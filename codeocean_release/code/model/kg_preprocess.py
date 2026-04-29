# kg_preprocess.py
# 依赖: torch, esm (fair-esm), requests, liana, pykeen (optional, R22)
# 被依赖: ms_hgatlink_method.py (via _load_esm2_features / _load_kge_features)
# 职责: 离线预计算 ESM-2 蛋白质嵌入和 OmniPath KGE 嵌入，保存到 data/ 目录
# 用法: python code/model/kg_preprocess.py [--step esm2|kge|all] [--model esm2_t6_8M_UR50D]

import argparse
import time
import os
import sys
from pathlib import Path

import torch
import numpy as np
import requests

# 路径设置（从任意目录运行均正确）
_HERE = Path(__file__).parent
_DATA_DIR = _HERE.parent.parent / 'data'


# ============================================================================
# UniProt 序列获取
# ============================================================================

def fetch_uniprot_batch(gene_names, organism_id=9606, batch_size=50):
    """批量从 UniProt REST API 获取蛋白质序列（限速重试）。

    Returns: dict {gene_name: aa_sequence}
    """
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    results = {}
    batches = [gene_names[i:i+batch_size] for i in range(0, len(gene_names), batch_size)]
    for batch_idx, batch in enumerate(batches):
        gene_query = " OR ".join([f'gene_exact:{g}' for g in batch])
        query = f'({gene_query}) AND organism_id:{organism_id} AND reviewed:true'
        params = {'query': query, 'fields': 'gene_names,sequence', 'format': 'tsv', 'size': 500}
        response = _fetch_with_retry(base_url, params)
        if response:
            _parse_uniprot_tsv(response.text, results)
        if (batch_idx + 1) % 5 == 0:
            print(f"  UniProt: {batch_idx+1}/{len(batches)} 批次完成, {len(results)} 条序列")
        time.sleep(0.3)
    return results


def _fetch_with_retry(url, params, max_retries=3):
    """带重试的 HTTP GET"""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                wait = 2 ** attempt
                print(f"  限速，等待 {wait}s...")
                time.sleep(wait)
        except requests.RequestException as e:
            print(f"  请求失败 ({attempt+1}/{max_retries}): {e}")
            time.sleep(1)
    return None


def _parse_uniprot_tsv(text, results):
    """解析 UniProt TSV 响应，提取基因名→序列映射"""
    lines = text.strip().split('\n')
    if len(lines) < 2:
        return
    for line in lines[1:]:
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        gene_field, seq = parts[0], parts[-1]
        if not seq or len(seq) < 10:
            continue
        for gene in gene_field.split():
            results[gene.upper()] = seq


# ============================================================================
# ESM-2 嵌入计算
# ============================================================================

def load_esm2_model(model_name='esm2_t6_8M_UR50D', device='cpu'):
    """加载 ESM-2 预训练模型（首次运行自动下载）"""
    import esm
    print(f"  加载 ESM-2 模型: {model_name} ...")
    loader = getattr(esm.pretrained, model_name)
    model, alphabet = loader()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


def compute_esm2_embeddings(seq_dict, model_name='esm2_t6_8M_UR50D', device='cpu',
                             batch_size=16, max_seq_len=1022):
    """对蛋白质序列字典计算 ESM-2 mean-pool 嵌入。

    Returns: dict {gene_name: numpy_array(esm2_dim)}
    """
    model, alphabet, batch_converter = load_esm2_model(model_name, device)
    genes = list(seq_dict.keys())
    sequences = [seq_dict[g][:max_seq_len] for g in genes]
    embeddings = {}
    batches = [list(range(i, min(i+batch_size, len(genes)))) for i in range(0, len(genes), batch_size)]
    for batch_idx, idxs in enumerate(batches):
        batch_genes = [genes[i] for i in idxs]
        batch_seqs = [sequences[i] for i in idxs]
        data = [(g, s) for g, s in zip(batch_genes, batch_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[model.num_layers])
        reps = out['representations'][model.num_layers]
        for j, gene in enumerate(batch_genes):
            seq_len = len(sequences[idxs[j]])
            emb = reps[j, 1:seq_len+1].mean(0).cpu().numpy()
            embeddings[gene] = emb
        if (batch_idx + 1) % 10 == 0:
            print(f"  ESM-2 推断: {(batch_idx+1)*batch_size}/{len(genes)}")
    return embeddings


def run_esm2_pipeline(output_path, model_name='esm2_t6_8M_UR50D', device=None):
    """完整 ESM-2 流水线：提取基因 -> 查序列 -> 计算嵌入 -> 保存"""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gene_names = _extract_lr_genes()
    print(f"LR 基因总数: {len(gene_names)}")
    print("从 UniProt 获取蛋白质序列 ...")
    seq_dict = fetch_uniprot_batch(gene_names)
    print(f"获取到序列: {len(seq_dict)}/{len(gene_names)}")
    print(f"计算 ESM-2 嵌入 (model={model_name}, device={device}) ...")
    embeddings = compute_esm2_embeddings(seq_dict, model_name, device)
    # 转为 tensor dict 保存
    tensor_dict = {g: torch.tensor(v, dtype=torch.float32) for g, v in embeddings.items()}
    os.makedirs(output_path.parent, exist_ok=True)
    torch.save(tensor_dict, output_path)
    emb_dim = next(iter(tensor_dict.values())).shape[0]
    print(f"保存完成: {output_path} ({len(tensor_dict)} 基因, dim={emb_dim})")
    return tensor_dict


def _extract_lr_genes():
    """从 LIANA consensus 资源提取全部 LR 基因名（大写）"""
    try:
        import liana as li
        resource = li.resource.select_resource('consensus')
        genes = set(resource['ligand'].str.upper()) | set(resource['receptor'].str.upper())
        return sorted(genes)
    except ImportError:
        print("警告: LIANA 未安装，使用空基因列表")
        return []


# ============================================================================
# OmniPath KGE 训练（R22）
# ============================================================================

def _build_lr_triples():
    """从 LIANA consensus 资源构建 LR 关系三元组。

    Returns: list of (head, relation, tail)，如 ('TGFB1', 'ligates', 'TGFBR1')
    """
    import liana as li
    resource = li.resource.select_resource('consensus')
    triples = [(row.ligand.upper(), 'ligates', row.receptor.upper())
               for _, row in resource.iterrows()]
    # 添加反向关系（受体被配体激活）
    triples += [(row.receptor.upper(), 'activated_by', row.ligand.upper())
                for _, row in resource.iterrows()]
    return triples


def _train_rotate(triples, emb_dim=128, n_epochs=100, device=None):
    """用 PyKEEN RotatE 训练知识图谱嵌入，返回实体嵌入字典。

    Args:
        triples: list of (head, relation, tail)
        emb_dim: 嵌入维度（实体嵌入）
        n_epochs: 训练轮数
        device: cuda 或 cpu
    Returns: dict {entity_name: tensor(emb_dim)}
    """
    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    triples_arr = np.array(triples)
    factory = TriplesFactory.from_labeled_triples(triples_arr)
    # 按 8:1:1 拆分 training/validation/testing
    train, valid, test = factory.split([0.8, 0.1, 0.1], random_state=42)
    result = pipeline(
        training=train,
        validation=valid,
        testing=test,
        model='RotatE',
        model_kwargs={'embedding_dim': emb_dim},
        training_kwargs={'num_epochs': n_epochs, 'batch_size': 512},
        optimizer='Adam',
        optimizer_kwargs={'lr': 0.001},
        device=device,
        random_seed=42,
        use_tqdm=True,
    )
    all_indices = torch.arange(factory.num_entities, device=result.model.device)
    entity_emb = result.model.entity_representations[0](indices=all_indices).detach().cpu()
    entity_to_id = factory.entity_to_id
    return {ent: entity_emb[idx] for ent, idx in entity_to_id.items()}


def run_kge_pipeline(output_path, emb_dim=128, n_epochs=100):
    """LR 知识图谱嵌入训练流水线（RotatE on consensus LR pairs）"""
    try:
        from pykeen.triples import TriplesFactory  # noqa: F401
    except ImportError:
        print("pykeen 未安装，跳过 KGE 训练。运行: pip install pykeen")
        return None
    print("构建 LR 三元组 ...")
    triples = _build_lr_triples()
    print(f"  三元组数: {len(triples)}, 关系数: 2 (ligates/activated_by)")
    unique_ents = len(set(t[0] for t in triples) | set(t[2] for t in triples))
    print(f"  实体数: {unique_ents}")
    print(f"训练 RotatE (dim={emb_dim}, epochs={n_epochs}) ...")
    emb_dict = _train_rotate(triples, emb_dim, n_epochs)
    tensor_dict = {g: v.float() for g, v in emb_dict.items()}
    os.makedirs(output_path.parent, exist_ok=True)
    torch.save(tensor_dict, output_path)
    print(f"保存完成: {output_path} ({len(tensor_dict)} 实体, dim={emb_dim})")
    return tensor_dict


# ============================================================================
# 主函数
# ============================================================================

def _parse_args():
    parser = argparse.ArgumentParser(description='预计算知识图谱嵌入')
    parser.add_argument('--step', choices=['esm2', 'kge', 'all'], default='esm2')
    parser.add_argument('--model', default='esm2_t6_8M_UR50D',
                        choices=['esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D', 'esm2_t30_150M_UR50D'],
                        help='ESM-2 模型名称')
    parser.add_argument('--device', default=None, help='cuda 或 cpu（默认自动检测）')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if args.step in ('esm2', 'all'):
        esm2_path = _DATA_DIR / 'esm2_gene_embeddings.pt'
        run_esm2_pipeline(esm2_path, args.model, device)
    if args.step in ('kge', 'all'):
        kge_path = _DATA_DIR / 'omnipath_kge.pt'
        run_kge_pipeline(kge_path)
