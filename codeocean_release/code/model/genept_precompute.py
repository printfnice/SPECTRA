# genept_precompute.py
# 依赖: mygene, sentence_transformers, torch, anndata, liana
# 被依赖: ms_hgatlink_method.py, e2e_pipeline.py
# 职责: 生成类 GenePT 基因嵌入（NCBI summary + 句向量模型），离线一次性运行

"""
用 NCBI 基因摘要 + 本地句向量模型生成基因功能嵌入。
原始 GenePT 使用 OpenAI text-embedding-ada-002（1536维），
此处用开源 all-MiniLM-L6-v2（384维）替代，无需 API。

用法:
    export HF_ENDPOINT=https://hf-mirror.com  # 中国大陆镜像
    python genept_precompute.py                # 生成嵌入
    python genept_precompute.py --check        # 检查覆盖率
"""

import os
import json
import argparse
import time
from pathlib import Path

import numpy as np
import torch

_CODE_DIR = Path(__file__).parent
_DATA_DIR = _CODE_DIR.parent.parent / 'data'


def _collect_all_genes():
    """收集所有数据集 + LR resource 中的基因名（去重、大写）。"""
    import anndata
    import liana as li

    genes = set()

    # Consensus LR 基因
    res = li.resource.select_resource('consensus')
    genes.update(res['ligand'].tolist())
    genes.update(res['receptor'].tolist())
    print(f"  Consensus LR 基因: {len(genes)}")

    # 各数据集基因
    datasets = ['velmeshev', 'habermann', 'kuppe', 'carraro']
    for ds in datasets:
        for suffix in ['_processed.h5ad', '_filtered.h5ad']:
            p = _DATA_DIR / 'interim' / f'{ds}{suffix}'
            if p.exists():
                adata = anndata.read_h5ad(str(p))
                ds_genes = set(adata.var_names)
                genes.update(ds_genes)
                print(f"  {ds}: +{len(ds_genes)} 基因")
                break

    # 转为大写去重
    genes_upper = sorted({g.upper() for g in genes if g and len(g) > 1})
    print(f"  总计去重基因: {len(genes_upper)}")
    return genes_upper


def _query_ncbi_summaries(gene_list, batch_size=500):
    """批量查询 NCBI 基因摘要，返回 {gene_name: summary_text}。"""
    import mygene
    mg = mygene.MyGeneInfo()

    summaries = {}
    n_batches = (len(gene_list) + batch_size - 1) // batch_size

    for i in range(n_batches):
        batch = gene_list[i * batch_size:(i + 1) * batch_size]
        print(f"  查询 NCBI batch {i+1}/{n_batches} ({len(batch)} genes)...")
        results = mg.querymany(
            batch, scopes='symbol', species='human',
            fields='summary,name,type_of_gene', as_dataframe=False
        )
        for r in results:
            if isinstance(r, dict) and 'query' in r:
                gene = r['query'].upper()
                parts = []
                if 'name' in r:
                    parts.append(r['name'])
                if 'summary' in r:
                    parts.append(r['summary'])
                elif 'type_of_gene' in r:
                    parts.append(f"Type: {r['type_of_gene']}")
                if parts:
                    summaries[gene] = '. '.join(parts)
        time.sleep(0.5)  # 避免 API 限速

    print(f"  NCBI 查询完成: {len(summaries)}/{len(gene_list)} 基因有描述")
    return summaries


def _encode_summaries(summaries, model_name='all-MiniLM-L6-v2'):
    """用句向量模型编码基因描述，返回 {gene: numpy_array}。"""
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    genes = sorted(summaries.keys())
    texts = [summaries[g] for g in genes]

    print(f"  编码 {len(texts)} 个基因描述 (model={model_name})...")
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True)
    print(f"  嵌入维度: {embeddings.shape[1]}")

    return {gene: embeddings[i] for i, gene in enumerate(genes)}


def _save_as_pt(emb_dict, output_path):
    """保存为 PyTorch .pt 格式（与 ESM-2 嵌入格式一致）。"""
    pt_dict = {k: torch.from_numpy(v).float() for k, v in emb_dict.items()}
    torch.save(pt_dict, output_path)
    print(f"  已保存: {output_path} ({len(pt_dict)} genes)")


def _check_coverage(emb_path):
    """检查嵌入对各数据集的覆盖率。"""
    import anndata

    raw = torch.load(emb_path, map_location='cpu', weights_only=True)
    emb_genes = {k.upper() for k in raw}
    dim = next(iter(raw.values())).shape[0]
    print(f"\nGenePT-local 嵌入: {len(raw)} genes, dim={dim}")

    # ESM-2 对比
    esm2_path = _DATA_DIR / 'esm2_gene_embeddings.pt'
    if esm2_path.exists():
        esm2 = torch.load(esm2_path, map_location='cpu', weights_only=True)
        esm2_genes = {k.upper() for k in esm2}
        print(f"ESM-2 嵌入: {len(esm2)} genes, dim={next(iter(esm2.values())).shape[0]}")
    else:
        esm2_genes = set()

    datasets = ['velmeshev', 'habermann', 'kuppe', 'carraro']
    for ds in datasets:
        for suffix in ['_processed.h5ad', '_filtered.h5ad']:
            p = _DATA_DIR / 'interim' / f'{ds}{suffix}'
            if p.exists():
                adata = anndata.read_h5ad(str(p))
                ds_genes = {g.upper() for g in adata.var_names}
                gpt_cov = sum(1 for g in ds_genes if g in emb_genes)
                esm_cov = sum(1 for g in ds_genes if g in esm2_genes)
                print(f"  {ds:12s}: GenePT={gpt_cov}/{len(ds_genes)} "
                      f"({100*gpt_cov/len(ds_genes):.1f}%) | "
                      f"ESM-2={esm_cov}/{len(ds_genes)} "
                      f"({100*esm_cov/len(ds_genes):.1f}%)")
                break


def main():
    parser = argparse.ArgumentParser(description='生成类 GenePT 基因嵌入')
    parser.add_argument('--check', action='store_true', help='仅检查覆盖率')
    parser.add_argument('--model', default='all-MiniLM-L6-v2',
                        help='句向量模型名称')
    args = parser.parse_args()

    output_path = _DATA_DIR / 'genept_gene_embeddings.pt'

    if args.check:
        _check_coverage(output_path)
        return

    print("=== 生成类 GenePT 基因嵌入 ===")

    # 1. 收集基因
    gene_list = _collect_all_genes()

    # 2. 查�� NCBI 摘要
    cache_path = _DATA_DIR / 'ncbi_gene_summaries.json'
    if cache_path.exists():
        print(f"  加载缓存的 NCBI 摘要: {cache_path}")
        with open(cache_path) as f:
            summaries = json.load(f)
        # 查询缺失基因
        missing = [g for g in gene_list if g not in summaries]
        if missing:
            print(f"  补充查询 {len(missing)} 个缺失基因...")
            new_summaries = _query_ncbi_summaries(missing)
            summaries.update(new_summaries)
            with open(cache_path, 'w') as f:
                json.dump(summaries, f)
    else:
        summaries = _query_ncbi_summaries(gene_list)
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(summaries, f)
        print(f"  NCBI 摘要已缓存: {cache_path}")

    # 3. 编码
    emb_dict = _encode_summaries(summaries, args.model)

    # 4. 保存
    _save_as_pt(emb_dict, output_path)

    # 5. 覆盖率检查
    _check_coverage(output_path)


if __name__ == '__main__':
    main()
