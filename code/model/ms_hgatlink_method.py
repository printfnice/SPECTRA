# ms_hgatlink_method.py
# 依赖: ms_hgatlink_encoders.py (MSHGATLink), liana, numpy, pandas, scipy
# 被依赖: simplified_pipeline_optimized.py, run_all_ms_hgatlink.py
# 职责: MS-HGATLink 推断接口（预训练 + 推断 + LIANA注册）

import warnings
import gc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import issparse

from model.ms_hgatlink_encoders import MSHGATLink

_MODEL_DIR = Path(__file__).parent
_DATA_DIR = _MODEL_DIR.parent.parent / 'data'

warnings.filterwarnings('ignore')

# ============================================================================
# 数据增强配置
# ============================================================================

_AUG_CONFIGS = {
    'velmeshev': {'noise_std': 0.1,  'dropout_rate': 0.05, 'n_augmentations': 8},
    'carraro':   {'noise_std': 0.08, 'dropout_rate': 0.05, 'n_augmentations': 8},
    'habermann': {'noise_std': 0.08, 'dropout_rate': 0.05, 'n_augmentations': 8},
    'default':   {'noise_std': 0.05, 'dropout_rate': 0.03, 'n_augmentations': 5},
}

# 数据集特异性模型超参数
# esm2_alpha: ESM-2 余弦相似度富集强度（0=禁用），需先运行 kg_preprocess.py --step esm2
# kge_alpha:  KGE 内积富集强度（0=禁用），需先运行 kg_preprocess.py --step kge
_DATASET_CONFIGS = {
    'habermann': {'gene_emb_dim': 32,  'dropout': 0.40, 'pretrain_epochs': 30, 'n_seeds': 1, 'specificity_alpha': 0.0, 't_alpha': 0.3, 'lfc_alpha': 0.0, 'disease_alpha': 0.0, 'use_esm2': True, 'use_kge': True, 'use_genept': False, 'esm2_alpha': 0.0, 'kge_alpha': 0.0, 'esm2_selection': False, 'enable_pretraining': False, 'condition_key': 'Status'},
    'velmeshev': {'gene_emb_dim': 96,  'dropout': 0.25, 'pretrain_epochs': 30, 'n_seeds': 1, 'specificity_alpha': 0.0, 't_alpha': 0.0, 'lfc_alpha': 0.0, 'disease_alpha': 0.0, 'quality_lr_cutoff': 150, 'score_formula': 'mean', 'use_esm2': True, 'use_kge': True, 'use_genept': False, 'esm2_alpha': 0.0, 'kge_alpha': 0.0, 'esm2_selection': False, 'enable_pretraining': False, 'condition_key': None},
    'kuppe':     {'gene_emb_dim': 128, 'dropout': 0.20, 'pretrain_epochs': 30, 'n_seeds': 1, 'specificity_alpha': 0.0, 't_alpha': 0.0, 'lfc_alpha': 0.0, 'use_esm2': True, 'use_kge': True, 'use_genept': False, 'esm2_alpha': 0.0, 'kge_alpha': 0.0, 'esm2_selection': False, 'enable_pretraining': False, 'condition_key': None},
    'carraro':   {'gene_emb_dim': 48,  'dropout': 0.40, 'pretrain_epochs': 30, 'n_seeds': 1, 'specificity_alpha': 0.0, 't_alpha': 0.0, 'lfc_alpha': 0.3, 'use_esm2': True, 'use_kge': True, 'use_genept': False, 'esm2_alpha': 0.0, 'kge_alpha': 0.0, 'esm2_selection': False, 'enable_pretraining': False, 'condition_key': None},
    'reichart':  {'gene_emb_dim': 64,  'dropout': 0.30, 'pretrain_epochs': 30, 'n_seeds': 1, 'specificity_alpha': 0.0, 't_alpha': 0.0, 'lfc_alpha': 0.0, 'use_esm2': True, 'use_kge': True, 'use_genept': False, 'esm2_alpha': 0.0, 'kge_alpha': 0.0, 'esm2_selection': False, 'enable_pretraining': False, 'condition_key': None},
    'default':   {'gene_emb_dim': 64,  'dropout': 0.30, 'pretrain_epochs': 30, 'n_seeds': 1, 'specificity_alpha': 0.0, 't_alpha': 0.0, 'lfc_alpha': 0.0, 'use_esm2': True, 'use_kge': True, 'use_genept': False, 'esm2_alpha': 0.0, 'kge_alpha': 0.0, 'esm2_selection': False, 'enable_pretraining': False, 'condition_key': None},
}

# 模块级缓存：避免每个样本重复加载大型嵌入文件
_ESM2_DICT_CACHE = {}
_KGE_DICT_CACHE = {}

# 多种子候选列表（用于平均推断，消除初始化随机性）
_SEED_LIST = [42, 123, 456]


# ============================================================================
# 辅助函数：配置获取
# ============================================================================

def _get_esm2_dict(emb_path=None):
    """加载 ESM-2 嵌入为 numpy dict（模块级缓存，避免重复 IO）。"""
    key = str(emb_path or _DATA_DIR / 'esm2_gene_embeddings.pt')
    if key not in _ESM2_DICT_CACHE:
        p = Path(key)
        if not p.exists():
            _ESM2_DICT_CACHE[key] = {}
        else:
            raw = torch.load(p, map_location='cpu', weights_only=True)
            _ESM2_DICT_CACHE[key] = {k.upper(): v.numpy() for k, v in raw.items()}
            print(f"  ESM-2 dict 已缓存: {len(_ESM2_DICT_CACHE[key])} 个蛋白质")
    return _ESM2_DICT_CACHE[key]


def _get_kge_dict(emb_path=None):
    """加载 KGE 嵌入为 numpy dict（模块级缓存）。"""
    key = str(emb_path or _DATA_DIR / 'omnipath_kge.pt')
    if key not in _KGE_DICT_CACHE:
        p = Path(key)
        if not p.exists():
            _KGE_DICT_CACHE[key] = {}
        else:
            raw = torch.load(p, map_location='cpu', weights_only=True)
            _KGE_DICT_CACHE[key] = {k.upper(): v.numpy() for k, v in raw.items()}
            print(f"  KGE dict 已缓存: {len(_KGE_DICT_CACHE[key])} 个实体")
    return _KGE_DICT_CACHE[key]


def _compute_esm2_enrichment(batch_data, esm2_dict):
    """ESM-2 蛋白质功能亲和先验：配体-受体 ESM-2 嵌入的余弦相似度。

    同一 LR 对的所有 CT pair 得到相同的 enrich 值（LR 级别先验，非 CT 特异）。
    返回 [n_batch]，范围 [-1, 1]，缺失基因对返回 0（中性）。
    """
    ligs = [d['ligand'].upper() for d in batch_data]
    recs = [d['receptor'].upper() for d in batch_data]
    zero = np.zeros(320 if esm2_dict else 1)
    lig_emb = np.stack([esm2_dict.get(g, zero) for g in ligs])
    rec_emb = np.stack([esm2_dict.get(g, zero) for g in recs])
    lig_n = lig_emb / (np.linalg.norm(lig_emb, axis=1, keepdims=True) + 1e-8)
    rec_n = rec_emb / (np.linalg.norm(rec_emb, axis=1, keepdims=True) + 1e-8)
    return (lig_n * rec_n).sum(axis=1)


def _compute_kge_enrichment(batch_data, kge_dict):
    """KGE 图结构先验：配体-受体 KGE 嵌入的余弦相似度。

    KGE 在 consensus LR 三元组上训练，余弦相似度反映 LR 对在 LR 互作图中的结构强度。
    返回 [n_batch]，范围 [-1, 1]，缺失实体返回 0。
    """
    ligs = [d['ligand'].upper() for d in batch_data]
    recs = [d['receptor'].upper() for d in batch_data]
    zero = np.zeros(128 if kge_dict else 1)
    lig_emb = np.stack([kge_dict.get(g, zero) for g in ligs])
    rec_emb = np.stack([kge_dict.get(g, zero) for g in recs])
    lig_n = lig_emb / (np.linalg.norm(lig_emb, axis=1, keepdims=True) + 1e-8)
    rec_n = rec_emb / (np.linalg.norm(rec_emb, axis=1, keepdims=True) + 1e-8)
    return (lig_n * rec_n).sum(axis=1)


def _load_kge_features(gene_to_idx, emb_path=None):
    """加载 OmniPath KGE 嵌入并组装为 [n_genes, kge_dim] 张量（缺失基因置零）。

    Args:
        gene_to_idx: dict {gene_name: idx}
        emb_path: .pt 文件路径，默认 data/omnipath_kge.pt
    Returns: tensor [n_genes, kge_dim] 或 None
    """
    if emb_path is None:
        emb_path = _DATA_DIR / 'omnipath_kge.pt'
    if not Path(emb_path).exists():
        return None
    raw = torch.load(emb_path, map_location='cpu', weights_only=True)
    n_genes = len(gene_to_idx)
    emb_dim = next(iter(raw.values())).shape[0]
    table = torch.zeros(n_genes, emb_dim)
    n_found = 0
    for gene, idx in gene_to_idx.items():
        key = gene.upper() if gene.upper() in raw else gene
        if key in raw:
            table[idx] = raw[key]
            n_found += 1
    print(f"  KGE 先验: {n_found}/{n_genes} 基因有嵌入 (dim={emb_dim})")
    return table


def _load_esm2_features(gene_to_idx, emb_path=None):
    """加载 ESM-2 嵌入并组装为 [n_genes, esm2_dim] 张量（缺失基因置零）。

    Args:
        gene_to_idx: dict {gene_name: idx}，来自 _build_gene_ct_index
        emb_path: .pt 文件路径，默认 data/esm2_gene_embeddings.pt
    Returns: tensor [n_genes, esm2_dim] 或 None（文件不存在时）
    """
    if emb_path is None:
        emb_path = _DATA_DIR / 'esm2_gene_embeddings.pt'
    if not Path(emb_path).exists():
        return None
    raw = torch.load(emb_path, map_location='cpu', weights_only=True)
    n_genes = len(gene_to_idx)
    emb_dim = next(iter(raw.values())).shape[0]
    table = torch.zeros(n_genes, emb_dim)
    n_found = sum(1 for g in gene_to_idx if g.upper() in raw or g in raw)
    for gene, idx in gene_to_idx.items():
        key = gene.upper() if gene.upper() in raw else gene
        if key in raw:
            table[idx] = raw[key]
    print(f"  ESM-2 先验: {n_found}/{n_genes} 基因有嵌入 (dim={emb_dim})")
    return table


def _load_genept_features(gene_to_idx, emb_path=None):
    """加载 GenePT 嵌入并组装为 [n_genes, genept_dim] 张量（缺失基因置零）。

    Args:
        gene_to_idx: dict {gene_name: idx}
        emb_path: .pt 文件路径，默认 data/genept_gene_embeddings.pt
    Returns: tensor [n_genes, genept_dim] 或 None
    """
    if emb_path is None:
        emb_path = _DATA_DIR / 'genept_gene_embeddings.pt'
    if not Path(emb_path).exists():
        return None
    raw = torch.load(emb_path, map_location='cpu', weights_only=True)
    n_genes = len(gene_to_idx)
    emb_dim = next(iter(raw.values())).shape[0]
    table = torch.zeros(n_genes, emb_dim)
    n_found = 0
    for gene, idx in gene_to_idx.items():
        key = gene.upper() if gene.upper() in raw else gene
        if key in raw:
            table[idx] = raw[key]
            n_found += 1
    print(f"  GenePT 先验: {n_found}/{n_genes} 基因有嵌入 (dim={emb_dim})")
    return table


def _get_aug_config(dataset_name, enable, n_aug_override):
    """获取数据增强配置"""
    if not enable:
        return {'noise_std': 0.0, 'dropout_rate': 0.0, 'n_augmentations': 1}
    cfg = _AUG_CONFIGS.get(dataset_name, _AUG_CONFIGS['default']).copy()
    if n_aug_override != 5:  # 非默认时覆盖
        cfg['n_augmentations'] = n_aug_override
    return cfg


def _get_model_config(dataset_name, pretrain_epochs_override):
    """获取模型超参数配置"""
    cfg = _DATASET_CONFIGS.get(dataset_name, _DATASET_CONFIGS['default']).copy()
    if pretrain_epochs_override != 50:
        cfg['pretrain_epochs'] = pretrain_epochs_override
    return cfg


# ============================================================================
# 辅助函数：数据准备
# ============================================================================

def _build_gene_ct_index(adata, groupby):
    """构建基因和细胞类型的索引映射"""
    all_genes = list(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    all_cts = list(adata.obs[groupby].unique())
    ct_to_idx = {ct: i for i, ct in enumerate(all_cts)}
    return gene_to_idx, ct_to_idx, len(all_genes), len(all_cts)


def _init_ms_model(n_genes, n_cell_types, cfg, device, dataset_name, enable_pretraining, verbose,
                   seed=42, gene_to_idx=None):
    """初始化 MS-HGATLink 模型（固定随机种子保证方向一致性）。

    当 cfg['use_esm2']=True 且 gene_to_idx 不为 None 时，自动加载 ESM-2 先验嵌入。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dim = cfg['gene_emb_dim']
    n_heads = 4 if dim >= 64 else 2
    esm2_feat = None
    if cfg.get('use_esm2', False) and gene_to_idx is not None:
        esm2_feat = _load_esm2_features(gene_to_idx)
        if esm2_feat is not None:
            esm2_feat = esm2_feat.to(device)
    kge_feat = None
    if cfg.get('use_kge', False) and gene_to_idx is not None:
        kge_feat = _load_kge_features(gene_to_idx)
        if kge_feat is not None:
            kge_feat = kge_feat.to(device)
    genept_feat = None
    if cfg.get('use_genept', False) and gene_to_idx is not None:
        genept_feat = _load_genept_features(gene_to_idx)
        if genept_feat is not None:
            genept_feat = genept_feat.to(device)
    model = MSHGATLink(
        n_genes=n_genes, n_cell_types=n_cell_types,
        gene_emb_dim=dim, pathway_dim=dim, ct_dim=dim,
        fusion_dim=dim * 2, n_heads=n_heads, dropout=cfg['dropout'],
        esm2_features=esm2_feat, kge_features=kge_feat,
        genept_features=genept_feat
    ).to(device)
    model.eval()
    if verbose:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        tags = []
        if esm2_feat is not None:
            tags.append(f'ESM-2 dim={esm2_feat.shape[1]}')
        if kge_feat is not None:
            tags.append(f'KGE dim={kge_feat.shape[1]}')
        if genept_feat is not None:
            tags.append(f'GenePT dim={genept_feat.shape[1]}')
        tag = (', ' + ', '.join(tags)) if tags else ''
        print(f"  模型参数量: {n_params:.2f}M{tag}")
    return model, enable_pretraining


def _get_sample_data(sample_adata, groupby, min_cells, use_raw):
    """提取细胞类型平均/标准差表达量及全样本均值。
    返回 (ct_expr_dict, ct_std_dict, ct_count_dict, valid_cts, gene_names, global_mean) 或 6×None"""
    expr = sample_adata.raw.X if (use_raw and sample_adata.raw) else sample_adata.X
    gene_names = list(sample_adata.raw.var_names if (use_raw and sample_adata.raw)
                      else sample_adata.var_names)
    if issparse(expr):
        expr = expr.toarray()
    counts = sample_adata.obs[groupby].value_counts()
    valid_cts = counts[counts >= min_cells].index.tolist()
    if len(valid_cts) < 2:
        return None, None, None, None, None, None
    ct_expr_dict, ct_std_dict, ct_count_dict = {}, {}, {}
    for ct in valid_cts:
        mask = sample_adata.obs[groupby] == ct
        ct_data = expr[mask, :]
        ct_expr_dict[ct]  = ct_data.mean(axis=0).ravel()
        ct_std_dict[ct]   = ct_data.std(axis=0).ravel()
        ct_count_dict[ct] = int(mask.sum())
    global_mean = expr.mean(axis=0).ravel()
    return ct_expr_dict, ct_std_dict, ct_count_dict, valid_cts, gene_names, global_mean


def _compute_lr_quality_scores(valid_lr):
    """计算 LR 对综合先验质量分（ESM-2 余弦相似度 + KGE 余弦相似度）。

    质量分越高 → 蛋白质功能先验和图结构先验越强。
    用于在 max_lr_pairs 截断时优先保留高质量对，降低噪声。
    返回 numpy array [n_valid_lr]，未覆盖基因得分为 0。
    """
    esm2_dict = _get_esm2_dict()
    kge_dict = _get_kge_dict()
    ligs = valid_lr['ligand'].str.upper().values
    recs = valid_lr['receptor'].str.upper().values
    quality = np.zeros(len(valid_lr))
    if esm2_dict:
        zero_e = np.zeros(next(iter(esm2_dict.values())).shape[0])
        lig_e = np.stack([esm2_dict.get(g, zero_e) for g in ligs])
        rec_e = np.stack([esm2_dict.get(g, zero_e) for g in recs])
        lig_n = lig_e / (np.linalg.norm(lig_e, axis=1, keepdims=True) + 1e-8)
        rec_n = rec_e / (np.linalg.norm(rec_e, axis=1, keepdims=True) + 1e-8)
        quality += (lig_n * rec_n).sum(axis=1)
    if kge_dict:
        zero_k = np.zeros(next(iter(kge_dict.values())).shape[0])
        lig_k = np.stack([kge_dict.get(g, zero_k) for g in ligs])
        rec_k = np.stack([kge_dict.get(g, zero_k) for g in recs])
        lig_n = lig_k / (np.linalg.norm(lig_k, axis=1, keepdims=True) + 1e-8)
        rec_n = rec_k / (np.linalg.norm(rec_k, axis=1, keepdims=True) + 1e-8)
        quality += (lig_n * rec_n).sum(axis=1)
    return quality


def _filter_valid_lr(resource, gene_set, max_lr_pairs, esm2_selection=False):
    """过滤有效配体-受体对并限制数量。

    esm2_selection=True 时，按 ESM-2+KGE 综合质量分降序选取 top-max_lr_pairs，
    替代默认的数据库顺序截断，降低噪声 LR 对进入特征矩阵的概率。
    """
    valid = resource[resource['ligand'].isin(gene_set) &
                     resource['receptor'].isin(gene_set)].copy()
    if not (max_lr_pairs and len(valid) > max_lr_pairs):
        return valid
    if esm2_selection:
        quality = _compute_lr_quality_scores(valid)
        top_idx = np.argsort(-quality)[:max_lr_pairs]
        return valid.iloc[top_idx].reset_index(drop=True)
    return valid.head(max_lr_pairs)


def _build_batch_entries(valid_lr_pairs, gene_names, ct_expr_dict,
                         gene_to_idx, ct_to_idx, valid_cts, lr_pathway_map=None):
    """构建批量推断数据"""
    batch_data = []
    for _, row in valid_lr_pairs.iterrows():
        lig, rec = row['ligand'], row['receptor']
        try:
            lig_local = gene_names.index(lig)
            rec_local = gene_names.index(rec)
        except ValueError:
            continue
        pw_idx = lr_pathway_map.get((lig, rec), 0) if lr_pathway_map else 0
        for src in valid_cts:
            for tgt in valid_cts:
                batch_data.append({
                    'ligand': lig, 'receptor': rec,
                    'source_ct': src, 'target_ct': tgt,
                    'lig_idx': gene_to_idx[lig], 'rec_idx': gene_to_idx[rec],
                    'lig_local': lig_local, 'rec_local': rec_local,
                    'lig_expr': float(ct_expr_dict[src][lig_local]),
                    'rec_expr': float(ct_expr_dict[tgt][rec_local]),
                    'source_ct_idx': ct_to_idx[src],
                    'target_ct_idx': ct_to_idx[tgt],
                    'pathway_idx': pw_idx,
                })
    return batch_data


def _normalize_expr(batch_data):
    """对批量数据的表达值做 Z-score 标准化"""
    lig_vals = [d['lig_expr'] for d in batch_data]
    rec_vals = [d['rec_expr'] for d in batch_data]
    lm, ls = np.mean(lig_vals), np.std(lig_vals) + 1e-8
    rm, rs = np.mean(rec_vals), np.std(rec_vals) + 1e-8
    lig_norm = [(x - lm) / ls for x in lig_vals]
    rec_norm = [(x - rm) / rs for x in rec_vals]
    return lig_norm, rec_norm


def _augment_single(lig_norm, rec_norm, noise_std, dropout_rate):
    """对归一化表达值添加噪声+dropout增强"""
    lig_aug, rec_aug = [], []
    for l, r in zip(lig_norm, rec_norm):
        l_new = 0.0 if np.random.random() < dropout_rate else l + np.random.normal(0, noise_std)
        r_new = 0.0 if np.random.random() < dropout_rate else r + np.random.normal(0, noise_std)
        lig_aug.append(l_new)
        rec_aug.append(r_new)
    return lig_aug, rec_aug


def _forward_pass(model, batch_data, lig_aug, rec_aug, device, score_formula='product'):
    """单次前向传播，返回 scores 和 scale_weights（numpy）。
    score_formula: 'product'（乘积，默认）| 'mean'（算术平均，类 CellPhoneDB，适合弱信号数据集）"""
    lig_idx = torch.tensor([d['lig_idx'] for d in batch_data], dtype=torch.long, device=device)
    rec_idx = torch.tensor([d['rec_idx'] for d in batch_data], dtype=torch.long, device=device)
    lig_expr = torch.tensor([[x] for x in lig_aug], dtype=torch.float32, device=device)
    rec_expr = torch.tensor([[x] for x in rec_aug], dtype=torch.float32, device=device)
    src_ct = torch.tensor([d['source_ct_idx'] for d in batch_data], dtype=torch.long, device=device)
    tgt_ct = torch.tensor([d['target_ct_idx'] for d in batch_data], dtype=torch.long, device=device)
    if score_formula == 'mean':
        base_vals = [[(d['lig_expr'] + d['rec_expr']) / 2] for d in batch_data]
    else:
        base_vals = [[d['lig_expr'] * d['rec_expr']] for d in batch_data]
    expr_prod = torch.tensor(base_vals, dtype=torch.float32, device=device)
    scores, weights = model(lig_idx, rec_idx, lig_expr, rec_expr, src_ct, tgt_ct,
                            expr_product=expr_prod)
    return scores.squeeze(-1).cpu().numpy(), weights.cpu().numpy()


def _run_augmented_inference(model, batch_data, aug_cfg, device, score_formula='product'):
    """带数据增强的推断，返回平均 scores 和 scale_weights"""
    lig_norm, rec_norm = _normalize_expr(batch_data)
    all_scores, all_weights = [], []
    with torch.no_grad():
        for i in range(aug_cfg['n_augmentations']):
            if i == 0:
                la, ra = lig_norm, rec_norm
            else:
                la, ra = _augment_single(lig_norm, rec_norm,
                                         aug_cfg['noise_std'], aug_cfg['dropout_rate'])
            s, w = _forward_pass(model, batch_data, la, ra, device, score_formula)
            all_scores.append(s)
            all_weights.append(w)
    return np.mean(all_scores, axis=0), np.mean(all_weights, axis=0)


def _collect_results(batch_data, scores, weights, sample, sample_key):
    """将推断结果整理为 list of dict"""
    return [{
        sample_key: sample,
        'ligand_complex': d['ligand'],
        'receptor_complex': d['receptor'],
        'source': d['source_ct'],
        'target': d['target_ct'],
        'ms_hgatlink_score': float(scores[i]),
        'gene_weight': float(weights[i, 0]),
        'pathway_weight': float(weights[i, 1]),
        'ct_weight': float(weights[i, 2]),
    } for i, d in enumerate(batch_data)]



def _apply_specificity_modulation(batch_data, scores, alpha):
    """对每个 LR 对内的 CT pair 应用 within-LR Z-score 特异性调制。
    高 Z-score = 该 CT pair 在该 LR 对上相对富集 -> boost；低 Z-score -> suppress。
    等价于 CellPhoneDB 置换检验捕获的特异性信号。"""
    if alpha <= 0:
        return scores
    from collections import defaultdict
    lr_groups = defaultdict(list)
    for i, d in enumerate(batch_data):
        lr_groups[(d['ligand'], d['receptor'])].append(i)
    modulated = scores.copy()
    for indices in lr_groups.values():
        if len(indices) < 2:
            continue
        group = scores[indices]
        std = group.std()
        if std < 1e-8:
            continue
        z = (group - group.mean()) / (std + 1e-8)
        modulated[indices] = group * (1 + alpha * np.tanh(z))
    return modulated


def _apply_disease_weights(batch_data, scores, alpha, disease_weights):
    """用疾病相关性权重调制 LR 评分（与 t_alpha/lfc_alpha 同构）。
    disease_weights: {(lig, rec): float ∈ [0, 1]}，0=无关，1=高度相关。
    调制公式: score *= (1 + alpha * tanh(weight - 0.5))，将 [0,1] 中心化到 [-0.5, 0.5]。"""
    if not disease_weights:
        return scores
    w = np.array([disease_weights.get((d['ligand'], d['receptor']), 0.0)
                  for d in batch_data])
    w_centered = (w - 0.3) * 2.0  # 中心化：0→-0.6, 0.5→0.4, 1.0→1.4
    return scores * (1 + alpha * np.tanh(w_centered))


def _apply_quality_cutoff(batch_data, valid_lr, scores, cutoff):
    """只保留前 cutoff 个 LR 对的真实分数，其余清零以维持 zero_frac > 0.6 触发 Z-score 归一化。
    等效于 '只用前 cutoff 对 + 强制 Z-score' 而不改变 step3/4 的归一化逻辑。"""
    top_pairs = set(zip(
        valid_lr['ligand'].head(cutoff).tolist(),
        valid_lr['receptor'].head(cutoff).tolist()
    ))
    mask = np.array([(d['ligand'], d['receptor']) in top_pairs for d in batch_data])
    result = scores.copy()
    result[~mask] = 0.0
    return result


def _compute_t_enrichment(batch_data, ct_std_dict, ct_count_dict, global_mean):
    """近似 CellPhoneDB 置换检验：对每个 (LR, CT pair) 计算 t-score 富集。
    t_lig = (mean_lig_src - sample_mean_lig) / (std_lig_src / sqrt(N_src))
    返回 t_lig * t_rec（均高于背景时为正，即细胞类型特异性富集）。"""
    lig_exprs = np.array([d['lig_expr'] for d in batch_data])
    rec_exprs = np.array([d['rec_expr'] for d in batch_data])
    lig_local = np.array([d['lig_local'] for d in batch_data])
    rec_local = np.array([d['rec_local'] for d in batch_data])
    gm_lig = global_mean[lig_local]
    gm_rec = global_mean[rec_local]
    src_std = np.array([ct_std_dict[d['source_ct']][d['lig_local']] for d in batch_data])
    tgt_std = np.array([ct_std_dict[d['target_ct']][d['rec_local']] for d in batch_data])
    n_src = np.array([ct_count_dict[d['source_ct']] for d in batch_data], dtype=float)
    n_tgt = np.array([ct_count_dict[d['target_ct']] for d in batch_data], dtype=float)
    t_lig = (lig_exprs - gm_lig) / (src_std / np.sqrt(n_src) + 1e-8)
    t_rec = (rec_exprs - gm_rec) / (tgt_std / np.sqrt(n_tgt) + 1e-8)
    return np.clip(t_lig, -10, 10) * np.clip(t_rec, -10, 10)


def _compute_lfc_enrichment(batch_data, global_mean, positive_only=False):
    """LogFC 式富集：在 log1p 空间做差，等价于对数倍变化，不依赖 std/sqrt(N)。
    适用于细胞数少或表达量低的 CT（如脑细胞）。
    positive_only=True: 只在配体/受体均高于全局均值时增强，否则中性（clip 下界为0）。
    positive_only=False（默认）: 双向调制（含负值乘积 = 抑制）。"""
    lig_exprs = np.array([d['lig_expr'] for d in batch_data])
    rec_exprs = np.array([d['rec_expr'] for d in batch_data])
    lig_local = np.array([d['lig_local'] for d in batch_data])
    rec_local = np.array([d['rec_local'] for d in batch_data])
    gm_lig = global_mean[lig_local]
    gm_rec = global_mean[rec_local]
    lfc_lig = lig_exprs - gm_lig
    lfc_rec = rec_exprs - gm_rec
    lo = 0.0 if positive_only else -5.0
    return np.clip(lfc_lig, lo, 5) * np.clip(lfc_rec, lo, 5)


def _process_sample_and_collect(sample, adata, model, resource, gene_to_idx, ct_to_idx,
                                 sample_key, groupby, aug_cfg, model_cfg, enable_pretraining,
                                 condition_key, max_lr_pairs, min_cells, use_raw, device, verbose):
    """处理单个样本并收集结果"""
    sample_adata = adata[adata.obs[sample_key] == sample].copy()
    ct_expr_dict, ct_std_dict, ct_count_dict, valid_cts, gene_names, global_mean = _get_sample_data(
        sample_adata, groupby, min_cells, use_raw)
    if ct_expr_dict is None:
        if verbose:
            print(f"  跳过 {sample}：有效细胞类型不足")
        return []
    valid_lr = _filter_valid_lr(resource, set(gene_names), max_lr_pairs,
                                esm2_selection=model_cfg.get('esm2_selection', False))
    if len(valid_lr) == 0:
        return []
    batch_data = _build_batch_entries(
        valid_lr, gene_names, ct_expr_dict, gene_to_idx, ct_to_idx, valid_cts)
    if not batch_data:
        return []
    if enable_pretraining:
        for d in batch_data:
            d['sample'] = sample
        model = pretrain_ms_hgatlink(
            model, batch_data, adata, sample_key, condition_key,
            n_epochs=model_cfg['pretrain_epochs'], device=device, verbose=verbose)
    score_formula = model_cfg.get('score_formula', 'product')
    scores, weights = _run_augmented_inference(model, batch_data, aug_cfg, device, score_formula)
    t_alpha = model_cfg.get('t_alpha', 0.0)
    if t_alpha > 0:
        t_enrich = _compute_t_enrichment(batch_data, ct_std_dict, ct_count_dict, global_mean)
        scores = scores * (1 + t_alpha * np.tanh(t_enrich / 4.0))
    lfc_alpha = model_cfg.get('lfc_alpha', 0.0)
    if lfc_alpha > 0:
        lfc_pos = model_cfg.get('lfc_positive_only', False)
        lfc_enrich = _compute_lfc_enrichment(batch_data, global_mean, positive_only=lfc_pos)
        scores = scores * (1 + lfc_alpha * np.tanh(lfc_enrich / 4.0))
    spec_alpha = model_cfg.get('specificity_alpha', 0.0)
    if spec_alpha > 0:
        scores = _apply_specificity_modulation(batch_data, scores, spec_alpha)
    esm2_alpha = model_cfg.get('esm2_alpha', 0.0)
    if esm2_alpha > 0:
        esm2_enrich = _compute_esm2_enrichment(batch_data, _get_esm2_dict())
        scores = scores * (1 + esm2_alpha * np.tanh(esm2_enrich))
    kge_alpha = model_cfg.get('kge_alpha', 0.0)
    if kge_alpha > 0:
        kge_enrich = _compute_kge_enrichment(batch_data, _get_kge_dict())
        scores = scores * (1 + kge_alpha * np.tanh(kge_enrich))
    quality_lr_cutoff = model_cfg.get('quality_lr_cutoff', 0)
    if quality_lr_cutoff > 0:
        scores = _apply_quality_cutoff(batch_data, valid_lr, scores, quality_lr_cutoff)
    disease_alpha = model_cfg.get('disease_alpha', 0.0)
    if disease_alpha > 0:
        scores = _apply_disease_weights(batch_data, scores, disease_alpha,
                                         model_cfg.get('_disease_weights', {}))
    results = _collect_results(batch_data, scores, weights, sample, sample_key)
    del sample_adata, ct_expr_dict, ct_std_dict, ct_count_dict, global_mean
    gc.collect()
    return results


# ============================================================================
# 预训练辅助函数
# ============================================================================

def _compute_specificity_targets(batch_data):
    """为每个 LR 对内的 CT pair 计算特异性软目标。
    对于每个 LR 对，计算 expr_product 在 CT pair 间的 Z-score -> sigmoid -> 软标签 (0,1)。
    Z-score 高 = 该 CT pair 相对富集 = 目标接近1；Z-score 低 = 不富集 = 目标接近0。"""
    from collections import defaultdict
    lr_groups = defaultdict(list)
    for i, d in enumerate(batch_data):
        key = (d['ligand'], d['receptor'])
        lr_groups[key].append((i, d['lig_expr'] * d['rec_expr']))
    targets = np.full(len(batch_data), 0.5)
    for entries in lr_groups.values():
        if len(entries) < 2:
            continue
        indices, values = zip(*entries)
        values = np.array(values)
        std = values.std()
        if std < 1e-8:
            continue
        z = (values - values.mean()) / (std + 1e-8)
        soft = 1.0 / (1.0 + np.exp(-z))
        for idx, t in zip(indices, soft):
            targets[idx] = t
    return targets


def _build_class_weights(batch_data, adata, sample_key, condition_key, device):
    """构建样本类别权重（用于类别平衡）"""
    if adata is None or sample_key is None or condition_key is None:
        return None
    labels = adata.obs[condition_key].values
    unique, counts = np.unique(labels, return_counts=True)
    w = (1.0 / counts) / (1.0 / counts).sum()
    lbl_w = dict(zip(unique, w))
    s2l = dict(zip(adata.obs[sample_key].values, adata.obs[condition_key].values))
    weights = [lbl_w[s2l.get(d['sample'], unique[0])] for d in batch_data]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _prepare_pretrain_tensors(batch_data, device):
    """将批量数据转为归一化的 Tensor 字典"""
    lig_vals = [d['lig_expr'] for d in batch_data]
    rec_vals = [d['rec_expr'] for d in batch_data]
    lm, ls = np.mean(lig_vals), np.std(lig_vals) + 1e-8
    rm, rs = np.mean(rec_vals), np.std(rec_vals) + 1e-8
    lig_n = [(x - lm) / ls for x in lig_vals]
    rec_n = [(x - rm) / rs for x in rec_vals]
    spec_targets = _compute_specificity_targets(batch_data)
    return {
        'lig_idx':   torch.tensor([d['lig_idx'] for d in batch_data], dtype=torch.long, device=device),
        'rec_idx':   torch.tensor([d['rec_idx'] for d in batch_data], dtype=torch.long, device=device),
        'lig_expr':  torch.tensor([[x] for x in lig_n], dtype=torch.float32, device=device),
        'rec_expr':  torch.tensor([[x] for x in rec_n], dtype=torch.float32, device=device),
        'source_ct': torch.tensor([d['source_ct_idx'] for d in batch_data], dtype=torch.long, device=device),
        'target_ct': torch.tensor([d['target_ct_idx'] for d in batch_data], dtype=torch.long, device=device),
        'expr_prod': torch.tensor([d['lig_expr'] * d['rec_expr']
                                    for d in batch_data], dtype=torch.float32, device=device),
        'spec_targets': torch.tensor(spec_targets, dtype=torch.float32, device=device),
    }


def _compute_pretrain_loss(model, t, sample_weights):
    """特异性感知预训练损失（主BCE + 三尺度辅助BCE + 重建 + 熵正则）"""
    expr_prod = t['expr_prod'].unsqueeze(-1)
    scores, scale_w = model(t['lig_idx'], t['rec_idx'], t['lig_expr'], t['rec_expr'],
                            t['source_ct'], t['target_ct'], expr_product=expr_prod)
    targets = t['spec_targets']
    # 主损失：融合后特异性预测（用 raw modulation 的 logit）
    spec_loss = F.binary_cross_entropy_with_logits(
        model._last_modulation.squeeze(-1), targets)
    # 辅助损失：三尺度独立特异性预测
    gene_cat = torch.cat([model._lig_feat, model._rec_feat], dim=-1)
    aux_gene = F.binary_cross_entropy_with_logits(
        model.aux_gene_head(gene_cat).squeeze(-1), targets)
    aux_pathway = F.binary_cross_entropy_with_logits(
        model.aux_pathway_head(model._pathway_feat).squeeze(-1), targets)
    aux_ct = F.binary_cross_entropy_with_logits(
        model.aux_ct_head(model._ct_feat).squeeze(-1), targets)
    # 辅助：重建 + 熵正则
    lig_emb = model.gene_encoder.gene_embedding(t['lig_idx'])
    rec_emb = model.gene_encoder.gene_embedding(t['rec_idx'])
    recon = (F.mse_loss(torch.sigmoid(lig_emb.mean(1, True)), torch.sigmoid(t['lig_expr'])) +
             F.mse_loss(torch.sigmoid(rec_emb.mean(1, True)), torch.sigmoid(t['rec_expr'])))
    probs = scale_w.mean(0)
    entropy = (probs * torch.log(probs + 1e-8)).sum()
    total = spec_loss + 0.3 * (aux_gene + aux_pathway + aux_ct) + 0.1 * recon + 0.01 * entropy
    return total * sample_weights.mean() if sample_weights is not None else total


def pretrain_ms_hgatlink(model, batch_data, adata, sample_key, condition_key,
                          n_epochs=50, device='cuda', verbose=False,
                          max_pretrain_entries=100000):
    """MS-HGATLink 自监督预训练（三目标：重建+排序+尺度熵正则）"""
    import torch.optim as optim
    if len(batch_data) > max_pretrain_entries:
        idx = np.random.choice(len(batch_data), max_pretrain_entries, replace=False)
        batch_data = [batch_data[i] for i in idx]
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    sample_weights = _build_class_weights(batch_data, adata, sample_key, condition_key, device)
    tensors = _prepare_pretrain_tensors(batch_data, device)
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = _compute_pretrain_loss(model, tensors, sample_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  预训练 [{epoch+1}/{n_epochs}] loss={loss.item():.4f}")
    model.eval()
    return model


# ============================================================================
# 主推断函数
# ============================================================================

def _run_single_seed(seed, adata, model_cfg, aug_cfg, gene_to_idx, ct_to_idx,
                     n_genes, n_cell_types, resource, sample_key, groupby,
                     enable_pretraining, condition_key, max_lr_pairs,
                     min_cells, use_raw, device, verbose):
    """用单个随机种子运行完整推断，返回��果列表"""
    model, enable_pt = _init_ms_model(
        n_genes, n_cell_types, model_cfg, device, '', enable_pretraining, False,
        seed=seed, gene_to_idx=gene_to_idx)
    results = []
    samples = adata.obs[sample_key].unique()
    for idx, sample in enumerate(samples):
        if verbose:
            print(f"  [seed={seed}] 样本 {idx+1}/{len(samples)}: {sample}")
        sample_results = _process_sample_and_collect(
            sample, adata, model, resource, gene_to_idx, ct_to_idx,
            sample_key, groupby, aug_cfg, model_cfg, enable_pt,
            condition_key, max_lr_pairs, min_cells, use_raw, device, verbose=False)
        results.extend(sample_results)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def _average_seed_results(seed_dfs, verbose):
    """对多种子结果取平均（分数和尺度权重）"""
    result = seed_dfs[0].copy()
    score_cols = ['ms_hgatlink_score', 'gene_weight', 'pathway_weight', 'ct_weight']
    for col in score_cols:
        vals = np.stack([df[col].values for df in seed_dfs])
        result[col] = vals.mean(axis=0)
    if verbose:
        stds = np.stack([df['ms_hgatlink_score'].values for df in seed_dfs]).std(axis=0)
        print(f"  多种子平均完成: {len(seed_dfs)} seeds, 分数std={stds.mean():.4f}")
    return result


def ms_hgatlink_inference(adata, groupby='cell_type', sample_key='sample',
                           use_raw=False, resource_name='consensus',
                           max_lr_pairs=300, min_cells=5, verbose=True,
                           device=None, enable_pretraining=False,
                           pretrain_epochs=50, condition_key=None,
                           enable_data_augmentation=False, n_augmentations=1,
                           **kwargs):
    """MS-HGATLink 推断接口（兼容 LIANA）。支持多种子平均以提升稳定性。"""
    try:
        import liana as li
    except ImportError:
        raise ImportError("需要 liana 包：pip install liana")
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_name = kwargs.get('dataset_name', 'default')
    aug_cfg = _get_aug_config(dataset_name, enable_data_augmentation, n_augmentations)
    model_cfg = _get_model_config(dataset_name, pretrain_epochs)
    # 加载疾病相关性 LR 权重（disease_alpha > 0 时生效）
    if model_cfg.get('disease_alpha', 0.0) > 0:
        from model.disease_lr_weights import load_weights as _load_disease_weights
        model_cfg['_disease_weights'] = _load_disease_weights(dataset_name)
        if verbose:
            n_w = sum(1 for v in model_cfg['_disease_weights'].values() if v > 0)
            print(f"  疾病 LR 权重: {n_w} 个 LR 对有先验")
    # 数据集配置可覆盖函数默认参数（pipeline 未显式传入时）
    enable_pretraining = model_cfg.get('enable_pretraining', enable_pretraining)
    condition_key = model_cfg.get('condition_key', condition_key)
    gene_to_idx, ct_to_idx, n_genes, n_cts = _build_gene_ct_index(adata, groupby)
    resource = li.resource.select_resource(resource_name)
    n_seeds = model_cfg.get('n_seeds', 1)
    seeds = _SEED_LIST[:n_seeds]
    if verbose:
        print(f"MS-HGATLink 推断: dataset={dataset_name}, device={device}, seeds={seeds}")
        print(f"  基因数={n_genes}, 细胞类型数={n_cts}, emb_dim={model_cfg['gene_emb_dim']}")
    seed_dfs = []
    for seed in seeds:
        if verbose and n_seeds > 1:
            print(f"\n--- Seed {seed} ---")
        results = _run_single_seed(
            seed, adata, model_cfg, aug_cfg, gene_to_idx, ct_to_idx,
            n_genes, n_cts, resource, sample_key, groupby,
            enable_pretraining, condition_key, max_lr_pairs,
            min_cells, use_raw, device, verbose)
        if results:
            seed_dfs.append(pd.DataFrame(results))
    if not seed_dfs:
        if verbose:
            print("未找到有效交互！")
        return pd.DataFrame()
    result_df = _average_seed_results(seed_dfs, verbose) if len(seed_dfs) > 1 else seed_dfs[0]
    if verbose:
        print(f"\n总计: {len(result_df)} 条交互（sigmoid门控特异性分数）")
        print(f"  平均尺度权重 - 基因:{result_df['gene_weight'].mean():.3f}, "
              f"通路:{result_df['pathway_weight'].mean():.3f}, "
              f"细胞类型:{result_df['ct_weight'].mean():.3f}")
    return result_df


# ============================================================================
# LIANA 方法注册
# ============================================================================

try:
    from liana.method.sc._Method import Method, MethodMeta

    _meta = MethodMeta(
        method_name="MS-HGATLink",
        complex_cols=["ligand_complex", "receptor_complex"],
        add_cols=[],
        fun=lambda adata, **kw: ms_hgatlink_inference(adata, **kw),
        magnitude="ms_hgatlink_score",
        magnitude_ascending=False,
        specificity=None,
        specificity_ascending=None,
        permute=False,
        reference="MS-HGATLink: Multi-Scale Heterogeneous Graph Attention Networks"
    )
    ms_hgatlink = Method(_method=_meta)
    print("MS-HGATLink 已注册到 LIANA")

except ImportError as e:
    class _SimpleInterface:
        magnitude = "ms_hgatlink_score"
        specificity = None
        def by_sample(self, adata, **kw):
            return ms_hgatlink_inference(adata, **kw)
    ms_hgatlink = _SimpleInterface()
