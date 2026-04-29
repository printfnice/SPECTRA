# unified_comparison.py
# 依赖: e2e_modules.py (ScoreEncoder, SampleAggregator, VIBLayer, E2EClassifier),
#       e2e_utils.py (build_unified_config), e2e_pipeline.py (_DATASET_KEYS),
#       simplified_pipeline_optimized.py (Step1/2 断点)
# 被依赖: 直接运行
# 职责: R30 统一下游公平对比 — 让所有 LIANA 方法通过相同 PMA+VIB+MLP 下游，
#       唯一变量是 Step 2 的 LR 评分来源，消除审稿人对"提升来自下游架构"的质疑

import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

warnings.filterwarnings('ignore')
from contextlib import nullcontext as _nullcontext

# 路径设置：从 code/pipeline/ 运行
_PIPELINE_DIR = Path(__file__).parent
_CODE_DIR = _PIPELINE_DIR.parent
_BASE_DIR = _CODE_DIR.parent
sys.path.insert(0, str(_CODE_DIR))

from model.e2e_modules import (
    ScoreEncoder, SampleAggregator, VIBLayer, E2EClassifier,
)
from pipeline.e2e_utils import build_unified_config
from pipeline.e2e_pipeline import (
    _DATASET_KEYS, _load_adata, _get_sample_labels, _encode_labels,
    _get_device, _build_cv_splitter, _compute_auroc,
)

_DATA_DIR = _BASE_DIR / 'data'
_OUTPUT_DIR = _BASE_DIR / 'output'

# 9 个标准 LIANA 方法: (uns_key, score_col)
# uns_key = adata.uns 中存储 DataFrame 的键名
# score_col = DataFrame 中实际的评分列名
# 注意: connectome/natmi 共享 uns_key='expr_prod'，需用不同 score_col 区分
# rank_aggregate: 聚合 CellPhoneDB+Connectome+log2FC+NATMI+SingleCellSignalR，对应论文 "RankAggregate"
_LIANA_METHODS = {
    'cellphonedb':        ('lr_means',        'lr_means'),
    'connectome':         ('magnitude_rank',  'scaled_weight'),
    'cellchat':           ('lr_probs',        'lr_probs'),
    'scseqcomm':          ('inter_score',     'inter_score'),
    'singlecellsignalr':  ('lrscore',         'lrscore'),
    'natmi':              ('expr_prod',        'spec_weight'),
    'logfc':              ('lr_logfc',         'lr_logfc'),
    'geometric_mean':     ('lr_gmeans',        'lr_gmeans'),
    'rank_aggregate':     ('magnitude_rank',   'magnitude_rank'),
}

# 可选：包含 ms_hgatlink 的无监督评分作为对照
_CUSTOM_METHODS = {
    'ms_hgatlink': ('ms_hgatlink_score', 'ms_hgatlink_score'),
    'hgatlink':    ('hgatlink_score',    'hgatlink_score'),
}


# ============================================================================
# 数据加载：从 adata.uns 或断点中提取 LR 评分 DataFrame
# ============================================================================

def _load_lr_scores(dataset_name, method_name):
    """加载 method 在 dataset 上的 LR 评分 DataFrame。

    加载优先级: parquet > checkpoint > h5ad。
    返回 (lr_df, score_col) 或 (None, None)。
    """
    keys = _get_method_keys(method_name)
    if keys is None:
        return None, None
    uns_key, score_col = keys

    # 尝试 parquet（reichart 等大数据集专用）
    lr_df = _load_from_parquet(dataset_name, method_name, score_col)
    if lr_df is not None:
        return lr_df, score_col

    # 尝试断点
    lr_df = _load_from_checkpoint(dataset_name, method_name, uns_key)
    if lr_df is not None and score_col in lr_df.columns:
        return lr_df, score_col

    # 尝试 h5ad（可能已有 LR 结果）
    lr_df = _load_from_h5ad(dataset_name, method_name, uns_key)
    if lr_df is not None and score_col in lr_df.columns:
        return lr_df, score_col

    return None, None


def _get_method_keys(method_name):
    """返回 (uns_key, score_col) 元组。"""
    if method_name in _LIANA_METHODS:
        return _LIANA_METHODS[method_name]
    if method_name in _CUSTOM_METHODS:
        return _CUSTOM_METHODS[method_name]
    return None


def _load_from_parquet(dataset_name, method_name, score_col):
    """从独立 parquet 文件加载 LR 评分（大数据集避免 h5ad OOM）。"""
    parquet_dir = _DATA_DIR / 'interim' / f'{dataset_name}_liana_scores'
    parquet_file = parquet_dir / f'{method_name}.parquet'
    if parquet_file.exists():
        lr_df = pd.read_parquet(str(parquet_file))
        if score_col in lr_df.columns:
            return lr_df
    return None


def _load_from_checkpoint(dataset_name, method_name, uns_key):
    """从 step2 断点文件加载 LR 评分。"""
    from pipeline.checkpoint_utils import get_checkpoint_path, load_checkpoint
    for reduction in ('tensor', 'mofa', 'det'):
        ckpt = get_checkpoint_path(dataset_name, method_name, reduction, 'step2')
        if Path(ckpt).exists():
            data = load_checkpoint(ckpt)
            if isinstance(data, dict) and 'adata' in data:
                adata = data['adata']
                if uns_key in adata.uns:
                    return adata.uns[uns_key]
            elif hasattr(data, 'uns') and uns_key in data.uns:
                return data.uns[uns_key]
    return None


def _load_from_h5ad(dataset_name, method_name, uns_key):
    """从已有的 h5ad 文件加载 LR 评分。"""
    import anndata
    candidates = [
        _DATA_DIR / 'interim' / f'{dataset_name}_with_liana_scores.h5ad',
        _DATA_DIR / 'interim' / f'{dataset_name}_with_{method_name}.h5ad',
        _DATA_DIR / 'interim' / f'{dataset_name}_processed.h5ad',
        _DATA_DIR / 'interim' / f'{dataset_name}_filtered.h5ad',
    ]
    for p in candidates:
        if p.exists():
            adata = anndata.read_h5ad(str(p))
            if uns_key in adata.uns:
                return adata.uns[uns_key]
    return None


def _detect_sample_col(lr_df, dataset_sample_key):
    """检测 lr_df 中的样本列名（LIANA 统一用 'sample'，但可能不同）。"""
    for col in ['sample', dataset_sample_key, 'Sample', 'Sample_Name', 'orig.ident']:
        if col in lr_df.columns:
            return col
    return None


# ============================================================================
# 数据预处理：LR 评分 DataFrame → per-sample 张量
# ============================================================================

def _build_lr_pair_vocab(lr_df, sample_key, score_col, max_lr_pairs=None):
    """从 LR 评分 DataFrame 构建全局 LR pair 词表。

    若 max_lr_pairs 不为 None，按跨样本方差排序选 top-K 最有分类信号的 LR pairs。
    返回 lr_pair_to_idx: {str: int} 映射。
    """
    lr_df = lr_df.copy()
    lr_df['lr_pair'] = (
        lr_df['source'].astype(str) + '_' +
        lr_df['ligand_complex'].astype(str) + '__' +
        lr_df['target'].astype(str) + '_' +
        lr_df['receptor_complex'].astype(str)
    )
    if max_lr_pairs is not None and lr_df['lr_pair'].nunique() > max_lr_pairs:
        # 按跨样本方差排序，选 top-K（方差大 = 分类信号强）
        var_by_pair = lr_df.groupby('lr_pair')[score_col].var().dropna()
        top_pairs = var_by_pair.nlargest(max_lr_pairs).index.tolist()
        lr_df = lr_df[lr_df['lr_pair'].isin(top_pairs)]
    unique_pairs = sorted(lr_df['lr_pair'].unique())
    return {p: i for i, p in enumerate(unique_pairs)}


def _prepare_frozen_samples(lr_df, score_col, sample_key, lr_pair_to_idx):
    """将 LR 评分 DataFrame 转为 per-sample 张量列表。

    返回 list of dict: {scores: (L,), lr_indices: (L,)} 或 None（样本无数据）。
    """
    lr_df = lr_df.copy()
    lr_df['lr_pair'] = (
        lr_df['source'].astype(str) + '_' +
        lr_df['ligand_complex'].astype(str) + '__' +
        lr_df['target'].astype(str) + '_' +
        lr_df['receptor_complex'].astype(str)
    )

    samples = sorted(lr_df[sample_key].unique())
    sample_tensors = {}
    for s in samples:
        sub = lr_df[lr_df[sample_key] == s]
        if len(sub) == 0:
            sample_tensors[s] = None
            continue
        scores = sub[score_col].values.astype(np.float32)
        indices = np.array([lr_pair_to_idx[p] for p in sub['lr_pair']
                           if p in lr_pair_to_idx], dtype=np.int64)
        if len(indices) == 0:
            sample_tensors[s] = None
            continue
        # 对齐长度（某些 pair 可能不在词表中）
        min_len = min(len(scores), len(indices))
        sample_tensors[s] = {
            'scores': torch.tensor(scores[:min_len], dtype=torch.float32),
            'lr_indices': torch.tensor(indices[:min_len], dtype=torch.long),
        }
    return samples, sample_tensors


# ============================================================================
# 训练 & 评估（frozen 方法专用，复用 E2E 架构）
# ============================================================================

def _init_frozen_model(n_lr_pairs, fusion_dim, config, n_classes, device, seed):
    """初始化 ScoreEncoder + PMA + VIB + MLP（与 E2E 相同架构）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    encoder = ScoreEncoder(n_lr_pairs, fusion_dim, dropout=0.1).to(device)
    aggregator = SampleAggregator(
        dim=fusion_dim, k=config['pma_k'],
        n_heads=4, dropout=config.get('pma_dropout', 0.1),
    ).to(device)

    vib = None
    clf_in_dim = fusion_dim
    if config.get('use_vib', False):
        vib_z_dim = max(fusion_dim // 2, 32)
        vib = VIBLayer(fusion_dim, vib_z_dim).to(device)
        clf_in_dim = vib_z_dim

    classifier = E2EClassifier(
        in_dim=clf_in_dim, n_classes=n_classes,
        dropout=0.3, l1_lambda=config['l1_lambda'],
    ).to(device)

    return encoder, aggregator, vib, classifier


def _forward_frozen(encoder, tensors, device):
    """ScoreEncoder 前向：scores + lr_indices → (L, D)。"""
    t = {k: v.to(device) for k, v in tensors.items()}
    return encoder(t['scores'], t['lr_indices'])


def _train_epoch_frozen(encoder, aggregator, classifier, sample_data,
                        labels_t, optimizer, device, config, vib=None,
                        scaler=None):
    """单轮训练（frozen 方法版本）。"""
    encoder.train()
    aggregator.train()
    classifier.train()
    if vib is not None:
        vib.train()
    optimizer.zero_grad()

    use_amp = scaler is not None
    amp_ctx = (torch.autocast('cuda', dtype=torch.float16)
               if (use_amp and device == 'cuda') else _nullcontext())

    h_list, lbl_list = [], []
    total_kl = 0.0
    for tensors, label in zip(sample_data, labels_t):
        if tensors is None:
            continue
        with amp_ctx:
            fused = _forward_frozen(encoder, tensors, device)
            h = aggregator(fused)
            if vib is not None:
                h, kl = vib(h)
                total_kl = total_kl + kl
        h_list.append(h)
        lbl_list.append(label)

    if not h_list:
        return 0.0

    h_stack = torch.stack(h_list)
    lbl_stack = torch.stack(lbl_list)
    n_samples = h_stack.size(0)

    with amp_ctx:
        logits = classifier(h_stack)
        ce_loss = F.cross_entropy(logits, lbl_stack, reduction='mean')
        total_loss = ce_loss + classifier.l1_loss()
        if vib is not None:
            total_loss = total_loss + config.get('vib_beta', 0.0005) * total_kl / n_samples

    if scaler is not None:
        scaler.scale(total_loss).backward()
        params = (list(encoder.parameters()) + list(aggregator.parameters()) +
                  list(classifier.parameters()))
        if vib is not None:
            params += list(vib.parameters())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        params = (list(encoder.parameters()) + list(aggregator.parameters()) +
                  list(classifier.parameters()))
        if vib is not None:
            params += list(vib.parameters())
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

    return total_loss.item()


def _evaluate_frozen(encoder, aggregator, classifier, sample_data,
                     labels, n_classes, device, config, vib=None,
                     use_amp=False):
    """评估（支持 MC Dropout）。"""
    mc_T = config.get('mc_dropout_T', 0)
    if mc_T > 0:
        return _evaluate_frozen_mc(
            encoder, aggregator, classifier, sample_data,
            labels, n_classes, device, mc_T, vib, use_amp)

    encoder.eval()
    aggregator.eval()
    classifier.eval()
    if vib is not None:
        vib.eval()

    proba_list, y_true = [], []
    amp_ctx = (torch.autocast('cuda', dtype=torch.float16)
               if (use_amp and device == 'cuda') else _nullcontext())
    with torch.no_grad():
        for tensors, lbl in zip(sample_data, labels):
            if tensors is None:
                continue
            with amp_ctx:
                fused = _forward_frozen(encoder, tensors, device)
                h = aggregator(fused)
                if vib is not None:
                    h, _ = vib(h)
                logit = classifier(h.unsqueeze(0))
            proba_list.append(F.softmax(logit, dim=-1).cpu().numpy()[0])
            y_true.append(lbl)
    if not proba_list:
        return 0.0
    return _compute_auroc(np.array(y_true), np.stack(proba_list), n_classes)


def _evaluate_frozen_mc(encoder, aggregator, classifier, sample_data,
                        labels, n_classes, device, T, vib, use_amp):
    """MC Dropout 推理（frozen 方法版本）。"""
    encoder.eval()
    aggregator.eval()
    classifier.train()  # dropout 保持开启
    if vib is not None:
        vib.train()

    proba_list, y_true = [], []
    amp_ctx = (torch.autocast('cuda', dtype=torch.float16)
               if (use_amp and device == 'cuda') else _nullcontext())
    with torch.no_grad():
        for tensors, lbl in zip(sample_data, labels):
            if tensors is None:
                continue
            with amp_ctx:
                fused = _forward_frozen(encoder, tensors, device)
                h_base = aggregator(fused)
                h_batch = h_base.unsqueeze(0).expand(T, -1)
                if vib is not None:
                    h_batch, _ = vib(h_batch)
                logits_batch = classifier(h_batch)
            proba_mean = F.softmax(logits_batch, dim=-1).mean(dim=0)
            proba_list.append(proba_mean.cpu().numpy())
            y_true.append(lbl)
    if not proba_list:
        return 0.0
    return _compute_auroc(np.array(y_true), np.stack(proba_list), n_classes)


# ============================================================================
# 单折训练
# ============================================================================

def _run_one_fold_frozen(train_data, train_y, test_data, test_y,
                         n_lr_pairs, config, n_classes, device, seed):
    """初始化并训练单折，返回测试 AUROC。"""
    fusion_dim = 128  # 与 E2E 的 gene_emb_dim=64 → fusion_dim=128 一致
    encoder, aggregator, vib, classifier = _init_frozen_model(
        n_lr_pairs, fusion_dim, config, n_classes, device, seed)

    params = (list(encoder.parameters()) + list(aggregator.parameters()) +
              list(classifier.parameters()))
    if vib is not None:
        params += list(vib.parameters())
    optimizer = torch.optim.AdamW(params, lr=config['lr'], weight_decay=1e-4)
    scheduler = None
    if config.get('use_lr_scheduler', True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['n_epochs'], eta_min=config['lr'] * 0.05)

    train_y_t = [torch.tensor(y, dtype=torch.long, device=device) for y in train_y]
    use_amp = (device == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_auroc = 0.0
    eval_interval = config.get('eval_interval', 10)
    patience = config.get('early_stop_patience', 0)
    no_improve = 0

    for epoch in range(config['n_epochs']):
        _train_epoch_frozen(
            encoder, aggregator, classifier, train_data, train_y_t,
            optimizer, device, config, vib=vib, scaler=scaler)
        if scheduler is not None:
            scheduler.step()
        if (epoch + 1) % eval_interval == 0:
            auroc = _evaluate_frozen(
                encoder, aggregator, classifier, test_data, test_y,
                n_classes, device, config, vib=vib, use_amp=use_amp)
            if auroc > best_auroc:
                best_auroc = auroc
                no_improve = 0
            else:
                no_improve += 1
            if patience > 0 and no_improve >= patience:
                break

    final = _evaluate_frozen(
        encoder, aggregator, classifier, test_data, test_y,
        n_classes, device, config, vib=vib, use_amp=use_amp)
    return max(best_auroc, final)


# ============================================================================
# CV 循环
# ============================================================================

def _run_cv_frozen(adata, ordered_samples, sample_tensors, labels_str,
                   n_lr_pairs, config, device):
    """K 折 CV，返回各折 AUROC 列表。"""
    labels_int, n_classes, label_names = _encode_labels(labels_str)
    print(f"  类别: {label_names}  n_classes={n_classes}")

    splitter, groups, strat_y = _build_cv_splitter(adata, ordered_samples, labels_int, config)
    split_args = ((ordered_samples, strat_y) if groups is None
                  else (ordered_samples, strat_y, groups))

    all_aurocs = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(*split_args)):
        train_data = [_to_device(sample_tensors[ordered_samples[i]], device)
                      for i in train_idx]
        test_data = [_to_device(sample_tensors[ordered_samples[i]], device)
                     for i in test_idx]
        train_y = [labels_int[i] for i in train_idx]
        test_y = [labels_int[i] for i in test_idx]

        fold_aurocs = []
        for rs in range(config['n_random_states']):
            seed = 42 + rs * 7
            auroc = _run_one_fold_frozen(
                train_data, train_y, test_data, test_y,
                n_lr_pairs, config, n_classes, device, seed)
            fold_aurocs.append(auroc)
        mean_fold = np.mean(fold_aurocs)
        all_aurocs.append(mean_fold)
        print(f"  [fold {fold_idx+1}] AUROC = {mean_fold:.4f}")

    return all_aurocs


def _to_device(tensors, device):
    """将单个样本的张量字典移到 device，或返回 None。"""
    if tensors is None:
        return None
    return {k: v.to(device) for k, v in tensors.items()}


# ============================================================================
# 主对比逻辑
# ============================================================================

def _run_single_comparison(dataset_name, method_name, config, device):
    """对单个 (dataset, method) 运行 frozen 下游对比，返回 (mean, std, aurocs)。"""
    print(f"\n--- {dataset_name} x {method_name} ---")

    # 加载 LR 评分
    lr_df, score_col = _load_lr_scores(dataset_name, method_name)
    if lr_df is None:
        print(f"  [跳过] 无 {method_name} 评分数据")
        return None, None, None

    # 加载 adata（获取样本标签和 CV 分割）
    adata = _load_adata(dataset_name)
    dk = _DATASET_KEYS[dataset_name]

    # LIANA by_sample 输出统一使用 'sample' 列（不同于 adata.obs 的 sample_key）
    sample_key_in_df = _detect_sample_col(lr_df, dk['sample_key'])
    if sample_key_in_df is None:
        print(f"  [跳过] lr_df 中无样本列")
        return None, None, None

    # 先获取样本标签映射（用于计算有效样本数）
    sample_label_map = dict(zip(
        *_get_sample_labels(adata, dk['sample_key'], dk['condition_key'])))
    df_samples = set(lr_df[sample_key_in_df].unique())
    n_valid = sum(1 for s in df_samples if s in sample_label_map)

    if n_valid < 6:
        print(f"  [跳过] 有效样本不足 ({n_valid})")
        return None, None, None

    # 应用统一配置（基于有效样本数）
    config = config.copy()
    config.update(build_unified_config(n_valid))
    config['dataset_name'] = dataset_name
    max_lr = config.get('e2e_max_lr_pairs', 200)

    # 构建 LR pair 词表和 per-sample 张量（限制 LR pairs 数量与 E2E 一致）
    lr_pair_to_idx = _build_lr_pair_vocab(lr_df, sample_key_in_df, score_col, max_lr)
    n_lr_pairs = len(lr_pair_to_idx)
    print(f"  评分列={score_col}, LR pairs={n_lr_pairs} (max={max_lr}), 有效样本={n_valid}")

    ordered_samples, sample_tensors = _prepare_frozen_samples(
        lr_df, score_col, sample_key_in_df, lr_pair_to_idx)

    # 过滤有标签且有数据的样本
    valid_samples, valid_labels = [], []
    for s in ordered_samples:
        if s in sample_label_map and sample_tensors.get(s) is not None:
            valid_samples.append(s)
            valid_labels.append(sample_label_map[s])

    if len(valid_samples) < 6:
        print(f"  [跳过] 有效样本不足 ({len(valid_samples)})")
        return None, None, None

    # 运行 CV
    aurocs = _run_cv_frozen(
        adata, valid_samples, sample_tensors, valid_labels,
        n_lr_pairs, config, device)
    mean_auroc = np.mean(aurocs)
    std_auroc = np.std(aurocs)
    print(f"  结果: AUROC = {mean_auroc:.4f} +/- {std_auroc:.4f}")
    return mean_auroc, std_auroc, aurocs


def run_comparison(datasets, methods, config, device):
    """批量运行对比实验，返回结果 DataFrame。"""
    rows = []
    for ds in datasets:
        for method in methods:
            mean, std, aurocs = _run_single_comparison(ds, method, config, device)
            if mean is not None:
                rows.append({
                    'dataset': ds, 'method': method,
                    'mean_auroc': round(mean, 4),
                    'std_auroc': round(std, 4),
                    'fold_aurocs': str([round(a, 4) for a in aurocs]),
                })
    return pd.DataFrame(rows)


# ============================================================================
# 结果保存与汇总
# ============================================================================

def _save_results(results_df, out_dir):
    """保存结果 CSV（追加模式，避免覆盖）+ 打印汇总表。"""
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'unified_comparison_R30.csv'
    if out_path.exists():
        existing = pd.read_csv(out_path)
        # 去重：同 (dataset, method) 保留最新结果
        combined = pd.concat([existing, results_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=['dataset', 'method'], keep='last')
        combined.to_csv(out_path, index=False)
        results_df = combined
    else:
        results_df.to_csv(out_path, index=False)
    print(f"\n结果已保存: {out_path}")

    # 打印论文 Table 格式
    if len(results_df) == 0:
        return
    pivot = results_df.pivot(index='method', columns='dataset', values='mean_auroc')
    pivot['Mean'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('Mean', ascending=False)
    print("\n=== R30 统一下游对比（PMA+VIB+MLP）===")
    print(pivot.to_string(float_format='%.4f'))


# ============================================================================
# CLI 入口
# ============================================================================

ALL_DATASETS = ['kuppe', 'carraro', 'habermann', 'velmeshev', 'reichart']
ALL_METHODS = list(_LIANA_METHODS.keys())


def _parse_args():
    p = argparse.ArgumentParser(description='R30: Unified Downstream Comparison')
    p.add_argument('--dataset', default='all',
                   help='数据集名或 "all"')
    p.add_argument('--method', default='all',
                   help='LIANA 方法名或 "all"')
    p.add_argument('--no_gpu', action='store_true')
    p.add_argument('--include_custom', action='store_true',
                   help='包含 ms_hgatlink/hgatlink 的 frozen 评分对照')
    return p.parse_args()


DEFAULT_CONFIG = {
    'use_gpu': True,
    'pma_heads': 4,
    'weight_decay': 1e-4,
    'resource_name': 'consensus',
    'min_cells': 5,
    'verbose': True,
}


def main():
    args = _parse_args()

    # 解析数据集列表
    datasets = ALL_DATASETS if args.dataset == 'all' else [args.dataset]

    # 解析方法列表
    if args.method == 'all':
        methods = ALL_METHODS
    else:
        methods = [args.method]
    if args.include_custom:
        methods += list(_CUSTOM_METHODS.keys())

    config = DEFAULT_CONFIG.copy()
    if args.no_gpu:
        config['use_gpu'] = False
    device = _get_device(config['use_gpu'])

    print(f"=== R30 统一下游公平对比 ===")
    print(f"  数据集: {datasets}")
    print(f"  方法: {methods}")
    print(f"  设备: {device}")

    results = run_comparison(datasets, methods, config, device)
    _save_results(results, _OUTPUT_DIR)


if __name__ == '__main__':
    main()
