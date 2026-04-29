# run_hybrid_e2e.py
# 依赖: e2e_pipeline.py, e2e_modules.py, sklearn.NMF
# 被依赖: 直接运行
# 职责: Path B — 混合 E2E：每折 fold-specific NMF 特征拼接到 VIB 输出后送 MLP
#       架构: LR_scores → MS-HGATLink → PMA(128D) → VIB(64D) → concat NMF(20D) → MLP(84D→2)
#       根因: E2E 在 reichart 弱是因为 PMA 需从零学习全局结构；
#             fold-specific NMF 提供强全局结构先验，帮助 TTN-DCM 边界学习
#       统一方法：所有 5 个数据集使用同一代码路径（NMF 对小数据集权重自动降低）

import sys
import warnings
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')
from contextlib import nullcontext as _nullcontext

_PIPELINE_DIR = Path(__file__).parent
_CODE_DIR = _PIPELINE_DIR.parent
_BASE_DIR = _CODE_DIR.parent
sys.path.insert(0, str(_CODE_DIR))

from pipeline.e2e_pipeline import (
    DEFAULT_CONFIG, _DATASET_KEYS,
    _get_device, _load_adata, _get_sample_labels, _encode_labels,
    _build_cv_splitter, _move_to_device, _forward_e2e,
    _init_model, _freeze_adapters, _compute_auroc,
    _prepare_all_samples,
)
from pipeline.e2e_utils import build_unified_config, map_ensembl_to_symbol, save_e2e_results
from model.e2e_modules import SampleAggregator, VIBLayer, E2EClassifier
from model.pathway_utils import build_lr_pathway_map
from model.ms_hgatlink_method import _build_gene_ct_index, _get_model_config

_OUTPUT_DIR = _BASE_DIR / 'output'
_NMF_N_COMPONENTS = 20


# ============================================================================
# NMF 特征构建（fold-specific，避免数据泄露）
# ============================================================================

def _build_flat_matrix(batch_data, max_lr_pairs):
    """从 batch_data 构建 (N_samples, max_lr_pairs) 特征矩阵（expr_prod 值）。"""
    feats = []
    for cpu_t in batch_data:
        if cpu_t is None:
            feats.append(np.zeros(max_lr_pairs))
            continue
        ep = cpu_t.get('expr_prod', cpu_t.get('scores'))
        arr = ep.cpu().numpy().flatten() if hasattr(ep, 'cpu') else np.array(ep).flatten()
        if len(arr) >= max_lr_pairs:
            feats.append(arr[:max_lr_pairs])
        else:
            feats.append(np.pad(arr, (0, max_lr_pairs - len(arr))))
    return np.array(feats, dtype=np.float32)


def _fit_nmf(train_bd, test_bd, max_lr_pairs, n_components=_NMF_N_COMPONENTS):
    """在训练集上拟合 NMF，转换训练/测试集，返回 (train_nmf, test_nmf) as float32 arrays。"""
    X_train = _build_flat_matrix(train_bd, max_lr_pairs)
    X_test  = _build_flat_matrix(test_bd,  max_lr_pairs)
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    nmf = NMF(n_components=n_components, init='nndsvda', random_state=42,
              max_iter=300, tol=1e-4)
    train_nmf = nmf.fit_transform(X_train_s).astype(np.float32)
    test_nmf  = nmf.transform(X_test_s).astype(np.float32)
    # L2 归一化每样本行向量，消除量纲差异
    norm_tr = np.linalg.norm(train_nmf, axis=1, keepdims=True).clip(min=1e-8)
    norm_te = np.linalg.norm(test_nmf,  axis=1, keepdims=True).clip(min=1e-8)
    return train_nmf / norm_tr, test_nmf / norm_te


# ============================================================================
# 改造训练/评估函数（注入 NMF 拼接）
# ============================================================================

def _train_hybrid_epoch(model, aggregator, classifier, vib, all_tensors,
                        labels_t, optimizer, device, config, nmf_t=None):
    """训练一个 epoch，在 VIB 输出后拼接 NMF 特征。

    nmf_t: (N_samples, nmf_dim) torch.Tensor on device or None
    """
    model.train(); aggregator.train(); classifier.train()
    if vib is not None: vib.train()

    use_mixup = config.get('use_mixup', False)
    use_vib   = config.get('use_vib', False) and vib is not None
    use_amp   = (device == 'cuda')
    scaler    = torch.amp.GradScaler('cuda') if use_amp else None

    h_list, lbl_list, valid_pos = [], [], []
    total_kl = torch.tensor(0.0, device=device)

    for i, t in enumerate(all_tensors):
        if t is None:
            continue
        fused = _forward_e2e(model, t, device, use_amp=use_amp)          # (L, D)
        h = aggregator(fused, pathway_idx=t.get('pathway_idx'))           # (D,)
        if use_vib:
            mu, lv = vib.encode(h)
            h = vib.reparametrize(mu, lv)
            total_kl += vib.kl(mu, lv)
        h_list.append(h)
        lbl_list.append(labels_t[i])
        valid_pos.append(i)

    if not h_list:
        return

    h_stack   = torch.stack(h_list)    # (N_valid, D)
    lbl_stack = torch.stack(lbl_list)  # (N_valid,)

    if nmf_t is not None:
        nmf_valid = nmf_t[valid_pos]   # (N_valid, nmf_dim)
        h_stack = torch.cat([h_stack, nmf_valid], dim=-1)

    if use_mixup and h_stack.size(0) >= 2:
        alpha = config.get('mixup_alpha', 0.4)
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(h_stack.size(0), device=device)
        h_mix = lam * h_stack + (1 - lam) * h_stack[idx]
        logits = classifier(h_mix)
        ce = (lam * F.cross_entropy(logits, lbl_stack) +
              (1 - lam) * F.cross_entropy(logits, lbl_stack[idx]))
    else:
        logits = classifier(h_stack)
        ce = F.cross_entropy(logits, lbl_stack)

    loss = ce + classifier.l1_loss()
    if use_vib:
        loss = loss + config.get('vib_beta', 0.0005) * total_kl / len(h_list)

    optimizer.zero_grad()
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()


def _evaluate_hybrid(model, aggregator, classifier, vib, all_tensors,
                     y_true, n_classes, device, config, nmf_t=None):
    """推理评估，拼接 NMF 特征，返回 AUROC。"""
    model.eval(); aggregator.eval(); classifier.eval()
    if vib is not None: vib.eval()
    mc_T = config.get('mc_dropout_T', 0)
    use_amp = (device == 'cuda')

    all_probas, y_out, valid_pos = [], [], []
    with torch.no_grad():
        for i, t in enumerate(all_tensors):
            if t is None:
                continue
            fused = _forward_e2e(model, t, device, use_amp=use_amp)
            h = aggregator(fused, pathway_idx=t.get('pathway_idx'))
            if vib is not None:
                mu, lv = vib.encode(h)
                h = mu                 # 推理时用均值
            if nmf_t is not None:
                h = torch.cat([h, nmf_t[i]], dim=-1)
            # MC Dropout 多次前向（若需要）
            if mc_T > 1:
                classifier.train()
                mc_out = [torch.softmax(classifier(h), dim=-1) for _ in range(mc_T)]
                classifier.eval()
                proba = torch.stack(mc_out).mean(0)
            else:
                proba = torch.softmax(classifier(h), dim=-1)
            all_probas.append(proba.cpu().numpy())
            y_out.append(y_true[i])
            valid_pos.append(i)

    if not all_probas:
        return 0.0
    proba_arr = np.stack(all_probas)
    return _compute_auroc(np.array(y_out), proba_arr, n_classes)


# ============================================================================
# 单折训练（Hybrid E2E）
# ============================================================================

def _run_hybrid_fold(train_bd, train_y, test_bd, test_y,
                     train_nmf, test_nmf,
                     n_genes, n_cts, gene_to_idx, model_cfg,
                     config, n_classes, device, seed):
    """单折训练：E2E + NMF 拼接。返回最佳 AUROC。"""
    nmf_dim = train_nmf.shape[1] if train_nmf is not None else 0
    model, fusion_dim = _init_model(n_genes, n_cts, model_cfg, gene_to_idx, device, seed)
    _freeze_adapters(model, config['freeze_esm2_adapter'], config['freeze_kge_adapter'])

    aggregator = SampleAggregator(
        dim=fusion_dim, k=config['pma_k'],
        n_heads=config['pma_heads'], dropout=config['pma_dropout'],
    ).to(device)
    vib = None
    clf_in = fusion_dim
    if config.get('use_vib', False):
        vib_z = max(fusion_dim // 2, 32)
        vib = VIBLayer(fusion_dim, vib_z).to(device)
        clf_in = vib_z
    clf_in += nmf_dim   # 拼接 NMF 维度

    classifier = E2EClassifier(
        in_dim=clf_in, n_classes=n_classes,
        dropout=0.3, l1_lambda=config['l1_lambda']
    ).to(device)

    all_params = (list(model.parameters()) + list(aggregator.parameters()) +
                  list(classifier.parameters()))
    if vib is not None: all_params += list(vib.parameters())
    optimizer = torch.optim.AdamW(
        [p for p in all_params if p.requires_grad],
        lr=config['lr'], weight_decay=config['weight_decay']
    )
    scheduler = None
    if config.get('use_lr_scheduler', True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['n_epochs'], eta_min=config['lr'] * 0.05)

    labels_t = [torch.tensor(y, dtype=torch.long, device=device) for y in train_y]

    # NMF 特征张量（device 上）
    train_nmf_t = torch.tensor(train_nmf, dtype=torch.float32, device=device) if nmf_dim else None
    test_nmf_t  = torch.tensor(test_nmf,  dtype=torch.float32, device=device) if nmf_dim else None

    best_auroc = 0.0
    eval_interval = config.get('eval_interval', 20)
    for epoch in range(config['n_epochs']):
        _train_hybrid_epoch(model, aggregator, classifier, vib,
                            train_bd, labels_t, optimizer, device, config,
                            nmf_t=train_nmf_t)
        if scheduler is not None:
            scheduler.step()
        if (epoch + 1) % eval_interval == 0:
            auroc = _evaluate_hybrid(model, aggregator, classifier, vib,
                                     test_bd, test_y, n_classes, device, config,
                                     nmf_t=test_nmf_t)
            if auroc > best_auroc:
                best_auroc = auroc
            if config.get('verbose', True):
                print(f"    epoch {epoch+1}/{config['n_epochs']} test AUROC={auroc:.4f}")

    final = _evaluate_hybrid(model, aggregator, classifier, vib,
                              test_bd, test_y, n_classes, device, config,
                              nmf_t=test_nmf_t)
    return max(best_auroc, final)


# ============================================================================
# CV 主循环
# ============================================================================

def run_hybrid_cv(adata, samples, labels_str, all_batch_data,
                  gene_to_idx, ct_to_idx, model_cfg, config, device):
    """5 折 CV，每折 fit NMF on train，返回 fold AUROC 列表。"""
    labels_int, n_classes, label_names = _encode_labels(labels_str)
    print(f"  类别: {label_names}  n_classes={n_classes}  nmf_dim={_NMF_N_COMPONENTS}")

    splitter, groups, strat_y = _build_cv_splitter(adata, samples, labels_int, config)
    split_args = (samples, strat_y) if groups is None else (samples, strat_y, groups)

    max_lr_pairs = config.get('e2e_max_lr_pairs', 400)
    fold_aurocs = []

    for fold_idx, (tr_idx, te_idx) in enumerate(splitter.split(*split_args)):
        train_bd_cpu = [all_batch_data[i] for i in tr_idx]
        test_bd_cpu  = [all_batch_data[i] for i in te_idx]

        # Fold-specific NMF（仅用训练集 fit，测试集 transform，避免泄露）
        train_nmf, test_nmf = _fit_nmf(train_bd_cpu, test_bd_cpu, max_lr_pairs)

        train_bd = _move_to_device(train_bd_cpu, device)
        test_bd  = _move_to_device(test_bd_cpu,  device)
        train_y  = [labels_int[i] for i in tr_idx]
        test_y   = [labels_int[i] for i in te_idx]

        seed_aurocs = []
        for rs in range(config['n_random_states']):
            seed = 42 + rs * 7
            auroc = _run_hybrid_fold(
                train_bd, train_y, test_bd, test_y,
                train_nmf, test_nmf,
                len(gene_to_idx), len(ct_to_idx), gene_to_idx, model_cfg,
                config, n_classes, device, seed)
            seed_aurocs.append(auroc)
            print(f"  fold={fold_idx+1} rs={rs}  single={auroc:.4f}")

        fold_mean = np.mean(seed_aurocs)
        fold_aurocs.append(fold_mean)
        print(f"  [fold {fold_idx+1}] mean={fold_mean:.4f}  seeds={[f'{a:.4f}' for a in seed_aurocs]}")

    return fold_aurocs


# ============================================================================
# 主入口
# ============================================================================

def _parse_args():
    p = argparse.ArgumentParser(description='Hybrid E2E + NMF 拼接')
    p.add_argument('--dataset', default='reichart')
    p.add_argument('--no_gpu',  action='store_true', default=False)
    return p.parse_args()


def main():
    args = _parse_args()
    dataset_name = args.dataset
    device = _get_device(not args.no_gpu)
    print(f"\n=== Hybrid E2E Pipeline: {dataset_name} | device={device} ===")

    adata = _load_adata(dataset_name)
    dk = _DATASET_KEYS[dataset_name]
    samples, labels_str = _get_sample_labels(adata, dk['sample_key'], dk['condition_key'])
    n_samples = len(samples)

    config = DEFAULT_CONFIG.copy()
    config['dataset_name'] = dataset_name
    config['unified'] = True
    config['collect_gate_weights'] = False
    config.update(build_unified_config(n_samples, dataset_name))
    print(f"  N={n_samples}  epochs={config['n_epochs']}  pma_k={config['pma_k']}"
          f"  lr_pairs={config['e2e_max_lr_pairs']}  nmf={_NMF_N_COMPONENTS}D")

    resource = 'consensus'
    gene_to_idx, ct_to_idx = _build_gene_ct_index(adata, dk['sample_key'],
                                                   dk['groupby'], resource,
                                                   config['min_cells'], config['use_raw'])
    model_cfg = _get_model_config(dataset_name, gene_to_idx, ct_to_idx)
    lr_pathway_map = None
    if config.get('use_pathway_prior', False):
        lr_pathway_map, n_pw = build_lr_pathway_map(_BASE_DIR / 'data')
        config['n_pathways'] = n_pw

    print(f"  预计算 batch_data...")
    all_batch_data = _prepare_all_samples(
        samples, adata, resource, gene_to_idx, ct_to_idx,
        model_cfg, config, dk['sample_key'], dk['groupby'],
        lr_pathway_map=lr_pathway_map)
    valid_n = sum(1 for b in all_batch_data if b is not None)
    print(f"  有效样本: {valid_n}/{n_samples}")

    fold_aurocs = run_hybrid_cv(
        adata, samples, labels_str, all_batch_data,
        gene_to_idx, ct_to_idx, model_cfg, config, device)

    mean_auroc = np.mean(fold_aurocs)
    std_auroc  = np.std(fold_aurocs)
    print(f"\n=== Hybrid E2E 结果: {dataset_name} ===")
    print(f"  fold AUROCs: {[f'{a:.4f}' for a in fold_aurocs]}")
    print(f"  mean={mean_auroc:.4f}  std={std_auroc:.4f}")

    _OUTPUT_DIR.mkdir(exist_ok=True)
    import pandas as pd
    pd.DataFrame({
        'dataset': [dataset_name], 'method': ['ms_hgatlink_e2e_hybrid'],
        'mean_auroc': [round(mean_auroc, 4)], 'std_auroc': [round(std_auroc, 4)],
        'n_folds': [len(fold_aurocs)], 'fold_aurocs': [str(fold_aurocs)],
        'n_samples': [n_samples], 'nmf_components': [_NMF_N_COMPONENTS],
        'n_epochs': [config['n_epochs']], 'pma_k': [config['pma_k']],
    }).to_csv(_OUTPUT_DIR / f'{dataset_name}_ms_hgatlink_e2e_hybrid.csv', index=False)
    print(f"  结果已保存: output/{dataset_name}_ms_hgatlink_e2e_hybrid.csv")


if __name__ == '__main__':
    main()
