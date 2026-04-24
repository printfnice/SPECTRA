# run_ensemble_reichart.py
# 依赖: e2e_pipeline.py, e2e_modules.py, sklearn.NMF, sklearn.RF
# 被依赖: 直接运行
# 职责: 方向三 — E2E(PMA+VIB) + NMF/RF 预测��融合
#       同���折内用相同 CV 分割，分别收集 soft probabilities，等权平均后计算 AUROC

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
from pipeline.e2e_utils import build_unified_config, map_ensembl_to_symbol
from model.e2e_modules import SampleAggregator, VIBLayer, E2EClassifier, supcon_loss
from model.pathway_utils import build_lr_pathway_map
from model.ms_hgatlink_method import _build_gene_ct_index, _get_model_config


# ============================================================================
# Part 1: E2E 预测收集（训练模型 + 返回 soft proba）
# ============================================================================

def _train_e2e_and_collect(train_bd, train_y, test_bd, test_y,
                           n_genes, n_cts, gene_to_idx, model_cfg,
                           config, n_classes, device, seed):
    """训练 E2E 单折，返回测试集 soft probabilities (N_test, n_classes)。"""
    n_pathways = config.get('n_pathways', 0)
    model, fusion_dim = _init_model(n_genes, n_cts, model_cfg, gene_to_idx,
                                    device, seed, n_pathways=n_pathways)
    _freeze_adapters(model, config['freeze_esm2_adapter'], config['freeze_kge_adapter'])

    vib_z = max(fusion_dim // 2, 32)
    vib = VIBLayer(fusion_dim, vib_z).to(device)
    clf_in = vib_z
    agg = SampleAggregator(dim=fusion_dim, k=config['pma_k'],
                           n_heads=config['pma_heads'],
                           dropout=config['pma_dropout']).to(device)
    clf = E2EClassifier(clf_in, n_classes, dropout=0.3,
                        l1_lambda=config['l1_lambda']).to(device)

    params = [p for m in [model, agg, vib, clf]
              for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=config['lr'],
                            weight_decay=config['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config['n_epochs'], eta_min=config['lr'] * 0.05)

    train_y_t = [torch.tensor(y, dtype=torch.long, device=device) for y in train_y]
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    use_amp = scaler is not None

    _run_training_loop(model, agg, vib, clf, train_bd, train_y_t,
                       opt, sched, scaler, config, device, use_amp)

    return _collect_mc_proba(model, agg, vib, clf, test_bd, test_y,
                             config, device, use_amp)


def _run_training_loop(model, agg, vib, clf, train_bd, train_y_t,
                       opt, sched, scaler, config, device, use_amp):
    """E2E 标准训练循环（无评估，纯前向+反向）。"""
    vib_beta = config.get('vib_beta', 0.0005)
    for _ in range(config['n_epochs']):
        for m in [model, agg, vib, clf]:
            m.train()
        opt.zero_grad()
        h_list, lbl_list, kl_total = [], [], 0.0
        for cpu_t, lbl in zip(train_bd, train_y_t):
            if cpu_t is None:
                continue
            fused = _forward_e2e(model, cpu_t, device, use_amp=use_amp)
            h = agg(fused)
            h, kl = vib(h)
            kl_total += kl
            h_list.append(h)
            lbl_list.append(lbl)
        if not h_list:
            continue
        h_s = torch.stack(h_list)
        lbl_s = torch.stack(lbl_list)
        if config.get('use_mixup', False) and h_s.size(0) >= 2:
            lam = np.random.beta(config.get('mixup_alpha', 0.4),
                                 config.get('mixup_alpha', 0.4))
            idx = torch.randperm(h_s.size(0), device=device)
            h_s = lam * h_s + (1 - lam) * h_s[idx]
            ce = (lam * F.cross_entropy(clf(h_s), lbl_s) +
                  (1 - lam) * F.cross_entropy(clf(h_s), lbl_s[idx]))
        else:
            ce = F.cross_entropy(clf(h_s), lbl_s)
        loss = ce + clf.l1_loss() + vib_beta * kl_total / len(h_list)
        if config.get('use_supcon', False) and len(h_list) >= 4:
            loss = loss + config.get('supcon_weight', 0.1) * supcon_loss(
                h_s, lbl_s, tau=config.get('supcon_tau', 0.07))
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for m in [model, agg, vib, clf]
                 for p in m.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for m in [model, agg, vib, clf]
                 for p in m.parameters() if p.requires_grad], 1.0)
            opt.step()
        sched.step()


def _collect_mc_proba(model, agg, vib, clf, test_bd, test_y,
                      config, device, use_amp):
    """MC Dropout 推理，返回 (y_true, proba)。"""
    T = config.get('mc_dropout_T', 15)
    model.eval(); agg.eval(); clf.train(); vib.train()
    y_true, proba_list = [], []
    with torch.no_grad():
        for cpu_t, lbl in zip(test_bd, test_y):
            if cpu_t is None:
                continue
            fused = _forward_e2e(model, cpu_t, device, use_amp=use_amp)
            h_base = agg(fused)
            h_b = h_base.unsqueeze(0).expand(T, -1)
            h_b, _ = vib(h_b)
            logits = clf(h_b)
            proba_list.append(F.softmax(logits, dim=-1).mean(0).cpu().numpy())
            y_true.append(lbl)
    return np.array(y_true), np.stack(proba_list)


# ============================================================================
# Part 2: NMF + RF（从 batch_data 构建 LR 特征矩阵）
# ============================================================================

def _build_vocab(all_batch_data):
    """构建全局 (lig, rec, src_ct, tgt_ct) 词表。"""
    vocab = set()
    for cpu_t in all_batch_data:
        if cpu_t is None:
            continue
        lig = cpu_t['lig_idx'].numpy()
        rec = cpu_t['rec_idx'].numpy()
        sc = cpu_t['src_ct'].numpy()
        tc = cpu_t['tgt_ct'].numpy()
        for i in range(len(lig)):
            vocab.add((int(lig[i]), int(rec[i]), int(sc[i]), int(tc[i])))
    return {k: idx for idx, k in enumerate(sorted(vocab))}


def _to_feature_vec(cpu_t, vocab):
    """单样本 batch_data → 稀疏特征向量 (|vocab|,)。"""
    if cpu_t is None:
        return None
    vec = np.zeros(len(vocab), dtype=np.float32)
    lig = cpu_t['lig_idx'].numpy()
    rec = cpu_t['rec_idx'].numpy()
    sc = cpu_t['src_ct'].numpy()
    tc = cpu_t['tgt_ct'].numpy()
    scores = cpu_t['expr_prod'].squeeze().numpy()
    for i in range(len(lig)):
        key = (int(lig[i]), int(rec[i]), int(sc[i]), int(tc[i]))
        if key in vocab:
            vec[vocab[key]] = max(vec[vocab[key]], float(scores[i]))
    return vec


def _run_nmf_rf(train_X, train_y, test_X, test_y, n_classes, seed=42):
    """NMF 降维 + RF 分类，返回 (y_true, proba)。"""
    from sklearn.decomposition import NMF
    from sklearn.ensemble import RandomForestClassifier
    n_comp = min(20, min(train_X.shape) - 1)
    nmf = NMF(n_components=n_comp, random_state=seed, max_iter=500)
    Xtr = nmf.fit_transform(train_X)
    Xte = nmf.transform(test_X)
    rf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    rf.fit(Xtr, train_y)
    proba = rf.predict_proba(Xte)
    if proba.shape[1] < n_classes:
        full = np.zeros((len(proba), n_classes))
        for j, cls in enumerate(rf.classes_):
            full[:, cls] = proba[:, j]
        proba = full
    return np.array(test_y), proba


# ============================================================================
# Part 3: 集成 CV 主循环
# ============================================================================

def run_ensemble_cv(adata, samples, labels_str, all_batch_data,
                    gene_to_idx, ct_to_idx, model_cfg, config, device):
    """K 折集成 CV：E2E + NMF/RF，等权平均 soft proba。"""
    labels_int, n_classes, label_names = _encode_labels(labels_str)
    print(f"  类别: {label_names}  n_classes={n_classes}")
    vocab = _build_vocab(all_batch_data)
    print(f"  LR×CT 特征维度: {len(vocab)}")

    splitter, groups, strat_y = _build_cv_splitter(adata, samples, labels_int, config)
    split_args = (samples, strat_y) if groups is None else (samples, strat_y, groups)
    n_genes, n_cts = len(gene_to_idx), len(ct_to_idx)

    all_aurocs_e2e, all_aurocs_nmf, all_aurocs_ens = [], [], []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(*split_args)):
        train_bd = _move_to_device([all_batch_data[i] for i in train_idx], device)
        test_bd  = _move_to_device([all_batch_data[i] for i in test_idx], device)
        train_y  = [labels_int[i] for i in train_idx]
        test_y   = [labels_int[i] for i in test_idx]

        # NMF 特征矩阵（CPU tensor → numpy）
        train_X = np.stack([_to_feature_vec(all_batch_data[i], vocab)
                            for i in train_idx if all_batch_data[i] is not None])
        test_X  = np.stack([_to_feature_vec(all_batch_data[i], vocab)
                            for i in test_idx  if all_batch_data[i] is not None])
        valid_train_y = [labels_int[i] for i in train_idx if all_batch_data[i] is not None]
        valid_test_y  = [labels_int[i] for i in test_idx  if all_batch_data[i] is not None]

        fold_e2e, fold_nmf, fold_ens = [], [], []
        for rs in range(config['n_random_states']):
            seed = 42 + rs * 7
            yt_e2e, p_e2e = _train_e2e_and_collect(
                train_bd, train_y, test_bd, test_y,
                n_genes, n_cts, gene_to_idx, model_cfg,
                config, n_classes, device, seed)
            yt_nmf, p_nmf = _run_nmf_rf(
                train_X, valid_train_y, test_X, valid_test_y, n_classes, seed)

            auroc_e2e = _compute_auroc(yt_e2e, p_e2e, n_classes)
            auroc_nmf = _compute_auroc(yt_nmf, p_nmf, n_classes)
            p_ens = 0.5 * p_e2e + 0.5 * p_nmf[:len(p_e2e)]
            auroc_ens = _compute_auroc(yt_e2e, p_ens, n_classes)

            fold_e2e.append(auroc_e2e)
            fold_nmf.append(auroc_nmf)
            fold_ens.append(auroc_ens)
            print(f"  fold={fold_idx+1} rs={rs}  "
                  f"E2E={auroc_e2e:.4f}  NMF/RF={auroc_nmf:.4f}  Ens={auroc_ens:.4f}")

        all_aurocs_e2e.append(np.mean(fold_e2e))
        all_aurocs_nmf.append(np.mean(fold_nmf))
        all_aurocs_ens.append(np.mean(fold_ens))
        print(f"  [fold {fold_idx+1}] E2E={np.mean(fold_e2e):.4f}  "
              f"NMF={np.mean(fold_nmf):.4f}  Ens={np.mean(fold_ens):.4f}")

    _report(all_aurocs_e2e, all_aurocs_nmf, all_aurocs_ens)
    return all_aurocs_ens


def _report(e2e, nmf, ens):
    print(f"\n{'='*55}")
    print(f"  E2E       : {np.mean(e2e):.4f} +/- {np.std(e2e):.4f}  {[f'{a:.3f}' for a in e2e]}")
    print(f"  NMF/RF    : {np.mean(nmf):.4f} +/- {np.std(nmf):.4f}  {[f'{a:.3f}' for a in nmf]}")
    print(f"  Ensemble  : {np.mean(ens):.4f} +/- {np.std(ens):.4f}  {[f'{a:.3f}' for a in ens]}")
    print(f"{'='*55}")


# ============================================================================
# 主入口
# ============================================================================

_DATA_DIR = _BASE_DIR / 'data'
_OUTPUT_DIR = _BASE_DIR / 'output'

def main():
    dataset_name = 'reichart'
    config = DEFAULT_CONFIG.copy()
    config['dataset_name'] = dataset_name
    device = _get_device(True)
    print(f"\n=== Ensemble Pipeline: {dataset_name} | device={device} ===")

    adata = _load_adata(dataset_name)
    dk = _DATASET_KEYS[dataset_name]
    samples, labels_str = _get_sample_labels(adata, dk['sample_key'], dk['condition_key'])
    n_samples = len(samples)
    config.update(build_unified_config(n_samples, dataset_name))
    config['dataset_name'] = dataset_name
    print(f"  N={n_samples}  VIB={config['use_vib']}  MC_T={config['mc_dropout_T']}  "
          f"epochs={config['n_epochs']}  pma_k={config['pma_k']}")

    import liana as li
    gene_to_idx, ct_to_idx, _, _ = _build_gene_ct_index(adata, dk['groupby'])
    resource = li.resource.select_resource(config['resource_name'])
    model_cfg = _get_model_config(dataset_name, 30)
    model_cfg['gene_emb_dim'] = 64
    model_cfg['dropout'] = 0.30

    lr_pathway_map, n_pathways = {}, 0
    lr_pathway_map, n_pathways = build_lr_pathway_map(_DATA_DIR)
    config['n_pathways'] = n_pathways
    config['use_pathway_prior'] = True

    print(f"  预计算 batch_data...")
    all_batch_data = _prepare_all_samples(
        samples, adata, resource, gene_to_idx, ct_to_idx,
        model_cfg, config, dk['sample_key'], dk['groupby'],
        lr_pathway_map=lr_pathway_map)
    n_valid = sum(1 for bd in all_batch_data if bd is not None)
    print(f"  有效样本: {n_valid}/{len(samples)}")

    aurocs = run_ensemble_cv(adata, samples, labels_str, all_batch_data,
                             gene_to_idx, ct_to_idx, model_cfg, config, device)
    import pandas as pd
    pd.DataFrame({'dataset': [dataset_name], 'method': ['ensemble_e2e+nmf'],
                  'mean_auroc': [round(float(np.mean(aurocs)), 4)],
                  'std_auroc': [round(float(np.std(aurocs)), 4)],
                  'fold_aurocs': [str([round(a, 4) for a in aurocs])]}
                 ).to_csv(_OUTPUT_DIR / 'reichart_ensemble_e2e_nmf.csv', index=False)
    print(f"  结果已保存")


if __name__ == '__main__':
    main()
