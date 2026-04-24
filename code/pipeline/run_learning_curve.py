# run_learning_curve.py
# 依赖: e2e_pipeline.py, e2e_utils.py, run_liana_baselines_5fold.py
# 被依赖: plot_learning_curve.py
# 职责: 学习曲线——按训练集比例对比 SPECTRA E2E vs RF(raw LR)
#       证据：性能提升与样本量负相关 → 小样本设计有效性

import sys, json, warnings
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

_PIPELINE_DIR = Path(__file__).parent
_CODE_DIR     = _PIPELINE_DIR.parent
_BASE_DIR     = _CODE_DIR.parent
sys.path.insert(0, str(_CODE_DIR))

DATASET   = 'reichart'
RATIOS    = [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
N_SPLITS  = 5
N_REPEATS = 3
SCORE_KEY = 'cellchat'
DATA_DIR  = _BASE_DIR / 'data'
OUT_DIR   = _PIPELINE_DIR / 'output' / 'data'


# ── LR 矩阵（供 RF 使用）────────────────────────────────────────────────────────

def build_X_y(adata):
    pq = DATA_DIR / 'interim' / 'reichart_liana_scores' / f'{SCORE_KEY}.parquet'
    df = pd.read_parquet(str(pq))
    adata.uns[SCORE_KEY] = df
    sk = 'Sample'; ck = 'disease'
    df = df.copy()
    df['lr_pair'] = (df['source'].astype(str) + '_' +
                     df['ligand_complex'].astype(str) + '__' +
                     df['target'].astype(str) + '_' +
                     df['receptor_complex'].astype(str))
    sc = SCORE_KEY if SCORE_KEY in df.columns else df.select_dtypes('number').columns[0]
    pivot = df.pivot_table(index=sk, columns='lr_pair', values=sc,
                           aggfunc='mean', fill_value=0)
    X     = pivot.values.astype(np.float32)
    names = np.array(pivot.index.tolist())
    cmap  = adata.obs[[sk, ck]].drop_duplicates().set_index(sk)[ck].to_dict()
    y     = LabelEncoder().fit_transform([cmap[s] for s in names])
    print(f"  LR X={X.shape}, 类别数={len(np.unique(y))}")
    return X, y, names


# ── RF baseline ───────────────────────────────────────────────────────────────

def rf_auroc(X_tr, y_tr, X_te, y_te, seed=0):
    n_cls = len(np.unique(np.concatenate([y_tr, y_te])))
    clf   = RandomForestClassifier(500, random_state=seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    prob  = clf.predict_proba(X_te)
    try:
        if n_cls == 2:
            return float(roc_auc_score(y_te, prob[:, 1]))
        return float(roc_auc_score(y_te, prob, multi_class='ovr', average='macro'))
    except Exception:
        return float('nan')


# ── SPECTRA E2E（复用 run_e2e_cv 公开接口）─────────────────────────────────────

def _setup_e2e(adata, device):
    """一次性初始化 E2E 所需的 resource、索引、batch_data（全样本）。"""
    import liana as li
    from e2e_pipeline import (
        _load_adata, _get_sample_labels, _build_gene_ct_index,
        _prepare_all_samples, _get_model_config, _DATASET_KEYS,
    )
    from pipeline.e2e_utils import build_unified_config

    dk         = _DATASET_KEYS[DATASET]
    sample_key = dk['sample_key']
    groupby    = dk['groupby']
    cond_key   = dk['condition_key']

    samples, labels_str = _get_sample_labels(adata, sample_key, cond_key)
    n_samples = len(samples)
    print(f"  E2E 初始化: {n_samples} 样本")

    from e2e_pipeline import DEFAULT_CONFIG
    from pipeline.e2e_utils import build_unified_config
    config = DEFAULT_CONFIG.copy()
    unified = build_unified_config(n_samples, DATASET)
    config.update(unified)
    config.update({'dataset_name': DATASET,
                   'use_gpu': device.type == 'cuda',
                   'collect_gate_weights': False,
                   'collect_embedding': False,
                   'collect_metrics': False,
                   'verbose': False,
                   'n_random_states': 1})

    gene_to_idx, ct_to_idx, _, _ = _build_gene_ct_index(adata, groupby)
    resource  = li.resource.select_resource(config['resource_name'])
    model_cfg = _get_model_config(DATASET, 30)
    model_cfg['gene_emb_dim'] = 64
    model_cfg['dropout']      = 0.30

    print("  预计算 batch_data（全样本，一次性）...")
    all_bd = _prepare_all_samples(
        samples, adata, resource, gene_to_idx, ct_to_idx,
        model_cfg, config, sample_key, groupby,
    )
    bd_map = {s: all_bd[i] for i, s in enumerate(samples)}
    return samples, labels_str, bd_map, gene_to_idx, ct_to_idx, model_cfg, config


def e2e_auroc_subset(adata, tr_names, te_names, labels_str_all,
                     bd_map, gene_to_idx, ct_to_idx,
                     model_cfg, config, device):
    """在给定训练/测试子集上跑 run_e2e_cv 的单折版本。"""
    from e2e_pipeline import run_e2e_cv
    from pipeline.e2e_utils import build_unified_config

    all_names  = list(tr_names) + list(te_names)
    labels_map = {s: l for s, l in zip(
        list(set(list(tr_names) + list(te_names))),
        [''] * len(all_names)   # placeholder
    )}
    # 重建标签串
    from e2e_pipeline import _DATASET_KEYS, _get_sample_labels
    dk       = _DATASET_KEYS[DATASET]
    sk       = dk['sample_key']
    ck       = dk['condition_key']
    from e2e_pipeline import _load_adata as _ep_load
    # 直接从 bd_map 键（sample name）还原 labels
    # labels_str_all 是全样本对应标签列表
    name2label = dict(zip(labels_str_all[0], labels_str_all[1]))

    sub_names  = list(tr_names)
    sub_labels = [name2label[s] for s in sub_names if s in name2label]
    if len(set(sub_labels)) < 2:
        return float('nan')

    sub_bd    = [bd_map.get(s) for s in sub_names]
    te_names_ = [s for s in te_names if s in bd_map]
    te_labels = [name2label[s] for s in te_names_ if s in name2label]

    # 用 n_cv_folds=2 的 run_e2e_cv 仅训练+评估一折
    cfg2 = config.copy()
    cfg2['n_cv_folds']      = 2
    cfg2['n_random_states'] = 1

    # 把 te 数据拼到 sub 后面，让 CV 自动把它分到测试集（workaround）
    all_subs   = sub_names + te_names_
    all_bds    = [bd_map.get(s) for s in all_subs]
    all_labels = sub_labels + te_labels

    try:
        result = run_e2e_cv(
            adata, all_subs, all_labels, all_bds,
            gene_to_idx, ct_to_idx, model_cfg, cfg2, device
        )
        aurocs = result[0] if isinstance(result, tuple) else result
        return float(np.mean(aurocs))
    except Exception as e:
        print(f"      E2E err: {e}")
        return float('nan')


# ── 主实验 ────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 加载数据
    from e2e_pipeline import _load_adata
    adata = _load_adata(DATASET)
    X, y, names = build_X_y(adata)
    n_total = len(names)

    # E2E 全局初始化（batch_data 只计算一次）
    samples_all, labels_list, bd_map, gene_to_idx, ct_to_idx, model_cfg, config = \
        _setup_e2e(adata, device)
    # 构建 name→label 映射
    from e2e_pipeline import _DATASET_KEYS, _get_sample_labels
    dk         = _DATASET_KEYS[DATASET]
    all_samples, all_labels_str = _get_sample_labels(
        adata, dk['sample_key'], dk['condition_key']
    )
    name2label = dict(zip(all_samples, all_labels_str))
    labels_ref = (all_samples, all_labels_str)  # tuple for passing around

    records = []
    outer_cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=0)

    for fold_i, (tr_pool_idx, te_idx) in enumerate(outer_cv.split(names, y)):
        tr_pool   = names[tr_pool_idx]
        te_names  = names[te_idx]
        y_tr_pool = y[tr_pool_idx]
        print(f"\n[Fold {fold_i+1}/{N_SPLITS}] pool={len(tr_pool)}, test={len(te_names)}")

        for ratio in RATIOS:
            n_sub = max(len(np.unique(y)), int(len(tr_pool) * ratio))
            for rep in range(N_REPEATS):
                seed = fold_i * 10000 + int(ratio * 1000) + rep
                if ratio >= 1.0:
                    sub_idx = np.arange(len(tr_pool))
                else:
                    sss = StratifiedShuffleSplit(1, train_size=n_sub, random_state=seed)
                    sub_idx, _ = next(sss.split(tr_pool, y_tr_pool))

                tr_sub = tr_pool[sub_idx]
                X_tr   = X[tr_pool_idx][sub_idx]
                y_tr   = y[tr_pool_idx][sub_idx]
                X_te   = X[te_idx]; y_te = y[te_idx]

                # RF
                try:
                    a_rf = rf_auroc(X_tr, y_tr, X_te, y_te, seed)
                except Exception as e:
                    a_rf = float('nan'); print(f"  RF err: {e}")

                # E2E
                try:
                    a_e2e = e2e_auroc_subset(
                        adata, tr_sub, te_names, labels_ref,
                        bd_map, gene_to_idx, ct_to_idx,
                        model_cfg, config, device
                    )
                except Exception as e:
                    a_e2e = float('nan'); print(f"  E2E err: {e}")

                rec = dict(fold=fold_i+1, ratio=ratio,
                           n_train=int(len(tr_sub)), repeat=rep,
                           auroc_rf=a_rf, auroc_e2e=a_e2e)
                records.append(rec)
                print(f"  {ratio:.0%} n={len(tr_sub):3d} rep={rep} "
                      f"RF={a_rf:.4f} E2E={a_e2e:.4f}")

    # 保存
    out = OUT_DIR / 'learning_curve_results.json'
    json.dump({'dataset': DATASET, 'ratios': RATIOS, 'records': records},
              open(str(out), 'w'), indent=2)
    print(f"\n已保存: {out}")

    df = pd.DataFrame(records)
    print(f"\n{'ratio':>6}  {'N':>6}  {'RF':>8}  {'E2E':>8}  {'Δ':>7}")
    for r in RATIOS:
        s = df[df['ratio'] == r]
        print(f"{r:>6.0%}  {s['n_train'].mean():>6.0f}  "
              f"{s['auroc_rf'].mean():>8.4f}  {s['auroc_e2e'].mean():>8.4f}  "
              f"{(s['auroc_e2e']-s['auroc_rf']).mean():>+7.4f}")


if __name__ == '__main__':
    main()
