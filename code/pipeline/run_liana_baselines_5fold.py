# run_liana_baselines_5fold.py
# 依赖: liana, scanpy, sklearn, cell2cell, utils/classify_utils.py, process/processer.py
# 被依赖: run_all_supplementary.sh
# 职责: 8 LIANA 方法 × 4 小数据集，5-fold StratifiedKFold 统一协议，丰富指标

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))
os.chdir(Path(__file__).parent)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef, confusion_matrix
)

# ============================================================================
# 配置
# ============================================================================

DATASETS = ['kuppe', 'carraro', 'habermann', 'velmeshev', 'reichart']
METHODS = ['cellphonedb', 'connectome', 'cellchat', 'scseqcomm',
           'singlecellsignalr', 'natmi', 'logfc', 'rank_aggregate']

N_CV_FOLDS = 5
N_RANDOM_STATES = 3   # reichart 大数据集用 3，小数据集也统一 3（节省时间）
N_ESTIMATORS = 500

_DATASET_META = {
    'kuppe':     {'sample_key': 'sample',      'groupby': 'cell_type_original',
                  'condition_key': 'patient_group', 'n_factors': 10},
    'habermann': {'sample_key': 'Sample_Name', 'groupby': 'celltype',
                  'condition_key': 'Status',         'n_factors': 15},
    'velmeshev': {'sample_key': 'sample',      'groupby': 'cluster',
                  'condition_key': 'diagnosis',      'n_factors': 30},
    'carraro':   {'sample_key': 'orig.ident',  'groupby': 'major',
                  'condition_key': 'type',            'n_factors': 8},
    # reichart: 171样本/70 donors，使用 StratifiedGroupKFold（与 SPECTRA 公平协议一致）
    'reichart':  {'sample_key': 'Sample',      'groupby': 'cell_type',
                  'condition_key': 'disease',         'n_factors': 10,
                  'donor_key': 'donor_id'},
}

OUTPUT_DIR = Path('output/liana_baselines_5fold')
DATA_DIR = Path('../../data')


# ============================================================================
# 工具函数
# ============================================================================

def _load_adata(dataset):
    candidates = [
        DATA_DIR / 'interim' / f'{dataset}_processed.h5ad',
        DATA_DIR / 'interim' / f'{dataset}_filtered.h5ad',
    ]
    for p in candidates:
        if p.exists():
            print(f"  加载: {p}")
            adata = sc.read_h5ad(str(p))
            # reichart 使用 ENSEMBL ID，需转换为 gene symbol
            first = str(adata.var_names[0]) if len(adata.var_names) > 0 else ''
            if first.startswith('ENSG') and 'feature_name' in adata.var.columns:
                new_names = adata.var['feature_name'].tolist()
                adata.var_names = new_names
                print(f"  ENSEMBL→Symbol 映射: {len(new_names)} 基因（via feature_name列）")
            return adata
    raise FileNotFoundError(f"找不到 {dataset} 的预处理数据: {candidates}")


def _get_method_map():
    from liana.method import (cellphonedb, connectome, cellchat, scseqcomm,
                               singlecellsignalr, natmi, logfc, rank_aggregate)
    return {
        'cellphonedb': cellphonedb, 'connectome': connectome,
        'cellchat': cellchat, 'scseqcomm': scseqcomm,
        'singlecellsignalr': singlecellsignalr, 'natmi': natmi,
        'logfc': logfc, 'rank_aggregate': rank_aggregate,
    }


def _get_score_key(method_obj):
    return method_obj.magnitude if method_obj.magnitude else method_obj.specificity


def _run_liana_method(adata, method_name, method_obj, meta, force=False):
    sk = _get_score_key(method_obj)
    if sk in adata.uns and not force:
        print(f"    [{method_name}] 已缓存 score_key={sk}，跳过推断")
        return sk
    print(f"    [{method_name}] 运行推断...")
    result = method_obj.by_sample(
        adata,
        groupby=meta['groupby'],
        sample_key=meta['sample_key'],
        use_raw=False,
        verbose=False,
        n_perms=None,
        inplace=False,
    )
    adata.uns[sk] = result
    print(f"    [{method_name}] 推断完成: {len(result)} 行")
    return sk


def _build_xlr_y(adata, score_key, meta):
    """构建 LR 评分矩阵和标签，返回 (X_lr, y, sample_names, groups_or_None)。"""
    df = adata.uns[score_key]
    sk = meta['sample_key']
    ck = meta['condition_key']
    # 确保有评分列
    score_col = score_key if score_key in df.columns else df.select_dtypes('number').columns[0]
    df['lr_pair'] = (df['source'].astype(str) + '_' +
                     df['ligand_complex'].astype(str) + '__' +
                     df['target'].astype(str) + '_' +
                     df['receptor_complex'].astype(str))
    pivot = df.pivot_table(index=sk, columns='lr_pair', values=score_col,
                           aggfunc='mean', fill_value=0)
    X = pivot.values
    sample_names = pivot.index.tolist()
    # y from adata.obs
    cond_map = adata.obs[[sk, ck]].drop_duplicates().set_index(sk)[ck].to_dict()
    y_raw = [cond_map[s] for s in sample_names]
    y = LabelEncoder().fit_transform(y_raw)
    # groups（donor_id）用于 reichart GroupKFold
    groups = None
    dk = meta.get('donor_key')
    if dk and dk in adata.obs.columns:
        donor_map = adata.obs[[sk, dk]].drop_duplicates().set_index(sk)[dk].to_dict()
        groups = np.array([donor_map.get(s, s) for s in sample_names])
    return X, y, sample_names, groups


def _run_tensor_reduction(adata, score_key, meta, dataset):
    """CV-外 tensor 降维，返回 (X_reduced, y_tensor, sample_order)。"""
    from utils.classify_utils import run_tensor_c2c
    run_tensor_c2c(
        adata, score_key=score_key,
        sample_key=meta['sample_key'],
        condition_key=meta['condition_key'],
        dataset_name=dataset,
        n_factors=meta['n_factors'],
        use_gpu=True,
    )
    # 加载保存的 factor scores
    csv_path = DATA_DIR / 'results' / 'tensor' / dataset / f'{score_key}.csv'
    if not csv_path.exists():
        # fallback: skip tensor, use raw X_lr
        print(f"    警告: tensor CSV 未找到 ({csv_path})，回退到原始 X_lr")
        return None, None, None
    factor_df = pd.read_csv(str(csv_path), index_col=0)
    if 'Category' not in factor_df.columns:
        print(f"    警告: Category 列缺失，回退到原始 X_lr")
        return None, None, None
    X_reduced = factor_df.drop(columns=['Category']).values
    y_raw = factor_df['Category'].values
    y = LabelEncoder().fit_transform(y_raw)
    sample_order = factor_df.index.tolist()
    return X_reduced, y, sample_order


def _compute_fold_metrics(y_true, y_pred, y_prob):
    n_classes = len(np.unique(y_true))
    result = {}
    try:
        if n_classes == 2:
            result['auroc'] = roc_auc_score(y_true, y_prob[:, 1])
            result['auprc'] = average_precision_score(y_true, y_prob[:, 1])
        else:
            result['auroc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            result['auprc'] = average_precision_score(y_true, y_prob, average='macro')
    except Exception:
        result['auroc'] = np.nan
        result['auprc'] = np.nan
    result['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    result['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    result['bacc'] = balanced_accuracy_score(y_true, y_pred)
    result['mcc'] = matthews_corrcoef(y_true, y_pred)
    if n_classes == 2:
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            result['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            result['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        except Exception:
            result['sensitivity'] = result['specificity'] = np.nan
    else:
        result['sensitivity'] = result['specificity'] = np.nan
    return result


def _add_roc_pr_to_metrics(fm, y_te, y_prob):
    """在折指标 dict 中追加 ROC/PR 曲线数据和原始预测，供后续绘图。"""
    from sklearn.metrics import roc_curve, precision_recall_curve
    n_classes = y_prob.shape[1]
    fm['y_true'] = y_te.tolist()
    fm['y_prob']  = y_prob.tolist()
    if n_classes == 2:
        try:
            pos = y_prob[:, 1]
            fpr_r, tpr_r, _ = roc_curve(y_te, pos)
            grid = np.linspace(0, 1, 100)
            fm['fpr'] = grid.tolist()
            fm['tpr'] = np.interp(grid, fpr_r, tpr_r).tolist()
            p_r, r_r, _ = precision_recall_curve(y_te, pos)
            fm['pr_recall']    = r_r.tolist()
            fm['pr_precision'] = p_r.tolist()
        except Exception:
            pass


def _run_cv_rich(X, y, n_folds=5, n_states=3, n_estimators=500, groups=None):
    """多次 5-fold CV，返回所有 fold 的丰富指标列表（含 ROC/PR 原始数据）。
    groups 非 None 时使用 StratifiedGroupKFold（用于 reichart 的 donor_id 分组）。
    """
    from sklearn.model_selection import StratifiedGroupKFold
    all_metrics = []
    use_group_kfold = groups is not None
    for rs in range(n_states):
        if use_group_kfold:
            skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=rs * 7)
            split_iter = skf.split(X, y, groups)
        else:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rs * 7)
            split_iter = skf.split(X, y)
        for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            if len(np.unique(y_tr)) < 2:
                continue
            clf = RandomForestClassifier(
                n_estimators=n_estimators, max_features='sqrt',
                random_state=rs, oob_score=False, n_jobs=-1, class_weight='balanced'
            )
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            y_prob = clf.predict_proba(X_te)
            if len(np.unique(y_te)) < 2:
                continue
            fm = _compute_fold_metrics(y_te, y_pred, y_prob)
            fm['rs'] = rs
            fm['fold'] = fold_idx
            _add_roc_pr_to_metrics(fm, y_te, y_prob)
            all_metrics.append(fm)
    return all_metrics


def _save_metrics_json(dataset, method_name, all_metrics, agg, out_dir):
    """保存完整折指标（含 ROC/PR）到 JSON，文件名：{dataset}_{method}_liana_metrics.json。"""
    import json
    data_dir = Path(out_dir).parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    # 汇总统计（从 agg 提取）
    summary = {k: v for k, v in agg.items() if k != 'fold_aurocs'}
    payload = {
        'dataset':      dataset,
        'method':       method_name,
        'pipeline':     'liana_tensor_rf',
        'summary':      summary,
        'fold_metrics': all_metrics,
    }
    out_path = data_dir / f'{dataset}_{method_name}_liana_metrics.json'
    with open(str(out_path), 'w') as f:
        json.dump(payload, f)
    print(f"  [Metrics JSON] 已保存: {out_path}")


def _agg_metrics(all_metrics):
    """汇总所有 fold 指标，返回 mean/std/fold_aurocs。"""
    if not all_metrics:
        return {}
    df = pd.DataFrame(all_metrics)
    result = {}
    for col in ['auroc', 'auprc', 'f1_macro', 'f1_weighted', 'bacc', 'mcc',
                'sensitivity', 'specificity']:
        vals = df[col].dropna().values
        result[f'mean_{col}'] = round(float(np.mean(vals)), 4) if len(vals) > 0 else np.nan
        result[f'std_{col}'] = round(float(np.std(vals)), 4) if len(vals) > 0 else np.nan
    result['fold_aurocs'] = str([round(float(v), 4) for v in df['auroc'].dropna().tolist()])
    result['n_folds_total'] = len(all_metrics)
    return result


# ============================================================================
# 主逻辑
# ============================================================================

def run_one(dataset, method_name):
    out_path = OUTPUT_DIR / f'{dataset}_{method_name}_5fold.csv'
    json_path = OUTPUT_DIR.parent / 'data' / f'{dataset}_{method_name}_liana_metrics.json'
    if out_path.exists() and json_path.exists():
        print(f"  [skip] {dataset}/{method_name} 已存在: {out_path}")
        return
    if out_path.exists() and not json_path.exists():
        print(f"  [rerun for metrics] {dataset}/{method_name} CSV存在但JSON缺失")

    print(f"\n{'='*60}")
    print(f"  {dataset} × {method_name}  [{datetime.now().strftime('%H:%M:%S')}]")
    print(f"{'='*60}")

    meta = _DATASET_META[dataset]
    method_map = _get_method_map()
    method_obj = method_map[method_name]

    adata = _load_adata(dataset)
    score_key = _run_liana_method(adata, method_name, method_obj, meta)

    # 构建 X_lr 矩阵（也用于 fallback）
    X_lr, y_lr, sample_names, groups = _build_xlr_y(adata, score_key, meta)
    print(f"  X_lr 矩阵: {X_lr.shape}" + (f"  groups(donors)={len(set(groups))}" if groups is not None else ""))

    # 尝试 tensor 降维
    X_reduced, y_tensor, _ = _run_tensor_reduction(adata, score_key, meta, dataset)
    if X_reduced is not None:
        X_feat, y_feat = X_reduced, y_tensor
        feat_type = 'tensor'
        print(f"  X_reduced (tensor): {X_reduced.shape}")
    else:
        X_feat, y_feat = X_lr, y_lr
        feat_type = 'raw_lr'
        print(f"  使用原始 X_lr: {X_lr.shape}")

    all_metrics = _run_cv_rich(X_feat, y_feat, N_CV_FOLDS, N_RANDOM_STATES, N_ESTIMATORS,
                               groups=groups)
    agg = _agg_metrics(all_metrics)
    print(f"  AUROC = {agg.get('mean_auroc', 'nan'):.4f} ± {agg.get('std_auroc', 'nan'):.4f}")
    _save_metrics_json(dataset, method_name, all_metrics, agg, OUTPUT_DIR)

    row = {
        'dataset': dataset, 'method': method_name, 'score_key': score_key,
        'feat_type': feat_type,
        'n_folds': N_CV_FOLDS, 'n_random_states': N_RANDOM_STATES,
        'n_samples': len(y_feat), 'n_features': X_feat.shape[1],
    }
    row.update(agg)
    result_df = pd.DataFrame([row])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(str(out_path), index=False)
    print(f"  保存: {out_path}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='all')
    p.add_argument('--method', default='all')
    args, _ = p.parse_known_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    methods = METHODS if args.method == 'all' else [args.method]

    total = len(datasets) * len(methods)
    done = 0
    for dataset in datasets:
        for method_name in methods:
            done += 1
            print(f"\n[{done}/{total}] {dataset} × {method_name}")
            try:
                run_one(dataset, method_name)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    # 汇总所有结果
    csvs = list(OUTPUT_DIR.glob('*_5fold.csv'))
    if csvs:
        all_df = pd.concat([pd.read_csv(str(f)) for f in csvs], ignore_index=True)
        summary_path = OUTPUT_DIR / 'summary_all.csv'
        all_df.to_csv(str(summary_path), index=False)
        print(f"\n汇总表已保存: {summary_path}  ({len(all_df)} 行)")


if __name__ == '__main__':
    main()
