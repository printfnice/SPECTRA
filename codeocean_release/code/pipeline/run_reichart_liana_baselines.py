# run_reichart_liana_baselines.py
# 依赖: liana, scanpy, sklearn, cell2cell, utils/classify_utils.py
# 被依赖: run_after_main_suite.sh (Extra E)
# 职责: 8 LIANA 方法 × reichart，5-fold StratifiedGroupKFold（按 donor_id），
#       Tensor+RF 下游，与 liana_baselines_5fold 其他4数据集协议一致

import os
import sys
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, confusion_matrix

warnings.filterwarnings('ignore')

code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))
os.chdir(Path(__file__).parent)

DATASET = 'reichart'
METHODS = ['cellphonedb', 'connectome', 'cellchat', 'scseqcomm',
           'singlecellsignalr', 'natmi', 'logfc', 'rank_aggregate']

N_CV_FOLDS = 5
N_RANDOM_STATES = 5
N_ESTIMATORS = 500
N_FACTORS = 20

META = {
    'sample_key':    'Sample',
    'groupby':       'cell_type',
    'condition_key': 'disease',
    'donor_key':     'donor_id',
    'n_factors':     N_FACTORS,
}

OUTPUT_DIR = Path('output/liana_baselines_5fold')
DATA_DIR   = Path('../../data')


def _load_adata():
    for p in [DATA_DIR / 'interim' / 'reichart_filtered.h5ad',
              DATA_DIR / 'interim' / 'reichart_processed.h5ad']:
        if p.exists():
            print(f"  加载: {p}")
            return sc.read_h5ad(str(p))
    raise FileNotFoundError("找不到 reichart h5ad")


def _get_method_map():
    from liana.method import (cellphonedb, connectome, cellchat, scseqcomm,
                               singlecellsignalr, natmi, logfc, rank_aggregate)
    return {
        'cellphonedb': cellphonedb, 'connectome': connectome,
        'cellchat': cellchat, 'scseqcomm': scseqcomm,
        'singlecellsignalr': singlecellsignalr, 'natmi': natmi,
        'logfc': logfc, 'rank_aggregate': rank_aggregate,
    }


def _run_liana_method(adata, method_name, method_obj):
    sk = method_obj.magnitude if method_obj.magnitude else method_obj.specificity
    if sk in adata.uns:
        print(f"    [{method_name}] 已缓存，跳过推断")
        return sk
    print(f"    [{method_name}] 运行推断...")
    result = method_obj.by_sample(
        adata, groupby=META['groupby'], sample_key=META['sample_key'],
        use_raw=False, verbose=False, n_perms=None, inplace=False,
    )
    adata.uns[sk] = result
    print(f"    [{method_name}] 完成: {len(result)} 行")
    return sk


def _build_xlr_y(adata, score_key):
    df = adata.uns[score_key]
    sk, ck, dk = META['sample_key'], META['condition_key'], META['donor_key']
    score_col = score_key if score_key in df.columns else df.select_dtypes('number').columns[0]
    df['lr_pair'] = (df['source'].astype(str) + '_' +
                     df['ligand_complex'].astype(str) + '__' +
                     df['target'].astype(str) + '_' +
                     df['receptor_complex'].astype(str))
    pivot = df.pivot_table(index=sk, columns='lr_pair', values=score_col,
                           aggfunc='mean', fill_value=0)
    X = pivot.values
    sample_names = pivot.index.tolist()
    obs_map = adata.obs[[sk, ck, dk]].drop_duplicates().set_index(sk)
    cond_map  = obs_map[ck].to_dict()
    donor_map = obs_map[dk].to_dict()
    y_raw    = [cond_map.get(s, 'unknown') for s in sample_names]
    groups   = [donor_map.get(s, s) for s in sample_names]
    y = LabelEncoder().fit_transform(y_raw)
    return X, y, np.array(groups)


def _run_tensor_reduction(adata, score_key):
    from utils.classify_utils import run_tensor_c2c
    run_tensor_c2c(
        adata, score_key=score_key,
        sample_key=META['sample_key'], condition_key=META['condition_key'],
        dataset_name=DATASET, n_factors=N_FACTORS, use_gpu=True,
    )
    csv_path = DATA_DIR / 'results' / 'tensor' / DATASET / f'{score_key}.csv'
    if not csv_path.exists():
        print(f"    警告: tensor CSV 未找到，回退到原始 X_lr")
        return None, None, None
    factor_df = pd.read_csv(str(csv_path), index_col=0)
    if 'Category' not in factor_df.columns:
        return None, None, None
    X_r = factor_df.drop(columns=['Category']).values
    y_r = LabelEncoder().fit_transform(factor_df['Category'].values)
    # groups: match sample order from tensor output
    obs_map = adata.obs[[META['sample_key'], META['donor_key']]].drop_duplicates().set_index(META['sample_key'])
    groups = np.array([obs_map.loc[s, META['donor_key']] if s in obs_map.index else s
                       for s in factor_df.index.tolist()])
    return X_r, y_r, groups


def _fold_metrics(y_true, y_pred, y_prob):
    res = {}
    try:
        res['auroc'] = roc_auc_score(y_true, y_prob[:, 1])
        res['auprc'] = average_precision_score(y_true, y_prob[:, 1])
    except Exception:
        res['auroc'] = res['auprc'] = np.nan
    res['f1_macro']    = f1_score(y_true, y_pred, average='macro', zero_division=0)
    res['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    res['bacc'] = balanced_accuracy_score(y_true, y_pred)
    res['mcc']  = matthews_corrcoef(y_true, y_pred)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        res['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        res['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    except Exception:
        res['sensitivity'] = res['specificity'] = np.nan
    return res


def _run_cv(X, y, groups):
    all_metrics = []
    for rs in range(N_RANDOM_STATES):
        sgkf = StratifiedGroupKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=rs * 7)
        for fold_idx, (tr, te) in enumerate(sgkf.split(X, y, groups)):
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue
            clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_features='sqrt',
                                         random_state=rs, n_jobs=-1, class_weight='balanced')
            clf.fit(X_tr, y_tr)
            fm = _fold_metrics(y_te, clf.predict(X_te), clf.predict_proba(X_te))
            fm['rs'] = rs; fm['fold'] = fold_idx
            all_metrics.append(fm)
    return all_metrics


def _agg(all_metrics):
    if not all_metrics:
        return {}
    df = pd.DataFrame(all_metrics)
    res = {}
    for col in ['auroc', 'auprc', 'f1_macro', 'f1_weighted', 'bacc', 'mcc',
                'sensitivity', 'specificity']:
        vals = df[col].dropna().values
        res[f'mean_{col}'] = round(float(np.mean(vals)), 4) if len(vals) else np.nan
        res[f'std_{col}']  = round(float(np.std(vals)), 4)  if len(vals) else np.nan
    res['fold_aurocs']   = str([round(float(v), 4) for v in df['auroc'].dropna().tolist()])
    res['n_folds_total'] = len(all_metrics)
    return res


def run_one(method_name, adata):
    out_path = OUTPUT_DIR / f'{DATASET}_{method_name}_5fold.csv'
    if out_path.exists():
        auroc = pd.read_csv(str(out_path))['mean_auroc'].iloc[0]
        print(f"  [skip] {method_name}  AUROC={auroc:.4f}")
        return

    print(f"\n{'='*60}")
    print(f"  {DATASET} × {method_name}  [{datetime.now().strftime('%H:%M:%S')}]")
    print(f"{'='*60}")

    method_map = _get_method_map()
    score_key = _run_liana_method(adata, method_name, method_map[method_name])
    X_lr, y_lr, groups_lr = _build_xlr_y(adata, score_key)
    print(f"  X_lr: {X_lr.shape}")

    X_r, y_r, groups_r = _run_tensor_reduction(adata, score_key)
    if X_r is not None:
        X_feat, y_feat, groups = X_r, y_r, groups_r
        feat_type = 'tensor'
        print(f"  X_tensor: {X_r.shape}")
    else:
        X_feat, y_feat, groups = X_lr, y_lr, groups_lr
        feat_type = 'raw_lr'

    all_metrics = _run_cv(X_feat, y_feat, groups)
    agg = _agg(all_metrics)
    print(f"  AUROC = {agg.get('mean_auroc', float('nan')):.4f} ± {agg.get('std_auroc', float('nan')):.4f}")

    row = {
        'dataset': DATASET, 'method': method_name, 'score_key': score_key,
        'feat_type': feat_type, 'n_folds': N_CV_FOLDS,
        'n_random_states': N_RANDOM_STATES,
        'n_samples': len(y_feat), 'n_features': X_feat.shape[1],
    }
    row.update(agg)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(str(out_path), index=False)
    print(f"  保存: {out_path}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--method', default='all')
    args, _ = p.parse_known_args()

    methods = METHODS if args.method == 'all' else [args.method]
    print(f"\n{'='*60}")
    print(f"  Reichart LIANA Baselines (GroupKFold by donor_id)")
    print(f"  {len(methods)} 方法，设备: GPU")
    print(f"{'='*60}")

    # 一次加载，所有方法共用（避免重复加载 20GB）
    adata = _load_adata()

    for i, m in enumerate(methods):
        print(f"\n[{i+1}/{len(methods)}] {m}")
        try:
            run_one(m, adata)
        except Exception as e:
            print(f"  ERROR [{m}]: {e}")
            import traceback; traceback.print_exc()

    # 更新 summary
    csvs = list(OUTPUT_DIR.glob('*_5fold.csv'))
    if csvs:
        all_df = pd.concat([pd.read_csv(str(f)) for f in csvs], ignore_index=True)
        all_df.to_csv(str(OUTPUT_DIR / 'summary_all.csv'), index=False)
        print(f"\n汇总已更新: {OUTPUT_DIR}/summary_all.csv  ({len(all_df)} 行)")


if __name__ == '__main__':
    main()
