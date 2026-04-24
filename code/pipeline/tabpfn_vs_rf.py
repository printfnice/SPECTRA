# tabpfn_vs_rf.py
# 依赖: tabpfn, sklearn, simplified_pipeline_optimized.py (步骤1-3数据)
# 被依赖: 直接运行
# 职责: 对比 TabPFN vs RF 在无监督流水线步骤4上的分类性能

import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

_PIPELINE_DIR = Path(__file__).parent
_BASE_DIR = _PIPELINE_DIR.parent.parent
_OUTPUT_DIR = _BASE_DIR / 'output'

# 步骤3输出路径（Tensor 降维后的因子矩阵）
_CHECKPOINT_DIR = _PIPELINE_DIR / 'data'

_DATASETS = ['carraro', 'habermann', 'kuppe', 'velmeshev', 'reichart']


def _load_lr_matrix(dataset):
    """加载 LR 分数矩阵 + PCA 降维到 n_components 维。"""
    import pickle
    from sklearn.decomposition import PCA
    ckpt_dir = _BASE_DIR / 'data' / 'results' / 'checkpoints'
    p = ckpt_dir / f'{dataset}_ms_hgatlink_tensor_step3.pkl'
    if not p.exists():
        return None, None
    with open(p, 'rb') as f:
        data = pickle.load(f)
    X_raw = np.array(data['X_lr']) if hasattr(data['X_lr'], 'toarray') \
        else (data['X_lr'].toarray() if hasattr(data['X_lr'], 'toarray')
              else np.array(data['X_lr']))
    y = np.array(data['y'])
    n_comp = min(min(X_raw.shape) - 1, 15)
    X = PCA(n_components=n_comp, random_state=42).fit_transform(X_raw)
    return X, y


def _run_cv(X, y, clf_factory, n_folds=3, n_states=5):
    """分层 K 折 CV，多种子取均值。"""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    all_aurocs = []
    for state in range(n_states):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=42 + state * 7)
        for train_idx, test_idx in skf.split(X, y_enc):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y_enc[train_idx], y_enc[test_idx]
            if len(np.unique(y_te)) < 2:
                continue
            clf = clf_factory()
            clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_te)
            if n_classes == 2:
                auroc = roc_auc_score(y_te, proba[:, 1])
            else:
                auroc = roc_auc_score(y_te, proba,
                                      multi_class='ovr', average='macro')
            all_aurocs.append(auroc)
    return np.mean(all_aurocs) if all_aurocs else 0.0


def _rf_factory():
    return RandomForestClassifier(n_estimators=500, random_state=1337,
                                  oob_score=True)


def _tabpfn_factory():
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier()


def main():
    p = argparse.ArgumentParser(description='TabPFN vs RF 对比实验')
    p.add_argument('--datasets', nargs='+', default=_DATASETS)
    p.add_argument('--n_folds', type=int, default=3)
    p.add_argument('--n_states', type=int, default=5)
    args = p.parse_args()

    results = []
    for ds in args.datasets:
        print(f"\n=== {ds} ===")
        X, y = _load_lr_matrix(ds)
        if X is None:
            print(f"  跳过: 未找到 {ds} 的 Tensor 因子数据")
            continue
        print(f"  样本={len(y)}, 特征={X.shape[1]}, 类别={np.unique(y)}")

        rf_auroc = _run_cv(X, y, _rf_factory, args.n_folds, args.n_states)
        print(f"  RF       AUROC = {rf_auroc:.4f}")

        try:
            tab_auroc = _run_cv(X, y, _tabpfn_factory, args.n_folds,
                                args.n_states)
            print(f"  TabPFN   AUROC = {tab_auroc:.4f}")
        except Exception as e:
            print(f"  TabPFN   失败: {e}")
            tab_auroc = np.nan

        diff = tab_auroc - rf_auroc if not np.isnan(tab_auroc) else np.nan
        print(f"  差异     = {diff:+.4f}" if not np.isnan(diff) else "")
        results.append({
            'dataset': ds, 'rf_auroc': rf_auroc,
            'tabpfn_auroc': tab_auroc, 'diff': diff
        })

    if results:
        df = pd.DataFrame(results)
        out = _OUTPUT_DIR / 'tabpfn_vs_rf_comparison.csv'
        _OUTPUT_DIR.mkdir(exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\n结果已保存: {out}")
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
