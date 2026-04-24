# plot_roc_curves.py
# 依赖: matplotlib, numpy, json, pub_style.py
# 被依赖: 直接运行
# 职责: 读取 output/data/*_metrics.json，绘制 5 子图 ROC 曲线（Fig2_roc_curves.pdf）
#       每子图一个数据集，多条曲线代表不同方法，legend 含 AUC 值

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

_PIPELINE_DIR = Path(__file__).parent
_BASE_DIR = _PIPELINE_DIR.parent.parent
sys.path.insert(0, str(_PIPELINE_DIR))

from pub_style import set_pub_style, NATURE_CATEGORICAL, FIGURES_DIR, DATA_DIR

# ============================================================================
# 配置
# ============================================================================

DATASETS = ['kuppe', 'carraro', 'habermann', 'velmeshev', 'reichart']
DATASET_LABELS = {
    'kuppe':     'Kuppe (MI, N=23)',
    'carraro':   'Carraro (CF, N=16)',
    'habermann': 'Habermann (PF, N=18)',
    'velmeshev': 'Velmeshev (ASD, N=38)',
    'reichart':  'Reichart (DCM, N=171)',
}

# 要展示的 LIANA 方法（从 liana_baselines_5fold JSON 加载）
LIANA_METHODS = ['cellphonedb', 'cellchat', 'natmi', 'geometric_mean']
LIANA_LABELS = {
    'cellphonedb':   'CellPhoneDB',
    'cellchat':      'CellChat',
    'natmi':         'NATMI',
    'geometric_mean': 'Geo. Mean',
}

# 颜色：LIANA 方法用灰色调，SPECTRA 用红色（突出）
LIANA_COLORS = ['#9ECAE1', '#74C476', '#FDD0A2', '#DADAEB']
SPECTRA_COLOR = '#E05252'


# ============================================================================
# 数据加载
# ============================================================================

def _load_json(path):
    with open(str(path)) as f:
        return json.load(f)


def _average_roc(fold_metrics):
    """将多个 (fold, rs) 的 ROC 曲线插值到统一 FPR 网格后取均值。"""
    grid = np.linspace(0, 1, 100)
    tpr_list = []
    for fm in fold_metrics:
        if 'fpr' not in fm or 'tpr' not in fm:
            continue
        tpr_list.append(np.interp(grid, fm['fpr'], fm['tpr']))
    if not tpr_list:
        return None, None, None
    tpr_mean = np.mean(tpr_list, axis=0)
    tpr_std  = np.std(tpr_list, axis=0)
    return grid, tpr_mean, tpr_std


def load_spectra_roc(dataset):
    """加载 SPECTRA E2E 的 ROC 数据，返回 (fpr_grid, tpr_mean, tpr_std, mean_auroc)。"""
    path = DATA_DIR / f'{dataset}_spectra_e2e_metrics.json'
    if not path.exists():
        print(f"  [跳过] SPECTRA JSON 不存在: {path}")
        return None
    data = _load_json(path)
    fpr, tpr_mean, tpr_std = _average_roc(data['fold_metrics'])
    if fpr is None:
        return None
    mean_auroc = data['summary']['mean_auroc']
    return fpr, tpr_mean, tpr_std, mean_auroc


def load_liana_roc(dataset, method):
    """加载 LIANA 基线方法的 ROC 数据。"""
    path = DATA_DIR / f'{dataset}_{method}_liana_metrics.json'
    if not path.exists():
        return None
    data = _load_json(path)
    fpr, tpr_mean, tpr_std = _average_roc(data['fold_metrics'])
    if fpr is None:
        return None
    mean_auroc = data['summary'].get('mean_auroc', float('nan'))
    return fpr, tpr_mean, tpr_std, mean_auroc


# ============================================================================
# 绘图
# ============================================================================

def plot_roc_curves(save_path=None):
    set_pub_style()
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8),
                             gridspec_kw={'wspace': 0.35})

    for ax, dataset in zip(axes, DATASETS):
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5, zorder=0)  # 随机基线

        # ---- LIANA 基线曲线（细，灰色调，后面）
        for method, color in zip(LIANA_METHODS, LIANA_COLORS):
            result = load_liana_roc(dataset, method)
            if result is None:
                continue
            fpr, tpr_mean, tpr_std, auc_val = result
            label = f'{LIANA_LABELS[method]} (AUC={auc_val:.3f})'
            ax.plot(fpr, tpr_mean, color=color, lw=1.2, label=label, zorder=2)
            ax.fill_between(fpr, tpr_mean - tpr_std, tpr_mean + tpr_std,
                            color=color, alpha=0.15, zorder=1)

        # ---- SPECTRA E2E 曲线（粗，红色，前面）
        result = load_spectra_roc(dataset)
        if result is not None:
            fpr, tpr_mean, tpr_std, auc_val = result
            label = f'SPECTRA E2E (AUC={auc_val:.3f})'
            ax.plot(fpr, tpr_mean, color=SPECTRA_COLOR, lw=2.0,
                    label=label, zorder=4)
            ax.fill_between(fpr, tpr_mean - tpr_std, tpr_mean + tpr_std,
                            color=SPECTRA_COLOR, alpha=0.15, zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('False Positive Rate', fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel('True Positive Rate', fontsize=8)
        ax.set_title(DATASET_LABELS[dataset], fontsize=8, pad=4)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.tick_params(labelsize=7)

        # 图例（每个子图单独，右下角）
        leg = ax.legend(fontsize=5.5, loc='lower right', frameon=True,
                        framealpha=0.85, edgecolor='#CCCCCC',
                        handlelength=1.5, handletextpad=0.4, borderpad=0.5)

    if save_path is None:
        save_path = FIGURES_DIR / 'Fig2_roc_curves.pdf'
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), bbox_inches='tight', dpi=300)
    png_path = str(save_path).replace('.pdf', '.png')
    fig.savefig(png_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  已保存: {save_path}")
    print(f"  已保存: {png_path}")
    return save_path


# ============================================================================
# 诊断：检查哪些 JSON 已存在
# ============================================================================

def check_data_availability():
    """列出哪些数据集的 metrics JSON 已准备好，哪些还需要跑。"""
    print("\n=== ROC 数据准备情况 ===")
    for ds in DATASETS:
        spec_path = DATA_DIR / f'{ds}_spectra_e2e_metrics.json'
        spec_ok = spec_path.exists()
        liana_ok = {}
        for m in LIANA_METHODS:
            liana_ok[m] = (DATA_DIR / f'{ds}_{m}_liana_metrics.json').exists()
        print(f"\n  {ds}:")
        print(f"    SPECTRA E2E:  {'OK' if spec_ok else '缺失 — 需运行 run_unified_e2e.py --collect_metrics'}")
        for m, ok in liana_ok.items():
            print(f"    {LIANA_LABELS[m]:15s}: {'OK' if ok else '缺失 — 需运行 run_liana_baselines_5fold.py'}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true', help='只检查数据准备情况，不绘图')
    args = p.parse_args()

    if args.check:
        check_data_availability()
    else:
        check_data_availability()
        print("\n=== 开始绘制 ROC 曲线 ===")
        plot_roc_curves()
