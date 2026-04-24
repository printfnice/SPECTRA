# run_statistical_tests.py
# 依赖: scipy, pandas, matplotlib, pub_style.py
# 职责: SPECTRA vs baselines 的数据集级配对统计检验（n=5 cohorts）

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

_PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(_PIPELINE_DIR))

from pub_style import set_pub_style, save_fig, FIGURES_DIR, DATA_DIR, NATURE_CATEGORICAL

DATASETS = ['kuppe', 'carraro', 'habermann', 'velmeshev', 'reichart']
DATASET_LABELS = ['Kuppe (MI)', 'Carraro (CF)', 'Habermann (PF)', 'Velmeshev (ASD)', 'Reichart (DCM)']

# SPECTRA mean AUROC (dataset-level)
OUR_MEAN = {
    'kuppe': 1.0000,
    'carraro': 0.9429,
    'habermann': 0.9286,
    'velmeshev': 0.8654,
    'reichart': 0.9883,
}

BASELINES = {
    'natmi':          {'kuppe': 0.9905, 'carraro': 0.8857, 'habermann': 0.9071, 'velmeshev': 0.8139, 'reichart': 0.9052},
    'cellchat':       {'kuppe': 0.9667, 'carraro': 0.7500, 'habermann': 0.8643, 'velmeshev': 0.6836, 'reichart': 0.9811},
    'cellphonedb':    {'kuppe': 0.8595, 'carraro': 0.7000, 'habermann': 0.7857, 'velmeshev': 0.7143, 'reichart': 0.8681},
    'geometric_mean': {'kuppe': 0.9762, 'carraro': 0.7286, 'habermann': 0.8000, 'velmeshev': 0.5242, 'reichart': 0.9556},
    'connectome':     {'kuppe': 1.0000, 'carraro': 0.8000, 'habermann': 0.5643, 'velmeshev': 0.6798, 'reichart': 0.8532},
    'scseqcomm':      {'kuppe': 0.4762, 'carraro': 0.8714, 'habermann': 0.6857, 'velmeshev': 0.6591, 'reichart': 0.7168},
    'logfc':          {'kuppe': 0.9286, 'carraro': 0.8429, 'habermann': 0.7357, 'velmeshev': 0.7333, 'reichart': 0.6526},
}

METHOD_LABELS = {
    'natmi': 'NATMI',
    'cellchat': 'CellChat',
    'cellphonedb': 'CellPhoneDB',
    'geometric_mean': 'Geometric Mean',
    'connectome': 'Connectome',
    'scseqcomm': 'scSeqComm',
    'logfc': 'LogFC',
}


def rank_biserial_from_diffs(diffs: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation for paired differences."""
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method='average')
    pos = ranks[nonzero > 0].sum()
    neg = ranks[nonzero < 0].sum()
    den = pos + neg
    return 0.0 if den == 0 else (pos - neg) / den


def test_vs_baseline(method_name: str):
    bl = BASELINES[method_name]
    our = np.array([OUR_MEAN[d] for d in DATASETS], dtype=float)
    base = np.array([bl[d] for d in DATASETS], dtype=float)
    diffs = our - base

    # one-sided (SPECTRA > baseline), exact for small n where available
    try:
        w = stats.wilcoxon(diffs, alternative='greater', zero_method='pratt', mode='exact')
        p = float(w.pvalue)
    except Exception:
        w = stats.wilcoxon(diffs, alternative='greater', zero_method='pratt', mode='approx')
        p = float(w.pvalue)

    r_rb = float(rank_biserial_from_diffs(diffs))

    row = {
        'method': method_name,
        'n_pairs': len(diffs),
        'p_value': p,
        'effect_r': r_rb,
        'mean_delta': float(np.mean(diffs)),
        'median_delta': float(np.median(diffs)),
    }
    for d, v in zip(DATASETS, diffs):
        row[f'delta_{d}'] = float(v)
    return row


def run_all_tests():
    rows = [test_vs_baseline(m) for m in BASELINES]
    df = pd.DataFrame(rows).sort_values('p_value').reset_index(drop=True)
    df['label'] = df['p_value'].apply(lambda x: '***' if x < 0.001 else ('**' if x < 0.01 else ('*' if x < 0.05 else 'ns')))
    return df


def plot_stat_figure(stat_df):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    set_pub_style()

    methods = stat_df['method'].tolist()
    x = np.arange(len(methods))
    colors = NATURE_CATEGORICAL[:len(DATASETS)]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), gridspec_kw={'width_ratios': [1.55, 1.0]})

    ax = axes[0]
    bar_w = 0.11
    offsets = np.linspace(-2 * bar_w, 2 * bar_w, len(DATASETS))
    for i, ds in enumerate(DATASETS):
        vals = [stat_df.loc[stat_df['method'] == m, f'delta_{ds}'].iloc[0] for m in methods]
        ax.bar(x + offsets[i], vals, width=bar_w, color=colors[i], alpha=0.85, label=DATASET_LABELS[i], edgecolor='none')

    for i, row in stat_df.iterrows():
        ymax = max(row[f'delta_{d}'] for d in DATASETS)
        if row['label'] != 'ns':
            ax.text(i, ymax + 0.007, row['label'], ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.axhline(0, color='#8A8A8A', linestyle='--', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=22, ha='right')
    ax.set_ylabel('ΔAUROC (SPECTRA − baseline)')
    ax.set_title('(A) Cohort-level paired deltas (n=5 cohorts)', loc='left', fontsize=10)
    ax.legend(frameon=False, ncol=2, fontsize=7)

    ax2 = axes[1]
    y = -np.log10(stat_df['p_value'].to_numpy() + 1e-12)
    x2 = stat_df['effect_r'].to_numpy()
    sig_colors = ['#D9534F' if p < 0.05 else '#9E9E9E' for p in stat_df['p_value']]
    ax2.scatter(x2, y, c=sig_colors, s=90, edgecolors='white', linewidths=0.5, zorder=3)
    for _, row in stat_df.iterrows():
        ax2.annotate(METHOD_LABELS[row['method']], (row['effect_r'], -np.log10(row['p_value'] + 1e-12)),
                     xytext=(5, 2), textcoords='offset points', fontsize=7)

    ax2.axhline(-np.log10(0.05), color='#D9534F', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Rank-biserial r')
    ax2.set_ylabel('−log10(p)')
    ax2.set_title('(B) Effect size and significance', loc='left', fontsize=10)

    plt.tight_layout()
    save_fig(fig, FIGURES_DIR / 'ExtFig_stat_test')
    plt.close(fig)


def main():
    print('\n' + '=' * 55)
    print('  统计显著性检验（cohort-level Wilcoxon, n=5）')
    print('=' * 55)

    stat_df = run_all_tests()
    out = DATA_DIR / 'statistical_tests.csv'
    stat_df.to_csv(out, index=False)

    cols = ['method', 'n_pairs', 'p_value', 'label', 'effect_r', 'mean_delta', 'median_delta']
    print('\n' + stat_df[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'\n  已保存: {out}')

    plot_stat_figure(stat_df)
    print('  完成。')


if __name__ == '__main__':
    main()
