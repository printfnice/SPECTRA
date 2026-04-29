# plot_gate_entropy.py
# 职责: Gate 权重 Shannon entropy 分析，量化多尺度利用程度

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(_PIPELINE_DIR))

from pub_style import (
    FIGURES_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    NATURE_CATEGORICAL,
    DATASET_LABELS,
    set_pub_style,
    save_fig,
    style_axis,
    add_panel_label,
    add_reference_line,
    finalize_figure,
)

DATASETS = ['kuppe', 'carraro', 'habermann', 'velmeshev', 'reichart']
MAX_ENTROPY = np.log2(3)


def compute_entropy(w_gene, w_pathway, w_celltype):
    eps = 1e-12
    weights = np.stack([w_gene, w_pathway, w_celltype], axis=1).clip(eps, 1)
    return -np.sum(weights * np.log2(weights), axis=1)


def load_gate_entropy(dataset, chunksize=500_000):
    path = OUTPUT_DIR / f'{dataset}_gate_weights.csv'
    if not path.exists():
        print(f'  [WARN] 未找到 {path}')
        return None
    rows_all = []
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=['w_gene', 'w_pathway', 'w_celltype']):
        entropy = compute_entropy(chunk['w_gene'].values, chunk['w_pathway'].values, chunk['w_celltype'].values)
        rows_all.append(entropy)
    return np.concatenate(rows_all)


def summarize_entropy(entropy, dataset):
    path = OUTPUT_DIR / f'{dataset}_gate_weights.csv'
    dom_counts = {'w_gene': 0, 'w_pathway': 0, 'w_celltype': 0}
    for chunk in pd.read_csv(path, chunksize=500_000, usecols=['w_gene', 'w_pathway', 'w_celltype']):
        dominant = chunk[['w_gene', 'w_pathway', 'w_celltype']].idxmax(axis=1)
        for key in dom_counts:
            dom_counts[key] += int((dominant == key).sum())
    total = sum(dom_counts.values())
    return {
        'dataset': dataset,
        'mean_entropy': float(np.mean(entropy)),
        'std_entropy': float(np.std(entropy)),
        'median_entropy': float(np.median(entropy)),
        'pct_low_entropy': float(np.mean(entropy < 0.5)),
        'norm_entropy': float(np.mean(entropy)) / MAX_ENTROPY,
        'pct_gene_dominant': dom_counts['w_gene'] / total,
        'pct_pathway_dominant': dom_counts['w_pathway'] / total,
        'pct_celltype_dominant': dom_counts['w_celltype'] / total,
    }


def plot_entropy_figure(stats_df, entropy_arrays):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    set_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.8), gridspec_kw={'width_ratios': [1.25, 1.0]})
    ax_violin, ax_bar = axes
    dataset_labels = [DATASET_LABELS[ds] for ds in DATASETS]
    colors = NATURE_CATEGORICAL[:len(DATASETS)]

    parts = ax_violin.violinplot(
        [entropy_arrays[ds] for ds in DATASETS],
        positions=np.arange(len(DATASETS)),
        widths=0.72,
        showmedians=True,
        showextrema=False,
    )
    for body, color in zip(parts['bodies'], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.70)
    parts['cmedians'].set_color('#1f1f1f')
    parts['cmedians'].set_linewidth(1.3)

    style_axis(ax_violin, grid_axis='y')
    add_panel_label(ax_violin, 'a')
    add_reference_line(ax_violin, y=MAX_ENTROPY, label=r'$\log_2 3$ theoretical maximum')
    ax_violin.set_xticks(range(len(DATASETS)))
    ax_violin.set_xticklabels(dataset_labels)
    ax_violin.set_ylabel('Gate entropy (bits)')
    # Entropy is bounded above by log2(3); clip the visual range to avoid
    # KDE tails being misread as values above the theoretical maximum.
    ax_violin.set_ylim(-0.03, MAX_ENTROPY)
    ax_violin.set_title('Distribution of retained sample-LR gate entropy', loc='left', pad=10)
    ax_violin.text(
        0.54,
        0.94,
        'Entropy summarizes cohort gate allocation at mapped LR pairs,\nnot pathway-specific predictive importance.',
        transform=ax_violin.transAxes,
        ha='left',
        va='top',
        fontsize=6.2,
        color='#3a3a3a',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#bdbdbd', linewidth=0.6, alpha=0.92),
    )

    x = np.arange(len(DATASETS))
    norm_entropy = stats_df['norm_entropy'].values
    bars = ax_bar.bar(x, norm_entropy, color=colors, edgecolor='white', linewidth=0.6, width=0.62, zorder=3)
    for bar, value in zip(bars, norm_entropy):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f'{value:.4f}',
                    ha='center', va='bottom', fontsize=7, color='#1f1f1f', fontweight='bold')

    style_axis(ax_bar, grid_axis='y')
    add_panel_label(ax_bar, 'b')
    add_reference_line(ax_bar, y=1.0, label='maximum normalized entropy')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(dataset_labels)
    ax_bar.set_ylabel('Normalized mean entropy')
    ax_bar.set_ylim(0, 1.10)
    ax_bar.set_title('Dataset-level summary of entropy magnitude', loc='left', pad=10)

    finalize_figure(fig, left=0.08, right=0.98, top=0.90, bottom=0.20, wspace=0.20)
    save_fig(fig, FIGURES_DIR / 'Fig6_gate_entropy')
    plt.close(fig)


def main():
    print('\n' + '=' * 55)
    print('  Gate Entropy 分析（多尺度利用程度量化）')
    print('=' * 55)

    entropy_arrays = {}
    stats_rows = []
    for ds in DATASETS:
        print(f'\n  {ds}:')
        entropy = load_gate_entropy(ds)
        if entropy is None:
            continue
        entropy_arrays[ds] = entropy
        row = summarize_entropy(entropy, ds)
        stats_rows.append(row)
        print(
            f'    mean_H={row["mean_entropy"]:.4f}  '
            f'norm_H={row["norm_entropy"]:.4f}  '
            f'pct_low={row["pct_low_entropy"]:.2%}'
        )

    if not stats_rows:
        print('  [ERROR] 无数据')
        return

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(DATA_DIR / 'gate_entropy_stats.csv', index=False)
    plot_entropy_figure(stats_df, entropy_arrays)
    print('\n  完成。')


if __name__ == '__main__':
    main()
