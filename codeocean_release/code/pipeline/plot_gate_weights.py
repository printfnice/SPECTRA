# plot_gate_weights.py
# 职责: 生成 Fig.4 Gate 三尺度权重图（Nature/Cell 风格）

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
    SCALE_COLORS,
    SCALE_LABELS,
    DATASET_LABELS,
    set_pub_style,
    save_fig,
    style_axis,
    add_panel_label,
    finalize_figure,
    SOFT_BG,
    MUTED,
)

DATASETS = ['kuppe', 'carraro', 'habermann', 'velmeshev', 'reichart']
BIO_ANNOTATIONS = {
    'kuppe': 'Gene-scale dominant gate with measurable cell-type contribution.',
    'carraro': 'Pathway-scale gate allocation in the cystic-fibrosis cohort.',
    'habermann': 'Pathway-scale features dominate the learned gate in this fibrosis cohort.',
    'velmeshev': 'Near-exclusive pathway-scale gate allocation in this cohort.',
    'reichart': 'Gene-scale gate dominant with a secondary pathway component.',
}


def load_gate_summary():
    rows = []
    for ds in DATASETS:
        p = OUTPUT_DIR / f'{ds}_gate_weights.csv'
        if not p.exists():
            continue
        df = pd.read_csv(p)
        rows.append({
            'dataset': ds,
            'w_gene': df['w_gene'].mean(),
            'w_pathway': df['w_pathway'].mean(),
            'w_celltype': df['w_celltype'].mean(),
            'n': len(df),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(DATA_DIR / 'gate_weights_summary.csv', index=False)
    return summary


def plot_gate_fig4(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    set_pub_style()
    fig = plt.figure(figsize=(8.1, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.40, 0.92], wspace=0.10)
    ax = fig.add_subplot(gs[0, 0])
    ax_note = fig.add_subplot(gs[0, 1])

    x = np.arange(len(summary))
    bottoms = np.zeros(len(summary))
    scales = ['w_gene', 'w_pathway', 'w_celltype']
    handles = []

    for scale in scales:
        vals = summary[scale].values
        ax.bar(
            x,
            vals,
            0.62,
            bottom=bottoms,
            color=SCALE_COLORS[scale],
            edgecolor='white',
            linewidth=0.6,
            zorder=3,
        )
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 0.035:
                text_color = 'white' if v >= 0.11 else '#1f1f1f'
                ax.text(
                    xi,
                    b + v / 2,
                    f'{v:.2%}',
                    ha='center',
                    va='center',
                    fontsize=7,
                    color=text_color,
                    fontweight='bold',
                )
        handles.append(mpatches.Patch(facecolor=SCALE_COLORS[scale], label=SCALE_LABELS[scale]))
        bottoms += vals

    style_axis(ax, grid_axis='y')
    add_panel_label(ax, 'a')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Mean gate weight')
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(ds, ds) for ds in summary['dataset']])
    ax.set_title('Scale preference learned by the gate network', loc='left', pad=10)
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.50, -0.14), ncol=3, frameon=False)

    ax_note.set_facecolor(SOFT_BG)
    ax_note.set_xticks([])
    ax_note.set_yticks([])
    for spine in ax_note.spines.values():
        spine.set_visible(False)
    add_panel_label(ax_note, 'b', x=0.00, y=1.02)
    ax_note.text(0.00, 0.96, 'Biological interpretation', transform=ax_note.transAxes,
                 ha='left', va='top', fontsize=8.8, fontweight='semibold', color='#1f1f1f')
    y_positions = np.linspace(0.90, 0.16, len(summary))
    for y, (_, row) in zip(y_positions, summary.iterrows()):
        dominant = max(['w_gene', 'w_pathway', 'w_celltype'], key=lambda key: row[key])
        block = (
            f"{row['dataset'].capitalize()}\n"
            f"{BIO_ANNOTATIONS[row['dataset']]}\n"
            f"Dominant scale: {SCALE_LABELS[dominant]} ({row[dominant]:.4f})"
        )
        ax_note.text(
            0.04,
            y,
            block,
            transform=ax_note.transAxes,
            va='top',
            ha='left',
            fontsize=6.95,
            color=MUTED,
            linespacing=1.18,
        )

    finalize_figure(fig, left=0.07, right=0.985, top=0.93, bottom=0.19, wspace=0.10)
    save_fig(fig, FIGURES_DIR / 'Fig4_gate_weights')
    plt.close(fig)
    print('  Fig4 完成。')


def main():
    print('\n' + '=' * 55)
    print('  Fig.4 Gate 权重可视化')
    print('=' * 55)
    summary = load_gate_summary()
    if summary.empty:
        print('  [ERROR] 无数据')
        return
    plot_gate_fig4(summary)
    print('\n  完成。')


if __name__ == '__main__':
    main()
