# plot_evi_fidelity.py
# 职责: 可视化 EVI 忠实性结果（Top-k 屏蔽 vs Random-k 屏蔽）

from pathlib import Path
import sys

import numpy as np
import pandas as pd

_PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(_PIPELINE_DIR))

from pub_style import (
    FIGURES_DIR,
    DATA_DIR,
    DATASET_LABELS,
    ACCENT,
    set_pub_style,
    save_fig,
    style_axis,
    add_panel_label,
    add_reference_line,
    finalize_figure,
)


def main() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p = DATA_DIR / 'evi_fidelity_table.csv'
    if not p.exists():
        raise FileNotFoundError(f'Missing {p}, run run_evi_fidelity.py first')

    df = pd.read_csv(p).sort_values(['k', 'dataset'])
    ks = sorted(df['k'].unique())

    set_pub_style()
    fig, axes = plt.subplots(1, len(ks), figsize=(7.8, 4.1), sharey=True)
    if len(ks) == 1:
        axes = [axes]

    comparison_colors = {'Top-k mask': ACCENT, 'Random-k mask': '#9AA7B8'}

    for idx, (ax, k) in enumerate(zip(axes, ks)):
        sub = df[df['k'] == k].copy()
        x = np.arange(len(sub))
        width = 0.34
        top_bars = ax.bar(x - width / 2, sub['drop_topk'], width, color=comparison_colors['Top-k mask'],
                          edgecolor='white', linewidth=0.5, zorder=3, label='Top-k mask')
        rnd_bars = ax.bar(x + width / 2, sub['drop_randomk'], width, color=comparison_colors['Random-k mask'],
                          edgecolor='white', linewidth=0.5, zorder=3, label='Random-k mask')
        for xpos, top_drop, rnd_drop in zip(x, sub['drop_topk'], sub['drop_randomk']):
            delta = top_drop - rnd_drop
            ax.text(xpos, max(top_drop, rnd_drop) + 0.010, f'{delta:+.4f}', ha='center', va='bottom',
                    fontsize=6.8, color=ACCENT if delta >= 0 else '#7A2E2E', fontweight='bold')

        style_axis(ax, grid_axis='y')
        add_panel_label(ax, chr(ord('a') + idx))
        add_reference_line(ax, y=0.0)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS.get(ds, ds) for ds in sub['dataset']])
        ax.set_title(f'Masking fidelity at k = {k}', loc='left', pad=10)
        ax.set_ylabel('AUROC drop' if idx == 0 else '')
        ax.set_ylim(min(df[['drop_topk', 'drop_randomk']].min().min(), -0.02) - 0.01,
                    df[['drop_topk', 'drop_randomk']].max().max() + 0.06)

    axes[0].legend(loc='upper left', bbox_to_anchor=(0.00, -0.16), ncol=2, frameon=False)
    finalize_figure(fig, left=0.08, right=0.98, top=0.90, bottom=0.25, wspace=0.18)
    save_fig(fig, FIGURES_DIR / 'FigS_evi_fidelity')
    plt.close(fig)
    print(f'Saved: {FIGURES_DIR / "FigS_evi_fidelity.pdf"}')


if __name__ == '__main__':
    main()
