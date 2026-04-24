# plot_supcon_ablation.py
# 职责: 生成 Fig.3 渐进式正则化消融图（Nature/Cell 风格）

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(_PIPELINE_DIR))

from pub_style import ACCENT, FIGURES_DIR, DATA_DIR, set_pub_style, save_fig, style_axis, add_panel_label, finalize_figure

ABLATION_CONFIGS = [
    {'tag': 'baseline_R38', 'label': 'VIB + MC', 'auroc': 0.9264, 'std': 0.0494, 'folds': [0.8468, 0.9033, 0.9583, 0.9318, 0.9917]},
    {'tag': 'supcon_w01', 'label': '+ SupCon 0.1', 'auroc': 0.9463, 'std': 0.0342, 'folds': [0.8957, 0.9304, 0.9712, 0.9394, 0.9947]},
    {'tag': 'supcon_w02', 'label': '+ SupCon 0.2', 'auroc': 0.9566, 'std': 0.0325, 'folds': [0.8977, 0.9612, 0.9712, 0.9568, 0.9962]},
    {'tag': 'supcon_w02_mixoff', 'label': '+ 0.2 / Mixup off', 'auroc': 0.9627, 'std': 0.0198, 'folds': [0.9445, 0.9487, 0.9758, 0.9492, 0.9955]},
    {'tag': 'supcon_w03_mixoff', 'label': '+ 0.3 / Mixup off', 'auroc': 0.9641, 'std': 0.0204, 'folds': [0.9431, 0.9531, 0.9758, 0.9500, 0.9985]},
    {'tag': 'sgdr', 'label': '+ SGDR final', 'auroc': 0.9883, 'std': 0.0121, 'folds': [0.9978, 0.9890, 0.9886, 0.9659, 1.0000]},
]
CELLCHAT_BASELINE = 0.9655
COLORS = ['#B9B9B9', '#A7C5DD', '#7EA9C9', '#5984B0', '#406B97', ACCENT]


def build_table():
    rows = []
    baseline = ABLATION_CONFIGS[0]['auroc']
    for cfg in ABLATION_CONFIGS:
        rows.append({
            'Tag': cfg['tag'],
            'AUROC': cfg['auroc'],
            '±std': cfg['std'],
            'Δ vs baseline': round(cfg['auroc'] - baseline, 4),
            'Δ vs CellChat': round(cfg['auroc'] - CELLCHAT_BASELINE, 4),
        })
    return pd.DataFrame(rows)


def plot_ablation_figure(table_df):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    set_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.8), gridspec_kw={'width_ratios': [1.15, 0.95]})
    x = np.arange(len(ABLATION_CONFIGS))
    labels = [cfg['label'] for cfg in ABLATION_CONFIGS]
    aurocs = [cfg['auroc'] for cfg in ABLATION_CONFIGS]
    stds = [cfg['std'] for cfg in ABLATION_CONFIGS]

    ax = axes[0]
    bars = ax.bar(x, aurocs, color=COLORS, edgecolor='white', linewidth=0.5, yerr=stds, error_kw=dict(elinewidth=0.8, ecolor='#555555'), zorder=3)
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.004, f'{val:.4f}', ha='center', va='bottom', fontsize=7, color=ACCENT if val == max(aurocs) else '#333333', fontweight='bold')
    style_axis(ax, grid_axis='y')
    add_panel_label(ax, 'a')
    ax.axhline(CELLCHAT_BASELINE, color='#909090', linestyle='--', linewidth=0.9, zorder=1)
    ax.set_ylim(0.82, 1.01)
    ax.set_ylabel('AUROC')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha='right')
    ax.set_title('Progressive gains from protocolized regularization', loc='left', pad=10)

    ax2 = axes[1]
    fold_matrix = np.array([cfg['folds'] for cfg in ABLATION_CONFIGS], dtype=float)
    fold_colors = ['#C9D3E0', '#AFBED2', '#94ABC5', '#7596B5', '#5278A0']
    for fold_idx in range(fold_matrix.shape[1]):
        vals = fold_matrix[:, fold_idx]
        ax2.plot(
            x, vals,
            color=fold_colors[fold_idx],
            linewidth=1.3,
            marker='o',
            markersize=4.2,
            markerfacecolor='white',
            markeredgewidth=0.8,
            zorder=2,
            alpha=0.95,
        )
    ax2.plot(
        x, aurocs,
        color=ACCENT,
        linewidth=2.2,
        marker='D',
        markersize=4.4,
        markerfacecolor=ACCENT,
        markeredgecolor='white',
        markeredgewidth=0.5,
        zorder=4,
    )
    style_axis(ax2, grid_axis='y')
    add_panel_label(ax2, 'b')
    ax2.axhline(CELLCHAT_BASELINE, color='#909090', linestyle='--', linewidth=0.9, zorder=1)
    ax2.set_ylim(0.82, 1.01)
    ax2.set_ylabel('Fold AUROC')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=22, ha='right')
    ax2.set_title('Fold trajectories contract under the final protocol', loc='left', pad=10)

    finalize_figure(fig, left=0.08, right=0.98, top=0.90, bottom=0.27, wspace=0.24)
    save_fig(fig, FIGURES_DIR / 'Fig3_supcon_ablation')
    plt.close(fig)


def main():
    print('\n' + '=' * 55)
    print('  SupCon 消融分析（Fig.3）')
    print('=' * 55)
    table_df = build_table()
    out_csv = DATA_DIR / 'supcon_ablation_table.csv'
    table_df.to_csv(out_csv, index=False)
    plot_ablation_figure(table_df)
    print('\n  完成。')


if __name__ == '__main__':
    main()
