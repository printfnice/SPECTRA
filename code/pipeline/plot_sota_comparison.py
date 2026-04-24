# plot_sota_comparison.py
# 职责: 生成 Fig.2 SOTA 对比图（Nature/Cell 风格）

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(_PIPELINE_DIR))

from pub_style import (
    ACCENT,
    FIGURES_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    NATURE_CATEGORICAL,
    set_pub_style,
    save_fig,
    style_axis,
    wrap_labels,
    add_panel_label,
    finalize_figure,
)

DATASETS = ['kuppe', 'carraro', 'habermann', 'velmeshev', 'reichart']
DATASET_LABELS = ['Kuppe (MI, N=23)', 'Carraro (CF, N=16)', 'Habermann (PF, N=18)', 'Velmeshev (ASD, N=38)', 'Reichart (DCM, N=171)']
METHODS_SHOW = ['cellphonedb', 'cellchat', 'natmi', 'geometric_mean', 'SPECTRA (E2E, SGDR)']
METHOD_LABELS = {
    'cellphonedb': 'CellPhoneDB',
    'cellchat': 'CellChat',
    'natmi': 'NATMI',
    'geometric_mean': 'Geometric Mean',
    'SPECTRA (E2E, SGDR)': 'SPECTRA (E2E)',
}
METHOD_COLORS = {
    'cellphonedb': NATURE_CATEGORICAL[3],
    'cellchat': NATURE_CATEGORICAL[2],
    'natmi': NATURE_CATEGORICAL[4],
    'geometric_mean': NATURE_CATEGORICAL[5],
    'SPECTRA (E2E, SGDR)': ACCENT,
}
E2E_SGDR_RESULTS = {
    'kuppe': {'mean_auroc': 1.0000, 'std_auroc': 0.0000},
    'carraro': {'mean_auroc': 0.9429, 'std_auroc': 0.0510},
    'habermann': {'mean_auroc': 0.9286, 'std_auroc': 0.0620},
    'velmeshev': {'mean_auroc': 0.8654, 'std_auroc': 0.0440},
    'reichart': {'mean_auroc': 0.9883, 'std_auroc': 0.0121},
}


def load_comparison_data():
    base_df = pd.read_csv(OUTPUT_DIR / 'unified_comparison_R30.csv')
    base_df = base_df[base_df['method'] != 'SPECTRA (E2E)']
    e2e_df = pd.DataFrame([
        {'dataset': ds, 'method': 'SPECTRA (E2E, SGDR)', 'mean_auroc': vals['mean_auroc'], 'std_auroc': vals['std_auroc']}
        for ds, vals in E2E_SGDR_RESULTS.items()
    ])
    return pd.concat([base_df, e2e_df], ignore_index=True)


def build_pivot(df):
    df = df[df['method'].isin(METHODS_SHOW)]
    pivot_mean = df.pivot(index='dataset', columns='method', values='mean_auroc').reindex(DATASETS)
    pivot_std = df.pivot(index='dataset', columns='method', values='std_auroc').reindex(DATASETS)
    return pivot_mean, pivot_std


def plot_grouped_bar(pivot_mean, pivot_std):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    set_pub_style()
    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    x = np.arange(len(DATASETS))
    total_width = 0.78
    bar_w = total_width / len(METHODS_SHOW)
    offsets = np.linspace(-total_width / 2 + bar_w / 2, total_width / 2 - bar_w / 2, len(METHODS_SHOW))

    for i, method in enumerate(METHODS_SHOW):
        means = pivot_mean[method].to_numpy()
        stds = pivot_std[method].fillna(0).to_numpy()
        bars = ax.bar(
            x + offsets[i], means, bar_w,
            color=METHOD_COLORS[method],
            edgecolor='white', linewidth=0.5,
            yerr=stds,
            error_kw=dict(elinewidth=0.8, ecolor='#555555', capthick=0.8),
            zorder=3, label=METHOD_LABELS[method],
        )
        if method == 'SPECTRA (E2E, SGDR)':
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f'{val:.4f}', ha='center', va='bottom', fontsize=7, color=ACCENT, fontweight='bold')

    style_axis(ax, grid_axis='y')
    ax.set_ylim(0.45, 1.06)
    ax.set_ylabel('AUROC')
    ax.set_xticks(x)
    ax.set_xticklabels(wrap_labels(DATASET_LABELS, width=14))
    ax.set_title('Performance comparison across five datasets', loc='left', pad=10)
    add_panel_label(ax, 'a')
    ax.axhline(1.0, color='#A7A7A7', linewidth=0.8, linestyle='--', zorder=1)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    finalize_figure(fig, left=0.09, right=0.98, top=0.90, bottom=0.24)
    save_fig(fig, FIGURES_DIR / 'Fig2_sota_comparison')
    plt.close(fig)


def save_table(df):
    pivot = df.pivot(index='method', columns='dataset', values='mean_auroc')
    pivot = pivot[[d for d in DATASETS if d in pivot.columns]]
    pivot['Mean'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('Mean', ascending=False)
    out_path = DATA_DIR / 'sota_comparison_table.csv'
    pivot.round(4).to_csv(out_path)
    print(f'  Table S1 已保存: {out_path}')


def main():
    print('\n' + '=' * 55)
    print('  SOTA 对比图（Fig.2）')
    print('=' * 55)
    df = load_comparison_data()
    pivot_mean, pivot_std = build_pivot(df)
    plot_grouped_bar(pivot_mean, pivot_std)
    save_table(df)
    print('\n  完成。')


if __name__ == '__main__':
    main()
