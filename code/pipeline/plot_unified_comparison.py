# plot_unified_comparison.py
# 职责: 生成补充全方法原始/E2E benchmark 热图（Nature/Cell 风格）

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(_PIPELINE_DIR))

from pub_style import FIGURES_DIR, DATA_DIR, OUTPUT_DIR, DATASET_LABELS, set_pub_style, save_fig, add_panel_label, finalize_figure

DATASETS = ['kuppe', 'carraro', 'habermann', 'velmeshev', 'reichart']
METHODS_ORDER = ['cellphonedb', 'connectome', 'cellchat', 'scseqcomm', 'singlecellsignalr', 'natmi', 'logfc', 'geometric_mean', 'rank_aggregate', 'SPECTRA (E2E, SGDR)']
METHOD_DISPLAY = {
    'cellphonedb': 'CellPhoneDB',
    'connectome': 'Connectome',
    'cellchat': 'CellChat',
    'scseqcomm': 'scSeqComm',
    'singlecellsignalr': 'SingleCellSignalR',
    'natmi': 'NATMI',
    'logfc': 'LogFC',
    'geometric_mean': 'Geometric Mean',
    'rank_aggregate': 'Rank Aggregate',
    'SPECTRA (E2E, SGDR)': 'SPECTRA (E2E)',
}
SOURCE_CSV = DATA_DIR / 'current_methods_performance_2026-04-07.csv'


def _load_comparison_data():
    return pd.read_csv(SOURCE_CSV)


def _build_pivot(df):
    if {'method', 'dataset', 'mean_auroc'}.issubset(df.columns):
        pivot = df.pivot_table(index='method', columns='dataset', values='mean_auroc', aggfunc='first')
    else:
        pivot = df.set_index('method')
    keep = [m for m in METHODS_ORDER if m in pivot.index]
    return pivot.reindex(keep)[DATASETS].astype(float)


def _compute_table_s2(pivot):
    rows = []
    for method in pivot.index:
        vals = {ds: pivot.loc[method, ds] for ds in DATASETS}
        mean_val = np.nanmean(list(vals.values()))
        rows.append({
            'Method': METHOD_DISPLAY.get(method, method),
            **{DATASET_LABELS[ds].replace('\n', ' '): round(vals[ds], 4) for ds in DATASETS},
            'Mean': round(mean_val, 4),
        })
    return pd.DataFrame(rows)


def plot_comparison_heatmap(pivot):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    set_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 5.6), gridspec_kw={'width_ratios': [3.0, 1.0]})
    ax, ax2 = axes
    data = pivot.values.astype(float)
    blue_cmap = LinearSegmentedColormap.from_list('nat_blues', ['#F4F8FC', '#AFC6DF', '#5F88B5', '#244A73'])
    highlight_blue = '#2D5F8B'
    text_blue = '#1F4E79'
    im = ax.imshow(data, cmap=blue_cmap, vmin=0.50, vmax=1.00, aspect='auto')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f'{v:.4f}', ha='center', va='center', fontsize=7, color='white' if v > 0.78 else '#12304A', fontweight='bold' if pivot.index[i] == 'SPECTRA (E2E, SGDR)' else 'normal')

    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([METHOD_DISPLAY.get(m, m) for m in pivot.index])
    for tick, method in zip(ax.get_yticklabels(), pivot.index):
        if method == 'SPECTRA (E2E, SGDR)':
            tick.set_color(text_blue)
            tick.set_fontweight('bold')
    ax.set_title('Expanded original-protocol benchmark', loc='left', pad=10)
    add_panel_label(ax, 'a')

    # Match panel (b) to the top-to-bottom visual order shown in panel (a).
    display_methods = list(reversed(list(pivot.index)))
    means = [np.nanmean(pivot.loc[m, DATASETS].values) for m in display_methods]
    colors = [highlight_blue if m == 'SPECTRA (E2E, SGDR)' else '#9DB1C9' for m in display_methods]
    ax2.barh(range(len(means)), means, color=colors, height=0.72)
    for i, (v, m) in enumerate(zip(means, display_methods)):
        ax2.text(v + 0.006, i, f'{v:.4f}', va='center', fontsize=7, color=text_blue if m == 'SPECTRA (E2E, SGDR)' else '#425466', fontweight='bold' if m == 'SPECTRA (E2E, SGDR)' else 'normal')
    ax2.set_yticks(range(len(means)))
    ax2.set_yticklabels([''] * len(means))
    ax.invert_yaxis()
    ax2.invert_yaxis()
    ax2.set_xlim(0.5, 1.05)
    ax2.set_xlabel('Mean AUROC')
    ax2.set_title('Across datasets (original protocol)', loc='left', pad=10)
    add_panel_label(ax2, 'b')
    for spine in ['top', 'right', 'left']:
        ax2.spines[spine].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    finalize_figure(fig, left=0.12, right=0.97, top=0.92, bottom=0.10, wspace=0.15)
    save_fig(fig, FIGURES_DIR / 'FigS_unified_comparison')
    plt.close(fig)


def main():
    print('\n' + '=' * 55)
    print('  Expanded SOTA Comparison 可视化')
    print('=' * 55)
    df = _load_comparison_data()
    pivot = _build_pivot(df)
    table_s2 = _compute_table_s2(pivot)
    table_s2.to_csv(DATA_DIR / 'expanded_sota_comparison_table.csv', index=False)
    plot_comparison_heatmap(pivot)
    print('\n  完成。')


if __name__ == '__main__':
    main()
