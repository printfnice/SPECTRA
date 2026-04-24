# plot_e2e_vs_unsupervised.py
# 职责: 生成 ExtFig1，比较 E2E 与两阶段流程

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(_PIPELINE_DIR))

from pub_style import ACCENT, FIGURES_DIR, DATA_DIR, set_pub_style, save_fig, style_axis, wrap_labels, add_panel_label, finalize_figure

DATASETS = ['kuppe', 'carraro', 'habermann', 'velmeshev', 'reichart']
DATASET_LABELS = ['Kuppe (MI)', 'Carraro (CF)', 'Habermann (PF)', 'Velmeshev (ASD)', 'Reichart (DCM)']
E2E_SGDR = {
    'kuppe': {'mean': 1.0000, 'std': 0.0000},
    'carraro': {'mean': 0.9429, 'std': 0.0510},
    'habermann': {'mean': 0.9286, 'std': 0.0620},
    'velmeshev': {'mean': 0.8654, 'std': 0.0440},
    'reichart': {'mean': 0.9883, 'std': 0.0121},
}


def load_unsupervised():
    rows = []
    pipeline_output = _PIPELINE_DIR / 'output'
    for ds in DATASETS:
        path = pipeline_output / f'{ds}_ms_hgatlink_tensor.csv'
        if not path.exists():
            rows.append({'dataset': ds, 'mean': np.nan, 'std': np.nan})
            continue
        df = pd.read_csv(path)
        rows.append({'dataset': ds, 'mean': df['auroc'].mean(), 'std': 0.0})
    return pd.DataFrame(rows)


def plot_paired_bar(unsup_df):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    set_pub_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(DATASETS))
    bar_w = 0.34

    bars1 = ax.bar(x - bar_w / 2, unsup_df['mean'], bar_w, color='#7C91B0', edgecolor='white', linewidth=0.5, zorder=3, label='Two-stage pipeline')
    e2e_means = [E2E_SGDR[ds]['mean'] for ds in DATASETS]
    e2e_stds = [E2E_SGDR[ds]['std'] for ds in DATASETS]
    bars2 = ax.bar(x + bar_w / 2, e2e_means, bar_w, color=ACCENT, edgecolor='white', linewidth=0.5, yerr=e2e_stds, error_kw=dict(elinewidth=0.8, ecolor='#555555'), zorder=3, label='E2E pipeline')

    for i, ds in enumerate(DATASETS):
        delta = E2E_SGDR[ds]['mean'] - unsup_df.loc[unsup_df['dataset'] == ds, 'mean'].values[0]
        ax.text(i + bar_w / 2, E2E_SGDR[ds]['mean'] + e2e_stds[i] + 0.014, f'{delta:+.4f}', ha='center', va='bottom', fontsize=7, color=ACCENT if delta >= 0 else '#7A2E2E', fontweight='bold')

    style_axis(ax, grid_axis='y')
    ax.set_ylabel('AUROC')
    ax.set_ylim(0.45, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(wrap_labels(DATASET_LABELS, width=12))
    ax.set_title('End-to-end learning improves phenotype classification', loc='left', pad=10)
    add_panel_label(ax, 'a')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    finalize_figure(fig, left=0.10, right=0.98, top=0.90, bottom=0.23)
    save_fig(fig, FIGURES_DIR / 'ExtFig1_e2e_vs_unsupervised')
    plt.close(fig)


def save_table(unsup_df):
    rows = []
    for ds in DATASETS:
        u = unsup_df.loc[unsup_df['dataset'] == ds].iloc[0]
        e = E2E_SGDR[ds]
        rows.append({
            'dataset': ds,
            'unsup_mean': round(u['mean'], 4),
            'unsup_std': round(u['std'], 4),
            'e2e_mean': e['mean'],
            'e2e_std': e['std'],
            'delta': round(e['mean'] - u['mean'], 4),
        })
    out = DATA_DIR / 'e2e_vs_unsupervised_table.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'  表格已保存: {out}')


def main():
    print('\n' + '=' * 55)
    print('  E2E vs 无监督两阶段对比（Extended Fig.1）')
    print('=' * 55)
    unsup_df = load_unsupervised()
    plot_paired_bar(unsup_df)
    save_table(unsup_df)
    print('\n  完成。')


if __name__ == '__main__':
    main()
