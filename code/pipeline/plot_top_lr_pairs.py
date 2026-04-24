# plot_top_lr_pairs.py
# 职责: 从 gate_weights CSV 提取 top LR pair 重要性（每数据集单图）

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PIPELINE_DIR = Path(__file__).parent
_CLASS_DIR = _PIPELINE_DIR.parent.parent
sys.path.insert(0, str(_PIPELINE_DIR))

from pub_style import (
    set_pub_style,
    save_fig,
    FIGURES_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    SCALE_COLORS,
    SCALE_LABELS,
    DATASET_FULL,
    wrap_labels,
    style_axis,
    add_reference_line,
    finalize_figure,
)

DATASETS = ['kuppe', 'carraro', 'habermann', 'velmeshev', 'reichart']
TOP_N = 20
_LR2PW = {}
_PW_COLORS = {}
_PW_PALETTE = ['#3F6C7B', '#C46B4E', '#7C9C86', '#6B7FA3', '#B88C5A', '#A06A7C', '#6D8F8B', '#9B6F53', '#8C8C8C']


def _ensure_cellchat():
    global _LR2PW, _PW_COLORS
    if _LR2PW:
        return
    ccdb_path = _CLASS_DIR / 'data' / 'cellchat_pathway_map.csv'
    if not ccdb_path.exists():
        return
    df = pd.read_csv(ccdb_path)
    for _, row in df.iterrows():
        _LR2PW[(row['ligand'], row['receptor'])] = row['pathway']
    for i, pathway in enumerate(sorted(set(_LR2PW.values()))):
        _PW_COLORS[pathway] = _PW_PALETTE[i % len(_PW_PALETTE)]


def assign_pathway(ligand, receptor):
    _ensure_cellchat()
    pw = _LR2PW.get((ligand, receptor))
    if pw:
        return pw
    if 'ITGA' in ligand or 'ITGB' in ligand or 'ITGA' in receptor or 'ITGB' in receptor:
        return 'Integrin (custom)'
    if ligand.startswith('FGF') or receptor.startswith('FGFR'):
        return 'FGF (custom)'
    if 'EGF' in ligand or receptor == 'EGFR' or 'ST6GAL' in ligand:
        return 'EGFR (custom)'
    if 'TGFB' in ligand or 'TGFB' in receptor or 'BMP' in ligand:
        return 'TGF-beta (custom)'
    if 'DLL' in ligand or 'JAG' in ligand or 'NOTCH' in receptor:
        return 'NOTCH (custom)'
    if ligand.startswith('WNT') or 'FZD' in receptor:
        return 'WNT (custom)'
    if 'PDGF' in ligand or 'PDGFR' in receptor:
        return 'PDGF (custom)'
    prefix = ligand[:4].rstrip('0123456789')
    return f'{prefix}… (uncharacterized)'


def load_gate_data(dataset):
    path = OUTPUT_DIR / f'{dataset}_gate_weights.csv'
    if not path.exists():
        print(f'  [WARN] 未找到 {path}')
        return None
    df = pd.read_csv(path)
    if 'lr_name' not in df.columns:
        print(f'  [WARN] {dataset} gate CSV 无 lr_name 列')
        return None
    return df


def compute_top_lr_pairs(df, top_n=TOP_N):
    parts = df['lr_name'].str.split('__', expand=True)
    df = df.copy()
    df['ligand'] = parts[0]
    df['receptor'] = parts[1]
    df['lr_pair'] = df['ligand'] + '__' + df['receptor']

    grouped = df.groupby('lr_pair').agg(
        w_gene=('w_gene', 'mean'),
        w_pathway=('w_pathway', 'mean'),
        w_celltype=('w_celltype', 'mean'),
        n_entries=('w_gene', 'count'),
    ).reset_index()

    grouped['ligand'] = grouped['lr_pair'].str.split('__').str[0]
    grouped['receptor'] = grouped['lr_pair'].str.split('__').str[1]
    grouped['pathway'] = grouped.apply(lambda r: assign_pathway(r['ligand'], r['receptor']), axis=1)
    grouped['max_gate'] = grouped[['w_gene', 'w_pathway', 'w_celltype']].max(axis=1)
    grouped['dominance'] = grouped['max_gate'] - 1 / 3
    grouped['dominant'] = grouped[['w_gene', 'w_pathway', 'w_celltype']].idxmax(axis=1)
    return grouped.nlargest(top_n, 'dominance').reset_index(drop=True)


def plot_top_lr(top_df, dataset):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    set_pub_style()
    plot_df = top_df.sort_values('dominance', ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    y = np.arange(len(plot_df))
    bar_colors = [SCALE_COLORS[key] for key in plot_df['dominant']]
    bars = ax.barh(y, plot_df['dominance'], color=bar_colors, edgecolor='white', linewidth=0.5, height=0.72, zorder=3)

    xmax = plot_df['dominance'].max()
    for yi, (_, row) in enumerate(plot_df.iterrows()):
        pathway_label = row['pathway'][:24] + ('…' if len(row['pathway']) > 24 else '')
        ax.text(row['dominance'] + xmax * 0.015, yi, pathway_label, va='center', ha='left', fontsize=6.8, color='#5B5B5B')

    style_axis(ax, grid_axis='x')
    add_reference_line(ax, x=0.0)
    ax.set_yticks(y)
    ax.set_yticklabels(wrap_labels(plot_df['lr_pair'].str.replace('__', ' -> '), width=22))
    ax.set_xlabel('Gate dominance score')
    ax.set_xlim(0, xmax * 1.55)
    ax.set_title(
        f"Top-{len(plot_df)} LR pairs ranked by scale dominance\n{DATASET_FULL.get(dataset, dataset)}",
        loc='left',
        pad=10,
    )

    handles = [mpatches.Patch(facecolor=SCALE_COLORS[key], label=SCALE_LABELS[key]) for key in ['w_gene', 'w_pathway', 'w_celltype']]
    ax.legend(handles=handles, title='Dominant scale', loc='upper center', bbox_to_anchor=(0.50, -0.10), ncol=3, frameon=False)

    finalize_figure(fig, left=0.27, right=0.98, top=0.93, bottom=0.16)
    save_fig(fig, FIGURES_DIR / f'Fig5a_top_lr_pairs_{dataset}')
    plt.close(fig)


def main():
    print('\n' + '=' * 55)
    print('  Top LR Pair 重要性分析（Fig.5a）')
    print('=' * 55)
    _ensure_cellchat()
    print(f'  CellChatDB: {len(_LR2PW)} LR pairs 已加载')

    for ds in DATASETS:
        print(f'\n  {ds}:')
        df = load_gate_data(ds)
        if df is None:
            continue
        top = compute_top_lr_pairs(df)
        csv_path = DATA_DIR / f'top_lr_pairs_{ds}.csv'
        top.to_csv(csv_path, index=False)
        plot_top_lr(top, ds)
        print(f'    Top-{TOP_N} 已保存: {csv_path}')

    print('\n  完成。')


if __name__ == '__main__':
    main()
