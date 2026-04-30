# plot_pathway_scale.py

import sys
from pathlib import Path
import numpy as np
import pandas as pd

_PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(_PIPELINE_DIR))

from pub_style import set_pub_style, save_fig, FIGURES_DIR, DATA_DIR, OUTPUT_DIR, DATASET_LABELS, add_panel_label, finalize_figure

DATASETS = ['kuppe', 'carraro', 'habermann', 'velmeshev', 'reichart']
READ_NROWS = {'kuppe': None, 'carraro': None, 'habermann': None, 'velmeshev': None, 'reichart': 800000}
TOP_N_PATHWAYS = 18


def _bh_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def _load_pw_map():
    mpath = Path(__file__).parent.parent.parent / 'data' / 'cellchat_pathway_map.csv'
    m = pd.read_csv(mpath)
    m = m[['ligand', 'receptor', 'pathway']].drop_duplicates()
    return m


def _load_gate_scores():
    pw_map = _load_pw_map()
    rows = []
    for ds in DATASETS:
        fp = OUTPUT_DIR / f'{ds}_gate_weights.csv'
        if not fp.exists():
            continue
        df = pd.read_csv(fp, usecols=['lr_name', 'w_gene', 'w_pathway', 'w_celltype'], nrows=READ_NROWS.get(ds))
        sp = df['lr_name'].str.split('__', expand=True)
        df['ligand'] = sp[0]
        df['receptor'] = sp[1]
        df = df.merge(pw_map, on=['ligand', 'receptor'], how='left')
        df['pathway'] = df['pathway'].fillna(df['ligand'].str[:4] + '_other')
        df['pathway_score'] = df[['w_gene', 'w_pathway', 'w_celltype']].max(axis=1)
        g = df.groupby('pathway', as_index=False).agg(pathway_score=('pathway_score', 'mean'), n_lr=('lr_name', 'nunique'))
        g['dataset'] = ds
        rows.append(g)
    if not rows:
        return pd.DataFrame(columns=['dataset', 'pathway', 'pathway_score', 'n_lr'])
    return pd.concat(rows, ignore_index=True)


def _load_enrichment_with_q():
    fp = DATA_DIR / 'pathway_enrichment_combined.csv'
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    outs = []
    for ds, sub in df.groupby('dataset'):
        sub = sub.copy()
        sub['qvalue_bh'] = _bh_adjust(sub['pvalue'].to_numpy())
        outs.append(sub)
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()


def plot_fig5():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    set_pub_style()

    gate = _load_gate_scores()
    enr = _load_enrichment_with_q()

    if gate.empty:
        print('  [WARN] no gate data')
        return

    piv = gate.pivot_table(index='pathway', columns='dataset', values='pathway_score', aggfunc='mean')
    keep = piv.fillna(0).mean(axis=1).sort_values(ascending=False).head(TOP_N_PATHWAYS).index.tolist()
    piv = piv.loc[keep, DATASETS]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 5.8), gridspec_kw={'width_ratios': [1.3, 1.0]})

    ax = axes[0]
    blue_cmap = plt.cm.get_cmap('Blues').copy()
    blue_cmap.set_bad(color='#E2E6EA')
    masked = np.ma.masked_invalid(piv.values.astype(float))
    vmax = float(np.nanmax(piv.values)) if np.isfinite(np.nanmax(piv.values)) else 1.0
    im = ax.imshow(masked, aspect='auto', cmap=blue_cmap, vmin=0.0, vmax=vmax)
    ax.set_xticks(np.arange(len(DATASETS)))
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in DATASETS], rotation=25, ha='right', fontsize=7)
    ax.set_yticks(np.arange(len(keep)))
    ax.set_yticklabels(keep, fontsize=6.6)
    ax.set_title('Mapped-pathway coverage weighted by cohort gate allocation', loc='left', fontsize=10)
    add_panel_label(ax, 'a')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label('Dominant-scale intensity at mapped LR pairs', fontsize=8)

    ax2 = axes[1]
    add_panel_label(ax2, 'b')
    if enr.empty:
        ax2.text(0.5, 0.5, 'No enrichment data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_xticks([]); ax2.set_yticks([])
    else:
        top = enr.sort_values('qvalue_bh').head(15).copy()
        top['neglog10_q'] = -np.log10(top['qvalue_bh'].clip(lower=1e-300))
        top = top.iloc[::-1].reset_index(drop=True)
        cmap = {d: c for d, c in zip(DATASETS, ['#4E79A7', '#F28E2B', '#59A14F', '#E15759', '#76B7B2'])}
        ax2.barh(np.arange(len(top)), top['neglog10_q'], color=[cmap.get(d, '#9AA0A6') for d in top['dataset']], alpha=0.86)
        ax2.set_yticks(np.arange(len(top)))
        ax2.set_yticklabels(top['pathway'], fontsize=6.6)
        ax2.set_xlabel('-log10(BH q-value)')
        ax2.set_title('Top enriched pathways (BH-adjusted)', loc='left', fontsize=10)
        ax2.axvline(-np.log10(0.05), color='#666666', linestyle='--', linewidth=0.8)
        handles = [mpatches.Patch(facecolor=cmap[d], label=DATASET_LABELS.get(d, d)) for d in DATASETS]
        ax2.legend(handles=handles, loc='lower right', fontsize=6.5, frameon=False, ncol=1)

    finalize_figure(fig, left=0.12, right=0.98, top=0.92, bottom=0.10, wspace=0.38)
    save_fig(fig, FIGURES_DIR / 'Fig5_pathway_scale')
    plt.close(fig)

    if not enr.empty:
        out = DATA_DIR / 'pathway_enrichment_with_q.csv'
        enr.to_csv(out, index=False)
        print(f'  保存: {out}')

    print('  保存: output/figures/Fig5_pathway_scale.pdf/png')


def main():
    print('\n' + '=' * 55)
    print('  Fig.5 重绘（heatmap + BH q-value）')
    print('=' * 55)
    plot_fig5()
    print('  完成。')


if __name__ == '__main__':
    main()
