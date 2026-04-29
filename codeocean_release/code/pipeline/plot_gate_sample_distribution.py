# plot_gate_sample_distribution.py

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

_PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(_PIPELINE_DIR))

from pub_style import set_pub_style, save_fig, FIGURES_DIR, DATA_DIR, OUTPUT_DIR, add_panel_label, finalize_figure

DATASETS = ['velmeshev', 'reichart']
READ_NROWS = {'velmeshev': None, 'reichart': None}
CHUNK_ROWS = 200000
COLORS = {'w_gene': '#4E79A7', 'w_pathway': '#F28E2B', 'w_celltype': '#59A14F'}


def load_sample_gate(ds):
    p = OUTPUT_DIR / f'{ds}_gate_weights.csv'
    if not p.exists():
        return None
    usecols = ['sample_id', 'w_gene', 'w_pathway', 'w_celltype']
    nrows = READ_NROWS.get(ds)
    if nrows is not None:
        df = pd.read_csv(p, usecols=usecols, nrows=nrows)
        g = df.groupby('sample_id', as_index=False)[['w_gene', 'w_pathway', 'w_celltype']].mean()
    elif p.stat().st_size > 200 * 1024 * 1024:
        accum = None
        for chunk in pd.read_csv(p, usecols=usecols, chunksize=CHUNK_ROWS):
            grp = chunk.groupby('sample_id')[['w_gene', 'w_pathway', 'w_celltype']].agg(['sum', 'count'])
            if accum is None:
                accum = grp
            else:
                accum = accum.add(grp, fill_value=0.0)
        if accum is None or accum.empty:
            return None
        sums = accum.xs('sum', axis=1, level=1)
        counts = accum.xs('count', axis=1, level=1)
        g = (sums / counts).reset_index()
    else:
        df = pd.read_csv(p, usecols=usecols)
        g = df.groupby('sample_id', as_index=False)[['w_gene', 'w_pathway', 'w_celltype']].mean()
    s = g[['w_gene', 'w_pathway', 'w_celltype']].sum(axis=1)
    g[['w_gene', 'w_pathway', 'w_celltype']] = g[['w_gene', 'w_pathway', 'w_celltype']].div(s, axis=0)
    g['dataset'] = ds
    return g


def to_xy(gene, pathway, cell):
    x = 0.5 * (2 * pathway + cell)
    y = (np.sqrt(3) / 2.0) * cell
    return x, y


def draw_ternary_frame(ax):
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
    ax.plot(tri[:, 0], tri[:, 1], color='#555555', linewidth=0.9, zorder=3)
    for level in (0.25, 0.50, 0.75):
        # constant gene
        p1 = to_xy(level, 1 - level, 0)
        p2 = to_xy(level, 0, 1 - level)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#D8DEE8', linewidth=0.6, zorder=1)
        # constant pathway
        p1 = to_xy(1 - level, level, 0)
        p2 = to_xy(0, level, 1 - level)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#D8DEE8', linewidth=0.6, zorder=1)
        # constant cell-type
        p1 = to_xy(1 - level, 0, level)
        p2 = to_xy(0, 1 - level, level)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#D8DEE8', linewidth=0.6, zorder=1)
    ax.text(-0.03, -0.03, 'Gene', fontsize=8)
    ax.text(1.01, -0.03, 'Pathway', fontsize=8, ha='right')
    ax.text(0.5, np.sqrt(3)/2 + 0.03, 'Cell-type', fontsize=8, ha='center')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, np.sqrt(3)/2 + 0.08)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')


def add_vertex_inset(ax, x, y, title):
    iax = inset_axes(ax, width="40%", height="40%", loc='upper right', borderpad=0.8)
    iax.scatter(x, y, s=18, color='#2F6C9E', alpha=0.9, edgecolors='white', linewidths=0.35)
    iax.set_xlim(0.78, 1.01)
    iax.set_ylim(-0.01, 0.18)
    iax.set_xticks([0.8, 0.9, 1.0])
    iax.set_yticks([0.0, 0.1])
    iax.tick_params(labelsize=5.5, length=2)
    iax.set_title(title, fontsize=6.2, pad=2)
    for spine in iax.spines.values():
        spine.set_color('#7A8796')
        spine.set_linewidth(0.7)
    return iax


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print('\n' + '=' * 55)
    print('  样本级 gate 分布图')
    print('=' * 55)

    set_pub_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2))
    out_rows = []

    for j, ds in enumerate(DATASETS):
        g = load_sample_gate(ds)
        if g is None or g.empty:
            for i in range(2):
                axes[i, j].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            continue

        ax = axes[0, j]
        x, y = to_xy(g['w_gene'].to_numpy(), g['w_pathway'].to_numpy(), g['w_celltype'].to_numpy())
        ax.scatter(x, y, s=28, color='#2F6C9E', alpha=0.85, edgecolors='white', linewidths=0.4)
        draw_ternary_frame(ax)
        add_vertex_inset(ax, x, y, 'Pathway-corner zoom')
        ax.set_title(f'{ds.capitalize()} sample-level ternary gate distribution', fontsize=8.5)

        ax2 = axes[1, j]
        g2 = g.sort_values('w_pathway', ascending=False).reset_index(drop=True)
        xs = np.arange(len(g2))
        b1 = g2['w_gene'].to_numpy(); b2 = g2['w_pathway'].to_numpy(); b3 = g2['w_celltype'].to_numpy()
        ax2.bar(xs, b1, color=COLORS['w_gene'], width=0.85, label='Gene')
        ax2.bar(xs, b2, bottom=b1, color=COLORS['w_pathway'], width=0.85, label='Pathway')
        ax2.bar(xs, b3, bottom=b1+b2, color=COLORS['w_celltype'], width=0.85, label='Cell-type')
        ax2.set_ylim(0, 1.0)
        ax2.set_xlim(-0.8, len(xs)-0.2)
        ax2.set_ylabel('Mean gate weight' if j == 0 else '')
        ax2.set_xlabel('Samples (sorted by pathway weight)')
        ax2.set_title(f'{ds.capitalize()} per-sample gate composition', fontsize=8.5)

        vec = g[['w_gene', 'w_pathway', 'w_celltype']].to_numpy(dtype=float)
        mean_vec = vec.mean(axis=0)
        mean_vec = mean_vec / mean_vec.sum()
        mean_norm = np.linalg.norm(mean_vec)
        vec_norm = np.linalg.norm(vec, axis=1)
        cosine = (vec @ mean_vec) / np.clip(vec_norm * mean_norm, 1e-12, None)
        l2 = np.linalg.norm(vec - mean_vec[None, :], axis=1)

        out_rows.append({
            'dataset': ds,
            'n_samples': len(g),
            'std_gene': round(float(g['w_gene'].std(ddof=1)), 4),
            'std_pathway': round(float(g['w_pathway'].std(ddof=1)), 4),
            'std_celltype': round(float(g['w_celltype'].std(ddof=1)), 4),
            'median_cosine_to_mean': round(float(np.median(cosine)), 4),
            'mean_cosine_to_mean': round(float(np.mean(cosine)), 4),
            'median_l2_to_mean': round(float(np.median(l2)), 4),
            'mean_l2_to_mean': round(float(np.mean(l2)), 4),
        })

    add_panel_label(axes[0, 0], 'a'); add_panel_label(axes[0, 1], 'b')
    add_panel_label(axes[1, 0], 'c'); add_panel_label(axes[1, 1], 'd')

    handles, labels = axes[1, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False)

    finalize_figure(fig, left=0.07, right=0.98, top=0.90, bottom=0.08, wspace=0.20, hspace=0.30)
    save_fig(fig, FIGURES_DIR / 'FigS_gate_sample_distribution')
    plt.close(fig)

    if out_rows:
        pd.DataFrame(out_rows).to_csv(DATA_DIR / 'gate_sample_dispersion.csv', index=False)
        print('  保存: output/data/gate_sample_dispersion.csv')

    print('  保存: output/figures/FigS_gate_sample_distribution.pdf/png')


if __name__ == '__main__':
    main()
