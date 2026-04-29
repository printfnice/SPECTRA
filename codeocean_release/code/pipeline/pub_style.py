# pub_style.py
# 被依赖: plot_*.py 所有可视化脚本
# 职责: 统一 Nature/Cell 风格绘图样式、版式辅助与导出策略

from pathlib import Path
from textwrap import fill

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

_PIPELINE_DIR = Path(__file__).parent
_CLASS_DIR = _PIPELINE_DIR.parent.parent
OUTPUT_DIR = _CLASS_DIR / 'output'
FIGURES_DIR = OUTPUT_DIR / 'figures'
DATA_DIR = OUTPUT_DIR / 'data'
RESULTS_DIR = OUTPUT_DIR / 'results'
GATE_DIR = RESULTS_DIR / 'gate_weights'
E2E_DIR = RESULTS_DIR / 'e2e'

WONG = {
    'black': '#000000',
    'orange': '#E69F00',
    'skyblue': '#56B4E9',
    'green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermilion': '#D55E00',
    'pink': '#CC79A7',
}

NATURE_CATEGORICAL = [
    '#3F6C7B', '#C46B4E', '#7C9C86', '#6B7FA3', '#B88C5A',
    '#A06A7C', '#6D8F8B', '#9B6F53', '#8C8C8C', '#D65F5F',
]

ACCENT = '#B3423E'
INK = '#1F1F1F'
MUTED = '#6F6F6F'
GRID = '#E6E2DD'
SOFT_BG = '#F7F5F2'

SCALE_COLORS = {
    'w_gene': '#335C81',
    'w_pathway': '#B85C38',
    'w_celltype': '#5C8A63',
}
SCALE_LABELS = {
    'w_gene': 'Gene-level',
    'w_pathway': 'Pathway-level',
    'w_celltype': 'Cell-type-level',
}

DATASET_LABELS = {
    'kuppe': 'Kuppe\n(MI)',
    'carraro': 'Carraro\n(CF)',
    'habermann': 'Habermann\n(PF)',
    'velmeshev': 'Velmeshev\n(ASD)',
    'reichart': 'Reichart\n(DCM)',
}
DATASET_FULL = {
    'kuppe': 'Kuppe (Myocardial Infarction)',
    'carraro': 'Carraro (Cystic Fibrosis)',
    'habermann': 'Habermann (Pulmonary Fibrosis)',
    'velmeshev': 'Velmeshev (Autism Spectrum Disorder)',
    'reichart': 'Reichart (Dilated Cardiomyopathy)',
}

FIGURE_SIZES = {
    'single': (3.35, 2.5),
    'single_tall': (3.35, 3.6),
    'double': (6.85, 4.2),
    'double_tall': (6.85, 5.6),
    'wide': (7.8, 4.8),
    'wide_tall': (7.8, 6.2),
}


def set_pub_style():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 8.5,
        'axes.titlesize': 9,
        'axes.titleweight': 'semibold',
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'legend.fontsize': 7.2,
        'legend.title_fontsize': 7.6,
        'axes.linewidth': 0.8,
        'axes.edgecolor': INK,
        'axes.labelcolor': INK,
        'axes.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'grid.color': GRID,
        'grid.linewidth': 0.6,
        'grid.alpha': 0.7,
        'xtick.color': INK,
        'ytick.color': INK,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.major.pad': 3,
        'ytick.major.pad': 3,
        'errorbar.capsize': 2,
        'figure.dpi': 300,
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
        'savefig.bbox': 'tight',
        'axes.prop_cycle': mpl.cycler(color=NATURE_CATEGORICAL),
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def get_figure_size(kind='double'):
    return FIGURE_SIZES.get(kind, FIGURE_SIZES['double'])


def wrap_label(text, width=16):
    return fill(str(text), width=width, break_long_words=False)


def wrap_labels(labels, width=16):
    return [wrap_label(label, width=width) for label in labels]


def style_axis(ax, grid_axis='y'):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, alpha=0.7, zorder=0)
    return ax


def add_panel_label(ax, label, x=-0.08, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='bottom', ha='left', color=INK)


def add_reference_line(ax, y=None, x=None, label=None, color=MUTED, linestyle='--'):
    if y is not None:
        ax.axhline(y, color=color, linewidth=0.9, linestyle=linestyle, zorder=1)
        if label:
            ax.text(0.99, y, label, transform=ax.get_yaxis_transform(),
                    ha='right', va='bottom', fontsize=7, color=color)
    if x is not None:
        ax.axvline(x, color=color, linewidth=0.9, linestyle=linestyle, zorder=1)
        if label:
            ax.text(x, 0.99, label, transform=ax.get_xaxis_transform(), rotation=90,
                    ha='left', va='top', fontsize=7, color=color)


def place_legend(ax, ncol=1, loc='upper left', bbox_to_anchor=(1.02, 1.0), title=None):
    return ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, frameon=False,
                     ncol=ncol, title=title, borderaxespad=0.0, handlelength=1.5)


def annotate_bar_values(ax, bars, fmt='{:.4f}', dy=0.01, color=INK, fontsize=7):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + dy, fmt.format(h),
                ha='center', va='bottom', fontsize=fontsize, color=color)


def finalize_figure(fig, left=None, right=None, top=None, bottom=None, wspace=None, hspace=None):
    fig.subplots_adjust(
        left=0.08 if left is None else left,
        right=0.98 if right is None else right,
        top=0.92 if top is None else top,
        bottom=0.12 if bottom is None else bottom,
        wspace=0.28 if wspace is None else wspace,
        hspace=0.28 if hspace is None else hspace,
    )
    return fig


def save_fig(fig, path_stem, formats=('pdf', 'png')):
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = path_stem.with_suffix(f'.{fmt}')
        fig.savefig(out, dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0.03, format=fmt)
        print(f'  保存: {out}')


def add_significance_bar(ax, x1, x2, y, pval, h=0.015):
    if pval < 0.001:
        sig = '***'
    elif pval < 0.01:
        sig = '**'
    elif pval < 0.05:
        sig = '*'
    else:
        return
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, color=INK)
    ax.text((x1 + x2) / 2, y + h, sig, ha='center', va='bottom', fontsize=7, color=INK)
