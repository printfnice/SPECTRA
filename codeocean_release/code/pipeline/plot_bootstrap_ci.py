# plot_bootstrap_ci.py
# 职责: 用 logit-transform CI 计算 AUROC 区间，避免上界超过 1

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

_PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(_PIPELINE_DIR))
from pub_style import set_pub_style, save_fig, FIGURES_DIR, DATA_DIR, NATURE_CATEGORICAL

DATASET_ORDER = ['kuppe', 'carraro', 'habermann', 'velmeshev', 'reichart']
DATASET_LABELS = {
    'kuppe': 'Kuppe\n(MI)',
    'carraro': 'Carraro\n(CF)',
    'habermann': 'Habermann\n(PF)',
    'velmeshev': 'Velmeshev\n(ASD)',
    'reichart': 'Reichart\n(DCM)',
}


def _logit(x):
    return np.log(x / (1 - x))


def _inv_logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_fold_aurocs():
    """Read the current manuscript-facing fold AUROCs from versioned metric JSONs."""
    fold_aurocs = {}
    for dataset in DATASET_ORDER:
        metrics_path = DATA_DIR / f'{dataset}_spectra_e2e_metrics.json'
        payload = json.loads(metrics_path.read_text())
        fold_aurocs[dataset] = {
            'folds': [float(v) for v in payload['summary']['fold_aurocs']],
            'mean_auroc': float(payload['summary']['mean_auroc']),
        }
    return fold_aurocs


def load_sota_baselines():
    """Use the strongest non-SPECTRA unified downstream baseline per cohort."""
    df = pd.read_csv(DATA_DIR / 'sota_comparison_table.csv')
    spectra_mask = df['method'].astype(str).str.contains('SPECTRA', case=False, na=False)
    baselines = {}
    for dataset in DATASET_ORDER:
        scores = pd.to_numeric(df.loc[~spectra_mask, dataset], errors='coerce').dropna()
        baselines[dataset] = float(scores.max())
    return baselines


def compute_logit_ci(folds, mean_override=None, confidence=0.95):
    x = np.array(folds, dtype=float)
    eps = 1e-4
    x = np.clip(x, eps, 1 - eps)
    z = _logit(x)
    mean = float(np.mean(x)) if mean_override is None else float(mean_override)
    if len(z) < 2 or float(np.std(z, ddof=1)) == 0.0:
        return mean, mean, mean
    z_mean = float(np.mean(z))
    se = stats.sem(z)
    t_crit = stats.t.ppf((1 + confidence) / 2.0, df=len(z) - 1)
    z_lo, z_hi = z_mean - t_crit * se, z_mean + t_crit * se
    lo = float(_inv_logit(z_lo))
    hi = float(_inv_logit(z_hi))
    # Keep the transformed interval consistent with the manuscript-facing mean AUROC.
    lo = min(lo, mean)
    hi = max(hi, mean)
    return mean, lo, hi


def build_ci_table():
    fold_aurocs = load_fold_aurocs()
    sota_baselines = load_sota_baselines()
    rows = []
    for ds in DATASET_ORDER:
        folds = fold_aurocs[ds]['folds']
        mean, lo, hi = compute_logit_ci(folds, mean_override=fold_aurocs[ds]['mean_auroc'])
        sota = sota_baselines[ds]
        rows.append({
            'dataset': ds,
            'n_folds': len(folds),
            'mean_auroc': round(mean, 4),
            'ci_lower_95': round(lo, 4),
            'ci_upper_95': round(hi, 4),
            'ci_width': round(hi - lo, 4),
            'sota_baseline': sota,
            'delta_vs_sota': round(mean - sota, 4),
            'ci_lower_vs_sota': round(lo - sota, 4),
            'ci_upper_vs_sota': round(hi - sota, 4),
            'fold_aurocs': str([round(v, 4) for v in folds]),
            'ci_method': 'logit-t',
        })
    return pd.DataFrame(rows)


def plot_ci_figure(df):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    set_pub_style()
    datasets = DATASET_ORDER
    x = np.arange(len(datasets))
    colors = NATURE_CATEGORICAL[:len(datasets)]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))

    ax = axes[0]
    means = df['mean_auroc'].to_numpy()
    lo = means - df['ci_lower_95'].to_numpy()
    hi = df['ci_upper_95'].to_numpy() - means
    ax.bar(x, means, color=colors, alpha=0.82, width=0.62, edgecolor='none', zorder=3)
    ax.errorbar(x, means, yerr=[lo, hi], fmt='none', ecolor='#333333', elinewidth=1.2, capsize=4, capthick=1.2, zorder=4)
    for i, row in df.iterrows():
        ax.text(i, row['ci_upper_95'] + 0.008, f"{row['mean_auroc']:.4f}\n[{row['ci_lower_95']:.3f},{row['ci_upper_95']:.3f}]", ha='center', va='bottom', fontsize=7)
        ax.plot([i - 0.3, i + 0.3], [row['sota_baseline'], row['sota_baseline']], linestyle='--', linewidth=1.0, color='#8C8C8C', zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets], fontsize=8.5)
    ax.set_ylim(0.6, 1.04)
    ax.set_ylabel('AUROC')
    ax.set_title('(A) Logit-transform 95% CI for AUROC', loc='left', fontsize=10)

    ax2 = axes[1]
    dmean = df['delta_vs_sota'].to_numpy()
    dlo = dmean - df['ci_lower_vs_sota'].to_numpy()
    dhi = df['ci_upper_vs_sota'].to_numpy() - dmean
    ax2.bar(x, dmean, color=colors, alpha=0.82, width=0.62, edgecolor='none', zorder=3)
    ax2.errorbar(x, dmean, yerr=[dlo, dhi], fmt='none', ecolor='#333333', elinewidth=1.2, capsize=4, capthick=1.2, zorder=4)
    ax2.axhline(0, linestyle='--', linewidth=0.8, color='#8C8C8C')
    for i, v in enumerate(dmean):
        ax2.text(i, v + (0.006 if v >= 0 else -0.01), f"{v:+.3f}", ha='center', va='bottom', fontsize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels([DATASET_LABELS[d] for d in datasets], fontsize=8.5)
    ax2.set_ylabel('ΔAUROC vs SOTA')
    ax2.set_title('(B) Delta with propagated logit CI', loc='left', fontsize=10)

    plt.tight_layout()
    save_fig(fig, FIGURES_DIR / 'FigS_bootstrap_ci')
    plt.close(fig)


def main():
    print('\n' + '=' * 55)
    print('  Logit-transform CI for AUROC')
    print('=' * 55)
    df = build_ci_table()
    out_csv = DATA_DIR / 'bootstrap_ci_table.csv'
    df.to_csv(out_csv, index=False)
    print(df[['dataset', 'mean_auroc', 'ci_lower_95', 'ci_upper_95', 'delta_vs_sota']].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'\n  已保存: {out_csv}')
    plot_ci_figure(df)
    print('  完成。')


if __name__ == '__main__':
    main()
