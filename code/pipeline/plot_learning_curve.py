# plot_learning_curve.py
# 依赖: run_learning_curve.py 生成的 output/data/learning_curve_results.json
# 被依赖: 直接运行
# 职责: 绘制学习曲线图 FigS_learning_curve.pdf，用于论文支撑小样本创新点

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import sys

_PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(_PIPELINE_DIR))

DATA_FILE   = _PIPELINE_DIR / 'output' / 'data' / 'learning_curve_results.json'
OUT_PDF     = _PIPELINE_DIR / 'output' / 'figures' / 'FigS_learning_curve.pdf'
OUT_PNG     = _PIPELINE_DIR / 'output' / 'figures' / 'FigS_learning_curve.png'

SPECTRA_COLOR = '#E05252'
RF_COLOR      = '#4A90D9'


def load_results():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"结果文件不存在: {DATA_FILE}\n请先运行 run_learning_curve.py")
    with open(str(DATA_FILE)) as f:
        data = json.load(f)
    df = pd.DataFrame(data['records'])
    return df, data['ratios']


def compute_stats(df, ratios):
    rows = []
    for r in ratios:
        sub = df[df['ratio'] == r]
        rows.append({
            'ratio':      r,
            'n_mean':     sub['n_train'].mean(),
            'rf_mean':    sub['auroc_rf'].mean(),
            'rf_std':     sub['auroc_rf'].std(),
            'e2e_mean':   sub['auroc_e2e'].mean(),
            'e2e_std':    sub['auroc_e2e'].std(),
            'delta_mean': (sub['auroc_e2e'] - sub['auroc_rf']).mean(),
            'delta_std':  (sub['auroc_e2e'] - sub['auroc_rf']).std(),
        })
    return pd.DataFrame(rows)


def plot(stats):
    try:
        from pub_style import set_pub_style
        set_pub_style()
    except Exception:
        plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ── Panel A: AUROC vs n_train ──────────────────────────────────────────
    ax = axes[0]
    ns = stats['n_mean'].values

    ax.plot(ns, stats['e2e_mean'], 'o-', color=SPECTRA_COLOR,
            linewidth=2, markersize=6, label='SPECTRA E2E', zorder=3)
    ax.fill_between(ns,
                    stats['e2e_mean'] - stats['e2e_std'],
                    stats['e2e_mean'] + stats['e2e_std'],
                    color=SPECTRA_COLOR, alpha=0.15, zorder=2)

    ax.plot(ns, stats['rf_mean'], 's--', color=RF_COLOR,
            linewidth=2, markersize=6, label='CellChat + RF (baseline)', zorder=3)
    ax.fill_between(ns,
                    stats['rf_mean'] - stats['rf_std'],
                    stats['rf_mean'] + stats['rf_std'],
                    color=RF_COLOR, alpha=0.15, zorder=2)

    ax.set_xlabel('Training set size (N)', fontsize=11)
    ax.set_ylabel('AUROC', fontsize=11)
    ax.set_title('A   AUROC vs. Training Set Size\n(Reichart, DCM classification)',
                 fontsize=11, loc='left', fontweight='bold')
    ax.set_ylim(0.45, 1.05)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # 标注每个点的训练集比例
    for _, row in stats.iterrows():
        ax.annotate(f"{row['ratio']:.0%}",
                    xy=(row['n_mean'], row['e2e_mean']),
                    xytext=(4, 6), textcoords='offset points',
                    fontsize=7.5, color=SPECTRA_COLOR)

    # ── Panel B: Delta AUROC vs n_train ───────────────────────────────────
    ax2 = axes[1]
    ax2.bar(ns, stats['delta_mean'], width=ns * 0.06,
            color=[SPECTRA_COLOR if d > 0 else '#aaaaaa' for d in stats['delta_mean']],
            alpha=0.85, zorder=3)
    ax2.errorbar(ns, stats['delta_mean'], yerr=stats['delta_std'],
                 fmt='none', color='#333333', capsize=4, linewidth=1.2, zorder=4)
    ax2.axhline(0, color='#555555', linewidth=0.8, linestyle='--')

    ax2.set_xlabel('Training set size (N)', fontsize=11)
    ax2.set_ylabel('ΔAUROC (SPECTRA − RF)', fontsize=11)
    ax2.set_title('B   Performance Gain vs. Training Set Size\n(SPECTRA E2E − CellChat+RF)',
                  fontsize=11, loc='left', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # 标注数值
    for _, row in stats.iterrows():
        ax2.text(row['n_mean'], row['delta_mean'] + row['delta_std'] + 0.005,
                 f"{row['delta_mean']:+.3f}",
                 ha='center', va='bottom', fontsize=8, color='#333333')

    plt.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUT_PDF), bbox_inches='tight', dpi=300)
    plt.savefig(str(OUT_PNG), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"图表已保存: {OUT_PDF}")
    print(f"图表已保存: {OUT_PNG}")


def print_summary(stats):
    print("\n=== 学习曲线结果汇总 ===")
    print(f"{'Ratio':>6}  {'N_train':>7}  {'RF':>8}  {'E2E':>8}  {'ΔAUROC':>8}")
    for _, row in stats.iterrows():
        print(f"{row['ratio']:>6.0%}  {row['n_mean']:>7.0f}  "
              f"{row['rf_mean']:>8.4f}  {row['e2e_mean']:>8.4f}  "
              f"{row['delta_mean']:>+8.4f}")
    corr = np.corrcoef(stats['n_mean'], stats['delta_mean'])[0, 1]
    print(f"\nΔAUROC 与 N_train 的 Pearson r = {corr:.3f}")
    if corr < -0.5:
        print("  → 强负相关：样本量越少，SPECTRA 优势越大（小样本设计有效性得证）")


if __name__ == '__main__':
    df, ratios = load_results()
    stats = compute_stats(df, ratios)
    print_summary(stats)
    plot(stats)
