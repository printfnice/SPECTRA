# plot_calibration.py
# 依赖: run_calibration.py 生成的 calibration_reliability.csv 和 calibration_table.csv
# 职责: 绘制校准图（可靠性图 + ECE 柱状图）

from pathlib import Path
import numpy as np
import pandas as pd

_PIPELINE_DIR = Path(__file__).parent
_CLASS_DIR = _PIPELINE_DIR.parent.parent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from pub_style import (
        set_pub_style,
        save_fig,
        FIGURES_DIR,
        DATA_DIR,
        style_axis,
        add_panel_label,
        add_reference_line,
        finalize_figure,
        NATURE_CATEGORICAL,
        ACCENT,
    )
    HAS_PUB = True
except Exception:
    HAS_PUB = False
    FIGURES_DIR = _CLASS_DIR / "output" / "figures"
    DATA_DIR = _CLASS_DIR / "output" / "data"

DATASETS = ["kuppe", "carraro", "habermann", "velmeshev", "reichart"]
DATASET_LABELS = {
    "kuppe": "Kuppe\n(MI)",
    "carraro": "Carraro\n(CF)",
    "habermann": "Habermann\n(PF)",
    "velmeshev": "Velmeshev\n(ASD)",
    "reichart": "Reichart\n(DCM)",
}


def plot_reliability_diagrams():
    rel_path = DATA_DIR / "calibration_reliability.csv"
    if not rel_path.exists():
        raise FileNotFoundError(f"Missing {rel_path}, run run_calibration.py first")

    df = pd.read_csv(rel_path)
    if HAS_PUB:
        set_pub_style()
    else:
        plt.rcParams.update({"font.family": "sans-serif", "font.size": 8})

    fig, axes = plt.subplots(1, 5, figsize=(7.8, 2.5), sharey=True)
    bar_color = '#8FC8DA'

    for idx, (ax, ds) in enumerate(zip(axes, DATASETS)):
        sub = df[df["dataset"] == ds]
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(DATASET_LABELS.get(ds, ds))
            continue

        avg = sub.groupby("bin_idx").agg(
            accuracy=("bin_accuracy", "mean"),
            accuracy_std=("bin_accuracy", "std"),
            confidence=("bin_center", "mean"),
        ).reset_index()

        ax.bar(avg["confidence"], avg["accuracy"], width=0.085, alpha=0.85, color=bar_color,
               edgecolor='white', linewidth=0.4, zorder=3)
        ax.errorbar(avg["confidence"], avg["accuracy"], yerr=avg["accuracy_std"].fillna(0),
                    fmt="none", color="#4A4A4A", capsize=2.5, linewidth=0.7, zorder=4)
        ax.plot([0, 1], [0, 1], "--", color="#9A9A9A", linewidth=0.9, zorder=2)

        style_axis(ax, grid_axis='y') if HAS_PUB else None
        if idx == 0 and HAS_PUB:
            add_panel_label(ax, 'a')
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=8.2, pad=5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Predicted prob.")
        if ds == DATASETS[0]:
            ax.set_ylabel("Observed freq.")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    axes[0].text(0.02, 0.96, 'Perfect calibration', transform=axes[0].transAxes,
                 ha='left', va='top', fontsize=6.8, color='#7A7A7A')
    finalize_figure(fig, left=0.06, right=0.99, top=0.90, bottom=0.24, wspace=0.10) if HAS_PUB else fig.tight_layout()

    out = FIGURES_DIR / "FigS_calibration_reliability"
    if HAS_PUB:
        save_fig(fig, out)
    else:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(out.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
        plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}.pdf")


def plot_ece_comparison():
    cal_path = DATA_DIR / "calibration_table.csv"
    if not cal_path.exists():
        raise FileNotFoundError(f"Missing {cal_path}, run run_calibration.py first")

    df = pd.read_csv(cal_path)
    if HAS_PUB:
        set_pub_style()
    else:
        plt.rcParams.update({"font.family": "sans-serif", "font.size": 8})

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    summary = df.groupby("dataset").agg(
        ece_mean=("ece", "mean"), ece_std=("ece", "std"),
        brier_mean=("brier", "mean"), brier_std=("brier", "std"),
    ).reindex(DATASETS)

    labels = [DATASET_LABELS.get(ds, ds) for ds in DATASETS]
    x = np.arange(len(DATASETS))
    w = 0.62

    ax = axes[0]
    bars = ax.bar(x, summary["ece_mean"], w, yerr=summary["ece_std"], capsize=3,
                  color=ACCENT if HAS_PUB else '#E05252', alpha=0.88, edgecolor='white', linewidth=0.5, zorder=3)
    if HAS_PUB:
        style_axis(ax, grid_axis='y')
        add_panel_label(ax, 'a')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ECE")
    ax.set_title("Calibration error by dataset", loc='left', pad=8)
    for bar, val in zip(bars, summary['ece_mean']):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(summary['ece_mean']) * 0.04, f'{val:.3f}', ha='center', va='bottom', fontsize=6.8)

    ax2 = axes[1]
    bars2 = ax2.bar(x, summary["brier_mean"], w, yerr=summary["brier_std"], capsize=3,
                    color='#7DB8D5', alpha=0.88, edgecolor='white', linewidth=0.5, zorder=3)
    if HAS_PUB:
        style_axis(ax2, grid_axis='y')
        add_panel_label(ax2, 'b')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Brier score")
    ax2.set_title("Brier score by dataset", loc='left', pad=8)
    for bar, val in zip(bars2, summary['brier_mean']):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + max(summary['brier_mean']) * 0.04, f'{val:.3f}', ha='center', va='bottom', fontsize=6.8)

    finalize_figure(fig, left=0.08, right=0.98, top=0.90, bottom=0.24, wspace=0.24) if HAS_PUB else fig.tight_layout()
    out = FIGURES_DIR / "FigS_calibration_metrics"
    if HAS_PUB:
        save_fig(fig, out)
    else:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(out.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
        plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}.pdf")


def main():
    print("Plotting reliability diagrams...")
    plot_reliability_diagrams()
    print("Plotting calibration metrics comparison...")
    plot_ece_comparison()
    print("Done.")


if __name__ == "__main__":
    main()
