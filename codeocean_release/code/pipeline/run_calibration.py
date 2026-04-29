# run_calibration.py
# 依赖: json, numpy, pandas, scikit-learn
# 被依赖: plot_calibration.py
# 职责: 从 E2E metrics JSON 文件中计算校准指标（ECE, Brier Score, MCE）
# 说明: 使用已有的 y_true/y_prob 数据，无需重新训练

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

PIPELINE_DIR = Path(__file__).parent
CLASS_DIR = PIPELINE_DIR.parent.parent
DATA_DIR = CLASS_DIR / "output" / "data"

DATASETS = ["kuppe", "carraro", "habermann", "velmeshev", "reichart"]
N_BINS = 10


def compute_ece(y_true, y_prob, n_bins=N_BINS):
    """Expected Calibration Error (binary classification)."""
    if len(np.unique(y_true)) > 2:
        return _ece_multiclass(y_true, y_prob, n_bins)
    pos_prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (pos_prob > bin_edges[i]) & (pos_prob <= bin_edges[i + 1])
        if i == 0:
            mask = (pos_prob >= bin_edges[i]) & (pos_prob <= bin_edges[i + 1])
        b = mask.sum()
        if b == 0:
            continue
        avg_conf = pos_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += (b / n) * abs(avg_acc - avg_conf)
    return float(ece)


def _ece_multiclass(y_true, y_prob, n_bins):
    """ECE for multiclass: use max probability as confidence."""
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (conf > bin_edges[i]) & (conf <= bin_edges[i + 1])
        if i == 0:
            mask = (conf >= bin_edges[i]) & (conf <= bin_edges[i + 1])
        b = mask.sum()
        if b == 0:
            continue
        ece += (b / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def compute_brier(y_true, y_prob):
    """Brier Score (binary: mean((p - y)^2); multiclass: mean sum((p_k - I_k)^2))."""
    if y_prob.ndim == 1 or y_prob.shape[1] == 2:
        pos_prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
        return float(np.mean((pos_prob - y_true) ** 2))
    n_classes = y_prob.shape[1]
    one_hot = np.zeros_like(y_prob)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def compute_mce(y_true, y_prob, n_bins=N_BINS):
    """Maximum Calibration Error."""
    if len(np.unique(y_true)) > 2:
        return _mce_multiclass(y_true, y_prob, n_bins)
    pos_prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    max_err = 0.0
    for i in range(n_bins):
        mask = (pos_prob > bin_edges[i]) & (pos_prob <= bin_edges[i + 1])
        if i == 0:
            mask = (pos_prob >= bin_edges[i]) & (pos_prob <= bin_edges[i + 1])
        b = mask.sum()
        if b == 0:
            continue
        max_err = max(max_err, abs(y_true[mask].mean() - pos_prob[mask].mean()))
    return float(max_err)


def _mce_multiclass(y_true, y_prob, n_bins):
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    max_err = 0.0
    for i in range(n_bins):
        mask = (conf > bin_edges[i]) & (conf <= bin_edges[i + 1])
        if i == 0:
            mask = (conf >= bin_edges[i]) & (conf <= bin_edges[i + 1])
        b = mask.sum()
        if b == 0:
            continue
        max_err = max(max_err, abs(correct[mask].mean() - conf[mask].mean()))
    return float(max_err)


def compute_reliability_data(y_true, y_prob, n_bins=N_BINS):
    """Compute per-bin accuracy and confidence for reliability diagram."""
    if len(np.unique(y_true)) > 2:
        return _reliability_multiclass(y_true, y_prob, n_bins)
    pos_prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accuracy = []
    bin_confidence = []
    bin_counts = []
    for i in range(n_bins):
        mask = (pos_prob > bin_edges[i]) & (pos_prob <= bin_edges[i + 1])
        if i == 0:
            mask = (pos_prob >= bin_edges[i]) & (pos_prob <= bin_edges[i + 1])
        b = mask.sum()
        if b == 0:
            bin_accuracy.append(0.0)
            bin_confidence.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_counts.append(0)
        else:
            bin_accuracy.append(float(y_true[mask].mean()))
            bin_confidence.append(float(pos_prob[mask].mean()))
            bin_counts.append(int(b))
    return {
        "bin_edges": bin_edges.tolist(),
        "bin_accuracy": bin_accuracy,
        "bin_confidence": bin_confidence,
        "bin_counts": bin_counts,
    }


def _reliability_multiclass(y_true, y_prob, n_bins):
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accuracy = []
    bin_confidence = []
    bin_counts = []
    for i in range(n_bins):
        mask = (conf > bin_edges[i]) & (conf <= bin_edges[i + 1])
        if i == 0:
            mask = (conf >= bin_edges[i]) & (conf <= bin_edges[i + 1])
        b = mask.sum()
        if b == 0:
            bin_accuracy.append(0.0)
            bin_confidence.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_counts.append(0)
        else:
            bin_accuracy.append(float(correct[mask].mean()))
            bin_confidence.append(float(conf[mask].mean()))
            bin_counts.append(int(b))
    return {
        "bin_edges": bin_edges.tolist(),
        "bin_accuracy": bin_accuracy,
        "bin_confidence": bin_confidence,
        "bin_counts": bin_counts,
    }


def process_dataset(dataset: str) -> list[dict]:
    json_path = DATA_DIR / f"{dataset}_spectra_e2e_metrics.json"
    if not json_path.exists():
        print(f"  [SKIP] {dataset}: metrics JSON not found at {json_path}")
        return []

    with open(json_path) as f:
        data = json.load(f)

    fold_metrics = data.get("fold_metrics", [])
    if not fold_metrics:
        print(f"  [SKIP] {dataset}: no fold_metrics in JSON")
        return []

    rows = []
    for idx, fm in enumerate(fold_metrics):
        y_true = np.array(fm.get("y_true", []))
        y_prob = np.array(fm.get("y_prob", []))
        if len(y_true) == 0 or len(y_prob) == 0:
            continue

        n_classes = len(np.unique(y_true))
        ece = compute_ece(y_true, y_prob)
        brier = compute_brier(y_true, y_prob)
        mce = compute_mce(y_true, y_prob)
        reliability = compute_reliability_data(y_true, y_prob)

        rows.append({
            "dataset": dataset,
            "fold_idx": idx,
            "n_samples": len(y_true),
            "n_classes": n_classes,
            "ece": round(ece, 6),
            "brier": round(brier, 6),
            "mce": round(mce, 6),
            "reliability": json.dumps(reliability),
        })

    print(f"  [OK] {dataset}: {len(rows)} folds processed")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["all"] + DATASETS)
    args = ap.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    all_rows = []
    for ds in datasets:
        all_rows.extend(process_dataset(ds))

    if not all_rows:
        print("No calibration data generated.")
        return

    df = pd.DataFrame(all_rows)
    out_csv = DATA_DIR / "calibration_table.csv"
    df.drop(columns=["reliability"]).to_csv(out_csv, index=False)
    print(f"\nCalibration table saved: {out_csv}")

    # Summary
    print("\n=== Calibration Summary (per dataset, mean +/- std) ===")
    print(f"{'Dataset':<12} {'ECE':>8} {'Brier':>8} {'MCE':>8} {'N_folds':>8}")
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            continue
        print(
            f"{ds:<12} {sub['ece'].mean():>8.4f} +/- {sub['ece'].std():.4f}  "
            f"{sub['brier'].mean():>8.4f} +/- {sub['brier'].std():.4f}  "
            f"{sub['mce'].mean():>8.4f} +/- {sub['mce'].std():.4f}  "
            f"{len(sub):>8}"
        )

    # Save reliability data separately for plotting
    rel_rows = []
    for _, row in df.iterrows():
        rel = json.loads(row["reliability"])
        for b in range(len(rel["bin_accuracy"])):
            rel_rows.append({
                "dataset": row["dataset"],
                "fold_idx": row["fold_idx"],
                "bin_idx": b,
                "bin_center": rel["bin_confidence"][b],
                "bin_accuracy": rel["bin_accuracy"][b],
                "bin_count": rel["bin_counts"][b],
            })
    rel_df = pd.DataFrame(rel_rows)
    rel_csv = DATA_DIR / "calibration_reliability.csv"
    rel_df.to_csv(rel_csv, index=False)
    print(f"Reliability data saved: {rel_csv}")


if __name__ == "__main__":
    main()
