# run_evi_fidelity.py
# 依赖: pandas, numpy, anndata, scikit-learn
# 职责: EVI 忠实性实验（Top-k LR 屏蔽 vs 随机 k 屏蔽）
# 说明: 使用 LIANA cellchat LR 矩阵 + 样本标签，评估屏蔽后 AUROC 下降幅度

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


DATASETS = ["kuppe", "carraro", "habermann", "velmeshev", "reichart"]
K_LIST_DEFAULT = [10, 20]

DATASET_KEYS = {
    "kuppe": {"sample_key": "sample", "condition_key": "patient_group"},
    "reichart": {"sample_key": "Sample", "condition_key": "disease"},
    "habermann": {"sample_key": "Sample_Name", "condition_key": "Status"},
    "velmeshev": {"sample_key": "sample", "condition_key": "diagnosis"},
    "carraro": {"sample_key": "orig.ident", "condition_key": "type"},
}


PIPELINE_DIR = Path(__file__).parent
CLASS_DIR = PIPELINE_DIR.parent.parent
DATA_DIR = CLASS_DIR / "data"
INTERIM_DIR = DATA_DIR / "interim"
TOP_DIR = CLASS_DIR / "output" / "data"
OUT_DIR = CLASS_DIR / "output" / "data"


def _h5ad_path(dataset: str) -> Path:
    cands = [
        INTERIM_DIR / f"{dataset}_processed.h5ad",
        INTERIM_DIR / f"{dataset}_filtered.h5ad",
        DATA_DIR / f"{dataset}_filtered.h5ad",
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f"No h5ad found for {dataset}: {cands}")


def load_sample_labels(dataset: str) -> pd.DataFrame:
    keys = DATASET_KEYS[dataset]
    p = _h5ad_path(dataset)
    adata = ad.read_h5ad(str(p), backed="r")
    obs = adata.obs[[keys["sample_key"], keys["condition_key"]]].copy()
    obs = obs.dropna().drop_duplicates(subset=[keys["sample_key"]])
    obs.columns = ["sample", "label"]
    obs["sample"] = obs["sample"].astype(str)
    obs["label"] = obs["label"].astype(str)
    return obs


def build_lr_matrix(dataset: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    keys = DATASET_KEYS[dataset]
    p = INTERIM_DIR / f"{dataset}_liana_scores" / "cellchat.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing parquet: {p}")

    cols = [
        keys["sample_key"],
        "source",
        "target",
        "ligand",
        "receptor",
        "ligand_complex",
        "receptor_complex",
        "lr_probs",
    ]
    df = pd.read_parquet(str(p), columns=cols)
    df = df.rename(columns={keys["sample_key"]: "sample"})
    df["sample"] = df["sample"].astype(str)
    # Use ligand__receptor (gene-level names) to match top_lr_pairs CSV format
    df["lr_simple"] = df["ligand"].astype(str) + "__" + df["receptor"].astype(str)
    df["feature"] = (
        df["source"].astype(str)
        + "_"
        + df["ligand_complex"].astype(str)
        + "__"
        + df["target"].astype(str)
        + "_"
        + df["receptor_complex"].astype(str)
    )

    # Build feature -> lr_simple mapping (handle duplicates by taking first)
    feat_map = (
        df[["feature", "lr_simple"]]
        .drop_duplicates()
        .groupby("feature")["lr_simple"]
        .first()
    )
    pivot = df.pivot_table(index="sample", columns="feature", values="lr_probs", aggfunc="mean", fill_value=0.0)

    label_df = load_sample_labels(dataset)
    merged = pivot.merge(label_df, left_index=True, right_on="sample", how="inner")
    merged = merged.set_index("sample")

    y = LabelEncoder().fit_transform(merged["label"].values)
    X = merged.drop(columns=["label"])
    lr_simple_per_feature = feat_map.reindex(X.columns).fillna("").values
    return X, y, lr_simple_per_feature


def cv_auroc(X: np.ndarray, y: np.ndarray, n_splits: int, n_repeats: int, n_trees: int) -> float:
    vals = []
    for rs in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=100 + rs)
        for tr, te in cv.split(X, y):
            clf = RandomForestClassifier(n_estimators=n_trees, random_state=1000 + rs, n_jobs=-1)
            clf.fit(X[tr], y[tr])
            prob = clf.predict_proba(X[te])
            if prob.shape[1] == 2:
                vals.append(float(roc_auc_score(y[te], prob[:, 1])))
            else:
                vals.append(float(roc_auc_score(y[te], prob, multi_class="ovr", average="macro")))
    return float(np.mean(vals))


def run_dataset(
    dataset: str,
    n_splits: int,
    n_repeats: int,
    n_trees: int,
    random_repeats: int,
    k_list: list[int],
) -> list[dict]:
    print(f"\n=== {dataset} ===")
    X_df, y, lr_simple = build_lr_matrix(dataset)
    X = X_df.values.astype(np.float32)
    all_lr = np.array(sorted(set(lr_simple.tolist())))
    all_lr = all_lr[all_lr != ""]

    # Compute top-k LR pairs from the SAME cellchat parquet by mean score
    # (not from E2E gate weights, which use a different LR set)
    keys = DATASET_KEYS[dataset]
    p = INTERIM_DIR / f"{dataset}_liana_scores" / "cellchat.parquet"
    score_df = pd.read_parquet(str(p), columns=["ligand", "receptor", "lr_probs"])
    score_df["lr_simple"] = score_df["ligand"].astype(str) + "__" + score_df["receptor"].astype(str)
    lr_mean_scores = score_df.groupby("lr_simple")["lr_probs"].mean().sort_values(ascending=False)
    top_lr = lr_mean_scores.index.tolist()
    print(f"Top-10 LR by score: {top_lr[:10]}")

    base = cv_auroc(X, y, n_splits=n_splits, n_repeats=n_repeats, n_trees=n_trees)
    print(f"base_auroc={base:.4f}")

    out_rows = []
    rng = np.random.default_rng(42)

    for k in k_list:
        topk = set(top_lr[:k])
        keep_top = np.array([s not in topk for s in lr_simple])
        X_top = X[:, keep_top]
        top_auroc = cv_auroc(X_top, y, n_splits=n_splits, n_repeats=n_repeats, n_trees=n_trees)
        drop_top = base - top_auroc

        rand_aurocs = []
        for _ in range(random_repeats):
            rand_set = set(rng.choice(all_lr, size=min(k, len(all_lr)), replace=False).tolist())
            keep_r = np.array([s not in rand_set for s in lr_simple])
            X_r = X[:, keep_r]
            rand_aurocs.append(cv_auroc(X_r, y, n_splits=n_splits, n_repeats=n_repeats, n_trees=n_trees))
        rand_auroc = float(np.mean(rand_aurocs))
        drop_rand = base - rand_auroc

        print(
            f"k={k:2d} | top={top_auroc:.4f} (drop {drop_top:+.4f}) | "
            f"rand={rand_auroc:.4f} (drop {drop_rand:+.4f})"
        )

        out_rows.append(
            {
                "dataset": dataset,
                "k": k,
                "auroc_base": round(base, 4),
                "auroc_topk_mask": round(top_auroc, 4),
                "drop_topk": round(drop_top, 4),
                "auroc_randomk_mask": round(rand_auroc, 4),
                "drop_randomk": round(drop_rand, 4),
                "delta_drop_top_vs_random": round(drop_top - drop_rand, 4),
                "n_features_base": int(X.shape[1]),
                "n_features_topk": int(X_top.shape[1]),
            }
        )
    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["all"] + DATASETS)
    ap.add_argument("--n_splits", type=int, default=3)
    ap.add_argument("--n_repeats", type=int, default=2)
    ap.add_argument("--n_trees", type=int, default=300)
    ap.add_argument("--random_repeats", type=int, default=3)
    ap.add_argument("--k_list", type=str, default="10,20",
                    help="Comma-separated mask sizes, e.g. 5,10,20,30")
    ap.add_argument("--out_name", type=str, default="evi_fidelity_table.csv")
    args = ap.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    k_list = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    if not k_list:
        k_list = K_LIST_DEFAULT
    rows = []
    for ds in datasets:
        rows.extend(
            run_dataset(
                ds,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                n_trees=args.n_trees,
                random_repeats=args.random_repeats,
                k_list=k_list,
            )
        )

    out_df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / args.out_name
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
