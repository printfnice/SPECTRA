"""
Generate sample-level NATMI embeddings for t-SNE/UMAP comparison with SPECTRA.

Outputs:
  - output/data/{dataset}_natmi_embeddings.npz
  - output/data/natmi_embedding_summary.csv
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler


DATASETS = ["kuppe", "carraro", "habermann", "velmeshev", "reichart"]
DATASET_KEYS = {
    "kuppe": {"sample_key": "sample", "condition_key": "patient_group"},
    "carraro": {"sample_key": "orig.ident", "condition_key": "type"},
    "habermann": {"sample_key": "Sample_Name", "condition_key": "Status"},
    "velmeshev": {"sample_key": "sample", "condition_key": "diagnosis"},
    "reichart": {"sample_key": "Sample", "condition_key": "disease"},
}

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
INTERIM_DIR = DATA_DIR / "interim"
OUT_DATA_DIR = BASE_DIR / "output" / "data"


def _load_sample_label_map(dataset: str) -> dict:
    sample_key = DATASET_KEYS[dataset]["sample_key"]
    condition_key = DATASET_KEYS[dataset]["condition_key"]
    candidates = [
        INTERIM_DIR / f"{dataset}_processed.h5ad",
        INTERIM_DIR / f"{dataset}_filtered.h5ad",
    ]
    src = None
    for p in candidates:
        if p.exists():
            src = p
            break
    if src is None:
        raise FileNotFoundError(f"Missing h5ad for {dataset}: {candidates}")

    adata = ad.read_h5ad(src, backed="r")
    obs = adata.obs[[sample_key, condition_key]].drop_duplicates()
    mapping = obs.set_index(sample_key)[condition_key].to_dict()
    adata.file.close()
    return mapping


def _build_natmi_matrix(dataset: str):
    p = INTERIM_DIR / f"{dataset}_liana_scores" / "natmi.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing NATMI parquet: {p}")
    df = pd.read_parquet(p)
    sample_key = DATASET_KEYS[dataset]["sample_key"]
    req = {sample_key, "source", "ligand_complex", "target", "receptor_complex", "spec_weight"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"{dataset}: missing columns in NATMI parquet: {miss}")

    df = df[[sample_key, "source", "ligand_complex", "target", "receptor_complex", "spec_weight"]].copy()
    df = df.rename(columns={sample_key: "sample"})
    df["lr_pair"] = (
        df["source"].astype(str)
        + "_"
        + df["ligand_complex"].astype(str)
        + "__"
        + df["target"].astype(str)
        + "_"
        + df["receptor_complex"].astype(str)
    )
    pivot = df.pivot_table(index="sample", columns="lr_pair", values="spec_weight", aggfunc="mean", fill_value=0.0)
    return pivot


def _reduce_features(x: np.ndarray, max_pairs: int = 4000, pca_dim: int = 50):
    # Variance filter to keep a stable and comparable feature budget across cohorts.
    if x.shape[1] > max_pairs:
        var = np.var(x, axis=0)
        top_idx = np.argsort(var)[-max_pairs:]
        x = x[:, top_idx]
    x = StandardScaler(with_mean=True, with_std=True).fit_transform(x)
    n_comp = min(pca_dim, x.shape[0] - 1, x.shape[1])
    if n_comp >= 2:
        x = PCA(n_components=n_comp, random_state=42).fit_transform(x)
    return x


def main():
    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    print("\n" + "=" * 66)
    print("  NATMI embedding control generation (for Fig7 SPECTRA vs NATMI)")
    print("=" * 66)

    for ds in DATASETS:
        print(f"\n[{ds}] loading NATMI sample matrix...")
        pivot = _build_natmi_matrix(ds)
        label_map = _load_sample_label_map(ds)

        samples = [s for s in pivot.index.tolist() if s in label_map]
        if len(samples) < 4:
            raise RuntimeError(f"{ds}: insufficient matched samples ({len(samples)})")
        pivot = pivot.loc[samples]

        y_raw = [label_map[s] for s in samples]
        y = LabelEncoder().fit_transform(y_raw).astype(np.int64)
        x_raw = pivot.values.astype(np.float32)
        x_embed = _reduce_features(x_raw, max_pairs=4000, pca_dim=50).astype(np.float32)

        out_npz = OUT_DATA_DIR / f"{ds}_natmi_embeddings.npz"
        np.savez_compressed(
            out_npz,
            embeddings=x_embed,
            sample_ids=np.array(samples, dtype=object),
            labels=y,
            labels_raw=np.array(y_raw, dtype=object),
        )
        print(f"  saved: {out_npz}")

        rows.append(
            {
                "dataset": ds,
                "n_samples": int(len(samples)),
                "n_lr_pairs_raw": int(x_raw.shape[1]),
                "n_features_embed": int(x_embed.shape[1]),
            }
        )

    out_csv = OUT_DATA_DIR / "natmi_embedding_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n  saved: {out_csv}")
    print("  done.")


if __name__ == "__main__":
    main()
