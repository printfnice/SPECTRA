# plot_tsne_umap.py
# Main + supplementary latent-space figures:
#   Fig7_tsne_umap.pdf/png           -> Reichart main figure
#   FigS_tsne_umap_velmeshev.pdf/png -> Velmeshev supplementary figure

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(_PIPELINE_DIR))

from pub_style import (
    DATA_DIR,
    FIGURES_DIR,
    OUTPUT_DIR,
    add_panel_label,
    finalize_figure,
    save_fig,
    set_pub_style,
)

DATASET_LABELS = {
    "kuppe": "Kuppe (MI)",
    "carraro": "Carraro (CF)",
    "habermann": "Habermann (PF)",
    "velmeshev": "Velmeshev (ASD)",
    "reichart": "Reichart (DCM)",
}
METHODS = [("SPECTRA", "e2e"), ("NATMI", "natmi")]
CLASS_COLORS = ["#4E79A7", "#E15759"]
FIGURE_SPECS = [
    ("reichart", "Fig7_tsne_umap"),
    ("velmeshev", "FigS_tsne_umap_velmeshev"),
]


def _run_tsne(x, perplexity=15, seed=42):
    from sklearn.manifold import TSNE

    n = len(x)
    perp = min(perplexity, max(2, n // 3))
    return TSNE(n_components=2, perplexity=perp, random_state=seed, max_iter=1200, init="pca").fit_transform(x)


def _run_umap(x, n_neighbors=10, seed=42):
    import umap

    n = len(x)
    nn = min(n_neighbors, max(2, n - 1))
    return umap.UMAP(n_components=2, n_neighbors=nn, random_state=seed).fit_transform(x)


def _silhouette(coords, labels):
    from sklearn.metrics import silhouette_score

    y = np.asarray(labels)
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(silhouette_score(coords, y))
    except Exception:
        return np.nan


def _normalize_square(coords):
    c = np.asarray(coords, dtype=float)
    c = c - c.mean(axis=0, keepdims=True)
    s = np.max(np.abs(c))
    if s <= 0:
        return c
    return c / s


def _load_embeddings(ds, tag):
    p = OUTPUT_DIR / "data" / f"{ds}_{tag}_embeddings.npz"
    if not p.exists():
        return None
    d = np.load(str(p), allow_pickle=True)
    return d["embeddings"], d["sample_ids"].astype(str), d["labels"].astype(int)


def _label_name(ds, y):
    m = {
        "kuppe": {0: "ischemic", 1: "myogenic"},
        "carraro": {0: "control", 1: "CF"},
        "habermann": {0: "control", 1: "ILD"},
        "velmeshev": {0: "TD", 1: "ASD"},
        "reichart": {0: "DCM", 1: "healthy"},
    }
    return m.get(ds, {}).get(int(y), f"class {int(y)}")


def _make_figure(dataset, out_name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_pub_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.6))
    rows = []

    cache = {}
    for method_name, tag in METHODS:
        loaded = _load_embeddings(dataset, tag)
        if loaded is None:
            cache[method_name] = None
            continue
        x, _, y = loaded
        ts = _run_tsne(x)
        um = _run_umap(x)
        cache[method_name] = {
            "y": y,
            "tsne": ts,
            "umap": um,
            "sil_ts": _silhouette(ts, y),
            "sil_um": _silhouette(um, y),
            "n": len(y),
            "pos_ratio": float(np.mean(y == 1)),
        }

    panel_map = [
        (0, 0, "SPECTRA", "tsne", "a"),
        (0, 1, "NATMI", "tsne", "b"),
        (1, 0, "SPECTRA", "umap", "c"),
        (1, 1, "NATMI", "umap", "d"),
    ]
    for r, c, method_name, view, plabel in panel_map:
        ax = axes[r, c]
        data = cache.get(method_name)
        if data is None:
            ax.text(0.5, 0.5, f"{method_name}\nNo embedding", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            add_panel_label(ax, plabel)
            continue

        coords = _normalize_square(data[view])
        y = data["y"]
        sil = data["sil_ts"] if view == "tsne" else data["sil_um"]
        for cls in sorted(np.unique(y)):
            idx = np.where(y == cls)[0]
            ax.scatter(
                coords[idx, 0],
                coords[idx, 1],
                s=26,
                alpha=0.88,
                color=CLASS_COLORS[int(cls) % len(CLASS_COLORS)],
                edgecolors="white",
                linewidths=0.4,
                label=_label_name(dataset, cls) if (r == 0 and c == 0) else None,
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(
            f"{method_name} {view.upper()}",
            fontsize=10.0,
            pad=4,
        )
        add_panel_label(ax, plabel)

        rows.append(
            {
                "dataset": dataset,
                "method": method_name,
                "view": view,
                "N": data["n"],
                "pos_ratio": round(data["pos_ratio"], 4),
                "silhouette": round(float(sil), 4),
            }
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        legend_handles, legend_labels = [], []
        for h, l in zip(handles, labels):
            if l not in legend_labels:
                legend_handles.append(h)
                legend_labels.append(l)
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(4, max(2, len(legend_labels))),
            frameon=False,
            bbox_to_anchor=(0.5, 0.985),
        )

    finalize_figure(fig, left=0.07, right=0.985, top=0.92, bottom=0.06, wspace=0.12, hspace=0.20)
    save_fig(fig, FIGURES_DIR / out_name)
    plt.close(fig)
    return rows


def main():
    print("\n" + "=" * 66)
    print("  Latent-space redraw: Reichart main + Velmeshev supplementary")
    print("=" * 66)

    rows = []
    for dataset, out_name in FIGURE_SPECS:
        rows.extend(_make_figure(dataset, out_name))

    out_csv = DATA_DIR / "tsne_umap_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("  saved: output/figures/Fig7_tsne_umap.pdf/png")
    print("  saved: output/figures/FigS_tsne_umap_velmeshev.pdf/png")
    print(f"  saved: {out_csv}")


if __name__ == "__main__":
    main()
