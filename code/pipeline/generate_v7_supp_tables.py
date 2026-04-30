from pathlib import Path

import pandas as pd


PIPELINE_DIR = Path(__file__).parent
CLASS_DIR = PIPELINE_DIR.parent.parent
DATA_DIR = CLASS_DIR / "output" / "data"
PIPELINE_OUT = PIPELINE_DIR / "output"


def build_vib_support_table() -> pd.DataFrame:
    src = PIPELINE_OUT / "core_ablation" / "summary_core_ablation.csv"
    df = pd.read_csv(src)

    full = (
        df[df["mode"] == "full"][["dataset", "mean_auroc", "std_auroc"]]
        .rename(columns={"mean_auroc": "full_auroc", "std_auroc": "full_std"})
    )
    wo_vib = (
        df[df["mode"] == "wo_vib"][["dataset", "mean_auroc", "std_auroc"]]
        .rename(columns={"mean_auroc": "wo_vib_auroc", "std_auroc": "wo_vib_std"})
    )
    out = full.merge(wo_vib, on="dataset", how="inner")
    out["delta_mean_full_minus_wo_vib"] = out["full_auroc"] - out["wo_vib_auroc"]
    out["delta_std_full_minus_wo_vib"] = out["full_std"] - out["wo_vib_std"]
    order = ["kuppe", "carraro", "habermann", "velmeshev", "reichart"]
    out["dataset"] = pd.Categorical(out["dataset"], categories=order, ordered=True)
    return out.sort_values("dataset").reset_index(drop=True)


def build_pathway_top5_table() -> pd.DataFrame:
    src = DATA_DIR / "pathway_enrichment_with_q.csv"
    df = pd.read_csv(src)
    df = df.sort_values(["dataset", "qvalue_bh", "pvalue", "pathway"], ascending=[True, True, True, True])
    top5 = df.groupby("dataset", as_index=False, group_keys=False).head(5).copy()
    order = ["kuppe", "carraro", "habermann", "velmeshev", "reichart"]
    top5["dataset"] = pd.Categorical(top5["dataset"], categories=order, ordered=True)
    return top5.sort_values(["dataset", "qvalue_bh", "pathway"]).reset_index(drop=True)


def main() -> None:
    vib = build_vib_support_table()
    vib_path = DATA_DIR / "v7_vib_support_table.csv"
    vib.to_csv(vib_path, index=False)
    print(f"saved: {vib_path}")

    pathway = build_pathway_top5_table()
    pathway_path = DATA_DIR / "v7_pathway_top5_table.csv"
    pathway.to_csv(pathway_path, index=False)
    print(f"saved: {pathway_path}")


if __name__ == "__main__":
    main()
