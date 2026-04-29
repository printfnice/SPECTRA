# SPECTRA Code Ocean Release

This directory is a Code Ocean-specific release package for the `SPECTRA` manuscript.

## Purpose

This package is designed to regenerate manuscript-facing figures and tables from attached seed assets. It is not the full training workspace and does not attempt to rerun the complete experimental suite from raw cohort data.

## Layout

```text
code/
  model/
  process/
  utils/
  pipeline/
  test/
  run_manuscript_repro.py
data/
  README.md
  reference/cellchat_pathway_map.csv
results/
  README.md
environment/
  environment.yml
metadata/
  artifact_manifest.tsv
assets/
  Fig1_custom_architecture.png
run_capsule.sh
```

## Recommended Code Ocean setup

1. Create a new Capsule.
2. Import this directory from Git or upload it manually.
3. Use `environment/environment.yml` as the runtime environment.
4. Attach one Data Asset containing the seed manuscript outputs.
5. Mount that asset at:

```text
/data/spectra_output_seed
```

If you use a different mount path, set:

```bash
SPECTRA_SEED_OUTPUT_DIR=/data/<your-asset-name>
```

## Required Data Asset contents

The mounted seed asset should mirror the subset of the current research `output/` directory that is needed as plotting/table input.

Minimum structure:

```text
spectra_output_seed/
  data/
    *.csv
    *.json
    *.npz
  carraro_gate_weights.csv
  habermann_gate_weights.csv
  kuppe_gate_weights.csv
  reichart_gate_weights.csv
  velmeshev_gate_weights.csv
  unified_comparison_R30.csv
  pathway_enrichment_carraro.csv
  pathway_enrichment_habermann.csv
  pathway_enrichment_kuppe.csv
  pathway_enrichment_reichart.csv
  pathway_enrichment_velmeshev.csv
```

## Run

Code Ocean entrypoint:

```bash
bash run_capsule.sh
```

This run will:

1. Stage the seed output asset into local `output/`
2. Copy the static Fig.1 architecture image
3. Regenerate manuscript-facing figures and tables
4. Export selected artifacts into `/results`

## Notes

- Fig.1 is a manually authored architecture figure and is included here as a static asset.
- Pathway-scale plots depend on the bundled `cellchat_pathway_map.csv` reference file.
- The full experimental training pipeline is intentionally excluded from this first Code Ocean release because the publication target is manuscript-level reproducibility.
