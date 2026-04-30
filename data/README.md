# Data Notes

This release package does not bundle raw cohort data in Git.

## Default setup

Place `spectra_output_seed.tar.gz` inside:

```text
data/seed_assets/
```

The runtime will unpack that archive automatically during `bash run_capsule.sh`.

## Optional external seed directory

If you later attach a Code Ocean Data Asset or mount another folder, point:

```bash
SPECTRA_SEED_OUTPUT_DIR=/data/<your-seed-directory>
```

That external directory takes precedence over the packaged tarball.

## Bundled static reference

This directory includes:

- `reference/cellchat_pathway_map.csv`

That file is copied locally during the run so pathway visualizations do not require an additional raw-data asset.

## Optional future assets

For a heavier future capsule, you may also attach:

- raw cohort `.h5ad` files
- intermediate LIANA outputs
- split manifests

Those are not required for the current manuscript-reproduction capsule.
