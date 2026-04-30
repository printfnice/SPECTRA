# GitHub and Code Ocean Steps

## Part 1: Publish the Code Ocean package to GitHub

Recommended approach:

1. Create a new branch in your existing `SPECTRA` repository, for example:

```bash
git checkout -b codeocean-release
```

2. Copy or commit the contents of `codeocean_release/` into that branch.

3. Verify that the following are included:

- `code/`
- `assets/`
- `data/reference/`
- `environment/environment.yml`
- `metadata/artifact_manifest.tsv`
- `run_capsule.sh`
- `README.md`

4. Verify that the following are **not** included:

- `seed_assets/spectra_output_seed/`
- `seed_assets/*.tar.gz`
- local result exports
- `__pycache__/`

5. Commit and push:

```bash
git add codeocean_release
git commit -m "Add Code Ocean manuscript reproduction package"
git push origin codeocean-release
```

## Part 2: Upload the seed Data Asset to Code Ocean

Use this file:

- `codeocean_release/seed_assets/spectra_output_seed.tar.gz`

Recommended Data Asset name:

- `spectra_output_seed`

Recommended mount path:

```text
/data/spectra_output_seed
```

## Part 3: Create the Capsule in Code Ocean

1. Log in to Code Ocean.
2. Create a new Capsule.
3. Choose **Copy from Public Git**.
4. Paste the GitHub repository or branch URL containing the `codeocean_release` package.
5. Use `environment/environment.yml` for the runtime.

## Part 4: Attach the Data Asset

1. Open the Capsule.
2. Attach the `spectra_output_seed` Data Asset.
3. Confirm the mount path is:

```text
/data/spectra_output_seed
```

If a different mount path is used, set:

```bash
SPECTRA_SEED_OUTPUT_DIR=/data/<your-mounted-asset>
```

## Part 5: Run the Capsule

Run:

```bash
bash run_capsule.sh
```

What the run does:

1. stages the seed output asset into local `output/`
2. restores the static Fig.1 asset
3. regenerates manuscript-facing figures and tables
4. exports selected artifacts into `/results`

## Part 6: Check the results

After the run, verify `/results` contains:

- main figures
- supplementary figures
- table `.tex` and `.md` files
- `metadata/artifact_manifest.tsv`
- `run_log.txt`

## Part 7: Recommended manuscript wording

Once the Capsule is live, the paper can cite:

- GitHub repository for public code
- Code Ocean capsule for executable reproduction

If Code Ocean assigns a DOI later, use that DOI in the final manuscript version.
