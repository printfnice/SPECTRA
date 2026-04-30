#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[SPECTRA] Code Ocean manuscript reproduction run"
echo "[SPECTRA] Working directory: $ROOT_DIR"
echo "[SPECTRA] Seed source override: ${SPECTRA_SEED_OUTPUT_DIR:-<not set>}"
echo "[SPECTRA] Fallback packaged seed: $ROOT_DIR/data/seed_assets/spectra_output_seed.tar.gz"

"$PYTHON_BIN" code/run_manuscript_repro.py
