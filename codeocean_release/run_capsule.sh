#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[SPECTRA] Code Ocean manuscript reproduction run"
echo "[SPECTRA] Working directory: $ROOT_DIR"
echo "[SPECTRA] Seed asset: ${SPECTRA_SEED_OUTPUT_DIR:-/data/spectra_output_seed}"

"$PYTHON_BIN" code/run_manuscript_repro.py
