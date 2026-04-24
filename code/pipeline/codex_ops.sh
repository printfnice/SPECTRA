#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIPE_DIR="$ROOT_DIR/code/pipeline"
OUT_DIR="$ROOT_DIR/output"
DOC_DIR="$ROOT_DIR/doc"

cmd="${1:-help}"

usage() {
  cat <<EOF
Usage: bash code/pipeline/codex_ops.sh <command>
Commands: status | snapshot | weekly | help
EOF
}

status() {
  echo "=== full suite status ==="
  test -f "$PIPE_DIR/output/full_suite_status.tsv" && cat "$PIPE_DIR/output/full_suite_status.tsv" || echo "(no status file)"
  echo
  echo "=== recent suite log ==="
  test -f /tmp/spectra_full_experiment_suite.log && tail -n 40 /tmp/spectra_full_experiment_suite.log || echo "(no suite log)"
  echo
  echo "=== running processes ==="
  ps -ef | grep -E "run_scale_ablation_v2.py|run_core_ablation_controls.py|unified_comparison.py|run_full_experiment_suite_v2.sh" | grep -v grep || true
}

snapshot() {
  for f in     "$OUT_DIR/data/sota_comparison_table.csv"     "$OUT_DIR/data/unified_comparison_table.csv"     "$OUT_DIR/data/e2e_vs_unsupervised_table.csv"     "$OUT_DIR/data/gate_weights_summary.csv"     "$OUT_DIR/data/evi_fidelity_table.csv"     "$OUT_DIR/data/bootstrap_ci_table.csv"     "$OUT_DIR/data/statistical_tests.csv"; do
    echo "--- $(basename "$f")"
    [ -f "$f" ] && head -n 6 "$f" || echo "(missing)"
  done
}

weekly() {
  echo "=== commits (7 days) ==="
  git -C "$ROOT_DIR" log --since="7 days ago" --oneline | head -n 100 || true
  echo
  echo "=== python files changed (mtime<=7d) ==="
  find "$ROOT_DIR/code" -name "*.py" -mtime -7 | sort
  echo
  echo "=== outputs (mtime<=7d) ==="
  find "$OUT_DIR" -type f -mtime -7 | sort | head -n 200
  echo
  echo "=== docs (mtime<=7d) ==="
  find "$DOC_DIR" -type f -mtime -7 | sort | head -n 200
}

case "$cmd" in
  status) status ;;
  snapshot) snapshot ;;
  weekly) weekly ;;
  help|*) usage ;;
esac
