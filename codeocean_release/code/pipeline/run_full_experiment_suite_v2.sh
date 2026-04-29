#!/bin/bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="/root/miniconda3/envs/LIANA/bin/python"
LOGFILE="/tmp/spectra_full_experiment_suite.log"
STATUS_FILE="$SCRIPT_DIR/output/full_suite_status.tsv"

mkdir -p "$SCRIPT_DIR/output"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

log() {
  echo "[$(timestamp)] $*" | tee -a "$LOGFILE"
}

mark() {
  # step	status	time
  echo -e "$1	$2	$(timestamp)" >> "$STATUS_FILE"
}

run_step() {
  local name="$1"
  shift
  log "===== START: $name ====="
  log "CMD: $*"
  mark "$name" "RUNNING"
  if "$@" 2>&1 | tee -a "$LOGFILE"; then
    log "===== DONE: $name ====="
    mark "$name" "DONE"
  else
    local rc=$?
    log "===== FAIL: $name (rc=$rc) ====="
    mark "$name" "FAIL"
  fi
  echo "" | tee -a "$LOGFILE"
}

: > "$STATUS_FILE"
log "=========================================="
log "SPECTRA full experiment suite started"
log "workdir: $SCRIPT_DIR"
log "python: $PYTHON"
log "=========================================="

run_step "A1_scale_ablation_v2_all"   "$PYTHON" -u run_scale_ablation_v2.py --dataset all --scale_mode all --force

run_step "A2_core_ablation_controls_all"   "$PYTHON" -u run_core_ablation_controls.py --dataset all --mode all --force

run_step "A3_unified_comparison_all_include_custom"   "$PYTHON" -u unified_comparison.py --dataset all --method all --include_custom

run_step "A4_collect_gate"   "$PYTHON" -u run_collect_gate.py

run_step "A5_evi_fidelity_all"   "$PYTHON" -u run_evi_fidelity.py --dataset all

run_step "A6_calibration_all"   "$PYTHON" -u run_calibration.py --dataset all

run_step "A7_bootstrap_ci"   "$PYTHON" -u plot_bootstrap_ci.py

run_step "A8_statistical_tests"   "$PYTHON" -u run_statistical_tests.py

run_step "A9_plot_unified_comparison"   "$PYTHON" -u plot_unified_comparison.py

run_step "A10_plot_gate_weights"   "$PYTHON" -u plot_gate_weights.py

run_step "A11_plot_gate_entropy"   "$PYTHON" -u plot_gate_entropy.py

run_step "A12_plot_evi_fidelity"   "$PYTHON" -u plot_evi_fidelity.py

run_step "A13_plot_calibration"   "$PYTHON" -u plot_calibration.py

run_step "A14_plot_bootstrap_ci"   "$PYTHON" -u plot_bootstrap_ci.py

run_step "A15_generate_tables"   "$PYTHON" -u generate_tables.py

log "=========================================="
log "SPECTRA full experiment suite finished"
log "status: $STATUS_FILE"
log "log: $LOGFILE"
log "=========================================="
