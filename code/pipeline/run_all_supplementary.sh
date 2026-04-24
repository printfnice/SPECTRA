#!/bin/bash
# run_all_supplementary.sh
# 职责: 依次运行补充实验（EVI忠实性 + 学习曲线出图 + 校准指标）
# 用法:
#   cd notebooks/classification/code/pipeline
#   bash run_all_supplementary.sh
#
# 后台运行:
#   nohup bash run_all_supplementary.sh > /tmp/supp_exp.log 2>&1 &
#
# 查看进度: tail -f /tmp/supp_exp.log
# 日志: /tmp/supp_exp.log

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="/root/miniconda3/envs/LIANA/bin/python"
LOGFILE="/tmp/supp_exp.log"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

log() {
    echo "[$(timestamp)] $*" | tee -a "$LOGFILE"
}

run_step() {
    local name="$1"
    shift
    log ">>>>>> 开始: $name <<<<<<"
    log "命令: $*"
    if "$@" 2>&1 | tee -a "$LOGFILE"; then
        log "<<<<<< 完成: $name <<<<<<"
    else
        log "!!!!!!!! 失败: $name !!!!!!!!"
        log "继续执行后续步骤..."
    fi
    echo "" | tee -a "$LOGFILE"
}

log "=========================================="
log "  补充实验流水线启动"
log "  工作目录: $SCRIPT_DIR"
log "=========================================="
echo "" | tee -a "$LOGFILE"

# ──────────────────────────────────────────────
# Step 1: EVI 忠实性实验（修复后重跑5数据集）
# 预计时间: 10-30分钟 (RF-based, CPU)
# 输出: output/data/evi_fidelity_table.csv
# ──────────────────────────────────────────────
run_step "Step 1: EVI 忠实性实验 (5数据集)" \
    "$PYTHON" -u run_evi_fidelity.py --dataset all

# ──────────────────────────────────────────────
# Step 1b: EVI 忠实性出图
# 预计时间: <1分钟
# 输出: output/figures/FigS_evi_fidelity.pdf
# ──────────────────────────────────────────────
run_step "Step 1b: EVI 忠实性出图" \
    "$PYTHON" -u plot_evi_fidelity.py

# ──────────────────────────────────────────────
# Step 2: 学习曲线出图（数据已存在）
# 预计时间: <1分钟
# 输出: output/figures/FigS_learning_curve.pdf
# ──────────────────────────────────────────────
run_step "Step 2: 学习曲线出图" \
    "$PYTHON" -u plot_learning_curve.py

# ──────────────────────────────────────────────
# Step 3: 校准指标计算（从已有JSON读取，无需重训练）
# 预计时间: <1分钟 (CPU)
# 输出: output/data/calibration_table.csv
#        output/data/calibration_reliability.csv
# ──────────────────────────────────────────────
run_step "Step 3: 校准指标计算 (5数据集)" \
    "$PYTHON" -u run_calibration.py --dataset all

# ──────────────────────────────────────────────
# Step 3b: 校准指标出图
# 预计时间: <1分钟
# 输出: output/figures/FigS_calibration_reliability.pdf
#        output/figures/FigS_calibration_metrics.pdf
# ──────────────────────────────────────────────
run_step "Step 3b: 校准指标出图" \
    "$PYTHON" -u plot_calibration.py

log "=========================================="
log "  全部补充实验完成"
log "=========================================="
echo "" | tee -a "$LOGFILE"
log "输出文件汇总:"
log "  output/data/evi_fidelity_table.csv"
log "  output/figures/FigS_evi_fidelity.pdf"
log "  output/figures/FigS_learning_curve.pdf"
log "  output/data/calibration_table.csv"
log "  output/data/calibration_reliability.csv"
log "  output/figures/FigS_calibration_reliability.pdf"
log "  output/figures/FigS_calibration_metrics.pdf"
log ""
log "日志: $LOGFILE"
