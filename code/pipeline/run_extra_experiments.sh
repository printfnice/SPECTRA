#!/bin/bash
# run_extra_experiments.sh
# 职责: 追加实验（在 run_all_supplementary.sh 完成后运行）
#   1. reichart Mixup=OFF 消融
#   2. 修复版尺度消融 v2（zero out 输入，彻底隔断残差路径）
# 用法:
#   cd notebooks/classification/code/pipeline
#   nohup bash run_extra_experiments.sh > /tmp/extra_experiments.log 2>&1 &

LOGFILE="/tmp/extra_experiments.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="/root/miniconda3/envs/LIANA/bin/python"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

log() {
    echo "[$(timestamp)] $*" | tee -a "$LOGFILE"
}

run_exp() {
    local name="$1"
    shift
    log "====== 开始: $name ======"
    log "命令: $*"
    if "$@" 2>&1 | tee -a "$LOGFILE"; then
        log "====== 完成: $name ======"
    else
        log "====== 失败(已跳过): $name ======"
    fi
}

log "=========================================="
log "  追加实验脚本启动"
log "  工作目录: $SCRIPT_DIR"
log "=========================================="

# ----------------------------------------
# Exp A: reichart Mixup=OFF
# 断点: output/mixup_ablation/reichart_mixup_off.csv
# ----------------------------------------
run_exp "reichart-Mixup-OFF" \
    "$PYTHON" -u run_reichart_mixup_off.py

# ----------------------------------------
# Exp B: 修复版尺度消融 v2（全部5数据集）
# 断点: output/scale_ablation_v2/{dataset}_{mode}.csv
# ----------------------------------------
run_exp "Scale-ablation-v2-all" \
    "$PYTHON" -u run_scale_ablation_v2.py --dataset all --scale_mode all

log "=========================================="
log "  追加实验全部完成"
log "  结果目录: output/mixup_ablation/reichart_mixup_off.csv"
log "            output/scale_ablation_v2/"
log "=========================================="
