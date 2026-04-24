#!/bin/bash
# run_after_main_suite.sh
# 职责: 等待主套件（run_all_supplementary.sh）完成后，依次运行追加实验
# 用法:
#   cd notebooks/classification/code/pipeline
#   nohup bash run_after_main_suite.sh > /tmp/after_main.log 2>&1 &
#
# 保证：轮询方式等待 PID 5172，绝不并发抢占 GPU

MAIN_PID=5172
LOGFILE="/tmp/after_main.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="/root/miniconda3/envs/LIANA/bin/python"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOGFILE"; }

log "=========================================="
log "  顺序实验守护脚本启动"
log "  等待主套件 PID=$MAIN_PID 完成..."
log "=========================================="

# 轮询等待主套件完成（每30秒检查一次）
while kill -0 "$MAIN_PID" 2>/dev/null; do
    log "  [等待] PID $MAIN_PID 仍在运行，30s 后再检查..."
    sleep 30
done

log "  主套件 PID=$MAIN_PID 已结束，开始追加实验"
log "=========================================="

# ----------------------------------------
# 追加实验 A: reichart Mixup=OFF
# 断点: output/mixup_ablation/reichart_mixup_off.csv
# ----------------------------------------
log "====== 开始: reichart-Mixup-OFF ======"
if "$PYTHON" -u run_reichart_mixup_off.py 2>&1 | tee -a "$LOGFILE"; then
    log "====== 完成: reichart-Mixup-OFF ======"
else
    log "====== 失败(继续): reichart-Mixup-OFF ======"
fi

# ----------------------------------------
# 追加实验 B: 修复版尺度消融 v2（全部5数据集 × 4模式）
# 断点: output/scale_ablation_v2/{dataset}_{mode}.csv
# ----------------------------------------
log "====== 开始: Scale-ablation-v2 ======"
if "$PYTHON" -u run_scale_ablation_v2.py --dataset all --scale_mode all 2>&1 | tee -a "$LOGFILE"; then
    log "====== 完成: Scale-ablation-v2 ======"
else
    log "====== 失败(继续): Scale-ablation-v2 ======"
fi

# ----------------------------------------
# 追加实验 C: 统计显著性检验（run_all_supplementary 已跑过，这里补跑一次含新数据）
# ----------------------------------------
log "====== 开始: Statistical-tests ======"
if "$PYTHON" -u run_statistical_tests.py --datasets all 2>&1 | tee -a "$LOGFILE"; then
    log "====== 完成: Statistical-tests ======"
else
    log "====== 失败(继续): Statistical-tests ======"
fi

# ----------------------------------------
# 追加实验 D: reichart SupCon 消融（5组对比，串行）
# 断点: output/supcon_ablation/reichart_{tag}.csv
# ----------------------------------------
log "====== 开始: reichart-SupCon-ablation ======"
if "$PYTHON" -u run_reichart_supcon.py 2>&1 | tee -a "$LOGFILE"; then
    log "====== 完成: reichart-SupCon-ablation ======"
else
    log "====== 失败(继续): reichart-SupCon-ablation ======"
fi

# ----------------------------------------
# 追加实验 E: reichart LIANA 基线（Tensor+RF，GroupKFold by donor_id）
# 断点: output/liana_baselines_5fold/reichart_{method}_5fold.csv
# ----------------------------------------
log "====== 开始: reichart-LIANA-baselines ======"
if "$PYTHON" -u run_reichart_liana_baselines.py 2>&1 | tee -a "$LOGFILE"; then
    log "====== 完成: reichart-LIANA-baselines ======"
else
    log "====== 失败(继续): reichart-LIANA-baselines ======"
fi

# ----------------------------------------
# 追加实验 F: reichart SupCon+Mixup=OFF 组合（2组：w=0.2/w=0.3，均关闭Mixup）
# 目标: 突破 CellChat 公平基线 0.9655
# 断点: output/supcon_ablation/reichart_{supcon_w02_mixoff,supcon_w03_mixoff}.csv
# ----------------------------------------
log "====== 开始: reichart-SupCon-MixOFF ======"
if "$PYTHON" -u run_reichart_supcon_mixoff.py 2>&1 | tee -a "$LOGFILE"; then
    log "====== 完成: reichart-SupCon-MixOFF ======"
else
    log "====== 失败(继续): reichart-SupCon-MixOFF ======"
fi

# ----------------------------------------
# 追加实验 G: SupCon w=0.2+MixOFF 全数据集验证（确认无退步，可写入统一配置）
# 断点: output/supcon_all_datasets/{dataset}_supcon_w02_mixoff.csv
# ----------------------------------------
log "====== 开始: SupCon-all-datasets ======"
if "$PYTHON" -u run_supcon_all_datasets.py --dataset all 2>&1 | tee -a "$LOGFILE"; then
    log "====== 完成: SupCon-all-datasets ======"
else
    log "====== 失败(继续): SupCon-all-datasets ======"
fi

log "=========================================="
log "  所有追加实验完成"
log "  结果:"
log "    output/mixup_ablation/reichart_mixup_off.csv"
log "    output/scale_ablation_v2/"
log "    output/statistical_tests/wilcoxon_results.csv"
log "    output/supcon_ablation/"
log "    output/liana_baselines_5fold/reichart_*_5fold.csv"
log "    output/supcon_ablation/reichart_supcon_{w02,w03}_mixoff.csv"
log "    output/supcon_all_datasets/summary_all.csv"
log "=========================================="
