#!/bin/bash
# overnight_runner.sh
# 等 Reichart ensemble 完成 → 跑其余4个数据集 ensemble → 跑 Reichart IRM
# 用法: bash overnight_runner.sh

PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="/tmp"
CONDA_RUN="conda run -n LIANA --no-capture-output python -u"

echo "=== Overnight Runner Started: $(date) ==="

# ---- 等待 Reichart Ensemble 完成 ----
REICHART_PID=13094
echo "[1] 等待 Reichart Ensemble (PID=$REICHART_PID) 完成..."
while kill -0 $REICHART_PID 2>/dev/null; do
    FOLD_LINE=$(tail -3 $LOG_DIR/r33_ensemble.log | grep "fold" | tail -1 || echo "")
    echo "  $(date +%H:%M) $FOLD_LINE"
    sleep 120
done
echo "[1] Reichart Ensemble 完成: $(date)"

# 打印最终结果
echo "--- Reichart Ensemble 最终结果 ---"
tail -15 $LOG_DIR/r33_ensemble.log | grep -E "(E2E|NMF|Ensemble|fold|\=\=\=)"

# ---- 并行跑小数据集 ensemble ----
echo "[2] 启动 kuppe+carraro+habermann ensemble (并行)..."
cd "$PIPELINE_DIR"
nohup $CONDA_RUN run_ensemble.py --dataset kuppe     > $LOG_DIR/ens_kuppe.log     2>&1 &
PID_KUPPE=$!
nohup $CONDA_RUN run_ensemble.py --dataset carraro   > $LOG_DIR/ens_carraro.log   2>&1 &
PID_CARRARO=$!
nohup $CONDA_RUN run_ensemble.py --dataset habermann > $LOG_DIR/ens_habermann.log 2>&1 &
PID_HABERMANN=$!

echo "  kuppe PID=$PID_KUPPE  carraro PID=$PID_CARRARO  habermann PID=$PID_HABERMANN"

# 等这三个完成
for PID in $PID_KUPPE $PID_CARRARO $PID_HABERMANN; do
    wait $PID 2>/dev/null || true
done
echo "[2] kuppe+carraro+habermann ensemble 完成: $(date)"

# ---- 跑 velmeshev ----
echo "[3] 启动 velmeshev ensemble..."
nohup $CONDA_RUN run_ensemble.py --dataset velmeshev > $LOG_DIR/ens_velmeshev.log 2>&1 &
PID_VEL=$!
echo "  velmeshev PID=$PID_VEL"

# ---- 跑 Reichart IRM（与 velmeshev 并行，IRM 使用 GPU）----
echo "[4] 启动 Reichart IRM..."
nohup $CONDA_RUN run_irm_reichart.py > $LOG_DIR/r35_irm.log 2>&1 &
PID_IRM=$!
echo "  IRM PID=$PID_IRM"

# 等 velmeshev + IRM 完成
wait $PID_VEL 2>/dev/null || true
wait $PID_IRM 2>/dev/null || true

echo "=== All experiments completed: $(date) ==="

# 汇总结果
echo ""
echo "=== RESULTS SUMMARY ==="
for DS in kuppe carraro habermann velmeshev; do
    CSV="/root/autodl-tmp/lianaplus_manuscript-main/output/${DS}_ensemble_e2e_nmf.csv"
    if [ -f "$CSV" ]; then
        echo "$DS: $(tail -1 $CSV)"
    else
        echo "$DS: NO RESULT"
    fi
done

REICH_CSV="/root/autodl-tmp/lianaplus_manuscript-main/output/reichart_ensemble_e2e_nmf.csv"
[ -f "$REICH_CSV" ] && echo "reichart_ensemble: $(tail -1 $REICH_CSV)" || echo "reichart_ensemble: NO RESULT"

IRM_CSV=/root/autodl-tmp/lianaplus_manuscript-main/output/reichart_irm_lambda1.0.csv
[ -f "$IRM_CSV" ] && echo "reichart_irm: $(tail -1 $IRM_CSV)" || echo "reichart_irm: NO RESULT"
