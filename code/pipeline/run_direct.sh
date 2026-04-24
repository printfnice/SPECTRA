#!/bin/bash
#================================================================
# 直接运行每个数据集（最简单可靠的方法）
#================================================================

# 激活LIANA环境
source /root/miniconda3/bin/activate LIANA

cd /root/autodl-tmp/lianaplus_manuscript-main/notebooks/classification/code/pipeline

LOG_DIR="../../output/sota_run_logs"
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/master_direct_$(date +%Y%m%d_%H%M%S).log"

echo "=====================================================================" | tee "$MASTER_LOG"
echo "SOTA测试 - 直接运行法" | tee -a "$MASTER_LOG"
echo "开始时间: $(date)" | tee -a "$MASTER_LOG"
echo "=====================================================================" | tee -a "$MASTER_LOG"

# 数据集列表
DATASETS=("habermann" "velmeshev" "kuppe" "carraro" "reichart")

for DATASET in "${DATASETS[@]}"; do
    echo "" | tee -a "$MASTER_LOG"
    echo "###################################################################" | tee -a "$MASTER_LOG"
    echo "# 运行: $DATASET" | tee -a "$MASTER_LOG"
    echo "###################################################################" | tee -a "$MASTER_LOG"

    # 修改脚本
    sed "s/dataset_name = 'reichart'/dataset_name = '$DATASET'/" simplified_pipeline_optimized-Copy2.py > temp_$DATASET.py
    sed -i "s/liana_method = 'cellchat'/liana_method = 'hgatlink'/" temp_$DATASET.py

    # 运行
    DATASET_LOG="$LOG_DIR/${DATASET}_$(date +%H%M%S).log"
    python temp_$DATASET.py 2>&1 | tee "$DATASET_LOG" | tee -a "$MASTER_LOG"

    # 清理
    rm -f temp_$DATASET.py

    echo "" | tee -a "$MASTER_LOG"
    echo "✅ $DATASET 完成" | tee -a "$MASTER_LOG"
done

echo "" | tee -a "$MASTER_LOG"
echo "=====================================================================" | tee -a "$MASTER_LOG"
echo "🎉 所有数据集完成！" | tee -a "$MASTER_LOG"
echo "结束时间: $(date)" | tee -a "$MASTER_LOG"
echo "=====================================================================" | tee -a "$MASTER_LOG"
