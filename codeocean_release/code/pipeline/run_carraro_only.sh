#!/bin/bash
# SOTA测试 - 仅运行carraro数据集
# 创建时间: 2026-02-03

set -e

echo "====================================================================="
echo "SOTA测试 - 仅运行carraro"
echo "开始时间: $(date)"
echo "====================================================================="
echo ""

DATASET="carraro"
TARGET_AUROC="0.88"
OUTPUT_DIR="../../output"
LOG_DIR="$OUTPUT_DIR/sota_run_logs"

mkdir -p "$LOG_DIR"

echo "###################################################################"
echo "# 运行: $DATASET (目标AUROC: $TARGET_AUROC)"
echo "###################################################################"

START_TIME=$(date +%s)

# 复制模板并运行
cp template_direct.py "temp_${DATASET}.py"
sed -i "s/DATASET_NAME = \".*\"/DATASET_NAME = \"$DATASET\"/" "temp_${DATASET}.py"

# 运行Python脚本，捕获输出和错误
LOG_FILE="${LOG_DIR}/${DATASET}_$(date +%H%M%S).log"
if python "temp_${DATASET}.py" > "$LOG_FILE" 2>&1; then
    END_TIME=$(date +%s)
    DURATION=$(( (END_TIME - START_TIME) / 60 ))
    echo "✅ $DATASET 成功完成！用时=${DURATION}分钟"

    # 检查结果文件
    RESULT_FILE="output/${DATASET}_hgatlink_det.csv"
    if [ -f "$RESULT_FILE" ]; then
        echo "📊 结果文件: $RESULT_FILE"
        echo "文件大小: $(ls -lh "$RESULT_FILE" | awk '{print $5}')"
    else
        echo "⚠️  警告: 未找到结果文件 $RESULT_FILE"
    fi
else
    END_TIME=$(date +%s)
    DURATION=$(( (END_TIME - START_TIME) / 60 ))
    echo "❌ $DATASET 失败！用时=${DURATION}分钟"
    echo "查看日志: $LOG_FILE"
    exit 1
fi

# 清理临时文件
rm -f "temp_${DATASET}.py"

echo ""
echo "====================================================================="
echo "SOTA测试完成！"
echo "结束时间: $(date)"
echo "====================================================================="
