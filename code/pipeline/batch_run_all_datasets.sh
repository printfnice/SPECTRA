#!/bin/bash
#================================================================
# 批量运行所有5个数据集以达到SOTA
# 使用简单版HGATLink + DET + 集成学习策略
#================================================================

set -e  # 遇到错误立即退出

# 日志目录
LOG_DIR="../../output/sota_run_logs"
mkdir -p "$LOG_DIR"

# 全局日志
MASTER_LOG="$LOG_DIR/master_$(date +%Y%m%d_%H%M%S).log"

echo "=====================================================================" | tee -a "$MASTER_LOG"
echo "SOTA测试批量运行 - 5个数据集" | tee -a "$MASTER_LOG"
echo "开始时间: $(date)" | tee -a "$MASTER_LOG"
echo "=====================================================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# 数据集列表（按优先级：困难数据集优先）
declare -a DATASETS=("habermann" "velmeshev" "kuppe" "carraro" "reichart")
declare -A TARGET_AUROC=(
    ["carraro"]="0.88"
    ["habermann"]="0.99"
    ["kuppe"]="1.00"
    ["reichart"]="1.00"
    ["velmeshev"]="0.77"
)

START_TIME=$(date +%s)

# 运行每个数据集
for DATASET in "${DATASETS[@]}"; do
    echo "" | tee -a "$MASTER_LOG"
    echo "###################################################################" | tee -a "$MASTER_LOG"
    echo "# 运行数据集: $DATASET" | tee -a "$MASTER_LOG"
    echo "# 目标AUROC: ${TARGET_AUROC[$DATASET]}" | tee -a "$MASTER_LOG"
    echo "###################################################################" | tee -a "$MASTER_LOG"

    DATASET_LOG="$LOG_DIR/${DATASET}_$(date +%H%M%S).log"
    DATASET_START=$(date +%s)

    # 修改脚本中的数据集名称并运行
    python -c "
import sys
from pathlib import Path

# 读取原始脚本
script = Path('simplified_pipeline_optimized-Copy2.py').read_text()

# 替换数据集名称
script = script.replace(
    \"dataset_name = 'reichart'\",
    \"dataset_name = '$DATASET'\"
)

# 确保使用正确的配置
script = script.replace(
    \"liana_method = 'cellchat'\",
    \"liana_method = 'hgatlink'\"
)
script = script.replace(
    \"speed_mode = 'rigorous'\",
    \"speed_mode = 'rigorous'\"
)
script = script.replace(
    \"enable_adaptive_config = True\",
    \"enable_adaptive_config = True\"
)
script = script.replace(
    \"enable_ensemble_learning = True\",
    \"enable_ensemble_learning = True\"
)

# 执行
exec(script)
" 2>&1 | tee "$DATASET_LOG" | tee -a "$MASTER_LOG"

    DATASET_END=$(date +%s)
    DATASET_ELAPSED=$((DATASET_END - DATASET_START))

    echo "" | tee -a "$MASTER_LOG"
    echo "✅ $DATASET 完成！用时: $((DATASET_ELAPSED / 60)) 分钟" | tee -a "$MASTER_LOG"
    echo "   日志: $DATASET_LOG" | tee -a "$MASTER_LOG"

    # 提取AUROC结果
    if grep -q "平均 AUROC" "$DATASET_LOG"; then
        AUROC_LINE=$(grep "平均 AUROC" "$DATASET_LOG" | tail -1)
        echo "   结果: $AUROC_LINE" | tee -a "$MASTER_LOG"
    fi

done

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

echo "" | tee -a "$MASTER_LOG"
echo "=====================================================================" | tee -a "$MASTER_LOG"
echo "🎉 所有数据集测试完成！" | tee -a "$MASTER_LOG"
echo "结束时间: $(date)" | tee -a "$MASTER_LOG"
echo "总用时: $((TOTAL_ELAPSED / 3600)) 小时 $((TOTAL_ELAPSED % 3600 / 60)) 分钟" | tee -a "$MASTER_LOG"
echo "=====================================================================" | tee -a "$MASTER_LOG"

# 生成汇总报告
python -c "
import pandas as pd
from pathlib import Path

log_dir = Path('$LOG_DIR')
print('\n生成汇总报告...')

results = []
for dataset in ['carraro', 'habermann', 'kuppe', 'reichart', 'velmeshev']:
    # 查找该数据集的CSV结果
    result_file = Path(f'../../output/{dataset}_hgatlink_det.csv')
    if result_file.exists():
        df = pd.read_csv(result_file)
        mean_auroc = df['auroc'].mean()
        std_auroc = df['auroc'].std()
        results.append({
            'dataset': dataset,
            'mean_auroc': mean_auroc,
            'std_auroc': std_auroc,
            'target_auroc': float('${TARGET_AUROC[$dataset]}'.replace('$dataset', dataset))
        })

if results:
    summary_df = pd.DataFrame(results)
    summary_file = log_dir / 'final_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f'✅ 汇总报告已保存: {summary_file}')
    print('\n结果预览:')
    print(summary_df.to_string(index=False))
" | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "📊 主日志: $MASTER_LOG" | tee -a "$MASTER_LOG"
