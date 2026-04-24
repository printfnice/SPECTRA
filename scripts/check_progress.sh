#!/bin/bash
#================================================================
# 检查SOTA测试进度
# 使用方法: bash check_progress.sh
#================================================================

LOG_DIR="code/pipeline/../../output/sota_run_logs"

echo "======================================================================"
echo "SOTA测试进度监控"
echo "======================================================================"
echo ""

# 检查主进程是否在运行
if pgrep -f "batch_run_all_datasets.sh" > /dev/null; then
    echo "✅ 后台任务正在运行中..."
else
    echo "⚠️ 后台任务未运行（可能已完成或出错）"
fi

echo ""

# 显示最新的主日志
if [ -d "$LOG_DIR" ]; then
    LATEST_MASTER_LOG=$(ls -t $LOG_DIR/master_*.log 2>/dev/null | head -1)

    if [ -n "$LATEST_MASTER_LOG" ]; then
        echo "📊 主日志: $LATEST_MASTER_LOG"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""

        # 显示最后50行
        echo "最新进度（最后50行）:"
        echo "----------------------------------------------------------------------"
        tail -50 "$LATEST_MASTER_LOG"
        echo "----------------------------------------------------------------------"

        echo ""
        echo "💡 查看完整日志: cat $LATEST_MASTER_LOG"
    else
        echo "⚠️  未找到主日志文件"
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # 检查各数据集日志
    echo "📁 数据集日志:"
    for DATASET in carraro habermann kuppe reichart velmeshev; do
        DATASET_LOG=$(ls -t $LOG_DIR/${DATASET}_*.log 2>/dev/null | head -1)

        if [ -n "$DATASET_LOG" ]; then
            # 检查是否有AUROC结果
            if grep -q "平均 AUROC" "$DATASET_LOG"; then
                AUROC=$(grep "平均 AUROC" "$DATASET_LOG" | tail -1 | awk '{print $3}')
                echo "  ✅ $DATASET: AUROC=$AUROC (已完成)"
            else
                # 显示最后一行进度
                LAST_LINE=$(tail -1 "$DATASET_LOG")
                echo "  🔄 $DATASET: 运行中... ($LAST_LINE)"
            fi
        else
            echo "  ⏳ $DATASET: 未开始"
        fi
    done

else
    echo "⚠️  日志目录不存在: $LOG_DIR"
fi

echo ""
echo "======================================================================"
echo "🔄 自动刷新: watch -n 30 bash check_progress.sh"
echo "📊 查看汇总: cat $LOG_DIR/final_summary.csv"
echo "======================================================================"
