#!/bin/bash
# 监控流程进度 - 更新版

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              实时流程进度监控                                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 检查进程是否在运行
if pgrep -f "continue_pipeline.py" > /dev/null; then
    echo "✓ 流程正在运行中..."

    # 显示进程信息
    ps aux | grep continue_pipeline.py | grep -v grep | awk '{printf "  PID: %s, CPU: %s%%, MEM: %s%%\n", $2, $3, $4}'
    echo ""
else
    echo "✗ 流程未运行或已完成"
    echo ""
fi

# 检查已生成的结果文件
echo "=== 已完成的数据集 ==="
for dataset in carraro reichart kuppe; do
    if [ -f "data/results/${dataset}.csv" ]; then
        size=$(ls -lh "data/results/${dataset}.csv" | awk '{print $5}')
        lines=$(wc -l < "data/results/${dataset}.csv")
        echo "  ✓ $dataset: $size, $lines 行"
    else
        echo "  ⏸ $dataset: 未完成"
    fi
done

echo ""
echo "=== 最新日志（最后 30 行）==="
tail -30 continue_pipeline.log 2>/dev/null || echo "  (日志文件为空或不存在)"

echo ""
echo "=== 命令 ==="
echo "  • 查看完整日志: cat continue_pipeline.log"
echo "  • 实时监控: tail -f continue_pipeline.log"
echo "  • 再次运行此脚本: bash monitor.sh"
echo "  • 检查进程: ps aux | grep continue_pipeline"
