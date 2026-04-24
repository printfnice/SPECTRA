#!/bin/bash
# 简单的状态查看命令

cd "$(dirname "$0")"

echo "========================================"
echo "  数据处理状态快速查看"
echo "========================================"
echo ""

# 1. 检查进程
echo "【进程状态】"

# 检查大数据集处理进程
if pgrep -f "process_large_dataset.py" > /dev/null; then
    PID=$(pgrep -f "process_large_dataset.py")
    MEM=$(ps -p $PID -o rss= | awk '{printf "%.1f GB", $1/1024/1024}')
    CPU=$(ps -p $PID -o %cpu= | awk '{printf "%.1f%%", $1}')
    echo "  ✓ 大数据集处理运行中 (PID: $PID, 内存: $MEM, CPU: $CPU)"
elif pgrep -f "run_full_pipeline.py" > /dev/null; then
    PID=$(pgrep -f "run_full_pipeline.py")
    MEM=$(ps -p $PID -o rss= | awk '{printf "%.1f GB", $1/1024/1024}')
    CPU=$(ps -p $PID -o %cpu= | awk '{printf "%.1f%%", $1}')
    echo "  ✓ 处理进程运行中 (PID: $PID, 内存: $MEM, CPU: $CPU)"
else
    echo "  ✗ 处理进程未运行"
fi

if pgrep -f "auto_process_remaining.py" > /dev/null; then
    echo "  ✓ 自动处理控制器运行中"
elif pgrep -f "continuous_monitor.py" > /dev/null; then
    echo "  ✓ 持续监控运行中"
else
    echo "  ○ 监控未运行"
fi

echo ""

# 2. 数据集状态
echo "【数据集状态】"
for dataset in habermann reichart velmeshev; do
    if [ -f "data/results/${dataset}.csv" ]; then
        SIZE=$(ls -lh "data/results/${dataset}.csv" | awk '{print $5}')
        echo "  ✓ $dataset - 完成 ($SIZE)"
    elif [ -f "data/interim/${dataset}_processed.h5ad" ]; then
        echo "  ◐ $dataset - 预处理完成，正在分类..."
    elif [ -f "data/interim/${dataset}_basic.h5ad" ]; then
        echo "  ◑ $dataset - 基础处理完成，正在运行 LIANA..."
    elif [ -f "data/interim/${dataset}_filtered.h5ad" ]; then
        echo "  ◔ $dataset - 已过滤，正在预处理..."
    else
        echo "  ○ $dataset - 未处理"
    fi
done

echo ""

# 3. 最新日志
echo "【最新处理日志】"
if [ -f "auto_process_remaining.log" ]; then
    echo "  [自动处理日志]"
    tail -3 auto_process_remaining.log | sed 's/^/    /'
elif [ -f "processing.log" ]; then
    echo "  [常规处理日志]"
    tail -3 processing.log | sed 's/^/    /'
fi

# 显示当前处理的数据集详细日志
for dataset in reichart velmeshev; do
    if [ -f "large_dataset_${dataset}.log" ]; then
        echo ""
        echo "  [${dataset} 详细日志]"
        tail -3 "large_dataset_${dataset}.log" | sed 's/^/    /'
    fi
done

echo ""

# 4. 系统资源
echo "【系统资源】"
MEM_USED=$(free -h | awk '/^Mem:/ {print $3}')
MEM_TOTAL=$(free -h | awk '/^Mem:/ {print $2}')
MEM_PERCENT=$(free | awk '/^Mem:/ {printf "%.1f%%", $3/$2*100}')
echo "  内存: $MEM_USED / $MEM_TOTAL ($MEM_PERCENT)"

echo ""
echo "========================================"
echo "命令说明："
echo "  tail -f large_dataset_reichart.log  # 查看 reichart 实时日志"
echo "  tail -f large_dataset_velmeshev.log # 查看 velmeshev 实时日志"
echo "  tail -f auto_process_remaining.log  # 查看自动处理日志"
echo "  ./show_status.sh                    # 再次查看状态"
echo "========================================"
