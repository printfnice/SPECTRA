#!/bin/bash
# 快速状态检查脚本

echo "=================================="
echo "  数据处理状态检查"
echo "=================================="
echo ""

cd "$(dirname "$0")"

# 检查处理进程
if [ -f processing.pid ]; then
    PID=$(cat processing.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✓ 处理进程运行中 (PID: $PID)"

        # 显示内存使用
        MEM=$(ps -p $PID -o rss= | awk '{print $1/1024/1024}')
        echo "  内存使用: ${MEM} GB"
    else
        echo "✗ 处理进程已停止"
    fi
else
    echo "○ 未找到处理进程"
fi

echo ""
echo "----------------------------------"

# 使用 Python 脚本检查详细状态
python3 auto_monitor.py status

echo ""
echo "----------------------------------"

# 显示最新日志
if [ -f processing.log ]; then
    echo "最新日志 (最后 10 行):"
    tail -10 processing.log
fi

echo ""
echo "=================================="
echo "查看完整日志: tail -f processing.log"
echo "手动检查状态: ./check_status.sh"
echo "=================================="
