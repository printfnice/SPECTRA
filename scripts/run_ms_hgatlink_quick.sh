#!/bin/bash
# MS-HGATLink快速测试脚本 - reichart数据集
# 只运行MS-HGATLink方法，不运行其他LIANA方法

echo "=========================================="
echo "MS-HGATLink 快速测试"
echo "=========================================="
echo ""

# 激活环境
echo "[1/3] 激活LIANA环境..."
source /root/miniconda3/bin/activate LIANA

# 切换到项目目录
cd /root/autodl-tmp/lianaplus_manuscript-main/notebooks/classification

echo ""
echo "[2/3] 当前配置:"
echo "  数据集: reichart"
echo "  方法: ms_hgatlink (带预训练)"
echo "  降维: det"
echo "  速度: quick模式 (~10分钟)"
echo ""

# 确保配置正确
echo "[3/3] 运行MS-HGATLink..."
echo ""

python code/pipeline/simplified_pipeline_optimized.py 2>&1 | tee log/train/ms_hgatlink_quick_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "✓ 测试完成！"
echo "=========================================="
echo ""
echo "📊 结果文件:"
ls -lh output/*ms_hgatlink*.csv 2>/dev/null || echo "  (未找到结果文件)"
echo ""
echo "📝 日志文件:"
ls -lht log/train/ms_hgatlink*.log 2>/dev/null | head -1 || echo "  (未找到日志文件)"
echo ""
