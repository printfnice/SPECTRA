#!/bin/bash
# 从checkpoint继续运行MS-HGATLink测试
# 使用tensor的CV外降维模式

echo "=========================================="
echo "MS-HGATLink 继续运行（从step3）"
echo "=========================================="
echo ""

# 激活环境
echo "[1/2] 激活LIANA环境..."
source /root/miniconda3/bin/activate LIANA

# 切换到项目目录
cd /root/autodl-tmp/lianaplus_manuscript-main/notebooks/classification

echo ""
echo "[2/2] 当前配置:"
echo "  ✅ 断点续传: 已启用（从step3继续）"
echo "  ✅ 降维模式: CV外降维（tensor方法）"
echo "  数据集: reichart"
echo "  方法: ms_hgatlink"
echo "  降维: tensor"
echo ""

echo "开始运行..."
echo ""

python code/pipeline/simplified_pipeline_optimized.py 2>&1 | tee log/train/ms_hgatlink_resume_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "✓ 运行完成！"
echo "=========================================="
echo ""
echo "📊 结果文件:"
ls -lh output/*ms_hgatlink*.csv 2>/dev/null || echo "  (未找到结果文件)"
echo ""
