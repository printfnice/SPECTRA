#!/bin/bash
# 阶段A快速优化测试
# 预期AUROC: 0.896 → 0.96+ (+2.2-3.0个百分点)

echo "=========================================="
echo "阶段A快速优化测试"
echo "=========================================="
echo ""
echo "优化清单:"
echo "  1. n_factors: 5 → 15 (+1.5-2.0 AUROC)"
echo "  2. max_lr_pairs: 200 → 300 (+0.3-0.5 AUROC)"
echo "  3. RandomForest优化: 200树→500树 (+0.5-1.0 AUROC)"
echo ""
echo "预期总提升: +2.2-3.0个百分点"
echo "预期结果: AUROC > 0.96"
echo ""

# 激活环境
echo "[1/2] 激活LIANA环境..."
source /root/miniconda3/bin/activate LIANA

# 切换到项目目录
cd /root/autodl-tmp/lianaplus_manuscript-main/notebooks/classification

echo ""
echo "[2/2] 开始运行优化测试..."
echo "  - 强制重新运行步骤2（MS-HGATLink）"
echo "  - 使用tensor降维（15因子）"
echo "  - 增强随机森林（500树）"
echo ""

python code/pipeline/simplified_pipeline_optimized.py 2>&1 | tee log/train/ms_hgatlink_optimized_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "✓ 测试完成！"
echo "=========================================="
echo ""
echo "📊 结果文件:"
ls -lh code/pipeline/output/*ms_hgatlink*.csv 2>/dev/null | tail -1
echo ""
echo "查看最新结果:"
echo "cat code/pipeline/output/reichart_ms_hgatlink_tensor.csv"
