#!/bin/bash
# 完整优化测试（清除checkpoint，从头运行）
# 应用所有3个优化

echo "=========================================="
echo "完整优化测试（清除checkpoint）"
echo "=========================================="
echo ""
echo "即将删除checkpoint并重新运行，确保所有优化生效"
echo ""

# 激活环境
source /root/miniconda3/bin/activate LIANA

# 切换到项目目录
cd /root/autodl-tmp/lianaplus_manuscript-main/notebooks/classification

echo "删除旧checkpoint..."
rm -f ../../data/results/checkpoints/reichart_ms_hgatlink_tensor_*.pkl
echo "✓ Checkpoint已清除"
echo ""

echo "开始完整测试（包含所有优化）..."
echo "  1. max_lr_pairs: 200 → 300"
echo "  2. n_factors: 5 → 15"
echo "  3. RandomForest: 200树 → 500树（优化参数）"
echo ""
echo "预计耗时: 60-90分钟"
echo ""

# 临时禁用force_rerun_step
sed -i 's/force_rerun_step = 2/force_rerun_step = None/' code/pipeline/simplified_pipeline_optimized.py

python code/pipeline/simplified_pipeline_optimized.py 2>&1 | tee log/train/ms_hgatlink_full_optimized_$(date +%Y%m%d_%H%M%S).log

# 恢复force_rerun_step
sed -i 's/force_rerun_step = None/force_rerun_step = 2/' code/pipeline/simplified_pipeline_optimized.py

echo ""
echo "=========================================="
echo "✓ 完整测试完成！"
echo "=========================================="
