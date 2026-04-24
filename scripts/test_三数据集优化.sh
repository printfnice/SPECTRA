#!/bin/bash
# 三数据集自动优化测试
# habermann, velmeshev, carraro

echo "================================================================"
echo "           三数据集性能优化测试"
echo "================================================================"
echo ""
echo "目标:"
echo "  habermann: 0.8642 → ≥0.87 (+0.6%)"
echo "  velmeshev: 0.5588 → ≥0.87 (+31.1%)"
echo "  carraro:   0.5469 → ≥0.97 (+42.3%)"
echo ""
echo "优化策略:"
echo "  habermann: n_factors=15"
echo "  velmeshev: max_lr_pairs=2000 + gene_emb_dim=96 + n_factors=15"
echo "  carraro:   RF优化 + ensemble=20"
echo ""

# 激活环境
source /root/miniconda3/bin/activate LIANA
cd /root/autodl-tmp/lianaplus_manuscript-main/notebooks/classification

# 创建结果目录
mkdir -p log/optimization
mkdir -p results/optimization

# 测试函数
run_test() {
    dataset=$1
    echo ""
    echo "================================================================"
    echo "  测试数据集: $dataset"
    echo "================================================================"

    # 修改配置文件
    sed -i "s/dataset_name = '.*'/dataset_name = '$dataset'/" code/pipeline/simplified_pipeline_optimized.py

    # 清除checkpoint（确保新配置生效）
    if [ "$dataset" = "velmeshev" ]; then
        echo "  清除velmeshev checkpoint（确保max_lr_pairs=2000生效）..."
        rm -f ../../data/results/checkpoints/velmeshev_ms_hgatlink_tensor_*.pkl
    fi

    # 运行测试
    echo "  开始测试..."
    python code/pipeline/simplified_pipeline_optimized.py 2>&1 | tee log/optimization/${dataset}_optimized_$(date +%Y%m%d_%H%M%S).log

    # 提取结果
    if [ -f "code/pipeline/output/${dataset}_ms_hgatlink_tensor.csv" ]; then
        auroc=$(tail -1 code/pipeline/output/${dataset}_ms_hgatlink_tensor.csv | cut -d',' -f5)
        echo ""
        echo "  ✓ $dataset 测试完成"
        echo "  AUROC: $auroc"
        echo ""
    else
        echo "  ⚠ 未找到结果文件"
    fi
}

# 询问用户测试哪些数据集
echo "选择测试模式:"
echo "  1) 仅测试 habermann (10-20分钟)"
echo "  2) 仅测试 velmeshev (60-120分钟)"
echo "  3) 仅测试 carraro (15-30分钟)"
echo "  4) 测试所有数据集 (90-170分钟)"
echo "  5) 快速验证 (habermann + carraro, 25-50分钟)"
echo ""
read -p "请选择 (1-5): " choice

case $choice in
    1)
        echo "开始测试 habermann..."
        run_test "habermann"
        ;;
    2)
        echo "开始测试 velmeshev..."
        run_test "velmeshev"
        ;;
    3)
        echo "开始测试 carraro..."
        run_test "carraro"
        ;;
    4)
        echo "开始测试所有数据集..."
        run_test "habermann"
        run_test "velmeshev"
        run_test "carraro"
        ;;
    5)
        echo "快速验证（跳过velmeshev）..."
        run_test "habermann"
        run_test "carraro"
        ;;
    *)
        echo "无效选择，退出"
        exit 1
        ;;
esac

echo ""
echo "================================================================"
echo "  所有测试完成！"
echo "================================================================"
echo ""
echo "结果文件位置:"
ls -lh code/pipeline/output/*_ms_hgatlink_tensor.csv 2>/dev/null
echo ""
echo "日志文件位置:"
ls -lh log/optimization/*.log 2>/dev/null | tail -5
echo ""
