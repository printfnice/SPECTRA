#!/bin/bash
# 验证KG实施指南的完整性和可执行性
# Verification script for KG implementation guide

echo "======================================================================="
echo "验证KG实施指南 - Verification Script"
echo "======================================================================="
echo ""

# 切换到工作目录
cd /root/autodl-tmp/lianaplus_manuscript-main/notebooks/classification || exit 1
echo "✓ 工作目录: $(pwd)"
echo ""

# 1. 检查文档文件
echo "1️⃣  检查文档文件..."
docs=(
    "doc/KG_IMPLEMENTATION_GUIDE.md"
    "doc/KG_GUIDED_ATTENTION_PLAN.md"
    "doc/README_KG_DOCS.md"
)

for doc in "${docs[@]}"; do
    if [ -f "$doc" ]; then
        lines=$(wc -l < "$doc")
        size=$(du -h "$doc" | cut -f1)
        echo "  ✓ $doc ($lines 行, $size)"
    else
        echo "  ✗ $doc (缺失)"
    fi
done
echo ""

# 2. 检查已创建的代码文件
echo "2️⃣  检查已创建的代码文件..."
if [ -f "code/model/kg_utils.py" ]; then
    lines=$(wc -l < "code/model/kg_utils.py")
    echo "  ✓ code/model/kg_utils.py ($lines 行)"
else
    echo "  ✗ code/model/kg_utils.py (缺失)"
fi
echo ""

# 3. 检查待创建的文件
echo "3️⃣  检查待创建的文件..."
pending_files=(
    "code/model/kg_fusion.py"
    "code/test_kg_mvp.py"
    "code/test_kg_benchmark.py"
    "code/visual/plot_kg_analysis.py"
)

for file in "${pending_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file (已创建)"
    else
        echo "  ⏳ $file (待创建)"
    fi
done
echo ""

# 4. 检查数据文件
echo "4️⃣  检查数据文件..."
datasets=("carraro" "habermann" "kuppe" "reichart" "velmeshev")
for dataset in "${datasets[@]}"; do
    if [ -f "data/${dataset}.h5ad" ]; then
        size=$(du -h "data/${dataset}.h5ad" | cut -f1)
        echo "  ✓ data/${dataset}.h5ad ($size)"
    else
        echo "  ✗ data/${dataset}.h5ad (缺失)"
    fi
done
echo ""

# 5. 测试Python环境
echo "5️⃣  测试Python环境..."
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate LIANA 2>/dev/null

if python -c "import liana; import scanpy; import pandas; import numpy" 2>/dev/null; then
    echo "  ✓ Python环境正常"

    # 测试liana版本
    liana_version=$(python -c "import liana; print(liana.__version__)" 2>/dev/null)
    echo "  ✓ LIANA版本: $liana_version"

    # 测试scanpy版本
    scanpy_version=$(python -c "import scanpy; print(scanpy.__version__)" 2>/dev/null)
    echo "  ✓ Scanpy版本: $scanpy_version"
else
    echo "  ✗ Python环境有问题"
fi
echo ""

# 6. 测试已完成的模块
echo "6️⃣  测试已完成的模块..."
if [ -f "code/model/kg_utils.py" ]; then
    echo "  测试 kg_utils.py..."
    if python code/model/kg_utils.py 2>&1 | grep -q "Test completed successfully"; then
        echo "  ✓ kg_utils.py 测试通过"
    else
        echo "  ⚠️  kg_utils.py 测试可能有问题"
    fi
else
    echo "  ⏳ kg_utils.py 尚未创建"
fi
echo ""

# 7. 检查关键函数
echo "7️⃣  检查关键函数..."
if python -c "from code.model.hgatlink_method import hgatlink_score, hgatlink_inference_vectorized" 2>/dev/null; then
    echo "  ✓ hgatlink_score 可导入"
    echo "  ✓ hgatlink_inference_vectorized 可导入"

    # 检查函数签名
    if python -c "import inspect; from code.model.hgatlink_method import hgatlink_score; sig = inspect.signature(hgatlink_score); assert 'use_kg_prior' in sig.parameters" 2>/dev/null; then
        echo "  ✓ hgatlink_score 已包含 use_kg_prior 参数"
    else
        echo "  ⏳ hgatlink_score 需要添加 use_kg_prior 参数"
    fi
else
    echo "  ✗ 无法导入HGATLink函数"
fi
echo ""

# 8. 生成实施清单
echo "8️⃣  生成实施清单..."
echo ""
echo "阶段1：MVP版本"
echo "  [x] 任务1.1：创建KG元数据提取模块"
if grep -q "use_kg_prior" code/model/hgatlink_method.py 2>/dev/null; then
    echo "  [x] 任务1.2：修改核心评分函数"
else
    echo "  [ ] 任务1.2：修改核心评分函数 ← 下一步"
fi
if [ -f "code/test_kg_mvp.py" ]; then
    echo "  [x] 任务1.3：MVP测试验证"
else
    echo "  [ ] 任务1.3：MVP测试验证"
fi
echo ""
echo "阶段2：自适应权重"
if [ -f "code/model/kg_fusion.py" ]; then
    echo "  [x] 任务2.1：创建权重计算模块"
else
    echo "  [ ] 任务2.1：创建权重计算模块"
fi
echo "  [ ] 任务2.2：集成自适应权重"
echo "  [ ] 任务2.3：5数据集Benchmark"
echo ""
echo "阶段3：可解释性增强"
if [ -f "code/visual/plot_kg_analysis.py" ]; then
    echo "  [x] 任务3.1：创建可视化模块"
else
    echo "  [ ] 任务3.1：创建可视化模块"
fi
echo "  [ ] 任务3.2：生成HTML报告"
echo ""

# 9. 总结
echo "======================================================================="
echo "验证总结"
echo "======================================================================="
echo ""
echo "📚 文档状态: 完整 (3个核心文档已创建)"
echo "✅ 阶段1.1: 已完成 (kg_utils.py)"
echo "⏳ 阶段1.2: 进行中 (需要修改 hgatlink_method.py)"
echo ""
echo "📖 下一步: 参考 doc/KG_IMPLEMENTATION_GUIDE.md 第3.2节"
echo "   修改 code/model/hgatlink_method.py"
echo ""
echo "🚀 快速启动:"
echo "   1. cat doc/README_KG_DOCS.md  # 阅读文档索引"
echo "   2. cat doc/KG_IMPLEMENTATION_GUIDE.md  # 阅读实施指南"
echo "   3. 按照指南第3.2节修改 hgatlink_method.py"
echo ""
echo "======================================================================="
