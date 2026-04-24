#!/bin/bash

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     MS-HGATLink优化项目 - 快速恢复指南                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# 切换到工作目录
cd /root/autodl-tmp/lianaplus_manuscript-main/notebooks/classification

echo "📍 当前目录: $(pwd)"
echo ""

echo "📊 最终测试结果："
echo "  ✅ Habermann (Simplified HGATlink): AUROC 0.9535 (目标≥0.87)"
echo "  ❌ Velmeshev (MS-HGATlink): AUROC 0.6597 (目标≥0.87, -24%)"
echo "  ❌ Carraro (MS-HGATlink): AUROC 0.7812 (目标≥0.97, -19%)"
echo ""

echo "📁 关键文件位置："
echo "  • 完整报告: MS_HGATLink_优化完整报告.md"
echo "  • 主脚本: code/pipeline/simplified_pipeline_optimized.py"
echo "  • 模型代码: code/model/ms_hgatlink_method.py"
echo "  • 结果文件: code/pipeline/output/*.csv"
echo ""

echo "🔧 当前配置状态："
echo "  • Dataset: $(grep "dataset_name = " code/pipeline/simplified_pipeline_optimized.py | head -1)"
echo "  • Method: MS-HGATLink"
echo "  • Velmeshev n_factors: 30 (测试后，性能下降)"
echo ""

echo "⚠️  建议恢复基线配置（Velmeshev最佳）："
echo "  1. n_factors: 30 → 15"
echo "  2. gene_emb_dim: 96, dropout: 0.3"
echo "  3. 禁用数据增强"
echo ""

echo "🚀 快速命令："
echo ""
echo "# 激活环境"
echo "source /root/miniconda3/bin/activate LIANA"
echo ""
echo "# 查看最佳结果"
echo "cat code/pipeline/output/carraro_ms_hgatlink_tensor.csv"
echo ""
echo "# 查看完整报告"
echo "cat MS_HGATLink_优化完整报告.md"
echo ""
echo "# 重新测试（如需要）"
echo "python code/pipeline/simplified_pipeline_optimized.py"
echo ""

echo "✅ 恢复指南生成完成！"
echo "详细信息请查看: MS_HGATLink_优化完整报告.md"
