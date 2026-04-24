#!/usr/bin/env python3
"""
MS-HGATLink 快速测试脚本
直接测试MS-HGATLink方法，不运行其他LIANA方法

使用方法:
    python test_ms_hgatlink_only.py                        # 默认reichart数据集
    python test_ms_hgatlink_only.py reichart               # 指定数据集
    python test_ms_hgatlink_only.py habermann det quick   # 完整参数
"""

import sys
import os
from pathlib import Path

# 添加code目录到路径
code_dir = Path(__file__).parent.parent / 'code'
sys.path.insert(0, str(code_dir))

# 切换到pipeline目录
os.chdir(Path(__file__).parent.parent / 'code' / 'pipeline')

print("=" * 50)
print("MS-HGATLink 性能测试")
print("=" * 50)
print()

# 解析命令行参数
dataset_name = sys.argv[1] if len(sys.argv) > 1 else 'reichart'
reduction_method = sys.argv[2] if len(sys.argv) > 2 else 'det'
speed_mode = sys.argv[3] if len(sys.argv) > 3 else 'quick'

print(f"📋 配置参数:")
print(f"  数据集: {dataset_name}")
print(f"  方法: ms_hgatlink (带预训练)")
print(f"  降维: {reduction_method}")
print(f"  速度模式: {speed_mode}")
print()

# 读取原始pipeline脚本
with open('simplified_pipeline_optimized.py', 'r', encoding='utf-8') as f:
    pipeline_code = f.read()

# 替换关键配置
pipeline_code = pipeline_code.replace(
    "dataset_name = 'reichart'",
    f"dataset_name = '{dataset_name}'"
)
pipeline_code = pipeline_code.replace(
    "liana_method = 'cellchat'",
    "liana_method = 'ms_hgatlink'"
)
pipeline_code = pipeline_code.replace(
    "reduction_method = 'det'",
    f"reduction_method = '{reduction_method}'"
)
pipeline_code = pipeline_code.replace(
    "speed_mode = 'rigorous'",
    f"speed_mode = '{speed_mode}'"
)

# 添加MS-HGATLink特殊配置
ms_hgatlink_config = """

# ============================================================================
# MS-HGATLink 特殊配置
# ============================================================================
ms_hgatlink_kwargs = {
    'enable_pretraining': True,      # 启用预训练
    'pretrain_epochs': 50,           # 预训练轮数（数据集特异性）
    'condition_key': None,           # 条件列（如果有）
    'dataset_name': dataset_name,    # 数据集名称（自动选择超参数）
}
print("\\n🚀 MS-HGATLink配置:")
print(f"  启用预训练: {ms_hgatlink_kwargs['enable_pretraining']}")
print(f"  预训练轮数: {ms_hgatlink_kwargs['pretrain_epochs']}")
print(f"  数据集配置: {ms_hgatlink_kwargs['dataset_name']}")
print()

"""

# 在导入部分后插入配置
import_end = pipeline_code.find("# ============================================================================\n# 🔧 配置参数")
if import_end != -1:
    pipeline_code = pipeline_code[:import_end] + ms_hgatlink_config + pipeline_code[import_end:]

print("🚀 开始运行MS-HGATLink测试...")
print()

# 执行修改后的pipeline
exec(pipeline_code)

print()
print("=" * 50)
print("✓ 测试完成！")
print("=" * 50)
print()
print(f"📊 结果位置:")
print(f"  output/{dataset_name}_ms_hgatlink_{reduction_method}.csv")
print()
