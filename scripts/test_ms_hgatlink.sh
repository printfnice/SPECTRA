#!/bin/bash
# MS-HGATLink快速测试脚本
# 直接运行MS-HGATLink方法，不运行其他LIANA方法

set -e

echo "=========================================="
echo "MS-HGATLink 性能测试"
echo "=========================================="
echo ""

# 激活环境
echo "[1/3] 激活LIANA环境..."
source /root/miniconda3/bin/activate LIANA

# 切换目录
cd /root/autodl-tmp/lianaplus_manuscript-main/notebooks/classification

# 获取数据集参数（默认reichart）
DATASET=${1:-reichart}
REDUCTION=${2:-det}
SPEED_MODE=${3:-quick}

echo ""
echo "[2/3] 配置参数:"
echo "  数据集: $DATASET"
echo "  降维方法: $REDUCTION"
echo "  速度模式: $SPEED_MODE"
echo "  方法: ms_hgatlink (带预训练)"
echo ""

# 创建临时Python脚本
echo "[3/3] 运行MS-HGATLink测试..."

python3 << EOF
import sys
from pathlib import Path
code_dir = Path('code')
sys.path.insert(0, str(code_dir))

# 修改配置
dataset_name = '$DATASET'
liana_method = 'ms_hgatlink'
reduction_method = '$REDUCTION'
speed_mode = '$SPEED_MODE'
use_gpu = True

print(f"\n🚀 开始测试:")
print(f"  数据集: {dataset_name}")
print(f"  方法: {liana_method}")
print(f"  降维: {reduction_method}")
print(f"  速度模式: {speed_mode}")
print("")

# 导入必要的模块
import warnings
warnings.filterwarnings('ignore')

import os
os.chdir('code/pipeline')

# 设置环境变量以覆盖配置
os.environ['DATASET_NAME'] = dataset_name
os.environ['LIANA_METHOD'] = liana_method
os.environ['REDUCTION_METHOD'] = reduction_method
os.environ['SPEED_MODE'] = speed_mode

# 运行pipeline的核心代码
exec(open('simplified_pipeline_optimized.py').read())
EOF

echo ""
echo "=========================================="
echo "✓ 测试完成！"
echo "=========================================="
echo ""
echo "结果位置:"
echo "  output/${DATASET}_ms_hgatlink_${REDUCTION}.csv"
echo ""
