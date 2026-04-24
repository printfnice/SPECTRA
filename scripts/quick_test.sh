#!/bin/bash
################################################################################
# LIANA方法快速测试脚本
################################################################################
# 功能：在1个数据集上运行1个方法，用于测试脚本是否正常工作
# 使用：bash quick_test.sh
################################################################################

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           LIANA方法快速测试（修复数据泄露版本）               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 工作目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PIPELINE_DIR="${SCRIPT_DIR}/../code/pipeline"

# 测试配置（使用最快的组合）
TEST_DATASET="carraro"      # 最小的数据集
TEST_METHOD="logfc"         # 最快的方法
TEST_REDUCTION="det"

echo -e "${GREEN}测试配置:${NC}"
echo "  数据集: ${TEST_DATASET} (最小数据集)"
echo "  方法: ${TEST_METHOD} (最快方法)"
echo "  降维: ${TEST_REDUCTION} (修复数据泄露)"
echo ""

# 激活环境
echo -e "${YELLOW}激活LIANA环境...${NC}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate LIANA
echo -e "${GREEN}✓ 环境已激活${NC}"
echo ""

# 创建临时测试脚本
TEMP_SCRIPT="${PIPELINE_DIR}/temp_quick_test.py"

cat > "${TEMP_SCRIPT}" << 'PYTHON_END'
import os
import sys

# 测试配置
dataset_name = 'TEST_DATASET'
liana_method = 'TEST_METHOD'
reduction_method = 'TEST_REDUCTION'
use_gpu = True
resume_from_checkpoint = False

# 修改为快速测试模式（减少训练时间）
import simplified_pipeline_optimized as spo
spo.dataset_name = dataset_name
spo.liana_method = liana_method
spo.reduction_method = reduction_method
spo.use_gpu = use_gpu
spo.resume_from_checkpoint = resume_from_checkpoint

# 设置为快速模式
spo.speed_mode = 'quick'
spo.SPEED_PRESETS['quick']['n_random_states'] = 1  # 只运行1次
spo.SPEED_PRESETS['quick']['n_cv_folds'] = 2       # 2折CV
spo.SPEED_PRESETS['quick']['n_estimators'] = 50    # 50棵树

# 运行
from simplified_pipeline_optimized import main
if __name__ == "__main__":
    main()
PYTHON_END

# 替换占位符
sed -i "s/TEST_DATASET/${TEST_DATASET}/g" "${TEMP_SCRIPT}"
sed -i "s/TEST_METHOD/${TEST_METHOD}/g" "${TEMP_SCRIPT}"
sed -i "s/TEST_REDUCTION/${TEST_REDUCTION}/g" "${TEMP_SCRIPT}"

# 运行测试
echo -e "${YELLOW}开始快速测试...${NC}"
echo "预计时间: 5-10分钟"
echo ""

START_TIME=$(date +%s)

if cd "${PIPELINE_DIR}" && python "${TEMP_SCRIPT}"; then
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))
    ELAPSED_SEC=$((ELAPSED % 60))

    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    测试成功！✓                              ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "耗时: ${ELAPSED_MIN}分${ELAPSED_SEC}秒"
    echo ""

    # 检查输出文件
    OUTPUT_FILE="${PIPELINE_DIR}/output/${TEST_DATASET}_${TEST_METHOD}_${TEST_REDUCTION}.csv"
    if [ -f "${OUTPUT_FILE}" ]; then
        echo -e "${GREEN}✓ 结果文件已生成:${NC}"
        echo "  ${OUTPUT_FILE}"
        echo ""

        # 显示结果摘要
        echo -e "${BLUE}结果预览:${NC}"
        if command -v csvtool &> /dev/null; then
            csvtool head 5 "${OUTPUT_FILE}" | column -t -s,
        else
            head -5 "${OUTPUT_FILE}"
        fi
        echo ""
    fi

    echo -e "${GREEN}✓ 脚本工作正常，可以运行完整批量测试${NC}"
    echo ""
    echo "运行完整批量测试:"
    echo "  bash run_all_liana_methods.sh"
    echo ""
else
    echo ""
    echo -e "${RED}✗ 测试失败${NC}"
    echo ""
    echo "请检查:"
    echo "  1. LIANA环境是否正确安装"
    echo "  2. 数据文件是否存在"
    echo "  3. 查看错误信息"
    exit 1
fi

# 清理
rm -f "${TEMP_SCRIPT}"

exit 0
