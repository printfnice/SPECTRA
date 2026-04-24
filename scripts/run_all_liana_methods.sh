#!/bin/bash
################################################################################
# LIANA方法批量测试脚本 (修复数据泄露版本)
################################################################################
# 功能：在5个数据集上运行8个LIANA方法
# 作者：Claude Code
# 日期：2026-01-30
#
# 数据集：kuppe, reichart, habermann, velmeshev, carraro
# 方法：cellphonedb, cellchat, scseqcomm, singlecellsignalr,
#       logfc, rank_aggregate, geometric_mean, connectome
################################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 工作目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PIPELINE_DIR="${SCRIPT_DIR}/../code/pipeline"
LOG_DIR="${SCRIPT_DIR}/../logs/batch_runs"
OUTPUT_DIR="${PIPELINE_DIR}/output"

# 创建日志目录
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     LIANA方法批量测试 - 5个数据集 × 8个方法 = 40次运行           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 数据集列表
DATASETS=(
    "kuppe"
    "reichart"
    "habermann"
    "velmeshev"
    "carraro"
)

# LIANA方法列表（对应用户要求的8个方法）
METHODS=(
    "cellphonedb"        # CellPhoneDB
    "cellchat"           # CellChat
    "scseqcomm"          # scSeqComm
    "singlecellsignalr"  # SingleCellSignalR
    "logfc"              # logFC
    "rank_aggregate"     # RankAggregate
    "geometric_mean"     # GeometricMean
    "connectome"         # Connectome (补充方法，可能对应用户的"Product")
)

# 方法显示名称映射
declare -A METHOD_NAMES=(
    ["cellphonedb"]="CellPhoneDB"
    ["cellchat"]="CellChat"
    ["scseqcomm"]="scSeqComm"
    ["singlecellsignalr"]="SingleCellSignalR"
    ["logfc"]="LogFC"
    ["rank_aggregate"]="RankAggregate"
    ["geometric_mean"]="GeometricMean"
    ["connectome"]="Connectome"
)

# 降维方法（使用DET，已修复数据泄露）
REDUCTION_METHOD="det"

# 统计信息
TOTAL_RUNS=$((${#DATASETS[@]} * ${#METHODS[@]}))
CURRENT_RUN=0
SUCCESS_COUNT=0
FAILED_COUNT=0
SKIPPED_COUNT=0

# 失败任务列表
declare -a FAILED_TASKS

# 开始时间
START_TIME=$(date +%s)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo -e "${GREEN}配置信息:${NC}"
echo "  数据集数量: ${#DATASETS[@]}"
echo "  方法数量: ${#METHODS[@]}"
echo "  降维方法: ${REDUCTION_METHOD} (DET - 修复数据泄露版本)"
echo "  总任务数: ${TOTAL_RUNS}"
echo "  日志目录: ${LOG_DIR}"
echo "  输出目录: ${OUTPUT_DIR}"
echo ""

# 激活conda环境
echo -e "${YELLOW}激活LIANA环境...${NC}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate LIANA
echo -e "${GREEN}✓ 环境已激活${NC}"
echo ""

# 主循环
for dataset in "${DATASETS[@]}"; do
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  数据集: ${dataset}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    for method in "${METHODS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        METHOD_NAME="${METHOD_NAMES[$method]}"

        # 进度显示
        PROGRESS=$((CURRENT_RUN * 100 / TOTAL_RUNS))
        echo -e "${YELLOW}[${CURRENT_RUN}/${TOTAL_RUNS}] (${PROGRESS}%) ${NC}运行: ${GREEN}${dataset}${NC} + ${BLUE}${METHOD_NAME}${NC}"

        # 输出文件
        OUTPUT_FILE="${OUTPUT_DIR}/${dataset}_${method}_${REDUCTION_METHOD}.csv"
        LOG_FILE="${LOG_DIR}/${dataset}_${method}_${REDUCTION_METHOD}_${TIMESTAMP}.log"

        # 检查是否已存在结果
        if [ -f "${OUTPUT_FILE}" ]; then
            echo -e "  ${YELLOW}⚠ 结果已存在，跳过${NC}"
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
            echo ""
            continue
        fi

        # 创建临时Python脚本
        TEMP_SCRIPT="${PIPELINE_DIR}/temp_run_${dataset}_${method}.py"

        cat > "${TEMP_SCRIPT}" << 'PYTHON_SCRIPT_END'
import os
import sys

# 修改配置
dataset_name = 'DATASET_PLACEHOLDER'
liana_method = 'METHOD_PLACEHOLDER'
reduction_method = 'REDUCTION_PLACEHOLDER'
use_gpu = True
resume_from_checkpoint = False

# 导入主函数
sys.path.insert(0, os.path.dirname(__file__))
from simplified_pipeline_optimized import main

# 覆盖全局配置
import simplified_pipeline_optimized as spo
spo.dataset_name = dataset_name
spo.liana_method = liana_method
spo.reduction_method = reduction_method
spo.use_gpu = use_gpu
spo.resume_from_checkpoint = resume_from_checkpoint

# 运行
if __name__ == "__main__":
    main()
PYTHON_SCRIPT_END

        # 替换占位符
        sed -i "s/DATASET_PLACEHOLDER/${dataset}/g" "${TEMP_SCRIPT}"
        sed -i "s/METHOD_PLACEHOLDER/${method}/g" "${TEMP_SCRIPT}"
        sed -i "s/REDUCTION_PLACEHOLDER/${REDUCTION_METHOD}/g" "${TEMP_SCRIPT}"

        # 运行pipeline
        echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

        if cd "${PIPELINE_DIR}" && python "${TEMP_SCRIPT}" > "${LOG_FILE}" 2>&1; then
            echo -e "  ${GREEN}✓ 成功完成${NC}"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

            # 显示结果摘要
            if [ -f "${OUTPUT_FILE}" ]; then
                echo "  结果文件: ${OUTPUT_FILE}"
            fi
        else
            echo -e "  ${RED}✗ 运行失败${NC}"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            FAILED_TASKS+=("${dataset} + ${METHOD_NAME}")
            echo "  错误日志: ${LOG_FILE}"
        fi

        # 清理临时文件
        rm -f "${TEMP_SCRIPT}"

        echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
    done
done

# 结束时间
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED_TIME / 60))
ELAPSED_SEC=$((ELAPSED_TIME % 60))

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                      批量运行完成                                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}统计信息:${NC}"
echo "  总任务数: ${TOTAL_RUNS}"
echo -e "  ${GREEN}成功: ${SUCCESS_COUNT}${NC}"
echo -e "  ${RED}失败: ${FAILED_COUNT}${NC}"
echo -e "  ${YELLOW}跳过: ${SKIPPED_COUNT}${NC}"
echo "  总耗时: ${ELAPSED_MIN}分${ELAPSED_SEC}秒"
echo ""

# 显示失败任务
if [ ${FAILED_COUNT} -gt 0 ]; then
    echo -e "${RED}失败的任务:${NC}"
    for task in "${FAILED_TASKS[@]}"; do
        echo "  - ${task}"
    done
    echo ""
fi

echo -e "${GREEN}结果文件位置:${NC} ${OUTPUT_DIR}"
echo -e "${GREEN}日志文件位置:${NC} ${LOG_DIR}"
echo ""

# 生成汇总报告
REPORT_FILE="${LOG_DIR}/batch_report_${TIMESTAMP}.txt"
cat > "${REPORT_FILE}" << EOF
LIANA方法批量测试报告
=====================
运行时间: $(date '+%Y-%m-%d %H:%M:%S')
总耗时: ${ELAPSED_MIN}分${ELAPSED_SEC}秒

统计:
- 总任务数: ${TOTAL_RUNS}
- 成功: ${SUCCESS_COUNT}
- 失败: ${FAILED_COUNT}
- 跳过: ${SKIPPED_COUNT}

数据集: ${DATASETS[@]}
方法: ${METHODS[@]}
降维: ${REDUCTION_METHOD}

结果目录: ${OUTPUT_DIR}
日志目录: ${LOG_DIR}
EOF

if [ ${FAILED_COUNT} -gt 0 ]; then
    echo "" >> "${REPORT_FILE}"
    echo "失败任务:" >> "${REPORT_FILE}"
    for task in "${FAILED_TASKS[@]}"; do
        echo "  - ${task}" >> "${REPORT_FILE}"
    done
fi

echo -e "${GREEN}✓ 报告已生成: ${REPORT_FILE}${NC}"

# 退出码
if [ ${FAILED_COUNT} -gt 0 ]; then
    exit 1
else
    exit 0
fi
