#!/bin/bash
#================================================================
# 使用正确的conda环境运行所有数据集
#================================================================

# 激活LIANA环境
source /root/miniconda3/bin/activate LIANA

cd /root/autodl-tmp/lianaplus_manuscript-main/notebooks/classification/code/pipeline

# 运行Python脚本
python run_all_sota.py
