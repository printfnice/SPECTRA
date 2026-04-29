"""
功能: 简化的分类流程入口脚本
数据集: 可配置多个数据集（默认KUPPE）
输入: data/{dataset}.h5ad (原始数据)
输出: 通过DatasetHandler处理和分类

使用方法:
    python code/pipeline/1.run_pipeline.py
"""

import os
import sys
import warnings
from pathlib import Path

# 添加code目录到Python路径
code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))

from process.processer import DatasetHandler

warnings.filterwarnings('ignore') # silence endless anndata spam

file_path = os.path.dirname(os.path.abspath(__file__))
# datasets = ["carraro", "velmeshev" ,"reichart", "kuppe", "habermann"]
datasets = ["kuppe"]  # carraro already processed
def run_classification(dataset_name, use_gpu=True):
    os.chdir(file_path)
    # Create an instance of DatasetHandler
    handler = DatasetHandler(dataset_name=dataset_name)
    
    # Process the dataset
    adata = handler.process_dataset()
    # import scanpy as sc
    # adata = sc.read_h5ad(os.path.join('data', 'interim', f'{dataset_name}_processed.h5ad'))
    
    # Perform dimensionality reduction using dim_reduction_pipe function
    handler.dim_reduction_pipe(adata, use_gpu=use_gpu)
    
    # Run classifier on the reduced dataset using run_classifier function
    handler.run_classifier(adata)
    print(adata.uns['evaluate'].head())

if __name__ == "__main__":    
    for dataset_name in datasets:
        run_classification(dataset_name)
