"""
完整的端到端流程：自动处理多个数据集
功能: 批量处理多个数据集的预处理→降维→分类流程
数据集: 支持CARRARO/REICHART/KUPPE等多个数据集
输入: data/{dataset}.h5ad (原始数据)
输出: output/{dataset}_*.csv (评估结果)

- 检查数据是否已处理
- 自动运行预处理（如果需要）
- 运行降维和分类流程
- 支持依次处理多个数据集

使用方法:
    python code/pipeline/run_full_pipeline.py

或者在 Python 中:
    from run_full_pipeline import run_datasets
    run_datasets(['carraro', 'reichart', 'kuppe'])
"""

import os
import sys
import gc
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, issparse
from pathlib import Path
import liana as li

# 添加code目录到Python路径
code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))

from process.processer import DatasetHandler
warnings.filterwarnings('ignore')

from liana.method import (
    cellphonedb, connectome, cellchat, scseqcomm,
    singlecellsignalr, natmi, logfc, rank_aggregate, geometric_mean
)

from process.prep_utils import (
    filter_samples,
    filter_celltypes,
    check_group_balance,
    map_gene_symbols
)

file_path = os.path.dirname(os.path.abspath(__file__))


def free_all_memory(use_gpu=True):
    """激进的内存清理"""
    gc.collect()
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass


def preprocess_dataset(dataset_name, handler):
    """
    预处理数据集：过滤、标准化、运行所有 LIANA 方法

    Args:
        dataset_name: 数据集名称
        handler: DatasetHandler 实例

    Returns:
        处理后的 adata
    """
    print("\n" + "="*70)
    print(f"  预处理数据集: {dataset_name}")
    print("="*70)

    os.chdir(file_path)

    # 加载原始数据
    print(f"\n[1] 加载原始数据...")
    raw_path = os.path.join('data', f"{dataset_name}.h5ad")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"原始数据文件不存在: {raw_path}")

    adata = sc.read_h5ad(raw_path)
    print(f"  原始数据形状: {adata.shape}")

    # 应用基本过滤和预处理
    print(f"\n[2] 应用数据过滤和预处理...")

    # 过滤条件
    if handler.conditions_to_keep is not None:
        msk = np.array([c in handler.conditions_to_keep for c in adata.obs[handler.condition_key]])
        adata = adata[msk]
        print(f"  过滤条件后: {adata.shape}")

    # 使用 raw 层（如果需要）
    if handler.use_raw:
        adata = adata.raw.to_adata()

    # 转换为稀疏矩阵
    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)

    # 基因符号映射
    if handler.change_var_to is not None:
        adata.var.index = adata.var[handler.change_var_to]
        # 删除原列以避免保存时的命名冲突
        if handler.change_var_to in adata.var.columns:
            adata.var = adata.var.drop(columns=[handler.change_var_to])
        # 重置 index.name 为 None
        adata.var.index.name = None
        # 确保 var.index 不是 categorical 类型
        if hasattr(adata.var.index, 'categories'):
            adata.var.index = adata.var.index.astype(str)

    if handler.map_path is not None:
        if dataset_name == 'velmeshev':
            adata.var.index = adata.var.index.str.split('\\|').str[0]
        map_df = pd.read_csv(os.path.join('data', handler.map_path))
        adata = map_gene_symbols(adata, map_df)
        print(f"  基因映射后: {adata.shape}")

    # 过滤样本
    adata = filter_samples(adata,
                          sample_key=handler.sample_key,
                          condition_key=handler.condition_key,
                          min_cells_per_sample=handler.min_cells_per_sample,
                          sample_zcounts_max=handler.sample_zcounts_max,
                          sample_zcounts_min=handler.sample_zcounts_min)
    print(f"  过滤样本后: {adata.shape}")

    # 检查组平衡
    adata = check_group_balance(adata,
                               condition_key=handler.condition_key,
                               sample_key=handler.sample_key)

    # 过滤细胞类型
    adata = filter_celltypes(adata=adata,
                            groupby=handler.groupby,
                            sample_key=handler.sample_key,
                            min_cells=handler.min_cells,
                            min_samples=handler.min_samples)
    print(f"  过滤细胞类型后: {adata.shape}")

    # 过滤基因和标准化
    sc.pp.filter_genes(adata, min_cells=handler.min_cells)
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print(f"  最终形状: {adata.shape}")

    free_all_memory()

    # 保存基础预处理数据
    basic_path = os.path.join('data', 'interim', f'{dataset_name}_basic.h5ad')
    print(f"\n[3] 保存基础预处理数据: {basic_path}")
    adata.write_h5ad(basic_path)

    del adata
    free_all_memory()

    # 运行所有 LIANA 方法（逐个处理）
    print(f"\n[4] 运行 LIANA 方法（逐个处理以节省内存）...")
    methods = [cellphonedb, connectome, cellchat, scseqcomm,
               singlecellsignalr, natmi, logfc, rank_aggregate, geometric_mean]

    all_results = {}

    for i, method in enumerate(methods, 1):
        print(f"\n  [{i}/{len(methods)}] 处理方法: {method.method_name}")

        # 重新加载基础数据
        adata = sc.read_h5ad(basic_path)

        score_key = method.magnitude if method.magnitude is not None else method.specificity
        print(f"    Score key: {score_key}")

        try:
            result = method.by_sample(adata,
                                     groupby=handler.groupby,
                                     use_raw=False,
                                     sample_key=handler.sample_key,
                                     verbose=False,
                                     n_perms=None,
                                     inplace=False)

            # 保存到临时文件
            temp_file = os.path.join('data', 'interim', f'{dataset_name}_{score_key}.csv')
            result.to_csv(temp_file, index=False)
            all_results[score_key] = temp_file
            print(f"    ✓ 完成")

        except Exception as e:
            print(f"    ✗ 错误: {str(e)}")

        finally:
            del adata
            free_all_memory()

    # 组合所有结果
    print(f"\n[5] 组合所有 LIANA 结果...")
    adata = sc.read_h5ad(basic_path)

    for score_key, temp_file in all_results.items():
        if os.path.exists(temp_file):
            adata.uns[score_key] = pd.read_csv(temp_file)

    # 保存最终处理数据
    final_path = os.path.join('data', 'interim', f'{dataset_name}_processed.h5ad')
    print(f"\n[6] 保存最终处理数据: {final_path}")
    adata.write_h5ad(final_path)

    # 清理临时文件
    for temp_file in all_results.values():
        if os.path.exists(temp_file):
            os.remove(temp_file)
    if os.path.exists(basic_path):
        os.remove(basic_path)

    print(f"\n✓ 预处理完成: {dataset_name}")

    return adata


def run_classification_pipeline(dataset_name, use_gpu=True):
    """
    运行降维和分类流程

    Args:
        dataset_name: 数据集名称
        use_gpu: 是否使用 GPU
    """
    print("\n" + "="*70)
    print(f"  运行分类流程: {dataset_name}")
    print("="*70)

    os.chdir(file_path)

    handler = DatasetHandler(dataset_name=dataset_name)
    n_factors = handler.n_factors

    # 加载处理后的数据
    processed_path = os.path.join('data', 'interim', f'{dataset_name}_processed.h5ad')
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"处理后的数据不存在: {processed_path}")

    print(f"\n[1] 加载处理后的数据...")
    adata_ref = sc.read_h5ad(processed_path)
    print(f"  数据形状: {adata_ref.shape}")

    # 获取 score keys
    scores = li.mt.show_methods()
    scores['score_key'] = scores["Magnitude Score"].fillna(scores["Specificity Score"])
    scores = scores.drop_duplicates(subset=['Method Name', 'score_key'])
    score_keys = scores['score_key'].tolist()

    print(f"\n[2] 处理 {len(score_keys)} 个 score_keys...")

    # 存储结果
    mofa_results = {'X': {}, 'X_0': {}, 'y_0': {}}
    tensor_results = {'X': {}, 'X_0': {}, 'y_0': {}}

    # 逐个处理 score_keys
    for i, score_key in enumerate(score_keys, 1):
        print(f"\n  [{i}/{len(score_keys)}] 处理 {score_key}...")

        free_all_memory(use_gpu)

        # 重新加载数据
        adata_temp = sc.read_h5ad(processed_path)

        try:
            from classify_utils import _dict_setup, run_mofatalk, run_tensor_c2c

            _dict_setup(adata_temp, 'mofa_res')
            _dict_setup(adata_temp, 'tensor_res')

            # MOFA
            print(f"    运行 MOFA...")
            run_mofatalk(adata=adata_temp,
                        score_key=score_key,
                        sample_key=handler.sample_key,
                        condition_key=handler.condition_key,
                        dataset_name=dataset_name,
                        n_factors=n_factors,
                        gpu_mode=False)

            mofa_results['X'][score_key] = adata_temp.uns['mofa_res']['X'][score_key].copy()
            mofa_results['X_0'][score_key] = adata_temp.uns['mofa_res']['X_0'][score_key].copy()
            mofa_results['y_0'][score_key] = adata_temp.uns['mofa_res']['y_0'][score_key].copy()

            del adata_temp.uns['mofa_res']
            free_all_memory(use_gpu)

            # Tensor
            print(f"    运行 Tensor-c2c...")
            _dict_setup(adata_temp, 'tensor_res')

            run_tensor_c2c(adata=adata_temp,
                          score_key=score_key,
                          sample_key=handler.sample_key,
                          condition_key=handler.condition_key,
                          dataset_name=dataset_name,
                          n_factors=n_factors,
                          use_gpu=use_gpu)

            tensor_results['X'][score_key] = adata_temp.uns['tensor_res']['X'][score_key].copy()
            tensor_results['X_0'][score_key] = adata_temp.uns['tensor_res']['X_0'][score_key].copy()
            tensor_results['y_0'][score_key] = adata_temp.uns['tensor_res']['y_0'][score_key].copy()

            print(f"    ✓ 完成")

        except Exception as e:
            print(f"    ✗ 错误: {str(e)}")
            import traceback
            traceback.print_exc()

        finally:
            del adata_temp
            free_all_memory(use_gpu)

    # 组合结果
    print(f"\n[3] 组合结果并运行分类器...")
    adata = sc.read_h5ad(processed_path)
    adata.uns['mofa_res'] = mofa_results
    adata.uns['tensor_res'] = tensor_results

    # 保存降维结果
    output_path = os.path.join('data', 'results', f'{dataset_name}_dimred.h5ad')
    adata.write(output_path)
    print(f"  降维结果保存至: {output_path}")

    # 运行分类器
    handler.run_classifier(adata)

    print(f"\n[4] 分类结果预览:")
    print(adata.uns['evaluate'].head(10))

    print(f"\n✓ 分类流程完成: {dataset_name}")
    print(f"  结果保存至: data/results/{dataset_name}.csv")

    return adata


def run_single_dataset(dataset_name, use_gpu=True, force_preprocess=False):
    """
    处理单个数据集（包括预处理和分类）

    Args:
        dataset_name: 数据集名称
        use_gpu: 是否使用 GPU
        force_preprocess: 是否强制重新预处理
    """
    print("\n" + "="*70)
    print(f"║  开始处理数据集: {dataset_name.upper()}")
    print("="*70)

    os.chdir(file_path)

    # 检查是否需要预处理
    processed_path = os.path.join('data', 'interim', f'{dataset_name}_processed.h5ad')
    needs_preprocessing = force_preprocess or not os.path.exists(processed_path)

    if needs_preprocessing:
        print(f"\n⚠ 未找到处理后的数据，将运行预处理...")
        handler = DatasetHandler(dataset_name=dataset_name)
        preprocess_dataset(dataset_name, handler)
        free_all_memory(use_gpu)
    else:
        print(f"\n✓ 找到处理后的数据: {processed_path}")
        print(f"  跳过预处理步骤")

    # 运行分类流程
    run_classification_pipeline(dataset_name, use_gpu=use_gpu)

    print(f"\n{'='*70}")
    print(f"║  ✓ 数据集 {dataset_name.upper()} 处理完成")
    print("="*70)


def run_datasets(dataset_names, use_gpu=True, force_preprocess=False):
    """
    依次处理多个数据集

    Args:
        dataset_names: 数据集名称列表
        use_gpu: 是否使用 GPU
        force_preprocess: 是否强制重新预处理所有数据集
    """
    import psutil
    import time

    process = psutil.Process()
    start_time = time.time()

    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*20 + "批量数据集处理流程" + " "*27 + "║")
    print("╚" + "="*68 + "╝")
    print(f"\n总数据集数量: {len(dataset_names)}")
    print(f"数据集列表: {', '.join(dataset_names)}")
    print(f"使用 GPU: {use_gpu}")
    print(f"初始内存: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    results = {}

    for idx, dataset_name in enumerate(dataset_names, 1):
        print(f"\n\n{'#'*70}")
        print(f"#  进度: [{idx}/{len(dataset_names)}]")
        print(f"{'#'*70}")

        try:
            run_single_dataset(dataset_name, use_gpu=use_gpu, force_preprocess=force_preprocess)
            results[dataset_name] = "成功"

        except Exception as e:
            print(f"\n✗ 处理 {dataset_name} 时发生错误:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
            results[dataset_name] = f"失败: {str(e)}"

        finally:
            free_all_memory(use_gpu)
            current_mem = process.memory_info().rss / 1024 / 1024
            print(f"\n当前内存: {current_mem:.2f} MB")

    # 最终总结
    elapsed_time = time.time() - start_time

    print("\n\n" + "╔" + "="*68 + "╗")
    print("║" + " "*25 + "处理总结" + " "*35 + "║")
    print("╚" + "="*68 + "╝")
    print(f"\n总耗时: {elapsed_time/60:.1f} 分钟")
    print(f"最终内存: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    print(f"\n结果:")
    for dataset, status in results.items():
        status_symbol = "✓" if status == "成功" else "✗"
        print(f"  {status_symbol} {dataset:15s}: {status}")

    # 生成结果文件列表
    print(f"\n生成的结果文件:")
    for dataset in dataset_names:
        csv_path = os.path.join('data', 'results', f'{dataset}.csv')
        h5ad_path = os.path.join('data', 'results', f'{dataset}_dimred.h5ad')
        if os.path.exists(csv_path):
            print(f"  ✓ {csv_path}")
        if os.path.exists(h5ad_path):
            print(f"  ✓ {h5ad_path}")

    return results


if __name__ == "__main__":
    # 要处理的数据集列表
    datasets_to_process = ['habermann', 'reichart', 'velmeshev']

    # 运行批量处理
    # force_preprocess=False: 如果已有处理后的数据则跳过预处理
    # force_preprocess=True: 强制重新预处理所有数据
    results = run_datasets(
        dataset_names=datasets_to_process,
        use_gpu=True,
        force_preprocess=False  # 改为 True 可强制重新预处理
    )

    print("\n✓ 所有数据集处理完成！")
