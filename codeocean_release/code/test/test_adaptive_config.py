#!/usr/bin/env python3
"""
测试自适应配置系统
功能: 验证自适应配置是否正确工作
数据集: CARRARO（默认）
输入: data/{dataset}.h5ad
输出: 控制台输出（配置对比结果）
"""

import os
import sys
from pathlib import Path

# 添加code目录到Python路径
code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))
# 添加项目根目录
project_root = code_dir.parent
os.chdir(project_root)

import scanpy as sc
from process.processer import DatasetHandler

def test_adaptive_config(dataset_name='carraro'):
    """测试单个数据集的自适应配置"""

    print("\n" + "=" * 80)
    print(f"测试数据集: {dataset_name.upper()}")
    print("=" * 80)

    # 创建处理器（启用自适应配置）
    handler_adaptive = DatasetHandler(dataset_name, use_adaptive_config=True)

    # 创建处理器（不启用自适应配置，用于对比）
    handler_original = DatasetHandler(dataset_name, use_adaptive_config=False)

    # 加载数据
    print(f"\n加载数据: data/{dataset_name}.h5ad")
    adata = sc.read_h5ad(os.path.join('data', f"{dataset_name}.h5ad"))

    print(f"数据集基本信息:")
    print(f"  细胞数: {adata.shape[0]:,}")
    print(f"  基因数: {adata.shape[1]:,}")
    print(f"  样本数: {adata.obs[handler_adaptive.sample_key].nunique()}")

    # 模拟调用dim_reduction_pipe（只触发自适应配置，不实际运行降维）
    print("\n触发自适应配置系统...")

    # 简单过滤（为了快速测试）
    if handler_adaptive.conditions_to_keep is not None:
        import numpy as np
        msk = np.array([patient in handler_adaptive.conditions_to_keep
                       for patient in adata.obs[handler_adaptive.condition_key]])
        adata = adata[msk]

    # 触发自适应配置
    print("\n【使用自适应配置】")
    _ = handler_adaptive.dim_reduction_pipe.__code__  # 获取方法代码对象
    # 手动触发配置逻辑
    from adaptive_config import get_adaptive_config_for_dataset

    config_adaptive, analysis = get_adaptive_config_for_dataset(
        adata,
        dataset_name=dataset_name,
        sample_key=handler_adaptive.sample_key,
        condition_key=handler_adaptive.condition_key,
        groupby=handler_adaptive.groupby,
        verbose=True
    )

    # 对比配置
    print("\n" + "=" * 80)
    print("配置对比")
    print("=" * 80)

    print(f"\n{'参数':<20} {'原始配置':<15} {'自适应配置':<15} {'变化':<15}")
    print("-" * 80)

    print(f"{'n_factors':<20} {handler_original.n_factors:<15} {config_adaptive['n_factors']:<15} "
          f"{config_adaptive['n_factors'] - handler_original.n_factors:+d}")

    if 'cv_folds' in config_adaptive:
        print(f"{'cv_folds':<20} {'3 (默认)':<15} {config_adaptive['cv_folds']:<15} "
              f"{config_adaptive['cv_folds'] - 3:+d}")

    if 'dropout' in config_adaptive:
        print(f"{'dropout':<20} {'0.3 (默认)':<15} {config_adaptive['dropout']:<15.2f} "
              f"{config_adaptive['dropout'] - 0.3:+.2f}")

    print("\n✅ 测试完成！")

    return config_adaptive, analysis


def test_all_datasets():
    """测试所有数据集"""

    datasets = ['carraro', 'habermann', 'kuppe', 'reichart', 'velmeshev']

    results = {}

    for dataset in datasets:
        try:
            config, analysis = test_adaptive_config(dataset)
            results[dataset] = {
                'config': config,
                'analysis': analysis,
                'status': '✅ 成功'
            }
        except Exception as e:
            print(f"\n⚠️ {dataset} 测试失败: {e}")
            results[dataset] = {
                'config': None,
                'analysis': None,
                'status': f'❌ 失败: {e}'
            }

    # 汇总报告
    print("\n\n" + "=" * 80)
    print(" " * 30 + "测试汇总报告")
    print("=" * 80)

    print(f"\n{'数据集':<15} {'样本数':<10} {'n_factors':<12} {'复杂度':<12} {'状态':<15}")
    print("-" * 80)

    for dataset, result in results.items():
        if result['config'] is not None:
            n_samples = result['analysis']['n_samples']
            n_factors = result['config']['n_factors']
            complexity = result['analysis']['complexity_level']
            status = result['status']

            print(f"{dataset:<15} {n_samples:<10} {n_factors:<12} {complexity.upper():<12} {status:<15}")
        else:
            print(f"{dataset:<15} {'N/A':<10} {'N/A':<12} {'N/A':<12} {result['status']:<15}")

    print("=" * 80)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='测试自适应配置系统')
    parser.add_argument('--dataset', type=str, default='carraro',
                       help='数据集名称 (carraro, habermann, kuppe, reichart, velmeshev, all)')

    args = parser.parse_args()

    if args.dataset == 'all':
        test_all_datasets()
    else:
        test_adaptive_config(args.dataset)
