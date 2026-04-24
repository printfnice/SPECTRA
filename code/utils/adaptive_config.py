#!/usr/bin/env python3
"""
自适应配置系统 - HGATLink模型参数自动优化

根据数据集特征（样本数、数据密度、任务复杂度）自动调整模型超参数，
以适配小数据集和大复杂数据集。

作者: Claude Code
日期: 2026-01-26
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class DatasetComplexityAnalyzer:
    """
    数据集复杂度分析器

    评估数据集的三个关键维度：
    1. 样本规模（n_samples）
    2. 数据密度（cells_per_sample）
    3. 任务复杂度（用快速基线模型估计）
    """

    def __init__(self, verbose=True):
        self.verbose = verbose

    def analyze(self, adata, sample_key, condition_key, groupby=None):
        """
        全面分析数据集特征

        Parameters
        ----------
        adata : AnnData
            单细胞数据对象
        sample_key : str
            样本列名
        condition_key : str
            条件/分组列名
        groupby : str, optional
            细胞类型列名

        Returns
        -------
        dict
            包含数据集特征的字典
        """
        # 1. 样本规模分析
        n_samples = adata.obs[sample_key].nunique()
        n_cells = adata.shape[0]
        n_genes = adata.shape[1]

        # 2. 数据密度分析
        cells_per_sample = n_cells / n_samples

        # 3. 类别平衡性分析
        class_counts = adata.obs[condition_key].value_counts()
        n_classes = len(class_counts)
        class_balance_ratio = class_counts.min() / class_counts.max()

        # 4. 细胞类型多样性（如果提供）
        if groupby is not None:
            n_celltypes = adata.obs[groupby].nunique()
            celltypes_per_sample = adata.obs.groupby([sample_key, groupby]).size().groupby(sample_key).size().mean()
        else:
            n_celltypes = None
            celltypes_per_sample = None

        # 5. 任务复杂度估计（快速基线模型）
        complexity_score, complexity_level = self._estimate_task_complexity(
            adata, sample_key, condition_key
        )

        # 6. 规模分类
        if n_samples < 20:
            scale = 'small'
        elif n_samples < 100:
            scale = 'medium'
        else:
            scale = 'large'

        results = {
            # 基本统计
            'n_samples': n_samples,
            'n_cells': n_cells,
            'n_genes': n_genes,
            'cells_per_sample': cells_per_sample,

            # 类别信息
            'n_classes': n_classes,
            'class_balance_ratio': class_balance_ratio,
            'class_counts': class_counts.to_dict(),

            # 细胞类型多样性
            'n_celltypes': n_celltypes,
            'celltypes_per_sample': celltypes_per_sample,

            # 复杂度评估
            'complexity_score': complexity_score,
            'complexity_level': complexity_level,

            # 规模分类
            'scale': scale
        }

        if self.verbose:
            self._print_analysis(results)

        return results

    def _estimate_task_complexity(self, adata, sample_key, condition_key, max_time_seconds=10):
        """
        使用快速基线模型估计任务复杂度

        策略：
        - 提取简单特征（样本级别的基本统计量）
        - 用逻辑回归快速评估
        - 根据AUROC判断任务难度

        Parameters
        ----------
        adata : AnnData
            单细胞数据对象
        sample_key : str
            样本列名
        condition_key : str
            条件/分组列名
        max_time_seconds : int
            最大运行时间（秒）

        Returns
        -------
        score : float
            基线AUROC分数
        level : str
            复杂度等级 ('low', 'medium', 'high')
        """
        try:
            # 提取简单特征：每个样本的基本统计量
            features = []
            labels = []
            samples = adata.obs[sample_key].unique()

            for sample in samples:
                mask = adata.obs[sample_key] == sample
                sample_data = adata[mask]

                # 特征1: 细胞数量
                n_cells = mask.sum()

                # 特征2-4: 基因表达统计（mean, std, median of means）
                if hasattr(sample_data.X, 'toarray'):
                    X_dense = sample_data.X.toarray()
                else:
                    X_dense = sample_data.X

                mean_expr = X_dense.mean(axis=0).mean()
                std_expr = X_dense.std(axis=0).mean()
                median_expr = np.median(X_dense, axis=0).mean()

                # 特征5: 非零基因比例
                nonzero_ratio = (X_dense > 0).mean()

                features.append([n_cells, mean_expr, std_expr, median_expr, nonzero_ratio])
                labels.append(sample_data.obs[condition_key].iloc[0])

            X = np.array(features)
            y = labels

            # 标签编码
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            # 快速逻辑回归（限制迭代次数）
            if len(np.unique(y_encoded)) < 2:
                # 只有一个类别，无法评估
                return 0.5, 'unknown'

            lr = LogisticRegression(
                max_iter=100,
                solver='lbfgs',
                random_state=42
            )

            # 3折交叉验证（快速）
            n_splits = min(3, len(X) // 2)
            if n_splits < 2:
                # 样本太少，无法交叉验证
                lr.fit(X, y_encoded)
                baseline_auroc = 0.5  # 默认值
            else:
                scores = cross_val_score(
                    lr, X, y_encoded,
                    cv=n_splits,
                    scoring='roc_auc'
                )
                baseline_auroc = scores.mean()

            # 根据AUROC判断复杂度
            if baseline_auroc > 0.85:
                complexity_level = 'low'  # 简单任务（如kuppe）
            elif baseline_auroc > 0.70:
                complexity_level = 'medium'  # 中等任务（如carraro）
            else:
                complexity_level = 'high'  # 困难任务（如velmeshev）

            return float(baseline_auroc), complexity_level

        except Exception as e:
            if self.verbose:
                print(f"⚠️ 复杂度估计失败: {e}")
                print("  使用默认复杂度: medium")
            return 0.7, 'medium'

    def _print_analysis(self, results):
        """打印分析结果"""
        print("\n" + "=" * 80)
        print(" " * 25 + "数据集复杂度分析报告")
        print("=" * 80)

        print(f"\n【基本统计】")
        print(f"  样本数量: {results['n_samples']}")
        print(f"  细胞总数: {results['n_cells']:,}")
        print(f"  基因数量: {results['n_genes']:,}")
        print(f"  平均每样本细胞数: {results['cells_per_sample']:.1f}")

        print(f"\n【类别信息】")
        print(f"  类别数量: {results['n_classes']}")
        print(f"  类别平衡比: {results['class_balance_ratio']:.2f}")
        for cls, count in results['class_counts'].items():
            print(f"    - {cls}: {count} 样本")

        if results['n_celltypes'] is not None:
            print(f"\n【细胞类型多样性】")
            print(f"  细胞类型总数: {results['n_celltypes']}")
            print(f"  平均每样本细胞类型数: {results['celltypes_per_sample']:.1f}")

        print(f"\n【复杂度评估】")
        print(f"  样本规模: {results['scale'].upper()}")
        print(f"  基线AUROC: {results['complexity_score']:.4f}")
        print(f"  任务复杂度: {results['complexity_level'].upper()}")

        # 复杂度说明
        complexity_desc = {
            'low': '✅ 简单任务，类别容易区分',
            'medium': '⚡ 中等任务，需要适当的模型复杂度',
            'high': '⚠️ 困难任务，需要复杂模型和特殊策略'
        }
        print(f"  说明: {complexity_desc.get(results['complexity_level'], '未知')}")

        print("=" * 80 + "\n")


class AdaptiveHGATLinkConfig:
    """
    自适应配置生成器

    根据数据集特征自动生成最优的HGATLink配置参数
    """

    def __init__(self):
        # 配置规则矩阵
        self.config_rules = {
            # 小数据集规则 (样本数 < 20)
            'small': {
                'low': {  # 简单任务
                    'n_factors': 3,
                    'hidden_dim': 32,
                    'dropout': 0.3,
                    'cv_folds': 2,
                    'max_lr_pairs': 1000,
                    'l2_reg': 0.001,
                    'description': '小数据集+简单任务：最简配置'
                },
                'medium': {  # 中等任务
                    'n_factors': 4,
                    'hidden_dim': 32,
                    'dropout': 0.5,
                    'cv_folds': 2,
                    'max_lr_pairs': 800,
                    'l2_reg': 0.01,
                    'description': '小数据集+中等任务：强正则化'
                },
                'high': {  # 困难任务
                    'n_factors': 5,
                    'hidden_dim': 48,
                    'dropout': 0.5,
                    'cv_folds': 2,
                    'max_lr_pairs': 600,
                    'l2_reg': 0.01,
                    'description': '小数据集+困难任务：增加容量但强正则化'
                }
            },
            # 中等数据集规则 (20 <= 样本数 < 100)
            'medium': {
                'low': {
                    'n_factors': 5,
                    'hidden_dim': 48,
                    'dropout': 0.2,
                    'cv_folds': 3,
                    'max_lr_pairs': 800,
                    'l2_reg': 0.0001,
                    'description': '中等数据集+简单任务：标准配置'
                },
                'medium': {
                    'n_factors': 6,
                    'hidden_dim': 64,
                    'dropout': 0.3,
                    'cv_folds': 3,
                    'max_lr_pairs': 500,
                    'l2_reg': 0.001,
                    'description': '中等数据集+中等任务：平衡配置'
                },
                'high': {
                    'n_factors': 12,
                    'hidden_dim': 128,
                    'dropout': 0.2,
                    'cv_folds': 3,
                    'max_lr_pairs': 1000,
                    'l2_reg': 0.0005,
                    'description': '中等数据集+困难任务：大幅增强容量（优化for VELMESHEV）'
                }
            },
            # 大数据集规则 (样本数 >= 100)
            'large': {
                'low': {
                    'n_factors': 8,
                    'hidden_dim': 96,
                    'dropout': 0.1,
                    'cv_folds': 5,
                    'max_lr_pairs': 500,
                    'l2_reg': 0.0001,
                    'description': '大数据集+简单任务：大容量低正则化'
                },
                'medium': {
                    'n_factors': 10,
                    'hidden_dim': 128,
                    'dropout': 0.2,
                    'cv_folds': 5,
                    'max_lr_pairs': 300,
                    'l2_reg': 0.0001,
                    'description': '大数据集+中等任务：标准大模型'
                },
                'high': {
                    'n_factors': 12,
                    'hidden_dim': 128,
                    'dropout': 0.2,
                    'cv_folds': 5,
                    'max_lr_pairs': 200,
                    'l2_reg': 0.001,
                    'description': '大数据集+困难任务：最大容量'
                }
            }
        }

    def generate_config(self, dataset_analysis, dataset_name=None, verbose=True):
        """
        生成自适应配置

        Parameters
        ----------
        dataset_analysis : dict
            DatasetComplexityAnalyzer的分析结果
        dataset_name : str, optional
            数据集名称（用于日志）
        verbose : bool
            是否打印配置信息

        Returns
        -------
        dict
            优化后的配置参数
        """
        scale = dataset_analysis['scale']
        complexity = dataset_analysis['complexity_level']

        # 获取基础配置
        base_config = self.config_rules[scale][complexity].copy()

        # 根据样本数微调
        n_samples = dataset_analysis['n_samples']

        # 微调1: 样本数接近边界时的平滑过渡
        if scale == 'small' and n_samples >= 15:
            # 接近中等数据集，略微增加容量
            base_config['n_factors'] = min(base_config['n_factors'] + 1, 6)
            base_config['dropout'] *= 0.9  # 略微降低dropout

        elif scale == 'medium' and n_samples >= 80:
            # 接近大数据集
            base_config['n_factors'] = min(base_config['n_factors'] + 2, 10)
            base_config['dropout'] *= 0.9

        # 微调2: 根据类别平衡调整
        class_balance_ratio = dataset_analysis['class_balance_ratio']
        if class_balance_ratio < 0.5:  # 不平衡数据集
            # 增加正则化
            base_config['l2_reg'] *= 1.5
            if verbose:
                print(f"  ⚡ 检测到类别不平衡（比例={class_balance_ratio:.2f}），增加L2正则化")

        # 微调3: 根据数据密度调整
        cells_per_sample = dataset_analysis['cells_per_sample']
        if cells_per_sample > 50000:  # 高密度数据（如velmeshev）
            # 可能需要更多特征提取能力
            if verbose:
                print(f"  ⚡ 检测到高密度数据（{cells_per_sample:.0f}细胞/样本），增加模型容量")
            base_config['hidden_dim'] = int(base_config['hidden_dim'] * 1.3)

        # 添加元信息
        base_config['dataset_name'] = dataset_name
        base_config['dataset_scale'] = scale
        base_config['dataset_complexity'] = complexity
        base_config['n_samples'] = n_samples

        if verbose:
            self._print_config(base_config, dataset_analysis)

        return base_config

    def _print_config(self, config, analysis):
        """打印配置信息"""
        print("\n" + "=" * 80)
        print(" " * 25 + "自适应配置生成结果")
        print("=" * 80)

        if config.get('dataset_name'):
            print(f"\n【数据集】: {config['dataset_name']}")

        print(f"\n【配置策略】: {config['description']}")
        print(f"  数据规模: {config['dataset_scale'].upper()} ({config['n_samples']} 样本)")
        print(f"  任务复杂度: {config['dataset_complexity'].upper()}")

        print(f"\n【核心参数】")
        print(f"  n_factors (降维维度): {config['n_factors']}")
        print(f"  hidden_dim (隐藏层维度): {config['hidden_dim']}")
        print(f"  dropout (Dropout率): {config['dropout']}")
        print(f"  l2_reg (L2正则化): {config['l2_reg']}")

        print(f"\n【训练参数】")
        print(f"  cv_folds (交叉验证折数): {config['cv_folds']}")
        print(f"  max_lr_pairs (最大LR对数): {config['max_lr_pairs']}")

        # 与默认配置对比
        default_n_factors = 10
        if config['n_factors'] < default_n_factors:
            print(f"\n【优化说明】")
            print(f"  ✓ n_factors从默认{default_n_factors}降低到{config['n_factors']}")
            print(f"    原因: {'小样本，避免过拟合' if config['dataset_scale'] == 'small' else '适配数据集规模'}")

        print("=" * 80 + "\n")

    def compare_configs(self, original_config, adaptive_config):
        """
        对比原始配置和自适应配置

        Parameters
        ----------
        original_config : dict
            原始配置
        adaptive_config : dict
            自适应配置

        Returns
        -------
        dict
            差异分析
        """
        differences = {}

        key_params = ['n_factors', 'hidden_dim', 'dropout', 'cv_folds']

        for param in key_params:
            if param in original_config and param in adaptive_config:
                orig = original_config[param]
                adapt = adaptive_config[param]

                if orig != adapt:
                    change_pct = ((adapt - orig) / orig) * 100 if orig != 0 else 0
                    differences[param] = {
                        'original': orig,
                        'adaptive': adapt,
                        'change': adapt - orig,
                        'change_pct': change_pct
                    }

        return differences


def get_adaptive_config_for_dataset(adata, dataset_name, sample_key, condition_key, groupby=None, verbose=True):
    """
    便捷函数：一键获取数据集的自适应配置

    Parameters
    ----------
    adata : AnnData
        单细胞数据对象
    dataset_name : str
        数据集名称
    sample_key : str
        样本列名
    condition_key : str
        条件列名
    groupby : str, optional
        细胞类型列名
    verbose : bool
        是否打印详细信息

    Returns
    -------
    dict
        自适应配置
    """
    # 步骤1: 分析数据集
    analyzer = DatasetComplexityAnalyzer(verbose=verbose)
    analysis = analyzer.analyze(adata, sample_key, condition_key, groupby)

    # 步骤2: 生成配置
    config_generator = AdaptiveHGATLinkConfig()
    config = config_generator.generate_config(analysis, dataset_name, verbose)

    return config, analysis


# 示例用法
if __name__ == '__main__':
    print("自适应配置系统 - 示例")
    print("\n使用方法:")
    print("""
    from adaptive_config import get_adaptive_config_for_dataset
    import scanpy as sc

    # 加载数据
    adata = sc.read_h5ad('data/carraro.h5ad')

    # 获取自适应配置
    config, analysis = get_adaptive_config_for_dataset(
        adata,
        dataset_name='carraro',
        sample_key='orig.ident',
        condition_key='type',
        groupby='major',
        verbose=True
    )

    # 使用配置
    print(f"推荐的n_factors: {config['n_factors']}")
    """)
