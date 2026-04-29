#!/usr/bin/env python3
"""
数据增强工具 - 扩充小数据集

专门针对配受体相互作用数据的增强策略：
1. SMOTE过采样 - 合成少数类样本
2. Mixup - 混合同类样本
3. 高斯噪声 - 添加技术噪声模拟
4. 特征Dropout - 随机丢弃特征增强鲁棒性

作者: Claude Code
日期: 2026-01-26
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')


class LRInteractionAugmentation:
    """
    配受体相互作用数据增强器

    针对小数据集（样本数<30）的增强策略
    """

    def __init__(self, random_state=42, verbose=True):
        """
        Parameters
        ----------
        random_state : int
            随机种子
        verbose : bool
            是否打印详细信息
        """
        self.random_state = random_state
        self.verbose = verbose
        np.random.seed(random_state)

    def augment(self, X, y, target_samples=None, methods=['smote', 'mixup', 'noise']):
        """
        综合数据增强

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            原始特征
        y : array-like, shape (n_samples,)
            原始标签
        target_samples : int, optional
            目标样本数（默认翻倍）
        methods : list
            使用的增强方法列表

        Returns
        -------
        X_aug : array
            增强后的特征
        y_aug : array
            增强后的标签
        """
        if target_samples is None:
            target_samples = len(X) * 2

        n_original = len(X)

        if self.verbose:
            print("\n" + "🔬 " * 30)
            print(f"数据增强: {n_original} → {target_samples} 样本")
            print(f"方法: {', '.join(methods)}")
            print("🔬 " * 30)

        X_aug = X.copy()
        y_aug = y.copy()

        remaining_samples = target_samples - n_original

        if remaining_samples <= 0:
            if self.verbose:
                print("  目标样本数已达到，跳过增强")
            return X_aug, y_aug

        # 按顺序应用增强方法
        for method in methods:
            if remaining_samples <= 0:
                break

            n_to_generate = remaining_samples // len(methods) + 1

            if method == 'smote':
                X_new, y_new = self._smote_augment(X, y, n_to_generate)
            elif method == 'mixup':
                X_new, y_new = self._mixup_augment(X, y, n_to_generate)
            elif method == 'noise':
                X_new, y_new = self._gaussian_noise_augment(X, y, n_to_generate)
            elif method == 'dropout':
                X_new, y_new = self._feature_dropout_augment(X, y, n_to_generate)
            else:
                if self.verbose:
                    print(f"  警告: 未知方法 '{method}'，跳过")
                continue

            X_aug = np.vstack([X_aug, X_new])
            y_aug = np.concatenate([y_aug, y_new])
            remaining_samples -= len(X_new)

            if self.verbose:
                print(f"  ✓ {method:12} 生成 {len(X_new):3} 样本")

        # 截断到目标样本数
        if len(X_aug) > target_samples:
            indices = np.random.choice(len(X_aug), target_samples, replace=False)
            X_aug = X_aug[indices]
            y_aug = y_aug[indices]

        if self.verbose:
            print(f"\n✅ 增强完成: {n_original} → {len(X_aug)} 样本")
            print("🔬 " * 30 + "\n")

        return X_aug, y_aug

    def _smote_augment(self, X, y, n_samples):
        """
        SMOTE过采样

        在特征空间中合成新样本

        Parameters
        ----------
        X : array
            特征
        y : array
            标签
        n_samples : int
            生成样本数

        Returns
        -------
        X_new : array
            新样本特征
        y_new : array
            新样本标签
        """
        try:
            from imblearn.over_sampling import SMOTE

            # 确定k_neighbors
            min_class_count = min(np.bincount(y.astype(int)))
            k_neighbors = min(5, min_class_count - 1)

            if k_neighbors < 1:
                # 样本太少，无法使用SMOTE
                if self.verbose:
                    print(f"    警告: 样本太少({min_class_count})，SMOTE回退到简单复制")
                return self._simple_oversample(X, y, n_samples)

            smote = SMOTE(
                k_neighbors=k_neighbors,
                random_state=self.random_state
            )

            # 计算采样策略（保持原始比例）
            class_counts = np.bincount(y.astype(int))
            total = len(y) + n_samples

            sampling_strategy = {}
            for cls in range(len(class_counts)):
                target_count = int(total * (class_counts[cls] / len(y)))
                sampling_strategy[cls] = max(class_counts[cls], target_count)

            smote.sampling_strategy = sampling_strategy

            X_resampled, y_resampled = smote.fit_resample(X, y)

            # 只返回新生成的样本
            X_new = X_resampled[len(X):]
            y_new = y_resampled[len(y):]

            # 如果生成的样本数不够，补充
            if len(X_new) < n_samples:
                additional = n_samples - len(X_new)
                X_add, y_add = self._simple_oversample(X, y, additional)
                X_new = np.vstack([X_new, X_add])
                y_new = np.concatenate([y_new, y_add])

            # 如果生成的样本数过多，截断
            if len(X_new) > n_samples:
                indices = np.random.choice(len(X_new), n_samples, replace=False)
                X_new = X_new[indices]
                y_new = y_new[indices]

            return X_new, y_new

        except ImportError:
            if self.verbose:
                print("    警告: imblearn未安装，回退到简单过采样")
            return self._simple_oversample(X, y, n_samples)

        except Exception as e:
            if self.verbose:
                print(f"    警告: SMOTE失败({e})，回退到简单过采样")
            return self._simple_oversample(X, y, n_samples)

    def _mixup_augment(self, X, y, n_samples, alpha=0.2):
        """
        Mixup数据增强

        混合同类样本生成新样本

        Parameters
        ----------
        X : array
            特征
        y : array
            标签
        n_samples : int
            生成样本数
        alpha : float
            Beta分布参数（控制混合比例）

        Returns
        -------
        X_new : array
            新样本特征
        y_new : array
            新样本标签
        """
        X_new = []
        y_new = []

        # 按类别分组
        unique_classes = np.unique(y)
        class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}

        for _ in range(n_samples):
            # 随机选择一个类别
            cls = np.random.choice(unique_classes)
            cls_indices = class_indices[cls]

            if len(cls_indices) < 2:
                # 该类只有一个样本，直接复制
                idx = cls_indices[0]
                X_new.append(X[idx])
                y_new.append(y[idx])
                continue

            # 随机选择同类的两个样本
            idx1, idx2 = np.random.choice(cls_indices, size=2, replace=False)

            # Beta分布采样混合权重
            lambda_mix = np.random.beta(alpha, alpha)

            # 混合
            x_mixed = lambda_mix * X[idx1] + (1 - lambda_mix) * X[idx2]

            X_new.append(x_mixed)
            y_new.append(y[idx1])  # 标签保持不变

        return np.array(X_new), np.array(y_new)

    def _gaussian_noise_augment(self, X, y, n_samples, noise_level=0.1):
        """
        高斯噪声增强

        添加高斯噪声模拟技术变异

        Parameters
        ----------
        X : array
            特征
        y : array
            标签
        n_samples : int
            生成样本数
        noise_level : float
            噪声水平（相对于标准差）

        Returns
        -------
        X_new : array
            新样本特征
        y_new : array
            新样本标签
        """
        # 计算每个特征的标准差
        std_devs = X.std(axis=0)

        X_new = []
        y_new = []

        for _ in range(n_samples):
            # 随机选择一个原始样本
            idx = np.random.randint(len(X))

            # 添加高斯噪声
            noise = np.random.normal(0, std_devs * noise_level)
            x_noisy = X[idx] + noise

            # 确保非负（如果原始数据非负）
            if (X >= 0).all():
                x_noisy = np.maximum(x_noisy, 0)

            X_new.append(x_noisy)
            y_new.append(y[idx])

        return np.array(X_new), np.array(y_new)

    def _feature_dropout_augment(self, X, y, n_samples, dropout_rate=0.2):
        """
        特征Dropout增强

        随机丢弃部分特征（设为0或均值）

        Parameters
        ----------
        X : array
            特征
        y : array
            标签
        n_samples : int
            生成样本数
        dropout_rate : float
            Dropout比例

        Returns
        -------
        X_new : array
            新样本特征
        y_new : array
            新样本标签
        """
        X_new = []
        y_new = []

        n_features = X.shape[1]
        feature_means = X.mean(axis=0)

        for _ in range(n_samples):
            # 随机选择一个原始样本
            idx = np.random.randint(len(X))
            x = X[idx].copy()

            # 随机dropout特征
            dropout_mask = np.random.binomial(1, dropout_rate, n_features)
            x[dropout_mask == 1] = feature_means[dropout_mask == 1]

            X_new.append(x)
            y_new.append(y[idx])

        return np.array(X_new), np.array(y_new)

    def _simple_oversample(self, X, y, n_samples):
        """
        简单过采样（有放回随机采样）

        用于SMOTE失败时的备用方案

        Parameters
        ----------
        X : array
            特征
        y : array
            标签
        n_samples : int
            生成样本数

        Returns
        -------
        X_new : array
            新样本特征
        y_new : array
            新样本标签
        """
        indices = np.random.choice(len(X), size=n_samples, replace=True)
        return X[indices], y[indices]

    def visualize_augmentation(self, X_original, y_original, X_augmented, y_augmented):
        """
        可视化增强效果（使用PCA降维）

        Parameters
        ----------
        X_original : array
            原始特征
        y_original : array
            原始标签
        X_augmented : array
            增强后特征
        y_augmented : array
            增强后标签
        """
        try:
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt

            # PCA降维到2D
            pca = PCA(n_components=2, random_state=self.random_state)
            X_all = np.vstack([X_original, X_augmented])
            X_2d = pca.fit_transform(X_all)

            X_orig_2d = X_2d[:len(X_original)]
            X_aug_2d = X_2d[len(X_original):]

            # 绘图
            plt.figure(figsize=(10, 6))

            # 原始样本
            for cls in np.unique(y_original):
                mask = y_original == cls
                plt.scatter(X_orig_2d[mask, 0], X_orig_2d[mask, 1],
                           label=f'Class {cls} (original)',
                           s=100, alpha=0.7, edgecolors='black')

            # 增强样本
            for cls in np.unique(y_augmented):
                mask = y_augmented == cls
                plt.scatter(X_aug_2d[mask, 0], X_aug_2d[mask, 1],
                           label=f'Class {cls} (augmented)',
                           s=30, alpha=0.3, marker='x')

            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            plt.title('数据增强效果可视化（PCA）')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            return plt.gcf()

        except ImportError:
            print("警告: matplotlib或sklearn未安装，无法可视化")
            return None


class AdaptiveAugmentation:
    """
    自适应数据增强

    根据数据集规模自动选择增强策略
    """

    def __init__(self, random_state=42, verbose=True):
        self.random_state = random_state
        self.verbose = verbose
        self.augmenter = LRInteractionAugmentation(random_state, verbose)

    def augment_adaptive(self, X, y, min_samples=30):
        """
        自适应增强

        Parameters
        ----------
        X : array
            特征
        y : array
            标签
        min_samples : int
            最小目标样本数

        Returns
        -------
        X_aug : array
            增强后特征（如果需要）
        y_aug : array
            增强后标签（如果需要）
        augmented : bool
            是否进行了增强
        """
        n_samples = len(X)

        if n_samples >= min_samples:
            if self.verbose:
                print(f"\n样本数充足({n_samples} >= {min_samples})，跳过增强")
            return X, y, False

        # 根据样本数选择策略
        if n_samples < 10:
            # 极小数据集：激进增强
            target = min_samples * 2
            methods = ['smote', 'mixup', 'noise', 'dropout']
            if self.verbose:
                print(f"\n极小数据集({n_samples})，激进增强 → {target}")

        elif n_samples < 20:
            # 小数据集：标准增强
            target = min_samples
            methods = ['smote', 'mixup', 'noise']
            if self.verbose:
                print(f"\n小数据集({n_samples})，标准增强 → {target}")

        else:
            # 接近阈值：轻微增强
            target = min_samples
            methods = ['smote', 'mixup']
            if self.verbose:
                print(f"\n接近阈值({n_samples})，轻微增强 → {target}")

        X_aug, y_aug = self.augmenter.augment(X, y, target, methods)

        return X_aug, y_aug, True


def check_augmentation_quality(X_train, y_train, X_aug, y_aug, model=None):
    """
    检查数据增强质量

    通过训练/测试差距检测过拟合

    Parameters
    ----------
    X_train : array
        原始训练集
    y_train : array
        原始训练标签
    X_aug : array
        增强后训练集
    y_aug : array
        增强后训练标签
    model : estimator, optional
        模型（默认RandomForest）

    Returns
    -------
    quality_metrics : dict
        质量指标
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 分割测试集（从原始数据）
    X_train_sub, X_test, y_train_sub, y_test = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42
    )

    # 训练1: 只用原始数据
    model1 = RandomForestClassifier(n_estimators=100, random_state=42)
    model1.fit(X_train_sub, y_train_sub)

    y_pred_train1 = model1.predict_proba(X_train_sub)[:, 1]
    y_pred_test1 = model1.predict_proba(X_test)[:, 1]

    auroc_train1 = roc_auc_score(y_train_sub, y_pred_train1)
    auroc_test1 = roc_auc_score(y_test, y_pred_test1)
    gap1 = auroc_train1 - auroc_test1

    # 训练2: 使用增强数据
    model2 = RandomForestClassifier(n_estimators=100, random_state=42)
    model2.fit(X_aug, y_aug)

    y_pred_train2 = model2.predict_proba(X_aug)[:, 1]
    y_pred_test2 = model2.predict_proba(X_test)[:, 1]

    auroc_train2 = roc_auc_score(y_aug, y_pred_train2)
    auroc_test2 = roc_auc_score(y_test, y_pred_test2)
    gap2 = auroc_train2 - auroc_test2

    # 质量评估
    quality_metrics = {
        'original_train_auroc': auroc_train1,
        'original_test_auroc': auroc_test1,
        'original_gap': gap1,
        'augmented_train_auroc': auroc_train2,
        'augmented_test_auroc': auroc_test2,
        'augmented_gap': gap2,
        'test_improvement': auroc_test2 - auroc_test1,
        'gap_change': gap2 - gap1,
        'quality_ok': gap2 < 0.10 and auroc_test2 >= auroc_test1
    }

    return quality_metrics


# 示例用法
if __name__ == '__main__':
    print("数据增强工具 - 使用示例")
    print("\n基本用法:")
    print("""
    from augmentation_utils import LRInteractionAugmentation, AdaptiveAugmentation

    # 方法1: 手动增强
    augmenter = LRInteractionAugmentation()
    X_aug, y_aug = augmenter.augment(
        X, y,
        target_samples=50,
        methods=['smote', 'mixup', 'noise']
    )

    # 方法2: 自适应增强（推荐）
    adaptive_aug = AdaptiveAugmentation()
    X_aug, y_aug, was_augmented = adaptive_aug.augment_adaptive(
        X, y,
        min_samples=30
    )

    # 质量检查
    from augmentation_utils import check_augmentation_quality
    quality = check_augmentation_quality(X, y, X_aug, y_aug)
    print(f"质量OK: {quality['quality_ok']}")
    print(f"测试集改进: {quality['test_improvement']:+.4f}")
    """)
