#!/usr/bin/env python3
"""
集成学习工具 - HGATLink模型性能提升

通过集成多个模型降低方差，提升预测稳定性和准确性。

核心策略：
1. Bootstrap聚合（Bagging）- 降低方差
2. 多样化集成 - 通过不同策略创建模型差异
3. 加权软投票 - 根据模型性能动态加权

作者: Claude Code
日期: 2026-01-26
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')


class BaggingHGATLink:
    """
    Bootstrap聚合集成

    通过Bootstrap采样训练多个模型，降低方差
    """

    def __init__(self, n_models=3, sample_ratio=0.8, random_state=42, verbose=True):
        """
        Parameters
        ----------
        n_models : int
            模型数量（默认3，平衡性能和计算成本）
        sample_ratio : float
            每个模型使用的样本比例（默认0.8）
        random_state : int
            随机种子
        verbose : bool
            是否打印详细信息
        """
        self.n_models = n_models
        self.sample_ratio = sample_ratio
        self.random_state = random_state
        self.verbose = verbose

        self.models = []
        self.weights = []
        self.oob_scores = []

    def fit(self, X, y, base_estimator=None):
        """
        训练集成模型

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            训练特征
        y : array-like, shape (n_samples,)
            训练标签
        base_estimator : estimator object, optional
            基础模型（默认使用RandomForest）
        """
        if self.verbose:
            print("\n" + "🎯 " * 30)
            print(f"Bootstrap集成训练: {self.n_models}个模型")
            print("🎯 " * 30)

        n_samples = len(X)
        bootstrap_size = int(n_samples * self.sample_ratio)

        for i in range(self.n_models):
            if self.verbose:
                print(f"\n训练模型 {i+1}/{self.n_models}...")

            # Bootstrap采样（有放回）
            np.random.seed(self.random_state + i)
            indices = np.random.choice(n_samples, size=bootstrap_size, replace=True)
            X_boot, y_boot = X[indices], y[indices]

            # 创建基础模型
            if base_estimator is None:
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    random_state=self.random_state + i,
                    oob_score=True,  # 启用OOB评分
                    n_jobs=-1
                )
            else:
                model = clone(base_estimator)
                if hasattr(model, 'random_state'):
                    model.random_state = self.random_state + i

            # 训练
            model.fit(X_boot, y_boot)

            # 保存模型和OOB分数
            self.models.append(model)

            if hasattr(model, 'oob_score_'):
                oob_score = model.oob_score_
                self.oob_scores.append(oob_score)
                if self.verbose:
                    print(f"  ✓ OOB分数: {oob_score:.4f}")
            else:
                self.oob_scores.append(1.0)  # 默认权重

        # 计算权重（基于OOB分数）
        self._compute_weights()

        if self.verbose:
            print("\n✅ 集成训练完成")
            print(f"模型数量: {len(self.models)}")
            print(f"权重分布: {[f'{w:.3f}' for w in self.weights]}")
            print("🎯 " * 30 + "\n")

    def _compute_weights(self):
        """根据OOB分数计算权重"""
        oob_array = np.array(self.oob_scores)

        # 归一化权重
        if oob_array.sum() > 0:
            self.weights = oob_array / oob_array.sum()
        else:
            # 如果所有OOB分数都是0，使用均匀权重
            self.weights = np.ones(self.n_models) / self.n_models

    def predict_proba(self, X):
        """
        预测概率（加权软投票）

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            测试特征

        Returns
        -------
        proba : array, shape (n_samples, n_classes)
            预测概率
        """
        if not self.models:
            raise ValueError("模型未训练，请先调用fit()")

        # 收集所有模型的预测
        all_probas = []
        for model in self.models:
            proba = model.predict_proba(X)
            all_probas.append(proba)

        # 加权平均
        all_probas = np.array(all_probas)  # shape: (n_models, n_samples, n_classes)
        weighted_proba = np.average(all_probas, axis=0, weights=self.weights)

        return weighted_proba

    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def save(self, filepath):
        """保存集成模型"""
        model_data = {
            'models': self.models,
            'weights': self.weights,
            'oob_scores': self.oob_scores,
            'n_models': self.n_models,
            'sample_ratio': self.sample_ratio,
            'random_state': self.random_state
        }
        joblib.dump(model_data, filepath)
        if self.verbose:
            print(f"✓ 模型已保存到: {filepath}")

    def load(self, filepath):
        """加载集成模型"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.weights = model_data['weights']
        self.oob_scores = model_data['oob_scores']
        self.n_models = model_data['n_models']
        self.sample_ratio = model_data['sample_ratio']
        self.random_state = model_data['random_state']
        if self.verbose:
            print(f"✓ 模型已从{filepath}加载")


class DiversifiedEnsemble:
    """
    多样化集成

    通过不同策略创建模型差异：
    1. 不同随机种子
    2. 不同超参数
    3. 不同特征子集
    4. 不同样本权重
    """

    def __init__(self, n_models=5, diversity_strategy='all', random_state=42, verbose=True):
        """
        Parameters
        ----------
        n_models : int
            模型数量
        diversity_strategy : str
            多样化策略 ('random_seed', 'hyperparameter', 'feature_subset', 'all')
        random_state : int
            随机种子
        verbose : bool
            是否打印详细信息
        """
        self.n_models = n_models
        self.diversity_strategy = diversity_strategy
        self.random_state = random_state
        self.verbose = verbose

        self.models = []
        self.feature_indices = []  # 每个模型使用的特征索引
        self.cv_scores = []  # 交叉验证分数
        self.weights = []

    def fit(self, X, y):
        """训练多样化集成"""
        if self.verbose:
            print("\n" + "🌟 " * 30)
            print(f"多样化集成训练: {self.n_models}个模型")
            print(f"多样化策略: {self.diversity_strategy.upper()}")
            print("🌟 " * 30)

        n_features = X.shape[1]

        for i in range(self.n_models):
            if self.verbose:
                print(f"\n训练模型 {i+1}/{self.n_models}...")

            # 创建多样化模型
            model, feature_mask = self._create_diverse_model(i, n_features)

            # 选择特征子集（如果使用特征多样化）
            if feature_mask is not None:
                X_subset = X[:, feature_mask]
                self.feature_indices.append(feature_mask)
            else:
                X_subset = X
                self.feature_indices.append(None)

            # 训练
            model.fit(X_subset, y)

            # 交叉验证评估
            cv_score = cross_val_score(model, X_subset, y, cv=3, scoring='roc_auc').mean()
            self.cv_scores.append(cv_score)

            self.models.append(model)

            if self.verbose:
                print(f"  ✓ CV AUROC: {cv_score:.4f}")

        # 计算权重
        self._compute_weights()

        if self.verbose:
            print("\n✅ 多样化集成训练完成")
            print(f"模型数量: {len(self.models)}")
            print(f"平均CV分数: {np.mean(self.cv_scores):.4f}")
            print(f"权重分布: {[f'{w:.3f}' for w in self.weights]}")
            print("🌟 " * 30 + "\n")

    def _create_diverse_model(self, model_idx, n_features):
        """
        创建多样化模型

        Returns
        -------
        model : estimator
            模型
        feature_mask : array or None
            特征掩码（如果使用特征子集）
        """
        np.random.seed(self.random_state + model_idx)

        feature_mask = None

        if self.diversity_strategy == 'random_seed':
            # 策略1: 只改变随机种子
            model = RandomForestClassifier(
                n_estimators=150,
                max_depth=None,
                random_state=self.random_state + model_idx,
                n_jobs=-1
            )

        elif self.diversity_strategy == 'hyperparameter':
            # 策略2: 不同超参数
            n_estimators_choices = [80, 100, 120, 150, 200]
            max_depth_choices = [10, 15, 20, None]
            min_samples_split_choices = [2, 5, 10]

            model = RandomForestClassifier(
                n_estimators=np.random.choice(n_estimators_choices),
                max_depth=np.random.choice(max_depth_choices),
                min_samples_split=np.random.choice(min_samples_split_choices),
                random_state=self.random_state + model_idx,
                n_jobs=-1
            )

            if self.verbose:
                print(f"  超参数: n_estimators={model.n_estimators}, "
                      f"max_depth={model.max_depth}, "
                      f"min_samples_split={model.min_samples_split}")

        elif self.diversity_strategy == 'feature_subset':
            # 策略3: 随机特征子集（70-100%特征）
            feature_ratio = np.random.uniform(0.7, 1.0)
            n_features_subset = max(int(n_features * feature_ratio), 1)
            feature_mask = np.random.choice(n_features, size=n_features_subset, replace=False)
            feature_mask = np.sort(feature_mask)

            model = RandomForestClassifier(
                n_estimators=150,
                max_depth=None,
                random_state=self.random_state + model_idx,
                n_jobs=-1
            )

            if self.verbose:
                print(f"  特征子集: {n_features_subset}/{n_features} ({feature_ratio:.1%})")

        elif self.diversity_strategy == 'all':
            # 策略4: 组合所有多样化策略
            # 超参数
            n_estimators_choices = [80, 100, 150]
            max_depth_choices = [15, 20, None]

            model = RandomForestClassifier(
                n_estimators=np.random.choice(n_estimators_choices),
                max_depth=np.random.choice(max_depth_choices),
                random_state=self.random_state + model_idx,
                n_jobs=-1
            )

            # 特征子集
            if model_idx % 2 == 0:  # 一半模型使用特征子集
                feature_ratio = np.random.uniform(0.7, 0.9)
                n_features_subset = max(int(n_features * feature_ratio), 1)
                feature_mask = np.random.choice(n_features, size=n_features_subset, replace=False)
                feature_mask = np.sort(feature_mask)

                if self.verbose:
                    print(f"  配置: n_est={model.n_estimators}, "
                          f"depth={model.max_depth}, "
                          f"features={n_features_subset}/{n_features}")
            else:
                if self.verbose:
                    print(f"  配置: n_est={model.n_estimators}, "
                          f"depth={model.max_depth}, "
                          f"features=all")

        else:
            # 默认: 只改变随机种子
            model = RandomForestClassifier(
                n_estimators=150,
                random_state=self.random_state + model_idx,
                n_jobs=-1
            )

        return model, feature_mask

    def _compute_weights(self):
        """根据CV分数计算权重"""
        cv_array = np.array(self.cv_scores)

        # 归一化权重
        if cv_array.sum() > 0:
            self.weights = cv_array / cv_array.sum()
        else:
            self.weights = np.ones(self.n_models) / self.n_models

    def predict_proba(self, X):
        """预测概率（加权软投票）"""
        if not self.models:
            raise ValueError("模型未训练，请先调用fit()")

        all_probas = []
        for i, model in enumerate(self.models):
            # 使用相应的特征子集
            if self.feature_indices[i] is not None:
                X_subset = X[:, self.feature_indices[i]]
            else:
                X_subset = X

            proba = model.predict_proba(X_subset)
            all_probas.append(proba)

        # 加权平均
        all_probas = np.array(all_probas)
        weighted_proba = np.average(all_probas, axis=0, weights=self.weights)

        return weighted_proba

    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


class FastEnsemble:
    """
    快速集成（计算成本最小化）

    策略：
    - 只训练3个模型（而非5个）
    - 使用较少的树（每个模型100棵）
    - 简单的随机种子多样化
    """

    def __init__(self, n_models=3, n_estimators=100, random_state=42, verbose=True):
        """
        Parameters
        ----------
        n_models : int
            模型数量（默认3）
        n_estimators : int
            每个模型的树数量（默认100）
        random_state : int
            随机种子
        verbose : bool
            是否打印详细信息
        """
        self.n_models = n_models
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.verbose = verbose

        self.models = []

    def fit(self, X, y):
        """训练快速集成"""
        if self.verbose:
            print("\n" + "⚡ " * 30)
            print(f"快速集成训练: {self.n_models}个模型 × {self.n_estimators}棵树")
            print("⚡ " * 30)

        for i in range(self.n_models):
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=None,
                random_state=self.random_state + i,
                n_jobs=-1
            )

            model.fit(X, y)
            self.models.append(model)

            if self.verbose:
                print(f"✓ 模型{i+1}训练完成")

        if self.verbose:
            print("⚡ " * 30 + "\n")

    def predict_proba(self, X):
        """预测概率（简单平均）"""
        all_probas = np.array([model.predict_proba(X) for model in self.models])
        return all_probas.mean(axis=0)

    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def compare_ensemble_methods(X_train, y_train, X_test, y_test, verbose=True):
    """
    对比不同集成方法的效果

    Parameters
    ----------
    X_train, y_train : array-like
        训练数据
    X_test, y_test : array-like
        测试数据
    verbose : bool
        是否打印详细信息

    Returns
    -------
    results : dict
        各方法的性能对比
    """
    from sklearn.metrics import roc_auc_score, accuracy_score

    results = {}

    methods = {
        'Baseline (单模型)': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'Bootstrap集成': BaggingHGATLink(n_models=3, verbose=verbose),
        '多样化集成': DiversifiedEnsemble(n_models=3, diversity_strategy='all', verbose=verbose),
        '快速集成': FastEnsemble(n_models=3, n_estimators=100, verbose=verbose)
    }

    if verbose:
        print("\n" + "=" * 80)
        print(" " * 30 + "集成方法对比")
        print("=" * 80)

    for name, model in methods.items():
        if verbose:
            print(f"\n测试: {name}")

        # 训练
        model.fit(X_train, y_train)

        # 预测
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.predict(X_test)

        y_pred = model.predict(X_test)

        # 评估
        auroc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)

        results[name] = {
            'AUROC': auroc,
            'Accuracy': accuracy
        }

        if verbose:
            print(f"  AUROC: {auroc:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")

    if verbose:
        print("\n" + "=" * 80)
        print("对比总结")
        print("=" * 80)

        baseline_auroc = results['Baseline (单模型)']['AUROC']

        for name, metrics in results.items():
            if name != 'Baseline (单模型)':
                improvement = metrics['AUROC'] - baseline_auroc
                print(f"{name:20} AUROC改进: {improvement:+.4f} ({improvement/baseline_auroc*100:+.2f}%)")

    return results


# 示例用法
if __name__ == '__main__':
    print("集成学习工具 - 使用示例")
    print("\n基本用法:")
    print("""
    from ensemble_utils import BaggingHGATLink, DiversifiedEnsemble, FastEnsemble

    # 方法1: Bootstrap集成（推荐用于小数据集）
    ensemble = BaggingHGATLink(n_models=3)
    ensemble.fit(X_train, y_train)
    y_pred_proba = ensemble.predict_proba(X_test)

    # 方法2: 多样化集成（推荐用于中大数据集）
    ensemble = DiversifiedEnsemble(n_models=5, diversity_strategy='all')
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)

    # 方法3: 快速集成（计算成本最小）
    ensemble = FastEnsemble(n_models=3, n_estimators=100)
    ensemble.fit(X_train, y_train)
    y_pred_proba = ensemble.predict_proba(X_test)

    # 保存/加载模型
    ensemble.save('models/ensemble.pkl')
    ensemble.load('models/ensemble.pkl')
    """)
