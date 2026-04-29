#!/usr/bin/env python3
"""
层次化分类器

针对复杂多类别分类任务（如velmeshev的38类细胞类型）的优化策略。
通过两阶段分类降低任务复杂度，提升性能。

Author: Claude Code
Date: 2026-01-26
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def auto_build_hierarchy(adata, groupby, n_groups=4, method='ward', metric='euclidean'):
    """
    自动构建细胞类型层次结构

    使用层次聚类将多个细胞类型分组为少数几个大类

    Parameters
    ----------
    adata : AnnData
        数据对象
    groupby : str
        细胞类型列名
    n_groups : int
        目标分组数量（默认4）
    method : str
        聚类方法（默认'ward'）
    metric : str
        距离度量（默认'euclidean'）

    Returns
    -------
    hierarchy : dict
        {group_id: [celltype1, celltype2, ...], ...}
    celltype_to_group : dict
        {celltype: group_id, ...}
    """

    print(f"\n{'='*60}")
    print(f"构建细胞类型层次结构")
    print(f"{'='*60}\n")

    # 获取所有细胞类型
    celltypes = adata.obs[groupby].unique()
    n_celltypes = len(celltypes)

    print(f"细胞类型数量: {n_celltypes}")
    print(f"目标分组数量: {n_groups}")

    # 计算每个细胞类型的平均表达谱
    celltype_profiles = []
    celltype_names = []

    for celltype in celltypes:
        mask = adata.obs[groupby] == celltype
        if mask.sum() > 0:
            # 使用平均表达作为细胞类型的特征
            profile = np.array(adata.X[mask].mean(axis=0)).flatten()
            celltype_profiles.append(profile)
            celltype_names.append(celltype)

    celltype_profiles = np.array(celltype_profiles)

    print(f"表达谱维度: {celltype_profiles.shape}")

    # 使用层次聚类分组
    if n_celltypes <= n_groups:
        print(f"⚠️ 细胞类型数量({n_celltypes})少于目标分组数({n_groups})，使用每类一组")
        clustering = AgglomerativeClustering(n_clusters=n_celltypes, linkage=method)
    else:
        clustering = AgglomerativeClustering(n_clusters=n_groups, linkage=method)

    labels = clustering.fit_predict(celltype_profiles)

    # 构建层次字典
    hierarchy = defaultdict(list)
    celltype_to_group = {}

    for celltype, group_id in zip(celltype_names, labels):
        hierarchy[group_id].append(celltype)
        celltype_to_group[celltype] = group_id

    # 打印分组结果
    print(f"\n分组结果:")
    print(f"{'-'*60}")
    for group_id, celltypes_in_group in sorted(hierarchy.items()):
        print(f"\n组 {group_id} ({len(celltypes_in_group)} 个细胞类型):")
        for ct in celltypes_in_group:
            n_cells = (adata.obs[groupby] == ct).sum()
            print(f"  - {ct}: {n_cells} 细胞")

    print(f"\n{'='*60}\n")

    return dict(hierarchy), celltype_to_group


class HierarchicalCellTypeClassifier:
    """
    层次化细胞类型分类器

    两阶段分类策略：
    1. 第一层：粗分类（分到大组）
    2. 第二层：细分类（组内细分）

    优势：
    - 降低单次分类的类别数量
    - 第一层高准确率确保整体性能
    - 每个组可以使用独立优化的模型
    """

    def __init__(self, hierarchy=None, celltype_to_group=None,
                 n_estimators=100, verbose=True):
        """
        Parameters
        ----------
        hierarchy : dict
            {group_id: [celltype1, celltype2, ...], ...}
        celltype_to_group : dict
            {celltype: group_id, ...}
        n_estimators : int
            随机森林树的数量
        verbose : bool
            是否打印详细信息
        """
        self.hierarchy = hierarchy or {}
        self.celltype_to_group = celltype_to_group or {}
        self.n_estimators = n_estimators
        self.verbose = verbose

        # 第一层：粗分类器（分到大组）
        self.coarse_classifier = None

        # 第二层：细分类器（每个组一个）
        self.fine_classifiers = {}

        # 标签编码器
        self.label_encoders = {}
        self.group_encoder = LabelEncoder()

    def fit(self, X, y):
        """
        训练层次化分类器

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            特征矩阵
        y : array-like, shape (n_samples,)
            细胞类型标签

        Returns
        -------
        self
        """

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"训练层次化分类器")
            print(f"{'='*60}\n")
            print(f"训练样本数: {len(X)}")
            print(f"细胞类型数: {len(np.unique(y))}")

        # 转换细胞类型到组标签
        y_groups = np.array([self.celltype_to_group[celltype] for celltype in y])

        # 训练第一层：粗分类器
        if self.verbose:
            print(f"\n[第一层] 训练粗分类器...")
            print(f"  分组数: {len(np.unique(y_groups))}")

        self.coarse_classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=42,
            n_jobs=-1,
            oob_score=True,
            class_weight='balanced'
        )

        # 编码组标签
        y_groups_encoded = self.group_encoder.fit_transform(y_groups)
        self.coarse_classifier.fit(X, y_groups_encoded)

        if self.verbose:
            print(f"  训练完成, OOB分数: {self.coarse_classifier.oob_score_:.4f}")

        # 训练第二层：为每个组训练细分类器
        if self.verbose:
            print(f"\n[第二层] 训练细分类器...")

        for group_id, celltypes_in_group in self.hierarchy.items():
            # 只保留属于当前组的样本
            mask = y_groups == group_id

            if mask.sum() == 0:
                if self.verbose:
                    print(f"  组 {group_id}: 跳过（无训练样本）")
                continue

            X_group = X[mask]
            y_group = y[mask]

            # 如果组内只有一个类别，使用简单分类器
            if len(np.unique(y_group)) == 1:
                if self.verbose:
                    print(f"  组 {group_id}: 单类别，使用默认分类器")
                # 创建一个总是返回该类别的简单分类器
                self.fine_classifiers[group_id] = None
                self.label_encoders[group_id] = LabelEncoder().fit(y_group)
                continue

            # 编码标签
            encoder = LabelEncoder()
            y_group_encoded = encoder.fit_transform(y_group)
            self.label_encoders[group_id] = encoder

            # 训练细分类器
            fine_clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=42,
                n_jobs=-1,
                oob_score=True,
                class_weight='balanced',
                max_depth=None
            )

            fine_clf.fit(X_group, y_group_encoded)
            self.fine_classifiers[group_id] = fine_clf

            if self.verbose:
                oob = fine_clf.oob_score_ if hasattr(fine_clf, 'oob_score_') else 0.0
                print(f"  组 {group_id}: {len(celltypes_in_group)} 类, "
                      f"{len(X_group)} 样本, OOB={oob:.4f}")

        if self.verbose:
            print(f"\n{'='*60}\n")

        return self

    def predict(self, X):
        """
        预测细胞类型

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            特征矩阵

        Returns
        -------
        y_pred : array-like, shape (n_samples,)
            预测的细胞类型
        """

        # 第一层：预测大组
        y_groups_pred_encoded = self.coarse_classifier.predict(X)
        y_groups_pred = self.group_encoder.inverse_transform(y_groups_pred_encoded)

        # 第二层：组内细分类
        y_pred = np.empty(len(X), dtype=object)

        for group_id in np.unique(y_groups_pred):
            mask = y_groups_pred == group_id

            if mask.sum() == 0:
                continue

            X_group = X[mask]

            # 获取该组的细分类器
            if group_id not in self.fine_classifiers:
                # 如果没有该组的分类器，使用组内第一个类别
                if group_id in self.hierarchy:
                    y_pred[mask] = self.hierarchy[group_id][0]
                else:
                    y_pred[mask] = "Unknown"
                continue

            fine_clf = self.fine_classifiers[group_id]
            encoder = self.label_encoders[group_id]

            # 单类别组
            if fine_clf is None:
                y_pred[mask] = encoder.classes_[0]
                continue

            # 多类别组
            y_group_pred_encoded = fine_clf.predict(X_group)
            y_group_pred = encoder.inverse_transform(y_group_pred_encoded)
            y_pred[mask] = y_group_pred

        return y_pred

    def predict_proba(self, X):
        """
        预测每个类别的概率

        策略：粗分类概率 × 细分类概率

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            特征矩阵

        Returns
        -------
        proba : array-like, shape (n_samples, n_classes)
            每个类别的概率
        """

        # 第一层：粗分类概率
        y_groups_proba = self.coarse_classifier.predict_proba(X)
        y_groups_pred_encoded = self.coarse_classifier.predict(X)
        y_groups_pred = self.group_encoder.inverse_transform(y_groups_pred_encoded)

        # 收集所有可能的细胞类型
        all_celltypes = []
        for celltypes in self.hierarchy.values():
            all_celltypes.extend(celltypes)
        all_celltypes = sorted(set(all_celltypes))

        # 初始化概率矩阵
        proba = np.zeros((len(X), len(all_celltypes)))
        celltype_to_idx = {ct: i for i, ct in enumerate(all_celltypes)}

        # 第二层：细分类概率
        for group_id in np.unique(y_groups_pred):
            mask = y_groups_pred == group_id

            if mask.sum() == 0:
                continue

            X_group = X[mask]

            # 获取该组的粗分类概率
            group_encoded = self.group_encoder.transform([group_id])[0]
            coarse_proba = y_groups_proba[mask, group_encoded]

            # 获取该组的细分类器
            if group_id not in self.fine_classifiers:
                # 没有细分类器，均匀分配到组内所有类别
                if group_id in self.hierarchy:
                    celltypes_in_group = self.hierarchy[group_id]
                    for ct in celltypes_in_group:
                        idx = celltype_to_idx[ct]
                        proba[mask, idx] = coarse_proba / len(celltypes_in_group)
                continue

            fine_clf = self.fine_classifiers[group_id]
            encoder = self.label_encoders[group_id]

            # 单类别组
            if fine_clf is None:
                ct = encoder.classes_[0]
                idx = celltype_to_idx[ct]
                proba[mask, idx] = coarse_proba
                continue

            # 多类别组
            fine_proba = fine_clf.predict_proba(X_group)

            # 组合概率：粗分类概率 × 细分类概率
            for i, ct in enumerate(encoder.classes_):
                idx = celltype_to_idx[ct]
                proba[mask, idx] = coarse_proba * fine_proba[:, i]

        return proba

    def evaluate(self, X, y):
        """
        评估分类器性能

        Parameters
        ----------
        X : array-like
            特征矩阵
        y : array-like
            真实标签

        Returns
        -------
        metrics : dict
            性能指标
        """

        # 预测
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        # 计算准确率
        accuracy = accuracy_score(y, y_pred)

        # 计算每层的准确率
        y_groups = np.array([self.celltype_to_group[celltype] for celltype in y])
        y_groups_pred_encoded = self.coarse_classifier.predict(X)
        y_groups_pred = self.group_encoder.inverse_transform(y_groups_pred_encoded)
        coarse_accuracy = accuracy_score(y_groups, y_groups_pred)

        # 计算AUROC (One-vs-Rest)
        from sklearn.preprocessing import label_binarize

        # 收集所有类别
        all_celltypes = []
        for celltypes in self.hierarchy.values():
            all_celltypes.extend(celltypes)
        all_celltypes = sorted(set(all_celltypes))

        try:
            y_bin = label_binarize(y, classes=all_celltypes)
            if y_bin.shape[1] > 1:
                auroc = roc_auc_score(y_bin, y_proba, average='macro')
            else:
                auroc = np.nan
        except:
            auroc = np.nan

        metrics = {
            'accuracy': accuracy,
            'coarse_accuracy': coarse_accuracy,
            'auroc': auroc,
            'n_classes': len(all_celltypes),
            'n_groups': len(self.hierarchy)
        }

        if self.verbose:
            print(f"\n性能评估:")
            print(f"  粗分类准确率: {coarse_accuracy:.4f}")
            print(f"  细分类准确率: {accuracy:.4f}")
            if not np.isnan(auroc):
                print(f"  AUROC (OvR): {auroc:.4f}")
            print(f"  类别数: {len(all_celltypes)}")
            print(f"  分组数: {len(self.hierarchy)}\n")

        return metrics


def should_use_hierarchical(adata, groupby, threshold=15):
    """
    判断是否应该使用层次化分类

    Parameters
    ----------
    adata : AnnData
        数据对象
    groupby : str
        细胞类型列名
    threshold : int
        类别数量阈值（默认15）

    Returns
    -------
    use_hierarchical : bool
        是否使用层次化分类
    """

    n_classes = adata.obs[groupby].nunique()
    use_hierarchical = n_classes >= threshold

    if use_hierarchical:
        print(f"✅ 检测到 {n_classes} 个类别，建议使用层次化分类")
    else:
        print(f"ℹ️ 检测到 {n_classes} 个类别，可直接使用普通分类器")

    return use_hierarchical


if __name__ == "__main__":
    print("层次化分类器模块")
    print("使用方法参见 test_hierarchical_classifier.py")
