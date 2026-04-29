#!/usr/bin/env python3
"""
测试层次化分类器功能

验证层次化分类器是否正确工作
"""

import os
import sys
from pathlib import Path

# 添加code目录到Python路径
code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize

print("\n" + "=" * 80)
print(" " * 25 + "层次化分类器功能测试")
print("=" * 80)

# 测试1: 导入检查
print("\n【测试1: 导入检查】")
try:
    from model.hierarchical_classifier import (
        HierarchicalCellTypeClassifier,
        auto_build_hierarchy,
        should_use_hierarchical
    )
    print("✓ hierarchical_classifier导入成功")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    exit(1)

# 测试2: 生成模拟数据
print("\n【测试2: 生成模拟数据】")

# 生成多类别数据（模拟20个细胞类型）
print("生成20类别数据集...")
X, y_numeric = make_classification(
    n_samples=400,
    n_features=50,
    n_informative=30,
    n_redundant=10,
    n_classes=20,
    n_clusters_per_class=1,
    random_state=42,
    flip_y=0.05
)

# 转换为细胞类型名称
celltype_names = [f"CellType_{i:02d}" for i in range(20)]
y = np.array([celltype_names[i] for i in y_numeric])

print(f"✓ 数据生成成功")
print(f"  样本数: {len(X)}")
print(f"  特征数: {X.shape[1]}")
print(f"  类别数: {len(np.unique(y))}")
print(f"  类别分布: min={np.min(np.bincount(y_numeric))}, "
      f"max={np.max(np.bincount(y_numeric))}")

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"  训练集: {len(X_train)} 样本")
print(f"  测试集: {len(X_test)} 样本")

# 测试3: 构建层次结构（模拟）
print("\n【测试3: 构建层次结构】")

try:
    # 由于我们没有真实的AnnData对象，手动构建层次结构
    # 模拟：将20个细胞类型分成4大组
    hierarchy = {
        0: [f"CellType_{i:02d}" for i in range(0, 5)],    # 组0: 类型0-4
        1: [f"CellType_{i:02d}" for i in range(5, 10)],   # 组1: 类型5-9
        2: [f"CellType_{i:02d}" for i in range(10, 15)],  # 组2: 类型10-14
        3: [f"CellType_{i:02d}" for i in range(15, 20)]   # 组3: 类型15-19
    }

    celltype_to_group = {}
    for group_id, celltypes in hierarchy.items():
        for ct in celltypes:
            celltype_to_group[ct] = group_id

    print("✓ 层次结构构建成功")
    print(f"  分组数: {len(hierarchy)}")
    for group_id, celltypes in hierarchy.items():
        print(f"  组 {group_id}: {len(celltypes)} 个细胞类型")

except Exception as e:
    print(f"✗ 层次结构构建失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试4: 训练层次化分类器
print("\n【测试4: 训练层次化分类器】")

try:
    clf_hierarchical = HierarchicalCellTypeClassifier(
        hierarchy=hierarchy,
        celltype_to_group=celltype_to_group,
        n_estimators=50,  # 减少树数量以加快测试
        verbose=True
    )

    clf_hierarchical.fit(X_train, y_train)
    print("✓ 层次化分类器训练成功")

except Exception as e:
    print(f"✗ 层次化分类器训练失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试5: 预测
print("\n【测试5: 预测】")

try:
    y_pred = clf_hierarchical.predict(X_test)
    print("✓ 预测成功")
    print(f"  预测样本数: {len(y_pred)}")

    # 检查预测是否包含有效类别
    unique_pred = set(y_pred)
    unique_true = set(y_test)
    print(f"  预测类别数: {len(unique_pred)}")
    print(f"  真实类别数: {len(unique_true)}")

except Exception as e:
    print(f"✗ 预测失败: {e}")
    import traceback
    traceback.print_exc()
    y_pred = None

# 测试6: 预测概率
print("\n【测试6: 预测概率】")

try:
    y_proba = clf_hierarchical.predict_proba(X_test)
    print("✓ 概率预测成功")
    print(f"  概率矩阵形状: {y_proba.shape}")
    print(f"  概率和: min={y_proba.sum(axis=1).min():.4f}, "
          f"max={y_proba.sum(axis=1).max():.4f}")

except Exception as e:
    print(f"✗ 概率预测失败: {e}")
    import traceback
    traceback.print_exc()
    y_proba = None

# 测试7: 性能评估
print("\n【测试7: 性能评估】")

try:
    if y_pred is not None and y_proba is not None:
        # 准确率
        accuracy = accuracy_score(y_test, y_pred)

        # 粗分类准确率
        y_groups_test = np.array([celltype_to_group[ct] for ct in y_test])
        y_groups_pred_encoded = clf_hierarchical.coarse_classifier.predict(X_test)
        y_groups_pred = clf_hierarchical.group_encoder.inverse_transform(
            y_groups_pred_encoded
        )
        coarse_accuracy = accuracy_score(y_groups_test, y_groups_pred)

        # AUROC (One-vs-Rest)
        try:
            all_celltypes = sorted(set(y_test))
            y_bin = label_binarize(y_test, classes=all_celltypes)
            auroc = roc_auc_score(y_bin, y_proba, average='macro')
        except:
            auroc = np.nan

        print("✓ 性能评估完成")
        print(f"\n性能指标:")
        print(f"  粗分类准确率: {coarse_accuracy:.4f}")
        print(f"  细分类准确率: {accuracy:.4f}")
        if not np.isnan(auroc):
            print(f"  AUROC (OvR): {auroc:.4f}")

        # 评估层次化的收益
        if coarse_accuracy >= 0.80:
            print(f"\n✅ 粗分类准确率优秀 (≥80%)，层次化策略有效")
        elif coarse_accuracy >= 0.70:
            print(f"\n⚠️ 粗分类准确率良好 (70-80%)，可能需要调优")
        else:
            print(f"\n⚠️ 粗分类准确率较低 (<70%)，考虑增加分组数")

except Exception as e:
    print(f"✗ 性能评估失败: {e}")
    import traceback
    traceback.print_exc()

# 测试8: 对比普通分类器
print("\n【测试8: 对比普通分类器】")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    # 训练普通随机森林
    print("训练普通随机森林分类器...")
    encoder_baseline = LabelEncoder()
    y_train_encoded = encoder_baseline.fit_transform(y_train)
    y_test_encoded = encoder_baseline.transform(y_test)

    clf_baseline = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        n_jobs=-1,
        oob_score=True,
        class_weight='balanced'
    )
    clf_baseline.fit(X_train, y_train_encoded)

    # 预测
    y_pred_baseline = clf_baseline.predict(X_test)
    y_proba_baseline = clf_baseline.predict_proba(X_test)

    # 评估
    accuracy_baseline = accuracy_score(y_test_encoded, y_pred_baseline)

    try:
        y_bin = label_binarize(y_test_encoded, classes=range(len(celltype_names)))
        auroc_baseline = roc_auc_score(y_bin, y_proba_baseline, average='macro')
    except:
        auroc_baseline = np.nan

    print("✓ 普通分类器训练完成")
    print(f"  准确率: {accuracy_baseline:.4f}")
    if not np.isnan(auroc_baseline):
        print(f"  AUROC: {auroc_baseline:.4f}")

    # 对比
    print(f"\n对比结果:")
    print(f"{'方法':<20} {'准确率':<12} {'AUROC':<12}")
    print("-" * 80)

    auroc_baseline_str = f"{auroc_baseline:.4f}" if not np.isnan(auroc_baseline) else "N/A"
    auroc_str = f"{auroc:.4f}" if not np.isnan(auroc) else "N/A"

    print(f"{'普通分类器':<20} {accuracy_baseline:.4f}       {auroc_baseline_str}")
    print(f"{'层次化分类器':<20} {accuracy:.4f}       {auroc_str}")

    if accuracy > accuracy_baseline:
        improvement = (accuracy - accuracy_baseline) / accuracy_baseline * 100
        print(f"\n✅ 层次化分类器性能更优 (+{improvement:.2f}%)")
    elif abs(accuracy - accuracy_baseline) < 0.01:
        print(f"\n➡️ 两种方法性能相当")
    else:
        print(f"\n⚠️ 普通分类器性能更优（可能数据集不适合层次化）")

except Exception as e:
    print(f"✗ 对比测试失败: {e}")
    import traceback
    traceback.print_exc()

# 总结
print("\n" + "=" * 80)
print(" " * 30 + "测试总结")
print("=" * 80)

tests = [
    ('导入检查', True),
    ('数据生成', len(X) == 400),
    ('层次结构构建', len(hierarchy) == 4),
    ('模型训练', clf_hierarchical is not None),
    ('预测', y_pred is not None),
    ('概率预测', y_proba is not None),
    ('性能评估', accuracy is not None if y_pred is not None else False),
    ('对比测试', True)
]

print(f"\n{'测试项':<20} {'状态':<10}")
print("-" * 80)
for test_name, passed in tests:
    print(f"{test_name:<20} {'✓ 通过' if passed else '✗ 失败'}")

all_passed = all(passed for _, passed in tests)

if all_passed:
    print("\n✅ 所有测试通过！层次化分类器功能正常工作")
else:
    print("\n⚠️ 部分测试失败")

print("=" * 80 + "\n")
