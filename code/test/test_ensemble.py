#!/usr/bin/env python3
"""
测试集成学习功能

验证集成学习是否正确集成到分类流程中
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
from sklearn.metrics import roc_auc_score

print("\n" + "=" * 80)
print(" " * 30 + "集成学习功能测试")
print("=" * 80)

# 测试1: 导入检查
print("\n【测试1: 导入检查】")
try:
    from utils.ensemble_utils import BaggingHGATLink, DiversifiedEnsemble, FastEnsemble
    print("✓ ensemble_utils导入成功")
    ENSEMBLE_OK = True
except ImportError as e:
    print(f"✗ ensemble_utils导入失败: {e}")
    ENSEMBLE_OK = False

try:
    from utils.classify_utils import _run_rf_auc
    print("✓ classify_utils导入成功")
    CLASSIFY_OK = True
except ImportError as e:
    print(f"✗ classify_utils导入失败: {e}")
    CLASSIFY_OK = False

if not (ENSEMBLE_OK and CLASSIFY_OK):
    print("\n⚠️ 导入失败，无法继续测试")
    exit(1)

# 测试2: 基本功能测试
print("\n【测试2: 基本功能测试】")

# 生成测试数据
X, y = make_classification(n_samples=200, n_features=20, n_informative=15,
                           n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"训练集: {X_train.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")

# 测试2.1: Bootstrap集成
print("\n【测试2.1: Bootstrap集成】")
try:
    bagging = BaggingHGATLink(n_models=3, verbose=False)
    bagging.fit(X_train, y_train)
    y_pred_proba = bagging.predict_proba(X_test)
    auroc_bagging = roc_auc_score(y_test, y_pred_proba[:, 1])
    print(f"✓ Bootstrap集成测试通过")
    print(f"  AUROC: {auroc_bagging:.4f}")
except Exception as e:
    print(f"✗ Bootstrap集成测试失败: {e}")
    auroc_bagging = None

# 测试2.2: 快速集成
print("\n【测试2.2: 快速集成】")
try:
    fast = FastEnsemble(n_models=3, n_estimators=50, verbose=False)
    fast.fit(X_train, y_train)
    y_pred_proba = fast.predict_proba(X_test)
    auroc_fast = roc_auc_score(y_test, y_pred_proba[:, 1])
    print(f"✓ 快速集成测试通过")
    print(f"  AUROC: {auroc_fast:.4f}")
except Exception as e:
    print(f"✗ 快速集成测试失败: {e}")
    auroc_fast = None

# 测试2.3: 多样化集成
print("\n【测试2.3: 多样化集成】")
try:
    diversified = DiversifiedEnsemble(n_models=3, diversity_strategy='all', verbose=False)
    diversified.fit(X_train, y_train)
    y_pred_proba = diversified.predict_proba(X_test)
    auroc_diversified = roc_auc_score(y_test, y_pred_proba[:, 1])
    print(f"✓ 多样化集成测试通过")
    print(f"  AUROC: {auroc_diversified:.4f}")
except Exception as e:
    print(f"✗ 多样化集成测试失败: {e}")
    auroc_diversified = None

# 测试3: 集成到classify_utils
print("\n【测试3: 集成到classify_utils】")

train_index = np.arange(len(X_train))
test_index = np.arange(len(X_train), len(X))

# 测试3.1: 单模型（baseline）
print("\n【测试3.1: 单模型baseline】")
try:
    auroc_single, _, _, _, _ = _run_rf_auc(
        X, y, train_index, test_index,
        n_estimators=100,
        use_ensemble=False
    )
    print(f"✓ 单模型测试通过")
    print(f"  AUROC: {auroc_single:.4f}")
except Exception as e:
    print(f"✗ 单模型测试失败: {e}")
    auroc_single = None

# 测试3.2: 快速集成（通过_run_rf_auc）
print("\n【测试3.2: 快速集成（通过_run_rf_auc）】")
try:
    auroc_ensemble, _, _, _, _ = _run_rf_auc(
        X, y, train_index, test_index,
        n_estimators=100,
        use_ensemble=True,
        ensemble_type='fast'
    )
    print(f"✓ 集成学习测试通过")
    print(f"  AUROC: {auroc_ensemble:.4f}")

    if auroc_single is not None:
        improvement = auroc_ensemble - auroc_single
        print(f"  相比单模型改进: {improvement:+.4f} ({improvement/auroc_single*100:+.2f}%)")
except Exception as e:
    print(f"✗ 集成学习测试失败: {e}")
    auroc_ensemble = None

# 测试3.3: Bootstrap集成（通过_run_rf_auc）
print("\n【测试3.3: Bootstrap集成（通过_run_rf_auc）】")
try:
    auroc_bagging_rf, _, _, _, _ = _run_rf_auc(
        X, y, train_index, test_index,
        n_estimators=100,
        use_ensemble=True,
        ensemble_type='bagging'
    )
    print(f"✓ Bootstrap集成测试通过")
    print(f"  AUROC: {auroc_bagging_rf:.4f}")

    if auroc_single is not None:
        improvement = auroc_bagging_rf - auroc_single
        print(f"  相比单模型改进: {improvement:+.4f} ({improvement/auroc_single*100:+.2f}%)")
except Exception as e:
    print(f"✗ Bootstrap集成测试失败: {e}")
    auroc_bagging_rf = None

# 总结
print("\n" + "=" * 80)
print(" " * 30 + "测试总结")
print("=" * 80)

print(f"\n{'方法':<25} {'AUROC':<10} {'状态':<10}")
print("-" * 80)

results = [
    ('单模型 (baseline)', auroc_single, '✓'),
    ('Bootstrap集成', auroc_bagging, '✓' if auroc_bagging else '✗'),
    ('快速集成', auroc_fast, '✓' if auroc_fast else '✗'),
    ('多样化集成', auroc_diversified, '✓' if auroc_diversified else '✗'),
    ('集成(via _run_rf_auc)', auroc_ensemble, '✓' if auroc_ensemble else '✗'),
    ('Bagging(via _run_rf_auc)', auroc_bagging_rf, '✓' if auroc_bagging_rf else '✗'),
]

for method, auroc, status in results:
    if auroc is not None:
        print(f"{method:<25} {auroc:.4f}     {status}")
    else:
        print(f"{method:<25} {'N/A':<10} {status}")

print("-" * 80)

# 检查所有测试是否通过
all_passed = all(auroc is not None for _, auroc, _ in results)

if all_passed:
    print("\n✅ 所有测试通过！集成学习功能正常工作")
else:
    print("\n⚠️ 部分测试失败，请检查错误信息")

print("=" * 80 + "\n")
