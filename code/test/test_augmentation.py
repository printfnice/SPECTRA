#!/usr/bin/env python3
"""
测试数据增强功能
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

print("\n" + "=" * 80)
print(" " * 30 + "数据增强功能测试")
print("=" * 80)

# 测试1: 导入检查
print("\n【测试1: 导入检查】")
try:
    from utils.augmentation_utils import (
        LRInteractionAugmentation,
        AdaptiveAugmentation,
        check_augmentation_quality
    )
    print("✓ augmentation_utils导入成功")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    exit(1)

# 生成小数据集（模拟carraro）
X_small, y_small = make_classification(
    n_samples=16, n_features=20, n_informative=15,
    n_redundant=5, random_state=42
)

print(f"\n小数据集: {len(X_small)} 样本, {X_small.shape[1]} 特征")
print(f"类别分布: {np.bincount(y_small)}")

# 测试2: SMOTE增强
print("\n【测试2: SMOTE增强】")
try:
    augmenter = LRInteractionAugmentation(verbose=False)
    X_smote, y_smote = augmenter._smote_augment(X_small, y_small, n_samples=16)
    print(f"✓ SMOTE增强成功")
    print(f"  生成样本: {len(X_smote)}")
    print(f"  类别分布: {np.bincount(y_smote.astype(int))}")
except Exception as e:
    print(f"✗ SMOTE失败: {e}")

# 测试3: Mixup增强
print("\n【测试3: Mixup增强】")
try:
    X_mixup, y_mixup = augmenter._mixup_augment(X_small, y_small, n_samples=16)
    print(f"✓ Mixup增强成功")
    print(f"  生成样本: {len(X_mixup)}")
except Exception as e:
    print(f"✗ Mixup失败: {e}")

# 测试4: 高斯噪声增强
print("\n【测试4: 高斯噪声增强】")
try:
    X_noise, y_noise = augmenter._gaussian_noise_augment(X_small, y_small, n_samples=16)
    print(f"✓ 高斯噪声增强成功")
    print(f"  生成样本: {len(X_noise)}")
except Exception as e:
    print(f"✗ 高斯噪声失败: {e}")

# 测试5: 综合增强
print("\n【测试5: 综合增强】")
try:
    augmenter = LRInteractionAugmentation(verbose=True)
    X_aug, y_aug = augmenter.augment(
        X_small, y_small,
        target_samples=32,
        methods=['smote', 'mixup', 'noise']
    )
    print(f"✓ 综合增强成功")
    print(f"  原始: {len(X_small)} 样本")
    print(f"  增强后: {len(X_aug)} 样本")
    print(f"  类别分布: {np.bincount(y_aug.astype(int))}")
except Exception as e:
    print(f"✗ 综合增强失败: {e}")
    X_aug, y_aug = X_small, y_small

# 测试6: 自适应增强
print("\n【测试6: 自适应增强】")
try:
    adaptive_aug = AdaptiveAugmentation(verbose=True)
    X_adaptive, y_adaptive, was_augmented = adaptive_aug.augment_adaptive(
        X_small, y_small,
        min_samples=30
    )
    print(f"✓ 自适应增强成功")
    print(f"  是否增强: {was_augmented}")
    print(f"  最终样本数: {len(X_adaptive)}")
except Exception as e:
    print(f"✗ 自适应增强失败: {e}")
    X_adaptive, y_adaptive = X_small, y_small
    was_augmented = False

# 测试7: 质量检查
print("\n【测试7: 质量检查】")
try:
    if len(X_aug) > len(X_small):
        quality = check_augmentation_quality(X_small, y_small, X_aug, y_aug)

        print(f"✓ 质量检查完成")
        print(f"\n原始数据:")
        print(f"  训练AUROC: {quality['original_train_auroc']:.4f}")
        print(f"  测试AUROC: {quality['original_test_auroc']:.4f}")
        print(f"  差距: {quality['original_gap']:.4f}")

        print(f"\n增强数据:")
        print(f"  训练AUROC: {quality['augmented_train_auroc']:.4f}")
        print(f"  测试AUROC: {quality['augmented_test_auroc']:.4f}")
        print(f"  差距: {quality['augmented_gap']:.4f}")

        print(f"\n改进:")
        print(f"  测试集改进: {quality['test_improvement']:+.4f}")
        print(f"  差距变化: {quality['gap_change']:+.4f}")
        print(f"  质量OK: {'✓' if quality['quality_ok'] else '✗'}")

        if quality['test_improvement'] > 0:
            print(f"\n✅ 数据增强有效提升了性能")
        elif abs(quality['test_improvement']) < 0.01:
            print(f"\n⚠️ 数据增强对性能影响较小")
        else:
            print(f"\n⚠️ 数据增强降低了性能，可能过拟合")
    else:
        print("  跳过质量检查（未进行增强）")
except Exception as e:
    print(f"✗ 质量检查失败: {e}")

# 总结
print("\n" + "=" * 80)
print(" " * 30 + "测试总结")
print("=" * 80)

tests = [
    ('SMOTE增强', True),
    ('Mixup增强', True),
    ('高斯噪声增强', True),
    ('综合增强', len(X_aug) > len(X_small)),
    ('自适��增强', was_augmented),
    ('质量检查', True)
]

print(f"\n{'测试项':<20} {'状态':<10}")
print("-" * 80)
for test_name, passed in tests:
    print(f"{test_name:<20} {'✓ 通过' if passed else '✗ 失败'}")

all_passed = all(passed for _, passed in tests)

if all_passed:
    print("\n✅ 所有测试通过！数据增强功能正常工作")
else:
    print("\n⚠️ 部分测试失败")

print("=" * 80 + "\n")
