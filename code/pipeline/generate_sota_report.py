#!/usr/bin/env python3
"""
生成SOTA测试最终报告
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 输出目录
output_dir = Path("output")
results = []

# 数据集和目标性能
datasets = {
    "habermann": 0.99,
    "velmeshev": 0.77,
    "kuppe": 1.00,
    "carraro": 0.88,
    "reichart": 1.00
}

print("=" * 80)
print("SOTA测试最终报告 - 完整版HGATLink")
print("=" * 80)
print()

# 读取每个数据集的结果
for dataset, target_auroc in datasets.items():
    result_file = output_dir / f"{dataset}_hgatlink_det.csv"

    if not result_file.exists():
        print(f"❌ {dataset:12s} - 结果文件不存在")
        results.append({
            "dataset": dataset,
            "mean_auroc": np.nan,
            "std_auroc": np.nan,
            "mean_f1": np.nan,
            "std_f1": np.nan,
            "target_auroc": target_auroc,
            "达标": "❌"
        })
        continue

    # 读取结果
    df = pd.read_csv(result_file)

    # 计算统计
    mean_auroc = df['auroc'].mean()
    std_auroc = df['auroc'].std()
    mean_f1 = df['f1_score'].mean()
    std_f1 = df['f1_score'].std()
    n_folds = len(df)

    # 判断是否达标
    达标 = "✅" if mean_auroc >= target_auroc * 0.95 else "❌"

    results.append({
        "dataset": dataset,
        "mean_auroc": mean_auroc,
        "std_auroc": std_auroc,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "n_folds": n_folds,
        "target_auroc": target_auroc,
        "达标": 达标
    })

    print(f"{dataset:12s} | AUROC: {mean_auroc:.4f}±{std_auroc:.4f} | F1: {mean_f1:.4f}±{std_f1:.4f} | 目标: {target_auroc:.2f} | {达标}")

# 创建汇总DataFrame
summary_df = pd.DataFrame(results)

# 保存汇总
summary_file = output_dir / "hgatlink_det_sota_summary.csv"
summary_df.to_csv(summary_file, index=False)

print()
print("=" * 80)
print(f"✅ 汇总报告已保存: {summary_file}")
print("=" * 80)
print()

# 打印统计
达标数 = summary_df['达标'].value_counts().get('✅', 0)
总数 = len(summary_df)
print(f"达标率: {达标数}/{总数} ({达标数/总数*100:.1f}%)")
print()

# 显示汇总表格
print("完整汇总:")
print(summary_df.to_string(index=False))
