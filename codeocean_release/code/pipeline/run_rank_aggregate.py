# run_rank_aggregate.py
# 依赖: subprocess, sys, simplified_pipeline_optimized.py, unified_comparison.py
# 被依赖: 无（顶层批量运行脚本）
# 职责: 在全部 5 个数据集上运行 rank_aggregate LR 推断（步骤1-2），
#       然后调用 unified_comparison.py 计算 AUROC，补全对比表

import subprocess
import sys
from datetime import datetime
from pathlib import Path

DATASETS = ['kuppe', 'habermann', 'carraro', 'velmeshev', 'reichart']

_DIR = Path(__file__).parent
PIPELINE_SCRIPT = _DIR / 'simplified_pipeline_optimized.py'
COMPARISON_SCRIPT = _DIR / 'unified_comparison.py'


def _run_liana_step2(dataset_name):
    """运行 simplified_pipeline 步骤1-2 生成 rank_aggregate LR 评分。"""
    cmd = [
        sys.executable, '-u', str(PIPELINE_SCRIPT),
        '--dataset',   dataset_name,
        '--method',    'rank_aggregate',
        '--reduction', 'tensor',
        # 不加 --force-step 2：有断点则复用（kuppe/habermann 已完成），无断点则从头跑
    ]
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {dataset_name}: 运行 rank_aggregate 步骤1-2 ...")
    t0 = datetime.now()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = (datetime.now() - t0).total_seconds()
    ok = result.returncode == 0
    print(f"  完成: {'OK' if ok else 'FAIL'} ({elapsed/60:.1f} min)")
    return ok, elapsed


def _run_unified_comparison():
    """调用 unified_comparison.py 更新 AUROC 对比表（含 rank_aggregate）。"""
    cmd = [sys.executable, '-u', str(COMPARISON_SCRIPT), '--method', 'rank_aggregate']
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 计算 rank_aggregate AUROC ...")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    print("=" * 60)
    print("  rank_aggregate 补充实验")
    print("  步骤: LR 推断(5 数据集) → unified_comparison AUROC")
    print("=" * 60)

    t_start = datetime.now()
    results = {}

    for dataset in DATASETS:
        ok, elapsed = _run_liana_step2(dataset)
        results[dataset] = (ok, elapsed)

    print(f"\n{'='*60}")
    print("  步骤1-2 汇总")
    for ds, (ok, t) in results.items():
        print(f"  {ds:12s}  {'OK' if ok else 'FAIL'}  ({t/60:.1f} min)")

    if all(ok for ok, _ in results.values()):
        _run_unified_comparison()
    else:
        failed = [ds for ds, (ok, _) in results.items() if not ok]
        print(f"\n警告: {failed} 步骤2 失败，跳过 unified_comparison")

    total = (datetime.now() - t_start).total_seconds()
    print(f"\n总耗时: {total/60:.1f} min")


if __name__ == '__main__':
    main()
