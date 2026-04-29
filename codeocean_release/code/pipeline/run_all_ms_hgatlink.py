# run_all_ms_hgatlink.py
# 依赖: subprocess, sys, os
# 被依赖: 无（顶层批量运行脚本）
# 职责: 在全部 5 个数据集上顺序运行 MS-HGATLink + tensor 流水线

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

DATASETS = ['kuppe', 'habermann', 'velmeshev', 'carraro']

PIPELINE_SCRIPT = Path(__file__).parent / 'simplified_pipeline_optimized.py'


def _run_one_dataset(dataset_name):
    """运行单个数据集，返回 (success, elapsed_seconds)"""
    cmd = [
        sys.executable, '-u', str(PIPELINE_SCRIPT),
        '--dataset',   dataset_name,
        '--method',    'ms_hgatlink',
        '--reduction', 'tensor',
    ]
    t0 = datetime.now()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = (datetime.now() - t0).total_seconds()
    return result.returncode == 0, elapsed


def _print_header(dataset_name, idx, total):
    """打印数据集运行标题"""
    print(f"\n{'#'*65}")
    print(f"# [{idx}/{total}]  数据集: {dataset_name.upper()}")
    print(f"# 开始时间: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'#'*65}")


def _print_final_report(results, total_elapsed):
    """打印最终汇总报告"""
    print(f"\n{'='*65}")
    print(f"  批量运行完成  总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"{'='*65}")
    for ds, (ok, t) in results.items():
        status = '成功' if ok else '失败'
        print(f"  {ds:12s}  {status}  ({t/60:.1f} 分钟)")


def main():
    print("MS-HGATLink + Tensor  批量运行 5 数据集")
    print(f"流水线脚本: {PIPELINE_SCRIPT}")
    t_start = datetime.now()
    results = {}

    for idx, dataset in enumerate(DATASETS, 1):
        _print_header(dataset, idx, len(DATASETS))
        ok, elapsed = _run_one_dataset(dataset)
        results[dataset] = (ok, elapsed)

    _print_final_report(results, (datetime.now() - t_start).total_seconds())


if __name__ == '__main__':
    main()
