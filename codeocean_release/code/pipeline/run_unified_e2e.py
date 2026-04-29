# run_unified_e2e.py
# 依赖: e2e_pipeline.py
# 被依赖: 直接运行
# 职责: 用统一配置跑全部5个数据集，满足审稿人要求（无cherry-picking）

import sys
import subprocess
from pathlib import Path

PIPELINE = Path(__file__).parent / 'e2e_pipeline.py'
DATASETS = ['carraro', 'habermann', 'kuppe', 'velmeshev', 'reichart']


def run_one(dataset):
    """运行单个数据集的 unified E2E。"""
    cmd = [
        sys.executable, '-u', str(PIPELINE),
        '--dataset', dataset,
        '--unified',
        '--collect_gate',
        '--collect_metrics',
    ]
    print(f"\n{'='*60}")
    print(f"  Running unified E2E: {dataset}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(PIPELINE.parent))
    return result.returncode


def main():
    results = {}
    for ds in DATASETS:
        rc = run_one(ds)
        results[ds] = 'OK' if rc == 0 else f'FAILED (rc={rc})'
        print(f"  [{ds}] {results[ds]}")

    print(f"\n{'='*60}")
    print("  Unified E2E 结果汇总:")
    for ds, status in results.items():
        print(f"    {ds}: {status}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
