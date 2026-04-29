# run_all_mofa.py
# 依赖: subprocess, sys, os, checkpoint_utils（间接）
# 被依赖: 无（顶层批量运行脚本）
# 职责: 为全部 5 个数据集创建 tensor→mofa 断点软链接，然后顺序运行 MOFA+ 流水线

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

DATASETS = ['kuppe', 'habermann', 'velmeshev', 'carraro', 'reichart']
METHOD = 'ms_hgatlink'
CKPT_DIR = Path(__file__).parent.parent.parent / 'data' / 'results' / 'checkpoints'
PIPELINE_SCRIPT = Path(__file__).parent / 'simplified_pipeline_optimized.py'


# ============================================================================
# 软链接工具
# ============================================================================

def _symlink_step(dataset, step_name):
    """将 tensor 断点软链接为 mofa 断点（step2 / step3 内容与降维方法无关）"""
    src = CKPT_DIR / f'{dataset}_{METHOD}_tensor_{step_name}.pkl'
    dst = CKPT_DIR / f'{dataset}_{METHOD}_mofa_{step_name}.pkl'
    if not src.exists():
        print(f"  [跳过] {src.name} 不存在")
        return False
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())
    print(f"  链接: {src.name} -> {dst.name}")
    return True


def create_mofa_symlinks():
    """为所有数据集创建 step2/step3 的 mofa 软链接"""
    print("\n创建 tensor → mofa 断点软链接")
    print("-" * 50)
    for ds in DATASETS:
        ok2 = _symlink_step(ds, 'step2')
        ok3 = _symlink_step(ds, 'step3')
        if ok2 or ok3:
            print(f"  {ds}: step2={'ok' if ok2 else '-'}  step3={'ok' if ok3 else '-'}")


# ============================================================================
# 批量运行
# ============================================================================

def _run_one_dataset(dataset_name):
    """运行单个数据集 MOFA+ 流水线，返回 (success, elapsed_seconds)"""
    cmd = [
        sys.executable, '-u', str(PIPELINE_SCRIPT),
        '--dataset',   dataset_name,
        '--method',    METHOD,
        '--reduction', 'mofa',
    ]
    t0 = datetime.now()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = (datetime.now() - t0).total_seconds()
    return result.returncode == 0, elapsed


def _print_header(dataset_name, idx, total):
    print(f"\n{'#'*65}")
    print(f"# [{idx}/{total}]  数据集: {dataset_name.upper()}  (MOFA+)")
    print(f"# 开始时间: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'#'*65}")


def _print_final_report(results, total_elapsed):
    print(f"\n{'='*65}")
    print(f"  MOFA+ 批量运行完成  总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"{'='*65}")
    for ds, (ok, t) in results.items():
        status = '成功' if ok else '失败'
        print(f"  {ds:12s}  {status}  ({t/60:.1f} 分钟)")


def main():
    print("MS-HGATLink + MOFA+  批量运行 5 数据集")
    print(f"流水线脚本: {PIPELINE_SCRIPT}")

    create_mofa_symlinks()

    t_start = datetime.now()
    results = {}
    for idx, dataset in enumerate(DATASETS, 1):
        _print_header(dataset, idx, len(DATASETS))
        ok, elapsed = _run_one_dataset(dataset)
        results[dataset] = (ok, elapsed)

    _print_final_report(results, (datetime.now() - t_start).total_seconds())


if __name__ == '__main__':
    main()
