#!/usr/bin/env python3
"""
自动化运行所有5个数据集
使用简单版HGATLink + DET + 集成学习
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

# 切换到工作目录
os.chdir('/root/autodl-tmp/lianaplus_manuscript-main/notebooks/classification/code/pipeline')

DATASETS = [
    ('habermann', 0.99, 15),  # (name, target_auroc, n_ensemble)
    ('velmeshev', 0.77, 15),
    ('kuppe', 1.00, 10),
    ('carraro', 0.88, 10),
    ('reichart', 1.00, 5),
]

def create_dataset_script(dataset_name):
    """为特定数据集创建临时脚本"""
    # 读取原始脚本
    template = Path('simplified_pipeline_optimized-Copy2.py').read_text()

    # 替换数据集名称
    script = template.replace(
        "dataset_name = 'reichart'",
        f"dataset_name = '{dataset_name}'"
    )

    # 确保使用hgatlink
    script = script.replace(
        "liana_method = 'cellchat'",
        "liana_method = 'hgatlink'"
    )

    # 保存临时脚本
    temp_file = f'temp_{dataset_name}_run.py'
    Path(temp_file).write_text(script)

    return temp_file

def run_dataset(dataset_name, target_auroc, n_ensemble):
    """运行单个数据集"""
    print("\n" + "="*80)
    print(f"🚀 运行: {dataset_name.upper()}")
    print(f"   目标AUROC: {target_auroc:.2f}, 集成模型数: {n_ensemble}")
    print("="*80)

    # 创建临时脚本
    temp_script = create_dataset_script(dataset_name)

    try:
        # 运行脚本
        start_time = time.time()
        result = subprocess.run(
            ['python', temp_script],
            capture_output=True,
            text=True,
            timeout=7200  # 2小时超时
        )
        elapsed = time.time() - start_time

        # 保存日志
        log_dir = Path('../../output/sota_run_logs')
        log_dir.mkdir(exist_ok=True, parents=True)

        log_file = log_dir / f'{dataset_name}_{datetime.now().strftime("%H%M%S")}.log'
        log_file.write_text(result.stdout + "\n\nSTDERR:\n" + result.stderr)

        if result.returncode == 0:
            print(f"✅ {dataset_name} 完成！用时: {elapsed/60:.1f} 分钟")

            # 尝试提取AUROC
            for line in result.stdout.split('\n'):
                if '平均 AUROC' in line:
                    print(f"   {line}")
                    break

            return True, elapsed
        else:
            print(f"❌ {dataset_name} 失败！")
            print("错误:", result.stderr[-500:] if result.stderr else "无错误输出")
            return False, elapsed

    except subprocess.TimeoutExpired:
        print(f"⚠️ {dataset_name} 超时（2小时）")
        return False, 7200
    except Exception as e:
        print(f"❌ {dataset_name} 异常: {e}")
        return False, 0
    finally:
        # 清理临时脚本
        if Path(temp_script).exists():
            Path(temp_script).unlink()

def main():
    """主函数"""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*15 + "自动化SOTA测试 - 5个数据集（Python版）" + " "*15 + "║")
    print("╚" + "="*78 + "╝")

    start_time = datetime.now()
    results = {}

    for dataset_name, target_auroc, n_ensemble in DATASETS:
        success, elapsed = run_dataset(dataset_name, target_auroc, n_ensemble)
        results[dataset_name] = {
            'success': success,
            'elapsed': elapsed,
            'target_auroc': target_auroc
        }

    # 生成报告
    end_time = datetime.now()
    total_elapsed = (end_time - start_time).total_seconds()

    print("\n\n" + "="*80)
    print("🎉 所有数据集测试完成！")
    print("="*80)
    print(f"\n开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总用时: {total_elapsed/3600:.2f} 小时")

    print("\n结果汇总:")
    print("-" * 80)
    for name, data in results.items():
        status = "✅" if data['success'] else "❌"
        print(f"{status} {name:12s}: 目标={data['target_auroc']:.2f}, "
              f"用时={data['elapsed']/60:.1f}分钟")

    # 保存汇总
    report_file = Path('../../output/sota_run_logs/final_report.txt')
    with open(report_file, 'w') as f:
        f.write("SOTA测试最终报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"开始时间: {start_time}\n")
        f.write(f"结束时间: {end_time}\n")
        f.write(f"总用时: {total_elapsed/3600:.2f} 小时\n\n")

        for name, data in results.items():
            status = "成功" if data['success'] else "失败"
            f.write(f"{name}: {status}, 目标AUROC={data['target_auroc']:.2f}, "
                   f"用时={data['elapsed']/60:.1f}分钟\n")

    print(f"\n📊 详细报告: {report_file}")

if __name__ == '__main__':
    main()
