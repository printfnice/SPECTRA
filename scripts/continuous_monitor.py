#!/usr/bin/env python3
"""
持续监控脚本：定期检查处理进度，自动处理 OOM 和其他错误
"""
import os
import sys
import time
import psutil
import subprocess
from datetime import datetime

DATASETS = ['habermann', 'reichart', 'velmeshev']
CHECK_INTERVAL = 300  # 5分钟检查一次
LOG_FILE = 'continuous_monitor.log'
PYTHON_PATH = '/root/miniconda3/envs/LIANA/bin/python'

def log(message):
    """记录日志"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_msg + '\n')

def get_process_info(pid):
    """获取进程信息"""
    try:
        process = psutil.Process(pid)
        mem_mb = process.memory_info().rss / (1024**2)
        cpu_percent = process.cpu_percent(interval=1)
        return {
            'status': process.status(),
            'mem_mb': mem_mb,
            'cpu_percent': cpu_percent
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None

def check_dataset_status():
    """检查数据集处理状态"""
    status = {}
    for dataset in DATASETS:
        result_path = os.path.join('data', 'results', f'{dataset}.csv')
        processed_path = os.path.join('data', 'interim', f'{dataset}_processed.h5ad')

        if os.path.exists(result_path):
            status[dataset] = '完成'
        elif os.path.exists(processed_path):
            status[dataset] = '预处理完成'
        else:
            status[dataset] = '未处理'
    return status

def get_current_processing_dataset():
    """从日志中获取当前正在处理的数据集"""
    try:
        with open('processing.log', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in reversed(lines[-200:]):
                for dataset in DATASETS:
                    if f"处理数据集: {dataset.upper()}" in line or f"预处理数据集: {dataset}" in line:
                        return dataset
    except:
        pass
    return None

def restart_if_needed():
    """检查是否需要重启处理"""
    # 检查监控进程是否运行
    monitor_running = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'auto_monitor.py' in cmdline and proc.info['pid'] != os.getpid():
                monitor_running = True
                break
        except:
            pass

    if not monitor_running:
        log("⚠ 监控进程未运行，正在重启...")
        subprocess.Popen(
            [PYTHON_PATH, 'auto_monitor.py'],
            stdout=open('monitor.log', 'a'),
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        time.sleep(5)
        return True

    return False

def main():
    """主监控循环"""
    log("="*70)
    log("启动持续监控")
    log(f"检查间隔: {CHECK_INTERVAL} 秒")
    log("="*70)

    iteration = 0

    while True:
        try:
            iteration += 1
            log(f"\n{'='*70}")
            log(f"检查 #{iteration}")
            log(f"{'='*70}")

            # 检查数据集状态
            dataset_status = check_dataset_status()
            log("\n数据集状态:")
            all_complete = True
            for dataset, status in dataset_status.items():
                symbol = "✓" if status == "完成" else ("◐" if status == "预处理完成" else "○")
                log(f"  {symbol} {dataset:12s}: {status}")
                if status != "完成":
                    all_complete = False

            # 如果所有数据集都完成了，退出监控
            if all_complete:
                log("\n" + "="*70)
                log("✓ 所有数据集处理完成！监控结束。")
                log("="*70)
                break

            # 检查进程状态
            current_dataset = get_current_processing_dataset()
            if current_dataset:
                log(f"\n当前处理: {current_dataset}")

            # 检查监控进程
            restarted = restart_if_needed()
            if restarted:
                log("  已重启监控进程")

            # 检查主处理进程
            if os.path.exists('processing.pid'):
                with open('processing.pid', 'r') as f:
                    pid = int(f.read().strip())

                proc_info = get_process_info(pid)
                if proc_info:
                    log(f"\n处理进程 (PID {pid}):")
                    log(f"  状态: {proc_info['status']}")
                    log(f"  内存: {proc_info['mem_mb']:.1f} MB")
                    log(f"  CPU: {proc_info['cpu_percent']:.1f}%")
                else:
                    log(f"\n⚠ 处理进程 (PID {pid}) 已停止")
                    # 进程已停止，等待监控脚本重启
                    time.sleep(30)
            else:
                log("\n○ 未找到处理进程 PID 文件")

            # 系统资源
            mem = psutil.virtual_memory()
            log(f"\n系统资源:")
            log(f"  内存使用: {mem.percent:.1f}% (可用: {mem.available / (1024**3):.1f} GB)")

            # 等待下次检查
            log(f"\n下次检查: {CHECK_INTERVAL} 秒后")
            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            log("\n用户中断监控")
            break
        except Exception as e:
            log(f"\n监控错误: {str(e)}")
            import traceback
            log(traceback.format_exc())
            time.sleep(60)  # 出错后等待1分钟再继续

    log("\n持续监控结束")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
