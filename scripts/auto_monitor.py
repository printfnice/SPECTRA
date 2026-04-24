#!/usr/bin/env python3
"""
智能监控脚本：监控数据处理进度、内存使用，自动处理 OOM 错误
"""
import os
import sys
import time
import psutil
import subprocess
from datetime import datetime

# 配置
DATASETS = ['habermann', 'reichart', 'velmeshev']
LOG_FILE = 'processing.log'
STATUS_FILE = 'processing_status.txt'
PID_FILE = 'processing.pid'
MAX_RETRIES = 3

class ProcessMonitor:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(self.script_dir)
        self.log_file = LOG_FILE
        self.status_file = STATUS_FILE
        self.pid_file = PID_FILE
        self.process = None
        self.retry_count = {}
        for dataset in DATASETS:
            self.retry_count[dataset] = 0

    def log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')

    def update_status(self, status):
        """更新状态文件"""
        with open(self.status_file, 'w', encoding='utf-8') as f:
            f.write(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"状态: {status}\n")
            f.write(f"\n已完成的数据集:\n")
            for dataset in DATASETS:
                processed_path = os.path.join('data', 'interim', f'{dataset}_processed.h5ad')
                result_path = os.path.join('data', 'results', f'{dataset}.csv')
                if os.path.exists(result_path):
                    f.write(f"  ✓ {dataset} - 完全完成\n")
                elif os.path.exists(processed_path):
                    f.write(f"  ◐ {dataset} - 预处理完成\n")
                else:
                    f.write(f"  ○ {dataset} - 未处理\n")

    def check_memory(self):
        """检查内存使用情况"""
        mem = psutil.virtual_memory()
        mem_percent = mem.percent
        mem_available_gb = mem.available / (1024**3)

        # 内存使用超过 90% 时警告
        if mem_percent > 90:
            self.log(f"⚠ 内存使用率高: {mem_percent:.1f}% (可用: {mem_available_gb:.1f} GB)")
            return True
        return False

    def start_processing(self):
        """启动处理进程"""
        self.log("="*70)
        self.log("启动数据处理流程")
        self.log(f"数据集: {', '.join(DATASETS)}")
        self.log("="*70)

        # 使用 LIANA conda 环境
        python_path = '/root/miniconda3/envs/LIANA/bin/python'
        cmd = [python_path, 'run_full_pipeline.py']

        with open(self.log_file, 'a') as log_f:
            self.process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=self.script_dir
            )

        # 保存 PID
        with open(self.pid_file, 'w') as f:
            f.write(str(self.process.pid))

        self.log(f"进程已启动 (PID: {self.process.pid})")
        self.update_status(f"运行中 (PID: {self.process.pid})")

    def monitor_loop(self):
        """监控循环"""
        check_interval = 30  # 每 30 秒检查一次

        while True:
            try:
                # 检查进程状态
                if self.process is None:
                    self.log("错误: 进程未启动")
                    break

                returncode = self.process.poll()

                if returncode is not None:
                    # 进程已结束
                    if returncode == 0:
                        self.log("✓ 处理流程正常完成")
                        self.update_status("完成")
                        return True
                    else:
                        self.log(f"✗ 进程异常退出 (返回码: {returncode})")
                        self.handle_failure(returncode)
                        return False

                # 检查内存
                self.check_memory()

                # 更新状态
                self.update_status(f"运行中 (PID: {self.process.pid})")

                # 等待
                time.sleep(check_interval)

            except KeyboardInterrupt:
                self.log("\n用户中断，正在清理...")
                if self.process:
                    self.process.terminate()
                    self.process.wait()
                break
            except Exception as e:
                self.log(f"监控错误: {str(e)}")
                time.sleep(check_interval)

    def handle_failure(self, returncode):
        """处理失败情况"""
        self.log(f"\n处理失败分析:")

        # 检查是否是 OOM
        if returncode == -9 or returncode == 137:
            self.log("  原因: 内存溢出 (OOM)")
            self.log("  建议: 该数据集可能过大，需要优化内存使用")
        else:
            self.log(f"  返回码: {returncode}")

        # 检查哪个数据集失败
        current_dataset = self.get_current_dataset()
        if current_dataset:
            self.log(f"  失败的数据集: {current_dataset}")

        self.update_status(f"失败 (返回码: {returncode})")

    def get_current_dataset(self):
        """从日志中判断当前处理的数据集"""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in reversed(lines[-100:]):  # 检查最后 100 行
                    for dataset in DATASETS:
                        if f"处理数据集: {dataset.upper()}" in line or f"开始处理数据集: {dataset.upper()}" in line:
                            return dataset
        except:
            pass
        return None

    def run(self):
        """主运行函数"""
        try:
            self.start_processing()
            success = self.monitor_loop()

            if success:
                self.log("\n" + "="*70)
                self.log("所有数据集处理完成！")
                self.log("="*70)
                self.print_summary()

        except Exception as e:
            self.log(f"监控程序错误: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            if os.path.exists(self.pid_file):
                os.remove(self.pid_file)

    def print_summary(self):
        """打印处理摘要"""
        self.log("\n处理结果摘要:")
        for dataset in DATASETS:
            result_path = os.path.join('data', 'results', f'{dataset}.csv')
            if os.path.exists(result_path):
                file_size = os.path.getsize(result_path) / (1024**2)
                self.log(f"  ✓ {dataset:12s}: {result_path} ({file_size:.1f} MB)")
            else:
                self.log(f"  ✗ {dataset:12s}: 未完成")


def check_status():
    """快速检查处理状态"""
    status_file = 'processing_status.txt'
    if os.path.exists(status_file):
        with open(status_file, 'r', encoding='utf-8') as f:
            print(f.read())
    else:
        print("状态文件不存在，处理可能尚未开始")

    print("\n" + "="*70)

    # 检查结果文件
    print("结果文件检查:")
    for dataset in DATASETS:
        result_path = os.path.join('data', 'results', f'{dataset}.csv')
        if os.path.exists(result_path):
            file_size = os.path.getsize(result_path) / (1024**2)
            mtime = datetime.fromtimestamp(os.path.getmtime(result_path))
            print(f"  ✓ {dataset}: 已完成 ({file_size:.1f} MB, {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            processed_path = os.path.join('data', 'interim', f'{dataset}_processed.h5ad')
            if os.path.exists(processed_path):
                print(f"  ◐ {dataset}: 预处理完成，分类进行中...")
            else:
                print(f"  ○ {dataset}: 未处理")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'status':
        check_status()
    else:
        monitor = ProcessMonitor()
        monitor.run()
