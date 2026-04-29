#!/usr/bin/env python3
# overnight_runner.py
# 职责: 自主实验决策循环（无时间限制，跑到超越目标或穷尽所有实验）
#   1. 等待 focal / rdrop_cw 完成
#   2. 检查是否超越 CellChat 0.9655
#   3. 若未超越，依次运行备选实验，自动记录决策日志

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

# ── 路径 ──────────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).parent
CODE_DIR = PIPELINE_DIR.parent
sys.path.insert(0, str(CODE_DIR))
os.chdir(PIPELINE_DIR)

OUTPUT_DIR = PIPELINE_DIR / 'output' / 'supcon_ablation'
LOG = Path('/tmp/overnight.log')
TARGET = 0.9655

# ── 实验队列（顺序执行，跳过已完成） ────────────────────────────────
# 前两个由 run_queue2.sh 启动，overnight_runner 只监控；
# 后续实验由 overnight_runner 自行启动。
EXPERIMENT_QUEUE = [
    # (tag,                  script,                              log)
    ('supcon_w03_focal',     'run_reichart_focal.py',             '/tmp/focal.log'),
    ('rdrop_cw',             'run_reichart_rdrop_cw.py',          '/tmp/rdrop_cw.log'),
    ('rdrop_only',           'run_reichart_rdrop_only.py',        '/tmp/rdrop_only.log'),
    ('focal_rdrop',          'run_reichart_focal_rdrop.py',       '/tmp/focal_rdrop.log'),
    ('cw_only',              'run_reichart_cw_only.py',           '/tmp/cw_only.log'),
    ('w03_finepeak',         'run_reichart_w03_finepeak.py',      '/tmp/w03_finepeak.log'),
]

# 前两个由外部队列管理，不重复启动
EXTERNALLY_LAUNCHED = {'supcon_w03_focal', 'rdrop_cw'}


def log(msg):
    ts = datetime.now().strftime('%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def read_result(tag):
    p = OUTPUT_DIR / f'reichart_{tag}.csv'
    if not p.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(str(p))
        auroc = float(df['mean_auroc'].iloc[0])
        folds = str(df['fold_aurocs'].iloc[0]) if 'fold_aurocs' in df.columns else 'N/A'
        return auroc, folds
    except Exception:
        return None


def is_done(tag):
    return (OUTPUT_DIR / f'reichart_{tag}.csv').exists()


def run_experiment(script, out_log):
    """在当前 conda 环境直接启动子进程"""
    cmd = f'python -u {script} > {out_log} 2>&1'
    log(f'  ▶ 启动: {script}')
    return subprocess.Popen(['bash', '-c', cmd], cwd=str(PIPELINE_DIR))


def wait_for_experiment(tag, proc=None, poll_sec=120):
    """轮询等待实验完成（CSV 出现 or 进程自然退出）"""
    log(f'  ⏳ 等待 {tag}...')
    while not is_done(tag):
        if proc is not None and proc.poll() is not None:
            log(f'  ⚠️  {tag} 进程已退出但无 CSV，跳过')
            return False
        time.sleep(poll_sec)
    return True


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log('=' * 55)
    log('overnight_runner 启动（无时间限制）')
    log(f'目标: AUROC > {TARGET}')
    log(f'实验队列: {[t for t, _, _ in EXPERIMENT_QUEUE]}')
    log('=' * 55)

    best_auroc = 0.9641   # supcon_w03_mixoff 已知基线
    best_tag = 'w03_mixoff'

    for tag, script, out_log in EXPERIMENT_QUEUE:

        # ── 已有结果：读取并判断 ─────────────────────────────────────
        if is_done(tag):
            res = read_result(tag)
            if res:
                auroc, folds = res
                delta = auroc - TARGET
                log(f'[已完成] {tag}  AUROC={auroc:.4f}  Δ={delta:+.4f}  {folds}')
                if auroc > best_auroc:
                    best_auroc, best_tag = auroc, tag
                if auroc > TARGET:
                    log(f'🎉 {tag} 超越 CellChat！停止。')
                    _summary(best_tag, best_auroc)
                    return
            continue

        # ── 启动或等待实验 ───────────────────────────────────────────
        if tag in EXTERNALLY_LAUNCHED:
            ok = wait_for_experiment(tag, proc=None)
        else:
            proc = run_experiment(script, out_log)
            ok = wait_for_experiment(tag, proc=proc)

        if not ok:
            continue

        # ── 读结果、决策 ─────────────────────────────────────────────
        res = read_result(tag)
        if res is None:
            log(f'  ⚠️  {tag} 无法读取结果，跳过')
            continue

        auroc, folds = res
        delta = auroc - TARGET
        log(f'[完成] {tag}  AUROC={auroc:.4f}  Δ_vs_CellChat={delta:+.4f}')
        log(f'       folds={folds}')

        if auroc > best_auroc:
            best_auroc, best_tag = auroc, tag
            log(f'  ✨ 新最优: {best_tag}  {best_auroc:.4f}')

        if auroc > TARGET:
            log(f'🎉 超越 CellChat！')
            _summary(best_tag, best_auroc)
            return

        log(f'  → 未超越，继续下一个 (当前最优={best_auroc:.4f}，差距={TARGET-best_auroc:.4f})')

    _summary(best_tag, best_auroc)


def _summary(best_tag, best_auroc):
    log('=' * 55)
    log(f'最终最优: {best_tag}  AUROC={best_auroc:.4f}')
    if best_auroc > TARGET:
        log(f'✅ 已超越 CellChat ({TARGET})')
    else:
        log(f'❌ 未超越 CellChat，差距 {TARGET - best_auroc:.4f}')
    log('所有实验已穷尽，overnight_runner 退出。')
    log('=' * 55)


if __name__ == '__main__':
    main()
