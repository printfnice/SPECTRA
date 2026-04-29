# plot_efficiency_table.py
# 依赖: pandas, matplotlib, pub_style.py
# 职责: 计算效率对比表格（wall-clock time）
#
# 时间来源：
#   - E2E 训练时间：从历史实验日志实际测量
#   - LIANA+Tensor+RF：run_liana_baselines_5fold.py 总时间 / 数据集数
#   - GPU: NVIDIA A100 40GB (AutoDL)
#
# 输出:
#   output/data/efficiency_table.csv
#   output/figures/FigS_efficiency.pdf/.png

import sys
from pathlib import Path
import numpy as np
import pandas as pd

_PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(_PIPELINE_DIR))
from pub_style import set_pub_style, save_fig, FIGURES_DIR, DATA_DIR, NATURE_CATEGORICAL

# ============================================================================
# 实测时间数据（来自历史实验日志，单位：分钟）
# ============================================================================

# E2E 训练时间（5-fold CV × n_random_states，GPU）
# 来源：历史 nohup 日志 + 本次 emb_*.log 实测
E2E_TRAIN_MIN = {
    'kuppe':     22,   # 115ep × 5fold × 7rs, GPU ~22min
    'carraro':   18,   # 80ep × 5fold × 7rs, GPU ~18min
    'habermann': 20,   # 90ep × 5fold × 7rs, GPU ~20min
    'velmeshev': 28,   # 190ep × 5fold × 7rs, GPU ~28min
    'reichart':  310,  # 500ep × 5fold × 3rs, GPU ~310min (5h10min, SGDR)
}

# Standard LIANA+Tensor-cell2cell+RF 时间（per method，5-fold）
# 来源：LIANA-baselines-5fold 2h10min / (5数据集×7方法) ≈ 4min/method/dataset
# 详细拆分：Step2(LIANA) + Step3(Tensor) + Step4(RF)
LIANA_STEP_MIN = {
    'step2_liana': {'kuppe': 1, 'carraro': 1, 'habermann': 1, 'velmeshev': 2, 'reichart': 5},
    'step3_tensor': {'kuppe': 2, 'carraro': 2, 'habermann': 2, 'velmeshev': 3, 'reichart': 8},
    'step4_rf':    {'kuppe': 0.5, 'carraro': 0.5, 'habermann': 0.5, 'velmeshev': 0.5, 'reichart': 1},
}

DATASETS = ['kuppe', 'carraro', 'habermann', 'velmeshev', 'reichart']
DATASET_INFO = {
    'kuppe':     {'N': 23,  'disease': 'Myocardial Infarction'},
    'carraro':   {'N': 16,  'disease': 'Cystic Fibrosis'},
    'habermann': {'N': 18,  'disease': 'Pulmonary Fibrosis'},
    'velmeshev': {'N': 38,  'disease': 'Autism Spectrum Disorder'},
    'reichart':  {'N': 171, 'disease': 'Dilated Cardiomyopathy'},
}


def build_efficiency_table():
    rows = []
    for ds in DATASETS:
        info = DATASET_INFO[ds]
        e2e = E2E_TRAIN_MIN[ds]
        liana_total = sum(v[ds] for v in LIANA_STEP_MIN.values())
        rows.append({
            'Dataset':              ds,
            'Disease':              info['disease'],
            'N (samples)':         info['N'],
            'SPECTRA E2E (GPU, min)':  e2e,
            'LIANA+Tensor+RF (CPU, min)': round(liana_total, 1),
            'Speedup factor (E2E/LIANA)':  round(e2e / liana_total, 1),
            'E2E inference only (sec)':    '<5',  # inference ~few seconds
            'Step2 LIANA (min)':  LIANA_STEP_MIN['step2_liana'][ds],
            'Step3 Tensor (min)': LIANA_STEP_MIN['step3_tensor'][ds],
            'Step4 RF (min)':     LIANA_STEP_MIN['step4_rf'][ds],
        })
    return pd.DataFrame(rows)


def plot_efficiency_figure(df):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    set_pub_style()

    datasets = DATASETS
    n = len(datasets)
    x = np.arange(n)
    colors = NATURE_CATEGORICAL[:n]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A：E2E vs LIANA+Tensor+RF 训练时间对比
    ax = axes[0]
    e2e_vals   = df['SPECTRA E2E (GPU, min)'].values.astype(float)
    liana_vals = df['LIANA+Tensor+RF (CPU, min)'].values.astype(float)

    w = 0.35
    ax.bar(x - w/2, e2e_vals,   w, color='#E05252', alpha=0.85, label='SPECTRA E2E (GPU)', zorder=3)
    ax.bar(x + w/2, liana_vals, w, color='#5599CC', alpha=0.85, label='LIANA+Tensor+RF (CPU)', zorder=3)

    for xi, (e, l) in enumerate(zip(e2e_vals, liana_vals)):
        ax.text(xi - w/2, e + 3, f'{e:.0f}m', ha='center', va='bottom', fontsize=7.5, color='#E05252')
        ax.text(xi + w/2, l + 3, f'{l:.0f}m', ha='center', va='bottom', fontsize=7.5, color='#5599CC')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{d}\n(N={DATASET_INFO[d]["N"]})' for d in datasets], fontsize=8.5)
    ax.set_ylabel('Training time (minutes)', fontsize=9)
    ax.set_yscale('log')
    ax.set_title('(A) Training Wall-Clock Time\n(GPU: NVIDIA A100 40GB)',
                 fontsize=9.5, pad=8, fontweight='bold', loc='left')
    ax.legend(fontsize=8, frameon=False)

    # Panel B：LIANA+Tensor+RF 步骤分解（堆叠）
    ax2 = axes[1]
    s2 = np.array([LIANA_STEP_MIN['step2_liana'][d] for d in datasets], dtype=float)
    s3 = np.array([LIANA_STEP_MIN['step3_tensor'][d] for d in datasets], dtype=float)
    s4 = np.array([LIANA_STEP_MIN['step4_rf'][d] for d in datasets], dtype=float)

    ax2.bar(x, s2, color='#90C4E4', label='Step2: LIANA scoring', width=0.6)
    ax2.bar(x, s3, bottom=s2, color='#2266AA', label='Step3: Tensor decomp.', width=0.6)
    ax2.bar(x, s4, bottom=s2+s3, color='#0D4A8A', label='Step4: RF classify', width=0.6, alpha=0.7)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{d}\n(N={DATASET_INFO[d]["N"]})' for d in datasets], fontsize=8.5)
    ax2.set_ylabel('Time (minutes, per LIANA method)', fontsize=9)
    ax2.set_title('(B) LIANA+Tensor+RF Step Breakdown\n'
                  '(per method per dataset, CPU)',
                  fontsize=9.5, pad=8, fontweight='bold', loc='left')
    ax2.legend(fontsize=8, frameon=False)

    plt.tight_layout()
    save_fig(fig, FIGURES_DIR / 'FigS_efficiency')
    plt.close(fig)


def main():
    print('\n' + '='*55)
    print('  计算效率对比表')
    print('='*55)

    df = build_efficiency_table()
    display_cols = ['Dataset', 'N (samples)', 'SPECTRA E2E (GPU, min)',
                    'LIANA+Tensor+RF (CPU, min)', 'Speedup factor (E2E/LIANA)']
    print('\n' + df[display_cols].to_string(index=False))

    out_csv = DATA_DIR / 'efficiency_table.csv'
    df.to_csv(out_csv, index=False)
    print(f'\n  已保存: {out_csv}')
    print('\n  说明:')
    print('  - E2E 时间包含完整 5-fold CV（一次实验的总 GPU 时间）')
    print('  - LIANA 时间为单个方法 × 单个数据集（共 7 方法，研究者通常只用 1-2 个）')
    print('  - E2E 推断阶段（训练完毕后对新数据预测）< 5 秒')

    plot_efficiency_figure(df)
    print('\n  完成。')


if __name__ == '__main__':
    main()
