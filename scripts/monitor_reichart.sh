#!/bin/bash
# 监控reichart运行并生成最终汇总

echo "等待reichart完成..."

while true; do
    # 检查进程是否还在运行
    if ps aux | grep "10971" | grep -v grep > /dev/null; then
        echo "[$(date +%H:%M:%S)] reichart还在运行中..."
        sleep 30
    else
        echo "[$(date +%H:%M:%S)] reichart已完成！"
        break
    fi
done

# 检查输出文件
if [ -f "../output/reichart_ms_hgatlink_mofa.csv" ]; then
    echo "✓ reichart结果文件已生成"

    # 生成最终汇总
    conda run -n LIANA python << 'EOF'
import pandas as pd
import numpy as np

# 读取所有5个数据集
datasets = ['carraro', 'habermann', 'kuppe', 'reichart', 'velmeshev']
all_results = []

print("="*80)
print("MS-HGATLink + MOFA+ 完整基准测试结果（5/5数据集）")
print("="*80)

for dataset in datasets:
    try:
        df = pd.read_csv(f'../output/{dataset}_ms_hgatlink_mofa.csv')
        mean_auroc = df['auroc'].mean()
        std_auroc = df['auroc'].std()
        mean_f1 = df['f1_score'].mean()
        std_f1 = df['f1_score'].std()

        print(f'\n{dataset:12s}:')
        print(f'  AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}')
        print(f'  F1:    {mean_f1:.4f} ± {std_f1:.4f}')
        print(f'  Folds: {len(df)}/10')

        all_results.append({
            'dataset': dataset,
            'mean_auroc': mean_auroc,
            'std_auroc': std_auroc,
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'n_folds': len(df)
        })
    except Exception as e:
        print(f'\n{dataset:12s}: ❌ 失败 ({e})')

summary_df = pd.DataFrame(all_results)

print(f'\n{"="*80}')
print(f'跨数据集平均:')
print(f'{"="*80}')
print(f'平均AUROC: {summary_df["mean_auroc"].mean():.4f} ± {summary_df["mean_auroc"].std():.4f}')
print(f'平均F1:    {summary_df["mean_f1"].mean():.4f} ± {summary_df["mean_f1"].std():.4f}')
print(f'完成数据集: {len(summary_df)}/5')

# 保存最终汇总
summary_df.to_csv('../output/ms_hgatlink_mofa_summary_final.csv', index=False)
print(f'\n✓ 最终汇总已保存: ../output/ms_hgatlink_mofa_summary_final.csv')
EOF

else
    echo "❌ reichart结果文件未生成"
    exit 1
fi
