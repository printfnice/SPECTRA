# run_mixup_ablation.py
# 依赖: e2e_pipeline.py, e2e_utils.py
# 被依赖: run_all_supplementary.sh
# 职责: N≤38 数据集 Mixup on/off 消融实验（Zhang et al. 2018 规则验证）
#
# 实验设计:
#   - 数据集: kuppe(23), carraro(16), habermann(18), velmeshev(38)
#   - 默认规则: use_mixup = (n_samples > 50) → 这四个数据集 Mixup=False
#   - 消融: 强制开启 Mixup，检验是否真的有害
#   - 结论预期: Mixup=False 优于 Mixup=True (支持 Zhang 2018 规则)

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))
os.chdir(Path(__file__).parent)

import liana as li
import e2e_pipeline as ep
from e2e_utils import build_unified_config, save_e2e_results

# ============================================================================
# 配置
# ============================================================================

SMALL_DATASETS = ['kuppe', 'carraro', 'habermann', 'velmeshev']
OUTPUT_DIR = Path('output/mixup_ablation')


# ============================================================================
# 核心逻辑
# ============================================================================

def _out_path(dataset, mixup):
    tag = 'mixup_on' if mixup else 'mixup_off'
    return OUTPUT_DIR / f'{dataset}_{tag}.csv'


def _run_one(dataset, mixup_on, device='cuda'):
    out_path = _out_path(dataset, mixup_on)
    if out_path.exists():
        df = pd.read_csv(str(out_path))
        print(f"  [skip] {dataset}/mixup={mixup_on}  已存在: {out_path}")
        return df

    print(f"\n{'='*60}")
    tag = 'ON' if mixup_on else 'OFF'
    print(f"  {dataset} | Mixup={tag}  [{datetime.now().strftime('%H:%M:%S')}]")
    print(f"{'='*60}")

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载数据
    adata = ep._load_adata(dataset)
    dk = ep._DATASET_KEYS[dataset]
    samples, labels_str = ep._get_sample_labels(adata, dk['sample_key'], dk['condition_key'])
    n_samples = len(samples)
    print(f"  样本数={n_samples}")

    # 构建配置 (unified 模式)
    config = ep.DEFAULT_CONFIG.copy()
    config['dataset_name'] = dataset
    config['unified'] = True
    config.update(build_unified_config(n_samples, dataset))
    # 消融：强制覆盖 use_mixup
    config['use_mixup'] = mixup_on
    config['verbose'] = False

    # 模型配置
    gene_to_idx, ct_to_idx, _, _ = ep._build_gene_ct_index(adata, dk['groupby'])
    resource = li.resource.select_resource(config['resource_name'])
    model_cfg = ep._get_model_config(dataset, 30)
    model_cfg['gene_emb_dim'] = 64
    model_cfg['dropout'] = 0.30

    print(f"  预计算 batch_data (use_mixup={config['use_mixup']})...")
    all_batch_data = ep._prepare_all_samples(
        samples, adata, resource, gene_to_idx, ct_to_idx,
        model_cfg, config, dk['sample_key'], dk['groupby'],
        lr_pathway_map=None,
    )
    n_valid = sum(1 for bd in all_batch_data if bd is not None)
    print(f"  有效样本: {n_valid}/{n_samples}")

    cv_result = ep.run_e2e_cv(
        adata, samples, labels_str, all_batch_data,
        gene_to_idx, ct_to_idx, model_cfg, config, device
    )
    aurocs, _ = cv_result if isinstance(cv_result, tuple) else (cv_result, None)
    mean_auroc = np.mean(aurocs)
    std_auroc = np.std(aurocs)
    print(f"  AUROC = {mean_auroc:.4f} ± {std_auroc:.4f}  (各折: {[f'{a:.4f}' for a in aurocs]})")

    row = {
        'dataset': dataset, 'use_mixup': mixup_on, 'mean_auroc': round(mean_auroc, 4),
        'std_auroc': round(std_auroc, 4), 'n_folds': len(aurocs),
        'fold_aurocs': str([round(a, 4) for a in aurocs]),
        'n_samples': n_samples,
        'n_epochs': config['n_epochs'], 'pma_k': config['pma_k'],
        'mixup_alpha': config.get('mixup_alpha', 0.0),
    }
    df = pd.DataFrame([row])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_path), index=False)
    print(f"  保存: {out_path}")
    return df


def main():
    import argparse
    import torch
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='all')
    p.add_argument('--no-gpu', action='store_true')
    args, _ = p.parse_known_args()

    device = 'cpu' if args.no_gpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = SMALL_DATASETS if args.dataset == 'all' else [args.dataset]

    all_rows = []
    for dataset in datasets:
        for mixup_on in [False, True]:
            try:
                df = _run_one(dataset, mixup_on, device)
                if df is not None:
                    all_rows.append(df)
            except Exception as e:
                print(f"  ERROR {dataset}/mixup={mixup_on}: {e}")
                import traceback
                traceback.print_exc()

    if all_rows:
        summary = pd.concat(all_rows, ignore_index=True)
        summary_path = OUTPUT_DIR / 'summary_mixup_ablation.csv'
        summary.to_csv(str(summary_path), index=False)
        print(f"\n汇总: {summary_path}")
        # 打印对比表
        print("\n=== Mixup 消融结果 ===")
        pivot = summary.pivot_table(
            index='dataset', columns='use_mixup', values='mean_auroc', aggfunc='first'
        )
        pivot.columns = ['Mixup=False', 'Mixup=True']
        pivot['delta'] = pivot['Mixup=True'] - pivot['Mixup=False']
        print(pivot.round(4))


if __name__ == '__main__':
    main()
