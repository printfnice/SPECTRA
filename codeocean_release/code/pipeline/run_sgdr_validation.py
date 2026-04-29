# run_sgdr_validation.py
# 职责: 验证 SGDR 统一配置在其他4个数据集（非reichart）上无退步
# 统一配置：SupCon(w=0.3) + VIB + MC_T15 + SGDR(T0=n_epochs//5) + warmup(n_epochs//10)
# 断点：output/{dataset}_ms_hgatlink_e2e_D_unified_sgdr.csv

import os, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')
code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))
os.chdir(Path(__file__).parent)

import liana as li
import e2e_pipeline as ep
from e2e_utils import build_unified_config

OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(exist_ok=True)

# 其他4个数据集（reichart 已单独运行 run_reichart_sgdr.py）
DATASETS = ['kuppe', 'carraro', 'habermann', 'velmeshev']

# E2E 线基线（w02_mixoff / w03_mixoff，用于退步检测）
BASELINES = {
    'kuppe':     1.0000,
    'carraro':   0.9500,
    'habermann': 0.9429,
    'velmeshev': 0.8608,
}
REGRESSION_THRESH = 0.02  # 退步超过此值标记警告


def run_one_dataset(dataset_name, device):
    out_path = OUTPUT_DIR / f'{dataset_name}_ms_hgatlink_e2e_D_unified_sgdr.csv'
    if out_path.exists():
        df = pd.read_csv(str(out_path))
        auroc = float(df['mean_auroc'].iloc[0])
        print(f'[skip] {dataset_name}  AUROC={auroc:.4f}')
        return auroc

    print(f'\n{"="*55}')
    print(f'  {dataset_name} | Unified SGDR 验证')
    print(f'  基线: {BASELINES[dataset_name]:.4f}  退步阈值: -{REGRESSION_THRESH}')
    print(f'{"="*55}')

    adata = ep._load_adata(dataset_name)
    dk = ep._DATASET_KEYS[dataset_name]
    samples, labels_str = ep._get_sample_labels(adata, dk['sample_key'], dk['condition_key'])
    n_samples = len(samples)

    config = ep.DEFAULT_CONFIG.copy()
    config['dataset_name'] = dataset_name
    config['unified'] = True
    config.update(build_unified_config(n_samples, dataset_name))
    config['verbose'] = True

    print(f'  N={n_samples}  n_epochs={config["n_epochs"]}  '
          f'T0={config["warm_restart_T0"]}  warmup={config["lr_warmup_epochs"]}  '
          f'pma_k={config["pma_k"]}  lr_pairs={config["e2e_max_lr_pairs"]}')

    gene_to_idx, ct_to_idx, _, _ = ep._build_gene_ct_index(adata, dk['groupby'])
    resource = li.resource.select_resource(config['resource_name'])
    model_cfg = ep._get_model_config(dataset_name, 30)
    model_cfg['gene_emb_dim'] = 64
    model_cfg['dropout'] = 0.30

    all_batch_data = ep._prepare_all_samples(
        samples, adata, resource, gene_to_idx, ct_to_idx,
        model_cfg, config, dk['sample_key'], dk['groupby'],
    )
    n_valid = sum(1 for bd in all_batch_data if bd is not None)
    print(f'  有效样本: {n_valid}/{n_samples}')

    cv_result = ep.run_e2e_cv(
        adata, samples, labels_str, all_batch_data,
        gene_to_idx, ct_to_idx, model_cfg, config, device
    )
    aurocs, _ = cv_result if isinstance(cv_result, tuple) else (cv_result, None)
    mean_auroc = float(np.mean(aurocs))
    std_auroc  = float(np.std(aurocs))
    baseline   = BASELINES[dataset_name]
    delta      = mean_auroc - baseline

    print(f'\n  {dataset_name}  AUROC = {mean_auroc:.4f} ± {std_auroc:.4f}')
    print(f'  vs baseline: {delta:+.4f}', end='')
    if delta < -REGRESSION_THRESH:
        print(f'  ⚠️  REGRESSION! 超过阈值 {REGRESSION_THRESH}')
    elif delta < 0:
        print(f'  (轻微退步，在阈值内)')
    else:
        print(f'  ✅ 无退步')
    print(f'  fold AUROCs: {[round(a, 4) for a in aurocs]}')

    row = {
        'dataset': dataset_name, 'tag': 'unified_sgdr',
        'mean_auroc': round(mean_auroc, 4), 'std_auroc': round(std_auroc, 4),
        'baseline': baseline, 'delta': round(delta, 4),
        'n_folds': len(aurocs), 'fold_aurocs': str([round(a, 4) for a in aurocs]),
        'n_samples': n_samples, 'n_epochs': config['n_epochs'],
        'warm_restart_T0': config['warm_restart_T0'],
        'lr_warmup_epochs': config['lr_warmup_epochs'],
        'use_supcon': True, 'supcon_weight': 0.3,
    }
    pd.DataFrame([row]).to_csv(str(out_path), index=False)
    print(f'  保存: {out_path}')
    return mean_auroc


def main():
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\n{"="*55}')
    print(f'  SGDR 统一配置验证：其他4个数据集')
    print(f'  目标：无退步（delta > -{REGRESSION_THRESH}）')
    print(f'  设备: {device}')
    print(f'{"="*55}')

    results = {}
    for ds in DATASETS:
        results[ds] = run_one_dataset(ds, device)

    print(f'\n{"="*55}')
    print(f'  汇总结果')
    print(f'{"="*55}')
    all_ok = True
    for ds, auroc in results.items():
        baseline = BASELINES[ds]
        delta = auroc - baseline
        flag = '✅' if delta >= -REGRESSION_THRESH else '⚠️ REGRESSION'
        print(f'  {ds:12s}  {auroc:.4f}  (vs {baseline:.4f}  {delta:+.4f})  {flag}')
        if delta < -REGRESSION_THRESH:
            all_ok = False
    print(f'\n  总体: {"✅ 全部通过" if all_ok else "⚠️ 存在退步，请检查"}')


if __name__ == '__main__':
    main()
