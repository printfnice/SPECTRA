# run_supcon_all_datasets.py
# 依赖: e2e_pipeline.py, e2e_utils.py
# 被依赖: run_after_main_suite.sh (Extra G)
# 职责: 在全部5个数据集上统一运行 SupCon w=0.2 + Mixup=OFF
#
# 目的: 确认 SupCon 作为统一配置对其他4个数据集无退步（无 cherry-picking）
#       若无退步，论文可将 SupCon(w=0.2)+Mixup=OFF 写入统一配置
#
# 断点: output/supcon_all_datasets/{dataset}_supcon_w02_mixoff.csv
# 退步阈值: 若任意数据集 AUROC 下降 > 0.02，打印 WARNING

import os
import sys
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
from e2e_utils import build_unified_config

OUTPUT_DIR = Path('output/supcon_all_datasets')

# E2E baseline AUROC（来自 CLAUDE.md）
BASELINES = {
    'kuppe':      1.0000,
    'carraro':    0.9357,
    'habermann':  0.9571,
    'velmeshev':  0.8665,
    'reichart':   0.9264,   # R38 E2E baseline（SupCon目标是超越这个）
}

REGRESSION_THRESHOLD = 0.02

TAG = 'supcon_w02_mixoff'
SUPCON_OVERRIDES = {
    'use_supcon':    True,
    'supcon_weight': 0.2,
    'supcon_tau':    0.07,
    'mc_dropout_T':  15,
    'use_mixup':     False,
}

DATASETS = ['kuppe', 'carraro', 'habermann', 'velmeshev', 'reichart']


def run_one_dataset(dataset):
    out_path = OUTPUT_DIR / f'{dataset}_{TAG}.csv'
    if out_path.exists():
        df = pd.read_csv(str(out_path))
        auroc_val = df['mean_auroc'].iloc[0]
        print(f'  [skip] {dataset}  AUROC={float(auroc_val):.4f}')
        return float(auroc_val)

    print(f'\n{"="*60}')
    print(f'  {dataset} | {TAG}  [{datetime.now().strftime("%H:%M:%S")}]')
    print(f'{"="*60}')

    adata = ep._load_adata(dataset)
    dk = ep._DATASET_KEYS[dataset]
    samples, labels_str = ep._get_sample_labels(adata, dk['sample_key'], dk['condition_key'])
    n_samples = len(samples)
    print(f'  样本数={n_samples}')

    base_config = ep.DEFAULT_CONFIG.copy()
    base_config['dataset_name'] = dataset
    base_config['unified'] = True
    base_config.update(build_unified_config(n_samples, dataset))
    base_config.update(SUPCON_OVERRIDES)
    base_config['verbose'] = False

    gene_to_idx, ct_to_idx, _, _ = ep._build_gene_ct_index(adata, dk['groupby'])
    resource = li.resource.select_resource(base_config['resource_name'])
    model_cfg = ep._get_model_config(dataset, 30)
    model_cfg['gene_emb_dim'] = 64
    model_cfg['dropout'] = 0.30

    all_batch_data = ep._prepare_all_samples(
        samples, adata, resource, gene_to_idx, ct_to_idx,
        model_cfg, base_config, dk['sample_key'], dk['groupby'],
        lr_pathway_map=None,
    )
    n_valid = sum(1 for bd in all_batch_data if bd is not None)
    print(f'  有效样本: {n_valid}/{n_samples}')

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cv_result = ep.run_e2e_cv(
        adata, samples, labels_str, all_batch_data,
        gene_to_idx, ct_to_idx, model_cfg, base_config, device
    )
    aurocs, _ = cv_result if isinstance(cv_result, tuple) else (cv_result, None)
    mean_auroc = float(np.mean(aurocs))
    std_auroc = float(np.std(aurocs))

    baseline = BASELINES.get(dataset, 0.0)
    delta = mean_auroc - baseline
    print(f'\n  {dataset}  AUROC = {mean_auroc:.4f} ± {std_auroc:.4f}')
    print(f'  vs baseline {baseline:.4f}: {delta:+.4f}')
    if delta < -REGRESSION_THRESHOLD:
        print(f'  WARNING: REGRESSION > {REGRESSION_THRESHOLD} !!!')
    print(f'  fold AUROCs: {[round(a, 4) for a in aurocs]}')

    row = {
        'dataset': dataset, 'tag': TAG,
        'mean_auroc': round(mean_auroc, 4),
        'std_auroc': round(std_auroc, 4),
        'baseline_auroc': baseline,
        'delta_vs_baseline': round(delta, 4),
        'regression_flag': delta < -REGRESSION_THRESHOLD,
        'n_folds': len(aurocs),
        'fold_aurocs': str([round(a, 4) for a in aurocs]),
        'n_samples': n_samples,
        'n_epochs': base_config['n_epochs'],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(str(out_path), index=False)
    print(f'  保存: {out_path}')
    return mean_auroc


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='all')
    args, _ = p.parse_known_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]

    print(f'\n{"="*60}')
    print(f'  SupCon(w=0.2)+Mixup=OFF — 全数据集验证')
    print(f'  目标: 确认可作为统一配置无退步')
    print(f'  退步阈值: {REGRESSION_THRESHOLD}')
    print(f'{"="*60}')

    results = {}
    for ds in datasets:
        try:
            auroc = run_one_dataset(ds)
            results[ds] = auroc
        except Exception as e:
            print(f'  ERROR [{ds}]: {e}')
            import traceback
            traceback.print_exc()
            results[ds] = None

    print(f'\n{"="*60}')
    print(f'  汇总结果 (SupCon w=0.2 + Mixup=OFF)')
    print(f'{"="*60}')
    regressions = []
    for ds in DATASETS:
        auroc = results.get(ds)
        baseline = BASELINES.get(ds, 0.0)
        if auroc is not None:
            delta = auroc - baseline
            flag = ' WARNING:REGRESSION' if delta < -REGRESSION_THRESHOLD else ''
            print(f'  {ds:12s}  {auroc:.4f} (baseline={baseline:.4f}, delta={delta:+.5f}){flag}')
            if delta < -REGRESSION_THRESHOLD:
                regressions.append(ds)
        else:
            print(f'  {ds:12s}  FAILED')

    if regressions:
        print(f'\n  REGRESSION detected on: {regressions}')
        print(f'  SupCon w=0.2+MixOFF 不适合作为统一配置！')
    else:
        print(f'\n  所有数据集无退步，SupCon w=0.2+MixOFF 可作为统一配置')

    # 更新 summary
    csvs = list(OUTPUT_DIR.glob(f'*_{TAG}.csv'))
    if csvs:
        all_df = pd.concat([pd.read_csv(str(f)) for f in csvs], ignore_index=True)
        all_df.to_csv(str(OUTPUT_DIR / 'summary_all.csv'), index=False)
        print(f'  汇总: {OUTPUT_DIR}/summary_all.csv')


if __name__ == '__main__':
    main()
