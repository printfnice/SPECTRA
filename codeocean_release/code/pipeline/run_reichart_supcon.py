# run_reichart_supcon.py
# 依赖: e2e_pipeline.py (use_supcon 已实现), e2e_utils.py
# 被依赖: run_after_main_suite.sh
# 职责: reichart SupCon消融——验证监督对比损失对跨DCM亚型泛化的提升效果
#
# 对比基线: R38 (Mixup+VIB+MC, use_supcon=False) AUROC=0.9264
# 实验组:   +SupCon(weight=0.1, tau=0.07), +SupCon(weight=0.2), +MC_T30
# 理论依据: SupCon拉近同类DCM样本嵌入，直接对抗fold1/fold2的亚型泛化失败
#           MC_T=30 增强 Bayesian 不确定性估计（单模型，非集成）

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

OUTPUT_DIR = Path('output/supcon_ablation')
DATASET = 'reichart'

# 实验��置矩阵
EXPERIMENTS = [
    # (tag, overrides)
    ('baseline_R38',    {'use_supcon': False, 'mc_dropout_T': 15}),
    ('supcon_w01',      {'use_supcon': True,  'supcon_weight': 0.1,  'supcon_tau': 0.07, 'mc_dropout_T': 15}),
    ('supcon_w02',      {'use_supcon': True,  'supcon_weight': 0.2,  'supcon_tau': 0.07, 'mc_dropout_T': 15}),
    ('supcon_w01_T30',  {'use_supcon': True,  'supcon_weight': 0.1,  'supcon_tau': 0.07, 'mc_dropout_T': 30}),
    ('mc_T30_only',     {'use_supcon': False, 'mc_dropout_T': 30}),
]


def _out_path(tag):
    return OUTPUT_DIR / f'{DATASET}_{tag}.csv'


def _run_one(tag, overrides, adata, samples, labels_str, gene_to_idx,
             ct_to_idx, model_cfg, base_config, all_batch_data, device):
    out_path = _out_path(tag)
    if out_path.exists():
        df = pd.read_csv(str(out_path))
        print(f'  [skip] {tag}  AUROC={df["mean_auroc"].iloc[0]:.4f}')
        return df

    print(f'\n{"="*60}')
    print(f'  {DATASET} | {tag}  [{datetime.now().strftime("%H:%M:%S")}]')
    print(f'{"="*60}')

    config = base_config.copy()
    config.update(overrides)
    config['verbose'] = False

    for k, v in overrides.items():
        print(f'  {k} = {v}')

    cv_result = ep.run_e2e_cv(
        adata, samples, labels_str, all_batch_data,
        gene_to_idx, ct_to_idx, model_cfg, config, device
    )
    aurocs, _ = cv_result if isinstance(cv_result, tuple) else (cv_result, None)
    mean_auroc = float(np.mean(aurocs))
    std_auroc = float(np.std(aurocs))
    print(f'\n  {tag}  AUROC = {mean_auroc:.4f} ± {std_auroc:.4f}')
    print(f'  vs R38 baseline 0.9264: {mean_auroc - 0.9264:+.4f}')
    print(f'  fold AUROCs: {[round(a, 4) for a in aurocs]}')

    row = {
        'dataset': DATASET, 'tag': tag,
        'use_supcon': overrides.get('use_supcon', False),
        'supcon_weight': overrides.get('supcon_weight', 0.0),
        'supcon_tau': overrides.get('supcon_tau', 0.07),
        'mc_dropout_T': overrides.get('mc_dropout_T', 15),
        'mean_auroc': round(mean_auroc, 4),
        'std_auroc': round(std_auroc, 4),
        'delta_vs_R38': round(mean_auroc - 0.9264, 4),
        'n_folds': len(aurocs),
        'fold_aurocs': str([round(a, 4) for a in aurocs]),
        'n_samples': len(samples),
        'n_epochs': config['n_epochs'],
        'pma_k': config['pma_k'],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(str(out_path), index=False)
    print(f'  保存: {out_path}')
    return df


def main():
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'\n{"="*60}')
    print(f'  reichart SupCon 消融实验')
    print(f'  共 {len(EXPERIMENTS)} 组，设备: {device}')
    print(f'{"="*60}')

    # 预加载数据（所有实验共用，只加载一次）
    adata = ep._load_adata(DATASET)
    dk = ep._DATASET_KEYS[DATASET]
    samples, labels_str = ep._get_sample_labels(adata, dk['sample_key'], dk['condition_key'])
    n_samples = len(samples)
    print(f'  样本数={n_samples}')

    base_config = ep.DEFAULT_CONFIG.copy()
    base_config['dataset_name'] = DATASET
    base_config['unified'] = True
    base_config.update(build_unified_config(n_samples, DATASET))

    gene_to_idx, ct_to_idx, _, _ = ep._build_gene_ct_index(adata, dk['groupby'])
    resource = li.resource.select_resource(base_config['resource_name'])
    model_cfg = ep._get_model_config(DATASET, 30)
    model_cfg['gene_emb_dim'] = 64
    model_cfg['dropout'] = 0.30

    print(f'  预计算 batch_data（全部实验共用）...')
    all_batch_data = ep._prepare_all_samples(
        samples, adata, resource, gene_to_idx, ct_to_idx,
        model_cfg, base_config, dk['sample_key'], dk['groupby'],
        lr_pathway_map=None,
    )
    n_valid = sum(1 for bd in all_batch_data if bd is not None)
    print(f'  有效样本: {n_valid}/{n_samples}')

    # 依次运行各实验
    all_rows = []
    for tag, overrides in EXPERIMENTS:
        try:
            df = _run_one(tag, overrides, adata, samples, labels_str,
                          gene_to_idx, ct_to_idx, model_cfg,
                          base_config, all_batch_data, device)
            if df is not None:
                all_rows.append(df)
        except Exception as e:
            print(f'  ERROR [{tag}]: {e}')
            import traceback
            traceback.print_exc()

    # 汇总
    if all_rows:
        summary = pd.concat(all_rows, ignore_index=True)
        out = OUTPUT_DIR / f'{DATASET}_supcon_summary.csv'
        summary.to_csv(str(out), index=False)
        print(f'\n{"="*60}')
        print(f'  SupCon 消融结果汇总')
        print(f'{"="*60}')
        cols = ['tag', 'mean_auroc', 'std_auroc', 'delta_vs_R38', 'use_supcon', 'supcon_weight', 'mc_dropout_T']
        print(summary[cols].sort_values('mean_auroc', ascending=False).to_string(index=False))
        print(f'\n  保存: {out}')


if __name__ == '__main__':
    main()
