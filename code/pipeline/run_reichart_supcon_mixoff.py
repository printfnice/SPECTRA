# run_reichart_supcon_mixoff.py
# 依赖: e2e_pipeline.py, e2e_utils.py
# 被依赖: run_after_main_suite.sh (Extra F)
# 职责: reichart SupCon+Mixup=OFF 组合消融
#
# ��景: SupCon w=0.2 已达 0.9566（使用 Mixup=ON）
#       Mixup=OFF 单独可达 0.9393
#       两者组合是否可突破 CellChat 公平基线 0.9655？
#
# 实验组:
#   supcon_w02_mixoff: SupCon weight=0.2, Mixup=OFF
#   supcon_w03_mixoff: SupCon weight=0.3, Mixup=OFF（更强对比损失）

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

EXPERIMENTS = [
    # (tag, overrides)
    ('supcon_w02_mixoff', {'use_supcon': True, 'supcon_weight': 0.2, 'supcon_tau': 0.07,
                           'mc_dropout_T': 15, 'use_mixup': False}),
    ('supcon_w03_mixoff', {'use_supcon': True, 'supcon_weight': 0.3, 'supcon_tau': 0.07,
                           'mc_dropout_T': 15, 'use_mixup': False}),
]


def _out_path(tag):
    return OUTPUT_DIR / f'{DATASET}_{tag}.csv'


def _run_one(tag, overrides, adata, samples, labels_str, gene_to_idx,
             ct_to_idx, model_cfg, base_config, all_batch_data, device):
    out_path = _out_path(tag)
    if out_path.exists():
        df = pd.read_csv(str(out_path))
        auroc_val = df['mean_auroc'].iloc[0]
        if str(auroc_val) != 'SKIPPED':
            print(f'  [skip] {tag}  AUROC={float(auroc_val):.4f}')
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
    print(f'  vs supcon_w02(ON) 0.9566: {mean_auroc - 0.9566:+.4f}')
    print(f'  vs CellChat 0.9655: {mean_auroc - 0.9655:+.4f}')
    print(f'  fold AUROCs: {[round(a, 4) for a in aurocs]}')

    row = {
        'dataset': DATASET, 'tag': tag,
        'use_supcon': overrides.get('use_supcon', False),
        'supcon_weight': overrides.get('supcon_weight', 0.0),
        'supcon_tau': overrides.get('supcon_tau', 0.07),
        'mc_dropout_T': overrides.get('mc_dropout_T', 15),
        'use_mixup': overrides.get('use_mixup', True),
        'mean_auroc': round(mean_auroc, 4),
        'std_auroc': round(std_auroc, 4),
        'delta_vs_supcon_w02': round(mean_auroc - 0.9566, 4),
        'delta_vs_cellchat': round(mean_auroc - 0.9655, 4),
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
    print(f'  reichart SupCon+Mixup=OFF 组合消融')
    print(f'  目标: 突破 CellChat 公平基线 0.9655')
    print(f'  当前最佳单模型: supcon_w02(ON)=0.9566')
    print(f'  共 {len(EXPERIMENTS)} 组，设备: {device}')
    print(f'{"="*60}')

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

    print(f'  预计算 batch_data...')
    all_batch_data = ep._prepare_all_samples(
        samples, adata, resource, gene_to_idx, ct_to_idx,
        model_cfg, base_config, dk['sample_key'], dk['groupby'],
        lr_pathway_map=None,
    )
    n_valid = sum(1 for bd in all_batch_data if bd is not None)
    print(f'  有效样本: {n_valid}/{n_samples}')

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

    if all_rows:
        summary = pd.concat(all_rows, ignore_index=True)
        out = OUTPUT_DIR / f'{DATASET}_supcon_mixoff_summary.csv'
        summary.to_csv(str(out), index=False)
        cols = ['tag', 'mean_auroc', 'std_auroc', 'delta_vs_cellchat', 'use_mixup', 'supcon_weight']
        print(f'\n  结果汇总:')
        print(summary[cols].sort_values('mean_auroc', ascending=False).to_string(index=False))
        print(f'  保存: {out}')


if __name__ == '__main__':
    main()
