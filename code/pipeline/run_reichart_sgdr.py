# run_reichart_sgdr.py
# 职责: w03_finepeak + SGDR(T0=100) + LR Warmup(30ep) + n_epochs=500
#   根因: fold1 早期训练震荡(ep10=0.41) + 300ep 不够(峰值在ep280-290)
#   SGDR: 每100ep重启LR，帮助逃出局部最小值，5次重启
#   Warmup: 前30ep从lr/10线性升至lr，消除早期震荡
#   500ep: 给fold1足够时间收敛(原300ep峰值在最后10ep)
# 断点: output/supcon_ablation/reichart_sgdr.csv

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

OUTPUT_DIR = Path('output/supcon_ablation')
DATASET = 'reichart'
TAG = 'sgdr'

OVERRIDES = {
    'use_supcon':          True,
    'supcon_weight':       0.3,
    'supcon_tau':          0.07,
    'mc_dropout_T':        15,
    'use_mixup':           False,
    'early_stop_patience': 0,
    'eval_interval':       10,
    'n_random_states':     3,
    'n_epochs':            500,
    'use_warm_restarts':   True,
    'warm_restart_T0':     100,
    'lr_warmup_epochs':    30,
}

FINEPEAK = 0.9654
CELLCHAT = 0.9655


def main():
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    out_path = OUTPUT_DIR / f'{DATASET}_{TAG}.csv'
    if out_path.exists():
        df = pd.read_csv(str(out_path))
        auroc = df['mean_auroc'].iloc[0]
        print(f'[skip] {TAG}  AUROC={float(auroc):.4f}')
        return

    print(f'\n{"="*60}')
    print(f'  reichart SGDR(T0=100) + Warmup(30ep) + 500ep')
    print(f'  根因修复: 早期震荡+300ep不足')
    print(f'  对比: finepeak={FINEPEAK}  目标: >{CELLCHAT}')
    print(f'  设备: {device}')
    print(f'{"="*60}')

    adata = ep._load_adata(DATASET)
    dk = ep._DATASET_KEYS[DATASET]
    samples, labels_str = ep._get_sample_labels(adata, dk['sample_key'], dk['condition_key'])
    n_samples = len(samples)

    base_config = ep.DEFAULT_CONFIG.copy()
    base_config['dataset_name'] = DATASET
    base_config['unified'] = True
    base_config.update(build_unified_config(n_samples, DATASET))
    base_config.update(OVERRIDES)
    base_config['verbose'] = True

    gene_to_idx, ct_to_idx, _, _ = ep._build_gene_ct_index(adata, dk['groupby'])
    resource = li.resource.select_resource(base_config['resource_name'])
    model_cfg = ep._get_model_config(DATASET, 30)
    model_cfg['gene_emb_dim'] = 64
    model_cfg['dropout'] = 0.30

    all_batch_data = ep._prepare_all_samples(
        samples, adata, resource, gene_to_idx, ct_to_idx,
        model_cfg, base_config, dk['sample_key'], dk['groupby'],
        lr_pathway_map=None,
    )

    cv_result = ep.run_e2e_cv(
        adata, samples, labels_str, all_batch_data,
        gene_to_idx, ct_to_idx, model_cfg, base_config, device
    )
    aurocs, _ = cv_result if isinstance(cv_result, tuple) else (cv_result, None)
    mean_auroc = float(np.mean(aurocs))
    std_auroc  = float(np.std(aurocs))

    print(f'\n  {TAG}  AUROC = {mean_auroc:.4f} ± {std_auroc:.4f}')
    print(f'  vs finepeak3: {mean_auroc - FINEPEAK:+.4f}')
    print(f'  vs CellChat:  {mean_auroc - CELLCHAT:+.4f}')
    print(f'  fold AUROCs: {[round(a, 4) for a in aurocs]}')

    row = {
        'dataset': DATASET, 'tag': TAG,
        'supcon_weight': 0.3, 'supcon_tau': 0.07,
        'use_warm_restarts': True, 'warm_restart_T0': 100,
        'lr_warmup_epochs': 30, 'n_epochs': 500,
        'eval_interval': 10, 'early_stop_patience': 0,
        'mc_dropout_T': 15, 'use_mixup': False, 'n_random_states': 3,
        'mean_auroc': round(mean_auroc, 4), 'std_auroc': round(std_auroc, 4),
        'delta_vs_finepeak': round(mean_auroc - FINEPEAK, 4),
        'delta_vs_cellchat': round(mean_auroc - CELLCHAT, 4),
        'n_folds': len(aurocs), 'fold_aurocs': str([round(a, 4) for a in aurocs]),
        'n_samples': n_samples,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(str(out_path), index=False)
    print(f'  保存: {out_path}')


if __name__ == '__main__':
    main()
