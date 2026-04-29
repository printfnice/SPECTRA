# run_reichart_ls.py
# 依赖: e2e_pipeline.py, e2e_utils.py
# 职责: reichart w=0.3 + Label Smoothing ε=0.1（零风险 +0.002-0.005 正则化）
#
# 背景: w03_mixoff=0.9641，LS 是 3行改动无副作用的叠加
# 断点: output/supcon_ablation/reichart_supcon_w03_ls01.csv

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
TAG = 'supcon_w03_ls01'

OVERRIDES = {
    'use_supcon':          True,
    'supcon_weight':       0.3,
    'supcon_tau':          0.07,
    'mc_dropout_T':        15,
    'use_mixup':           False,
    'early_stop_patience': 5,
    'label_smoothing':     0.1,
}

BASELINE_W03 = 0.9641
CELLCHAT     = 0.9655


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
    print(f'  reichart w=0.3 + Label Smoothing ε=0.1')
    print(f'  对比: w03_mixoff={BASELINE_W03}  目标: >{CELLCHAT}')
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
    print(f'  vs w03: {mean_auroc - BASELINE_W03:+.4f}')
    print(f'  vs CellChat: {mean_auroc - CELLCHAT:+.4f}')
    print(f'  fold AUROCs: {[round(a, 4) for a in aurocs]}')

    row = {
        'dataset': DATASET, 'tag': TAG,
        'supcon_weight': 0.3, 'supcon_tau': 0.07, 'label_smoothing': 0.1,
        'mc_dropout_T': 15, 'use_mixup': False,
        'mean_auroc': round(mean_auroc, 4), 'std_auroc': round(std_auroc, 4),
        'delta_vs_w03': round(mean_auroc - BASELINE_W03, 4),
        'delta_vs_cellchat': round(mean_auroc - CELLCHAT, 4),
        'n_folds': len(aurocs), 'fold_aurocs': str([round(a, 4) for a in aurocs]),
        'n_samples': n_samples, 'n_epochs': base_config['n_epochs'],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(str(out_path), index=False)
    print(f'  保存: {out_path}')


if __name__ == '__main__':
    main()
