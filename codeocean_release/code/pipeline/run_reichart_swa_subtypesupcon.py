# run_reichart_swa_subtypesupcon.py
# 依赖: e2e_pipeline.py, e2e_utils.py
# 职责: reichart SWA + 亚型双重 SupCon 实验
#
# 背景: w03_mixoff = 0.9641，距 CellChat 0.9655 差 0.0014
#       tau005 失败（fold1 0.9231），w04 提前终止（fold1 partial 0.9476 有希望但被误判停止）
#       SWA: 稳定后期训练（rs2 在 ep220 达 0.9766，ep300 跌至 0.7893）
#       亚型 SupCon: 针对 fold1/4 DCM 亚型构成薄弱，引入精细亚型正样本对
#
# 配置: disease_supcon_w=0.3, subtype_supcon_w=0.1, tau=0.07, SWA(80%), MixOFF
# 断点: output/supcon_ablation/reichart_swa_subtypesupcon.csv

import os
import sys
import warnings
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
TAG = 'swa_subtypesupcon'

OVERRIDES = {
    'use_supcon':            True,
    'supcon_weight':         0.3,    # disease-level SupCon（与 w03 一致）
    'supcon_tau':            0.07,
    'use_subtype_supcon':    True,   # 亚型级双重 SupCon
    'subtype_supcon_weight': 0.1,
    'use_swa':               True,   # Stochastic Weight Averaging
    'swa_start_frac':        0.8,    # 最后 20% epoch（即 ep240/300）开始 SWA
    'mc_dropout_T':          15,
    'use_mixup':             False,
    'early_stop_patience':   5,
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
    print(f'  reichart SWA + 亚型双重 SupCon 实验')
    print(f'  disease SupCon w=0.3, subtype SupCon w=0.1, SWA(80%)')
    print(f'  对比: w03_mixoff = {BASELINE_W03}')
    print(f'  目标: 超越 CellChat {CELLCHAT}')
    print(f'  设备: {device}')
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
    base_config.update(OVERRIDES)
    base_config['verbose'] = True

    print(f'  配置: disease_w={OVERRIDES["supcon_weight"]}, subtype_w={OVERRIDES["subtype_supcon_weight"]}, '
          f'tau={OVERRIDES["supcon_tau"]}, SWA={OVERRIDES["use_swa"]}, '
          f'swa_start={int(base_config["n_epochs"] * OVERRIDES["swa_start_frac"])}ep/{base_config["n_epochs"]}ep')

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

    cv_result = ep.run_e2e_cv(
        adata, samples, labels_str, all_batch_data,
        gene_to_idx, ct_to_idx, model_cfg, base_config, device
    )
    aurocs, _ = cv_result if isinstance(cv_result, tuple) else (cv_result, None)
    mean_auroc = float(np.mean(aurocs))
    std_auroc  = float(np.std(aurocs))

    print(f'\n  {TAG}  AUROC = {mean_auroc:.4f} ± {std_auroc:.4f}')
    print(f'  vs w03_mixoff (baseline): {mean_auroc - BASELINE_W03:+.4f}')
    print(f'  vs CellChat {CELLCHAT}: {mean_auroc - CELLCHAT:+.4f}')
    print(f'  fold AUROCs: {[round(a, 4) for a in aurocs]}')

    row = {
        'dataset': DATASET, 'tag': TAG,
        'use_supcon': True, 'supcon_weight': OVERRIDES['supcon_weight'],
        'supcon_tau': OVERRIDES['supcon_tau'],
        'use_subtype_supcon': True, 'subtype_supcon_weight': OVERRIDES['subtype_supcon_weight'],
        'use_swa': True, 'swa_start_frac': OVERRIDES['swa_start_frac'],
        'mc_dropout_T': OVERRIDES['mc_dropout_T'], 'use_mixup': False,
        'mean_auroc': round(mean_auroc, 4),
        'std_auroc':  round(std_auroc, 4),
        'delta_vs_w03': round(mean_auroc - BASELINE_W03, 4),
        'delta_vs_cellchat': round(mean_auroc - CELLCHAT, 4),
        'n_folds': len(aurocs),
        'fold_aurocs': str([round(a, 4) for a in aurocs]),
        'n_samples': n_samples,
        'n_epochs': base_config['n_epochs'],
        'early_stop_patience': OVERRIDES['early_stop_patience'],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(str(out_path), index=False)
    print(f'  保存: {out_path}')


if __name__ == '__main__':
    main()
