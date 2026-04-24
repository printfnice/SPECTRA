# run_ensemble.py
# 依赖: e2e_pipeline.py, run_ensemble_reichart.py (复用核心函数)
# 被依赖: 直接运行（命令行 --dataset 参数）
# 职责: 通用 E2E + NMF/RF 集成流水线（支持全部 5 个数据集）
#       与 run_ensemble_reichart.py 共享核心逻辑，额外支持 CLI

import sys
import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')

_PIPELINE_DIR = Path(__file__).parent
_CODE_DIR = _PIPELINE_DIR.parent
_BASE_DIR = _CODE_DIR.parent
sys.path.insert(0, str(_CODE_DIR))

from pipeline.e2e_pipeline import (
    DEFAULT_CONFIG, _DATASET_KEYS,
    _get_device, _load_adata, _get_sample_labels, _prepare_all_samples,
)
from pipeline.e2e_utils import build_unified_config
from pipeline.run_ensemble_reichart import run_ensemble_cv
from model.e2e_modules import SampleAggregator, VIBLayer, E2EClassifier
from model.pathway_utils import build_lr_pathway_map
from model.ms_hgatlink_method import _build_gene_ct_index, _get_model_config

_DATA_DIR = _BASE_DIR / 'data'
_OUTPUT_DIR = _BASE_DIR / 'output'

_REICHART_ONLY = {'reichart'}    # 需要 pathway_prior 的数据集


def _parse_args():
    p = argparse.ArgumentParser(description='Ensemble E2E+NMF/RF Pipeline')
    p.add_argument('--dataset', required=True,
                   choices=['kuppe', 'reichart', 'habermann', 'velmeshev', 'carraro'])
    p.add_argument('--no_gpu', action='store_true', default=False)
    return p.parse_args()


def main():
    args = _parse_args()
    dataset_name = args.dataset
    config = DEFAULT_CONFIG.copy()
    config['dataset_name'] = dataset_name
    device = _get_device(not args.no_gpu)
    print(f"\n=== Ensemble Pipeline: {dataset_name} | device={device} ===")

    adata = _load_adata(dataset_name)
    dk = _DATASET_KEYS[dataset_name]
    samples, labels_str = _get_sample_labels(adata, dk['sample_key'], dk['condition_key'])
    n_samples = len(samples)
    config.update(build_unified_config(n_samples, dataset_name))
    config['dataset_name'] = dataset_name
    print(f"  N={n_samples}  VIB={config['use_vib']}  MC_T={config['mc_dropout_T']}  "
          f"epochs={config['n_epochs']}  pma_k={config['pma_k']}")

    import liana as li
    gene_to_idx, ct_to_idx, _, _ = _build_gene_ct_index(adata, dk['groupby'])
    resource = li.resource.select_resource(config['resource_name'])
    model_cfg = _get_model_config(dataset_name, 30)
    model_cfg['gene_emb_dim'] = 64
    model_cfg['dropout'] = 0.30

    # CellChat 通路先验（仅 reichart）
    lr_pathway_map, n_pathways = {}, 0
    if dataset_name in _REICHART_ONLY:
        lr_pathway_map, n_pathways = build_lr_pathway_map(_DATA_DIR)
        config['n_pathways'] = n_pathways
        config['use_pathway_prior'] = True
        print(f"  [CellChat] {n_pathways} 通路, {len(lr_pathway_map)} LR pairs 覆盖")

    print("  预计算 batch_data...")
    all_batch_data = _prepare_all_samples(
        samples, adata, resource, gene_to_idx, ct_to_idx,
        model_cfg, config, dk['sample_key'], dk['groupby'],
        lr_pathway_map=lr_pathway_map if lr_pathway_map else None)
    n_valid = sum(1 for bd in all_batch_data if bd is not None)
    print(f"  有效样本: {n_valid}/{len(samples)}")

    aurocs = run_ensemble_cv(adata, samples, labels_str, all_batch_data,
                             gene_to_idx, ct_to_idx, model_cfg, config, device)

    import pandas as pd
    pd.DataFrame({
        'dataset': [dataset_name],
        'method': ['ensemble_e2e+nmf'],
        'mean_auroc': [round(float(np.mean(aurocs)), 4)],
        'std_auroc': [round(float(np.std(aurocs)), 4)],
        'fold_aurocs': [str([round(a, 4) for a in aurocs])],
    }).to_csv(_OUTPUT_DIR / f'{dataset_name}_ensemble_e2e_nmf.csv', index=False)
    print(f"  结果已保存: {dataset_name}_ensemble_e2e_nmf.csv")


if __name__ == '__main__':
    main()
