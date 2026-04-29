# run_irm_reichart.py
# 依赖: e2e_pipeline.py, e2e_irm.py, e2e_utils.py, model/
# 被依赖: 直接运行
# 职责: Direction 2 — IRM（不变风险最小化）在 reichart 上的运行入口
#       遗传亚型作为环境，训练对所有亚型不变的疾病分类特征

import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings('ignore')

_PIPELINE_DIR = Path(__file__).parent
_CODE_DIR = _PIPELINE_DIR.parent
_BASE_DIR = _CODE_DIR.parent
sys.path.insert(0, str(_CODE_DIR))

from pipeline.e2e_pipeline import (
    DEFAULT_CONFIG, _DATASET_KEYS,
    _get_device, _load_adata, _get_sample_labels, _encode_labels,
    _build_cv_splitter, _move_to_device,
    _init_model, _compute_auroc,
    _prepare_all_samples,
)
from pipeline.e2e_utils import build_unified_config, map_ensembl_to_symbol
from pipeline.e2e_irm import run_irm_fold
from model.pathway_utils import build_lr_pathway_map
from model.ms_hgatlink_method import _build_gene_ct_index, _get_model_config


_DATA_DIR = _BASE_DIR / 'data'
_OUTPUT_DIR = _BASE_DIR / 'output'


def get_subtype_labels_for_irm(adata, samples, sample_key):
    """从 Primary.Genetic.Diagnosis 提取亚型标签（与 e2e_multitask 相同）。"""
    col = 'Primary.Genetic.Diagnosis'
    if col not in adata.obs.columns:
        print(f"  警告: {col} 列不存在，无法使用 IRM")
        return None
    pgd_map = adata.obs.groupby(sample_key)[col].first()
    labels = [str(pgd_map.get(s, 'Unknown')) for s in samples]
    unique = sorted(set(labels))
    print(f"  [IRM] 亚型标签: {unique}  n_subtypes={len(unique)}")
    return labels


def run_irm_cv(adata, samples, labels_str, all_batch_data,
               gene_to_idx, ct_to_idx, model_cfg, config, device,
               irm_lambda=1.0):
    """K 折 IRM CV：遗传亚型分组 + 不变特征学习。"""
    labels_int, n_classes, label_names = _encode_labels(labels_str)
    print(f"  类别: {label_names}  n_classes={n_classes}")

    sample_key = _DATASET_KEYS['reichart']['sample_key']
    sub_labels = get_subtype_labels_for_irm(adata, samples, sample_key)
    if sub_labels is None:
        raise RuntimeError("reichart 缺少亚型标签，无法运行 IRM")

    splitter, groups, strat_y = _build_cv_splitter(adata, samples, labels_int, config)
    split_args = (samples, strat_y) if groups is None else (samples, strat_y, groups)
    n_genes, n_cts = len(gene_to_idx), len(ct_to_idx)
    all_aurocs = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(*split_args)):
        train_bd = _move_to_device([all_batch_data[i] for i in train_idx], device)
        test_bd  = _move_to_device([all_batch_data[i] for i in test_idx], device)
        train_y  = [labels_int[i] for i in train_idx]
        test_y   = [labels_int[i] for i in test_idx]
        train_sub_y = [sub_labels[i] for i in train_idx]

        fold_aurocs = []
        for rs in range(config['n_random_states']):
            seed = 42 + rs * 7
            auroc = run_irm_fold(
                train_bd, train_y, train_sub_y, test_bd, test_y,
                n_genes, n_cts, gene_to_idx, model_cfg,
                config, n_classes, device, seed, irm_lambda=irm_lambda)
            fold_aurocs.append(auroc)
            print(f"  fold={fold_idx+1} rs={rs}  AUROC={auroc:.4f}")

        mean_f = np.mean(fold_aurocs)
        all_aurocs.append(mean_f)
        print(f"  [fold {fold_idx+1}] 平均 AUROC = {mean_f:.4f}")

    print(f"\n{'='*55}")
    print(f"  IRM (lambda={irm_lambda})  "
          f"Mean={np.mean(all_aurocs):.4f} +/- {np.std(all_aurocs):.4f}")
    print(f"  各折: {[f'{a:.3f}' for a in all_aurocs]}")
    print(f"{'='*55}")
    return all_aurocs


def main():
    irm_lambda = 1.0          # IRM 惩罚权重（可调：0.1/1.0/10.0）
    dataset_name = 'reichart'
    config = DEFAULT_CONFIG.copy()
    config['dataset_name'] = dataset_name
    device = _get_device(True)
    print(f"\n=== IRM Pipeline: {dataset_name} | device={device} | lambda={irm_lambda} ===")

    adata = _load_adata(dataset_name)
    dk = _DATASET_KEYS[dataset_name]
    samples, labels_str = _get_sample_labels(adata, dk['sample_key'], dk['condition_key'])
    n_samples = len(samples)
    config.update(build_unified_config(n_samples, dataset_name))
    config['dataset_name'] = dataset_name
    config['use_pathway_prior'] = True
    print(f"  N={n_samples}  epochs={config['n_epochs']}  pma_k={config['pma_k']}")

    import liana as li
    gene_to_idx, ct_to_idx, _, _ = _build_gene_ct_index(adata, dk['groupby'])
    resource = li.resource.select_resource(config['resource_name'])
    model_cfg = _get_model_config(dataset_name, 30)
    model_cfg['gene_emb_dim'] = 64
    model_cfg['dropout'] = 0.30

    lr_pathway_map, n_pathways = build_lr_pathway_map(_DATA_DIR)
    config['n_pathways'] = n_pathways

    print("  预计算 batch_data...")
    all_batch_data = _prepare_all_samples(
        samples, adata, resource, gene_to_idx, ct_to_idx,
        model_cfg, config, dk['sample_key'], dk['groupby'],
        lr_pathway_map=lr_pathway_map)
    n_valid = sum(1 for bd in all_batch_data if bd is not None)
    print(f"  有效样本: {n_valid}/{len(samples)}")

    aurocs = run_irm_cv(adata, samples, labels_str, all_batch_data,
                        gene_to_idx, ct_to_idx, model_cfg, config, device,
                        irm_lambda=irm_lambda)

    import pandas as pd
    pd.DataFrame({
        'dataset': [dataset_name], 'method': [f'irm_lambda{irm_lambda}'],
        'mean_auroc': [round(float(np.mean(aurocs)), 4)],
        'std_auroc': [round(float(np.std(aurocs)), 4)],
        'fold_aurocs': [str([round(a, 4) for a in aurocs])],
    }).to_csv(_OUTPUT_DIR / f'reichart_irm_lambda{irm_lambda}.csv', index=False)
    print("  结果已保存")


if __name__ == '__main__':
    main()
