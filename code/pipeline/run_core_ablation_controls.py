#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_core_ablation_controls.py

核心对照消融（全数据集）：
1) full:        SPECTRA unified 默认配置
2) wo_gate:     固定均匀门控 (1/3,1/3,1/3)
3) wo_vib:      关闭 VIB

输出:
- output/core_ablation/{dataset}_{mode}.csv
- output/core_ablation/summary_core_ablation.csv
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))
os.chdir(Path(__file__).parent)

import liana as li
import e2e_pipeline as ep
from e2e_utils import build_unified_config

ALL_DATASETS = ['kuppe', 'carraro', 'habermann', 'velmeshev', 'reichart']
MODES = ['full', 'wo_gate', 'wo_vib']
OUTPUT_DIR = Path('output/core_ablation')

_ORIG_INIT_MODEL = ep._init_model


class _UniformGate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.full((x.size(0), 3), 1.0 / 3.0, dtype=x.dtype, device=x.device)


def _make_patched_init_model(mode):
    def patched(n_genes, n_cts, model_cfg, gene_to_idx, device, seed, n_pathways=0, prior_features=None):
        model, fusion_dim = _ORIG_INIT_MODEL(
            n_genes, n_cts, model_cfg, gene_to_idx, device, seed, n_pathways,
            prior_features=prior_features,
        )
        if mode == 'wo_gate' and hasattr(model, 'fusion_layer'):
            model.fusion_layer.gate_net = _UniformGate().to(device)
        return model, fusion_dim

    return patched


def _restore():
    ep._init_model = _ORIG_INIT_MODEL


def _run_one(dataset, mode, force=False):
    out_path = OUTPUT_DIR / f'{dataset}_{mode}.csv'
    if out_path.exists() and not force:
        return pd.read_csv(out_path)

    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] dataset={dataset} mode={mode}")
    print(f"{'='*70}")

    ep._init_model = _make_patched_init_model(mode)

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        adata = ep._load_adata(dataset)
        dk = ep._DATASET_KEYS[dataset]
        samples, labels_str = ep._get_sample_labels(adata, dk['sample_key'], dk['condition_key'])
        n_samples = len(samples)

        config = ep.DEFAULT_CONFIG.copy()
        config['dataset_name'] = dataset
        config['unified'] = True
        config.update(build_unified_config(n_samples, dataset))
        config['verbose'] = False

        if mode == 'wo_vib':
            config['use_vib'] = False
            config['mc_dropout'] = 0
            config['mc_dropout_T'] = 1

        gene_to_idx, ct_to_idx, _, _ = ep._build_gene_ct_index(adata, dk['groupby'])
        resource = li.resource.select_resource(config['resource_name'])
        model_cfg = ep._get_model_config(dataset, 30)
        model_cfg['gene_emb_dim'] = 64
        model_cfg['dropout'] = 0.30

        all_batch_data = ep._prepare_all_samples(
            samples, adata, resource, gene_to_idx, ct_to_idx,
            model_cfg, config, dk['sample_key'], dk['groupby'],
            lr_pathway_map=None,
        )

        cv_result = ep.run_e2e_cv(
            adata, samples, labels_str, all_batch_data,
            gene_to_idx, ct_to_idx, model_cfg, config, device
        )
        aurocs = cv_result[0] if isinstance(cv_result, tuple) else cv_result

        row = {
            'dataset': dataset,
            'mode': mode,
            'mean_auroc': round(float(np.mean(aurocs)), 4),
            'std_auroc': round(float(np.std(aurocs)), 4),
            'n_folds': len(aurocs),
            'fold_aurocs': str([round(float(a), 4) for a in aurocs]),
            'n_samples': n_samples,
            'use_vib': bool(config.get('use_vib', False)),
        }

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([row])
        df.to_csv(out_path, index=False)
        print(f"saved: {out_path}")
        return df

    finally:
        _restore()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all')
    parser.add_argument('--mode', default='all')
    parser.add_argument('--force', action='store_true')
    args, _ = parser.parse_known_args()

    datasets = ALL_DATASETS if args.dataset == 'all' else [args.dataset]
    modes = MODES if args.mode == 'all' else [args.mode]

    all_rows = []
    for ds in datasets:
        for md in modes:
            try:
                df = _run_one(ds, md, force=args.force)
                all_rows.append(df)
            except Exception as e:
                _restore()
                print(f"ERROR {ds}/{md}: {e}")
                import traceback

                traceback.print_exc()

    if all_rows:
        summary = pd.concat(all_rows, ignore_index=True)
        out = OUTPUT_DIR / 'summary_core_ablation.csv'
        summary.to_csv(out, index=False)
        print(f"\nsummary: {out}")
        try:
            pivot = summary.pivot_table(index='dataset', columns='mode', values='mean_auroc', aggfunc='first')
            print(pivot[MODES].round(4))
        except Exception:
            print(summary[['dataset', 'mode', 'mean_auroc']].to_string(index=False))


if __name__ == '__main__':
    main()
