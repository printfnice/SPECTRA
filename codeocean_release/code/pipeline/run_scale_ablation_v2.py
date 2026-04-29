# run_scale_ablation_v2.py
# 依赖: e2e_pipeline.py, ms_hgatlink_encoders.py (需支持 scale_mode_v2)
# 被依赖: run_extra_experiments.sh
# 职责: 修复版尺度消融——在 cross-attention 前 zero out 非目标 proj，
#        彻底隔断残差路径信息泄露，得到干净的单尺度 AUROC。
#
# v1 问题: residual_proj(all_flat) 仍包含全部三尺度信息
# v2 修复: scale_mode_v2 在 stack 之前直接 zero 掉非目标 proj

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

ALL_DATASETS = ['kuppe', 'carraro', 'habermann', 'velmeshev', 'reichart']
SCALE_MODES = ['full', 'gene_only', 'pathway_only', 'ct_only']
OUTPUT_DIR = Path('output/scale_ablation_v2')

_ORIG_INIT_MODEL = ep._init_model


def _make_patched_init_model(scale_mode):
    def patched(n_genes, n_cts, model_cfg, gene_to_idx, device, seed, n_pathways=0, prior_features=None):
        model, fusion_dim = _ORIG_INIT_MODEL(
            n_genes, n_cts, model_cfg, gene_to_idx, device, seed, n_pathways,
            prior_features=prior_features
        )
        if scale_mode != 'full' and hasattr(model, 'fusion_layer'):
            # v2: 使用 scale_mode_v2 属性（zero out 输入，非覆盖权重）
            model.fusion_layer.scale_mode_v2 = scale_mode
        return model, fusion_dim
    return patched


def _restore():
    ep._init_model = _ORIG_INIT_MODEL


def _out_path(dataset, scale_mode):
    return OUTPUT_DIR / f'{dataset}_{scale_mode}.csv'


def _run_one(dataset, scale_mode, device='cuda', force=False):
    out_path = _out_path(dataset, scale_mode)
    if out_path.exists() and not force:
        df = pd.read_csv(str(out_path))
        print(f"  [skip] {dataset}/{scale_mode}  已存在")
        return df

    print(f"\n{'='*60}")
    print(f"  {dataset} | scale_mode_v2={scale_mode}  [{datetime.now().strftime('%H:%M:%S')}]")
    print(f"{'='*60}")

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ep._init_model = _make_patched_init_model(scale_mode)

    try:
        adata = ep._load_adata(dataset)
        dk = ep._DATASET_KEYS[dataset]
        samples, labels_str = ep._get_sample_labels(adata, dk['sample_key'], dk['condition_key'])
        n_samples = len(samples)
        print(f"  样本数={n_samples}")

        config = ep.DEFAULT_CONFIG.copy()
        config['dataset_name'] = dataset
        config['unified'] = True
        config.update(build_unified_config(n_samples, dataset))
        config['verbose'] = False

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
        n_valid = sum(1 for bd in all_batch_data if bd is not None)
        print(f"  有效样本: {n_valid}/{n_samples}")

        cv_result = ep.run_e2e_cv(
            adata, samples, labels_str, all_batch_data,
            gene_to_idx, ct_to_idx, model_cfg, config, device
        )
        if isinstance(cv_result, tuple):
            aurocs = cv_result[0]
        else:
            aurocs = cv_result
        mean_auroc = np.mean(aurocs)
        std_auroc = np.std(aurocs)
        print(f"  AUROC = {mean_auroc:.4f} ± {std_auroc:.4f}")

    finally:
        _restore()

    row = {
        'dataset': dataset, 'scale_mode': scale_mode,
        'mean_auroc': round(mean_auroc, 4), 'std_auroc': round(std_auroc, 4),
        'n_folds': len(aurocs),
        'fold_aurocs': str([round(a, 4) for a in aurocs]),
        'n_samples': n_samples,
        'ablation_version': 'v2_zero_input',
    }
    df = pd.DataFrame([row])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_path), index=False)
    print(f"  保存: {out_path}")
    return df


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='all')
    p.add_argument('--scale_mode', default='all')
    p.add_argument('--force', action='store_true', help='强制重跑并覆盖已有输出')
    args, _ = p.parse_known_args()

    datasets = ALL_DATASETS if args.dataset == 'all' else [args.dataset]
    modes = SCALE_MODES if args.scale_mode == 'all' else [args.scale_mode]

    all_rows = []
    total = len(datasets) * len(modes)
    done = 0
    for dataset in datasets:
        for mode in modes:
            done += 1
            print(f"\n[{done}/{total}] {dataset} × {mode}")
            try:
                df = _run_one(dataset, mode, force=args.force)
                if df is not None:
                    all_rows.append(df)
            except Exception as e:
                _restore()
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    if all_rows:
        summary = pd.concat(all_rows, ignore_index=True)
        out = OUTPUT_DIR / 'summary_scale_ablation_v2.csv'
        summary.to_csv(str(out), index=False)
        print(f"\n汇总: {out}")
        print("\n=== v2 尺度消融结果 (mean_auroc) ===")
        try:
            pivot = summary.pivot_table(
                index='dataset', columns='scale_mode', values='mean_auroc', aggfunc='first'
            )
            print(pivot[SCALE_MODES].round(4))
        except Exception:
            print(summary[['dataset', 'scale_mode', 'mean_auroc']].to_string())


if __name__ == '__main__':
    main()
