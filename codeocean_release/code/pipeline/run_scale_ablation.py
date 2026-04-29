# run_scale_ablation.py
# 依赖: e2e_pipeline.py, ms_hgatlink_encoders.py (需支持 scale_mode)
# 被依赖: run_all_supplementary.sh
# 职责: 三尺度消融实验（gene_only / pathway_only / ct_only / full），验证多尺度架构必要性
#
# 实现: monkeypatch e2e_pipeline._init_model，模型创建后设置 fusion_layer.scale_mode
# 注意: ms_hgatlink_encoders.py 已添加 scale_mode 支持（MultiScaleFusionLayer.forward）

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

# ============================================================================
# 配置
# ============================================================================

ALL_DATASETS = ['kuppe', 'carraro', 'habermann', 'velmeshev', 'reichart']
SCALE_MODES = ['full', 'gene_only', 'pathway_only', 'ct_only']
OUTPUT_DIR = Path('output/scale_ablation')

# 保存原始 _init_model
_ORIG_INIT_MODEL = ep._init_model


# ============================================================================
# Monkeypatch 工具
# ============================================================================

def _make_patched_init_model(scale_mode):
    """返回一个在 _init_model 基础上设置 scale_mode 的版本。"""
    def patched(n_genes, n_cts, model_cfg, gene_to_idx, device, seed, n_pathways=0, prior_features=None):
        model, fusion_dim = _ORIG_INIT_MODEL(
            n_genes, n_cts, model_cfg, gene_to_idx, device, seed, n_pathways,
            prior_features=prior_features
        )
        if scale_mode != 'full' and hasattr(model, 'fusion_layer'):
            model.fusion_layer.scale_mode = scale_mode
        return model, fusion_dim
    return patched


def _restore_init_model():
    ep._init_model = _ORIG_INIT_MODEL


# ============================================================================
# 核心逻辑
# ============================================================================

def _out_path(dataset, scale_mode):
    return OUTPUT_DIR / f'{dataset}_{scale_mode}.csv'


def _run_one(dataset, scale_mode, device='cuda', force=False):
    out_path = _out_path(dataset, scale_mode)
    if out_path.exists() and not force:
        df = pd.read_csv(str(out_path))
        print(f"  [skip] {dataset}/{scale_mode}  已存在: {out_path}")
        return df

    print(f"\n{'='*60}")
    print(f"  {dataset} | scale_mode={scale_mode}  [{datetime.now().strftime('%H:%M:%S')}]")
    print(f"{'='*60}")

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 注入 scale_mode
    ep._init_model = _make_patched_init_model(scale_mode)

    try:
        adata = ep._load_adata(dataset)
        dk = ep._DATASET_KEYS[dataset]
        samples, labels_str = ep._get_sample_labels(adata, dk['sample_key'], dk['condition_key'])
        n_samples = len(samples)
        print(f"  样本数={n_samples}  scale_mode={scale_mode}")

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

        print(f"  预计算 batch_data...")
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
            gate_data = cv_result[1] if len(cv_result) > 1 else None
        else:
            aurocs, gate_data = cv_result, None
        mean_auroc = np.mean(aurocs)
        std_auroc = np.std(aurocs)
        print(f"  AUROC = {mean_auroc:.4f} ± {std_auroc:.4f}")

    finally:
        _restore_init_model()

    row = {
        'dataset': dataset, 'scale_mode': scale_mode,
        'mean_auroc': round(mean_auroc, 4), 'std_auroc': round(std_auroc, 4),
        'n_folds': len(aurocs),
        'fold_aurocs': str([round(a, 4) for a in aurocs]),
        'n_samples': n_samples,
        'n_epochs': config['n_epochs'], 'pma_k': config['pma_k'],
    }
    # 若 gate_data 可用（full 模式），记录平均 gate 权重
    if gate_data is not None and scale_mode == 'full':
        try:
            gdf = pd.concat(gate_data.values()) if isinstance(gate_data, dict) else gate_data
            row['w_gene_mean'] = round(float(gdf['w_gene'].mean()), 4)
            row['w_pathway_mean'] = round(float(gdf['w_pathway'].mean()), 4)
            row['w_celltype_mean'] = round(float(gdf['w_celltype'].mean()), 4)
        except Exception:
            pass

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
    p.add_argument('--no-gpu', action='store_true')
    p.add_argument('--force', action='store_true', help='强制重跑并覆盖已有输出')
    args, _ = p.parse_known_args()

    import torch
    device = 'cpu' if args.no_gpu else ('cuda' if torch.cuda.is_available() else 'cpu')
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
                df = _run_one(dataset, mode, device, force=args.force)
                if df is not None:
                    all_rows.append(df)
            except Exception as e:
                _restore_init_model()
                print(f"  ERROR {dataset}/{mode}: {e}")
                import traceback
                traceback.print_exc()

    if all_rows:
        summary = pd.concat(all_rows, ignore_index=True)
        summary_path = OUTPUT_DIR / 'summary_scale_ablation.csv'
        summary.to_csv(str(summary_path), index=False)
        print(f"\n汇总: {summary_path}")
        # 打印消融表
        print("\n=== 尺度消融结果 (mean_auroc) ===")
        try:
            pivot = summary.pivot_table(
                index='dataset', columns='scale_mode', values='mean_auroc', aggfunc='first'
            )
            print(pivot[SCALE_MODES].round(4))
        except Exception:
            print(summary[['dataset', 'scale_mode', 'mean_auroc']].to_string())


if __name__ == '__main__':
    main()
