# run_reichart_mixup_off.py
# 职责: reichart Mixup=OFF 消融——验证 N>50 时 Mixup 开启规则的必要性
# 对比: reichart 正常运行 use_mixup=True (R38 AUROC=0.9264)
#        本实验: use_mixup=False，预期 AUROC 下降

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

OUTPUT_DIR = Path('output/mixup_ablation')


def main():
    import torch
    out_path = OUTPUT_DIR / 'reichart_mixup_off.csv'
    if out_path.exists():
        print(f"[skip] 已存在: {out_path}")
        df = pd.read_csv(str(out_path))
        print(f"  AUROC = {df['mean_auroc'].iloc[0]:.4f}")
        return

    print(f"\n{'='*60}")
    print(f"  reichart | Mixup=OFF  [{datetime.now().strftime('%H:%M:%S')}]")
    print(f"{'='*60}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = 'reichart'

    adata = ep._load_adata(dataset)
    dk = ep._DATASET_KEYS[dataset]
    samples, labels_str = ep._get_sample_labels(adata, dk['sample_key'], dk['condition_key'])
    n_samples = len(samples)
    print(f"  样本数={n_samples}")

    config = ep.DEFAULT_CONFIG.copy()
    config['dataset_name'] = dataset
    config['unified'] = True
    config.update(build_unified_config(n_samples, dataset))
    # 消融：强制关闭 Mixup（默认 reichart 因 N>50 会开启）
    config['use_mixup'] = False
    config['verbose'] = False
    print(f"  use_mixup=False (默认规则应为 True，N={n_samples}>50)")

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
    aurocs, _ = cv_result if isinstance(cv_result, tuple) else (cv_result, None)
    mean_auroc = np.mean(aurocs)
    std_auroc = np.std(aurocs)
    print(f"\n  reichart Mixup=OFF  AUROC = {mean_auroc:.4f} ± {std_auroc:.4f}")
    print(f"  对比 Mixup=ON (R38): 0.9264 ± 0.0494")
    print(f"  delta = {mean_auroc - 0.9264:+.4f}")

    row = {
        'dataset': dataset, 'use_mixup': False,
        'mean_auroc': round(mean_auroc, 4), 'std_auroc': round(std_auroc, 4),
        'n_folds': len(aurocs),
        'fold_aurocs': str([round(a, 4) for a in aurocs]),
        'n_samples': n_samples,
        'n_epochs': config['n_epochs'], 'pma_k': config['pma_k'],
        'mixup_alpha': 0.0,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(str(out_path), index=False)
    print(f"  保存: {out_path}")


if __name__ == '__main__':
    main()
