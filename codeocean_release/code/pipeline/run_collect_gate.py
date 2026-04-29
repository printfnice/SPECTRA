# run_collect_gate.py
# 依赖: e2e_pipeline.py, e2e_utils.py, liana
# 被依赖: 直接运行
# 职责: 5个数据集统一配置跑完整 E2E + 保存含 LR 名称的 gate 权重 CSV
#       中间结果逐数据集保存，可随时中断恢复
#
# 输出：output/{dataset}_gate_weights.csv  （含 lr_name, w_gene, w_pathway, w_celltype）
# 完成后自动调用 plot_gate_weights.py 生成 Fig.4

import os, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')
_PIPELINE_DIR = Path(__file__).parent
_CODE_DIR = _PIPELINE_DIR.parent
sys.path.insert(0, str(_CODE_DIR))
os.chdir(_PIPELINE_DIR)

import liana as li
import e2e_pipeline as ep
from e2e_utils import build_unified_config, save_gate_weights

OUTPUT_DIR = _PIPELINE_DIR.parent.parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

DATASETS = ['kuppe', 'carraro', 'habermann', 'velmeshev', 'reichart']


def run_one(dataset_name, device):
    gate_path = OUTPUT_DIR / f'{dataset_name}_gate_weights.csv'
    if gate_path.exists():
        df = pd.read_csv(gate_path)
        has_names = 'lr_name' in df.columns
        print(f'[skip] {dataset_name}  (rows={len(df)}, has_lr_name={has_names})')
        if has_names:
            return  # 已是新格式，跳过
        print(f'  旧格式（无 lr_name），重新收集...')
        gate_path.unlink()

    print(f'\n{"="*55}')
    print(f'  {dataset_name} | collect_gate')
    print(f'{"="*55}')

    adata = ep._load_adata(dataset_name)
    dk = ep._DATASET_KEYS[dataset_name]
    samples, labels_str = ep._get_sample_labels(adata, dk['sample_key'], dk['condition_key'])
    n_samples = len(samples)

    config = ep.DEFAULT_CONFIG.copy()
    config['dataset_name'] = dataset_name
    config['unified'] = True
    config['collect_gate_weights'] = True
    config['verbose'] = False
    config.update(build_unified_config(n_samples, dataset_name))

    print(f'  N={n_samples}  n_epochs={config["n_epochs"]}  '
          f'pma_k={config["pma_k"]}  lr_pairs={config["e2e_max_lr_pairs"]}')

    gene_to_idx, ct_to_idx, _, _ = ep._build_gene_ct_index(adata, dk['groupby'])
    resource = li.resource.select_resource(config['resource_name'])
    model_cfg = ep._get_model_config(dataset_name, 30)
    model_cfg['gene_emb_dim'] = 64
    model_cfg['dropout'] = 0.30

    lr_pathway_map = None
    if config.get('use_pathway_prior'):
        from model.pathway_utils import build_lr_pathway_map
        lr_pathway_map, _ = build_lr_pathway_map(str(_CODE_DIR.parent / 'data'))

    all_batch_data = ep._prepare_all_samples(
        samples, adata, resource, gene_to_idx, ct_to_idx,
        model_cfg, config, dk['sample_key'], dk['groupby'],
        lr_pathway_map=lr_pathway_map,
    )
    n_valid = sum(1 for bd in all_batch_data if bd is not None)
    print(f'  有效样本: {n_valid}/{n_samples}')

    # 验证��一个样本是否含 lr_names
    first_valid = next(bd for bd in all_batch_data if bd is not None)
    if 'lr_names' not in first_valid:
        print('  [ERROR] lr_names 字段缺失，请确认 e2e_pipeline.py 已更新')
        return

    cv_result = ep.run_e2e_cv(
        adata, samples, labels_str, all_batch_data,
        gene_to_idx, ct_to_idx, model_cfg, config, device
    )
    aurocs, gate_data = cv_result if isinstance(cv_result, tuple) else (cv_result, None)
    mean_auroc = float(np.mean(aurocs))
    print(f'  AUROC = {mean_auroc:.4f} ± {float(np.std(aurocs)):.4f}')

    if gate_data is not None:
        save_gate_weights(gate_data, dataset_name, OUTPUT_DIR)
        n_rows = len(gate_data)
        has_names = 'lr_name' in gate_data.columns
        print(f'  Gate CSV: {n_rows} 行  has_lr_name={has_names}')
    else:
        print('  [WARN] gate_data 为空，未保存')


def main():
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\n{"="*55}')
    print(f'  collect_gate — 5个数据集')
    print(f'  设备: {device}')
    print(f'{"="*55}')

    for ds in DATASETS:
        run_one(ds, device)

    print('\n  所有数据集 gate 收集完成，开始生成图表...')
    import subprocess
    subprocess.run([sys.executable, str(_PIPELINE_DIR / 'plot_gate_weights.py')], check=True)


if __name__ == '__main__':
    main()
