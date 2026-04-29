# run_liana_scores.py
# 依赖: liana, anndata, pandas
# 被依赖: unified_comparison.py（需要这些 parquet 文件）
# 职责: 为4个小数据集（kuppe/carraro/habermann/velmeshev）批量运行7个 LIANA 方法，
#       保存 LR 评分到 data/interim/{dataset}_liana_scores/{method}.parquet

import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

_PIPELINE_DIR = Path(__file__).parent
_CODE_DIR = _PIPELINE_DIR.parent
_BASE_DIR = _CODE_DIR.parent
sys.path.insert(0, str(_CODE_DIR))

import anndata
import pandas as pd
from pipeline.e2e_pipeline import _DATASET_KEYS, map_ensembl_to_symbol

_DATA_DIR = _BASE_DIR / 'data'

DATASETS = ['kuppe', 'carraro', 'habermann', 'velmeshev']

# method_name -> (liana_attr, uns_key, score_col)
METHODS = {
    'cellphonedb':       ('cellphonedb',       'lr_means',       'lr_means'),
    'connectome':        ('connectome',         'magnitude_rank', 'scaled_weight'),
    'cellchat':          ('cellchat',           'lr_probs',       'lr_probs'),
    'scseqcomm':         ('scseqcomm',          'inter_score',    'inter_score'),
    'singlecellsignalr': ('singlecellsignalr',  'lrscore',        'lrscore'),
    'natmi':             ('natmi',              'expr_prod',      'spec_weight'),
    'logfc':             ('logfc',              'lr_logfc',       'lr_logfc'),
    'geometric_mean':    ('geometric_mean',     'lr_gmeans',      'lr_gmeans'),
}


def _load_adata(dataset):
    candidates = [
        _DATA_DIR / 'interim' / f'{dataset}_processed.h5ad',
        _DATA_DIR / 'interim' / f'{dataset}_filtered.h5ad',
        _DATA_DIR / f'{dataset}_filtered.h5ad',
    ]
    for p in candidates:
        if p.exists():
            print(f'  加载: {p}')
            adata = anndata.read_h5ad(str(p))
            return map_ensembl_to_symbol(adata, _DATA_DIR)
    raise FileNotFoundError(f'找不到 {dataset} 处理后数据')


def run_dataset(dataset, methods_to_run=None):
    print(f'\n{"="*60}')
    print(f'  数据集: {dataset}')
    print(f'{"="*60}')

    dk = _DATASET_KEYS[dataset]
    sample_key = dk['sample_key']
    groupby    = dk['groupby']

    out_dir = _DATA_DIR / 'interim' / f'{dataset}_liana_scores'
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = _load_adata(dataset)
    print(f'  样本: {adata.obs[sample_key].nunique()}, 细胞类型: {adata.obs[groupby].nunique()}')

    from liana.method import (cellphonedb, connectome, cellchat, scseqcomm,
                               singlecellsignalr, natmi, logfc, geometric_mean)
    method_objs = {
        'cellphonedb': cellphonedb, 'connectome': connectome,
        'cellchat': cellchat, 'scseqcomm': scseqcomm,
        'singlecellsignalr': singlecellsignalr, 'natmi': natmi,
        'logfc': logfc, 'geometric_mean': geometric_mean,
    }

    todo = methods_to_run or list(METHODS.keys())
    for method_name in todo:
        out_path = out_dir / f'{method_name}.parquet'
        if out_path.exists():
            print(f'  [{method_name}] 已存在，跳过')
            continue

        liana_attr, uns_key, score_col = METHODS[method_name]
        method_obj = method_objs.get(liana_attr)
        if method_obj is None:
            print(f'  [{method_name}] 未找到 liana 方法，跳过')
            continue

        print(f'  [{method_name}] 运行中...')
        try:
            result = method_obj.by_sample(
                adata, groupby=groupby, use_raw=False,
                sample_key=sample_key, verbose=False,
                n_perms=None, inplace=False)
            if result is not None and len(result) > 0:
                result.to_parquet(str(out_path), index=False)
                print(f'  [{method_name}] 完成 → {out_path.name} ({len(result)} 行)')
            else:
                print(f'  [{method_name}] 结果为空，跳过')
        except Exception as e:
            print(f'  [{method_name}] 失败: {e}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default=None, help='指定数据集，默认跑全部4个')
    p.add_argument('--method',  default=None, help='指定方法，默认跑全部8个')
    args = p.parse_args()

    datasets = [args.dataset] if args.dataset else DATASETS
    methods  = [args.method]  if args.method  else None

    for ds in datasets:
        run_dataset(ds, methods)

    print('\n全部完成。')


if __name__ == '__main__':
    main()
