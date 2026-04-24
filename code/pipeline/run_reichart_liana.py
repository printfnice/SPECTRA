# run_reichart_liana.py
# 依赖: liana, anndata, pandas
# 被依赖: unified_comparison.py（通过 parquet 文件）
# 职责: 一次加载 reichart 20GB h5ad，依次跑 8 个 LIANA 方法的 by_sample，
#       每个方法完成后立即保存为 parquet（避免最终 h5ad 写入 OOM）

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

_PIPELINE_DIR = Path(__file__).parent
_CODE_DIR = _PIPELINE_DIR.parent
_BASE_DIR = _CODE_DIR.parent
sys.path.insert(0, str(_CODE_DIR))

_DATA_DIR = _BASE_DIR / 'data'
_SCORE_DIR = _DATA_DIR / 'interim' / 'reichart_liana_scores'

SAMPLE_KEY = 'Sample'
GROUPBY = 'cell_type'
CONDITION_KEY = 'disease'


def _map_ensembl_to_symbol(adata):
    """将 ENSEMBL ID 转为基因符号（reichart 必需）。"""
    import pandas as pd
    mapping_file = _DATA_DIR / 'ensembl_to_symbol.csv'
    if not mapping_file.exists():
        print("  警告: ensembl_to_symbol.csv 不存在，跳过基因名转换")
        return adata
    mapping = pd.read_csv(mapping_file, index_col=0).squeeze()
    mapping_dict = mapping.to_dict()
    new_names = [str(mapping_dict.get(g, g)) for g in adata.var_names]
    seen = {}
    unique_names = []
    for n in new_names:
        if n in seen:
            seen[n] += 1
            unique_names.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            unique_names.append(n)
    n_mapped = sum(1 for old, new in zip(adata.var_names, unique_names)
                   if str(old) != str(new))
    adata.var_names = unique_names
    print(f"  ENSEMBL->Symbol: {n_mapped}/{len(adata.var_names)} genes mapped")
    return adata


def _load_reichart():
    """加载 reichart 数据（只加载一次），含 ENSEMBL->Symbol 转换。"""
    import anndata
    path = _DATA_DIR / 'interim' / 'reichart_filtered.h5ad'
    print(f"Loading reichart: {path}")
    t0 = time.time()
    adata = anndata.read_h5ad(str(path))
    print(f"  Loaded: {adata.shape}, {time.time()-t0:.0f}s")
    if str(adata.var_names[0]).startswith('ENSG'):
        adata = _map_ensembl_to_symbol(adata)
    return adata


def _save_parquet(result, method_name):
    """将单个方法的评分 DataFrame 保存为 parquet。"""
    _SCORE_DIR.mkdir(parents=True, exist_ok=True)
    out = _SCORE_DIR / f'{method_name}.parquet'
    result.to_parquet(str(out), index=False)
    print(f"  [{method_name}] saved: {out} ({len(result)} rows)")


def _run_one_method(adata, method, method_name):
    """运行单个 LIANA 方法并保存 parquet，支持断点续跑。"""
    out_file = _SCORE_DIR / f'{method_name}.parquet'
    if out_file.exists():
        print(f"  [{method_name}] parquet exists, skip")
        return method_name

    print(f"  [{method_name}] running...")
    t0 = time.time()
    result = method.by_sample(
        adata, groupby=GROUPBY, use_raw=False,
        sample_key=SAMPLE_KEY, verbose=False,
        n_perms=None, inplace=False,
    )
    elapsed = time.time() - t0
    n_rows = len(result) if result is not None else 0
    print(f"  [{method_name}] done: {n_rows} rows, {elapsed:.0f}s")
    _save_parquet(result, method_name)
    return method_name


def main():
    from liana.method import (
        cellphonedb, connectome, cellchat, scseqcomm,
        singlecellsignalr, natmi, logfc, geometric_mean,
    )
    methods = [
        (cellphonedb, 'cellphonedb'),
        (connectome, 'connectome'),
        (cellchat, 'cellchat'),
        (scseqcomm, 'scseqcomm'),
        (singlecellsignalr, 'singlecellsignalr'),
        (natmi, 'natmi'),
        (logfc, 'logfc'),
        (geometric_mean, 'geometric_mean'),
    ]

    adata = _load_reichart()
    print(f"\nSamples: {adata.obs[SAMPLE_KEY].nunique()}")
    print(f"Cell types: {adata.obs[GROUPBY].nunique()}")
    print(f"Conditions: {adata.obs[CONDITION_KEY].unique().tolist()}")

    completed = []
    for method, name in methods:
        try:
            _run_one_method(adata, method, name)
            completed.append(name)
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")

    print(f"\n=== Done ===")
    print(f"Completed: {completed}")
    existing = list(_SCORE_DIR.glob('*.parquet'))
    print(f"Saved parquets: {[f.stem for f in existing]}")


if __name__ == '__main__':
    main()
