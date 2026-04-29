# pathway_utils.py
# 依赖: pandas, pathlib, urllib.request
# 被依赖: ms_hgatlink_method.py, e2e_pipeline.py
# 职责: CellChatDB 通路映射构建与本地缓存

import pandas as pd
from pathlib import Path
import urllib.request

_CELLCHAT_URL = (
    'https://raw.githubusercontent.com/LewisLabUCSD/'
    'Ligand-Receptor-Pairs/master/Human/Human-2020-Jin-LR-pairs.csv'
)
_CACHE_FILENAME = 'cellchat_pathway_map.csv'


def build_lr_pathway_map(data_dir):
    """下载/加载 CellChatDB，返回 ({(lig,rec): pathway_idx}, n_pathways)。

    pathway_idx 从 1 开始，0 保留给 unknown/padding。
    结果缓存到 data_dir/cellchat_pathway_map.csv。
    """
    data_dir = Path(data_dir)
    cache_path = data_dir / _CACHE_FILENAME

    if cache_path.exists():
        df = pd.read_csv(cache_path)
        print(f"  [CellChat] 从缓存加载: {cache_path} ({len(df)} 行)")
    else:
        df = _download_and_parse(data_dir, cache_path)
        if df is None:
            return {}, 0

    # 构建通路索引（1-based，0=unknown）
    pathways = sorted(df['pathway'].dropna().unique())
    path_to_idx = {p: i + 1 for i, p in enumerate(pathways)}
    n_pathways = len(pathways)

    lr_map = {}
    for _, row in df.iterrows():
        if pd.isna(row['pathway']):
            continue
        key = (str(row['ligand']).upper(), str(row['receptor']).upper())
        lr_map[key] = path_to_idx[row['pathway']]

    print(f"  [CellChat] {n_pathways} 通路, {len(lr_map)} LR pairs 有通路注释")
    return lr_map, n_pathways


def _download_and_parse(data_dir, cache_path):
    """下载 CellChatDB CSV，解析并缓存，返回标准化 DataFrame 或 None。"""
    print(f"  [CellChat] 下载 CellChatDB LR pairs...")
    tmp_path = data_dir / 'cellchat_raw.csv'
    try:
        urllib.request.urlretrieve(_CELLCHAT_URL, str(tmp_path))
        raw = pd.read_csv(tmp_path)
    except Exception as e:
        print(f"  [CellChat] 下载失败: {e}，返回空映射")
        tmp_path.unlink(missing_ok=True)
        return None

    col_map = _detect_columns(raw)
    if col_map is None:
        print(f"  [CellChat] 列名识别失败，列: {list(raw.columns)}")
        tmp_path.unlink(missing_ok=True)
        return None

    df = raw[[col_map['ligand'], col_map['receptor'], col_map['pathway']]].copy()
    df.columns = ['ligand', 'receptor', 'pathway']
    df = df.dropna()
    df['ligand'] = df['ligand'].str.upper()
    df['receptor'] = df['receptor'].str.upper()
    df.to_csv(cache_path, index=False)
    tmp_path.unlink(missing_ok=True)
    print(f"  [CellChat] 缓存已保存: {cache_path}")
    return df


def _detect_columns(df):
    """自动识别 ligand/receptor/pathway 列名（不同版本列名不同）。"""
    cols_lower = [c.lower() for c in df.columns]

    def find_col(candidates):
        for c in candidates:
            if c.lower() in cols_lower:
                return df.columns[cols_lower.index(c.lower())]
        return None

    lig = find_col(['ligand_symbol', 'ligand', 'gene1'])
    rec = find_col(['receptor_symbol', 'receptor', 'gene2'])
    path = find_col(['pathway_name', 'pathway', 'category'])

    if lig and rec and path:
        return {'ligand': lig, 'receptor': rec, 'pathway': path}
    return None
