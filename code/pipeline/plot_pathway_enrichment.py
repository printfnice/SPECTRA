# plot_pathway_enrichment.py
# 依赖: pandas, matplotlib, numpy, scipy
# 被依赖: 直接运行（可反复执行，不依赖模型）
# 职责: 通路富集分析（Fisher's exact test）+ dot plot（Fig.5b）
#
# 数据来源：
#   data/cellchat_pathway_map.csv  （CellChatDB 228 通路注释）
#   output/top_lr_pairs_{dataset}.csv  （top LR pairs，由 plot_top_lr_pairs.py 生成）
#
# 输出：
#   output/Fig5b_pathway_enrichment_{dataset}.pdf/.png
#   output/pathway_enrichment_{dataset}.csv

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

_PIPELINE_DIR = Path(__file__).parent
_CODE_DIR = _PIPELINE_DIR.parent
_CLASS_DIR = _CODE_DIR.parent
sys.path.insert(0, str(_CODE_DIR))

OUTPUT_DIR = _CLASS_DIR / 'output'
DATA_DIR   = _CLASS_DIR / 'data'

DATASETS = ['kuppe', 'carraro', 'habermann', 'velmeshev', 'reichart']
DATASET_LABELS = {
    'kuppe':     'Kuppe (MI)',
    'carraro':   'Carraro (CF)',
    'habermann': 'Habermann (PF)',
    'velmeshev': 'Velmeshev (ASD)',
    'reichart':  'Reichart (DCM)',
}
TOP_N_ENRICHED = 15  # 展示前 N 个富集通路


def load_pathway_map():
    """加载 CellChatDB 通路注释。返回 {(ligand, receptor): pathway_name} dict。"""
    from model.pathway_utils import build_lr_pathway_map
    lr_map, n_pw = build_lr_pathway_map(str(DATA_DIR))
    # 反转：index → pathway_name
    csv_path = DATA_DIR / 'cellchat_pathway_map.csv'
    pw_df = pd.read_csv(csv_path)
    # pathway_idx → pathway_name 映射
    idx2name = {}
    for _, row in pw_df.iterrows():
        idx2name[row.get('pathway_idx', row.get('index', None))] = row.get('pathway_name', row.get('Var2', ''))

    # (lig, rec) → pathway_name
    lr2pathway = {}
    for (lig, rec), idx in lr_map.items():
        name = idx2name.get(idx, f'pathway_{idx}')
        lr2pathway[(lig, rec)] = name
    return lr2pathway, n_pw


def load_pathway_map_simple():
    """直接从 CSV 读取通路映射，不依赖 pathway_utils。"""
    csv_path = DATA_DIR / 'cellchat_pathway_map.csv'
    if not csv_path.exists():
        print(f'  [WARN] 未找到 {csv_path}')
        return {}, 0
    df = pd.read_csv(csv_path)
    print(f'  CellChatDB CSV 列名: {list(df.columns)}')
    print(f'  行数: {len(df)}')
    # 尝试识别列名
    lig_col = next((c for c in df.columns if 'lig' in c.lower()), None)
    rec_col = next((c for c in df.columns if 'rec' in c.lower()), None)
    pw_col  = next((c for c in df.columns if 'path' in c.lower()), None)
    if lig_col and rec_col and pw_col:
        lr2pathway = {(row[lig_col], row[rec_col]): row[pw_col]
                      for _, row in df.iterrows()}
        n_pw = df[pw_col].nunique()
        print(f'  已加载 {len(lr2pathway)} LR pairs, {n_pw} 通路')
        return lr2pathway, n_pw
    # 若列名无���识别，尝试前3列
    if len(df.columns) >= 3:
        df.columns = ['ligand', 'receptor', 'pathway_name'] + list(df.columns[3:])
        lr2pathway = {(row['ligand'], row['receptor']): row['pathway_name']
                      for _, row in df.iterrows()}
        n_pw = df['pathway_name'].nunique()
        print(f'  已加载（按位置）{len(lr2pathway)} LR pairs, {n_pw} 通路')
        return lr2pathway, n_pw
    return {}, 0


def fisher_enrichment(top_lr_list, all_lr_set, lr2pathway):
    """
    对 top LR pairs 做通路富集（Fisher's exact test）。
    top_lr_list: list of (ligand, receptor) tuples
    all_lr_set:  set of all (ligand, receptor) from resource
    """
    all_pathways = {}
    for (lig, rec), pw in lr2pathway.items():
        if pw not in all_pathways:
            all_pathways[pw] = set()
        all_pathways[pw].add((lig, rec))

    N = len(all_lr_set)
    K = len(top_lr_list)
    top_set = set(top_lr_list)

    results = []
    for pw, pw_pairs in all_pathways.items():
        pw_in_all  = len(pw_pairs & all_lr_set)
        pw_in_top  = len(pw_pairs & top_set)
        if pw_in_all == 0:
            continue
        # 2×2 contingency table
        a = pw_in_top
        b = K - pw_in_top
        c = pw_in_all - pw_in_top
        d = N - K - c
        if d < 0:
            d = 0
        _, p = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
        results.append({
            'pathway': pw,
            'n_pw_in_top': pw_in_top,
            'n_pw_in_all': pw_in_all,
            'pvalue': p,
            'neg_log10_p': -np.log10(p + 1e-300),
        })
    df = pd.DataFrame(results).sort_values('pvalue')
    return df


def load_all_lr_pairs():
    """加载 consensus resource 的全部 LR pairs。"""
    try:
        import liana as li
        resource = li.resource.select_resource('consensus')
        pairs = set(zip(resource['ligand'], resource['receptor']))
        return pairs
    except Exception as e:
        print(f'  [WARN] 无法加载 liana resource: {e}')
        return set()


def plot_dot(enrich_df, dataset, top_n=TOP_N_ENRICHED, out_prefix=None):
    """Dot plot：x=-log10(p)，y=通路名，点大小=overlapping LR 数量。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(_PIPELINE_DIR))
    from pub_style import set_pub_style, FIGURES_DIR as FIG_DIR

    set_pub_style()

    if out_prefix is None:
        out_prefix = f'Fig5b_pathway_enrichment_{dataset}'

    df = enrich_df[enrich_df['n_pw_in_top'] > 0].head(top_n).copy()
    if df.empty:
        print(f'  [WARN] {dataset} 无富集通路')
        return

    df = df.sort_values('neg_log10_p', ascending=True)
    y = np.arange(len(df))
    sizes = df['n_pw_in_top'].values * 50

    # 图宽根据 x 轴数据范围自适应（紧凑）
    x_max = df['neg_log10_p'].max()
    fig_w = max(4.5, min(6.5, x_max * 0.6 + 2.5))
    fig_h = max(3.5, len(df) * 0.38 + 1.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sc = ax.scatter(df['neg_log10_p'], y, s=sizes, c=df['neg_log10_p'],
                    cmap='Blues', vmin=0, vmax=x_max,
                    edgecolors='#333333', linewidths=0.5, zorder=3)
    ax.axvline(-np.log10(0.05), color='#999999', linestyle='--', linewidth=0.8,
               label='p = 0.05')

    # 紧凑 x 轴：右侧只留 5% 空白
    ax.set_xlim(0, x_max * 1.08)

    ax.set_yticks(y)
    ax.set_yticklabels(df['pathway'], fontsize=8.5)
    ax.set_xlabel('−log₁₀(p-value)', fontsize=9)
    ax.set_title(f'Pathway Enrichment — {DATASET_LABELS.get(dataset, dataset)}',
                 fontsize=10, pad=8, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 网格线
    for yi in y:
        ax.axhline(yi, color='#f0f0f0', linewidth=0.4, zorder=0)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.03, aspect=18)
    cbar.set_label('−log₁₀(p)', fontsize=8)

    # 点大小图例
    for size_val in [1, 3, 5]:
        ax.scatter([], [], s=size_val*50, c='#888888', label=f'{size_val} LR pair{"s" if size_val>1 else ""}',
                   edgecolors='#444444', linewidths=0.5)
    ax.legend(loc='lower right', fontsize=7.5, frameon=False)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        out_path = FIG_DIR / f'{out_prefix}.{ext}'
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f'    保存: {out_path}')
    plt.close(fig)


def main():
    print('\n' + '='*55)
    print('  通路富集分析（Fig.5b）')
    print('='*55)

    lr2pathway, n_pw = load_pathway_map_simple()
    if not lr2pathway:
        print('  [ERROR] 无法加载通路映射，退出')
        return

    all_lr_set = load_all_lr_pairs()
    print(f'  背景 LR pairs: {len(all_lr_set)}')

    for ds in DATASETS:
        print(f'\n  {ds}:')
        # top_lr_pairs 由 plot_top_lr_pairs.py 保存到 output/data/
        top_path = OUTPUT_DIR / 'data' / f'top_lr_pairs_{ds}.csv'
        if not top_path.exists():
            top_path = OUTPUT_DIR / f'top_lr_pairs_{ds}.csv'   # 旧路径兼容
        if not top_path.exists():
            print(f'    [WARN] 未找到 {top_path}，请先运行 plot_top_lr_pairs.py')
            continue
        top_df = pd.read_csv(top_path)
        top_lr = [(row['ligand'], row['receptor']) for _, row in top_df.iterrows()
                  if 'ligand' in top_df.columns]
        if not top_lr:
            # 从 lr_pair 列解析
            top_lr = [(r.split('__')[0], r.split('__')[1])
                      for r in top_df['lr_pair']]
        print(f'    Top-{len(top_lr)} LR pairs')

        enrich = fisher_enrichment(top_lr, all_lr_set, lr2pathway)
        csv_path = OUTPUT_DIR / f'pathway_enrichment_{ds}.csv'
        enrich.to_csv(csv_path, index=False)
        print(f'    富集结果已保存: {csv_path}')
        sig = enrich[enrich['pvalue'] < 0.05]
        print(f'    显著富集通路（p<0.05）: {len(sig)}')
        if not sig.empty:
            print(sig[['pathway', 'n_pw_in_top', 'pvalue']].head(5).to_string(index=False))
        plot_dot(enrich, ds)

    print('\n  完成。')


if __name__ == '__main__':
    main()
