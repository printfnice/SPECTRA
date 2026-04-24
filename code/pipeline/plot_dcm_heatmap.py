# plot_dcm_heatmap.py
# 依赖: pandas, matplotlib, numpy, scipy, pub_style.py
# 被依赖: 直接运行
# 职责: Extended Fig.3 Reichart(DCM) vs Healthy 通讯热图
#
# 设计：
#   1. 从 data/pathway_enrichment_reichart.csv 取显著富集通路（p<0.05）
#   2. 从 data/cellchat_pathway_map.csv 取这些通路的所有 LR pairs
#   3. 从 reichart_gate_weights.csv 提取 w_pathway
#      - 若有 sample_id 列：计算 DCM 均值 − Healthy 均值差值热图
#      - 否则：全体均值热图
#   4. 绘制 (LR pair × cell-type pair) 热图
#
# 参考: CellChat Nature Commun 2021 Fig.4 heatmap 样式
# 输出: output/figures/ExtFig3_dcm_heatmap.pdf/.png

import sys
from pathlib import Path
import numpy as np
import pandas as pd

_PIPELINE_DIR = Path(__file__).parent
_CODE_DIR     = _PIPELINE_DIR.parent
_CLASS_DIR    = _CODE_DIR.parent
sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_PIPELINE_DIR))

from pub_style import (set_pub_style, save_fig, FIGURES_DIR, DATA_DIR,
                       OUTPUT_DIR, NATURE_CATEGORICAL)

DATA_DIR_MAIN = _CLASS_DIR / 'data'   # data/cellchat_pathway_map.csv

TOP_LR_PER_PW = 3
TOP_CT_PAIRS  = 10

# Reichart sample → condition mapping（从 filtered h5ad 派生，硬编码 cache）
SAMPLE_CONDITION_CSV = DATA_DIR / 'reichart_sample_conditions.csv'


def _build_sample_condition_cache():
    """从 h5ad 构建 sample_id → condition 映射，保存到 data/。"""
    try:
        import scanpy as sc
        h5ad = _CLASS_DIR / 'data' / 'interim' / 'reichart_filtered.h5ad'
        if not h5ad.exists():
            return None
        adata = sc.read_h5ad(h5ad)
        obs = adata.obs
        sc_map = (obs[['Sample', 'disease']]
                  .drop_duplicates(subset='Sample')
                  .rename(columns={'Sample': 'sample_id', 'disease': 'condition'}))
        sc_map['condition'] = sc_map['condition'].astype(str).map({
            'dilated cardiomyopathy': 'DCM',
            'normal': 'Healthy',
        }).fillna('Unknown')
        DATA_DIR.mkdir(exist_ok=True)
        sc_map.to_csv(SAMPLE_CONDITION_CSV, index=False)
        print(f'  条件映射已保存: {SAMPLE_CONDITION_CSV}  '
              f'(DCM={sum(sc_map.condition=="DCM")}, '
              f'Healthy={sum(sc_map.condition=="Healthy")})')
        return sc_map
    except Exception as e:
        print(f'  [WARN] 无法构建条件映射: {e}')
        return None


def load_sample_conditions():
    """加载 sample_id → condition 映射（DCM / Healthy）。"""
    if SAMPLE_CONDITION_CSV.exists():
        return pd.read_csv(SAMPLE_CONDITION_CSV)
    return _build_sample_condition_cache()


def load_enriched_lr_pairs():
    """从显著富集通路中提取 LR pairs。"""
    enrich_path = DATA_DIR / 'pathway_enrichment_reichart.csv'
    if not enrich_path.exists():
        print(f'  [ERROR] 未找到 {enrich_path}')
        return {}, []
    enrich = pd.read_csv(enrich_path)
    sig_pw = enrich[enrich['pvalue'] < 0.05]['pathway'].tolist()
    print(f'  显著通路（p<0.05）: {sig_pw}')

    ccdb = pd.read_csv(DATA_DIR_MAIN / 'cellchat_pathway_map.csv')
    target_lr = ccdb[ccdb['pathway'].isin(sig_pw)].copy()
    lr_info = {}
    for _, row in target_lr.iterrows():
        lr_info[(row['ligand'], row['receptor'])] = row['pathway']
    print(f'  目标 LR pairs: {len(lr_info)}')
    return lr_info, sig_pw


def load_gate_for_lr_pairs(lr_info, chunk_size=500_000):
    """逐块读取 reichart gate CSV，只保留目标 LR pairs。"""
    gate_path = OUTPUT_DIR / 'reichart_gate_weights.csv'
    target_pairs = set(f'{l}__{r}' for (l, r) in lr_info.keys())
    print(f'  读取 gate CSV（逐块，目标 {len(target_pairs)} LR pairs）...')

    collected = []
    for chunk in pd.read_csv(gate_path, chunksize=chunk_size):
        parts = chunk['lr_name'].str.split('__', expand=True)
        chunk = chunk.copy()
        chunk['ligand']    = parts[0]
        chunk['receptor']  = parts[1]
        chunk['source_ct'] = parts[2]
        chunk['target_ct'] = parts[3]
        chunk['lr_pair']   = chunk['ligand'] + '__' + chunk['receptor']
        mask = chunk['lr_pair'].isin(target_pairs)
        if mask.any():
            collected.append(chunk[mask])
    if not collected:
        return None
    df = pd.concat(collected, ignore_index=True)
    df['pathway'] = df.apply(
        lambda r: lr_info.get((r['ligand'], r['receptor']), 'Unknown'), axis=1)
    df['ct_pair'] = df['source_ct'] + ' → ' + df['target_ct']
    has_sid = 'sample_id' in df.columns
    print(f'  匹配行数: {len(df):,}  has_sample_id={has_sid}')
    return df


def build_heatmap_matrix(df, sig_pw, top_ct=TOP_CT_PAIRS, sample_cond=None):
    """
    构建热图矩阵。
    若有 sample_id + sample_cond：返回 (dcm_pivot - healthy_pivot, dcm_pivot, healthy_pivot)
    否则：返回 (agg_pivot, None, None)
    """
    lr_order, lr_pw_map = [], {}
    for pw in sig_pw:
        sub = df[df['pathway'] == pw]
        if sub.empty:
            continue
        top = sub.groupby('lr_pair')['w_pathway'].mean().nlargest(TOP_LR_PER_PW).index
        for lr in top:
            lr_order.append(lr)
            lr_pw_map[lr] = pw

    ct_rank = df.groupby('ct_pair')['w_pathway'].mean().nlargest(top_ct).index.tolist()

    sub = df[df['lr_pair'].isin(lr_order) & df['ct_pair'].isin(ct_rank)]

    use_diff = (sample_cond is not None and 'sample_id' in df.columns
                and sub['sample_id'].notna().any())

    if use_diff:
        merged = sub.merge(sample_cond, on='sample_id', how='left')
        dcm = merged[merged['condition'] == 'DCM']
        hlt = merged[merged['condition'] == 'Healthy']
        dcm_piv = (dcm.groupby(['lr_pair', 'ct_pair'])['w_pathway']
                   .mean().unstack(fill_value=np.nan)
                   .reindex(index=lr_order, columns=ct_rank))
        hlt_piv = (hlt.groupby(['lr_pair', 'ct_pair'])['w_pathway']
                   .mean().unstack(fill_value=np.nan)
                   .reindex(index=lr_order, columns=ct_rank))
        diff_piv = dcm_piv.fillna(0) - hlt_piv.fillna(0)
        return diff_piv, dcm_piv, hlt_piv, lr_pw_map
    else:
        pivot = (sub.groupby(['lr_pair', 'ct_pair'])['w_pathway']
                 .mean().unstack(fill_value=np.nan)
                 .reindex(index=lr_order, columns=ct_rank))
        return pivot, None, None, lr_pw_map


def _draw_single_heatmap(ax, data, title, cmap, vmin, vmax,
                         lr_pw_map, sig_pw, show_ylab=True):
    """共用单面板绘图函数。"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches2
    n_lr, n_ct = data.shape

    # aspect='equal' 保证每个格子是正方形
    im = ax.imshow(data.values, aspect='equal', cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation='nearest')

    ax.set_xticks(range(n_ct))
    ax.set_xticklabels(data.columns, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n_lr))
    if show_ylab:
        lr_labels = []
        for lr in data.index:
            pw = lr_pw_map.get(lr, '')
            l, r = lr.split('__')
            lr_labels.append(f'{l}–{r}  [{pw}]')
        ax.set_yticklabels(lr_labels, fontsize=7.5)
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=8.5, pad=6, fontweight='bold')

    # 通路分割线（细线，仅在通路切换处）
    pw_assign = [lr_pw_map.get(lr, '') for lr in data.index]
    for i in range(1, n_lr):
        if pw_assign[i] != pw_assign[i-1]:
            ax.axhline(i - 0.5, color='white', linewidth=0.6, zorder=4)

    # 通路颜色：直接着色 y 轴标签（避免 Rectangle 与 imshow 坐标冲突）
    pw_colors = {pw: c for pw, c in zip(sig_pw, NATURE_CATEGORICAL)}
    if show_ylab:
        for tick, lr in zip(ax.get_yticklabels(), data.index):
            c = pw_colors.get(lr_pw_map.get(lr, ''), '#444444')
            tick.set_color(c)
    return im, pw_colors


def plot_dcm_heatmap(result, dcm_piv, hlt_piv, lr_pw_map, sig_pw):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    set_pub_style()

    if result is None or result.empty:
        print('  [WARN] 矩阵为空，跳过')
        return

    n_lr, n_ct = result.shape
    fig_h = max(4.5, n_lr * 0.55 + 2.5)

    use_diff = (dcm_piv is not None and hlt_piv is not None)

    if use_diff:
        # 三面板：DCM | Healthy | DCM−Healthy
        # aspect='equal' 时 axes 高度 ≈ n_lr/n_ct * axes_width
        # 给每个面板 (n_ct+1.5) * 0.55 宽（+1.5 容纳左侧色条）
        cell_w = 0.55
        panel_w = (n_ct + 1.5) * cell_w
        fig_w = panel_w * 3 + 4.0
        fig_h = max(5.0, n_lr * cell_w + 3.0)
        fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h),
                                 gridspec_kw={'width_ratios': [1, 1, 1]})
        ax_dcm, ax_hlt, ax_diff = axes

        vmax_abs = max(
            np.nanpercentile(np.abs(dcm_piv.values[~np.isnan(dcm_piv.values)]), 95),
            np.nanpercentile(np.abs(hlt_piv.values[~np.isnan(hlt_piv.values)]), 95),
        ) if not np.all(np.isnan(dcm_piv.values)) else 1.0
        vdiff = np.nanpercentile(np.abs(result.values[~np.isnan(result.values)]), 95) \
            if not np.all(np.isnan(result.values)) else 1.0

        im_d, pw_colors = _draw_single_heatmap(
            ax_dcm, dcm_piv, '(A) DCM (mean w_pathway)',
            'Reds', 0, vmax_abs, lr_pw_map, sig_pw, show_ylab=True)
        im_h, _ = _draw_single_heatmap(
            ax_hlt, hlt_piv, '(B) Healthy (mean w_pathway)',
            'Blues', 0, vmax_abs, lr_pw_map, sig_pw, show_ylab=False)
        im_f, _ = _draw_single_heatmap(
            ax_diff, result, '(C) DCM − Healthy (Δw_pathway)',
            'RdBu_r', -vdiff, vdiff, lr_pw_map, sig_pw, show_ylab=False)

        for im, ax in [(im_d, ax_dcm), (im_h, ax_hlt), (im_f, ax_diff)]:
            cb = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02, aspect=20)
            cb.ax.tick_params(labelsize=6.5)

        fig.suptitle(
            'Reichart (DCM): Pathway Gate Weights — DCM vs Healthy\n'
            '(LR pairs from significantly enriched pathways, Fisher\'s p < 0.05)',
            fontsize=9.5, y=1.01, fontweight='bold')

    else:
        # 单面板：全体均值（无 sample_id 时兜底）
        vmax = np.nanpercentile(result.values[~np.isnan(result.values)], 95) \
            if not np.all(np.isnan(result.values)) else 1.0
        fig_w = max(8, n_ct * 0.75 + 3)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im, pw_colors = _draw_single_heatmap(
            ax, result, 'Reichart (DCM): Mean Pathway Gate Weights',
            'RdYlBu_r', 0, vmax, lr_pw_map, sig_pw, show_ylab=True)
        cb = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02, aspect=20)
        cb.set_label('Mean w_pathway', fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax.set_title(
            'Reichart (DCM): Pathway Gate Weights\n'
            '(LR pairs from significantly enriched pathways, Fisher\'s p < 0.05)',
            fontsize=9, pad=10, fontweight='bold')
        ax.set_xlabel('Cell-type pair (source → target)', fontsize=9)

    # 通路图例（共用）
    pw_colors = {pw: c for pw, c in zip(sig_pw, NATURE_CATEGORICAL)}
    legend_handles = [mpatches.Patch(facecolor=pw_colors.get(pw, '#ccc'), label=pw)
                      for pw in sig_pw if pw in pw_colors]
    (axes[-1] if use_diff else ax).legend(
        handles=legend_handles, title='Enriched Pathway',
        loc='upper left', bbox_to_anchor=(1.05, 1.0),
        frameon=False, fontsize=7.5, title_fontsize=8)

    plt.tight_layout()
    save_fig(fig, FIGURES_DIR / 'ExtFig3_dcm_heatmap')
    plt.close(fig)
    print(f'  ExtFig3 完成（{"DCM vs Healthy 三面板" if use_diff else "全体均值（无 sample_id）"})。')


def main():
    print('\n' + '='*55)
    print('  Extended Fig.3 DCM vs Healthy 通讯热图（发表级）')
    print('='*55)

    lr_info, sig_pw = load_enriched_lr_pairs()
    if not lr_info:
        print('  [ERROR] 无显著通路或 LR pairs')
        return

    # 尝试加载 sample → condition 映射
    sample_cond = load_sample_conditions()
    if sample_cond is not None:
        print(f'  样本标签: {len(sample_cond)} 样本')
    else:
        print('  [INFO] gate CSV 无 sample_id，退回全体均值模式')

    df = load_gate_for_lr_pairs(lr_info)
    if df is None:
        print('  [ERROR] gate CSV 中未找到目标 LR pairs')
        return

    result, dcm_piv, hlt_piv, lr_pw_map = build_heatmap_matrix(
        df, sig_pw, sample_cond=sample_cond)
    print(f'  热图矩阵: {result.shape}')
    plot_dcm_heatmap(result, dcm_piv, hlt_piv, lr_pw_map, sig_pw)
    print('\n  完成。')


if __name__ == '__main__':
    main()
