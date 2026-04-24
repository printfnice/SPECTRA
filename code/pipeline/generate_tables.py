# generate_tables.py
# 职责: 将实验结果数据生成三线格 LaTeX 表格，保存到 output/tables/
# 输入: output/data/*.csv
# 输出: output/tables/Table_*.tex + Table_*.md (Markdown 预览版)

from pathlib import Path
import pandas as pd
import numpy as np

_PIPELINE_DIR = Path(__file__).parent
_CLASS_DIR = _PIPELINE_DIR.parent.parent
OUTPUT_DIR = _CLASS_DIR / 'output'
DATA_DIR = OUTPUT_DIR / 'data'
TABLES_DIR = OUTPUT_DIR / 'tables'
TABLES_DIR.mkdir(exist_ok=True)


# ── 工具函数 ────────────────────────────────────────────────────────────────

def write_table(name: str, tex: str, md: str):
    (TABLES_DIR / f'{name}.tex').write_text(tex)
    (TABLES_DIR / f'{name}.md').write_text(md)
    print(f'  ✓ {name}.tex / .md')


def booktabs_tex(caption: str, label: str, header: list[str],
                 rows: list[list[str]], notes: str = '') -> str:
    col_fmt = 'l' + 'c' * (len(header) - 1)
    h = ' & '.join(header) + r' \\'
    body_lines = []
    for row in rows:
        body_lines.append(' & '.join(str(c) for c in row) + r' \\')
    body = '\n        '.join(body_lines)
    note_block = (f'\n    \\begin{{tablenotes}}\n      \\small\\item {notes}\n'
                  r'    \end{tablenotes}') if notes else ''
    return rf"""\begin{{table}}[htbp]
  \centering
  \caption{{{caption}}}
  \label{{{label}}}
  \begin{{threeparttable}}
  \begin{{tabular}}{{{col_fmt}}}
    \toprule
    {h}
    \midrule
        {body}
    \bottomrule
  \end{{tabular}}{note_block}
  \end{{threeparttable}}
\end{{table}}
"""


def markdown_table(header: list[str], rows: list[list[str]],
                   title: str = '', notes: str = '') -> str:
    lines = []
    if title:
        lines.append(f'## {title}\n')
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('|' + '---|' * len(header))
    for row in rows:
        lines.append('| ' + ' | '.join(str(c) for c in row) + ' |')
    if notes:
        lines.append(f'\n*{notes}*')
    return '\n'.join(lines) + '\n'


# ── Table 1: SupCon 消融（Fig3）─────────────────────────────────────────────

def make_supcon_table():
    df = pd.read_csv(DATA_DIR / 'supcon_ablation_table.csv')

    config_names = {
        'baseline_R38':    'VIB + MC Dropout (baseline)',
        'supcon_w01':      r'+SupCon ($w$=0.1)',
        'supcon_w02':      r'+SupCon ($w$=0.2)',
        'supcon_w02_mixoff': r'+SupCon ($w$=0.2), Mixup OFF',
        'supcon_w03_mixoff': r'+SupCon ($w$=0.3), Mixup OFF',
        'sgdr':            r'\textbf{+SGDR (final)}',
    }
    config_names_md = {
        'baseline_R38':    'VIB + MC Dropout (baseline)',
        'supcon_w01':      '+SupCon (w=0.1)',
        'supcon_w02':      '+SupCon (w=0.2)',
        'supcon_w02_mixoff': '+SupCon (w=0.2), Mixup OFF',
        'supcon_w03_mixoff': '+SupCon (w=0.3), Mixup OFF',
        'sgdr':            '**+SGDR (final)**',
    }

    header = ['Configuration', 'AUROC', r'$\pm$ Std',
              r'$\Delta$ vs Baseline', r'$\Delta$ vs CellChat']
    header_md = ['Configuration', 'AUROC', '±Std', 'Δ vs Baseline', 'Δ vs CellChat']

    rows_tex, rows_md = [], []
    for _, r in df.iterrows():
        tag = r['Tag']
        auroc = f"{r['AUROC']:.4f}"
        std   = f"{r['±std']:.4f}"
        d_base = f"{r['Δ vs baseline']:+.4f}"
        d_cc   = f"{r['Δ vs CellChat']:+.4f}"
        rows_tex.append([config_names.get(tag, tag), auroc, std, d_base, d_cc])
        rows_md.append([config_names_md.get(tag, tag), auroc, std, d_base, d_cc])

    caption = ('Progressive Regularization Ablation on Reichart (DCM, $N$=171). '
               'CellChat baseline AUROC = 0.9655.')
    notes = r'All results: 5-fold CV, mean $\pm$ std. SGDR = cosine annealing with warm restarts.'
    tex = booktabs_tex(caption, 'tab:supcon_ablation', header, rows_tex, notes)
    md  = markdown_table(header_md, rows_md,
                         'Table S-Ablation: Progressive Regularization (Reichart DCM)',
                         'All results: 5-fold CV, mean ± std. CellChat baseline = 0.9655.')
    write_table('Table_S1_supcon_ablation', tex, md)


# ── Table 2: Gate 权重 + 生物学含义（Fig4）──────────────────────────────────

def make_gate_weights_table():
    df = pd.read_csv(DATA_DIR / 'gate_weights_summary.csv')

    bio_meanings = {
        'kuppe':     ('心肌梗死由特定基因表达驱动，且表现为细胞类型特异性的通讯模式。'
                      'MI is driven by specific gene expression changes with '
                      'cell-type-specific communication patterns.'),
        'carraro':   ('囊性纤维化由 CFTR 通路系统性异常跨多细胞类型驱动。'
                      'CF is driven by systemic CFTR pathway dysregulation '
                      'across cell types.'),
        'habermann': ('肺纤维化由 TGF-β/PDGF 信号通路的跨细胞类型激活主导。'
                      'PF is dominated by cross-cell-type activation of TGF-β/PDGF pathways.'),
        'velmeshev': ('自闭症由神经发育信号通路极高度主导，基因和细胞类型贡献极小。'
                      'ASD is almost entirely dominated by neurodevelopmental '
                      'signaling pathways, with minimal gene/cell-type contribution.'),
        'reichart':  ('扩张型心肌病由基因突变（TTN/LMNA）主导的基因层面信号驱动。'
                      'DCM is driven by gene-level signals reflecting TTN/LMNA mutations.'),
    }

    dataset_labels = {
        'kuppe':     'Kuppe (MI)',
        'carraro':   'Carraro (CF)',
        'habermann': 'Habermann (PF)',
        'velmeshev': 'Velmeshev (ASD)',
        'reichart':  'Reichart (DCM)',
    }

    header = ['Dataset', 'Gene-level', 'Pathway-level', 'Cell-type-level',
              'Dominant Scale', 'Biological Interpretation']
    rows_tex, rows_md = [], []

    for _, r in df.iterrows():
        ds = r['dataset']
        wg = f"{r['w_gene']:.3f}"
        wp = f"{r['w_pathway']:.3f}"
        wc = f"{r['w_celltype']:.3f}"
        # 主导尺度
        vals = {'Gene': r['w_gene'], 'Pathway': r['w_pathway'], 'Cell-type': r['w_celltype']}
        dom = max(vals, key=vals.get)
        bio_tex = bio_meanings[ds].split('. ', 1)[-1]  # 英文部分
        bio_md  = bio_meanings[ds]
        label = dataset_labels[ds]
        rows_tex.append([label, wg, wp, wc, dom, bio_tex])
        rows_md.append([label, wg, wp, wc, dom, bio_md])

    caption = ('SPECTRA Gate Network learned scale weights per disease dataset. '
               r'Weights sum to 1.0 for each dataset ($w_\text{gene} + '
               r'w_\text{pathway} + w_\text{celltype} = 1$). '
               'The dominant scale reflects the primary biological mechanism of each disease.')
    notes = ('Weights averaged over all LR pairs in the dataset. '
             'MI=Myocardial Infarction, CF=Cystic Fibrosis, PF=Pulmonary Fibrosis, '
             'ASD=Autism Spectrum Disorder, DCM=Dilated Cardiomyopathy.')
    tex = booktabs_tex(caption, 'tab:gate_weights', header, rows_tex, notes)
    md  = markdown_table(header, rows_md,
                         'Table 2: Gate Network Scale Weights per Disease',
                         notes)
    write_table('Table_2_gate_weights', tex, md)


# ── Table 3: 统计显著性（ExtFig_stat）────────────────────────────────────────

def make_stat_test_table():
    df = pd.read_csv(DATA_DIR / 'statistical_tests.csv')

    method_display = {
        'geometric_mean':   'Geometric Mean',
        'cellchat':         'CellChat',
        'cellphonedb':      'CellPhoneDB',
        'scseqcomm':        'scSeqComm',
        'natmi':            'NATMI',
        'connectome':       'Connectome',
        'logfc':            'LogFC',
    }
    ds_cols = ['delta_kuppe', 'delta_carraro', 'delta_habermann',
               'delta_velmeshev', 'delta_reichart']
    ds_labels = ['Kuppe\n(MI)', 'Carraro\n(CF)', 'Habermann\n(PF)',
                 'Velmeshev\n(ASD)', 'Reichart\n(DCM)']
    ds_labels_md = ['Kuppe (MI)', 'Carraro (CF)', 'Habermann (PF)',
                    'Velmeshev (ASD)', 'Reichart (DCM)']

    header_tex = (['Baseline Method', r'$p$-value', r'Effect size ($r$)',
                   r'Mean $\Delta$AUROC'] +
                  [f'$\\Delta$ {l}' for l in ['Kuppe', 'Carraro', 'Habermann',
                                               'Velmeshev', 'Reichart']] + ['Sig.'])
    header_md = (['Baseline Method', 'p-value', 'Effect size (r)', 'Mean ΔAUROC'] +
                 ds_labels_md + ['Sig.'])

    rows_tex, rows_md = [], []
    for _, r in df.iterrows():
        p = r['p_value']
        p_str = f'{p:.2e}'
        eff   = f"{r['effect_r']:.3f}"
        mean  = f"{r['mean_delta']:+.4f}"
        deltas_tex = [f"{r[c]:+.4f}" for c in ds_cols]
        deltas_md  = [f"{r[c]:+.4f}" for c in ds_cols]
        sig   = r['label']
        mname = method_display.get(r['method'], r['method'])
        rows_tex.append([mname, p_str, eff, mean] + deltas_tex + [sig])
        rows_md.append([mname, p_str, eff, mean] + deltas_md + [sig])

    caption = (r'Statistical significance of SPECTRA over LIANA baseline methods. '
               r'Wilcoxon one-sided signed-rank test, $n$=25 (5 folds $\times$ 5 datasets). '
               r'$\Delta$AUROC = SPECTRA $-$ baseline.')
    notes = ('*** p < 0.001. Effect size: rank-biserial correlation r. '
             'All comparisons significant at p < 0.001.')
    tex = booktabs_tex(caption, 'tab:stat_test', header_tex, rows_tex, notes)
    md  = markdown_table(header_md, rows_md,
                         'Table S2: Statistical Significance vs Baseline Methods',
                         notes)
    write_table('Table_S2_statistical_tests', tex, md)


# ── Table 4: Bootstrap 95% CI（FigS_bootstrap）───────────────────────────────

def make_bootstrap_table():
    df = pd.read_csv(DATA_DIR / 'bootstrap_ci_table.csv')

    sota = {
        'kuppe': 1.0000, 'carraro': 0.8797, 'habermann': 0.7600,
        'velmeshev': 0.6957, 'reichart': 0.9655,
    }
    sota_method = {
        'kuppe': 'NATMI', 'carraro': 'Geo. Mean', 'habermann': 'NATMI',
        'velmeshev': 'NATMI', 'reichart': 'CellChat',
    }
    ds_labels = {
        'kuppe': 'Kuppe (MI)', 'carraro': 'Carraro (CF)',
        'habermann': 'Habermann (PF)', 'velmeshev': 'Velmeshev (ASD)',
        'reichart': 'Reichart (DCM)',
    }

    header = ['Dataset', 'N', 'AUROC', '95% CI', 'SOTA Baseline',
              'SOTA Method', 'ΔAUROC vs SOTA']
    rows_tex, rows_md = [], []

    n_map = {'kuppe': 23, 'carraro': 16, 'habermann': 18,
             'velmeshev': 38, 'reichart': 171}

    for _, r in df.iterrows():
        ds = r['dataset']
        ci = f"[{r['ci_lower_95']:.3f}, {r['ci_upper_95']:.3f}]"
        delta = f"{r['delta_vs_sota']:+.4f}"
        rows_tex.append([
            ds_labels[ds], n_map[ds],
            f"{r['mean_auroc']:.4f}", ci,
            f"{sota[ds]:.4f}", sota_method[ds], delta,
        ])
        rows_md.append([
            ds_labels[ds], n_map[ds],
            f"{r['mean_auroc']:.4f}", ci,
            f"{sota[ds]:.4f}", sota_method[ds], delta,
        ])

    caption = ('SPECTRA mean AUROC with 95\\% confidence intervals (t-distribution, '
               '5-fold CV). SOTA baselines use the same PMA+VIB+MLP downstream classifier.')
    notes = ('CI computed from 5-fold AUROC values using t-distribution. '
             'SOTA: best-performing LIANA baseline per dataset under fair comparison.')
    tex = booktabs_tex(caption, 'tab:bootstrap_ci', header, rows_tex, notes)
    md  = markdown_table(header, rows_md,
                         'Table S3: SPECTRA AUROC with 95% Confidence Intervals',
                         notes)
    write_table('Table_S3_bootstrap_ci', tex, md)


# ── Table 5: 计算效率（FigS_efficiency）──────────────────────────────────────

def make_efficiency_table():
    df = pd.read_csv(DATA_DIR / 'efficiency_table.csv')

    header = ['Dataset', 'Disease', 'N',
              'SPECTRA E2E (GPU, min)', 'LIANA+Tensor+RF (CPU, min)',
              'Speedup ratio', 'Inference only (sec)']
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r['Dataset'], r['Disease'], int(r['N (samples)']),
            r["SPECTRA E2E (GPU, min)"],
            r["LIANA+Tensor+RF (CPU, min)"],
            r["Speedup factor (E2E/LIANA)"],
            r["E2E inference only (sec)"],
        ])

    caption = ('Computational efficiency comparison between SPECTRA (end-to-end, GPU) '
               'and the LIANA+Tensor-cell2cell+RF pipeline (CPU). '
               'E2E time includes full 5-fold cross-validation. '
               'LIANA time is per single method (7 methods can run in parallel).')
    notes = ('GPU: NVIDIA A100 40GB. CPU: 32-core Intel Xeon. '
             'Inference: time to predict one new sample after training. '
             'Speedup = E2E total / LIANA single-method time.')
    tex = booktabs_tex(caption, 'tab:efficiency', header, rows, notes)
    md  = markdown_table(header, rows,
                         'Table S4: Computational Efficiency',
                         notes)
    write_table('Table_S4_efficiency', tex, md)


# ── 主入口 ──────────────────────────────────────────────────────────────────

def main():
    print('生成三线格表格...')
    make_supcon_table()
    make_gate_weights_table()
    make_stat_test_table()
    make_bootstrap_table()
    make_efficiency_table()
    print(f'\n全部完成 → {TABLES_DIR}')


if __name__ == '__main__':
    main()
