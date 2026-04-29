# disease_lr_weights.py
# 依赖: liana, numpy, json, sklearn (TfidfVectorizer)
# 被依赖: ms_hgatlink_method.py
# 职责: 基于疾病通路知识生成 LR 对重要性权重（离线计算，无需 API）

"""
方案 D: 利用已有生物学知识（疾病相关通路/基因列表）为每个 LR 对
生成疾病相关性权重，作为评分后处理的先验调制因子。

知识来源：
1. 人工整理的疾病核心通路/基因（来自文献，覆盖 5 个数据集）
2. LR 对与通路基因的匹配度 → 归一化为 [0, 1] 权重

用法:
    python disease_lr_weights.py                # 生成全部 5 个数据集的权重
    python disease_lr_weights.py --dataset kuppe # 生成单个数据集
"""

import json
import argparse
import numpy as np
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent.parent / 'data'

# ============================================================================
# 疾病相关通路/基因知识库（文献来源，人工整理）
# ============================================================================

# 每个数据集的疾病关键通路及其核心基因
# 来源: PubMed 文献综述 + KEGG/Reactome 通路数据库
_DISEASE_KNOWLEDGE = {
    'kuppe': {
        'disease': 'acute myocardial infarction',
        'pathways': {
            'TGF-beta/fibrosis': {
                'genes': ['TGFB1', 'TGFB2', 'TGFB3', 'TGFBR1', 'TGFBR2',
                          'ACVR1', 'ACVR2A', 'SMAD2', 'SMAD3', 'SMAD4',
                          'COL1A1', 'COL1A2', 'COL3A1', 'FN1', 'POSTN'],
                'weight': 1.0,
            },
            'inflammatory': {
                'genes': ['TNF', 'TNFRSF1A', 'TNFRSF1B', 'IL1B', 'IL1R1',
                          'IL6', 'IL6R', 'IL6ST', 'CCL2', 'CCR2',
                          'CXCL12', 'CXCR4', 'CSF1', 'CSF1R'],
                'weight': 0.9,
            },
            'angiogenesis': {
                'genes': ['VEGFA', 'VEGFB', 'VEGFC', 'FLT1', 'KDR',
                          'FGF2', 'FGFR1', 'PDGFA', 'PDGFB', 'PDGFRA',
                          'PDGFRB', 'ANGPT1', 'ANGPT2', 'TEK'],
                'weight': 0.8,
            },
            'WNT/cardiac_remodeling': {
                'genes': ['WNT3A', 'WNT5A', 'WNT11', 'FZD1', 'FZD2',
                          'FZD4', 'FZD7', 'LRP5', 'LRP6', 'DKK1',
                          'SFRP1', 'SFRP2'],
                'weight': 0.7,
            },
            'Notch/cell_fate': {
                'genes': ['DLL1', 'DLL4', 'JAG1', 'JAG2', 'NOTCH1',
                          'NOTCH2', 'NOTCH3', 'NOTCH4'],
                'weight': 0.6,
            },
        },
    },
    'habermann': {
        'disease': 'idiopathic pulmonary fibrosis',
        'pathways': {
            'TGF-beta/fibrosis': {
                'genes': ['TGFB1', 'TGFB2', 'TGFB3', 'TGFBR1', 'TGFBR2',
                          'ACVR1', 'BMP2', 'BMP4', 'BMP7', 'BMPR1A',
                          'BMPR2', 'SMAD2', 'SMAD3'],
                'weight': 1.0,
            },
            'FGF/epithelial_repair': {
                'genes': ['FGF1', 'FGF2', 'FGF7', 'FGF10', 'FGFR1',
                          'FGFR2', 'FGFR3', 'FGFR4'],
                'weight': 0.9,
            },
            'WNT/fibrosis': {
                'genes': ['WNT2', 'WNT3A', 'WNT5A', 'WNT7B', 'WNT10A',
                          'FZD1', 'FZD2', 'FZD4', 'FZD7', 'LRP5', 'LRP6'],
                'weight': 0.9,
            },
            'PDGF/myofibroblast': {
                'genes': ['PDGFA', 'PDGFB', 'PDGFC', 'PDGFD', 'PDGFRA',
                          'PDGFRB'],
                'weight': 0.85,
            },
            'Hedgehog/fibrosis': {
                'genes': ['SHH', 'IHH', 'DHH', 'PTCH1', 'PTCH2', 'SMO'],
                'weight': 0.7,
            },
            'inflammatory': {
                'genes': ['TNF', 'TNFRSF1A', 'IL1B', 'IL1R1', 'IL6',
                          'IL6R', 'IL13', 'IL13RA1', 'IL13RA2',
                          'CCL2', 'CCR2', 'CXCL12', 'CXCR4'],
                'weight': 0.8,
            },
            'ECM/integrin': {
                'genes': ['COL1A1', 'COL1A2', 'COL3A1', 'COL4A1', 'FN1',
                          'LAMA1', 'LAMB1', 'LAMC1', 'ITGA1', 'ITGA2',
                          'ITGA5', 'ITGAV', 'ITGB1', 'ITGB3', 'ITGB6'],
                'weight': 0.75,
            },
        },
    },
    'velmeshev': {
        'disease': 'autism spectrum disorder',
        'pathways': {
            'synaptic_adhesion': {
                'genes': ['NRXN1', 'NRXN2', 'NRXN3', 'NLGN1', 'NLGN2',
                          'NLGN3', 'NLGN4X', 'LRRTM1', 'LRRTM2',
                          'CNTNAP2', 'CNTN1', 'CNTN2'],
                'weight': 1.0,
            },
            'neurotrophic': {
                'genes': ['BDNF', 'NTRK2', 'NGF', 'NTRK1', 'NTF3',
                          'NTRK3', 'GDNF', 'GFRA1', 'RET'],
                'weight': 0.9,
            },
            'WNT/neurodevelopment': {
                'genes': ['WNT3', 'WNT5A', 'WNT7A', 'WNT7B', 'FZD3',
                          'FZD5', 'FZD9', 'LRP5', 'LRP6', 'RSPO1',
                          'RSPO3', 'ROR1', 'ROR2'],
                'weight': 0.85,
            },
            'Notch/neural_patterning': {
                'genes': ['DLL1', 'DLL3', 'JAG1', 'JAG2', 'NOTCH1',
                          'NOTCH2', 'NOTCH3'],
                'weight': 0.8,
            },
            'immune/microglia': {
                'genes': ['CX3CL1', 'CX3CR1', 'CSF1', 'CSF1R', 'TREM2',
                          'TYROBP', 'CD47', 'SIRPA', 'C1QA', 'C1QB',
                          'C3', 'C3AR1', 'IL34'],
                'weight': 0.85,
            },
            'semaphorin/axon_guidance': {
                'genes': ['SEMA3A', 'SEMA3B', 'SEMA3C', 'SEMA3D', 'SEMA3E',
                          'SEMA3F', 'SEMA4D', 'SEMA6A', 'NRP1', 'NRP2',
                          'PLXNA1', 'PLXNA2', 'PLXND1'],
                'weight': 0.8,
            },
            'ephrin/synapse_formation': {
                'genes': ['EFNA1', 'EFNA2', 'EFNA3', 'EFNA5', 'EFNB1',
                          'EFNB2', 'EFNB3', 'EPHA1', 'EPHA2', 'EPHA3',
                          'EPHA4', 'EPHB1', 'EPHB2', 'EPHB3'],
                'weight': 0.75,
            },
        },
    },
    'carraro': {
        'disease': 'cystic fibrosis',
        'pathways': {
            'inflammatory/CF': {
                'genes': ['TNF', 'TNFRSF1A', 'TNFRSF1B', 'IL1B', 'IL1R1',
                          'IL6', 'IL6R', 'IL6ST', 'IL8', 'CXCL8',
                          'CXCR1', 'CXCR2', 'IL17A', 'IL17RA'],
                'weight': 1.0,
            },
            'TGF-beta/airway_remodeling': {
                'genes': ['TGFB1', 'TGFB2', 'TGFBR1', 'TGFBR2',
                          'BMP2', 'BMP4', 'BMPR1A', 'BMPR2'],
                'weight': 0.9,
            },
            'mucin/epithelial': {
                'genes': ['MUC5AC', 'MUC5B', 'EGFR', 'EGF', 'ERBB2',
                          'ERBB3', 'HB-EGF', 'HBEGF', 'AREG', 'TGFA'],
                'weight': 0.85,
            },
            'WNT/regeneration': {
                'genes': ['WNT3A', 'WNT5A', 'WNT7A', 'WNT7B',
                          'FZD1', 'FZD7', 'LRP5', 'LRP6'],
                'weight': 0.7,
            },
            'neutrophil_chemotaxis': {
                'genes': ['CXCL1', 'CXCL2', 'CXCL3', 'CXCL5', 'CCL2',
                          'CCL3', 'CCL4', 'CCL5', 'CCR1', 'CCR2', 'CCR5'],
                'weight': 0.8,
            },
        },
    },
    'reichart': {
        'disease': 'dilated cardiomyopathy',
        'pathways': {
            'TGF-beta/cardiac_fibrosis': {
                'genes': ['TGFB1', 'TGFB2', 'TGFB3', 'TGFBR1', 'TGFBR2',
                          'ACVR1', 'ACVR2A', 'BMP2', 'BMP4', 'BMPR2'],
                'weight': 1.0,
            },
            'neuregulin/cardiomyocyte': {
                'genes': ['NRG1', 'NRG2', 'NRG3', 'NRG4', 'ERBB2',
                          'ERBB3', 'ERBB4', 'EGFR'],
                'weight': 0.9,
            },
            'adrenergic/contractility': {
                'genes': ['EDN1', 'EDNRA', 'EDNRB', 'AGT', 'AGTR1',
                          'AGTR2', 'ACE', 'ACE2'],
                'weight': 0.85,
            },
            'inflammatory': {
                'genes': ['TNF', 'TNFRSF1A', 'IL1B', 'IL1R1', 'IL6',
                          'IL6R', 'IL6ST', 'CXCL12', 'CXCR4'],
                'weight': 0.8,
            },
            'WNT/remodeling': {
                'genes': ['WNT3A', 'WNT5A', 'WNT11', 'FZD1', 'FZD2',
                          'FZD4', 'FZD7', 'LRP5', 'LRP6', 'SFRP1'],
                'weight': 0.7,
            },
            'Notch/cardiac_differentiation': {
                'genes': ['DLL1', 'DLL4', 'JAG1', 'JAG2', 'NOTCH1',
                          'NOTCH2', 'NOTCH3'],
                'weight': 0.7,
            },
        },
    },
}


# ============================================================================
# 权重计算
# ============================================================================

def compute_lr_weights(dataset_name, lr_pairs=None):
    """计算每个 LR 对的疾病相关性权重。

    对每个 (lig, rec) 对，检查是否匹配疾病通路中的基因：
    - 两个基因都在同一通路 → 取该通路的 weight
    - 一个基因在通路中 → weight × 0.5
    - 都不在 → 0.0（中性，不调制）

    多通路取最大匹配。返回 dict: {(lig, rec): float ∈ [0, 1]}
    """
    if dataset_name not in _DISEASE_KNOWLEDGE:
        print(f"  警告: {dataset_name} 无疾病知识库，跳过 LLM 权重")
        return {}
    knowledge = _DISEASE_KNOWLEDGE[dataset_name]
    # 构建基因→(通路, 权重)索引
    gene_pathway_map = {}
    for pathway_name, info in knowledge['pathways'].items():
        for gene in info['genes']:
            gene_upper = gene.upper()
            if gene_upper not in gene_pathway_map:
                gene_pathway_map[gene_upper] = []
            gene_pathway_map[gene_upper].append(
                (pathway_name, info['weight'])
            )
    if lr_pairs is None:
        import liana as li
        resource = li.resource.select_resource('consensus')
        lr_pairs = list(zip(resource['ligand'], resource['receptor']))
    weights = {}
    for lig, rec in lr_pairs:
        lig_u, rec_u = lig.upper(), rec.upper()
        lig_pathways = gene_pathway_map.get(lig_u, [])
        rec_pathways = gene_pathway_map.get(rec_u, [])
        if not lig_pathways and not rec_pathways:
            weights[(lig, rec)] = 0.0
            continue
        # 检查同通路匹配（最强信号）
        lig_pw_names = {pw for pw, _ in lig_pathways}
        rec_pw_names = {pw for pw, _ in rec_pathways}
        shared = lig_pw_names & rec_pw_names
        if shared:
            max_w = max(w for pw, w in lig_pathways + rec_pathways
                        if pw in shared)
            weights[(lig, rec)] = max_w
        else:
            # 单侧匹配 → 半权
            all_w = [w for _, w in lig_pathways + rec_pathways]
            weights[(lig, rec)] = max(all_w) * 0.5
    return weights


def save_weights(dataset_name, output_dir=None):
    """计算并保存权重到 JSON 文件。"""
    if output_dir is None:
        output_dir = _DATA_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    weights = compute_lr_weights(dataset_name)
    # 转为可序列化格式
    serializable = {f"{k[0]}|{k[1]}": v for k, v in weights.items()}
    out_path = output_dir / f'disease_lr_weights_{dataset_name}.json'
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    n_nonzero = sum(1 for v in weights.values() if v > 0)
    print(f"  {dataset_name}: {n_nonzero}/{len(weights)} 个 LR 对有疾病相关性权重"
          f" → {out_path}")
    return weights


def load_weights(dataset_name, data_dir=None):
    """加载预计算的权重。返回 dict {(lig, rec): float}。"""
    if data_dir is None:
        data_dir = _DATA_DIR
    path = Path(data_dir) / f'disease_lr_weights_{dataset_name}.json'
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {tuple(k.split('|')): v for k, v in raw.items()}


# ============================================================================
# 统计
# ============================================================================

def print_coverage_stats(dataset_name):
    """打印权重覆盖统计。"""
    weights = compute_lr_weights(dataset_name)
    total = len(weights)
    nonzero = sum(1 for v in weights.values() if v > 0)
    full_match = sum(1 for v in weights.values() if v >= 0.6)
    half_match = sum(1 for v in weights.values() if 0 < v < 0.6)
    print(f"\n=== {dataset_name} LR 疾病权重统计 ===")
    print(f"  总 LR 对: {total}")
    print(f"  有权重 (>0): {nonzero} ({100*nonzero/total:.1f}%)")
    print(f"  强匹配 (>=0.6): {full_match} ({100*full_match/total:.1f}%)")
    print(f"  弱匹配 (<0.6): {half_match} ({100*half_match/total:.1f}%)")
    # 打印 top LR 对
    sorted_lr = sorted(weights.items(), key=lambda x: -x[1])[:10]
    print(f"  Top 10 LR 对:")
    for (lig, rec), w in sorted_lr:
        print(f"    {lig:>10} - {rec:<10}  weight={w:.2f}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='生成疾病特异性 LR 权重')
    parser.add_argument('--dataset', default=None,
                        help='数据集名称（默认全部 5 个）')
    args = parser.parse_args()
    datasets = ([args.dataset] if args.dataset
                else list(_DISEASE_KNOWLEDGE.keys()))
    for ds in datasets:
        print_coverage_stats(ds)
        save_weights(ds)


if __name__ == '__main__':
    main()
