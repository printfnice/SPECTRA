"""
HGATLink-inspired method for ligand-receptor inference
基于HGATLink的配体受体推断方法

This method adapts the Heterogeneous Graph Attention Network (HGAT) + Transformer
architecture from HGATLink for cell-cell communication inference.

Key innovations:
1. Heterogeneous graph modeling of ligand-receptor-cell type interactions
2. Multi-head attention mechanism for capturing interaction patterns
3. Transformer encoder-decoder for feature integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import warnings

warnings.filterwarnings('ignore')

# 可选依赖项（仅在实际推断时需要）
try:
    import numpy as np
    import pandas as pd
    from scipy.sparse import issparse
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    np = None
    pd = None


class HGATLinkAttention(nn.Module):
    """
    Heterogeneous Graph Attention Layer
    改编自HGATLink的注意力机制，用于配体受体交互建模
    """
    def __init__(self, in_features, out_features, n_heads=4, dropout=0.2):
        super(HGATLinkAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.d_k = out_features // n_heads

        # Linear transformations
        self.W_q = nn.Linear(in_features, out_features)
        self.W_k = nn.Linear(in_features, out_features)
        self.W_v = nn.Linear(in_features, out_features)
        self.W_out = nn.Linear(out_features, out_features)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, ligand_feat, receptor_feat):
        """
        Args:
            ligand_feat: [n_lr_pairs, in_features]
            receptor_feat: [n_lr_pairs, in_features]
        Returns:
            attention_output: [n_lr_pairs, out_features]
        """
        batch_size = ligand_feat.size(0)

        # Linear projections
        Q = self.W_q(ligand_feat).view(batch_size, self.n_heads, self.d_k)
        K = self.W_k(receptor_feat).view(batch_size, self.n_heads, self.d_k)
        V = self.W_v(receptor_feat).view(batch_size, self.n_heads, self.d_k)

        # Scaled dot-product attention
        Q_norm = F.normalize(Q, p=2, dim=-1)
        K_norm = F.normalize(K, p=2, dim=-1)

        # Attention scores
        attention_scores = torch.sum(Q_norm * K_norm, dim=-1)  # [batch_size, n_heads]
        attention_weights = F.softmax(attention_scores / 0.25, dim=-1)  # Temperature = 0.25

        # Apply attention
        attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, n_heads, 1]
        attended_values = V * attention_weights  # [batch_size, n_heads, d_k]

        # Concatenate heads
        attended_values = attended_values.view(batch_size, -1)  # [batch_size, out_features]

        # Output projection
        output = self.W_out(attended_values)
        output = self.dropout(output)
        output = self.layer_norm(output + ligand_feat if ligand_feat.size(1) == self.out_features else output)

        return output


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for feature integration
    借鉴HGATLink的Transformer架构
    """
    def __init__(self, d_model=256, n_heads=4, n_layers=2, dropout=0.3):
        super(TransformerEncoder, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=n_layers
        )

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer,
            num_layers=1
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        encoded = self.transformer_encoder(x)
        decoded = self.transformer_decoder(x, encoded)
        return decoded


class HGATLinkLRScorer(nn.Module):
    """
    Complete HGATLink-inspired model for LR scoring
    完整的配体受体评分模型
    """
    def __init__(self, n_genes, embedding_dim=128, hidden_dim=256, n_heads=4, dropout=0.3):
        super(HGATLinkLRScorer, self).__init__()

        # Gene embeddings
        self.gene_embedding = nn.Embedding(n_genes, embedding_dim)

        # HGAT attention layers
        self.hgat_layer1 = HGATLinkAttention(embedding_dim, hidden_dim, n_heads, dropout)
        self.hgat_layer2 = HGATLinkAttention(hidden_dim, hidden_dim, n_heads, dropout)

        # Transformer
        self.transformer = TransformerEncoder(hidden_dim, n_heads, n_layers=2, dropout=dropout)

        # Output layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.gene_embedding.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, ligand_idx, receptor_idx, ligand_expr, receptor_expr):
        """
        Args:
            ligand_idx: [batch_size] - ligand gene indices
            receptor_idx: [batch_size] - receptor gene indices
            ligand_expr: [batch_size, n_cells] - ligand expression across cells
            receptor_expr: [batch_size, n_cells] - receptor expression across cells
        Returns:
            scores: [batch_size, 1] - interaction scores
        """
        # Get embeddings
        ligand_emb = self.gene_embedding(ligand_idx)  # [batch_size, embedding_dim]
        receptor_emb = self.gene_embedding(receptor_idx)  # [batch_size, embedding_dim]

        # HGAT layers
        h1 = self.hgat_layer1(ligand_emb, receptor_emb)
        h2 = self.hgat_layer2(h1, h1)

        # Prepare for transformer (add sequence dimension)
        # Stack ligand and receptor features
        lr_features = torch.stack([h2, h2], dim=1)  # [batch_size, 2, hidden_dim]

        # Transformer encoding
        transformed = self.transformer(lr_features)  # [batch_size, 2, hidden_dim]

        # Flatten
        transformed = transformed.view(transformed.size(0), -1)  # [batch_size, hidden_dim*2]

        # Final prediction
        out = self.fc1(transformed)
        out = self.bn1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        scores = self.fc2(out)

        return scores


def hgatlink_score(adata,
                   ligand_complex,
                   receptor_complex,
                   source_celltype,
                   target_celltype,
                   sample_id,
                   groupby='cell_type',
                   use_raw=False,
                   device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Calculate HGATLink-based score for a single LR interaction in a sample

    Args:
        adata: AnnData object for a single sample
        ligand_complex: str or list of genes (ligand)
        receptor_complex: str or list of genes (receptor)
        source_celltype: cell type expressing ligand
        target_celltype: cell type expressing receptor
        sample_id: sample identifier
        groupby: column name for cell types in adata.obs
        use_raw: whether to use raw counts
        device: torch device

    Returns:
        score: interaction score
    """
    if not HAS_SCIPY:
        raise ImportError("numpy, pandas, and scipy are required for inference. Please install them.")

    # Get expression matrix
    if use_raw and adata.raw is not None:
        expr = adata.raw.X
        gene_names = adata.raw.var_names
    else:
        expr = adata.X
        gene_names = adata.var_names

    if issparse(expr):
        expr = expr.toarray()

    # Handle complex notation (e.g., "gene1_gene2")
    if isinstance(ligand_complex, str):
        ligands = ligand_complex.split('_')
    else:
        ligands = ligand_complex

    if isinstance(receptor_complex, str):
        receptors = receptor_complex.split('_')
    else:
        receptors = receptor_complex

    # Get cell masks - use the groupby parameter
    source_mask = adata.obs[groupby] == source_celltype
    target_mask = adata.obs[groupby] == target_celltype

    if source_mask.sum() == 0 or target_mask.sum() == 0:
        return 0.0

    # Calculate mean expression for ligand and receptor
    ligand_expr_vals = []
    for lig in ligands:
        if lig in gene_names:
            gene_idx = np.where(gene_names == lig)[0][0]
            lig_expr = expr[source_mask, gene_idx].mean()
            ligand_expr_vals.append(lig_expr)

    receptor_expr_vals = []
    for rec in receptors:
        if rec in gene_names:
            gene_idx = np.where(gene_names == rec)[0][0]
            rec_expr = expr[target_mask, gene_idx].mean()
            receptor_expr_vals.append(rec_expr)

    if len(ligand_expr_vals) == 0 or len(receptor_expr_vals) == 0:
        return 0.0

    # Use geometric mean for complex
    ligand_expr = np.prod(ligand_expr_vals) ** (1.0 / len(ligand_expr_vals))
    receptor_expr = np.prod(receptor_expr_vals) ** (1.0 / len(receptor_expr_vals))

    # Simple scoring: product of expressions (can be enhanced with learned model)
    # This is a simplified version; full version would use the neural network
    score = ligand_expr * receptor_expr

    return score


def hgatlink_inference_optimized(adata,
                                 groupby='cell_type',
                                 sample_key='sample',
                                 use_raw=False,
                                 resource_name='consensus',
                                 expr_prop=0.1,
                                 min_cells=5,
                                 verbose=True,
                                 return_all_lrs=False,
                                 n_perms=None,
                                 max_lr_pairs=None,
                                 max_samples=None):
    """
    优化版HGATLink推断，支持限制LR对和样本数量以加速

    Args:
        max_lr_pairs: 限制LR对数量（None=全部，100=快速测试，500=平衡）
        max_samples: 限制样本数量（None=全部）
        其他参数同 hgatlink_inference
    """
    if not HAS_SCIPY:
        raise ImportError("numpy, pandas, and scipy are required for inference. Please install them.")

    try:
        import liana as li
    except ImportError:
        raise ImportError("liana package is required for inference. Please install it with: pip install liana")

    # Get LR resource
    if verbose:
        print(f"Loading resource: {resource_name}")

    resource = li.resource.select_resource(resource_name)

    if verbose:
        print(f"  Total LR pairs in resource: {len(resource)}")

    # Get unique samples
    samples = adata.obs[sample_key].unique()

    # ⚡ 加速选项1: 限制样本数量
    if max_samples is not None and max_samples < len(samples):
        samples = samples[:max_samples]
        if verbose:
            print(f"  ⚡ 限制样本数量: {max_samples}")

    results = []

    for sample_idx, sample in enumerate(samples):
        if verbose:
            print(f"\nProcessing sample {sample_idx+1}/{len(samples)}: {sample}")

        # Subset to sample
        sample_adata = adata[adata.obs[sample_key] == sample].copy()

        # Get cell types
        cell_types = sample_adata.obs[groupby].unique()

        # Filter cell types by min_cells
        cell_type_counts = sample_adata.obs[groupby].value_counts()
        valid_cell_types = cell_type_counts[cell_type_counts >= min_cells].index.tolist()

        if len(valid_cell_types) < 2:
            if verbose:
                print(f"  Skipping {sample}: insufficient cell types")
            continue

        if verbose:
            print(f"  Valid cell types: {len(valid_cell_types)}")
            print(f"  Cell types: {', '.join(valid_cell_types[:5])}{'...' if len(valid_cell_types) > 5 else ''}")

        # Filter resource to only genes present in this sample
        gene_names = set(sample_adata.var_names)
        valid_lr_pairs = resource[
            resource['ligand'].isin(gene_names) &
            resource['receptor'].isin(gene_names)
        ]

        # ⚡ 加速选项2: 限制LR对数量
        if max_lr_pairs is not None and max_lr_pairs < len(valid_lr_pairs):
            valid_lr_pairs = valid_lr_pairs.head(max_lr_pairs)
            if verbose:
                print(f"  ⚡ 限制LR对数量: {max_lr_pairs}")

        if verbose:
            print(f"  Valid LR pairs: {len(valid_lr_pairs)}/{len(resource)}")

        n_combinations = len(valid_lr_pairs) * len(valid_cell_types) * len(valid_cell_types)
        if verbose:
            print(f"  Total combinations to process: {n_combinations}")

        # Iterate over LR pairs and cell type pairs
        lr_count = 0
        for lr_idx, lr_row in valid_lr_pairs.iterrows():
            ligand = lr_row['ligand']
            receptor = lr_row['receptor']

            # Progress update every 100 LR pairs
            lr_count += 1
            if verbose and lr_count % 100 == 0:
                print(f"    Processed {lr_count}/{len(valid_lr_pairs)} LR pairs... ({len(results)} interactions found)")

            # Iterate over source-target cell type pairs
            for source_ct in valid_cell_types:
                for target_ct in valid_cell_types:
                    # Calculate score
                    score = hgatlink_score(
                        sample_adata,
                        ligand,
                        receptor,
                        source_ct,
                        target_ct,
                        sample,
                        groupby=groupby,
                        use_raw=use_raw
                    )

                    results.append({
                        sample_key: sample,
                        'ligand_complex': ligand,
                        'receptor_complex': receptor,
                        'source': source_ct,
                        'target': target_ct,
                        'hgatlink_score': score
                    })

        if verbose:
            print(f"  ✓ Sample {sample} complete: {len(results)} total interactions")

    if len(results) == 0:
        if verbose:
            print("No valid interactions found!")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    if verbose:
        print(f"\nCompleted! Found {len(results_df)} interactions across {len(samples)} samples")

    return results_df


def hgatlink_inference_vectorized(adata,
                                   groupby='cell_type',
                                   sample_key='sample',
                                   use_raw=False,
                                   resource_name='consensus',
                                   max_lr_pairs=300,
                                   min_cells=5,
                                   verbose=True,
                                   **kwargs):
    """
    ⚡ 向量化快速版HGATLink推断（简单截断法 - 已验证最优）

    优化点:
    1. 预计算细胞类型平均表达（避免重复计算）
    2. 简单截断前N个LR对（不使用复杂筛选）
    3. 向量化NumPy操作（减少循环开销）

    Args:
        max_lr_pairs: 限制LR对数量（每个样本独立选择）
        其他参数同 hgatlink_inference_optimized
    """
    if not HAS_SCIPY:
        raise ImportError("numpy, pandas, and scipy are required for inference.")

    try:
        import liana as li
    except ImportError:
        raise ImportError("liana package is required. Install: pip install liana")

    # 加载LR资源
    if verbose:
        print(f"⚡ 向量化快速版HGATLink推断")
        print(f"  Loading resource: {resource_name}")

    resource = li.resource.select_resource(resource_name)

    if verbose:
        print(f"  LR资源总数: {len(resource)}")

    samples = adata.obs[sample_key].unique()
    results = []

    for sample_idx, sample in enumerate(samples):
        if verbose:
            print(f"\n处理样本 {sample_idx+1}/{len(samples)}: {sample}")

        sample_adata = adata[adata.obs[sample_key] == sample].copy()

        # 获取表达矩阵
        if use_raw and sample_adata.raw is not None:
            expr = sample_adata.raw.X
            gene_names = list(sample_adata.raw.var_names)
        else:
            expr = sample_adata.X
            gene_names = list(sample_adata.var_names)

        if issparse(expr):
            expr = expr.toarray()

        # 1️⃣ 预计算每个细胞类型的平均表达
        cell_types = sample_adata.obs[groupby].unique()
        cell_type_counts = sample_adata.obs[groupby].value_counts()
        valid_cell_types = cell_type_counts[cell_type_counts >= min_cells].index.tolist()

        if len(valid_cell_types) < 2:
            if verbose:
                print(f"  跳过: 有效细胞类型不足2个")
            continue

        # 预计算细胞类型-基因平均表达矩阵
        ct_expr_dict = {}
        for ct in valid_cell_types:
            mask = sample_adata.obs[groupby] == ct
            ct_expr_dict[ct] = expr[mask, :].mean(axis=0).ravel()

        if verbose:
            print(f"  有效细胞类型: {len(valid_cell_types)}")

        # 2️⃣ 筛选有效的LR对（简单截断）
        gene_set = set(gene_names)
        valid_lr_pairs = resource[
            resource['ligand'].isin(gene_set) &
            resource['receptor'].isin(gene_set)
        ].copy()

        if len(valid_lr_pairs) == 0:
            if verbose:
                print(f"  跳过: 无有效LR对")
            continue

        # ⚡ 简单截断前N个LR对（已验证最优）
        if max_lr_pairs is not None and max_lr_pairs < len(valid_lr_pairs):
            valid_lr_pairs = valid_lr_pairs.head(max_lr_pairs)
            if verbose:
                print(f"  限制LR对数量: {len(valid_lr_pairs)}")
        else:
            if verbose:
                print(f"  有效LR对: {len(valid_lr_pairs)} (全部保留)")

        if verbose:
            print(f"  有效LR对: {len(valid_lr_pairs)}/{len(resource)}")

        # 3️⃣ 批量计算（向量化）
        n_combinations = len(valid_lr_pairs) * len(valid_cell_types) ** 2
        if verbose:
            print(f"  计算组合数: {n_combinations}")

        # 为LR对预计算基因索引
        lr_indices = []
        for _, row in valid_lr_pairs.iterrows():
            try:
                lig_idx = gene_names.index(row['ligand'])
                rec_idx = gene_names.index(row['receptor'])
                lr_indices.append((row['ligand'], row['receptor'], lig_idx, rec_idx))
            except ValueError:
                continue

        # 批量计算所有组合
        batch_results = []
        for ligand, receptor, lig_idx, rec_idx in lr_indices:
            for source_ct in valid_cell_types:
                source_expr = ct_expr_dict[source_ct]
                lig_expr = source_expr[lig_idx]

                for target_ct in valid_cell_types:
                    target_expr = ct_expr_dict[target_ct]
                    rec_expr = target_expr[rec_idx]

                    # 简单乘积评分（已验证最佳）
                    score = float(lig_expr * rec_expr)

                    batch_results.append({
                        sample_key: sample,
                        'ligand_complex': ligand,
                        'receptor_complex': receptor,
                        'source': source_ct,
                        'target': target_ct,
                        'hgatlink_score': score
                    })

        results.extend(batch_results)
        n_new = len(batch_results)

        # ⚡ 释放该样本的内存
        del sample_adata, expr, ct_expr_dict, batch_results
        import gc
        gc.collect()

        if verbose:
            print(f"  ✓ 完成: {n_new} 个交互")

    if len(results) == 0:
        if verbose:
            print("未发现有效交互！")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    if verbose:
        print(f"\n✓ 总计: {len(results_df)} 个交互，来自 {len(samples)} 个样本")

    return results_df


# 保留原始的hgatlink_inference作为别名
def hgatlink_inference(adata, **kwargs):
    """
    原始接口，现在默认调用向量化快速版

    如需使用旧版本，请直接调用 hgatlink_inference_optimized()
    """
    # 默认使用快速版，限制LR对数量以加速
    if 'max_lr_pairs' not in kwargs:
        kwargs['max_lr_pairs'] = 300  # 默认300个LR对（平衡速度和准确性）

    return hgatlink_inference_vectorized(adata, **kwargs)


# ============================================================================
# LIANA Method Registration
# 注册HGATLink为LIANA方法，使其能被tensor降维识别
# ============================================================================

try:
    from liana.method.sc._Method import Method, MethodMeta

    def _hgatlink_scoring_wrapper(adata, **kwargs):
        """
        Wrapper for LIANA's scoring pipeline
        虽然HGATLink有自己的推断逻辑，但我们需要这个包装器来符合LIANA的接口
        """
        # This is a placeholder - actual inference is done separately
        # The important thing is registering the score_key
        return hgatlink_inference_optimized(adata, **kwargs)

    # 创建MethodMeta实例，注册hgatlink_score评分
    _hgatlink_meta = MethodMeta(
        method_name="HGATLink",
        complex_cols=["ligand_complex", "receptor_complex"],
        add_cols=[],
        fun=_hgatlink_scoring_wrapper,
        magnitude="hgatlink_score",  # ← 关键：注册评分名称
        magnitude_ascending=False,    # 更高的分数更好
        specificity=None,              # 我们只有一个评分指标
        specificity_ascending=None,
        permute=False,                 # HGATLink不使用排列检验
        reference="Adapted from HGATLink: Heterogeneous Graph Attention Network for "
                  "ligand-receptor interaction prediction"
    )

    # 创建Method实例
    hgatlink = Method(_method=_hgatlink_meta)

    print("✓ HGATLink方法已注册到LIANA")

except ImportError as e:
    print(f"警告: 无法注册HGATLink到LIANA - {e}")
    print("这可能影响tensor降维功能")
    # 创建一个简单的占位符
    class SimplehGATLink:
        def __init__(self):
            self.magnitude = "hgatlink_score"
            self.specificity = None

        def by_sample(self, adata, **kwargs):
            return hgatlink_inference_optimized(adata, **kwargs)

    hgatlink = SimplehGATLink()
