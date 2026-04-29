# ms_hgatlink_encoders.py
# 依赖: torch, torch.nn, torch.nn.functional
# 被依赖: ms_hgatlink_method.py
# 职责: MS-HGATLink 三层编码器和融合层的 PyTorch 实现

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Gene-Level Encoder
# ============================================================================

class GeneEncoder(nn.Module):
    """基因级编码器：学习基因功能嵌入并编码表达值。

    先验融合模式：
    - ESM-2（残差加法，零初始化 adapter，ReZero 风格）
    - GenePT（门控融合，gate=sigmoid(0)=0.5 起步，避免共线性）
    """

    def __init__(self, n_genes, gene_emb_dim=64, expr_hidden_dim=128, dropout=0.2,
                 esm2_features=None, genept_features=None):
        super().__init__()
        self.gene_embedding = nn.Embedding(n_genes, gene_emb_dim)
        self.expr_encoder = nn.Sequential(
            nn.Linear(1, expr_hidden_dim),
            nn.LayerNorm(expr_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expr_hidden_dim, gene_emb_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(gene_emb_dim * 2, expr_hidden_dim),
            nn.LayerNorm(expr_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expr_hidden_dim, gene_emb_dim)
        )
        # ESM-2 蛋白质先验（残差加法，零初始化 adapter）
        self.esm2_table = None
        if esm2_features is not None:
            self.esm2_table = nn.Embedding.from_pretrained(esm2_features, freeze=True)
            self.esm2_adapter = nn.Linear(esm2_features.shape[1], gene_emb_dim)
            nn.init.zeros_(self.esm2_adapter.weight)
            nn.init.zeros_(self.esm2_adapter.bias)
        # GenePT 先验（门控融合，独立梯度路径）
        self.genept_table = None
        if genept_features is not None:
            self.genept_table = nn.Embedding.from_pretrained(genept_features, freeze=True)
            self.genept_proj = nn.Linear(genept_features.shape[1], gene_emb_dim)
            self.genept_gate = nn.Linear(gene_emb_dim * 2, gene_emb_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.gene_embedding.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # ESM-2 adapter 重置为零（_init_weights Xavier 会覆盖）
        if self.esm2_table is not None:
            nn.init.zeros_(self.esm2_adapter.weight)
            nn.init.zeros_(self.esm2_adapter.bias)
        # GenePT gate bias=0 → sigmoid(0)=0.5（初始 50/50 混合）
        if self.genept_table is not None:
            nn.init.zeros_(self.genept_gate.bias)

    def forward(self, gene_indices, gene_expressions):
        gene_emb = self.gene_embedding(gene_indices)
        if self.esm2_table is not None:
            esm2_prior = self.esm2_adapter(self.esm2_table(gene_indices))
            gene_emb = gene_emb + esm2_prior
        if self.genept_table is not None:
            gene_emb = self._genept_gated_fusion(gene_emb, gene_indices)
        expr_emb = self.expr_encoder(gene_expressions)
        combined = torch.cat([gene_emb, expr_emb], dim=-1)
        return self.fusion(combined)

    def _genept_gated_fusion(self, free_emb, gene_indices):
        """门控融合：gate * free_emb + (1-gate) * genept_proj。
        gate 由两个嵌入的拼接决定，per-dimension 独立门控。"""
        prior = self.genept_proj(self.genept_table(gene_indices))
        gate_input = torch.cat([free_emb, prior], dim=-1)
        gate = torch.sigmoid(self.genept_gate(gate_input))
        return gate * free_emb + (1 - gate) * prior


# ============================================================================
# Pathway-Level Encoder
# ============================================================================

class PathwayEncoder(nn.Module):
    """通路级编码器：用多头注意力建模配体-受体相互作用。

    可选 KGE 先验：将 OmniPath 知识图谱嵌入作为残差注入，
    提供通路关系结构先验（零初始化 adapter，不干扰 baseline）。
    """

    def __init__(self, gene_emb_dim=64, pathway_dim=64, n_heads=4, dropout=0.3,
                 kg_dim=0, n_pathways=0):
        super().__init__()
        self.temperature = 0.5
        self.attention = nn.MultiheadAttention(
            embed_dim=gene_emb_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(gene_emb_dim, pathway_dim * 2),
            nn.LayerNorm(pathway_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pathway_dim * 2, pathway_dim)
        )
        self.layer_norm = nn.LayerNorm(pathway_dim)
        # KGE 先验（可选，kg_dim > 0 时启用，零初始化保留 baseline）
        self.kg_adapter = None
        if kg_dim > 0:
            self.kg_adapter = nn.Linear(kg_dim, pathway_dim)
            nn.init.zeros_(self.kg_adapter.weight)
            nn.init.zeros_(self.kg_adapter.bias)
        # CellChat 通路先验（可选，n_pathways>0 时启用，零初始化保留 baseline）
        self.pathway_prior = None
        if n_pathways > 0:
            self.pathway_prior = nn.Embedding(n_pathways + 1, pathway_dim, padding_idx=0)
            nn.init.zeros_(self.pathway_prior.weight)

    def forward(self, ligand_features, receptor_features, coexpr_scores=None,
                kg_features=None, pathway_idx=None):
        lr_seq = torch.stack([ligand_features, receptor_features], dim=1)
        lr_seq_norm = F.normalize(lr_seq, p=2, dim=-1)
        attended, _ = self.attention(lr_seq_norm, lr_seq_norm, lr_seq)
        attended = attended / self.temperature
        if coexpr_scores is not None:
            attended = attended * coexpr_scores.unsqueeze(-1).unsqueeze(-1)
        aggregated = attended.mean(dim=1)
        context = self.layer_norm(self.context_encoder(aggregated))
        if kg_features is not None and self.kg_adapter is not None:
            context = context + self.kg_adapter(kg_features)
        if pathway_idx is not None and self.pathway_prior is not None:
            context = context + self.pathway_prior(pathway_idx)
        return context


# ============================================================================
# Cell-Type-Level Encoder
# ============================================================================

class CellTypeEncoder(nn.Module):
    """细胞类型级编码器：建模细胞类型对特异性通讯模式"""

    def __init__(self, n_cell_types, ct_emb_dim=64, dropout=0.2):
        super().__init__()
        self.ct_embedding = nn.Embedding(n_cell_types, ct_emb_dim)
        self.pair_encoder = nn.Sequential(
            nn.Linear(ct_emb_dim * 2, ct_emb_dim * 2),
            nn.LayerNorm(ct_emb_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ct_emb_dim * 2, ct_emb_dim)
        )
        self.affinity_scorer = nn.Sequential(
            nn.Linear(ct_emb_dim, ct_emb_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ct_emb_dim // 2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ct_embedding.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, source_ct_indices, target_ct_indices):
        src = self.ct_embedding(source_ct_indices)
        tgt = self.ct_embedding(target_ct_indices)
        ct_features = self.pair_encoder(torch.cat([src, tgt], dim=-1))
        affinity = self.affinity_scorer(ct_features)
        return ct_features, affinity


# ============================================================================
# Multi-Scale Fusion Layer
# ============================================================================

class MultiScaleFusionLayer(nn.Module):
    """自适应多尺度融合层：学习基因/通路/细胞类型三个尺度的权重"""

    def __init__(self, gene_dim=64, pathway_dim=64, ct_dim=64,
                 fusion_dim=128, n_heads=4, dropout=0.3):
        super().__init__()
        self.gene_proj = nn.Linear(gene_dim * 2, fusion_dim)
        self.pathway_proj = nn.Linear(pathway_dim, fusion_dim)
        self.ct_proj = nn.Linear(ct_dim, fusion_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.gate_net = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 3),
            nn.Softmax(dim=-1)
        )
        self.residual_proj = nn.Linear(fusion_dim * 3, fusion_dim)
        self.layer_norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ligand_gene_feat, receptor_gene_feat, pathway_feat, ct_feat):
        gene_feat = torch.cat([ligand_gene_feat, receptor_gene_feat], dim=-1)
        gene_proj = self.gene_proj(gene_feat)
        pathway_proj = self.pathway_proj(pathway_feat)
        ct_proj = self.ct_proj(ct_feat)

        # v2 消融：在 cross-attention 前 zero out 非目标尺度，彻底隔断残差信息
        if getattr(self, 'scale_mode_v2', None) is not None:
            _SMv2 = {'gene_only': (True, False, False),
                     'pathway_only': (False, True, False),
                     'ct_only': (False, False, True)}
            keep = _SMv2[self.scale_mode_v2]
            if not keep[0]:
                gene_proj = torch.zeros_like(gene_proj)
            if not keep[1]:
                pathway_proj = torch.zeros_like(pathway_proj)
            if not keep[2]:
                ct_proj = torch.zeros_like(ct_proj)

        multi_scale = torch.stack([gene_proj, pathway_proj, ct_proj], dim=1)
        attended, _ = self.cross_attention(multi_scale, multi_scale, multi_scale)

        all_flat = attended.reshape(attended.size(0), -1)
        scale_weights = self.gate_net(all_flat)

        # scale_mode: None=learned, 'gene_only'=[1,0,0], 'pathway_only'=[0,1,0], 'ct_only'=[0,0,1]
        # v1: 仅覆盖 scale_weights（残差路径仍泄露信息，用于 scale_ablation/）
        # v2: 在 multi_scale stack 之前已 zero 掉非目标 proj（用于 scale_ablation_v2/）
        if getattr(self, 'scale_mode', None) is not None:
            _SM = {'gene_only': [1., 0., 0.], 'pathway_only': [0., 1., 0.], 'ct_only': [0., 0., 1.]}
            w = torch.tensor(_SM[self.scale_mode], dtype=all_flat.dtype, device=all_flat.device)
            scale_weights = w.unsqueeze(0).expand(all_flat.size(0), -1)

        weighted = attended * scale_weights.unsqueeze(-1)
        fused = weighted.sum(dim=1) + self.residual_proj(all_flat)
        return self.layer_norm(self.dropout(fused)), scale_weights


# ============================================================================
# Complete MS-HGATLink Model
# ============================================================================

class MSHGATLink(nn.Module):
    """
    Multi-Scale Heterogeneous Graph Attention Network.

    三层编码架构：基因级 → 通路级 → 细胞类型级 → 自适应融合 → 评分
    """

    def __init__(self, n_genes, n_cell_types, gene_emb_dim=64,
                 pathway_dim=64, ct_dim=64, fusion_dim=128,
                 n_heads=4, dropout=0.3, esm2_features=None, kge_features=None,
                 genept_features=None, n_pathways=0):
        super().__init__()
        self.gene_encoder = GeneEncoder(n_genes, gene_emb_dim, dropout=dropout,
                                        esm2_features=esm2_features,
                                        genept_features=genept_features)
        kg_dim = kge_features.shape[1] if kge_features is not None else 0
        self.pathway_encoder = PathwayEncoder(gene_emb_dim, pathway_dim, n_heads, dropout,
                                              kg_dim=kg_dim, n_pathways=n_pathways)
        # KGE 嵌入表（冻结，用于查询 LR pair 对应的关系嵌入）
        self.kge_table = None
        if kge_features is not None:
            self.kge_table = nn.Embedding.from_pretrained(kge_features, freeze=True)
        self.ct_encoder = CellTypeEncoder(n_cell_types, ct_dim, dropout)
        self.fusion_layer = MultiScaleFusionLayer(
            gene_emb_dim, pathway_dim, ct_dim, fusion_dim, n_heads, dropout
        )
        self.scorer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1)
        )
        # 辅助特异性头（仅预训练使用，确保三尺度均被训练）
        self.aux_gene_head = nn.Linear(gene_emb_dim * 2, 1)
        self.aux_pathway_head = nn.Linear(pathway_dim, 1)
        self.aux_ct_head = nn.Linear(ct_dim, 1)
        self._init_weights()
        self._zero_init_residual()
        self._zero_init_prior_adapters()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def _zero_init_residual(self):
        """ReZero: scorer 末层零初始化，确保未训练模型 = 精确恒等函数"""
        with torch.no_grad():
            self.scorer[-1].weight.zero_()
            self.scorer[-1].bias.zero_()

    def _zero_init_prior_adapters(self):
        """零初始化所有先验 adapter（ESM-2 / KGE / CellChat），防止 Xavier 覆盖。"""
        with torch.no_grad():
            if (hasattr(self.gene_encoder, 'esm2_table') and
                    self.gene_encoder.esm2_table is not None):
                self.gene_encoder.esm2_adapter.weight.zero_()
                self.gene_encoder.esm2_adapter.bias.zero_()
            if (hasattr(self.pathway_encoder, 'kg_adapter') and
                    self.pathway_encoder.kg_adapter is not None):
                self.pathway_encoder.kg_adapter.weight.zero_()
                self.pathway_encoder.kg_adapter.bias.zero_()
            if (hasattr(self.pathway_encoder, 'pathway_prior') and
                    self.pathway_encoder.pathway_prior is not None):
                self.pathway_encoder.pathway_prior.weight.zero_()

    def forward(self, ligand_idx, receptor_idx, ligand_expr, receptor_expr,
                source_ct_idx, target_ct_idx, expr_product=None,
                return_embeddings=False, pathway_idx=None):
        lig_feat = self.gene_encoder(ligand_idx, ligand_expr)
        rec_feat = self.gene_encoder(receptor_idx, receptor_expr)
        kg_feat = None
        if self.kge_table is not None:
            kg_feat = (self.kge_table(ligand_idx) + self.kge_table(receptor_idx)) / 2
        pathway_feat = self.pathway_encoder(lig_feat, rec_feat, kg_features=kg_feat,
                                            pathway_idx=pathway_idx)
        ct_feat, ct_affinity = self.ct_encoder(source_ct_idx, target_ct_idx)
        fused, scale_weights = self.fusion_layer(lig_feat, rec_feat, pathway_feat, ct_feat)
        modulation = self.scorer(fused) * torch.sigmoid(ct_affinity)
        # 缓存中间结果供预训练损失使用
        self._last_modulation = modulation
        self._lig_feat = lig_feat
        self._rec_feat = rec_feat
        self._pathway_feat = pathway_feat
        self._ct_feat = ct_feat
        # Tanh 调制：modulation=0 时 score=expr_product（恒等函数基线）
        if expr_product is not None:
            score = expr_product * (1 + torch.tanh(modulation))
        else:
            score = modulation
        if return_embeddings:
            return score, scale_weights, fused  # fused: (L, fusion_dim) 用于 E2E
        return score, scale_weights
