# e2e_modules.py
# 依赖: torch, torch.nn
# 被依赖: e2e_pipeline.py
# 职责: 端到端分类所需的聚合器、VIB瓶颈、DTFD伪袋、ABMIL和MLP分类头

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Multihead Attention Block（MAB）用于 PMA
# ============================================================================

class MAB(nn.Module):
    """Multihead Attention Block: Q attends to (K, V)。"""

    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )
        self.layer_norm2 = nn.LayerNorm(dim)

    def forward(self, Q, KV):
        """Q: (1, k, D)  KV: (1, L, D) → (1, k, D)"""
        out, _ = self.attn(Q, KV, KV)
        out = self.layer_norm1(Q + out)
        out = self.layer_norm2(out + self.ff(out))
        return out


# ============================================================================
# PMA：Pooling by Multihead Attention（Lee et al., ICML 2019）
# ============================================================================

class PMA(nn.Module):
    """将变长集合 Z: (L, D) 聚合为固定大小 (k, D)。"""

    def __init__(self, dim, k=30, n_heads=4, dropout=0.1):
        super().__init__()
        self.k = k
        self.seeds = nn.Parameter(torch.randn(k, dim))
        nn.init.xavier_uniform_(self.seeds.unsqueeze(0))
        self.mab = MAB(dim, n_heads, dropout)

    def forward(self, Z):
        """Z: (L, D) → (k, D)"""
        Z_b = Z.unsqueeze(0)               # (1, L, D)
        S_b = self.seeds.unsqueeze(0)       # (1, k, D)
        out = self.mab(S_b, Z_b)            # (1, k, D)
        return out.squeeze(0)               # (k, D)


# ============================================================================
# SampleAggregator：PMA + ACMIL 注意力 Dropout + 均值压缩
# ============================================================================

class SampleAggregator(nn.Module):
    """封装 PMA，将 (L, D) 压缩为 (D,) 样本嵌入。

    ACMIL 注意力 Dropout（ECCV 2024）：训练时随机丢弃 top-K% 高注意力实例，
    迫使模型从次要 LR-CT 对中学习，防止过拟合到少数显著实例。
    """

    def __init__(self, dim, k=30, n_heads=4, dropout=0.1,
                 acmil_drop_ratio=0.0, use_pathway_pool=False):
        super().__init__()
        self.pathway_pool = (PathwayPool(dim, n_heads=n_heads, dropout=dropout, k=k)
                             if use_pathway_pool else None)
        self.pma = PMA(dim, k, n_heads, dropout)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.acmil_drop_ratio = acmil_drop_ratio
        # 轻量级注意力评分器（用于 ACMIL 排序，不影响 PMA 本身）
        if acmil_drop_ratio > 0:
            self.instance_scorer = nn.Linear(dim, 1)

    def forward(self, Z, pathway_idx=None):
        """Z: (L, D) → h: (D,)"""
        if self.pathway_pool is not None and pathway_idx is not None:
            return self.pathway_pool(Z, pathway_idx)
        if self.training and self.acmil_drop_ratio > 0:
            Z = self._acmil_drop(Z)
        pma_out = self.pma(Z)       # (k, D)
        h = pma_out.mean(dim=0)     # (D,)
        return self.proj(h)          # (D,)

    def _acmil_drop(self, Z):
        """丢弃 top-K 高重要性实例（ACMIL）。"""
        scores = self.instance_scorer(Z.detach()).squeeze(-1)   # (L,)
        k_drop = max(1, int(Z.size(0) * self.acmil_drop_ratio))
        if k_drop >= Z.size(0) - 1:
            return Z
        _, top_idx = scores.topk(k_drop)
        keep_mask = torch.ones(Z.size(0), dtype=torch.bool, device=Z.device)
        keep_mask[top_idx] = False
        return Z[keep_mask]


# ============================================================================
# VIBLayer：变分信息瓶颈（Alemi et al., 2016）
# ============================================================================

class VIBLayer(nn.Module):
    """在聚合输出和分类器之间插入随机瓶颈层。

    将确定性 h 映射为 (mu, sigma)，采样 z ~ N(mu, sigma)，
    KL(q(z|x) || N(0,I)) 惩罚强制压缩，防止记忆训练样本。
    """

    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.mu_head = nn.Linear(in_dim, z_dim)
        self.logvar_head = nn.Linear(in_dim, z_dim)
        nn.init.xavier_uniform_(self.mu_head.weight)
        nn.init.xavier_uniform_(self.logvar_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.constant_(self.logvar_head.bias, -2.0)  # 初始 sigma 较小

    def forward(self, h):
        """h: (D,) 或 (B, D) → (z, kl_loss)
        训练时采样，推理时用 mu（确定性）。
        """
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return z, kl


# ============================================================================
# GatedABMIL：门控注意力 MIL 聚合（Ilse et al., ICML 2018）
# ============================================================================

class GatedABMIL(nn.Module):
    """Gated Attention MIL: 可解释的注意力加权实例聚合。

    注意力权重可导出，用于可视化哪些 LR-CT pair 驱动疾病分类。
    参数量极少（<10K），适合小样本。
    """

    def __init__(self, in_dim, hidden_dim=128, acmil_drop_ratio=0.0):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Sigmoid()
        )
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.acmil_drop_ratio = acmil_drop_ratio

    def forward(self, x):
        """x: (L, D) → (D,) weighted sum"""
        h_V = self.attention_V(x)                          # (L, H)
        h_U = self.attention_U(x)                          # (L, H)
        attn_logits = self.attention_w(h_V * h_U).squeeze(-1)  # (L,)
        # ACMIL: 训练时 mask top-K 实例
        if self.training and self.acmil_drop_ratio > 0:
            k_drop = max(1, int(len(attn_logits) * self.acmil_drop_ratio))
            if k_drop < len(attn_logits) - 1:
                _, top_idx = attn_logits.detach().topk(k_drop)
                attn_logits = attn_logits.clone()
                attn_logits[top_idx] = float('-inf')
        attn = F.softmax(attn_logits, dim=0)               # (L,)
        return (attn.unsqueeze(-1) * x).sum(0)             # (D,)

    def get_attention_weights(self, x):
        """推理时获取注意力权重（用于可解释性分析）。"""
        with torch.no_grad():
            h_V = self.attention_V(x)
            h_U = self.attention_U(x)
            logits = self.attention_w(h_V * h_U).squeeze(-1)
            return F.softmax(logits, dim=0)


# ============================================================================
# PseudoBagSplitter：DTFD 伪袋拆分（Zhang et al., CVPR 2022）
# ============================================================================

class PseudoBagSplitter:
    """将变长实例集合拆分为 K 个伪袋。

    训练时随机拆分以增加有效训练样本数（N→N*K），
    推理时不拆分（K=1，返回全量实例）。
    """

    def __init__(self, K=4):
        self.K = K

    def split(self, instances, training=True):
        """instances: (L, D) → list of K tensors, each (~L/K, D)"""
        if not training or self.K <= 1:
            return [instances]
        L = instances.size(0)
        if L < self.K * 2:  # 实例太少，不拆分
            return [instances]
        indices = torch.randperm(L, device=instances.device)
        chunks = torch.chunk(indices, self.K)
        return [instances[idx] for idx in chunks]


# ============================================================================
# E2EClassifier：MLP 分类头 + L1 正则
# ============================================================================

class E2EClassifier(nn.Module):
    """两层 MLP 分类器，适合小样本（L1 正则化防过拟合）。"""

    def __init__(self, in_dim, n_classes, dropout=0.3, l1_lambda=0.01):
        super().__init__()
        hidden = max(in_dim // 2, n_classes * 2)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )
        self.l1_lambda = l1_lambda
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h):
        """h: (D,) 或 (B, D) → logits: (n_classes,) 或 (B, n_classes)"""
        return self.mlp(h)

    def l1_loss(self):
        """L1 正则项（对第一层权重）"""
        return self.l1_lambda * self.mlp[0].weight.abs().sum()


# ============================================================================
# ScoreEncoder：标量 LR 评分 → 嵌入空间（R30 统一下游公平对比）
# ============================================================================

# ============================================================================
# PathwayPool：通路级聚合（Direction 1，模拟 CellChat 通路信号机制）
# ============================================================================

class PathwayPool(nn.Module):
    """通路内注意力池化后再做 PMA，模拟 CellChat 通路级信号聚合（可学习版）。

    CellChat 优势根因：同一通路的 LR pair 信号在通路级聚合后去噪，
    等价于对 L 个 LR pair 做了 L/P 倍的信号增强（中心极限定理）。
    本模块通过可学习注意力权重复现这一机制。
    """

    def __init__(self, dim, n_heads=4, dropout=0.1, k=30):
        super().__init__()
        self.intra_attn = nn.Linear(dim, 1)   # 通路内轻量级评分
        self.norm = nn.LayerNorm(dim)
        self.pma = PMA(dim, k=k, n_heads=n_heads, dropout=dropout)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, Z, pathway_idx):
        """Z: (L, D), pathway_idx: (L,) → (D,)

        向量化实现（scatter_add_），全程保持梯度图：
        1. 全批次计算 intra_attn scores → 通路内 softmax（数值稳定）
        2. scatter_add_ 加权求�� → 每通路向量
        3. (P, D) 通路向量 → PMA → 样本嵌入
        """
        L, D = Z.shape
        n_p = int(pathway_idx.max().item()) + 1

        # Step 1: attention scores（全批次，保持梯度）
        raw = self.intra_attn(Z).squeeze(-1)                        # (L,)

        # Step 2: 通路内 softmax（数值稳定，scatter_reduce_ amax）
        max_p = torch.full((n_p,), -1e9, device=Z.device, dtype=Z.dtype)
        max_p.scatter_reduce_(0, pathway_idx, raw, reduce='amax', include_self=True)
        exp_s = torch.exp(raw - max_p[pathway_idx])                 # (L,)
        sum_p = torch.zeros(n_p, device=Z.device, dtype=Z.dtype)
        sum_p.scatter_add_(0, pathway_idx, exp_s)
        attn = exp_s / (sum_p[pathway_idx] + 1e-8)                 # (L,)

        # Step 3: 通路内加权求和（scatter_add_ 保持梯度）
        weighted = attn.unsqueeze(-1) * Z                           # (L, D)
        path_vecs = torch.zeros(n_p, D, device=Z.device, dtype=Z.dtype)
        path_vecs.scatter_add_(0, pathway_idx.unsqueeze(-1).expand(-1, D), weighted)
        path_vecs = self.norm(path_vecs)                            # (P, D)

        # Step 4: PMA 聚合 → 样本嵌入
        return self.proj(self.pma(path_vecs).mean(0))               # (D,)


class ScoreEncoder(nn.Module):
    """将 frozen LIANA 方法的标量 LR 评分映射到与 MS-HGATLink 相同维度的嵌入空间。

    用于 R30 公平对比实验：所有方法共享 PMA+VIB+MLP 下游，
    唯一变量是 LR 评分来源（CellPhoneDB/logFC/connectome 等 vs MS-HGATLink）。
    """

    def __init__(self, n_lr_pairs, fusion_dim, dropout=0.1):
        super().__init__()
        self.lr_embed = nn.Embedding(n_lr_pairs, fusion_dim)
        self.score_proj = nn.Sequential(
            nn.Linear(1, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.lr_embed.weight)
        for m in self.score_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, scores, lr_indices):
        """scores: (L,)  lr_indices: (L,) → (L, fusion_dim)

        乘法交互：LR 身份嵌入 * 评分投影，使模型同时感知"哪个 LR 对"和"分数多高"。
        """
        emb = self.lr_embed(lr_indices)                    # (L, D)
        proj = self.score_proj(scores.unsqueeze(-1))       # (L, D)
        return emb * proj                                  # (L, D)


# ============================================================================
# SupCon Loss（Khosla et al., NeurIPS 2020）
# ============================================================================

def supcon_loss(h, labels, tau=0.07):
    """Supervised Contrastive Loss：拉近同类嵌入，推离异类嵌入。

    h: (N, D) L2-normalized 样本嵌入
    labels: (N,) 整数类别标签
    tau: 温度参数（default 0.07）
    返回标量 loss，N<4 时返回 0（避免无正对的情况）。
    """
    N = h.size(0)
    if N < 4:
        return h.new_zeros(1).squeeze()
    h = F.normalize(h, dim=-1)                              # (N, D)
    sim = torch.matmul(h, h.T) / tau                        # (N, N)
    # 去掉对角线（自相似）
    mask_self = ~torch.eye(N, dtype=torch.bool, device=h.device)
    labels_col = labels.unsqueeze(1)                         # (N, 1)
    pos_mask = (labels_col == labels.unsqueeze(0)) & mask_self  # (N, N)
    # 若某样本没有正对（独类），跳过
    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return h.new_zeros(1).squeeze()
    # log-sum-exp 分母（所有非自身）
    sim_masked = sim.masked_fill(~mask_self, -1e9)
    log_denom = torch.logsumexp(sim_masked, dim=1)          # (N,)
    # 正对对数概率均值
    log_prob = sim - log_denom.unsqueeze(1)                  # (N, N)
    n_pos = pos_mask.float().sum(dim=1).clamp(min=1)
    loss_per = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos
    return loss_per[has_pos].mean()
