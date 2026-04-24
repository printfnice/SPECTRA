# e2e_pipeline.py
# 依赖: ms_hgatlink_encoders.py, ms_hgatlink_method.py, e2e_modules.py, liana, sklearn
# 被依赖: 直接运行
# 职责: 端到端（E2E）LR推断→样本分类流水线（替代 Step3 Tensor + Step4 RF）
#       支持跨数据集迁移学习：先在源数据集预训练编码器，再迁移至目标数据集

import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

warnings.filterwarnings('ignore')
from contextlib import nullcontext as _nullcontext

# 路径设置：从 code/pipeline/ 运行
_PIPELINE_DIR = Path(__file__).parent
_CODE_DIR = _PIPELINE_DIR.parent
_BASE_DIR = _CODE_DIR.parent
sys.path.insert(0, str(_CODE_DIR))

from model.ms_hgatlink_encoders import MSHGATLink
from model.ms_hgatlink_method import (
    _build_gene_ct_index, _get_model_config, _get_aug_config,
    _get_sample_data, _filter_valid_lr, _build_batch_entries,
    _normalize_expr, _load_esm2_features, _load_kge_features,
    _load_genept_features,
)
from model.pathway_utils import build_lr_pathway_map
from model.e2e_modules import (
    SampleAggregator, E2EClassifier, VIBLayer,
    GatedABMIL, PseudoBagSplitter, supcon_loss,
)
from pipeline.e2e_utils import (
    build_unified_config, build_output_tag,
    collect_all_gate_weights, transfer_weights,
    save_e2e_results, save_gate_weights, map_ensembl_to_symbol,
)

# --- 默认配置
DEFAULT_CONFIG = {
    'dataset_name':         'kuppe',
    'use_gpu':              True,
    'n_cv_folds':           3,
    'n_epochs':             80,
    'lr':                   1e-3,
    'weight_decay':         1e-4,
    'l1_lambda':            0.01,
    'pma_k':                30,
    'pma_heads':            4,
    'pma_dropout':          0.1,
    'max_lr_pairs':         300,
    'e2e_max_lr_pairs':     100,
    'min_cells':            5,
    'use_raw':              False,
    'resource_name':        'consensus',
    'n_random_states':      3,
    'freeze_esm2_adapter':  False,     # 消融 A/B/C/D 控制
    'freeze_kge_adapter':   False,
    'freeze_encoder':       False,
    'use_lr_scheduler':     True,
    'use_transfer':         False,
    'transfer_source':      'kuppe',
    'transfer_epochs':      40,
    'use_mixup':            False,     # R26 正则化
    'use_vib':              False,
    'vib_beta':             0.001,
    'acmil_drop_ratio':     0.0,
    'mc_dropout_T':         0,
    'use_dtfd':             False,     # R26 DTFD 伪袋
    'dtfd_K':               4,
    'early_stop_patience':  0,         # 0=禁用
    'verbose':              True,
    'unified':              False,
    'collect_gate_weights': False,
    'collect_embedding':    False,     # 收集 VIB 层嵌入用于 t-SNE/UMAP 可视化
    'collect_metrics':      False,     # 保存完整评估指标（ROC/PR/F1 等），用于 ROC 曲线绘图
    'use_pathway_prior':    False,     # CellChat 通路先验（仅 reichart 默认启用）
    'use_multitask':        False,     # Tier 4: 主任务+遗传亚型辅助分类头（R32）
    'multitask_weight':     0.3,       # 辅助任务损失权重
    'use_pathway_pool':     False,     # Direction 1: 通路级聚合替代 LR pair 级 PMA
    'use_supcon':           False,     # R37: 监督对比辅助损失（Khosla NeurIPS 2020）
    'supcon_weight':        0.1,       # SupCon 损失权重
    'supcon_tau':           0.07,      # SupCon 温度参数
    'use_swa':              False,     # Stochastic Weight Averaging（最后 swa_start_frac 比例 epoch）
    'swa_start_frac':       0.8,       # SWA 开始比例（最后 20% epoch）
    'use_subtype_supcon':   False,     # 亚型双重 SupCon（Primary.Genetic.Diagnosis 标签）
    'subtype_supcon_weight': 0.1,      # 亚型 SupCon 权重
    'label_smoothing':      0.0,       # CE Loss Label Smoothing（ε=0.1 推荐）
    'use_supcon_proj':      False,     # SupCon Projection Head（SimCLR 风格，解耦对比梯度）
    'supcon_proj_dim':      64,        # Projection Head 输出维度
    'use_focal_loss':       False,     # Focal Loss（γ=2，聚焦难样本）替代 CE
    'focal_gamma':          2.0,       # Focal Loss focusing 参数
    'use_class_weight':     False,     # 类别权重再平衡（少数类加权）
    'use_rdrop':            False,     # R-Drop（两次 dropout 前向 + KL 一致性，NeurIPS 2021）
    'rdrop_alpha':          0.3,       # R-Drop KL 惩罚权重
    'use_warm_restarts':    False,     # SGDR: CosineAnnealingWarmRestarts，帮助逃出局部最小值
    'warm_restart_T0':      100,       # SGDR 重启周期（每 T0 epoch 重启一次 LR）
    'lr_warmup_epochs':     0,         # LR 线性热身 epoch 数（0=禁用），从 lr/10 升至 lr
}

# 数据集特异性 E2E 超参（非 unified 模式时使用，覆盖 DEFAULT_CONFIG）
_R26_FULL = {'dtfd_K': 4, 'use_mixup': True, 'use_vib': True,
             'acmil_drop_ratio': 0.2, 'use_dtfd': True, 'mc_dropout_T': 30}
_E2E_DATASET_CONFIGS = {
    'carraro':   {**_R26_FULL, 'pma_k': 30, 'l1_lambda': 0.01, 'pma_dropout': 0.1,
                  'e2e_max_lr_pairs': 100, 'n_random_states': 5, 'lr': 1e-3, 'n_epochs': 80},
    'habermann': {**_R26_FULL, 'pma_k': 30, 'l1_lambda': 0.01, 'pma_dropout': 0.1,
                  'e2e_max_lr_pairs': 100, 'n_random_states': 5, 'lr': 1e-3,
                  'use_lr_scheduler': False},
    'kuppe':     {'pma_k': 10, 'l1_lambda': 0.02, 'pma_dropout': 0.15,
                  'e2e_max_lr_pairs': 60, 'n_random_states': 5, 'lr': 5e-4},
    'velmeshev': {'pma_k': 30, 'l1_lambda': 0.01, 'pma_dropout': 0.1,
                  'e2e_max_lr_pairs': 100, 'n_random_states': 3, 'lr': 1e-3,
                  'n_epochs': 150, 'mc_dropout_T': 30},
    'reichart':  {'pma_k': 20, 'l1_lambda': 0.01, 'pma_dropout': 0.1,
                  'e2e_max_lr_pairs': 80, 'n_random_states': 3, 'lr': 1e-3, 'n_epochs': 60},
}

# 各数据集实际 obs 列名（来自 process/processer.py DatasetHandler，已验证）
_DATASET_KEYS = {
    'kuppe':     {'sample_key': 'sample',      'groupby': 'cell_type_original', 'condition_key': 'patient_group'},
    'reichart':  {'sample_key': 'Sample',      'groupby': 'cell_type',          'condition_key': 'disease'},
    'habermann': {'sample_key': 'Sample_Name', 'groupby': 'celltype',           'condition_key': 'Status'},
    'velmeshev': {'sample_key': 'sample',      'groupby': 'cluster',            'condition_key': 'diagnosis'},
    'carraro':   {'sample_key': 'orig.ident',  'groupby': 'major',              'condition_key': 'type'},
}

_DATA_DIR = _BASE_DIR / 'data'
_OUTPUT_DIR = _BASE_DIR / 'output'
def _get_device(use_gpu):
    return 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
def _load_adata(dataset_name):
    """加载预处理后的 h5ad 文件（Step1 输出）。"""
    import anndata
    candidates = [
        _DATA_DIR / 'interim' / f'{dataset_name}_processed.h5ad',
        _DATA_DIR / 'interim' / f'{dataset_name}_filtered.h5ad',  # reichart
        _DATA_DIR / f'{dataset_name}_filtered.h5ad',
    ]
    for p in candidates:
        if p.exists():
            print(f"  加载数据: {p}")
            adata = anndata.read_h5ad(str(p))
            return map_ensembl_to_symbol(adata, _DATA_DIR)
    raise FileNotFoundError(f"找不到 {dataset_name} 的处理后数据，候选路径: {candidates}")
def _get_sample_labels(adata, sample_key, condition_key):
    """提取每个样本的条件标签（字符串）。"""
    mapping = adata.obs.groupby(sample_key)[condition_key].first()
    samples = list(mapping.index)
    labels = list(mapping.values)
    return samples, labels
def _encode_labels(labels):
    """字符串标签 → 整数，返回 (int_labels, n_classes, label_names)。"""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return y, len(le.classes_), list(le.classes_)
def _load_prior_features(model_cfg, gene_to_idx, device):
    """一次性加载所有先验特征（ESM-2/KGE/GenePT），跨 fold×rs 复用。"""
    esm2_feat = kge_feat = genept_feat = None
    if model_cfg.get('use_esm2'):
        esm2_feat = _load_esm2_features(gene_to_idx)
        if esm2_feat is not None:
            esm2_feat = esm2_feat.to(device)
    if model_cfg.get('use_kge'):
        kge_feat = _load_kge_features(gene_to_idx)
        if kge_feat is not None:
            kge_feat = kge_feat.to(device)
    if model_cfg.get('use_genept'):
        genept_feat = _load_genept_features(gene_to_idx)
        if genept_feat is not None:
            genept_feat = genept_feat.to(device)
    return esm2_feat, kge_feat, genept_feat

def _init_model(n_genes, n_cts, model_cfg, gene_to_idx, device, seed, n_pathways=0,
                prior_features=None):
    """初始化 MSHGATLink。prior_features=(esm2,kge,genept) 由外部预加载以避免重复 I/O。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dim = model_cfg['gene_emb_dim']
    n_heads = 4 if dim >= 64 else 2
    if prior_features is not None:
        esm2_feat, kge_feat, genept_feat = prior_features
    else:
        esm2_feat, kge_feat, genept_feat = _load_prior_features(model_cfg, gene_to_idx, device)
    model = MSHGATLink(
        n_genes=n_genes, n_cell_types=n_cts,
        gene_emb_dim=dim, pathway_dim=dim, ct_dim=dim,
        fusion_dim=dim * 2, n_heads=n_heads, dropout=model_cfg['dropout'],
        esm2_features=esm2_feat, kge_features=kge_feat,
        genept_features=genept_feat, n_pathways=n_pathways,
    ).to(device)
    return model, dim * 2  # model, fusion_dim
def _freeze_adapters(model, freeze_esm2, freeze_kge):
    """消融实验：冻结指定 adapter 参数。"""
    if freeze_esm2 and hasattr(model.gene_encoder, 'esm2_adapter'):
        for p in model.gene_encoder.esm2_adapter.parameters():
            p.requires_grad = False
    if freeze_kge and hasattr(model.pathway_encoder, 'kg_adapter'):
        if model.pathway_encoder.kg_adapter is not None:
            for p in model.pathway_encoder.kg_adapter.parameters():
                p.requires_grad = False
def _pretrain_source_encoder(src_name, tgt_model_cfg, config, device):
    """在 src_name 上预训练 E2E（迁移学习），返回 (model, gene_to_idx)。"""
    import liana as li  # noqa: 迁移学习独立调用，需本地导入
    adata_src = _load_adata(src_name)
    dk = _DATASET_KEYS[src_name]
    samples_src, labels_src = _get_sample_labels(
        adata_src, dk['sample_key'], dk['condition_key']
    )
    g2i_src, ct2i_src, n_genes_src, n_cts_src = _build_gene_ct_index(
        adata_src, dk['groupby']
    )
    resource = li.resource.select_resource(config['resource_name'])
    bd_src = _prepare_all_samples(
        samples_src, adata_src, resource, g2i_src, ct2i_src,
        tgt_model_cfg, config, dk['sample_key'], dk['groupby']
    )
    labels_int, n_cls, _ = _encode_labels(labels_src)
    model, fusion_dim = _init_model(n_genes_src, n_cts_src, tgt_model_cfg, g2i_src, device, 42)
    _freeze_adapters(model, config['freeze_esm2_adapter'], config['freeze_kge_adapter'])
    agg = SampleAggregator(
        dim=fusion_dim, k=config['pma_k'],
        n_heads=config['pma_heads'], dropout=config['pma_dropout']
    ).to(device)
    clf = E2EClassifier(in_dim=fusion_dim, n_classes=n_cls,
                        dropout=0.3, l1_lambda=config['l1_lambda']).to(device)
    params = list(model.parameters()) + list(agg.parameters()) + list(clf.parameters())
    opt = torch.optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
    labels_t = [torch.tensor(y, dtype=torch.long, device=device) for y in labels_int]
    for ep in range(config['transfer_epochs']):
        _train_epoch(model, agg, clf, bd_src, labels_t, opt, device)
    return model, g2i_src
def _prepare_one_sample(sample, adata, resource, gene_to_idx, ct_to_idx,
                         model_cfg, config, sample_key, groupby,
                         lr_pathway_map=None):
    """为单个样本构建 batch_data，返回预张量化的 CPU 字典（避免每 epoch 重建）。"""
    max_lr = config.get('e2e_max_lr_pairs', config['max_lr_pairs'])
    mask = adata.obs[sample_key] == sample
    sample_adata = adata[mask].copy()
    ct_expr_dict, _, _, valid_cts, gene_names, _ = _get_sample_data(
        sample_adata, groupby, config['min_cells'], config['use_raw']
    )
    if ct_expr_dict is None:
        return None
    valid_lr = _filter_valid_lr(resource, set(gene_names), max_lr)
    if len(valid_lr) == 0:
        return None
    batch_data = _build_batch_entries(
        valid_lr, gene_names, ct_expr_dict, gene_to_idx, ct_to_idx, valid_cts,
        lr_pathway_map=lr_pathway_map,
    )
    if not batch_data:
        return None
    return _tensorize_batch_data(batch_data, model_cfg)  # CPU 张量，训练时移到 GPU
def _tensorize_batch_data(batch_data, model_cfg):
    """一次性将 batch_data 列表转为 CPU 张量字典（避免每 epoch 重建 Python 对象）。"""
    lig_norm, rec_norm = _normalize_expr(batch_data)
    score_formula = model_cfg.get('score_formula', 'product')
    if score_formula == 'mean':
        base_vals = [(d['lig_expr'] + d['rec_expr']) / 2 for d in batch_data]
    else:
        base_vals = [d['lig_expr'] * d['rec_expr'] for d in batch_data]
    result = {
        'lig_idx':   torch.tensor([d['lig_idx'] for d in batch_data], dtype=torch.long),
        'rec_idx':   torch.tensor([d['rec_idx'] for d in batch_data], dtype=torch.long),
        'lig_expr':  torch.tensor([[x] for x in lig_norm], dtype=torch.float32),
        'rec_expr':  torch.tensor([[x] for x in rec_norm], dtype=torch.float32),
        'src_ct':    torch.tensor([d['source_ct_idx'] for d in batch_data], dtype=torch.long),
        'tgt_ct':    torch.tensor([d['target_ct_idx'] for d in batch_data], dtype=torch.long),
        'expr_prod': torch.tensor([[v] for v in base_vals], dtype=torch.float32),
    }
    if any(d.get('pathway_idx', 0) > 0 for d in batch_data):
        result['pathway_idx'] = torch.tensor([d.get('pathway_idx', 0) for d in batch_data], dtype=torch.long)
    result['lr_names'] = [f"{d['ligand']}__{d['receptor']}__{d['source_ct']}__{d['target_ct']}"
                          for d in batch_data]
    return result
def _prepare_all_samples(samples, adata, resource, gene_to_idx, ct_to_idx,
                          model_cfg, config, sample_key, groupby,
                          lr_pathway_map=None):
    """批量构建所有样本的预张量化数据（CPU，一次性，无梯度，避免全量 copy）。"""
    all_tensors = []
    for s in samples:
        t = _prepare_one_sample(
            s, adata, resource, gene_to_idx, ct_to_idx,
            model_cfg, config, sample_key, groupby,
            lr_pathway_map=lr_pathway_map,
        )
        all_tensors.append(t)
    return all_tensors
def _move_to_device(batch_data_list, device):
    """Pre-move batch_data to device, reused across random states."""
    return [({k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in t.items()} if t is not None else None)
            for t in batch_data_list]
def _forward_e2e(model, tensors, device, use_amp=False):
    """前向传播，返回 fused (L, fusion_dim)。使用梯度检查点降低显存峰值。
    梯度检查点：不保存中间激活，backward 时重新计算，峰值从 ~132 MB/样本降至 ~26 MB/样本。
    """
    from torch.utils.checkpoint import checkpoint as grad_checkpoint
    pi = tensors.get('pathway_idx')
    def _run(*args):
        lig_idx, rec_idx, lig_expr, rec_expr, src_ct, tgt_ct, expr_prod = args[:7]
        pathway_idx = args[7] if len(args) > 7 else None
        _, _, fused = model(lig_idx, rec_idx, lig_expr, rec_expr, src_ct, tgt_ct,
                            expr_product=expr_prod, pathway_idx=pathway_idx,
                            return_embeddings=True)
        return fused
    run_args = (tensors['lig_idx'], tensors['rec_idx'], tensors['lig_expr'], tensors['rec_expr'],
                tensors['src_ct'], tensors['tgt_ct'], tensors['expr_prod'])
    if pi is not None:
        run_args = run_args + (pi,)
    return grad_checkpoint(_run, *run_args, use_reentrant=False)
def _batch_forward_e2e(model, tensors_list, device, use_amp=False, chunk_size=8):
    """批量编码：按 chunk_size 分组，避免大 tensor 触发显存碎片 OOM。
    每组独立 GPU call，梯度正常流动。返回与 tensors_list 等长的 list。"""
    valid_pairs = [(i, t) for i, t in enumerate(tensors_list) if t is not None]
    if not valid_pairs:
        return [None] * len(tensors_list)
    result = [None] * len(tensors_list)
    keys = ['lig_idx', 'rec_idx', 'lig_expr', 'rec_expr', 'src_ct', 'tgt_ct', 'expr_prod']
    amp_ctx = torch.autocast('cuda', dtype=torch.float16) if (use_amp and device == 'cuda') else _nullcontext()
    for c in range(0, len(valid_pairs), chunk_size):
        chunk = valid_pairs[c:c + chunk_size]
        idxs, valid_ts = zip(*chunk)
        lens = [t['lig_idx'].size(0) for t in valid_ts]
        cat = {k: torch.cat([t[k] for t in valid_ts], dim=0) for k in keys}
        if all('pathway_idx' in t for t in valid_ts):
            cat['pathway_idx'] = torch.cat([t['pathway_idx'] for t in valid_ts], dim=0)
        with amp_ctx:
            _, _, fused_chunk = model(
                cat['lig_idx'], cat['rec_idx'], cat['lig_expr'], cat['rec_expr'],
                cat['src_ct'], cat['tgt_ct'], expr_product=cat['expr_prod'],
                pathway_idx=cat.get('pathway_idx'), return_embeddings=True)
        for orig_i, fused in zip(idxs, torch.split(fused_chunk, lens, dim=0)):
            result[orig_i] = fused
    return result
def _collect_embeddings(model, aggregator, vib, all_batch_data, labels_int,
                        sample_ids, device, config):
    """Collect per-sample VIB embeddings for t-SNE/UMAP. Returns dict with arrays."""
    model.eval(); aggregator.eval()
    if vib is not None: vib.eval()
    use_amp = (device == 'cuda')
    amp_ctx = torch.autocast('cuda', dtype=torch.float16) if use_amp else _nullcontext()
    embs, sids, lbls = [], [], []
    with torch.no_grad():
        for i, cpu_t in enumerate(all_batch_data):
            if cpu_t is None:
                continue
            t = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in cpu_t.items()}
            with amp_ctx:
                pi = t.get('pathway_idx')
                _, _, fused = model(
                    t['lig_idx'], t['rec_idx'], t['lig_expr'], t['rec_expr'],
                    t['src_ct'], t['tgt_ct'], expr_product=t['expr_prod'],
                    pathway_idx=pi, return_embeddings=True)
                h = aggregator(fused, pi)
                if vib is not None:
                    h, _ = vib(h)
            embs.append(h.float().cpu().numpy())
            sids.append(sample_ids[i] if i < len(sample_ids) else str(i))
            lbls.append(labels_int[i] if i < len(labels_int) else -1)
    if not embs:
        return None
    return {'embeddings': np.stack(embs), 'sample_ids': np.array(sids),
            'labels': np.array(lbls)}


def _compute_auroc(y_true, proba, n_classes):
    """计算 AUROC（支持二分类和多分类 OvR）。"""
    from sklearn.metrics import roc_auc_score
    if n_classes == 2:
        return roc_auc_score(y_true, proba[:, 1])
    return roc_auc_score(y_true, proba, multi_class='ovr', average='macro')


def _compute_full_metrics(y_true_arr, proba, n_classes):
    """计算完整评估指标（含 ROC/PR 曲线原始数据），供后续绘图无需重跑实验。"""
    from sklearn.metrics import (
        roc_curve, precision_recall_curve, f1_score, precision_score,
        recall_score, balanced_accuracy_score, matthews_corrcoef,
        average_precision_score,
    )
    y_pred = np.argmax(proba, axis=1)
    m = {'auroc': float(_compute_auroc(y_true_arr, proba, n_classes))}
    m['f1_macro']        = float(f1_score(y_true_arr, y_pred, average='macro',    zero_division=0))
    m['f1_weighted']     = float(f1_score(y_true_arr, y_pred, average='weighted', zero_division=0))
    m['precision_macro'] = float(precision_score(y_true_arr, y_pred, average='macro',    zero_division=0))
    m['recall_macro']    = float(recall_score(y_true_arr, y_pred,    average='macro', zero_division=0))
    m['bacc']            = float(balanced_accuracy_score(y_true_arr, y_pred))
    m['mcc']             = float(matthews_corrcoef(y_true_arr, y_pred))
    if n_classes == 2:
        pos_prob = proba[:, 1]
        try:
            m['auprc'] = float(average_precision_score(y_true_arr, pos_prob))
            fpr_r, tpr_r, _ = roc_curve(y_true_arr, pos_prob)
            grid = np.linspace(0, 1, 100)
            m['fpr'] = grid.tolist()
            m['tpr'] = np.interp(grid, fpr_r, tpr_r).tolist()
            p_r, r_r, _ = precision_recall_curve(y_true_arr, pos_prob)
            m['pr_recall']    = r_r.tolist()
            m['pr_precision'] = p_r.tolist()
        except Exception:
            pass
    else:
        try:
            m['auprc'] = float(average_precision_score(y_true_arr, proba, average='macro'))
        except Exception:
            pass
    m['y_true'] = y_true_arr.tolist()
    m['y_prob'] = proba.tolist()
    return m
def _train_epoch(model, aggregator, classifier, all_tensors, labels_t, optimizer, device,
                 freeze_encoder=False, config=None, vib=None, dtfd_splitter=None,
                 tier1_agg=None, scaler=None, trainable_params=None, subtype_labels_t=None,
                 supcon_proj=None, class_weights=None):
    """单轮训练：编码器→[DTFD]→PMA→[VIB]→[Mixup]→分类器。scaler: AMP GradScaler。
    trainable_params: 预构建的可训练参数列表（避免每 epoch 重建，由 _run_one_fold 传入）。
    """
    cfg = config or {}
    use_amp = scaler is not None
    if freeze_encoder:
        model.eval()
    else:
        model.train()
    aggregator.train()
    classifier.train()
    if vib is not None:
        vib.train()
    if tier1_agg is not None:
        tier1_agg.train()
    optimizer.zero_grad()

    use_mixup = cfg.get('use_mixup', False)
    use_vib = cfg.get('use_vib', False)
    use_dtfd = cfg.get('use_dtfd', False) and dtfd_splitter is not None
    amp_ctx = torch.autocast('cuda', dtype=torch.float16) if (use_amp and device == 'cuda') else _nullcontext()
    trainable = trainable_params
    if trainable is None:
        all_params = (list(model.parameters()) + list(aggregator.parameters()) +
                      list(classifier.parameters()))
        if vib is not None: all_params += list(vib.parameters())
        if tier1_agg is not None: all_params += list(tier1_agg.parameters())
        trainable = [p for p in all_params if p.requires_grad]

    def _step():
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

    def _encode_h(cpu_t):
        if freeze_encoder:
            with torch.no_grad():
                with amp_ctx:
                    fused = _forward_e2e(model, cpu_t, device)
            fused = fused.detach()
        else:
            with amp_ctx:
                fused = _forward_e2e(model, cpu_t, device)
        if use_dtfd and tier1_agg is not None:
            pb_vecs = torch.stack([tier1_agg(pb) for pb in dtfd_splitter.split(fused, training=True)])
            return aggregator(pb_vecs)
        return aggregator(fused, cpu_t.get('pathway_idx'))

    # --- 统一批量路径：梯度检查点保证 SupCon 梯度完整传播至编码器 ---
    h_list, lbl_list, sub_lbl_list = [], [], []
    total_kl = 0.0
    for i, (cpu_t, label) in enumerate(zip(all_tensors, labels_t)):
        if cpu_t is None:
            continue
        h = _encode_h(cpu_t)
        if use_vib and vib is not None:
            h, kl = vib(h)
            total_kl = total_kl + kl
        h_list.append(h)
        lbl_list.append(label)
        if subtype_labels_t is not None:
            sub_lbl_list.append(subtype_labels_t[i])
    if not h_list:
        return 0.0
    h_stack = torch.stack(h_list)
    lbl_stack = torch.stack(lbl_list)
    n_samples = h_stack.size(0)
    ls = cfg.get('label_smoothing', 0.0)
    use_focal = cfg.get('use_focal_loss', False)

    def _ce(logits, targets):
        if use_focal:
            gamma = cfg.get('focal_gamma', 2.0)
            ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=ls,
                                 weight=class_weights)
            pt = torch.exp(-ce)
            return ((1 - pt) ** gamma * ce).mean()
        return F.cross_entropy(logits, targets, reduction='mean', label_smoothing=ls,
                               weight=class_weights)

    if use_mixup:
        alpha = cfg.get('mixup_alpha', 0.4)
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(n_samples, device=device)
        h_mix = lam * h_stack + (1 - lam) * h_stack[idx]
        logits = classifier(h_mix)
        ce_loss = lam * _ce(logits, lbl_stack) + (1 - lam) * _ce(logits, lbl_stack[idx])
    else:
        logits = classifier(h_stack)
        ce_loss = _ce(logits, lbl_stack)
    total_loss = ce_loss + classifier.l1_loss()
    if use_vib and vib is not None:
        total_loss = total_loss + cfg.get('vib_beta', 0.001) * total_kl / n_samples
    if cfg.get('use_supcon', False) and n_samples >= 4:
        sc_feat = h_stack
        if supcon_proj is not None:
            sc_feat = F.normalize(supcon_proj(h_stack), dim=-1)
        sc = supcon_loss(sc_feat, lbl_stack, tau=cfg.get('supcon_tau', 0.07))
        total_loss = total_loss + cfg.get('supcon_weight', 0.1) * sc
    if cfg.get('use_subtype_supcon', False) and sub_lbl_list and n_samples >= 4:
        sub_stack = torch.stack(sub_lbl_list)
        sc_sub = supcon_loss(h_stack, sub_stack, tau=cfg.get('supcon_tau', 0.07))
        total_loss = total_loss + cfg.get('subtype_supcon_weight', 0.1) * sc_sub
    if cfg.get('use_rdrop', False) and not use_mixup:
        logits2 = classifier(h_stack)   # 第二次前向，不同 dropout mask
        p1 = F.softmax(logits, dim=-1)
        p2 = F.softmax(logits2, dim=-1)
        kl_loss = 0.5 * (
            F.kl_div(F.log_softmax(logits, dim=-1), p2.detach(), reduction='batchmean') +
            F.kl_div(F.log_softmax(logits2, dim=-1), p1.detach(), reduction='batchmean')
        )
        total_loss = total_loss + cfg.get('rdrop_alpha', 0.3) * kl_loss
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()
    _step()
    return total_loss.item()
def _evaluate(model, aggregator, classifier, all_tensors, labels, n_classes, device,
              config=None, vib=None, dtfd_splitter=None, tier1_agg=None, use_amp=False,
              return_full=False):
    """评估：收集所有样本 logits → AUROC。支持 MC Dropout 和 AMP。
    return_full=True 时额外返回 (auroc, full_metrics_dict)。
    """
    cfg = config or {}
    mc_T = cfg.get('mc_dropout_T', 0)
    use_dtfd = cfg.get('use_dtfd', False) and dtfd_splitter is not None

    if mc_T > 0:
        return _evaluate_mc(model, aggregator, classifier, all_tensors, labels,
                            n_classes, device, mc_T, vib, dtfd_splitter, tier1_agg,
                            use_amp=use_amp, return_full=return_full)

    for m in [model, aggregator, classifier]: m.eval()
    if vib is not None: vib.eval()
    if tier1_agg is not None: tier1_agg.eval()
    proba_list, y_true = [], []
    with torch.no_grad():
        fused_list = _batch_forward_e2e(model, all_tensors, device, use_amp=use_amp)
        for fused, cpu_t, lbl in zip(fused_list, all_tensors, labels):
            if fused is None:
                continue
            if use_dtfd and tier1_agg is not None:
                h_t1 = tier1_agg(fused).unsqueeze(0)
                h = aggregator(h_t1)
            else:
                h = aggregator(fused, cpu_t.get('pathway_idx') if cpu_t is not None else None)
            if vib is not None:
                h, _ = vib(h)
            logit = classifier(h.unsqueeze(0))
            proba_list.append(F.softmax(logit, dim=-1).cpu().numpy()[0])
            y_true.append(lbl)
    if not proba_list:
        return (0.0, {}) if return_full else 0.0
    proba = np.stack(proba_list)
    y_arr = np.array(y_true)
    auroc = _compute_auroc(y_arr, proba, n_classes)
    if return_full:
        return auroc, _compute_full_metrics(y_arr, proba, n_classes)
    return auroc
def _evaluate_mc(model, aggregator, classifier, all_tensors, labels,
                 n_classes, device, T, vib, dtfd_splitter, tier1_agg,
                 use_amp=False, return_full=False):
    """MC Dropout 推理：T 次前向取均值。保持 dropout 开启降低预测方差。"""
    model.eval()
    aggregator.eval()
    # 分类器保持 train 模式（dropout 开启）
    classifier.train()
    if vib is not None:
        vib.train()
    use_dtfd = dtfd_splitter is not None and tier1_agg is not None
    all_probas, y_true = [], []
    with torch.no_grad():
        fused_list = _batch_forward_e2e(model, all_tensors, device, use_amp=use_amp)
        for fused, cpu_t, lbl in zip(fused_list, all_tensors, labels):
            if fused is None:
                continue
            if use_dtfd and tier1_agg is not None:
                tier1_agg.eval()
                h_t1 = tier1_agg(fused).unsqueeze(0)
                h_base = aggregator(h_t1)
            else:
                h_base = aggregator(fused, cpu_t.get('pathway_idx') if cpu_t is not None else None)
            # 向量化 MC：复制 T 份 → 一次批量前向
            h_batch = h_base.unsqueeze(0).expand(T, -1)        # (T, D)
            if vib is not None:
                h_batch, _ = vib(h_batch)
            logits_batch = classifier(h_batch)                  # (T, C)
            proba_mean = F.softmax(logits_batch, dim=-1).mean(dim=0)
            all_probas.append(proba_mean.cpu().numpy())
            y_true.append(lbl)
    if not all_probas:
        return (0.0, {}) if return_full else 0.0
    proba = np.stack(all_probas)
    y_arr = np.array(y_true)
    auroc = _compute_auroc(y_arr, proba, n_classes)
    if return_full:
        return auroc, _compute_full_metrics(y_arr, proba, n_classes)
    return auroc
# --- 单折训练
def _run_one_fold(train_bd, train_y, test_bd, test_y, labels_int,
                  n_genes, n_cts, gene_to_idx, model_cfg, config, n_classes, device, seed,
                  transfer_state=None, prior_features=None, sub_train_y=None):
    """初始化模型并完成单折 E2E 训练+评估，返回测试 AUROC。"""
    n_pathways = config.get('n_pathways', 0)
    if device == 'cuda':
        torch.cuda.empty_cache()  # 每折前清空碎片化缓存
    model, fusion_dim = _init_model(n_genes, n_cts, model_cfg, gene_to_idx, device, seed,
                                    n_pathways=n_pathways, prior_features=prior_features)
    if transfer_state is not None:
        src_model, src_g2i = transfer_state
        n_t = transfer_weights(src_model, src_g2i, model, gene_to_idx)
        if config['verbose']:
            print(f"    [Transfer] 迁移 {n_t} 个共同基因嵌入")
    _freeze_adapters(model, config['freeze_esm2_adapter'], config['freeze_kge_adapter'])
    freeze_enc = config.get('freeze_encoder', False)
    if freeze_enc:
        for p in model.parameters():
            p.requires_grad = False

    acmil_ratio = config.get('acmil_drop_ratio', 0.0)
    aggregator = SampleAggregator(
        dim=fusion_dim, k=config['pma_k'],
        n_heads=config['pma_heads'], dropout=config['pma_dropout'],
        acmil_drop_ratio=acmil_ratio,
        use_pathway_pool=config.get('use_pathway_pool', False)
    ).to(device)
    vib = None
    clf_in_dim = fusion_dim
    if config.get('use_vib', False):
        vib_z_dim = max(fusion_dim // 2, 32)
        vib = VIBLayer(fusion_dim, vib_z_dim).to(device)
        clf_in_dim = vib_z_dim
    dtfd_splitter = tier1_agg = None
    if config.get('use_dtfd', False):
        dtfd_splitter = PseudoBagSplitter(K=config.get('dtfd_K', 4))
        tier1_agg = GatedABMIL(
            in_dim=fusion_dim, hidden_dim=max(fusion_dim // 2, 64),
            acmil_drop_ratio=acmil_ratio
        ).to(device)
    classifier = E2EClassifier(
        in_dim=clf_in_dim, n_classes=n_classes,
        dropout=0.3, l1_lambda=config['l1_lambda']
    ).to(device)

    # SupCon Projection Head（解耦对比梯度，SimCLR 风格）
    # 注意：h 维度是 clf_in_dim（VIB 后为 vib_z_dim，否则为 fusion_dim）
    supcon_proj = None
    if config.get('use_supcon_proj', False) and config.get('use_supcon', False):
        proj_dim = config.get('supcon_proj_dim', 64)
        import torch.nn as nn
        supcon_proj = nn.Sequential(
            nn.Linear(clf_in_dim, clf_in_dim), nn.GELU(),
            nn.Linear(clf_in_dim, proj_dim)
        ).to(device)

    # --- 优化器（收集所有可训练参数） ---
    all_params = list(aggregator.parameters()) + list(classifier.parameters())
    if not freeze_enc:
        all_params = list(model.parameters()) + all_params
    if vib is not None: all_params += list(vib.parameters())
    if tier1_agg is not None: all_params += list(tier1_agg.parameters())
    if supcon_proj is not None: all_params += list(supcon_proj.parameters())
    trainable_params = [p for p in all_params if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    scheduler = None
    if config.get('use_lr_scheduler', True):
        warmup_ep = config.get('lr_warmup_epochs', 0)
        if config.get('use_warm_restarts', False):
            base_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=config.get('warm_restart_T0', 100), eta_min=config['lr'] * 0.05)
        else:
            base_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config['n_epochs'], eta_min=config['lr'] * 0.05)
        if warmup_ep > 0:
            wu = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_ep)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[wu, base_sched], milestones=[warmup_ep])
        else:
            scheduler = base_sched
    train_y_t = [torch.tensor(y, dtype=torch.long, device=device) for y in train_y]
    class_weights = None
    if config.get('use_class_weight', False):
        from collections import Counter
        cnt = Counter(train_y)
        n_cls = n_classes
        w = [len(train_y) / (n_cls * cnt.get(c, 1)) for c in range(n_cls)]
        class_weights = torch.tensor(w, dtype=torch.float32, device=device)
    sub_train_y_t = None
    if sub_train_y is not None:
        sub_train_y_t = [torch.tensor(y, dtype=torch.long, device=device) for y in sub_train_y]
    use_amp = (device == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    # SWA 初始化
    use_swa = config.get('use_swa', False)
    swa_model = swa_scheduler = None
    if use_swa:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        swa_start = int(config['n_epochs'] * config.get('swa_start_frac', 0.8))
        swa_scheduler = SWALR(optimizer, swa_lr=config.get('lr', 5e-4) * 0.1, anneal_epochs=10)
    best_auroc = 0.0
    eval_interval = config.get('eval_interval', 10)
    patience = config.get('early_stop_patience', 0)  # 0=禁用
    no_improve_count = 0
    for epoch in range(config['n_epochs']):
        _train_epoch(model, aggregator, classifier, train_bd, train_y_t, optimizer, device,
                     freeze_encoder=freeze_enc, config=config, vib=vib,
                     dtfd_splitter=dtfd_splitter, tier1_agg=tier1_agg, scaler=scaler,
                     trainable_params=trainable_params, subtype_labels_t=sub_train_y_t,
                     supcon_proj=supcon_proj, class_weights=class_weights)
        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            if swa_scheduler: swa_scheduler.step()
        elif scheduler is not None:
            scheduler.step()
        if (epoch + 1) % eval_interval == 0:
            auroc = _evaluate(model, aggregator, classifier, test_bd, test_y,
                              n_classes, device, config=config, vib=vib,
                              dtfd_splitter=dtfd_splitter, tier1_agg=tier1_agg,
                              use_amp=use_amp)
            if auroc > best_auroc:
                best_auroc = auroc
                no_improve_count = 0
            else:
                no_improve_count += 1
            if config['verbose']:
                print(f"    epoch {epoch+1}/{config['n_epochs']} test AUROC={auroc:.4f}")
            if patience > 0 and no_improve_count >= patience:
                if config['verbose']:
                    print(f"    Early stop at epoch {epoch+1} (patience={patience})")
                break
    collect_metrics = config.get('collect_metrics', False)
    final_auroc = _evaluate(model, aggregator, classifier, test_bd, test_y,
                            n_classes, device, config=config, vib=vib,
                            dtfd_splitter=dtfd_splitter, tier1_agg=tier1_agg,
                            use_amp=use_amp, return_full=collect_metrics)
    if collect_metrics:
        final_auroc, fold_metrics = final_auroc
    else:
        fold_metrics = None
    best = max(best_auroc, final_auroc)
    if use_swa and swa_model is not None:
        swa_auroc = _evaluate(swa_model, aggregator, classifier, test_bd, test_y,
                              n_classes, device, config=config, vib=vib,
                              dtfd_splitter=dtfd_splitter, tier1_agg=tier1_agg,
                              use_amp=use_amp)
        if config['verbose']:
            print(f"    SWA AUROC={swa_auroc:.4f}  final AUROC={final_auroc:.4f}")
        best = max(best, swa_auroc)
    collect_gate_emb = config.get('collect_gate_weights') or config.get('collect_embedding')
    if collect_gate_emb and collect_metrics:
        return best, model, aggregator, vib, classifier, fold_metrics
    if collect_gate_emb:
        return best, model, aggregator, vib, classifier
    if collect_metrics:
        return best, fold_metrics
    return best
# --- 外层 CV 循环
def _build_cv_splitter(adata, samples, labels_int, config):
    """构建 CV 分割器。含 donor_id 时用 StratifiedGroupKFold；含遗传亚型时用亚型分层。"""
    from sklearn.model_selection import StratifiedKFold
    n_splits = config['n_cv_folds']
    if 'donor_id' in adata.obs.columns:
        from sklearn.model_selection import StratifiedGroupKFold
        sk = _DATASET_KEYS[config['dataset_name']]['sample_key']
        groups = np.array([adata.obs.groupby(sk)['donor_id'].first()[s] for s in samples])
        strat_y, label = labels_int, f"{len(set(groups))} donors, {n_splits} folds"
        if 'Primary.Genetic.Diagnosis' in adata.obs.columns:
            from sklearn.preprocessing import LabelEncoder
            pgd = adata.obs.groupby(sk)['Primary.Genetic.Diagnosis'].first()
            strat_y = LabelEncoder().fit_transform([pgd[s] for s in samples])
            label += ' (亚型分层)'
        print(f"  [CV] StratifiedGroupKFold: {label}")
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42), groups, strat_y
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42), None, labels_int

def run_e2e_cv(adata, samples, labels_str, all_batch_data, gene_to_idx, ct_to_idx,
               model_cfg, config, device):
    """K 折分层 CV，返回所有折的 AUROC 列表。"""
    labels_int, n_classes, label_names = _encode_labels(labels_str)
    n_genes = len(gene_to_idx)
    n_cts = len(ct_to_idx)
    if config['verbose']:
        print(f"  类别: {label_names}  n_classes={n_classes}")

    transfer_state = None
    if config.get('use_transfer', False):
        src_name = config.get('transfer_source', 'kuppe')
        print(f"  [Transfer] 在 {src_name} 上预训练 {config['transfer_epochs']} epochs...")
        src_model, src_g2i = _pretrain_source_encoder(src_name, model_cfg, config, device)
        src_model.eval(); transfer_state = (src_model, src_g2i)

    # Tier 4: 多任务学习（主任务+遗传亚型辅助头）
    use_mt = config.get('use_multitask', False)
    sub_labels = None
    if use_mt:
        from pipeline.e2e_multitask import get_subtype_labels, run_multitask_fold
        sub_labels = get_subtype_labels(adata, samples, _DATASET_KEYS[config['dataset_name']]['sample_key'])
        use_mt = sub_labels is not None
    # 亚型双重 SupCon：加载 Primary.Genetic.Diagnosis 标签
    supcon_sub_labels = None
    if config.get('use_subtype_supcon', False):
        sk = _DATASET_KEYS[config['dataset_name']]['sample_key']
        if 'Primary.Genetic.Diagnosis' in adata.obs.columns:
            from sklearn.preprocessing import LabelEncoder
            pgd = adata.obs.groupby(sk)['Primary.Genetic.Diagnosis'].first()
            sub_raw = [pgd.get(s, 'unknown') for s in samples]
            supcon_sub_labels = LabelEncoder().fit_transform(sub_raw)
            print(f"  [SubtypeSupCon] 加载亚型标签: {len(set(sub_raw))} 类")

    splitter, groups, strat_y = _build_cv_splitter(adata, samples, labels_int, config)
    split_args = (samples, strat_y) if groups is None else (samples, strat_y, groups)
    all_aurocs = []
    last_model = last_aggregator = last_vib = last_classifier = None
    collect_gate = config.get('collect_gate_weights', False)
    collect_emb     = config.get('collect_embedding', False)
    collect_any     = collect_gate or collect_emb
    collect_metrics = config.get('collect_metrics', False)
    all_fold_metrics = []   # list of dicts, one per (fold, rs)
    # 先验特征一次性加载（避免每个 fold×rs 重复 I/O）
    prior_features = _load_prior_features(model_cfg, gene_to_idx, device)
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(*split_args)):
        train_bd_cpu = [all_batch_data[i] for i in train_idx]
        test_bd_cpu  = [all_batch_data[i] for i in test_idx]
        # 预移到 device（每折一次，跨 random_states 复用）
        train_bd = _move_to_device(train_bd_cpu, device)
        test_bd  = _move_to_device(test_bd_cpu, device)
        train_y  = [labels_int[i] for i in train_idx]
        test_y   = [labels_int[i] for i in test_idx]
        fold_aurocs = []
        for rs in range(config['n_random_states']):
            seed = 42 + rs * 7
            if use_mt:
                train_sub_y = [sub_labels[i] for i in train_idx]
                auroc = run_multitask_fold(
                    train_bd, train_y, train_sub_y, test_bd, test_y,
                    n_genes, n_cts, gene_to_idx, model_cfg,
                    config, n_classes, device, seed)
            else:
                sub_train_y = ([supcon_sub_labels[i] for i in train_idx]
                               if supcon_sub_labels is not None else None)
                result = _run_one_fold(
                    train_bd, train_y, test_bd, test_y, labels_int,
                    n_genes, n_cts, gene_to_idx, model_cfg, config, n_classes, device, seed,
                    transfer_state=transfer_state, prior_features=prior_features,
                    sub_train_y=sub_train_y)
                if isinstance(result, tuple):
                    auroc = result[0]
                    if collect_any and len(result) >= 5:
                        last_model, last_aggregator, last_vib, last_classifier = (
                            result[1], result[2], result[3], result[4])
                    if collect_metrics:
                        fm = result[-1]  # last element is metrics dict
                        fm['fold'] = fold_idx
                        fm['rs'] = rs
                        all_fold_metrics.append(fm)
                else:
                    auroc = result
            fold_aurocs.append(auroc)
            if config['verbose']:
                print(f"  fold={fold_idx+1} rs={rs} AUROC={auroc:.4f}")
        mean_auroc = np.mean(fold_aurocs)
        all_aurocs.append(mean_auroc)
        print(f"  [fold {fold_idx+1}] 平均 AUROC = {mean_auroc:.4f}")

    gate_data = (collect_all_gate_weights(last_model, all_batch_data, device,
                                          sample_ids=list(samples))
                 if collect_gate and last_model is not None else None)
    emb_data = (_collect_embeddings(last_model, last_aggregator, last_vib,
                                    all_batch_data, labels_int, list(samples), device, config)
                if collect_emb and last_model is not None else None)
    metrics_data = all_fold_metrics if collect_metrics else None
    return all_aurocs, gate_data, emb_data, metrics_data
# --- 主入口
def _parse_args():
    p = argparse.ArgumentParser(description='E2E LR→Classification Pipeline')
    p.add_argument('--dataset',         default=None)
    p.add_argument('--n_epochs',        type=int, default=None)
    p.add_argument('--pma_k',           type=int, default=None)
    p.add_argument('--freeze_esm2',     action='store_true', default=False)
    p.add_argument('--freeze_kge',      action='store_true', default=False)
    p.add_argument('--n_cv_folds',      type=int, default=None)
    p.add_argument('--no_gpu',          action='store_true', default=False)
    p.add_argument('--use_transfer',    action='store_true', default=False)
    p.add_argument('--transfer_source', default=None)
    p.add_argument('--use_genept',      action='store_true', default=False)
    p.add_argument('--use_mixup',       action='store_true', default=False)
    p.add_argument('--use_vib',         action='store_true', default=False)
    p.add_argument('--vib_beta',        type=float, default=None)
    p.add_argument('--acmil_drop',      type=float, default=None)
    p.add_argument('--mc_dropout',      type=int, default=None)
    p.add_argument('--use_dtfd',        action='store_true', default=False)
    p.add_argument('--dtfd_K',          type=int, default=None)
    p.add_argument('--unified',             action='store_true', default=False)
    p.add_argument('--collect_gate',        action='store_true', default=False)
    p.add_argument('--collect_embedding',   action='store_true', default=False)
    p.add_argument('--collect_metrics',     action='store_true', default=False)
    p.add_argument('--e2e_max_lr_pairs',    type=int, default=None)
    p.add_argument('--early_stop_patience', type=int, default=None)
    p.add_argument('--use_pathway_prior',   action='store_true', default=False)
    p.add_argument('--use_multitask',       action='store_true', default=False)
    p.add_argument('--multitask_weight',    type=float, default=None)
    p.add_argument('--use_pathway_pool',    action='store_true', default=False)
    p.add_argument('--use_supcon',              action='store_true', default=False)
    p.add_argument('--supcon_weight',           type=float, default=None)
    p.add_argument('--use_swa',                 action='store_true', default=False)
    p.add_argument('--use_subtype_supcon',      action='store_true', default=False)
    p.add_argument('--subtype_supcon_weight',   type=float, default=None)
    p.add_argument('--label_smoothing',         type=float, default=None)
    p.add_argument('--use_supcon_proj',         action='store_true', default=False)
    p.add_argument('--supcon_proj_dim',         type=int, default=None)
    p.add_argument('--use_focal_loss',          action='store_true', default=False)
    p.add_argument('--focal_gamma',             type=float, default=None)
    p.add_argument('--use_rdrop',               action='store_true', default=False)
    p.add_argument('--rdrop_alpha',             type=float, default=None)
    p.add_argument('--use_class_weight',        action='store_true', default=False)
    return p.parse_args()
def _apply_args(config, args):
    """CLI 参数覆盖 DEFAULT_CONFIG。"""
    # 直接映射（CLI arg name -> config key）
    _DIRECT = {'dataset': 'dataset_name', 'n_epochs': 'n_epochs', 'pma_k': 'pma_k',
               'n_cv_folds': 'n_cv_folds', 'transfer_source': 'transfer_source',
               'vib_beta': 'vib_beta', 'dtfd_K': 'dtfd_K',
               'e2e_max_lr_pairs': 'e2e_max_lr_pairs',
               'early_stop_patience': 'early_stop_patience',
               'multitask_weight': 'multitask_weight'}
    for arg_name, cfg_key in _DIRECT.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            config[cfg_key] = val
    # 布尔标志
    _FLAGS = {'freeze_esm2': 'freeze_esm2_adapter', 'freeze_kge': 'freeze_kge_adapter',
              'use_transfer': 'use_transfer', 'use_genept': 'use_genept',
              'use_mixup': 'use_mixup', 'use_vib': 'use_vib',
              'use_dtfd': 'use_dtfd', 'unified': 'unified',
              'use_pathway_prior': 'use_pathway_prior',
              'use_multitask': 'use_multitask',
              'use_pathway_pool': 'use_pathway_pool',
              'use_supcon': 'use_supcon',
              'use_swa': 'use_swa',
              'use_subtype_supcon': 'use_subtype_supcon',
              'use_supcon_proj': 'use_supcon_proj',
              'use_focal_loss': 'use_focal_loss',
              'use_rdrop': 'use_rdrop',
              'use_class_weight': 'use_class_weight'}
    for arg_name, cfg_key in _FLAGS.items():
        if getattr(args, arg_name, False):
            config[cfg_key] = True
    if args.no_gpu:          config['use_gpu'] = False
    if args.acmil_drop is not None: config['acmil_drop_ratio'] = args.acmil_drop
    if args.mc_dropout is not None: config['mc_dropout_T'] = args.mc_dropout
    if args.collect_gate:      config['collect_gate_weights'] = True
    if getattr(args, 'collect_embedding', False): config['collect_embedding'] = True
    if getattr(args, 'collect_metrics',   False): config['collect_metrics']   = True
    if getattr(args, 'supcon_weight', None) is not None:
        config['supcon_weight'] = args.supcon_weight
    if getattr(args, 'subtype_supcon_weight', None) is not None:
        config['subtype_supcon_weight'] = args.subtype_supcon_weight
    if getattr(args, 'label_smoothing', None) is not None:
        config['label_smoothing'] = args.label_smoothing
    if getattr(args, 'supcon_proj_dim', None) is not None:
        config['supcon_proj_dim'] = args.supcon_proj_dim
    if getattr(args, 'focal_gamma', None) is not None:
        config['focal_gamma'] = args.focal_gamma
    if getattr(args, 'rdrop_alpha', None) is not None:
        config['rdrop_alpha'] = args.rdrop_alpha
    return config
def _apply_dataset_config(config, n_samples=None):
    """按数据集名称应用超参覆盖。

    unified=True 时使用统一配置（基于样本量自动公式），
    unified=False 时使用数据集特异性配置。
    CLI 传入的消融控制参数（freeze_esm2_adapter/freeze_kge_adapter）不被覆盖。
    """
    preserved = {k: config[k] for k in ('freeze_esm2_adapter', 'freeze_kge_adapter')}
    if config.get('unified') and n_samples is not None:
        config.update(build_unified_config(n_samples, config.get('dataset_name')))
    else:
        dataset = config['dataset_name']
        if dataset in _E2E_DATASET_CONFIGS:
            config.update(_E2E_DATASET_CONFIGS[dataset])
    config.update(preserved)
    return config

def main():
    args = _parse_args()
    config = DEFAULT_CONFIG.copy()
    if args.dataset:
        config['dataset_name'] = args.dataset
    # unified 标志先通过 CLI 注入
    config = _apply_args(config, args)
    dataset_name = config['dataset_name']
    device = _get_device(config['use_gpu'])
    print(f"\n=== E2E Pipeline: {dataset_name} | device={device} ===")

    # 加载数据
    adata = _load_adata(dataset_name)
    dk = _DATASET_KEYS[dataset_name]
    sample_key    = dk['sample_key']
    groupby       = dk['groupby']
    condition_key = dk['condition_key']
    samples, labels_str = _get_sample_labels(adata, sample_key, condition_key)
    n_samples = len(samples)
    print(f"  样本数={n_samples}, 类别={set(labels_str)}")

    # 应用配置：unified 模式使用样本量公式，否则用数据集特异性
    config = _apply_dataset_config(config, n_samples=n_samples)
    config = _apply_args(config, args)  # CLI 再次覆盖（最高优先级）

    if config.get('unified'):
        print(f"  [Unified] 统一配置: VIB={config['use_vib']}, MC_T={config['mc_dropout_T']}, "
              f"epochs={config['n_epochs']}, pma_k={config['pma_k']}, "
              f"lr_pairs={config['e2e_max_lr_pairs']}, lr={config['lr']}")

    # 构建索引
    import liana as li
    gene_to_idx, ct_to_idx, n_genes, n_cts = _build_gene_ct_index(adata, groupby)
    resource = li.resource.select_resource(config['resource_name'])
    # unified 模式统一 gene_emb_dim=64
    if config.get('unified'):
        model_cfg = _get_model_config(dataset_name, 30)
        model_cfg['gene_emb_dim'] = 64
        model_cfg['dropout'] = 0.30
    else:
        model_cfg = _get_model_config(dataset_name, 30)
    if config.get('use_genept'):
        model_cfg['use_genept'] = True

    print(f"  基因数={n_genes}, 细胞类型数={n_cts}, fusion_dim={model_cfg['gene_emb_dim']*2}")
    # CellChat 通路先验（可选，opt-in，零初始化保证不影响 baseline）
    lr_pathway_map, n_pathways = {}, 0
    if config.get('use_pathway_prior'):
        lr_pathway_map, n_pathways = build_lr_pathway_map(_DATA_DIR)
        config['n_pathways'] = n_pathways
        print(f"  [CellChat] 通路先验: {n_pathways} 通路, {len(lr_pathway_map)} LR pairs 覆盖")
    print(f"  预计算 batch_data...")
    all_batch_data = _prepare_all_samples(
        samples, adata, resource, gene_to_idx, ct_to_idx,
        model_cfg, config, sample_key, groupby,
        lr_pathway_map=lr_pathway_map if lr_pathway_map else None,
    )
    n_valid = sum(1 for bd in all_batch_data if bd is not None)
    print(f"  有效样本: {n_valid}/{len(samples)}")

    # 运行 CV
    cv_result = run_e2e_cv(
        adata, samples, labels_str, all_batch_data,
        gene_to_idx, ct_to_idx, model_cfg, config, device
    )
    if isinstance(cv_result, tuple):
        aurocs       = cv_result[0]
        gate_data    = cv_result[1] if len(cv_result) > 1 else None
        emb_data     = cv_result[2] if len(cv_result) > 2 else None
        metrics_data = cv_result[3] if len(cv_result) > 3 else None
    else:
        aurocs, gate_data, emb_data, metrics_data = cv_result, None, None, None
    mode = 'Unified-E2E' if config.get('unified') else 'E2E'
    mean_auroc = np.mean(aurocs)
    std_auroc = np.std(aurocs)
    print(f"\n[{dataset_name}] {mode} AUROC = {mean_auroc:.4f} +/- {std_auroc:.4f}  "
          f"(各折: {[f'{a:.4f}' for a in aurocs]})")

    out_dir = _BASE_DIR / 'output'
    out_dir.mkdir(exist_ok=True)
    _, _, out_path = save_e2e_results(
        dataset_name, aurocs, config, model_cfg, n_samples, out_dir
    )
    print(f"  结果已保存: {out_path}")
    save_gate_weights(gate_data, dataset_name, out_dir)
    if emb_data is not None:
        emb_path = out_dir / 'data' / f'{dataset_name}_e2e_embeddings.npz'
        emb_path.parent.mkdir(exist_ok=True)
        np.savez(str(emb_path), **emb_data)
        print(f"  嵌入已保存: {emb_path} ({len(emb_data['embeddings'])} samples)")
    if metrics_data is not None:
        from pipeline.e2e_utils import save_e2e_metrics
        save_e2e_metrics(dataset_name, aurocs, metrics_data, out_dir)
        print(f"  完整指标已保存: output/data/{dataset_name}_spectra_e2e_metrics.json")
if __name__ == '__main__':
    main()
