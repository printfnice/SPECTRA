# e2e_multitask.py
# 依赖: e2e_pipeline.py (内部��数), e2e_modules.py, torch, sklearn
# 被依赖: e2e_pipeline.py (条件导入，--use_multitask 开关)
# 职责: 多任务学习扩展 - 主分类任务 + 遗传亚型辅助分类头（导师思路A，R32）
#       主任务: 疾病/健康 CE Loss（不变，性能指标来源）
#       辅助任务: 遗传亚型分类（TTN/LMNA/RBM20/PLN/DES/NF等）CE Loss × weight
#       两任务共享 MS-HGATLink + PMA + VIB 编码器，分别接独立分类头

import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder


def get_subtype_labels(adata, samples, sample_key):
    """提取 Primary.Genetic.Diagnosis 亚型标签，与 samples 一一对应。
    若列不存在返回 None（调用方应禁用多任务）。
    """
    col = 'Primary.Genetic.Diagnosis'
    if col not in adata.obs.columns:
        return None
    pgd_map = adata.obs.groupby(sample_key)[col].first()
    labels = [str(pgd_map.get(s, 'Unknown')) for s in samples]
    print(f"  [多任务] 亚型标签: {sorted(set(labels))}  n_subtypes={len(set(labels))}")
    return labels


def _init_multitask_components(n_genes, n_cts, model_cfg, gene_to_idx,
                                config, device, seed, n_pathways, n_classes, n_subtypes):
    """初始化多任务所需所有模块，返回 (model, agg, vib, clf, sub_clf)。"""
    from pipeline.e2e_pipeline import _init_model, _freeze_adapters
    from model.e2e_modules import SampleAggregator, E2EClassifier, VIBLayer
    model, fusion_dim = _init_model(
        n_genes, n_cts, model_cfg, gene_to_idx, device, seed, n_pathways)
    _freeze_adapters(model, config['freeze_esm2_adapter'], config['freeze_kge_adapter'])
    agg = SampleAggregator(
        dim=fusion_dim, k=config['pma_k'],
        n_heads=config['pma_heads'], dropout=config['pma_dropout']
    ).to(device)
    vib, clf_in = None, fusion_dim
    if config.get('use_vib', False):
        vib_z = max(fusion_dim // 2, 32)
        vib = VIBLayer(fusion_dim, vib_z).to(device)
        clf_in = vib_z
    clf = E2EClassifier(clf_in, n_classes, dropout=0.3, l1_lambda=config['l1_lambda']).to(device)
    sub_clf = E2EClassifier(clf_in, n_subtypes, dropout=0.3, l1_lambda=0.0).to(device)
    return model, agg, vib, clf, sub_clf


def _build_multitask_optimizer(model, agg, vib, clf, sub_clf, config):
    """构建 AdamW 优化器和 CosineAnnealingLR 调度器。"""
    params = (list(model.parameters()) + list(agg.parameters()) +
              list(clf.parameters()) + list(sub_clf.parameters()))
    if vib is not None:
        params += list(vib.parameters())
    opt = torch.optim.AdamW(
        [p for p in params if p.requires_grad],
        lr=config['lr'], weight_decay=config['weight_decay'])
    sched = None
    if config.get('use_lr_scheduler', True):
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=config['n_epochs'], eta_min=config['lr'] * 0.05)
    return opt, sched


def _backward(loss, all_params, opt, scaler):
    """统一反向传播步骤（支持 AMP GradScaler）。"""
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()


def _train_epoch_multitask(model, agg, vib, clf, sub_clf, train_bd,
                            train_y_t, sub_y_t, opt, device, config, mt_weight, scaler):
    """多任务单轮训练：主CE + mt_weight * 辅助亚型CE + [VIB KL]。"""
    for m in [model, agg, clf, sub_clf]:
        m.train()
    if vib is not None:
        vib.train()
    opt.zero_grad()
    from pipeline.e2e_pipeline import _forward_e2e
    h_list, lbl_list, sub_list, kl_total = [], [], [], 0.0
    for cpu_t, lbl, sub in zip(train_bd, train_y_t, sub_y_t):
        if cpu_t is None:
            continue
        fused = _forward_e2e(model, cpu_t, device, use_amp=(scaler is not None))
        h = agg(fused)
        if vib is not None:
            h, kl = vib(h)
            kl_total = kl_total + kl
        h_list.append(h); lbl_list.append(lbl); sub_list.append(sub)
    if not h_list:
        return
    h_s = torch.stack(h_list)
    loss = (F.cross_entropy(clf(h_s), torch.stack(lbl_list)) + clf.l1_loss() +
            mt_weight * F.cross_entropy(sub_clf(h_s), torch.stack(sub_list)))
    if vib is not None:
        loss = loss + config.get('vib_beta', 0.0005) * kl_total / len(h_list)
    modules = [model, agg, clf, sub_clf] + ([vib] if vib is not None else [])
    all_p = [p for m in modules for p in m.parameters() if p.requires_grad]
    _backward(loss, all_p, opt, scaler)


def run_multitask_fold(train_bd, train_y, train_sub_y,
                       test_bd, test_y,
                       n_genes, n_cts, gene_to_idx, model_cfg,
                       config, n_classes, device, seed):
    """多任务单折训练+评估。以主任务测试 AUROC 为性能指标（与基线可比）。"""
    from pipeline.e2e_pipeline import _evaluate
    le = LabelEncoder()
    train_sub_int = le.fit_transform(train_sub_y).tolist()
    n_subtypes = len(le.classes_)
    model, agg, vib, clf, sub_clf = _init_multitask_components(
        n_genes, n_cts, model_cfg, gene_to_idx, config, device, seed,
        config.get('n_pathways', 0), n_classes, n_subtypes)
    opt, sched = _build_multitask_optimizer(model, agg, vib, clf, sub_clf, config)
    train_y_t = [torch.tensor(y, dtype=torch.long, device=device) for y in train_y]
    sub_y_t   = [torch.tensor(y, dtype=torch.long, device=device) for y in train_sub_int]
    use_amp   = (device == 'cuda')
    scaler    = torch.amp.GradScaler('cuda') if use_amp else None
    mt_w      = config.get('multitask_weight', 0.3)
    best, eval_iv = 0.0, config.get('eval_interval', 10)
    for ep in range(config['n_epochs']):
        _train_epoch_multitask(model, agg, vib, clf, sub_clf, train_bd,
                                train_y_t, sub_y_t, opt, device, config, mt_w, scaler)
        if sched is not None:
            sched.step()
        if (ep + 1) % eval_iv == 0:
            auroc = _evaluate(model, agg, clf, test_bd, test_y,
                              n_classes, device, config=config, vib=vib, use_amp=use_amp)
            best = max(best, auroc)
    return max(best, _evaluate(model, agg, clf, test_bd, test_y,
                               n_classes, device, config=config, vib=vib, use_amp=use_amp))
