# e2e_irm.py
# 依赖: e2e_pipeline.py (内部函数), e2e_modules.py, torch, sklearn
# 被依赖: run_irm_reichart.py（直接运行）
# 职责: Direction 2 — 不变风险最小化（IRM）
#       将 reichart 的遗传亚型当作"环境"，训练对所有亚型不变的特征表示
#       损失 = Σ_e CE^e + lambda * Σ_e ||∇_w CE^e||²
#       核心直觉：迫使编码器学习跨亚型稳健的疾病-通讯模式

import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder


# ============================================================================
# IRM 惩罚计算
# ============================================================================

def _irm_penalty(logits, labels, classifier):
    """计算单个环境的 IRM 梯度惩罚。

    IRM 要求对任意环境 e，当前分类权重 w 都是最优的。
    惩罚 = ||∇_w L^e(w∘Φ)||²，用 scale=1.0 的虚拟线性头近似。
    （参考 Arjovsky et al. 2019，式3的简化版）
    """
    scale = torch.ones(1, requires_grad=True, device=logits.device)
    scaled_logits = logits * scale
    loss = F.cross_entropy(scaled_logits, labels)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


# ============================================================================
# IRM 单折训练
# ============================================================================

def _group_by_subtype(bd_list, labels_int, sub_labels_int, device):
    """按遗传亚型 (sub_labels_int) 将样本分组，返回 {subtype: (bd_list, labels)}。"""
    groups = {}
    for bd, lbl, sub in zip(bd_list, labels_int, sub_labels_int):
        if bd is None:
            continue
        if sub not in groups:
            groups[sub] = ([], [])
        groups[sub][0].append(bd)
        groups[sub][1].append(torch.tensor(lbl, dtype=torch.long, device=device))
    return groups


def _forward_group(model, agg, vib, bd_list, device, use_amp):
    """对一个亚型组的所有样本做前向，返回 (h_stack, lbl_stack)。"""
    from pipeline.e2e_pipeline import _forward_e2e
    h_list, lbl_list = [], []
    for cpu_t, lbl in zip(*bd_list):
        fused = _forward_e2e(model, cpu_t, device, use_amp=use_amp)
        pi = cpu_t.get('pathway_idx')
        h = agg(fused, pi)
        if vib is not None:
            h, _ = vib(h)
        h_list.append(h)
        lbl_list.append(lbl)
    if not h_list:
        return None, None
    return torch.stack(h_list), torch.stack(lbl_list)


def _train_epoch_irm(model, agg, vib, clf, groups, opt, scaler, irm_lambda, cfg, device):
    """IRM 单轮训练：每个亚型环境独立计算 CE + 梯度惩罚，再汇总。"""
    for m in [model, agg, clf]:
        m.train()
    if vib is not None:
        vib.train()
    opt.zero_grad()
    use_amp = scaler is not None
    total_ce, total_penalty, n_env = 0.0, 0.0, 0
    kl_total = 0.0

    for sub, (bd_list, lbl_list) in groups.items():
        bd_lbl = (bd_list, lbl_list)
        h_s, y_s = _forward_group(model, agg, vib, bd_lbl, device, use_amp)
        if h_s is None or h_s.size(0) < 2:
            continue
        logits = clf(h_s)
        ce = F.cross_entropy(logits, y_s)
        penalty = _irm_penalty(logits.detach(), y_s, clf) if irm_lambda > 0 else 0.0
        total_ce += ce
        total_penalty += penalty
        n_env += 1

    if n_env == 0:
        return
    loss = total_ce / n_env + irm_lambda * total_penalty / n_env + clf.l1_loss()

    all_params = [p for m in ([model, agg, clf] + ([vib] if vib else []))
                  for p in m.parameters() if p.requires_grad]
    if scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()


# ============================================================================
# IRM 单折主函数
# ============================================================================

def run_irm_fold(train_bd, train_y, train_sub_y,
                 test_bd, test_y,
                 n_genes, n_cts, gene_to_idx, model_cfg,
                 config, n_classes, device, seed, irm_lambda=1.0):
    """IRM 单折：亚型分组 + 不变特征学习，返回测试 AUROC。"""
    from pipeline.e2e_pipeline import (
        _init_model, _freeze_adapters, _evaluate,
    )
    from model.e2e_modules import SampleAggregator, E2EClassifier, VIBLayer

    le = LabelEncoder()
    train_sub_int = le.fit_transform(train_sub_y).tolist()
    n_pathways = config.get('n_pathways', 0)

    model, fusion_dim = _init_model(
        n_genes, n_cts, model_cfg, gene_to_idx, device, seed, n_pathways)
    _freeze_adapters(model, config['freeze_esm2_adapter'], config['freeze_kge_adapter'])

    agg = SampleAggregator(
        dim=fusion_dim, k=config['pma_k'],
        n_heads=config['pma_heads'], dropout=config['pma_dropout'],
        use_pathway_pool=config.get('use_pathway_pool', False)
    ).to(device)

    vib, clf_in = None, fusion_dim
    if config.get('use_vib', False):
        vib_z = max(fusion_dim // 2, 32)
        vib = VIBLayer(fusion_dim, vib_z).to(device)
        clf_in = vib_z

    clf = E2EClassifier(clf_in, n_classes, dropout=0.3,
                        l1_lambda=config['l1_lambda']).to(device)

    all_params = (list(model.parameters()) + list(agg.parameters()) +
                  list(clf.parameters()) + (list(vib.parameters()) if vib else []))
    opt = torch.optim.AdamW([p for p in all_params if p.requires_grad],
                            lr=config['lr'], weight_decay=config['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config['n_epochs'], eta_min=config['lr'] * 0.05)

    use_amp = (device == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # 按亚型分组（每个环境独立计算损失）
    train_y_int = list(train_y)
    groups = _group_by_subtype(train_bd, train_y_int, train_sub_int, device)
    print(f"    IRM 环境数: {len(groups)}，各亚型样本: "
          f"{[(k, len(v[0])) for k, v in groups.items()]}")

    best, eval_iv = 0.0, config.get('eval_interval', 10)
    for ep in range(config['n_epochs']):
        _train_epoch_irm(model, agg, vib, clf, groups, opt, scaler,
                         irm_lambda, config, device)
        sched.step()
        if (ep + 1) % eval_iv == 0:
            auroc = _evaluate(model, agg, clf, test_bd, test_y,
                              n_classes, device, config=config, vib=vib,
                              use_amp=use_amp)
            best = max(best, auroc)

    final = _evaluate(model, agg, clf, test_bd, test_y,
                      n_classes, device, config=config, vib=vib, use_amp=use_amp)
    return max(best, final)
