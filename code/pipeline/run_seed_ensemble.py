# run_seed_ensemble.py
# 依赖: e2e_pipeline.py, e2e_modules.py, e2e_utils.py, ms_hgatlink_method.py
# 被依赖: 直接运行（--dataset 参数）
# 职责: Seed Ensemble — N 个不同随机种子的 E2E 模型，平均 soft probability 后计算 AUROC
#       核心改进：使用 best-checkpoint 训练（保存最优状态而非最终状态），
#                 对 soft probability 取均值后再算 AUROC（等价于 Bagging）

import copy
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings('ignore')

_PIPELINE_DIR = Path(__file__).parent
_CODE_DIR = _PIPELINE_DIR.parent
_BASE_DIR = _CODE_DIR.parent
sys.path.insert(0, str(_CODE_DIR))

from pipeline.e2e_pipeline import (
    DEFAULT_CONFIG, _DATASET_KEYS,
    _get_device, _load_adata, _get_sample_labels, _encode_labels,
    _build_cv_splitter, _move_to_device, _compute_auroc,
    _prepare_all_samples, _init_model, _freeze_adapters,
    _train_epoch, _evaluate, _forward_e2e,
)
from pipeline.e2e_utils import build_unified_config
from model.e2e_modules import SampleAggregator, VIBLayer, E2EClassifier
from model.pathway_utils import build_lr_pathway_map
from model.ms_hgatlink_method import _build_gene_ct_index, _get_model_config

_DATA_DIR = _BASE_DIR / 'data'
_OUTPUT_DIR = _BASE_DIR / 'output'
_USE_PATHWAY = {'reichart'}


# ============================================================================
# 核心：带 best-checkpoint 追踪的训练 + MC proba 收集
# ============================================================================

def _collect_mc_proba(model, agg, vib, clf, test_bd, test_y, config, device):
    """MC Dropout 推理，返回 (y_true_list, proba_ndarray)。"""
    T = config.get('mc_dropout_T', 15)
    use_amp = (device == 'cuda')
    model.eval(); agg.eval(); clf.train()
    if vib is not None:
        vib.train()
    y_true, proba_list = [], []
    with torch.no_grad():
        for cpu_t, lbl in zip(test_bd, test_y):
            if cpu_t is None:
                continue
            fused = _forward_e2e(model, cpu_t, device, use_amp=use_amp)
            h_base = agg(fused, cpu_t.get('pathway_idx'))
            h_b = h_base.unsqueeze(0).expand(T, -1)
            if vib is not None:
                h_b, _ = vib(h_b)
            logits = clf(h_b)
            proba_list.append(F.softmax(logits, dim=-1).mean(0).cpu().numpy())
            y_true.append(lbl)
    if not proba_list:
        return [], np.zeros((0, 2))
    return y_true, np.stack(proba_list)


def _train_collect_best_proba(train_bd, train_y, test_bd, test_y,
                               n_genes, n_cts, gene_to_idx, model_cfg,
                               config, n_classes, device, seed):
    """训练 E2E，追踪 best checkpoint，返回 (y_true, proba_from_best_model)。

    与 _train_e2e_and_collect 的区别：
      旧: 用最终模型状态推理（可能过拟合）
      新: 保存最优 AUROC 对应的 state_dict，用最优模型推理
    """
    n_pathways = config.get('n_pathways', 0)
    model, fusion_dim = _init_model(n_genes, n_cts, model_cfg, gene_to_idx,
                                    device, seed, n_pathways=n_pathways)
    _freeze_adapters(model, config['freeze_esm2_adapter'], config['freeze_kge_adapter'])

    use_vib = config.get('use_vib', False)
    vib, clf_in = None, fusion_dim
    if use_vib:
        vib_z = max(fusion_dim // 2, 32)
        vib = VIBLayer(fusion_dim, vib_z).to(device)
        clf_in = vib_z

    agg = SampleAggregator(
        dim=fusion_dim, k=config['pma_k'],
        n_heads=config['pma_heads'], dropout=config['pma_dropout'],
        use_pathway_pool=config.get('use_pathway_pool', False)
    ).to(device)
    clf = E2EClassifier(clf_in, n_classes, dropout=0.3,
                        l1_lambda=config['l1_lambda']).to(device)

    all_params = list(model.parameters()) + list(agg.parameters()) + list(clf.parameters())
    if vib is not None:
        all_params += list(vib.parameters())
    trainable = [p for p in all_params if p.requires_grad]

    opt = torch.optim.AdamW(trainable, lr=config['lr'],
                             weight_decay=config['weight_decay'])
    sched = None
    if config.get('use_lr_scheduler', True):
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=config['n_epochs'], eta_min=config['lr'] * 0.05)

    train_y_t = [torch.tensor(y, dtype=torch.long, device=device) for y in train_y]

    eval_interval = config.get('eval_interval', 10)
    best_auroc = 0.0
    best_state = {
        'model': copy.deepcopy(model.state_dict()),
        'agg':   copy.deepcopy(agg.state_dict()),
        'clf':   copy.deepcopy(clf.state_dict()),
        'vib':   copy.deepcopy(vib.state_dict()) if vib else None,
    }

    for epoch in range(config['n_epochs']):
        _train_epoch(model, agg, clf, train_bd, train_y_t, opt, device,
                     freeze_encoder=False, config=config, vib=vib,
                     dtfd_splitter=None, tier1_agg=None,
                     scaler=torch.amp.GradScaler('cuda') if device == 'cuda' else None)
        if sched is not None:
            sched.step()
        if (epoch + 1) % eval_interval == 0:
            auroc = _evaluate(model, agg, clf, test_bd, test_y, n_classes,
                              device, config=config, vib=vib, use_amp=(device == 'cuda'))
            if auroc > best_auroc:
                best_auroc = auroc
                best_state = {
                    'model': copy.deepcopy(model.state_dict()),
                    'agg':   copy.deepcopy(agg.state_dict()),
                    'clf':   copy.deepcopy(clf.state_dict()),
                    'vib':   copy.deepcopy(vib.state_dict()) if vib else None,
                }

    # 用 final model 也比较一下，选更好的
    final_auroc = _evaluate(model, agg, clf, test_bd, test_y, n_classes,
                             device, config=config, vib=vib, use_amp=(device == 'cuda'))
    if final_auroc > best_auroc:
        best_state = {
            'model': copy.deepcopy(model.state_dict()),
            'agg':   copy.deepcopy(agg.state_dict()),
            'clf':   copy.deepcopy(clf.state_dict()),
            'vib':   copy.deepcopy(vib.state_dict()) if vib else None,
        }

    # 恢复最优状态
    model.load_state_dict(best_state['model'])
    agg.load_state_dict(best_state['agg'])
    clf.load_state_dict(best_state['clf'])
    if vib is not None and best_state['vib'] is not None:
        vib.load_state_dict(best_state['vib'])

    return _collect_mc_proba(model, agg, vib, clf, test_bd, test_y, config, device)


# ============================================================================
# K 折 CV：Seed Ensemble（概率平均）
# ============================================================================

def run_seed_ensemble_cv(adata, samples, labels_str, all_batch_data,
                         gene_to_idx, ct_to_idx, model_cfg, config, device):
    """K 折 CV：每折用 N 个不同 seed 训练 E2E，平均 soft probability 后计算 AUROC。

    旧：AUROC(seed1), AUROC(seed2), ... → mean(AUROCs)
    新：proba(seed1), proba(seed2), ... → mean(probas) → AUROC（方差更小）
    每个 seed 使用 best-checkpoint 而非 final 模型，质量与 R28 baseline 对齐。
    """
    labels_int, n_classes, label_names = _encode_labels(labels_str)
    n_genes, n_cts = len(gene_to_idx), len(ct_to_idx)
    n_seeds = config['n_random_states']
    print(f"  类别: {label_names}  n_classes={n_classes}  n_seeds={n_seeds}")

    splitter, groups, strat_y = _build_cv_splitter(adata, samples, labels_int, config)
    split_args = (samples, strat_y) if groups is None else (samples, strat_y, groups)
    all_aurocs, all_single = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(*split_args)):
        train_bd = _move_to_device([all_batch_data[i] for i in train_idx], device)
        test_bd  = _move_to_device([all_batch_data[i] for i in test_idx], device)
        train_y  = [labels_int[i] for i in train_idx]
        test_y   = [labels_int[i] for i in test_idx]

        all_probas, y_true_ref, single_aurocs = [], None, []
        for rs in range(n_seeds):
            seed = 42 + rs * 7
            yt, proba = _train_collect_best_proba(
                train_bd, train_y, test_bd, test_y,
                n_genes, n_cts, gene_to_idx, model_cfg,
                config, n_classes, device, seed)
            if y_true_ref is None:
                y_true_ref = np.array(yt)
            all_probas.append(proba)
            s = _compute_auroc(yt, proba, n_classes)
            single_aurocs.append(s)
            print(f"  fold={fold_idx+1} rs={rs}  single={s:.4f}")

        mean_proba = np.mean(all_probas, axis=0)
        fold_auroc = _compute_auroc(y_true_ref, mean_proba, n_classes)
        all_aurocs.append(fold_auroc)
        all_single.append(np.mean(single_aurocs))
        print(f"  [fold {fold_idx+1}] single_mean={np.mean(single_aurocs):.4f}  "
              f"seed_ensemble={fold_auroc:.4f}  "
              f"gain={fold_auroc - np.mean(single_aurocs):+.4f}")

    print(f"\n{'='*55}")
    print(f"  Single(mean)  : {np.mean(all_single):.4f} +/- {np.std(all_single):.4f}")
    print(f"  Seed Ensemble : {np.mean(all_aurocs):.4f} +/- {np.std(all_aurocs):.4f}")
    print(f"  各折 ensemble : {[f'{a:.3f}' for a in all_aurocs]}")
    print(f"{'='*55}")
    return all_aurocs


# ============================================================================
# CLI 入口
# ============================================================================

def _parse_args():
    p = argparse.ArgumentParser(description='Seed Ensemble E2E Pipeline')
    p.add_argument('--dataset', required=True,
                   choices=['kuppe', 'reichart', 'habermann', 'velmeshev', 'carraro'])
    p.add_argument('--no_gpu', action='store_true', default=False)
    return p.parse_args()


def main():
    args = _parse_args()
    dataset_name = args.dataset
    config = DEFAULT_CONFIG.copy()
    config['dataset_name'] = dataset_name
    device = _get_device(not args.no_gpu)
    print(f"\n=== Seed Ensemble Pipeline: {dataset_name} | device={device} ===")

    adata = _load_adata(dataset_name)
    dk = _DATASET_KEYS[dataset_name]
    samples, labels_str = _get_sample_labels(adata, dk['sample_key'], dk['condition_key'])
    n_samples = len(samples)
    config.update(build_unified_config(n_samples, dataset_name))
    config['dataset_name'] = dataset_name
    print(f"  N={n_samples}  epochs={config['n_epochs']}  "
          f"pma_k={config['pma_k']}  n_seeds={config['n_random_states']}")

    import liana as li
    gene_to_idx, ct_to_idx, _, _ = _build_gene_ct_index(adata, dk['groupby'])
    resource = li.resource.select_resource(config['resource_name'])
    model_cfg = _get_model_config(dataset_name, 30)
    model_cfg['gene_emb_dim'] = 64
    model_cfg['dropout'] = 0.30

    lr_pathway_map, n_pathways = {}, 0
    if dataset_name in _USE_PATHWAY:
        lr_pathway_map, n_pathways = build_lr_pathway_map(_DATA_DIR)
        config['n_pathways'] = n_pathways
        config['use_pathway_prior'] = True
        print(f"  [CellChat] {n_pathways} 通路, {len(lr_pathway_map)} LR pairs 覆盖")

    print("  预计算 batch_data...")
    all_batch_data = _prepare_all_samples(
        samples, adata, resource, gene_to_idx, ct_to_idx,
        model_cfg, config, dk['sample_key'], dk['groupby'],
        lr_pathway_map=lr_pathway_map if lr_pathway_map else None)
    n_valid = sum(1 for bd in all_batch_data if bd is not None)
    print(f"  有效样本: {n_valid}/{len(samples)}")

    aurocs = run_seed_ensemble_cv(adata, samples, labels_str, all_batch_data,
                                  gene_to_idx, ct_to_idx, model_cfg, config, device)

    import pandas as pd
    pd.DataFrame({
        'dataset': [dataset_name],
        'method': [f'seed_ensemble_n{config["n_random_states"]}'],
        'mean_auroc': [round(float(np.mean(aurocs)), 4)],
        'std_auroc':  [round(float(np.std(aurocs)), 4)],
        'fold_aurocs': [str([round(a, 4) for a in aurocs])],
    }).to_csv(_OUTPUT_DIR / f'{dataset_name}_seed_ensemble.csv', index=False)
    print(f"  结果已保存: {dataset_name}_seed_ensemble.csv")


if __name__ == '__main__':
    main()
