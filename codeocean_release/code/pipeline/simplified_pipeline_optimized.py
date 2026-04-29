# simplified_pipeline_optimized.py
# 依赖: checkpoint_utils.py, process/processer.py, model/*_method.py, utils/classify_utils.py
# 被依赖: run_all_ms_hgatlink.py
# 职责: 4步分类流水线主入口（预处理→LR推断→LR矩阵→降维+分类）
#
# 使用方式:
#   直接运行（修改下方 DEFAULT_CONFIG）:  python simplified_pipeline_optimized.py
#   命令行参数覆盖:  python simplified_pipeline_optimized.py --dataset kuppe --method ms_hgatlink

import os
import sys
import gc
import warnings
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from datetime import datetime
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))
os.chdir(Path(__file__).parent)

from pipeline.checkpoint_utils import (
    get_checkpoint_path, save_checkpoint, load_checkpoint, should_skip_step
)

warnings.filterwarnings('ignore')

# ============================================================================
# 默认配置（直接在此修改，或通过 --arg 覆盖）
# ============================================================================
DEFAULT_CONFIG = {
    'dataset_name':           'velmeshev',
    'liana_method':           'ms_hgatlink',
    'reduction_method':       'tensor',
    'use_gpu':                True,
    'speed_mode':             'rigorous',
    'enable_adaptive_config': False,
    'enable_data_augmentation': True,
    'resume_from_checkpoint': True,
    'force_rerun_step':       None,
    'save_checkpoints':       True,
    'load_preprocessed':      True,
    'use_tabpfn':             False,   # True=用 TabPFN 替换 RF 做分类
}

SPEED_PRESETS = {
    'quick':    {'n_random_states': 2, 'n_cv_folds': 2, 'n_estimators': 50,  'n_jobs': -1},
    'balanced': {'n_random_states': 5, 'n_cv_folds': 2, 'n_estimators': 100, 'n_jobs': -1},
    'rigorous': {'n_random_states': 5, 'n_cv_folds': 2, 'n_estimators': 500, 'n_jobs': -1},
}

# 每个数据集自适应的 max_lr_pairs
_MAX_LR_BY_DATASET = {
    'velmeshev': 2000,  # 恢复2000：实验确认400更差（零值率下降导致-log10而非Z-score）
    'carraro':   800,
    'habermann': 800,
    'kuppe':     400,
    'reichart':  300,
}

# 数据集特异性随机森林参数（仅 carraro 需要更强正则化）
_RF_CONFIGS = {
    'carraro': dict(n_estimators=1000, max_depth=20,
                    min_samples_split=3, min_samples_leaf=2),
    'default': dict(n_estimators=500,  max_depth=30,
                    min_samples_split=2, min_samples_leaf=1),
}

LIANA_METHOD_MAP = None  # lazy init


# ============================================================================
# CLI 参数解析
# ============================================================================

def parse_args():
    """解析命令行参数（允许未知参数，兼容 Jupyter 环境）"""
    p = argparse.ArgumentParser(description='LR 分类流水线', add_help=False)
    p.add_argument('--dataset',  choices=['kuppe', 'reichart', 'habermann', 'velmeshev', 'carraro'])
    p.add_argument('--method',   default=None)
    p.add_argument('--reduction', choices=['tensor', 'mofa'], default=None)
    p.add_argument('--speed',    choices=['quick', 'balanced', 'rigorous'], default=None)
    p.add_argument('--no-gpu',   action='store_true')
    p.add_argument('--force-step', type=int, default=None)
    p.add_argument('--no-resume', action='store_true')
    p.add_argument('--use-tabpfn', action='store_true', help='用 TabPFN 替换 RF')
    args, _ = p.parse_known_args()
    return args


def build_run_config():
    """合并默认配置与 CLI 参数，返回运行配置 dict"""
    cfg = DEFAULT_CONFIG.copy()
    args = parse_args()
    if args.dataset:    cfg['dataset_name'] = args.dataset
    if args.method:     cfg['liana_method'] = args.method
    if args.reduction:  cfg['reduction_method'] = args.reduction
    if args.speed:      cfg['speed_mode'] = args.speed
    if args.no_gpu:     cfg['use_gpu'] = False
    if args.force_step: cfg['force_rerun_step'] = args.force_step
    if args.no_resume:  cfg['resume_from_checkpoint'] = False
    if args.use_tabpfn: cfg['use_tabpfn'] = True
    cfg.update(SPEED_PRESETS.get(cfg['speed_mode'], SPEED_PRESETS['rigorous']))
    return cfg


# ============================================================================
# 步骤 1：数据预处理
# ============================================================================

def _map_ensembl_to_symbol(adata):
    """ENSEMBL ID 转 Gene Symbol（若需要）"""
    map_path = '../../data/ensembl_to_symbol.csv'
    if not os.path.exists(map_path):
        return adata
    mapping = dict(zip(*(pd.read_csv(map_path)[col] for col in ['alias', 'gene'])))
    adata.var_names = [mapping.get(g, g) for g in adata.var_names]
    adata.var_names_make_unique()
    return adata


def _try_load_existing(cfg):
    """尝试加载已处理的数据文件，成功则返回 adata，否则返回 None"""
    ds, method = cfg['dataset_name'], cfg['liana_method']
    candidates = [
        f'../../data/interim/{ds}_with_{method}.h5ad',
        f'../../data/interim/{ds}_processed.h5ad',
        f'../../data/interim/{ds}_filtered.h5ad',
    ]
    for path in candidates:
        if os.path.exists(path):
            adata = sc.read_h5ad(path)
            print(f"  加载已有数据: {path}  [{adata.shape[0]}细胞]")
            if adata.var_names[0].startswith('ENSG'):
                adata = _map_ensembl_to_symbol(adata)
            return adata
    return None


def run_step1(cfg, handler):
    """步骤1：数据预处理（支持断点续传）"""
    ckpt = get_checkpoint_path(cfg['dataset_name'], cfg['liana_method'],
                               cfg['reduction_method'], 'step1')
    if should_skip_step(1, ckpt, cfg['force_rerun_step'], cfg['resume_from_checkpoint']):
        return load_checkpoint(ckpt)

    print("\n" + "=" * 65)
    print("  步骤 1: 数据预处理")
    print("=" * 65)

    if cfg['load_preprocessed']:
        adata = _try_load_existing(cfg)
        if adata is not None:
            large = adata.shape[0] > 500_000
            save_checkpoint(ckpt, adata, large_data=large)
            return adata

    from process.processer import DatasetHandler
    adata = DatasetHandler(cfg['dataset_name'], use_adaptive_config=False).process_dataset()
    save_checkpoint(ckpt, adata)
    return adata


# ============================================================================
# 步骤 2：LR 推断
# ============================================================================

def _get_liana_method_map():
    """懒加载 LIANA 方法映射"""
    global LIANA_METHOD_MAP
    if LIANA_METHOD_MAP is None:
        from liana.method import (cellphonedb, connectome, cellchat, scseqcomm,
                                   singlecellsignalr, natmi, logfc, rank_aggregate, geometric_mean)
        LIANA_METHOD_MAP = {
            'cellphonedb': cellphonedb, 'connectome': connectome,
            'cellchat': cellchat, 'scseqcomm': scseqcomm,
            'singlecellsignalr': singlecellsignalr, 'natmi': natmi,
            'logfc': logfc, 'rank_aggregate': rank_aggregate,
            'geometric_mean': geometric_mean,
        }
    return LIANA_METHOD_MAP


def _get_max_lr_pairs(cfg, handler):
    """根据数据集名获取 max_lr_pairs（大数据集用更多特征）"""
    n = handler.sample_key  # access to check handler is valid
    return _MAX_LR_BY_DATASET.get(cfg['dataset_name'], 400)


def _run_graph_method(cfg, adata, handler):
    """运行 HGATLink 或 MS-HGATLink"""
    method = cfg['liana_method']
    if method == 'ms_hgatlink':
        from model.ms_hgatlink_method import ms_hgatlink_inference
        infer_fn, score_key = ms_hgatlink_inference, 'ms_hgatlink_score'
    else:
        from model.hgatlink_method import hgatlink_inference
        infer_fn, score_key = hgatlink_inference, 'hgatlink_score'

    force = (cfg['force_rerun_step'] is not None and cfg['force_rerun_step'] <= 2)
    if score_key in adata.uns and not force:
        print(f"  已有 {score_key} 结果，跳过计算")
        return adata, score_key

    result = infer_fn(
        adata, groupby=handler.groupby, sample_key=handler.sample_key,
        use_raw=False, verbose=True, n_perms=None,
        max_lr_pairs=_get_max_lr_pairs(cfg, handler),
        dataset_name=cfg['dataset_name']
    )
    adata.uns[score_key] = result
    return adata, score_key


def _run_liana_standard(cfg, adata, handler):
    """运行标准 LIANA 方法"""
    method_map = _get_liana_method_map()
    name = cfg['liana_method']
    if name not in method_map:
        raise ValueError(f"未知方法: {name}，可选: {list(method_map)}")
    method = method_map[name]
    score_key = method.magnitude if method.magnitude else method.specificity
    force = (cfg['force_rerun_step'] is not None and cfg['force_rerun_step'] <= 2)
    if score_key in adata.uns and not force:
        print(f"  已有 {score_key} 结果，跳过计算")
        return adata, score_key
    result = method.by_sample(adata, groupby=handler.groupby, use_raw=False,
                               sample_key=handler.sample_key, verbose=True,
                               n_perms=None, inplace=False)
    adata.uns[score_key] = result
    return adata, score_key


def run_step2(cfg, adata, handler):
    """步骤2：LR 推断（支持断点续传）"""
    ckpt = get_checkpoint_path(cfg['dataset_name'], cfg['liana_method'],
                               cfg['reduction_method'], 'step2')
    if should_skip_step(2, ckpt, cfg['force_rerun_step'], cfg['resume_from_checkpoint']):
        data = load_checkpoint(ckpt)
        return data['adata'], data['score_key']

    print("\n" + "=" * 65)
    print(f"  步骤 2: LR 推断 - {cfg['liana_method']}")
    print("=" * 65)

    if cfg['liana_method'] in ('hgatlink', 'ms_hgatlink'):
        adata, score_key = _run_graph_method(cfg, adata, handler)
    else:
        adata, score_key = _run_liana_standard(cfg, adata, handler)

    save_checkpoint(ckpt, {'adata': adata, 'score_key': score_key})
    return adata, score_key


# ============================================================================
# 步骤 3：准备 LR 矩阵
# ============================================================================

def run_step3(cfg, adata, score_key, handler):
    """步骤3：构建样本×LR对矩阵（支持断点续传）"""
    ckpt = get_checkpoint_path(cfg['dataset_name'], cfg['liana_method'],
                               cfg['reduction_method'], 'step3')
    if should_skip_step(3, ckpt, cfg['force_rerun_step'], cfg['resume_from_checkpoint']):
        data = load_checkpoint(ckpt)
        return data['X_lr'], data['y'], data['sample_names']

    print("\n" + "=" * 65)
    print("  步骤 3: 构建 LR 评分矩阵")
    print("=" * 65)

    from utils.classify_utils import _get_lr_scores_matrix
    X_lr, y, names = _get_lr_scores_matrix(
        adata.uns[score_key], sample_key=handler.sample_key,
        condition_key=handler.condition_key, adata=adata)

    print(f"  矩阵形状: {X_lr.shape}  |  类别分布: {np.bincount(y)}")
    save_checkpoint(ckpt, {'X_lr': X_lr, 'y': y, 'sample_names': names})
    return X_lr, y, names


# ============================================================================
# 步骤 4：降维 + 分类
# ============================================================================

def _reduce_dimensions(cfg, adata, score_key, handler):
    """对整个数据集做降维（CV 外降维策略）"""
    from utils.classify_utils import run_tensor_c2c, run_mofatalk
    method = cfg['reduction_method']
    print(f"  降维方法: {method.upper()}  目标维度: {handler.n_factors}")
    if method == 'tensor':
        return run_tensor_c2c(adata, score_key=score_key, sample_key=handler.sample_key,
                              condition_key=handler.condition_key,
                              dataset_name=cfg['dataset_name'],
                              n_factors=handler.n_factors, use_gpu=cfg['use_gpu'])
    if method == 'mofa':
        from utils.classify_utils import _dict_setup
        _dict_setup(adata, 'mofa_res')
        run_mofatalk(adata, score_key=score_key, sample_key=handler.sample_key,
                     condition_key=handler.condition_key,
                     dataset_name=cfg['dataset_name'],
                     n_factors=handler.n_factors, gpu_mode=cfg['use_gpu'])
        return adata.uns['mofa_res']['X_0'][score_key]
    raise ValueError(f"不支持的降维方法: {method}")


def _build_classifier(cfg):
    """构建分类器（TabPFN 或数据集特异性 RF）"""
    if cfg.get('use_tabpfn', False):
        from tabpfn import TabPFNClassifier
        print("  分类器: TabPFN")
        return TabPFNClassifier()
    from sklearn.ensemble import RandomForestClassifier
    rf_kw = _RF_CONFIGS.get(cfg['dataset_name'], _RF_CONFIGS['default'])
    return RandomForestClassifier(
        **rf_kw, max_features='sqrt', random_state=1337,
        oob_score=True, n_jobs=cfg['n_jobs'], class_weight='balanced'
    )


def _cv_classify(clf, X, y, cfg):
    """交叉验证分类，返回预测概率"""
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    cv = StratifiedKFold(n_splits=cfg['n_cv_folds'], shuffle=True, random_state=42)
    n_jobs = 1 if cfg.get('use_tabpfn', False) else -1
    return cross_val_predict(clf, X, y, cv=cv, method='predict_proba', n_jobs=n_jobs)


def _compute_metrics(y, y_pred_proba, clf, X):
    """计算 AUROC、F1、精确率、召回率、OOB 分数"""
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    clf.fit(X, y)
    auroc = roc_auc_score(y, y_pred_proba[:, 1]) if y_pred_proba.shape[1] == 2 else 0.5
    y_pred = clf.predict(X)
    return {
        'auroc':     auroc,
        'f1_score':  f1_score(y, y_pred, average='weighted'),
        'precision': precision_score(y, y_pred, average='weighted'),
        'recall':    recall_score(y, y_pred, average='weighted'),
        'oob_score': getattr(clf, 'oob_score_', 0.0),
    }


def _ensure_method_registered(method_name):
    """确保自定义方法已注册到 LIANA（断点恢复跳过 step2 时需要手动触发导入）"""
    if method_name == 'ms_hgatlink':
        from model.ms_hgatlink_method import ms_hgatlink  # noqa: F401
    elif method_name == 'hgatlink':
        from model.hgatlink_method import hgatlink  # noqa: F401


def run_step4(cfg, adata, score_key, X_lr, y, handler):
    """步骤4：CV 外降维 + 随机森林分类"""
    print("\n" + "=" * 65)
    print("  步骤 4: 降维 + 分类评估（CV 外降维）")
    print("=" * 65)

    _ensure_method_registered(cfg['liana_method'])
    X_reduced = _reduce_dimensions(cfg, adata, score_key, handler)
    print(f"  降维完成: {X_lr.shape} → {X_reduced.shape}")

    clf = _build_classifier(cfg)
    y_pred_proba = _cv_classify(clf, X_reduced, y, cfg)
    metrics = _compute_metrics(y, y_pred_proba, clf, X_reduced)

    del adata
    gc.collect()

    result_row = {
        'dataset': cfg['dataset_name'], 'method': cfg['liana_method'],
        'reduction': cfg['reduction_method'], 'n_factors': handler.n_factors,
        'n_samples': len(y), 'n_features_original': X_lr.shape[1],
        'n_features_reduced': X_reduced.shape[1],
    }
    result_row.update(metrics)
    return pd.DataFrame([result_row])


# ============================================================================
# 结果保存
# ============================================================================

def _save_results(cfg, evaluate_df):
    """保存结果到 output/ 和 data/results/"""
    ds, method, red = cfg['dataset_name'], cfg['liana_method'], cfg['reduction_method']
    os.makedirs('output', exist_ok=True)
    out_file = f'output/{ds}_{method}_{red}.csv'
    evaluate_df.to_csv(out_file, index=False)

    results_dir = '../../data/results'
    os.makedirs(results_dir, exist_ok=True)
    results_file = f'{results_dir}/{ds}.csv'

    if os.path.exists(results_file):
        existing = pd.read_csv(results_file)
        mask = ~((existing.get('score_key') == method) &
                 (existing.get('reduction_name') == red))
        evaluate_df = pd.concat([existing[mask], evaluate_df], ignore_index=True)
    evaluate_df.to_csv(results_file, index=False)
    return out_file


def _print_summary(evaluate_df, elapsed):
    """打印结果汇总"""
    print("\n" + "=" * 65)
    print("  结果汇总")
    print("=" * 65)
    valid_auroc = evaluate_df['auroc'].dropna()
    valid_f1 = evaluate_df['f1_score'].dropna()
    print(f"  平均 AUROC: {valid_auroc.mean():.4f} ± {valid_auroc.std():.4f}")
    print(f"  平均 F1:    {valid_f1.mean():.4f} ± {valid_f1.std():.4f}")
    if 'oob_score' in evaluate_df.columns:
        print(f"  OOB score: {evaluate_df['oob_score'].mean():.4f}")
    print(f"  总耗时: {elapsed:.1f}s ({elapsed/60:.1f} 分钟)")


# ============================================================================
# 主函数
# ============================================================================

def _setup_dirs():
    """创建必要目录"""
    os.makedirs('../../data/interim', exist_ok=True)
    os.makedirs('../../data/results/checkpoints', exist_ok=True)
    os.makedirs('output', exist_ok=True)


def _print_config(cfg):
    """打印当前运行配置"""
    print("\n" + "=" * 65)
    print("  4步分类流水线")
    print("=" * 65)
    print(f"  数据集: {cfg['dataset_name']}  方法: {cfg['liana_method']}  "
          f"降维: {cfg['reduction_method']}")
    print(f"  速度模式: {cfg['speed_mode']}  GPU: {cfg['use_gpu']}  "
          f"断点续传: {cfg['resume_from_checkpoint']}")
    if cfg['force_rerun_step']:
        print(f"  强制重跑步骤: {cfg['force_rerun_step']}")


def _flush_gpu():
    """释放 GPU 缓存"""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    start = datetime.now()
    cfg = build_run_config()
    _setup_dirs()
    _print_config(cfg)

    from process.processer import DatasetHandler
    handler = DatasetHandler(cfg['dataset_name'], use_adaptive_config=False)

    # 智能断点加载：优先从 step3+step2 联合恢复
    step3_ckpt = get_checkpoint_path(cfg['dataset_name'], cfg['liana_method'],
                                     cfg['reduction_method'], 'step3')
    step2_ckpt = get_checkpoint_path(cfg['dataset_name'], cfg['liana_method'],
                                     cfg['reduction_method'], 'step2')

    adata, score_key, X_lr, y = None, None, None, None

    if should_skip_step(3, step3_ckpt, cfg['force_rerun_step'], cfg['resume_from_checkpoint']):
        data3 = load_checkpoint(step3_ckpt)
        data2 = load_checkpoint(step2_ckpt)
        if data2:
            X_lr, y = data3['X_lr'], data3['y']
            adata, score_key = data2['adata'], data2['score_key']
            print("  从 step3 断点恢复，跳过步骤1-3")
        else:
            print("  step3 断点存在但 step2 缺失，回退到正常路径")

    if adata is None:
        adata = run_step1(cfg, handler)
        gc.collect()
        adata, score_key = run_step2(cfg, adata, handler)
        gc.collect()
    if X_lr is None:
        X_lr, y, _ = run_step3(cfg, adata, score_key, handler)

    _flush_gpu()

    evaluate_df = run_step4(cfg, adata, score_key, X_lr, y, handler)

    out_file = _save_results(cfg, evaluate_df)
    elapsed = (datetime.now() - start).total_seconds()
    _print_summary(evaluate_df, elapsed)
    print(f"\n  结果已保存: {out_file}")
    return evaluate_df


if __name__ == '__main__':
    results = main()
