# checkpoint_utils.py
# 依赖: os, pickle
# 被依赖: simplified_pipeline_optimized.py
# 职责: 流水线断点续传管理（路径构建、保存、加载、跳步判断）

import os
import pickle

_CHECKPOINT_BASE = os.path.join('..', '..', 'data', 'results', 'checkpoints')


def get_checkpoint_path(dataset, method, reduction, step_name):
    """构建检查点文件路径"""
    filename = f'{dataset}_{method}_{reduction}_{step_name}.pkl'
    return os.path.join(_CHECKPOINT_BASE, filename)


def save_checkpoint(path, data, large_data=False):
    """保存检查点；large_data=True 时跳过（防止大数据集 OOM）"""
    if large_data:
        print(f"  大数据集，跳过断点保存")
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  断点已保存: {path}")


def load_checkpoint(path):
    """加载检查点，不存在时返回 None"""
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"  断点已加载: {path}")
    return data


def should_skip_step(step_num, checkpoint_path, force_rerun=None, resume=True):
    """
    判断是否跳过某步骤。

    Rules:
    - force_rerun=N: step_num >= N 时强制重跑，< N 时按断点判断
    - resume=True: 存在断点文件则跳过
    """
    if force_rerun is not None and step_num >= force_rerun:
        return False
    if resume and os.path.exists(checkpoint_path):
        return True
    return False
