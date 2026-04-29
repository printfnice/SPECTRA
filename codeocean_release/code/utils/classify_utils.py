import os
import gc
import sys
from pathlib import Path

# 添加code目录到Python路径（如果需要）
# 对于utils模块，通常不需要添加路径，因为它们在code目录下
# 但为了保持一致性，我们添加这个
if __name__ == "__main__":
    code_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(code_dir))

import numpy as np
import pandas as pd
import scanpy as sc

import muon as mu
import liana as li

import cell2cell as c2c
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# 🌟 集成学习工具
try:
    from utils.ensemble_utils import BaggingHGATLink, FastEnsemble
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    print("⚠️ ensemble_utils未找到，集成学习功能不可用")

N_SPLITS = 3
INVERSE_FUN = lambda x: -np.log10(x)

# TODO: run method -> classify; next method (not loop over all methods)
def _dict_setup(adata, uns_key):
    adata.uns[uns_key] = dict()
    adata.uns[uns_key] = {'X': {}, 'X_0': {}, 'y_0': {}, 'dimred_extra': {}}


def _encode_y(y):
    # create a LabelEncoder object & and transform the labels
    le = LabelEncoder()
    le.fit(y)
    return le.transform(y)


def _generate_splits(n_samples, random_state, n_factors):
    
    X_dummy = np.ones((n_samples, n_factors), dtype=np.int16)
    y_dummy = np.ones(n_samples, dtype=np.int16)
    
    splits = pd.DataFrame(columns=['fold', 'train', 'test'])

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    fold = 0

    for train_index, test_index in skf.split(X_dummy, y_dummy):
        splits.loc[len(splits)] = [fold, train_index, test_index]

        print(f"{fold}: {train_index}, {test_index}")
        fold += 1
    
    # write splits
    return splits


def _get_lr_scores_matrix(lr_scores_df, sample_key, condition_key, adata=None):
    """
    将配体-受体评分DataFrame转换为矩阵形式（用于DET降维）

    Parameters
    ----------
    lr_scores_df : pd.DataFrame
        LIANA或HGATLink的输出结果，包含sample, source, target, ligand_complex, receptor_complex和score
    sample_key : str
        样本列名
    condition_key : str
        条件列名（用于生成标签）
    adata : AnnData, optional
        AnnData对象，用于从.obs中获取条件信息

    Returns
    -------
    X_lr : np.ndarray
        LR评分矩阵 (n_samples, n_lr_pairs)
    y : np.ndarray
        编码后的标签 (n_samples,)
    sample_names : list
        样本名称列表
    """
    # 获取评分列名（假设除了sample_key等元数据外，第一个数值列是评分）
    score_col = None
    for col in lr_scores_df.columns:
        if col not in [sample_key, 'source', 'target', 'ligand_complex', 'receptor_complex', condition_key]:
            if pd.api.types.is_numeric_dtype(lr_scores_df[col]):
                score_col = col
                break

    if score_col is None:
        # 如果没有找到，使用最后一列
        score_col = lr_scores_df.columns[-1]

    print(f"    使用评分列: {score_col}")

    # 创建LR pair标识
    lr_scores_df['lr_pair'] = (
        lr_scores_df['source'].astype(str) + '_' +
        lr_scores_df['ligand_complex'].astype(str) + '__' +
        lr_scores_df['target'].astype(str) + '_' +
        lr_scores_df['receptor_complex'].astype(str)
    )

    # Pivot: 行=样本, 列=LR对
    pivot_table = lr_scores_df.pivot_table(
        index=sample_key,
        columns='lr_pair',
        values=score_col,
        aggfunc='mean',  # 如果有重复，取平均值
        fill_value=0      # 填充缺失值
    )

    X_lr = pivot_table.values
    sample_names = pivot_table.index.tolist()

    # 🔧 修复：从adata.obs获取条件信息（如果提供了adata）
    if adata is not None:
        # 从adata.obs中提取样本-条件映射
        sample_condition_map = adata.obs[[sample_key, condition_key]].drop_duplicates()
        sample_condition_map = dict(zip(sample_condition_map[sample_key], sample_condition_map[condition_key]))

        # 根据sample_names获取条件
        y_raw = [sample_condition_map[s] for s in sample_names]
    else:
        # 回退方案：尝试从lr_scores_df获取（可能不存在）
        if condition_key in lr_scores_df.columns:
            sample_conditions = lr_scores_df[[sample_key, condition_key]].drop_duplicates()
            sample_conditions = sample_conditions.set_index(sample_key).loc[sample_names]
            y_raw = sample_conditions[condition_key].values
        else:
            raise ValueError(f"condition_key '{condition_key}' 不在lr_scores_df中，且未提供adata参数")

    # 编码标签
    y = _encode_y(y_raw)

    return X_lr, y, sample_names


def run_mofatalk(adata, score_key, sample_key, condition_key, dataset_name, n_factors, gpu_mode=False):

    mdata = li.multi.lrs_to_views(adata,
                                  sample_key=sample_key,
                                  score_key=score_key,
                                  inverse_fun=INVERSE_FUN,
                                  obs_keys=[condition_key], # add those to mdata.obs
                                  lr_prop = 0.33, # minimum required proportion of samples to keep an LR
                                  lrs_per_sample = 3, # minimum number of interactions to keep a sample in a specific view
                                  lrs_per_view = 15, # minimum number of interactions to keep a view
                                  samples_per_view = 5, # minimum number of samples to keep a view
                                  min_variance = 0, # minimum variance to keep an interaction
                                  lr_fill = 0, # fill missing LR values across samples with this
                                  verbose=True,
                                  uns_key=score_key
                                  ).copy()

    if dataset_name is not None:
        # 🔧 修复路径：从code/pipeline运行，data在../../data/
        outfile = os.path.join('..', '..', 'data', 'results', 'models', dataset_name, f'{score_key}.hdf5')
    else:
        outfile = None
    mu.tl.mofa(mdata,
               use_obs='union',
               convergence_mode='medium',
               n_factors=n_factors,
               outfile=outfile,
               seed=1337,
               gpu_mode=gpu_mode,
               )

    y = mdata.obs[condition_key].copy()

    # save results - extract what we need first
    factor_scores = li.ut.get_factor_scores(mdata, obsm_key='X_mofa').copy()
    X_mofa = mdata.obsm['X_mofa'].copy()

    # Delete mdata as early as possible
    del mdata
    gc.collect()

    adata.uns['mofa_res']['X'][score_key] = factor_scores

    ## create & write to dataset_name folder
    if dataset_name is not None:
        # 🔧 修复路径：从code/pipeline运行，data在../../data/
        os.makedirs(os.path.join('..', '..', 'data', 'results', 'mofa', dataset_name), exist_ok=True)
        factor_scores.to_csv(os.path.join('..', '..', 'data', 'results', 'mofa', dataset_name, f'{score_key}.csv'))

    adata.uns['mofa_res']['X_0'][score_key] = X_mofa
    adata.uns['mofa_res']['y_0'][score_key] = _encode_y(y)

    # Clean up
    del factor_scores, X_mofa, y
    gc.collect()
    


def run_tensor_c2c(adata, score_key, sample_key, condition_key, dataset_name, n_factors, use_gpu=True):
    if use_gpu:
        import tensorly as tl
        tl.set_backend('pytorch')
        device = 'cuda'
    else:
        device = 'cpu'

    # ⚠️ 关键修复：确保DataFrame格式正确
    if score_key in adata.uns:
        result_df = adata.uns[score_key]

        # 确保有必需的列
        required_cols = [sample_key, 'source', 'target', 'ligand_complex', 'receptor_complex', score_key]
        missing_cols = [col for col in required_cols if col not in result_df.columns]

        if missing_cols:
            raise ValueError(f"Result DataFrame缺少必需的列: {missing_cols}\n当前列: {result_df.columns.tolist()}")

        # 确保数据类型正确
        if score_key in result_df.columns:
            result_df[score_key] = result_df[score_key].astype(float)

        # HGATLink/MS-HGATLink归一化：根据零值比例自适应选择策略
        if score_key in ('hgatlink_score', 'ms_hgatlink_score'):
            original_len = len(result_df)
            zero_frac = (result_df[score_key] == 0).mean()
            n_ct = result_df['source'].nunique() * result_df['target'].nunique()
            use_zscore = (zero_frac > 0.6) or (score_key == 'hgatlink_score')
            print(f"\n  归一化: 零值比例={zero_frac:.1%}, 策略={'Z-score' if use_zscore else '-log10'}")
            if use_zscore:
                result_df = result_df[result_df[score_key] > 0].copy()
                print(f"    - 过滤零值: {original_len} -> {len(result_df)}")
                result_df[score_key] = result_df.groupby(sample_key)[score_key].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-10))
                mn, mx = result_df[score_key].min(), result_df[score_key].max()
                result_df[score_key] = ((result_df[score_key] - mn) / (mx - mn)) * 10
                print(f"    - Z-score + min-max 完成")

        # 更新回adata
        adata.uns[score_key] = result_df

        print(f"  ✓ 数据格式验证通过:")
        print(f"    - 形状: {result_df.shape}")
        print(f"    - 评分列: {score_key} (存在)")
        print(f"    - 样本数: {result_df[sample_key].nunique()}")
    else:
        raise ValueError(f"adata.uns中没有'{score_key}'")

    # ⚠️ 关键修复：为自定义方法(如hgatlink)注册score_key
    # LIANA的to_tensor_c2c会调用process_scores检查score_key是否在已注册方法中
    # 对于自定义方法，我们需要monkey-patch get_method_scores来添加score_key
    if score_key not in li.method.get_method_scores():
        print(f"  ⚠️  '{score_key}' 不在LIANA已注册方法中，添加临时注册...")
        original_get_method_scores = li.method.get_method_scores

        def patched_get_method_scores():
            scores = original_get_method_scores()
            # 添加自定义score_key（False表示降序，即更高的分数更好）
            if score_key not in scores:
                scores[score_key] = False  # False = descending (higher is better)
            return scores

        # 替换函数
        li.method.get_method_scores = patched_get_method_scores
        print(f"  ✓ 已临时注册 '{score_key}' (降序排列)")

    # 根据评分类型选择inverse_fun
    # Z-score归一化后的数据已经是标准化强度值，不需要-log10
    _zscore_applied = (score_key in ('hgatlink_score', 'ms_hgatlink_score')
                       and (result_df[score_key] >= 0).all()
                       and result_df[score_key].max() <= 10.01)
    if _zscore_applied:
        inverse_fun_to_use = None
        print(f"  已归一化，不使用inverse_fun")
    elif score_key in ('hgatlink_score', 'ms_hgatlink_score'):
        # MS-HGATLink 分数是强度值（可>1），-log10(>1)产生负值，tensor对负值敏感
        inverse_fun_to_use = lambda x: np.maximum(-np.log10(x), 0)
        print(f"  使用clipped -log10（负值clip为0）")
    else:
        inverse_fun_to_use = INVERSE_FUN

    tensor = li.multi.to_tensor_c2c(adata,
                                    sample_key = sample_key,
                                    inverse_fun = inverse_fun_to_use,  # 🔧 动态选择
                                    score_key = score_key,
                                    how='outer',
                                    non_expressed_fill = 0,
                                    outer_fraction = 0.15,  # ✅ 恢复到0.15（已验证最佳）
                                    uns_key=score_key
                                    )

    context_dict = adata.obs[[sample_key, condition_key]].drop_duplicates()
    context_dict = dict(zip(context_dict[sample_key], context_dict[condition_key]))
    context_dict = defaultdict(lambda: 'Unknown', context_dict)

    tensor_meta = c2c.tensor.generate_tensor_metadata(interaction_tensor=tensor,
                                                    metadata_dicts=[context_dict, None, None, None],
                                                    fill_with_order_elements=True,
                                                    )

    # 🔧 修复路径：从code/pipeline运行，data在../../data/
    output_folder = os.path.join('..', '..', 'data', 'results', 'tensor', dataset_name)
    os.makedirs(output_folder, exist_ok=True)
    tensor = c2c.analysis.run_tensor_cell2cell_pipeline(tensor,
                                                        tensor_meta,
                                                        copy_tensor=True, # Whether to output a new tensor or modifying the original
                                                        rank=n_factors,
                                                        tf_optimization='regular', # To define how robust we want the analysis to be.
                                                        random_state=1337, # Random seed for reproducibility
                                                        device=device,
                                                        elbow_metric='error',
                                                        smooth_elbow=False,
                                                        upper_rank=20,
                                                        tf_init='random',
                                                        tf_svd='numpy_svd',
                                                        cmaps=None,
                                                        sample_col='Element',
                                                        group_col='Category',
                                                        output_fig=False,
                                                        output_folder=output_folder
                                                        )

    # Extract what we need before cleaning up
    factor_scores = tensor.factors['Contexts'].join(tensor_meta[0].set_index('Element'))
    y = factor_scores['Category'].copy()
    contexts_values = tensor.factors['Contexts'].values.copy()

    # Save to CSV
    factor_scores.to_csv(os.path.join(output_folder, f'{score_key}.csv'))

    # Extract features (factor scores) before cleanup
    X_reduced = factor_scores.drop(columns=['Category']).values

    # Delete large objects
    del tensor, tensor_meta, factor_scores, y, contexts_values
    gc.collect()

    # Clear GPU cache if using GPU
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

    return X_reduced
    
    
def _run_rf_auc(X, y, train_index, test_index, n_estimators=100, use_ensemble=False, ensemble_type='fast'):
    """
    训练随机森林并评估AUROC

    Parameters
    ----------
    X : array-like
        特征矩阵
    y : array-like
        标签
    train_index : array-like
        训练集索引
    test_index : array-like
        测试集索引
    n_estimators : int
        树的数量（用于单模型）
    use_ensemble : bool
        是否使用集成学习（默认False）
    ensemble_type : str
        集成类型 ('fast' 或 'bagging')

    Returns
    -------
    auroc : float
        AUROC分数
    tpr : array
        真正率
    fpr : array
        假正率
    f1 : float
        F1分数
    oob_score : float
        袋外分数
    """
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 检查类别数量
    n_classes_train = len(np.unique(y_train))
    n_classes_test = len(np.unique(y_test))

    if n_classes_train < 2:
        print(f"    Warning: Train set only has {n_classes_train} class(es), skipping fold")
        return np.nan, np.array([]), np.array([]), np.nan, np.nan

    # 选择模型类型
    if use_ensemble and ENSEMBLE_AVAILABLE:
        # 🌟 使用集成学习
        if ensemble_type == 'fast':
            clf = FastEnsemble(n_models=3, n_estimators=100, random_state=1337, verbose=False)
        elif ensemble_type == 'bagging':
            clf = BaggingHGATLink(n_models=3, sample_ratio=0.8, random_state=1337, verbose=False)
        else:
            print(f"    Warning: Unknown ensemble_type '{ensemble_type}', using fast")
            clf = FastEnsemble(n_models=3, n_estimators=100, random_state=1337, verbose=False)

        clf.fit(X_train, y_train)

        # 计算OOB分数（集成的平均）
        if hasattr(clf, 'oob_scores') and len(clf.oob_scores) > 0:
            oob_score = np.mean(clf.oob_scores)
        else:
            oob_score = np.nan

    else:
        # 原始单模型
        if use_ensemble and not ENSEMBLE_AVAILABLE:
            print("    Warning: 集成学习不可用，使用单模型")

        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=1337, oob_score=True)
        clf.fit(X_train, y_train)
        oob_score = clf.oob_score_

    # 测试集预测
    if n_classes_test < 2:
        # Test set only has one class - cannot compute AUROC
        print(f"    Warning: Test set only has {n_classes_test} class(es), AUROC=nan")
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        return np.nan, np.array([]), np.array([]), f1, oob_score

    # 预测概率
    y_prob = clf.predict_proba(X_test)

    # Handle binary vs multi-class
    if y_prob.shape[1] == 2:
        y_prob_positive = y_prob[:, 1]
    else:
        y_prob_positive = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0]

    # 计算AUROC
    fpr, tpr, _ = roc_curve(y_test, y_prob_positive)
    auroc = auc(fpr, tpr)

    # 计算F1
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')

    return auroc, tpr, fpr, f1, oob_score


def _assign_dict(reduction_name, score_key, state, fold, auroc, tpr, fpr, f1_score, oob_score, train_split, test_split, test_classes):
    return {'reduction_name': reduction_name,
            'score_key': score_key, 
            'state': state, 
            'fold': fold,
            'auroc': auroc, 
            'tpr': tpr,
            'fpr': fpr,
            'f1_score': f1_score, 
            'oob_score': oob_score,
            'train_split': train_split, 
            'test_split': test_split, 
            'test_classes' : test_classes
            }
