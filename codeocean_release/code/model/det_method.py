"""
Deep Ensemble Transformers (DETs) for Dimensionality Reduction
================================================================
基于论文: Deep Ensemble Transformers for Dimensionality Reduction
(IEEE TNNLS, 2025)

实现三种中间层类型:
1. Forward-thinking: 只传递决策树预测
2. Input-output: 决策树预测 + 原始输入拼接
3. agBoost: AdaBoost + 梯度提升组合

适用于: 高维配体-受体相互作用数据的降维
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Literal, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class IntermediateLayer:
    """DET中间层基类"""

    def __init__(
        self,
        n_trees: int = 1000,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.trees = []
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练中间层"""
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        """转换特征"""
        raise NotImplementedError


class ForwardThinkingLayer(IntermediateLayer):
    """
    Forward-thinking层: 只传递决策树预测

    适用于: 高维稀疏数据（如基因表达、配体-受体相互作用）
    """

    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练随机森林，生成个体树预测"""
        print(f"    训练Forward-thinking层: {self.n_trees}棵树...")

        # 使用随机森林
        self.rf = RandomForestRegressor(
            n_estimators=self.n_trees,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            oob_score=True,
            verbose=0
        )

        self.rf.fit(X, y)

        # 保存各个树（用于单独预测）
        self.trees = self.rf.estimators_

        print(f"    ✓ OOB Score: {self.rf.oob_score_:.4f}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """提取每棵树的预测作为新特征"""
        # 每棵树单独预测（论文公式5）
        tree_predictions = np.array([tree.predict(X) for tree in self.trees]).T

        # 标准化
        tree_predictions = self.scaler.fit_transform(tree_predictions)

        return tree_predictions


class InputOutputLayer(IntermediateLayer):
    """
    Input-output层: 决策树预测 + 原始输入拼接

    适用于: 大部分特征都相关的高维数据（如EEG）
    """

    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练随机森林"""
        print(f"    训练Input-output层: {self.n_trees}棵树...")

        self.rf = RandomForestRegressor(
            n_estimators=self.n_trees,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            oob_score=True,
            verbose=0
        )

        self.rf.fit(X, y)
        self.trees = self.rf.estimators_

        print(f"    ✓ OOB Score: {self.rf.oob_score_:.4f}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """拼接树预测和原始输入"""
        # 树预测
        tree_predictions = np.array([tree.predict(X) for tree in self.trees]).T
        tree_predictions = self.scaler.fit_transform(tree_predictions)

        # 拼接原始输入（论文第4页描述）
        concatenated = np.concatenate([tree_predictions, X], axis=1)

        return concatenated


class AgBoostLayer(IntermediateLayer):
    """
    agBoost层: AdaBoost + 梯度提升组合

    适用于: 高度相关的高维数据（如生物医学数据）
    """

    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练AdaBoost和GradientBoosting"""
        print(f"    训练agBoost层: {self.n_trees}棵树（AdaBoost + GBM）...")

        # AdaBoost部分
        self.ada = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=self.max_depth),
            n_estimators=self.n_trees,
            random_state=self.random_state
        )

        # Gradient Boosting部分
        self.gbm = GradientBoostingRegressor(
            n_estimators=self.n_trees,
            max_depth=self.max_depth,
            random_state=self.random_state,
            verbose=0
        )

        self.ada.fit(X, y)
        self.gbm.fit(X, y)

        print(f"    ✓ AdaBoost和GBM训练完成")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """提取AdaBoost各树预测 + GBM预测"""
        # AdaBoost各树预测
        ada_predictions = []
        for estimator in self.ada.estimators_:
            pred = estimator.predict(X)
            ada_predictions.append(pred)
        ada_predictions = np.array(ada_predictions).T

        # GBM最终预测
        gbm_pred = self.gbm.predict(X).reshape(-1, 1)

        # 拼接
        combined = np.concatenate([ada_predictions, gbm_pred], axis=1)
        combined = self.scaler.fit_transform(combined)

        return combined


class ReLUNetwork(nn.Module):
    """
    前馈神经网络（带ReLU激活）

    用于处理中间层输出，生成最终降维特征
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        dropout: float = 0.2
    ):
        super(ReLUNetwork, self).__init__()

        layers = []
        prev_dim = input_dim

        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DeepEnsembleTransformer:
    """
    Deep Ensemble Transformers (DETs) 主类

    Parameters
    ----------
    layer_type : str
        中间层类型，可选 'forward', 'input_output', 'agboost'
    n_intermediate_layers : int
        中间层数量（论文推荐1-3层）
    n_trees_per_layer : int
        每层的决策树数量
    hidden_dims : list
        神经网络隐藏层维度列表
    output_dim : int
        最终降维后的特征维度
    max_depth : int, optional
        决策树最大深度
    dropout : float
        神经网络Dropout率
    learning_rate : float
        神经网络学习率
    n_epochs : int
        神经网络训练轮数
    device : str
        'cpu' 或 'cuda'
    verbose : bool
        是否打印详细信息

    Examples
    --------
    >>> det = DeepEnsembleTransformer(
    ...     layer_type='forward',
    ...     n_intermediate_layers=1,
    ...     n_trees_per_layer=1000,
    ...     hidden_dims=[128, 64],
    ...     output_dim=10
    ... )
    >>> det.fit(X_train, y_train)
    >>> X_reduced = det.transform(X_test)
    """

    def __init__(
        self,
        layer_type: Literal['forward', 'input_output', 'agboost'] = 'forward',
        n_intermediate_layers: int = 1,
        n_trees_per_layer: int = 1000,
        hidden_dims: list = [128, 64],
        output_dim: int = 10,
        max_depth: Optional[int] = None,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        n_epochs: int = 100,
        batch_size: int = 32,
        device: str = 'cpu',
        verbose: bool = True,
        random_state: int = 42
    ):
        self.layer_type = layer_type
        self.n_intermediate_layers = n_intermediate_layers
        self.n_trees_per_layer = n_trees_per_layer
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose
        self.random_state = random_state

        self.intermediate_layers = []
        self.nn_model = None
        self.input_scaler = StandardScaler()

    def _create_layer(self) -> IntermediateLayer:
        """创建中间层"""
        layer_map = {
            'forward': ForwardThinkingLayer,
            'input_output': InputOutputLayer,
            'agboost': AgBoostLayer
        }

        if self.layer_type not in layer_map:
            raise ValueError(f"未知的层类型: {self.layer_type}")

        return layer_map[self.layer_type](
            n_trees=self.n_trees_per_layer,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DeepEnsembleTransformer':
        """
        训练DET模型

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            训练特征矩阵
        y : array-like, shape (n_samples,)
            训练标签（0/1）

        Returns
        -------
        self : DeepEnsembleTransformer
            训练后的模型
        """
        if self.verbose:
            print(f"\n🌲 训练DET模型 ({self.layer_type}层)")
            print(f"  输入: {X.shape[0]}样本 × {X.shape[1]}特征")
            print(f"  中间层数: {self.n_intermediate_layers}")
            print(f"  每层树数: {self.n_trees_per_layer}")

        # 标准化输入
        X_current = self.input_scaler.fit_transform(X)

        # 步骤1-2: 训练中间层（论文Algorithm 1的步骤1-4）
        for i in range(self.n_intermediate_layers):
            if self.verbose:
                print(f"\n  中间层 {i+1}/{self.n_intermediate_layers}:")

            layer = self._create_layer()
            layer.fit(X_current, y)
            self.intermediate_layers.append(layer)

            # 转换为新特征空间
            X_current = layer.transform(X_current)

            if self.verbose:
                print(f"    输出形状: {X_current.shape}")

        # 步骤3: 训练神经网络（论文Algorithm 1的步骤5-6）
        if self.verbose:
            print(f"\n  🧠 训练前馈神经网络:")
            print(f"    隐藏层: {self.hidden_dims}")
            print(f"    输出维度: {self.output_dim}")

        input_dim = X_current.shape[1]
        self.nn_model = ReLUNetwork(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            dropout=self.dropout
        ).to(self.device)

        # 使用OLS初始化第一层权重（论文Proposition 1）
        self._initialize_with_ols(X_current, y)

        # 训练神经网络
        self._train_nn(X_current, y)

        if self.verbose:
            print(f"\n✅ DET训练完成！")

        return self

    def _initialize_with_ols(self, X: np.ndarray, y: np.ndarray):
        """使用OLS初始化神经网络权重（论文算法1）"""
        try:
            ols = LinearRegression()
            ols.fit(X, y)

            # 初始化第一层权重
            with torch.no_grad():
                first_layer = self.nn_model.network[0]
                if X.shape[1] == first_layer.weight.shape[1]:
                    # 只初始化部分权重（OLS系数对应的维度）
                    n_features = min(len(ols.coef_), first_layer.weight.shape[0])
                    first_layer.weight[:n_features, :] = torch.tensor(
                        ols.coef_.reshape(-1, 1).T,
                        dtype=torch.float32,
                        device=self.device
                    )[:n_features, :]

                    if first_layer.bias is not None:
                        first_layer.bias[0] = torch.tensor(
                            ols.intercept_,
                            dtype=torch.float32,
                            device=self.device
                        )

            if self.verbose:
                print(f"    ✓ OLS初始化完成")
        except Exception as e:
            if self.verbose:
                print(f"    ⚠️  OLS初始化失败，使用默认初始化: {e}")

    def _train_nn(self, X: np.ndarray, y: np.ndarray):
        """训练神经网络（论文公式6）"""
        # 转换为PyTorch张量
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).view(-1, 1)

        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(X)),
            shuffle=True
        )

        # 优化器和损失函数（论文公式6：MSE + L2正则）
        optimizer = optim.Adam(self.nn_model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # 训练循环
        self.nn_model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                # 前向传播
                output = self.nn_model(batch_X)

                # 计算损失（MSE + L2正则）
                mse_loss = criterion(output, batch_y)

                # L2正则化（论文公式6）
                l2_reg = 0
                for param in self.nn_model.parameters():
                    l2_reg += torch.norm(param, p=2)

                loss = mse_loss + 0.01 * l2_reg  # λ=0.01

                # 反向传播
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if self.verbose and (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"    Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.4f}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        转换数据到低维特征空间

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            输入特征矩阵

        Returns
        -------
        X_reduced : array-like, shape (n_samples, output_dim)
            降维后的特征矩阵
        """
        # 标准化
        X_current = self.input_scaler.transform(X)

        # 通过中间层
        for layer in self.intermediate_layers:
            X_current = layer.transform(X_current)

        # 通过神经网络
        self.nn_model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_current, dtype=torch.float32, device=self.device)
            X_reduced = self.nn_model(X_tensor).cpu().numpy()

        return X_reduced

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """训练并转换"""
        self.fit(X, y)
        return self.transform(X)


def det_dimensionality_reduction(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 10,
    layer_type: str = 'forward',
    n_intermediate_layers: int = 1,
    n_trees: int = 1000,
    hidden_dims: list = None,
    use_gpu: bool = False,
    verbose: bool = True,
    **kwargs
) -> np.ndarray:
    """
    DET降维的便捷接口

    Parameters
    ----------
    X : array-like
        输入特征矩阵 (n_samples, n_features)
    y : array-like
        标签 (n_samples,)
    n_components : int
        输出降维后的特征数量
    layer_type : str
        中间层类型: 'forward', 'input_output', 'agboost'
    n_intermediate_layers : int
        中间层数量
    n_trees : int
        每层的决策树数量
    hidden_dims : list, optional
        神经网络隐藏层维度，默认[128, 64]
    use_gpu : bool
        是否使用GPU
    verbose : bool
        是否打印详细信息

    Returns
    -------
    X_reduced : array-like
        降维后的特征矩阵 (n_samples, n_components)

    Examples
    --------
    >>> X_reduced = det_dimensionality_reduction(
    ...     X, y,
    ...     n_components=10,
    ...     layer_type='forward',
    ...     n_trees=1000
    ... )
    """
    if hidden_dims is None:
        hidden_dims = [128, 64]

    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

    det = DeepEnsembleTransformer(
        layer_type=layer_type,
        n_intermediate_layers=n_intermediate_layers,
        n_trees_per_layer=n_trees,
        hidden_dims=hidden_dims,
        output_dim=n_components,
        device=device,
        verbose=verbose,
        **kwargs
    )

    X_reduced = det.fit_transform(X, y)

    return X_reduced
