#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 设置LOKY_MAX_CPU_COUNT环境变量以避免CPU核心检测问题
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '8'  # 将数字调整为你想使用的核心数

"""
AutoMatFlow.py
--------------------------


GPU加速说明:
- 本脚本支持使用GPU加速模型训练，但需要满足以下条件：
  1. 硬件要求：安装了NVIDIA GPU和相应的驱动程序
  2. CUDA要求：安装了与所用库兼容的CUDA版本
  3. 软件要求：
     - 对于XGBoost：安装支持GPU的版本 (pip install xgboost --upgrade)
     - 对于LightGBM：需要特别编译支持GPU的版本 
     - 对于CatBoost：GPU版本开箱即用
     - 对于TabPFN：需要安装支持CUDA的PyTorch
  4. 注意事项：
     - 大多数scikit-learn模型（如RandomForest、SVM等）不支持GPU加速
     - TabPFN会自动使用可用的GPU
     - GPU加速可以通过配置项"use_gpu"启用或禁用

  若使用GPU后出现问题，可将配置中的"use_gpu"设置为False以使用CPU模式

数据预处理选项:
- 本脚本现在支持在数据预处理阶段为数值特征添加高斯噪声，通过以下配置项控制：
  - "add_gaussian_noise": 布尔值，控制是否启用噪声添加功能
  - "gaussian_noise_scale": 浮点数，表示噪声标准差与特征标准差的比例
  这一功能可以增强模型对于数据的鲁棒性，特别是在处理过拟合问题时
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
import shap
import warnings
import re
import os
from datetime import datetime
import random
import optuna
from optuna.pruners import MedianPruner
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
import inspect
import time
import gc
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import clone
from sklearn.base import BaseEstimator, TransformerMixin
# GPU配置
GPU_CONFIG = {
    "use_gpu": True,  # 是否尝试使用GPU加速, 当使用TabPFN时候建议使用 GPU显著增加速度
}

# 噪声配置
NOISE_CONFIG = {
    "add_gaussian_noise": False,  # 是否在数据预处理时为数值特征添加高斯噪声
    "gaussian_noise_scale": 0.05,  # 噪声标准差与特征标准差的比例
}
# ===================== 配置参数（请在此处修改配置） =====================
# 数据配置
DATA_CONFIG = {
    "data_file": "1234.xlsx",  # 数据文件名（支持.xlsx, .xls, .csv格式）
    "file_type": None,  # 可选: "excel", "csv"，若为None则自动根据文件后缀判断
    "sheet_name": 0,  # Excel工作表名称或索引（只对Excel有效）
    "target_column": -1,  # 目标变量列索引，默认最后一列
    "header": 0,  # 数据头所在行号，None表示无头
    "csv_separator": ",",  # CSV分隔符（只对CSV有效）
    "encoding": "utf-8",  # 文件编码
}

# 基本配置
BASIC_CONFIG = {
    "task_type": "regression",  # 任务类型: "regression" 或 "classification"
    "test_size": 0.2,  # 测试集比例
    "random_state": 42,  # 随机种子
    "results_dir": None,  # 结果保存目录，None则自动创建
}

# 模型训练与评估配置
MODEL_CONFIG = {
    "cv": 2,  # 交叉验证折数
    "cv_outer": 2,  # 嵌套交叉验证-外层折数
    "cv_inner": 2,  # 嵌套交叉验证-内层折数
    "permutation_repeats": 1, #置换重要性评估重复次数（如果有）
    "n_jobs": -1,  # 并行任务数（-1表示使用所有CPU核心）
    "bayes_iter": 50,  # 贝叶斯优化迭代次数
    "memory_threshold": 80,  # 内存使用阈值（百分比）
    "memory_warning": 70,  # 内存警告阈值（百分比）
}

# 特征选择配置
FEATURE_CONFIG = {
    "importance_threshold": 0.05,  # 特征重要性过滤阈值，保留前百分之20%的特征
    "correlation_threshold": 0.95,  # 特征相关性过滤阈值
}

# SHAP分析配置
SHAP_CONFIG = {
    "shap_sample_size": 200,  # SHAP分析样本数量
    "shap_plot_type": "all",  # SHAP可视化类型：'all', 'basic', 'advanced'
    "shap_dependency_plot_max_features": 5,  # 依赖图最大特征数
    "shap_interaction_max_features": 10,  # 交互图最大特征数
}

# 图形配置
PLOT_CONFIG = {
    "dpi": 300,  # 图像DPI
    "fig_width": 12,  # 图像宽度
    "fig_height": 8,  # 图像高度
    "font_family": "DejaVu Sans",  # 字体
    "save_format": "png",  # 图像保存格式
}



# 其他配置（合并到主配置中）
config = {
    **DATA_CONFIG,
    **BASIC_CONFIG,
    **MODEL_CONFIG,
    **FEATURE_CONFIG,
    **SHAP_CONFIG,
    **PLOT_CONFIG,
    **GPU_CONFIG,
    **NOISE_CONFIG,
    # 特征类型配置
    # 可选项：填写用数字表示类别的列名列表。
    # 未在此列表中的列将被自动视为数值列（进行填充和缩放）。
    # 如果列表为空或不提供，所有列都按数值列处理。
    "categorical_num_cols": [],  # 例如 ["是否热处理"]
    # 遗传算法配置
    "genetic_population_size": 10,  # 种群大小
    "genetic_generations": 2,  # 最大代数
    "genetic_initial_rate": 0.5,  # 初始特征选择率
    "genetic_tournament_size": 3,  # 锦标赛选择大小
    "genetic_initial_crossover_rate": 0.8,  # 初始交叉率
    "genetic_initial_mutation_rate": 0.2,  # 初始变异率
    "genetic_min_mutation_rate": 0.05,  # 最小变异率
    "genetic_max_mutation_rate": 0.3,  # 最大变异率
    "genetic_min_crossover_rate": 0.5,  # 最小交叉率
    "genetic_max_crossover_rate": 0.9  # 最大交叉率
}


# 配置管理函数
def update_config(**kwargs):
    """
    更新全局配置

    参数:
        **kwargs: 要更新的配置参数
    """
    for key, value in kwargs.items():
        if key in config:
            config[key] = value
            print(f"已更新配置: {key} = {value}")
        else:
            print(f"警告: 配置项 {key} 不存在，将被忽略")


def get_config(key, default=None):
    """
    获取配置值，如果不存在则返回默认值

    参数:
        key: 配置键名
        default: 默认值

    返回:
        配置值或默认值
    """
    return config.get(key, default)


def validate_config():
    """
    验证配置是否有效，检查必需的参数是否存在，
    如果存在问题，返回错误信息列表

    返回:
        errors: 错误信息列表，如果没有错误则为空列表
    """
    errors = []

    # 必需的配置项
    required_configs = [
        "data_file",
        "task_type",
        "test_size",
        "random_state",
        "cv",
        "n_jobs",
        "bayes_iter",
        "importance_threshold",
        "correlation_threshold"
    ]

    # 检查必需的配置项
    for key in required_configs:
        if key not in config:
            errors.append(f"缺少必需的配置项: {key}")

    # 检查值范围
    if "test_size" in config and (config["test_size"] <= 0 or config["test_size"] >= 1):
        errors.append(f"test_size必须在0和1之间，当前值为: {config['test_size']}")

    if "importance_threshold" in config and config["importance_threshold"] < 0:
        errors.append(f"importance_threshold不能为负数，当前值为: {config['importance_threshold']}")

    if "correlation_threshold" in config and (
            config["correlation_threshold"] < 0 or config["correlation_threshold"] > 1):
        errors.append(f"correlation_threshold必须在0和1之间，当前值为: {config['correlation_threshold']}")

    if "task_type" in config and config["task_type"] not in ["regression", "classification"]:
        errors.append(f"task_type必须为'regression'或'classification'，当前值为: {config['task_type']}")

    # 验证遗传算法参数
    if "genetic_population_size" in config and config["genetic_population_size"] <= 0:
        errors.append(f"genetic_population_size必须大于0，当前值为: {config['genetic_population_size']}")

    if "genetic_generations" in config and config["genetic_generations"] <= 0:
        errors.append(f"genetic_generations必须大于0，当前值为: {config['genetic_generations']}")

    if "genetic_initial_rate" in config and (config["genetic_initial_rate"] <= 0 or config["genetic_initial_rate"] > 1):
        errors.append(f"genetic_initial_rate必须在0和1之间，当前值为: {config['genetic_initial_rate']}")

    if "genetic_tournament_size" in config and config["genetic_tournament_size"] <= 1:
        errors.append(f"genetic_tournament_size必须大于1，当前值为: {config['genetic_tournament_size']}")

    if "genetic_initial_crossover_rate" in config and (
            config["genetic_initial_crossover_rate"] <= 0 or config["genetic_initial_crossover_rate"] > 1):
        errors.append(
            f"genetic_initial_crossover_rate必须在0和1之间，当前值为: {config['genetic_initial_crossover_rate']}")

    if "genetic_initial_mutation_rate" in config and (
            config["genetic_initial_mutation_rate"] <= 0 or config["genetic_initial_mutation_rate"] > 1):
        errors.append(
            f"genetic_initial_mutation_rate必须在0和1之间，当前值为: {config['genetic_initial_mutation_rate']}")

    if "genetic_min_mutation_rate" in config and (
            config["genetic_min_mutation_rate"] <= 0 or config["genetic_min_mutation_rate"] > 1):
        errors.append(f"genetic_min_mutation_rate必须在0和1之间，当前值为: {config['genetic_min_mutation_rate']}")

    if "genetic_max_mutation_rate" in config and (
            config["genetic_max_mutation_rate"] <= 0 or config["genetic_max_mutation_rate"] > 1):
        errors.append(f"genetic_max_mutation_rate必须在0和1之间，当前值为: {config['genetic_max_mutation_rate']}")

    if "genetic_min_crossover_rate" in config and (
            config["genetic_min_crossover_rate"] <= 0 or config["genetic_min_crossover_rate"] > 1):
        errors.append(f"genetic_min_crossover_rate必须在0和1之间，当前值为: {config['genetic_min_crossover_rate']}")

    if "genetic_max_crossover_rate" in config and (
            config["genetic_max_crossover_rate"] <= 0 or config["genetic_max_crossover_rate"] > 1):
        errors.append(f"genetic_max_crossover_rate必须在0和1之间，当前值为: {config['genetic_max_crossover_rate']}")

    # 验证遗传算法参数的逻辑关系
    if all(key in config for key in ["genetic_min_mutation_rate", "genetic_max_mutation_rate"]):
        if config["genetic_min_mutation_rate"] >= config["genetic_max_mutation_rate"]:
            errors.append(f"genetic_min_mutation_rate必须小于genetic_max_mutation_rate")

    if all(key in config for key in ["genetic_min_crossover_rate", "genetic_max_crossover_rate"]):
        if config["genetic_min_crossover_rate"] >= config["genetic_max_crossover_rate"]:
            errors.append(f"genetic_min_crossover_rate必须小于genetic_max_crossover_rate")

    # 验证噪声参数
    if "add_gaussian_noise" in config and not isinstance(config["add_gaussian_noise"], bool):
        errors.append(f"add_gaussian_noise必须是布尔值，当前值为: {config['add_gaussian_noise']}")
        
    if "gaussian_noise_scale" in config and (not isinstance(config["gaussian_noise_scale"], (int, float)) or config["gaussian_noise_scale"] < 0):
        errors.append(f"gaussian_noise_scale必须是非负浮点数，当前值为: {config['gaussian_noise_scale']}")

    return errors


# 如果不需要交互式窗口，可改为Agg后端，避免Tkinter问题
plt.switch_backend('Agg')

# ===================== 导入所需库 =====================
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GroupKFold, \
    cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, \
    classification_report, roc_auc_score
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import clone

from joblib import Parallel, delayed, dump
import psutil
import seaborn as sns
from scipy.special import comb  # 用于计算组合数

# 定义高斯噪声添加器
class GaussianNoiseAdder(BaseEstimator, TransformerMixin):
    """
    为数值特征添加高斯噪声的自定义转换器
    
    参数:
    ----------
    scale : float
        噪声的标准差与特征标准差的比例
    random_state : int, optional
        随机数生成器的种子
    """
    def __init__(self, scale, random_state=None):
        self.scale = scale
        self.random_state = random_state
        self.apply_on_training_only = True
        self._is_fitting = False
        
    def fit(self, X, y=None):
        """
        计算并存储每个特征的标准差
        
        参数:
        ----------
        X : array-like
            输入特征矩阵
        y : array-like, optional
            目标变量，不使用但为了兼容性保留

        返回:
        ----------
        self
        """
        # 将X转换为NumPy数组以确保一致性
        X_arr = np.asarray(X)
        
        # 计算每列的标准差
        self.feature_stds_ = np.std(X_arr, axis=0)
        
        # 处理标准差为零的情况
        self.feature_stds_[self.feature_stds_ == 0] = 1.0
        
        # 初始化随机数生成器
        self.random_generator_ = np.random.RandomState(self.random_state)
        
        # 设置拟合标志
        self._is_fitting = True
        
        return self
        
    def transform(self, X):
        """
        添加高斯噪声到输入特征
        
        参数:
        ----------
        X : array-like
            输入特征矩阵

        返回:
        ----------
        X_noisy : array-like
            添加噪声后的特征矩阵
        """
        # 确保X是NumPy数组并创建副本
        X_copy = np.asarray(X).copy()
        
        # 检查噪声比例是否大于0
        if self.scale > 0:
            # 判断是否需要添加噪声
            if self.apply_on_training_only and hasattr(self, '_is_fitting') and self._is_fitting:
                # 基于特征标准差和scale生成噪声
                noise = self.random_generator_.normal(
                    0, 
                    self.scale * self.feature_stds_, 
                    size=X_copy.shape
                )
                # 添加噪声
                X_copy += noise
                
        # 重置拟合标志
        if hasattr(self, '_is_fitting'):
            self._is_fitting = False
            
        return X_copy
        
    def fit_transform(self, X, y=None, **fit_params):
        """
        拟合并转换数据
        
        参数:
        ----------
        X : array-like
            输入特征矩阵
        y : array-like, optional
            目标变量，不使用但为了兼容性保留

        返回:
        ----------
        X_noisy : array-like
            添加噪声后的特征矩阵
        """
        return self.fit(X, y).transform(X)

# 尝试导入可选模型
try:
    import lightgbm as lgb

    LGBMRegressor = lgb.LGBMRegressor
    LGBMClassifier = lgb.LGBMClassifier
except ImportError:
    LGBMRegressor = None
    LGBMClassifier = None
    print("LightGBM models are not available. Please install lightgbm package.")

try:
    import catboost as cb

    CatBoostRegressor = cb.CatBoostRegressor
    CatBoostClassifier = cb.CatBoostClassifier
except ImportError:
    CatBoostRegressor = None
    CatBoostClassifier = None
    print("CatBoost models are not available. Please install catboost package.")

try:
    from tabpfn import TabPFNRegressor
    from tabpfn import TabPFNClassifier
except ImportError:
    TabPFNRegressor = None
    TabPFNClassifier = None
    print("TabPFN models are not available. Please install the tabpfn package.")

# 如果TabPFN可用，创建带有进度条的包装类
if TabPFNRegressor is not None and TabPFNClassifier is not None:
    # 保存原始类引用
    OriginalTabPFNRegressor = TabPFNRegressor
    OriginalTabPFNClassifier = TabPFNClassifier
    
    # 创建带有简单进度条的TabPFNRegressor包装类
    class ProgressTabPFNRegressor(OriginalTabPFNRegressor):
        def fit(self, X, y):
            """添加简单进度条的TabPFNRegressor拟合方法"""
            print("\n开始TabPFN回归模型训练（估计需要1-2分钟）...")
            
            # 创建简单进度条 - 不尝试修改内部方法
            progress_bar = tqdm(total=10, desc="TabPFN训练进度估计", 
                               bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}")
            
            # 使用定时器更新进度条
            def update_progress():
                for i in range(10):
                    time.sleep(6)  # 大约60秒，每6秒更新一次
                    progress_bar.update(1)
            
            # 在后台线程中运行定时器
            import threading
            timer_thread = threading.Thread(target=update_progress)
            timer_thread.daemon = True  # 设为守护线程，不阻塞主程序退出
            timer_thread.start()
            
            try:
                # 使用单线程模式避免权限问题
                import os
                old_n_jobs = os.environ.get('JOBLIB_NUM_THREADS', None)
                os.environ['JOBLIB_NUM_THREADS'] = '1'  # 强制joblib使用单线程
                
                # 调用原始的fit方法
                result = super().fit(X, y)
                
                # 恢复原始设置
                if old_n_jobs is not None:
                    os.environ['JOBLIB_NUM_THREADS'] = old_n_jobs
                else:
                    os.environ.pop('JOBLIB_NUM_THREADS', None)
                
                # 关闭进度条
                progress_bar.close()
                print("TabPFN回归模型训练完成!")
                
                # 清理内存
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return result
                
            except Exception as e:
                progress_bar.close()
                print(f"TabPFN训练错误: {str(e)}")
                
                # 清理环境变量
                if old_n_jobs is not None:
                    os.environ['JOBLIB_NUM_THREADS'] = old_n_jobs
                else:
                    os.environ.pop('JOBLIB_NUM_THREADS', None)
                    
                # 尝试正常训练作为回退
                try:
                    return super().fit(X, y)
                except Exception as inner_e:
                    print(f"无法训练TabPFN模型: {str(inner_e)}")
                    raise
    
    # 创建带有简单进度条的TabPFNClassifier包装类
    class ProgressTabPFNClassifier(OriginalTabPFNClassifier):
        def fit(self, X, y):
            """添加简单进度条的TabPFNClassifier拟合方法"""
            print("\n开始TabPFN分类模型训练（估计需要1-2分钟）...")
            
            # 创建简单进度条 - 不尝试修改内部方法
            progress_bar = tqdm(total=10, desc="TabPFN训练进度估计", 
                              bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}")
            
            # 使用定时器更新进度条
            def update_progress():
                for i in range(10):
                    time.sleep(6)  # 大约60秒，每6秒更新一次
                    progress_bar.update(1)
            
            # 在后台线程中运行定时器
            import threading
            timer_thread = threading.Thread(target=update_progress)
            timer_thread.daemon = True  # 设为守护线程，不阻塞主程序退出
            timer_thread.start()
            
            try:
                # 使用单线程模式避免权限问题
                import os
                old_n_jobs = os.environ.get('JOBLIB_NUM_THREADS', None)
                os.environ['JOBLIB_NUM_THREADS'] = '1'  # 强制joblib使用单线程
                
                # 调用原始的fit方法
                result = super().fit(X, y)
                
                # 恢复原始设置
                if old_n_jobs is not None:
                    os.environ['JOBLIB_NUM_THREADS'] = old_n_jobs
                else:
                    os.environ.pop('JOBLIB_NUM_THREADS', None)
                
                # 关闭进度条
                progress_bar.close()
                print("TabPFN分类模型训练完成!")
                
                # 清理内存
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return result
                
            except Exception as e:
                progress_bar.close()
                print(f"TabPFN训练错误: {str(e)}")
                
                # 清理环境变量
                if old_n_jobs is not None:
                    os.environ['JOBLIB_NUM_THREADS'] = old_n_jobs
                else:
                    os.environ.pop('JOBLIB_NUM_THREADS', None)
                    
                # 尝试正常训练作为回退
                try:
                    return super().fit(X, y)
                except Exception as inner_e:
                    print(f"无法训练TabPFN模型: {str(inner_e)}")
                    raise

try:
    from xgboost import XGBRegressor, XGBClassifier
except ImportError:
    XGBRegressor = None
    XGBClassifier = None
    print("XGBoost models are not available. Please install xgboost package.")


# ===================== 内存检查函数 =====================
def check_memory_usage(threshold=None, warning_threshold=None):
    """检查内存使用情况并提供警告"""
    # 使用最新的配置值
    threshold = threshold if threshold is not None else get_config("memory_threshold")
    warning_threshold = warning_threshold if warning_threshold is not None else get_config("memory_warning")

    memory_percent = psutil.virtual_memory().percent
    if memory_percent > threshold:
        print(f"Critical Warning: Memory usage at {memory_percent}%, exceeding danger threshold {threshold}%!")
        print("Please close other applications or reduce feature combination evaluation scope.")
        return True
    elif memory_percent > warning_threshold:
        print(f"Warning: Memory usage at {memory_percent}%, exceeding warning threshold {warning_threshold}%!")
        print("Consider closing unnecessary applications.")
    return False


# ===================== 超参数建议函数 =====================
def suggest_params(trial, param_space):
    # 保持原函数不变
    params = {}
    for key, val in param_space.items():
        if isinstance(val, tuple):
            if len(val) == 2:
                if isinstance(val[0], int) and isinstance(val[1], int):
                    params[key] = trial.suggest_int(key, val[0], val[1])
                else:
                    params[key] = trial.suggest_float(key, val[0], val[1])
            elif len(val) == 3:
                low, high, dist = val
                if dist == 'log-uniform':
                    params[key] = trial.suggest_float(key, low, high, log=True)
                else:
                    params[key] = trial.suggest_float(key, low, high)
            else:
                raise ValueError(f"Unsupported tuple length for parameter {key}")
        elif isinstance(val, list):
            params[key] = trial.suggest_categorical(key, val)
        else:
            raise ValueError(f"Unsupported parameter type for {key}")
    return params


# ===================== 数据读取与预处理函数 =====================
def read_data(file_path=None, task_type=None, sheet_name=None, target_column=None, header=None,
              csv_separator=None, encoding=None, file_type=None):
    """
    读取并预处理数据，支持Excel和CSV格式

    参数:
        file_path (str): 数据文件路径，None则使用配置中的路径
        task_type (str): 任务类型，'regression'或'classification'，None则使用配置中的类型
        sheet_name (int/str): Excel工作表名称或索引，None则使用配置中的值
        target_column (int): 目标变量列索引，None则使用配置中的值
        header (int): 表头行索引，None则使用配置中的值
        csv_separator (str): CSV分隔符，None则使用配置中的值
        encoding (str): 文件编码，None则使用配置中的值
        file_type (str): 文件类型，'excel'或'csv'，None则自动判断

    返回:
        X (DataFrame): 特征变量
        y (Series): 目标变量
    """
    # 使用配置中的默认值
    file_path = file_path or get_config("data_file")
    task_type = task_type or get_config("task_type")
    sheet_name = sheet_name if sheet_name is not None else get_config("sheet_name")
    target_column = target_column if target_column is not None else get_config("target_column")
    header = header if header is not None else get_config("header")
    csv_separator = csv_separator if csv_separator is not None else get_config("csv_separator")
    encoding = encoding if encoding is not None else get_config("encoding")
    file_type = file_type or get_config("file_type")

    print(f"正在读取数据文件: {file_path}")

    # 如果文件类型未指定，从文件扩展名判断
    if file_type is None:
        if file_path.lower().endswith(('.xlsx', '.xls')):
            file_type = 'excel'
        elif file_path.lower().endswith('.csv'):
            file_type = 'csv'
        else:
            raise ValueError(f"无法从文件扩展名判断文件类型: {file_path}，请显式指定file_type='excel'或'csv'")

    # 根据文件类型读取数据
    if file_type.lower() == 'excel':
        print(f"读取Excel文件，工作表: {sheet_name}")
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header, engine='openpyxl')
    elif file_type.lower() == 'csv':
        print(f"读取CSV文件，分隔符: {csv_separator}")
        df = pd.read_csv(file_path, sep=csv_separator, header=header, encoding=encoding)
    else:
        raise ValueError(f"不支持的文件类型: {file_type}，请使用'excel'或'csv'")

    # 检测缺失值，按列均值填充（仅对数值型有效）
    if df.isnull().values.any():
        print("检测到缺失值，将在模型管道中处理...")

    # 数据基本信息
    n_features = df.shape[1] - 1 if target_column == -1 else df.shape[1] - 1
    n_samples = df.shape[0]
    print(f"数据维度: {df.shape}, 特征数: {n_features}, 样本数: {n_samples}")

    # 分离特征和目标变量
    if target_column == -1:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    else:
        X = df.drop(df.columns[target_column], axis=1)
        y = df.iloc[:, target_column]

    # 显示特征名
    print(f"特征列表: {X.columns.tolist()}")
    print(f"目标变量: {df.columns[target_column if target_column != -1 else -1]}")

    # 分类任务需要对标签进行编码
    if task_type == "classification":
        if not np.issubdtype(y.dtype, np.number) or (len(np.unique(y)) < 10 and len(y) / len(np.unique(y)) > 5):
            print("检测到分类任务，正在对标签进行编码...")
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            print(f"类别映射: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
            print(f"检测到 {len(label_encoder.classes_)} 个类别")

    return X, y


# ===================== 创建预处理器函数 =====================
def create_preprocessor(numerical_cols, categorical_num_cols):
    """
    创建用于预处理数据的ColumnTransformer

    参数:
    numerical_cols: 真正的数值型特征列名（将被标准化处理）
    categorical_num_cols: 用数字表示类别的特征列名（将被独热编码处理）

    返回:
    preprocessor: ColumnTransformer对象，用于数据预处理
    """
    # 获取噪声配置
    add_noise = get_config("add_gaussian_noise", False)
    noise_scale = get_config("gaussian_noise_scale", 0.05)
    random_state = get_config("random_state", 42)
    
    # 数值型特征的转换器
    numeric_pipeline_steps = [('imputer', SimpleImputer(strategy='mean'))]
    
    # 只有当需要添加噪声时才添加GaussianNoiseAdder步骤
    if add_noise and noise_scale > 0:
        numeric_pipeline_steps.append(
            ('noise', GaussianNoiseAdder(scale=noise_scale, random_state=random_state))
        )
        print(f"配置：为数值特征添加高斯噪声 (scale={noise_scale})")
    
    # 添加标准化步骤
    numeric_pipeline_steps.append(('scaler', StandardScaler()))
    
    # 创建数值特征的Pipeline
    numeric_transformer = Pipeline(numeric_pipeline_steps)

    # 数值表示的类别特征的转换器
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    transformers = []

    # 只有当存在数值列时才添加数值转换器
    if numerical_cols:
        transformers.append(('num', numeric_transformer, numerical_cols))

    # 只有当存在类别列时才添加类别转换器
    if categorical_num_cols:
        transformers.append(('cat', categorical_transformer, categorical_num_cols))

    # 组合两种转换器
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough',  # 将其他列直接传递（如果有的话）
        sparse_threshold=0,  # 避免稀疏输出
        verbose_feature_names_out=False  # 简化特征名
    )

    return preprocessor


# ===================== 超参数优化与模型训练函数 =====================
def optimize_model(X_train, y_train, base_model, param_space, n_trials, cv, n_jobs, results_dir, task_type="regression",
                   preprocessor=None):
    """
    使用Optuna进行超参数优化

    参数:
    ----------
    X_train : pandas.DataFrame
        训练数据
    y_train : array-like
        目标变量
    base_model : estimator
        基础模型
    param_space : dict
        参数空间
    n_trials : int
        Optuna试验次数
    cv : int
        交叉验证折数
    n_jobs : int
        并行任务数
    results_dir : str
        结果保存目录
    task_type : str, default="regression"
        任务类型，"regression"或"classification"
    preprocessor : ColumnTransformer, default=None
        特征预处理器

    返回:
    ----------
    tuple
        (优化后的管道, 最佳参数)
    """
    import optuna
    from optuna.pruners import MedianPruner

    # 确保结果目录存在
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        
    print(f"\n开始超参数优化 ({n_trials} 次试验)...")
    
    # 创建包含预处理器的Pipeline
    if preprocessor is not None:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", base_model)
        ])
    else:
        # 兼容旧版代码的默认行为
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy='mean')),
            ("scaler", StandardScaler()),
            ("model", base_model)
        ])

    def objective(trial):
        params = suggest_params(trial, param_space)
        pipeline.set_params(**params)
        if task_type == "regression":
            scoring = "neg_mean_squared_error"
        else:  # classification
            scoring = "f1_weighted" if len(np.unique(y_train)) > 2 else "f1"
        scores = cross_val_score(pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=n_jobs)
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_warmup_steps=5))
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_trial.params
    print("Best parameters from Optuna:", best_params)
    pipeline.set_params(**best_params)
    pipeline.fit(X_train, y_train)
    # 保存优化历史
    optimization_history = []
    for i, trial in enumerate(study.trials):
        optimization_history.append({
            "iteration": i + 1,
            "score": trial.value,
            "params": trial.params
        })
    pd.DataFrame(optimization_history).to_csv(os.path.join(results_dir, "hyperparameter_optimization_history.csv"),
                                              index=False)
    # 绘制优化历史曲线
    best_score = study.best_value
    plt.figure(figsize=(10, 6))
    plt.plot([h["iteration"] for h in optimization_history],
             [h["score"] for h in optimization_history], 'b-o')
    plt.axhline(y=best_score, color='r', linestyle='--', label=f'Best Score: {best_score:.6f}')
    plt.xlabel("Iteration")
    if task_type == "regression":
        plt.ylabel("Score (neg_mean_squared_error)")
    else:
        plt.ylabel("Score (F1)")
    plt.title("Hyperparameter Optimization Progress (Optuna)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_dir, "hyperparameter_optimization_history.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Optimized parameters:", best_params)
    return pipeline, best_params


# ===================== 特征重要性过滤函数 =====================
def feature_importance_filter(X_train, y_train, importance_threshold, model_pipeline, n_jobs, results_dir, plot=True):
    """
    基于模型特征重要性的特征筛选
    """
    import warnings
    import gc
    
    # 严格限制joblib并行度，避免权限问题
    import os
    old_n_jobs = os.environ.get('JOBLIB_NUM_THREADS', None)
    
    if os.name == 'nt':  # Windows系统
        print(f"Windows系统检测: 限制并行处理以避免权限问题")
        n_jobs_inner = 1  # 强制单线程
        os.environ['JOBLIB_NUM_THREADS'] = '1'  # 强制joblib使用单线程
    else:
        n_jobs_inner = min(n_jobs, 2)  # 非Windows系统也限制最多2个线程
        os.environ['JOBLIB_NUM_THREADS'] = str(n_jobs_inner)
    
    # 设置joblib并行处理的超时参数
    try:
        from joblib import parallel_backend, Parallel, delayed
        # 使用loky后端但增加超时设置
        parallel_params = {'timeout': 300}  # 5分钟超时
    except ImportError:
        parallel_params = {}
    
    # 获取任务类型
    task_type = get_config("task_type", "regression")
    original_feature_names = X_train.columns.tolist()
    
    print("\n" + "="*70)
    print("| {:^66} |".format("开始特征重要性分析"))
    print("| {:^66} |".format(f"特征数量: {len(original_feature_names)}"))
    print("="*70)
    
    # 创建基于交叉验证的特征重要性分析
    importances = []
    
    # 检查是否使用TabPFN模型以提供更详细的进度条
    is_tabpfn = False
    if hasattr(model_pipeline, 'named_steps') and 'model' in model_pipeline.named_steps:
        model = model_pipeline.named_steps['model']
        model_name = type(model).__name__
        is_tabpfn = 'TabPFN' in model_name
    
    # 为TabPFN模型配置更详细的进度条
    if is_tabpfn:
        tqdm_args = {
            "desc": "TabPFN特征重要性分析", 
            "unit": "fold", 
            "leave": True, 
            "ncols": 100,
            "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            "colour": "green",
            "ascii": " ▏▎▍▌▋▊▉█"  # 使用更丰富的ASCII字符来显示进度
        }
        print("使用TabPFN模型进行特征重要性分析，这可能需要一些时间...")
    else:
        tqdm_args = {"desc": "特征重要性分析", "unit": "fold", "leave": True, "ncols": 100}
    
    cv = get_config("cv", 5)
    kf = KFold(n_splits=cv, shuffle=True, random_state=get_config("random_state"))
    
    # 确保X_train是DataFrame以便能够通过列名索引
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
    
    # 检查y_train是DataFrame/Series还是ndarray
    is_y_pandas = isinstance(y_train, (pd.Series, pd.DataFrame))
    
    for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X_train)), total=cv, **tqdm_args):
        # 使用索引分割数据，考虑到y_train可能是Series或ndarray
        X_fold = X_train.iloc[train_idx] if isinstance(X_train, pd.DataFrame) else X_train[train_idx]
        
        # 根据y_train类型选择正确的索引方法
        if is_y_pandas:
            # 如果是pandas对象，使用iloc索引
            y_fold = y_train.iloc[train_idx]
        else:
            # 如果是numpy数组，使用标准索引
            y_fold = y_train[train_idx]
        
        # 如果是TabPFN模型，在每个折叠的开始和结束添加额外的进度信息
        if is_tabpfn:
            print(f"开始处理第 {fold+1}/{cv} 折 (TabPFN模型)...")
        
        # 需要创建一个新的管道实例，以避免交叉泄露
        model_copy = clone(model_pipeline.named_steps['model'])
        model_pipeline.named_steps['model'] = model_copy
        model_pipeline.fit(X_fold, y_fold)
        
        # 获取特征重要性（如果模型支持）
        if hasattr(model_pipeline.named_steps['model'], 'feature_importances_'):
            fold_importances = model_pipeline.named_steps['model'].feature_importances_
        elif hasattr(model_pipeline.named_steps['model'], 'coef_'):
            coef = model_pipeline.named_steps['model'].coef_
            fold_importances = np.abs(coef).reshape(-1) if len(coef.shape) > 1 else np.abs(coef)
        else:
            # 使用permutation importance作为备用
            # 为TabPFN添加一个内部进度条来显示置换重要性进度
            if is_tabpfn:
                print(f"TabPFN模型不支持直接特征重要性，正在计算置换重要性...")
                
                # 手动实现置换重要性，以便添加详细进度条
                baseline_score = model_pipeline.score(X_fold, y_fold)
                feature_importances = np.zeros(X_fold.shape[1])
                n_repeats = get_config("permutation_repeats", 5)  # 默认值为5
                
                # 为每个特征的置换重要性计算添加进度条
                with tqdm(total=len(X_fold.columns), 
                          desc="  特征置换分析", 
                          unit="特征", 
                          leave=False, 
                          colour="cyan",
                          ncols=80) as feature_pbar:
                    
                    for col_idx, col_name in enumerate(X_fold.columns):
                        col_importance = []
                        
                        # 为每次重复添加小型进度条
                        with tqdm(total=n_repeats, 
                                  desc=f"  重复 ({col_name})", 
                                  unit="次", 
                                  leave=False, 
                                  colour="yellow",
                                  ncols=70) as repeat_pbar:
                            
                            for i in range(n_repeats):
                                X_permuted = X_fold.copy()
                                X_permuted[col_name] = np.random.permutation(X_permuted[col_name].values)
                                permuted_score = model_pipeline.score(X_permuted, y_fold)
                                importance = baseline_score - permuted_score
                                col_importance.append(importance)
                                repeat_pbar.update(1)
                        
                        feature_importances[col_idx] = np.mean(col_importance)
                        feature_pbar.update(1)
                        
                        # 每5个特征或最后一个特征时显示进度
                        if (col_idx + 1) % 5 == 0 or col_idx == len(X_fold.columns) - 1:
                            percent = (col_idx + 1) / len(X_fold.columns) * 100
                            print(f"  置换进度: {col_idx + 1}/{len(X_fold.columns)} 特征 ({percent:.1f}%)")
                
                fold_importances = feature_importances
            else:
                result = permutation_importance(model_pipeline, X_fold, y_fold, n_repeats=5,
                                            random_state=get_config("random_state"))
                fold_importances = result.importances_mean
        
        importances.append(fold_importances)
        
        # 如果是TabPFN模型，在每个折叠结束时显示进度
        if is_tabpfn:
            print(f"第 {fold+1}/{cv} 折分析完成 ({(fold+1)/cv*100:.1f}%)")

    # 计算平均特征重要性和标准差
    importance_values = np.mean(importances, axis=0)
    importance_std = np.std(importances, axis=0)

    print("\n" + "="*70)
    print("| {:^66} |".format("特征重要性分析完成"))
    print("="*70)
    
    # 确保所有数组长度一致
    if len(original_feature_names) != len(importance_values) or len(original_feature_names) != len(importance_std):
        print(f"警告: 数组长度不匹配! 特征名: {len(original_feature_names)}, 重要性值: {len(importance_values)}, 标准差: {len(importance_std)}")
        # 截断或填充数组以匹配长度
        min_length = min(len(original_feature_names), len(importance_values), len(importance_std))
        original_feature_names = original_feature_names[:min_length]
        importance_values = importance_values[:min_length]
        importance_std = importance_std[:min_length]

    # 创建特征重要性数据框，添加标准差信息
    importance_df = pd.DataFrame({
        'Feature': original_feature_names,
        'Importance': importance_values,
        'Importance_std': importance_std
    })

    importance_df['Rank'] = importance_df['Importance'].rank(ascending=False)
    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)

    # 计算阈值 (百分位数)
    threshold = np.percentile(importance_values, 100 * (1 - importance_threshold))
    filtered_features = importance_df[importance_df['Importance'] >= threshold]['Feature'].tolist()

    # 将变换后的特征名映射回原始特征名（适用于独热编码的特征）
    if len(filtered_features) != len(original_feature_names):
        print(f"\n调试: 根据重要性阈值 {100 * (1 - importance_threshold):.1f}% 筛选后有 {len(filtered_features)} 个转换后特征")
        
        # 创建映射字典，记录每个转换后特征名应该对应到哪个原始特征
        transform_to_original_map = {}
        
        # 首先尝试为每个转换后特征名找到可能的原始特征
        for trans_feat in filtered_features:
            mapped = False
            
            # 情况1: 使用双下划线分割的特征名 (例如 "cat__SMOKING_1" 或 "num__AGE")
            if '__' in trans_feat:
                parts = trans_feat.split('__', 1)
                if len(parts) == 2:
                    prefix, name = parts
                    
                    # 情况1.1: 独热编码特征，格式为 prefix__feature_value
                    if '_' in name:
                        # 尝试提取可能的原始特征名 (去掉最后的 _数字 部分)
                        base_name = re.sub(r'_\d+$', '', name)
                        
                        # 检查是否是原始特征或原始特征的一部分
                        for orig_feat in original_feature_names:
                            if base_name == orig_feat or name == orig_feat:
                                transform_to_original_map[trans_feat] = orig_feat
                                mapped = True
                                print(f"  调试: 映射独热编码特征 '{trans_feat}' -> '{orig_feat}' (通过feature_value模式)")
                                break
                    
                    # 情况1.2: 数值特征，格式为 prefix__feature
                    if not mapped:
                        for orig_feat in original_feature_names:
                            if name == orig_feat:
                                transform_to_original_map[trans_feat] = orig_feat
                                mapped = True
                                print(f"  调试: 映射数值特征 '{trans_feat}' -> '{orig_feat}' (通过prefix__feature模式)")
                                break
            
            # 情况2: 无前缀的转换后特征名 (例如直接保留的特征名或scikit-learn直接使用的名称)
            if not mapped:
                # 情况2.1: 独热编码但没有前缀，可能是 "SMOKING_1"
                if '_' in trans_feat:
                    # 尝试提取基本名称 (去掉 _数字 后缀)
                    base_name = re.sub(r'_\d+$', '', trans_feat)
                    
                    for orig_feat in original_feature_names:
                        if base_name == orig_feat:
                            transform_to_original_map[trans_feat] = orig_feat
                            mapped = True
                            print(f"  调试: 映射独热编码特征 '{trans_feat}' -> '{orig_feat}' (通过无前缀模式)")
                            break
                
                # 情况2.2: 可能是直接透传的特征 (无任何处理)
                if not mapped:
                    for orig_feat in original_feature_names:
                        if trans_feat == orig_feat:
                            transform_to_original_map[trans_feat] = orig_feat
                            mapped = True
                            print(f"  调试: 映射透传特征 '{trans_feat}' -> '{orig_feat}' (直接匹配)")
                            break
            
            # 情况3: 未能通过以上规则匹配，尝试最宽松的匹配 - 如果转换后特征名包含原始特征名
            if not mapped:
                for orig_feat in original_feature_names:
                    # 确保我们匹配完整的单词/部分，避免将 'AGE' 匹配到 'AVERAGE'
                    trans_words = re.findall(r'\b\w+\b', trans_feat)
                    if orig_feat in trans_words or trans_feat.find(orig_feat) >= 0:
                        transform_to_original_map[trans_feat] = orig_feat
                        print(f"  调试: 映射特征 '{trans_feat}' -> '{orig_feat}' (通过包含关系匹配)")
                        mapped = True
                        break
            
            if not mapped:
                print(f"  警告: 未能映射转换后特征 '{trans_feat}' 到任何原始特征")
        
        # 现在我们有了映射字典，将筛选后的转换后特征名映射回原始特征
        original_filtered_features = set()
        for feat in filtered_features:
            if feat in transform_to_original_map:
                original_filtered_features.add(transform_to_original_map[feat])
                print(f"  应用: 重要特征 '{feat}' -> 原始特征 '{transform_to_original_map[feat]}'")
            else:
                print(f"  警告: 无法找到重要特征 '{feat}' 的映射关系")
        
        # 如果没有找到任何映射，回退到更简单的策略
        if not original_filtered_features and importance_threshold >= 0.99:
            print("  注意: 未找到任何映射关系，但阈值接近1.0，返回所有原始特征")
            original_filtered_features = set(original_feature_names)
        
        # 转换为列表并保持原始顺序
        filtered_features = [f for f in original_feature_names if f in original_filtered_features]
        print(f"将 {len(original_filtered_features)} 个转换后的特征名映射回原始特征名")

    # 输出重要性最高的几个特征
    print(f"特征重要性最高的10个特征:")
    for i, row in importance_df.head(min(10, len(importance_df))).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.6f} ± {row['Importance_std']:.6f}")

    # 显示重要性过滤的结果
    print(f"\n特征重要性过滤结果:")
    print(f"  原始特征数量: {len(original_feature_names)}")
    print(f"  重要性阈值(百分位数 {100 * (1 - importance_threshold):.1f}%): {threshold:.6f}")
    print(f"  保留特征数量: {len(filtered_features)}")

    # 可视化特征重要性
    if plot:
        top_n = min(30, len(importance_df))  # 显示前30个特征或全部（如果少于30个）
        plt.figure(figsize=(10, max(8, top_n * 0.3)))

        # 绘制水平条形图，包含误差条
        importance_plot_df = importance_df.head(top_n).sort_values('Importance')
        plt.barh(importance_plot_df['Feature'], importance_plot_df['Importance'],
                 xerr=importance_plot_df['Importance_std'],
                 color='skyblue', ecolor='black', capsize=5)

        plt.xlabel('Feature Importance (with Cross-Validation)')
        plt.ylabel('Features')
        plt.title('Feature Importance (Top Features)')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(results_dir, "feature_importance.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 创建特征重要性分布的散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(importance_df)), importance_df['Importance'], alpha=0.7)
        plt.axhline(y=threshold, color='r', linestyle='--',
                    label=f'Threshold ({importance_threshold * 100:.1f}%): {threshold:.6f}')
        plt.xlabel('Feature Index (sorted by importance)')
        plt.ylabel('Feature Importance')
        plt.title('Feature Importance Distribution')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "feature_importance_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()

    return filtered_features, importance_df


def _evaluate_removal(features_to_keep, X_train, y_train, optimized_pipeline, cv, n_jobs_inner, task_type):
    """
    评估移除某个特征后的模型性能

    参数:
    ----------
    features_to_keep : list
        要保留的特征列表
    X_train : pandas.DataFrame
        训练数据
    y_train : array-like
        目标变量
    optimized_pipeline : sklearn.pipeline.Pipeline
        原始优化过的管道
    cv : int 或 cross-validator
        交叉验证设置
    n_jobs_inner : int
        内部交叉验证的并行任务数
    task_type : str
        任务类型，"regression"或"classification"

    返回:
    ----------
    float
        交叉验证的平均性能分数
    """
    try:
        # 内存监控 - 在处理前检查可用内存
        import psutil
        before_mem = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB

        # 生成特征子集的哈希值用于缓存
        import hashlib
        features_str = ','.join(sorted(features_to_keep))
        cache_key = hashlib.md5(features_str.encode()).hexdigest()

        # 检查是否有缓存结果
        cache_dir = ".feature_eval_cache"
        import os
        import pickle
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{cache_key}_{task_type}.pkl")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_result = pickle.load(f)
                print(f"  使用缓存结果: {len(features_to_keep)}个特征")
                return cached_result
            except Exception as e:
                print(f"  缓存读取失败: {e}")
                # 继续执行评估

        # 创建特征子集的专用预处理器
        categorical_cols = get_config("categorical_num_cols", [])
        categorical_subset = [col for col in features_to_keep if col in categorical_cols]
        numerical_subset = [col for col in features_to_keep if col not in categorical_cols]

        # 创建子集的预处理器
        subset_preprocessor = create_preprocessor(numerical_subset, categorical_subset)

        # 获取原始模型类型和参数
        model_class = optimized_pipeline.named_steps["model"].__class__
        model_params = optimized_pipeline.named_steps["model"].get_params()

        # 创建临时管道
        temp_pipeline = Pipeline([
            ('preprocessor', subset_preprocessor),
            ('model', model_class(**model_params))
        ])

        # 设置评分指标
        if task_type == "regression":
            scoring = "neg_root_mean_squared_error"
        else:  # 分类任务
            if len(np.unique(y_train)) <= 2:
                scoring = "f1"  # 二分类
            else:
                scoring = "f1_weighted"  # 多分类

        # 根据数据规模动态调整并行策略
        X_subset = X_train[features_to_keep]
        data_size = X_subset.shape[0] * X_subset.shape[1]

        # 对于大型数据集，使用批处理避免内存问题
        if data_size > 1000000 and n_jobs_inner > 1:  # 100万个数据点，约8MB内存（假设float64）
            from sklearn.model_selection import KFold, StratifiedKFold

            # 为大型数据集创建分批CV评估函数
            print(f"  大型数据集检测: {X_subset.shape[0]}行 x {X_subset.shape[1]}列，使用批处理CV")

            # 创建适当的CV分割器
            if task_type == "regression":
                cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
            else:
                cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

            # 分批处理CV
            all_scores = []
            for train_idx, val_idx in cv_splitter.split(X_subset, y_train):
                X_train_fold, X_val_fold = X_subset.iloc[train_idx], X_subset.iloc[val_idx]
                y_train_fold, y_val_fold = y_train[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx], \
                    y_train[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]

                # 在每个折上训练和评估
                temp_pipeline.fit(X_train_fold, y_train_fold)

                # 根据任务类型计算分数
                if task_type == "regression":
                    from sklearn.metrics import mean_squared_error
                    y_pred = temp_pipeline.predict(X_val_fold)
                    score = -np.sqrt(mean_squared_error(y_val_fold, y_pred))  # 负RMSE，与sklearn一致
                else:
                    from sklearn.metrics import f1_score
                    y_pred = temp_pipeline.predict(X_val_fold)
                    if len(np.unique(y_train)) <= 2:
                        score = f1_score(y_val_fold, y_pred, zero_division=0)
                    else:
                        score = f1_score(y_val_fold, y_pred, average='weighted', zero_division=0)

                all_scores.append(score)

            scores = all_scores
        else:
            # 对于小型数据集，使用标准cross_val_score
            scores = cross_val_score(
                temp_pipeline,
                X_subset,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs_inner
            )

        # 计算平均分数
        mean_score = np.mean(scores)

        # 检查处理后的内存使用情况
        after_mem = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
        memory_used = before_mem - after_mem
        if memory_used > 0.1:  # 如果使用了超过100MB内存
            print(f"  内存使用: {memory_used:.2f}GB (特征数: {len(features_to_keep)})")

        # 缓存结果
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(mean_score, f)
        except Exception as e:
            print(f"  缓存写入失败: {e}")

        return mean_score

    except Exception as e:
        print(f"评估特征子集时出错: {e}")
        # 返回一个非常差的分数作为失败的标志
        return -np.inf if task_type == 'regression' else 0


# ===================== 特征相关性过滤函数 =====================
def correlation_filtering(X_train, y_train, features, optimized_pipeline, cv, n_jobs, corr_threshold,
                          task_type="regression", results_dir=None):
    """
    基于相关性和模型性能移除高度相关的特征（并行优化版本）

    参数:
    ----------
    X_train : pandas.DataFrame
        训练数据
    y_train : array-like
        目标变量
    features : list
        要分析的特征列表
    optimized_pipeline : sklearn.pipeline.Pipeline
        优化过的基础管道
    cv : int
        交叉验证折数
    n_jobs : int
        并行任务数
    corr_threshold : float
        相关性筛选阈值
    task_type : str, default="regression"
        任务类型，"regression"或"classification"
    results_dir : str, default=None
        结果保存目录

    返回:
    ----------
    list
        筛选后保留的特征列表
    """
    from joblib import Parallel, delayed
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    print(f"\n=== 相关性过滤 ===")
    print(f"过滤相关性系数 > {corr_threshold} 的特征")

    # 获取系统可用CPU核心数
    import multiprocessing
    total_cores = multiprocessing.cpu_count()
    print(f"系统可用CPU核心数: {total_cores}")

    # 限制总并行任务数，避免过度使用资源
    max_jobs = 4
    print(f"总并行任务数上限: {max_jobs}")

    selected_features = features.copy()
    
    # 计算特征相关性矩阵
    corr_matrix = X_train[features].corr()
    
    # 生成并保存过滤前的相关性热图
    plt.figure(figsize=(12, 10))
    
    # 创建掩码只显示下三角部分
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # 绘制热图
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False if len(features) > 30 else True, 
                fmt='.2f', square=True, linewidths=.5, vmin=-1, vmax=1)
    
    plt.title('Feature Correlation Heatmap (Before Filtering)', fontsize=16)
    plt.tight_layout()
    
    # 如果提供了结果目录，保存图像
    if results_dir:
        # 确保结果目录存在
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存图像
        plt.savefig(os.path.join(results_dir, "correlation_heatmap_before.png"), dpi=300, bbox_inches='tight')
        print("已保存过滤前的相关性热图")
    plt.close()

    # 开始相关性过滤
    while True:
        removed = False
        # 计算相关性矩阵
        corr_matrix = X_train[selected_features].corr().abs()
        # 获取上三角矩阵
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # 找出高相关特征对
        pairs = [(col, row) for col in upper.columns for row in upper.index
                 if pd.notnull(upper.loc[row, col]) and upper.loc[row, col] > corr_threshold]
        if not pairs:
            break

        for feat1, feat2 in pairs:
            print(f"评估特征对: '{feat1}' 和 '{feat2}' (相关性: {corr_matrix.loc[feat1, feat2]:.4f})")
            current_features = selected_features.copy()

            # 智能分配并行资源
            # 策略：优先分配给外层并行，然后根据计算复杂度给内层分配资源
            dataset_size = X_train.shape[0]
            feature_count = len(current_features)

            # 根据数据规模动态调整资源分配
            if dataset_size * feature_count > 100000:  # 大型数据集
                # 对于大型数据集，减少外层并行度，增加内层并行度
                n_jobs_outer = 1
                n_jobs_inner = max_jobs
            else:
                # 对于小型数据集，外层最多使用2个作业，内层合理分配
                n_jobs_outer = min(2, max_jobs)
                # 如果CV折数较多，给内层更多资源
                if cv > 5:
                    n_jobs_inner = max(2, max_jobs // n_jobs_outer)
                else:
                    # 对于简单交叉验证，减少内层资源分配
                    n_jobs_inner = max(1, max_jobs // (n_jobs_outer * 2))

            print(f"并行设置: 外层并行数={n_jobs_outer}, 内层交叉验证并行数={n_jobs_inner}")
            print(f"数据集特征: 样本数={dataset_size}, 特征数={feature_count}")

            # 准备两个特征子集：一个不包含feat1，一个不包含feat2
            features_without_feat1 = [f for f in current_features if f != feat1]
            features_without_feat2 = [f for f in current_features if f != feat2]

            # 并行评估两个特征子集
            results = Parallel(n_jobs=n_jobs_outer)(
                delayed(_evaluate_removal)(
                    features_subset, X_train, y_train, optimized_pipeline, cv, n_jobs_inner, task_type
                )
                for features_subset in [features_without_feat1, features_without_feat2]
            )

            # 解析结果
            perf1, perf2 = results

            if task_type == "regression":
                # 对于回归，neg_root_mean_squared_error 越高越好
                if perf1 >= perf2:  # 移除 feat1 后性能更好或相同
                    print(f"  移除特征 '{feat1}' (移除后性能: {perf1:.4f} vs {perf2:.4f})")
                    selected_features.remove(feat1)
                else:  # 移除 feat2 后性能更好
                    print(f"  移除特征 '{feat2}' (移除后性能: {perf2:.4f} vs {perf1:.4f})")
                    selected_features.remove(feat2)
            else:  # 分类任务
                # 对于分类，f1或f1_weighted 越高越好
                if perf1 >= perf2:  # 移除 feat1 后性能更好或相同
                    print(f"  移除特征 '{feat1}' (移除后性能: {perf1:.4f} vs {perf2:.4f})")
                    selected_features.remove(feat1)
                else:  # 移除 feat2 后性能更好
                    print(f"  移除特征 '{feat2}' (移除后性能: {perf2:.4f} vs {perf1:.4f})")
                    selected_features.remove(feat2)

            removed = True
            break

        if not removed:
            break

    print(f"相关性过滤完成，剩余特征: {len(selected_features)}")
    
    # 生成并保存过滤后的相关性热图
    if len(selected_features) > 1:  # 确保至少有两个特征用于计算相关性
        corr_matrix_after = X_train[selected_features].corr()
        
        plt.figure(figsize=(12, 10))
        
        # 创建掩码只显示下三角部分
        mask = np.triu(np.ones_like(corr_matrix_after, dtype=bool))
        
        # 绘制热图
        sns.heatmap(corr_matrix_after, mask=mask, cmap='coolwarm', 
                    annot=False if len(selected_features) > 30 else True,
                    fmt='.2f', square=True, linewidths=.5, vmin=-1, vmax=1)
        
        plt.title('Feature Correlation Heatmap (After Filtering)', fontsize=16)
        plt.tight_layout()
        
        # 如果提供了结果目录，保存图像
        if results_dir:
            # 保存图像
            plt.savefig(os.path.join(results_dir, "correlation_heatmap_after.png"), dpi=300, bbox_inches='tight')
            print("已保存过滤后的相关性热图")
        plt.close()
    else:
        print("剩余特征少于2个，无法生成相关性热图")
    
    return selected_features


# ===================== 固定超参数组合评估函数 =====================
def evaluate_combination_fixed(X_train, y_train, X_test, y_test, features, fixed_pipeline, task_type="regression"):
    """
    利用固定的最优超参数结果，对给定特征组合进行模型训练和评估，
    不再对每个组合进行单独的超参数优化。
    """
    # 获取交叉验证折数
    cv = get_config("cv", 5)

    # 创建交叉验证对象
    if task_type == "regression":
        cv_obj = KFold(n_splits=cv, shuffle=True, random_state=get_config("random_state"))
        scoring = 'neg_root_mean_squared_error'
    else:  # 分类问题
        cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=get_config("random_state"))
        if len(np.unique(y_train)) <= 2:
            scoring = 'f1'
        else:
            scoring = 'f1_weighted'

    # 为当前特征子集创建特定的预处理器
    categorical_cols = get_config("categorical_num_cols", [])
    categorical_subset = [col for col in features if col in categorical_cols]
    numerical_subset = [col for col in features if col not in categorical_cols]
    subset_preprocessor = create_preprocessor(numerical_subset, categorical_subset)

    # 获取原始模型实例
    if hasattr(fixed_pipeline, 'named_steps') and 'model' in fixed_pipeline.named_steps:
        model_instance = clone(fixed_pipeline.named_steps["model"])
    else:
        model_instance = clone(fixed_pipeline)

    # 创建使用特征子集专用预处理器的pipeline
    pipeline = Pipeline([
        ("preprocessor", subset_preprocessor),
        ("model", model_instance)
    ])

    # 使用交叉验证评估模型
    print(f"使用{cv}折交叉验证评估特征组合: {len(features)}个特征")

    if task_type == "regression":
        # 回归任务评估
        # 交叉验证计算RMSE
        cv_scores = cross_val_score(
            pipeline,
            X_train[features],
            y_train,
            cv=cv_obj,
            scoring=scoring,
            n_jobs=get_config("n_jobs", -1)
        )
        cv_rmse = -np.mean(cv_scores)  # 转换为正的RMSE

        # 交叉验证计算R²
        r2_scores = cross_val_score(
            pipeline,
            X_train[features],
            y_train,
            cv=cv_obj,
            scoring='r2',
            n_jobs=get_config("n_jobs", -1)
        )
        cv_r2 = np.mean(r2_scores)

        # 计算交叉验证的标准差(用于评估稳定性)
        cv_rmse_std = np.std(-cv_scores)
        cv_r2_std = np.std(r2_scores)

        # 在全部训练集上训练模型，并评估测试集性能
        pipeline.fit(X_train[features], y_train)
        y_pred = pipeline.predict(X_test[features])
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)

        return {
            "features": features,
            "rmse": cv_rmse,  # 主要使用交叉验证的RMSE
            "rmse_std": cv_rmse_std,  # RMSE的标准差
            "r2": cv_r2,  # 交叉验证的R²
            "r2_std": cv_r2_std,  # R²的标准差
            "test_rmse": test_rmse,  # 测试集上的RMSE
            "test_r2": test_r2,  # 测试集上的R²
            "num_features": len(features)
        }
    else:  # classification
        # 分类任务评估
        # 准确率
        accuracy_scores = cross_val_score(
            pipeline,
            X_train[features],
            y_train,
            cv=cv_obj,
            scoring='accuracy',
            n_jobs=get_config("n_jobs", -1)
        )
        cv_accuracy = np.mean(accuracy_scores)
        cv_accuracy_std = np.std(accuracy_scores)

        # F1分数
        if len(np.unique(y_train)) <= 2:
            # 二分类
            f1_scores = cross_val_score(
                pipeline,
                X_train[features],
                y_train,
                cv=cv_obj,
                scoring='f1',
                n_jobs=get_config("n_jobs", -1)
            )
            precision_scores = cross_val_score(
                pipeline,
                X_train[features],
                y_train,
                cv=cv_obj,
                scoring='precision',
                n_jobs=get_config("n_jobs", -1)
            )
            recall_scores = cross_val_score(
                pipeline,
                X_train[features],
                y_train,
                cv=cv_obj,
                scoring='recall',
                n_jobs=get_config("n_jobs", -1)
            )
        else:
            # 多分类
            f1_scores = cross_val_score(
                pipeline,
                X_train[features],
                y_train,
                cv=cv_obj,
                scoring='f1_weighted',
                n_jobs=get_config("n_jobs", -1)
            )
            precision_scores = cross_val_score(
                pipeline,
                X_train[features],
                y_train,
                cv=cv_obj,
                scoring='precision_weighted',
                n_jobs=get_config("n_jobs", -1)
            )
            recall_scores = cross_val_score(
                pipeline,
                X_train[features],
                y_train,
                cv=cv_obj,
                scoring='recall_weighted',
                n_jobs=get_config("n_jobs", -1)
            )

        cv_f1 = np.mean(f1_scores)
        cv_f1_std = np.std(f1_scores)
        cv_precision = np.mean(precision_scores)
        cv_recall = np.mean(recall_scores)

        # 在全部训练集上训练模型，并评估测试集性能（用于AUC和最终评估）
        pipeline.fit(X_train[features], y_train)
        y_pred = pipeline.predict(X_test[features])

        test_accuracy = accuracy_score(y_test, y_pred)
        if len(np.unique(y_train)) <= 2:
            test_f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            test_precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            test_recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        else:
            test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            test_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

        # AUC score if the model has predict_proba
        auc = 0
        if hasattr(pipeline, 'predict_proba'):
            try:
                y_proba = pipeline.predict_proba(X_test[features])
                if len(np.unique(y_train)) <= 2:
                    if y_proba.shape[1] >= 2:  # Binary classification
                        auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            except:
                pass

        return {
            "features": features,
            "accuracy": cv_accuracy,  # 交叉验证的准确率
            "accuracy_std": cv_accuracy_std,  # 准确率的标准差
            "f1": cv_f1,  # 交叉验证的F1
            "f1_std": cv_f1_std,  # F1的标准差
            "precision": cv_precision,  # 交叉验证的精确率
            "recall": cv_recall,  # 交叉验证的召回率
            "test_accuracy": test_accuracy,  # 测试集上的准确率
            "test_f1": test_f1,  # 测试集上的F1
            "test_precision": test_precision,  # 测试集上的精确率
            "test_recall": test_recall,  # 测试集上的召回率
            "auc": auc,  # AUC (仅在测试集上)
            "num_features": len(features)
        }


# ===================== 固定超参数组合搜索函数 =====================
def combination_search_parallel_fixed(X_train, y_train, X_test, y_test, selected_features, fixed_pipeline, cv, n_jobs,
                                      results_dir, task_type="regression"):
    """
    并行方式使用固定超参数进行特征组合搜索。
    利用已经优化好的pipeline对不同特征组合进行测试，找出最佳组合。
    """
    import itertools
    from joblib import Parallel, delayed

    print(f"正在对 {len(selected_features)} 个特征的组合进行搜索...")
    print(f"将使用交叉验证 ({cv} 折) 来评估每个特征组合的性能")

    if task_type == "regression":
        cv_obj = KFold(n_splits=cv, shuffle=True, random_state=get_config("random_state"))
        scoring = 'neg_root_mean_squared_error'
    else:  # 分类任务
        cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=get_config("random_state"))
        scoring = 'f1_weighted' if len(np.unique(y_train)) > 2 else 'f1'

    # 添加函数来使用交叉验证评估特征组合
    def evaluate_with_cv(features):
        # 创建特定于当前特征子集的预处理器
        categorical_cols = get_config("categorical_num_cols", [])
        categorical_subset = [col for col in features if col in categorical_cols]
        numerical_subset = [col for col in features if col not in categorical_cols]

        # 创建新的预处理器
        subset_preprocessor = create_preprocessor(numerical_subset, categorical_subset)

        # 获取原始模型实例
        if hasattr(fixed_pipeline, 'named_steps') and 'model' in fixed_pipeline.named_steps:
            model_instance = clone(fixed_pipeline.named_steps["model"])
        else:
            model_instance = clone(fixed_pipeline)

        # 创建新的pipeline，使用特征子集专用的预处理器
        pipeline = Pipeline([
            ("preprocessor", subset_preprocessor),
            ("model", model_instance)
        ])

        try:
            cv_scores = cross_val_score(
                pipeline,
                X_train[features],
                y_train,
                cv=cv_obj,
                scoring=scoring,
                n_jobs=1  # 设为1因为我们已经在外部并行化了
            )

            # 对于回归任务，我们使用交叉验证评估的RMSE和R2
            if task_type == "regression":
                cv_rmse = -np.mean(cv_scores)  # 负的RMSE变成正的

                # 计算R2，需要另一次交叉验证
                r2_scores = cross_val_score(
                    pipeline,
                    X_train[features],
                    y_train,
                    cv=cv_obj,
                    scoring='r2',
                    n_jobs=1
                )
                cv_r2 = np.mean(r2_scores)

                # 最后再训练一次模型，并在测试集上评估，用于最终报告
                pipeline.fit(X_train[features], y_train)
                y_pred = pipeline.predict(X_test[features])
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                test_r2 = r2_score(y_test, y_pred)

                return {
                    "features": features,
                    "rmse": cv_rmse,  # 主要性能指标：交叉验证的RMSE
                    "r2": cv_r2,  # 交叉验证的R2
                    "test_rmse": test_rmse,  # 测试集上的RMSE（仅供参考）
                    "test_r2": test_r2,  # 测试集上的R2（仅供参考）
                    "num_features": len(features)
                }
            else:  # 分类任务
                if len(np.unique(y_train)) <= 2:
                    # 二分类任务
                    f1_cv = np.mean(cv_scores)  # F1已经是正的，不需要取反

                    # 评估其他指标
                    accuracy_scores = cross_val_score(pipeline, X_train[features], y_train,
                                                      cv=cv_obj, scoring='accuracy', n_jobs=1)
                    precision_scores = cross_val_score(pipeline, X_train[features], y_train,
                                                       cv=cv_obj, scoring='precision', n_jobs=1)
                    recall_scores = cross_val_score(pipeline, X_train[features], y_train,
                                                    cv=cv_obj, scoring='recall', n_jobs=1)

                    # 计算平均值
                    accuracy_cv = np.mean(accuracy_scores)
                    precision_cv = np.mean(precision_scores)
                    recall_cv = np.mean(recall_scores)

                    # 最后再训练一次模型，并在测试集上评估AUC，用于最终报告
                    pipeline.fit(X_train[features], y_train)
                    y_pred = pipeline.predict(X_test[features])
                    test_accuracy = accuracy_score(y_test, y_pred)
                    test_f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

                    # 尝试计算AUC
                    auc = 0
                    if hasattr(pipeline, 'predict_proba'):
                        try:
                            y_proba = pipeline.predict_proba(X_test[features])
                            if y_proba.shape[1] >= 2:
                                auc = roc_auc_score(y_test, y_proba[:, 1])
                        except:
                            pass
                else:
                    # 多分类任务
                    f1_cv = np.mean(cv_scores)

                    # 评估其他指标
                    accuracy_scores = cross_val_score(pipeline, X_train[features], y_train,
                                                      cv=cv_obj, scoring='accuracy', n_jobs=1)
                    precision_scores = cross_val_score(pipeline, X_train[features], y_train,
                                                       cv=cv_obj, scoring='precision_weighted', n_jobs=1)
                    recall_scores = cross_val_score(pipeline, X_train[features], y_train,
                                                    cv=cv_obj, scoring='recall_weighted', n_jobs=1)

                    # 计算平均值
                    accuracy_cv = np.mean(accuracy_scores)
                    precision_cv = np.mean(precision_scores)
                    recall_cv = np.mean(recall_scores)

                    # 最后再训练一次模型，并在测试集上评估
                    pipeline.fit(X_train[features], y_train)
                    y_pred = pipeline.predict(X_test[features])
                    test_accuracy = accuracy_score(y_test, y_pred)
                    test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                    # 尝试计算AUC
                    auc = 0
                    if hasattr(pipeline, 'predict_proba'):
                        try:
                            y_proba = pipeline.predict_proba(X_test[features])
                            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                        except:
                            pass

                return {
                    "features": features,
                    "accuracy": accuracy_cv,  # 交叉验证的准确率
                    "f1": f1_cv,  # 主要性能指标：交叉验证的F1
                    "precision": precision_cv,  # 交叉验证的精确率
                    "recall": recall_cv,  # 交叉验证的召回率
                    "test_accuracy": test_accuracy,  # 测试集上的准确率（仅供参考）
                    "test_f1": test_f1,  # 测试集上的F1（仅供参考）
                    "auc": auc,  # 测试集上的AUC（仅供参考）
                    "num_features": len(features)
                }
        except Exception as e:
            print(f"评估特征子集时出错: {str(e)} - 特征: {features}")
            if task_type == "regression":
                return {"features": features, "rmse": float('inf'), "r2": -float('inf'), "num_features": len(features)}
            else:
                return {"features": features, "accuracy": 0, "f1": 0, "precision": 0, "recall": 0, "auc": 0,
                        "num_features": len(features)}

    max_combinations = 10000  # 设置组合上限以避免内存溢出

    # 准备所有可能的特征组合
    print("生成特征组合...")
    all_combinations = []
    for r in range(1, len(selected_features) + 1):
        combos = list(itertools.combinations(selected_features, r))
        all_combinations.extend(combos)

        if len(all_combinations) > max_combinations:
            print(f"警告: 组合数量超过上限 ({max_combinations})，将只评估一部分组合。")
            all_combinations = all_combinations[:max_combinations]
            break

    total_combinations = len(all_combinations)
    print(f"开始评估 {total_combinations} 个特征组合...")

    # 并行评估所有组合
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_with_cv)(list(combo)) for combo in all_combinations
    )

    # 排序结果
    if task_type == "regression":
        results = sorted(results, key=lambda x: x["rmse"])
    else:  # classification
        results = sorted(results, key=lambda x: x["f1"], reverse=True)

    # 保存临时结果
    if task_type == "regression":
        interim_results = pd.DataFrame([{
            "features": ",".join(r["features"]),
            "num_features": r["num_features"],
            "cv_rmse": r["rmse"],
            "cv_r2": r["r2"],
            "test_rmse": r.get("test_rmse", 0),
            "test_r2": r.get("test_r2", 0)
        } for r in results])
    else:  # classification
        interim_results = pd.DataFrame([{
            "features": ",".join(r["features"]),
            "num_features": r["num_features"],
            "cv_accuracy": r["accuracy"],
            "cv_f1": r["f1"],
            "cv_precision": r["precision"],
            "cv_recall": r["recall"],
            "test_accuracy": r.get("test_accuracy", 0),
            "test_f1": r.get("test_f1", 0),
            "auc": r.get("auc", 0)
        } for r in results])

    interim_results.to_csv(os.path.join(results_dir, "feature_combinations_results.csv"), index=False)

    # 打印顶部结果
    if task_type == "regression":
        print("\n顶部5个特征组合:")
        for r in results[:5]:
            print(
                f"特征: {r['features']}, 特征数量: {r['num_features']}, CV-RMSE: {r['rmse']:.4f}, CV-R2: {r['r2']:.4f}")
    else:
        print("\n顶部5个特征组合:")
        for r in results[:5]:
            print(
                f"特征: {r['features']}, 特征数量: {r['num_features']}, CV-F1: {r['f1']:.4f}, CV-Accuracy: {r['accuracy']:.4f}")

    return results


# ===================== 组合结果可视化函数 =====================
def plot_combination_results(results, results_dir, task_type="regression"):
    """
    Plot feature combination search results

    Parameters:
    results: List of dictionaries with results from feature combination evaluation
    results_dir: Directory to save plots
    task_type: Type of task (regression or classification)
    """
    # Create a color palette for plotting
    colors = plt.cm.viridis(np.linspace(0, 1, min(10, len(results))))

    # Sort results by performance metric
    if task_type == "regression":
        results_sorted = sorted(results, key=lambda x: x.get('rmse', float('inf')))
        metric_name = "RMSE"
        metric_key = "rmse"
        best_label = "Lower is better"
    else:
        results_sorted = sorted(results, key=lambda x: x.get('f1', 0), reverse=True)
        metric_name = "F1 Score"
        metric_key = "f1"
        best_label = "Higher is better"

    # Extract top N results for plotting
    top_n = min(10, len(results_sorted))
    top_results = results_sorted[:top_n]

    # Create a figure for plotting
    plt.figure(figsize=(12, 8))

    # Plotting performance metrics
    metrics = [result[metric_key] for result in top_results]
    x_pos = np.arange(len(top_results))

    # Adjust for metric direction (higher is better or lower is better)
    if task_type == "regression":
        # For regression, lower RMSE is better
        plt.bar(x_pos, metrics, color=colors, alpha=0.8)
        plt.axhline(y=min(metrics), color='r', linestyle='--',
                    label=f'Best {metric_name}: {min(metrics):.4f}')
    else:
        # For classification, higher metrics are better
        plt.bar(x_pos, metrics, color=colors, alpha=0.8)
        plt.axhline(y=max(metrics), color='r', linestyle='--',
                    label=f'Best {metric_name}: {max(metrics):.4f}')

    # Add feature information as labels
    labels = [', '.join(result['features'][:3]) +
              (f' + {len(result["features"]) - 3} more' if len(result['features']) > 3 else '')
              for result in top_results]

    plt.xticks(x_pos, labels, rotation=45, ha='right')
    plt.ylabel(metric_name)
    plt.title(f'Top {top_n} Feature Combinations ({metric_name})')
    plt.legend([best_label])
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"top_combinations_{metric_key}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Create detailed feature frequency analysis
    feature_counts = {}
    for result in results_sorted[:min(20, len(results_sorted))]:
        for feature in result['features']:
            if feature in feature_counts:
                feature_counts[feature] += 1
            else:
                feature_counts[feature] = 1

    # Sort feature counts for plotting
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    features = [item[0] for item in sorted_features]
    counts = [item[1] for item in sorted_features]

    # Plot feature frequency
    plt.figure(figsize=(12, 8))
    plt.bar(np.arange(len(features)), counts, color=plt.cm.viridis(np.linspace(0, 1, len(features))))
    plt.xticks(np.arange(len(features)), features, rotation=45, ha='right')
    plt.ylabel('Frequency in Top Combinations')
    plt.title('Feature Frequency in Top Combinations')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "feature_frequency.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot number of features vs performance
    feature_counts = [len(result['features']) for result in results]
    metrics_values = [result[metric_key] for result in results]

    plt.figure(figsize=(10, 6))
    if task_type == "regression":
        # For regression, negative correlation is better (lower RMSE)
        plt.scatter(feature_counts, metrics_values, alpha=0.6, c=metrics_values, cmap='viridis_r')
        plt.ylabel(f"{metric_name} (Lower is better)")
    else:
        # For classification, positive correlation is better (higher F1, accuracy, etc.)
        plt.scatter(feature_counts, metrics_values, alpha=0.6, c=metrics_values, cmap='viridis')
        plt.ylabel(f"{metric_name} (Higher is better)")

    plt.xlabel('Number of Features')
    plt.title('Number of Features vs. Performance')
    plt.colorbar(label=metric_name)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"features_vs_{metric_key}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Save top results to CSV
    top_results_df = pd.DataFrame(top_results)
    # Make sure the important columns come first
    cols = top_results_df.columns.tolist()
    for col in ['features', metric_key]:
        if col in cols:
            cols.remove(col)
    cols = ['features', metric_key] + cols

    # Save only the columns that exist
    existing_cols = [col for col in cols if col in top_results_df.columns]
    top_results_df = top_results_df[existing_cols]

    top_results_df.to_csv(os.path.join(results_dir, "top_combinations_detailed.csv"), index=False)

    # Create a pairwise correlation matrix of top features if possible
    if len(top_results) > 0 and 'X_train_sample' in top_results[0]:
        X_sample = top_results[0]['X_train_sample']
        top_features = list(set(feature for result in top_results for feature in result['features']))

        if len(top_features) > 1 and all(feature in X_sample.columns for feature in top_features):
            corr_matrix = X_sample[top_features].corr()

            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", annot=True,
                        fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "feature_correlation.png"), dpi=300, bbox_inches='tight')
            plt.close()


# ===================== 最终模型评估函数 =====================
def final_model_evaluation_full(X_train, y_train, X_test, y_test, selected_features, base_model, best_params, cv,
                                n_jobs, results_dir):
    """
    Perform final model evaluation using the best parameters for the selected feature set

    Parameters:
    X_train: Training features
    y_train: Training labels
    X_test: Test features
    y_test: Test labels
    selected_features: Selected feature subset
    base_model: Base model object
    best_params: Dictionary of best hyperparameters (from optimize_model function)
    cv: Number of cross-validation folds
    n_jobs: Number of parallel jobs
    results_dir: Directory to save results

    Returns:
    pipeline: Trained model pipeline
    """
    print("\n=== Final Model Evaluation ===")
    print(f"Evaluating {len(selected_features)} selected features...")

    # Get categorical and numerical features from selected features
    categorical_num_cols = get_config("categorical_num_cols", [])
    categorical_subset = [col for col in selected_features if col in categorical_num_cols]
    numerical_subset = [col for col in selected_features if col not in categorical_num_cols]

    # Create preprocessor using only the selected feature subset
    preprocessor = create_preprocessor(numerical_subset, categorical_subset)

    # Create Pipeline with preprocessor
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", base_model)
    ])

    # Set model with previously optimized best parameters
    pipeline.set_params(**best_params)
    print("Applying best hyperparameters:", best_params)

    # Evaluate final model using cross-validation
    print("Performing cross-validation evaluation on training set...")
    cv_scores = cross_val_score(
        pipeline,
        X_train[selected_features],
        y_train,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=n_jobs
    )
    cv_rmse = np.sqrt(-np.mean(cv_scores))
    cv_rmse_std = np.std([np.sqrt(-score) for score in cv_scores])
    print(f"Cross-validation RMSE: {cv_rmse:.4f} ± {cv_rmse_std:.4f}")

    # Train model on full training set
    print("Training final model on complete training set...")
    pipeline.fit(X_train[selected_features], y_train)

    # Save model and feature information as dictionary
    model_dict = {
        "model": pipeline,
        "feature_names": selected_features
    }
    dump(model_dict, os.path.join(results_dir, "final_model_with_metadata.pkl"))
    print("Final model saved as 'final_model_with_metadata.pkl'")

    # Evaluate model performance
    y_train_pred = pipeline.predict(X_train[selected_features])
    y_test_pred = pipeline.predict(X_test[selected_features])
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    # Print performance metrics
    print("\nFinal Model Performance:")
    print(f"Training set: RMSE = {train_rmse:.4f}, R² = {train_r2:.4f}")
    print(f"Test set: RMSE = {test_rmse:.4f}, R² = {test_r2:.4f}")

    # Visualize predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train, y_train_pred, color="blue", label="Training Set", alpha=0.6)
    plt.scatter(y_test, y_test_pred, color="red", label="Test Set", alpha=0.6)
    plt.plot([min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
             [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
             'k--', lw=2, label="Ideal Line")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Final Model: Predicted vs Actual Values")
    metrics_text = (f"Training: RMSE = {train_rmse:.2f}, R² = {train_r2:.2f}\n"
                    f"Test: RMSE = {test_rmse:.2f}, R² = {test_r2:.2f}")
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "final_model_evaluation.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Create performance report CSV file
    report_df = pd.DataFrame({
        "Metric": ["Train RMSE", "Train R²", "Test RMSE", "Test R²", "CV RMSE"],
        "Value": [train_rmse, train_r2, test_rmse, test_r2, cv_rmse],
        "Std": [None, None, None, None, cv_rmse_std]
    })
    report_df.to_csv(os.path.join(results_dir, "final_model_performance.csv"), index=False)

    # Return trained pipeline model
    return pipeline


# ===================== 最终模型评估函数 =====================
def final_model_evaluation_tabpfn(X_train, y_train, X_test, y_test, selected_features, fixed_pipeline, results_dir):
    """For TabPFN models, train and evaluate directly using default parameters without hyperparameter optimization"""

    print("\n开始TabPFN回归模型的最终评估...")
    
    # Check if pipeline already has a preprocessor, if not, create one
    if 'preprocessor' not in [step[0] for step in fixed_pipeline.steps]:
        # Get categorical and numerical features from selected features
        categorical_num_cols = get_config("categorical_num_cols", [])
        categorical_subset = [col for col in selected_features if col in categorical_num_cols]
        numerical_subset = [col for col in selected_features if col not in categorical_num_cols]

        # Create preprocessor using only the selected feature subset
        print("创建特征子集专用预处理器...")
        preprocessor = create_preprocessor(numerical_subset, categorical_subset)

        # Create new pipeline with preprocessor
        new_steps = [("preprocessor", preprocessor)]
        for name, transformer in fixed_pipeline.steps:
            if name not in ["imputer", "scaler"]:  # Skip original preprocessing steps
                new_steps.append((name, transformer))
        fixed_pipeline = Pipeline(new_steps)

    # 训练TabPFN模型并添加进度条显示
    print("正在训练TabPFN回归模型...")
    print("[进度]: 准备数据")
    X_train_selected = X_train[selected_features]
    
    # 使用自定义进度条显示训练过程
    print("[进度]: 模型适配中 ████████████ 0%")
    # TabPFN模型适配过程可能需要一些时间，添加进度指示
    try:
        # 使用time.time()记录开始时间
        import time
        start_time = time.time()
        
        # 适配模型
        fixed_pipeline.fit(X_train_selected, y_train)
        
        # 计算并显示耗时
        elapsed_time = time.time() - start_time
        print(f"[进度]: 模型适配完成 ████████████████████████████ 100% (耗时: {elapsed_time:.2f}秒)")
    except Exception as e:
        print(f"[错误]: 模型训练失败 - {str(e)}")
        return None

    model_dict = {
        "model": fixed_pipeline,
        "feature_names": selected_features
    }
    dump(model_dict, os.path.join(results_dir, "final_model_with_metadata.pkl"))
    print("Final model saved as 'final_model_with_metadata.pkl'")

    print("[进度]: 模型评估中...")
    y_train_pred = fixed_pipeline.predict(X_train[selected_features])
    y_test_pred = fixed_pipeline.predict(X_test[selected_features])
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    print("[进度]: 模型评估完成 ✓")
    
    print("\n评估结果:")
    print("Training RMSE:", train_rmse, "Training R2:", train_r2)
    print("Test RMSE:", test_rmse, "Test R2:", test_r2)
    
    print("[进度]: 生成可视化图表...")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train, y_train_pred, color="blue", label="Training Set", alpha=0.6)
    plt.scatter(y_test, y_test_pred, color="red", label="Test Set", alpha=0.6)
    plt.plot([min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
             [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
             'k--', lw=2, label="Ideal")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Final Model: Predicted vs Actual Values (TabPFN)")
    metrics_text = (f"Training RMSE: {train_rmse:.2f}\nTraining R2: {train_r2:.2f}\n"
                    f"Test RMSE: {test_rmse:.2f}\nTest R2: {test_r2:.2f}")
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "final_model_evaluation.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("[进度]: 图表生成完成 ✓")
    
    print("TabPFN回归模型评估完成!")
    
    # Return trained pipeline model
    return fixed_pipeline


# ===================== Final Model Evaluation Functions =====================
def final_model_evaluation_tabpfn_classification(X_train, y_train, X_test, y_test, selected_features, fixed_pipeline,
                                                 results_dir):
    """For TabPFN classifier models, train and evaluate directly using default parameters without hyperparameter optimization"""

    print("\n开始TabPFN分类模型的最终评估...")
    
    # Check if pipeline already has a preprocessor, if not, create one
    if 'preprocessor' not in [step[0] for step in fixed_pipeline.steps]:
        # Get categorical and numerical features from selected features
        categorical_num_cols = get_config("categorical_num_cols", [])
        categorical_subset = [col for col in selected_features if col in categorical_num_cols]
        numerical_subset = [col for col in selected_features if col not in categorical_num_cols]

        # Create preprocessor using only the selected feature subset
        print("创建特征子集专用预处理器...")
        preprocessor = create_preprocessor(numerical_subset, categorical_subset)

        # Create new pipeline with preprocessor
        new_steps = [("preprocessor", preprocessor)]
        for name, transformer in fixed_pipeline.steps:
            if name not in ["imputer", "scaler"]:  # Skip original preprocessing steps
                new_steps.append((name, transformer))
        fixed_pipeline = Pipeline(new_steps)

    # 训练TabPFN分类模型并添加进度条显示
    print("正在训练TabPFN分类模型...")
    print("[进度]: 准备数据")
    X_train_selected = X_train[selected_features]
    
    # 显示不同类别的分布情况
    class_counts = np.unique(y_train, return_counts=True)
    print(f"类别分布: {dict(zip(class_counts[0], class_counts[1]))}")
    
    # 使用自定义进度条显示训练过程
    print("[进度]: 模型适配中 ████████████ 0%")
    # TabPFN模型适配过程可能需要一些时间，添加进度指示
    try:
        # 使用time.time()记录开始时间
        import time
        start_time = time.time()
        
        # 适配模型
        fixed_pipeline.fit(X_train_selected, y_train)
        
        # 计算并显示耗时
        elapsed_time = time.time() - start_time
        print(f"[进度]: 模型适配完成 ████████████████████████████ 100% (耗时: {elapsed_time:.2f}秒)")
    except Exception as e:
        print(f"[错误]: 模型训练失败 - {str(e)}")
        return None

    model_dict = {
        "model": fixed_pipeline,
        "feature_names": selected_features
    }
    dump(model_dict, os.path.join(results_dir, "final_model_with_metadata.pkl"))
    print("Final model saved as 'final_model_with_metadata.pkl'")

    # Evaluate performance on training and test sets
    print("[进度]: 模型评估中...")
    y_train_pred = fixed_pipeline.predict(X_train[selected_features])
    y_test_pred = fixed_pipeline.predict(X_test[selected_features])

    # Calculate training set metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    if len(np.unique(y_train)) <= 2:
        train_f1 = f1_score(y_train, y_train_pred, average='binary', zero_division=0)
        train_precision = precision_score(y_train, y_train_pred, average='binary', zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, average='binary', zero_division=0)
    else:
        train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
        train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)

    # Calculate test set metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    if len(np.unique(y_test)) <= 2:
        test_f1 = f1_score(y_test, y_test_pred, average='binary', zero_division=0)
        test_precision = precision_score(y_test, y_test_pred, average='binary', zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average='binary', zero_division=0)
    else:
        test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    print("[进度]: 模型评估完成 ✓")

    print("\n评估结果:")
    print("Training set metrics:")
    print(f"  Accuracy: {train_accuracy:.4f}")
    print(f"  F1 Score: {train_f1:.4f}")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  Recall: {train_recall:.4f}")

    print("Test set metrics:")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")

    # Confusion matrix
    print("[进度]: 生成混淆矩阵和ROC曲线...")
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (TabPFN)')
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # For binary classification with probability prediction support, plot ROC curve
    if len(np.unique(y_test)) == 2 and hasattr(fixed_pipeline, 'predict_proba'):
        from sklearn.metrics import roc_curve, auc
        y_proba = fixed_pipeline.predict_proba(X_test[selected_features])[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (TabPFN)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(results_dir, "roc_curve.png"), dpi=300, bbox_inches='tight')
        plt.close()
    print("[进度]: 图表生成完成 ✓")
    
    print("TabPFN分类模型评估完成!")

    return fixed_pipeline


# ===================== 创建结果目录函数 =====================
def create_results_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"feature_selection_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


# ===================== 特征集评估函数 =====================
def evaluate_feature_sets(X, y, feature_sets, base_model, param_space, cv_outer=None, cv_inner=None, n_trials=None,
                          n_jobs=None):
    # 使用配置参数替代硬编码的默认值
    cv_outer = cv_outer if cv_outer is not None else get_config("cv_outer", get_config("cv"))
    cv_inner = cv_inner if cv_inner is not None else get_config("cv_inner", 3)
    n_trials = n_trials if n_trials is not None else get_config("bayes_iter", 20)
    n_jobs = n_jobs if n_jobs is not None else get_config("n_jobs", -1)

    results = []
    cv_outer_obj = KFold(n_splits=cv_outer, shuffle=True, random_state=get_config("random_state"))
    for feature_set_name, features in feature_sets.items():
        print(f"Evaluating feature set: {feature_set_name} with {len(features)} features")
        cv_scores = []
        feature_importances = []
        for train_idx, test_idx in cv_outer_obj.split(X):
            X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
            y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
            pipeline = Pipeline([("scaler", StandardScaler()), ("model", base_model)])

            def objective(trial):
                params = suggest_params(trial, param_space)
                pipeline.set_params(**params)
                scores = cross_val_score(pipeline, X_train_cv[features], y_train_cv, scoring="neg_mean_squared_error",
                                         cv=cv_inner, n_jobs=n_jobs)
                return np.mean(scores)

            study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_warmup_steps=5))
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_trial.params
            pipeline.set_params(**best_params)
            pipeline.fit(X_train_cv[features], y_train_cv)
            y_pred = pipeline.predict(X_test_cv[features])
            mse = mean_squared_error(y_test_cv, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_cv, y_pred)
            cv_scores.append({
                "mse": mse,
                "rmse": rmse,
                "r2": r2
            })
            model_fitted = pipeline.named_steps["model"]
            if hasattr(model_fitted, 'feature_importances_'):
                feature_importances.append(dict(zip(features, model_fitted.feature_importances_)))
            elif hasattr(model_fitted, 'coef_'):
                if len(model_fitted.coef_.shape) > 1:
                    importances = np.sqrt(np.mean(model_fitted.coef_ ** 2, axis=0))
                else:
                    importances = np.abs(model_fitted.coef_)
                feature_importances.append(dict(zip(features, importances)))
        mean_rmse = np.mean([s["rmse"] for s in cv_scores])
        std_rmse = np.std([s["rmse"] for s in cv_scores])
        mean_r2 = np.mean([s["r2"] for s in cv_scores])
        std_r2 = np.std([s["r2"] for s in cv_scores])
        agg_importances = {}
        if feature_importances:
            all_features = set()
            for imp_dict in feature_importances:
                all_features.update(imp_dict.keys())
            for feature in all_features:
                values = [imp.get(feature, 0) for imp in feature_importances if feature in imp]
                if values:
                    agg_importances[feature] = np.mean(values)
        results.append({
            "feature_set": feature_set_name,
            "features": features,
            "mean_rmse": mean_rmse,
            "std_rmse": std_rmse,
            "mean_r2": mean_r2,
            "std_r2": std_r2,
            "num_features": len(features),
            "feature_importances": agg_importances
        })
        print(f"  Results: RMSE = {mean_rmse:.4f} ± {std_rmse:.4f}, R² = {mean_r2:.4f} ± {std_r2:.4f}")
    results = sorted(results, key=lambda x: x["mean_rmse"])
    return results


# ===================== 最终模型评估函数 =====================
def final_model_evaluation_classification(X_train, y_train, X_test, y_test, selected_features, base_model, best_params,
                                          cv, n_jobs, results_dir):
    """
    Perform classification model evaluation using the best parameters for the selected feature set

    Parameters:
    X_train: Training features
    y_train: Training labels
    X_test: Test features
    y_test: Test labels
    selected_features: Selected feature subset
    base_model: Base model object
    best_params: Dictionary of best hyperparameters (from optimize_model function)
    cv: Number of cross-validation folds
    n_jobs: Number of parallel jobs
    results_dir: Directory to save results

    Returns:
    pipeline: Trained model pipeline
    """
    print("\n=== Final Classification Model Evaluation ===")
    print(f"Evaluating {len(selected_features)} selected features...")

    # Get categorical and numerical features from selected features
    categorical_num_cols = get_config("categorical_num_cols", [])
    categorical_subset = [col for col in selected_features if col in categorical_num_cols]
    numerical_subset = [col for col in selected_features if col not in categorical_num_cols]

    # Create preprocessor using only the selected feature subset
    preprocessor = create_preprocessor(numerical_subset, categorical_subset)

    # Create Pipeline with preprocessor
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", base_model)
    ])

    # Set model with previously optimized best parameters
    pipeline.set_params(**best_params)
    print("Applying best hyperparameters:", best_params)

    # Use stratified cross-validation for model evaluation
    print("Performing cross-validation evaluation on training set...")
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=get_config("random_state"))

    # Choose appropriate scoring metric
    if len(np.unique(y_train)) <= 2:
        scoring = "f1"
    else:
        scoring = "f1_weighted"

    # Perform cross-validation
    cv_scores = cross_val_score(
        pipeline,
        X_train[selected_features],
        y_train,
        scoring=scoring,
        cv=cv_obj,
        n_jobs=n_jobs
    )
    cv_f1 = np.mean(cv_scores)
    cv_f1_std = np.std(cv_scores)
    print(f"Cross-validation {scoring}: {cv_f1:.4f} ± {cv_f1_std:.4f}")

    # Train model on full training set
    print("Training final model on complete training set...")
    pipeline.fit(X_train[selected_features], y_train)

    # Save model and feature information as dictionary
    model_dict = {
        "model": pipeline,
        "feature_names": selected_features
    }
    dump(model_dict, os.path.join(results_dir, "final_model_with_metadata.pkl"))
    print("Final model saved as 'final_model_with_metadata.pkl'")

    # Evaluate performance on training and test sets
    y_train_pred = pipeline.predict(X_train[selected_features])
    y_test_pred = pipeline.predict(X_test[selected_features])

    # Calculate training set metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    if len(np.unique(y_train)) <= 2:
        train_f1 = f1_score(y_train, y_train_pred, average='binary', zero_division=0)
        train_precision = precision_score(y_train, y_train_pred, average='binary', zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, average='binary', zero_division=0)
    else:
        train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
        train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)

    # Calculate test set metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    if len(np.unique(y_test)) <= 2:
        test_f1 = f1_score(y_test, y_test_pred, average='binary', zero_division=0)
        test_precision = precision_score(y_test, y_test_pred, average='binary', zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average='binary', zero_division=0)
    else:
        test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)

    # Print performance metrics
    print("\nFinal Model Performance:")
    print("Training set metrics:")
    print(f"  Accuracy: {train_accuracy:.4f}")
    print(f"  F1 Score: {train_f1:.4f}")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  Recall: {train_recall:.4f}")

    print("\nTest set metrics:")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")

    # Confusion matrix visualization
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Test Set Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # For binary classification with probability prediction support, plot ROC curve
    if len(np.unique(y_test)) == 2 and hasattr(pipeline, 'predict_proba'):
        from sklearn.metrics import roc_curve, auc
        y_proba = pipeline.predict_proba(X_test[selected_features])[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(results_dir, "roc_curve.png"), dpi=300, bbox_inches='tight')
        plt.close()

    return pipeline


# ===================== 递归特征消除 (RFE) 函数 =====================
def rfe_feature_selection(X_train, y_train, features, fixed_pipeline, cv, n_jobs, results_dir, task_type="regression",
                          original_preprocessor=None):
    """
    使用递归特征消除(RFE)方法进行特征选择。
    RFE从所有特征开始，每次移除最不重要的特征，直到达到指定数量的特征。
    使用交叉验证评估每个特征子集的性能。

    参数:
    X_train: 训练数据特征 DataFrame
    y_train: 训练数据标签 Series
    features: 初始特征列表，在此基础上进行RFE
    fixed_pipeline: 优化好的模型管道，主要用于提取最终模型的训练步骤
    cv: 交叉验证折数
    n_jobs: 并行作业数
    results_dir: 结果保存目录
    task_type: 任务类型，'regression'或'classification'
    original_preprocessor: 针对所有初始特征创建的预处理器，如果为None，将根据features重新创建
    """
    from sklearn.feature_selection import RFE

    print("\n=== 递归特征消除 (RFE) ===")
    print(f"开始对 {len(features)} 个特征进行递归特征消除...")

    # 使用配置的随机种子确保结果可复现
    random_state = get_config("random_state")

    # 初始化结果列表
    results = []

    # 确保有预处理器
    if original_preprocessor is None:
        # 获取所有列和类别列
        categorical_num_cols = get_config("categorical_num_cols", [])

        # 计算数值列和类别列
        categorical_subset = [col for col in features if col in categorical_num_cols]
        numerical_subset = [col for col in features if col not in categorical_num_cols]

        # 创建预处理器
        original_preprocessor = create_preprocessor(numerical_subset, categorical_subset)
        print("已创建针对所有初始特征的预处理器")

    # 创建简单的RFE估计器（不包含预处理）
    if task_type == "regression":
        simple_rfe_estimator = RandomForestRegressor(
            n_estimators=10,
            max_depth=5,
            random_state=random_state
        )
    else:
        simple_rfe_estimator = RandomForestClassifier(
            n_estimators=10,
            max_depth=5,
            random_state=random_state
        )

    print(f"使用 {type(simple_rfe_estimator).__name__} 作为RFE基础估计器")

    # 对不同特征数量进行评估
    max_features = min(20, len(features))  # 最多考虑20个特征，避免计算量过大

    # 创建进度条
    from tqdm import tqdm
    print(f"评估从1到{max_features}个特征的性能:")

    for n_features_to_select in tqdm(range(1, max_features + 1)):
        # 构建RFE评估Pipeline
        evaluation_pipeline = Pipeline([
            # 第1步: 应用针对所有初始特征的预处理器
            ("preprocessor", clone(original_preprocessor)),
            # 第2步: 使用RFE进行特征选择
            ("feature_selection", RFE(
                estimator=clone(simple_rfe_estimator),
                n_features_to_select=n_features_to_select,
                step=1
            )),
            # 第3步: 使用最终优化好的模型进行训练/预测
            ("final_model", clone(fixed_pipeline.named_steps['model']))
        ])

        # 使用交叉验证评估当前特征数量
        cv_results = robust_cross_validation(
            X_train[features],
            y_train,
            evaluation_pipeline,
            cv=cv,
            task_type=task_type,
            n_jobs=n_jobs
        )

        # 根据任务类型提取性能指标
        if task_type == "regression":
            # 回归任务指标
            rmse = -cv_results['neg_rmse']['mean']  # 转换为正的RMSE
            rmse_std = cv_results['neg_rmse']['std']
            r2 = cv_results['r2']['mean']
            r2_std = cv_results['r2']['std']

            # 记录中间结果
            print(
                f"  特征数量: {n_features_to_select}, CV-RMSE: {rmse:.4f} ± {rmse_std:.4f}, CV-R2: {r2:.4f} ± {r2_std:.4f}")

            # 创建最终选择器Pipeline获取特征名称
            selection_pipeline = Pipeline([
                ("preprocessor", clone(original_preprocessor)),
                ("rfe", RFE(
                    estimator=clone(simple_rfe_estimator),
                    n_features_to_select=n_features_to_select,
                    step=1
                ))
            ])

            # 训练选择器Pipeline
            selection_pipeline.fit(X_train[features], y_train)

            # 获取RFE支持特征的掩码
            mask = selection_pipeline.named_steps['rfe'].support_

            # 获取转换后的特征名
            try:
                transformed_names = selection_pipeline.named_steps['preprocessor'].get_feature_names_out()
                # 应用掩码获取选定的转换后特征名
                selected_transformed_names = transformed_names[mask]

                # 将转换后的特征名映射回原始特征名
                original_selected_features = set()
                for feat in selected_transformed_names:
                    # 处理独热编码特征格式，如 "cat__feature_value"
                    if '__' in feat:
                        parts = feat.split('__', 1)
                        if len(parts) == 2:
                            prefix, name = parts
                            # 从完整名称中提取原始特征名
                            for orig_feat in features:
                                if name.startswith(orig_feat + '_') or name == orig_feat:
                                    original_selected_features.add(orig_feat)
                                    break
                    else:
                        # 对于数值特征，尝试直接匹配
                        for orig_feat in features:
                            if feat == orig_feat or feat.endswith('__' + orig_feat):
                                original_selected_features.add(orig_feat)
                                break

                # 转换为列表并保持原始顺序
                selected_features = [f for f in features if f in original_selected_features]
                print(f"    选择了 {len(selected_features)} 个原始特征")

            except Exception as e:
                print(f"警告: 无法获取转换后的特征名: {e}")
                # 直接使用RFE的support_掩码选择原始特征
                selected_features = [features[i] for i in range(len(features)) if mask[i]]

            # 存储结果
            results.append({
                "n_features": n_features_to_select,
                "features": selected_features,
                "rmse": rmse,
                "rmse_std": rmse_std,
                "r2": r2,
                "r2_std": r2_std,
                "num_features": len(selected_features)
            })

        else:  # 分类任务
            # 分类任务指标
            if len(np.unique(y_train)) <= 2:
                # 二分类指标
                accuracy = cv_results['accuracy']['mean']
                accuracy_std = cv_results['accuracy']['std']
                f1 = cv_results['f1']['mean']
                f1_std = cv_results['f1']['std']
                precision = cv_results.get('precision', {}).get('mean', 0)
                recall = cv_results.get('recall', {}).get('mean', 0)

                # 记录中间结果
                print(
                    f"  特征数量: {n_features_to_select}, CV-Accuracy: {accuracy:.4f} ± {accuracy_std:.4f}, CV-F1: {f1:.4f} ± {f1_std:.4f}")
            else:
                # 多分类指标
                accuracy = cv_results['accuracy']['mean']
                accuracy_std = cv_results['accuracy']['std']
                f1 = cv_results['f1_weighted']['mean']
                f1_std = cv_results['f1_weighted']['std']
                precision = cv_results.get('precision_weighted', {}).get('mean', 0)
                recall = cv_results.get('recall_weighted', {}).get('mean', 0)

                # 记录中间结果
                print(
                    f"  特征数量: {n_features_to_select}, CV-Accuracy: {accuracy:.4f} ± {accuracy_std:.4f}, CV-F1: {f1:.4f} ± {f1_std:.4f}")

            # 创建最终选择器Pipeline获取特征名称
            selection_pipeline = Pipeline([
                ("preprocessor", clone(original_preprocessor)),
                ("rfe", RFE(
                    estimator=clone(simple_rfe_estimator),
                    n_features_to_select=n_features_to_select,
                    step=1
                ))
            ])

            # 训练选择器Pipeline
            selection_pipeline.fit(X_train[features], y_train)

            # 获取RFE支持特征的掩码
            mask = selection_pipeline.named_steps['rfe'].support_

            # 获取转换后的特征名
            try:
                transformed_names = selection_pipeline.named_steps['preprocessor'].get_feature_names_out()
                # 应用掩码获取选定的转换后特征名
                selected_transformed_names = transformed_names[mask]

                # 将转换后的特征名映射回原始特征名
                original_selected_features = set()
                for feat in selected_transformed_names:
                    # 处理独热编码特征格式，如 "cat__feature_value"
                    if '__' in feat:
                        parts = feat.split('__', 1)
                        if len(parts) == 2:
                            prefix, name = parts
                            # 从完整名称中提取原始特征名
                            for orig_feat in features:
                                if name.startswith(orig_feat + '_') or name == orig_feat:
                                    original_selected_features.add(orig_feat)
                                    break
                    else:
                        # 对于数值特征，尝试直接匹配
                        for orig_feat in features:
                            if feat == orig_feat or feat.endswith('__' + orig_feat):
                                original_selected_features.add(orig_feat)
                                break

                # 转换为列表并保持原始顺序
                selected_features = [f for f in features if f in original_selected_features]
                print(f"    选择了 {len(selected_features)} 个原始特征")

            except Exception as e:
                print(f"警告: 无法获取转换后的特征名: {e}")
                # 直接使用RFE的support_掩码选择原始特征
                selected_features = [features[i] for i in range(len(features)) if mask[i]]

            # 存储结果
            results.append({
                "n_features": n_features_to_select,
                "features": selected_features,
                "accuracy": accuracy,
                "accuracy_std": accuracy_std,
                "f1": f1,
                "f1_std": f1_std,
                "precision": precision,
                "recall": recall,
                "num_features": len(selected_features)
            })

        # 保存中间结果到CSV
        try:
            if task_type == "regression":
                interim_results = pd.DataFrame([{
                    "n_features": r["n_features"],
                    "features": ",".join(r["features"]),
                    "num_features": r["num_features"],
                    "cv_rmse": r["rmse"],
                    "cv_rmse_std": r["rmse_std"],
                    "cv_r2": r["r2"],
                    "cv_r2_std": r["r2_std"]
                } for r in results])
            else:  # classification
                interim_results = pd.DataFrame([{
                    "n_features": r["n_features"],
                    "features": ",".join(r["features"]),
                    "num_features": r["num_features"],
                    "cv_accuracy": r["accuracy"],
                    "cv_accuracy_std": r["accuracy_std"],
                    "cv_f1": r["f1"],
                    "cv_f1_std": r["f1_std"],
                    "cv_precision": r["precision"],
                    "cv_recall": r["recall"]
                } for r in results])

            interim_results.to_csv(os.path.join(results_dir, "rfe_feature_selection_interim.csv"), index=False)
        except Exception as e:
            print(f"警告: 保存中间结果时出错: {e}")

    # 排序结果
    if task_type == "regression":
        results = sorted(results, key=lambda x: x["rmse"])
    else:  # classification
        results = sorted(results, key=lambda x: x["f1"], reverse=True)

    # 使用专门的RFE可视化函数
    plot_rfe_results(results, results_dir, task_type)

    print("\n递归特征消除完成!")
    print(f"最佳特征组合: {results[0]['features']}")
    if task_type == "regression":
        print(f"最佳性能: RMSE = {results[0]['rmse']:.4f}, R2 = {results[0]['r2']:.4f}")
    else:
        print(f"最佳性能: F1 = {results[0]['f1']:.4f}, Accuracy = {results[0]['accuracy']:.4f}")

    return results


# ===================== 递归特征消除 (RFE) 可视化函数 =====================
def plot_rfe_results(results, results_dir, task_type="regression"):
    """
    Plot results from Recursive Feature Elimination (RFE)

    Parameters:
    results: List of dictionaries with results from RFE evaluation
    results_dir: Directory to save plots
    task_type: Type of task (regression or classification)
    """
    # Sort results by feature count for plotting
    results_sorted = sorted(results, key=lambda x: x['n_features'])

    # Extract metrics based on task type
    if task_type == "regression":
        metric_name = "RMSE"
        values = [result.get('rmse', float('inf')) for result in results_sorted]
        std_values = [result.get('rmse_std', 0) for result in results_sorted]
        higher_is_better = False
        best_idx = np.argmin(values)
    else:
        metric_name = "F1 Score"
        values = [result.get('f1', 0) for result in results_sorted]
        std_values = [result.get('f1_std', 0) for result in results_sorted]
        higher_is_better = True
        best_idx = np.argmax(values)

    # Number of features in each set
    n_features = [result['n_features'] for result in results_sorted]

    # Plot performance vs number of features
    plt.figure(figsize=(12, 8))

    # Plot main metric line
    plt.errorbar(n_features, values, yerr=std_values, fmt='-o', ecolor='gray', capsize=5)

    # Highlight best point
    if not higher_is_better:
        # For metrics where lower is better (like RMSE)
        best_value = min(values)
        plt.plot(n_features[best_idx], best_value, 'ro', markersize=10,
                 label=f'Best: {n_features[best_idx]} features, {metric_name}={best_value:.4f}')
    else:
        # For metrics where higher is better (like F1 Score)
        best_value = max(values)
        plt.plot(n_features[best_idx], best_value, 'ro', markersize=10,
                 label=f'Best: {n_features[best_idx]} features, {metric_name}={best_value:.4f}')

    # Add secondary metric if available (e.g., R² for regression)
    if task_type == "regression" and 'r2' in results_sorted[0]:
        secondary_values = [result.get('r2', 0) for result in results_sorted]
        plt.figure(figsize=(12, 8))
        plt.plot(n_features, secondary_values, '-s', color='green', label='R²')
        plt.axhline(y=max(secondary_values), color='green', linestyle='--',
                    alpha=0.5, label=f'Best R²: {max(secondary_values):.4f}')
        plt.xlabel('Number of Features')
        plt.ylabel('R² Score (Higher is better)')
        plt.title('Feature Count vs. R² Score in RFE')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "rfe_r2_curve.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # Back to main metric plot
    plt.figure(figsize=(12, 8))
    plt.errorbar(n_features, values, yerr=std_values, fmt='-o', ecolor='gray', capsize=5)
    plt.plot(n_features[best_idx], values[best_idx], 'ro', markersize=10,
             label=f'Best: {n_features[best_idx]} features, {metric_name}={values[best_idx]:.4f}')

    plt.xlabel('Number of Features')
    plt.ylabel(f'{metric_name} ({("Lower" if not higher_is_better else "Higher")} is better)')
    plt.title(f'Feature Count vs. {metric_name} in RFE')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"rfe_{metric_name.lower()}_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot feature importance if available
    best_features = results_sorted[best_idx].get('selected_features', [])
    feature_importance = results_sorted[best_idx].get('feature_importance', {})

    if best_features and feature_importance:
        # Sort features by importance
        importance_values = [feature_importance.get(feature, 0) for feature in best_features]
        sorted_indices = np.argsort(importance_values)
        sorted_features = [best_features[i] for i in sorted_indices]
        sorted_importance = [importance_values[i] for i in sorted_indices]

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(sorted_features)), sorted_importance, color='skyblue')
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title(f'Feature Importance for Best RFE Model ({n_features[best_idx]} features)')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "rfe_feature_importance.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # Save results to CSV
    results_df = pd.DataFrame(results_sorted)
    results_df.to_csv(os.path.join(results_dir, "rfe_results.csv"), index=False)

    # Save best feature set separately
    if best_features:
        with open(os.path.join(results_dir, "rfe_best_features.txt"), 'w') as f:
            f.write(f"Best number of features: {n_features[best_idx]}\n")
            f.write(f"Best {metric_name}: {values[best_idx]:.4f}\n\n")
            f.write("Selected features:\n")
            for i, feature in enumerate(best_features):
                imp = feature_importance.get(feature, 'N/A')
                if isinstance(imp, (int, float)):
                    f.write(f"{i + 1}. {feature} (Importance: {imp:.4f})\n")
                else:
                    f.write(f"{i + 1}. {feature}\n")


# ===================== 遗传算法特征选择函数 =====================
def genetic_algorithm_feature_selection(X_train, y_train, X_test, y_test, features, fixed_pipeline, cv, n_jobs,
                                        results_dir, task_type="regression"):
    """
    使用遗传算法进行特征选择。
    遗传算法通过模拟自然选择过程来找到最优特征子集。
    """
    import random
    import numpy as np
    import torch
    import gc
    import os
    import warnings
    
    # 强制单线程模式避免Windows上的权限问题
    old_n_jobs = os.environ.get('JOBLIB_NUM_THREADS', None)
    os.environ['JOBLIB_NUM_THREADS'] = '1'  # 强制joblib使用单线程
    
    n_jobs = 1  # 修改函数参数，强制设为单线程
    print(f"遗传算法使用单线程模式以避免权限问题")
    
    # 在Windows系统上降低并行度
    if os.name == 'nt' and n_jobs > 2:
        print(f"在Windows系统上将遗传算法的并行度从{n_jobs}降至2以避免权限问题")
        n_jobs = 2
    
    # 临时禁用GPU加速以避免内存溢出
    print("\n===== 遗传算法阶段临时禁用GPU以避免内存溢出 =====")
    original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 临时禁用所有GPU
    
    # 如果使用的是TabPFN模型，强制使用CPU
    if hasattr(fixed_pipeline, 'named_steps') and 'model' in fixed_pipeline.named_steps:
        model = fixed_pipeline.named_steps['model']
        model_name = type(model).__name__
        if 'TabPFN' in model_name and hasattr(model, 'device'):
            print(f"检测到TabPFN模型，强制设置为CPU模式")
            # 记录原始设备设置
            original_device = model.device
            model.device = "cpu"
    
    # 设置随机种子，确保结果可复现
    random_state = get_config("random_state")
    random.seed(random_state)
    np.random.seed(random_state)

    # 初始化结果列表
    results = []

    print("开始使用遗传算法进行特征选择...")

    # 从配置中获取遗传算法参数
    POPULATION_SIZE = get_config("genetic_population_size", 50)
    GENERATIONS = get_config("genetic_generations", 30)
    INITIAL_SELECTION_RATE = get_config("genetic_initial_rate", 0.5)
    TOURNAMENT_SIZE = get_config("genetic_tournament_size", 3)
    INITIAL_CROSSOVER_RATE = get_config("genetic_initial_crossover_rate", 0.8)
    INITIAL_MUTATION_RATE = get_config("genetic_initial_mutation_rate", 0.1)
    MIN_MUTATION_RATE = get_config("genetic_min_mutation_rate", 0.05)
    MAX_MUTATION_RATE = get_config("genetic_max_mutation_rate", 0.3)
    MIN_CROSSOVER_RATE = get_config("genetic_min_crossover_rate", 0.5)
    MAX_CROSSOVER_RATE = get_config("genetic_max_crossover_rate", 0.9)

    print(f"遗传算法参数:")
    print(f"  种群大小: {POPULATION_SIZE}")
    print(f"  最大代数: {GENERATIONS}")
    print(f"  初始选择率: {INITIAL_SELECTION_RATE}")
    print(f"  锦标赛大小: {TOURNAMENT_SIZE}")
    print(f"  初始交叉率: {INITIAL_CROSSOVER_RATE}")
    print(f"  初始变异率: {INITIAL_MUTATION_RATE}")
    print(f"  变异率范围: [{MIN_MUTATION_RATE}, {MAX_MUTATION_RATE}]")
    print(f"  交叉率范围: [{MIN_CROSSOVER_RATE}, {MAX_CROSSOVER_RATE}]")
    print(f"  随机种子: {random_state}")
    print(f"  交叉验证: {cv}折")

    # 适应度函数 - 评估一个特征子集的性能
    def fitness_function(chromosome):
        # 从染色体获取特征子集
        selected_indices = [i for i, bit in enumerate(chromosome) if bit]

        # 如果没有选择任何特征，返回一个很差的适应度
        if len(selected_indices) == 0:
            return -float('inf') if task_type == "regression" else 0

        selected_features_subset = [features[i] for i in selected_indices]
        
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 获取原始模型实例，但不使用预处理器部分
        if hasattr(fixed_pipeline, 'named_steps') and 'model' in fixed_pipeline.named_steps:
            model_class = fixed_pipeline.named_steps["model"].__class__
            model_params = fixed_pipeline.named_steps["model"].get_params()
            model_instance = model_class(**model_params)
        else:
            # 如果没有model步骤，克隆整个pipeline作为模型
            model_instance = clone(fixed_pipeline)

        # 根据特征子集创建新的预处理器
        # 从选定特征子集中决定哪些是类别特征
        categorical_cols = get_config("categorical_num_cols", [])
        categorical_subset = [col for col in selected_features_subset if col in categorical_cols]
        numerical_subset = [col for col in selected_features_subset if col not in categorical_cols]

        # 创建新的预处理器，专门针对当前特征子集
        current_preprocessor = create_preprocessor(numerical_subset, categorical_subset)

        # 创建包含当前特征子集预处理器的新Pipeline
        if 'preprocessor' in fixed_pipeline.named_steps:
            temp_pipeline = Pipeline([
                ("preprocessor", current_preprocessor),
                ("model", model_instance)
            ])
        else:
            # 兼容旧版代码的默认行为
            temp_pipeline = Pipeline([
                ("preprocessor", current_preprocessor),
                ("model", model_instance)
            ])

        # 创建交叉验证对象
        if task_type == "regression":
            cv_obj = KFold(n_splits=cv, shuffle=True, random_state=get_config('random_state'))
            scoring = 'neg_root_mean_squared_error'
        else:  # 分类问题
            cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=get_config('random_state'))
            scoring = 'f1_weighted' if len(np.unique(y_train)) > 2 else 'f1'

        try:
            # 根据X_train的类型选择特征子集
            if hasattr(X_train, 'loc'):  # DataFrame
                X_train_subset = X_train[selected_features_subset]
                X_test_subset = X_test[selected_features_subset] if X_test is not None else None
            else:  # NumPy数组
                feature_indices = [features.index(feat) for feat in selected_features_subset]
                X_train_subset = X_train[:, feature_indices]
                X_test_subset = X_test[:, feature_indices] if X_test is not None else None

            # 进行交叉验证
            cv_scores = cross_val_score(
                temp_pipeline,
                X_train_subset,
                y_train,
                cv=cv_obj,
                scoring=scoring,
                n_jobs=n_jobs
            )

            # 计算平均得分作为适应度
            if task_type == "regression":
                # 对于回归，得分是负的RMSE，需要转换为正数以适应最大化适应度
                fitness = np.mean(cv_scores)  # 已经是负的RMSE
            else:  # 分类问题
                # 对于分类，得分是F1，已经是越高越好
                fitness = np.mean(cv_scores)

        except Exception as e:
            print(f"交叉验证评估出错: {str(e)}")
            # 出错时回退到单次评估
            try:
                # 根据X_train和X_test的类型选择特征子集
                if hasattr(X_train, 'loc'):  # DataFrame
                    X_train_subset = X_train[selected_features_subset]
                    X_test_subset = X_test[selected_features_subset]
                else:  # NumPy数组
                    feature_indices = [features.index(feat) for feat in selected_features_subset]
                    X_train_subset = X_train[:, feature_indices]
                    X_test_subset = X_test[:, feature_indices]

                temp_pipeline.fit(X_train_subset, y_train)
                y_pred = temp_pipeline.predict(X_test_subset)

                if task_type == "regression":
                    mse = mean_squared_error(y_test, y_pred)
                    fitness = -np.sqrt(mse)
                else:  # 分类问题
                    if len(np.unique(y_train)) <= 2:
                        fitness = f1_score(y_test, y_pred, average='binary', zero_division=0)
                    else:
                        fitness = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            except Exception as inner_e:
                print(f"回退评估也失败: {str(inner_e)}")
                # 如果所有评估方法都失败，返回一个非常差的适应度
                return -float('inf') if task_type == "regression" else 0

        # 添加特征数量的惩罚/奖励
        feature_count_penalty = -0.01 * len(selected_indices)

        return fitness + feature_count_penalty

    # 初始化种群
    def initialize_population(size, n_features):
        population = []
        for _ in range(size):
            # 使用INITIAL_SELECTION_RATE来决定选择多少特征
            chromosome = [False] * n_features
            # 至少选择一个特征，最多选择基于初始选择率的比例
            max_features = max(1, int(n_features * INITIAL_SELECTION_RATE))
            num_ones = random.randint(1, max_features)
            indices = random.sample(range(n_features), num_ones)
            for idx in indices:
                chromosome[idx] = True
            population.append(chromosome)
        return population

    # 锦标赛选择
    def tournament_selection(population, fitnesses, tournament_size):
        selected = []
        for _ in range(len(population)):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
            selected.append(population[winner_idx])
        return selected

    # 交叉
    def crossover(parent1, parent2, crossover_rate):
        if random.random() > crossover_rate:
            return parent1, parent2

        n = len(parent1)
        crossover_point = random.randint(1, n - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        # 确保每个子代至少有一个特征被选择
        if True not in child1:
            random_index = random.randint(0, n - 1)
            child1[random_index] = True
        if True not in child2:
            random_index = random.randint(0, n - 1)
            child2[random_index] = True

        return child1, child2

    # 变异
    def mutation(chromosome, mutation_rate):
        mutated = chromosome.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = not mutated[i]

        # 确保至少有一个特征被选择
        if True not in mutated:
            random_index = random.randint(0, len(mutated) - 1)
            mutated[random_index] = True

        return mutated

    # 计算种群多样性 (用于自适应参数调整)
    def calculate_diversity(population):
        if not population:
            return 0

        # 计算每个位置的平均值
        avg_bits = [sum(chr[i] for chr in population) / len(population) for i in range(len(population[0]))]

        # 计算多样性 (偏离0.5越远，多样性越低)
        diversity = 1 - (sum(abs(p - 0.5) for p in avg_bits) / len(avg_bits)) * 2
        return diversity

    # 自适应参数调整
    def adaptive_parameters(generation, max_generations, diversity, best_fitness, avg_fitness):
        # 根据进化阶段调整参数
        progress = generation / max_generations

        # 根据多样性和进化进度调整变异率
        # 多样性低或进化早期，提高变异率以增加探索
        mutation_rate = INITIAL_MUTATION_RATE
        if diversity < 0.3:  # 低多样性
            mutation_rate = MIN_MUTATION_RATE + (MAX_MUTATION_RATE - MIN_MUTATION_RATE) * (1 - diversity)

        # 根据适应度差距调整交叉率
        # 如果最佳适应度和平均适应度差距大，降低交叉率以保护优秀个体
        crossover_rate = INITIAL_CROSSOVER_RATE
        if avg_fitness != 0 and best_fitness != 0:
            fitness_ratio = abs(best_fitness - avg_fitness) / max(abs(best_fitness), 1e-10)
            crossover_rate = MAX_CROSSOVER_RATE - fitness_ratio * (MAX_CROSSOVER_RATE - MIN_CROSSOVER_RATE)

        return max(MIN_MUTATION_RATE, min(MAX_MUTATION_RATE, mutation_rate)), max(MIN_CROSSOVER_RATE,
                                                                                  min(MAX_CROSSOVER_RATE,
                                                                                      crossover_rate))

    # 均匀交叉
    def uniform_crossover(parent1, parent2, crossover_rate):
        if random.random() > crossover_rate:
            return parent1, parent2

        n = len(parent1)
        child1 = []
        child2 = []

        for i in range(n):
            # 对每个基因位置，有50%的概率交换父母的基因
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])

        # 确保每个子代至少有一个特征被选择
        if True not in child1:
            random_index = random.randint(0, n - 1)
            child1[random_index] = True
        if True not in child2:
            random_index = random.randint(0, n - 1)
            child2[random_index] = True

        return child1, child2

    # 双点交叉
    def two_point_crossover(parent1, parent2, crossover_rate):
        if random.random() > crossover_rate:
            return parent1, parent2

        n = len(parent1)
        # 确保第一个点小于第二个点
        point1 = random.randint(1, n - 2)
        point2 = random.randint(point1 + 1, n - 1)

        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

        # 确保每个子代至少有一个特征被选择
        if True not in child1:
            random_index = random.randint(0, n - 1)
            child1[random_index] = True
        if True not in child2:
            random_index = random.randint(0, n - 1)
            child2[random_index] = True

        return child1, child2

    # 高级交叉操作 - 根据问题特点自动选择交叉方法
    def advanced_crossover(parent1, parent2, crossover_rate, feature_count):
        # 特征较少时使用单点交叉，特征较多时使用均匀交叉
        if feature_count < 20:
            return crossover(parent1, parent2, crossover_rate)  # 使用原始的单点交叉
        elif feature_count < 50:
            return two_point_crossover(parent1, parent2, crossover_rate)  # 使用双点交叉
        else:
            return uniform_crossover(parent1, parent2, crossover_rate)  # 使用均匀交叉

    # 主遗传算法循环
    n_features = len(features)
    population = initialize_population(POPULATION_SIZE, n_features)

    best_chromosomes = []  # 保存每代的最佳染色体
    best_fitnesses = []  # 保存每代的最佳适应度

    # 初始化交叉率和变异率
    current_crossover_rate = INITIAL_CROSSOVER_RATE
    current_mutation_rate = INITIAL_MUTATION_RATE

    for generation in range(GENERATIONS):
        print(f"遗传算法第 {generation + 1}/{GENERATIONS} 代")

        # 评估适应度
        fitnesses = []
        for chromosome in population:
            fit = fitness_function(chromosome)
            fitnesses.append(fit)

        # 找出当前种群中的最佳染色体
        best_idx = fitnesses.index(max(fitnesses))
        best_chromosome = population[best_idx]
        best_fitness = fitnesses[best_idx]

        best_chromosomes.append(best_chromosome.copy())
        best_fitnesses.append(best_fitness)

        # 计算平均适应度
        avg_fitness = sum(fitnesses) / len(fitnesses)

        # 计算种群多样性并更新参数
        diversity = calculate_diversity(population)
        current_mutation_rate, current_crossover_rate = adaptive_parameters(
            generation, GENERATIONS, diversity, best_fitness, avg_fitness)

        print(
            f"  当前多样性: {diversity:.2f}, 变异率: {current_mutation_rate:.2f}, 交叉率: {current_crossover_rate:.2f}")

        # 保存选定的特征和性能
        selected_indices = [i for i, bit in enumerate(best_chromosome) if bit]
        selected_features_subset = [features[i] for i in selected_indices]

        print(f"  最佳适应度: {best_fitness:.4f}, 特征数量: {len(selected_features_subset)}")
        print(f"  选择的特征: {selected_features_subset}")

        # 选择
        selected = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)

        # 创建新一代
        next_generation = []
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):
                parent1, parent2 = selected[i], selected[i + 1]
                # 使用高级交叉操作
                child1, child2 = advanced_crossover(parent1, parent2, current_crossover_rate, n_features)
                next_generation.append(mutation(child1, current_mutation_rate))
                next_generation.append(mutation(child2, current_mutation_rate))
            else:
                next_generation.append(mutation(selected[i], current_mutation_rate))

        # 精英保留：确保最佳染色体进入下一代
        next_generation[0] = best_chromosome

        population = next_generation

    # 找出所有代中的最佳染色体
    best_generation_idx = best_fitnesses.index(max(best_fitnesses))
    overall_best_chromosome = best_chromosomes[best_generation_idx]
    
    # 从最佳染色体中提取最佳特征列表
    best_indices = [i for i, bit in enumerate(overall_best_chromosome) if bit]
    best_features = [features[i] for i in best_indices]
    best_fitness = max(best_fitnesses)

    # 计算最终结果
    results = []

    # 将所有生成的独特染色体评估并添加到结果中
    unique_chromosomes = []
    for chromosome in best_chromosomes:
        if chromosome not in unique_chromosomes:
            unique_chromosomes.append(chromosome)

    for chromosome in unique_chromosomes:
        # 从染色体获取特征子集
        selected_indices = [i for i, bit in enumerate(chromosome) if bit]
        if len(selected_indices) == 0:
            continue

        selected_features_subset = [features[i] for i in selected_indices]

        # 使用交叉验证评估最终结果
        if hasattr(fixed_pipeline, 'named_steps') and 'model' in fixed_pipeline.named_steps:
            model_class = fixed_pipeline.named_steps["model"].__class__
            model_params = fixed_pipeline.named_steps["model"].get_params()
            model_instance = model_class(**model_params)
        else:
            model_instance = clone(fixed_pipeline)

        temp_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy='mean')),
            ("scaler", StandardScaler()),
            ("model", model_instance)
        ])

        if task_type == "regression":
            # 回归任务
            # 进行交叉验证评估
            cv_obj = KFold(n_splits=cv, shuffle=True, random_state=get_config('random_state'))

            try:
                # 交叉验证评估RMSE
                rmse_scores = -cross_val_score(
                    temp_pipeline,
                    X_train[selected_features_subset],
                    y_train,
                    cv=cv_obj,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=n_jobs
                )
                rmse = np.mean(rmse_scores)

                # 交叉验证评估R²
                r2_scores = cross_val_score(
                    temp_pipeline,
                    X_train[selected_features_subset],
                    y_train,
                    cv=cv_obj,
                    scoring='r2',
                    n_jobs=n_jobs
                )
                r2 = np.mean(r2_scores)

                print(f"特征子集 {len(selected_features_subset)}个特征，交叉验证RMSE: {rmse:.4f}, R²: {r2:.4f}")

            except Exception as e:
                print(f"交叉验证评估出错: {str(e)}，使用单次评估")
                # 出错时使用传统的单次训练测试分割
                temp_pipeline.fit(X_train[selected_features_subset], y_train)
                y_pred = temp_pipeline.predict(X_test[selected_features_subset])
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

            results.append({
                "features": selected_features_subset,
                "rmse": rmse,
                "r2": r2,
                "num_features": len(selected_features_subset)
            })
        else:  # 分类任务
            # 创建交叉验证对象
            cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=get_config('random_state'))

            try:
                # 交叉验证评估准确率
                accuracy_scores = cross_val_score(
                    temp_pipeline,
                    X_train[selected_features_subset],
                    y_train,
                    cv=cv_obj,
                    scoring='accuracy',
                    n_jobs=n_jobs
                )
                accuracy = np.mean(accuracy_scores)

                # 交叉验证评估F1分数
                if len(np.unique(y_train)) <= 2:
                    f1_scores = cross_val_score(
                        temp_pipeline,
                        X_train[selected_features_subset],
                        y_train,
                        cv=cv_obj,
                        scoring='f1',
                        n_jobs=n_jobs
                    )
                    precision_scores = cross_val_score(
                        temp_pipeline,
                        X_train[selected_features_subset],
                        y_train,
                        cv=cv_obj,
                        scoring='precision',
                        n_jobs=n_jobs
                    )
                    recall_scores = cross_val_score(
                        temp_pipeline,
                        X_train[selected_features_subset],
                        y_train,
                        cv=cv_obj,
                        scoring='recall',
                        n_jobs=n_jobs
                    )
                else:
                    f1_scores = cross_val_score(
                        temp_pipeline,
                        X_train[selected_features_subset],
                        y_train,
                        cv=cv_obj,
                        scoring='f1_weighted',
                        n_jobs=n_jobs
                    )
                    precision_scores = cross_val_score(
                        temp_pipeline,
                        X_train[selected_features_subset],
                        y_train,
                        cv=cv_obj,
                        scoring='precision_weighted',
                        n_jobs=n_jobs
                    )
                    recall_scores = cross_val_score(
                        temp_pipeline,
                        X_train[selected_features_subset],
                        y_train,
                        cv=cv_obj,
                        scoring='recall_weighted',
                        n_jobs=n_jobs
                    )

                f1 = np.mean(f1_scores)
                precision = np.mean(precision_scores)
                recall = np.mean(recall_scores)

                print(f"特征子集 {len(selected_features_subset)}个特征，交叉验证准确率: {accuracy:.4f}, F1: {f1:.4f}")

                # 训练模型用于预测测试集进行最终评估并获取AUC
                temp_pipeline.fit(X_train[selected_features_subset], y_train)
                y_pred = temp_pipeline.predict(X_test[selected_features_subset])

                # 尝试计算AUC
                auc = 0
                if hasattr(temp_pipeline, 'predict_proba'):
                    try:
                        y_proba = temp_pipeline.predict_proba(X_test[selected_features_subset])
                        if len(np.unique(y_train)) <= 2:
                            if y_proba.shape[1] >= 2:
                                auc = roc_auc_score(y_test, y_proba[:, 1])
                        else:
                            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                    except:
                        pass

            except Exception as e:
                print(f"交叉验证评估出错: {str(e)}，使用单次评估")
                # 出错时使用传统的单次训练测试分割
                temp_pipeline.fit(X_train[selected_features_subset], y_train)
                y_pred = temp_pipeline.predict(X_test[selected_features_subset])

                accuracy = accuracy_score(y_test, y_pred)
                if len(np.unique(y_train)) <= 2:
                    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
                    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
                    auc = 0
                    if hasattr(temp_pipeline, 'predict_proba'):
                        y_proba = temp_pipeline.predict_proba(X_test[selected_features_subset])
                        if y_proba.shape[1] >= 2:
                            auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    auc = 0
                    if hasattr(temp_pipeline, 'predict_proba'):
                        try:
                            y_proba = temp_pipeline.predict_proba(X_test[selected_features_subset])
                            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                        except:
                            pass

            results.append({
                "features": selected_features_subset,
                "accuracy": accuracy,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "auc": auc,
                "num_features": len(selected_features_subset)
            })

    # 排序结果
    if task_type == "regression":
        results = sorted(results, key=lambda x: x["rmse"])
    else:  # classification
        results = sorted(results, key=lambda x: x["f1"], reverse=True)

    # 可视化遗传算法结果
    plot_genetic_algorithm_results(results, best_chromosomes, best_fitnesses, results_dir, task_type)

    # 保存结果
    if task_type == "regression":
        interim_results = pd.DataFrame([{
            "features": ",".join(r["features"]),
            "num_features": r["num_features"],
            "rmse": r["rmse"],
            "r2": r["r2"]
        } for r in results])
    else:  # classification
        interim_results = pd.DataFrame([{
            "features": ",".join(r["features"]),
            "num_features": r["num_features"],
            "accuracy": r["accuracy"],
            "f1": r["f1"],
            "precision": r["precision"],
            "recall": r["recall"],
            "auc": r["auc"] if "auc" in r else 0
        } for r in results])

    interim_results.to_csv(os.path.join(results_dir, "genetic_algorithm_results.csv"), index=False)

    # 返回结果
    print("\n遗传算法特征选择完成!")
    print(f"最佳特征子集 ({len(best_features)}): {best_features}")
    print(f"最佳适应度: {best_fitness}")
    
    # 恢复原始GPU设置
    print("恢复原始GPU设置")
    os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
    
    # 如果之前修改了TabPFN模型的设备设置，恢复它
    if hasattr(fixed_pipeline, 'named_steps') and 'model' in fixed_pipeline.named_steps:
        model = fixed_pipeline.named_steps['model']
        model_name = type(model).__name__
        if 'TabPFN' in model_name and hasattr(model, 'device') and 'original_device' in locals():
            print(f"恢复TabPFN模型到原始设备: {original_device}")
            model.device = original_device
    
    # 清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return best_features, results, best_chromosomes, best_fitnesses


# ===================== 遗传算法特征选择可视化函数 =====================
def plot_genetic_algorithm_results(results, best_chromosomes, best_fitnesses, results_dir, task_type="regression"):
    """
    Visualize genetic algorithm feature selection results
    """
    # 1. Plot evolution of best fitness over generations
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(best_fitnesses) + 1), best_fitnesses, 'b-o')
    plt.xlabel('Generation')

    if task_type == "regression":
        plt.ylabel('Fitness (Negative RMSE)')
        plt.title('Genetic Algorithm Evolution - Best Fitness (Regression)')
    else:
        plt.ylabel('Fitness (F1 Score)')
        plt.title('Genetic Algorithm Evolution - Best Fitness (Classification)')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "ga_evolution.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Performance comparison for different feature counts
    feature_counts = {}

    if task_type == "regression":
        for res in results:
            count = res["num_features"]
            if count not in feature_counts:
                feature_counts[count] = {"rmse": [], "r2": []}
            feature_counts[count]["rmse"].append(res["rmse"])
            feature_counts[count]["r2"].append(res["r2"])

        if feature_counts:  # Only proceed if we have data
            counts = sorted(feature_counts.keys())
            best_rmse = [min(feature_counts[count]["rmse"]) for count in counts]
            best_r2 = [max(feature_counts[count]["r2"]) for count in counts]

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()

            line1, = ax1.plot(counts, best_rmse, 'b-o', label='Best RMSE')
            ax1.set_xlabel('Number of Features')
            ax1.set_ylabel('RMSE', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')

            line2, = ax2.plot(counts, best_r2, 'r-o', label='Best R²')
            ax2.set_ylabel('R²', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            # Find best feature count
            if best_rmse:  # Check if list is not empty
                best_idx = best_rmse.index(min(best_rmse))
                best_features_count = counts[best_idx]

                plt.axvline(x=best_features_count, color='green', linestyle='--',
                            label=f'Best Feature Count: {best_features_count}')

            lines = [line1, line2]
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='best')

            plt.title('Genetic Algorithm - Performance by Feature Count (Regression)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "ga_performance_by_feature_count.png"), dpi=300, bbox_inches='tight')
            plt.close()

    else:  # classification
        for res in results:
            count = res["num_features"]
            if count not in feature_counts:
                feature_counts[count] = {"accuracy": [], "f1": []}
            feature_counts[count]["accuracy"].append(res["accuracy"])
            feature_counts[count]["f1"].append(res["f1"])

        if feature_counts:  # Only proceed if we have data
            counts = sorted(feature_counts.keys())
            best_accuracy = [max(feature_counts[count]["accuracy"]) for count in counts]
            best_f1 = [max(feature_counts[count]["f1"]) for count in counts]

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()

            line1, = ax1.plot(counts, best_accuracy, 'b-o', label='Best Accuracy')
            ax1.set_xlabel('Number of Features')
            ax1.set_ylabel('Accuracy', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')

            line2, = ax2.plot(counts, best_f1, 'r-o', label='Best F1 Score')
            ax2.set_ylabel('F1 Score', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            # Find best feature count
            if best_f1:  # Check if list is not empty
                best_idx = best_f1.index(max(best_f1))
                best_features_count = counts[best_idx]

                plt.axvline(x=best_features_count, color='green', linestyle='--',
                            label=f'Best Feature Count: {best_features_count}')

            lines = [line1, line2]
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='best')

            plt.title('Genetic Algorithm - Performance by Feature Count (Classification)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "ga_performance_by_feature_count.png"), dpi=300, bbox_inches='tight')
            plt.close()

    # 3. Visualize best feature combination - SIMPLIFIED VERSION
    if len(results) > 0:
        best_result = results[0]
        best_features = best_result["features"]

        if len(best_features) > 1:
            plt.figure(figsize=(10, 6))
            y_pos = np.arange(len(best_features))
            # Use a fixed importance for visualization based on feature order
            importances = np.linspace(1.0, 0.1, len(best_features))

            # 优化图形大小和特征排列，解决特征名称重叠问题
            plt.figure(figsize=(10, max(6, len(best_features) * 0.4)))  # 动态调整图高

            # 根据特征数量调整字体大小
            font_size = max(8, min(12, 14 - 0.4 * len(best_features)))

            # 反转顺序，使重要特征显示在顶部
            plt.barh(best_features[::-1], importances[::-1], color='lightgreen')
            plt.xlabel('Feature Importance (Based on Selection Order)')
            plt.ylabel('Features')
            plt.title('Best Feature Combination Selected by Genetic Algorithm')
            plt.yticks(fontsize=font_size)  # 调整特征名称字体大小
            plt.grid(True, linestyle='--', alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "ga_feature_importance.png"), dpi=300, bbox_inches='tight')
            plt.close()


# ===================== 主函数 =====================
def main():
    # 验证配置项
    config_errors = validate_config()
    if config_errors:
        print("配置错误，请修复以下问题后重试:")
        for error in config_errors:
            print(f"  - {error}")
        return

    # GPU 可用性检查
    try:
        import torch
        is_gpu_available = torch.cuda.is_available()
        gpu_device_name = torch.cuda.get_device_name(0) if is_gpu_available else "N/A"
        print("-" * 50)
        print(f"GPU 可用性检查:")
        if is_gpu_available and get_config("use_gpu"):
            print(f"  状态: 检测到可用 GPU ({gpu_device_name}) 并且配置允许使用。")
            print(f"  注意: TabPFN 将自动尝试使用 GPU。")
            print(f"  检查: PyTorch CUDA 版本: {torch.version.cuda}")
        elif is_gpu_available and not get_config("use_gpu"):
            print(f"  状态: 检测到可用 GPU ({gpu_device_name}) 但配置 'use_gpu' 为 False，将使用 CPU。")
        else:
            print(f"  状态: 未检测到可用 GPU 或配置不允许使用，将使用 CPU。")
        print("-" * 50)
    except ImportError:
        print("-" * 50)
        print("警告: 未安装 PyTorch，无法检查 GPU 可用性。TabPFN 可能无法运行或无法使用 GPU。")
        print("-" * 50)
        is_gpu_available = False # 假设不可用
    except Exception as e:
        print(f"GPU 检查时出错: {e}")
        is_gpu_available = False

    results_dir = create_results_dir()
    print(f"Results will be saved in: {results_dir}")

    # 选择任务类型
    task_type_choice = input("选择任务类型 (1: 回归, 2: 分类) [默认: 1]: ").strip()
    if task_type_choice == "2":
        update_config(task_type="classification")
        print("已选择分类任务")
    else:
        update_config(task_type="regression")
        print("已选择回归任务")

    # 读取数据时传递任务类型
    X, y = read_data(task_type=get_config("task_type"))

    # 对于分类任务使用分层分割
    if get_config("task_type") == "classification":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=get_config("test_size"),
                                                            random_state=get_config("random_state"), stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=get_config("test_size"),
                                                            random_state=get_config("random_state"))

    # 获取数值型和类别型特征列
    categorical_num_cols = get_config("categorical_num_cols", [])

    # 获取所有特征列
    all_columns = X_train.columns.tolist()

    # 动态计算数值型特征列（所有列中排除类别型特征列）
    numerical_cols = [col for col in all_columns if col not in categorical_num_cols]

    if not categorical_num_cols:
        print("\n信息: 未指定用数字表示的类别特征列。所有列将作为数值型特征处理。")
        print("  - 数值型特征 ({0}): {1}".format(len(all_columns), all_columns))
        print("  - 数字表示的类别特征 (0): []")
        print("  提示: 如果数据中有用数字表示的类别特征，请在配置中正确设置 'categorical_num_cols'")

    # 创建预处理器
    preprocessor = create_preprocessor(numerical_cols, categorical_num_cols)
    print(f"\n创建预处理器:")
    print(f"  - 数值型特征 ({len(numerical_cols)}): {numerical_cols}")
    print(f"  - 数字表示的类别特征 ({len(categorical_num_cols)}): {categorical_num_cols}")

    # 定义回归模型选项
    regression_models = {
        "1": {"name": "LinearRegression", "model": LinearRegression(), "param_space": {}},
        "2": {"name": "DecisionTreeRegressor", "model": DecisionTreeRegressor(random_state=get_config("random_state")),
              "param_space": {"model__max_depth": (3, 10),
                              "model__min_samples_split": (2, 20),
                              "model__min_samples_leaf": (1, 10),
                              "model__max_features": (0.1, 1.0),
                              "model__ccp_alpha": (0.0, 0.05)}},
        "3": {"name": "RandomForestRegressor",
              "model": RandomForestRegressor(n_jobs=get_config("n_jobs"), random_state=get_config("random_state")),
              "param_space": {"model__n_estimators": (50, 200),
                              "model__max_depth": (3, 10),
                              "model__min_samples_split": (2, 20),
                              "model__min_samples_leaf": (1, 10),
                              "model__max_features": (0.1, 1.0),
                              "model__bootstrap": [True, False],
                              "model__ccp_alpha": (0.0, 0.05)}},
        "4": {"name": "GradientBoostingRegressor",
              "model": GradientBoostingRegressor(random_state=get_config("random_state")),
              "param_space": {"model__n_estimators": (50, 200),
                              "model__learning_rate": (0.01, 0.3, 'log-uniform'),
                              "model__max_depth": (3, 8),
                              "model__min_samples_split": (2, 20),
                              "model__min_samples_leaf": (1, 10),
                              "model__subsample": (0.5, 1.0),
                              "model__max_features": (0.1, 1.0),
                              "model__alpha": (0.1, 0.9)}},
        "5": {"name": "AdaBoostRegressor", "model": AdaBoostRegressor(random_state=get_config("random_state")),
              "param_space": {"model__n_estimators": (50, 200),
                              "model__learning_rate": (0.01, 0.3, 'log-uniform'),
                              "model__loss": ["linear", "square", "exponential"]}},
        "6": {"name": "XGBRegressor",
              "model": XGBRegressor(
                  n_jobs=get_config("n_jobs"),
                  random_state=get_config("random_state"),
                  # 添加GPU参数（如果可用且已配置）
                  **({"device": "cuda"} if get_config("use_gpu") and is_gpu_available and XGBRegressor is not None else {})
              ) if XGBRegressor is not None else None,
              "param_space": {"model__n_estimators": (50, 200),
                              "model__max_depth": (3, 8),
                              "model__learning_rate": (0.01, 0.3, 'log-uniform'),
                              "model__subsample": (0.5, 1.0),
                              "model__colsample_bytree": (0.5, 1.0),
                              "model__min_child_weight": (1, 10),
                              "model__reg_alpha": (0.0001, 1.0, 'log-uniform'),
                              "model__reg_lambda": (0.0001, 1.0, 'log-uniform'),
                              "model__gamma": (0.0, 5.0)}},
        "7": {"name": "SVR", "model": SVR(),
              "param_space": {"model__C": (0.1, 100, 'log-uniform'),
                              "model__gamma": (0.001, 1, 'log-uniform'),
                              "model__kernel": ["rbf", "linear", "poly"],
                              "model__epsilon": (0.01, 0.5)}},
        "8": {"name": "TabPFN",
              "model": TabPFNRegressor(
                  # TabPFN会自动使用PyTorch后端的GPU，如果可用
                  device="cuda" if is_gpu_available and get_config("use_gpu") else "cpu"
              ) if TabPFNRegressor is not None else None,
              "param_space": {}},  # TabPFN 不进行超参优化
        "9": {"name": "ElasticNet", "model": ElasticNet(random_state=get_config("random_state")),
              "param_space": {"model__alpha": (0.0001, 1.0, 'log-uniform'),
                              "model__l1_ratio": (0.1, 0.9),
                              "model__max_iter": (1000, 5000),
                              "model__tol": (1e-5, 1e-3, 'log-uniform')}},
        "10": {"name": "Ridge", "model": Ridge(random_state=get_config("random_state")),
               "param_space": {"model__alpha": (0.1, 10.0, 'log-uniform'),
                               "model__solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]}},
        "11": {"name": "Lasso", "model": Lasso(random_state=get_config("random_state")),
               "param_space": {"model__alpha": (0.0001, 1.0, 'log-uniform'),
                               "model__max_iter": (1000, 5000),
                               "model__tol": (1e-5, 1e-3, 'log-uniform')}},
        "12": {"name": "KNeighborsRegressor", "model": KNeighborsRegressor(),
               "param_space": {"model__n_neighbors": (3, 15),
                               "model__weights": ["uniform", "distance"],
                               "model__p": [1, 2],
                               "model__leaf_size": (10, 50)}},
        "13": {"name": "MLPRegressor",
               "model": MLPRegressor(hidden_layer_sizes=(100, 50),
                                     activation='relu',
                                     solver='adam',
                                     alpha=0.001,
                                     learning_rate='adaptive',
                                     max_iter=2000,
                                     early_stopping=True,
                                     validation_fraction=0.1,
                                     n_iter_no_change=10,
                                     random_state=get_config("random_state")),
               "param_space": {"model__alpha": (0.0001, 0.1, 'log-uniform'),
                               "model__learning_rate_init": (0.0001, 0.1, 'log-uniform'),
                               "model__hidden_layer_sizes": [(50,), (100,), (50, 25), (100, 50)],
                               "model__activation": ["tanh", "relu"],
                               "model__batch_size": [32, 64, 128, 256],
                               "model__beta_1": (0.8, 0.999),
                               "model__beta_2": (0.99, 0.9999)}},
        "14": {"name": "HistGradientBoostingRegressor",
               "model": HistGradientBoostingRegressor(random_state=get_config("random_state")),
               "param_space": {"model__learning_rate": (0.01, 0.3, 'log-uniform'),
                               "model__max_depth": (3, 10),
                               "model__max_iter": (50, 200),
                               "model__min_samples_leaf": (1, 20),
                               "model__l2_regularization": (0.0, 10.0),
                               "model__max_bins": (100, 255)}},
        "15": {"name": "ExtraTreesRegressor",
               "model": ExtraTreesRegressor(n_jobs=get_config("n_jobs"), random_state=get_config("random_state")),
               "param_space": {"model__n_estimators": (50, 200),
                               "model__max_depth": (3, 15),
                               "model__min_samples_split": (2, 20),
                               "model__min_samples_leaf": (1, 10),
                               "model__max_features": (0.1, 1.0),
                               "model__bootstrap": [True, False],
                               "model__ccp_alpha": (0.0, 0.05)}},
        "16": {"name": "GaussianProcessRegressor",
               "model": GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), random_state=get_config("random_state")),
               "param_space": {"model__alpha": (1e-10, 1.0, 'log-uniform'),
                               "model__kernel__k1__length_scale": (0.1, 10.0, 'log-uniform'),
                               "model__kernel__k2__noise_level": (1e-10, 1.0, 'log-uniform')}},
        "17": {"name": "LGBMRegressor",
               "model": LGBMRegressor(
                   n_jobs=get_config("n_jobs"),
                   random_state=get_config("random_state"),
                   # 添加GPU参数
                   # 注意：LightGBM使用GPU需要专门编译支持GPU的版本
                   # 如果出现错误，可以尝试改回CPU模式
                   **({"device": "gpu"} if get_config("use_gpu") and is_gpu_available and LGBMRegressor is not None else {})
               ) if LGBMRegressor is not None else None,
               "param_space": {"model__learning_rate": (0.01, 0.3, 'log-uniform'),
                               "model__n_estimators": (50, 200),
                               "model__num_leaves": (20, 50),
                               "model__subsample": (0.5, 1.0),
                               "model__colsample_bytree": (0.5, 1.0),
                               "model__min_child_samples": (5, 30),
                               "model__reg_alpha": (0.0, 1.0),
                               "model__reg_lambda": (0.0, 1.0),
                               "model__max_depth": (3, 8),
                               "model__min_split_gain": (0.0, 0.5)}},
        "18": {"name": "CatBoostRegressor",
               "model": CatBoostRegressor(
                   thread_count=get_config("n_jobs"), 
                   random_seed=get_config("random_state"),
                   verbose=False,
                   # 添加GPU参数
                   **({"task_type": "GPU", "devices": "0"} if get_config("use_gpu") and is_gpu_available and CatBoostRegressor is not None else {})
               ) if CatBoostRegressor is not None else None,
               "param_space": {"model__learning_rate": (0.01, 0.3, 'log-uniform'),
                               "model__depth": (4, 10),
                               "model__iterations": (50, 200),
                               "model__l2_leaf_reg": (1, 10),
                               "model__random_strength": (0.1, 10.0),
                               "model__bagging_temperature": (0, 10),
                               "model__subsample": (0.5, 1.0)}},
    }

    # 定义分类模型选项
    classification_models = {
        "1": {"name": "LogisticRegression",
              "model": LogisticRegression(random_state=get_config("random_state")),
              "param_space": {"model__C": (0.1, 10.0, 'log-uniform'),
                              "model__solver": ["liblinear", "saga"],
                              "model__penalty": ["l1", "l2", "elasticnet"],
                              "model__l1_ratio": (0.0, 1.0),
                              "model__max_iter": (1000, 5000)}},
        "2": {"name": "DecisionTreeClassifier",
              "model": DecisionTreeClassifier(random_state=get_config("random_state")),
              "param_space": {"model__max_depth": (3, 10),
                              "model__min_samples_split": (2, 20),
                              "model__min_samples_leaf": (1, 10),
                              "model__max_features": (0.1, 1.0),
                              "model__class_weight": [None, "balanced"],
                              "model__ccp_alpha": (0.0, 0.05)}},
        "3": {"name": "RandomForestClassifier",
              "model": RandomForestClassifier(n_jobs=get_config("n_jobs"), random_state=get_config("random_state")),
              "param_space": {"model__n_estimators": (50, 200),
                              "model__max_depth": (3, 10),
                              "model__min_samples_split": (2, 20),
                              "model__min_samples_leaf": (1, 10),
                              "model__max_features": (0.1, 1.0),
                              "model__class_weight": [None, "balanced", "balanced_subsample"],
                              "model__bootstrap": [True, False],
                              "model__ccp_alpha": (0.0, 0.05)}},
        "4": {"name": "GradientBoostingClassifier",
              "model": GradientBoostingClassifier(random_state=get_config("random_state")),
              "param_space": {"model__n_estimators": (50, 200),
                              "model__learning_rate": (0.01, 0.3, 'log-uniform'),
                              "model__max_depth": (3, 8),
                              "model__min_samples_split": (2, 20),
                              "model__min_samples_leaf": (1, 10),
                              "model__subsample": (0.5, 1.0),
                              "model__max_features": (0.1, 1.0)}},
        "5": {"name": "AdaBoostClassifier",
              "model": AdaBoostClassifier(random_state=get_config("random_state")),
              "param_space": {"model__n_estimators": (50, 200),
                              "model__learning_rate": (0.01, 0.3, 'log-uniform')}},
        "6": {"name": "XGBClassifier",
              "model": XGBClassifier(
                  n_jobs=get_config("n_jobs"),
                  random_state=get_config("random_state"),
                  # 添加GPU参数（如果可用且已配置）
                  **({"device": "cuda"} if get_config("use_gpu") and is_gpu_available and XGBClassifier is not None else {})
              ) if XGBClassifier is not None else None,
              "param_space": {"model__n_estimators": (50, 200),
                              "model__max_depth": (3, 8),
                              "model__learning_rate": (0.01, 0.3, 'log-uniform'),
                              "model__subsample": (0.5, 1.0),
                              "model__colsample_bytree": (0.5, 1.0),
                              "model__min_child_weight": (1, 10),
                              "model__reg_alpha": (0.0001, 1.0, 'log-uniform'),
                              "model__reg_lambda": (0.0001, 1.0, 'log-uniform'),
                              "model__gamma": (0.0, 5.0),
                              "model__scale_pos_weight": (0.5, 2.0)}},
        "7": {"name": "SVC",
              "model": SVC(probability=True, random_state=get_config("random_state")),
              "param_space": {"model__C": (0.1, 100, 'log-uniform'),
                              "model__gamma": (0.001, 1, 'log-uniform'),
                              "model__kernel": ["rbf", "linear", "poly"],
                              "model__degree": (2, 5),
                              "model__class_weight": [None, "balanced"]}},
        "8": {"name": "TabPFN",
              "model": TabPFNClassifier(
                  # TabPFN会自动使用PyTorch后端的GPU，如果可用
                  device="cuda" if is_gpu_available and get_config("use_gpu") else "cpu"
              ) if TabPFNClassifier is not None else None,
              "param_space": {}},
        "9": {"name": "KNeighborsClassifier",
              "model": KNeighborsClassifier(),
              "param_space": {"model__n_neighbors": (3, 15),
                              "model__weights": ["uniform", "distance"],
                              "model__p": [1, 2],
                              "model__leaf_size": (10, 50),
                              "model__metric": ["euclidean", "manhattan", "chebyshev", "minkowski"]}},
        "10": {"name": "MLPClassifier",
               "model": MLPClassifier(hidden_layer_sizes=(100, 50),
                                      activation='relu',
                                      solver='adam',
                                      alpha=0.001,
                                      learning_rate='adaptive',
                                      max_iter=2000,
                                      early_stopping=True,
                                      validation_fraction=0.1,
                                      n_iter_no_change=10,
                                      random_state=get_config("random_state")),
               "param_space": {"model__alpha": (0.0001, 0.1, 'log-uniform'),
                               "model__learning_rate_init": (0.0001, 0.1, 'log-uniform'),
                               "model__hidden_layer_sizes": [(50,), (100,), (50, 25), (100, 50)],
                               "model__activation": ["tanh", "relu"],
                               "model__batch_size": [32, 64, 128, 256],
                               "model__beta_1": (0.8, 0.999),
                               "model__beta_2": (0.99, 0.9999)}},
        "11": {"name": "ExtraTreesClassifier",
               "model": ExtraTreesClassifier(n_jobs=get_config("n_jobs"), random_state=get_config("random_state")),
               "param_space": {"model__n_estimators": (50, 200),
                               "model__max_depth": (3, 15),
                               "model__min_samples_split": (2, 20),
                               "model__min_samples_leaf": (1, 10),
                               "model__max_features": (0.1, 1.0),
                               "model__class_weight": [None, "balanced", "balanced_subsample"],
                               "model__bootstrap": [True, False],
                               "model__ccp_alpha": (0.0, 0.05)}},
        "12": {"name": "GaussianProcessClassifier",
               "model": GaussianProcessClassifier(kernel=RBF(), random_state=get_config("random_state")),
               "param_space": {"model__kernel__length_scale": (0.1, 10.0, 'log-uniform'),
                               "model__max_iter_predict": (100, 500),
                               "model__n_restarts_optimizer": (0, 5)}},
        "13": {"name": "LGBMClassifier",
               "model": LGBMClassifier(
                   n_jobs=get_config("n_jobs"),
                   random_state=get_config("random_state"),
                   # 添加GPU参数
                   # 注意：LightGBM使用GPU需要专门编译支持GPU的版本
                   # 如果出现错误，可以尝试改回CPU模式
                   **({"device": "gpu"} if get_config("use_gpu") and is_gpu_available and LGBMClassifier is not None else {})
               ) if LGBMClassifier is not None else None,
               "param_space": {"model__learning_rate": (0.01, 0.3, 'log-uniform'),
                               "model__n_estimators": (50, 200),
                               "model__num_leaves": (20, 50),
                               "model__subsample": (0.5, 1.0),
                               "model__colsample_bytree": (0.5, 1.0),
                               "model__min_child_samples": (5, 30),
                               "model__reg_alpha": (0.0, 1.0),
                               "model__reg_lambda": (0.0, 1.0),
                               "model__max_depth": (3, 8),
                               "model__min_split_gain": (0.0, 0.5),
                               "model__class_weight": [None, "balanced"]}},
        "14": {"name": "CatBoostClassifier",
               "model": CatBoostClassifier(
                   thread_count=get_config("n_jobs"), 
                   random_seed=get_config("random_state"),
                   verbose=False,
                   # 添加GPU参数
                   **({"task_type": "GPU", "devices": "0"} if get_config("use_gpu") and is_gpu_available and CatBoostClassifier is not None else {})
               ) if CatBoostClassifier is not None else None,
               "param_space": {"model__learning_rate": (0.01, 0.3, 'log-uniform'),
                               "model__depth": (4, 10),
                               "model__iterations": (50, 200),
                               "model__l2_leaf_reg": (1, 10),
                               "model__random_strength": (0.1, 10.0),
                               "model__bagging_temperature": (0, 10),
                               "model__subsample": (0.5, 1.0),
                               "model__auto_class_weights": ["None", "Balanced", "SqrtBalanced"]}}
    }

    # 根据任务类型选择模型列表
    model_choices = regression_models if get_config("task_type") == "regression" else classification_models

    print("选择基准模型:")
    for key, model_info in sorted(model_choices.items(), key=lambda x: int(x[0])):
        if model_info["model"] is not None:
            print(f"{key}: {model_info['name']}")
    max_choice = max([int(k) for k in model_choices.keys() if model_choices[k]["model"] is not None])
    choice = input(f"Enter your choice (1-{max_choice}): ").strip()
    if choice not in model_choices:
        print("Invalid choice. Exiting.")
        return
    selected_model_info = model_choices[choice]
    if selected_model_info["model"] is None:
        print(f"{selected_model_info['name']} is not available. Please install the required package.")
        return
    base_model = selected_model_info["model"]
    param_space = selected_model_info["param_space"]
    print(f"Selected model: {selected_model_info['name']}")
    # 如果选择的是TabPFN，则跳过超参数优化
    if selected_model_info["name"] == "TabPFN":
        print("TabPFN model selected: skipping hyperparameter optimization.")
        optimized_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", base_model)
        ])
        optimized_pipeline.fit(X_train, y_train)
    else:
        optimized_pipeline, best_params = optimize_model(X_train, y_train, base_model, param_space,
                                                         get_config("bayes_iter"), get_config("cv"),
                                                         get_config("n_jobs"),
                                                         results_dir, get_config("task_type"),
                                                         preprocessor=preprocessor)
        print("Best parameters obtained:", best_params)
    filtered_features, importance_df = feature_importance_filter(X_train, y_train, get_config("importance_threshold"),
                                                                 optimized_pipeline, get_config("n_jobs"), results_dir)
    print("Features after importance filtering:", filtered_features)
    importance_df.to_csv(os.path.join(results_dir, "feature_importance_results.csv"), index=False)
    filtered_features_corr = correlation_filtering(X_train, y_train, filtered_features, optimized_pipeline,
                                                   get_config("cv"), get_config("n_jobs"),
                                                   get_config("correlation_threshold"),
                                                   get_config("task_type"), results_dir)
    print("Features after correlation filtering:", filtered_features_corr)
    pd.DataFrame({"Selected_Features": filtered_features_corr}).to_csv(
        os.path.join(results_dir, "correlation_filtered_features.csv"), index=False)

    # 选择特征选择方法
    print("\n选择特征选择方法:")
    print("0: 跳过高级筛选 (使用当前特征集)")
    print("1: 穷举法 (Exhaustive Search)")
    print("2: 递归特征消除 (RFE)")
    print("3: 遗传算法 (Genetic Algorithm)")
    feature_selection_method = input("请选择特征选择方法 (0-3) [默认: 1]: ").strip()

    # 根据用户选择执行相应的特征选择方法
    if feature_selection_method == "0":
        print(f"已选择跳过高级筛选，使用 {len(filtered_features_corr)} 个当前特征进行后续步骤。")
        # 直接使用correlation_filtering返回的特征列表
        best_feature_combo = filtered_features_corr
    elif feature_selection_method == "2":
        print("使用递归特征消除 (RFE) 进行特征选择...")
        results = rfe_feature_selection(X_train, y_train, filtered_features_corr,
                                        optimized_pipeline, get_config("cv"), get_config("n_jobs"), results_dir,
                                        get_config("task_type"), original_preprocessor=preprocessor)
        # 选择RFE结果中的最佳特征组合
        best_feature_combo = results[0]["features"]
    elif feature_selection_method == "3":
        print("使用遗传算法进行特征选择...")
        best_features, results, best_chromosomes, best_fitnesses = genetic_algorithm_feature_selection(
            X_train, y_train, X_test, y_test, filtered_features_corr,
            optimized_pipeline, get_config("cv"), get_config("n_jobs"),
            results_dir,
            get_config("task_type"))
        # 选择遗传算法结果中的最佳特征组合
        best_feature_combo = best_features
    else:
        print("使用穷举法进行特征选择...")
        # 使用固定管道进行特征组合评估，不重复超参数优化
        results = combination_search_parallel_fixed(X_train, y_train, X_test, y_test, filtered_features_corr,
                                                    optimized_pipeline, get_config("cv"), get_config("n_jobs"),
                                                    results_dir,
                                                    get_config("task_type"))
        # 为穷举法使用原始可视化函数
        plot_combination_results(results, results_dir, get_config("task_type"))
        # 选择穷举法结果中的最佳特征组合
        best_feature_combo = results[0]["features"]

    # 显示顶级组合结果(仅当选择非0选项时)
    if feature_selection_method != "0":
        print("Top 5 combinations:")
        if get_config("task_type") == "regression":
            for res in results[:5]:
                print(f"Features: {res['features']}, RMSE: {res['rmse']:.4f}, R2: {res['r2']:.4f}")
        else:  # classification
            for res in results[:5]:
                print(f"Features: {res['features']}, Accuracy: {res['accuracy']:.4f}, F1: {res['f1']:.4f}")

        top_results_df = pd.DataFrame(results[:5])
        top_results_df.to_csv(os.path.join(results_dir, "top_feature_combinations.csv"), index=False)

    # 最终特征集超参数优化选项
    final_features_params_choice = input("\n是否为最终特征子集重新优化超参数? [1: 使用原始超参数, 2: 重新优化] [默认: 1]: ").strip()
    
    # 如果用户选择为最终特征子集重新优化超参数
    if final_features_params_choice == "2" and selected_model_info["name"] != "TabPFN":
        print(f"\n为最终选定的 {len(best_feature_combo)} 个特征重新优化超参数...")
        
        # 创建特征子集的专用预处理器
        categorical_subset = [col for col in best_feature_combo if col in categorical_num_cols]
        numerical_subset = [col for col in best_feature_combo if col not in categorical_num_cols]
        subset_preprocessor = create_preprocessor(numerical_subset, categorical_subset)
        
        # 创建保存最终优化结果的目录
        final_optimization_dir = os.path.join(results_dir, "final_features_optimization")
        os.makedirs(final_optimization_dir, exist_ok=True)
        
        # 重新优化超参数
        optimized_pipeline, best_params = optimize_model(
            X_train[best_feature_combo], y_train, base_model, param_space,
            get_config("bayes_iter"), get_config("cv"), get_config("n_jobs"),
            final_optimization_dir, 
            get_config("task_type"), preprocessor=subset_preprocessor
        )
        print("最终特征子集优化后的超参数:", best_params)
    elif selected_model_info["name"] == "TabPFN" and final_features_params_choice == "2":
        print("TabPFN模型不需要超参数优化，将使用原始模型。")

    # 最终模型评估
    final_eval = input("\n是否进行最终模型评估? (y/n) [默认: y]: ").strip().lower()
    final_model = None

    if final_eval != 'n':
        print(f"对最佳特征组合进行最终模型评估: {best_feature_combo}")

        if get_config("task_type") == "classification":
            if selected_model_info["name"] == "TabPFN":
                final_model = final_model_evaluation_tabpfn_classification(
                    X_train, y_train, X_test, y_test, best_feature_combo,
                    optimized_pipeline, results_dir
                )
            else:
                final_model = final_model_evaluation_classification(
                    X_train, y_train, X_test, y_test, best_feature_combo,
                    base_model, best_params, get_config("cv"), get_config("n_jobs"), results_dir
                )
        else:  # regression
            if selected_model_info["name"] == "TabPFN":
                final_model = final_model_evaluation_tabpfn(
                    X_train, y_train, X_test, y_test, best_feature_combo,
                    optimized_pipeline, results_dir
                )
            else:
                final_model = final_model_evaluation_full(
                    X_train, y_train, X_test, y_test, best_feature_combo,
                    base_model, best_params, get_config("cv"), get_config("n_jobs"), results_dir
                )

    # 可选进行SHAP分析 - 移动到最终模型评估后
    perform_shap = input("\n是否进行SHAP分析? (y/n) [默认: n]: ").strip().lower()
    if perform_shap == 'y':
        print(f"对最佳特征组合进行SHAP分析: {best_feature_combo}")

        # 如果用户跳过了最终模型评估，但希望进行SHAP分析，创建一个临时模型
        if final_model is None:
            print("由于跳过了最终模型评估，正在为SHAP分析创建临时模型...")

            # 创建特征子集的专用预处理器
            categorical_subset = [col for col in best_feature_combo if col in categorical_num_cols]
            numerical_subset = [col for col in best_feature_combo if col not in categorical_num_cols]
            subset_preprocessor = create_preprocessor(numerical_subset, categorical_subset)

            # 创建并训练临时管道
            try:
                if selected_model_info["name"] == "TabPFN":
                    temp_pipeline = Pipeline([
                        ("preprocessor", subset_preprocessor),
                        ("model", base_model)
                    ])
                else:
                    temp_pipeline = Pipeline([
                        ("preprocessor", subset_preprocessor),
                        ("model", base_model)
                    ])
                    # 如果有最佳参数，应用它们
                    if 'best_params' in locals():
                        temp_pipeline.set_params(**best_params)

                # 训练模型
                temp_pipeline.fit(X_train[best_feature_combo], y_train)
                final_model = temp_pipeline
                print("临时模型创建并训练完成，可用于SHAP分析。")
            except Exception as e:
                print(f"创建临时模型时出错: {str(e)}")
                print("无法执行SHAP分析，需要有训练好的最终模型。")
                final_model = None

        # 如果有可用的最终模型，执行SHAP分析
        if final_model is not None:
            # 使用重构后的create_shap_report函数，直接传入最终的pipeline
            shap_importance = create_shap_report(
                final_pipeline=final_model,
                features=best_feature_combo,
                results_dir=results_dir,
                X_train_full=X_train,
                X_test_full=X_test,
                task_type=get_config("task_type"),
                sample_size=get_config("shap_sample_size"),
                plot_type=get_config("shap_plot_type")
            )
        else:
            print("SHAP分析已取消，因为没有可用的模型。")

    print("\nAll feature selection analyses completed! Results saved in:", results_dir)


# ===================== SHAP分析功能 =====================
# perform_shap_analysis函数已被删除，使用新的create_shap_report函数替代

# 一个额外的函数用于创建详细的SHAP分析报告
def create_shap_report(final_pipeline, features, results_dir, X_train_full, X_test_full, task_type="regression",
                       sample_size=200, plot_type="all"):
    """
    对最终训练好的Pipeline进行SHAP分析，解释模型预测并生成可视化报告

    Parameters:
    -----------
    final_pipeline : sklearn Pipeline
        完整的训练好的Pipeline对象，包含预处理器和模型
    features : list
        用于训练模型的原始特征列表
    results_dir : str
        结果保存目录
    X_train_full : pandas.DataFrame
        包含所有原始特征的训练数据
    X_test_full : pandas.DataFrame
        包含所有原始特征的测试数据
    task_type : str, default="regression"
        任务类型，"regression"或"classification"
    sample_size : int, default=200
        用于SHAP分析的样本数量
    plot_type : str, default="all"
        可视化类型，可选"all", "basic", "advanced"

    Returns:
    --------
    importance_df : pandas.DataFrame
        包含特征重要性信息的DataFrame
    """
    print("创建SHAP分析报告...")

    # 确保导入所需库
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import datetime

    try:
        import shap
    except ImportError:
        print("错误: SHAP库未安装。请使用 'pip install shap' 安装。")
        return None

    # 验证Pipeline结构
    if 'preprocessor' not in final_pipeline.named_steps or 'model' not in final_pipeline.named_steps:
        print("错误：传入的final_pipeline结构不完整，缺少'preprocessor'或'model'步骤。")
        return None

    # 提取Pipeline组件
    model_step = final_pipeline.named_steps['model']
    print(f"使用最终Pipeline中的Model ({type(model_step).__name__}) 进行SHAP分析")

    # 创建结果目录
    shap_dir = os.path.join(results_dir, "shap_analysis")
    os.makedirs(shap_dir, exist_ok=True)

    # 确保只使用选定的特征
    X_train_selected = X_train_full[features].copy()
    X_test_selected = X_test_full[features].copy()

    # 为特征子集创建专用的预处理器
    categorical_num_cols_config = get_config("categorical_num_cols", [])
    categorical_subset = [col for col in features if col in categorical_num_cols_config]
    numerical_subset = [col for col in features if col not in categorical_num_cols_config]
    print(f"为SHAP分析创建针对子集的预处理器:")
    print(f"  - 数值型子集特征: {numerical_subset}")
    print(f"  - 类别数值子集特征: {categorical_subset}")

    # 为特征子集创建专用预处理器
    subset_preprocessor = create_preprocessor(numerical_subset, categorical_subset)

    # 准备SHAP分析的背景数据（用于计算期望值）
    background_size = min(100, len(X_train_selected))
    X_background_orig = X_train_selected.sample(background_size, random_state=get_config("random_state"))

    # 使用子集预处理器转换背景数据
    try:
        # 拟合子集预处理器
        subset_preprocessor.fit(X_train_selected)
        background_data_transformed = subset_preprocessor.transform(X_background_orig)
        print(f"已使用子集预处理器准备 {background_size} 个样本作为背景数据")
    except Exception as e:
        print(f"警告: 使用子集预处理器转换背景数据时出错: {str(e)}。尝试使用原始数据。")
        background_data_transformed = X_background_orig.values

    # 准备SHAP解释样本
    explanation_size = min(sample_size, len(X_test_selected))
    X_sample_orig = X_test_selected.sample(explanation_size, random_state=get_config("random_state"))

    # 使用子集预处理器转换解释样本
    try:
        X_sample_transformed = subset_preprocessor.transform(X_sample_orig)
        print(f"已使用子集预处理器准备 {explanation_size} 个样本用于SHAP分析")
    except Exception as e:
        print(f"警告: 使用子集预处理器转换样本数据时出错: {str(e)}。尝试使用原始数据。")
        X_sample_transformed = X_sample_orig.values

    # 从子集预处理器获取转换后的特征名称
    try:
        transformed_feature_names = subset_preprocessor.get_feature_names_out()
        # 验证转换后的维度是否与特征名称数量匹配
        if X_sample_transformed.shape[1] != len(transformed_feature_names):
            print(
                f"警告：转换后数据的列数 ({X_sample_transformed.shape[1]}) 与子集预处理器获取到的特征名数量 ({len(transformed_feature_names)}) 不匹配！")
            # 创建通用特征名
            transformed_feature_names = [f'feature_{i}' for i in range(X_sample_transformed.shape[1])]
    except Exception as e:
        print(f"警告: 获取子集转换后特征名时出错: {str(e)}。使用通用特征名。")
        transformed_feature_names = [f'feature_{i}' for i in range(X_sample_transformed.shape[1])]

    print(f"子集转换后特征数量: {len(transformed_feature_names)}")

    # 创建解释专用Pipeline
    explainer_pipeline = Pipeline([
        ("preprocessor", subset_preprocessor),
        ("model", model_step)
    ])

    # 选择并创建SHAP解释器
    explainer = None
    shap_values = None

    try:
        # SVC模型特殊处理
        if isinstance(model_step, SVC):
            print("检测到SVC模型，使用专用处理方法...")
            
            # 首先使用预处理器转换数据
            X_train_transformed = subset_preprocessor.transform(X_train_selected)
            X_sample_transformed = subset_preprocessor.transform(X_sample_orig)
            
            # 仅对模型部分使用KernelExplainer
            def svc_predict_proba(X):
                # 对于已转换的数据直接传给模型
                if hasattr(model_step, 'predict_proba'):
                    return model_step.predict_proba(X)[:, 1]
                else:
                    # 如果没有概率输出，使用决策函数
                    return model_step.decision_function(X)
            
            print("为SVC模型创建KernelExplainer")
            # 使用转换后的数据创建解释器
            background_sample = shap.sample(X_train_transformed, min(50, len(X_train_transformed)))
            explainer = shap.KernelExplainer(svc_predict_proba, background_sample)
            
            # 计算SHAP值
            print("计算SVC模型的SHAP值")
            shap_values = explainer.shap_values(X_sample_transformed, nsamples=100)
            
            # 创建DataFrame以显示特征重要性
            importance_df = pd.DataFrame({
                'feature': transformed_feature_names,
                'importance': np.abs(shap_values).mean(0),
                'original_feature': [f.split('__')[0] if '__' in f else f for f in transformed_feature_names]
            })
            
            # 按重要性排序
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # 基于plot_type生成不同类型的可视化
            print(f"生成SVC模型的SHAP可视化图表 (图表类型: {plot_type})")
            
            # 1. 摘要图 (基本的,总是生成)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample_transformed, feature_names=transformed_feature_names, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, "shap_summary.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 柱状图摘要 (基本的)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample_transformed, feature_names=transformed_feature_names, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, "shap_summary_bar.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 如果要求更多图表
            if plot_type in ["all", "advanced"]:
                # 3. 依赖图（对前5个重要特征）
                top_features = importance_df['feature'].head(5).values
                for feature in top_features:
                    try:
                        feature_idx = list(transformed_feature_names).index(feature)
                        plt.figure(figsize=(10, 6))
                        shap.dependence_plot(
                            feature_idx, 
                            shap_values, 
                            X_sample_transformed,
                            feature_names=transformed_feature_names,
                            show=False
                        )
                        plt.tight_layout()
                        safe_feature_name = feature.replace('/', '_').replace('\\', '_')
                        plt.savefig(os.path.join(shap_dir, f"shap_dependence_{safe_feature_name}.png"), dpi=300, bbox_inches='tight')
                        plt.close()
                    except Exception as dep_err:
                        print(f"创建特征'{feature}'的依赖图时出错: {str(dep_err)}")
                
                # 4. 瀑布图(对第一个样本)
                try:
                    if hasattr(shap, "waterfall_plot"):  # 新版SHAP
                        plt.figure(figsize=(12, 8))
                        shap.waterfall_plot(
                            shap.Explanation(
                                values=shap_values[0], 
                                base_values=explainer.expected_value,
                                data=X_sample_transformed[0],
                                feature_names=transformed_feature_names
                            ),
                            show=False
                        )
                        plt.tight_layout()
                        plt.savefig(os.path.join(shap_dir, "shap_waterfall.png"), dpi=300, bbox_inches='tight')
                        plt.close()
                    else:
                        print("此版本的SHAP库不支持waterfall_plot")
                except Exception as wf_err:
                    print(f"创建瀑布图时出错: {str(wf_err)}")
                
                # 5. 决策图(对前3个样本)
                try:
                    if hasattr(shap, "decision_plot"):  # 新版SHAP
                        plt.figure(figsize=(12, 8))
                        shap.decision_plot(
                            explainer.expected_value, 
                            shap_values[:3], 
                            X_sample_transformed[:3],
                            feature_names=transformed_feature_names,
                            show=False
                        )
                        plt.tight_layout()
                        plt.savefig(os.path.join(shap_dir, "shap_decision.png"), dpi=300, bbox_inches='tight')
                        plt.close()
                    else:
                        print("此版本的SHAP库不支持decision_plot")
                except Exception as dec_err:
                    print(f"创建决策图时出错: {str(dec_err)}")
                
                # 6. 力图(对第一个样本)
                try:
                    sample_idx = 0
                    plt.figure(figsize=(20, 3))
                    shap.force_plot(
                        explainer.expected_value, 
                        shap_values[sample_idx], 
                        X_sample_transformed[sample_idx],
                        feature_names=transformed_feature_names,
                        matplotlib=True,
                        show=False
                    )
                    plt.tight_layout()
                    plt.savefig(os.path.join(shap_dir, "shap_force.png"), dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as force_err:
                    print(f"创建力图时出错: {str(force_err)}")
                
                # 7. 保存HTML交互式力图
                try:
                    import IPython
                    from IPython.display import HTML
                    import base64
                    from io import BytesIO
                    
                    # 创建交互式力图
                    force_plot = shap.force_plot(
                        explainer.expected_value, 
                        shap_values[:10], 
                        X_sample_transformed[:10],
                        feature_names=transformed_feature_names
                    )
                    
                    # 保存为HTML
                    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
                    with open(os.path.join(shap_dir, "shap_force_interactive.html"), "w", encoding="utf-8") as f:
                        f.write(shap_html)
                    print("已保存交互式HTML力图")
                except Exception as html_err:
                    print(f"创建HTML交互式图时出错: {str(html_err)}")
            
            # 保存SHAP重要性
            importance_df.to_csv(os.path.join(shap_dir, "shap_importance.csv"), index=False)
            print(f"SHAP分析完成，结果保存在 {shap_dir} 目录")
            print(f"生成的图表: {os.listdir(shap_dir)}")
            
            return importance_df
            
        # 尝试1: 使用自动检测的Explainer
        print("尝试使用shap.Explainer自动检测最佳解释器...")
        try:
            # 创建可分离的Pipeline用于SHAP
            X_train_processed = subset_preprocessor.transform(X_train_selected)
            X_sample_processed = subset_preprocessor.transform(X_sample_orig)
            
            # 针对不同模型类型选择适当的解释器
            if hasattr(model_step, 'predict_proba'):
                explainer = shap.KernelExplainer(
                    model_step.predict_proba, 
                    shap.sample(X_train_processed, 50)
                )
                shap_values = explainer.shap_values(X_sample_processed)[1]  # 取正类概率
            else:
                explainer = shap.KernelExplainer(
                    model_step.predict, 
                    shap.sample(X_train_processed, 50)
                )
                shap_values = explainer.shap_values(X_sample_processed)
                
            print("成功使用KernelExplainer计算SHAP值")
            
        except Exception as e:
            print(f"使用简化模型解释失败: {str(e)}")
            
            try:
                # 使用TreeExplainer作为备选
                print("尝试使用TreeExplainer...")
                tree_based = any(x in type(model_step).__name__.lower() for x in 
                               ['tree', 'forest', 'boost', 'xgb', 'lgbm', 'catboost'])
                
                if tree_based:
                    # 对预处理后的数据使用TreeExplainer
                    X_train_processed = subset_preprocessor.transform(X_train_selected)
                    X_sample_processed = subset_preprocessor.transform(X_sample_orig)
                    
                    explainer = shap.TreeExplainer(model_step)
                    shap_values = explainer.shap_values(X_sample_processed)
                    
                    # 对于XGBoost特殊处理
                    if len(np.array(shap_values).shape) == 3:
                        shap_values = shap_values[1]  # 取正类的SHAP值
                        
                    print("成功使用TreeExplainer计算SHAP值")
                else:
                    raise ValueError("模型不是基于树的模型，无法使用TreeExplainer")
                    
            except Exception as tree_err:
                print(f"TreeExplainer失败: {str(tree_err)}")
                print("所有SHAP方法都失败了，无法生成解释")
                return None

    except Exception as ex_err:
        print(f"创建SHAP解释器时出错: {str(ex_err)}")
        return None

    if shap_values is None:
        print("无法计算SHAP值，分析终止")
        return None

    print("SHAP值计算成功，创建可视化...")

    # 处理分类任务的SHAP值可能是列表的情况
    if task_type == "classification" and isinstance(shap_values, list):
        # 对于二分类，通常取第二个类的SHAP值（通常是正类）
        shap_values_for_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_values_for_plot = shap_values

    # 生成可视化
    plots = []

    # 基本可视化 (Summary Plot和Bar Plot)
    if plot_type in ["all", "basic"]:
        try:
            # 1. Summary Plot (蜂群图)
            plt.figure(figsize=(12, max(8, len(transformed_feature_names) * 0.4)))
            
            # 处理分类问题的SHAP值
            if task_type == "classification":
                # 准备数据用于可视化
                if len(shap_values_for_plot.shape) == 3:
                    # 如果是多维数组，选择第一个类别的SHAP值
                    plot_shap_values = shap_values_for_plot[:, :, 0]
                elif isinstance(shap_values_for_plot, list):
                    # 如果是列表，选择第一个类别
                    plot_shap_values = shap_values_for_plot[0]
                else:
                    # 已经是二维数组
                    plot_shap_values = shap_values_for_plot
            else:
                # 回归问题，直接使用
                plot_shap_values = shap_values_for_plot
            
            # 使用修复后的数据创建Summary Plot
            shap.summary_plot(
                plot_shap_values,
                X_sample_transformed,
                feature_names=transformed_feature_names,
                plot_size=(12, max(8, len(transformed_feature_names) * 0.4)),
                show=False,
                plot_type="dot",
                max_display=20,   # 限制显示的特征数量
                color_bar=True,   # 显示颜色条
                layered_violin_max_num_bins=20  # 优化小提琴图层级
            )
            plt.gcf().set_size_inches(12, max(10, len(transformed_feature_names) * 0.4))
            plt.title('SHAP Summary Plot - Feature Impact', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, "summary_plot.png"), dpi=300, bbox_inches='tight')
            plt.close()
            plots.append("summary_plot.png")

            # 2. Bar Plot (特征重要性)
            plt.figure(figsize=(12, max(8, len(transformed_feature_names) * 0.3)))
            shap.summary_plot(
                plot_shap_values,
                X_sample_transformed,
                feature_names=transformed_feature_names,
                plot_type="bar",
                max_display=20,  # 限制显示的特征数量
                show=False
            )
            plt.gcf().set_size_inches(12, max(10, len(transformed_feature_names) * 0.3))
            plt.title('SHAP Feature Importance', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, "bar_plot.png"), dpi=300, bbox_inches='tight')
            plt.close()
            plots.append("bar_plot.png")

            # 3. Waterfall Plot (单样本解释)
            try:
                # 为第一个样本创建瀑布图
                sample_idx = 0

                # 获取期望值 (base value)
                if hasattr(explainer, 'expected_value'):
                    expected_value = explainer.expected_value
                    if isinstance(expected_value, list):
                        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
                else:
                    expected_value = 0

                # 修复分类问题的SHAP值处理
                if task_type == "classification":
                    if isinstance(shap_values_for_plot, list):
                        # 多分类情况，选择第一个类别
                        sample_shap_values = shap_values_for_plot[0][sample_idx]
                    elif len(shap_values_for_plot.shape) == 3:
                        # 多维数组，获取第一个样本的第一个类别
                        sample_shap_values = shap_values_for_plot[sample_idx, :, 0]
                    else:
                        # 正常二维数组
                        sample_shap_values = shap_values_for_plot[sample_idx]
                else:
                    # 回归问题
                    sample_shap_values = shap_values_for_plot[sample_idx]

                # 创建Explanation对象
                waterfall_explanation = shap.Explanation(
                    values=sample_shap_values,
                    base_values=expected_value,
                    data=X_sample_transformed[sample_idx] if isinstance(X_sample_transformed, np.ndarray) 
                         else X_sample_transformed.iloc[sample_idx].values,
                    feature_names=transformed_feature_names
                )

                plt.figure(figsize=(12, max(8, len(transformed_feature_names) * 0.3)))

                # 根据SHAP版本选择适当的API
                if hasattr(shap.plots, 'waterfall'):
                    shap.plots.waterfall(waterfall_explanation, show=False)
                else:
                    # 兼容旧版本SHAP
                    from shap.plots import waterfall
                    waterfall(
                        expected_value,
                        sample_shap_values,
                        feature_names=transformed_feature_names,
                        show=False
                    )

                plt.title("SHAP Waterfall Plot - Feature Impact for a Single Sample", fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(shap_dir, "shap_waterfall.png"), dpi=300, bbox_inches='tight')
                plt.close()
                plots.append("waterfall_plot.png")
            except Exception as waterfall_err:
                print(f"创建瀑布图时出错: {str(waterfall_err)}")

        except Exception as basic_viz_err:
            print(f"创建基本可视化时出错: {str(basic_viz_err)}")

    # 高级可视化
    if plot_type in ["all", "advanced"]:
        try:
            # 4. 依赖图 (Feature Dependence Plots)
            # 计算特征重要性
            if len(shap_values_for_plot.shape) > 2:  # 多类输出
                importance_values = np.abs(shap_values_for_plot).mean(axis=(0, 2))
            else:  # 单一输出或二维数组
                importance_values = np.abs(shap_values_for_plot).mean(0)
                
            top_indices = np.argsort(-importance_values)[:min(5, len(transformed_feature_names))]

            for idx in top_indices:
                feature_name = transformed_feature_names[idx]
                plt.figure(figsize=(10, 6))
                try:
                    # 准备数据用于内置依赖图
                    if task_type == "classification":
                        # 处理分类问题的SHAP值
                        if len(shap_values_for_plot.shape) == 3:
                            # 如果是多维数组，选择第一个类别的SHAP值
                            plot_shap_values = shap_values_for_plot[:, :, 0]
                        elif isinstance(shap_values_for_plot, list):
                            # 如果是列表，选择第一个类别
                            plot_shap_values = shap_values_for_plot[0]
                        else:
                            # 已经是二维数组
                            plot_shap_values = shap_values_for_plot
                    else:
                        # 回归问题，直接使用
                        plot_shap_values = shap_values_for_plot
                    
                    # 使用SHAP内置依赖图
                    try:
                        # 尝试使用新版API
                        if hasattr(shap.plots, "dependence"):
                            sample_data = X_sample_transformed
                            shap.plots.dependence(
                                plot_shap_values[:, idx],
                                sample_data[:, idx],
                                feature_names=[feature_name],
                                show=False
                            )
                        else:
                            # 回退到旧版API
                            shap.dependence_plot(
                                idx,
                                plot_shap_values,
                                X_sample_transformed,
                                feature_names=transformed_feature_names,
                                show=False
                            )
                        plt.title(f'SHAP Dependence Plot - {feature_name}', fontsize=14)
                        plt.tight_layout()
                        plt.savefig(os.path.join(shap_dir, f"dependence_plot_{feature_name.replace('/', '_')}.png"),
                                    dpi=300, bbox_inches='tight')
                        plt.close()
                        plots.append(f"dependence_plot_{feature_name}.png")
                    except Exception as native_err:
                        print(f"使用内置依赖图失败: {str(native_err)}，切换到自定义图")
                        
                        # 回退到自定义依赖图
                        plt.figure(figsize=(10, 6))
                        if task_type == "classification" and len(shap_values_for_plot.shape) > 2:
                            # 处理多类输出，选择第一个类
                            dep_values = shap_values_for_plot[:, idx, 0]
                        else:
                            # 单一输出
                            dep_values = shap_values_for_plot[:, idx]
                            
                        # 创建自定义依赖图
                        plt.scatter(X_sample_transformed[:, idx], dep_values, s=20, alpha=0.6)
                        plt.xlabel(feature_name)
                        plt.ylabel(f'SHAP value for {feature_name}')
                        plt.title(f'SHAP Dependence Plot - {feature_name} (Custom)', fontsize=14)
                        plt.grid(alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(shap_dir, f"dependence_plot_{feature_name.replace('/', '_')}.png"),
                                    dpi=300, bbox_inches='tight')
                        plt.close()
                        plots.append(f"dependence_plot_{feature_name}.png")
                except Exception as dep_err:
                    print(f"创建特征 {feature_name} 的依赖图时出错: {str(dep_err)}")

            # 5. 特征交互图 (最重要的两个特征)
            if len(top_indices) >= 2:
                try:
                    # 准备数据用于内置交互图
                    if task_type == "classification":
                        # 处理分类问题的SHAP值
                        if len(shap_values_for_plot.shape) == 3:
                            # 如果是多维数组，选择第一个类别的SHAP值
                            plot_shap_values = shap_values_for_plot[:, :, 0]
                        elif isinstance(shap_values_for_plot, list):
                            # 如果是列表，选择第一个类别
                            plot_shap_values = shap_values_for_plot[0]
                        else:
                            # 已经是二维数组
                            plot_shap_values = shap_values_for_plot
                    else:
                        # 回归问题，直接使用
                        plot_shap_values = shap_values_for_plot
                    
                    # 使用SHAP内置交互图
                    try:
                        plt.figure(figsize=(10, 8))
                        shap.dependence_plot(
                            top_indices[0],
                            plot_shap_values,
                            X_sample_transformed,
                            feature_names=transformed_feature_names,
                            interaction_index=top_indices[1],
                            show=False
                        )
                        plt.title(
                            f'SHAP Interaction Plot - {transformed_feature_names[top_indices[0]]} & {transformed_feature_names[top_indices[1]]}',
                            fontsize=14)
                        plt.tight_layout()
                        plt.savefig(os.path.join(shap_dir, "interaction_plot.png"), dpi=300, bbox_inches='tight')
                        plt.close()
                        plots.append("interaction_plot.png")
                    except Exception as native_int_err:
                        print(f"使用内置交互图失败: {str(native_int_err)}，切换到自定义图")
                        
                        # 回退到自定义交互图
                        plt.figure(figsize=(10, 8))
                        if task_type == "classification" and len(shap_values_for_plot.shape) > 2:
                            # 处理多类输出，选择第一个类
                            values_idx0 = shap_values_for_plot[:, top_indices[0], 0]
                            values_idx1 = shap_values_for_plot[:, top_indices[1], 0]
                        else:
                            # 单一输出
                            values_idx0 = shap_values_for_plot[:, top_indices[0]]
                            values_idx1 = shap_values_for_plot[:, top_indices[1]]
                            
                        # 创建交互散点图
                        plt.scatter(values_idx0, values_idx1, s=30, alpha=0.7)
                        plt.xlabel(f'SHAP value for {transformed_feature_names[top_indices[0]]}')
                        plt.ylabel(f'SHAP value for {transformed_feature_names[top_indices[1]]}')
                        plt.title(f'SHAP Interaction Plot - Top Two Features (Custom)', fontsize=14)
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(shap_dir, "interaction_plot.png"), dpi=300, bbox_inches='tight')
                        plt.close()
                        plots.append("interaction_plot.png")
                except Exception as int_err:
                    print(f"创建特征交互图时出错: {str(int_err)}")

        except Exception as adv_viz_err:
            print(f"创建高级可视化时出错: {str(adv_viz_err)}")

    # 创建特征重要性表格
    try:
        # 计算特征重要性，适应多种SHAP值格式
        if len(shap_values_for_plot.shape) > 2:  # 多类输出
            importance_values = np.abs(shap_values_for_plot).mean(axis=(0, 2))
        else:  # 单一输出或二维数组
            importance_values = np.abs(shap_values_for_plot).mean(0)
            
        importance_df = pd.DataFrame({
            'Feature': transformed_feature_names,
            'Importance': importance_values,
            'Rank': np.argsort(np.argsort(-importance_values)) + 1
        })

        # 保存特征重要性到CSV
        importance_df.to_csv(os.path.join(shap_dir, "shap_feature_importance.csv"), index=False)

        # 创建综合报告文本文件
        with open(os.path.join(shap_dir, "shap_analysis_report.txt"), "w") as f:
            f.write("SHAP Feature Importance Analysis\n")
            f.write("==============================\n\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Type: {type(model_step).__name__}\n")
            f.write(f"Number of Features: {len(transformed_feature_names)}\n\n")

            # 按重要性排序写入特征
            f.write("Features by Importance:\n")
            for idx, row in importance_df.sort_values('Importance', ascending=False).iterrows():
                f.write(f"{int(row['Rank'])}. {row['Feature']}: {row['Importance']:.6f}\n")

            f.write("\nGenerated Visualizations:\n")
            for plot in plots:
                f.write(f"- {plot}\n")

        print(f"SHAP分析完成，结果保存在: {shap_dir}")
        return importance_df

    except Exception as report_err:
        print(f"创建SHAP报告时出错: {str(report_err)}")
        return None


def robust_cross_validation(X, y, model_pipeline, cv=None, task_type="regression", stratify=None, groups=None,
                            n_jobs=-1):
    """
    执行严格的交叉验证，防止数据泄露

    参数:
    X: 特征数据
    y: 目标变量
    model_pipeline: 模型管道
    cv: 交叉验证折数或交叉验证对象
    task_type: 任务类型('regression'或'classification')
    stratify: 用于分层的数组（仅分类问题）
    groups: 用于GroupKFold的分组数组
    n_jobs: 并行作业数

    返回:
    包含交叉验证结果的字典
    """
    if cv is None:
        cv = get_config("cv", 5)

    # 确定使用哪种交叉验证策略
    if isinstance(cv, int):
        if groups is not None:
            # 使用分组交叉验证以尊重数据分组
            cv_obj = GroupKFold(n_splits=cv)
            cv_args = {"groups": groups}
            print(f"Using GroupKFold cross-validation with {cv} splits")
        elif task_type == "classification" and stratify is not None:
            # 使用分层交叉验证来处理不平衡的分类问题
            cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=get_config("random_state"))
            cv_args = {"y": y}
            print(f"Using StratifiedKFold cross-validation with {cv} splits")
        else:
            # 使用标准K折交叉验证
            cv_obj = KFold(n_splits=cv, shuffle=True, random_state=get_config("random_state"))
            cv_args = {}
            print(f"Using KFold cross-validation with {cv} splits")
    else:
        # 用户提供了自定义的交叉验证对象
        cv_obj = cv
        cv_args = {}
        print(f"Using custom cross-validation: {cv_obj.__class__.__name__}")

    # 定义评估指标
    if task_type == "regression":
        scoring = {
            'neg_rmse': 'neg_root_mean_squared_error',
            'r2': 'r2'
        }
    else:
        if len(np.unique(y)) > 2:  # 多分类
            scoring = {
                'accuracy': 'accuracy',
                'f1_weighted': 'f1_weighted',
                'precision_weighted': 'precision_weighted',
                'recall_weighted': 'recall_weighted'
            }
        else:  # 二分类
            scoring = {
                'accuracy': 'accuracy',
                'f1': 'f1',
                'precision': 'precision',
                'recall': 'recall',
                'roc_auc': 'roc_auc'
            }

    # 执行交叉验证，确保在每个折内完成完整的预处理，防止数据泄露
    scores = cross_validate(
        model_pipeline,
        X, y,
        cv=cv_obj,
        scoring=scoring,
        return_estimator=True,
        n_jobs=n_jobs,
        **cv_args
    )

    # 计算平均分数和标准差
    results = {}
    for metric in scoring.keys():
        score_key = f'test_{metric}'
        results[metric] = {
            'mean': scores[score_key].mean(),
            'std': scores[score_key].std(),
            'values': scores[score_key].tolist()
        }

    # 返回交叉验证的估计器，以便后续使用
    results['estimators'] = scores['estimator']

    return results


if __name__ == "__main__":
    main()
