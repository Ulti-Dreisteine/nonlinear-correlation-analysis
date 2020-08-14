# -*- coding: utf-8 -*-
"""
Created on 2020/8/12 11:15 上午

@Project -> File: nonlinear-correlation-analysis -> cart_tree_mie.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 使用CART决策树计算相关性
"""

import logging

logging.basicConfig(level = logging.INFO)

import matplotlib.pyplot as plt
# from sklearn.tree import DecisionTreeRegressor
# from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
from lightgbm import plot_importance, create_tree_digraph, plot_tree
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor as LGBM
import numpy as np
import sys, os

sys.path.append('../..')

if __name__ == '__main__':
	
	# ---- 载入测试数据和参数 -----------------------------------------------------------------------
	
	from collections import defaultdict
	import matplotlib.pyplot as plt
	import json
	
	from src import proj_dir, proj_cmap
	from src.local_data.build_local_test_data import load_local_test_data
	
	data = load_local_test_data(label = 'weather')
	
	continuous_cols = ['pm25', 'pm10', 'so2', 'co', 'no2', 'o3', 'aqi', 'ws', 'temp', 'sd']
	discrete_cols = ['weather', 'wd', 'month', 'weekday', 'clock_num']
	
	x_cols = ['pm25', 'pm10', 'so2', 'co', 'no2', 'o3', 'aqi', 'ws', 'temp', 'sd', 'weather', 'wd',
	          'month', 'weekday', 'clock_num']
	y_cols = ['pm25', 'pm10', 'so2', 'co', 'no2', 'o3']
	
	# ---- 测试 ------------------------------------------------------------------------------------
	
	x_col, y_col = 'pm25', 'wd'
	x_type, y_type = 'continuous', 'discrete'
	X, y = data[x_col].values.reshape(-1, 1), data[y_col].values
	
	# y要做归一化.
	y = (y - np.min(y)) / (np.max(y) - np.min(y))
	
	# lightgbm包中的模型.
	params = {
		'task': 'train',
		'max_depth': 1,
		# 'boosting_type': 'gbdt',  # 设置提升类型
		'min_data_in_leaf': 20,
		'objective': 'binary',  # 'regression_l1' if y_type == 'continuous' else 'cross_entropy',  # 目标函数
		'num_leaves': 31,  # 叶子节点数, 剪枝操作相关
		'learning_rate': 1.0,  # 学习速率
		'verbose': 0,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
	}

	train_data = lgb.Dataset(
		data = X,
		label = y,
		feature_name = [x_col],
		categorical_feature = [x_col] if x_type == 'discrete' else None
	)

	rgsr = lgb.train(
		params,
		train_data,
		num_boost_round = 1,  # 只进行一次分割
	)
	
	y_pred = rgsr.predict(data = X)
	
	plt.plot(y[:3000])
	plt.plot(y_pred[:3000])
	
	
	rgsr_info = rgsr.dump_model(num_iteration = -1)  # 转为json格式, 提取树节点和分裂特征
	
	if x_type == 'discrete':
		thres = rgsr_info['tree_info'][0]['tree_structure']['threshold'].split(sep = '||')
	else:
		thres = rgsr_info['tree_info'][0]['tree_structure']['threshold']

	
	
