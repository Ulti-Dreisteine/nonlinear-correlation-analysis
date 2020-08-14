# -*- coding: utf-8 -*-
"""
Created on 2020/8/13 4:32 下午

@Project -> File: nonlinear-correlation-analysis -> binary_tree_mie.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 二叉树互信息熵
"""

import lightgbm as lgb
import numpy as np
import sys, os

sys.path.append('../..')


class BinaryMutualInfoEntropy(object):
	"""二叉分箱计算互信息熵"""
	
	def __init__(self, var_values: list, var_types: list):
		try:
			assert len(var_values) == 2
			assert len(var_types) == 2
		except Exception as e:
			raise ValueError('Input variable num is not 2, {}'.format(e))
		
		self.var_values = var_values
		self.var_types = var_types
		self.arr = np.array(self.var_values).T
		self._normalize_y()
		self.N, self.D = self.arr.shape
		
		# 数据归一化
		
	def _normalize_y(self):
		if self.var_types[1] == 'discrete':
			min_, max_ = np.max(self.arr[:, i]), np.min(self.arr[:, i])
			self.arr[:, i] = np.round((self.arr[:, i] - min_) / (max_ - min_), 6)
			
	
		
		
	
	

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

	# ---- 计算互信息熵 -----------------------------------------------------------------------------
	
	x_col, y_col = 'weather', 'pm25'
	
	var_cols = [x_col, y_col]
	var_values, var_types = [], []
	for i in range(2):
		var_values.append(list(data[var_cols[i]]))
		_var_type = 'continuous' if var_cols[i] in continuous_cols else 'discrete'
		var_types.append(_var_type)
		
	# ---- 类测试 -----------------------------------------------------------------------------------
	
	self = BinaryMutualInfoEntropy(var_values, var_types)
	
	# 临时.
	lst = self.var_values[0]
