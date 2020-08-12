# -*- coding: utf-8 -*-
"""
Created on 2020/8/11 10:10 上午

@Project -> File: nonlinear-correlation-analysis -> binary_mie.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 二分箱的互信息熵计算
"""

import logging

logging.basicConfig(level = logging.INFO)

from lake.decorator import time_cost
import pandas as pd
import numpy as np
import copy
import sys, os

sys.path.append('../..')

from src.data_binning.univariate_binning import UnivarBinning


# ---- 概率和互信息熵计算 ---------------------------------------------------------------------------

def _cal_confusion_matrix(binned_arr):
	"""仅支持二分类的混淆矩阵计算"""
	c_matrix = np.zeros([2, 2])
	for i in range(2):
		for j in range(2):
			c_matrix[i, j] = binned_arr[(binned_arr[:, 0] == i) & (binned_arr[:, 1] == j)].shape[0]
	return c_matrix


def _do_binary_cut(arr: np.ndarray, label_pair: list, var_types: list):
	"""根据选定的分割标识进行二分"""
	_arr = np.zeros_like(arr)
	for i in range(2):
		if var_types[i] == 'continuous':
			_arr[np.where(arr[:, i] >= label_pair[i]), i] = 1.0
		else:
			_arr[np.where(arr[:, i] == label_pair[i]), i] = 1.0
	
	c_matrix = _cal_confusion_matrix(_arr)
	proba_matrix = c_matrix / np.sum(c_matrix)
	
	return c_matrix, proba_matrix


def _cal_grid_entropy(P):
	if P == 0:
		return 0.0
	else:
		return -P * np.log2(P)


def _cal_mie_from_proba_matrix(proba_matrix) -> float:
	H_x = _cal_grid_entropy(np.sum(proba_matrix[0, :])) + _cal_grid_entropy(
		np.sum(proba_matrix[1, :]))
	H_y = _cal_grid_entropy(np.sum(proba_matrix[:, 0])) + _cal_grid_entropy(
		np.sum(proba_matrix[:, 1]))
	H_xy = _cal_grid_entropy(proba_matrix[0, 0]) + _cal_grid_entropy(
		proba_matrix[0, 1]) + _cal_grid_entropy(proba_matrix[1, 0]) + \
	       _cal_grid_entropy(proba_matrix[1, 1])
	
	mie = H_x + H_y - H_xy
	return mie


def cal_mie(arr: np.ndarray, label_pair: list, var_types: list) -> float:
	"""
	根据设定的切分label对已有数据集进行二分分割, 并计算此时对应的互信息熵
	
	:param arr: 待切分数据集, shape = (N, D = 2), 第一列为x, 第二列为y
	:param label_pair: 设定切分位置的标识对
	:param var_types: 各维度变量值类型
	"""
	c_matrix, proba_matrix = _do_binary_cut(arr, label_pair, var_types)
	mie = _cal_mie_from_proba_matrix(proba_matrix)
	return mie


# ---- 时滞序列生成 ---------------------------------------------------------------------------------

def _gen_time_delayed_series(arr: np.ndarray, td_lag: int):
	"""
	生成时间延迟序列

	:param arr: 样本数组, shape = (N, D = 2), 第一列为x, 第二列为y
	:param td_lag: 时间平移样本点数, 若td_lag > 0, 则x对应右方td_lag个样本点后的y;
				   若td_lag < 0, 则y对应右方td_lag个样本点后的x
	"""
	lag_remain = np.abs(td_lag) % arr.shape[0]  # 整除后的余数
	x_td = copy.deepcopy(arr[:, 0])
	y_td = copy.deepcopy(arr[:, 1])
	
	if lag_remain == 0:
		pass
	else:
		if td_lag > 0:
			y_td = np.hstack((y_td[lag_remain:], y_td[:lag_remain]))
		else:
			x_td = np.hstack((x_td[lag_remain:], x_td[:lag_remain]))
	
	return x_td, y_td


# ---- 二分互信息熵 ---------------------------------------------------------------------------------

class BinaryMutualInfoEntropy(object):
	"""使用二分分箱计算互信息熵值"""
	
	def __init__(self, var_values: list, var_types: list):
		try:
			assert len(var_values) == 2
			assert len(var_types) == 2
		except Exception as e:
			raise ValueError('Input variable num is not 2, {}'.format(e))
		
		self.var_values = var_values
		self.var_types = var_types
		self.arr = np.array(self.var_values).T
		self.N, self.D = self.arr.shape
	
	def deter_candidate_binary_cut_locs(self, bins: int = 10):
		"""
		确定候选分箱二分方案
		
		:param bins: 连续值分箱的个数; 离散值分箱使用label_binning方法, 与该参数无关
		"""
		self.labels_grid = []
		for i in range(2):
			binning_ = UnivarBinning(self.var_values[i], self.var_types[i])
			if self.var_types[i] == 'continuous':
				_, labels = binning_.do_univar_binning(method = 'isometric', bins = bins)
			else:
				_, labels = binning_.do_univar_binning(method = 'label')
			self.labels_grid.append(labels)
	
	def do_grid_search_mie(self, arr: np.ndarray) -> np.ndarray:
		"""网格化搜索"""
		_labels_pairs = [[p, q] for p in self.labels_grid[0] for q in self.labels_grid[1]]
		
		grid_search_mie_results = pd.DataFrame(_labels_pairs)
		grid_search_mie_results['mie'] = grid_search_mie_results.apply(
			lambda x: cal_mie(arr, x, self.var_types),
			axis = 1
		)
		
		# 转为数组.
		grid_search_mie_arr = grid_search_mie_results['mie'].values.reshape(
			len(self.labels_grid[0]), len(self.labels_grid[1]))
		return grid_search_mie_arr
	
	def cal_mie_opt(self, arr: np.ndarray) -> float:
		"""
		计算两个变量间的最大MIE值
		
		:param arr: x和y序列构成的arr数组, shape = (N, D = 2)
		"""
		grid_search_mie_arr = self.do_grid_search_mie(arr)
		
		# 对于多分类变量, 信息熵应该是各个互斥子二分类互信息熵熵值之和.
		# 如果对应维度上变量为离散类别值, 则该维度上的MIE值需要加和合并.
		# 依据来自信息熵和互信息熵的链式法则:
		# https://www.cs.uic.edu/pub/ECE534/WebHome/ch2.pdf
		_axis2sum = []
		for i in range(len(self.var_types)):
			_axis2sum.append(True if self.var_types[i] == 'discrete' else False)
		
		for i in range(self.D):
			if _axis2sum[i]:
				_reshape_lst = [-1] * self.D
				_reshape_lst[i] = 1
				grid_search_mie_arr = np.sum(grid_search_mie_arr, axis = i).reshape(
					tuple(_reshape_lst))
		
		mie_opt = np.max(grid_search_mie_arr)
		return mie_opt
	
	@time_cost
	def time_delay_mie(self, td_lags: list or np.ndarray) -> dict:
		"""
		含时滞的互信息熵检测
		
		:param td_lags: 待计算的时滞lag序列, 当lag > 0计算x对y的正向时间影响; 当lag
						< 0计算y对x的正向时间影响
		"""
		td_mie = {}
		for i in range(len(td_lags)):
			print('cal time delay mie: {}'.format(str(int(i / len(td_lags) * 100)) + '%') + "\r",
			      end = "")
			_td_lag = td_lags[i]
			_x_td, _y_td = _gen_time_delayed_series(self.arr, _td_lag)
			_td_arr = np.vstack((_x_td, _y_td)).T
			
			mie_opt = self.cal_mie_opt(_td_arr)
			td_mie[int(_td_lag)] = float(mie_opt)
		return td_mie
			

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
	
	# ---- 计算时滞互信息熵 -------------------------------------------------------------------------
	
	td_lags = np.arange(-800, 800 + 1, 1)
	
	for y_col in y_cols:
		print('Analyzing {}'.format(y_col))
		plt.figure(figsize = [12, 8])
		plt.suptitle('Time Delay MIE Analysis for {}'.format(y_col))
		td_mie_results = defaultdict(dict)
		for x_col in x_cols:
			print('x_col: {}'.format(x_col))
			var_cols = [x_col, y_col]
			var_values, var_types = [], []
			for i in range(2):
				var_values.append(list(data[var_cols[i]]))
				_var_type = 'continuous' if var_cols[i] in continuous_cols else 'discrete'
				var_types.append(_var_type)

			# ---- 进行序列平移和互信息熵计算 --------------------------------------------------------
			
			self = BinaryMutualInfoEntropy(var_values, var_types)
			self.deter_candidate_binary_cut_locs(bins = 5)
			td_mie = self.time_delay_mie(td_lags)
			td_mie_results[x_col][y_col] = td_mie

			plt.subplot(len(x_cols) // 3 + 1, 3, x_cols.index(x_col) + 1)
			plt.plot(list(td_mie.keys()), list(td_mie.values()), linewidth = 0.8)
			plt.xlim(td_lags[0], td_lags[-1])
			if max(td_mie.values()) < 0.1 :
				plt.ylim([0.0 - 0.025, 0.1 + 0.05])
			elif (max(td_mie.values()) < 0.3) & (max(td_mie.values()) >= 0.1) :
				plt.ylim([0.0 - 0.05, 0.3 + 0.05])
			else:
				plt.ylim([0.0 - 0.1, 0.65 + 0.05])
			plt.legend([x_col], fontsize = 8.0, loc = 'upper right')
			plt.axvline(c = proj_cmap['grey'], linewidth = 0.3)
			plt.tight_layout()
			plt.subplots_adjust(top = 0.94)
			plt.xticks(fontsize = 8.0)
			plt.yticks(fontsize = 8.0)
			plt.show()
			plt.pause(0.5)
		plt.savefig(os.path.join(proj_dir, 'img/weather/y_col_{}.png'.format(y_col)), dpi = 450)
		plt.close()
		
		# 保存结果.
		with open(os.path.join(proj_dir, 'file/weather_td_mie_results/{}_td_mie.json'.format(y_col)), 'w') as f:
			json.dump(td_mie_results, f)
	
