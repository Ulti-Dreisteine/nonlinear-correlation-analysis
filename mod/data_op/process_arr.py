# -*- coding: utf-8 -*-
"""
Created on 2020/8/10 5:46 下午

@Project -> File: nonlinear-correlation-analysis -> process_arr.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 处理arr数组
"""

from collections import defaultdict
import pandas as pd
import numpy as np
import sys

sys.path.append('../')

from mod.data_op import EPS


def _convert_series_label2number(x: list or np.ndarray):
	"""将离散值序列中的类别转为数值"""
	x = np.array(x).flatten()
	_map = defaultdict(int)
	_labels = list(np.unique(x))
	for i in range(len(_labels)):
		_label = _labels[i]
		_map[_label] = i
	
	for k, v in _map.items():
		x[x == k] = int(v)
	
	x = np.array(x, dtype = np.float16)
	return x


def convert_series_values(x: list or np.ndarray, x_type: str):
	"""单序列值转换, 连续值转为np.float64, 离散值转换为np.float16"""
	_d_types = {'continuous': np.float64, 'discrete': np.float16}
	try:
		x = np.array(x, dtype = _d_types[x_type]).flatten()
		return x
	except:
		if x_type == 'discrete':
			try:
				x = _convert_series_label2number(x)
				return x
			except:
				raise RuntimeError(
					'Cannot convert x into numpy.ndarray with numerical values np.float64 or '
					'np.float16'
				)
		else:
			raise RuntimeError(
				'Cannot convert x into numpy.ndarray with numerical values np.float64 or np.float16'
			)


def drop_nans_and_add_noise(arr: np.ndarray, var_types: list):
	"""在数组中去掉异常值, 并对连续值加入噪声"""
	_D = arr.shape[1]
	_df = pd.DataFrame(arr).dropna(axis = 0, how = 'any')
	arr = np.array(_df)
	
	for i in range(_D):
		if var_types[i] == 'continuous':
			arr[:, i] += EPS * np.random.random(arr.shape[0])
	return arr
