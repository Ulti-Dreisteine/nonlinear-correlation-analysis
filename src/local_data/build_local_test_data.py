# -*- coding: utf-8 -*-
"""
Created on 2020/8/10 5:55 下午

@Project -> File: nonlinear-correlation-analysis -> build_local_test_data.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 测试用函数
"""

import pandas as pd
import sys, os

sys.path.append('../..')

from src import proj_dir


def load_local_test_data(label: str) -> pd.DataFrame:
	"""
	载入本地测试数据
	
	:param label: 数据标签, {'weather', 'patient}
	:return: data: 测试数据表
	"""
	fp = os.path.join(proj_dir, 'data/raw/{}/data.csv'.format(label))
	return pd.read_csv(fp)

