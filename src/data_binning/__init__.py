# -*- coding: utf-8 -*-
"""
Created on 2020/8/10 5:43 下午

@Project -> File: nonlinear-correlation-analysis -> __init__.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 初始化配置
"""

VALUE_TYPES_AVAILABLE = ['continuous', 'discrete']

METHODS_AVAILABLE = {
	'continuous': ['isometric', 'equifreq', 'quasi_chi2'],
	'discrete': ['label']
}

EPS = 1e-12

__all__ = ['VALUE_TYPES_AVAILABLE', 'METHODS_AVAILABLE', 'EPS']