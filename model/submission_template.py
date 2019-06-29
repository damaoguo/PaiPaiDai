#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: submission_template.py
@time: 19-6-29 下午4:08
@desc: 生成数据提交模板
"""
import pandas as pd
import numpy as np

test_df = pd.read_csv('../data/test.csv', parse_dates=['auditing_date', 'due_date'])
