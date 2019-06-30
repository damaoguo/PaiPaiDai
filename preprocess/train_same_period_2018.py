#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: train_same_period_2018.py
@time: 19-6-30 下午6:58
@desc: 和test数据集处在同一个时间段的2018年的标记录,train总共一百万条数据,同一个时期的数据14万条
使用小表驱动大表
"""
import pandas as pd
import numpy as np

train=pd.read_csv('../data/train.csv')
train_same_period_2018=train[train['auditing_date']<='2018-03-31']
train_same_period_2018=train_same_period_2018[train_same_period_2018['auditing_date']>='2018-02-01']
print('train_same_period_2018 shape:',train_same_period_2018.shape)
train_same_period_2018.to_csv("../data/train_same_period_2018.csv",index=False)