#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: heavy_user_test.py
@time: 19-6-30 下午7:43
@desc: 训练集和测试集中贷款次数大于N?
"""

import pandas as pd
import numpy as np

################################################################
# 标信息,找出全部的重度用户
################################################################
user_repay_logs=pd.read_csv("../data/user_repay_logs.csv")
print("user_repay_logs.shape:",user_repay_logs.shape)


# 删除重复的标
user_repay_logs_= user_repay_logs.drop_duplicates('listing_id')
print("user_repay_logs_.shape",user_repay_logs_.shape)

# 统计用户的标的数目
user_repay_logs_count=user_repay_logs_.groupby(['user_id'])['user_id'].count()
print("user_repay_logs_count",user_repay_logs_count.shape)



print(user_repay_logs_count.head(1000))

# heavy_user=set([])


################################################################
# 找到train和test数据
################################################################
# train=pd.read_csv("../data/train.csv")
# test=pd.read_csv("../data/test.csv")
#
#
#
# train_user_id=train['user_id'].tolist()
# test_user_id=test['user_id'].tolist()
#
#
#
#
# heavy_train_user=set(train_user_id)&heavy_user
