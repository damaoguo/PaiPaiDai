#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: load_data.py
@time: 19-7-1 上午9:01
@desc:
"""
import pandas as pd
def load():
    train_values = pd.read_csv("../data/train_values.csv")
    test_values = pd.read_csv("../data/test_values.csv")

    clf_labels = train_values['label_32'].values  # 标签一列，为0-32的数
    clf_labels_r = train_values['label_32_r'].values  # 标签一列，为0-32的数
    clf_labels_2 = train_values['label_2'].values  # 标签一列，为0-32的数

    del train_values['user_id'], train_values['listing_id'], train_values['label_32'], train_values['label_32_r'],train_values['label_2']
    del test_values['user_id'], test_values['listing_id']

    return train_values,test_values,clf_labels,clf_labels_r,clf_labels_2
