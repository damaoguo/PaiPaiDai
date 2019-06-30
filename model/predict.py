#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: predict.py
@time: 19-6-29 下午5:04
@desc: 加载保存的模型,存储结果
"""

import time
from sklearn.externals import joblib
import numpy as np
import pandas as pd

test_values=pd.read_csv("../data/test_values.csv")

del test_values['user_id'], test_values['listing_id']


test_pred_prob = np.zeros((test_values.shape[0], 33))#测试集个数作为行数，33为列数，测试集每天的还款概率

for i in range(5):#特征数据，标签

    print(i, 'fold...')

    t = time.time()

    clf=joblib.load('../data/paipaidai_v2_%d.pkl'%i)

    test_pred_prob += clf.predict_proba(test_values.values, num_iteration=clf.best_iteration_) / 5#每个测试集样本的33个类别概率

y_pred=[list(x).index(max(x)) for x in test_pred_prob]


test_df = pd.read_csv('../data/test.csv', parse_dates=['auditing_date', 'due_date'])
sub = test_df['listing_id']


prob_cols = ['prob_{}'.format(i) for i in range(33)]#prob_0 至 prob_32

for i, f in enumerate(prob_cols):
    sub[f] = test_pred_prob[:, i]

sub['y']=y_pred


sub_example = pd.read_csv('../data/submission_template.csv')
sub_example = sub_example.merge(sub, on='listing_id', how='left')


test_prob = sub_example[prob_cols].values

test_labels = sub_example['days'].values


test_prob = [test_prob[i][test_labels[i]] for i in range(test_prob.shape[0])]

sub_example['repay_amt'] = sub_example['due_amt'] * test_prob

# 清除没有还钱的行

drop_left=sub_example[sub_example['y']==32].drop_duplicates(subset='listing_id', keep='first')
sub_example.drop(sub_example[sub_example['y']==32].index)

drop_left['repay_date']=None
drop_left['repay_amt']=None

sub_example=pd.concat([sub_example,drop_left],axis=0)

sub_example[['listing_id', 'repay_date', 'repay_amt']].to_csv('../data/paipaidai_04.csv', index=False)