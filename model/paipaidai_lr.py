#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: paipaidai_logisti_regression.py
@time: 19-7-3 下午2:40
@desc: 逻辑回归进行多分类
"""
from load_data import load_2
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

train_values, test_values,clf_labels,amt_labels,train_due_amt_df,sub=load_2()
# print(train_values.isnull().sum())

train_values=train_values.fillna(0)
test_values=test_values.fillna(0)


X_train = train_values.values
X_test = test_values.values

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


lr = LogisticRegression(C=1000.0, random_state=0)

lr.fit(X_train_std, clf_labels)


test_pred_prob = np.zeros((test_values.shape[0], 33))
test_pred_prob = lr.predict_proba(X_test_std)

prob_cols = ['prob_{}'.format(i) for i in range(33)]  # prob_0 至 prob_32

for i, f in enumerate(prob_cols):  # 遍历每一个prob_i

    sub[f] = test_pred_prob[:, i]

sub_example = pd.read_csv('../data/submission.csv', parse_dates=['repay_date'])
test = pd.read_csv('../data/test.csv', parse_dates=['due_date'])

due_date = pd.DataFrame()
due_date['listing_id'] = test['listing_id']
due_date['due_date'] = test['due_date']

sub_example = sub_example.merge(sub, on='listing_id', how='left')
sub_example = sub_example.merge(due_date, on='listing_id', how='left')

sub_example['days'] = (sub_example['due_date'] - sub_example['repay_date']).dt.days

# shape = (-1, 33)

test_prob = sub_example[prob_cols].values

test_labels = sub_example['days'].values

# 第i个样本第test_labels[i]天的还款概率
test_prob = [test_prob[i][test_labels[i]] for i in range(test_prob.shape[0])]

# 第i个样本第test_labels[i]天的预测的还款金额
sub_example['repay_amt'] = sub_example['due_amt'] * test_prob

sub_example[['listing_id', 'repay_date', 'repay_amt']].to_csv('../data/submission_lr.csv', index=False)


