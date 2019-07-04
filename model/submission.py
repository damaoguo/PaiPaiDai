#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: submission.py
@time: 19-7-4 上午8:47
@desc:
"""

import time
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from load_data import load_2
import figures
import xgboost as xgb


train_values, test_values,clf_labels,amt_labels,train_due_amt_df,sub=load_2()

test=pd.read_csv("../data/test.csv")
listing_id=test['listing_id']
repay_amt=test['due_amt']

# test_pred_prob = np.zeros((test_values.shape[0], 33))#测试集个数作为行数，33为列数，测试集每天的还款概率
#
#
# for i in range(5):#特征数据，标签
#
#     print(i, 'fold...')
#
#     t = time.time()
#     clf=joblib.load('../data/paipaidai_v3_%d.pkl'%i)
#
#     test_pred_prob += clf.predict_proba(test_values.values, num_iteration=clf.best_iteration_) / 5#每个测试集样本的33个类别概率


test_pred_prob = np.zeros((test_values.shape[0], 33))

# 测试集数据
xgb_test=xgb.DMatrix(test_values.values)
for index in range(0,5):
    print(index," fold...")
    bst=xgb.Booster({'nthread':4})
    bst.load_model('../data/xgb%s.model'%str(index))
    # print(bst.predict(xgb_test))
    test_pred_prob += bst.predict(xgb_test)/5



y_pred=[list(x).index(max(x)) for x in test_pred_prob]



#############################################################################
# 具体到固定的天数
#############################################################################
# 新建提交数据
out=pd.DataFrame()

# (1)提交列
out['listing_id']=listing_id
out['y']=y_pred

# 计算还款日期
due_date=test['due_date']
out['due_date']=pd.to_datetime(due_date)

# (2)提交列
out['repay_date']=out['due_date']-pd.to_timedelta(out['y'],unit='d')

# (3)提交列
out['repay_amt']=test_values['due_amt']

# 将逾期的置空
out.loc[out['y']==32,'repay_date']=None
out.loc[out['y']==32,'repay_amt']=None

# 删除多余的列
del out['y'],out['due_date']
out.to_csv("../data/paipaidai_specific.csv",index=False)



########################################################################
# 分配到33天中
########################################################################

test_df = pd.read_csv('../data/test.csv', parse_dates=['auditing_date', 'due_date'])#读入测试数据，将成交日期和应还款日期解析为日期格式

sub = test_df[['listing_id', 'due_date', 'due_amt']]#标的id+成交日期+应还款金额

prob_cols = ['prob_{}'.format(i) for i in range(33)]#prob_0 至 prob_32

for i, f in enumerate(prob_cols):
    sub[f] = test_pred_prob[:, i]

prob_cols_2 = ['prob_{}'.format(i) for i in range(32)]

sub['sum'] = sub[prob_cols_2].apply(lambda x: x.sum(), axis=1)

for f in prob_cols_2:
    sub[f] = sub[f]/sub['sum']


del sub['sum']

sub['y']=y_pred
sub_example = pd.read_csv('../data/submission.csv', parse_dates=['repay_date'])

sub_example = sub_example.merge(sub, on='listing_id', how='left')

sub_example['days'] = (sub_example['due_date'] - sub_example['repay_date']).dt.days

# shape = (-1, 33)

test_prob = sub_example[prob_cols].values

test_labels = sub_example['days'].values


test_prob = [test_prob[i][test_labels[i]] for i in range(test_prob.shape[0])]#第i个样本第test_labels[i]天的还款概率

sub_example['repay_amt'] = sub_example['due_amt'] * test_prob#第i个样本第test_labels[i]天的预测的还款金额

# 清除没有还钱的行
drop_left=sub_example[sub_example['y']==32].drop_duplicates(subset='listing_id', keep='first')
sub_example.drop(sub_example[sub_example['y']==32].index)

drop_left['repay_date']=None
drop_left['repay_amt']=None

sub_example=pd.concat([sub_example,drop_left],axis=0)

sub_example[['listing_id', 'repay_date', 'repay_amt']].to_csv('../data/paipaidai_xgboost.csv', index=False)