#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: baseline_33class_v2.py
@time: 19-6-28 上午8:28
@desc:
"""

import pandas as pd

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
from lightgbm.sklearn import LGBMClassifier


from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score

from sklearn.feature_extraction.text import CountVectorizer

from scipy import sparse

from scipy.stats import kurtosis

import time

import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
from load_data import load_2


train_values, test_values,clf_labels,amt_labels,train_due_amt_df,sub,train_num=load_2()

print(train_values.shape)

# 五折验证也可以改成一次验证，按时间划分训练集和验证集，以避免由于时序引起的数据穿越问题。



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)#5折交叉验证



clf = LGBMClassifier(#lightGBM分类模型

    learning_rate=0.05,

    n_estimators=10000,

    subsample=0.8,

    subsample_freq=1,

    colsample_bytree=0.8

)

amt_oof = np.zeros(train_num)#训练集大小，保存实际还款那一天的预测还款金额

prob_oof = np.zeros((train_num, 33))#训练集大小作为行数，33作为列数，就是每个训练集对应的每天的还款概率

test_pred_prob = np.zeros((test_values.shape[0], 33))#测试集个数作为行数，33为列数，测试集每天的还款概率

for i, (trn_idx, val_idx) in enumerate(skf.split(train_values, clf_labels)):#特征数据，标签

    print(i, 'fold...')

    t = time.time()



    trn_x, trn_y = train_values.values[trn_idx], clf_labels[trn_idx]#训练集

    val_x, val_y = train_values.values[val_idx], clf_labels[val_idx]#测试集

    val_repay_amt = amt_labels[val_idx]#训练集对应的实际还款金额

    val_due_amt = train_due_amt_df.iloc[val_idx]#训练集对应的应还款金额



    clf.fit(

        trn_x, trn_y,

        eval_set=[(trn_x, trn_y), (val_x, val_y)],

        early_stopping_rounds=100, verbose=5

    )#交叉验证进行训练

    # shepe = (-1, 33)

    val_pred_prob_everyday = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)#预测，是一个shepe = (-1, 33)的结果，每一行的33个数代表每一类别的概率

    prob_oof[val_idx] = val_pred_prob_everyday#一折训练集的预测结果，将预测结果填到相对应的验证集的位置上



    val_pred_prob_today = [val_pred_prob_everyday[i][val_y[i]] for i in range(val_pred_prob_everyday.shape[0])]#i表示第i条验证集的预测结果，该结果包含33个概率，val_y[i]表示验证集真实的label，所以val_pred_prob_today保存的就是第i条验证集样本的33个预测类别中，真实类别的概率

    val_pred_repay_amt = val_due_amt['due_amt'].values * val_pred_prob_today#实际还款那一天的预测还款金额=实际还款那一天对应的概率*应还款金额

    print('val rmse:', np.sqrt(mean_squared_error(val_repay_amt, val_pred_repay_amt)))#计算实际还款金额与预测出来的那一天的还款金额(那一天的预测概率*应还款金额)的均方差

    print('val mae:', mean_absolute_error(val_repay_amt, val_pred_repay_amt))#计算实际还款金额与预测出来的那一天的还款金额(那一天的预测概率*应还款金额)的绝对值之差的平均值

    amt_oof[val_idx] = val_pred_repay_amt#实际还款那一天的预测还款金额

    test_pred_prob += clf.predict_proba(test_values.values, num_iteration=clf.best_iteration_) / skf.n_splits#每个测试集样本的33个类别概率

    joblib.dump(clf, '../data/paipaidai_v3_%d.pkl'%i)

    print('runtime: {}\n'.format(time.time() - t))



print('\ncv rmse:', np.sqrt(mean_squared_error(amt_labels, amt_oof)))#实际还款金额与预测出来的还款金额的均方差

print('cv mae:', mean_absolute_error(amt_labels, amt_oof))#实际还款金额与预测出来的还款金额的绝对值误差的均值

print('cv logloss:', log_loss(clf_labels, prob_oof))#训练集标签，

print('cv acc:', accuracy_score(clf_labels, np.argmax(prob_oof, axis=1)))



prob_cols = ['prob_{}'.format(i) for i in range(33)]#prob_0 至 prob_32

for i, f in enumerate(prob_cols):#遍历每一个prob_i

    sub[f] = test_pred_prob[:, i] #sub是测试集的['listing_id', 'auditing_date', 'due_amt'],也就是在后面增加prob_0 至 prob_32共33列，每一列填充对应的概率值



sub_example = pd.read_csv('../data/submission.csv', parse_dates=['repay_date'])
test=pd.read_csv('../data/test.csv',parse_dates=['due_date'])

due_date=pd.DataFrame()
due_date['listing_id']=test['listing_id']
due_date['due_date']=test['due_date']

sub_example = sub_example.merge(sub, on='listing_id', how='left')
sub_example = sub_example.merge(due_date, on='listing_id', how='left')


sub_example['days'] = (sub_example['due_date'] - sub_example['repay_date']).dt.days

# shape = (-1, 33)

test_prob = sub_example[prob_cols].values

test_labels = sub_example['days'].values



test_prob = [test_prob[i][test_labels[i]] for i in range(test_prob.shape[0])]#第i个样本第test_labels[i]天的还款概率


sub_example['repay_amt'] = sub_example['due_amt'] * test_prob#第i个样本第test_labels[i]天的预测的还款金额

sub_example[['listing_id', 'repay_date', 'repay_amt']].to_csv('../data/sub_baseline_v8.csv', index=False)