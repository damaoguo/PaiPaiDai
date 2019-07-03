#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: paipaidai_xgboost.py
@time: 19-7-3 上午9:12
@desc: 使用xgboost模型
"""
from load_data import load_2
import xgboost as xgb
import sys
import pandas as pd
import numpy as np
import time
import threading

start_time=time.time()
print("start load data...")
train_values, test_values,clf_labels,amt_labels,train_due_amt_df,sub,train_num=load_2()
print("load data over,time cost:",time.time()-start_time)

X,Y=train_values[:1000],clf_labels[:1000]

params={
    'max_depth':12,
    'learning_rate':0.05,
    'n_estimators':752,
    'silent':True,
    'objective':"multi:softmax",
    'nthread':4,
    'gamma':0,
    'max_delta_step':0,
    'subsample':1,
    'colsample_bytree':0.9,
    'colsample_bylevel':0.9,
    'reg_alpha':1,
    'reg_lambda':1,
    'scale_pos_weight':1,
    'base_score':0.5,
    'seed':2019,
    'missing':None,
    'num_class':33
}

plst = list(params.items())
num_rounds = 5000 # 迭代次数

# 测试集数据
xgb_test=xgb.DMatrix(test_values)

from sklearn.model_selection import StratifiedKFold

n_splits = 5
seed = 42

skf = StratifiedKFold(n_splits=n_splits,random_state=seed,shuffle=True)

def run(index,train_index,test_index):
    print(index)
    X_train, X_valid, y_train, y_valid = X[train_index],X[test_index],Y[train_index],Y[test_index]
    xgb_train=xgb.DMatrix(X_train,label=y_train)
    xgb_val=xgb.DMatrix(X_valid,y_valid)
    watchlist = [(xgb_val, 'val')]
    model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=50)
    model.save_model('../data/xgb%s.model'%str(index))

def main():
    threads=[]
    for index,(train_index,test_index) in enumerate(skf.split(X,Y)):
        t=threading.Thread(target=run,args=(index,train_index,test_index,))
        print("start thread:%d"%index)
        t.start()
        threads.append(t)
    for k in threads:
        k.join()

if __name__ == '__main__':
    start_time=time.time()
    main()

    test_pred_prob = np.zeros((test_values.shape[0], 33))
    for index in range(0,5):
        bst=xgb.Booster({'nthread':4})
        bst.load_model('../data/xgb%s.model'%str(index))
        test_pred_prob += bst.predict(xgb_test)/5
    prob_cols = ['prob_{}'.format(i) for i in range(33)]  # prob_0 至 prob_32

    for i, f in enumerate(prob_cols):  # 遍历每一个prob_i

        sub[f] = test_pred_prob[:,i]

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

    sub_example[['listing_id', 'repay_date', 'repay_amt']].to_csv('../data/submission_xgboost.csv', index=False)

print("over! time cost:%s"%str(time.time()-start_time))

