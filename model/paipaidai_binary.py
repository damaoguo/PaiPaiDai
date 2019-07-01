#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: paipaidai_binary.py
@time: 19-7-1 上午9:46
@desc: 二分类,判断是否会超期
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
from lightgbm.sklearn import LGBMClassifier
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score
from load_data import load

train_values,test_values,clf_labels,clf_labels_r,clf_labels_2=load()

train_num=1000000

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



test_pred_prob = np.zeros((test_values.shape[0], 2))#测试集个数作为行数，33为列数，测试集每天的还款概率



for i, (trn_idx, val_idx) in enumerate(skf.split(train_values, clf_labels_2)):#特征数据，标签

    print(i, 'fold...')

    t = time.time()

    trn_x, trn_y = train_values.values[trn_idx], clf_labels_2[trn_idx]#训练集

    val_x, val_y = train_values.values[val_idx], clf_labels_2[val_idx]#测试集

    clf.fit(

        trn_x, trn_y,

        eval_set=[(trn_x, trn_y), (val_x, val_y)],

        early_stopping_rounds=100, verbose=5

    )#交叉验证进行训练

    # shepe = (-1, 33)


    test_pred_prob += clf.predict_proba(test_values.values, num_iteration=clf.best_iteration_) / skf.n_splits#每个测试集样本的33个类别概率

    joblib.dump(clf, '../data/paipaidai_binary_%d.pkl'%i)

    print('runtime: {}\n'.format(time.time() - t))