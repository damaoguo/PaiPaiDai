#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: predict.py
@time: 19-6-27 上午10:37
@desc: 使用训练好的模型进行预测
"""
from sklearn.externals import joblib
import pandas as pd
import numpy as np

test1=pd.read_csv("../../data/reference_01_test.csv")

user_id=test1['user_id']

test_X=test1.drop(['user_id', 'listing_id','auditing_date'],axis=1)
for col in test_X.columns:
    test_X[col]=test_X[col].fillna(test_X[col].mean())


clf=joblib.load('../../data/paipaidai_01.pkl')

pred_y=clf.predict(test_X)
pred_y=[list(x).index(max(x)) for x in pred_y]
pred_y=np.array(pred_y)

pred_y=pd.DataFrame(pred_y)

out=pd.DataFrame()
out['user_id']=user_id
out['y']=pred_y
out.to_csv("../../data/paipaidai_01.csv")