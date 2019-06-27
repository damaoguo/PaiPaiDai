# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 09:27:15 2019

@author: huyue
"""

#专门用于用户历史数据集
import pandas as pd
import numpy as np
from method import getFeatures
from method import getProData
from method import getTrain
from method import getRatio
from method import getUserInfo
# 后台运行代码
# nohup python -u user_log.py > nohup.out 2>&1 &
def getRepayLogs(data,user_repay_logs):
    
    #对用户历史数据集的处理
    #只保留第一期的数据
    user_repay_logs=user_repay_logs[user_repay_logs['order_id']==1]
    #先删除2019年的数据，也就是根本不能用的数据
    user_repay_logs['due_date']=pd.to_datetime(user_repay_logs['due_date'])
    user_repay_logs1=user_repay_logs.sort_values("due_date").set_index('due_date')
    user_repay_logs1=user_repay_logs1.truncate(after="2019-01").reset_index()

    #将两个数据集进行连接
    user_repay_logs_train=pd.merge(data[['user_id','auditing_date']],user_repay_logs1, how='left', on=['user_id'])
    user_repay_logs_train1=getProData(user_repay_logs_train)
    #开始做特征了  
    data1=getFeatures(user_repay_logs_train1)
    return data1

###############################################训练集#############################

print("start train data")
#%%读入用户历史数据集
user_repay_logs=pd.read_csv("../../data/user_repay_logs.csv")
#读入训练集
train=pd.read_csv("../../data/train.csv")
#读入用户历史数据集
user_info=pd.read_csv("../../data/user_info.csv")
#读入标的的属性
listing_info=pd.read_csv("../../data/listing_info.csv")
listing_info=listing_info[['listing_id','rate','principal']]
#得到训练集的特征
data1=getRepayLogs(train,user_repay_logs)
#加入用户的个人信息特征
user_info_train=getUserInfo(train,user_info)
    
#对训练集的处理
train1=getTrain(train)
#得到分布比例
ratio=getRatio(train1)
#得到最终的训练数据集，并查漏补缺
train2=pd.merge(train1[['user_id','auditing_date','listing_id','due_amt','y']],data1, how='left', on=['user_id','auditing_date'])
train2=pd.merge(train2,user_info_train, how='left', on=['user_id','auditing_date','listing_id'])
train2=pd.merge(train2,listing_info, how='left', on=['listing_id'])
train2['due_amt']=pd.DataFrame(train2['due_amt'],dtype=np.float)
train2['y']=pd.DataFrame(train2['y'],dtype=np.int)
train2['age_cut']=pd.DataFrame(train2['age_cut'],dtype=np.int)

train2.to_csv("../../data/reference_01_train.csv",index=False)

#用均值填充缺失值
X=train2.drop(['user_id', 'listing_id','y','auditing_date'],axis=1)
for col in X.columns:  
    X[col]=X[col].fillna(X[col].mean())
    #X[col]=X[col].fillna(0)
y=train1['y']

print("train data over")
########################################测试集####################################
#%%读入测试集用户历史数据集
user_repay_logs=pd.read_csv("../../data/user_repay_logs.csv")
#读入训练集
test=pd.read_csv("../../data/test.csv")
#读入用户历史数据集
user_info=pd.read_csv("../../data/user_info.csv")

#加入用户的个人信息特征
user_info_test=getUserInfo(test,user_info)


print("start test data...")
#得到测试集的特征
data1=getRepayLogs(test,user_repay_logs)
test['auditing_date']=pd.to_datetime(test['auditing_date'])
test1=pd.merge(test[['user_id','auditing_date','listing_id','due_amt']],data1, how='left', on=['user_id','auditing_date'])
test1=pd.merge(test1,user_info_test, how='left', on=['user_id','auditing_date','listing_id'])
test1['due_amt']=pd.DataFrame(test1['due_amt'],dtype=np.float)
test1['age_cut']=pd.DataFrame(test1['age_cut'],dtype=np.int)
#用均值填充缺失值

test1.to_csv("../../data/reference_01_test.csv",index=False)

test_X=test1.drop(['user_id', 'listing_id','auditing_date'],axis=1)
for col in test_X.columns:  
    test_X[col]=test_X[col].fillna(test_X[col].mean())
print("test data over...")
#######################################模型训练###################################
#%%模型训练
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

train_data=lgb.Dataset(X_train,label=y_train)
validation_data=lgb.Dataset(X_test,label=y_test)
params={
    'boosting_type': 'gbdt',
    'learning_rate':0.1,
    'lambda_l1':0.1,
    'lambda_l2':0.2,
    'max_depth':5,
    #'metric': 'multi_error',  
    'objective':'multiclass',
    'num_class':3,  #lightgbm.basic.LightGBMError: b‘Number of classes should be specified and greater than 1 for multiclass training‘
}

print("start train...")
clf=lgb.train(params,train_data,valid_sets=[validation_data])
joblib.dump(clf,'../../data/paipaidai_01.pkl')


print("start predict...")
y_pred=clf.predict(X_test)
y_pred=[list(x).index(max(x)) for x in y_pred]
print(y_pred)
print("accuracy_score:{}".format(accuracy_score(y_test,y_pred)))
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))





pred_y=clf.predict(test_X)
pred_y=np.array(pred_y)

ouput=pd.DataFrame(pred_y)
output.to_csv("../../data/paipaidai_01.csv")