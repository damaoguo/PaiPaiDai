#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: user_tags.py
@time: 19-6-26 下午7:37
@desc:
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# 用户画像只有60万,有的用户的画像找不到!!!
user_taglist=pd.read_csv("../data/user_taglist.csv")
# 聚类或者降维
# 用户画像有5000多个维度,使用聚类算法聚类,将缺省的统一算为0,tag取值范围为range(1,5986)
# user_taglist_detail=user_taglist['taglist'].tolist()
# user_tags=[]
# print("start convert")
# for s in user_taglist_detail:
#     tags=np.zeros(5986)
#     temp=s.split("|")
#     index=[]
#     for i in temp:
#         if(i=="\\N"):
#             break
#         index.append(int(i))
#     tags[index]=1
#     user_tags.append(tags)
# print("finish convert")
#
# user_tags=np.array(user_tags)
# f=open('../data/user_tags.pkl','wb')
# pickle.dump(user_tags,f,protocol = 4)

pkl_file = open('../data/user_tags.pkl', 'rb')

user_tags = pickle.load(pkl_file)
print("finish load data...")

estimator = KMeans(n_clusters=10)#构造聚类器
print("start training...")
estimator.fit(user_tags)#聚类

label_pred = estimator.labels_ #获取聚类标签

label_file = open('../data/user_tags_label.pkl', 'wb')
pickle.dump(label_pred,label_file,protocol = 4)

print(label_pred)