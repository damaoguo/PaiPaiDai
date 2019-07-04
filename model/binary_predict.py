#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: binary_predict.py
@time: 19-7-2 下午7:27
@desc: 二分类预测结果
"""
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
from sklearn.externals import joblib
import time
from load_data import load
import figures
train_values,test_values,clf_labels,clf_labels_r,clf_labels_2=load()


test_pred_prob = np.zeros((test_values.shape[0], 2))#测试集个数作为行数，33为列数，测试集每天的还款概率
train_pred_prob = np.zeros((train_values.shape[0], 2))#测试集个数作为行数，33为列数，测试集每天的还款概率


for i in range(5):#特征数据，标签

    print(i, 'fold...')

    t = time.time()
    clf=joblib.load('../data/paipaidai_binary_%d.pkl'%i)
    train_pred_prob += clf.predict_proba(train_values.values, num_iteration=clf.best_iteration_) / 5
    test_pred_prob += clf.predict_proba(test_values.values, num_iteration=clf.best_iteration_) / 5#每个测试集样本的33个类别概率

y_pred=[list(x).index(max(x)) for x in test_pred_prob]

y_train=[list(x).index(max(x)) for x in train_pred_prob]


outcome=pd.DataFrame()
outcome['true']=clf_labels_2
outcome['predict']=np.array(y_train)
outcome.to_csv("../data/binary_train_predict.csv",index=False)

# 绘制混淆矩阵

alphabet=softwares=["yes","no"]
figures.plot_confusion_matrix(y_train, clf_labels_2,alphabet, "../data/confusion.png")



from collections import Counter
counter=Counter(y_pred)
print(counter)

test=pd.read_csv("../data/test.csv")



listing_id=test['listing_id']
# 新建提交数据
out=pd.DataFrame()

out['listing_id']=listing_id
out['binary']=y_pred


# 判断出Counter({0: 125062, 1: 4938}),只有4938个用户没有按时还款?约为4%不还款
out.to_csv("../data/paipaidai_binary.csv",index=False)

