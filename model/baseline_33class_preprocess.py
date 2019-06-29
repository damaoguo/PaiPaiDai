#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: baseline_33class_preprocess.py
@time: 19-6-28 下午7:15
@desc: 预处理 train和test
"""

import pandas as pd

import numpy as np


from scipy.stats import kurtosis


import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)


#训练集，将标的成交日期、应还款日期和实际还款日期解析为时间格式

train_df = pd.read_csv('../data/train.csv', parse_dates=['auditing_date', 'due_date', 'repay_date'])

train_df['repay_date'] = train_df[['due_date', 'repay_date']].apply(

    lambda x: x['repay_date'] if x['repay_date'] != '\\N' else x['due_date'], axis=1

)#实际还款日期若为空，表示逾期，则将其赋值为应还款日期，并且实际还款金额置为0

#实际还款金额,如果实际还款金额为空，则赋值为0

train_df['repay_amt'] = train_df['repay_amt'].apply(lambda x: x if x != '\\N' else 0).astype('float32')

train_df['label'] = (train_df['repay_date'] - train_df['auditing_date']).dt.days#实际还款日期-成交日期，将其作为标签label

train_df.loc[train_df['repay_amt'] == 0, 'label'] = 32#实际还款金额为0的，都是逾期的，将其分类label变为32


# 构建不同的标签数据
train_labels=pd.DataFrame()
train_labels['label_32']=train_df['label']
train_labels['label_2']=train_df['label']

train_labels.loc[train_labels['label_2']!=32,'label_2']=0
train_labels.loc[train_labels['label_2']==32,'label_2']=1

clf_labels = train_df['label'].values#标签一列，为0-32的数

amt_labels = train_df['repay_amt'].values#实际还款金额一列

del train_df['label'], train_df['repay_amt'], train_df['repay_date']#将没有用的三列删除，这样训练集和测试集的字段就相同了




train_due_amt_df = train_df[['due_amt']]#应还款金额

train_num = train_df.shape[0]#样本数量

test_df = pd.read_csv('../data/test.csv', parse_dates=['auditing_date', 'due_date'])#读入测试数据，将成交日期和应还款日期解析为日期格式

sub = test_df[['listing_id', 'auditing_date', 'due_amt']]#标的id+成交日期+应还款金额

df = pd.concat([train_df, test_df], axis=0, ignore_index=True)



listing_info_df = pd.read_csv('../data/listing_info.csv')#标的属性表

del listing_info_df['user_id'], listing_info_df['auditing_date']#将用户id和标的成交日期删除，因为这两项在训练集和测试集中都存在

df = df.merge(listing_info_df, on='listing_id', how='left')



# 表中有少数user不止一条记录，因此按日期排序，去重，只保留最新的一条记录。

user_info_df = pd.read_csv('../data/user_info.csv', parse_dates=['reg_mon', 'insertdate'])#用户基础信息表

user_info_df.rename(columns={'insertdate': 'info_insert_date'}, inplace=True)#将用户数据的插入日期重命名

user_info_df_1 = user_info_df.sort_values(by='info_insert_date', ascending=False).drop_duplicates('user_id').reset_index(drop=True)#按照插入日期降序排列，去重，只保留最新的一条

user_info_df_1['foreign_land']=np.where(user_info_df_1['cell_province']==user_info_df_1['id_province'],'n','y')

modifyInfoNum=user_info_df.groupby('user_id').count()['info_insert_date'].to_frame().rename(columns={'info_insert_date':'modify_info_num'})

user_info_df_2=pd.merge(user_info_df_1,modifyInfoNum,how='left',on='user_id')

# 对年龄进行分桶
def map_age(s):

    if s < 25:

        return 'Young'

    elif s>24 and s < 36:

        return 'Middle1'

    elif s>35 and s < 51:

        return 'Middle2'

    else:

        return 'Old'

user_info_df_2['map_age']=user_info_df_2['age'].map(map_age)

df = df.merge(user_info_df_2, on='user_id', how='left')#将用户基础信息表合并到训练集之中



#用户画像标签列表，同样如上操作，排序去重合并

user_tag_df = pd.read_csv('../data/user_taglist.csv', parse_dates=['insertdate'])

user_tag_df.rename(columns={'insertdate': 'tag_insert_date'}, inplace=True)

user_tag_df_1 = user_tag_df.sort_values(by='tag_insert_date', ascending=False).drop_duplicates('user_id').reset_index(drop=True)

modifyTagListNum=user_tag_df.groupby('user_id').count()['tag_insert_date'].to_frame().rename(columns={'tag_insert_date':'modify_taglist_num'})

user_tag_df_2=pd.merge(user_tag_df_1,modifyTagListNum,how='left',on='user_id')

df = df.merge(user_tag_df_2, on='user_id', how='left')



user_behavior_logs = pd.read_csv('../data/user_behavior_logs.csv', parse_dates=['behavior_time'])

user_behavior_logs_1=user_behavior_logs.groupby('user_id').count()['behavior_type'].to_frame().rename(columns={'behavior_type':'behavior_num'})

df = df.merge(user_behavior_logs_1, on='user_id', how='left')



#基于全部还款记录计算每位user的逾期率

user_repay_logs=pd.read_csv('../data/user_repay_logs.csv',index_col=None)

user_repay_logs['expire']=np.where(user_repay_logs['repay_date']=='2200-01-01',1,0)

expire_cnt_ratio=user_repay_logs.groupby('user_id')['expire'].agg({'repay_mean':'mean'}).reset_index()

df = df.merge(expire_cnt_ratio, on='user_id', how='left')



# 历史记录表能做的特征远不止这些

repay_log_df = pd.read_csv('../data/user_repay_logs.csv', parse_dates=['due_date', 'repay_date'])#用户还款日志表

# 由于题目任务只预测第一期的还款情况，因此这里只保留第一期的历史记录。当然非第一期的记录也能提取很多特征。

repay_log_df = repay_log_df[repay_log_df['order_id'] == 1].reset_index(drop=True)#这里采用了简化处理的方法，由于训练集和测试集均为第一期，所以这里就只保留了第一期的数据，当然，从其他期中也可以提取出很多其他的特征

repay_log_df['repay'] = repay_log_df['repay_date'].astype('str').apply(lambda x: 1 if x != '2200-01-01' else 0)#标记是否如期还款，是，为1，逾期，为0

repay_log_df['early_repay_days'] = (repay_log_df['due_date'] - repay_log_df['repay_date']).dt.days#应还款日期-实际还款日期

repay_log_df['early_repay_days'] = repay_log_df['early_repay_days'].apply(lambda x: x if x >= 0 else -1)#提前还款的就填写原始天数，否则就是逾期的，这里标上-1

for f in ['listing_id', 'order_id', 'due_date', 'repay_date', 'repay_amt']:

    del repay_log_df[f]#删除标的id、标的的应还款期数序号、应还款日期、实际还款日期和实际还款金额

group = repay_log_df.groupby('user_id', as_index=False)#此时该日志的字段有，user_id用户id，due_amt应还款金额，repay是否逾期，early_repay_days应还款日期-实际还款日期，然后按照user_id用户进行聚合

repay_log_df = repay_log_df.merge(#将上面聚合之后的group，对repay这一列求平均值，计算出来的就是每个用户的按时还款的概率，将其以user_id为聚合键，返回到原列表中

    group['repay'].agg({'repay_mean': 'mean'}), on='user_id', how='left'

)

repay_log_df = repay_log_df.merge(

    group['early_repay_days'].agg({#求每个用户的：应还款日期-实际还款日期的最大值、中间值、总和、均值、方差，分别以user_id为键，返回到原列表中

        'early_repay_days_max': 'max', 'early_repay_days_median': 'median', 'early_repay_days_sum': 'sum',

        'early_repay_days_mean': 'mean', 'early_repay_days_std': 'std'

    }), on='user_id', how='left'

)

repay_log_df = repay_log_df.merge(

    group['due_amt'].agg({#每个user_id应还款金额的最大值、最小值、中位数、均值、总和、方差、偏度、峰度和取值范围

        'due_amt_max': 'max', 'due_amt_min': 'min', 'due_amt_median': 'median',

        'due_amt_mean': 'mean', 'due_amt_sum': 'sum', 'due_amt_std': 'std',

        'due_amt_skew': 'skew', 'due_amt_kurt': kurtosis, 'due_amt_ptp': np.ptp

    }), on='user_id', how='left'

)

del repay_log_df['repay'], repay_log_df['early_repay_days'], repay_log_df['due_amt']#对逾期与否、应还款日与实际还款日期之差、应还款金额进行了相关统计，所以就对这三列原始值进行了删除

repay_log_df = repay_log_df.drop_duplicates('user_id').reset_index(drop=True)#删除了重复user_id样本

df = df.merge(repay_log_df, on='user_id', how='left')#将由用户还款日志提取的用户相关统计信息，返回加到样本集中



cate_cols = ['gender', 'cell_province', 'id_province', 'id_city','foreign_land','map_age']#性别、手机号码归属省份、身份证归属省份、身份证归属城市


for f in cate_cols:

    df[f] = df[f].map(dict(zip(df[f].unique(), range(df[f].nunique())))).astype('int32')#这里相当于将对所有的值做了特征编码，通过map传入了字典，将对应的值替换为编码，并转化为int型



df['due_amt_per_days'] = df['due_amt'] / (train_df['due_date'] - train_df['auditing_date']).dt.days#应还款金额除以(应还款日期-成交日期)

date_cols = ['auditing_date', 'due_date', 'reg_mon', 'info_insert_date', 'tag_insert_date']#成交日期、应还款日期、用户注册年月、基础信息数据更新日期、用户画像标签更新日期

for f in date_cols:#处理各个日期数据

    if f in ['reg_mon', 'info_insert_date', 'tag_insert_date']:

        df[f + '_year'] = df[f].dt.year#取出年

    df[f + '_month'] = df[f].dt.month#取出月

    if f in ['auditing_date', 'due_date', 'info_insert_date', 'tag_insert_date']:

        df[f + '_day'] = df[f].dt.day#取出天

        df[f + '_dayofweek'] = df[f].dt.dayofweek#取出星期

df.drop(columns=date_cols, axis=1, inplace=True)#将原始数据删除



del df['taglist']#这下就可以将用户id、标的id和用户画像标签删除



df = pd.get_dummies(df, columns=cate_cols)#做独热编码

train_values, test_values = df[:train_num], df[train_num:]#拆分训练集和测试集，之前训练集和测试集是放在一起学习的

train_values['label_32']=train_labels['label_32']
train_values['label_2']=train_labels['label_2']

print("process over")
train_values.to_csv("../data/train_values.csv",index=False)
test_values.to_csv("../data/test_values.csv",index=False)