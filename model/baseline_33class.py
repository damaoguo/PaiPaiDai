#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: baseline_33class.py
@time: 19-6-22 下午12:25
@desc:
    方案思路：33分类。
              label为还款日期距成交日期的天数，可能的情况有0天到31天，未还款定义为32，一共33个类别。
              预测出每个label对应的概率，然后分别乘以应还的金额，就是每天需要还的金额。
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from scipy.stats import kurtosis
import time
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)




#########################################################################################
# 加载训练数据
#########################################################################################
train_df = pd.read_csv('../data/train.csv', parse_dates=['auditing_date', 'due_date', 'repay_date'])

train_df['repay_date'] = train_df[['due_date', 'repay_date']].apply(
    lambda x: x['repay_date'] if x['repay_date'] != '\\N' else x['due_date'], axis=1
)

train_df['repay_amt'] = train_df['repay_amt'].apply(lambda x: x if x != '\\N' else 0).astype('float32')

# 标签是到还款日距离这测日的天数
train_df['label'] = (train_df['repay_date'] - train_df['auditing_date']).dt.days

# 还款金额为0的行标签为0
train_df.loc[train_df['repay_amt'] == 0, 'label'] = 32

clf_labels = train_df['label'].values
amt_labels = train_df['repay_amt'].values

del train_df['label'], train_df['repay_amt'], train_df['repay_date']

train_due_amt_df = train_df[['due_amt']]
# 记录训练数据集合的条数
train_num = train_df.shape[0]

#########################################################################################
# 加载测试数据
#########################################################################################

test_df = pd.read_csv('../data/test.csv', parse_dates=['auditing_date', 'due_date'])
sub = test_df[['listing_id', 'auditing_date', 'due_amt']]
# 训练集和测试集合合并
df = pd.concat([train_df, test_df], axis=0, ignore_index=True)


#########################################################################################
# 加载标信息
#########################################################################################
listing_info_df = pd.read_csv('../data/listing_info.csv')
del listing_info_df['user_id'], listing_info_df['auditing_date']

# 拼接历史log
df = df.merge(listing_info_df, on='listing_id', how='left')


#########################################################################################
# 加载用户数据
#########################################################################################
# 表中有少数user不止一条记录，因此按日期排序，去重，只保留最新的一条记录。
user_info_df = pd.read_csv('../data/user_info.csv', parse_dates=['reg_mon', 'insertdate'])
user_info_df.rename(columns={'insertdate': 'info_insert_date'}, inplace=True)
user_info_df = user_info_df.sort_values(by='info_insert_date', ascending=False).drop_duplicates('user_id').reset_index(drop=True)

# 拼接用户信息
df = df.merge(user_info_df, on='user_id', how='left')


#########################################################################################
# 加载用户画像
#########################################################################################
# 同上
user_tag_df = pd.read_csv('../data/user_taglist.csv', parse_dates=['insertdate'])
user_tag_df.rename(columns={'insertdate': 'tag_insert_date'}, inplace=True)
user_tag_df = user_tag_df.sort_values(by='tag_insert_date', ascending=False).drop_duplicates('user_id').reset_index(drop=True)
df = df.merge(user_tag_df, on='user_id', how='left')



#########################################################################################
# 加载历史数据
#########################################################################################
# 历史记录表能做的特征远不止这些
repay_log_df = pd.read_csv('../data/user_repay_logs.csv', parse_dates=['due_date', 'repay_date'])
# 由于题目任务只预测第一期的还款情况，因此这里只保留第一期的历史记录。当然非第一期的记录也能提取很多特征。
repay_log_df = repay_log_df[repay_log_df['order_id'] == 1].reset_index(drop=True)
repay_log_df['repay'] = repay_log_df['repay_date'].astype('str').apply(lambda x: 1 if x != '2200-01-01' else 0)
repay_log_df['early_repay_days'] = (repay_log_df['due_date'] - repay_log_df['repay_date']).dt.days
repay_log_df['early_repay_days'] = repay_log_df['early_repay_days'].apply(lambda x: x if x >= 0 else -1)
for f in ['listing_id', 'order_id', 'due_date', 'repay_date', 'repay_amt']:
    del repay_log_df[f]
group = repay_log_df.groupby('user_id', as_index=False)
repay_log_df = repay_log_df.merge(
    group['repay'].agg({'repay_mean': 'mean'}), on='user_id', how='left'
)
repay_log_df = repay_log_df.merge(
    group['early_repay_days'].agg({
        'early_repay_days_max': 'max', 'early_repay_days_median': 'median', 'early_repay_days_sum': 'sum',
        'early_repay_days_mean': 'mean', 'early_repay_days_std': 'std'
    }), on='user_id', how='left'
)
repay_log_df = repay_log_df.merge(
    group['due_amt'].agg({
        'due_amt_max': 'max', 'due_amt_min': 'min', 'due_amt_median': 'median',
        'due_amt_mean': 'mean', 'due_amt_sum': 'sum', 'due_amt_std': 'std',
        'due_amt_skew': 'skew', 'due_amt_kurt': kurtosis, 'due_amt_ptp': np.ptp
    }), on='user_id', how='left'
)
del repay_log_df['repay'], repay_log_df['early_repay_days'], repay_log_df['due_amt']
repay_log_df = repay_log_df.drop_duplicates('user_id').reset_index(drop=True)
df = df.merge(repay_log_df, on='user_id', how='left')




#########################################################################################
# 处理合并后的数据
#########################################################################################
# 性别,电话,省份,城市
cate_cols = ['gender', 'cell_province', 'id_province', 'id_city']
for f in cate_cols:
    df[f] = df[f].map(dict(zip(df[f].unique(), range(df[f].nunique())))).astype('int32')

df['due_amt_per_days'] = df['due_amt'] / (train_df['due_date'] - train_df['auditing_date']).dt.days

# 时间相关特征
date_cols = ['auditing_date', 'due_date', 'reg_mon', 'info_insert_date', 'tag_insert_date']
for f in date_cols:
    if f in ['reg_mon', 'info_insert_date', 'tag_insert_date']:
        df[f + '_year'] = df[f].dt.year
    df[f + '_month'] = df[f].dt.month
    if f in ['auditing_date', 'due_date', 'info_insert_date', 'tag_insert_date']:
        df[f + '_day'] = df[f].dt.day
        df[f + '_dayofweek'] = df[f].dt.dayofweek
df.drop(columns=date_cols, axis=1, inplace=True)

# 用户画像
df['taglist'] = df['taglist'].astype('str').apply(lambda x: x.strip().replace('|', ' ').strip())
tag_cv = CountVectorizer(min_df=10, max_df=0.9).fit_transform(df['taglist'])

del df['user_id'], df['listing_id'], df['taglist']

# 将数据离散化
df = pd.get_dummies(df, columns=cate_cols)
df = sparse.hstack((df.values, tag_cv), format='csr', dtype='float32')
# 切分数据



#########################################################################################
# 将之前合并的数据进行分开,进行保存!!!
#########################################################################################
train_values, test_values = df[:train_num], df[train_num:]






#########################################################################################
# 训练模型
#########################################################################################
print(train_values.shape)
# 五折验证也可以改成一次验证，按时间划分训练集和验证集，以避免由于时序引起的数据穿越问题。
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
clf = LGBMClassifier(
    learning_rate=0.05,
    n_estimators=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=2019
)
amt_oof = np.zeros(train_num)
prob_oof = np.zeros((train_num, 33))

# 保存测试集的预测结果
test_pred_prob = np.zeros((test_values.shape[0], 33))

print("start training...")
for i, (trn_idx, val_idx) in enumerate(skf.split(train_values, clf_labels)):
    print(i, 'fold...')
    t = time.time()

    trn_x, trn_y = train_values[trn_idx], clf_labels[trn_idx]
    val_x, val_y = train_values[val_idx], clf_labels[val_idx]
    val_repay_amt = amt_labels[val_idx]
    val_due_amt = train_due_amt_df.iloc[val_idx]

    clf.fit(
        trn_x, trn_y,
        eval_set=[(trn_x, trn_y), (val_x, val_y)],
        early_stopping_rounds=100, verbose=5
    )
    # shepe = (-1, 33)
    val_pred_prob_everyday = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
    prob_oof[val_idx] = val_pred_prob_everyday
    val_pred_prob_today = [val_pred_prob_everyday[i][val_y[i]] for i in range(val_pred_prob_everyday.shape[0])]
    val_pred_repay_amt = val_due_amt['due_amt'].values * val_pred_prob_today
    print('val rmse:', np.sqrt(mean_squared_error(val_repay_amt, val_pred_repay_amt)))
    print('val mae:', mean_absolute_error(val_repay_amt, val_pred_repay_amt))
    amt_oof[val_idx] = val_pred_repay_amt

    # 对于测试集预测五次,直接在这里求了五次计算的均值
    test_pred_prob += clf.predict_proba(test_values, num_iteration=clf.best_iteration_) / skf.n_splits

    print('runtime: {}\n'.format(time.time() - t))

print('\ncv rmse:', np.sqrt(mean_squared_error(amt_labels, amt_oof)))
print('cv mae:', mean_absolute_error(amt_labels, amt_oof))
print('cv logloss:', log_loss(clf_labels, prob_oof))
print('cv acc:', accuracy_score(clf_labels, np.argmax(prob_oof, axis=1)))

prob_cols = ['prob_{}'.format(i) for i in range(33)]
for i, f in enumerate(prob_cols):
    sub[f] = test_pred_prob[:, i]
sub_example = pd.read_csv('../data/submission.csv', parse_dates=['repay_date'])
sub_example = sub_example.merge(sub, on='listing_id', how='left')
sub_example['days'] = (sub_example['repay_date'] - sub_example['auditing_date']).dt.days

# shape = (-1, 33)
test_prob = sub_example[prob_cols].values
test_labels = sub_example['days'].values
test_prob = [test_prob[i][test_labels[i]] for i in range(test_prob.shape[0])]

#
sub_example['repay_amt'] = sub_example['due_amt'] * test_prob
sub_example[['listing_id', 'repay_date', 'repay_amt']].to_csv('sub.csv', index=False)