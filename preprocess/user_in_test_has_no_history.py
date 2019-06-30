#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: user_in_test_has_no_history.py
@time: 19-6-30 下午2:58
@desc: 没有任何历史记录的用户,共11295个
"""

import pandas as pd
import numpy as np

# 去掉listing_log中2019-02-01及以后的标之后,求剩下的数据

listing_info=pd.read_csv("../data/listing_info.csv")
print(listing_info.shape)

listing_info_before=listing_info[listing_info['auditing_date']<'2019-02-01']
print(listing_info_before.shape)
listing_info_before_user_id=listing_info_before['user_id'].tolist()


test=pd.read_csv("../data/test.csv")
print("test.shape:",test.shape)
test_user_id=test['user_id'].tolist()


without_history=set(test_user_id)-set(listing_info_before_user_id)

print('test user without history:',len(without_history))

test_user_without_history=pd.DataFrame()

test_user_without_history['user_id']=np.array(list(without_history))

test_user_without_history=test_user_without_history.merge(test,on='user_id',how='left')
print('test_user_without_history.shape:',test_user_without_history.shape)
test_user_without_history.to_csv('../data/test_user_without_history.csv',index=False)