#!/usr/bin/env python
# coding: utf-8
数据来源：
本地化BTC_USTD
文件1:BTC_USTD 1h  from 2024-01-01 to 2025-04-13 

模型：LSTM
参考来源：群里面的project 样本
特征工程：滞后性3h
# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

读取原始BTC_USDT 1h 数据。from 2024-01-01 to 2025-04-13 
# In[2]:


file_path = "/Users/rbw/Desktop/binanceus/BTC_USDT_1h.csv"  #路径
df = pd.read_csv(file_path)

# 转换时间戳为datetime并设为索引
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# 原始列：timestamp, datetime, open, high, low, close, volume
features = ['open', 'high', 'low', 'close', 'volume']  # 选择需要生成滞后特征的列

标准化数据
# In[3]:


scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df[features])

生成滞后特征和监督学习格式数据（过去3小时预测未来2小时）
# In[4]:


def series_to_supervised(data, features, n_in=3, n_out=2, dropnan=True):
    """
    将时间序列转换为监督学习格式
    :param data: 标准化后的数据（NumPy数组）
    :param features: 原始特征名列表（如 ['open', 'high', 'close', ...]）
    :param n_in: 滞后步数（过去n_in个时间点）
    :param n_out: 预测步数（未来n_out个时间点）
    :param dropnan: 是否删除NaN行
    :return: 包含滞后特征的DataFrame
    """
    n_vars = len(features)
    df = pd.DataFrame(data)
    cols, names = [], []
    
    # 输入序列 (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f'{features[j]}(t-{i})' for j in range(n_vars)]
    
    # 预测序列 (t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [f'{features[j]}(t)' for j in range(n_vars)]
        else:
            names += [f'{features[j]}(t+{i})' for j in range(n_vars)]
    
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# 生成监督学习格式数据（过去3小时预测未来2小时）
n_lag = 3
n_pred = 2
reframed = series_to_supervised(scaled, features, n_lag, n_pred)

提取输入(X)和输出(Y) ，存入reframed文件 （格式：过去三小时的特征+之后两小时）
# In[5]:


# 输入特征：所有滞后列（如 open(t-3), high(t-3), ..., volume(t-1)）
input_features = [col for col in reframed.columns if any(f in col for f in features) and 't-' in col]
X = reframed[input_features]

# 输出目标：未来close价格（close(t)和close(t+1)）
output_features = ['close(t)', 'close(t+1)']
Y = reframed[output_features]

输出结果
# In[6]:


print("\n--------------- 特征工程后的数据样例 ---------------")
print(reframed.head())

print("\n--------------- 输入特征(X) ---------------")
print(X.head())

print("\n--------------- 输出目标(Y) ---------------")
print(Y.head())

