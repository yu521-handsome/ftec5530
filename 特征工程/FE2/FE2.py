#!/usr/bin/env python
# coding: utf-8
数据来源：
本地化BTC_USTD
文件1:BTC_USTD 1h  from 2024-01-01 to 2025-04-13 

模型：梯度提升树（XGBoost/LightGBM）
参考来源：deepseek简单提供几个特征工程处理
特征工程：1.价格波动范围 2.涨跌幅（百分比变化） 3.简单移动平均（4小时窗口） 4.交易量变化  5.时间特征
# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

加载数据
# In[2]:


# 1. 加载数据
file_path = "/Users/rbw/Desktop/binanceus/BTC_USDT_1h.csv"
df = pd.read_csv(file_path)

数据清洗
# In[3]:


# 转换时间戳为datetime格式（假设timestamp列为毫秒级）
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('datetime', inplace=True)

# 检查缺失值
print("缺失值统计:")
print(df.isnull().sum())

# 前向填充缺失值（如果有）
df = df.ffill()

# 验证填充结果
print("\n填充后缺失值统计:")
print(df.isnull().sum())

特征工程
# In[4]:


# 价格波动范围
df['price_range'] = df['high'] - df['low']

# 涨跌幅（百分比变化）
df['price_change_pct'] = df['close'].pct_change() * 100

# 简单移动平均（4小时窗口）
df['sma_4'] = df['close'].rolling(window=4).mean()

# 交易量变化
df['volume_change'] = df['volume'].pct_change()

# 时间特征
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6

处理特征中的NaN（由于rolling计算产生）
# In[5]:


df.dropna(inplace=True)

存入新对象df_clean。删除包含NaN的行（确保所有特征有效）
# In[6]:


df_clean = df.dropna().copy()  # 存入新对象df_clean

验证处理后的数据：直接显示索引（datetime）和其他列
# In[7]:


print("\n（datetime作为索引）：")
print(df_clean[['open', 'close', 'price_range', 'price_change_pct', 'sma_4', 'volume_change', 'hour']].head())
print("对应的时间索引：")
print(df_clean.index[:5])

# 检查形状变化
print(f"\n原始数据行数: {len(df)} → 处理后行数: {len(df_clean)} (删除了{len(df) - len(df_clean)}行NaN)")

