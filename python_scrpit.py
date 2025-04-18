#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.pipeline import Pipeline
import lightgbm as lgb

# ====================== 1. 加载预训练模型与特征列表 ======================
# 加载训练好的Pipeline模型（包含标准化、PCA、LightGBM）
MODEL_PATH = "best_model.pkl"
FEATURES_PATH = "selected_features.pkl"

best_model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "rb") as f:
    EXPECTED_FEATURES = pickle.load(f)


# ====================== 2. 模拟API数据获取（需替换为真实API调用） ======================
def fetch_raw_data_from_api(api_endpoint: str) -> pd.DataFrame:
   
    return pd.DataFrame(mock_data)


# ====================== 3. 特征工程函数（严格复现训练时的逻辑） ======================
def process_raw_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    将原始数据转换为模型所需的特征矩阵（与训练时完全一致）
    :param raw_df: 包含API原始数据的DataFrame
    :return: 处理后的特征矩阵（用于模型预测）
    """
    # ====================== A. 基础数据处理 ======================
    # 将时间戳转换为datetime并设置索引
    raw_df['datetime'] = pd.to_datetime(raw_df['timestamp'], unit='ms')
    raw_df.set_index('datetime', inplace=True)
    
    # ====================== B. 价格相关特征 ======================
    # 价格波动范围
    raw_df['price_range'] = raw_df['high'] - raw_df['low']
    # 涨跌幅（百分比变化）
    raw_df['price_change_pct'] = raw_df['close'].pct_change() * 100
    # 简单移动平均（4小时窗口）
    raw_df['sma_4'] = raw_df['close'].rolling(window=4).mean()
    # 交易量变化
    raw_df['volume_change'] = raw_df['volume'].pct_change()
    
    # ====================== C. 时间特征 ======================
    raw_df['hour'] = raw_df.index.hour
    raw_df['day_of_week'] = raw_df.index.dayofweek  # Monday=0, Sunday=6
    raw_df['month'] = raw_df.index.month
    raw_df['quarter'] = raw_df.index.quarter
    
    # ====================== D. 交易量相关特征 ======================
    raw_df['volume_sma_4'] = raw_df['volume'].rolling(window=4).mean()
    raw_df['volume_std_4'] = raw_df['volume'].rolling(window=4).std()
    
    # ====================== E. 高级时间特征（正弦/余弦编码） ======================
    raw_df['hour_sin'] = np.sin(2 * np.pi * raw_df['hour'] / 24)
    raw_df['hour_cos'] = np.cos(2 * np.pi * raw_df['hour'] / 24)
    raw_df['dayofweek_sin'] = np.sin(2 * np.pi * raw_df['day_of_week'] / 7)
    raw_df['dayofweek_cos'] = np.cos(2 * np.pi * raw_df['day_of_week'] / 7)
    
    # ====================== F. 交易日标识 ======================
    raw_df['is_trading_day'] = np.where(raw_df['day_of_week'].isin([5, 6]), 0, 1)
    
    # ====================== G. 复杂技术指标 ======================
    # MACD指标（12, 26, 9）
    close_price = raw_df['close']
    ema12 = close_price.rolling(window=12).mean()
    ema26 = close_price.rolling(window=26).mean()
    raw_df['macd'] = ema12 - ema26
    raw_df['macd_signal'] = raw_df['macd'].rolling(window=9).mean()
    raw_df['macd_diff'] = raw_df['macd'] - raw_df['macd_signal']
    
    # 动量指标（14期）
    n = 14
    raw_df['momentum'] = raw_df['close'] / raw_df['close'].shift(n) - 1
    
    # ====================== H. 量价交叉特征 ======================
    raw_df['volatility_vol'] = raw_df['price_range'] * raw_df['volume']
    raw_df['return_vol_corr'] = raw_df['price_change_pct'].rolling(window=5).corr(raw_df['volume_change'])
    
    # ====================== I. 数据清洗（与训练时一致） ======================
    # 删除包含NaN的行（由于rolling计算产生）
    processed_df = raw_df.dropna().reset_index()
    
    # ====================== J. 筛选目标特征（按训练时的特征列表） ======================
    # 确保特征顺序和名称与训练时完全一致
    features_df = processed_df[EXPECTED_FEATURES].copy()
    
    return features_df


# ====================== 4. 预测主函数 ======================
def predict_next_volume(api_endpoint: str) -> float:
    """
    预测下一时刻的volume（t到t+60分钟）
    :param api_endpoint: API数据接口地址
    :return: 预测的volume值（标量）
    """
    # 1. 获取原始数据（模拟API调用，需替换为真实实现）
    raw_data = fetch_raw_data_from_api(api_endpoint)
    
    # 2. 特征工程（复现训练时的逻辑）
    features = process_raw_data(raw_data)
    
    # 3. 检查数据有效性（至少包含1条有效特征）
    if features.empty:
        raise ValueError("处理后的数据为空，无法预测")
    
    # 4. 执行预测（模型自动处理标准化+PCA+LightGBM）
    # 取最后一条数据（假设API返回按时间排序，最后一条为最新数据）
    last_features = features.iloc[-1:].reset_index(drop=True)
    prediction = best_model.predict(last_features)
    
    return float(prediction[0])


# ====================== 5. 主程序入口 ======================
if __name__ == "__main__":
    # 替换为真实API地址（示例中使用模拟数据）
    API_ENDPOINT = "https://api.example.com/historical_data"
    
    try:
        # 执行预测
        predicted_volume = predict_next_volume(API_ENDPOINT)
        print(f"预测的下一时刻volume：{predicted_volume:.6f}")
        
        # 可选：将结果写入文件/发送到API/存储到数据库
    except Exception as e:
        print(f"预测失败：{str(e)}")
        raise  # 确保错误被捕获，避免静默失败


# In[4]:


import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.pipeline import Pipeline
import lightgbm as lgb

# ====================== 1. 加载预训练模型与特征列表 ======================
# 加载训练好的Pipeline模型（包含标准化、PCA、LightGBM）
MODEL_PATH = "best_model.pkl"
FEATURES_PATH = "selected_features.pkl"

best_model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "rb") as f:
    EXPECTED_FEATURES = pickle.load(f)


# In[5]:


best_model


# In[6]:


EXPECTED_FEATURES


# In[ ]:




