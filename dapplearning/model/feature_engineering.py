import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import MinMaxScaler

def add_technical_indicators(df):
    """添加技术指标"""
    # 确保数据按时间排序
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 基础特征
    df['price_range'] = df['high'] - df['low']
    df['price_change_pct'] = df['close'].pct_change() * 100
    df['volume_change'] = df['volume'].pct_change()
    
    # 移动平均
    df['sma_4'] = df['close'].rolling(window=4).mean()
    df['volume_sma_4'] = df['volume'].rolling(window=4).mean()
    df['volume_std_4'] = df['volume'].rolling(window=4).std()
    
    # 时间特征
    df['hour'] = pd.to_datetime(df['timestamp'], unit='ms').dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp'], unit='ms').dt.dayofweek
    
    # 周期性编码
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # MACD
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # RSI
    rsi = RSIIndicator(close=df['close'])
    df['rsi'] = rsi.rsi()
    
    # 布林带
    bb = BollingerBands(close=df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    
    # 删除NaN值
    df = df.dropna()
    
    return df

def load_and_process_data(file_path):
    """加载并处理数据"""
    df = pd.read_csv(file_path)
    df = add_technical_indicators(df)
    
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'price_range', 'price_change_pct', 'sma_4',
        'volume_change', 'volume_sma_4', 'volume_std_4',
        'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
        'macd', 'macd_signal', 'macd_diff', 'rsi',
        'bb_high', 'bb_low'
    ]
    
    return df[feature_columns]

def series_to_supervised(data, features, n_in=3, n_out=2, dropnan=True):
    """将时间序列转换为监督学习格式"""
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

def process_data(file_path, n_in=3, n_out=1):
    """处理数据并生成特征"""
    # 读取数据
    df = pd.read_csv(file_path)
    
    # 转换时间戳为datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    
    # 基础特征
    features = ['open', 'high', 'low', 'close', 'volume']
    
    # 添加时间特征
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek  # 添加星期几特征
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)  # 添加是否周末特征
    
    # 交易量特征
    df['volume_sma_4'] = df['volume'].rolling(window=4).mean()  # 4小时移动平均
    df['volume_std_4'] = df['volume'].rolling(window=4).std()   # 4小时标准差
    df['volume_momentum'] = df['volume'] / df['volume'].rolling(24).mean()  # 相对于24小时平均的动量
    
    # 价格和波动性特征
    df['price_change'] = df['close'].pct_change()
    df['volatility'] = df['price_change'].rolling(24).std()  # 24小时波动率
    
    # 更新特征列表
    features = features + [
        'hour', 'day_of_week', 'is_weekend',
        'volume_sma_4', 'volume_std_4', 'volume_momentum',
        'price_change', 'volatility'
    ]
    
    # 处理缺失值
    df = df.ffill()
    df = df.dropna()
    
    # 标准化数据
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features])
    
    # 创建序列数据
    X, y = [], []
    for i in range(len(scaled_data) - n_in - n_out + 1):
        X.append(scaled_data[i:(i + n_in)])
        y.append(scaled_data[i + n_in, 4])  # 预测volume（第5列）
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"特征列表: {features}")
    
    return X, y, scaler

if __name__ == "__main__":
    # 测试数据处理
    data_path = '../../DATA/new/binance/BTC_USDT_1h.csv'  # 使用DATA目录的数据
    X_hourly, y_hourly, scaler_hourly = process_data(data_path)
    print("\n处理结果:")
    print(f"X shape: {X_hourly.shape}")
    print(f"y shape: {y_hourly.shape}") 