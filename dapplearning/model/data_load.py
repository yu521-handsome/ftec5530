# model/dataset.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from feature_engineering import process_data
import os

class BitcoinDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, window_size=6):
        self.scaler = MinMaxScaler()
        features = self.scaler.fit_transform(df[feature_cols])
        targets = df[target_col].values
        self.X, self.y = [], []
        for i in range(window_size, len(df)):
            self.X.append(features[i-window_size:i])
            self.y.append(targets[i])
        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data(file_path, sequence_length=3):
    """直接调用特征工程处理数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到数据文件: {file_path}")
        
    print(f"正在加载数据: {file_path}")
    X, y, scaler = process_data(file_path, n_in=sequence_length, n_out=2)
    print(f"数据加载完成: X shape={X.shape}, y shape={y.shape}")
    
    return X, y, scaler
