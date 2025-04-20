# model/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from data_load import load_data
from lstm_model import LSTMModel
import numpy as np
import matplotlib.pyplot as plt

def calculate_volume_error(y_true, y_pred, scaler):
    """计算交易量预测的对数误差"""
    # 确保输入是正确的形状
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    # 创建完整的特征数组（用0填充其他特征）
    y_true_full = np.zeros((len(y_true), 13))
    y_true_full[:, 4] = y_true  # volume 是第5个特征
    
    y_pred_full = np.zeros((len(y_pred), 13))
    y_pred_full[:, 4] = y_pred
    
    # 反标准化
    y_true_volume = scaler.inverse_transform(y_true_full)[:, 4]
    y_pred_volume = scaler.inverse_transform(y_pred_full)[:, 4]
    
    # 计算对数误差
    log_error = np.abs(np.log(y_pred_volume / y_true_volume))
    mean_error = log_error.mean()
    
    print(f"\n预测评估:")
    print(f"平均对数误差: {mean_error:.4f}")
    print(f"最大对数误差: {log_error.max():.4f}")
    print(f"最小对数误差: {log_error.min():.4f}")
    
    return mean_error

def train_model(model_params, train_params):
    """训练模型"""
    # 加载数据
    X, y, scaler = load_data(train_params['data_path'], 
                            sequence_length=train_params['sequence_length'])
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)  # 添加维度变成 (batch_size, 1)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val).unsqueeze(1)      # 添加维度变成 (batch_size, 1)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], 
                            shuffle=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=train_params['batch_size'])
    
    # 初始化模型
    model = LSTMModel(**model_params)
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['learning_rate'])
    
    # 训练循环
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("\n开始训练...")
    for epoch in range(train_params['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            if torch.cuda.is_available():
                batch_X = batch_X.cuda()
                batch_y = batch_y.cuda()
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                if torch.cuda.is_available():
                    batch_X = batch_X.cuda()
                    batch_y = batch_y.cuda()
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                
                # 收集预测结果
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        # 计算损失
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 计算交易量预测误差
        volume_error = calculate_volume_error(
            np.array(val_targets),
            np.array(val_predictions),
            scaler
        )
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), train_params['model_save_path'])
            print(f"保存最佳模型，验证损失: {val_loss:.4f}")
        
        print(f'Epoch [{epoch+1}/{train_params["epochs"]}], '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Volume Error: {volume_error:.4f}')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()
    
    return model, scaler

if __name__ == "__main__":
    model_params = {
        'input_size': 13,  # 13个特征
        'hidden_size': 64,
        'num_layers': 2,
        'output_size': 1,  # 预测一个时间点的交易量
        'dropout': 0.2
    }
    
    train_params = {
        'data_path': '../../DATA/new/binance/BTC_USDT_1h.csv',
        'sequence_length': 3,  # 使用过去3个小时的数据
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 100,
        'model_save_path': './best_model.pth'
    }
    
    model, scaler = train_model(model_params, train_params)
