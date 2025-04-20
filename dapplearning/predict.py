import torch
import pandas as pd
import numpy as np
from model.lstm_model import LSTMModel
from model.feature_engineering import process_data
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def predict_next_hour_volume(model, data, scaler):
    """预测下一个小时的交易量"""
    model.eval()
    with torch.no_grad():
        # 转换输入数据为张量
        X = torch.FloatTensor(data).unsqueeze(0)  # 添加batch维度
        if torch.cuda.is_available():
            X = X.cuda()
            model = model.cuda()
        
        # 预测
        pred = model(X)
        pred = pred.cpu().numpy()
        
        # 反标准化得到实际交易量
        pred_full = np.zeros((len(pred), 13))
        pred_full[:, 4] = pred.reshape(-1)  # volume是第5个特征
        pred_volume = scaler.inverse_transform(pred_full)[:, 4]
        
        return pred_volume[0]

def evaluate_predictions(predictions, actual_values, timestamps):
    """评估预测结果"""
    log_errors = np.abs(np.log(predictions / actual_values))
    mean_error = np.mean(log_errors)
    
    # 创建结果DataFrame
    results = pd.DataFrame({
        'Timestamp': timestamps,
        'Actual_Volume': actual_values,
        'Predicted_Volume': predictions,
        'Log_Error': log_errors
    })
    
    # 打印评估指标
    print("\n预测评估结果:")
    print(f"平均对数误差: {mean_error:.4f}")
    print(f"最大对数误差: {log_errors.max():.4f}")
    print(f"最小对数误差: {log_errors.min():.4f}")
    
    # 保存结果
    results.to_csv('prediction_results.csv', index=False)
    
    # 绘制预测vs实际值图表
    plt.figure(figsize=(15, 6))
    plt.plot(timestamps[-100:], actual_values[-100:], label='Actual Volume')
    plt.plot(timestamps[-100:], predictions[-100:], label='Predicted Volume')
    plt.title('Last 100 Hours: Predicted vs Actual Trading Volume')
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('prediction_plot.png')
    plt.close()
    
    return mean_error, results

def main():
    # 1. 加载模型
    model = LSTMModel(
        input_size=13,  # 特征数量
        hidden_size=64,
        num_layers=2,
        output_size=1,
        dropout=0.2
    )
    model.load_state_dict(torch.load('model/best_model.pth'))
    model.eval()
    
    # 2. 加载测试数据
    test_data_path = '../DATA/new/binance/BTC_USDT_1h.csv'
    X_test, y_test, scaler = process_data(test_data_path, n_in=3, n_out=1)
    
    # 3. 读取时间戳
    df = pd.read_csv(test_data_path)
    timestamps = pd.to_datetime(df['timestamp'], unit='ms')
    timestamps = timestamps[3:]  # 跳过前3个小时（用于特征）
    
    # 4. 进行预测
    predictions = []
    actuals = []
    
    print("\n开始预测...")
    for i in range(len(X_test)):
        # 预测下一个小时的交易量
        pred_volume = predict_next_hour_volume(model, X_test[i], scaler)
        
        # 修改这部分代码
        actual_full = np.zeros((1, 13))
        actual_full[0, 4] = y_test[i]  # volume是第5个特征
        actual_volume = scaler.inverse_transform(actual_full)[0, 4]
        
        predictions.append(pred_volume)
        actuals.append(actual_volume)
        
        if i % 100 == 0:
            print(f"已完成 {i}/{len(X_test)} 个预测")
    
    # 5. 评估结果
    error, results = evaluate_predictions(
        np.array(predictions),
        np.array(actuals),
        timestamps[:len(predictions)]
    )
    
    # 6. 预测未来60分钟
    last_data = X_test[-1]
    future_pred = predict_next_hour_volume(model, last_data, scaler)
    next_hour = timestamps.iloc[-1] + timedelta(hours=1)
    
    print("\n未来60分钟预测:")
    print(f"时间: {next_hour}")
    print(f"预测交易量: {future_pred:.2f}")
    
    return error, results

if __name__ == "__main__":
    error, results = main()
    print(f"\n最终预测误差: {error:.4f}")
