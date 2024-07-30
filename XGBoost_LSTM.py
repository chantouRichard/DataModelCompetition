import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor

# 设置中文字体以解决字体缺失问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
file_path_A3 = 'A3.xlsx'
file_path_A4 = 'A4.xlsx'
data_A3 = pd.read_excel(file_path_A3)
data_A4 = pd.read_excel(file_path_A4)

def preprocess_data(data):
    """
    预处理数据，将月份转换为日期格式，并设置为索引，填充缺失值。
    """
    data['月份'] = pd.to_datetime(data['月份'], format='%Y%m')
    data.set_index('月份', inplace=True)
    data = data.asfreq('MS')  # 使用 MS 代替 M 设置月度开始频率
    data = data.ffill()  # 使用推荐的ffill方法填充缺失值
    return data

data_A3 = preprocess_data(data_A3)
data_A4 = preprocess_data(data_A4)

def create_supervised_data(data, n_in=1, n_out=1):
    """
    准备监督学习的数据集。
    """
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f'{data.name}(t-{i})']
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [f'{data.name}(t)']
        else:
            names += [f'{data.name}(t+{i})']
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)
    return agg

# 准备 A3 数据
n_lags = 12
supervised_A3 = create_supervised_data(data_A3['金额（元）'], n_in=n_lags, n_out=1)
X_A3, y_A3 = supervised_A3.iloc[:, :-1], supervised_A3.iloc[:, -1]

# 准备 A4 数据
supervised_A4 = create_supervised_data(data_A4['金额（元）'], n_in=n_lags, n_out=1)
X_A4, y_A4 = supervised_A4.iloc[:, :-1], supervised_A4.iloc[:, -1]

# 标准化数据
scaler_X_A3 = StandardScaler()
X_A3_scaled = scaler_X_A3.fit_transform(X_A3)
scaler_y_A3 = StandardScaler()
y_A3_scaled = scaler_y_A3.fit_transform(y_A3.values.reshape(-1, 1)).flatten()

scaler_X_A4 = StandardScaler()
X_A4_scaled = scaler_X_A4.fit_transform(X_A4)
scaler_y_A4 = StandardScaler()
y_A4_scaled = scaler_y_A4.fit_transform(y_A4.values.reshape(-1, 1)).flatten()

def reshape_for_lstm(X):
    """
    将输入数据重塑为 LSTM 的输入格式。
    """
    return X.reshape((X.shape[0], X.shape[1], 1))

# 重塑数据以适应 LSTM
X_A3_lstm = reshape_for_lstm(X_A3_scaled)
X_A4_lstm = reshape_for_lstm(X_A4_scaled)

def build_lstm_model(input_shape):
    """
    构建 LSTM 模型。
    """
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# 定义并拟合 LSTM 模型
lstm_model_A3 = build_lstm_model((n_lags, 1))
lstm_model_A3.fit(X_A3_lstm, y_A3_scaled, epochs=200, batch_size=32, verbose=0)

lstm_model_A4 = build_lstm_model((n_lags, 1))
lstm_model_A4.fit(X_A4_lstm, y_A4_scaled, epochs=200, batch_size=32, verbose=0)

# 预测 A3 和 A4 的 LSTM 结果
lstm_forecast_A3_scaled = lstm_model_A3.predict(X_A3_lstm[-12:])
lstm_forecast_A3 = scaler_y_A3.inverse_transform(lstm_forecast_A3_scaled).flatten()

lstm_forecast_A4_scaled = lstm_model_A4.predict(X_A4_lstm[-12:])
lstm_forecast_A4 = scaler_y_A4.inverse_transform(lstm_forecast_A4_scaled).flatten()

# XGBoost 模型预测
xgb_model_A3 = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model_A3.fit(X_A3_scaled, y_A3)
xgb_forecast_A3 = xgb_model_A3.predict(X_A3_scaled[-12:])

xgb_model_A4 = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model_A4.fit(X_A4_scaled, y_A4)
xgb_forecast_A4 = xgb_model_A4.predict(X_A4_scaled[-12:])

def plot_forecasts(data, xgb_forecast, lstm_forecast, title):
    """
    绘制预测结果与实际数据的对比图。
    """
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['金额（元）'], label='历史数据')
    plt.plot(data.index[-12:], xgb_forecast, label='XGBoost预测', color='red')
    plt.plot(data.index[-12:], lstm_forecast, label='LSTM预测', color='green')
    plt.xlabel('日期')
    plt.ylabel('销售金额（元）')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# 绘制 A3 和 A4 的预测结果
plot_forecasts(data_A3, xgb_forecast_A3, lstm_forecast_A3, 'A3品牌销售金额预测 - XGBoost和LSTM模型')
plot_forecasts(data_A4, xgb_forecast_A4, lstm_forecast_A4, 'A4品牌销售金额预测 - XGBoost和LSTM模型')

# 创建预测值的 DataFrame
forecast_df_A3 = pd.DataFrame({
    '日期': data_A3.index[-12:],
    'XGBoost预测': xgb_forecast_A3,
    'LSTM预测': lstm_forecast_A3
})

forecast_df_A4 = pd.DataFrame({
    '日期': data_A4.index[-12:],
    'XGBoost预测': xgb_forecast_A4,
    'LSTM预测': lstm_forecast_A4
})

# 打印并保存预测结果
print("A3品牌销售金额预测结果：")
print(forecast_df_A3)
print("\nA4品牌销售金额预测结果：")
print(forecast_df_A4)

forecast_df_A3.to_excel('forecast_A3.xlsx', index=False)
forecast_df_A4.to_excel('forecast_A4.xlsx', index=False)
print("预测结果已保存到 'forecast_A3.xlsx' 和 'forecast_A4.xlsx'")
