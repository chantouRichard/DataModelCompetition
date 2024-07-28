import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# Load the datasets
file_path_A3 = 'A3.xlsx'
file_path_A4 = 'A4.xlsx'
data_A3 = pd.read_excel(file_path_A3)
data_A4 = pd.read_excel(file_path_A4)


# Function to prepare data for supervised learning
def create_supervised_data(data, n_in=1, n_out=1):
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(f'{data.name}(t-{i})')]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(f'{data.name}(t)')]
        else:
            names += [(f'{data.name}(t+{i})')]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)
    return agg


# Prepare data for A3
n_lags = 12
supervised_A3 = create_supervised_data(data_A3['金额（元）'], n_in=n_lags, n_out=1)
X_A3, y_A3 = supervised_A3.iloc[:, :-1], supervised_A3.iloc[:, -1]

# Prepare data for A4
supervised_A4 = create_supervised_data(data_A4['金额（元）'], n_in=n_lags, n_out=1)
X_A4, y_A4 = supervised_A4.iloc[:, :-1], supervised_A4.iloc[:, -1]

# Standardize the data
scaler_A3 = StandardScaler()
X_A3_scaled = scaler_A3.fit_transform(X_A3)
y_A3_scaled = scaler_A3.fit_transform(y_A3.values.reshape(-1, 1)).flatten()

scaler_A4 = StandardScaler()
X_A4_scaled = scaler_A4.fit_transform(X_A4)
y_A4_scaled = scaler_A4.fit_transform(y_A4.values.reshape(-1, 1)).flatten()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Function to reshape input for LSTM
def reshape_for_lstm(X):
    return X.reshape((X.shape[0], X.shape[1], 1))


# Reshape data for LSTM
X_A3_lstm = reshape_for_lstm(X_A3_scaled)
X_A4_lstm = reshape_for_lstm(X_A4_scaled)

# Define and fit LSTM model for A3
lstm_model_A3 = Sequential()
lstm_model_A3.add(LSTM(50, activation='relu', input_shape=(n_lags, 1)))
lstm_model_A3.add(Dense(1))
lstm_model_A3.compile(optimizer='adam', loss='mse')
lstm_model_A3.fit(X_A3_lstm, y_A3_scaled, epochs=200, batch_size=32, verbose=0)

# Predict with LSTM for A3
lstm_forecast_A3_scaled = lstm_model_A3.predict(X_A3_lstm[-12:])
lstm_forecast_A3 = scaler_A3.inverse_transform(lstm_forecast_A3_scaled).flatten()

# Define and fit LSTM model for A4
lstm_model_A4 = Sequential()
lstm_model_A4.add(LSTM(50, activation='relu', input_shape=(n_lags, 1)))
lstm_model_A4.add(Dense(1))
lstm_model_A4.compile(optimizer='adam', loss='mse')
lstm_model_A4.fit(X_A4_lstm, y_A4_scaled, epochs=200, batch_size=32, verbose=0)

# Predict with LSTM for A4
lstm_forecast_A4_scaled = lstm_model_A4.predict(X_A4_lstm[-12:])
lstm_forecast_A4 = scaler_A4.inverse_transform(lstm_forecast_A4_scaled).flatten()

from xgboost import XGBRegressor

# XGBoost Model for A3
xgb_model_A3 = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model_A3.fit(X_A3_scaled, y_A3)
xgb_forecast_A3 = xgb_model_A3.predict(X_A3_scaled[-12:])

# XGBoost Model for A4
xgb_model_A4 = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model_A4.fit(X_A4_scaled, y_A4)
xgb_forecast_A4 = xgb_model_A4.predict(X_A4_scaled[-12:])

import matplotlib.pyplot as plt


# Function to plot the results
def plot_forecasts(data, xgb_forecast, lstm_forecast, title):
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


# Plot results for A3
plot_forecasts(data_A3, xgb_forecast_A3, lstm_forecast_A3, 'A3品牌销售金额预测 - XGBoost和LSTM模型')

# Plot results for A4
plot_forecasts(data_A4, xgb_forecast_A4, lstm_forecast_A4, 'A4品牌销售金额预测 - XGBoost和LSTM模型')
# Create a DataFrame for the forecast values
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

# Display the results
print("A3品牌销售金额预测结果：")
print(forecast_df_A3)
print("\nA4品牌销售金额预测结果：")
print(forecast_df_A4)
forecast_df_A3.to_excel('forecast_A3.xlsx', index=False)
forecast_df_A4.to_excel('forecast_A4.xlsx', index=False)