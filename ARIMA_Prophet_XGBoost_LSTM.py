import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import xgboost as xgb

# 设置中文字体以解决字体缺失问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
file_path = 'A5.xlsx'
data = pd.read_excel(file_path)


def preprocess_data(data):
    data['月份'] = pd.to_datetime(data['月份'], format='%Y%m')
    data.set_index('月份', inplace=True)
    data = data.asfreq('MS')
    data = data.ffill()
    return data


data = preprocess_data(data)


# 参数遍历选择
def find_best_arima(data, column):
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    best_aic = np.inf
    best_pdq = None
    best_model = None
    for param in pdq:
        try:
            model = ARIMA(data[column], order=param)
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_pdq = param
                best_model = model_fit
        except:
            continue
    return best_pdq, best_model


# Prophet模型预测
def prophet_forecast(data, column, steps=10):
    df = data.reset_index().rename(columns={'月份': 'ds', column: 'y'})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=steps, freq='MS')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].set_index('ds')['yhat']


# LSTM模型预测
def lstm_forecast(data, column, steps=10):
    series = data[column].values
    n_steps = 3
    X, y = [], []
    for i in range(len(series)):
        end_ix = i + n_steps
        if end_ix > len(series) - 1:
            break
        seq_x, seq_y = series[i:end_ix], series[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)
    x_input = series[-n_steps:]
    temp_input = list(x_input)
    lst_output = []
    for i in range(steps):
        x_input = np.array(temp_input[-n_steps:])
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])
    forecast_index = pd.date_range(start=data.index[-1], periods=steps + 1, freq='MS')[1:]
    lst_output = pd.Series(lst_output, index=forecast_index)
    return lst_output


# XGBoost模型预测
def xgboost_forecast(data, column, steps=10):
    series = data[column].values
    n_steps = 3
    X, y = [], []
    for i in range(len(series)):
        end_ix = i + n_steps
        if end_ix > len(series) - 1:
            break
        seq_x, seq_y = series[i:end_ix], series[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    X, y = np.array(X), np.array(y)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y)
    x_input = series[-n_steps:]
    temp_input = list(x_input)
    xgb_output = []
    for i in range(steps):
        x_input = np.array(temp_input[-n_steps:])
        yhat = model.predict(x_input.reshape(1, -1))
        temp_input.append(yhat[0])
        xgb_output.append(yhat[0])
    forecast_index = pd.date_range(start=data.index[-1], periods=steps + 1, freq='MS')[1:]
    xgb_output = pd.Series(xgb_output, index=forecast_index)
    return xgb_output


# 构建集成学习模型（Stacking）
def stacking_forecast(data, column, steps=10):
    # 训练ARIMA模型
    best_pdq, best_arima_model = find_best_arima(data, column)
    arima_forecast_values = best_arima_model.forecast(steps=steps)
    arima_forecast_index = pd.date_range(start=data.index[-1], periods=steps + 1, freq='MS')[1:]
    arima_forecast_values = pd.Series(arima_forecast_values, index=arima_forecast_index)

    # 训练Prophet模型
    prophet_forecast_values = prophet_forecast(data, column, steps)

    # 训练LSTM模型
    lstm_forecast_values = lstm_forecast(data, column, steps)

    # 训练XGBoost模型
    xgboost_forecast_values = xgboost_forecast(data, column, steps)

    # 确保预测结果长度一致
    common_index = arima_forecast_index.intersection(prophet_forecast_values.index).intersection(
        lstm_forecast_values.index).intersection(xgboost_forecast_values.index)
    arima_forecast_values = arima_forecast_values[common_index]
    prophet_forecast_values = prophet_forecast_values[common_index]
    lstm_forecast_values = lstm_forecast_values[common_index]
    xgboost_forecast_values = xgboost_forecast_values[common_index]

    # 创建训练数据
    X_train = pd.DataFrame({
        'ARIMA': arima_forecast_values,
        'Prophet': prophet_forecast_values,
        'LSTM': lstm_forecast_values,
        'XGBoost': xgboost_forecast_values
    })

    # 实际值
    actual_values = data[column].iloc[-steps:]

    # 使用线性回归作为元学习器
    meta_model = LinearRegression()
    meta_model.fit(X_train, actual_values[:len(X_train)])

    # 生成最终预测
    final_forecast = meta_model.predict(X_train)

    return final_forecast, arima_forecast_values, prophet_forecast_values, lstm_forecast_values, xgboost_forecast_values, actual_values


# 预测并评价A5品牌的销量和销售金额
def forecast_brand(data, column):
    steps = 10
    final_forecast, arima_forecast_values, prophet_forecast_values, lstm_forecast_values, xgboost_forecast_values, actual_values = stacking_forecast(
        data, column, steps)

    # 评价模型
    mse = mean_squared_error(actual_values[:len(final_forecast)], final_forecast)
    mae = mean_absolute_error(actual_values[:len(final_forecast)], final_forecast)

    print(f'{column} 销量和销售金额预测:')
    print(f'集成学习模型 MSE: {mse}, MAE: {mae}')

    # 绘制预测结果
    plt.figure(figsize=(14, 8))
    plt.plot(data.index, data[column], label='实际值', color='blue')
    plt.plot(arima_forecast_values.index, arima_forecast_values, label='ARIMA预测值', color='red', linestyle='--')
    plt.plot(prophet_forecast_values.index, prophet_forecast_values, label='Prophet预测值', color='green',
             linestyle='--')
    plt.plot(lstm_forecast_values.index, lstm_forecast_values, label='LSTM预测值', color='orange', linestyle='--')
    plt.plot(xgboost_forecast_values.index, xgboost_forecast_values, label='XGBoost预测值', color='cyan',
             linestyle='--')
    plt.plot(arima_forecast_values.index, final_forecast, label='集成学习预测值', color='purple', linestyle='--')
    plt.title(f'{column} 预测')
    plt.xlabel('月份')
    plt.ylabel(column)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

    # 预测结果保存为表格

    forecast_df = pd.DataFrame({
        '日期': arima_forecast_values.index,
        'ARIMA预测': arima_forecast_values.values,
        'Prophet预测': prophet_forecast_values.values,
        'LSTM预测': lstm_forecast_values.values,
        'XGBoost预测': xgboost_forecast_values.values,
        '集成学习预测': final_forecast
    })
    forecast_df.set_index('日期', inplace=True)
    forecast_df.to_excel(f'{column}_预测结果.xlsx')

    # 显示模型评价结果
    model_evaluation_results = pd.DataFrame({
        'Model': ['ARIMA', 'Prophet', 'LSTM', 'XGBoost', 'Stacking'],
        'MSE': [
            mean_squared_error(actual_values, arima_forecast_values[:len(actual_values)]),
            mean_squared_error(actual_values, prophet_forecast_values[:len(actual_values)]),
            mean_squared_error(actual_values, lstm_forecast_values[:len(actual_values)]),
            mean_squared_error(actual_values, xgboost_forecast_values[:len(actual_values)]),
            mse
        ],
        'MAE': [
            mean_absolute_error(actual_values, arima_forecast_values[:len(actual_values)]),
            mean_absolute_error(actual_values, prophet_forecast_values[:len(actual_values)]),
            mean_absolute_error(actual_values, lstm_forecast_values[:len(actual_values)]),
            mean_absolute_error(actual_values, xgboost_forecast_values[:len(actual_values)]),
            mae
        ]
    })
    print(model_evaluation_results)


# 预测A5品牌的销量和销售金额
forecast_brand(data, '销量（箱）')
forecast_brand(data, '金额（元）')