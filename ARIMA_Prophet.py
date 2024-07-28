import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
import numpy as np

# 设置中文字体以解决字体缺失问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
file_path_a3 = 'A3.xlsx'
file_path_a4 = 'A4.xlsx'
data_a3 = pd.read_excel(file_path_a3)
data_a4 = pd.read_excel(file_path_a4)


def preprocess_data(data):
    data['月份'] = pd.to_datetime(data['月份'], format='%Y%m')
    data.set_index('月份', inplace=True)
    data = data.asfreq('MS')  # 使用 MS 代替 M 设置月度开始频率
    data = data.ffill()  # 使用推荐的ffill方法填充缺失值
    return data


data_a3 = preprocess_data(data_a3)
data_a4 = preprocess_data(data_a4)


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
def prophet_forecast(data, column, steps=12):
    df = data.reset_index().rename(columns={'月份': 'ds', column: 'y'})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=steps, freq='MS')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].set_index('ds')['yhat']


# 评价模型
def evaluate_forecast(actual, forecast):
    mse = mean_squared_error(actual, forecast)
    mae = mean_absolute_error(actual, forecast)
    return mse, mae


# 绘制预测结果
def plot_forecast(data, column, arima_forecast, prophet_forecast):
    plt.figure(figsize=(14, 8))
    plt.plot(data.index, data[column], label='实际值', color='blue')
    plt.plot(arima_forecast.index, arima_forecast, label='ARIMA预测值', color='red', linestyle='--')
    plt.plot(prophet_forecast.index, prophet_forecast, label='Prophet预测值', color='green', linestyle='--')
    plt.title(f'{column} 预测')
    plt.xlabel('月份')
    plt.ylabel(column)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


# 对单个品牌的销售金额进行预测并输出结果
def forecast_brand(data, brand):
    column = '金额（元）'
    steps = 12

    # ARIMA参数选择
    best_pdq, best_arima_model = find_best_arima(data, column)

    # ARIMA预测
    arima_forecast_values = best_arima_model.forecast(steps=steps)
    arima_forecast_index = pd.date_range(start=data.index[-1], periods=steps + 1, freq='MS')[1:]
    arima_forecast_values = pd.Series(arima_forecast_values, index=arima_forecast_index)

    # Prophet预测
    prophet_forecast_values = prophet_forecast(data, column, steps)

    # 调整预测结果长度一致
    min_length = min(len(arima_forecast_values), len(prophet_forecast_values))
    arima_forecast_values = arima_forecast_values.iloc[:min_length]
    prophet_forecast_values = prophet_forecast_values.iloc[:min_length]

    # 实际值（这里假设实际值存在于数据中，调整索引以匹配预测）
    actual_values = data[column].iloc[-steps:]

    # 评价模型
    arima_mse, arima_mae = evaluate_forecast(actual_values, arima_forecast_values[:len(actual_values)])
    prophet_mse, prophet_mae = evaluate_forecast(actual_values, prophet_forecast_values[:len(actual_values)])

    # 打印评价结果
    print(f'{brand} 销售金额预测:')
    print(f'最佳ARIMA参数: {best_pdq}')
    print(f'ARIMA MSE: {arima_mse}, MAE: {arima_mae}')
    print(f'Prophet MSE: {prophet_mse}, MAE: {prophet_mae}')

    # 绘制预测结果
    plot_forecast(data, column, arima_forecast_values, prophet_forecast_values)

    # 预测结果保存为表格
    forecast_df = pd.DataFrame({
        '日期': arima_forecast_values.index,
        'ARIMA预测': arima_forecast_values.values,
        'Prophet预测': prophet_forecast_values.values
    })
    forecast_df.set_index('日期', inplace=True)
    forecast_df.to_excel(f'{brand}_预测结果.xlsx')

    # 显示模型评价结果
    model_evaluation_results = pd.DataFrame({
        'Model': ['ARIMA', 'Prophet'],
        'MSE': [arima_mse, prophet_mse],
        'MAE': [arima_mae, prophet_mae]
    })
    print(model_evaluation_results)


# 预测A3和A4品牌的销售金额
forecast_brand(data_a3, 'A3')
forecast_brand(data_a4, 'A4')