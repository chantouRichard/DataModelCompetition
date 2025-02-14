import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
    """
    预处理数据，将月份转换为日期格式，并设置为索引，填充缺失值。
    """
    data['月份'] = pd.to_datetime(data['月份'], format='%Y%m')
    data.set_index('月份', inplace=True)
    data = data.asfreq('MS')  # 使用 MS 代替 M 设置月度开始频率
    data = data.ffill()  # 使用推荐的ffill方法填充缺失值
    return data

data_a3 = preprocess_data(data_a3)
data_a4 = preprocess_data(data_a4)

def find_best_sarima(data, column):
    """
    选择最佳的SARIMA模型参数。
    """
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
    best_aic = np.inf
    best_param = None
    best_seasonal_param = None
    best_model = None
    for param in pdq:
        for seasonal_param in seasonal_pdq:
            try:
                model = SARIMAX(data[column], order=param, seasonal_order=seasonal_param)
                model_fit = model.fit(disp=False)
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_param = param
                    best_seasonal_param = seasonal_param
                    best_model = model_fit
            except Exception as e:
                print(f"SARIMA 参数 {param} 季节性参数 {seasonal_param} 失败: {e}")
                continue
    return best_param, best_seasonal_param, best_model

def prophet_forecast(data, column, steps=12):
    """
    使用 Prophet 模型进行预测。
    """
    df = data.reset_index().rename(columns={'月份': 'ds', column: 'y'})
    model = Prophet(weekly_seasonality=True, daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=steps, freq='MS')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].set_index('ds')['yhat']

def evaluate_forecast(actual, forecast):
    """
    计算预测结果的评估指标。
    """
    mse = mean_squared_error(actual, forecast)
    mae = mean_absolute_error(actual, forecast)
    return mse, mae

def plot_forecast(data, column, sarima_forecast, prophet_forecast):
    """
    绘制预测结果与实际数据的对比图。
    """
    plt.figure(figsize=(14, 8))
    plt.plot(data.index, data[column], label='实际值', color='blue')
    plt.plot(sarima_forecast.index, sarima_forecast, label='SARIMA预测值', color='red', linestyle='--')
    plt.plot(prophet_forecast.index, prophet_forecast, label='Prophet预测值', color='green', linestyle='--')
    plt.title(f'{column} 预测')
    plt.xlabel('月份')
    plt.ylabel(column)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def forecast_brand(data, brand):
    """
    对单个品牌的销售金额进行预测并输出结果。
    """
    column = '金额（元）'
    steps = 12

    # SARIMA参数选择
    best_param, best_seasonal_param, best_sarima_model = find_best_sarima(data, column)
    if best_sarima_model is None:
        print(f"{brand} 的SARIMA模型训练失败。")
        return

    # SARIMA预测
    try:
        sarima_forecast_values = best_sarima_model.forecast(steps=steps)
        sarima_forecast_index = pd.date_range(start=data.index[-1], periods=steps + 1, freq='MS')[1:]
        sarima_forecast_values = pd.Series(sarima_forecast_values, index=sarima_forecast_index)
    except Exception as e:
        print(f"{brand} 的SARIMA模型预测失败: {e}")
        return

    # Prophet预测
    try:
        prophet_forecast_values = prophet_forecast(data, column, steps)
    except Exception as e:
        print(f"{brand} 的Prophet模型预测失败: {e}")
        return

    # 实际值（这里假设实际值存在于数据中，调整索引以匹配预测）
    min_length = min(len(sarima_forecast_values), len(prophet_forecast_values))
    actual_values = data[column].iloc[-min_length:]

    # 调整预测结果长度一致
    sarima_forecast_values = sarima_forecast_values.iloc[:min_length]
    prophet_forecast_values = prophet_forecast_values.iloc[:min_length]

    # 评价模型
    sarima_mse, sarima_mae = evaluate_forecast(actual_values, sarima_forecast_values)
    prophet_mse, prophet_mae = evaluate_forecast(actual_values, prophet_forecast_values)

    # 打印评价结果
    print(f'{brand} 销售金额预测:')
    print(f'最佳SARIMA参数: {best_param}, 季节性参数: {best_seasonal_param}')
    print(f'SARIMA MSE: {sarima_mse}, MAE: {sarima_mae}')
    print(f'Prophet MSE: {prophet_mse}, MAE: {prophet_mae}')

    # 绘制预测结果
    plot_forecast(data, column, sarima_forecast_values, prophet_forecast_values)

    # 预测结果保存为表格
    forecast_df = pd.DataFrame({
        '日期': sarima_forecast_values.index,
        'SARIMA预测': sarima_forecast_values.values,
        'Prophet预测': prophet_forecast_values.values
    })
    forecast_df.set_index('日期', inplace=True)
    forecast_df.to_excel(f'{brand}_预测结果.xlsx')

    # 显示模型评价结果
    model_evaluation_results = pd.DataFrame({
        'Model': ['SARIMA', 'Prophet'],
        'MSE': [sarima_mse, prophet_mse],
        'MAE': [sarima_mae, prophet_mae]
    })
    print(model_evaluation_results)

# 预测A3和A4品牌的销售金额
forecast_brand(data_a3, 'A3')
forecast_brand(data_a4, 'A4')
