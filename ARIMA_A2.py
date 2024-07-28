import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 加载数据
file_path = 'A2.xlsx'
data = pd.read_excel(file_path)

# 提取时间序列数据
data['月份'] = pd.to_datetime(data['月份'], format='%Y%m')
data.set_index('月份', inplace=True)
sales_data = data['销量（箱）']
sales_data = sales_data.asfreq('MS')  # 设置时间序列频率为月初

# SARIMA模型
model = SARIMAX(sales_data,
                order=(1, 2, 2),
                seasonal_order=(2, 0, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)
model_fit = model.fit(disp=False, maxiter=1000)  # 增加最大迭代次数

# 预测未来20步
forecast = model_fit.get_forecast(steps=20)
forecast_index = pd.date_range(start=sales_data.index[-1] + pd.DateOffset(months=1), periods=20, freq='MS')
forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)

# 绘制结果
plt.figure(figsize=(12, 6))
plt.plot(sales_data, label='历史数据')
plt.plot(forecast_series, label='预测', color='red')
plt.title('销量（箱）20步预测')
plt.xlabel('时间')
plt.ylabel('销量（箱）')
plt.legend()
plt.grid(True)
plt.show()

# 输出预测数据
forecast_df = forecast_series.reset_index()
forecast_df.columns = ['月份', '预测销量（箱）']
forecast_df.to_excel('a1f2.xlsx', index=False)  # 将预测结果保存为Excel文件
print("预测结果已保存为 'a1f2.xlsx'")