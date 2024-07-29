import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 加载数据
file_path = 'C:/Users/34353/Desktop/2024年钉钉杯A题/A1.xlsx'
data = pd.read_excel(file_path)

# 提取时间序列数据
data['月份'] = pd.to_datetime(data['月份'], format='%Y%m')
data.set_index('月份', inplace=True)
sales_data = data['销量（箱）']
sales_data = sales_data.asfreq('MS')  # 设置时间序列频率为月初

# 检查数据的缺失值
if sales_data.isnull().sum() > 0:
    sales_data = sales_data.ffill().bfill()  # 前向填充和后向填充

# 平稳性检测
result = adfuller(sales_data)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# 如果p-value大于0.05，表示数据非平稳，需要进行差分
d = 0
while result[1] > 0.05:
    sales_data = sales_data.diff().dropna()
    result = adfuller(sales_data)
    d += 1

print(f'经过 {d} 阶差分后，时间序列数据平稳。')

# 建立SARIMAX模型
model = SARIMAX(sales_data,
                order=(1, d, 2),
                seasonal_order=(2, 0, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)
model_fit = model.fit(disp=False, maxiter=1000)  # 增加最大迭代次数

# 模型诊断
model_fit.plot_diagnostics(figsize=(12, 8))
plt.show()

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

# 输出预测结果
forecast_df = forecast_series.reset_index()
forecast_df.columns = ['月份', '预测销量（箱）']
forecast_df.to_excel('a1f2.xlsx', index=False)  # 将预测结果保存为Excel文件
print("预测结果已保存为 'a1f21.xlsx'")
