import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号 '-' 显示为方块的问题

# 读取数据
file_path = "A题.xlsx"
sheet_names = ["A1", "A2", "A3", "A4", "A5"]
dataframes = [pd.read_excel(file_path, sheet_name=sheet) for sheet in sheet_names]
data = pd.concat(dataframes, ignore_index=True)

# 数据预处理
data['月份'] = pd.to_datetime(data['月份'], format='%Y%m')  # 假设月份为YYYYMM格式
data = data.sort_values(by='月份')

# 按月份计算销售总额
monthly_sales = data.groupby('月份')['金额（元）'].sum()

# 将 Series 转换为 DataFrame 并设置时间索引和频率
monthly_sales_df = monthly_sales.to_frame()
monthly_sales_df.index = pd.DatetimeIndex(monthly_sales_df.index)
monthly_sales_df = monthly_sales_df.asfreq('M')  # 设置频率为月度

# 处理缺失值
monthly_sales_df = monthly_sales_df.fillna(method='ffill')  # 前向填充缺失值

# 计算价格与销量的相关性
data['价格'] = data['金额（元）'] / data['销量（箱）']  # 计算价格
price_sales = data[['价格', '销量（箱）']].dropna()

# 线性回归模型
X = sm.add_constant(price_sales['价格'])  # 添加常数项
y = price_sales['销量（箱）']
model = sm.OLS(y, X).fit()
print(model.summary())

# 可视化结果
plt.scatter(price_sales['价格'], price_sales['销量（箱）'], color='blue')
plt.plot(price_sales['价格'], model.predict(X), color='red')
plt.xlabel('价格')
plt.ylabel('销量（箱）')
plt.title('价格对销量的影响')
plt.show()


from scipy import stats

# 提取年份
data['年份'] = data['月份'].dt.year

# 按年份计算销售总额
annual_sales = data.groupby('年份')['金额（元）'].sum().reset_index()

# 方差分析
years = data['年份'].unique()
sales_by_year = [data[data['年份'] == year]['金额（元）'] for year in years]
f_value, p_value = stats.f_oneway(*sales_by_year)
print(f'ANOVA F-value: {f_value}, p-value: {p_value}')

# 可视化结果
plt.boxplot([sales_by_year[i] for i in range(len(years))], labels=years)
plt.xlabel('年份')
plt.ylabel('销售额（元）')
plt.title('不同年份销售额的分布')
plt.show()


from statsmodels.tsa.seasonal import seasonal_decompose

# 按月份计算销售总额
monthly_sales = data.groupby('月份')['金额（元）'].sum()

# 将 Series 转换为 DataFrame 并设置时间索引和频率
monthly_sales_df = monthly_sales.to_frame()
monthly_sales_df.index = pd.DatetimeIndex(monthly_sales_df.index)
monthly_sales_df = monthly_sales_df.asfreq('M')  # 设置频率为月度

# 处理缺失值
monthly_sales_df = monthly_sales_df.fillna(method='ffill')  # 前向填充缺失值

# 时间序列分解
decomposition = seasonal_decompose(monthly_sales_df['金额（元）'], model='additive')

# 可视化结果
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(311)
ax1.plot(decomposition.observed)
ax1.set_title('Observed')
ax2 = fig.add_subplot(312)
ax2.plot(decomposition.trend)
ax2.set_title('Trend')
ax3 = fig.add_subplot(313)
ax3.plot(decomposition.seasonal)
ax3.set_title('Seasonal')
plt.tight_layout()
plt.show()