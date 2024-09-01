import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 一、数据预处理
# 读取Excel文件
excel_file = 'A题.xlsx'
sheets = pd.read_excel(excel_file, sheet_name=None)
data = pd.concat(sheets.values())

# 检查数据缺失值
data.isnull().sum()

# 数据变换：确保数值型数据正确
data['销量（箱）'] = pd.to_numeric(data['销量（箱）'], errors='coerce')
data['金额（元）'] = pd.to_numeric(data['金额（元）'], errors='coerce')

# 处理日期转换
data['月份'] = pd.to_datetime(data['月份'], format='%Y-%m')
data['年份'] = data['月份'].dt.year
data['月'] = data['月份'].dt.month

# 计算月均销量、月均销售额、销售额增长率、价格波动率
data['平均单价'] = data['金额（元）'] / data['销量（箱）']
data['月均销量'] = data.groupby('名称')['销量（箱）'].transform('mean')
data['月均销售额'] = data.groupby('名称')['金额（元）'].transform('mean')
data['销售额增长率'] = data.groupby('名称')['金额（元）'].pct_change()
data['价格波动率'] = data.groupby('名称')['平均单价'].transform('std') / data['平均单价']

# 填充可能的缺失值
data.fillna(0, inplace=True)

# 选择用于聚类的特征
features = ['月均销量', '月均销售额', '销售额增长率', '价格波动率']
scaled_features = StandardScaler().fit_transform(data[features])

# 二、聚类分析

# 1. K-Means聚类分析
kmeans = KMeans(n_clusters=3, random_state=42)
data['KMeans_Cluster'] = kmeans.fit_predict(scaled_features)

# 2. DBSCAN聚类分析
dbscan = DBSCAN(eps=0.5, min_samples=5)
data['DBSCAN_Cluster'] = dbscan.fit_predict(scaled_features)

# 聚类结果可视化 - K-Means
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='月均销量', y='月均销售额', hue='KMeans_Cluster', palette='viridis', marker='o')
plt.title('K-Means Clustering Result')
plt.xlabel('Average Monthly Sales (Boxes)')
plt.ylabel('Average Monthly Revenue (Yuan)')
plt.show()

# 聚类结果可视化 - DBSCAN
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='月均销量', y='月均销售额', hue='DBSCAN_Cluster', palette='viridis', marker='o')
plt.title('DBSCAN Clustering Result')
plt.xlabel('Average Monthly Sales (Boxes)')
plt.ylabel('Average Monthly Revenue (Yuan)')
plt.show()

# 三、影响分析

# 定性分析：
# 价格影响：观察价格与销量的关系
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='平均单价', y='销量（箱）')
plt.title('Price vs Sales')
plt.xlabel('Price (Yuan)')
plt.ylabel('Sales (Boxes)')
plt.show()

# 年份影响：比较不同年份的销量变化
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='年份', y='销量（箱）', hue='名称')
plt.title('Sales Over Years')
plt.xlabel('Year')
plt.ylabel('Sales (Boxes)')
plt.show()

# 时间影响：观察月份数据
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='月', y='销量（箱）', hue='名称')
plt.title('Seasonal Sales')
plt.xlabel('Month')
plt.ylabel('Sales (Boxes)')
plt.show()

# 定量分析 - 多元线性回归模型

# 选择自变量和因变量
X = data[['平均单价', '年份', '月']]
y = data['销量（箱）']

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测与评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Linear Regression - MSE: {mse}, R^2: {r2}')
print('Model Coefficients:', model.coef_)
print('Model Intercept:', model.intercept_)

# 结果可视化
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Linear Regression: Actual vs Predicted Sales')
plt.show()
