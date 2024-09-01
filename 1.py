import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# 读取Excel文件中的所有表格
file_path = "A题.xlsx"
sheet_names = ["A1", "A2", "A3", "A4", "A5"]

# 读取每个表格到一个数据框列表中
dataframes = [pd.read_excel(file_path, sheet_name=sheet) for sheet in sheet_names]

# 合并所有数据框
data = pd.concat(dataframes, ignore_index=True)

# 计算指标
data['总销售额'] = data.groupby('名称')['金额（元）'].transform('sum')
data['平均月销量'] = data.groupby('名称')['销量（箱）'].transform('mean')
data['销售额波动系数'] = data.groupby('名称')['金额（元）'].transform(lambda x: x.std() / x.mean())
data['销量波动系数'] = data.groupby('名称')['销量（箱）'].transform(lambda x: x.std() / x.mean())
data['月销售增长率'] = data.groupby('名称')['金额（元）'].transform(lambda x: x.pct_change().mean())

# 选择特征指标
features = data[['总销售额', '平均月销量', '销售额波动系数', '销量波动系数', '月销售增长率']].drop_duplicates()

# 数据标准化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


# K-Means 聚类
kmeans = KMeans(n_clusters=3, random_state=0)  # 选择3个聚类中心作为示例
kmeans_labels = kmeans.fit_predict(scaled_features)

# 将聚类结果合并到原数据中
features['KMeans_Cluster'] = kmeans_labels

# 可视化结果
plt.figure(figsize=(10, 6))
sns.scatterplot(x='总销售额', y='平均月销量', hue='KMeans_Cluster', data=features, palette='viridis')
plt.title('K-Means 聚类结果')
plt.xlabel('总销售额')
plt.ylabel('平均月销量')
plt.legend(title='Cluster')
plt.show()


# DBSCAN 聚类
dbscan = DBSCAN(eps=0.5, min_samples=5)  # eps 和 min_samples 需要调整
dbscan_labels = dbscan.fit_predict(scaled_features)

# 将聚类结果合并到原数据中
features['DBSCAN_Cluster'] = dbscan_labels

# 可视化结果
plt.figure(figsize=(10, 6))
sns.scatterplot(x='总销售额', y='平均月销量', hue='DBSCAN_Cluster', data=features, palette='viridis')
plt.title('DBSCAN 聚类结果')
plt.xlabel('总销售额')
plt.ylabel('平均月销量')
plt.legend(title='Cluster')
plt.show()

