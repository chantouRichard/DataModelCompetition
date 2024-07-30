import pandas as pd
import numpy as np
from skimage.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# 加载数据
file_path = 'A1.xlsx'
data = pd.read_excel(file_path)

# 将'月份'列转换为日期格式并设置为索引
data['月份'] = pd.to_datetime(data['月份'], format='%Y%m')
data.set_index('月份', inplace=True)

# 提取并确保销量数据为浮点格式
sales_volume = data['销量（箱）'].astype(float)

# 填充空值
sales_volume = sales_volume.fillna(method='ffill')

# 将数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(sales_volume.values.reshape(-1, 1))

# 创建数据集
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# 设置时间步
time_step = 12
X, y = create_dataset(scaled_data, time_step)

# 将数据重塑为LSTM的输入格式
X = X.reshape(X.shape[0], X.shape[1], 1)

# 分割训练和测试数据
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# 进行预测
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 反归一化
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# 计算评估指标
train_mse = mean_squared_error(y_train, train_predict)
train_mae = mean_absolute_error(y_train, train_predict)
test_mse = mean_squared_error(y_test, test_predict)
test_mae = mean_absolute_error(y_test, test_predict)

print(f'训练集 MSE: {train_mse}, MAE: {train_mae}')
print(f'测试集 MSE: {test_mse}, MAE: {test_mae}')

# 准备预测未来的20个月数据
x_input = scaled_data[-time_step:].reshape(1, -1)
temp_input = list(x_input[0])
lst_output = []
n_steps = time_step
i = 0
while i < 20:
    if len(temp_input) > time_step:
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i += 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i += 1

# 反归一化预测值
future_predictions = scaler.inverse_transform(lst_output)

# 准备结果数据
future_dates = pd.date_range(start=sales_volume.index[-1] + pd.DateOffset(months=1), periods=20, freq='M')
future_df = pd.DataFrame({
    'Date': future_dates,
    'Sales': future_predictions.flatten()
})

# 合并观测数据和未来预测数据
observed_data = sales_volume.reset_index()
observed_data.columns = ['Date', 'Sales']

result_df = pd.concat([observed_data, future_df])

# 保存结果到Excel文件
output_file_path = 'a1f1.xlsx'
result_df.to_excel(output_file_path, index=False)

# 绘制结果
plt.figure(figsize=(14, 7))
plt.plot(observed_data['Date'], observed_data['Sales'], label='Observed', color='blue', marker='o')
plt.plot(future_df['Date'], future_df['Sales'], label='Forecast', color='red', linestyle='--', marker='x')
plt.fill_between(future_df['Date'], future_df['Sales'], color='red', alpha=0.1)
plt.title('Sales Volume Forecast using LSTM')
plt.xlabel('Date')
plt.ylabel('Sales Volume (Boxes)')
plt.legend()
plt.grid(True)
plt.show()

print(f'预测结果已保存到: {output_file_path}')
