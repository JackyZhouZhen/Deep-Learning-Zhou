import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import mean_squared_error
from math import sqrt

# # Get the AAPL data
# ticker_symbol = "AAPL"
# aapl_data = yf.Ticker(ticker_symbol)
# data = aapl_data.history(period="5y")
# # Save the data as 'aapl_stock_data.csv'
# data.to_csv('aapl_stock_data.csv', index=True)

data = pd.read_csv('aapl_stock_data.csv')

# get the close price in the data and change it to numpy
close_prices = data['Close'].values.reshape(-1, 1)


scaler = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler.fit_transform(close_prices)

# create Sequential Data
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# # basic
# seq_length = 60
# X, y = create_sequences(close_prices_scaled, seq_length)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # change to PyTorch
# X_train = torch.from_numpy(X_train).float()
# y_train = torch.from_numpy(y_train).float()
# X_test = torch.from_numpy(X_test).float()
# y_test = torch.from_numpy(y_test).float()
#
# # create DataLoader
# train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)  # batch_size 是可调节参数


# # change the seq_length
# seq_length_1 = 20
# seq_length_2 = 30
# seq_length_3 = 40
# seq_length_4 = 50
# seq_length_5 = 60
#
# x1, y1 = create_sequences(close_prices_scaled, seq_length_1)
# x2, y2 = create_sequences(close_prices_scaled, seq_length_2)
# x3, y3 = create_sequences(close_prices_scaled, seq_length_3)
# x4, y4 = create_sequences(close_prices_scaled, seq_length_4)
# x5, y5 = create_sequences(close_prices_scaled, seq_length_5)
#
# # divide the data set
# x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x1, y1, test_size=0.2, random_state=42)
# x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x2, y2, test_size=0.2, random_state=42)
# x_train_3, x_test_3, y_train_3, y_test_3 = train_test_split(x3, y3, test_size=0.2, random_state=42)
# x_train_4, x_test_4, y_train_4, y_test_4 = train_test_split(x4, y4, test_size=0.2, random_state=42)
# x_train_5, x_test_5, y_train_5, y_test_5 = train_test_split(x5, y5, test_size=0.2, random_state=42)
#
#
# # change the data to PyTorch side
# x_train_1 = torch.from_numpy(x_train_1).float()
# y_train_1 = torch.from_numpy(y_train_1).float()
# x_test_1 = torch.from_numpy(x_test_1).float()
# y_test_1 = torch.from_numpy(y_test_1).float()
#
# x_train_2 = torch.from_numpy(x_train_2).float()
# y_train_2 = torch.from_numpy(y_train_2).float()
# x_test_2 = torch.from_numpy(x_test_2).float()
# y_test_2 = torch.from_numpy(y_test_2).float()
#
# x_train_3 = torch.from_numpy(x_train_3).float()
# y_train_3 = torch.from_numpy(y_train_3).float()
# x_test_3 = torch.from_numpy(x_test_3).float()
# y_test_3 = torch.from_numpy(y_test_3).float()
#
# x_train_4 = torch.from_numpy(x_train_4).float()
# y_train_4 = torch.from_numpy(y_train_4).float()
# x_test_4 = torch.from_numpy(x_test_4).float()
# y_test_4 = torch.from_numpy(y_test_4).float()
#
# x_train_5 = torch.from_numpy(x_train_5).float()
# y_train_5 = torch.from_numpy(y_train_5).float()
# x_test_5 = torch.from_numpy(x_test_5).float()
# y_test_5 = torch.from_numpy(y_test_5).float()
#
#
#
# # create DataLoader
# train_loader_1 = DataLoader(TensorDataset(x_train_1, y_train_1), batch_size=64, shuffle=True)
# train_loader_2 = DataLoader(TensorDataset(x_train_2, y_train_2), batch_size=64, shuffle=True)
# train_loader_3 = DataLoader(TensorDataset(x_train_3, y_train_3), batch_size=64, shuffle=True)
# train_loader_4 = DataLoader(TensorDataset(x_train_4, y_train_4), batch_size=64, shuffle=True)
# train_loader_5 = DataLoader(TensorDataset(x_train_5, y_train_5), batch_size=64, shuffle=True)


# new
seq_length = 60
X, y = create_sequences(close_prices_scaled, seq_length)

# divide the data set to three part, as 7:2:1
x_1, x_final_test, y_1, y_final_test = train_test_split(X, y, test_size=0.1, random_state=42)
validation_size = 0.2 / 0.9
x_train, x_val, y_train, y_val = train_test_split(x_1, y_1, test_size=validation_size, random_state=42)
# # divide the data set to two part
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
X_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(x_val).float()
y_test = torch.from_numpy(y_val).float()
x_final_test = torch.from_numpy(x_final_test).float()
y_final_test = torch.from_numpy(y_final_test).float()

# create the DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)  # batch_size 是可调节参数

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        # RNN
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def train_model(model, train_loader, criterion, optimizer, n_epochs):
    for epoch in range(n_epochs):
        for seq, labels in train_loader:
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

n_epochs = 100  # epochs times


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        # LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out




# # basic model
# model = RNNModel(input_size = 1, hidden_size = 50, num_layers = 2, output_size = 1)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# train_model(model, train_loader, criterion, optimizer, n_epochs)
# model.eval()  # change the model to eval

# with torch.no_grad():
#     predictions = model(X_test)
#     # 反标准化预测
#     predictions = scaler.inverse_transform(predictions.numpy())
#     actual = scaler.inverse_transform(y_test.numpy())
#
# residuals = actual - predictions
#
# mse = mean_squared_error(actual, predictions)
# rmse = sqrt(mse)
# print(f"MSE: {mse}")
# print(f"RMSE: {rmse}")
#
# dates = data.index[-len(residuals):]
# plt.figure(figsize=(10, 6))
# plt.plot(dates, residuals, label='Residuals')
# plt.axhline(y=0, color='r', linestyle='--')
# plt.title('Residuals of AAPL Stock Price Predictions')
# plt.xlabel('Date')
# plt.ylabel('Residuals')
# plt.legend()
# plt.show()



# # hidden select
# model_hidden1 = RNNModel(input_size = 1, hidden_size = 20, num_layers = 2, output_size = 1)
# model_hidden2 = RNNModel(input_size = 1, hidden_size = 50, num_layers = 2, output_size = 1)
# model_hidden3 = RNNModel(input_size = 1, hidden_size = 80, num_layers = 2, output_size = 1)
# model_hidden4 = RNNModel(input_size = 1, hidden_size = 130, num_layers = 2, output_size = 1)
# model_hidden5 = RNNModel(input_size = 1, hidden_size = 180, num_layers = 2, output_size = 1)
#
# optimizer1 = torch.optim.Adam(model_hidden1.parameters(), lr=0.001)
# optimizer2 = torch.optim.Adam(model_hidden2.parameters(), lr=0.001)
# optimizer3 = torch.optim.Adam(model_hidden3.parameters(), lr=0.001)
# optimizer4 = torch.optim.Adam(model_hidden4.parameters(), lr=0.001)
# optimizer5 = torch.optim.Adam(model_hidden5.parameters(), lr=0.001)
#
# # hidden select
# train_model(model_hidden1, train_loader, criterion, optimizer1, n_epochs)
# train_model(model_hidden2, train_loader, criterion, optimizer2, n_epochs)
# train_model(model_hidden3, train_loader, criterion, optimizer3, n_epochs)
# train_model(model_hidden4, train_loader, criterion, optimizer4, n_epochs)
# train_model(model_hidden5, train_loader, criterion, optimizer5, n_epochs)
#
# model_hidden1.eval()
# model_hidden2.eval()
# model_hidden3.eval()
# model_hidden4.eval()
# model_hidden5.eval()
#
# with torch.no_grad():
#     predictions = model_hidden1(X_test)
#     predictions_1 = scaler.inverse_transform(predictions.numpy())
#     actual_1 = scaler.inverse_transform(y_test.numpy())
#
# with torch.no_grad():
#     predictions = model_hidden2(X_test)
#     predictions_2 = scaler.inverse_transform(predictions.numpy())
#     actual_2 = scaler.inverse_transform(y_test.numpy())
#
# with torch.no_grad():
#     predictions = model_hidden3(X_test)
#     predictions_3 = scaler.inverse_transform(predictions.numpy())
#     actual_3 = scaler.inverse_transform(y_test.numpy())
#
# with torch.no_grad():
#     predictions = model_hidden4(X_test)
#     predictions_4 = scaler.inverse_transform(predictions.numpy())
#     actual_4 = scaler.inverse_transform(y_test.numpy())
#
# with torch.no_grad():
#     predictions = model_hidden5(X_test)
#     predictions_5 = scaler.inverse_transform(predictions.numpy())
#     actual_5 = scaler.inverse_transform(y_test.numpy())
#
# # 计算残差
# residuals_1 = actual_1 - predictions_1
# residuals_2 = actual_2 - predictions_2
# residuals_3 = actual_3 - predictions_3
# residuals_4 = actual_4 - predictions_4
# residuals_5 = actual_5 - predictions_5
#
# mse_1 = mean_squared_error(actual_1, predictions_1)
# rmse_1 = sqrt(mse_1)
# print(f"RMSE_1: {rmse_1}")
#
# mse_2 = mean_squared_error(actual_2, predictions_2)
# rmse_2 = sqrt(mse_2)
# print(f"RMSE_2: {rmse_2}")
#
# mse_3 = mean_squared_error(actual_3, predictions_3)
# rmse_3 = sqrt(mse_3)
# print(f"RMSE_3: {rmse_3}")
#
# mse_4 = mean_squared_error(actual_4, predictions_4)
# rmse_4 = sqrt(mse_4)
# print(f"RMSE_4: {rmse_4}")
#
# mse_5 = mean_squared_error(actual_5, predictions_5)
# rmse_5 = sqrt(mse_5)
# print(f"RMSE_5: {rmse_5}")
#
# dates = data.index[-len(residuals_1):]
#
# plt.figure(figsize=(12, 6))
# plt.plot(dates, residuals_1, label='Residuals Model 1 (Hidden Size 20)')
# plt.plot(dates, residuals_2, label='Residuals Model 2 (Hidden Size 50)')
# plt.plot(dates, residuals_3, label='Residuals Model 2 (Hidden Size 80)')
# plt.plot(dates, residuals_4, label='Residuals Model 2 (Hidden Size 130)')
# plt.plot(dates, residuals_5, label='Residuals Model 2 (Hidden Size 180)')
# plt.axhline(y=0, color='r', linestyle='--')
# plt.title('Comparison of Residuals')
# plt.xlabel('Date')
# plt.ylabel('Residuals')
# plt.legend()
# plt.show()




# # layers select
# model_l1 = RNNModel(input_size = 1, hidden_size = 80, num_layers = 1, output_size = 1)
# model_l2 = RNNModel(input_size = 1, hidden_size = 80, num_layers = 2, output_size = 1)
# model_l3 = RNNModel(input_size = 1, hidden_size = 80, num_layers = 3, output_size = 1)
# model_l4 = RNNModel(input_size = 1, hidden_size = 80, num_layers = 4, output_size = 1)
# model_l5 = RNNModel(input_size = 1, hidden_size = 80, num_layers = 5, output_size = 1)
#
# optimizer1 = torch.optim.Adam(model_l1.parameters(), lr=0.001)
# optimizer2 = torch.optim.Adam(model_l2.parameters(), lr=0.001)
# optimizer3 = torch.optim.Adam(model_l3.parameters(), lr=0.001)
# optimizer4 = torch.optim.Adam(model_l4.parameters(), lr=0.001)
# optimizer5 = torch.optim.Adam(model_l5.parameters(), lr=0.001)
#
# # layers select
# train_model(model_l1, train_loader, criterion, optimizer1, n_epochs)
# train_model(model_l2, train_loader, criterion, optimizer2, n_epochs)
# train_model(model_l3, train_loader, criterion, optimizer3, n_epochs)
# train_model(model_l4, train_loader, criterion, optimizer4, n_epochs)
# train_model(model_l5, train_loader, criterion, optimizer5, n_epochs)
#
#
#
# model_l1.eval()
# model_l2.eval()
# model_l3.eval()
# model_l4.eval()
# model_l5.eval()
#
# with torch.no_grad():
#     predictions = model_l1(X_test)
#     predictions_1 = scaler.inverse_transform(predictions.numpy())
#     actual_1 = scaler.inverse_transform(y_test.numpy())
#
# with torch.no_grad():
#     predictions = model_l2(X_test)
#     predictions_2 = scaler.inverse_transform(predictions.numpy())
#     actual_2 = scaler.inverse_transform(y_test.numpy())
#
# with torch.no_grad():
#     predictions = model_l3(X_test)
#     predictions_3 = scaler.inverse_transform(predictions.numpy())
#     actual_3 = scaler.inverse_transform(y_test.numpy())
#
# with torch.no_grad():
#     predictions = model_l4(X_test)
#     predictions_4 = scaler.inverse_transform(predictions.numpy())
#     actual_4 = scaler.inverse_transform(y_test.numpy())
#
# with torch.no_grad():
#     predictions = model_l5(X_test)
#     predictions_5 = scaler.inverse_transform(predictions.numpy())
#     actual_5 = scaler.inverse_transform(y_test.numpy())
#
# # calculate the residual
# residuals_1 = actual_1 - predictions_1
# residuals_2 = actual_2 - predictions_2
# residuals_3 = actual_3 - predictions_3
# residuals_4 = actual_4 - predictions_4
# residuals_5 = actual_5 - predictions_5
#
#
# mse_1 = mean_squared_error(actual_1, predictions_1)
# rmse_1 = sqrt(mse_1)
# print(f"RMSE_1: {rmse_1}")
#
# mse_2 = mean_squared_error(actual_2, predictions_2)
# rmse_2 = sqrt(mse_2)
# print(f"RMSE_2: {rmse_2}")
#
# mse_3 = mean_squared_error(actual_3, predictions_3)
# rmse_3 = sqrt(mse_3)
# print(f"RMSE_3: {rmse_3}")
#
# mse_4 = mean_squared_error(actual_4, predictions_4)
# rmse_4 = sqrt(mse_4)
# print(f"RMSE_4: {rmse_4}")
#
# mse_5 = mean_squared_error(actual_5, predictions_5)
# rmse_5 = sqrt(mse_5)
# print(f"RMSE_5: {rmse_5}")
#
# dates = data.index[-len(residuals_1):]
# plt.figure(figsize=(12, 6))
# plt.plot(dates, residuals_1, label='Residuals Model 1 (Layer Size 1)')
# plt.plot(dates, residuals_2, label='Residuals Model 2 (Layer Size 2)')
# plt.plot(dates, residuals_3, label='Residuals Model 2 (Layer Size 3)')
# plt.plot(dates, residuals_4, label='Residuals Model 2 (Layer Size 4)')
# plt.plot(dates, residuals_5, label='Residuals Model 2 (Layer Size 5)')
# plt.axhline(y=0, color='r', linestyle='--')
# plt.title('Comparison of Residuals')
# plt.xlabel('Date')
# plt.ylabel('Residuals')
# plt.legend()
# plt.show()




# # change the seq_length
# model_1 = RNNModel(input_size = 1, hidden_size = 80, num_layers = 3, output_size = 1)
# model_2 = RNNModel(input_size = 1, hidden_size = 80, num_layers = 3, output_size = 1)
# model_3 = RNNModel(input_size = 1, hidden_size = 80, num_layers = 3, output_size = 1)
# model_4 = RNNModel(input_size = 1, hidden_size = 80, num_layers = 3, output_size = 1)
# model_5 = RNNModel(input_size = 1, hidden_size = 80, num_layers = 3, output_size = 1)
#
# optimizer = torch.optim.Adam(model_1.parameters(), lr=0.001)
#
# train_model(model_1, train_loader_1, criterion, optimizer, n_epochs)
# train_model(model_2, train_loader_2, criterion, optimizer, n_epochs)
# train_model(model_3, train_loader_3, criterion, optimizer, n_epochs)
# train_model(model_4, train_loader_4, criterion, optimizer, n_epochs)
# train_model(model_5, train_loader_5, criterion, optimizer, n_epochs)
#
# model_1.eval()
# model_2.eval()
# model_3.eval()
# model_4.eval()
# model_5.eval()
#
# with torch.no_grad():
#     predictions = model_1(x_test_1)
#     predictions_1 = scaler.inverse_transform(predictions.numpy())
#     actual_1 = scaler.inverse_transform(y_test_1.numpy())
#
# with torch.no_grad():
#     predictions = model_2(x_test_2)
#     predictions_2 = scaler.inverse_transform(predictions.numpy())
#     actual_2 = scaler.inverse_transform(y_test_2.numpy())
#
# with torch.no_grad():
#     predictions = model_3(x_test_3)
#     predictions_3 = scaler.inverse_transform(predictions.numpy())
#     actual_3 = scaler.inverse_transform(y_test_3.numpy())
#
# with torch.no_grad():
#     predictions = model_4(x_test_4)
#     predictions_4 = scaler.inverse_transform(predictions.numpy())
#     actual_4 = scaler.inverse_transform(y_test_4.numpy())
#
# with torch.no_grad():
#     predictions = model_5(x_test_5)
#     predictions_5 = scaler.inverse_transform(predictions.numpy())
#     actual_5 = scaler.inverse_transform(y_test_5.numpy())
#
# # calculate the residual
# residuals_1 = actual_1 - predictions_1
# residuals_2 = actual_2 - predictions_2
# residuals_3 = actual_3 - predictions_3
# residuals_4 = actual_4 - predictions_4
# residuals_5 = actual_5 - predictions_5
#
#
# mse_1 = mean_squared_error(actual_1, predictions_1)
# rmse_1 = sqrt(mse_1)
# print(f"RMSE_1: {rmse_1}")
#
# mse_2 = mean_squared_error(actual_2, predictions_2)
# rmse_2 = sqrt(mse_2)
# print(f"RMSE_2: {rmse_2}")
#
# mse_3 = mean_squared_error(actual_3, predictions_3)
# rmse_3 = sqrt(mse_3)
# print(f"RMSE_3: {rmse_3}")
#
# mse_4 = mean_squared_error(actual_4, predictions_4)
# rmse_4 = sqrt(mse_4)
# print(f"RMSE_4: {rmse_4}")
#
# mse_5 = mean_squared_error(actual_5, predictions_5)
# rmse_5 = sqrt(mse_5)
# print(f"RMSE_5: {rmse_5}")

criterion = nn.MSELoss()  # loss function


# new model
model = RNNModel(input_size = 1, hidden_size = 80, num_layers = 3, output_size = 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_model(model, train_loader, criterion, optimizer, n_epochs)
model.eval()  # change the model to eval
with torch.no_grad():
    predictions = model(X_test)
    predictions = scaler.inverse_transform(predictions.numpy())
    actual = scaler.inverse_transform(y_test.numpy())
# calculate the residual
residuals = actual - predictions
mse = mean_squared_error(actual, predictions)
rmse = sqrt(mse)
print(f"RMSE_BASIC: {rmse}")

#LSTM
model2 = LSTMModel(input_size=1, hidden_size = 80, num_layers=3, output_size=1)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
train_model(model2, train_loader, criterion, optimizer2, n_epochs)
model2.eval()
with torch.no_grad():
    predictions = model2(X_test)
    predictions2 = scaler.inverse_transform(predictions.numpy())
    actual2 = scaler.inverse_transform(y_test.numpy())
# calculate the residual
residuals2 = actual2 - predictions2

mse2 = mean_squared_error(actual2, predictions2)
rmse2 = sqrt(mse2)
print(f"RMSE_LSTM: {rmse2}")


model.eval()  # change the model to eval
with torch.no_grad():
    predictions_final = model(x_final_test)
    predictions_final = scaler.inverse_transform(predictions_final.numpy())
    actual_final = scaler.inverse_transform(y_final_test.numpy())
# calculate the residual
residuals_final = actual_final - predictions_final
mse_final = mean_squared_error(actual_final, predictions_final)
rmse_final = sqrt(mse_final)
print(f"RMSE_BASIC_finaltest: {rmse_final}")


# # basic model
# dates = data.index[-len(residuals):]  # 假设数据集中有一个日期索引
# plt.figure(figsize=(10, 6))
# plt.plot(dates, residuals, label='Residuals')
# plt.axhline(y=0, color='r', linestyle='--')
# plt.title('Residuals of AAPL Stock Price Predictions')
# plt.xlabel('Date')
# plt.ylabel('Residuals')
# plt.legend()
# plt.show()


# basic model
dates = data.index[-len(residuals_final):]
plt.figure(figsize=(10, 6))
plt.plot(dates, residuals_final, label='Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals of AAPL Stock Price Predictions')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.legend()
plt.show()
# dates = data.index[-len(residuals):]
#
# plt.figure(figsize=(12, 6))
# plt.plot(dates, residuals, label='Residuals Model 1 (basic RNN)')
# plt.plot(dates, residuals2, label='Residuals Model 2 (LSTM)')
# plt.axhline(y=0, color='r', linestyle='--')
# plt.title('Comparison of Residuals')
# plt.xlabel('Date')
# plt.ylabel('Residuals')
# plt.legend()
# plt.show()
