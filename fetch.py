# -*- coding: utf-8 -*-
"""Fetch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13j0vwZc_q9yLdBBwnwDTMpBRRPrUKIm0
"""

import pandas as pd

# fill your own path
data = pd.read_csv('data_daily.csv')


data.head()

print(data)

data['# Date'] = pd.to_datetime(data['# Date'])

# Feature Engineering: Extracting month and day of the week
data['Month'] = data['# Date'].dt.month
data['DayOfWeek'] = data['# Date'].dt.dayofweek

# Aggregating the data to monthly totals
monthly_data = data.groupby('Month').sum().reset_index()

# Check for missing months and fill if necessary
all_months = pd.DataFrame({'Month': range(1, 13)})
monthly_data = all_months.merge(monthly_data, on='Month', how='left')
monthly_data.fillna(0, inplace=True)

print(monthly_data)

import torch
import torch.nn as nn
import torch.optim as optim

# Defining the Neural Network
class ReceiptPredictionModel(nn.Module):
    def __init__(self):
        super(ReceiptPredictionModel, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # 2 input features: Month and DayOfWeek
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)  # 1 output: Receipt_Count

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model
model = ReceiptPredictionModel()

# Convert data to PyTorch tensors
X_train = torch.tensor(monthly_data[['Month', 'DayOfWeek']].values, dtype=torch.float32)
y_train = torch.tensor(monthly_data['Receipt_Count'].values, dtype=torch.float32)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(1000):  # Number of epochs
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/1000], Loss: {loss.item():.4f}')

# Predicting for 2022 - Assuming you need a similar structure as 2021 data
X_test = torch.tensor([[month, 0] for month in range(1, 13)], dtype=torch.float32)  # Placeholder DayOfWeek
predictions = model(X_test).detach().numpy()

# Display predictions
print(predictions)

torch.save(model.state_dict(), 'receipt_prediction_model.pth')