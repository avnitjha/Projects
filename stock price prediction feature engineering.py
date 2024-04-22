#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Function to fetch historical stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to calculate EV/EBITDA ratio
def calculate_ev_ebitda(stock_data):
    stock_data['EV'] = stock_data['Market Cap'] + stock_data['Total Debt'] - stock_data['Cash']
    stock_data['EBITDA'] = stock_data['EBITDA']
    stock_data['EV/EBITDA'] = stock_data['EV'] / stock_data['EBITDA']
    return stock_data

# Function to create additional features
def create_features(stock_data):
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
    stock_data['Return'] = stock_data['Close'].pct_change()
    stock_data['Volatility'] = stock_data['Return'].rolling(window=50).std()
    return stock_data.dropna()

# Load historical data for a stock (e.g., Apple)
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2023-12-31'
stock_data = get_stock_data(ticker, start_date, end_date)

# Calculate EV/EBITDA ratio
stock_data = calculate_ev_ebitda(stock_data)

# Create additional features
stock_data = create_features(stock_data)

# Define features and target variable
features = ['EV/EBITDA', 'SMA_50', 'SMA_200', 'Volatility']
target = 'Close'

X = stock_data[features]
y = stock_data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

# Predict stock prices for future dates
future_dates = pd.date_range(start='2024-01-01', end='2024-12-31')
future_features = pd.DataFrame(index=future_dates, columns=features)
future_features['EV/EBITDA'] = 20  # For demonstration purposes, assuming a constant EV/EBITDA ratio
future_features['SMA_50'] = stock_data['SMA_50'].mean()  # Using mean value for demonstration
future_features['SMA_200'] = stock_data['SMA_200'].mean()  # Using mean value for demonstration
future_features['Volatility'] = stock_data['Volatility'].mean()  # Using mean value for demonstration

future_predictions = model.predict(future_features)

# Display future price predictions
future_predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_predictions})
print(future_predictions_df)

