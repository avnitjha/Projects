#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import yfinance as yf

# Function to fetch financial ratios for US stocks from Yahoo Finance
def fetch_financial_data(tickers):
    data = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            data[ticker] = {
                'PE_Ratio': info.get('forwardPE', np.nan),
                'PB_Ratio': info.get('priceToBook', np.nan),
                'DE_Ratio': info.get('debtToEquity', np.nan)
            }
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return data

# Sample US stocks (replace with your own list)
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']

# Fetch financial data
data = fetch_financial_data(tickers)

# Create DataFrame
df = pd.DataFrame(data).transpose()

# Drop rows with missing values
df.dropna(inplace=True)

# Fetch historical stock prices
stock_prices = yf.download(tickers, start="2023-01-01", end="2023-12-31")['Adj Close']

# Calculate daily returns
returns = stock_prices.pct_change().dropna()

# Calculate average daily return as the target variable
df['Average_Return'] = returns.mean(axis=0)

# Define features (financial ratios) and target variable
features = ['PE_Ratio', 'PB_Ratio', 'DE_Ratio']
target = 'Average_Return'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Define Random Forest regressor
rf = RandomForestRegressor(random_state=42)

# Define hyperparameters grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform GridSearchCV with reduced number of splits
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=min(5, len(X_train)), scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions
train_pred = best_model.predict(X_train)
test_pred = best_model.predict(X_test)

# Evaluate the model
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

print("Best Model Parameters:", grid_search.best_params_)
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

# Rank US stocks based on predicted values
df['Predicted_Average_Return'] = best_model.predict(df[features])
ranked_stocks = df.sort_values(by='Predicted_Average_Return')

print("Ranked Stocks:")
print(ranked_stocks[['Predicted_Average_Return']])


# In[ ]:




