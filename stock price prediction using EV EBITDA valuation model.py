#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[2]:


# Sample data
data = {
    'EV/EBITDA': [10, 12, 15, 18, 20],
    'Stock_Price': [100, 120, 150, 180, 200]
}



# In[3]:


# Create a DataFrame
df = pd.DataFrame(data)



# In[4]:


# Splitting the data into features and target variable
X = df[['EV/EBITDA']]
y = df['Stock_Price']



# In[5]:


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# In[6]:


# Model training
model = LinearRegression()
model.fit(X_train, y_train)



# In[7]:


# Model evaluation
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)



# In[10]:


# Predicting stock prices
ev_ebitda = 25
predicted_price = model.predict([[ev_ebitda]])
print("Predicted stock price for EV/EBITDA ratio {}: ${}".format(ev_ebitda, predicted_price[0]))


# In[ ]:




