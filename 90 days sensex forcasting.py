#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta


# In[2]:


file_path = "E:\\CSVForDate.csv"
data = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True, index_col='Date')
data = data.sort_index()


# In[3]:


# Define and fit the model
model = ExponentialSmoothing(data['Close'], trend=None, seasonal=None)
model_fit = model.fit()


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta

# Load and prepare data
file_path = "E:\\CSVForDate.csv"
data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
data.index = pd.to_datetime(data.index)  # Convert index to datetime
data = data.sort_index()

# Ensure the index has daily frequency
data = data.asfreq('D')  # Adjust frequency if needed

# Fill missing values if necessary
data = data.fillna(method='ffill')  # Forward fill missing values

# Define and fit the model
model = ExponentialSmoothing(data['Close'], trend=None, seasonal=None)
model_fit = model.fit()

# Generate in-sample forecast
data['in_sample_forecast'] = model_fit.fittedvalues

# Generate out-of-sample forecast
forecast_steps = 90
forecast = model_fit.forecast(steps=forecast_steps)

# Create a date range for the forecast
forecast_index = [data.index[-1] + timedelta(days=i) for i in range(1, forecast_steps + 1)]
forecast_df = pd.DataFrame({
    'Forecast': forecast
}, index=forecast_index)

# Plot the results
plt.figure(figsize=(14, 7))

# Historical Prices
plt.plot(data.index, data['Close'], label='Historical Prices', color='blue')

# In-Sample Forecast
plt.plot(data.index, data['in_sample_forecast'], label='In-Sample Forecast', color='orange')

# Out-of-Sample Forecast
plt.plot(forecast_df.index, forecast_df['Forecast'], label='Out-of-Sample Forecast (90 days)', color='green', linestyle='--')

# Formatting the plot
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('BSE Sensex Price Forecast')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[6]:


get_ipython().system('pip install pandas matplotlib prophet')


# In[10]:


import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# File path
file_path = "E:\\CSVForDate.csv"

# Load data with correct date format
date_format = '%d-%b-%y'  # Adjust this to match your CSV date format
data = pd.read_csv(file_path, parse_dates=['Date'], date_parser=lambda x: pd.to_datetime(x, format=date_format))

# Rename columns for Prophet
data = data.rename(columns={'Date': 'ds', 'Close': 'y'})

# Ensure the data is sorted by date
data = data.sort_values('ds')

# Create and fit the model
model = Prophet()
model.fit(data)

# Create future dataframe for out-of-sample forecast
future = model.make_future_dataframe(periods=90)  # Forecasting for the next 90 days
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.title('BSE Sensex Forecast')
plt.show()

# Plot the components (trend, yearly seasonality, etc.)
fig2 = model.plot_components(forecast)
plt.show()

# Extract in-sample forecast
in_sample_forecast = forecast[forecast['ds'] <= data['ds'].max()]
in_sample_actual = data[data['ds'] <= data['ds'].max()]

# Plot actual vs. in-sample forecast
plt.figure(figsize=(12, 6))
plt.plot(in_sample_actual['ds'], in_sample_actual['y'], label='Actual')
plt.plot(in_sample_forecast['ds'], in_sample_forecast['yhat'], label='In-Sample Forecast', linestyle='--')
plt.legend()
plt.title('In-Sample Forecast vs Actual')
plt.show()

# Extract out-of-sample forecast
out_sample_forecast = forecast[forecast['ds'] > data['ds'].max()]

# Plot out-of-sample forecast
plt.figure(figsize=(12, 6))
plt.plot(out_sample_forecast['ds'], out_sample_forecast['yhat'], label='Out-of-Sample Forecast')
plt.title('Out-of-Sample Forecast (Next 90 Days)')
plt.show()


# In[ ]:




