from google.colab import drive
drive.mount('/content/drive')

# Libraries and data

!pip install numpy==1.24.4 pmdarima==2.0.3

# Change directory
%cd /content/drive/MyDrive/Data Analysis Science/Time Series Forecasting Product

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm

# Load the data
# YYYY-MM-DD
df = pd.read_csv('nyc_data.csv', index_col = 0, parse_dates = True)
future_df = pd.read_csv('future.csv', index_col = 0, parse_dates = True)
future_df.head()

# Rename variable
df = df.rename(columns = {'Demand': 'y'})
df.head(0)

# Extract regressors
train_X = df.iloc[:, 1:]
future_X = future_df.iloc[:, 1:]
future_X.head()

# Sarimax Model

# Get the best parameters
parameters = pd.read_csv("Forecasting Product/best_params_sarimax.csv",
                         index_col = 0)
parameters

# Store the individuals parameters
p = parameters.loc["p"][0]
d = parameters.loc["d"][0]
q = parameters.loc["q"][0]
P = parameters.loc["P"][0]
D = parameters.loc["D"][0]
Q = parameters.loc["Q"][0]


# Model
# hourly: 24, daily: 7, weekly:52, monthly: 12, quarterly: 4
model = pm.ARIMA(order = (p, d, q),
                 seasonal_order = (P, D, Q, 7),
                 X = train_X,
                 suppress_warnings = True,
                 force_stationarity = False)

model.fit(df.y)

# Forecasting

# Predictions
predictions_sarimax = pd.Series(model.predict(n_periods = len(future_df),
              X = future_X)).rename("sarimax")

predictions_sarimax.index = future_df.index
predictions_sarimax

# Visualization
df['y']['2020-01-01':].plot(figsize = (15, 6), legend = True)
predictions_sarimax.plot(legend = True)

# Exporting
# predictions_sarimax.to_csv('Forecasting Product/Ensemble/predictions_sarimax.csv')
