# Libraries and data

# Change directory
%cd /content/drive/MyDrive/Data Analysis Science/Time Series Forecasting Product/

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
# YYYY-MM-DD
df = pd.read_csv('nyc_data.csv')
future_df = pd.read_csv('future.csv')
future_df.head()

# Merge these data sets
df = pd.concat([df, future_df])
df = df.reset_index(drop = True)
df.tail()

# Rename variable
df = df.rename(columns = {'Demand': 'y',
                          'Date': 'ds'})
df.head(0)

# Transforming the date variable: YYYY-MM-DD
df.ds = pd.to_datetime(df.ds, format= "%m/%d/%Y")

## Holidays

# Easter Holiday
dates = df[df.Easter == 1].ds
easter = pd.DataFrame({'holiday': 'easter',
              'ds': dates,
              'lower_window': -5,
              'upper_window': 2})
easter

# Thanksgiving Holiday
dates = df[df.Thanksgiving == 1].ds
thanksgiving = pd.DataFrame({'holiday': 'thanksgiving',
              'ds': dates,
              'lower_window': -3,
              'upper_window': 5})
thanksgiving

# Christmas Holiday
dates = df[df.Christmas == 1].ds
christmas = pd.DataFrame({'holiday': 'christmas',
              'ds': dates,
              'lower_window': -7,
              'upper_window': 7})
christmas

# Combine all events
holidays = pd.concat([easter, thanksgiving, christmas])
holidays.head()

# Drop holidays from df
df_final = df.drop(columns = ["Easter", "Thanksgiving", "Christmas"])
df_final.head()

## Prophet Model

# Load the tuned parameters
parameters = pd.read_csv("Forecasting Product/best_params_prophet.csv", index_col = 0)
parameters

# Extracting the parameters
cps = float(parameters.loc['changepoint_prior_scale'])
hps = parameters.loc['holidays_prior_scale']
sps = parameters.loc['seasonality_prior_scale']
sm = parameters.loc['seasonality_mode'][0]

# Data
training = df_final.iloc[:-31, :] # all before 31 days
future_df = df_final.iloc[-31:, :]

from prophet import Prophet

sm[0]

# Building the model
m = Prophet(holidays = holidays,
            seasonality_mode = sm,
            seasonality_prior_scale = sps,
            holidays_prior_scale = hps,
            changepoint_prior_scale= cps)

m.add_regressor("Temperature")
m.add_regressor("Marketing")
m.fit(training)

# Forecasting

df_final.head()

df_final.iloc[:, 2:]

# Make a future dataframe
# It has the training and the test dataset.
# Then we will populate it with the forecast itself.

future = m.make_future_dataframe(periods = len(future_df),
                                 freq = "D") # W or M for weekly or monthly
future = pd.concat([future, df_final.iloc[:, 2:]], axis=1)
future.head()

# Forecasting
forecast = m.predict(future)
forecast.head()

# Components
m.plot_components(forecast);

# Let's take the last 31 days of prediction
predictions_prophet = forecast.yhat[-len(future_df):].rename("prophet")
predictions_prophet.index = future_df['ds']
predictions_prophet

# Exporting
# predictions_prophet.to_csv('Forecasting Product Ensemble/predictions_prophet.csv')
