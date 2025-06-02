# Libraries and data

# Change directory
%cd /content/drive/MyDrive/Time Series Forecasting Product

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
# YYYY-MM-DD
df = pd.read_csv('nyc_data.csv')
df.head()

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

from prophet import Prophet

# Building the model
m = Prophet(holidays = holidays,
            seasonality_mode = "multiplicative",
            seasonality_prior_scale = 10,
            holidays_prior_scale = 10,
            changepoint_prior_scale=0.05)

m.add_regressor("Temperature")
m.add_regressor("Marketing")
m.fit(df_final)

from prophet.diagnostics import cross_validation

# How many days we have in the data set
df.shape[0] - 180

# Cross-validation
df_cv = cross_validation(model = m,
                         horizon = '31 days',
                         period = '16 days',
                         initial = '2012 days',
                         parallel = 'processes')

#df_cv.head()

# Look at the output
df_cv.head()

# Performance metrics
from prophet.diagnostics import performance_metrics

performance_metrics(df_cv)

# RMSE and MAPE
RMSE = round(performance_metrics(df_cv)["rmse"].mean(), 1)
MAPE = round(performance_metrics(df_cv)["mape"].mean() * 100, 3)
print(f"RMSE: {RMSE}")
print(f"MAPE: {MAPE}%")

# Plotting the performance metrics over time
from prophet.plot import plot_cross_validation_metric
plot_cross_validation_metric(df_cv, metric = "rmse");

 # Parameter Tuning

from sklearn.model_selection import ParameterGrid
param_grid = {'seasonality_mode': ['additive', 'multiplicative'],
              'seasonality_prior_scale': [5, 10, 20],
              'holidays_prior_scale': [5, 10, 20],
              'changepoint_prior_scale': [0.01, 0.05, 0.1]}

grid = ParameterGrid(param_grid)
print(len(list(grid)))
print(list(grid))

# Store the results
rmse = []
i = 0

# Loop
for params in grid:
  print(f"{i} out of {len(list(grid))}")
  # Build the model
  m = Prophet(holidays = holidays,
              seasonality_mode = params['seasonality_mode'],
              seasonality_prior_scale = params['seasonality_prior_scale'],
              holidays_prior_scale = params['holidays_prior_scale'],
              changepoint_prior_scale=params['changepoint_prior_scale'])

  m.add_regressor("Temperature")
  m.add_regressor("Marketing")
  m.fit(df_final)

  # CV
  df_cv = cross_validation(model = m,
                          horizon = '31 days',
                          period = '16 days',
                          initial = '2012 days',
                          parallel = 'processes')

  # Measure and store error
  error = round(performance_metrics(df_cv)["rmse"].mean(), 1)
  rmse.append(error)

  i+= 1

# Parameter tuning outcome
tuning_results = pd.DataFrame(grid)
tuning_results['rmse'] = rmse
tuning_results

# Exporting the tuned parameters
best_params = tuning_results[tuning_results.rmse == tuning_results.rmse.min()].transpose()
best_params.to_csv('Forecasting Product/best_params_prophet.csv')
