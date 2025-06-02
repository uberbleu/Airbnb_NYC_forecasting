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
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from pmdarima import model_selection

# Load the data
# YYYY-MM-DD
df = pd.read_csv('nyc_data.csv', index_col = 0, parse_dates = True)
df.head()

# Rename variable
df = df.rename(columns = {'Demand': 'y'})
df.head(0)

# Extract regressors
X = df.iloc[:, 1:]
X.head()

# Stationarity

# Test
pvalue = adfuller(x = df.y)[1] # to get only the p value

# Condition to read test
if pvalue < 0.05:
  print(f'The series is stationary. The p-value is {pvalue}')
else:
  print(f'The series is not stationary. The p-value is {pvalue}')

# Differencing
df.y.diff().dropna()

# Test
pvalue = adfuller(x = df.y.diff().dropna())[1] # to get only the p value

# Condition to read test
if pvalue < 0.05:
  print(f'The series is stationary. The p-value is {pvalue}')
else:
  print(f'The series is not stationary. The p-value is {pvalue}')

# Sarimax Model

# Model
# hourly: 24, daily: 7, weekly:52, monthly: 12, quarterly: 4
model = pm.ARIMA(order = (1, 1, 1),
                 seasonal_order = (1, 1, 1, 7),
                 X = X,
                 suppress_warnings = True,
                 force_stationarity = False)

# Cross-validation
cv = model_selection.RollingForecastCV(h = 31,
                                       step = 16,
                                       initial = df.shape[0] - 180)
cv_score = model_selection.cross_val_score(model,
                                           y = df.y,
                                           scoring = 'mean_squared_error',
                                           cv = cv,
                                           verbose = 1,
                                           error_score = 10000000000)

# CV performance
error = np.sqrt(np.average(cv_score))

Parameter Tuning

# Grid
param_grid = {'p': [0, 1],
              'd': [1],
              'q': [0, 1],
              'P': [0, 1],
              'D': [0, 1],
              'Q': [0, 1]}

grid = ParameterGrid(param_grid)
len(list(grid))

# Parameter Tuning
rmse = []
i = 1
# Parameter loop
for params in grid:
  print(f" {i} / {len(list(grid))}")
  # model
  model = pm.ARIMA(order = (params['p'], params['d'], params['q']),
                 seasonal_order = (params['P'], params['D'], params['Q'], 7),
                 X = X,
                 suppress_warnings = True,
                 force_stationarity = False)
  # CV
  cv = model_selection.RollingForecastCV(h = 31,
                                       step = 16,
                                       initial = df.shape[0] - 180)
  cv_score = model_selection.cross_val_score(model,
                                           y = df.y,
                                           scoring = 'mean_squared_error',
                                           cv = cv,
                                           verbose = 1,
                                           error_score = 10000000000)
  # Error
  error = np.sqrt(np.average(cv_score))
  rmse.append(error)
  i += 1

# Check the results
tuning_results = pd.DataFrame(grid)
tuning_results['rmse'] = rmse
tuning_results

# Export best parameters
best_params = tuning_results[tuning_results.rmse == tuning_results.rmse.min()].transpose()
best_params.to_csv("Forecasting Product/best_params_sarimax.csv")
