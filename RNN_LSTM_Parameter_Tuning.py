# Libraries and data

# Change directory
%cd /content/drive/MyDrive/Data Analysis Science/Time Series Forecasting Product

! pip install darts

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from darts.timeseries import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel
from sklearn.metrics import mean_squared_error



# Load the data
# YYYY-MM-DD
df = pd.read_csv('nyc_data.csv', index_col = 0, parse_dates = True)
df.head()

# Rename variable
df = df.rename(columns = {'Demand': 'y'})
df.head(0)

# Prepare for LSTM

# Creating Time Series
series = TimeSeries.from_series(df.y)
# Anything that is not a "y" is a covariate
covariates = TimeSeries.from_dataframe(df.iloc[:, 1:])

# Seasonality / trend
# Year
year_series = datetime_attribute_timeseries(pd.date_range(start = series.start_time(),
                                            freq = series.freq_str,
                                            periods = df.shape[0]),
                                            attribute = "year",
                                            one_hot = False)

# Month
month_series = datetime_attribute_timeseries(year_series,
                                             attribute = 'month',
                                             one_hot = True)

# Weekday
weekday_series = datetime_attribute_timeseries(year_series,
                                             attribute = 'weekday',
                                             one_hot = True)

series.start_time()

series.freq_str

df.shape[0]

year_series

month_series

# Scaling
transformer1 = Scaler()
transformer2 = Scaler()

# Scaling Y
y_transformed = transformer1.fit_transform(series)

# Stacking the covariates
covariates = covariates.stack(year_series)

# Scaling the covariates
covariates_transformed = transformer2.fit_transform(covariates)
covariates_transformed

# Stack the seasonal variables
covariates_transformed = covariates_transformed.stack(month_series)
covariates_transformed = covariates_transformed.stack(weekday_series)
covariates_transformed

# LSTM Model

# Build the LSTM model
model = RNNModel(model="LSTM",
                 hidden_dim = 20,
                 n_rnn_layers = 2,
                 dropout = 0.2,
                 n_epochs = 20,
                 optimizer_kwargs = {"lr": 0.003},
                 random_state = 42,
                 training_length = 20,
                 input_chunk_length = 15,
                 pl_trainer_kwargs = {"accelerator": "gpu", "devices": [0]})

# Fit the model to the data
model.fit(y_transformed,
          future_covariates = covariates_transformed)

# Cross-Validation

# CV
cv = model.historical_forecasts(series = y_transformed,
                                future_covariates = covariates_transformed,
                                start = df.shape[0] - 180,
                                forecast_horizon = 31,
                                stride = 16,
                                retrain = True,
                                last_points_only = False)

!pip install scikit-learn --upgrade

# Store the results
rmse_cv = []

for i in range(len(cv)):

  # Compute the RMSE for the CV
  # Get the forecast as a timeseries object
  ts = transformer1.inverse_transform(cv[i])

  # Manually convert to a pandas Series (values + time index)
  predictions = pd.Series(ts.values().squeeze(), index = ts.time_index)
  # this is deprecated --> predictions = transformer1.inverse_transform(cv[i]).to_pandas()

  # Actual values
  start = predictions.index.min()
  end = predictions.index.max()

  actuals = df.y[start:end]

  # Compute the error
  error_cv = mean_squared_error(actuals, predictions) ** 0.5 # squared = False doesn't work for some reason so I used ** 0.5

  # Save the error
  rmse_cv.append(error_cv)

print(f"The RMSE is {np.mean(rmse_cv)}")

# Parameter Tuning

# Grid
param_grid = {'n_rnn_layers': [1, 2],
              'hidden_dim': [10, 20],
              'dropout': [0.1, 0.2],
              'n_epochs': [10, 20],
              'lr': [0.003],
              'training_length': [20],
              'input_chunk_length': [15]}

grid = ParameterGrid(param_grid)
len(list(grid))

# Parameter Tuning
rmse = []

# Loop
for params in grid:
  # Build the model

  model = RNNModel(model="LSTM",
                  hidden_dim = params['hidden_dim'],
                  n_rnn_layers = params['n_rnn_layers'],
                  dropout = params['dropout'],
                  n_epochs = params['n_epochs'],
                  optimizer_kwargs = {"lr": 0.003},
                  random_state = 42,
                  training_length = 20,
                  input_chunk_length = 15,
                  pl_trainer_kwargs = {"accelerator": "gpu", "devices": [0]})

  # Fit the model to the data
  model.fit(y_transformed, future_covariates = covariates_transformed)

  # CV
  cv = model.historical_forecasts(series = y_transformed,
                                future_covariates = covariates_transformed,
                                start = df.shape[0] - 180,
                                forecast_horizon = 31,
                                stride = 16,
                                retrain = True,
                                last_points_only = False)
  # Measure and store the error
  # Store the results
  rmse_cv = []

  for i in range(len(cv)):

    # Compute the RMSE for the CV
    # Get the forecast as a timeseries object
    ts = transformer1.inverse_transform(cv[i])

    # Manually convert to a pandas Series (values + time index)
    predictions = pd.Series(ts.values().squeeze(), index = ts.time_index)
    # this is deprecated --> predictions = transformer1.inverse_transform(cv[i]).to_pandas()

    # Actual values
    start = predictions.index.min()
    end = predictions.index.max()

    actuals = df.y[start:end]

    # Compute the error
    error_cv = mean_squared_error(actuals, predictions) ** 0.5 # squared = False doesn't work for some reason so I used ** 0.5

    # Save the error
    rmse_cv.append(error_cv)

  error = np.mean(rmse_cv)
  rmse.append(error)


# Parameter Tuning outcome
tuning_results = pd.DataFrame(grid)
tuning_results['rmse'] = rmse
tuning_results

# Exporting the tuned parameters
best_params = tuning_results[tuning_results.rmse == tuning_results.rmse.min()].transpose()
best_params.to_csv('Forecasting Product/best_params_XXX.csv')

# Isolate the params
n_rnn_layers = int(best_params.loc['n_rnn_layers'])
hidden_dim = int(best_params.loc['hidden_dim'])
dropout = float(best_params.loc['dropout'])

# Parameter tuning part 2

# Grid
param_grid = {'n_rnn_layers': [n_rnn_layers],
              'hidden_dim': [hidden_dim],
              'dropout': [dropout],
              'n_epochs': [10, 20],
              'lr': [0,001, 0.003],
              'training_length': [20, 30],
              'input_chunk_length': [15, 29]}

grid = ParameterGrid(param_grid)
len(list(grid))

# Parameter Tuning
rmse = []

# Loop
for params in grid:
  # Build the model

  model = RNNModel(model="LSTM",
                  hidden_dim = params['hidden_dim'],
                  n_rnn_layers = params['n_rnn_layers'],
                  dropout = params['dropout'],
                  n_epochs = params['n_epochs'],
                  optimizer_kwargs = {"lr": params['lr']},
                  random_state = 42,
                  training_length = params['training_length'],
                  input_chunk_length = params['input_chunk_length'],
                  pl_trainer_kwargs = {"accelerator": "gpu", "devices": [0]})

  # Fit the model to the data
  model.fit(y_transformed, future_covariates = covariates_transformed)

  # CV
  cv = model.historical_forecasts(series = y_transformed,
                                future_covariates = covariates_transformed,
                                start = df.shape[0] - 180,
                                forecast_horizon = 31,
                                stride = 16,
                                retrain = True,
                                last_points_only = False)
  # Measure and store the error
  # Store the results
  rmse_cv = []

  for i in range(len(cv)):

    # Compute the RMSE for the CV
    # Get the forecast as a timeseries object
    ts = transformer1.inverse_transform(cv[i])

    # Manually convert to a pandas Series (values + time index)
    predictions = pd.Series(ts.values().squeeze(), index = ts.time_index)
    # this is deprecated --> predictions = transformer1.inverse_transform(cv[i]).to_pandas()

    # Actual values
    start = predictions.index.min()
    end = predictions.index.max()

    actuals = df.y[start:end]

    # Compute the error
    error_cv = mean_squared_error(actuals, predictions) ** 0.5 # squared = False doesn't work for some reason so I used ** 0.5

    # Save the error
    rmse_cv.append(error_cv)

  error = np.mean(rmse_cv)
  rmse.append(error)


# Parameter Tuning outcome
tuning_results = pd.DataFrame(grid)
tuning_results['rmse'] = rmse
tuning_results

# Exporting the tuned parameters
best_params = tuning_results[tuning_results.rmse == tuning_results.rmse.min()].transpose()
best_params.to_csv("Forecasting Product/best_params/lstm.csv")
