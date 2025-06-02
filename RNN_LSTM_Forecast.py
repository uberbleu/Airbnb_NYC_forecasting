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
future_df = pd.read_csv('future.csv', index_col = 0, parse_dates = True)

df.head()

# Extract the regressors
X_train = df.iloc[:, 1:]
X_future = future_df.iloc[:, 1:]

# Merge the 2 inputs
X = pd.concat([X_train, X_future])

# Rename variable
df = df.rename(columns = {'Demand': 'y'})
df.head(0)

# Prepare for LSTM

# Creating Time Series
series = TimeSeries.from_series(df.y)
# Anything that is not a "y" is a covariate
covariates = TimeSeries.from_dataframe(X)

# Seasonality / trend
# Year
year_series = datetime_attribute_timeseries(pd.date_range(start = series.start_time(),
                                            freq = series.freq_str,
                                            periods = X.shape[0]),
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

# Scaling
transformer1 = Scaler()
transformer2 = Scaler()

# Scaling Y
y_transformed = transformer1.fit_transform(series)

# Stacking the covariates
covariates = covariates.stack(year_series)

# Scaling the covariates
covariates_transformed = transformer2.fit_transform(covariates)

# Stack the seasonal variables
covariates_transformed = covariates_transformed.stack(month_series)
covariates_transformed = covariates_transformed.stack(weekday_series)

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

# Load the tuned parameters
parameters = pd.read_csv("Forecasting Product/best_params_lstm.csv",
                         index_col = 0)
parameters

# Isolate the params
n_rnn_layers = int(parameters.loc['n_rnn_layers'])
hidden_dim = int(parameters.loc['hidden_dim'])
dropout = float(parameters.loc['dropout'])
lr = float(parameters.loc['lr'])
input_chunk_length = int(parameters.loc['input_chunk_length'])
n_epochs = int(parameters.loc['n_epochs'])
training_length = int(parameters.loc['training_length'])

# Build the LSTM model
model = RNNModel(model="LSTM",
                 hidden_dim = hidden_dim,
                 n_rnn_layers = n_rnn_layers,
                 dropout = dropout,
                 n_epochs = n_epochs,
                 optimizer_kwargs = {"lr": lr},
                 random_state = 42,
                 training_length = training_length,
                 input_chunk_length = input_chunk_length,
                 pl_trainer_kwargs = {"accelerator": "gpu", "devices": [0]})

# Fit the model to the data
model.fit(y_transformed,
          future_covariates = covariates_transformed)

# Forecasting and Exporting

predictions_lstm = model.predict(n = len(future_df),
              future_covariates = covariates_transformed)
predictions_lstm = transformer1.inverse_transform(predictions_lstm)
predictions_series = pd.Series(predictions_lstm.values().squeeze(), index=predictions_lstm.time_index).rename("LSTM")

# Exporting
# predictions_lstm.to_csv("Forecasting Product/Ensemble/predictions_lstm.csv ")
