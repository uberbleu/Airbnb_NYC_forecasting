from google.colab import drive
drive.mount('/content/drive')

# Libraries and data

%cd /content/drive/MyDrive/Data Analysis Science/Time Series Forecasting Product

# Install Greykite
!pip install greykite

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from greykite.framework.templates.autogen.forecast_config import *
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.common.features.timeseries_features import *
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.framework.utils.result_summary import summarize_grid_search_results
from plotly.offline import iplot

import yaml
from greykite.common.constants import DETAILED_SEASONALITY_COMPONENTS_REGEX_DICT

# Load the data
# YYYY-MM-DD
df = pd.read_csv('nyc_data.csv')
future_df = pd.read_csv('future.csv')
future_df.head()

# Merging both
df = pd.concat([df, future_df])
df = df.reset_index(drop = True)
len(df)

# Inspecting df
df.tail()

# Rename variable
df = df.rename(columns = {'Demand': 'y'})
df.head(0)

# Silverkite Preparations

# Load the tuned parameters
parameters = pd.read_csv("/content/drive/MyDrive/Data Analysis Science/Time Series Forecasting Product/Forecasting Product/best_params_silverkite.csv",
                         index_col = 0)
parameters

# Isolate the tuned parameters
growth_term_param = parameters.loc["param_estimator__growth_term"][0]
fit_algorithm_param = parameters.loc["param_estimator__fit_algorithm_dict"][0]

# Specifying time series names
metadata = MetadataParam(time_col = "Date",
                        value_col = "y",
                        freq = "D",
                        train_end_date = pd.to_datetime("2020-12-31"))
metadata

# Growth terms possibilitites
growth = dict(growth_term = growth_term_param)
growth

# Seasonalities
seasonality = dict(yearly_seasonality = "auto",
                   quarterly_seasonality = "auto",
                   monthly_seasonality = "auto",
                   weekly_seasonality = "auto",
                   daily_seasonality = "auto")
seasonality

# Checking which countries are available and their holidays
get_available_holiday_lookup_countries(["US"])
get_available_holidays_across_countries(countries = ["US"],
                                        year_start = 2015,
                                        year_end = 2021)

# Specifying events
events = dict(holidays_to_model_separately = ["New Year's Day"],
              holiday_lookup_countries = ["US"],
              holiday_pre_num_days = 2,
              holiday_post_num_days = 2,
              holiday_pre_post_num_dict = {"New Year's Day": (3, 1)},
              daily_event_df_dict = {"elections": pd.DataFrame({
                  "date": ["2016-11-8", "2020-11-03"],
                  "event_name": ["elections"] * 2
              })})

events

# Changepoints -> reflects the changes in the trend
changepoints = dict(changepoints_dict = dict(method = "auto"))
changepoints

# Regressors
regressors = dict(regressor_cols = ["Easter", "Temperature", "Marketing"])
regressors

# Lagged Regressors
lagged_regressors = dict(lagged_regressor_dict = {"Temperature": "auto",
                                                  "Easter": "auto",
                                                  "Marketing": "auto"})

# Autoregression > dependent on the forecasting horizon
autoregression = dict(autoreg_dict = "auto")

# Fitting algorithms
import yaml
custom = dict(fit_algorithm_dict = yaml.load(fit_algorithm_param,
                                             Loader = yaml.SafeLoader))
custom

# Silverkite model

# Build the model
model_components = ModelComponentsParam(growth = growth,
                                        seasonality = seasonality,
                                        events = events,
                                        changepoints = changepoints,
                                        regressors = regressors,
                                        lagged_regressors = lagged_regressors,
                                        autoregression = autoregression,
                                        custom = custom)

 # Cross-validation
 # df.shape[0] - 180 - 31 which is the future_df.
 evaluation_period = EvaluationPeriodParam(cv_min_train_periods= df.shape[0] - 180 - 31,
                                           cv_expanding_window = True,
                                           cv_max_splits = 50,
                                           cv_periods_between_splits = 16)


# Evaluation metric
evaluation_metric = EvaluationMetricParam(cv_selection_metric = EvaluationMetricEnum.RootMeanSquaredError.name)

# Configuration
config = ForecastConfig(model_template = ModelTemplateEnum.SILVERKITE.name,
                         forecast_horizon = 31,
                         metadata_param = metadata,
                         model_components_param = model_components,
                         evaluation_period_param = evaluation_period,
                         evaluation_metric_param = evaluation_metric)

# Forecasting
forecaster = Forecaster()
result = forecaster.run_forecast_config(df = df,
                                        config = config)

# Visualization
fig = result.backtest.plot()
iplot(fig)

# Look at summary
print(result.model[-1].summary()) # -1 retrieves just the estimators

# Components
fig = result.forecast.plot_components(grouping_regex_patterns_dict = DETAILED_SEASONALITY_COMPONENTS_REGEX_DICT)
iplot(fig)

# Retrieving the forecast
forecast = result.forecast.df[['ts', 'forecast']]
forecast = forecast.rename(columns = {'forecast': 'silverkite'})
forecast_silverkite = forecast.iloc[-len(future_df):, :]
forecast_silverkite

# Exporting
# forecast_silverkite.to_csv("Forecasting Product/Ensemble/predictions_silverkite.csv")
