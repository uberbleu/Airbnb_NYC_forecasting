# Airbnb NYC Forecasting

## Intro

In this project, we aim to predict future Airbnb demand in New York City by leveraging various forecasting models, including Prophet, Silverkite, RNN LSTM, and SARIMAX. By comparing the performance of these models, we hope to provide accurate insights into future booking trends, helping Airbnb hosts and stakeholders make informed decisions for optimizing occupancy and revenue.

Herein, only the Prophet model will be displayed with a final comparison amongst all the models. The rest of the work is listed below.
Full notebooks:
* EDA: https://colab.research.google.com/drive/14NZatkdotPVngiH4qsxRpy-dB1Hp8u70?usp=sharing
* Prophet: https://drive.google.com/file/d/1DvDFt-vdGq7ZA8O3h86SDUoIV5wD-J-4/view?usp=sharing
* Prophet hyperparameter tuning: https://colab.research.google.com/drive/1NAd9HWAVsDTLyWq9OZMqrrSrpo5b46Kl?usp=sharing
* Silverkite: https://colab.research.google.com/drive/1abF88ZNZbmigW0-N-GX2gyOmsuyzJlfm?usp=sharing
* Silverkite hyperparameter tuning: https://colab.research.google.com/drive/162kfl43PogjtbQyBThtEl45pJZ5VDA4C?usp=sharing
* RNN LSTM: https://drive.google.com/file/d/1aBaGn0NSsNquE_K57SKC-QTwYUCXDEql/view?usp=sharing
* RNN LSTM hyperparameter tuning: https://colab.research.google.com/drive/1KHyxQy_dLwdSot15mJ3UHfB9BuQxpoyq?usp=sharing
* SARIMAX: https://colab.research.google.com/drive/1KBS35PsAbLG-pU4pf0BlKHPpQSnwm5BU?usp=sharing
* SARIMAX hyperparameter tuning: https://colab.research.google.com/drive/1izp9bBWNbOUSR95wn59TJd-JdKZmjuYh?usp=sharing

Project link with graphs and more summaries:
https://huszartony.super.site/95c0b3a5d941461987c6b41872a2bab4

## 1. Problem Definition
In a statement, can we accurately forecast future Airbnb demand in NYC to help hosts optimize pricing, availability and occupancy rates?

## 2. Data
Past data: https://colab.research.google.com/drive/1izp9bBWNbOUSR95wn59TJd-JdKZmjuYh?usp=sharing
Future data: https://drive.google.com/file/d/1YoCjpls-n_STsdw3INegjUCP97smDUe1/view?usp=sharing

## 3. Evaluation
We aim to achieve a low Root Mean Squared Error (RMSE) to ensure accurate and reliable demand forecasts for Airbnb in NYC.

## 4. Features
* Date
* Demand
* Easter
* Thanksgiving
* Christmas
* Temperature
* Marketing

## Forecasting

Forecasting was achieved by leveraging Prophet for trend-based predictions, Silverkite for complex seasonality adjustments, SARIMAX for incorporating seasonal effects and external regressors, and LSTM for capturing intricate temporal dependencies and non-linear patterns in the data.

## Conclusion
The findings from the analysis and modeling include:

## 1. Seasonality and Trends: 
* Seasonal decomposition showed clear patterns and trends in the NYC demand data. Events such as Easter, Thanksgiving, and Christmas significantly affected demand.

## 2. Impact of External Factors: 
* The relationship between temperature, marketing efforts, and demand was examined, with regressors like temperature and marketing impacting demand.

## 3. Model Diversity:
* By incorporating forecasts from Prophet, SARIMAX, Silverkite, and LSTM, we harnessed the strengths of each model. This diversity helps capture different aspects of the time series data.

## 4. Error-Based Weighting:
* Weights were assigned to each model based on its error performance relative to the average error. This approach ensures that models with better performance have a higher influence on the final forecast.

## 5. Ensemble Effectiveness:
* The ensemble forecast combines individual predictions in a weighted manner, potentially leading to improved accuracy and robustness compared to any single model.

## 6. Results:
* The calculated weights reflect the relative performance of each model, with Prophet and Silverkite receiving slightly higher weights due to their lower errors. The total weight slightly exceeding 1 indicates a slight imbalance in the weight distribution but should not significantly affect the ensemble forecast's reliability.

## 7. Visualization:
* Plotting the ensemble forecast alongside individual model forecasts provides a clear visual comparison, showing how the ensemble forecast integrates the strengths of all models.

## Retrospection
## 1. Have we met our goal?
* We achieved our goal of a low RMSE score of 48 by optimizing individual models through hyperparameter tuning.

## 2.What did we learn from our experience?
* From this experience, we learned that integrating multiple forecasting models into an ensemble approach can significantly enhance predictive accuracy and reduce errors. 
* Additionally, meticulous hyperparameter tuning and error-based weighting are crucial for optimizing model performance and achieving reliable forecasts.

## 3.What are some future improvements?
* Future improvements could include exploring additional forecasting models and techniques to further enhance prediction accuracy, such as incorporating advanced machine learning algorithms or external data sources. 
* Refining the ensemble method with more sophisticated weighting strategies and dynamic adjustments could also optimize the model's performance. 
* Additionally, increasing the frequency and granularity of the data used for training could help capture more nuanced patterns and improve forecast precision.
