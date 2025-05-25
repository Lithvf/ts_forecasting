To be updated:
This repository is made to apply different time series forecasting models on a Kaggle dataset that contains energy consumption data of PJM - East.

The XGBoost model insights:
The trained model generates on the training set: RMSE = 661.68, R2 = 0.99, MAPE = 1.50.
The trained model generates on the test set: RMSE = 677.68, R2 = 0.99, MAPE = 1.66.
With a forecasting horizon of 48 hours, the trained model forecasts recursively on the test data: RMSE = 6419.45, R2 = -0.72, MAPE = 18.09. See the plot for the forecast vs. actual data.
![Figure_1](https://github.com/user-attachments/assets/8528be87-5de4-44a7-888a-8c3f15dac090)

