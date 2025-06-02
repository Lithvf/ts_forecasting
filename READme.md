# Forecasting of multi-horizon timeseries - energy usage

This project uses machine learning (XGBoost model) to forecast hourly energy usage for the PJM Interconnection based on this Kaggle dataset https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption.  

The project contains an automatic pipeline for training and testing. It supports multi-horizon forecasting (up to x hours ahead) and tracks experiments with MLflow, currently for training and testing solely the XGBoostRegressor is included. This can be expanded with other models if and only if they are compatible with skicit-learn GridSearchCV.  
The model is trained on lag features, see config.yaml for the lags included. Gridsearch is applied to find the best model, see the gridsearch parameters in config.yaml.  

Todo: Add an inference pipeline, the functions are already included to conduct inference based on a registered model.  

## MLflow Experiment Summary

**Experiment Name:** `Multi horizon - Energy forecast`  
**Best Model:** `XGBRegressor`  
**Horizon:** `24 hours`  
**Best Run ID:** `ee62e28759324fb6a84f77e69a09bc1f`  
**Logged in MLflow**: -  

Average performance per horizon on the test dataset RMSE, R2, MAPE:  


![image](https://github.com/user-attachments/assets/2d546dec-e53a-44c7-932d-f1e949e78ec3)

## Installation

```bash
git clone https://github.com/Lithvf/ts_forecasting.git
poetry install
