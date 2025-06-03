# Multi-horizon forecasting of timeseries - energy usage

This project uses machine learning (XGBoost model) to forecast hourly energy usage for the PJM Interconnection based on this Kaggle dataset https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption.  

The project contains an automatic pipeline for training and testing. It supports multi-horizon forecasting (up to x hours ahead) and tracks experiments with MLflow, currently for training and testing solely the XGBoostRegressor is included. This can be expanded with other models if and only if they are compatible with skicit-learn GridSearchCV.  
The model is trained on lag features, see config.yaml for the lags included. Gridsearch is applied to find the best model, see the gridsearch parameters in config.yaml.  

Todo: Add an inference pipeline, the functions are already included to conduct inference based on a registered model.  

Next step (software): Set workflow up with Airflow or Snowflake  

Next step (model): Create probabilistic forecasts

## MLflow Experiment Summary

**Experiment Name:** `Multi horizon - Energy forecast`  
**Best Model:** `XGBRegressor`  
**Horizon:** `24 hours`  
**Best Run ID:** `ee62e28759324fb6a84f77e69a09bc1f`  
**Logged in MLflow**: -  

Average performance per horizon on the test dataset RMSE, R2, MAPE:  


![image](https://github.com/user-attachments/assets/2d546dec-e53a-44c7-932d-f1e949e78ec3)

## Project structure
```bash
ts_forecasting/  
├── README.md # Project overview and instructions  
├── config.yaml # Configuration for model and features  
├── data/  
│ └── PJME_hourly.csv # Input dataset  
├── pipeline_train_evaluate.py # Script to train and evaluate the model  
├── poetry.lock # Poetry lock file (dependencies)  
├── pyproject.toml # Project and dependency configuration  
├── scripts/ #To add  
│ └── pipeline_inference.py # To add  
├── src/  
│ ├── features.py # Feature engineering functions  
│ ├── forecast.py # Forecasting logic  
│ ├── helpers.py # Metrics and plotting tools  
│ ├── models.py # Model training and abstraction  
│ └── utils.py # Data loading and preprocessing utilities  
└── test.py # Quick script for testing functionality
```

## Installation

```bash
git clone https://github.com/Lithvf/ts_forecasting.git
pip install poetry
poetry env use python
poetry shell
poetry install 
