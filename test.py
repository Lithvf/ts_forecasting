from src.utils import EnergyUsageData, SplitData
from src.features import CreateFeatures
from src.models import Model
from src.helpers import MetricsReporting, PlotPredictionActual
from src.forecast import MultiHorizonForecaster
from xgboost import XGBRegressor
import numpy as np

if __name__ == "__main__":
    timestamp = "06-Oct-2016"
    horizon = 24
    raw_data = EnergyUsageData(path="data/PJME_hourly.csv").get_data()
    split_data = SplitData(raw_data, timestamp, False)
    X_train, y_train = CreateFeatures(
        split_data.train, target="PJME_MW", purpose="training", horizon=horizon
    ).get_features_and_target()
    X_test, y_test = CreateFeatures(
        split_data.test, target="PJME_MW", purpose="training", horizon=horizon
    ).get_features_and_target()

    # param_grid = {
    #     "n_estimators": [100, 200],
    #     "learning_rate": [0.05, 0.1],
    #     "max_depth": [3, 5],
    # }

    trained_model = Model(XGBRegressor)
    trained_model.train(X_train, y_train)

    reporting = MetricsReporting(
        trained_model.y_pred_train, trained_model.y_train, purpose="train"
    ).visualize_metrics()

    PlotPredictionActual(
        split_data.train, trained_model.y_pred_train, "01-11-2014"
    ).plot_sample()

    forecaster = MultiHorizonForecaster(
        trained_model.trained_model, split_data.train, horizon=horizon
    )
    forecast = forecaster.forecaster()
    reporting = MetricsReporting(
        forecast, split_data.test, purpose="test"
    ).visualize_metrics()
    PlotPredictionActual(split_data.test, forecast).plot_sample()
