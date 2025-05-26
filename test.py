from src.utils import EnergyUsageData, SplitData
from src.features import CreateFeatures
from src.models import Model
from src.helpers import MetricsReporting
from xgboost import XGBRegressor
import pandas as pd


if __name__ == "__main__":
    timestamp = "01-Jan-2016"
    horizon = 48
    raw_data = EnergyUsageData(path="data/PJME_hourly.csv").get_data()
    split_data = SplitData(raw_data, timestamp, False)
    X_train, y_train = CreateFeatures(
        split_data.train, target="PJME_MW", horizon=horizon
    ).get_features_and_target()
    X_test, y_test = CreateFeatures(
        split_data.test, target="PJME_MW", horizon=horizon
    ).get_features_and_target()

    trained_model = Model(XGBRegressor)

    trained_model.train(X_train, y_train, split_data.train)

    reporting = MetricsReporting(trained_model.y_pred_train, trained_model.y_train)
    reporting.print()
    reporting.plot_metrics()
