from src.utils import EnergyUsageData, SplitData
from src.features import CreateFeatures
from src.models import Model
from src.helpers import MetricsReporting, PlotPredictionActual
from src.forecast import MultiHorizonForecaster
from xgboost import XGBRegressor
import yaml
import mlflow


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config("config.yaml")
    mlflow.set_experiment("EnergyForecasting_MultiHorizon")

    with mlflow.start_run():
        model_name = config["model"]["name"]
        grid_params = config["model"]["grid_params"]
        lag_features = config["features"]["lag_features"]
        timestamp_split = config["split"]["timestamp_start"]
        horizon = config["forecasting"]["horizon"]

        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("horizon", horizon)
        mlflow.log_params(grid_params)
        mlflow.log_param("lag_features", str(lag_features))

        raw_data = EnergyUsageData(path="data/PJME_hourly.csv").get_data()
        split_data = SplitData(raw_data, timestamp_split, plot=False)
        X_train, y_train = CreateFeatures(
            split_data.train,
            lag_features,
            target="PJME_MW",
            purpose="training",
            horizon=horizon,
        ).get_features_and_target()

        trained_model = Model(XGBRegressor)
        trained_model.train(X_train, y_train)

        metrics = MetricsReporting(
            trained_model.y_pred_train, trained_model.y_train, purpose="train"
        ).calculate_metrics()

        print(metrics)

        # Log overall metrics
        mlflow.log_metric("rmse_overall", float(metrics["rmse_overall"]))
        mlflow.log_metric("r2_overall", metrics["r2_overall"])

        # Log per-horizon metrics
        for i, horizon in enumerate(metrics["horizon"]):
            mlflow.log_metric(f"rmse_{horizon}", float(metrics["rmse_per_horizon"][i]))
            mlflow.log_metric(f"r2_{horizon}", metrics["r2_per_horizon"][i])
            mlflow.log_metric(f"mape_{horizon}", float(metrics["mape_per_horizon"][i]))

    # reporting = MetricsReporting(
    #     trained_model.y_pred_train, trained_model.y_train, purpose="train"
    # ).visualize_metrics()

    # PlotPredictionActual(
    #     split_data.train, trained_model.y_pred_train, "01-11-2014"
    # ).plot_sample()
