import argparse
import yaml
import mlflow
import pandas as pd
from src.utils import EnergyUsageData, SplitData
from src.helpers import MetricsReporting, PlotPredictionActual
from src.forecast import MultiHorizonForecaster


def load_config(path: str) -> dict:
    """Load YAML configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_inference(run_id: str, config_path: str) -> pd.DataFrame:
    """Run inference using a registered MLflow model."""
    config = load_config(config_path)
    horizon = config["forecasting"]["horizon"]
    timestamp_split = config["split"]["timestamp_start"]

    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    data = EnergyUsageData(path="data/PJME_hourly.csv").get_data()
    split_data = SplitData(data, timestamp_split, plot=False)

    forecaster = MultiHorizonForecaster(
        model, split_data.train, horizon=horizon
    )
    forecast_df = forecaster.forecasts

    metrics = MetricsReporting(
        forecast_df,
        split_data.test.head(horizon),
        purpose="inference",
    )
    print(metrics.calculate_metrics())

    PlotPredictionActual(
        split_data.test.head(horizon), forecast_df
    ).plot_sample()

    return forecast_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model inference")
    parser.add_argument("--run-id", required=True, help="MLflow run ID")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to configuration file"
    )
    args = parser.parse_args()

    run_inference(args.run_id, args.config)


if __name__ == "__main__":
    main()
