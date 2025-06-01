from src.utils import EnergyUsageData, SplitData
from src.features import CreateFeatures
from src.models import Model
from src.helpers import MetricsReporting
import yaml
import mlflow
import pandas as pd

# TODO break the script up in multiple files


def load_config(path):
    """
    Loads configuration from a YAML file.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def unpack_config(config, run):
    """
    Extracts necessary configuration values and adds run_id.

    Args:
        config (dict): Full configuration dictionary.
        run (mlflow): MLflow run object.

    Returns:
        dict: Reduced config with relevant parameters and run ID.
    """
    return {
        "run_id": run.info.run_id,
        "model_name": config["model"]["name"],
        "grid_params": config["model"]["grid_params"],
        "lag_features": config["features"]["lag_features"],
        "timestamp_split": config["split"]["timestamp_start"],
        "horizon": config["forecasting"]["horizon"],
    }


def log_params(config_dict):
    """
    Logs model and training parameters to MLflow.

    Args:
        config_dict (dict): Dictionary containing config parameters.
    """
    mlflow.log_param("model_name", config_dict["model_name"])
    mlflow.log_param("horizon", config_dict["horizon"])
    mlflow.log_params(config_dict["grid_params"])
    mlflow.log_param("lag_features", str(config_dict["lag_features"]))


def train_model(config_dict, purpose):
    """
    Trains the model on the training split.

    Args:
        config_dict (dict): Configuration dictionary.
        purpose (str).

    Returns:
        Model: Trained model object.
    """
    raw_data = EnergyUsageData(path="data/PJME_hourly.csv").get_data()
    split_data = SplitData(raw_data, config_dict["timestamp_split"], plot=False)
    X_train, y_train = CreateFeatures(
        split_data.train,
        config_dict["lag_features"],
        target="PJME_MW",
        purpose=purpose,
        horizon=config_dict["horizon"],
    ).get_features_and_target()

    trained_model = Model(config_dict["model_name"])
    trained_model.train(X_train, y_train, param_grid=config_dict["grid_params"])
    return trained_model


def log_results(y_pred, y_actual, run_id, purpose):
    """
    Calculates metrics, visualizes them, and logs to MLflow.

    Args:
        y_pred (pd.DataFrame): Model predictions.
        y_actual (pd.DataFrame): Actual target values.
        run_id (str): MLflow run ID for logging.
        purpose (str).
    """
    reporting = MetricsReporting(y_pred, y_actual, purpose=purpose)
    metrics = reporting.calculate_metrics()
    reporting.visualize_metrics(run_id=run_id, save=True)

    mlflow.log_metric(f"rmse_{purpose}_overall", float(metrics["rmse_overall"]))
    mlflow.log_metric(f"r2_{purpose}_overall", metrics["r2_overall"])
    mlflow.log_artifact(f"plots/metrics_{purpose}_horizons_{run_id}.png")


def test_model(trained_model, config_dict, purpose):
    """
    Applies a trained model to the test set and returns predictions.

    Args:
        trained_model (Model): Trained model wrapper.
        config_dict (dict): Configuration dictionary.
        purpose (str).

    Returns:
        tuple: (Predictions, Ground truth values) as DataFrames.
    """
    raw_data = EnergyUsageData(path="data/PJME_hourly.csv").get_data()
    split_data = SplitData(raw_data, config_dict["timestamp_split"], plot=False)
    X_test, y_test = CreateFeatures(
        split_data.test,
        config_dict["lag_features"],
        target="PJME_MW",
        purpose=purpose,
        horizon=config_dict["horizon"],
    ).get_features_and_target()

    y_pred_test = trained_model.trained_model.predict(X_test)
    y_pred_test = pd.DataFrame(y_pred_test, index=y_test.index)
    return y_pred_test, y_test


def main_train(config_path, purpose):
    """
    Orchestrates the training workflow including MLflow logging.

    Args:
        config_path (str): Path to the configuration YAML.
        purpose (str).

    Returns:
        Trained model and config dictionary.
    """
    config = load_config(config_path)
    mlflow.set_experiment("Multi horizon - Energy forecast")

    with mlflow.start_run() as run:
        config_dict = unpack_config(config, run)
        log_params(config_dict)
        trained_model = train_model(config_dict, purpose=purpose)
        log_results(
            trained_model.y_pred_train,
            trained_model.y_train,
            purpose=purpose,
            run_id=config_dict["run_id"],
        )
    return trained_model, config_dict


def main_test(trained_model, config_dict, purpose):
    """
    Orchestrates model testing and logs results to the same MLflow run.

    Args:
        trained_model (Model): Model object from training.
        config_dict (dict): Configuration used in training.
        purpose (str).
    """
    mlflow.set_experiment("Multi horizon - Energy forecast")

    with mlflow.start_run(run_id=config_dict["run_id"]) as run:
        y_pred, y_actual = test_model(trained_model, config_dict, purpose=purpose)
        log_results(
            y_pred,
            y_actual,
            purpose=purpose,
            run_id=config_dict["run_id"],
        )


if __name__ == "__main__":
    trained_model, config_dict = main_train(
        purpose="training", config_path="config.yaml"
    )
    main_test(trained_model, config_dict, purpose="testing")
