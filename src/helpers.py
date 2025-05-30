"""Script with helper classes/functions."""

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MetricsReporting:
    def __init__(self, y_pred: pd.DataFrame, y_true: pd.DataFrame, purpose: str):
        y_true = y_true.copy()
        y_pred = y_pred.copy()
        if purpose == "test":
            y_true, y_pred = self._align_indices(y_true, y_pred)
        self.y_pred = np.asarray(y_pred)
        self.y_true = np.asarray(y_true)
        if self.y_pred.ndim == 2 and self.y_pred.shape[1] == 1:
            self.y_pred = self.y_pred.flatten()
        self.metrics = {}
        self.sample = {}

    def _align_indices(self, y_true, y_pred):
        y_true.set_index("datetime", inplace=True)
        y_true = y_true.loc[y_pred.index]
        return y_true, y_pred

    def calculate_metrics(self):
        """Internal method to calculate all relevant metrics."""

        if self.y_pred.ndim > 1:
            num_horizons = self.y_pred.shape[1]
            self.metrics["rmse_overall"] = np.sqrt(
                mean_squared_error(self.y_true, self.y_pred)
            )
            self.metrics["r2_overall"] = r2_score(self.y_true, self.y_pred)
            self.metrics["horizon"] = []
            self.metrics["rmse_per_horizon"] = []
            self.metrics["r2_per_horizon"] = []
            self.metrics["mape_per_horizon"] = []

            self.sample["horizon"] = []
            self.sample["prediction"] = []
            self.sample["actual"] = []

            for i in range(num_horizons):
                self.metrics["horizon"].append(f"t{i + 1}")
                y_true_h = self.y_true[:, i]
                y_pred_h = self.y_pred[:, i]

                valid_mask = ~np.isnan(y_true_h) & ~np.isnan(y_pred_h)
                y_true_h_clean = y_true_h[valid_mask]
                y_pred_h_clean = y_pred_h[valid_mask]

                if len(y_true_h_clean) > 0:
                    self.metrics["rmse_per_horizon"].append(
                        np.sqrt(mean_squared_error(y_true_h_clean, y_pred_h_clean))
                    )
                    self.metrics["r2_per_horizon"].append(
                        r2_score(y_true_h_clean, y_pred_h_clean)
                    )

                    abs_percentage_error = (
                        np.abs(
                            (y_true_h_clean - y_pred_h_clean)
                            / np.where(y_true_h_clean == 0, np.nan, y_true_h_clean)
                        )
                        * 100
                    )
                    self.metrics["mape_per_horizon"].append(
                        np.nanmean(abs_percentage_error)
                    )

                else:
                    self.metrics["rmse_per_horizon"].append(np.nan)
                    self.metrics["r2_per_horizon"].append(np.nan)
                    self.metrics["mape_per_horizon"].append(np.nan)
        else:
            self.metrics["rmse"] = [
                np.sqrt(mean_squared_error(self.y_true, self.y_pred))
            ]
            self.metrics["r2"] = [r2_score(self.y_true, self.y_pred)]
            self.metrics["mape"] = [
                np.mean(np.abs((self.y_true - self.y_pred) / self.y_true)) * 100
            ]

        return self.metrics

    def visualize_metrics(self):
        metrics = self.calculate_metrics()
        self.metrics_df = pd.DataFrame(metrics)
        print(self.metrics_df)
        if self.y_pred.ndim <= 1:
            print(
                "One dimensional reporting data is not plotted, look at the logging instead."
            )
        else:
            fig, axes = plt.subplots(1, 3, figsize=(24, 6))
            axes[0].plot(
                self.metrics_df[["horizon", "rmse_per_horizon"]].set_index("horizon"),
            )
            axes[0].set_title("RMSE")
            axes[0].set_ylabel("MW")
            axes[1].plot(
                self.metrics_df[["horizon", "r2_per_horizon"]].set_index("horizon"),
            )
            axes[1].set_title("R2")
            axes[1].set_ylabel("Per unit")
            axes[2].plot(
                self.metrics_df[["horizon", "mape_per_horizon"]].set_index("horizon"),
            )
            axes[2].set_title("MAPE")
            axes[2].set_ylabel("Percentage")

            for ax in axes:
                ax.tick_params(axis="x", labelsize=8)
                for label in ax.get_xticklabels():
                    label.set_rotation(90)

            plt.tight_layout()
            plt.show()


class PlotPredictionActual:
    def __init__(
        self, actual: pd.DataFrame, y_pred: pd.DataFrame, timestamp: str = None
    ):
        self.actual = actual.copy()
        self.actual.set_index("datetime", inplace=True)
        self.y_pred = y_pred
        if timestamp:
            self.timestamp = pd.to_datetime(timestamp)
        else:
            self.timestamp = self.y_pred.reset_index()["datetime"].iloc[0]

    def restructure_prediction(self):
        filtered_y_pred = self.y_pred.loc[[self.timestamp]]
        transformed_pred = filtered_y_pred.iloc[0].T
        num_rows = len(transformed_pred)
        new_index = pd.date_range(
            start=self.timestamp + pd.Timedelta(hours=1), periods=num_rows, freq="h"
        )
        df_predict = transformed_pred.to_frame(name=f"Prediction {self.timestamp}")
        df_predict["datetime"] = new_index
        df_predict.set_index("datetime", inplace=True)
        return df_predict

    def plot_sample(self):
        if self.y_pred.shape[1] > 1:
            pred = self.restructure_prediction()
        else:
            pred = self.y_pred
        if not self.actual.index.intersection(pred.index).empty:
            actual_prediction = pd.merge(
                self.actual,
                pred,
                left_index=True,
                right_index=True,
                how="inner",
            )
            actual_prediction.plot()
        else:
            pred.plot()
        plt.show()
