"""Script with helper classes/functions."""

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MetricsReporting:
    def __init__(self, y_pred, y_true):
        self.y_pred = np.asarray(y_pred)
        self.y_true = np.asarray(y_true)
        self.metrics = {}

    def _calculate_metrics(self):
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

            for i in range(num_horizons):
                self.metrics["horizon"].append(f"t+{i + 1}")
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

    def print(self):
        self._calculate_metrics()
        self.metrics_df = pd.DataFrame(self.metrics)
        print(self.metrics_df)

    def plot_metrics(self):
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

        plt.show()
        plt.tight_layout()
