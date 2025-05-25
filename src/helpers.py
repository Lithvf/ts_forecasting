"""Script with helper classes/functions."""

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


class MetricsReporting:
    def __init__(self, y_pred, y_true):
        self.y_pred = np.asarray(y_pred)
        self.y_true = np.asarray(y_true)
        self.metrics = {}

    # def _calculate_metrics(self, dim):
    #     if dim == "multi":
    #         self.rmse = np.sqrt(mean_squared_error(self.y_true, self.y_pred))
    #         self.r2 = r2_score(self.y_true, self.y_pred)
    #     else:
    #         pass

    def _calculate_metrics(self):
        """Internal method to calculate all relevant metrics."""

        self.metrics["rmse"] = np.nan
        self.metrics["r2"] = np.nan
        self.metrics["mape"] = np.nan

        if self.y_pred.ndim > 1:
            num_horizons = self.y_pred.shape[1]
            self.metrics["rmse_overall"] = np.sqrt(
                mean_squared_error(self.y_true, self.y_pred)
            )
            self.metrics["r2_overall"] = r2_score(self.y_true, self.y_pred)

            self.metrics["rmse_per_horizon"] = []
            self.metrics["r2_per_horizon"] = []
            self.metrics["mape_per_horizon"] = []

            for i in range(num_horizons):
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
        print(self.metrics)
        # print(f"RMSE = {self.rmse:0.2f}")
        # print(f"R2 = {self.r2:0.2f}")


# class ForecastPlotter:
#     pass
