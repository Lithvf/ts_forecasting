"""Script with classes to forecast on unknown states with trained models."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from src.features import CreateFeatures


class RecursiveForecaster:
    def __init__(self, model, initial_data, y_test, test, steps, plot=False):
        self.model = model
        self.initial_data = initial_data
        self.y_test = y_test
        self.test = test
        self.steps = steps
        self.plot = plot
        self.stats, self.forecasts = self.forecaster()
        self.metrics()
        self.plot_predict()

    def forecaster(self):
        history = self.initial_data.copy()
        stats = pd.DataFrame()
        forecasts = []

        for i in range(1, self.steps):
            X_last = CreateFeatures(history, target="PJME_MW").X.iloc[[-1]]
            stats = pd.concat([stats, X_last])

            y_pred = self.model.predict(X_last)[0]
            forecasts.append(y_pred)

            next_index = history.index[-1] + pd.Timedelta(hours=1)
            next_row = pd.DataFrame({"PJME_MW": [y_pred]}, index=[next_index])
            history = pd.concat([history, next_row])
        stats["forecast"] = forecasts
        return stats, forecasts

    def metrics(self):
        self.y_test = self.y_test.iloc[: self.steps - 1]
        rmse = np.sqrt(mean_squared_error(self.y_test, self.forecasts))
        r2 = r2_score(self.y_test, self.forecasts)
        mape = np.mean(np.abs((self.y_test - self.forecasts) / self.y_test)) * 100
        print(
            f"Result forecast set:\nRMSE = {rmse:0.2f}, R2 = {r2:0.2f}, MAPE = {mape:0.2f}"
        )

    def plot_predict(self):
        if self.plot == True:
            self.test = self.test.iloc[: self.steps - 1]
            self.test["Forecast"] = self.forecasts
            self.test.plot()
            plt.show()


class MultiHorizonForecaster:
    def __init__(self, model, history_data, horizon):
        """
        Parameters:
        - model: trained MultiOutputRegressor
        - history_data: DataFrame with latest known time series values
        - y_test: true multi-step targets for evaluation
        - test: original test set for plotting
        - steps: number of steps to forecast (should match training horizon)
        - plot: whether to plot forecast
        """
        self.model = model
        self.history_data = history_data.copy()
        self.horizon = horizon
        self.forecasts = self.forecaster()

    def forecaster(self):
        cf = CreateFeatures(
            self.history_data,
            target="PJME_MW",
            purpose="inference",
            horizon=self.horizon,
        )
        X_latest = cf.X.iloc[[-1]]
        timestamp = X_latest.reset_index()["datetime"].iloc[0]
        y_preds = self.model.predict(X_latest)[0]
        datetime_index = pd.date_range(
            start=timestamp + pd.Timedelta(hours=1),
            periods=self.horizon,
            freq="h",
        )
        y_preds = pd.Series(y_preds)
        df_forecast = y_preds.to_frame(name=f"Prediction {timestamp}")
        df_forecast["datetime"] = datetime_index
        df_forecast.set_index("datetime", inplace=True)
        return df_forecast
