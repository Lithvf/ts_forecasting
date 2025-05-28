"""Script with class to engineer features."""

import pandas as pd


class CreateFeatures:
    """
    A class to generate time-based and lag features for time-series forecasting.
    It prepares the data into X (features) and y (target) for model training.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        lag_features: list,
        target: str,
        purpose: str,
        horizon: int = 1,
    ):
        """
        Initializes the CreateFeatures instance and builds the features.

        Args:
            data (pd.DataFrame): The input DataFrame.
            target (str): The name of the target column in the DataFrame.
            horizon (int, optional): The number of future steps to forecast.
        """
        self.data = data.copy()
        self.lags = lag_features.copy()
        self.target = target
        self.purpose = purpose
        self.horizon = horizon
        if horizon < 1:
            raise ValueError("Horizon must be a positive integer (>= 1).")
        self.X, self.y = self._build_features()

    def _build_features(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Internal method to orchestrate the feature creation process.
        """
        self._add_time_features()
        self._add_lag_features(self.lags)
        self._create_multi_horizon_targets()
        if self.purpose == "training":
            self.data.dropna(inplace=True)
        return self._split_X_y()

    def _add_time_features(self):
        """
        Adds common time-based features to the DataFrame based on its datetime index.
        """
        self.data.set_index("datetime", inplace=True)
        self.data["hour"] = self.data.index.hour
        self.data["dayofweek"] = self.data.index.dayofweek
        self.data["quarter"] = self.data.index.quarter
        self.data["month"] = self.data.index.month
        self.data["dayofyear"] = self.data.index.dayofyear
        self.data["dayofmonth"] = self.data.index.day
        self.data["weekofyear"] = self.data.index.isocalendar().week.astype(int)

    def _add_lag_features(self, lags: list[int] = None):
        """
        Adds lag features for the target variable.
        """
        # TODO: Create config that has a list of lag features
        if lags is None:
            lags = list(range(0, 8)) + [24, 24 * 7]

        for lag in lags:
            self.data[f"lag_{lag}_h"] = self.data[self.target].shift(lag)

    def _create_multi_horizon_targets(self):
        """
        Creates multiple target columns for multi-step forecasting.
        """
        for h in range(1, self.horizon + 1):
            self.data[f"target_t+{h}"] = self.data[self.target].shift(-h)

    def _split_X_y(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the DataFrame into features (X) and target(s) (y).
        """
        feature_columns = [
            col
            for col in self.data.columns
            if col != self.target and not col.startswith("target_t+")
        ]

        X = self.data[feature_columns]

        if self.horizon > 1:
            y_cols = [f"target_t+{h}" for h in range(1, self.horizon + 1)]
            y = self.data[y_cols]
        else:
            y = self.data["target_t+1"]

        return X, y

    def get_features_and_target(self):
        """
        Returns the prepared features (X) and target(s) (y) DataFrames.
        """
        return self.X, self.y
