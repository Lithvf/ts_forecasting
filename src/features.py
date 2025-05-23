"""Script with class to engineer features."""


class CreateFeatures:
    def __init__(self, data, target):
        self.data = data.copy()
        self.target = target
        self.time_indicators()
        self.lag_features()
        self.X, self.y = self.split_X_y()

    def time_indicators(self):
        self.data["date"] = self.data.index
        self.data["hour"] = self.data["date"].dt.hour
        self.data["dayofweek"] = self.data["date"].dt.dayofweek
        self.data["quarter"] = self.data["date"].dt.quarter
        self.data["month"] = self.data["date"].dt.month
        self.data["dayofyear"] = self.data["date"].dt.dayofyear
        self.data["dayofmonth"] = self.data["date"].dt.day
        self.data["weekofyear"] = self.data["date"].dt.isocalendar().week

    def lag_features(self):
        self.data["lag_1_h"] = self.data[self.target].shift(1)
        self.data["lag_2_h"] = self.data[self.target].shift(2)
        self.data["lag_3_h"] = self.data[self.target].shift(3)
        self.data["lag_4_h"] = self.data[self.target].shift(4)
        self.data["lag_5_h"] = self.data[self.target].shift(5)
        self.data["lag_6_h"] = self.data[self.target].shift(6)
        self.data["lag_7_h"] = self.data[self.target].shift(7)
        self.data["lag_1_day"] = self.data[self.target].shift(24)
        self.data["lag_1_week"] = self.data[self.target].shift(24 * 7)
        self.data.dropna(inplace=True)

    def split_X_y(self):
        X = self.data[
            [
                "hour",
                "dayofweek",
                "quarter",
                "month",
                "dayofyear",
                "dayofmonth",
                "weekofyear",
                "lag_1_h",
                "lag_2_h",
                "lag_3_h",
                "lag_4_h",
                "lag_5_h",
                "lag_6_h",
                "lag_7_h",
                "lag_1_day",
                "lag_1_week",
            ]
        ]
        y = self.data[self.target]
        return X, y
