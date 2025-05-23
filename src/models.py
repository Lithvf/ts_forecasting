"""Script with classes to train, test and forecast models."""

from src.features import CreateFeatures

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


class FitModel:
    def __init__(self, model_cls, X_train, y_train, train, param_grid=None, plot=False):
        self.model = model_cls
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.train = train.copy()
        self.param_grid = param_grid
        self.plot = plot
        if self.param_grid:
            self.fit_gridsearch()
        else:
            self.fit()
        self.metrics()
        self.plot_fit()

    def fit(self):
        self.trained_model = self.model.fit(X=self.X_train, y=self.y_train)
        self.y_pred_train = self.trained_model.predict(self.X_train)

    def fit_gridsearch(self):
        tscv = TimeSeriesSplit(n_splits=3)

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            scoring="neg_mean_squared_error",
            cv=tscv,
            verbose=1,
            n_jobs=-1,
        )
        grid_search.fit(self.X_train, self.y_train)
        self.trained_model = grid_search.best_estimator_
        self.y_pred_train = self.trained_model.predict(self.X_train)

    def metrics(self):
        rmse = np.sqrt(mean_squared_error(self.y_train, self.y_pred_train))
        r2 = r2_score(self.y_train, self.y_pred_train)
        mape = np.mean(np.abs((self.y_train - self.y_pred_train) / self.y_train)) * 100
        print(
            f"Result training set:\nRMSE = {rmse:0.2f}, R2 = {r2:0.2f}, MAPE = {mape:0.2f}"
        )

    def plot_fit(self):
        if self.plot == True:
            common_index = self.train.index.intersection(self.y_train.index)
            self.train = self.train.loc[common_index]
            self.train["Predicted training"] = self.y_pred_train
            self.train.plot()


def grouped_mape(y_true, y_pred, time_index, group="dayofweek"):
    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "hour": time_index.hour,
            "dayofweek": time_index.dayofweek,
        }
    )
    return df.groupby(group).apply(
        lambda x: np.mean(np.abs((x.y_true - x.y_pred) / x.y_true)) * 100
    )


class PredictModel:
    def __init__(self, trained_model, X_test, y_test, test, plot=True):
        self.trained_model = trained_model
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.test = test.copy()
        self.plot = plot
        self.predict()
        self.metrics()
        self.plot_predict()

    def predict(self):
        self.y_pred = self.trained_model.predict(self.X_test)

    def metrics(self):
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        r2 = r2_score(self.y_test, self.y_pred)
        mape = np.mean(np.abs((self.y_test - self.y_pred) / self.y_test)) * 100
        print(
            f"Result test set:\nRMSE = {rmse:0.2f}, R2 = {r2:0.2f}, MAPE = {mape:0.2f}"
        )
        # print(grouped_mape(self.y_test, self.y_pred, X_test.index))

    def plot_predict(self):
        if self.plot == True:
            common_index = self.test.index.intersection(self.y_test.index)
            self.test = self.test.loc[common_index]
            self.test["Prediction"] = self.y_pred
            self.test.iloc[:48].plot()
            plt.show()
