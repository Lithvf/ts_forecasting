"""Script with classes to train, test and forecast models."""

from xgboost import XGBRegressor
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


class Model:
    """
    Handles model training (including hyperparameter tuning based on params)
    and gives metrics for the trained model.
    """

    def __init__(self, model_cls):
        """
        Initializes the ModelHandler with a model class name or object.

        Args:
            model_cls (str or type): Model name string (e.g. "XGBRegressor") or a class.
            model_params (dict, optional): Parameters to initialize the model with.
        """
        self.model_registry = {
            "XGBRegressor": XGBRegressor,
        }

        if isinstance(model_cls, str):
            if model_cls not in self.model_registry:
                raise ValueError(f"Model '{model_cls}' is not supported.")
            model_cls = self.model_registry[model_cls]
        self.model_cls = model_cls
        self.trained_model = None
        self.metrics_results = {}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        param_grid: dict = None,
    ):
        """
        Trains the model. Performs GridSearchCV if param_grid is provided, else standard fit.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            param_grid (dict, optional): Dictionary of hyperparameters for GridSearchCV.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.model_instance = self.model_cls()
        if param_grid:
            self._fit_gridsearch(param_grid)
        else:
            self._fit_standard()

    def _fit_standard(self):
        """Internal method for standard model fitting."""
        self.trained_model = self.model_instance.fit(X=self.X_train, y=self.y_train)
        self.y_pred_train = self.trained_model.predict(self.X_train)
        self.y_pred_train = pd.DataFrame(self.y_pred_train, index=self.y_train.index)

    def _fit_gridsearch(self, param_grid: dict):
        """Internal method for fitting with GridSearchCV."""
        print("starting grid search...")
        tscv = TimeSeriesSplit(n_splits=3)

        grid_search = GridSearchCV(
            estimator=self.model_instance,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=tscv,
            verbose=0,
            n_jobs=-1,
        )
        grid_search.fit(self.X_train, self.y_train)
        self.trained_model = grid_search.best_estimator_
        self.y_pred_train = self.trained_model.predict(self.X_train)
        self.y_pred_train = pd.DataFrame(self.y_pred_train, index=self.y_train.index)
