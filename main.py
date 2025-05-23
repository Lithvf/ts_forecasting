from src.utils import EnergyUsageData, SplitData
from src.features import CreateFeatures
from src.models import FitModel, PredictModel
from src.forecast import RecursiveForecaster

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

if __name__ == "__main__":
    # param_grid = {
    #     "n_estimators": [100, 300],
    #     "max_depth": [3, 5],
    #     "learning_rate": [0.01, 0.1],
    # }
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [2, 3],
        "learning_rate": [0.01, 0.05],
        "subsample": [0.6, 0.8],
        "reg_lambda": [5, 10],
        "reg_alpha": [0.1, 1],
    }
    raw_data = EnergyUsageData(path="data/PJME_hourly.csv", plot=False)
    raw_data = raw_data.raw_data
    timestamp_start = "01-Oct-2017"
    timestamp_end = "01-Jan-2050"
    split_data = SplitData(raw_data, timestamp_start, timestamp_end, plot=False)

    train, test = split_data.train, split_data.test
    target = "PJME_MW"
    X_train, y_train = CreateFeatures(train, target).X, CreateFeatures(train, target).y
    X_test, y_test = CreateFeatures(test, target).X, CreateFeatures(test, target).y
    # fit_model = FitModel(
    #     XGBRegressor(
    #         base_score=0.5,
    #         booster="gbtree",
    #         n_estimators=1000,
    #         objective="reg:linear",
    #         max_depth=3,
    #         learning_rate=0.01,
    #     ),
    #     X_train,
    #     y_train,
    #     train,
    #     plot=False,
    # )

    fit_model = FitModel(
        XGBRegressor(),
        X_train,
        y_train,
        train,
        param_grid=param_grid,
        plot=False,
    )
    # fit_model = FitModel(RandomForestRegressor(), X_train, y_train, train, plot=True)
    predict = PredictModel(fit_model.trained_model, X_test, y_test, test, plot=True)
    steps = 48
    forecaster = RecursiveForecaster(
        fit_model.trained_model, train, y_test, test, steps=steps, plot=True
    )
