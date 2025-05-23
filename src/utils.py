"""Script that contains classes to read and prepare data."""

import pandas as pd


class EnergyUsageData:
    def __init__(self, path, plot=False):
        self.path = path
        self.plot = plot
        self.raw_data = self.load_raw_data()
        self.plot_raw_data()

    def load_raw_data(self):
        raw_data = pd.read_csv(self.path)
        raw_data["datetime"] = pd.to_datetime(raw_data["Datetime"])
        raw_data.drop("Datetime", axis=1, inplace=True)
        return raw_data.sort_values(by="datetime")

    def plot_raw_data(self):
        if self.plot == True:
            self.raw_data.set_index("datetime").plot(
                ylabel="MW", title="PJME hourly energy usage"
            )


class SplitData:
    def __init__(self, raw_data, timestamp_start, timestamp_end, plot=False):
        self.raw_data = raw_data
        self.timestamp_start = timestamp_start
        self.timestamp_end = timestamp_end
        self.plot = plot
        self.train, self.test = self.split_data_set()
        self.plot_train_test()

    def split_data_set(self):
        self.raw_data.set_index("datetime", inplace=True)
        train = self.raw_data.loc[self.raw_data.index <= self.timestamp_start].copy()
        test = self.raw_data.loc[self.raw_data.index > self.timestamp_start].copy()
        return train, test

    def split_data_cv(self):
        pass

    def plot_train_test(self):
        if self.plot == True:
            self.train.rename(columns={"PJME_MW": "train_set"}).join(
                self.test.rename(columns={"PJME_MW": "test_set"}), how="outer"
            ).plot(style=".", xlabel="MW", title="train-test split PJME")
