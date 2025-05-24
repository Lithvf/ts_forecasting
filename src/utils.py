"""Script that contains classes to read and prepare data."""

import pandas as pd
import matplotlib.pyplot as plt


class EnergyUsageData:
    """
    A concise class to load and optionally visualize energy usage data.
    """

    def __init__(self, path: str, plot: bool = False):
        """
        Initializes the EnergyUsageData instance, loads data, and optionally plots it.

        Args:
            path (str): The file path to the CSV data.
            plot (bool, optional): If True, a plot of the raw data will be generated. Defaults to False.
        """
        self.path = path
        self.plot = plot
        self.raw_data = self._load_data()
        if self.plot:
            self._plot_data()

    def _load_data(self) -> pd.DataFrame:
        """
        Loads the CSV data, processes the datetime column, and sorts.
        """
        try:
            df = pd.read_csv(self.path)
            df["datetime"] = pd.to_datetime(df["Datetime"])
            df.drop("Datetime", axis=1, inplace=True)
            return df.sort_values(by="datetime").reset_index(drop=True)
        except Exception as e:
            print(f"Error loading or processing data: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def _plot_data(self):
        """
        Internal method to plot the energy usage data.
        """
        if not self.raw_data.empty:
            self.raw_data.set_index("datetime").plot(
                ylabel="MW",
                title="PJME hourly energy usage",
                figsize=(12, 6),
                grid=True,
            )
            plt.tight_layout()
            plt.show()
        else:
            print("No data to plot.")

    def get_data(self) -> pd.DataFrame:
        """
        Returns the loaded and processed energy usage data.
        """
        return self.raw_data


class SplitData:
    """
    A class to split time-series data into training and testing sets,
    and optionally visualize the split.
    """

    def __init__(
        self, raw_data: pd.DataFrame, split_timestamp: str, plot: bool = False
    ):
        """
        Initializes the SplitData instance, performs the train-test split,
        and optionally plots the result.

        Args:
            raw_data: The input DataFrame containing time-series data.
            split_timestamp (str): The timestamp string (e.g., 'YYYY-MM-DD HH:MM:SS')
            plot (bool, optional): If True, a plot of the train-test split will be generated.
        """

        self.raw_data = raw_data.copy()
        self.plot = plot
        self._split_point = pd.to_datetime(split_timestamp)
        self.train, self.test = self.split_data_set()
        if self.plot:
            self._plot_train_test()

    def split_data_set(self):
        """
        Splits the raw data into training and testing sets based on the split_timestamp.
        Data up to and including `split_timestamp` goes into the training set.
        Data after `split_timestamp` goes into the testing set.

        Returns:
            train_df, test_df
        """
        df_indexed = self.raw_data.set_index("datetime")

        train = df_indexed.loc[df_indexed.index <= self._split_point].copy()
        test = df_indexed.loc[df_indexed.index > self._split_point].copy()

        return train.reset_index(), test.reset_index()

    def split_data_cv(self):
        """
        Placeholder for cross-validation timeseries splitting logic.
        """
        print("Note: Cross-validation splitting not implemented yet.")
        pass

    def _plot_train_test(self, figsize=(15, 7)):
        """
        Internal method to plot the train and test sets.
        """
        train_plot = self.train.set_index("datetime")
        test_plot = self.test.set_index("datetime")

        plot_df = train_plot.rename(columns={"PJME_MW": "train_set"}).join(
            test_plot.rename(columns={"PJME_MW": "test_set"}), how="outer"
        )

        fig, ax = plt.subplots(figsize=figsize)
        plot_df.plot(
            ax=ax,
            style=".",
            ylabel="MW",
            title="Train-Test Split of PJME Energy Usage",
        )
        ax.axvline(self._split_point, color="red", linestyle="--", label="Split Point")
        ax.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
