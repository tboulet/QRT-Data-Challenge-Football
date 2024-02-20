from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Type, Callable

import numpy as np
import pandas as pd
import xgboost as xgb

from features_loaders.base_feature_loader import BaseLoader
from src.constants import SPECIFIC_TEAMFEATURES
from src.data_analysis import get_metrics_names_to_fn_names


class LoaderTeamfeatures(BaseLoader):
    """The class for loading the team features."""

    def load_features(self, data_path: str) -> Dict[str, Any]:
        """Load the features in memory. They can then be used by any feature creator for creating additional features.

        Args:
            data_path (str): the path where all data is.

        Returns:
            Tuple[Dict[str, Any]]: the features, as a one-element dictionnary of dataframes.
        """
        verbose: int = self.config["verbose"]
        if verbose >= 1:
            print("Loading team features...")

        # Load the team features
        home_team_statistics_df = pd.read_csv(
            data_path + f"/home_team_statistics_df.csv"
        )
        away_team_statistics_df = pd.read_csv(
            data_path + f"/away_team_statistics_df.csv"
        )

        # Concatenate the home and away team features
        if verbose >= 1:
            print("Concatenating home and away team features")
        home_teamfeatures = home_team_statistics_df.iloc[:]
        away_teamfeatures = away_team_statistics_df.iloc[:]
        home_teamfeatures.columns = "HOME_" + home_teamfeatures.columns
        away_teamfeatures.columns = "AWAY_" + away_teamfeatures.columns
        dataframe_teamfeatures = pd.concat(
            [home_teamfeatures, away_teamfeatures], join="inner", axis=1
        )
        if verbose >= 1:
            print("Teamfeatures shape: ", dataframe_teamfeatures.shape)

        # Replace infinities with NaN
        if verbose >= 1:
            print("Replacing infinities with NaN")
        dataframe_teamfeatures = dataframe_teamfeatures.replace(
            {np.inf: np.nan, -np.inf: np.nan}
        )

        # Drop teammetrics
        if verbose >= 1:
            print("Dropping teamfeatures from teammetrics")
        n_dropped_teamfeatures_from_teammetrics = 0
        teammetrics_names_to_fn_names = get_metrics_names_to_fn_names(
            dataframe_teamfeatures
        )
        for teammetric in self.config["metrics_to_drop"]:
            if teammetric not in teammetrics_names_to_fn_names:
                print(
                    f"WARNING: tried to drop {teammetric}, but it is not in teammetrics_names_to_fn_names"
                )
            else:
                if verbose >= 2: 
                    print(f"Dropping features corresponding to {teammetric}")
                for aggregate_function_names in teammetrics_names_to_fn_names[
                    teammetric
                ]:
                    teamfeature = f"{teammetric}_{aggregate_function_names}"
                    if teamfeature in dataframe_teamfeatures.columns:
                        dataframe_teamfeatures = dataframe_teamfeatures.drop(
                            columns=[teamfeature]
                        )
                    n_dropped_teamfeatures_from_teammetrics += 1
        if verbose >= 1:
            print(f"\tDropped {n_dropped_teamfeatures_from_teammetrics} teammetrics")

        # Drop teamfeatures
        if verbose >= 1:
            print("Dropping teamfeatures")
        n_dropped_teamfeatures = 0
        for teamfeature in self.config["features_to_drop"]:
            if teamfeature in dataframe_teamfeatures.columns:
                if verbose >= 2:
                    print(f"Dropping {teamfeature}")
                dataframe_teamfeatures = dataframe_teamfeatures.drop(
                    columns=[teamfeature]
                )
                n_dropped_teamfeatures += 1
        if verbose >= 1:
            print(f"\tDropped {n_dropped_teamfeatures} teamfeatures")

        # Add feature_is_not_null features (indicator features for missing values)
        if self.config["add_non_null_indicator_feature"]:
            if verbose >= 1:
                print(f"Adding {len(dataframe_teamfeatures.columns)} <feature>_is_not_null features")
            for teamfeature in dataframe_teamfeatures.columns:
                if verbose >= 2:
                    print(f"Adding feature is_not_null for {teamfeature}")
                dataframe_teamfeatures[teamfeature + "_is_not_null"] = (
                    dataframe_teamfeatures[teamfeature].notnull().astype(int)
                )

        # Add metric_is_not_null features (indicator feature for data missing for a given metric (for all aggregate functions))
        if self.config["add_non_null_indicator_metric"]:
            if verbose >= 1:
                print("Adding <metric>_is_not_null features")
            n_added_metric_is_not_null = 0
            for teammetric, fn_names in get_metrics_names_to_fn_names(dataframe_teamfeatures).items():
                if teammetric not in SPECIFIC_TEAMFEATURES:
                    if verbose >= 2:
                        print(f"Adding feature is_not_null for metric {teammetric}")
                    dataframe_teamfeatures[teammetric + "_is_not_null"] = dataframe_teamfeatures[[f"{teammetric}_{fn_name}" for fn_name in fn_names]].notnull().astype(int).sum(axis=1)
                    n_added_metric_is_not_null += 1
            if verbose >= 1:
                print(f"\tAdded {n_added_metric_is_not_null} metric_is_not_null features")
            
        # Impute missing values, but only where column name is not in SPECIFIC_TEAMFEATURES
        if verbose >= 1:
            print("Imputing missing values")
        imputation_method = self.config["imputation_method"]
        for teamfeature in dataframe_teamfeatures.columns:
            if dataframe_teamfeatures[teamfeature].isnull().sum() > 0 and teamfeature not in SPECIFIC_TEAMFEATURES:
                if verbose >= 2:
                    print(f"Imputing missing values for {teamfeature} with {imputation_method}")
                if imputation_method == "mean":
                    dataframe_teamfeatures[teamfeature] = dataframe_teamfeatures[teamfeature].fillna(dataframe_teamfeatures[teamfeature].mean())
                elif imputation_method == "median":
                    dataframe_teamfeatures[teamfeature] = dataframe_teamfeatures[teamfeature].fillna(dataframe_teamfeatures[teamfeature].median())
                elif imputation_method == "zero":
                    dataframe_teamfeatures[teamfeature] = dataframe_teamfeatures[teamfeature].fillna(0)
                else:
                    raise ValueError(f"Unknown imputation method {imputation_method}")

        # Save the features in this object for later use
        self.dataframe_teamfeatures = dataframe_teamfeatures
        # Return the features
        if verbose >= 1:
            print(f"Team features loaded, shape : {self.dataframe_teamfeatures.shape}")
        return {"teamfeatures": self.dataframe_teamfeatures}

    def get_usable_features(self) -> Dict[str, pd.DataFrame]:
        # Insert a column "ID" copy of the column "HOME_ID" at first position
        # self.dataframe_teamfeatures.insert(0, "ID", self.dataframe_teamfeatures["HOME_ID"])
        
        # Try dropping columns that are not for training
        for name_feature in SPECIFIC_TEAMFEATURES:
            if name_feature in self.dataframe_teamfeatures.columns:
                self.dataframe_teamfeatures = self.dataframe_teamfeatures.drop(
                    columns=[name_feature]
                )
        # Return the features
        return {"teamfeatures": self.dataframe_teamfeatures}
