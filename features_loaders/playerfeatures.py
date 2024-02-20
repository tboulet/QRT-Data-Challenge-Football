from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Type, Callable

import numpy as np
import pandas as pd
import xgboost as xgb

from features_loaders.base_feature_loader import BaseLoader
from src.constants import SPECIFIC_PLAYERFEATURES
from src.data_analysis import get_metrics_names_to_fn_names


class LoaderPlayerfeatures(BaseLoader):
    """The class for loading the player features."""

    def load_features(self, data_path: str) -> Dict[str, Any]:
        """Load the player features in memory. This will construct two intermediary objects :
        - a dataset of player features, for potentially post unsupervised learning
        - player features grouped by match and by team, for post aggregation of player features.

        Args:
            data_path (str): the path where all data is.

        Returns:
            Tuple[Dict[str, Any]]: the features, as a one-element dictionnary of dataframes.
        """
        self.verbose: int = self.config["verbose"]
        self.n_data_max = 10

        if self.verbose >= 1:
            print("Loading player features...")

        # Load the player features
        df_playerfeatures_home = pd.read_csv(
            data_path + f"/home_player_statistics_df.csv",
            nrows=self.n_data_max,
        )
        df_playerfeatures_away = pd.read_csv(
            data_path + f"/away_player_statistics_df.csv",
            nrows=self.n_data_max,
        )

        # Preprocess the player features
        df_playerfeatures_home = self.preprocess_playerfeatures(df_playerfeatures_home)
        df_playerfeatures_away = self.preprocess_playerfeatures(df_playerfeatures_away)

        # Add the two dataframes
        df_playerfeatures = pd.concat(
            [df_playerfeatures_home, df_playerfeatures_away], ignore_index=True
        )
        
        # Group the player features by match and by side
        matchesIDs = df_playerfeatures["ID"].unique()
        matchID_to_side_to_playerfeatures : Dict[int, Dict[str, pd.DataFrame]] = {
            matchID: {
                "home": df_playerfeatures_home[df_playerfeatures_home["ID"] == matchID],
                "away": df_playerfeatures_away[df_playerfeatures_away["ID"] == matchID]
                }
            for matchID in matchesIDs
        }
        
        # Return the features
        return {"playerfeatures": df_playerfeatures, "matchID_to_side_to_playerfeatures": matchID_to_side_to_playerfeatures}

    def preprocess_playerfeatures(
        self, df_playerfeatures: pd.DataFrame
    ) -> pd.DataFrame:
        """Preprocess the player features. This will include the following steps :
        - Replace infinities with NaN
        - Drop playermetrics
        - Drop playerfeatures
        - Add feature_is_not_null features (indicator features for missing values)
        - Add metric_is_not_null features (indicator feature for data missing for a given metric (for all aggregate functions))
        """
        # Replace ID_team with ID (we do that because one of the 4 player dataset has ID_team instead of ID)
        if "ID_team" in df_playerfeatures.columns:
            df_playerfeatures.rename(columns={"ID_team": "ID"}, inplace=True)        

        # Replace infinities with NaN
        df_playerfeatures = df_playerfeatures.replace(
            {np.inf: np.nan, -np.inf: np.nan}
        )

        # Drop playermetrics
        if self.verbose >= 1:
            print("Dropping playerfeatures from playermetrics")
        n_dropped_playerfeatures_from_playermetrics = 0
        playermetrics_names_to_fn_names = get_metrics_names_to_fn_names(
            df_playerfeatures
        )
        for playermetric in self.config["metrics_to_drop"]:
            if playermetric not in playermetrics_names_to_fn_names:
                print(
                    f"WARNING: tried to drop {playermetric}, but it is not in playermetrics_names_to_fn_names"
                )
            else:
                if self.verbose >= 2:
                    print(f"Dropping features corresponding to {playermetric}")
                for aggregate_function_names in playermetrics_names_to_fn_names[
                    playermetric
                ]:
                    playerfeatures = f"{playermetric}_{aggregate_function_names}"
                    if playerfeatures in df_playerfeatures.columns:
                        df_playerfeatures = df_playerfeatures.drop(
                            columns=[playerfeatures]
                        )
                    n_dropped_playerfeatures_from_playermetrics += 1
        if self.verbose >= 1:
            print(f"\tDropped {n_dropped_playerfeatures_from_playermetrics} playermetrics")

        # Drop playerfeatures
        if self.verbose >= 1:
            print("Dropping playerfeatures")
        n_dropped_playerfeatures = 0
        for playerfeatures in self.config["features_to_drop"]:
            if playerfeatures in df_playerfeatures.columns:
                if self.verbose >= 2:
                    print(f"Dropping {playerfeatures}")
                df_playerfeatures = df_playerfeatures.drop(
                    columns=[playerfeatures]
                )
                n_dropped_playerfeatures += 1
        if self.verbose >= 1:
            print(f"\tDropped {n_dropped_playerfeatures} playerfeatures")

        # Add feature_is_not_null features (indicator features for missing values)
        if self.config["add_non_null_indicator_feature"]:
            if self.verbose >= 1:
                print(
                    f"Adding {len(df_playerfeatures.columns)} <feature>_is_not_null features"
                )
            for playerfeatures in df_playerfeatures.columns:
                if self.verbose >= 2:
                    print(f"Adding feature is_not_null for {playerfeatures}")
                df_playerfeatures[playerfeatures + "_is_not_null"] = (
                    df_playerfeatures[playerfeatures].notnull().astype(int)
                )

        # Add metric_is_not_null features (indicator feature for data missing for a given metric (for all aggregate functions))
        if self.config["add_non_null_indicator_metric"]:
            if self.verbose >= 1:
                print("Adding <metric>_is_not_null features")
            n_added_metric_is_not_null = 0
            for playermetric, fn_names in get_metrics_names_to_fn_names(
                df_playerfeatures
            ).items():
                if playermetric not in SPECIFIC_PLAYERFEATURES:
                    if self.verbose >= 2:
                        print(f"Adding feature is_not_null for metric {playermetric}")
                    df_playerfeatures[playermetric + "_is_not_null"] = (
                        df_playerfeatures[
                            [f"{playermetric}_{fn_name}" for fn_name in fn_names]
                        ]
                        .notnull()
                        .astype(int)
                        .sum(axis=1)
                    )
                    n_added_metric_is_not_null += 1
            if self.verbose >= 1:
                print(
                    f"\tAdded {n_added_metric_is_not_null} metric_is_not_null features"
                )

        # Impute missing values, but only where column name is not in SPECIFIC_PLAYERFEATURES
        if self.verbose >= 1:
            print("Imputing missing values")
        imputation_method = self.config["imputation_method"]
        for playerfeatures in df_playerfeatures.columns:
            if (
                df_playerfeatures[playerfeatures].isnull().sum() > 0
                and playerfeatures not in SPECIFIC_PLAYERFEATURES
            ):
                if self.verbose >= 2:
                    print(
                        f"Imputing missing values for {playerfeatures} with {imputation_method}"
                    )
                if imputation_method == "mean":
                    df_playerfeatures[playerfeatures] = df_playerfeatures[
                        playerfeatures
                    ].fillna(df_playerfeatures[playerfeatures].mean())
                elif imputation_method == "median":
                    df_playerfeatures[playerfeatures] = df_playerfeatures[
                        playerfeatures
                    ].fillna(df_playerfeatures[playerfeatures].median())
                elif imputation_method == "zero":
                    df_playerfeatures[playerfeatures] = df_playerfeatures[
                        playerfeatures
                    ].fillna(0)
                else:
                    raise ValueError(f"Unknown imputation method {imputation_method}")

        # Save the features in this object for later use
        self.dataframe_playerfeatures = df_playerfeatures
        # Return the features
        if self.verbose >= 1:
            print(f"Player features loaded, shape : {self.dataframe_playerfeatures.shape}")
        return df_playerfeatures
    
    def get_usable_features(self) -> Dict[str, np.ndarray]:
        return {}