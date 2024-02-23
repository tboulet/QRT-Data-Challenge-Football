# Logging
from collections import defaultdict
import random
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
import wandb
from tensorboardX import SummaryWriter

# Config system
import hydra
from omegaconf import OmegaConf, DictConfig

# Utils
from tqdm import tqdm
import datetime
from time import time
from typing import Any, Dict, List, Tuple, Type
import cProfile

# ML libraries
import numpy as np
from src.constants import SPECIFIC_PLAYERFEATURES, SPECIFIC_TEAMFEATURES
from src.data_analysis import get_metrics_names_to_fn_names
from src.time_measure import RuntimeMeter
from src.utils import get_name_trainer_and_features, try_get, try_get_seed
from sklearn.utils import shuffle


def drop_features(df_features: pd.DataFrame, features_config: dict) -> pd.DataFrame:
    """Drop features from the dataframe, according to the config.
    It can either drop column by column or by aggregated features (i.e. by metrics)

    Args:
        df_features (pd.DataFrame): the dataframe to drop features from
        features_config (dict): the config for the features

    Returns:
        pd.DataFrame: the dataframe with the features dropped
    """
    verbose = try_get("verbose", features_config, default=0)
    if verbose >= 1:
        print("\tDropping features")
    n_dropped_features = 0
    metrics_names_to_fn_names = get_metrics_names_to_fn_names(df_features=df_features)

    # Drop metrics
    names_feature_dropped = set()
    for metric in features_config["metrics_to_drop"]:
        if metric not in metrics_names_to_fn_names:
            print(
                f"\tWARNING: tried to drop metric {metric}, but this metric doesn't appear in the dataframe"
            )
        else:
            if verbose >= 2:
                print(f"\t\tDropping features corresponding to {metric}")
            for aggregate_function_names in metrics_names_to_fn_names[metric]:
                name_feature = f"{metric}_{aggregate_function_names}"
                names_feature_dropped.add(name_feature)
                if name_feature in df_features.columns:
                    df_features = df_features.drop(columns=[name_feature])
                n_dropped_features += 1
    if verbose >= 1:
        print(f"\tDropped {n_dropped_features} features from metrics")

    # Drop features
    if verbose >= 1:
        print("\tDropping features")
    n_dropped_features = 0
    for name_feature in features_config["features_to_drop"]:
        if (
            name_feature not in df_features.columns
            and name_feature not in names_feature_dropped
        ):
            print(
                f"\tWARNING: tried to drop {name_feature}, but this feature doesn't appear in the dataframe"
            )
        else:
            if verbose >= 2:
                print(f"\t\tDropping {name_feature}")
            df_features = df_features.drop(columns=[name_feature])
            n_dropped_features += 1
    if verbose >= 1:
        print(f"\tDropped {n_dropped_features} features")

    return df_features


def add_non_null_indicator_features(
    df_features: pd.DataFrame,
    features_config: dict,
) -> pd.DataFrame:
    """Add new features to the dataset, which corresponds to indicator features for missing values.

    Args:
        df_features (pd.DataFrame): the dataframe to add the features to
        features_config (dict): the config for the features

    Returns:
        pd.DataFrame: a new dataframe with the indicator features added
    """
    verbose = try_get("verbose", features_config, default=0)

    # Add feature_is_not_null features (indicator features for missing values)
    if features_config["add_non_null_indicator_feature"]:
        if verbose >= 1:
            print(f"\tAdding {len(df_features.columns)} <feature>_is_not_null features")
        for feature in df_features.columns:
            if feature not in SPECIFIC_PLAYERFEATURES + SPECIFIC_TEAMFEATURES:
                if verbose >= 2:
                    print(f"\t\tAdding feature is_not_null for {feature}")
                df_features[feature + "_is_not_null"] = (
                    df_features[feature].notnull().astype(int)
                )

    # Add metric_is_not_null features (indicator feature for data missing for a given metric (for all aggregate functions))
    if features_config["add_non_null_indicator_metric"]:
        if verbose >= 1:
            print("\tAdding <metric>_is_not_null features")
        n_added_metric_is_not_null = 0
        for metric, fn_names in get_metrics_names_to_fn_names(df_features).items():
            if metric not in SPECIFIC_PLAYERFEATURES + SPECIFIC_TEAMFEATURES:
                if verbose >= 2:
                    print(f"\t\tAdding feature is_not_null for metric {metric}")
                df_features[metric + "_is_not_null"] = (
                    df_features[[f"{metric}_{fn_name}" for fn_name in fn_names]]
                    .notnull()
                    .all(axis=1)
                    .astype(int)
                )
                n_added_metric_is_not_null += 1
        if verbose >= 1:
            print(f"\tAdded {n_added_metric_is_not_null} metric_is_not_null features")

    return df_features


def impute_missing_values(
    df_features: pd.DataFrame,
    features_config: dict,
) -> pd.DataFrame:
    """Impute missing values in the dataframe, according to the config.

    Args:
        df_features (pd.DataFrame): the dataframe to impute the missing values in
        features_config (dict): the config for the features

    Returns:
        pd.DataFrame: the dataframe with the missing values imputed
    """
    verbose = try_get("verbose", features_config, default=0)
    if verbose >= 1:
        print("\tImputing missing values")

    imputation_method = features_config["imputation_method"]
    for name_feature in df_features.columns:
        if name_feature not in SPECIFIC_TEAMFEATURES + SPECIFIC_PLAYERFEATURES:
            if verbose >= 2:
                print(
                    f"\t\tImputing missing values for {name_feature} with {imputation_method}"
                )
            if imputation_method == "mean":
                df_features[name_feature] = df_features[name_feature].fillna(
                    df_features[name_feature].mean()
                )
            elif imputation_method == "median":
                df_features[name_feature] = df_features[name_feature].fillna(
                    df_features[name_feature].median()
                )
            elif imputation_method == "zero":
                df_features[name_feature] = df_features[name_feature].fillna(0)
            else:
                raise ValueError(f"Unknown imputation method {imputation_method}")
    return df_features


def get_agg_playerfeatures_by_operation(
    df_playerfeatures: pd.DataFrame,
    aggregator_config: dict,
) -> pd.DataFrame:
    list_df_agg_playerfeatures: List[pd.DataFrame] = []
    # Try dropping the columns that are not used
    columns = [
        "LEAGUE",
        "TEAM_NAME",
        "POSITION",
        "PLAYER_NAME",
    ]
    df_playerfeatures_filtered = df_playerfeatures.drop(
        columns=columns, errors="ignore"
    )
    df_playerfeatures_grouped = df_playerfeatures_filtered.groupby("ID")
    for operation in aggregator_config["operations"]:
        if operation == "mean":
            list_df_agg_playerfeatures.append(df_playerfeatures_grouped.mean())
        elif operation == "median":
            list_df_agg_playerfeatures.append(df_playerfeatures_grouped.median())
        elif operation == "sum":
            list_df_agg_playerfeatures.append(df_playerfeatures_grouped.sum())
        elif operation == "max":
            list_df_agg_playerfeatures.append(df_playerfeatures_grouped.max())
        elif operation == "min":
            list_df_agg_playerfeatures.append(df_playerfeatures_grouped.min())
        elif operation == "std":
            list_df_agg_playerfeatures.append(df_playerfeatures_grouped.std())
        else:
            raise ValueError(f"Unknown operation {operation}")

    return pd.concat(list_df_agg_playerfeatures, axis=1)


def group_playerfeatures_by_match_and_by_team(
    df_playerfeatures_home: pd.DataFrame,
    df_playerfeatures_away: pd.DataFrame,
) -> Dict[int, Dict[str, pd.DataFrame]]:
    """Group in a dictionnary the player features by match and by team.
    This function takes a lot of time.
    You can access to the dataframe of player features for a given match and a given team with:
    df_playerfeatures_matchID_team = matchID_to_side_to_playerfeatures[matchID][team]

    Where matchID_to_side_to_playerfeatures is the output of this function.

    Args:
        df_playerfeatures_home (pd.DataFrame): the dataframe of player features for the home team
        df_playerfeatures_away (pd.DataFrame): the dataframe of player features for the away team

    Returns:
        Dict[int, Dict[str, pd.DataFrame]]: the mapping from matchID to team to player features
    """
    matchID_to_side_to_playerfeatures = {}
    for matchID in df_playerfeatures_home["ID"].unique():
        matchID_to_side_to_playerfeatures[matchID] = {
            "HOME": df_playerfeatures_home[df_playerfeatures_home["ID"] == matchID],
            "AWAY": df_playerfeatures_away[df_playerfeatures_away["ID"] == matchID],
        }
    return matchID_to_side_to_playerfeatures