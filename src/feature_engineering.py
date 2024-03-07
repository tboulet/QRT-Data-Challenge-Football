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
from src.data_loading import load_dataframe_labels
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
    metrics_names_to_fn_names = get_metrics_names_to_fn_names(df_features=df_features)

    # Drop features if their correlation with the target is too low
    thr = features_config["threshold_correlation_drop"]
    if verbose >= 1:
        print(f"\tDropping features with correlation with the target below {thr}...")
    n_dropped_features = 0
    features_blocked = set()
    if thr is not None and thr > 0:
        max_correlations_df = pd.read_csv(
            "data/max_correlations_global.csv", index_col=0
        )
        features_to_keep = df_features.columns.tolist().copy()
        for feature in df_features.columns:
            # Do not drop specific features
            if feature in SPECIFIC_PLAYERFEATURES + SPECIFIC_TEAMFEATURES:
                continue
            # Do not drop the _season_sum feature
            if feature.endswith("_season_sum"):
                continue
            max_corr_feature_row = max_correlations_df.loc[feature]
            # Do not drop if the max_correlated_feature is in the features_to_keep set
            if max_corr_feature_row["Max Correlated Feature"] in features_blocked:
                continue
            # Drop if the correlation is above the threshold
            if max_corr_feature_row["Correlation"] > thr:
                if verbose >= 2:
                    print(
                        f"\t\tDropping {feature} because its correlation with {max_corr_feature_row['Max Correlated Feature']} is above {thr}"
                    )
                # df_features = df_features.drop(columns=[feature])
                features_to_keep.remove(feature)
                features_blocked.add(max_corr_feature_row["Max Correlated Feature"])
                n_dropped_features += 1
        df_features = df_features[features_to_keep]

    if verbose >= 1:
        print(
            f"\tDropped {n_dropped_features} features with correlation with the target below {thr}"
        )

    # Drop specified metrics
    if verbose >= 1:
        print("\tDropping metrics...")
    n_dropped_features = 0
    names_feature_dropped = set()
    features_to_keep = df_features.columns.tolist().copy()
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
                    features_to_keep.remove(name_feature)
                n_dropped_features += 1
    df_features = df_features[features_to_keep]
    if verbose >= 1:
        print(f"\tDropped {n_dropped_features} features from metrics")

    # Drop specified features
    if verbose >= 1:
        print("\tDropping features...")
    n_dropped_features = 0
    features_to_keep = df_features.columns.tolist().copy()
    for name_feature in features_config["features_to_drop"]:
        if (
            name_feature not in features_to_keep
            and name_feature not in names_feature_dropped
        ):
            print(
                f"\tWARNING: tried to drop {name_feature}, but this feature doesn't appear in the dataframe"
            )
        else:
            if verbose >= 2:
                print(f"\t\tDropping {name_feature}")
            features_to_keep.remove(name_feature)
            n_dropped_features += 1
    df_features = df_features[features_to_keep]
    if verbose >= 1:
        print(f"\tDropped {n_dropped_features} features")

    # Drop specified aggregate features
    if verbose >= 1:
        print("\tDropping aggregate features...")
    n_dropped_features = 0
    features_to_keep = df_features.columns.tolist().copy()
    for fn_agg in features_config["fn_agg_to_drop"]:
        if verbose >= 2:
            print(f"\t\tDropping aggregate features with {fn_agg}:")
        for metric in metrics_names_to_fn_names.keys():
            feature = f"{metric}_{fn_agg}"
            if feature in features_to_keep:
                if verbose >= 2:
                    print(f"\t\t\tDropping aggregate feature {feature}")
                features_to_keep.remove(f"{metric}_{fn_agg}")
                n_dropped_features += 1
    df_features = df_features[features_to_keep]
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

    if verbose >= 1:
        print("\tAdding <metric or feature>_is_not_null features")
    n_added_feature_is_not_null = 0
    for metric, fn_names in get_metrics_names_to_fn_names(df_features).items():
        if metric not in SPECIFIC_PLAYERFEATURES + SPECIFIC_TEAMFEATURES:
            if verbose >= 2:
                print(f"\t\tAdding features is_not_null for metric {metric}")
            # Adding the metric is not null feature if features_config["add_non_null_indicator_metric"]. Don't add if features_config["add_non_null_indicator_feature"] and there is only one aggregate function for the metric
            if features_config["add_non_null_indicator_metric"] and (
                len(fn_names) > 1
                or not features_config["add_non_null_indicator_feature"]
            ):
                df_features[metric + "_is_not_null"] = (
                    df_features[[f"{metric}_{fn_name}" for fn_name in fn_names]]
                    .notnull()
                    .any(axis=1)
                    .astype(int)
                )
                n_added_feature_is_not_null += 1
            # Adding the feature is not null for each aggregate function if features_config["add_non_null_indicator_feature"] and there is more than one aggregate function for the metric (or if features_config["add_non_null_indicator_metric"] is False)
            if features_config["add_non_null_indicator_feature"]:
                for fn_name in fn_names:
                    feature_name = f"{metric}_{fn_name}" if fn_name != "" else metric
                    df_features[f"{feature_name}_is_not_null"] = (
                        df_features[f"{feature_name}"].notnull().astype(int)
                    )
                    n_added_feature_is_not_null += 1
    if verbose >= 1:
        print(f"\tAdded {n_added_feature_is_not_null} features")
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
            elif imputation_method in [None, False]:
                pass
            else:
                raise ValueError(f"Unknown imputation method {imputation_method}")
    return df_features


def add_team_couple_info(
    df_features: pd.DataFrame,
    features_config: dict,
    data_path: str,
) -> pd.DataFrame:
    """Try to add features to the dataframe to identify the home and away team names and the prior winrate of the home team.
    It requires to have previously run "python compute_team_name_to_id_mapping" to generate the team_mapping.csv file, which is used to map the team names to their identifiers.
    For test data, because it does not contains the team names, it will requires to use a predictor to predict the team names.

    Args:
        df_features (pd.DataFrame): the dataframe to add the features to
        features_config (dict): the config for the features
        data_path (str): the path to the data folder

    Returns:
        pd.DataFrame: the dataframe with the home and away team identifier features added
    """

    verbose = try_get("verbose", features_config, default=0)
    if verbose >= 1:
        print("\tAdding team couple info")

    # Load the team mapping CSV file
    try:
        team_mapping_df = pd.read_csv(f"data/team_mapping.csv")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The file team_mapping.csv is missing. Please run 'python compute_team_name_to_id_mapping.py' to generate it."
        )
    try:
        win_rates_df = pd.read_csv(f"data/win_rates.csv")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The file data/win_rates.csv is missing. Please run 'python compute_win_rates.py' to generate it."
        )

    if features_config["add_team_name_identifier"]:

        # Merge the team mapping with df_features based on home team name
        df_features = pd.merge(
            df_features,
            team_mapping_df,
            left_on="HOME_TEAM_NAME",
            right_on="Team_Name",
            how="left",
        )
        df_features.rename(columns={"Identifier": "HOME_TEAM_ID"}, inplace=True)
        df_features.drop(columns=["Team_Name"], inplace=True)

        # Merge the team mapping with df_features based on away team name
        df_features = pd.merge(
            df_features,
            team_mapping_df,
            left_on="AWAY_TEAM_NAME",
            right_on="Team_Name",
            how="left",
        )
        df_features.rename(columns={"Identifier": "AWAY_TEAM_ID"}, inplace=True)
        df_features.drop(columns=["Team_Name"], inplace=True)

    if features_config["add_team_name_indicator"]:
        # Create indicator features for each team name
        team_names = team_mapping_df["Team_Name"].tolist()
        for team_name in team_names:
            # Create indicator feature for home team
            df_features[f"{team_name}_HOME"] = (
                df_features["HOME_TEAM_NAME"] == team_name
            ).astype(int)
            # Create indicator feature for away team
            df_features[f"{team_name}_AWAY"] = (
                df_features["AWAY_TEAM_NAME"] == team_name
            ).astype(int)

    if features_config["add_team_winrate"]:

        # Add team identifier from the team_mapping.csv file (name to identifier)
        df_features = df_features.merge(
            team_mapping_df,
            how="left",
            left_on="HOME_TEAM_NAME",
            right_on="Team_Name",
        ).rename(columns={"Identifier": "identifier_HOME_ID"})
        df_features = df_features.merge(
            team_mapping_df,
            how="left",
            left_on="AWAY_TEAM_NAME",
            right_on="Team_Name",
        ).rename(columns={"Identifier": "identifier_AWAY_ID"})

        # Merge with win rates dataframe to add win rates for home teams HOME_WINS_RATE
        df_features = df_features.merge(
            win_rates_df[
                [
                    "HOME_TEAM_ID",
                    "AWAY_TEAM_ID",
                    "HOME_WINS",
                    "DRAW",
                    "AWAY_WINS",
                    "TOTAL_MATCHES",
                ]
            ],
            how="left",
            left_on=["identifier_HOME_ID", "identifier_AWAY_ID"],
            right_on=["HOME_TEAM_ID", "AWAY_TEAM_ID"],
        )
        # Drop the columns from the merge
        df_features.drop(
            columns=[
                "identifier_HOME_ID",
                "identifier_AWAY_ID",
                "Team_Name_x",
                "Team_Name_y",
                "HOME_TEAM_ID",
                "AWAY_TEAM_ID",
            ],
            inplace=True,
        )

        # Unbiased version for train data :
        if data_path == "data_train" and features_config["balance_train_stats"]:
            # Adapt the winrates for unbiasedness using the labels
            df_labels = load_dataframe_labels(global_data_path="data_train")
            df_labels.columns = [
                "ID",
                "HOME_HAS_WINS",
                "DRAW_HAS_OCCURED",
                "AWAY_HAS_WINS",
            ]
            df_features = pd.merge(df_features, df_labels, on="ID", how="left")
            df_features["HOME_WINS"] -= df_features["HOME_HAS_WINS"]
            df_features["DRAW"] -= df_features["DRAW_HAS_OCCURED"]
            df_features["AWAY_WINS"] -= df_features["AWAY_HAS_WINS"]
            df_features["TOTAL_MATCHES"] -= 1
            df_features.drop(
                columns=["HOME_HAS_WINS", "AWAY_HAS_WINS", "DRAW_HAS_OCCURED"],
                inplace=True,
            )

        df_features["HOME_WINS_RATE"] = (
            df_features["HOME_WINS"] / df_features["TOTAL_MATCHES"]
        )
        df_features["DRAW_RATE"] = df_features["DRAW"] / df_features["TOTAL_MATCHES"]
        df_features["AWAY_WINS_RATE"] = (
            df_features["AWAY_WINS"] / df_features["TOTAL_MATCHES"]
        )

        # Eventually drop the columns that are not used
        df_features.drop(columns=["HOME_WINS", "DRAW", "AWAY_WINS"], inplace=True)

        # Print values that are inf or NaN
        print(
            f"\t\tNumber of NaN values for win rates: {df_features['HOME_WINS_RATE'].isna().sum()} out of {len(df_features)}"
        )
        print(
            f"\t\tNumber of inf values for win rates: {np.isinf(df_features['HOME_WINS_RATE']).sum()} out of {len(df_features)}"
        )
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
