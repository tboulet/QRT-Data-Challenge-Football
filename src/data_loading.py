# ML libraries
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Utils
import os
import sys
from collections import defaultdict
from tqdm import tqdm
import warnings
from typing import List, Dict, Any, Tuple, Union, Optional, Callable, TypeVar
from src.data_management import add_prefix_to_columns, merge_dfs
from src.utils import try_get


def load_teamfeatures(
    teamfeatures_config: dict,
    data_path: str,
) -> pd.DataFrame:
    """Load the team features from the dataset, as a dataframe.
    This method load the home dataframe and the away dataframe, add "HOME_" or "AWAY_" prefix to the columns, and concatenate them.

    Args:
        teamfeatures_config (dict): the config for the teamfeatures
        data_path (str): the path where all data is.

    Returns:
        pd.DataFrame: the team features, as a dataframe
    """
    n_data_max = try_get("n_data_max", teamfeatures_config, default=sys.maxsize)
    verbose = try_get("verbose", teamfeatures_config, default=0)
    if verbose >= 1:
        print("\tLoading team features...")

    # Load the team features
    df_teamstatistics_home = pd.read_csv(
        data_path + f"/home_team_statistics_df.csv",
        nrows=n_data_max,
    )
    df_teamstatistics_away = pd.read_csv(
        data_path + f"/away_team_statistics_df.csv",
        nrows=n_data_max,
    )

    # Concatenate the home and away team features
    if verbose >= 1:
        print("\tConcatenating home and away team features")
    # Get the team features values
    home_teamfeatures = df_teamstatistics_home.iloc[:]
    away_teamfeatures = df_teamstatistics_away.iloc[:]
    # Add the prefix to the columns
    add_prefix_to_columns(home_teamfeatures, "HOME_")
    add_prefix_to_columns(away_teamfeatures, "AWAY_")
    # Concatenate the home and away team features
    dataframe_teamfeatures = merge_dfs([home_teamfeatures, away_teamfeatures], on = "ID")
    
    if verbose >= 1:
        print("\t[Shapes] Loaded teamfeatures shape: ", dataframe_teamfeatures.shape)
    return dataframe_teamfeatures


def load_playerfeatures(
    playerfeatures_config: dict,
    data_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the player features from the dataset, as a dataframe.
    This method load the home dataframe and the away dataframe, add "HOME_" or "AWAY_" prefix to the columns, and concatenate them.

    Args:
        playerfeatures_config (dict): the config for the playerfeatures
        data_path (str): the path where all data is.

    Returns:
        pd.DataFrame: the home player features, as a dataframe
        pd.DataFrame: the away player features, as a dataframe
    """
    n_data_max = try_get("n_data_max", playerfeatures_config, default=sys.maxsize)
    verbose = try_get("verbose", playerfeatures_config, default=0)
    if verbose >= 1:
        print("\tLoading player features...")

    # Load the player features
    df_playerfeatures_home = pd.read_csv(
        data_path + f"/home_player_statistics_df.csv",
        nrows=n_data_max,
    )
    df_playerfeatures_away = pd.read_csv(
        data_path + f"/away_player_statistics_df.csv",
        nrows=n_data_max,
    )

    # Replace ID_team with ID (we do that because one of the 4 player dataset has ID_team instead of ID)
    if "ID_team" in df_playerfeatures_home.columns:
        df_playerfeatures_home.rename(columns={"ID_team": "ID"}, inplace=True)
    if "ID_team" in df_playerfeatures_away.columns:
        df_playerfeatures_away.rename(columns={"ID_team": "ID"}, inplace=True)

    if verbose >= 1:
        print(
            "\t[Shapes] Loaded playerfeatures shape (home): ",
            df_playerfeatures_home.shape,
        )
        print(
            "\t[Shapes] Loaded playerfeatures shape (away): ",
            df_playerfeatures_away.shape,
        )

    return df_playerfeatures_home, df_playerfeatures_away


def load_dataframe_teamfeatures(
    dataset_prefix: str,
    global_data_path="datas_final/",
) -> pd.DataFrame:
    """Load team features from the dataset, as a dataframe.

    Args:
        dataset_prefix (str): either 'train' or 'test'.
        global_data_path (str, optional): the path where all CSVs are. Defaults to 'datas_final/'.
    """
    assert dataset_prefix in [
        "train",
        "test",
    ], 'dataset_name should be either "train" or "test"'

    home_team_statistics_df = pd.read_csv(
        global_data_path + f"/{dataset_prefix}_home_team_statistics_df.csv"
    )
    away_team_statistics_df = pd.read_csv(
        global_data_path + f"/{dataset_prefix}_away_team_statistics_df.csv"
    )

    home_teamfeatures = home_team_statistics_df.iloc[:]
    away_teamfeatures = away_team_statistics_df.iloc[:]

    home_teamfeatures.columns = "HOME_" + home_teamfeatures.columns
    away_teamfeatures.columns = "AWAY_" + away_teamfeatures.columns

    dataframe_teamfeatures = pd.concat(
        [home_teamfeatures, away_teamfeatures], join="inner", axis=1
    )
    dataframe_teamfeatures = dataframe_teamfeatures.replace(
        {np.inf: np.nan, -np.inf: np.nan}
    )
    return dataframe_teamfeatures


def load_dataframe_playersfeatures(
    dataset_prefix: str,
    global_data_path="datas_final/",
    n_rows_to_load=sys.maxsize,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load players features from the dataset, as a dataframe.

    Args:
        dataset_prefix (str): either 'train' or 'test'.
        global_data_path (str, optional): the path where all CSVs are. Defaults to 'datas_final/'.

    Returns:
        pd.DataFrame : the home players features
        pd.DataFrame : the away players features
    """
    assert dataset_prefix in [
        "train",
        "test",
    ], 'dataset_name should be either "train" or "test"'
    # Load max 1000 data points
    home_players_statistics_df = pd.read_csv(
        global_data_path + f"/{dataset_prefix}_home_player_statistics_df.csv",
        nrows=n_rows_to_load,
    )
    away_players_statistics_df = pd.read_csv(
        global_data_path + f"/{dataset_prefix}_away_player_statistics_df.csv",
        nrows=n_rows_to_load,
    )

    home_playersfeatures = home_players_statistics_df.iloc[:n_rows_to_load, :].replace(
        {np.inf: np.nan, -np.inf: np.nan}
    )
    away_playersfeatures = away_players_statistics_df.iloc[:n_rows_to_load, :].replace(
        {np.inf: np.nan, -np.inf: np.nan}
    )
    return home_playersfeatures, away_playersfeatures


def load_dataframe_labels(
    global_data_path: str,
) -> pd.DataFrame:
    """Load labels from the dataset

    Args:
        global_data_path (str): the path where all CSVs are.
    """
    labels = pd.read_csv(global_data_path + "/labels.csv")
    return labels


def load_index_numpy_labels(
    global_data_path: str,
) -> np.ndarray:
    """Load the labels as an index numpy array.

    Args:
        global_data_path (str): the path where all CSVs are. Defaults to 'datas_final/'.

    Returns:
        np.ndarray: the labels, as an index numpy array of shape (n_data,).
    """
    print("\tLoading labels...")
    labels = load_dataframe_labels(global_data_path)
    labels = labels.drop(columns=["ID"])
    labels = labels.to_numpy()
    labels = np.argmax(labels, axis=1)
    print("\t[Shapes] Loaded labels shape: ", labels.shape)
    return labels
