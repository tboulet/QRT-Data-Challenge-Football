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



def load_dataframe_teamfeatures(
        dataset_prefix : str,
        global_data_path = 'datas_final/',
        ) -> pd.DataFrame:
    """Load team features from the dataset, as a dataframe.

    Args:
        dataset_prefix (str): either 'train' or 'test'.
        global_data_path (str, optional): the path where all CSVs are. Defaults to 'datas_final/'.
    """
    assert dataset_prefix in ['train', 'test'], 'dataset_name should be either "train" or "test"'
    
    home_team_statistics_df = pd.read_csv(global_data_path + f'/{dataset_prefix}_home_team_statistics_df.csv')
    away_team_statistics_df = pd.read_csv(global_data_path + f'/{dataset_prefix}_away_team_statistics_df.csv')

    home_teamfeatures = home_team_statistics_df.iloc[:]
    away_teamfeatures = away_team_statistics_df.iloc[:]

    home_teamfeatures.columns = 'HOME_' + home_teamfeatures.columns
    away_teamfeatures.columns = 'AWAY_' + away_teamfeatures.columns

    dataframe_teamfeatures =  pd.concat([home_teamfeatures,away_teamfeatures],join='inner',axis=1)
    dataframe_teamfeatures = dataframe_teamfeatures.replace({np.inf:np.nan,-np.inf:np.nan})
    return dataframe_teamfeatures


def load_dataframe_playersfeatures(
        dataset_prefix : str,
        global_data_path = 'datas_final/',
        n_rows_to_load = sys.maxsize,
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load players features from the dataset, as a dataframe.
    
    Args:
        dataset_prefix (str): either 'train' or 'test'.
        global_data_path (str, optional): the path where all CSVs are. Defaults to 'datas_final/'.
        
    Returns:
        pd.DataFrame : the home players features
        pd.DataFrame : the away players features
    """
    assert dataset_prefix in ['train', 'test'], 'dataset_name should be either "train" or "test"'
    # Load max 1000 data points
    home_players_statistics_df = pd.read_csv(global_data_path + f'/{dataset_prefix}_home_player_statistics_df.csv', nrows=n_rows_to_load)
    away_players_statistics_df = pd.read_csv(global_data_path + f'/{dataset_prefix}_away_player_statistics_df.csv', nrows=n_rows_to_load)
    
    home_playersfeatures = home_players_statistics_df.iloc[:1000,:].replace({np.inf:np.nan,-np.inf:np.nan})
    away_playersfeatures = away_players_statistics_df.iloc[:1000,:].replace({np.inf:np.nan,-np.inf:np.nan})
    return home_playersfeatures, away_playersfeatures

    
    
def load_dataframe_labels(
        global_data_path = 'datas_final/',
        ) -> pd.DataFrame:
    """Load labels from the dataset

    Args:
        global_data_path (str, optional): the path where all CSVs are. Defaults to 'datas_final/'.
    """
    labels = pd.read_csv(global_data_path + 'Y_train.csv')
    return labels


