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
from sklearn.utils import shuffle


def cut_data_to_n_data_max(dataframe: pd.DataFrame, n_data_max: int) -> None:
    """Cut the data to a maximum number of samples."""
    if n_data_max is not None:
        dataframe = dataframe[:n_data_max]
    return dataframe


def shuffle_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Shuffle the data.

    Args:
        dataframe (pd.DataFrame): the final features

    Returns:
        pd.DataFrame: the shuffled features.
    """
    return shuffle(dataframe)


def save_predictions(
    list_label_preds_test: List[np.ndarray],
    path: str = "predictions.csv",
) -> None:
    """Save the predictions to a CSV file.

    Args:
        list_label_preds_test (List[np.ndarray]): a list of predictions from the different K folds.
        path (str): the path where to save the predictions. Defaults to "predictions.csv".
    """
    labels_preds_test = np.stack(list_label_preds_test, axis=1)
    n_data_test = labels_preds_test.shape[0]
    majority_elements = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), axis=1, arr=labels_preds_test
    )
    majority_elements_one_hot = np.eye(3, dtype=int)[majority_elements]
    df = pd.DataFrame(
        majority_elements_one_hot,
        columns=["HOME_WINS", "DRAW", "AWAY_WINS"],
        index=range(12303, 12303 + n_data_test, 1),
    )
    df.index.name = "ID"
    df.to_csv(path)
    print(f"Predictions saved to {path}")


def add_prefix_to_columns(
    dataframe: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """Change in place the columns of the dataframe by adding a prefix to them.

    Args:
        dataframe (pd.DataFrame): the dataframe to change the columns of
        prefix (str): the prefix to add to the columns, in the form "prefix_"

    Returns:
        pd.DataFrame: the dataframe with the columns changed
    """
    dataframe.columns = [
        prefix + col if col != "ID" else col for col in dataframe.columns
    ]
    
    
def merge_dfs(list_dataframes : List[pd.DataFrame], on : str) -> pd.DataFrame:
    """Merge a list of dataframes on a given column.

    Args:
        list_dataframes (List[pd.DataFrame]): the list of dataframes to merge
        on (str): the column to merge on, usually "ID"

    Returns:
        pd.DataFrame: the merged dataframe
    """
    df = list_dataframes[0]
    for i in range(1, len(list_dataframes)):
        df = df.merge(list_dataframes[i], on=on, how="outer")
    return df