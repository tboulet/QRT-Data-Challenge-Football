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
from features_loaders.base_feature_loader import BaseLoader
from src.time_measure import RuntimeMeter
from src.utils import get_name_trainer_and_features, try_get_seed


def get_train_val_split(
    features_dict_final_arrays: Dict[str, np.ndarray],
    labels: np.ndarray,
    k: int,
    K: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Generate the k-th train and validation splits from the features and labels.

    Args:
        features_dict_final_arrays (Dict[str, np.ndarray]): the final features, as a dictionnary of numpy arrays of shape (n_data, *_).
        labels (np.ndarray): the labels, as a numpy array of shape (n_data,).
        k (int): the index of the fold.
        K (int): the number of folds.

    Returns:
        features_dict_final_arrays_train (Dict[str, np.ndarray]): the final features for the training set.
        features_dict_final_arrays_val (Dict[str, np.ndarray]): the final features for the validation set, or None if K == 1.
        labels_train (np.ndarray): the labels for the training set.
        labels_val (np.ndarray): the labels for the validation set, or None if K == 1.
    """
    if K == 1:
        assert k == 0
        return features_dict_final_arrays, None, labels, None

    n_data = len(next(iter(features_dict_final_arrays.values())))
    indices_val = np.arange(n_data)[k * n_data // K : (k + 1) * n_data // K]
    indices_train = np.concatenate(
        [
            np.arange(n_data)[: k * n_data // K],
            np.arange(n_data)[(k + 1) * n_data // K :],
        ]
    )
    features_dict_final_arrays_train = {
        name_feature: feature_final_array[indices_train]
        for name_feature, feature_final_array in features_dict_final_arrays.items()
    }
    features_dict_final_arrays_val = {
        name_feature: feature_final_array[indices_val]
        for name_feature, feature_final_array in features_dict_final_arrays.items()
    }
    labels_train = labels[indices_train]
    labels_val = labels[indices_val]

    return (
        features_dict_final_arrays_train,
        features_dict_final_arrays_val,
        labels_train,
        labels_val,
    )


def cut_data_to_n_data_max(
    features_dict_final_arrays: Dict[str, np.ndarray], n_data_max: int
) -> None:
    """Cut the data to a maximum number of samples.

    Args:
        features_dict_final_arrays (Dict[str, np.ndarray]): the final features, as a dictionnary of numpy arrays of shape (n_data, *_).
        n_data_max (int): the maximum number of samples.
    """
    n_data = len(next(iter(features_dict_final_arrays.values())))
    if isinstance(n_data_max, int) and n_data_max < n_data:
        for name_feature, feature_final_array in features_dict_final_arrays.items():
            features_dict_final_arrays[name_feature] = feature_final_array[:n_data_max]
        n_data = min(n_data, n_data_max)
        print(f"Limiting the data to {n_data} samples.")


def shuffle_data(features_dict_final_arrays: Dict[str, np.ndarray]) -> None:
    n_data = len(next(iter(features_dict_final_arrays.values())))
    shuffled_indices = np.random.permutation(n_data)
    for name_feature, feature_final_array in features_dict_final_arrays.items():
        features_dict_final_arrays[name_feature] = feature_final_array[shuffled_indices]


def save_predictions(list_label_preds_test: List[np.ndarray]) -> None:
    """Save the predictions to a CSV file.

    Args:
        list_label_preds_test (List[np.ndarray]): a list of predictions from the different K folds.
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
    df.to_csv("predictions.csv")

