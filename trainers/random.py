from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Type, Callable

import numpy as np
import xgboost as xgb

from trainers.base_trainer import BaseTrainer


class RandomTrainer(BaseTrainer):
    """A trainer that predicts random values. It is used as a baseline."""

    def __init__(self, config: dict):
        self.config = config

    def train(
        self, features_dict_final_arrays: Dict[str, np.ndarray], labels: np.ndarray
    ):
        """Train the model.

        Args:
            features_dict_final_arrays (Dict[str, np.ndarray]): the input data, as dictionnary of numpy arrays of shape (n_data_train, n_features_training).
            labels (np.ndarray): the output data, as a numpy array of shape (n_data_train,)
        """
        pass
    
    def predict(self, feature_name_to_array: Dict[str, np.ndarray]) -> np.ndarray:
        """Make predictions.

        Args:
            feature_name_to_array (Dict[str, np.ndarray]): the input data, as dictionnary of numpy arrays of shape (n_data_train, n_features_training).

        Returns:
            np.ndarray: the predictions, as a numpy array of shape (n_data_train, n_output_features).
        """
        n_data = feature_name_to_array[list(feature_name_to_array.keys())[0]].shape[0]
        return np.random.randint(0, 3, n_data)
        