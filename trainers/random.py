from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Type, Callable

import numpy as np
import pandas as pd
import xgboost as xgb

from trainers.base_trainer import BaseTrainer


class RandomTrainer(BaseTrainer):
    """A trainer that predicts random values. It is used as a baseline."""

    def __init__(self, config: dict):
        self.config = config

    def train(
        self,
        dataframe_train: pd.DataFrame,
        labels_train: np.ndarray,
    ):
        """Train the model.

        Args:
            dataframe_train (pd.DataFrame): the input data, as a dataframe of shape (n_data_train, n_features_training).
            labels_train (pd.Series): the output data, as a series of shape (n_data_train,)
        """
        pass
    
    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            dataframe (pd.DataFrame): the input data, as a dataframe of shape (n_data_train, n_features_training).

        Returns:
            np.ndarray: the predictions, as a numpy array of shape (n_data_train,).
        """
        return np.random.randint(0, 3, dataframe.shape[0])
        