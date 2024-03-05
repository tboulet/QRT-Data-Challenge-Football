from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Type, Callable

import numpy as np
import pandas as pd

from trainers.base_trainer import BaseTrainer


class SelectedClassTrainer(BaseTrainer):
    """A trainer that predicts one selected class. It is used for exploratory purposes."""

    def __init__(self, config: dict):
        self.config = config

    def train(
        self,
        dataframe_train: pd.DataFrame,
        labels_train: np.ndarray,
    ):
        """Set the desired class, in [0 : 'HOME_WINS', 1 : 'DRAW', 2 : 'AWAY_WINS'].

        Args:
            dataframe_train (pd.DataFrame): the input data, as a dataframe of shape (n_data_train, n_features_training).
            labels_train (pd.Series): the output data, as a series of shape (n_data_train,)
        """
        # self.majority_class = np.argmax(np.bincount(labels_train))
        self.selected_class = 1

    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            dataframe (pd.DataFrame): the input data, as a dataframe of shape (n_data_train, n_features_training).

        Returns:
            np.ndarray: the predictions, as a numpy array of shape (n_data_train,).
        """
        return np.full(dataframe.shape[0], self.selected_class)
