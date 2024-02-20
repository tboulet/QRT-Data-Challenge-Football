from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Type, Callable

import numpy as np
import pandas as pd


class BaseTrainer(ABC):
    """The base class for all trainers. It should be able to train a model and return the metrics."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def train(
        self,
        dataframe: pd.DataFrame,
        labels_train: np.ndarray,
    ):
        """Train the model.

        Args:
            dataframe_train (pd.DataFrame): the input data, as a dataframe of shape (n_data_train, n_features_training).
            labels (pd.Series): the output data, as a series of shape (n_data_train,)
        """

    @abstractmethod
    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            dataframe (pd.DataFrame): the input data, as a dataframe of shape (n_data_train, n_features_training).

        Returns:
            np.ndarray: the predictions, as a numpy array of shape (n_data_train,).
        """
