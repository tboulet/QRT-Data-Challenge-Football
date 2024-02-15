from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Type, Callable

import numpy as np


class BaseTrainer(ABC):
    """The base class for all trainers. It should be able to train a model and return the metrics."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def train(
        self, x_data: Dict[str, np.ndarray], y_data: np.ndarray
    ):
        """Train the model.

        Args:
            x_data (Any): the input data, as dictionnary of numpy arrays of shape (n_data_train, n_features_training).
            y_data (Any): the output data, as a numpy array of shape (n_data_train,)
        """

    @abstractmethod
    def predict(self, feature_name_to_array: Dict[str, np.ndarray]) -> np.ndarray:
        """Make predictions on the given features, and return the predictions.
        The predictions should be of shape (n_data,) and be in the form of indexes, not one-hot encoded nor probabilities.

        Args:
            x_data (Any): the input data, as dictionnary of numpy arrays of shape (n_data_train, n_features_training).

        Returns:
            np.ndarray: the predictions, as a numpy array of shape (n_data_train,)
        """