from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Type, Callable

import numpy as np
import xgboost as xgb


class BaseLoader(ABC):
    """The base class for all feature loaders. It should be able to load the features and return them."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def load_features(self, data_path: str) -> Dict[str, Any]:
        """Load the features in memory. They could then be used by any feature creator for creating additional features,
        or by this (or other?) feature loader for getting the usable features for training the model.

        Args:
            dataset (str): either 'train' or 'test'.

        Returns:
            Dict[str, Any]: the features, as a dictionnary of any type.
        """

    @abstractmethod
    def get_usable_features(self) -> Dict[str, np.ndarray]:
        """Get the usable features (from the memory) that can be used for training the model.

        At this stage, the features can be modified in place without problems for any feature creator, as they have
        already computed their features from the loaded features.

        Returns:
            Dict[str, np.ndarray]: a dictionnary of numpy arrays of shape (n_data, n_features).
        """
