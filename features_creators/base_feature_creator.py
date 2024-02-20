from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Type, Callable

import numpy as np


class BaseFeatureCreator(ABC):
    """The base class for all feature creators. It should be able to create features from the loaded features."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def create_usable_features(self, feature_dict_intermediary_objects : Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Create the usable features from the already loaded features. Using a particular feature creator may need to
        have some features already loaded in the 'feature_name_to_feature_object' dictionnary (memory), and then create
        additional features from them that cannot be kept on disk.

        Args:
            feature_dict_intermediary_objects (Dict[str, Any]): a dictionnary of feature names to feature objects, as loaded by the feature loader.

        Returns:
            Dict[str, np.ndarray]: the features, as a dictionnary of numpy arrays of shape (n_data, n_features).
        """
