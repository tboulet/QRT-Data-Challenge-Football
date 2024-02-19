from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Type, Callable

import numpy as np
import pandas as pd

from src.constants import (OFFENSIVE_POSITIVE_PLAYERFEATURES,
                           OFFENSIVE_NEGATIVE_PLAYERFEATURES,
                           DEFENSIVE_POSITIVE_PLAYERFEATURES, 
                           DEFENSIVE_NEGATIVE_PLAYERFEATURES, 
                           SHOOTING_POSITIVE_PLAYERFEATURES, 
                           SHOOTING_NEGATIVE_PLAYERFEATURES, 
                           FOOTWORK_POSITIVE_PLAYERFEATURES, 
                           FOOTWORK_NEGATIVE_PLAYERFEATURES, 
                           TIREDNESS_PLAYERFEATURES, 
                           GOALKEEPER_PLAYERFEATURES, 
                           BEHAVIOR_POSITIVE_PLAYERFEATURES, 
                           BEHAVIOR_NEGATIVE_PLAYERFEATURES, 
                           MISC_PLAYERFEATURES)


class PlayerFeaturesCreator(ABC):
    """The class for creating the player features."""

    def create_usable_features(self, features_dict_final_arrays : Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Create the usable features from the already loaded features. Using a particular feature creator may need to
        have some features already loaded in the 'feature_name_to_feature_object' dictionnary (memory), and then create
        additional features from them that cannot be kept on disk.

        Args:
            features_dict_final_arrays (Dict[str, np.ndarray]): a dictionnary of feature names to feature objects, as loaded by the feature loader.

        Returns:
            Dict[str, np.ndarray]: the features, as a dictionnary of numpy arrays of shape (n_data, n_features).
        """
        self.dataframe_playerfeatures = features_dict_final_arrays

        # Create smart features
        offensive_positive_playerfeature = extract_sum(OFFENSIVE_POSITIVE_PLAYERFEATURES)
        offensive_negative_playerfeature = extract_sum(OFFENSIVE_NEGATIVE_PLAYERFEATURES)
        defensive_positive_playerfeature = extract_sum(DEFENSIVE_POSITIVE_PLAYERFEATURES)
        defensive_negative_playerfeature = extract_sum(DEFENSIVE_NEGATIVE_PLAYERFEATURES)
        shooting_positive_playerfeature = extract_sum(SHOOTING_POSITIVE_PLAYERFEATURES)
        shooting_negative_playerfeature = extract_sum(SHOOTING_NEGATIVE_PLAYERFEATURES)
        footwork_positive_playerfeature = extract_sum(FOOTWORK_POSITIVE_PLAYERFEATURES)
        footwork_negative_playerfeature = extract_sum(FOOTWORK_NEGATIVE_PLAYERFEATURES)
        tiredness_playerfeature = extract_sum(TIREDNESS_PLAYERFEATURES)
        goalkeeper_playerfeature = extract_sum(GOALKEEPER_PLAYERFEATURES)
        behavior_positive_playerfeature = extract_sum(BEHAVIOR_POSITIVE_PLAYERFEATURES)
        behavior_negative_playerfeature = extract_sum(BEHAVIOR_NEGATIVE_PLAYERFEATURES)
        misc_playerfeature = extract_sum(MISC_PLAYERFEATURES)

        # Add the new features to the dataframe
        self.dataframe_playerfeatures["offensive_playerfeature"] = offensive_positive_playerfeature / offensive_negative_playerfeature
        self.dataframe_playerfeatures["defensive_playerfeature"] = defensive_positive_playerfeature / defensive_negative_playerfeature
        self.dataframe_playerfeatures["shooting_playerfeature"] = shooting_positive_playerfeature / shooting_negative_playerfeature
        self.dataframe_playerfeatures["footwork_playerfeature"] = footwork_positive_playerfeature / footwork_negative_playerfeature
        self.dataframe_playerfeatures["tiredness_playerfeature"] = tiredness_playerfeature
        self.dataframe_playerfeatures["goalkeeper_playerfeature"] = goalkeeper_playerfeature
        self.dataframe_playerfeatures["behavior_playerfeature"] = behavior_positive_playerfeature / behavior_negative_playerfeature
        self.dataframe_playerfeatures["misc_playerfeature"] = misc_playerfeature

        return {"playerfeatures" : self.dataframe_playerfeatures}
    
    def extract_sum(self, list: List[float]) -> float:
        """Extract the sum of a list of floats."""
        return 1 + np.sum(self.dataframe_playerfeatures[list], axis=1)