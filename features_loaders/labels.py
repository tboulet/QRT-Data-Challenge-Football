from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Type, Callable

import numpy as np
import pandas as pd
import xgboost as xgb

from features_loaders.base_feature_loader import BaseLoader


class LoaderLabels(BaseLoader):
    """The class for loading the labels."""
    
    def load_features(self, data_path : str) -> Dict[str, Any]:
        # Load without index column
        labels_csv = pd.read_csv(data_path + '/labels.csv', index_col=0, header=0)
        # Convert to numpy
        self.labels_numpy = labels_csv.to_numpy()
        # Convert to indexes
        self.labels_numpy = np.argmax(self.labels_numpy, axis=1)
        return {"labels_numpy": self.labels_numpy}
    
    def get_usable_features(self) -> Dict[str, np.ndarray]:
        return {"labels": self.labels_numpy}