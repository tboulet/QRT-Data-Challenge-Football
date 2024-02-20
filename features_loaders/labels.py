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
        self.labels_df = pd.read_csv(data_path + '/labels.csv', index_col=0, header=0)
        # We have three one-hot encoded columns, convert to indices
        self.labels_df['labels'] = None
        self.labels_df['labels'][self.labels_df['HOME_WINS'] == 1] = 0
        self.labels_df['labels'][self.labels_df['DRAW'] == 1] = 1
        self.labels_df['labels'][self.labels_df['AWAY_WINS'] == 1] = 2
        self.labels_df = self.labels_df.drop(columns=['HOME_WINS', 'DRAW', 'AWAY_WINS'])
        return {"labels": self.labels_df}
    
    def get_usable_features(self) -> Dict[str, pd.DataFrame]:
        return {"labels": self.labels_df}