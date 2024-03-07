from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Type, Callable

import numpy as np
import pandas as pd
import xgboost as xgb

from trainers.base_trainer import BaseTrainer


class StatisticalTrainer(BaseTrainer):

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
        # Count the number of each label
        self.count_labels = np.bincount(labels_train)

    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            dataframe (pd.DataFrame): the input data, as a dataframe of shape (n_data_train, n_features_training).

        Returns:
            np.ndarray: the predictions, as a numpy array of shape (n_data_train,).
        """
        
        
        if self.config["bayes"]:
            # Return the max of the three columns, converted to 0, 1 or 2
            return dataframe[["HOME_WINS", "DRAW", "AWAY_WINS"]].idxmax(axis=1).apply(lambda x: 0 if x == "HOME_WINS" else (1 if x == "DRAW" else 2))
        else:
            count_labels = self.count_labels
            # Return the max of each column i divided by count_labels[i]
            return dataframe[["HOME_WINS_RATE", "DRAW_RATE", "AWAY_WINS_RATE"]].apply(lambda x: np.argmax(x / count_labels), axis=1)