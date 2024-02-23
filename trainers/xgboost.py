from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Type, Callable

import numpy as np
import pandas as pd
import xgboost as xgb

from trainers.base_trainer import BaseTrainer


class XGBoostTrainer(BaseTrainer):
    """The class for training a XGBoost model."""

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

        # (n_data_train,) = labels_train.shape
        # x_data_array = self.convert_feature_name_to_array_to_x_data_array(
        #     features_dict_final_arrays,
        # )

        # print("Shape of x_data_array: ", x_data_array.shape)

        # dmatrix_train = xgb.DMatrix(x_data_array, label=labels_train)

        # # Parameters
        # params = {
        #     "objective": "multi:softmax",  # for multi-class classification
        #     "num_class": 3,  # number of classes
        #     "eta": 0.1,  # learning rate
        #     "max_depth": 10,  # maximum depth of a tree
        #     "subsample": 0.8,  # subsample ratio of the training instances
        #     "colsample_bytree": 0.8,  # subsample ratio of columns when constructing each tree
        #     "eval_metric": "merror",  # evaluation metric
        # }

        # num_rounds = 100  # number of boosting rounds
        # self.model = xgb.train(params, dmatrix_train, num_rounds)

        self.clf = xgb.XGBClassifier(
            objective="multi:softmax",  # Specify multiclass classification
            num_class=3,  # Number of classes in the dataset
            learning_rate=0.1,  # Learning rate
            max_depth=3,  # Maximum depth of a tree
        )
        self.clf.fit(dataframe_train, labels_train)


    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            dataframe (pd.DataFrame): the input data, as a dataframe of shape (n_data_train, n_features_training).

        Returns:
            np.ndarray: the predictions, as a numpy array of shape (n_data_train,).
        """
        return self.clf.predict(dataframe)