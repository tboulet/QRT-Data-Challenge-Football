from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Type, Callable

import numpy as np
import xgboost as xgb

from trainers.base_trainer import BaseTrainer


class XGBoostTrainer(BaseTrainer):
    """The class for training a XGBoost model."""
    
    def __init__(self, config: dict):
        self.config = config

    def train(
        self, features_dict_final_arrays: Dict[str, np.ndarray], labels: np.ndarray
    ):
        """Train the model.

        Args:
            features_dict_final_arrays (Dict[str, np.ndarray]): the input data, as dictionnary of numpy arrays of shape (n_data_train, n_features_training).
            labels (np.ndarray): the output data, as a numpy array of shape (n_data_train,)
        """

        n_data_train, = labels.shape
        x_data_array = self.convert_feature_name_to_array_to_x_data_array(
            features_dict_final_arrays,
        )

        print("Shape of x_data_array: ", x_data_array.shape)

        dmatrix_train = xgb.DMatrix(x_data_array, label=labels)

        # Parameters
        params = {
            "objective": "reg:squarederror",  # for regression task
            "eval_metric": "rmse",  # evaluation metric
            "eta": 0.1,  # learning rate
            "max_depth": 6,  # maximum depth of the tree
            "subsample": 0.8,  # subsample ratio of the training instances
            "colsample_bytree": 0.8,  # subsample ratio of columns when constructing each tree
            "seed": 42,  # random seed for reproducibility
        }

        num_rounds = 100  # number of boosting rounds
        self.model = xgb.train(params, dmatrix_train, num_rounds)

    def predict(self, feature_name_to_array: Dict[str, np.ndarray]) -> np.ndarray:
        """Make predictions.

        Args:
            feature_name_to_array (Dict[str, np.ndarray]): the input data, as dictionnary of numpy arrays of shape (n_data_train, n_features_training).

        Returns:
            np.ndarray: the predictions, as a numpy array of shape (n_data_train, n_output_features).
        """
        # Convert the features dict to a x_data_array
        x_data_array = self.convert_feature_name_to_array_to_x_data_array(
            feature_name_to_array
        )
        # Predict the values, as a regression task
        values_predicted = self.predict_y_values(x_data_array)
        # Round the values to the nearest integer to obtain the indexes
        indexes_predicted = np.round(values_predicted).astype(int)
        return indexes_predicted
        

    def convert_feature_name_to_array_to_x_data_array(
        self,
        features_dict_final_arrays: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Convert the features dict to a x_data_array"""
        return np.concatenate(
            [
                features_dict_final_arrays[feature_name]
                for feature_name in features_dict_final_arrays
            ],
            axis=1,
        )
        
    def predict_y_values(self, x_data_array: np.ndarray) -> np.ndarray:
        """Make predictions on the given features, and return the predictions."""
        dmatrix = xgb.DMatrix(x_data_array)
        values_predicted = self.model.predict(dmatrix)
        return values_predicted