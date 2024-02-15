from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Type, Callable

import numpy as np
import pandas as pd
import xgboost as xgb

from features_loaders.base_feature_loader import BaseLoader


class LoaderTeamfeatures(BaseLoader):
    """The class for loading the team features."""
    
    def load_features(self, data_path : str) -> Dict[str, Any]:
        """Load the features in memory. They can then be used by any feature creator for creating additional features.

        Args:
            data_path (str): the path where all data is.
            
        Returns:
            Tuple[Dict[str, Any]]: the features, as a dictionnary of numpy arrays of shape (n_data, n_features).
        """
        pass
    
        home_team_statistics_df = pd.read_csv(data_path + f'/home_team_statistics_df.csv')
        away_team_statistics_df = pd.read_csv(data_path + f'/away_team_statistics_df.csv')

        home_teamfeatures = home_team_statistics_df.iloc[:]
        away_teamfeatures = away_team_statistics_df.iloc[:]

        home_teamfeatures.columns = 'HOME_' + home_teamfeatures.columns
        away_teamfeatures.columns = 'AWAY_' + away_teamfeatures.columns

        dataframe_teamfeatures =  pd.concat([home_teamfeatures,away_teamfeatures],join='inner',axis=1)
        dataframe_teamfeatures = dataframe_teamfeatures.replace({np.inf:np.nan,-np.inf:np.nan})
        self.dataframe_teamfeatures = dataframe_teamfeatures
        return {"teamfeatures": self.dataframe_teamfeatures}
    
    def get_usable_features(self) -> Dict[str, np.ndarray]:
        # Drop columns that are not for training
        for side in ["HOME", "AWAY"]:
            for suffix_feature in ["ID", "LEAGUE", "TEAM_NAME"]:
                name_feature = f"{side}_{suffix_feature}"
                self.dataframe_teamfeatures = self.dataframe_teamfeatures.drop(name_feature, axis=1)
        # Turn into numpy array
        self.dataframe_teamfeatures = self.dataframe_teamfeatures.to_numpy()
        # Return the features
        return {"teamfeatures": self.dataframe_teamfeatures}