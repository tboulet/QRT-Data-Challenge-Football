# ML libraries
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import xgboost as xgb
# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
# Utils
import os
import sys
from collections import defaultdict
from tqdm import tqdm
import warnings
from typing import List, Dict, Any, Tuple, Union, Optional, Callable, TypeVar
# Project modules
from src.data_loading import load_dataframe_teamfeatures, load_dataframe_playersfeatures
from src.data_analysis import get_metrics_names_to_fn_names
from src.constants import SPECIFIC_TEAMFEATURES, SPECIFIC_PLAYERFEATURES

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    
    # Initialize an empty dictionary to store maximum correlations
    max_correlations_all = {}
    
    # Compute the maximum correlations among same metric, for the team features
    df_teamfeatures_train = load_dataframe_teamfeatures('train')
    metric_names_to_fn_names = get_metrics_names_to_fn_names(df_features=df_teamfeatures_train)

    for metric_name, fn_names in metric_names_to_fn_names.items():
        if metric_name not in SPECIFIC_TEAMFEATURES:
            names_features_from_metric = [f"{metric_name}_{fn_name}" for fn_name in fn_names]
            corr = df_teamfeatures_train[names_features_from_metric].corr()

            # Find the highest correlation for each feature
            max_correlations = {}
            for feature in corr.columns:
                max_corr = corr[feature].drop(feature).max()
                max_correlated_feature = corr[feature].drop(feature).idxmax()
                max_correlations[feature] = (max_correlated_feature, max_corr)

            # Update the global dictionary with maximum correlations for this metric
            max_correlations_all[metric_name] = max_correlations

    # Compute the maximum correlations among same metric, for the players features
    df_playersfeatures_train, _ = load_dataframe_playersfeatures('train')
    metric_names_to_fn_names = get_metrics_names_to_fn_names(df_features=df_playersfeatures_train)
     
    for metric_name, fn_names in metric_names_to_fn_names.items():
        if metric_name not in SPECIFIC_PLAYERFEATURES:
            names_features_from_metric = [f"{metric_name}_{fn_name}" for fn_name in fn_names]
            corr = df_playersfeatures_train[names_features_from_metric].corr()

            # Find the highest correlation for each feature
            max_correlations = {}
            for feature in corr.columns:
                max_corr = corr[feature].drop(feature).max()
                max_correlated_feature = corr[feature].drop(feature).idxmax()
                max_correlations[feature] = (max_correlated_feature, max_corr)

            # Update the global dictionary with maximum correlations for this metric
            max_correlations_all[metric_name] = max_correlations


    # Convert the dictionary to a DataFrame
    max_correlations_df = pd.DataFrame.from_dict({feature: values 
                                                   for metric_name, features in max_correlations_all.items() 
                                                   for feature, values in features.items()}, 
                                                  orient='index', columns=['Max Correlated Feature', 'Correlation'])

    # Save the DataFrame to a CSV file
    csv_filename = 'data/max_correlations_global.csv'
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    max_correlations_df.to_csv(csv_filename)
