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
pass

warnings.filterwarnings('ignore')


def get_metrics_names_to_fn_names(df_teamfeatures : pd.DataFrame) -> Dict[str, List[str]]:
    """Get the mapping of team metrics to their corresponding feature/column names in the dataframe.

    Args:
        df_teamfeatures (pd.DataFrame): a team dataframe

    Returns:
        Dict[str, List[str]]: a dictionary of team metrics to their corresponding feature/column names in the dataframe
    """
    teammetrics_names_to_fn_names = defaultdict(list)
    for aggregate_function_name in df_teamfeatures.columns:
        words_splitted = aggregate_function_name.split("_")
        higher_idx_word_capital_letter = np.sum([int(word.isupper()) for word in words_splitted])
        metric_name = "_".join(words_splitted[:higher_idx_word_capital_letter])
        aggregate_function_name = '_'.join(words_splitted[higher_idx_word_capital_letter:])
        teammetrics_names_to_fn_names[metric_name].append(aggregate_function_name)
    return teammetrics_names_to_fn_names


# Those metrics are not aggregated, and have to be treated differently. Some of them don't appear in the test set.
non_aggregated_teammetric_names = ["HOME_ID", "HOME_TEAM_NAME", "HOME_LEAGUE", "AWAY_ID", "AWAY_TEAM_NAME", "AWAY_LEAGUE"]
non_aggregated_playermetric_names = ["ID_team", "LEAGUE", "TEAM_NAME", "POSITION", "PLAYER_NAME"]




def pd_serie_to_distribution(serie : pd.Series) -> np.array:
    """Transform a pandas serie whose values are integers in the range [0, 10] into a distribution in [0, 10], as a numpy array of shape (11,)

    Args:
        serie (pd.Series): the serie to transform

    Returns:
        np.array: the distribution of the values in the serie
    """
    res = serie.value_counts()
    values = res.index
    assert all([v in range(11) for v in values])   # Check that the values are in the range [0, 10] and are integers
    counts = res.values
    counts = counts[np.argsort(values)]
    vector_distrib = counts / np.sum(counts)
    assert np.isclose(np.sum(vector_distrib), 1)
    assert (vector_distrib >= 0).all()
    assert (vector_distrib <= 1).all()
    # Add 0s if some values are missing
    if len(vector_distrib) < 11:
        vector_distrib = np.concatenate([vector_distrib, np.zeros(11 - len(vector_distrib))])
    return vector_distrib



def get_typical_distribution_abs_shift(vector_distrib1 : np.array, n : int):
    return np.sqrt(vector_distrib1 * (1 - vector_distrib1) / n)



def compute_distribution_difference(
    feature_serie1 : pd.Series,
    feature_serie2 : pd.Series,
    n_monte_carlo : int = 10,
) -> Dict[str, Any]:
    """Compute the difference between two distributions

    Args:
        feature_serie1 (pd.Series): the first distribution
        feature_serie2 (pd.Series): the second distribution

    Returns:
        Dict[str, Any]: a dictionary with the difference metrics
    """
    # Get the vector distribution
    vector_distribution1 = pd_serie_to_distribution(feature_serie1)
    vector_distribution2 = pd_serie_to_distribution(feature_serie2)
    n1 = len(feature_serie1)
    n2 = len(feature_serie2)
    # Compute the typical shift
    typical_abs_shift1 = get_typical_distribution_abs_shift(vector_distribution1, n1)
    typical_abs_shift2 = get_typical_distribution_abs_shift(vector_distribution2, n2)
    # Losses
    loss_name_to_loss_fn = {
    'l1_loss' : lambda x, y: np.mean(np.abs(x - y)),
    'l2_loss' : lambda x, y: np.mean((x - y) ** 2),
    'kl_divergence' : lambda x, y: np.sum(x * np.log(x / y)),
    }
    loss_name_to_normalized_difference : Dict[str, float] = {}
    
    for loss_name, loss_fn in loss_name_to_loss_fn.items():
        loss_value = loss_fn(vector_distribution1, vector_distribution2)
        
        estimated_typical_loss = 0
        for i in range(n_monte_carlo):
            vector_distribution1_noisy = np.random.normal(vector_distribution1, typical_abs_shift1)
            vector_distribution2_noisy = np.random.normal(vector_distribution2, typical_abs_shift2)
            vector_distribution1_noisy /= np.sum(vector_distribution1_noisy)
            vector_distribution2_noisy /= np.sum(vector_distribution2_noisy)
            estimated_typical_loss += (loss_fn(vector_distribution1, vector_distribution1_noisy) + loss_fn(vector_distribution2, vector_distribution2_noisy)) / 2
        estimated_typical_loss /= n_monte_carlo
        normalized_difference = loss_value / (estimated_typical_loss + sys.float_info.epsilon)
        loss_name_to_normalized_difference[loss_name] = normalized_difference
    return loss_name_to_normalized_difference