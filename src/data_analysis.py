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

warnings.filterwarnings("ignore")


def get_metrics_names_to_fn_names(df_features: pd.DataFrame) -> Dict[str, List[str]]:
    """Get the mapping of metrics to their corresponding feature/column names in the dataframe.

    Args:
        df_features (pd.DataFrame): a dataframe of features (be it team or player features)

    Returns:
        Dict[str, List[str]]: a dictionary of metrics to their corresponding feature/column names in the dataframe
    """
    metrics_names_to_fn_names = {}
    for aggregate_function_name in df_features.columns:
        words_splitted = aggregate_function_name.split("_")
        higher_idx_word_capital_letter = np.sum(
            [int(word.isupper()) for word in words_splitted]
        )
        metric_name = "_".join(words_splitted[:higher_idx_word_capital_letter])
        aggregate_function_name = "_".join(
            words_splitted[higher_idx_word_capital_letter:]
        )
        if metric_name not in metrics_names_to_fn_names:
            metrics_names_to_fn_names[metric_name] = []
        metrics_names_to_fn_names[metric_name].append(aggregate_function_name)
    return metrics_names_to_fn_names


def pd_serie_to_distribution(serie: pd.Series, n_value_max: int) -> np.array:
    """Transform a pandas serie whose values are integers in the range [0, n_value_max] into a distribution in [0, n_value_max], as a numpy array of shape (n_value_max+1,)


    Args:
        serie (pd.Series): the serie to transform
        n_value_max (int): the maximum value of the serie

    Returns:
        np.array: the distribution of the values in the serie
    """
    # Drop NaNs and return uniform distribution if no value
    serie = serie.dropna()
    if len(serie) == 0:
        return np.ones(n_value_max + 1) / (n_value_max + 1)
    # Check that the values are in the range [0, n_value_max]
    assert (
        0 <= serie.min() and serie.max() <= n_value_max
    ), f"Values should be in the range [0, {n_value_max}], found {serie.min()} and {serie.max()} instead."
    res = serie.value_counts(
        sort=True, normalize=True, ascending=False, bins=range(n_value_max + 2)
    )
    vector_distrib = res.values / np.sum(res.values)
    assert vector_distrib.shape == (
        n_value_max + 1,
    ), f"Expected shape {(n_value_max + 1,)}, got {vector_distrib.shape} instead."
    return vector_distrib


def get_typical_distribution_abs_shift(vector_distrib1: np.array, n: int):
    return np.sqrt(vector_distrib1 * (1 - vector_distrib1) / n)


def sample_empirical_distribution(p, n):
    """Sample one empirical distribution vector from n samples.

    Args:
        p (ndarray): Probability vector of shape (K,)
        n (int): Number of samples

    Returns:
        ndarray: Empirical distribution vector sampled from the given distribution.
    """
    # Sample n indices based on the probabilities defined by p
    indices = np.random.choice(len(p), size=n, p=p)
    # Count occurrences of each index
    counts = np.bincount(indices, minlength=len(p))
    # Normalize counts to get the empirical distribution vector
    empirical_distribution = counts / n
    return empirical_distribution


loss_name_to_loss_fn: Dict[str, Callable[[np.array, np.array], float]] = {
    "l1_loss": lambda x, y: np.mean(np.abs(x - y)),
    "l2_loss": lambda x, y: np.mean((x - y) ** 2),
    "kl_divergence": lambda x, y: np.sum(x * np.log(x / y)),
    "sum_ratio_abs_diff": lambda x, y: np.sum(
        np.abs(x - y) / (x + sys.float_info.epsilon)
    )
    / len(x),
}


def compute_distribution_difference(
    feature_serie1: pd.Series,
    feature_serie2: pd.Series,
    n_value_max: int = 10,
    do_compute_normalized_difference: bool = True,
    n_monte_carlo: int = 10,
    normalization_method: str = "mc_estimated_loss",
) -> Dict[str, Dict[str, float]]:
    """Compute the difference between two distributions as pandas series, and optionally the normalized difference.
    This function assumes that each serie takes integer values in the range [0, n_value_max].

    Args:
        feature_serie1 (pd.Series): the first distribution
        feature_serie2 (pd.Series): the second distribution
        n_value_max (int, optional): the maximum value of the serie.
        do_compute_normalized_difference (bool, optional): whether to compute the normalized difference. Defaults to True.
        n_monte_carlo (int, optional): the number of monte carlo simulations to estimate the typical loss. Defaults to 10.
        normalization_method (str, optional): the method to normalize the difference. Among ["typical_shift", "mc_estimated_loss"]. Defaults to "mc_estimated_loss".

    Returns:
        Dict[str, Dict[str, float]]: a dictionary that maps loss name and stat name (loss_value, estimated_typical_loss_value, loss_value_normalized)
    """
    # Get the vector distribution
    vector_distribution1 = pd_serie_to_distribution(
        feature_serie1, n_value_max=n_value_max
    )
    vector_distribution2 = pd_serie_to_distribution(
        feature_serie2, n_value_max=n_value_max
    )

    n1 = len(feature_serie1)
    n2 = len(feature_serie2)
    K = len(vector_distribution1)

    # Losses
    loss_name_to_loss_values: Dict[str, Dict[str, float]] = {}

    for loss_name, loss_fn in loss_name_to_loss_fn.items():
        loss_value = loss_fn(vector_distribution1, vector_distribution2)

        # If we don't want to compute the normalized difference, just return the loss value
        if not do_compute_normalized_difference or n_monte_carlo == 0:
            loss_name_to_loss_values[loss_name] = {
                "loss_value": loss_value,
            }
            continue

        if normalization_method == "typical_shift":
            # Method 1 : E[L(p_empirical,p_empirical')] is estimated by L(p_observed, N(p_observed, typical_abs_shift))
            estimated_typical_loss_value = 0
            typical_abs_shift1 = get_typical_distribution_abs_shift(
                vector_distribution1, n1
            )
            typical_abs_shift2 = get_typical_distribution_abs_shift(
                vector_distribution2, n2
            )
            for i in range(n_monte_carlo):
                vector_distribution1_noisy = np.random.normal(
                    vector_distribution1, typical_abs_shift1
                )
                vector_distribution2_noisy = np.random.normal(
                    vector_distribution2, typical_abs_shift2
                )
                vector_distribution1_noisy /= np.sum(vector_distribution1_noisy)
                vector_distribution2_noisy /= np.sum(vector_distribution2_noisy)
                estimated_typical_loss_value += (
                    loss_fn(vector_distribution1, vector_distribution1_noisy)
                    + loss_fn(vector_distribution2, vector_distribution2_noisy)
                ) / 2
            estimated_typical_loss_value /= n_monte_carlo

        elif normalization_method == "mc_estimated_loss":
            # Method 2 : E[L(p_empirical,p_empirical')] is estimated by E[L(p_sampled,p_sampled)] with p and p' sampled from p_empirical (with n1 data) half the time and p_empirical' (with n2 data) the other half
            estimated_typical_loss_value = 0
            for i in range(n_monte_carlo):
                vector_distribution1_sampled_a = sample_empirical_distribution(
                    p=vector_distribution1, n=n1
                )
                vector_distribution1_sampled_b = sample_empirical_distribution(
                    p=vector_distribution1, n=n1
                )
                vector_distribution2_sampled_a = sample_empirical_distribution(
                    p=vector_distribution2, n=n2
                )
                vector_distribution2_sampled_b = sample_empirical_distribution(
                    p=vector_distribution2, n=n2
                )
                estimated_typical_loss_value += (
                    loss_fn(
                        vector_distribution1_sampled_a, vector_distribution1_sampled_b
                    )
                    + loss_fn(
                        vector_distribution2_sampled_a, vector_distribution2_sampled_b
                    )
                ) / 2
            estimated_typical_loss_value /= n_monte_carlo

        else:
            raise ValueError(
                f"normalization_method should be either 'typical_shift' or 'mc_estimated_loss', found {normalization_method} instead."
            )

        loss_value_normalized = loss_value / (
            estimated_typical_loss_value + sys.float_info.epsilon
        )
        loss_name_to_loss_values[loss_name] = {
            "loss_value": loss_value,
            "estimated_typical_loss_value": estimated_typical_loss_value,
            "loss_value_normalized": loss_value_normalized,
        }
    return loss_name_to_loss_values
