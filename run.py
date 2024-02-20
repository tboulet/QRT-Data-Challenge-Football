# Logging
from collections import defaultdict
import wandb
from tensorboardX import SummaryWriter

# Config system
import hydra
from omegaconf import OmegaConf, DictConfig

# Utils
from tqdm import tqdm
import datetime
from time import time
from typing import Any, Dict, List, Tuple, Type
import cProfile

# ML libraries
import numpy as np
import random
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold
from src.data_loading import load_dataframe_labels, load_playerfeatures, load_teamfeatures
from src.feature_engineering import (
    add_non_null_indicator_features,
    drop_features,
    impute_missing_values,
)

# Project imports
from trainers import trainer_name_to_TrainerClass
from features_loaders.base_feature_loader import BaseLoader
from src.time_measure import RuntimeMeter
from src.data_management import (
    cut_data_to_n_data_max,
    save_predictions,
    shuffle_data,
)
from src.utils import get_name_trainer_and_features, try_get_seed


def create_features(
    teamfeatures_config: Dict[str, dict],
    playerfeatures_config: Dict[str, dict],
    aggregator_config: Dict[str, dict],
    data_path: str,
) -> pd.DataFrame:
    """Create the features from the config. It does the following :
    - first, preprocess and engineer the team features
    - then, preprocess and engineer the player features
    - finally, aggregate the playerfeatures by match and by team, and concatenate them with the teamfeatures.

    Args:
        teamfeatures_config (Dict[str, dict]): the config for the teamfeatures
        playerfeatures_config (Dict[str, dict]): the config for the playerfeatures
        aggregator_config (Dict[str, dict]): the config for the playerfeatures aggregator method
        data_path (str): the path where all data is.

    Returns:
        pd.DataFrame: the final features, as a dataframe of shape (n_data, n_features)
    """

    # From the datasets, create the teamfeatures
    with RuntimeMeter("teamfeatures creation") as rm:
        print("Creating teamfeatures :")
        # Load the initial data
        df_teamfeatures = load_teamfeatures(
            teamfeatures_config=teamfeatures_config,
            data_path=data_path,
        )
        # Replace infinities with NaN
        df_teamfeatures = df_teamfeatures.replace({np.inf: np.nan, -np.inf: np.nan})
        # Drop features
        df_teamfeatures = drop_features(
            df_features=df_teamfeatures,
            features_config=teamfeatures_config,
        )
        # Add non-null indicator features
        df_teamfeatures = add_non_null_indicator_features(
            df_features=df_teamfeatures,
            features_config=teamfeatures_config,
        )
        # Impute missing values
        df_teamfeatures = impute_missing_values(
            df_features=df_teamfeatures,
            features_config=teamfeatures_config,
        )

    # From the datasets, create the playerfeatures and the grouped playerfeatures
    with RuntimeMeter("playerfeatures creation") as rm:
        print("Creating playerfeatures...")
        # Load the initial data
        df_playerfeatures_home, df_playerfeatures_away = load_playerfeatures(
            playerfeatures_config=playerfeatures_config,
            data_path=data_path,
        )
        # Replace infinities with NaN
        dfs = []
        for df_playerfeature_side in [df_playerfeatures_home, df_playerfeatures_away]:
            df_playerfeature_side = df_playerfeature_side.replace({np.inf: np.nan, -np.inf: np.nan})
            # Drop features
            df_playerfeature_side = drop_features(
                df_features=df_playerfeature_side,
                features_config=playerfeatures_config,
            )
            # Add non-null indicator features
            df_playerfeature_side = add_non_null_indicator_features(
                df_features=df_playerfeature_side,
                features_config=playerfeatures_config,
            )
            # Impute missing values
            df_playerfeature_side = impute_missing_values(
                df_features=df_playerfeature_side,
                features_config=playerfeatures_config,
            )
            # Append to the list
            dfs.append(df_playerfeature_side)
            
        df_playerfeatures_home, df_playerfeatures_away = dfs
            
        # Group playerfeatures by match and by team
        matchID_to_side_to_playerfeatures : Dict[int, Dict[str, pd.DataFrame]] = {}
        for matchID in df_playerfeatures_home["ID"].unique():
            matchID_to_side_to_playerfeatures[matchID] = {
                "home": df_playerfeatures_home[df_playerfeatures_home["ID"] == matchID],
                "away": df_playerfeatures_away[df_playerfeatures_away["ID"] == matchID],
            }
            
        # Concatenate the playerfeatures
        df_playerfeatures = pd.concat([df_playerfeatures_home, df_playerfeatures_away], ignore_index=True)

    # Aggregate the playerfeatures
    with RuntimeMeter("aggregation") as rm:
        print("Aggregating playerfeatures...")
        df_aggplayerfeatures = None

    # Concatenate the teamfeatures and the playerfeatures
    with RuntimeMeter("concatenation") as rm:
        print("Concatenating teamfeatures and playerfeatures...")
        dataframe = pd.concat([df_teamfeatures, df_aggplayerfeatures], join="inner", axis=1)
        print(f"Final features shape: {dataframe.shape}")

    return dataframe


@hydra.main(config_path="configs", config_name="config_default.yaml")
def main(config: DictConfig):

    # Get the config values from the config object.
    config = OmegaConf.to_container(config, resolve=True)
    do_shuffle: bool = config["do_shuffle"]
    n_data_max: int = config["n_data_max"]
    K: int = config["cross_val_folds"]
    do_test_pred: bool = config["do_test_pred"]

    do_cli: bool = config["do_cli"]
    do_wandb: bool = config["do_wandb"]
    do_tb: bool = config["do_tb"]
    do_tqdm: bool = config["do_tqdm"]

    # Set the seeds
    seed = try_get_seed(config)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Using seed: {seed}")

    # Get the trainer and initialize it
    name_trainer: str = config["trainer"]["name"]
    TrainerClass = trainer_name_to_TrainerClass[name_trainer]
    trainer_config = config["trainer"]["config"]
    trainer = TrainerClass(trainer_config)

    # Get the teamfeature config, playerfeature config, and aggregator config

    # Initialize loggers
    run_name = f"[{name_trainer}]_{datetime.datetime.now().strftime('%dth%mmo_%Hh%Mmin%Ss')}_seed{seed}"
    print(f"\nStarting run {run_name}")

    # Load the features
    df = create_features(
        teamfeatures_config=config["teamfeatures_config"],
        playerfeatures_config=config["playerfeatures_config"],
        aggregator_config=config["aggregator_config"],
        data_path="data_train",
    )
    if do_test_pred:
        df_test = create_features(
            dict_loaders=dict_loaders_without_labels,
            dict_creators=dict_creators,
            data_path="data_test",
        )

    # Shuffle the data
    print(df)
    if do_shuffle:
        print("Shuffling the data...")
        shuffle_data(df)

    # Limit the data to a subset for debugging
    cut_data_to_n_data_max(df, n_data_max)

    # Get the y_data out of the features, and remove it from the features.
    labels = load_dataframe_labels()

    # Start the KFold loop for Cross Validation
    dict_list_metrics = defaultdict(list)
    list_label_preds_test: List[np.ndarray] = []
    kf = KFold(n_splits=5)
    # for k in range(K):
    for k, (train_index, val_index) in enumerate(kf.split(df)):
        print(f"\nStarting fold {k+1}/{config['cross_val_folds']}")

        # Split the data
        df_train = df.iloc[train_index]
        df_val = df.iloc[val_index]
        labels_train = labels[train_index]
        labels_val = labels[val_index]

        # Train the model
        with RuntimeMeter("training") as rm:
            trainer.train(
                dataframe=df_train,
                labels_train=labels_train,
            )

        # Evaluate the model
        metric_results = {}
        with RuntimeMeter("evaluation") as rm:

            # Train metrics
            preds_train = trainer.predict(df_train)
            accuracy_train = accuracy_score(labels_train, preds_train)
            metric_results["accuracy_train"] = accuracy_train

            # Cross validation metrics
            if K >= 2:
                preds_test = trainer.predict(dataframe=df_val)
                accuracy_test = accuracy_score(labels_val, preds_test)
                metric_results["accuracy"] = accuracy_test
                print(f"Accuracy: {accuracy_test}")

            # Test metrics
            if do_test_pred:
                labels_pred_test = trainer.predict(df_test)
                list_label_preds_test.append(labels_pred_test)
                save_predictions(list_label_preds_test)

            # Save time metrics
            metric_results["loading_time"] = rm.get_stage_runtime("loading")
            metric_results["creation_time"] = rm.get_stage_runtime("creation")
            metric_results["training_time"] = rm.get_stage_runtime("training")
            metric_results["evaluation_time"] = rm.get_stage_runtime("evaluation")
            metric_results["log_time"] = rm.get_stage_runtime("log")

        # Log metrics
        if do_cli:
            print(f"Metrics: {metric_results}")

    # Conclude
    print()
    for metric_name, list_metric in dict_list_metrics.items():
        print(f"Metric {metric_name}: {np.mean(list_metric)} +- {np.std(list_metric)}")


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()
    pr.dump_stats("logs/profile_stats.prof")
    print("\nProfile stats dumped to profile_stats.prof")
    print(
        "You can visualize the profile stats using snakeviz by running 'snakeviz logs/profile_stats.prof'"
    )
