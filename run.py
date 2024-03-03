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
from src.constants import SPECIFIC_PLAYERFEATURES


# Project imports
from trainers import trainer_name_to_TrainerClass
from src.time_measure import RuntimeMeter
from src.data_loading import (
    load_dataframe_labels,
    load_index_numpy_labels,
    load_playerfeatures,
    load_teamfeatures,
)
from src.feature_engineering import (
    add_home_and_away_team_name_identifier_features,
    add_non_null_indicator_features,
    get_agg_playerfeatures_by_operation,
    drop_features,
    impute_missing_values,
    group_playerfeatures_by_match_and_by_team,
)
from src.data_management import (
    add_prefix_to_columns,
    cut_data_to_n_data_max,
    merge_dfs,
    save_predictions,
    shuffle_data,
)
from src.utils import (
    get_name_trainer_and_features,
    to_numeric,
    try_get_seed,
)


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
    print(f"\n============ Creating features from {data_path} ============")

    # From the datasets, create the teamfeatures
    with RuntimeMeter("teamfeatures creation") as rm:
        print("\nCreating teamfeatures :")
        # Load the initial data
        df_teamfeatures = load_teamfeatures(
            teamfeatures_config=teamfeatures_config,
            data_path=data_path,
        )

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
        # Add team name features
        df_teamfeatures = add_home_and_away_team_name_identifier_features(
            df_features=df_teamfeatures,
            features_config=teamfeatures_config,
        )

        print("[Shapes] Teamfeatures shape: ", df_teamfeatures.shape)

    # From the datasets, create the playerfeatures and the grouped playerfeatures
    with RuntimeMeter("playerfeatures creation") as rm:
        print("\nCreating playerfeatures...")
        # Load the initial data
        df_playerfeatures_home, df_playerfeatures_away = load_playerfeatures(
            playerfeatures_config=playerfeatures_config,
            data_path=data_path,
        )

        dfs = []
        for df_playerfeature_side in [df_playerfeatures_home, df_playerfeatures_away]:

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

        # # Group playerfeatures by match and by team
        # print("Grouping playerfeatures by match and by team...")
        # matchID_to_side_to_playerfeatures = group_playerfeatures_by_match_and_by_team(
        #     df_playerfeatures_home=df_playerfeatures_home,
        #     df_playerfeatures_away=df_playerfeatures_away,
        # )

        # Concatenate the playerfeatures
        print("\tConcatenating playerfeatures...")
        df_playerfeatures = pd.concat(
            [df_playerfeatures_home, df_playerfeatures_away], ignore_index=True
        )
        print(f"\t[Shapes] Playerfeatures shape: {df_playerfeatures.shape}")

    # Aggregate the playerfeatures
    list_df_agg_playerfeatures: List[pd.DataFrame] = []
    with RuntimeMeter("aggregation") as rm:
        print("\nAggregating playerfeatures...")

        # Aggregate the team of home players with operations (mean, sum, etc.)
        df_agg_playerfeatures_home = get_agg_playerfeatures_by_operation(
            df_playerfeatures=df_playerfeatures_home,
            aggregator_config=aggregator_config,
        )
        add_prefix_to_columns(df_agg_playerfeatures_home, "HOME_")
        list_df_agg_playerfeatures.append(df_agg_playerfeatures_home)

        # Aggregate the team of away players
        df_agg_playerfeatures_away = get_agg_playerfeatures_by_operation(
            df_playerfeatures=df_playerfeatures_away,
            aggregator_config=aggregator_config,
        )
        add_prefix_to_columns(df_agg_playerfeatures_away, "AWAY_")
        list_df_agg_playerfeatures.append(df_agg_playerfeatures_away)

    # Merge and clean the teamfeatures and the aggregated playerfeatures
    with RuntimeMeter("merging") as rm:
        print("\nMerge and clean the teamfeatures and the aggregated playerfeatures...")

        # Concatenate the teamfeatures and the agg_playerfeatures
        dataframe = merge_dfs(
            list_dataframes=[df_teamfeatures] + list_df_agg_playerfeatures,
            on="ID",
        )

        # Drop non numeric and ID columns
        dataframe.drop(
            columns=[
                "ID",
                "HOME_ID",
                "HOME_LEAGUE",
                "HOME_TEAM_NAME",
                "AWAY_ID",
                "AWAY_LEAGUE",
                "AWAY_TEAM_NAME",
            ],
            inplace=True,
            errors="ignore",
        )

    print(f"[Shapes] Final features shape: {dataframe.shape}")
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
            teamfeatures_config=config["teamfeatures_config"],
            playerfeatures_config=config["playerfeatures_config"],
            aggregator_config=config["aggregator_config"],
            data_path="data_test",
        )

    # Shuffle the data
    if do_shuffle:
        print("Shuffling the data...")
        shuffle_data(df)

    # Limit the data to a subset for debugging
    cut_data_to_n_data_max(df, n_data_max)

    # Get the y_data out of the features, and remove it from the features.
    labels = load_index_numpy_labels(global_data_path="./data_train/")

    # Start the KFold loop for Cross Validation
    dict_list_metrics = defaultdict(list)
    list_label_preds_test: List[np.ndarray] = []
    kf = KFold(n_splits=K)

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
                dataframe_train=df_train,
                labels_train=labels_train,
            )

        # Evaluate the model
        metric_results = {}
        with RuntimeMeter("evaluation") as rm:

            # Train metrics
            preds_train = trainer.predict(df_train)
            accuracy_train = accuracy_score(labels_train, preds_train)
            metric_results["accuracy_train"] = accuracy_train
            print(f"Accuracy train: {accuracy_train}")
            
            # Cross validation metrics
            if K >= 2:
                preds_val = trainer.predict(dataframe=df_val)
                accuracy_val = accuracy_score(labels_val, preds_val)
                metric_results["accuracy_val"] = accuracy_val
                print(f"Accuracy val: {accuracy_val}")

            # Test metrics
            if do_test_pred:
                labels_pred_test = trainer.predict(df_test)
                list_label_preds_test.append(labels_pred_test)
                

            # Save time metrics
            metric_results.update({f"time {key}": value for key, value in rm.stage_name_to_runtime.items()})
            for metric_name, metric_value in metric_results.items():
                dict_list_metrics[metric_name].append(metric_value)
                
        # Log metrics
        if do_cli:
            print(f"Metrics: {metric_results}")

    if do_test_pred:
        save_predictions(list_label_preds_test, path="predictions.csv")
        
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
