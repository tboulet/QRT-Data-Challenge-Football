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

# Project imports
from trainers import trainer_name_to_TrainerClass
from features_loaders import loader_name_to_LoaderClass
from features_creators import feature_creator_name_to_FeatureCreatorClass
from features_loaders.base_feature_loader import BaseLoader
from src.time_measure import RuntimeMeter
from src.data_management import (
    cut_data_to_n_data_max,
    get_train_val_split,
    save_predictions,
    shuffle_data,
)
from src.utils import get_name_trainer_and_features, try_get_seed


def create_features_dict_final_arrays(
    dict_loaders: Dict[str, dict],
    dict_creators: Dict[str, dict],
    data_path: str,
) -> Dict[str, np.ndarray]:
    """Load and create the features to be given as input to the model, using the loaders and creators configurations.

    Args:
        dict_loaders (Dict[str, dict]): the dict of loaders configurations. Each configuration should at least contain the booleans "load" field and "use" field.
        dict_creators (Dict[str, dict]): the dict of creators configurations. Each configuration should at least contain a boolean "use" field.
        data_path (str): the path to the data.

    Returns:
        Dict[str, np.ndarray]: the final features, as a dictionnary of numpy arrays of shape (n_data, *_), that can be used as input to the model.
    """

    # Initialize as empty the intermediary objects and the final arrays
    feature_dict_intermediary_objects: Dict[str, Any] = {}
    features_dict_final_arrays: Dict[str, np.ndarray] = {}

    # Load the features with the feature loaders
    with RuntimeMeter("loading") as rm:
        print("Loading features...")
        loader_name_to_loader: Dict[str, BaseLoader] = {}
        for name_loader, config_loader in dict_loaders.items():
            if config_loader["load"]:
                LoaderClass = loader_name_to_LoaderClass[name_loader]
                loader: BaseLoader = LoaderClass(config_loader)
                feature_dict_intermediary_objects.update(
                    loader.load_features(data_path=data_path)
                )
                loader_name_to_loader[name_loader] = loader

    # Run the feature creators to create final features from the loaded features
    with RuntimeMeter("creation") as rm:
        print("Creating features...")
        for name_creator, config_creator in dict_creators.items():
            if config_creator["use"]:
                creator = feature_creator_name_to_FeatureCreatorClass[name_creator]
                features_dict_final_arrays.update(
                    creator.create_usable_features(feature_dict_intermediary_objects)
                )

    # Get the final features from the feature creators
    with RuntimeMeter("reloadings") as rm:
        print("Reloading features...")
        for name_loader, loader in loader_name_to_loader.items():
            if dict_loaders[name_loader]["use"]:
                features_dict_final_arrays.update(loader.get_usable_features())
        n_data = len(next(iter(features_dict_final_arrays.values())))
        print(f"Number of data: {n_data}")
        print()

    return features_dict_final_arrays


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

    # Get the loaders and creators config
    dict_loaders: Dict[str, dict] = config["loaders"]
    dict_loaders_without_labels = dict_loaders.copy()
    dict_loaders_without_labels.pop("labels")
    dict_creators: Dict[str, dict] = config["creators"]

    # Initialize loggers
    name_trainer_and_features = get_name_trainer_and_features(
        name_trainer, dict_loaders, dict_creators
    )
    run_name = f"[{name_trainer_and_features}]_{datetime.datetime.now().strftime('%dth%mmo_%Hh%Mmin%Ss')}_seed{seed}"
    print(f"\nStarting run {run_name}")
    if do_wandb:
        run = wandb.init(
            name=run_name,
            config=config,
            **config["wandb_config"],
        )
    if do_tb:
        tb_writer = SummaryWriter(log_dir=f"tensorboard/{run_name}")

    # Load the features
    features_dict_final_arrays = create_features_dict_final_arrays(
        dict_loaders=dict_loaders,
        dict_creators=dict_creators,
        data_path="data_train",
    )
    features_dict_final_arrays_test = create_features_dict_final_arrays(
        dict_loaders=dict_loaders_without_labels,
        dict_creators=dict_creators,
        data_path="data_test",
    )

    # Shuffle the data
    if do_shuffle:
        print("Shuffling the data...")
        shuffle_data(features_dict_final_arrays)

    # Limit the data to a subset for debugging
    cut_data_to_n_data_max(features_dict_final_arrays, n_data_max)

    # Get the y_data out of the features, and remove it from the features.
    assert (
        "labels" in features_dict_final_arrays
    ), "The features 'labels' should be present in the final features."
    labels = features_dict_final_arrays.pop("labels")

    # Start the KFold loop for Cross Validation
    dict_list_metrics = defaultdict(list)
    list_label_preds_test: List[np.ndarray] = []
    for k in range(K):
        print(f"\nStarting fold {k+1}/{config['cross_val_folds']}")

        (
            features_dict_final_arrays_train,
            features_dict_final_arrays_val,
            labels_train,
            labels_val,
        ) = get_train_val_split(features_dict_final_arrays, labels, k, K)

        # Train the model
        with RuntimeMeter("training") as rm:
            trainer.train(
                features_dict_final_arrays=features_dict_final_arrays_train,
                labels=labels_train,
            )

        # Evaluate the model
        with RuntimeMeter("evaluation") as rm:
            metric_results = {}
            if K >= 2:
                # Cross validation
                labels_pred = trainer.predict(features_dict_final_arrays_val)
                accuracy = accuracy_score(labels_val, labels_pred)
                metric_results["accuracy"] = accuracy
                print(f"Accuracy: {accuracy}")
            # Test prediction
            labels_pred_test = trainer.predict(features_dict_final_arrays_test)
            list_label_preds_test.append(labels_pred_test)
            # Log metrics
            metric_results["loading_time"] = rm.get_stage_runtime("loading")
            metric_results["creation_time"] = rm.get_stage_runtime("creation")
            metric_results["training_time"] = rm.get_stage_runtime("training")
            metric_results["evaluation_time"] = rm.get_stage_runtime("evaluation")
            metric_results["log_time"] = rm.get_stage_runtime("log")

        # Log metrics
        with RuntimeMeter("log") as rm:
            for metric_name, metric_result in metric_results.items():
                dict_list_metrics[metric_name].append(metric_result)
            if do_wandb:
                cumulative_solver_time_in_ms = int(
                    rm.get_stage_runtime("solver") * 1000
                )
                wandb.log(metric_results, step=cumulative_solver_time_in_ms)
            if do_tb:
                for metric_name, metric_result in metric_results.items():
                    tb_writer.add_scalar(
                        f"metrics/{metric_name}",
                        metric_result,
                        global_step=rm.get_stage_runtime("solver"),
                    )
            if do_cli:
                print(f"Metrics: {metric_results}")

    # Finish the WandB run.
    if do_wandb:
        run.finish()

    # Conclude
    print()
    for metric_name, list_metric in dict_list_metrics.items():
        print(f"Metric {metric_name}: {np.mean(list_metric)} +- {np.std(list_metric)}")

    # Save the predictions
    if do_test_pred:
        print("\nSaving the predictions...")
        save_predictions(list_label_preds_test)


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()
    pr.dump_stats("logs/profile_stats.prof")
    print("\nProfile stats dumped to profile_stats.prof")
    print(
        "You can visualize the profile stats using snakeviz by running 'snakeviz logs/profile_stats.prof'"
    )
