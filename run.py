# Logging
from collections import defaultdict
import random
from sklearn.metrics import accuracy_score, mean_squared_error
import wandb
from tensorboardX import SummaryWriter

# Config system
import hydra
from omegaconf import OmegaConf, DictConfig

# Utils
from tqdm import tqdm
import datetime
from time import time
from typing import Any, Dict, Type
import cProfile

# ML libraries
import numpy as np
from features_loaders.base_feature_loader import BaseLoader
from src.utils import try_get_seed

# Project imports
from trainers import trainer_name_to_TrainerClass
from features_loaders import loader_name_to_LoaderClass
from features_creators import feature_creator_name_to_FeatureCreatorClass
from src.time_measure import RuntimeMeter


@hydra.main(config_path="configs", config_name="config_default.yaml")
def main(config: DictConfig):

    # Get the config values from the config object.
    config = OmegaConf.to_container(config, resolve=True)
    do_shuffle: bool = config["do_shuffle"]
    n_data_max: int = config["n_data_max"]
    K: int = config["cross_val_folds"]

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

    # Initialize loggers
    run_name = f"[{name_trainer}]]_{datetime.datetime.now().strftime('%dth%mmo_%Hh%Mmin%Ss')}_seed{seed}"
    print(f"\nStarting run {run_name}")
    if do_wandb:
        run = wandb.init(
            name=run_name,
            config=config,
            **config["wandb_config"],
        )
    if do_tb:
        tb_writer = SummaryWriter(log_dir=f"tensorboard/{run_name}")

    # Load the features both from the disk and create some on the fly using creators.
    feature_dict_intermediary_objects: Dict[str, Any] = {}
    features_dict_final_arrays: Dict[str, np.ndarray] = {}

    # Load the features with the feature loaders
    with RuntimeMeter("loading") as rm:
        print("Loading features...")
        loader_name_to_loader: Dict[str, BaseLoader] = {}
        loaders: Dict[str, dict] = config["loaders"]
        for name_loader, config_loader in loaders.items():
            if config_loader["load"]:
                LoaderClass = loader_name_to_LoaderClass[name_loader]
                loader: BaseLoader = LoaderClass(config_loader)
                feature_dict_intermediary_objects.update(
                    loader.load_features(data_path="data_train")
                )
                loader_name_to_loader[name_loader] = loader

    # Run the feature creators to create final features from the loaded features
    with RuntimeMeter("creation") as rm:
        print("Creating features...")
        creators: Dict[str, dict] = config["creators"]
        for name_creator, config_creator in creators.items():
            if config_creator["use"]:
                creator = feature_creator_name_to_FeatureCreatorClass[name_creator]
                features_dict_final_arrays.update(
                    creator.create_usable_features(feature_dict_intermediary_objects)
                )

    # Get the final features from the feature creators
    with RuntimeMeter("reloadings") as rm:
        print("Reloading features...")
        for name_loader, loader in loader_name_to_loader.items():
            if config["loaders"][name_loader]["use"]:
                features_dict_final_arrays.update(loader.get_usable_features())
        n_data = len(next(iter(features_dict_final_arrays.values())))
        print(f"Number of data: {n_data}")
        print()

    # Shuffle the data
    if do_shuffle:
        print("Shuffling the data...")
        shuffled_indices = np.random.permutation(n_data)
        for name_feature, feature_final_array in features_dict_final_arrays.items():
            features_dict_final_arrays[name_feature] = feature_final_array[
                shuffled_indices
            ]

    # Limit the data to a subset for debugging
    if isinstance(n_data_max, int) and n_data_max < n_data:
        for name_feature, feature_final_array in features_dict_final_arrays.items():
            features_dict_final_arrays[name_feature] = feature_final_array[:n_data_max]
        n_data = min(n_data, n_data_max)
        print(f"Limiting the data to {n_data} samples.")

    # Get the y_data out of the features, and remove it from the features.
    assert (
        "labels" in features_dict_final_arrays
    ), "The features 'labels' should be present in the final features."
    labels = features_dict_final_arrays.pop("labels")

    # Start the KFold loop for Cross Validation
    dict_list_metrics = defaultdict(list)
    for k in range(K):
        print(f"\nStarting fold {k+1}/{config['cross_val_folds']}")

        indices_val = np.arange(n_data)[k * n_data // K : (k + 1) * n_data // K]
        indices_train = np.concatenate(
            [
                np.arange(n_data)[: k * n_data // K],
                np.arange(n_data)[(k + 1) * n_data // K :],
            ]
        )
        features_dict_final_arrays_train = {
            name_feature: feature_final_array[indices_train]
            for name_feature, feature_final_array in features_dict_final_arrays.items()
        }
        features_dict_final_arrays_val = {
            name_feature: feature_final_array[indices_val]
            for name_feature, feature_final_array in features_dict_final_arrays.items()
        }
        labels_train = labels[indices_train]
        labels_val = labels[indices_val]
        # Train the model
        with RuntimeMeter("training") as rm:
            trainer.train(
                features_dict_final_arrays=features_dict_final_arrays_train, labels=labels_train
            )

        # Evaluate the model
        with RuntimeMeter("evaluation") as rm:
            metric_results = {}
            labels_pred = trainer.predict(features_dict_final_arrays_val)
            print("Labels pred:", labels_pred[:10])
            print("Labels true:", labels_val[:10])
            accuracy = accuracy_score(labels_val, labels_pred)
            metric_results["accuracy"] = accuracy
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


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()
    pr.dump_stats("logs/profile_stats.prof")
    print("\nProfile stats dumped to profile_stats.prof")
    print(
        "You can visualize the profile stats using snakeviz by running 'snakeviz logs/profile_stats.prof'"
    )
