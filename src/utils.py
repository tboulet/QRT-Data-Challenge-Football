from typing import Dict, Union

import numpy as np


def to_numeric(x: Union[int, float, str, None]) -> Union[int, float]:
    if isinstance(x, int) or isinstance(x, float):
        return x
    elif isinstance(x, str):
        return float(x)
    elif x == "inf":
        return float("inf")
    elif x == "-inf":
        return float("-inf")
    elif x is None:
        return None
    else:
        raise ValueError(f"Cannot convert {x} to numeric")


def try_get_seed(config: Dict) -> int:
    """Will try to extract the seed from the config, or return a random one if not found

    Args:
        config (Dict): the run config

    Returns:
        int: the seed
    """
    try:
        seed = config["seed"]
        if not isinstance(seed, int):
            seed = np.random.randint(0, 1000)
    except KeyError:
        seed = np.random.randint(0, 1000)
    return seed


def get_name_trainer_and_features(
    name_trainer : str, 
    dict_loaders : Dict[str, dict], 
    dict_creators : Dict[str, dict],
) -> str:
    """Create the name of the {trainer + features} combination

    Args:
        name_trainer (str): the name tag of the trainer
        dict_loaders (Dict[str, dict]): the dict of loaders
        dict_creators (Dict[str, dict]): the dict of creators

    Returns:
        str: the name of the combination of trainer and features
    """
    name_combination = name_trainer
    name_combination += "_("
    for name_loader, config_loader in dict_loaders.items():
        if config_loader["use"] :
            name_combination += name_loader + ","
    for name_creator, config_creator in dict_creators.items():
        if config_creator["use"] :
            name_combination += name_creator + ","
    name_combination += ")"
    return name_combination