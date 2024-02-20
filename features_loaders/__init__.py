from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Type, Callable

import numpy as np
import xgboost as xgb


from features_loaders.base_feature_loader import BaseLoader
from features_loaders.labels import LoaderLabels
from features_loaders.teamfeatures import LoaderTeamfeatures
from features_loaders.playerfeatures import LoaderPlayerfeatures

loader_name_to_LoaderClass: Dict[str, Type[BaseLoader]] = {
    "teamfeatures": LoaderTeamfeatures,
    "playerfeatures": LoaderPlayerfeatures,
    "labels": LoaderLabels,
}
