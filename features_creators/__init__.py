from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Type, Callable

import numpy as np
import xgboost as xgb

from .base_feature_creator import BaseFeatureCreator

feature_creator_name_to_FeatureCreatorClass : Dict[str, Type[BaseFeatureCreator]] = {
    # "teamfeatures" : TeamfeaturesLoader,
    # "playerfeatures" : PlayerfeaturesLoader,
}