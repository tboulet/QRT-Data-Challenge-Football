from typing import Type, Dict

from trainers.base_trainer import BaseTrainer
from trainers.majority import MajorityClassTrainer
from trainers.random import RandomTrainer
from trainers.statistical import StatisticalTrainer
from trainers.xgboost import XGBoostTrainer
from trainers.selected import SelectedClassTrainer

trainer_name_to_TrainerClass : Dict[str, Type[BaseTrainer]] = {
    "Random" : RandomTrainer,
    "Majority Class": MajorityClassTrainer,
    "XGBoost": XGBoostTrainer,
    "Selected Class": SelectedClassTrainer,
    "Statistical Trainer" : StatisticalTrainer,
}