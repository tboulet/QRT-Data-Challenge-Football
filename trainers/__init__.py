from typing import Type, Dict

from trainers.base_trainer import BaseTrainer
from trainers.majority import MajorityClassTrainer
from trainers.random import RandomTrainer
from trainers.xgboost import XGBoostTrainer

trainer_name_to_TrainerClass : Dict[str, Type[BaseTrainer]] = {
    "Random" : RandomTrainer,
    "Majority Class": MajorityClassTrainer,
    "XGBoost": XGBoostTrainer,
}