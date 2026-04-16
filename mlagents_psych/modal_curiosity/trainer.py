from __future__ import annotations

from typing import cast

from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.ppo.trainer import PPOTrainer

from .config import PPOModalCuriositySettings
from .optimizer import TorchPPOModalCuriosityOptimizer

TRAINER_NAME = "ppo_modal_curiosity"


class PPOModalCuriosityTrainer(PPOTrainer):
    def create_optimizer(self) -> TorchOptimizer:
        return TorchPPOModalCuriosityOptimizer(
            cast(TorchPolicy, self.policy), self.trainer_settings
        )

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME


def get_type_and_setting():
    return {TRAINER_NAME: PPOModalCuriosityTrainer}, {
        TRAINER_NAME: PPOModalCuriositySettings
    }
