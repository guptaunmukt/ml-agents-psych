from __future__ import annotations

from typing import cast

import cattr
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.settings import strict_to_cls

from .config import (
    ModalCuriosityBranchSettings,
    ModalCuriositySettings,
    PPOModalCuriositySettings,
)
from .optimizer import TorchPPOModalCuriosityOptimizer

TRAINER_NAME = "ppo_modal_curiosity"


def _register_modal_curiosity_structure_hooks() -> None:
    cattr.register_structure_hook(ModalCuriosityBranchSettings, strict_to_cls)
    cattr.register_structure_hook(ModalCuriositySettings, strict_to_cls)


class PPOModalCuriosityTrainer(PPOTrainer):
    def create_optimizer(self) -> TorchOptimizer:
        return TorchPPOModalCuriosityOptimizer(
            cast(TorchPolicy, self.policy), self.trainer_settings
        )

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME


def get_type_and_setting():
    _register_modal_curiosity_structure_hooks()
    return {TRAINER_NAME: PPOModalCuriosityTrainer}, {
        TRAINER_NAME: PPOModalCuriositySettings
    }
