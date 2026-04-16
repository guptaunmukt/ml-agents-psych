from __future__ import annotations

from typing import Dict, cast

from mlagents.trainers.ppo.optimizer_torch import TorchPPOOptimizer
from mlagents.trainers.settings import RewardSignalSettings, RewardSignalType
from mlagents.trainers.torch_entities.components.reward_providers import (
    create_reward_provider,
)

from .config import PPOModalCuriositySettings
from .provider import ModalCuriosityRewardProvider


class TorchPPOModalCuriosityOptimizer(TorchPPOOptimizer):
    def create_reward_signals(
        self, reward_signal_configs: Dict[RewardSignalType, RewardSignalSettings]
    ) -> None:
        hyperparameters = cast(
            PPOModalCuriositySettings, self.trainer_settings.hyperparameters
        )
        for reward_signal, settings in reward_signal_configs.items():
            if reward_signal == RewardSignalType.CURIOSITY:
                self.reward_signals[reward_signal.value] = ModalCuriosityRewardProvider(
                    self.policy.behavior_spec,
                    settings,
                    hyperparameters.modal_curiosity,
                )
            else:
                self.reward_signals[reward_signal.value] = create_reward_provider(
                    reward_signal,
                    self.policy.behavior_spec,
                    settings,
                )
