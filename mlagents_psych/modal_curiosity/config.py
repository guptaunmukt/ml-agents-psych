from __future__ import annotations

from typing import List, Optional, Tuple

import attr

from mlagents.trainers.exception import TrainerConfigError
from mlagents.trainers.ppo.optimizer_torch import PPOSettings

VISUAL_SENSOR_CANDIDATES: Tuple[str, ...] = ("RatVisualField", "CameraSensor")
AUDITORY_SENSOR_CANDIDATES: Tuple[str, ...] = (
    "RatAuditoryField",
    "Auditory Spectrogram",
)
BRANCH_ORDER: Tuple[str, ...] = ("visual", "auditory", "remaining")

BRANCH_SENSOR_CANDIDATES = {
    "visual": VISUAL_SENSOR_CANDIDATES,
    "auditory": AUDITORY_SENSOR_CANDIDATES,
}


@attr.s(auto_attribs=True)
class ModalCuriosityBranchSettings:
    strength: float = 1.0

    def __attrs_post_init__(self) -> None:
        if self.strength <= 0.0:
            raise TrainerConfigError(
                "modal_curiosity branch strength must be greater than 0."
            )


@attr.s(auto_attribs=True)
class ModalCuriositySettings:
    visual: Optional[ModalCuriosityBranchSettings] = None
    auditory: Optional[ModalCuriosityBranchSettings] = None
    remaining: Optional[ModalCuriosityBranchSettings] = None

    def active_items(self) -> List[Tuple[str, ModalCuriosityBranchSettings]]:
        active: List[Tuple[str, ModalCuriosityBranchSettings]] = []
        for branch_name in BRANCH_ORDER:
            branch_settings = getattr(self, branch_name)
            if branch_settings is not None:
                active.append((branch_name, branch_settings))
        return active


@attr.s(auto_attribs=True)
class PPOModalCuriositySettings(PPOSettings):
    modal_curiosity: ModalCuriositySettings = attr.ib(
        factory=ModalCuriositySettings
    )
