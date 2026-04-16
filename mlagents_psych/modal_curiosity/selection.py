from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence, Tuple

from mlagents.trainers.exception import TrainerConfigError

from .config import BRANCH_SENSOR_CANDIDATES, ModalCuriositySettings


@dataclass(frozen=True)
class ResolvedModalCuriosityBranch:
    name: str
    raw_strength: float
    weight: float
    observation_indices: Tuple[int, ...]
    observation_names: Tuple[str, ...]


def _sensor_name(spec: object) -> str:
    sensor_name = getattr(spec, "name", None)
    if sensor_name is None or sensor_name == "":
        return "<unnamed>"
    return str(sensor_name)


def _canonical_sensor_name(sensor_name: str) -> str:
    return re.sub(r"^StackingSensor_size\d+_", "", sensor_name)


def _available_sensor_names(observation_specs: Sequence[object]) -> Tuple[str, ...]:
    return tuple(_sensor_name(spec) for spec in observation_specs)


def resolve_modal_curiosity_branches(
    observation_specs: Sequence[object],
    modal_curiosity: ModalCuriositySettings,
) -> Tuple[ResolvedModalCuriosityBranch, ...]:
    active_items = modal_curiosity.active_items()
    if not active_items:
        raise TrainerConfigError(
            "ppo_modal_curiosity requires hyperparameters.modal_curiosity to define at least one branch."
        )

    resolved_unweighted = []
    claimed_indices = set()

    for branch_name, branch_settings in active_items:
        if branch_name == "remaining":
            continue

        candidate_names = BRANCH_SENSOR_CANDIDATES[branch_name]
        matches = tuple(
            (index, _sensor_name(spec))
            for index, spec in enumerate(observation_specs)
            if _canonical_sensor_name(_sensor_name(spec)) in candidate_names
        )
        if not matches:
            available_names = ", ".join(_available_sensor_names(observation_specs))
            raise TrainerConfigError(
                f"modal_curiosity.{branch_name} could not resolve any sensors. "
                f"Expected one of {candidate_names}, available sensors were [{available_names}]."
            )

        indices = tuple(index for index, _ in matches)
        names = tuple(name for _, name in matches)
        for index in indices:
            if index in claimed_indices:
                raise TrainerConfigError(
                    f"Observation index {index} was claimed by more than one modal curiosity branch."
                )
            claimed_indices.add(index)

        resolved_unweighted.append(
            (branch_name, branch_settings.strength, indices, names)
        )

    if modal_curiosity.remaining is not None:
        remaining_indices = tuple(
            index for index in range(len(observation_specs)) if index not in claimed_indices
        )
        if not remaining_indices:
            raise TrainerConfigError(
                "modal_curiosity.remaining was requested, but no unclaimed sensors remain."
            )
        resolved_unweighted.append(
            (
                "remaining",
                modal_curiosity.remaining.strength,
                remaining_indices,
                tuple(_sensor_name(observation_specs[index]) for index in remaining_indices),
            )
        )

    total_strength = sum(item[1] for item in resolved_unweighted)
    if total_strength <= 0.0:
        raise TrainerConfigError(
            "modal_curiosity branch strengths must sum to a positive value."
        )

    return tuple(
        ResolvedModalCuriosityBranch(
            name=branch_name,
            raw_strength=raw_strength,
            weight=raw_strength / total_strength,
            observation_indices=indices,
            observation_names=names,
        )
        for branch_name, raw_strength, indices, names in resolved_unweighted
    )
