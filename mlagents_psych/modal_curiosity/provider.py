from __future__ import annotations

import copy
from typing import Dict, List, NamedTuple, Sequence

import numpy as np

from mlagents.torch_utils import default_device, torch
from mlagents_envs import logging_util
from mlagents_envs.base_env import BehaviorSpec, ObservationSpec
from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents.trainers.settings import CuriositySettings
from mlagents.trainers.torch_entities.action_flattener import ActionFlattener
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.components.reward_providers.base_reward_provider import (
    BaseRewardProvider,
)
from mlagents.trainers.torch_entities.layers import LinearEncoder, linear_layer
from mlagents.trainers.torch_entities.networks import NetworkBody
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil

from .config import ModalCuriositySettings
from .selection import ResolvedModalCuriosityBranch, resolve_modal_curiosity_branches

logger = logging_util.get_logger(__name__)


class ActionPredictionTuple(NamedTuple):
    continuous: torch.Tensor
    discrete: torch.Tensor


class BranchEvaluation(NamedTuple):
    rewards: torch.Tensor
    forward_loss: torch.Tensor
    inverse_loss: torch.Tensor


class ModalCuriosityUpdateStats(NamedTuple):
    rewards: torch.Tensor
    forward_loss: torch.Tensor
    inverse_loss: torch.Tensor
    branch_reward_means: Dict[str, float]
    branch_forward_losses: Dict[str, float]
    branch_inverse_losses: Dict[str, float]
    branch_weights: Dict[str, float]


class ModalCuriosityBranchNetwork(torch.nn.Module):
    EPSILON = 1e-10

    def __init__(
        self,
        branch: ResolvedModalCuriosityBranch,
        observation_specs: Sequence[ObservationSpec],
        action_spec,
        network_settings,
    ) -> None:
        super().__init__()
        self.branch = branch
        self._action_spec = action_spec
        self._observation_indices = branch.observation_indices
        self._state_encoder = NetworkBody(list(observation_specs), network_settings)
        self._action_flattener = ActionFlattener(self._action_spec)

        self.inverse_model_action_encoding = torch.nn.Sequential(
            LinearEncoder(2 * network_settings.hidden_units, 1, 256)
        )

        if self._action_spec.continuous_size > 0:
            self.continuous_action_prediction = linear_layer(
                256, self._action_spec.continuous_size
            )
        if self._action_spec.discrete_size > 0:
            self.discrete_action_prediction = linear_layer(
                256, sum(self._action_spec.discrete_branches)
            )

        self.forward_model_next_state_prediction = torch.nn.Sequential(
            LinearEncoder(
                network_settings.hidden_units
                + self._action_flattener.flattened_size,
                1,
                256,
            ),
            linear_layer(256, network_settings.hidden_units),
        )

    def select_tensors(self, tensor_obs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return [tensor_obs[index] for index in self._observation_indices]

    def encode(self, tensor_obs: Sequence[torch.Tensor]) -> torch.Tensor:
        hidden, _ = self._state_encoder.forward(list(tensor_obs))
        return hidden

    def predict_action(
        self, current_state: torch.Tensor, next_state: torch.Tensor
    ) -> ActionPredictionTuple:
        inverse_model_input = torch.cat((current_state, next_state), dim=1)
        hidden = self.inverse_model_action_encoding(inverse_model_input)

        continuous_pred = None
        discrete_pred = None
        if self._action_spec.continuous_size > 0:
            continuous_pred = self.continuous_action_prediction(hidden)
        if self._action_spec.discrete_size > 0:
            raw_discrete_pred = self.discrete_action_prediction(hidden)
            branches = ModelUtils.break_into_branches(
                raw_discrete_pred, self._action_spec.discrete_branches
            )
            branches = [torch.softmax(branch, dim=1) for branch in branches]
            discrete_pred = torch.cat(branches, dim=1)
        return ActionPredictionTuple(continuous_pred, discrete_pred)

    def predict_next_state(
        self, current_state: torch.Tensor, actions: AgentAction
    ) -> torch.Tensor:
        flattened_action = self._action_flattener.forward(actions)
        forward_model_input = torch.cat((current_state, flattened_action), dim=1)
        return self.forward_model_next_state_prediction(forward_model_input)

    def compute_reward(
        self,
        current_state: torch.Tensor,
        next_state: torch.Tensor,
        actions: AgentAction,
    ) -> torch.Tensor:
        predicted_next_state = self.predict_next_state(current_state, actions)
        sq_difference = 0.5 * (next_state - predicted_next_state) ** 2
        return torch.sum(sq_difference, dim=1)

    def compute_forward_loss(
        self, rewards: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        return torch.mean(ModelUtils.dynamic_partition(rewards, masks, 2)[1])

    def compute_inverse_loss(
        self,
        current_state: torch.Tensor,
        next_state: torch.Tensor,
        actions: AgentAction,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        predicted_action = self.predict_action(current_state, next_state)
        inverse_loss = 0
        if self._action_spec.continuous_size > 0:
            sq_difference = (
                actions.continuous_tensor - predicted_action.continuous
            ) ** 2
            sq_difference = torch.sum(sq_difference, dim=1)
            inverse_loss += torch.mean(ModelUtils.dynamic_partition(sq_difference, masks, 2)[1])
        if self._action_spec.discrete_size > 0:
            true_action = torch.cat(
                ModelUtils.actions_to_onehot(
                    actions.discrete_tensor, self._action_spec.discrete_branches
                ),
                dim=1,
            )
            cross_entropy = torch.sum(
                -torch.log(predicted_action.discrete + self.EPSILON) * true_action,
                dim=1,
            )
            inverse_loss += torch.mean(ModelUtils.dynamic_partition(cross_entropy, masks, 2)[1])
        return inverse_loss


class ModalCuriosityNetwork(torch.nn.Module):
    def __init__(
        self,
        specs: BehaviorSpec,
        settings: CuriositySettings,
        modal_curiosity: ModalCuriositySettings,
    ) -> None:
        super().__init__()
        self._n_obs_total = len(specs.observation_specs)
        self._resolved_branches = resolve_modal_curiosity_branches(
            specs.observation_specs,
            modal_curiosity,
        )

        branch_network_settings = copy.deepcopy(settings.network_settings)
        if branch_network_settings.memory is not None:
            branch_network_settings.memory = None
            logger.warning(
                "memory was specified in network_settings but is not supported by Curiosity. It is being ignored."
            )

        self._branches = torch.nn.ModuleDict(
            {
                branch.name: ModalCuriosityBranchNetwork(
                    branch,
                    [specs.observation_specs[index] for index in branch.observation_indices],
                    specs.action_spec,
                    branch_network_settings,
                )
                for branch in self._resolved_branches
            }
        )
        logger.info(
            "Modal curiosity branch resolution: %s",
            "; ".join(
                f"{branch.name}={list(branch.observation_names)} weight={branch.weight:.3f}"
                for branch in self._resolved_branches
            ),
        )

    def _tensor_obs_from_buffer(
        self, mini_batch: AgentBuffer, next_state: bool
    ) -> List[torch.Tensor]:
        np_obs = (
            ObsUtil.from_buffer_next(mini_batch, self._n_obs_total)
            if next_state
            else ObsUtil.from_buffer(mini_batch, self._n_obs_total)
        )
        return [ModelUtils.list_to_tensor(obs) for obs in np_obs]

    def _evaluate_branches(
        self, mini_batch: AgentBuffer
    ) -> Dict[str, BranchEvaluation]:
        current_obs = self._tensor_obs_from_buffer(mini_batch, next_state=False)
        next_obs = self._tensor_obs_from_buffer(mini_batch, next_state=True)
        actions = AgentAction.from_buffer(mini_batch)
        masks = ModelUtils.list_to_tensor(mini_batch[BufferKey.MASKS], dtype=torch.float)

        branch_results: Dict[str, BranchEvaluation] = {}
        for branch in self._resolved_branches:
            branch_network = self._branches[branch.name]
            current_state = branch_network.encode(branch_network.select_tensors(current_obs))
            next_state = branch_network.encode(branch_network.select_tensors(next_obs))
            rewards = branch_network.compute_reward(current_state, next_state, actions)
            branch_results[branch.name] = BranchEvaluation(
                rewards=rewards,
                forward_loss=branch_network.compute_forward_loss(rewards, masks),
                inverse_loss=branch_network.compute_inverse_loss(
                    current_state,
                    next_state,
                    actions,
                    masks,
                ),
            )
        return branch_results

    def compute_reward(self, mini_batch: AgentBuffer) -> torch.Tensor:
        branch_results = self._evaluate_branches(mini_batch)
        combined_rewards = None
        for branch in self._resolved_branches:
            weighted_rewards = branch.weight * branch_results[branch.name].rewards
            combined_rewards = (
                weighted_rewards
                if combined_rewards is None
                else combined_rewards + weighted_rewards
            )
        return combined_rewards

    def compute_update_stats(self, mini_batch: AgentBuffer) -> ModalCuriosityUpdateStats:
        branch_results = self._evaluate_branches(mini_batch)

        combined_rewards = None
        combined_forward_loss = None
        combined_inverse_loss = None
        branch_reward_means: Dict[str, float] = {}
        branch_forward_losses: Dict[str, float] = {}
        branch_inverse_losses: Dict[str, float] = {}
        branch_weights: Dict[str, float] = {}

        for branch in self._resolved_branches:
            branch_result = branch_results[branch.name]
            weighted_rewards = branch.weight * branch_result.rewards
            weighted_forward_loss = branch.weight * branch_result.forward_loss
            weighted_inverse_loss = branch.weight * branch_result.inverse_loss

            combined_rewards = (
                weighted_rewards
                if combined_rewards is None
                else combined_rewards + weighted_rewards
            )
            combined_forward_loss = (
                weighted_forward_loss
                if combined_forward_loss is None
                else combined_forward_loss + weighted_forward_loss
            )
            combined_inverse_loss = (
                weighted_inverse_loss
                if combined_inverse_loss is None
                else combined_inverse_loss + weighted_inverse_loss
            )

            branch_reward_means[branch.name] = float(torch.mean(branch_result.rewards).item())
            branch_forward_losses[branch.name] = float(branch_result.forward_loss.item())
            branch_inverse_losses[branch.name] = float(branch_result.inverse_loss.item())
            branch_weights[branch.name] = branch.weight

        return ModalCuriosityUpdateStats(
            rewards=combined_rewards,
            forward_loss=combined_forward_loss,
            inverse_loss=combined_inverse_loss,
            branch_reward_means=branch_reward_means,
            branch_forward_losses=branch_forward_losses,
            branch_inverse_losses=branch_inverse_losses,
            branch_weights=branch_weights,
        )


class ModalCuriosityRewardProvider(BaseRewardProvider):
    beta = 0.2
    loss_multiplier = 10.0

    def __init__(
        self,
        specs: BehaviorSpec,
        settings: CuriositySettings,
        modal_curiosity: ModalCuriositySettings,
    ) -> None:
        super().__init__(specs, settings)
        self._ignore_done = True
        self._network = ModalCuriosityNetwork(specs, settings, modal_curiosity)
        self._network.to(default_device())
        self.optimizer = torch.optim.Adam(
            self._network.parameters(), lr=settings.learning_rate
        )
        self._has_updated_once = False

    @property
    def name(self) -> str:
        return "curiosity"

    def evaluate(self, mini_batch: AgentBuffer) -> np.ndarray:
        with torch.no_grad():
            rewards = ModelUtils.to_numpy(self._network.compute_reward(mini_batch))
        rewards = np.minimum(rewards, 1.0 / self.strength)
        return rewards * self._has_updated_once

    def update(self, mini_batch: AgentBuffer) -> Dict[str, np.ndarray]:
        self._has_updated_once = True
        update_stats = self._network.compute_update_stats(mini_batch)
        loss = self.loss_multiplier * (
            self.beta * update_stats.forward_loss
            + (1.0 - self.beta) * update_stats.inverse_loss
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        stats: Dict[str, np.ndarray] = {
            "Losses/Curiosity Forward Loss": update_stats.forward_loss.item(),
            "Losses/Curiosity Inverse Loss": update_stats.inverse_loss.item(),
        }
        for branch_name, branch_weight in update_stats.branch_weights.items():
            branch_label = branch_name.capitalize()
            stats[f"Curiosity/Branch Reward/{branch_label}"] = update_stats.branch_reward_means[
                branch_name
            ]
            stats[f"Losses/Curiosity Forward Loss/{branch_label}"] = update_stats.branch_forward_losses[
                branch_name
            ]
            stats[f"Losses/Curiosity Inverse Loss/{branch_label}"] = update_stats.branch_inverse_losses[
                branch_name
            ]
            stats[f"Curiosity/Branch Weight/{branch_label}"] = branch_weight
        return stats

    def get_modules(self):
        return {f"Module:{self.name}": self._network}
