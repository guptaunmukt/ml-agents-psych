from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from mlagents.trainers.settings import RunOptions
from mlagents.trainers.stats import StatsSummary, StatsWriter
from mlagents_envs.logging_util import get_logger
from torch.utils.tensorboard import SummaryWriter

logger = get_logger(__name__)

_TIME_MARKER = "/Raw/Time/"
_STEP_MARKER = "/Raw/Step/"
_VALUE_MARKER = "/Raw/Value/"
_SESSION_START = "SessionStart"


def _split_time_key(key: str) -> Optional[Tuple[str, str]]:
    if _TIME_MARKER not in key:
        return None
    task, _, event = key.partition(_TIME_MARKER)
    if not task or not event:
        return None
    return task, event


class PsychStatsWriter(StatsWriter):
    """
    Write task-specific transition values as TensorBoard scalars.

    This writer intentionally targets the documented StatsWriter interface
    instead of subclassing TensorboardWriter directly.
    """

    def __init__(
        self,
        base_dir: str,
        clear_past_data: bool = False,
        hidden_keys: Optional[List[str]] = None,
    ) -> None:
        self.base_dir = base_dir
        self._clear_past_data = clear_past_data
        self.hidden_keys = set(hidden_keys or [])
        self.summary_writers: Dict[str, SummaryWriter] = {}
        self.step_offsets: Dict[str, int] = {}

    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        self._maybe_create_summary_writer(category)
        self._update_step_offsets(values, step)

        writer = self.summary_writers[category]

        for key, summary in values.items():
            if key in self.hidden_keys:
                continue

            split = _split_time_key(key)
            if split is None:
                continue

            task, event = split
            step_key = f"{task}{_STEP_MARKER}{event}"
            value_key = f"{task}{_VALUE_MARKER}{event}"

            step_summary = values.get(step_key)
            if step_summary is None or not step_summary.full_dist:
                logger.debug("Skipping psych stat without step series: %s", key)
                continue

            value_summary = values.get(value_key)
            time_count = len(summary.full_dist)
            step_count = len(step_summary.full_dist)
            value_count = len(value_summary.full_dist) if value_summary else time_count
            count = min(time_count, step_count, value_count)

            if count == 0:
                continue

            if count != time_count or count != step_count or (value_summary and count != value_count):
                logger.warning(
                    "Psych stat series length mismatch for %s; using truncated count %s.",
                    key,
                    count,
                )

            step_offset = self.step_offsets.get(task, 0)

            for ix in range(count):
                scalar_value = value_summary.full_dist[ix] if value_summary else 0.0
                global_step = int(step_summary.full_dist[ix]) + step_offset
                writer.add_scalar(f"{task}/Psych/{event}", scalar_value, global_step)

        writer.flush()

    def _update_step_offsets(
        self, values: Dict[str, StatsSummary], step: int
    ) -> None:
        for key, summary in values.items():
            if not key.endswith(f"{_VALUE_MARKER}{_SESSION_START}"):
                continue
            if not summary.full_dist or summary.full_dist[0] != 1:
                continue

            task = key[: -len(f"{_VALUE_MARKER}{_SESSION_START}")]
            session_step_key = f"{task}{_STEP_MARKER}{_SESSION_START}"
            session_step_summary = values.get(session_step_key)
            session_step = 0
            if session_step_summary and session_step_summary.full_dist:
                session_step = int(session_step_summary.full_dist[0])

            self.step_offsets[task] = int(step) - session_step

    def _maybe_create_summary_writer(self, category: str) -> None:
        if category in self.summary_writers:
            return

        filewriter_dir = os.path.join(self.base_dir, category)
        os.makedirs(filewriter_dir, exist_ok=True)
        if self._clear_past_data:
            self._delete_all_events_files(filewriter_dir)
        self.summary_writers[category] = SummaryWriter(filewriter_dir)

    def _delete_all_events_files(self, directory_name: str) -> None:
        for file_name in os.listdir(directory_name):
            if not file_name.startswith("events.out"):
                continue
            logger.warning(
                "Deleting TensorBoard data %s left over from a previous run.",
                file_name,
            )
            full_path = os.path.join(directory_name, file_name)
            try:
                os.remove(full_path)
            except OSError:
                logger.error("Unable to delete stale TensorBoard data %s.", full_path)

    def add_property(self, category: str, property_type: Any, value: Any) -> None:
        # This plugin currently does not emit additional property summaries.
        return None



def get_psych_stats_writer(run_options: RunOptions) -> List[StatsWriter]:
    """
    Registration function: must return a list of StatsWriters.
    """

    logger.info("Attached Psych-Stats Writer")

    checkpoint_settings = run_options.checkpoint_settings
    return [
        PsychStatsWriter(
            checkpoint_settings.write_path,
            clear_past_data=not checkpoint_settings.resume,
            hidden_keys=["Is Training", "Step"],
        )
    ]
