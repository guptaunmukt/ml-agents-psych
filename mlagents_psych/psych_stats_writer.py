#   Changelog:
##  Removed task-name dependence to ensure general code

from typing import Dict, List
from mlagents.trainers.settings import RunOptions
from mlagents.trainers.stats import (
    StatsWriter,
    StatsSummary,
    TensorboardWriter,
)

from mlagents_envs.logging_util import get_logger
import time

logger = get_logger(__name__)


class PsychStatsWriter(TensorboardWriter):
    """
    Write task-specific variables as tensorboard summaries
        # Select keys with _task prefix:
            # Must have "Time", "Step" sub-keys
            # Optionally has "Value" sub-key
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Introduce instance-wide "reference" parameters to ensure continuity
        # between resumed runs
        self.step_ref = 0
        self.time_ref = 0.0

    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        self._maybe_create_summary_writer(category)

        # If run is resumed, all step/time info is lost from C# end
        # Session counter is restarted each time env is re-initialized
        if (f"Raw/Value/SessionStart" in values.keys()) and (
            values[f"Raw/Value/SessionStart"].full_dist[0] == 1
        ):
            # Move all steps/timers (in a chunked fashion) w.r.t. reference vars
            self.step_ref = step
            self.time_ref = time.time()

        for key, summary in values.items():
            # Get event name for which time is stamped
            if (key in self.hidden_keys) or (not (f"Raw/Time" in key)):
                continue
            tgroups = key.split("/")
            # Task name
            task = tgroups[0]
            # Event name as str
            event = "/".join(tgroups[tgroups.index("Time") + 1 :])

            for ix, ts in enumerate(summary.full_dist):
                _val = (
                    values[f"{task}/Raw/Value/{event}"].full_dist[ix]
                    if f"{task}/Raw/Value/{event}" in values
                    else 0
                )

                self.summary_writers[category].add_scalar(
                    f"{task}/Psych/{event}",
                    _val,
                    global_step=values[f"{task}/Raw/Step/{event}"].full_dist[ix]
                    + self.step_ref,
                    walltime=ts + self.time_ref,
                )
        self.summary_writers[category].flush()


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
