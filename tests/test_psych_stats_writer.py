import importlib
import sys
import types
from collections import namedtuple
from pathlib import Path


class FakeSummaryWriter:
    def __init__(self, path):
        self.path = path
        self.scalars = []
        self.flushed = False

    def add_scalar(self, tag, value, global_step=None, walltime=None):
        self.scalars.append((tag, value, global_step, walltime))

    def flush(self):
        self.flushed = True


def install_stub_modules():
    stats_module = types.ModuleType("mlagents.trainers.stats")
    StatsSummary = namedtuple("StatsSummary", ["full_dist", "aggregation_method"])

    class StatsWriter:
        pass

    stats_module.StatsSummary = StatsSummary
    stats_module.StatsWriter = StatsWriter

    settings_module = types.ModuleType("mlagents.trainers.settings")

    class RunOptions:
        pass

    settings_module.RunOptions = RunOptions

    logging_module = types.ModuleType("mlagents_envs.logging_util")

    class Logger:
        def info(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    logging_module.get_logger = lambda name: Logger()

    tensorboard_module = types.ModuleType("torch.utils.tensorboard")
    tensorboard_module.SummaryWriter = FakeSummaryWriter

    sys.modules["mlagents"] = types.ModuleType("mlagents")
    sys.modules["mlagents.trainers"] = types.ModuleType("mlagents.trainers")
    sys.modules["mlagents.trainers.stats"] = stats_module
    sys.modules["mlagents.trainers.settings"] = settings_module
    sys.modules["mlagents_envs"] = types.ModuleType("mlagents_envs")
    sys.modules["mlagents_envs.logging_util"] = logging_module
    sys.modules["torch"] = types.ModuleType("torch")
    sys.modules["torch.utils"] = types.ModuleType("torch.utils")
    sys.modules["torch.utils.tensorboard"] = tensorboard_module

    return StatsSummary


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.modules.pop("mlagents_psych.psych_stats_writer", None)
    return importlib.import_module("mlagents_psych.psych_stats_writer")


def test_split_time_key_extracts_task_and_event():
    install_stub_modules()
    module = load_module()
    assert module._split_time_key("2AFC/Raw/Time/Enter/TrialStart") == (
        "2AFC",
        "Enter/TrialStart",
    )
    assert module._split_time_key("Environment/Cumulative Reward") is None


def test_writer_uses_prefixed_session_start_to_offset_steps_and_walltime(tmp_path):
    StatsSummary = install_stub_modules()
    module = load_module()
    module.time.time = lambda: 1000.0

    writer = module.PsychStatsWriter(str(tmp_path), clear_past_data=False)
    values = {
        "2AFC/Raw/Value/SessionStart": StatsSummary([1.0], None),
        "2AFC/Raw/Step/SessionStart": StatsSummary([0.0], None),
        "2AFC/Raw/Time/SessionStart": StatsSummary([0.0], None),
        "2AFC/Raw/Time/Enter/TrialStart": StatsSummary([2.0], None),
        "2AFC/Raw/Step/Enter/TrialStart": StatsSummary([5.0], None),
        "2AFC/Raw/Value/Enter/TrialStart": StatsSummary([1.0], None),
    }

    writer.write_stats("2DPoke", values, step=100)
    summary_writer = writer.summary_writers["2DPoke"]

    assert (
        "2AFC/Psych/Enter/TrialStart",
        1.0,
        105,
        1002.0,
    ) in summary_writer.scalars
    assert summary_writer.flushed is True


def test_writer_skips_missing_step_series_without_crashing(tmp_path):
    StatsSummary = install_stub_modules()
    module = load_module()

    writer = module.PsychStatsWriter(str(tmp_path), clear_past_data=False)
    values = {
        "2AFC/Raw/Time/Enter/TrialStart": StatsSummary([2.0], None),
        "2AFC/Raw/Value/Enter/TrialStart": StatsSummary([1.0], None),
    }

    writer.write_stats("2DPoke", values, step=100)
    summary_writer = writer.summary_writers["2DPoke"]
    assert summary_writer.scalars == []
    assert summary_writer.flushed is True
