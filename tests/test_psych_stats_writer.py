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


class FakeLogger:
    def __init__(self):
        self.messages = {
            "info": [],
            "debug": [],
            "warning": [],
            "error": [],
        }

    def _record(self, level, msg, *args, **kwargs):
        if args:
            msg = msg % args
        self.messages[level].append(msg)

    def info(self, *args, **kwargs):
        self._record("info", *args, **kwargs)

    def debug(self, *args, **kwargs):
        self._record("debug", *args, **kwargs)

    def warning(self, *args, **kwargs):
        self._record("warning", *args, **kwargs)

    def error(self, *args, **kwargs):
        self._record("error", *args, **kwargs)


FAKE_LOGGER = FakeLogger()


def install_stub_modules():
    global FAKE_LOGGER
    FAKE_LOGGER = FakeLogger()

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
    logging_module.get_logger = lambda name: FAKE_LOGGER

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


def test_writer_warns_and_skips_missing_step_series(tmp_path):
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
    assert any(
        "Skipping psych stat without step series" in msg
        for msg in module.logger.messages["warning"]
    )
    assert summary_writer.flushed is True


def test_writer_emits_zero_fallback_when_value_series_is_missing(tmp_path):
    StatsSummary = install_stub_modules()
    module = load_module()

    writer = module.PsychStatsWriter(str(tmp_path), clear_past_data=False)
    values = {
        "2AFC/Raw/Time/PerceptSignal": StatsSummary([2.0], None),
        "2AFC/Raw/Step/PerceptSignal": StatsSummary([5.0], None),
    }

    writer.write_stats("2DPoke", values, step=100)
    summary_writer = writer.summary_writers["2DPoke"]

    assert (
        "2AFC/Psych/PerceptSignal",
        0.0,
        5,
        2.0,
    ) in summary_writer.scalars
    assert module.logger.messages["warning"] == []


def test_writer_uses_last_session_start_when_multiple_markers_exist(tmp_path):
    StatsSummary = install_stub_modules()
    module = load_module()
    module.time.time = lambda: 5000.0

    writer = module.PsychStatsWriter(str(tmp_path), clear_past_data=False)
    values = {
        "2AFC/Raw/Value/SessionStart": StatsSummary([1.0, 2.0], None),
        "2AFC/Raw/Step/SessionStart": StatsSummary([10.0, 20.0], None),
        "2AFC/Raw/Time/SessionStart": StatsSummary([100.0, 200.0], None),
        "2AFC/Raw/Time/Enter/TrialStart": StatsSummary([201.0], None),
        "2AFC/Raw/Step/Enter/TrialStart": StatsSummary([21.0], None),
        "2AFC/Raw/Value/Enter/TrialStart": StatsSummary([1.0], None),
    }

    writer.write_stats("2DPoke", values, step=1000)
    summary_writer = writer.summary_writers["2DPoke"]

    assert (
        "2AFC/Psych/Enter/TrialStart",
        1.0,
        1001,
        5001.0,
    ) in summary_writer.scalars
    assert any(
        "Multiple SessionStart markers for 2AFC in one summary write" in msg
        for msg in module.logger.messages["warning"]
    )


def test_writer_offsets_from_session_start_even_when_session_value_is_not_one(tmp_path):
    StatsSummary = install_stub_modules()
    module = load_module()
    module.time.time = lambda: 8000.0

    writer = module.PsychStatsWriter(str(tmp_path), clear_past_data=False)
    values = {
        "2AFC/Raw/Value/SessionStart": StatsSummary([3.0], None),
        "2AFC/Raw/Step/SessionStart": StatsSummary([40.0], None),
        "2AFC/Raw/Time/SessionStart": StatsSummary([400.0], None),
        "2AFC/Raw/Time/TrialStart": StatsSummary([401.0, 402.0], None),
        "2AFC/Raw/Step/TrialStart": StatsSummary([41.0, 42.0], None),
        "2AFC/Raw/Value/TrialStart": StatsSummary([1.0, 2.0], None),
        "2AFC/Raw/Time/TrialEnd": StatsSummary([410.0], None),
        "2AFC/Raw/Step/TrialEnd": StatsSummary([50.0], None),
        "2AFC/Raw/Value/TrialEnd": StatsSummary([2.0], None),
    }

    writer.write_stats("2DPoke", values, step=1000)
    summary_writer = writer.summary_writers["2DPoke"]

    assert (
        "2AFC/Psych/TrialStart",
        1.0,
        1001,
        8001.0,
    ) in summary_writer.scalars
    assert (
        "2AFC/Psych/TrialStart",
        2.0,
        1002,
        8002.0,
    ) in summary_writer.scalars
    assert (
        "2AFC/Psych/TrialEnd",
        2.0,
        1010,
        8010.0,
    ) in summary_writer.scalars
