import dataclasses
import importlib
import sys
import types
from collections import namedtuple
from pathlib import Path


def install_stub_modules():
    exception_module = types.ModuleType("mlagents.trainers.exception")

    class TrainerConfigError(Exception):
        pass

    exception_module.TrainerConfigError = TrainerConfigError

    ppo_optimizer_module = types.ModuleType("mlagents.trainers.ppo.optimizer_torch")

    class PPOSettings:
        pass

    ppo_optimizer_module.PPOSettings = PPOSettings

    attr_module = types.ModuleType("attr")

    def s(*, auto_attribs=True):
        def decorator(cls):
            attrs_post_init = getattr(cls, "__attrs_post_init__", None)
            if attrs_post_init is not None and "__post_init__" not in cls.__dict__:
                cls.__post_init__ = attrs_post_init
            return dataclasses.dataclass(cls)

        return decorator

    def ib(*, default=dataclasses.MISSING, factory=dataclasses.MISSING):
        kwargs = {}
        if default is not dataclasses.MISSING:
            kwargs["default"] = default
        if factory is not dataclasses.MISSING:
            kwargs["default_factory"] = factory
        return dataclasses.field(**kwargs)

    attr_module.s = s
    attr_module.ib = ib

    sys.modules["attr"] = attr_module
    sys.modules["mlagents"] = types.ModuleType("mlagents")
    sys.modules["mlagents.trainers"] = types.ModuleType("mlagents.trainers")
    sys.modules["mlagents.trainers.exception"] = exception_module
    sys.modules["mlagents.trainers.ppo"] = types.ModuleType("mlagents.trainers.ppo")
    sys.modules["mlagents.trainers.ppo.optimizer_torch"] = ppo_optimizer_module

    return TrainerConfigError


def load_modules():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    for module_name in [
        "mlagents_psych.modal_curiosity.selection",
        "mlagents_psych.modal_curiosity.config",
        "mlagents_psych.modal_curiosity",
    ]:
        sys.modules.pop(module_name, None)
    config = importlib.import_module("mlagents_psych.modal_curiosity.config")
    selection = importlib.import_module("mlagents_psych.modal_curiosity.selection")
    return config, selection


def test_visual_and_auditory_strengths_normalize_with_legacy_aliases():
    install_stub_modules()
    config, selection = load_modules()
    ObservationSpec = namedtuple("ObservationSpec", ["name"])

    settings = config.ModalCuriositySettings(
        visual=config.ModalCuriosityBranchSettings(strength=1.0),
        auditory=config.ModalCuriosityBranchSettings(strength=0.5),
    )
    resolved = selection.resolve_modal_curiosity_branches(
        [ObservationSpec("RatVisualField"), ObservationSpec("Auditory Spectrogram")],
        settings,
    )

    assert [branch.name for branch in resolved] == ["visual", "auditory"]
    assert resolved[0].weight == 2.0 / 3.0
    assert resolved[1].weight == 1.0 / 3.0
    assert resolved[1].observation_names == ("Auditory Spectrogram",)


def test_remaining_claims_only_unclaimed_sensors():
    install_stub_modules()
    config, selection = load_modules()
    ObservationSpec = namedtuple("ObservationSpec", ["name"])

    settings = config.ModalCuriositySettings(
        visual=config.ModalCuriosityBranchSettings(strength=1.0),
        remaining=config.ModalCuriosityBranchSettings(strength=1.0),
    )
    resolved = selection.resolve_modal_curiosity_branches(
        [
            ObservationSpec("RatVisualField"),
            ObservationSpec("TaskVector"),
            ObservationSpec("AuxiliarySensor"),
        ],
        settings,
    )

    assert [branch.name for branch in resolved] == ["visual", "remaining"]
    assert resolved[0].observation_indices == (0,)
    assert resolved[1].observation_indices == (1, 2)
    assert resolved[0].weight == 0.5
    assert resolved[1].weight == 0.5


def test_missing_visual_sensor_raises_config_error():
    TrainerConfigError = install_stub_modules()
    config, selection = load_modules()
    ObservationSpec = namedtuple("ObservationSpec", ["name"])

    settings = config.ModalCuriositySettings(
        visual=config.ModalCuriosityBranchSettings(strength=1.0)
    )

    try:
        selection.resolve_modal_curiosity_branches(
            [ObservationSpec("UnrelatedSensor")],
            settings,
        )
    except TrainerConfigError as exc:
        assert "modal_curiosity.visual" in str(exc)
    else:
        raise AssertionError("Expected TrainerConfigError for missing visual sensor")


def test_stacking_sensor_alias_resolves_to_visual_branch():
    install_stub_modules()
    config, selection = load_modules()
    ObservationSpec = namedtuple("ObservationSpec", ["name"])

    settings = config.ModalCuriositySettings(
        visual=config.ModalCuriosityBranchSettings(strength=1.0),
        auditory=config.ModalCuriosityBranchSettings(strength=0.5),
    )
    resolved = selection.resolve_modal_curiosity_branches(
        [
            ObservationSpec("Auditory Spectrogram"),
            ObservationSpec("StackingSensor_size4_RatVisualField"),
        ],
        settings,
    )

    assert [branch.name for branch in resolved] == ["visual", "auditory"]
    assert resolved[0].observation_names == ("StackingSensor_size4_RatVisualField",)
    assert resolved[1].observation_names == ("Auditory Spectrogram",)
