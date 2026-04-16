import pytest


mlagents = pytest.importorskip("mlagents")
from mlagents.plugins.trainer_type import register_trainer_plugins
from mlagents.trainers.settings import RunOptions


@pytest.fixture(autouse=True)
def register_modal_trainer_plugin():
    register_trainer_plugins()


def test_modal_curiosity_runoptions_parse_registers_nested_settings_hooks():
    options = RunOptions.from_dict(
        {
            "behaviors": {
                "2DPoke": {
                    "trainer_type": "ppo_modal_curiosity",
                    "hyperparameters": {
                        "batch_size": 64,
                        "buffer_size": 2048,
                        "learning_rate": 3.0e-4,
                        "beta": 1.0e-2,
                        "epsilon": 0.2,
                        "lambd": 0.95,
                        "num_epoch": 3,
                        "modal_curiosity": {
                            "visual": {"strength": 1.0},
                            "auditory": {"strength": 0.5},
                        },
                    },
                    "network_settings": {
                        "normalize": True,
                        "hidden_units": 128,
                        "num_layers": 2,
                    },
                    "reward_signals": {
                        "extrinsic": {"gamma": 0.995, "strength": 1.0},
                        "curiosity": {"gamma": 0.99, "strength": 0.1},
                    },
                    "max_steps": 1000,
                    "time_horizon": 64,
                    "summary_freq": 100,
                }
            }
        }
    )

    trainer_settings = options.behaviors["2DPoke"]
    assert trainer_settings.trainer_type == "ppo_modal_curiosity"
    assert trainer_settings.hyperparameters.modal_curiosity.visual.strength == 1.0
    assert trainer_settings.hyperparameters.modal_curiosity.auditory.strength == 0.5
