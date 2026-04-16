from setuptools import find_packages, setup

ML_AGENTS_STATS_WRITER = "mlagents.stats_writer"
ML_AGENTS_TRAINER_TYPE = "mlagents.trainer_type"

setup(
    name="mlagents_psych",
    version="0.0.4",
    packages=find_packages(exclude=["tests", "tests.*"]),
    zip_safe=False,
    entry_points={
        ML_AGENTS_STATS_WRITER: [
            "psychometric=mlagents_psych.psych_stats_writer:get_psych_stats_writer"
        ],
        ML_AGENTS_TRAINER_TYPE: [
            "ppo_modal_curiosity=mlagents_psych.modal_curiosity.trainer:get_type_and_setting"
        ],
    },
)
