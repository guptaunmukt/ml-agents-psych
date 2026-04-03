from setuptools import setup
from mlagents.plugins import ML_AGENTS_STATS_WRITER

setup(
    name="mlagents_psych",
    version="0.0.1",
    py_modules=["mlagents_psych.psych_stats_writer"],
    entry_points={
        ML_AGENTS_STATS_WRITER: [
            "psychometric=mlagents_psych.psych_stats_writer:get_psych_stats_writer"
        ]
    },
)
