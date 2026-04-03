from setuptools import find_packages, setup

ML_AGENTS_STATS_WRITER = "mlagents.stats_writer"

setup(
    name="mlagents_psych",
    version="0.0.2",
    packages=find_packages(exclude=["tests", "tests.*"]),
    zip_safe=False,
    entry_points={
        ML_AGENTS_STATS_WRITER: [
            "psychometric=mlagents_psych.psych_stats_writer:get_psych_stats_writer"
        ]
    },
)
