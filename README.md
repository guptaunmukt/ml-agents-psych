# ML-Agents Psych

Standalone package for project-specific ML-Agents plugins.

This repository is the extracted source-of-truth lane for custom trainer-side and
logging-side extensions used by the PDTW project.

## Current Plugins
- `psychometric`
  - entry point group: `mlagents.stats_writer`
  - implementation: `mlagents_psych.psych_stats_writer`
  - purpose: write task transition series into TensorBoard with psych-specific step and time alignment
- `ppo_modal_curiosity`
  - entry point group: `mlagents.trainer_type`
  - implementation: `mlagents_psych.modal_curiosity`
  - purpose: keep stock PPO semantics largely intact while replacing fused curiosity with weighted modality-specific curiosity branches

## Architecture Notes
- The modal curiosity implementation is intentionally a trainer shim, not a base `mlagents` fork.
- It keeps one outer `curiosity` reward signal and one outer curiosity return stream.
- Modality-specific branch strengths are normalized internally, so they define mixture ratios rather than total intrinsic scale.
- Total curiosity-vs-extrinsic scaling still lives in upstream `reward_signals.curiosity.strength`.

## Docs
- `docs/PPOModalCuriosityTutorial.ipynb`

## Local Validation
Create an isolated Python 3.10 environment with `uv` and run the lightweight tests:

```bash
uv venv --python python3.10 tests/.venv
uv pip install --python tests/.venv/bin/python -e . pytest
tests/.venv/bin/python -m pytest -q tests/test_psych_stats_writer.py tests/test_modal_curiosity_selection.py
```

## Notes
- local machine-specific absolute paths should stay out of tracked package metadata and docs
- this package is intended to be installed independently from the trainer source tree
