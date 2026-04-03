# ML-Agents Psych

Standalone package for the PDTW psychometric stats-writer plugin.

This repository is the extracted source-of-truth lane for the custom
ML-Agents logging plugin used by the project.

## Status
- extracted from the legacy local trainer tree
- now refactored to target the documented `StatsWriter` interface directly
- includes lightweight smoke tests for key parsing and step-offset behavior
- not yet validated against the final pinned upgraded trainer fork

## Near-Term Plan
1. validate against the pinned ML-Agents `1.1.0` trainer target
2. harden resume behavior and malformed-key handling further if needed
3. bound logging overhead under real training workloads
4. publish the repository remotely once the compatibility pass is stable

## Notes
- local machine-specific paths should stay out of tracked package metadata and docs
- this package is intended to be installed independently from the trainer source tree
