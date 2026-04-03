# ML-Agents Psych

Standalone package seed for the PDTW psychometric stats-writer plugin.

This package is the extracted source-of-truth lane for the current custom
ML-Agents logging plugin. It is intentionally seeded from the existing local
plugin in its current form before compatibility refactors begin.

Current status:
- extracted from the local `ml-agents-local` tree
- behavior intentionally unchanged from the seeded source
- not yet upgraded to the newer documented `StatsWriter`-first shape

Planned next steps:
1. keep this initial extracted form as the seed version
2. port it to the documented `StatsWriter` interface
3. add tests for resume behavior, malformed-key safety, and bounded logging

Seed source:
- [Docker/v0.65/ml-agents-local/ml-agents-psych](/Users/unmukt/Documents/KepecsLab/PDTW/Unity/AuditoryNosePoke_v0.1/Docker/v0.65/ml-agents-local/ml-agents-psych)
