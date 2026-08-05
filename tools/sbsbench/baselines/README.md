# Core-suite baselines

The earlier core snapshots were clean Host SBS V2 evidence, but they predated the current
schema-25 geometry and still declared controls removed with the legacy pipeline. They remain in
Git history rather than masquerading as current gate inputs.

Use `run_eval.py --comparison-only` while developing this migration. After the complete result is
reviewed and the exact implementation is committed, `--update-baselines` must publish one coherent
replacement set from that clean commit. Baseline-gated runs intentionally fail closed until every
required clip has a current snapshot.
