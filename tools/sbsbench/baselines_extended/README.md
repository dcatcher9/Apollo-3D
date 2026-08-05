# Extended-suite baselines

The retired Host SBS V1 snapshots were removed. Run the V2 extended suite with
`--suite extended --comparison-only` until a complete result has been reviewed. Publish a new
baseline set only with `--suite extended --update-baselines`; the baseline-gated command fails
closed while any required snapshot is absent.
