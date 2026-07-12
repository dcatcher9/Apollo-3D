# Legacy warpsim (removed)

The hand-maintained CPU warp replicas and scene-specific diagnostic programs were permanently
removed after the validated Bestv2 field became the only disparity path. They modeled deleted
divergence/focal-plane/border/depth-floor branches and had drifted from the real D3D pipeline.

Use `python tools/sbsbench/run_eval.py` for all SBS evaluation. It executes the actual estimator
and shaders, checks both eyes over the committed temporal suites, and generates the visual report
and decision sidecar.
