# Extended-suite baselines

This directory is intentionally empty until the Apollo-vs-VD3D warp decision is made. Public
profile comparisons must use `--suite extended --comparison-only`; after choosing the retained
profile, establish its baselines with `--suite extended --update-baselines` and commit the
resulting JSON files here. A normal gated extended run refuses to report PASS while any baseline
is missing.
