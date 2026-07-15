"""Dependency-free numerical contract for Apollo's artistic-policy controls."""

# These are model-output bounds, not the final comfort envelope.  Keep them in a
# lightweight module so data preparation and provenance checks do not require
# importing PyTorch or the DA-V2 training runtime.
ART_SCALE_DELTA_MAX = 0.50
ARTISTIC_GLOBAL_SIZE = 2
