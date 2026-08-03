// Live Host SBS renderer for the authenticated Depth Coordinate V2 final field.
//
// Keep this as a separate compile unit so enabling the experimental renderer cannot silently
// change SBS_DIRECT_PARALLAX replay or the legacy production shader. The included implementation
// consumes t1 as the signed, one-sided slope-limited R32 source-U displacement and validates the
// V2 state bound at t2. The sub-unit slope makes the inverse unique; no owner/fill backend exists.
#define SBS_LIVE_V2_SIGNED_PARALLAX 1
#include "sbs_reprojection_ps.hlsl"
