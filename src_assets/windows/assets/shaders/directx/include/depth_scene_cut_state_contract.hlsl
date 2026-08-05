#ifndef DEPTH_SCENE_CUT_STATE_CONTRACT_HLSL
#define DEPTH_SCENE_CUT_STATE_CONTRACT_HLSL

// Live cut rearm/refractory calibration. Keep it narrow-scoped so changing cut analysis cannot
// invalidate the published V2 coordinate/preprocess shader identities or the offline evaluator.
#define CUT_ARM_SETTLE_STREAM_FRAMES 8.0f

#endif
