#ifndef DEPTH_SCENE_CUT_EVIDENCE_CONTRACT_HLSL
#define DEPTH_SCENE_CUT_EVIDENCE_CONTRACT_HLSL

// Counter order shared by the live cut evidence producer and its one-thread resolver. Keep this
// contract in one file: a same-sized reorder would otherwise compile while silently changing the
// detector's meaning.
#define CUT_EVIDENCE_TOTAL 0u
#define CUT_EVIDENCE_DEPTH_CHANGE 1u
#define CUT_EVIDENCE_STRUCTURAL_CHANGE 2u
#define CUT_EVIDENCE_RAW_RGB_CHANGE 3u
#define CUT_EVIDENCE_CURRENT_STRUCTURAL_SUPPORT 4u
#define CUT_EVIDENCE_PREVIOUS_STRUCTURAL_SUPPORT 5u
#define CUT_EVIDENCE_COMMON_STRUCTURAL_SUPPORT 6u
#define CUT_EVIDENCE_BRIGHTNESS_RISE 7u
#define CUT_EVIDENCE_BRIGHTNESS_FALL 8u
#define CUT_EVIDENCE_WORD_COUNT 9u

#endif
