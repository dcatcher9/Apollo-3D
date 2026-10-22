#ifndef HOST_SBS_SUBTITLE_HISTORY_COMPATIBILITY_HLSL
#define HOST_SBS_SUBTITLE_HISTORY_COMPATIBILITY_HLSL

// Source identity is shared by the probability witness and native RGB advancement.
// Callers separately authenticate their current publication and resource dimensions.
bool V2SubtitleHistoryClockLess(uint2 a, uint2 b) {
    return a.y < b.y || (a.y == b.y && a.x < b.x);
}

bool V2SubtitleHistoryCompatible(
    StructuredBuffer<uint> identity,
    uint4 source, uint4 field, uint4 content, uint4 frame, uint2 observation,
    uint domain_reset, uint epoch, out bool duplicate) {
    duplicate = false;
    uint words, stride;
    identity.GetDimensions(words, stride);
    if (domain_reset != 0u || words != V2_SUBTITLE_CONTOUR_HISTORY_WORD_COUNT || stride != 4u ||
        !any(frame.xy != 0u) || !any(observation != 0u) ||
        identity[0u] != V2_SUBTITLE_CONTOUR_SCHEMA ||
        identity[1u] != V2_SUBTITLE_CONTOUR_HISTORY_TAG || identity[2u] > 1u ||
        identity[V2_SUBTITLE_CONTOUR_HISTORY_SNAPSHOT_VALID_WORD] != 1u ||
        identity[3u] != epoch || identity[6u] != frame.z || identity[7u] != frame.w ||
        identity[22u] != V2_SUBTITLE_CONTOUR_RECIPE) return false;
    [unroll]
    for (uint component = 0u; component < 4u; ++component) {
        if (identity[8u + component] != source[component] ||
            identity[12u + component] != field[component] ||
            identity[16u + component] != content[component]) return false;
    }
    [unroll]
    for (uint word = V2_SUBTITLE_CONTOUR_HISTORY_RESERVED_OFFSET;
         word < V2_SUBTITLE_CONTOUR_HISTORY_WORD_COUNT; ++word) {
        if (identity[word] != 0u) return false;
    }
    uint2 previous_frame = uint2(identity[4u], identity[5u]);
    uint2 previous_time = uint2(identity[20u], identity[21u]);
    if (!any(previous_frame != 0u) || !any(previous_time != 0u)) return false;
    duplicate = all(previous_frame == frame.xy) && all(previous_time == observation);
    return duplicate || (V2SubtitleHistoryClockLess(previous_frame, frame.xy) &&
                         V2SubtitleHistoryClockLess(previous_time, observation));
}

#endif
