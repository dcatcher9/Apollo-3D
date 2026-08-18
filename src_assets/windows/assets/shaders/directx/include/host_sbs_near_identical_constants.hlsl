#ifndef HOST_SBS_NEAR_IDENTICAL_CONSTANTS_HLSL
#define HOST_SBS_NEAR_IDENTICAL_CONSTANTS_HLSL

// The detector owns these twenty words at b1.  The fused depth-coordinate producer reuses the
// exact same upload at b2 because b1 is already the DVC constant buffer there.  Define
// HOST_SBS_NEAR_IDENTICAL_CONSTANTS_REGISTER before including this file to select the latter
// binding without cloning the ABI.
#ifndef HOST_SBS_NEAR_IDENTICAL_CONSTANTS_REGISTER
#define HOST_SBS_NEAR_IDENTICAL_CONSTANTS_REGISTER b1
#endif

cbuffer NearIdenticalConstants : register(HOST_SBS_NEAR_IDENTICAL_CONSTANTS_REGISTER) {
    uint near_identical_request_flags;
    uint near_identical_tile_group_width;
    uint near_identical_tile_group_height;
    uint near_identical_tile_group_count;
    uint near_identical_current_frame_low;
    uint near_identical_current_frame_high;
    uint near_identical_baseline_frame_low;
    uint near_identical_baseline_frame_high;
    uint near_identical_domain_tag_low;
    uint near_identical_domain_tag_high;
    uint near_identical_request_token_low;
    uint near_identical_request_token_high;
    uint near_identical_reduce_groups;
    uint near_identical_stream_frame_delta;
    uint near_identical_expected_work;
    uint near_identical_expected_work_cookie;
    uint near_identical_observation_timestamp_low;
    uint near_identical_observation_timestamp_high;
    uint near_identical_timestamp_padding0;
    uint near_identical_timestamp_padding1;
};

#endif
