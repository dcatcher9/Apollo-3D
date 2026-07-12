#ifndef DEPTH_PLANE_CONSTANTS_HLSL
#define DEPTH_PLANE_CONSTANTS_HLSL

cbuffer PlaneConstants : register(b1) {
    uint plane_w;
    uint plane_h;
    float plane_strength;
    float plane_width;
    uint filter_axis;
    uint filter_radius;
    uint filter_op;
    uint plane_group_count;
    uint plane_subject_stretch;
    uint3 plane_reserved;
};

#endif
