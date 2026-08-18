#ifndef DEPTH_SOURCE_REGION_HLSL
#define DEPTH_SOURCE_REGION_HLSL

// Physical texel rectangle inside the bound retained source texture. Model and ownership
// coordinates remain local to this rectangle; only the final integer Texture2D.Load address is
// translated. A full-source input uses {0, 0, texture_width, texture_height}.
cbuffer DepthSourceRegionConstants : register(b2) {
    uint4 depth_source_region;
};

bool DepthSourceRegionValid(uint2 texture_size) {
    return depth_source_region.x < depth_source_region.z &&
        depth_source_region.y < depth_source_region.w &&
        depth_source_region.z <= texture_size.x &&
        depth_source_region.w <= texture_size.y;
}

uint2 DepthSourceRegionSize() {
    return depth_source_region.zw - depth_source_region.xy;
}

int2 DepthSourceRegionLoadPosition(int2 local_position, uint2 region_size) {
    int2 clamped = clamp(
        local_position,
        int2(0, 0),
        max(int2(region_size), int2(1, 1)) - 1);
    return clamped + int2(depth_source_region.xy);
}

#endif
