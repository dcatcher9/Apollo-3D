// Test-only oracle for the retired two-dispatch padded preprocessing topology. Production owns
// exactly one calibrated rgb_to_nchw entry point; this fixture exists only to prove that its
// padded output remains bit-identical to the removed optimization.

#include "../../src_assets/windows/assets/shaders/directx/rgb_to_nchw_cs.hlsl"

[numthreads(16, 16, 1)]
void oracle_content_main(uint3 dispatch_thread : SV_DispatchThreadID) {
    if (!DepthAnalysisContentValid())
        return;

    uint2 content_extent = uint2(
        analysis_content_right - analysis_content_left,
        analysis_content_bottom - analysis_content_top);
    if (dispatch_thread.x >= content_extent.x || dispatch_thread.y >= content_extent.y)
        return;

    uint2 output_position =
        uint2(analysis_content_left, analysis_content_top) + dispatch_thread.xy;
    uint base_index = output_position.y * target_w + output_position.x;

    uint texture_width, texture_height;
    InputTexture.GetDimensions(texture_width, texture_height);
    uint2 texture_size = uint2(texture_width, texture_height);
    if (!DepthSourceRegionValid(texture_size)) {
        uint invalid_stride = target_w * target_h;
        OutputBuffer[base_index] = 0.0f;
        OutputBuffer[base_index + invalid_stride] = 0.0f;
        OutputBuffer[base_index + 2u * invalid_stride] = 0.0f;
        OutputAppearanceOrdinal[base_index] = 0.0f;
        OutputTensorExclusion[output_position] = 1u;
        return;
    }
    uint2 source_size = DepthSourceRegionSize();
    float3 pixel = SampleLetterboxedModelFootprint(output_position, source_size);
    OutputTensorExclusion[output_position] = 0u;

    float2 appearance_model_uv =
        (float2(output_position) + 0.5f) / float2(target_w, target_h);
    float2 appearance_uv = LetterboxSourceUv(appearance_model_uv);
    uint2 source_point = min(
        uint2(appearance_uv * float2(source_size)),
        source_size - 1u);
    float3 capture_rgb = InputTexture.Load(int3(
        DepthSourceRegionLoadPosition(int2(source_point), source_size), 0)).rgb;
    OutputAppearanceOrdinal[base_index] =
        DepthAppearanceOrdinal(capture_rgb, color_mode);

    float r = (pixel.r - 0.485f) / 0.229f;
    float g = (pixel.g - 0.456f) / 0.224f;
    float b = (pixel.b - 0.406f) / 0.225f;
    uint channel_stride = target_w * target_h;
    OutputBuffer[base_index] = r;
    OutputBuffer[base_index + channel_stride] = g;
    OutputBuffer[base_index + 2u * channel_stride] = b;
}

[numthreads(16, 16, 1)]
void oracle_pad_main(uint3 dispatch_thread : SV_DispatchThreadID) {
    if (dispatch_thread.x >= target_w || dispatch_thread.y >= target_h ||
        !DepthAnalysisContentValid() || DepthAnalysisCellIsContent(dispatch_thread.xy))
        return;

    uint2 source_position = DepthAnalysisClampCell(dispatch_thread.xy);
    uint destination_index = dispatch_thread.y * target_w + dispatch_thread.x;
    uint source_index = source_position.y * target_w + source_position.x;
    uint channel_stride = target_w * target_h;

    OutputBuffer[destination_index] = OutputBuffer[source_index];
    OutputBuffer[destination_index + channel_stride] =
        OutputBuffer[source_index + channel_stride];
    OutputBuffer[destination_index + 2u * channel_stride] =
        OutputBuffer[source_index + 2u * channel_stride];
    OutputAppearanceOrdinal[destination_index] = OutputAppearanceOrdinal[source_index];
    OutputTensorExclusion[dispatch_thread.xy] = 1u;
}
