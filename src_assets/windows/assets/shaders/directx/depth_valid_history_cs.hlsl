// Preserve the exact preprocessed NCHW color only for frames whose TensorRT output contained a
// valid depth sample. This keeps cut detection paired to the last valid depth instead of allowing
// an all-NaN frame to advance color history independently.
StructuredBuffer<float4> MinMaxEma : register(t0);  // w = current-frame validity
StructuredBuffer<float> CurrentModelInput : register(t1);
RWStructuredBuffer<float> PreviousModelInput : register(u0);

#include "include/depth_constants.hlsl"

[numthreads(16, 16, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= target_w || dtid.y >= target_h || MinMaxEma[0].w < 0.5f)
        return;

    uint plane = target_w * target_h;
    uint idx = dtid.y * target_w + dtid.x;
    PreviousModelInput[idx] = CurrentModelInput[idx];
    PreviousModelInput[idx + plane] = CurrentModelInput[idx + plane];
    PreviousModelInput[idx + 2u * plane] = CurrentModelInput[idx + 2u * plane];
}
