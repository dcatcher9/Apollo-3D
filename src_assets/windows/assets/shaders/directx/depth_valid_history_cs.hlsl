// Preserve the exact preprocessed NCHW color only for frames whose TensorRT output contained a
// valid depth sample and whose resolve pass selected history advance. State 2 holds the last
// structurally reliable color/ordinal/depth tuple for one black or fully clipped update, allowing
// an immediate supported return to be compared with the last visible scene. State 3 advances a
// persistent-low endpoint, so persistent content cannot remain vetoed indefinitely. State 4
// holds a geometry-only candidate's pre-change endpoint for one confirmation update.
StructuredBuffer<float4> MinMaxEma : register(t0);  // w = current-frame validity
StructuredBuffer<float> CurrentModelInput : register(t1);
StructuredBuffer<float> CurrentAppearanceOrdinal : register(t2);
StructuredBuffer<float4> CutBridgeState : register(t3);  // [2].w: 0 empty, 1 advance, 2/4 hold, 3 low
Texture2D<float> CurrentDepth : register(t4);
RWStructuredBuffer<float> PreviousModelInput : register(u0);
RWStructuredBuffer<float> PreviousAppearanceOrdinal : register(u1);
RWTexture2D<float> PreviousReliableDepth : register(u2);

#include "include/depth_constants.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"

[numthreads(16, 16, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= target_w || dtid.y >= target_h ||
        MinMaxEma[0].w < 0.5f ||
        ((SBS_STATE_MODEL_INPUT_HISTORY_STATE(
              CutBridgeState[SBS_STATE_VECTOR_MODEL_INPUT_HISTORY_STATE]) > 1.5f &&
          SBS_STATE_MODEL_INPUT_HISTORY_STATE(
              CutBridgeState[SBS_STATE_VECTOR_MODEL_INPUT_HISTORY_STATE]) < 2.5f) ||
         (SBS_STATE_MODEL_INPUT_HISTORY_STATE(
              CutBridgeState[SBS_STATE_VECTOR_MODEL_INPUT_HISTORY_STATE]) > 3.5f &&
          SBS_STATE_MODEL_INPUT_HISTORY_STATE(
              CutBridgeState[SBS_STATE_VECTOR_MODEL_INPUT_HISTORY_STATE]) < 4.5f)))
        return;

    uint plane = target_w * target_h;
    uint idx = dtid.y * target_w + dtid.x;
    PreviousModelInput[idx] = CurrentModelInput[idx];
    PreviousModelInput[idx + plane] = CurrentModelInput[idx + plane];
    PreviousModelInput[idx + 2u * plane] = CurrentModelInput[idx + 2u * plane];
    PreviousAppearanceOrdinal[idx] = CurrentAppearanceOrdinal[idx];
    PreviousReliableDepth[dtid.xy] = CurrentDepth[dtid.xy];
}
