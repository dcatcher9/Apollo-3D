// WARP-only adversarial validator for an already-published subtitle-locator state. Keeping this
// entrypoint outside the production shader specs makes it impossible to include in the live
// closure while still exercising the exact production snapshot validator and verdict writer.
#include "../host_sbs_subtitle_locator_cs.hlsl"

[numthreads(1, 1, 1)]
void condition_validate_test_main(uint3 dispatch_id : SV_DispatchThreadID) {
    [loop]
    for (uint index = 0u; index < V2_SUBTITLE_LOCATOR_STATE_WORD_COUNT; ++index) {
        ConditionStateSnapshot[index] = LocatorStateRead[index];
    }
    GroupMemoryBarrierWithGroupSync();
    PublishConditionParamsFromSnapshot();
}
