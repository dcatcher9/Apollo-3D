// Opt-in RawMatte receipt authorization with the same preparation as Rule.
#define HOST_SBS_RAW_MODEL_SELECTED_MATTE 1
#include "host_sbs_shared_ui_warp.hlsl"

[numthreads(1,1,1)]
void prepare_raw_model_bench_main(uint3 id : SV_DispatchThreadID) {
    prepare_main(id);
}
