// Opt-in direct glyph receipt authorization for the completed shared warp.
#define HOST_SBS_RAW_MODEL_SELECTED_MATTE 1
#define HOST_SBS_DIRECT_GLYPH_SELECTED_MATTE 1
#include "host_sbs_shared_ui_warp.hlsl"

[numthreads(1, 1, 1)]
void prepare_direct_glyph_main(uint3 id : SV_DispatchThreadID) {
    prepare_main(id);
}
