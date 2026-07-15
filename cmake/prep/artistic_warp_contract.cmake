# Hash the complete implementation boundary that can change Apollo's processed depth, subject
# state, disparity field, or the benchmark evidence used to approve a learned camera policy.
# Normalize line endings so an otherwise-identical checkout does not invalidate a checkpoint.
set(APOLLO_ARTISTIC_WARP_CONTRACT_FILES
        "${CMAKE_SOURCE_DIR}/src/video_depth_estimator.cpp"
        "${CMAKE_SOURCE_DIR}/src/platform/windows/display_vram.cpp"
        "${CMAKE_SOURCE_DIR}/src/sbs_bench_harness.cpp"
        "${CMAKE_SOURCE_DIR}/src_assets/windows/assets/shaders/directx/rgb_to_nchw_cs.hlsl"
        "${CMAKE_SOURCE_DIR}/src_assets/windows/assets/shaders/directx/buffer_to_tex_cs.hlsl"
        "${CMAKE_SOURCE_DIR}/src_assets/windows/assets/shaders/directx/depth_ema_motion_cs.hlsl"
        "${CMAKE_SOURCE_DIR}/src_assets/windows/assets/shaders/directx/depth_minmax_cs.hlsl"
        "${CMAKE_SOURCE_DIR}/src_assets/windows/assets/shaders/directx/depth_minmax_ema_cs.hlsl"
        "${CMAKE_SOURCE_DIR}/src_assets/windows/assets/shaders/directx/depth_hist_cs.hlsl"
        "${CMAKE_SOURCE_DIR}/src_assets/windows/assets/shaders/directx/depth_subject_hist_cs.hlsl"
        "${CMAKE_SOURCE_DIR}/src_assets/windows/assets/shaders/directx/depth_subject_resolve_cs.hlsl"
        "${CMAKE_SOURCE_DIR}/src_assets/windows/assets/shaders/directx/depth_warp_prefilter_cs.hlsl"
        "${CMAKE_SOURCE_DIR}/src_assets/windows/assets/shaders/directx/sbs_forward_coverage_cs.hlsl"
        "${CMAKE_SOURCE_DIR}/src_assets/windows/assets/shaders/directx/sbs_reprojection_vs.hlsl"
        "${CMAKE_SOURCE_DIR}/src_assets/windows/assets/shaders/directx/sbs_reprojection_ps.hlsl"
        "${CMAKE_SOURCE_DIR}/src_assets/windows/assets/shaders/directx/include/bestv2_curve.hlsl"
        "${CMAKE_SOURCE_DIR}/src_assets/windows/assets/shaders/directx/include/depth_color.hlsl"
        "${CMAKE_SOURCE_DIR}/src_assets/windows/assets/shaders/directx/include/depth_constants.hlsl"
        "${CMAKE_SOURCE_DIR}/src_assets/windows/assets/shaders/directx/include/sbs_warp_common.hlsl")

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        ${APOLLO_ARTISTIC_WARP_CONTRACT_FILES})

set(_apollo_artistic_warp_contract "")
foreach(_contract_file IN LISTS APOLLO_ARTISTIC_WARP_CONTRACT_FILES)
    file(RELATIVE_PATH _contract_relative "${CMAKE_SOURCE_DIR}" "${_contract_file}")
    file(READ "${_contract_file}" _contract_contents)
    string(REPLACE "\r\n" "\n" _contract_contents "${_contract_contents}")
    string(REPLACE "\r" "\n" _contract_contents "${_contract_contents}")
    string(APPEND _apollo_artistic_warp_contract
            "${_contract_relative}\n${_contract_contents}\n")
endforeach()
string(SHA256 APOLLO_ARTISTIC_WARP_CONTRACT_SHA256
        "${_apollo_artistic_warp_contract}")
list(APPEND SUNSHINE_DEFINITIONS
        APOLLO_ARTISTIC_WARP_CONTRACT_SHA256="${APOLLO_ARTISTIC_WARP_CONTRACT_SHA256}")
message(STATUS "Apollo artistic warp contract: ${APOLLO_ARTISTIC_WARP_CONTRACT_SHA256}")

# A learned policy is approved against both the renderer above and the exact evaluator contract
# that selected its safe frontier. Match tools/sbsbench/run_eval.py's semantic hash:
# basename + normalized contents, in this fixed order, truncated to 16 hex characters.
set(APOLLO_ARTISTIC_METRIC_CONTRACT_FILES
        "${CMAKE_SOURCE_DIR}/tools/sbsbench/sbsbench.py"
        "${CMAKE_SOURCE_DIR}/tools/sbsbench/thresholds.json"
        "${CMAKE_SOURCE_DIR}/tools/sbsbench/run_eval.py")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        ${APOLLO_ARTISTIC_METRIC_CONTRACT_FILES})
set(_apollo_artistic_metric_contract "")
foreach(_metric_file IN LISTS APOLLO_ARTISTIC_METRIC_CONTRACT_FILES)
    get_filename_component(_metric_name "${_metric_file}" NAME)
    file(READ "${_metric_file}" _metric_contents)
    string(REPLACE "\r\n" "\n" _metric_contents "${_metric_contents}")
    string(REPLACE "\r" "\n" _metric_contents "${_metric_contents}")
    string(APPEND _apollo_artistic_metric_contract
            "${_metric_name}${_metric_contents}")
endforeach()
string(SHA256 _apollo_artistic_metric_contract_full
        "${_apollo_artistic_metric_contract}")
string(SUBSTRING "${_apollo_artistic_metric_contract_full}" 0 16
        APOLLO_ARTISTIC_METRIC_CONTRACT_SHA256)
list(APPEND SUNSHINE_DEFINITIONS
        APOLLO_ARTISTIC_METRIC_CONTRACT_SHA256="${APOLLO_ARTISTIC_METRIC_CONTRACT_SHA256}")
message(STATUS "Apollo artistic metric contract: ${APOLLO_ARTISTIC_METRIC_CONTRACT_SHA256}")
