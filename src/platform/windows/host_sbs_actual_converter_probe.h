#pragma once
// PRIVATE opt-in scheduling fixture. No generic encoder/config/capture API is changed.
#ifdef _WIN32
  #include "src/config.h"
  #include "src/video_depth_estimator.h"
  #include "src/host_sbs_pending_current_colour_lane.h"
  #include <array>
  #include <chrono>
  #include <cstdint>
  #include <filesystem>
  #include <string>
  #include <vector>
  #include <wrl/client.h>

namespace platf::dxgi::actual_converter_probe_v1 {
  struct frame_t {
    std::uint32_t width = 0, height = 0;
    std::vector<std::uint8_t> bgra;
    std::vector<std::uint16_t> rgba16; // Private HDR source payload; never quantized to BGRA8.
    std::string source_file_sha256;
    std::uint64_t declared_timeline_us = 0; // metadata only, never the model clock
  };
  struct request_t {
    bool enabled = false, queued = false;
    bool background_observer = false; // Explicit V7 HDR qualification only; no default readbacks.
    bool hdr_input = false; // Explicit three-PFM current-outline probe only; default SDR unchanged.
    bool explicit_adapter = false; // Private probe plan binding for SDR or HDR; absent SDR keeps its legacy default.
    std::uint32_t hdr_adapter_luid_low=0u,hdr_adapter_vendor_id=0u,hdr_adapter_device_id=0u;
    std::int32_t hdr_adapter_luid_high=0; // Explicit HDR plan identity; no implicit adapter choice.
    models::research_pending_current_colour_v1::configuration_t pending_current_colour {};
    // Explicit source-bridge activation only; native-image fixtures without DDup remain UNKNOWN.
    std::optional<models::subtitle_temporal_source_research_configuration_t> temporal_source_bridge;
    std::optional<models::subtitle_selected_matte_research_configuration_t> selected_matte;
    std::array<frame_t, 3> frames; // immutable A, B, C inputs, NOT presence labels
    config::video_t::sbs_t settings;
    std::filesystem::path output;
  };
  // Same-TU implementation owns the actual d3d_base_encode_device RGB path.
  int run(const request_t &request, std::string &error) noexcept;

  struct context_t {
    std::uint32_t width = 0, height = 0;
    std::uint64_t instance = 0;
    // This is a declared synthetic scheduling domain, NEVER actual Win32 video eligibility.
  };
  enum class event_e : std::uint32_t { admission, queue_begin, queued, ack, draw, packed_repeat, convert_end };
  // PRIVATE observed actual draw bindings; this carries no new UI/EMPTY authority.
  struct background_draw_t {
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> field,proof;
    Microsoft::WRL::ComPtr<ID3D11Buffer> expected,source_c24;
    bool queue_issued=false,role_matched=false,bound=false,ps_observed_after_draw=false;
    std::uint64_t owner_serial=0,transport_serial=0,context_address=0,pixel_shader_address=0;
    std::array<std::uint64_t,4> colour_key{};
    std::array<std::uint64_t,7> actual_role_pointers{};
    std::array<std::uint32_t,4> analysis_shape{};
    std::string pixel_shader_closure;
  };
  struct event_t {
    event_e kind {};
    std::uint64_t conversion = 0, model_frame = 0, observation_us = 0;
    std::int64_t host_event_us = 0;
    std::int64_t copied_at_ticks = 0, copied_at_us = 0;
    std::int64_t source_stamp_ticks = 0, content_stamp_ticks = 0;
    std::int64_t current_source_stamp_ticks = 0, current_content_stamp_ticks = 0;
    std::uint64_t context_generation = 0, transaction_token = 0;
    std::uint64_t watermark_before = 0, watermark_after = 0, pixel_source_frame = 0;
    std::uint64_t actual_draw_pixel_source = 0;
    std::int64_t pixel_source_stamp_ticks = 0, packed_age_us = 0;
    std::uint32_t pending_slots = 0, optional_work = 0, request_flags = 0;
    std::array<std::uint64_t, 2> slot_frames {}, slot_observations {}, slot_contexts {}, slot_source_addresses {};
    std::array<bool, 2> slot_pending {};
    std::uint32_t selected_slot = 0xffffffffu;
    bool force_infer = false, gpu_undecided = false, ddup_proof = false;
    bool queued = false, flat = false, repeat = false, route_matches = false, source_matches = false;
    bool cached_current_colour = false;
    bool pending_current_colour=false, material_interval_contiguous=false;
    std::uint64_t material_epoch=0,material_serial=0,material_capture_clock_us=0;
    std::uint32_t negative_bank=0xffffffffu;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> negative_current_source,negative_current_receipt,negative_receipt;
    Microsoft::WRL::ComPtr<ID3D11Buffer> negative_current_constants,negative_expected;
    std::array<std::array<Microsoft::WRL::ComPtr<ID3D11Resource>,16>,2> negative_owner_resources;
    bool packed_valid = false, query_observed = false, gate_entered = false, gate_released = false;
    std::int32_t literal_last_query_result = 0;
    std::uint32_t literal_query_count = 0, literal_not_ready_count = 0, literal_ready_count = 0;
    std::uint64_t literal_query_target_frame = 0;
    bool publication_current = false, ticket_publication_current = false;
    models::host_sbs_gpu_completion_receipt::expected_t publication {};
    models::host_sbs_gpu_completion_receipt::expected_t planned_publication {}; // value only, never proof
    models::depth_input_region_t region {};
    models::depth_tensor_content_rect_t field_content {};
    std::uint32_t field_width = 0, field_height = 0;
    // Selected DRAW source (null for packed-repeat), Base,state,M,RGB,SCP,SCT,OCR,UIP,cut,frame,EMA.
    std::array<Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>, 12> views;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> current_input;
    // b2,b3,b5,raw,args,depthCB,sourceRegionCB,contourCB. Export ACTUAL GetDesc byte lengths.
    std::array<Microsoft::WRL::ComPtr<ID3D11Buffer>, 8> buffers;
    // Selected-sidecar observer contract, appended; original views/buffers keep their roles.
    // selected_views: M,receipt,actualDrawM,baseBody,baseM,numeric32,P,S,T_M,T_receipt,Frame256.
    std::array<Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>,11> selected_views;
    // Diagnostic references only; the recorder snapshots before the next queue.
    std::array<Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>,12> current_outline_stages;
    Microsoft::WRL::ComPtr<ID3D11Buffer> selected_contour_constants;
    bool selected_draw=false,selected_baseline_current=false,selected_temporal_commands=false,selected_source_certificate_valid=false;
    std::uint64_t actual_draw_selected_receipt_address=0u;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> packed;
    background_draw_t background;
  };
  struct observer_t {
    void *state = nullptr;
    void (*record)(void *, const event_t &) noexcept = nullptr;
    void emit(const event_t &event) const noexcept { if (record) record(state, event); }
  };
}
namespace platf::dxgi::pending_current_colour_probe_v1 {
  int run(const actual_converter_probe_v1::request_t &request,std::string &error) noexcept;
}
#endif

// PRIVATE fixed-six sequence protocol. The original three-source request and scheduler above
// remain unchanged. No authored P/S/OCR/C24 or draw-mask input is accepted here.
#ifdef _WIN32
namespace platf::dxgi::actual_converter_probe_v1 {
  struct sequence6_request_t {
    request_t common;
    std::array<frame_t,6> frames;
  };
  int run_sequence6(const sequence6_request_t &,std::string &error) noexcept;
}
#endif
