#pragma once
// Observer and synthetic scheduling context for the converter test fixtures
// (model_preparation_converter_test.h). Ordinary streaming sets neither.
#ifdef _WIN32
  #include "src/video_depth_estimator.h"
  #include <array>
  #include <cstdint>
  #include <wrl/client.h>

namespace platf::dxgi::converter_test_observer {
  struct context_t {
    std::uint32_t width = 0, height = 0;
    std::uint64_t instance = 0;
    // This is a declared synthetic scheduling domain, NEVER actual Win32 video eligibility.
    bool borrow_matched_capture = false;
  };
  enum class event_e : std::uint32_t { admission, ack, draw, packed_repeat, convert_end };
  struct event_t {
    event_e kind {};
    std::uint64_t conversion = 0, model_frame = 0, observation_us = 0;
    std::int64_t source_stamp_ticks = 0, content_stamp_ticks = 0;
    std::int64_t current_source_stamp_ticks = 0, current_content_stamp_ticks = 0;
    std::uint64_t watermark_before = 0, watermark_after = 0;
    std::uint64_t actual_draw_pixel_source = 0;
    std::int64_t pixel_source_stamp_ticks = 0;
    std::uint32_t pending_slots = 0, optional_work = 0;
    std::array<std::uint64_t, 2> slot_frames {};
    std::array<bool, 2> slot_pending {};
    bool force_infer = false, gpu_undecided = false;
    bool flat = false, repeat = false, route_matches = false;
    bool native_planes_written = false; // Output-event evidence; independent of packed RGB reuse.
    bool current_capture_key = false, prior_capture_key = false;
    bool cached_current_colour = false;
    bool packed_valid = false, query_observed = false, gate_entered = false, gate_released = false;
    std::int32_t literal_last_query_result = 0;
    std::uint64_t literal_query_target_frame = 0;
    bool publication_current = false;
    models::host_sbs_gpu_completion_receipt::expected_t publication {};
    // The selected draw source (null for a packed repeat) and the completed EMA motion mask.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> draw_source, ema_motion_mask;
  };
  struct observer_t {
    void *state = nullptr;
    void (*record)(void *, const event_t &) noexcept = nullptr;
    void emit(const event_t &event) const noexcept { if (record) record(state, event); }
  };
}
#endif
