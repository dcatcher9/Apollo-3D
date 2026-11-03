#pragma once
// PRIVATE measurement only. No queued-view API, renderer dispatch or semantic authority.
#include "video_depth_estimator.h"
#include <array>
#include <chrono>
#include <stdexcept>
#include <thread>

namespace external_pre_unmap_feasibility_v1 {
  namespace gate = models::joined_completion_gate_benchmark;
  using clock_t = std::chrono::steady_clock;
  inline std::int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(clock_t::now().time_since_epoch()).count();
  }
  inline void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
  inline gate::snapshot_t status(models::video_depth_estimator &est) {
    const auto s = est.joined_completion_gate_status_for_benchmark();
    require(s.has_value(), "Actual private estimator gate unavailable"); return *s;
  }
  inline models::host_sbs_gpu_completion_receipt::receipt_t receipt(
    models::video_depth_estimator &est, const models::estimate_result &owner) {
    const auto deadline = clock_t::now() + std::chrono::milliseconds {250};
    do {
      const auto p = est.poll_publication_receipts();
      if (p.current_publication && p.current_publication->expected == owner.publication &&
          p.current_publication->depth_cache_authorized()) return *p.current_publication;
      std::this_thread::yield();
    } while (clock_t::now() < deadline);
    throw std::runtime_error("Genuine completed-owner receipt unavailable");
  }
  struct result_t {
    models::estimate_result warmup, A, B;
    models::host_sbs_gpu_completion_receipt::receipt_t warmup_receipt, A_receipt, B_receipt;
    gate::snapshot_t before_B, at_submit_return, after_first_real_poll, final_gate;
    std::int64_t B_submit_begin_us = 0, B_submit_return_us = 0;
    std::int64_t first_poll_begin_us = 0, first_poll_return_us = 0;
    bool first_poll_ready = false;
    std::uint32_t first_poll_query_count = 0;
    const char *classification = "UNAVAILABLE";
  };
  // Configure before ANY input. Source0/1/2 are immutable actual A/B/empty uploads.
  // The common-stream callback is outside graph capture and before the first participating
  // interop unmap. It delays the release tail, not a conditional child. The companion native
  // conditional-graph tests own sibling-order coverage. A callback stall is not a kernel model.
  inline result_t run(models::video_depth_estimator &est,
                      const std::array<ID3D11ShaderResourceView *,3> &source,
                      gate::placement_e placement) {
    require(source[0] && source[1] && source[2] && source[0] != source[1] &&
            source[0] != source[2] && source[1] != source[2], "Three distinct immutable source SRVs required");
    gate::configuration_t config; config.enabled = true; config.placement = placement;
    config.target_frame_id = 3; config.hold_after_callback_enter_ms = 120;
    config.callback_entry_timeout_ms = 30000; config.callback_safety_timeout_ms = 500;
    require(est.configure_joined_completion_gate_for_benchmark(config), "Pre-input gate configuration failed");
    struct release_t {
      models::video_depth_estimator &est;
      ~release_t() { est.release_joined_completion_gate_for_benchmark(); }
    } release {est};
    result_t out;
    const auto submit = [&](std::size_t input, std::uint64_t frame, std::uint64_t obs) {
      const models::gpu_adaptive_reuse_request force {.observation_timestamp_us = obs};
      return est.estimate_depth(source[input], models::input_color_space::srgb, frame, false, {},
                               models::depth_optional_work_mode_e::ordinary, force, 0);
    };
    const auto validate = [](const models::estimate_result &r, std::uint64_t frame, std::uint64_t obs) {
      require(r.completed_frame_valid && r.completed_frame_id == frame &&
              r.publication.observation_timestamp_us == obs && models::parallax_v2_result_is_authenticated(r),
              "Genuine exact-frame completion failed");
    };
    try {
      require(submit(2,1,1).inference_enqueued, "Actual empty warmup did not enqueue");
      out.warmup = est.finish_pending_depth_for_evaluation(models::input_color_space::srgb,false);
      validate(out.warmup,1,1); out.warmup_receipt = receipt(est,out.warmup);
      require(submit(0,2,33334).inference_enqueued, "Actual normal A did not enqueue");
      out.A = est.finish_pending_depth_for_evaluation(models::input_color_space::srgb,false);
      validate(out.A,2,33334); out.A_receipt = receipt(est,out.A);
      out.before_B = status(est);
      require(!out.before_B.attempted && !out.before_B.queued && !out.before_B.released &&
              !out.before_B.cancelled, "Gate was consumed before genuine B submission");

      // No hash, disk output, D3D readback or queued-continuation API in this interval.
      out.B_submit_begin_us = now_us();
      const auto submitted = submit(1,3,66667);
      out.B_submit_return_us = now_us();
      out.at_submit_return = status(est);
      require(submitted.inference_enqueued && !est.has_terminal_failure(), "Actual B submission failed");
      out.first_poll_begin_us = now_us();
      const auto poll = est.try_finish_pending_depth_nonblocking(models::input_color_space::srgb,false);
      out.first_poll_return_us = now_us();
      out.first_poll_ready = poll.ready; out.first_poll_query_count = poll.query_count;
      out.after_first_real_poll = status(est);
      if (poll.ready) out.B = poll.result;
      else out.B = est.finish_pending_depth_for_evaluation(models::input_color_space::srgb,false);
      validate(out.B,3,66667); out.B_receipt = receipt(est,out.B);
      out.final_gate = status(est);
      const auto &s = out.at_submit_return; const auto &f = out.final_gate;
      require(f.attempted && f.queued && f.enqueue_result == CUDA_SUCCESS && f.callback_entered &&
              f.callback_exited && f.released && !f.callback_timed_out && !f.callback_entry_timed_out &&
              f.capture_state_checked && f.capture_state_query_result == CUDA_SUCCESS && f.capture_state == 0 &&
              f.entered_wall_us <= f.released_wall_us && f.released_wall_us <= f.exited_wall_us,
              "Actual one-shot callback entry/release/exit proof failed");
      require(out.B_receipt.depth_owner_frame_id == 3 && out.B_receipt.depth_owner_timestamp_us == 66667,
              "Actual B receipt names another source owner");
      if (placement == gate::placement_e::before_interop_unmaps) {
        const auto valid_unmap = [&](const gate::unmap_probe_t &u, bool required) {
          require(!required || u.issued, "Mandatory real depth-unmap instrumentation missing");
          if (u.issued) require(u.result == CUDA_SUCCESS && out.B_submit_begin_us <= u.begin_wall_us &&
              u.begin_wall_us <= u.return_wall_us && u.return_wall_us <= out.B_submit_return_us,
              "Real interop-unmap failed or timing order invalid");
        };
        valid_unmap(s.optional_ocr_unmap,false); valid_unmap(s.depth_unmap,true);
        require(s.post_unmap_query_recorded && (s.post_unmap_query_result == CUDA_SUCCESS ||
                s.post_unmap_query_result == CUDA_ERROR_NOT_READY) &&
                s.depth_unmap.return_wall_us <= s.post_unmap_query_wall_us &&
                s.post_unmap_query_wall_us <= out.B_submit_return_us,
                "Real post-unmap recorded-event query missing or failed");
        const auto held_during = [&](const gate::unmap_probe_t &u) {
          return u.issued && u.callback_released_when_return_observed &&
                 u.begin_wall_us < f.released_wall_us && f.released_wall_us <= u.return_wall_us;
        };
        if (!s.released && s.post_unmap_query_result == CUDA_ERROR_NOT_READY)
          out.classification = "EARLY_PENDING_WINDOW_OBSERVED_AT_ESTIMATOR_RETURN";
        else if (held_during(s.optional_ocr_unmap) || held_during(s.depth_unmap))
          out.classification = "HOST_UNMAP_STALLED_UNTIL_CALLBACK_RELEASE";
        else if (!s.depth_unmap.callback_released_when_return_observed && s.released)
          out.classification = "UNMAP_RETURNED_EARLY_BUT_SUBMIT_EXPOSED_NO_PENDING_WINDOW";
        else out.classification = "NO_EARLY_PENDING_WINDOW_OBSERVED";
      } else out.classification = "AFTER_UNMAP_CONTROL_ONLY";
      // This result says nothing about queued D3D ordering, latency gain, child scheduling,
      // admission/ACK splitting, current cue presence, or production rendering correctness.
      return out;
    } catch (...) {
      est.release_joined_completion_gate_for_benchmark();
      if (!est.has_terminal_failure()) {
        (void) est.finish_pending_depth_for_evaluation(models::input_color_space::srgb,false);
      }
      throw;
    }
  }
}
