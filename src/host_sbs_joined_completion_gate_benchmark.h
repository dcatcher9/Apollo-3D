#pragma once
// PRIVATE BENCHMARK ONLY. Default-off, immutable pre-input configuration.
// Not a producer receipt, renderer authority, CPU mock, or model-performance measurement.
#include "cuda_driver_api.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <windows.h>

namespace models::joined_completion_gate_benchmark {
  enum class placement_e : std::uint32_t { after_interop_unmaps = 0u, before_interop_unmaps = 1u };
  enum class unmap_e : std::uint32_t { optional_ocr = 0u, depth = 1u };
  struct unmap_probe_t {
    bool issued = false;
    int result = 0;
    std::int64_t begin_wall_us = 0, return_wall_us = 0;
    bool callback_entered_when_return_observed = false, callback_released_when_return_observed = false;
  };
  struct configuration_t {
    bool enabled = false;
    placement_e placement = placement_e::after_interop_unmaps;
    std::uint64_t target_frame_id = 0;
    std::uint32_t hold_after_callback_enter_ms = 120;
    std::uint32_t callback_entry_timeout_ms = 30000;
    std::uint32_t callback_safety_timeout_ms = 500;
    bool valid() const noexcept {
      // The explicit packed-continuity regression crosses its 250 ms presentation cap with a
      // 350 ms gate. This private allowance stays below the independent 500 ms callback fuse.
      return !enabled || ((placement == placement_e::after_interop_unmaps || placement == placement_e::before_interop_unmaps) && target_frame_id != 0 && hold_after_callback_enter_ms > 0 &&
        hold_after_callback_enter_ms <= 400 && callback_entry_timeout_ms > 0 &&
        callback_entry_timeout_ms <= 30000 && callback_safety_timeout_ms > hold_after_callback_enter_ms &&
        callback_safety_timeout_ms <= 500);
    }
  };
  struct snapshot_t {
    configuration_t configuration;
    unmap_probe_t optional_ocr_unmap, depth_unmap;
    bool post_unmap_query_recorded = false;
    int post_unmap_query_result = CUDA_ERROR_NOT_READY;
    std::int64_t post_unmap_query_wall_us = 0;
    bool attempted = false, queued = false, callback_entered = false, callback_exited = false;
    bool released = false, cancelled = false, callback_timed_out = false, callback_entry_timed_out = false;
    int enqueue_result = 0, last_actual_event_query = CUDA_ERROR_NOT_READY;
    bool capture_state_checked = false;
    int capture_state_query_result = 0, capture_state = -1;
    std::uint32_t actual_event_queries = 0, actual_not_ready_queries = 0, actual_ready_queries = 0, actual_failed_queries = 0;
    // CPU benchmark walltime only. NEVER observation/capture/geometry clock authority.
    std::int64_t entered_wall_us = 0, released_wall_us = 0, exited_wall_us = 0;
  };

  namespace detail {
    inline std::int64_t now_us() noexcept {
      return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    struct event_t {
      HANDLE value = CreateEventW(nullptr, TRUE, FALSE, nullptr);
      ~event_t() { if (value) CloseHandle(value); }
      event_t() = default;
      event_t(const event_t &) = delete;
      event_t &operator=(const event_t &) = delete;
    };
    struct state_t {
      explicit state_t(configuration_t value): config(value) {}
      configuration_t config;
      struct unmap_atomic_t {
        std::atomic<bool> issued {false}, entered {false}, released {false};
        std::atomic<int> result {0};
        std::atomic<std::int64_t> begin {0}, returned {0};
      };
      unmap_atomic_t ocr_unmap, depth_unmap;
      std::atomic<bool> post_query_recorded {false};
      std::atomic<int> post_query_result {CUDA_ERROR_NOT_READY};
      std::atomic<std::int64_t> post_query_wall {0};
      event_t entered, release, exited;
      std::atomic<bool> attempted {false}, queued {false}, entered_flag {false}, exited_flag {false};
      std::atomic<bool> released {false}, cancelled {false}, callback_timeout {false}, entry_timeout {false};
      std::atomic<int> enqueue_result {0}, last_query {CUDA_ERROR_NOT_READY};
      std::atomic<bool> capture_checked {false};
      std::atomic<int> capture_query_result {0}, capture_status {-1};
      std::atomic<std::uint32_t> queries {0}, busy_queries {0}, ready_queries {0}, failed_queries {0};
      std::atomic<std::int64_t> entered_at {0}, released_at {0}, exited_at {0};
      bool valid_events() const noexcept { return entered.value && release.value && exited.value; }
      void signal_release() noexcept {
        if (!released.exchange(true)) released_at.store(now_us());
        if (release.value) SetEvent(release.value);
      }
    };
    using host_function_t = void (__stdcall *)(void *);
    using launch_t = CUresult (__stdcall *)(CUstream, host_function_t, void *);
    inline launch_t resolve_launch(cuda_driver_api &api) noexcept {
      launch_t fn = nullptr;
      const auto symbol = api.hMod ? GetProcAddress(api.hMod, "cuLaunchHostFunc") : nullptr;
      static_assert(sizeof(fn) == sizeof(symbol));
      std::memcpy(&fn, &symbol, sizeof(fn));
      return fn;
    }
    // CUstreamCaptureStatus is a four-byte enum; NONE=0. Resolve privately so the
    // production CUDA loader/ABI and ordinary disabled path remain untouched.
    using is_capturing_t = CUresult (__stdcall *)(CUstream, int *);
    inline is_capturing_t resolve_is_capturing(cuda_driver_api &api) noexcept {
      is_capturing_t fn = nullptr;
      const auto symbol = api.hMod ? GetProcAddress(api.hMod, "cuStreamIsCapturing") : nullptr;
      static_assert(sizeof(fn) == sizeof(symbol));
      std::memcpy(&fn, &symbol, sizeof(fn));return fn;
    }
    struct callback_payload_t { std::shared_ptr<state_t> state; };
    inline void __stdcall host_callback(void *opaque) noexcept {
      // The callback owns this payload. If a CUDA context error prevents callback invocation,
      // this ONE private benchmark payload is intentionally retained: never free queued userdata
      // on the assumption that a failed root/teardown made it safe. Existing context quarantine
      // remains unchanged. Normal callback exit releases every handle through shared ownership.
      std::unique_ptr<callback_payload_t> payload(static_cast<callback_payload_t *>(opaque));
      const auto state = payload->state;
      state->entered_at.store(now_us());
      state->entered_flag.store(true);
      SetEvent(state->entered.value);
      const DWORD waited = WaitForSingleObject(state->release.value, state->config.callback_safety_timeout_ms);
      if (waited != WAIT_OBJECT_0) { state->callback_timeout.store(true); state->signal_release(); }
      state->exited_at.store(now_us());
      state->exited_flag.store(true);
      SetEvent(state->exited.value);
      // No CUDA, D3D, logging, model access, or GPU-dependent synchronization in this callback.
    }
  }

  class controller_t {
  public:
    explicit controller_t(configuration_t config): state_(std::make_shared<detail::state_t>(config)) {}
    ~controller_t() { cancel_and_release(); }
    controller_t(const controller_t &) = delete;
    controller_t &operator=(const controller_t &) = delete;
    bool available(cuda_driver_api &api) const noexcept {
      return state_->config.valid() && state_->valid_events() && detail::resolve_launch(api) != nullptr &&
        detail::resolve_is_capturing(api) != nullptr && api.cuEventRecord && api.cuEventQuery;
    }
    bool targets(std::uint64_t frame) const noexcept {
      return state_->config.enabled && frame == state_->config.target_frame_id && !state_->attempted.load();
    }
    bool is_pre_unmap_probe(std::uint64_t frame) const noexcept {
      return state_->config.enabled && state_->config.placement == placement_e::before_interop_unmaps && frame == state_->config.target_frame_id;
    }
    CUresult enqueue_before_interop_unmaps(cuda_driver_api &api, CUstream stream, std::uint64_t frame) noexcept {
      return enqueue_at_placement(api, stream, frame, placement_e::before_interop_unmaps);
    }
    CUresult enqueue_after_interop_unmaps(cuda_driver_api &api, CUstream stream, std::uint64_t frame) noexcept {
      return enqueue_at_placement(api, stream, frame, placement_e::after_interop_unmaps);
    }
    CUresult enqueue_at_placement(cuda_driver_api &api, CUstream stream, std::uint64_t frame, placement_e actual_site) noexcept {
      if (state_->config.placement != actual_site || !targets(frame)) return CUDA_SUCCESS;
      state_->attempted.store(true);
      const auto launch = detail::resolve_launch(api);
      if (!launch || !stream || !state_->valid_events()) {
        state_->enqueue_result.store(CUDA_ERROR_NOT_SUPPORTED);state_->signal_release();return CUDA_ERROR_NOT_SUPPORTED;
      }
      // Never capture this one-shot owning callback into a replayable graph.
      // A real driver query immediately at the actual enqueue site is mandatory.
      const auto is_capturing = detail::resolve_is_capturing(api);
      int capture_status = -1;
      const auto capture_result = is_capturing ? is_capturing(stream, &capture_status) : CUDA_ERROR_NOT_SUPPORTED;
      state_->capture_query_result.store(capture_result);state_->capture_status.store(capture_status);
      state_->capture_checked.store(true);
      if (capture_result != CUDA_SUCCESS || capture_status != 0) {
        const auto result = capture_result != CUDA_SUCCESS ? capture_result : CUDA_ERROR_NOT_SUPPORTED;
        state_->enqueue_result.store(result);state_->signal_release();return result;
      }
      detail::callback_payload_t *payload = nullptr;
      try {
        payload = new detail::callback_payload_t {state_};
        const auto state = state_;
        // This thread never depends on successful capture, rendering, CUDA, or callback completion.
        // Once entry is observed, release is fixed walltime later even if the render thread fails.
        timer_ = std::thread([state] {
          HANDLE handles[] {state->entered.value, state->release.value};
          const DWORD entry = WaitForMultipleObjects(2, handles, FALSE, state->config.callback_entry_timeout_ms);
          if (entry == WAIT_OBJECT_0) {
            (void) WaitForSingleObject(state->release.value, state->config.hold_after_callback_enter_ms);
          } else if (entry != WAIT_OBJECT_0 + 1) {
            state->entry_timeout.store(true);
          }
          state->signal_release();
        });
        const CUresult result = launch(stream, &detail::host_callback, payload);
        state_->enqueue_result.store(result);
        state_->queued.store(result == CUDA_SUCCESS);
        // A later CUDA call may surface an earlier asynchronous fault. Conservatively retain
        // the one payload on an error return rather than infer that userdata cannot be live.
        if (result != CUDA_SUCCESS) state_->signal_release();
        return result;
      } catch (...) {
        delete payload;state_->enqueue_result.store(-1);state_->signal_release();return static_cast<CUresult>(-1);
      }
    }
    void note_unmap_begin(std::uint64_t frame, unmap_e kind) noexcept {
      if (!is_pre_unmap_probe(frame)) return;
      auto &u = kind == unmap_e::optional_ocr ? state_->ocr_unmap : state_->depth_unmap;
      u.begin.store(detail::now_us()); u.issued.store(true);
    }
    void note_unmap_return(std::uint64_t frame, unmap_e kind, CUresult result) noexcept {
      if (!is_pre_unmap_probe(frame)) return;
      auto &u = kind == unmap_e::optional_ocr ? state_->ocr_unmap : state_->depth_unmap;
      u.returned.store(detail::now_us()); u.result.store(result);
      u.entered.store(state_->entered_flag.load()); u.released.store(state_->released.load());
    }
    void query_after_successful_unmap_calls(cuda_driver_api &api, CUevent actual_done_event, std::uint64_t frame) noexcept {
      if (!is_pre_unmap_probe(frame) || !actual_done_event || !api.cuEventQuery) return;
      // Real query only; result never changes admission, completion, or branch state.
      const auto result = api.cuEventQuery(actual_done_event);
      state_->post_query_result.store(result); state_->post_query_wall.store(detail::now_us());
      state_->post_query_recorded.store(true);
    }
    void note_actual_event_query(std::uint64_t frame, CUresult result) noexcept {
      if (frame != state_->config.target_frame_id || !state_->queued.load()) return;
      state_->last_query.store(result);++state_->queries;
      if (result == CUDA_ERROR_NOT_READY) ++state_->busy_queries;
      else if (result == CUDA_SUCCESS) ++state_->ready_queries;
      else ++state_->failed_queries;
    }
    void cancel_and_release() noexcept {
      state_->cancelled.store(true);state_->signal_release();
      if (timer_.joinable()) timer_.join();
    }
    snapshot_t snapshot() const noexcept {
      const auto read_unmap = [](const detail::state_t::unmap_atomic_t &u) {
        unmap_probe_t p; p.issued=u.issued.load(); p.result=u.result.load();
        p.begin_wall_us=u.begin.load(); p.return_wall_us=u.returned.load();
        p.callback_entered_when_return_observed=u.entered.load(); p.callback_released_when_return_observed=u.released.load(); return p;
      };
      snapshot_t s;s.optional_ocr_unmap=read_unmap(state_->ocr_unmap);s.depth_unmap=read_unmap(state_->depth_unmap);
      s.post_unmap_query_recorded=state_->post_query_recorded.load();s.post_unmap_query_result=state_->post_query_result.load();
      s.post_unmap_query_wall_us=state_->post_query_wall.load();s.configuration=state_->config;s.attempted=state_->attempted.load();s.queued=state_->queued.load();
      s.callback_entered=state_->entered_flag.load();s.callback_exited=state_->exited_flag.load();s.released=state_->released.load();
      s.cancelled=state_->cancelled.load();s.callback_timed_out=state_->callback_timeout.load();s.callback_entry_timed_out=state_->entry_timeout.load();
      s.enqueue_result=state_->enqueue_result.load();s.last_actual_event_query=state_->last_query.load();
      s.capture_state_checked=state_->capture_checked.load();s.capture_state_query_result=state_->capture_query_result.load();s.capture_state=state_->capture_status.load();
      s.actual_event_queries=state_->queries.load();s.actual_not_ready_queries=state_->busy_queries.load();
      s.actual_ready_queries=state_->ready_queries.load();s.actual_failed_queries=state_->failed_queries.load();
      s.entered_wall_us=state_->entered_at.load();s.released_wall_us=state_->released_at.load();s.exited_wall_us=state_->exited_at.load();return s;
    }
  private:
    std::shared_ptr<detail::state_t> state_;
    std::thread timer_;
  };
}
