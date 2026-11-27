/**
 * @file src/host_sbs_depth_gpu_timing.h
 * @brief Nonblocking D3D11 timestamp and CUDA event rings for the depth estimator's diagnostics.
 */
#pragma once

#include "cuda_driver_api.h"
#include "host_sbs_telemetry_perf.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <d3d11.h>
#include <memory>
#include <optional>
#include <string_view>
#include <wrl/client.h>

namespace models::host_sbs_depth_gpu_timing {
  // The timestamp queries of one estimator frame. The estimator ends the start and end markers
  // of each pass it records; the has_* flags say which pairs the frame contains.
  struct d3d_slot_t {
    Microsoft::WRL::ComPtr<ID3D11Query> disjoint;
    Microsoft::WRL::ComPtr<ID3D11Query> post_start;
    Microsoft::WRL::ComPtr<ID3D11Query> post_end;
    Microsoft::WRL::ComPtr<ID3D11Query> parallax_frame_stats_start;
    Microsoft::WRL::ComPtr<ID3D11Query> parallax_frame_stats_end;
    Microsoft::WRL::ComPtr<ID3D11Query> parallax_start;
    Microsoft::WRL::ComPtr<ID3D11Query> parallax_end;
    Microsoft::WRL::ComPtr<ID3D11Query> parallax_map_start;
    Microsoft::WRL::ComPtr<ID3D11Query> parallax_limits_start;
    Microsoft::WRL::ComPtr<ID3D11Query> parallax_subtitle_start;
    Microsoft::WRL::ComPtr<ID3D11Query> pre_start;
    Microsoft::WRL::ComPtr<ID3D11Query> pre_end;
    bool pending = false;
    bool has_post = false;
    bool has_parallax_frame_stats = false;
    bool has_parallax = false;
    bool has_parallax_map_start = false;
    bool has_parallax_limits_start = false;
    bool has_parallax_subtitle_start = false;
    bool has_pre = false;
    std::uint64_t perf_generation = 0;
  };

  // Slots retire out of order once their queries complete, never by waiting on the GPU.
  class d3d_ring_t {
  public:
    // Allocates the queries only for a diagnostics estimator; any failure leaves timing off.
    void initialize(ID3D11Device *device, bool diagnostics_enabled);

    // Publishes every completed slot's samples and frees it.
    void resolve(
      ID3D11DeviceContext *context,
      const std::shared_ptr<host_sbs_telemetry::collector> &telemetry_performance
    );

    // Resolves completed slots, then starts the disjoint and first timestamp of a free slot.
    // Returns nullptr when timing is off or every slot is still in flight.
    d3d_slot_t *begin(
      ID3D11DeviceContext *context,
      const std::shared_ptr<host_sbs_telemetry::collector> &telemetry_performance,
      bool has_post,
      bool has_pre
    );

  private:
    static constexpr std::size_t d3d_perf_ring_size = 16;
    std::array<d3d_slot_t, d3d_perf_ring_size> d3d_perf_slots;
    std::size_t d3d_perf_next = 0;
    bool d3d_perf_ready = false;
  };

  // One failed CUDA call of the event ring. The ring only abandons the measurement it was
  // making; its owner decides what the failure means for the stream.
  struct cuda_event_failure_t {
    std::string_view operation;
    CUresult result;
  };

  // GPU-stream timing of an async enqueue (diagnostics only). A small ring of CUDA event pairs
  // lets several enqueues be in flight; each elapsed time is read back lazily once its stop
  // event completes. Every call runs on the owner's thread with the owner's CUDA context
  // current. The ring owns its events, their recording and their readback, and returns every
  // CUDA failure instead of applying a policy to it.
  class cuda_event_ring_t {
  public:
    static constexpr int slot_count = 4;

    struct resolution_t {
      std::optional<float> elapsed_ms;  // The slot's sample; the slot is free again.
      std::optional<cuda_event_failure_t> failure;  // The slot stays in flight.
    };

    // Reads back the slot's sample once its stop event has completed. An idle slot, a slot
    // still in flight and a driver without event queries return neither.
    resolution_t try_resolve(int slot, cuda_driver_api &cuda);

    // The slot the next measurement uses.
    [[nodiscard]] int next_slot() const noexcept {
      return head;
    }

    [[nodiscard]] bool in_flight(int slot) const noexcept {
      return busy[slot];
    }

    // Whether the driver exports the calls record_start makes.
    [[nodiscard]] static bool recording_available(const cuda_driver_api &cuda) noexcept {
      return cuda.cuEventCreate && cuda.cuEventRecord;
    }

    // Creates the slot's events on first use, then records its start event.
    std::optional<cuda_event_failure_t> record_start(int slot, CUstream stream, cuda_driver_api &cuda);

    // Records the slot's stop event, marks the slot in flight and advances the ring.
    std::optional<cuda_event_failure_t> record_stop(int slot, CUstream stream, cuda_driver_api &cuda);

    // Destroys every created event. Returns false at the first failed destroy and keeps that
    // and every later handle, so no other CUDA teardown call follows a possibly deferred fault.
    [[nodiscard]] bool destroy_events(cuda_driver_api &cuda);

  private:
    CUevent start[slot_count] {};
    CUevent stop[slot_count] {};
    bool busy[slot_count] {};
    int head = 0;
  };
}  // namespace models::host_sbs_depth_gpu_timing
