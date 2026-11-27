/**
 * @file src/host_sbs_depth_gpu_timing.h
 * @brief Nonblocking D3D11 timestamp ring for the depth estimator's diagnostics.
 */
#pragma once

#include "host_sbs_telemetry_perf.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <d3d11.h>
#include <memory>
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
}  // namespace models::host_sbs_depth_gpu_timing
