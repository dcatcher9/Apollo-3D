/**
 * @file src/host_sbs_depth_gpu_timing.cpp
 * @brief Nonblocking D3D11 timestamp ring for the depth estimator's diagnostics.
 */
#include "host_sbs_depth_gpu_timing.h"

#include "logging.h"
#include "sbs_perf.h"

#include <chrono>

namespace models::host_sbs_depth_gpu_timing {
  void d3d_ring_t::initialize(ID3D11Device *device, bool diagnostics_enabled) {
    if (!diagnostics_enabled) {
      return;
    }
    for (auto &slot : d3d_perf_slots) {
      D3D11_QUERY_DESC desc {D3D11_QUERY_TIMESTAMP_DISJOINT, 0};
      if (FAILED(device->CreateQuery(&desc, &slot.disjoint))) {
        BOOST_LOG(warning) << "Depth D3D11 timing unavailable: could not create disjoint query.";
        return;
      }
      desc.Query = D3D11_QUERY_TIMESTAMP;
      if (FAILED(device->CreateQuery(&desc, &slot.post_start)) ||
          FAILED(device->CreateQuery(&desc, &slot.post_end)) ||
          FAILED(device->CreateQuery(&desc, &slot.parallax_frame_stats_start)) ||
          FAILED(device->CreateQuery(&desc, &slot.parallax_frame_stats_end)) ||
          FAILED(device->CreateQuery(&desc, &slot.parallax_start)) ||
          FAILED(device->CreateQuery(&desc, &slot.parallax_end)) ||
          FAILED(device->CreateQuery(&desc, &slot.parallax_map_start)) ||
          FAILED(device->CreateQuery(&desc, &slot.parallax_limits_start)) ||
          FAILED(device->CreateQuery(&desc, &slot.parallax_subtitle_start)) ||
          FAILED(device->CreateQuery(&desc, &slot.pre_start)) ||
          FAILED(device->CreateQuery(&desc, &slot.pre_end))) {
        BOOST_LOG(warning) << "Depth D3D11 timing unavailable: could not create timestamp queries.";
        return;
      }
    }
    d3d_perf_ready = true;
  }

  void d3d_ring_t::resolve(
    ID3D11DeviceContext *context,
    const std::shared_ptr<host_sbs_telemetry::collector> &telemetry_performance
  ) {
    if (!d3d_perf_ready) {
      return;
    }
    for (auto &slot : d3d_perf_slots) {
      if (!slot.pending) {
        continue;
      }
      D3D11_QUERY_DATA_TIMESTAMP_DISJOINT timing {};
      const auto ready = context->GetData(
        slot.disjoint.Get(),
        &timing,
        sizeof(timing),
        D3D11_ASYNC_GETDATA_DONOTFLUSH
      );
      if (ready == S_FALSE) {
        continue;
      }
      if (FAILED(ready)) {
        slot.pending = false;
        continue;
      }

      UINT64 post_start = 0;
      UINT64 post_end = 0;
      UINT64 parallax_frame_stats_start = 0;
      UINT64 parallax_frame_stats_end = 0;
      UINT64 parallax_start = 0;
      UINT64 parallax_end = 0;
      UINT64 parallax_map_start = 0;
      UINT64 parallax_limits_start = 0;
      UINT64 parallax_subtitle_start = 0;
      UINT64 pre_start = 0;
      UINT64 pre_end = 0;
      constexpr UINT nonblocking = D3D11_ASYNC_GETDATA_DONOTFLUSH;
      const auto post_start_status = context->GetData(
        slot.post_start.Get(), &post_start, sizeof(post_start),
        nonblocking
      );
      const auto post_end_status = context->GetData(
        slot.post_end.Get(), &post_end, sizeof(post_end),
        nonblocking
      );
      const auto pre_start_status = context->GetData(
        slot.pre_start.Get(), &pre_start, sizeof(pre_start),
        nonblocking
      );
      const auto pre_end_status = context->GetData(
        slot.pre_end.Get(), &pre_end, sizeof(pre_end),
        nonblocking
      );
      HRESULT parallax_start_status = S_OK;
      HRESULT parallax_end_status = S_OK;
      HRESULT parallax_frame_stats_start_status = S_OK;
      HRESULT parallax_frame_stats_end_status = S_OK;
      HRESULT parallax_map_start_status = S_OK;
      HRESULT parallax_limits_start_status = S_OK;
      HRESULT parallax_subtitle_start_status = S_OK;
      if (slot.has_parallax_frame_stats) {
        parallax_frame_stats_start_status = context->GetData(
          slot.parallax_frame_stats_start.Get(),
          &parallax_frame_stats_start,
          sizeof(parallax_frame_stats_start),
          nonblocking
        );
        parallax_frame_stats_end_status = context->GetData(
          slot.parallax_frame_stats_end.Get(),
          &parallax_frame_stats_end,
          sizeof(parallax_frame_stats_end),
          nonblocking
        );
      }
      if (slot.has_parallax) {
        parallax_start_status = context->GetData(
          slot.parallax_start.Get(), &parallax_start, sizeof(parallax_start),
          nonblocking
        );
        parallax_end_status = context->GetData(
          slot.parallax_end.Get(), &parallax_end, sizeof(parallax_end),
          nonblocking
        );
      }
      if (slot.has_parallax_map_start) {
        parallax_map_start_status = context->GetData(
          slot.parallax_map_start.Get(),
          &parallax_map_start,
          sizeof(parallax_map_start),
          nonblocking
        );
      }
      if (slot.has_parallax_limits_start) {
        parallax_limits_start_status = context->GetData(
          slot.parallax_limits_start.Get(),
          &parallax_limits_start,
          sizeof(parallax_limits_start),
          nonblocking
        );
      }
      if (slot.has_parallax_subtitle_start) {
        parallax_subtitle_start_status = context->GetData(
          slot.parallax_subtitle_start.Get(),
          &parallax_subtitle_start,
          sizeof(parallax_subtitle_start),
          nonblocking
        );
      }
      const bool any_pending =
        post_start_status == S_FALSE || post_end_status == S_FALSE ||
        pre_start_status == S_FALSE || pre_end_status == S_FALSE ||
        parallax_frame_stats_start_status == S_FALSE ||
        parallax_frame_stats_end_status == S_FALSE ||
        parallax_start_status == S_FALSE || parallax_end_status == S_FALSE ||
        parallax_map_start_status == S_FALSE ||
        parallax_limits_start_status == S_FALSE ||
        parallax_subtitle_start_status == S_FALSE;
      if (any_pending) {
        continue;
      }
      if (post_start_status == S_OK && post_end_status == S_OK &&
          pre_start_status == S_OK && pre_end_status == S_OK &&
          parallax_frame_stats_start_status == S_OK &&
          parallax_frame_stats_end_status == S_OK &&
          parallax_start_status == S_OK && parallax_end_status == S_OK &&
          parallax_map_start_status == S_OK &&
          parallax_limits_start_status == S_OK &&
          parallax_subtitle_start_status == S_OK &&
          !timing.Disjoint && timing.Frequency > 0 && post_end >= post_start &&
          pre_start >= post_end && pre_end >= pre_start &&
          (!slot.has_parallax_frame_stats ||
            (slot.has_parallax && parallax_frame_stats_start >= post_start &&
             parallax_frame_stats_end >= parallax_frame_stats_start &&
             parallax_start >= parallax_frame_stats_end)) &&
          (!slot.has_parallax ||
            (parallax_start >= post_start && parallax_end >= parallax_start &&
             post_end >= parallax_end)) &&
          (!slot.has_parallax_map_start ||
           (slot.has_parallax && parallax_map_start >= parallax_start &&
             parallax_end >= parallax_map_start)) &&
          (!slot.has_parallax_limits_start ||
           (slot.has_parallax && parallax_limits_start >= parallax_start &&
            parallax_end >= parallax_limits_start)) &&
          (!slot.has_parallax_subtitle_start ||
           (slot.has_parallax &&
            parallax_subtitle_start >= parallax_start &&
            parallax_end >= parallax_subtitle_start)) &&
          (!slot.has_parallax_map_start || !slot.has_parallax_limits_start ||
           parallax_limits_start >= parallax_map_start) &&
          (!slot.has_parallax_subtitle_start || !slot.has_parallax_limits_start ||
           parallax_subtitle_start >= parallax_limits_start)) {
        const double to_ms = 1000.0 / static_cast<double>(timing.Frequency);
        if (slot.has_post) {
          if (telemetry_performance) {
            telemetry_performance->record(host_sbs_telemetry::stage::postprocess, static_cast<double>(post_end - post_start) * to_ms, std::chrono::steady_clock::now());
          }
          sbs_perf::add_sample_ms_if_current(
            "depth_postprocess_gpu",
            static_cast<double>(post_end - post_start) * to_ms,
            slot.perf_generation
          );
        }
        if (slot.has_parallax) {
          const UINT64 frame_stats_ticks = slot.has_parallax_frame_stats ?
                                             parallax_frame_stats_end -
                                               parallax_frame_stats_start :
                                             0u;
          sbs_perf::add_sample_ms_if_current(
            "depth_parallax_gpu",
            static_cast<double>(
              frame_stats_ticks + parallax_end - parallax_start
            ) * to_ms,
            slot.perf_generation
          );
        }
        if (slot.has_parallax_map_start) {
          const UINT64 frame_stats_ticks = slot.has_parallax_frame_stats ?
                                             parallax_frame_stats_end -
                                               parallax_frame_stats_start :
                                             0u;
          sbs_perf::add_sample_ms_if_current(
            "depth_parallax_stats_gpu",
            static_cast<double>(
              frame_stats_ticks + parallax_map_start - parallax_start
            ) * to_ms,
            slot.perf_generation
          );
        }
        if (slot.has_parallax_map_start && slot.has_parallax_limits_start) {
          sbs_perf::add_sample_ms_if_current(
            "depth_parallax_map_gpu",
            static_cast<double>(parallax_limits_start - parallax_map_start) * to_ms,
            slot.perf_generation
          );
        }
        if (slot.has_parallax_subtitle_start && slot.has_parallax_limits_start) {
          sbs_perf::add_sample_ms_if_current(
            "depth_parallax_limits_gpu",
            static_cast<double>(parallax_subtitle_start - parallax_limits_start) * to_ms,
            slot.perf_generation
          );
        }
        if (slot.has_parallax_subtitle_start) {
          sbs_perf::add_sample_ms_if_current(
            "depth_parallax_subtitle_gpu",
            static_cast<double>(parallax_end - parallax_subtitle_start) * to_ms,
            slot.perf_generation
          );
        }
        if (slot.has_pre) {
          if (telemetry_performance) {
            telemetry_performance->record(host_sbs_telemetry::stage::preprocess, static_cast<double>(pre_end - pre_start) * to_ms, std::chrono::steady_clock::now());
          }
          sbs_perf::add_sample_ms_if_current(
            "depth_preprocess_gpu",
            static_cast<double>(pre_end - pre_start) * to_ms,
            slot.perf_generation
          );
        }
      }
      slot.pending = false;
    }
  }

  d3d_slot_t *d3d_ring_t::begin(
    ID3D11DeviceContext *context,
    const std::shared_ptr<host_sbs_telemetry::collector> &telemetry_performance,
    bool has_post,
    bool has_pre
  ) {
    resolve(context, telemetry_performance);
    if (!d3d_perf_ready) {
      return nullptr;
    }
    for (std::size_t i = 0; i < d3d_perf_slots.size(); ++i) {
      const std::size_t index = (d3d_perf_next + i) % d3d_perf_slots.size();
      auto &slot = d3d_perf_slots[index];
      if (slot.pending) {
        continue;
      }
      d3d_perf_next = (index + 1) % d3d_perf_slots.size();
      slot.has_post = has_post;
      slot.has_parallax_frame_stats = false;
      slot.has_parallax = false;
      slot.has_parallax_map_start = false;
      slot.has_parallax_limits_start = false;
      slot.has_parallax_subtitle_start = false;
      slot.has_pre = has_pre;
      slot.perf_generation = sbs_perf::generation();
      context->Begin(slot.disjoint.Get());
      context->End(slot.post_start.Get());
      return &slot;
    }
    return nullptr;  // Never stall the encode thread merely to collect telemetry.
  }
}  // namespace models::host_sbs_depth_gpu_timing
