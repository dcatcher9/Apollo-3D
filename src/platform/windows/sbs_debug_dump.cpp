/**
 * @file src/platform/windows/sbs_debug_dump.cpp
 * @brief Dump 3D request triggers, same-frame capture submission and readback polling.
 */
#include "sbs_debug_dump.h"

// standard includes
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

// platform includes
#include <windows.h>
#include <wrl/client.h>

// local includes
#include "src/generated/sbs_adaptive_state_contract.h"
#include "src/depth_coordinate_v2.h"
#include "src/host_sbs_subtitle_plane.h"
#include "src/host_sbs_subtitle_glyph.h"
#include "src/host_sbs_gpu_trace.h"
#include "src/host_sbs_shader_cache.h"
#include "src/logging.h"
#include "src/video_depth_estimator.h"
#include "sbs_debug_dump_async.h"
#include "sbs_debug_dump_package.h"
#include "sbs_debug_dump_staging.h"
#include "sbs_debug_dump_validate.h"

namespace platf::sbs_debug {

  using namespace std::literals;

  namespace {

    using namespace models::depth_coordinate_v2;

    constexpr auto retry_backoff = std::chrono::seconds(1);

    using detail::actual_draw_artifacts;
    using detail::actual_draw_is_texture;
    using detail::actual_draw_storage_index;
    using detail::actual_draw_texture_count;
    using detail::calibrated_dav2_shape_for_dump;
    using detail::collect_buffer;
    using detail::collect_float_buffer;
    using detail::collect_status_e;
    using detail::collect_texture;
    using detail::collection_budget;
    using detail::decode_normalization_state;
    using detail::depth_dumpability;
    using detail::depth_input_region_error;
    using detail::dump_publish_result;
    using detail::gpu_trace_shader_identity_matches_contract;
    using detail::normalize_contour_history_for_publication;
    using detail::parallax_v2_shader_identity_matches_contract;
    using detail::publish_captured_dump;
    using detail::shadow_state_is_dumpable;
    using detail::stage_buffer;
    using detail::stage_contour_texture;
    using detail::stage_exact_constants;
    using detail::stage_exact_scalar_buffer;
    using detail::stage_texture;

  }  // namespace

  namespace detail {

    struct diagnostic_roi_crop {
      Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
      Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> srv;
    };

    enum class collection_stage_e : std::uint8_t {
      shadow_state,
      shadow_frame_stats,
      subtitle_ocr_record,
      subtitle_locator_state,
      contour_support,
      subtitle_glyph_matte,
      subtitle_condition_params,
      contour_field_pin,
      contour_distance,
      contour_identity,
      contour_history,
      contour_source,
      ocr_probability,
      gaussian_weight,
      gaussian_blend,
      gaussian_vertical_conditioned,
      gaussian_control,
      gpu_trace_ring,
      validate_gpu_trace,
      validate_evidence,
      scene_normalization,
      scene_adaptive,
      scene_decode,
      model_input,
      raw_depth,
      shadow_coordinate,
      shadow_candidate,
      shadow_vertical,
      shadow_vertical_conditioned,
      shadow_base_final,
      shadow_final,
      source,
      depth_input_source,
      sbs,
      warp_map,
      warp_mask,
      actual_draw,
      complete,
    };

    /**
     * One exact-frame D3D11 staging batch. Every copy and the terminal event query are submitted
     * together on the owning immediate context before any live texture can be reused.
     */
    struct pending_gpu_capture {
      captured_dump_job job;
      Microsoft::WRL::ComPtr<ID3D11Query> completion;

      staged_texture source;
      staged_texture depth_input_source;
      staged_buffer model_input;
      staged_buffer raw_depth;
      staged_texture sbs;
      staged_texture shadow_coordinate;
      staged_texture shadow_candidate;
      staged_texture shadow_vertical;
      staged_texture shadow_vertical_conditioned;
      staged_texture shadow_base_final;
      staged_texture shadow_final;
      staged_buffer subtitle_ocr_record;
      staged_buffer subtitle_locator_state;
      staged_texture contour_support;
      staged_texture subtitle_glyph_matte;
      staged_buffer subtitle_condition_params;
      staged_texture contour_field_pin;
      staged_texture contour_distance;
      staged_texture gaussian_weight;
      staged_texture gaussian_blend;
      staged_texture gaussian_vertical_conditioned;
      staged_texture contour_history;
      staged_texture contour_source;
      staged_buffer contour_identity;
      staged_buffer ocr_probability;
      staged_buffer gaussian_control;
      staged_buffer gpu_trace_ring;
      staged_buffer shadow_state;
      staged_buffer shadow_frame_stats;
      staged_texture warp_map;
      staged_texture warp_mask;
      staged_buffer depth_frame_state;
      staged_buffer adaptive_state;
      std::array<staged_texture, actual_draw_texture_count> actual_draw_textures;
      std::array<staged_buffer, actual_draw_buffer_count> actual_draw_buffers;
      std::size_t actual_draw_collection_index = 0u;

      std::vector<std::uint8_t> normalization_state_bytes;
      collection_stage_e collection_stage = collection_stage_e::shadow_state;

      bool subtitle_plane_active = false;
      bool gpu_trace_requested = false;
      bool depth_input_source_available = false;
      bool depth_health_state_requested = false;
      bool gpu_ready = false;
      std::uint64_t retry_token = 0;
      std::chrono::steady_clock::time_point submitted_at {};
      std::chrono::steady_clock::time_point collection_started_at {};
      double gpu_ready_age_ms = 0.0;
      double cpu_collection_ms = 0.0;
      std::uint32_t collection_poll_count = 0;
    };

  }  // namespace detail

  dumper::dumper():
      async_(detail::publication_state::create()),
      diagnostic_roi_crop_(std::make_unique<detail::diagnostic_roi_crop>()) {
    if (const char *override_dir = std::getenv("APOLLO_SBS_DUMP"); override_dir && *override_dir) {
      dir_ = override_dir;
      file_trigger_enabled_ = config::sunshine.diagnostics_enabled;
    } else if (!config::sunshine.log_file.empty()) {
      dir_ =
        std::filesystem::path(config::sunshine.log_file).parent_path() /
        "sbs_dump";
    } else {
      dir_ = "sbs_dump";
    }
  }

  dumper::~dumper() {
    // Invalidate retry callbacks first, then release the session handle. Accepted publication
    // owns only CPU snapshots plus shared state and finishes on the process-lifetime queue.
    cancel_pending_request();
    async_.reset();
  }

  void dumper::set_button_request(std::shared_ptr<std::atomic<bool>> request) {
    button_request_ = std::move(request);
    if (async_) {
      async_->allow_retries_and_token();
    }
  }

  void dumper::cancel_pending_request() noexcept {
    if (async_) {
      async_->cancel_retries(button_request_);
    }
    const bool remove_file_trigger =
      file_trigger_enabled_ || file_trigger_pending_;
    snapshot_armed_for_dump_ = false;
    prepared_frame_id_ = 0;
    pending_gpu_capture_.reset();
    release_diagnostic_roi_crop();
    retry_not_before_ = {};
    if (auto *button = button_request_.get()) {
      button->store(false, std::memory_order_relaxed);
    }
    file_trigger_pending_ = false;
    file_trigger_enabled_ = false;
    if (!remove_file_trigger) {
      return;
    }
    try {
      std::error_code error;
      std::filesystem::remove(dir_ / "dump.trigger", error);
    } catch (...) {
      // Permanent estimator failure already makes this request impossible. Cleanup is best-effort
      // and must never turn the encode-thread failure handling into another exception path.
    }
  }

  void dumper::reject_pending_request() noexcept {
    snapshot_armed_for_dump_ = false;
    prepared_frame_id_ = 0;
    pending_gpu_capture_.reset();
    release_diagnostic_roi_crop();
    retry_not_before_ = {};
    if (auto *button = button_request_.get()) {
      button->store(false, std::memory_order_relaxed);
    }
    if (file_trigger_pending_) {
      try {
        std::error_code error;
        std::filesystem::remove(dir_ / "dump.trigger", error);
      } catch (...) {
      }
    }
    file_trigger_pending_ = false;
  }

  bool dumper::snapshot_requested() {
    snapshot_armed_for_dump_ = false;
    prepared_frame_id_ = 0;
    release_diagnostic_roi_crop();
    if (async_ && async_->take_file_retry_pending()) {
      file_trigger_pending_ = true;
    }
    if (async_ && async_->take_trigger_remove_failed()) {
      file_trigger_enabled_ = false;
    }
    if (async_ && async_->take_publication_failed()) {
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
    }
    // The button latch remains armed while publication is active, so a later click is retained
    // for the next frame instead of replacing or aliasing the in-flight package.
    if (pending_gpu_capture_ || (async_ && async_->busy())) {
      return false;
    }
    if (std::chrono::steady_clock::now() < retry_not_before_) {
      return false;
    }
    const auto *button = button_request_.get();
    if (button && button->load(std::memory_order_relaxed)) {
      if (async_) {
        async_->allow_retries_and_token();
      }
      snapshot_armed_for_dump_ = true;
      return true;
    }
    if (!file_trigger_enabled_) {
      return false;
    }
    if (file_trigger_pending_) {
      if (async_) {
        async_->allow_retries_and_token();
      }
      snapshot_armed_for_dump_ = true;
      return true;
    }
    (void) needs_conversion_poll();
    snapshot_armed_for_dump_ = file_trigger_pending_;
    if (file_trigger_pending_ && async_) {
      async_->allow_retries_and_token();
    }
    return file_trigger_pending_;
  }

  bool dumper::needs_conversion_poll() const noexcept {
    if (pending_gpu_capture_) {
      return true;
    }
    if (async_ && async_->busy()) {
      return false;
    }
    if (std::chrono::steady_clock::now() < retry_not_before_) {
      return false;
    }
    const auto *button = button_request_.get();
    if (button && button->load(std::memory_order_acquire)) {
      return true;
    }
    if (!file_trigger_enabled_) {
      return false;
    }
    if (file_trigger_pending_) {
      return true;
    }

    const auto now = std::chrono::steady_clock::now();
    if (now < next_file_trigger_poll_) {
      return false;
    }
    next_file_trigger_poll_ = now + std::chrono::seconds(1);
    try {
      std::error_code error;
      file_trigger_pending_ =
        std::filesystem::exists(dir_ / "dump.trigger", error) && !error;
    } catch (...) {
      file_trigger_pending_ = false;
    }
    return file_trigger_pending_;
  }

  bool dumper::prepare_requested_v2_frame(
    const std::uint64_t matched_frame_id
  ) noexcept {
    auto *button = button_request_.get();
    const bool requested =
      (button && button->load(std::memory_order_relaxed)) ||
      file_trigger_pending_;
    if (!requested || !snapshot_armed_for_dump_ || pending_gpu_capture_ ||
        matched_frame_id == 0) {
      return false;
    }
    // Do not read shadow_state here. The subsequent staging batch captures that buffer and the
    // packed SBS in one command-stream transaction, then validates the camera from those exact
    // bytes once the terminal event query reports completion.
    prepared_frame_id_ = matched_frame_id;
    return true;
  }

  ID3D11ShaderResourceView *dumper::prepare_diagnostic_roi_crop(
    ID3D11Device *device,
    ID3D11DeviceContext *ctx,
    ID3D11Texture2D *source,
    const models::depth_input_region_t &region
  ) noexcept {
    release_diagnostic_roi_crop();
    if (!diagnostic_roi_crop_ || !device || !ctx || !source || !region.valid() ||
        !region.is_video_region()) {
      return nullptr;
    }

    D3D11_TEXTURE2D_DESC source_desc {};
    source->GetDesc(&source_desc);
    if (source_desc.Width != region.source_width ||
        source_desc.Height != region.source_height || source_desc.MipLevels != 1u ||
        source_desc.ArraySize != 1u || source_desc.SampleDesc.Count != 1u ||
        region.right > source_desc.Width || region.bottom > source_desc.Height) {
      return nullptr;
    }

    auto crop_desc = source_desc;
    crop_desc.Width = region.width();
    crop_desc.Height = region.height();
    crop_desc.MipLevels = 1u;
    crop_desc.ArraySize = 1u;
    crop_desc.Usage = D3D11_USAGE_DEFAULT;
    crop_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    crop_desc.CPUAccessFlags = 0u;
    crop_desc.MiscFlags = 0u;

    Microsoft::WRL::ComPtr<ID3D11Texture2D> crop_texture;
    auto status = device->CreateTexture2D(&crop_desc, nullptr, &crop_texture);
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> crop_srv;
    if (SUCCEEDED(status)) {
      status = device->CreateShaderResourceView(crop_texture.Get(), nullptr, &crop_srv);
    }
    if (FAILED(status)) {
      try {
        BOOST_LOG(warning)
          << "Dump 3D could not allocate its diagnostic window-region source crop "sv
          << region.width() << 'x' << region.height() << " [HRESULT "sv
          << static_cast<std::uint32_t>(status)
          << "]; live ROI analysis remains active and this dump request is rejected."sv;
      } catch (...) {
      }
      return nullptr;
    }

    const D3D11_BOX source_box {
      region.left,
      region.top,
      0u,
      region.right,
      region.bottom,
      1u,
    };
    ctx->CopySubresourceRegion(
      crop_texture.Get(),
      0u,
      0u,
      0u,
      0u,
      source,
      0u,
      &source_box
    );
    diagnostic_roi_crop_->texture = std::move(crop_texture);
    diagnostic_roi_crop_->srv = std::move(crop_srv);
    return diagnostic_roi_crop_->srv.Get();
  }

  void dumper::release_diagnostic_roi_crop() noexcept {
    if (!diagnostic_roi_crop_) {
      return;
    }
    diagnostic_roi_crop_->srv.Reset();
    diagnostic_roi_crop_->texture.Reset();
  }

  void dumper::poll_pending_readback(ID3D11DeviceContext *ctx) noexcept {
    if (!pending_gpu_capture_ || !ctx || !async_) {
      return;
    }

    const auto fail_capture = [this](const char *reason, const bool retryable = true) noexcept {
      if (!pending_gpu_capture_) {
        return;
      }
      const auto retry_token = pending_gpu_capture_->retry_token;
      const bool by_button = pending_gpu_capture_->job.by_button;
      const bool by_file = pending_gpu_capture_->job.by_file;
      const auto button = pending_gpu_capture_->job.button_request;
      pending_gpu_capture_.reset();
      prepared_frame_id_ = 0;
      retry_not_before_ = retryable ?
        std::chrono::steady_clock::now() + retry_backoff :
        std::chrono::steady_clock::time_point {};
      const bool rearmed = retryable && async_ && async_->record_publication_failure(
        retry_token, by_button, by_file, button
      );
      if (!retryable && async_) {
        // Invalidate only this capture's retry epoch. A later click may already have re-armed the
        // shared button latch after the original request was consumed and must remain pending.
        async_->cancel_retries(nullptr);
      }
      if (!retryable) {
        snapshot_armed_for_dump_ = false;
        if (by_file) {
          try {
            std::error_code error;
            std::filesystem::remove(dir_ / "dump.trigger", error);
            if (error) {
              file_trigger_enabled_ = false;
            }
          } catch (...) {
            file_trigger_enabled_ = false;
          }
        }
        file_trigger_pending_ = false;
      }
      try {
        BOOST_LOG(warning) << "SBS debug dump staged GPU readback failed; request "sv
                           << (rearmed ? "re-armed" :
                               (retryable ? "discarded after cancellation" :
                                            "discarded as deterministic invalid evidence"))
                           << ": " << reason;
      } catch (...) {
      }
    };

    try {
      auto &pending = *pending_gpu_capture_;
      if (!pending.gpu_ready) {
        if (!pending.completion) {
          fail_capture("completion query is unavailable");
          return;
        }
        const HRESULT query_status = ctx->GetData(
          pending.completion.Get(),
          nullptr,
          0,
          D3D11_ASYNC_GETDATA_DONOTFLUSH
        );
        if (query_status == S_FALSE) {
          return;
        }
        if (query_status != S_OK) {
          fail_capture("completion query failed");
          return;
        }
        const auto ready_observed = std::chrono::steady_clock::now();
        pending.gpu_ready = true;
        pending.gpu_ready_age_ms = std::chrono::duration<double, std::milli>(
          ready_observed - pending.submitted_at
        ).count();
        pending.collection_started_at = ready_observed;
        pending.completion.Reset();
      }

      const auto poll_started = std::chrono::steady_clock::now();
      ++pending.collection_poll_count;
      collection_budget budget(poll_started);
      auto &job = pending.job;
      const auto finish_poll = [&]() noexcept {
        pending.cpu_collection_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - poll_started
        ).count();
      };

      const auto collect_required = [&]
      (
        const collect_status_e status,
        const detail::collection_stage_e next
      ) noexcept {
        if (status == collect_status_e::ready) {
          pending.collection_stage = next;
          return true;
        }
        finish_poll();
        if (status == collect_status_e::failed) {
          fail_capture("required staging resource could not be mapped");
        }
        // A terminal EVENT should normally make every preceding staging resource mappable.
        // WAS_STILL_DRAWING and a deliberately partial chunk both retain the immutable batch.
        return false;
      };

      while (pending_gpu_capture_) {
        if (budget.exhausted()) {
          finish_poll();
          return;
        }

        using stage_e = detail::collection_stage_e;
        switch (pending.collection_stage) {
          case stage_e::shadow_state:
            if (!collect_required(
                  collect_float_buffer(
                    ctx,
                    pending.shadow_state,
                    models::depth_coordinate_v2::state_float_count,
                    job.shadow_state,
                    budget
                  ),
                  stage_e::shadow_frame_stats
                )) return;
            break;

          case stage_e::shadow_frame_stats:
            if (!collect_required(
                  collect_float_buffer(
                    ctx,
                    pending.shadow_frame_stats,
                    models::depth_coordinate_v2::frame_stats_float_count,
                    job.shadow_frame_stats,
                    budget
                  ),
                  stage_e::subtitle_ocr_record
                )) return;
            break;

          case stage_e::subtitle_ocr_record:
            if (!pending.subtitle_plane_active) {
              pending.collection_stage = stage_e::gpu_trace_ring;
              break;
            }
            if (!collect_required(
                  collect_buffer(
                    ctx,
                    pending.subtitle_ocr_record,
                    job.subtitle_ocr_record,
                    budget
                  ),
                  stage_e::subtitle_locator_state
                )) return;
            break;

          case stage_e::subtitle_locator_state:
            if (!collect_required(
                  collect_buffer(
                    ctx,
                    pending.subtitle_locator_state,
                    job.subtitle_locator_state,
                    budget
                  ),
                  stage_e::contour_support
                )) return;
            break;

          case stage_e::contour_support:
            if (!collect_required(
                  collect_texture(ctx, pending.contour_support, job.contour_support, budget),
                  stage_e::subtitle_glyph_matte
                )) return;
            break;

          case stage_e::subtitle_glyph_matte:
            if (!collect_required(
                  collect_texture(ctx, pending.subtitle_glyph_matte, job.subtitle_glyph_matte, budget),
                  stage_e::subtitle_condition_params
                )) return;
            break;

          case stage_e::subtitle_condition_params:
            if (!collect_required(
                  collect_buffer(ctx, pending.subtitle_condition_params, job.subtitle_condition_params, budget),
                  stage_e::contour_field_pin
                )) return;
            break;

          case stage_e::contour_field_pin:
            if (!collect_required(
                  collect_texture(ctx, pending.contour_field_pin, job.contour_field_pin, budget),
                  stage_e::contour_distance
                )) return;
            break;

          case stage_e::contour_distance:
            if (!collect_required(
                  collect_texture(ctx, pending.contour_distance, job.contour_distance, budget),
                  stage_e::contour_identity
                )) return;
            break;

          case stage_e::contour_identity:
            if (!collect_required(
                  collect_buffer(ctx, pending.contour_identity, job.contour_identity, budget),
                  stage_e::contour_history
                )) return;
            break;

          case stage_e::contour_history:
            if (!collect_required(
                  collect_texture(ctx, pending.contour_history, job.contour_history, budget),
                  stage_e::contour_source
                )) return;
            break;

          case stage_e::contour_source:
            if (!collect_required(
                  collect_texture(ctx, pending.contour_source, job.contour_source, budget),
                  stage_e::ocr_probability
                )) return;
            break;

          case stage_e::ocr_probability:
            if (!collect_required(
                  collect_buffer(ctx, pending.ocr_probability, job.ocr_probability, budget),
                  stage_e::gaussian_weight
                )) return;
            break;

          case stage_e::gaussian_weight:
            if (!collect_required(
                  collect_texture(ctx, pending.gaussian_weight, job.gaussian_weight, budget),
                  stage_e::gaussian_blend
                )) return;
            break;

          case stage_e::gaussian_blend:
            if (!collect_required(
                  collect_texture(ctx, pending.gaussian_blend, job.gaussian_blend, budget),
                  stage_e::gaussian_vertical_conditioned
                )) return;
            break;

          case stage_e::gaussian_vertical_conditioned:
            if (!collect_required(
                  collect_texture(ctx, pending.gaussian_vertical_conditioned, job.gaussian_vertical_conditioned, budget),
                  stage_e::gaussian_control
                )) return;
            break;

          case stage_e::gaussian_control:
            if (!collect_required(
                  collect_buffer(ctx, pending.gaussian_control, job.gaussian_control, budget),
                  stage_e::gpu_trace_ring
                )) return;
            break;

          case stage_e::gpu_trace_ring:
            if (!pending.gpu_trace_requested) {
              pending.collection_stage = stage_e::validate_evidence;
              break;
            }
            {
              const auto status = collect_buffer(
                ctx,
                pending.gpu_trace_ring,
                job.gpu_trace_ring,
                budget
              );
              if (status == collect_status_e::ready) {
                pending.collection_stage = stage_e::validate_gpu_trace;
              } else if (status == collect_status_e::failed) {
                pending.gpu_trace_requested = false;
                pending.gpu_trace_ring = {};
                job.gpu_trace_ring.clear();
                pending.collection_stage = stage_e::validate_evidence;
              } else {
                finish_poll();
                return;
              }
            }
            break;

          case stage_e::validate_gpu_trace:
            if (!detail::gpu_trace_ring_is_canonical(
                  job.gpu_trace_ring,
                  job.completed
                )) {
              BOOST_LOG(warning)
                << "SBS debug dump: optional GPU completion trace failed canonical "sv
                   "validation and was omitted; the core dump remains complete."sv;
              pending.gpu_trace_requested = false;
              job.gpu_trace_ring.clear();
            }
            pending.collection_stage = stage_e::validate_evidence;
            break;

          case stage_e::validate_evidence:
            if (!shadow_state_is_dumpable(
                  job.shadow_state,
                  job.completed.parallax_v2_raw_coordinate_scale
                )) {
              finish_poll();
              fail_capture(
                "selected completion is not a current valid V2 camera/output pair"
              );
              return;
            }
            if (pending.subtitle_plane_active && !detail::subtitle_records_match_completion(
                  job.subtitle_ocr_record,
                  job.subtitle_locator_state,
                  job.gpu_trace_ring,
                  job.completed
                )) {
              finish_poll();
              fail_capture("OCR8/UIP4 record identity or layout is invalid", false);
              return;
            }
            if (pending.subtitle_plane_active && !detail::subtitle_contour_identity_matches_completion(
                  job.contour_identity, job.subtitle_ocr_record, job.subtitle_locator_state, job.shadow_state,
                  job.completed)) {
              finish_poll();
              fail_capture("SCT5 contour identity does not match the joined completed tuple", false);
              return;
            }
            pending.collection_stage = stage_e::scene_normalization;
            break;

          case stage_e::scene_normalization:
            if (!pending.depth_health_state_requested) {
              pending.collection_stage = stage_e::model_input;
              break;
            }
            {
              const auto status = collect_buffer(
                ctx,
                pending.depth_frame_state,
                pending.normalization_state_bytes,
                budget
              );
              if (status == collect_status_e::ready) {
                pending.collection_stage = stage_e::scene_adaptive;
              } else if (status == collect_status_e::failed) {
                pending.depth_health_state_requested = false;
                pending.depth_frame_state.resource.Reset();
                pending.adaptive_state.resource.Reset();
                pending.normalization_state_bytes.clear();
                job.adaptive_state.clear();
                pending.collection_stage = stage_e::model_input;
              } else {
                finish_poll();
                return;
              }
            }
            break;

          case stage_e::scene_adaptive:
            {
              const auto status = collect_buffer(
                ctx,
                pending.adaptive_state,
                job.adaptive_state,
                budget
              );
              if (status == collect_status_e::ready) {
                pending.collection_stage = stage_e::scene_decode;
              } else if (status == collect_status_e::failed) {
                pending.depth_health_state_requested = false;
                pending.adaptive_state.resource.Reset();
                pending.normalization_state_bytes.clear();
                job.adaptive_state.clear();
                pending.collection_stage = stage_e::model_input;
              } else {
                finish_poll();
                return;
              }
            }
            break;

          case stage_e::scene_decode:
            // The retained depth-health state is optional comparison evidence. Malformed bytes do
            // not reject the exact-frame core package.
            job.depth_health_state_available =
              decode_normalization_state(
                pending.normalization_state_bytes, job.normalization
              ) == depth_dumpability::valid;
            pending.normalization_state_bytes.clear();
            if (!job.depth_health_state_available) {
              job.adaptive_state.clear();
            }
            pending.collection_stage = stage_e::model_input;
            break;

          case stage_e::model_input:
            if (!collect_required(
                  collect_float_buffer(
                    ctx,
                    pending.model_input,
                    static_cast<std::size_t>(job.completed.model_width) *
                      static_cast<std::size_t>(job.completed.model_height) * 3u,
                    job.model_input,
                    budget
                  ),
                  stage_e::raw_depth
                )) return;
            break;

          case stage_e::raw_depth:
            if (!collect_required(
                  collect_float_buffer(
                    ctx,
                    pending.raw_depth,
                    static_cast<std::size_t>(job.completed.raw_width) *
                      static_cast<std::size_t>(job.completed.raw_height),
                    job.raw_depth,
                    budget
                  ),
                  stage_e::shadow_coordinate
                )) return;
            break;

          case stage_e::shadow_coordinate:
            if (!collect_required(
                  collect_texture(
                    ctx, pending.shadow_coordinate, job.shadow_coordinate, budget
                  ),
                  stage_e::shadow_candidate
                )) return;
            break;

          case stage_e::shadow_candidate:
            if (!collect_required(
                  collect_texture(
                    ctx, pending.shadow_candidate, job.shadow_candidate, budget
                  ),
                  stage_e::shadow_vertical
                )) return;
            break;

          case stage_e::shadow_vertical:
            if (!collect_required(
                  collect_texture(
                    ctx, pending.shadow_vertical, job.shadow_vertical, budget
                  ),
                  stage_e::shadow_vertical_conditioned
                )) return;
            break;

          case stage_e::shadow_vertical_conditioned:
            if (!collect_required(
                  collect_texture(
                    ctx,
                    pending.shadow_vertical_conditioned,
                    job.shadow_vertical_conditioned,
                    budget
                  ),
                  stage_e::shadow_base_final
                )) return;
            break;

          case stage_e::shadow_base_final:
            if (!pending.subtitle_plane_active) {
              pending.collection_stage = stage_e::shadow_final;
              break;
            }
            if (!collect_required(
                  collect_texture(
                    ctx, pending.shadow_base_final, job.shadow_base_final, budget
                  ),
                  stage_e::shadow_final
                )) return;
            break;

          case stage_e::shadow_final:
            if (!collect_required(
                  collect_texture(ctx, pending.shadow_final, job.shadow_final, budget),
                  stage_e::source
                )) return;
            break;

          case stage_e::source:
            if (!collect_required(
                  collect_texture(ctx, pending.source, job.source, budget),
                  stage_e::depth_input_source
                )) return;
            break;

          case stage_e::depth_input_source:
            if (!pending.depth_input_source_available) {
              pending.collection_stage = stage_e::sbs;
              break;
            }
            if (!collect_required(
                  collect_texture(
                    ctx, pending.depth_input_source, job.depth_input_source, budget
                  ),
                  stage_e::sbs
                )) return;
            break;

          case stage_e::sbs:
            if (!collect_required(
                  collect_texture(ctx, pending.sbs, job.sbs, budget),
                  stage_e::warp_map
                )) return;
            break;

          case stage_e::warp_map:
            if (!job.warp_map_available) {
              pending.collection_stage = stage_e::warp_mask;
              break;
            }
            if (!collect_required(
                  collect_texture(ctx, pending.warp_map, job.warp_map, budget),
                  stage_e::warp_mask
                )) return;
            break;

          case stage_e::warp_mask:
            if (!job.warp_mask_available) {
              pending.collection_stage = stage_e::actual_draw;
              break;
            }
            if (!collect_required(
                  collect_texture(ctx, pending.warp_mask, job.warp_mask, budget),
                  stage_e::actual_draw
                )) return;
            break;

          case stage_e::actual_draw: {
            const auto i = pending.actual_draw_collection_index;
            if (i == actual_draw_artifacts.size()) {
              pending.collection_stage = stage_e::complete;
              break;
            }
            if (job.actual_draw_available[i]) {
              const auto storage = actual_draw_storage_index(i);
              const auto status = actual_draw_is_texture(i) ?
                collect_texture(ctx, pending.actual_draw_textures[storage], job.actual_draw_textures[storage], budget) :
                collect_buffer(ctx, pending.actual_draw_buffers[storage], job.actual_draw_buffers[storage], budget);
              if (status == collect_status_e::partial || status == collect_status_e::not_ready) {
                finish_poll();
                return;
              }
              if (status == collect_status_e::failed) {
                job.actual_draw_available[i] = false;
                job.actual_draw_unavailable[i] = "optional staging resource could not be mapped";
                if (actual_draw_is_texture(i)) {
                  job.actual_draw_textures[storage] = {};
                  pending.actual_draw_textures[storage] = {};
                } else {
                  job.actual_draw_buffers[storage].clear();
                  pending.actual_draw_buffers[storage] = {};
                }
              }
            }
            ++pending.actual_draw_collection_index;
            break;
          }

          case stage_e::complete:
            finish_poll();
            {
              const auto queued_frame_id = job.completed.matched_frame_id;
              const auto retry_token = pending.retry_token;
              const bool by_button = job.by_button;
              const bool by_file = job.by_file;
              const auto button = job.button_request;
              const double gpu_ready_age_ms = pending.gpu_ready_age_ms;
              const double collection_ms = pending.cpu_collection_ms;
              const auto collection_poll_count = pending.collection_poll_count;
              const double collection_wall_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - pending.collection_started_at
              ).count();
              auto cpu_job = std::move(job);
              pending_gpu_capture_.reset();

              const auto worker_state = async_;
              bool queued = false;
              try {
                std::function<void()> publish_task =
                  [job = std::move(cpu_job), worker_state, retry_token]() mutable {
                  dump_publish_result result;
                  try {
                    if (!normalize_contour_history_for_publication(job)) {
                      result.error = "invalid native contour-history storage";
                    } else {
                      result = publish_captured_dump(job);
                    }
                  } catch (const std::exception &exception) {
                    result.error = exception.what();
                  } catch (...) {
                    result.error = "unknown background publication exception";
                  }
                  if (!result.success) {
                    const bool rearmed = worker_state->record_publication_failure(
                      retry_token,
                      job.by_button,
                      job.by_file,
                      job.button_request
                    );
                    try {
                      BOOST_LOG(warning)
                        << "SBS debug dump background publication failed; request "
                        << (rearmed ? "re-armed" : "discarded after cancellation")
                        << (result.error.empty() ? "." : ": " + result.error);
                    } catch (...) {
                    }
                  }
                  if (result.trigger_remove_failed) {
                    worker_state->record_trigger_remove_failure();
                  }
                };
                queued = worker_state->enqueue(std::move(publish_task));
              } catch (...) {
                // The exact-frame CPU snapshot is already complete, but allocating the queued
                // task itself can still fail. Preserve the request through the common retry path.
                queued = false;
              }
              if (!queued) {
                retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
                worker_state->record_publication_failure(
                  retry_token,
                  by_button,
                  by_file,
                  button
                );
                return;
              }

              try {
                BOOST_LOG(info)
                  << "SBS debug dump stable GPU snapshot queued for background publication (frame "sv
                  << queued_frame_id << ", GPU-ready age " << gpu_ready_age_ms
                  << " ms, CPU collection " << collection_ms << " ms across "
                  << collection_poll_count << " polls, ready-to-CPU age "
                  << collection_wall_ms << " ms)."sv;
              } catch (...) {
              }
            }
            return;
        }
      }
    } catch (const std::exception &exception) {
      try {
        BOOST_LOG(warning) << "SBS debug dump staging collection threw: "
                           << exception.what();
      } catch (...) {
      }
      fail_capture("CPU snapshot allocation or collection failed");
    } catch (...) {
      fail_capture("CPU snapshot collection failed with an unknown exception");
    }
  }

  bool dumper::maybe_dump(
    ID3D11Device *device,
    ID3D11DeviceContext *ctx,
    const frame &completed,
    const config::video_t::sbs_t &cfg
  ) {
    const bool by_file = file_trigger_pending_;
    if (!snapshot_armed_for_dump_ || !async_ || async_->busy() ||
        pending_gpu_capture_) {
      return false;
    }
    snapshot_armed_for_dump_ = false;
    detail::button_request_guard button_request(button_request_);
    if (!button_request.consumed() && !by_file) {
      return false;
    }
    try {
      const auto submission_started = std::chrono::steady_clock::now();
      const std::uint64_t retry_token = async_->allow_retries_and_token();
      if (
        !device || !ctx || !completed.source || !completed.depth_input_source ||
        !completed.model_input ||
        !completed.raw_depth || !completed.warp_depth || !completed.sbs ||
        completed.model_width <= 0 || completed.model_height <= 0 ||
        completed.raw_width <= 0 || completed.raw_height <= 0 ||
        completed.field_width <= 0 || completed.field_height <= 0 ||
        !completed.field_content.valid(
          {completed.field_width, completed.field_height}
        ) ||
        !completed.raw_model_provenance ||
        completed.raw_model_provenance->depth_model.empty() ||
        completed.raw_model_provenance->depth_model != completed.depth_model ||
        completed.raw_model_provenance->onnx_sha256.empty() ||
        completed.raw_model_provenance->preprocess_source_closure_sha256.empty() ||
        completed.raw_width != completed.model_width ||
        completed.raw_height != completed.model_height ||
        prepared_frame_id_ != completed.matched_frame_id ||
        completed.matched_observation_timestamp_us == 0u ||
        completed.warp_depth != (completed.shadow_base_final_parallax ?
          completed.shadow_base_final_parallax : completed.shadow_final_parallax)
      ) {
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        return false;
      }
      // The fused runtime exposes one common capture/model/depth/geometry raster: exactly the
      // frozen 2x realization of the source-selected DAV2 shape. A stale split-grid frame or an
      // arbitrary high-resolution claim fails closed instead of being mislabeled in Dump 3D.
      const auto calibrated_dav2_shape = calibrated_dav2_shape_for_dump(completed);
      if (!calibrated_dav2_shape ||
          !detail::capture_tensor_grid_is_authenticated(completed) ||
          !detail::composite_runtime_provenance_is_authenticated(completed)) {
        BOOST_LOG(warning)
          << "SBS debug dump: completed model/depth geometry does not match the authenticated "sv
             "schema-2 single-high fused runtime contract; dump rejected."sv;
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        return false;
      }
      if (completed.field_width != completed.raw_width ||
          completed.field_height != completed.raw_height ||
          completed.field_content != completed.depth_input_region.tensor_content) {
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        return false;
      }
      const auto input_region_error = depth_input_region_error(completed);
      if (!input_region_error.empty()) {
        BOOST_LOG(warning) << "SBS debug dump: authenticated input-region metadata is "sv
                              "incomplete or inconsistent; dump rejected: "sv
                           << input_region_error << '.';
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        return false;
      }
      if (!completed.parallax_v2_render_selected ||
          !completed.parallax_v2_producer_active ||
          !completed.shadow_candidate_parallax ||
          !completed.shadow_vertical_majorant || !completed.shadow_vertical_conditioned ||
          !completed.shadow_final_parallax || !completed.shadow_state ||
          !completed.shadow_frame_stats) {
        BOOST_LOG(warning)
          << "SBS debug dump: production V2 renderer is not selected or has an incomplete "sv
             "authenticated resource set; dump rejected (legacy live rendering is unsupported)."sv;
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        return false;
      }
      const bool subtitle_ocr_present = completed.ocr_box_record != nullptr;
      const bool subtitle_locator_present = completed.subtitle_locator_state != nullptr;
      const bool subtitle_plane_active =
        !completed.subtitle_work_suppressed &&
        subtitle_ocr_present && subtitle_locator_present;
      if (!detail::subtitle_capture_resources_are_authenticated(completed)) {
        BOOST_LOG(warning)
          << "SBS debug dump: OCR8/UIP4/contour resources are partial; dump rejected."sv;
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        return false;
      }
      if (!completed.shadow_coordinate) {
        BOOST_LOG(warning)
          << "SBS debug dump: the explicit Dump 3D canonical-coordinate snapshot is "sv
             "unavailable; live V2 rendering remains authenticated and unaffected."sv;
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        return false;
      }
      if (completed.parallax_v2_live_renderer_source_closure_sha256 !=
            models::host_sbs_shader_cache::
              parallax_v2_live_renderer_source_closure_sha256) {
        BOOST_LOG(warning)
          << "SBS debug dump: production V2 renderer source closure is missing or "sv
             "mismatched; dump rejected."sv;
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        return false;
      }

      const auto &model_identity = *completed.raw_model_provenance;
      const auto *capture_calibration =
        models::depth_coordinate_v2::find_capture_calibration(
          model_identity.depth_model,
          model_identity.depth_model_url,
          model_identity.onnx_sha256,
          model_identity.preprocess_profile,
          model_identity.preprocess_source_closure_sha256,
          static_cast<std::uint32_t>(calibrated_dav2_shape->width),
          static_cast<std::uint32_t>(calibrated_dav2_shape->height)
        );
      if (!capture_calibration) {
        BOOST_LOG(warning)
          << "SBS debug dump: production V2 resources do not resolve to exactly one "sv
             "authenticated model/preprocess/shape calibration; dump rejected."sv;
        // Back off like every sibling rejection: the request latch is restored on return, so
        // without it this warning and the snapshot arming would repeat on every frame.
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        return false;
      }
      if (!parallax_v2_shader_identity_matches_contract(
            completed.parallax_v2_shader_provenance
          )) {
        BOOST_LOG(warning)
          << "SBS debug dump: production V2 resources have missing or mismatched "sv
             "shader-source provenance; dump rejected."sv;
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        return false;
      }

      const std::uint64_t model_pixels =
        static_cast<std::uint64_t>(completed.model_width) *
        static_cast<std::uint64_t>(completed.model_height);
      const std::uint64_t raw_values =
        static_cast<std::uint64_t>(completed.raw_width) *
        static_cast<std::uint64_t>(completed.raw_height);
      if (model_pixels > SIZE_MAX / (3u * sizeof(float)) ||
          raw_values > SIZE_MAX / sizeof(float)) {
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        return false;
      }

      auto pending = std::make_unique<detail::pending_gpu_capture>();
      auto &job = pending->job;
      job.root = dir_;
      job.trigger = dir_ / "dump.trigger";
      job.button_request = button_request_;
      job.by_button = button_request.consumed();
      job.by_file = by_file;
      job.completed = completed;
      job.cfg = cfg;
      job.preprocess = &capture_calibration->preprocess;
      job.warp_map_available = completed.warp_map != nullptr;
      job.warp_mask_available = completed.warp_mask != nullptr;
      pending->subtitle_plane_active = subtitle_plane_active;
      pending->gpu_trace_requested = completed.gpu_trace_ring != nullptr &&
        gpu_trace_shader_identity_matches_contract(completed.gpu_trace_provenance);
      pending->depth_input_source_available =
        completed.depth_input_region.is_video_region();
      pending->retry_token = retry_token;
      if (completed.depth_input_region.is_video_region() &&
          (!job.warp_map_available || !job.warp_mask_available)) {
        BOOST_LOG(warning)
          << "SBS debug dump: ROI completion lacks the required full-source inverse map "sv
             "or mask; dump rejected."sv;
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        return false;
      }

      D3D11_QUERY_DESC query_desc {D3D11_QUERY_EVENT, 0};
      if (FAILED(device->CreateQuery(&query_desc, &pending->completion))) {
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        return false;
      }

      bool staged =
        stage_texture(device, ctx, completed.source, pending->source) &&
        stage_buffer(
          device,
          ctx,
          completed.model_input,
          static_cast<std::size_t>(model_pixels) * 3u * sizeof(float),
          pending->model_input
        ) &&
        stage_buffer(
          device,
          ctx,
          completed.raw_depth,
          static_cast<std::size_t>(raw_values) * sizeof(float),
          pending->raw_depth
        ) &&
        stage_texture(device, ctx, completed.sbs, pending->sbs) &&
        stage_texture(
          device, ctx, completed.shadow_coordinate, pending->shadow_coordinate
        ) &&
        stage_texture(
          device,
          ctx,
          completed.shadow_candidate_parallax,
          pending->shadow_candidate
        ) &&
        stage_texture(
          device,
          ctx,
          completed.shadow_vertical_majorant,
          pending->shadow_vertical
        ) &&
        stage_texture(
          device,
          ctx,
          completed.shadow_vertical_conditioned,
          pending->shadow_vertical_conditioned
        ) &&
        stage_texture(
          device, ctx, completed.shadow_final_parallax, pending->shadow_final
        ) &&
        stage_buffer(
          device,
          ctx,
          completed.shadow_state,
          models::depth_coordinate_v2::state_float_count * sizeof(float),
          pending->shadow_state
        ) &&
        stage_buffer(
          device,
          ctx,
          completed.shadow_frame_stats,
          models::depth_coordinate_v2::frame_stats_float_count * sizeof(float),
          pending->shadow_frame_stats
        );
      if (staged && subtitle_plane_active) {
        staged =
          stage_texture(
            device,
            ctx,
            completed.shadow_base_final_parallax,
            pending->shadow_base_final
          ) &&
          stage_buffer(
            device,
            ctx,
            completed.ocr_box_record,
            subtitle_ocr_record_word_count * sizeof(std::uint32_t),
            pending->subtitle_ocr_record
          ) &&
          stage_buffer(
            device,
            ctx,
            completed.subtitle_locator_state,
            subtitle_plane_state_word_count * sizeof(std::uint32_t),
            pending->subtitle_locator_state
          ) &&
          stage_contour_texture(
            device, ctx, completed.subtitle_contour_support, detail::subtitle_contour_texture_e::support,
            completed, pending->contour_support
          ) &&
          stage_contour_texture(
            device, ctx, completed.subtitle_glyph_matte, detail::subtitle_contour_texture_e::glyph_matte,
            completed, pending->subtitle_glyph_matte
          ) &&
          stage_exact_scalar_buffer(
            device, ctx, completed.subtitle_condition_params, models::depth_coordinate_v2::subtitle_condition_param_word_count, pending->subtitle_condition_params
          ) &&
          stage_contour_texture(
            device, ctx, completed.subtitle_contour_field_pin, detail::subtitle_contour_texture_e::field_pin,
            completed, pending->contour_field_pin
          ) &&
          stage_contour_texture(
            device, ctx, completed.subtitle_contour_distance, detail::subtitle_contour_texture_e::distance,
            completed, pending->contour_distance
          ) &&
          stage_exact_scalar_buffer(
            device, ctx, completed.subtitle_contour_identity, subtitle_contour_history_word_count, pending->contour_identity
          ) &&
          stage_contour_texture(
            device, ctx, completed.subtitle_contour_history, detail::subtitle_contour_texture_e::history,
            completed, pending->contour_history
          ) &&
          stage_contour_texture(
            device, ctx, completed.subtitle_contour_source, detail::subtitle_contour_texture_e::source,
            completed, pending->contour_source
          ) &&
          stage_exact_scalar_buffer(
            device, ctx, completed.subtitle_ocr_probability, subtitle_ocr_output_width * subtitle_ocr_output_height, pending->ocr_probability
          ) &&
          stage_contour_texture(
            device, ctx, completed.subtitle_gaussian_weight, detail::subtitle_contour_texture_e::gaussian_weight,
            completed, pending->gaussian_weight
          ) &&
          stage_contour_texture(
            device, ctx, completed.subtitle_gaussian_blend, detail::subtitle_contour_texture_e::gaussian_blend,
            completed, pending->gaussian_blend
          ) &&
          stage_contour_texture(
            device, ctx, completed.subtitle_gaussian_vertical_conditioned, detail::subtitle_contour_texture_e::gaussian_vertical_conditioned,
            completed, pending->gaussian_vertical_conditioned
          ) &&
          stage_exact_scalar_buffer(
            device, ctx, completed.subtitle_gaussian_control, 1u, pending->gaussian_control
          );
      }
      if (staged && pending->gpu_trace_requested) {
        pending->gpu_trace_requested = stage_buffer(
          device,
          ctx,
          completed.gpu_trace_ring,
          models::host_sbs_gpu_trace::ring_byte_count,
          pending->gpu_trace_ring
        );
        if (!pending->gpu_trace_requested) {
          pending->gpu_trace_ring = {};
        }
      }
      if (staged && pending->depth_input_source_available) {
        staged = stage_texture(
          device,
          ctx,
          completed.depth_input_source,
          pending->depth_input_source
        );
      }
      if (staged && job.warp_map_available) {
        staged = stage_texture(device, ctx, completed.warp_map, pending->warp_map);
      }
      if (staged && job.warp_mask_available) {
        staged = stage_texture(device, ctx, completed.warp_mask, pending->warp_mask);
      }
      if (!staged) {
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        BOOST_LOG(warning)
          << "SBS debug dump: GPU staging submission failed; request retained for retry."sv;
        return false;
      }

      if (completed.adaptive_state && completed.depth_frame_state) {
        pending->depth_health_state_requested =
          stage_buffer(
            device,
            ctx,
            completed.depth_frame_state,
            4u * sizeof(float),
            pending->depth_frame_state
          ) &&
          stage_buffer(
            device,
            ctx,
            completed.adaptive_state,
            sbs_adaptive_state::word_count * sizeof(std::uint32_t),
            pending->adaptive_state
          );
        if (!pending->depth_health_state_requested) {
          pending->depth_frame_state = {};
          pending->adaptive_state = {};
        }
      }

      if (completed.actual_draw.requested) {
        const auto &e = completed.actual_draw;
        const std::array<ID3D11ShaderResourceView *, actual_draw_texture_count> textures {
          e.matte.Get(),
          e.shared_pin_weights.Get(), e.shared_pin_field.Get(), e.shared_pin_rows.Get(),
          e.shared_pin_mapping.Get(), e.shared_pin_boundary.Get()
        };
        const std::array<std::size_t, actual_draw_texture_count> texture_artifacts {0u, 4u, 5u, 6u, 8u, 9u};
        for (std::size_t i = 0u; i < textures.size(); ++i) {
          const auto artifact = texture_artifacts[i];
          job.actual_draw_available[artifact] = e.captured && textures[i] &&
            stage_texture(device, ctx, textures[i], pending->actual_draw_textures[i]);
          if (!job.actual_draw_available[artifact]) job.actual_draw_unavailable[artifact] = !e.captured ? e.unavailable_reason :
            !textures[i] ? "not bound by observed draw" : "optional texture staging failed";
        }
        job.actual_draw_available[1u] = e.captured && e.selected_receipt &&
          stage_exact_scalar_buffer(device, ctx, e.selected_receipt.Get(), 64u, pending->actual_draw_buffers[0u]);
        if (!job.actual_draw_available[1u]) job.actual_draw_unavailable[1u] = !e.captured ? e.unavailable_reason :
          !e.selected_receipt ? "not bound by observed draw" : "optional structured-buffer layout/staging failed";
        const std::array<ID3D11Buffer *, 3u> constants {e.geometry_constants.Get(), e.subtitle_constants.Get(), e.shared_pin_geometry.Get()};
        const std::array<UINT, 3u> constant_bytes {48u, 32u, 16u};
        const std::array<std::size_t, 3u> constant_artifacts {2u, 3u, 7u};
        for (std::size_t i = 0u; i < constants.size(); ++i) {
          const auto artifact = constant_artifacts[i];
          job.actual_draw_available[artifact] = e.captured && constants[i] &&
            stage_exact_constants(device, ctx, constants[i], constant_bytes[i], pending->actual_draw_buffers[1u + i]);
          if (!job.actual_draw_available[artifact]) job.actual_draw_unavailable[artifact] = !e.captured ? e.unavailable_reason :
            !constants[i] ? "not bound/issued for observed draw" : "optional constant-buffer layout/staging failed";
        }
      }

      // No later frame can race these bytes: every CopyResource precedes this event, and all
      // future live writes are submitted after it on the same immediate-context command stream.
      ctx->End(pending->completion.Get());
      pending->submitted_at = std::chrono::steady_clock::now();

      // No background code may observe a live COM pointer. The pending GPU object owns only
      // staging resources; the eventual publication job owns only copied scalar metadata.
      job.completed.source = nullptr;
      job.completed.actual_draw.release_resources();
      job.completed.depth_input_source = nullptr;
      job.completed.model_input = nullptr;
      job.completed.raw_depth = nullptr;
      job.completed.warp_depth = nullptr;
      job.completed.adaptive_state = nullptr;
      job.completed.depth_frame_state = nullptr;
      job.completed.warp_map = nullptr;
      job.completed.warp_mask = nullptr;
      job.completed.sbs = nullptr;
      job.completed.shadow_coordinate = nullptr;
      job.completed.shadow_candidate_parallax = nullptr;
      job.completed.shadow_vertical_majorant = nullptr;
      job.completed.shadow_vertical_conditioned = nullptr;
      job.completed.shadow_base_final_parallax = nullptr;
      job.completed.shadow_final_parallax = nullptr;
      job.completed.ocr_box_record = nullptr;
      job.completed.subtitle_locator_state = nullptr;
      job.completed.subtitle_contour_support = nullptr;
      job.completed.subtitle_glyph_matte = nullptr;
      job.completed.subtitle_condition_params = nullptr;
      job.completed.subtitle_contour_field_pin = nullptr;
      job.completed.subtitle_contour_distance = nullptr;
      job.completed.subtitle_contour_identity = nullptr;
      job.completed.subtitle_contour_history = nullptr;
      job.completed.subtitle_contour_source = nullptr;
      job.completed.subtitle_ocr_probability = nullptr;
      job.completed.subtitle_gaussian_weight = nullptr;
      job.completed.subtitle_gaussian_blend = nullptr;
      job.completed.subtitle_gaussian_vertical_conditioned = nullptr;
      job.completed.subtitle_gaussian_control = nullptr;
      job.completed.gpu_trace_ring = nullptr;
      job.completed.shadow_state = nullptr;
      job.completed.shadow_frame_stats = nullptr;

      const auto submitted_frame_id = job.completed.matched_frame_id;
      pending_gpu_capture_ = std::move(pending);
      button_request.commit();
      if (by_file) {
        file_trigger_pending_ = false;
      }
      prepared_frame_id_ = 0;
      try {
        const double submission_ms = std::chrono::duration<double, std::milli>(
          pending_gpu_capture_->submitted_at - submission_started
        ).count();
        BOOST_LOG(info)
          << "SBS debug dump exact-frame GPU staging submitted without waiting (frame "sv
          << submitted_frame_id << ", CPU allocation/submission " << submission_ms
          << " ms)."sv;
      } catch (...) {
      }
      return true;
    } catch (const std::exception &exception) {
      retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
      try {
        BOOST_LOG(warning)
          << "SBS debug dump render-thread staging submission failed; request retained: "
          << exception.what();
      } catch (...) {
      }
      return false;
    } catch (...) {
      retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
      try {
        BOOST_LOG(warning)
          << "SBS debug dump render-thread staging submission failed with an unknown "sv
             "exception; request retained."sv;
      } catch (...) {
      }
      return false;
    }
  }

}  // namespace platf::sbs_debug
