/**
 * @file src/host_sbs_direct_glyph_route.cpp
 * @brief Per-observation dispatch and retirement of the Direct Glyph subtitle model.
 */
#include "host_sbs_direct_glyph_route.h"

#include "host_sbs_direct_glyph_engine.h"
#include "host_sbs_resolution.h"
#include "logging.h"
#include "tensorrt_engine_pool.h"

#include <algorithm>
#include <mutex>
#include <thread>

namespace models::host_sbs_direct_glyph_route {
  namespace {
    [[nodiscard]] display_snapshot_rgb8_analysis_gpu_v1::request_t make_source_request(const frame_t &frame) {
      display_snapshot_rgb8_analysis_gpu_v1::request_t source;
      source.display_snapshot_band = frame.display_snapshot_band;
      // Descriptor-only parent reference; source pixels come from the joined snapshot above.
      source.parent_raw_source = frame.parent_raw_source;
      source.actual_contour_constants96 = frame.contour_constants;
      source.actual_sct32 = frame.contour_identity;
      source.actual_final256 = frame.root_dispatch;
      source.roi_ltrb = {frame.region.left, frame.region.top, frame.region.right, frame.region.bottom};
      source.frame = frame.frame;
      source.observation_us = frame.observation_us;
      source.analysis_generation = frame.region.analysis_generation;
      source.token = frame.token;
      source.transfer_mode = frame.color_space;
      return source;
    }

    [[nodiscard]] host_sbs_direct_glyph_publication::request_t make_publication_request(
      const frame_t &frame, const display_snapshot_rgb8_analysis_gpu_v1::request_t &source, UINT width, UINT crop
    ) {
      host_sbs_direct_glyph_publication::request_t model;
      model.width = width;
      model.height = crop;
      model.frame = source.frame;
      model.observation_us = source.observation_us;
      model.analysis_generation = source.analysis_generation;
      model.token = source.token;
      model.current_constants = source.actual_contour_constants96;
      model.ocr = frame.ocr_box_record;
      model.sct = source.actual_sct32;
      model.condition = frame.condition_params;
      model.root = frame.root;
      model.uip = frame.locator_state;
      // Actual pending function result and work, never guessed from the root.
      model.postprocess_mode = frame.postprocess_mode;
      model.expected_work = frame.expected_work;
      model.use_empty_gate = true;
      return model;
    }
  }  // namespace

  route_t::route_t() = default;

  route_t::~route_t() = default;

  void route_t::report_failure(const std::string &error) {
    if (error_ != error) {
      BOOST_LOG(warning) << "Subtitle glyph protection unavailable for this observation: " << error;
      error_ = error;
    }
  }

  std::optional<bool> route_t::dispatch(ID3D11Device *device, ID3D11DeviceContext *context, const frame_t &frame) {
    namespace glyph = host_sbs_direct_glyph_gpu;
    namespace snapshot = display_snapshot_rgb8_analysis_gpu_v1;
    score_retired_ = false;
    source_audit_.Reset();
    const auto width = frame.region.width();
    const auto height = frame.region.height();
    const auto crop = subtitle_ocr_source_crop_height(width, height);
    if (!crop || crop > height) return std::nullopt;
    const auto source_request = make_source_request(frame);
    auto publication = make_publication_request(frame, source_request, width, crop);
    std::string error;
    if (late_score_) {
      glyph::result_t discarded;
      const auto status = score_.poll(nullptr, discarded, error);
      late_score_ = status == glyph::poll_status_e::pending;
      score_failed_ = status == glyph::poll_status_e::failed;
    }
    const bool can_score = !late_score_ && !score_failed_;
    auto *schedule = publication_.prepare_empty_gate(context, publication, error);
    publication.use_empty_gate = schedule != nullptr;
    const auto &shader = source_shaders_;
    const snapshot::program_t program {shader[0].Get(), shader[1].Get(), shader[2].Get(), shader[3].Get()};
    std::optional<snapshot::direct_audit_submission_t> audit;
    if (can_score && source_request.parent_raw_source && source_request.display_snapshot_band &&
        source_.ensure_direct_audit(device, width, height, height - crop, crop, error)) {
      audit = source_.queue_current_audit(context, program, source_request,
                                          publication.uip, error, schedule);
      if (audit && capture_plan_) {
        source_audit_ = audit->uip_envelope();
      }
    }
    auto *work = publication_.plan_direct(context, publication,
                                          audit ? &*audit : nullptr, error);
    if (!work) {
      report_failure(error);
      return std::nullopt;
    }
    const auto &region = frame.region;
    const auto line_reference = frame.video_caption ?
      subtitle_ocr_line_reference(width, height, region.source_width, region.source_height, region.left, region.top, region.is_video_region()) :
      std::nullopt;
    const glyph::request_t score {
      .source_rgb = audit ? audit->display_snapshot_band() : nullptr, .work = work, .width = width, .band_height = crop,
      .crop_top = height - crop, .token = publication.token,
      .detector_caps = publication_.direct_detector_caps(),
      .video_caption_ocr = line_reference ? publication.ocr : nullptr,
      .line_center_twice_source = line_reference ? static_cast<UINT>(std::max(0, line_reference->center_twice_source)) : 0u,
      .generation = publication.analysis_generation, .frame = publication.frame,
      .subtitle_context_generation = frame.subtitle_context_generation,
      .origin_x = frame.region.left, .origin_y = frame.region.top,
      .color_domain = frame.color_space,
      .reset_history = frame.input_domain_reset,
      .carry_unchanged_lines = frame.carry_unchanged_lines,
    };
    if (can_score) {
      const auto status = score_.try_submit(score, error);
      if (status == glyph::submit_status_e::accepted) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds {250};
        settled_.reset();
        job_.emplace(job_t {publication, score, deadline});
        return std::nullopt;
      }
      score_failed_ = status == glyph::submit_status_e::failed;
    }
    if (!error.empty()) report_failure(error);
    return publication_.publish_direct(context, publication, nullptr);
  }

  bool route_t::work_score(const std::optional<std::chrono::steady_clock::time_point> wait_until) {
    namespace glyph = host_sbs_direct_glyph_gpu;
    if (settled_) return false;
    auto &job = *job_;
    settled_t polled;
    polled.status = score_.poll(&job.score, polled.result, polled.error);
    bool waited = false;
    if (wait_until && polled.status == glyph::poll_status_e::pending) {
      // A score takes two GPU steps or more, each queued by the poll that finds the one before
      // it finished. Within the caller's budget the steps follow one another here, at the pace
      // of the GPU, and not one to a conversion. Past that budget the observation stays
      // reserved and a later call goes on from the step it has reached.
      const auto limit = std::min(*wait_until, job.deadline);
      while (polled.status == glyph::poll_status_e::pending && std::chrono::steady_clock::now() < limit) {
        waited = true;
        score_.wait_until(limit);
        polled.status = score_.poll(&job.score, polled.result, polled.error);
        if (polled.status == glyph::poll_status_e::pending) {
          // A finished step whose readback is not yet mapped has no fence left to wait for.
          std::this_thread::yield();
        }
      }
    }
    // The transport hands a finished score out once. Keep it for the call that publishes it.
    if (polled.status != glyph::poll_status_e::pending) settled_.emplace(std::move(polled));
    return waited;
  }

  bool route_t::advance(const std::chrono::steady_clock::time_point wait_until) {
    if (!job_) return true;
    (void) work_score(wait_until);
    return settled_.has_value();
  }

  retirement_t route_t::retire(ID3D11DeviceContext *context, const std::optional<std::chrono::steady_clock::time_point> wait_until) {
    if (!job_) return {};
    namespace glyph = host_sbs_direct_glyph_gpu;
    auto &job = *job_;
    const bool waited = work_score(wait_until);
    std::string error;
    glyph::result_t result;
    auto status = glyph::poll_status_e::pending;
    if (settled_) {
      status = settled_->status;
      result = std::move(settled_->result);
      error = std::move(settled_->error);
      settled_.reset();
    }
    if (status == glyph::poll_status_e::pending) {
      if (std::chrono::steady_clock::now() < job.deadline) return {.retired = false, .waited = waited};
      // Publish the observation as unavailable while the late score keeps its private
      // buffers until a subsequent poll can discard it without copying Work.
      late_score_ = true;
      error = "Direct glyph scoring did not retire before its deadline";
    }
    score_failed_ = status == glyph::poll_status_e::failed;
    score_retired_ = status == glyph::poll_status_e::ready;
    const bool ready = status == glyph::poll_status_e::ready;
    if (ready) error_.clear();
    else if (!error.empty()) report_failure(error);
    const bool selected_matte_commands = publication_.publish_direct(
      context, job.publication, ready ? result.alpha8.Get() : nullptr);
    score_.retire();
    job_.reset();
    return {.retired = true, .waited = waited, .selected_matte_commands = selected_matte_commands};
  }

  void route_t::register_interop_early() noexcept {
    score_.register_interop_early();
  }

  void route_t::release() noexcept {
    if (!execution_) return;
    settled_.reset();
    host_sbs_direct_glyph_gpu::result_t discarded;
    std::string error;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds {250};
    // Retirement may run alongside flat rendering. A null request only checks the
    // device fence (or the CUDA event) and discards the finished submission.
    while (score_.poll(nullptr, discarded, error) ==
             host_sbs_direct_glyph_gpu::poll_status_e::pending &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds {1});
    }
    const bool released = score_.release();
    std::lock_guard lock(tensorrt_engine_pool::g_trt_mutex);
    tensorrt_engine_pool::recycle_or_quarantine_execution_context_locked(
      engine_key_, execution_, true, !released);
    engine_key_.clear();
  }

  bool route_t::prepare(
    ID3D11Device *device,
    ID3D11DeviceContext *context,
    const subtitle_direct_glyph_configuration_t &cfg,
    const std::filesystem::path &assets_dir,
    CUdevice cuda_device,
    CUcontext cuda_ctx,
    std::string &error
  ) {
    if (cfg.shader_root.empty()) {
      error = "Direct glyph shader root missing";
      return false;
    }
    namespace audit = host_sbs_shader_cache;
    const std::filesystem::path model_shaders(cfg.shader_root);
    const auto sources = audit::snapshot_sources(model_shaders / audit::direct_glyph_source_audit_root, audit::direct_glyph_source_audit_specs);
    if (!sources || audit::source_closure_sha256(sources) != audit::direct_glyph_source_audit_source_closure_sha256) {
      error = "Direct glyph source audit closure mismatch";
      return false;
    }
    for (std::size_t i = 0; i < source_shaders_.size(); ++i) {
      const auto bytecode = audit::get(sources, audit::direct_glyph_source_audit_specs[i]);
      if (!bytecode || FAILED(device->CreateComputeShader(bytecode->data(), bytecode->size(), nullptr, &source_shaders_[i]))) {
        error = "Direct glyph source audit shader creation failed";
        return false;
      }
    }
    if (!publication_.load_direct(device, model_shaders, error)) return false;
    auto &cuda = cuda_driver_api::get();
    nvinfer1::ICudaEngine *engine = nullptr;  // The global engine cache owns its lifetime.
    if (!acquire_direct_glyph_context(assets_dir, cuda, cuda_device, engine,
                                      execution_, engine_key_)) {
      release();
      error = "Direct glyph TensorRT asset or execution context unavailable";
      return false;
    }
    if (!score_.prepare(device, context, model_shaders, cuda_ctx,
                        execution_, error)) {
      release();
      return false;
    }
    enabled_ = true;
    capture_plan_ = cfg.capture_plan;
    return true;
  }

  void route_t::fill_diagnostic_views(subtitle_direct_glyph_diagnostic_views_t &views) const noexcept {
    const auto direct = publication_.views();
    views.alpha8 = direct.alpha8;
    views.work = direct.work;
    views.source_audit = source_audit_;
    if (capture_plan_ && score_retired_) {
      views.native_lines = score_.native_lines_for_evaluation();
    }
  }
}  // namespace models::host_sbs_direct_glyph_route
