/**
 * @file src/platform/windows/display_vram.cpp
 * @brief Definitions for handling video ram.
 */
// standard includes
#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <future>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string_view>
#include <thread>

// platform includes
#include <d3d11_3.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <dwmapi.h>

// lib includes
#include <boost/algorithm/string/predicate.hpp>

// local includes
#include "display.h"
#include "display_config.h"
#include "display_shader_manifest.h"
#include "foreground_window_region.h"
#include "subtitle_context_policy.h"
#include "game_export_pipeline.h"
#include "game3d_setup.h"
#include "game3d_text_hook.h"
#include "tools/reshade/game3d_ui_adaptive.h"
#include "host_sbs_v2_renderer.h"
#include "misc.h"
#include "local_rgb_converter.h"
#include "packed_output_fit.h"
#include "reshade_bridge.h"
#include "sbs_debug_dump.h"
#include "text_game_pin_controller.h"
#include "utils.h"
#include "src/capture_image_pool.h"
#include "src/config.h"
#include "src/depth_coordinate_v2.h"
#include "src/generated/sbs_adaptive_state_contract.h"
#include "src/host_sbs_drawn_source_order.h"
#include "src/host_sbs_observation_handoff.h"
#include "src/host_sbs_packed_presentation_clock.h"
#include "host_sbs_converter_test_observer.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <cstring>
#include <stdexcept>
#include "src/host_sbs_provider.h"
#include "src/host_sbs_completed_subtitle_programs.h"
#include "src/host_sbs_shared_ui_warp_runtime.h"
#include "src/host_sbs_resolution.h"
#include "src/host_sbs_shader_cache.h"
#include "src/host_sbs_subtitle_render.h"
#include "src/host_sbs_v2_geometry.h"
#include "src/host_subtitle_context.h"
#include "src/live_model_preparation.h"
#include "src/logging.h"
#include "src/model_manager.h"
#include "src/nvenc/nvenc_config.h"
#include "src/nvenc/nvenc_d3d11_native.h"
#include "src/nvenc/nvenc_utils.h"
#include "src/sbs_perf.h"
#include "src/system_tray.h"
#include "src/video.h"
#include "src/video_depth_estimator.h"
#include "video_dom_client.h"
#include "host3d_subtitle_apps.h"
#include "video_player_context.h"

#ifdef SUNSHINE_TESTS
  #include "src/capture_display_retirement.h"
  #include "src/tracked_async_worker.h"
  #include "model_preparation_converter_test.h"
  #include "conversion_poll_test.h"
#endif

#if !defined(SUNSHINE_SHADERS_DIR)  // for testing this needs to be defined in cmake as we don't do an install
  #define SUNSHINE_SHADERS_DIR SUNSHINE_ASSETS_DIR "/shaders/directx"
#endif
namespace platf {
  using namespace std::literals;
}

namespace platf::dxgi {

  using d3d_query_t = util::safe_ptr<ID3D11Query, Release<ID3D11Query>>;

  [[nodiscard]] std::uint64_t host_sbs_observation_timestamp_us(
    const std::chrono::steady_clock::time_point observed_at
  ) noexcept {
    if (observed_at == std::chrono::steady_clock::time_point {}) {
      return 0u;
    }
    const auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(
                                observed_at.time_since_epoch()
    )
                                .count();
    if (microseconds < 0 || static_cast<std::uint64_t>(microseconds) == std::numeric_limits<std::uint64_t>::max()) {
      return 0u;
    }
    // Zero remains the cross-ABI invalid sentinel. The matched capture observation, rather than
    // a later admission/submission clock, owns this timestamp.
    return static_cast<std::uint64_t>(microseconds) + 1u;
  }

  struct alignas(16) sdr_color_transform_t {
    std::uint32_t target_bt2020;
    std::uint32_t source_is_hdr;
    float source_sdr_white_scrgb;
    std::uint32_t target_is_hdr;
  };

  struct alignas(16) stream_gamma_transform_t {
    std::uint32_t mode;
    float white_scrgb;
    float padding[2] {};
  };

  template<class T>
  buf_t make_buffer(device_t::pointer device, const T &t) {
    static_assert(sizeof(T) % 16 == 0, "Buffer needs to be aligned on a 16-byte alignment");

    D3D11_BUFFER_DESC buffer_desc {
      sizeof(T),
      D3D11_USAGE_IMMUTABLE,
      D3D11_BIND_CONSTANT_BUFFER
    };

    D3D11_SUBRESOURCE_DATA init_data {
      &t
    };

    buf_t::pointer buf_p;
    auto status = device->CreateBuffer(&buffer_desc, &init_data, &buf_p);
    if (status) {
      BOOST_LOG(error) << "Failed to create buffer: [0x"sv << util::hex(status).to_string_view() << ']';
      return nullptr;
    }

    return buf_t {buf_p};
  }

  blend_t make_blend(device_t::pointer device, bool enable, bool invert) {
    D3D11_BLEND_DESC bdesc {};
    auto &rt = bdesc.RenderTarget[0];
    rt.BlendEnable = enable;
    rt.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

    if (enable) {
      rt.BlendOp = D3D11_BLEND_OP_ADD;
      rt.BlendOpAlpha = D3D11_BLEND_OP_ADD;

      if (invert) {
        // Invert colors
        rt.SrcBlend = D3D11_BLEND_INV_DEST_COLOR;
        rt.DestBlend = D3D11_BLEND_INV_SRC_COLOR;
      } else {
        // Regular alpha blending
        rt.SrcBlend = D3D11_BLEND_SRC_ALPHA;
        rt.DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
      }

      rt.SrcBlendAlpha = D3D11_BLEND_ZERO;
      rt.DestBlendAlpha = D3D11_BLEND_ZERO;
    }

    blend_t blend;
    auto status = device->CreateBlendState(&bdesc, &blend);
    if (status) {
      BOOST_LOG(error) << "Failed to create blend state: [0x"sv << util::hex(status).to_string_view() << ']';
      return nullptr;
    }

    return blend;
  }

#define DISPLAY_SHADER_BLOB(stem, stage) blob_t stem##_hlsl;
  WINDOWS_DISPLAY_SHADERS(DISPLAY_SHADER_BLOB)
#undef DISPLAY_SHADER_BLOB

  // Authenticated production Host SBS V2 shaders. The renderer closure covers the warp pixel
  // shader plus the shared fullscreen vertex shader; the independent fallback closure covers the
  // flat-identity pixel shader plus the same vertex shader. Dump-only entry points are
  // snapshotted and compiled lazily by the per-device dumper, outside startup and production
  // frame timing.
  models::host_sbs_shader_cache::bytecode_t sbs_reprojection_v2_live_ps_hlsl;
  std::string sbs_reprojection_v2_live_source_closure_sha256;
  models::host_sbs_shader_cache::bytecode_t sbs_reprojection_v2_p010_y_ps_hlsl;
  std::string sbs_reprojection_v2_p010_y_source_closure_sha256;
  models::host_sbs_shader_cache::bytecode_t sbs_flat_identity_ps_hlsl;
  models::host_sbs_shader_cache::bytecode_t sbs_reprojection_vs_hlsl;

  /** Resolve the model-grid realization of one display-selected analysis region.
   *
   * The display selects source pixels and semantic authority before the estimator has accepted a
   * model transaction, so matched slots intentionally retain the calibrated DAV2 content raster.
   * The single-high fused estimator returns the same source/authority tuple with that content
   * raster doubled exactly.  Keep that raster conversion explicit: source position, authority and
   * analysis generation are never inferred from the model output.
   */
  [[nodiscard]] static std::optional<models::depth_input_region_t>
    realize_depth_input_region(
      const models::depth_input_region_t &submitted_region,
      const int field_width,
      const int field_height
    ) noexcept {
    if (!submitted_region.valid() || field_width <= 0 || field_height <= 0) {
      return std::nullopt;
    }

    const auto calibrated_shape = models::fit_host_sbs_v2_depth_tensor_shape(
      submitted_region.source_width,
      submitted_region.source_height
    );
    if (!models::host_sbs_v2_depth_shape_is_authenticated(calibrated_shape)) {
      return std::nullopt;
    }

    const auto realized_shape = models::host_sbs_convex2x_field_shape(calibrated_shape);
    const auto high_content = models::host_sbs_convex2x_content_rect(
      submitted_region.tensor_content,
      calibrated_shape
    );
    if (!realized_shape || !high_content) {
      return std::nullopt;
    }
    auto realized = submitted_region;
    realized.tensor_content = *high_content;

    if (field_width != realized_shape->width || field_height != realized_shape->height || !realized.tensor_content.valid(*realized_shape) || !realized.valid()) {
      return std::nullopt;
    }
    return realized;
  }

  /** Exact completion-to-color binding, including the estimator-owned model-grid raster. */
  [[nodiscard]] static bool depth_completion_region_matches_submission(
    const models::depth_observation_views_t &estimate,
    const models::depth_input_region_t &submitted_region
  ) noexcept {
    if (!estimate.input_region.valid() || !estimate.composite_depth_runtime_provenance || estimate.raw_width != estimate.field_width || estimate.raw_height != estimate.field_height) {
      return false;
    }
    const auto expected = realize_depth_input_region(
      submitted_region,
      estimate.field_width,
      estimate.field_height
    );
    return expected && estimate.input_region == *expected;
  }

  /** Stable analysis-domain comparison across coarse submission and fused high-grid publication. */
  [[nodiscard]] static bool depth_analysis_domain_matches_realization(
    const models::depth_input_region_t &submitted_region,
    const models::depth_input_region_t &realized_region,
    const int field_width,
    const int field_height
  ) noexcept {
    const auto expected = realize_depth_input_region(
      submitted_region,
      field_width,
      field_height
    );
    return expected && expected->same_analysis_domain(realized_region);
  }

  struct img_d3d_t: public platf::img_t {
    // These objects are owned by the display_t's ID3D11Device
    texture2d_t capture_texture;
    render_target_t capture_rt;
    keyed_mutex_t capture_mutex;

    // Shared handle used by an encode device to open capture_texture.
    HANDLE encoder_texture_handle = {};
    // Changes when this pooled image replaces its shared resource.
    std::uint64_t capture_texture_generation = 0;

    // Set to true if the image corresponds to a dummy texture used prior to
    // the first successful capture of a desktop frame
    bool dummy = false;

    // Set to true if the image is blank (contains no content at all, including a cursor)
    bool blank = true;

    // Unique identifier for this image
    uint32_t id = 0;

    // DXGI format of this image texture
    DXGI_FORMAT format;

    // DDup-only identity and exact bounded damage history for the committed desktop surface.
    // Cursor-only outputs retain their base surface's snapshot. WGC and dummy images leave this
    // disengaged so consumers fail open.
    std::optional<detail::ddup_damage_snapshot_t> ddup_damage;

    // DDup capture owns this pixel history for the lifetime of the texture. A metadata-only
    // delivery can change the published cursor without touching the baked cursor below.
    std::uint64_t ddup_desktop_serial = 0;
    D3D11_BOX ddup_cursor_box {};

    // The desktop capture already contains this cursor. External SBS bypasses those pixels,
    // so retain its immutable CPU shape and position for composition on the encoder device.
    sbs_cursor::snapshot_t cursor;

    virtual ~img_d3d_t() override {
      if (encoder_texture_handle) {
        CloseHandle(encoder_texture_handle);
      }
    };
  };

  blob_t compile_shader(LPCSTR file, LPCSTR entrypoint, LPCSTR shader_model) {
    blob_t::pointer msg_p = nullptr;
    blob_t::pointer compiled_p;

    DWORD flags = D3DCOMPILE_ENABLE_STRICTNESS;

#ifndef NDEBUG
    flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
    // Keep production shader compilation aligned with the SBS evaluator and the depth-compute
    // path. The D3DCompile default is only optimization level 1, which leaves avoidable work in
    // the full-resolution reprojection loop.
    flags |= D3DCOMPILE_OPTIMIZATION_LEVEL3;
#endif

    auto wFile = from_utf8(file);
    auto status = D3DCompileFromFile(wFile.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, entrypoint, shader_model, flags, 0, &compiled_p, &msg_p);

    if (msg_p) {
      BOOST_LOG(warning) << std::string_view {(const char *) msg_p->GetBufferPointer(), msg_p->GetBufferSize() - 1};
      msg_p->Release();
    }

    if (status) {
      BOOST_LOG(error) << "Couldn't compile ["sv << file << "] [0x"sv << util::hex(status).to_string_view() << ']';
      return nullptr;
    }

    return blob_t {compiled_p};
  }

  blob_t compile_pixel_shader(LPCSTR file) {
    return compile_shader(file, "main_ps", "ps_5_0");
  }

  blob_t compile_vertex_shader(LPCSTR file) {
    return compile_shader(file, "main_vs", "vs_5_0");
  }

  struct selected_matte_programs_t {
    models::host_sbs_shared_ui_warp::programs_t shared_warp;
    models::subtitle_direct_glyph_configuration_t configuration;
    std::string error;
    bool ready = false;
  };

  // Device-object preparation only, before the estimator's first input. The
  // shared subtitle warp is the only renderer of the Direct Glyph matte.
  // Historical selector receipt ABIs do not authorize this renderer.
  selected_matte_programs_t prepare_selected_matte_programs(
    models::video_depth_estimator &estimator,
    ID3D11Device *device,
    const models::subtitle_direct_glyph_configuration_t &configuration
  ) {
    selected_matte_programs_t result;
    result.configuration = configuration;
    result.ready = models::host_sbs_completed_subtitle_programs::prepare(
      device, estimator, configuration, result.shared_warp, result.error);
    return result;
  }

  struct depth_pipeline_build_result_t {
    std::unique_ptr<models::video_depth_estimator> estimator;
    selected_matte_programs_t selected_matte;
  };

  using depth_estimator_build_task_t =
    std::packaged_task<depth_pipeline_build_result_t()>;

  // Keep abandoned per-device estimator builds/results alive without blocking a presentation
  // session teardown. The shared process-lifetime worker service joins them together with async
  // encoder teardown before logging/TensorRT/D3D globals are destroyed.
  void launch_depth_estimator_build(
    depth_estimator_build_task_t task,
    std::function<void()> on_complete = {}
  ) {
    std::packaged_task<void()> worker(
      [task = std::move(task), on_complete = std::move(on_complete)]() mutable {
        task();
        // packaged_task makes its future ready before operator() returns. Signal the encoder
        // only after that point so it can consume the result without a readiness race.
        if (on_complete) {
          on_complete();
        }
      }
    );
    ::video::launch_async_teardown_worker(std::move(worker));
  }

  // Retire everything a device's depth pipeline owned once nothing draws with the device. CUDA and
  // TensorRT teardown can wait for device quiescence and return a large execution context to the
  // pool, and a build future owns its estimator result, so this never runs on the encode or
  // local-presenter thread. One worker releases them all and then clears and flushes the device's
  // immediate context: D3D11 destroys a device only after its bound state is cleared and its
  // deferred destruction flushed, and without that every Host 3D session kept its device, VRAM
  // and driver threads.
  void retire_depth_pipeline(
    std::vector<std::unique_ptr<models::video_depth_estimator>> estimators,
    std::future<depth_pipeline_build_result_t> build,
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context
  ) {
    std::packaged_task<void()> worker([estimators = std::move(estimators), build = std::move(build), context = std::move(context)]() mutable {
      estimators.clear();
      if (build.valid()) {
        try {
          auto pipeline = build.get();
          pipeline.estimator.reset();
          // The prepared device programs are destroyed with `pipeline`, before the flush.
        } catch (const std::exception &e) {
          BOOST_LOG(warning) << "Retired Host SBS pipeline build failed: "sv << e.what();
        } catch (...) {
          BOOST_LOG(warning) << "Retired Host SBS pipeline build failed with an unknown exception."sv;
        }
      }
      if (context) {
        context->ClearState();
        context->Flush();
      }
    });
    ::video::launch_async_teardown_worker(std::move(worker));
  }

#ifdef SUNSHINE_TESTS
  namespace {
    struct model_preparation_watch_probe_t {
      std::uint64_t calls = 0;
      RECT last_rect {};
      std::weak_ptr<void> token;
    };
    thread_local model_preparation_watch_probe_t *active_model_preparation_watch_probe = nullptr;

    std::shared_ptr<void> model_preparation_watch_for_test(RECT rect) {
      if (!active_model_preparation_watch_probe) {
        return platf::game3d::watch_game_setup(rect);
      }
      auto &probe = *active_model_preparation_watch_probe;
      ++probe.calls;
      probe.last_rect = rect;
      auto token = std::make_shared<unsigned char>(0u);
      probe.token = token;
      return token;
    }
  }  // namespace
#endif
#ifdef SUNSHINE_TESTS
  namespace {
    struct conversion_poll_model_read_t {
      const std::function<models::engine_build_status()> &read;
      std::uint32_t calls = 0;
    };

    thread_local conversion_poll_model_read_t *active_conversion_poll_model_read = nullptr;

    class conversion_poll_model_read_scope_t {
    public:
      explicit conversion_poll_model_read_scope_t(conversion_poll_model_read_t &read):
          previous_(std::exchange(active_conversion_poll_model_read, &read)) {}

      ~conversion_poll_model_read_scope_t() {
        active_conversion_poll_model_read = previous_;
      }

    private:
      conversion_poll_model_read_t *previous_;
    };
  }  // namespace
#endif
  class d3d_base_encode_device final {
  public:
    // Declared first, so destroyed last: the destructor hands the depth pipeline over here, and it
    // goes to its retirement worker only after every other member has released its D3D11
    // objects, so that worker's final flush follows every release on this device.
    struct pipeline_retirement_t {
      std::vector<std::unique_ptr<models::video_depth_estimator>> estimators;
      std::future<depth_pipeline_build_result_t> build;
      Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;

      ~pipeline_retirement_t() {
        // A device that never built a depth pipeline leaves nothing bound that keeps it alive.
        if (!estimators.empty() || build.valid()) {
          retire_depth_pipeline(std::move(estimators), std::move(build), std::move(context));
        }
      }
    } pipeline_retirement;

    ~d3d_base_encode_device() {
      claim_capture_pixels(false);
      capture_damage_observer.reset();
      // The encode session is already leaving the live path, so a short bounded drain cannot
      // stall capture. It preserves the final GPU timing samples; the generation token only
      // rejects results that cross an explicit collector reset (used by the offline harness).
      drain_sbs_gpu_timers();

      // The background task owns D3D references and its packaged-task future does not block here.
      // If this device is torn down while construction is in flight, the process-lifetime tracked
      // worker service lets it finish independently and destroys the unused result there.
      pipeline_retirement.estimators = std::move(failed_depth_estimators);
      if (depth_estimator) {
        pipeline_retirement.estimators.push_back(std::move(depth_estimator));
      }
      pipeline_retirement.build = std::move(depth_estimator_build);
      pipeline_retirement.context = device_ctx.get();
    }

    void release_capture_display() noexcept {
      // Capture must copy the desktop again for whoever converts next.
      claim_capture_pixels(false);
      capture_damage_observer.reset();
      // Encoder teardown may remain blocked inside the NVIDIA driver. The shared capture-display
      // owner is not a D3D/NVENC operand, so release it before derived member destruction to let
      // Desktop Duplication reinitialize while device/context resources remain alive.
      game_export.setup_watch.reset();
      display.reset();
    }

    // Installed only by the streaming encode loop, which converts when woken, on new captures and
    // at its minimum-FPS keepalive. A live export then drives conversion from its ready fence and
    // capture skips the desktop pixels it never reads. Local AR never installs it.
    void set_external_frame_wake(std::function<void()> wake) {
      external_frame_wake = std::move(wake);
      if (!external_frame_wake) {
        // The streaming loop is leaving: capture must copy the desktop again for whoever
        // converts next, including a replacement encoder on this display.
        claim_capture_pixels(false);
      }
      if (game_export.receiver && ::video::is_packed_mode(sbs_mode)) {
        game_export.receiver->set_frame_wake(external_frame_wake);
      }
    }

    bool conversion_kept_input() const {
      return conversion_missing_source;
    }

    // The streaming loop converts a held live export, whose next frame it re-checks for.
    bool external_frame_held() const {
      return game_export.receiver && ::video::is_packed_mode(sbs_mode) && external_frame_wake && game_export.receiver->frame_held();
    }

    // A text game in front is converted from its captured picture by the Host AI path.
    bool capture_paced() const {
      return text_game_ai_active;
    }

    void encoder_consumed_input() {
      // A replaced export slot returns to the producer when the GPU completes the conversion's
      // reads of it (the receiver's read-fence wait), which with pictures in flight can come after
      // the next claim. This is the owner's fallback check, should that wait not be armed.
      if (game_export.receiver) {
        game_export.receiver->retire();
        if (diagnostics_enabled && ::video::is_packed_mode(sbs_mode)) {
          game_export.log_wakes();
        }
      }
    }

    // Tells capture whether this encoder currently reads its desktop pixels.
    void claim_capture_pixels(bool owned) noexcept {
      if (!display) {
        return;
      }
      auto &owner = display->external_pixels_owner;
      if (owned) {
        if (!capture_pixels_token) {
          static std::atomic<std::uint64_t> next_token {1};
          capture_pixels_token = next_token.fetch_add(1, std::memory_order_relaxed);
        }
        if (owner.load(std::memory_order_relaxed) != capture_pixels_token) {
          owner.store(capture_pixels_token, std::memory_order_release);
        }
      } else if (capture_pixels_token) {
        auto expected = capture_pixels_token;
        owner.compare_exchange_strong(expected, 0, std::memory_order_acq_rel);
      }
    }

    void observe_capture_damage() noexcept {
      if (auto *capture = dynamic_cast<display_ddup_vram_t *>(display.get())) {
        capture->add_damage_observer();
        capture_damage_observer.reset(capture);
      }
    }

    using rgb_conversion_e = local_rgb_converter_t::conversion_e;
    using rgb_destination_e = local_rgb_converter_t::destination_e;

    struct rgb_output_t {
      ID3D11Texture2D *texture;
      ID3D11RenderTargetView *target;
      bool linear;
      rgb_destination_e destination;
      bool retained_pair = false;
    };

    // Only a shutter sink with a published pair from this converter can retain RGB output.
    rgb_conversion_e convert_rgb(
      platf::img_t &img,
      ID3D11Texture2D *target_texture,
      ID3D11RenderTargetView *target,
      bool target_is_linear,
      const std::optional<std::chrono::steady_clock::time_point> presentation_target,
      rgb_destination_e destination = rgb_destination_e::backbuffer
    ) {
      rgb_output_t output {target_texture, target, target_is_linear, destination};
      if (convert(img, presentation_target, &output) != 0) {
        return rgb_conversion_e::failed;
      }
      return output.retained_pair ? rgb_conversion_e::retained : rgb_conversion_e::converted;
    }

    bool needs_conversion_poll() const {
      if (stream_gamma_conversion_pending) {
        return true;
      }
      const bool white_query_due = stream_gamma_mode_ != ::video::stream_gamma_mode_e::windows_default && std::chrono::steady_clock::now() >= stream_gamma_white_query_at;
      // ReShade publishes independently of desktop presents. Local AR polls it at its own cadence.
      // A streaming encoder converts a live export when its ready fence wakes it (or at stream
      // cadence while that wake is not armed), and otherwise only for a changed connection:
      // without a live export, captures and the minimum-FPS keepalive refresh the flat fallback,
      // never a repeat of a static desktop at stream cadence. Each of those conversions also runs a
      // due white-level query, so the query never forces a repeat of its own.
      if (game_export.receiver && ::video::is_packed_mode(sbs_mode)) {
        if (!external_frame_wake || game_export.dumper.needs_conversion_poll()) {
          return true;
        }
        if ((game_export.receiver->frame_held() && !game_export.receiver->frame_wake_active()) || game_export.receiver->frame_pending()) {
          return true;
        }
        // A text game is converted by the Host AI path, whose completed depth must be drawn even
        // while the game shows a still picture. Without such a game in front this stays false,
        // so the flat fallback is never repeated at stream cadence.
        // Its picture comes from capture, and a still one would otherwise never be converted
        // again: a few times a second the conversion looks again at which window is in front
        // and at the owner's consent, and reports what protects the text.
        return text_game_ai_active && (depth_work_needs_service() || std::chrono::steady_clock::now() - text_game_serviced_at >= std::chrono::milliseconds(250));
      }
      if (white_query_due) {
        return true;
      }
      if (::video::is_game_mode(sbs_mode)) {
        // Game mono encodes the desktop and only reports whether an export is available. The
        // streaming loop refreshes that on captures and keepalives; Local AR keeps polling.
        return !external_frame_wake || game_export.dumper.needs_conversion_poll();
      }
      return depth_work_needs_service();
    }

    bool depth_work_needs_service() const {
      // This predicate is sampled on the normal D3D owner: the encode thread for streaming,
      // or the capture/presentation thread for Local AR. Drain completed publications there
      // without creating conversion demand or changing the caller's wait policy.
      if (depth_estimator) {
        depth_estimator->poll_gpu_completion();
      }
      const models::host_sbs_conversion_work_t work {
        .pipeline_enabled = sbs_mode != ::video::SBS_OFF &&
                            models::host_sbs_renderer_uses_depth_pipeline(host_sbs_renderer),
        .estimator_present = static_cast<bool>(depth_estimator),
        .pipeline_build_pending = depth_estimator_build.valid(),
        .depth_completion_pending = pending_matched_frame != nullptr,
        .authority_reprocess_pending = depth_authority_reprocess_pending || subtitle_context_reprocess_pending ||
                                       (sbs_mode != ::video::SBS_OFF &&
                                        detail::host_sbs_window_authority_observation_needed(
                                          depth_estimator != nullptr,
                                          models::host_sbs_renderer_uses_depth_pipeline(host_sbs_renderer)
                                        ) &&
                                        video_player_registry && !video_player_registry_ready_observed && video_player_registry->ready()),
        .dump_pending = sbs_dumper.needs_conversion_poll(),
      };
      return work.needs_service([]() {
#ifdef SUNSHINE_TESTS
        if (active_conversion_poll_model_read) {
          ++active_conversion_poll_model_read->calls;
          return active_conversion_poll_model_read->read();
        }
#endif
        return models::tensorrt_model_prepare_status(::video::host_sbs_v2_depth_model());
      });
    }

    std::optional<std::chrono::steady_clock::time_point> rendered_content_timestamp() const {
      return rendered_content_timestamp_;
    }

    int convert(
      platf::img_t &img_base,
      const std::optional<std::chrono::steady_clock::time_point> next_encode_target = std::nullopt,
      rgb_output_t *rgb = nullptr
    ) {
      auto &img = (img_d3d_t &) img_base;
      conversion_missing_source = false;
      std::optional<unsigned> converted_input_surface;
      // Select only when this call will write native planes. A packed repeat, skipped capture,
      // or blank image keeps the last converted surface, even when every other input is busy.
      const auto select_native_output = [&]() {
        if (!native_input_owner || rgb || converted_input_surface) {
          return true;
        }
        const auto surface = native_input_owner->begin_input_conversion();
        if (!surface || *surface >= output_surface_views.size()) {
          BOOST_LOG(error) << "No writable native encoder input surface."sv;
          return false;
        }
        converted_input_surface = surface;
        active_output_surface = *surface;
        return true;
      };
      // Reference white belongs to the captured Windows display, not its EDID peak or
      // the headset. Poll at most once a second, including when the desktop is static.
      const auto gamma_now = std::chrono::steady_clock::now();
      if (stream_gamma_mode_ != ::video::stream_gamma_mode_e::windows_default && gamma_now >= stream_gamma_white_query_at) {
        stream_gamma_white_query_at = gamma_now + 1s;
        if (!set_stream_gamma(stream_gamma_mode_)) {
          return -1;
        }
      }
      if (::video::is_game_mode(sbs_mode)) {
        // Missing depth/exports are diagnostic use cases too. Poll before the blank-image
        // guard and without sharing the AI dumper's request/retry ownership.
        game_export.dumper.poll(device.get(), device_ctx.get(),
          display->screen_rect(),
          game_export.source_width, game_export.source_height, sbs_mode);
      }
      auto converted_content_timestamp =
        ::video::detail::select_rendered_content_timestamp(
          false,
          std::nullopt,
          false,
          std::nullopt,
          std::nullopt,
          img_base.content_timestamp,
          img_base.frame_timestamp
        );
      if (!img.blank) {
        std::optional<platf::reshade_bridge::frame_t> external;
        std::optional<::video::game_source_identity_t> game_source;
        if (game_export.receiver) {
          game_source = game_export.observe(sbs_mode, display->screen_rect(), sbs_viewport, external);
        }
        // Without an add-on export, the game in front may be a text game. It is reported ready
        // only while this session can really convert it, so the client never widens for nothing.
        bool text_game_in_front = false;
        if (!game_source && text_game_eligible && models::host_sbs_renderer_uses_depth_pipeline(host_sbs_renderer)) {
          game_source = text_game_pin.observe(
            std::chrono::steady_clock::now(),
            {display->screen_x, display->screen_y, display->screen_x + display->width, display->screen_y + display->height}
          );
          text_game_in_front = game_source.has_value();
        }
        // Building the depth pipeline is a burst of GPU work. Local presentation follows the
        // display's own refresh clock and loses it under such a burst, so build the pipeline when
        // the session starts, as Host 3D does, and not when the first text game is detected.
        if (local_presentation && text_game_eligible && models::host_sbs_renderer_uses_depth_pipeline(host_sbs_renderer)) {
          ensure_depth_estimator();
        }
        text_game_ai_active = sbs_mode == ::video::SBS_GAME_SBS && text_game_in_front;
        text_game_serviced_at = std::chrono::steady_clock::now();
        if (sbs_mode == ::video::SBS_GAME_SBS) {
          // The client stays in packed Game 3D while games come and go, so one encoder session
          // moves between the Host AI path and the export or flat path. Both draw into the same
          // packed intermediate and encoder input: pixels and geometry of one route must never
          // be repeated or reused as the other's.
          if (text_game_pin.switch_route(text_game_ai_active)) {
            matched_presentation_cache.reset();
            latest_v2_lineage.reset();
            host_sbs_encoder_input_state.reset();
            // Depth work submitted before the switch is not serviced while the other route
            // draws. Whatever completes meanwhile is retired, not shown, when this route resumes.
            subtitle_context.revoke();
            subtitle_context_reprocess_pending = true;
            // Dump 3D follows the route: a text game is converted by the Host path, so its dump
            // is the Host package; an add-on game's is the add-on's own.
            game_export.dumper.set_button_request(text_game_ai_active ? nullptr : dump_button_request);
            sbs_dumper.set_button_request(text_game_ai_active ? dump_button_request : nullptr);
          }
        }
        game_export.publish_status(sbs_mode, game_source);
        // Every read of the export is recorded on this context before convert() returns, on every
        // path; the claim that replaces the frame can then return its slot without waiting.
        auto export_reads_recorded = util::fail_guard([&]() {
          if (external) {
            game_export.receiver->reads_recorded();
          }
        });

        // A streaming encoder that converts a live packed export never reads the desktop.
        const bool export_owns_output = external && ::video::is_packed_mode(sbs_mode);
        claim_capture_pixels(export_owns_output && external_frame_wake);
        if (img.pixels_skipped && !export_owns_output) {
          // Capture skipped these pixels while an export owned the stream, and it no longer does.
          // The encoder input stays as it was; capture now copies the whole desktop again.
          conversion_missing_source = true;
          return 0;
        }

        // A valid packed export owns its texture and only uses immutable CPU cursor metadata
        // from capture. Do not open or wait on an unrelated desktop resource on that path.
        // Flat fallback, unpacked discovery and Host AI retain the desktop context and lock.
        encoder_img_ctx_t *img_ctx = nullptr;
        if (!external || !::video::is_packed_mode(sbs_mode)) {
          auto [it, inserted] = img_ctx_map.try_emplace(img.id);
          if (inserted) {
            // Capture's pool bounds the cache. Collect expired images only on insertion;
            // erasing other map nodes leaves this image's context valid.
            for (auto gc = img_ctx_map.begin(); gc != img_ctx_map.end();) {
              if (gc != it && gc->second.img_weak.expired()) {
                gc = img_ctx_map.erase(gc);
              } else {
                ++gc;
              }
            }
          }
          img_ctx = &it->second;
          if (initialize_image_context(img, *img_ctx)) {
            return -1;
          }
        }
        detail::keyed_mutex_lock_t encoder_mutex_lock {img_ctx ? img_ctx->encoder_mutex.get() : nullptr};
        const auto acquire_current_capture = [&]() {
          const auto status = encoder_mutex_lock.lock();
          if (status != S_OK) {
            BOOST_LOG(error) << "Failed to acquire encoder mutex [0x"sv << util::hex(status).to_string_view() << ']';
          }
          return status == S_OK;
        };
        const bool depth_route = sbs_mode == ::video::SBS_AI || text_game_ai_active;
        // Snapshot probes may read the current source even on an otherwise source-free repeat.
        if (img_ctx && (!depth_route || test_context) && !acquire_current_capture()) {
          return -1;
        }

        // `planes` selects Y (1) and/or UV (2); the cursor patch re-runs one plane at a time.
        auto draw = [&](auto &input, const D3D11_VIEWPORT &y_or_yuv_viewport, const D3D11_VIEWPORT &uv_viewport, bool input_is_linear, bool y_already_written = false, bool external_input = false, bool input_is_pq = false, int planes = 3) {
          device_ctx->PSSetShaderResources(0, 1, &input);
          ID3D11Buffer *converter_buffers[] = {
            color_matrix.get(),
            external_input ? external_color_transform.get() : sdr_color_transform.get(),
            subsample_offset.get(),
          };
          // Bind all converter constants for every shader variant. Other passes use their own
          // PS constant-buffer slots, so never rely on stale context state between SBS/RGB draws.
          device_ctx->PSSetConstantBuffers(0, 3, converter_buffers);
          ID3D11Buffer *gamma_constants = stream_gamma_transform.get();
          device_ctx->PSSetConstantBuffers(4, 1, &gamma_constants);

          // The optional HDR Host-SBS MRT has already populated luma while producing the packed
          // RGB raster. Every other path retains the established standalone Y/YUV draw.
          auto &output = active_output_views();
          if (!y_already_written && (planes & 1)) {
            auto *luma = output.y_or_yuv.get();
            device_ctx->OMSetRenderTargets(1, &luma, nullptr);
            device_ctx->VSSetShader(convert_Y_or_YUV_vs.get(), nullptr, 0);
            auto *y_shader = input_is_linear ? convert_Y_or_YUV_fp16_ps.get() : convert_Y_or_YUV_ps.get();
            device_ctx->PSSetShader(input_is_pq ? convert_Y_or_YUV_pq_ps.get() : y_shader, nullptr, 0);
            device_ctx->RSSetViewports(1, &y_or_yuv_viewport);
            device_ctx->Draw(3, 0);
          }

          // Draw UV if needed
          if (output.uv && (planes & 2)) {
            assert(format == DXGI_FORMAT_NV12 || format == DXGI_FORMAT_P010);
            auto *chroma = output.uv.get();
            device_ctx->OMSetRenderTargets(1, &chroma, nullptr);
            device_ctx->VSSetShader(convert_UV_vs.get(), nullptr, 0);
            auto *uv_shader = input_is_linear ? convert_UV_fp16_ps.get() : convert_UV_ps.get();
            device_ctx->PSSetShader(input_is_pq ? convert_UV_pq_ps.get() : uv_shader, nullptr, 0);
            device_ctx->RSSetViewports(1, &uv_viewport);
            device_ctx->Draw(3, 0);
          }
        };

        auto draw_rgb = [&](ID3D11ShaderResourceView *input, bool input_is_linear, bool input_is_hdr = false, bool input_is_pq = false) {
          device_ctx->OMSetRenderTargets(1, &rgb->target, nullptr);
          device_ctx->VSSetShader(sbs_reprojection_vs.get(), nullptr, 0);
          ID3D11PixelShader *presentation_shader = rgb_present_ps.get();
          if (input_is_pq) {
            // A PQ export is HDR: decode it for a linear output, or tone map it for an SDR one.
            presentation_shader = rgb->linear ? rgb_present_pq_to_linear_ps.get() : rgb_present_pq_to_srgb_ps.get();
            input_is_hdr = true;
          } else if (input_is_linear != rgb->linear) {
            presentation_shader = input_is_linear ?
                                    (input_is_hdr ? rgb_present_hdr_to_srgb_ps.get() : rgb_present_linear_to_srgb_ps.get()) :
                                    rgb_present_srgb_to_linear_ps.get();
          }
          device_ctx->PSSetShader(presentation_shader, nullptr, 0);
          if ((!input_is_linear && !input_is_pq && rgb->linear) || (input_is_hdr && !rgb->linear)) {
            ID3D11Buffer *sdr_white = rgb_present_sdr_white.get();
            device_ctx->PSSetConstantBuffers(1, 1, &sdr_white);
          }
          device_ctx->RSSetViewports(1, &rgb_present_viewport);
          device_ctx->PSSetSamplers(0, 1, &sampler_linear);
          device_ctx->PSSetShaderResources(0, 1, &input);
          device_ctx->Draw(3, 0);

          ID3D11RenderTargetView *null_rtv = nullptr;
          ID3D11ShaderResourceView *null_srv = nullptr;
          device_ctx->OMSetRenderTargets(1, &null_rtv, nullptr);
          device_ctx->PSSetShaderResources(0, 1, &null_srv);
        };

        auto copy_rgb = [&](ID3D11Texture2D *input, bool input_is_linear) {
          if (!input || !rgb->texture) {
            return false;
          }

          D3D11_TEXTURE2D_DESC input_desc {};
          D3D11_TEXTURE2D_DESC output_desc {};
          input->GetDesc(&input_desc);
          rgb->texture->GetDesc(&output_desc);
          const bool compatible = source_signatures_match(input_desc, output_desc) &&
                                  input_is_linear == rgb->linear;
          if (!compatible) {
            if (!rgb_copy_fallback_logged) {
              BOOST_LOG(info) << "Local AR exact-copy path unavailable (source "sv
                              << input_desc.Width << 'x' << input_desc.Height << ' '
                              << (int) input_desc.Format << ", target "sv
                              << output_desc.Width << 'x' << output_desc.Height << ' '
                              << (int) output_desc.Format << ", transfer "sv
                              << (input_is_linear ? "linear" : "sRGB") << " -> "sv
                              << (rgb->linear ? "linear" : "sRGB")
                              << "); using the RGB presentation shader."sv;
              rgb_copy_fallback_logged = true;
            }
            return false;
          }

          // Neither resource may remain bound while CopyResource reads/writes it. The local
          // swapchain is an exact-size, exact-format sink, so a copy preserves SDR/HDR values
          // bit-for-bit and avoids another full-screen texture sample and render-target write.
          ID3D11RenderTargetView *null_rtv = nullptr;
          std::array<ID3D11ShaderResourceView *, 5> null_srvs {};
          device_ctx->OMSetRenderTargets(1, &null_rtv, nullptr);
          device_ctx->PSSetShaderResources(0, (UINT) null_srvs.size(), null_srvs.data());
          device_ctx->CopyResource(rgb->texture, input);
          if (!rgb_copy_path_logged) {
            BOOST_LOG(info) << "Local AR exact texture-copy presentation path active."sv;
            rgb_copy_path_logged = true;
          }
          return true;
        };

        // Re-records one pass with the scissor enabled and restores the caller's rasterizer state.
        auto with_scissor = [&](const D3D11_RECT &rect, auto &&record) {
          Microsoft::WRL::ComPtr<ID3D11RasterizerState> previous;
          device_ctx->RSGetState(&previous);
          device_ctx->RSSetState(cursor_scissor_raster.Get());
          device_ctx->RSSetScissorRects(1, &rect);
          const bool recorded = record();
          device_ctx->RSSetState(previous.Get());
          return recorded;
        };

        if (game_export.receiver && ::video::is_packed_mode(sbs_mode) && !text_game_ai_active) {
          ID3D11Texture2D *packed_texture = nullptr;
          ID3D11ShaderResourceView *packed_view = nullptr;
          bool packed_linear = false;
          bool packed_pq = false;
          // A visible Windows cursor lives in a frame-sized patch that is valid only around the
          // texels it changed. Every pass reads the export itself, then re-runs over the
          // cursor's output pixels from the patch. Those pixels read only patch texels equal to
          // the composited frame, so the output is the same as converting a full composited copy.
          std::optional<sbs_cursor::result_t> cursor_patch;
          if (external) {
            packed_texture = external->texture;
            packed_view = external->view;
            packed_linear = external->linear;
            packed_pq = external->transfer == platf::reshade_bridge::transfer_e::pq;
            converted_content_timestamp = external->timestamp;
            // The final conversion reads exact output-sized texels. Eyes authored at another size
            // are resampled first, each output eye from its own source half: scaled over the whole
            // eye, or fitted into an even rectangle of it surrounded by black when the aspect
            // differs (receiver_t::poll() refused every other placement for this raster). FP16
            // keeps linear and encoded values alike; the transfer stays tracked by packed_linear.
            D3D11_TEXTURE2D_DESC packed_desc {};
            packed_texture->GetDesc(&packed_desc);
            const auto output_width = static_cast<std::uint32_t>(std::lround(sbs_viewport.Width));
            const auto output_height = static_cast<std::uint32_t>(std::lround(sbs_viewport.Height));
            const auto placement = ::reshade_bridge::place_output(packed_desc.Width, packed_desc.Height, output_width, output_height);
            const auto fit_passes = platf::packed_output::resample_passes(placement, output_width, output_height);
            const bool resample = placement.fit != ::reshade_bridge::output_fit::exact;
            // The patch must hold the footprint of the most minified pass that reads the
            // export: the resample, the Local AR presentation, or the half-size chroma plane.
            const auto scale = [&](const D3D11_VIEWPORT &viewport) {
              return std::min(viewport.Width / float(packed_desc.Width), viewport.Height / float(packed_desc.Height));
            };
            const float minimum_scale = resample           ? platf::packed_output::minimum_scale(fit_passes, packed_desc.Width, packed_desc.Height) :
                                        rgb ? scale(rgb_present_viewport) :
                                                             std::min(scale(out_Y_or_YUV_viewport), scale(out_UV_viewport));
            auto transfer = packed_linear ? sbs_cursor::source_transfer_e::scrgb : sbs_cursor::source_transfer_e::srgb;
            if (packed_pq) {
              transfer = sbs_cursor::source_transfer_e::pq;
            }
            const auto composited = game_export.cursor->compose(
              packed_texture,
              packed_view,
              img.cursor,
              transfer,
              game_export.cursor_white_multiplier,
              external->ui_parallax_uv,
              sbs_cursor::patch_margin(minimum_scale)
            );
            if (!composited) {
              BOOST_LOG(error) << "Failed to composite the Windows cursor over external Game 3D."sv;
              return -1;
            }
            if (composited->regions) {
              cursor_patch = composited;
              if (!game_export.cursor_logged) {
                const char *blend = packed_linear ? "linear HDR" : "SDR";
                if (packed_pq) {
                  blend = "PQ HDR10, blended in linear light";
                }
                BOOST_LOG(info) << "Game 3D: compositing the Windows cursor into both eyes (" << blend << ").";
                game_export.cursor_logged = true;
              }
            }
            if (resample) {
              if (!ensure_sbs_intermediate_storage(true)) {
                return -1;
              }
              const auto resample_from = [&](const platf::packed_output::pass_t &pass, ID3D11ShaderResourceView *source, ID3D11PixelShader *shader) {
                const host_sbs_v2_draw_command_t resample_draw {
                  .render_targets = {sbs_intermediate_rtv.get(), nullptr},
                  .vertex_shader = sbs_reprojection_vs.get(),
                  .pixel_shader = shader,
                  .viewport = pass.viewport,
                  .sampler = sampler_linear.get(),
                  .shader_resources = {source},
                  .geometry_constants = sbs_reprojection_cbuffer.get(),
                };
                return record_host_sbs_v2_draw(device_ctx.get(), resample_draw);
              };
              if (fit_passes.clear) {
                // The black around fitted eyes: zero is black in sRGB code values and in linear
                // scRGB, the two encodings this FP16 intermediate holds (PQ is decoded below).
                constexpr float black[4] {0.0f, 0.0f, 0.0f, 1.0f};
                device_ctx->ClearRenderTargetView(sbs_intermediate_rtv.get(), black);
              }
              // A PQ export is decoded per tap, so the intermediate is FP16 scRGB filtered in
              // linear light. The cursor patch of a PQ export is already scRGB. A fitted eye's
              // pass and its cursor regions stay inside that eye's rectangle (its scissor).
              auto *const resample_shader = packed_pq ? game_export.resample_pq_ps.get() : game_export.resample_ps.get();
              for (std::uint32_t pass_index = 0; pass_index < fit_passes.count; ++pass_index) {
                const auto &pass = fit_passes.passes[pass_index];
                const auto draw_export = [&]() {
                  return resample_from(pass, packed_view, resample_shader);
                };
                if (!(pass.scissor ? with_scissor(*pass.scissor, draw_export) : draw_export())) {
                  return -1;
                }
                for (std::uint32_t i = 0; cursor_patch && i < cursor_patch->regions; ++i) {
                  const auto region = platf::packed_output::pass_region(pass, cursor_patch->changed[i], packed_desc.Width, packed_desc.Height);
                  if (region && !with_scissor(*region, [&]() {
                        return resample_from(pass, cursor_patch->view, game_export.resample_ps.get());
                      })) {
                    return -1;
                  }
                }
              }
              packed_texture = sbs_intermediate_texture.get();
              packed_view = sbs_intermediate_srv.get();
              packed_linear = packed_linear || packed_pq;
              packed_pq = false;
              cursor_patch.reset();
            }
          } else {
            // An unavailable, incompatible, or unfocused add-on publisher never selects Host AI,
            // and neither does the desktop: only a text game in front, which took the branch
            // below, is converted by the host. Keep everything else usable by duplicating it
            // into two flat eyes.
            packed_linear = img.format == DXGI_FORMAT_R16G16B16A16_FLOAT;
            if (!ensure_sbs_intermediate_storage(packed_linear)) {
              return -1;
            }
            const host_sbs_v2_draw_command_t flat_draw {
              .render_targets = {sbs_intermediate_rtv.get(), nullptr},
              .vertex_shader = sbs_reprojection_vs.get(),
              .pixel_shader = sbs_flat_identity_ps.get(),
              .viewport = sbs_viewport,
              .sampler = sampler_linear.get(),
              .shader_resources = {img_ctx->encoder_input_res.get()},
              .geometry_constants = sbs_reprojection_cbuffer.get(),
            };
            if (!record_host_sbs_v2_draw(device_ctx.get(), flat_draw)) {
              return -1;
            }
            packed_texture = sbs_intermediate_texture.get();
            packed_view = sbs_intermediate_srv.get();
          }
          D3D11_TEXTURE2D_DESC final_desc {};
          packed_texture->GetDesc(&final_desc);
          if (rgb) {
            // A PQ export never matches a local sRGB or scRGB swapchain's transfer.
            if (!packed_pq && copy_rgb(packed_texture, packed_linear)) {
              // The patch has the export's format and transfer; copy the cursor's texels exactly.
              for (std::uint32_t i = 0; cursor_patch && i < cursor_patch->regions; ++i) {
                const auto &changed = cursor_patch->changed[i];
                const D3D11_BOX box {UINT(changed.left), UINT(changed.top), 0, UINT(changed.right), UINT(changed.bottom), 1};
                device_ctx->CopySubresourceRegion(rgb->texture, 0, box.left, box.top, 0, cursor_patch->texture, 0, &box);
              }
            } else {
              draw_rgb(packed_view, packed_linear, packed_linear && (external.has_value() || display_is_hdr), packed_pq);
              for (std::uint32_t i = 0; cursor_patch && i < cursor_patch->regions; ++i) {
                if (const auto region = sbs_cursor::output_region(cursor_patch->changed[i], final_desc.Width, final_desc.Height, rgb_present_viewport)) {
                  with_scissor(*region, [&]() {
                    draw_rgb(cursor_patch->view, cursor_patch->linear, cursor_patch->linear);
                    return true;
                  });
                }
              }
            }
          } else {
            if (!select_native_output()) {
              return -1;
            }
            draw(packed_view, out_Y_or_YUV_viewport, out_UV_viewport, packed_linear, false, external.has_value(), packed_pq);
            for (std::uint32_t i = 0; cursor_patch && i < cursor_patch->regions; ++i) {
              auto *patch_view = cursor_patch->view;
              const auto &changed = cursor_patch->changed[i];
              if (const auto region = sbs_cursor::output_region(changed, final_desc.Width, final_desc.Height, out_Y_or_YUV_viewport)) {
                with_scissor(*region, [&]() {
                  draw(patch_view, out_Y_or_YUV_viewport, out_UV_viewport, cursor_patch->linear, false, true, false, 1);
                  return true;
                });
              }
              const auto chroma = active_output_views().uv ?
                                    sbs_cursor::output_region(changed, final_desc.Width, final_desc.Height, out_UV_viewport) :
                                    std::nullopt;
              if (chroma) {
                with_scissor(*chroma, [&]() {
                  draw(patch_view, out_Y_or_YUV_viewport, out_UV_viewport, cursor_patch->linear, false, true, false, 2);
                  return true;
                });
              }
            }
          }
        } else if (depth_route) {
          const bool input_is_linear =
            img.format == DXGI_FORMAT_R16G16B16A16_FLOAT;

          // Retire an earlier Dump 3D staging batch only when its terminal GPU event is ready.
          // GPU readiness never flushes or waits. Once ready, this explicit debug action copies
          // the complete snapshot here and remains outside production timing telemetry.
          sbs_dumper.poll_pending_readback(device_ctx.get());

          // Perf benchmark: CPU wall time of the whole SBS block (estimator dispatch + composite
          // draw submission). GPU-side inference times are measured separately via CUDA events in
          // the estimator. No-op unless runtime diagnostics are on.
          const bool perf = diagnostics_enabled && sbs_perf::enabled();
          const auto perf_t0 = perf ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
          const bool snapshot_debug_inputs = sbs_dumper.snapshot_requested();
          if (!snapshot_debug_inputs && sbs_debug_mapping_texture) {
            // Dump 3D copied what it needed from these full-size targets into its own staging
            // textures before maybe_dump() returned. Without a request they hold only memory
            // (8 bytes per packed pixel); the next request creates them again.
            sbs_debug_mapping_texture.Reset();
            sbs_debug_mapping_rtv.Reset();
            sbs_debug_mapping_srv.Reset();
            sbs_debug_mask_texture.Reset();
            sbs_debug_mask_rtv.Reset();
            sbs_debug_mask_srv.Reset();
            sbs_debug_geometry_ready = false;
            sbs_debug_geometry_init_attempted = false;
          }

          // Lazy-create the depth estimator on the first SBS frame.
          if (models::host_sbs_renderer_uses_depth_pipeline(host_sbs_renderer)) {
            ensure_depth_estimator();
          } else {
            sbs_dumper.cancel_pending_request();
          }
          // Only a pending inference may keep a capture image between conversions. Inside this
          // call the completed source stays alive through its last shader or dump read.
          auto release_completed_capture_images = util::fail_guard([this]() {
            for (auto &slot : matched_frame_slots) {
              if (&slot != pending_matched_frame) {
                slot.release_capture_image();
              }
            }
          });
          std::optional<detail::keyed_mutex_lock_t<IDXGIKeyedMutex>> prior_capture_lock;
          const auto *prior_pending_frame = pending_matched_frame;
          HRESULT pending_source_status = S_OK;
          const auto acquire_pending_capture = [&]() {
            if (pending_source_status != S_OK || !prior_pending_frame || !prior_pending_frame->capture_mutex) {
              return pending_source_status == S_OK;
            }
            // Preserve current-before-prior ordering. Polling a busy root needs neither key.
            if (!acquire_current_capture()) {
              pending_source_status = E_FAIL;
              return false;
            }
            if (prior_pending_frame->capture_mutex != img_ctx->encoder_mutex.get() && !prior_capture_lock) {
              prior_capture_lock.emplace(prior_pending_frame->capture_mutex);
              pending_source_status = prior_capture_lock->lock(0, 0);
              if (pending_source_status != S_OK && pending_source_status != WAIT_TIMEOUT) {
                BOOST_LOG(error) << "Failed to acquire pending Host SBS texture mutex [0x"sv
                                 << util::hex(pending_source_status).to_string_view() << ']';
              }
            }
            return pending_source_status == S_OK;
          };
          const models::pending_depth_source_access_t pending_source_access {
            &acquire_pending_capture,
            [](const void *owner) { return (*static_cast<decltype(acquire_pending_capture) *>(owner))(); },
          };
          // Stable tensor copies are inserted into the same D3D command stream before the current
          // preprocess. Skip whole-frame performance publication for an explicit dump frame so
          // diagnostic work cannot appear as a production regression.
          auto *gpu_timer =
            perf && !snapshot_debug_inputs ? begin_sbs_gpu_timer() : nullptr;

          // Production uses bounded matched pairing: retain the exact capture image for
          // asynchronous inference, then warp only its completion. The DDup reuse exception can prove
          // either an identical full desktop-content timestamp or an exact ROI untouched by every
          // dirty/move record since the accepted crop. It still fresh-warps current color.
          const auto input_color_space = input_is_linear ?
                                           (display_is_hdr ? models::input_color_space::scrgb_hdr :
                                                             models::input_color_space::linear_sdr) :
                                           models::input_color_space::srgb;
          const auto current_source_timestamp = img_base.frame_timestamp;
          const auto current_content_timestamp = img_base.content_timestamp;
          // Observe a changed source before any completion wait. An old, exactly unchanged
          // packed image must not charge its static lifetime to this first changed delivery.
          matched_presentation_cache.observe_current_source(current_source_timestamp, gamma_now);
          const auto &current_ddup_damage = img.ddup_damage;
          // Each lineage copy advances its own transitive proof. Repeated same-token queries
          // are constant-time, so a cross-slot memo would only leave another copy's anchor stale.
          const auto current_input_reuse_kind = [&](const matched_frame_attribution_t &attribution) {
            return matched_input_reuse_kind(
              attribution,
              current_content_timestamp,
              current_ddup_damage
            );
          };
          // Once the per-stream estimator exists, observe foreground continuity on every SBS
          // conversion, not merely when TensorRT can accept a new matched frame. Before then
          // there is no ROI completion/cache to revoke, so synchronous USER32/DWM work has no
          // consumer. A focus transition between accepted inferences still revokes authority.
          live_window_authority_observation_t initial_source_authority;
          const auto &live_source_desc = img_ctx->encoder_texture.description();
          if (detail::host_sbs_window_authority_observation_needed(depth_estimator != nullptr, models::host_sbs_renderer_uses_depth_pipeline(host_sbs_renderer))) {
            initial_source_authority = observe_live_window_authority(
              live_source_desc,
              current_content_timestamp
            );
          }
          const auto same_source = [](
                                     const std::optional<std::chrono::steady_clock::time_point> &left,
                                     const std::optional<std::chrono::steady_clock::time_point> &right
                                   ) {
            // A missing identity is never evidence that two source images are the same.
            return left && right && *left == *right;
          };
          const auto frame_id = ++sbs_frame_sequence;
          const auto test_watermark_at_entry = test_observer.record ? drawn_source_order.last_drawn() : 0u;
          if (test_observer.record) {
            test_current_source_ticks = img_base.frame_timestamp ? img_base.frame_timestamp->time_since_epoch().count() : 0;
            test_current_content_ticks = img_base.content_timestamp ? img_base.content_timestamp->time_since_epoch().count() : 0;
          }
          // Cached geometry borrows only the attribution. A completion may also have the
          // physical slot needed for matched color, capture lifetime, and exact Dump 3D inputs.
          matched_frame_slot_t *matched_render_capture = nullptr;
          matched_frame_attribution_t *retained_render_attribution = nullptr;
          const auto render_attribution = [&]() {
            return matched_render_capture ? &matched_render_capture->attribution : retained_render_attribution;
          };
          matched_frame_slot_t *matched_candidate_slot = nullptr;
          bool cached_current_color_warp = false;
          host_sbs::observation_handoff_t handoff;
          const auto &depth_step = handoff.step();
          const auto &step_observation = depth_step.observation;
          const host_sbs::observation_handoff_t *render_handoff = &handoff;
          const auto clear_matched_selection = [&]() {
            retained_render_attribution = nullptr;
            matched_render_capture = nullptr;
            render_handoff = &handoff;
            handoff.reset();
            cached_current_color_warp = false;
          };
          const auto adopt_matched_completion = [&](
                                                   matched_frame_slot_t &slot,
                                                   models::depth_step_result_t *polled = nullptr,
                                                   std::optional<std::chrono::steady_clock::time_point> age_sample_at = std::nullopt
                                                 ) {
            matched_render_capture = &slot;
            // An explicit poll supplies the new step. Admission has already composed its step
            // and can report a completion together with a separately enqueued submission.
            if (polled) {
              if (!slot.attribution.depth_input_region.is_video_region()) {
                depth_authority_reprocess_pending = false;
              }
              handoff.select(std::move(*polled));
              if (test_observer.record) {
                converter_test_observer::event_t event;
                event.kind = converter_test_observer::event_e::ack;
                emit_test_observation(std::move(event), &slot.attribution, &step_observation, slot.srv.get(), frame_id);
              }
            }
            if (diagnostics_enabled) {
              const auto sampled_at = age_sample_at ? *age_sample_at : std::chrono::steady_clock::now();
              const double age_ms = std::chrono::duration<double, std::milli>(sampled_at - slot.attribution.captured_at).count();
              matched_stats_age_sum_ms += age_ms;
              matched_stats_age_max_ms = std::max(matched_stats_age_max_ms, age_ms);
              ++matched_stats_completions;
              if (perf) {
                sbs_perf::add_sample_ms("matched_frame_age", age_ms);
              }
            }
          };
          const auto poll_pending_depth = [&](const detail::host_sbs_same_frame_poll_plan_t *plan, bool snapshot = false, models::pending_depth_source_access_t access = {}) {
            if (plan && plan->eligible) {
              // Remote encoders and local presenters may both wait on the completion
              // notification; the estimator applies the owner thread's scheduling qualification.
              return depth_estimator->try_finish_pending_depth_until(
                input_color_space,
                plan->deadline,
                detail::host_sbs_same_frame_poll_max_queries,
                false,
                true,
                access
              );
            }
            return depth_estimator->try_finish_pending_depth_nonblocking(input_color_space, snapshot, access);
          };
          bool unchanged_content_submission_suppressed = false;
          bool damage_guided_submission_suppressed = false;
          bool force_fresh_current_color = false;
          bool completion_finalized_before_admission = false;
          bool block_current_submission_after_early_poll = false;
          bool same_frame_poll_missed_for_current_draw = false;
          // A prior B and a newly admitted C share one conversion's absolute wait cap.
          std::optional<detail::host_sbs_same_frame_poll_plan_t> prior_poll_plan;
#ifdef SUNSHINE_TESTS
          const auto record_native_poll = [&](const auto &plan, const auto &poll, bool prior) noexcept {
            if (test_native_pacing.record_poll) {
              test_native_pacing.record_poll(test_native_pacing.converter_observer.state, {
                frame_id, prior, plan.eligible, plan.deadline,
                poll.ready, poll.timed_out, poll.query_count, poll.wait_duration,
                poll.notification_armed, poll.notification_waited, poll.notification_callback_pending,
              });
            }
          };
#endif
          detail::host_sbs_depth_reuse_authorization_t depth_reuse_authorization;
          const auto consume_matched_completion = [this](const models::depth_observation_views_t &observation) -> matched_frame_slot_t * {
            // A ready result consumed the root even when no usable observation was produced.
            // Its postprocess can overwrite the resources behind every prior borrowed view.
            latest_v2_lineage.reset();
            auto *completed_slot = std::exchange(pending_matched_frame, nullptr);
            if (!observation.completed_frame_valid) {
              return nullptr;
            }
            const auto completed_frame_id = observation.completed_frame_id;
            if (completed_slot && completed_slot->attribution.frame_id == completed_frame_id) {
              return completed_slot;
            }
            const auto error_now = std::chrono::steady_clock::now();
            if (matched_unknown_frame_error_last.time_since_epoch().count() == 0 || error_now - matched_unknown_frame_error_last >= std::chrono::seconds(30)) {
              if (matched_unknown_frame_errors_suppressed) {
                BOOST_LOG(error) << "Matched depth completed unknown frame "sv
                                 << completed_frame_id << "; repeating the last output ("sv
                                 << matched_unknown_frame_errors_suppressed
                                 << " similar occurrence(s) suppressed)."sv;
              } else {
                BOOST_LOG(error) << "Matched depth completed unknown frame "sv
                                 << completed_frame_id << "; repeating the last output."sv;
              }
              matched_unknown_frame_error_last = error_now;
              matched_unknown_frame_errors_suppressed = 0;
            } else {
              ++matched_unknown_frame_errors_suppressed;
            }
            return nullptr;
          };

          const auto validate_completed_slot = [&]() {
            if (!render_attribution()) {
              return false;
            }
            if (!subtitle_context.matches(render_attribution()->subtitle_context_generation)) {
              // Normal focus/video transitions retire old work without displaying its protected field.
              clear_matched_selection();
              matched_presentation_cache.invalidate();
              return false;
            }
            const auto &est = render_handoff->step().observation;
            if (
              !est.completed_frame_valid ||
              est.completed_frame_id != render_attribution()->frame_id ||
              !gpu_transaction_class_matches(est, *render_attribution()) ||
              !depth_completion_region_matches_submission(
                est,
                render_attribution()->depth_input_region
              )
            ) {
              BOOST_LOG(error)
                << "Host SBS rejected a depth completion whose region or GPU transaction class "sv
                   "does not match its exact buffered color frame; "sv
                   "rendering current-frame identity."sv;
              clear_matched_selection();
              matched_presentation_cache.invalidate();
              return false;
            }
            if (
              render_attribution()->window_region &&
              !window_region_authorized_for_render(*render_attribution())
            ) {
              if (render_attribution()->depth_input_region.is_video_region() ||
                  render_attribution()->subtitle_context_eligible) {
                // The completion is still consumed and releases its pending slot, but geometry
                // observed before a focus/move/resize/monitor transition cannot warp newer pixels.
                matched_presentation_cache.reset();
                clear_matched_selection();
                latest_v2_lineage.reset();
                return false;
              }
              // Full-source depth is independent of optional exact-full window provenance.
              render_attribution()->window_region.reset();
              render_attribution()->window_region_observer_status = "not-observed";
              render_attribution()->window_region_mapping_status = "not-mapped";
              render_attribution()->live_window_authority_generation = 0u;
            }
            subtitle_context_reprocess_pending = false;
            return true;
          };

          const auto apply_completed_domain_and_renderer = [&]() {
            if (!render_attribution()) {
              return false;
            }
            const auto &render_step = render_handoff->step();
            const auto &est = render_step.observation;
            if (render_step.completion.input_domain_reset) {
              sbs_telemetry_min_frame_id = std::max(
                sbs_telemetry_min_frame_id,
                est.completed_frame_id
              );
            }

            // A poisoned CUDA/TensorRT producer cannot recover by repeating its last stereo pair:
            // that turns a fail-closed renderer into a frozen stream.
            if (depth_estimator && depth_estimator->has_terminal_failure()) {
              const auto failed_renderer =
                models::fail_host_sbs_renderer_flat(host_sbs_renderer);
              if (failed_renderer != host_sbs_renderer) {
                clear_matched_selection();
                fail_depth_pipeline_flat();
                BOOST_LOG(error)
                  << "Host SBS parallax-v2 producer reported a terminal CUDA/TensorRT failure; "sv
                     "this stream is now live flat identity with depth submissions disabled."sv;
              }
              return false;
            }

            // Authenticate the first completed V2 field for which the exact buffered color
            // remains. A later invalid frame is handled inside the V2 shader.
            if (host_sbs_renderer == models::host_sbs_renderer_e::awaiting_v2) {
              const bool result_authenticated = render_handoff->authenticate_structure();
              host_sbs_renderer = models::latch_host_sbs_renderer(
                host_sbs_renderer,
                result_authenticated
              );
              if (host_sbs_renderer == models::host_sbs_renderer_e::parallax_v2) {
                BOOST_LOG(info)
                  << "Host SBS renderer latched to authenticated parallax-v2 for this stream "sv
                  << "(contract schema "sv
                  << models::depth_coordinate_v2::contract_schema << ", tag 0x"sv
                  << util::hex(models::depth_coordinate_v2::contract_tag).to_string_view()
                  << ", shader closure "sv
                  << models::depth_coordinate_v2::shader_source_closure_sha256
                  << ", live renderer closure "sv
                  << sbs_reprojection_v2_live_source_closure_sha256
                  << ", fixed cfg.pop_strength "sv
                  << est.parallax_v2_requested_pop_strength
                  << ", model shape "sv << est.raw_width << 'x' << est.raw_height
                  << ", live field "sv << est.field_width << 'x' << est.field_height
                  << ")."sv;
              } else {
                BOOST_LOG(warning)
                  << "Host SBS V2 authentication failed closed; this stream is permanently "sv
                     "latched to live flat identity (authenticated_result="sv
                  << (result_authenticated ? "true" : "false")
                  << ", producer_active="sv
                  << (est.parallax_v2_producer_active ? "true" : "false")
                  << ", model_shape="sv << est.raw_width << 'x' << est.raw_height
                  << ", field_shape="sv << est.field_width << 'x' << est.field_height
                  << ")."sv;
                clear_matched_selection();
                fail_depth_pipeline_flat();
                return false;
              }
            }
            return render_attribution() != nullptr;
          };

          const auto retain_completed_lineage = [&, this]() {
            if (!matched_render_capture || !step_observation.completed_frame_valid || render_handoff != &handoff) {
              return;
            }
            const auto &completed_source_desc = matched_render_capture->texture ?
                                                   matched_render_capture->texture.description() :
                                                   live_source_desc;
            (void) retain_latest_v2_lineage(
              handoff,
              matched_render_capture->attribution,
              completed_source_desc,
              live_source_desc,
              input_color_space
            );
          };

          // Consume a completed publication before another root can overwrite its borrowed
          // fields. Missing/invalid receipts leave the ordinary force-infer path available.
          const auto consume_publications = [&, this](const bool late) {
            if (!depth_estimator) {
              return;
            }
            const auto publications = depth_estimator->poll_publication_receipts();
            if (diagnostics_enabled && publications.failed) {
              ++matched_stats_publication_failure_polls;
            }
            if (publications.current_publication) {
              const auto &receipt = *publications.current_publication;
              if (depth_estimator->publication_is_current(receipt.expected) && models::host_sbs_gpu_completion_receipt::matches_publication(receipt, latest_v2_lineage.estimate().publication)) {
                if (late && diagnostics_enabled && !latest_v2_lineage.receipt) {
                  ++matched_stats_publication_late_ready;
                }
                latest_v2_lineage.receipt = receipt;
              }
            }
          };
          consume_publications(false);

          // Retire a ready prior root before deriving this frame's DDup/adaptive proof. Its new
          // field publication invalidates the old borrowed aliases; a later nonblocking receipt
          // authenticates the new publication without turning root readiness into GPU validity.
          if (
            depth_estimator &&
            models::host_sbs_renderer_uses_depth_pipeline(host_sbs_renderer)
          ) {
            if (prior_pending_frame) {
              prior_poll_plan = detail::host_sbs_same_frame_poll_plan(
                true, snapshot_debug_inputs, next_encode_target, std::chrono::steady_clock::now()
              );
              auto polled = poll_pending_depth(&*prior_poll_plan, snapshot_debug_inputs, pending_source_access);
              if (pending_source_status != S_OK && pending_source_status != WAIT_TIMEOUT) {
                return -1;
              }
#ifdef SUNSHINE_TESTS
              record_native_poll(*prior_poll_plan, polled, true);
#endif
              if (polled.wait_attempted) {
                // C's admission must see authority sampled after the possible B wait.
                initial_source_authority = observe_live_window_authority(
                  live_source_desc, current_content_timestamp, &initial_source_authority.foreground
                );
              }
              if (polled.ready) {
                if (auto *completed_slot = consume_matched_completion(polled.result.observation)) {
                  adopt_matched_completion(*completed_slot, &polled.result);
                  completion_finalized_before_admission =
                    validate_completed_slot() && apply_completed_domain_and_renderer();
                  if (completion_finalized_before_admission) {
                    retain_completed_lineage();
                  }
                }
                block_current_submission_after_early_poll = !completion_finalized_before_admission;
              }
            }
          }
          // Derive C proof only after B harvest and the fresh authority observation.
          const auto current_authority = live_presentation_authority();
          const bool producer_terminal =
            depth_estimator && depth_estimator->has_terminal_failure();
          const detail::host_sbs_source_admission_t source_admission {
            .pipeline_active = depth_estimator &&
                               models::host_sbs_renderer_uses_depth_pipeline(host_sbs_renderer),
            .parallax_v2_renderer =
              host_sbs_renderer == models::host_sbs_renderer_e::parallax_v2,
            .snapshot_debug_inputs = snapshot_debug_inputs,
            .authority_reprocess_pending = depth_authority_reprocess_pending || subtitle_context_reprocess_pending,
            .producer_terminal = producer_terminal,
            .interactive_move_size = current_authority.interactive_move_size,
          };
          const bool dedup_gate_open =
            source_admission.ddup_reuse_allowed(current_content_timestamp);
          const bool cache_reuse_gate_open =
            dedup_gate_open && latest_v2_lineage.known_depth();
          const auto latest_v2_route_matches = latest_v2_lineage_route_matches_current(
            live_source_desc,
            input_color_space,
            current_authority
          );
          const auto cached_reuse_kind = latest_v2_route_matches ?
                                           current_input_reuse_kind(
                                             latest_v2_lineage.attribution
                                           ) :
                                           detail::host_sbs_ddup_reuse_proof_e::none;
          const auto reuse_now = std::chrono::steady_clock::now();
          // An authenticated exact DDup proof remains valid until its source/domain changes.
          // Elapsed time and delivery count alone cannot require new model work.
          depth_reuse_authorization = detail::select_host_sbs_cached_depth_reuse(
            !cache_reuse_gate_open ?
              detail::host_sbs_ddup_reuse_proof_e::none :
              cached_reuse_kind,
            latest_v2_lineage.attribution.frame_id,
            frame_id
          );
          const bool completed_current_route_matches =
            completion_finalized_before_admission && matched_render_capture &&
            same_source(matched_render_capture->attribution.source_timestamp, current_source_timestamp) &&
            handoff.authenticate_structure() &&
            matched_route_matches_current(
              *matched_render_capture,
              live_source_desc,
              input_color_space,
              current_authority
            );
          // Classify any root still pending after the early poll for packed repeats, DDup reuse,
          // and admission of the next root.
          const bool inference_pending = source_admission.pipeline_active && pending_matched_frame;
          const bool retained_source_pending =
            inference_pending && same_source(pending_matched_frame->attribution.source_timestamp, current_source_timestamp);
          matched_frame_slot_t *unchanged_input_pending_slot = nullptr;
          auto pending_reuse_kind = detail::host_sbs_ddup_reuse_proof_e::none;
          if (
            inference_pending && !retained_source_pending && dedup_gate_open &&
            matched_route_matches_current(
              *pending_matched_frame,
              live_source_desc,
              input_color_space,
              current_authority
            )
          ) {
            pending_reuse_kind = current_input_reuse_kind(pending_matched_frame->attribution);
            if (pending_reuse_kind != detail::host_sbs_ddup_reuse_proof_e::none) {
              unchanged_input_pending_slot = pending_matched_frame;
            }
          }
          const auto completed_source_action = detail::host_sbs_completed_source_action(
            current_source_timestamp,
            source_admission,
            {
              render_attribution() ? render_attribution()->frame_id : 0u,
              render_attribution() ? render_attribution()->source_timestamp : std::nullopt,
              completed_current_route_matches,
            },
            {
              matched_presentation_cache.frame_id(),
              latest_v2_lineage.attribution.source_timestamp,
              dedup_gate_open && latest_v2_route_matches &&
                !inference_pending &&
                matched_presentation_cache.frame_id() == latest_v2_lineage.attribution.frame_id &&
                current_ddup_damage && latest_v2_lineage.attribution.inference_ddup_damage &&
                current_ddup_damage->history == latest_v2_lineage.attribution.inference_ddup_damage->history &&
                current_ddup_damage->token == latest_v2_lineage.attribution.inference_ddup_damage->token &&
                matched_presentation_cache.source_matches(current_source_timestamp) &&
                matched_presentation_cache.route_matches(
                  live_source_desc.Width,
                  live_source_desc.Height,
                  input_color_space,
                  depth_authority_reprocess_pending,
                  producer_terminal,
                  subtitle_context.generation(),
                  current_authority
                ),
            }
          );
          const bool current_source_already_completed =
            completed_source_action != detail::host_sbs_completed_source_action_e::observe;
          if (detail::host_sbs_latest_v2_lineage_reset_required(latest_v2_lineage.authenticated(), latest_v2_route_matches, snapshot_debug_inputs, depth_authority_reprocess_pending, producer_terminal)) {
            latest_v2_lineage.reset();
            // Route/dump/reprocess/terminal revocation invalidates the cached resource aliases
            // regardless of any current-frame proof computed above.
          }
          const bool cached_geometry_render_allowed =
            detail::host_sbs_cached_geometry_render_allowed(
              cache_reuse_gate_open && latest_v2_route_matches,
              host_sbs_renderer == models::host_sbs_renderer_e::parallax_v2,
              depth_reuse_authorization,
              latest_v2_lineage.attribution.frame_id,
              frame_id,
              latest_v2_lineage.estimate().shadow_base_final_parallax &&
                latest_v2_lineage.estimate().shadow_state
            );
          if (source_admission.pipeline_active) {
            // The encode loop calls convert() on a retained source when accepted inference is
            // still pending. The early nonblocking poll above already found this exact unit busy,
            // so keep its slot and poll owner intact and do not enqueue duplicate work. Presentation
            // remains fail-open through the bounded packed-output hold (or current flat output once
            // that hold expires); a later convert() adopts the exact completion when its event is
            // ready. Live capture must never turn this retry into an unbounded CUDA stream wait.
            if (current_source_already_completed) {
              // The early poll already consumed this exact captured image, or its exact packed
              // pixels remain available. Repeating them creates no new publication or depth owner
              // and cannot force another root for an unchanged captured source.
              mark_sbs_matched_copy_end(gpu_timer, false);
            } else if (retained_source_pending) {
              mark_sbs_matched_copy_end(gpu_timer, false);
            } else if (unchanged_input_pending_slot && pending_source_status == S_OK) {
              // Never enqueue an input already represented by the sole pending inference. This is
              // either a DDup content-clock duplicate or a changed desktop whose entire damage
              // history stayed outside the exact accepted ROI. Poll once without waiting so a
              // ready completion can immediately drive a fresh current-color warp.
              mark_sbs_matched_copy_end(gpu_timer, false);
              auto polled = poll_pending_depth(nullptr, false, pending_source_access);
              if (pending_source_status == S_OK) {
                unchanged_content_submission_suppressed = true;
                depth_reuse_authorization =
                  detail::select_host_sbs_cached_depth_reuse(
                    pending_reuse_kind,
                    unchanged_input_pending_slot->attribution.frame_id,
                    frame_id
                  );
                damage_guided_submission_suppressed =
                  pending_reuse_kind == detail::host_sbs_ddup_reuse_proof_e::roi_damage;
                force_fresh_current_color = cached_geometry_render_allowed;
              } else if (pending_source_status != WAIT_TIMEOUT) {
                return -1;
              }
              if (polled.ready) {
                if (auto *completed_slot = consume_matched_completion(polled.result.observation)) {
                  force_fresh_current_color =
                    current_input_reuse_kind(completed_slot->attribution) != detail::host_sbs_ddup_reuse_proof_e::none;
                  adopt_matched_completion(*completed_slot, &polled.result, reuse_now);
                }
              }
            } else if (inference_pending || block_current_submission_after_early_poll) {
              // A changed input must never race a prior root that was busy when admission was
              // derived. Even if can_accept_frame() would turn ready later in this call, the DDup
              // proof still names the older lineage. Retire it on the next conversion instead of
              // submitting with stale authority. A ready-but-invalid early completion takes the
              // same one-call fail-closed path.
              mark_sbs_matched_copy_end(gpu_timer, false);
            } else if (cached_geometry_render_allowed) {
              // These pixels have not changed since the authorized publication. Its held
              // analysis remains valid without an age limit, even after a GPU near reuse.
              // Skip analysis work; current capture color is still warped below.
              mark_sbs_matched_copy_end(gpu_timer, false);
              unchanged_content_submission_suppressed = true;
              damage_guided_submission_suppressed =
                depth_reuse_authorization.kind ==
                detail::host_sbs_depth_reuse_kind_e::exact_roi_damage;
              force_fresh_current_color = cached_geometry_render_allowed;
            } else {
              matched_candidate_slot = available_matched_slot(
                completion_finalized_before_admission ? matched_render_capture : nullptr
              );
              // Only a candidate the estimator can accept reserves a matched source.
              const bool estimator_ready =
                matched_candidate_slot && depth_estimator->can_accept_frame();
              if (estimator_ready && !acquire_current_capture()) {
                return -1;
              }
              const bool matched_source_prepared =
                estimator_ready &&
                prepare_matched_frame(
                  *img_ctx,
                  live_source_desc,
                  *matched_candidate_slot,
                  frame_id,
                  input_color_space,
                  current_source_timestamp,
                  current_content_timestamp,
                  current_ddup_damage,
                  initial_source_authority,
                  gpu_timer
                );
              if (!matched_source_prepared) {
                // Every timing slot needs a complete timestamp tuple. A skipped/failed copy gets
                // adjacent markers and publishes no matched_frame_copy_gpu sample.
                mark_sbs_matched_copy_end(gpu_timer, false);
              }
              if (
                completion_finalized_before_admission && matched_render_capture &&
                !matched_route_matches_current(
                  *matched_render_capture,
                  live_source_desc,
                  input_color_space,
                  live_presentation_authority()
                )
              ) {
                // prepare_matched_frame() performs a second live authority observation. If that
                // closes a route race after the early completion was accepted, revoke its cache
                // and delivery now. A route mismatch cannot restore publication authority.
                latest_v2_lineage.reset();
                clear_matched_selection();
                completion_finalized_before_admission = false;
              }
              if (matched_source_prepared) {
                // B's final event may have completed during ordinary source preparation. Give it
                // one nonblocking opportunity before forming C's adaptive proof against owner A.
                consume_publications(true);
                const bool known_publication_at_admission = latest_v2_lineage.known_depth() &&
                                                            depth_estimator->publication_is_current(latest_v2_lineage.estimate().publication);
                const auto observation_timestamp_us =
                  host_sbs_observation_timestamp_us(
                    matched_candidate_slot->attribution.captured_at
                  );
                const auto optional_work = models::select_depth_optional_work_mode(
                  matched_candidate_slot->attribution.observed_authority.interactive_move_size,
                  snapshot_debug_inputs,
                  matched_candidate_slot->attribution.subtitle_context_eligible ||
                    matched_candidate_slot->attribution.subtitle_context_suppression_eligible
                );
                const bool depth_only_history =
                  dedup_gate_open && matched_candidate_slot->attribution.inference_ddup_damage &&
                  !matched_candidate_slot->attribution.observed_authority.interactive_move_size &&
                  optional_work == models::depth_optional_work_mode_e::disable_subtitle;
                const bool authorize_gpu_undecided =
                  dedup_gate_open &&
                  models::depth_optional_work_allows_gpu_undecided(
                    optional_work, depth_only_history
                  ) &&
                  !inference_pending &&
                  gpu_undecided_candidate_matches(
                    *matched_candidate_slot,
                    live_source_desc
                  );
                // The detector decides joint depth/subtitle reuse for eligible dirty content.
                // Native USER32 suppression remains an explicit eligibility rule.
                const auto adaptive_request =
                  models::make_gpu_adaptive_request(
                    frame_id,
                    authorize_gpu_undecided,
                    latest_v2_lineage.receipt ? &*latest_v2_lineage.receipt : nullptr,
                    depth_estimator->publication_is_current(latest_v2_lineage.estimate().publication) ?
                      latest_v2_lineage.estimate().publication :
                      models::host_sbs_gpu_completion_receipt::expected_t {},
                    observation_timestamp_us,
                    depth_only_history
                  );
#ifdef SUNSHINE_TESTS
                if (!test_native_pacing_gate_configured && test_native_pacing.held_source_stamp_ticks &&
                    current_source_timestamp &&
                    current_source_timestamp->time_since_epoch().count() ==
                      test_native_pacing.held_source_stamp_ticks->load(std::memory_order_acquire)) {
                  models::joined_completion_test_gate::configuration_t gate {
                    .enabled = true,
                    .target_frame_id = frame_id,
                    .hold_after_callback_enter_ms = test_native_pacing.hold_ms,
                  };
                  if (!depth_estimator->configure_joined_completion_gate_for_native_pacing_test(gate)) {
                    BOOST_LOG(error) << "Could not arm the native pacing test's joined depth gate";
                    return -1;
                  }
                  test_native_pacing_gate_configured = true;
                }
#endif
                auto submitted = depth_estimator->estimate_depth(
                  matched_candidate_slot->analysis_input_srv(),
                  input_color_space,
                  frame_id,
                  snapshot_debug_inputs,
                  matched_candidate_slot->attribution.depth_input_region,
                  optional_work,
                  adaptive_request,
                  matched_candidate_slot->attribution.subtitle_context_generation,
                  matched_candidate_slot->attribution.subtitle_line_policy
                );
                if (completion_finalized_before_admission &&
                    !depth_estimator->publication_is_current(step_observation.publication)) {
                  // Admission can reset subtitle state without changing the depth domain.
                  // Borrowed views from the early completion no longer authorize this draw.
                  clear_matched_selection();
                  matched_presentation_cache.reset();
                  latest_v2_lineage.reset();
                  completion_finalized_before_admission = false;
                }
                if (
                  diagnostics_enabled &&
                  matched_candidate_slot->attribution.depth_input_region.is_video_region() &&
                  (submitted.submission.inference_enqueued ||
                   submitted.submission.gpu_undecided_transaction_enqueued)
                ) {
                  ++matched_stats_roi_direct_inputs;
                }
                completion_finalized_before_admission = handoff.accept_submission(
                  std::move(submitted),
                  completion_finalized_before_admission
                );
                if (
                  step_observation.completed_frame_valid &&
                  !completion_finalized_before_admission
                ) {
                  if (auto *completed = consume_matched_completion(step_observation)) {
                    adopt_matched_completion(*completed);
                  } else {
                    matched_render_capture = nullptr;
                    retained_render_attribution = nullptr;
                  }
                }
                const bool gpu_undecided_transaction_enqueued =
                  depth_step.submission.gpu_undecided_transaction_enqueued;
                const bool force_infer_transaction_enqueued = depth_step.submission.inference_enqueued;
                const bool depth_transaction_enqueued =
                  force_infer_transaction_enqueued || gpu_undecided_transaction_enqueued;
                const auto adaptive_submission_class =
                  models::classify_gpu_adaptive_submission(
                    frame_id,
                    adaptive_request,
                    force_infer_transaction_enqueued,
                    gpu_undecided_transaction_enqueued
                  );
                if (
                  depth_transaction_enqueued &&
                  adaptive_submission_class ==
                    models::gpu_adaptive_submission_class_e::invalid
                ) {
                  BOOST_LOG(error)
                    << "Host SBS estimator returned an invalid adaptive submission class; "sv
                       "revoking adaptive reuse authority."sv;
                  latest_v2_lineage.reset();
                }
                if (force_infer_transaction_enqueued) {
                  if (diagnostics_enabled && !snapshot_debug_inputs && !known_publication_at_admission) {
                    ++matched_stats_force_missing_publication;
                  }
                  if (
                    !matched_candidate_slot->attribution.inference_content_timestamp ||
                    (matched_candidate_slot->attribution.depth_input_region.is_video_region() &&
                     !matched_candidate_slot->attribution.inference_ddup_damage)
                  ) {
                    latest_v2_lineage.reset();
                  }
                  // The forced full-source submission has retired the old ROI authority. Return
                  // subsequent observations to ordinary causality and route selection.
                  depth_authority_reprocess_pending = false;
                }
                if (depth_transaction_enqueued) {
                  pending_matched_frame = matched_candidate_slot;
                  matched_candidate_slot->attribution.gpu_undecided_transaction =
                    gpu_undecided_transaction_enqueued;
                }

                if (test_observer.record) {
                  converter_test_observer::event_t e; e.kind=converter_test_observer::event_e::admission;
                  e.force_infer=force_infer_transaction_enqueued; e.gpu_undecided=gpu_undecided_transaction_enqueued;
                  e.optional_work=static_cast<std::uint32_t>(optional_work);
                  emit_test_observation(std::move(e),&matched_candidate_slot->attribution,nullptr,matched_candidate_slot->srv.get(),frame_id);
                }
                // Give the just-submitted matched-frame unit one immediate joined query. For a
                // conditional transaction this queries only root completion; it never reads the
                // GPU-owned reuse/infer decision.
                // Repeated polling is allowed only inside encode-loop cadence slack; a timeout
                // leaves the sole pending inference, its matched slot, and the prior completion
                // untouched. No older busy admission can enter this path because estimate_depth()
                // must have successfully enqueued this candidate in the current call.
                if (
                  depth_transaction_enqueued &&
                  pending_matched_frame == matched_candidate_slot && !snapshot_debug_inputs
                ) {
                  const auto poll_started = std::chrono::steady_clock::now();
                  if (diagnostics_enabled) {
                    ++matched_stats_same_frame_poll_submissions;
                  }
                  const auto poll_plan = detail::host_sbs_same_frame_poll_plan(
                    true,
                    false,
                    next_encode_target,
                    poll_started,
                    prior_poll_plan
                  );
                  auto polled = poll_pending_depth(&poll_plan);
#ifdef SUNSHINE_TESTS
                  record_native_poll(poll_plan, polled, false);
#endif
                  const double waited_ms = std::chrono::duration<double, std::milli>(
                                             polled.wait_duration
                  )
                                             .count();
                  if (diagnostics_enabled) {
                    matched_stats_same_frame_poll_queries += polled.query_count;
                    if (poll_plan.eligible) {
                      const double planned_budget_ms =
                        std::chrono::duration<double, std::milli>(poll_plan.budget).count();
                      ++matched_stats_same_frame_poll_plans;
                      matched_stats_same_frame_poll_budget_sum_ms += planned_budget_ms;
                      matched_stats_same_frame_poll_budget_max_ms = std::max(
                        matched_stats_same_frame_poll_budget_max_ms,
                        planned_budget_ms
                      );
                    }
                    if (polled.wait_attempted) {
                      ++matched_stats_same_frame_poll_attempts;
                      matched_stats_same_frame_poll_wait_sum_ms += waited_ms;
                      matched_stats_same_frame_poll_wait_max_ms = std::max(
                        matched_stats_same_frame_poll_wait_max_ms,
                        waited_ms
                      );
                      if (perf) {
                        sbs_perf::add_sample_ms("same_frame_poll_wait", waited_ms);
                      }
                    }
                  }

                  const auto completion = detail::host_sbs_same_frame_completion(
                    polled.ready,
                    polled.result.observation.completed_frame_valid,
                    polled.result.observation.completed_frame_id,
                    matched_candidate_slot->attribution.frame_id
                  );
                  const auto poll_outcome = detail::host_sbs_same_frame_poll_outcome(
                    poll_plan.eligible,
                    polled.wait_attempted,
                    polled.timed_out,
                    completion
                  );
                  same_frame_poll_missed_for_current_draw =
                    completion == detail::host_sbs_same_frame_completion_e::keep_pending;
                  if (diagnostics_enabled) {
                    switch (poll_outcome) {
                      case detail::host_sbs_same_frame_poll_outcome_e::immediate_hit:
                        ++matched_stats_same_frame_immediate_hits;
                        break;
                      case detail::host_sbs_same_frame_poll_outcome_e::wait_hit:
                        ++matched_stats_same_frame_poll_hits;
                        switch (detail::host_sbs_same_frame_poll_hit_bucket(
                          polled.wait_duration
                        )) {
                          case detail::host_sbs_same_frame_poll_hit_bucket_e::within_2_ms:
                            ++matched_stats_same_frame_poll_hits_le_2_ms;
                            break;
                          case detail::host_sbs_same_frame_poll_hit_bucket_e::between_2_and_2_5_ms:
                            ++matched_stats_same_frame_poll_hits_2_to_2_5_ms;
                            break;
                          case detail::host_sbs_same_frame_poll_hit_bucket_e::between_2_5_and_3_ms:
                            ++matched_stats_same_frame_poll_hits_2_5_to_3_ms;
                            break;
                          case detail::host_sbs_same_frame_poll_hit_bucket_e::over_3_ms:
                            ++matched_stats_same_frame_poll_hits_over_3_ms;
                            break;
                        }
                        if (perf) {
                          sbs_perf::add_sample_ms(
                            "same_frame_poll_hit_wait",
                            waited_ms
                          );
                        }
                        break;
                      case detail::host_sbs_same_frame_poll_outcome_e::cadence_ineligible_busy:
                        ++matched_stats_same_frame_poll_cadence_ineligible_busy;
                        break;
                      case detail::host_sbs_same_frame_poll_outcome_e::eligible_timeout:
                        ++matched_stats_same_frame_poll_timeouts;
                        if (poll_plan.limited_by_hard_cap) {
                          ++matched_stats_same_frame_poll_hard_cap_timeouts;
                        } else {
                          ++matched_stats_same_frame_poll_cadence_timeouts;
                        }
                        if (perf) {
                          sbs_perf::add_sample_ms(
                            "same_frame_poll_timeout_wait",
                            waited_ms
                          );
                        }
                        break;
                      case detail::host_sbs_same_frame_poll_outcome_e::ready_failure:
                        ++matched_stats_same_frame_poll_failures;
                        break;
                      case detail::host_sbs_same_frame_poll_outcome_e::wait_unavailable_busy:
                        // Event-unavailable fallback made its required immediate query but could
                        // not spend the otherwise valid repeated-wait plan.
                        ++matched_stats_same_frame_poll_wait_unavailable_busy;
                        break;
                    }
                  }
                  if (
                    completion !=
                    detail::host_sbs_same_frame_completion_e::keep_pending
                  ) {
                    // finish_pending() rewrites the estimator's singleton V2 resources. The prior
                    // completion returned by estimate_depth() can no longer render after any ready
                    // current-frame query, including a ready failure or corrupt frame identity.
                    auto *completed_slot = consume_matched_completion(polled.result.observation);
                    clear_matched_selection();
                    completion_finalized_before_admission = false;
                    if (completed_slot) {
                      adopt_matched_completion(*completed_slot, &polled.result);
                    }
                  }
                }
              }
            }
          } else {
            mark_sbs_matched_copy_end(gpu_timer, false);
          }

          if (unchanged_content_submission_suppressed && depth_reuse_authorization.valid() && diagnostics_enabled) {
            ++matched_stats_content_skips;
            if (damage_guided_submission_suppressed) {
              ++matched_stats_damage_skips;
            }
          }

          // Cached fields alias the estimator's singleton V2 textures, so recompute eligibility
          // only after every path that could have consumed/normalized a completion above.
          const bool post_completion_cache_route_matches =
            latest_v2_lineage_route_matches_current(
              live_source_desc,
              input_color_space,
              live_presentation_authority()
            );
          const bool post_completion_cache_gate_open =
            dedup_gate_open && latest_v2_lineage.known_depth();
          const auto post_completion_cache_reuse_kind =
            post_completion_cache_gate_open && post_completion_cache_route_matches ?
              current_input_reuse_kind(latest_v2_lineage.attribution) :
              detail::host_sbs_ddup_reuse_proof_e::none;
          const auto post_completion_reuse_authorization =
            detail::select_host_sbs_cached_depth_reuse(
              post_completion_cache_reuse_kind,
              latest_v2_lineage.attribution.frame_id,
              frame_id
            );
          const bool post_completion_cache_render_allowed =
            detail::host_sbs_cached_geometry_render_allowed(
              post_completion_cache_gate_open && post_completion_cache_route_matches,
              host_sbs_renderer == models::host_sbs_renderer_e::parallax_v2,
              post_completion_reuse_authorization,
              latest_v2_lineage.attribution.frame_id,
              frame_id,
              latest_v2_lineage.estimate().shadow_base_final_parallax &&
                latest_v2_lineage.estimate().shadow_state
            );
          if (
            !render_attribution() && post_completion_cache_render_allowed &&
            (unchanged_content_submission_suppressed || depth_step.submission.inference_enqueued ||
             depth_step.submission.gpu_undecided_transaction_enqueued)
          ) {
            // All estimator work is finished for this draw. Borrow the current lineage on the
            // converter thread, and keep this call's submission and completion events separate.
            render_handoff = &latest_v2_lineage.completion;
            retained_render_attribution = &latest_v2_lineage.attribution;
            cached_current_color_warp = true;
            depth_reuse_authorization = post_completion_reuse_authorization;
          } else if (post_completion_cache_gate_open && force_fresh_current_color && render_attribution()) {
            const auto completed_reuse_authorization = detail::select_host_sbs_cached_depth_reuse(
              current_input_reuse_kind(*render_attribution()),
              step_observation.completed_frame_id,
              frame_id
            );
            if (completed_reuse_authorization.valid()) {
              cached_current_color_warp = true;
              depth_reuse_authorization = completed_reuse_authorization;
            }
          }

          // A completion retired before admission was already authenticated and retained so the
          // current proof could name it. All other completions pass through the same helpers here.
          if (!completion_finalized_before_admission && render_attribution()) {
            (void) validate_completed_slot();
          }
          if (
            render_attribution() && matched_candidate_slot &&
            pending_matched_frame == matched_candidate_slot &&
            (
              // Both operands are display-selected submission regions here. Their calibrated
              // coarse raster is intentional; no estimator publication participates in this
              // pending-domain transition check.
              !render_attribution()->depth_input_region.same_analysis_domain(
                matched_candidate_slot->attribution.depth_input_region
              ) ||
              render_attribution()->color_space != matched_candidate_slot->attribution.color_space
            )
          ) {
            // estimate_depth may retire domain A while enqueueing the first frame of domain B in
            // the same call. Once B is observed, do not resurrect or repeat A; show current flat
            // color until the exact B completion arrives and rearms its scene camera.
            clear_matched_selection();
            if (completion_finalized_before_admission) {
              latest_v2_lineage.reset();
            }
            completion_finalized_before_admission = false;
          } else if (!completion_finalized_before_admission && render_attribution()) {
            (void) apply_completed_domain_and_renderer();
          }

          // A current submission made after an early completion may itself fail terminally, so
          // keep this post-submit guard even when the prior completion was already finalized.
          if (depth_estimator && depth_estimator->has_terminal_failure()) {
            const auto failed_renderer =
              models::fail_host_sbs_renderer_flat(host_sbs_renderer);
            if (failed_renderer != host_sbs_renderer) {
              clear_matched_selection();
              fail_depth_pipeline_flat();
              BOOST_LOG(error)
                << "Host SBS parallax-v2 producer reported a terminal CUDA/TensorRT failure; "sv
                   "this stream is now live flat identity with depth submissions disabled."sv;
            }
          }

          // Retain the latest completion that survived exact frame/region/route checks and the V2
          // authentication latch. This lineage aliases the estimator's singleton V2 resources but
          // does not itself authorize rendering; a typed current-frame reuse proof remains
          // separate.
          if (!completion_finalized_before_admission) {
            retain_completed_lineage();
          }

          // A busy CUDA stream may legitimately repeat one matched pair for a few source frames,
          // but NOT_READY is not proof of eventual progress. Bound live V2 by the source frame's
          // age; once stale, render current color through the V2 shader's flat-identity path while
          // continuing nonblocking queries so a later fresh completion can recover automatically.
          // Model owner frame and actual rendered RGB conversion are different on authorized
          // cached-current RGB reuse. Reject a late private B only after a newer source was drawn.
          if (const auto *attribution = render_attribution();
              attribution && !cached_current_color_warp &&
              !same_source(attribution->source_timestamp, current_source_timestamp) &&
              !drawn_source_order.permits(attribution->frame_id)) {
            clear_matched_selection();
          }
          const auto repeat_now = std::chrono::steady_clock::now();
          bool stale_v2_completion = false;
          if (render_attribution() && host_sbs_renderer == models::host_sbs_renderer_e::parallax_v2) {
            const auto &source = cached_current_color_warp ? current_source_timestamp : render_attribution()->source_timestamp;
            const auto captured_at = cached_current_color_warp ? reuse_now : render_attribution()->captured_at;
            if (!models::host_sbs_matched_completion_is_current(same_source(source, current_source_timestamp), repeat_now - captured_at)) {
              stale_v2_completion = true;
              clear_matched_selection();
            }
          }
          if (render_attribution()) {
            publish_window_region_transition(*render_attribution());
          }
          const auto repeat_source_age =
            matched_presentation_cache.source_age(repeat_now);
          const auto repeat_hold_age =
            matched_presentation_cache.hold_age(current_source_timestamp, repeat_now);
          const bool output_source_unchanged =
            matched_presentation_cache.source_matches(current_source_timestamp);
          // Packed-presentation retention is deliberately weaker than reusable V2 lineage: it
          // may replay only the already-rendered pixels. Recheck the live transfer/source domain
          // here even though prepare_matched_frame() also invalidates mismatched domains, so no route
          // or reset race can turn presentation continuity into geometry authority.
          const bool packed_presentation_route_matches =
            matched_presentation_cache.route_matches(
              live_source_desc.Width,
              live_source_desc.Height,
              input_color_space,
              depth_authority_reprocess_pending,
              producer_terminal,
              subtitle_context.generation(),
              live_presentation_authority()
            );
          const bool can_retain_shutter_pair = rgb && rgb->destination == rgb_destination_e::shutter_with_pair;
          const bool repeat_matched_output = (!packed_output_in_shutter_pair || can_retain_shutter_pair) &&
                                             drawn_source_order.permits(matched_presentation_cache.pixel_source_frame_id()) &&
                                             models::host_sbs_should_repeat_packed_presentation(
                                               host_sbs_renderer,
                                               render_attribution() != nullptr,
                                               matched_presentation_cache.has_fresh_pixels(),
                                               packed_presentation_route_matches,
                                               repeat_hold_age,
                                               output_source_unchanged
                                             );
          const bool output_conversion_required =
            host_sbs_encoder_input_state.conversion_required(
              repeat_matched_output,
              rgb != nullptr
            );
          const bool retain_rgb_pair = can_retain_shutter_pair && repeat_matched_output;
          if (!repeat_matched_output && !acquire_current_capture()) {
            return -1;
          }
          // The optional warp+luma MRT and the later chroma/cursor draws must address the same
          // selected allocation. Ordinary packed repeats do not select or invalidate an input.
          if (output_conversion_required && !select_native_output()) {
            return -1;
          }
          const bool v2_renderer_selected = host_sbs_renderer == models::host_sbs_renderer_e::parallax_v2;
          const bool v2_renderer_failed_flat = host_sbs_renderer == models::host_sbs_renderer_e::failed_flat;
          const auto &render_step = render_handoff->step();
          const auto &est = render_step.observation;
          const bool v2_result_authenticated = render_attribution() && render_handoff->authenticate_structure();
          const bool v2_live_warp_selected =
            v2_renderer_selected && v2_result_authenticated && sbs_reprojection_v2_live_ps &&
            est.shadow_base_final_parallax && est.shadow_state;
          const auto geometry = select_sbs_draw_geometry(
            matched_render_capture, retained_render_attribution, v2_live_warp_selected ? &est : nullptr
          );
          const auto *draw_source_attribution = geometry.source_attribution;
          // Match geometry to its true submission while resolving colour and its presentation
          // identity independently. Reuse binds the current capture.
          const bool uses_current_colour = !geometry.source_capture || cached_current_color_warp;
          const auto draw_colour = uses_current_colour ? sbs_draw_colour_t {
            img_ctx->encoder_input_res.get(), input_color_space, current_source_timestamp,
            current_content_timestamp, reuse_now,
            frame_id,
          } : sbs_draw_colour_t {
            geometry.source_capture->srv.get(),
            draw_source_attribution->color_space, draw_source_attribution->source_timestamp, draw_source_attribution->content_timestamp,
            draw_source_attribution->captured_at,
            !same_source(draw_source_attribution->source_timestamp, current_source_timestamp) ?
              draw_source_attribution->frame_id : frame_id,
          };
          converted_content_timestamp = repeat_matched_output ? matched_presentation_cache.content_timestamp() :
                                                                draw_colour.rendered_content_timestamp();
          const bool v2_repeat_age_expired =
            stale_v2_completion ||
            (host_sbs_renderer == models::host_sbs_renderer_e::parallax_v2 &&
             !render_attribution() && matched_presentation_cache.has_fresh_pixels() &&
             !output_source_unchanged &&
             repeat_hold_age > models::host_sbs_v2_max_matched_repeat_age);
          using presentation_clock_t = detail::host_sbs_packed_presentation_clock_t;
          using presentation_output_e = presentation_clock_t::output_e;
          const auto presentation_output = repeat_matched_output ? presentation_output_e::packed :
                                           v2_live_warp_selected ? presentation_output_e::completed :
                                                                   presentation_output_e::flat;
          if (!v2_live_warp_selected && v2_renderer_selected && render_attribution() && v2_live_warp_seen && !v2_live_warp_loss_logged) {
            BOOST_LOG(error)
              << "Host SBS lost authenticated live-warp authority after parallax-v2 was active"
              << " (result_authenticated=" << (v2_result_authenticated ? "true" : "false")
              << ", live_shader=" << (sbs_reprojection_v2_live_ps ? "present" : "missing")
              << ", base_field=" << (est.shadow_base_final_parallax ? "present" : "missing")
              << ", state=" << (est.shadow_state ? "present" : "missing")
              << "); rendering current-frame flat identity. Further warnings are suppressed for "
                 "this stream.";
            v2_live_warp_loss_logged = true;
          }
          if (cached_current_color_warp && v2_live_warp_selected && diagnostics_enabled) {
            switch (depth_reuse_authorization.kind) {
              case detail::host_sbs_depth_reuse_kind_e::exact_content:
                ++matched_stats_content_reuses;
                break;
              case detail::host_sbs_depth_reuse_kind_e::exact_roi_damage:
                ++matched_stats_content_reuses;
                ++matched_stats_damage_reuses;
                break;
              case detail::host_sbs_depth_reuse_kind_e::none:
                break;
            }
          }
          // The same fullscreen pass performs either authenticated reprojection or current-frame
          // flat identity. Time both; a packed repeat performs neither draw.
          const bool timing_has_sbs_composite = !repeat_matched_output;
          mark_sbs_warp_start(gpu_timer, timing_has_sbs_composite);
          ID3D11ShaderResourceView *final_sbs_srv = nullptr;
          ID3D11Texture2D *final_sbs_texture = nullptr;
          bool final_sbs_is_linear = sbs_intermediate_is_linear;
          bool final_sbs_has_p010_y = false;
          bool rendered_directly_to_rgb = false;
          std::array<ID3D11ShaderResourceView *, 7u> dump_warp_inputs {};
          ID3D11Buffer *dump_subtitle_constants = nullptr;
          bool dump_subtitle_render_enabled = false;
          std::optional<sbs_debug::actual_draw_evidence> dump_actual_draw;
          if (snapshot_debug_inputs) {
            dump_actual_draw.emplace();
            dump_actual_draw->requested = true;
          }
          if (diagnostics_enabled) {
            if (repeat_matched_output) {
              ++matched_stats_output_packed_repeat;
            } else if (geometry.region) {
              ++matched_stats_output_warped_new;
            } else {
              ++matched_stats_output_flat;
              if (
                same_frame_poll_missed_for_current_draw &&
                pending_matched_frame
              ) {
                // This should remain zero once a valid packed presentation exists. Keep the
                // counter explicit so a route/reset invalidation can be distinguished from the
                // former adaptive timeout flash in live logs.
                ++matched_stats_output_flat_pending_miss;
              }
            }
          }
          if (repeat_matched_output) {
            if (!retain_rgb_pair) {
              final_sbs_srv = sbs_intermediate_srv.get();
              final_sbs_texture = sbs_intermediate_texture.get();
            }
            if (diagnostics_enabled) {
              ++matched_stats_repeats;
            }
          } else {
            // Before the first matched completion, provide a flat current-frame SBS image. An
            // ordinary completion renders its matched color; an authorized cache reuse
            // deliberately renders current color/cursor with authenticated cached geometry.
            if (!render_attribution()) {
              handoff.reset();
            }

            // Output transfer follows the color frame actually rendered. Matched inference may
            // complete later than the current capture; repeated output retains the transfer of
            // the packed texture already stored in the intermediate.
            const bool render_input_is_linear = models::input_color_space_is_linear(draw_colour.color_space);

            if (rgb && rgb->destination != rgb_destination_e::backbuffer &&
                rgb->linear == render_input_is_linear &&
                !snapshot_debug_inputs && rgb->texture) {
              D3D11_TEXTURE2D_DESC target_desc {};
              D3D11_RENDER_TARGET_VIEW_DESC target_view_desc {};
              rgb->texture->GetDesc(&target_desc);
              rgb->target->GetDesc(&target_view_desc);
              const auto packed_format = render_input_is_linear ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_B8G8R8A8_UNORM;
              rendered_directly_to_rgb = target_desc.Format == packed_format && target_view_desc.Format == packed_format &&
                                         target_view_desc.ViewDimension == D3D11_RTV_DIMENSION_TEXTURE2D &&
                                         target_desc.ArraySize == 1 && target_desc.MipLevels == 1 &&
                                         target_desc.SampleDesc.Count == 1 &&
                                         sbs_viewport.TopLeftX == 0 && sbs_viewport.TopLeftY == 0 &&
                                         sbs_viewport.Width == static_cast<float>(target_desc.Width) &&
                                         sbs_viewport.Height == static_cast<float>(target_desc.Height);
            }

            // DDup reveals the current capture transfer only after the first real frame, while
            // asynchronous depth completion can select an older buffered color frame. Size the
            // packed target from the frame we are about to render, not from the newest capture.
            // Repeated output deliberately skips this path so a transfer transition cannot
            // reallocate and erase the last completed SBS image before it is presented.
            if (!rendered_directly_to_rgb && !ensure_sbs_intermediate_storage(render_input_is_linear)) {
              return -1;
            }

            if (!update_sbs_constant_buffer(
                  sbs_content_scale_x,
                  sbs_content_scale_y,
                  geometry.region,
                  geometry.field_content
                )) {
              return -1;
            }
            models::host_sbs_subtitle_render_constants_t subtitle_constants;
            if (geometry.region) {
              const auto &region = *geometry.region;
              subtitle_constants.enabled = draw_source_attribution->subtitle_context_eligible &&
                                               subtitle_context.matches(draw_source_attribution->subtitle_context_generation) &&
                                               !current_authority.interactive_move_size && geometry.has_subtitle_inputs() ?
                                             1u :
                                             0u;
              subtitle_constants.color_mode = static_cast<std::uint32_t>(draw_colour.color_space);
              subtitle_constants.analysis_generation_low = static_cast<std::uint32_t>(region.analysis_generation);
              subtitle_constants.analysis_generation_high = static_cast<std::uint32_t>(region.analysis_generation >> 32u);
              subtitle_constants.source_left = region.left;
              subtitle_constants.source_top = region.top;
              subtitle_constants.source_width = region.width();
              subtitle_constants.source_height = region.height();
            }
            if (!update_sbs_subtitle_constant_buffer(subtitle_constants)) {
              return -1;
            }

            const bool subtitle_pin_candidate = selected_matte_ready && geometry.selected_matte &&
              geometry.selected_receipt && selected_shared_warp_programs.direct_glyph_completed_ready();
            // A frame taken for a Dump 3D package is drawn without the text pin: the package
            // describes the Host geometry, and its format has no place for this route.
            const bool text_pin_candidate = text_game_ai_active && selected_shared_warp_programs.text_game_ready() &&
              v2_live_warp_selected && draw_colour.srv && !dump_actual_draw;
            Microsoft::WRL::ComPtr<ID3D11Predicate> predicate;
            if (subtitle_pin_candidate || text_pin_candidate) {
              BOOL value = FALSE;
              device_ctx->GetPredication(&predicate, &value);
            }
            // A text game's own text takes the place of the picture-based protection while the
            // picture shows it in the colours its helper reports. Whenever the helper reports
            // text, that is checked against the picture; until it holds, the draw stays as it
            // was, with the picture-based protection where that is in force.
            std::array<UINT, 4u> text_pin_shape {};
            bool text_pin_prepared = false;
            if (text_pin_candidate && !predicate) {
              Microsoft::WRL::ComPtr<ID3D11Resource> colour_resource;
              Microsoft::WRL::ComPtr<ID3D11Texture2D> colour_texture;
              draw_colour.srv->GetResource(&colour_resource);
              if (SUCCEEDED(colour_resource.As(&colour_texture))) {
                D3D11_TEXTURE2D_DESC colour_desc {};
                colour_texture->GetDesc(&colour_desc);
                text_pin_prepared = text_game_pin.prepare(device.get(), device_ctx.get(), colour_desc, default_source_sdr_white_scrgb, text_pin_shape);
              }
            }
            bool text_pin_selected = text_pin_prepared && text_game_pin.text_shown();
            if (text_game_ai_active && !predicate) {
              text_game_pin.note_protection(text_pin_candidate, text_pin_prepared, text_pin_selected, std::chrono::steady_clock::now());
            }
            // The shared pin is prepared only for an enabled subtitle band (below). Without one
            // this is the ordinary draw from the start, so it keeps the fused P010 luma target
            // instead of giving it up for a pin that is then never prepared.
            auto *selected_receipt = subtitle_pin_candidate && subtitle_constants.enabled == 1u &&
                                         !text_pin_selected && !predicate ?
                                       geometry.selected_receipt : nullptr;
            bool selected_draw = selected_receipt != nullptr;
            // Only an authenticated V2 stream selects the production warp. Awaiting, stale, and
            // terminal failure states use the independent current-frame identity shader, so V2
            // shader authentication/device-creation failures cannot freeze or end the stream.
            auto *native_luma = rgb ? nullptr : active_output_views().y_or_yuv.get();
            const bool p010_y_mrt_authorized =
              detail::host_sbs_p010_y_mrt_eligible(
                v2_live_warp_selected,
                render_input_is_linear,
                output_is_hdr,
                format == DXGI_FORMAT_P010,
                rgb == nullptr,
                snapshot_debug_inputs,
                sbs_reprojection_v2_p010_y_ps != nullptr &&
                  stream_gamma_mode_ == ::video::stream_gamma_mode_e::windows_default,
                native_luma != nullptr
              );
            const bool p010_y_mrt_selected =
              !selected_draw && !text_pin_selected && p010_y_mrt_authorized && qualify_p010_y_mrt_targets();
            if (p010_y_mrt_selected && diagnostics_enabled) {
              ++matched_stats_output_p010_y_mrt;
            }
            if (p010_y_mrt_selected) {
              ID3D11Buffer *matrix = color_matrix.get();
              device_ctx->PSSetConstantBuffers(0, 1, &matrix);
            }
            host_sbs_v2_draw_command_t draw_command {
              .render_targets = {
                rendered_directly_to_rgb ? rgb->target : sbs_intermediate_rtv.get(),
                p010_y_mrt_selected ? native_luma : nullptr,
              },
              .render_target_count = p010_y_mrt_selected ? 2u : 1u,
              .vertex_shader = sbs_reprojection_vs.get(),
              .pixel_shader = p010_y_mrt_selected ? sbs_reprojection_v2_p010_y_ps.get() : geometry.pixel_shader,
              .viewport = sbs_viewport,
              .sampler = sampler_linear.get(),
              .shader_resources = {
                draw_colour.srv, geometry.base, geometry.state, selected_draw ? geometry.selected_matte : nullptr,
                geometry.display_band, geometry.condition_params, geometry.contour_identity,
              },
              .geometry_constants = sbs_reprojection_cbuffer.get(),
              .subtitle_constants = sbs_subtitle_render_cbuffer.get(),
            };
            // One fresh pin field uses the current actual draw source. The immutable draw
            // keeps the selected receipt/current RGB checks and performs no RGB reconstruction.
            namespace shared_warp = models::host_sbs_shared_ui_warp;
            shared_warp::views_t current_shared_warp;
            bool shared_warp_selected = false;
            // Preparation and the pre-Draw observer derive the same literal request from
            // the recorder command, rather than maintaining a second copy of its operands.
            const auto shared_request_for_draw = [&] {
              shared_warp::request_t request;
              std::copy(draw_command.shader_resources.begin(), draw_command.shader_resources.end(), request.draw_srvs.begin());
              request.draw_srvs[7u] = selected_receipt;
              request.geometry_constants = draw_command.geometry_constants;
              request.subtitle_constants = draw_command.subtitle_constants;
              request.sampler = draw_command.sampler;
              request.materialize_inactive = snapshot_debug_inputs;
              return request;
            };
            if (selected_draw && subtitle_constants.enabled == 1u) {
              const auto request = shared_request_for_draw();
              const auto program = selected_shared_warp_programs.direct_glyph_completed_program();
              std::string shared_error;
              if (selected_shared_warp.prepare_subtitle(
                    device.get(), device_ctx.get(), program, request,
                    {subtitle_constants.source_width, subtitle_constants.source_height, subtitle_constants.source_top},
                    shared_error
                  ) && selected_shared_warp.matches(device_ctx.get(), request)) {
                current_shared_warp = selected_shared_warp.views();
                shared_warp_selected = true;
                draw_command.pixel_shader = program.pixel;
              }
            }
            if (!shared_warp_selected) {
              // Shared programs being ready does not guarantee THIS frame's
              // preparation. This draw uses ordinary authenticated Base geometry
              // and current RGB, with no native render mask or subtitle permission.
              // Finalize before receipt scopes, recorder binding and evidence.
              draw_command.shader_resources[3u] = nullptr;
              selected_receipt = nullptr;
              selected_draw = false;
              if (subtitle_constants.enabled != 0u) {
                subtitle_constants.enabled = 0u;
                if (!update_sbs_subtitle_constant_buffer(subtitle_constants)) {
                  return -1;
                }
              }
            }
            shared_warp::views_t text_pin_views;
            if (text_pin_prepared) {
              shared_warp::request_t request;
              for (std::size_t slot = 0u; slot < 3u; ++slot) {
                request.draw_srvs[slot] = draw_command.shader_resources[slot];
              }
              request.geometry_constants = draw_command.geometry_constants;
              request.subtitle_constants = draw_command.subtitle_constants;
              request.sampler = draw_command.sampler;
              const auto program = selected_shared_warp_programs.text_game_program();
              auto text_pin_draw = text_game_pin.queue(device.get(), device_ctx.get(), program, request, text_pin_shape, text_pin_selected, diagnostics_enabled);
              text_pin_selected = text_pin_draw.selected;
              if (text_pin_selected) {
                text_pin_views = std::move(text_pin_draw.views);
                draw_command.pixel_shader = program.pixel;
              }
            }
            if (shared_warp_selected || text_pin_selected) {
              const auto &pin = shared_warp_selected ? current_shared_warp : text_pin_views;
              draw_command.shared_pin = host_sbs_v2_shared_pin_bindings_t {
                {pin.weights.Get(), pin.field.Get(), pin.rows.Get()},
                pin.geometry.Get(),
                selected_receipt,
              };
            }
            bool draw_recorded = false;
            {
              auto observe_dump_bindings = [&](ID3D11DeviceContext *observed_context) noexcept {
                if (!dump_actual_draw) {
                  return;
                }
                try {
                  // Observe the recorder's actual operands before Draw and cleanup. Strong references survive unbinding; maybe_dump below
                  // issues their GPU copies before the next conversion can reuse these buffers.
                  std::array<Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>, 15u> actual;
                  ID3D11ShaderResourceView *raw[15] {};
                  observed_context->PSGetShaderResources(0u, 15u, raw);
                  for (std::size_t i = 0u; i < actual.size(); ++i) {
                    actual[i].Attach(raw[i]);
                  }
                  Microsoft::WRL::ComPtr<ID3D11PixelShader> actual_shader;
                  observed_context->PSGetShader(actual_shader.GetAddressOf(), nullptr, nullptr);
                  std::array<Microsoft::WRL::ComPtr<ID3D11Buffer>, 5u> constants;
                  ID3D11Buffer *raw_constants[5] {};
                  observed_context->PSGetConstantBuffers(2u, 5u, raw_constants);
                  for (std::size_t i = 0u; i < constants.size(); ++i) {
                    constants[i].Attach(raw_constants[i]);
                  }
                  auto &e = *dump_actual_draw;

                  e.conversion_frame = frame_id;
                  e.model_frame = draw_source_attribution ? draw_source_attribution->frame_id : 0u;
                  e.observation_us = draw_source_attribution ? host_sbs_observation_timestamp_us(draw_source_attribution->captured_at) : 0u;
                  e.selected_shader = selected_draw && actual_shader.Get() == draw_command.pixel_shader;
                  e.shared_pin_selected = e.selected_shader && shared_warp_selected;
                  e.pixel_shader_address = reinterpret_cast<std::uintptr_t>(actual_shader.Get());
                  if (e.shared_pin_selected) {
                    e.program = shared_warp::renderer_kind;
                    e.pixel_shader_closure = models::host_sbs_shader_cache::shared_ui_warp_completed_source_closure_sha256;
                  } else if (actual_shader.Get() == sbs_reprojection_v2_live_ps.get()) {
                    e.program = "ordinary-v2";
                    e.pixel_shader_closure = sbs_reprojection_v2_live_source_closure_sha256;
                  } else if (actual_shader.Get() == sbs_flat_identity_ps.get()) {
                    e.program = "flat-identity";
                  } else {
                    e.program = "other-observed-draw";
                  }
                  e.matte = actual[3];
                  e.selected_receipt = actual[7];
                  if (e.shared_pin_selected) {
                    e.shared_pin_weights = actual[12];
                    e.shared_pin_field = actual[13];
                    e.shared_pin_rows = actual[14];
                    e.shared_pin_geometry = constants[4];
                    e.shared_pin_source_closure = e.pixel_shader_closure;
                    e.shared_pin_model_prepare_closure = selected_shared_warp_programs.model_prepare_closure();
                  }
                  e.geometry_constants = constants[0];
                  e.subtitle_constants = constants[1];
                  e.owner_serial = e.shared_pin_selected ? current_shared_warp.serial : 0u;
                  e.colour_key = {frame_id, draw_source_attribution ? draw_source_attribution->frame_id : 0u, draw_source_attribution ? static_cast<std::uint64_t>(draw_colour.captured_at.time_since_epoch().count()) : 0u, static_cast<std::uint64_t>(subtitle_constants.color_mode)};
                  e.analysis_source_rect = {subtitle_constants.source_left, subtitle_constants.source_top, subtitle_constants.source_width, subtitle_constants.source_height};
                  e.analysis_shape = e.shared_pin_selected ? current_shared_warp.shape : std::array<std::uint32_t, 4u> {};
                  e.context_address = reinterpret_cast<std::uintptr_t>(observed_context);
                  Microsoft::WRL::ComPtr<ID3D11Resource> source_resource;
                  if (actual[0]) {
                    actual[0]->GetResource(&source_resource);
                  }
                  e.actual_role_pointers = {reinterpret_cast<std::uintptr_t>(actual[0].Get()), reinterpret_cast<std::uintptr_t>(source_resource.Get()), reinterpret_cast<std::uintptr_t>(actual[3].Get()), reinterpret_cast<std::uintptr_t>(actual[7].Get()), reinterpret_cast<std::uintptr_t>(actual[4].Get()), reinterpret_cast<std::uintptr_t>(actual[6].Get()), e.context_address};

                  e.captured = true;
                  e.unavailable_reason.clear();
                } catch (...) {
                  if (dump_actual_draw) {
                    dump_actual_draw->release_resources();
                    dump_actual_draw->captured = false;
                  }
                }
              };
              if (snapshot_debug_inputs) {
                using observer_t = decltype(observe_dump_bindings);
                draw_command.bound_draw_observer_state = &observe_dump_bindings;
                draw_command.observe_bound_draw = +[](void *state, ID3D11DeviceContext *context) noexcept {
                  (*static_cast<observer_t *>(state))(context);
                };
              }
              draw_recorded = record_host_sbs_v2_draw(device_ctx.get(), draw_command);
              // The diagnostic draws below reuse draw_command after this stack observer has
              // expired. Never let them call a dangling state.
              draw_command.observe_bound_draw = nullptr;
              draw_command.bound_draw_observer_state = nullptr;
              if (draw_recorded && shared_warp_selected && dump_actual_draw && dump_actual_draw->captured) {
                // Requested-only diagnostics reuse this exact field/header/receipt before the
                // owner can be queued again. They do not allocate or run on ordinary draws.
                (void) render_shared_ui_debug_geometry(draw_command, *dump_actual_draw);
              }
            }
            if (!draw_recorded) {
              BOOST_LOG(error) << "Host SBS V2 draw operands are incomplete."sv;
              return -1;
            }
            if (presentation_output != presentation_output_e::flat) {
              v2_live_warp_seen = true;
            }
            final_sbs_has_p010_y = draw_recorded && p010_y_mrt_selected;
            if (snapshot_debug_inputs) {
              dump_warp_inputs = draw_command.shader_resources;
              dump_subtitle_constants = draw_command.subtitle_constants;
              dump_subtitle_render_enabled = subtitle_constants.enabled != 0u;
            }

            packed_output_in_shutter_pair = rendered_directly_to_rgb;
            if (rendered_directly_to_rgb) {
              final_sbs_texture = rgb->texture;
            } else {
              final_sbs_srv = sbs_intermediate_srv.get();
              final_sbs_texture = sbs_intermediate_texture.get();
              sbs_intermediate_is_linear = render_input_is_linear;
            }
            final_sbs_is_linear = render_input_is_linear;
            (void) drawn_source_order.note_drawn(draw_colour.pixel_source_frame_id);
            if (test_observer.record) {
              converter_test_observer::event_t e; e.kind=converter_test_observer::event_e::draw;
              e.flat=presentation_output == presentation_output_e::flat;
              e.cached_current_colour=cached_current_color_warp; e.watermark_before=test_watermark_at_entry;
              e.watermark_after=drawn_source_order.last_drawn(); e.actual_draw_pixel_source=draw_colour.pixel_source_frame_id;
              emit_test_observation(std::move(e),draw_source_attribution,&est,
                draw_colour.srv,frame_id, cached_current_color_warp ? &draw_colour : nullptr);
            }
            const bool cache_recorded_warp = presentation_output != presentation_output_e::flat &&
              models::host_sbs_packed_output_can_enter_presentation_cache(render_attribution() != nullptr, v2_live_warp_selected);
            if (cache_recorded_warp) {
              // Only packed PIXELS enter this continuity cache. Planned publication remains
              // unavailable to latest_v2_lineage / known_depth / estimator publication checks.
              matched_presentation_cache.publish(
                geometry.frame_id,
                draw_colour.pixel_source_frame_id,
                draw_colour.captured_at,
                draw_colour.source_timestamp,
                converted_content_timestamp,
                *geometry.region,
                draw_colour.color_space,
                geometry.field_width,
                geometry.field_height,
                draw_source_attribution->subtitle_context_generation,
                subtitle_constants.enabled != 0u,
                draw_source_attribution->observed_authority,
                current_source_timestamp,
                gamma_now
              );
            } else {
              // Flat identity cannot seed packed presentation continuity. This cache remains
              // completely separate from semantic depth/OCR/DDup publication authority.
              matched_presentation_cache.invalidate();
            }
          }

          // Observe the successfully recorded output, after a queued recorder failure may have
          // selected current flat and invalidated packed pixels. Invalidation retains the log
          // latch; the saved age decision must not be lost with that pixel cache.
          const auto presentation_timeout =
            matched_presentation_cache.observe_timeout(v2_repeat_age_expired, presentation_output);
          if (presentation_timeout == presentation_clock_t::timeout_event_e::expired) {
            BOOST_LOG(error)
              << "Host SBS packed presentation continuity exhausted after "sv
              << models::host_sbs_v2_max_matched_repeat_age.count()
              << " ms for a changed source; rendering live current-frame flat identity until "sv
                 "depth recovers."sv;
          } else if (presentation_timeout == presentation_clock_t::timeout_event_e::recovered) {
            BOOST_LOG(info) << "Host SBS V2 depth completion recovered; stereo warp resumed."sv;
          }

          auto *tiled_chroma = !rgb && output_conversion_required && final_sbs_is_linear &&
                                 !snapshot_debug_inputs ?
                                 active_output_views().p010_chroma.Get() :
                                 nullptr;
          mark_sbs_warp_end(gpu_timer);

          if (rgb) {
            // A shutter ring can keep its last complete packet. A rotating backbuffer still
            // needs pixels even when the packed image is an exact repeat.
            if (retain_rgb_pair) {
              rgb->retained_pair = true;
            } else if (!rendered_directly_to_rgb && !copy_rgb(final_sbs_texture, final_sbs_is_linear)) {
              draw_rgb(final_sbs_srv, final_sbs_is_linear);
            }
          } else if (output_conversion_required) {
            // Host SBS accepts identity-oriented sources only. Portrait is represented by actual
            // W < H display dimensions, so this final conversion never rotates a packed 2W frame.
            if (!tiled_chroma || !final_sbs_has_p010_y) {
              draw(
                final_sbs_srv,
                out_Y_or_YUV_viewport,
                out_UV_viewport,
                final_sbs_is_linear,
                final_sbs_has_p010_y,
                false, false, tiled_chroma ? 1 : 3
              );
            }
            if (tiled_chroma) {
              dispatch_p010_chroma_tiles(final_sbs_srv, tiled_chroma);
            }
            // Packed repeats retain this last converted input through native surface rotation.
            host_sbs_encoder_input_state.mark_converted();
          }
          end_sbs_gpu_timer(gpu_timer);
          if (test_observer.record) {
            converter_test_observer::event_t e; e.kind=repeat_matched_output ? converter_test_observer::event_e::packed_repeat:converter_test_observer::event_e::convert_end;
            e.current_capture_key = encoder_mutex_lock.owns_lock();
            e.prior_capture_key = prior_capture_lock && prior_capture_lock->owns_lock();
            e.native_planes_written = !rgb && output_conversion_required;
            e.repeat=repeat_matched_output; e.watermark_before=test_watermark_at_entry;
            e.actual_draw_pixel_source=repeat_matched_output ? 0u : draw_colour.pixel_source_frame_id; e.flat=presentation_output == presentation_output_e::flat;
            e.cached_current_colour=cached_current_color_warp; e.watermark_after=drawn_source_order.last_drawn();
            e.packed_valid=matched_presentation_cache.has_fresh_pixels();
            e.route_matches=packed_presentation_route_matches;
            emit_test_observation(std::move(e),draw_source_attribution,&est,
              repeat_matched_output ? nullptr:draw_colour.srv,frame_id,
              cached_current_color_warp ? &draw_colour : nullptr);
          }
          // Publication receipts already captured the diagnostic state. External Stats reads
          // that immutable CPU snapshot after output without scheduling any GPU work.
          publish_sbs_telemetry_after_output();

          // Close the live CPU sample before any requested dump work. A dump may perform lazy
          // shader compilation and submit a large diagnostic staging batch; none of that
          // represents production Host-SBS frame cost. GPU readiness polling is nonblocking
          // above; explicit dump CPU copies are excluded, while PNG/JSON serialization and
          // filesystem publication run on the worker.
          if (perf && !snapshot_debug_inputs) {
            const auto dt = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - perf_t0
            )
                              .count();
            sbs_perf::add_sample_ms("sbs_convert_cpu", dt);
            sbs_perf::tick();  // once per SBS frame: periodic min/mean/p50/p95/max summary
          }

          // Dump-only geometry repeats the exact completed-frame warp after all production timing
          // and output work. It is allocated and executed only when both immutable estimator
          // snapshots prove that this is the requested matched frame.
          if (v2_renderer_failed_flat && snapshot_debug_inputs) {
            BOOST_LOG(warning)
              << "Dump 3D request rejected: this parallax-v2 render stream is latched flat "sv
                 "after initialization/authentication failure, so no attributable V2 geometry "sv
                 "package exists."sv;
            sbs_dumper.reject_pending_request();
          }
          if (snapshot_debug_inputs && !repeat_matched_output) {
            const bool complete_dump_snapshot =
              matched_render_capture && dump_warp_inputs[1] && est.model_input_snapshot &&
              est.raw_model_depth_snapshot && est.raw_model_provenance &&
              est.shadow_candidate_parallax && est.shadow_vertical_majorant &&
              est.shadow_vertical_conditioned &&
              est.final_parallax() &&
              (est.subtitle_conditioning_diagnostics_current || est.subtitle_work_suppressed) &&
              v2_renderer_selected &&
              v2_result_authenticated;
            ID3D11ShaderResourceView *dump_depth_input_source =
              matched_render_capture ? matched_render_capture->analysis_input_srv() : nullptr;
            if (complete_dump_snapshot && est.input_region.is_video_region() && matched_render_capture->texture) {
              // This explicit diagnostic copy targets the exact completed slot after all live
              // timing has closed. The dumper revalidates the retained texture against the
              // completed region and owns the crop; ordinary ROI analysis allocates no crop.
              dump_depth_input_source = sbs_dumper.prepare_diagnostic_roi_crop(
                device.get(),
                device_ctx.get(),
                matched_render_capture->texture.get(),
                est.input_region
              );
              if (dump_depth_input_source && diagnostics_enabled) {
                ++matched_stats_roi_dump_copies;
              }
            }
            const bool dump_roi_source_ready =
              !complete_dump_snapshot || !est.input_region.is_video_region() ||
              dump_depth_input_source;
            // maybe_dump queues its staging copy on this immediate context before returning.
            // Release the dumper-owned crop on every later success, rejection, or exception path.
            auto release_diagnostic_roi_crop = util::fail_guard([this]() {
              sbs_dumper.release_diagnostic_roi_crop();
            });
            if (complete_dump_snapshot && !dump_roi_source_ready) {
              BOOST_LOG(warning)
                << "Dump 3D request rejected: the completed ROI has no exact diagnostic "sv
                   "source crop; live direct ROI analysis remains active."sv;
              sbs_dumper.reject_pending_request();
            }
            const bool dump_frame_valid =
              complete_dump_snapshot && dump_roi_source_ready &&
              sbs_dumper.prepare_requested_v2_frame(est.completed_frame_id);
            if (dump_frame_valid) {
              const auto &dump_attribution = matched_render_capture->attribution;
              ID3D11Buffer *completed_constants = sbs_reprojection_cbuffer.get();
              bool geometry_available = false;
              try {
                geometry_available = render_sbs_debug_geometry(
                  dump_warp_inputs,
                  completed_constants,
                  dump_subtitle_constants
                );
              } catch (const std::exception &error) {
                try {
                  BOOST_LOG(warning)
                    << "Dump 3D geometry preparation failed; publishing the core package without mapping/coverage: "
                    << error.what();
                } catch (...) {
                }
              } catch (...) {
                try {
                  BOOST_LOG(warning)
                    << "Dump 3D geometry preparation failed; publishing the core package without mapping/coverage."sv;
                } catch (...) {
                }
              }

              const bool roi_mapping_unavailable =
                est.input_region.is_video_region() && !geometry_available;
              if (roi_mapping_unavailable) {
                BOOST_LOG(warning)
                  << "Dump 3D request rejected: the authenticated window-region ROI frame "sv
                     "does not have its required full-source inverse map and zero-plane "sv
                     "evidence."sv;
                sbs_dumper.reject_pending_request();
              }

              if (!roi_mapping_unavailable) {
                sbs_debug::frame dump_frame;
                if (dump_actual_draw) dump_frame.actual_draw = *dump_actual_draw;
                dump_frame.source = draw_colour.srv;
                dump_frame.depth_input_source = dump_depth_input_source;
                dump_frame.model_input = est.model_input_snapshot.Get();
                dump_frame.raw_depth = est.raw_model_depth_snapshot.Get();
                dump_frame.warp_depth = dump_warp_inputs[1];
                dump_frame.adaptive_state = est.depth_health_state.Get();
                dump_frame.depth_frame_state = est.depth_frame_state.Get();
                dump_frame.warp_map =
                  geometry_available ? sbs_debug_mapping_srv.Get() : nullptr;
                dump_frame.warp_mask =
                  geometry_available ? sbs_debug_mask_srv.Get() : nullptr;
                dump_frame.sbs = final_sbs_srv;
                dump_frame.shadow_coordinate = est.shadow_coordinate.Get();
                dump_frame.shadow_candidate_parallax =
                  est.shadow_candidate_parallax.Get();
                dump_frame.shadow_vertical_majorant =
                  est.shadow_vertical_majorant.Get();
                dump_frame.shadow_vertical_conditioned =
                  est.shadow_vertical_conditioned.Get();
                dump_frame.shadow_base_final_parallax =
                  est.shadow_base_final_parallax.Get();
                dump_frame.shadow_final_parallax = est.final_parallax();
                dump_frame.ocr_box_record = est.ocr_box_record.Get();
                dump_frame.subtitle_locator_state =
                  est.subtitle_locator_state.Get();
                dump_frame.subtitle_contour_support = est.subtitle_contour_support.Get();
                dump_frame.subtitle_glyph_matte = est.subtitle_glyph_matte.Get();
                dump_frame.subtitle_condition_params = est.subtitle_condition_params.Get();
                dump_frame.subtitle_render_enabled = dump_subtitle_render_enabled;
                dump_frame.subtitle_contour_field_pin = est.subtitle_contour_field_pin.Get();
                dump_frame.subtitle_contour_distance = est.subtitle_contour_distance.Get();
                dump_frame.subtitle_contour_identity = est.subtitle_contour_identity.Get();
                dump_frame.subtitle_contour_history = est.subtitle_contour_history.Get();
                dump_frame.subtitle_contour_source = est.subtitle_contour_source.Get();
                dump_frame.subtitle_ocr_probability = est.subtitle_ocr_probability.Get();
                dump_frame.subtitle_gaussian_weight = est.subtitle_gaussian_weight.Get();
                dump_frame.subtitle_gaussian_blend = est.subtitle_gaussian_blend.Get();
                dump_frame.subtitle_gaussian_vertical_conditioned = est.subtitle_gaussian_vertical_conditioned.Get();
                dump_frame.subtitle_gaussian_control = est.subtitle_gaussian_control.Get();
                dump_frame.gpu_trace_ring = est.gpu_trace_ring.Get();
                dump_frame.shadow_state = est.shadow_state.Get();
                dump_frame.shadow_frame_stats = est.shadow_frame_stats.Get();
                dump_frame.raw_model_provenance = est.raw_model_provenance;
                dump_frame.composite_depth_runtime_provenance =
                  est.composite_depth_runtime_provenance;
                dump_frame.parallax_v2_shader_provenance =
                  est.parallax_v2_shader_provenance;
                dump_frame.gpu_trace_provenance = est.gpu_trace_provenance;
                dump_frame.model_width = est.raw_width;
                dump_frame.model_height = est.raw_height;
                dump_frame.raw_width = est.raw_width;
                dump_frame.raw_height = est.raw_height;
                dump_frame.field_width = est.field_width;
                dump_frame.field_height = est.field_height;
                dump_frame.field_content = est.field_content;
                dump_frame.matched_frame_id = est.completed_frame_id;
                dump_frame.matched_observation_timestamp_us =
                  est.publication.observation_timestamp_us;
                dump_frame.depth_input_region = est.input_region;
                dump_frame.depth_video_plan = dump_attribution.depth_video_plan;
                dump_frame.input_domain_reset = render_step.completion.input_domain_reset;
                dump_frame.gpu_undecided_completion =
                  est.gpu_undecided_completion;
                dump_frame.subtitle_work_suppressed =
                  est.subtitle_work_suppressed;
                dump_frame.window_region = dump_attribution.window_region;
                dump_frame.window_region_observer_status =
                  dump_attribution.window_region_observer_status;
                dump_frame.window_region_mapping_status =
                  dump_attribution.window_region_mapping_status;
                dump_frame.cuda_graph_active = render_step.status.cuda_graph_active;
                dump_frame.parallax_v2_producer_active =
                  est.parallax_v2_producer_active;
                dump_frame.parallax_v2_render_selected =
                  v2_renderer_selected;
                dump_frame.parallax_v2_live_renderer_source_closure_sha256 =
                  v2_renderer_selected ?
                    sbs_reprojection_v2_live_source_closure_sha256 :
                    std::string {};
                dump_frame.parallax_v2_raw_coordinate_scale =
                  est.parallax_v2_raw_coordinate_scale;
                dump_frame.parallax_v2_requested_pop_strength =
                  est.parallax_v2_requested_pop_strength;
                dump_frame.parallax_v2_requested_gain =
                  est.parallax_v2_requested_gain;
                dump_frame.parallax_v2_joint_plane_mode =
                  est.parallax_v2_joint_plane_mode;
                dump_frame.color_space = dump_attribution.color_space;
                dump_frame.depth_model = est.raw_model_provenance ?
                                           est.raw_model_provenance->depth_model :
                                           ::video::host_sbs_v2_depth_model().name;
                sbs_dumper.maybe_dump(
                  device.get(),
                  device_ctx.get(),
                  dump_frame,
                  sbs_config
                );
              }
            }
          }

          if (diagnostics_enabled) {
            ++matched_stats_calls;
            const bool video_roi_output =
              repeat_matched_output ?
                matched_presentation_cache.is_video_region() :
                render_attribution() && geometry.base &&
                  est.input_region.is_video_region();
            if (video_roi_output) {
              ++matched_stats_video_roi_route_outputs;
            }
            const auto now = std::chrono::steady_clock::now();
            if (now - matched_stats_started >= std::chrono::seconds(5)) {
              const double avg_age_ms = matched_stats_completions ?
                                          matched_stats_age_sum_ms / matched_stats_completions :
                                          0.0;
              const double repeat_pct = matched_stats_calls ?
                                          100.0 * matched_stats_repeats / matched_stats_calls :
                                          0.0;
              const double content_skip_pct = matched_stats_calls ?
                                                100.0 * matched_stats_content_skips /
                                                  matched_stats_calls :
                                                0.0;
              const double content_reuse_pct = matched_stats_calls ?
                                                 100.0 * matched_stats_content_reuses /
                                                   matched_stats_calls :
                                                 0.0;
              const double damage_skip_pct = matched_stats_calls ?
                                               100.0 * matched_stats_damage_skips /
                                                 matched_stats_calls :
                                               0.0;
              const double damage_reuse_pct = matched_stats_calls ?
                                                100.0 * matched_stats_damage_reuses /
                                                  matched_stats_calls :
                                                0.0;
              const double video_roi_pct = matched_stats_calls ?
                                             100.0 * matched_stats_video_roi_route_outputs /
                                               matched_stats_calls :
                                             0.0;
              const double same_frame_poll_wait_avg_ms =
                matched_stats_same_frame_poll_attempts ?
                  matched_stats_same_frame_poll_wait_sum_ms /
                    matched_stats_same_frame_poll_attempts :
                  0.0;
              const double same_frame_poll_budget_avg_ms =
                matched_stats_same_frame_poll_plans ?
                  matched_stats_same_frame_poll_budget_sum_ms /
                    matched_stats_same_frame_poll_plans :
                  0.0;
              BOOST_LOG(info) << "SBS cadence: conversions="sv << matched_stats_calls
                              << " completed="sv << matched_stats_completions
                              << " repeats="sv << matched_stats_repeats
                              << " ("sv << repeat_pct << "%) content_skips="sv
                              << matched_stats_content_skips << " ("sv
                              << content_skip_pct << "%) current_color_reuses="sv
                              << matched_stats_content_reuses << " ("sv
                              << content_reuse_pct << "%) damage_skips="sv
                              << matched_stats_damage_skips << " ("sv
                              << damage_skip_pct << "%) damage_reuses="sv
                              << matched_stats_damage_reuses << " ("sv
                              << damage_reuse_pct << "%) video_roi_route_outputs="sv
                              << matched_stats_video_roi_route_outputs << " ("sv
                              << video_roi_pct << "%) age_ms_avg/max="sv
                              << avg_age_ms << '/' << matched_stats_age_max_ms
                              << " roi_direct_inputs/dump_copies="sv
                              << matched_stats_roi_direct_inputs << '/'
                              << matched_stats_roi_dump_copies
                              << " output_warped_new/packed_repeat/flat="sv
                              << matched_stats_output_warped_new << '/'
                              << matched_stats_output_packed_repeat << '/'
                              << matched_stats_output_flat
                              << " p010_y_mrt="sv
                              << matched_stats_output_p010_y_mrt
                              << " flat_pending_miss="sv
                              << matched_stats_output_flat_pending_miss
                              << " same_frame_repeated_wait_attempt/hit="sv
                              << matched_stats_same_frame_poll_attempts << '/'
                              << matched_stats_same_frame_poll_hits
                              << " timeout/failure="sv
                              << matched_stats_same_frame_poll_timeouts << '/'
                              << matched_stats_same_frame_poll_failures
                              << " immediate_hits="sv
                              << matched_stats_same_frame_immediate_hits
                              << " poll_queries="sv
                              << matched_stats_same_frame_poll_queries
                              << " same_frame_wait_hit_buckets_le2/2to2_5/2_5to3/over3="sv
                              << matched_stats_same_frame_poll_hits_le_2_ms << '/'
                              << matched_stats_same_frame_poll_hits_2_to_2_5_ms << '/'
                              << matched_stats_same_frame_poll_hits_2_5_to_3_ms << '/'
                              << matched_stats_same_frame_poll_hits_over_3_ms
                              << " repeated_wait_ms_avg/max="sv
                              << same_frame_poll_wait_avg_ms << '/'
                              << matched_stats_same_frame_poll_wait_max_ms
                              << " planned_budget_ms_avg/max="sv
                              << same_frame_poll_budget_avg_ms << '/'
                              << matched_stats_same_frame_poll_budget_max_ms
                              << " hard_cap/cadence_timeouts="sv
                              << matched_stats_same_frame_poll_hard_cap_timeouts << '/'
                              << matched_stats_same_frame_poll_cadence_timeouts
                              << " same_frame_outcomes="sv
                              << "submissions/immediate_hit/wait_hit/cadence_ineligible_busy/"sv
                              << "eligible_timeout/ready_failure/wait_unavailable_busy="sv
                              << matched_stats_same_frame_poll_submissions << '/'
                              << matched_stats_same_frame_immediate_hits << '/'
                              << matched_stats_same_frame_poll_hits << '/'
                              << matched_stats_same_frame_poll_cadence_ineligible_busy << '/'
                              << matched_stats_same_frame_poll_timeouts << '/'
                              << matched_stats_same_frame_poll_failures << '/'
                              << matched_stats_same_frame_poll_wait_unavailable_busy
                              << " publication_late_ready/force_missing_proof/failure_polls="sv
                              << matched_stats_publication_late_ready << '/'
                              << matched_stats_force_missing_publication << '/'
                              << matched_stats_publication_failure_polls;
              reset_matched_stats(now);
            }
            if (sbs_telemetry_performance) {
              sbs_telemetry_performance->record_output(repeat_matched_output, geometry.region != nullptr, now);
            }
          }

        } else {
          if (rgb) {
            const bool input_is_linear =
              img.format == DXGI_FORMAT_R16G16B16A16_FLOAT;
            if (!copy_rgb(img_ctx->encoder_texture.get(), input_is_linear)) {
              draw_rgb(
                img_ctx->encoder_input_res.get(),
                input_is_linear
              );
            }
          } else {
            // Plain 2D: draw the captured frame straight into the encoder output.
            if (!select_native_output()) {
              return -1;
            }
            draw(img_ctx->encoder_input_res, out_Y_or_YUV_viewport, out_UV_viewport, img.format == DXGI_FORMAT_R16G16B16A16_FLOAT);
          }
        }

        ID3D11ShaderResourceView *emptyShaderResourceView = nullptr;
        device_ctx->PSSetShaderResources(0, 1, &emptyShaderResourceView);
        rendered_content_timestamp_ = converted_content_timestamp;
        stream_gamma_conversion_pending = false;
      } else {
        rendered_content_timestamp_.reset();
      }

      if (converted_input_surface) {
        device_ctx->OMSetRenderTargets(0, nullptr, nullptr);
        native_input_owner->set_converted_input(*converted_input_surface);
      }
      return 0;
    }

    bool set_stream_gamma(::video::stream_gamma_mode_e mode) {
      if (!::video::valid_stream_gamma_mode(static_cast<std::uint8_t>(mode)) || (!output_is_hdr && mode != ::video::stream_gamma_mode_e::windows_default)) {
        return false;
      }
      if (stream_gamma_transform && mode == ::video::stream_gamma_mode_e::windows_default && stream_gamma_mode_ == mode) {
        return true;
      }
      // A transient query failure (for example during a mode change) keeps the level already in
      // use; falling back to the 203-nit default would be a visible white-level jump.
      float white_nits = stream_gamma_transform && stream_gamma_white_nits_ > 0.0f ? stream_gamma_white_nits_ : 203.0f;
      if (display) {
        const auto queried = display->get_sdr_white_nits();
        if (queried && std::isfinite(*queried) && *queried > 0.0f) {
          white_nits = std::clamp(*queried, 40.0f, 1000.0f);
        }
      }
      stream_gamma_white_query_at = std::chrono::steady_clock::now() + 1s;
      if (stream_gamma_transform && mode == stream_gamma_mode_ && white_nits == stream_gamma_white_nits_) {
        return true;
      }
      const stream_gamma_transform_t gamma_constants {
        static_cast<std::uint32_t>(mode),
        white_nits / 80.0f,
      };
      auto gamma_buffer = make_buffer(device.get(), gamma_constants);
      if (!gamma_buffer) {
        return false;
      }
      // BGRA SDR frames in an HDR stream need the same current white used by the
      // correction. FP16 scRGB already contains absolute light and needs no rescaling.
      auto transform = source_color_transform;
      transform.source_sdr_white_scrgb = mode == ::video::stream_gamma_mode_e::windows_default ?
                                           default_source_sdr_white_scrgb :
                                           white_nits / 80.0f;
      auto sdr_buffer = make_buffer(device.get(), transform);
      auto external_transform = transform;
      external_transform.source_is_hdr = !output_is_hdr;
      auto external_buffer = make_buffer(device.get(), external_transform);
      if (!sdr_buffer || !external_buffer) {
        return false;
      }
      stream_gamma_transform = std::move(gamma_buffer);
      sdr_color_transform = std::move(sdr_buffer);
      external_color_transform = std::move(external_buffer);
      source_color_transform = transform;
      stream_gamma_mode_ = mode;
      stream_gamma_white_nits_ = white_nits;
      stream_gamma_conversion_pending = true;
      // The native warp remains reusable, but its cached encoder Y/UV planes carry
      // the previous gamma. Force only the final conversion on a packed repeat.
      host_sbs_encoder_input_state.reset();
      return true;
    }

    ::video::stream_gamma_mode_e stream_gamma_mode() const {
      return stream_gamma_mode_;
    }

    float stream_gamma_white_nits() const {
      return stream_gamma_white_nits_;
    }

    bool apply_colorspace(const ::video::sunshine_colorspace_t &colorspace) {
      output_is_hdr = ::video::colorspace_is_hdr(colorspace);
      auto color_vectors = ::video::color_vectors_from_colorspace(colorspace, true);

      if (!color_vectors) {
        BOOST_LOG(error) << "No vector data for colorspace"sv;
        return false;
      }

      auto color_matrix = make_buffer(device.get(), *color_vectors);
      if (!color_matrix) {
        BOOST_LOG(warning) << "Failed to create color matrix"sv;
        return false;
      }

      // This flag is consumed only by the linear-input shader. The encoded BGRA shader
      // categorically treats its actual input as SDR, which is required for WGC SDR capture on
      // an HDR desktop and also avoids relying on DDup's initially-unknown capture format.
      const bool source_is_hdr = ::video::hdr_to_sdr_tonemap_required(
        ::video::colorspace_is_hdr(colorspace),
        display_is_hdr,
        true
      );
      constexpr float fallback_sdr_white_nits = 203.0f;
      float sdr_white_nits = fallback_sdr_white_nits;
      if (display_is_hdr) {
        const auto queried_sdr_white_nits = display->get_sdr_white_nits();
        if (queried_sdr_white_nits && std::isfinite(*queried_sdr_white_nits) && *queried_sdr_white_nits > 0.0f) {
          sdr_white_nits = *queried_sdr_white_nits;
        } else {
          BOOST_LOG(warning)
            << "Failed to query the display's SDR reference white; using "sv
            << fallback_sdr_white_nits << " nits for SDR/HDR color conversion."sv;
        }
      }

      const sdr_color_transform_t transform {
        colorspace.colorspace == ::video::colorspace_e::bt2020sdr,
        source_is_hdr,
        sdr_white_nits / 80.0f,
        output_is_hdr,
      };
      auto sdr_color_transform = make_buffer(device.get(), transform);
      if (!sdr_color_transform) {
        BOOST_LOG(warning) << "Failed to create SDR color-transform constants"sv;
        return false;
      }
      // External linear frames are explicitly scRGB HDR even when Windows capture is SDR.
      // Encoded external frames ignore this source flag and retain their declared sRGB transfer.
      auto external_transform = transform;
      external_transform.source_is_hdr = !output_is_hdr;
      external_color_transform = make_buffer(device.get(), external_transform);
      if (!external_color_transform) {
        BOOST_LOG(warning) << "Failed to create external color-transform constants"sv;
        return false;
      }

      device_ctx->VSSetConstantBuffers(3, 1, &color_matrix);
      device_ctx->PSSetConstantBuffers(0, 1, &color_matrix);
      this->color_matrix = std::move(color_matrix);
      this->sdr_color_transform = std::move(sdr_color_transform);
      source_color_transform = transform;
      default_source_sdr_white_scrgb = transform.source_sdr_white_scrgb;
      stream_gamma_white_nits_ = std::clamp(sdr_white_nits, 40.0f, 1000.0f);
      const stream_gamma_transform_t gamma_constants {
        0,
        stream_gamma_white_nits_ / 80.0f,
      };
      stream_gamma_transform = make_buffer(device.get(), gamma_constants);
      if (!stream_gamma_transform) {
        return false;
      }
      output_black_y = color_vectors->color_vec_y[3];
      output_neutral_uv = color_vectors->color_vec_u[3];
      return true;
    }

    void fail_depth_pipeline_flat() {
      // The picture stays 2D for the rest of this stream: say so once, where the user sees it.
      if (!flat_fallback_notified) {
        flat_fallback_notified = true;
        system_tray::update_tray_notice(
          "Host 3D is off for this session",
          "Host 3D could not start, so the picture is 2D. Click to open the log."
        );
      }
      host_sbs_renderer = models::fail_host_sbs_renderer_flat(host_sbs_renderer);
      if (depth_estimator) {
        failed_depth_estimators.push_back(std::move(depth_estimator));
      }
      selected_matte_ready = false;
      pending_matched_frame = nullptr;
      for (auto &slot : matched_frame_slots) {
        slot.attribution.gpu_undecided_transaction = false;
      }
      latest_v2_lineage.reset();
      matched_presentation_cache.reset();
      depth_authority_reprocess_pending = false;
      subtitle_context_reprocess_pending = false;
      sbs_dumper.cancel_pending_request();
      publish_depth_status(0);
    }

    bool ensure_depth_estimator() {
      if (!models::host_sbs_renderer_uses_depth_pipeline(host_sbs_renderer)) {
        sbs_dumper.cancel_pending_request();
        return false;
      }
      if (!video_player_registry && !test_context) {
#ifdef SUNSHINE_TESTS
        if (test_registry_executables.valid()) {
          video_player_registry = std::make_unique<video_player::registry_t>(test_registry_executables);
          ++test_registry_factory_calls;
        } else {
          video_player_registry = std::make_unique<video_player::registry_t>();
        }
#else
        video_player_registry = std::make_unique<video_player::registry_t>();
#endif
      }
      if (depth_estimator) {
        return true;
      }

      // An engaged future is the sole owner of an in-flight build. Take the result once ready;
      // get() invalidates the future, so no parallel flag can drift from the shared state.
      if (depth_estimator_build.valid()) {
        if (depth_estimator_build.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
          try {
            auto pipeline = depth_estimator_build.get();
            depth_estimator = std::move(pipeline.estimator);
            if (depth_estimator) {
              depth_estimator->set_telemetry_performance(sbs_telemetry_performance);
              if (!test_context) {
                if (pipeline.selected_matte.ready) {
                  BOOST_LOG(info) << "Adaptive subtitle glyph model active.";
                } else {
                  BOOST_LOG(warning) << "Subtitle glyph initialization unavailable;retaining ordinary authenticated Base: "
                                     << pipeline.selected_matte.error;
                }
                adopt_selected_matte_programs(std::move(pipeline.selected_matte));
              }
            }
          } catch (const std::exception &e) {
            // Don't let a background-build exception propagate on the encode thread (it would end
            // the stream); log it, clear the client's "loading" indicator, and stream flat.
            BOOST_LOG(error) << "Host SBS per-stream GPU pipeline initialization failed: "sv << e.what();
            fail_depth_pipeline_flat();
            return false;
          }
          if (depth_estimator && depth_estimator->is_valid() && !depth_estimator->has_terminal_failure()) {
            BOOST_LOG(info) << "Host SBS per-stream GPU pipeline ready; depth is now live."sv;
            publish_depth_status(2);  // ready -> client hides indicator
          } else {
            BOOST_LOG(error)
              << (depth_estimator && depth_estimator->has_terminal_failure() ?
                    "Host SBS V2 producer is unavailable for this stream; streaming flat SBS."sv :
                    "Host SBS per-stream GPU pipeline initialization returned an invalid result; streaming flat SBS."sv);
            fail_depth_pipeline_flat();
          }
          return (bool) depth_estimator;
        }
        return false;
      }

      // Model GPU preparation is process-global and may take minutes on first AI use. It includes
      // engine compilation, deserialization, execution-context creation, and CUDA warmup. Report
      // that complete wait as loading, but don't duplicate it on this encode device.
      auto active = ::video::host_sbs_v2_depth_model();
      auto build_status = models::tensorrt_model_prepare_status(active);
      if (build_status == models::engine_build_status::unknown) {
        if (!models::request_live_model_preparation()) {
          return false;
        }
        build_status = models::tensorrt_model_prepare_status(active);
      }
      if (build_status != models::engine_build_status::ready) {
        if (build_status == models::engine_build_status::failed) {
          BOOST_LOG(error) << "Startup TensorRT model preparation failed for '"sv << active.name
                           << "'; streaming flat SBS."sv;
          fail_depth_pipeline_flat();
        } else {
          if (engine_poll_counter++ % 1800 == 0) {  // ~every 20 s at 90 fps
            BOOST_LOG(warning) << "Waiting for startup TensorRT model preparation for '"sv << active.name
                               << "'; streaming flat until it is ready."sv;
          }
          publish_depth_status(1);
        }
        return false;
      }

      // Kick construction on a background thread. The device is free-threaded for the resource
      // creation the build does. The build makes no immediate-context calls and leaves CUDA's
      // D3D11 registration, which reaches the immediate context, to this drawing thread. Capture
      // owning ComPtrs so a presentation-session teardown never has to wait for construction.
      auto sbs_cfg = sbs_config;
      // Every estimator consumer uses the authenticated Host SBS tensor shapes and private
      // analysis contract.
      BOOST_LOG(info) << "Host SBS enabled; initializing the per-stream GPU pipeline for resident model \""sv
                      << active.name
                      << "\" in the background (streaming flat until ready)..."sv;
      Microsoft::WRL::ComPtr<ID3D11Device> dev(device.get());
      Microsoft::WRL::ComPtr<ID3D11DeviceContext> ctx(device_ctx.get());
      const bool prepare_live_subtitles = !test_context;
      using normalized_depth_diagnostics_e = models::video_depth_estimator::normalized_depth_diagnostics_e;
      const auto normalized_depth_diagnostics = test_context ?
                                                  normalized_depth_diagnostics_e::enabled :
                                                  normalized_depth_diagnostics_e::disabled;
      using subtitle_legacy_matte_e = models::video_depth_estimator::subtitle_legacy_matte_e;
      const auto legacy_matte = test_context ? subtitle_legacy_matte_e::always : subtitle_legacy_matte_e::on_demand;
      const bool prepare_text_game_programs = sbs_mode == ::video::SBS_GAME_SBS && text_game_eligible;
      depth_estimator_build_task_t build_task([dev = std::move(dev),
                                               ctx = std::move(ctx),
                                               active,
                                               sbs_cfg,
                                               normalized_depth_diagnostics,
                                               legacy_matte,
                                               prepare_live_subtitles,
                                               prepare_text_game_programs]() mutable {
        depth_pipeline_build_result_t result;
        result.estimator = std::make_unique<models::video_depth_estimator>(
          dev,
          std::move(ctx),
          std::filesystem::path(SUNSHINE_ASSETS_DIR),
          sbs_cfg,
          active,
          normalized_depth_diagnostics,
          legacy_matte
        );
        // Live rendering binds Base. Conditioned-field diagnostics run only for Dump 3D.
        result.estimator->set_subtitle_conditioning_diagnostics_always(false);
        if (prepare_live_subtitles && result.estimator->is_valid() &&
            !result.estimator->has_terminal_failure()) {
          namespace completed_subtitles = models::host_sbs_completed_subtitle_programs;
          const std::filesystem::path root(SUNSHINE_SHADERS_DIR);
          result.selected_matte = prepare_selected_matte_programs(
            *result.estimator, dev.Get(), completed_subtitles::canonical_configuration(root)
          );
          // Keep the optional program creation on the background worker for packed Game 3D,
          // including its early local warmup before a text game appears.
          if (std::string error; prepare_text_game_programs && !result.selected_matte.shared_warp.load_text_game(dev.Get(), root, error)) {
            BOOST_LOG(warning) << "Game 3D text-game pin unavailable; a text game keeps the picture-based protection: " << error;
          }
        }
        return result;
      });
      depth_estimator_build = build_task.get_future();
      auto pipeline_ready_event = sbs_depth_pipeline_ready_event;
      launch_depth_estimator_build(
        std::move(build_task),
        [pipeline_ready_event = std::move(pipeline_ready_event)]() {
          if (pipeline_ready_event) {
            pipeline_ready_event->raise(true);
          }
        }
      );
      publish_depth_status(3);  // device-specific pipeline initialization (engine is already ready)
      return false;
    }

    void publish_depth_status(int phase) {
      if (sbs_depth_status_event) {
        sbs_depth_status_event->raise(phase);
      }
    }

    bool update_sbs_constant_buffer(
      const float content_scale_x,
      const float content_scale_y,
      const models::depth_input_region_t *input_region = nullptr,
      const models::depth_tensor_content_rect_t *field_content = nullptr
    ) {
      auto geometry = models::make_host_sbs_v2_full_frame_geometry(
        content_scale_x,
        content_scale_y
      );
      if (input_region && input_region->is_video_region() && input_region->valid()) {
        const auto &renderer_content = field_content ?
                                         *field_content :
                                         input_region->tensor_content;
        if (!renderer_content.valid()) {
          return false;
        }
        const auto source_width = static_cast<float>(input_region->source_width);
        const auto source_height = static_cast<float>(input_region->source_height);
        geometry.video_roi_active = 1.0f;
        geometry.video_roi_left = static_cast<float>(input_region->left) / source_width;
        geometry.video_roi_top = static_cast<float>(input_region->top) / source_height;
        geometry.video_roi_right = static_cast<float>(input_region->right) / source_width;
        geometry.video_roi_bottom = static_cast<float>(input_region->bottom) / source_height;
        geometry.tensor_content_left = renderer_content.left;
        geometry.tensor_content_top = renderer_content.top;
        geometry.tensor_content_right = renderer_content.right;
        geometry.tensor_content_bottom = renderer_content.bottom;
      }

      if (!sbs_reprojection_cbuffer) {
        D3D11_BUFFER_DESC desc {};
        desc.ByteWidth = sizeof(geometry);
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA initial {};
        initial.pSysMem = &geometry;
        buf_t::pointer buffer = nullptr;
        const auto status = device->CreateBuffer(&desc, &initial, &buffer);
        if (FAILED(status)) {
          BOOST_LOG(error) << "Failed to create Host SBS V2 geometry constants [0x"sv
                           << util::hex(status).to_string_view() << ']';
          return false;
        }
        sbs_reprojection_cbuffer = buf_t {buffer};
        sbs_reprojection_constants.commit(geometry);
      } else if (!sbs_reprojection_constants.is_current(geometry)) {
        device_ctx->UpdateSubresource(
          sbs_reprojection_cbuffer.get(),
          0,
          nullptr,
          &geometry,
          0,
          0
        );
        sbs_reprojection_constants.commit(geometry);
      }
      return true;
    }

    bool update_sbs_subtitle_constant_buffer(
      const models::host_sbs_subtitle_render_constants_t &constants
    ) {
      if (!sbs_subtitle_render_cbuffer) {
        D3D11_BUFFER_DESC desc {};
        desc.ByteWidth = sizeof(constants);
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA initial {};
        initial.pSysMem = &constants;
        buf_t::pointer buffer = nullptr;
        if (FAILED(device->CreateBuffer(&desc, &initial, &buffer))) {
          return false;
        }
        sbs_subtitle_render_cbuffer = buf_t {buffer};
        sbs_subtitle_render_constants.commit(constants);
      } else if (!sbs_subtitle_render_constants.is_current(constants)) {
        device_ctx->UpdateSubresource(sbs_subtitle_render_cbuffer.get(), 0u, nullptr,
                                     &constants, 0u, 0u);
        sbs_subtitle_render_constants.commit(constants);
      }
      return true;
    }

    std::uint32_t next_sbs_telemetry_sequence() {
      if (++sbs_telemetry_sequence == 0) {
        ++sbs_telemetry_sequence;
      }
      return sbs_telemetry_sequence;
    }

    ::video::sbs_telemetry_snapshot_t base_sbs_telemetry_snapshot(
      ::video::sbs_telemetry_sample_status_e status
    ) {
      ::video::sbs_telemetry_snapshot_t snapshot;
      snapshot.status = status;
      snapshot.generation = sbs_telemetry_generation;
      snapshot.sequence = next_sbs_telemetry_sequence();
      snapshot.performance = sbs_telemetry_performance;
      ::video::apply_sbs_telemetry_config(snapshot, sbs_config);
      return snapshot;
    }

    void publish_sbs_telemetry_failure() {
      if (sbs_telemetry_event) {
        // A recovery must supersede the failure even if its cached sample is unchanged.
        sbs_telemetry_sample_gate.reset();
        sbs_telemetry_event->raise(base_sbs_telemetry_snapshot(
          ::video::sbs_telemetry_sample_status_e::failed
        ));
      }
    }

    void publish_sbs_telemetry_sample(const models::depth_telemetry_sample &sample) {
      if (!sbs_telemetry_event || !sbs_telemetry_sample_gate.accept(sample.sampled_frame_id, sample.sampled_at)) {
        return;
      }

      auto snapshot = base_sbs_telemetry_snapshot(
        ::video::sbs_telemetry_sample_status_e::ready
      );
      snapshot.valid_fields =
        ::video::sbs_telemetry_valid_field::config |
        ::video::sbs_telemetry_valid_field::effective_pop |
        ::video::sbs_telemetry_valid_field::valid_fraction |
        ::video::sbs_telemetry_valid_field::faults;
      snapshot.depth_width = static_cast<std::uint16_t>(std::clamp(
        sample.depth_width,
        0,
        static_cast<int>(std::numeric_limits<std::uint16_t>::max())
      ));
      snapshot.depth_height = static_cast<std::uint16_t>(std::clamp(
        sample.depth_height,
        0,
        static_cast<int>(std::numeric_limits<std::uint16_t>::max())
      ));
      snapshot.effective_pop = snapshot.pop_floor;
      snapshot.edge_fraction = sample.edge_fraction;
      snapshot.zero_anchor_shift_px = sample.zero_anchor_shift_px;
      snapshot.subject_depth = sample.subject_depth;
      snapshot.valid_depth_fraction = sample.valid_depth_fraction;
      snapshot.effective_range_width = sample.effective_range_width;
      snapshot.empty_raw_count = sample.empty_raw_count;
      snapshot.collapsed_raw_count = sample.collapsed_raw_count;
      snapshot.sampled_frame_id = static_cast<std::uint32_t>(sample.sampled_frame_id);

      if (sample.profile_initialized) {
        snapshot.runtime_flags |= ::video::sbs_telemetry_runtime_flag::profile_initialized;
      }
      sbs_telemetry_event->raise(std::move(snapshot));
    }

    void publish_sbs_telemetry_after_output() {
      if (!sbs_telemetry_subscription || !sbs_telemetry_event || !sbs_telemetry_subscription->enabled()) {
        return;
      }
      const auto now = std::chrono::steady_clock::now();
      const auto interval = std::chrono::milliseconds(std::max<std::uint16_t>(sbs_telemetry_subscription->interval_ms(), 1));
      if (sbs_telemetry_last_publish_attempt.time_since_epoch().count() != 0 && now - sbs_telemetry_last_publish_attempt < interval) {
        return;
      }
      sbs_telemetry_last_publish_attempt = now;
      const bool producer_active =
        models::host_sbs_renderer_uses_depth_pipeline(host_sbs_renderer);
      if (!producer_active) {
        // A fail-flat producer's cached snapshot is historical, not live health data.
        if (!sbs_telemetry_producer_failure_published) {
          publish_sbs_telemetry_failure();
          sbs_telemetry_producer_failure_published = true;
        }
        return;
      }
      if (!depth_estimator) {
        return;
      }
      sbs_telemetry_producer_failure_published = false;
      // Only a new captured sample creates a health sequence. The control loop repeats the
      // existing snapshot as a heartbeat and samples performance independently at send time.
      const auto result = depth_estimator->latest_depth_telemetry();
      if (result.failed) {
        publish_sbs_telemetry_failure();
        return;
      }
      if (result.sample && result.sample->sampled_frame_id >= sbs_telemetry_min_frame_id) {
        publish_sbs_telemetry_sample(*result.sample);
      }
    }

    struct sbs_gpu_timer_slot_t {
      d3d_query_t disjoint;
      d3d_query_t start;
      d3d_query_t matched_copy_start;
      d3d_query_t matched_copy_end;
      d3d_query_t warp_start;
      d3d_query_t warp_end;
      d3d_query_t convert_end;
      bool pending = false;
      bool matched_copy_started = false;
      bool matched_copy_ended = false;
      bool has_matched_copy = false;
      bool has_sbs_composite = false;
      std::uint64_t perf_generation = 0;
    };

    void initialize_sbs_gpu_timers() {
      sbs_gpu_timing_ready = false;
      sbs_gpu_timer_next = 0;
      for (auto &slot : sbs_gpu_timers) {
        slot = {};
      }
      if (!diagnostics_enabled) {
        return;
      }

      for (auto &slot : sbs_gpu_timers) {
        D3D11_QUERY_DESC desc {D3D11_QUERY_TIMESTAMP_DISJOINT, 0};
        if (FAILED(device->CreateQuery(&desc, &slot.disjoint))) {
          BOOST_LOG(warning) << "Host SBS GPU timing unavailable: could not create disjoint query."sv;
          return;
        }
        desc.Query = D3D11_QUERY_TIMESTAMP;
        if (FAILED(device->CreateQuery(&desc, &slot.start)) || FAILED(device->CreateQuery(&desc, &slot.matched_copy_start)) || FAILED(device->CreateQuery(&desc, &slot.matched_copy_end)) || FAILED(device->CreateQuery(&desc, &slot.warp_start)) || FAILED(device->CreateQuery(&desc, &slot.warp_end)) || FAILED(device->CreateQuery(&desc, &slot.convert_end))) {
          BOOST_LOG(warning) << "Host SBS GPU timing unavailable: could not create timestamp queries."sv;
          return;
        }
      }
      sbs_gpu_timing_ready = true;
    }

    void resolve_sbs_gpu_timers() {
      if (!sbs_gpu_timing_ready) {
        return;
      }
      for (auto &slot : sbs_gpu_timers) {
        if (!slot.pending) {
          continue;
        }
        D3D11_QUERY_DATA_TIMESTAMP_DISJOINT timing {};
        const auto ready = device_ctx->GetData(slot.disjoint.get(), &timing, sizeof(timing), D3D11_ASYNC_GETDATA_DONOTFLUSH);
        if (ready == S_FALSE) {
          continue;
        }
        if (ready != S_OK) {
          slot.pending = false;
          continue;
        }

        UINT64 start = 0;
        UINT64 matched_copy_start = 0;
        UINT64 matched_copy_end = 0;
        UINT64 warp_start = 0;
        UINT64 warp_end = 0;
        UINT64 convert_end = 0;
        const auto start_status = device_ctx->GetData(slot.start.get(), &start, sizeof(start), D3D11_ASYNC_GETDATA_DONOTFLUSH);
        const auto matched_copy_start_status = device_ctx->GetData(slot.matched_copy_start.get(), &matched_copy_start, sizeof(matched_copy_start), D3D11_ASYNC_GETDATA_DONOTFLUSH);
        const auto matched_copy_status = device_ctx->GetData(slot.matched_copy_end.get(), &matched_copy_end, sizeof(matched_copy_end), D3D11_ASYNC_GETDATA_DONOTFLUSH);
        const auto warp_start_status = device_ctx->GetData(slot.warp_start.get(), &warp_start, sizeof(warp_start), D3D11_ASYNC_GETDATA_DONOTFLUSH);
        const auto warp_status = device_ctx->GetData(slot.warp_end.get(), &warp_end, sizeof(warp_end), D3D11_ASYNC_GETDATA_DONOTFLUSH);
        const auto convert_status = device_ctx->GetData(slot.convert_end.get(), &convert_end, sizeof(convert_end), D3D11_ASYNC_GETDATA_DONOTFLUSH);
        if (start_status == S_FALSE || matched_copy_start_status == S_FALSE || matched_copy_status == S_FALSE || warp_start_status == S_FALSE || warp_status == S_FALSE || convert_status == S_FALSE) {
          continue;
        }
        if (start_status != S_OK || matched_copy_start_status != S_OK || matched_copy_status != S_OK || warp_start_status != S_OK || warp_status != S_OK || convert_status != S_OK) {
          slot.pending = false;
          continue;
        }
        if (!timing.Disjoint && timing.Frequency > 0 && matched_copy_start >= start && matched_copy_end >= matched_copy_start && warp_start >= matched_copy_end && warp_end >= warp_start && convert_end >= warp_end) {
          const double to_ms = 1000.0 / static_cast<double>(timing.Frequency);
          if (slot.has_matched_copy) {
            sbs_perf::add_sample_ms_if_current(
              "matched_frame_copy_gpu",
              static_cast<double>(matched_copy_end - matched_copy_start) * to_ms,
              slot.perf_generation
            );
          }
          if (slot.has_sbs_composite) {
            sbs_perf::add_sample_ms_if_current(
              "sbs_warp_gpu",
              static_cast<double>(warp_end - warp_start) * to_ms,
              slot.perf_generation
            );
            if (sbs_telemetry_performance) {
              sbs_telemetry_performance->record(host_sbs_telemetry::stage::warp, static_cast<double>(warp_end - warp_start) * to_ms, std::chrono::steady_clock::now());
            }
          }
          sbs_perf::add_sample_ms_if_current(
            "sbs_output_gpu",
            static_cast<double>(convert_end - warp_end) * to_ms,
            slot.perf_generation
          );
          if (sbs_telemetry_performance) {
            sbs_telemetry_performance->record(host_sbs_telemetry::stage::output, static_cast<double>(convert_end - warp_end) * to_ms, std::chrono::steady_clock::now());
          }
        }
        slot.pending = false;
      }
    }

    sbs_gpu_timer_slot_t *begin_sbs_gpu_timer() {
      resolve_sbs_gpu_timers();
      if (!sbs_gpu_timing_ready) {
        return nullptr;
      }
      for (std::size_t i = 0; i < sbs_gpu_timers.size(); ++i) {
        const std::size_t index = (sbs_gpu_timer_next + i) % sbs_gpu_timers.size();
        auto &slot = sbs_gpu_timers[index];
        if (slot.pending) {
          continue;
        }
        sbs_gpu_timer_next = (index + 1) % sbs_gpu_timers.size();
        slot.matched_copy_started = false;
        slot.matched_copy_ended = false;
        slot.has_matched_copy = false;
        slot.has_sbs_composite = false;
        slot.perf_generation = sbs_perf::generation();
        device_ctx->Begin(slot.disjoint.get());
        device_ctx->End(slot.start.get());
        return &slot;
      }
      return nullptr;  // GPU is more than one timing-ring behind; never stall the encode thread.
    }

    void drain_sbs_gpu_timers() {
      if (!sbs_gpu_timing_ready || !device_ctx) {
        return;
      }
      const auto pending_count = [&]() {
        return std::count_if(sbs_gpu_timers.begin(), sbs_gpu_timers.end(), [](const auto &slot) {
          return slot.pending;
        });
      };
      if (pending_count() == 0) {
        return;
      }

      device_ctx->Flush();
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
      do {
        resolve_sbs_gpu_timers();
        if (pending_count() == 0) {
          return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } while (std::chrono::steady_clock::now() < deadline);

      BOOST_LOG(warning) << "Host SBS GPU timing snapshot is partial; "sv << pending_count()
                         << " tail sample(s) were still pending after the bounded teardown drain."sv;
    }

    void mark_sbs_warp_end(sbs_gpu_timer_slot_t *slot) {
      if (slot) {
        device_ctx->End(slot->warp_end.get());
      }
    }

    void end_sbs_gpu_timer(sbs_gpu_timer_slot_t *slot) {
      if (!slot) {
        return;
      }
      device_ctx->End(slot->convert_end.get());
      device_ctx->End(slot->disjoint.get());
      slot->pending = true;
    }

    class matched_presentation_cache_t {
    public:
      void invalidate() noexcept {
        // Pixel continuity can be invalidated without reopening the one-shot timeout log latch.
        valid_ = false;
        frame_id_ = 0u;
        pixel_source_frame_id_ = 0u;
        source_at_ = {};
        presentation_clock_.invalidate();
        source_timestamp_.reset();
        content_timestamp_.reset();
        input_region_ = {};
        color_space_ = models::input_color_space::srgb;
        field_width_ = 0;
        field_height_ = 0;
        positioned_subtitles_ = false;
        presentation_authority_ = {};
      }

      void reset() noexcept {
        invalidate();
        presentation_clock_.reset();
      }

      void publish(
        const std::uint64_t presented_frame_id,
        const std::uint64_t pixel_source_frame_id,
        const std::chrono::steady_clock::time_point presented_at,
        const std::optional<std::chrono::steady_clock::time_point> &presented_source_timestamp,
        const std::optional<std::chrono::steady_clock::time_point> &presented_content_timestamp,
        const models::depth_input_region_t &presented_input_region,
        const models::input_color_space presented_color_space,
        const int presented_field_width,
        const int presented_field_height,
        const std::uint64_t subtitle_context_generation,
        const bool positioned_subtitles,
        const models::host_subtitle_context::presentation_authority_t presentation_authority,
        const std::optional<std::chrono::steady_clock::time_point> &current_source_timestamp,
        const std::chrono::steady_clock::time_point current_source_observed_at
      ) noexcept {
        // The selected completed draw reports recovery independently of pixel publication.
        valid_ = true;
        frame_id_ = presented_frame_id;
        pixel_source_frame_id_ = pixel_source_frame_id;
        source_at_ = presented_at;
        source_timestamp_ = presented_source_timestamp;
        observe_current_source(current_source_timestamp, current_source_observed_at);
        content_timestamp_ = presented_content_timestamp;
        input_region_ = presented_input_region;
        color_space_ = presented_color_space;
        field_width_ = presented_field_width;
        field_height_ = presented_field_height;
        subtitle_context_generation_ = subtitle_context_generation;
        positioned_subtitles_ = positioned_subtitles;
        presentation_authority_ = presentation_authority;
      }

      [[nodiscard]] bool has_fresh_pixels() const noexcept {
        return valid_;
      }

      [[nodiscard]] std::uint64_t frame_id() const noexcept {
        return valid_ ? frame_id_ : 0u;
      }

      [[nodiscard]] std::uint64_t pixel_source_frame_id() const noexcept {
        return valid_ ? pixel_source_frame_id_ : 0u;
      }

      [[nodiscard]] const std::optional<std::chrono::steady_clock::time_point> &source_timestamp() const noexcept { return source_timestamp_; }
      [[nodiscard]] std::chrono::steady_clock::duration source_age(
        const std::chrono::steady_clock::time_point now
      ) const noexcept {
        return source_at_.time_since_epoch().count() != 0 ?
                 now - source_at_ :
                 std::chrono::steady_clock::duration::max();
      }

      [[nodiscard]] bool source_matches(
        const std::optional<std::chrono::steady_clock::time_point> &source
      ) const noexcept {
        return source_timestamp_ && source && *source_timestamp_ == *source;
      }

      void observe_current_source(
        const std::optional<std::chrono::steady_clock::time_point> &source,
        const std::chrono::steady_clock::time_point observed_at
      ) noexcept {
        if (valid_) {
          presentation_clock_.observe_source(source_timestamp_, source, observed_at);
        }
      }

      [[nodiscard]] std::chrono::steady_clock::duration hold_age(
        const std::optional<std::chrono::steady_clock::time_point> &source,
        const std::chrono::steady_clock::time_point now
      ) const noexcept {
        return presentation_clock_.hold_age(source_age(now), source_timestamp_, source, now);
      }

      [[nodiscard]] bool route_matches(
        const UINT source_width,
        const UINT source_height,
        const models::input_color_space color_space,
        const bool reprocess_pending,
        const bool producer_terminal,
        const std::uint64_t subtitle_context_generation,
        const models::host_subtitle_context::presentation_authority_t presentation_authority
      ) const noexcept {
        return valid_ && input_region_.valid() &&
               input_region_.source_width == source_width &&
               input_region_.source_height == source_height &&
               color_space_ == color_space && !reprocess_pending && !producer_terminal &&
               subtitle_context_generation_ == subtitle_context_generation &&
               presentation_authority_.permits_packed_repeat(
                 presentation_authority, input_region_.is_video_region() || positioned_subtitles_);
      }

      [[nodiscard]] const std::optional<std::chrono::steady_clock::time_point> &
        content_timestamp() const noexcept {
        return content_timestamp_;
      }

      detail::host_sbs_packed_presentation_clock_t::timeout_event_e observe_timeout(
        const bool age_expired,
        const detail::host_sbs_packed_presentation_clock_t::output_e output
      ) noexcept {
        return presentation_clock_.observe_timeout(age_expired, output);
      }

      [[nodiscard]] bool is_video_region() const noexcept {
        return valid_ && input_region_.is_video_region();
      }

      [[nodiscard]] bool has_positioned_subtitles() const noexcept {
        return valid_ && positioned_subtitles_;
      }

      [[nodiscard]] bool has_authority(
        const models::depth_analysis_authority_e authority
      ) const noexcept {
        return valid_ && input_region_.authority == authority;
      }

      [[nodiscard]] bool matches_analysis_domain(
        const models::depth_input_region_t &region,
        const models::input_color_space color_space
      ) const noexcept {
        return valid_ && depth_analysis_domain_matches_realization(region, input_region_, field_width_, field_height_) &&
               color_space == color_space_;
      }

    private:
      bool valid_ = false;
      std::uint64_t frame_id_ = 0u;
      std::uint64_t pixel_source_frame_id_ = 0u;
      std::chrono::steady_clock::time_point source_at_ {};
      std::optional<std::chrono::steady_clock::time_point> source_timestamp_;
      std::optional<std::chrono::steady_clock::time_point> content_timestamp_;
      models::depth_input_region_t input_region_ {};
      models::input_color_space color_space_ = models::input_color_space::srgb;
      int field_width_ = 0;
      int field_height_ = 0;
      detail::host_sbs_packed_presentation_clock_t presentation_clock_;
      std::uint64_t subtitle_context_generation_ = 0u;
      bool positioned_subtitles_ = false;
      models::host_subtitle_context::presentation_authority_t presentation_authority_;
    };

    struct encoder_img_ctx_t;

    class described_texture_t {
    public:
      [[nodiscard]] explicit operator bool() const noexcept {
        return static_cast<bool>(texture_);
      }

      [[nodiscard]] ID3D11Texture2D *get() noexcept {
        return texture_.get();
      }

      void reset(texture2d_t texture = {}) noexcept {
        texture_ = std::move(texture);
        description_.reset();
      }

      void borrow_from(described_texture_t &source) noexcept {
        auto *texture = source.get();
        texture->AddRef();
        texture_.reset(texture);
        description_ = source.description();
      }

      [[nodiscard]] const D3D11_TEXTURE2D_DESC &description() noexcept {
        // Ordinary SDR conversion samples the SRV without needing a description.
        if (!description_) {
          texture_.get()->GetDesc(&description_.emplace());
        }
        return *description_;
      }

    private:
      texture2d_t texture_;
      std::optional<D3D11_TEXTURE2D_DESC> description_;
    };

    struct sbs_draw_colour_t {
      ID3D11ShaderResourceView *srv;
      models::input_color_space color_space;
      ::video::detail::optional_frame_time_point_t source_timestamp;
      ::video::detail::optional_frame_time_point_t content_timestamp;
      std::chrono::steady_clock::time_point captured_at;
      std::uint64_t pixel_source_frame_id;

      [[nodiscard]] ::video::detail::optional_frame_time_point_t rendered_content_timestamp() const noexcept {
        return content_timestamp ? content_timestamp : source_timestamp;
      }
    };

    struct matched_frame_attribution_t {
      std::uint64_t frame_id = 0;
      models::input_color_space color_space = models::input_color_space::srgb;
      std::optional<std::chrono::steady_clock::time_point> source_timestamp;
      std::optional<std::chrono::steady_clock::time_point> content_timestamp;
      // These fields identify the submitted analysis pixels. The receipt separately identifies
      // the actual DAV2 owner when this transaction reused depth. A draw can use another colour
      // source while leaving the analysis slot and source-continuity anchor intact.
      std::optional<std::chrono::steady_clock::time_point> inference_content_timestamp;
      std::optional<detail::ddup_damage_snapshot_t> inference_ddup_damage;
      // Advances only exact clean-ROI evidence, never the submitted source attribution above.
      mutable detail::ddup_unchanged_roi_proof_t unchanged_roi_proof;
      std::chrono::steady_clock::time_point captured_at {};
      std::optional<sbs_debug::window_region_snapshot> window_region;
      std::optional<models::depth_video_region_plan_t> depth_video_plan;
      models::depth_input_region_t depth_input_region {};
      std::string_view window_region_observer_status = "not-observed";
      std::string_view window_region_mapping_status = "not-mapped";
      std::uint64_t live_window_authority_generation = 0u;
      std::uint64_t live_browser_authority_generation = 0u;
      std::int32_t live_browser_screen_left = 0;
      std::int32_t live_browser_screen_top = 0;
      std::int32_t live_browser_screen_right = 0;
      std::int32_t live_browser_screen_bottom = 0;
      std::chrono::steady_clock::time_point live_browser_geometry_valid_since {};
      video_dom::geometry_authority_class_e live_browser_authority_class =
        video_dom::geometry_authority_class_e::none;
      models::host_subtitle_context::presentation_authority_t observed_authority;
      std::uint64_t subtitle_context_generation = 0u;
      bool subtitle_context_eligible = false;
      // Semantic retention during a freshly verified modal loop permits no positioned glyph
      // authority. Keep it separate from the normal per-frame eligibility used by rendering.
      bool subtitle_context_suppression_eligible = false;
      // Submitted with the same context generation, so it cannot change within one context.
      models::subtitle_line_policy_e subtitle_line_policy = models::subtitle_line_policy_e::video_caption;
      std::uint64_t test_instance = 0u;
      // CPU-owned submission class authenticates matched completion metadata. The exact final
      // publication receipt independently reports the actual infer/reuse branch and depth owner.
      bool gpu_undecided_transaction = false;
    };

    struct matched_frame_slot_t {
      // The pool may recycle only unreferenced images. This pin also keeps the encoder
      // context mapped, so its keyed mutex survives until the matched draw is submitted.
      std::shared_ptr<const platf::img_t> capture_image;
      IDXGIKeyedMutex *capture_mutex = nullptr;
      described_texture_t texture;
      shader_res_t srv;
      matched_frame_attribution_t attribution;

      ID3D11ShaderResourceView *analysis_input_srv() noexcept {
        return srv.get();
      }

      void release_capture_image() noexcept {
        if (capture_image) {
          srv.reset();
          texture.reset();
          capture_mutex = nullptr;
          capture_image.reset();
        }
      }
    };

    // Resolved once per conversion. Geometry and attribution borrow from the same owner.
    // A physical source can provide matched color on a flat draw as well.
    struct sbs_draw_geometry_t {
      const matched_frame_attribution_t *source_attribution = nullptr;
      matched_frame_slot_t *source_capture = nullptr;
      ID3D11PixelShader *pixel_shader = nullptr;
      const models::depth_input_region_t *region = nullptr;
      const models::depth_tensor_content_rect_t *field_content = nullptr;
      ID3D11ShaderResourceView *base = nullptr;
      ID3D11ShaderResourceView *state = nullptr;
      ID3D11ShaderResourceView *display_band = nullptr;
      ID3D11ShaderResourceView *condition_params = nullptr;
      ID3D11ShaderResourceView *contour_identity = nullptr;
      ID3D11ShaderResourceView *selected_matte = nullptr;
      ID3D11ShaderResourceView *selected_receipt = nullptr;
      std::uint64_t frame_id = 0u;
      int field_width = 0, field_height = 0;
      bool subtitle_suppressed = false;

      [[nodiscard]] bool has_subtitle_inputs() const noexcept {
        return !subtitle_suppressed && display_band && condition_params && contour_identity;
      }
    };

    [[nodiscard]] sbs_draw_geometry_t select_sbs_draw_geometry(
      matched_frame_slot_t *matched_capture,
      const matched_frame_attribution_t *retained_attribution,
      const models::depth_observation_views_t *completed
    ) noexcept {
      const auto *matched_attribution = matched_capture ? &matched_capture->attribution : retained_attribution;
      if (completed) {
        return {
          .source_attribution = matched_attribution,
          .source_capture = matched_capture,
          .pixel_shader = sbs_reprojection_v2_live_ps.get(),
          .region = &completed->input_region,
          .field_content = &completed->field_content,
          .base = completed->shadow_base_final_parallax.Get(),
          .state = completed->shadow_state.Get(),
          .display_band = completed->subtitle_contour_source.Get(),
          .condition_params = completed->subtitle_condition_params.Get(),
          .contour_identity = completed->subtitle_contour_identity.Get(),
          .selected_matte = completed->subtitle_selected_matte.Get(),
          .selected_receipt = completed->subtitle_selected_matte_receipt.Get(),
          .frame_id = completed->completed_frame_id,
          .field_width = completed->field_width,
          .field_height = completed->field_height,
          .subtitle_suppressed = completed->subtitle_work_suppressed,
        };
      }
      return {
        .source_attribution = matched_attribution,
        .source_capture = matched_capture,
        .pixel_shader = sbs_flat_identity_ps.get(),
      };
    }

    void emit_test_observation(converter_test_observer::event_t event, const matched_frame_attribution_t *slot, const models::depth_observation_views_t *result, ID3D11ShaderResourceView *selected, std::uint64_t conversion, const sbs_draw_colour_t *colour = nullptr) noexcept {
      if (!test_observer.record) { return; }
      event.conversion = conversion;
      event.current_source_stamp_ticks=test_current_source_ticks;
      event.current_content_stamp_ticks=test_current_content_ticks;
      if (matched_presentation_cache.source_timestamp()) event.pixel_source_stamp_ticks=matched_presentation_cache.source_timestamp()->time_since_epoch().count();
      event.draw_source = selected;
      for (std::size_t i=0;i<matched_frame_slots.size();++i) {
        const auto &s = matched_frame_slots[i]; event.slot_frames[i]=s.attribution.frame_id;
        event.slot_pending[i] = &s == pending_matched_frame;
        event.pending_slots += event.slot_pending[i] ? 1u : 0u;
      }
      if (slot) {
        event.model_frame=slot->frame_id;
        event.observation_us=host_sbs_observation_timestamp_us(slot->captured_at);
        const auto &source = colour ? colour->source_timestamp : slot->source_timestamp;
        const auto &content = colour ? colour->content_timestamp : slot->content_timestamp;
        if (source) event.source_stamp_ticks=source->time_since_epoch().count();
        if (content) event.content_stamp_ticks=content->time_since_epoch().count();
      }
      if (result && result->completed_frame_valid) {
        event.publication=result->publication;
        event.publication_current=depth_estimator && depth_estimator->publication_is_current(result->publication);
        if (event.kind!=converter_test_observer::event_e::draw || !event.flat) event.ema_motion_mask=result->ema_motion_mask;
      }
#ifdef SUNSHINE_TESTS
      if (depth_estimator) if (const auto q=depth_estimator->joined_completion_gate_status_for_benchmark()) {
        event.query_observed=q->actual_event_queries!=0u; event.literal_last_query_result=q->last_actual_event_query;
        event.literal_query_target_frame=q->configuration.target_frame_id;
        event.gate_entered=q->callback_entered; event.gate_released=q->released;
      }
#endif
      test_observer.emit(event);
    }

    struct latest_v2_lineage_t {
      host_sbs::observation_handoff_t completion;
      matched_frame_attribution_t attribution;
      std::optional<models::host_sbs_gpu_completion_receipt::receipt_t> receipt;

      [[nodiscard]] const models::depth_observation_views_t &estimate() const noexcept {
        return completion.step().observation;
      }

      [[nodiscard]] bool authenticated() const noexcept {
        return completion.has_authenticated_structure();
      }

      [[nodiscard]] bool known_depth() const noexcept {
        return authenticated() && receipt && receipt->depth_cache_authorized() &&
               models::host_sbs_gpu_completion_receipt::matches_publication(*receipt, estimate().publication);
      }

      D3D11_TEXTURE2D_DESC completed_source_desc {};
      void reset() noexcept {
        *this = {};
      }

      [[nodiscard]] bool route_matches(
        const D3D11_TEXTURE2D_DESC &source_desc,
        const models::input_color_space current_color_space,
        const models::host_subtitle_context::presentation_authority_t current_authority
      ) const noexcept {
        const auto &estimate = this->estimate();
        return authenticated() && estimate.completed_frame_valid &&
               estimate.completed_frame_id == attribution.frame_id && estimate.shadow_base_final_parallax &&
               estimate.shadow_state && estimate.input_region.valid() &&
               depth_completion_region_matches_submission(
                 estimate,
                 attribution.depth_input_region
               ) &&
               attribution.inference_content_timestamp && attribution.color_space == current_color_space &&
               source_signatures_match(completed_source_desc, source_desc) &&
               attribution.observed_authority == current_authority;
      }
    };

    [[nodiscard]] static bool source_signatures_match(
      const D3D11_TEXTURE2D_DESC &left,
      const D3D11_TEXTURE2D_DESC &right
    ) noexcept {
      return left.Width == right.Width && left.Height == right.Height &&
             left.MipLevels == right.MipLevels && left.ArraySize == right.ArraySize &&
             left.Format == right.Format &&
             left.SampleDesc.Count == right.SampleDesc.Count &&
             left.SampleDesc.Quality == right.SampleDesc.Quality;
    }

    [[nodiscard]] static bool matched_source_signature_matches(
      matched_frame_slot_t &slot,
      const D3D11_TEXTURE2D_DESC &source_desc
    ) noexcept {
      return slot.texture && source_signatures_match(slot.texture.description(), source_desc);
    }

    // Shared subtitle warp activation for Direct Glyph, before the first input.
    void adopt_selected_matte_programs(selected_matte_programs_t programs) {
      selected_shared_warp_programs = std::move(programs.shared_warp);
      selected_shared_warp_root = std::filesystem::path(programs.configuration.shader_root);
      selected_matte_ready = programs.ready;
      if (selected_matte_ready && selected_shared_warp_programs.direct_glyph_completed_ready()) {
        BOOST_LOG(info) << "Host SBS Direct Glyph prepare source closure "
                        << selected_shared_warp_programs.model_prepare_closure();
      }
    }

    [[nodiscard]] static bool gpu_transaction_class_matches(
      const models::depth_observation_views_t &estimate,
      const matched_frame_attribution_t &slot
    ) noexcept {
      return estimate.gpu_undecided_completion == slot.gpu_undecided_transaction;
    }

    [[nodiscard]] static std::uint64_t authority_generation(
      const std::optional<foreground_window::snapshot_t> &authority
    ) noexcept {
      return authority ? authority->generation : 0u;
    }

    [[nodiscard]] static std::optional<RECT> damage_region(
      const models::depth_input_region_t &region
    ) noexcept {
      constexpr auto long_max = static_cast<std::uint32_t>(
        std::numeric_limits<LONG>::max()
      );
      if (!region.valid() || region.right > long_max || region.bottom > long_max) {
        return std::nullopt;
      }
      return RECT {
        static_cast<LONG>(region.left),
        static_cast<LONG>(region.top),
        static_cast<LONG>(region.right),
        static_cast<LONG>(region.bottom),
      };
    }

    [[nodiscard]] detail::host_sbs_ddup_reuse_proof_e matched_input_reuse_kind(
      const matched_frame_attribution_t &attribution,
      const std::optional<std::chrono::steady_clock::time_point> &current_content,
      const std::optional<detail::ddup_damage_snapshot_t> &current_damage
    ) const noexcept {
      const auto region = damage_region(attribution.depth_input_region);
      if (!region) {
        return detail::host_sbs_ddup_reuse_proof_e::none;
      }
      if (attribution.depth_input_region.is_video_region()) {
        if (!attribution.inference_content_timestamp || !current_content || !attribution.unchanged_roi_proof.observe(current_damage, *region)) {
          return detail::host_sbs_ddup_reuse_proof_e::none;
        }
        return attribution.inference_content_timestamp == current_content ?
                 detail::host_sbs_ddup_reuse_proof_e::content_clock :
                 detail::host_sbs_ddup_reuse_proof_e::roi_damage;
      }
      return detail::classify_host_sbs_ddup_reuse(
        attribution.inference_content_timestamp,
        attribution.inference_ddup_damage,
        attribution.depth_input_region.is_video_region(),
        *region,
        current_content,
        current_damage
      );
    }

    /** Form the host proof once, after private-copy ROI/authority selection and the late receipt
     * poll. B owns complete DDup source continuity; the receipt separately names actual depth A.
     */
    [[nodiscard]] bool gpu_undecided_candidate_matches(
      matched_frame_slot_t &slot,
      const D3D11_TEXTURE2D_DESC &source_desc
    ) const noexcept {
      const auto current_authority = live_presentation_authority();
      const auto &candidate = slot.attribution;
      const auto &baseline = latest_v2_lineage.attribution;
      if (!latest_v2_lineage.known_depth() || candidate.frame_id <= baseline.frame_id ||
          candidate.observed_authority.interactive_move_size || pending_matched_frame ||
          !candidate.inference_content_timestamp ||
          candidate.inference_content_timestamp == baseline.inference_content_timestamp ||
          !candidate.inference_ddup_damage || !baseline.inference_ddup_damage ||
          candidate.inference_ddup_damage->history != baseline.inference_ddup_damage->history ||
          baseline.inference_ddup_damage->token == 0u ||
          candidate.inference_ddup_damage->token <= baseline.inference_ddup_damage->token ||
          candidate.depth_input_region != baseline.depth_input_region || candidate.color_space != baseline.color_space ||
          !matched_route_matches_current(slot, source_desc, candidate.color_space, current_authority) ||
          !latest_v2_lineage_route_matches_current(source_desc, candidate.color_space, current_authority)) {
        return false;
      }
      return baseline.inference_content_timestamp && damage_region(baseline.depth_input_region) &&
             detail::has_complete_ddup_damage_between(baseline.inference_ddup_damage, candidate.inference_ddup_damage);
    }

    [[nodiscard]] models::host_subtitle_context::presentation_authority_t live_presentation_authority() const noexcept {
      return {
        authority_generation(live_window_authority),
        authority_generation(live_foreground_region),
        live_browser_authority_epoch,
        interactive_move_size_observed,
      };
    }

    [[nodiscard]] bool matched_route_matches_current(
      matched_frame_slot_t &slot,
      const D3D11_TEXTURE2D_DESC &source_desc,
      const models::input_color_space current_color_space,
      const models::host_subtitle_context::presentation_authority_t current_authority
    ) const noexcept {
      const auto &attribution = slot.attribution;
      if (
        !subtitle_context.matches(attribution.subtitle_context_generation) ||
        !attribution.depth_input_region.valid() ||
        attribution.depth_input_region.source_width != source_desc.Width ||
        attribution.depth_input_region.source_height != source_desc.Height ||
        !matched_source_signature_matches(slot, source_desc) ||
        attribution.color_space != current_color_space ||
        attribution.observed_authority != current_authority
      ) {
        return false;
      }
      return (!attribution.depth_input_region.is_video_region() && !attribution.subtitle_context_eligible) ||
             window_region_authorized_for_render(attribution);
    }

    [[nodiscard]] bool latest_v2_lineage_route_matches_current(
      const D3D11_TEXTURE2D_DESC &source_desc,
      const models::input_color_space current_color_space,
      const models::host_subtitle_context::presentation_authority_t current_authority
    ) const noexcept {
      return depth_estimator && subtitle_context.matches(latest_v2_lineage.attribution.subtitle_context_generation) &&
             depth_estimator->publication_is_current(latest_v2_lineage.estimate().publication) &&
             latest_v2_lineage.route_matches(
               source_desc,
               current_color_space,
               current_authority
             ) &&
             ((!latest_v2_lineage.attribution.depth_input_region.is_video_region() &&
               !latest_v2_lineage.attribution.subtitle_context_eligible) ||
              window_region_authorized_for_render(latest_v2_lineage.attribution));
    }

    [[nodiscard]] bool retain_latest_v2_lineage(
      const host_sbs::observation_handoff_t &observation,
      const matched_frame_attribution_t &attribution,
      const D3D11_TEXTURE2D_DESC &completed_source_desc,
      const D3D11_TEXTURE2D_DESC &current_source_desc,
      const models::input_color_space current_color_space
    ) {
      const auto &estimate = observation.step().observation;
      const bool completion_route_matches_current =
        subtitle_context.matches(attribution.subtitle_context_generation) &&
        attribution.inference_content_timestamp && estimate.completed_frame_id == attribution.frame_id &&
        depth_completion_region_matches_submission(estimate, attribution.depth_input_region) &&
        estimate.color_space == attribution.color_space && estimate.color_space == current_color_space &&
        estimate.input_region.source_width == completed_source_desc.Width &&
        estimate.input_region.source_height == completed_source_desc.Height &&
        source_signatures_match(completed_source_desc, current_source_desc) &&
        attribution.observed_authority == live_presentation_authority() &&
        (!attribution.subtitle_context_eligible || window_region_authorized_for_render(attribution)) &&
        (!attribution.depth_input_region.is_video_region() ||
         (attribution.inference_ddup_damage && window_region_authorized_for_render(attribution)));
      if (!detail::host_sbs_latest_v2_completion_retention_allowed(
            host_sbs_renderer == models::host_sbs_renderer_e::parallax_v2,
            observation.authenticate_structure(),
            completion_route_matches_current,
            gpu_transaction_class_matches(estimate, attribution)
          )) {
        latest_v2_lineage.reset();
        return false;
      }

      if (latest_v2_lineage.estimate().publication != estimate.publication) {
        latest_v2_lineage.receipt.reset();
      }
      // Retain observation views and diagnostic status only. Repeated presentation cannot
      // replay a submission or a completion event because neither belongs to this cache.
      latest_v2_lineage.completion.select_retained(observation);
      // The cached draw uses current color. Copy the submitted attribution, including its
      // independent rolling ROI proof. Physical capture ownership stays with the matched slot.
      latest_v2_lineage.attribution = attribution;
      latest_v2_lineage.completed_source_desc = completed_source_desc;
      return true;
    }

    bool ensure_sbs_intermediate_storage(const bool input_is_linear) {
      const DXGI_FORMAT required_format = input_is_linear ?
                                            DXGI_FORMAT_R16G16B16A16_FLOAT :
                                            DXGI_FORMAT_B8G8R8A8_UNORM;
      D3D11_TEXTURE2D_DESC current_desc {};
      if (sbs_intermediate_texture) {
        sbs_intermediate_texture->GetDesc(&current_desc);
        if (current_desc.Format == required_format) {
          return true;
        }
      }

      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = static_cast<UINT>(std::lround(sbs_viewport.Width));
      desc.Height = static_cast<UINT>(std::lround(sbs_viewport.Height));
      desc.MipLevels = 1;
      desc.ArraySize = 1;
      desc.Format = required_format;
      desc.SampleDesc.Count = 1;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
      if (desc.Width == 0 || desc.Height == 0) {
        BOOST_LOG(error) << "Host SBS intermediate has invalid dimensions."sv;
        return false;
      }

      texture2d_t texture;
      render_target_t rtv;
      shader_res_t srv;
      HRESULT status = device->CreateTexture2D(&desc, nullptr, &texture);
      if (SUCCEEDED(status)) {
        status = device->CreateRenderTargetView(texture.get(), nullptr, &rtv);
      }
      if (SUCCEEDED(status)) {
        status = device->CreateShaderResourceView(texture.get(), nullptr, &srv);
      }
      if (FAILED(status)) {
        // FP16 can safely store either linear values or SDR code values because transfer state is
        // tracked separately. Retain an existing FP16 target if only the optional SDR downsizing
        // failed; never retain BGRA8 for genuine linear input.
        if (!input_is_linear && sbs_intermediate_texture && current_desc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
          BOOST_LOG(warning)
            << "Could not resize the Host SBS intermediate to BGRA8 [0x"sv
            << util::hex(status).to_string_view()
            << "]; retaining transfer-safe FP16 storage."sv;
          return true;
        }
        BOOST_LOG(error) << "Failed to create the Host SBS intermediate [0x"sv
                         << util::hex(status).to_string_view() << ']';
        return false;
      }

      sbs_intermediate_texture = std::move(texture);
      sbs_intermediate_rtv = std::move(rtv);
      sbs_intermediate_srv = std::move(srv);
      // The optional MRT qualification binds the exact view pair. A replacement intermediate
      // invalidates that evidence even when its dimensions and format are unchanged.
      for (auto &surface : output_surface_views) {
        surface.p010_y_mrt_targets_checked = false;
        surface.p010_y_mrt_targets_qualified = false;
      }
      sbs_intermediate_is_linear = false;
      packed_output_in_shutter_pair = false;
      matched_presentation_cache.reset();
      BOOST_LOG(info) << "Host SBS intermediate storage finalized as "sv
                      << (input_is_linear ? "FP16 linear-capable."sv : "BGRA8 SDR."sv);
      return true;
    }

    bool prepare_p010_chroma_tiles() {
      UINT view_support = 0;
      D3D11_FEATURE_DATA_FORMAT_SUPPORT2 support {DXGI_FORMAT_R16G16_UNORM, 0};
      if (format != DXGI_FORMAT_P010 ||
          FAILED(device->CheckFormatSupport(support.InFormat, &view_support)) ||
          !(view_support & D3D11_FORMAT_SUPPORT_TYPED_UNORDERED_ACCESS_VIEW) ||
          FAILED(device->CheckFeatureSupport(D3D11_FEATURE_FORMAT_SUPPORT2, &support, sizeof(support))) ||
          !(support.OutFormatSupport2 & D3D11_FORMAT_SUPPORT2_UAV_TYPED_STORE)) {
        return false;
      }

      // Loaded with the display shaders at startup, from the packaged bytecode where the
      // installed source matches it. Without it raster chroma stays available.
      auto *const bytecode = p010_uv_tile_cs_hlsl.get();
      return bytecode && SUCCEEDED(device->CreateComputeShader(
        bytecode->GetBufferPointer(), bytecode->GetBufferSize(), nullptr, &p010_chroma_shader
      ));
    }

    void dispatch_p010_chroma_tiles(ID3D11ShaderResourceView *input, ID3D11UnorderedAccessView *output) {
      // D3D11 treats the two native P010 planes as one resource for binding hazards.
      device_ctx->OMSetRenderTargets(0, nullptr, nullptr);
      ID3D11Buffer *constants[] = {
        color_matrix.get(), nullptr, subsample_offset.get(), nullptr, stream_gamma_transform.get(),
      };
      device_ctx->CSSetConstantBuffers(0, std::size(constants), constants);
      device_ctx->CSSetShader(p010_chroma_shader.Get(), nullptr, 0);
      device_ctx->CSSetShaderResources(0, 1, &input);
      device_ctx->CSSetUnorderedAccessViews(0, 1, &output, nullptr);
      const auto eye_chroma_width = static_cast<UINT>(out_UV_viewport.Width) / 2u;
      const auto chroma_height = static_cast<UINT>(out_UV_viewport.Height);
      device_ctx->Dispatch((eye_chroma_width + 15u) / 16u, (chroma_height + 7u) / 8u, 2u);
      output = nullptr;
      input = nullptr;
      device_ctx->CSSetUnorderedAccessViews(0, 1, &output, nullptr);
      device_ctx->CSSetShaderResources(0, 1, &input);
      device_ctx->CSSetShader(nullptr, nullptr, 0);
    }

    bool qualify_p010_y_mrt_targets() {
      if (active_output_surface >= output_surface_views.size()) {
        return false;
      }
      auto &surface = output_surface_views[active_output_surface];
      auto &p010_y_mrt_targets_checked = surface.p010_y_mrt_targets_checked;
      auto &p010_y_mrt_targets_qualified = surface.p010_y_mrt_targets_qualified;
      if (p010_y_mrt_targets_checked) {
        return p010_y_mrt_targets_qualified;
      }
      p010_y_mrt_targets_checked = true;
      p010_y_mrt_targets_qualified = false;
      if (!sbs_intermediate_rtv || !surface.y_or_yuv) {
        return false;
      }

      ID3D11RenderTargetView *requested[] = {
        sbs_intermediate_rtv.get(),
        surface.y_or_yuv.get(),
      };
      device_ctx->OMSetRenderTargets((UINT) std::size(requested), requested, nullptr);
      ID3D11RenderTargetView *observed[] = {nullptr, nullptr};
      device_ctx->OMGetRenderTargets((UINT) std::size(observed), observed, nullptr);
      p010_y_mrt_targets_qualified =
        observed[0] == requested[0] && observed[1] == requested[1];
      for (auto *view : observed) {
        if (view) {
          view->Release();
        }
      }
      device_ctx->OMSetRenderTargets(0, nullptr, nullptr);

      if (p010_y_mrt_targets_qualified) {
        BOOST_LOG(info)
          << "Host SBS HDR P010 warp+luma MRT enabled; chroma retains the canonical "sv
             "intermediate-based pass."sv;
      } else {
        BOOST_LOG(warning)
          << "Host SBS HDR P010 warp+luma MRT target qualification failed; retaining the "sv
             "established RGB-to-P010 path."sv;
      }
      return p010_y_mrt_targets_qualified;
    }

    matched_frame_slot_t *available_matched_slot(
      const matched_frame_slot_t *reserved_render_slot = nullptr
    ) {
      for (auto &slot : matched_frame_slots) {
        if (&slot != pending_matched_frame && !slot.capture_image && &slot != reserved_render_slot) {
          return &slot;
        }
      }
      return nullptr;
    }

    static models::depth_input_region_t full_depth_input_region(
      const D3D11_TEXTURE2D_DESC &source_desc
    ) noexcept {
      const auto tensor_shape = models::fit_host_sbs_v2_depth_tensor_shape(
        source_desc.Width,
        source_desc.Height
      );
      return models::depth_input_region_t {
        .source_width = source_desc.Width,
        .source_height = source_desc.Height,
        .left = 0u,
        .top = 0u,
        .right = source_desc.Width,
        .bottom = source_desc.Height,
        .tensor_content = {
          0u,
          0u,
          static_cast<std::uint32_t>(std::max(tensor_shape.width, 0)),
          static_cast<std::uint32_t>(std::max(tensor_shape.height, 0)),
        },
        .analysis_generation = 0u,
        .authority = models::depth_analysis_authority_e::full_source,
      };
    }

    std::uint64_t depth_analysis_generation(
      const models::depth_analysis_domain_key_t &key
    ) noexcept {
      return depth_analysis_generation_tracker.select(key);
    }

    struct browser_live_authority_key_t {
      video_dom::status_e status {video_dom::status_e::starting};
      std::uint64_t window = 0u;
      std::uint32_t process_id = 0u;
      std::int32_t document_id = 0;
      std::int32_t video_id = 0;
      video_dom::rect_t screen_rect {};
      std::chrono::steady_clock::time_point geometry_valid_since {};

      bool operator==(const browser_live_authority_key_t &) const = default;
    };

    struct live_window_authority_observation_t {
      foreground_window::snapshot_t foreground;
      std::optional<foreground_window::capture_target_t> capture_target;
      foreground_window::mapping_result_t foreground_projection {
        .status = foreground_window::mapping_status_e::invalid_capture_target,
      };
      video_dom::snapshot_ptr browser_snapshot;
      std::optional<video_dom::mapping_result_t> browser_projection;
      std::optional<browser_live_authority_key_t> browser_authority;
      std::uint64_t browser_authority_epoch = 0u;
    };

    struct subtitle_admission_diagnostic_key_t {
      std::uint64_t context_generation = 0u;
      std::string_view reason;
      foreground_window::status_e foreground_status {};
      foreground_window::mapping_status_e foreground_mapping_status {};
      video_dom::status_e browser_status {};
      video_dom::mapping_status_e browser_mapping_status {};
      std::uint64_t foreground_window = 0u;
      std::uint32_t foreground_process = 0u;
      std::uint64_t browser_window = 0u;
      std::uint32_t browser_process = 0u;
      std::int32_t browser_document = 0;
      std::int32_t browser_video = 0;
      bool foreground_root = false;
      bool foreground_region = false;
      bool foreground_target_usable = false;
      bool foreground_usable = false;
      bool content_timestamp_available = false;
      bool browser_snapshot = false;
      bool browser_heartbeat_fresh = false;
      bool browser_after_move = false;
      bool browser_authority = false;
      bool browser_identity_matches = false;
      bool browser_usable = false;
      bool mapping_attempted = false;
      bool mapping_succeeded = false;
      bool fullscreen_checked = false;
      bool fullscreen_match = false;
      bool registered_player_candidate = false;
      bool registry_ready = false;
      bool retained_for_interactive_move_size = false;

      bool operator==(const subtitle_admission_diagnostic_key_t &) const = default;
    };

    void observe_live_browser_authority(
      const video_dom::snapshot_ptr &browser
    ) {
      std::optional<browser_live_authority_key_t> next;
      if (browser) {
        constexpr auto maximum_age = std::chrono::milliseconds {2500};
        if (
          video_dom::carries_video_geometry(browser->status) &&
          browser->received_at >= browser_roi_not_before &&
          video_dom::usable(*browser, std::chrono::steady_clock::now(), maximum_age)
        ) {
          next = browser_live_authority_key_t {
            .status = browser->status,
            .window = browser->window,
            .process_id = browser->process_id,
            .document_id = browser->document_id,
            .video_id = browser->video_id,
            .screen_rect = browser->screen_rect,
            .geometry_valid_since = browser->geometry_valid_since,
          };
        }
      }
      if (live_browser_authority == next) {
        return;
      }
      const auto same_analysis_identity_shape = [](const auto &left, const auto &right) {
        const auto width = [](const video_dom::rect_t rect) {
          return static_cast<std::int64_t>(rect.right) - rect.left;
        };
        const auto height = [](const video_dom::rect_t rect) {
          return static_cast<std::int64_t>(rect.bottom) - rect.top;
        };
        return left.window == right.window && left.process_id == right.process_id &&
               left.document_id == right.document_id && left.video_id == right.video_id &&
               video_dom::geometry_authority_class(left.status) ==
                 video_dom::geometry_authority_class(right.status) &&
               width(left.screen_rect) == width(right.screen_rect) &&
               height(left.screen_rect) == height(right.screen_rect);
      };
      const bool analysis_rearm =
        live_browser_authority &&
        (!next || !same_analysis_identity_shape(*live_browser_authority, *next));
      live_browser_authority = next;
      if (++live_browser_authority_epoch == 0u) {
        ++live_browser_authority_epoch;
      }
      if (
        matched_presentation_cache.has_authority(
          models::depth_analysis_authority_e::chromium_video
        ) || matched_presentation_cache.has_positioned_subtitles()
      ) {
        clear_cached_roi_output();
      }
      if (analysis_rearm || !next) {
        depth_analysis_generation_tracker.select_full_source();
      }
    }

    static bool same_window_analysis_identity_shape(
      const foreground_window::snapshot_t &left,
      const foreground_window::snapshot_t &right
    ) noexcept {
      const auto width = [](const foreground_window::rect_t rect) {
        return static_cast<std::int64_t>(rect.right) - rect.left;
      };
      const auto height = [](const foreground_window::rect_t rect) {
        return static_cast<std::int64_t>(rect.bottom) - rect.top;
      };
      return left.window == right.window && left.process_id == right.process_id &&
             left.monitor == right.monitor && left.source_window() == right.source_window() &&
             width(left.source_screen_rect()) == width(right.source_screen_rect()) &&
             height(left.source_screen_rect()) == height(right.source_screen_rect());
    }

    void clear_cached_roi_output() noexcept {
      // Full-capture depth may still have rendered positioned subtitle treatment. Its packed
      // pixels must leave with the old window authority even while semantic memory survives.
      if (!matched_presentation_cache.is_video_region() && !matched_presentation_cache.has_positioned_subtitles()) {
        return;
      }
      matched_presentation_cache.reset();
    }

    live_window_authority_observation_t observe_live_window_authority(
      const D3D11_TEXTURE2D_DESC &source_desc,
      const std::optional<std::chrono::steady_clock::time_point> &content_timestamp,
      const foreground_window::snapshot_t *revalidate_expected = nullptr,
      bool sampling_for_attribution = false
    ) {
      if (test_context) {
        if (source_desc.Width != test_context->width || source_desc.Height != test_context->height || !test_context->instance) {
          subtitle_context.update({});
        } else {
          subtitle_context.update({models::host_subtitle_context::kind_e::registered_video_player,
            test_context->instance, GetCurrentProcessId()});
        }
        // Deliberately no actual USER32/browser/player claim. Only this labelled fixture
        // instance uses the synthetic scheduling domain; source/receipt authority stays real.
        live_window_authority.reset(); live_foreground_region.reset();
        live_browser_authority.reset(); live_browser_authority_epoch = 0u;
        interactive_move_size_observed = false;
        return {};
      }
      foreground_window::snapshot_t observed;
      if (content_timestamp) {
        const auto observation = revalidate_expected ?
                                   foreground_window::sample_after_copy(
                                     *revalidate_expected
                                   ) :
                                   foreground_window::sample();
        observed = foreground_window_tracker.update(observation);
      } else {
        const bool interactive_move_size =
          foreground_window::interactive_move_size_active();
        observed = foreground_window_tracker.update({
          .status = interactive_move_size ?
                      foreground_window::status_e::interactive_move_size :
                      foreground_window::status_e::no_foreground,
          .observed_at = std::chrono::steady_clock::now(),
        });
      }

      const bool interactive_move_size =
        observed.status == foreground_window::status_e::interactive_move_size;
      if (interactive_move_size) {
        if (!interactive_move_size_observed) {
          BOOST_LOG(info)
            << "Host SBS selected full-source 3D for an interactive foreground-window "sv
               "move/resize."sv;
          if (pending_matched_frame && pending_matched_frame->attribution.depth_input_region.is_video_region()) {
            // Consuming this exact-source ROI completion does not enqueue a replacement. Ask
            // the retained-source owner for one more conversion so full-source depth starts
            // even if USER32 stops producing desktop damage during the modal move loop.
            depth_authority_reprocess_pending = true;
          }
        }
        interactive_move_size_observed = true;
        browser_roi_not_before = observed.observed_at;
      } else if (interactive_move_size_observed) {
        BOOST_LOG(info)
          << "Host SBS foreground-window move/resize ended; waiting for fresh ROI authority."sv;
        interactive_move_size_observed = false;
        browser_roi_not_before = observed.observed_at;
      }

      std::optional<foreground_window::snapshot_t> next_root;
      if (foreground_window::carries_geometry(observed.status)) {
        next_root = observed;
      }
      std::optional<foreground_window::snapshot_t> next_region;
      foreground_window::mapping_result_t foreground_mapping {
        .status = foreground_window::mapping_status_e::invalid_capture_target,
      };
      std::optional<foreground_window::capture_target_t> capture_target;
      foreground_window::mapping_result_t foreground_projection {
        .status = foreground_window::mapping_status_e::invalid_capture_target,
      };
      const bool can_map_live_target = display &&
        display->display_rotation == DXGI_MODE_ROTATION_IDENTITY &&
        static_cast<std::uint32_t>(display->width) == source_desc.Width &&
        static_cast<std::uint32_t>(display->height) == source_desc.Height;
      DXGI_OUTPUT_DESC output_desc {};
      if (next_root && display && display->output && (can_map_live_target || sampling_for_attribution) &&
          SUCCEEDED(display->output->GetDesc(&output_desc))) {
        capture_target = foreground_window::capture_target_t {
          .screen_rect = {
            output_desc.DesktopCoordinates.left,
            output_desc.DesktopCoordinates.top,
            output_desc.DesktopCoordinates.right,
            output_desc.DesktopCoordinates.bottom,
          },
          .width = source_desc.Width,
          .height = source_desc.Height,
          .monitor = reinterpret_cast<std::uintptr_t>(output_desc.Monitor),
          .identity_orientation = display->display_rotation == DXGI_MODE_ROTATION_IDENTITY,
        };
        foreground_projection = foreground_window::map_to_capture(observed, *capture_target);
      }
      if (capture_target && can_map_live_target) {
        // Keep liveness's stricter target gate and its diagnostic status. Attribution below
        // reuses the raw projection, including unsupported orientation/extent failures.
        // Liveness follows the current root/client/content geometry, not the content-causality
        // gate. A move must revoke an old positioned completion immediately, while the later
        // private-copy path still requires geometry_valid_since <= content_timestamp before
        // authorizing a new foreground ROI. This separation preserves the position-independent
        // DAV2 domain.
        foreground_mapping = foreground_projection;
        if (foreground_mapping) {
          next_region = observed;
        }
      }

      const auto changed = [](const auto &before, const auto &after) {
        return before.has_value() != after.has_value() ||
               (before && after && before->generation != after->generation);
      };
      const bool programmatic_causal_gap =
        live_foreground_region && next_region &&
        live_foreground_region->generation != next_region->generation &&
        foreground_window::requires_full_source_causal_fallback(
          *live_foreground_region,
          *next_region,
          content_timestamp
        );
      if (programmatic_causal_gap) {
        // This observation runs even while TensorRT owns the prior ROI slot. Keep requesting
        // retained-source conversions after that old completion is retired, so one exact
        // full-source submission occurs even if Desktop Duplication never presents again.
        depth_authority_reprocess_pending = true;
      }
      const bool root_changed = changed(live_window_authority, next_root);
      const bool region_changed = changed(live_foreground_region, next_region);
      const bool analysis_source_description_changed = [&]() {
        if (!next_root) {
          return false;
        }
        if (!live_window_authority) {
          return true;
        }
        const auto relative_source = [](const foreground_window::snapshot_t &snapshot) {
          const auto source = snapshot.source_screen_rect();
          return foreground_window::rect_t {
            source.left - snapshot.client_screen_rect.left,
            source.top - snapshot.client_screen_rect.top,
            source.right - snapshot.client_screen_rect.left,
            source.bottom - snapshot.client_screen_rect.top,
          };
        };
        return live_window_authority->window != next_root->window ||
               live_window_authority->process_id != next_root->process_id ||
               live_window_authority->source_window() != next_root->source_window() ||
               relative_source(*live_window_authority) != relative_source(*next_root);
      }();
      const bool analysis_authority_changed =
        live_window_authority &&
        (!next_root ||
         !same_window_analysis_identity_shape(*live_window_authority, *next_root));
      if (root_changed || region_changed) {
        clear_cached_roi_output();
        if (analysis_source_description_changed) {
          const auto source_rect = next_root->source_screen_rect();
          const bool structural_child = next_root->source_window() != next_root->window;
          BOOST_LOG(info)
            << "Host SBS foreground analysis source changed: source="sv
            << (structural_child ? "structural-child"sv : "whole-client"sv)
            << " root_client="sv << next_root->client_screen_rect.left << ':'
            << next_root->client_screen_rect.top << '-'
            << next_root->client_screen_rect.right << ':'
            << next_root->client_screen_rect.bottom << " analysis="sv
            << source_rect.left << ':' << source_rect.top << '-'
            << source_rect.right << ':' << source_rect.bottom << " root_hwnd=0x"sv
            << util::hex(next_root->window).to_string_view() << " content_hwnd=0x"sv
            << util::hex(next_root->source_window()).to_string_view()
            << " generation="sv << next_root->generation << '.';
        }
      }
      if (
        analysis_authority_changed ||
        (live_foreground_region && !next_region) ||
        !next_root
      ) {
        depth_analysis_generation_tracker.select_full_source();
      }
      live_window_authority = next_root;
      live_foreground_region = next_region;
      // Read the helper exactly once for this Win32 observation. The immutable shared snapshot,
      // the live epoch derived from it, and any later ROI mapping remain one attribution unit.
      const auto browser_snapshot = video_dom_lease ? video_dom_lease->latest() : nullptr;
      observe_live_browser_authority(browser_snapshot);
      const auto subtitle_admission = observe_subtitle_context(
        observed, browser_snapshot, output_desc, source_desc, content_timestamp, foreground_mapping);
      return {
        .foreground = observed,
        .capture_target = capture_target,
        .foreground_projection = foreground_projection,
        .browser_snapshot = browser_snapshot,
        .browser_projection = subtitle_admission.browser_mapping,
        .browser_authority = live_browser_authority,
        .browser_authority_epoch = live_browser_authority_epoch,
      };
    }

    // The image path of the process that owns the foreground window. A window belongs to one
    // process for its whole life, so the path is asked for again only for another window or
    // process, and once a second in case both identifiers were reused together. Only the path
    // is kept: who may have subtitle protection is still decided on every observation.
    const std::optional<std::wstring> &subtitle_process_image_path(
      const foreground_window::snapshot_t &foreground,
      const std::chrono::steady_clock::time_point now
    ) {
      auto &cached = subtitle_process_image;
      if (cached.window != foreground.window || cached.process_id != foreground.process_id ||
          now - cached.queried_at >= std::chrono::seconds(1)) {
        cached = {foreground.window, foreground.process_id, now,
                  foreground_window::process_image_path(foreground.process_id)};
      }
      return cached.path;
    }

    // Subtitle permission is independent of the generic foreground depth ROI. Fullscreen keeps
    // its semantic video identity even when depth analysis canonicalizes to the whole source.
    subtitle_context_policy::evaluation_t observe_subtitle_context(
      const foreground_window::snapshot_t &foreground,
      const video_dom::snapshot_ptr &browser,
      const DXGI_OUTPUT_DESC &output,
      const D3D11_TEXTURE2D_DESC &source,
      const std::optional<std::chrono::steady_clock::time_point> &content_timestamp,
      const foreground_window::mapping_result_t &foreground_mapping
    ) {
      using namespace models::host_subtitle_context;
      key_t next;
      video_player_registry_ready_observed = video_player_registry && video_player_registry->ready();
      const auto now = std::chrono::steady_clock::now();
      const bool foreground_region_present = static_cast<bool>(live_foreground_region);
      const auto admission = subtitle_context_policy::evaluate(
        foreground.status == foreground_window::status_e::interactive_move_size ? &foreground :
          live_window_authority ? &*live_window_authority : nullptr,
        foreground_region_present, foreground_mapping, browser.get(),
        live_browser_authority ? live_browser_authority_epoch : 0u,
        {output.DesktopCoordinates.left, output.DesktopCoordinates.top,
         output.DesktopCoordinates.right, output.DesktopCoordinates.bottom},
        source.Width, source.Height, content_timestamp, now, subtitle_context.key());
      const bool foreground_usable = admission.foreground_usable;
      const bool browser_identity_matches = admission.browser_identity_matches;
      const bool browser_usable = admission.browser_usable;
      const bool mapping_attempted = admission.browser_mapping.has_value();
      const auto mapping_status = mapping_attempted ?
        admission.browser_mapping->status : video_dom::mapping_status_e::invalid_video_rect;
      const bool mapping_succeeded = mapping_attempted && static_cast<bool>(*admission.browser_mapping);
      const bool fullscreen_check_evaluated = admission.fullscreen_check_evaluated;
      const bool fullscreen_match = admission.fullscreen_match;
      const bool browser_video = admission.browser_key.kind == kind_e::browser_video;
      if (browser_video) {
        next = admission.browser_key;
      } else if (admission.registered_player_candidate) {
        // A Game 3D text game is an application its owner approved for exactly this purpose. The
        // approval names that process: the game's answer is refreshed a few times a second, so
        // another window that has just come in front must not inherit it meanwhile.
        // Unless its helper has lately shown that it sees the text, the picture-based search
        // covers it.
        const bool approved_text_game = text_game_ai_active &&
                                        text_game_pin.picture_search_covers(foreground.process_id, std::chrono::steady_clock::now());
        const auto registered_executables = video_player_registry ?
          video_player_registry->executables_if_ready() : std::span<const std::wstring> {};
        const auto executable = !registered_executables.empty() || !approved_text_game ?
          subtitle_process_image_path(foreground, now) : std::nullopt;
        // Registration keeps precedence. Both policies inspect the same process image while
        // the owner's current settings still take effect on every authority observation.
        const bool registered = executable && video_player::matches_registered_executable(registered_executables, *executable);
        const bool owner_enabled = !registered &&
          (approved_text_game || (executable && host3d_subtitle_apps::observe_executable(*executable)));
        next = {native_kind(registered, owner_enabled), foreground.window, foreground.process_id};
      }
      const bool context_changed = subtitle_context.update(next);
      if (context_changed) {
        // Revoke even full-source completions and already-packed output. The generation survives
        // busy drops, so returning to the same video cannot revive pre-transition protection.
        matched_presentation_cache.reset();
        latest_v2_lineage.reset();
        subtitle_context_reprocess_pending = true;
      }
      const char *reason = "no-video-authority";
      if (subtitle_context.kind() == kind_e::browser_video) {
        reason = "browser-video-authorized";
      } else if (subtitle_context.kind() == kind_e::registered_video_player) {
        reason = "registered-player-authorized";
      } else if (subtitle_context.kind() == kind_e::foreground_app) {
        reason = "foreground-app-authorized";
      } else if (!live_window_authority) {
        reason = "foreground-root-unavailable";
      } else if (!foreground_usable) {
        reason = "foreground-unusable-for-content";
      } else if (!admission.foreground_target_usable) {
        reason = "foreground-target-unavailable";
      } else if (!foreground_region_present && !browser_identity_matches) {
        reason = "foreground-region-unavailable";
      } else if (!browser) {
        reason = "browser-snapshot-unavailable";
      } else if (!live_browser_authority) {
        reason = "browser-authority-unavailable";
      } else if (!browser_identity_matches) {
        reason = "browser-identity-mismatch";
      } else if (!browser_usable) {
        reason = "browser-unusable-for-content";
      } else if (!mapping_succeeded) {
        reason = "browser-mapping-failed";
      } else if (!fullscreen_match) {
        reason = "browser-fullscreen-mismatch";
      } else if (browser_video) {
        reason = "browser-key-rejected";
      }
      const bool heartbeat_age_available = browser &&
                                           browser->received_at.time_since_epoch().count() != 0;
      const bool browser_heartbeat_fresh = heartbeat_age_available && now >= browser->received_at &&
                                           now - browser->received_at <= std::chrono::milliseconds {2500};
      const bool browser_after_move = browser && browser->received_at >= browser_roi_not_before;
      const subtitle_admission_diagnostic_key_t diagnostic_key {
        .context_generation = subtitle_context.generation(),
        .reason = reason,
        .foreground_status = foreground.status,
        .foreground_mapping_status = foreground_mapping.status,
        .browser_status = browser ? browser->status : video_dom::status_e::stopped,
        .browser_mapping_status = mapping_status,
        .foreground_window = foreground.window,
        .foreground_process = foreground.process_id,
        .browser_window = browser ? browser->window : 0u,
        .browser_process = browser ? browser->process_id : 0u,
        .browser_document = browser ? browser->document_id : 0,
        .browser_video = browser ? browser->video_id : 0,
        .foreground_root = static_cast<bool>(live_window_authority),
        .foreground_region = foreground_region_present,
        .foreground_target_usable = admission.foreground_target_usable,
        .foreground_usable = foreground_usable,
        .content_timestamp_available = content_timestamp.has_value(),
        .browser_snapshot = static_cast<bool>(browser),
        .browser_heartbeat_fresh = browser_heartbeat_fresh,
        .browser_after_move = browser_after_move,
        .browser_authority = static_cast<bool>(live_browser_authority),
        .browser_identity_matches = browser_identity_matches,
        .browser_usable = browser_usable,
        .mapping_attempted = mapping_attempted,
        .mapping_succeeded = mapping_succeeded,
        .fullscreen_checked = fullscreen_check_evaluated,
        .fullscreen_match = fullscreen_match,
        .registered_player_candidate = admission.registered_player_candidate,
        .registry_ready = video_player_registry_ready_observed,
        .retained_for_interactive_move_size = admission.retained_for_interactive_move_size,
      };
      // Admission can change between disabled reasons without changing semantic context. Keep
      // that evidence visible without resetting history or logging every heartbeat/geometry tick.
      // Rapid nonsemantic fluctuations coalesce to the latest state at most once per second;
      // actual context transitions retain their immediate log and existing revocation semantics.
      if (!context_changed && last_subtitle_admission_diagnostic &&
          (*last_subtitle_admission_diagnostic == diagnostic_key ||
           now - last_subtitle_admission_diagnostic_at < std::chrono::seconds {1})) {
        return admission;
      }
      last_subtitle_admission_diagnostic = diagnostic_key;
      last_subtitle_admission_diagnostic_at = now;
      const double heartbeat_age_ms = heartbeat_age_available ?
                                        std::chrono::duration<double, std::milli>(now - browser->received_at).count() :
                                        0.0;
      const bool geometry_content_delta_available = browser && content_timestamp &&
                                                    browser->geometry_valid_since.time_since_epoch().count() != 0;
      const double content_minus_geometry_ms = geometry_content_delta_available ?
                                                 std::chrono::duration<double, std::milli>(*content_timestamp - browser->geometry_valid_since).count() :
                                                 0.0;
      BOOST_LOG(info) << "Host SBS subtitle context="sv
                      << kind_name(subtitle_context.kind())
                      << " generation="sv << subtitle_context.generation() << '.'
                      << " reason="sv << reason
                      << " context_changed="sv << context_changed
                      << " foreground_status="sv << foreground_window::status_name(foreground.status)
                      << " foreground_root="sv << static_cast<bool>(live_window_authority)
                      << " foreground_region="sv << foreground_region_present
                      << " foreground_target_usable="sv << admission.foreground_target_usable
                      << " foreground_mapping_status="sv << foreground_window::mapping_status_name(foreground_mapping.status)
                      << " foreground_usable="sv << foreground_usable
                      << " foreground_hwnd=0x"sv << util::hex(foreground.window).to_string_view()
                      << " foreground_pid="sv << foreground.process_id
                      << " foreground_generation="sv << foreground.generation
                      << " foreground_client="sv << foreground.client_screen_rect.left << ':'
                      << foreground.client_screen_rect.top << '-'
                      << foreground.client_screen_rect.right << ':' << foreground.client_screen_rect.bottom
                      << " capture="sv << output.DesktopCoordinates.left << ':'
                      << output.DesktopCoordinates.top << '-' << output.DesktopCoordinates.right << ':'
                      << output.DesktopCoordinates.bottom
                      << " source="sv << source.Width << 'x' << source.Height
                      << " content_timestamp_available="sv << content_timestamp.has_value()
                      << " content_age_ms="sv << (content_timestamp ? std::chrono::duration<double, std::milli>(now - *content_timestamp).count() : 0.0)
                      << " browser_snapshot="sv << static_cast<bool>(browser)
                      << " browser_status="sv << (browser ? video_dom::status_name(browser->status) : "none")
                      << " live_browser_authority="sv << static_cast<bool>(live_browser_authority)
                      << " browser_heartbeat_fresh="sv << browser_heartbeat_fresh
                      << " browser_after_move="sv << browser_after_move
                      << " browser_epoch="sv << live_browser_authority_epoch
                      << " browser_identity_checked="sv << (foreground_usable && admission.foreground_target_usable)
                      << " browser_identity_match="sv << browser_identity_matches
                      << " browser_usable_checked="sv << (foreground_usable && browser_identity_matches)
                      << " browser_usable="sv << browser_usable
                      << " mapping_attempted="sv << mapping_attempted
                      << " mapping_status="sv << (mapping_attempted ? video_dom::mapping_status_name(mapping_status) : "not-evaluated")
                      << " fullscreen_checked="sv << fullscreen_check_evaluated
                      << " fullscreen_match="sv << fullscreen_match
                      << " registered_player_candidate="sv << admission.registered_player_candidate
                      << " registry_ready="sv << video_player_registry_ready_observed
                      << " retained_for_interactive_move_size="sv << admission.retained_for_interactive_move_size
                      << " browser_sequence="sv << (browser ? browser->helper_sequence : 0u)
                      << " browser_hwnd=0x"sv << util::hex(browser ? browser->window : 0u).to_string_view()
                      << " browser_pid="sv << (browser ? browser->process_id : 0u)
                      << " browser_document="sv << (browser ? browser->document_id : 0)
                      << " browser_video="sv << (browser ? browser->video_id : 0)
                      << " browser_rect="sv << (browser ? browser->screen_rect.left : 0) << ':'
                      << (browser ? browser->screen_rect.top : 0) << '-'
                      << (browser ? browser->screen_rect.right : 0) << ':'
                      << (browser ? browser->screen_rect.bottom : 0)
                      << " heartbeat_age_available="sv << heartbeat_age_available
                      << " heartbeat_age_ms="sv << heartbeat_age_ms
                      << " geometry_content_delta_available="sv << geometry_content_delta_available
                      << " content_minus_geometry_ms="sv << content_minus_geometry_ms;
      return admission;
    }

    bool window_region_matches_live_authority(
      const sbs_debug::window_region_snapshot &region
    ) const noexcept {
      const auto &authority =
        region.authority_kind == sbs_debug::window_region_authority_kind_e::chromium_video ?
          live_window_authority :
          live_foreground_region;
      return authority && authority->window == region.hwnd &&
             authority->process_id == region.process_id;
    }

    bool window_region_authorized_for_render(
      const matched_frame_attribution_t &slot
    ) const noexcept {
      if (test_context) {
        return slot.test_instance == test_context->instance && slot.test_instance != 0u &&
          slot.depth_input_region.source_width == test_context->width &&
          slot.depth_input_region.source_height == test_context->height &&
          subtitle_context.matches(slot.subtitle_context_generation);
      }
      if (!slot.window_region || !window_region_matches_live_authority(*slot.window_region) || slot.live_window_authority_generation == 0u) {
        return false;
      }
      if (
        slot.window_region->authority_kind ==
        sbs_debug::window_region_authority_kind_e::chromium_video
      ) {
        const auto browser = video_dom_lease ? video_dom_lease->latest() : nullptr;
        return live_window_authority->generation == slot.live_window_authority_generation &&
               slot.live_browser_authority_generation == live_browser_authority_epoch &&
               browser && video_dom::carries_video_geometry(browser->status) &&
               browser->window == slot.window_region->hwnd &&
               browser->process_id == slot.window_region->process_id &&
               browser->document_id == slot.window_region->document_id &&
               browser->video_id == slot.window_region->video_id &&
               browser->screen_rect.left == slot.live_browser_screen_left &&
               browser->screen_rect.top == slot.live_browser_screen_top &&
               browser->screen_rect.right == slot.live_browser_screen_right &&
               browser->screen_rect.bottom == slot.live_browser_screen_bottom &&
               browser->geometry_valid_since ==
                 slot.live_browser_geometry_valid_since &&
               video_dom::geometry_authority_class(browser->status) ==
                 slot.live_browser_authority_class;
      }
      return live_foreground_region->generation == slot.live_window_authority_generation;
    }

    void select_depth_input_region(
      matched_frame_attribution_t &slot,
      const D3D11_TEXTURE2D_DESC &source_desc,
      const bool finalize_full_source = true
    ) {
      slot.depth_video_plan.reset();
      slot.depth_input_region = full_depth_input_region(source_desc);

      if (slot.window_region) {
        const auto &region = *slot.window_region;
        const models::depth_source_rect_t semantic_rect {
          static_cast<std::uint32_t>(region.left),
          static_cast<std::uint32_t>(region.top),
          static_cast<std::uint32_t>(region.right),
          static_cast<std::uint32_t>(region.bottom),
        };
        const auto authority =
          region.authority_kind == sbs_debug::window_region_authority_kind_e::chromium_video ?
            models::depth_analysis_authority_e::chromium_video :
            models::depth_analysis_authority_e::foreground_client;
        const auto make_domain_key =
          [&](const models::depth_source_rect_t analysis_rect) {
            return models::depth_analysis_domain_key_t {
              .source_width = source_desc.Width,
              .source_height = source_desc.Height,
              .semantic_width = semantic_rect.width(),
              .semantic_height = semantic_rect.height(),
              .crop_width = analysis_rect.width(),
              .crop_height = analysis_rect.height(),
              .hwnd = region.hwnd,
              .process_id = region.process_id,
              .document_id = region.document_id,
              .video_id = region.video_id,
              .authority = authority,
            };
          };
        const auto active_shape = models::fit_host_sbs_v2_depth_tensor_shape(
          source_desc.Width,
          source_desc.Height
        );
        auto plan = models::plan_host_sbs_v2_video_region(
          semantic_rect,
          source_desc.Width,
          source_desc.Height,
          active_shape
        );
        if (plan) {
          const auto key = make_domain_key(plan->source_rect);
          slot.depth_input_region = models::depth_input_region_t {
            .source_width = source_desc.Width,
            .source_height = source_desc.Height,
            .left = plan->source_rect.left,
            .top = plan->source_rect.top,
            .right = plan->source_rect.right,
            .bottom = plan->source_rect.bottom,
            .tensor_content = plan->tensor_content,
            .analysis_generation = depth_analysis_generation(key),
            .authority = authority,
          };
          slot.depth_video_plan = std::move(plan);
        }
      }

      if (!slot.depth_input_region.is_video_region()) {
        if (finalize_full_source) {
          depth_analysis_generation_tracker.select_full_source();
        }
        if (
          slot.window_region &&
          !(
            slot.window_region->left == 0 && slot.window_region->top == 0 &&
            slot.window_region->right == slot.window_region->source_width &&
            slot.window_region->bottom == slot.window_region->source_height
          )
        ) {
          slot.window_region.reset();
        }
      }
      if (
        matched_presentation_cache.has_fresh_pixels() &&
        !matched_presentation_cache.matches_analysis_domain(
          slot.depth_input_region,
          slot.color_space
        )
      ) {
        matched_presentation_cache.reset();
      }
    }

    template<typename CaptureRect>
    static sbs_debug::window_region_snapshot make_window_region_snapshot(
      sbs_debug::window_region_snapshot identity,
      const D3D11_TEXTURE2D_DESC &source_desc,
      const std::uint64_t frame_id,
      const CaptureRect &capture_pixels,
      const std::chrono::steady_clock::time_point observed_at,
      const std::chrono::steady_clock::time_point geometry_valid_since,
      const std::chrono::steady_clock::time_point content_timestamp,
      const std::chrono::steady_clock::time_point captured_at,
      const std::chrono::milliseconds maximum_age
    ) noexcept {
      identity.matched_frame_id = frame_id;
      identity.source_width = source_desc.Width;
      identity.source_height = source_desc.Height;
      identity.left = capture_pixels.left;
      identity.top = capture_pixels.top;
      identity.right = capture_pixels.right;
      identity.bottom = capture_pixels.bottom;
      identity.latest_observation_age_ms_at_capture = static_cast<std::uint32_t>(
        std::clamp<std::int64_t>(
          std::chrono::duration_cast<std::chrono::milliseconds>(
            captured_at - observed_at
          )
            .count(),
          0,
          std::numeric_limits<std::uint32_t>::max()
        )
      );
      identity.maximum_observation_age_ms = static_cast<std::uint32_t>(
        maximum_age.count()
      );
      identity.geometry_continuity_ms_at_capture = static_cast<std::uint64_t>(
        std::max<std::int64_t>(
          std::chrono::duration_cast<std::chrono::milliseconds>(
            captured_at - geometry_valid_since
          )
            .count(),
          0
        )
      );
      identity.source_content_age_ms_at_capture = static_cast<std::uint64_t>(
        std::max<std::int64_t>(
          std::chrono::duration_cast<std::chrono::milliseconds>(
            captured_at - content_timestamp
          )
            .count(),
          0
        )
      );
      return identity;
    }

    bool has_pending_depth_pipeline_build() const {
      return depth_estimator_build.valid();
    }

    std::optional<sbs_debug::window_region_snapshot>
      capture_browser_window_region(
        const D3D11_TEXTURE2D_DESC &source_desc,
        const live_window_authority_observation_t &authority,
        const std::uint64_t frame_id,
        const std::optional<std::chrono::steady_clock::time_point> &content_timestamp,
        const std::chrono::steady_clock::time_point captured_at,
        video_dom::status_e &video_status,
        video_dom::mapping_status_e &mapping_status
      ) const {
      const auto &observed = authority.browser_snapshot;
      video_status = video_dom::status_e::stopped;
      mapping_status = video_dom::mapping_status_e::invalid_video_rect;
      if (!observed || !display) {
        return std::nullopt;
      }
      video_status = observed->status;
      if (observed->received_at < browser_roi_not_before) {
        if (video_dom::carries_video_geometry(video_status)) {
          video_status = video_dom::status_e::stale;
        }
        return std::nullopt;
      }
      constexpr auto maximum_age = std::chrono::milliseconds {2500};
      // The newest heartbeat proves liveness; the uninterrupted exact-geometry run proves
      // causality. An identical heartbeat may accompany unchanged pixels, but a newly changed
      // identity or rectangle must never be attached retroactively to an older matched frame.
      if (
        !video_dom::detail::usable_for_content(
          *observed,
          content_timestamp,
          captured_at,
          maximum_age
        )
      ) {
        if (video_dom::carries_video_geometry(video_status)) {
          video_status = video_dom::status_e::stale;
        }
        return std::nullopt;
      }
      const auto observed_window = reinterpret_cast<HWND>(
        static_cast<std::uintptr_t>(observed->window)
      );
      DWORD foreground_process_id = 0;
      const auto foreground_window = GetAncestor(GetForegroundWindow(), GA_ROOT);
      if (
        !observed_window || !IsWindow(observed_window) ||
        foreground_window != observed_window ||
        GetWindowThreadProcessId(observed_window, &foreground_process_id) == 0 ||
        foreground_process_id != observed->process_id
      ) {
        video_status = video_dom::status_e::changed;
        mapping_status = video_dom::mapping_status_e::foreground_mismatch;
        return std::nullopt;
      }
      if (
        display->display_rotation != DXGI_MODE_ROTATION_IDENTITY ||
        display->width <= 0 || display->height <= 0 ||
        static_cast<std::uint32_t>(display->width) != source_desc.Width ||
        static_cast<std::uint32_t>(display->height) != source_desc.Height
      ) {
        mapping_status =
          display->display_rotation == DXGI_MODE_ROTATION_IDENTITY ?
            video_dom::mapping_status_e::extent_mismatch :
            video_dom::mapping_status_e::unsupported_rotation;
        return std::nullopt;
      }

      if (!authority.capture_target) {
        mapping_status = video_dom::mapping_status_e::invalid_capture_rect;
        return std::nullopt;
      }
      const auto &target = *authority.capture_target;
      const video_dom::rect_t desktop_rect {
        target.screen_rect.left,
        target.screen_rect.top,
        target.screen_rect.right,
        target.screen_rect.bottom,
      };
      const auto mapped = authority.browser_projection ?
                            *authority.browser_projection :
                            video_dom::map_screen_rect_to_capture(
                              observed->screen_rect, desktop_rect,
                              source_desc.Width, source_desc.Height,
                              video_dom::rotation_e::identity);
      mapping_status = mapped.status;
      if (!mapped) {
        return std::nullopt;
      }

      return make_window_region_snapshot(
        {
          .authority_kind = sbs_debug::window_region_authority_kind_e::chromium_video,
          .hwnd = observed->window,
          .process_id = observed->process_id,
          .document_id = observed->document_id,
          .video_id = observed->video_id,
          .generation = observed->generation,
        },
        source_desc,
        frame_id,
        mapped.capture_pixels,
        observed->received_at,
        observed->geometry_valid_since,
        *content_timestamp,
        captured_at,
        maximum_age
      );
    }

    std::optional<sbs_debug::window_region_snapshot>
      capture_foreground_window_region(
        const D3D11_TEXTURE2D_DESC &source_desc,
        const std::uint64_t frame_id,
        const std::optional<std::chrono::steady_clock::time_point> &content_timestamp,
        const std::chrono::steady_clock::time_point captured_at,
        const live_window_authority_observation_t &authority,
        std::string_view &observer_status,
        std::string_view &mapping_status
      ) {
      const auto &snapshot = authority.foreground;
      constexpr auto maximum_age = std::chrono::milliseconds {250};
      observer_status = foreground_window::status_name(snapshot.status);
      mapping_status = "not-mapped";
      if (
        !display ||
        !foreground_window::usable_for_content(
          snapshot,
          content_timestamp,
          captured_at,
          maximum_age
        )
      ) {
        return std::nullopt;
      }

      const auto &mapped = authority.foreground_projection;
      mapping_status = foreground_window::mapping_status_name(mapped.status);
      if (!mapped) {
        return std::nullopt;
      }

      return make_window_region_snapshot(
        {
          .authority_kind = sbs_debug::window_region_authority_kind_e::foreground_client,
          .hwnd = snapshot.window,
          .process_id = snapshot.process_id,
          .generation = snapshot.generation,
        },
        source_desc,
        frame_id,
        mapped.capture_pixels,
        snapshot.observed_at,
        snapshot.geometry_valid_since,
        *content_timestamp,
        captured_at,
        maximum_age
      );
    }

    static bool same_window_region(
      const sbs_debug::window_region_snapshot &left,
      const sbs_debug::window_region_snapshot &right
    ) noexcept {
      return left.authority_kind == right.authority_kind &&
             left.source_width == right.source_width &&
             left.source_height == right.source_height &&
             left.left == right.left && left.top == right.top &&
             left.right == right.right && left.bottom == right.bottom &&
             left.hwnd == right.hwnd && left.process_id == right.process_id &&
             left.document_id == right.document_id && left.video_id == right.video_id;
    }

    void publish_window_region_transition(const matched_frame_attribution_t &slot) {
      const bool observer_state_changed =
        !last_window_region_observer_status ||
        *last_window_region_observer_status != slot.window_region_observer_status ||
        !last_window_region_mapping_status ||
        *last_window_region_mapping_status != slot.window_region_mapping_status;
      if (slot.window_region) {
        if (
          !last_window_region ||
          !same_window_region(*last_window_region, *slot.window_region) ||
          observer_state_changed
        ) {
          const auto &region = *slot.window_region;
          BOOST_LOG(info)
            << "Host SBS window region attributed to matched frame "sv
            << region.matched_frame_id << ": authority="sv
            << sbs_debug::window_region_authority_kind_name(region.authority_kind)
            << " capture="sv << region.left << ':' << region.top << '-'
            << region.right << ':' << region.bottom << " source="sv
            << region.source_width << 'x' << region.source_height << " hwnd=0x"sv
            << util::hex(region.hwnd).to_string_view() << " pid="sv
            << region.process_id << " document="sv << region.document_id
            << " video="sv << region.video_id << " observation_age_ms="sv
            << region.latest_observation_age_ms_at_capture
            << " geometry_before_content_ms="sv
            << region.geometry_continuity_ms_at_capture -
                 region.source_content_age_ms_at_capture
            << '.';
          if (slot.depth_video_plan) {
            const auto &plan = *slot.depth_video_plan;
            BOOST_LOG(info)
              << "Host SBS window-region DAV2 analysis active: crop="sv
              << plan.source_rect.left << ':' << plan.source_rect.top << '-'
              << plan.source_rect.right << ':' << plan.source_rect.bottom
              << " tensor="sv << plan.tensor_shape.width << 'x'
              << plan.tensor_shape.height << " letterbox_padding_pct="sv
              << plan.padded_area_fraction * 100.0f
              << "; the scene center is region-local."sv;
          } else if (
            region.left == 0 && region.top == 0 &&
            region.right == region.source_width &&
            region.bottom == region.source_height
          ) {
            BOOST_LOG(info)
              << "Host SBS window region covers the exact capture extent; using canonical "sv
                 "ordinary full-frame V2."sv;
          } else if (
            region.authority_kind ==
              sbs_debug::window_region_authority_kind_e::chromium_video &&
            slot.window_region_observer_status == "ok-fullscreen"
          ) {
            BOOST_LOG(info)
              << "Host SBS fullscreen-only browser-client authority does not map to the exact "sv
                 "capture extent; the windowed ROI remains disabled."sv;
          } else {
            BOOST_LOG(info)
              << "Host SBS window region is not eligible for the active authenticated tensor; "sv
                 "using ordinary full-frame V2."sv;
          }
        }
        last_window_region = slot.window_region;
        last_window_region_observer_status = slot.window_region_observer_status;
        last_window_region_mapping_status = slot.window_region_mapping_status;
        return;
      }

      if (last_window_region) {
        BOOST_LOG(info)
          << "Host SBS window region released at matched frame "sv
          << slot.frame_id << " (observer_status="sv
          << slot.window_region_observer_status
          << ", mapping_status="sv << slot.window_region_mapping_status
          << "); full-frame V2 remains active."sv;
        last_window_region.reset();
      } else if (observer_state_changed) {
        BOOST_LOG(debug)
          << "Host SBS window region unavailable at matched frame "sv
          << slot.frame_id << " (observer_status="sv
          << slot.window_region_observer_status
          << ", mapping_status="sv << slot.window_region_mapping_status
          << "); full-frame V2 remains active."sv;
      }
      last_window_region_observer_status = slot.window_region_observer_status;
      last_window_region_mapping_status = slot.window_region_mapping_status;
    }

    bool ensure_private_matched_texture(
      matched_frame_slot_t &slot,
      const D3D11_TEXTURE2D_DESC &source_desc
    ) {
      if (slot.srv && matched_source_signature_matches(slot, source_desc)) {
        return true;
      }

      auto private_desc = source_desc;
      private_desc.Usage = D3D11_USAGE_DEFAULT;
      private_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      private_desc.CPUAccessFlags = 0;
      private_desc.MiscFlags = 0;

      texture2d_t texture;
      auto status = device->CreateTexture2D(&private_desc, nullptr, &texture);
      if (FAILED(status)) {
        BOOST_LOG(error) << "Failed to create matched-frame texture [0x"sv
                         << util::hex(status).to_string_view() << ']';
        return false;
      }

      shader_res_t srv;
      status = device->CreateShaderResourceView(texture.get(), nullptr, &srv);
      if (FAILED(status)) {
        BOOST_LOG(error) << "Failed to create matched-frame resource view [0x"sv
                         << util::hex(status).to_string_view() << ']';
        return false;
      }
      slot.texture.reset(std::move(texture));
      slot.srv = std::move(srv);
      return true;
    }

    bool prepare_matched_frame(
      encoder_img_ctx_t &image_context,
      const D3D11_TEXTURE2D_DESC &source_desc,
      matched_frame_slot_t &slot,
      std::uint64_t frame_id,
      models::input_color_space color_space,
      std::optional<std::chrono::steady_clock::time_point> source_timestamp,
      std::optional<std::chrono::steady_clock::time_point> content_timestamp,
      std::optional<detail::ddup_damage_snapshot_t> ddup_damage,
      const live_window_authority_observation_t &initial_source_authority,
      sbs_gpu_timer_slot_t *gpu_timer
    ) {
      auto *source = image_context.encoder_texture.get();
      auto *input = image_context.encoder_input_res.get();
      if (!source || !input) {
        return false;
      }

      const bool borrow_capture = !test_context || test_context->borrow_matched_capture;
      if (borrow_capture) {
        slot.capture_image = image_context.img_weak.lock();
      }
      if (slot.capture_image) {
        input->AddRef();
        slot.texture.borrow_from(image_context.encoder_texture);
        slot.srv.reset(input);
        slot.capture_mutex = image_context.encoder_mutex.get();
      } else if (!ensure_private_matched_texture(slot, source_desc)) {
        return false;
      }

      const auto captured_at = std::chrono::steady_clock::now();
      if (slot.capture_image) {
        mark_sbs_matched_copy_end(gpu_timer, false);
      } else {
        mark_sbs_matched_copy_start(gpu_timer);
        device_ctx->CopyResource(slot.texture.get(), source);
        mark_sbs_matched_copy_end(gpu_timer, true);
      }
      // Recheck live authority after securing the exact source. During an authenticated native
      // move/resize the existing observation already forces full-frame analysis. Ordinary
      // whole-client observations revalidate their exact identity and geometry; structural
      // child authority still requires the full census to revalidate sibling uniqueness.
      const bool initial_interactive_move_size =
        initial_source_authority.foreground.status ==
        foreground_window::status_e::interactive_move_size;
      const auto matched_authority = initial_interactive_move_size ?
                                      initial_source_authority :
                                      observe_live_window_authority(
                                        source_desc,
                                        content_timestamp,
                                        &initial_source_authority.foreground,
                                        true
                                      );
      const auto &matched_foreground_snapshot = matched_authority.foreground;
      // A pre-drag ROI completion may still own the estimator after USER32 stops producing desktop
      // damage. Full-source analysis is independent of window geometry, so submit these exact
      // retained pixels once without letting post-drag ROI causality block completion retirement.
      const bool preexisting_force_full_source =
        depth_authority_reprocess_pending || initial_interactive_move_size;
      auto &attribution = slot.attribution;
      const auto attributed_at = std::chrono::steady_clock::now();
      attribution.frame_id = frame_id;
      attribution.color_space = color_space;
      attribution.source_timestamp = source_timestamp;
      attribution.content_timestamp = content_timestamp;
      attribution.inference_content_timestamp = content_timestamp;
      attribution.inference_ddup_damage = std::move(ddup_damage);
      // Source age bounds older matched completions. Packed-pixel continuity separately excludes
      // exact-static time before its first changed source. Age diagnostics remain opt-in.
      attribution.captured_at = captured_at;
      attribution.window_region.reset();
      attribution.window_region_observer_status = "not-observed";
      attribution.window_region_mapping_status = "not-mapped";
      attribution.live_browser_geometry_valid_since = {};
      if (initial_interactive_move_size) {
        attribution.window_region_observer_status = foreground_window::status_name(
          foreground_window::status_e::interactive_move_size
        );
      }
      video_dom::status_e browser_status = video_dom::status_e::stopped;
      video_dom::mapping_status_e browser_mapping_status =
        video_dom::mapping_status_e::invalid_video_rect;
      const bool foreground_causal_wait =
        live_foreground_region &&
        foreground_window::requires_full_source_causal_fallback(
          matched_foreground_snapshot,
          *live_foreground_region,
          content_timestamp
        );
      const bool force_full_source =
        preexisting_force_full_source || foreground_causal_wait;
      if (foreground_causal_wait) {
        // The captured desktop content predates this same-size programmatic move. Desktop
        // Duplication may never publish another content timestamp after the move, so submit this
        // exact frame through full-source V2 instead of starving depth indefinitely.
        attribution.window_region_observer_status = "causal-wait-full-source";
        attribution.window_region_mapping_status = "not-mapped";
      }
      auto browser_region = force_full_source ?
                              std::optional<sbs_debug::window_region_snapshot> {} :
                              capture_browser_window_region(
                                source_desc,
                                matched_authority,
                                frame_id,
                                content_timestamp,
                                captured_at,
                                browser_status,
                                browser_mapping_status
                              );
      const bool browser_fullscreen_only_mismatch =
        browser_region && browser_status == video_dom::status_e::ok_fullscreen &&
        !(
          browser_region->left == 0 && browser_region->top == 0 &&
          browser_region->right == browser_region->source_width &&
          browser_region->bottom == browser_region->source_height
        );
      if (browser_fullscreen_only_mismatch) {
        browser_region.reset();
      }
      if (browser_region && !window_region_matches_live_authority(*browser_region)) {
        browser_status = video_dom::status_e::changed;
        browser_mapping_status = video_dom::mapping_status_e::foreground_mismatch;
        browser_region.reset();
      }

      if (browser_region) {
        attribution.window_region = std::move(browser_region);
        attribution.window_region_observer_status = video_dom::status_name(browser_status);
        attribution.window_region_mapping_status =
          video_dom::mapping_status_name(browser_mapping_status);
      } else if (!force_full_source) {
        std::string_view foreground_observer_status;
        std::string_view foreground_mapping_status;
        attribution.window_region = capture_foreground_window_region(
          source_desc,
          frame_id,
          content_timestamp,
          attributed_at,
          matched_authority,
          foreground_observer_status,
          foreground_mapping_status
        );
        attribution.window_region_observer_status = foreground_observer_status;
        attribution.window_region_mapping_status = foreground_mapping_status;
      }
      const bool retry_foreground_after_browser =
        attribution.window_region &&
        attribution.window_region->authority_kind ==
          sbs_debug::window_region_authority_kind_e::chromium_video &&
        !(
          attribution.window_region->left == 0 && attribution.window_region->top == 0 &&
          attribution.window_region->right == attribution.window_region->source_width &&
          attribution.window_region->bottom == attribution.window_region->source_height
        );
      select_depth_input_region(
        attribution,
        source_desc,
        !retry_foreground_after_browser
      );
      if (!attribution.depth_input_region.is_video_region() && retry_foreground_after_browser) {
        std::string_view foreground_observer_status;
        std::string_view foreground_mapping_status;
        attribution.window_region = capture_foreground_window_region(
          source_desc,
          frame_id,
          content_timestamp,
          attributed_at,
          matched_authority,
          foreground_observer_status,
          foreground_mapping_status
        );
        attribution.window_region_observer_status = foreground_observer_status;
        attribution.window_region_mapping_status = foreground_mapping_status;
        select_depth_input_region(attribution, source_desc);
      }
      attribution.live_window_authority_generation =
        attribution.window_region && window_region_matches_live_authority(*attribution.window_region) ?
          (
            attribution.window_region->authority_kind ==
                sbs_debug::window_region_authority_kind_e::chromium_video ?
              live_window_authority->generation :
              live_foreground_region->generation
          ) :
          0u;
      attribution.live_browser_authority_generation =
        attribution.window_region &&
            attribution.window_region->authority_kind ==
              sbs_debug::window_region_authority_kind_e::chromium_video ?
          matched_authority.browser_authority_epoch :
          0u;
      if (attribution.live_browser_authority_generation != 0u && video_dom_lease) {
        if (matched_authority.browser_authority) {
          attribution.live_browser_screen_left =
            matched_authority.browser_authority->screen_rect.left;
          attribution.live_browser_screen_top =
            matched_authority.browser_authority->screen_rect.top;
          attribution.live_browser_screen_right =
            matched_authority.browser_authority->screen_rect.right;
          attribution.live_browser_screen_bottom =
            matched_authority.browser_authority->screen_rect.bottom;
          attribution.live_browser_geometry_valid_since =
            matched_authority.browser_authority->geometry_valid_since;
          attribution.live_browser_authority_class =
            video_dom::geometry_authority_class(
              matched_authority.browser_authority->status
            );
        }
      }
      // Capture the complete live route epoch even for a full-source frame. A later unchanged-
      // content reuse may retain that geometry only while the same focus/ROI/move authority is
      // still live; otherwise a new matched copy must select the current route.
      attribution.observed_authority = {
        authority_generation(live_window_authority),
        authority_generation(live_foreground_region),
        matched_authority.browser_authority_epoch,
        interactive_move_size_observed,
      };
      attribution.subtitle_context_generation = subtitle_context.generation();
      attribution.subtitle_context_suppression_eligible =
        attribution.observed_authority.interactive_move_size && subtitle_context.eligible();
      attribution.test_instance = test_context ? test_context->instance : 0u;
      attribution.subtitle_line_policy =
        models::host_subtitle_context::requires_centered_lines(subtitle_context.kind()) ?
          models::subtitle_line_policy_e::video_caption :
          models::subtitle_line_policy_e::application_text;
      attribution.subtitle_context_eligible = test_context ?
        subtitle_context.eligible() && source_desc.Width == test_context->width && source_desc.Height == test_context->height :
        subtitle_context.eligible() && attribution.window_region &&
        window_region_matches_live_authority(*attribution.window_region) &&
        ((subtitle_context.kind() == models::host_subtitle_context::kind_e::browser_video) ==
         (attribution.window_region->authority_kind == sbs_debug::window_region_authority_kind_e::chromium_video));
      if (attribution.depth_input_region.is_video_region() && attribution.live_window_authority_generation == 0u) {
        attribution.window_region.reset();
        attribution.window_region_observer_status = "not-observed";
        attribution.window_region_mapping_status = "not-mapped";
        select_depth_input_region(attribution, source_desc);
      }
      if (
        !sbs_telemetry_input_domain_valid ||
        // Telemetry admission is keyed before estimator submission, so both operands retain the
        // display-selected raster. Completed estimator/dump metadata carries est.input_region.
        !attribution.depth_input_region.same_analysis_domain(sbs_telemetry_input_region) ||
        attribution.color_space != sbs_telemetry_input_color_space
      ) {
        // Stop an old-domain staging readback from publishing while the renderer is deliberately
        // flat and waiting for this newly observed ROI/full/transfer domain. The first completion
        // in the new domain establishes the new health-counter baseline.
        sbs_telemetry_input_region = attribution.depth_input_region;
        sbs_telemetry_input_color_space = attribution.color_space;
        sbs_telemetry_input_domain_valid = true;
        sbs_telemetry_min_frame_id = frame_id;
      }
      attribution.gpu_undecided_transaction = false;
      const auto proof_region = damage_region(attribution.depth_input_region);
      if (attribution.depth_input_region.is_video_region() && proof_region) {
        attribution.unchanged_roi_proof.reset(attribution.inference_ddup_damage, *proof_region);
      } else {
        attribution.unchanged_roi_proof.reset();
      }
      return true;
    }

    void mark_sbs_matched_copy_start(sbs_gpu_timer_slot_t *slot) {
      if (slot && !slot->matched_copy_started) {
        slot->matched_copy_started = true;
        device_ctx->End(slot->matched_copy_start.get());
      }
    }

    void mark_sbs_matched_copy_end(sbs_gpu_timer_slot_t *slot, bool submitted) {
      if (slot) {
        if (slot->matched_copy_ended) {
          // Preserve the fact that CopyResource was submitted if a later authority/result path
          // closes the logical copy as rejected. A timestamp query must be ended exactly once.
          slot->has_matched_copy = slot->has_matched_copy || submitted;
          return;
        }
        // Skipped copies still close every query in the timing tuple, but only an actual
        // CopyResource interval is published as matched_frame_copy_gpu.
        mark_sbs_matched_copy_start(slot);
        slot->matched_copy_ended = true;
        slot->has_matched_copy = slot->has_matched_copy || submitted;
        device_ctx->End(slot->matched_copy_end.get());
      }
    }

    void mark_sbs_warp_start(sbs_gpu_timer_slot_t *slot, bool has_sbs_composite) {
      if (slot) {
        slot->has_sbs_composite = has_sbs_composite;
        device_ctx->End(slot->warp_start.get());
      }
    }

    bool ensure_sbs_debug_geometry_resources() {
      if (!sbs_intermediate_texture) {
        return false;
      }
      D3D11_TEXTURE2D_DESC intermediate_desc {};
      sbs_intermediate_texture->GetDesc(&intermediate_desc);
      if (sbs_debug_geometry_ready) {
        D3D11_TEXTURE2D_DESC mapping_desc {};
        D3D11_TEXTURE2D_DESC mask_desc {};
        if (sbs_debug_mapping_texture) {
          sbs_debug_mapping_texture->GetDesc(&mapping_desc);
        }
        if (sbs_debug_mask_texture) {
          sbs_debug_mask_texture->GetDesc(&mask_desc);
        }
        const bool cached_resources_match =
          sbs_debug_mapping_texture && sbs_debug_mapping_rtv && sbs_debug_mapping_srv &&
          sbs_debug_mask_texture && sbs_debug_mask_rtv && sbs_debug_mask_srv &&
          mapping_desc.Width == intermediate_desc.Width &&
          mapping_desc.Height == intermediate_desc.Height &&
          mapping_desc.Format == DXGI_FORMAT_R32_FLOAT &&
          mapping_desc.MipLevels == intermediate_desc.MipLevels &&
          mapping_desc.ArraySize == intermediate_desc.ArraySize &&
          mapping_desc.SampleDesc.Count == intermediate_desc.SampleDesc.Count &&
          mapping_desc.SampleDesc.Quality == intermediate_desc.SampleDesc.Quality &&
          mask_desc.Width == intermediate_desc.Width &&
          mask_desc.Height == intermediate_desc.Height &&
          mask_desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM &&
          mask_desc.MipLevels == intermediate_desc.MipLevels &&
          mask_desc.ArraySize == intermediate_desc.ArraySize &&
          mask_desc.SampleDesc.Count == intermediate_desc.SampleDesc.Count &&
          mask_desc.SampleDesc.Quality == intermediate_desc.SampleDesc.Quality;
        if (cached_resources_match) {
          return true;
        }

        // The packed intermediate can be resized/reformatted without reinitializing the output
        // object. Never render dump-only evidence into stale targets from the prior extent.
        BOOST_LOG(warning)
          << "Dump 3D geometry resources no longer match the packed output; recreating them."sv;
        sbs_debug_v2_mapping_ps.Reset();
        sbs_debug_v2_mask_ps.Reset();
        sbs_debug_mapping_texture.Reset();
        sbs_debug_mapping_rtv.Reset();
        sbs_debug_mapping_srv.Reset();
        sbs_debug_mask_texture.Reset();
        sbs_debug_mask_rtv.Reset();
        sbs_debug_mask_srv.Reset();
        sbs_debug_geometry_ready = false;
        sbs_debug_geometry_init_attempted = false;
      }
      if (sbs_debug_geometry_init_attempted) {
        return false;
      }
      sbs_debug_geometry_init_attempted = true;

      namespace cache = models::host_sbs_shader_cache;
      const auto diagnostic_sources = cache::snapshot_sources(
        SUNSHINE_SHADERS_DIR,
        cache::parallax_v2_live_diagnostic_specs
      );
      const bool diagnostic_sources_authenticated =
        cache::source_closure_sha256(diagnostic_sources) ==
        cache::parallax_v2_live_diagnostic_source_closure_sha256;
      const auto mapping_bytecode = diagnostic_sources_authenticated ?
                                      cache::get(diagnostic_sources, cache::parallax_v2_live_mapping) :
                                      cache::bytecode_t {};
      const auto mask_bytecode = diagnostic_sources_authenticated ?
                                   cache::get(diagnostic_sources, cache::parallax_v2_live_mask) :
                                   cache::bytecode_t {};
      const bool geometry_shaders_ready =
        mapping_bytecode && mask_bytecode &&
        SUCCEEDED(device->CreatePixelShader(
          mapping_bytecode->data(),
          mapping_bytecode->size(),
          nullptr,
          &sbs_debug_v2_mapping_ps
        )) &&
        SUCCEEDED(device->CreatePixelShader(
          mask_bytecode->data(),
          mask_bytecode->size(),
          nullptr,
          &sbs_debug_v2_mask_ps
        ));
      if (!geometry_shaders_ready) {
        BOOST_LOG(warning) << "Dump 3D geometry shaders are unavailable; the core dump will "
                              "record warp mapping/coverage as unavailable."sv;
        return false;
      }

      D3D11_TEXTURE2D_DESC desc = intermediate_desc;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.CPUAccessFlags = 0;
      desc.MiscFlags = 0;
      desc.Format = DXGI_FORMAT_R32_FLOAT;
      desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
      if (FAILED(device->CreateTexture2D(&desc, nullptr, &sbs_debug_mapping_texture)) || FAILED(device->CreateRenderTargetView(sbs_debug_mapping_texture.Get(), nullptr, &sbs_debug_mapping_rtv)) || FAILED(device->CreateShaderResourceView(sbs_debug_mapping_texture.Get(), nullptr, &sbs_debug_mapping_srv))) {
        BOOST_LOG(warning) << "Dump 3D warp-map resources are unavailable."sv;
        return false;
      }

      desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
      if (FAILED(device->CreateTexture2D(&desc, nullptr, &sbs_debug_mask_texture)) || FAILED(device->CreateRenderTargetView(sbs_debug_mask_texture.Get(), nullptr, &sbs_debug_mask_rtv)) || FAILED(device->CreateShaderResourceView(sbs_debug_mask_texture.Get(), nullptr, &sbs_debug_mask_srv))) {
        BOOST_LOG(warning) << "Dump 3D coverage-mask resources are unavailable."sv;
        return false;
      }

      sbs_debug_geometry_ready = true;
      return true;
    }

    bool render_shared_ui_debug_geometry(
      const host_sbs_v2_draw_command_t &actual_draw,
      sbs_debug::actual_draw_evidence &evidence
    ) {
      std::string error;
      if (!evidence.shared_pin_selected || !sbs_intermediate_texture || !selected_shared_warp_programs.load_diagnostics(device.get(), selected_shared_warp_root, error)) {
        return false;
      }
      D3D11_TEXTURE2D_DESC desc {};
      sbs_intermediate_texture->GetDesc(&desc);
      desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
      desc.CPUAccessFlags = desc.MiscFlags = 0u;
      std::array<Microsoft::WRL::ComPtr<ID3D11Texture2D>, 2u> textures;
      std::array<Microsoft::WRL::ComPtr<ID3D11RenderTargetView>, 2u> targets;
      std::array<Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>, 2u> views;
      for (std::size_t i = 0u; i < textures.size(); ++i) {
        if (FAILED(device->CreateTexture2D(&desc, nullptr, &textures[i])) || FAILED(device->CreateRenderTargetView(textures[i].Get(), nullptr, &targets[i])) || FAILED(device->CreateShaderResourceView(textures[i].Get(), nullptr, &views[i]))) {
          return false;
        }
      }
      auto draw = actual_draw;
      draw.render_target_count = 1u;
      draw.render_targets[1] = nullptr;
      draw.bound_draw_observer_state = nullptr;
      draw.observe_bound_draw = nullptr;
      device_ctx->OMSetBlendState(blend_disable.get(), nullptr, 0xFFFFFFFFu);
      const std::array<ID3D11PixelShader *, 2u> shaders {
        selected_shared_warp_programs.mapping(),
        selected_shared_warp_programs.boundary()
      };
      for (std::size_t i = 0u; i < targets.size(); ++i) {
        draw.render_targets[0] = targets[i].Get();
        draw.pixel_shader = shaders[i];
        if (!record_host_sbs_v2_draw(device_ctx.get(), draw)) {
          return false;
        }
      }
      evidence.shared_pin_mapping = views[0];
      evidence.shared_pin_boundary = views[1];
      return true;
    }

    bool render_sbs_debug_geometry(
      const std::array<ID3D11ShaderResourceView *, 7u> &inputs,
      ID3D11Buffer *constants,
      ID3D11Buffer *subtitle_constants
    ) {
      if (!inputs[0] || !inputs[1] || !inputs[2] || !constants || !ensure_sbs_debug_geometry_resources()) {
        return false;
      }
      device_ctx->OMSetBlendState(blend_disable.get(), nullptr, 0xFFFFFFFFu);
      host_sbs_v2_draw_command_t command {
        .render_targets = {sbs_debug_mapping_rtv.Get(), nullptr},
        .render_target_count = 1u,
        .vertex_shader = sbs_reprojection_vs.get(),
        .pixel_shader = sbs_debug_v2_mapping_ps.Get(),
        .viewport = sbs_viewport,
        .sampler = sampler_linear.get(),
        .shader_resources = inputs,
        .geometry_constants = constants,
        .subtitle_constants = subtitle_constants,
      };
      if (!record_host_sbs_v2_draw(device_ctx.get(), command)) {
        return false;
      }
      command.render_targets[0] = sbs_debug_mask_rtv.Get();
      command.pixel_shader = sbs_debug_v2_mask_ps.Get();
      return record_host_sbs_v2_draw(device_ctx.get(), command);
    }

    void reset_matched_stats(std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now()) {
      matched_stats_started = now;
      matched_stats_calls = 0;
      matched_stats_completions = 0;
      matched_stats_repeats = 0;
      matched_stats_content_skips = 0;
      matched_stats_content_reuses = 0;
      matched_stats_damage_skips = 0;
      matched_stats_damage_reuses = 0;
      matched_stats_video_roi_route_outputs = 0;
      matched_stats_roi_direct_inputs = 0;
      matched_stats_roi_dump_copies = 0;
      matched_stats_output_warped_new = 0;
      matched_stats_output_packed_repeat = 0;
      matched_stats_output_flat = 0;
      matched_stats_output_p010_y_mrt = 0;
      matched_stats_output_flat_pending_miss = 0;
      matched_stats_same_frame_poll_attempts = 0;
      matched_stats_same_frame_poll_hits = 0;
      matched_stats_same_frame_poll_timeouts = 0;
      matched_stats_same_frame_poll_submissions = 0;
      matched_stats_same_frame_poll_cadence_ineligible_busy = 0;
      matched_stats_same_frame_poll_wait_unavailable_busy = 0;
      matched_stats_publication_late_ready = 0;
      matched_stats_force_missing_publication = 0;
      matched_stats_publication_failure_polls = 0;
      matched_stats_same_frame_poll_plans = 0;
      matched_stats_same_frame_poll_hard_cap_timeouts = 0;
      matched_stats_same_frame_poll_cadence_timeouts = 0;
      matched_stats_same_frame_poll_failures = 0;
      matched_stats_same_frame_immediate_hits = 0;
      matched_stats_same_frame_poll_hits_le_2_ms = 0;
      matched_stats_same_frame_poll_hits_2_to_2_5_ms = 0;
      matched_stats_same_frame_poll_hits_2_5_to_3_ms = 0;
      matched_stats_same_frame_poll_hits_over_3_ms = 0;
      matched_stats_same_frame_poll_queries = 0;
      matched_stats_same_frame_poll_wait_sum_ms = 0.0;
      matched_stats_same_frame_poll_wait_max_ms = 0.0;
      matched_stats_same_frame_poll_budget_sum_ms = 0.0;
      matched_stats_same_frame_poll_budget_max_ms = 0.0;
      matched_stats_age_sum_ms = 0.0;
      matched_stats_age_max_ms = 0.0;
    }

    // What init_output() sets a converter up for. Named fields: several neighbours share a type,
    // and a local presenter leaves most of them at their defaults.
    struct output_setup_t {
      ID3D11Texture2D *frame_texture = nullptr;
      int width = 0;
      int height = 0;
      int sbs_mode_param = ::video::SBS_OFF;
      config::video_t::sbs_t settings {};
      safe::mail_raw_t::event_t<int> depth_status_event {};
      std::shared_ptr<safe::event_t<bool>> depth_pipeline_ready_event {};
      safe::mail_raw_t::event_t<::video::sbs_telemetry_snapshot_t> telemetry_event {};
      std::shared_ptr<::video::sbs_telemetry_subscription_t> telemetry_subscription {};
      std::uint32_t telemetry_generation = 0;
      std::shared_ptr<std::atomic<bool>> sbs_debug_dump_request {};
      bool rgb_only = false;
      std::shared_ptr<host_sbs_telemetry::collector> telemetry_performance {};
      safe::mail_raw_t::event_t<::video::game_source_state_t> game_status_event {};
      std::shared_ptr<::video::effective_video_mode_publisher_t> effective_mode {};
      bool game_transport_supported = false;
      int game_width = 0;
      int game_height = 0;
      nvenc::nvenc_d3d11_native *native_input = nullptr;
    };

    int init_output(output_setup_t setup) {
      capture_damage_observer.reset();
      native_input_owner = setup.rgb_only ? nullptr : setup.native_input;
      output_surface_views.clear();
      active_output_surface = 0;
      if (setup.frame_texture) {
        // The underlying frame pool owns the texture, so we must reference it for ourselves.
        setup.frame_texture->AddRef();
        output_texture.reset(setup.frame_texture);
      } else if (!setup.rgb_only) {
        BOOST_LOG(error) << "Missing encoder output texture."sv;
        return -1;
      }
      sbs_mode = setup.sbs_mode_param;
      sbs_config = setup.settings;
      local_presentation = setup.rgb_only;
      game_export.status_event = std::move(setup.game_status_event);
      game_export.effective_mode = std::move(setup.effective_mode);
      game_export.source_width = setup.game_width;
      game_export.source_height = setup.game_height;
      game_export.output_width = setup.width;
      game_export.output_height = setup.height;
      game_export.source_tracker = {};
      game_export.source_supported = setup.game_transport_supported &&
                              (display->display_rotation == DXGI_MODE_ROTATION_UNSPECIFIED ||
                               display->display_rotation == DXGI_MODE_ROTATION_IDENTITY);
#ifdef SUNSHINE_TESTS
      game_export.setup_watch = ::video::is_game_mode(sbs_mode) ?
                           model_preparation_watch_for_test({display->offset_x, display->offset_y, display->offset_x + display->width, display->offset_y + display->height}) :
                           nullptr;
#else
      game_export.setup_watch = ::video::is_game_mode(sbs_mode) ?
                           platf::game3d::watch_game_setup(display->screen_rect()) :
                           nullptr;
#endif
      // This converter is used by live local/remote presentation only. The offline worker and
      // benchmark own their independent AI renderer and never instantiate this receiver.
      // A fullscreen game may render at the negotiated resolution while Windows presents it at
      // another desktop resolution. The receiver checks fullscreen ownership independently from
      // the exact authored stereo raster; missing exports retain the packed flat fallback.
      game_export.receiver = ::video::is_game_mode(sbs_mode) && game_export.source_supported ?
                           std::make_unique<platf::reshade_bridge::receiver_t>(device.get(), device_ctx.get()) :
                           nullptr;
      if (game_export.receiver) {
        // An HDR stream is encoded as HDR10 PQ, so the producer may pack an scRGB game in that
        // format directly. Local presentation keeps the producer's own choice.
        game_export.receiver->set_stream_pq(!setup.rgb_only && output_is_hdr);
      }
      if (game_export.receiver && ::video::is_packed_mode(sbs_mode) && external_frame_wake) {
        game_export.receiver->set_frame_wake(external_frame_wake);
      }
      // One encoder converts this display at a time. A token left by a predecessor whose teardown
      // has not run yet must not keep this one from receiving desktop pixels.
      display->external_pixels_owner.store(0, std::memory_order_release);
      game_export.cursor.reset();
      game_export.cursor_logged = false;
      if (game_export.receiver) {
        game_export.cursor = std::make_unique<sbs_cursor::compositor_t>(
          device.get(),
          device_ctx.get(),
          cursor_vs_hlsl.get(),
          cursor_ps_hlsl.get(),
          cursor_ps_normalize_white_hlsl.get(),
          pq_input_to_scrgb_ps_hlsl.get()
        );
        const auto white_nits = display->get_sdr_white_nits();
        game_export.cursor_white_multiplier = white_nits && *white_nits > 0.0f ? *white_nits / 80.0f : 203.0f / 80.0f;
      }
      host_sbs_renderer = models::host_sbs_renderer_e::awaiting_v2;
      v2_live_warp_seen = false;
      v2_live_warp_loss_logged = false;
      diagnostics_enabled = config::sunshine.diagnostics_enabled;
      sbs_depth_status_event = std::move(setup.depth_status_event);
      sbs_depth_pipeline_ready_event = std::move(setup.depth_pipeline_ready_event);
      sbs_telemetry_event = std::move(setup.telemetry_event);
      sbs_telemetry_subscription = std::move(setup.telemetry_subscription);
      sbs_telemetry_performance = diagnostics_enabled ? std::move(setup.telemetry_performance) : nullptr;
      game_export.dumper.set_button_request(::video::is_game_mode(sbs_mode) ? setup.sbs_debug_dump_request : nullptr);
      sbs_dumper.set_button_request(::video::is_game_mode(sbs_mode) ? nullptr : setup.sbs_debug_dump_request);
      dump_button_request = std::move(setup.sbs_debug_dump_request);
      sbs_flat_identity_ps.reset();
      game_export.resample_ps.reset();
      game_export.resample_pq_ps.reset();
      cursor_scissor_raster.Reset();
      sbs_reprojection_v2_live_ps.reset();
      selected_shared_warp={};
      text_game_pin.reset();
      sbs_reprojection_v2_p010_y_ps.reset();
      sbs_debug_v2_mapping_ps.Reset();
      sbs_debug_v2_mask_ps.Reset();
      sbs_debug_mapping_texture.Reset();
      sbs_debug_mapping_rtv.Reset();
      sbs_debug_mapping_srv.Reset();
      sbs_debug_mask_texture.Reset();
      sbs_debug_mask_rtv.Reset();
      sbs_debug_mask_srv.Reset();
      sbs_debug_geometry_init_attempted = false;
      sbs_debug_geometry_ready = false;
      sbs_telemetry_generation = setup.telemetry_generation;
      sbs_telemetry_sequence = 0;
      sbs_telemetry_sample_gate.reset();
      sbs_telemetry_min_frame_id = 0;
      sbs_telemetry_input_region = {};
      sbs_telemetry_input_color_space = models::input_color_space::srgb;
      sbs_telemetry_input_domain_valid = false;
      sbs_telemetry_producer_failure_published = false;
      sbs_telemetry_last_publish_attempt = {};
      pending_matched_frame = nullptr;
      matched_frame_slots = {};
      depth_analysis_generation_tracker = {};
      foreground_window_tracker.reset();
      live_window_authority.reset();
      live_foreground_region.reset();
      live_browser_authority.reset();
      live_browser_authority_epoch = 0u;
      subtitle_context = {};
      subtitle_context_reprocess_pending = false;
      last_subtitle_admission_diagnostic.reset();
      last_subtitle_admission_diagnostic_at = {};
      video_player_registry_ready_observed = false;
      interactive_move_size_observed = false;
      browser_roi_not_before = {};
      last_window_region.reset();
      last_window_region_observer_status.reset();
      last_window_region_mapping_status.reset();
      sbs_frame_sequence = 0;
      drawn_source_order.reset();
      latest_v2_lineage.reset();
      matched_presentation_cache.reset();
      depth_authority_reprocess_pending = false;
      sbs_intermediate_is_linear = false;
      packed_output_in_shutter_pair = false;
      host_sbs_encoder_input_state.reset();
      matched_unknown_frame_error_last = {};
      matched_unknown_frame_errors_suppressed = 0;
      if (diagnostics_enabled) {
        reset_matched_stats();
      }
      // DDup discovers its capture format on the first frame, after these resources are created.
      // Preserve FP16 storage while HDR/10-bit discovery is pending; WGC's already-known BGRA
      // format remains BGRA even on a physical HDR display. The transfer state is deliberately
      // separate and follows each actual img.format in convert().
      HRESULT status = S_OK;

#define create_vertex_shader_helper(x, y) \
  if (FAILED(status = device->CreateVertexShader(x->GetBufferPointer(), x->GetBufferSize(), nullptr, &y))) { \
    BOOST_LOG(error) << "Failed to create vertex shader " << #x << ": " << util::log_hex(status); \
    return -1; \
  }
#define create_pixel_shader_helper(x, y) \
  if (FAILED(status = device->CreatePixelShader(x->GetBufferPointer(), x->GetBufferSize(), nullptr, &y))) { \
    BOOST_LOG(error) << "Failed to create pixel shader " << #x << ": " << util::log_hex(status); \
    return -1; \
  }
#define create_compute_shader_helper(x, y) \
  if (FAILED(status = device->CreateComputeShader(x->GetBufferPointer(), x->GetBufferSize(), nullptr, &y))) { \
    BOOST_LOG(error) << "Failed to create compute shader " << #x << ": " << util::log_hex(status); \
    return -1; \
  }
      const bool sbs_on = ::video::is_packed_mode(sbs_mode);
      const bool ai_on = sbs_mode == ::video::SBS_AI;
      if (sbs_mode == ::video::SBS_GAME_SBS &&
          (!game_export.source_supported ||
           !::video::external_sbs_dimensions_match(setup.game_width, setup.game_height, setup.width, setup.height))) {
        BOOST_LOG(error) << "Game 3D requires an unrotated source and supported exact full-SBS output raster."sv;
        return -1;
      }
#if !defined(SUNSHINE_TESTS)
      if (ai_on && !test_context) {
        // Cross-process accessibility traversal is isolated in the supervised helper. This lease
        // only exposes an atomic snapshot to the matched-frame path and cannot block streaming.
        // A missing, stale, or ambiguous rectangle simply selects ordinary full-frame V2.
        if (!video_dom_lease) {
          video_dom_lease.emplace();
        }
      } else {
        video_dom_lease.reset();
      }
#endif
      const bool downscaling = display->width > setup.width || display->height > setup.height;
      if (sbs_on && display->display_rotation != DXGI_MODE_ROTATION_UNSPECIFIED && display->display_rotation != DXGI_MODE_ROTATION_IDENTITY) {
        BOOST_LOG(error) << "Host SBS requires an identity-oriented source display. "
                         << "Use an explicit portrait resolution instead of Windows display "
                            "rotation.";
        return -1;
      }
      const auto host_sbs_rejection =
        display->width > 0 && display->height > 0 ?
          models::host_sbs_v2_source_resolution_rejection_reason(
            static_cast<std::uint32_t>(display->width),
            static_cast<std::uint32_t>(display->height)
          ) :
          std::string_view {"source extent is empty"};
      // A text game needs the same depth path as Host AI. A surface that path rejects simply
      // makes this Game 3D session unable to convert one; it stays valid for add-on games.
      text_game_eligible = ::video::is_game_mode(sbs_mode) && game_export.source_supported && host_sbs_rejection.empty() && !test_context;
      text_game_ai_active = false;
      text_game_pin.restart_session();
      const bool depth_path_on = ai_on || (sbs_mode == ::video::SBS_GAME_SBS && text_game_eligible);
      if (ai_on && !host_sbs_rejection.empty()) {
        // Launch, RTSP, and live controls have already preflighted the negotiated mode. Reaching
        // this guard therefore means capture attached to a different surface/topology; never let
        // that internal mismatch proceed to inference and become an unexplained flat stream.
        const auto fitted = display->width > 0 && display->height > 0 ?
                              models::fit_host_sbs_v2_depth_tensor_shape(
                                static_cast<std::uint32_t>(display->width),
                                static_cast<std::uint32_t>(display->height)
                              ) :
                              models::depth_tensor_shape_t {};
        BOOST_LOG(error) << "Host SBS V2 rejects capture surface "sv
                         << display->width << 'x' << display->height
                         << ": "sv << host_sbs_rejection
                         << "; fitted depth tensor "sv << fitted.width << 'x'
                         << fitted.height << '.';
        return -1;
      }

      // FP16 capture is linear for both HDR and Advanced Color SDR. Runtime shader selection uses
      // the actual input transfer plus the negotiated display mode; resource format alone is not
      // an HDR-mode signal.
      if (setup.rgb_only) {
        create_pixel_shader_helper(cursor_ps_hlsl, rgb_present_ps);
        create_pixel_shader_helper(
          rgb_present_linear_to_srgb_ps_hlsl,
          rgb_present_linear_to_srgb_ps
        );
        create_pixel_shader_helper(rgb_present_hdr_to_srgb_ps_hlsl, rgb_present_hdr_to_srgb_ps);
        create_pixel_shader_helper(
          rgb_present_srgb_to_linear_ps_hlsl,
          rgb_present_srgb_to_linear_ps
        );
        create_pixel_shader_helper(rgb_present_pq_input_to_linear_ps_hlsl, rgb_present_pq_to_linear_ps);
        create_pixel_shader_helper(rgb_present_pq_input_to_srgb_ps_hlsl, rgb_present_pq_to_srgb_ps);
        // Local HDR presentation can receive BGRA/sRGB frames from WGC even though its swapchain
        // is linear scRGB. Match the existing HDR cursor convention: scRGB 1.0 is 80 nits, while
        // captured SDR white follows the user's display setting (203 nits is Windows' fallback).
        constexpr float fallback_sdr_white_nits = 203.0f;
        float sdr_white_nits = fallback_sdr_white_nits;
        if (display_is_hdr) {
          const auto queried_sdr_white_nits = display->get_sdr_white_nits();
          if (queried_sdr_white_nits && *queried_sdr_white_nits > 0.0f) {
            sdr_white_nits = *queried_sdr_white_nits;
          } else {
            BOOST_LOG(warning)
              << "Failed to query the display's SDR reference white; using "sv
              << fallback_sdr_white_nits
              << " nits for local HDR RGB presentation."sv;
          }
        }
        const std::array<float, 4> white_multiplier {
          sdr_white_nits / 80.0f,
          0.0f,
          0.0f,
          0.0f,
        };
        rgb_present_sdr_white = make_buffer(device.get(), white_multiplier);
        if (!rgb_present_sdr_white) {
          BOOST_LOG(error)
            << "Failed to create local RGB presentation white-level constants."sv;
          return -1;
        }
      }
      if (!sbs_reprojection_vs_hlsl || FAILED(status = device->CreateVertexShader(sbs_reprojection_vs_hlsl->data(), sbs_reprojection_vs_hlsl->size(), nullptr, &sbs_reprojection_vs))) {
        BOOST_LOG(error) << "Failed to create vertex shader sbs_reprojection_vs_hlsl: "
                         << util::log_hex(status);
        return -1;
      }
      if (sbs_on) {
        if (sbs_flat_identity_ps_hlsl) {
          const auto flat_pixel_status = device->CreatePixelShader(
            sbs_flat_identity_ps_hlsl->data(),
            sbs_flat_identity_ps_hlsl->size(),
            nullptr,
            &sbs_flat_identity_ps
          );
          if (FAILED(flat_pixel_status)) {
            BOOST_LOG(warning)
              << "Failed to create the Host SBS flat-identity renderer [0x"sv
              << util::hex(flat_pixel_status).to_string_view() << "]"sv;
            sbs_flat_identity_ps.reset();
          }
        }

        if (!sbs_flat_identity_ps) {
          BOOST_LOG(error) << "Host SBS flat-identity safety renderer is unavailable."sv;
          return -1;
        }
        create_pixel_shader_helper(sbs_packed_resample_ps_hlsl, game_export.resample_ps);
        create_pixel_shader_helper(sbs_packed_resample_pq_input_ps_hlsl, game_export.resample_pq_ps);
        // Re-runs a conversion pass from the Game 3D cursor patch over the cursor's pixels only.
        // Otherwise identical to the default rasterizer state the conversion passes use.
        D3D11_RASTERIZER_DESC scissor_desc {};
        scissor_desc.FillMode = D3D11_FILL_SOLID;
        scissor_desc.CullMode = D3D11_CULL_BACK;
        scissor_desc.DepthClipEnable = TRUE;
        scissor_desc.ScissorEnable = TRUE;
        if (FAILED(status = device->CreateRasterizerState(&scissor_desc, &cursor_scissor_raster))) {
          BOOST_LOG(error) << "Failed to create the Game 3D cursor scissor state: " << util::log_hex(status);
          return -1;
        }
      }
      if (depth_path_on) {
        HRESULT v2_pixel_status = E_FAIL;
        if (sbs_reprojection_v2_live_ps_hlsl) {
          v2_pixel_status = device->CreatePixelShader(
            sbs_reprojection_v2_live_ps_hlsl->data(),
            sbs_reprojection_v2_live_ps_hlsl->size(),
            nullptr,
            &sbs_reprojection_v2_live_ps
          );
        }
        if (!sbs_reprojection_v2_live_ps) {
          if (!sbs_flat_identity_ps) {
            BOOST_LOG(error)
              << "Host SBS cannot create either the authenticated V2 renderer or its "sv
                 "independent flat-identity safety renderer."sv;
            return -1;
          }
          BOOST_LOG(error)
            << "Host SBS V2 renderer is unavailable [0x"sv
            << util::hex(v2_pixel_status).to_string_view()
            << "]; continuing this stream with current-frame flat identity."sv;
          fail_depth_pipeline_flat();
        } else if (!sbs_flat_identity_ps) {
          // V2 can render identity while it awaits valid state, but a later V2 device-creation
          // failure cannot be recovered without the independent safety shader. Reject this Host
          // SBS device rather than pretending its fail-flat guarantee exists.
          BOOST_LOG(error)
            << "Host SBS flat-identity safety renderer is unavailable."sv;
          return -1;
        }
        if (format == DXGI_FORMAT_P010 && sbs_reprojection_v2_p010_y_ps_hlsl) {
          const auto p010_y_status = device->CreatePixelShader(
            sbs_reprojection_v2_p010_y_ps_hlsl->data(),
            sbs_reprojection_v2_p010_y_ps_hlsl->size(),
            nullptr,
            &sbs_reprojection_v2_p010_y_ps
          );
          if (FAILED(p010_y_status)) {
            BOOST_LOG(warning)
              << "Host SBS direct P010 luma optimization is unavailable [0x"sv
              << util::hex(p010_y_status).to_string_view()
              << "]; retaining the established RGB-to-P010 path."sv;
            sbs_reprojection_v2_p010_y_ps.reset();
          }
        }
      }

      if (!setup.rgb_only) {
        switch (format) {
          case DXGI_FORMAT_NV12:
            // Semi-planar 8-bit YUV 4:2:0
            create_vertex_shader_helper(convert_yuv420_planar_y_vs_hlsl, convert_Y_or_YUV_vs);
            create_pixel_shader_helper(convert_yuv420_planar_y_ps_hlsl, convert_Y_or_YUV_ps);
            create_pixel_shader_helper(convert_yuv420_planar_y_ps_linear_hlsl, convert_Y_or_YUV_fp16_ps);
            create_pixel_shader_helper(convert_yuv420_planar_y_ps_pq_input_linear_hlsl, convert_Y_or_YUV_pq_ps);
            if (downscaling && !sbs_on) {
              create_vertex_shader_helper(convert_yuv420_packed_uv_type0s_vs_hlsl, convert_UV_vs);
              create_pixel_shader_helper(convert_yuv420_packed_uv_type0s_ps_hlsl, convert_UV_ps);
              create_pixel_shader_helper(convert_yuv420_packed_uv_type0s_ps_linear_hlsl, convert_UV_fp16_ps);
              create_pixel_shader_helper(convert_yuv420_packed_uv_type0s_ps_pq_input_linear_hlsl, convert_UV_pq_ps);
            } else {
              create_vertex_shader_helper(convert_yuv420_packed_uv_type0_vs_hlsl, convert_UV_vs);
              create_pixel_shader_helper(convert_yuv420_packed_uv_type0_ps_hlsl, convert_UV_ps);
              create_pixel_shader_helper(convert_yuv420_packed_uv_type0_ps_linear_hlsl, convert_UV_fp16_ps);
              create_pixel_shader_helper(convert_yuv420_packed_uv_type0_ps_pq_input_linear_hlsl, convert_UV_pq_ps);
            }
            break;

          case DXGI_FORMAT_P010:
            // Semi-planar 16-bit YUV 4:2:0, 10 most significant bits store the value
            create_vertex_shader_helper(convert_yuv420_planar_y_vs_hlsl, convert_Y_or_YUV_vs);
            create_pixel_shader_helper(convert_yuv420_planar_y_ps_hlsl, convert_Y_or_YUV_ps);
            if (output_is_hdr) {
              create_pixel_shader_helper(convert_yuv420_planar_y_ps_perceptual_quantizer_hlsl, convert_Y_or_YUV_fp16_ps);
              create_pixel_shader_helper(convert_yuv420_planar_y_ps_pq_input_perceptual_quantizer_hlsl, convert_Y_or_YUV_pq_ps);
            } else {
              create_pixel_shader_helper(convert_yuv420_planar_y_ps_linear_hlsl, convert_Y_or_YUV_fp16_ps);
              create_pixel_shader_helper(convert_yuv420_planar_y_ps_pq_input_linear_hlsl, convert_Y_or_YUV_pq_ps);
            }
            if (downscaling && !sbs_on) {
              create_vertex_shader_helper(convert_yuv420_packed_uv_type0s_vs_hlsl, convert_UV_vs);
              create_pixel_shader_helper(convert_yuv420_packed_uv_type0s_ps_hlsl, convert_UV_ps);
              if (output_is_hdr) {
                create_pixel_shader_helper(convert_yuv420_packed_uv_type0s_ps_perceptual_quantizer_hlsl, convert_UV_fp16_ps);
                create_pixel_shader_helper(convert_yuv420_packed_uv_type0s_ps_pq_input_perceptual_quantizer_hlsl, convert_UV_pq_ps);
              } else {
                create_pixel_shader_helper(convert_yuv420_packed_uv_type0s_ps_linear_hlsl, convert_UV_fp16_ps);
                create_pixel_shader_helper(convert_yuv420_packed_uv_type0s_ps_pq_input_linear_hlsl, convert_UV_pq_ps);
              }
            } else {
              create_vertex_shader_helper(convert_yuv420_packed_uv_type0_vs_hlsl, convert_UV_vs);
              create_pixel_shader_helper(convert_yuv420_packed_uv_type0_ps_hlsl, convert_UV_ps);
              if (output_is_hdr) {
                create_pixel_shader_helper(convert_yuv420_packed_uv_type0_ps_perceptual_quantizer_hlsl, convert_UV_fp16_ps);
                create_pixel_shader_helper(convert_yuv420_packed_uv_type0_ps_pq_input_perceptual_quantizer_hlsl, convert_UV_pq_ps);
              } else {
                create_pixel_shader_helper(convert_yuv420_packed_uv_type0_ps_linear_hlsl, convert_UV_fp16_ps);
                create_pixel_shader_helper(convert_yuv420_packed_uv_type0_ps_pq_input_linear_hlsl, convert_UV_pq_ps);
              }
            }
            break;

          default:
            BOOST_LOG(error) << "Unable to create shaders because of the unrecognized surface format";
            return -1;
        }
      }

#undef create_vertex_shader_helper
#undef create_pixel_shader_helper
#undef create_compute_shader_helper

      auto out_width = setup.width;
      auto out_height = setup.height;
      rgb_present_viewport = {0.0f, 0.0f, (float) out_width, (float) out_height, 0.0f, 1.0f};

      // When SBS is on the source content is a double-width side-by-side frame; otherwise it
      // is the plain captured frame. The output (width x height) already carries the doubling
      // (the client-negotiated width was doubled by the encode pipeline for SBS).
      float in_width = sbs_on ? display->width * 2 : display->width;
      float in_height = display->height;

      // Ensure aspect ratio is maintained. SBS owns a full-size packed intermediate and performs
      // the fit independently inside each eye in the warp shaders. Applying one packed viewport
      // would put a pillarbox only at the far left of the left eye and far right of the right eye,
      // which appears as a large false disparity. Plain 2D retains the normal fitted viewport.
      auto scalar = std::fminf(out_width / in_width, out_height / in_height);
      auto fitted_width = in_width * scalar;
      auto fitted_height = in_height * scalar;
      float content_scale_x = 1.0f;
      float content_scale_y = 1.0f;
      if (sbs_on) {
        const float source_aspect = (float) display->width / (float) display->height;
        const float eye_aspect = ((float) out_width * 0.5f) / (float) out_height;
        if (eye_aspect > source_aspect) {
          content_scale_x = source_aspect / eye_aspect;
        } else {
          content_scale_y = eye_aspect / source_aspect;
        }
      }
      sbs_content_scale_x = content_scale_x;
      sbs_content_scale_y = content_scale_y;
      if (!update_sbs_constant_buffer(content_scale_x, content_scale_y)) {
        BOOST_LOG(error) << "Failed to create the Host SBS V2 geometry constants";
        return -1;
      }

      auto out_width_f = sbs_on ? (float) out_width : fitted_width;
      auto out_height_f = sbs_on ? (float) out_height : fitted_height;

      // result is always positive
      auto offsetX = (out_width - out_width_f) / 2;
      auto offsetY = (out_height - out_height_f) / 2;

      // The SBS reprojection intermediate is only needed when host SBS is active. Plain 2D
      // (SBS_OFF) draws the captured frame straight into the output.
      //
      // Size the intermediate to the full encoded output. The warp renders directly at the
      // (possibly capped) encode resolution and applies identical aspect-fit bars inside each eye.
      if (sbs_on) {
        if (depth_path_on) {
          initialize_sbs_gpu_timers();
          BOOST_LOG(info)
            << "Host SBS warp: authenticated Depth Coordinate V2 contractive renderer, fixed pop "
            << sbs_config.pop_strength << ".";
        } else {
          BOOST_LOG(info) << "Host SBS source: ReShade final full-SBS texture, with flat desktop fallback."sv;
        }
        sbs_viewport = {0.0f, 0.0f, out_width_f, out_height_f, 0.0f, 1.0f};
        // RGB sinks can draw directly into the shutter packet. Allocate their scratch
        // texture only when conversion actually needs it. DDup also defers until its
        // first real frame reveals the capture format.
        if (!setup.rgb_only && display->capture_format != DXGI_FORMAT_UNKNOWN &&
            !ensure_sbs_intermediate_storage(display->capture_format == DXGI_FORMAT_R16G16B16A16_FLOAT)) {
          return -1;
        }
      }

      if (setup.rgb_only) {
        // Local AR presents RGB directly. It needs the common capture/SBS resources above, but
        // no encoder output texture, chroma shaders, planar RTVs, or YUV conversion constants.
        if (depth_path_on) {
          observe_capture_damage();
        }
        return 0;
      }

      out_Y_or_YUV_viewport = {offsetX, offsetY, out_width_f, out_height_f, 0.0f, 1.0f};  // Y plane

      out_UV_viewport = {offsetX / 2, offsetY / 2, out_width_f / 2, out_height_f / 2, 0.0f, 1.0f};

      if (sbs_on && (out_width < 4 || out_width % 4 != 0 || out_height < 2 || out_height % 2 != 0)) {
        BOOST_LOG(error) << "Packed Host SBS 4:2:0 requires width divisible by four and even height.";
        return -1;
      }
      // The warp has already fitted/resized each eye into this exact packed raster, and external
      // Game 3D eyes of another size are resampled into it before conversion. The chroma
      // shader must load its real texels and clamp each eye separately, regardless of whether
      // capture-to-encode sizing was classified as downscaling earlier in initialization.
      // Ordinary unrotated 1:1 video (including Client SBS input) needs the same transfer-before-
      // filtering rule, with whole-texture clamping. Actual scaling/rotation retains its generic
      // sampled-source path; its footprint cannot be inferred from output pixel coordinates.
      const bool exact_chroma_texels = sbs_on ||
                                       (out_width == display->width && out_height == display->height &&
                                        (display->display_rotation == DXGI_MODE_ROTATION_UNSPECIFIED ||
                                         display->display_rotation == DXGI_MODE_ROTATION_IDENTITY));
      const ::video::chroma_sampling_t subsample_offset_in {
        1.0f / out_width_f,
        1.0f / out_height_f,
        exact_chroma_texels ? static_cast<unsigned>(out_width) : 0u,
        exact_chroma_texels ? static_cast<unsigned>(out_height) : 0u,
        sbs_on ? static_cast<unsigned>(out_width / 2) : 0u,
      };
      subsample_offset = make_buffer(device.get(), subsample_offset_in);

      if (!subsample_offset) {
        BOOST_LOG(error) << "Failed to create subsample offset vertex constant buffer";
        return -1;
      }
      device_ctx->VSSetConstantBuffers(0, 1, &subsample_offset);

      {
        int32_t rotation_modifier = display->display_rotation == DXGI_MODE_ROTATION_UNSPECIFIED ? 0 : display->display_rotation - 1;
        int32_t rotation_data[16 / sizeof(int32_t)] {-rotation_modifier};  // aligned to 16-byte
        auto rotation = make_buffer(device.get(), rotation_data);
        if (!rotation) {
          BOOST_LOG(error) << "Failed to create display rotation vertex constant buffer";
          return -1;
        }
        device_ctx->VSSetConstantBuffers(1, 1, &rotation);
      }

      DXGI_FORMAT rtv_Y_or_YUV_format = DXGI_FORMAT_UNKNOWN;
      DXGI_FORMAT rtv_UV_format = DXGI_FORMAT_UNKNOWN;

      switch (format) {
        case DXGI_FORMAT_NV12:
          rtv_Y_or_YUV_format = DXGI_FORMAT_R8_UNORM;
          rtv_UV_format = DXGI_FORMAT_R8G8_UNORM;
          break;

        case DXGI_FORMAT_P010:
          rtv_Y_or_YUV_format = DXGI_FORMAT_R16_UNORM;
          rtv_UV_format = DXGI_FORMAT_R16G16_UNORM;
          break;

        default:
          BOOST_LOG(error) << "Unable to create render target views because of the unrecognized surface format";
          return -1;
      }

      auto create_rtv = [&](ID3D11Texture2D *texture, auto &rt, DXGI_FORMAT rt_format) -> bool {
        D3D11_RENDER_TARGET_VIEW_DESC rtv_desc = {};
        rtv_desc.Format = rt_format;
        rtv_desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;

        auto status = device->CreateRenderTargetView(texture, &rtv_desc, &rt);
        if (FAILED(status)) {
          BOOST_LOG(error) << "Failed to create render target view: " << util::log_hex(status);
          return false;
        }

        return true;
      };

      const unsigned surface_count = native_input_owner ? native_input_owner->input_surface_count() : 1u;
      if (surface_count == 0) {
        BOOST_LOG(error) << "Missing native encoder input surfaces."sv;
        return -1;
      }
      output_surface_views.resize(surface_count);
      Microsoft::WRL::ComPtr<ID3D11Device3> chroma_device;
      if (native_input_owner && ai_on && output_is_hdr && p010_chroma_shader) {
        device->QueryInterface(IID_PPV_ARGS(&chroma_device));
      }
      const float y_black[] = {output_black_y, output_black_y, output_black_y, output_black_y};
      const float uv_black[] = {
        output_neutral_uv,
        output_neutral_uv,
        output_neutral_uv,
        output_neutral_uv,
      };
      for (unsigned index = 0; index < surface_count; ++index) {
        auto *texture = native_input_owner ? native_input_owner->get_input_texture(index) : output_texture.get();
        auto &surface = output_surface_views[index];
        if (!texture || !create_rtv(texture, surface.y_or_yuv, rtv_Y_or_YUV_format) ||
            (rtv_UV_format != DXGI_FORMAT_UNKNOWN && !create_rtv(texture, surface.uv, rtv_UV_format))) {
          return -1;
        }
        if (chroma_device) {
          D3D11_TEXTURE2D_DESC desc {};
          texture->GetDesc(&desc);
          if (desc.Format == DXGI_FORMAT_P010 && desc.Width == static_cast<UINT>(out_width) &&
              desc.Height == static_cast<UINT>(out_height) && (desc.BindFlags & D3D11_BIND_UNORDERED_ACCESS)) {
            D3D11_UNORDERED_ACCESS_VIEW_DESC1 view {};
            view.Format = DXGI_FORMAT_R16G16_UNORM;
            view.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
            view.Texture2D.PlaneSlice = 1;
            // Failure leaves this surface's regular chroma RTV available.
            chroma_device->CreateUnorderedAccessView1(texture, &view, &surface.p010_chroma);
          }
        }
        // Every allocation receives black/neutral aspect-ratio padding once.
        device_ctx->ClearRenderTargetView(surface.y_or_yuv.get(), y_black);
        if (surface.uv) {
          device_ctx->ClearRenderTargetView(surface.uv.get(), uv_black);
        }
      }
      if (depth_path_on) {
        observe_capture_damage();
      }
      return 0;
    }

    int init_rgb_output(
      int width,
      int height,
      int sbs_mode_param,
      const config::video_t::sbs_t &settings,
      std::shared_ptr<safe::event_t<bool>> depth_pipeline_ready_event = {}
    ) {
      return init_output({
        .width = width,
        .height = height,
        .sbs_mode_param = sbs_mode_param,
        .settings = settings,
        .depth_pipeline_ready_event = std::move(depth_pipeline_ready_event),
        .rgb_only = true,
        .game_transport_supported = ::video::is_game_mode(sbs_mode_param),
        .game_width = width / 2,
        .game_height = height,
      });
    }

    int init(std::shared_ptr<platf::display_t> display, adapter_t::pointer adapter_p, pix_fmt_e pix_fmt) {
      switch (pix_fmt) {
        case pix_fmt_e::nv12:
          format = DXGI_FORMAT_NV12;
          break;

        case pix_fmt_e::p010:
          format = DXGI_FORMAT_P010;
          break;

        default:
          BOOST_LOG(error) << "D3D11 backend doesn't support pixel format: " << from_pix_fmt(pix_fmt);
          return -1;
      }

      D3D_FEATURE_LEVEL featureLevels[] {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
        D3D_FEATURE_LEVEL_9_3,
        D3D_FEATURE_LEVEL_9_2,
        D3D_FEATURE_LEVEL_9_1
      };

      HRESULT status = D3D11CreateDevice(
        adapter_p,
        D3D_DRIVER_TYPE_UNKNOWN,
        nullptr,
        D3D11_CREATE_DEVICE_FLAGS | D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
        featureLevels,
        sizeof(featureLevels) / sizeof(D3D_FEATURE_LEVEL),
        D3D11_SDK_VERSION,
        &device,
        nullptr,
        &device_ctx
      );

      if (FAILED(status)) {
        BOOST_LOG(error) << "Failed to create encoder D3D11 device [0x"sv << util::hex(status).to_string_view() << ']';
        return -1;
      }

      dxgi::dxgi_t dxgi;
      status = device->QueryInterface(IID_IDXGIDevice, (void **) &dxgi);
      if (FAILED(status)) {
        BOOST_LOG(warning) << "Failed to query DXGI interface from device [0x"sv << util::hex(status).to_string_view() << ']';
        return -1;
      }

      // Honor the capture adapter's process scheduling policy on the encoder
      // device too; absolute priority 30 would force non-yielding hard realtime.
      status = dxgi->SetGPUThreadPriority(7);
      if (FAILED(status)) {
        BOOST_LOG(warning) << "Failed to request relative encoding GPU thread priority. Please run application as administrator for optimal performance.";
      } else {
        BOOST_LOG(info) << "Encoder GPU context uses relative priority 7; process scheduling policy retained.";
      }

      auto default_color_vectors = ::video::color_vectors_from_colorspace({::video::colorspace_e::rec601, false, 8}, true);
      if (!default_color_vectors) {
        BOOST_LOG(error) << "Missing color vectors for Rec. 601"sv;
        return -1;
      }

      color_matrix = make_buffer(device.get(), *default_color_vectors);
      if (!color_matrix) {
        BOOST_LOG(error) << "Failed to create color matrix buffer"sv;
        return -1;
      }
      device_ctx->VSSetConstantBuffers(3, 1, &color_matrix);
      device_ctx->PSSetConstantBuffers(0, 1, &color_matrix);

      capture_damage_observer.reset();
      this->display = std::dynamic_pointer_cast<display_base_t>(display);
      if (!this->display) {
        return -1;
      }
      display = nullptr;
      // Windows HDR/color-mode changes rebuild this display/capture object. Cache its immutable
      // session value instead of issuing QueryInterface + GetDesc1 for every FP16 SBS frame.
      display_is_hdr = this->display->is_hdr();

      blend_disable = make_blend(device.get(), false, false);
      if (!blend_disable) {
        return -1;
      }

      D3D11_SAMPLER_DESC sampler_desc {};
      sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
      sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
      sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
      sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
      sampler_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
      sampler_desc.MinLOD = 0;
      sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;

      status = device->CreateSamplerState(&sampler_desc, &sampler_linear);
      if (FAILED(status)) {
        BOOST_LOG(error) << "Failed to create point sampler state [0x"sv << util::hex(status).to_string_view() << ']';
        return -1;
      }

      device_ctx->OMSetBlendState(blend_disable.get(), nullptr, 0xFFFFFFFFu);
      device_ctx->PSSetSamplers(0, 1, &sampler_linear);
      device_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

      // The depth estimator (heavy: model download + TensorRT engine build/load) is created
      // lazily on the first SBS frame via ensure_depth_estimator(), so plain 2D (SBS_OFF)
      // sessions never pay for it.

      // Rebuilt by init_output() once the source/output aspect relationship is known.
      sbs_content_scale_x = 1.0f;
      sbs_content_scale_y = 1.0f;
      if (!update_sbs_constant_buffer(1.0f, 1.0f)) {
        return -1;
      }

      return 0;
    }

    struct encoder_img_ctx_t {
      std::uint64_t capture_texture_generation = 0;

      described_texture_t encoder_texture;
      shader_res_t encoder_input_res;
      keyed_mutex_t encoder_mutex;

      std::weak_ptr<const platf::img_t> img_weak;

      void reset() {
        capture_texture_generation = 0;
        encoder_texture.reset();
        encoder_input_res.reset();
        encoder_mutex.reset();
        img_weak.reset();
      }
    };

    int initialize_image_context(const img_d3d_t &img, encoder_img_ctx_t &img_ctx) {
      // If we've already opened the shared texture, we're done
      if (img_ctx.encoder_texture && img.capture_texture_generation == img_ctx.capture_texture_generation) {
        return 0;
      }

      // Reset this image context in case it was used before with a different texture.
      // Textures can change when transitioning from a dummy image to a real image.
      img_ctx.reset();

      device1_t device1;
      auto status = device->QueryInterface(__uuidof(ID3D11Device1), (void **) &device1);
      if (FAILED(status)) {
        BOOST_LOG(error) << "Failed to query ID3D11Device1 [0x"sv << util::hex(status).to_string_view() << ']';
        return -1;
      }

      // Open a handle to the shared texture
      texture2d_t texture;
      status = device1->OpenSharedResource1(img.encoder_texture_handle, __uuidof(ID3D11Texture2D), (void **) &texture);
      if (FAILED(status)) {
        BOOST_LOG(error) << "Failed to open shared image texture [0x"sv << util::hex(status).to_string_view() << ']';
        return -1;
      }

      // Get the keyed mutex to synchronize with the capture code
      keyed_mutex_t mutex;
      status = texture->QueryInterface(__uuidof(IDXGIKeyedMutex), (void **) &mutex);
      if (FAILED(status)) {
        BOOST_LOG(error) << "Failed to query IDXGIKeyedMutex [0x"sv << util::hex(status).to_string_view() << ']';
        return -1;
      }

      // Create the SRV for the encoder texture
      shader_res_t srv;
      status = device->CreateShaderResourceView(texture.get(), nullptr, &srv);
      if (FAILED(status)) {
        BOOST_LOG(error) << "Failed to create shader resource view for encoding [0x"sv << util::hex(status).to_string_view() << ']';
        return -1;
      }

      img_ctx.encoder_texture.reset(std::move(texture));
      img_ctx.encoder_mutex = std::move(mutex);
      img_ctx.encoder_input_res = std::move(srv);
      img_ctx.capture_texture_generation = img.capture_texture_generation;
      img_ctx.img_weak = img.weak_from_this();

      return 0;
    }

    buf_t subsample_offset;
    buf_t color_matrix;
    buf_t sdr_color_transform;
    buf_t external_color_transform;
    buf_t stream_gamma_transform;
    sdr_color_transform_t source_color_transform {};
    float default_source_sdr_white_scrgb = 203.0f / 80.0f;
    ::video::stream_gamma_mode_e stream_gamma_mode_ = ::video::stream_gamma_mode_e::windows_default;
    float stream_gamma_white_nits_ = 203.0f;
    bool stream_gamma_conversion_pending = false;
    std::chrono::steady_clock::time_point stream_gamma_white_query_at {};
    float output_black_y = 0.0f;
    float output_neutral_uv = 0.5f;

    blend_t blend_disable;
    sampler_state_t sampler_linear;

    struct output_surface_views_t {
      render_target_t y_or_yuv;
      render_target_t uv;
      Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView1> p010_chroma;
      bool p010_y_mrt_targets_checked = false;
      bool p010_y_mrt_targets_qualified = false;
    };
    std::vector<output_surface_views_t> output_surface_views;
    unsigned active_output_surface = 0;
    nvenc::nvenc_d3d11_native *native_input_owner = nullptr;  ///< Owned by the native wrapper.

    output_surface_views_t &active_output_views() {
      return output_surface_views[active_output_surface];
    }

    // d3d_img_t::id -> encoder_img_ctx_t
    // These store the encoder textures for each img_t that passes through
    // convert(). We can't store them in the img_t itself because it can be shared
    // among multiple encode devices (and therefore multiple ID3D11Devices).
    std::map<uint32_t, encoder_img_ctx_t> img_ctx_map;

    std::shared_ptr<display_base_t> display;
    util::safe_ptr<display_ddup_vram_t, [](display_ddup_vram_t *capture) {
      capture->remove_damage_observer();
    }> capture_damage_observer;
    bool display_is_hdr = false;
    bool output_is_hdr = false;

    vs_t convert_Y_or_YUV_vs;
    ps_t convert_Y_or_YUV_ps;
    ps_t convert_Y_or_YUV_fp16_ps;
    ps_t convert_Y_or_YUV_pq_ps;  ///< Game 3D PQ export; passthrough for an HDR stream.

    vs_t convert_UV_vs;
    ps_t convert_UV_ps;
    ps_t convert_UV_fp16_ps;
    ps_t convert_UV_pq_ps;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> p010_chroma_shader;

    D3D11_VIEWPORT out_Y_or_YUV_viewport;
    D3D11_VIEWPORT out_UV_viewport;

    DXGI_FORMAT format;

    device_t device;
    device_ctx_t device_ctx;

    texture2d_t output_texture;
    bool rgb_copy_path_logged = false;
    bool rgb_copy_fallback_logged = false;
    ps_t rgb_present_ps;
    ps_t rgb_present_linear_to_srgb_ps;
    ps_t rgb_present_hdr_to_srgb_ps;
    ps_t rgb_present_srgb_to_linear_ps;
    ps_t rgb_present_pq_to_linear_ps;
    ps_t rgb_present_pq_to_srgb_ps;
    buf_t rgb_present_sdr_white;
    D3D11_VIEWPORT rgb_present_viewport {};

    std::unique_ptr<models::video_depth_estimator> depth_estimator;
    // Estimators that failed while this device keeps drawing flat. Their teardown unregisters
    // CUDA's D3D11 interop, which must not run beside this drawing thread, so they retire with
    // the device, once nothing draws with it.
    std::vector<std::unique_ptr<models::video_depth_estimator>> failed_depth_estimators;
    // The per-device D3D estimator is built on a background thread and borrows the startup-warmed
    // TensorRT context. The process-lifetime tracked worker service owns and joins the worker; the
    // task itself captures owning D3D references, so retiring this future during teardown is safe.
    std::future<depth_pipeline_build_result_t> depth_estimator_build;
    unsigned engine_poll_counter = 0;  ///< Rate-limits the startup model-preparation wait warning.
    int sbs_mode = ::video::SBS_OFF;  ///< Host SBS mode for this encode device (set in init_output).
    config::video_t::sbs_t sbs_config {};  ///< Immutable Host SBS settings for this device.
    game_export_pipeline_t game_export;  ///< The Game 3D add-on export, its cursor and its source state.
    std::function<void()> external_frame_wake;  ///< See set_external_frame_wake().
    std::uint64_t capture_pixels_token = 0;  ///< This encoder's display->external_pixels_owner value.
    bool conversion_missing_source = false;  ///< Skipped capture pixels have no external replacement.
    // A Game 3D text game has no depth and publishes no stereo of its own. It supplies where its
    // text is, through the text helper loaded into it, and the host supplies the rest: while such
    // a game is in front, packed Game 3D converts the captured picture with the Host AI path.
    bool text_game_eligible = false;  ///< This session can convert a text game; decided at init.
    bool text_game_ai_active = false;  ///< The current frame takes the Host AI path for one.
    bool local_presentation = false;  ///< Output goes to a local display, not to an encoder.
    std::shared_ptr<std::atomic<bool>> dump_button_request;
    /// The latest conversion that looked at which window is in front; see needs_conversion_poll().
    std::chrono::steady_clock::time_point text_game_serviced_at {};
    text_game_pin_controller_t text_game_pin;  ///< The game in front, its helper's text and its pin.
    bool diagnostics_enabled = false;  ///< Cached once per device; the disabled hot path only branches.
    safe::mail_raw_t::event_t<int> sbs_depth_status_event;
    std::shared_ptr<safe::event_t<bool>> sbs_depth_pipeline_ready_event;
    safe::mail_raw_t::event_t<::video::sbs_telemetry_snapshot_t> sbs_telemetry_event;
    std::shared_ptr<::video::sbs_telemetry_subscription_t> sbs_telemetry_subscription;
    std::shared_ptr<host_sbs_telemetry::collector> sbs_telemetry_performance;
    std::uint32_t sbs_telemetry_generation = 0;
    std::uint32_t sbs_telemetry_sequence = 0;
    detail::host_sbs_telemetry_sample_gate_t sbs_telemetry_sample_gate;
    std::uint64_t sbs_telemetry_min_frame_id = 0;
    models::depth_input_region_t sbs_telemetry_input_region {};
    models::input_color_space sbs_telemetry_input_color_space =
      models::input_color_space::srgb;
    bool sbs_telemetry_input_domain_valid = false;
    bool sbs_telemetry_producer_failure_published = false;
    std::chrono::steady_clock::time_point sbs_telemetry_last_publish_attempt {};
    vs_t sbs_reprojection_vs;
    ps_t sbs_flat_identity_ps;
    Microsoft::WRL::ComPtr<ID3D11RasterizerState> cursor_scissor_raster;
    ps_t sbs_reprojection_v2_live_ps;
    ps_t sbs_reprojection_v2_p010_y_ps;
    models::host_sbs_renderer_e host_sbs_renderer =
      models::host_sbs_renderer_e::awaiting_v2;
    bool v2_live_warp_seen = false;
    bool v2_live_warp_loss_logged = false;
    buf_t sbs_reprojection_cbuffer;
    buf_t sbs_subtitle_render_cbuffer;
    bool selected_matte_ready = false;
    bool flat_fallback_notified = false;
    models::host_sbs_shared_ui_warp::programs_t selected_shared_warp_programs;
    models::host_sbs_shared_ui_warp::resources_t selected_shared_warp;
    std::filesystem::path selected_shared_warp_root;
    detail::host_sbs_drawn_source_order_t drawn_source_order;
    detail::uploaded_value_state_t<models::host_sbs_subtitle_render_constants_t>
      sbs_subtitle_render_constants;
    float sbs_content_scale_x = 1.0f;
    float sbs_content_scale_y = 1.0f;
    texture2d_t sbs_intermediate_texture;
    render_target_t sbs_intermediate_rtv;
    shader_res_t sbs_intermediate_srv;
    bool sbs_intermediate_is_linear = false;
    bool packed_output_in_shutter_pair = false;
    detail::host_sbs_encoder_input_state_t host_sbs_encoder_input_state;
    detail::uploaded_value_state_t<models::host_sbs_v2_geometry_t>
      sbs_reprojection_constants;
    // Lazily created only for an explicit Dump 3D. These repeat the exact production inverse map
    // after its timing window without allocating the removed V1 forward-coverage resources.
    Microsoft::WRL::ComPtr<ID3D11PixelShader> sbs_debug_v2_mapping_ps;
    Microsoft::WRL::ComPtr<ID3D11PixelShader> sbs_debug_v2_mask_ps;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> sbs_debug_mapping_texture;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> sbs_debug_mapping_rtv;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> sbs_debug_mapping_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> sbs_debug_mask_texture;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> sbs_debug_mask_rtv;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> sbs_debug_mask_srv;
    bool sbs_debug_geometry_init_attempted = false;
    bool sbs_debug_geometry_ready = false;
    D3D11_VIEWPORT sbs_viewport;
    std::array<matched_frame_slot_t, 2> matched_frame_slots;
    // The estimator's sole pending root pins this source and keeps conversion polling active.
    matched_frame_slot_t *pending_matched_frame = nullptr;
    std::optional<video_dom::lease_t> video_dom_lease;
#ifdef SUNSHINE_TESTS
    std::shared_future<std::vector<std::wstring>> test_registry_executables;
    std::uint64_t test_registry_factory_calls = 0;
#endif
    std::unique_ptr<video_player::registry_t> video_player_registry;
    bool video_player_registry_ready_observed = false;
    /// See subtitle_process_image_path().
    struct {
      std::uintptr_t window = 0;
      std::uint32_t process_id = 0;
      std::chrono::steady_clock::time_point queried_at {};
      std::optional<std::wstring> path;
    } subtitle_process_image;
    models::host_subtitle_context::tracker_t subtitle_context;
    std::optional<subtitle_admission_diagnostic_key_t> last_subtitle_admission_diagnostic;
    std::chrono::steady_clock::time_point last_subtitle_admission_diagnostic_at {};
    const converter_test_observer::context_t *test_context = nullptr;
    converter_test_observer::observer_t test_observer {};
#ifdef SUNSHINE_TESTS
    testing::native_pacing_fixture_t test_native_pacing;
    bool test_native_pacing_gate_configured = false;
#endif
    std::int64_t test_current_source_ticks = 0, test_current_content_ticks = 0;
    bool subtitle_context_reprocess_pending = false;
    models::depth_analysis_generation_tracker_t depth_analysis_generation_tracker;
    foreground_window::continuity_tracker_t foreground_window_tracker;
    std::optional<foreground_window::snapshot_t> live_window_authority;
    std::optional<foreground_window::snapshot_t> live_foreground_region;
    std::optional<browser_live_authority_key_t> live_browser_authority;
    std::uint64_t live_browser_authority_epoch = 0u;
    bool interactive_move_size_observed = false;
    std::chrono::steady_clock::time_point browser_roi_not_before {};
    std::optional<sbs_debug::window_region_snapshot> last_window_region;
    std::optional<std::string_view> last_window_region_observer_status;
    std::optional<std::string_view> last_window_region_mapping_status;
    std::uint64_t sbs_frame_sequence = 0;
    latest_v2_lineage_t latest_v2_lineage;
    matched_presentation_cache_t matched_presentation_cache;
    bool depth_authority_reprocess_pending = false;
    std::chrono::steady_clock::time_point matched_stats_started {};
    unsigned matched_stats_calls = 0;
    unsigned matched_stats_completions = 0;
    unsigned matched_stats_repeats = 0;
    unsigned matched_stats_content_skips = 0;
    unsigned matched_stats_content_reuses = 0;
    unsigned matched_stats_damage_skips = 0;
    unsigned matched_stats_damage_reuses = 0;
    unsigned matched_stats_video_roi_route_outputs = 0;
    unsigned matched_stats_roi_direct_inputs = 0;
    unsigned matched_stats_roi_dump_copies = 0;
    unsigned matched_stats_output_warped_new = 0;
    unsigned matched_stats_output_packed_repeat = 0;
    unsigned matched_stats_output_flat = 0;
    unsigned matched_stats_output_p010_y_mrt = 0;
    unsigned matched_stats_output_flat_pending_miss = 0;
    unsigned matched_stats_same_frame_poll_attempts = 0;
    unsigned matched_stats_same_frame_poll_hits = 0;
    unsigned matched_stats_same_frame_poll_timeouts = 0;
    unsigned matched_stats_same_frame_poll_submissions = 0;
    unsigned matched_stats_same_frame_poll_cadence_ineligible_busy = 0;
    unsigned matched_stats_same_frame_poll_wait_unavailable_busy = 0;
    unsigned matched_stats_publication_late_ready = 0;
    unsigned matched_stats_force_missing_publication = 0;
    unsigned matched_stats_publication_failure_polls = 0;
    unsigned matched_stats_same_frame_poll_plans = 0;
    unsigned matched_stats_same_frame_poll_hard_cap_timeouts = 0;
    unsigned matched_stats_same_frame_poll_cadence_timeouts = 0;
    unsigned matched_stats_same_frame_poll_failures = 0;
    unsigned matched_stats_same_frame_immediate_hits = 0;
    unsigned matched_stats_same_frame_poll_hits_le_2_ms = 0;
    unsigned matched_stats_same_frame_poll_hits_2_to_2_5_ms = 0;
    unsigned matched_stats_same_frame_poll_hits_2_5_to_3_ms = 0;
    unsigned matched_stats_same_frame_poll_hits_over_3_ms = 0;
    std::uint64_t matched_stats_same_frame_poll_queries = 0;
    double matched_stats_same_frame_poll_wait_sum_ms = 0.0;
    double matched_stats_same_frame_poll_wait_max_ms = 0.0;
    double matched_stats_same_frame_poll_budget_sum_ms = 0.0;
    double matched_stats_same_frame_poll_budget_max_ms = 0.0;
    double matched_stats_age_sum_ms = 0.0;
    double matched_stats_age_max_ms = 0.0;
    std::chrono::steady_clock::time_point matched_unknown_frame_error_last {};
    unsigned matched_unknown_frame_errors_suppressed = 0;
    static constexpr std::size_t sbs_gpu_timer_ring_size = 16;
    std::array<sbs_gpu_timer_slot_t, sbs_gpu_timer_ring_size> sbs_gpu_timers;
    std::size_t sbs_gpu_timer_next = 0;
    bool sbs_gpu_timing_ready = false;

    platf::sbs_debug::dumper sbs_dumper;  ///< Debug: dumps SBS frames on the client button (see sbs_debug_dump.h).
    std::optional<std::chrono::steady_clock::time_point> rendered_content_timestamp_;
  };

  int init();

#ifdef SUNSHINE_TESTS
  namespace testing {
    namespace {
      using Microsoft::WRL::ComPtr;

      void preparation_fixture_require(bool condition, const char *message) {
        if (!condition) {
          throw std::runtime_error(message);
        }
      }

      void preparation_fixture_hr(HRESULT status, const char *message) {
        if (FAILED(status)) {
          throw std::runtime_error(std::string(message) + ": " + std::to_string(static_cast<long>(status)));
        }
      }

      ComPtr<IDXGIAdapter1> preparation_fixture_adapter(const model_preparation_adapter_t &required) {
        preparation_fixture_require((required.luid_low || required.luid_high) && required.vendor_id && required.device_id,
          "Preparation probe requires explicit adapter identity");
        ComPtr<IDXGIFactory1> factory;
        preparation_fixture_hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "Preparation DXGI factory failed");
        ComPtr<IDXGIAdapter1> selected;
        unsigned matches = 0;
        for (UINT index = 0; index != 64u; ++index) {
          ComPtr<IDXGIAdapter1> candidate;
          const auto status = factory->EnumAdapters1(index, &candidate);
          if (status == DXGI_ERROR_NOT_FOUND) {
            preparation_fixture_require(matches == 1u, "Preparation adapter identity missing or ambiguous");
            return selected;
          }
          preparation_fixture_hr(status, "Preparation adapter enumeration failed");
          DXGI_ADAPTER_DESC1 description {};
          preparation_fixture_hr(candidate->GetDesc1(&description), "Preparation adapter description failed");
          if (description.AdapterLuid.LowPart != required.luid_low || description.AdapterLuid.HighPart != required.luid_high) {
            continue;
          }
          preparation_fixture_require(description.VendorId == required.vendor_id && description.DeviceId == required.device_id &&
              !(description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE),
            "Preparation adapter hardware identity differs");
          selected = candidate;
          ++matches;
        }
        throw std::runtime_error("Preparation adapter census exceeded its finite bound");
      }

      void preparation_fixture_device_adapter(ID3D11Device *device, const model_preparation_adapter_t &required) {
        ComPtr<IDXGIDevice> dxgi;
        ComPtr<IDXGIAdapter> adapter;
        DXGI_ADAPTER_DESC description {};
        preparation_fixture_hr(device->QueryInterface(IID_PPV_ARGS(&dxgi)), "Preparation device DXGI query failed");
        preparation_fixture_hr(dxgi->GetAdapter(&adapter), "Preparation device adapter query failed");
        preparation_fixture_hr(adapter->GetDesc(&description), "Preparation actual adapter description failed");
        preparation_fixture_require(description.AdapterLuid.LowPart == required.luid_low && description.AdapterLuid.HighPart == required.luid_high &&
            description.VendorId == required.vendor_id && description.DeviceId == required.device_id,
          "Preparation actual D3D device differs from explicit adapter");
      }

      struct preparation_fixture_display_t final: display_vram_t {
        native_pacing_fixture_t native_pacing;

        bool is_hdr() override { return false; }
        capture_e snapshot(const pull_free_image_cb_t &, std::shared_ptr<platf::img_t> &, std::chrono::milliseconds, bool) override {
          throw std::logic_error("Preparation probe must never enter physical capture");
        }
        capture_e release_snapshot() override {
          throw std::logic_error("Preparation probe owns no physical capture");
        }
      };
    }  // namespace

    std::shared_ptr<display_vram_t> make_native_pacing_display(
      const model_preparation_adapter_t &required, unsigned width, unsigned height,
      const native_pacing_fixture_t &probe
    ) {
      const auto adapter = preparation_fixture_adapter(required);
      preparation_fixture_require(platf::dxgi::init() == 0, "Native pacing shader initialization failed");
      auto display = platf::make_unarmed_capture_display<preparation_fixture_display_t>();
      display->native_pacing = probe;
      adapter->AddRef();
      display->adapter.reset(adapter.Get());
      display->width = display->width_before_rotation = width;
      display->height = display->height_before_rotation = height;
      display->env_width = width;
      display->env_height = height;
      display->offset_x = display->offset_y = 0;
      display->display_rotation = DXGI_MODE_ROTATION_IDENTITY;
      display->capture_format = DXGI_FORMAT_B8G8R8A8_UNORM;
      display->next_image_id = 1u;
      constexpr D3D_FEATURE_LEVEL levels[] {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
      preparation_fixture_hr(
        D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
          levels, 2u, D3D11_SDK_VERSION, &display->device, &display->feature_level, &display->device_ctx),
        "Native pacing capture device creation failed"
      );
      preparation_fixture_device_adapter(display->device.get(), required);
      return display;
    }

    std::shared_ptr<platf::img_t> upload_native_pacing_image(
      display_vram_t &display, std::span<const std::uint32_t> bgra
    ) {
      preparation_fixture_require(bgra.size() == std::size_t(display.width) * display.height,
        "Native pacing source shape differs from its capture device");
      auto image = display.alloc_img();
      preparation_fixture_require(image && display.complete_img(image.get(), false) == 0,
        "Native pacing shared image allocation failed");
      auto &native = static_cast<img_d3d_t &>(*image);
      detail::keyed_mutex_lock_t lock {native.capture_mutex.get()};
      preparation_fixture_require(lock.lock() == S_OK, "Native pacing capture image keyed mutex failed");
      display.device_ctx->UpdateSubresource(native.capture_texture.get(), 0u, nullptr,
        bgra.data(), native.row_pitch, 0u);
      display.device_ctx->Flush();
      native.blank = false;
      return image;
    }

    class native_pacing_local_converter_t::impl_t {
    public:
      impl_t(std::shared_ptr<display_vram_t> display, ::video::sbs_mode_e mode) {
        const auto *fixture = dynamic_cast<const preparation_fixture_display_t *>(display.get());
        preparation_fixture_require(fixture != nullptr, "Local pacing converter requires a native pacing display");
        const auto width = mode == ::video::SBS_AI ? 2 * display->width : display->width;
        const auto height = display->height;
        config::video_t::sbs_t settings;
        settings.reshade = false;
        // The local presenter's own production RGB converter initialization.
        preparation_fixture_require(
          converter_.init(display, display->adapter.get(), pix_fmt_e::nv12) == 0 &&
            converter_.init_rgb_output(width, height, mode, settings, ready_event_) == 0,
          "Local pacing RGB converter initialization failed"
        );
        converter_.test_native_pacing = fixture->native_pacing;
        converter_.test_observer = fixture->native_pacing.converter_observer;
        D3D11_TEXTURE2D_DESC description {};
        description.Width = static_cast<UINT>(width);
        description.Height = static_cast<UINT>(height);
        description.MipLevels = description.ArraySize = description.SampleDesc.Count = 1u;
        description.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        description.Usage = D3D11_USAGE_DEFAULT;
        description.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        preparation_fixture_hr(converter_.device->CreateTexture2D(&description, nullptr, &target_), "Local pacing output allocation failed");
        preparation_fixture_hr(converter_.device->CreateRenderTargetView(target_.Get(), nullptr, &target_rtv_), "Local pacing output RTV failed");
      }

      bool convert(platf::img_t &image, const std::chrono::steady_clock::time_point presentation_target) {
        return converter_.convert_rgb(image, target_.Get(), target_rtv_.Get(), false, presentation_target) ==
               d3d_base_encode_device::rgb_conversion_e::converted;
      }

      void fail_depth_pipeline_flat() {
        converter_.fail_depth_pipeline_flat();
      }

      bool estimator_present() const {
        return static_cast<bool>(converter_.depth_estimator);
      }

      std::size_t parked_failed_estimators() const {
        return converter_.failed_depth_estimators.size();
      }

      ID3D11Device *device() const {
        return const_cast<ID3D11Device *>(converter_.device.get());
      }

    private:
      std::shared_ptr<safe::event_t<bool>> ready_event_ = std::make_shared<safe::event_t<bool>>();
      d3d_base_encode_device converter_;
      ComPtr<ID3D11Texture2D> target_;
      ComPtr<ID3D11RenderTargetView> target_rtv_;
    };

    native_pacing_local_converter_t::native_pacing_local_converter_t(
      std::shared_ptr<display_vram_t> display,
      ::video::sbs_mode_e mode
    ):
        impl_(std::make_unique<impl_t>(std::move(display), mode)) {
    }

    native_pacing_local_converter_t::~native_pacing_local_converter_t() = default;

    bool native_pacing_local_converter_t::convert(
      platf::img_t &image,
      const std::chrono::steady_clock::time_point presentation_target
    ) {
      return impl_->convert(image, presentation_target);
    }

    void native_pacing_local_converter_t::fail_depth_pipeline_flat() {
      impl_->fail_depth_pipeline_flat();
    }

    bool native_pacing_local_converter_t::estimator_present() const {
      return impl_->estimator_present();
    }

    std::size_t native_pacing_local_converter_t::parked_failed_estimators() const {
      return impl_->parked_failed_estimators();
    }

    ID3D11Device *native_pacing_local_converter_t::device() const {
      return impl_->device();
    }

    model_preparation_memory_t query_model_preparation_memory(const model_preparation_adapter_t &required) {
      const auto adapter = preparation_fixture_adapter(required);
      DXGI_ADAPTER_DESC1 description {};
      preparation_fixture_hr(adapter->GetDesc1(&description), "Preparation memory adapter description failed");
      model_preparation_memory_t result;
      result.adapter = required;
      result.description = platf::to_utf8(description.Description);
      ComPtr<IDXGIAdapter3> adapter3;
      preparation_fixture_hr(adapter.As(&adapter3), "Preparation process memory interface unavailable");
      result.local_status = adapter3->QueryVideoMemoryInfo(0u, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &result.local);
      result.nonlocal_status = adapter3->QueryVideoMemoryInfo(0u, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &result.nonlocal);
      return result;
    }

    class model_preparation_converter_t::impl_t {
    public:
      explicit impl_t(model_preparation_converter_request_t request):
          request_(std::move(request)),
          owner_thread_(std::this_thread::get_id()),
          ready_event_(std::make_shared<safe::event_t<bool>>()) {
        const auto width = request_.width, height = request_.height;
        preparation_fixture_require(width && height && width <= 4096u && height <= 4096u &&
            request_.bgra.size() == std::size_t(width) * height * 4u,
          "Preparation probe source shape or immutable bytes invalid");
        const auto attributes = GetFileAttributesW(request_.output_root.c_str());
        preparation_fixture_require(request_.output_root.is_absolute() && attributes != INVALID_FILE_ATTRIBUTES &&
            (attributes & FILE_ATTRIBUTE_DIRECTORY) && !(attributes & FILE_ATTRIBUTE_REPARSE_POINT),
          "Preparation probe requires an owned absolute ordinary output directory");
        source_artifact_ = write_artifact("source.bgra8", request_.bgra.data(), request_.bgra.size(), width, height);

        const auto adapter = preparation_fixture_adapter(request_.adapter);
        preparation_fixture_require(::platf::dxgi::init() == 0, "Preparation platform shader initialization failed");
        display_ = std::make_shared<preparation_fixture_display_t>();
        display_->width = display_->width_before_rotation = width;
        display_->height = display_->height_before_rotation = height;
        display_->env_width = width;
        display_->env_height = height;
        display_->offset_x = display_->offset_y = 0;
        display_->display_rotation = DXGI_MODE_ROTATION_IDENTITY;
        display_->capture_format = DXGI_FORMAT_B8G8R8A8_UNORM;
        display_->next_image_id = 1u;
        constexpr D3D_FEATURE_LEVEL levels[] {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
        preparation_fixture_hr(D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
          levels, 2u, D3D11_SDK_VERSION, &display_->device, nullptr, &display_->device_ctx), "Preparation source device creation failed");
        preparation_fixture_device_adapter(display_->device.get(), request_.adapter);
        converter_ = std::make_unique<d3d_base_encode_device>();
        preparation_fixture_require(converter_->init(display_, adapter.Get(), pix_fmt_e::nv12) == 0, "Preparation actual converter initialization failed");
        preparation_fixture_device_adapter(converter_->device.get(), request_.adapter);
        test_context_ = {width, height, 0x5052455041524531ull};
        if (request_.hard_disabled_dirty_source) {
          preparation_fixture_require(!request_.production_observer_path, "Dirty source probe requires the synthetic context");
          // The existing zero-instance branch disables subtitles without native authority.
          test_context_.instance = 0u;
          test_context_.borrow_matched_capture = true;
          damage_history_ = std::make_shared<detail::ddup_damage_history_t>();
        }
        if (request_.production_observer_path) {
          preparation_fixture_require(request_.registry_executables.valid(), "Production-observer probe requires its finite registry future");
          converter_->test_registry_executables = request_.registry_executables;
        } else {
          converter_->test_context = &test_context_;
        }
        converter_->test_observer = {this, &impl_t::record};

        image_ = display_->alloc_img();
        preparation_fixture_require(image_ && display_->complete_img(image_.get(), false) == 0, "Preparation shared source image allocation failed");
        auto &image = static_cast<img_d3d_t &>(*image_);
        preparation_fixture_require(image.capture_mutex.get() != nullptr, "Preparation source keyed mutex unavailable");
        preparation_fixture_require(image.capture_mutex->AcquireSync(0u, 5000u) == S_OK, "Preparation immutable source mutex failed");
        {
          auto release = util::fail_guard([&image]() { image.capture_mutex->ReleaseSync(0u); });
          display_->device_ctx->UpdateSubresource(image.capture_texture.get(), 0u, nullptr, request_.bgra.data(), width * 4u, 0u);
          display_->device_ctx->Flush();
        }
        ++source_uploads_;
        if (damage_history_) {
          image.ddup_damage = damage_history_->commit({true, {{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)}}});
        }
        image.blank = false;
        const auto stamp = std::chrono::steady_clock::now();
        image_->frame_timestamp = stamp;
        image_->content_timestamp = stamp;
        set_mode(::video::SBS_OFF);
      }

      ~impl_t() {
        // The real destructor transfers GPU construction/teardown to its existing tracked owner.
        // Destroy it while observer state, the source and the synthetic context still exist.
        converter_.reset();
      }

      void set_mode(::video::sbs_mode_e mode) {
        assert_owner();
        preparation_fixture_require(mode == ::video::SBS_OFF || mode == ::video::SBS_GAME_MONO || mode == ::video::SBS_AI,
          "Preparation probe supports flat, no-provider Game mono and actual Host AI only");
        const auto output_width = mode == ::video::SBS_AI ? request_.width * 2u : request_.width;
        // Only synchronous initialization sees this finite external factory edge. The real
        // Game branch still runs; unsupported transport creates no ReShade receiver/provider.
        config::video_t::sbs_t settings;
        settings.reshade = false;
        {
          auto *previous = std::exchange(active_model_preparation_watch_probe, &watch_probe_);
          auto restore_watch = util::fail_guard([previous]() { active_model_preparation_watch_probe = previous; });
          preparation_fixture_require(converter_->init_rgb_output(static_cast<int>(output_width), static_cast<int>(request_.height),
            mode, settings, ready_event_) == 0, "Preparation actual RGB mode initialization failed");
        }
        preparation_fixture_require(!converter_->game_export.receiver, "Preparation probe unexpectedly constructed a provider receiver");
        target_.Reset();
        target_rtv_.Reset();
        D3D11_TEXTURE2D_DESC description {};
        description.Width = output_width;
        description.Height = request_.height;
        description.MipLevels = description.ArraySize = description.SampleDesc.Count = 1u;
        description.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        description.Usage = D3D11_USAGE_DEFAULT;
        description.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        preparation_fixture_hr(converter_->device->CreateTexture2D(&description, nullptr, &target_), "Preparation output allocation failed");
        preparation_fixture_hr(converter_->device->CreateRenderTargetView(target_.Get(), nullptr, &target_rtv_), "Preparation output RTV failed");
        constexpr float clear[4] {};
        converter_->device_ctx->ClearRenderTargetView(target_rtv_.Get(), clear);
        output_width_ = output_width;
        draw_ = {};
      }

      model_preparation_converter_snapshot_t observe() const {
        assert_owner();
        auto result = draw_;
        result.conversions = conversions_;
        result.source_uploads = source_uploads_;
        result.source_stamp_ticks = image_->frame_timestamp ? image_->frame_timestamp->time_since_epoch().count() : 0;
        result.content_stamp_ticks = image_->content_timestamp ? image_->content_timestamp->time_since_epoch().count() : 0;
        if (const auto stamp = converter_->rendered_content_timestamp()) {
          result.rendered_stamp_ticks = stamp->time_since_epoch().count();
        }
        result.mode = converter_->sbs_mode;
        result.model_status = models::tensorrt_model_prepare_status(::video::host_sbs_v2_depth_model());
        result.renderer = converter_->host_sbs_renderer;
        result.pipeline_ready = ready_event_->peek();
        result.build_pending = converter_->has_pending_depth_pipeline_build();
        result.estimator_present = static_cast<bool>(converter_->depth_estimator);
        result.estimator_valid = result.estimator_present && converter_->depth_estimator->is_valid();
        result.estimator_terminal = result.estimator_present && converter_->depth_estimator->has_terminal_failure();
        result.subtitle_warp_ready =converter_->selected_matte_ready;
        result.text_game_warp_ready = converter_->selected_shared_warp_programs.text_game_ready();
        result.v2_live_warp_seen = converter_->v2_live_warp_seen;
        result.latest_lineage_authenticated = converter_->latest_v2_lineage.authenticated();
        result.latest_lineage_known_depth = converter_->latest_v2_lineage.known_depth();
        const auto &lineage = converter_->latest_v2_lineage;
        result.receipt = lineage.receipt;
        result.latest_lineage_receipt_present = lineage.receipt.has_value();
        result.latest_lineage_receipt_depth_cache_authorized = lineage.receipt && lineage.receipt->depth_cache_authorized();
        const auto &publication = lineage.estimate().publication;
        result.latest_lineage_receipt_matches_publication = lineage.receipt && models::host_sbs_gpu_completion_receipt::matches_publication(*lineage.receipt, publication);
        result.latest_lineage_publication_frame_id = publication.frame_id;
        result.sbs_frame_sequence = converter_->sbs_frame_sequence;
        result.flat_outputs = converter_->matched_stats_output_flat;
        result.warped_outputs = converter_->matched_stats_output_warped_new;
        result.repeat_outputs = converter_->matched_stats_output_packed_repeat;
        result.setup_watch_factory_calls = watch_probe_.calls;
        result.setup_watch_tokens_alive = watch_probe_.token.use_count();
        result.setup_watch_rect = watch_probe_.last_rect;
        result.setup_watch_present = static_cast<bool>(converter_->game_export.setup_watch);
        result.receiver_present = static_cast<bool>(converter_->game_export.receiver);
        result.production_observer_path = converter_->test_context == nullptr;
        result.registry_present = static_cast<bool>(converter_->video_player_registry);
        result.registry_ready = result.registry_present && converter_->video_player_registry->ready();
        result.registry_ready_observed = converter_->video_player_registry_ready_observed;
        result.registry_factory_calls = converter_->test_registry_factory_calls;
        result.registry_consumption_eligible = detail::host_sbs_window_authority_observation_needed(
          result.estimator_present,
          models::host_sbs_renderer_uses_depth_pipeline(converter_->host_sbs_renderer)
        );
        result.pipeline_enabled = converter_->sbs_mode != ::video::SBS_OFF && models::host_sbs_renderer_uses_depth_pipeline(converter_->host_sbs_renderer);
        result.depth_completion_pending = converter_->pending_matched_frame != nullptr;
        result.authority_reprocess_pending = converter_->depth_authority_reprocess_pending;
        result.subtitle_reprocess_pending = converter_->subtitle_context_reprocess_pending;
        result.dump_pending = converter_->sbs_dumper.needs_conversion_poll();
        result.gamma_pending = converter_->stream_gamma_conversion_pending;
        result.gamma_white_query_due =
          converter_->stream_gamma_mode_ != ::video::stream_gamma_mode_e::windows_default &&
          std::chrono::steady_clock::now() >= converter_->stream_gamma_white_query_at;
        return result;
      }

      model_preparation_converter_snapshot_t convert_retained() {
        assert_owner();
        const bool sampled_ready = ready_event_->peek();
        preparation_fixture_require(converter_->convert_rgb(*image_, target_.Get(), target_rtv_.Get(), false, std::nullopt) == d3d_base_encode_device::rgb_conversion_e::converted,
          "Preparation actual retained-source conversion failed");
        ++conversions_;
        if (sampled_ready) {
          (void) ready_event_->try_pop();
        }
        return observe();
      }

      bool needs_conversion_poll() const {
        assert_owner();
        return converter_->needs_conversion_poll();
      }

      bool pipeline_ready() const {
        assert_owner();
        return ready_event_->peek();
      }

      model_preparation_artifact_t snapshot_output(std::string_view label) {
        assert_owner();
        preparation_fixture_require(!label.empty() && label.size() <= 48u && std::all_of(label.begin(), label.end(), [](unsigned char value) {
          return (value >= 'a' && value <= 'z') || (value >= '0' && value <= '9') || value == '-';
        }), "Preparation artifact label invalid");
        D3D11_TEXTURE2D_DESC description {};
        target_->GetDesc(&description);
        description.Usage = D3D11_USAGE_STAGING;
        description.BindFlags = 0u;
        description.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Texture2D> staging;
        preparation_fixture_hr(converter_->device->CreateTexture2D(&description, nullptr, &staging), "Preparation output staging allocation failed");
        converter_->device_ctx->CopyResource(staging.Get(), target_.Get());
        std::vector<std::uint8_t> bytes(std::size_t(output_width_) * request_.height * 4u);
        D3D11_MAPPED_SUBRESOURCE mapped {};
        preparation_fixture_hr(converter_->device_ctx->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped), "Preparation explicit output readback failed");
        {
          auto unmap = util::fail_guard([&]() { converter_->device_ctx->Unmap(staging.Get(), 0u); });
          preparation_fixture_require(mapped.RowPitch >= output_width_ * 4u, "Preparation output readback pitch invalid");
          for (std::uint32_t row = 0; row != request_.height; ++row) {
            std::memcpy(bytes.data() + std::size_t(row) * output_width_ * 4u,
              static_cast<const std::uint8_t *>(mapped.pData) + std::size_t(row) * mapped.RowPitch, output_width_ * 4u);
          }
        }
        return write_artifact(std::string(label) + ".bgra8", bytes.data(), bytes.size(), output_width_, request_.height);
      }

      void upload_dirty_source(std::span<const std::uint8_t> bgra, const RECT &damage) {
        assert_owner();
        preparation_fixture_require(damage_history_ && bgra.size() == request_.bgra.size() && damage.left >= 0 && damage.top >= 0 && damage.right <= static_cast<LONG>(request_.width) && damage.bottom <= static_cast<LONG>(request_.height) && damage.left < damage.right && damage.top < damage.bottom, "Dirty source probe requires bounded known damage");
        bool changed = false;
        for (std::uint32_t y = 0; y != request_.height; ++y) {
          for (std::uint32_t x = 0; x != request_.width; ++x) {
            const auto offset = (std::size_t(y) * request_.width + x) * 4u;
            if (std::memcmp(bgra.data() + offset, request_.bgra.data() + offset, 4u) != 0) {
              preparation_fixture_require(x >= static_cast<unsigned>(damage.left) && x < static_cast<unsigned>(damage.right) && y >= static_cast<unsigned>(damage.top) && y < static_cast<unsigned>(damage.bottom), "Changed pixels escape declared damage");
              changed = true;
            }
          }
        }
        preparation_fixture_require(changed, "Dirty source probe refuses identical uploads");
        // Retain A's immutable pool image while the real matched lineage owns it. B is a
        // distinct shared capture allocation, as on the ordinary borrowed-capture path.
        auto replacement = display_->alloc_img();
        preparation_fixture_require(replacement && display_->complete_img(replacement.get(), false) == 0, "Dirty source allocation failed");
        auto &image = static_cast<img_d3d_t &>(*replacement);
        preparation_fixture_require(image.capture_mutex->AcquireSync(0u, 5000u) == S_OK, "Dirty source capture mutex failed");
        {
          auto release = util::fail_guard([&image]() {
            image.capture_mutex->ReleaseSync(0u);
          });
          display_->device_ctx->UpdateSubresource(image.capture_texture.get(), 0u, nullptr, bgra.data(), request_.width * 4u, 0u);
          display_->device_ctx->Flush();
        }
        request_.bgra.assign(bgra.begin(), bgra.end());
        image.ddup_damage = damage_history_->commit({true, {damage}});
        image.blank = false;
        replacement->frame_timestamp = replacement->content_timestamp = std::chrono::steady_clock::now();
        image_ = std::move(replacement);
        ++source_uploads_;
        draw_ = {};
        drawn_source_.Reset();
      }

      bool drawn_source_matches_upload() const {
        assert_owner();
        if (!drawn_source_) {
          return false;
        }
        ComPtr<ID3D11Resource> resource;
        drawn_source_->GetResource(&resource);
        ComPtr<ID3D11Texture2D> texture;
        preparation_fixture_hr(resource.As(&texture), "Draw source is not a texture");
        D3D11_TEXTURE2D_DESC description {};
        texture->GetDesc(&description);
        preparation_fixture_require(description.Width == request_.width && description.Height == request_.height && description.Format == DXGI_FORMAT_B8G8R8A8_UNORM, "Unexpected actual draw source shape");
        description.Usage = D3D11_USAGE_STAGING;
        description.BindFlags = description.MiscFlags = 0u;
        description.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Texture2D> staging;
        preparation_fixture_hr(converter_->device->CreateTexture2D(&description, nullptr, &staging), "Draw source staging failed");
        ComPtr<IDXGIKeyedMutex> mutex;
        preparation_fixture_hr(texture.As(&mutex), "Borrowed draw source keyed mutex unavailable");
        preparation_fixture_require(mutex->AcquireSync(0u, 5000u) == S_OK, "Draw source readback mutex failed");
        auto release = util::fail_guard([&]() { mutex->ReleaseSync(0u); });
        converter_->device_ctx->CopyResource(staging.Get(), texture.Get());
        D3D11_MAPPED_SUBRESOURCE mapped {};
        preparation_fixture_hr(converter_->device_ctx->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped), "Draw source readback failed");
        auto unmap = util::fail_guard([&]() {
          converter_->device_ctx->Unmap(staging.Get(), 0u);
        });
        preparation_fixture_require(mapped.RowPitch >= request_.width * 4u, "Draw source pitch invalid");
        for (std::uint32_t y = 0; y != request_.height; ++y) {
          if (std::memcmp(static_cast<const std::uint8_t *>(mapped.pData) + std::size_t(y) * mapped.RowPitch, request_.bgra.data() + std::size_t(y) * request_.width * 4u, request_.width * 4u) != 0) {
            return false;
          }
        }
        return true;
      }

      model_preparation_artifact_t source_artifact() const {
        assert_owner();
        return source_artifact_;
      }

      model_preparation_memory_t memory() const {
        assert_owner();
        return query_model_preparation_memory(request_.adapter);
      }

    private:
      void assert_owner() const {
        preparation_fixture_require(std::this_thread::get_id() == owner_thread_, "Preparation converter used outside its D3D owner");
      }

      model_preparation_artifact_t write_artifact(const std::string &name, const std::uint8_t *bytes, std::size_t size,
        std::uint32_t width, std::uint32_t height) const {
        const auto path = request_.output_root / name;
        preparation_fixture_require(!std::filesystem::exists(path), "Preparation probe refuses artifact overwrite");
        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<const char *>(bytes), static_cast<std::streamsize>(size));
        output.flush();
        preparation_fixture_require(output.good(), "Preparation probe artifact write failed");
        output.close();
        const auto hash = models::file_sha256_hex(path);
        preparation_fixture_require(hash.size() == 64u, "Preparation probe artifact hash failed");
        return {path, hash, width, height, width * 4u};
      }

      static void record(void *state, const converter_test_observer::event_t &event) noexcept {
        auto &self = *static_cast<impl_t *>(state);
        if (event.kind == converter_test_observer::event_e::admission) {
          self.draw_.last_admission_gpu_undecided = event.gpu_undecided;
          self.draw_.last_admission_optional_work = event.optional_work;
        }
        if (event.kind != converter_test_observer::event_e::draw) {
          return;
        }
        if (self.request_.hard_disabled_dirty_source) {
          self.drawn_source_ = event.draw_source;
        }
        self.draw_.last_draw_present = true;
        self.draw_.last_draw_flat = event.flat;
        self.draw_.last_draw_repeat = event.repeat;
        self.draw_.last_draw_publication_current = event.publication_current;
        self.draw_.last_draw_ema_motion_mask_present = event.ema_motion_mask != nullptr;
        self.draw_.last_draw_model_frame = event.model_frame;
        self.draw_.last_draw_observation_us = event.observation_us;
        self.draw_.last_draw_expected_work = event.publication.expected_work;
        self.draw_.last_draw_host_subtitle_outcome = static_cast<std::uint32_t>(event.publication.host_subtitle_outcome);
        self.draw_.last_draw_source_stamp_ticks = event.source_stamp_ticks;
        self.draw_.last_draw_content_stamp_ticks = event.content_stamp_ticks;
        self.draw_.last_draw_current_source_stamp_ticks = event.current_source_stamp_ticks;
        self.draw_.last_draw_current_content_stamp_ticks = event.current_content_stamp_ticks;
      }

      model_preparation_converter_request_t request_;
      std::thread::id owner_thread_;
      std::shared_ptr<safe::event_t<bool>> ready_event_;
      model_preparation_watch_probe_t watch_probe_;
      converter_test_observer::context_t test_context_;
      model_preparation_converter_snapshot_t draw_;
      model_preparation_artifact_t source_artifact_;
      std::shared_ptr<detail::ddup_damage_history_t> damage_history_;
      ComPtr<ID3D11ShaderResourceView> drawn_source_;
      std::shared_ptr<preparation_fixture_display_t> display_;
      std::shared_ptr<platf::img_t> image_;
      ComPtr<ID3D11Texture2D> target_;
      ComPtr<ID3D11RenderTargetView> target_rtv_;
      std::uint32_t output_width_ = 0;
      std::uint64_t conversions_ = 0;
      std::uint64_t source_uploads_ = 0;
      std::unique_ptr<d3d_base_encode_device> converter_;
    };

    model_preparation_converter_t::model_preparation_converter_t(model_preparation_converter_request_t request):
        impl_(std::make_unique<impl_t>(std::move(request))) {}
    model_preparation_converter_t::~model_preparation_converter_t() = default;
    void model_preparation_converter_t::set_mode(::video::sbs_mode_e mode) { impl_->set_mode(mode); }
    model_preparation_converter_snapshot_t model_preparation_converter_t::observe() const { return impl_->observe(); }
    model_preparation_converter_snapshot_t model_preparation_converter_t::convert_retained() { return impl_->convert_retained(); }
    bool model_preparation_converter_t::needs_conversion_poll() const { return impl_->needs_conversion_poll(); }
    bool model_preparation_converter_t::pipeline_ready() const { return impl_->pipeline_ready(); }
    model_preparation_artifact_t model_preparation_converter_t::snapshot_output(std::string_view label) { return impl_->snapshot_output(label); }

    void model_preparation_converter_t::upload_dirty_source(std::span<const std::uint8_t> bgra, const RECT &damage) {
      impl_->upload_dirty_source(bgra, damage);
    }

    bool model_preparation_converter_t::drawn_source_matches_upload() const {
      return impl_->drawn_source_matches_upload();
    }
    model_preparation_artifact_t model_preparation_converter_t::source_artifact() const { return impl_->source_artifact(); }
    model_preparation_memory_t model_preparation_converter_t::memory() const { return impl_->memory(); }
  }  // namespace testing
#endif

#ifdef SUNSHINE_TESTS
  namespace testing {
    class conversion_poll_probe_t::impl_t {
    public:
      explicit impl_t(std::optional<std::shared_future<std::vector<std::wstring>>> registry_future) {
        const char *dump_directory = std::getenv("APOLLO_SBS_DUMP");
        if (dump_directory && *dump_directory && config::sunshine.diagnostics_enabled) {
          throw std::invalid_argument("CPU conversion-poll probe requires diagnostic file triggers disabled");
        }
        // No init method is called: all D3D handles and the estimator remain empty.
        converter_ = std::make_unique<d3d_base_encode_device>();
        if (registry_future) {
          converter_->video_player_registry = std::make_unique<video_player::registry_t>(std::move(*registry_future));
        }
        converter_->sbs_dumper.set_button_request(button_);
      }

      ~impl_t() {
        clear_build();
      }

      conversion_poll_snapshot_t observe(
        const conversion_poll_inputs_t &inputs,
        const std::function<models::engine_build_status()> &read_model_status
      ) {
        if (!read_model_status) {
          throw std::invalid_argument("CPU conversion-poll probe requires a finite model-status reader");
        }
        if (converter_->device || converter_->device_ctx || converter_->depth_estimator) {
          throw std::logic_error("CPU conversion-poll probe acquired a graphics owner");
        }
        clear_build();
        if (inputs.build != conversion_poll_build_e::absent) {
          build_completion_.emplace();
          converter_->depth_estimator_build = build_completion_->get_future();
          if (inputs.build == conversion_poll_build_e::ready) {
            build_completion_->set_value({});
          }
        }
        converter_->sbs_mode = inputs.mode;
        converter_->host_sbs_renderer = inputs.renderer;
        converter_->video_player_registry_ready_observed = inputs.registry_ready_observed;
        converter_->pending_matched_frame = inputs.depth_completion_pending ?
                                             &converter_->matched_frame_slots.front() : nullptr;
        converter_->depth_authority_reprocess_pending = inputs.depth_authority_reprocess_pending;
        converter_->subtitle_context_reprocess_pending = inputs.subtitle_context_reprocess_pending;
        converter_->stream_gamma_conversion_pending = inputs.stream_gamma_conversion_pending;
        converter_->stream_gamma_mode_ = inputs.white_query_due ? ::video::stream_gamma_mode_e::gamma_2_2 : ::video::stream_gamma_mode_e::windows_default;
        converter_->stream_gamma_white_query_at = inputs.white_query_due.value_or(false) ?
                                                    std::chrono::steady_clock::time_point::min() :
                                                    std::chrono::steady_clock::time_point::max();
        button_->store(inputs.dump_button_requested, std::memory_order_release);

        conversion_poll_model_read_t model_read {read_model_status};
        conversion_poll_model_read_scope_t scope {model_read};
        const bool needs_conversion = converter_->needs_conversion_poll();
        const bool build_valid = converter_->depth_estimator_build.valid();
        return {
          .needs_conversion = needs_conversion,
          .registry_present = static_cast<bool>(converter_->video_player_registry),
          .registry_ready = converter_->video_player_registry && converter_->video_player_registry->ready(),
          .registry_ready_observed = converter_->video_player_registry_ready_observed,
          .build_valid = build_valid,
          .build_ready = build_valid && converter_->depth_estimator_build.wait_for(std::chrono::seconds {0}) == std::future_status::ready,
          .estimator_present = static_cast<bool>(converter_->depth_estimator),
          .device_present = static_cast<bool>(converter_->device),
          .context_present = static_cast<bool>(converter_->device_ctx),
          .model_status_reads = model_read.calls,
        };
      }

    private:
      void clear_build() {
        if (converter_ && converter_->depth_estimator_build.valid()) {
          if (converter_->depth_estimator_build.wait_for(std::chrono::seconds {0}) != std::future_status::ready) {
            build_completion_->set_value({});
          }
          // Consume the actual empty future before the real converter destructor: it must not
          // hand test-owned work to the process-lifetime estimator retirement service.
          (void) converter_->depth_estimator_build.get();
        }
        build_completion_.reset();
      }

      std::shared_ptr<std::atomic<bool>> button_ = std::make_shared<std::atomic<bool>>(false);
      std::optional<std::promise<depth_pipeline_build_result_t>> build_completion_;
      std::unique_ptr<d3d_base_encode_device> converter_;
    };

    conversion_poll_probe_t::conversion_poll_probe_t(std::optional<std::shared_future<std::vector<std::wstring>>> registry_future):
        impl_(std::make_unique<impl_t>(std::move(registry_future))) {}

    conversion_poll_probe_t::~conversion_poll_probe_t() = default;

    conversion_poll_snapshot_t conversion_poll_probe_t::observe(
      const conversion_poll_inputs_t &inputs,
      const std::function<models::engine_build_status()> &read_model_status
    ) {
      return impl_->observe(inputs, read_model_status);
    }
  }  // namespace testing
#endif

  local_rgb_converter_t::local_rgb_converter_t() noexcept = default;
  local_rgb_converter_t::local_rgb_converter_t(local_rgb_converter_t &&) noexcept = default;
  local_rgb_converter_t &local_rgb_converter_t::operator=(local_rgb_converter_t &&) noexcept = default;
  local_rgb_converter_t::~local_rgb_converter_t() = default;

  local_rgb_converter_t local_rgb_converter_t::create() {
    local_rgb_converter_t created;
    created.converter_ = std::make_unique<d3d_base_encode_device>();
    return created;
  }

  void local_rgb_converter_t::reset() noexcept {
    converter_.reset();
  }

  int local_rgb_converter_t::init(std::shared_ptr<platf::display_t> display, adapter_t::pointer adapter, pix_fmt_e pix_fmt) {
    return converter_->init(std::move(display), adapter, pix_fmt);
  }

  int local_rgb_converter_t::init_rgb_output(
    int width,
    int height,
    int sbs_mode,
    const config::video_t::sbs_t &settings,
    std::shared_ptr<safe::event_t<bool>> depth_pipeline_ready_event
  ) {
    return converter_->init_rgb_output(width, height, sbs_mode, settings, std::move(depth_pipeline_ready_event));
  }

  ID3D11Device *local_rgb_converter_t::device() const noexcept {
    return converter_->device.get();
  }

  ID3D11DeviceContext *local_rgb_converter_t::device_ctx() const noexcept {
    return converter_->device_ctx.get();
  }

  bool local_rgb_converter_t::needs_conversion_poll() const {
    return converter_->needs_conversion_poll();
  }

  bool local_rgb_converter_t::has_pending_depth_pipeline_build() const {
    return converter_->has_pending_depth_pipeline_build();
  }

  local_rgb_converter_t::conversion_e local_rgb_converter_t::convert_rgb(
    platf::img_t &img,
    ID3D11Texture2D *target_texture,
    ID3D11RenderTargetView *target,
    bool target_is_linear,
    const std::optional<std::chrono::steady_clock::time_point> presentation_target,
    destination_e destination
  ) {
    return converter_->convert_rgb(img, target_texture, target, target_is_linear, presentation_target, destination);
  }

  class d3d_nvenc_encode_device_t: public nvenc_encode_device_t {
  public:
    ~d3d_nvenc_encode_device_t() override {
      // The destructor body runs before nvenc_d3d is destroyed. Do not let a hung NVENC teardown
      // retain the capture display and deadlock capture-thread reinitialization.
      base.release_capture_display();
#ifdef SUNSHINE_TESTS
      if (test_after_capture_release_) {
        test_after_capture_release_();
      }
#endif
    }

    static native_encode_owner_t create(
      std::shared_ptr<platf::display_t> display,
      adapter_t::pointer adapter,
      const ::video::config_t &client_config,
      const ::video::sunshine_colorspace_t &colorspace
    ) {
      auto candidate = std::make_unique<d3d_nvenc_encode_device_t>();
      auto &device = *candidate;
      native_encode_owner_t owner {std::move(candidate)};
      const auto pix_fmt = colorspace.bit_depth == 10 ? pix_fmt_e::p010 : pix_fmt_e::nv12;
      if (!device.init_device(display, adapter, pix_fmt)) {
        return {};
      }
      device.colorspace = colorspace;
      if (::video::colorspace_is_hdr(colorspace)) {
        SS_HDR_METADATA metadata {};
        if (display->get_hdr_metadata(metadata)) {
          device.hdr_metadata = metadata;
        } else {
          BOOST_LOG(error) << "Couldn't get display HDR metadata for the native encoder bitstream."sv;
        }
      }
      if (!device.init_encoder(client_config, colorspace)) {
        return {};
      }
#ifdef SUNSHINE_TESTS
      if (const auto *fixture = dynamic_cast<const testing::preparation_fixture_display_t *>(display.get())) {
        device.base.test_native_pacing = fixture->native_pacing;
        device.base.test_observer = fixture->native_pacing.converter_observer;
      }
#endif
      return owner;
    }

    nvenc::nvenc_base &encoder() override {
      return *nvenc_d3d;
    }

  private:
    bool init_device(std::shared_ptr<platf::display_t> display, adapter_t::pointer adapter_p, pix_fmt_e pix_fmt) {
      buffer_format = nvenc::nvenc_format_from_sunshine_format(pix_fmt);
      if (buffer_format == NV_ENC_BUFFER_FORMAT_UNDEFINED) {
        BOOST_LOG(error) << "Unexpected pixel format for NvENC ["sv << from_pix_fmt(pix_fmt) << ']';
        return false;
      }

      if (base.init(display, adapter_p, pix_fmt)) {
        return false;
      }

      return true;
    }

    bool init_encoder(const ::video::config_t &client_config, const ::video::sunshine_colorspace_t &colorspace) {
      const bool tiled_chroma = client_config.sbs_mode == ::video::SBS_AI &&
                                ::video::colorspace_is_hdr(colorspace) && base.prepare_p010_chroma_tiles();
      nvenc_d3d = std::make_unique<nvenc::nvenc_d3d11_native>(base.device.get(), tiled_chroma);

      auto nvenc_colorspace = nvenc::nvenc_colorspace_from_sunshine_colorspace(colorspace);
      if (!nvenc_d3d->create_encoder(
            config::video.nv,
            client_config,
            nvenc_colorspace,
            buffer_format,
            hdr_metadata
          )) {
        return false;
      }

      if (!base.apply_colorspace(colorspace)) {
        return false;
      }
      return base.init_output({
               .frame_texture = nvenc_d3d->get_input_texture(),
               .width = client_config.width,
               .height = client_config.height,
               .sbs_mode_param = client_config.sbs_mode,
               .settings = client_config.sbs_config,
               .depth_status_event = client_config.sbs_depth_status_event,
               .depth_pipeline_ready_event = client_config.sbs_depth_pipeline_ready_event,
               .telemetry_event = client_config.sbs_telemetry_event,
               .telemetry_subscription = client_config.sbs_telemetry_subscription,
               .telemetry_generation = client_config.sbs_telemetry_generation,
               .sbs_debug_dump_request = client_config.sbs_debug_dump_pending,
               .telemetry_performance = client_config.sbs_telemetry_performance,
               .game_status_event = client_config.game_source_status_event,
               .effective_mode = client_config.effective_mode,
               .game_transport_supported = client_config.game_source_transport_supported,
               .game_width = client_config.game_source_width,
               .game_height = client_config.game_source_height,
               .native_input = nvenc_d3d.get(),
             }) == 0;
    }

  public:
    int convert(platf::img_t &img_base) override {
#ifdef SUNSHINE_TESTS
      return convert_with_native_pacing_probe(img_base, std::nullopt);
#else
      return base.convert(img_base);
#endif
    }

    bool set_stream_gamma(::video::stream_gamma_mode_e mode) override {
      return base.set_stream_gamma(mode);
    }

    ::video::stream_gamma_mode_e stream_gamma_mode() const override {
      return base.stream_gamma_mode();
    }

    float stream_gamma_white_nits() const override {
      return base.stream_gamma_white_nits();
    }

    int convert_with_encode_target(
      platf::img_t &img_base,
      const std::chrono::steady_clock::time_point next_encode_target
    ) override {
#ifdef SUNSHINE_TESTS
      return convert_with_native_pacing_probe(img_base, next_encode_target);
#else
      return base.convert(img_base, next_encode_target);
#endif
    }

    bool needs_conversion_poll() const override {
      return base.needs_conversion_poll();
    }

    void set_external_frame_wake(std::function<void()> wake) override {
      base.set_external_frame_wake(std::move(wake));
    }

    bool external_frame_held() const override {
      return base.external_frame_held();
    }

    bool capture_paced() const override {
      return base.capture_paced();
    }

    void encoder_consumed_input() override {
      base.encoder_consumed_input();
    }

    bool conversion_kept_input() const override {
      return base.conversion_kept_input();
    }

    std::optional<std::chrono::steady_clock::time_point> rendered_content_timestamp() const override {
      return base.rendered_content_timestamp();
    }

  private:
#ifdef SUNSHINE_TESTS
    int convert_with_native_pacing_probe(
      platf::img_t &image, std::optional<std::chrono::steady_clock::time_point> next_encode_target
    ) {
      const auto &probe = base.test_native_pacing;
      if (!probe.record_delivery) {
        return base.convert(image, next_encode_target);
      }
      const auto started = std::chrono::steady_clock::now();
      const int result = base.convert(image, next_encode_target);
      probe.record_delivery(probe.converter_observer.state, {
        started, std::chrono::steady_clock::now(), image.frame_timestamp, result,
      });
      return result;
    }
#endif
    d3d_base_encode_device base;
    std::unique_ptr<nvenc::nvenc_d3d11_native> nvenc_d3d;
    NV_ENC_BUFFER_FORMAT buffer_format = NV_ENC_BUFFER_FORMAT_UNDEFINED;
#ifdef SUNSHINE_TESTS
    friend struct native_capture_release_test_access;
    std::function<void()> test_after_capture_release_;
#endif
  };

#ifdef SUNSHINE_TESTS
  #include "display_vram_native_owner_test.inc"
#endif

  bool set_cursor_texture(device_t::pointer device, gpu_cursor_t &cursor, const std::vector<std::uint8_t> &cursor_img, const sbs_cursor::shape_t &shape) {
    // This cursor image may not be used
    if (cursor_img.size() == 0) {
      cursor.input_res.reset();
      cursor.set_texture(0, 0, nullptr);
      return true;
    }

    D3D11_SUBRESOURCE_DATA data {
      cursor_img.data(),
      4 * shape.width,
      0
    };

    // Create texture for cursor
    D3D11_TEXTURE2D_DESC t {};
    t.Width = shape.width;
    t.Height = shape.height;
    t.MipLevels = 1;
    t.ArraySize = 1;
    t.SampleDesc.Count = 1;
    t.Usage = D3D11_USAGE_IMMUTABLE;
    t.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    t.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    texture2d_t texture;
    auto status = device->CreateTexture2D(&t, &data, &texture);
    if (FAILED(status)) {
      BOOST_LOG(error) << "Failed to create mouse texture [0x"sv << util::hex(status).to_string_view() << ']';
      return false;
    }

    // Free resources before allocating on the next line.
    cursor.input_res.reset();
    status = device->CreateShaderResourceView(texture.get(), nullptr, &cursor.input_res);
    if (FAILED(status)) {
      BOOST_LOG(error) << "Failed to create cursor shader resource view [0x"sv << util::hex(status).to_string_view() << ']';
      return false;
    }

    cursor.set_texture(t.Width, t.Height, std::move(texture));
    return true;
  }

  void display_ddup_vram_t::add_damage_observer() noexcept {
    damage_observers_.fetch_add(1, std::memory_order_release);
  }

  void display_ddup_vram_t::remove_damage_observer() noexcept {
    damage_observers_.fetch_sub(1, std::memory_order_release);
  }

  static D3D11_BOX ddup_cursor_bounds(const sbs_cursor::snapshot_t &cursor) {
    // The alpha and XOR planes of one decoded shape use the same integer viewport. It is
    // already in capture texture coordinates, including rotation and offscreen placement.
    const auto clip = [](float coordinate, UINT extent) {
      return static_cast<UINT>(std::clamp(coordinate, 0.0f, static_cast<float>(extent)));
    };
    const auto &view = cursor.viewport;
    return {
      clip(view.TopLeftX, cursor.capture_width),
      clip(view.TopLeftY, cursor.capture_height),
      0u,
      clip(view.TopLeftX + view.Width, cursor.capture_width),
      clip(view.TopLeftY + view.Height, cursor.capture_height),
      1u,
    };
  }

  capture_e display_ddup_vram_t::snapshot(const pull_free_image_cb_t &pull_free_image_cb, std::shared_ptr<platf::img_t> &img_out, std::chrono::milliseconds timeout, bool cursor_visible) {
    DXGI_OUTDUPL_FRAME_INFO frame_info {};

    resource_t::pointer res_p {};
    const auto capture_status = dup.next_frame(frame_info, timeout, &res_p);
    resource_t res {res_p};

    const bool record_damage = damage_observers_.load(std::memory_order_acquire) != 0;
    if (!record_damage && damage_history) {
      // Subsequent captures can omit metadata, so the next observer needs a fresh history.
      last_ddup_damage.reset();
      damage_history.reset();
    }
    // Sampled once per acquisition; the encode thread may claim or release the output any time.
    const bool export_owns_output = external_pixels_owner.load(std::memory_order_acquire) != 0;
    // Only an idle source wait re-duplicates. The short pacing probe keeps the duplication.
    if (capture_status == capture_e::timeout && detail::capture_wait_policy_t::waited_for_source(timeout) && dropped_desktop.reduplicate_on_timeout(export_owns_output)) {
      return reduplicate_output();
    }
    if (capture_status != capture_e::ok) {
      return capture_status;
    }

    const bool mouse_update = frame_info.LastMouseUpdateTime.QuadPart != 0 || frame_info.PointerShapeBufferSize > 0;
    const bool desktop_present = frame_info.LastPresentTime.QuadPart != 0;
    const auto acquisition = dropped_desktop.acquire(export_owns_output, desktop_present);
    const bool prior_damage_chain_valid = damage_chain_valid;
    const bool desktop_update = desktop_present ||
                                (res && (acquisition.read_desktop ||
                                         (!prior_damage_chain_valid && !acquisition.skip_pixels)));
    const bool uncertain_desktop = frame_info.ProtectedContentMaskedOut ||
                                   (!desktop_present && (frame_info.AccumulatedFrames || frame_info.TotalMetadataBufferSize || frame_info.RectsCoalesced));
    if (desktop_update || uncertain_desktop) {
      // Retire every pooled copy even if this desktop is subsequently dropped or fails to copy.
      ++desktop_serial_;
    }
    if (!mouse_update && !desktop_update) {
      return capture_e::timeout;
    }
    // A present becomes content after the copy. An uncopied present breaks damage continuity.
    if (desktop_update) {
      damage_chain_valid = false;
    }

    if (frame_info.PointerShapeBufferSize > 0) {
      DXGI_OUTDUPL_POINTER_SHAPE_INFO shape_info {};

      util::buffer_t<std::uint8_t> img_data {frame_info.PointerShapeBufferSize};

      UINT dummy;
      const auto status = dup.dup->GetFramePointerShape(img_data.size(), std::begin(img_data), &dummy, &shape_info);
      if (FAILED(status)) {
        BOOST_LOG(error) << "Failed to get new pointer shape [0x"sv << util::hex(status).to_string_view() << ']';

        return capture_e::error;
      }

      auto decoded = dummy <= img_data.size() ?
                       sbs_cursor::decode_shape(shape_info, std::begin(img_data), dummy) : std::nullopt;
      if (!decoded) {
        // A malformed cursor must not interrupt the desktop/game stream or leave the previous
        // pointer shape visible. Empty planes clear both capture textures and the SBS snapshot.
        BOOST_LOG(warning) << "Ignoring invalid Desktop Duplication cursor shape."sv;
        decoded.emplace();
      }
      auto shape = std::make_shared<sbs_cursor::shape_t>(std::move(*decoded));
      if (!set_cursor_texture(device.get(), cursor_alpha, shape->alpha_bgra, *shape) || !set_cursor_texture(device.get(), cursor_xor, shape->xor_bgra, *shape)) {
        return capture_e::error;
      }
      cursor_shape = std::move(shape);
    }

    if (frame_info.LastMouseUpdateTime.QuadPart) {
      cursor_alpha.set_pos(frame_info.PointerPosition.Position.x, frame_info.PointerPosition.Position.y, width, height, display_rotation, frame_info.PointerPosition.Visible);

      cursor_xor.set_pos(frame_info.PointerPosition.Position.x, frame_info.PointerPosition.Position.y, width, height, display_rotation, frame_info.PointerPosition.Visible);
    }

    const bool blend_mouse_cursor = (cursor_alpha.visible || cursor_xor.visible) && cursor_visible;
    const bool skip_pixels =
      (acquisition.skip_pixels || (dropped_desktop.dropped() && !desktop_update)) &&
      (desktop_update || capture_format != DXGI_FORMAT_UNKNOWN);
    const bool dummy = !desktop_update && !last_content_timestamp;
    const bool redraw_pointer = !dummy && (blend_mouse_cursor || last_pixel_cursor_visible);
    if (!skip_pixels && !desktop_update && !redraw_pointer && !dummy) {
      // The consumer may still hold the preceding image. Leave its pixels and metadata intact.
      img_out.reset();
      return capture_e::timeout;
    }

    texture2d_t src;
    if (desktop_update || (!skip_pixels && redraw_pointer)) {
      // A successful DDup acquisition supplies the complete desktop, even on cursor-only
      // updates. Read it while this frame is held and keep content identity independent.
      if (!res) {
        BOOST_LOG(warning) << "Desktop Duplication did not supply the required desktop bitmap."sv;
        return capture_e::reinit;
      }
      const auto status = res->QueryInterface(IID_ID3D11Texture2D, (void **) &src);
      if (FAILED(status)) {
        BOOST_LOG(error) << "Couldn't query interface [0x"sv << util::hex(status).to_string_view() << ']';
        return capture_e::error;
      }

      D3D11_TEXTURE2D_DESC desc {};
      src->GetDesc(&desc);
      // A mode change can race enumeration and invalidate the image pool.
      if (desc.Width != width_before_rotation || desc.Height != height_before_rotation) {
        BOOST_LOG(info) << "Capture size changed ["sv << width << 'x' << height << " -> "sv << desc.Width << 'x' << desc.Height << ']';
        return capture_e::reinit;
      }
      if (capture_format == DXGI_FORMAT_UNKNOWN) {
        capture_format = desc.Format;
        BOOST_LOG(info) << "Capture format ["sv << dxgi_format_to_string(capture_format) << ']';
      }
      if (capture_format != desc.Format) {
        BOOST_LOG(info) << "Capture format changed ["sv << dxgi_format_to_string(capture_format) << " -> "sv << dxgi_format_to_string(desc.Format) << ']';
        return capture_e::reinit;
      }
    }

    sbs_cursor::snapshot_t cursor;
    if (blend_mouse_cursor) {
      cursor = {
        .shape = cursor_shape,
        .viewport = cursor_alpha.texture ? cursor_alpha.cursor_view : cursor_xor.cursor_view,
        .capture_width = static_cast<std::uint32_t>(width_before_rotation),
        .capture_height = static_cast<std::uint32_t>(height_before_rotation),
        .rotation = display_rotation,
        .visible = true,
      };
    }
    // Metadata-only delivery leaves the encoder without a reusable desktop image. Let the first
    // otherwise useful redraw after export restore its pixels even when the cursor is unchanged.
    const bool compare_cursor = skip_pixels || (!desktop_update && redraw_pointer && !last_delivery_skipped_pixels);
    const bool same_cursor = compare_cursor && sbs_cursor::same_presentation(cursor, last_delivered_cursor);
    if (!desktop_update && same_cursor) {
      img_out.reset();
      return capture_e::timeout;
    }

    const auto timestamp_now = std::chrono::steady_clock::now();
    const auto timestamp_qpc = qpc_counter();
    const auto from_qpc = [&](const std::int64_t value) {
      return timestamp_now - qpc_time_difference(timestamp_qpc, value);
    };
    const auto present_timestamp = desktop_present ?
                                     std::optional {from_qpc(frame_info.LastPresentTime.QuadPart)} :
                                     std::nullopt;
    auto frame_timestamp = detail::select_ddup_timestamps(
      present_timestamp,
      frame_info.LastMouseUpdateTime.QuadPart != 0 ?
        std::optional {from_qpc(frame_info.LastMouseUpdateTime.QuadPart)} :
        std::nullopt,
      last_content_timestamp
    ).presentation_timestamp;
    if (desktop_update && !frame_timestamp) {
      frame_timestamp = timestamp_now;
    }

    detail::ddup_damage_update_t damage;
    if (damage_history && !skip_pixels && desktop_present && prior_damage_chain_valid && !acquisition.unknown_damage) {
      damage = dup.damage_update(frame_info, display_rotation, width_before_rotation, height_before_rotation);
      damage.single_desktop_present = detail::ddup_frame_has_one_desktop_present(frame_info);
    }

    if (skip_pixels && desktop_update) {
      last_ddup_damage.reset();
    }
    if (!pull_free_image_cb(img_out)) {
      return capture_e::interrupted;
    }
    auto *image = static_cast<img_d3d_t *>(img_out.get());
    if (skip_pixels) {
      // Export reads only metadata. Keep the cursor in the last real desktop image independent.
      image->ddup_damage.reset();
    } else {
      const bool reclear_dummy = !src && dummy && !image->blank && image->capture_texture;
      if (complete_img(image, dummy)) {
        return capture_e::error;
      }
      detail::ddup_capture_copy_plan_t copy {detail::ddup_capture_copy_kind_e::none};
      if (src) {
        const auto &cursor_box = image->ddup_cursor_box;
        const RECT old_cursor {static_cast<LONG>(cursor_box.left), static_cast<LONG>(cursor_box.top),
                               static_cast<LONG>(cursor_box.right), static_cast<LONG>(cursor_box.bottom)};
        copy = detail::plan_ddup_capture_copy(
          image->ddup_desktop_serial, desktop_serial_, image->ddup_damage,
          last_ddup_damage, damage, old_cursor, width_before_rotation, height_before_rotation
        );
      }
      if (copy.kind != detail::ddup_capture_copy_kind_e::none || reclear_dummy || blend_mouse_cursor) {
        detail::keyed_mutex_lock_t lock {image->capture_mutex.get()};
        if (const auto status = lock.lock(); status != S_OK) {
          BOOST_LOG(error) << "Failed to lock capture texture [0x"sv << util::hex(status).to_string_view() << ']';
          return capture_e::error;
        }
        if (copy.kind == detail::ddup_capture_copy_kind_e::full) {
          device_ctx->CopyResource(image->capture_texture.get(), src.get());
        } else if (copy.kind == detail::ddup_capture_copy_kind_e::region) {
          // The exact retained interval covers every changed desktop pixel. Read one
          // conservative rectangle from the current full bitmap, including this slot's old
          // cursor. Moves need no replay or temporary surface; the source is already current.
          const D3D11_BOX box {static_cast<UINT>(copy.region.left), static_cast<UINT>(copy.region.top), 0u,
                               static_cast<UINT>(copy.region.right), static_cast<UINT>(copy.region.bottom), 1u};
          device_ctx->CopySubresourceRegion(image->capture_texture.get(), 0u, box.left, box.top, 0u, src.get(), 0u, &box);
        } else if (reclear_dummy) {
          const float rgb_black[4] {};
          device_ctx->ClearRenderTargetView(image->capture_rt.get(), rgb_black);
        }

        if (blend_mouse_cursor) {
          device_ctx->VSSetShader(cursor_vs.get(), nullptr, 0);
          device_ctx->PSSetShader(cursor_ps.get(), nullptr, 0);
          device_ctx->OMSetRenderTargets(1, &image->capture_rt, nullptr);

          if (cursor_alpha.texture) {
            device_ctx->OMSetBlendState(blend_alpha.get(), nullptr, 0xFFFFFFFFu);
            device_ctx->PSSetShaderResources(0, 1, &cursor_alpha.input_res);
            device_ctx->RSSetViewports(1, &cursor_alpha.cursor_view);
            device_ctx->Draw(3, 0);
          }
          if (cursor_xor.texture) {
            device_ctx->OMSetBlendState(blend_invert.get(), nullptr, 0x00FFFFFFu);
            device_ctx->PSSetShaderResources(0, 1, &cursor_xor.input_res);
            device_ctx->RSSetViewports(1, &cursor_xor.cursor_view);
            device_ctx->Draw(3, 0);
          }

          device_ctx->OMSetBlendState(blend_disable.get(), nullptr, 0xFFFFFFFFu);
          ID3D11RenderTargetView *empty_target = nullptr;
          device_ctx->OMSetRenderTargets(1, &empty_target, nullptr);
          device_ctx->RSSetViewports(0, nullptr);
          ID3D11ShaderResourceView *empty_srv = nullptr;
          device_ctx->PSSetShaderResources(0, 1, &empty_srv);
        }
      }
      image->ddup_desktop_serial = src && !uncertain_desktop ? desktop_serial_ : 0;
      image->ddup_cursor_box = cursor.visible ? ddup_cursor_bounds(cursor) : D3D11_BOX {};

      if (desktop_update) {
        last_ddup_damage.reset();
        if (record_damage) {
          if (!damage_history) {
            // This copied desktop is the first baseline. Only later surfaces need interval metadata.
            damage_history = std::make_shared<detail::ddup_damage_history_t>();
          }
          auto snapshot = damage_history->commit(std::move(damage));
          if (snapshot.history && snapshot.token != 0u) {
            last_ddup_damage = std::move(snapshot);
          }
        }
        last_content_timestamp = present_timestamp ? present_timestamp : frame_timestamp;
        damage_chain_valid = true;
        dropped_desktop.copied();
      }
      image->ddup_damage = last_ddup_damage;
      last_pixel_cursor_visible = blend_mouse_cursor;
    }

    image->cursor = cursor;
    image->pixels_skipped = skip_pixels;
    image->conversion_needed = !skip_pixels || !same_cursor;
    image->blank = false;
    image->frame_timestamp = frame_timestamp;
    image->content_timestamp = last_content_timestamp;
    last_delivered_cursor = std::move(cursor);
    last_delivery_skipped_pixels = skip_pixels;
    return capture_e::ok;
  }

  capture_e display_ddup_vram_t::release_snapshot() {
    return dup.release_frame();
  }

  capture_e display_ddup_vram_t::reduplicate_output() {
    BOOST_LOG(info) << "Desktop Duplication: re-duplicating the output to read the desktop a Game 3D export covered."sv;
    ++desktop_serial_;
    // Duplicating resets the detected format. Keep it so the first acquisition still detects a
    // real format change (and reinitializes) instead of delivering a dummy frame.
    const auto format = capture_format;
    if (const auto status = dup.reset(); status != capture_e::ok) {
      return status;
    }
    if (dup.init(this)) {
      return capture_e::reinit;
    }
    capture_format = format;
    return capture_e::timeout;
  }

  int display_ddup_vram_t::init(const ::video::config_t &config, const std::string &display_name) {
    ++desktop_serial_;
    cursor_shape.reset();
    last_content_timestamp.reset();
    last_ddup_damage.reset();
    dropped_desktop = {};
    last_delivered_cursor = {};
    last_delivery_skipped_pixels = false;
    last_pixel_cursor_visible = false;
    damage_history.reset();
    damage_chain_valid = true;
    if (display_base_t::init(config, display_name, capture_backend_e::ddup) || dup.init(this)) {
      return -1;
    }

    D3D11_SAMPLER_DESC sampler_desc {};
    sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    sampler_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    sampler_desc.MinLOD = 0;
    sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;

    auto status = device->CreateSamplerState(&sampler_desc, &sampler_linear);
    if (FAILED(status)) {
      BOOST_LOG(error) << "Failed to create point sampler state [0x"sv << util::hex(status).to_string_view() << ']';
      return -1;
    }

    status = device->CreateVertexShader(cursor_vs_hlsl->GetBufferPointer(), cursor_vs_hlsl->GetBufferSize(), nullptr, &cursor_vs);
    if (status) {
      BOOST_LOG(error) << "Failed to create scene vertex shader [0x"sv << util::hex(status).to_string_view() << ']';
      return -1;
    }

    {
      int32_t rotation_modifier = display_rotation == DXGI_MODE_ROTATION_UNSPECIFIED ? 0 : display_rotation - 1;
      int32_t rotation_data[16 / sizeof(int32_t)] {rotation_modifier};  // aligned to 16-byte
      auto rotation = make_buffer(device.get(), rotation_data);
      if (!rotation) {
        BOOST_LOG(error) << "Failed to create display rotation vertex constant buffer";
        return -1;
      }
      device_ctx->VSSetConstantBuffers(2, 1, &rotation);
    }

    if (config.dynamicRange && is_hdr()) {
      // This shader will normalize scRGB white levels to a user-defined white level
      status = device->CreatePixelShader(cursor_ps_normalize_white_hlsl->GetBufferPointer(), cursor_ps_normalize_white_hlsl->GetBufferSize(), nullptr, &cursor_ps);
      if (status) {
        BOOST_LOG(error) << "Failed to create cursor blending (normalized white) pixel shader [0x"sv << util::hex(status).to_string_view() << ']';
        return -1;
      }

      constexpr float fallback_sdr_white_nits = 203.0f;
      const auto queried_sdr_white_nits = get_sdr_white_nits();
      const float sdr_white_nits =
        queried_sdr_white_nits && *queried_sdr_white_nits > 0.0f ?
          *queried_sdr_white_nits :
          fallback_sdr_white_nits;
      if (!queried_sdr_white_nits || *queried_sdr_white_nits <= 0.0f) {
        BOOST_LOG(warning)
          << "Failed to query the display's SDR reference white; using "sv
          << fallback_sdr_white_nits << " nits for the HDR cursor."sv;
      }

      // scRGB 1.0 represents 80 nits. Cursor texture RGB was decoded from sRGB by the shader.
      float white_multiplier_data[16 / sizeof(float)] {sdr_white_nits / 80.0f};  // aligned to 16-byte
      auto white_multiplier = make_buffer(device.get(), white_multiplier_data);
      if (!white_multiplier) {
        BOOST_LOG(warning) << "Failed to create cursor blending (normalized white) white multiplier constant buffer";
        return -1;
      }

      device_ctx->PSSetConstantBuffers(1, 1, &white_multiplier);
    } else {
      status = device->CreatePixelShader(cursor_ps_hlsl->GetBufferPointer(), cursor_ps_hlsl->GetBufferSize(), nullptr, &cursor_ps);
      if (status) {
        BOOST_LOG(error) << "Failed to create cursor blending pixel shader [0x"sv << util::hex(status).to_string_view() << ']';
        return -1;
      }
    }

    blend_alpha = make_blend(device.get(), true, false);
    blend_invert = make_blend(device.get(), true, true);
    blend_disable = make_blend(device.get(), false, false);

    if (!blend_disable || !blend_alpha || !blend_invert) {
      return -1;
    }

    device_ctx->OMSetBlendState(blend_disable.get(), nullptr, 0xFFFFFFFFu);
    device_ctx->PSSetSamplers(0, 1, &sampler_linear);
    device_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    return 0;
  }

  /**
   * Get the next frame from the Windows.Graphics.Capture API and copy it into a new snapshot texture.
   * @param pull_free_image_cb call this to get a new free image from the video subsystem.
   * @param img_out the captured frame is returned here
   * @param timeout how long to wait for the next frame
   * @param cursor_visible
   */
  capture_e display_wgc_vram_t::snapshot(const pull_free_image_cb_t &pull_free_image_cb, std::shared_ptr<platf::img_t> &img_out, std::chrono::milliseconds timeout, bool cursor_visible) {
    texture2d_t src;
    detail::wgc_timestamp_t frame_time {};
    dup.set_cursor_visible(cursor_visible);
    auto capture_status = dup.next_frame(timeout, &src, frame_time);
    // Sampled once per frame; the encode thread may claim or release the output any time.
    const bool export_owns_output = external_pixels_owner.load(std::memory_order_acquire) != 0;
    // A static screen sends no frame once an export releases the output. Copy the newest frame
    // forwarded without pixels meanwhile, which is still the current screen: a timeout means no
    // newer frame is waiting. Unlike a re-duplication this is one cheap copy, so a short pacing
    // probe may make it too, and the desktop returns without waiting for an idle source timeout.
    const bool kept_frame = capture_status == capture_e::timeout && !export_owns_output && dup.kept_frame_texture(&src, frame_time);
    if (capture_status != capture_e::ok && !kept_frame) {
      return capture_status;
    }

    const auto timestamp_now = std::chrono::steady_clock::now();
    const auto frame_timestamp = kept_frame ? timestamp_now :
                                              timestamp_now - detail::wgc_frame_age(
                                                                qpc_counter(),
                                                                qpc_frequency(),
                                                                frame_time
                                                              );
    D3D11_TEXTURE2D_DESC desc {};
    src->GetDesc(&desc);

    // It's possible for our display enumeration to race with mode changes and result in
    // mismatched image pool and desktop texture sizes. If this happens, just reinit again.
    if (desc.Width != width_before_rotation || desc.Height != height_before_rotation) {
      BOOST_LOG(info) << "Capture size changed ["sv << width << 'x' << height << " -> "sv << desc.Width << 'x' << desc.Height << ']';
      return capture_e::reinit;
    }

    // It's also possible for the capture format to change on the fly. If that happens,
    // reinitialize capture to try format detection again and create new images.
    if (capture_format != desc.Format) {
      BOOST_LOG(info) << "Capture format changed ["sv << dxgi_format_to_string(capture_format) << " -> "sv << dxgi_format_to_string(desc.Format) << ']';
      return capture_e::reinit;
    }

    std::shared_ptr<platf::img_t> img;
    if (!pull_free_image_cb(img)) {
      return capture_e::interrupted;
    }

    auto d3d_img = std::static_pointer_cast<img_d3d_t>(img);
    d3d_img->ddup_damage.reset();
    d3d_img->cursor = {};
    d3d_img->blank = false;  // image is always ready for capture
    if (export_owns_output) {
      // A live external export owns the encoded pixels. WGC exposes no cursor metadata, so the
      // frame carries nothing new for that encoder: forward its timestamp without a copy, and
      // keep the frame for the copy once the export releases the output.
      d3d_img->pixels_skipped = true;
      d3d_img->conversion_needed = false;
      dup.keep_frame();
    } else if (complete_img(d3d_img.get(), false) == 0) {
      detail::keyed_mutex_lock_t lock {d3d_img->capture_mutex.get()};
      if (const auto status = lock.lock(); status != S_OK) {
        BOOST_LOG(error) << "Failed to lock capture texture [0x"sv << util::hex(status).to_string_view() << ']';
        return capture_e::error;
      }
      device_ctx->CopyResource(d3d_img->capture_texture.get(), src.get());
      // This copy, or a newer frame's, supersedes any kept frame.
      dup.drop_kept_frame();
    } else {
      return capture_e::error;
    }
    img_out = img;
    if (img_out) {
      img_out->frame_timestamp = frame_timestamp;
      // WGC does not expose a LastPresentTime equivalent that distinguishes desktop content from
      // cursor-only compositor frames. Window-region attribution therefore abstains on WGC.
      img_out->content_timestamp.reset();
    }

    return capture_e::ok;
  }

  capture_e display_wgc_vram_t::release_snapshot() {
    return dup.release_frame();
  }

  int display_wgc_vram_t::init(const ::video::config_t &config, const std::string &display_name) {
    if (display_base_t::init(config, display_name, capture_backend_e::wgc) || dup.init(this, config)) {
      return -1;
    }

    return 0;
  }

  std::shared_ptr<platf::img_t> display_vram_t::alloc_img() {
    auto img = std::make_shared<img_d3d_t>();

    // Initialize format-independent fields
    img->width = width_before_rotation;
    img->height = height_before_rotation;
    img->id = next_image_id++;
    img->blank = true;

    return img;
  }

  // This cannot use ID3D11DeviceContext because it can be called concurrently by the encoding thread
  int display_vram_t::complete_img(platf::img_t *img_base, bool dummy) {
    auto img = (img_d3d_t *) img_base;

    if (dummy) {
      img->ddup_damage.reset();
      img->ddup_desktop_serial = 0;
    }

    // If this already has a capture texture and it's not switching dummy state, nothing to do
    if (img->capture_texture && img->dummy == dummy) {
      return 0;
    }

    // If this is not a dummy image, we must know the format by now
    if (!dummy && capture_format == DXGI_FORMAT_UNKNOWN) {
      BOOST_LOG(error) << "display_vram_t::complete_img() called with unknown capture format!";
      return -1;
    }

    // Reset the image (in case this was previously a dummy)
    ++img->capture_texture_generation;
    img->ddup_desktop_serial = 0;
    img->capture_texture.reset();
    img->capture_rt.reset();
    img->capture_mutex.reset();
    img->data = nullptr;
    if (img->encoder_texture_handle) {
      CloseHandle(img->encoder_texture_handle);
      img->encoder_texture_handle = nullptr;
    }

    // Initialize format-dependent fields
    img->pixel_pitch = get_pixel_pitch();
    img->row_pitch = img->pixel_pitch * img->width;
    img->dummy = dummy;
    img->format = (capture_format == DXGI_FORMAT_UNKNOWN) ? DXGI_FORMAT_B8G8R8A8_UNORM : capture_format;

    // The early-out above treats a set capture_texture as a complete image. A failure part-way
    // must therefore leave none of it behind, or a reused pooled image would be accepted with a
    // null render target, keyed mutex or shared handle.
    auto incomplete = util::fail_guard([img]() {
      img->capture_texture.reset();
      img->capture_rt.reset();
      img->capture_mutex.reset();
      img->data = nullptr;
      if (img->encoder_texture_handle) {
        CloseHandle(img->encoder_texture_handle);
        img->encoder_texture_handle = nullptr;
      }
    });

    D3D11_TEXTURE2D_DESC t {};
    t.Width = img->width;
    t.Height = img->height;
    t.MipLevels = 1;
    t.ArraySize = 1;
    t.SampleDesc.Count = 1;
    t.Usage = D3D11_USAGE_DEFAULT;
    t.Format = img->format;
    t.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    t.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;

    auto status = device->CreateTexture2D(&t, nullptr, &img->capture_texture);
    if (FAILED(status)) {
      BOOST_LOG(error) << "Failed to create img buf texture [0x"sv << util::hex(status).to_string_view() << ']';
      return -1;
    }

    status = device->CreateRenderTargetView(img->capture_texture.get(), nullptr, &img->capture_rt);
    if (FAILED(status)) {
      BOOST_LOG(error) << "Failed to create render target view [0x"sv << util::hex(status).to_string_view() << ']';
      return -1;
    }

    // Get the keyed mutex to synchronize with the encoding code
    status = img->capture_texture->QueryInterface(__uuidof(IDXGIKeyedMutex), (void **) &img->capture_mutex);
    if (FAILED(status)) {
      BOOST_LOG(error) << "Failed to query IDXGIKeyedMutex [0x"sv << util::hex(status).to_string_view() << ']';
      return -1;
    }

    resource1_t resource;
    status = img->capture_texture->QueryInterface(__uuidof(IDXGIResource1), (void **) &resource);
    if (FAILED(status)) {
      BOOST_LOG(error) << "Failed to query IDXGIResource1 [0x"sv << util::hex(status).to_string_view() << ']';
      return -1;
    }

    // Create a handle for the encoder device to use to open this texture
    status = resource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ, nullptr, &img->encoder_texture_handle);
    if (FAILED(status)) {
      BOOST_LOG(error) << "Failed to create shared texture handle [0x"sv << util::hex(status).to_string_view() << ']';
      return -1;
    }

    img->data = (std::uint8_t *) img->capture_texture.get();

    incomplete.disable();
    return 0;
  }

  // This cannot use ID3D11DeviceContext because it can be called concurrently by the encoding thread
  /**
   * @memberof platf::dxgi::display_vram_t
   */
  int display_vram_t::dummy_img(platf::img_t *img_base) {
    return complete_img(img_base, true);
  }

  std::vector<DXGI_FORMAT> display_vram_t::get_supported_capture_formats() {
    return {
      // scRGB FP16 is the ideal format for Wide Color Gamut and Advanced Color
      // displays (both SDR and HDR). This format uses linear gamma, so we will
      // use a linear->PQ shader for HDR and the declared BT.709/BT.2020 transfer for SDR.
      DXGI_FORMAT_R16G16B16A16_FLOAT,

      // DXGI_FORMAT_R10G10B10A2_UNORM seems like it might give us frames already
      // converted to SMPTE 2084 PQ, however it seems to actually just clamp the
      // scRGB FP16 values that DWM is using when the desktop format is scRGB FP16.
      //
      // If there is a case where the desktop format is really SMPTE 2084 PQ, it
      // might make sense to support capturing it without conversion to scRGB,
      // but we avoid it for now.

      // We include the 8-bit modes too for when the display is in SDR mode,
      // while the client stream is HDR-capable. These UNORM formats can
      // use our normal pixel shaders that expect sRGB input.
      DXGI_FORMAT_B8G8R8A8_UNORM,
      DXGI_FORMAT_B8G8R8X8_UNORM,
      DXGI_FORMAT_R8G8B8A8_UNORM,
    };
  }

  /**
   * @brief Check that a given codec is supported by the display device.
   * @param name The native NVENC codec name.
   * @param config The codec configuration.
   * @return `true` if supported, `false` otherwise.
   */
  bool display_vram_t::is_codec_supported(std::string_view name, const ::video::config_t &config) {
    DXGI_ADAPTER_DESC adapter_desc {};
    const auto status = adapter->GetDesc(&adapter_desc);
    if (FAILED(status)) {
      BOOST_LOG(error) << "Failed to query the display adapter description [0x"sv
                       << util::hex(status).to_string_view() << ']';
      return false;
    }

    if (adapter_desc.VendorId != 0x10de) {
      BOOST_LOG(error) << "Sunshine 3D requires an NVIDIA display adapter; detected vendor ID "
                       << util::hex(adapter_desc.VendorId).to_string_view();
      return false;
    }
    return boost::algorithm::ends_with(name, "_nvenc");
  }

  native_encode_owner_t display_vram_t::make_nvenc_encode_device(
    const ::video::config_t &client_config,
    const ::video::sunshine_colorspace_t &colorspace
  ) {
    return d3d_nvenc_encode_device_t::create(shared_from_this(), adapter.get(), client_config, colorspace);
  }

  int init() {
    BOOST_LOG(info) << "Loading Windows display shaders..."sv;
    namespace cache = models::host_sbs_shader_cache;

#define DISPLAY_SHADER_TARGET(stem, stage) std::addressof(stem##_hlsl),
    const std::array targets {WINDOWS_DISPLAY_SHADERS(DISPLAY_SHADER_TARGET)};
#undef DISPLAY_SHADER_TARGET
    static_assert(targets.size() == display_shader_specs.size());

    std::size_t embedded_count = 0u;
#ifdef NDEBUG
    const auto display_sources = cache::snapshot_sources(SUNSHINE_SHADERS_DIR, display_shader_specs);
#endif
    for (std::size_t index = 0u; index < targets.size(); ++index) {
      const auto &spec = display_shader_specs[index];
      blob_t bytecode;
#ifdef NDEBUG
      const auto embedded = cache::get_embedded(display_sources, spec);
      if (embedded && SUCCEEDED(D3DCreateBlob(embedded->size(), &bytecode))) {
        std::memcpy(bytecode->GetBufferPointer(), embedded->data(), embedded->size());
        ++embedded_count;
      }
#endif
      if (!bytecode) {
        const auto path = SUNSHINE_SHADERS_DIR "/" + std::string {spec.filename};
        bytecode = compile_shader(path.c_str(), spec.entrypoint.data(), spec.target.data());
        // The tiled P010 chroma program is optional: its encoder falls back to raster chroma.
        if (!bytecode && spec.filename != "p010_uv_tile_cs.hlsl"sv) {
          return -1;
        }
      }
      *targets[index] = std::move(bytecode);
    }

    sbs_reprojection_v2_live_ps_hlsl.reset();
    sbs_reprojection_v2_live_source_closure_sha256.clear();
    sbs_reprojection_v2_p010_y_ps_hlsl.reset();
    sbs_reprojection_v2_p010_y_source_closure_sha256.clear();
    sbs_flat_identity_ps_hlsl.reset();
    sbs_reprojection_vs_hlsl.reset();

    {
      // Host AI remains available independently of the local-AR provider selection. Game
      // converters never instantiate this renderer, but a later Host AI session may need it.
      const auto renderer_sources = cache::snapshot_sources(
        SUNSHINE_SHADERS_DIR,
        cache::parallax_v2_live_renderer_specs
      );
      sbs_reprojection_v2_live_source_closure_sha256 =
        cache::source_closure_sha256(renderer_sources);
      const bool renderer_authenticated = renderer_sources &&
                                          sbs_reprojection_v2_live_source_closure_sha256 ==
                                            cache::parallax_v2_live_renderer_source_closure_sha256;
      if (renderer_authenticated) {
        sbs_reprojection_v2_live_ps_hlsl = cache::get(
          renderer_sources,
          cache::parallax_v2_live_renderer
        );
        sbs_reprojection_vs_hlsl = cache::get(
          renderer_sources,
          cache::sbs_reprojection_vertex
        );
      }
      if (!sbs_reprojection_v2_live_ps_hlsl) {
        BOOST_LOG(warning)
          << "Host SBS V2 renderer authentication or compilation failed; Host SBS will stream "sv
             "current-frame flat identity (observed closure "sv
          << sbs_reprojection_v2_live_source_closure_sha256 << ", expected "sv
          << cache::parallax_v2_live_renderer_source_closure_sha256 << ")."sv;
      }
      const auto p010_y_sources = cache::snapshot_sources(
        SUNSHINE_SHADERS_DIR,
        cache::parallax_v2_p010_y_specs
      );
      sbs_reprojection_v2_p010_y_source_closure_sha256 =
        cache::source_closure_sha256(p010_y_sources);
      const bool p010_y_authenticated =
        p010_y_sources &&
        sbs_reprojection_v2_p010_y_source_closure_sha256 ==
          cache::parallax_v2_p010_y_source_closure_sha256;
      if (p010_y_authenticated) {
        sbs_reprojection_v2_p010_y_ps_hlsl = cache::get(
          p010_y_sources,
          cache::parallax_v2_p010_y_renderer
        );
      }
      if (!sbs_reprojection_v2_p010_y_ps_hlsl) {
        BOOST_LOG(info)
          << "Host SBS direct P010 luma optimization is unavailable; retaining the established "sv
             "RGB-to-P010 path (observed closure "sv
          << sbs_reprojection_v2_p010_y_source_closure_sha256 << ", expected "sv
          << cache::parallax_v2_p010_y_source_closure_sha256 << ")."sv;
      }
    }

    // This independent current-frame identity renderer is the safety net for Host SBS. Its
    // failure must not disable ordinary non-SBS video; a Host SBS device will reject creation
    // only if neither this shader nor the authenticated V2 renderer can be created. It shares
    // the pinned vertex shader through its own closure so an edited flat renderer (or vertex
    // shader) cannot silently alter stereo geometry.
    const auto fallback_sources = cache::snapshot_sources(
      SUNSHINE_SHADERS_DIR,
      cache::sbs_flat_fallback_specs
    );
    const auto fallback_closure = cache::source_closure_sha256(fallback_sources);
    const bool fallback_authenticated = fallback_sources &&
                                        fallback_closure ==
                                          cache::sbs_flat_fallback_source_closure_sha256;
    if (fallback_authenticated) {
      sbs_flat_identity_ps_hlsl = cache::get(fallback_sources, cache::sbs_flat_identity);
      if (!sbs_reprojection_vs_hlsl) {
        sbs_reprojection_vs_hlsl = cache::get(
          fallback_sources,
          cache::sbs_reprojection_vertex
        );
      }
    }
    if (!sbs_flat_identity_ps_hlsl) {
      BOOST_LOG(warning)
        << "Host SBS flat-identity authentication or compilation failed; ordinary video "sv
           "remains enabled (observed closure "sv
        << fallback_closure << ", expected "sv
        << cache::sbs_flat_fallback_source_closure_sha256 << ")."sv;
    }
    if (!sbs_reprojection_vs_hlsl) {
      // Both Host SBS closures failed. Non-SBS presentation still needs the fullscreen-triangle
      // vertex shader, so compile it unpinned; Host SBS device creation will reject separately
      // because neither pinned pixel shader is available.
      const auto vertex_only_sources = cache::snapshot_sources(
        SUNSHINE_SHADERS_DIR,
        std::array {cache::sbs_reprojection_vertex}
      );
      if (vertex_only_sources) {
        sbs_reprojection_vs_hlsl = cache::get(
          vertex_only_sources,
          cache::sbs_reprojection_vertex
        );
      }
      if (!sbs_reprojection_vs_hlsl) {
        BOOST_LOG(error) << "Failed to compile the shared fullscreen vertex shader"sv;
        return -1;
      }
    }
    BOOST_LOG(info) << "Windows display shaders ready (embedded="sv << embedded_count
                    << ", runtime_compiled="sv << display_shader_specs.size() - embedded_count << ")."sv;

    return 0;
  }
}  // namespace platf::dxgi
