/**
 * @file src/platform/windows/display_vram.cpp
 * @brief Definitions for handling video ram.
 */
// standard includes
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string_view>
#include <thread>

// platform includes
#include <d3dcompiler.h>
#include <DirectXMath.h>

// lib includes
#include <boost/algorithm/string/predicate.hpp>

// local includes
#include "display.h"
#include "display_config.h"
#include "foreground_window_region.h"
#include "host_sbs_v2_renderer.h"
#include "misc.h"
#include "sbs_debug_dump.h"
#include "video_dom_client.h"
#include "src/config.h"
#include "src/depth_coordinate_v2.h"
#include "src/generated/sbs_adaptive_state_contract.h"
#include "src/host_sbs_resolution.h"
#include "src/host_sbs_shader_cache.h"
#include "src/host_sbs_v2_geometry.h"
#include "src/logging.h"
#include "src/model_manager.h"
#include "src/nvenc/nvenc_config.h"
#include "src/nvenc/nvenc_d3d11_native.h"
#include "src/nvenc/nvenc_utils.h"
#include "src/sbs_perf.h"
#include "src/video.h"
#include "src/video_depth_estimator.h"

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
    ).count();
    if (microseconds < 0 ||
        static_cast<std::uint64_t>(microseconds) ==
          std::numeric_limits<std::uint64_t>::max()) {
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
    float padding;
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

  blob_t convert_yuv420_packed_uv_type0_ps_hlsl;
  blob_t convert_yuv420_packed_uv_type0_ps_linear_hlsl;
  blob_t convert_yuv420_packed_uv_type0_ps_perceptual_quantizer_hlsl;
  blob_t convert_yuv420_packed_uv_type0_vs_hlsl;
  blob_t convert_yuv420_packed_uv_type0s_ps_hlsl;
  blob_t convert_yuv420_packed_uv_type0s_ps_linear_hlsl;
  blob_t convert_yuv420_packed_uv_type0s_ps_perceptual_quantizer_hlsl;
  blob_t convert_yuv420_packed_uv_type0s_vs_hlsl;
  blob_t convert_yuv420_planar_y_ps_hlsl;
  blob_t convert_yuv420_planar_y_ps_linear_hlsl;
  blob_t convert_yuv420_planar_y_ps_perceptual_quantizer_hlsl;
  blob_t convert_yuv420_planar_y_vs_hlsl;
  blob_t cursor_ps_hlsl;
  blob_t cursor_ps_normalize_white_hlsl;
  blob_t rgb_present_linear_to_srgb_ps_hlsl;
  blob_t rgb_present_srgb_to_linear_ps_hlsl;
  blob_t cursor_vs_hlsl;
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

  struct img_d3d_t: public platf::img_t {
    // These objects are owned by the display_t's ID3D11Device
    texture2d_t capture_texture;
    render_target_t capture_rt;
    keyed_mutex_t capture_mutex;

    // Shared handle used by an encode device to open capture_texture.
    HANDLE encoder_texture_handle = {};

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

    virtual ~img_d3d_t() override {
      if (encoder_texture_handle) {
        CloseHandle(encoder_texture_handle);
      }
    };
  };

  struct texture_lock_helper {
    keyed_mutex_t _mutex;
    bool _locked = false;

    texture_lock_helper(const texture_lock_helper &) = delete;
    texture_lock_helper &operator=(const texture_lock_helper &) = delete;

    texture_lock_helper(texture_lock_helper &&other) {
      _mutex.reset(other._mutex.release());
      _locked = other._locked;
      other._locked = false;
    }

    texture_lock_helper &operator=(texture_lock_helper &&other) {
      if (_locked) {
        _mutex->ReleaseSync(0);
      }
      _mutex.reset(other._mutex.release());
      _locked = other._locked;
      other._locked = false;
      return *this;
    }

    texture_lock_helper(IDXGIKeyedMutex *mutex):
        _mutex(mutex) {
      if (_mutex) {
        _mutex->AddRef();
      }
    }

    ~texture_lock_helper() {
      if (_locked) {
        _mutex->ReleaseSync(0);
      }
    }

    bool lock() {
      if (_locked) {
        return true;
      }
      HRESULT status = _mutex->AcquireSync(0, INFINITE);
      if (status == S_OK) {
        _locked = true;
      } else {
        BOOST_LOG(error) << "Failed to acquire texture mutex [0x"sv << util::hex(status).to_string_view() << ']';
      }
      return _locked;
    }
  };

  util::buffer_t<std::uint8_t> make_cursor_xor_image(const util::buffer_t<std::uint8_t> &img_data, DXGI_OUTDUPL_POINTER_SHAPE_INFO shape_info) {
    constexpr std::uint32_t inverted = 0xFFFFFFFF;
    constexpr std::uint32_t transparent = 0;

    switch (shape_info.Type) {
      case DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR:
        // This type doesn't require any XOR-blending
        return {};
      case DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MASKED_COLOR:
        {
          util::buffer_t<std::uint8_t> cursor_img = img_data;
          std::for_each((std::uint32_t *) std::begin(cursor_img), (std::uint32_t *) std::end(cursor_img), [](auto &pixel) {
            auto alpha = (std::uint8_t) ((pixel >> 24) & 0xFF);
            if (alpha == 0xFF) {
              // Pixels with 0xFF alpha will be XOR-blended as is.
            } else if (alpha == 0x00) {
              // Pixels with 0x00 alpha will be blended by make_cursor_alpha_image().
              // We make them transparent for the XOR-blended cursor image.
              pixel = transparent;
            } else {
              // Other alpha values are illegal in masked color cursors
              BOOST_LOG(warning) << "Illegal alpha value in masked color cursor: " << alpha;
            }
          });
          return cursor_img;
        }
      case DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MONOCHROME:
        // Monochrome is handled below
        break;
      default:
        BOOST_LOG(error) << "Invalid cursor shape type: " << shape_info.Type;
        return {};
    }

    shape_info.Height /= 2;

    util::buffer_t<std::uint8_t> cursor_img {shape_info.Width * shape_info.Height * 4};

    auto bytes = shape_info.Pitch * shape_info.Height;
    auto pixel_begin = (std::uint32_t *) std::begin(cursor_img);
    auto pixel_data = pixel_begin;
    auto and_mask = std::begin(img_data);
    auto xor_mask = std::begin(img_data) + bytes;

    for (auto x = 0; x < bytes; ++x) {
      for (auto c = 7; c >= 0 && ((std::uint8_t *) pixel_data) != std::end(cursor_img); --c) {
        auto bit = 1 << c;
        auto color_type = ((*and_mask & bit) ? 1 : 0) + ((*xor_mask & bit) ? 2 : 0);

        switch (color_type) {
          case 0:  // Opaque black (handled by alpha-blending)
          case 2:  // Opaque white (handled by alpha-blending)
          case 1:  // Color of screen (transparent)
            *pixel_data = transparent;
            break;
          case 3:  // Inverse of screen
            *pixel_data = inverted;
            break;
        }

        ++pixel_data;
      }
      ++and_mask;
      ++xor_mask;
    }

    return cursor_img;
  }

  util::buffer_t<std::uint8_t> make_cursor_alpha_image(const util::buffer_t<std::uint8_t> &img_data, DXGI_OUTDUPL_POINTER_SHAPE_INFO shape_info) {
    constexpr std::uint32_t black = 0xFF000000;
    constexpr std::uint32_t white = 0xFFFFFFFF;
    constexpr std::uint32_t transparent = 0;

    switch (shape_info.Type) {
      case DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MASKED_COLOR:
        {
          util::buffer_t<std::uint8_t> cursor_img = img_data;
          std::for_each((std::uint32_t *) std::begin(cursor_img), (std::uint32_t *) std::end(cursor_img), [](auto &pixel) {
            auto alpha = (std::uint8_t) ((pixel >> 24) & 0xFF);
            if (alpha == 0xFF) {
              // Pixels with 0xFF alpha will be XOR-blended by make_cursor_xor_image().
              // We make them transparent for the alpha-blended cursor image.
              pixel = transparent;
            } else if (alpha == 0x00) {
              // Pixels with 0x00 alpha will be blended as opaque with the alpha-blended image.
              pixel |= 0xFF000000;
            } else {
              // Other alpha values are illegal in masked color cursors
              BOOST_LOG(warning) << "Illegal alpha value in masked color cursor: " << alpha;
            }
          });
          return cursor_img;
        }
      case DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR:
        // Color cursors are just an ARGB bitmap which requires no processing.
        return img_data;
      case DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MONOCHROME:
        // Monochrome cursors are handled below.
        break;
      default:
        BOOST_LOG(error) << "Invalid cursor shape type: " << shape_info.Type;
        return {};
    }

    shape_info.Height /= 2;

    util::buffer_t<std::uint8_t> cursor_img {shape_info.Width * shape_info.Height * 4};

    auto bytes = shape_info.Pitch * shape_info.Height;
    auto pixel_begin = (std::uint32_t *) std::begin(cursor_img);
    auto pixel_data = pixel_begin;
    auto and_mask = std::begin(img_data);
    auto xor_mask = std::begin(img_data) + bytes;

    for (auto x = 0; x < bytes; ++x) {
      for (auto c = 7; c >= 0 && ((std::uint8_t *) pixel_data) != std::end(cursor_img); --c) {
        auto bit = 1 << c;
        auto color_type = ((*and_mask & bit) ? 1 : 0) + ((*xor_mask & bit) ? 2 : 0);

        switch (color_type) {
          case 0:  // Opaque black
            *pixel_data = black;
            break;
          case 2:  // Opaque white
            *pixel_data = white;
            break;
          case 3:  // Inverse of screen (handled by XOR blending)
          case 1:  // Color of screen (transparent)
            *pixel_data = transparent;
            break;
        }

        ++pixel_data;
      }
      ++and_mask;
      ++xor_mask;
    }

    return cursor_img;
  }

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

  blob_t compile_compute_shader(LPCSTR file) {
    return compile_shader(file, "main", "cs_5_0");
  }

  using depth_estimator_build_task_t =
    std::packaged_task<std::unique_ptr<models::video_depth_estimator>()>;

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

  // CUDA/TensorRT teardown can wait for device quiescence and return a large execution context to
  // the pool. Never perform that work on the encode/local-presenter thread.
  void retire_depth_estimator(std::unique_ptr<models::video_depth_estimator> estimator) {
    if (!estimator) {
      return;
    }
    std::packaged_task<void()> worker([estimator = std::move(estimator)]() mutable {
      estimator.reset();
    });
    ::video::launch_async_teardown_worker(std::move(worker));
  }

  // A ready-but-unconsumed future owns its estimator result. Move that shared state to a worker so
  // abandoning a local/encode device never runs the estimator destructor on its owner thread.
  void retire_depth_estimator_build(
    std::future<std::unique_ptr<models::video_depth_estimator>> build
  ) {
    if (!build.valid()) {
      return;
    }
    std::packaged_task<void()> worker([build = std::move(build)]() mutable {
      try {
        auto estimator = build.get();
        estimator.reset();
      } catch (const std::exception &e) {
        BOOST_LOG(warning) << "Retired Host SBS pipeline build failed: "sv << e.what();
      } catch (...) {
        BOOST_LOG(warning) << "Retired Host SBS pipeline build failed with an unknown exception."sv;
      }
    });
    ::video::launch_async_teardown_worker(std::move(worker));
  }

  class d3d_base_encode_device final {
  public:
    ~d3d_base_encode_device() {
      // The encode session is already leaving the live path, so a short bounded drain cannot
      // stall capture. It preserves the final GPU timing samples; the generation token only
      // rejects results that cross an explicit collector reset (used by the offline harness).
      drain_sbs_gpu_timers();

      // The background task owns D3D references and its packaged-task future does not block here.
      // If this device is torn down while construction is in flight, the process-lifetime tracked
      // worker service lets it finish independently and destroys the unused result there.
      retire_depth_estimator(std::move(depth_estimator));
      retire_depth_estimator_build(std::move(depth_estimator_build));
    }

    void release_capture_display() noexcept {
      // Encoder teardown may remain blocked inside the NVIDIA driver. The shared capture-display
      // owner is not a D3D/NVENC operand, so release it before derived member destruction to let
      // Desktop Duplication reinitialize while device/context resources remain alive.
      display.reset();
    }

    int convert_rgb(
      platf::img_t &img,
      ID3D11Texture2D *target_texture,
      ID3D11RenderTargetView *target,
      bool target_is_linear
    ) {
      rgb_present_texture = target_texture;
      rgb_present_target = target;
      rgb_present_target_is_linear = target_is_linear;
      auto clear_target = util::fail_guard([this]() {
        rgb_present_texture = nullptr;
        rgb_present_target = nullptr;
        rgb_present_target_is_linear = false;
      });
      return convert(img);
    }

    bool needs_conversion_poll() const {
      return depth_completion_poll_pending || depth_authority_reprocess_pending ||
             gpu_observation_barrier.active() ||
             sbs_dumper.needs_conversion_poll();
    }

    std::optional<std::chrono::steady_clock::time_point> rendered_content_timestamp() const {
      return rendered_content_timestamp_;
    }

    int convert(
      platf::img_t &img_base,
      const std::optional<std::chrono::steady_clock::time_point> next_encode_target =
        std::nullopt
    ) {
      auto &img = (img_d3d_t &) img_base;
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
        // Look up (or create) this image's encoder context. Only when a new id first appears do
        // we garbage-collect contexts whose capture img_t has since expired -- the set of live
        // ids is bounded by the capture pool, so this replaces an every-frame scan with one that
        // runs only on the rare new-image insertion. (Erasing other map nodes leaves `it` valid.)
        auto [it, inserted] = img_ctx_map.try_emplace(img.id);
        if (inserted) {
          for (auto gc = img_ctx_map.begin(); gc != img_ctx_map.end();) {
            if (gc != it && gc->second.img_weak.expired()) {
              gc = img_ctx_map.erase(gc);
            } else {
              ++gc;
            }
          }
        }
        auto &img_ctx = it->second;

        // Open the shared capture texture with our ID3D11Device
        if (initialize_image_context(img, img_ctx)) {
          return -1;
        }

        // Acquire encoder mutex to synchronize with capture code
        detail::keyed_mutex_lock_t encoder_mutex_lock {img_ctx.encoder_mutex.get()};
        auto status = encoder_mutex_lock.lock(0, INFINITE);
        if (status != S_OK) {
          BOOST_LOG(error) << "Failed to acquire encoder mutex [0x"sv << util::hex(status).to_string_view() << ']';
          return -1;
        }

        auto draw = [&](auto &input, const D3D11_VIEWPORT &y_or_yuv_viewport, const D3D11_VIEWPORT &uv_viewport, bool input_is_linear, bool y_already_written = false) {
          device_ctx->PSSetShaderResources(0, 1, &input);
          ID3D11Buffer *converter_buffers[] = {
            color_matrix.get(),
            sdr_color_transform.get(),
          };
          // Bind both converter constants for every shader variant. Other passes use their own
          // PS constant-buffer slots, so never rely on stale context state between SBS/RGB draws.
          device_ctx->PSSetConstantBuffers(0, 2, converter_buffers);

          // The optional HDR Host-SBS MRT has already populated luma while producing the packed
          // RGB raster. Every other path retains the established standalone Y/YUV draw.
          if (!y_already_written) {
            device_ctx->OMSetRenderTargets(1, &out_Y_or_YUV_rtv, nullptr);
            device_ctx->VSSetShader(convert_Y_or_YUV_vs.get(), nullptr, 0);
            device_ctx->PSSetShader(input_is_linear ? convert_Y_or_YUV_fp16_ps.get() : convert_Y_or_YUV_ps.get(), nullptr, 0);
            device_ctx->RSSetViewports(1, &y_or_yuv_viewport);
            device_ctx->Draw(3, 0);
          }

          // Draw UV if needed
          if (out_UV_rtv) {
            assert(format == DXGI_FORMAT_NV12 || format == DXGI_FORMAT_P010);
            device_ctx->OMSetRenderTargets(1, &out_UV_rtv, nullptr);
            device_ctx->VSSetShader(convert_UV_vs.get(), nullptr, 0);
            device_ctx->PSSetShader(input_is_linear ? convert_UV_fp16_ps.get() : convert_UV_ps.get(), nullptr, 0);
            device_ctx->RSSetViewports(1, &uv_viewport);
            device_ctx->Draw(3, 0);
          }
        };

        auto draw_rgb = [&](ID3D11ShaderResourceView *input, bool input_is_linear) {
          device_ctx->OMSetRenderTargets(1, &rgb_present_target, nullptr);
          device_ctx->VSSetShader(sbs_reprojection_vs.get(), nullptr, 0);
          ID3D11PixelShader *presentation_shader = rgb_present_ps.get();
          if (input_is_linear != rgb_present_target_is_linear) {
            presentation_shader = input_is_linear ?
                                    rgb_present_linear_to_srgb_ps.get() :
                                    rgb_present_srgb_to_linear_ps.get();
          }
          device_ctx->PSSetShader(presentation_shader, nullptr, 0);
          if (!input_is_linear && rgb_present_target_is_linear) {
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
          if (!input || !rgb_present_texture) {
            return false;
          }

          D3D11_TEXTURE2D_DESC input_desc {};
          D3D11_TEXTURE2D_DESC output_desc {};
          input->GetDesc(&input_desc);
          rgb_present_texture->GetDesc(&output_desc);
          const bool compatible = input_desc.Width == output_desc.Width &&
                                  input_desc.Height == output_desc.Height &&
                                  input_desc.MipLevels == output_desc.MipLevels &&
                                  input_desc.ArraySize == output_desc.ArraySize &&
                                  input_desc.Format == output_desc.Format &&
                                  input_desc.SampleDesc.Count == output_desc.SampleDesc.Count &&
                                  input_desc.SampleDesc.Quality == output_desc.SampleDesc.Quality &&
                                  input_is_linear == rgb_present_target_is_linear;
          if (!compatible) {
            if (!rgb_copy_fallback_logged) {
              BOOST_LOG(info) << "Local AR exact-copy path unavailable (source "sv
                              << input_desc.Width << 'x' << input_desc.Height << ' '
                              << (int) input_desc.Format << ", target "sv
                              << output_desc.Width << 'x' << output_desc.Height << ' '
                              << (int) output_desc.Format << ", transfer "sv
                              << (input_is_linear ? "linear" : "sRGB") << " -> "sv
                              << (rgb_present_target_is_linear ? "linear" : "sRGB")
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
          device_ctx->CopyResource(rgb_present_texture, input);
          if (!rgb_copy_path_logged) {
            BOOST_LOG(info) << "Local AR exact texture-copy presentation path active."sv;
            rgb_copy_path_logged = true;
          }
          return true;
        };

        if (sbs_mode != ::video::SBS_OFF) {
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

          // Lazy-create the depth estimator on the first SBS frame.
          if (models::host_sbs_renderer_uses_depth_pipeline(host_sbs_renderer)) {
            ensure_depth_estimator();
          } else {
            sbs_dumper.cancel_pending_request();
          }
          // Stable tensor copies are inserted into the same D3D command stream before the current
          // preprocess. Skip whole-frame performance publication for an explicit dump frame so
          // diagnostic work cannot appear as a production regression.
          auto *gpu_timer =
            perf && !snapshot_debug_inputs ? begin_sbs_gpu_timer() : nullptr;

          // Production uses bounded matched pairing: infer asynchronously from a private color
          // slot, then warp only its completion. The bounded DDup reuse exception below can prove
          // either an identical full desktop-content timestamp or an exact ROI untouched by every
          // dirty/move record since the accepted crop. It still fresh-warps current color.
          const auto input_color_space = input_is_linear ?
                                           (display_is_hdr ? models::input_color_space::scrgb_hdr :
                                                                models::input_color_space::linear_sdr) :
                                           models::input_color_space::srgb;
          const auto current_source_timestamp = img_base.frame_timestamp;
          const auto current_content_timestamp = img_base.content_timestamp;
          const auto &current_ddup_damage = img.ddup_damage;
          struct reuse_damage_key_t {
            const detail::ddup_damage_history_t *history;
            std::uint64_t token;
            bool present;
            explicit reuse_damage_key_t(
              const std::optional<detail::ddup_damage_snapshot_t> &damage
            ):
                history {damage ? damage->history.get() : nullptr},
                token {damage ? damage->token : 0u},
                present {damage.has_value()} {}
            bool operator==(const reuse_damage_key_t &) const = default;
          };
          struct reuse_key_t {
            std::optional<std::chrono::steady_clock::time_point> inference_content;
            reuse_damage_key_t inference_damage;
            models::depth_input_region_t input_region;
            std::optional<std::chrono::steady_clock::time_point> current_content;
            reuse_damage_key_t current_damage;
            bool operator==(const reuse_key_t &) const = default;
          };
          using reuse_entry_t = std::pair<
            reuse_key_t, detail::host_sbs_ddup_reuse_proof_e>;
          std::array<std::optional<reuse_entry_t>, 3u> reuse_cache;
          std::size_t next_reuse_cache_entry = 0u;
          const auto memoized_input_reuse_kind = [&](const matched_frame_slot_t &slot) {
            if (!slot.depth_input_region.is_video_region()) {
              return matched_input_reuse_kind(
                slot, current_content_timestamp, current_ddup_damage
              );
            }
            const reuse_key_t key {
              slot.inference_content_timestamp,
              reuse_damage_key_t {slot.inference_ddup_damage},
              slot.depth_input_region,
              current_content_timestamp,
              reuse_damage_key_t {current_ddup_damage},
            };
            for (const auto &entry : reuse_cache) {
              if (entry && entry->first == key) {
                return entry->second;
              }
            }
            const auto kind = matched_input_reuse_kind(
              slot, current_content_timestamp, current_ddup_damage
            );
            reuse_cache[next_reuse_cache_entry] = reuse_entry_t {key, kind};
            next_reuse_cache_entry = (next_reuse_cache_entry + 1u) % reuse_cache.size();
            return kind;
          };
          // Once the per-stream estimator exists, observe foreground continuity on every SBS
          // conversion, not merely when TensorRT can accept a new matched frame. Before then
          // there is no ROI completion/cache to revoke, so synchronous USER32/DWM work has no
          // consumer. A focus transition between accepted inferences still revokes authority.
          live_window_authority_observation_t pre_copy_authority;
          D3D11_TEXTURE2D_DESC live_source_desc {};
          img_ctx.encoder_texture->GetDesc(&live_source_desc);
          if (detail::host_sbs_window_authority_observation_needed(
                depth_estimator != nullptr,
                models::host_sbs_renderer_uses_depth_pipeline(host_sbs_renderer)
              )) {
            pre_copy_authority = observe_live_window_authority(
              live_source_desc,
              current_content_timestamp
            );
          }
          const auto current_root_authority_generation =
            authority_generation(live_window_authority);
          const auto current_region_authority_generation =
            authority_generation(live_foreground_region);
          const auto current_browser_authority_epoch = live_browser_authority_epoch;
          const bool current_interactive_move_size = interactive_move_size_observed;
          const auto same_source = [](
                                     const std::optional<std::chrono::steady_clock::time_point> &left,
                                     const std::optional<std::chrono::steady_clock::time_point> &right
                                   ) {
            // A missing identity is never evidence that two source images are the same.
            return left && right && *left == *right;
          };
          const auto frame_id = ++sbs_frame_sequence;
          ID3D11ShaderResourceView *render_input_srv = img_ctx.encoder_input_res.get();
          matched_frame_slot_t *matched_render_slot = nullptr;
          matched_frame_slot_t *matched_candidate_slot = nullptr;
          models::estimate_result est;
          bool unchanged_content_submission_suppressed = false;
          bool damage_guided_submission_suppressed = false;
          bool cached_current_color_warp = false;
          bool force_fresh_current_color = false;
          bool using_cached_estimate = false;
          bool completion_finalized_before_admission = false;
          bool block_current_submission_after_early_poll = false;
          bool same_frame_poll_missed_for_current_draw = false;
          std::uint64_t early_cleared_barrier_frame_id = 0u;
          std::optional<std::chrono::steady_clock::time_point> force_infer_enqueued_at;
          detail::host_sbs_depth_reuse_authorization_t depth_reuse_authorization;
          auto adaptive_hold_decision =
            detail::host_sbs_adaptive_hold_decision_e::infer;
          std::uint64_t adaptive_gpu_baseline_frame_id = 0u;
          bool adaptive_gpu_opaque_followup = false;
          bool adaptive_gpu_followup_rejection_counted = false;
          detail::host_sbs_gpu_undecided_candidate_t adaptive_gpu_candidate;
          matched_frame_slot_t current_color_reuse_slot;
          const auto release_unknown_completion = [this](
                                                    const std::uint64_t completed_frame_id,
                                                    const matched_frame_slot_t *preserve_slot
                                                  ) {
            const auto error_now = std::chrono::steady_clock::now();
            if (matched_unknown_frame_error_last.time_since_epoch().count() == 0 ||
                error_now - matched_unknown_frame_error_last >= std::chrono::seconds(30)) {
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
            // The completion consumed its inference and no longer owns the source texture.
            // Preserve only a separately enqueued current candidate, if one exists.
            for (auto &slot : matched_frame_slots) {
              if (&slot != preserve_slot) {
                slot.pending = false;
              }
            }
          };

          const auto validate_completed_slot = [&]() {
            if (!matched_render_slot) {
              return false;
            }
            if (
              !est.completed_frame_valid ||
              est.completed_frame_id != matched_render_slot->frame_id ||
              !gpu_transaction_class_matches(est, *matched_render_slot) ||
              !est.input_region.valid() ||
              est.input_region != matched_render_slot->depth_input_region
            ) {
              BOOST_LOG(error)
                << "Host SBS rejected a depth completion whose region or GPU transaction class "sv
                   "does not match its exact buffered color frame; "sv
                   "rendering current-frame identity."sv;
              matched_render_slot = nullptr;
              est = {};
              render_input_srv = img_ctx.encoder_input_res.get();
              matched_presentation_cache.invalidate();
              return false;
            }
            if (
              matched_render_slot->window_region &&
              !window_region_authorized_for_render(*matched_render_slot)
            ) {
              if (matched_render_slot->depth_input_region.is_video_region()) {
                // The completion is still consumed and releases its pending slot, but geometry
                // observed before a focus/move/resize/monitor transition cannot warp newer pixels.
                clear_cached_roi_output();
                matched_render_slot = nullptr;
                est = {};
                render_input_srv = img_ctx.encoder_input_res.get();
                return false;
              }
              // Full-source depth is independent of optional exact-full window provenance.
              matched_render_slot->window_region.reset();
              matched_render_slot->window_region_observer_status = "not-observed";
              matched_render_slot->window_region_mapping_status = "not-mapped";
              matched_render_slot->live_window_authority_generation = 0u;
            }
            return true;
          };

          const auto apply_completed_domain_and_renderer = [&]() {
            if (!matched_render_slot) {
              return false;
            }
            if (est.input_domain_reset) {
              // Cut counters and scene-camera history are domain-scoped. The first ROI/full/
              // transfer completion starts a fresh telemetry sample without pretending the rearm
              // was an editorial cut.
              reset_adaptive_reuse_runtime();
              opaque_gpu_followup_anchor.reset();
              if (force_infer_enqueued_at) {
                // estimate_depth() may retire the old domain and enqueue this frame's known
                // force-infer transaction in one call. Preserve that successful new-domain arm
                // after resetting the completed domain's cadence.
                adaptive_hold_cadence.record_successful_enqueue(
                  current_content_timestamp,
                  *force_infer_enqueued_at
                );
              }
              sbs_telemetry_has_sample = false;
              sbs_telemetry_last_hard_cut_count = 0;
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
                fail_depth_pipeline_flat();
                matched_render_slot = nullptr;
                est = {};
                render_input_srv = img_ctx.encoder_input_res.get();
                BOOST_LOG(error)
                  << "Host SBS parallax-v2 producer reported a terminal CUDA/TensorRT failure; "sv
                     "this stream is now live flat identity with depth submissions disabled."sv;
              }
              return false;
            }

            // Authenticate the first completed V2 field for which the exact buffered color
            // remains. A later invalid frame is handled inside the V2 shader.
            if (host_sbs_renderer == models::host_sbs_renderer_e::awaiting_v2) {
              const bool result_authenticated =
                models::parallax_v2_result_is_authenticated(est);
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
                  << est.parallax_v2_requested_pop_strength << ")."sv;
              } else {
                BOOST_LOG(warning)
                  << "Host SBS V2 authentication failed closed; this stream is permanently "sv
                     "latched to live flat identity (authenticated_result="sv
                  << (result_authenticated ? "true" : "false")
                  << ", producer_active="sv
                  << (est.parallax_v2_producer_active ? "true" : "false")
                  << ", model_shape="sv << est.raw_width << 'x' << est.raw_height
                  << ")."sv;
                fail_depth_pipeline_flat();
                matched_render_slot = nullptr;
                est = {};
                render_input_srv = img_ctx.encoder_input_res.get();
                return false;
              }
            }
            return matched_render_slot != nullptr;
          };

          const auto retain_completed_lineage = [&, this](
                                                  const bool current_reusable_enqueue_pending,
                                                  const bool before_current_admission
                                                ) {
            if (
              !matched_render_slot || !est.completed_frame_valid ||
              using_cached_estimate
            ) {
              return;
            }
            D3D11_TEXTURE2D_DESC completed_source_desc = live_source_desc;
            if (matched_render_slot->texture) {
              matched_render_slot->texture->GetDesc(&completed_source_desc);
            }
            const auto completed_reuse_kind =
              memoized_input_reuse_kind(*matched_render_slot);
            const bool barrier_force_infer_completion_accepted =
              known_force_infer_completion(est, *matched_render_slot) &&
              host_sbs_renderer == models::host_sbs_renderer_e::parallax_v2 &&
              models::parallax_v2_result_is_authenticated(est) &&
              est.completed_frame_id == matched_render_slot->frame_id &&
              est.input_region == matched_render_slot->depth_input_region &&
              est.color_space == matched_render_slot->color_space &&
              matched_route_matches_current(
                *matched_render_slot,
                live_source_desc,
                input_color_space,
                authority_generation(live_window_authority),
                authority_generation(live_foreground_region),
                live_browser_authority_epoch,
                interactive_move_size_observed
              );
            const bool latest_lineage_retained = retain_latest_v2_lineage(
              est,
              *matched_render_slot,
              completed_source_desc,
              live_source_desc,
              input_color_space
            );
            const auto barrier_frame_id =
              gpu_observation_barrier.conditional_frame_id();
            const bool barrier_cleared =
              gpu_observation_barrier.record_known_force_infer_completion(
                est.completed_frame_id,
                barrier_force_infer_completion_accepted
              );
            if (before_current_admission && barrier_cleared) {
              early_cleared_barrier_frame_id = barrier_frame_id;
            }
            const bool completion_reuse_authorized =
              latest_lineage_retained &&
              completed_reuse_kind != detail::host_sbs_ddup_reuse_proof_e::none;
            if (!completion_reuse_authorized && !current_reusable_enqueue_pending) {
              content_reuse_refresh.reset();
            }
          };

          // Retire a ready prior root before deriving this frame's DDup/adaptive proof. Without
          // this phase, a force completion that narrowly misses the bounded same-frame poll makes
          // the following frame compare against the older lineage and can suppress GPU admission
          // indefinitely on slower shapes. This query observes root readiness only; it never maps
          // the GPU-owned reuse/infer decision.
          if (
            depth_estimator &&
            models::host_sbs_renderer_uses_depth_pipeline(host_sbs_renderer)
          ) {
            auto pending_it = std::find_if(
              matched_frame_slots.begin(),
              matched_frame_slots.end(),
              [](const matched_frame_slot_t &slot) {
                return slot.pending;
              }
            );
            if (pending_it != matched_frame_slots.end()) {
              auto polled = depth_estimator->try_finish_pending_depth_nonblocking(
                input_color_space,
                snapshot_debug_inputs
              );
              if (polled.ready) {
                latest_v2_lineage.reset();
                depth_completion_poll_pending = false;
                if (!polled.result.completed_frame_valid) {
                  for (auto &slot : matched_frame_slots) {
                    slot.pending = false;
                  }
                  block_current_submission_after_early_poll = true;
                } else {
                  auto *completed_slot =
                    find_pending_matched_slot(polled.result.completed_frame_id);
                  if (!completed_slot) {
                    release_unknown_completion(polled.result.completed_frame_id, nullptr);
                    block_current_submission_after_early_poll = true;
                  } else {
                    completed_slot->pending = false;
                    if (!completed_slot->depth_input_region.is_video_region()) {
                      depth_authority_reprocess_pending = false;
                    }
                    matched_render_slot = completed_slot;
                    render_input_srv = completed_slot->srv.get();
                    if (known_force_infer_completion(polled.result, *completed_slot)) {
                      sbs_telemetry_last_sampled_frame_id =
                        polled.result.completed_frame_id;
                    }
                    est = std::move(polled.result);
                    if (diagnostics_enabled) {
                      const double age_ms =
                        std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - completed_slot->captured_at
                        )
                          .count();
                      matched_stats_age_sum_ms += age_ms;
                      matched_stats_age_max_ms = std::max(matched_stats_age_max_ms, age_ms);
                      ++matched_stats_completions;
                      if (perf) {
                        sbs_perf::add_sample_ms("matched_frame_age", age_ms);
                      }
                    }
                    if (
                      validate_completed_slot() &&
                      apply_completed_domain_and_renderer()
                    ) {
                      retain_completed_lineage(false, true);
                      completion_finalized_before_admission = true;
                    } else {
                      block_current_submission_after_early_poll = true;
                    }
                  }
                }
              }
            }
          }
          const bool producer_terminal =
            depth_estimator && depth_estimator->has_terminal_failure();
          const bool dedup_gate_open =
            depth_estimator && models::host_sbs_renderer_uses_depth_pipeline(host_sbs_renderer) &&
            current_content_timestamp && !snapshot_debug_inputs &&
            !depth_authority_reprocess_pending && !producer_terminal;
          const bool cache_reuse_gate_open =
            dedup_gate_open && !gpu_observation_barrier.active();
          const auto latest_v2_route_matches = latest_v2_lineage_route_matches_current(
            live_source_desc,
            input_color_space,
            current_root_authority_generation,
            current_region_authority_generation,
            current_browser_authority_epoch,
            current_interactive_move_size
          );
          const bool matched_inference_pending_at_entry = std::any_of(
            matched_frame_slots.begin(),
            matched_frame_slots.end(),
            [](const matched_frame_slot_t &slot) {
              return slot.pending;
            }
          );
          const auto cached_reuse_kind = latest_v2_route_matches ?
                                           memoized_input_reuse_kind(
                                             latest_v2_lineage.slot
                                           ) :
                                           detail::host_sbs_ddup_reuse_proof_e::none;
          const auto reuse_now = std::chrono::steady_clock::now();
          const bool adaptive_refresh_required_at_entry =
            adaptive_hold_cadence.refresh_required();
          const bool adaptive_route_observable =
            !gpu_observation_barrier.active() &&
            dedup_gate_open && latest_v2_lineage.authenticated &&
            latest_v2_route_matches && current_ddup_damage &&
            !current_interactive_move_size;
          const bool motion_damage_comparable =
            dedup_gate_open && latest_v2_lineage.authenticated &&
            !matched_inference_pending_at_entry && latest_v2_route_matches &&
            cached_reuse_kind == detail::host_sbs_ddup_reuse_proof_e::none;
          const auto cached_motion_damage =
            motion_damage_comparable && adaptive_route_observable ?
              matched_motion_damage(
                latest_v2_lineage.slot,
                current_ddup_damage
              ) :
              std::optional<matched_motion_damage_t> {};
          const bool content_refresh_due = content_reuse_refresh.refresh_due(reuse_now);
          depth_reuse_authorization = detail::select_host_sbs_cached_depth_reuse(
            content_refresh_due || !cache_reuse_gate_open ?
              detail::host_sbs_ddup_reuse_proof_e::none : cached_reuse_kind,
            latest_v2_lineage.slot.frame_id,
            frame_id
          );
          const auto current_adaptive_route_epoch = [&]() {
            return detail::host_sbs_adaptive_motion_route_epoch_t {
                .source_width = live_source_desc.Width,
                .source_height = live_source_desc.Height,
                .mip_levels = live_source_desc.MipLevels,
                .array_size = live_source_desc.ArraySize,
                .source_format = static_cast<std::uint32_t>(live_source_desc.Format),
                .sample_count = live_source_desc.SampleDesc.Count,
                .sample_quality = live_source_desc.SampleDesc.Quality,
                .input_color_space = static_cast<std::uint32_t>(input_color_space),
                .root_authority_generation = authority_generation(live_window_authority),
                .region_authority_generation = authority_generation(live_foreground_region),
                .browser_authority_epoch = live_browser_authority_epoch,
                .interactive_move_size = interactive_move_size_observed,
              };
          };
          if (adaptive_motion_route_state.observe(current_adaptive_route_epoch())) {
            // Track route epochs independently of cache authentication. A focus/ROI/browser,
            // source-signature, transfer-domain, or interactive transition must not let an
            // adaptive candidate armed on the old route combine with the new route's completion.
            if (diagnostics_enabled && opaque_gpu_followup_anchor.authenticated) {
              ++matched_stats_gpu_followup_host_rejected;
            }
            revoke_adaptive_reuse();
          }
          const bool opaque_followup_route_matches =
            gpu_observation_barrier.active() && dedup_gate_open &&
            opaque_gpu_followup_anchor.route_matches(
              live_source_desc,
              input_color_space,
              current_root_authority_generation,
              current_region_authority_generation,
              current_browser_authority_epoch,
              current_interactive_move_size
            ) &&
            current_ddup_damage;
          const bool opaque_followup_route_observable =
            opaque_followup_route_matches && !matched_inference_pending_at_entry;
          const bool opaque_followup_route_rejected =
            opaque_gpu_followup_anchor.authenticated &&
            !opaque_followup_route_matches;
          if (
            !dedup_gate_open || !current_ddup_damage || current_interactive_move_size ||
              (latest_v2_lineage.authenticated && !latest_v2_route_matches) ||
              opaque_followup_route_rejected)
          {
            // Missing DDup proof, route/authority changes, dump/reprocess, terminal failure, and
            // native move/size all revoke predictive state even before a cache is authenticated.
            if (diagnostics_enabled && opaque_followup_route_rejected) {
              ++matched_stats_gpu_followup_host_rejected;
            }
            revoke_adaptive_reuse();
          }
          if (
            opaque_followup_route_observable &&
            opaque_gpu_followup_anchor.authenticated
          ) {
            if (!detail::host_sbs_gpu_followup_fresh(
                  opaque_gpu_followup_anchor.frame_id,
                  gpu_observation_barrier.conditional_frame_id(),
                  opaque_gpu_followup_anchor.enqueued_at,
                  reuse_now
                )) {
              if (diagnostics_enabled) {
                ++matched_stats_gpu_followup_expired;
              }
              revoke_adaptive_reuse();
            } else {
              const auto followup_damage = matched_motion_damage(
                opaque_gpu_followup_anchor,
                current_ddup_damage
              );
              const bool followup_proof_eligible =
                followup_damage && followup_damage->coverage.known &&
                current_content_timestamp != opaque_gpu_followup_anchor.content_timestamp &&
                detail::host_sbs_adaptive_motion_damage_candidate(
                  followup_damage->coverage
                ) &&
                opaque_gpu_followup_anchor.damage && current_ddup_damage &&
                opaque_gpu_followup_anchor.damage->history == current_ddup_damage->history &&
                current_ddup_damage->token > opaque_gpu_followup_anchor.damage->token;
              if (followup_proof_eligible) {
                adaptive_gpu_baseline_frame_id = opaque_gpu_followup_anchor.frame_id;
                adaptive_gpu_opaque_followup = true;
                adaptive_gpu_candidate = {
                  .baseline_frame_id = opaque_gpu_followup_anchor.frame_id,
                  .current_frame_id = frame_id,
                  .damage_history = current_ddup_damage->history.get(),
                  .baseline_damage_token = opaque_gpu_followup_anchor.damage->token,
                  .current_damage_token = current_ddup_damage->token,
                  .damage_history_complete = true,
                  .opaque_followup = true,
                };
              } else {
                if (diagnostics_enabled) {
                  ++matched_stats_gpu_followup_host_rejected;
                }
                // An incomplete successor cannot be retried as a later follow-up; the
                // immediately-prior transaction contract now falls back to CPU-known inference.
                revoke_adaptive_reuse();
              }
            }
          }
          if (
            adaptive_route_observable && !matched_inference_pending_at_entry &&
            cached_reuse_kind == detail::host_sbs_ddup_reuse_proof_e::none
          ) {
            const auto &damage = cached_motion_damage;
            if (!damage || !damage->coverage.known) {
              // A matched route without a complete retained sequence is a discontinuity, not a
              // merely ineligible candidate. Do not let an old adaptive arm bridge the
              // force-infer transaction that repairs the baseline.
              revoke_adaptive_reuse();
            } else {
              const bool complete_damage_history =
                detail::host_sbs_adaptive_motion_damage_candidate(damage->coverage);
              // DDup proves only continuity and route attribution. Localized and broad changes
              // both reach the authenticated device comparison; similarity remains GPU-owned.
              const bool gpu_undecided_candidate =
                !gpu_observation_barrier.active() && complete_damage_history &&
                detail::host_sbs_approximate_reuse_provider_allowed(
                  approximate_reuse_since_enqueue,
                  detail::host_sbs_approximate_reuse_provider_e::gpu_undecided,
                  adaptive_refresh_required_at_entry
                );
              adaptive_hold_decision = adaptive_hold_cadence.observe_changed(
                current_content_timestamp,
                gpu_undecided_candidate,
                reuse_now
              );
              if (
                adaptive_hold_decision ==
                detail::host_sbs_adaptive_hold_decision_e::hold_candidate
              ) {
                if (
                  latest_v2_lineage.slot.inference_ddup_damage &&
                  current_ddup_damage &&
                  latest_v2_lineage.slot.inference_ddup_damage->history ==
                    current_ddup_damage->history &&
                  latest_v2_lineage.slot.inference_ddup_damage->token != 0u &&
                  current_ddup_damage->token >
                    latest_v2_lineage.slot.inference_ddup_damage->token
                ) {
                  // The baseline and DDup tuple are host-owned prefilter evidence only. The exact
                  // current-vs-history comparison and final branch remain on the GPU.
                  adaptive_gpu_baseline_frame_id =
                    latest_v2_lineage.slot.frame_id;
                  adaptive_gpu_candidate = {
                    .baseline_frame_id = latest_v2_lineage.slot.frame_id,
                    .current_frame_id = frame_id,
                    .damage_history = current_ddup_damage->history.get(),
                    .baseline_damage_token =
                      latest_v2_lineage.slot.inference_ddup_damage->token,
                    .current_damage_token = current_ddup_damage->token,
                    .damage_history_complete = complete_damage_history,
                  };
                }
              }
            }
          }
          if (detail::host_sbs_latest_v2_lineage_reset_required(
                latest_v2_lineage.authenticated,
                latest_v2_route_matches,
                snapshot_debug_inputs,
                depth_authority_reprocess_pending,
                producer_terminal
              )) {
            latest_v2_lineage.reset();
            // Route/dump/reprocess/terminal revocation invalidates the cached resource aliases
            // regardless of any current-frame proof computed above.
          }
          const bool cached_geometry_matches =
            latest_v2_lineage.authenticated && depth_reuse_authorization.valid();
          const auto host_depth_admission = detail::host_sbs_depth_admission(
            cached_geometry_matches,
            adaptive_gpu_candidate.valid(),
            gpu_observation_barrier.active(),
            adaptive_gpu_opaque_followup
          );
          const bool cached_geometry_render_allowed =
            detail::host_sbs_cached_geometry_render_allowed(
              cache_reuse_gate_open && latest_v2_route_matches,
              host_sbs_renderer == models::host_sbs_renderer_e::parallax_v2,
              depth_reuse_authorization,
              latest_v2_lineage.slot.frame_id,
              frame_id,
              latest_v2_lineage.estimate.shadow_final_parallax &&
                latest_v2_lineage.estimate.shadow_state
            );
          if (depth_estimator && models::host_sbs_renderer_uses_depth_pipeline(host_sbs_renderer)) {
            matched_frame_slot_t *retained_source_pending_slot = nullptr;
            matched_frame_slot_t *unchanged_input_pending_slot = nullptr;
            auto pending_reuse_kind = detail::host_sbs_ddup_reuse_proof_e::none;
            matched_frame_slot_t *any_pending_slot = nullptr;
            for (auto &slot : matched_frame_slots) {
              if (slot.pending && !any_pending_slot) {
                any_pending_slot = &slot;
              }
              if (slot.pending && same_source(slot.source_timestamp, current_source_timestamp)) {
                retained_source_pending_slot = &slot;
                break;
              }
              if (dedup_gate_open && slot.pending && matched_route_matches_current(
                    slot,
                    live_source_desc,
                    input_color_space,
                    current_root_authority_generation,
                    current_region_authority_generation,
                    current_browser_authority_epoch,
                    current_interactive_move_size
                  )) {
                const auto reuse_kind = memoized_input_reuse_kind(slot);
                if (reuse_kind != detail::host_sbs_ddup_reuse_proof_e::none) {
                  unchanged_input_pending_slot = &slot;
                  pending_reuse_kind = reuse_kind;
                }
              }
            }

            // The encode loop calls convert() on a retained source when accepted inference is
            // still pending. The early nonblocking poll above already found this exact unit busy,
            // so keep its slot and poll owner intact and do not enqueue duplicate work. Presentation
            // remains fail-open through the bounded packed-output hold (or current flat output once
            // that hold expires); a later convert() adopts the exact completion when its event is
            // ready. Live capture must never turn this retry into an unbounded CUDA stream wait.
            if (retained_source_pending_slot) {
              mark_sbs_matched_copy_end(gpu_timer, false);
            } else if (unchanged_input_pending_slot) {
              // Never enqueue an input already represented by the sole pending inference. This is
              // either a DDup content-clock duplicate or a changed desktop whose entire damage
              // history stayed outside the exact accepted ROI. Poll once without waiting so a
              // ready completion can immediately drive a fresh current-color warp.
              mark_sbs_matched_copy_end(gpu_timer, false);
              unchanged_content_submission_suppressed = true;
              depth_reuse_authorization =
                detail::select_host_sbs_cached_depth_reuse(
                  pending_reuse_kind,
                  unchanged_input_pending_slot->frame_id,
                  frame_id
                );
              damage_guided_submission_suppressed =
                pending_reuse_kind == detail::host_sbs_ddup_reuse_proof_e::roi_damage;
              force_fresh_current_color = cached_geometry_render_allowed;
              auto polled = depth_estimator->try_finish_pending_depth_nonblocking(
                input_color_space,
                false
              );
              if (polled.ready) {
                // A ready poll either normalized a completion or latched a producer failure. Do
                // not retain resource aliases authenticated before that state transition.
                latest_v2_lineage.reset();
                depth_completion_poll_pending = false;
                if (!polled.result.completed_frame_valid) {
                  // The verified slot represented the estimator's sole pending inference. A ready
                  // empty/failure result has consumed or invalidated that ownership too; retaining
                  // the bit would suppress every later submission indefinitely.
                  unchanged_input_pending_slot->pending = false;
                }
                if (polled.result.completed_frame_valid) {
                  auto *completed_slot =
                    find_pending_matched_slot(polled.result.completed_frame_id);
                  if (completed_slot == unchanged_input_pending_slot) {
                    completed_slot->pending = false;
                    if (!completed_slot->depth_input_region.is_video_region()) {
                      depth_authority_reprocess_pending = false;
                    }
                    matched_render_slot = completed_slot;
                    render_input_srv = completed_slot->srv.get();
                    if (known_force_infer_completion(polled.result, *completed_slot)) {
                      sbs_telemetry_last_sampled_frame_id =
                        polled.result.completed_frame_id;
                    }
                    const auto completed_reuse_kind =
                      memoized_input_reuse_kind(*completed_slot);
                    force_fresh_current_color =
                      completed_reuse_kind != detail::host_sbs_ddup_reuse_proof_e::none;
                    est = std::move(polled.result);
                    if (diagnostics_enabled) {
                      const double age_ms =
                        std::chrono::duration<double, std::milli>(
                          reuse_now - completed_slot->captured_at
                        )
                          .count();
                      matched_stats_age_sum_ms += age_ms;
                      matched_stats_age_max_ms =
                        std::max(matched_stats_age_max_ms, age_ms);
                      ++matched_stats_completions;
                      if (perf) {
                        sbs_perf::add_sample_ms("matched_frame_age", age_ms);
                      }
                    }
                  } else {
                    release_unknown_completion(polled.result.completed_frame_id, nullptr);
                  }
                }
              }
            } else if (any_pending_slot || block_current_submission_after_early_poll) {
              // A changed input must never race a prior root that was busy when admission was
              // derived. Even if can_accept_frame() would turn ready later in this call, the DDup
              // proof still names the older lineage. Retire it on the next conversion instead of
              // submitting with stale authority. A ready-but-invalid early completion takes the
              // same one-call fail-closed path.
              mark_sbs_matched_copy_end(gpu_timer, false);
            } else if (
              !any_pending_slot &&
              host_depth_admission == detail::host_sbs_depth_admission_e::reuse_cached
            ) {
              // The cached field owns this exact input. Skip the private copy, both preprocesses,
              // TensorRT, OCR, and postprocess; current capture color is still warped below.
              mark_sbs_matched_copy_end(gpu_timer, false);
              unchanged_content_submission_suppressed = true;
              damage_guided_submission_suppressed =
                depth_reuse_authorization.kind ==
                detail::host_sbs_depth_reuse_kind_e::exact_roi_damage;
              force_fresh_current_color = cached_geometry_render_allowed;
            } else {
              matched_candidate_slot = available_matched_slot(
                completion_finalized_before_admission ? matched_render_slot : nullptr
              );
              // Readiness is checked before the full-resolution private-slot copy. When TensorRT is
              // still using the previous matched frame, the output is repeated without spending a
              // D3D11 CopyResource on a source frame that cannot be enqueued.
              const bool estimator_ready =
                matched_candidate_slot && depth_estimator->can_accept_frame();
              const bool matched_copy_submitted =
                estimator_ready &&
                copy_matched_frame(
                  img_ctx.encoder_texture.get(),
                  *matched_candidate_slot,
                  frame_id,
                  input_color_space,
                  current_source_timestamp,
                  current_content_timestamp,
                  current_ddup_damage,
                  pre_copy_authority,
                  gpu_timer
                );
              if (!matched_copy_submitted) {
                // Every timing slot needs a complete timestamp tuple. A skipped/failed copy gets
                // adjacent markers and publishes no matched_frame_copy_gpu sample.
                mark_sbs_matched_copy_end(gpu_timer, false);
              }
              if (
                completion_finalized_before_admission && matched_render_slot &&
                !matched_route_matches_current(
                  *matched_render_slot,
                  live_source_desc,
                  input_color_space,
                  authority_generation(live_window_authority),
                  authority_generation(live_foreground_region),
                  live_browser_authority_epoch,
                  interactive_move_size_observed
                )
              ) {
                // copy_matched_frame() performs a second live authority observation. If that
                // closes a route race after the early completion was accepted, revoke its cache
                // and delivery now. Restore the prior opaque watermark as well: only a force
                // completion accepted on the final live route may release that barrier.
                latest_v2_lineage.reset();
                if (early_cleared_barrier_frame_id != 0u) {
                  gpu_observation_barrier.record_gpu_undecided_enqueue(
                    early_cleared_barrier_frame_id
                  );
                }
                matched_render_slot = nullptr;
                est = {};
                render_input_srv = img_ctx.encoder_input_res.get();
                completion_finalized_before_admission = false;
              }
              if (matched_copy_submitted) {
                // copy_matched_frame() closes the authority race with a second live observation.
                // Consume that final epoch now so the successful force-infer below can re-arm the
                // cadence without being revoked again by the next frame.
                if (adaptive_motion_route_state.observe(current_adaptive_route_epoch())) {
                  if (diagnostics_enabled && adaptive_gpu_opaque_followup) {
                    ++matched_stats_gpu_followup_host_rejected;
                    adaptive_gpu_followup_rejection_counted = true;
                  }
                  revoke_adaptive_reuse();
                }
                const auto observation_timestamp_us =
                  host_sbs_observation_timestamp_us(
                    matched_candidate_slot->captured_at
                  );
                const auto selected_optional_work = adaptive_ocr_cadence.select_mode(
                  observation_timestamp_us,
                  matched_candidate_slot->observed_interactive_move_size,
                  snapshot_debug_inputs
                );
                const auto final_admission_now = std::chrono::steady_clock::now();
                const bool initial_gpu_authority =
                  !adaptive_gpu_opaque_followup &&
                  adaptive_hold_decision ==
                    detail::host_sbs_adaptive_hold_decision_e::hold_candidate &&
                  adaptive_hold_cadence.hold_candidate_still_fresh(
                    current_content_timestamp,
                    final_admission_now
                  );
                const bool opaque_followup_authority =
                  adaptive_gpu_opaque_followup &&
                  detail::host_sbs_gpu_followup_fresh(
                    opaque_gpu_followup_anchor.frame_id,
                    gpu_observation_barrier.conditional_frame_id(),
                    opaque_gpu_followup_anchor.enqueued_at,
                    final_admission_now
                  );
                const bool authorize_gpu_undecided =
                  host_depth_admission == detail::host_sbs_depth_admission_e::gpu_undecided &&
                  models::depth_optional_work_allows_gpu_undecided(
                    selected_optional_work
                  ) &&
                  (initial_gpu_authority || opaque_followup_authority) &&
                  !matched_inference_pending_at_entry && !any_pending_slot &&
                  gpu_undecided_candidate_matches(
                    *matched_candidate_slot,
                    live_source_desc,
                    adaptive_gpu_candidate
                  );
                if (
                  diagnostics_enabled && adaptive_gpu_opaque_followup &&
                  !authorize_gpu_undecided &&
                  !adaptive_gpu_followup_rejection_counted
                ) {
                  if (!opaque_followup_authority) {
                    ++matched_stats_gpu_followup_expired;
                  } else {
                    ++matched_stats_gpu_followup_host_rejected;
                  }
                }
                // OCR scheduling is independent of the DAV2 decision: ordinary observations are
                // infer-coupled, while cadence-due observations run on either depth branch.
                // Native USER32 suppression advances neither subtitle tuple nor cadence.
                const auto optional_work = selected_optional_work;
                const auto adaptive_request =
                  gpu_observation_barrier.make_request(
                    frame_id,
                    authorize_gpu_undecided,
                    adaptive_gpu_opaque_followup,
                    adaptive_gpu_baseline_frame_id,
                    observation_timestamp_us
                  );
                auto submitted = depth_estimator->estimate_depth(
                  matched_candidate_slot->analysis_input_srv(),
                  input_color_space,
                  frame_id,
                  snapshot_debug_inputs,
                  matched_candidate_slot->depth_input_region,
                  optional_work,
                  adaptive_request
                );
                if (
                  diagnostics_enabled &&
                  matched_candidate_slot->depth_input_region.is_video_region() &&
                  (submitted.inference_enqueued ||
                   submitted.gpu_undecided_transaction_enqueued)
                ) {
                  ++matched_stats_roi_direct_inputs;
                }
                if (
                  completion_finalized_before_admission && est.completed_frame_valid &&
                  !submitted.completed_frame_valid
                ) {
                  // Preserve the already-authenticated prior completion for this draw while the
                  // current call's submission flags retain ownership of the new matched slot.
                  // The wrapper result owns no per-enqueue resources beyond these call metadata.
                  est.inference_enqueued = submitted.inference_enqueued;
                  est.gpu_undecided_transaction_enqueued =
                    submitted.gpu_undecided_transaction_enqueued;
                  est.subtitle_ocr_inference_enqueued =
                    submitted.subtitle_ocr_inference_enqueued;
                  est.cuda_graph_active =
                    est.cuda_graph_active || submitted.cuda_graph_active;
                } else {
                  est = std::move(submitted);
                  completion_finalized_before_admission = false;
                }
                if (
                  est.completed_frame_valid &&
                  !completion_finalized_before_admission
                ) {
                  latest_v2_lineage.reset();
                  matched_render_slot = find_pending_matched_slot(est.completed_frame_id);
                  if (matched_render_slot) {
                    matched_render_slot->pending = false;
                    render_input_srv = matched_render_slot->srv.get();
                    if (known_force_infer_completion(est, *matched_render_slot)) {
                      sbs_telemetry_last_sampled_frame_id = est.completed_frame_id;
                    }
                    if (diagnostics_enabled) {
                      const double age_ms = std::chrono::duration<double, std::milli>(
                                              std::chrono::steady_clock::now() - matched_render_slot->captured_at
                      )
                                              .count();
                      matched_stats_age_sum_ms += age_ms;
                      matched_stats_age_max_ms = std::max(matched_stats_age_max_ms, age_ms);
                      ++matched_stats_completions;
                      if (perf) {
                        sbs_perf::add_sample_ms("matched_frame_age", age_ms);
                      }
                    }
                  } else {
                    release_unknown_completion(
                      est.completed_frame_id, matched_candidate_slot);
                  }
                }
                const bool gpu_undecided_transaction_enqueued =
                  est.gpu_undecided_transaction_enqueued;
                const bool force_infer_transaction_enqueued = est.inference_enqueued;
                const bool depth_transaction_enqueued =
                  force_infer_transaction_enqueued || gpu_undecided_transaction_enqueued;
                const auto adaptive_submission_class =
                  gpu_observation_barrier.record_submission(
                    frame_id,
                    adaptive_request,
                    force_infer_transaction_enqueued,
                    gpu_undecided_transaction_enqueued
                  );
                if (
                  diagnostics_enabled && adaptive_gpu_opaque_followup &&
                  authorize_gpu_undecided && force_infer_transaction_enqueued
                ) {
                  ++matched_stats_gpu_followup_force_fallbacks;
                }
                if (gpu_undecided_transaction_enqueued) {
                  opaque_gpu_followup_anchor.record(
                    *matched_candidate_slot,
                    live_source_desc,
                    // Use the pre-submission observation so the shared owner-age window is
                    // conservative even when mapping/wrapper submission itself is delayed.
                    final_admission_now
                  );
                  approximate_reuse_since_enqueue =
                    detail::host_sbs_approximate_reuse_provider_e::gpu_undecided;
                }
                if (
                  depth_transaction_enqueued &&
                  adaptive_submission_class ==
                    models::gpu_adaptive_submission_class_e::invalid
                ) {
                  BOOST_LOG(error)
                    << "Host SBS estimator returned an invalid adaptive submission class; "sv
                       "revoking opaque follow-up authority."sv;
                  revoke_adaptive_reuse();
                }
                if (force_infer_transaction_enqueued) {
                  const auto enqueued_at = std::chrono::steady_clock::now();
                  opaque_gpu_followup_anchor.reset();
                  force_infer_enqueued_at = enqueued_at;
                  approximate_reuse_since_enqueue =
                    detail::host_sbs_approximate_reuse_provider_e::none;
                  if (
                    matched_candidate_slot->inference_content_timestamp &&
                    (!matched_candidate_slot->depth_input_region.is_video_region() ||
                     matched_candidate_slot->inference_ddup_damage)
                  ) {
                    content_reuse_refresh.record_successful_enqueue(enqueued_at);
                  } else {
                    latest_v2_lineage.reset();
                    content_reuse_refresh.reset();
                  }
                  adaptive_hold_cadence.record_successful_enqueue(
                    current_content_timestamp,
                    enqueued_at
                  );
                  // The forced full-source submission has retired the old ROI authority. Return
                  // subsequent observations to ordinary causality and route selection.
                  depth_authority_reprocess_pending = false;
                }
                if (depth_transaction_enqueued) {
                  adaptive_ocr_cadence.record_accepted(
                    optional_work,
                    adaptive_submission_class,
                    observation_timestamp_us
                  );
                  matched_candidate_slot->pending = true;
                  matched_candidate_slot->gpu_undecided_transaction =
                    gpu_undecided_transaction_enqueued;
                  depth_completion_poll_pending = true;
                }

                // Give the just-submitted matched-frame unit one immediate joined query. For a
                // conditional transaction this queries only root completion; it never reads the
                // GPU-owned reuse/infer decision.
                // Repeated polling is allowed only inside encode-loop cadence slack; a timeout
                // leaves the sole pending inference, its matched slot, and the prior completion
                // untouched. No older busy admission can enter this path because estimate_depth()
                // must have successfully enqueued this candidate in the current call.
                if (
                  depth_transaction_enqueued && matched_candidate_slot &&
                  matched_candidate_slot->pending && !snapshot_debug_inputs
                ) {
                  const auto poll_started = std::chrono::steady_clock::now();
                  if (diagnostics_enabled) {
                    ++matched_stats_same_frame_poll_submissions;
                  }
                  const auto poll_plan = detail::host_sbs_same_frame_poll_plan(
                    true,
                    false,
                    next_encode_target,
                    poll_started
                  );
                  auto polled = poll_plan.eligible ?
                                  depth_estimator->try_finish_pending_depth_until(
                                    input_color_space,
                                    poll_plan.deadline,
                                    detail::host_sbs_same_frame_poll_max_queries,
                                    false
                                  ) :
                                  depth_estimator->try_finish_pending_depth_nonblocking(
                                    input_color_space,
                                    false
                                  );
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
                    polled.result.completed_frame_valid,
                    polled.result.completed_frame_id,
                    matched_candidate_slot->frame_id
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
                    latest_v2_lineage.reset();
                    depth_completion_poll_pending = false;
                    matched_render_slot = nullptr;
                    render_input_srv = img_ctx.encoder_input_res.get();
                    est = {};
                    completion_finalized_before_admission = false;
                    if (
                      completion ==
                      detail::host_sbs_same_frame_completion_e::adopt_exact
                    ) {
                      matched_candidate_slot->pending = false;
                      if (!matched_candidate_slot->depth_input_region.is_video_region()) {
                        depth_authority_reprocess_pending = false;
                      }
                      matched_render_slot = matched_candidate_slot;
                      render_input_srv = matched_candidate_slot->srv.get();
                      if (known_force_infer_completion(
                            polled.result,
                            *matched_candidate_slot
                          )) {
                        sbs_telemetry_last_sampled_frame_id =
                          polled.result.completed_frame_id;
                      }
                      est = std::move(polled.result);
                      if (diagnostics_enabled) {
                        const double age_ms =
                          std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() -
                            matched_candidate_slot->captured_at
                          )
                            .count();
                        matched_stats_age_sum_ms += age_ms;
                        matched_stats_age_max_ms =
                          std::max(matched_stats_age_max_ms, age_ms);
                        ++matched_stats_completions;
                        if (perf) {
                          sbs_perf::add_sample_ms("matched_frame_age", age_ms);
                        }
                      }
                    } else {
                      // A ready empty/failure result consumed or invalidated B. Clear its exact
                      // slot and let the existing terminal-flat/authentication path decide output.
                      matched_candidate_slot->pending = false;
                      if (polled.result.completed_frame_valid) {
                        release_unknown_completion(
                          polled.result.completed_frame_id,
                          nullptr
                        );
                      }
                    }
                  }
                }

              }
            }
          } else {
            mark_sbs_matched_copy_end(gpu_timer, false);
            depth_completion_poll_pending = false;
          }

          if (unchanged_content_submission_suppressed) {
            switch (depth_reuse_authorization.refresh) {
              case detail::host_sbs_depth_reuse_refresh_e::none:
                // Pending-input suppression without a complete typed proof is not cache reuse.
                break;
              case detail::host_sbs_depth_reuse_refresh_e::bounded_content:
                content_reuse_refresh.record_reuse();
                if (diagnostics_enabled) {
                  ++matched_stats_content_skips;
                  if (damage_guided_submission_suppressed) {
                    ++matched_stats_damage_skips;
                  }
                }
                break;
            }
          }

          // Cached fields alias the estimator's singleton V2 textures, so recompute eligibility
          // only after every path that could have consumed/normalized a completion above.
          const bool post_completion_cache_route_matches =
            latest_v2_lineage_route_matches_current(
            live_source_desc,
            input_color_space,
            authority_generation(live_window_authority),
            authority_generation(live_foreground_region),
            live_browser_authority_epoch,
            interactive_move_size_observed
          );
          const bool post_completion_cache_gate_open =
            dedup_gate_open && !gpu_observation_barrier.active();
          const auto post_completion_cache_reuse_kind =
            post_completion_cache_gate_open && post_completion_cache_route_matches ?
              memoized_input_reuse_kind(latest_v2_lineage.slot) :
              detail::host_sbs_ddup_reuse_proof_e::none;
          auto post_completion_reuse_authorization =
            detail::select_host_sbs_cached_depth_reuse(
              content_refresh_due ? detail::host_sbs_ddup_reuse_proof_e::none :
                                    post_completion_cache_reuse_kind,
              latest_v2_lineage.slot.frame_id,
              frame_id
            );
          const bool post_completion_cache_render_allowed =
            detail::host_sbs_cached_geometry_render_allowed(
              post_completion_cache_gate_open && post_completion_cache_route_matches,
              host_sbs_renderer == models::host_sbs_renderer_e::parallax_v2,
              post_completion_reuse_authorization,
              latest_v2_lineage.slot.frame_id,
              frame_id,
              latest_v2_lineage.estimate.shadow_final_parallax &&
                latest_v2_lineage.estimate.shadow_state
            );
          if (
            !matched_render_slot && post_completion_cache_render_allowed &&
            (unchanged_content_submission_suppressed || est.inference_enqueued ||
             est.gpu_undecided_transaction_enqueued)
          ) {
            est = latest_v2_lineage.estimate;
            copy_matched_frame_attribution(
              current_color_reuse_slot,
              latest_v2_lineage.slot
            );
            using_cached_estimate = true;
            force_fresh_current_color = true;
            matched_render_slot = &current_color_reuse_slot;
          }
          const auto fresh_current_color_reuse_kind =
            post_completion_cache_gate_open && force_fresh_current_color &&
                matched_render_slot ?
              memoized_input_reuse_kind(*matched_render_slot) :
              detail::host_sbs_ddup_reuse_proof_e::none;
          if (
            post_completion_cache_gate_open && force_fresh_current_color &&
            matched_render_slot
          ) {
            post_completion_reuse_authorization =
              detail::select_host_sbs_cached_depth_reuse(
                fresh_current_color_reuse_kind,
                est.completed_frame_id,
                frame_id
              );
          }
          if (
            force_fresh_current_color && matched_render_slot &&
            post_completion_reuse_authorization.valid() &&
            post_completion_reuse_authorization.baseline_frame_id ==
              est.completed_frame_id
          ) {
            if (matched_render_slot != &current_color_reuse_slot) {
              copy_matched_frame_attribution(
                current_color_reuse_slot,
                *matched_render_slot
              );
            }
            current_color_reuse_slot.frame_id = est.completed_frame_id;
            current_color_reuse_slot.color_space = input_color_space;
            current_color_reuse_slot.source_timestamp = current_source_timestamp;
            current_color_reuse_slot.content_timestamp = current_content_timestamp;
            current_color_reuse_slot.captured_at = reuse_now;
            current_color_reuse_slot.pending = false;
            matched_render_slot = &current_color_reuse_slot;
            render_input_srv = img_ctx.encoder_input_res.get();
            cached_current_color_warp = true;
            depth_reuse_authorization = post_completion_reuse_authorization;
          }

          // A completion retired before admission was already authenticated and retained so the
          // current proof could name it. All other completions pass through the same helpers here.
          if (!completion_finalized_before_admission && matched_render_slot) {
            (void) validate_completed_slot();
          }
          if (
            matched_render_slot && matched_candidate_slot &&
            matched_candidate_slot->pending &&
            (
              !matched_render_slot->depth_input_region.same_analysis_domain(
                matched_candidate_slot->depth_input_region
              ) ||
              matched_render_slot->color_space != matched_candidate_slot->color_space
            )
          ) {
            // estimate_depth may retire domain A while enqueueing the first frame of domain B in
            // the same call. Once B is observed, do not resurrect or repeat A; show current flat
            // color until the exact B completion arrives and rearms its scene camera.
            if (completion_finalized_before_admission) {
              latest_v2_lineage.reset();
            }
            matched_render_slot = nullptr;
            est = {};
            render_input_srv = img_ctx.encoder_input_res.get();
            completion_finalized_before_admission = false;
          } else if (!completion_finalized_before_admission && matched_render_slot) {
            (void) apply_completed_domain_and_renderer();
          }

          // A current submission made after an early completion may itself fail terminally, so
          // keep this post-submit guard even when the prior completion was already finalized.
          if (depth_estimator && depth_estimator->has_terminal_failure()) {
            const auto failed_renderer =
              models::fail_host_sbs_renderer_flat(host_sbs_renderer);
            if (failed_renderer != host_sbs_renderer) {
              fail_depth_pipeline_flat();
              matched_render_slot = nullptr;
              est = {};
              render_input_srv = img_ctx.encoder_input_res.get();
              BOOST_LOG(error)
                << "Host SBS parallax-v2 producer reported a terminal CUDA/TensorRT failure; "sv
                   "this stream is now live flat identity with depth submissions disabled."sv;
            }
          }

          // Retain the latest completion that survived exact frame/region/route checks and the V2
          // authentication latch. This lineage aliases the estimator's singleton V2 resources but
          // does not itself authorize rendering; a typed current-frame reuse proof remains
          // separate.
          if (
            !completion_finalized_before_admission && matched_render_slot &&
            est.completed_frame_valid && !using_cached_estimate
          ) {
            const bool current_reusable_enqueue_pending =
              matched_candidate_slot && matched_candidate_slot->pending &&
              matched_route_matches_current(
                *matched_candidate_slot,
                live_source_desc,
                input_color_space,
                authority_generation(live_window_authority),
                authority_generation(live_foreground_region),
                live_browser_authority_epoch,
                interactive_move_size_observed
              ) &&
              memoized_input_reuse_kind(*matched_candidate_slot) !=
                detail::host_sbs_ddup_reuse_proof_e::none;
            retain_completed_lineage(
              current_reusable_enqueue_pending,
              false
            );
          }

          // A busy CUDA stream may legitimately repeat one matched pair for a few source frames,
          // but NOT_READY is not proof of eventual progress. Bound live V2 by the source frame's
          // age; once stale, render current color through the V2 shader's flat-identity path while
          // continuing nonblocking queries so a later fresh completion can recover automatically.
          const auto repeat_now = std::chrono::steady_clock::now();
          bool stale_v2_completion = false;
          if (matched_render_slot &&
              host_sbs_renderer == models::host_sbs_renderer_e::parallax_v2 &&
              !models::host_sbs_matched_completion_is_current(
                same_source(
                  matched_render_slot->source_timestamp,
                  current_source_timestamp
                ),
                repeat_now - matched_render_slot->captured_at
              )) {
            stale_v2_completion = true;
            matched_render_slot = nullptr;
            est = {};
            render_input_srv = img_ctx.encoder_input_res.get();
          }
          if (matched_render_slot) {
            publish_window_region_transition(*matched_render_slot);
          }
          const auto repeat_source_age =
            matched_presentation_cache.source_age(repeat_now);
          const bool output_source_unchanged =
            matched_presentation_cache.source_matches(current_source_timestamp);
          // Packed-presentation retention is deliberately weaker than reusable V2 lineage: it
          // may replay only the already-rendered pixels. Recheck the live transfer/source domain
          // here even though copy_matched_frame() also invalidates mismatched domains, so no route
          // or reset race can turn presentation continuity into geometry authority.
          const bool packed_presentation_route_matches =
            matched_presentation_cache.route_matches(
              live_source_desc.Width,
              live_source_desc.Height,
              input_color_space,
              depth_authority_reprocess_pending,
              producer_terminal
            );
          const bool repeat_matched_output =
            models::host_sbs_should_repeat_packed_presentation(
              host_sbs_renderer,
              matched_render_slot != nullptr,
              matched_presentation_cache.has_fresh_pixels(),
              packed_presentation_route_matches,
              repeat_source_age,
              output_source_unchanged
            );
          converted_content_timestamp =
            ::video::detail::select_rendered_content_timestamp(
              repeat_matched_output,
              matched_presentation_cache.content_timestamp(),
              matched_render_slot != nullptr,
              matched_render_slot ? matched_render_slot->content_timestamp : std::nullopt,
              matched_render_slot ? matched_render_slot->source_timestamp : std::nullopt,
              current_content_timestamp,
              current_source_timestamp
            );
          const bool v2_repeat_timed_out =
            stale_v2_completion ||
            (host_sbs_renderer == models::host_sbs_renderer_e::parallax_v2 &&
             !matched_render_slot && matched_presentation_cache.has_fresh_pixels() &&
             !output_source_unchanged &&
             repeat_source_age > models::host_sbs_v2_max_matched_repeat_age);
          if (v2_repeat_timed_out && !matched_presentation_cache.timeout_active()) {
            matched_presentation_cache.mark_timeout();
            BOOST_LOG(error)
              << "Host SBS packed presentation continuity exhausted after "sv
              << models::host_sbs_v2_max_matched_repeat_age.count()
              << " ms for a changed source; rendering live current-frame flat identity until "sv
                 "depth recovers."sv;
          } else if (!v2_repeat_timed_out && matched_presentation_cache.timeout_active() &&
                     matched_render_slot) {
            matched_presentation_cache.clear_timeout();
            BOOST_LOG(info) << "Host SBS V2 depth completion recovered; stereo warp resumed."sv;
          }
          const bool v2_renderer_selected =
            host_sbs_renderer == models::host_sbs_renderer_e::parallax_v2;
          const bool v2_renderer_failed_flat =
            host_sbs_renderer == models::host_sbs_renderer_e::failed_flat;
          const bool v2_live_resources_complete =
            est.shadow_final_parallax && est.shadow_state;
          const bool v2_live_warp_selected =
            v2_renderer_selected && matched_render_slot &&
            models::parallax_v2_result_is_authenticated(est) &&
            sbs_reprojection_v2_live_ps && v2_live_resources_complete;
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
          ID3D11ShaderResourceView *selected_parallax_field =
            v2_live_warp_selected ? est.shadow_final_parallax.Get() : nullptr;
          // The same fullscreen pass performs either authenticated reprojection or current-frame
          // flat identity. Time both; a packed repeat performs neither draw.
          const bool timing_has_sbs_composite = !repeat_matched_output;
          mark_sbs_warp_start(gpu_timer, timing_has_sbs_composite);
          ID3D11ShaderResourceView *final_sbs_srv = nullptr;
          ID3D11Texture2D *final_sbs_texture = nullptr;
          bool final_sbs_is_linear = sbs_intermediate_is_linear;
          bool final_sbs_has_p010_y = false;
          ID3D11ShaderResourceView *dump_warp_depth = nullptr;
          if (diagnostics_enabled) {
            if (repeat_matched_output) {
              ++matched_stats_output_packed_repeat;
            } else if (v2_live_warp_selected) {
              ++matched_stats_output_warped_new;
            } else {
              ++matched_stats_output_flat;
              if (
                same_frame_poll_missed_for_current_draw &&
                depth_completion_poll_pending
              ) {
                // This should remain zero once a valid packed presentation exists. Keep the
                // counter explicit so a route/reset invalidation can be distinguished from the
                // former adaptive timeout flash in live logs.
                ++matched_stats_output_flat_pending_miss;
              }
            }
          }
          if (repeat_matched_output) {
            final_sbs_srv = sbs_intermediate_srv.get();
            final_sbs_texture = sbs_intermediate_texture.get();
            if (diagnostics_enabled) {
              ++matched_stats_repeats;
            }
          } else {
            // Before the first matched completion, provide a flat current-frame SBS image. An
            // ordinary completion renders its private matched color; an authorized cache reuse
            // deliberately renders current color/cursor with authenticated cached geometry.
            if (!matched_render_slot) {
              est = {};
              render_input_srv = img_ctx.encoder_input_res.get();
            }

            // Output transfer follows the color frame actually rendered. Matched inference may
            // complete later than the current capture; repeated output retains the transfer of
            // the packed texture already stored in the intermediate.
            const bool render_input_is_linear = matched_render_slot ?
                                                    models::input_color_space_is_linear(
                                                      matched_render_slot->color_space
                                                    ) :
                                                    input_is_linear;

            // DDup reveals the current capture transfer only after the first real frame, while
            // asynchronous depth completion can select an older buffered color frame. Size the
            // packed target from the frame we are about to render, not from the newest capture.
            // Repeated output deliberately skips this path so a transfer transition cannot
            // reallocate and erase the last completed SBS image before it is presented.
            if (!ensure_sbs_intermediate_storage(render_input_is_linear)) {
              return -1;
            }

            ID3D11ShaderResourceView *warp_depth = selected_parallax_field;
            dump_warp_depth = warp_depth;
            if (!update_sbs_constant_buffer(
                  sbs_content_scale_x,
                  sbs_content_scale_y,
                  v2_live_warp_selected ? &est.input_region : nullptr
                )) {
              return -1;
            }
            ID3D11Buffer *sbs_constants = sbs_reprojection_cbuffer.get();

            // Only an authenticated V2 stream selects the production warp. Awaiting, stale, and
            // terminal failure states use the independent current-frame identity shader, so V2
            // shader authentication/device-creation failures cannot freeze or end the stream.
            const bool p010_y_mrt_authorized =
              detail::host_sbs_p010_y_mrt_eligible(
                v2_live_warp_selected,
                render_input_is_linear,
                display_is_hdr,
                format == DXGI_FORMAT_P010,
                rgb_present_target == nullptr,
                snapshot_debug_inputs,
                sbs_reprojection_v2_p010_y_ps != nullptr,
                out_Y_or_YUV_rtv != nullptr
              );
            const bool p010_y_mrt_selected =
              p010_y_mrt_authorized && qualify_p010_y_mrt_targets();
            if (p010_y_mrt_selected && diagnostics_enabled) {
              ++matched_stats_output_p010_y_mrt;
            }
            if (p010_y_mrt_selected) {
              ID3D11Buffer *matrix = color_matrix.get();
              device_ctx->PSSetConstantBuffers(0, 1, &matrix);
            }
            const host_sbs_v2_draw_command_t draw_command {
              .render_targets = {
                sbs_intermediate_rtv.get(),
                p010_y_mrt_selected ? out_Y_or_YUV_rtv.get() : nullptr,
              },
              .render_target_count = p010_y_mrt_selected ? 2u : 1u,
              .vertex_shader = sbs_reprojection_vs.get(),
              .pixel_shader = p010_y_mrt_selected ?
                                sbs_reprojection_v2_p010_y_ps.get() :
                                v2_live_warp_selected ?
                                  sbs_reprojection_v2_live_ps.get() :
                                  sbs_flat_identity_ps.get(),
              .viewport = sbs_viewport,
              .sampler = sampler_linear.get(),
              .shader_resources = {
                render_input_srv,
                warp_depth,
                v2_live_warp_selected ? est.shadow_state.Get() : nullptr,
                nullptr,
                nullptr,
                nullptr,
              },
              .geometry_constants = sbs_constants,
            };
            if (!record_host_sbs_v2_draw(device_ctx.get(), draw_command)) {
              BOOST_LOG(error) << "Host SBS V2 draw operands are incomplete."sv;
              return -1;
            }
            final_sbs_has_p010_y = p010_y_mrt_selected;

            final_sbs_srv = sbs_intermediate_srv.get();
            final_sbs_texture = sbs_intermediate_texture.get();
            sbs_intermediate_is_linear = render_input_is_linear;
            final_sbs_is_linear = render_input_is_linear;
            if (models::host_sbs_packed_output_can_enter_presentation_cache(
                  matched_render_slot != nullptr,
                  v2_live_warp_selected
                )) {
              const auto rendered_content_timestamp =
                ::video::detail::select_rendered_content_timestamp(
                  false,
                  std::nullopt,
                  true,
                  matched_render_slot->content_timestamp,
                  matched_render_slot->source_timestamp,
                  std::nullopt,
                  std::nullopt
                );
              matched_presentation_cache.publish(
                matched_render_slot->captured_at,
                matched_render_slot->source_timestamp,
                rendered_content_timestamp,
                est.input_region,
                matched_render_slot->color_space
              );
            } else {
              // Flat identity cannot seed packed presentation continuity. This cache remains
              // completely separate from semantic depth/OCR/DDup lineage and the opaque
              // observation barrier.
              matched_presentation_cache.invalidate();
            }
          }

          mark_sbs_warp_end(gpu_timer);

          const bool output_conversion_required =
            host_sbs_encoder_input_state.conversion_required(
              repeat_matched_output,
              rgb_present_target != nullptr
            );
          if (rgb_present_target) {
            // The local AR presenter consumes the production RGB warp directly, avoiding an
            // encode/decode round trip. Exact layouts take the copy fast path; retain the shader
            // for any future mode whose source and physical target require scaling or conversion.
            if (!copy_rgb(final_sbs_texture, final_sbs_is_linear)) {
              draw_rgb(final_sbs_srv, final_sbs_is_linear);
            }
          } else if (output_conversion_required) {
            // Host SBS accepts identity-oriented sources only. Portrait is represented by actual
            // W < H display dimensions, so this final conversion never rotates a packed 2W frame.
            draw(
              final_sbs_srv,
              out_Y_or_YUV_viewport,
              out_UV_viewport,
              final_sbs_is_linear,
              final_sbs_has_p010_y
            );
            // Native NVENC owns one registered input texture for this whole encode session. The
            // completed draw remains there until a later conversion overwrites it, so a repeated
            // packed SBS frame can be encoded again without identical Y and UV draws.
            host_sbs_encoder_input_state.mark_converted();
          }
          end_sbs_gpu_timer(gpu_timer);
          // Telemetry is deliberately submitted after the production warp/output work. The
          // estimator uses a nonblocking staging/query ring, so this can neither flush nor wait
          // on the D3D11 queue. Without an external subscription it schedules no diagnostic copy;
          // GPU adaptive arbitration owns no telemetry readback.
          poll_sbs_telemetry_after_output();

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
          if (!repeat_matched_output) {
            const bool complete_dump_snapshot =
              matched_render_slot && dump_warp_depth && est.model_input_snapshot &&
              est.raw_model_depth_snapshot && est.raw_model_provenance &&
              est.shadow_final_parallax &&
              v2_renderer_selected &&
              models::parallax_v2_result_is_authenticated(est);
            ID3D11ShaderResourceView *dump_depth_input_source =
              matched_render_slot ? matched_render_slot->analysis_input_srv() : nullptr;
            if (complete_dump_snapshot && est.input_region.is_video_region() &&
                matched_render_slot->texture) {
              // This explicit diagnostic copy targets the exact completed slot after all live
              // timing has closed. The dumper revalidates the retained texture against the
              // completed region and owns the crop; ordinary ROI analysis allocates no crop.
              dump_depth_input_source = sbs_dumper.prepare_diagnostic_roi_crop(
                device.get(),
                device_ctx.get(),
                matched_render_slot->texture.get(),
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
              ID3D11Buffer *completed_constants = sbs_reprojection_cbuffer.get();
              bool geometry_available = false;
              try {
                geometry_available = render_sbs_debug_geometry(
                  render_input_srv,
                  dump_warp_depth,
                  est.shadow_state.Get(),
                  completed_constants
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
                dump_frame.source = render_input_srv;
                dump_frame.depth_input_source = dump_depth_input_source;
                dump_frame.model_input = est.model_input_snapshot.Get();
                dump_frame.raw_depth = est.raw_model_depth_snapshot.Get();
                dump_frame.warp_depth = dump_warp_depth;
                dump_frame.adaptive_state = est.cut_state.Get();
                dump_frame.depth_frame_state = est.depth_frame_state.Get();
                dump_frame.warp_map =
                  geometry_available ? sbs_debug_mapping_srv.Get() : nullptr;
                dump_frame.warp_mask =
                  geometry_available ? sbs_debug_mask_srv.Get() : nullptr;
                dump_frame.sbs = final_sbs_srv;
                dump_frame.shadow_coordinate = est.shadow_coordinate.Get();
                dump_frame.shadow_candidate_parallax =
                  est.shadow_candidate_parallax.Get();
                dump_frame.shadow_ownership_refined_parallax =
                  est.shadow_ownership_refined_parallax.Get();
                dump_frame.shadow_vertical_majorant =
                  est.shadow_vertical_majorant.Get();
                dump_frame.shadow_vertical_conditioned =
                  est.shadow_vertical_conditioned.Get();
                dump_frame.shadow_base_final_parallax =
                  est.shadow_base_final_parallax.Get();
                dump_frame.shadow_final_parallax = est.shadow_final_parallax.Get();
                dump_frame.ocr_box_record = est.ocr_box_record.Get();
                dump_frame.subtitle_locator_state =
                  est.subtitle_locator_state.Get();
                dump_frame.gpu_trace_ring = est.gpu_trace_ring.Get();
                dump_frame.shadow_state = est.shadow_state.Get();
                dump_frame.shadow_frame_stats = est.shadow_frame_stats.Get();
                dump_frame.raw_model_provenance = est.raw_model_provenance;
                dump_frame.parallax_v2_shader_provenance =
                  est.parallax_v2_shader_provenance;
                dump_frame.gpu_trace_provenance = est.gpu_trace_provenance;
                dump_frame.model_width = est.raw_width;
                dump_frame.model_height = est.raw_height;
                dump_frame.raw_width = est.raw_width;
                dump_frame.raw_height = est.raw_height;
                dump_frame.matched_frame_id = est.completed_frame_id;
                dump_frame.depth_input_region = est.input_region;
                dump_frame.depth_video_plan = matched_render_slot->depth_video_plan;
                dump_frame.input_domain_reset = est.input_domain_reset;
                dump_frame.gpu_undecided_completion =
                  est.gpu_undecided_completion;
                dump_frame.subtitle_work_suppressed =
                  est.subtitle_work_suppressed;
                dump_frame.window_region = matched_render_slot->window_region;
                dump_frame.window_region_observer_status =
                  matched_render_slot->window_region_observer_status;
                dump_frame.window_region_mapping_status =
                  matched_render_slot->window_region_mapping_status;
                dump_frame.cuda_graph_active = est.cuda_graph_active;
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
                dump_frame.color_space = matched_render_slot->color_space;
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
                matched_render_slot && selected_parallax_field &&
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
                               << " opaque_followup_expired/rejected/force_fallback="sv
                               << matched_stats_gpu_followup_expired << '/'
                               << matched_stats_gpu_followup_host_rejected << '/'
                               << matched_stats_gpu_followup_force_fallbacks;
              reset_matched_stats(now);
            }
          }

        } else {
          if (rgb_present_target) {
            const bool input_is_linear =
              img.format == DXGI_FORMAT_R16G16B16A16_FLOAT;
            if (!copy_rgb(img_ctx.encoder_texture.get(), input_is_linear)) {
              draw_rgb(
                img_ctx.encoder_input_res.get(),
                input_is_linear
              );
            }
          } else {
            // Plain 2D: draw the captured frame straight into the encoder output.
            draw(img_ctx.encoder_input_res, out_Y_or_YUV_viewport, out_UV_viewport, img.format == DXGI_FORMAT_R16G16B16A16_FLOAT);
          }
        }

        ID3D11ShaderResourceView *emptyShaderResourceView = nullptr;
        device_ctx->PSSetShaderResources(0, 1, &emptyShaderResourceView);
        rendered_content_timestamp_ = converted_content_timestamp;
      } else {
        rendered_content_timestamp_.reset();
      }

      return 0;
    }

    bool apply_colorspace(const ::video::sunshine_colorspace_t &colorspace) {
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
      if (source_is_hdr) {
        const auto queried_sdr_white_nits = display->get_sdr_white_nits();
        if (queried_sdr_white_nits && *queried_sdr_white_nits > 0.0f) {
          sdr_white_nits = *queried_sdr_white_nits;
        } else {
          BOOST_LOG(warning)
            << "Failed to query the display's SDR reference white; using "sv
            << fallback_sdr_white_nits << " nits for HDR-to-SDR tone mapping."sv;
        }
      }

      const sdr_color_transform_t transform {
        colorspace.colorspace == ::video::colorspace_e::bt2020sdr,
        source_is_hdr,
        sdr_white_nits / 80.0f,
        0.0f,
      };
      auto sdr_color_transform = make_buffer(device.get(), transform);
      if (!sdr_color_transform) {
        BOOST_LOG(warning) << "Failed to create SDR color-transform constants"sv;
        return false;
      }

      device_ctx->VSSetConstantBuffers(3, 1, &color_matrix);
      device_ctx->PSSetConstantBuffers(0, 1, &color_matrix);
      this->color_matrix = std::move(color_matrix);
      this->sdr_color_transform = std::move(sdr_color_transform);
      output_black_y = color_vectors->color_vec_y[3];
      output_neutral_uv = color_vectors->color_vec_u[3];
      return true;
    }

    void fail_depth_pipeline_flat() {
      host_sbs_renderer = models::fail_host_sbs_renderer_flat(host_sbs_renderer);
      retire_depth_estimator(std::move(depth_estimator));
      for (auto &slot : matched_frame_slots) {
        slot.gpu_undecided_transaction = false;
        slot.pending = false;
      }
      gpu_observation_barrier.reset();
      opaque_gpu_followup_anchor.reset();
      latest_v2_lineage.reset();
      content_reuse_refresh.reset();
      approximate_reuse_since_enqueue =
        detail::host_sbs_approximate_reuse_provider_e::none;
      reset_adaptive_reuse_runtime();
      matched_presentation_cache.reset();
      depth_completion_poll_pending = false;
      depth_authority_reprocess_pending = false;
      sbs_dumper.cancel_pending_request();
      publish_depth_status(0);
    }

    // Create the D3D depth pipeline on demand (first SBS frame). The heavy TensorRT engine,
    // execution context, and CUDA modules were prepared at host startup; construction now borrows
    // that warm context and creates only device/session resources on a background thread.
    bool ensure_depth_estimator() {
      if (!models::host_sbs_renderer_uses_depth_pipeline(host_sbs_renderer)) {
        sbs_dumper.cancel_pending_request();
        return false;
      }
      if (depth_estimator) {
        return true;
      }

      // An engaged future is the sole owner of an in-flight build. Take the result once ready;
      // get() invalidates the future, so no parallel flag can drift from the shared state.
      if (depth_estimator_build.valid()) {
        if (depth_estimator_build.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
          try {
            depth_estimator = depth_estimator_build.get();
          } catch (const std::exception &e) {
            // Don't let a background-build exception propagate on the encode thread (it would end
            // the stream); log it, clear the client's "loading" indicator, and stream flat.
            BOOST_LOG(error) << "Host SBS per-stream GPU pipeline initialization failed: "sv << e.what();
            fail_depth_pipeline_flat();
            return false;
          }
          if (depth_estimator && depth_estimator->is_valid() &&
              !depth_estimator->has_terminal_failure()) {
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

      // Startup model preparation is process-global and may take minutes on first use. It includes
      // engine compilation, deserialization, execution-context creation, and CUDA warmup. Report
      // that complete wait as loading, but don't duplicate it on this encode device.
      auto active = ::video::host_sbs_v2_depth_model();
      auto build_status = models::tensorrt_model_prepare_status(active);
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

      // Kick construction on a background thread. The D3D device/context are free-threaded for the
      // resource creation the constructor does (it makes no immediate-context calls). Capture
      // owning ComPtrs so a presentation-session teardown never has to wait for construction.
      auto sbs_cfg = sbs_config;
      // Every estimator consumer uses the authenticated Host SBS tensor shapes and private cut
      // analysis contract.
      BOOST_LOG(info) << "Host SBS enabled; initializing the per-stream GPU pipeline for resident model \""sv
                      << active.name
                      << "\" in the background (streaming flat until ready)..."sv;
      Microsoft::WRL::ComPtr<ID3D11Device> dev(device.get());
      Microsoft::WRL::ComPtr<ID3D11DeviceContext> ctx(device_ctx.get());
      depth_estimator_build_task_t build_task([
        dev = std::move(dev),
        ctx = std::move(ctx),
        active,
        sbs_cfg
      ]() mutable {
        return std::make_unique<models::video_depth_estimator>(
          std::move(dev),
          std::move(ctx),
          std::filesystem::path(SUNSHINE_ASSETS_DIR),
          sbs_cfg,
          active
        );
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
      const models::depth_input_region_t *input_region = nullptr
    ) {
      auto geometry = models::make_host_sbs_v2_full_frame_geometry(
        content_scale_x,
        content_scale_y
      );
      if (input_region && input_region->is_video_region() && input_region->valid()) {
        const auto source_width = static_cast<float>(input_region->source_width);
        const auto source_height = static_cast<float>(input_region->source_height);
        geometry.video_roi_active = 1.0f;
        geometry.video_roi_left = static_cast<float>(input_region->left) / source_width;
        geometry.video_roi_top = static_cast<float>(input_region->top) / source_height;
        geometry.video_roi_right = static_cast<float>(input_region->right) / source_width;
        geometry.video_roi_bottom = static_cast<float>(input_region->bottom) / source_height;
        geometry.tensor_content_left = input_region->tensor_content.left;
        geometry.tensor_content_top = input_region->tensor_content.top;
        geometry.tensor_content_right = input_region->tensor_content.right;
        geometry.tensor_content_bottom = input_region->tensor_content.bottom;
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
      ::video::apply_sbs_telemetry_config(snapshot, sbs_config);
      return snapshot;
    }

    void publish_sbs_telemetry_failure() {
      if (sbs_telemetry_event) {
        sbs_telemetry_event->raise(base_sbs_telemetry_snapshot(
          ::video::sbs_telemetry_sample_status_e::failed
        ));
      }
    }

    void publish_sbs_telemetry_sample(const models::depth_telemetry_sample &sample) {
      if (!sbs_telemetry_event) {
        return;
      }

      auto snapshot = base_sbs_telemetry_snapshot(
        ::video::sbs_telemetry_sample_status_e::ready
      );
      snapshot.valid_fields =
        ::video::sbs_telemetry_valid_field::config |
        ::video::sbs_telemetry_valid_field::effective_pop |
        ::video::sbs_telemetry_valid_field::change |
        ::video::sbs_telemetry_valid_field::valid_fraction |
        ::video::sbs_telemetry_valid_field::cuts |
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
      snapshot.change_fraction = sample.change_fraction;
      snapshot.zero_anchor_shift_px = sample.zero_anchor_shift_px;
      snapshot.subject_depth = sample.subject_depth;
      snapshot.valid_depth_fraction = sample.valid_depth_fraction;
      snapshot.effective_range_width = sample.effective_range_width;
      snapshot.scene_age = sample.scene_age;
      snapshot.hard_cut_count = sample.hard_cut_count;
      snapshot.external_cut_count = sample.external_cut_count;
      snapshot.empty_raw_count = sample.empty_raw_count;
      snapshot.collapsed_raw_count = sample.collapsed_raw_count;
      snapshot.sampled_frame_id = static_cast<std::uint32_t>(sample.sampled_frame_id);

      if (sample.profile_initialized) {
        snapshot.runtime_flags |= ::video::sbs_telemetry_runtime_flag::profile_initialized;
        snapshot.valid_fields |= ::video::sbs_telemetry_valid_field::scene;
      }
      if (
        (sample.cut_flags & sbs_adaptive_state::cut_flag_geometry_armed) != 0
      ) {
        snapshot.runtime_flags |= ::video::sbs_telemetry_runtime_flag::geometry_armed;
      }
      if (
        (sample.cut_flags & sbs_adaptive_state::cut_flag_appearance_armed) != 0
      ) {
        snapshot.runtime_flags |= ::video::sbs_telemetry_runtime_flag::appearance_armed;
      }
      const bool unseen_cut =
        sbs_telemetry_has_sample &&
        sample.hard_cut_count != sbs_telemetry_last_hard_cut_count;
      if (sample.hard_cut_pulse || unseen_cut) {
        snapshot.runtime_flags |= ::video::sbs_telemetry_runtime_flag::hard_cut_pulse;
      }
      sbs_telemetry_last_hard_cut_count = sample.hard_cut_count;
      sbs_telemetry_has_sample = true;
      sbs_telemetry_event->raise(std::move(snapshot));
    }

    void reset_adaptive_reuse_runtime() noexcept {
      adaptive_hold_cadence.reset();
      adaptive_ocr_cadence.reset();
    }

    void revoke_adaptive_reuse() noexcept {
      reset_adaptive_reuse_runtime();
      opaque_gpu_followup_anchor.reset();
    }

    void poll_sbs_telemetry_after_output() {
      const auto now = std::chrono::steady_clock::now();
      const bool external_available =
        sbs_telemetry_subscription && sbs_telemetry_event;
      if (!external_available) {
        return;
      }

      const bool enabled = sbs_telemetry_subscription->enabled();
      const bool producer_active =
        models::host_sbs_renderer_uses_depth_pipeline(host_sbs_renderer);
      if (!depth_estimator) {
        if (enabled && !producer_active &&
            !sbs_telemetry_producer_failure_published) {
          publish_sbs_telemetry_failure();
          sbs_telemetry_producer_failure_published = true;
        }
        return;
      }
      const auto interval = enabled ?
                              std::chrono::milliseconds(std::max<std::uint16_t>(
                                sbs_telemetry_subscription->interval_ms(),
                                1
                              )) :
                              std::chrono::milliseconds {1};
      const bool external_due =
        enabled &&
        (sbs_telemetry_last_copy.time_since_epoch().count() == 0 ||
         now - sbs_telemetry_last_copy >= interval);
      const bool schedule_snapshot =
        external_due && producer_active && !gpu_observation_barrier.active();
      auto result = depth_estimator->poll_depth_telemetry(
        schedule_snapshot,
        sbs_telemetry_last_sampled_frame_id
      );
      // A terminal fail-flat renderer still retires an already-submitted readback, but its last
      // ready depth sample is no longer live data. Publish one explicit failure and suppress that
      // retired sample so subscribers cannot mistake a frozen chart for an active producer.
      if (!producer_active) {
        if (external_due) {
          sbs_telemetry_last_copy = now;
        }
        if (enabled && !sbs_telemetry_producer_failure_published) {
          publish_sbs_telemetry_failure();
          sbs_telemetry_producer_failure_published = true;
        }
        return;
      }
      // A failed lazy allocation/query creation is also an attempt. Advancing the cadence here
      // prevents a permanent device error from publishing a fresh failed sequence every render
      // frame, while still retrying (and therefore recovering) at the requested interval.
      if (result.copy_scheduled || (schedule_snapshot && result.failed)) {
        if (external_due) {
          sbs_telemetry_last_copy = now;
        }
      }
      if (result.failed) {
        if (enabled) {
          publish_sbs_telemetry_failure();
        }
      }
      if (
        result.sample &&
        result.sample->sampled_frame_id >= sbs_telemetry_min_frame_id
      ) {
        if (enabled) {
          publish_sbs_telemetry_sample(*result.sample);
        }
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
          }
          sbs_perf::add_sample_ms_if_current(
            "sbs_output_gpu",
            static_cast<double>(convert_end - warp_end) * to_ms,
            slot.perf_generation
          );
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
        source_at_ = {};
        source_timestamp_.reset();
        content_timestamp_.reset();
        input_region_ = {};
        color_space_ = models::input_color_space::srgb;
      }

      void reset() noexcept {
        invalidate();
        timeout_active_ = false;
      }

      void publish(
        const std::chrono::steady_clock::time_point presented_at,
        const std::optional<std::chrono::steady_clock::time_point> &presented_source_timestamp,
        const std::optional<std::chrono::steady_clock::time_point> &presented_content_timestamp,
        const models::depth_input_region_t &presented_input_region,
        const models::input_color_space presented_color_space
      ) noexcept {
        // A recovered matched completion clears timeout_active at its existing policy site.
        valid_ = true;
        source_at_ = presented_at;
        source_timestamp_ = presented_source_timestamp;
        content_timestamp_ = presented_content_timestamp;
        input_region_ = presented_input_region;
        color_space_ = presented_color_space;
      }

      [[nodiscard]] bool has_fresh_pixels() const noexcept {
        return valid_;
      }

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

      [[nodiscard]] bool route_matches(
        const UINT source_width,
        const UINT source_height,
        const models::input_color_space color_space,
        const bool reprocess_pending,
        const bool producer_terminal
      ) const noexcept {
        return valid_ && input_region_.valid() &&
               input_region_.source_width == source_width &&
               input_region_.source_height == source_height &&
               color_space_ == color_space && !reprocess_pending && !producer_terminal;
      }

      [[nodiscard]] const std::optional<std::chrono::steady_clock::time_point> &
      content_timestamp() const noexcept {
        return content_timestamp_;
      }

      [[nodiscard]] bool timeout_active() const noexcept {
        return timeout_active_;
      }

      void mark_timeout() noexcept {
        timeout_active_ = true;
      }

      void clear_timeout() noexcept {
        timeout_active_ = false;
      }

      [[nodiscard]] bool is_video_region() const noexcept {
        return valid_ && input_region_.is_video_region();
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
        return valid_ && region.same_analysis_domain(input_region_) &&
               color_space == color_space_;
      }

    private:
      bool valid_ = false;
      std::chrono::steady_clock::time_point source_at_ {};
      std::optional<std::chrono::steady_clock::time_point> source_timestamp_;
      std::optional<std::chrono::steady_clock::time_point> content_timestamp_;
      models::depth_input_region_t input_region_ {};
      models::input_color_space color_space_ = models::input_color_space::srgb;
      bool timeout_active_ = false;
    };

    struct matched_frame_slot_t {
      texture2d_t texture;
      shader_res_t srv;
      std::uint64_t frame_id = 0;
      models::input_color_space color_space = models::input_color_space::srgb;
      std::optional<std::chrono::steady_clock::time_point> source_timestamp;
      std::optional<std::chrono::steady_clock::time_point> content_timestamp;
      // These two fields identify the pixels actually submitted to DAV2. Current-color reuse may
      // restamp the presentation attribution above, but it must never advance this baseline.
      std::optional<std::chrono::steady_clock::time_point> inference_content_timestamp;
      std::optional<detail::ddup_damage_snapshot_t> inference_ddup_damage;
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
      std::uint64_t observed_root_authority_generation = 0u;
      std::uint64_t observed_region_authority_generation = 0u;
      std::uint64_t observed_browser_authority_epoch = 0u;
      bool observed_interactive_move_size = false;
      // CPU-owned submission class only. The conditional branch itself is deliberately never
      // read back; this bit authenticates the completion metadata returned by the estimator.
      bool gpu_undecided_transaction = false;
      bool pending = false;

      ID3D11ShaderResourceView *analysis_input_srv() noexcept {
        return srv.get();
      }
    };

    /** Scalar-only authority for one immediately preceding GPU-opaque transaction.
     *
     * The barrier deliberately prevents this from becoming render, OCR, telemetry, or host-cache
     * lineage. It retains only the route and DDup tuple needed to let the device compare the next
     * copied tensor against its private history owner without any branch readback.
     */
    struct opaque_gpu_followup_anchor_t {
      std::uint64_t frame_id = 0u;
      models::input_color_space color_space = models::input_color_space::srgb;
      std::optional<std::chrono::steady_clock::time_point> content_timestamp;
      std::optional<detail::ddup_damage_snapshot_t> damage;
      std::chrono::steady_clock::time_point enqueued_at {};
      models::depth_input_region_t input_region {};
      UINT source_width = 0u;
      UINT source_height = 0u;
      UINT mip_levels = 0u;
      UINT array_size = 0u;
      DXGI_FORMAT source_format = DXGI_FORMAT_UNKNOWN;
      UINT sample_count = 0u;
      UINT sample_quality = 0u;
      std::uint64_t root_authority_generation = 0u;
      std::uint64_t region_authority_generation = 0u;
      std::uint64_t browser_authority_epoch = 0u;
      bool interactive_move_size = false;
      bool authenticated = false;

      void reset() noexcept {
        *this = {};
      }

      void record(
        const matched_frame_slot_t &slot,
        const D3D11_TEXTURE2D_DESC &source_desc,
        const std::chrono::steady_clock::time_point transaction_enqueued_at
      ) noexcept {
        frame_id = slot.frame_id;
        color_space = slot.color_space;
        content_timestamp = slot.inference_content_timestamp;
        damage = slot.inference_ddup_damage;
        enqueued_at = transaction_enqueued_at;
        input_region = slot.depth_input_region;
        source_width = source_desc.Width;
        source_height = source_desc.Height;
        mip_levels = source_desc.MipLevels;
        array_size = source_desc.ArraySize;
        source_format = source_desc.Format;
        sample_count = source_desc.SampleDesc.Count;
        sample_quality = source_desc.SampleDesc.Quality;
        root_authority_generation = slot.observed_root_authority_generation;
        region_authority_generation = slot.observed_region_authority_generation;
        browser_authority_epoch = slot.observed_browser_authority_epoch;
        interactive_move_size = slot.observed_interactive_move_size;
        authenticated = frame_id != 0u && content_timestamp && damage && damage->history &&
                        damage->token != 0u && input_region.valid() &&
                        enqueued_at.time_since_epoch().count() != 0 && !interactive_move_size;
      }

      [[nodiscard]] bool route_matches(
        const D3D11_TEXTURE2D_DESC &source_desc,
        const models::input_color_space current_color_space,
        const std::uint64_t current_root_authority_generation,
        const std::uint64_t current_region_authority_generation,
        const std::uint64_t current_browser_authority_epoch,
        const bool current_interactive_move_size
      ) const noexcept {
        return authenticated && frame_id != 0u && content_timestamp && damage &&
               damage->history && damage->token != 0u && input_region.valid() &&
               color_space == current_color_space && source_width == source_desc.Width &&
               source_height == source_desc.Height && mip_levels == source_desc.MipLevels &&
               array_size == source_desc.ArraySize && source_format == source_desc.Format &&
               sample_count == source_desc.SampleDesc.Count &&
               sample_quality == source_desc.SampleDesc.Quality &&
               root_authority_generation == current_root_authority_generation &&
               region_authority_generation == current_region_authority_generation &&
               browser_authority_epoch == current_browser_authority_epoch &&
               interactive_move_size == current_interactive_move_size &&
               !current_interactive_move_size;
      }
    };

    struct latest_v2_lineage_t {
      models::estimate_result estimate;
      matched_frame_slot_t slot;
      UINT source_width = 0u;
      UINT source_height = 0u;
      UINT mip_levels = 0u;
      UINT array_size = 0u;
      DXGI_FORMAT source_format = DXGI_FORMAT_UNKNOWN;
      UINT sample_count = 0u;
      UINT sample_quality = 0u;
      std::uint64_t root_authority_generation = 0u;
      std::uint64_t region_authority_generation = 0u;
      std::uint64_t browser_authority_epoch = 0u;
      bool interactive_move_size = false;
      bool authenticated = false;

      void reset() noexcept {
        *this = {};
      }

      [[nodiscard]] bool route_matches(
        const D3D11_TEXTURE2D_DESC &source_desc,
        const models::input_color_space current_color_space,
        const std::uint64_t current_root_authority_generation,
        const std::uint64_t current_region_authority_generation,
        const std::uint64_t current_browser_authority_epoch,
        const bool current_interactive_move_size
      ) const noexcept {
        return authenticated && estimate.completed_frame_valid &&
               estimate.completed_frame_id == slot.frame_id && estimate.shadow_final_parallax &&
               estimate.shadow_state && estimate.input_region.valid() &&
               estimate.input_region == slot.depth_input_region &&
               slot.inference_content_timestamp && slot.color_space == current_color_space &&
               source_width == source_desc.Width && source_height == source_desc.Height &&
               mip_levels == source_desc.MipLevels && array_size == source_desc.ArraySize &&
               source_format == source_desc.Format && sample_count == source_desc.SampleDesc.Count &&
               sample_quality == source_desc.SampleDesc.Quality &&
               root_authority_generation == current_root_authority_generation &&
               region_authority_generation == current_region_authority_generation &&
               browser_authority_epoch == current_browser_authority_epoch &&
               interactive_move_size == current_interactive_move_size;
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
      if (!slot.texture) {
        return false;
      }
      D3D11_TEXTURE2D_DESC slot_desc {};
      slot.texture->GetDesc(&slot_desc);
      return source_signatures_match(slot_desc, source_desc);
    }

    static void copy_matched_frame_attribution(
      matched_frame_slot_t &destination,
      const matched_frame_slot_t &source
    ) {
      destination.texture.reset();
      destination.srv.reset();
      destination.frame_id = source.frame_id;
      destination.color_space = source.color_space;
      destination.source_timestamp = source.source_timestamp;
      destination.content_timestamp = source.content_timestamp;
      destination.inference_content_timestamp = source.inference_content_timestamp;
      destination.inference_ddup_damage = source.inference_ddup_damage;
      destination.captured_at = source.captured_at;
      destination.window_region = source.window_region;
      destination.depth_video_plan = source.depth_video_plan;
      destination.depth_input_region = source.depth_input_region;
      destination.window_region_observer_status = source.window_region_observer_status;
      destination.window_region_mapping_status = source.window_region_mapping_status;
      destination.live_window_authority_generation =
        source.live_window_authority_generation;
      destination.live_browser_authority_generation =
        source.live_browser_authority_generation;
      destination.live_browser_screen_left = source.live_browser_screen_left;
      destination.live_browser_screen_top = source.live_browser_screen_top;
      destination.live_browser_screen_right = source.live_browser_screen_right;
      destination.live_browser_screen_bottom = source.live_browser_screen_bottom;
      destination.live_browser_geometry_valid_since =
        source.live_browser_geometry_valid_since;
      destination.live_browser_authority_class = source.live_browser_authority_class;
      destination.observed_root_authority_generation =
        source.observed_root_authority_generation;
      destination.observed_region_authority_generation =
        source.observed_region_authority_generation;
      destination.observed_browser_authority_epoch =
        source.observed_browser_authority_epoch;
      destination.observed_interactive_move_size = source.observed_interactive_move_size;
      destination.gpu_undecided_transaction = false;
      destination.pending = false;
    }

    [[nodiscard]] static bool gpu_transaction_class_matches(
      const models::estimate_result &estimate,
      const matched_frame_slot_t &slot
    ) noexcept {
      return estimate.gpu_undecided_completion == slot.gpu_undecided_transaction;
    }

    [[nodiscard]] static bool known_force_infer_completion(
      const models::estimate_result &estimate,
      const matched_frame_slot_t &slot
    ) noexcept {
      return gpu_transaction_class_matches(estimate, slot) &&
             !estimate.gpu_undecided_completion;
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
      const matched_frame_slot_t &slot,
      const std::optional<std::chrono::steady_clock::time_point> &current_content,
      const std::optional<detail::ddup_damage_snapshot_t> &current_damage
    ) const noexcept {
      const auto region = damage_region(slot.depth_input_region);
      if (!region) {
        return detail::host_sbs_ddup_reuse_proof_e::none;
      }
      return detail::classify_host_sbs_ddup_reuse(
        slot.inference_content_timestamp,
        slot.inference_ddup_damage,
        slot.depth_input_region.is_video_region(),
        *region,
        current_content,
        current_damage
      );
    }

    struct matched_motion_damage_t {
      detail::ddup_damage_coverage_t coverage;
    };

    [[nodiscard]] std::optional<matched_motion_damage_t> matched_motion_damage(
      const models::depth_input_region_t &input_region,
      const std::optional<std::chrono::steady_clock::time_point> &baseline_content,
      const std::optional<detail::ddup_damage_snapshot_t> &baseline_damage,
      const std::optional<detail::ddup_damage_snapshot_t> &current_damage
    ) const noexcept {
      if (!baseline_content || !baseline_damage || !current_damage) {
        return std::nullopt;
      }
      const auto region = damage_region(input_region);
      if (!region) {
        return std::nullopt;
      }
      const auto coverage = detail::query_ddup_damage_coverage_between(
        baseline_damage,
        current_damage,
        *region
      );
      return matched_motion_damage_t {coverage};
    }

    /** DDup evidence accumulated from the pixels actually submitted to DAV2, not the last
     * delivery. A GPU-undecided depth branch remains private, so its depth follow-up uses the
     * separately retained opaque anchor. Complete DDup history is route authority only; the
     * device owns the current-vs-history similarity decision.
     */
    [[nodiscard]] std::optional<matched_motion_damage_t> matched_motion_damage(
      const matched_frame_slot_t &slot,
      const std::optional<detail::ddup_damage_snapshot_t> &current_damage
    ) const noexcept {
      return matched_motion_damage(
        slot.depth_input_region,
        slot.inference_content_timestamp,
        slot.inference_ddup_damage,
        current_damage
      );
    }

    [[nodiscard]] std::optional<matched_motion_damage_t> matched_motion_damage(
      const opaque_gpu_followup_anchor_t &anchor,
      const std::optional<detail::ddup_damage_snapshot_t> &current_damage
    ) const noexcept {
      return matched_motion_damage(
        anchor.input_region,
        anchor.content_timestamp,
        anchor.damage,
        current_damage
      );
    }

    /** Re-authenticate the GPU-undecided host tuple after private-copy ROI selection. */
    [[nodiscard]] bool gpu_undecided_candidate_matches(
      matched_frame_slot_t &candidate,
      const D3D11_TEXTURE2D_DESC &source_desc,
      const detail::host_sbs_gpu_undecided_candidate_t &proof
    ) const noexcept {
      const auto current_root_generation = authority_generation(live_window_authority);
      const auto current_region_generation = authority_generation(live_foreground_region);
      if (
        !proof.valid() || candidate.frame_id != proof.current_frame_id || candidate.pending ||
        candidate.observed_interactive_move_size ||
        proof.opaque_followup != gpu_observation_barrier.active() ||
        depth_completion_poll_pending ||
        std::any_of(
          matched_frame_slots.begin(),
          matched_frame_slots.end(),
          [](const matched_frame_slot_t &slot) {
            return slot.pending;
          }
        ) ||
        !candidate.inference_ddup_damage ||
        candidate.inference_ddup_damage->history.get() != proof.damage_history ||
        candidate.inference_ddup_damage->token != proof.current_damage_token ||
        !matched_route_matches_current(
          candidate,
          source_desc,
          candidate.color_space,
          current_root_generation,
          current_region_generation,
          live_browser_authority_epoch,
          interactive_move_size_observed
        )
      ) {
        return false;
      }

      if (proof.opaque_followup) {
        return opaque_gpu_followup_anchor.authenticated &&
               opaque_gpu_followup_anchor.frame_id == proof.baseline_frame_id &&
               opaque_gpu_followup_anchor.damage &&
               opaque_gpu_followup_anchor.damage->history.get() == proof.damage_history &&
               opaque_gpu_followup_anchor.damage->token == proof.baseline_damage_token &&
               opaque_gpu_followup_anchor.input_region == candidate.depth_input_region &&
               opaque_gpu_followup_anchor.color_space == candidate.color_space &&
               detail::host_sbs_gpu_followup_fresh(
                 opaque_gpu_followup_anchor.frame_id,
                 gpu_observation_barrier.conditional_frame_id(),
                 opaque_gpu_followup_anchor.enqueued_at,
                 std::chrono::steady_clock::now()
               ) &&
               opaque_gpu_followup_anchor.route_matches(
                 source_desc,
                 candidate.color_space,
                 current_root_generation,
                 current_region_generation,
                 live_browser_authority_epoch,
                 interactive_move_size_observed
               );
      }

      return latest_v2_lineage.authenticated &&
             latest_v2_lineage.slot.inference_ddup_damage &&
             latest_v2_lineage.slot.inference_ddup_damage->history.get() ==
               proof.damage_history &&
             latest_v2_lineage.slot.inference_ddup_damage->token ==
               proof.baseline_damage_token &&
             latest_v2_lineage.slot.frame_id == proof.baseline_frame_id &&
             latest_v2_lineage.estimate.completed_frame_id == proof.baseline_frame_id &&
             latest_v2_lineage.estimate.input_region ==
               latest_v2_lineage.slot.depth_input_region &&
             candidate.depth_input_region == latest_v2_lineage.slot.depth_input_region &&
             candidate.color_space == latest_v2_lineage.slot.color_space &&
             latest_v2_lineage_route_matches_current(
               source_desc,
               candidate.color_space,
               current_root_generation,
               current_region_generation,
               live_browser_authority_epoch,
               interactive_move_size_observed
             );
    }

    [[nodiscard]] bool matched_route_matches_current(
      matched_frame_slot_t &slot,
      const D3D11_TEXTURE2D_DESC &source_desc,
      const models::input_color_space current_color_space,
      const std::uint64_t current_root_authority_generation,
      const std::uint64_t current_region_authority_generation,
      const std::uint64_t current_browser_authority_epoch,
      const bool current_interactive_move_size
    ) const noexcept {
      if (
        !slot.depth_input_region.valid() ||
        slot.depth_input_region.source_width != source_desc.Width ||
        slot.depth_input_region.source_height != source_desc.Height ||
        !matched_source_signature_matches(slot, source_desc) ||
        slot.color_space != current_color_space ||
        slot.observed_root_authority_generation != current_root_authority_generation ||
        slot.observed_region_authority_generation != current_region_authority_generation ||
        slot.observed_browser_authority_epoch != current_browser_authority_epoch ||
        slot.observed_interactive_move_size != current_interactive_move_size
      ) {
        return false;
      }
      return !slot.depth_input_region.is_video_region() ||
             window_region_authorized_for_render(slot);
    }

    [[nodiscard]] bool latest_v2_lineage_route_matches_current(
      const D3D11_TEXTURE2D_DESC &source_desc,
      const models::input_color_space current_color_space,
      const std::uint64_t current_root_authority_generation,
      const std::uint64_t current_region_authority_generation,
      const std::uint64_t current_browser_authority_epoch,
      const bool current_interactive_move_size
    ) const noexcept {
      return latest_v2_lineage.route_matches(
               source_desc,
               current_color_space,
               current_root_authority_generation,
               current_region_authority_generation,
               current_browser_authority_epoch,
               current_interactive_move_size
             ) &&
             (!latest_v2_lineage.slot.depth_input_region.is_video_region() ||
              window_region_authorized_for_render(latest_v2_lineage.slot));
    }

    [[nodiscard]] bool retain_latest_v2_lineage(
      const models::estimate_result &estimate,
      const matched_frame_slot_t &slot,
      const D3D11_TEXTURE2D_DESC &completed_source_desc,
      const D3D11_TEXTURE2D_DESC &current_source_desc,
      const models::input_color_space current_color_space
    ) {
      const bool completion_route_matches_current =
        slot.inference_content_timestamp && estimate.completed_frame_id == slot.frame_id &&
        estimate.input_region == slot.depth_input_region &&
        estimate.color_space == slot.color_space && estimate.color_space == current_color_space &&
        estimate.input_region.source_width == completed_source_desc.Width &&
        estimate.input_region.source_height == completed_source_desc.Height &&
        source_signatures_match(completed_source_desc, current_source_desc) &&
        slot.observed_root_authority_generation ==
          authority_generation(live_window_authority) &&
        slot.observed_region_authority_generation ==
          authority_generation(live_foreground_region) &&
        slot.observed_browser_authority_epoch == live_browser_authority_epoch &&
        slot.observed_interactive_move_size == interactive_move_size_observed &&
        (!slot.depth_input_region.is_video_region() ||
         (slot.inference_ddup_damage && window_region_authorized_for_render(slot)));
      if (!detail::host_sbs_latest_v2_completion_retention_allowed(
            host_sbs_renderer == models::host_sbs_renderer_e::parallax_v2,
            models::parallax_v2_result_is_authenticated(estimate),
            completion_route_matches_current,
            known_force_infer_completion(estimate, slot)
          )) {
        latest_v2_lineage.reset();
        return false;
      }

      latest_v2_lineage.estimate = estimate;
      // A reused presentation is not a new inference/completion transition. Preserve the completed
      // geometry identity while preventing reset/cut side effects from replaying on every cursor
      // delivery that shares the same DDup desktop-content clock.
      latest_v2_lineage.estimate.inference_enqueued = false;
      latest_v2_lineage.estimate.gpu_undecided_transaction_enqueued = false;
      latest_v2_lineage.estimate.gpu_undecided_completion = false;
      latest_v2_lineage.estimate.input_domain_reset = false;
      latest_v2_lineage.estimate.subtitle_ocr_inference_enqueued = false;
      // The cache renders current color, never the old private color copy. Retain only the scalar
      // attribution needed by the V2 renderer; matched slots own move-only D3D resources.
      copy_matched_frame_attribution(latest_v2_lineage.slot, slot);
      latest_v2_lineage.source_width = completed_source_desc.Width;
      latest_v2_lineage.source_height = completed_source_desc.Height;
      latest_v2_lineage.mip_levels = completed_source_desc.MipLevels;
      latest_v2_lineage.array_size = completed_source_desc.ArraySize;
      latest_v2_lineage.source_format = completed_source_desc.Format;
      latest_v2_lineage.sample_count = completed_source_desc.SampleDesc.Count;
      latest_v2_lineage.sample_quality = completed_source_desc.SampleDesc.Quality;
      latest_v2_lineage.root_authority_generation =
        slot.observed_root_authority_generation;
      latest_v2_lineage.region_authority_generation =
        slot.observed_region_authority_generation;
      latest_v2_lineage.browser_authority_epoch =
        slot.observed_browser_authority_epoch;
      latest_v2_lineage.interactive_move_size =
        slot.observed_interactive_move_size;
      latest_v2_lineage.authenticated = true;
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
        if (!input_is_linear &&
            sbs_intermediate_texture &&
            current_desc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
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
      p010_y_mrt_targets_checked = false;
      p010_y_mrt_targets_qualified = false;
      sbs_intermediate_is_linear = false;
      matched_presentation_cache.reset();
      BOOST_LOG(info) << "Host SBS intermediate storage finalized as "sv
                      << (input_is_linear ? "FP16 linear-capable."sv : "BGRA8 SDR."sv);
      return true;
    }

    bool qualify_p010_y_mrt_targets() {
      if (p010_y_mrt_targets_checked) {
        return p010_y_mrt_targets_qualified;
      }
      p010_y_mrt_targets_checked = true;
      p010_y_mrt_targets_qualified = false;
      if (!sbs_intermediate_rtv || !out_Y_or_YUV_rtv) {
        return false;
      }

      ID3D11RenderTargetView *requested[] = {
        sbs_intermediate_rtv.get(),
        out_Y_or_YUV_rtv.get(),
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
        if (!slot.pending && &slot != reserved_render_slot) {
          return &slot;
        }
      }
      return nullptr;
    }

    matched_frame_slot_t *find_pending_matched_slot(std::uint64_t frame_id) {
      for (auto &slot : matched_frame_slots) {
        if (slot.pending && slot.frame_id == frame_id) {
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
      video_dom::snapshot_ptr browser_snapshot;
      std::optional<browser_live_authority_key_t> browser_authority;
      std::uint64_t browser_authority_epoch = 0u;
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
        )
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
             left.monitor == right.monitor &&
             width(left.client_screen_rect) == width(right.client_screen_rect) &&
             height(left.client_screen_rect) == height(right.client_screen_rect);
    }

    void clear_cached_roi_output() noexcept {
      if (!matched_presentation_cache.is_video_region()) {
        return;
      }
      matched_presentation_cache.reset();
    }

    live_window_authority_observation_t observe_live_window_authority(
      const D3D11_TEXTURE2D_DESC &source_desc,
      const std::optional<std::chrono::steady_clock::time_point> &content_timestamp
    ) {
      foreground_window::snapshot_t observed;
      if (content_timestamp) {
        observed = foreground_window_tracker.update(foreground_window::sample());
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
          for (const auto &slot : matched_frame_slots) {
            if (slot.pending && slot.depth_input_region.is_video_region()) {
              // Consuming this exact-source ROI completion does not enqueue a replacement. Ask
              // the retained-source owner for one more conversion so full-source depth starts
              // even if USER32 stops producing desktop damage during the modal move loop.
              depth_authority_reprocess_pending = true;
              break;
            }
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
      DXGI_OUTPUT_DESC output_desc {};
      if (
        next_root &&
        display && display->output &&
        display->display_rotation == DXGI_MODE_ROTATION_IDENTITY &&
        static_cast<std::uint32_t>(display->width) == source_desc.Width &&
        static_cast<std::uint32_t>(display->height) == source_desc.Height &&
        SUCCEEDED(display->output->GetDesc(&output_desc))
      ) {
        const foreground_window::capture_target_t target {
          .screen_rect = {
            output_desc.DesktopCoordinates.left,
            output_desc.DesktopCoordinates.top,
            output_desc.DesktopCoordinates.right,
            output_desc.DesktopCoordinates.bottom,
          },
          .width = source_desc.Width,
          .height = source_desc.Height,
          .monitor = reinterpret_cast<std::uintptr_t>(output_desc.Monitor),
          .identity_orientation = true,
        };
        // Liveness follows the current root/client geometry, not the content-causality gate. A
        // move must revoke an old positioned completion immediately, while the later private-copy
        // path still requires geometry_valid_since <= content_timestamp before authorizing a new
        // foreground ROI. This separation preserves the position-independent DAV2 domain.
        if (foreground_window::map_to_capture(observed, target)) {
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
      const bool analysis_authority_changed =
        live_window_authority &&
        (!next_root ||
         !same_window_analysis_identity_shape(*live_window_authority, *next_root));
      if (root_changed || region_changed) {
        clear_cached_roi_output();
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
      return {
        .foreground = observed,
        .browser_snapshot = browser_snapshot,
        .browser_authority = live_browser_authority,
        .browser_authority_epoch = live_browser_authority_epoch,
      };
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
      const matched_frame_slot_t &slot
    ) const noexcept {
      if (!slot.window_region || !window_region_matches_live_authority(*slot.window_region) ||
          slot.live_window_authority_generation == 0u) {
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
      matched_frame_slot_t &slot,
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


    std::optional<sbs_debug::window_region_snapshot>
    capture_browser_window_region(
      const D3D11_TEXTURE2D_DESC &source_desc,
      const video_dom::snapshot_ptr &observed,
      const std::uint64_t frame_id,
      const std::optional<std::chrono::steady_clock::time_point> &content_timestamp,
      const std::chrono::steady_clock::time_point captured_at,
      video_dom::status_e &video_status,
      video_dom::mapping_status_e &mapping_status
    ) const {
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

      DXGI_OUTPUT_DESC output_desc {};
      if (!display->output || FAILED(display->output->GetDesc(&output_desc))) {
        mapping_status = video_dom::mapping_status_e::invalid_capture_rect;
        return std::nullopt;
      }
      const video_dom::rect_t desktop_rect {
        output_desc.DesktopCoordinates.left,
        output_desc.DesktopCoordinates.top,
        output_desc.DesktopCoordinates.right,
        output_desc.DesktopCoordinates.bottom,
      };
      const auto mapped = video_dom::map_screen_rect_to_capture(
        observed->screen_rect,
        desktop_rect,
        source_desc.Width,
        source_desc.Height,
        video_dom::rotation_e::identity
      );
      mapping_status = mapped.status;
      if (!mapped) {
        return std::nullopt;
      }

      const auto heartbeat_age = std::chrono::duration_cast<std::chrono::milliseconds>(
        captured_at - observed->received_at
      );
      const auto geometry_continuity = std::chrono::duration_cast<std::chrono::milliseconds>(
        captured_at - observed->geometry_valid_since
      );
      const auto source_content_age = std::chrono::duration_cast<std::chrono::milliseconds>(
        captured_at - *content_timestamp
      );
      return sbs_debug::window_region_snapshot {
        .authority_kind = sbs_debug::window_region_authority_kind_e::chromium_video,
        .matched_frame_id = frame_id,
        .source_width = source_desc.Width,
        .source_height = source_desc.Height,
        .left = mapped.capture_pixels.left,
        .top = mapped.capture_pixels.top,
        .right = mapped.capture_pixels.right,
        .bottom = mapped.capture_pixels.bottom,
        .hwnd = observed->window,
        .process_id = observed->process_id,
        .document_id = observed->document_id,
        .video_id = observed->video_id,
        .generation = observed->generation,
        .latest_observation_age_ms_at_capture = static_cast<std::uint32_t>(
          std::clamp<std::int64_t>(
            heartbeat_age.count(),
            0,
            std::numeric_limits<std::uint32_t>::max()
          )
        ),
        .maximum_observation_age_ms = static_cast<std::uint32_t>(maximum_age.count()),
        .geometry_continuity_ms_at_capture = static_cast<std::uint64_t>(
          std::max<std::int64_t>(geometry_continuity.count(), 0)
        ),
        .source_content_age_ms_at_capture = static_cast<std::uint64_t>(
          std::max<std::int64_t>(source_content_age.count(), 0)
        ),
      };
    }

    std::optional<sbs_debug::window_region_snapshot>
    capture_foreground_window_region(
      const D3D11_TEXTURE2D_DESC &source_desc,
      const std::uint64_t frame_id,
      const std::optional<std::chrono::steady_clock::time_point> &content_timestamp,
      const std::chrono::steady_clock::time_point captured_at,
      const foreground_window::snapshot_t &snapshot,
      std::string_view &observer_status,
      std::string_view &mapping_status
    ) {
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

      DXGI_OUTPUT_DESC output_desc {};
      if (!display->output || FAILED(display->output->GetDesc(&output_desc))) {
        mapping_status = foreground_window::mapping_status_name(
          foreground_window::mapping_status_e::invalid_capture_target
        );
        return std::nullopt;
      }
      const foreground_window::capture_target_t target {
        .screen_rect = {
          output_desc.DesktopCoordinates.left,
          output_desc.DesktopCoordinates.top,
          output_desc.DesktopCoordinates.right,
          output_desc.DesktopCoordinates.bottom,
        },
        .width = source_desc.Width,
        .height = source_desc.Height,
        .monitor = reinterpret_cast<std::uintptr_t>(output_desc.Monitor),
        .identity_orientation =
          display->display_rotation == DXGI_MODE_ROTATION_IDENTITY,
      };
      const auto mapped = foreground_window::map_to_capture(snapshot, target);
      mapping_status = foreground_window::mapping_status_name(mapped.status);
      if (!mapped) {
        return std::nullopt;
      }

      const auto observation_age = std::chrono::duration_cast<std::chrono::milliseconds>(
        captured_at - snapshot.observed_at
      );
      const auto geometry_continuity = std::chrono::duration_cast<std::chrono::milliseconds>(
        captured_at - snapshot.geometry_valid_since
      );
      const auto source_content_age = std::chrono::duration_cast<std::chrono::milliseconds>(
        captured_at - *content_timestamp
      );
      return sbs_debug::window_region_snapshot {
        .authority_kind = sbs_debug::window_region_authority_kind_e::foreground_client,
        .matched_frame_id = frame_id,
        .source_width = source_desc.Width,
        .source_height = source_desc.Height,
        .left = mapped.capture_pixels.left,
        .top = mapped.capture_pixels.top,
        .right = mapped.capture_pixels.right,
        .bottom = mapped.capture_pixels.bottom,
        .hwnd = snapshot.window,
        .process_id = snapshot.process_id,
        .document_id = 0,
        .video_id = 0,
        .generation = snapshot.generation,
        .latest_observation_age_ms_at_capture = static_cast<std::uint32_t>(
          std::clamp<std::int64_t>(
            observation_age.count(),
            0,
            std::numeric_limits<std::uint32_t>::max()
          )
        ),
        .maximum_observation_age_ms = static_cast<std::uint32_t>(maximum_age.count()),
        .geometry_continuity_ms_at_capture = static_cast<std::uint64_t>(
          std::max<std::int64_t>(geometry_continuity.count(), 0)
        ),
        .source_content_age_ms_at_capture = static_cast<std::uint64_t>(
          std::max<std::int64_t>(source_content_age.count(), 0)
        ),
      };
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

    void publish_window_region_transition(const matched_frame_slot_t &slot) {
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
              << "; scene cuts and scene center are region-local."sv;
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

    bool copy_matched_frame(
      ID3D11Texture2D *source,
      matched_frame_slot_t &slot,
      std::uint64_t frame_id,
      models::input_color_space color_space,
      std::optional<std::chrono::steady_clock::time_point> source_timestamp,
      std::optional<std::chrono::steady_clock::time_point> content_timestamp,
      std::optional<detail::ddup_damage_snapshot_t> ddup_damage,
      const live_window_authority_observation_t &pre_copy_authority,
      sbs_gpu_timer_slot_t *gpu_timer
    ) {
      if (!source) {
        return false;
      }

      D3D11_TEXTURE2D_DESC source_desc {};
      source->GetDesc(&source_desc);
      bool recreate = !slot.texture || !slot.srv;
      if (!recreate) {
        D3D11_TEXTURE2D_DESC slot_desc {};
        slot.texture->GetDesc(&slot_desc);
        recreate = slot_desc.Width != source_desc.Width ||
                   slot_desc.Height != source_desc.Height ||
                   slot_desc.MipLevels != source_desc.MipLevels ||
                   slot_desc.ArraySize != source_desc.ArraySize ||
                   slot_desc.Format != source_desc.Format ||
                   slot_desc.SampleDesc.Count != source_desc.SampleDesc.Count ||
                   slot_desc.SampleDesc.Quality != source_desc.SampleDesc.Quality;
      }

      if (recreate) {
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
        slot.texture = std::move(texture);
        slot.srv = std::move(srv);
      }

      const auto captured_at = std::chrono::steady_clock::now();
      mark_sbs_matched_copy_start(gpu_timer);
      device_ctx->CopyResource(slot.texture.get(), source);
      mark_sbs_matched_copy_end(gpu_timer, true);
      // Recheck after the copy as the causal authority for this exact matched image. The earlier
      // per-convert observation breaks continuity across busy periods; this second observation
      // closes a focus/move/resize race between admission and CopyResource. During an already
      // authenticated native move/resize, ROI authority has been withdrawn and this frame is
      // forced to full-source analysis, so repeating the synchronous USER32/DWM sample cannot
      // improve attribution. If the move starts after the first observation, this recheck still
      // runs and detects it.
      const bool pre_copy_interactive_move_size =
        pre_copy_authority.foreground.status ==
        foreground_window::status_e::interactive_move_size;
      const auto copied_authority = pre_copy_interactive_move_size ?
                                      pre_copy_authority :
                                      observe_live_window_authority(
                                        source_desc,
                                        content_timestamp
                                      );
      const auto &copied_foreground_snapshot = copied_authority.foreground;
      // A pre-drag ROI completion may still own the estimator after USER32 stops producing desktop
      // damage. Full-source analysis is independent of window geometry, so submit these exact
      // retained pixels once without letting post-drag ROI causality block completion retirement.
      const bool preexisting_force_full_source =
        depth_authority_reprocess_pending || pre_copy_interactive_move_size;
      const auto attributed_at = std::chrono::steady_clock::now();
      slot.frame_id = frame_id;
      slot.color_space = color_space;
      slot.source_timestamp = source_timestamp;
      slot.content_timestamp = content_timestamp;
      slot.inference_content_timestamp = content_timestamp;
      slot.inference_ddup_damage = std::move(ddup_damage);
      // Production uses source age to bound V2 output repetition; this is no longer diagnostics-
      // only metadata. The aggregate age counters below remain gated by diagnostics_enabled.
      slot.captured_at = captured_at;
      slot.window_region.reset();
      slot.window_region_observer_status = "not-observed";
      slot.window_region_mapping_status = "not-mapped";
      slot.live_browser_geometry_valid_since = {};
      if (pre_copy_interactive_move_size) {
        slot.window_region_observer_status = foreground_window::status_name(
          foreground_window::status_e::interactive_move_size
        );
      }
      video_dom::status_e browser_status = video_dom::status_e::stopped;
      video_dom::mapping_status_e browser_mapping_status =
        video_dom::mapping_status_e::invalid_video_rect;
      const bool foreground_causal_wait =
        live_foreground_region &&
        foreground_window::requires_full_source_causal_fallback(
          copied_foreground_snapshot,
          *live_foreground_region,
          content_timestamp
        );
      const bool force_full_source =
        preexisting_force_full_source || foreground_causal_wait;
      if (foreground_causal_wait) {
        // The copied desktop content predates this same-size programmatic move. Desktop
        // Duplication may never publish another content timestamp after the move, so submit this
        // exact frame through full-source V2 instead of starving depth indefinitely.
        slot.window_region_observer_status = "causal-wait-full-source";
        slot.window_region_mapping_status = "not-mapped";
      }
      auto browser_region = force_full_source ?
                              std::optional<sbs_debug::window_region_snapshot> {} :
                              capture_browser_window_region(
                                source_desc,
                                copied_authority.browser_snapshot,
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
        slot.window_region = std::move(browser_region);
        slot.window_region_observer_status = video_dom::status_name(browser_status);
        slot.window_region_mapping_status =
          video_dom::mapping_status_name(browser_mapping_status);
      } else if (!force_full_source) {
        std::string_view foreground_observer_status;
        std::string_view foreground_mapping_status;
        slot.window_region = capture_foreground_window_region(
          source_desc,
          frame_id,
          content_timestamp,
          attributed_at,
          copied_foreground_snapshot,
          foreground_observer_status,
          foreground_mapping_status
        );
        slot.window_region_observer_status = foreground_observer_status;
        slot.window_region_mapping_status = foreground_mapping_status;
      }
      const bool retry_foreground_after_browser =
        slot.window_region &&
        slot.window_region->authority_kind ==
          sbs_debug::window_region_authority_kind_e::chromium_video &&
        !(
          slot.window_region->left == 0 && slot.window_region->top == 0 &&
          slot.window_region->right == slot.window_region->source_width &&
          slot.window_region->bottom == slot.window_region->source_height
        );
      select_depth_input_region(
        slot,
        source_desc,
        !retry_foreground_after_browser
      );
      if (!slot.depth_input_region.is_video_region() && retry_foreground_after_browser) {
        std::string_view foreground_observer_status;
        std::string_view foreground_mapping_status;
        slot.window_region = capture_foreground_window_region(
          source_desc,
          frame_id,
          content_timestamp,
          attributed_at,
          copied_foreground_snapshot,
          foreground_observer_status,
          foreground_mapping_status
        );
        slot.window_region_observer_status = foreground_observer_status;
        slot.window_region_mapping_status = foreground_mapping_status;
        select_depth_input_region(slot, source_desc);
      }
      slot.live_window_authority_generation =
        slot.window_region && window_region_matches_live_authority(*slot.window_region) ?
          (
            slot.window_region->authority_kind ==
                sbs_debug::window_region_authority_kind_e::chromium_video ?
              live_window_authority->generation :
              live_foreground_region->generation
          ) :
          0u;
      slot.live_browser_authority_generation =
        slot.window_region &&
            slot.window_region->authority_kind ==
              sbs_debug::window_region_authority_kind_e::chromium_video ?
          copied_authority.browser_authority_epoch :
          0u;
      if (slot.live_browser_authority_generation != 0u && video_dom_lease) {
        if (copied_authority.browser_authority) {
          slot.live_browser_screen_left =
            copied_authority.browser_authority->screen_rect.left;
          slot.live_browser_screen_top =
            copied_authority.browser_authority->screen_rect.top;
          slot.live_browser_screen_right =
            copied_authority.browser_authority->screen_rect.right;
          slot.live_browser_screen_bottom =
            copied_authority.browser_authority->screen_rect.bottom;
          slot.live_browser_geometry_valid_since =
            copied_authority.browser_authority->geometry_valid_since;
          slot.live_browser_authority_class =
            video_dom::geometry_authority_class(
              copied_authority.browser_authority->status
            );
        }
      }
      // Capture the complete live route epoch even for a full-source frame. A later unchanged-
      // content reuse may retain that geometry only while the same focus/ROI/move authority is
      // still live; otherwise a new matched copy must select the current route.
      slot.observed_root_authority_generation =
        authority_generation(live_window_authority);
      slot.observed_region_authority_generation =
        authority_generation(live_foreground_region);
      slot.observed_browser_authority_epoch = copied_authority.browser_authority_epoch;
      slot.observed_interactive_move_size = interactive_move_size_observed;
      if (slot.depth_input_region.is_video_region() &&
          slot.live_window_authority_generation == 0u) {
        slot.window_region.reset();
        slot.window_region_observer_status = "not-observed";
        slot.window_region_mapping_status = "not-mapped";
        select_depth_input_region(slot, source_desc);
      }
      if (
        !sbs_telemetry_input_domain_valid ||
        !slot.depth_input_region.same_analysis_domain(sbs_telemetry_input_region) ||
        slot.color_space != sbs_telemetry_input_color_space
      ) {
        // Stop an old-domain staging readback from publishing while the renderer is deliberately
        // flat and waiting for this newly observed ROI/full/transfer domain. The first completion
        // in the new domain establishes the new cut-counter baseline.
        sbs_telemetry_input_region = slot.depth_input_region;
        sbs_telemetry_input_color_space = slot.color_space;
        sbs_telemetry_input_domain_valid = true;
        sbs_telemetry_min_frame_id = frame_id;
        sbs_telemetry_has_sample = false;
        sbs_telemetry_last_hard_cut_count = 0;
      }
      slot.gpu_undecided_transaction = false;
      slot.pending = false;
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

    bool render_sbs_debug_geometry(
      ID3D11ShaderResourceView *source,
      ID3D11ShaderResourceView *warp_depth,
      ID3D11ShaderResourceView *parallax_state,
      ID3D11Buffer *constants
    ) {
      if (!source || !warp_depth || !parallax_state || !constants ||
          !ensure_sbs_debug_geometry_resources()) {
        return false;
      }

      ID3D11SamplerState *sampler = sampler_linear.get();
      device_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      device_ctx->VSSetShader(sbs_reprojection_vs.get(), nullptr, 0);
      device_ctx->RSSetViewports(1, &sbs_viewport);
      device_ctx->OMSetBlendState(blend_disable.get(), nullptr, 0xFFFFFFFFu);
      device_ctx->PSSetSamplers(0, 1, &sampler);
      device_ctx->PSSetConstantBuffers(2, 1, &constants);

      device_ctx->OMSetRenderTargets(
        1,
        sbs_debug_mapping_rtv.GetAddressOf(),
        nullptr
      );
      device_ctx->PSSetShader(sbs_debug_v2_mapping_ps.Get(), nullptr, 0);
      ID3D11ShaderResourceView *mapping_inputs[] = {
        source,
        warp_depth,
        parallax_state,
        nullptr,
        nullptr,
        nullptr,
      };
      device_ctx->PSSetShaderResources(
        0,
        (UINT) std::size(mapping_inputs),
        mapping_inputs
      );
      device_ctx->Draw(3, 0);

      device_ctx->OMSetRenderTargets(
        1,
        sbs_debug_mask_rtv.GetAddressOf(),
        nullptr
      );
      device_ctx->PSSetShader(sbs_debug_v2_mask_ps.Get(), nullptr, 0);
      ID3D11ShaderResourceView *mask_inputs[] = {
        source,
        warp_depth,
        parallax_state,
        nullptr,
        nullptr,
        nullptr,
      };
      device_ctx->PSSetShaderResources(
        0,
        (UINT) std::size(mask_inputs),
        mask_inputs
      );
      device_ctx->Draw(3, 0);

      ID3D11RenderTargetView *null_rtv = nullptr;
      ID3D11ShaderResourceView *null_ps_srvs[] = {
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
      };
      device_ctx->OMSetRenderTargets(1, &null_rtv, nullptr);
      device_ctx->PSSetShaderResources(
        0,
        (UINT) std::size(null_ps_srvs),
        null_ps_srvs
      );
      return true;
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
      matched_stats_gpu_followup_expired = 0;
      matched_stats_gpu_followup_host_rejected = 0;
      matched_stats_gpu_followup_force_fallbacks = 0;
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

    int init_output(
      ID3D11Texture2D *frame_texture,
      int width,
      int height,
      int sbs_mode_param = ::video::SBS_OFF,
      const config::video_t::sbs_t &settings = {},
      safe::mail_raw_t::event_t<int> depth_status_event = {},
      std::shared_ptr<safe::event_t<bool>> depth_pipeline_ready_event = {},
      safe::mail_raw_t::event_t<::video::sbs_telemetry_snapshot_t> telemetry_event = {},
      std::shared_ptr<::video::sbs_telemetry_subscription_t> telemetry_subscription = {},
      std::uint32_t telemetry_generation = 0,
      std::shared_ptr<std::atomic<bool>> sbs_debug_dump_request = {},
      bool rgb_only = false
    ) {
      if (frame_texture) {
        // The underlying frame pool owns the texture, so we must reference it for ourselves.
        frame_texture->AddRef();
        output_texture.reset(frame_texture);
      } else if (!rgb_only) {
        BOOST_LOG(error) << "Missing encoder output texture."sv;
        return -1;
      }
      sbs_mode = sbs_mode_param;
      sbs_config = settings;
      host_sbs_renderer = models::host_sbs_renderer_e::awaiting_v2;
      diagnostics_enabled = config::sunshine.diagnostics_enabled;
      sbs_depth_status_event = std::move(depth_status_event);
      sbs_depth_pipeline_ready_event = std::move(depth_pipeline_ready_event);
      sbs_telemetry_event = std::move(telemetry_event);
      sbs_telemetry_subscription = std::move(telemetry_subscription);
      sbs_dumper.set_button_request(std::move(sbs_debug_dump_request));
      sbs_flat_identity_ps.reset();
      sbs_reprojection_v2_live_ps.reset();
      sbs_reprojection_v2_p010_y_ps.reset();
      p010_y_mrt_targets_checked = false;
      p010_y_mrt_targets_qualified = false;
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
      sbs_telemetry_generation = telemetry_generation;
      sbs_telemetry_sequence = 0;
      sbs_telemetry_last_sampled_frame_id = 0;
      sbs_telemetry_min_frame_id = 0;
      sbs_telemetry_last_hard_cut_count = 0;
      sbs_telemetry_has_sample = false;
      sbs_telemetry_input_region = {};
      sbs_telemetry_input_color_space = models::input_color_space::srgb;
      sbs_telemetry_input_domain_valid = false;
      sbs_telemetry_producer_failure_published = false;
      sbs_telemetry_last_copy = {};
      adaptive_motion_route_state.reset();
      adaptive_hold_cadence.reset();
      adaptive_ocr_cadence.reset();
      gpu_observation_barrier.reset();
      opaque_gpu_followup_anchor.reset();
      matched_frame_slots = {};
      depth_analysis_generation_tracker = {};
      foreground_window_tracker.reset();
      live_window_authority.reset();
      live_foreground_region.reset();
      live_browser_authority.reset();
      live_browser_authority_epoch = 0u;
      interactive_move_size_observed = false;
      browser_roi_not_before = {};
      last_window_region.reset();
      last_window_region_observer_status.reset();
      last_window_region_mapping_status.reset();
      sbs_frame_sequence = 0;
      latest_v2_lineage.reset();
      content_reuse_refresh.reset();
      approximate_reuse_since_enqueue =
        detail::host_sbs_approximate_reuse_provider_e::none;
      matched_presentation_cache.reset();
      depth_completion_poll_pending = false;
      depth_authority_reprocess_pending = false;
      sbs_intermediate_is_linear = false;
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
      const bool sbs_on = sbs_mode != ::video::SBS_OFF;
#if !defined(SUNSHINE_TESTS)
      if (sbs_on) {
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
      const bool downscaling = display->width > width || display->height > height;
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
      if (sbs_on && !host_sbs_rejection.empty()) {
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
      if (rgb_only) {
        create_pixel_shader_helper(cursor_ps_hlsl, rgb_present_ps);
        create_pixel_shader_helper(
          rgb_present_linear_to_srgb_ps_hlsl,
          rgb_present_linear_to_srgb_ps
        );
        create_pixel_shader_helper(
          rgb_present_srgb_to_linear_ps_hlsl,
          rgb_present_srgb_to_linear_ps
        );
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
      if (!sbs_reprojection_vs_hlsl ||
          FAILED(status = device->CreateVertexShader(
                   sbs_reprojection_vs_hlsl->data(),
                   sbs_reprojection_vs_hlsl->size(),
                   nullptr,
                   &sbs_reprojection_vs
                 ))) {
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

      if (!rgb_only) {
        switch (format) {
          case DXGI_FORMAT_NV12:
            // Semi-planar 8-bit YUV 4:2:0
            create_vertex_shader_helper(convert_yuv420_planar_y_vs_hlsl, convert_Y_or_YUV_vs);
            create_pixel_shader_helper(convert_yuv420_planar_y_ps_hlsl, convert_Y_or_YUV_ps);
            create_pixel_shader_helper(convert_yuv420_planar_y_ps_linear_hlsl, convert_Y_or_YUV_fp16_ps);
            if (downscaling) {
              create_vertex_shader_helper(convert_yuv420_packed_uv_type0s_vs_hlsl, convert_UV_vs);
              create_pixel_shader_helper(convert_yuv420_packed_uv_type0s_ps_hlsl, convert_UV_ps);
              create_pixel_shader_helper(convert_yuv420_packed_uv_type0s_ps_linear_hlsl, convert_UV_fp16_ps);
            } else {
              create_vertex_shader_helper(convert_yuv420_packed_uv_type0_vs_hlsl, convert_UV_vs);
              create_pixel_shader_helper(convert_yuv420_packed_uv_type0_ps_hlsl, convert_UV_ps);
              create_pixel_shader_helper(convert_yuv420_packed_uv_type0_ps_linear_hlsl, convert_UV_fp16_ps);
            }
            break;

          case DXGI_FORMAT_P010:
            // Semi-planar 16-bit YUV 4:2:0, 10 most significant bits store the value
            create_vertex_shader_helper(convert_yuv420_planar_y_vs_hlsl, convert_Y_or_YUV_vs);
            create_pixel_shader_helper(convert_yuv420_planar_y_ps_hlsl, convert_Y_or_YUV_ps);
            if (display_is_hdr) {
              create_pixel_shader_helper(convert_yuv420_planar_y_ps_perceptual_quantizer_hlsl, convert_Y_or_YUV_fp16_ps);
            } else {
              create_pixel_shader_helper(convert_yuv420_planar_y_ps_linear_hlsl, convert_Y_or_YUV_fp16_ps);
            }
            if (downscaling) {
              create_vertex_shader_helper(convert_yuv420_packed_uv_type0s_vs_hlsl, convert_UV_vs);
              create_pixel_shader_helper(convert_yuv420_packed_uv_type0s_ps_hlsl, convert_UV_ps);
              if (display_is_hdr) {
                create_pixel_shader_helper(convert_yuv420_packed_uv_type0s_ps_perceptual_quantizer_hlsl, convert_UV_fp16_ps);
              } else {
                create_pixel_shader_helper(convert_yuv420_packed_uv_type0s_ps_linear_hlsl, convert_UV_fp16_ps);
              }
            } else {
              create_vertex_shader_helper(convert_yuv420_packed_uv_type0_vs_hlsl, convert_UV_vs);
              create_pixel_shader_helper(convert_yuv420_packed_uv_type0_ps_hlsl, convert_UV_ps);
              if (display_is_hdr) {
                create_pixel_shader_helper(convert_yuv420_packed_uv_type0_ps_perceptual_quantizer_hlsl, convert_UV_fp16_ps);
              } else {
                create_pixel_shader_helper(convert_yuv420_packed_uv_type0_ps_linear_hlsl, convert_UV_fp16_ps);
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

      auto out_width = width;
      auto out_height = height;
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
        initialize_sbs_gpu_timers();
        BOOST_LOG(info)
          << "Host SBS warp: authenticated Depth Coordinate V2 contractive renderer, fixed pop "
          << sbs_config.pop_strength << ".";
        sbs_viewport = {0.0f, 0.0f, out_width_f, out_height_f, 0.0f, 1.0f};
        // DDup starts with UNKNOWN and exposes the actual format on its first real frame. Defer
        // allocation in that case; known WGC/DDup formats can be finalized now.
        if (display->capture_format != DXGI_FORMAT_UNKNOWN &&
            !ensure_sbs_intermediate_storage(
              display->capture_format == DXGI_FORMAT_R16G16B16A16_FLOAT
            )) {
          return -1;
        }
      }

      if (rgb_only) {
        // Local AR presents RGB directly. It needs the common capture/SBS resources above, but
        // no encoder output texture, chroma shaders, planar RTVs, or YUV conversion constants.
        return 0;
      }

      out_Y_or_YUV_viewport = {offsetX, offsetY, out_width_f, out_height_f, 0.0f, 1.0f};  // Y plane

      out_UV_viewport = {offsetX / 2, offsetY / 2, out_width_f / 2, out_height_f / 2, 0.0f, 1.0f};

      float subsample_offset_in[16 / sizeof(float)] {1.0f / (float) out_width_f, 1.0f / (float) out_height_f};  // aligned to 16-byte
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

      auto create_rtv = [&](auto &rt, DXGI_FORMAT rt_format) -> bool {
        D3D11_RENDER_TARGET_VIEW_DESC rtv_desc = {};
        rtv_desc.Format = rt_format;
        rtv_desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;

        auto status = device->CreateRenderTargetView(output_texture.get(), &rtv_desc, &rt);
        if (FAILED(status)) {
          BOOST_LOG(error) << "Failed to create render target view: " << util::log_hex(status);
          return false;
        }

        return true;
      };

      // Create Y/YUV render target view
      if (!create_rtv(out_Y_or_YUV_rtv, rtv_Y_or_YUV_format)) {
        return -1;
      }

      // Create UV render target view if needed
      if (rtv_UV_format != DXGI_FORMAT_UNKNOWN && !create_rtv(out_UV_rtv, rtv_UV_format)) {
        return -1;
      }

      // NV12 and P010 support native RTV clears. Initialize aspect-ratio padding once.
      const float y_black[] = {output_black_y, output_black_y, output_black_y, output_black_y};
      device_ctx->ClearRenderTargetView(out_Y_or_YUV_rtv.get(), y_black);
      if (out_UV_rtv) {
        const float uv_black[] = {
          output_neutral_uv,
          output_neutral_uv,
          output_neutral_uv,
          output_neutral_uv,
        };
        device_ctx->ClearRenderTargetView(out_UV_rtv.get(), uv_black);
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
      return init_output(
        nullptr,
        width,
        height,
        sbs_mode_param,
        settings,
        {},
        std::move(depth_pipeline_ready_event),
        {},
        {},
        0,
        {},
        true
      );
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

      status = dxgi->SetGPUThreadPriority(0x4000001E);
      if (FAILED(status)) {
        BOOST_LOG(info) << "Failed to request absoloute encoding GPU thread priority. Trying relative priority.";
        status = dxgi->SetGPUThreadPriority(7);
        if (FAILED(status)) {
          BOOST_LOG(warning) << "Failed to request relative encoding GPU thread priority. Please run application as administrator for optimal performance.";
        } else {
          BOOST_LOG(info) << "Relative encoding GPU thread priority request success.";
        }
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
      // Used to determine if the underlying texture changes.
      // Not safe for actual use by the encoder!
      texture2d_t::const_pointer capture_texture_p;

      texture2d_t encoder_texture;
      shader_res_t encoder_input_res;
      keyed_mutex_t encoder_mutex;

      std::weak_ptr<const platf::img_t> img_weak;

      void reset() {
        capture_texture_p = nullptr;
        encoder_texture.reset();
        encoder_input_res.reset();
        encoder_mutex.reset();
        img_weak.reset();
      }
    };

    int initialize_image_context(const img_d3d_t &img, encoder_img_ctx_t &img_ctx) {
      // If we've already opened the shared texture, we're done
      if (img_ctx.encoder_texture && img.capture_texture.get() == img_ctx.capture_texture_p) {
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
      status = device1->OpenSharedResource1(img.encoder_texture_handle, __uuidof(ID3D11Texture2D), (void **) &img_ctx.encoder_texture);
      if (FAILED(status)) {
        BOOST_LOG(error) << "Failed to open shared image texture [0x"sv << util::hex(status).to_string_view() << ']';
        return -1;
      }

      // Get the keyed mutex to synchronize with the capture code
      status = img_ctx.encoder_texture->QueryInterface(__uuidof(IDXGIKeyedMutex), (void **) &img_ctx.encoder_mutex);
      if (FAILED(status)) {
        BOOST_LOG(error) << "Failed to query IDXGIKeyedMutex [0x"sv << util::hex(status).to_string_view() << ']';
        return -1;
      }

      // Create the SRV for the encoder texture
      status = device->CreateShaderResourceView(img_ctx.encoder_texture.get(), nullptr, &img_ctx.encoder_input_res);
      if (FAILED(status)) {
        BOOST_LOG(error) << "Failed to create shader resource view for encoding [0x"sv << util::hex(status).to_string_view() << ']';
        return -1;
      }

      img_ctx.capture_texture_p = img.capture_texture.get();

      img_ctx.img_weak = img.weak_from_this();

      return 0;
    }

    ::video::color_t *color_p;

    buf_t subsample_offset;
    buf_t color_matrix;
    buf_t sdr_color_transform;
    float output_black_y = 0.0f;
    float output_neutral_uv = 0.5f;

    blend_t blend_disable;
    sampler_state_t sampler_linear;

    render_target_t out_Y_or_YUV_rtv;
    render_target_t out_UV_rtv;

    // d3d_img_t::id -> encoder_img_ctx_t
    // These store the encoder textures for each img_t that passes through
    // convert(). We can't store them in the img_t itself because it can be shared
    // among multiple encode devices (and therefore multiple ID3D11Devices).
    std::map<uint32_t, encoder_img_ctx_t> img_ctx_map;

    std::shared_ptr<display_base_t> display;
    bool display_is_hdr = false;

    vs_t convert_Y_or_YUV_vs;
    ps_t convert_Y_or_YUV_ps;
    ps_t convert_Y_or_YUV_fp16_ps;

    vs_t convert_UV_vs;
    ps_t convert_UV_ps;
    ps_t convert_UV_fp16_ps;

    D3D11_VIEWPORT out_Y_or_YUV_viewport;
    D3D11_VIEWPORT out_UV_viewport;

    DXGI_FORMAT format;

    device_t device;
    device_ctx_t device_ctx;

    texture2d_t output_texture;
    ID3D11Texture2D *rgb_present_texture = nullptr;  ///< Non-owning swapchain texture during convert_rgb().
    ID3D11RenderTargetView *rgb_present_target = nullptr;  ///< Non-owning swapchain RTV during convert_rgb().
    bool rgb_present_target_is_linear = false;  ///< True for a linear scRGB swapchain, independent of the physical output mode.
    bool rgb_copy_path_logged = false;
    bool rgb_copy_fallback_logged = false;
    ps_t rgb_present_ps;
    ps_t rgb_present_linear_to_srgb_ps;
    ps_t rgb_present_srgb_to_linear_ps;
    buf_t rgb_present_sdr_white;
    D3D11_VIEWPORT rgb_present_viewport {};

    std::unique_ptr<models::video_depth_estimator> depth_estimator;
    // The per-device D3D estimator is built on a background thread and borrows the startup-warmed
    // TensorRT context. The process-lifetime tracked worker service owns and joins the worker; the
    // task itself captures owning D3D references, so retiring this future during teardown is safe.
    std::future<std::unique_ptr<models::video_depth_estimator>> depth_estimator_build;
    unsigned engine_poll_counter = 0;  ///< Rate-limits the startup model-preparation wait warning.
    int sbs_mode = ::video::SBS_OFF;  ///< Host SBS mode for this encode device (set in init_output).
    config::video_t::sbs_t sbs_config {};  ///< Immutable Host SBS settings for this device.
    bool diagnostics_enabled = false;  ///< Cached once per device; the disabled hot path only branches.
    safe::mail_raw_t::event_t<int> sbs_depth_status_event;
    std::shared_ptr<safe::event_t<bool>> sbs_depth_pipeline_ready_event;
    safe::mail_raw_t::event_t<::video::sbs_telemetry_snapshot_t> sbs_telemetry_event;
    std::shared_ptr<::video::sbs_telemetry_subscription_t> sbs_telemetry_subscription;
    std::uint32_t sbs_telemetry_generation = 0;
    std::uint32_t sbs_telemetry_sequence = 0;
    std::uint64_t sbs_telemetry_last_sampled_frame_id = 0;
    std::uint64_t sbs_telemetry_min_frame_id = 0;
    std::uint32_t sbs_telemetry_last_hard_cut_count = 0;
    bool sbs_telemetry_has_sample = false;
    models::depth_input_region_t sbs_telemetry_input_region {};
    models::input_color_space sbs_telemetry_input_color_space =
      models::input_color_space::srgb;
    bool sbs_telemetry_input_domain_valid = false;
    bool sbs_telemetry_producer_failure_published = false;
    std::chrono::steady_clock::time_point sbs_telemetry_last_copy {};
    detail::host_sbs_adaptive_motion_route_state_t adaptive_motion_route_state;
    detail::host_sbs_adaptive_hold_cadence_t adaptive_hold_cadence;
    detail::host_sbs_gpu_observation_barrier_t gpu_observation_barrier;
    models::gpu_adaptive_ocr_cadence_t adaptive_ocr_cadence;
    opaque_gpu_followup_anchor_t opaque_gpu_followup_anchor;
    vs_t sbs_reprojection_vs;
    ps_t sbs_flat_identity_ps;
    ps_t sbs_reprojection_v2_live_ps;
    ps_t sbs_reprojection_v2_p010_y_ps;
    bool p010_y_mrt_targets_checked = false;
    bool p010_y_mrt_targets_qualified = false;
    models::host_sbs_renderer_e host_sbs_renderer =
      models::host_sbs_renderer_e::awaiting_v2;
    buf_t sbs_reprojection_cbuffer;
    float sbs_content_scale_x = 1.0f;
    float sbs_content_scale_y = 1.0f;
    texture2d_t sbs_intermediate_texture;
    render_target_t sbs_intermediate_rtv;
    shader_res_t sbs_intermediate_srv;
    bool sbs_intermediate_is_linear = false;
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
    std::optional<video_dom::lease_t> video_dom_lease;
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
    detail::host_sbs_content_refresh_state_t content_reuse_refresh;
    detail::host_sbs_approximate_reuse_provider_e approximate_reuse_since_enqueue =
      detail::host_sbs_approximate_reuse_provider_e::none;
    matched_presentation_cache_t matched_presentation_cache;
    bool depth_completion_poll_pending = false;
    bool depth_authority_reprocess_pending = false;
    std::chrono::steady_clock::time_point matched_stats_started {};
    unsigned matched_stats_calls = 0;
    unsigned matched_stats_completions = 0;
    unsigned matched_stats_repeats = 0;
    unsigned matched_stats_content_skips = 0;
    unsigned matched_stats_content_reuses = 0;
    unsigned matched_stats_damage_skips = 0;
    unsigned matched_stats_damage_reuses = 0;
    unsigned matched_stats_gpu_followup_expired = 0;
    unsigned matched_stats_gpu_followup_host_rejected = 0;
    unsigned matched_stats_gpu_followup_force_fallbacks = 0;
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

  namespace {
    constexpr wchar_t local_presenter_window_class[] = L"ApolloLocalArPresenter";

    bool same_local_presenter_rect(const RECT &left, const RECT &right) {
      return left.left == right.left && left.top == right.top &&
             left.right == right.right && left.bottom == right.bottom;
    }

    bool valid_local_presenter_rect(const RECT &rect) {
      return rect.right > rect.left && rect.bottom > rect.top;
    }

    RECT local_presenter_virtual_screen_rect() {
      const LONG left = GetSystemMetrics(SM_XVIRTUALSCREEN);
      const LONG top = GetSystemMetrics(SM_YVIRTUALSCREEN);
      return {
        left,
        top,
        left + GetSystemMetrics(SM_CXVIRTUALSCREEN),
        top + GetSystemMetrics(SM_CYVIRTUALSCREEN),
      };
    }

    std::optional<RECT> intersect_local_presenter_rects(const RECT &left, const RECT &right) {
      RECT intersection {
        std::max(left.left, right.left),
        std::max(left.top, right.top),
        std::min(left.right, right.right),
        std::min(left.bottom, right.bottom),
      };
      if (!valid_local_presenter_rect(intersection)) {
        return std::nullopt;
      }
      return intersection;
    }

    std::optional<RECT> query_local_presenter_cursor_clip(int attempts = 1) {
      RECT clip {};
      for (int attempt = 0; attempt < std::max(1, attempts); ++attempt) {
        if (GetClipCursor(&clip)) {
          return clip;
        }
        if (attempt + 1 < attempts) {
          Sleep(2);
        }
      }
      return std::nullopt;
    }

    struct deferred_cursor_restore_t {
      RECT previous_clip {};
      RECT owned_clip {};
      bool previous_clip_unbounded = false;
    };

    class deferred_cursor_restore_manager_t {
    public:
      deferred_cursor_restore_manager_t():
          worker_([this]() {
            run();
          }) {
      }

      ~deferred_cursor_restore_manager_t() {
        {
          std::lock_guard lock(mutex_);
          stopping_ = true;
        }
        changed_.notify_all();
        if (worker_.joinable()) {
          worker_.join();
        }
      }

      void defer(const deferred_cursor_restore_t &restore) {
        std::lock_guard lock(mutex_);
        if (pending_ && !same_local_presenter_rect(pending_->owned_clip, restore.owned_clip)) {
          BOOST_LOG(error) << "Local AR refused to overwrite an unresolved cursor-restoration record."sv;
          return;
        }
        pending_ = restore;
        retry_delay_ = 25ms;
        changed_.notify_all();
      }

      // A new owner must not save Apollo's unresolved clip as the user's prior restriction.
      bool ready_for_new_owner() {
        std::lock_guard lock(mutex_);
        resolve_once_locked();
        return !pending_;
      }

    private:
      bool resolve_once_locked() {
        if (!pending_) {
          return true;
        }
        const auto current = query_local_presenter_cursor_clip(3);
        if (!current) {
          return false;
        }
        if (!same_local_presenter_rect(*current, pending_->owned_clip)) {
          // Another application replaced Apollo's clip. Its state is authoritative, so restoring
          // the older rectangle now would corrupt that application's cursor confinement.
          pending_.reset();
          return true;
        }
        const BOOL restored = pending_->previous_clip_unbounded ?
                                ClipCursor(nullptr) :
                                ClipCursor(&pending_->previous_clip);
        if (!restored) {
          return false;
        }
        pending_.reset();
        return true;
      }

      void run() {
        std::unique_lock lock(mutex_);
        while (!stopping_) {
          changed_.wait(lock, [&]() {
            return stopping_ || pending_.has_value();
          });
          while (!stopping_ && pending_) {
            if (resolve_once_locked()) {
              retry_delay_ = 25ms;
              continue;
            }
            const auto delay = retry_delay_;
            retry_delay_ = std::min(retry_delay_ * 2, 1000ms);
            changed_.wait_for(lock, delay, [&]() {
              return stopping_ || !pending_;
            });
          }
        }
        // Make one final best effort while Win32 and the process desktop are still alive.
        resolve_once_locked();
      }

      std::mutex mutex_;
      std::condition_variable changed_;
      std::optional<deferred_cursor_restore_t> pending_;
      std::chrono::milliseconds retry_delay_ {25};
      bool stopping_ = false;
      std::thread worker_;
    };

    deferred_cursor_restore_manager_t &deferred_cursor_restore_manager() {
      static deferred_cursor_restore_manager_t manager;
      return manager;
    }

    void defer_cursor_retry(local_presenter_cursor_clip_t &clip) {
      const unsigned shift = std::min(clip.retry_failures, 6u);
      const auto delay = std::min(20ms * (1u << shift), 1000ms);
      clip.retry_failures = std::min(clip.retry_failures + 1, 16u);
      clip.retry_after = std::chrono::steady_clock::now() + delay;
    }

    void clear_cursor_retry(local_presenter_cursor_clip_t &clip) {
      clip.retry_failures = 0;
      clip.retry_after = {};
    }

    struct local_presenter_display_identity_t {
      LUID adapter_id {};
      UINT32 target_id = 0;
      std::wstring gdi_name;
      std::wstring device_path;
      std::wstring friendly_name;
    };

    bool same_local_presenter_luid(const LUID &left, const LUID &right) {
      return left.HighPart == right.HighPart && left.LowPart == right.LowPart;
    }

    std::optional<local_presenter_display_identity_t> query_local_presenter_display_identity(
      std::wstring_view display_name
    ) {
      if (display_name.empty()) {
        return std::nullopt;
      }
      std::vector<DISPLAYCONFIG_PATH_INFO> paths;
      std::vector<DISPLAYCONFIG_MODE_INFO> modes;
      if (!platf::display_config::query_display_config(
            QDC_ONLY_ACTIVE_PATHS,
            paths,
            modes,
            {
              4,
              platf::display_config::retry_delay_e::none,
            }
          )) {
        return std::nullopt;
      }
      for (const auto &path : paths) {
        DISPLAYCONFIG_SOURCE_DEVICE_NAME source_name {};
        source_name.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
        source_name.header.size = sizeof(source_name);
        source_name.header.adapterId = path.sourceInfo.adapterId;
        source_name.header.id = path.sourceInfo.id;
        if (DisplayConfigGetDeviceInfo(&source_name.header) != ERROR_SUCCESS || display_name != source_name.viewGdiDeviceName) {
          continue;
        }

        DISPLAYCONFIG_TARGET_DEVICE_NAME target_name {};
        target_name.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME;
        target_name.header.size = sizeof(target_name);
        target_name.header.adapterId = path.targetInfo.adapterId;
        target_name.header.id = path.targetInfo.id;
        if (DisplayConfigGetDeviceInfo(&target_name.header) != ERROR_SUCCESS || !target_name.monitorDevicePath[0]) {
          return std::nullopt;
        }
        return local_presenter_display_identity_t {
          path.targetInfo.adapterId,
          path.targetInfo.id,
          source_name.viewGdiDeviceName,
          target_name.monitorDevicePath,
          target_name.monitorFriendlyDeviceName,
        };
      }
      return std::nullopt;
    }

    bool local_presenter_identity_is_coherent(
      const std::shared_ptr<local_presenter_config_t::target_t> &live_target,
      const local_presenter_display_identity_t &source,
      const local_presenter_display_identity_t &target,
      const LUID &expected_adapter
    ) {
      // The source is a SudoVDA output: its CCD path lives on the IddCx adapter, whose LUID is
      // not the render GPU's. Only the physical sink is required to stay on the expected adapter;
      // the source is pinned by its driver-learned device path and Apollo-owned friendly name.
      if (!live_target || source.device_path.empty() || target.device_path.empty() || source.device_path == target.device_path || !same_local_presenter_luid(target.adapter_id, expected_adapter) || !boost::algorithm::istarts_with(source.friendly_name, std::wstring(L"Apollo AR Des"))) {
        return false;
      }
      std::lock_guard lock(live_target->mutex);
      return !live_target->source_device_path.empty() &&
             !live_target->target_device_path.empty() &&
             live_target->source_device_path == source.device_path &&
             live_target->target_device_path == target.device_path;
    }

    std::optional<RECT> query_local_presenter_display_rect(std::wstring_view display_name) {
      if (display_name.empty()) {
        return std::nullopt;
      }

      DEVMODEW mode {};
      mode.dmSize = sizeof(mode);
      const std::wstring owned_name {display_name};
      if (!EnumDisplaySettingsExW(owned_name.c_str(), ENUM_CURRENT_SETTINGS, &mode, 0)) {
        return std::nullopt;
      }

      RECT rect {
        mode.dmPosition.x,
        mode.dmPosition.y,
        mode.dmPosition.x + static_cast<LONG>(mode.dmPelsWidth),
        mode.dmPosition.y + static_cast<LONG>(mode.dmPelsHeight),
      };
      if (!valid_local_presenter_rect(rect)) {
        return std::nullopt;
      }
      return rect;
    }

    struct local_presenter_pointer_isolation_t {
      struct clip_request_t {
        bool geometry_known = false;
        std::optional<RECT> clip;
      };

      std::wstring source_display_name;
      RECT source_fallback_rect {};
      RECT target_fallback_rect {};
      std::shared_ptr<local_presenter_config_t::target_t> live_target;
      std::shared_ptr<local_presenter_cursor_clip_t> cursor_clip;

      RECT current_source_rect() const {
        if (const auto rect = query_local_presenter_display_rect(source_display_name)) {
          return *rect;
        }
        return source_fallback_rect;
      }

      RECT current_target_rect() const {
        RECT fallback = target_fallback_rect;
        std::wstring display_name;
        if (live_target) {
          std::lock_guard lock(live_target->mutex);
          fallback = live_target->rect;
          display_name = platf::from_utf8(live_target->display_name);
        }
        if (const auto rect = query_local_presenter_display_rect(display_name)) {
          return *rect;
        }
        return fallback;
      }

      clip_request_t desired_clip() const {
        const auto source_rect = current_source_rect();
        const auto target_rect = current_target_rect();
        if (!valid_local_presenter_rect(source_rect) || !valid_local_presenter_rect(target_rect)) {
          return {};
        }

        // The local-AR topology is interactive displays -> private virtual source -> physical
        // presentation sink. A rectangular cursor clip can exclude the sink without excluding an
        // interactive display only when the source and sink share that exact vertical boundary.
        const bool vertically_connected = std::max(source_rect.top, target_rect.top) <
                                          std::min(source_rect.bottom, target_rect.bottom);
        if (target_rect.left != source_rect.right || !vertically_connected) {
          return {true, std::nullopt};
        }

        RECT clip = local_presenter_virtual_screen_rect();
        clip.right = source_rect.right;
        if (!valid_local_presenter_rect(clip)) {
          return {true, std::nullopt};
        }
        return {true, clip};
      }

      void acquire_clip() {
        if (!cursor_clip || std::chrono::steady_clock::now() < cursor_clip->retry_after) {
          return;
        }
        if (!deferred_cursor_restore_manager().ready_for_new_owner()) {
          defer_cursor_retry(*cursor_clip);
          return;
        }
        if (cursor_clip->clip_yielded) {
          const auto current_clip = query_local_presenter_cursor_clip();
          if (!current_clip) {
            defer_cursor_retry(*cursor_clip);
            return;
          }
          const auto released_clip = cursor_clip->previous_clip_unbounded ?
                                       local_presenter_virtual_screen_rect() :
                                       cursor_clip->previous_clip;
          const auto unbounded_clip = local_presenter_virtual_screen_rect();
          const bool released_to_unbounded = same_local_presenter_rect(*current_clip, unbounded_clip);
          if (!same_local_presenter_rect(*current_clip, released_clip) && !released_to_unbounded) {
            cursor_clip->retry_after = std::chrono::steady_clock::now() + 250ms;
            return;  // another application still owns cursor confinement
          }
          if (released_to_unbounded) {
            // ClipCursor has no ownership stack. Once another application replaces the saved
            // restriction and releases to unbounded, that is the new state Apollo must preserve.
            cursor_clip->previous_clip = unbounded_clip;
            cursor_clip->previous_clip_unbounded = true;
          }
        }
        auto request = desired_clip();
        if (!request.geometry_known) {
          return;
        }
        auto requested_clip = request.clip;
        if (!requested_clip) {
          if (!cursor_clip->clip_unavailable_logged) {
            BOOST_LOG(debug) << "Local AR pointer confinement is unavailable for the current topology; "sv
                                "using event-driven edge clamping."sv;
            cursor_clip->clip_unavailable_logged = true;
          }
          return;
        }
        if (!cursor_clip->clip_saved) {
          const auto previous_clip = query_local_presenter_cursor_clip(3);
          if (!previous_clip) {
            BOOST_LOG(warning) << "Local AR presenter could not save the existing cursor clip: "sv
                               << GetLastError();
            defer_cursor_retry(*cursor_clip);
            return;
          }
          cursor_clip->previous_clip = *previous_clip;
          cursor_clip->clip_saved = true;
          cursor_clip->previous_clip_unbounded = same_local_presenter_rect(
            *previous_clip,
            local_presenter_virtual_screen_rect()
          );
        }
        if (!cursor_clip->previous_clip_unbounded) {
          requested_clip = intersect_local_presenter_rects(
            *requested_clip,
            cursor_clip->previous_clip
          );
          if (!requested_clip) {
            BOOST_LOG(debug) << "Local AR left another application's disjoint cursor confinement untouched."sv;
            cursor_clip->clip_yielded = true;
            cursor_clip->retry_after = std::chrono::steady_clock::now() + 250ms;
            return;
          }
        }
        if (!ClipCursor(&*requested_clip)) {
          BOOST_LOG(warning) << "Local AR presenter could not confine the cursor: "sv
                             << GetLastError();
          defer_cursor_retry(*cursor_clip);
          return;
        }

        // ClipCursor succeeded, so Apollo may own this clip even if the verification query is
        // transiently unavailable. Track tentative ownership so teardown can still restore it.
        cursor_clip->owned_clip = *requested_clip;
        cursor_clip->owns_clip = true;
        cursor_clip->clip_yielded = false;
        clear_cursor_retry(*cursor_clip);
        const auto applied_clip = query_local_presenter_cursor_clip(3);
        if (applied_clip && !same_local_presenter_rect(*applied_clip, *requested_clip)) {
          BOOST_LOG(warning) << "Local AR cursor confinement was overridden while being applied; "sv
                                "using event-driven edge clamping."sv;
          cursor_clip->owns_clip = false;
          cursor_clip->clip_yielded = true;
          return;
        }
        BOOST_LOG(debug) << "Local AR cursor confined through virtual-source right edge x="sv
                         << cursor_clip->owned_clip.right << '.';
      }

      void refresh_clip() {
        if (!cursor_clip) {
          return;
        }
        if (!cursor_clip->owns_clip) {
          acquire_clip();
          return;
        }

        const auto current_clip = query_local_presenter_cursor_clip();
        if (!current_clip) {
          // A query failure is not evidence that another application replaced Apollo's clip.
          // Keep tentative ownership and retry on the next pointer/topology event.
          defer_cursor_retry(*cursor_clip);
          return;
        }
        if (!same_local_presenter_rect(*current_clip, cursor_clip->owned_clip)) {
          // ClipCursor is shared system state. Never fight an application that replaces Apollo's
          // clip, and never restore over that application's state during teardown.
          cursor_clip->owns_clip = false;
          cursor_clip->clip_yielded = true;
          cursor_clip->retry_after = std::chrono::steady_clock::now() + 250ms;
          BOOST_LOG(debug) << "Local AR cursor confinement was replaced by another application; "sv
                              "leaving it untouched and using event-driven edge clamping."sv;
          return;
        }

        const auto request = desired_clip();
        if (!request.geometry_known) {
          // Display queries commonly fail while DXGI is rebuilding. Preserve the existing clip;
          // indeterminate geometry is not proof that the source/sink row stopped being valid.
          return;
        }
        const auto requested_clip = request.clip;
        if (!requested_clip) {
          if (!cursor_clip->restore()) {
            defer_cursor_retry(*cursor_clip);
          }
          cursor_clip->clip_unavailable_logged = false;
          return;
        }
        auto bounded_clip = requested_clip;
        if (!cursor_clip->previous_clip_unbounded) {
          bounded_clip = intersect_local_presenter_rects(*requested_clip, cursor_clip->previous_clip);
          if (!bounded_clip) {
            if (cursor_clip->restore()) {
              cursor_clip->retry_after = std::chrono::steady_clock::now() + 250ms;
            } else {
              defer_cursor_retry(*cursor_clip);
            }
            return;
          }
        }
        if (same_local_presenter_rect(*bounded_clip, cursor_clip->owned_clip)) {
          return;
        }
        if (!ClipCursor(&*bounded_clip)) {
          BOOST_LOG(warning) << "Local AR presenter could not update cursor confinement: "sv
                             << GetLastError();
          defer_cursor_retry(*cursor_clip);
          return;
        }

        cursor_clip->owned_clip = *bounded_clip;
        cursor_clip->owns_clip = true;
        clear_cursor_retry(*cursor_clip);
        const auto applied_clip = query_local_presenter_cursor_clip(3);
        if (applied_clip && !same_local_presenter_rect(*applied_clip, *bounded_clip)) {
          cursor_clip->owns_clip = false;
          cursor_clip->clip_yielded = true;
        }
      }

      bool clamp_pointer_to_source_edge() const {
        const auto source_rect = current_source_rect();
        const auto target_rect = current_target_rect();
        if (!valid_local_presenter_rect(source_rect) || !valid_local_presenter_rect(target_rect)) {
          return false;
        }

        POINT cursor {};
        if (!GetCursorPos(&cursor) || !PtInRect(&target_rect, cursor)) {
          return false;
        }

        const LONG clamped_y = std::clamp(cursor.y, source_rect.top, source_rect.bottom - 1);
        return SetCursorPos(source_rect.right - 1, clamped_y) != FALSE;
      }

      void cancel_active_pointer_interaction(HWND sink_window) const {
        HWND recipient = GetCapture();
        if (!recipient || recipient == sink_window) {
          recipient = GetForegroundWindow();
        }
        if (!recipient || recipient == sink_window) {
          return;
        }
        DWORD_PTR ignored = 0;
        SendMessageTimeoutW(
          recipient,
          WM_CANCELMODE,
          0,
          0,
          SMTO_ABORTIFHUNG | SMTO_BLOCK,
          50,
          &ignored
        );
      }
    };

    LRESULT CALLBACK local_presenter_window_proc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
      if (message == WM_NCCREATE) {
        const auto *create = reinterpret_cast<const CREATESTRUCTW *>(lparam);
        SetWindowLongPtrW(
          hwnd,
          GWLP_USERDATA,
          reinterpret_cast<LONG_PTR>(create->lpCreateParams)
        );
      }
      auto *pointer_isolation = reinterpret_cast<local_presenter_pointer_isolation_t *>(
        GetWindowLongPtrW(hwnd, GWLP_USERDATA)
      );
      switch (message) {
        case WM_CLOSE:
          DestroyWindow(hwnd);
          return 0;
        case WM_DESTROY:
          PostQuitMessage(0);
          return 0;
        case WM_NCHITTEST:
          // ClipCursor is shared state and another application may replace it. Clamp at the first
          // presenter hit-test as an event-driven fallback, before a click can reach the sink.
          if (pointer_isolation) {
            pointer_isolation->refresh_clip();
            pointer_isolation->clamp_pointer_to_source_edge();
          }
          return HTCLIENT;
        case WM_MOUSEACTIVATE:
          return MA_NOACTIVATEANDEAT;
        case WM_MOUSEMOVE:
          if (pointer_isolation) {
            pointer_isolation->refresh_clip();
            pointer_isolation->clamp_pointer_to_source_edge();
          }
          return 0;
        case WM_DISPLAYCHANGE:
        case WM_SETTINGCHANGE:
        case WM_WINDOWPOSCHANGED:
          if (pointer_isolation) {
            pointer_isolation->refresh_clip();
          }
          break;
        case WM_LBUTTONUP:
        case WM_RBUTTONUP:
        case WM_MBUTTONUP:
        case WM_XBUTTONUP:
          // If an application replaced ClipCursor during a drag, swallowing its release on the
          // presentation sink could leave the source application logically pressed. Cancel that
          // interaction explicitly before consuming the sink event.
          if (pointer_isolation) {
            pointer_isolation->cancel_active_pointer_interaction(hwnd);
            pointer_isolation->refresh_clip();
            pointer_isolation->clamp_pointer_to_source_edge();
          }
          return 0;
        case WM_LBUTTONDOWN:
        case WM_RBUTTONDOWN:
        case WM_MBUTTONDOWN:
        case WM_XBUTTONDOWN:
        case WM_MOUSEWHEEL:
        case WM_MOUSEHWHEEL:
          // The sink is presentation-only. If cursor confinement was replaced between movement
          // and this queued action, clamp it back and consume the sink-targeted action.
          if (pointer_isolation) {
            pointer_isolation->refresh_clip();
            pointer_isolation->clamp_pointer_to_source_edge();
          }
          return 0;
        default:
          break;
      }
      return DefWindowProcW(hwnd, message, wparam, lparam);
    }

    bool register_local_presenter_window_class() {
      WNDCLASSEXW window_class {};
      window_class.cbSize = sizeof(window_class);
      window_class.lpfnWndProc = local_presenter_window_proc;
      window_class.hInstance = GetModuleHandleW(nullptr);
      window_class.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
      window_class.lpszClassName = local_presenter_window_class;
      if (RegisterClassExW(&window_class)) {
        return true;
      }
      return GetLastError() == ERROR_CLASS_ALREADY_EXISTS;
    }
  }  // namespace

  bool local_presenter_cursor_clip_t::restore() {
    if (!owns_clip) {
      clip_saved = false;
      clip_yielded = false;
      previous_clip_unbounded = false;
      clear_cursor_retry(*this);
      return true;
    }

    const auto current_clip = query_local_presenter_cursor_clip(3);
    if (current_clip && same_local_presenter_rect(*current_clip, owned_clip)) {
      const BOOL restored = previous_clip_unbounded ? ClipCursor(nullptr) : ClipCursor(&previous_clip);
      if (!restored) {
        BOOST_LOG(warning) << "Local AR presenter could not restore the previous cursor clip: "sv
                           << GetLastError();
        return false;
      }
    } else if (!current_clip) {
      return false;
    }
    owns_clip = false;
    clip_saved = false;
    clip_yielded = false;
    previous_clip_unbounded = false;
    clear_cursor_retry(*this);
    return true;
  }

  local_presenter_cursor_clip_t::~local_presenter_cursor_clip_t() {
    for (int attempt = 0; attempt < 8; ++attempt) {
      if (restore()) {
        return;
      }
      Sleep(5);
    }
    BOOST_LOG(warning) << "Local AR presenter could not verify and restore cursor confinement after bounded retries; "sv
                          "deferring ownership-safe restoration until the desktop topology settles."sv;
    deferred_cursor_restore_manager().defer({previous_clip, owned_clip, previous_clip_unbounded});
    owns_clip = false;
  }

  void refresh_local_presenter_pointer_isolation(
    const std::string &source_display_name,
    const std::shared_ptr<local_presenter_config_t::target_t> &live_target,
    const std::shared_ptr<local_presenter_cursor_clip_t> &cursor_clip
  ) {
    if (source_display_name.empty() || !live_target || !cursor_clip) {
      return;
    }
    RECT target_rect {};
    {
      std::lock_guard lock(live_target->mutex);
      target_rect = live_target->rect;
    }
    const auto source_name_wide = platf::from_utf8(source_display_name);
    local_presenter_pointer_isolation_t pointer_isolation {
      source_name_wide,
      query_local_presenter_display_rect(source_name_wide).value_or(RECT {}),
      target_rect,
      live_target,
      cursor_clip,
    };
    pointer_isolation.refresh_clip();
  }

  local_presenter_result_e run_local_presenter(const local_presenter_config_t &config, std::stop_token stop_token) {
    auto read_target_rect = [&]() {
      if (!config.live_target) {
        return config.target_rect;
      }
      std::lock_guard lock(config.live_target->mutex);
      return config.live_target->rect;
    };
    auto read_target_display_name = [&]() {
      if (!config.live_target) {
        return std::string {};
      }
      std::lock_guard lock(config.live_target->mutex);
      return config.live_target->display_name;
    };
    auto target_rect = read_target_rect();
    auto target_display_name = read_target_display_name();
    const auto source_name_wide = platf::from_utf8(config.source_display_name);
    const auto target_name_wide = platf::from_utf8(target_display_name);
    const auto source_identity_at_start = query_local_presenter_display_identity(source_name_wide);
    const auto target_identity_at_start = query_local_presenter_display_identity(target_name_wide);
    if (!source_identity_at_start || !target_identity_at_start || !local_presenter_identity_is_coherent(config.live_target, *source_identity_at_start, *target_identity_at_start, config.target_adapter_id)) {
      BOOST_LOG(warning) << "Local AR presenter rejected a stale source/sink display-name generation."sv;
      return local_presenter_result_e::reinit;
    }
    const auto source_rect = query_local_presenter_display_rect(source_name_wide).value_or(RECT {});
    local_presenter_pointer_isolation_t pointer_isolation {
      source_name_wide,
      source_rect,
      target_rect,
      config.live_target,
      config.cursor_clip,
    };
    // A cursor clip belongs to the complete local session, not one DXGI attempt. Reconcile it
    // before any early return so repeated pre-window reinitializations cannot leave stale bounds.
    pointer_isolation.refresh_clip();
    const int output_width = target_rect.right - target_rect.left;
    const int output_height = target_rect.bottom - target_rect.top;
    if (output_width <= 0 || output_height <= 0 || config.source_display_name.empty()) {
      BOOST_LOG(error) << "Local AR presenter received invalid source or target geometry."sv;
      return local_presenter_result_e::error;
    }

    ::video::config_t capture_config {};
    capture_config.width = config.sbs_mode == ::video::SBS_AI ? output_width / 2 : output_width;
    capture_config.height = output_height;
    capture_config.framerate = std::max(1, (config.target_refresh_millihz + 500) / 1000);
    capture_config.dynamicRange = config.hdr ? 1 : 0;
    capture_config.encodingFramerate = config.target_refresh_millihz;

    const auto preferred_capture_backend = config.capture_failover ?
                                             config.capture_failover->preferred_backend() :
                                             capture_backend_e::ddup;
    auto display = platf::display(
      config.source_display_name,
      capture_config,
      preferred_capture_backend
    );
    if (display && config.capture_failover) {
      config.capture_failover->note_backend_opened(display->capture_backend());
    }
    auto dxgi_display = std::dynamic_pointer_cast<display_base_t>(display);
    if (!dxgi_display) {
      BOOST_LOG(warning) << "Local AR presenter could not yet open virtual source display "sv
                         << config.source_display_name << "; refreshing display topology."sv;
      return local_presenter_result_e::reinit;
    }
    if (dxgi_display->is_hdr() != config.hdr) {
      BOOST_LOG(info) << "Local AR virtual source has not settled to "sv
                      << (config.hdr ? "HDR"sv : "SDR"sv) << "; reinitializing."sv;
      return local_presenter_result_e::reinit;
    }
    const auto source_identity_after_open = query_local_presenter_display_identity(source_name_wide);
    DXGI_ADAPTER_DESC1 source_adapter_desc {};
    // DXGI attributes the SudoVDA output to its render GPU, while CCD reports the IddCx adapter.
    // The zero-copy contract is that capture opened on the same GPU that presents to the sink.
    if (!source_identity_after_open || source_identity_after_open->device_path != source_identity_at_start->device_path || !dxgi_display->adapter || FAILED(dxgi_display->adapter->GetDesc1(&source_adapter_desc)) || !same_local_presenter_luid(source_adapter_desc.AdapterLuid, config.target_adapter_id)) {
      BOOST_LOG(info) << "Local AR virtual source identity changed while DXGI capture was opening; reinitializing."sv;
      return local_presenter_result_e::reinit;
    }

    d3d_base_encode_device converter;
    if (converter.init(display, dxgi_display->adapter.get(), platf::pix_fmt_e::nv12)) {
      BOOST_LOG(error) << "Local AR presenter could not initialize the production SBS converter."sv;
      return local_presenter_result_e::error;
    }

    auto depth_pipeline_ready_event = std::make_shared<safe::event_t<bool>>();
    if (converter.init_rgb_output(
          output_width,
          output_height,
          config.sbs_mode,
          config.sbs_config,
          depth_pipeline_ready_event
        )) {
      BOOST_LOG(error) << "Local AR presenter could not initialize RGB presentation resources."sv;
      return local_presenter_result_e::error;
    }

    DXGI_OUTPUT_DESC source_output_desc {};
    if (!dxgi_display->output || FAILED(dxgi_display->output->GetDesc(&source_output_desc))) {
      BOOST_LOG(error) << "Local AR presenter could not query the virtual source geometry."sv;
      return local_presenter_result_e::error;
    }
    pointer_isolation.source_fallback_rect = source_output_desc.DesktopCoordinates;
    pointer_isolation.refresh_clip();

    if (!register_local_presenter_window_class()) {
      BOOST_LOG(error) << "Local AR presenter could not register its window class: "sv
                       << GetLastError();
      return local_presenter_result_e::error;
    }

    std::promise<std::pair<HWND, DWORD>> window_ready;
    auto window_future = window_ready.get_future();
    std::jthread window_thread([&pointer_isolation,
                                target_rect,
                                output_width,
                                output_height,
                                ready = std::move(window_ready)](std::stop_token window_stop_token) mutable {
      HWND created_window = CreateWindowExW(
        WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
        local_presenter_window_class,
        L"Sunshine 3D Local SBS AI",
        WS_POPUP,
        target_rect.left,
        target_rect.top,
        output_width,
        output_height,
        nullptr,
        nullptr,
        GetModuleHandleW(nullptr),
        &pointer_isolation
      );
      const DWORD create_error = created_window ? ERROR_SUCCESS : GetLastError();
      if (created_window) {
        pointer_isolation.acquire_clip();
      }
      ready.set_value({created_window, create_error});
      if (!created_window) {
        return;
      }

      // Capture may block for 200 ms on a static desktop. Keep Win32 input on this dedicated
      // message loop so pointer entry is handled immediately regardless of frame
      // cadence. Stopping the owner posts WM_CLOSE and wakes GetMessageW().
      std::stop_callback stop_window(window_stop_token, [created_window]() {
        PostMessageW(created_window, WM_CLOSE, 0, 0);
      });
      MSG message {};
      while (GetMessageW(&message, nullptr, 0, 0) > 0) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
      }
      if (IsWindow(created_window)) {
        DestroyWindow(created_window);
      }
    });
    const auto [window, window_create_error] = window_future.get();
    if (!window) {
      BOOST_LOG(error) << "Local AR presenter could not create its output window: "sv
                       << window_create_error;
      return local_presenter_result_e::error;
    }
    auto destroy_window = util::fail_guard([&]() {
      window_thread.request_stop();
      if (IsWindow(window)) {
        PostMessageW(window, WM_CLOSE, 0, 0);
      }
      if (window_thread.joinable()) {
        window_thread.join();
      }
      SetThreadExecutionState(ES_CONTINUOUS);
    });

    if (!SetWindowPos(
          window,
          HWND_TOPMOST,
          target_rect.left,
          target_rect.top,
          output_width,
          output_height,
          SWP_NOACTIVATE
        )) {
      BOOST_LOG(warning) << "Local AR presenter could not place its hidden output window: "sv
                         << GetLastError();
      return local_presenter_result_e::reinit;
    }
    SetThreadExecutionState(ES_CONTINUOUS | ES_DISPLAY_REQUIRED);

    Microsoft::WRL::ComPtr<IDXGIDevice> dxgi_device;
    Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
    Microsoft::WRL::ComPtr<IDXGIFactory2> factory;
    Microsoft::WRL::ComPtr<IDXGISwapChain1> swapchain;
    auto status = converter.device->QueryInterface(IID_PPV_ARGS(&dxgi_device));
    if (SUCCEEDED(status)) {
      status = dxgi_device->GetAdapter(&adapter);
    }
    if (SUCCEEDED(status)) {
      status = adapter->GetParent(IID_PPV_ARGS(&factory));
    }
    if (FAILED(status)) {
      BOOST_LOG(error) << "Local AR presenter could not obtain its DXGI factory: "sv
                       << util::log_hex(status);
      return local_presenter_result_e::error;
    }

    DXGI_ADAPTER_DESC presenter_adapter_desc {};
    status = adapter->GetDesc(&presenter_adapter_desc);
    if (FAILED(status)) {
      BOOST_LOG(error) << "Local AR presenter could not identify its graphics adapter: "sv
                       << util::log_hex(status);
      return local_presenter_result_e::error;
    }
    if (presenter_adapter_desc.AdapterLuid.HighPart != config.target_adapter_id.HighPart || presenter_adapter_desc.AdapterLuid.LowPart != config.target_adapter_id.LowPart) {
      BOOST_LOG(info) << "Local AR physical output migrated to another graphics adapter; "sv
                         "requesting a complete presentation-session rebuild."sv;
      return local_presenter_result_e::error;
    }

    Microsoft::WRL::ComPtr<IDXGIOutput> target_output;
    DXGI_OUTPUT_DESC target_output_desc {};
    const auto target_display_name_wide = target_name_wide;
    for (UINT output_index = 0;; ++output_index) {
      Microsoft::WRL::ComPtr<IDXGIOutput> candidate;
      const auto enum_status = adapter->EnumOutputs(output_index, &candidate);
      if (enum_status == DXGI_ERROR_NOT_FOUND) {
        break;
      }
      if (FAILED(enum_status) || FAILED(candidate->GetDesc(&target_output_desc))) {
        continue;
      }
      if (std::wstring_view(target_output_desc.DeviceName) == target_display_name_wide) {
        target_output = std::move(candidate);
        break;
      }
    }
    if (!target_output) {
      BOOST_LOG(warning) << "Local AR physical output "sv << target_display_name
                         << " is not yet available through DXGI; refreshing topology."sv;
      return local_presenter_result_e::reinit;
    }
    const auto source_identity_before_present = query_local_presenter_display_identity(source_name_wide);
    const auto target_identity_before_present = query_local_presenter_display_identity(target_display_name_wide);
    if (!source_identity_before_present || !target_identity_before_present || source_identity_before_present->device_path != source_identity_at_start->device_path || target_identity_before_present->device_path != target_identity_at_start->device_path || !local_presenter_identity_is_coherent(config.live_target, *source_identity_before_present, *target_identity_before_present, config.target_adapter_id)) {
      BOOST_LOG(info) << "Local AR source/sink identity changed while presentation resources were opening; reinitializing."sv;
      return local_presenter_result_e::reinit;
    }

    const auto actual_target_rect = target_output_desc.DesktopCoordinates;
    const auto actual_target_width = actual_target_rect.right - actual_target_rect.left;
    const auto actual_target_height = actual_target_rect.bottom - actual_target_rect.top;
    if (actual_target_width != output_width || actual_target_height != output_height) {
      BOOST_LOG(info) << "Local AR physical output mode changed from "sv << output_width << 'x'
                      << output_height << " to "sv << actual_target_width << 'x'
                      << actual_target_height << "; reinitializing."sv;
      return local_presenter_result_e::reinit;
    }
    Microsoft::WRL::ComPtr<IDXGIOutput6> target_output6;
    DXGI_OUTPUT_DESC1 target_desc1 {};
    const bool target_color_volume_available =
      SUCCEEDED(target_output.As(&target_output6)) && SUCCEEDED(target_output6->GetDesc1(&target_desc1));
    if (target_color_volume_available) {
      BOOST_LOG(info) << "Local AR physical color volume: colorspace="sv
                      << dxgi_display->colorspace_to_string(target_desc1.ColorSpace)
                      << " bits="sv << target_desc1.BitsPerColor
                      << " luminance="sv << target_desc1.MinLuminance << '-'
                      << target_desc1.MaxLuminance << " nits full_frame="sv
                      << target_desc1.MaxFullFrameLuminance << " nits."sv;
    } else {
      BOOST_LOG(warning) << "Could not query the local AR physical output color volume."sv;
    }
    if (config.hdr && (!target_color_volume_available || target_desc1.ColorSpace != DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020)) {
      BOOST_LOG(info) << "Local AR physical output has not settled to HDR10/PQ; reinitializing."sv;
      return local_presenter_result_e::reinit;
    }
    target_rect = actual_target_rect;
    if (!SetWindowPos(
          window,
          HWND_TOPMOST,
          target_rect.left,
          target_rect.top,
          output_width,
          output_height,
          SWP_NOACTIVATE | SWP_SHOWWINDOW
        )) {
      BOOST_LOG(warning) << "Local AR presenter could not place its output window on the verified sink: "sv
                         << GetLastError();
      return local_presenter_result_e::reinit;
    }
    ShowWindow(window, SW_SHOWNOACTIVATE);

    DXGI_SWAP_CHAIN_DESC1 swapchain_desc {};
    swapchain_desc.Width = output_width;
    swapchain_desc.Height = output_height;
    // HDR capture is linear scRGB. Preserve it in an FP16 scRGB swapchain and let DWM perform the
    // final conversion to the restricted physical output's HDR10/PQ color volume.
    swapchain_desc.Format = config.hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT :
                                         DXGI_FORMAT_B8G8R8A8_UNORM;
    swapchain_desc.SampleDesc.Count = 1;
    swapchain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapchain_desc.BufferCount = 2;
    swapchain_desc.Scaling = DXGI_SCALING_STRETCH;
    swapchain_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapchain_desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    swapchain_desc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
    status = factory->CreateSwapChainForHwnd(
      converter.device.get(),
      window,
      &swapchain_desc,
      nullptr,
      target_output.Get(),
      &swapchain
    );
    if (FAILED(status)) {
      BOOST_LOG(error) << "Local AR presenter could not create its swapchain: "sv
                       << util::log_hex(status);
      return local_presenter_result_e::error;
    }
    factory->MakeWindowAssociation(window, DXGI_MWA_NO_ALT_ENTER | DXGI_MWA_NO_WINDOW_CHANGES);

    Microsoft::WRL::ComPtr<IDXGISwapChain2> swapchain2;
    status = swapchain.As(&swapchain2);
    if (SUCCEEDED(status)) {
      status = swapchain2->SetMaximumFrameLatency(1);
    }
    if (FAILED(status)) {
      BOOST_LOG(error) << "Local AR presenter could not configure bounded frame latency: "sv
                       << util::log_hex(status);
      return local_presenter_result_e::error;
    }
    const auto frame_latency_waitable = swapchain2->GetFrameLatencyWaitableObject();
    if (!frame_latency_waitable) {
      BOOST_LOG(error) << "Local AR presenter could not obtain its frame-latency waitable object: "sv
                       << GetLastError();
      return local_presenter_result_e::error;
    }
    auto close_frame_latency_waitable = util::fail_guard([&]() {
      CloseHandle(frame_latency_waitable);
    });

    Microsoft::WRL::ComPtr<IDXGISwapChain3> swapchain3;
    status = swapchain.As(&swapchain3);
    if (FAILED(status)) {
      BOOST_LOG(error) << "Local AR presenter could not query IDXGISwapChain3: "sv
                       << util::log_hex(status);
      return local_presenter_result_e::error;
    }
    const auto presentation_color_space = config.hdr ? DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709 :
                                                       DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
    UINT color_space_support = 0;
    status = swapchain3->CheckColorSpaceSupport(presentation_color_space, &color_space_support);
    if (FAILED(status) || !(color_space_support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT)) {
      BOOST_LOG(error) << "Local AR presenter does not support the required "sv
                       << (config.hdr ? "linear scRGB"sv : "Rec.709"sv)
                       << " swapchain color space: "sv << util::log_hex(status);
      return local_presenter_result_e::error;
    }
    status = swapchain3->SetColorSpace1(presentation_color_space);
    if (FAILED(status)) {
      BOOST_LOG(error) << "Local AR presenter could not set its swapchain color space: "sv
                       << util::log_hex(status);
      return local_presenter_result_e::error;
    }

    // D3D11 flip-model swapchains expose the current writable buffer as buffer 0.
    // Unlike D3D12, querying every physical buffer by index is not supported.
    Microsoft::WRL::ComPtr<ID3D11Texture2D> backbuffer;
    status = swapchain->GetBuffer(0, IID_PPV_ARGS(&backbuffer));
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> backbuffer_rtv;
    if (SUCCEEDED(status)) {
      status = converter.device->CreateRenderTargetView(backbuffer.Get(), nullptr, &backbuffer_rtv);
    }
    if (FAILED(status)) {
      BOOST_LOG(error) << "Local AR presenter could not create the swapchain render target: "sv
                       << util::log_hex(status);
      return local_presenter_result_e::error;
    }

    BOOST_LOG(info) << "Local AR presentation started: source="sv << config.source_display_name
                    << " target="sv << target_display_name << " ["sv << target_rect.left << ','
                    << target_rect.top << "] "sv << output_width << 'x' << output_height << '@'
                    << capture_config.framerate << " mode="sv
                    << (config.sbs_mode == ::video::SBS_AI ? "SBS AI"sv : "normal"sv)
                    << " color="sv << (config.hdr ? "HDR linear scRGB FP16"sv : "SDR Rec.709 BGRA8"sv)
                    << "; physical output is presentation-only."sv;

    std::array<std::shared_ptr<platf::img_t>, 3> image_pool;
    const bool diagnostics_enabled = ::config::sunshine.diagnostics_enabled;
    bool capture_cursor = true;
    bool window_closed = false;
    auto next_pointer_fallback = std::chrono::steady_clock::now();
    auto poll_presenter_state = [&]() {
      window_closed = !IsWindow(window);
      // ClipCursor normally prevents entry without any polling. Keep a throttled global fallback
      // for applications that replace it while owning Win32 mouse capture: in that case movement
      // is routed to the captured source window and the presenter HWND receives no mouse message.
      const auto now = std::chrono::steady_clock::now();
      if (!window_closed && now >= next_pointer_fallback) {
        pointer_isolation.clamp_pointer_to_source_edge();
        next_pointer_fallback = now + 100ms;
      }
    };

    auto pull_image = [&](std::shared_ptr<platf::img_t> &image) {
      if (stop_token.stop_requested() || window_closed) {
        image.reset();
        return false;
      }
      for (auto &candidate : image_pool) {
        if (!candidate) {
          candidate = display->alloc_img();
        }
        if (candidate && candidate.use_count() == 1) {
            image = candidate;
            image->frame_timestamp.reset();
            image->content_timestamp.reset();
            return true;
        }
      }
      image.reset();
      return false;
    };

    bool presentation_reinit_requested = false;
    std::uint64_t capture_attempt_frames = 0;
    std::uint64_t captured_frames = 0;
    std::uint64_t presented_frames = 0;
    std::uint64_t busy_present_retries = 0;
    std::shared_ptr<platf::img_t> retained_presenter_source;
    detail::local_presenter_retry_state_t presenter_retry;
    auto present_stats_started = diagnostics_enabled ? std::chrono::steady_clock::now() :
                                                       std::chrono::steady_clock::time_point {};
    auto log_present_stats = [&]() {
      if (!diagnostics_enabled) {
        return;
      }
      const auto now = std::chrono::steady_clock::now();
      const auto elapsed = now - present_stats_started;
      if (elapsed < std::chrono::seconds(5)) {
        return;
      }
      const double elapsed_seconds = std::chrono::duration<double>(elapsed).count();
      BOOST_LOG(info) << "Local AR presenter stats: captured="sv << captured_frames
                      << " presented="sv << presented_frames
                      << " busy_retries="sv << busy_present_retries
                      << " output_fps="sv << (presented_frames / elapsed_seconds);
      captured_frames = 0;
      presented_frames = 0;
      busy_present_retries = 0;
      present_stats_started = now;
    };
    auto push_image = [&](std::shared_ptr<platf::img_t> &&image, bool frame_captured) {
      poll_presenter_state();
      if (stop_token.stop_requested() || window_closed) {
        return false;
      }
      if (frame_captured) {
        if (!image) {
          BOOST_LOG(error) << "Local AR capture reported a frame without an image."sv;
          return false;
        }
        retained_presenter_source = std::move(image);
        presenter_retry.observe_source();
        ++capture_attempt_frames;
        if (diagnostics_enabled) {
          ++captured_frames;
        }
      }

      const bool depth_pipeline_ready =
        depth_pipeline_ready_event && depth_pipeline_ready_event->peek();
      const bool conversion_poll_pending = converter.needs_conversion_poll();
      if (!presenter_retry.should_process(
            static_cast<bool>(retained_presenter_source),
            depth_pipeline_ready,
            conversion_poll_pending
          )) {
        return true;
      }

      const auto latest_target_rect = read_target_rect();
      if (latest_target_rect.left != target_rect.left || latest_target_rect.top != target_rect.top || latest_target_rect.right != target_rect.right || latest_target_rect.bottom != target_rect.bottom) {
        const auto latest_width = latest_target_rect.right - latest_target_rect.left;
        const auto latest_height = latest_target_rect.bottom - latest_target_rect.top;
        if (latest_width != output_width || latest_height != output_height) {
          BOOST_LOG(info) << "Local AR target size changed; reinitializing presentation resources."sv;
          presentation_reinit_requested = true;
          return false;
        }
        if (!SetWindowPos(
              window,
              HWND_TOPMOST,
              latest_target_rect.left,
              latest_target_rect.top,
              output_width,
              output_height,
              SWP_NOACTIVATE | SWP_SHOWWINDOW
            )) {
          BOOST_LOG(warning) << "Local AR presenter could not follow the physical sink position: "sv
                             << GetLastError();
          presentation_reinit_requested = true;
          return false;
        }
        target_rect = latest_target_rect;
      }

      // Never let a slower physical output back-pressure capture, asynchronous depth processing,
      // or the desktop being interacted with. Retain only the newest source and retry it on the
      // next capture timeout if DWM has not retired the previous flip yet.
      const auto frame_latency_status = WaitForSingleObject(frame_latency_waitable, 0);
      if (frame_latency_status == WAIT_FAILED) {
        BOOST_LOG(error) << "Local AR presenter frame-latency wait failed: "sv << GetLastError();
        return false;
      }
      if (frame_latency_status != WAIT_OBJECT_0) {
        if (diagnostics_enabled) {
          ++busy_present_retries;
          log_present_stats();
        }
        return true;
      }

      const bool conversion_required = presenter_retry.should_convert(
        static_cast<bool>(retained_presenter_source),
        depth_pipeline_ready,
        conversion_poll_pending
      );
      if (conversion_required) {
        // The swapchain transfer contract controls RGB presentation. In SDR, DWM may composite
        // the G22 swapchain onto an HDR physical output, so the output's live HDR state is not
        // relevant.
        if (depth_pipeline_ready && depth_pipeline_ready_event) {
          // Consume only the notification sampled before should_process(). A build that becomes
          // ready during conversion must remain armed for the next static-source timeout.
          depth_pipeline_ready_event->pop(0ms);
        }
        if (converter.convert_rgb(
              *retained_presenter_source,
              backbuffer.Get(),
              backbuffer_rtv.Get(),
              config.hdr
            )) {
          BOOST_LOG(error) << "Local AR presenter failed to convert a captured frame."sv;
          return false;
        }
        // A ready/poll-triggered conversion may update a source whose earlier flat output was
        // already presented. Keep this newly rendered backbuffer retryable if Present is busy;
        // only a successful Present clears the owner.
        presenter_retry.record_converted();
      }
      DXGI_PRESENT_PARAMETERS present_parameters {};
      status = swapchain->Present1(
        0,
        DXGI_PRESENT_RESTRICT_TO_OUTPUT | DXGI_PRESENT_DO_NOT_WAIT,
        &present_parameters
      );
      if (status == DXGI_ERROR_WAS_STILL_DRAWING) {
        if (diagnostics_enabled) {
          ++busy_present_retries;
          log_present_stats();
        }
        return true;
      }
      if (status == DXGI_ERROR_RESTRICT_TO_OUTPUT_STALE) {
        BOOST_LOG(info) << "Local AR restricted output changed; reinitializing presentation resources."sv;
        presentation_reinit_requested = true;
        return false;
      }
      if (FAILED(status)) {
        BOOST_LOG(error) << "Local AR presenter Present() failed: "sv << util::log_hex(status);
        return false;
      }
      if (diagnostics_enabled) {
        ++presented_frames;
        log_present_stats();
      }
      if (config.presented_frames) {
        config.presented_frames->fetch_add(1, std::memory_order_relaxed);
      }
      presenter_retry.record_presented();
      return true;
    };

    const auto capture_backend = display->capture_backend();
    const auto capture_started = std::chrono::steady_clock::now();
    const auto capture_status = display->capture(push_image, pull_image, &capture_cursor);
    if (config.capture_failover) {
      const auto previous_preference = config.capture_failover->preferred_backend();
      config.capture_failover->note_capture_result(
        capture_backend,
        capture_status,
        capture_attempt_frames,
        std::chrono::steady_clock::now() - capture_started
      );
      if (previous_preference != config.capture_failover->preferred_backend()) {
        BOOST_LOG(warning) << "Local AR Desktop Duplication failed repeatedly before stable capture; "sv
                              "using Windows.Graphics.Capture for the rest of this session."sv;
      }
    }
    if (presentation_reinit_requested && !stop_token.stop_requested()) {
      return local_presenter_result_e::reinit;
    }
    if ((capture_status == capture_e::reinit || capture_status == capture_e::error) && !stop_token.stop_requested()) {
      BOOST_LOG(info) << "Local AR capture backend stopped; reinitializing the presenter."sv;
      return local_presenter_result_e::reinit;
    }
    if (!stop_token.stop_requested() && !window_closed) {
      BOOST_LOG(warning) << "Local AR capture stopped unexpectedly with status "sv
                         << (int) capture_status << '.';
      return local_presenter_result_e::error;
    }

    BOOST_LOG(info) << "Local AR presentation stopped."sv;
    return local_presenter_result_e::stopped;
  }

  class d3d_nvenc_encode_device_t: public nvenc_encode_device_t {
  public:
    ~d3d_nvenc_encode_device_t() override {
      // The destructor body runs before nvenc_d3d is destroyed. Do not let a hung NVENC teardown
      // retain the capture display and deadlock capture-thread reinitialization.
      base.release_capture_display();
    }

    bool init_device(std::shared_ptr<platf::display_t> display, adapter_t::pointer adapter_p, pix_fmt_e pix_fmt) {
      buffer_format = nvenc::nvenc_format_from_sunshine_format(pix_fmt);
      if (buffer_format == NV_ENC_BUFFER_FORMAT_UNDEFINED) {
        BOOST_LOG(error) << "Unexpected pixel format for NvENC ["sv << from_pix_fmt(pix_fmt) << ']';
        return false;
      }

      if (base.init(display, adapter_p, pix_fmt)) {
        return false;
      }

      nvenc_d3d = std::make_unique<nvenc::nvenc_d3d11_native>(base.device.get());
      nvenc = nvenc_d3d.get();

      return true;
    }

    bool init_encoder(const ::video::config_t &client_config, const ::video::sunshine_colorspace_t &colorspace) override {
      if (!nvenc_d3d) {
        return false;
      }

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
      return base.init_output(
               nvenc_d3d->get_input_texture(),
               client_config.width,
               client_config.height,
               client_config.sbs_mode,
               client_config.sbs_config,
               client_config.sbs_depth_status_event,
               client_config.sbs_depth_pipeline_ready_event,
               client_config.sbs_telemetry_event,
               client_config.sbs_telemetry_subscription,
               client_config.sbs_telemetry_generation,
               client_config.sbs_debug_dump_pending
             ) == 0;
    }

    int convert(platf::img_t &img_base) override {
      return base.convert(img_base);
    }

    int convert_with_encode_target(
      platf::img_t &img_base,
      const std::chrono::steady_clock::time_point next_encode_target
    ) override {
      return base.convert(img_base, next_encode_target);
    }

    bool needs_conversion_poll() const override {
      return base.needs_conversion_poll();
    }

    std::optional<std::chrono::steady_clock::time_point> rendered_content_timestamp() const override {
      return base.rendered_content_timestamp();
    }

  private:
    d3d_base_encode_device base;
    std::unique_ptr<nvenc::nvenc_d3d11_native> nvenc_d3d;
    NV_ENC_BUFFER_FORMAT buffer_format = NV_ENC_BUFFER_FORMAT_UNDEFINED;
  };

  bool set_cursor_texture(device_t::pointer device, gpu_cursor_t &cursor, util::buffer_t<std::uint8_t> &&cursor_img, DXGI_OUTDUPL_POINTER_SHAPE_INFO &shape_info) {
    // This cursor image may not be used
    if (cursor_img.size() == 0) {
      cursor.input_res.reset();
      cursor.set_texture(0, 0, nullptr);
      return true;
    }

    D3D11_SUBRESOURCE_DATA data {
      std::begin(cursor_img),
      4 * shape_info.Width,
      0
    };

    // Create texture for cursor
    D3D11_TEXTURE2D_DESC t {};
    t.Width = shape_info.Width;
    t.Height = cursor_img.size() / data.SysMemPitch;
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

  capture_e display_ddup_vram_t::snapshot(const pull_free_image_cb_t &pull_free_image_cb, std::shared_ptr<platf::img_t> &img_out, std::chrono::milliseconds timeout, bool cursor_visible) {
    HRESULT status;
    DXGI_OUTDUPL_FRAME_INFO frame_info;

    resource_t::pointer res_p {};
    auto capture_status = dup.next_frame(frame_info, timeout, &res_p);
    resource_t res {res_p};

    if (capture_status != capture_e::ok) {
      return capture_status;
    }

    const bool mouse_update_flag = frame_info.LastMouseUpdateTime.QuadPart != 0 || frame_info.PointerShapeBufferSize > 0;
    const bool frame_update_flag = frame_info.LastPresentTime.QuadPart != 0;
    const bool update_flag = mouse_update_flag || frame_update_flag;

    if (!update_flag) {
      return capture_e::timeout;
    }

    const auto timestamp_now = std::chrono::steady_clock::now();
    const auto timestamp_qpc = qpc_counter();
    const auto from_qpc = [&](const std::int64_t value) {
      return timestamp_now - qpc_time_difference(timestamp_qpc, value);
    };
    const auto present_timestamp = frame_info.LastPresentTime.QuadPart != 0 ?
                                     std::optional {from_qpc(frame_info.LastPresentTime.QuadPart)} :
                                     std::nullopt;
    const auto timestamp_selection = detail::select_ddup_timestamps(
      present_timestamp,
      frame_info.LastMouseUpdateTime.QuadPart != 0 ?
        std::optional {from_qpc(frame_info.LastMouseUpdateTime.QuadPart)} :
        std::nullopt,
      last_content_timestamp
    );
    auto frame_timestamp = timestamp_selection.presentation_timestamp;
    // A present becomes content only after its CopyResource is actually submitted. Mark the
    // metadata chain discontinuous now so any early return makes the next successful commit
    // explicitly UNKNOWN instead of omitting an acquired-but-uncopied surface.
    const bool prior_damage_chain_valid = damage_chain_valid;
    if (frame_update_flag) {
      damage_chain_valid = false;
    }

    if (frame_info.PointerShapeBufferSize > 0) {
      DXGI_OUTDUPL_POINTER_SHAPE_INFO shape_info {};

      util::buffer_t<std::uint8_t> img_data {frame_info.PointerShapeBufferSize};

      UINT dummy;
      status = dup.dup->GetFramePointerShape(img_data.size(), std::begin(img_data), &dummy, &shape_info);
      if (FAILED(status)) {
        BOOST_LOG(error) << "Failed to get new pointer shape [0x"sv << util::hex(status).to_string_view() << ']';

        return capture_e::error;
      }

      auto alpha_cursor_img = make_cursor_alpha_image(img_data, shape_info);
      auto xor_cursor_img = make_cursor_xor_image(img_data, shape_info);

      if (!set_cursor_texture(device.get(), cursor_alpha, std::move(alpha_cursor_img), shape_info) || !set_cursor_texture(device.get(), cursor_xor, std::move(xor_cursor_img), shape_info)) {
        return capture_e::error;
      }
    }

    if (frame_info.LastMouseUpdateTime.QuadPart) {
      cursor_alpha.set_pos(frame_info.PointerPosition.Position.x, frame_info.PointerPosition.Position.y, width, height, display_rotation, frame_info.PointerPosition.Visible);

      cursor_xor.set_pos(frame_info.PointerPosition.Position.x, frame_info.PointerPosition.Position.y, width, height, display_rotation, frame_info.PointerPosition.Visible);
    }

    const bool blend_mouse_cursor_flag = (cursor_alpha.visible || cursor_xor.visible) && cursor_visible;

    texture2d_t src {};
    std::optional<detail::ddup_damage_update_t> pending_damage;
    if (frame_update_flag) {
      // Get the texture object from this frame
      status = res->QueryInterface(IID_ID3D11Texture2D, (void **) &src);
      if (FAILED(status)) {
        BOOST_LOG(error) << "Couldn't query interface [0x"sv << util::hex(status).to_string_view() << ']';
        return capture_e::error;
      }

      D3D11_TEXTURE2D_DESC desc {};
      src->GetDesc(&desc);

      // It's possible for our display enumeration to race with mode changes and result in
      // mismatched image pool and desktop texture sizes. If this happens, just reinit again.
      if (desc.Width != width_before_rotation || desc.Height != height_before_rotation) {
        BOOST_LOG(info) << "Capture size changed ["sv << width << 'x' << height << " -> "sv << desc.Width << 'x' << desc.Height << ']';
        return capture_e::reinit;
      }

      // If we don't know the capture format yet, grab it from this texture
      if (capture_format == DXGI_FORMAT_UNKNOWN) {
        capture_format = desc.Format;
        BOOST_LOG(info) << "Capture format ["sv << dxgi_format_to_string(capture_format) << ']';
      }

      // It's also possible for the capture format to change on the fly. If that happens,
      // reinitialize capture to try format detection again and create new images.
      if (capture_format != desc.Format) {
        BOOST_LOG(info) << "Capture format changed ["sv << dxgi_format_to_string(capture_format) << " -> "sv << dxgi_format_to_string(desc.Format) << ']';
        return capture_e::reinit;
      }

      pending_damage = dup.damage_update(
        frame_info,
        display_rotation,
        width_before_rotation,
        height_before_rotation
      );
      if (!prior_damage_chain_valid) {
        pending_damage->known = false;
        pending_damage->rects.clear();
      }
    }

    auto commit_desktop_surface = [&]() {
      if (!pending_damage || !present_timestamp) {
        return;
      }

      if (damage_history) {
        auto snapshot = damage_history->commit(std::move(*pending_damage));
        if (snapshot.history && snapshot.token != 0u) {
          last_ddup_damage = std::move(snapshot);
        } else {
          last_ddup_damage.reset();
        }
      } else {
        last_ddup_damage.reset();
      }
      pending_damage.reset();
      last_content_timestamp = present_timestamp;
      damage_chain_valid = true;
    };

    enum class lfa {
      nothing,
      replace_surface_with_img,
      replace_img_with_surface,
      copy_src_to_img,
      copy_src_to_surface,
    };

    enum class ofa {
      forward_last_img,
      copy_last_surface_and_blend_cursor,
      dummy_fallback,
    };

    auto last_frame_action = lfa::nothing;
    auto out_frame_action = ofa::dummy_fallback;

    if (capture_format == DXGI_FORMAT_UNKNOWN) {
      // We don't know the final capture format yet, so we will encode a black dummy image
      last_frame_action = lfa::nothing;
      out_frame_action = ofa::dummy_fallback;
    } else {
      if (src) {
        // We got a new frame from DesktopDuplication...
        if (blend_mouse_cursor_flag) {
          // ...and we need to blend the mouse cursor onto it.
          // Copy the frame to intermediate surface so we can blend this and future mouse cursor updates
          // without new frames from DesktopDuplication. We use direct3d surface directly here and not
          // an image from pull_free_image_cb mainly because it's lighter (surface sharing between
          // direct3d devices produce significant memory overhead).
          last_frame_action = lfa::copy_src_to_surface;
          // Copy the intermediate surface to a new image from pull_free_image_cb and blend the mouse cursor onto it.
          out_frame_action = ofa::copy_last_surface_and_blend_cursor;
        } else {
          // ...and we don't need to blend the mouse cursor.
          // Copy the frame to a new image from pull_free_image_cb and save the shared pointer to the image
          // in case the mouse cursor appears without a new frame from DesktopDuplication.
          last_frame_action = lfa::copy_src_to_img;
          // Use saved last image shared pointer as output image evading copy.
          out_frame_action = ofa::forward_last_img;
        }
      } else if (!std::holds_alternative<std::monostate>(last_frame_variant)) {
        // We didn't get a new frame from DesktopDuplication...
        if (blend_mouse_cursor_flag) {
          // ...but we need to blend the mouse cursor.
          if (std::holds_alternative<std::shared_ptr<platf::img_t>>(last_frame_variant)) {
            // We have the shared pointer of the last image, replace it with intermediate surface
            // while copying contents so we can blend this and future mouse cursor updates.
            last_frame_action = lfa::replace_img_with_surface;
          }
          // Copy the intermediate surface which contains last DesktopDuplication frame
          // to a new image from pull_free_image_cb and blend the mouse cursor onto it.
          out_frame_action = ofa::copy_last_surface_and_blend_cursor;
        } else {
          // ...and we don't need to blend the mouse cursor.
          // This happens when the mouse cursor disappears from screen,
          // or there's mouse cursor on screen, but its drawing is disabled in sunshine.
          if (std::holds_alternative<texture2d_t>(last_frame_variant)) {
            // We have the intermediate surface that was used as the mouse cursor blending base.
            // Replace it with an image from pull_free_image_cb copying contents and freeing up the surface memory.
            // Save the shared pointer to the image in case the mouse cursor reappears.
            last_frame_action = lfa::replace_surface_with_img;
          }
          // Use saved last image shared pointer as output image evading copy.
          out_frame_action = ofa::forward_last_img;
        }
      }
    }

    auto create_surface = [&](texture2d_t &surface) -> bool {
      // Try to reuse the old surface if it hasn't been destroyed yet.
      if (old_surface_delayed_destruction) {
        surface.reset(old_surface_delayed_destruction.release());
        return true;
      }

      // Otherwise create a new surface.
      D3D11_TEXTURE2D_DESC t {};
      t.Width = width_before_rotation;
      t.Height = height_before_rotation;
      t.MipLevels = 1;
      t.ArraySize = 1;
      t.SampleDesc.Count = 1;
      t.Usage = D3D11_USAGE_DEFAULT;
      t.Format = capture_format;
      t.BindFlags = 0;
      status = device->CreateTexture2D(&t, nullptr, &surface);
      if (FAILED(status)) {
        BOOST_LOG(error) << "Failed to create frame copy texture [0x"sv << util::hex(status).to_string_view() << ']';
        return false;
      }

      return true;
    };

    auto get_locked_d3d_img = [&](std::shared_ptr<platf::img_t> &img, bool dummy = false) -> std::tuple<std::shared_ptr<img_d3d_t>, texture_lock_helper> {
      auto d3d_img = std::static_pointer_cast<img_d3d_t>(img);

      // Finish creating the image (if it hasn't happened already),
      // also creates synchronization primitives for shared access from multiple direct3d devices.
      if (complete_img(d3d_img.get(), dummy)) {
        return {nullptr, nullptr};
      }

      // This image is shared between capture direct3d device and encoders direct3d devices,
      // we must acquire lock before doing anything to it.
      texture_lock_helper lock_helper(d3d_img->capture_mutex.get());
      if (!lock_helper.lock()) {
        BOOST_LOG(error) << "Failed to lock capture texture";
        return {nullptr, nullptr};
      }

      // Clear the blank flag now that we're ready to capture into the image
      d3d_img->blank = false;

      return {std::move(d3d_img), std::move(lock_helper)};
    };

    switch (last_frame_action) {
      case lfa::nothing:
        {
          break;
        }

      case lfa::replace_surface_with_img:
        {
          auto p_surface = std::get_if<texture2d_t>(&last_frame_variant);
          if (!p_surface) {
            BOOST_LOG(error) << "Logical error at " << __FILE__ << ":" << __LINE__;
            return capture_e::error;
          }

          std::shared_ptr<platf::img_t> img;
          if (!pull_free_image_cb(img)) {
            return capture_e::interrupted;
          }

          auto [d3d_img, lock] = get_locked_d3d_img(img);
          if (!d3d_img) {
            return capture_e::error;
          }

          device_ctx->CopyResource(d3d_img->capture_texture.get(), p_surface->get());
          d3d_img->ddup_damage = last_ddup_damage;

          // We delay the destruction of intermediate surface in case the mouse cursor reappears shortly.
          old_surface_delayed_destruction.reset(p_surface->release());
          old_surface_timestamp = std::chrono::steady_clock::now();

          last_frame_variant = img;
          break;
        }

      case lfa::replace_img_with_surface:
        {
          auto p_img = std::get_if<std::shared_ptr<platf::img_t>>(&last_frame_variant);
          if (!p_img) {
            BOOST_LOG(error) << "Logical error at " << __FILE__ << ":" << __LINE__;
            return capture_e::error;
          }
          auto [d3d_img, lock] = get_locked_d3d_img(*p_img);
          if (!d3d_img) {
            return capture_e::error;
          }

          p_img = nullptr;
          last_frame_variant = texture2d_t {};
          auto &surface = std::get<texture2d_t>(last_frame_variant);
          if (!create_surface(surface)) {
            return capture_e::error;
          }

          device_ctx->CopyResource(surface.get(), d3d_img->capture_texture.get());
          break;
        }

      case lfa::copy_src_to_img:
        {
          last_frame_variant = {};

          std::shared_ptr<platf::img_t> img;
          if (!pull_free_image_cb(img)) {
            return capture_e::interrupted;
          }

          auto [d3d_img, lock] = get_locked_d3d_img(img);
          if (!d3d_img) {
            return capture_e::error;
          }

          device_ctx->CopyResource(d3d_img->capture_texture.get(), src.get());
          commit_desktop_surface();
          d3d_img->ddup_damage = last_ddup_damage;
          last_frame_variant = img;
          break;
        }

      case lfa::copy_src_to_surface:
        {
          auto p_surface = std::get_if<texture2d_t>(&last_frame_variant);
          if (!p_surface) {
            last_frame_variant = texture2d_t {};
            p_surface = std::get_if<texture2d_t>(&last_frame_variant);
            if (!create_surface(*p_surface)) {
              return capture_e::error;
            }
          }
          device_ctx->CopyResource(p_surface->get(), src.get());
          commit_desktop_surface();
          break;
        }
    }

    auto blend_cursor = [&](img_d3d_t &d3d_img) {
      device_ctx->VSSetShader(cursor_vs.get(), nullptr, 0);
      device_ctx->PSSetShader(cursor_ps.get(), nullptr, 0);
      device_ctx->OMSetRenderTargets(1, &d3d_img.capture_rt, nullptr);

      if (cursor_alpha.texture.get()) {
        // Perform an alpha blending operation
        device_ctx->OMSetBlendState(blend_alpha.get(), nullptr, 0xFFFFFFFFu);

        device_ctx->PSSetShaderResources(0, 1, &cursor_alpha.input_res);
        device_ctx->RSSetViewports(1, &cursor_alpha.cursor_view);
        device_ctx->Draw(3, 0);
      }

      if (cursor_xor.texture.get()) {
        // Perform an invert blending without touching alpha values
        device_ctx->OMSetBlendState(blend_invert.get(), nullptr, 0x00FFFFFFu);

        device_ctx->PSSetShaderResources(0, 1, &cursor_xor.input_res);
        device_ctx->RSSetViewports(1, &cursor_xor.cursor_view);
        device_ctx->Draw(3, 0);
      }

      device_ctx->OMSetBlendState(blend_disable.get(), nullptr, 0xFFFFFFFFu);

      ID3D11RenderTargetView *emptyRenderTarget = nullptr;
      device_ctx->OMSetRenderTargets(1, &emptyRenderTarget, nullptr);
      device_ctx->RSSetViewports(0, nullptr);
      ID3D11ShaderResourceView *emptyShaderResourceView = nullptr;
      device_ctx->PSSetShaderResources(0, 1, &emptyShaderResourceView);
    };

    switch (out_frame_action) {
      case ofa::forward_last_img:
        {
          auto p_img = std::get_if<std::shared_ptr<platf::img_t>>(&last_frame_variant);
          if (!p_img) {
            BOOST_LOG(error) << "Logical error at " << __FILE__ << ":" << __LINE__;
            return capture_e::error;
          }
          img_out = *p_img;
          break;
        }

      case ofa::copy_last_surface_and_blend_cursor:
        {
          auto p_surface = std::get_if<texture2d_t>(&last_frame_variant);
          if (!p_surface) {
            BOOST_LOG(error) << "Logical error at " << __FILE__ << ":" << __LINE__;
            return capture_e::error;
          }
          if (!blend_mouse_cursor_flag) {
            BOOST_LOG(error) << "Logical error at " << __FILE__ << ":" << __LINE__;
            return capture_e::error;
          }

          if (!pull_free_image_cb(img_out)) {
            return capture_e::interrupted;
          }

          auto [d3d_img, lock] = get_locked_d3d_img(img_out);
          if (!d3d_img) {
            return capture_e::error;
          }

          device_ctx->CopyResource(d3d_img->capture_texture.get(), p_surface->get());
          d3d_img->ddup_damage = last_ddup_damage;
          blend_cursor(*d3d_img);
          break;
        }

      case ofa::dummy_fallback:
        {
          if (!pull_free_image_cb(img_out)) {
            return capture_e::interrupted;
          }

          // Clear the image if it has been used as a dummy.
          // It can have the mouse cursor blended onto it.
          auto old_d3d_img = (img_d3d_t *) img_out.get();
          bool reclear_dummy = !old_d3d_img->blank && old_d3d_img->capture_texture;

          auto [d3d_img, lock] = get_locked_d3d_img(img_out, true);
          if (!d3d_img) {
            return capture_e::error;
          }
          d3d_img->ddup_damage.reset();

          if (reclear_dummy) {
            const float rgb_black[] = {0.0f, 0.0f, 0.0f, 0.0f};
            device_ctx->ClearRenderTargetView(d3d_img->capture_rt.get(), rgb_black);
          }

          if (blend_mouse_cursor_flag) {
            blend_cursor(*d3d_img);
          }

          break;
        }
    }

    // Perform delayed destruction of the unused surface if the time is due.
    if (old_surface_delayed_destruction && old_surface_timestamp + 10s < std::chrono::steady_clock::now()) {
      old_surface_delayed_destruction.reset();
    }

    if (img_out) {
      img_out->frame_timestamp = frame_timestamp;
      img_out->content_timestamp = last_content_timestamp;
    }

    return capture_e::ok;
  }

  capture_e display_ddup_vram_t::release_snapshot() {
    return dup.release_frame();
  }

  int display_ddup_vram_t::init(const ::video::config_t &config, const std::string &display_name) {
    last_content_timestamp.reset();
    last_ddup_damage.reset();
    damage_history = std::make_shared<detail::ddup_damage_history_t>();
    damage_chain_valid = true;
    if (display_base_t::init(config, display_name, capture_backend_e::ddup) || dup.init(this, config)) {
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
    uint64_t frame_qpc;
    dup.set_cursor_visible(cursor_visible);
    auto capture_status = dup.next_frame(timeout, &src, frame_qpc);
    if (capture_status != capture_e::ok) {
      return capture_status;
    }

    auto frame_timestamp = std::chrono::steady_clock::now() - qpc_time_difference(qpc_counter(), frame_qpc);
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
    d3d_img->blank = false;  // image is always ready for capture
    if (complete_img(d3d_img.get(), false) == 0) {
      texture_lock_helper lock_helper(d3d_img->capture_mutex.get());
      if (lock_helper.lock()) {
        device_ctx->CopyResource(d3d_img->capture_texture.get(), src.get());
      } else {
        BOOST_LOG(error) << "Failed to lock capture texture";
        return capture_e::error;
      }
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

  std::unique_ptr<nvenc_encode_device_t> display_vram_t::make_nvenc_encode_device(pix_fmt_e pix_fmt) {
    auto device = std::make_unique<d3d_nvenc_encode_device_t>();
    if (!device->init_device(shared_from_this(), adapter.get(), pix_fmt)) {
      return nullptr;
    }
    return device;
  }

  int init() {
    BOOST_LOG(info) << "Compiling shaders..."sv;

#define compile_vertex_shader_helper(x) \
  if (!(x##_hlsl = compile_vertex_shader(SUNSHINE_SHADERS_DIR "/" #x ".hlsl"))) \
    return -1;
#define compile_pixel_shader_helper(x) \
  if (!(x##_hlsl = compile_pixel_shader(SUNSHINE_SHADERS_DIR "/" #x ".hlsl"))) \
    return -1;
#define compile_compute_shader_helper(x) \
  if (!(x##_hlsl = compile_compute_shader(SUNSHINE_SHADERS_DIR "/" #x ".hlsl"))) \
    return -1;
    compile_pixel_shader_helper(convert_yuv420_packed_uv_type0_ps);
    compile_pixel_shader_helper(convert_yuv420_packed_uv_type0_ps_linear);
    compile_pixel_shader_helper(convert_yuv420_packed_uv_type0_ps_perceptual_quantizer);
    compile_vertex_shader_helper(convert_yuv420_packed_uv_type0_vs);
    compile_pixel_shader_helper(convert_yuv420_packed_uv_type0s_ps);
    compile_pixel_shader_helper(convert_yuv420_packed_uv_type0s_ps_linear);
    compile_pixel_shader_helper(convert_yuv420_packed_uv_type0s_ps_perceptual_quantizer);
    compile_vertex_shader_helper(convert_yuv420_packed_uv_type0s_vs);
    compile_pixel_shader_helper(convert_yuv420_planar_y_ps);
    compile_pixel_shader_helper(convert_yuv420_planar_y_ps_linear);
    compile_pixel_shader_helper(convert_yuv420_planar_y_ps_perceptual_quantizer);
    compile_vertex_shader_helper(convert_yuv420_planar_y_vs);
    compile_pixel_shader_helper(cursor_ps);
    compile_pixel_shader_helper(cursor_ps_normalize_white);
    compile_pixel_shader_helper(rgb_present_linear_to_srgb_ps);
    compile_pixel_shader_helper(rgb_present_srgb_to_linear_ps);
    sbs_reprojection_v2_live_ps_hlsl.reset();
    sbs_reprojection_v2_live_source_closure_sha256.clear();
    sbs_reprojection_v2_p010_y_ps_hlsl.reset();
    sbs_reprojection_v2_p010_y_source_closure_sha256.clear();
    sbs_flat_identity_ps_hlsl.reset();
    sbs_reprojection_vs_hlsl.reset();
    namespace cache = models::host_sbs_shader_cache;
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
    compile_vertex_shader_helper(cursor_vs);

    BOOST_LOG(info) << "Compiled shaders"sv;

#undef compile_vertex_shader_helper
#undef compile_pixel_shader_helper
#undef compile_compute_shader_helper

    return 0;
  }
}  // namespace platf::dxgi
