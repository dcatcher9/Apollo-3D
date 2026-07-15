/**
 * @file src/platform/windows/display_vram.cpp
 * @brief Definitions for handling video ram.
 */
// standard includes
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <thread>

// platform includes
#include <d3dcompiler.h>
#include <DirectXMath.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext_d3d11va.h>
}

// lib includes
#include <AMF/core/Factory.h>
#include <boost/algorithm/string/predicate.hpp>

// local includes
#include "display.h"
#include "misc.h"
#include "sbs_debug_dump.h"
#include "src/config.h"
#include "src/logging.h"
#include "src/model_manager.h"
#include "src/nvenc/nvenc_config.h"
#include "src/nvenc/nvenc_d3d11_native.h"
#include "src/nvenc/nvenc_d3d11_on_cuda.h"
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

static void free_frame(AVFrame *frame) {
  av_frame_free(&frame);
}

using frame_t = util::safe_ptr<AVFrame, free_frame>;

namespace platf::dxgi {

  using d3d_query_t = util::safe_ptr<ID3D11Query, Release<ID3D11Query>>;

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
  blob_t convert_yuv444_packed_ayuv_ps_hlsl;
  blob_t convert_yuv444_packed_ayuv_ps_linear_hlsl;
  blob_t convert_yuv444_packed_vs_hlsl;
  blob_t convert_yuv444_planar_ps_hlsl;
  blob_t convert_yuv444_planar_ps_linear_hlsl;
  blob_t convert_yuv444_planar_ps_perceptual_quantizer_hlsl;
  blob_t convert_yuv444_packed_y410_ps_hlsl;
  blob_t convert_yuv444_packed_y410_ps_linear_hlsl;
  blob_t convert_yuv444_packed_y410_ps_perceptual_quantizer_hlsl;
  blob_t convert_yuv444_planar_vs_hlsl;
  blob_t cursor_ps_hlsl;
  blob_t cursor_ps_normalize_white_hlsl;
  blob_t cursor_vs_hlsl;
  blob_t depth_warp_prefilter_cs_hlsl;
  blob_t sbs_reprojection_ps_hlsl;
  blob_t sbs_reprojection_vs_hlsl;

  struct img_d3d_t: public platf::img_t {
    // These objects are owned by the display_t's ID3D11Device
    texture2d_t capture_texture;
    render_target_t capture_rt;
    keyed_mutex_t capture_mutex;

    // This is the shared handle used by hwdevice_t to open capture_texture
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

  // Keep abandoned per-device estimator builds alive without blocking a presentation-session
  // teardown. The manager joins any remaining workers during process shutdown, before the
  // function-static object and the TensorRT process globals are destroyed.
  class depth_estimator_build_manager_t {
  public:
    ~depth_estimator_build_manager_t() {
      for (auto &worker : workers_) {
        if (worker.thread.joinable()) {
          worker.thread.join();
        }
      }
    }

    void launch(depth_estimator_build_task_t task) {
      auto completed = std::make_shared<std::atomic<bool>>(false);
      std::thread thread([task = std::move(task), completed]() mutable {
        task();
        completed->store(true, std::memory_order_release);
      });

      std::lock_guard lock(mutex_);
      for (auto worker = workers_.begin(); worker != workers_.end();) {
        if (!worker->completed->load(std::memory_order_acquire)) {
          ++worker;
          continue;
        }
        worker->thread.join();
        worker = workers_.erase(worker);
      }
      workers_.push_back({std::move(thread), std::move(completed)});
    }

  private:
    struct worker_t {
      std::thread thread;
      std::shared_ptr<std::atomic<bool>> completed;
    };

    std::mutex mutex_;
    std::vector<worker_t> workers_;
  };

  depth_estimator_build_manager_t &depth_estimator_build_manager() {
    static depth_estimator_build_manager_t manager;
    return manager;
  }

  class d3d_base_encode_device final {
  public:
    ~d3d_base_encode_device() {
      // The encode session is already leaving the live path, so a short bounded drain cannot
      // stall capture. It preserves the final GPU timing samples while the generation token
      // prevents a racing replacement session from receiving this device's late results.
      drain_sbs_gpu_timers();

      // The background task owns D3D references and its packaged-task future does not block here.
      // If this device is torn down while construction is in flight, the process-level build
      // manager lets it finish independently and destroys the unused result on that worker.
    }

    int convert_rgb(
      platf::img_t &img,
      ID3D11Texture2D *target_texture,
      ID3D11RenderTargetView *target
    ) {
      rgb_present_texture = target_texture;
      rgb_present_target = target;
      auto clear_target = util::fail_guard([this]() {
        rgb_present_texture = nullptr;
        rgb_present_target = nullptr;
      });
      return convert(img);
    }

    int convert(platf::img_t &img_base) {
      auto &img = (img_d3d_t &) img_base;
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
        auto status = img_ctx.encoder_mutex->AcquireSync(0, INFINITE);
        if (status != S_OK) {
          BOOST_LOG(error) << "Failed to acquire encoder mutex [0x"sv << util::hex(status).to_string_view() << ']';
          return -1;
        }

        auto draw = [&](auto &input, auto &y_or_yuv_viewports, auto &uv_viewport, bool input_is_linear) {
          device_ctx->PSSetShaderResources(0, 1, &input);

          // Draw Y/YUV
          device_ctx->OMSetRenderTargets(1, &out_Y_or_YUV_rtv, nullptr);
          device_ctx->VSSetShader(convert_Y_or_YUV_vs.get(), nullptr, 0);
          device_ctx->PSSetShader(input_is_linear ? convert_Y_or_YUV_fp16_ps.get() : convert_Y_or_YUV_ps.get(), nullptr, 0);
          auto viewport_count = (format == DXGI_FORMAT_R16_UINT) ? 3 : 1;
          assert(viewport_count <= y_or_yuv_viewports.size());
          device_ctx->RSSetViewports(viewport_count, y_or_yuv_viewports.data());
          device_ctx->Draw(3 * viewport_count, 0);  // vertex shader will spread vertices across viewports

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

        auto draw_rgb = [&](ID3D11ShaderResourceView *input) {
          device_ctx->OMSetRenderTargets(1, &rgb_present_target, nullptr);
          device_ctx->VSSetShader(sbs_reprojection_vs.get(), nullptr, 0);
          device_ctx->PSSetShader(rgb_present_ps.get(), nullptr, 0);
          device_ctx->RSSetViewports(1, &rgb_present_viewport);
          device_ctx->PSSetSamplers(0, 1, &sampler_linear);
          device_ctx->PSSetShaderResources(0, 1, &input);
          device_ctx->Draw(3, 0);

          ID3D11RenderTargetView *null_rtv = nullptr;
          ID3D11ShaderResourceView *null_srv = nullptr;
          device_ctx->OMSetRenderTargets(1, &null_rtv, nullptr);
          device_ctx->PSSetShaderResources(0, 1, &null_srv);
        };

        auto copy_rgb = [&](ID3D11Texture2D *input) {
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
                                  input_desc.SampleDesc.Quality == output_desc.SampleDesc.Quality;
          if (!compatible) {
            if (!rgb_copy_fallback_logged) {
              BOOST_LOG(info) << "Local AR exact-copy path unavailable (source "sv
                              << input_desc.Width << 'x' << input_desc.Height << ' '
                              << (int) input_desc.Format << ", target "sv
                              << output_desc.Width << 'x' << output_desc.Height << ' '
                              << (int) output_desc.Format << "); using the RGB presentation shader."sv;
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

        // Clear render target view(s) once so that the aspect ratio mismatch "bars" appear black
        if (!rtvs_cleared) {
          auto black = create_black_texture_for_rtv_clear();
          if (black) {
            draw(black, out_Y_or_YUV_viewports_for_clear, out_UV_viewport_for_clear, false);
          }
          rtvs_cleared = true;
        }

        if (sbs_mode != ::video::SBS_OFF) {
          // Perf benchmark: CPU wall time of the whole SBS block (estimator dispatch + composite
          // draw submission). GPU-side inference times are measured separately via CUDA events in
          // the estimator. No-op unless sbs_3d_perf_stats is on.
          const bool perf = sbs_perf::enabled();
          const auto perf_t0 = perf ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};

          // Lazy-create the depth estimator on the first SBS frame.
          ensure_depth_estimator();
          auto *gpu_timer = perf ? begin_sbs_gpu_timer() : nullptr;

          // Production always uses bounded matched pairing: infer asynchronously from a private
          // color slot, then warp only the slot whose frame identity completed.
          const auto input_color_space = img.format == DXGI_FORMAT_R16G16B16A16_FLOAT ?
                                           (display->is_hdr() ? models::input_color_space::scrgb_hdr :
                                                                models::input_color_space::linear_sdr) :
                                           models::input_color_space::srgb;
          const auto frame_id = ++sbs_frame_sequence;
          ID3D11ShaderResourceView *render_input_srv = img_ctx.encoder_input_res.get();
          matched_frame_slot_t *matched_render_slot = nullptr;
          matched_frame_slot_t *matched_candidate_slot = nullptr;
          models::estimate_result est;

          if (depth_estimator) {
            matched_candidate_slot = available_matched_slot();
            const bool matched_copy_submitted =
              matched_candidate_slot &&
              copy_matched_frame(img_ctx.encoder_texture.get(), *matched_candidate_slot, frame_id);
            mark_sbs_matched_copy_end(gpu_timer, matched_copy_submitted);
            if (matched_copy_submitted) {
              est = depth_estimator->estimate_depth(matched_candidate_slot->srv.get(), input_color_space, frame_id);
              if (est.completed_frame_valid) {
                matched_render_slot = find_pending_matched_slot(est.completed_frame_id);
                if (matched_render_slot) {
                  matched_render_slot->pending = false;
                  render_input_srv = matched_render_slot->srv.get();
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
                } else {
                  BOOST_LOG(error) << "Matched depth completed unknown frame "sv
                                   << est.completed_frame_id << "; repeating the last output."sv;
                  // The completed inference is no longer using its source texture. Recover the
                  // old pending slot so a metadata mismatch cannot permanently exhaust the
                  // bounded two-slot queue. The current candidate is handled below if enqueued.
                  for (auto &slot : matched_frame_slots) {
                    if (&slot != matched_candidate_slot) {
                      slot.pending = false;
                    }
                  }
                }
              }
              if (est.inference_enqueued) {
                matched_candidate_slot->pending = true;
              }
            }
          } else {
            mark_sbs_matched_copy_end(gpu_timer, false);
          }

          const bool repeat_matched_output = !matched_render_slot && matched_output_valid;
          const bool timing_has_depth_warp = !repeat_matched_output && est.depth;
          mark_sbs_warp_start(gpu_timer, timing_has_depth_warp);
          ID3D11ShaderResourceView *final_sbs_srv = nullptr;
          ID3D11Texture2D *final_sbs_texture = nullptr;
          if (repeat_matched_output) {
            final_sbs_srv = sbs_intermediate_srv.get();
            final_sbs_texture = sbs_intermediate_texture.get();
            ++matched_stats_repeats;
          } else {
            // Before the first matched completion, provide a flat current-frame SBS image. Once a
            // pair has completed, this branch only renders the buffered color that owns est.depth.
            if (!matched_render_slot) {
              est = {};
              render_input_srv = img_ctx.encoder_input_res.get();
            }

            ID3D11ShaderResourceView *warp_depth = est.depth.Get();
            if (est.depth) {
              if (auto *filtered_depth = prefilter_warp_depth(est.depth.Get())) {
                warp_depth = filtered_depth;
              }
            }

            // Draw Apollo's occlusion-aware geometry into the shared SBS intermediate.
            device_ctx->OMSetRenderTargets(1, &sbs_intermediate_rtv, nullptr);
            device_ctx->VSSetShader(sbs_reprojection_vs.get(), nullptr, 0);
            device_ctx->PSSetShader(sbs_reprojection_ps.get(), nullptr, 0);
            device_ctx->RSSetViewports(1, &sbs_viewport);
            // Bind the sampler explicitly rather than relying on it persisting from init().
            device_ctx->PSSetSamplers(0, 1, &sampler_linear);

            ID3D11ShaderResourceView *srvs[] = {render_input_srv, warp_depth, est.subject.Get()};
            device_ctx->PSSetShaderResources(0, 3, srvs);
            ID3D11Buffer *sbs_cb[] = {sbs_reprojection_cbuffer.get()};
            device_ctx->PSSetConstantBuffers(2, 1, sbs_cb);
            device_ctx->Draw(3, 0);  // Fullscreen triangle

            // Unbind the Render Target so D3D11 doesn't nullify our SRV in the next pass!
            ID3D11RenderTargetView *null_rtvs[] = {nullptr};
            device_ctx->OMSetRenderTargets(1, null_rtvs, nullptr);

            // Clear shader resources
            ID3D11ShaderResourceView *null_srvs[] = {nullptr, nullptr, nullptr};
            device_ctx->PSSetShaderResources(0, 3, null_srvs);

            final_sbs_srv = sbs_intermediate_srv.get();
            final_sbs_texture = sbs_intermediate_texture.get();
            if (matched_render_slot && est.depth) {
              matched_output_valid = true;
            }
          }

          mark_sbs_warp_end(gpu_timer);

          if (rgb_present_target) {
            // The local AR presenter consumes the production RGB warp directly, avoiding an
            // encode/decode round trip. Exact layouts take the copy fast path; retain the shader
            // for any future mode whose source and physical target require scaling or conversion.
            if (!copy_rgb(final_sbs_texture)) {
              draw_rgb(final_sbs_srv);
            }
          } else {
            // Draw the SBS intermediate into encoder YUV.
            draw(final_sbs_srv, out_Y_or_YUV_viewports, out_UV_viewport, sbs_intermediate_linear);
          }
          end_sbs_gpu_timer(gpu_timer);

          // Debug frame dump (offline artifact inspection): on the client "Dump 3D" button or a
          // "dump.trigger" file, save this frame's 2D source, depth map and SBS result. See
          // sbs_debug_dump.h. No-op unless APOLLO_SBS_DUMP is set.
          if (!repeat_matched_output) {
            sbs_dumper.maybe_dump(device.get(), device_ctx.get(), render_input_srv, est.depth.Get(), final_sbs_srv, display->is_hdr(), sbs_config.depth_model);
          }

          ++matched_stats_calls;
          const auto now = std::chrono::steady_clock::now();
          if (now - matched_stats_started >= std::chrono::seconds(5)) {
            const double avg_age_ms = matched_stats_completions ?
                                        matched_stats_age_sum_ms / matched_stats_completions :
                                        0.0;
            BOOST_LOG(info) << "SBS matched-frame stats: calls="sv << matched_stats_calls
                            << " completed="sv << matched_stats_completions
                            << " repeats="sv << matched_stats_repeats
                            << " age_avg_ms="sv << avg_age_ms
                            << " age_max_ms="sv << matched_stats_age_max_ms;
            reset_matched_stats(now);
          }

          if (perf) {
            auto dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - perf_t0).count();
            sbs_perf::add_sample_ms("sbs_convert_cpu", dt);
            sbs_perf::tick();  // once per SBS frame: periodic p50/p95 summary + JSON snapshot
          }
        } else {
          if (rgb_present_target) {
            if (!copy_rgb(img_ctx.encoder_texture.get())) {
              draw_rgb(img_ctx.encoder_input_res.get());
            }
          } else {
            // Plain 2D: draw the captured frame straight into the encoder output.
            draw(img_ctx.encoder_input_res, out_Y_or_YUV_viewports, out_UV_viewport, img.format == DXGI_FORMAT_R16G16B16A16_FLOAT);
          }
        }

        // Release encoder mutex to allow capture code to reuse this image
        img_ctx.encoder_mutex->ReleaseSync(0);

        ID3D11ShaderResourceView *emptyShaderResourceView = nullptr;
        device_ctx->PSSetShaderResources(0, 1, &emptyShaderResourceView);
      }

      return 0;
    }

    void apply_colorspace(const ::video::sunshine_colorspace_t &colorspace) {
      auto color_vectors = ::video::color_vectors_from_colorspace(colorspace);

      if (format == DXGI_FORMAT_AYUV || format == DXGI_FORMAT_R16_UINT || format == DXGI_FORMAT_Y410) {
        color_vectors = ::video::new_color_vectors_from_colorspace(colorspace);
      }

      if (!color_vectors) {
        BOOST_LOG(error) << "No vector data for colorspace"sv;
        return;
      }

      auto color_matrix = make_buffer(device.get(), *color_vectors);
      if (!color_matrix) {
        BOOST_LOG(warning) << "Failed to create color matrix"sv;
        return;
      }

      device_ctx->VSSetConstantBuffers(3, 1, &color_matrix);
      device_ctx->PSSetConstantBuffers(0, 1, &color_matrix);
      this->color_matrix = std::move(color_matrix);
    }

    std::string make_depth_estimator_fingerprint(
      std::uint32_t eye_width,
      std::uint32_t eye_height,
      float content_scale_x,
      float content_scale_y
    ) const {
      const auto model = ::video::depth_model_for_profile(sbs_config);
      std::ostringstream stream;
      stream << std::setprecision(std::numeric_limits<double>::max_digits10)
             << model.name << '\n' << model.url << '\n'
             << display->width << 'x' << display->height << '\n'
             << eye_width << 'x' << eye_height << '\n'
             << content_scale_x << ',' << content_scale_y << '\n'
             << sbs_intermediate_linear << '\n'
             << sbs_config.profile << '\n'
             << sbs_config.pop_strength << '\n'
             << sbs_config.adaptive_pop << '\n'
             << sbs_config.adaptive_pop_max << '\n'
             << sbs_config.ema << '\n'
             << sbs_config.ema_edge_change << '\n'
             << sbs_config.ema_edge_gradient << '\n'
             << sbs_config.ema_edge_strength << '\n'
             << sbs_config.minmax_ema << '\n'
             << sbs_config.subject_lock << '\n'
             << sbs_config.subject_recenter << '\n'
             << sbs_config.subject_stretch << '\n'
             << sbs_config.depth_short_side << '\n'
             << sbs_config.depth_max_aspect << '\n'
             << sbs_config.zero_plane << '\n'
             << sbs_config.cuda_graph << '\n'
             << sbs_config.perf_stats << '\n'
             << sbs_config.artistic_style << '\n'
             << sbs_config.artistic_live_review;
      return stream.str();
    }

    void refresh_depth_estimator_fingerprint(
      std::uint32_t eye_width,
      std::uint32_t eye_height,
      float content_scale_x,
      float content_scale_y
    ) {
      const std::string next = make_depth_estimator_fingerprint(
        eye_width,
        eye_height,
        content_scale_x,
        content_scale_y
      );
      if (!depth_estimator_fingerprint.empty() && depth_estimator_fingerprint != next) {
        depth_estimator.reset();
        depth_estimator_failed = false;
        engine_poll_counter = 0;
        publish_depth_status(0);
        BOOST_LOG(info) << "Host SBS estimator configuration/output geometry changed; the old "
                           "device pipeline will be rebuilt while the process-global model stays "
                           "resident.";
      }
      depth_estimator_fingerprint = next;
    }

    // Create the D3D depth pipeline on demand (first SBS frame). The heavy TensorRT engine,
    // execution context, and CUDA modules were prepared at host startup; construction now borrows
    // that warm context and creates only device/session resources on a background thread.
    bool ensure_depth_estimator() {
      if (depth_estimator) {
        return true;
      }

      // A failed build streams flat SBS for the rest of this encode device's life instead of
      // re-kicking a doomed build every frame; a later mode rebuild creates a fresh device.
      if (depth_estimator_failed) {
        return false;
      }

      // A build is already in flight: take the result once ready, otherwise keep streaming flat.
      if (depth_estimator_building) {
        if (depth_estimator_build.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
          depth_estimator_building = false;
          const bool stale_build = depth_estimator_build_fingerprint !=
                                   depth_estimator_fingerprint;
          try {
            auto built_estimator = depth_estimator_build.get();
            if (stale_build) {
              BOOST_LOG(info) << "Discarding a depth estimator completed for an obsolete SBS "
                                 "configuration/output geometry.";
              built_estimator.reset();
              publish_depth_status(0);
              return false;
            }
            depth_estimator = std::move(built_estimator);
          } catch (const std::exception &e) {
            if (stale_build) {
              BOOST_LOG(info) << "Ignoring a failed depth estimator build for an obsolete SBS "
                                 "configuration/output geometry: "sv
                              << e.what();
              publish_depth_status(0);
              return false;
            }
            // Don't let a background-build exception propagate on the encode thread (it would end
            // the stream); log it, clear the client's "loading" indicator, and stream flat.
            BOOST_LOG(error) << "Depth estimator build failed: "sv << e.what();
            depth_estimator_failed = true;
            publish_depth_status(0);
            return false;
          }
          if (depth_estimator && depth_estimator->is_valid()) {
            BOOST_LOG(info) << "Depth estimator ready; host SBS depth is now live."sv;
            publish_depth_status(2);  // ready -> client hides indicator
          } else {
            BOOST_LOG(error) << "Depth estimator build returned an invalid pipeline; streaming flat SBS."sv;
            depth_estimator.reset();
            depth_estimator_failed = true;
            publish_depth_status(0);
          }
          return (bool) depth_estimator;
        }
        return false;
      }

      // Startup model preparation is process-global and may take minutes on first use. It includes
      // engine compilation, deserialization, execution-context creation, and CUDA warmup. Report
      // that complete wait as loading, but don't duplicate it on this encode device.
      auto active = ::video::depth_model_for_profile(sbs_config);
      auto build_status = models::tensorrt_model_prepare_status(active);
      if (build_status != models::engine_build_status::ready) {
        if (build_status == models::engine_build_status::failed) {
          BOOST_LOG(error) << "Startup TensorRT model preparation failed for '"sv << active.name
                           << "'; streaming flat SBS."sv;
          depth_estimator_failed = true;
          publish_depth_status(0);
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
      const auto artistic_eye_width = (std::uint32_t) std::lround(sbs_viewport.Width * 0.5f);
      const auto artistic_eye_height = (std::uint32_t) std::lround(sbs_viewport.Height);
      const float artistic_content_scale_x = sbs_content_scale_x;
      const float artistic_content_scale_y = sbs_content_scale_y;
      BOOST_LOG(info) << "Host SBS enabled; building depth model \""sv << active.name
                      << "\" for matched-frame presentation in the background "
                         "(streaming flat until ready)..."sv;
      Microsoft::WRL::ComPtr<ID3D11Device> dev(device.get());
      Microsoft::WRL::ComPtr<ID3D11DeviceContext> ctx(device_ctx.get());
      depth_estimator_build_task_t build_task([dev = std::move(dev), ctx = std::move(ctx), active, sbs_cfg,
                                                artistic_eye_width, artistic_eye_height,
                                                artistic_content_scale_x,
                                                artistic_content_scale_y]() mutable {
        auto estimator = std::make_unique<models::video_depth_estimator>(
          std::move(dev),
          std::move(ctx),
          std::filesystem::path(SUNSHINE_ASSETS_DIR),
          sbs_cfg,
          active,
          true,
          0.0f,
          sbs_cfg.artistic_live_review ?
            models::artistic_policy_authorization::headset_review :
            models::artistic_policy_authorization::deployment
        );
        estimator->set_artistic_output_geometry(
          artistic_eye_width,
          artistic_eye_height,
          artistic_content_scale_x,
          artistic_content_scale_y
        );
        return estimator;
      });
      depth_estimator_build = build_task.get_future();
      depth_estimator_build_fingerprint = depth_estimator_fingerprint;
      depth_estimator_build_manager().launch(std::move(build_task));
      depth_estimator_building = true;
      publish_depth_status(3);  // device-specific pipeline initialization (engine is already ready)
      return false;
    }

    void publish_depth_status(int phase) {
      if (sbs_depth_status_event) {
        sbs_depth_status_event->raise(phase);
      }
    }

    void update_sbs_constant_buffer(float content_scale_x, float content_scale_y) {
      float sbs_params[8] {
        (float) sbs_config.subject_lock,
        sbs_config.subject_stretch ? 1.0f : 0.0f,
        content_scale_x,
        content_scale_y,
        (float) sbs_config.pop_strength,
        0.0f,
        sbs_config.adaptive_pop ? 1.0f : 0.0f,
        (float) std::max(sbs_config.adaptive_pop_max, sbs_config.pop_strength)
      };
      sbs_reprojection_cbuffer = make_buffer(device.get(), sbs_params);
    }

    struct sbs_gpu_timer_slot_t {
      d3d_query_t disjoint;
      d3d_query_t start;
      d3d_query_t matched_copy_end;
      d3d_query_t warp_start;
      d3d_query_t warp_end;
      d3d_query_t convert_end;
      bool pending = false;
      bool has_matched_copy = false;
      bool has_depth_warp = false;
      std::uint64_t perf_generation = 0;
    };

    void initialize_sbs_gpu_timers() {
      sbs_gpu_timing_ready = false;
      sbs_gpu_timer_next = 0;
      for (auto &slot : sbs_gpu_timers) {
        slot = {};
      }
      if (!sbs_config.perf_stats) {
        return;
      }

      for (auto &slot : sbs_gpu_timers) {
        D3D11_QUERY_DESC desc {D3D11_QUERY_TIMESTAMP_DISJOINT, 0};
        if (FAILED(device->CreateQuery(&desc, &slot.disjoint))) {
          BOOST_LOG(warning) << "Host SBS GPU timing unavailable: could not create disjoint query."sv;
          return;
        }
        desc.Query = D3D11_QUERY_TIMESTAMP;
        if (FAILED(device->CreateQuery(&desc, &slot.start)) || FAILED(device->CreateQuery(&desc, &slot.matched_copy_end)) || FAILED(device->CreateQuery(&desc, &slot.warp_start)) || FAILED(device->CreateQuery(&desc, &slot.warp_end)) || FAILED(device->CreateQuery(&desc, &slot.convert_end))) {
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
        if (FAILED(ready)) {
          slot.pending = false;
          continue;
        }

        UINT64 start = 0;
        UINT64 matched_copy_end = 0;
        UINT64 warp_start = 0;
        UINT64 warp_end = 0;
        UINT64 convert_end = 0;
        const auto start_status = device_ctx->GetData(slot.start.get(), &start, sizeof(start), 0);
        const auto matched_copy_status = device_ctx->GetData(slot.matched_copy_end.get(), &matched_copy_end, sizeof(matched_copy_end), 0);
        const auto warp_start_status = device_ctx->GetData(slot.warp_start.get(), &warp_start, sizeof(warp_start), 0);
        const auto warp_status = device_ctx->GetData(slot.warp_end.get(), &warp_end, sizeof(warp_end), 0);
        const auto convert_status = device_ctx->GetData(slot.convert_end.get(), &convert_end, sizeof(convert_end), 0);
        if (SUCCEEDED(start_status) && SUCCEEDED(matched_copy_status) && SUCCEEDED(warp_start_status) && SUCCEEDED(warp_status) && SUCCEEDED(convert_status) && !timing.Disjoint && timing.Frequency > 0 && matched_copy_end >= start && warp_start >= matched_copy_end && warp_end >= warp_start && convert_end >= warp_end) {
          const double to_ms = 1000.0 / static_cast<double>(timing.Frequency);
          if (slot.has_matched_copy) {
            sbs_perf::add_sample_ms_if_current(
              "matched_frame_copy_gpu",
              static_cast<double>(matched_copy_end - start) * to_ms,
              slot.perf_generation
            );
          }
          if (slot.has_depth_warp) {
            sbs_perf::add_sample_ms_if_current(
              "sbs_warp_gpu",
              static_cast<double>(warp_end - warp_start) * to_ms,
              slot.perf_generation
            );
          }
          sbs_perf::add_sample_ms_if_current(
            "sbs_color_convert_gpu",
            static_cast<double>(convert_end - warp_end) * to_ms,
            slot.perf_generation
          );
          sbs_perf::add_sample_ms_if_current(
            "sbs_render_gpu",
            static_cast<double>(convert_end - warp_start) * to_ms,
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
        slot.has_matched_copy = false;
        slot.has_depth_warp = false;
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

    struct matched_frame_slot_t {
      texture2d_t texture;
      shader_res_t srv;
      std::uint64_t frame_id = 0;
      std::chrono::steady_clock::time_point captured_at {};
      bool pending = false;
    };

    matched_frame_slot_t *available_matched_slot() {
      for (auto &slot : matched_frame_slots) {
        if (!slot.pending) {
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

    bool copy_matched_frame(ID3D11Texture2D *source, matched_frame_slot_t &slot, std::uint64_t frame_id) {
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

      device_ctx->CopyResource(slot.texture.get(), source);
      slot.frame_id = frame_id;
      slot.captured_at = std::chrono::steady_clock::now();
      slot.pending = false;
      return true;
    }

    void mark_sbs_matched_copy_end(sbs_gpu_timer_slot_t *slot, bool submitted) {
      if (slot) {
        slot->has_matched_copy = submitted;
        device_ctx->End(slot->matched_copy_end.get());
      }
    }

    void mark_sbs_warp_start(sbs_gpu_timer_slot_t *slot, bool has_depth_warp) {
      if (slot) {
        slot->has_depth_warp = has_depth_warp;
        device_ctx->End(slot->warp_start.get());
      }
    }

    ID3D11ShaderResourceView *prefilter_warp_depth(ID3D11ShaderResourceView *source_srv) {
      if (!source_srv || !sbs_depth_prefilter_cs) {
        return nullptr;
      }

      Microsoft::WRL::ComPtr<ID3D11Resource> source_resource;
      source_srv->GetResource(&source_resource);
      Microsoft::WRL::ComPtr<ID3D11Texture2D> source_texture;
      if (FAILED(source_resource.As(&source_texture))) {
        return nullptr;
      }

      D3D11_TEXTURE2D_DESC source_desc {};
      source_texture->GetDesc(&source_desc);
      bool recreate = !sbs_warp_depth_texture || !sbs_warp_depth_uav || !sbs_warp_depth_srv;
      if (!recreate) {
        D3D11_TEXTURE2D_DESC current_desc {};
        sbs_warp_depth_texture->GetDesc(&current_desc);
        recreate = current_desc.Width != source_desc.Width ||
                   current_desc.Height != source_desc.Height ||
                   current_desc.Format != source_desc.Format;
      }
      if (recreate) {
        auto filtered_desc = source_desc;
        filtered_desc.Usage = D3D11_USAGE_DEFAULT;
        filtered_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        filtered_desc.CPUAccessFlags = 0;
        filtered_desc.MiscFlags = 0;
        sbs_warp_depth_texture.reset();
        sbs_warp_depth_uav.reset();
        sbs_warp_depth_srv.reset();
        if (FAILED(device->CreateTexture2D(&filtered_desc, nullptr, &sbs_warp_depth_texture)) || FAILED(device->CreateUnorderedAccessView(sbs_warp_depth_texture.get(), nullptr, &sbs_warp_depth_uav)) || FAILED(device->CreateShaderResourceView(sbs_warp_depth_texture.get(), nullptr, &sbs_warp_depth_srv))) {
          BOOST_LOG(error) << "Failed to create the SBS warp-depth prefilter resources."sv;
          return nullptr;
        }
      }

      device_ctx->CSSetShader(sbs_depth_prefilter_cs.get(), nullptr, 0);
      device_ctx->CSSetShaderResources(0, 1, &source_srv);
      device_ctx->CSSetUnorderedAccessViews(0, 1, &sbs_warp_depth_uav, nullptr);
      device_ctx->Dispatch((source_desc.Width + 15u) / 16u, (source_desc.Height + 15u) / 16u, 1u);
      ID3D11UnorderedAccessView *null_uav = nullptr;
      ID3D11ShaderResourceView *null_srv = nullptr;
      device_ctx->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
      device_ctx->CSSetShaderResources(0, 1, &null_srv);
      return sbs_warp_depth_srv.get();
    }

    void reset_matched_stats(std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now()) {
      matched_stats_started = now;
      matched_stats_calls = 0;
      matched_stats_completions = 0;
      matched_stats_repeats = 0;
      matched_stats_age_sum_ms = 0.0;
      matched_stats_age_max_ms = 0.0;
    }

    int init_output(
      ID3D11Texture2D *frame_texture,
      int width,
      int height,
      int sbs_mode_param = ::video::SBS_OFF,
      const config::video_t::sbs_t &profile = {},
      safe::mail_raw_t::event_t<int> depth_status_event = {},
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
      sbs_config = profile;
      sbs_depth_status_event = std::move(depth_status_event);
      matched_frame_slots = {};
      sbs_frame_sequence = 0;
      matched_output_valid = false;
      reset_matched_stats();
      // WGC uses FP16 for both HDR and 10-bit SDR capture. In either case the texture is linear;
      // HDR only changes the later linear-scRGB -> Rec.2020/PQ conversion.
      sbs_intermediate_linear = display->is_hdr() ||
                                display->capture_format == DXGI_FORMAT_R16G16B16A16_FLOAT;

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
      const bool downscaling = display->width > width || display->height > height;

      create_pixel_shader_helper(cursor_ps_hlsl, rgb_present_ps);
      create_vertex_shader_helper(sbs_reprojection_vs_hlsl, sbs_reprojection_vs);
      if (sbs_on) {
        create_compute_shader_helper(depth_warp_prefilter_cs_hlsl, sbs_depth_prefilter_cs);
        create_pixel_shader_helper(sbs_reprojection_ps_hlsl, sbs_reprojection_ps);
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
            if (display->is_hdr()) {
              create_pixel_shader_helper(convert_yuv420_planar_y_ps_perceptual_quantizer_hlsl, convert_Y_or_YUV_fp16_ps);
            } else {
              create_pixel_shader_helper(convert_yuv420_planar_y_ps_linear_hlsl, convert_Y_or_YUV_fp16_ps);
            }
            if (downscaling) {
              create_vertex_shader_helper(convert_yuv420_packed_uv_type0s_vs_hlsl, convert_UV_vs);
              create_pixel_shader_helper(convert_yuv420_packed_uv_type0s_ps_hlsl, convert_UV_ps);
              if (display->is_hdr()) {
                create_pixel_shader_helper(convert_yuv420_packed_uv_type0s_ps_perceptual_quantizer_hlsl, convert_UV_fp16_ps);
              } else {
                create_pixel_shader_helper(convert_yuv420_packed_uv_type0s_ps_linear_hlsl, convert_UV_fp16_ps);
              }
            } else {
              create_vertex_shader_helper(convert_yuv420_packed_uv_type0_vs_hlsl, convert_UV_vs);
              create_pixel_shader_helper(convert_yuv420_packed_uv_type0_ps_hlsl, convert_UV_ps);
              if (display->is_hdr()) {
                create_pixel_shader_helper(convert_yuv420_packed_uv_type0_ps_perceptual_quantizer_hlsl, convert_UV_fp16_ps);
              } else {
                create_pixel_shader_helper(convert_yuv420_packed_uv_type0_ps_linear_hlsl, convert_UV_fp16_ps);
              }
            }
            break;

          case DXGI_FORMAT_R16_UINT:
            // Planar 16-bit YUV 4:4:4, 10 most significant bits store the value
            create_vertex_shader_helper(convert_yuv444_planar_vs_hlsl, convert_Y_or_YUV_vs);
            create_pixel_shader_helper(convert_yuv444_planar_ps_hlsl, convert_Y_or_YUV_ps);
            if (display->is_hdr()) {
              create_pixel_shader_helper(convert_yuv444_planar_ps_perceptual_quantizer_hlsl, convert_Y_or_YUV_fp16_ps);
            } else {
              create_pixel_shader_helper(convert_yuv444_planar_ps_linear_hlsl, convert_Y_or_YUV_fp16_ps);
            }
            break;

          case DXGI_FORMAT_AYUV:
            // Packed 8-bit YUV 4:4:4
            create_vertex_shader_helper(convert_yuv444_packed_vs_hlsl, convert_Y_or_YUV_vs);
            create_pixel_shader_helper(convert_yuv444_packed_ayuv_ps_hlsl, convert_Y_or_YUV_ps);
            create_pixel_shader_helper(convert_yuv444_packed_ayuv_ps_linear_hlsl, convert_Y_or_YUV_fp16_ps);
            break;

          case DXGI_FORMAT_Y410:
            // Packed 10-bit YUV 4:4:4
            create_vertex_shader_helper(convert_yuv444_packed_vs_hlsl, convert_Y_or_YUV_vs);
            create_pixel_shader_helper(convert_yuv444_packed_y410_ps_hlsl, convert_Y_or_YUV_ps);
            if (display->is_hdr()) {
              create_pixel_shader_helper(convert_yuv444_packed_y410_ps_perceptual_quantizer_hlsl, convert_Y_or_YUV_fp16_ps);
            } else {
              create_pixel_shader_helper(convert_yuv444_packed_y410_ps_linear_hlsl, convert_Y_or_YUV_fp16_ps);
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
        sbs_content_scale_x = content_scale_x;
        sbs_content_scale_y = content_scale_y;
      }
      update_sbs_constant_buffer(content_scale_x, content_scale_y);
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

        D3D11_TEXTURE2D_DESC tex_desc = {};
        tex_desc.Width = (UINT) std::lround(out_width_f);
        tex_desc.Height = (UINT) std::lround(out_height_f);
        tex_desc.MipLevels = 1;
        tex_desc.ArraySize = 1;
        tex_desc.Format = sbs_intermediate_linear ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_B8G8R8A8_UNORM;
        tex_desc.SampleDesc.Count = 1;
        tex_desc.Usage = D3D11_USAGE_DEFAULT;
        tex_desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;

        status = device->CreateTexture2D(&tex_desc, nullptr, &sbs_intermediate_texture);
        if (FAILED(status)) {
          BOOST_LOG(error) << "Failed to create SBS texture";
          return -1;
        }
        if (FAILED(device->CreateRenderTargetView(sbs_intermediate_texture.get(), nullptr, &sbs_intermediate_rtv)) || FAILED(device->CreateShaderResourceView(sbs_intermediate_texture.get(), nullptr, &sbs_intermediate_srv))) {
          BOOST_LOG(error) << "Failed to create SBS texture views";
          return -1;
        }

        if (sbs_config.adaptive_pop) {
          BOOST_LOG(info) << "Host SBS warp: Apollo occlusion-aware probe, scene-latched pop "
                          << sbs_config.pop_strength << "-"
                          << std::max(sbs_config.adaptive_pop_max, sbs_config.pop_strength) << ".";
        } else {
          BOOST_LOG(info) << "Host SBS warp: Apollo occlusion-aware probe, fixed pop "
                          << sbs_config.pop_strength << ".";
        }

        sbs_viewport = {0.0f, 0.0f, (float) tex_desc.Width, (float) tex_desc.Height, 0.0f, 1.0f};
        refresh_depth_estimator_fingerprint(
          (std::uint32_t) std::lround(sbs_viewport.Width * 0.5f),
          (std::uint32_t) std::lround(sbs_viewport.Height),
          sbs_content_scale_x,
          sbs_content_scale_y
        );
      }

      if (rgb_only) {
        // Local AR presents RGB directly. It needs the common capture/SBS resources above, but
        // no encoder output texture, chroma shaders, planar RTVs, or YUV conversion constants.
        rtvs_cleared = true;
        return 0;
      }

      out_Y_or_YUV_viewports[0] = {offsetX, offsetY, out_width_f, out_height_f, 0.0f, 1.0f};  // Y plane
      out_Y_or_YUV_viewports[1] = out_Y_or_YUV_viewports[0];  // U plane
      out_Y_or_YUV_viewports[1].TopLeftY += out_height;
      out_Y_or_YUV_viewports[2] = out_Y_or_YUV_viewports[1];  // V plane
      out_Y_or_YUV_viewports[2].TopLeftY += out_height;

      out_Y_or_YUV_viewports_for_clear[0] = {0, 0, (float) out_width, (float) out_height, 0.0f, 1.0f};  // Y plane
      out_Y_or_YUV_viewports_for_clear[1] = out_Y_or_YUV_viewports_for_clear[0];  // U plane
      out_Y_or_YUV_viewports_for_clear[1].TopLeftY += out_height;
      out_Y_or_YUV_viewports_for_clear[2] = out_Y_or_YUV_viewports_for_clear[1];  // V plane
      out_Y_or_YUV_viewports_for_clear[2].TopLeftY += out_height;

      out_UV_viewport = {offsetX / 2, offsetY / 2, out_width_f / 2, out_height_f / 2, 0.0f, 1.0f};
      out_UV_viewport_for_clear = {0, 0, (float) out_width / 2, (float) out_height / 2, 0.0f, 1.0f};

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
      bool rtv_simple_clear = false;

      switch (format) {
        case DXGI_FORMAT_NV12:
          rtv_Y_or_YUV_format = DXGI_FORMAT_R8_UNORM;
          rtv_UV_format = DXGI_FORMAT_R8G8_UNORM;
          rtv_simple_clear = true;
          break;

        case DXGI_FORMAT_P010:
          rtv_Y_or_YUV_format = DXGI_FORMAT_R16_UNORM;
          rtv_UV_format = DXGI_FORMAT_R16G16_UNORM;
          rtv_simple_clear = true;
          break;

        case DXGI_FORMAT_AYUV:
          rtv_Y_or_YUV_format = DXGI_FORMAT_R8G8B8A8_UINT;
          break;

        case DXGI_FORMAT_R16_UINT:
          rtv_Y_or_YUV_format = DXGI_FORMAT_R16_UINT;
          break;

        case DXGI_FORMAT_Y410:
          rtv_Y_or_YUV_format = DXGI_FORMAT_R10G10B10A2_UINT;
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

      if (rtv_simple_clear) {
        // Clear the RTVs to ensure the aspect ratio padding is black
        const float y_black[] = {0.0f, 0.0f, 0.0f, 0.0f};
        device_ctx->ClearRenderTargetView(out_Y_or_YUV_rtv.get(), y_black);
        if (out_UV_rtv) {
          const float uv_black[] = {0.5f, 0.5f, 0.5f, 0.5f};
          device_ctx->ClearRenderTargetView(out_UV_rtv.get(), uv_black);
        }
        rtvs_cleared = true;
      } else {
        // Can't use ClearRenderTargetView(), will clear on first convert()
        rtvs_cleared = false;
      }

      return 0;
    }

    int init_rgb_output(
      int width,
      int height,
      int sbs_mode_param,
      const config::video_t::sbs_t &profile
    ) {
      return init_output(
        nullptr,
        width,
        height,
        sbs_mode_param,
        profile,
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

        case pix_fmt_e::ayuv:
          format = DXGI_FORMAT_AYUV;
          break;

        case pix_fmt_e::yuv444p16:
          format = DXGI_FORMAT_R16_UINT;
          break;

        case pix_fmt_e::y410:
          format = DXGI_FORMAT_Y410;
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

      auto default_color_vectors = ::video::color_vectors_from_colorspace(::video::colorspace_e::rec601, false);
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
      update_sbs_constant_buffer(1.0f, 1.0f);

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

    shader_res_t create_black_texture_for_rtv_clear() {
      constexpr auto width = 32;
      constexpr auto height = 32;

      D3D11_TEXTURE2D_DESC texture_desc = {};
      texture_desc.Width = width;
      texture_desc.Height = height;
      texture_desc.MipLevels = 1;
      texture_desc.ArraySize = 1;
      texture_desc.SampleDesc.Count = 1;
      texture_desc.Usage = D3D11_USAGE_IMMUTABLE;
      texture_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
      texture_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

      std::vector<uint8_t> mem(4 * width * height, 0);
      D3D11_SUBRESOURCE_DATA texture_data = {mem.data(), 4 * width, 0};

      texture2d_t texture;
      auto status = device->CreateTexture2D(&texture_desc, &texture_data, &texture);
      if (FAILED(status)) {
        BOOST_LOG(error) << "Failed to create black texture: " << util::log_hex(status);
        return {};
      }

      shader_res_t resource_view;
      status = device->CreateShaderResourceView(texture.get(), nullptr, &resource_view);
      if (FAILED(status)) {
        BOOST_LOG(error) << "Failed to create black texture resource view: " << util::log_hex(status);
        return {};
      }

      return resource_view;
    }

    ::video::color_t *color_p;

    buf_t subsample_offset;
    buf_t color_matrix;

    blend_t blend_disable;
    sampler_state_t sampler_linear;

    render_target_t out_Y_or_YUV_rtv;
    render_target_t out_UV_rtv;
    bool rtvs_cleared = false;

    // d3d_img_t::id -> encoder_img_ctx_t
    // These store the encoder textures for each img_t that passes through
    // convert(). We can't store them in the img_t itself because it is shared
    // amongst multiple hwdevice_t objects (and therefore multiple ID3D11Devices).
    std::map<uint32_t, encoder_img_ctx_t> img_ctx_map;

    std::shared_ptr<display_base_t> display;

    vs_t convert_Y_or_YUV_vs;
    ps_t convert_Y_or_YUV_ps;
    ps_t convert_Y_or_YUV_fp16_ps;

    vs_t convert_UV_vs;
    ps_t convert_UV_ps;
    ps_t convert_UV_fp16_ps;

    std::array<D3D11_VIEWPORT, 3> out_Y_or_YUV_viewports, out_Y_or_YUV_viewports_for_clear;
    D3D11_VIEWPORT out_UV_viewport, out_UV_viewport_for_clear;

    DXGI_FORMAT format;

    device_t device;
    device_ctx_t device_ctx;

    texture2d_t output_texture;
    ID3D11Texture2D *rgb_present_texture = nullptr;  ///< Non-owning swapchain texture during convert_rgb().
    ID3D11RenderTargetView *rgb_present_target = nullptr;  ///< Non-owning swapchain RTV during convert_rgb().
    bool rgb_copy_path_logged = false;
    bool rgb_copy_fallback_logged = false;
    ps_t rgb_present_ps;
    D3D11_VIEWPORT rgb_present_viewport {};

    std::unique_ptr<models::video_depth_estimator> depth_estimator;
    // The per-device D3D estimator is built on a background thread and borrows the startup-warmed
    // TensorRT context. The future is declared after device/device_ctx so its destructor (which
    // joins the build) runs while those are still alive.
    std::future<std::unique_ptr<models::video_depth_estimator>> depth_estimator_build;
    bool depth_estimator_building = false;
    bool depth_estimator_failed = false;  ///< Build threw; stream flat, don't retry on this device.
    std::string depth_estimator_fingerprint;
    std::string depth_estimator_build_fingerprint;
    unsigned engine_poll_counter = 0;  ///< Rate-limits the startup model-preparation wait warning.
    int sbs_mode = ::video::SBS_OFF;  ///< Host SBS mode for this encode device (set in init_output).
    config::video_t::sbs_t sbs_config {};  ///< Immutable profile snapshot for this device.
    safe::mail_raw_t::event_t<int> sbs_depth_status_event;
    vs_t sbs_reprojection_vs;
    ps_t sbs_reprojection_ps;
    cs_t sbs_depth_prefilter_cs;
    buf_t sbs_reprojection_cbuffer;
    texture2d_t sbs_intermediate_texture;
    render_target_t sbs_intermediate_rtv;
    shader_res_t sbs_intermediate_srv;
    texture2d_t sbs_warp_depth_texture;
    unordered_access_t sbs_warp_depth_uav;
    shader_res_t sbs_warp_depth_srv;
    bool sbs_intermediate_linear = false;
    D3D11_VIEWPORT sbs_viewport;
    float sbs_content_scale_x = 1.0f;
    float sbs_content_scale_y = 1.0f;
    std::array<matched_frame_slot_t, 2> matched_frame_slots;
    std::uint64_t sbs_frame_sequence = 0;
    bool matched_output_valid = false;
    std::chrono::steady_clock::time_point matched_stats_started =
      std::chrono::steady_clock::now();
    unsigned matched_stats_calls = 0;
    unsigned matched_stats_completions = 0;
    unsigned matched_stats_repeats = 0;
    double matched_stats_age_sum_ms = 0.0;
    double matched_stats_age_max_ms = 0.0;
    static constexpr std::size_t sbs_gpu_timer_ring_size = 16;
    std::array<sbs_gpu_timer_slot_t, sbs_gpu_timer_ring_size> sbs_gpu_timers;
    std::size_t sbs_gpu_timer_next = 0;
    bool sbs_gpu_timing_ready = false;

    platf::sbs_debug::dumper sbs_dumper;  ///< Debug: dumps SBS frames on the client button (see sbs_debug_dump.h).
  };

  namespace {
    constexpr wchar_t local_presenter_window_class[] = L"ApolloLocalArPresenter";

    LRESULT CALLBACK local_presenter_window_proc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
      switch (message) {
        case WM_CLOSE:
          DestroyWindow(hwnd);
          return 0;
        case WM_DESTROY:
          return 0;
        case WM_NCHITTEST:
          return HTTRANSPARENT;
        case WM_MOUSEACTIVATE:
          return MA_NOACTIVATE;
        default:
          return DefWindowProcW(hwnd, message, wparam, lparam);
      }
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

    auto display = platf::display(platf::mem_type_e::dxgi, config.source_display_name, capture_config);
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

    d3d_base_encode_device converter;
    if (converter.init(display, dxgi_display->adapter.get(), platf::pix_fmt_e::nv12)) {
      BOOST_LOG(error) << "Local AR presenter could not initialize the production SBS converter."sv;
      return local_presenter_result_e::error;
    }

    if (converter.init_rgb_output(output_width, output_height, config.sbs_mode, config.sbs_config)) {
      BOOST_LOG(error) << "Local AR presenter could not initialize RGB presentation resources."sv;
      return local_presenter_result_e::error;
    }

    if (!register_local_presenter_window_class()) {
      BOOST_LOG(error) << "Local AR presenter could not register its window class: "sv
                       << GetLastError();
      return local_presenter_result_e::error;
    }

    HWND window = CreateWindowExW(
      WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_TRANSPARENT,
      local_presenter_window_class,
      L"Apollo Local SBS AI",
      WS_POPUP,
      target_rect.left,
      target_rect.top,
      output_width,
      output_height,
      nullptr,
      nullptr,
      GetModuleHandleW(nullptr),
      nullptr
    );
    if (!window) {
      BOOST_LOG(error) << "Local AR presenter could not create its output window: "sv
                       << GetLastError();
      return local_presenter_result_e::error;
    }
    auto destroy_window = util::fail_guard([&]() {
      if (IsWindow(window)) {
        DestroyWindow(window);
      }
      SetThreadExecutionState(ES_CONTINUOUS);
    });

    SetWindowPos(
      window,
      HWND_TOPMOST,
      target_rect.left,
      target_rect.top,
      output_width,
      output_height,
      SWP_NOACTIVATE | SWP_SHOWWINDOW
    );
    ShowWindow(window, SW_SHOWNOACTIVATE);
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

    Microsoft::WRL::ComPtr<IDXGIOutput> target_output;
    DXGI_OUTPUT_DESC target_output_desc {};
    const auto target_display_name_wide = platf::from_utf8(target_display_name);
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
    SetWindowPos(
      window,
      HWND_TOPMOST,
      target_rect.left,
      target_rect.top,
      output_width,
      output_height,
      SWP_NOACTIVATE | SWP_SHOWWINDOW
    );

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
    bool capture_cursor = true;
    bool window_closed = false;
    auto redirect_pointer_from_target = [&]() {
      const auto live_rect = read_target_rect();
      POINT cursor {};
      if (!GetCursorPos(&cursor) || !PtInRect(&live_rect, cursor)) {
        return;
      }

      DXGI_OUTPUT_DESC source_desc {};
      if (FAILED(dxgi_display->output->GetDesc(&source_desc))) {
        return;
      }
      const auto source_rect = source_desc.DesktopCoordinates;
      const LONG source_width = source_rect.right - source_rect.left;
      const LONG source_height = source_rect.bottom - source_rect.top;
      const LONG sink_width = live_rect.right - live_rect.left;
      const LONG sink_height = live_rect.bottom - live_rect.top;
      if (source_width <= 0 || source_height <= 0 || sink_width <= 0 || sink_height <= 0) {
        return;
      }

      // The physical sink must remain active for DP scanout, so Windows requires it to touch the
      // desktop. If the pointer crosses that one-pixel join, redirect it to the corresponding
      // location on the private virtual source before it can interact with the sink desktop.
      const auto relative_x = std::clamp(cursor.x - live_rect.left, 0L, sink_width - 1);
      const auto relative_y = std::clamp(cursor.y - live_rect.top, 0L, sink_height - 1);
      const LONG mapped_x = source_rect.left + std::min(
                                                 source_width - 1,
                                                 (LONG) ((std::int64_t) relative_x * source_width / sink_width)
                                               );
      const LONG mapped_y = source_rect.top + std::min(
                                                source_height - 1,
                                                (LONG) ((std::int64_t) relative_y * source_height / sink_height)
                                              );
      SetCursorPos(mapped_x, mapped_y);
    };
    auto pump_messages = [&]() {
      redirect_pointer_from_target();
      MSG message {};
      while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
      }
      window_closed = !IsWindow(window);
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
          return true;
        }
      }
      image.reset();
      return false;
    };

    bool presentation_reinit_requested = false;
    std::uint64_t captured_frames = 0;
    std::uint64_t presented_frames = 0;
    std::uint64_t busy_present_drops = 0;
    auto present_stats_started = std::chrono::steady_clock::now();
    auto log_present_stats = [&]() {
      const auto now = std::chrono::steady_clock::now();
      const auto elapsed = now - present_stats_started;
      if (elapsed < std::chrono::seconds(5)) {
        return;
      }
      const double elapsed_seconds = std::chrono::duration<double>(elapsed).count();
      BOOST_LOG(debug) << "Local AR presenter stats: captured="sv << captured_frames
                       << " presented="sv << presented_frames
                       << " busy_drops="sv << busy_present_drops
                       << " output_fps="sv << (presented_frames / elapsed_seconds);
      captured_frames = 0;
      presented_frames = 0;
      busy_present_drops = 0;
      present_stats_started = now;
    };
    auto push_image = [&](std::shared_ptr<platf::img_t> &&image, bool frame_captured) {
      pump_messages();
      if (stop_token.stop_requested() || window_closed) {
        return false;
      }
      if (!frame_captured) {
        return true;
      }
      ++captured_frames;

      const auto &d3d_image = static_cast<const img_d3d_t &>(*image);
      const bool frame_is_hdr = d3d_image.format == DXGI_FORMAT_R16G16B16A16_FLOAT;
      if (frame_is_hdr != config.hdr) {
        BOOST_LOG(error) << "Local AR capture frame does not match the negotiated "sv
                         << (config.hdr ? "HDR"sv : "SDR"sv) << " mode: "sv
                         << dxgi_display->dxgi_format_to_string(d3d_image.format);
        return false;
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
        SetWindowPos(
          window,
          HWND_TOPMOST,
          latest_target_rect.left,
          latest_target_rect.top,
          output_width,
          output_height,
          SWP_NOACTIVATE | SWP_SHOWWINDOW
        );
        target_rect = latest_target_rect;
      }

      // Never let a slower physical output back-pressure capture, synchronous depth inference,
      // or the desktop being interacted with. If DWM has not retired the previous flip yet, keep
      // the newest source frame in capture and skip this presentation opportunity.
      const auto frame_latency_status = WaitForSingleObject(frame_latency_waitable, 0);
      if (frame_latency_status == WAIT_FAILED) {
        BOOST_LOG(error) << "Local AR presenter frame-latency wait failed: "sv << GetLastError();
        return false;
      }
      if (frame_latency_status != WAIT_OBJECT_0) {
        ++busy_present_drops;
        log_present_stats();
        return true;
      }

      if (converter.convert_rgb(*image, backbuffer.Get(), backbuffer_rtv.Get())) {
        BOOST_LOG(error) << "Local AR presenter failed to convert a captured frame."sv;
        return false;
      }
      DXGI_PRESENT_PARAMETERS present_parameters {};
      const bool perf = sbs_perf::enabled();
      const auto present_started = perf ? std::chrono::steady_clock::now() :
                                          std::chrono::steady_clock::time_point {};
      status = swapchain->Present1(
        0,
        DXGI_PRESENT_RESTRICT_TO_OUTPUT | DXGI_PRESENT_DO_NOT_WAIT,
        &present_parameters
      );
      if (perf) {
        sbs_perf::add_sample_ms(
          "local_present_call_cpu",
          std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - present_started
          )
            .count()
        );
      }
      if (status == DXGI_ERROR_WAS_STILL_DRAWING) {
        ++busy_present_drops;
        log_present_stats();
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
      ++presented_frames;
      if (config.presented_frames) {
        config.presented_frames->fetch_add(1, std::memory_order_relaxed);
      }
      log_present_stats();
      return true;
    };

    const auto capture_status = display->capture(push_image, pull_image, &capture_cursor);
    if (presentation_reinit_requested && !stop_token.stop_requested()) {
      return local_presenter_result_e::reinit;
    }
    if (capture_status == capture_e::reinit && !stop_token.stop_requested()) {
      BOOST_LOG(info) << "Local AR capture topology changed; reinitializing the presenter."sv;
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

  class d3d_avcodec_encode_device_t: public avcodec_encode_device_t {
  public:
    int init(std::shared_ptr<platf::display_t> display, adapter_t::pointer adapter_p, pix_fmt_e pix_fmt) {
      int result = base.init(display, adapter_p, pix_fmt);
      data = base.device.get();
      return result;
    }

    int convert(platf::img_t &img_base) override {
      return base.convert(img_base);
    }

    void apply_colorspace() override {
      base.apply_colorspace(colorspace);
    }

    void init_hwframes(AVHWFramesContext *frames) override {
      // We may be called with a QSV or D3D11VA context
      if (frames->device_ctx->type == AV_HWDEVICE_TYPE_D3D11VA) {
        auto d3d11_frames = (AVD3D11VAFramesContext *) frames->hwctx;

        // The encoder requires textures with D3D11_BIND_RENDER_TARGET set
        d3d11_frames->BindFlags = D3D11_BIND_RENDER_TARGET;
        d3d11_frames->MiscFlags = 0;
      }

      // We require a single texture
      frames->initial_pool_size = 1;
    }

    int prepare_to_derive_context(int hw_device_type) override {
      // QuickSync requires our device to be multithread-protected
      if (hw_device_type == AV_HWDEVICE_TYPE_QSV) {
        multithread_t mt;

        auto status = base.device->QueryInterface(IID_ID3D11Multithread, (void **) &mt);
        if (FAILED(status)) {
          BOOST_LOG(warning) << "Failed to query ID3D11Multithread interface from device [0x"sv << util::hex(status).to_string_view() << ']';
          return -1;
        }

        mt->SetMultithreadProtected(TRUE);
      }

      return 0;
    }

    int set_frame(AVFrame *frame, AVBufferRef *hw_frames_ctx) override {
      this->hwframe.reset(frame);
      this->frame = frame;

      // Populate this frame with a hardware buffer if one isn't there already
      if (!frame->buf[0]) {
        auto err = av_hwframe_get_buffer(hw_frames_ctx, frame, 0);
        if (err) {
          char err_str[AV_ERROR_MAX_STRING_SIZE] {0};
          BOOST_LOG(error) << "Failed to get hwframe buffer: "sv << av_make_error_string(err_str, AV_ERROR_MAX_STRING_SIZE, err);
          return -1;
        }
      }

      // If this is a frame from a derived context, we'll need to map it to D3D11
      ID3D11Texture2D *frame_texture;
      if (frame->format != AV_PIX_FMT_D3D11) {
        frame_t d3d11_frame {av_frame_alloc()};

        d3d11_frame->format = AV_PIX_FMT_D3D11;

        auto err = av_hwframe_map(d3d11_frame.get(), frame, AV_HWFRAME_MAP_WRITE | AV_HWFRAME_MAP_OVERWRITE);
        if (err) {
          char err_str[AV_ERROR_MAX_STRING_SIZE] {0};
          BOOST_LOG(error) << "Failed to map D3D11 frame: "sv << av_make_error_string(err_str, AV_ERROR_MAX_STRING_SIZE, err);
          return -1;
        }

        // Get the texture from the mapped frame
        frame_texture = (ID3D11Texture2D *) d3d11_frame->data[0];
      } else {
        // Otherwise, we can just use the texture inside the original frame
        frame_texture = (ID3D11Texture2D *) frame->data[0];
      }

      return base.init_output(frame_texture, frame->width, frame->height);
    }

  private:
    d3d_base_encode_device base;
    frame_t hwframe;
  };

  class d3d_nvenc_encode_device_t: public nvenc_encode_device_t {
  public:
    bool init_device(std::shared_ptr<platf::display_t> display, adapter_t::pointer adapter_p, pix_fmt_e pix_fmt) {
      buffer_format = nvenc::nvenc_format_from_sunshine_format(pix_fmt);
      if (buffer_format == NV_ENC_BUFFER_FORMAT_UNDEFINED) {
        BOOST_LOG(error) << "Unexpected pixel format for NvENC ["sv << from_pix_fmt(pix_fmt) << ']';
        return false;
      }

      if (base.init(display, adapter_p, pix_fmt)) {
        return false;
      }

      if (pix_fmt == pix_fmt_e::yuv444p16) {
        nvenc_d3d = std::make_unique<nvenc::nvenc_d3d11_on_cuda>(base.device.get());
      } else {
        nvenc_d3d = std::make_unique<nvenc::nvenc_d3d11_native>(base.device.get());
      }
      nvenc = nvenc_d3d.get();

      return true;
    }

    bool init_encoder(const ::video::config_t &client_config, const ::video::sunshine_colorspace_t &colorspace) override {
      if (!nvenc_d3d) {
        return false;
      }

      auto nvenc_colorspace = nvenc::nvenc_colorspace_from_sunshine_colorspace(colorspace);
      if (!nvenc_d3d->create_encoder(config::video.nv, client_config, nvenc_colorspace, buffer_format)) {
        return false;
      }

      base.apply_colorspace(colorspace);
      return base.init_output(nvenc_d3d->get_input_texture(), client_config.width, client_config.height, client_config.sbs_mode, client_config.sbs_config, client_config.sbs_depth_status_event) == 0;
    }

    int convert(platf::img_t &img_base) override {
      return base.convert(img_base);
    }

  private:
    d3d_base_encode_device base;
    std::unique_ptr<nvenc::nvenc_d3d11> nvenc_d3d;
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

    std::optional<std::chrono::steady_clock::time_point> frame_timestamp;
    if (auto qpc_displayed = std::max(frame_info.LastPresentTime.QuadPart, frame_info.LastMouseUpdateTime.QuadPart)) {
      // Translate QueryPerformanceCounter() value to steady_clock time point
      frame_timestamp = std::chrono::steady_clock::now() - qpc_time_difference(qpc_counter(), qpc_displayed);
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
    if (frame_update_flag) {
      // Get the texture object from this frame
      status = res->QueryInterface(IID_ID3D11Texture2D, (void **) &src);
      if (FAILED(status)) {
        BOOST_LOG(error) << "Couldn't query interface [0x"sv << util::hex(status).to_string_view() << ']';
        return capture_e::error;
      }

      D3D11_TEXTURE2D_DESC desc;
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
    }

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
    }

    return capture_e::ok;
  }

  capture_e display_ddup_vram_t::release_snapshot() {
    return dup.release_frame();
  }

  int display_ddup_vram_t::init(const ::video::config_t &config, const std::string &display_name) {
    if (display_base_t::init(config, display_name) || dup.init(this, config)) {
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

      // Use a 300 nit target for the mouse cursor. We should really get
      // the user's SDR white level in nits, but there is no API that
      // provides that information to Win32 apps.
      float white_multiplier_data[16 / sizeof(float)] {300.0f / 80.f};  // aligned to 16-byte
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
    D3D11_TEXTURE2D_DESC desc;
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
    }

    return capture_e::ok;
  }

  capture_e display_wgc_vram_t::release_snapshot() {
    return dup.release_frame();
  }

  int display_wgc_vram_t::init(const ::video::config_t &config, const std::string &display_name) {
    if (display_base_t::init(config, display_name) || dup.init(this, config)) {
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
      // use a linear->PQ shader for HDR and a linear->sRGB shader for SDR.
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
   * @param name The FFmpeg codec name (or similar for non-FFmpeg codecs).
   * @param config The codec configuration.
   * @return `true` if supported, `false` otherwise.
   */
  bool display_vram_t::is_codec_supported(std::string_view name, const ::video::config_t &config) {
    DXGI_ADAPTER_DESC adapter_desc;
    adapter->GetDesc(&adapter_desc);

    if (adapter_desc.VendorId == 0x1002) {  // AMD
      // If it's not an AMF encoder, it's not compatible with an AMD GPU
      if (!boost::algorithm::ends_with(name, "_amf")) {
        return false;
      }

      // Perform AMF version checks if we're using an AMD GPU. This check is placed in display_vram_t
      // to avoid hitting the display_ram_t path which uses software encoding and doesn't touch AMF.
      HMODULE amfrt = LoadLibraryW(AMF_DLL_NAME);
      if (amfrt) {
        auto unload_amfrt = util::fail_guard([amfrt]() {
          FreeLibrary(amfrt);
        });

        auto fnAMFQueryVersion = (AMFQueryVersion_Fn) GetProcAddress(amfrt, AMF_QUERY_VERSION_FUNCTION_NAME);
        if (fnAMFQueryVersion) {
          amf_uint64 version;
          auto result = fnAMFQueryVersion(&version);
          if (result == AMF_OK) {
            if (config.videoFormat == 2 && version < AMF_MAKE_FULL_VERSION(1, 4, 30, 0)) {
              // AMF 1.4.30 adds ultra low latency mode for AV1. Don't use AV1 on earlier versions.
              // This corresponds to driver version 23.5.2 (23.10.01.45) or newer.
              BOOST_LOG(warning) << "AV1 encoding is disabled on AMF version "sv
                                 << AMF_GET_MAJOR_VERSION(version) << '.'
                                 << AMF_GET_MINOR_VERSION(version) << '.'
                                 << AMF_GET_SUBMINOR_VERSION(version) << '.'
                                 << AMF_GET_BUILD_VERSION(version);
              BOOST_LOG(warning) << "If your AMD GPU supports AV1 encoding, update your graphics drivers!"sv;
              return false;
            } else if (config.dynamicRange && version < AMF_MAKE_FULL_VERSION(1, 4, 23, 0)) {
              // Older versions of the AMD AMF runtime can crash when fed P010 surfaces.
              // Fail if AMF version is below 1.4.23 where HEVC Main10 encoding was introduced.
              // AMF 1.4.23 corresponds to driver version 21.12.1 (21.40.11.03) or newer.
              BOOST_LOG(warning) << "HDR encoding is disabled on AMF version "sv
                                 << AMF_GET_MAJOR_VERSION(version) << '.'
                                 << AMF_GET_MINOR_VERSION(version) << '.'
                                 << AMF_GET_SUBMINOR_VERSION(version) << '.'
                                 << AMF_GET_BUILD_VERSION(version);
              BOOST_LOG(warning) << "If your AMD GPU supports HEVC Main10 encoding, update your graphics drivers!"sv;
              return false;
            }
          } else {
            BOOST_LOG(warning) << "AMFQueryVersion() failed: "sv << result;
          }
        } else {
          BOOST_LOG(warning) << "AMF DLL missing export: "sv << AMF_QUERY_VERSION_FUNCTION_NAME;
        }
      } else {
        BOOST_LOG(warning) << "Detected AMD GPU but AMF failed to load"sv;
      }
    } else if (adapter_desc.VendorId == 0x8086) {  // Intel
      // If it's not a QSV encoder, it's not compatible with an Intel GPU
      if (!boost::algorithm::ends_with(name, "_qsv")) {
        return false;
      }
      if (config.chromaSamplingType == 1) {
        if (config.videoFormat == 0 || config.videoFormat == 2) {
          // QSV doesn't support 4:4:4 in H.264 or AV1
          return false;
        }
        // TODO: Blacklist HEVC 4:4:4 based on adapter model
      }
    } else if (adapter_desc.VendorId == 0x10de) {  // Nvidia
      // If it's not an NVENC encoder, it's not compatible with an Nvidia GPU
      if (!boost::algorithm::ends_with(name, "_nvenc")) {
        return false;
      }
    } else {
      BOOST_LOG(warning) << "Unknown GPU vendor ID: " << util::hex(adapter_desc.VendorId).to_string_view();
    }

    return true;
  }

  std::unique_ptr<avcodec_encode_device_t> display_vram_t::make_avcodec_encode_device(pix_fmt_e pix_fmt) {
    auto device = std::make_unique<d3d_avcodec_encode_device_t>();
    if (device->init(shared_from_this(), adapter.get(), pix_fmt) != 0) {
      return nullptr;
    }
    return device;
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
    compile_pixel_shader_helper(convert_yuv444_packed_ayuv_ps);
    compile_pixel_shader_helper(convert_yuv444_packed_ayuv_ps_linear);
    compile_vertex_shader_helper(convert_yuv444_packed_vs);
    compile_pixel_shader_helper(convert_yuv444_planar_ps);
    compile_pixel_shader_helper(convert_yuv444_planar_ps_linear);
    compile_pixel_shader_helper(convert_yuv444_planar_ps_perceptual_quantizer);
    compile_pixel_shader_helper(convert_yuv444_packed_y410_ps);
    compile_pixel_shader_helper(convert_yuv444_packed_y410_ps_linear);
    compile_pixel_shader_helper(convert_yuv444_packed_y410_ps_perceptual_quantizer);
    compile_vertex_shader_helper(convert_yuv444_planar_vs);
    compile_pixel_shader_helper(cursor_ps);
    compile_pixel_shader_helper(cursor_ps_normalize_white);
    compile_compute_shader_helper(depth_warp_prefilter_cs);
    compile_pixel_shader_helper(sbs_reprojection_ps);
    compile_vertex_shader_helper(sbs_reprojection_vs);
    compile_vertex_shader_helper(cursor_vs);

    BOOST_LOG(info) << "Compiled shaders"sv;

#undef compile_vertex_shader_helper
#undef compile_pixel_shader_helper
#undef compile_compute_shader_helper

    return 0;
  }
}  // namespace platf::dxgi
