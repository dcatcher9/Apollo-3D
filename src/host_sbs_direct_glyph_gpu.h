/**
 * @file src/host_sbs_direct_glyph_gpu.h
 * @brief Single-slot GPU scoring transport for the current direct subtitle glyph plan.
 */
#pragma once

#include "cuda_driver_api.h"
#include "host_sbs_compute_state_guard.h"
#include "host_sbs_direct_glyph_tiles.h"
#include "host_sbs_shader_cache.h"

#include <NvInferRuntime.h>
#include <d3d11_4.h>
#include <wrl/client.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace models::host_sbs_direct_glyph_gpu {
  using Microsoft::WRL::ComPtr;
  namespace cache = host_sbs_shader_cache;

  inline constexpr UINT work_bytes = 144u;

  struct request_t {
    ID3D11ShaderResourceView *source_rgb = nullptr;
    ID3D11Buffer *work = nullptr;
    UINT width = 0;
    UINT band_height = 0;
    UINT crop_top = 0;
    std::uint64_t token = 0;
    ID3D11ShaderResourceView *detector_caps = nullptr;
    ID3D11ShaderResourceView *video_caption_ocr = nullptr;
    UINT line_center_twice_source = 0;
    std::uint64_t generation = 0, frame = 0;
    std::uint64_t subtitle_context_generation = 0;
    UINT origin_x = 0, origin_y = 0, color_domain = 0;
    bool reset_history = false;

    bool operator==(const request_t &) const = default;
  };

  struct result_t {
    ComPtr<ID3D11ShaderResourceView> alpha8;
  };

  enum class submit_status_e { accepted, busy, failed };
  enum class poll_status_e { idle, pending, ready, discarded, failed };

  namespace detail {
    inline void require(bool condition, const char *reason) {
      if (!condition) throw std::runtime_error(reason);
    }

    inline void check_cuda(CUresult status, const char *reason) {
      if (status != CUDA_SUCCESS) {
        throw std::runtime_error(std::string(reason) + " (CUDA " + std::to_string(status) + ")");
      }
    }

    // The caller may already be using another primary context on this thread.
    class current_cuda_context_t {
     public:
      current_cuda_context_t(cuda_driver_api &cuda, CUcontext primary): cuda_(cuda) {
        check_cuda(cuda_.cuCtxGetCurrent(&previous_), "Read current CUDA context");
        if (previous_ != primary) {
          check_cuda(cuda_.cuCtxSetCurrent(primary), "Select the direct glyph CUDA context");
          changed_ = true;
        }
      }

      ~current_cuda_context_t() {
        if (changed_) cuda_.cuCtxSetCurrent(previous_);
      }

      current_cuda_context_t(const current_cuda_context_t &) = delete;
      current_cuda_context_t &operator=(const current_cuda_context_t &) = delete;

     private:
      cuda_driver_api &cuda_;
      CUcontext previous_ = nullptr;
      bool changed_ = false;
    };

    // The compute slots the palette passes bind, restored around each queue.
    using palette_state_t = host_sbs_compute_state::guard_t<16u, 5u, 1u>;

    // Defined with the resources below in host_sbs_direct_glyph_gpu.cpp.
    class palette_transport_t;
  }  // namespace detail

  // Serialized by the caller's immediate context. It owns publication and gives each
  // observation a unique token. Work must stay reserved until ready or discarded.
  class resources_t {
   public:
    resources_t();
    resources_t(const resources_t &) = delete;
    resources_t &operator=(const resources_t &) = delete;

    ~resources_t();

    bool prepare(ID3D11Device *device, ID3D11DeviceContext *context, const std::filesystem::path &shader_root,
                 CUcontext primary, nvinfer1::IExecutionContext *execution, std::string &error);

    /** Register the native score buffers with CUDA ahead of the first submission, on the thread
     *  that draws. A failure is left to that submission, which retries and reports it. */
    void register_interop_early() noexcept;

    submit_status_e try_submit(const request_t &request, std::string &error);

    // Returns a score lease only for this exact request. GPU Work separately attests
    // EMPTY, valid score, or unavailable. A ready event alone is never matte authority.
    poll_status_e poll(const request_t *current, result_t &result, std::string &error);

    // Call after recording every same-context consumer of alpha8 and updated Work.
    void retire() noexcept;

    [[nodiscard]] ID3D11Buffer *native_lines_for_evaluation() const noexcept;
#ifdef SUNSHINE_TESTS
    [[nodiscard]] std::uint64_t native_fences_for_tests() const noexcept;

    [[nodiscard]] bool native_interop_registered_for_tests() const noexcept;
#endif

    bool release() noexcept;

    bool safe_to_recycle_context() const noexcept;

   private:
    std::unique_ptr<detail::palette_transport_t> palette_;
    bool failed_ = false;
  };
}  // namespace models::host_sbs_direct_glyph_gpu
