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

    #include "host_sbs_direct_glyph_palette.inc"
  }  // namespace detail

  // Serialized by the caller's immediate context. It owns publication and gives each
  // observation a unique token. Work must stay reserved until ready or discarded.
  class resources_t {
   public:
    resources_t() = default;
    resources_t(const resources_t &) = delete;
    resources_t &operator=(const resources_t &) = delete;

    ~resources_t() {
      // Retain all operands if CUDA may still be using them. The execution lease must
      // be quarantined by the caller whenever release() cannot prove safe teardown.
      if (!release()) {
        (void)palette_.release();
      }
    }

    bool prepare(ID3D11Device *device, ID3D11DeviceContext *context, const std::filesystem::path &shader_root,
                 CUcontext primary, nvinfer1::IExecutionContext *execution, std::string &error) {
      if (palette_ || failed_) {
        error = "Direct glyph score transport already owns an execution lease";
        return false;
      }
      try {
        palette_ = std::make_unique<detail::palette_transport_t>();
        palette_->prepare(device, context, shader_root, primary, execution);
        error.clear();
        return true;
      } catch (const std::exception &why) {
        failed_ = true;
        error = why.what();
        return false;
      }
    }

    /** Register the native score buffers with CUDA ahead of the first submission, on the thread
     *  that draws. A failure is left to that submission, which retries and reports it. */
    void register_interop_early() noexcept {
      if (!palette_ || failed_) return;
      try {
        palette_->register_interop();
      } catch (const std::exception &) {
      }
    }

    submit_status_e try_submit(const request_t &request, std::string &error) {
      if (!palette_ || failed_) {
        error = "Direct glyph score transport is unavailable";
        return submit_status_e::failed;
      }
      error.clear();
      if (palette_->busy()) return submit_status_e::busy;
      try {
        palette_->submit(request);
        return submit_status_e::accepted;
      } catch (const std::exception &why) {
        failed_ = true;
        error = why.what();
        return submit_status_e::failed;
      }
    }

    // Returns a score lease only for this exact request. GPU Work separately attests
    // EMPTY, valid score, or unavailable. A ready event alone is never matte authority.
    poll_status_e poll(const request_t *current, result_t &result, std::string &error) {
      result = {};
      error.clear();
      if (failed_) {
        error = "Direct glyph score transport failed";
        return poll_status_e::failed;
      }
      if (!palette_) return poll_status_e::idle;
      try {
        return palette_->poll(current, result);
      } catch (const std::exception &why) {
        failed_ = true;
        error = why.what();
        return poll_status_e::failed;
      }
    }

    // Call after recording every same-context consumer of alpha8 and updated Work.
    void retire() noexcept {
      if (palette_) palette_->retire();
    }

    [[nodiscard]] ID3D11Buffer *native_lines_for_evaluation() const noexcept {
      return palette_ ? palette_->committed_lines() : nullptr;
    }
#ifdef SUNSHINE_TESTS
    [[nodiscard]] std::uint64_t native_fences_for_tests() const noexcept {
      return palette_ ? palette_->submitted_fences_for_tests() : 0u;
    }

    [[nodiscard]] bool native_interop_registered_for_tests() const noexcept {
      return palette_ && palette_->interop_registered_for_tests();
    }
#endif

    bool release() noexcept {
      if (palette_ && !palette_->release()) return false;
      palette_.reset();
      return true;
    }

    bool safe_to_recycle_context() const noexcept {
      return !palette_;
    }

   private:
    std::unique_ptr<detail::palette_transport_t> palette_;
    bool failed_ = false;
  };
}  // namespace models::host_sbs_direct_glyph_gpu
