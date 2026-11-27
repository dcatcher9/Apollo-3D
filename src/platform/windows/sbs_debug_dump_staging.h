/**
 * @file src/platform/windows/sbs_debug_dump_staging.h
 * @brief Dump 3D internals: D3D11 staging copies and bounded nonblocking readback.
 *
 * Staging and collection run only on the render thread that owns the immediate context.
 */
#pragma once

// platform includes
#include <d3d11.h>
#include <wrl/client.h>

// standard includes
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

// local includes
#include "sbs_debug_dump.h"
#include "sbs_debug_dump_package.h"
#include "sbs_debug_dump_validate.h"

namespace platf::sbs_debug {

  namespace detail {

    struct staged_buffer {
      Microsoft::WRL::ComPtr<ID3D11Buffer> resource;
      std::size_t byte_count = 0;
    };

    struct staged_texture {
      Microsoft::WRL::ComPtr<ID3D11Texture2D> resource;
      D3D11_TEXTURE2D_DESC desc {};
      std::size_t row_bytes = 0;
    };

    enum class collect_status_e {
      ready,
      partial,
      not_ready,
      failed,
    };

    class collection_budget {
    public:
      explicit collection_budget(const std::chrono::steady_clock::time_point started) noexcept:
          deadline_(started + std::chrono::milliseconds(2)) {
      }

      std::size_t next_chunk(
        const std::size_t remaining_bytes,
        const std::size_t alignment
      ) const noexcept {
        if (remaining_maps_ == 0 ||
            std::chrono::steady_clock::now() >= deadline_) {
          return 0;
        }
        return detail::bounded_collection_chunk_bytes(
          remaining_bytes,
          alignment,
          remaining_bytes_,
          copied_bytes_ == 0
        );
      }

      void record_copy(const std::size_t bytes) noexcept {
        copied_bytes_ += bytes;
        remaining_bytes_ = bytes >= remaining_bytes_ ? 0 : remaining_bytes_ - bytes;
        if (remaining_maps_ != 0) {
          --remaining_maps_;
        }
      }

      bool exhausted() const noexcept {
        return remaining_bytes_ == 0 || remaining_maps_ == 0 ||
               std::chrono::steady_clock::now() >= deadline_;
      }

    private:
      std::chrono::steady_clock::time_point deadline_ {};
      std::size_t remaining_bytes_ = detail::cpu_collection_byte_budget;
      std::size_t copied_bytes_ = 0;
      unsigned remaining_maps_ = 8;
    };

    /** Queue a full copy of an SRV's buffer; `required_bytes` of it are collected. */
    bool stage_buffer(ID3D11Device *device, ID3D11DeviceContext *ctx, ID3D11ShaderResourceView *srv, std::size_t required_bytes, staged_buffer &staging);

    /** Queue a copy of a constant buffer that must be exactly `bytes` long. */
    bool stage_exact_constants(ID3D11Device *device, ID3D11DeviceContext *ctx, ID3D11Buffer *buffer, UINT bytes, staged_buffer &staging);

    /** Queue a copy of a whole structured uint buffer of exactly `word_count` words. */
    bool stage_exact_scalar_buffer(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv,
      std::size_t word_count,
      staged_buffer &staging
    );

    /** Queue a copy of a single-mip, single-sample texture in a supported format. */
    bool stage_texture(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv,
      staged_texture &staging
    );

    /** Queue a contour texture after authenticating its view and captured layout. */
    bool stage_contour_texture(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv,
      subtitle_contour_texture_e role,
      const frame &completed,
      staged_texture &staging
    );

    /** Map without waiting and append the next budgeted chunk of a staged buffer. */
    collect_status_e collect_buffer(
      ID3D11DeviceContext *ctx,
      staged_buffer &staging,
      std::vector<std::uint8_t> &bytes,
      collection_budget &budget
    );

    /** Map without waiting and append the next budgeted floats of a staged buffer. */
    collect_status_e collect_float_buffer(
      ID3D11DeviceContext *ctx,
      staged_buffer &staging,
      std::size_t value_count,
      std::vector<float> &values,
      collection_budget &budget
    );

    /** Map without waiting and append the next budgeted whole rows of a staged texture. */
    collect_status_e collect_texture(
      ID3D11DeviceContext *ctx,
      staged_texture &staging,
      texture_snapshot &snapshot,
      collection_budget &budget
    );

    /** Expand compact RGBA16F contour history to the RGBA32F package layout. */
    bool normalize_contour_history_for_publication(captured_dump_job &job);

  }  // namespace detail
}  // namespace platf::sbs_debug
