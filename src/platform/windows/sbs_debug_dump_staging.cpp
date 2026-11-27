/**
 * @file src/platform/windows/sbs_debug_dump_staging.cpp
 * @brief Dump 3D D3D11 staging copies and bounded nonblocking readback.
 */
#include "sbs_debug_dump_staging.h"

// standard includes
#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

// platform includes
#include <windows.h>
#include <wrl/client.h>

// local includes
#include "src/depth_coordinate_v2.h"
#include "src/host_sbs_contour_history_readback.h"
#include "debug_dump_common.h"

namespace platf::sbs_debug {

  namespace {

    using namespace models::depth_coordinate_v2;
    using detail::staged_buffer;

    std::uint32_t format_bytes_per_pixel(const DXGI_FORMAT format) {
      switch (format) {
        case DXGI_FORMAT_R32_FLOAT:
        case DXGI_FORMAT_R32_UINT:
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8X8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
          return 4;
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
          return 8;
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
          return 16;
        default:
          return 0;
      }
    }

    bool stage_buffer_resource(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      ID3D11Buffer *buffer,
      const std::size_t required_bytes,
      staged_buffer &staging
    ) {
      if (!device || !ctx || !buffer || required_bytes == 0) {
        return false;
      }

      D3D11_BUFFER_DESC source_desc {};
      buffer->GetDesc(&source_desc);
      if (required_bytes > source_desc.ByteWidth) {
        return false;
      }
      D3D11_BUFFER_DESC staging_desc = source_desc;
      staging_desc.Usage = D3D11_USAGE_STAGING;
      staging_desc.BindFlags = 0;
      staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      staging_desc.MiscFlags = 0;
      staging_desc.StructureByteStride = 0;
      if (FAILED(device->CreateBuffer(&staging_desc, nullptr, &staging.resource))) {
        return false;
      }
      staging.byte_count = required_bytes;
      ctx->CopyResource(staging.resource.Get(), buffer);
      return true;
    }

  }  // namespace

  namespace detail {

    std::size_t bounded_collection_chunk_bytes(
      const std::size_t remaining_bytes,
      const std::size_t alignment,
      const std::size_t available_budget,
      const bool poll_is_empty
    ) noexcept {
      if (remaining_bytes == 0 || alignment == 0 ||
          remaining_bytes % alignment != 0) {
        return 0;
      }
      if (available_budget >= alignment) {
        return std::min(
          remaining_bytes,
          available_budget - available_budget % alignment
        );
      }
      return poll_is_empty ? std::min(remaining_bytes, alignment) : 0;
    }

    // Called only on the publication worker after bounded raw-row collection has retired
    // every GPU operand. Failed conversion leaves the owned snapshot unmodified.
    bool normalize_contour_history_for_publication(captured_dump_job &job) {
      auto &history = job.contour_history;
      if (history.desc.Format != DXGI_FORMAT_R16G16B16A16_FLOAT) return true;
      if (history.desc.MipLevels != 1u || history.desc.ArraySize != 1u ||
          history.desc.SampleDesc.Count != 1u ||
          !contour_texture_capture_layout_is_authenticated(
            detail::subtitle_contour_texture_e::history,
            history.desc.Format, history.desc.Width, history.desc.Height,
            history.row_bytes, history.bytes.size(), job.completed
          )) return false;
      const auto logical_row_bytes = history.row_bytes * 2u;
      std::vector<std::uint8_t> logical(history.bytes.size() * 2u);
      for (std::size_t row = 0u; row < history.desc.Height; ++row) {
        const std::span<const std::uint8_t> physical_row(
          history.bytes.data() + row * history.row_bytes, history.row_bytes
        );
        const std::span<std::uint8_t> logical_row(
          logical.data() + row * logical_row_bytes, logical_row_bytes
        );
        if (!models::contour_history_readback::expand_rgba16f_row_to_rgba32f(
              physical_row, logical_row
            )) return false;
      }
      history.bytes = std::move(logical);
      history.row_bytes = logical_row_bytes;
      history.desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
      return true;
    }

    bool stage_buffer(ID3D11Device *device, ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv, const std::size_t required_bytes, staged_buffer &staging) {
      if (!srv) return false;
      Microsoft::WRL::ComPtr<ID3D11Resource> resource;
      Microsoft::WRL::ComPtr<ID3D11Buffer> buffer;
      srv->GetResource(&resource);
      return resource && SUCCEEDED(resource.As(&buffer)) &&
        stage_buffer_resource(device, ctx, buffer.Get(), required_bytes, staging);
    }

    bool stage_exact_constants(ID3D11Device *device, ID3D11DeviceContext *ctx,
      ID3D11Buffer *buffer, const UINT bytes, staged_buffer &staging) {
      if (!buffer) return false;
      D3D11_BUFFER_DESC desc {};
      buffer->GetDesc(&desc);
      return desc.ByteWidth == bytes && (desc.BindFlags & D3D11_BIND_CONSTANT_BUFFER) != 0u &&
        stage_buffer_resource(device, ctx, buffer, bytes, staging);
    }

    bool stage_exact_scalar_buffer(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv,
      const std::size_t word_count,
      staged_buffer &staging
    ) {
      if (!srv || word_count == 0u ||
          word_count > std::numeric_limits<UINT>::max() / sizeof(std::uint32_t)) return false;
      Microsoft::WRL::ComPtr<ID3D11Resource> resource;
      Microsoft::WRL::ComPtr<ID3D11Buffer> buffer;
      srv->GetResource(&resource);
      if (!resource || FAILED(resource.As(&buffer))) return false;
      D3D11_BUFFER_DESC desc {};
      D3D11_SHADER_RESOURCE_VIEW_DESC view {};
      buffer->GetDesc(&desc);
      srv->GetDesc(&view);
      if (desc.ByteWidth != word_count * sizeof(std::uint32_t) ||
          desc.StructureByteStride != sizeof(std::uint32_t) ||
          (desc.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED) == 0u ||
          view.ViewDimension != D3D11_SRV_DIMENSION_BUFFER ||
          view.Format != DXGI_FORMAT_UNKNOWN || view.Buffer.FirstElement != 0u ||
          view.Buffer.NumElements != word_count) return false;
      return stage_buffer(device, ctx, srv, desc.ByteWidth, staging);
    }

    bool stage_texture(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv,
      staged_texture &staging
    ) {
      if (!device || !ctx || !srv) {
        return false;
      }
      Microsoft::WRL::ComPtr<ID3D11Resource> resource;
      srv->GetResource(&resource);
      Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
      if (!resource || FAILED(resource.As(&texture))) {
        return false;
      }
      texture->GetDesc(&staging.desc);
      const std::uint32_t bytes_per_pixel = format_bytes_per_pixel(staging.desc.Format);
      if (
        bytes_per_pixel == 0 || staging.desc.Width == 0 || staging.desc.Height == 0 ||
        staging.desc.ArraySize != 1 || staging.desc.MipLevels != 1 ||
        staging.desc.SampleDesc.Count != 1
      ) {
        return false;
      }
      staging.row_bytes =
        static_cast<std::size_t>(staging.desc.Width) * bytes_per_pixel;
      if (
        static_cast<std::size_t>(staging.desc.Height) >
        SIZE_MAX / staging.row_bytes
      ) {
        return false;
      }

      const auto staging_desc = debug_dump::readback_staging_desc(staging.desc);
      if (FAILED(device->CreateTexture2D(&staging_desc, nullptr, &staging.resource))) {
        return false;
      }
      ctx->CopyResource(staging.resource.Get(), texture.Get());
      return true;
    }

    bool stage_contour_texture(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv,
      const detail::subtitle_contour_texture_e role,
      const frame &completed,
      staged_texture &staging
    ) {
      if (!srv) return false;
      Microsoft::WRL::ComPtr<ID3D11Resource> resource;
      Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
      srv->GetResource(&resource);
      if (!resource || FAILED(resource.As(&texture))) return false;
      D3D11_TEXTURE2D_DESC desc {};
      D3D11_SHADER_RESOURCE_VIEW_DESC view {};
      texture->GetDesc(&desc);
      srv->GetDesc(&view);
      const auto row_bytes = static_cast<std::size_t>(desc.Width) * format_bytes_per_pixel(desc.Format);
      if (view.ViewDimension != D3D11_SRV_DIMENSION_TEXTURE2D || view.Format != desc.Format ||
          view.Texture2D.MostDetailedMip != 0u || view.Texture2D.MipLevels != 1u ||
          !contour_texture_capture_layout_is_authenticated(
            role, desc.Format, desc.Width, desc.Height, row_bytes,
            row_bytes * desc.Height, completed)) return false;
      return stage_texture(device, ctx, srv, staging);
    }

    collect_status_e collect_buffer(
      ID3D11DeviceContext *ctx,
      staged_buffer &staging,
      std::vector<std::uint8_t> &bytes,
      collection_budget &budget
    ) {
      if (!ctx || !staging.resource || staging.byte_count == 0) {
        return collect_status_e::failed;
      }
      if (bytes.size() > staging.byte_count) {
        return collect_status_e::failed;
      }
      if (bytes.size() == staging.byte_count) {
        staging.resource.Reset();
        return collect_status_e::ready;
      }
      bytes.reserve(staging.byte_count);
      const std::size_t copy_bytes = budget.next_chunk(
        staging.byte_count - bytes.size(), 1u
      );
      if (copy_bytes == 0) {
        return collect_status_e::partial;
      }
      D3D11_MAPPED_SUBRESOURCE mapped {};
      const HRESULT status = ctx->Map(
        staging.resource.Get(),
        0,
        D3D11_MAP_READ,
        D3D11_MAP_FLAG_DO_NOT_WAIT,
        &mapped
      );
      if (status == DXGI_ERROR_WAS_STILL_DRAWING) {
        return collect_status_e::not_ready;
      }
      if (FAILED(status) || !mapped.pData) {
        return collect_status_e::failed;
      }
      try {
        const auto *source = static_cast<const std::uint8_t *>(mapped.pData) + bytes.size();
        bytes.insert(bytes.end(), source, source + copy_bytes);
      } catch (...) {
        ctx->Unmap(staging.resource.Get(), 0);
        throw;
      }
      ctx->Unmap(staging.resource.Get(), 0);
      budget.record_copy(copy_bytes);
      if (bytes.size() == staging.byte_count) {
        staging.resource.Reset();
        return collect_status_e::ready;
      }
      return collect_status_e::partial;
    }

    collect_status_e collect_float_buffer(
      ID3D11DeviceContext *ctx,
      staged_buffer &staging,
      const std::size_t value_count,
      std::vector<float> &values,
      collection_budget &budget
    ) {
      if (value_count > SIZE_MAX / sizeof(float) ||
          staging.byte_count != value_count * sizeof(float)) {
        return collect_status_e::failed;
      }
      if (!ctx || !staging.resource || values.size() > value_count) {
        return collect_status_e::failed;
      }
      if (values.size() == value_count) {
        staging.resource.Reset();
        return collect_status_e::ready;
      }
      values.reserve(value_count);
      const std::size_t copy_bytes = budget.next_chunk(
        (value_count - values.size()) * sizeof(float), sizeof(float)
      );
      if (copy_bytes == 0) {
        return collect_status_e::partial;
      }
      D3D11_MAPPED_SUBRESOURCE mapped {};
      const HRESULT status = ctx->Map(
        staging.resource.Get(),
        0,
        D3D11_MAP_READ,
        D3D11_MAP_FLAG_DO_NOT_WAIT,
        &mapped
      );
      if (status == DXGI_ERROR_WAS_STILL_DRAWING) {
        return collect_status_e::not_ready;
      }
      if (FAILED(status) || !mapped.pData) {
        return collect_status_e::failed;
      }
      try {
        const auto *source = static_cast<const float *>(mapped.pData) + values.size();
        values.insert(values.end(), source, source + copy_bytes / sizeof(float));
      } catch (...) {
        ctx->Unmap(staging.resource.Get(), 0);
        throw;
      }
      ctx->Unmap(staging.resource.Get(), 0);
      budget.record_copy(copy_bytes);
      if (values.size() == value_count) {
        staging.resource.Reset();
        return collect_status_e::ready;
      }
      return collect_status_e::partial;
    }

    collect_status_e collect_texture(
      ID3D11DeviceContext *ctx,
      staged_texture &staging,
      texture_snapshot &snapshot,
      collection_budget &budget
    ) {
      if (!ctx || !staging.resource || staging.row_bytes == 0) {
        return collect_status_e::failed;
      }
      const std::size_t total_bytes =
        static_cast<std::size_t>(staging.desc.Height) * staging.row_bytes;
      if (snapshot.bytes.size() > total_bytes ||
          snapshot.bytes.size() % staging.row_bytes != 0) {
        return collect_status_e::failed;
      }
      snapshot.desc = staging.desc;
      snapshot.row_bytes = staging.row_bytes;
      if (snapshot.bytes.size() == total_bytes) {
        staging.resource.Reset();
        return collect_status_e::ready;
      }
      snapshot.bytes.reserve(total_bytes);
      const std::size_t copy_bytes = budget.next_chunk(
        total_bytes - snapshot.bytes.size(), staging.row_bytes
      );
      if (copy_bytes == 0) {
        return collect_status_e::partial;
      }
      D3D11_MAPPED_SUBRESOURCE mapped {};
      const HRESULT status = ctx->Map(
        staging.resource.Get(),
        0,
        D3D11_MAP_READ,
        D3D11_MAP_FLAG_DO_NOT_WAIT,
        &mapped
      );
      if (status == DXGI_ERROR_WAS_STILL_DRAWING) {
        return collect_status_e::not_ready;
      }
      if (FAILED(status) || !mapped.pData || mapped.RowPitch < staging.row_bytes) {
        if (SUCCEEDED(status)) {
          ctx->Unmap(staging.resource.Get(), 0);
        }
        return collect_status_e::failed;
      }

      const std::size_t first_row = snapshot.bytes.size() / snapshot.row_bytes;
      const std::size_t row_count = copy_bytes / snapshot.row_bytes;
      try {
        debug_dump::append_mapped_rows(
          snapshot.bytes, mapped, first_row, row_count, snapshot.row_bytes
        );
      } catch (...) {
        ctx->Unmap(staging.resource.Get(), 0);
        throw;
      }
      ctx->Unmap(staging.resource.Get(), 0);
      budget.record_copy(copy_bytes);
      if (snapshot.bytes.size() == total_bytes) {
        staging.resource.Reset();
        return collect_status_e::ready;
      }
      return collect_status_e::partial;
    }

  }  // namespace detail
}  // namespace platf::sbs_debug
