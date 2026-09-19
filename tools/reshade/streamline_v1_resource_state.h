// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "scene_depth_source.h"
#include <d3d12.h>
#include <cstdint>

// This private property is an exact Streamline 1.1.1 adapter, not a general
// D3D12 state tracker. That version's DLSS::cacheState uses the same property
// when Resource::state is zero. Newer/unknown versions must not query it.
// Sources: NVIDIA-RTX/Streamline, tag v1.1.1:
// source/platforms/sl.chi/{compute.h,d3d12.cpp}, source/plugins/sl.dlss/dlssEntry.cpp.
namespace sunshine_streamline::v1_resource_state {
  inline constexpr GUID property_id{0x694b3e1c, 0x0e33, 0x416f,
    {0xba, 0x83, 0xfe, 0x24, 0x8d, 0xa1, 0xe8, 0x5d}};

  // chi::ResourceState bits are NOT D3D12_RESOURCE_STATES. Limit translation
  // to the ordinary texture states supported by our native capture owner.
  enum chi : std::uint32_t {
    general = 1u << 0, texture_read = 1u << 5, storage_read = 1u << 6,
    storage_write = 1u << 7, color_write = 1u << 9,
    depth_write = 1u << 10, depth_read = 1u << 11,
    copy_source = 1u << 12, copy_destination = 1u << 13, present = 1u << 18
  };

  inline bool decode(std::uint32_t value, std::uint32_t &native) {
    native = UINT32_MAX;
    constexpr std::uint32_t allowed = general | texture_read | storage_read |
      storage_write | color_write | depth_write | depth_read | copy_source |
      copy_destination | present;
    // eUnknown=0 is different from ePresent, which maps to native COMMON=0.
    if (!value || (value & ~allowed) || ((value & storage_write) && !(value & storage_read))) return false;
    std::uint32_t result = D3D12_RESOURCE_STATE_COMMON;
    if (value & texture_read) result |= D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    if (value & storage_read) result |= D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    if ((value & storage_write) && (value & storage_read)) {
      // Match getNativeResourceState exactly: StorageRW takes precedence over
      // the shader-read bits, rather than interpreting chi bits as native bits.
      result |= D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      result &= ~(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    if (value & color_write) result |= D3D12_RESOURCE_STATE_RENDER_TARGET;
    if (value & depth_write) result |= D3D12_RESOURCE_STATE_DEPTH_WRITE;
    if (value & depth_read) result |= D3D12_RESOURCE_STATE_DEPTH_READ;
    if (value & copy_source) result |= D3D12_RESOURCE_STATE_COPY_SOURCE;
    if (value & copy_destination) result |= D3D12_RESOURCE_STATE_COPY_DEST;
    constexpr std::uint32_t writes = D3D12_RESOURCE_STATE_UNORDERED_ACCESS |
      D3D12_RESOURCE_STATE_RENDER_TARGET | D3D12_RESOURCE_STATE_DEPTH_WRITE | D3D12_RESOURCE_STATE_COPY_DEST;
    if ((result & writes) && (result & (result - 1))) return false;
    native = result;
    return true;
  }

  inline bool supported_resource(const D3D12_RESOURCE_DESC &desc) {
    // Pinned DLSS applies this provider state to ALL subresources, including
    // packed depth/stencil. Capture uses that same input-state contract but
    // copies/transitions only subresource 0 (depth), leaving stencil untouched.
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || !desc.Width || !desc.Height ||
        desc.MipLevels != 1 || desc.DepthOrArraySize != 1 || desc.SampleDesc.Count != 1) return false;
    switch (desc.Format) {
    case DXGI_FORMAT_R32_FLOAT: case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_D32_FLOAT:
    case DXGI_FORMAT_R16_UNORM: case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_D16_UNORM:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: case DXGI_FORMAT_R32G8X24_TYPELESS: case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT: case DXGI_FORMAT_R24G8_TYPELESS: case DXGI_FORMAT_R24_UNORM_X8_TYPELESS: return true;
    default: return false;
    }
  }

  inline bool decode_property(HRESULT result, UINT bytes, std::uint32_t value, std::uint32_t &native) {
    native = UINT32_MAX;
    return result == S_OK && bytes == sizeof(value) && decode(value, native);
  }

  struct read_details {
    const char *stage{"not_queried"};
    D3D12_RESOURCE_DESC description{};
    HRESULT result{E_PENDING};
    UINT bytes{};
    std::uint32_t encoded{};
  };

  inline bool read(IUnknown *object, std::uint32_t &native, read_details *details = nullptr) {
    native = UINT32_MAX;
    ID3D12Resource *resource{};
    // A retained backing resource does not make the original opaque pointer a
    // Resource interface: wrappers may return a different interface from QI.
    const HRESULT queried = object ? object->QueryInterface(__uuidof(ID3D12Resource), reinterpret_cast<void **>(&resource)) : E_POINTER;
    if (FAILED(queried) || !resource) {
      if (details) { details->stage = "resource_interface"; details->result = queried; }
      return false;
    }
    const auto description = resource->GetDesc();
    std::uint32_t property{};
    UINT bytes = sizeof(property);
    const HRESULT result = resource->GetPrivateData(property_id, &bytes, &property);
    const bool shape = supported_resource(description);
    const bool valid = shape && decode_property(result, bytes, property, native);
    if (details) {
      details->description = description; details->result = result;
      details->bytes = bytes; details->encoded = property;
      details->stage = !shape ? "resource_shape_or_format" : result != S_OK ? "property_read" :
        bytes != sizeof(property) ? "property_size" : !valid ? "property_encoding" : "ready";
    }
    resource->Release();
    return valid;
  }

  struct resolved {
    std::uint32_t native{UINT32_MAX};
    sunshine_scene_depth::state_proof proof{sunshine_scene_depth::state_proof::unavailable};
  };

  // The caller supplies a validated exact-1.1.1 ABI identity. Query runs only
  // after retaining the actual resource at evaluation entry; no COM access
  // belongs in the metadata-only normalization/diagnostic path.
  template<class Query>
  inline resolved resolve_with(std::uint32_t tagged, bool pinned_v1_1_1, Query &&query) {
    using proof = sunshine_scene_depth::state_proof;
    resolved out{tagged, tagged != UINT32_MAX ? proof::declared : proof::unavailable};
    if (!pinned_v1_1_1 || tagged != 0) return out;
    out.proof = proof::observed_nonzero;
    std::uint32_t native{};
    if (query(native)) out = {native, proof::declared};
    return out;
  }

  inline resolved resolve(std::uint32_t tagged, bool pinned_v1_1_1, IUnknown *resource, read_details *details = nullptr) {
    return resolve_with(tagged, pinned_v1_1_1, [resource, details](std::uint32_t &native) { return read(resource, native, details); });
  }
}
