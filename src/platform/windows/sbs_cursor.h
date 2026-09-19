// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <d3d11.h>
#include <d3dcommon.h>
#include <dxgi1_2.h>
#include <memory>
#include <optional>
#include <vector>

namespace platf::sbs_cursor {
  // Published once per shape change. No capture-device GPU objects cross this boundary.
  struct shape_t {
    std::uint32_t width = 0, height = 0;
    std::vector<std::uint8_t> alpha_bgra;
    std::vector<std::uint8_t> xor_bgra;
  };

  // Decode the DXGI row pitch and optional stacked monochrome masks once. Both desktop
  // capture and SBS composition consume these same tightly packed BGRA images.
  std::optional<shape_t> decode_shape(const DXGI_OUTDUPL_POINTER_SHAPE_INFO &info, const std::uint8_t *bytes, std::size_t size);

  struct snapshot_t {
    std::shared_ptr<const shape_t> shape;
    // Already in capture-texture coordinates, including the pointer's top-left/hotspot offset.
    D3D11_VIEWPORT viewport {};
    std::uint32_t capture_width = 0, capture_height = 0;
    DXGI_MODE_ROTATION rotation = DXGI_MODE_ROTATION_IDENTITY;
    // Caller combines OS visibility and the client's cursor-display preference.
    bool visible = false;
  };

  struct placement_t {
    std::array<D3D11_VIEWPORT, 2> viewports;
    std::array<D3D11_RECT, 2> scissors;
  };

  // Game 3D accepts identity-oriented displays only. Reject unsupported rotations explicitly.
  std::optional<placement_t> place_cursor(const snapshot_t &cursor, std::uint32_t packed_width, std::uint32_t packed_height);

  struct result_t {
    ID3D11Texture2D *texture = nullptr;
    ID3D11ShaderResourceView *view = nullptr;
  };

  class compositor_t {
  public:
    compositor_t(ID3D11Device *device, ID3D11DeviceContext *context, ID3DBlob *vertex_shader, ID3DBlob *pixel_shader, ID3DBlob *hdr_pixel_shader);
    ~compositor_t();
    compositor_t(const compositor_t &) = delete;
    compositor_t &operator=(const compositor_t &) = delete;

    // Hidden/disabled cursors return the unchanged input without GPU allocation/copy.
    // Visible output is private scratch, valid until the next call or destruction. Source is
    // never modified. All device-context state is restored. Nullopt means invalid input/failure.
    // Linear output uses the existing cursor HDR shader: sRGB decode * SDR-white-nits / 80.
    std::optional<result_t> compose(ID3D11Texture2D *texture, ID3D11ShaderResourceView *view, const snapshot_t &cursor, bool linear, float white_multiplier);

  private:
    struct impl_t;
    std::unique_ptr<impl_t> impl_;
  };
}  // namespace platf::sbs_cursor
