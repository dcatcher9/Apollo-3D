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

  // Whether two snapshots composite identically. Hidden cursors are equal wherever they are.
  [[nodiscard]] inline bool same_presentation(const snapshot_t &a, const snapshot_t &b) noexcept {
    if (a.visible != b.visible) {
      return false;
    }
    return !a.visible ||
           (a.shape == b.shape && a.viewport.TopLeftX == b.viewport.TopLeftX && a.viewport.TopLeftY == b.viewport.TopLeftY &&
            a.viewport.Width == b.viewport.Width && a.viewport.Height == b.viewport.Height &&
            a.capture_width == b.capture_width && a.capture_height == b.capture_height && a.rotation == b.rotation);
  }

  struct placement_t {
    std::array<D3D11_VIEWPORT, 2> viewports;
    std::array<D3D11_RECT, 2> scissors;
  };

  // Game 3D accepts identity-oriented displays only. Positive source-UV parallax places the
  // cursor in front of the screen: left eye moves right, right eye moves left. The supplied
  // value belongs to the exact packed frame, independently of the current cursor position.
  std::optional<placement_t> place_cursor(const snapshot_t &cursor, std::uint32_t packed_width, std::uint32_t packed_height, float ui_parallax_uv = 0.0f);

  // Encoding of the frame the cursor is composited over.
  enum class source_transfer_e {
    srgb,  ///< Display-referred code values (any UNORM/FP16 format).
    scrgb,  ///< Linear Rec.709, 1.0 = 80 cd/m2.
    pq,  ///< Rec.2020 ST 2084 code values; the patch is decoded to scRGB first.
  };

  // A margin that makes compose() populate the whole frame, so `texture` is the complete
  // composited frame (the reference full-frame composition).
  inline constexpr std::uint32_t whole_frame_margin = UINT32_MAX;

  struct result_t {
    ID3D11Texture2D *texture = nullptr;
    ID3D11ShaderResourceView *view = nullptr;
    // Encoding of `texture` when `regions` is nonzero: the source's for sRGB/scRGB; scRGB for PQ.
    bool linear = false;
    // Eyes whose cursor changed texels. Zero means `texture` is the unchanged source.
    std::uint32_t regions = 0;
    // Texels the cursor changed in each drawn eye (right/bottom exclusive, frame coordinates).
    std::array<D3D11_RECT, 2> changed {};
    // Texels of `texture` that equal the composited frame: `changed` grown by the requested
    // margin and clamped to the frame. `texture` has the frame's size; outside these
    // rectangles it is stale. A reader re-runs its pass from `texture` only over
    // output_region(changed), whose footprint stays inside `valid`.
    std::array<D3D11_RECT, 2> valid {};
  };

  // Margin, in source texels, around a changed rectangle that holds the bilinear (or 4:2:0
  // [1,2,1]x[1,1] texel) footprint of every output pixel output_region() returns, for passes
  // whose viewport maps at least `minimum_scale` output pixels to one source texel.
  std::uint32_t patch_margin(float minimum_scale);

  // Render-target pixels of a pass that reads a `source_width` x `source_height` frame through
  // `viewport` whose footprint may include a texel of `changed`, clamped to the viewport.
  std::optional<D3D11_RECT> output_region(const D3D11_RECT &changed, std::uint32_t source_width, std::uint32_t source_height, const D3D11_VIEWPORT &viewport);

  class compositor_t {
  public:
    compositor_t(ID3D11Device *device, ID3D11DeviceContext *context, ID3DBlob *vertex_shader, ID3DBlob *pixel_shader, ID3DBlob *hdr_pixel_shader, ID3DBlob *pq_decode_pixel_shader = nullptr);
    ~compositor_t();
    compositor_t(const compositor_t &) = delete;
    compositor_t &operator=(const compositor_t &) = delete;

    // Hidden/disabled cursors return the unchanged input without GPU allocation/copy.
    // A visible cursor is drawn into a private frame-sized patch, valid until the next call or
    // destruction: only `valid` is copied from the source (or decoded, for PQ) before the
    // blend, so no full-frame copy is made unless `margin` is whole_frame_margin. Source is
    // never modified. All device-context state is restored. Nullopt means invalid
    // input/failure. Linear output uses the existing cursor HDR shader: sRGB decode *
    // SDR-white-nits / 80; a PQ source is blended in that same linear light.
    std::optional<result_t> compose(ID3D11Texture2D *texture, ID3D11ShaderResourceView *view, const snapshot_t &cursor, source_transfer_e transfer, float white_multiplier, float ui_parallax_uv, std::uint32_t margin);

    // Whole-frame composition of an sRGB or scRGB source: `texture` is the composited frame.
    std::optional<result_t> compose(ID3D11Texture2D *texture, ID3D11ShaderResourceView *view, const snapshot_t &cursor, bool linear, float white_multiplier, float ui_parallax_uv = 0.0f) {
      return compose(texture, view, cursor, linear ? source_transfer_e::scrgb : source_transfer_e::srgb, white_multiplier, ui_parallax_uv, whole_frame_margin);
    }

  private:
    struct impl_t;
    std::unique_ptr<impl_t> impl_;
  };
}  // namespace platf::sbs_cursor
