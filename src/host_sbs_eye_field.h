/**
 * @file src/host_sbs_eye_field.h
 * @brief Eye-specific limiter field that accompanies the Base field.
 *
 * The producer attaches the eye field to the Base field's shader resource view as private data,
 * so every holder of the Base view (live, queued, shadow and harness routes) reaches the same
 * companion without a second publication path. A Base view without a companion renders exactly
 * as before.
 */
#pragma once

// standard includes
#include <cstdint>

// platform includes
#include <d3d11.h>
#include <wrl/client.h>

namespace models::host_sbs_eye_field {

  /** Private-data key of the companion view on the Base view. */
  inline constexpr GUID companion_private_data {
    0x6f4d2a51,
    0x93c7,
    0x4e0b,
    {0xa1, 0x5e, 0x2b, 0x7c, 0x44, 0x9d, 0x30, 0xe8}
  };

  /** Pixel-shader slot of `EyeParallax` in the Host-SBS V2 reprojection shaders. */
  inline constexpr UINT pixel_slot = 15u;

  /**
   * Largest slope, in parallax per source U, on the side of a nearer object that the object covers
   * in that eye. Below one, so each eye map keeps a unique contractive inverse. The side the object
   * uncovers keeps depth_coordinate_v2::max_horizontal_slope.
   */
  inline constexpr float covered_side_slope = 0.9f;

  /**
   * Cells over which a foreground value is carried unchanged before the ramp starts, on the side the
   * object covers and on the side it uncovers. At most 8 each (the shader's history length).
   * docs/host-sbs.md (eye-specific horizontal limiter) records the measured trade-off.
   */
  inline constexpr std::uint32_t covered_extension_cells = 4u;
  inline constexpr std::uint32_t uncovered_extension_cells = 4u;

  /** Both counts as the shader's `eye_extension_cells` word. */
  constexpr std::uint32_t extension_word(std::uint32_t covered, std::uint32_t uncovered) noexcept {
    return (covered & 0xffffu) | ((uncovered & 0xffffu) << 16u);
  }

  /** Attach (or, with null, detach) the companion. The Base view keeps one reference. */
  inline bool attach(ID3D11ShaderResourceView *base, ID3D11ShaderResourceView *eye) noexcept {
    return base && SUCCEEDED(base->SetPrivateDataInterface(companion_private_data, eye));
  }

  /** The companion of this Base view, or null. */
  inline Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> companion(ID3D11ShaderResourceView *base) noexcept {
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> eye;
    if (!base) {
      return eye;
    }
    IUnknown *raw = nullptr;
    UINT size = sizeof(raw);
    if (FAILED(base->GetPrivateData(companion_private_data, &size, &raw)) || !raw) {
      return eye;
    }
    // GetPrivateData returns an interface with one added reference.
    Microsoft::WRL::ComPtr<IUnknown> owned;
    owned.Attach(raw);
    owned.As(&eye);
    return eye;
  }

}  // namespace models::host_sbs_eye_field
