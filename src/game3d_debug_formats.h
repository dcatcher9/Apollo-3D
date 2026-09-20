// SPDX-License-Identifier: GPL-3.0-only
#pragma once

namespace game3d_debug {
  // Exact storage widths for the uncompressed, single-plane typed DXGI
  // formats admitted by native diagnostic capture. No color/mask decoding is
  // implied. Numeric DXGI IDs keep this contract independent of Windows headers.
  inline constexpr unsigned pixel_bytes(unsigned format) noexcept {
    switch (format) {
      case 61: case 62: case 63: case 64: // R8 UNORM/UINT/SNORM/SINT
        return 1;
      case 49: case 50: case 51: case 52: // RG8
      case 54: case 56: case 57: case 58: case 59: // R16
        return 2;
      case 28: case 29: case 30: case 31: case 32: // RGBA8
      case 87: case 91: case 88: case 93: // BGRA/BGRX UNORM and sRGB
      case 24: case 25: case 26: case 67: // RGB10A2/R11G11B10/RGB9E5
      case 34: case 35: case 36: case 37: case 38: // RG16
      case 41: case 42: case 43: // R32
        return 4;
      case 10: case 11: case 12: case 13: case 14: // RGBA16
      case 16: case 17: case 18: // RG32
        return 8;
      case 2: case 3: case 4: // RGBA32
        return 16;
      default:
        return 0;
    }
  }
}
