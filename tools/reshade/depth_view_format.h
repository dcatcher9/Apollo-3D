// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <dxgiformat.h>

// The one rule for how an API depth input is stored and read. The capture's
// snapshot storage and shader-resource view, the content sampler's view and
// the Streamline 1.1.1 adapter's shape gate all use it, so they accept the
// same set of formats.
namespace sunshine_depth_format {
  // The typeless storage family of a depth input's allocation format (the
  // capture copies into it), or UNKNOWN for a format that is not depth.
  constexpr DXGI_FORMAT storage(DXGI_FORMAT format) {
    switch (format) {
      case DXGI_FORMAT_D32_FLOAT: case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_R32_FLOAT:
        return DXGI_FORMAT_R32_TYPELESS;
      case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: case DXGI_FORMAT_R32G8X24_TYPELESS: case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
        return DXGI_FORMAT_R32G8X24_TYPELESS;
      case DXGI_FORMAT_D24_UNORM_S8_UINT: case DXGI_FORMAT_R24G8_TYPELESS: case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
        return DXGI_FORMAT_R24G8_TYPELESS;
      case DXGI_FORMAT_D16_UNORM: case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_R16_UNORM: case DXGI_FORMAT_R16_FLOAT:
        return DXGI_FORMAT_R16_TYPELESS;
      default: return DXGI_FORMAT_UNKNOWN;
    }
  }
  // The view that reads depth from an allocation, in this order:
  // 1. the allocation's own typed format when it is R32_FLOAT, R16_FLOAT or R16_UNORM;
  // 2. the depth alias of a depth or depth/stencil family (D32, D16, D24S8 and
  //    D32S8 with their typeless forms; R32_TYPELESS reads as R32_FLOAT, its
  //    only float view);
  // 3. for R16_TYPELESS, a declared typed format of that family that reads
  //    depth (Streamline Resource.nativeFormat: R16_FLOAT, R16_UNORM, D16);
  // 4. for R16_TYPELESS without one, R16_UNORM only for device depth (a depth
  //    buffer's own encoding). A linear input is never guessed: UNKNOWN.
  // A declaration outside the allocation's family is ignored.
  constexpr DXGI_FORMAT view(DXGI_FORMAT allocation, DXGI_FORMAT declared = DXGI_FORMAT_UNKNOWN, bool device_depth = true) {
    switch (allocation) {
      case DXGI_FORMAT_R32_FLOAT: case DXGI_FORMAT_R16_FLOAT: case DXGI_FORMAT_R16_UNORM: return allocation;
      case DXGI_FORMAT_D32_FLOAT: case DXGI_FORMAT_R32_TYPELESS: return DXGI_FORMAT_R32_FLOAT;
      case DXGI_FORMAT_D16_UNORM: return DXGI_FORMAT_R16_UNORM;
      case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: case DXGI_FORMAT_R32G8X24_TYPELESS: case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
        return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
      case DXGI_FORMAT_D24_UNORM_S8_UINT: case DXGI_FORMAT_R24G8_TYPELESS: case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
        return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
      case DXGI_FORMAT_R16_TYPELESS:
        if (declared == DXGI_FORMAT_R16_FLOAT) return DXGI_FORMAT_R16_FLOAT;
        if (declared == DXGI_FORMAT_R16_UNORM || declared == DXGI_FORMAT_D16_UNORM) return DXGI_FORMAT_R16_UNORM;
        return device_depth ? DXGI_FORMAT_R16_UNORM : DXGI_FORMAT_UNKNOWN;
      default: return DXGI_FORMAT_UNKNOWN;
    }
  }
  // A view the content sampler may read a snapshot of storage through.
  constexpr bool readable(DXGI_FORMAT storage_format, DXGI_FORMAT view_format) {
    return view_format != DXGI_FORMAT_UNKNOWN && storage(view_format) == storage(storage_format);
  }

  static_assert(view(DXGI_FORMAT_R16_FLOAT) == DXGI_FORMAT_R16_FLOAT && storage(DXGI_FORMAT_R16_FLOAT) == DXGI_FORMAT_R16_TYPELESS);
  static_assert(view(DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_UNKNOWN, false) == DXGI_FORMAT_UNKNOWN &&
    view(DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_FLOAT, false) == DXGI_FORMAT_R16_FLOAT &&
    view(DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R32_FLOAT, true) == DXGI_FORMAT_R16_UNORM &&
    view(DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_UNKNOWN, false) == DXGI_FORMAT_R32_FLOAT);
}
