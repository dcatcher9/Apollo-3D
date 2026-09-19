// SPDX-License-Identifier: GPL-3.0-only
#pragma once

// Test-only C boundary around Sunshine's production receiver. No C++/ReShade objects or
// ownership cross this boundary; the texture is borrowed until the next poll or destruction.
#include <cstdint>
#include <d3d11.h>

struct sunshine_receiver_test_frame {
  ID3D11Texture2D *texture = nullptr;
  std::uint32_t linear = 0;
  std::uint64_t sequence = 0;
  std::int64_t timestamp_ns = 0;
};

using sunshine_receiver_test_create = void *(*) (ID3D11Device *, ID3D11DeviceContext *, HWND, DWORD, const RECT *);
using sunshine_receiver_test_focus = void (*)(void *, BOOL);
using sunshine_receiver_test_poll = int (*)(void *, sunshine_receiver_test_frame *);
using sunshine_receiver_test_destroy = void (*)(void *);
