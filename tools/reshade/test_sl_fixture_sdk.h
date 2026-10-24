// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// What every synthetic Streamline SDK the tests build shares: the noinline
// marker for entry points that MinHook patches, and the structure type GUIDs
// of PrecisionInfo (sl_core_types.h) and DLSSGOptions (sl_dlss_g.h), which
// the probe matches by value. Each executable keeps its own entry points,
// ReShade exports and structure defaults, which differ by design: the probe
// tests' entry points record every call and can block or lose observation
// inside it, and their ReShade exports keep log lines and settable
// configuration; the contract matrix's entry points only return the result
// set for their boundary, and its exports drop log lines.
#include "streamline_camera_data.h"

#if defined(__GNUC__) && !defined(__clang__)
  #define FIXTURE_NOINLINE __attribute__((noinline, noipa))
#else
  #define FIXTURE_NOINLINE __declspec(noinline)
#endif

namespace sunshine_streamline::fixture_sdk {
  inline constexpr guid precision_type {0x98f6e9ba, 0x8d16, 0x4831, {0xa8, 0x02, 0x4d, 0x3b, 0x52, 0xff, 0x26, 0xbf}};
  inline constexpr guid fg_options_type {0xfac5f1cb, 0x2dfd, 0x4f36, {0xa1, 0xe6, 0x3a, 0x9e, 0x86, 0x52, 0x56, 0xc5}};
}  // namespace sunshine_streamline::fixture_sdk
