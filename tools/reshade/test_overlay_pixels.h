// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

// Independent pixel oracle for the actual-runtime fixtures. Readback exists only in tests.
namespace sunshine_sbs_overlay_pixels {
  inline float decode_srgb(float value) {
    return value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
  }

  inline float encode_srgb(float value) {
    return value <= 0.0031308f ? value * 12.92f : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
  }

  inline float half(std::uint16_t value) {
    const unsigned exponent = (value >> 10) & 31, fraction = value & 1023;
    if (exponent == 31) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    const float result = exponent == 0 ? std::ldexp(static_cast<float>(fraction), -24) :
                                         std::ldexp(1.0f + fraction / 1024.0f, static_cast<int>(exponent) - 15);
    return value & 0x8000 ? -result : result;
  }

  inline bool check(const void *data, unsigned pitch, unsigned format, unsigned width, unsigned height, bool overlay, bool patches, float gain) {
    const auto pixel = [&](unsigned x, unsigned y) {
      std::array<float, 3> result {};
      const auto *row = static_cast<const std::uint8_t *>(data) + y * pitch;
      for (unsigned c = 0; c < 3; ++c) {
        result[c] = format == 24 ? ((reinterpret_cast<const std::uint32_t *>(row)[x] >> (c * 10)) & 1023) / 1023.0f :
                                   half(reinterpret_cast<const std::uint16_t *>(row)[x * 4 + c]);
      }
      return result;
    };
    const float sdr[2][3] {{1, 0.25f, 0}, {0, 0.5f, 1}};
    const float hdr[2][3] {{4, 0.5f, -0.125f}, {0.125f, 2, 8}};
    const auto &base = format == 24 ? sdr : hdr;
    // ReShade.ini fixes its native UI white at 203 nits; scRGB 1.0 is 80 nits.
    const float white = format == 24 ? 1.0f : 203.0f / 80.0f;
    const float tolerance = format == 24 ? 0.004f : 0.012f;
    const unsigned locations[3][2] {{width - 200, height - 64}, {width - 96, height - 144}, {width - 96, height - 80}};
    for (unsigned eye = 0; eye < 2; ++eye) {
      for (unsigned probe = 0; probe < 3; ++probe) {
        const float alpha = !patches || probe == 0 ? 0.0f : probe == 1 ? 1.0f :
                                                                         128.0f / 255.0f;
        const auto actual = pixel(eye * width + locations[probe][0], locations[probe][1]);
        for (unsigned c = 0; c < 3; ++c) {
          // The exported UI blends in linear light; the game's native UI is left alone.
          const float background = base[eye][c] * gain;
          const float expected = format == 24 && alpha > 0 ? encode_srgb(decode_srgb(background) * (1 - alpha) + alpha) :
                                                             background * (1 - alpha) + white * alpha;
          if (!std::isfinite(actual[c]) || std::abs(actual[c] - expected) > tolerance) {
            std::printf("overlay pixel mismatch eye=%u probe=%u channel=%u actual=%.6f expected=%.6f overlay=%d gain=%.2f\n", eye, probe, c, actual[c], expected, overlay, gain);
            return false;
          }
        }
      }
    }
    for (unsigned eye = 0; eye < 2; ++eye) {
      // These points are in the native ReShade controls dock, far from the test patches.
      // Requiring many changed pixels rejects a renderer that mirrors only our markers.
      unsigned controls = 0;
      for (unsigned y = 96; y < height - 32; y += 24) {
        for (unsigned x = 24; x < width / 3 - 24; x += 24) {
          const auto actual = pixel(eye * width + x, y);
          bool changed = false;
          for (unsigned c = 0; c < 3; ++c) {
            if (!std::isfinite(actual[c])) {
              std::printf("non-finite native GUI pixel: eye=%u x=%u y=%u channel=%u\n", eye, x, y, c);
              return false;
            }
            changed |= std::abs(actual[c] - base[eye][c] * gain) > 0.05f;
          }
          controls += changed;
        }
      }
      if ((overlay && controls < 20) || (!overlay && controls != 0)) {
        std::printf("native ReShade controls visibility mismatch: eye=%u overlay=%d changed probes=%u\n", eye, overlay, controls);
        return false;
      }
    }
    return true;
  }

  // Native add-on output of a flat game frame with no depth: both eyes carry the
  // source color, encoded for the export (color 1 = 8-bit sRGB game, 2 = scRGB,
  // 3 = PQ BT.2020 game exported as scRGB), with the optional UI patches.
  inline bool check_native(const void *data, unsigned pitch, unsigned format, unsigned width, unsigned height,
    unsigned color, std::array<float, 3> expected, bool overlay, bool patches) {
    if (color == 1) {
      for (unsigned c = 0; c < 3; ++c) expected[c] = std::round(expected[c] * 255.f) / 255.f;
    } else if (color == 3) {
      std::array<double, 3> linear {};
      for (unsigned c = 0; c < 3; ++c) {
        const double code = std::round(expected[c] * 1023.0) / 1023.0;
        const double p = std::pow(code, 32.0 / 2523.0);
        linear[c] = 125.0 * std::pow(std::max(p - 3424.0 / 4096.0, 0.0) /
          (2413.0 / 128.0 - 2392.0 / 128.0 * p), 16384.0 / 2610.0);
      }
      expected[0] = float(1.6604910021 * linear[0] - .5876411388 * linear[1] - .0728498633 * linear[2]);
      expected[1] = float(-.1245504745 * linear[0] + 1.1328998971 * linear[1] - .0083494226 * linear[2]);
      expected[2] = float(-.0181507634 * linear[0] - .1005788980 * linear[1] + 1.1187296614 * linear[2]);
    }
    const auto pixel = [&](unsigned x, unsigned y, unsigned c) {
      const auto row = static_cast<const std::uint8_t *>(data) + y * pitch;
      return format == 24 ? float((reinterpret_cast<const std::uint32_t *>(row)[x] >> (10 * c)) & 1023) / 1023.f :
        half(reinterpret_cast<const std::uint16_t *>(row)[x * 4 + c]);
    };
    const float tolerance = format == 24 ? .004f : .012f;
    const float white = format == 24 ? 1.f : 203.f / 80.f;
    const unsigned probes[][2] {{width - 200, height - 64}, {width - 96, height - 144}, {width - 96, height - 80}};
    for (unsigned eye = 0; eye < 2; ++eye) {
      for (unsigned probe = 0; probe < 3; ++probe) {
        const float alpha = !overlay || !patches || probe == 0 ? 0.f : probe == 1 ? 1.f : 128.f / 255.f;
        for (unsigned c = 0; c < 3; ++c) {
          const float wanted = format == 24 && alpha > 0 ?
            encode_srgb(decode_srgb(expected[c]) * (1 - alpha) + alpha) : expected[c] * (1 - alpha) + white * alpha;
          const float actual = pixel(eye * width + probes[probe][0], probes[probe][1], c);
          if (!std::isfinite(actual) || std::abs(actual - wanted) > tolerance) {
            std::printf("native no-FX pixel mismatch eye=%u probe=%u channel=%u actual=%.6g expected=%.6g overlay=%u\n",
              eye, probe, c, actual, wanted, unsigned(overlay));
            return false;
          }
        }
      }
      unsigned changed = 0;
      for (unsigned y = 96; y < height - 32; y += 24) for (unsigned x = 24; x < width / 3 - 24; x += 24) {
        bool different = false;
        for (unsigned c = 0; c < 3; ++c) {
          const float actual = pixel(eye * width + x, y, c);
          if (!std::isfinite(actual)) return false;
          different |= std::abs(actual - expected[c]) > .05f;
        }
        changed += different;
      }
      if ((overlay && changed < 20) || (!overlay && changed != 0)) return false;
    }
    return true;
  }
}  // namespace sunshine_sbs_overlay_pixels
