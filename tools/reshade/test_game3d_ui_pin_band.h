// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// GPU checks of the soft UI pin band (docs/reshade-sbs.md, UI pin band), shared
// by the D3D12 and D3D11 native renderer fixtures. Each case renders an R32
// red-channel UI mask and compares the pinned final field, through the
// unpinned field h of the same scene, with CPU references: the band in double
// precision, and for binary masks the distance rule the band replaced, in the
// shader's float32 arithmetic (each bound one multiply-add, fused or not).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sunshine_game3d_test {
  struct ui_pin_band_fixture {
    unsigned width{}, height{};
    float gain{}; // SUNSHINE_UI_SOFT_PIN_GAIN of the shader under test.
    // Uploads uniform (false) or structured (true) scene depth.
    std::function<void(bool structured)> depth;
    // The final field of one render: protection off without a mask, else on
    // with this R32 mask read through its red channel.
    std::function<std::vector<float>(const std::vector<float> *mask)> render;
  };

  namespace ui_pin_band_detail {
    inline void check(bool condition, const std::string &message) {
      if (!condition) throw std::runtime_error(message);
    }

    // field = clamp(h, p - b, p + b), b(x) = min_u r(u) + 0.5*max(|x - u| - 1, 0)/W,
    // r(u) = (1 - w(u))|h(u) - p| and w = saturate(gain * alpha) for finite
    // positive alpha. Rows without UI keep h.
    inline std::vector<double> band(const std::vector<float> &h, const std::vector<float> &alpha,
        unsigned width, unsigned height, double plane, double gain) {
      std::vector<double> out(h.begin(), h.end()), slack(width), bound(width);
      const double step = .5 / width, none = std::numeric_limits<double>::infinity();
      for (unsigned y = 0; y < height; ++y) {
        const size_t row = size_t(y) * width;
        bool ui = false;
        for (unsigned x = 0; x < width; ++x) {
          const float a = alpha[row + x];
          const double w = std::isfinite(a) && a > 0.f ? std::min(1., double(a) * gain) : 0.;
          slack[x] = w > 0 ? (1 - w) * std::abs(double(h[row + x]) - plane) : none;
          ui = ui || w > 0;
        }
        if (!ui) continue;
        // The collar shares each texel's slack with its neighbors; the ramp grows beyond it.
        for (unsigned x = 0; x < width; ++x)
          bound[x] = std::min({slack[x], x ? slack[x - 1] : none, x + 1 < width ? slack[x + 1] : none});
        for (unsigned x = 1; x < width; ++x) bound[x] = std::min(bound[x], bound[x - 1] + step);
        for (unsigned x = width - 1; x-- > 0;) bound[x] = std::min(bound[x], bound[x + 1] + step);
        for (unsigned x = 0; x < width; ++x)
          out[row + x] = std::clamp(double(h[row + x]), plane - bound[x], plane + bound[x]);
      }
      return out;
    }

    // The binary rule the band replaced: plane within one texel of UI, else
    // clamp(h, p -/+ 0.5*(d - 1)/W), each bound one float32 multiply-add.
    inline std::vector<float> distance_rule(const std::vector<float> &h, const std::vector<float> &alpha,
        unsigned width, unsigned height, float plane, bool fused) {
      const float ramp = float(.5 / width);
      std::vector<float> out = h;
      std::vector<int> left(width);
      for (unsigned y = 0; y < height; ++y) {
        const size_t row = size_t(y) * width;
        const auto ui = [&](unsigned x) { return std::isfinite(alpha[row + x]) && alpha[row + x] > 0.f; };
        int nearest = -1;
        for (unsigned x = 0; x < width; ++x) left[x] = nearest = ui(x) ? int(x) : nearest;
        nearest = -1;
        for (unsigned x = width; x-- > 0;) {
          nearest = ui(x) ? int(x) : nearest;
          const int distance = std::min(left[x] >= 0 ? int(x) - left[x] : int(width), nearest >= 0 ? nearest - int(x) : int(width));
          if (distance >= int(width)) continue;
          if (distance <= 1) { out[row + x] = plane; continue; }
          const float k = float(distance - 1);
          float product = k * ramp;
          const float low = fused ? std::fma(-k, ramp, plane) : plane - product;
          const float high = fused ? std::fma(k, ramp, plane) : plane + product;
          out[row + x] = std::min(std::max(h[row + x], low), high);
        }
      }
      return out;
    }

    inline double worst(const std::vector<float> &field, const std::vector<double> &expected) {
      double result = 0;
      for (size_t i = 0; i < field.size(); ++i) result = std::max(result, std::abs(double(field[i]) - expected[i]));
      return result;
    }

    inline bool same_row(const std::vector<float> &a, const std::vector<float> &b, unsigned width, unsigned y) {
      return !std::memcmp(a.data() + size_t(y) * width, b.data() + size_t(y) * width, width * sizeof(float));
    }
  }

  inline void check_ui_pin_band(const ui_pin_band_fixture &f, std::ostream &report, const std::string &api) {
    using namespace ui_pin_band_detail;
    const unsigned width = f.width, height = f.height;
    const size_t pixels = size_t(width) * height;
    // Fields are signed source-U displacements; one source pixel is 1/W.
    const double px = width, tolerance = 1e-5 / px;
    check(f.gain >= 1.f && width >= 64 && height >= 64, api + " UI pin band fixture is not configured");

    // A uniform scene: the all-ones plane, an isolated faint texel and a
    // vertical alpha ramp from zero to full weight.
    f.depth(false);
    const auto flat = f.render(nullptr);
    std::vector<float> mask(pixels, 1.f);
    const auto all = f.render(&mask);
    const float plane = all.front();
    for (const float value : all) check(value == plane, api + " all-ones UI mask is not one rigid plane");
    for (const float value : flat) check(value == flat.front(), api + " UI pin band scene is not uniform");
    const double offset = std::abs(double(flat.front()) - plane);
    check(offset * px >= 2, api + " UI pin band scene lies within two pixels of the UI plane");

    std::fill(mask.begin(), mask.end(), 0.f);
    const unsigned faint_x = width / 2, faint_y = height / 2;
    const size_t faint = size_t(faint_y) * width + faint_x;
    mask[faint] = 1.f / 255.f;
    auto field = f.render(&mask);
    const double weight = std::min(1., f.gain / 255.);
    const double lerp = double(flat[faint]) + weight * (double(plane) - flat[faint]);
    const double faint_error = std::abs(double(field[faint]) - lerp);
    check(faint_error <= tolerance, api + " a 1/255 UI texel does not pin in proportion to its weight");
    for (unsigned y = 0; y < height; ++y)
      check(y == faint_y || same_row(field, flat, width, y), api + " a row without UI changed");
    check(worst(field, band(flat, mask, width, height, plane, f.gain)) <= tolerance,
      api + " a faint UI texel's collar and ramp differ from the band");

    // The knee, one texel per row: the first 8-bit code at or above
    // 1/gain and 1/gain itself pin exactly; the code below it and the float
    // just below 1/gain pin in proportion (the shader's float32 weight).
    std::fill(mask.begin(), mask.end(), 0.f);
    float knee = 1.f / f.gain;
    if (knee * f.gain < 1.f) knee = std::nextafter(knee, 1.f);
    const float code = std::ceil(255.f / f.gain);
    const float knee_alpha[]{(code - 1.f) / 255.f, code / 255.f, std::nextafter(knee, 0.f), knee};
    const auto knee_at = [&](unsigned i) { return size_t(height / 8 + 4 * i) * width + width / 4 + 8 * i; };
    for (unsigned i = 0; i < 4; ++i) mask[knee_at(i)] = knee_alpha[i];
    field = f.render(&mask);
    for (unsigned i = 0; i < 4; ++i) {
      const float a = knee_alpha[i], w = std::min(1.f, a * f.gain);
      check((w >= 1.f) == (i % 2 == 1), api + " the knee fixture does not straddle 1/gain");
      if (w >= 1.f) check(field[knee_at(i)] == plane, api + " UI alpha at the soft pin knee did not pin exactly");
      else if (i == 0) check(field[knee_at(i)] != plane, api + " UI alpha just below the soft pin knee pinned exactly");
    }
    const double knee_error = worst(field, band(flat, mask, width, height, plane, f.gain));
    check(knee_error <= tolerance, api + " UI alpha around the soft pin knee differs from the band");

    // Across the whole width, weights k/16 over 16 rows, then full weight to
    // the bottom of the frame.
    std::fill(mask.begin(), mask.end(), 0.f);
    const unsigned ramp_y = height / 4;
    for (unsigned y = ramp_y + 1; y < height; ++y)
      std::fill_n(mask.begin() + size_t(y) * width, width, float(std::min(y - ramp_y, 16u)) / (16.f * f.gain));
    field = f.render(&mask);
    double steepest = 0;
    for (unsigned y = 0; y < height; ++y) {
      if (y + 1 < height) steepest = std::max(steepest, std::abs(double(field[size_t(y + 1) * width]) - field[size_t(y) * width]));
      check(y > ramp_y ? !same_row(field, flat, width, y) : same_row(field, flat, width, y),
        api + " the UI alpha ramp pinned the wrong rows");
    }
    check(steepest <= offset / 16 + 1e-6 / px, api + " an alpha ramp to full weight over 16 rows steps by more than 1/16 of the offset");
    check(field[size_t(ramp_y + 20) * width] == plane, api + " full-weight UI rows left the plane");
    check(worst(field, band(flat, mask, width, height, plane, f.gain)) <= tolerance, api + " the UI alpha ramp differs from the band");

    // A structured scene: random soft alpha keeps the horizontal slope bound
    // and matches the band; binary masks keep the distance rule bit for bit.
    f.depth(true);
    const auto h = f.render(nullptr);
    std::uint32_t state = 0x2545f491u;
    const auto next = [&] { state = state * 1664525u + 1013904223u; return state >> 8; };
    std::fill(mask.begin(), mask.end(), 0.f);
    for (unsigned y = height / 4; y < height * 3 / 4; ++y)
      for (unsigned x = width / 4; x < width * 3 / 4; ++x) {
        const auto pick = next() % 10u;
        mask[size_t(y) * width + x] = pick < 4u ? 0.f : pick < 7u ? float(next() % 255u + 1u) / (255.f * f.gain) :
          float(next() % 1024u + 1u) / 1024.f;
      }
    field = f.render(&mask);
    const double slope = .5 / px;
    double steepest_x = 0;
    for (unsigned y = 0; y < height; ++y)
      for (unsigned x = 0; x + 1 < width; ++x)
        steepest_x = std::max(steepest_x, std::abs(double(field[size_t(y) * width + x + 1]) - field[size_t(y) * width + x]));
    // Allow float32 rounding of the field values (a few ulps of 0.04).
    check(steepest_x <= slope * (1 + 1e-4), api + " soft UI pinning exceeded the 0.5/W horizontal slope bound");
    const double random_error = worst(field, band(h, mask, width, height, plane, f.gain));
    check(random_error <= tolerance, api + " random soft UI alpha differs from the band");
    for (unsigned y = 0; y < height; ++y)
      check((y >= height / 4 && y < height * 3 / 4) || same_row(field, h, width, y), api + " a row without UI changed");

    // A rectangle of quarter alpha (at or above the knee) and sparse opaque
    // texels. A device rounds the rule's multiply-adds once or twice, so either
    // replica is exact here. reshade_game3d_native_shader_test pins the
    // shader's instructions to the rule's non-precise multiply-adds, and the
    // source-alpha fixture compares binary masks with a frozen control shader.
    std::fill(mask.begin(), mask.end(), 0.f);
    for (unsigned y = height / 4; y < height * 3 / 4; ++y)
      for (unsigned x = width / 3; x < width / 2; ++x) mask[size_t(y) * width + x] = .25f;
    for (unsigned y = 3; y < height; y += 11)
      for (unsigned x = 7 + y % 5; x < width; x += 53) mask[size_t(y) * width + x] = 1.f;
    field = f.render(&mask);
    const auto unfused = distance_rule(h, mask, width, height, plane, false);
    const auto fused = distance_rule(h, mask, width, height, plane, true);
    const bool exact_unfused = !std::memcmp(field.data(), unfused.data(), pixels * sizeof(float));
    const bool exact_fused = !std::memcmp(field.data(), fused.data(), pixels * sizeof(float));
    check(exact_unfused || exact_fused, api + " a binary UI mask no longer reproduces the distance rule bit for bit");
    check(field != h, api + " the binary UI mask had no effect");
    size_t rounding_differs = 0;
    for (size_t i = 0; i < pixels; ++i) rounding_differs += fused[i] != unfused[i];

    report << "ui-pin-band " << api << " gain=" << f.gain << " offset_px=" << offset * px << " faint_error_px=" << faint_error * px
      << " knee_band_error_px=" << knee_error * px << " ramp_step_px=" << steepest * px << " ramp_limit_px=" << offset * px / 16
      << " slope_px=" << steepest_x * px << " random_band_error_px=" << random_error * px
      << " binary_exact=" << (exact_fused ? "fused" : "unfused") << " rounding_sensitive_texels=" << rounding_differs << '\n';
    std::printf("PASS %s UI pin band: 1/255 texel pins by %.9g, knee exact at %g/255 and 1/gain, 16-row ramp steps %.6g px <= %.6g, "
      "random soft alpha within %.3g px of the band and slope %.9g px, binary masks exact (%s multiply-add, %zu texels "
      "rounding-sensitive), rows without UI untouched\n", api.c_str(), weight, code, steepest * px, offset * px / 16, random_error * px,
      steepest_x * px, exact_fused ? "fused" : "unfused", rounding_differs);
  }
}
