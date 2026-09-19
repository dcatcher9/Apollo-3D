// SPDX-License-Identifier: GPL-3.0-only
#include "raw_reference_statistics.h"

#include <cstdio>
#include <stdexcept>

namespace {
  using namespace sunshine_raw_reference;
  void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
  }
  void close(double actual, double expected, double tolerance, const char *message) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, message);
  }
  void print(const char *label, const statistics &v) {
    std::printf("MEASURE %s valid=%u usable=%u mean=%.12g center=%.12g sigma=%.12g mean_gain=%.12g sigma_gain=%.12g participation=%.12g endpoints=%u\n",
      label, v.valid, usable_reference(v), v.mean, v.center_mean, v.sigma,
      v.scene_mean_gain, v.sigma_gain, v.effective_contributors, v.endpoint_cells);
  }
  grid gradient() {
    grid result;
    for (unsigned i = 0; i < cell_count; ++i) result[i] = .125f + (i % width) / 256.0f;
    return result;
  }
  grid affine(grid value, float a, float b) {
    for (float &t : value) t = a * t + b;
    return value;
  }
  void moments_and_endpoints() {
    grid depth;
    for (unsigned i = 0; i < cell_count; ++i) depth[i] = i < cell_count / 2 ? .25f : .5f;
    const auto measured = measure(depth);
    require(usable_reference(measured), "Known two-plane reference invalid");
    close(measured.mean, .375, 1e-14, "Wrong scene mean");
    close(measured.variance, .015625, 1e-14, "Wrong spatial variance");
    close(measured.sigma_gain, 8.0, 1e-12, "Wrong sigma gain");
    close(measured.effective_contributors, cell_count, 1e-9, "Wrong participation");
    print("two-plane", measured);
    for (unsigned i = 0; i < cell_count; ++i) depth[i] = i < cell_count / 2 ? 0.0f : .5f;
    const auto sky = measure(depth), normal = measure(depth, 1, true);
    require(usable_reference(sky) && usable_reference(normal), "Mixed sky scene rejected");
    close(sky.mean, .25, 1e-14, "Sky endpoints dropped");
    close(sky.sigma, .25, 1e-14, "Sky endpoints clipped");
    close(normal.sigma, sky.sigma, 1e-14, "Exact FP32 flipped scene spread differs");
    require(sky.endpoint_cells == cell_count / 2, "Sky endpoints missing");
    print("half-sky-endpoint", sky);
    for (float endpoint : {0.0f, 1.0f}) {
      depth.fill(endpoint);
      require(!usable_reference(measure(depth)), "Clear-only reference accepted");
    }
  }
  void affine_and_foreground() {
    const auto source = gradient();
    const auto base = measure(source), shifted = measure(affine(source, 1, .25f));
    const auto scaled = measure(affine(source, .5f, .125f));
    print("base-gradient", base); print("additive-offset", shifted); print("affine-scale-offset", scaled);
    close(shifted.sigma_gain, base.sigma_gain, 1e-10, "Additive offset changed gain");
    close(scaled.sigma_gain, 2 * base.sigma_gain, 1e-10, "Raw scale did not invert gain");
    require(shifted.scene_mean_gain != base.scene_mean_gain, "Mean offset dependence concealed");
    auto foreground = source;
    for (unsigned y = 7; y < 11; ++y)
      for (unsigned x = 14; x < 18; ++x) foreground[y * width + x] = .75f;
    const auto fg = measure(foreground);
    require(usable_reference(fg), "Distributed localized foreground rejected");
    print("localized-center-foreground", fg);
    const double old_ratio = base.center_mean / fg.center_mean;
    const double new_ratio = fg.sigma_gain / base.sigma_gain;
    require(new_ratio > old_ratio && new_ratio < 1, "Center domination was not reduced in this fixture");
    require(fg.sigma_gain != base.sigma_gain, "Real startup-scene dependence concealed");
    close(measure(source, 2).sigma_gain, 2 * base.sigma_gain, 1e-10, "Artistic target not explicit");
  }
  void unusable_and_fp32_orientation() {
    grid depth; depth.fill(.25f);
    require(measure(depth).valid && !usable_reference(measure(depth)), "Flat scene invented gain");
    depth[123] = .875f;
    require(!usable_reference(measure(depth)), "Single outlier controlled reference");
    print("single-outlier", measure(depth));
    for (unsigned i = 0; i < 4; ++i) depth[i] = .875f;
    require(!usable_reference(measure(depth)), "Few outliers controlled reference");
    // These are valid small foreground objects on a planar background, not
    // known corruption. The participation gate cannot distinguish the two.
    // Its conservative rejection can delay calibration in a legitimate view.
    for (unsigned foreground_cells : {15u, 16u}) {
      grid small_foreground;
      small_foreground.fill(.25f);
      for (unsigned i = 0; i < foreground_cells; ++i) small_foreground[i] = .75f;
      const auto original = small_foreground;
      const auto measured = measure(small_foreground);
      require(measured.valid && small_foreground == original,
        "Sparse valid foreground was modified or called invalid depth");
      require(usable_reference(measured) == (foreground_cells == 16),
        "Valid small foreground did not expose the participation admission boundary");
      print(foreground_cells == 15 ? "valid-small-foreground-15-uncertain-rejected" :
        "valid-small-foreground-16-uncertain-admitted", measured);
    }
    for (unsigned i = 0; i < cell_count; ++i) depth[i] = i & 1 ? .5f : std::nextafter(.5f, 1.0f);
    require(usable_reference(measure(depth)) && measure(depth).sigma_gain > 16383,
      "Statistics silently capped unrepresentable gain");
    print("one-ulp-noise-renderer-domain-rejection-required", measure(depth));
    for (unsigned i = 0; i < cell_count; ++i) depth[i] = i & 1 ? .25f : .5f;
    require(usable_reference(measure(depth)), "Persistent spatial-noise ambiguity concealed");
    print("persistent-spatial-noise-ambiguity", measure(depth));
    for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -.01f, 1.01f}) {
      auto malformed = gradient(); malformed[0] = bad;
      require(!measure(malformed).valid && !measure(malformed, 1, true).valid, "Invalid raw cell discarded or clamped");
    }
    for (unsigned i = 0; i < cell_count; ++i) depth[i] = i & 1 ? 0x1p-28f : 0x1p-27f;
    const auto input = depth;
    const auto unflipped = measure(depth), flipped = measure(depth, 1, true);
    require(unflipped.sigma > 0 && flipped.sigma == 0 && !usable_reference(flipped),
      "FP32 1-raw quantization was treated as exact mathematical flip invariance");
    require(depth == input, "Measuring orientation modified input grid");
    print("shader-fp32-flip-collapse", flipped);
  }
}
int main() {
  try {
    moments_and_endpoints(); affine_and_foreground(); unusable_and_fp32_orientation();
    std::puts("PASS stateless raw reference statistics. Temporal/source/readiness coverage belongs to raw_scene_policy; no physical scale or shipping quality claim.");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL raw reference statistics: %s\n", error.what());
    return 1;
  }
}
