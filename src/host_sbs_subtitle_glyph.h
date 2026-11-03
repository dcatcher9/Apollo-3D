/** @file Independent current-source glyph-mask authentication for saved GPU artifacts. */
#pragma once

#include "generated/depth_coordinate_v2_contract.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace models::host_sbs_subtitle_glyph {
  namespace v2 = depth_coordinate_v2;
  using rgb_t = std::array<float, 3u>;
  using rect_t = std::array<std::uint32_t, 4u>;
  struct pair_t { rect_t core, cover; bool ribbon = false; };
  struct sample_t { rgb_t rgb {}; bool valid {}; };

  // This verifier runs only on immutable dump bytes. It does not supply live render state.
  template<class LoadColor, class LoadProbability>
  std::vector<std::uint8_t> current_support(
    std::uint32_t width, std::uint32_t height, std::uint32_t source_height,
    std::uint32_t crop_top, const rect_t &content, std::span<const pair_t> pairs,
    LoadColor load_color, LoadProbability load_probability,
    std::optional<std::int32_t> current_reference_center_twice_source = std::nullopt
  ) {
    using qrgb_t = std::array<std::uint32_t, 3u>;
    struct mode_t { qrgb_t sum {}; std::uint32_t count {}; std::uint32_t bin {}; };
    const auto pixels = static_cast<std::size_t>(width) * height;
    // A nonzero classifier payload is validated by the OCR owner before this helper is
    // called. Only its authenticated signed center is accepted here; nullopt is zero-tail
    // source-center fallback, never a fallback for a malformed nonzero payload.
    if (!width || !height || !source_height || width > v2::subtitle_contour_source_max_extent ||
        source_height > v2::subtitle_contour_source_max_extent || crop_top > source_height ||
        height > source_height - crop_top || content[0] >= content[2] || content[1] >= content[3] ||
        (current_reference_center_twice_source &&
          (*current_reference_center_twice_source < -static_cast<std::int32_t>(v2::subtitle_contour_source_max_extent) ||
           *current_reference_center_twice_source > static_cast<std::int32_t>(v2::subtitle_contour_source_max_extent))))
      return std::vector<std::uint8_t>(pixels);
    std::vector<sample_t> colors(pixels);
    std::vector<qrgb_t> quantized(pixels);
    std::vector<std::uint8_t> edges(pixels), result(pixels);
    const auto quantize = [](float value) {
      const volatile float product = value * static_cast<float>(v2::subtitle_contour_palette_channel_max);
      return static_cast<std::uint32_t>(std::floor(product + .5f));
    };
    const auto distance = [](const rgb_t &a, const rgb_t &b) {
      return std::max({std::abs(a[0] - b[0]), std::abs(a[1] - b[1]), std::abs(a[2] - b[2])});
    };
    const auto inside = [&](int x, int y) {
      return x >= 0 && y >= 0 && x < static_cast<int>(width) && y < static_cast<int>(height);
    };
    for (std::size_t p = 0; p < pixels; ++p) {
      colors[p] = load_color(p);
      if (colors[p].valid) for (std::size_t c = 0; c < 3; ++c) quantized[p][c] = quantize(colors[p].rgb[c]);
    }
    constexpr int radius = v2::subtitle_contour_spatial_edge_radius_source_pixels;
    for (std::size_t p = 0; p < pixels; ++p) {
      if (!colors[p].valid) continue;
      const int x = p % width, y = p / width;
      for (const auto &[dx, dy] : std::array {std::pair {radius, 0}, std::pair {-radius, 0},
             std::pair {0, radius}, std::pair {0, -radius}}) {
        if (!inside(x + dx, y + dy)) continue;
        const auto q = static_cast<std::size_t>(y + dy) * width + x + dx;
        if (colors[q].valid && distance(colors[p].rgb, colors[q].rgb) > v2::subtitle_contour_spatial_edge_rgb_distance)
          edges[p] |= dx == 0 ? 2u : 1u;
      }
    }
    const auto content_width = content[2] - content[0], content_height = content[3] - content[1];
    const auto covered = [&](std::size_t p, const rect_t &rect) {
      const auto x = p % width, y = p / width;
      const std::uint64_t sx = 2ull * content[0] * width + (2ull * x + 1) * content_width;
      const std::uint64_t sy = 2ull * content[1] * source_height + (2ull * (y + crop_top) + 1) * content_height;
      return sx >= 2ull * rect[0] * width && sx < 2ull * rect[2] * width &&
        sy >= 2ull * rect[1] * source_height && sy < 2ull * rect[3] * source_height;
    };
    const auto peak = [](const mode_t &mode) { return std::max({mode.sum[0], mode.sum[1], mode.sum[2]}); };
    constexpr auto channel_max = v2::subtitle_contour_palette_channel_max;
    const auto matches = [&](const qrgb_t &color, const mode_t &mode) {
      std::array<std::int64_t, 3> delta {};
      for (std::size_t c = 0; c < 3; ++c) delta[c] = static_cast<std::int64_t>(color[c]) * mode.count - mode.sum[c];
      const auto absolute = std::max({std::abs(delta[0]), std::abs(delta[1]), std::abs(delta[2])});
      const std::int64_t scale = static_cast<std::int64_t>(channel_max) * mode.count;
      return v2::subtitle_contour_palette_match_rgb_denominator * absolute <= v2::subtitle_contour_palette_match_rgb_numerator * scale;
    };
    for (const auto &pair : pairs) {
      auto search = pair.cover;
      if (!pair.ribbon) {
        auto left = pair.core[0], right = pair.core[2];
        for (const auto &other : pairs) {
          if (other.ribbon) continue;
          const auto top = std::max(pair.core[1], other.core[1]), bottom = std::min(pair.core[3], other.core[3]);
          if (top >= bottom) continue;
          const auto union_height = std::max(pair.core[3], other.core[3]) - std::min(pair.core[1], other.core[1]);
          if (static_cast<std::uint64_t>(bottom - top) * v2::subtitle_ocr_line_group_min_vertical_iou_denominator <
              static_cast<std::uint64_t>(union_height) * v2::subtitle_ocr_line_group_min_vertical_iou_numerator) continue;
          left = std::min(left, other.core[0]);
          right = std::max(right, other.core[2]);
        }
        const std::int64_t center = current_reference_center_twice_source.value_or(static_cast<std::int32_t>(width));
        const std::int64_t line = (static_cast<std::int64_t>(left) + right - 2ll * content[0]) * width * v2::subtitle_ocr_output_width;
        const std::int64_t reference = center * v2::subtitle_ocr_output_width;
        const std::int64_t bias = 2ll * v2::subtitle_contour_ordinary_search_center_bias_detector_pixels * width;
        const bool right_biased = line > (reference + bias) * content_width;
        const bool left_biased = (reference - bias) * content_width > line;
        const auto inset = static_cast<std::uint32_t>(static_cast<std::uint64_t>(v2::subtitle_contour_ordinary_search_inset_detector_pixels) *
          content_width / v2::subtitle_ocr_output_width);
        if (search[0] != content[0] && !right_biased) search[0] = std::min(pair.core[0], search[0] + inset);
        if (search[2] != content[2] && !left_biased) search[2] = std::max(pair.core[2], search[2] - std::min(inset, search[2]));
        const auto pad = ((pair.core[3] - pair.core[1]) * v2::subtitle_contour_ordinary_search_pad_height_numerator +
          v2::subtitle_contour_ordinary_search_pad_height_denominator - 1u) / v2::subtitle_contour_ordinary_search_pad_height_denominator;
        search[1] = std::max(search[1], pair.core[1] - std::min(pad, pair.core[1]));
      }
      std::array<mode_t, v2::subtitle_contour_palette_bin_count> bins {};
      std::uint32_t anchor_peak = 0;
      for (std::uint32_t sy = 0; sy < v2::subtitle_contour_palette_grid_rows; ++sy) {
        for (std::uint32_t sx = 0; sx < v2::subtitle_contour_palette_grid_columns; ++sx) {
          const auto grid_x = v2::subtitle_contour_palette_grid_columns, grid_y = v2::subtitle_contour_palette_grid_rows;
          const auto project = [](std::uint32_t low, std::uint32_t high, std::uint32_t origin,
                                 std::uint32_t sample, std::uint32_t grid, std::uint32_t source, std::uint32_t extent) {
            const std::int64_t numerator = ((static_cast<std::int64_t>(low) - origin) * (2 * grid) +
              (2ull * sample + 1) * (high - low)) * source;
            const auto denominator = static_cast<std::int64_t>(2 * grid) * extent;
            const auto floor = numerator / denominator - (numerator < 0 && numerator % denominator != 0);
            return floor;
          };
          const auto x = std::clamp<std::int64_t>(project(pair.core[0], pair.core[2], content[0], sx, grid_x, width, content_width), 0, width - 1);
          const auto y = std::clamp<std::int64_t>(project(pair.core[1], pair.core[3], content[1], sy, grid_y, source_height, content_height) - crop_top, 0, height - 1);
          const auto p = static_cast<std::size_t>(y) * width + x;
          float probability {};
          if (!colors[p].valid || edges[p] == 0 || !load_probability(p, probability) || probability <= v2::subtitle_ocr_active_probability_threshold) continue;
          const auto &rgb = colors[p].rgb;
          const auto bin = std::min(3u, static_cast<unsigned>(rgb[0] * 4.f)) * 16u +
            std::min(3u, static_cast<unsigned>(rgb[1] * 4.f)) * 4u + std::min(3u, static_cast<unsigned>(rgb[2] * 4.f));
          auto &mode = bins[bin]; mode.bin = bin;
          const auto current_peak = std::max({quantized[p][0], quantized[p][1], quantized[p][2]});
          // Row-major traversal retains the first grid index on equal max-channel Q16.
          if (!mode.count || current_peak > peak(mode)) mode.sum = quantized[p];
          ++mode.count;
          anchor_peak = std::max(anchor_peak, current_peak);
        }
      }
      std::vector<mode_t> eligible, modes;
      for (auto mode : bins) if (mode.count &&
          static_cast<std::uint64_t>(v2::subtitle_contour_palette_brightness_denominator) * peak(mode) >=
          static_cast<std::uint64_t>(v2::subtitle_contour_palette_brightness_numerator) * anchor_peak) {
        // The selected actual RGB is represented as a virtual sum/count for the existing
        // exact rational color and dark-outline predicates.
        for (auto &channel : mode.sum) channel *= mode.count;
        eligible.push_back(mode);
      }
      std::sort(eligible.begin(), eligible.end(), [&](const auto &a, const auto &b) {
        const auto left = static_cast<std::uint64_t>(peak(a)) * b.count, right = static_cast<std::uint64_t>(peak(b)) * a.count;
        if (left != right) return left > right;
        if (a.count != b.count) return a.count > b.count;
        return a.bin < b.bin;
      });
      for (const auto &mode : eligible) {
        const auto redundant = std::any_of(modes.begin(), modes.end(), [&](const auto &prior) {
          const auto mode_peak = std::max(1u, (peak(mode) + mode.count / 2u) / mode.count);
          const auto prior_peak = std::max(1u, (peak(prior) + prior.count / 2u) / prior.count);
          std::uint32_t maximum = 0;
          for (std::size_t c = 0; c < 3; ++c) {
            const auto mode_channel = (mode.sum[c] + mode.count / 2u) / mode.count;
            const auto prior_channel = (prior.sum[c] + prior.count / 2u) / prior.count;
            const auto a = static_cast<std::int64_t>((mode_channel * v2::subtitle_contour_palette_hue_channel_max + mode_peak / 2u) / mode_peak);
            const auto b = static_cast<std::int64_t>((prior_channel * v2::subtitle_contour_palette_hue_channel_max + prior_peak / 2u) / prior_peak);
            maximum = std::max(maximum, static_cast<std::uint32_t>(std::abs(a - b)));
          }
          return static_cast<std::uint64_t>(v2::subtitle_contour_palette_merge_hue_denominator) * maximum <=
            static_cast<std::uint64_t>(v2::subtitle_contour_palette_merge_hue_numerator) * v2::subtitle_contour_palette_hue_channel_max;
        });
        if (!redundant) modes.push_back(mode);
        if (modes.size() == v2::subtitle_contour_palette_max_modes) break;
      }
      if (modes.empty()) continue;
      const auto palette_match = [&](std::size_t p) {
        return colors[p].valid && std::any_of(modes.begin(), modes.end(), [&](const auto &mode) { return matches(quantized[p], mode); });
      };
      const auto &brightest = modes.front();
      const auto short_run = [&](std::size_t p, int axis) {
        std::uint64_t length = 1;
        const auto limit = static_cast<std::uint64_t>(v2::subtitle_contour_long_line_core_height_numerator) *
          (pair.core[3] - pair.core[1]) * source_height;
        const auto scale = static_cast<std::uint64_t>(content_height) * v2::subtitle_contour_long_line_core_height_denominator;
        const int x = p % width, y = p / width;
        for (int sign : {-1, 1}) {
          for (int step = 1; length * scale <= limit; ++step) {
            const int qx = x + (axis == 0 ? sign * step : 0), qy = y + (axis == 1 ? sign * step : 0);
            if (!inside(qx, qy)) break;
            const auto q = static_cast<std::size_t>(qy) * width + qx;
            if (!colors[q].valid || distance(colors[p].rgb, colors[q].rgb) > v2::subtitle_contour_spatial_edge_rgb_distance) break;
            ++length;
          }
        }
        return length * scale <= limit;
      };
      for (std::size_t p = 0; p < pixels; ++p) {
        float probability {};
        if (result[p] || !colors[p].valid || edges[p] == 0 || !covered(p, search) || !load_probability(p, probability)) continue;
        if (palette_match(p) && (((edges[p] & 1u) && short_run(p, 1)) || ((edges[p] & 2u) && short_run(p, 0)))) { result[p] = 1; continue; }
        const auto &q = quantized[p];
        const auto high = std::max({q[0], q[1], q[2]}), low = std::min({q[0], q[1], q[2]});
        if (static_cast<std::uint64_t>(v2::subtitle_contour_outline_value_denominator) * high * brightest.count >
              static_cast<std::uint64_t>(v2::subtitle_contour_outline_value_numerator) * peak(brightest) ||
            static_cast<std::uint64_t>(v2::subtitle_contour_outline_chroma_denominator) * (high - low) >
              static_cast<std::uint64_t>(v2::subtitle_contour_outline_chroma_numerator) * channel_max) continue;
        const int x = p % width, y = p / width;
        constexpr int outline = v2::subtitle_contour_outline_probe_radius_source_pixels;
        for (int dy : {-outline, 0, outline}) for (int dx : {-outline, 0, outline}) {
          if ((dx == 0 && dy == 0) || !inside(x + dx, y + dy)) continue;
          const auto neighbor = static_cast<std::size_t>(y + dy) * width + x + dx;
          if (palette_match(neighbor) && distance(colors[p].rgb, colors[neighbor].rgb) > v2::subtitle_contour_spatial_edge_rgb_distance) result[p] = 1;
        }
      }
    }
    return result;
  }
  // Independent immutable-byte replay of native dense membership. The caller authenticates
  // the current OCR/source tuple and supplies zero support for empty/invalid current authority.
  // This verifier never supplies a live shader resource or a stereo-renderer implementation.
  template<class LoadColor>
  std::vector<std::uint8_t> dense_matte(
    std::uint32_t width, std::uint32_t height, std::span<const std::uint32_t> support,
    LoadColor load_color
  ) {
    if (!width || !height || width > v2::subtitle_contour_source_max_extent ||
        height > v2::subtitle_contour_source_max_extent) return {};
    const auto pixels = static_cast<std::size_t>(width) * height;
    if (support.size() != pixels) return {};
    std::vector<sample_t> colors(pixels);
    std::vector<std::uint8_t> current(pixels), next(pixels), result(pixels);
    const auto peak = [](const rgb_t &rgb) { return std::max({rgb[0], rgb[1], rgb[2]}); };
    const auto inside = [&](int x, int y) {
      return x >= 0 && y >= 0 && x < static_cast<int>(width) && y < static_cast<int>(height);
    };
    for (std::size_t p = 0; p < pixels; ++p) {
      colors[p] = load_color(p);
      for (const auto channel : colors[p].rgb)
        colors[p].valid = colors[p].valid && std::isfinite(channel) && channel >= 0.f && channel <= 1.f;
      current[p] = colors[p].valid && support[p] == 1u ? 1u : 0u;
    }
    for (std::uint32_t step = 0; step < v2::subtitle_glyph_matte_steps; ++step) {
      next = current;
      for (std::size_t p = 0; p < pixels; ++p) {
        if (current[p] || !colors[p].valid) continue;
        const int x = static_cast<int>(p % width), y = static_cast<int>(p / width);
        const bool bright = peak(colors[p].rgb) >= v2::subtitle_glyph_matte_brightness_split;
        for (int dy = -1; dy <= 1 && !next[p]; ++dy) for (int dx = -1; dx <= 1; ++dx) {
          if ((!dx && !dy) || !inside(x + dx, y + dy)) continue;
          const auto q = static_cast<std::size_t>(y + dy) * width + x + dx;
          if (!current[q] || !colors[q].valid ||
              bright != (peak(colors[q].rgb) >= v2::subtitle_glyph_matte_brightness_split)) continue;
          const auto &a = colors[p].rgb;
          const auto &b = colors[q].rgb;
          const auto distance = std::max({std::abs(a[0] - b[0]), std::abs(a[1] - b[1]), std::abs(a[2] - b[2])});
          if (distance <= v2::subtitle_glyph_matte_rgb_distance) next[p] = 1u;
        }
      }
      current.swap(next);
    }
    constexpr int radius = v2::subtitle_glyph_matte_collar_radius_source_pixels;
    for (std::size_t p = 0; p < pixels; ++p) {
      if (!colors[p].valid) continue;
      const int x = static_cast<int>(p % width), y = static_cast<int>(p / width);
      for (int dy = -radius; dy <= radius && !result[p]; ++dy) for (int dx = -radius; dx <= radius; ++dx) {
        if (dx * dx + dy * dy > static_cast<int>(v2::subtitle_glyph_matte_collar_radius_squared) ||
            !inside(x + dx, y + dy)) continue;
        const auto q = static_cast<std::size_t>(y + dy) * width + x + dx;
        if (current[q]) result[p] = 1u;
      }
    }
    return result;
  }
}  // namespace models::host_sbs_subtitle_glyph
