// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <span>
#include <string>

namespace platf::game3d_debug::preview {
  struct image_view {
    std::string kind;
    std::uint32_t width, height, dxgi_format;
    std::span<const std::uint8_t> bytes;
  };

  // CPU-only; call on the dump publication worker. Native artifacts are never modified.
  // Images are tightly packed, full-allocation, top-to-bottom native DXGI bytes.
  nlohmann::json generate(const std::filesystem::path &directory, const nlohmann::json &manifest, std::span<const image_view> images);

  namespace detail {
    using rgb = std::array<std::uint8_t, 3>;
    std::uint32_t bytes_per_pixel(std::uint32_t format);
    std::array<float, 4> sample(const image_view &image, std::uint32_t x, std::uint32_t y);
    rgb display_color(std::array<float, 4> value, unsigned color_space);
    rgb scalar_color(float value, float low, float high);
    rgb signed_color(float value, float limit);
    std::array<std::uint32_t, 2> aligned_depth_pixel(std::uint32_t x, std::uint32_t y, std::uint32_t color_width, std::uint32_t color_height, std::uint32_t depth_width, std::uint32_t depth_height, const std::array<float, 4> &rect, const std::array<float, 2> &jitter);
    bool safe_artifact_filename(const std::string &name);
  }  // namespace detail
}  // namespace platf::game3d_debug::preview
