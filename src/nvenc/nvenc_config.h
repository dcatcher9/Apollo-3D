/**
 * @file src/nvenc/nvenc_config.h
 * @brief Declarations for NVENC encoder configuration.
 */
#pragma once

#ifndef SUNSHINE_NVENC_CONFIG_H
  #define SUNSHINE_NVENC_CONFIG_H

  #include <cstdint>
  #include <limits>
  #include <optional>

namespace nvenc {

  struct nvenc_rate_control_values_t {
    std::uint32_t average_bitrate = 0;
    std::optional<std::uint32_t> vbv_buffer_size;
  };

  /**
   * Derive the rate-control fields shared by initial creation and bitrate-only reconfiguration.
   * Keeping this calculation in one overflow-checked helper prevents a live change from using a
   * different VBV contract than a fresh session at the same bitrate and cadence.
   */
  constexpr std::optional<nvenc_rate_control_values_t> nvenc_rate_control_values(
    int bitrate_kbps,
    int framerate,
    int vbv_percentage_increase,
    bool custom_vbv_supported
  ) noexcept {
    if (bitrate_kbps <= 0 || framerate <= 0) {
      return std::nullopt;
    }

    constexpr auto max_u32 = std::numeric_limits<std::uint32_t>::max();
    const auto average_bitrate = static_cast<std::uint64_t>(bitrate_kbps) * 1000u;
    if (average_bitrate > max_u32) {
      return std::nullopt;
    }

    nvenc_rate_control_values_t values {
      static_cast<std::uint32_t>(average_bitrate),
      std::nullopt,
    };
    if (!custom_vbv_supported) {
      return values;
    }

    auto vbv_buffer_size = average_bitrate / static_cast<std::uint64_t>(framerate);
    if (vbv_percentage_increase > 0) {
      vbv_buffer_size +=
        vbv_buffer_size * static_cast<std::uint64_t>(vbv_percentage_increase) / 100u;
    }
    values.vbv_buffer_size = static_cast<std::uint32_t>(
      vbv_buffer_size > max_u32 ? max_u32 : vbv_buffer_size
    );
    return values;
  }

  enum class nvenc_two_pass {
    disabled,  ///< Single pass, the fastest and no extra vram
    quarter_resolution,  ///< Larger motion vectors being caught, faster and uses less extra vram
    full_resolution,  ///< Better overall statistics, slower and uses more extra vram
  };

  /**
   * @brief NVENC encoder configuration.
   */
  struct nvenc_config {
    // Quality preset from 1 to 7, higher is slower
    int quality_preset = 1;

    // Use optional preliminary pass for better motion vectors, bitrate distribution and stricter VBV(HRD), uses CUDA cores
    nvenc_two_pass two_pass = nvenc_two_pass::quarter_resolution;

    // Percentage increase of VBV/HRD from the default single frame, allows low-latency variable bitrate
    int vbv_percentage_increase = 100;

    // Allocate more bitrate to flat regions since they're visually more perceptible, uses CUDA cores
    bool adaptive_quantization = false;

    // Replace HEVC P pictures with past-reference-only B pictures for better low-latency compression
    bool hevc_unidirectional_b = true;
  };

  constexpr bool should_enable_hevc_unidirectional_b(
    const nvenc_config &config,
    int video_format,
    bool supported
  ) {
    constexpr int hevc_video_format = 1;
    return config.hevc_unidirectional_b &&
           video_format == hevc_video_format &&
           supported;
  }

  constexpr bool should_force_split_frame_encoding(
    bool packed_host_sbs,
    int video_format,
    int encode_width,
    int encoder_engine_count
  ) {
    const bool split_capable_codec = video_format == 1 || video_format == 2;
    return packed_host_sbs &&
           split_capable_codec &&
           encode_width > 4096 &&
           encoder_engine_count > 1;
  }

}  // namespace nvenc

#endif  // SUNSHINE_NVENC_CONFIG_H
