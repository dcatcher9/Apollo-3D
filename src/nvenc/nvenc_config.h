/**
 * @file src/nvenc/nvenc_config.h
 * @brief Declarations for NVENC encoder configuration.
 */
#pragma once

namespace nvenc {

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
