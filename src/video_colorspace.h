/**
 * @file src/video_colorspace.h
 * @brief Declarations for colorspace functions.
 */
#pragma once

namespace video {

  enum class colorspace_e {
    rec601,  ///< Rec. 601
    rec709,  ///< Rec. 709
    bt2020sdr,  ///< Rec. 2020 SDR
    bt2020,  ///< Rec. 2020 HDR
  };

  struct sunshine_colorspace_t {
    colorspace_e colorspace;
    bool full_range;
    unsigned bit_depth;
  };

  bool colorspace_is_hdr(const sunshine_colorspace_t &colorspace);

  /**
   * Bounded polynomial composite of the sRGB EOTF and BT.709 OETF used by the optimized BGRA8
   * shader path. This scalar reference is exposed for dense numerical contract tests.
   */
  float srgb_code_to_bt709_code(float value);

  /**
   * Whether an actual input frame needs the scRGB HDR-to-SDR tone map.
   * Encoded BGRA capture remains SDR even when the physical display has HDR enabled.
   */
  bool hdr_to_sdr_tonemap_required(
    bool target_is_hdr,
    bool source_display_is_hdr,
    bool input_is_linear
  ) noexcept;

  /**
   * Whether an SBS intermediate must retain FP16 precision while capture format discovery
   * is still pending. Transfer state is tracked separately from the actual frame format.
   */
  bool sbs_intermediate_requires_fp16(
    bool capture_is_fp16,
    bool capture_format_is_unknown,
    bool source_display_is_hdr,
    bool output_is_10bit
  ) noexcept;

  // Declared in video.h
  struct config_t;

  sunshine_colorspace_t colorspace_from_client_config(const config_t &config, bool hdr_display);

  struct alignas(16) color_t {
    float color_vec_y[4];
    float color_vec_u[4];
    float color_vec_v[4];
    float range_y[2];
    float range_uv[2];
  };

  /**
   * @brief Get static RGB->YUV color conversion matrix.
   *        This matrix expects RGB input in UNORM (0.0 to 1.0) range and doesn't perform any
   *        gamut mapping or gamma correction.
   * @param colorspace Targeted YUV colorspace.
   * @param unorm_output Whether the matrix should produce output in UNORM or UINT range.
   * @return `const color_t*` that contains RGB->YUV transformation vectors.
   *         Components `range_y` and `range_uv` are there for backwards compatibility
   *         and can be ignored in the computation.
   */
  const color_t *color_vectors_from_colorspace(const sunshine_colorspace_t &colorspace, bool unorm_output);
}  // namespace video
