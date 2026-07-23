/**
 * @file src/video_colorspace.cpp
 * @brief Definitions for colorspace functions.
 */
// this include
#include "video_colorspace.h"

// standard includes
#include <algorithm>
#include <cmath>

// local includes
#include "logging.h"
#include "video.h"

namespace video {

  bool colorspace_is_hdr(const sunshine_colorspace_t &colorspace) {
    return colorspace.colorspace == colorspace_e::bt2020;
  }

  float srgb_code_to_bt709_code(float value) {
    const float x = std::clamp(value, 0.0f, 1.0f);
    if (x <= 0.04045f) {
      return x * (4.5f / 12.92f);
    }

    if (x < 0.142825681303039f) {
      const float t = (2.0f * x - 0.183275681303039f) / 0.102375681303039f;
      return (((-0.0000203208382756466f * t + 0.000379028747322516f) * t +
               0.00808284888652290f) *
                t +
              0.0330765198090362f) *
               t +
             0.0394817883386249f;
    }

    const float t = (2.0f * x - 1.142825681303039f) / 0.857174318696961f;
    return std::clamp(
      (((0.00127867013357397f * t - 0.00322804757744502f) * t +
        0.0124427001451592f) *
         t +
       0.462640801365181f) *
          t +
        0.526902601731455f,
      0.0f,
      1.0f
    );
  }

  bool hdr_to_sdr_tonemap_required(
    bool target_is_hdr,
    bool source_display_is_hdr,
    bool input_is_linear
  ) noexcept {
    return !target_is_hdr && source_display_is_hdr && input_is_linear;
  }

  bool sbs_intermediate_requires_fp16(
    bool capture_is_fp16,
    bool capture_format_is_unknown,
    bool source_display_is_hdr,
    bool output_is_10bit
  ) noexcept {
    return capture_is_fp16 ||
           (capture_format_is_unknown && (source_display_is_hdr || output_is_10bit));
  }

  sunshine_colorspace_t colorspace_from_client_config(const config_t &config, bool hdr_display) {
    sunshine_colorspace_t colorspace;

    /* See video::config_t declaration for details */

    BOOST_LOG(info) << "Client dynamicRange: " << config.dynamicRange << ", Display is HDR: " << hdr_display;

    if (config.dynamicRange > 0 && hdr_display) {
      // Rec. 2020 with ST 2084 perceptual quantizer
      colorspace.colorspace = colorspace_e::bt2020;
    } else {
      switch (config.encoderCscMode >> 1) {
        case 0:
          // Rec. 601
          colorspace.colorspace = colorspace_e::rec601;
          break;

        case 1:
          // Rec. 709
          colorspace.colorspace = colorspace_e::rec709;
          break;

        case 2:
          // Rec. 2020
          colorspace.colorspace = colorspace_e::bt2020sdr;
          break;

        default:
          BOOST_LOG(error) << "Unknown video colorspace in csc, falling back to Rec. 709";
          colorspace.colorspace = colorspace_e::rec709;
          break;
      }
    }

    colorspace.full_range = (config.encoderCscMode & 0x1);

    switch (config.dynamicRange) {
      case 0:
        colorspace.bit_depth = 8;
        break;

      case 1:
        colorspace.bit_depth = 10;
        break;

      default:
        BOOST_LOG(error) << "Unknown dynamicRange value, falling back to 10-bit color depth";
        colorspace.bit_depth = 10;
        break;
    }

    if (colorspace.colorspace == colorspace_e::bt2020sdr && colorspace.bit_depth != 10) {
      BOOST_LOG(error) << "BT.2020 SDR colorspace expects 10-bit color depth, falling back to Rec. 709";
      colorspace.colorspace = colorspace_e::rec709;
    }

    return colorspace;
  }

  const color_t *color_vectors_from_colorspace(const sunshine_colorspace_t &colorspace, bool unorm_output) {
    constexpr auto generate_color_vectors = [](const sunshine_colorspace_t &colorspace, bool unorm_output) -> color_t {
      // "Table 4 – Interpretation of matrix coefficients (MatrixCoefficients) value" section of ITU-T H.273
      double Kr, Kb;
      switch (colorspace.colorspace) {
        case colorspace_e::rec601:
          Kr = 0.299;
          Kb = 0.114;
          break;
        case colorspace_e::rec709:
        default:
          Kr = 0.2126;
          Kb = 0.0722;
          break;
        case colorspace_e::bt2020:
        case colorspace_e::bt2020sdr:
          Kr = 0.2627;
          Kb = 0.0593;
          break;
      }
      double Kg = 1.0 - Kr - Kb;

      double y_mult, y_add;
      double uv_mult, uv_add;

      // "8.3 Matrix coefficients" section of ITU-T H.273
      if (colorspace.full_range) {
        y_mult = (1 << colorspace.bit_depth) - 1;
        y_add = 0;
        uv_mult = (1 << colorspace.bit_depth) - 1;
        uv_add = (1 << (colorspace.bit_depth - 1));
      } else {
        y_mult = (1 << (colorspace.bit_depth - 8)) * 219;
        y_add = (1 << (colorspace.bit_depth - 8)) * 16;
        uv_mult = (1 << (colorspace.bit_depth - 8)) * 224;
        uv_add = (1 << (colorspace.bit_depth - 8)) * 128;
      }

      if (unorm_output) {
        const double unorm_range = (1 << colorspace.bit_depth) - 1;
        y_mult /= unorm_range;
        y_add /= unorm_range;
        uv_mult /= unorm_range;
        uv_add /= unorm_range;
      } else {
        // For rounding when the shader converts the result to an integer.
        y_add += 0.5;
        uv_add += 0.5;
      }

      color_t color_vectors;

      color_vectors.color_vec_y[0] = Kr * y_mult;
      color_vectors.color_vec_y[1] = Kg * y_mult;
      color_vectors.color_vec_y[2] = Kb * y_mult;
      color_vectors.color_vec_y[3] = y_add;

      color_vectors.color_vec_u[0] = -0.5 * Kr / (1.0 - Kb) * uv_mult;
      color_vectors.color_vec_u[1] = -0.5 * Kg / (1.0 - Kb) * uv_mult;
      color_vectors.color_vec_u[2] = 0.5 * uv_mult;
      color_vectors.color_vec_u[3] = uv_add;

      color_vectors.color_vec_v[0] = 0.5 * uv_mult;
      color_vectors.color_vec_v[1] = -0.5 * Kg / (1.0 - Kr) * uv_mult;
      color_vectors.color_vec_v[2] = -0.5 * Kb / (1.0 - Kr) * uv_mult;
      color_vectors.color_vec_v[3] = uv_add;

      // Unused
      color_vectors.range_y[0] = 1;
      color_vectors.range_y[1] = 0;
      color_vectors.range_uv[0] = 1;
      color_vectors.range_uv[1] = 0;

      return color_vectors;
    };

    static constexpr color_t colors[] = {
      generate_color_vectors({colorspace_e::rec601, false, 8}, false),
      generate_color_vectors({colorspace_e::rec601, true, 8}, false),
      generate_color_vectors({colorspace_e::rec601, false, 10}, false),
      generate_color_vectors({colorspace_e::rec601, true, 10}, false),
      generate_color_vectors({colorspace_e::rec709, false, 8}, false),
      generate_color_vectors({colorspace_e::rec709, true, 8}, false),
      generate_color_vectors({colorspace_e::rec709, false, 10}, false),
      generate_color_vectors({colorspace_e::rec709, true, 10}, false),
      generate_color_vectors({colorspace_e::bt2020, false, 8}, false),
      generate_color_vectors({colorspace_e::bt2020, true, 8}, false),
      generate_color_vectors({colorspace_e::bt2020, false, 10}, false),
      generate_color_vectors({colorspace_e::bt2020, true, 10}, false),

      generate_color_vectors({colorspace_e::rec601, false, 8}, true),
      generate_color_vectors({colorspace_e::rec601, true, 8}, true),
      generate_color_vectors({colorspace_e::rec601, false, 10}, true),
      generate_color_vectors({colorspace_e::rec601, true, 10}, true),
      generate_color_vectors({colorspace_e::rec709, false, 8}, true),
      generate_color_vectors({colorspace_e::rec709, true, 8}, true),
      generate_color_vectors({colorspace_e::rec709, false, 10}, true),
      generate_color_vectors({colorspace_e::rec709, true, 10}, true),
      generate_color_vectors({colorspace_e::bt2020, false, 8}, true),
      generate_color_vectors({colorspace_e::bt2020, true, 8}, true),
      generate_color_vectors({colorspace_e::bt2020, false, 10}, true),
      generate_color_vectors({colorspace_e::bt2020, true, 10}, true),
    };

    const color_t *result = nullptr;

    switch (colorspace.colorspace) {
      case colorspace_e::rec601:
        result = &colors[0];
        break;
      case colorspace_e::rec709:
      default:
        result = &colors[4];
        break;
      case colorspace_e::bt2020:
      case colorspace_e::bt2020sdr:
        result = &colors[8];
        break;
    }

    if (colorspace.bit_depth == 10) {
      result += 2;
    }
    if (colorspace.full_range) {
      result += 1;
    }
    if (unorm_output) {
      result += 12;
    }

    return result;
  }
}  // namespace video
