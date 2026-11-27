/**
 * @file src/platform/windows/sbs_debug_dump_package.cpp
 * @brief Dump 3D package writer: JSON documents, previews and the atomic package folder.
 */
#include "sbs_debug_dump_package.h"

// standard includes
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

// platform includes
#include <windows.h>
#include <wrl/client.h>

// lib includes
#include <nlohmann/json.hpp>
#include <zlib.h>

// local includes
#include "src/generated/sbs_adaptive_state_contract.h"
#include "src/depth_coordinate_v2.h"
#include "src/host_sbs_resolution.h"
#include "src/host_sbs_contour_history_readback.h"
#include "src/host_sbs_subtitle_plane.h"
#include "src/host_sbs_subtitle_glyph.h"
#include "src/host_sbs_gpu_trace.h"
#include "src/host_sbs_shader_cache.h"
#include "src/logging.h"
#include "src/model_manager.h"
#include "src/prod_zipdepth_convex2x.h"
#include "src/video_depth_estimator.h"
#include "sbs_debug_dump_validate.h"

namespace platf::sbs_debug {

  using namespace std::literals;

  namespace {

    using namespace models::depth_coordinate_v2;
    using detail::actual_draw_artifacts;
    using detail::actual_draw_is_texture;
    using detail::actual_draw_storage_index;
    using detail::captured_dump_job;
    using detail::depth_input_region_error;
    using detail::dump_publish_result;
    using detail::gpu_trace_shader_identity_matches_contract;
    using detail::normalization_state;
    using detail::parallax_v2_shader_identity_matches_contract;
    using detail::texture_snapshot;

    nlohmann::json parallax_v2_shader_identity_json(
      const models::parallax_v2_shader_provenance_t &identity
    ) {
      return {
        {"source_closure_schema", identity.source_closure_schema},
        {"source_compile_flags", identity.source_compile_flags},
        {"source_macro_count", identity.source_macro_count},
        {"source_closure_sha256", identity.source_closure_sha256},
      };
    }

    nlohmann::json parallax_v2_coordinate_binding(
      const models::parallax_v2_shader_provenance_t &identity,
      const char *word_count_key,
      const std::size_t word_count
    ) {
      using namespace models::depth_coordinate_v2;
      auto binding = parallax_v2_shader_identity_json(identity);
      binding["schema"] = contract_schema;
      binding["tag"] = contract_tag;
      binding[word_count_key] = word_count;
      return binding;
    }

    nlohmann::json subtitle_shader_contract_json(
      const models::parallax_v2_shader_provenance_t &identity,
      nlohmann::json source_specs
    ) {
      using namespace models::depth_coordinate_v2;
      auto contract = parallax_v2_shader_identity_json(identity);
      contract["depth_coordinate_v2_schema"] = contract_schema;
      contract["depth_coordinate_v2_tag"] = contract_tag;
      contract["source_specs"] = std::move(source_specs);
      return contract;
    }

    nlohmann::json subtitle_ocr_producer_contract_json(
      const models::parallax_v2_shader_provenance_t &identity
    ) {
      using namespace models::depth_coordinate_v2;
      return {
        {"contract_schema", subtitle_ocr_contract_schema},
        {"record_schema", subtitle_ocr_record_schema},
        {"record_tag", subtitle_ocr_record_tag},
        {"record_word_count", subtitle_ocr_record_word_count},
        {"current_prefix_word_count", subtitle_ocr_current_prefix_word_count},
        {"raw_box_capacity", subtitle_ocr_raw_box_capacity},
        {"final_box_capacity", subtitle_ocr_final_box_capacity},
        {"classifier", nlohmann::json::parse(subtitle_ocr_classifier_policy_json)},
        {"model", {
          {"name", std::string {subtitle_ocr_model_name}},
          {"asset_path", std::string {subtitle_ocr_asset_path}},
          {"artifact_onnx_sha256", std::string {subtitle_ocr_artifact_onnx_sha256}},
          {"source_url", std::string {subtitle_ocr_source_url}},
          {"source_onnx_sha256", std::string {subtitle_ocr_source_onnx_sha256}},
          {"conversion_tool", std::string {subtitle_ocr_conversion_tool}},
          {"conversion_version", std::string {subtitle_ocr_conversion_version}},
          {"conversion_recipe", std::string {subtitle_ocr_conversion_recipe}},
          {"conversion_calibration_profile", std::string {
            subtitle_ocr_conversion_calibration_profile
          }},
          {"engine_recipe", std::string {subtitle_ocr_engine_recipe}},
          {"preprocess_profile", std::string {subtitle_ocr_preprocess_profile}},
          {"source_crop", std::string {subtitle_ocr_source_crop}},
          {"input", {
            {"name", std::string {subtitle_ocr_input_name}},
            {"dtype", std::string {subtitle_ocr_input_dtype}},
            {"layout", std::string {subtitle_ocr_input_layout}},
            {"shape", {
              subtitle_ocr_input_n,
              subtitle_ocr_input_c,
              subtitle_ocr_input_height,
              subtitle_ocr_input_width,
            }},
            {"channels", {
              std::string {subtitle_ocr_input_channels[0]},
              std::string {subtitle_ocr_input_channels[1]},
              std::string {subtitle_ocr_input_channels[2]},
            }},
            {"imagenet_mean", {
              subtitle_ocr_imagenet_mean[0],
              subtitle_ocr_imagenet_mean[1],
              subtitle_ocr_imagenet_mean[2],
            }},
            {"imagenet_std", {
              subtitle_ocr_imagenet_std[0],
              subtitle_ocr_imagenet_std[1],
              subtitle_ocr_imagenet_std[2],
            }},
          }},
          {"output", {
            {"name", std::string {subtitle_ocr_output_name}},
            {"dtype", std::string {subtitle_ocr_output_dtype}},
            {"layout", std::string {subtitle_ocr_output_layout}},
            {"shape", {
              subtitle_ocr_output_n,
              subtitle_ocr_output_c,
              subtitle_ocr_output_height,
              subtitle_ocr_output_width,
            }},
          }},
        }},
        {"shader_contract", subtitle_shader_contract_json(identity, {
          {
            {"source_file", "host_sbs_ocr_preprocess_cs.hlsl"},
            {"entrypoint", "main"},
            {"target", "cs_5_0"},
          },
          {
            {"source_file", "host_sbs_ocr_boxes_cs.hlsl"},
            {"entrypoint", "cells_main"},
            {"target", "cs_5_0"},
          },
          {
            {"source_file", "host_sbs_ocr_boxes_cs.hlsl"},
            {"entrypoint", "resolve_main"},
            {"target", "cs_5_0"},
          },
        })},
      };
    }

    nlohmann::json subtitle_plane_resolver_contract_json(
      const models::parallax_v2_shader_provenance_t &identity
    ) {
      return {
        {"state_schema", subtitle_plane_state_schema},
        {"state_tag", subtitle_plane_state_tag},
        {"state_word_count", subtitle_plane_state_word_count},
        {"cover_capacity", subtitle_plane_cover_capacity},
        {"classifier", nlohmann::json::parse(subtitle_ocr_classifier_policy_json)},
        {"adaptive_ui", {
          {"policy_id", adaptive_policy_id}, {"state_offset", subtitle_plane_adaptive_offset},
          {"state_word_count", subtitle_plane_adaptive_word_count},
          {"clock", "source-observation-microseconds"},
          {"geometry_authority", subtitle_geometry_authority},
          {"forward_policy", subtitle_ui_plane_policy},
          {"known_flags", subtitle_plane_adaptive_known_flags},
          {"probe", "union-of-current-authorized-covers-in-content"},
          {"clearance_per_eye_source_pixels", 2u},
          {"enter_covered_fraction", 0.20f}, {"enter_content_fraction", 0.02f},
          {"release_covered_fraction", 0.0f}, {"release_content_fraction", 0.0f},
          {"approach_delay_ms", 0u}, {"retreat_delay_ms", 0u},
          {"approach_source_u_per_second", 0.0f},
          {"retreat_source_u_per_second", 0.0f},
          {"retreat_time_constant_ms", subtitle_plane_retreat_time_constant_ms},
          {"smoothing_time_constant_ms", subtitle_plane_smoothing_time_constant_ms},
          {"slew_per_eye_source_pixels_per_second", subtitle_plane_slew_source_pixels_per_second},
          {"rate_tick_cap_us", subtitle_plane_max_tick_gap_ms * 1000u},
          {"first_visible_policy", "current-goal-immediate"},
          {"current_clearance_is_presentation_floor", false},
          {"geometric_pin", "full-on-current-contour-authority"},
          {"host_model_policy", {
            {"mode", 3u}, {"maximum_plane_source_u", direct_container_limit},
            {"candidate_step_per_eye_source_pixels", 4u},
            {"candidate_step_scope", "serialized-coarse-diagnostic-index"},
            {"last_candidate", "exact-representation-cap"},
            {"maximum_candidate_index", "ceil-analysis-source-width-over-100"},
            {"core_range_per_eye_source_pixels", 0u},
            {"retreat_requires_applied_target", false},
            {"capped_conflict", "derived-from-top-candidate-counts"},
          }},
        }},
        {"current_cover_policy", {
          {"selection", "all-valid-current-ocr8-core-cover-pairs"},
          {"capacity", subtitle_plane_cover_capacity},
          {"geometry_source", "exact-current-ocr8-record-final-payload"},
          {"history_authority", "diagnostic-fixed-rgb-duration-positive-count-only-for-duplicate-admission"},
          {"box_identity_resets_history", false},
          {"held_reuse", "byte-identical-latest-published-ocr8-and-uip2-tuple"},
        }},
        {"target_policy", {
          {"units", "one-eye-analysis-source-U"},
          {"selection", "authenticated-current-cover-conflict-histogram"},
          {"forward_policy", subtitle_ui_plane_policy},
          {"candidate_step_per_eye_source_pixels", 1u},
          {"coarse_diagnostic_step_per_eye_source_pixels", 4u},
          {"clearance_per_eye_source_pixels", 2u},
          {"approach_dwell_us", 0u},
          {"retreat_dwell_us", 0u},
          {"approach_rate_source_u_per_second", 0.0f},
          {"retreat_rate_source_u_per_second", 0.0f},
          {"retreat_time_constant_ms", subtitle_plane_retreat_time_constant_ms},
          {"smoothing_time_constant_ms", subtitle_plane_smoothing_time_constant_ms},
          {"slew_per_eye_source_pixels_per_second", subtitle_plane_slew_source_pixels_per_second},
          {"current_clearance_is_presentation_floor", false},
          {"rate_tick_cap_us", subtitle_plane_max_tick_gap_ms * 1000u},
          {"current_cover_pin", "current-evidence-native-support-rectangle-slope-budget"},
          {"collar", "minimum-ui-depth-influence-within-current-support-slope-feasible-interval-before-shared-conditioning"},
          {"representation_limit", "direct-parallax-container"},
        }},
        {"contour_policy", nlohmann::json::parse(subtitle_contour_policy_json)},
        {"shader_contract", subtitle_shader_contract_json(identity, {
          {
            {"source_file", "host_sbs_subtitle_locator_cs.hlsl"},
            {"entrypoint", "resolve_main"},
            {"target", "cs_5_0"},
          },
          {
            {"source_file", "host_sbs_subtitle_locator_cs.hlsl"},
            {"entrypoint", "condition_main"},
            {"target", "cs_5_0"},
          },
          {
            {"source_file", "host_sbs_subtitle_contour_cs.hlsl"},
            {"entrypoint", "shared_palette_main"},
            {"target", "cs_5_0"},
          },
          {
            {"source_file", "host_sbs_subtitle_contour_cs.hlsl"},
            {"entrypoint", "temporal_main"},
            {"target", "cs_5_0"},
          },
          {
            {"source_file", "host_sbs_subtitle_contour_cs.hlsl"},
            {"entrypoint", "dense_main"},
            {"target", "cs_5_0"},
          },
          {
            {"source_file", "host_sbs_subtitle_contour_cs.hlsl"},
            {"entrypoint", "project_main"},
            {"target", "cs_5_0"},
          },
          {
            {"source_file", "host_sbs_subtitle_contour_cs.hlsl"},
            {"entrypoint", "history_main"},
            {"target", "cs_5_0"},
          },
          {
            {"source_file", "host_sbs_subtitle_contour_cs.hlsl"},
            {"entrypoint", "history_identity_main"},
            {"target", "cs_5_0"},
          },
          {
            {"source_file", "host_sbs_subtitle_contour_distance_cs.hlsl"},
            {"entrypoint", "horizontal_main"},
            {"target", "cs_5_0"},
          },
          {
            {"source_file", "host_sbs_subtitle_contour_distance_cs.hlsl"},
            {"entrypoint", "vertical_main"},
            {"target", "cs_5_0"},
          },
          {
            {"source_file", "host_sbs_subtitle_contour_cs.hlsl"},
            {"entrypoint", "snapshot_main"},
            {"target", "cs_5_0"},
          },
          {
            {"source_file", "depth_coordinate_v2_vertical_limit_cs.hlsl"},
            {"entrypoint", "subtitle_main"},
            {"target", "cs_5_0"},
          },
          {
            {"source_file", "depth_coordinate_v2_limit_cs.hlsl"},
            {"entrypoint", "subtitle_main"},
            {"target", "cs_5_0"},
          },
        })},
      };
    }

    bool subtitle_target_is_representable(const float target) {
      return std::isfinite(target) && target >= 0.0f && target <= direct_container_limit;
    }

    const char *normalization_frame_state_name(const float value) {
      if (value < 0.5f) {
        return "invalid-held";
      }
      if (value < 1.5f) {
        return "valid-with-history";
      }
      return "first-valid";
    }

    struct scalar_stats {
      std::size_t finite_count = 0;
      float minimum = std::numeric_limits<float>::quiet_NaN();
      float maximum = std::numeric_limits<float>::quiet_NaN();
    };

    struct raw_depth_dump_stats: scalar_stats {
      std::uint32_t width = 0;
      std::uint32_t height = 0;
    };

    struct warp_map_dump_stats {
      std::uint32_t width = 0;
      std::uint32_t height = 0;
      std::uint32_t eye_width = 0;
      std::uint32_t eye_height = 0;
      float content_scale_x = 1.0f;
      float content_scale_y = 1.0f;
    };

    std::string format_name(const DXGI_FORMAT format) {
      switch (format) {
        case DXGI_FORMAT_R32_FLOAT:
          return "DXGI_FORMAT_R32_FLOAT";
        case DXGI_FORMAT_R32_UINT:
          return "DXGI_FORMAT_R32_UINT";
        case DXGI_FORMAT_B8G8R8A8_UNORM:
          return "DXGI_FORMAT_B8G8R8A8_UNORM";
        case DXGI_FORMAT_B8G8R8X8_UNORM:
          return "DXGI_FORMAT_B8G8R8X8_UNORM";
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
          return "DXGI_FORMAT_B8G8R8A8_UNORM_SRGB";
        case DXGI_FORMAT_R8G8B8A8_UNORM:
          return "DXGI_FORMAT_R8G8B8A8_UNORM";
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
          return "DXGI_FORMAT_R8G8B8A8_UNORM_SRGB";
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
          return "DXGI_FORMAT_R16G16B16A16_FLOAT";
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
          return "DXGI_FORMAT_R32G32B32A32_FLOAT";
        default:
          return "DXGI_FORMAT_UNKNOWN_" + std::to_string(static_cast<unsigned>(format));
      }
    }

    nlohmann::json texture_description(const texture_snapshot &snapshot) {
      return {
        {"width", snapshot.desc.Width},
        {"height", snapshot.desc.Height},
        {"format", format_name(snapshot.desc.Format)},
        {"format_value", static_cast<unsigned>(snapshot.desc.Format)},
      };
    }

    using models::contour_history_readback::half_to_float;

    inline std::uint8_t encode_srgb(float value) {
      value = std::clamp(value, 0.0f, 1.0f);
      const float encoded = value <= 0.0031308f ?
                              12.92f * value :
                              1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
      return static_cast<std::uint8_t>(std::lround(encoded * 255.0f));
    }

    inline std::uint8_t encode_unit(float value) {
      return static_cast<std::uint8_t>(
        std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f)
      );
    }

    inline void tonemap_scrgb(float &r, float &g, float &b) {
      r = std::max(r, 0.0f);
      g = std::max(g, 0.0f);
      b = std::max(b, 0.0f);
      const float luminance =
        std::max(0.2126f * r + 0.7152f * g + 0.0722f * b, 0.0f);
      const float tone_scale = 1.0f / (1.0f + luminance);
      r *= tone_scale;
      g *= tone_scale;
      b *= tone_scale;
      const float gamut_scale =
        1.0f / std::max(1.0f, std::max(r, std::max(g, b)));
      r *= gamut_scale;
      g *= gamut_scale;
      b *= gamut_scale;
    }

    bool write_bytes(
      const std::filesystem::path &path,
      const void *data,
      const std::size_t size
    ) {
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      if (!out) {
        return false;
      }
      if (size != 0) {
        out.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
      }
      out.flush();
      return out.good();
    }

    bool write_text(const std::filesystem::path &path, const std::string &text) {
      return write_bytes(path, text.data(), text.size());
    }

    bool write_json(const std::filesystem::path &path, const nlohmann::json &value) {
      try {
        return write_text(path, value.dump(2) + "\n");
      } catch (const std::exception &error) {
        BOOST_LOG(warning) << "SBS debug dump: JSON serialization failed for "sv
                           << path.string() << ": " << error.what();
        return false;
      }
    }

    bool write_png(
      const std::filesystem::path &path,
      const std::uint32_t width,
      const std::uint32_t height,
      const std::vector<std::uint8_t> &rgb
    ) {
      if (
        width == 0 || height == 0 ||
        static_cast<std::uint64_t>(width) * height >
          std::numeric_limits<std::size_t>::max() / 3u ||
        rgb.size() != static_cast<std::size_t>(width) * height * 3u
      ) {
        return false;
      }

      const std::size_t scanline = 1u + static_cast<std::size_t>(width) * 3u;
      if (static_cast<std::size_t>(height) > SIZE_MAX / scanline) {
        return false;
      }
      std::vector<std::uint8_t> raw(static_cast<std::size_t>(height) * scanline);
      for (std::uint32_t y = 0; y < height; ++y) {
        const std::size_t output_offset = static_cast<std::size_t>(y) * scanline;
        raw[output_offset] = 0;
        std::memcpy(
          raw.data() + output_offset + 1u,
          rgb.data() + static_cast<std::size_t>(y) * width * 3u,
          static_cast<std::size_t>(width) * 3u
        );
      }

      if (
        raw.size() > std::numeric_limits<uLong>::max() ||
        raw.size() > std::numeric_limits<uLongf>::max()
      ) {
        return false;
      }
      uLongf compressed_size = compressBound(static_cast<uLong>(raw.size()));
      std::vector<std::uint8_t> compressed(compressed_size);
      if (
        compress2(
          compressed.data(),
          &compressed_size,
          raw.data(),
          static_cast<uLong>(raw.size()),
          Z_BEST_SPEED
        ) != Z_OK
      ) {
        return false;
      }

      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      if (!out) {
        return false;
      }
      const auto write_be32 = [](const std::uint32_t value, std::uint8_t *bytes) {
        bytes[0] = static_cast<std::uint8_t>(value >> 24u);
        bytes[1] = static_cast<std::uint8_t>(value >> 16u);
        bytes[2] = static_cast<std::uint8_t>(value >> 8u);
        bytes[3] = static_cast<std::uint8_t>(value);
      };
      const auto write_chunk = [&](const char *type, const std::uint8_t *data, const std::uint32_t size) {
        std::uint8_t encoded_size[4];
        write_be32(size, encoded_size);
        out.write(reinterpret_cast<const char *>(encoded_size), sizeof(encoded_size));
        out.write(type, 4);
        if (size != 0) {
          out.write(reinterpret_cast<const char *>(data), size);
        }
        uLong crc = crc32(0, reinterpret_cast<const Bytef *>(type), 4);
        if (size != 0) {
          crc = crc32(crc, data, size);
        }
        std::uint8_t encoded_crc[4];
        write_be32(static_cast<std::uint32_t>(crc), encoded_crc);
        out.write(reinterpret_cast<const char *>(encoded_crc), sizeof(encoded_crc));
        return out.good();
      };

      static constexpr std::uint8_t signature[8] {
        0x89,
        'P',
        'N',
        'G',
        0x0D,
        0x0A,
        0x1A,
        0x0A
      };
      out.write(reinterpret_cast<const char *>(signature), sizeof(signature));
      std::uint8_t header[13] {};
      write_be32(width, header);
      write_be32(height, header + 4);
      header[8] = 8;
      header[9] = 2;
      const bool ok =
        out.good() &&
        write_chunk("IHDR", header, sizeof(header)) &&
        write_chunk(
          "IDAT",
          compressed.data(),
          static_cast<std::uint32_t>(compressed_size)
        ) &&
        write_chunk("IEND", nullptr, 0);
      out.flush();
      return ok && out.good();
    }

    bool texture_to_rgb(
      const texture_snapshot &snapshot,
      const models::input_color_space color_space,
      std::vector<std::uint8_t> &rgb
    ) {
      const auto width = snapshot.desc.Width;
      const auto height = snapshot.desc.Height;
      if (
        static_cast<std::uint64_t>(width) * height >
        SIZE_MAX / 3u
      ) {
        return false;
      }
      rgb.resize(static_cast<std::size_t>(width) * height * 3u);
      for (std::uint32_t y = 0; y < height; ++y) {
        const std::uint8_t *input =
          snapshot.bytes.data() + static_cast<std::size_t>(y) * snapshot.row_bytes;
        std::uint8_t *output =
          rgb.data() + static_cast<std::size_t>(y) * width * 3u;
        for (std::uint32_t x = 0; x < width; ++x) {
          std::uint8_t r;
          std::uint8_t g;
          std::uint8_t b;
          switch (snapshot.desc.Format) {
            case DXGI_FORMAT_R16G16B16A16_FLOAT:
              {
                const auto *pixel =
                  reinterpret_cast<const std::uint16_t *>(input + static_cast<std::size_t>(x) * 8u);
                float rf = half_to_float(pixel[0]);
                float gf = half_to_float(pixel[1]);
                float bf = half_to_float(pixel[2]);
                if (!std::isfinite(rf) || !std::isfinite(gf) || !std::isfinite(bf)) {
                  r = 255;
                  g = 0;
                  b = 255;
                } else {
                  if (color_space == models::input_color_space::scrgb_hdr) {
                    tonemap_scrgb(rf, gf, bf);
                  }
                  if (color_space == models::input_color_space::srgb) {
                    // Some capture paths conservatively retain FP16 storage for sRGB code
                    // values. The warp copies those values unchanged, so applying an OETF here
                    // would double-gamma both source.png and sbs.png.
                    r = encode_unit(rf);
                    g = encode_unit(gf);
                    b = encode_unit(bf);
                  } else {
                    r = encode_srgb(rf);
                    g = encode_srgb(gf);
                    b = encode_srgb(bf);
                  }
                }
                break;
              }
            case DXGI_FORMAT_B8G8R8A8_UNORM:
            case DXGI_FORMAT_B8G8R8X8_UNORM:
            case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
              {
                const std::uint8_t *pixel = input + static_cast<std::size_t>(x) * 4u;
                b = pixel[0];
                g = pixel[1];
                r = pixel[2];
                break;
              }
            case DXGI_FORMAT_R8G8B8A8_UNORM:
            case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
              {
                const std::uint8_t *pixel = input + static_cast<std::size_t>(x) * 4u;
                r = pixel[0];
                g = pixel[1];
                b = pixel[2];
                break;
              }
            default:
              return false;
          }
          output[static_cast<std::size_t>(x) * 3u + 0u] = r;
          output[static_cast<std::size_t>(x) * 3u + 1u] = g;
          output[static_cast<std::size_t>(x) * 3u + 2u] = b;
        }
      }
      return true;
    }

    bool write_color_preview(
      const std::filesystem::path &path,
      const texture_snapshot &snapshot,
      const models::input_color_space color_space
    ) {
      std::vector<std::uint8_t> rgb;
      return texture_to_rgb(snapshot, color_space, rgb) &&
             write_png(path, snapshot.desc.Width, snapshot.desc.Height, rgb);
    }

    bool texture_float_values(
      const texture_snapshot &snapshot,
      std::vector<float> &values
    ) {
      if (snapshot.desc.Format != DXGI_FORMAT_R32_FLOAT) {
        return false;
      }
      const std::size_t count =
        static_cast<std::size_t>(snapshot.desc.Width) * snapshot.desc.Height;
      if (snapshot.bytes.size() != count * sizeof(float)) {
        return false;
      }
      values.resize(count);
      std::memcpy(values.data(), snapshot.bytes.data(), snapshot.bytes.size());
      return true;
    }

    scalar_stats calculate_scalar_stats(const std::vector<float> &values);

    bool write_float_texture(
      const std::filesystem::path &data_path,
      const texture_snapshot &snapshot
    ) {
      std::vector<float> values;
      if (!texture_float_values(snapshot, values)) {
        return false;
      }
      const scalar_stats stats = calculate_scalar_stats(values);
      if (stats.finite_count != values.size()) {
        return false;
      }
      return write_bytes(data_path, values.data(), values.size() * sizeof(float));
    }

    scalar_stats calculate_scalar_stats(const std::vector<float> &values) {
      scalar_stats stats;
      std::vector<float> finite;
      finite.reserve(values.size());
      for (const float value : values) {
        if (std::isfinite(value)) {
          finite.push_back(value);
        }
      }
      stats.finite_count = finite.size();
      if (finite.empty()) {
        return stats;
      }
      const auto bounds = std::minmax_element(finite.begin(), finite.end());
      stats.minimum = *bounds.first;
      stats.maximum = *bounds.second;
      return stats;
    }

    bool dump_model_input(
      const std::vector<float> &values,
      const int width,
      const int height,
      const std::filesystem::path &dir,
      const models::depth_coordinate_v2::model_preprocess_contract_t &preprocess
    ) {
      if (width <= 0 || height <= 0) {
        return false;
      }
      const std::uint64_t pixel_count =
        static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
      if (pixel_count > SIZE_MAX / (3u * sizeof(float))) {
        return false;
      }
      if (
        values.size() != static_cast<std::size_t>(pixel_count) * 3u ||
        !std::all_of(values.begin(), values.end(), [](const float value) {
          return std::isfinite(value);
        }) ||
        !write_bytes(
          dir / "model_input.f32",
          values.data(),
          values.size() * sizeof(float)
        )
      ) {
        return false;
      }

      const nlohmann::json shape {
        {"schema", preprocess.model_input_schema},
        {"width", width},
        {"height", height},
        {"dtype", std::string {preprocess.dtype}},
        {"layout", std::string {preprocess.layout}},
        {"channels", {
                       std::string {preprocess.channels[0]},
                       std::string {preprocess.channels[1]},
                       std::string {preprocess.channels[2]},
                     }},
        {"stage", std::string {preprocess.stage}},
        {"imagenet_mean", preprocess.imagenet_mean},
        {"imagenet_std", preprocess.imagenet_std},
      };
      return write_json(dir / "model_input_shape.json", shape);
    }

    bool dump_raw_depth(
      const std::vector<float> &values,
      const int width,
      const int height,
      const std::filesystem::path &dir,
      raw_depth_dump_stats &stats
    ) {
      if (width <= 0 || height <= 0) {
        return false;
      }
      const std::uint64_t value_count =
        static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
      if (value_count > SIZE_MAX / sizeof(float)) {
        return false;
      }
      if (values.size() != static_cast<std::size_t>(value_count)) {
        return false;
      }
      const scalar_stats scalar = calculate_scalar_stats(values);
      static_cast<scalar_stats &>(stats) = scalar;
      stats.width = static_cast<std::uint32_t>(width);
      stats.height = static_cast<std::uint32_t>(height);
      if (
        scalar.finite_count == 0 ||
        !write_bytes(
          dir / "raw_depth.f32",
          values.data(),
          values.size() * sizeof(float)
        )
      ) {
        return false;
      }
      return true;
    }

    bool dump_shadow_float_texture(
      const texture_snapshot &snapshot,
      const std::filesystem::path &dir,
      const std::string_view stem
    ) {
      const std::string name(stem);
      return write_float_texture(dir / (name + ".f32"), snapshot);
    }

    bool dump_parallax_v2_state(
      const frame &completed,
      const std::vector<float> &state,
      const std::vector<float> &frame_stats,
      const std::filesystem::path &dir,
      nlohmann::json &summary
    ) {
      using namespace models::depth_coordinate_v2;
      if (!parallax_v2_shader_identity_matches_contract(
            completed.parallax_v2_shader_provenance
          )) {
        return false;
      }
      const auto &shader_identity = *completed.parallax_v2_shader_provenance;
      if (!detail::parallax_v2_state_matches_frame(state, frame_stats, completed)) {
        return false;
      }
      state_words_t state_words {};
      std::memcpy(state_words.data(), state.data(), sizeof(state_words));
      const bool state_frame_valid = state[frame_valid] > 0.5f;
      const auto calibration_revision_value = state_words[calibration_revision];
      const auto camera_center_integrity_value = state_words[camera_center_integrity_bits];
      const auto renderer_authorization_value = state_words[renderer_authorization_bits];
      const bool camera_initialized =
        state[inverse_scale] > 0.0f &&
        acquired_calibration_revision_is_valid(calibration_revision_value);

      nlohmann::json fields = nlohmann::json::array();
      nlohmann::json named_values = nlohmann::json::object();
      for (std::size_t index = 0; index < state.size(); ++index) {
        const auto &descriptor = state_fields[index];
        const std::string name {descriptor.name};
        if (descriptor.gpu_encoding == state_gpu_encoding_e::uint_bits) {
          const auto bits = std::bit_cast<std::uint32_t>(state[index]);
          fields.push_back({
            {"index", index},
            {"name", name},
            {"type", "uint32-bitcast"},
            {"value", bits},
          });
          named_values[name] = bits;
          continue;
        }
        fields.push_back({
          {"index", index},
          {"name", name},
          {"type", "float32"},
          {"value", state[index]},
        });
        named_values[name] = state[index];
      }

      const auto joint_plane_mode_value = state_words[joint_plane_mode_bits];
      const float effective_gain_value = state_frame_valid ?
        completed.parallax_v2_requested_gain : 0.0f;
      const float latched_scale_value = camera_initialized ?
        1.0f / state[inverse_scale] : 0.0f;
      const nlohmann::json decoded {
        {"frame_valid", state_frame_valid},
        {"camera_valid", camera_initialized},
        {"calibration_revision", calibration_revision_value},
        {"contract_tag", contract_tag},
        {"requested_gain", completed.parallax_v2_requested_gain},
        {"requested_pop_strength", completed.parallax_v2_requested_pop_strength},
        {"latched_scale", latched_scale_value},
        {"convergence_curve", state[convergence_curve]},
        {"container_scale", state[container_scale]},
        {"effective_gain", effective_gain_value},
        {"camera_center_integrity_bits", camera_center_integrity_value},
        {"renderer_authorization_bits", renderer_authorization_value},
        {"joint_plane_mode", joint_plane_mode_value},
      };
      const nlohmann::json state_json {
        {"schema", shadow_state_dump_schema},
        {"coordinate_contract", parallax_v2_coordinate_binding(
                                  shader_identity,
                                  "state_word_count",
                                  state_float_count
                                )},
        {"source", std::string {shadow_state_source}},
        {"capture", std::string {shadow_state_capture}},
        {"rendered_output_selected", true},
        {"wire_contract", "authenticated live Host-SBS renderer input; not a client wire contract"},
        {"units", {
                    {"coordinate", "dimensionless canonical coordinate derived from raw depth"},
                    {"gain", completed.depth_input_region.is_video_region() ?
                        "one-eye ROI-local source-U per coordinate unit" :
                        "one-eye full-source-U per coordinate unit"},
                    {"parallax", completed.depth_input_region.is_video_region() ?
                      "signed one-eye ROI-local source-U; full-source renderer authority additionally requires depth_input_region embedding" :
                      "signed one-eye full-source-U"},
                  }},
        {"constants", {
                        {"raw_coordinate_scale", completed.parallax_v2_raw_coordinate_scale},
                        {"collapse_abs_epsilon", collapse_abs_epsilon},
                        {"gain_per_pop", gain_per_pop},
                        {"reference_pop_strength", reference_pop_strength},
                        {"reference_gain_at_reference_pop", parallax_gain},
                        {"requested_gain", completed.parallax_v2_requested_gain},
                        {"requested_pop_strength", completed.parallax_v2_requested_pop_strength},
                        {"direct_container_limit", direct_container_limit},
                        {"max_horizontal_slope", max_horizontal_slope},
                        {"max_vertical_shear", max_vertical_shear},
                        {"vertical_majorant_share", vertical_majorant_share},
                        {"convergence_curve_default", convergence_curve_default},
                      }},
        {"fields", std::move(fields)},
        {"named_values", std::move(named_values)},
        {"decoded", decoded},
        {"adaptation_semantics", {
                                    {"coordinate", "bounded-source-time-linear-raw-mean-and-robust-amplitude-reference"},
                                    {"convergence_curve", "continuously-tracked-mean-is-zero-plane"},
                                    {"requested_gain", "immutable-cfg-pop-strength-independent-of-display-limit"},
                                    {"container_scale", "abi-retained-identity-mode3-hard-representation-cap"},
                                    {"near_curve", "linear-coordinate-no-depth-curve-before-hard-representation-cap"},
                                    {"spatial_conditioner", "fixed-75pct-vertical-majorant-share-then-horizontal-majorant"},
                                  }},
      };

      nlohmann::json frame_named = nlohmann::json::object();
      for (std::size_t index = 0; index < frame_stats.size(); ++index) {
        frame_named[frame_stat_names[index]] = frame_stats[index];
      }
      const nlohmann::json frame_json {
        {"schema", shadow_frame_stats_dump_schema},
        {"coordinate_contract", parallax_v2_coordinate_binding(
                                  shader_identity,
                                  "frame_stats_word_count",
                                  frame_stats_float_count
                                )},
        {"source", std::string {frame_stats_source}},
        {"named_values", std::move(frame_named)},
      };
      summary = decoded;
      summary["raw_coordinate_scale"] = completed.parallax_v2_raw_coordinate_scale;
      summary["rendered_output_selected"] = true;
      return write_json(dir / "shadow_state.json", state_json) &&
             write_json(dir / "shadow_frame_stats.json", frame_json);
    }

    const char *encoding_name(const sbs_adaptive_state::gpu_encoding_e encoding) {
      switch (encoding) {
        case sbs_adaptive_state::gpu_encoding_e::float_value:
          return "float_value";
        case sbs_adaptive_state::gpu_encoding_e::uint_bits:
          return "uint_bits";
        case sbs_adaptive_state::gpu_encoding_e::uint_valued_float:
          return "uint_valued_float";
      }
      return "unknown";
    }

    bool dump_adaptive_state(
      const std::vector<std::uint8_t> &bytes,
      const normalization_state &normalization,
      const frame &completed,
      const std::filesystem::path &dir,
      nlohmann::json &adaptive_summary
    ) {
      if (bytes.size() !=
          sbs_adaptive_state::word_count * sizeof(std::uint32_t)) {
        return false;
      }
      sbs_adaptive_state::words_t words {};
      std::memcpy(words.data(), bytes.data(), bytes.size());
      nlohmann::json fields = nlohmann::json::array();
      nlohmann::json values = nlohmann::json::array();
      nlohmann::json named_values = nlohmann::json::object();
      std::array<float, sbs_adaptive_state::word_count> scalars {};
      for (const auto &descriptor : sbs_adaptive_state::fields) {
        const std::size_t index = sbs_adaptive_state::index(descriptor.word);
        const std::uint32_t raw_word = words[index];
        nlohmann::json value;
        if (
          descriptor.gpu_encoding ==
          sbs_adaptive_state::gpu_encoding_e::uint_bits
        ) {
          value = raw_word;
        } else {
          const float scalar = std::bit_cast<float>(raw_word);
          if (!std::isfinite(scalar)) {
            return false;
          }
          scalars[index] = scalar;
          if (
            descriptor.gpu_encoding ==
            sbs_adaptive_state::gpu_encoding_e::uint_valued_float
          ) {
            if (
              scalar < 0.0f ||
              scalar > static_cast<float>(std::numeric_limits<std::uint32_t>::max()) ||
              std::trunc(scalar) != scalar
            ) {
              return false;
            }
            value = static_cast<std::uint32_t>(scalar);
          } else {
            value = scalar;
          }
        }
        values.push_back(value);
        named_values[std::string(descriptor.name)] = value;
        fields.push_back({
          {"index", index},
          {"name", std::string(descriptor.name)},
          {"json_type", std::string(descriptor.json_type)},
          {"gpu_encoding", encoding_name(descriptor.gpu_encoding)},
          {"raw_word", raw_word},
          {"value", std::move(value)},
        });
      }

      using sbs_adaptive_state::word_e;
      adaptive_summary = {
        {"schema", sbs_adaptive_state::schema_version},
        {"source", std::string(sbs_adaptive_state::source)},
        {"capture", std::string(sbs_adaptive_state::capture)},
        {"role", "comparison-only depth-health state; no live V2 geometry authority"},
        {"matched_frame_id", completed.matched_frame_id},
        {"depth_model", completed.depth_model},
        {"fields", std::move(fields)},
        {"values", std::move(values)},
        {"named_values", std::move(named_values)},
        {"decoded", {
                      {"empty_raw_count", words[sbs_adaptive_state::index(word_e::empty_raw_count)]},
                      {"collapsed_raw_count", words[sbs_adaptive_state::index(word_e::collapsed_raw_count)]},
                      {"geometry_authority", false},
                    }},
        {"normalization", {
                            {"role", "comparison-only depth normalization state"},
                            {"effective_lower", normalization.lower},
                            {"effective_upper", normalization.upper},
                            {"initialized", normalization.initialized > 0.5f},
                            {"initialized_value", normalization.initialized},
                            {"frame_state", normalization_frame_state_name(normalization.frame_state)},
                            {"frame_state_value", normalization.frame_state},
                          }},
      };
      return write_json(dir / "adaptive_state.json", adaptive_summary);
    }

    bool dump_warp_map(
      const texture_snapshot &mapping,
      const std::uint32_t source_width,
      const std::uint32_t source_height,
      const std::filesystem::path &dir,
      warp_map_dump_stats &stats
    ) {
      std::vector<float> map;
      if (
        !texture_float_values(mapping, map) ||
        mapping.desc.Width < 2u || (mapping.desc.Width & 1u) != 0u ||
        source_width == 0u || source_height == 0u
      ) {
        return false;
      }
      stats.width = mapping.desc.Width;
      stats.height = mapping.desc.Height;
      stats.eye_width = mapping.desc.Width / 2u;
      stats.eye_height = mapping.desc.Height;
      const float source_aspect =
        static_cast<float>(source_width) / static_cast<float>(source_height);
      const float eye_aspect =
        static_cast<float>(stats.eye_width) / static_cast<float>(stats.eye_height);
      stats.content_scale_x =
        eye_aspect > source_aspect ? source_aspect / eye_aspect : 1.0f;
      stats.content_scale_y =
        eye_aspect < source_aspect ? eye_aspect / source_aspect : 1.0f;
      if (!std::all_of(map.begin(), map.end(), [](const float value) {
            return std::isfinite(value);
          })) {
        return false;
      }
      return write_bytes(
        dir / "warp_map.f32",
        map.data(),
        map.size() * sizeof(float)
      );
    }

    nlohmann::json config_json(
      const config::video_t::sbs_t &cfg,
      const frame &completed,
      const std::string &model_name,
      const std::string &effective_model_url
    ) {
      return {
        {"schema", 3},
        {"shared_configured", {
          {"pop_strength", cfg.pop_strength},
          {"max_packed_encode_width", cfg.max_encode_width},
        }},
        {"live_effective", {
          {"renderer", "depth-coordinate-v2"},
          {"pop_strength", completed.parallax_v2_requested_pop_strength},
          {"adaptive_pop", false},
          {"zero_plane_authority", "continuously source-time tracked raw mean"},
          {"depth_model", model_name},
          {"depth_model_url", effective_model_url},
          {"model_input_width", completed.model_width},
          {"model_input_height", completed.model_height},
          {"cuda_graph_active", completed.cuda_graph_active},
          {"normalization_analysis", {
            {"depth_ema", config::host_sbs_v2_live_calibration::depth_ema},
            {"ema_edge_change", config::host_sbs_v2_live_calibration::edge_change},
            {"ema_edge_gradient", config::host_sbs_v2_live_calibration::edge_gradient},
            {"ema_edge_strength", config::host_sbs_v2_live_calibration::edge_strength},
            {"minmax_ema", config::host_sbs_v2_live_calibration::minmax_ema},
          }},
        }},
      };
    }

    nlohmann::json artifact_description(
      const bool available,
      const bool required,
      const std::string_view stage,
      const std::string_view description
    ) {
      return {
        {"available", available},
        {"required", required},
        {"stage", std::string(stage)},
        {"description", std::string(description)},
      };
    }

    nlohmann::json hashed_artifact_description(
      const bool available,
      const bool required,
      const std::string_view stage,
      const std::string_view description,
      const std::string &sha256
    ) {
      auto descriptor = artifact_description(available, required, stage, description);
      descriptor["sha256"] = sha256;
      return descriptor;
    }

    void write_actual_draw_evidence(const captured_dump_job &job, const std::filesystem::path &dir,
      nlohmann::json &artifacts) {
      const auto &e = job.completed.actual_draw;
      if (!e.requested) return;
      nlohmann::json entries = nlohmann::json::object();
      for (std::size_t i = 0u; i < actual_draw_artifacts.size(); ++i) {
        const auto storage = actual_draw_storage_index(i);
        const auto &bytes = actual_draw_is_texture(i) ? job.actual_draw_textures[storage].bytes : job.actual_draw_buffers[storage];
        bool available = job.actual_draw_available[i];
        std::string unavailable = job.actual_draw_unavailable[i];
        std::string digest;
        if (available) {
          available = write_bytes(dir / actual_draw_artifacts[i], bytes.data(), bytes.size());
          if (available) digest = models::file_sha256_hex(dir / actual_draw_artifacts[i]);
          available = available && !digest.empty();
          if (!available) unavailable = "optional artifact write/hash failed";
        }
        auto entry = hashed_artifact_description(available, false, "actual draw binding observation",
          "Literal requested-dump GPU readback; presence is not shader authorization or a per-pixel route.", digest);
        entry["byte_count"] = available ? bytes.size() : 0u;
        entry["unavailable_reason"] = available ? "" : unavailable;
        if (actual_draw_is_texture(i) && available) entry["texture"] = texture_description(job.actual_draw_textures[storage]);
        entries[actual_draw_artifacts[i]] = entry;
        artifacts[actual_draw_artifacts[i]] = entry;
      }
      const auto words = [](const std::vector<std::uint8_t> &bytes, std::size_t first, std::size_t count) {
        nlohmann::json values = nlohmann::json::array();
        if ((first + count) * sizeof(std::uint32_t) > bytes.size()) return values;
        for (std::size_t i = first; i < first + count; ++i) {
          std::uint32_t value;
          std::memcpy(&value, bytes.data() + i * sizeof(value), sizeof(value));
          values.push_back(value);
        }
        return values;
      };
      const auto &receipt = job.actual_draw_buffers[0];
      nlohmann::json metadata {
        {"schema", 2u}, {"diagnostic_only", true}, {"captured", e.captured},
        {"unavailable_reason", e.unavailable_reason},
        {"observation_point", "PSGet bindings inside the recorder immediately before Draw and cleanup"},
        {"snapshot_order", "same immediate-context conversion; copies precede terminal dump event and next live reuse"},
        {"conversion_frame", e.conversion_frame}, {"model_frame", e.model_frame}, {"observation_us", e.observation_us},
        {"program", e.program}, {"selected_shader", e.selected_shader},
        {"pixel_shader_address", e.pixel_shader_address}, {"pixel_shader_closure", e.pixel_shader_closure},
        {"pixel_route_map", e.shared_pin_selected ? "actual_shared_pin_map.rgba32f" : "not_exported"},
        {"shared_pin", {
          {"selected", e.shared_pin_selected}, {"source_closure_sha256", e.shared_pin_source_closure},
          {"source_shape", words(job.actual_draw_buffers[3u], 0u, 4u)},
          {"source_shape_layout", "fullSourceWidth,fullSourceHeight,nativeBandTop,nativeBandHeight"},
          {"map_artifact", "actual_shared_pin_map.rgba32f"}, {"boundary_artifact", "actual_shared_pin_boundary.rgba32f"},
          {"map_layout", "packed-output row-major float32-le RGBA: sourceU,sourceV,pinnedRow,contentValid"},
          {"boundary_layout", "packed-output row-major float32-le RGBA: finiteSourceBoundary,0,0,1"},
          {"map_authority", "same-draw shared-field inverse before native source clamp; warp_map.f32 remains a separate ordinary Base diagnostic"},
          {"replay", "native field, current authorized weights, rows/header and exact b6 are literal GPU readbacks; no CPU pin-field replica"},
        }},
        {"fallback_policy", "Unavailable shared preparation or invalid selected authority preserves authenticated ordinary Base with subtitle permission disabled; empty or revoked pin rows also preserve Base. Independent flat rendering is reserved for invalid or unavailable depth. Binding flags alone do not establish GPU authorization."},
        {"published_matte_artifact", "subtitle_glyph_matte.u32 remains an independent mask-construction diagnostic; actual_draw_M.u32 is observed t3"},
        {"binding_roles", {{"t3", "actual_draw_M.u32"}, {"t7", "actual_selected_receipt.u32"},
          {"t12", "actual_shared_pin_weights.f32"}, {"t13", "actual_shared_pin_field.f32"},
          {"t14", "actual_shared_pin_rows.u32"}, {"b6", "actual_shared_pin_geometry.u32"}}},
        {"owner_serial", e.owner_serial},
        {"context_address", e.context_address}, {"colour_key", e.colour_key},
        {"actual_role_pointers", e.actual_role_pointers}, {"analysis_source_rect", e.analysis_source_rect}, {"analysis_shape", e.analysis_shape},
        {"analysis_source_rect_layout", "left,top,width,height in full-source native pixels"},
        {"analysis_shape_layout", e.shared_pin_selected ? "fullSourceWidth,fullSourceHeight,nativeBandTop,nativeBandHeight" : "analysisWidth,analysisHeight,cropTop,bandHeight"},
        {"selected_receipt_header", words(receipt, 0u, 4u)}, {"selected_receipt_token_algorithm", words(receipt, 60u, 4u)},
        {"artifacts", entries},
      };
      if (!e.shared_pin_model_prepare_closure.empty()) {
        metadata["shared_pin"]["model_prepare_source_closure_sha256"] = e.shared_pin_model_prepare_closure;
      }
      const auto path = dir / "actual_draw_evidence.json";
      const bool written = write_json(path, metadata);
      const auto digest = written ? models::file_sha256_hex(path) : std::string {};
      const bool available = written && !digest.empty();
      auto descriptor = hashed_artifact_description(available, false,
        "actual draw binding observation", "Optional schema-2 literal selected/shared bindings; does not replace tuple authentication.", digest);
      descriptor["unavailable_reason"] = available ? "" : "optional metadata write/hash failed";
      artifacts["actual_draw_evidence.json"] = std::move(descriptor);
    }

    const char *gpu_trace_submission_class_name(const std::uint32_t value) noexcept {
      using models::host_sbs_gpu_trace::submission_class_e;
      switch (static_cast<submission_class_e>(value)) {
        case submission_class_e::force_infer:
          return "force-infer";
        case submission_class_e::gpu_undecided:
          return "gpu-undecided";
        default:
          return "invalid";
      }
    }

    const char *gpu_trace_depth_disposition_name(const std::uint32_t value) noexcept {
      using models::host_sbs_gpu_trace::depth_disposition_e;
      switch (static_cast<depth_disposition_e>(value)) {
        case depth_disposition_e::reuse:
          return "reuse";
        case depth_disposition_e::infer:
          return "infer";
        default:
          return "invalid";
      }
    }

    const char *gpu_trace_subtitle_disposition_name(
      const std::uint32_t value
    ) noexcept {
      using models::host_sbs_gpu_trace::subtitle_disposition_e;
      switch (static_cast<subtitle_disposition_e>(value)) {
        case subtitle_disposition_e::suppressed:
          return "suppressed";
        case subtitle_disposition_e::optional_ocr:
          return "optional-ocr";
        case subtitle_disposition_e::abstention:
          return "abstention";
        case subtitle_disposition_e::held_with_depth:
          return "held-with-depth";
        default:
          return "invalid";
      }
    }

    const char *gpu_trace_host_outcome_name(const std::uint32_t value) noexcept {
      using models::host_sbs_gpu_trace::host_subtitle_outcome_e;
      switch (static_cast<host_subtitle_outcome_e>(value)) {
        case host_subtitle_outcome_e::suppressed:
          return "suppressed";
        case host_subtitle_outcome_e::ordinary_record:
          return "ordinary-record";
      }
      return "invalid";
    }

    const char *gpu_trace_expected_work_name(const std::uint32_t value) noexcept {
      using cuda_conditional_graph::work_flag_e;
      using cuda_conditional_graph::work_flags_value;
      if (value == work_flags_value(work_flag_e::none)) {
        return "none";
      }
      if (value == work_flags_value(work_flag_e::optional_ocr)) {
        return "optional-ocr";
      }
      if (value == work_flags_value(work_flag_e::subtitle_observation)) {
        return "subtitle-observation";
      }
      return "invalid";
    }

    std::uint32_t gpu_trace_word(
      const std::vector<std::uint8_t> &ring,
      const std::size_t index
    ) noexcept {
      std::uint32_t value = 0u;
      if (index < ring.size() / sizeof(value)) {
        std::memcpy(&value, ring.data() + index * sizeof(value), sizeof(value));
      }
      return value;
    }

    nlohmann::json gpu_trace_word_array(
      const std::vector<std::uint8_t> &ring,
      const std::size_t begin,
      const std::size_t count
    ) {
      auto result = nlohmann::json::array();
      for (std::size_t index = 0u; index < count; ++index) {
        result.push_back(gpu_trace_word(ring, begin + index));
      }
      return result;
    }

    nlohmann::json gpu_trace_contract_document(
      const models::host_sbs_gpu_trace_provenance_t &provenance
    ) {
      using namespace models::host_sbs_gpu_trace;
      using models::host_sbs_shader_cache::host_sbs_gpu_trace;
      using cuda_conditional_graph::work_flag_e;
      using cuda_conditional_graph::work_flags_value;
      const auto offset = [](const auto value) {
        return static_cast<std::uint32_t>(value);
      };
      return {
        {"schema", dump_contract_schema},
        {"role", "diagnostic-only accepted-root completion history; never rendering authority"},
        {"byte_order", "little-endian"},
        {"decoded_trace_schema", decoded_trace_schema},
        {"ring", {
          {"schema", ring_schema},
          {"tag", ring_tag},
          {"capacity", capacity},
          {"header_words", header_word_count},
          {"record_words", record_word_count},
          {"record_bytes", record_word_count * sizeof(std::uint32_t)},
          {"ring_words", ring_word_count},
          {"ring_bytes", ring_byte_count},
          {"record_array_word_offset", header_word_count},
          {"record_array_byte_offset", header_word_count * sizeof(std::uint32_t)},
          {"record_slot_word_stride", record_word_count},
          {"record_slot_byte_stride", record_word_count * sizeof(std::uint32_t)},
          {"commit_protocol", "header tag invalidated before slot overwrite; payload then record tag; cursor then header tag last"},
        }},
        {"header_word_offsets", {
          {"schema", offset(header_word_e::schema)},
          {"tag", offset(header_word_e::tag)},
          {"capacity", offset(header_word_e::capacity)},
          {"record_words", offset(header_word_e::record_words)},
          {"next_sequence_low", offset(header_word_e::next_sequence_low)},
          {"next_sequence_high", offset(header_word_e::next_sequence_high)},
          {"next_slot", offset(header_word_e::next_slot)},
          {"committed_count", offset(header_word_e::committed_count)},
          {"reserved_begin", offset(header_word_e::reserved_begin)},
          {"end", header_word_count},
        }},
        {"record_word_offsets", {
          {"schema", offset(record_word_e::schema)},
          {"commit_tag", offset(record_word_e::commit_tag)},
          {"sequence_low", offset(record_word_e::sequence_low)},
          {"sequence_high", offset(record_word_e::sequence_high)},
          {"frame_low", offset(record_word_e::frame_low)},
          {"frame_high", offset(record_word_e::frame_high)},
          {"analysis_generation_low", offset(record_word_e::analysis_generation_low)},
          {"analysis_generation_high", offset(record_word_e::analysis_generation_high)},
          {"domain_tag_low", offset(record_word_e::domain_tag_low)},
          {"domain_tag_high", offset(record_word_e::domain_tag_high)},
          {"transaction_token_low", offset(record_word_e::transaction_token_low)},
          {"transaction_token_high", offset(record_word_e::transaction_token_high)},
          {"submission_class", offset(record_word_e::submission_class)},
          {"depth_disposition", offset(record_word_e::depth_disposition)},
          {"expected_work", offset(record_word_e::expected_work)},
          {"subtitle_disposition", offset(record_word_e::subtitle_disposition)},
          {"flags", offset(record_word_e::flags)},
          {"host_subtitle_outcome", offset(record_word_e::host_subtitle_outcome)},
          {"source_width", offset(record_word_e::source_width)},
          {"source_height", offset(record_word_e::source_height)},
          {"field_width", offset(record_word_e::field_width)},
          {"field_height", offset(record_word_e::field_height)},
          {"transaction_words", offset(record_word_e::transaction_words)},
          {"reserved0", offset(record_word_e::reserved0)},
          {"transaction_begin", offset(record_word_e::transaction_begin)},
          {"subtitle_locator_begin", offset(record_word_e::subtitle_locator_begin)},
          {"subtitle_condition_begin", offset(record_word_e::subtitle_condition_begin)},
          {"observation_timestamp_low", offset(record_word_e::observation_timestamp_low)},
          {"observation_timestamp_high", offset(record_word_e::observation_timestamp_high)},
          {"reserved_begin", offset(record_word_e::reserved_begin)},
          {"end", record_word_count},
        }},
        {"record_sections", {
          {"transaction", {
            {"word_offset", offset(record_word_e::transaction_begin)},
            {"word_count", transaction_word_count},
            {"validity", "immutable postprocessed transaction snapshot; branch remains invalid unless RQST/CBRG/cookies/token/class all authenticate"},
          }},
          {"subtitle_locator", {
            {"word_offset", offset(record_word_e::subtitle_locator_begin)},
            {"word_count", subtitle_locator_word_count},
            {"validity", "post-finalization UIP4 for an authenticated subtitle observation; held_with_depth preserves the byte-identical immediately prior tuple and frame identity; subtitle_suppressed bytes are frozen/unused"},
          }},
          {"subtitle_condition", {
            {"word_offset", offset(record_word_e::subtitle_condition_begin)},
            {"word_count", models::depth_coordinate_v2::subtitle_condition_param_word_count},
            {"validity", "post-finalization SCP4 for an authenticated subtitle observation; active payload matches current UIP4 slots and applied target, inactive payload is zero with fresh schema/tag; held_with_depth preserves the byte-identical immediately prior params; suppressed words are unused; actual OCR8 covers are verified separately"},
          }},
          {"observation_timestamp", {
            {"word_offset", offset(record_word_e::observation_timestamp_low)},
            {"word_count", 2u},
            {"validity", "nonzero monotonic source-observation microseconds; zero is invalid"},
          }},
          {"reserved", {
            {"word_offset", offset(record_word_e::reserved_begin)},
            {"word_count", record_word_count - offset(record_word_e::reserved_begin)},
            {"validity", "must be zero"},
          }},
        }},
        {"transaction_word_offsets", {
          {"receipt", {{"word_offset", 0u}, {"word_count", 8u}}},
          {"request", {{"word_offset", 8u}, {"word_count", 8u}}},
          {"indirect_dispatch_groups", {{"word_offset", 16u}, {"word_count", 48u}, {"group_word_stride", 4u}}},
        }},
        {"trace_constant_word_offsets", {
          {"frame_low", 0u}, {"frame_high", 1u},
          {"analysis_generation_low", 2u}, {"analysis_generation_high", 3u},
          {"domain_tag_low", 4u}, {"domain_tag_high", 5u},
          {"transaction_token_low", 6u}, {"transaction_token_high", 7u},
          {"expected_work", 8u}, {"submission_class", 9u},
          {"flags", 10u}, {"host_subtitle_outcome", 11u},
          {"source_width", 12u}, {"source_height", 13u},
          {"field_width", 14u}, {"field_height", 15u},
          {"observation_timestamp_low", 16u}, {"observation_timestamp_high", 17u},
          {"padding0", 18u}, {"padding1", 19u},
        }},
        {"tags", {{"header", ring_tag}, {"record", record_tag}}},
        {"receipt_abi", {
          {"decision_cookie", cuda_conditional_graph::decision_cookie},
          {"token_low_cookie", cuda_conditional_graph::token_low_cookie},
          {"token_high_cookie", cuda_conditional_graph::token_high_cookie},
          {"work_flags_cookie", cuda_conditional_graph::work_flags_cookie},
          {"proposal_magic", cuda_conditional_graph::proposal_magic},
          {"request_magic", cuda_conditional_graph::request_magic},
          {"receipt_magic", cuda_conditional_graph::receipt_magic},
          {"optional_ocr_receipt_magic", cuda_conditional_graph::optional_ocr_receipt_magic},
          {"branch", {{"reuse", 0u}, {"infer", 1u}}},
          {"work", {
            {"none", work_flags_value(work_flag_e::none)},
            {"optional_ocr", work_flags_value(work_flag_e::optional_ocr)},
            {"subtitle_observation", work_flags_value(work_flag_e::subtitle_observation)},
          }},
        }},
        {"enums", {
          {"submission_class", {{"invalid", 0u}, {"force_infer", 1u}, {"gpu_undecided", 2u}}},
          {"depth_disposition", {{"invalid", 0u}, {"reuse", 1u}, {"infer", 2u}}},
          {"subtitle_disposition", {
            {"suppressed", 0u}, {"optional_ocr", 1u}, {"abstention", 2u},
            {"held_with_depth", 5u}, {"invalid", 6u},
          }},
          {"host_subtitle_outcome", {
            {"suppressed", 0u}, {"ordinary_record", 1u},
          }},
          {"flags", {
            {"input_domain_reset", input_domain_reset},
            {"dump_forced_at_enqueue", dump_forced},
            {"ocr_record_submitted", ocr_record_submitted},
            {"subtitle_suppressed", subtitle_suppressed},
            {"condition_executed", condition_executed},
            {"subtitle_branch_gated", subtitle_branch_gated},
            {"known_mask", known_record_flags},
          }},
        }},
        {"subtitle_locator", {
          {"schema", subtitle_plane_state_schema},
          {"tag", subtitle_plane_state_tag},
          {"word_count", subtitle_locator_word_count},
        }},
        {"subtitle_condition", {
          {"schema", subtitle_condition_param_schema},
          {"tag", subtitle_condition_param_tag},
          {"word_count", models::depth_coordinate_v2::subtitle_condition_param_word_count},
        }},
        {"shader", {
          {"source_file", std::string {host_sbs_gpu_trace.filename}},
          {"entrypoint", std::string {host_sbs_gpu_trace.entrypoint}},
          {"profile", std::string {host_sbs_gpu_trace.target}},
          {"source_closure_schema", provenance.source_closure_schema},
          {"source_compile_flags", provenance.source_compile_flags},
          {"source_macro_count", provenance.source_macro_count},
          {"source_closure_sha256", provenance.source_closure_sha256},
        }},
      };
    }

    nlohmann::json gpu_trace_decoded_document(
      const std::vector<std::uint8_t> &ring,
      const frame &completed
    ) {
      using namespace models::host_sbs_gpu_trace;
      const auto next_sequence = join_u64(
        gpu_trace_word(ring, word_index(header_word_e::next_sequence_low)),
        gpu_trace_word(ring, word_index(header_word_e::next_sequence_high))
      );
      const auto next_slot = gpu_trace_word(
        ring, word_index(header_word_e::next_slot)
      );
      const auto count = gpu_trace_word(
        ring, word_index(header_word_e::committed_count)
      );
      const auto oldest_sequence = next_sequence - count;
      const auto oldest_slot = (next_slot + capacity - count) % capacity;
      auto records = nlohmann::json::array();
      std::uint64_t matched_sequence = 0u;
      for (std::uint32_t ordinal = 0u; ordinal < count; ++ordinal) {
        const auto slot = (oldest_slot + ordinal) % capacity;
        const auto base = record_base(slot);
        const auto word = [&ring, base](const record_word_e field) {
          return gpu_trace_word(ring, base + word_index(field));
        };
        const auto sequence = join_u64(
          word(record_word_e::sequence_low), word(record_word_e::sequence_high)
        );
        const auto frame_id = join_u64(
          word(record_word_e::frame_low), word(record_word_e::frame_high)
        );
        const auto analysis_generation = join_u64(
          word(record_word_e::analysis_generation_low),
          word(record_word_e::analysis_generation_high)
        );
        const auto domain_tag = join_u64(
          word(record_word_e::domain_tag_low), word(record_word_e::domain_tag_high)
        );
        const auto token = join_u64(
          word(record_word_e::transaction_token_low),
          word(record_word_e::transaction_token_high)
        );
        const auto flags = word(record_word_e::flags);
        const auto locator_base =
          base + word_index(record_word_e::subtitle_locator_begin);
        const auto condition_base =
          base + word_index(record_word_e::subtitle_condition_begin);
        const auto locator_word = [&ring, locator_base](const std::size_t index) {
          return gpu_trace_word(ring, locator_base + index);
        };
        const auto condition_word = [&ring, condition_base](const std::size_t index) {
          return gpu_trace_word(ring, condition_base + index);
        };
        const auto plane_flags = locator_word(subtitle_plane_flags_word);
        const auto slot_mask = locator_word(subtitle_plane_current_slot_mask_word);
        const auto target_bits = locator_word(subtitle_plane_adaptive_offset + 3u);
        const auto target = std::bit_cast<float>(target_bits);
        const bool suppressed = (flags & subtitle_suppressed) != 0u;
        const bool condition_valid = !suppressed &&
          condition_word(0u) == subtitle_condition_param_schema &&
          condition_word(1u) == subtitle_condition_param_tag;
        const auto condition_target_bits = condition_word(5u);
        const auto condition_target = std::bit_cast<float>(condition_target_bits);
        const bool condition_executed_host_proven =
          (flags & condition_executed) != 0u;
        const bool branch_gated = (flags & subtitle_branch_gated) != 0u;
        const auto subtitle_disposition = static_cast<subtitle_disposition_e>(
          word(record_word_e::subtitle_disposition)
        );
        const bool subtitle_held_with_depth =
          subtitle_disposition == subtitle_disposition_e::held_with_depth;
        const bool subtitle_observation_executed =
          subtitle_disposition == subtitle_disposition_e::optional_ocr ||
          subtitle_disposition == subtitle_disposition_e::abstention;
        const bool condition_was_executed = condition_executed_host_proven ||
          (branch_gated && subtitle_observation_executed);
        const bool condition_was_published =
          condition_was_executed || subtitle_held_with_depth;
        const bool condition_active = condition_was_published && condition_valid &&
          condition_word(2u) != 0u &&
          condition_word(4u) == 2u &&
          subtitle_target_is_representable(condition_target);
        const bool matched = frame_id == completed.matched_frame_id;
        if (matched) {
          matched_sequence = sequence;
        }
        records.push_back({
          {"sequence", sequence},
          {"ring_slot", slot},
          {"frame_id", frame_id},
          {"analysis_generation", analysis_generation},
          {"domain_tag", domain_tag},
          {"transaction_token", token},
          {"observation_timestamp_us", join_u64(
            word(record_word_e::observation_timestamp_low),
            word(record_word_e::observation_timestamp_high)
          )},
          {"submission_class", {
            {"value", word(record_word_e::submission_class)},
            {"name", gpu_trace_submission_class_name(word(record_word_e::submission_class))},
          }},
          {"depth_disposition", {
            {"value", word(record_word_e::depth_disposition)},
            {"name", gpu_trace_depth_disposition_name(word(record_word_e::depth_disposition))},
          }},
          {"expected_work", {
            {"value", word(record_word_e::expected_work)},
            {"name", gpu_trace_expected_work_name(word(record_word_e::expected_work))},
          }},
          {"subtitle_disposition", {
            {"value", word(record_word_e::subtitle_disposition)},
            {"name", gpu_trace_subtitle_disposition_name(word(record_word_e::subtitle_disposition))},
          }},
          {"host_subtitle_outcome", {
            {"value", word(record_word_e::host_subtitle_outcome)},
            {"name", gpu_trace_host_outcome_name(word(record_word_e::host_subtitle_outcome))},
          }},
          {"flags", {
            {"value", flags},
            {"input_domain_reset", (flags & input_domain_reset) != 0u},
            {"dump_forced_at_enqueue", (flags & dump_forced) != 0u},
            {"ocr_record_submitted", (flags & ocr_record_submitted) != 0u},
            {"subtitle_suppressed", (flags & subtitle_suppressed) != 0u},
            {"condition_executed", condition_was_executed},
            {"condition_executed_host_proven", condition_executed_host_proven},
            {"subtitle_branch_gated", branch_gated},
          }},
          {"analysis_source", {
            {"width", word(record_word_e::source_width)},
            {"height", word(record_word_e::source_height)},
          }},
          {"field", {
            {"width", word(record_word_e::field_width)},
            {"height", word(record_word_e::field_height)},
          }},
          {"matched_dump_frame", matched},
          {"subtitle_plane", {
            {"schema", locator_word(0u)}, {"tag", locator_word(1u)},
            {"flags", {{"value", plane_flags}, {"current_valid", (plane_flags & subtitle_plane_current_valid_flag) != 0u}}},
            {"current_slot_mask", slot_mask}, {"current_count", std::popcount(slot_mask)},
            {"target", target}, {"target_bits", target_bits},
            {"analysis_generation", join_u64(locator_word(subtitle_plane_analysis_generation_offset), locator_word(subtitle_plane_analysis_generation_offset + 1u))},
            {"frame_id", join_u64(locator_word(subtitle_plane_frame_offset), locator_word(subtitle_plane_frame_offset + 1u))},
            {"observation_timestamp_us", join_u64(locator_word(subtitle_plane_observation_timestamp_offset), locator_word(subtitle_plane_observation_timestamp_offset + 1u))},
            {"adaptive", {
              {"initialized", locator_word(subtitle_plane_adaptive_offset)},
              {"flags", locator_word(subtitle_plane_adaptive_offset + 1u)},
              {"current_level", locator_word(subtitle_plane_adaptive_offset + 2u)},
              {"current_entry_requirement_level", locator_word(subtitle_plane_adaptive_offset + 2u)},
              {"presented_active", (locator_word(subtitle_plane_adaptive_offset + 1u) & subtitle_plane_adaptive_presented_flag) != 0u},
              {"smoothing_time_constant_ms", subtitle_plane_smoothing_time_constant_ms},
              {"slew_per_eye_source_pixels_per_second", subtitle_plane_slew_source_pixels_per_second},
              {"retreat_time_constant_ms", subtitle_plane_retreat_time_constant_ms},
              {"accepted_observation_timestamp_us", join_u64(locator_word(subtitle_plane_adaptive_offset + 4u), locator_word(subtitle_plane_adaptive_offset + 5u))},
              {"approach_since_us", join_u64(locator_word(subtitle_plane_adaptive_offset + 6u), locator_word(subtitle_plane_adaptive_offset + 7u))},
              {"retreat_since_us", join_u64(locator_word(subtitle_plane_adaptive_offset + 8u), locator_word(subtitle_plane_adaptive_offset + 9u))},
              {"approach_level", locator_word(subtitle_plane_adaptive_offset + 10u)},
              {"retreat_level", locator_word(subtitle_plane_adaptive_offset + 11u)},
              {"display_budget", std::bit_cast<float>(locator_word(subtitle_plane_adaptive_offset + 12u))},
              {"covered_count", locator_word(subtitle_plane_adaptive_offset + 13u)},
              {"conflict_count", locator_word(subtitle_plane_adaptive_offset + 14u)},
              {"invalid_count", locator_word(subtitle_plane_adaptive_offset + 15u)},
            }},
          }},
          {"subtitle_condition", suppressed ?
            nlohmann::json {
              {"active", false},
              {"unused", true},
              {"reason", "subtitle-suppressed transaction freezes UIP4 and skips conditioning"},
            } :
            nlohmann::json {
              {"active", condition_active},
              {"valid", condition_valid},
              {"executed", condition_was_executed},
              {"held_with_depth", subtitle_held_with_depth},
              {"unused", false},
              {"schema", condition_word(0u)},
              {"tag", condition_word(1u)},
              {"current_count", condition_word(2u)},
              {"current_kinds", condition_word(3u)},
              {"active_level", condition_word(4u)},
              {"target", condition_target},
              {"target_bits", condition_target_bits},
            }},
          {"transaction_words", gpu_trace_word_array(
            ring,
            base + word_index(record_word_e::transaction_begin),
            transaction_word_count
          )},
          {"subtitle_plane_words", gpu_trace_word_array(
            ring,
            base + word_index(record_word_e::subtitle_locator_begin),
            subtitle_locator_word_count
          )},
          {"subtitle_condition_words", gpu_trace_word_array(
            ring,
            base + word_index(record_word_e::subtitle_condition_begin),
            models::depth_coordinate_v2::subtitle_condition_param_word_count
          )},
        });
      }
      return {
        {"schema", decoded_trace_schema},
        {"role", "decoded diagnostic history; no rendering authority"},
        {"ring", {
          {"schema", ring_schema}, {"tag", ring_tag}, {"capacity", capacity},
          {"record_words", record_word_count}, {"next_sequence", next_sequence},
          {"next_slot", next_slot}, {"committed_count", count},
          {"oldest_sequence", oldest_sequence},
        }},
        {"capture_match", {
          {"matched_frame_id", completed.matched_frame_id},
          {"analysis_generation", completed.depth_input_region.analysis_generation},
          {"source_width", completed.depth_input_region.width()},
          {"source_height", completed.depth_input_region.height()},
          {"field_width", completed.field_width},
          {"field_height", completed.field_height},
          {"sequence", matched_sequence},
        }},
        {"records", std::move(records)},
      };
    }

    struct gpu_trace_publication_t {
      bool available = false;
      std::string raw_sha256;
      std::string decoded_sha256;
      std::string contract_sha256;
      nlohmann::json summary = nullptr;
    };

    gpu_trace_publication_t publish_gpu_trace_artifacts(
      const captured_dump_job &job,
      const std::filesystem::path &directory
    ) {
      gpu_trace_publication_t result;
      if (job.gpu_trace_ring.empty() || !job.completed.gpu_trace_provenance ||
          !gpu_trace_shader_identity_matches_contract(
            job.completed.gpu_trace_provenance
          ) || !detail::gpu_trace_ring_is_canonical(
            job.gpu_trace_ring, job.completed
          )) {
        return result;
      }
      const auto raw_path = directory / "gpu_trace_ring.u32";
      const auto decoded_path = directory / "gpu_trace.json";
      const auto contract_path = directory / "gpu_trace_contract.json";
      const auto decoded = gpu_trace_decoded_document(
        job.gpu_trace_ring, job.completed
      );
      const auto contract = gpu_trace_contract_document(
        *job.completed.gpu_trace_provenance
      );
      const auto remove_optional_artifacts = [&]() {
        std::error_code ignored;
        std::filesystem::remove(raw_path, ignored);
        ignored.clear();
        std::filesystem::remove(decoded_path, ignored);
        ignored.clear();
        std::filesystem::remove(contract_path, ignored);
      };
      if (!write_bytes(
            raw_path, job.gpu_trace_ring.data(), job.gpu_trace_ring.size()
          ) || !write_json(decoded_path, decoded) ||
          !write_json(contract_path, contract)) {
        remove_optional_artifacts();
        return result;
      }
      result.raw_sha256 = models::file_sha256_hex(raw_path);
      result.decoded_sha256 = models::file_sha256_hex(decoded_path);
      result.contract_sha256 = models::file_sha256_hex(contract_path);
      if (result.raw_sha256.empty() || result.decoded_sha256.empty() ||
          result.contract_sha256.empty()) {
        remove_optional_artifacts();
        return {};
      }
      result.available = true;
      result.summary = {
        {"available", true},
        {"required", false},
        {"rendering_authority", false},
        {"raw_artifact", "gpu_trace_ring.u32"},
        {"decoded_artifact", "gpu_trace.json"},
        {"contract_artifact", "gpu_trace_contract.json"},
        {"record_count", decoded.at("ring").at("committed_count")},
        {"oldest_sequence", decoded.at("ring").at("oldest_sequence")},
        {"next_sequence", decoded.at("ring").at("next_sequence")},
        {"matched_sequence", decoded.at("capture_match").at("sequence")},
        {"source_closure_sha256",
          job.completed.gpu_trace_provenance->source_closure_sha256},
      };
      return result;
    }

    nlohmann::json window_region_document(
      const window_region_snapshot &region
    ) {
      std::ostringstream hwnd;
      hwnd << "0x" << std::hex << std::uppercase << region.hwnd;
      return {
        {"schema", window_region_schema},
        {"capture", "same matched source/color/depth/render frame as the parent Dump 3D package"},
        {"role", "matched-window region provenance; no independent geometry or renderer authority"},
        {"authority_kind", window_region_authority_kind_name(region.authority_kind)},
        {"matched_frame_id", region.matched_frame_id},
        {"coordinate_space", {
          {"name", "matched-source-pixels"},
          {"rect_semantics", "half-open [left, top, right, bottom)"},
          {"source_extent_px", {
            {"width", region.source_width},
            {"height", region.source_height},
          }},
          {"capture_rect_px", {
            {"left", region.left},
            {"top", region.top},
            {"right", region.right},
            {"bottom", region.bottom},
          }},
        }},
        {"identity", {
          {"hwnd", hwnd.str()},
          {"process_id", region.process_id},
          {"document_id", region.document_id},
          {"video_id", region.video_id},
          {"generation", region.generation},
        }},
        {"freshness", {
          {"latest_observation_age_ms_at_capture", region.latest_observation_age_ms_at_capture},
          {"maximum_observation_age_ms", region.maximum_observation_age_ms},
          {"geometry_continuity_ms_at_capture", region.geometry_continuity_ms_at_capture},
          {"source_content_age_ms_at_capture", region.source_content_age_ms_at_capture},
          {"fresh", true},
          {"causal_geometry", true},
        }},
      };
    }

    bool scalar_tensor_snapshot_matches(
      const texture_snapshot &snapshot,
      const std::uint32_t width,
      const std::uint32_t height
    ) noexcept {
      return snapshot.desc.Width == width && snapshot.desc.Height == height &&
             snapshot.desc.Format == DXGI_FORMAT_R32_FLOAT;
    }

    bool bgra_mask_snapshot_matches(
      const texture_snapshot &snapshot,
      const std::uint32_t width,
      const std::uint32_t height
    ) noexcept {
      return snapshot.desc.Width == width && snapshot.desc.Height == height &&
             (snapshot.desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM ||
              snapshot.desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB);
    }

    nlohmann::json source_rect_document(
      const std::uint32_t left,
      const std::uint32_t top,
      const std::uint32_t right,
      const std::uint32_t bottom
    ) {
      return {
        {"left", left},
        {"top", top},
        {"right", right},
        {"bottom", bottom},
      };
    }

    float video_region_vertical_slope_source_u_per_source_v(
      const models::depth_input_region_t &region
    ) {
      const float source_width = static_cast<float>(region.source_width);
      const float source_height = static_cast<float>(region.source_height);
      const float roi_left = static_cast<float>(region.left) / source_width;
      const float roi_top = static_cast<float>(region.top) / source_height;
      const float roi_right = static_cast<float>(region.right) / source_width;
      const float roi_bottom = static_cast<float>(region.bottom) / source_height;
      const float roi_width = roi_right - roi_left;
      const float roi_height = roi_bottom - roi_top;
      const float source_height_in_source_u =
        source_height / std::max(source_width, 1.0f);
      const float roi_pixel_aspect =
        (roi_width * source_width) /
        std::max(roi_height * source_height, 1.0e-6f);
      const float vertical_slope =
        models::depth_coordinate_v2::max_vertical_shear * roi_pixel_aspect *
        (static_cast<float>(region.tensor_content.height()) /
         std::max(static_cast<float>(region.tensor_content.width()), 1.0f));
      return vertical_slope * source_height_in_source_u;
    }

    nlohmann::json depth_input_region_document(const frame &completed) {
      const auto &region = completed.depth_input_region;
      const bool roi = region.is_video_region();
      nlohmann::json authorization = nullptr;
      if (roi) {
        const auto &provenance = *completed.window_region;
        std::ostringstream hwnd;
        hwnd << "0x" << std::hex << std::uppercase << provenance.hwnd;
        authorization = {
          {"authority_kind", window_region_authority_kind_name(provenance.authority_kind)},
          {"observer_generation", provenance.generation},
          {"hwnd", hwnd.str()},
          {"process_id", provenance.process_id},
          {"document_id", provenance.document_id},
          {"video_id", provenance.video_id},
        };
      }
      const nlohmann::json semantic_rect = roi ?
        source_rect_document(
          static_cast<std::uint32_t>(completed.window_region->left),
          static_cast<std::uint32_t>(completed.window_region->top),
          static_cast<std::uint32_t>(completed.window_region->right),
          static_cast<std::uint32_t>(completed.window_region->bottom)
        ) :
        nlohmann::json(nullptr);
      const nlohmann::json outside = roi ?
        nlohmann::json {
          {"construction", "signed soft-threshold collar"},
          {"horizontal_slope_source_u_per_source_u",
           models::depth_coordinate_v2::max_horizontal_slope},
          {"vertical_slope_source_u_per_source_v",
           video_region_vertical_slope_source_u_per_source_v(region)},
          {"beyond_collar", "exact zero parallax"},
        } :
        nlohmann::json(nullptr);
      // Match the actual b2/HLSL path bit-for-bit in operation order. The live shader subtracts
      // two normalized float edges; serializing integer-width/source-width as a double would
      // describe a subtly different embedding for most nonzero-left rectangles.
      const float field_to_source_scale = roi ?
        static_cast<float>(region.right) / static_cast<float>(region.source_width) -
          static_cast<float>(region.left) / static_cast<float>(region.source_width) :
        1.0f;
      return {
        {"schema", 4},
        {"capture", "same matched source/color/model/depth/render frame as the parent Dump 3D package"},
        {"role", "authoritative analysis-domain placement and live-render embedding contract"},
        {"matched_frame_id", completed.matched_frame_id},
        {"mode", roi ? "window-region" : "full-source"},
        {"authorization", authorization},
        {"coordinate_space", {
          {"name", "matched-source-pixels"},
          {"rect_semantics", "half-open [left, top, right, bottom)"},
          {"source_extent_px", {
            {"width", region.source_width},
            {"height", region.source_height},
          }},
          {"semantic_rect_px", semantic_rect},
          {"inference_rect_px", source_rect_document(
            region.left,
            region.top,
            region.right,
            region.bottom
          )},
        }},
        {"analysis", {
          {"analysis_generation", region.analysis_generation},
          {"input_domain_reset", completed.input_domain_reset},
          {"tensor_extent_px", {
            {"width", completed.model_width},
            {"height", completed.model_height},
          }},
          {"tensor_content_rect_px", source_rect_document(
            region.tensor_content.left,
            region.tensor_content.top,
            region.tensor_content.right,
            region.tensor_content.bottom
          )},
          {"padded_area_fraction",
           roi ? completed.depth_video_plan->padded_area_fraction : 0.0f},
          {"fit_method", roi ?
            "centered-integer-contain-with-edge-replicated-excluded-padding" :
            "full-tensor"},
          {"crop_method", roi ?
            "direct retained-source rectangle sampling" : "full-source"},
          {"scene_analysis_domain", roi ?
            "inference-rectangle-only" : "full-source"},
        }},
        {"renderer", {
          {"final_parallax_units", roi ?
            "roi-local-source-u" : "full-source-u"},
          {"full_source_parallax_scale", field_to_source_scale},
          {"inside_inference_rect", "no taper"},
          {"outside", outside},
          {"source_sampling", "full matched source; never clamp to inference rectangle"},
          {"inverse_iterations", 11},
        }},
      };
    }

    std::string timestamp_string() {
      char text[32] = "unknown";
      const std::time_t now = std::time(nullptr);
      std::tm local {};
      if (localtime_s(&local, &now) == 0) {
        std::strftime(text, sizeof(text), "%Y%m%d_%H%M%S", &local);
      }
      return text;
    }

    struct output_paths {
      std::filesystem::path temporary;
      std::filesystem::path final;
    };

    bool make_output_paths(
      const std::filesystem::path &root,
      output_paths &paths,
      std::error_code &error
    ) {
      static std::atomic<std::uint64_t> sequence {0};
      const auto ticks = static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
      );
      for (unsigned attempt = 0; attempt < 128u; ++attempt) {
        const std::uint64_t serial =
          sequence.fetch_add(1u, std::memory_order_relaxed);
        std::ostringstream suffix;
        suffix << timestamp_string() << '_' << GetCurrentProcessId() << '_'
               << std::hex << ticks << '_' << serial;
        paths.final = root / ("dump_" + suffix.str());
        paths.temporary = root / (".partial_" + suffix.str());
        error.clear();
        if (
          !std::filesystem::exists(paths.final, error) && !error &&
          std::filesystem::create_directory(paths.temporary, error) && !error
        ) {
          return true;
        }
        if (error) {
          return false;
        }
      }
      return false;
    }

  }  // namespace

  namespace detail {

    std::string gpu_trace_contract_json(
      const models::host_sbs_gpu_trace_provenance_t &provenance
    ) {
      return gpu_trace_contract_document(provenance).dump();
    }

    std::string gpu_trace_decoded_json(
      const std::vector<std::uint8_t> &ring,
      const frame &completed
    ) {
      return gpu_trace_decoded_document(ring, completed).dump();
    }

  }  // namespace detail

  namespace detail {

  dump_publish_result publish_captured_dump(const captured_dump_job &job) {
    dump_publish_result result;
    const auto &completed = job.completed;
    const bool video_region = completed.depth_input_region.is_video_region();
    const auto &cfg = job.cfg;
    const auto &capture_preprocess = *job.preprocess;
    const bool hdr =
      completed.color_space == models::input_color_space::scrgb_hdr;
    const bool by_button = job.by_button;
    const bool by_file = job.by_file;
    const auto &model_identity = *completed.raw_model_provenance;
    nlohmann::json composite_runtime_provenance = nullptr;
    if (completed.composite_depth_runtime_provenance) {
      const auto &runtime = *completed.composite_depth_runtime_provenance;
      composite_runtime_provenance = {
        {"schema", models::prod_zipdepth_convex2x::contract_schema},
        {"runtime", "dav2_zipdepth_convex2x_composite"},
        {"model", runtime.model},
        {"onnx_sha256", runtime.onnx_sha256},
        {"embedded_dav2_onnx_sha256", runtime.embedded_dav2_onnx_sha256},
        {"zipdepth_checkpoint_sha256", runtime.zipdepth_checkpoint_sha256},
        {"guidance_preprocess_source_closure_sha256",
         runtime.guidance_preprocess_source_closure_sha256},
        {"engine_recipe", runtime.engine_recipe},
        {"engine_artifact", runtime.engine_artifact},
        {"active_engine_manifest", runtime.active_engine_manifest},
      };
    }

    const std::filesystem::path &trigger = job.trigger;
    std::error_code error;
    output_paths paths;
    bool success = false;
    try {
      const auto &normalization = job.normalization;
      const bool depth_health_state_available =
        job.depth_health_state_available;

      if (
        !std::filesystem::create_directories(job.root, error) && error
      ) {
        BOOST_LOG(warning) << "SBS debug dump: cannot create root "sv
                           << job.root.string() << ": " << error.message();
        result.error = error.message();
        return result;
      }
      if (!make_output_paths(job.root, paths, error)) {
        BOOST_LOG(warning) << "SBS debug dump: cannot reserve a unique output folder in "sv
                           << job.root.string()
                           << (error ? ": " + error.message() : "."s);
        result.error = error ? error.message() : "cannot reserve output folder";
        return result;
      }

      do {
        const auto &source = job.source;
        const auto &depth_input_source = completed.depth_input_region.is_video_region() ?
                                           job.depth_input_source :
                                           job.source;
        const auto &sbs = job.sbs;
        const bool subtitle_plane_active =
          !job.subtitle_ocr_record.empty() &&
          !job.subtitle_locator_state.empty();
        const nlohmann::json subtitle_conditioning = subtitle_plane_active ?
          nlohmann::json {
            {"mode", "subtitle-contours-v1"},
            {"request", true},
            {"producer", subtitle_ocr_producer_contract_json(
              *completed.parallax_v2_shader_provenance
            )},
            {"resolver", subtitle_plane_resolver_contract_json(
              *completed.parallax_v2_shader_provenance
            )},
            {"artifacts", {
              {"ocr_record", "subtitle_ocr_record.u32"},
              {"locator_state", "subtitle_locator_state.u32"},
              {"base_field", "shadow_base_final_parallax.f32"},
              {"conditioned_field", "shadow_final_parallax.f32"},
              {"contour_support", "contour_support.u32"},
              {"subtitle_glyph_matte", "subtitle_glyph_matte.u32"},
              {"subtitle_condition_params", "subtitle_condition_params.u32"},
              {"contour_field_pin", "contour_field_pin.u32"},
              {"contour_distance", "contour_distance.f32"},
              {"contour_identity", "contour_identity.u32"},
              {"contour_history", "contour_history.rgba32f"},
              {"contour_source", "contour_source.rgba16f"},
              {"ocr_probability", "ocr_probability.f32"},
              {"gaussian_weight", "gaussian_weight.f32"},
              {"gaussian_blend", "gaussian_blend.f32"},
              {"gaussian_vertical_conditioned", "gaussian_vertical_conditioned.f32"},
              {"gaussian_control", "gaussian_control.u32"},
            }},
          } :
          nlohmann::json {
            {"mode", "none"},
            {"request", nullptr},
            {"producer", nullptr},
            {"resolver", nullptr},
            {"artifacts", nlohmann::json::object()},
          };
        const nlohmann::json glyph_matte_descriptor {
          {"schema", subtitle_glyph_matte_schema}, {"artifact", "subtitle_glyph_matte.u32"},
          {"recipe", nlohmann::json::parse(subtitle_glyph_matte_policy_json)},
          {"identity_artifact", "contour_identity.u32"}, {"source_artifact", "contour_source.rgba16f"},
          {"source_support_artifact", "contour_support.u32"},
        };
        const nlohmann::json condition_descriptor {
          {"schema", subtitle_condition_param_schema}, {"artifact", "subtitle_condition_params.u32"},
          {"word_count", models::depth_coordinate_v2::subtitle_condition_param_word_count}, {"identity_artifact", "contour_identity.u32"},
        };
        const auto region_error = depth_input_region_error(completed);
        if (!region_error.empty()) {
          BOOST_LOG(warning) << "SBS debug dump: depth input region rejected: "sv
                             << region_error << '.';
          break;
        }
        if (
          source.desc.Width != completed.depth_input_region.source_width ||
          source.desc.Height != completed.depth_input_region.source_height ||
          depth_input_source.desc.Width != completed.depth_input_region.width() ||
          depth_input_source.desc.Height != completed.depth_input_region.height() ||
          depth_input_source.desc.Format != source.desc.Format
        ) {
          BOOST_LOG(warning)
            << "SBS debug dump: full-source or analysis-source texture extent/format does "sv
               "not match the authenticated depth input region."sv;
          break;
        }
        const auto tensor_width = static_cast<std::uint32_t>(completed.field_width);
        const auto tensor_height = static_cast<std::uint32_t>(completed.field_height);
        if (
          !scalar_tensor_snapshot_matches(
            job.shadow_coordinate,
            tensor_width,
            tensor_height
          ) ||
          !scalar_tensor_snapshot_matches(
            job.shadow_candidate,
            tensor_width,
            tensor_height
          ) ||
          !scalar_tensor_snapshot_matches(
            job.shadow_vertical,
            tensor_width,
            tensor_height
          ) ||
          !scalar_tensor_snapshot_matches(
            job.shadow_vertical_conditioned,
            tensor_width,
            tensor_height
          ) ||
          (subtitle_plane_active && !scalar_tensor_snapshot_matches(
            job.shadow_base_final,
            tensor_width,
            tensor_height
          )) ||
          !scalar_tensor_snapshot_matches(job.shadow_final, tensor_width, tensor_height)
        ) {
          BOOST_LOG(warning)
            << "SBS debug dump: a completed parallax-v2 tensor texture does not match "sv
               "the authenticated model extent/R32_FLOAT contract."sv;
          break;
        }
        const std::array<std::pair<const char *, const std::vector<std::uint8_t> *>, 13u> contour_artifacts {{
          {"contour_support.u32", &job.contour_support.bytes},
          {"subtitle_glyph_matte.u32", &job.subtitle_glyph_matte.bytes},
          {"subtitle_condition_params.u32", &job.subtitle_condition_params},
          {"contour_field_pin.u32", &job.contour_field_pin.bytes},
          {"contour_distance.f32", &job.contour_distance.bytes},
          {"contour_identity.u32", &job.contour_identity},
          {"contour_history.rgba32f", &job.contour_history.bytes},
          {"contour_source.rgba16f", &job.contour_source.bytes},
          {"ocr_probability.f32", &job.ocr_probability},
          {"gaussian_weight.f32", &job.gaussian_weight.bytes},
          {"gaussian_blend.f32", &job.gaussian_blend.bytes},
          {"gaussian_vertical_conditioned.f32", &job.gaussian_vertical_conditioned.bytes},
          {"gaussian_control.u32", &job.gaussian_control},
        }};
        std::array<std::string, 13u> contour_sha256 {};
        if (subtitle_plane_active) {
          bool contours_valid = job.contour_identity.size() == subtitle_contour_history_word_count * sizeof(std::uint32_t) &&
            job.ocr_probability.size() == subtitle_ocr_output_width * subtitle_ocr_output_height * sizeof(float);
          contours_valid = contours_valid && detail::subtitle_contour_texture_layout_is_authenticated(
            detail::subtitle_contour_texture_e::support, job.contour_support.desc.Format,
            job.contour_support.desc.Width, job.contour_support.desc.Height, job.contour_support.row_bytes, job.contour_support.bytes.size(), completed);
          contours_valid = contours_valid && detail::subtitle_contour_texture_layout_is_authenticated(
            detail::subtitle_contour_texture_e::glyph_matte, job.subtitle_glyph_matte.desc.Format,
            job.subtitle_glyph_matte.desc.Width, job.subtitle_glyph_matte.desc.Height, job.subtitle_glyph_matte.row_bytes, job.subtitle_glyph_matte.bytes.size(), completed);
          contours_valid = contours_valid && job.subtitle_condition_params.size() == models::depth_coordinate_v2::subtitle_condition_param_word_count * sizeof(std::uint32_t);
          contours_valid = contours_valid && detail::subtitle_contour_texture_layout_is_authenticated(
            detail::subtitle_contour_texture_e::field_pin, job.contour_field_pin.desc.Format,
            job.contour_field_pin.desc.Width, job.contour_field_pin.desc.Height, job.contour_field_pin.row_bytes, job.contour_field_pin.bytes.size(), completed);
          contours_valid = contours_valid && detail::subtitle_contour_texture_layout_is_authenticated(
            detail::subtitle_contour_texture_e::distance, job.contour_distance.desc.Format,
            job.contour_distance.desc.Width, job.contour_distance.desc.Height, job.contour_distance.row_bytes, job.contour_distance.bytes.size(), completed);
          contours_valid = contours_valid && detail::subtitle_contour_texture_layout_is_authenticated(
            detail::subtitle_contour_texture_e::history, job.contour_history.desc.Format,
            job.contour_history.desc.Width, job.contour_history.desc.Height, job.contour_history.row_bytes, job.contour_history.bytes.size(), completed);
          contours_valid = contours_valid && detail::subtitle_contour_texture_layout_is_authenticated(
            detail::subtitle_contour_texture_e::source, job.contour_source.desc.Format,
            job.contour_source.desc.Width, job.contour_source.desc.Height, job.contour_source.row_bytes, job.contour_source.bytes.size(), completed);
          contours_valid = contours_valid && detail::subtitle_contour_texture_layout_is_authenticated(
            detail::subtitle_contour_texture_e::gaussian_weight, job.gaussian_weight.desc.Format,
            job.gaussian_weight.desc.Width, job.gaussian_weight.desc.Height, job.gaussian_weight.row_bytes, job.gaussian_weight.bytes.size(), completed);
          contours_valid = contours_valid && detail::subtitle_contour_texture_layout_is_authenticated(
            detail::subtitle_contour_texture_e::gaussian_blend, job.gaussian_blend.desc.Format,
            job.gaussian_blend.desc.Width, job.gaussian_blend.desc.Height, job.gaussian_blend.row_bytes, job.gaussian_blend.bytes.size(), completed);
          contours_valid = contours_valid && detail::subtitle_contour_texture_layout_is_authenticated(
            detail::subtitle_contour_texture_e::gaussian_vertical_conditioned, job.gaussian_vertical_conditioned.desc.Format,
            job.gaussian_vertical_conditioned.desc.Width, job.gaussian_vertical_conditioned.desc.Height, job.gaussian_vertical_conditioned.row_bytes, job.gaussian_vertical_conditioned.bytes.size(), completed);
          contours_valid = contours_valid && job.gaussian_control.size() == sizeof(std::uint32_t);
          if (!contours_valid || !detail::subtitle_contour_identity_matches_completion(
                job.contour_identity, job.subtitle_ocr_record, job.subtitle_locator_state, job.shadow_state, completed) ||
              !detail::subtitle_contour_clear_policy_is_authenticated(
                job.contour_identity, job.subtitle_ocr_record, job.shadow_state, job.contour_support.bytes,
                job.contour_field_pin.bytes, job.contour_distance.bytes, job.contour_history.bytes,
                job.contour_source.bytes, job.ocr_probability) ||
              !detail::subtitle_gaussian_fields_are_authenticated(
                job.contour_identity, job.subtitle_locator_state, job.subtitle_ocr_record, job.contour_distance.bytes,
                job.gaussian_weight.bytes, job.gaussian_blend.bytes, job.gaussian_vertical_conditioned.bytes,
                job.gaussian_control, job.shadow_base_final.bytes, job.shadow_final.bytes) ||
              !detail::subtitle_analytic_tuple_is_authenticated(
                job.contour_identity, job.subtitle_locator_state, job.subtitle_ocr_record,
                job.contour_support.bytes, job.contour_source.bytes, job.subtitle_glyph_matte.bytes,
                job.subtitle_condition_params)) {
            BOOST_LOG(warning) << "SBS debug dump: joined contour tuple ABI, identity, or abstention policy is invalid."sv;
            break;
          }
          bool written = true;
          for (const auto &[name, bytes] : contour_artifacts) {
            written = written && write_bytes(paths.temporary / name, bytes->data(), bytes->size());
          }
          written = written && write_json(paths.temporary / "subtitle_glyph_matte.json", glyph_matte_descriptor) &&
            write_json(paths.temporary / "subtitle_condition_params.json", condition_descriptor) &&
            write_bytes(paths.temporary / "render_source.bin", source.bytes.data(), source.bytes.size());
          if (!written) break;
        }
        if (sbs.desc.Width < 2u || (sbs.desc.Width & 1u) != 0u) {
          BOOST_LOG(warning)
            << "SBS debug dump: packed output width is not an even two-eye extent."sv;
          break;
        }
        if (
          !write_color_preview(
            paths.temporary / "source.png",
            source,
            completed.color_space
          ) ||
          !write_color_preview(
            paths.temporary / "depth_input_source.png",
            depth_input_source,
            completed.color_space
          ) ||
          !write_json(
            paths.temporary / "depth_input_region.json",
            depth_input_region_document(completed)
          ) ||
          !write_json(
            paths.temporary / "subtitle_conditioning.json",
            subtitle_conditioning
          ) ||
          (subtitle_plane_active && (
            !write_bytes(
              paths.temporary / "subtitle_ocr_record.u32",
              job.subtitle_ocr_record.data(),
              job.subtitle_ocr_record.size()
            ) ||
            !write_bytes(
              paths.temporary / "subtitle_locator_state.u32",
              job.subtitle_locator_state.data(),
              job.subtitle_locator_state.size()
            )
          )) ||
          !dump_model_input(
            job.model_input,
            completed.model_width,
            completed.model_height,
            paths.temporary,
            capture_preprocess
          )
        ) {
          break;
        }

        raw_depth_dump_stats raw_stats;
        if (
          !dump_raw_depth(
            job.raw_depth,
            completed.raw_width,
            completed.raw_height,
            paths.temporary,
            raw_stats
          ) ||
          !write_color_preview(
            paths.temporary / "sbs.png",
            sbs,
            completed.color_space
          )
        ) {
          break;
        }

        nlohmann::json adaptive = nullptr;
        bool adaptive_available = false;
        if (depth_health_state_available) {
          try {
            adaptive_available = dump_adaptive_state(
              job.adaptive_state,
              normalization,
              completed,
              paths.temporary,
              adaptive
            );
          } catch (...) {
            // Depth-health state evidence is optional and has no live V2 geometry authority.
          }
        }
        // The completion ring is optional diagnostic evidence. Missing resources, a torn ring,
        // provenance mismatch, or publication failure omits only these three files; it must never
        // reject the authenticated core Dump 3D package.
        const auto gpu_trace = publish_gpu_trace_artifacts(
          job, paths.temporary
        );
        const bool held_subtitle_evidence = subtitle_plane_active &&
          detail::subtitle_u64(job.subtitle_locator_state, subtitle_plane_frame_offset) !=
            completed.matched_frame_id;
        if (subtitle_plane_active && held_subtitle_evidence &&
            !gpu_trace.available) {
          BOOST_LOG(warning)
            << "SBS debug dump: held OCR8/UIP4 evidence requires its authenticated "sv
               "GPU trace artifact; publication aborted."sv;
          break;
        }
        // Bind provenance to the exact bytes that were successfully written into this
        // transaction directory. Hashing the files (rather than reconstructed vectors) also
        // covers byte order and the canonical JSON serialization of the input-shape contract.
        const std::string raw_depth_sha256 = models::file_sha256_hex(
          paths.temporary / "raw_depth.f32"
        );
        const std::string model_input_sha256 = models::file_sha256_hex(
          paths.temporary / "model_input.f32"
        );
        const std::string model_input_shape_sha256 = models::file_sha256_hex(
          paths.temporary / "model_input_shape.json"
        );
        const std::string depth_input_region_sha256 = models::file_sha256_hex(
          paths.temporary / "depth_input_region.json"
        );
        const std::string subtitle_conditioning_sha256 = models::file_sha256_hex(
          paths.temporary / "subtitle_conditioning.json"
        );
        const std::string subtitle_ocr_record_sha256 = subtitle_plane_active ?
          models::file_sha256_hex(paths.temporary / "subtitle_ocr_record.u32") :
          std::string {};
        const std::string subtitle_locator_state_sha256 = subtitle_plane_active ?
          models::file_sha256_hex(paths.temporary / "subtitle_locator_state.u32") :
          std::string {};
        bool contour_hashes_complete = true;
        if (subtitle_plane_active) {
          for (std::size_t index = 0u; index < contour_artifacts.size(); ++index) {
            contour_sha256[index] = models::file_sha256_hex(paths.temporary / contour_artifacts[index].first);
            contour_hashes_complete = contour_hashes_complete && !contour_sha256[index].empty();
          }
        }
        if (!contour_hashes_complete || raw_depth_sha256.empty() || model_input_sha256.empty() ||
            model_input_shape_sha256.empty() || depth_input_region_sha256.empty() ||
            subtitle_conditioning_sha256.empty() ||
            (subtitle_plane_active && (
              subtitle_ocr_record_sha256.empty() ||
              subtitle_locator_state_sha256.empty()
            ))) {
          BOOST_LOG(warning)
            << "SBS debug dump: cannot hash the exact raw-model/domain artifacts; "sv
               "publication aborted."sv;
          break;
        }

        const auto &shadow_coordinate = job.shadow_coordinate;
        const auto &shadow_candidate = job.shadow_candidate;
        const auto &shadow_vertical = job.shadow_vertical;
        const auto &shadow_vertical_conditioned = job.shadow_vertical_conditioned;
        const auto &shadow_base_final = job.shadow_base_final;
        const auto &shadow_final = job.shadow_final;
        nlohmann::json shadow_summary = nullptr;
        if (
           !dump_shadow_float_texture(
             shadow_coordinate,
             paths.temporary,
             "shadow_coordinate"
           ) ||
           !dump_shadow_float_texture(
             shadow_candidate,
             paths.temporary,
             "shadow_candidate_parallax"
           ) ||
           !dump_shadow_float_texture(
             shadow_vertical,
             paths.temporary,
             "shadow_vertical_majorant"
           ) ||
           !dump_shadow_float_texture(
             shadow_vertical_conditioned,
             paths.temporary,
             "shadow_vertical_conditioned"
           ) ||
           (subtitle_plane_active && !dump_shadow_float_texture(
             shadow_base_final,
             paths.temporary,
             "shadow_base_final_parallax"
           )) ||
           !dump_shadow_float_texture(
             shadow_final,
             paths.temporary,
             "shadow_final_parallax"
           ) ||
           !dump_parallax_v2_state(
             completed,
             job.shadow_state,
             job.shadow_frame_stats,
             paths.temporary,
             shadow_summary
           )
        ) {
          break;
        }

        // Mapping and boundary attribution form one diagnostic transaction. A partial optional
        // pass is omitted rather than publishing evidence that the current reader cannot join.
        const bool warp_map_available =
          job.warp_map_available && job.warp_mask_available;
        const bool warp_mask_available = warp_map_available;
        warp_map_dump_stats warp_map_stats;
        const auto &warp_map = job.warp_map;
        const auto &warp_mask = job.warp_mask;
        if (
          (warp_map_available &&
           !scalar_tensor_snapshot_matches(
             warp_map,
             sbs.desc.Width,
             sbs.desc.Height
           )) ||
          (warp_mask_available &&
           !bgra_mask_snapshot_matches(
             warp_mask,
             sbs.desc.Width,
             sbs.desc.Height
           ))
        ) {
          BOOST_LOG(warning)
            << "SBS debug dump: dump-only map/mask resources do not match the packed "sv
               "output extent or format; publication aborted."sv;
          break;
        }
        if (
          warp_map_available &&
          (!dump_warp_map(
             warp_map,
             source.desc.Width,
             source.desc.Height,
             paths.temporary,
             warp_map_stats
           ))
        ) {
          break;
        }
        if (
          warp_mask_available &&
          (!write_color_preview(
             paths.temporary / "warp_mask.png",
             warp_mask,
             models::input_color_space::srgb
           ))
        ) {
          break;
        }
        const std::string warp_map_sha256 = warp_map_available ?
          models::file_sha256_hex(paths.temporary / "warp_map.f32") : std::string {};
        const std::string warp_mask_sha256 = warp_mask_available ?
          models::file_sha256_hex(paths.temporary / "warp_mask.png") : std::string {};
        if (completed.depth_input_region.is_video_region()) {
          if (!warp_map_available || !warp_mask_available) {
            BOOST_LOG(warning)
              << "SBS debug dump: ROI publication requires a full-source inverse map and "sv
                 "boundary mask; publication aborted."sv;
            break;
          }
          if (warp_map_sha256.empty() || warp_mask_sha256.empty()) {
            BOOST_LOG(warning)
              << "SBS debug dump: ROI inverse-map evidence could not be hashed; "sv
                 "publication aborted."sv;
            break;
          }
        } else if ((warp_map_available && warp_map_sha256.empty()) ||
                   (warp_mask_available && warp_mask_sha256.empty())) {
          BOOST_LOG(warning)
            << "SBS debug dump: optional inverse-map evidence could not be hashed; "sv
               "publication aborted."sv;
          break;
        }

        if (
          sbs.desc.Width < 2u || (sbs.desc.Width & 1u) != 0u ||
          source.desc.Width == 0u || source.desc.Height == 0u ||
          sbs.desc.Height == 0u
        ) {
          break;
        }
        bool window_region_available = false;
        std::string window_region_sha256;
        if (completed.window_region) {
          const auto validation = validate_window_region(
            *completed.window_region,
            completed.matched_frame_id,
            source.desc.Width,
            source.desc.Height
          );
          if (validation == window_region_error::none) {
            window_region_available = write_json(
              paths.temporary / "window_region.json",
              window_region_document(*completed.window_region)
            );
            if (!window_region_available) {
              BOOST_LOG(warning)
                << "SBS debug dump: optional matched-frame window-region provenance could not be written; continuing without it."sv;
            } else if (completed.depth_input_region.is_video_region()) {
              window_region_sha256 = models::file_sha256_hex(
                paths.temporary / "window_region.json"
              );
            }
          } else {
            BOOST_LOG(warning)
              << "SBS debug dump: optional window-region provenance rejected ("sv
              << window_region_error_name(validation)
              << "); continuing without it."sv;
          }
        }
        if (
          completed.depth_input_region.is_video_region() &&
          (!window_region_available || window_region_sha256.empty())
        ) {
          BOOST_LOG(warning)
            << "SBS debug dump: ROI publication requires exact matched-frame window-region "sv
               "provenance; publication aborted."sv;
          break;
        }
        const std::uint32_t eye_width = sbs.desc.Width / 2u;
        const std::uint32_t eye_height = sbs.desc.Height;
        const float source_aspect =
          static_cast<float>(source.desc.Width) /
          static_cast<float>(source.desc.Height);
        const float eye_aspect =
          static_cast<float>(eye_width) / static_cast<float>(eye_height);
        const float content_scale_x =
          eye_aspect > source_aspect ? source_aspect / eye_aspect : 1.0f;
        const float content_scale_y =
          eye_aspect < source_aspect ? eye_aspect / source_aspect : 1.0f;

        // Bind every V2 geometry field to the exact bytes written into this transaction
        // directory. Metadata-only descriptors let a truncated or internally inconsistent
        // geometry dump validate cleanly, which silently poisons every downstream offline
        // investigation that trusts validated dumps.
        const std::string shadow_coordinate_sha256 =
          models::file_sha256_hex(paths.temporary / "shadow_coordinate.f32");
        const std::string shadow_candidate_sha256 =
          models::file_sha256_hex(paths.temporary / "shadow_candidate_parallax.f32");
        const std::string shadow_vertical_majorant_sha256 =
          models::file_sha256_hex(paths.temporary / "shadow_vertical_majorant.f32");
        const std::string shadow_vertical_conditioned_sha256 =
          models::file_sha256_hex(paths.temporary / "shadow_vertical_conditioned.f32");
        const std::string shadow_base_final_sha256 = subtitle_plane_active ?
          models::file_sha256_hex(
            paths.temporary / "shadow_base_final_parallax.f32"
          ) :
          std::string {};
        const std::string shadow_final_sha256 =
          models::file_sha256_hex(paths.temporary / "shadow_final_parallax.f32");
        const std::string shadow_state_sha256 =
          models::file_sha256_hex(paths.temporary / "shadow_state.json");
        const std::string shadow_frame_stats_sha256 =
          models::file_sha256_hex(paths.temporary / "shadow_frame_stats.json");
        if (shadow_coordinate_sha256.empty() ||
            shadow_candidate_sha256.empty() ||
            shadow_vertical_majorant_sha256.empty() ||
            shadow_vertical_conditioned_sha256.empty() || shadow_final_sha256.empty() ||
            shadow_state_sha256.empty() || shadow_frame_stats_sha256.empty() ||
            (subtitle_plane_active && shadow_base_final_sha256.empty())) {
          BOOST_LOG(warning)
            << "SBS debug dump: failed to hash a written V2 geometry field; dump rejected."sv;
          break;
        }
        nlohmann::json artifacts = nlohmann::json::object();
        write_actual_draw_evidence(job, paths.temporary, artifacts);
        artifacts["source.png"] = artifact_description(
          true,
          true,
          "captured color supplied to Host SBS",
          "Preview decoded according to the matched frame's declared transfer: direct sRGB, linear-SDR OETF, or HDR tone map plus OETF."
        );
        artifacts["depth_input_source.png"] = artifact_description(
          true,
          false,
          "model-depth input source preview",
          "Spatially exact full-source preview or post-completion reconstruction of the logical ROI source. Live ROI preprocessing samples the retained full matched texture directly; this transfer-aware PNG is diagnostic only, while model_input.f32 is numeric authority."
        );
        artifacts["depth_input_region.json"] = hashed_artifact_description(
          true,
          true,
          "depth analysis input region",
          "Authoritative full-source placement, crop-local analysis domain, and live-render embedding for every model/depth/parallax artifact in this matched package.",
          depth_input_region_sha256
        );
        artifacts["subtitle_conditioning.json"] = hashed_artifact_description(
          true,
          true,
          "subtitle conditioning authority",
          "Canonical current-schema subtitle authority descriptor: none or exact joined OCR8/UIP4/native contour tuple.",
          subtitle_conditioning_sha256
        );
        if (subtitle_plane_active) {
          artifacts["subtitle_ocr_record.u32"] = hashed_artifact_description(
            true,
            true,
            "OCR8 subtitle boxes for atomic target",
            "Exact 256-word little-endian OCR8 record with a 208-word current core/cover prefix and a 48-word layout-learning history tail, bound to the atomic target's publication frame (current authenticated observation or held ordinary reuse), source clock, analysis generation, source, field, content, and reference. Layout history never authorizes pixels by itself.",
            subtitle_ocr_record_sha256
          );
          artifacts["subtitle_locator_state.u32"] = hashed_artifact_description(
            true,
            true,
            "compact UIP4 current subtitle plane state",
            "Exact 40-word little-endian UIP4 current OCR8 slot authority, source/domain identity, and independent source-time adaptive UI controller state.",
            subtitle_locator_state_sha256
          );
          constexpr std::array<const char *, 13u> contour_stages {{
            "native current subtitle contour support",
            "native completed subtitle glyph matte",
            "joined subtitle analytic plane condition params",
            "projected subtitle contour field pins",
            "native support rectangle completion budget in parallax pixels",
            "SCT5 subtitle temporal-mask identity",
            "native subtitle contour history",
            "joined subtitle source display crop",
            "joined OCR detector probability",
            "bounded native support completion presentation weight",
            "direct bounded UI-Base completion proposal",
            "bit-exact subtitle proposal copy; compatibility vertical name",
            "actual completion smoothing-enabled control",
          }};
          for (std::size_t index = 0u; index < contour_artifacts.size(); ++index) {
            auto descriptor = hashed_artifact_description(
              true, true, contour_stages[index],
              "Exact joined completed subtitle contour evidence; raw little-endian payload.",
              contour_sha256[index]);
            descriptor["byte_count"] = contour_artifacts[index].second->size();
            artifacts[contour_artifacts[index].first] = std::move(descriptor);
          }
        }
        if (subtitle_plane_active) {
          for (const auto &[name, stage] : std::array<std::pair<const char *, const char *>, 3u> {{
                 {"subtitle_glyph_matte.json", "native glyph matte recipe metadata"},
                 {"subtitle_condition_params.json", "joined analytic plane params metadata"},
                 {"render_source.bin", "exact rendered native color source"},
               }}) {
            const auto digest = models::file_sha256_hex(paths.temporary / name);
            if (digest.empty()) { contour_hashes_complete = false; break; }
            auto descriptor = hashed_artifact_description(true, true, stage,
              "Exact immutable selected-map input; raw native pixels or versioned tuple metadata.", digest);
            descriptor["byte_count"] = std::filesystem::file_size(paths.temporary / name);
            artifacts[name] = std::move(descriptor);
          }
          if (!contour_hashes_complete) break;
        }
        artifacts["gpu_trace_ring.u32"] = gpu_trace.available ?
          hashed_artifact_description(
            true,
            false,
            "diagnostic GPU accepted-root completion history",
            "Raw little-endian 300-slot completion ring. This is diagnostic evidence only and has no rendering authority.",
            gpu_trace.raw_sha256
          ) :
          artifact_description(
            false,
            false,
            "diagnostic GPU accepted-root completion history",
            "Optional diagnostic ring was unavailable, untrusted, torn, or could not be published; the core dump remains complete."
          );
        artifacts["gpu_trace.json"] = gpu_trace.available ?
          hashed_artifact_description(
            true,
            false,
            "decoded diagnostic GPU history",
            "Chronological decoded records with raw transaction, UIP4, and condition words plus authenticated device dispositions.",
            gpu_trace.decoded_sha256
          ) :
          artifact_description(
            false,
            false,
            "decoded diagnostic GPU history",
            "Optional decoded trace is absent with the raw diagnostic ring."
          );
        artifacts["gpu_trace_contract.json"] = gpu_trace.available ?
          hashed_artifact_description(
            true,
            false,
            "diagnostic GPU trace wire contract",
            "Exact ring offsets, receipt ABI, enum values, embedded state schemas, and authenticated shader source closure.",
            gpu_trace.contract_sha256
          ) :
          artifact_description(
            false,
            false,
            "diagnostic GPU trace wire contract",
            "Optional trace contract is absent with the raw diagnostic ring."
          );
        artifacts["model_input.f32"] = hashed_artifact_description(
          true,
          true,
          "exact neural-network input",
          "Float32-le NCHW tensor after preprocessing and ImageNet normalization.",
          model_input_sha256
        );
        artifacts["model_input_shape.json"] = hashed_artifact_description(
          true,
          true,
          "model-input contract",
          "Calibrated preprocess dimensions, layout, channels, normalization, and producer stage.",
          model_input_shape_sha256
        );
        artifacts["raw_depth.f32"] = hashed_artifact_description(
          true,
          true,
          "exact model output",
          "Float32-le raw depth before normalization or temporal processing.",
          raw_depth_sha256
        );
        artifacts["adaptive_state.json"] = artifact_description(
          adaptive_available,
          false,
          "depth-health comparison state",
          adaptive_available ?
            "Comparison-only depth-health counters and normalization state; no live V2 geometry authority." :
            "Unavailable; depth-health evidence never gates an authenticated live V2 dump."
        );
        artifacts["window_region.json"] = video_region ?
          hashed_artifact_description(
            true,
            true,
            "matched-frame window region provenance",
            "Validated semantic rectangle, authority kind, stable identity, freshness, and causal provenance from which the authoritative ROI input region was planned; not renderer authority by itself.",
            window_region_sha256
          ) :
          artifact_description(
            window_region_available,
            false,
            "matched-frame window region provenance",
            "Validated half-open capture rectangle, source extent, authority kind, identity, and freshness; provenance only and never independent renderer authority."
          );
        artifacts["shadow_coordinate.f32"] = hashed_artifact_description(
          true,
          true,
          "parallax-v2 canonical coordinate diagnostic",
          "Exact float32-le unbounded canonical coordinate u; diagnostic only and never used by the live renderer.",
          shadow_coordinate_sha256
        );
        artifacts["shadow_candidate_parallax.f32"] = hashed_artifact_description(
          true,
          true,
          "parallax-v2 pre-limiter candidate displacement",
          video_region ?
            "Exact immutable signed one-eye ROI-local source-U before the spatial limiter; geometry evidence only and never renderer authority without depth_input_region embedding." :
            "Exact immutable signed one-eye full-source-U before the spatial limiter; geometry evidence only, never live render authority.",
          shadow_candidate_sha256
        );
        artifacts["shadow_vertical_majorant.f32"] = hashed_artifact_description(
          true,
          false,
          "parallax-v2 vertical shear-limiter intermediate",
          video_region ?
            "Exact signed one-eye ROI-local source-U for the least column-wise upper envelope v+ >= candidate with |dv+/dy| <= max_vertical_shear/content_width; crop-local diagnostic evidence only." :
            "Exact signed one-eye full-source-U for the least column-wise upper envelope v+ >= candidate with |dv+/dy| <= max_vertical_shear/content_width; diagnostic evidence only.",
          shadow_vertical_majorant_sha256
        );
        artifacts["shadow_vertical_conditioned.f32"] = hashed_artifact_description(
          true,
          false,
          "parallax-v2 orientation-selective vertical conditioner",
          video_region ?
            "Exact signed one-eye ROI-local source-U after the fixed 75/25 share of the column upper/lower envelopes; may raise or lower the crop-local candidate while preserving the vertical shear bound." :
            "Exact signed one-eye full-source-U after the fixed 75/25 share of the column upper/lower envelopes; may raise or lower candidate while preserving the vertical shear bound.",
          shadow_vertical_conditioned_sha256
        );
        if (subtitle_plane_active) {
          artifacts["shadow_base_final_parallax.f32"] = hashed_artifact_description(
            true,
            true,
            "ordinary post-limiter V2 field before UIP4 conditioning",
            video_region ?
              "Exact ordinary crop-local row-majorant field before UIP4; not renderer authority without current UIP4 conditioning and depth_input_region embedding." :
              "Exact ordinary full-source-U row-majorant field before UIP4; not selected renderer authority.",
            shadow_base_final_sha256
          );
        }
        artifacts["shadow_final_parallax.f32"] = hashed_artifact_description(
          true,
          true,
          "parallax-v2 independent protected final displacement diagnostic",
          video_region ?
            "Exact complete atomic one-eye ROI-local source-U field. Live renderer authority is this field together with depth_input_region embedding." :
            "Exact complete atomic one-eye full-source-U field; sole live V2 render position authority.",
          shadow_final_sha256
        );
        artifacts["shadow_state.json"] = hashed_artifact_description(
          true,
          true,
          "parallax-v2 shot calibration and attenuation state",
          "Independent typed state bound to the exact coordinate contract tag.",
          shadow_state_sha256
        );
        artifacts["shadow_frame_stats.json"] = hashed_artifact_description(
          true,
          true,
          "parallax-v2 current-frame moments",
          "Independent mean/std/min/max state bound to the exact coordinate contract tag.",
          shadow_frame_stats_sha256
        );
        artifacts["warp_map.f32"] = warp_map_available ?
          hashed_artifact_description(
            true,
            video_region,
            video_region ? "exact full-source inverse-warp mapping" :
                           "exact inverse-warp mapping",
            video_region ?
              "Raw normalized full-source U selected by production Reproject after crop-local field embedding and the outside-only zero-plane collar." :
              "Raw normalized source-U selected by the production Reproject function.",
            warp_map_sha256
          ) :
          artifact_description(
            false,
            false,
            "exact inverse-warp mapping",
            "Unavailable because the matching dump-only mapping pass could not be created."
          );
        artifacts["warp_mask.png"] = warp_mask_available ?
          hashed_artifact_description(
            true,
            video_region,
            "V2 boundary-extrapolation mask",
            "Red marks inverse samples outside the finite source interval that the live renderer clamps to the nearest boundary column; V2 has no internal owner selection or synthetic fill.",
            warp_mask_sha256
          ) :
          artifact_description(
            false,
            false,
            "V2 boundary-extrapolation mask",
            "Unavailable because the matching dump-only mask pass could not be created."
          );
        artifacts["sbs.png"] = artifact_description(
          true,
          true,
          "packed Host-SBS output",
          "Final packed stereo preview using the same matched-frame transfer handling as source.png."
        );
        artifacts["meta.txt"] = artifact_description(
          true,
          true,
          "human-readable summary",
          "Compact compatibility summary; dump_manifest.json is authoritative."
        );
        artifacts["dump_manifest.json"] = artifact_description(
          true,
          true,
          "package contract",
          "Authoritative dimensions, formats, settings, stage descriptions, and availability."
        );

        nlohmann::json dimensions {
          {"source", texture_description(source)},
          {"analysis_source", texture_description(depth_input_source)},
          {"model_input", {
                            {"width", completed.model_width},
                            {"height", completed.model_height},
                            {"channels", 3},
                            {"layout", "NCHW"},
                            {"dtype", "float32-le"},
                          }},
          {"raw_depth", {
                          {"width", completed.raw_width},
                          {"height", completed.raw_height},
                          {"format", "float32-le structured buffer"},
                        }},
          {"normalized_depth", nullptr},
          {"packed_sbs", texture_description(sbs)},
          {"eye", {
                    {"width", eye_width},
                    {"height", eye_height},
                  }},
          {"content_fit", {
                            {"scale_x", content_scale_x},
                            {"scale_y", content_scale_y},
                          }},
        };
        if (warp_map_available) {
          dimensions["warp_map"] = texture_description(warp_map);
        } else {
          dimensions["warp_map"] = nullptr;
        }
        if (warp_mask_available) {
          dimensions["warp_mask"] = texture_description(warp_mask);
        } else {
          dimensions["warp_mask"] = nullptr;
        }
        dimensions["shadow_coordinate"] = texture_description(shadow_coordinate);
        dimensions["shadow_candidate_parallax"] =
          texture_description(shadow_candidate);
        dimensions["shadow_vertical_majorant"] =
          texture_description(shadow_vertical);
        dimensions["shadow_vertical_conditioned"] =
          texture_description(shadow_vertical_conditioned);
        if (subtitle_plane_active) {
          dimensions["shadow_base_final_parallax"] =
            texture_description(shadow_base_final);
          dimensions["contour_support"] = texture_description(job.contour_support);
          dimensions["subtitle_glyph_matte"] = texture_description(job.subtitle_glyph_matte);
          dimensions["render_source"] = texture_description(source);
          dimensions["subtitle_condition_params"] = {{"word_count", models::depth_coordinate_v2::subtitle_condition_param_word_count}, {"format", "uint32-le structured buffer"}};
          dimensions["contour_field_pin"] = texture_description(job.contour_field_pin);
          dimensions["contour_distance"] = texture_description(job.contour_distance);
          dimensions["gaussian_weight"] = texture_description(job.gaussian_weight);
          dimensions["gaussian_blend"] = texture_description(job.gaussian_blend);
          dimensions["gaussian_vertical_conditioned"] = texture_description(job.gaussian_vertical_conditioned);
          dimensions["gaussian_control"] = {{"word_count", 1u}, {"format", "uint32-le structured buffer"}};
          dimensions["contour_history"] = texture_description(job.contour_history);
          dimensions["contour_source"] = texture_description(job.contour_source);
          dimensions["contour_identity"] = {
            {"word_count", subtitle_contour_history_word_count},
            {"format", "uint32-le structured buffer"},
          };
          dimensions["ocr_probability"] = {
            {"width", subtitle_ocr_output_width}, {"height", subtitle_ocr_output_height},
            {"format", "float32-le structured buffer"},
          };
        }
        dimensions["shadow_final_parallax"] = texture_description(shadow_final);

        const std::string color_mode =
          completed.color_space == models::input_color_space::scrgb_hdr  ? "scrgb_hdr" :
          completed.color_space == models::input_color_space::linear_sdr ? "linear_sdr" :
                                                                           "srgb";
        const std::string color_preview_transform =
          completed.color_space == models::input_color_space::scrgb_hdr ?
            "luminance-preserving diagnostic tone map then sRGB OETF" :
          completed.color_space == models::input_color_space::linear_sdr ?
            "sRGB OETF" :
            "none; values are already sRGB code values";
        const std::string trigger_source =
          by_button && by_file ? "button+file" : by_button ? "button" :
                                                             "file";
        const nlohmann::json shadow_shader_source =
          parallax_v2_shader_identity_json(
            *completed.parallax_v2_shader_provenance
          );
        const nlohmann::json warp_map_contract = warp_map_available ?
          nlohmann::json {
            {"available", true},
            {"schema", 3},
            {"artifact", "warp_map.f32"},
            {"width", warp_map_stats.width},
            {"height", warp_map_stats.height},
            {"eye_width", warp_map_stats.eye_width},
            {"eye_height", warp_map_stats.eye_height},
            {"source_width", source.desc.Width},
            {"source_height", source.desc.Height},
            {"content_scale_x", warp_map_stats.content_scale_x},
            {"content_scale_y", warp_map_stats.content_scale_y},
            {"dtype", "float32-le"},
            {"layout", "row-major"},
            {"channels", {"raw_reproject_source_u_normalized"}},
            {"validity", {
              {"content", "derive from content_scale_x/content_scale_y and packed output coordinate"},
              {"inverse", video_region ?
                "pre-clamp U from crop-local ordinary Base inverse with depth_input_region embedding" :
                "pre-clamp U from ordinary Base inverse"},
              {"mask", "warp_mask.png red marks finite-source boundary extrapolation; V2 has no internal owner or synthetic-fill path"},
            }},
            {"live_sample_source_u_normalized", "clamp(raw_reproject_source_u_normalized, 0, 1)"},
            {"derived_inverse_displacement_output_eye_px", "(raw_reproject_source_u_normalized - aspect_fitted_unwarped_source_u) * content_scale_x * eye_width"},
            {"derived_signed_binocular_disparity_px", "invert both eye maps at common source-U samples; x_right - x_left"},
          } :
          nlohmann::json {
            {"available", false},
            {"schema", 3},
            {"artifact", nullptr},
          };
        auto selected_map = nlohmann::json::parse(models::depth_coordinate_v2::selected_map_contract_json);
        selected_map["render_enabled"] = completed.subtitle_render_enabled;
        nlohmann::json manifest {
          {"schema", 52},
          {"capture", "one matched, completed Host-SBS frame"},
          {"capture_status", "complete"},
          {"published_atomically", true},
          {"host_sbs_mode", "ai"},
          {"trigger", trigger_source},
          {"matched_frame_id", completed.matched_frame_id},
          {"matched_observation_timestamp_us", completed.matched_observation_timestamp_us},
          {"depth_model", completed.depth_model},
          {"composite_runtime_provenance", composite_runtime_provenance},
          {"color_mode", color_mode},
          {"color_preview_transform", color_preview_transform},
          {"hdr_preview", hdr ? color_preview_transform : "not applied"},
          {"cuda_graph_active", completed.cuda_graph_active},
          {"float_previews", {
            {"packaged", false},
            {"generator", "tools/sbsbench/generate_dump_previews.py"},
            {"normalization", "finite p2-p98 computed on demand"},
          }},
          {"renderer", {
                         {"authority", video_region ?
                           "authenticated ordinary Base inverse with depth-input-region embedding" :
                           "authenticated-ordinary-base-inverse"},
                         {"parallax_v2_render_requested", true},
                         {"parallax_v2_render_selected", true},
                         {"mapping_artifacts_match_selected_renderer", !completed.actual_draw.shared_pin_selected && warp_map_available && warp_mask_available},
                         {"actual_draw_renderer", {
                           {"schema", 2u},
                           {"kind", completed.actual_draw.shared_pin_selected ? completed.actual_draw.program : "ordinary-base-v1"},
                           {"source_closure_sha256", completed.actual_draw.shared_pin_selected ? completed.actual_draw.shared_pin_source_closure : std::string {}},
                           {"mapping_role", completed.actual_draw.shared_pin_selected ? "independent Base diagnostic; does not describe rendered shared warp" : "authenticated Base inverse diagnostic"},
                           {"evidence_artifact", "actual_draw_evidence.json"},
                         }},
                         {"parallax_v2_position_field", video_region ? "ordinary Base inverse + depth_input_region embedding" : "ordinary Base inverse"},
                         {"parallax_v2_coordinate_role", "shadow_coordinate is diagnostic only; it has no renderer authority"},
                         {"parallax_v2_vertical_majorant_role", "least column-wise upper envelope v+ >= candidate with adjacent-row source-U change <= max_vertical_shear/content_width; diagnostic evidence only"},
                         {"parallax_v2_vertical_conditioned_role", "fixed 75/25 share of column upper/lower envelopes; may raise or lower candidate and feeds the row majorant"},
                         {"parallax_v2_conditioner_role", "ordinary Base retains its existing slope bounds; protected Final is an independent diagnostic; shared pin field alone adds subtitle displacement"},
                         {"parallax_v2_inverse", "at-most-11-step authenticated Base inverse; no subtitle selector or synthetic fill"},
                         {"collar_defocus", nlohmann::json {
                              {"enabled", false},
                              {"role", "disabled after live hand-boundary halo regression; live color uses one linear sample at the inverse-warped coordinate"},
                              {"kernel", "none"},
                              {"hdr", "native source sample; no clamp, tone map, or gamma conversion"},
                            }},
                         {"live_shader_source", nlohmann::json {
                              {"source_closure_schema", models::host_sbs_shader_cache::source_closure_schema},
                              {"source_compile_flags", models::host_sbs_shader_cache::shader_compile_flags},
                              {"source_macro_count", 0u},
                              {"source_closure_sha256", completed.parallax_v2_live_renderer_source_closure_sha256},
                              {"source_file", std::string {models::host_sbs_shader_cache::parallax_v2_live_renderer.filename}},
                              {"entrypoint", std::string {models::host_sbs_shader_cache::parallax_v2_live_renderer.entrypoint}},
                              {"target", std::string {models::host_sbs_shader_cache::parallax_v2_live_renderer.target}},
                              {"diagnostic_source_closure_sha256", std::string {models::host_sbs_shader_cache::parallax_v2_live_diagnostic_source_closure_sha256}},
                              {"mapping_source_file", std::string {models::host_sbs_shader_cache::parallax_v2_live_mapping.filename}},
                              {"mapping_entrypoint", std::string {models::host_sbs_shader_cache::parallax_v2_live_mapping.entrypoint}},
                              {"mask_source_file", std::string {models::host_sbs_shader_cache::parallax_v2_live_mask.filename}},
                              {"mask_entrypoint", std::string {models::host_sbs_shader_cache::parallax_v2_live_mask.entrypoint}},
                            }},
                       }},
          {"dimensions", std::move(dimensions)},
          {"warp_map_contract", warp_map_contract},
          {"selected_map", std::move(selected_map)},
          {"final_parallax", {
            {"contract_schema",
             models::depth_coordinate_v2::final_parallax_contract_schema},
            {"artifact", "shadow_final_parallax.f32"},
            {"warp_input_artifact", nullptr},
            {"authority", std::string {
              models::depth_coordinate_v2::final_parallax_authority
            }},
            {"publication_policy", std::string {
              models::depth_coordinate_v2::final_parallax_publication_policy
            }},
            {"reuse_policy", std::string {
              models::depth_coordinate_v2::final_parallax_reuse_policy
            }},
            {"invalid_policy", std::string {
              models::depth_coordinate_v2::final_parallax_invalid_policy
            }},
            {"current_rgb_policy", std::string {
              models::depth_coordinate_v2::final_parallax_current_rgb_policy
            }},
            {"warp_relation", "independent diagnostic; ordinary mapping consumes immutable Base only"},
          }},
          {"normalization", adaptive_available ?
             nlohmann::json {
               {"role", "comparison-only depth-health state; no live V2 geometry authority"},
               {"effective_lower", normalization.lower},
               {"effective_upper", normalization.upper},
               {"initialized", normalization.initialized > 0.5f},
               {"initialized_value", normalization.initialized},
               {"frame_state", normalization_frame_state_name(normalization.frame_state)},
               {"frame_state_value", normalization.frame_state},
             } : nlohmann::json {nullptr}},
          {"raw_depth_statistics", {
                                     {"finite_count", raw_stats.finite_count},
                                     {"sample_count", static_cast<std::uint64_t>(raw_stats.width) * raw_stats.height},
                                      {"finite_fraction", static_cast<double>(raw_stats.finite_count) / (static_cast<double>(raw_stats.width) * raw_stats.height)},
                                      {"minimum", raw_stats.minimum},
                                      {"maximum", raw_stats.maximum},
                                    }},
          {"adaptive_summary", adaptive_available ? adaptive["decoded"] :
                                                     nlohmann::json {nullptr}},
          {"depth_input_region", {
            {"available", true},
            {"artifact", "depth_input_region.json"},
            {"mode", video_region ? "window-region" : "full-source"},
            {"geometry_authority", true},
            {"renderer_authority", true},
          }},
          {"window_region", {
            {"available", window_region_available},
            {"artifact", window_region_available ?
              nlohmann::json("window_region.json") : nlohmann::json(nullptr)},
            {"observer_status", completed.window_region_observer_status},
            {"mapping_status", completed.window_region_mapping_status},
            {"geometry_authority", false},
            {"renderer_authority", false},
          }},
          {"subtitle_conditioning", subtitle_conditioning},
          {"gpu_trace", gpu_trace.available ?
            gpu_trace.summary :
            nlohmann::json {
              {"available", false},
              {"required", false},
              {"rendering_authority", false},
              {"raw_artifact", nullptr},
              {"decoded_artifact", nullptr},
              {"contract_artifact", nullptr},
            }},
          {"parallax_v2_shadow", {
                                    {"requested", false},
                                    {"active", true},
                                    {"rendered_output_selected", true},
                                    {"shader_source", shadow_shader_source},
                                    {"state", std::move(shadow_summary)},
                                  }},
          {"config", config_json(
                       cfg,
                       completed,
                       completed.depth_model,
                       completed.raw_model_provenance->depth_model_url
                     )},
          {"artifacts", std::move(artifacts)},
        };
        manifest[std::string {
            models::depth_coordinate_v2::capture_provenance_manifest_key
          }] = {
            {"schema", models::depth_coordinate_v2::capture_provenance_schema},
            {"binding", std::string {
                          models::depth_coordinate_v2::capture_provenance_binding
                        }},
            {"depth_model", model_identity.depth_model},
            {"depth_model_url", model_identity.depth_model_url},
            {"onnx_sha256", model_identity.onnx_sha256},
            {"preprocess_profile", model_identity.preprocess_profile},
            {"preprocess_source_closure_sha256",
             model_identity.preprocess_source_closure_sha256},
            {"raw_depth_sha256", raw_depth_sha256},
            {"model_input_sha256", model_input_sha256},
            {"model_input_shape_sha256", model_input_shape_sha256},
          };

        std::ostringstream meta;
        meta.imbue(std::locale::classic());
        meta << std::setprecision(std::numeric_limits<float>::max_digits10)
             << "depth_model=" << completed.depth_model << '\n'
             << "color_mode=" << color_mode << '\n'
             << "trigger=" << trigger_source << '\n'
             << "matched_frame_id=" << completed.matched_frame_id << '\n'
             << "source_width=" << source.desc.Width << '\n'
             << "source_height=" << source.desc.Height << '\n'
             << "depth_input_mode="
             << (video_region ? "window-region" : "full-source") << '\n'
             << "depth_input_left=" << completed.depth_input_region.left << '\n'
             << "depth_input_top=" << completed.depth_input_region.top << '\n'
             << "depth_input_right=" << completed.depth_input_region.right << '\n'
             << "depth_input_bottom=" << completed.depth_input_region.bottom << '\n'
             << "depth_input_analysis_generation="
             << completed.depth_input_region.analysis_generation << '\n'
             << "depth_input_domain_reset="
             << (completed.input_domain_reset ? "true" : "false") << '\n'
             << "depth_input_region_sha256=" << depth_input_region_sha256 << '\n'
             << "packed_sbs_width=" << sbs.desc.Width << '\n'
             << "packed_sbs_height=" << sbs.desc.Height << '\n'
             << "eye_width=" << eye_width << '\n'
             << "eye_height=" << eye_height << '\n'
             << "model_input_width=" << completed.model_width << '\n'
             << "model_input_height=" << completed.model_height << '\n'
             << "raw_depth_width=" << raw_stats.width << '\n'
             << "raw_depth_height=" << raw_stats.height << '\n'
             << "raw_depth_finite_fraction="
             << static_cast<double>(raw_stats.finite_count) /
                  (static_cast<double>(raw_stats.width) * raw_stats.height)
             << '\n'
             << "raw_depth_min=" << raw_stats.minimum << '\n'
             << "raw_depth_max=" << raw_stats.maximum << '\n'
             << "depth_health_diagnostics_available="
             << (adaptive_available ? "true" : "false") << '\n'
             << "normalization_effective_lower=" << normalization.lower << '\n'
             << "normalization_effective_upper=" << normalization.upper << '\n'
             << "normalization_initialized=" << normalization.initialized << '\n'
             << "normalization_frame_state=" << normalization.frame_state << '\n'
             << "cuda_graph_active="
             << (completed.cuda_graph_active ? "true" : "false") << '\n'
             << "final_parallax_artifact=shadow_final_parallax.f32\n"
             << "warp_map_available=" << (warp_map_available ? "true" : "false")
             << '\n'
             << "warp_mask_available=" << (warp_mask_available ? "true" : "false")
             << '\n'
             << "window_region_available="
             << (window_region_available ? "true" : "false") << '\n'
             << "window_region_observer_status="
             << completed.window_region_observer_status << '\n'
             << "window_region_mapping_status="
             << completed.window_region_mapping_status << '\n'
             << "parallax_v2_shadow_requested="
             << "false\n"
             << "parallax_v2_shadow_active=true\n"
             << "parallax_v2_render_requested=true\n"
             << "parallax_v2_render_selected=true\n"
             << "renderer_authority="
             << (video_region ?
                   "authenticated ordinary Base inverse with depth-input-region embedding" :
                   "authenticated-ordinary-base-inverse")
             << '\n'
             << "final_parallax_warp_relation=independent-diagnostic-no-render-authority\n"
             << "collar_defocus_enabled=false\n"
             << "collar_defocus_kernel=none\n"
             << "parallax_v2_live_renderer_source_closure_sha256="
             << completed.parallax_v2_live_renderer_source_closure_sha256 << '\n'
             << "raw_model_provenance=authoritative\n"
             << "raw_model_preprocess_profile="
             << model_identity.preprocess_profile
             << '\n'
             << "raw_model_preprocess_source_closure_sha256="
             << model_identity.preprocess_source_closure_sha256
             << '\n';
        const auto &shader_identity = *completed.parallax_v2_shader_provenance;
        meta << "parallax_v2_shader_source_closure_schema="
             << shader_identity.source_closure_schema << '\n'
             << "parallax_v2_shader_source_compile_flags="
             << shader_identity.source_compile_flags << '\n'
             << "parallax_v2_shader_source_macro_count="
             << shader_identity.source_macro_count << '\n'
             << "parallax_v2_shader_source_closure_sha256="
             << shader_identity.source_closure_sha256 << '\n';
        if (
          !write_text(paths.temporary / "meta.txt", meta.str()) ||
          !write_json(paths.temporary / "dump_manifest.json", manifest)
        ) {
          break;
        }

        error.clear();
        std::filesystem::rename(paths.temporary, paths.final, error);
        if (error) {
          BOOST_LOG(warning) << "SBS debug dump: atomic directory publication failed: "
                             << error.message();
          break;
        }
        success = true;
      } while (false);

      if (!success) {
        error.clear();
        std::filesystem::remove_all(paths.temporary, error);
        BOOST_LOG(warning)
          << "SBS debug dump failed; request retained for a rate-limited retry."sv;
        result.error = "artifact publication failed";
        return result;
      }
    } catch (const std::exception &exception) {
      try {
        if (!paths.temporary.empty()) {
          std::error_code cleanup_error;
          std::filesystem::remove_all(paths.temporary, cleanup_error);
        }
      } catch (...) {
      }
      try {
        BOOST_LOG(warning)
          << "SBS debug dump transaction threw; partial output removed and request retained: "
          << exception.what();
      } catch (...) {
      }
      result.error = exception.what();
      return result;
    } catch (...) {
      try {
        if (!paths.temporary.empty()) {
          std::error_code cleanup_error;
          std::filesystem::remove_all(paths.temporary, cleanup_error);
        }
      } catch (...) {
      }
      try {
        BOOST_LOG(warning)
          << "SBS debug dump transaction threw an unknown exception; partial output removed and request retained."sv;
      } catch (...) {
      }
      result.error = "unknown publication exception";
      return result;
    }

    if (by_file) {
      try {
        error.clear();
        std::filesystem::remove(trigger, error);
      } catch (...) {
        error = std::make_error_code(std::errc::not_enough_memory);
      }
      if (error) {
        result.trigger_remove_failed = true;
        try {
          BOOST_LOG(warning)
            << "SBS debug dump: could not remove dump.trigger; file polling disabled for this session: "
            << error.message();
        } catch (...) {
        }
      }
    }
    try {
      BOOST_LOG(info) << "SBS debug dump written to "sv << paths.final.string()
                      << " (model "sv << completed.depth_model << ", frame "
                      << completed.matched_frame_id << ')';
    } catch (...) {
    }
    result.success = true;
    result.published_path = paths.final;
    return result;
  }

  }  // namespace detail
}  // namespace platf::sbs_debug
