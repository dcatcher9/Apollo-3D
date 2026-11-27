/**
 * @file src/platform/windows/sbs_debug_dump_package.h
 * @brief Dump 3D internals: the CPU-owned copy of one matched frame and its package writer.
 */
#pragma once

// platform includes
#include <d3d11.h>

// standard includes
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

// local includes
#include "src/config.h"
#include "src/depth_coordinate_v2.h"
#include "sbs_debug_dump.h"
#include "sbs_debug_dump_validate.h"

namespace platf::sbs_debug {

  namespace detail {

    static_assert(
      std::endian::native == std::endian::little,
      "Dump contracts explicitly use little-endian float and integer words."
    );

    struct texture_snapshot {
      D3D11_TEXTURE2D_DESC desc {};
      std::vector<std::uint8_t> bytes;
      std::size_t row_bytes = 0;
    };

    // Schema 2 records only maintained bindings: no shader reads the retired t8 mask or the
    // t9/t11/b4 background roles.
    inline constexpr std::array<const char *, 10u> actual_draw_artifacts {{
      "actual_draw_M.u32", "actual_selected_receipt.u32",
      "actual_geometry_constants.u32", "actual_subtitle_constants.u32",
      "actual_shared_pin_weights.f32", "actual_shared_pin_field.f32", "actual_shared_pin_rows.u32",
      "actual_shared_pin_geometry.u32", "actual_shared_pin_map.rgba32f", "actual_shared_pin_boundary.rgba32f",
    }};
    inline constexpr std::size_t actual_draw_texture_count = 6u, actual_draw_buffer_count = 4u;
    constexpr bool actual_draw_is_texture(std::size_t index) {
      return index == 0u || (index >= 4u && index <= 6u) || index >= 8u;
    }
    constexpr std::size_t actual_draw_storage_index(std::size_t index) {
      return index == 0u ? 0u : index < 4u ? index - 1u : index < 7u ? index - 3u : index == 7u ? 3u : index - 4u;
    }
    static_assert([] {
      std::array<bool, actual_draw_texture_count> textures {};
      std::array<bool, actual_draw_buffer_count> buffers {};
      for (std::size_t index = 0u; index < actual_draw_artifacts.size(); ++index) {
        const auto storage = actual_draw_storage_index(index);
        const bool texture = actual_draw_is_texture(index);
        if (storage >= (texture ? textures.size() : buffers.size())) return false;
        bool &slot = texture ? textures[storage] : buffers[storage];
        if (slot) return false;
        slot = true;
      }
      return true;
    }());

    /**
     * CPU-owned copy of one authenticated render pair. D3D11 resources are read while the
     * immediate context is still owned by the render thread; the background publisher never
     * touches the device or context and therefore cannot race the live pipeline.
     */
    struct captured_dump_job {
      std::filesystem::path root;
      std::filesystem::path trigger;
      std::shared_ptr<std::atomic<bool>> button_request;
      bool by_button = false;
      bool by_file = false;
      frame completed;
      config::video_t::sbs_t cfg;
      const models::depth_coordinate_v2::model_preprocess_contract_t *preprocess = nullptr;

      normalization_state normalization {};
      bool depth_health_state_available = false;
      std::vector<std::uint8_t> adaptive_state;

      texture_snapshot source;
      texture_snapshot depth_input_source;
      std::vector<float> model_input;
      std::vector<float> raw_depth;
      texture_snapshot sbs;
      texture_snapshot shadow_coordinate;
      texture_snapshot shadow_candidate;
      texture_snapshot shadow_vertical;
      texture_snapshot shadow_vertical_conditioned;
      texture_snapshot shadow_base_final;
      texture_snapshot shadow_final;
      std::vector<std::uint8_t> subtitle_ocr_record;
      std::vector<std::uint8_t> subtitle_locator_state;
      texture_snapshot contour_support;
      texture_snapshot subtitle_glyph_matte;
      std::vector<std::uint8_t> subtitle_condition_params;
      texture_snapshot contour_field_pin;
      texture_snapshot contour_distance;
      texture_snapshot gaussian_weight;
      texture_snapshot gaussian_blend;
      texture_snapshot gaussian_vertical_conditioned;
      texture_snapshot contour_history;
      texture_snapshot contour_source;
      std::vector<std::uint8_t> contour_identity;
      std::vector<std::uint8_t> ocr_probability;
      std::vector<std::uint8_t> gaussian_control;
      std::vector<std::uint8_t> gpu_trace_ring;
      std::vector<float> shadow_state;
      std::vector<float> shadow_frame_stats;
      bool warp_map_available = false;
      texture_snapshot warp_map;
      bool warp_mask_available = false;
      texture_snapshot warp_mask;
      std::array<texture_snapshot, actual_draw_texture_count> actual_draw_textures;
      std::array<std::vector<std::uint8_t>, actual_draw_buffer_count> actual_draw_buffers;
      std::array<bool, actual_draw_artifacts.size()> actual_draw_available {};
      std::array<std::string, actual_draw_artifacts.size()> actual_draw_unavailable;
    };

    struct dump_publish_result {
      bool success = false;
      bool trigger_remove_failed = false;
      std::filesystem::path published_path;
      std::string error;
    };

    /**
     * Write one complete package folder from CPU-owned bytes. Runs only on the publication
     * worker and never touches the D3D11 device or context.
     */
    dump_publish_result publish_captured_dump(const captured_dump_job &job);

  }  // namespace detail
}  // namespace platf::sbs_debug
