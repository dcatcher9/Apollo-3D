/**
 * @file src/host_sbs_shader_cache.h
 * @brief Process-wide bytecode cache for the fixed-shape Host SBS depth shaders.
 */
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace models::host_sbs_shader_cache {
  inline constexpr std::uint32_t source_closure_schema = 2u;
  inline constexpr std::uint32_t shader_compile_flags = 0x00008800u;

  struct shader_spec {
    std::string_view filename;
    std::string_view entrypoint = "main";
    std::string_view target = "cs_5_0";
  };

  inline constexpr shader_spec rgb_to_nchw {"rgb_to_nchw_cs.hlsl"};
  inline constexpr shader_spec buffer_to_tex {"buffer_to_tex_cs.hlsl"};
  inline constexpr shader_spec depth_ema_motion {"depth_ema_motion_cs.hlsl"};
  inline constexpr shader_spec depth_minmax {"depth_minmax_cs.hlsl"};
  inline constexpr shader_spec depth_minmax_ema {"depth_minmax_ema_cs.hlsl"};
  inline constexpr shader_spec depth_hist {"depth_hist_cs.hlsl"};
  inline constexpr shader_spec depth_subject_hist {"depth_subject_hist_cs.hlsl"};
  inline constexpr shader_spec depth_subject_resolve {"depth_subject_resolve_cs.hlsl"};
  inline constexpr shader_spec depth_valid_history {"depth_valid_history_cs.hlsl"};
  inline constexpr shader_spec depth_coordinate_v2_moments {"depth_coordinate_v2_moments_cs.hlsl"};
  inline constexpr shader_spec depth_coordinate_v2_frame_resolve {"depth_coordinate_v2_frame_resolve_cs.hlsl"};
  inline constexpr shader_spec depth_coordinate_v2_near_coverage {"depth_coordinate_v2_near_coverage_cs.hlsl"};
  inline constexpr shader_spec depth_coordinate_v2_state_resolve {"depth_coordinate_v2_state_resolve_cs.hlsl"};
  inline constexpr shader_spec depth_coordinate_v2_map {"depth_coordinate_v2_map_cs.hlsl"};
  inline constexpr shader_spec depth_coordinate_v2_vertical_limit {"depth_coordinate_v2_vertical_limit_cs.hlsl"};
  inline constexpr shader_spec depth_coordinate_v2_limit {"depth_coordinate_v2_limit_cs.hlsl"};
  inline constexpr shader_spec parallax_v2_live_renderer {
    "sbs_reprojection_v2_live_ps.hlsl", "main_ps", "ps_5_0"
  };
  inline constexpr shader_spec parallax_v2_live_mapping {
    "sbs_reprojection_v2_live_ps.hlsl", "mapping_ps", "ps_5_0"
  };
  inline constexpr shader_spec parallax_v2_live_mask {
    "sbs_reprojection_v2_live_ps.hlsl", "mask_ps", "ps_5_0"
  };
  inline constexpr std::string_view parallax_v2_live_renderer_source_closure_sha256 =
    "25061ab84be309d97910deffa8dc41ec31cbf95af8c9e5ef844ee0113a7b2b83";

  inline constexpr std::array preprocess_specs {
    rgb_to_nchw,
  };

  inline constexpr std::array core_specs {
    rgb_to_nchw,
    buffer_to_tex,
    depth_ema_motion,
    depth_minmax,
    depth_minmax_ema,
    depth_hist,
    depth_subject_hist,
    depth_subject_resolve,
    depth_valid_history,
  };

  // Kept out of core_specs on purpose: the v2 path is config-file-only and default-off. A normal
  // Host SBS startup must neither compile these experimental shaders nor inherit their failures.
  inline constexpr std::array parallax_v2_shadow_specs {
    depth_coordinate_v2_moments,
    depth_coordinate_v2_frame_resolve,
    depth_coordinate_v2_near_coverage,
    depth_coordinate_v2_state_resolve,
    depth_coordinate_v2_map,
    depth_coordinate_v2_vertical_limit,
    depth_coordinate_v2_limit,
  };

  inline constexpr std::array parallax_v2_live_renderer_specs {
    parallax_v2_live_renderer,
    parallax_v2_live_mapping,
    parallax_v2_live_mask,
  };

  struct source_snapshot;
  using source_snapshot_t = std::shared_ptr<const source_snapshot>;
  using bytecode_t = std::shared_ptr<const std::vector<unsigned char>>;

  /**
   * Own the selected roots and their authenticated quoted-include graph as one immutable source
   * snapshot. Only dependencies reachable from `specs` participate; unrelated optional shaders
   * cannot invalidate the fixed-shape Host SBS cache.
   */
  source_snapshot_t snapshot_sources(
    const std::filesystem::path &shader_root,
    std::span<const shader_spec> specs
  );

  /**
   * Standard lowercase SHA-256 of the snapshot's canonical root/spec/transitive-source closure.
   * The identity is stable across installation roots and changes when any reachable source,
   * resolved include edge, shader entrypoint/target, or compiler policy changes.
   */
  std::string source_closure_sha256(const source_snapshot_t &sources);

  /** Return cached bytecode, compiling this exact source snapshot once process-wide if needed. */
  bytecode_t get(
    const source_snapshot_t &sources,
    const shader_spec &spec
  );

  /** Precompile every shader consumed by the ordinary fixed-shape depth-estimator constructor. */
  bool prewarm(const std::filesystem::path &assets_dir);
}  // namespace models::host_sbs_shader_cache
