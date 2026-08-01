/**
 * @file src/host_sbs_shader_cache.h
 * @brief Process-wide bytecode cache for the fixed-shape Host SBS depth shaders.
 */
#pragma once

#include <array>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace models::host_sbs_shader_cache {
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

  struct source_snapshot;
  using source_snapshot_t = std::shared_ptr<const source_snapshot>;
  using bytecode_t = std::shared_ptr<const std::vector<unsigned char>>;

  /**
   * Hash the selected root shaders and their quoted includes into one immutable source identity.
   * Only dependencies reachable from `specs` participate; unrelated optional shaders cannot
   * invalidate the fixed-shape Host SBS cache.
   */
  source_snapshot_t snapshot_sources(
    const std::filesystem::path &shader_root,
    std::span<const shader_spec> specs
  );

  /** Return cached bytecode, compiling this exact source snapshot once process-wide if needed. */
  bytecode_t get(
    const source_snapshot_t &sources,
    const shader_spec &spec
  );

  /** Precompile every shader consumed by the ordinary fixed-shape depth-estimator constructor. */
  bool prewarm(const std::filesystem::path &assets_dir);
}  // namespace models::host_sbs_shader_cache
