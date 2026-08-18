/**
 * @file src/host_sbs_shader_cache.h
 * @brief Closure-keyed memory and persistent bytecode cache for fixed-shape Host SBS shaders.
 */
#pragma once

#include <array>
#include <cstddef>
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

  inline constexpr shader_spec buffer_to_tex {"buffer_to_tex_cs.hlsl"};
  inline constexpr shader_spec buffer_to_tex_pad {
    "buffer_to_tex_cs.hlsl", "pad_main", "cs_5_0"
  };
  inline constexpr shader_spec depth_minmax_ema {"depth_minmax_ema_cs.hlsl"};
  inline constexpr shader_spec depth_hist {"depth_hist_cs.hlsl"};
  inline constexpr shader_spec depth_scene_cut_evidence {"depth_scene_cut_evidence_cs.hlsl"};
  inline constexpr shader_spec depth_scene_cut_resolve {"depth_scene_cut_resolve_cs.hlsl"};
  inline constexpr shader_spec depth_coordinate_v2_moments {"depth_coordinate_v2_moments_cs.hlsl"};
  inline constexpr shader_spec depth_coordinate_v2_frame_resolve {"depth_coordinate_v2_frame_resolve_cs.hlsl"};
  inline constexpr shader_spec depth_coordinate_v2_state_resolve {"depth_coordinate_v2_state_resolve_cs.hlsl"};
  inline constexpr shader_spec depth_coordinate_v2_map {"depth_coordinate_v2_map_cs.hlsl"};
  inline constexpr shader_spec depth_coordinate_v2_ownership {
    "depth_coordinate_v2_ownership_cs.hlsl"
  };
  inline constexpr shader_spec depth_coordinate_v2_coordinate_diagnostic {
    "depth_coordinate_v2_map_cs.hlsl", "coordinate_main", "cs_5_0"
  };
  inline constexpr shader_spec depth_coordinate_v2_vertical_limit {"depth_coordinate_v2_vertical_limit_cs.hlsl"};
  inline constexpr shader_spec depth_coordinate_v2_limit {"depth_coordinate_v2_limit_cs.hlsl"};
  inline constexpr shader_spec host_sbs_ocr_preprocess {
    "host_sbs_ocr_preprocess_cs.hlsl", "main", "cs_5_0"
  };
  inline constexpr shader_spec host_sbs_ocr_cells {
    "host_sbs_ocr_boxes_cs.hlsl", "cells_main", "cs_5_0"
  };
  inline constexpr shader_spec host_sbs_ocr_resolve {
    "host_sbs_ocr_boxes_cs.hlsl", "resolve_main", "cs_5_0"
  };
  inline constexpr shader_spec host_sbs_subtitle_locator_resolve {
    "host_sbs_subtitle_locator_cs.hlsl", "resolve_main", "cs_5_0"
  };
  inline constexpr shader_spec host_sbs_subtitle_condition {
    "host_sbs_subtitle_locator_cs.hlsl", "condition_main", "cs_5_0"
  };
  inline constexpr shader_spec host_sbs_near_identical_fused_preprocess {
    "rgb_to_nchw_near_identical_cs.hlsl", "fused_main", "cs_5_0"
  };
  inline constexpr shader_spec host_sbs_near_identical_resolve {
    "host_sbs_near_identical_detector_cs.hlsl", "resolve_main", "cs_5_0"
  };
  inline constexpr shader_spec host_sbs_near_identical_scene_seed {
    "host_sbs_near_identical_detector_cs.hlsl", "scene_seed_main", "cs_5_0"
  };
  inline constexpr shader_spec host_sbs_near_identical_finalize {
    "host_sbs_near_identical_detector_cs.hlsl", "finalize_main", "cs_5_0"
  };
  inline constexpr shader_spec host_sbs_near_identical_reuse_depth {
    "host_sbs_near_identical_detector_cs.hlsl", "reuse_depth_main", "cs_5_0"
  };
  inline constexpr shader_spec host_sbs_gpu_trace {
    "host_sbs_gpu_trace_cs.hlsl", "main", "cs_5_0"
  };
  inline constexpr shader_spec parallax_v2_live_renderer {
    "sbs_reprojection_v2_live_ps.hlsl", "main_ps", "ps_5_0"
  };
  inline constexpr shader_spec parallax_v2_p010_y_renderer {
    "sbs_reprojection_v2_p010_y_ps.hlsl", "main_p010_y_ps", "ps_5_0"
  };
  inline constexpr shader_spec parallax_v2_live_mapping {
    "sbs_reprojection_v2_diagnostics_ps.hlsl", "mapping_ps", "ps_5_0"
  };
  inline constexpr shader_spec parallax_v2_live_mask {
    "sbs_reprojection_v2_diagnostics_ps.hlsl", "mask_ps", "ps_5_0"
  };
  // The shared fullscreen-triangle vertex shader produces the TexCoord that both Host SBS pixel
  // shaders invert; it co-defines live stereo geometry and therefore belongs to both pinned
  // closures below. Non-SBS presentation may still compile it unpinned.
  inline constexpr shader_spec sbs_reprojection_vertex {
    "sbs_reprojection_vs.hlsl", "main_vs", "vs_5_0"
  };
  inline constexpr shader_spec sbs_flat_identity {
    "sbs_flat_identity_ps.hlsl", "main_ps", "ps_5_0"
  };
  inline constexpr std::string_view parallax_v2_live_renderer_source_closure_sha256 =
    "7553350163dfb80cee85b70689b768a46a5c15b44997e2753e24bebedaa51ffd";
  inline constexpr std::string_view parallax_v2_p010_y_source_closure_sha256 =
    "f14993382c296ee07ce556c3dbebba29d021d8555ea446fae9a8b06b17a09813";
  inline constexpr std::string_view parallax_v2_diagnostic_source_closure_sha256 =
    "1c7cb433f667c990e4d55f254a11cc9a40590e3a1e97e62fc261c2e0069b9513";
  inline constexpr std::string_view sbs_flat_fallback_source_closure_sha256 =
    "7e45f7ca78b170c2d6c33ab5c5e20d9f45cece71a5c84e6e7fc4f0f42cfde8d4";
  inline constexpr std::string_view near_identical_detector_source_closure_sha256 =
    "87b63ffb753e9cc39e90b385ab997134402596eb5c074a4f3a3680619015910b";
  inline constexpr std::string_view gpu_trace_source_closure_sha256 =
    "a1e6a21286b5a898b1dcdd419056faa8e3a5351b21edef0ef288d716979c0cb9";

  // Identity-only minimal closure used to match the model/preprocess calibration. Production
  // bytecode is never compiled from this smaller snapshot: the sole fused entry point is compiled
  // with every other producer root from parallax_v2_producer_specs below.
  inline constexpr std::array preprocess_specs {
    host_sbs_near_identical_fused_preprocess,
  };

  // Complete production V2 producer set. The normalized depth is private scene-cut evidence; the
  // retired subject shaping, hard-mask sanitizer/exclusion, and adaptive-pop paths remain absent.
  // The OCR8 producer and compact SLR13 post-limit conditioner share this authenticated snapshot,
  // so no analysis or geometry pass can be sampled from a weaker closure. This array is the sole
  // canonical producer registry: its order is authenticated by the generated DVC contract and is
  // shared unchanged by prewarm, snapshotting, and compute-object creation.
  inline constexpr std::array parallax_v2_producer_specs {
    host_sbs_near_identical_fused_preprocess,
    buffer_to_tex,
    buffer_to_tex_pad,
    depth_minmax_ema,
    depth_hist,
    depth_scene_cut_evidence,
    depth_scene_cut_resolve,
    depth_coordinate_v2_moments,
    depth_coordinate_v2_frame_resolve,
    depth_coordinate_v2_state_resolve,
    depth_coordinate_v2_map,
    depth_coordinate_v2_ownership,
    depth_coordinate_v2_vertical_limit,
    depth_coordinate_v2_limit,
    host_sbs_ocr_preprocess,
    host_sbs_ocr_cells,
    host_sbs_ocr_resolve,
    host_sbs_subtitle_locator_resolve,
    host_sbs_subtitle_condition,
  };

  // The canonical-coordinate field is not a production input or output. Keep its alternate
  // entrypoint outside the authenticated live producer set so a normal frame never depends on
  // diagnostic shader availability.
  inline constexpr std::array parallax_v2_diagnostic_specs {
    depth_coordinate_v2_coordinate_diagnostic,
  };

  // Required GPU-only reuse arbitration. The mandatory producer entry points have already emitted
  // tile evidence from the current and authenticated-history NCHW tensors. This independent
  // closure resolves that evidence and owns receipt/postprocess gating. Authentication,
  // compilation, or resource failure is terminal fail-flat; post-bootstrap DAV2 is never
  // submitted outside the conditional wrapper.
  inline constexpr std::array near_identical_detector_specs {
    host_sbs_near_identical_resolve,
    host_sbs_near_identical_scene_seed,
    host_sbs_near_identical_finalize,
    host_sbs_near_identical_reuse_depth,
  };

  // Optional diagnostic-only completion trace. It is intentionally absent from every pinned
  // rendering/arbitration closure and from prewarm: source/resource failure only omits Dump 3D
  // trace evidence and can never reject the production Host SBS path.
  inline constexpr std::array gpu_trace_specs {
    host_sbs_gpu_trace,
  };

  inline constexpr std::array parallax_v2_live_renderer_specs {
    parallax_v2_live_renderer,
    sbs_reprojection_vertex,
  };

  // Optional HDR bandwidth optimization. Its closure includes the production V2 renderer via
  // the shader's quoted include, but remains independent of the required renderer closure: an
  // edit, compile failure, or unsupported MRT must select the established RGB -> P010 path rather
  // than latching Host SBS flat.
  inline constexpr std::array parallax_v2_p010_y_specs {
    parallax_v2_p010_y_renderer,
  };

  // Independent fail-flat closure. Kept separate from the live-renderer closure so an edited V2
  // pixel shader still degrades to authenticated flat streaming, while an edited vertex shader
  // (shared geometry) invalidates both closures and rejects Host SBS device creation outright.
  inline constexpr std::array sbs_flat_fallback_specs {
    sbs_flat_identity,
    sbs_reprojection_vertex,
  };

  // Dump-only shader roots. These are intentionally absent from prewarm() and from the pinned
  // production renderer closure; a diagnostics failure must not affect live Host SBS.
  inline constexpr std::array parallax_v2_live_diagnostic_specs {
    parallax_v2_live_mapping,
    parallax_v2_live_mask,
  };

  struct source_snapshot;
  using source_snapshot_t = std::shared_ptr<const source_snapshot>;
  using bytecode_t = std::shared_ptr<const std::vector<unsigned char>>;

  struct cache_statistics_t {
    std::uint64_t memory_hits = 0u;
    std::uint64_t persistent_hits = 0u;
    std::uint64_t compiled = 0u;
    std::uint64_t persistent_writes = 0u;
    std::uint64_t rejected_artifacts = 0u;
  };

  /**
   * Select the writable directory used for closure-keyed bytecode artifacts. An empty path
   * disables persistence without affecting the process-local cache. The caller must configure
   * this once, before prewarm() or any concurrent get() calls.
   */
  void configure_persistent_cache(const std::filesystem::path &directory);

  /** Monotonic process counters used by startup logging and focused cache tests. */
  cache_statistics_t cache_statistics() noexcept;

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

  /**
   * Return cached bytecode, loading a validated closure-keyed artifact or compiling this exact
   * source snapshot once process-wide if needed.
   */
  bytecode_t get(
    const source_snapshot_t &sources,
    const shader_spec &spec
  );

  /** Precompile the production V2, mandatory GPU arbitration, and fail-flat shader sets. */
  bool prewarm(const std::filesystem::path &assets_dir);
}  // namespace models::host_sbs_shader_cache
