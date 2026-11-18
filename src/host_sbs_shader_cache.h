/**
 * @file src/host_sbs_shader_cache.h
 * @brief Closure-keyed memory and persistent bytecode cache for fixed-shape Host SBS shaders.
 */
#pragma once

#include "generated/host_sbs_shader_manifest.h"
#include <array>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace models::host_sbs_shader_cache {
  struct source_snapshot;
  using source_snapshot_t = std::shared_ptr<const source_snapshot>;
  using bytecode_t = std::shared_ptr<const std::vector<unsigned char>>;

  struct shader_set_t {
    std::string_view name;
    std::span<const shader_spec> specs;
    std::string_view closure;
  };

  // The package builder and live startup use this exact order and group identity. The shared
  // vertex program belongs to both the completed and flat source closures.
  inline constexpr std::array<shader_set_t, 4u> production_prewarm_sets {{
    {"depth producer", parallax_v2_producer_specs, parallax_v2_producer_source_closure_sha256},
    {"near-identical arbitration", near_identical_detector_specs, near_identical_detector_source_closure_sha256},
    {"completed renderer", parallax_v2_live_renderer_specs, parallax_v2_live_renderer_source_closure_sha256},
    {"flat identity", sbs_flat_fallback_specs, sbs_flat_fallback_source_closure_sha256},
  }};

  struct cache_statistics_t {
    std::uint64_t memory_hits = 0u;
    std::uint64_t persistent_hits = 0u;
    std::uint64_t compiled = 0u;
    std::uint64_t persistent_writes = 0u;
    std::uint64_t rejected_artifacts = 0u;
    std::uint64_t embedded_hits = 0u;
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

  /** Files in the authenticated snapshot, for the build generator's dependency file. */
  std::vector<std::filesystem::path> source_paths(const source_snapshot_t &sources);

  /** PE resource name for an entry owned by the given snapshot. */
  std::string bytecode_resource_name(const source_snapshot_t &sources, const shader_spec &spec);

  /** Load only the executable resource for this exact source snapshot and program. */
  bytecode_t get_embedded(const source_snapshot_t &sources, const shader_spec &spec);

  /**
   * Return cached bytecode from the executable, the writable cache, or compilation of this exact
   * source snapshot. The producer resolves each process-local key once.
   */
  bytecode_t get(
    const source_snapshot_t &sources,
    const shader_spec &spec
  );

  /** Authenticate and resolve the production V2, mandatory GPU arbitration, and fail-flat shader sets. */
  bool prewarm(const std::filesystem::path &assets_dir);
}  // namespace models::host_sbs_shader_cache
