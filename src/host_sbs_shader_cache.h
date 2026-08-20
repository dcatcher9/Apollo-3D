/**
 * @file src/host_sbs_shader_cache.h
 * @brief Closure-keyed memory and persistent bytecode cache for fixed-shape Host SBS shaders.
 */
#pragma once

#include "generated/host_sbs_shader_manifest.h"

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
