/** Typed production entry to the shared causal native SBS engine. */
#pragma once

#include <cstdint>
#include <string>

namespace offline_sbs {
  enum class raw_pixel_format : std::uint32_t;
  struct raw_frame_transport;

  namespace wire {
    struct whole_clip_contract_t;
  }

  enum class native_pass_operation {
    analyze,
    convert,
  };

  struct native_pass_settings {
    // Preserve the worker's existing UTF-8 path representation and frame identities.
    std::string source_identity_directory;
    std::string transient_output_directory;
    std::string observation_timeline;
    raw_pixel_format source_format {};
    std::uint64_t frame_count = 0;
    native_pass_operation operation = native_pass_operation::analyze;
    std::uint64_t transient_raster_hard_cap_bytes = 0;
  };

  /**
   * One GPU owner thread calls this with the worker's already initialized config. A successful
   * pass (zero) leaves its validated whole-clip contract in `contract`; no file carries it.
   */
  int run_native_pass(
    const native_pass_settings &settings,
    raw_frame_transport &transport,
    wire::whole_clip_contract_t &contract
  );
}  // namespace offline_sbs
