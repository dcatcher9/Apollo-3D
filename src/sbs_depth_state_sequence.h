#pragma once

#ifdef _WIN32

  #include "video_depth_estimator.h"

  #include <cstdint>
  #include <d3d11.h>
  #include <filesystem>
  #include <memory>
  #include <string>
  #include <vector>
  #include <wrl/client.h>

namespace sbs_bench::depth_state {

  inline constexpr int sequence_schema = 1;
  inline constexpr char sequence_contract[] =
    "apollo-production-depth-state-sequence-v1";

  /**
   * Offline-only writer for the geometry-independent boundary immediately before
   * depth_warp_prefilter_cs.  It stores exact float/uint resources only for the sparse
   * selected output frames, while retaining the tiny runtime-scene row for every completed
   * frame.  The manifest is written last, so a partial sequence is never replayable.
   */
  class sequence_writer {
  public:
    sequence_writer(
      Microsoft::WRL::ComPtr<ID3D11Device> device,
      Microsoft::WRL::ComPtr<ID3D11DeviceContext> context,
      std::filesystem::path root,
      std::string cache_key,
      std::vector<std::uint64_t> source_frame_ids,
      std::vector<std::uint64_t> selected_frame_ids
    );
    ~sequence_writer();

    sequence_writer(const sequence_writer &) = delete;
    sequence_writer &operator=(const sequence_writer &) = delete;

    bool valid() const;
    const std::string &error() const;
    bool capture_runtime_scene(
      std::size_t source_ordinal,
      std::uint64_t source_frame_id,
      const models::runtime_scene_evidence &evidence
    );
    bool capture_selected_state(
      std::size_t source_ordinal,
      std::uint64_t source_frame_id,
      const models::estimate_result &estimate
    );
    bool finish(bool cuda_graph_captured);
    const std::string &manifest_sha256() const;

  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
  };

  /**
   * Fail-closed reader for one verified sequence.  The constructor authenticates every
   * payload byte before any frame can be loaded, preventing mixed/corrupt cache replay.
   */
  class sequence_reader {
  public:
    sequence_reader(
      Microsoft::WRL::ComPtr<ID3D11Device> device,
      std::filesystem::path root,
      std::string expected_cache_key,
      std::string expected_manifest_sha256,
      std::vector<std::uint64_t> expected_source_frame_ids,
      std::vector<std::uint64_t> expected_selected_frame_ids
    );
    ~sequence_reader();

    sequence_reader(const sequence_reader &) = delete;
    sequence_reader &operator=(const sequence_reader &) = delete;

    bool valid() const;
    const std::string &error() const;
    bool runtime_scene(
      std::size_t source_ordinal,
      std::uint64_t source_frame_id,
      models::runtime_scene_evidence &evidence
    ) const;
    bool load_selected_state(
      std::size_t source_ordinal,
      std::uint64_t source_frame_id,
      models::estimate_result &estimate
    );
    bool cuda_graph_captured() const;
    const std::string &manifest_sha256() const;

  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
  };

}  // namespace sbs_bench::depth_state

#endif  // _WIN32
