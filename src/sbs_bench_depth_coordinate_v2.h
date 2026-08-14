/**
 * @file src/sbs_bench_depth_coordinate_v2.h
 * @brief Harness-only whole-sequence execution of the production depth-coordinate-v2 shaders.
 *
 * This is deliberately not a second mapping implementation. The helper loads authenticated raw
 * DAV2 fields and hard-cut generations, keeps one GPU state buffer alive for the sequence, and
 * dispatches the same seven base V2 coordinate entrypoints used by video_depth_estimator. Its
 * replay contract supplies canonical full-content analysis geometry and omits the live-only
 * OCR/SLR12 subtitle ownership stage. Its candidate texture is retained as immutable
 * pre-limiter evidence. The column pass publishes
 * both an upper-envelope diagnostic and the authenticated 75/25 vertical share consumed by the
 * pure row majorant. The encoded final texture is the live render input for the eleven-step
 * contractive inverse. A separate
 * diagnostic-only map entrypoint materializes canonical_values for the replay report; it is not
 * part of the authenticated production producer closure or live authentication.
 */
#pragma once

#ifdef _WIN32

  #include <cstddef>
  #include <cstdint>
  #include <filesystem>
  #include <memory>
  #include <string>
  #include <string_view>
  #include <vector>

  #include <d3d11.h>
  #include <wrl/client.h>

namespace sbs_bench {

  inline constexpr unsigned depth_coordinate_v2_state_trace_schema = 18;
  inline constexpr std::string_view depth_coordinate_v2_state_trace_policy =
    "immediate-first-usable-arithmetic-mean-zero-fixed-scale-fixed-near-curve-retained-camera-source-ownership-pointwise-soft-container-vertical-share75-row-majorant-v17";
  inline constexpr std::string_view depth_coordinate_v2_diagnostic_role =
    "non-controlling-fixed-scale-camera-audit-v4";
  inline constexpr std::string_view depth_coordinate_v2_gpu_authority =
    "authenticated-raw-source-color-plus-seven-v2-compute-shaders-persistent-gpu-state-v8";
  inline constexpr std::string_view depth_coordinate_v2_gpu_execution =
    "authenticated-raw-source-color-plus-seven-v2-compute-shaders-persistent-state-v8";

  struct depth_coordinate_v2_gpu_frame {
    std::string frame_id;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> raw_depth;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> canonical_order;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> candidate_parallax;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> ownership_refined_parallax;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> vertical_majorant;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> vertical_conditioned;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> encoded_final_parallax;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> cut_state;
    std::vector<float> raw_values;
    std::vector<float> canonical_values;
    std::vector<float> candidate_parallax_values;
    std::vector<float> ownership_refined_parallax_values;
    std::vector<float> vertical_majorant_values;
    std::vector<float> vertical_conditioned_values;
    std::vector<float> encoded_parallax_values;
    float encoded_minimum = 0.5f;
    float encoded_maximum = 0.5f;
    float maximum_absolute_source_u = 0.0f;
    float order_minimum = 0.0f;
    float order_maximum = 0.0f;
    std::string raw_sha256;
    std::string order_sha256;
    std::string candidate_sha256;
    std::string ownership_refined_sha256;
    std::string vertical_majorant_sha256;
    std::string vertical_conditioned_sha256;
    std::string parallax_sha256;
  };

  class depth_coordinate_v2_gpu_replay {
  public:
    static std::unique_ptr<depth_coordinate_v2_gpu_replay> create(
      ID3D11Device *device,
      ID3D11DeviceContext *context,
      const std::filesystem::path &manifest_path,
      std::string &error
    );

    ~depth_coordinate_v2_gpu_replay();

    depth_coordinate_v2_gpu_replay(const depth_coordinate_v2_gpu_replay &) = delete;
    depth_coordinate_v2_gpu_replay &operator=(const depth_coordinate_v2_gpu_replay &) = delete;

    std::size_t frame_count() const;
    unsigned width() const;
    unsigned height() const;
    const std::string &model_name() const;
    const std::string &manifest_sha256() const;

    /**
     * Dispatch one caller-owned source SRV. `expected_source_sha256` authenticates the caller's
     * provenance binding; D3D11 cannot hash an arbitrary SRV here without a synchronous readback.
     * The exact harness therefore hashes and decodes one immutable byte snapshot before upload.
     */
    bool dispatch(
      std::size_t sequence_index,
      std::string_view expected_frame_id,
      ID3D11ShaderResourceView *source_color,
      std::uint32_t source_color_mode,
      float source_linear_scale,
      std::string_view expected_source_sha256,
      depth_coordinate_v2_gpu_frame &result,
      std::string &error
    );

    bool write_state_trace(const std::filesystem::path &path, std::string &error) const;

  #ifdef SUNSHINE_TESTS
    /** Replace the persistent GPU state before a dispatch to exercise fail-closed recovery. */
    bool overwrite_state_for_testing(
      const std::vector<std::uint32_t> &words,
      std::string &error
    );
  #endif

  private:
    depth_coordinate_v2_gpu_replay();
    struct impl;
    std::unique_ptr<impl> impl_;
  };

}  // namespace sbs_bench

#endif  // _WIN32
