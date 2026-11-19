/**
 * @file src/host_sbs_v2_gpu_executor.h
 * @brief Non-owning D3D11 command recorder for the shared base Host-SBS V2 GPU stages.
 */
#pragma once

#ifdef _WIN32

  // standard includes
  #include <cstdint>
  #include <optional>

  // platform includes
  #include <d3d11.h>

  // local includes
  #include "generated/depth_coordinate_v2_contract.h"

namespace models::host_sbs_v2_gpu {

  /** One caller-selected direct or indirect dispatch. */
  struct dispatch_command_t {
    enum class mode_e : std::uint8_t {
      invalid,
      direct,
      indirect,
    };

    mode_e mode = mode_e::invalid;
    UINT group_count_x = 0u;
    UINT group_count_y = 0u;
    UINT group_count_z = 0u;
    ID3D11Buffer *indirect_arguments = nullptr;
    UINT indirect_byte_offset = 0u;

    static dispatch_command_t direct(UINT x, UINT y, UINT z) noexcept;
    static dispatch_command_t indirect(
      ID3D11Buffer *arguments,
      UINT byte_offset
    ) noexcept;

    [[nodiscard]] bool valid() const noexcept;
  };

  /** The authenticated depth constants at b0 and V2 constants at b1. */
  struct base_constants_t {
    ID3D11Buffer *depth = nullptr;
    ID3D11Buffer *coordinate_v2 = nullptr;

    [[nodiscard]] bool valid() const noexcept;
  };

  // One structured element, in the same order as TensorAnalysisDomain in HLSL.
  struct analysis_domain_record_t {
    std::uint32_t valid = 0u;
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    std::uint32_t reserved = 0u;
    std::uint32_t left = 0u;
    std::uint32_t top = 0u;
    std::uint32_t right = 0u;
    std::uint32_t bottom = 0u;
  };
  static_assert(sizeof(analysis_domain_record_t) == 32u);

  struct frame_publication_outputs_t {
    ID3D11UnorderedAccessView *minmax_ema = nullptr;
    ID3D11UnorderedAccessView *depth_health = nullptr;
    ID3D11UnorderedAccessView *camera_state = nullptr;
  };

  struct moments_frame_command_t {
    base_constants_t constants;
    ID3D11ComputeShader *moments_shader = nullptr;
    ID3D11ComputeShader *frame_resolve_shader = nullptr;
    ID3D11ShaderResourceView *raw_depth = nullptr;
    /** The analysis domain expected by the selected scan shaders. */
    ID3D11ShaderResourceView *analysis_domain = nullptr;
    ID3D11UnorderedAccessView *partials_output = nullptr;
    ID3D11ShaderResourceView *partials = nullptr;
    ID3D11UnorderedAccessView *frame_stats_output = nullptr;
    ID3D11UnorderedAccessView *minmax_raw_output = nullptr;
    dispatch_command_t moments_dispatch;
    dispatch_command_t frame_resolve_dispatch;
    /** Required adaptive GPU quantiles; reuse preserves the complete prior publication. */
    ID3D11ComputeShader *histogram_shader = nullptr;
    ID3D11ComputeShader *quantile_shader = nullptr;
    ID3D11ShaderResourceView *frame_stats = nullptr;
    ID3D11UnorderedAccessView *histogram_output = nullptr;
    ID3D11ShaderResourceView *histogram = nullptr;
    /** Replay's quantile-only shader exports the independent P02/P98 bins. */
    ID3D11UnorderedAccessView *normalization_histogram_output = nullptr;
    /** Live publish_main consumes those bins in shared memory and writes the complete state. */
    std::optional<frame_publication_outputs_t> publication_outputs;
  };

  struct state_command_t {
    base_constants_t constants;
    ID3D11ComputeShader *shader = nullptr;
    ID3D11ShaderResourceView *frame_stats = nullptr;
    ID3D11ShaderResourceView *depth_health_state = nullptr;
    ID3D11UnorderedAccessView *state_output = nullptr;
    dispatch_command_t dispatch;
  };

  struct map_history_command_t {
    base_constants_t constants;
    /** Exact Host-SBS near-identical constants at b2. */
    ID3D11Buffer *near_identical_constants = nullptr;
    ID3D11ComputeShader *shader = nullptr;
    ID3D11ShaderResourceView *raw_depth = nullptr;
    ID3D11ShaderResourceView *state = nullptr;
    ID3D11ShaderResourceView *analysis_domain = nullptr;
    ID3D11ShaderResourceView *minmax_ema = nullptr;
    ID3D11ShaderResourceView *current_model_input = nullptr;
    ID3D11ShaderResourceView *depth_health_state = nullptr;
    ID3D11UnorderedAccessView *candidate_output = nullptr;
    ID3D11UnorderedAccessView *previous_model_input_output = nullptr;
    ID3D11UnorderedAccessView *history_owner_output = nullptr;
    dispatch_command_t dispatch;
  };

  /** Direct group count for a limiter pass over `lines` columns (vertical) or rows (horizontal). */
  [[nodiscard]] constexpr UINT limiter_groups(const UINT lines) noexcept {
    constexpr UINT group_lines = depth_coordinate_v2::limiter_group_lines;
    return (lines + group_lines - 1u) / group_lines;
  }

  struct vertical_command_t {
    base_constants_t constants;
    ID3D11ComputeShader *shader = nullptr;
    ID3D11ShaderResourceView *candidate = nullptr;
    ID3D11UnorderedAccessView *vertical_majorant_output = nullptr;
    ID3D11UnorderedAccessView *vertical_conditioned_output = nullptr;
    dispatch_command_t dispatch;
  };

  struct horizontal_command_t {
    base_constants_t constants;
    ID3D11ComputeShader *shader = nullptr;
    ID3D11ShaderResourceView *vertical_conditioned = nullptr;
    ID3D11UnorderedAccessView *final_output = nullptr;
    dispatch_command_t dispatch;
  };

  /** Record the shared raw scan and either replay quantiles or the complete live publication. */
  [[nodiscard]] bool record_moments_frame(
    ID3D11DeviceContext *context,
    const moments_frame_command_t &command
  ) noexcept;

  /** Record persistent V2 camera-state resolve. */
  [[nodiscard]] bool record_state(
    ID3D11DeviceContext *context,
    const state_command_t &command
  ) noexcept;

  /** Record candidate mapping plus exact history/owner publication. */
  [[nodiscard]] bool record_map_history(
    ID3D11DeviceContext *context,
    const map_history_command_t &command
  ) noexcept;

  /** Record the column upper/lower envelope and fixed vertical share. */
  [[nodiscard]] bool record_vertical(
    ID3D11DeviceContext *context,
    const vertical_command_t &command
  ) noexcept;

  /** Record the final row-wise horizontal majorant. */
  [[nodiscard]] bool record_horizontal(
    ID3D11DeviceContext *context,
    const horizontal_command_t &command
  ) noexcept;

}  // namespace models::host_sbs_v2_gpu

#endif  // _WIN32
