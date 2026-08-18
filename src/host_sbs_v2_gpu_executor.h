/**
 * @file src/host_sbs_v2_gpu_executor.h
 * @brief Non-owning D3D11 command recorder for the shared base Host-SBS V2 GPU stages.
 */
#pragma once

#ifdef _WIN32

  // standard includes
  #include <cstdint>

  // platform includes
  #include <d3d11.h>

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

  struct moments_frame_command_t {
    base_constants_t constants;
    ID3D11ComputeShader *moments_shader = nullptr;
    ID3D11ComputeShader *frame_resolve_shader = nullptr;
    ID3D11ShaderResourceView *raw_depth = nullptr;
    /** Always explicit: live exclusion or replay's zero-cleared exclusion texture. */
    ID3D11ShaderResourceView *tensor_exclusion = nullptr;
    ID3D11UnorderedAccessView *partials_output = nullptr;
    ID3D11ShaderResourceView *partials = nullptr;
    ID3D11UnorderedAccessView *frame_stats_output = nullptr;
    ID3D11UnorderedAccessView *minmax_raw_output = nullptr;
    dispatch_command_t moments_dispatch;
    dispatch_command_t frame_resolve_dispatch;
  };

  struct state_command_t {
    base_constants_t constants;
    ID3D11ComputeShader *shader = nullptr;
    ID3D11ShaderResourceView *frame_stats = nullptr;
    ID3D11ShaderResourceView *cut_bridge_state = nullptr;
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
    ID3D11ShaderResourceView *tensor_exclusion = nullptr;
    ID3D11ShaderResourceView *minmax_ema = nullptr;
    ID3D11ShaderResourceView *current_model_input = nullptr;
    ID3D11ShaderResourceView *current_appearance_ordinal = nullptr;
    ID3D11ShaderResourceView *cut_bridge_state = nullptr;
    ID3D11ShaderResourceView *current_depth = nullptr;
    ID3D11UnorderedAccessView *candidate_output = nullptr;
    ID3D11UnorderedAccessView *previous_model_input_output = nullptr;
    ID3D11UnorderedAccessView *previous_appearance_ordinal_output = nullptr;
    ID3D11UnorderedAccessView *previous_reliable_depth_output = nullptr;
    ID3D11UnorderedAccessView *previous_tensor_exclusion_output = nullptr;
    ID3D11UnorderedAccessView *history_owner_output = nullptr;
    dispatch_command_t dispatch;
  };

  struct ownership_command_t {
    base_constants_t constants;
    /** Exact depth-source-region constants at b2. */
    ID3D11Buffer *source_region_constants = nullptr;
    ID3D11ComputeShader *shader = nullptr;
    ID3D11ShaderResourceView *candidate = nullptr;
    ID3D11ShaderResourceView *source_color = nullptr;
    /** Always explicit: live exclusion or replay's zero-cleared exclusion texture. */
    ID3D11ShaderResourceView *tensor_exclusion = nullptr;
    ID3D11UnorderedAccessView *ownership_refined_output = nullptr;
    dispatch_command_t dispatch;
  };

  struct vertical_command_t {
    base_constants_t constants;
    ID3D11ComputeShader *shader = nullptr;
    ID3D11ShaderResourceView *ownership_refined = nullptr;
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

  /** Record moments followed by the one-thread frame resolve. */
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

  /** Record conservative source-color ownership refinement. */
  [[nodiscard]] bool record_ownership(
    ID3D11DeviceContext *context,
    const ownership_command_t &command
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
