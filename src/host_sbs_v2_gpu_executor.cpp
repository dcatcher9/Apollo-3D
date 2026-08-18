/**
 * @file src/host_sbs_v2_gpu_executor.cpp
 * @brief Non-owning D3D11 command recorder for the shared base Host-SBS V2 GPU stages.
 */
#include "host_sbs_v2_gpu_executor.h"

#ifdef _WIN32

namespace models::host_sbs_v2_gpu {

  namespace {

    bool record_dispatch(
      ID3D11DeviceContext *context,
      const dispatch_command_t &command
    ) noexcept {
      if (!context || !command.valid()) {
        return false;
      }
      if (command.mode == dispatch_command_t::mode_e::direct) {
        context->Dispatch(
          command.group_count_x,
          command.group_count_y,
          command.group_count_z
        );
      } else {
        context->DispatchIndirect(
          command.indirect_arguments,
          command.indirect_byte_offset
        );
      }
      return true;
    }

    void bind_base_constants(
      ID3D11DeviceContext *context,
      const base_constants_t &constants
    ) noexcept {
      ID3D11Buffer *buffers[2] = {
        constants.depth,
        constants.coordinate_v2,
      };
      context->CSSetConstantBuffers(0u, 2u, buffers);
    }

    void bind_stage_constants(
      ID3D11DeviceContext *context,
      const base_constants_t &constants,
      ID3D11Buffer *stage_constants
    ) noexcept {
      ID3D11Buffer *buffers[3] = {
        constants.depth,
        constants.coordinate_v2,
        stage_constants,
      };
      context->CSSetConstantBuffers(0u, 3u, buffers);
    }

  }  // namespace

  dispatch_command_t dispatch_command_t::direct(
    const UINT x,
    const UINT y,
    const UINT z
  ) noexcept {
    return {
      .mode = mode_e::direct,
      .group_count_x = x,
      .group_count_y = y,
      .group_count_z = z,
    };
  }

  dispatch_command_t dispatch_command_t::indirect(
    ID3D11Buffer *arguments,
    const UINT byte_offset
  ) noexcept {
    return {
      .mode = mode_e::indirect,
      .indirect_arguments = arguments,
      .indirect_byte_offset = byte_offset,
    };
  }

  bool dispatch_command_t::valid() const noexcept {
    switch (mode) {
      case mode_e::direct:
        return group_count_x > 0u && group_count_y > 0u && group_count_z > 0u &&
               !indirect_arguments && indirect_byte_offset == 0u;
      case mode_e::indirect:
        return group_count_x == 0u && group_count_y == 0u &&
               group_count_z == 0u && indirect_arguments &&
               (indirect_byte_offset & 3u) == 0u;
      case mode_e::invalid:
        return false;
    }
    return false;
  }

  bool base_constants_t::valid() const noexcept {
    return depth && coordinate_v2;
  }

  bool record_moments_frame(
    ID3D11DeviceContext *context,
    const moments_frame_command_t &command
  ) noexcept {
    if (!context || !command.constants.valid() || !command.moments_shader ||
        !command.frame_resolve_shader || !command.moments_dispatch.valid() ||
        !command.frame_resolve_dispatch.valid()) {
      return false;
    }
    if (!command.raw_depth || !command.tensor_exclusion ||
        !command.partials_output || !command.partials ||
        !command.frame_stats_output || !command.minmax_raw_output) {
      return false;
    }

    bind_base_constants(context, command.constants);
    context->CSSetShader(command.moments_shader, nullptr, 0u);
    ID3D11ShaderResourceView *moments_inputs[2] = {
      command.raw_depth,
      command.tensor_exclusion,
    };
    context->CSSetShaderResources(0u, 2u, moments_inputs);
    context->CSSetUnorderedAccessViews(0u, 1u, &command.partials_output, nullptr);
    (void) record_dispatch(context, command.moments_dispatch);
    ID3D11ShaderResourceView *null_inputs[2] = {};
    ID3D11UnorderedAccessView *null_outputs[2] = {};
    context->CSSetShaderResources(0u, 2u, null_inputs);
    context->CSSetUnorderedAccessViews(0u, 1u, null_outputs, nullptr);

    context->CSSetShader(command.frame_resolve_shader, nullptr, 0u);
    context->CSSetShaderResources(0u, 1u, &command.partials);
    ID3D11UnorderedAccessView *frame_outputs[2] = {
      command.frame_stats_output,
      command.minmax_raw_output,
    };
    context->CSSetUnorderedAccessViews(0u, 2u, frame_outputs, nullptr);
    (void) record_dispatch(context, command.frame_resolve_dispatch);
    context->CSSetShaderResources(0u, 1u, null_inputs);
    context->CSSetUnorderedAccessViews(0u, 2u, null_outputs, nullptr);
    return true;
  }

  bool record_state(
    ID3D11DeviceContext *context,
    const state_command_t &command
  ) noexcept {
    if (!context || !command.constants.valid() || !command.shader || !command.dispatch.valid()) {
      return false;
    }
    if (!command.frame_stats || !command.cut_bridge_state || !command.state_output) {
      return false;
    }

    bind_base_constants(context, command.constants);
    context->CSSetShader(command.shader, nullptr, 0u);
    ID3D11ShaderResourceView *inputs[2] = {
      command.frame_stats,
      command.cut_bridge_state,
    };
    context->CSSetShaderResources(0u, 2u, inputs);
    context->CSSetUnorderedAccessViews(0u, 1u, &command.state_output, nullptr);
    (void) record_dispatch(context, command.dispatch);
    ID3D11ShaderResourceView *null_inputs[2] = {};
    ID3D11UnorderedAccessView *null_output = nullptr;
    context->CSSetShaderResources(0u, 2u, null_inputs);
    context->CSSetUnorderedAccessViews(0u, 1u, &null_output, nullptr);
    return true;
  }

  bool record_map_history(
    ID3D11DeviceContext *context,
    const map_history_command_t &command
  ) noexcept {
    if (!context || !command.constants.valid() || !command.near_identical_constants || !command.shader || !command.dispatch.valid()) {
      return false;
    }
    if (!command.raw_depth || !command.state || !command.tensor_exclusion ||
        !command.minmax_ema || !command.current_model_input ||
        !command.current_appearance_ordinal || !command.cut_bridge_state ||
        !command.current_depth) {
      return false;
    }
    if (!command.candidate_output || !command.previous_model_input_output ||
        !command.previous_appearance_ordinal_output ||
        !command.previous_reliable_depth_output ||
        !command.previous_tensor_exclusion_output ||
        !command.history_owner_output) {
      return false;
    }

    bind_stage_constants(context, command.constants, command.near_identical_constants);
    context->CSSetShader(command.shader, nullptr, 0u);
    ID3D11ShaderResourceView *inputs[8] = {
      command.raw_depth,
      command.state,
      command.tensor_exclusion,
      command.minmax_ema,
      command.current_model_input,
      command.current_appearance_ordinal,
      command.cut_bridge_state,
      command.current_depth,
    };
    ID3D11UnorderedAccessView *outputs[6] = {
      command.candidate_output,
      command.previous_model_input_output,
      command.previous_appearance_ordinal_output,
      command.previous_reliable_depth_output,
      command.previous_tensor_exclusion_output,
      command.history_owner_output,
    };
    context->CSSetShaderResources(0u, 8u, inputs);
    context->CSSetUnorderedAccessViews(0u, 6u, outputs, nullptr);
    (void) record_dispatch(context, command.dispatch);

    // This full unbind is the cross-dispatch publication boundary for the history tuple/owner.
    ID3D11ShaderResourceView *null_inputs[8] = {};
    ID3D11UnorderedAccessView *null_outputs[6] = {};
    context->CSSetShaderResources(0u, 8u, null_inputs);
    context->CSSetUnorderedAccessViews(0u, 6u, null_outputs, nullptr);
    return true;
  }

  bool record_ownership(
    ID3D11DeviceContext *context,
    const ownership_command_t &command
  ) noexcept {
    if (!context || !command.constants.valid() || !command.source_region_constants || !command.shader || !command.dispatch.valid()) {
      return false;
    }
    if (!command.candidate || !command.source_color || !command.tensor_exclusion || !command.ownership_refined_output) {
      return false;
    }

    bind_stage_constants(context, command.constants, command.source_region_constants);
    context->CSSetShader(command.shader, nullptr, 0u);
    ID3D11ShaderResourceView *inputs[3] = {
      command.candidate,
      command.source_color,
      command.tensor_exclusion,
    };
    context->CSSetShaderResources(0u, 3u, inputs);
    context->CSSetUnorderedAccessViews(
      0u,
      1u,
      &command.ownership_refined_output,
      nullptr
    );
    (void) record_dispatch(context, command.dispatch);
    ID3D11ShaderResourceView *null_inputs[3] = {};
    ID3D11UnorderedAccessView *null_output = nullptr;
    context->CSSetShaderResources(0u, 3u, null_inputs);
    context->CSSetUnorderedAccessViews(0u, 1u, &null_output, nullptr);
    return true;
  }

  bool record_vertical(
    ID3D11DeviceContext *context,
    const vertical_command_t &command
  ) noexcept {
    if (!context || !command.constants.valid() || !command.shader || !command.dispatch.valid()) {
      return false;
    }
    if (!command.ownership_refined || !command.vertical_majorant_output || !command.vertical_conditioned_output) {
      return false;
    }

    bind_base_constants(context, command.constants);
    context->CSSetShader(command.shader, nullptr, 0u);
    context->CSSetShaderResources(0u, 1u, &command.ownership_refined);
    ID3D11UnorderedAccessView *outputs[2] = {
      command.vertical_majorant_output,
      command.vertical_conditioned_output,
    };
    context->CSSetUnorderedAccessViews(0u, 2u, outputs, nullptr);
    (void) record_dispatch(context, command.dispatch);
    ID3D11ShaderResourceView *null_input = nullptr;
    ID3D11UnorderedAccessView *null_outputs[2] = {};
    context->CSSetShaderResources(0u, 1u, &null_input);
    context->CSSetUnorderedAccessViews(0u, 2u, null_outputs, nullptr);
    return true;
  }

  bool record_horizontal(
    ID3D11DeviceContext *context,
    const horizontal_command_t &command
  ) noexcept {
    if (!context || !command.constants.valid() || !command.shader || !command.dispatch.valid()) {
      return false;
    }
    if (!command.vertical_conditioned || !command.final_output) {
      return false;
    }

    bind_base_constants(context, command.constants);
    context->CSSetShader(command.shader, nullptr, 0u);
    context->CSSetShaderResources(0u, 1u, &command.vertical_conditioned);
    context->CSSetUnorderedAccessViews(0u, 1u, &command.final_output, nullptr);
    (void) record_dispatch(context, command.dispatch);
    ID3D11ShaderResourceView *null_input = nullptr;
    ID3D11UnorderedAccessView *null_output = nullptr;
    context->CSSetShaderResources(0u, 1u, &null_input);
    context->CSSetUnorderedAccessViews(0u, 1u, &null_output, nullptr);
    return true;
  }

}  // namespace models::host_sbs_v2_gpu

#endif  // _WIN32
