/**
 * @file src/host_sbs_v2_gpu_executor.cpp
 * @brief Non-owning D3D11 command recorder for the shared base Host-SBS V2 GPU stages.
 */
#include "host_sbs_v2_gpu_executor.h"

#ifdef _WIN32

namespace models::host_sbs_v2_gpu {

  namespace {

    void record_validated_dispatch(
      ID3D11DeviceContext *context,
      const dispatch_command_t &command
    ) noexcept {
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

  bool record_eye_limit(
    ID3D11DeviceContext *context,
    const eye_limit_command_t &command
  ) noexcept {
    if (!context || !command.constants.valid() || !command.near_identical_constants ||
        !command.eye_constants || !command.shader || !command.vertical_conditioned ||
        !command.eye_output || !command.dispatch.valid()) {
      return false;
    }
    ID3D11Buffer *constants[4] = {
      command.constants.depth,
      command.constants.coordinate_v2,
      command.near_identical_constants,
      command.eye_constants,
    };
    context->CSSetConstantBuffers(0u, 4u, constants);
    context->CSSetShader(command.shader, nullptr, 0u);
    context->CSSetShaderResources(0u, 1u, &command.vertical_conditioned);
    context->CSSetUnorderedAccessViews(0u, 1u, &command.eye_output, nullptr);
    record_validated_dispatch(context, command.dispatch);
    ID3D11Buffer *null_constants = nullptr;
    ID3D11ShaderResourceView *null_input = nullptr;
    ID3D11UnorderedAccessView *null_output = nullptr;
    context->CSSetConstantBuffers(3u, 1u, &null_constants);
    context->CSSetShaderResources(0u, 1u, &null_input);
    context->CSSetUnorderedAccessViews(0u, 1u, &null_output, nullptr);
    return true;
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
    if (!command.raw_depth || !command.analysis_domain ||
        !command.partials_output || !command.partials ||
        !command.frame_stats_output || !command.minmax_raw_output) {
      return false;
    }
    if (!command.histogram_shader || !command.quantile_shader || !command.frame_stats ||
        !command.histogram_output || !command.histogram) {
      return false;
    }
    const auto &publication = command.publication_outputs;
    if (publication ?
          (!publication->minmax_ema || !publication->depth_health || !publication->camera_state ||
           command.normalization_histogram_output) :
          !command.normalization_histogram_output) {
      return false;
    }

    bind_base_constants(context, command.constants);
    context->CSSetShader(command.moments_shader, nullptr, 0u);
    ID3D11ShaderResourceView *moments_inputs[2] = {
      command.raw_depth,
      command.analysis_domain,
    };
    context->CSSetShaderResources(0u, 2u, moments_inputs);
    context->CSSetUnorderedAccessViews(0u, 1u, &command.partials_output, nullptr);
    record_validated_dispatch(context, command.moments_dispatch);
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
    record_validated_dispatch(context, command.frame_resolve_dispatch);
    context->CSSetShaderResources(0u, 1u, null_inputs);
    context->CSSetUnorderedAccessViews(0u, 2u, null_outputs, nullptr);
    {
      context->CSSetShader(command.histogram_shader, nullptr, 0u);
      ID3D11ShaderResourceView *histogram_inputs[3] = {
        command.raw_depth, command.analysis_domain, command.frame_stats,
      };
      context->CSSetShaderResources(0u, 3u, histogram_inputs);
      ID3D11UnorderedAccessView *histogram_outputs[2] = {
        command.histogram_output, command.minmax_raw_output,
      };
      context->CSSetUnorderedAccessViews(0u, 2u, histogram_outputs, nullptr);
      record_validated_dispatch(context, command.moments_dispatch);
      ID3D11ShaderResourceView *null_histogram_inputs[3] = {};
      context->CSSetShaderResources(0u, 3u, null_histogram_inputs);
      context->CSSetUnorderedAccessViews(0u, 2u, null_outputs, nullptr);
      context->CSSetShader(command.quantile_shader, nullptr, 0u);
      context->CSSetShaderResources(0u, 1u, &command.histogram);
      ID3D11UnorderedAccessView *quantile_outputs[6] = {
        command.frame_stats_output, command.normalization_histogram_output,
      };
      UINT output_count = 2u;
      if (publication) {
        quantile_outputs[2] = publication->minmax_ema;
        quantile_outputs[3] = command.minmax_raw_output;
        quantile_outputs[4] = publication->depth_health;
        quantile_outputs[5] = publication->camera_state;
        output_count = 6u;
      }
      context->CSSetUnorderedAccessViews(0u, output_count, quantile_outputs, nullptr);
      record_validated_dispatch(context, command.frame_resolve_dispatch);
      context->CSSetShaderResources(0u, 1u, null_inputs);
      ID3D11UnorderedAccessView *null_quantile_outputs[6] = {};
      context->CSSetUnorderedAccessViews(0u, output_count, null_quantile_outputs, nullptr);
    }
    return true;
  }

  bool record_state(
    ID3D11DeviceContext *context,
    const state_command_t &command
  ) noexcept {
    if (!context || !command.constants.valid() || !command.shader || !command.dispatch.valid()) {
      return false;
    }
    if (!command.frame_stats || !command.depth_health_state || !command.state_output) {
      return false;
    }

    bind_base_constants(context, command.constants);
    context->CSSetShader(command.shader, nullptr, 0u);
    ID3D11ShaderResourceView *inputs[2] = {
      command.frame_stats,
      command.depth_health_state,
    };
    context->CSSetShaderResources(0u, 2u, inputs);
    context->CSSetUnorderedAccessViews(0u, 1u, &command.state_output, nullptr);
    record_validated_dispatch(context, command.dispatch);
    ID3D11ShaderResourceView *null_inputs[2] = {};
    ID3D11UnorderedAccessView *null_output = nullptr;
    context->CSSetShaderResources(0u, 2u, null_inputs);
    context->CSSetUnorderedAccessViews(0u, 1u, &null_output, nullptr);
    return true;
  }

  bool record_base_geometry(
    ID3D11DeviceContext *context,
    const base_geometry_command_t &command
  ) noexcept {
    if (!context || !command.constants.valid() || !command.near_identical_constants ||
        !command.map_shader || !command.vertical_shader || !command.horizontal_shader ||
        !command.map_dispatch.valid() || !command.vertical_dispatch.valid() ||
        !command.horizontal_dispatch.valid()) {
      return false;
    }
    if (!command.raw_depth || !command.state || !command.analysis_domain ||
        !command.minmax_ema || !command.current_model_input || !command.depth_health_state ||
        !command.candidate || !command.vertical_conditioned) {
      return false;
    }
    if (!command.candidate_output || !command.previous_model_input_output || !command.history_owner_output ||
        !command.vertical_majorant_output || !command.vertical_conditioned_output || !command.final_output) {
      return false;
    }

    if (command.map_start) {
      context->End(command.map_start);
    }
    ID3D11Buffer *constants[3] = {
      command.constants.depth,
      command.constants.coordinate_v2,
      command.near_identical_constants,
    };
    context->CSSetConstantBuffers(0u, 3u, constants);
    context->CSSetShader(command.map_shader, nullptr, 0u);
    ID3D11ShaderResourceView *map_inputs[6] = {
      command.raw_depth,
      command.state,
      command.analysis_domain,
      command.minmax_ema,
      command.current_model_input,
      command.depth_health_state,
    };
    ID3D11UnorderedAccessView *map_outputs[3] = {
      command.candidate_output,
      command.previous_model_input_output,
      command.history_owner_output,
    };
    context->CSSetShaderResources(0u, 6u, map_inputs);
    context->CSSetUnorderedAccessViews(0u, 3u, map_outputs, nullptr);
    record_validated_dispatch(context, command.map_dispatch);

    // Release the history outputs and candidate UAV before the next pass reads it as an SRV.
    ID3D11UnorderedAccessView *null_outputs[3] = {};
    context->CSSetUnorderedAccessViews(0u, 3u, null_outputs, nullptr);
    if (command.limits_start) {
      context->End(command.limits_start);
    }
    context->CSSetShader(command.vertical_shader, nullptr, 0u);
    // Both limiters use the same b0/b1 as mapping. Replace t0 and clear t1-t5 together.
    ID3D11ShaderResourceView *vertical_inputs[6] = {command.candidate};
    context->CSSetShaderResources(0u, 6u, vertical_inputs);
    ID3D11UnorderedAccessView *vertical_outputs[2] = {
      command.vertical_majorant_output,
      command.vertical_conditioned_output,
    };
    context->CSSetUnorderedAccessViews(0u, 2u, vertical_outputs, nullptr);
    record_validated_dispatch(context, command.vertical_dispatch);
    // Release the conditioned UAV before rebinding it for the horizontal pass.
    context->CSSetUnorderedAccessViews(0u, 2u, null_outputs, nullptr);

    context->CSSetShader(command.horizontal_shader, nullptr, 0u);
    context->CSSetShaderResources(0u, 1u, &command.vertical_conditioned);
    context->CSSetUnorderedAccessViews(0u, 1u, &command.final_output, nullptr);
    record_validated_dispatch(context, command.horizontal_dispatch);
    ID3D11ShaderResourceView *null_input = nullptr;
    context->CSSetShaderResources(0u, 1u, &null_input);
    context->CSSetUnorderedAccessViews(0u, 1u, null_outputs, nullptr);
    return true;
  }

}  // namespace models::host_sbs_v2_gpu

#endif  // _WIN32
