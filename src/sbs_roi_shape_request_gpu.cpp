/**
 * @file src/sbs_roi_shape_request_gpu.cpp
 * @brief Nonblocking D3D11 dispatcher/readback ring for Host SBS ROI shape requests.
 */

#include "sbs_roi_shape_request_gpu.h"

#include "generated/sbs_scene_controller_contract.h"

#include <array>
#include <cmath>
#include <cstring>
#include <d3dcompiler.h>
#include <limits>
#include <utility>
#include <vector>

namespace models {
  namespace {
    using Microsoft::WRL::ComPtr;

    constexpr std::size_t readback_slot_count = 3u;
    constexpr UINT rule_state_vector_count =
      static_cast<UINT>(
        sbs_scene_controller::rule_state_vector_count
      );
    constexpr UINT uint4_stride = 4u * sizeof(std::uint32_t);
    constexpr UINT rule_state_byte_width =
      rule_state_vector_count * uint4_stride;
    static_assert(
      sbs_scene_controller::rule_state_vector_count <=
        std::numeric_limits<UINT>::max()
    );

    struct alignas(16) shape_request_constants {
      std::uint32_t source_width;
      std::uint32_t source_height;
      std::uint32_t canonical_model_width;
      std::uint32_t canonical_model_height;

      std::uint32_t target_pixel_budget;
      std::uint32_t profile_max_width;
      std::uint32_t profile_max_height;
      std::uint32_t expected_backend_generation;

      float quiet_halo_cells;
      std::uint32_t analysis_canvas_size;
      float max_model_aspect;
      std::uint32_t active_rules;

      std::array<std::uint32_t, 4> reserved {};
    };

    static_assert(sizeof(shape_request_constants) == 64u);
    static_assert(alignof(shape_request_constants) == 16u);
    static_assert(
      offsetof(shape_request_constants, source_width) == 0u
    );
    static_assert(
      offsetof(shape_request_constants, target_pixel_budget) == 16u
    );
    static_assert(
      offsetof(shape_request_constants, quiet_halo_cells) == 32u
    );
    static_assert(
      offsetof(shape_request_constants, reserved) == 48u
    );

    [[nodiscard]] shape_request_constants make_constants(
      const sbs_roi_shape_request_gpu_submission &submission
    ) {
      return {
        submission.source_width,
        submission.source_height,
        submission.canonical_model_width,
        submission.canonical_model_height,
        submission.target_pixel_budget,
        submission.profile_max_width,
        submission.profile_max_height,
        submission.expected_backend_generation,
        submission.quiet_halo_cells,
        submission.analysis_canvas_size,
        submission.max_model_aspect,
        submission.active_rules,
        {},
      };
    }

    [[nodiscard]] bool submission_contract_valid(
      const sbs_roi_shape_request_gpu_submission &submission
    ) {
      if (
        submission.source_frame_id == 0u ||
        submission.source_width == 0u ||
        submission.source_height == 0u ||
        !sbs_roi_shape_patch_aligned(
          submission.canonical_model_width
        ) ||
        !sbs_roi_shape_patch_aligned(
          submission.canonical_model_height
        ) ||
        submission.canonical_model_width > submission.source_width ||
        submission.canonical_model_height > submission.source_height ||
        submission.canonical_model_width >
          sbs_roi_shape_request_engine_max_dimension ||
        submission.canonical_model_height >
          sbs_roi_shape_request_engine_max_dimension ||
        submission.canonical_model_width >
          submission.profile_max_width ||
        submission.canonical_model_height >
          submission.profile_max_height ||
        submission.target_pixel_budget <
          sbs_roi_shape_request_patch_size *
            sbs_roi_shape_request_patch_size ||
        submission.target_pixel_budget >
          sbs_roi_shape_request_engine_max_dimension *
            sbs_roi_shape_request_engine_max_dimension ||
        submission.profile_max_width <
          sbs_roi_shape_request_patch_size ||
        submission.profile_max_height <
          sbs_roi_shape_request_patch_size ||
        !std::isfinite(submission.quiet_halo_cells) ||
        submission.quiet_halo_cells < 0.0f ||
        submission.analysis_canvas_size == 0u ||
        submission.quiet_halo_cells >
          static_cast<float>(submission.analysis_canvas_size) ||
        !std::isfinite(submission.max_model_aspect) ||
        submission.max_model_aspect < 1.0f ||
        submission.max_model_aspect >
          sbs_roi_shape_request_max_aspect_limit
      ) {
        return false;
      }

      // max_model_aspect caps active ROI tensors only. Canonical full-frame tensors preserve the
      // source geometry and use that cap as an area budget, so wide sources may exceed it.
      return sbs_roi_full_frame_shape_matches(
        submission.source_width,
        submission.source_height,
        submission.canonical_model_width,
        submission.canonical_model_height
      );
    }

    [[nodiscard]] bool create_structured_buffer(
      ID3D11Device *device,
      const UINT byte_width,
      const UINT bind_flags,
      const void *initial_data,
      ComPtr<ID3D11Buffer> &buffer
    ) {
      D3D11_BUFFER_DESC descriptor {};
      descriptor.Usage = D3D11_USAGE_DEFAULT;
      descriptor.ByteWidth = byte_width;
      descriptor.BindFlags = bind_flags;
      descriptor.MiscFlags =
        D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      descriptor.StructureByteStride = uint4_stride;
      D3D11_SUBRESOURCE_DATA initial {
        initial_data,
        0,
        0,
      };
      return SUCCEEDED(device->CreateBuffer(
        &descriptor,
        initial_data ? &initial : nullptr,
        &buffer
      ));
    }

    [[nodiscard]] std::shared_ptr<const std::vector<std::uint8_t>>
    compile_shape_request_shader(const std::filesystem::path &path) {
      ComPtr<ID3DBlob> shader;
      ComPtr<ID3DBlob> diagnostics;
      constexpr DWORD flags =
        D3DCOMPILE_ENABLE_STRICTNESS |
        D3DCOMPILE_OPTIMIZATION_LEVEL3;
      if (FAILED(D3DCompileFromFile(
            path.wstring().c_str(),
            nullptr,
            D3D_COMPILE_STANDARD_FILE_INCLUDE,
            "main",
            "cs_5_0",
            flags,
            0,
            &shader,
            &diagnostics
          ))) {
        return {};
      }
      const auto *begin =
        static_cast<const std::uint8_t *>(shader->GetBufferPointer());
      return std::make_shared<const std::vector<std::uint8_t>>(
        begin,
        begin + shader->GetBufferSize()
      );
    }
  }  // namespace

  struct sbs_roi_shape_request_gpu::impl {
    struct slot_t {
      ComPtr<ID3D11Buffer> constants;
      ComPtr<ID3D11Buffer> staging;
      ComPtr<ID3D11Query> completion;
      bool pending = false;
      std::uint64_t submission_sequence = 0u;
      std::uint64_t source_frame_id = 0u;
      std::uint32_t source_width = 0u;
      std::uint32_t source_height = 0u;
      std::uint32_t expected_backend_generation = 0u;
      sbs_roi_shape_request_limits limits;
    };

    impl(
      ComPtr<ID3D11Device> device_value,
      ComPtr<ID3D11DeviceContext> context_value,
      const std::filesystem::path &assets_dir
    ):
        device(std::move(device_value)),
        context(std::move(context_value)) {
      if (!device || !context) {
        return;
      }

      const auto bytecode = compile_shape_request_shader(
        assets_dir / "shaders" / "directx" /
          "sbs_roi_shape_request_cs.hlsl"
      );
      if (
        !bytecode ||
        FAILED(device->CreateComputeShader(
          bytecode->data(),
          bytecode->size(),
          nullptr,
          &shader
        ))
      ) {
        return;
      }

      if (
        !create_structured_buffer(
          device.Get(),
          sizeof(sbs_roi_shape_request),
          D3D11_BIND_UNORDERED_ACCESS,
          nullptr,
          output
        ) ||
        FAILED(device->CreateUnorderedAccessView(
          output.Get(),
          nullptr,
          &output_uav
        ))
      ) {
        return;
      }

      const std::array<std::uint32_t, rule_state_byte_width / 4u>
        zero_words {};
      if (
        !create_structured_buffer(
          device.Get(),
          rule_state_byte_width,
          D3D11_BIND_SHADER_RESOURCE,
          zero_words.data(),
          zero_rule_state
        ) ||
        FAILED(device->CreateShaderResourceView(
          zero_rule_state.Get(),
          nullptr,
          &zero_rule_state_srv
        ))
      ) {
        return;
      }

      D3D11_QUERY_DESC query_descriptor {
        D3D11_QUERY_EVENT,
        0,
      };
      for (auto &slot : slots) {
        D3D11_BUFFER_DESC constants_descriptor {};
        constants_descriptor.Usage = D3D11_USAGE_DEFAULT;
        constants_descriptor.ByteWidth =
          sizeof(shape_request_constants);
        constants_descriptor.BindFlags =
          D3D11_BIND_CONSTANT_BUFFER;

        D3D11_BUFFER_DESC staging_descriptor {};
        staging_descriptor.Usage = D3D11_USAGE_STAGING;
        staging_descriptor.ByteWidth =
          sizeof(sbs_roi_shape_request);
        staging_descriptor.CPUAccessFlags =
          D3D11_CPU_ACCESS_READ;

        if (
          FAILED(device->CreateBuffer(
            &constants_descriptor,
            nullptr,
            &slot.constants
          )) ||
          FAILED(device->CreateBuffer(
            &staging_descriptor,
            nullptr,
            &slot.staging
          )) ||
          FAILED(device->CreateQuery(
            &query_descriptor,
            &slot.completion
          ))
        ) {
          return;
        }
      }

      ready = true;
    }

    [[nodiscard]] bool rule_state_view_usable(
      ID3D11ShaderResourceView *view
    ) const {
      if (!view) {
        return false;
      }

      ComPtr<ID3D11Device> view_device;
      view->GetDevice(&view_device);
      if (view_device.Get() != device.Get()) {
        return false;
      }

      D3D11_SHADER_RESOURCE_VIEW_DESC view_descriptor {};
      view->GetDesc(&view_descriptor);
      if (
        view_descriptor.ViewDimension !=
          D3D11_SRV_DIMENSION_BUFFER ||
        view_descriptor.Buffer.NumElements <
          rule_state_vector_count
      ) {
        return false;
      }

      ComPtr<ID3D11Resource> resource;
      view->GetResource(&resource);
      ComPtr<ID3D11Buffer> buffer;
      if (!resource || FAILED(resource.As(&buffer))) {
        return false;
      }
      D3D11_BUFFER_DESC buffer_descriptor {};
      buffer->GetDesc(&buffer_descriptor);
      return buffer_descriptor.ByteWidth >= rule_state_byte_width &&
             (
               buffer_descriptor.MiscFlags &
               D3D11_RESOURCE_MISC_BUFFER_STRUCTURED
             ) != 0u &&
             buffer_descriptor.StructureByteStride == uint4_stride;
    }

    [[nodiscard]] sbs_roi_shape_request_gpu_result poll() {
      sbs_roi_shape_request_gpu_result result;
      result.request = newest_completed;
      result.completed_sequence = newest_completed_sequence;
      result.completed_source_frame_id =
        newest_completed_source_frame_id;
      if (!ready) {
        result.failed = true;
        return result;
      }

      for (auto &slot : slots) {
        if (!slot.pending) {
          continue;
        }

        BOOL complete = FALSE;
        const auto query_status = context->GetData(
          slot.completion.Get(),
          &complete,
          sizeof(complete),
          D3D11_ASYNC_GETDATA_DONOTFLUSH
        );
        if (
          query_status == S_FALSE ||
          (SUCCEEDED(query_status) && !complete)
        ) {
          continue;
        }
        if (FAILED(query_status)) {
          slot.pending = false;
          result.failed = true;
          continue;
        }

        D3D11_MAPPED_SUBRESOURCE mapped {};
        const auto map_status = context->Map(
          slot.staging.Get(),
          0,
          D3D11_MAP_READ,
          D3D11_MAP_FLAG_DO_NOT_WAIT,
          &mapped
        );
        if (map_status == DXGI_ERROR_WAS_STILL_DRAWING) {
          continue;
        }
        slot.pending = false;
        if (FAILED(map_status) || !mapped.pData) {
          if (SUCCEEDED(map_status)) {
            context->Unmap(slot.staging.Get(), 0);
          }
          result.failed = true;
          continue;
        }

        sbs_roi_shape_request decoded;
        std::memcpy(&decoded, mapped.pData, sizeof(decoded));
        context->Unmap(slot.staging.Get(), 0);

        const bool expected_source =
          decoded.shape[0] == slot.source_width &&
          decoded.shape[1] == slot.source_height;
        const bool expected_active_backend =
          !sbs_roi_shape_has_flag(
            decoded,
            sbs_roi_shape_request_flag::active_roi
          ) ||
          decoded.identity[0] ==
            slot.expected_backend_generation;
        if (
          !expected_source ||
          !expected_active_backend ||
          !sbs_roi_shape_request_valid(decoded, slot.limits)
        ) {
          result.failed = true;
          continue;
        }

        if (
          !newest_completed ||
          slot.submission_sequence >=
            newest_completed_sequence
        ) {
          newest_completed = decoded;
          newest_completed_sequence =
            slot.submission_sequence;
          newest_completed_source_frame_id =
            slot.source_frame_id;
          result.request = decoded;
          result.completed_sequence =
            slot.submission_sequence;
          result.completed_source_frame_id =
            slot.source_frame_id;
          result.fresh_sample = true;
        }
      }

      return result;
    }

    [[nodiscard]] sbs_roi_shape_request_gpu_result submit(
      const sbs_roi_shape_request_gpu_submission &submission
    ) {
      auto result = poll();
      if (!ready || !submission_contract_valid(submission)) {
        result.failed = true;
        return result;
      }
      if (FAILED(device->GetDeviceRemovedReason())) {
        result.failed = true;
        return result;
      }

      slot_t *selected = nullptr;
      std::size_t selected_index = 0u;
      for (std::size_t offset = 0u; offset < slots.size(); ++offset) {
        const auto index =
          (next_slot + offset) % slots.size();
        if (!slots[index].pending) {
          selected = &slots[index];
          selected_index = index;
          break;
        }
      }
      if (!selected) {
        return result;
      }

      ID3D11ShaderResourceView *rule_state =
        submission.rule_state;
      if (
        rule_state &&
        !rule_state_view_usable(rule_state)
      ) {
        rule_state = zero_rule_state_srv.Get();
        result.failed = true;
      } else if (!rule_state) {
        rule_state = zero_rule_state_srv.Get();
      }

      const auto constants = make_constants(submission);
      context->UpdateSubresource(
        selected->constants.Get(),
        0,
        nullptr,
        &constants,
        0,
        0
      );

      ID3D11ShaderResourceView *inputs[] {rule_state};
      ID3D11UnorderedAccessView *outputs[] {
        output_uav.Get(),
      };
      ID3D11Buffer *constant_buffers[] {
        selected->constants.Get(),
      };
      context->CSSetShader(shader.Get(), nullptr, 0);
      context->CSSetShaderResources(0, 1, inputs);
      context->CSSetUnorderedAccessViews(
        0,
        1,
        outputs,
        nullptr
      );
      context->CSSetConstantBuffers(
        0,
        1,
        constant_buffers
      );
      context->Dispatch(1, 1, 1);

      ID3D11ShaderResourceView *null_inputs[] {nullptr};
      ID3D11UnorderedAccessView *null_outputs[] {nullptr};
      ID3D11Buffer *null_constant_buffers[] {nullptr};
      context->CSSetShaderResources(0, 1, null_inputs);
      context->CSSetUnorderedAccessViews(
        0,
        1,
        null_outputs,
        nullptr
      );
      context->CSSetConstantBuffers(
        0,
        1,
        null_constant_buffers
      );
      context->CSSetShader(nullptr, nullptr, 0);

      context->CopyResource(
        selected->staging.Get(),
        output.Get()
      );
      context->End(selected->completion.Get());

      selected->pending = true;
      selected->submission_sequence =
        next_submission_sequence++;
      if (next_submission_sequence == 0u) {
        next_submission_sequence = 1u;
      }
      selected->source_frame_id =
        submission.source_frame_id;
      selected->source_width = submission.source_width;
      selected->source_height = submission.source_height;
      selected->expected_backend_generation =
        submission.expected_backend_generation;
      selected->limits = {
        submission.canonical_model_width,
        submission.canonical_model_height,
        submission.profile_max_width,
        submission.profile_max_height,
        submission.max_model_aspect,
      };
      next_slot = (selected_index + 1u) % slots.size();
      result.copy_scheduled = true;

      if (FAILED(device->GetDeviceRemovedReason())) {
        selected->pending = false;
        result.failed = true;
        result.copy_scheduled = false;
      }
      return result;
    }

    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11ComputeShader> shader;
    ComPtr<ID3D11Buffer> output;
    ComPtr<ID3D11UnorderedAccessView> output_uav;
    ComPtr<ID3D11Buffer> zero_rule_state;
    ComPtr<ID3D11ShaderResourceView> zero_rule_state_srv;
    std::array<slot_t, readback_slot_count> slots;
    std::size_t next_slot = 0u;
    std::uint64_t next_submission_sequence = 1u;
    std::optional<sbs_roi_shape_request> newest_completed;
    std::uint64_t newest_completed_sequence = 0u;
    std::uint64_t newest_completed_source_frame_id = 0u;
    bool ready = false;
  };

  sbs_roi_shape_request_gpu::sbs_roi_shape_request_gpu(
    ComPtr<ID3D11Device> device,
    ComPtr<ID3D11DeviceContext> context,
    const std::filesystem::path &assets_dir
  ):
      pimpl_(std::make_unique<impl>(
        std::move(device),
        std::move(context),
        assets_dir
      )) {
  }

  sbs_roi_shape_request_gpu::~sbs_roi_shape_request_gpu() = default;

  bool sbs_roi_shape_request_gpu::valid() const {
    return pimpl_ && pimpl_->ready;
  }

  sbs_roi_shape_request_gpu_result
  sbs_roi_shape_request_gpu::submit(
    const sbs_roi_shape_request_gpu_submission &submission
  ) {
    if (!pimpl_) {
      sbs_roi_shape_request_gpu_result result;
      result.failed = true;
      return result;
    }
    return pimpl_->submit(submission);
  }

  sbs_roi_shape_request_gpu_result
  sbs_roi_shape_request_gpu::poll() {
    if (!pimpl_) {
      sbs_roi_shape_request_gpu_result result;
      result.failed = true;
      return result;
    }
    return pimpl_->poll();
  }
}  // namespace models
