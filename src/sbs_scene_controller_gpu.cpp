/**
 * @file src/sbs_scene_controller_gpu.cpp
 * @brief GPU-only deterministic scene-controller backend for Host SBS.
 */

#include "sbs_scene_controller_gpu.h"

#include "generated/sbs_scene_controller_contract.h"
#include "logging.h"
#include "sbs_perf.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <d3dcompiler.h>
#include <limits>
#include <map>
#include <mutex>
#include <string_view>
#include <utility>
#include <vector>

namespace {
  using Microsoft::WRL::ComPtr;

  struct shader_cache_entry_t {
    std::filesystem::file_time_type modified;
    std::shared_ptr<const std::vector<std::uint8_t>> bytecode;
  };

  std::mutex scene_shader_cache_mutex;
  std::map<std::filesystem::path, shader_cache_entry_t> scene_shader_cache;

  std::shared_ptr<const std::vector<std::uint8_t>> compile_scene_shader(
    const std::filesystem::path &path
  ) {
    std::error_code ec;
    const auto modified = std::filesystem::last_write_time(path, ec);
    {
      std::lock_guard lock(scene_shader_cache_mutex);
      const auto found = scene_shader_cache.find(path);
      if (
        found != scene_shader_cache.end() && !ec &&
        found->second.modified == modified
      ) {
        return found->second.bytecode;
      }
    }

    ComPtr<ID3DBlob> shader;
    ComPtr<ID3DBlob> diagnostics;
    constexpr DWORD flags =
      D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3;
    const HRESULT status = D3DCompileFromFile(
      path.wstring().c_str(),
      nullptr,
      D3D_COMPILE_STANDARD_FILE_INCLUDE,
      "main",
      "cs_5_0",
      flags,
      0,
      &shader,
      &diagnostics
    );
    if (FAILED(status)) {
      BOOST_LOG(error) << "Scene-controller shader compile error (" << path
                       << "): "
                       << (diagnostics ?
                             static_cast<const char *>(
                               diagnostics->GetBufferPointer()
                             ) :
                             "no compiler diagnostics");
      return {};
    }

    const auto *begin =
      static_cast<const std::uint8_t *>(shader->GetBufferPointer());
    auto bytecode = std::make_shared<const std::vector<std::uint8_t>>(
      begin,
      begin + shader->GetBufferSize()
    );
    if (!ec) {
      std::lock_guard lock(scene_shader_cache_mutex);
      scene_shader_cache.insert_or_assign(
        path,
        shader_cache_entry_t {modified, bytecode}
      );
    }
    return bytecode;
  }

  struct gpu_float_buffer_t {
    ComPtr<ID3D11Buffer> buffer;
    ComPtr<ID3D11ShaderResourceView> srv;
    ComPtr<ID3D11UnorderedAccessView> uav;
    std::size_t float_count = 0;
  };

  bool create_float_buffer(
    ID3D11Device *device,
    std::size_t float_count,
    gpu_float_buffer_t &out,
    const void *initial_values = nullptr,
    UINT structure_byte_stride = sizeof(float)
  ) {
    if (
      !device || float_count == 0 ||
      structure_byte_stride == 0 ||
      (float_count * sizeof(float)) % structure_byte_stride != 0 ||
      float_count >
        static_cast<std::size_t>(
          std::numeric_limits<UINT>::max() / sizeof(float)
        )
    ) {
      return false;
    }

    D3D11_BUFFER_DESC descriptor {};
    descriptor.Usage = D3D11_USAGE_DEFAULT;
    descriptor.ByteWidth =
      static_cast<UINT>(float_count * sizeof(float));
    descriptor.BindFlags =
      D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    descriptor.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    descriptor.StructureByteStride = structure_byte_stride;
    D3D11_SUBRESOURCE_DATA initial {
      initial_values,
      0,
      0,
    };
    if (
      FAILED(device->CreateBuffer(
        &descriptor,
        initial_values ? &initial : nullptr,
        &out.buffer
      )) ||
      FAILED(device->CreateShaderResourceView(
        out.buffer.Get(),
        nullptr,
        &out.srv
      )) ||
      FAILED(device->CreateUnorderedAccessView(
        out.buffer.Get(),
        nullptr,
        &out.uav
      ))
    ) {
      out = {};
      return false;
    }
    out.float_count = float_count;
    return true;
  }

  struct alignas(16) scene_constants_t {
    std::uint32_t source_width = 0;
    std::uint32_t source_height = 0;
    std::uint32_t depth_width = 0;
    std::uint32_t depth_height = 0;

    std::uint32_t color_mode = 0;
    std::uint32_t backend_generation = 1;
    std::uint32_t history_valid = 0;
    std::uint32_t reset_flags = 0;

    float elapsed_seconds = 0.0f;
    float pop_floor = 1.2f;
    float pop_ceiling = 2.0f;
    float zero_plane_mode = 2.0f;

    float acquire_seconds = 0.12f;
    float challenger_seconds = 0.30f;
    float release_seconds = 60.0f;
    float scroll_enter_seconds = 0.05f;

    std::array<std::uint32_t, 8> ordered_abi_hash_words {};
  };
  static_assert(sizeof(scene_constants_t) == 96);
}  // namespace

namespace models {
  struct sbs_scene_controller_gpu::impl {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    config::sbs_scene_controller_e backend =
      config::sbs_scene_controller_e::off;
    float pop_floor = 1.2f;
    float pop_ceiling = 2.0f;
    float zero_plane_mode = 2.0f;
    bool initialized = false;
    bool history_valid = false;
    bool prepared = false;
    bool pending = false;
    bool snapshot_available = false;
    std::uint64_t prepared_frame_id = 0;
    std::uint64_t pending_frame_id = 0;
    std::uint64_t completed_frame_id = 0;
    std::uint32_t source_width = 0;
    std::uint32_t source_height = 0;
    std::uint32_t pending_source_width = 0;
    std::uint32_t pending_source_height = 0;
    std::uint32_t completed_source_width = 0;
    std::uint32_t completed_source_height = 0;
    std::uint32_t prepared_reset_flags = 0;
    std::uint32_t pending_reset_flags = 0;
    input_color_space prepared_color_space = input_color_space::srgb;
    input_color_space pending_color_space = input_color_space::srgb;
    input_color_space completed_color_space = input_color_space::srgb;
    float prepared_elapsed_seconds = 0.0f;
    float pending_elapsed_seconds = 0.0f;
    std::chrono::steady_clock::time_point prepared_at {};
    std::chrono::steady_clock::time_point last_enqueued_at {};
    static constexpr std::uint32_t backend_generation = 1;

    ComPtr<ID3D11ComputeShader> prepare_cs;
    ComPtr<ID3D11ComputeShader> features_cs;
    ComPtr<ID3D11ComputeShader> evidence_cs;
    ComPtr<ID3D11ComputeShader> reduce_cs;
    ComPtr<ID3D11ComputeShader> resolve_cs;
    ComPtr<ID3D11ComputeShader> history_commit_cs;
    ComPtr<ID3D11SamplerState> linear_sampler;

    std::array<gpu_float_buffer_t, 2> scene_rgb;
    std::array<gpu_float_buffer_t, 2> scene_ordinal;
    std::array<gpu_float_buffer_t, 2> analysis_grid;
    std::array<gpu_float_buffer_t, 2> layout_history;
    std::array<gpu_float_buffer_t, 2> depth_history;
    gpu_float_buffer_t dense_output;
    gpu_float_buffer_t evidence_global;
    gpu_float_buffer_t global_output;
    gpu_float_buffer_t meta;
    gpu_float_buffer_t hidden_output;
    std::array<gpu_float_buffer_t, 2> rule_state;
    std::size_t prepared_scene_bank = 0;
    std::size_t pending_scene_bank = 0;
    std::size_t completed_scene_bank = 0;
    std::size_t current_history_bank = 0;
    std::size_t current_state_bank = 0;

    static constexpr std::size_t constant_buffer_ring_size = 4;
    std::array<ComPtr<ID3D11Buffer>, constant_buffer_ring_size> constant_buffers;
    std::size_t next_constant_buffer = 0;

    struct timing_slot_t {
      ComPtr<ID3D11Query> disjoint;
      ComPtr<ID3D11Query> start;
      ComPtr<ID3D11Query> end;
      bool pending = false;
      std::uint64_t perf_generation = 0;
    };
    struct timing_ring_t {
      std::array<timing_slot_t, 8> slots;
      std::size_t next = 0;
      bool ready = false;
      const char *stage = nullptr;
    };
    timing_ring_t prepare_timing;
    timing_ring_t rules_timing;

    impl(
      ComPtr<ID3D11Device> d,
      ComPtr<ID3D11DeviceContext> c,
      const std::filesystem::path &assets_dir,
      config::sbs_scene_controller_e selected_backend,
      const config::video_t::sbs_t &sbs_config
    ):
        device(std::move(d)),
        context(std::move(c)),
        backend(selected_backend),
        pop_floor(static_cast<float>(sbs_config.pop_strength)),
        pop_ceiling(static_cast<float>(
          sbs_config.adaptive_pop ?
            std::max(
              sbs_config.pop_strength,
              sbs_config.adaptive_pop_max
            ) :
            sbs_config.pop_strength
        )),
        zero_plane_mode(
          sbs_config.zero_plane == "subject" ? 1.0f :
          sbs_config.zero_plane == "background" ? 3.0f :
                                                   2.0f
        ) {
      if (backend == config::sbs_scene_controller_e::off) {
        initialized = true;
        return;
      }
      if (!device || !context) {
        return;
      }

      const auto shader_root =
        assets_dir / "shaders" / "directx";
      const auto create_shader =
        [&](std::string_view filename, ComPtr<ID3D11ComputeShader> &target) {
          const auto bytecode =
            compile_scene_shader(shader_root / filename);
          return bytecode &&
                 SUCCEEDED(device->CreateComputeShader(
                   bytecode->data(),
                   bytecode->size(),
                   nullptr,
                   &target
                 ));
        };
      if (
        !create_shader("sbs_scene_prepare_cs.hlsl", prepare_cs) ||
        !create_shader("sbs_scene_features_cs.hlsl", features_cs) ||
        !create_shader("sbs_scene_rules_evidence_cs.hlsl", evidence_cs) ||
        !create_shader("sbs_scene_rules_reduce_cs.hlsl", reduce_cs) ||
        !create_shader("sbs_scene_rules_resolve_cs.hlsl", resolve_cs) ||
        !create_shader(
          "sbs_scene_history_commit_cs.hlsl",
          history_commit_cs
        )
      ) {
        BOOST_LOG(error)
          << "Host SBS scene-controller initialization failed: rule shader compilation.";
        return;
      }

      D3D11_SAMPLER_DESC sampler_descriptor {};
      sampler_descriptor.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
      sampler_descriptor.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
      sampler_descriptor.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
      sampler_descriptor.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
      if (
        FAILED(device->CreateSamplerState(
          &sampler_descriptor,
          &linear_sampler
        ))
      ) {
        return;
      }

      constexpr std::size_t appearance_pixels =
        sbs_scene_controller::appearance_canvas_size *
        sbs_scene_controller::appearance_canvas_size;
      constexpr std::size_t analysis_pixels =
        sbs_scene_controller::analysis_canvas_size *
        sbs_scene_controller::analysis_canvas_size;
      const bool buffers_ready =
        create_float_buffer(
          device.Get(),
          3 * appearance_pixels,
          scene_rgb[0]
        ) &&
        create_float_buffer(
          device.Get(),
          3 * appearance_pixels,
          scene_rgb[1]
        ) &&
        create_float_buffer(
          device.Get(),
          appearance_pixels,
          scene_ordinal[0]
        ) &&
        create_float_buffer(
          device.Get(),
          appearance_pixels,
          scene_ordinal[1]
        ) &&
        create_float_buffer(
          device.Get(),
          sbs_scene_controller::analysis_grid_channel_count *
            analysis_pixels,
          analysis_grid[0]
        ) &&
        create_float_buffer(
          device.Get(),
          sbs_scene_controller::analysis_grid_channel_count *
            analysis_pixels,
          analysis_grid[1]
        ) &&
        create_float_buffer(
          device.Get(),
          sbs_scene_controller::layout_history_channel_count *
            analysis_pixels,
          layout_history[0]
        ) &&
        create_float_buffer(
          device.Get(),
          sbs_scene_controller::layout_history_channel_count *
            analysis_pixels,
          layout_history[1]
        ) &&
        create_float_buffer(
          device.Get(),
          sbs_scene_controller::depth_history_channel_count *
            analysis_pixels,
          depth_history[0]
        ) &&
        create_float_buffer(
          device.Get(),
          sbs_scene_controller::depth_history_channel_count *
            analysis_pixels,
          depth_history[1]
        ) &&
        create_float_buffer(
          device.Get(),
          sbs_scene_controller::dense_out_channel_count *
            analysis_pixels,
          dense_output
        ) &&
        create_float_buffer(
          device.Get(),
          sbs_scene_controller::global_out_word_count,
          evidence_global
        ) &&
        create_float_buffer(
          device.Get(),
          sbs_scene_controller::global_out_word_count,
          global_output
        ) &&
        create_float_buffer(
          device.Get(),
          sbs_scene_controller::meta_word_count,
          meta
        ) &&
        create_float_buffer(
          device.Get(),
          sbs_scene_controller::hidden_channel_count *
            sbs_scene_controller::recurrent_canvas_size *
            sbs_scene_controller::recurrent_canvas_size,
          hidden_output
        ) &&
        create_float_buffer(
          device.Get(),
          sbs_scene_controller::rule_state_word_count,
          rule_state[0],
          sbs_scene_controller::initial_word_bits.data(),
          4u * sizeof(float)
        ) &&
        create_float_buffer(
          device.Get(),
          sbs_scene_controller::rule_state_word_count,
          rule_state[1],
          sbs_scene_controller::initial_word_bits.data(),
          4u * sizeof(float)
        );
      if (!buffers_ready) {
        BOOST_LOG(error)
          << "Host SBS scene-controller initialization failed: fixed GPU resources.";
        return;
      }

      const float clear_float[4] = {};
      for (auto *buffer : {
             &scene_rgb[0],
             &scene_rgb[1],
             &scene_ordinal[0],
             &scene_ordinal[1],
             &analysis_grid[0],
             &analysis_grid[1],
             &layout_history[0],
             &layout_history[1],
             &depth_history[0],
             &depth_history[1],
             &dense_output,
             &evidence_global,
             &global_output,
             &meta,
             &hidden_output,
           }) {
        context->ClearUnorderedAccessViewFloat(buffer->uav.Get(), clear_float);
      }

      D3D11_BUFFER_DESC constant_descriptor {};
      constant_descriptor.Usage = D3D11_USAGE_DEFAULT;
      constant_descriptor.ByteWidth = sizeof(scene_constants_t);
      constant_descriptor.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      constant_descriptor.CPUAccessFlags = 0;
      for (auto &constant_buffer : constant_buffers) {
        if (
          FAILED(device->CreateBuffer(
            &constant_descriptor,
            nullptr,
            &constant_buffer
          ))
        ) {
          BOOST_LOG(error)
            << "Host SBS scene-controller initialization failed: constant-buffer ring.";
          return;
        }
      }

      initialize_timing(prepare_timing, "scene_prepare_gpu");
      initialize_timing(rules_timing, "scene_rules_gpu");
      initialized = true;
      BOOST_LOG(info)
        << "Host SBS GPU scene controller enabled in shadow_rules mode (ABI "
        << sbs_scene_controller::ordered_abi_hash << ", rule "
        << sbs_scene_controller::rule_revision << ").";
    }

    bool is_enabled() const {
      return backend != config::sbs_scene_controller_e::off;
    }

    bool is_valid() const {
      return initialized;
    }

    void initialize_timing(timing_ring_t &ring, const char *stage) {
      ring.stage = stage;
      if (!config::sunshine.diagnostics_enabled) {
        return;
      }
      for (auto &slot : ring.slots) {
        D3D11_QUERY_DESC descriptor {
          D3D11_QUERY_TIMESTAMP_DISJOINT,
          0,
        };
        if (FAILED(device->CreateQuery(&descriptor, &slot.disjoint))) {
          return;
        }
        descriptor.Query = D3D11_QUERY_TIMESTAMP;
        if (
          FAILED(device->CreateQuery(&descriptor, &slot.start)) ||
          FAILED(device->CreateQuery(&descriptor, &slot.end))
        ) {
          return;
        }
      }
      ring.ready = true;
    }

    void resolve_timing(timing_ring_t &ring) {
      if (!ring.ready) {
        return;
      }
      for (auto &slot : ring.slots) {
        if (!slot.pending) {
          continue;
        }
        D3D11_QUERY_DATA_TIMESTAMP_DISJOINT disjoint {};
        const auto disjoint_status = context->GetData(
          slot.disjoint.Get(),
          &disjoint,
          sizeof(disjoint),
          D3D11_ASYNC_GETDATA_DONOTFLUSH
        );
        if (disjoint_status == S_FALSE) {
          continue;
        }
        if (FAILED(disjoint_status)) {
          slot.pending = false;
          continue;
        }
        UINT64 start = 0;
        UINT64 end = 0;
        const auto start_status = context->GetData(
          slot.start.Get(),
          &start,
          sizeof(start),
          D3D11_ASYNC_GETDATA_DONOTFLUSH
        );
        const auto end_status = context->GetData(
          slot.end.Get(),
          &end,
          sizeof(end),
          D3D11_ASYNC_GETDATA_DONOTFLUSH
        );
        if (start_status == S_FALSE || end_status == S_FALSE) {
          continue;
        }
        if (
          start_status == S_OK && end_status == S_OK &&
          !disjoint.Disjoint && disjoint.Frequency != 0 && end >= start
        ) {
          sbs_perf::add_sample_ms_if_current(
            ring.stage,
            static_cast<double>(end - start) * 1000.0 /
              static_cast<double>(disjoint.Frequency),
            slot.perf_generation
          );
        }
        slot.pending = false;
      }
    }

    timing_slot_t *begin_timing(timing_ring_t &ring) {
      resolve_timing(ring);
      if (!ring.ready) {
        return nullptr;
      }
      for (std::size_t offset = 0; offset < ring.slots.size(); ++offset) {
        const std::size_t index = (ring.next + offset) % ring.slots.size();
        auto &slot = ring.slots[index];
        if (slot.pending) {
          continue;
        }
        ring.next = (index + 1) % ring.slots.size();
        slot.perf_generation = sbs_perf::generation();
        context->Begin(slot.disjoint.Get());
        context->End(slot.start.Get());
        return &slot;
      }
      return nullptr;
    }

    void end_timing(timing_slot_t *slot) {
      if (!slot) {
        return;
      }
      context->End(slot->end.Get());
      context->End(slot->disjoint.Get());
      slot->pending = true;
    }

    ID3D11Buffer *upload_constants(const scene_constants_t &values) {
      auto &buffer = constant_buffers[next_constant_buffer];
      next_constant_buffer =
        (next_constant_buffer + 1) % constant_buffers.size();
      // Full-buffer UpdateSubresource keeps the live path free of Map/Unmap and lets the
      // driver stage or rename the small ring element without a CPU/GPU readback boundary.
      context->UpdateSubresource(
        buffer.Get(),
        0,
        nullptr,
        &values,
        0,
        0
      );
      return buffer.Get();
    }

    scene_constants_t constants(
      std::uint32_t depth_width = 0,
      std::uint32_t depth_height = 0
    ) const {
      scene_constants_t values;
      values.source_width = pending ? pending_source_width : source_width;
      values.source_height = pending ? pending_source_height : source_height;
      values.depth_width = depth_width;
      values.depth_height = depth_height;
      values.color_mode = static_cast<std::uint32_t>(
        pending ? pending_color_space : prepared_color_space
      );
      values.backend_generation = backend_generation;
      values.history_valid = history_valid ? 1u : 0u;
      values.reset_flags =
        pending ? pending_reset_flags : prepared_reset_flags;
      values.elapsed_seconds =
        pending ? pending_elapsed_seconds : prepared_elapsed_seconds;
      values.pop_floor = pop_floor;
      values.pop_ceiling = pop_ceiling;
      values.zero_plane_mode = zero_plane_mode;
      values.ordered_abi_hash_words =
        sbs_scene_controller::ordered_abi_hash_words;
      return values;
    }

    bool prepare(
      ID3D11ShaderResourceView *input,
      input_color_space color_space,
      std::uint64_t source_frame_id
    ) {
      if (!is_enabled()) {
        return true;
      }
      if (!initialized || !input || pending || prepared) {
        return false;
      }

      ComPtr<ID3D11Resource> resource;
      input->GetResource(&resource);
      ComPtr<ID3D11Texture2D> texture;
      D3D11_TEXTURE2D_DESC descriptor {};
      if (
        FAILED(resource.As(&texture)) || !texture ||
        (texture->GetDesc(&descriptor), descriptor.Width == 0) ||
        descriptor.Height == 0
      ) {
        return false;
      }

      prepared_at = std::chrono::steady_clock::now();
      prepared_elapsed_seconds =
        last_enqueued_at.time_since_epoch().count() == 0 ?
          0.0f :
          std::clamp(
            std::chrono::duration<float>(
              prepared_at - last_enqueued_at
            ).count(),
            0.0f,
            1.0f
          );
      prepared_color_space = color_space;
      prepared_frame_id = source_frame_id;
      source_width = descriptor.Width;
      source_height = descriptor.Height;
      prepared_reset_flags = 0;
      if (!history_valid) {
        prepared_reset_flags =
          sbs_scene_controller::reset_flags_layout |
          sbs_scene_controller::reset_flags_depth_shot |
          sbs_scene_controller::reset_flags_backend;
      } else if (
        source_width != completed_source_width ||
        source_height != completed_source_height ||
        prepared_color_space != completed_color_space
      ) {
        prepared_reset_flags =
          sbs_scene_controller::reset_flags_layout |
          sbs_scene_controller::reset_flags_depth_shot |
          sbs_scene_controller::reset_flags_geometry |
          sbs_scene_controller::reset_flags_display_or_hdr;
      }
      prepared_scene_bank = snapshot_available ?
                              1u - completed_scene_bank :
                              (pending ? 1u - pending_scene_bank : 0u);

      const auto values = constants();
      ID3D11Buffer *constant_buffer = upload_constants(values);
      if (!constant_buffer) {
        return false;
      }

      auto *timer = begin_timing(prepare_timing);
      context->CSSetShader(prepare_cs.Get(), nullptr, 0);
      context->CSSetConstantBuffers(0, 1, &constant_buffer);
      context->CSSetShaderResources(0, 1, &input);
      ID3D11UnorderedAccessView *prepare_outputs[2] = {
        scene_rgb[prepared_scene_bank].uav.Get(),
        scene_ordinal[prepared_scene_bank].uav.Get(),
      };
      context->CSSetUnorderedAccessViews(
        0,
        static_cast<UINT>(std::size(prepare_outputs)),
        prepare_outputs,
        nullptr
      );
      context->CSSetSamplers(0, 1, linear_sampler.GetAddressOf());
      constexpr UINT appearance_groups =
        (sbs_scene_controller::appearance_canvas_size + 15u) / 16u;
      context->Dispatch(appearance_groups, appearance_groups, 1);

      ID3D11ShaderResourceView *null_prepare_inputs[1] = {};
      ID3D11UnorderedAccessView *null_prepare_outputs[2] = {};
      context->CSSetShaderResources(0, 1, null_prepare_inputs);
      context->CSSetUnorderedAccessViews(0, 2, null_prepare_outputs, nullptr);

      context->CSSetShader(features_cs.Get(), nullptr, 0);
      ID3D11ShaderResourceView *feature_inputs[4] = {
        scene_rgb[prepared_scene_bank].srv.Get(),
        scene_ordinal[prepared_scene_bank].srv.Get(),
        layout_history[current_history_bank].srv.Get(),
        rule_state[current_state_bank].srv.Get(),
      };
      context->CSSetShaderResources(0, 4, feature_inputs);
      context->CSSetUnorderedAccessViews(
        0,
        1,
        analysis_grid[prepared_scene_bank].uav.GetAddressOf(),
        nullptr
      );
      constexpr UINT analysis_groups =
        (sbs_scene_controller::analysis_canvas_size + 15u) / 16u;
      context->Dispatch(analysis_groups, analysis_groups, 1);

      ID3D11ShaderResourceView *null_feature_inputs[4] = {};
      ID3D11UnorderedAccessView *null_feature_output[1] = {};
      context->CSSetShaderResources(0, 4, null_feature_inputs);
      context->CSSetUnorderedAccessViews(
        0,
        1,
        null_feature_output,
        nullptr
      );
      context->CSSetShader(nullptr, nullptr, 0);
      end_timing(timer);
      prepared = true;
      return true;
    }

    void commit_prepared(std::uint64_t source_frame_id) {
      if (!is_enabled()) {
        return;
      }
      if (!prepared || prepared_frame_id != source_frame_id) {
        snapshot_available = false;
        return;
      }
      pending = true;
      prepared = false;
      pending_scene_bank = prepared_scene_bank;
      pending_frame_id = prepared_frame_id;
      pending_source_width = source_width;
      pending_source_height = source_height;
      pending_color_space = prepared_color_space;
      pending_reset_flags = prepared_reset_flags;
      pending_elapsed_seconds = prepared_elapsed_seconds;
      last_enqueued_at = prepared_at;
    }

    void discard(std::uint64_t source_frame_id) {
      if (
        prepared &&
        (source_frame_id == 0 || source_frame_id == prepared_frame_id)
      ) {
        prepared = false;
      }
    }

    bool resolve(
      std::uint64_t source_frame_id,
      ID3D11ShaderResourceView *roi_rgb_tensor,
      ID3D11ShaderResourceView *raw_depth,
      ID3D11ShaderResourceView *normalized_depth,
      ID3D11ShaderResourceView *depth_frame_state,
      ID3D11ShaderResourceView *adaptive_state,
      int depth_width,
      int depth_height
    ) {
      if (!is_enabled()) {
        return true;
      }
      if (
        !initialized || !pending || pending_frame_id != source_frame_id ||
        !roi_rgb_tensor || !raw_depth || !normalized_depth ||
        !depth_frame_state ||
        !adaptive_state || depth_width <= 0 || depth_height <= 0
      ) {
        snapshot_available = false;
        if (pending && pending_frame_id == source_frame_id) {
          pending = false;
        }
        return false;
      }

      const auto values = constants(
        static_cast<std::uint32_t>(depth_width),
        static_cast<std::uint32_t>(depth_height)
      );
      ID3D11Buffer *constant_buffer = upload_constants(values);
      if (!constant_buffer) {
        snapshot_available = false;
        pending = false;
        return false;
      }

      const std::size_t next_history_bank = 1u - current_history_bank;
      const std::size_t next_state_bank = 1u - current_state_bank;
      auto *timer = begin_timing(rules_timing);

      context->CSSetShader(evidence_cs.Get(), nullptr, 0);
      context->CSSetConstantBuffers(0, 1, &constant_buffer);
      ID3D11ShaderResourceView *evidence_inputs[9] = {
        analysis_grid[pending_scene_bank].srv.Get(),
        layout_history[current_history_bank].srv.Get(),
        normalized_depth,
        depth_history[current_history_bank].srv.Get(),
        depth_frame_state,
        rule_state[current_state_bank].srv.Get(),
        raw_depth,
        roi_rgb_tensor,
        adaptive_state,
      };
      ID3D11UnorderedAccessView *evidence_outputs[4] = {
        dense_output.uav.Get(),
        layout_history[next_history_bank].uav.Get(),
        depth_history[next_history_bank].uav.Get(),
        meta.uav.Get(),
      };
      context->CSSetShaderResources(0, 9, evidence_inputs);
      context->CSSetUnorderedAccessViews(0, 4, evidence_outputs, nullptr);
      constexpr UINT analysis_groups =
        (sbs_scene_controller::analysis_canvas_size + 15u) / 16u;
      context->Dispatch(analysis_groups, analysis_groups, 1);

      ID3D11ShaderResourceView *null_evidence_inputs[9] = {};
      ID3D11UnorderedAccessView *null_evidence_outputs[4] = {};
      context->CSSetShaderResources(0, 9, null_evidence_inputs);
      context->CSSetUnorderedAccessViews(
        0,
        4,
        null_evidence_outputs,
        nullptr
      );

      context->CSSetShader(reduce_cs.Get(), nullptr, 0);
      ID3D11ShaderResourceView *reduce_inputs[7] = {
        analysis_grid[pending_scene_bank].srv.Get(),
        dense_output.srv.Get(),
        layout_history[next_history_bank].srv.Get(),
        depth_history[next_history_bank].srv.Get(),
        adaptive_state,
        meta.srv.Get(),
        rule_state[current_state_bank].srv.Get(),
      };
      context->CSSetShaderResources(0, 7, reduce_inputs);
      context->CSSetUnorderedAccessViews(
        0,
        1,
        evidence_global.uav.GetAddressOf(),
        nullptr
      );
      context->Dispatch(1, 1, 1);

      ID3D11ShaderResourceView *null_reduce_inputs[7] = {};
      ID3D11UnorderedAccessView *null_reduce_output[1] = {};
      context->CSSetShaderResources(0, 7, null_reduce_inputs);
      context->CSSetUnorderedAccessViews(
        0,
        1,
        null_reduce_output,
        nullptr
      );

      context->CSSetShader(resolve_cs.Get(), nullptr, 0);
      ID3D11ShaderResourceView *resolve_inputs[4] = {
        evidence_global.srv.Get(),
        rule_state[current_state_bank].srv.Get(),
        meta.srv.Get(),
        adaptive_state,
      };
      ID3D11UnorderedAccessView *resolve_outputs[2] = {
        global_output.uav.Get(),
        rule_state[next_state_bank].uav.Get(),
      };
      context->CSSetShaderResources(0, 4, resolve_inputs);
      context->CSSetUnorderedAccessViews(0, 2, resolve_outputs, nullptr);
      context->Dispatch(1, 1, 1);

      ID3D11ShaderResourceView *null_resolve_inputs[4] = {};
      ID3D11UnorderedAccessView *null_resolve_outputs[2] = {};
      context->CSSetShaderResources(0, 4, null_resolve_inputs);
      context->CSSetUnorderedAccessViews(
        0,
        2,
        null_resolve_outputs,
        nullptr
      );
      context->CSSetShader(nullptr, nullptr, 0);

      // The evidence pass writes candidate next histories. Promotion is a GPU decision: this
      // final pass preserves the prior banks during scroll/invalid output and performs shot
      // resets without a CPU readback or an encode-thread branch.
      context->CSSetShader(history_commit_cs.Get(), nullptr, 0);
      ID3D11ShaderResourceView *history_commit_inputs[3] = {
        layout_history[current_history_bank].srv.Get(),
        depth_history[current_history_bank].srv.Get(),
        rule_state[next_state_bank].srv.Get(),
      };
      ID3D11UnorderedAccessView *history_commit_outputs[2] = {
        layout_history[next_history_bank].uav.Get(),
        depth_history[next_history_bank].uav.Get(),
      };
      context->CSSetShaderResources(0, 3, history_commit_inputs);
      context->CSSetUnorderedAccessViews(
        0,
        2,
        history_commit_outputs,
        nullptr
      );
      context->Dispatch(analysis_groups, analysis_groups, 1);

      ID3D11ShaderResourceView *null_history_commit_inputs[3] = {};
      ID3D11UnorderedAccessView *null_history_commit_outputs[2] = {};
      context->CSSetShaderResources(
        0,
        3,
        null_history_commit_inputs
      );
      context->CSSetUnorderedAccessViews(
        0,
        2,
        null_history_commit_outputs,
        nullptr
      );
      context->CSSetShader(nullptr, nullptr, 0);
      end_timing(timer);

      current_history_bank = next_history_bank;
      current_state_bank = next_state_bank;
      completed_scene_bank = pending_scene_bank;
      completed_source_width = pending_source_width;
      completed_source_height = pending_source_height;
      completed_color_space = pending_color_space;
      history_valid = true;
      pending = false;
      snapshot_available = true;
      completed_frame_id = source_frame_id;
      return true;
    }

    scene_controller_gpu_snapshot make_snapshot() const {
      scene_controller_gpu_snapshot result;
      if (!is_enabled() || !initialized) {
        return result;
      }
      result.scene_rgb = scene_rgb[completed_scene_bank].srv;
      result.analysis_grid = analysis_grid[completed_scene_bank].srv;
      result.dense_output = dense_output.srv;
      result.global_output = global_output.srv;
      result.layout_history = layout_history[current_history_bank].srv;
      result.depth_history = depth_history[current_history_bank].srv;
      result.hidden_output = hidden_output.srv;
      result.meta = meta.srv;
      result.rule_state = rule_state[current_state_bank].srv;
      result.source_frame_id = completed_frame_id;
      result.backend_generation = backend_generation;
      result.snapshot_available = snapshot_available;
      result.shadow = true;
      return result;
    }
  };

  sbs_scene_controller_gpu::sbs_scene_controller_gpu(
    ComPtr<ID3D11Device> device,
    ComPtr<ID3D11DeviceContext> context,
    const std::filesystem::path &assets_dir,
    config::sbs_scene_controller_e backend,
    const config::video_t::sbs_t &sbs_config
  ):
      pimpl_(std::make_unique<impl>(
        std::move(device),
        std::move(context),
        assets_dir,
        backend,
        sbs_config
      )) {}

  sbs_scene_controller_gpu::~sbs_scene_controller_gpu() = default;

  bool sbs_scene_controller_gpu::enabled() const {
    return pimpl_ && pimpl_->is_enabled();
  }

  bool sbs_scene_controller_gpu::valid() const {
    return pimpl_ && pimpl_->is_valid();
  }

  bool sbs_scene_controller_gpu::prepare_scene(
    ID3D11ShaderResourceView *input,
    input_color_space color_space,
    std::uint64_t source_frame_id
  ) {
    return pimpl_ &&
           pimpl_->prepare(input, color_space, source_frame_id);
  }

  void sbs_scene_controller_gpu::mark_enqueued(
    std::uint64_t source_frame_id
  ) {
    if (pimpl_) {
      pimpl_->commit_prepared(source_frame_id);
    }
  }

  void sbs_scene_controller_gpu::discard_prepared(
    std::uint64_t source_frame_id
  ) {
    if (pimpl_) {
      pimpl_->discard(source_frame_id);
    }
  }

  bool sbs_scene_controller_gpu::resolve_completed(
    std::uint64_t source_frame_id,
    ID3D11ShaderResourceView *roi_rgb_tensor,
    ID3D11ShaderResourceView *raw_depth,
    ID3D11ShaderResourceView *normalized_depth,
    ID3D11ShaderResourceView *depth_frame_state,
    ID3D11ShaderResourceView *adaptive_state,
    int depth_width,
    int depth_height
  ) {
    return pimpl_ &&
           pimpl_->resolve(
             source_frame_id,
             roi_rgb_tensor,
             raw_depth,
             normalized_depth,
             depth_frame_state,
             adaptive_state,
             depth_width,
             depth_height
           );
  }

  scene_controller_gpu_snapshot sbs_scene_controller_gpu::snapshot() const {
    return pimpl_ ? pimpl_->make_snapshot() :
                    scene_controller_gpu_snapshot {};
  }
}  // namespace models
