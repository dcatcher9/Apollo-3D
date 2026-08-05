/**
 * @file tests/unit/test_host_sbs_cut_only_gpu.cpp
 * @brief Executes the live Host SBS cut-only resolver on the D3D11 WARP device.
 */
#include "../tests_common.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <src/generated/sbs_adaptive_state_contract.h>

#ifdef _WIN32
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;

  template<typename T, std::size_t Size>
  bool create_structured_uav(
    ID3D11Device *device,
    const std::array<T, Size> &initial,
    const UINT stride,
    ComPtr<ID3D11Buffer> &buffer,
    ComPtr<ID3D11UnorderedAccessView> &uav
  ) {
    D3D11_BUFFER_DESC desc {};
    desc.ByteWidth = static_cast<UINT>(sizeof(initial));
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = stride;
    D3D11_SUBRESOURCE_DATA data {initial.data(), 0, 0};
    return SUCCEEDED(device->CreateBuffer(&desc, &data, &buffer)) &&
           SUCCEEDED(device->CreateUnorderedAccessView(buffer.Get(), nullptr, &uav));
  }

  template<typename T, std::size_t Size>
  bool read_buffer(
    ID3D11Device *device,
    ID3D11DeviceContext *context,
    ID3D11Buffer *source,
    std::array<T, Size> &output
  ) {
    D3D11_BUFFER_DESC desc {};
    source->GetDesc(&desc);
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0;
    desc.StructureByteStride = 0;
    ComPtr<ID3D11Buffer> staging;
    if (FAILED(device->CreateBuffer(&desc, nullptr, &staging))) {
      return false;
    }
    context->CopyResource(staging.Get(), source);
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
      return false;
    }
    std::memcpy(output.data(), mapped.pData, sizeof(output));
    context->Unmap(staging.Get(), 0);
    return true;
  }
}

TEST(HostSbsCutOnlyGpuTest, ResolvesCutPulseAndClearsEvidence) {
  ComPtr<ID3D11Device> device;
  ComPtr<ID3D11DeviceContext> context;
  D3D_FEATURE_LEVEL feature_level {};
  constexpr D3D_FEATURE_LEVEL requested_levels[] = {D3D_FEATURE_LEVEL_11_0};
  ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(
    nullptr,
    D3D_DRIVER_TYPE_WARP,
    nullptr,
    0,
    requested_levels,
    static_cast<UINT>(std::size(requested_levels)),
    D3D11_SDK_VERSION,
    &device,
    &feature_level,
    &context
  )));

  const std::filesystem::path shader_path =
    SUNSHINE_SHADERS_DIR "/depth_scene_cut_resolve_cs.hlsl";
  ComPtr<ID3DBlob> shader_blob;
  ComPtr<ID3DBlob> shader_errors;
  const auto compile_status = D3DCompileFromFile(
    shader_path.c_str(),
    nullptr,
    D3D_COMPILE_STANDARD_FILE_INCLUDE,
    "main",
    "cs_5_0",
    D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
    0,
    &shader_blob,
    &shader_errors
  );
  ASSERT_TRUE(SUCCEEDED(compile_status))
    << (shader_errors ?
          static_cast<const char *>(shader_errors->GetBufferPointer()) :
          "no compiler diagnostics");

  ComPtr<ID3D11ComputeShader> shader;
  ASSERT_TRUE(SUCCEEDED(device->CreateComputeShader(
    shader_blob->GetBufferPointer(),
    shader_blob->GetBufferSize(),
    nullptr,
    &shader
  )));

  auto state = sbs_adaptive_state::initial_values;
  const auto index = sbs_adaptive_state::index;
  state[index(sbs_adaptive_state::word_e::initialized)] = 1.0f;
  state[index(sbs_adaptive_state::word_e::scene_age)] = 8.0f;
  state[index(sbs_adaptive_state::word_e::depth_change_baseline_ema)] = 0.05f;
  state[index(sbs_adaptive_state::word_e::cut_flags)] =
    static_cast<float>(sbs_adaptive_state::cut_flag_appearance_armed);
  state[index(sbs_adaptive_state::word_e::model_input_history_state)] = 1.0f;
  state[index(sbs_adaptive_state::word_e::appearance_change_baseline_ema)] = 0.05f;

  // Shared evidence contract order: total, depth, ordinal, RGB, current/previous/common support,
  // brightness rise/fall. This is a broad semantic replacement, not an exposure-like transition.
  std::array<std::uint32_t, 10> evidence {
    100u, 30u, 10u, 80u, 100u, 100u, 100u, 40u, 40u, 1u
  };
  ComPtr<ID3D11Buffer> state_buffer;
  ComPtr<ID3D11UnorderedAccessView> state_uav;
  ComPtr<ID3D11Buffer> evidence_buffer;
  ComPtr<ID3D11UnorderedAccessView> evidence_uav;
  ASSERT_TRUE(create_structured_uav(
    device.Get(),
    state,
    4u * sizeof(float),
    state_buffer,
    state_uav
  ));
  ASSERT_TRUE(create_structured_uav(
    device.Get(),
    evidence,
    sizeof(std::uint32_t),
    evidence_buffer,
    evidence_uav
  ));

  const auto dispatch = [&]() {
    ID3D11UnorderedAccessView *uavs[] = {state_uav.Get(), evidence_uav.Get()};
    context->CSSetShader(shader.Get(), nullptr, 0);
    context->CSSetUnorderedAccessViews(0, 2, uavs, nullptr);
    context->Dispatch(1, 1, 1);
    ID3D11UnorderedAccessView *null_uavs[] = {nullptr, nullptr};
    context->CSSetUnorderedAccessViews(0, 2, null_uavs, nullptr);
  };

  dispatch();
  std::array<float, sbs_adaptive_state::word_count> resolved {};
  std::array<std::uint32_t, 10> cleared {};
  ASSERT_TRUE(read_buffer(device.Get(), context.Get(), state_buffer.Get(), resolved));
  ASSERT_TRUE(read_buffer(device.Get(), context.Get(), evidence_buffer.Get(), cleared));
  EXPECT_FLOAT_EQ(resolved[index(sbs_adaptive_state::word_e::scene_age)], 0.0f);
  EXPECT_EQ(
    std::bit_cast<std::uint32_t>(
      resolved[index(sbs_adaptive_state::word_e::cut_contract_tag_bits)]
    ),
    sbs_adaptive_state::cut_contract_tag
  );
  EXPECT_FLOAT_EQ(
    resolved[index(sbs_adaptive_state::word_e::cut_flags)],
    static_cast<float>(sbs_adaptive_state::cut_flag_latched)
  );
  EXPECT_FLOAT_EQ(resolved[index(sbs_adaptive_state::word_e::hard_cut_pulse)], 1.0f);
  EXPECT_EQ(
    std::bit_cast<std::uint32_t>(resolved[index(sbs_adaptive_state::word_e::hard_cut_count)]),
    1u
  );
  EXPECT_TRUE(std::ranges::all_of(cleared, [](const auto value) { return value == 0u; }));

  // An invalid/empty update clears only the one-frame pulse and preserves the generation count.
  dispatch();
  ASSERT_TRUE(read_buffer(device.Get(), context.Get(), state_buffer.Get(), resolved));
  EXPECT_FLOAT_EQ(resolved[index(sbs_adaptive_state::word_e::hard_cut_pulse)], 0.0f);
  EXPECT_EQ(
    std::bit_cast<std::uint32_t>(resolved[index(sbs_adaptive_state::word_e::hard_cut_count)]),
    1u
  );

  // A same-sized foreign cut buffer is replaced from the generated schema defaults.
  auto corrupt_state = sbs_adaptive_state::initial_values;
  corrupt_state[index(sbs_adaptive_state::word_e::cut_contract_tag_bits)] = 0.0f;
  corrupt_state[index(sbs_adaptive_state::word_e::initialized)] = 1.0f;
  corrupt_state[index(sbs_adaptive_state::word_e::scene_age)] = 123.0f;
  context->UpdateSubresource(
    state_buffer.Get(), 0, nullptr, corrupt_state.data(), 0, 0);
  dispatch();
  ASSERT_TRUE(read_buffer(device.Get(), context.Get(), state_buffer.Get(), resolved));
  EXPECT_EQ(
    std::bit_cast<std::uint32_t>(
      resolved[index(sbs_adaptive_state::word_e::cut_contract_tag_bits)]
    ),
    sbs_adaptive_state::cut_contract_tag
  );
  EXPECT_FLOAT_EQ(resolved[index(sbs_adaptive_state::word_e::initialized)], 0.0f);
  EXPECT_FLOAT_EQ(resolved[index(sbs_adaptive_state::word_e::scene_age)], 0.0f);

  // Rearm age follows source-stream frames even when depth observations were skipped.
  auto aged_state = sbs_adaptive_state::initial_values;
  aged_state[index(sbs_adaptive_state::word_e::initialized)] = 1.0f;
  std::array<std::uint32_t, 10> aged_evidence {
    100u, 0u, 0u, 0u, 100u, 100u, 100u, 0u, 0u, 8u
  };
  context->UpdateSubresource(state_buffer.Get(), 0, nullptr, aged_state.data(), 0, 0);
  context->UpdateSubresource(
    evidence_buffer.Get(), 0, nullptr, aged_evidence.data(), 0, 0);
  dispatch();
  ASSERT_TRUE(read_buffer(device.Get(), context.Get(), state_buffer.Get(), resolved));
  EXPECT_FLOAT_EQ(resolved[index(sbs_adaptive_state::word_e::scene_age)], 8.0f);
  EXPECT_FLOAT_EQ(
    resolved[index(sbs_adaptive_state::word_e::cut_flags)],
    static_cast<float>(
      sbs_adaptive_state::cut_flag_geometry_armed |
      sbs_adaptive_state::cut_flag_appearance_armed
    )
  );
}
#endif
