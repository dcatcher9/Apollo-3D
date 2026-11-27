/**
 * @file tests/unit/test_host_sbs_compute_state_guard_gpu.cpp
 * @brief Sentinel tests for the compute-state guard of each Host SBS pass family.
 */
#ifdef _WIN32
  #include "src/host_sbs_compute_state_guard.h"
  #include "src/host_sbs_direct_glyph_gpu.h"
  #include "src/host_sbs_direct_glyph_publication_gpu.h"
  #include "src/host_sbs_display_snapshot_rgb8_analysis_gpu.h"

  #include <d3dcompiler.h>
  #include <gtest/gtest.h>
  #include <type_traits>
  #include <vector>

namespace {
  using Microsoft::WRL::ComPtr;
  namespace guard = models::host_sbs_compute_state;

  // Each pass family restores exactly the slots it binds.
  static_assert(std::is_same_v<models::host_sbs_direct_glyph_gpu::detail::palette_state_t, guard::guard_t<16u, 5u, 1u>>);
  static_assert(std::is_same_v<models::host_sbs_direct_glyph_publication::compute_state_t, guard::guard_t<25u, 8u, 3u>>);
  static_assert(std::is_same_v<models::display_snapshot_rgb8_analysis_gpu_v1::compute_state_t, guard::guard_t<16u, 8u, 3u>>);

  template<class Guard>
  struct counts_t;

  template<UINT Srvs, UINT Uavs, UINT Constants>
  struct counts_t<guard::guard_t<Srvs, Uavs, Constants>> {
    static constexpr UINT srvs = Srvs, uavs = Uavs, constants = Constants;
  };

  struct views_t {
    std::vector<ComPtr<ID3D11Buffer>> buffers;
    std::vector<ComPtr<ID3D11ShaderResourceView>> srvs;
    std::vector<ComPtr<ID3D11UnorderedAccessView>> uavs;
  };

  views_t structured(ID3D11Device *device, UINT count) {
    views_t views;
    for (UINT i = 0u; i < count; ++i) {
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = 16u;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      desc.StructureByteStride = 4u;
      auto &buffer = views.buffers.emplace_back();
      auto &srv = views.srvs.emplace_back();
      auto &uav = views.uavs.emplace_back();
      EXPECT_HRESULT_SUCCEEDED(device->CreateBuffer(&desc, nullptr, &buffer));
      EXPECT_HRESULT_SUCCEEDED(device->CreateShaderResourceView(buffer.Get(), nullptr, &srv));
      EXPECT_HRESULT_SUCCEEDED(device->CreateUnorderedAccessView(buffer.Get(), nullptr, &uav));
    }
    return views;
  }

  std::vector<ComPtr<ID3D11Buffer>> constants(ID3D11Device *device, UINT count) {
    std::vector<ComPtr<ID3D11Buffer>> buffers(count);
    D3D11_BUFFER_DESC desc {};
    desc.ByteWidth = 16u;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    for (auto &buffer : buffers) {
      EXPECT_HRESULT_SUCCEEDED(device->CreateBuffer(&desc, nullptr, &buffer));
    }
    return buffers;
  }

  template<class T>
  std::vector<T *> raw(const std::vector<ComPtr<T>> &objects) {
    std::vector<T *> pointers;
    for (const auto &object : objects) {
      pointers.push_back(object.Get());
    }
    return pointers;
  }

  template<class Guard>
  void expect_exact_restore() {
    constexpr auto srvs = counts_t<Guard>::srvs, uavs = counts_t<Guard>::uavs, cbs = counts_t<Guard>::constants;
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    D3D_FEATURE_LEVEL level {};
    const D3D_FEATURE_LEVEL wanted = D3D_FEATURE_LEVEL_11_1;
    ASSERT_HRESULT_SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0u, &wanted, 1u, D3D11_SDK_VERSION, &device, &level, &context));
    constexpr char source[] = "[numthreads(1,1,1)] void main() {}";
    ComPtr<ID3DBlob> bytecode, errors;
    ASSERT_HRESULT_SUCCEEDED(D3DCompile(source, sizeof(source) - 1u, nullptr, nullptr, nullptr, "main", "cs_5_0", 0u, 0u, &bytecode, &errors));
    ComPtr<ID3D11ComputeShader> caller_shader, pass_shader;
    ASSERT_HRESULT_SUCCEEDED(device->CreateComputeShader(bytecode->GetBufferPointer(), bytecode->GetBufferSize(), nullptr, &caller_shader));
    ASSERT_HRESULT_SUCCEEDED(device->CreateComputeShader(bytecode->GetBufferPointer(), bytecode->GetBufferSize(), nullptr, &pass_shader));
    D3D11_QUERY_DESC query {D3D11_QUERY_OCCLUSION_PREDICATE, 0u};
    ComPtr<ID3D11Predicate> predicate;
    ASSERT_HRESULT_SUCCEEDED(device->CreatePredicate(&query, &predicate));

    // The caller holds one sentinel past every guarded range; the pass must never see it change.
    const auto caller_srvs = structured(device.Get(), srvs + 1u);
    const auto caller_uavs = structured(device.Get(), uavs + 1u);
    const auto caller_cbs = constants(device.Get(), cbs + 1u);
    const auto pass = structured(device.Get(), 2u);
    const auto pass_cbs = constants(device.Get(), 1u);
    context->CSSetShaderResources(0u, srvs + 1u, raw(caller_srvs.srvs).data());
    context->CSSetUnorderedAccessViews(0u, uavs + 1u, raw(caller_uavs.uavs).data(), nullptr);
    context->CSSetConstantBuffers(0u, cbs + 1u, raw(caller_cbs).data());
    context->CSSetShader(caller_shader.Get(), nullptr, 0u);
    context->SetPredication(predicate.Get(), TRUE);
    {
      const Guard saved(context.Get());
      ComPtr<ID3D11Predicate> active;
      BOOL value = TRUE;
      context->GetPredication(&active, &value);
      EXPECT_EQ(active.Get(), nullptr) << "The pass must not run under the caller's predicate";
      // The pass clobbers every guarded slot and still holds the caller's SRV 0 resource as a
      // UAV when the guard restores, which would refuse that SRV without the unbind first.
      const std::vector<ID3D11ShaderResourceView *> pass_srvs(srvs, pass.srvs[0].Get());
      std::vector<ID3D11UnorderedAccessView *> pass_uavs(uavs, pass.uavs[1].Get());
      pass_uavs[0] = caller_srvs.uavs[0].Get();
      const std::vector<ID3D11Buffer *> pass_constants(cbs, pass_cbs[0].Get());
      context->CSSetShaderResources(0u, srvs, pass_srvs.data());
      context->CSSetUnorderedAccessViews(0u, uavs, pass_uavs.data(), nullptr);
      context->CSSetConstantBuffers(0u, cbs, pass_constants.data());
      context->CSSetShader(pass_shader.Get(), nullptr, 0u);
    }
    std::vector<ID3D11ShaderResourceView *> bound_srvs(srvs + 1u);
    std::vector<ID3D11UnorderedAccessView *> bound_uavs(uavs + 1u);
    std::vector<ID3D11Buffer *> bound_cbs(cbs + 1u);
    context->CSGetShaderResources(0u, srvs + 1u, bound_srvs.data());
    context->CSGetUnorderedAccessViews(0u, uavs + 1u, bound_uavs.data());
    context->CSGetConstantBuffers(0u, cbs + 1u, bound_cbs.data());
    for (UINT i = 0u; i <= srvs; ++i) {
      EXPECT_EQ(bound_srvs[i], caller_srvs.srvs[i].Get()) << "SRV " << i;
      if (bound_srvs[i]) {
        bound_srvs[i]->Release();
      }
    }
    for (UINT i = 0u; i <= uavs; ++i) {
      EXPECT_EQ(bound_uavs[i], caller_uavs.uavs[i].Get()) << "UAV " << i;
      if (bound_uavs[i]) {
        bound_uavs[i]->Release();
      }
    }
    for (UINT i = 0u; i <= cbs; ++i) {
      EXPECT_EQ(bound_cbs[i], caller_cbs[i].Get()) << "constant buffer " << i;
      if (bound_cbs[i]) {
        bound_cbs[i]->Release();
      }
    }
    ComPtr<ID3D11ComputeShader> shader;
    UINT instances = 0u;
    context->CSGetShader(&shader, nullptr, &instances);
    EXPECT_EQ(shader.Get(), caller_shader.Get());
    ComPtr<ID3D11Predicate> restored;
    BOOL value = FALSE;
    context->GetPredication(&restored, &value);
    EXPECT_EQ(restored.Get(), predicate.Get());
    EXPECT_TRUE(value);

    // Unbinding clears only the guarded range.
    context->SetPredication(nullptr, FALSE);
    Guard::unbind(context.Get());
    context->CSGetShaderResources(0u, srvs + 1u, bound_srvs.data());
    context->CSGetUnorderedAccessViews(0u, uavs + 1u, bound_uavs.data());
    context->CSGetConstantBuffers(0u, cbs + 1u, bound_cbs.data());
    for (UINT i = 0u; i <= srvs; ++i) {
      EXPECT_EQ(bound_srvs[i], i == srvs ? caller_srvs.srvs[i].Get() : nullptr) << "SRV " << i;
      if (bound_srvs[i]) {
        bound_srvs[i]->Release();
      }
    }
    for (UINT i = 0u; i <= uavs; ++i) {
      EXPECT_EQ(bound_uavs[i], i == uavs ? caller_uavs.uavs[i].Get() : nullptr) << "UAV " << i;
      if (bound_uavs[i]) {
        bound_uavs[i]->Release();
      }
    }
    for (UINT i = 0u; i <= cbs; ++i) {
      EXPECT_EQ(bound_cbs[i], i == cbs ? caller_cbs[i].Get() : nullptr) << "constant buffer " << i;
      if (bound_cbs[i]) {
        bound_cbs[i]->Release();
      }
    }
  }

  TEST(HostSbsComputeStateGuardGpuTest, DirectGlyphPaletteRestoresExactlyItsSlots) {
    expect_exact_restore<models::host_sbs_direct_glyph_gpu::detail::palette_state_t>();
  }

  TEST(HostSbsComputeStateGuardGpuTest, DirectGlyphPublicationRestoresExactlyItsSlots) {
    expect_exact_restore<models::host_sbs_direct_glyph_publication::compute_state_t>();
  }

  TEST(HostSbsComputeStateGuardGpuTest, DisplaySnapshotRgb8RestoresExactlyItsSlots) {
    expect_exact_restore<models::display_snapshot_rgb8_analysis_gpu_v1::compute_state_t>();
  }
}  // namespace
#endif
