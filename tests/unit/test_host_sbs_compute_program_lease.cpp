#ifdef _WIN32
  #include "src/host_sbs_compute_program_lease.h"

  #include <d3dcompiler.h>
  #include <gtest/gtest.h>

namespace {
  using Microsoft::WRL::ComPtr;

  TEST(HostSbsComputeProgramLeaseTest, RetainsValidatedTupleAndRejectsMissingOrForeignReplacement) {
    ComPtr<ID3D11Device> first, second;
    ASSERT_HRESULT_SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0u,
      nullptr, 0u, D3D11_SDK_VERSION, &first, nullptr, nullptr));
    ASSERT_HRESULT_SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0u,
      nullptr, 0u, D3D11_SDK_VERSION, &second, nullptr, nullptr));
    constexpr char source[] = "[numthreads(1,1,1)] void main() {}";
    ComPtr<ID3DBlob> bytecode, errors;
    ASSERT_HRESULT_SUCCEEDED(D3DCompile(source, sizeof(source) - 1u, nullptr, nullptr, nullptr,
      "main", "cs_5_0", 0u, 0u, &bytecode, &errors));
    const auto shader = [&](ID3D11Device *device) {
      ComPtr<ID3D11ComputeShader> result;
      EXPECT_HRESULT_SUCCEEDED(device->CreateComputeShader(bytecode->GetBufferPointer(),
        bytecode->GetBufferSize(), nullptr, &result));
      return result;
    };
    auto a = shader(first.Get()), b = shader(first.Get()), replacement = shader(first.Get());
    auto foreign = shader(second.Get());
    models::host_sbs_compute_program_lease::group_t<2u> lease;
    const std::array original {a.Get(), b.Get()};
    ASSERT_TRUE(lease.acquire(first.Get(), original));
    a.Reset();b.Reset(); // The validated owner retains both shader objects.
    EXPECT_TRUE(lease.acquire(first.Get(), original));
    EXPECT_FALSE(lease.acquire(first.Get(), {original[0u], nullptr}));
    EXPECT_FALSE(lease.acquire(first.Get(), {original[0u], foreign.Get()}));
    EXPECT_FALSE(lease.acquire(second.Get(), original));
    EXPECT_TRUE(lease.acquire(first.Get(), original)); // Failed replacements preserve the old lease.
    EXPECT_TRUE(lease.acquire(first.Get(), {original[0u], replacement.Get()}));
    EXPECT_FALSE(lease.acquire(nullptr, {original[0u], replacement.Get()}));
  }
}
#endif
