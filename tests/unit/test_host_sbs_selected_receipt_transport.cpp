#include <gtest/gtest.h>

#ifdef _WIN32
  #include "src/host_sbs_current_outline_selected_matte_gpu.h"
  #include "src/host_sbs_selected_glyph_receipt_c24_transport.h"
  #include "src/host_sbs_selected_matte_gpu.h"
  #include "src/sbs_bench_current_outline_qualification.h"

  #include <array>
  #include <cstring>
  #include <stdexcept>

namespace {
  namespace transport = models::selected_glyph_receipt_c24_transport_v1;
  using Microsoft::WRL::ComPtr;

  class HostSbsSelectedReceiptTransportTest: public testing::Test {
  protected:
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11Buffer> receipt;
    ComPtr<ID3D11ShaderResourceView> whole;
    transport::resources_t owner;
    std::array<std::uint32_t, 64u> words {};

    void SetUp() override {
      D3D_FEATURE_LEVEL level {};
      ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0u, nullptr, 0u, D3D11_SDK_VERSION, &device, &level, &context)));
      for (std::uint32_t i = 0u; i < words.size(); ++i) {
        words[i] = 0xc2400000u + i;
      }
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = sizeof(words);
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      desc.StructureByteStride = sizeof(std::uint32_t);
      const D3D11_SUBRESOURCE_DATA data {words.data(), 0u, 0u};
      ASSERT_TRUE(SUCCEEDED(device->CreateBuffer(&desc, &data, &receipt)));
      ASSERT_TRUE(SUCCEEDED(device->CreateShaderResourceView(receipt.Get(), nullptr, &whole)));
      std::string error;
      ASSERT_TRUE(owner.ensure(device.Get(), error)) << error;
    }

    void queue() {
      std::string error;
      ASSERT_TRUE(owner.queue(context.Get(), whole.Get(), error)) << error;
      ASSERT_EQ(owner.source_receipt(), whole.Get());
      ASSERT_NE(owner.copy_serial(), 0u);
    }

    std::array<std::uint32_t, 24u> read(ID3D11Buffer *source) {
      std::array<std::uint32_t, 24u> result {};
      if (!source) {
        ADD_FAILURE() << "C24 copy was not issued";
        return result;
      }
      D3D11_BUFFER_DESC desc {};
      source->GetDesc(&desc);
      EXPECT_EQ(desc.ByteWidth, 96u);
      EXPECT_EQ(desc.BindFlags, D3D11_BIND_CONSTANT_BUFFER);
      desc.Usage = D3D11_USAGE_STAGING;
      desc.BindFlags = desc.MiscFlags = desc.StructureByteStride = 0u;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Buffer> staging;
      if (FAILED(device->CreateBuffer(&desc, nullptr, &staging))) {
        ADD_FAILURE() << "C24 staging allocation failed";
        return result;
      }
      context->CopyResource(staging.Get(), source);
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) {
        ADD_FAILURE() << "C24 staging read failed";
        return result;
      }
      std::memcpy(result.data(), mapped.pData, sizeof(result));
      context->Unmap(staging.Get(), 0u);
      return result;
    }

    void expect_unavailable() {
      EXPECT_EQ(owner.constants(), nullptr);
      EXPECT_EQ(owner.source_receipt(), nullptr);
      EXPECT_EQ(owner.copy_serial(), 0u);
    }
  };

  TEST_F(HostSbsSelectedReceiptTransportTest, CopiesOnlyWholeReceiptWordsFourThroughTwentySeven) {
    queue();
    const auto copied = read(owner.constants());
    // Deliberately independent of the shared layout constants: header, SCT and token sentinel
    // words must not enter the C24 buffer consumed by the actual background GPU join.
    for (std::uint32_t i = 0u; i < copied.size(); ++i) {
      EXPECT_EQ(copied[i], 0xc2400004u + i) << i;
    }
  }

  TEST_F(HostSbsSelectedReceiptTransportTest, BothSelectorOwnersAllocateTheWholeReceiptRole) {
    const auto check = [&](const auto &views) {
      ASSERT_NE(views.receipt.Get(), nullptr);
      D3D11_SHADER_RESOURCE_VIEW_DESC view {};
      views.receipt->GetDesc(&view);
      EXPECT_EQ(view.ViewDimension, D3D11_SRV_DIMENSION_BUFFER);
      EXPECT_EQ(view.Format, DXGI_FORMAT_UNKNOWN);
      EXPECT_EQ(view.Buffer.FirstElement, 0u);
      EXPECT_EQ(view.Buffer.NumElements, 64u);
      ComPtr<ID3D11Resource> resource;
      ComPtr<ID3D11Buffer> buffer;
      views.receipt->GetResource(&resource);
      ASSERT_TRUE(SUCCEEDED(resource.As(&buffer)));
      D3D11_BUFFER_DESC desc {};
      buffer->GetDesc(&desc);
      EXPECT_EQ(desc.ByteWidth, 256u);
      EXPECT_EQ(desc.StructureByteStride, 4u);
      EXPECT_NE(desc.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, 0u);
    };
    models::research_current_outline_selected_matte_v2::resources_t current;
    models::research_selected_matte_v1::resources_t legacy;
    std::string error;
    ASSERT_TRUE(current.ensure(device.Get(), 8u, 4u, error)) << error;
    ASSERT_TRUE(legacy.ensure(device.Get(), 8u, 4u, error)) << error;
    check(current.views());
    check(legacy.views());
  }

  TEST_F(HostSbsSelectedReceiptTransportTest, RejectsSubrangeAndOversizedReceiptWithoutExposingOldCopy) {
    queue();
    D3D11_SHADER_RESOURCE_VIEW_DESC view {};
    whole->GetDesc(&view);
    view.Buffer.FirstElement = 1u;
    view.Buffer.NumElements = 63u;
    ComPtr<ID3D11ShaderResourceView> subrange;
    ASSERT_TRUE(SUCCEEDED(device->CreateShaderResourceView(receipt.Get(), &view, &subrange)));
    std::string error;
    EXPECT_FALSE(owner.queue(context.Get(), subrange.Get(), error));
    expect_unavailable();

    queue();
    D3D11_BUFFER_DESC desc {};
    receipt->GetDesc(&desc);
    desc.ByteWidth = 65u * 4u;
    ComPtr<ID3D11Buffer> oversized;
    ASSERT_TRUE(SUCCEEDED(device->CreateBuffer(&desc, nullptr, &oversized)));
    view.Buffer.FirstElement = 0u;
    view.Buffer.NumElements = 64u;
    ComPtr<ID3D11ShaderResourceView> prefix;
    ASSERT_TRUE(SUCCEEDED(device->CreateShaderResourceView(oversized.Get(), &view, &prefix)));
    EXPECT_FALSE(owner.queue(context.Get(), prefix.Get(), error));
    expect_unavailable();
  }

  TEST_F(HostSbsSelectedReceiptTransportTest, ReusedStorageChangesOnlyAfterExplicitCopyAndEnsureRevokesAvailability) {
    queue();
    const auto serial = owner.copy_serial();
    ComPtr<ID3D11Buffer> previous = owner.constants();
    const auto before = read(previous.Get());
    for (auto &word : words) {
      word ^= 0x00ff0000u;
    }
    context->UpdateSubresource(receipt.Get(), 0u, nullptr, words.data(), 0u, 0u);
    EXPECT_EQ(read(previous.Get()), before) << "Source-pointer identity does not copy or certify changed contents";
    std::string error;
    ASSERT_TRUE(owner.ensure(device.Get(), error)) << error;
    expect_unavailable();
    queue();
    EXPECT_GT(owner.copy_serial(), serial);
    const auto copied = read(owner.constants());
    for (std::uint32_t i = 0u; i < copied.size(); ++i) {
      EXPECT_EQ(copied[i], words[4u + i]) << i;
    }
  }

  TEST_F(HostSbsSelectedReceiptTransportTest, RejectsForeignDeferredAndPredicatedContexts) {
    queue();
    ComPtr<ID3D11Device> other_device;
    ComPtr<ID3D11DeviceContext> other_context;
    ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0u, nullptr, 0u, D3D11_SDK_VERSION, &other_device, nullptr, &other_context)));
    std::string error;
    EXPECT_FALSE(owner.queue(other_context.Get(), whole.Get(), error));
    expect_unavailable();

    queue();
    D3D11_BUFFER_DESC receipt_desc {};
    receipt->GetDesc(&receipt_desc);
    ComPtr<ID3D11Buffer> foreign_receipt;
    ComPtr<ID3D11ShaderResourceView> foreign_view;
    ASSERT_TRUE(SUCCEEDED(other_device->CreateBuffer(&receipt_desc, nullptr, &foreign_receipt)));
    ASSERT_TRUE(SUCCEEDED(other_device->CreateShaderResourceView(foreign_receipt.Get(), nullptr, &foreign_view)));
    EXPECT_FALSE(owner.queue(context.Get(), foreign_view.Get(), error));
    expect_unavailable();

    queue();
    ComPtr<ID3D11DeviceContext> deferred;
    ASSERT_TRUE(SUCCEEDED(device->CreateDeferredContext(0u, &deferred)));
    EXPECT_FALSE(owner.queue(deferred.Get(), whole.Get(), error));
    expect_unavailable();

    queue();
    const D3D11_QUERY_DESC desc {D3D11_QUERY_OCCLUSION_PREDICATE, 0u};
    ComPtr<ID3D11Predicate> predicate;
    ASSERT_TRUE(SUCCEEDED(device->CreatePredicate(&desc, &predicate)));
    context->SetPredication(predicate.Get(), FALSE);
    EXPECT_FALSE(owner.queue(context.Get(), whole.Get(), error));
    context->SetPredication(nullptr, FALSE);
    expect_unavailable();
  }

  TEST_F(HostSbsSelectedReceiptTransportTest, ProducerStageDescriptorsRequireExactWholeStructuredOrRawRoles) {
    namespace qualification = sbs_bench::current_outline_qualification;
    // These views contain no invented producer result. Exercise only actual D3D11
    // storage descriptors used by the diagnostic copy gate, independently of JSON.
    const auto make = [](ID3D11Device *owner, const UINT words, const bool raw, const UINT first, const UINT count, const bool typed_raw = false) {
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = words * 4u;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      desc.MiscFlags = raw ? D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS : D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      desc.StructureByteStride = raw ? 0u : 4u;
      ComPtr<ID3D11Buffer> buffer;
      if (FAILED(owner->CreateBuffer(&desc, nullptr, &buffer))) {
        throw std::runtime_error("Producer descriptor buffer creation failed");
      }
      D3D11_SHADER_RESOURCE_VIEW_DESC view {};
      view.ViewDimension = raw && !typed_raw ? D3D11_SRV_DIMENSION_BUFFEREX : D3D11_SRV_DIMENSION_BUFFER;
      view.Format = raw ? (typed_raw ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R32_TYPELESS) : DXGI_FORMAT_UNKNOWN;
      if (raw && !typed_raw) {
        view.BufferEx.FirstElement = first;
        view.BufferEx.NumElements = count;
        view.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
      } else {
        view.Buffer.FirstElement = first;
        view.Buffer.NumElements = count;
      }
      ComPtr<ID3D11ShaderResourceView> result;
      if (FAILED(owner->CreateShaderResourceView(buffer.Get(), &view, &result))) {
        throw std::runtime_error("Producer descriptor view creation failed");
      }
      return result;
    };
    ComPtr<ID3D11Device> foreign;
    ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0u, nullptr, 0u, D3D11_SDK_VERSION, &foreign, nullptr, nullptr)));
    ComPtr<ID3D11Buffer> actual;
    for (const auto &spec : qualification::producer_stage_specs) {
      SCOPED_TRACE(spec.name);
      const auto valid = make(device.Get(), spec.words, spec.raw, 0u, spec.words);
      ASSERT_TRUE(qualification::producer_stage_buffer(device.Get(), valid.Get(), spec, actual));
      ASSERT_NE(actual.Get(), nullptr);
      D3D11_BUFFER_DESC desc {};
      actual->GetDesc(&desc);
      EXPECT_EQ(desc.ByteWidth, spec.words * 4u);
      EXPECT_EQ(desc.StructureByteStride, spec.raw ? 0u : 4u);
      const UINT offset = spec.raw ? 4u : 1u;
      const std::array invalid {
        make(device.Get(), spec.words, spec.raw, offset, spec.words - offset),
        make(device.Get(), spec.words + (spec.raw ? 4u : 1u), spec.raw, 0u, spec.words),
        make(device.Get(), spec.words, !spec.raw, 0u, spec.words),
        make(foreign.Get(), spec.words, spec.raw, 0u, spec.words),
      };
      for (const auto &view : invalid) {
        ASSERT_TRUE(qualification::producer_stage_buffer(device.Get(), valid.Get(), spec, actual));
        EXPECT_FALSE(qualification::producer_stage_buffer(device.Get(), view.Get(), spec, actual));
        EXPECT_EQ(actual.Get(), nullptr) << "Rejected descriptor must not expose a preceding buffer";
      }
      const auto &wrong_role = spec.raw ? qualification::producer_stage_specs[0u] : qualification::producer_stage_specs[2u];
      EXPECT_FALSE(qualification::producer_stage_buffer(device.Get(), valid.Get(), wrong_role, actual));
      EXPECT_EQ(actual.Get(), nullptr);
      if (spec.raw) {
        const auto typed = make(device.Get(), spec.words, true, 0u, spec.words, true);
        EXPECT_FALSE(qualification::producer_stage_buffer(device.Get(), typed.Get(), spec, actual));
        EXPECT_EQ(actual.Get(), nullptr);
      }
      EXPECT_FALSE(qualification::producer_stage_buffer(device.Get(), nullptr, spec, actual));
      EXPECT_FALSE(qualification::producer_stage_buffer(nullptr, valid.Get(), spec, actual));
    }
  }
}  // namespace
#endif
