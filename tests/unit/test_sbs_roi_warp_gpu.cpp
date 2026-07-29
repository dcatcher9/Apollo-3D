/**
 * @file tests/unit/test_sbs_roi_warp_gpu.cpp
 * @brief WARP execution tests for the retained-transform Host SBS warp contract.
 */

#ifdef _WIN32

  #include <algorithm>
  #include <array>
  #include <bit>
  #include <cstdint>
  #include <cstring>
  #include <d3d11.h>
  #include <d3dcompiler.h>
  #include <filesystem>
  #include <gtest/gtest.h>
  #include <src/sbs_frame_roi_transform.h>
  #include <vector>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;

  constexpr std::uint32_t transform_flag_valid = 1u << 0u;
  constexpr std::uint32_t transform_flag_full_frame = 1u << 1u;
  constexpr std::uint32_t transform_flag_active_roi = 1u << 2u;

  struct alignas(16) transform_t {
    std::array<std::uint32_t, 4> header {};
    std::array<std::uint32_t, 4> identity {};
    std::array<std::uint32_t, 4> model {};
    std::array<std::uint32_t, 4> focus {};
    std::array<std::uint32_t, 4> crop {};
    std::array<std::uint32_t, 4> accepted {};
    std::array<std::uint32_t, 4> feather {};
    std::array<std::uint32_t, 4> lifecycle {};
  };

  struct alignas(16) warp_constants_t {
    float subject_stretch = 0.0f;
    float content_scale_x = 1.0f;
    float content_scale_y = 1.0f;
    float pop_strength = 1.0f;
    float literal_bestv2 = 1.0f;
    float adaptive_pop = 0.0f;
    float adaptive_pop_max = 2.0f;
    float preserve_previous_on_invalid = 0.0f;
  };

  using subject_state_t =
    std::array<std::array<float, 4>, 3>;

  static_assert(
    sizeof(transform_t) ==
      models::frame_roi_transform_vector_count *
        4u * sizeof(std::uint32_t)
  );
  static_assert(alignof(transform_t) == 16u);
  static_assert(sizeof(warp_constants_t) == 32u);
  static_assert(alignof(warp_constants_t) == 16u);

  std::uint32_t float_bits(const float value) {
    return std::bit_cast<std::uint32_t>(value);
  }

  std::array<std::uint32_t, 4> rect_bits(
    const std::array<float, 4> &value
  ) {
    return {
      float_bits(value[0]),
      float_bits(value[1]),
      float_bits(value[2]),
      float_bits(value[3]),
    };
  }

  transform_t canonical_transform(
    const std::uint32_t source_width,
    const std::uint32_t source_height,
    const std::uint32_t model_width,
    const std::uint32_t model_height
  ) {
    transform_t value;
    value.header = {
      models::frame_roi_transform_contract_version,
      transform_flag_valid | transform_flag_full_frame,
      1u,
      0u,
    };
    value.identity = {
      0u,
      0u,
      source_width,
      source_height,
    };
    value.model = {
      model_width,
      model_height,
      model_width * model_height,
      0u,
    };
    value.focus = rect_bits({0.0f, 0.0f, 1.0f, 1.0f});
    value.crop = value.focus;
    value.accepted = {
      0u,
      0u,
      model_width,
      model_height,
    };
    value.lifecycle = {1u, 0u, 0u, 1u};
    return value;
  }

  transform_t active_transform() {
    constexpr std::uint32_t source_width = 56u;
    constexpr std::uint32_t source_height = 28u;
    constexpr std::uint32_t model_width = 28u;
    constexpr std::uint32_t model_height = 28u;
    constexpr std::uint32_t accepted_side = 14u;

    transform_t value;
    value.header = {
      models::frame_roi_transform_contract_version,
      transform_flag_valid | transform_flag_active_roi,
      2u,
      0u,
    };
    value.identity = {
      3u,
      7u,
      source_width,
      source_height,
    };
    value.model = {
      model_width,
      model_height,
      accepted_side * accepted_side,
      0x12345678u,
    };
    value.focus = rect_bits({0.375f, 0.25f, 0.625f, 0.75f});
    value.crop = rect_bits({0.25f, 0.0f, 0.75f, 1.0f});
    value.accepted = {7u, 7u, 21u, 21u};
    value.feather = rect_bits({0.05f, 0.05f, 0.05f, 0.05f});
    value.lifecycle = {2u, 0u, 1u, 0u};
    return value;
  }

  subject_state_t uninitialized_subject() {
    return {};
  }

  subject_state_t initialized_subject() {
    subject_state_t value {};
    // {recenter delta, scene age, subject EMA, initialized}
    value[0] = {0.0f, 0.0f, 0.5f, 1.0f};
    // {stretch low, inverse range, baseline, adaptive-pop ratio}
    value[1] = {0.0f, 1.0f, 0.0f, 1.0f};
    // {zero-anchor shift pixels, anchor valid, cut flags, history}
    value[2] = {0.0f, 1.0f, 0.0f, 0.0f};
    return value;
  }

  std::uint32_t packed_depth_key(const std::uint32_t packed) {
    return packed >> 16u;
  }

  std::uint32_t packed_source_x(const std::uint32_t packed) {
    return packed & 0xffffu;
  }

  class SbsRoiWarpGpu: public testing::Test {
  protected:
    void SetUp() override {
      constexpr D3D_FEATURE_LEVEL levels[] {D3D_FEATURE_LEVEL_11_0};
      ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_WARP,
        nullptr,
        D3D11_CREATE_DEVICE_SINGLETHREADED,
        levels,
        static_cast<UINT>(std::size(levels)),
        D3D11_SDK_VERSION,
        &device_,
        &feature_level_,
        &context_
      )));

      const auto shader_path =
        std::filesystem::path(SUNSHINE_SHADERS_DIR) /
        "sbs_forward_coverage_cs.hlsl";
      ComPtr<ID3DBlob> bytecode;
      ComPtr<ID3DBlob> diagnostics;
      const auto compile_status = D3DCompileFromFile(
        shader_path.c_str(),
        nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "main",
        "cs_5_0",
        D3DCOMPILE_ENABLE_STRICTNESS |
          D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0,
        &bytecode,
        &diagnostics
      );
      ASSERT_TRUE(SUCCEEDED(compile_status))
        << (diagnostics ?
              static_cast<const char *>(diagnostics->GetBufferPointer()) :
              "no compiler diagnostics");
      ASSERT_TRUE(SUCCEEDED(device_->CreateComputeShader(
        bytecode->GetBufferPointer(),
        bytecode->GetBufferSize(),
        nullptr,
        &coverage_shader_
      )));

      D3D11_SAMPLER_DESC sampler_descriptor {};
      sampler_descriptor.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
      sampler_descriptor.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
      sampler_descriptor.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
      sampler_descriptor.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
      sampler_descriptor.MaxLOD = D3D11_FLOAT32_MAX;
      ASSERT_TRUE(SUCCEEDED(device_->CreateSamplerState(
        &sampler_descriptor,
        &linear_sampler_
      )));
    }

    ComPtr<ID3D11ShaderResourceView> make_color_view(
      const std::uint32_t width,
      const std::uint32_t height
    ) {
      const std::vector<std::array<float, 4>> pixels(
        static_cast<std::size_t>(width) * height,
        {0.25f, 0.5f, 0.75f, 1.0f}
      );
      D3D11_TEXTURE2D_DESC descriptor {};
      descriptor.Width = width;
      descriptor.Height = height;
      descriptor.MipLevels = 1u;
      descriptor.ArraySize = 1u;
      descriptor.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
      descriptor.SampleDesc.Count = 1u;
      descriptor.Usage = D3D11_USAGE_IMMUTABLE;
      descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA initial {
        pixels.data(),
        width * static_cast<UINT>(sizeof(pixels.front())),
        0u,
      };
      ComPtr<ID3D11Texture2D> texture;
      if (FAILED(device_->CreateTexture2D(
            &descriptor,
            &initial,
            &texture
          ))) {
        ADD_FAILURE() << "Could not create color texture";
        return {};
      }
      ComPtr<ID3D11ShaderResourceView> view;
      if (FAILED(device_->CreateShaderResourceView(
            texture.Get(),
            nullptr,
            &view
          ))) {
        ADD_FAILURE() << "Could not create color SRV";
        return {};
      }
      return view;
    }

    ComPtr<ID3D11ShaderResourceView> make_depth_view(
      const std::uint32_t width,
      const std::uint32_t height,
      const std::vector<float> &pixels
    ) {
      if (pixels.size() !=
          static_cast<std::size_t>(width) * height) {
        ADD_FAILURE() << "Depth input has the wrong size";
        return {};
      }
      D3D11_TEXTURE2D_DESC descriptor {};
      descriptor.Width = width;
      descriptor.Height = height;
      descriptor.MipLevels = 1u;
      descriptor.ArraySize = 1u;
      descriptor.Format = DXGI_FORMAT_R32_FLOAT;
      descriptor.SampleDesc.Count = 1u;
      descriptor.Usage = D3D11_USAGE_IMMUTABLE;
      descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA initial {
        pixels.data(),
        width * static_cast<UINT>(sizeof(float)),
        0u,
      };
      ComPtr<ID3D11Texture2D> texture;
      if (FAILED(device_->CreateTexture2D(
            &descriptor,
            &initial,
            &texture
          ))) {
        ADD_FAILURE() << "Could not create depth texture";
        return {};
      }
      ComPtr<ID3D11ShaderResourceView> view;
      if (FAILED(device_->CreateShaderResourceView(
            texture.Get(),
            nullptr,
            &view
          ))) {
        ADD_FAILURE() << "Could not create depth SRV";
        return {};
      }
      return view;
    }

    template<typename T>
    ComPtr<ID3D11ShaderResourceView> make_structured_view(
      const T &value,
      const std::uint32_t vector_count
    ) {
      D3D11_BUFFER_DESC descriptor {};
      descriptor.Usage = D3D11_USAGE_IMMUTABLE;
      descriptor.ByteWidth = sizeof(value);
      descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      descriptor.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      descriptor.StructureByteStride = 4u * sizeof(std::uint32_t);
      D3D11_SUBRESOURCE_DATA initial {&value, 0u, 0u};
      ComPtr<ID3D11Buffer> buffer;
      if (FAILED(device_->CreateBuffer(
            &descriptor,
            &initial,
            &buffer
          ))) {
        ADD_FAILURE() << "Could not create structured buffer";
        return {};
      }
      D3D11_SHADER_RESOURCE_VIEW_DESC view_descriptor {};
      view_descriptor.Format = DXGI_FORMAT_UNKNOWN;
      view_descriptor.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
      view_descriptor.Buffer.FirstElement = 0u;
      view_descriptor.Buffer.NumElements = vector_count;
      ComPtr<ID3D11ShaderResourceView> view;
      if (FAILED(device_->CreateShaderResourceView(
            buffer.Get(),
            &view_descriptor,
            &view
          ))) {
        ADD_FAILURE() << "Could not create structured SRV";
        return {};
      }
      return view;
    }

    std::vector<std::uint32_t> dispatch(
      const std::uint32_t source_width,
      const std::uint32_t source_height,
      const std::uint32_t depth_width,
      const std::uint32_t depth_height,
      const std::vector<float> &depth,
      const transform_t &transform,
      const subject_state_t &subject
    ) {
      const auto color_view =
        make_color_view(source_width, source_height);
      const auto depth_view =
        make_depth_view(depth_width, depth_height, depth);
      const auto subject_view =
        make_structured_view(subject, subject.size());
      const auto transform_view = make_structured_view(
        transform,
        models::frame_roi_transform_vector_count
      );
      if (!color_view || !depth_view ||
          !subject_view || !transform_view) {
        return {};
      }

      D3D11_TEXTURE2D_DESC output_descriptor {};
      output_descriptor.Width = 2u * source_width;
      output_descriptor.Height = source_height;
      output_descriptor.MipLevels = 1u;
      output_descriptor.ArraySize = 1u;
      output_descriptor.Format = DXGI_FORMAT_R32_UINT;
      output_descriptor.SampleDesc.Count = 1u;
      output_descriptor.Usage = D3D11_USAGE_DEFAULT;
      output_descriptor.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
      ComPtr<ID3D11Texture2D> output;
      if (FAILED(device_->CreateTexture2D(
            &output_descriptor,
            nullptr,
            &output
          ))) {
        ADD_FAILURE() << "Could not create coverage texture";
        return {};
      }
      ComPtr<ID3D11UnorderedAccessView> output_view;
      if (FAILED(device_->CreateUnorderedAccessView(
            output.Get(),
            nullptr,
            &output_view
          ))) {
        ADD_FAILURE() << "Could not create coverage UAV";
        return {};
      }
      constexpr UINT clear[] {0u, 0u, 0u, 0u};
      context_->ClearUnorderedAccessViewUint(
        output_view.Get(),
        clear
      );

      const warp_constants_t constants;
      D3D11_BUFFER_DESC constants_descriptor {};
      constants_descriptor.Usage = D3D11_USAGE_IMMUTABLE;
      constants_descriptor.ByteWidth = sizeof(constants);
      constants_descriptor.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      D3D11_SUBRESOURCE_DATA constants_initial {&constants, 0u, 0u};
      ComPtr<ID3D11Buffer> constants_buffer;
      if (FAILED(device_->CreateBuffer(
            &constants_descriptor,
            &constants_initial,
            &constants_buffer
          ))) {
        ADD_FAILURE() << "Could not create warp constants";
        return {};
      }

      ID3D11ShaderResourceView *inputs[] {
        color_view.Get(),
        depth_view.Get(),
        subject_view.Get(),
        nullptr,
        nullptr,
        transform_view.Get(),
      };
      ID3D11UnorderedAccessView *outputs[] {output_view.Get()};
      ID3D11Buffer *constant_buffers[] {constants_buffer.Get()};
      ID3D11SamplerState *samplers[] {linear_sampler_.Get()};
      context_->CSSetShader(coverage_shader_.Get(), nullptr, 0u);
      context_->CSSetShaderResources(0u, 6u, inputs);
      context_->CSSetUnorderedAccessViews(
        0u,
        1u,
        outputs,
        nullptr
      );
      context_->CSSetConstantBuffers(2u, 1u, constant_buffers);
      context_->CSSetSamplers(0u, 1u, samplers);
      context_->Dispatch(
        (source_width + 15u) / 16u,
        (source_height + 15u) / 16u,
        1u
      );

      ID3D11ShaderResourceView *null_inputs[] {
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
      };
      ID3D11UnorderedAccessView *null_outputs[] {nullptr};
      ID3D11Buffer *null_constants[] {nullptr};
      ID3D11SamplerState *null_samplers[] {nullptr};
      context_->CSSetShaderResources(0u, 6u, null_inputs);
      context_->CSSetUnorderedAccessViews(
        0u,
        1u,
        null_outputs,
        nullptr
      );
      context_->CSSetConstantBuffers(2u, 1u, null_constants);
      context_->CSSetSamplers(0u, 1u, null_samplers);
      context_->CSSetShader(nullptr, nullptr, 0u);

      auto staging_descriptor = output_descriptor;
      staging_descriptor.Usage = D3D11_USAGE_STAGING;
      staging_descriptor.BindFlags = 0u;
      staging_descriptor.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Texture2D> staging;
      if (FAILED(device_->CreateTexture2D(
            &staging_descriptor,
            nullptr,
            &staging
          ))) {
        ADD_FAILURE() << "Could not create coverage staging texture";
        return {};
      }
      context_->CopyResource(staging.Get(), output.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context_->Map(
            staging.Get(),
            0u,
            D3D11_MAP_READ,
            0u,
            &mapped
          ))) {
        ADD_FAILURE() << "Could not map coverage staging texture";
        return {};
      }
      std::vector<std::uint32_t> result(
        static_cast<std::size_t>(output_descriptor.Width) *
          output_descriptor.Height
      );
      for (std::uint32_t y = 0u;
           y < output_descriptor.Height;
           ++y) {
        std::memcpy(
          result.data() +
            static_cast<std::size_t>(y) *
              output_descriptor.Width,
          static_cast<const std::byte *>(mapped.pData) +
            static_cast<std::size_t>(y) * mapped.RowPitch,
          output_descriptor.Width * sizeof(std::uint32_t)
        );
      }
      context_->Unmap(staging.Get(), 0u);
      return result;
    }

    static std::uint32_t at(
      const std::vector<std::uint32_t> &coverage,
      const std::uint32_t packed_width,
      const std::uint32_t x,
      const std::uint32_t y
    ) {
      return coverage[
        static_cast<std::size_t>(y) * packed_width + x
      ];
    }

    static void expect_identity_occupancy(
      const std::vector<std::uint32_t> &coverage
    ) {
      ASSERT_FALSE(coverage.empty());
      EXPECT_TRUE(std::all_of(
        coverage.begin(),
        coverage.end(),
        [](const std::uint32_t value) {
          return value == 1u;
        }
      ));
    }

    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    D3D_FEATURE_LEVEL feature_level_ {};
    ComPtr<ID3D11ComputeShader> coverage_shader_;
    ComPtr<ID3D11SamplerState> linear_sampler_;
  };

  TEST_F(
    SbsRoiWarpGpu,
    AllZeroAndExplicitCanonicalFullProduceIdenticalLegacyCoverage
  ) {
    constexpr std::uint32_t source_width = 56u;
    constexpr std::uint32_t source_height = 28u;
    constexpr std::uint32_t depth_width = 28u;
    constexpr std::uint32_t depth_height = 14u;
    std::vector<float> depth(depth_width * depth_height);
    for (std::uint32_t y = 0u; y < depth_height; ++y) {
      for (std::uint32_t x = 0u; x < depth_width; ++x) {
        depth[y * depth_width + x] =
          static_cast<float>(x) /
          static_cast<float>(depth_width - 1u);
      }
    }

    const auto zero = dispatch(
      source_width,
      source_height,
      depth_width,
      depth_height,
      depth,
      {},
      initialized_subject()
    );
    const auto explicit_full = dispatch(
      source_width,
      source_height,
      depth_width,
      depth_height,
      depth,
      canonical_transform(
        source_width,
        source_height,
        depth_width,
        depth_height
      ),
      initialized_subject()
    );

    ASSERT_FALSE(zero.empty());
    EXPECT_EQ(zero, explicit_full);
    EXPECT_TRUE(std::any_of(
      zero.begin(),
      zero.end(),
      [](const std::uint32_t value) {
        return value > 1u;
      }
    ));
  }

  TEST_F(
    SbsRoiWarpGpu,
    MissingTransformWithSquareOrPortraitDepthIsInert
  ) {
    constexpr std::uint32_t source_width = 56u;
    constexpr std::uint32_t source_height = 28u;
    for (const auto dimensions :
         {std::array<std::uint32_t, 2> {28u, 28u},
          std::array<std::uint32_t, 2> {14u, 28u}}) {
      const std::vector<float> depth(
        dimensions[0] * dimensions[1],
        1.0f
      );
      const auto coverage = dispatch(
        source_width,
        source_height,
        dimensions[0],
        dimensions[1],
        depth,
        {},
        initialized_subject()
      );
      expect_identity_occupancy(coverage);
    }
  }

  TEST_F(
    SbsRoiWarpGpu,
    ActiveCropMapsDepthAndLeavesFocusExteriorAsUnorderedIdentity
  ) {
    constexpr std::uint32_t source_width = 56u;
    constexpr std::uint32_t source_height = 28u;
    constexpr std::uint32_t depth_width = 28u;
    constexpr std::uint32_t depth_height = 28u;
    std::vector<float> depth(depth_width * depth_height, 0.1f);
    // The crop maps source x=34 to depth u~=0.73. A mistaken full-frame
    // mapping would sample u~=0.62 and remain on the low-depth side.
    for (std::uint32_t y = 0u; y < depth_height; ++y) {
      for (std::uint32_t x = 20u; x < depth_width; ++x) {
        depth[y * depth_width + x] = 0.9f;
      }
    }
    const auto coverage = dispatch(
      source_width,
      source_height,
      depth_width,
      depth_height,
      depth,
      active_transform(),
      uninitialized_subject()
    );
    ASSERT_EQ(coverage.size(), 2u * source_width * source_height);

    constexpr std::uint32_t row = 13u;
    const auto low = at(coverage, 2u * source_width, 22u, row);
    const auto mapped_high =
      at(coverage, 2u * source_width, 34u, row);
    EXPECT_EQ(packed_source_x(low), 22u);
    EXPECT_EQ(packed_source_x(mapped_high), 34u);
    EXPECT_LT(packed_depth_key(low), 20000u);
    EXPECT_GT(packed_depth_key(mapped_high), 50000u);

    // Exterior coverage is nonzero but carries no fabricated depth key
    // or source-ordering payload.
    EXPECT_EQ(at(coverage, 2u * source_width, 0u, row), 1u);
    EXPECT_EQ(
      at(
        coverage,
        2u * source_width,
        source_width,
        row
      ),
      1u
    );
    EXPECT_EQ(at(coverage, 2u * source_width, 35u, row), 1u);
    EXPECT_EQ(
      at(
        coverage,
        2u * source_width,
        source_width + 35u,
        row
      ),
      1u
    );
  }

  TEST_F(
    SbsRoiWarpGpu,
    ActiveFeatherSuppressesBoundaryShiftButPreservesInteriorShift
  ) {
    constexpr std::uint32_t source_width = 56u;
    constexpr std::uint32_t source_height = 28u;
    constexpr std::uint32_t depth_width = 28u;
    constexpr std::uint32_t depth_height = 28u;
    const std::vector<float> depth(
      depth_width * depth_height,
      1.0f
    );
    const auto coverage = dispatch(
      source_width,
      source_height,
      depth_width,
      depth_height,
      depth,
      active_transform(),
      initialized_subject()
    );
    ASSERT_EQ(coverage.size(), 2u * source_width * source_height);

    constexpr std::uint32_t row = 13u;
    constexpr std::uint32_t packed_width = 2u * source_width;
    // x=21 is the first source-pixel center inside the focus. Its
    // smoothstep feather is below the half-pixel rounding threshold.
    EXPECT_EQ(
      packed_source_x(at(coverage, packed_width, 21u, row)),
      21u
    );
    // The corresponding right-eye identity destination is legitimately
    // depth-occluded by a fully feathered interior source; it is not a
    // stable per-source oracle. The left destination has no such collision.

    // At the focus center the same depth reaches full feather strength
    // and the calibrated literal path shifts three output pixels.
    EXPECT_EQ(
      packed_source_x(at(coverage, packed_width, 30u, row)),
      27u
    );
    EXPECT_EQ(
      packed_source_x(at(
        coverage,
        packed_width,
        source_width + 24u,
        row
      )),
      27u
    );
  }

  TEST_F(
    SbsRoiWarpGpu,
    MalformedOrStaleSourceAndModelTransformsAreInert
  ) {
    constexpr std::uint32_t source_width = 56u;
    constexpr std::uint32_t source_height = 28u;
    constexpr std::uint32_t depth_width = 28u;
    constexpr std::uint32_t depth_height = 14u;
    const std::vector<float> depth(
      depth_width * depth_height,
      1.0f
    );

    auto malformed = canonical_transform(
      source_width,
      source_height,
      depth_width,
      depth_height
    );
    malformed.header[0] =
      models::frame_roi_transform_contract_version + 1u;

    // This record is internally valid for a 112x56 source, but stale
    // against the bound 56x28 color texture.
    const auto stale_source = canonical_transform(
      112u,
      56u,
      depth_width,
      depth_height
    );

    // This record is internally valid for 56x28 depth, but stale
    // against the bound 28x14 depth texture.
    const auto stale_model = canonical_transform(
      source_width,
      source_height,
      56u,
      28u
    );

    for (const auto &transform :
         {malformed, stale_source, stale_model}) {
      const auto coverage = dispatch(
        source_width,
        source_height,
        depth_width,
        depth_height,
        depth,
        transform,
        initialized_subject()
      );
      expect_identity_occupancy(coverage);
    }
  }
}  // namespace

#endif
