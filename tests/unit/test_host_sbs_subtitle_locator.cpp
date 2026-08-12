/**
 * @file tests/unit/test_host_sbs_subtitle_locator.cpp
 * @brief Deterministic WARP coverage for the compact OCR6/SLR6 subtitle authority.
 */
#include <gtest/gtest.h>

#ifdef _WIN32

  #include <algorithm>
  #include <array>
  #include <bit>
  #include <cmath>
  #include <cstddef>
  #include <cstdint>
  #include <cstring>
  #include <d3d11.h>
  #include <d3dcompiler.h>
  #include <filesystem>
  #include <limits>
  #include <string>
  #include <vector>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;

  constexpr std::uint32_t field_width = 770u;
  constexpr std::uint32_t field_height = 434u;
  constexpr std::uint32_t roi_top = 325u;
  constexpr std::uint32_t roi_bottom = 430u;
  constexpr std::uint32_t ocr_words = 208u;
  constexpr std::uint32_t state_words = 80u;
  constexpr std::uint32_t ocr_schema = 1u;
  constexpr std::uint32_t ocr_tag = 0x3652434Fu;
  constexpr std::uint32_t slr_schema = 6u;
  constexpr std::uint32_t slr_tag = 0x36524C53u;
  constexpr std::uint32_t cut_tag = 0x28632D48u;
  constexpr std::uint32_t final_box_offset = 144u;

  constexpr std::uint32_t flag_owner = 1u;
  constexpr std::uint32_t flag_pending = 2u;
  constexpr std::uint32_t flag_target_valid = 4u;

  struct depth_constants_t {
    std::uint32_t target_w;
    std::uint32_t target_h;
    std::uint32_t color_mode;
    float ema_alpha;
    float minmax_alpha;
    std::uint32_t reduce_threads;
    float ema_edge_change;
    float ema_edge_gradient;
    float ema_edge_strength;
    std::array<float, 3u> reserved;
  };

  struct v2_constants_t {
    float raw_coordinate_scale;
    float collapse_abs_epsilon;
    float far_tau;
    float near_log_tau;
    float requested_gain;
    float max_horizontal_slope;
    float direct_container_limit;
    float convergence_curve_default;
  };

  struct subtitle_constants_t {
    std::array<std::uint32_t, 4u> field;
    std::array<std::uint32_t, 4u> source;
    std::array<std::uint32_t, 4u> frame;
  };

  struct line_box_t {
    std::uint32_t left;
    std::uint32_t top;
    std::uint32_t right;
    std::uint32_t bottom;
  };

  static_assert(sizeof(depth_constants_t) == 48u);
  static_assert(sizeof(v2_constants_t) == 32u);
  static_assert(sizeof(subtitle_constants_t) == 48u);

  bool compile_compute_shader(
    ID3D11Device *device,
    const std::filesystem::path &path,
    const char *entry,
    ComPtr<ID3D11ComputeShader> &shader,
    std::string &error
  ) {
    ComPtr<ID3DBlob> bytecode;
    ComPtr<ID3DBlob> diagnostics;
    const auto status = D3DCompileFromFile(
      path.c_str(),
      nullptr,
      D3D_COMPILE_STANDARD_FILE_INCLUDE,
      entry,
      "cs_5_0",
      D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
      0u,
      &bytecode,
      &diagnostics
    );
    if (FAILED(status)) {
      error = diagnostics ?
                std::string(
                  static_cast<const char *>(diagnostics->GetBufferPointer()),
                  diagnostics->GetBufferSize()
                ) :
                std::string("D3DCompileFromFile failed without diagnostics");
      return false;
    }
    const auto create_status = device->CreateComputeShader(
      bytecode->GetBufferPointer(), bytecode->GetBufferSize(), nullptr, &shader
    );
    if (FAILED(create_status)) {
      error = std::string("CreateComputeShader failed for ") + entry;
      return false;
    }
    return true;
  }

  bool create_structured_buffer(
    ID3D11Device *device,
    const void *initial,
    const std::size_t byte_count,
    const UINT stride,
    const UINT bind_flags,
    ComPtr<ID3D11Buffer> &buffer,
    ComPtr<ID3D11ShaderResourceView> *srv,
    ComPtr<ID3D11UnorderedAccessView> *uav
  ) {
    if (byte_count == 0u || byte_count > std::numeric_limits<UINT>::max()) return false;
    D3D11_BUFFER_DESC desc {};
    desc.ByteWidth = static_cast<UINT>(byte_count);
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = bind_flags;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = stride;
    D3D11_SUBRESOURCE_DATA data {initial, 0u, 0u};
    if (FAILED(device->CreateBuffer(&desc, initial ? &data : nullptr, &buffer))) return false;
    if (srv && FAILED(device->CreateShaderResourceView(
                 buffer.Get(), nullptr, srv->ReleaseAndGetAddressOf()))) return false;
    return !uav || SUCCEEDED(device->CreateUnorderedAccessView(
                     buffer.Get(), nullptr, uav->ReleaseAndGetAddressOf()));
  }

  bool create_constant_buffer(
    ID3D11Device *device,
    const void *initial,
    const UINT byte_count,
    ComPtr<ID3D11Buffer> &buffer
  ) {
    D3D11_BUFFER_DESC desc {};
    desc.ByteWidth = byte_count;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA data {initial, 0u, 0u};
    return SUCCEEDED(device->CreateBuffer(&desc, &data, &buffer));
  }

  template<typename T>
  bool read_buffer(
    ID3D11Device *device,
    ID3D11DeviceContext *context,
    ID3D11Buffer *source,
    std::vector<T> &output
  ) {
    D3D11_BUFFER_DESC source_desc {};
    source->GetDesc(&source_desc);
    D3D11_BUFFER_DESC staging_desc = source_desc;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0u;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    staging_desc.MiscFlags = 0u;
    staging_desc.StructureByteStride = 0u;
    ComPtr<ID3D11Buffer> staging;
    if (FAILED(device->CreateBuffer(&staging_desc, nullptr, &staging))) return false;
    context->CopyResource(staging.Get(), source);
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) return false;
    output.resize(source_desc.ByteWidth / sizeof(T));
    std::memcpy(output.data(), mapped.pData, source_desc.ByteWidth);
    context->Unmap(staging.Get(), 0u);
    return true;
  }

  bool read_texture(
    ID3D11Device *device,
    ID3D11DeviceContext *context,
    ID3D11Texture2D *source,
    std::vector<float> &output
  ) {
    D3D11_TEXTURE2D_DESC source_desc {};
    source->GetDesc(&source_desc);
    auto staging_desc = source_desc;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0u;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> staging;
    if (FAILED(device->CreateTexture2D(&staging_desc, nullptr, &staging))) return false;
    context->CopyResource(staging.Get(), source);
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) return false;
    output.resize(static_cast<std::size_t>(source_desc.Width) * source_desc.Height);
    for (std::uint32_t y = 0u; y < source_desc.Height; ++y) {
      const auto *row = reinterpret_cast<const float *>(
        static_cast<const std::byte *>(mapped.pData) +
        static_cast<std::size_t>(y) * mapped.RowPitch
      );
      std::copy_n(
        row,
        source_desc.Width,
        output.begin() + static_cast<std::size_t>(y) * source_desc.Width
      );
    }
    context->Unmap(staging.Get(), 0u);
    return true;
  }

  class slr6_warp_fixture_t {
   public:
    bool initialize(std::string &error) {
      constexpr D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_0};
      D3D_FEATURE_LEVEL actual {};
      if (FAILED(D3D11CreateDevice(
            nullptr,
            D3D_DRIVER_TYPE_WARP,
            nullptr,
            0u,
            requested,
            static_cast<UINT>(std::size(requested)),
            D3D11_SDK_VERSION,
            &device_,
            &actual,
            &context_
          )) || actual < D3D_FEATURE_LEVEL_11_0) return false;

      const auto shader_path = std::filesystem::path(SUNSHINE_SHADERS_DIR) /
                               "host_sbs_subtitle_locator_cs.hlsl";
      if (!compile_compute_shader(device_.Get(), shader_path, "resolve_main", resolve_, error) ||
          !compile_compute_shader(device_.Get(), shader_path, "condition_main", condition_, error)) {
        return false;
      }

      std::vector<std::uint32_t> zeros(state_words, 0u);
      if (!create_structured_buffer(
            device_.Get(),
            zeros.data(),
            zeros.size() * sizeof(std::uint32_t),
            sizeof(std::uint32_t),
            D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
            state_buffer_,
            &state_srv_,
            &state_uav_
          )) return false;

      zeros.assign(ocr_words, 0u);
      if (!create_structured_buffer(
            device_.Get(),
            zeros.data(),
            zeros.size() * sizeof(std::uint32_t),
            sizeof(std::uint32_t),
            D3D11_BIND_SHADER_RESOURCE,
            ocr_buffer_,
            &ocr_srv_,
            nullptr
          )) return false;

      std::array<std::uint32_t, 32u> cut_words {};
      cut_words[0u] = cut_tag;
      if (!create_structured_buffer(
            device_.Get(),
            cut_words.data(),
            cut_words.size() * sizeof(std::uint32_t),
            4u * sizeof(std::uint32_t),
            D3D11_BIND_SHADER_RESOURCE,
            cut_buffer_,
            &cut_srv_,
            nullptr
          )) return false;

      base_.assign(static_cast<std::size_t>(field_width) * field_height, 0.03f);
      for (std::uint32_t y = 330u; y <= 356u; ++y) {
        std::fill_n(base_.begin() + static_cast<std::size_t>(y) * field_width,
                    field_width, 0.01f);
      }
      D3D11_TEXTURE2D_DESC texture_desc {};
      texture_desc.Width = field_width;
      texture_desc.Height = field_height;
      texture_desc.MipLevels = 1u;
      texture_desc.ArraySize = 1u;
      texture_desc.Format = DXGI_FORMAT_R32_FLOAT;
      texture_desc.SampleDesc.Count = 1u;
      texture_desc.Usage = D3D11_USAGE_DEFAULT;
      texture_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA base_data {base_.data(), field_width * sizeof(float), 0u};
      if (FAILED(device_->CreateTexture2D(&texture_desc, &base_data, &base_texture_)) ||
          FAILED(device_->CreateShaderResourceView(base_texture_.Get(), nullptr, &base_srv_))) {
        return false;
      }
      texture_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
      if (FAILED(device_->CreateTexture2D(&texture_desc, nullptr, &output_texture_)) ||
          FAILED(device_->CreateUnorderedAccessView(output_texture_.Get(), nullptr, &output_uav_))) {
        return false;
      }

      const depth_constants_t depth_constants {
        field_width, field_height, 0u, 0.0f, 0.0f, 0u, 0.0f, 0.0f, 0.0f, {}
      };
      const v2_constants_t v2_constants {
        0.04f, 0.0001f, 0.0f, 0.0f, 1.0f, 0.5f, 0.04f, 0.0f
      };
      const subtitle_constants_t subtitle_constants {
        {field_width, field_height, roi_top, roi_bottom},
        {1920u, 1080u, 1u, 0u},
        {0u, 0u, 0u, 0u},
      };
      return create_constant_buffer(
               device_.Get(), &depth_constants, sizeof(depth_constants), depth_cb_) &&
             create_constant_buffer(
               device_.Get(), &v2_constants, sizeof(v2_constants), v2_cb_) &&
             create_constant_buffer(
               device_.Get(), &subtitle_constants, sizeof(subtitle_constants), subtitle_cb_);
    }

    bool observe(
      const std::uint64_t identity,
      const std::vector<line_box_t> &boxes,
      const bool reset,
      const bool authoritative = true,
      const bool corrupt_frame_identity = false
    ) {
      std::array<std::uint32_t, 32u> cut_words {};
      cut_words[0u] = cut_tag;
      cut_words[16u] = scene_epoch_;
      cut_words[22u] = std::bit_cast<std::uint32_t>(cut_pulse_ ? 1.0f : 0.0f);
      context_->UpdateSubresource(
        cut_buffer_.Get(), 0u, nullptr, cut_words.data(), 0u, 0u
      );
      cut_pulse_ = false;

      std::vector<std::uint32_t> record(ocr_words, 0u);
      record[0u] = ocr_schema;
      record[1u] = ocr_tag;
      record[2u] = authoritative ? 1u : 0u;
      record[3u] = authoritative ? static_cast<std::uint32_t>(boxes.size()) : 0u;
      record[4u] = authoritative ? static_cast<std::uint32_t>(boxes.size()) : 0u;
      record[5u] = static_cast<std::uint32_t>(identity) + (corrupt_frame_identity ? 1u : 0u);
      record[6u] = static_cast<std::uint32_t>(identity >> 32u);
      record[7u] = static_cast<std::uint32_t>(identity);
      record[8u] = static_cast<std::uint32_t>(identity >> 32u);
      record[9u] = 1920u;
      record[10u] = 1080u;
      record[11u] = field_width;
      record[12u] = field_height;
      record[13u] = roi_top;
      record[14u] = roi_bottom;
      if (authoritative) {
        for (std::size_t slot = 0u; slot < boxes.size(); ++slot) {
          write_box(record, 16u + static_cast<std::uint32_t>(slot) * 8u, boxes[slot]);
          write_box(record, final_box_offset + static_cast<std::uint32_t>(slot) * 8u, boxes[slot]);
        }
      }
      context_->UpdateSubresource(ocr_buffer_.Get(), 0u, nullptr, record.data(), 0u, 0u);

      const subtitle_constants_t subtitle_constants {
        {field_width, field_height, roi_top, roi_bottom},
        {1920u, 1080u, 1u, reset ? 1u : 0u},
        {
          static_cast<std::uint32_t>(identity),
          static_cast<std::uint32_t>(identity >> 32u),
          static_cast<std::uint32_t>(identity),
          static_cast<std::uint32_t>(identity >> 32u),
        },
      };
      context_->UpdateSubresource(
        subtitle_cb_.Get(), 0u, nullptr, &subtitle_constants, 0u, 0u
      );
      ID3D11Buffer *constant_buffers[] = {depth_cb_.Get(), v2_cb_.Get(), subtitle_cb_.Get()};
      context_->CSSetConstantBuffers(0u, 3u, constant_buffers);

      context_->CSSetShader(resolve_.Get(), nullptr, 0u);
      std::array<ID3D11ShaderResourceView *, 8u> resolve_srvs {
        nullptr, cut_srv_.Get(), base_srv_.Get(), nullptr, nullptr, nullptr, nullptr, ocr_srv_.Get()
      };
      context_->CSSetShaderResources(0u, resolve_srvs.size(), resolve_srvs.data());
      context_->CSSetUnorderedAccessViews(2u, 1u, state_uav_.GetAddressOf(), nullptr);
      context_->Dispatch(1u, 1u, 1u);
      unbind();

      context_->CSSetShader(condition_.Get(), nullptr, 0u);
      std::array<ID3D11ShaderResourceView *, 4u> condition_srvs {
        nullptr, nullptr, base_srv_.Get(), state_srv_.Get()
      };
      context_->CSSetShaderResources(0u, condition_srvs.size(), condition_srvs.data());
      context_->CSSetUnorderedAccessViews(3u, 1u, output_uav_.GetAddressOf(), nullptr);
      context_->Dispatch((field_width + 15u) / 16u, (field_height + 15u) / 16u, 1u);
      unbind();

      return read_buffer(device_.Get(), context_.Get(), state_buffer_.Get(), state_) &&
             read_texture(device_.Get(), context_.Get(), output_texture_.Get(), output_);
    }

    void set_cut(const std::uint32_t scene_epoch, const bool pulse) {
      scene_epoch_ = scene_epoch;
      cut_pulse_ = pulse;
    }

    const std::vector<std::uint32_t> &state() const { return state_; }
    const std::vector<float> &base() const { return base_; }
    const std::vector<float> &output() const { return output_; }

    float output_at(const std::uint32_t x, const std::uint32_t y) const {
      return output_.at(static_cast<std::size_t>(y) * field_width + x);
    }

    bool output_is_exact_base() const {
      if (output_.size() != base_.size()) return false;
      for (std::size_t index = 0u; index < base_.size(); ++index) {
        if (std::bit_cast<std::uint32_t>(output_[index]) !=
            std::bit_cast<std::uint32_t>(base_[index])) return false;
      }
      return true;
    }

   private:
    static void write_box(
      std::vector<std::uint32_t> &record,
      const std::uint32_t offset,
      const line_box_t box
    ) {
      record[offset + 0u] = box.left;
      record[offset + 1u] = box.top;
      record[offset + 2u] = box.right;
      record[offset + 3u] = box.bottom;
      record[offset + 4u] = std::bit_cast<std::uint32_t>(0.9f);
    }

    void unbind() {
      std::array<ID3D11ShaderResourceView *, 8u> null_srvs {};
      std::array<ID3D11UnorderedAccessView *, 4u> null_uavs {};
      context_->CSSetShaderResources(0u, null_srvs.size(), null_srvs.data());
      context_->CSSetUnorderedAccessViews(0u, null_uavs.size(), null_uavs.data(), nullptr);
    }

    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<ID3D11ComputeShader> resolve_;
    ComPtr<ID3D11ComputeShader> condition_;
    ComPtr<ID3D11Buffer> state_buffer_;
    ComPtr<ID3D11ShaderResourceView> state_srv_;
    ComPtr<ID3D11UnorderedAccessView> state_uav_;
    ComPtr<ID3D11Buffer> ocr_buffer_;
    ComPtr<ID3D11ShaderResourceView> ocr_srv_;
    ComPtr<ID3D11Buffer> cut_buffer_;
    ComPtr<ID3D11ShaderResourceView> cut_srv_;
    ComPtr<ID3D11Texture2D> base_texture_;
    ComPtr<ID3D11ShaderResourceView> base_srv_;
    ComPtr<ID3D11Texture2D> output_texture_;
    ComPtr<ID3D11UnorderedAccessView> output_uav_;
    ComPtr<ID3D11Buffer> depth_cb_;
    ComPtr<ID3D11Buffer> v2_cb_;
    ComPtr<ID3D11Buffer> subtitle_cb_;
    std::vector<std::uint32_t> state_;
    std::vector<float> base_;
    std::vector<float> output_;
    std::uint32_t scene_epoch_ = 0u;
    bool cut_pulse_ = false;
  };

  TEST(HostSbsSubtitleSlr6GpuTest, ConfirmsExactFrameStacksAndConditionsOnlyCurrentLines) {
    slr6_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    const line_box_t first {180u, 360u, 590u, 370u};
    // Deliberately shorter and left-aligned: live captions commonly append a translated line
    // without recentering the first line.
    const line_box_t second {180u, 374u, 500u, 384u};

    // A single observation is pending and has no conditioning authority.
    ASSERT_TRUE(fixture.observe(1u, {first}, false));
    ASSERT_EQ(fixture.state().size(), state_words);
    EXPECT_EQ(fixture.state()[0u], slr_schema);
    EXPECT_EQ(fixture.state()[1u], slr_tag);
    EXPECT_EQ(fixture.state()[2u], flag_pending);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_EQ(fixture.state()[12u], 1u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());

    // Redispatching the same exact frame/domain identity cannot self-confirm.
    ASSERT_TRUE(fixture.observe(1u, {first}, false));
    EXPECT_EQ(fixture.state()[2u], flag_pending);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());

    // A distinct compatible observation births the owner at half fade.
    ASSERT_TRUE(fixture.observe(2u, {first}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[3u], 1u);
    EXPECT_EQ(fixture.state()[4u], 1u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 1u);
    EXPECT_EQ(fixture.state()[21u], 1u);
    EXPECT_EQ(fixture.state()[24u], 1u);
    EXPECT_LT(fixture.output_at(300u, 364u), 0.025f);
    EXPECT_EQ(
      std::bit_cast<std::uint32_t>(fixture.output_at(10u, 100u)),
      std::bit_cast<std::uint32_t>(0.03f)
    );

    // The next observation reaches full strength.
    ASSERT_TRUE(fixture.observe(3u, {first}, false));
    EXPECT_EQ(fixture.state()[24u], 2u);
    EXPECT_NEAR(fixture.output_at(300u, 364u), 0.0102604f, 0.00002f);

    // A newly appended translation line is pending.  Only the matched old line is current;
    // the new line receives at most the old line's analytic collar, never a fabricated core.
    ASSERT_TRUE(fixture.observe(4u, {first, second}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_pending | flag_target_valid);
    EXPECT_EQ(fixture.state()[4u], 1u);
    EXPECT_EQ(fixture.state()[12u], 2u);
    EXPECT_EQ(fixture.state()[20u], 1u);
    EXPECT_GT(fixture.output_at(300u, 377u), 0.020f);

    // The second distinct stack observation confirms a two-line handoff.  Both dense line cores
    // are active, while their four-row gap remains a distance collar rather than a filled box.
    ASSERT_TRUE(fixture.observe(5u, {first, second}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[3u], 2u);
    EXPECT_EQ(fixture.state()[4u], 2u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 2u);
    EXPECT_EQ(fixture.state()[21u], 3u);
    EXPECT_EQ(fixture.state()[24u], 1u);
    const float first_core = fixture.output_at(300u, 364u);
    const float gap = fixture.output_at(300u, 372u);
    const float second_core = fixture.output_at(300u, 377u);
    EXPECT_LT(first_core, gap);
    EXPECT_LT(second_core, gap);
    EXPECT_LT(gap, 0.03f);

    // Authoritative empty OCR removes geometry immediately.  The cached target/grace is not
    // geometry authority, so every texel is exact Base on the death frame and thereafter.
    ASSERT_TRUE(fixture.observe(6u, {}, false));
    EXPECT_EQ(fixture.state()[2u], 0u);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_EQ(fixture.state()[21u], 2u);
    EXPECT_EQ(fixture.state()[25u], 6u);
    EXPECT_TRUE(fixture.output_is_exact_base());
    ASSERT_TRUE(fixture.observe(7u, {}, false));
    EXPECT_EQ(fixture.state()[25u], 5u);
    EXPECT_TRUE(fixture.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr6GpuTest, ResetLogoAndInvalidRecordFailToExactBase) {
    slr6_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    // A square bottom-corner badge is rejected by generic line geometry; reset state publishes
    // no pending stack and the conditioner is bit-exact Base.
    const line_box_t square_badge {700u, 350u, 740u, 390u};
    ASSERT_TRUE(fixture.observe(10u, {square_badge}, true));
    EXPECT_EQ(fixture.state()[0u], slr_schema);
    EXPECT_EQ(fixture.state()[1u], slr_tag);
    EXPECT_EQ(fixture.state()[2u], 0u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());

    // Exact-frame identity is mandatory.  A structurally valid but stale/mismatched record resets
    // to a canonical current state envelope and cannot retain pending or owner authority.
    const line_box_t line {180u, 360u, 590u, 370u};
    ASSERT_TRUE(fixture.observe(11u, {line}, false));
    EXPECT_EQ(fixture.state()[2u], flag_pending);
    ASSERT_TRUE(fixture.observe(12u, {line}, false, true, true));
    EXPECT_EQ(fixture.state()[0u], slr_schema);
    EXPECT_EQ(fixture.state()[1u], slr_tag);
    EXPECT_EQ(fixture.state()[2u], 0u);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr6GpuTest, HardCutSurvivorResamplesButDisjointStackRestartsPending) {
    slr6_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    const line_box_t survivor {180u, 360u, 590u, 370u};
    const line_box_t disjoint {80u, 405u, 380u, 417u};
    ASSERT_TRUE(fixture.observe(30u, {survivor}, false));
    ASSERT_TRUE(fixture.observe(31u, {survivor}, false));
    ASSERT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);

    fixture.set_cut(1u, true);
    ASSERT_TRUE(fixture.observe(32u, {survivor}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[3u], 1u);
    EXPECT_EQ(fixture.state()[4u], 1u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 1u);
    EXPECT_EQ(fixture.state()[21u], 0u);
    EXPECT_EQ(fixture.state()[24u], 2u);
    EXPECT_EQ(fixture.state()[26u], 1u);

    fixture.set_cut(2u, true);
    ASSERT_TRUE(fixture.observe(33u, {disjoint}, false));
    EXPECT_EQ(fixture.state()[2u], flag_pending);
    EXPECT_EQ(fixture.state()[3u], 0u);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_EQ(fixture.state()[12u], 1u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_EQ(fixture.state()[21u], 2u);
    EXPECT_EQ(fixture.state()[26u], 2u);
    EXPECT_TRUE(fixture.output_is_exact_base());

    // The cut pulse is one observation only; the retained epoch lets the next distinct matching
    // observation confirm a new birth instead of being mistaken for another cut.
    ASSERT_TRUE(fixture.observe(34u, {disjoint}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[3u], 1u);
    EXPECT_EQ(fixture.state()[4u], 1u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 1u);
    EXPECT_EQ(fixture.state()[21u], 1u);
    EXPECT_EQ(fixture.state()[26u], 2u);
  }

  TEST(HostSbsSubtitleSlr6GpuTest, DomainResetRequiresPendingThenNextDistinctFrameBirth) {
    slr6_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    const line_box_t line {180u, 360u, 590u, 370u};
    ASSERT_TRUE(fixture.observe(40u, {line}, false));
    ASSERT_TRUE(fixture.observe(41u, {line}, false));
    ASSERT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);

    ASSERT_TRUE(fixture.observe(42u, {line}, true));
    EXPECT_EQ(fixture.state()[2u], flag_pending);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_EQ(fixture.state()[12u], 1u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_EQ(fixture.state()[21u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());

    // The same exact record cannot satisfy the second-observation transaction after reset.
    ASSERT_TRUE(fixture.observe(42u, {line}, false));
    EXPECT_EQ(fixture.state()[2u], flag_pending);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());

    ASSERT_TRUE(fixture.observe(43u, {line}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[3u], 1u);
    EXPECT_EQ(fixture.state()[4u], 1u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 1u);
    EXPECT_EQ(fixture.state()[21u], 1u);
    EXPECT_EQ(fixture.state()[24u], 1u);
  }
}  // namespace

#else

TEST(HostSbsSubtitleSlr6GpuTest, WindowsOnly) {
  GTEST_SKIP() << "D3D11 WARP is Windows-only";
}

#endif
