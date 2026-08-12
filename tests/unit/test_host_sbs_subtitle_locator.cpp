/**
 * @file tests/unit/test_host_sbs_subtitle_locator.cpp
 * @brief Deterministic WARP coverage for the compact OCR8/SLR8 lower-text authority.
 */
#include <gtest/gtest.h>

#ifdef _WIN32

  #include "src/generated/depth_coordinate_v2_contract.h"

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
  namespace v2 = models::depth_coordinate_v2;

  constexpr std::uint32_t field_width = v2::model_calibrated_shapes[0u].width;
  constexpr std::uint32_t field_height = v2::model_calibrated_shapes[0u].height;
  constexpr auto default_roi =
    v2::subtitle_ocr_dynamic_roi(1920u, 1080u, field_width, field_height);
  constexpr std::uint32_t roi_top = default_roi.top;
  constexpr std::uint32_t roi_bottom = default_roi.bottom;
  constexpr std::uint32_t ocr_words = v2::subtitle_ocr_record_word_count;
  constexpr std::uint32_t state_words = v2::subtitle_locator_state_word_count;
  constexpr std::uint32_t ocr_schema = v2::subtitle_ocr_record_schema;
  constexpr std::uint32_t ocr_tag = v2::subtitle_ocr_record_tag;
  constexpr std::uint32_t slr_schema = v2::subtitle_locator_state_schema;
  constexpr std::uint32_t slr_tag = v2::subtitle_locator_state_tag;
  constexpr std::uint32_t cut_tag = 0x28632D48u;
  constexpr std::uint32_t final_box_offset = 144u;

  constexpr std::uint32_t flag_owner = 1u;
  constexpr std::uint32_t flag_pending = 2u;
  constexpr std::uint32_t flag_target_valid = 4u;
  constexpr std::uint32_t box_flag_ribbon = 1u;
  constexpr std::uint32_t owner_kind_shift = 0u;
  constexpr std::uint32_t pending_kind_shift = 4u;
  constexpr std::uint32_t current_kind_shift = 8u;

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

  struct ocr_box_t {
    line_box_t core;
    line_box_t cover;
    std::uint32_t kind_flags;
    std::uint32_t island_count;
    std::uint32_t structural_gap_count;

    constexpr ocr_box_t(const line_box_t box) :
        core(box), cover(box), kind_flags(0u), island_count(1u), structural_gap_count(0u) {}

    constexpr ocr_box_t(
      const line_box_t core_box,
      const line_box_t cover_box,
      const std::uint32_t flags,
      const std::uint32_t islands,
      const std::uint32_t structural_gaps
    ) :
        core(core_box),
        cover(cover_box),
        kind_flags(flags),
        island_count(islands),
        structural_gap_count(structural_gaps) {}
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

  class slr8_warp_fixture_t {
   public:
    explicit slr8_warp_fixture_t(
      const std::uint32_t field_w = field_width,
      const std::uint32_t field_h = field_height,
      const std::uint32_t source_w = 1920u,
      const std::uint32_t source_h = 1080u,
      const std::uint32_t roi_top_override = std::numeric_limits<std::uint32_t>::max(),
      const std::uint32_t roi_bottom_override = std::numeric_limits<std::uint32_t>::max()
    ) :
        field_width_(field_w),
        field_height_(field_h),
        source_width_(source_w),
        source_height_(source_h) {
      const auto dynamic_roi =
        v2::subtitle_ocr_dynamic_roi(source_w, source_h, field_w, field_h);
      roi_top_ = roi_top_override == std::numeric_limits<std::uint32_t>::max() ?
                   dynamic_roi.top :
                   roi_top_override;
      roi_bottom_ = roi_bottom_override == std::numeric_limits<std::uint32_t>::max() ?
                      dynamic_roi.bottom :
                      roi_bottom_override;
    }

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

      base_.assign(static_cast<std::size_t>(field_width_) * field_height_, 0.03f);
      const auto sample_top = std::min(roi_top_ + 5u, field_height_);
      const auto sample_bottom = std::min(roi_top_ + 32u, field_height_);
      for (std::uint32_t y = sample_top; y < sample_bottom; ++y) {
        std::fill_n(base_.begin() + static_cast<std::size_t>(y) * field_width_,
                    field_width_, 0.01f);
      }
      D3D11_TEXTURE2D_DESC texture_desc {};
      texture_desc.Width = field_width_;
      texture_desc.Height = field_height_;
      texture_desc.MipLevels = 1u;
      texture_desc.ArraySize = 1u;
      texture_desc.Format = DXGI_FORMAT_R32_FLOAT;
      texture_desc.SampleDesc.Count = 1u;
      texture_desc.Usage = D3D11_USAGE_DEFAULT;
      texture_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA base_data {
        base_.data(), static_cast<UINT>(field_width_ * sizeof(float)), 0u
      };
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
        field_width_, field_height_, 0u, 0.0f, 0.0f, 0u, 0.0f, 0.0f, 0.0f, {}
      };
      const v2_constants_t v2_constants {
        0.04f, 0.0001f, 0.0f, 0.0f, 1.0f, 0.5f, 0.04f, 0.0f
      };
      const subtitle_constants_t subtitle_constants {
        {field_width_, field_height_, roi_top_, roi_bottom_},
        {source_width_, source_height_, 1u, 0u},
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
      const std::vector<ocr_box_t> &boxes,
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
      record[9u] = source_width_;
      record[10u] = source_height_;
      record[11u] = field_width_;
      record[12u] = field_height_;
      record[13u] = roi_top_;
      record[14u] = roi_bottom_;
      if (authoritative) {
        for (std::size_t slot = 0u; slot < boxes.size(); ++slot) {
          write_box(
            record,
            16u + static_cast<std::uint32_t>(slot) * 8u,
            boxes[slot].core,
            boxes[slot]
          );
          write_box(
            record,
            final_box_offset + static_cast<std::uint32_t>(slot) * 8u,
            boxes[slot].cover,
            boxes[slot]
          );
        }
      }
      context_->UpdateSubresource(ocr_buffer_.Get(), 0u, nullptr, record.data(), 0u, 0u);

      const subtitle_constants_t subtitle_constants {
        {field_width_, field_height_, roi_top_, roi_bottom_},
        {source_width_, source_height_, 1u, reset ? 1u : 0u},
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
      context_->Dispatch((field_width_ + 15u) / 16u, (field_height_ + 15u) / 16u, 1u);
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
      return output_.at(static_cast<std::size_t>(y) * field_width_ + x);
    }

    void set_base(const float value) {
      std::fill(base_.begin(), base_.end(), value);
      const auto sample_top = std::min(roi_top_ + 5u, field_height_);
      const auto sample_bottom = std::min(roi_top_ + 32u, field_height_);
      for (std::uint32_t y = sample_top; y < sample_bottom; ++y) {
        std::fill_n(base_.begin() + static_cast<std::size_t>(y) * field_width_,
                    field_width_, value);
      }
      context_->UpdateSubresource(
        base_texture_.Get(), 0u, nullptr, base_.data(),
        static_cast<UINT>(field_width_ * sizeof(float)), 0u
      );
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
      const line_box_t box,
      const ocr_box_t &metadata
    ) {
      record[offset + 0u] = box.left;
      record[offset + 1u] = box.top;
      record[offset + 2u] = box.right;
      record[offset + 3u] = box.bottom;
      record[offset + 4u] = std::bit_cast<std::uint32_t>(0.9f);
      record[offset + 5u] = metadata.kind_flags;
      record[offset + 6u] = metadata.island_count;
      record[offset + 7u] = metadata.structural_gap_count;
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
    std::uint32_t field_width_;
    std::uint32_t field_height_;
    std::uint32_t source_width_;
    std::uint32_t source_height_;
    std::uint32_t roi_top_;
    std::uint32_t roi_bottom_;
    std::uint32_t scene_epoch_ = 0u;
    bool cut_pulse_ = false;
  };

  TEST(HostSbsSubtitleSlr8GpuTest, ConfirmsExactFrameStacksAndConditionsOnlyCurrentLines) {
    slr8_warp_fixture_t fixture;
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

  TEST(HostSbsSubtitleSlr8GpuTest, SameBaselineSegmentsShareOwnerWithoutUnioningCovers) {
    slr8_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    // A detector gap just beyond the four-cell ordinary join becomes two OCR8 pairs. Their cores
    // remain strongly baseline-aligned and near enough to form one SLR8 owner, but each paired
    // cover stays independent and the horizontal gap is conditioned only by analytic collars.
    const ocr_box_t left {
      {180u, 360u, 380u, 372u},
      {176u, 356u, 384u, 376u},
      0u,
      1u,
      0u,
    };
    const ocr_box_t right {
      {420u, 360u, 600u, 372u},
      {416u, 356u, 604u, 376u},
      0u,
      1u,
      0u,
    };
    ASSERT_TRUE(fixture.observe(8u, {left, right}, false));
    ASSERT_EQ(fixture.state()[2u], flag_pending);
    ASSERT_EQ(fixture.state()[12u], 2u);
    ASSERT_TRUE(fixture.observe(9u, {left, right}, false));
    ASSERT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    ASSERT_EQ(fixture.state()[4u], 2u);
    ASSERT_EQ(fixture.state()[20u], 2u);
    EXPECT_EQ(fixture.state()[32u], left.core.left);
    EXPECT_EQ(fixture.state()[36u], right.core.left);
    EXPECT_EQ(fixture.state()[64u], left.cover.left);
    EXPECT_EQ(fixture.state()[66u], left.cover.right);
    EXPECT_EQ(fixture.state()[68u], right.cover.left);
    EXPECT_EQ(fixture.state()[70u], right.cover.right);
    const float left_core = fixture.output_at(250u, 365u);
    const float cover_gap = fixture.output_at(400u, 365u);
    const float right_core = fixture.output_at(500u, 365u);
    EXPECT_LT(left_core, cover_gap);
    EXPECT_LT(right_core, cover_gap);
    EXPECT_LT(cover_gap, 0.03f);

    // A nearby scene-text line with no strong vertical overlap cannot bridge into this baseline
    // component. The larger two-segment subtitle wins deterministic area selection by itself.
    const line_box_t vertically_offset_scene_text {200u, 330u, 500u, 340u};
    slr8_warp_fixture_t guarded;
    ASSERT_TRUE(guarded.initialize(error)) << error;
    ASSERT_TRUE(guarded.observe(
      10u, {left, right, vertically_offset_scene_text}, false));
    EXPECT_EQ(guarded.state()[2u], flag_pending);
    EXPECT_EQ(guarded.state()[12u], 2u);
    EXPECT_EQ(guarded.state()[48u], left.core.left);
    EXPECT_EQ(guarded.state()[52u], right.core.left);

    // Component closure remains fail-flat at compact-state capacity. It must not retain an
    // arbitrary four of five same-baseline segments.
    const line_box_t one {50u, 360u, 150u, 370u};
    const line_box_t two {170u, 360u, 270u, 370u};
    const line_box_t three {290u, 360u, 390u, 370u};
    const line_box_t four {410u, 360u, 510u, 370u};
    const line_box_t five {530u, 360u, 630u, 370u};
    slr8_warp_fixture_t overflow;
    ASSERT_TRUE(overflow.initialize(error)) << error;
    ASSERT_TRUE(overflow.observe(11u, {one, two, three, four, five}, false));
    EXPECT_EQ(overflow.state()[2u], 0u);
    EXPECT_EQ(overflow.state()[4u], 0u);
    EXPECT_EQ(overflow.state()[12u], 0u);
    EXPECT_EQ(overflow.state()[20u], 0u);
    EXPECT_TRUE(overflow.output_is_exact_base());

    // Pairwise-valid edges must not let transitive closure exceed the ordinary 0.9-field span.
    // A-B and B-C satisfy SameBaselineSegments, but A-C does not and the whole component is wider
    // than the authenticated ordinary-line maximum; reject it rather than selecting a bridge chain.
    const line_box_t chain_a {10u, 380u, 210u, 390u};
    const line_box_t chain_b {250u, 380u, 450u, 390u};
    const line_box_t chain_c {490u, 380u, 730u, 390u};
    slr8_warp_fixture_t bridge;
    ASSERT_TRUE(bridge.initialize(error)) << error;
    ASSERT_TRUE(bridge.observe(12u, {chain_a, chain_b, chain_c}, false));
    EXPECT_EQ(bridge.state()[2u], 0u);
    EXPECT_EQ(bridge.state()[12u], 0u);
    EXPECT_TRUE(bridge.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr8GpuTest, OwnerTargetLatchesUntilCutResetOrFullDeath) {
    slr8_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    const line_box_t first {180u, 360u, 590u, 370u};
    const line_box_t jittered {182u, 361u, 592u, 371u};
    const line_box_t handoff {80u, 360u, 380u, 370u};
    fixture.set_base(0.011f);
    ASSERT_TRUE(fixture.observe(100u, {first}, false));
    ASSERT_TRUE(fixture.observe(101u, {first}, false));
    ASSERT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    const std::uint32_t birth_target = fixture.state()[18u];
    EXPECT_EQ(birth_target, std::bit_cast<std::uint32_t>(0.011f));

    // Neither current Base changes nor compatible core jitter resample an established lifetime.
    fixture.set_base(0.025f);
    ASSERT_TRUE(fixture.observe(102u, {jittered}, false));
    EXPECT_EQ(fixture.state()[18u], birth_target);

    // An overlapping material handoff remains pending for one observation, then changes owner
    // generation. Both phases inherit the exact target bits instead of updating from BaseField.
    ASSERT_TRUE(fixture.observe(103u, {handoff}, false));
    EXPECT_EQ(fixture.state()[18u], birth_target);
    ASSERT_TRUE(fixture.observe(104u, {handoff}, false));
    ASSERT_EQ(fixture.state()[21u], 3u);
    EXPECT_EQ(fixture.state()[18u], birth_target);

    // Missing current evidence clears conditioning immediately but caches the latch through death
    // grace. A two-observation reacquisition during grace resumes the same lifetime bit-for-bit.
    ASSERT_TRUE(fixture.observe(105u, {}, false));
    ASSERT_EQ(fixture.state()[25u], 6u);
    EXPECT_EQ(fixture.state()[18u], birth_target);
    fixture.set_base(0.031f);
    ASSERT_TRUE(fixture.observe(106u, {handoff}, false));
    EXPECT_EQ(fixture.state()[18u], birth_target);
    ASSERT_TRUE(fixture.observe(107u, {handoff}, false));
    ASSERT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[18u], birth_target);

    // Once the owner dies and all six distinct grace observations expire, a future confirmed birth
    // samples a new plane from the then-current BaseField.
    ASSERT_TRUE(fixture.observe(108u, {}, false));
    for (std::uint64_t identity = 109u; identity <= 114u; ++identity) {
      ASSERT_TRUE(fixture.observe(identity, {}, false));
    }
    ASSERT_EQ(fixture.state()[25u], 0u);
    ASSERT_EQ(fixture.state()[18u], 0u);
    fixture.set_base(0.027f);
    ASSERT_TRUE(fixture.observe(115u, {handoff}, false));
    ASSERT_TRUE(fixture.observe(116u, {handoff}, false));
    const std::uint32_t post_death_target = fixture.state()[18u];
    EXPECT_EQ(post_death_target, std::bit_cast<std::uint32_t>(0.027f));
    EXPECT_NE(post_death_target, birth_target);

    // A hard-cut survivor is the other deliberate resampling boundary and becomes full-strength
    // immediately. The same current geometry now samples the new Base rather than inheriting.
    fixture.set_base(0.018f);
    fixture.set_cut(1u, true);
    ASSERT_TRUE(fixture.observe(117u, {handoff}, false));
    EXPECT_EQ(fixture.state()[18u], std::bit_cast<std::uint32_t>(0.018f));
    EXPECT_EQ(fixture.state()[24u], 2u);
    EXPECT_EQ(fixture.state()[3u], 2u);

    // An input-domain reset clears owner and latch. Present geometry starts pending, and its second
    // observation births a new owner sampled in the new source/transfer/field domain.
    fixture.set_base(0.024f);
    ASSERT_TRUE(fixture.observe(118u, {handoff}, true));
    EXPECT_EQ(fixture.state()[2u], flag_pending);
    EXPECT_EQ(fixture.state()[18u], 0u);
    ASSERT_TRUE(fixture.observe(119u, {handoff}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[18u], std::bit_cast<std::uint32_t>(0.024f));

    // Even a disjoint confirmed handoff remains in the same owner lifetime and carries the exact
    // plane latch. Geometry changes never sample BaseField by themselves.
    slr8_warp_fixture_t disjoint;
    ASSERT_TRUE(disjoint.initialize(error)) << error;
    disjoint.set_base(0.010f);
    ASSERT_TRUE(disjoint.observe(200u, {first}, false));
    ASSERT_TRUE(disjoint.observe(201u, {first}, false));
    const std::uint32_t old_disjoint_target = disjoint.state()[18u];
    disjoint.set_base(0.026f);
    const line_box_t unrelated {40u, 395u, 340u, 405u};
    ASSERT_TRUE(disjoint.observe(202u, {unrelated}, false));
    EXPECT_EQ(disjoint.state()[18u], old_disjoint_target);
    ASSERT_TRUE(disjoint.observe(203u, {unrelated}, false));
    EXPECT_EQ(disjoint.state()[21u], 3u);
    EXPECT_EQ(disjoint.state()[18u], old_disjoint_target);

    // The same unconditional lifetime rule applies across the miss/grace transaction. A disjoint
    // rebirth before grace expiry inherits the cached bits; only full grace expiry may resample.
    ASSERT_TRUE(disjoint.observe(204u, {}, false));
    ASSERT_EQ(disjoint.state()[25u], 6u);
    EXPECT_EQ(disjoint.state()[18u], old_disjoint_target);
    disjoint.set_base(0.032f);
    const line_box_t other_side {430u, 395u, 730u, 405u};
    ASSERT_TRUE(disjoint.observe(205u, {other_side}, false));
    EXPECT_EQ(disjoint.state()[18u], old_disjoint_target);
    ASSERT_TRUE(disjoint.observe(206u, {other_side}, false));
    EXPECT_EQ(disjoint.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(disjoint.state()[18u], old_disjoint_target);
  }

  TEST(HostSbsSubtitleSlr8GpuTest, TracksSubtitleAndBottomRibbonOnOnePlane) {
    slr8_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    const line_box_t subtitle_core {180u, 360u, 590u, 370u};
    const line_box_t subtitle_cover {176u, 356u, 594u, 374u};
    const line_box_t changed_core {80u, 360u, 380u, 370u};
    const line_box_t changed_cover {76u, 356u, 384u, 374u};
    const line_box_t ribbon_core {6u, 401u, 693u, roi_bottom};
    const line_box_t ribbon_cover {0u, 397u, field_width, field_height};
    const ocr_box_t subtitle {subtitle_core, subtitle_cover, 0u, 1u, 0u};
    const ocr_box_t changed {changed_core, changed_cover, 0u, 1u, 0u};
    const ocr_box_t ribbon {ribbon_core, ribbon_cover, box_flag_ribbon, 7u, 4u};

    // Tight/core geometry owns selection and temporal confirmation. The paired full-field ribbon
    // cover is not allowed to merge with the subtitle merely because its conditioning strip meets
    // the bottom of the field.
    ASSERT_TRUE(fixture.observe(20u, {subtitle, ribbon}, false));
    EXPECT_EQ(fixture.state()[2u], flag_pending);
    EXPECT_EQ(fixture.state()[12u], 2u);
    EXPECT_EQ(
      fixture.state()[31u],
      2u << pending_kind_shift
    );
    EXPECT_TRUE(fixture.output_is_exact_base());

    ASSERT_TRUE(fixture.observe(21u, {subtitle, ribbon}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[4u], 2u);
    EXPECT_EQ(fixture.state()[20u], 2u);
    EXPECT_EQ(
      fixture.state()[31u],
      (2u << owner_kind_shift) | (2u << current_kind_shift)
    );
    // Owner/pending rectangles are cores; only current same-frame rectangles are covers.
    EXPECT_EQ(fixture.state()[32u], subtitle_core.left);
    EXPECT_EQ(fixture.state()[33u], subtitle_core.top);
    EXPECT_EQ(fixture.state()[36u], ribbon_core.left);
    EXPECT_EQ(fixture.state()[37u], ribbon_core.top);
    EXPECT_EQ(fixture.state()[64u], subtitle_cover.left);
    EXPECT_EQ(fixture.state()[65u], subtitle_cover.top);
    EXPECT_EQ(fixture.state()[68u], ribbon_cover.left);
    EXPECT_EQ(fixture.state()[69u], ribbon_cover.top);

    // Both regions use exactly one sampled scalar plane. The ribbon's full cover includes its
    // gaps and reaches the field bottom; its only analytic collar is above the corrected top.
    EXPECT_NEAR(fixture.output_at(300u, 364u), fixture.output_at(300u, 410u), 0.000001f);
    EXPECT_LT(fixture.output_at(0u, 395u), 0.03f);
    EXPECT_LT(fixture.output_at(300u, field_height - 1u), 0.03f);
    EXPECT_LT(fixture.output_at(174u, 364u), 0.03f);  // Ordinary left collar remains active.
    // The analytic collar is slope-bounded rather than finite-radius, so it decays continuously
    // above the strip instead of creating a hard edge. A farther row must be closer to Base than
    // a row immediately above the corrected top.
    EXPECT_GT(fixture.output_at(0u, 380u), fixture.output_at(0u, 395u));
    EXPECT_LE(fixture.output_at(0u, 380u), 0.03f);

    // A material subtitle handoff leaves the constant ribbon authoritative while the new subtitle
    // waits for its second distinct observation.
    ASSERT_TRUE(fixture.observe(22u, {changed, ribbon}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_pending | flag_target_valid);
    EXPECT_EQ(fixture.state()[4u], 2u);
    EXPECT_EQ(fixture.state()[12u], 2u);
    EXPECT_EQ(fixture.state()[20u], 1u);
    EXPECT_EQ(
      fixture.state()[31u],
      (2u << owner_kind_shift) | (2u << pending_kind_shift) |
        (1u << current_kind_shift)
    );
    EXPECT_EQ(
      std::bit_cast<std::uint32_t>(fixture.output_at(200u, 364u)),
      std::bit_cast<std::uint32_t>(0.03f)
    );
    EXPECT_LT(fixture.output_at(300u, 410u), 0.03f);

    ASSERT_TRUE(fixture.observe(23u, {changed, ribbon}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[21u], 3u);
    EXPECT_EQ(fixture.state()[4u], 2u);
    EXPECT_EQ(fixture.state()[20u], 2u);
    EXPECT_EQ(
      fixture.state()[31u],
      (2u << owner_kind_shift) | (2u << current_kind_shift)
    );

    // A same-frame mixed owner survives a hard cut as two independently matched core rectangles,
    // resamples the one shared target, and keeps the current cover kinds.
    fixture.set_cut(1u, true);
    ASSERT_TRUE(fixture.observe(24u, {changed, ribbon}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[4u], 2u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 2u);
    EXPECT_EQ(fixture.state()[24u], 2u);
    EXPECT_EQ(fixture.state()[26u], 1u);
    EXPECT_EQ(
      fixture.state()[31u],
      (2u << owner_kind_shift) | (2u << current_kind_shift)
    );
  }

  TEST(HostSbsSubtitleSlr8GpuTest, ResetLogoAndInvalidRecordFailToExactBase) {
    slr8_warp_fixture_t fixture;
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

  TEST(HostSbsSubtitleSlr8GpuTest, MixedSelectionOverCompactCapacityFailsFlat) {
    slr8_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    const ocr_box_t ribbon {
      {6u, 401u, 693u, roi_bottom},
      {0u, 397u, field_width, field_height},
      box_flag_ribbon,
      7u,
      4u,
    };
    // Four coherent ordinary lines still fit the established compact owner by themselves. Adding
    // an independently authoritative ribbon would require five entities, so the observation must
    // abstain as a whole instead of silently dropping either kind.
    const line_box_t first {180u, 330u, 500u, 338u};
    const line_box_t second {180u, 340u, 500u, 348u};
    const line_box_t third {180u, 350u, 500u, 358u};
    const line_box_t fourth {180u, 360u, 500u, 368u};
    ASSERT_TRUE(fixture.observe(25u, {first, second, third, fourth, ribbon}, false));
    EXPECT_EQ(fixture.state()[2u], 0u);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_EQ(fixture.state()[31u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr8GpuTest, HardCutSurvivorResamplesButDisjointStackRestartsPending) {
    slr8_warp_fixture_t fixture;
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
    EXPECT_EQ(fixture.state()[3u], 2u);
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

  TEST(HostSbsSubtitleSlr8GpuTest, DomainResetRequiresPendingThenNextDistinctFrameBirth) {
    slr8_warp_fixture_t fixture;
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

  TEST(HostSbsSubtitleSlr8GpuTest, SupportsAuthenticatedWideAndPortraitFields) {
    struct field_case_t {
      std::uint32_t field_width;
      std::uint32_t field_height;
      std::uint32_t source_width;
      std::uint32_t source_height;
      std::uint32_t line_left;
      std::uint32_t line_right;
    };
    constexpr std::array cases {
      field_case_t {1022u, 434u, 2560u, 1080u, 100u, 850u},
      field_case_t {434u, 770u, 1080u, 1920u, 40u, 360u},
    };

    std::uint64_t identity = 100u;
    for (const auto &field_case : cases) {
      SCOPED_TRACE(
        std::to_string(field_case.field_width) + "x" +
        std::to_string(field_case.field_height)
      );
      const auto dynamic_roi = v2::subtitle_ocr_dynamic_roi(
        field_case.source_width,
        field_case.source_height,
        field_case.field_width,
        field_case.field_height
      );
      ASSERT_TRUE(static_cast<bool>(dynamic_roi));
      const line_box_t line {
        field_case.line_left,
        dynamic_roi.top + 35u,
        field_case.line_right,
        dynamic_roi.top + 45u,
      };
      ASSERT_LE(line.bottom, dynamic_roi.bottom);

      slr8_warp_fixture_t fixture(
        field_case.field_width,
        field_case.field_height,
        field_case.source_width,
        field_case.source_height
      );
      std::string error;
      ASSERT_TRUE(fixture.initialize(error)) << error;
      ASSERT_TRUE(fixture.observe(identity++, {line}, false));
      EXPECT_EQ(fixture.state()[2u], flag_pending);
      EXPECT_TRUE(fixture.output_is_exact_base());
      ASSERT_TRUE(fixture.observe(identity++, {line}, false));
      EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
      EXPECT_EQ(fixture.state()[4u], 1u);
      EXPECT_EQ(fixture.state()[20u], 1u);
      EXPECT_EQ(fixture.state()[27u], field_case.field_width);
      EXPECT_EQ(fixture.state()[28u], field_case.field_height);
      EXPECT_LT(
        fixture.output_at((line.left + line.right) / 2u, line.top + 4u),
        0.025f
      );
    }
  }

  TEST(HostSbsSubtitleSlr8GpuTest, RibbonBottomToleranceProjectsExactlyAcrossFields) {
    struct field_case_t {
      std::uint32_t field_width;
      std::uint32_t field_height;
      std::uint32_t source_width;
      std::uint32_t source_height;
    };
    constexpr std::array cases {
      field_case_t {770u, 434u, 1920u, 1080u},
      field_case_t {1022u, 434u, 2560u, 1080u},
      field_case_t {1036u, 434u, 3840u, 1600u},
      field_case_t {434u, 770u, 1080u, 1920u},
      field_case_t {434u, 1022u, 1080u, 2560u},
      field_case_t {434u, 1036u, 1600u, 3840u},
    };

    std::uint64_t identity = 300u;
    for (const auto &field_case : cases) {
      SCOPED_TRACE(
        std::to_string(field_case.field_width) + "x" +
        std::to_string(field_case.field_height)
      );
      const auto dynamic_roi = v2::subtitle_ocr_dynamic_roi(
        field_case.source_width,
        field_case.source_height,
        field_case.field_width,
        field_case.field_height
      );
      ASSERT_TRUE(static_cast<bool>(dynamic_roi));
      const auto minimum_bottom_result = v2::subtitle_ocr_ribbon_min_bottom(
        field_case.source_width,
        field_case.source_height,
        field_case.field_width,
        field_case.field_height
      );
      ASSERT_TRUE(static_cast<bool>(minimum_bottom_result));
      const auto minimum_bottom = minimum_bottom_result.value;
      ASSERT_GE(minimum_bottom, dynamic_roi.top + 7u);
      ASSERT_LE(minimum_bottom, dynamic_roi.bottom);
      const auto left = field_case.field_width / 20u;
      const auto right = field_case.field_width - left;
      const auto core_top = minimum_bottom - 6u;
      const auto cover_top = core_top > 4u ? core_top - 4u : 0u;
      const ocr_box_t boundary_ribbon {
        {left, core_top, right, minimum_bottom},
        {0u, cover_top, field_case.field_width, field_case.field_height},
        box_flag_ribbon,
        7u,
        4u,
      };

      slr8_warp_fixture_t accepted(
        field_case.field_width,
        field_case.field_height,
        field_case.source_width,
        field_case.source_height
      );
      std::string error;
      ASSERT_TRUE(accepted.initialize(error)) << error;
      ASSERT_TRUE(accepted.observe(identity++, {boundary_ribbon}, false));
      EXPECT_EQ(accepted.state()[2u], flag_pending);
      ASSERT_TRUE(accepted.observe(identity++, {boundary_ribbon}, false));
      EXPECT_EQ(accepted.state()[2u], flag_owner | flag_target_valid);
      EXPECT_EQ(accepted.state()[4u], 1u);
      EXPECT_EQ(accepted.state()[20u], 1u);
      EXPECT_EQ(
        accepted.state()[31u],
        (1u << owner_kind_shift) | (1u << current_kind_shift)
      );

    }

    // One field row below the exact projected detector tolerance has no OCR8 ribbon authority,
    // even if all other topology metadata and the canonical cover are well formed. One rejection
    // fixture is sufficient because the generated helper's six-field values were exercised above.
    constexpr auto rejection_case = cases[1u];
    const auto rejection_roi = v2::subtitle_ocr_dynamic_roi(
      rejection_case.source_width,
      rejection_case.source_height,
      rejection_case.field_width,
      rejection_case.field_height
    );
    const auto rejection_minimum = v2::subtitle_ocr_ribbon_min_bottom(
      rejection_case.source_width,
      rejection_case.source_height,
      rejection_case.field_width,
      rejection_case.field_height
    );
    ASSERT_TRUE(static_cast<bool>(rejection_roi));
    ASSERT_TRUE(static_cast<bool>(rejection_minimum));
    ASSERT_GT(rejection_minimum.value, rejection_roi.top + 6u);
    const auto rejection_left = rejection_case.field_width / 20u;
    const auto rejection_right = rejection_case.field_width - rejection_left;
    const ocr_box_t below_boundary {
      {
        rejection_left,
        rejection_minimum.value - 7u,
        rejection_right,
        rejection_minimum.value - 1u,
      },
      {
        0u,
        rejection_minimum.value - 11u,
        rejection_case.field_width,
        rejection_case.field_height,
      },
      box_flag_ribbon,
      7u,
      4u,
    };
    slr8_warp_fixture_t rejected(
      rejection_case.field_width,
      rejection_case.field_height,
      rejection_case.source_width,
      rejection_case.source_height
    );
    std::string error;
    ASSERT_TRUE(rejected.initialize(error)) << error;
    ASSERT_TRUE(rejected.observe(identity++, {below_boundary}, false));
    EXPECT_EQ(rejected.state()[2u], 0u);
    EXPECT_EQ(rejected.state()[4u], 0u);
    EXPECT_EQ(rejected.state()[12u], 0u);
    EXPECT_EQ(rejected.state()[20u], 0u);
    EXPECT_TRUE(rejected.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr8GpuTest, DynamicMaximumWidthAndMalformedRoiFailFlat) {
    constexpr std::uint32_t portrait_field_width = 434u;
    constexpr std::uint32_t portrait_field_height = 770u;
    constexpr std::uint32_t portrait_source_width = 1080u;
    constexpr std::uint32_t portrait_source_height = 1920u;
    constexpr auto dynamic_roi = v2::subtitle_ocr_dynamic_roi(
      portrait_source_width,
      portrait_source_height,
      portrait_field_width,
      portrait_field_height
    );
    static_assert(static_cast<bool>(dynamic_roi));

    // floor(0.9 * 434) == 390; this otherwise valid 400-cell horizontal line must not acquire.
    slr8_warp_fixture_t overwide_fixture(
      portrait_field_width,
      portrait_field_height,
      portrait_source_width,
      portrait_source_height
    );
    std::string error;
    ASSERT_TRUE(overwide_fixture.initialize(error)) << error;
    const line_box_t overwide {
      17u, dynamic_roi.top + 35u, 417u, dynamic_roi.top + 45u
    };
    ASSERT_TRUE(overwide_fixture.observe(200u, {overwide}, false));
    ASSERT_TRUE(overwide_fixture.observe(201u, {overwide}, false));
    EXPECT_EQ(overwide_fixture.state()[2u], 0u);
    EXPECT_TRUE(overwide_fixture.output_is_exact_base());

    // Even when OCR8 repeats the same malformed bounds, SLR8 independently derives the safe ROI
    // from source/field geometry and rejects the record instead of trusting its header.
    const auto wrong_top = dynamic_roi.top + 1u;
    slr8_warp_fixture_t malformed_fixture(
      portrait_field_width,
      portrait_field_height,
      portrait_source_width,
      portrait_source_height,
      wrong_top,
      dynamic_roi.bottom
    );
    error.clear();
    ASSERT_TRUE(malformed_fixture.initialize(error)) << error;
    const line_box_t line {40u, wrong_top + 35u, 360u, wrong_top + 45u};
    ASSERT_TRUE(malformed_fixture.observe(210u, {line}, false));
    EXPECT_EQ(malformed_fixture.state()[0u], slr_schema);
    EXPECT_EQ(malformed_fixture.state()[1u], slr_tag);
    EXPECT_EQ(malformed_fixture.state()[2u], 0u);
    EXPECT_EQ(malformed_fixture.state()[4u], 0u);
    EXPECT_EQ(malformed_fixture.state()[20u], 0u);
    EXPECT_TRUE(malformed_fixture.output_is_exact_base());
  }
}  // namespace

#else

TEST(HostSbsSubtitleSlr8GpuTest, WindowsOnly) {
  GTEST_SKIP() << "D3D11 WARP is Windows-only";
}

#endif
