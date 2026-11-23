/**
 * @file tests/unit/test_depth_coordinate_v2_eye_limit_gpu.cpp
 * @brief D3D11-WARP checks for the eye-specific horizontal limiter.
 */
#include "../tests_common.h"

#ifdef _WIN32

  #include <algorithm>
  #include <array>
  #include <cmath>
  #include <cstdint>
  #include <d3d11.h>
  #include <d3dcompiler.h>
  #include <filesystem>
  #include <limits>
  #include <src/depth_coordinate_v2.h>
  #include <src/host_sbs_eye_field.h>
  #include <src/host_sbs_subtitle_publication.h>
  #include <vector>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;

  struct eye_limit_constants_t {
    float occlusion_slope;
    float hole_slope;
    std::uint32_t extension_cells;  // low 16 bits: covered side, high 16 bits: hole side
    std::uint32_t compact;
  };

  /** Largest of `value` and the `cells` inputs that precede index `x` in sweep direction `step`. */
  float extended(const float *row, const std::uint32_t width, const std::uint32_t x, const int step, const std::uint32_t cells) {
    float result = row[x];
    for (std::uint32_t back = 1u; back <= cells; ++back) {
      const auto source = static_cast<std::int64_t>(x) - static_cast<std::int64_t>(step) * back;
      // Before the start of a sweep the shader's history holds the first input.
      result = std::max(result, row[static_cast<std::size_t>(std::clamp<std::int64_t>(source, 0, width - 1))]);
    }
    return result;
  }

  struct eye_fields_t {
    std::vector<float> left;
    std::vector<float> right;
  };

  /** Content columns [left, right) of the field; the whole width when empty. */
  struct content_columns_t {
    std::uint32_t left = 0u;
    std::uint32_t right = 0u;
  };

  /**
   * The shader's own recurrences, in the same operation order. The slopes are per content width;
   * with `compact` set, a column outside the content reads the nearest content column.
   */
  eye_fields_t eye_limit_reference(
    const std::vector<float> &field,
    const std::uint32_t width,
    const std::uint32_t height,
    const eye_limit_constants_t &constants,
    const content_columns_t content = {}
  ) {
    const std::uint32_t left = content.right ? content.left : 0u;
    const std::uint32_t right = content.right ? content.right : width;
    std::vector<float> share = field;
    if (constants.compact) {
      for (std::uint32_t y = 0u; y < height; ++y) {
        for (std::uint32_t x = 0u; x < width; ++x) {
          share[static_cast<std::size_t>(y) * width + x] =
            field[static_cast<std::size_t>(y) * width + std::clamp(x, left, right - 1u)];
        }
      }
    }
    eye_fields_t result {std::vector<float>(share.size()), std::vector<float>(share.size())};
    const float hole_step = constants.hole_slope / static_cast<float>(right - left);
    const float covered_step = constants.occlusion_slope / static_cast<float>(right - left);
    const std::uint32_t covered_cells = constants.extension_cells & 0xffffu;
    const std::uint32_t hole_cells = constants.extension_cells >> 16u;
    for (std::uint32_t y = 0u; y < height; ++y) {
      const auto row = static_cast<std::size_t>(y) * width;
      const float *input = share.data() + row;
      float gentle = input[0];
      float steep = input[0];
      result.left[row] = steep;
      result.right[row] = gentle;
      for (std::uint32_t x = 1u; x < width; ++x) {
        gentle = std::max(extended(input, width, x, 1, hole_cells), gentle - hole_step);
        steep = std::max(extended(input, width, x, 1, covered_cells), steep - covered_step);
        result.left[row + x] = steep;
        result.right[row + x] = gentle;
      }
      gentle = input[width - 1u];
      steep = input[width - 1u];
      for (std::uint32_t x = width; x-- > 0u;) {
        if (x + 1u < width) {
          gentle = std::max(extended(input, width, x, -1, hole_cells), gentle - hole_step);
          steep = std::max(extended(input, width, x, -1, covered_cells), steep - covered_step);
        }
        result.left[row + x] = std::max(result.left[row + x], gentle);
        result.right[row + x] = std::max(result.right[row + x], steep);
      }
    }
    return result;
  }

  bool dispatch_eye_limit(
    ID3D11Device *device,
    ID3D11DeviceContext *context,
    ID3D11ComputeShader *shader,
    const std::uint32_t width,
    const std::uint32_t height,
    const std::vector<float> &share,
    const eye_limit_constants_t &eye_constants,
    eye_fields_t &result,
    const content_columns_t content = {}
  ) {
    namespace v2 = models::depth_coordinate_v2;
    D3D11_TEXTURE2D_DESC share_desc {};
    share_desc.Width = width;
    share_desc.Height = height;
    share_desc.MipLevels = 1u;
    share_desc.ArraySize = 1u;
    share_desc.Format = DXGI_FORMAT_R32_FLOAT;
    share_desc.SampleDesc.Count = 1u;
    share_desc.Usage = D3D11_USAGE_IMMUTABLE;
    share_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA share_data {share.data(), static_cast<UINT>(width * sizeof(float)), 0u};
    ComPtr<ID3D11Texture2D> share_texture;
    ComPtr<ID3D11ShaderResourceView> share_srv;

    auto eye_desc = share_desc;
    eye_desc.Height = height * 2u;
    eye_desc.Usage = D3D11_USAGE_DEFAULT;
    eye_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    ComPtr<ID3D11Texture2D> eye_texture;
    ComPtr<ID3D11UnorderedAccessView> eye_uav;

    auto staging_desc = eye_desc;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0u;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> staging;

    std::array<std::uint32_t, 16> constants {};
    constants[0] = width;
    constants[1] = height;
    constants[9] = content.right ? content.left : 0u;
    constants[11] = content.right ? content.right : width;
    constants[12] = height;
    const v2::constants_t v2_constants {
      v2::model_calibrations.front().raw_coordinate_scale,
      v2::collapse_abs_epsilon,
      0.0f,
      0.0f,
      v2::requested_gain_for_config(v2::reference_pop_strength),
      v2::max_horizontal_slope,
      v2::direct_container_limit,
      v2::convergence_curve_default,
      v2::adaptive_policy_id,
    };
    const auto constant_buffer = [&](const void *bytes, const std::size_t size, ComPtr<ID3D11Buffer> &buffer) {
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = static_cast<UINT>(size);
      desc.Usage = D3D11_USAGE_IMMUTABLE;
      desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      D3D11_SUBRESOURCE_DATA data {bytes, 0u, 0u};
      return SUCCEEDED(device->CreateBuffer(&desc, &data, &buffer));
    };
    ComPtr<ID3D11Buffer> depth_buffer;
    ComPtr<ID3D11Buffer> v2_buffer;
    ComPtr<ID3D11Buffer> eye_buffer;
    if (FAILED(device->CreateTexture2D(&share_desc, &share_data, &share_texture)) ||
        FAILED(device->CreateShaderResourceView(share_texture.Get(), nullptr, &share_srv)) ||
        FAILED(device->CreateTexture2D(&eye_desc, nullptr, &eye_texture)) ||
        FAILED(device->CreateUnorderedAccessView(eye_texture.Get(), nullptr, &eye_uav)) ||
        FAILED(device->CreateTexture2D(&staging_desc, nullptr, &staging)) ||
        !constant_buffer(constants.data(), sizeof(constants), depth_buffer) ||
        !constant_buffer(&v2_constants, sizeof(v2_constants), v2_buffer) ||
        !constant_buffer(&eye_constants, sizeof(eye_constants), eye_buffer)) {
      return false;
    }

    // Sentinel: every texel must be written.
    const float sentinel[4] = {std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f, 0.0f};
    context->ClearUnorderedAccessViewFloat(eye_uav.Get(), sentinel);
    ID3D11Buffer *base_constants[2] = {depth_buffer.Get(), v2_buffer.Get()};
    auto *eye_constant_buffer = eye_buffer.Get();
    auto *srv = share_srv.Get();
    auto *uav = eye_uav.Get();
    context->CSSetShader(shader, nullptr, 0u);
    context->CSSetConstantBuffers(0u, 2u, base_constants);
    context->CSSetConstantBuffers(3u, 1u, &eye_constant_buffer);
    context->CSSetShaderResources(0u, 1u, &srv);
    context->CSSetUnorderedAccessViews(0u, 1u, &uav, nullptr);
    context->Dispatch((height + v2::limiter_group_lines - 1u) / v2::limiter_group_lines, 1u, 1u);
    ID3D11UnorderedAccessView *null_uav = nullptr;
    context->CSSetUnorderedAccessViews(0u, 1u, &null_uav, nullptr);
    context->CopyResource(staging.Get(), eye_texture.Get());
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) {
      return false;
    }
    result.left.resize(share.size());
    result.right.resize(share.size());
    for (std::uint32_t y = 0u; y < height * 2u; ++y) {
      const auto *row = reinterpret_cast<const float *>(static_cast<const std::uint8_t *>(mapped.pData) + mapped.RowPitch * y);
      auto &half = y < height ? result.left : result.right;
      std::copy_n(row, width, half.begin() + static_cast<std::size_t>(y % height) * width);
    }
    context->Unmap(staging.Get(), 0u);
    return true;
  }
}  // namespace

TEST(DepthCoordinateV2EyeLimitGpuTest, EachEyeGetsItsOwnOneSidedSlopesAndForegroundExtension) {
  namespace v2 = models::depth_coordinate_v2;
  ComPtr<ID3D11Device> device;
  ComPtr<ID3D11DeviceContext> context;
  constexpr D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_0};
  D3D_FEATURE_LEVEL actual {};
  ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(
    nullptr,
    D3D_DRIVER_TYPE_WARP,
    nullptr,
    0,
    requested,
    1u,
    D3D11_SDK_VERSION,
    &device,
    &actual,
    &context
  )));

  const auto shader_path = std::filesystem::path(SUNSHINE_SHADERS_DIR) / "depth_coordinate_v2_eye_limit_cs.hlsl";
  ComPtr<ID3DBlob> blob;
  ComPtr<ID3DBlob> errors;
  ASSERT_TRUE(SUCCEEDED(D3DCompileFromFile(
    shader_path.c_str(),
    nullptr,
    D3D_COMPILE_STANDARD_FILE_INCLUDE,
    "main",
    "cs_5_0",
    D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
    0u,
    &blob,
    &errors
  ))) << (errors ? static_cast<const char *>(errors->GetBufferPointer()) : "no compiler diagnostics");
  ComPtr<ID3D11ComputeShader> shader;
  ASSERT_TRUE(SUCCEEDED(device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &shader)));

  constexpr std::uint32_t width = 160u;
  constexpr std::uint32_t height = 11u;  // not a multiple of the group size
  std::vector<float> share(static_cast<std::size_t>(width) * height, -0.03f);
  for (std::uint32_t y = 0u; y < height; ++y) {
    for (std::uint32_t x = 0u; x < width; ++x) {
      auto &value = share[static_cast<std::size_t>(y) * width + x];
      if (y % 4u == 0u) {
        value = x >= 60u && x < 64u ? 0.035f : -0.03f;  // thin near bar on a far plane
      } else if (y % 4u == 1u) {
        value = x < 70u ? 0.03f : -0.035f;  // one cliff
      } else if (y % 4u == 2u) {
        value = -0.03f + 0.06f * static_cast<float>(x) / static_cast<float>(width);  // gentle ramp
      } else {
        value = 0.02f * std::sin(0.37f * static_cast<float>(x + 3u * y));  // texture
      }
    }
  }

  const eye_limit_constants_t constants {
    models::host_sbs_eye_field::covered_side_slope,
    v2::max_horizontal_slope,
    models::host_sbs_eye_field::extension_word(
      models::host_sbs_eye_field::covered_extension_cells,
      models::host_sbs_eye_field::uncovered_extension_cells
    ),
    0u,
  };
  const std::uint32_t covered_cells = models::host_sbs_eye_field::covered_extension_cells;
  const std::uint32_t uncovered_cells = models::host_sbs_eye_field::uncovered_extension_cells;
  ASSERT_LE(covered_cells, 8u);
  ASSERT_LE(uncovered_cells, 8u);
  ASSERT_LT(constants.occlusion_slope, 1.0f);
  ASSERT_GT(constants.occlusion_slope, constants.hole_slope);
  eye_fields_t gpu;
  ASSERT_TRUE(dispatch_eye_limit(device.Get(), context.Get(), shader.Get(), width, height, share, constants, gpu));
  const auto reference = eye_limit_reference(share, width, height, constants);
  const float covered_step = constants.occlusion_slope / static_cast<float>(width);
  const float hole_step = constants.hole_slope / static_cast<float>(width);
  constexpr float tolerance = 2.0e-7f;
  for (std::uint32_t y = 0u; y < height; ++y) {
    for (std::uint32_t x = 0u; x < width; ++x) {
      const auto index = static_cast<std::size_t>(y) * width + x;
      ASSERT_TRUE(std::isfinite(gpu.left[index]) && std::isfinite(gpu.right[index])) << "unwritten x=" << x << ", y=" << y;
      EXPECT_NEAR(gpu.left[index], reference.left[index], tolerance) << "left x=" << x << ", y=" << y;
      EXPECT_NEAR(gpu.right[index], reference.right[index], tolerance) << "right x=" << x << ", y=" << y;
      // Majorants of the input, inside the container.
      EXPECT_GE(gpu.left[index], share[index]);
      EXPECT_GE(gpu.right[index], share[index]);
      EXPECT_LE(std::abs(gpu.left[index]), v2::direct_container_limit);
      EXPECT_LE(std::abs(gpu.right[index]), v2::direct_container_limit);
      // The foreground value is carried unchanged onto the covered side (right of it for the left
      // eye, left of it for the right eye) and onto the uncovered side.
      for (std::uint32_t reach = 1u; reach <= std::max(covered_cells, uncovered_cells); ++reach) {
        if (x >= reach) {
          if (reach <= covered_cells) {
            EXPECT_GE(gpu.left[index], share[index - reach]);
            EXPECT_GE(gpu.right[index - reach], share[index]);
          }
          if (reach <= uncovered_cells) {
            EXPECT_GE(gpu.right[index], share[index - reach]);
            EXPECT_GE(gpu.left[index - reach], share[index]);
          }
        }
      }
      if (x > 0u) {
        // Left eye (-1): falls towards the right at most at the covered-side slope and rises at
        // most at the hole-side slope. The right eye is the mirror image.
        const float left_rise = gpu.left[index] - gpu.left[index - 1u];
        const float right_rise = gpu.right[index] - gpu.right[index - 1u];
        EXPECT_LE(left_rise, hole_step + tolerance) << "x=" << x << ", y=" << y;
        EXPECT_LE(-left_rise, covered_step + tolerance) << "x=" << x << ", y=" << y;
        EXPECT_LE(right_rise, covered_step + tolerance) << "x=" << x << ", y=" << y;
        EXPECT_LE(-right_rise, hole_step + tolerance) << "x=" << x << ", y=" << y;
      }
    }
  }
  // The thin bar (row 0) keeps its full value in both eyes and drags less background on the
  // covered side than on the hole side.
  EXPECT_FLOAT_EQ(gpu.left[61u], 0.035f);
  EXPECT_FLOAT_EQ(gpu.right[61u], 0.035f);
  EXPECT_LT(gpu.right[50u], gpu.left[50u]);  // left of the bar: covered for the right eye
  EXPECT_LT(gpu.left[74u], gpu.right[74u]);  // right of the bar: covered for the left eye

  // An extension of zero cells on one side leaves that side's ramp at the foreground value itself.
  const eye_limit_constants_t covered_only {
    models::host_sbs_eye_field::covered_side_slope,
    v2::max_horizontal_slope,
    models::host_sbs_eye_field::extension_word(3u, 0u),
    0u,
  };
  eye_fields_t one_sided;
  ASSERT_TRUE(dispatch_eye_limit(device.Get(), context.Get(), shader.Get(), width, height, share, covered_only, one_sided));
  const auto one_sided_reference = eye_limit_reference(share, width, height, covered_only);
  for (std::size_t index = 0u; index < share.size(); ++index) {
    ASSERT_NEAR(one_sided.left[index], one_sided_reference.left[index], tolerance) << "one-sided left " << index;
    ASSERT_NEAR(one_sided.right[index], one_sided_reference.right[index], tolerance) << "one-sided right " << index;
  }

  // The pass splits every row into eight chunks. Widths that are not a multiple of eight, narrower
  // than the extension, or as wide as the production field must still equal the serial recurrence.
  for (const std::uint32_t ragged_width : {5u, 9u, 31u, 157u, 1541u}) {
    constexpr std::uint32_t ragged_height = 3u;
    std::vector<float> ragged(static_cast<std::size_t>(ragged_width) * ragged_height);
    for (std::uint32_t y = 0u; y < ragged_height; ++y) {
      for (std::uint32_t x = 0u; x < ragged_width; ++x) {
        const bool object = ((x + 7u * y) / 23u) % 3u == 1u;
        ragged[static_cast<std::size_t>(y) * ragged_width + x] =
          (object ? 0.03f : -0.02f) + 0.004f * std::sin(0.61f * static_cast<float>(x + 5u * y));
      }
    }
    eye_fields_t chunked;
    ASSERT_TRUE(dispatch_eye_limit(device.Get(), context.Get(), shader.Get(), ragged_width, ragged_height, ragged, constants, chunked));
    const auto serial = eye_limit_reference(ragged, ragged_width, ragged_height, constants);
    for (std::size_t index = 0u; index < ragged.size(); ++index) {
      ASSERT_NEAR(chunked.left[index], serial.left[index], tolerance) << "width " << ragged_width << " left " << index;
      ASSERT_NEAR(chunked.right[index], serial.right[index], tolerance) << "width " << ragged_width << " right " << index;
    }
  }

  // Cropped content (the compact limiter path): columns outside the content read its edge column
  // and the slopes are per content width, for the whole width and across chunk borders.
  for (const std::uint32_t cropped_width : {160u, 157u}) {
    constexpr std::uint32_t cropped_height = 3u;
    const content_columns_t content {11u, cropped_width - 23u};
    std::vector<float> cropped(static_cast<std::size_t>(cropped_width) * cropped_height);
    for (std::uint32_t y = 0u; y < cropped_height; ++y) {
      for (std::uint32_t x = 0u; x < cropped_width; ++x) {
        const bool object = ((x + 5u * y) / 19u) % 3u == 1u;
        // Values outside the content are deliberately different: they must not be read.
        const bool outside = x < content.left || x >= content.right;
        cropped[static_cast<std::size_t>(y) * cropped_width + x] =
          outside ? 0.039f : (object ? 0.03f : -0.02f) + 0.004f * std::sin(0.47f * static_cast<float>(x + 3u * y));
      }
    }
    auto compact_constants = constants;
    compact_constants.compact = 1u;
    eye_fields_t compact;
    ASSERT_TRUE(dispatch_eye_limit(device.Get(), context.Get(), shader.Get(), cropped_width, cropped_height, cropped, compact_constants, compact, content));
    const auto compact_reference = eye_limit_reference(cropped, cropped_width, cropped_height, compact_constants, content);
    for (std::size_t index = 0u; index < cropped.size(); ++index) {
      ASSERT_NEAR(compact.left[index], compact_reference.left[index], tolerance) << "cropped width " << cropped_width << " left " << index;
      ASSERT_NEAR(compact.right[index], compact_reference.right[index], tolerance) << "cropped width " << cropped_width << " right " << index;
      ASSERT_LT(compact.left[index], 0.039f) << "a value outside the content was read";
    }
  }

  // With equal slopes and no extension both eyes equal the symmetric least majorant.
  const eye_limit_constants_t symmetric {v2::max_horizontal_slope, v2::max_horizontal_slope, 0u, 0u};
  eye_fields_t same;
  ASSERT_TRUE(dispatch_eye_limit(device.Get(), context.Get(), shader.Get(), width, height, share, symmetric, same));
  const float step = v2::max_horizontal_slope / static_cast<float>(width);
  for (std::uint32_t y = 0u; y < height; ++y) {
    const auto row = static_cast<std::size_t>(y) * width;
    for (std::uint32_t x = 0u; x < width; ++x) {
      float least = -std::numeric_limits<float>::infinity();
      for (std::uint32_t source = 0u; source < width; ++source) {
        least = std::max(least, share[row + source] - step * static_cast<float>(x > source ? x - source : source - x));
      }
      EXPECT_NEAR(same.left[row + x], least, 4.0e-7f) << "x=" << x << ", y=" << y;
      EXPECT_NEAR(same.right[row + x], least, 4.0e-7f) << "x=" << x << ", y=" << y;
    }
  }
}

TEST(DepthCoordinateV2EyeLimitGpuTest, CompanionTravelsWithTheBaseViewAndCanBeDetached) {
  ComPtr<ID3D11Device> device;
  ComPtr<ID3D11DeviceContext> context;
  constexpr D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_0};
  D3D_FEATURE_LEVEL actual {};
  ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(
    nullptr,
    D3D_DRIVER_TYPE_WARP,
    nullptr,
    0,
    requested,
    1u,
    D3D11_SDK_VERSION,
    &device,
    &actual,
    &context
  )));
  D3D11_TEXTURE2D_DESC desc {};
  desc.Width = 8u;
  desc.Height = 4u;
  desc.MipLevels = 1u;
  desc.ArraySize = 1u;
  desc.Format = DXGI_FORMAT_R32_FLOAT;
  desc.SampleDesc.Count = 1u;
  desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
  ComPtr<ID3D11Texture2D> base_texture;
  ComPtr<ID3D11Texture2D> eye_texture;
  ComPtr<ID3D11ShaderResourceView> base;
  ComPtr<ID3D11ShaderResourceView> eye;
  ASSERT_TRUE(SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &base_texture)));
  desc.Height = 8u;
  ASSERT_TRUE(SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &eye_texture)));
  ASSERT_TRUE(SUCCEEDED(device->CreateShaderResourceView(base_texture.Get(), nullptr, &base)));
  ASSERT_TRUE(SUCCEEDED(device->CreateShaderResourceView(eye_texture.Get(), nullptr, &eye)));

  namespace field = models::host_sbs_eye_field;
  EXPECT_EQ(field::companion(nullptr).Get(), nullptr);
  EXPECT_EQ(field::companion(base.Get()).Get(), nullptr);
  ASSERT_TRUE(field::attach(base.Get(), eye.Get()));
  // A second holder of the same Base view reaches the same companion.
  ComPtr<ID3D11ShaderResourceView> holder = base;
  EXPECT_EQ(field::companion(holder.Get()).Get(), eye.Get());

  // A completed subtitle draw must retain the same pair of eye fields as its Base,
  // even when the depth producer rewrites the live texture for the next frame.
  namespace publication = models::subtitle_publication;
  publication::pool_t pool;
  publication::inputs_t inputs;
  inputs.fill(base.Get());
  std::array<float, 64u> values;
  std::fill(values.begin(), values.begin() + 32, 1.0f);
  std::fill(values.begin() + 32, values.end(), 2.0f);
  context->UpdateSubresource(eye_texture.Get(), 0u, nullptr, values.data(), desc.Width * sizeof(float), 0u);
  auto held = pool.publish(device.Get(), context.Get(), inputs);
  ASSERT_TRUE(held);
  auto held_eye = field::companion(held->srv[publication::base].Get());
  ASSERT_TRUE(held_eye);
  EXPECT_NE(held_eye.Get(), eye.Get());
  values.fill(3.0f);
  context->UpdateSubresource(eye_texture.Get(), 0u, nullptr, values.data(), desc.Width * sizeof(float), 0u);
  ASSERT_TRUE(field::attach(base.Get(), nullptr));
  EXPECT_EQ(field::companion(base.Get()).Get(), nullptr);
  auto next = pool.publish(device.Get(), context.Get(), inputs);
  ASSERT_TRUE(next);
  EXPECT_FALSE(field::companion(next->srv[publication::base].Get()));
  ComPtr<ID3D11Resource> held_resource;
  held_eye->GetResource(&held_resource);
  desc.Usage = D3D11_USAGE_STAGING;
  desc.BindFlags = 0u;
  desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
  ComPtr<ID3D11Texture2D> staging;
  ASSERT_TRUE(SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &staging)));
  context->CopyResource(staging.Get(), held_resource.Get());
  D3D11_MAPPED_SUBRESOURCE mapped {};
  ASSERT_TRUE(SUCCEEDED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped)));
  for (UINT y = 0; y < desc.Height; ++y) {
    const auto *row = reinterpret_cast<const float *>(static_cast<const char *>(mapped.pData) + y * mapped.RowPitch);
    for (UINT x = 0; x < desc.Width; ++x) EXPECT_EQ(row[x], y < desc.Height / 2 ? 1.0f : 2.0f);
  }
  context->Unmap(staging.Get(), 0u);
  held.reset();
  auto recycled = pool.publish(device.Get(), context.Get(), inputs);
  ASSERT_TRUE(recycled);
  EXPECT_FALSE(field::companion(recycled->srv[publication::base].Get()));
}

#endif  // _WIN32
