/**
 * @file src/platform/windows/sbs_debug_dump.cpp
 * @brief Transactional, same-frame Host-SBS diagnostic package writer.
 */
#include "sbs_debug_dump.h"

// standard includes
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

// platform includes
#include <windows.h>
#include <wrl/client.h>

// lib includes
#include <nlohmann/json.hpp>
#include <zlib.h>

// local includes
#include "src/generated/sbs_adaptive_state_contract.h"
#include "src/generated/sbs_scene_controller_contract.h"
#include "src/logging.h"
#include "src/sbs_scene_cache_contract.h"
#include "src/video_depth_estimator.h"

namespace platf::sbs_debug {

  using namespace std::literals;

  namespace {

    static_assert(
      std::endian::native == std::endian::little,
      "Dump contracts explicitly use little-endian float and integer words."
    );

    constexpr unsigned retry_backoff_frames = 60;
    constexpr std::array<float, 3> imagenet_mean {0.485f, 0.456f, 0.406f};
    constexpr std::array<float, 3> imagenet_std {0.229f, 0.224f, 0.225f};

    struct normalization_state {
      float lower = 0.0f;
      float upper = 0.0f;
      float initialized = 0.0f;
      float frame_state = 0.0f;
    };

    enum class depth_dumpability {
      valid,
      invalid,
      unreadable,
    };

    const char *normalization_frame_state_name(const float value) {
      if (value < 0.5f) {
        return "invalid-held";
      }
      if (value < 1.5f) {
        return "valid-with-history";
      }
      return "first-valid";
    }

    struct texture_snapshot {
      D3D11_TEXTURE2D_DESC desc {};
      std::vector<std::uint8_t> bytes;
      std::size_t row_bytes = 0;
    };

    struct scalar_stats {
      std::size_t finite_count = 0;
      float minimum = std::numeric_limits<float>::quiet_NaN();
      float maximum = std::numeric_limits<float>::quiet_NaN();
      float preview_low = std::numeric_limits<float>::quiet_NaN();
      float preview_high = std::numeric_limits<float>::quiet_NaN();
    };

    struct raw_depth_dump_stats: scalar_stats {
      std::uint32_t width = 0;
      std::uint32_t height = 0;
    };

    struct warp_map_dump_stats: scalar_stats {
      std::uint32_t width = 0;
      std::uint32_t height = 0;
      std::uint32_t eye_width = 0;
      std::uint32_t eye_height = 0;
      float content_scale_x = 1.0f;
      float content_scale_y = 1.0f;
      float displacement_preview_abs_px = 0.0f;
    };

    std::uint32_t format_bytes_per_pixel(const DXGI_FORMAT format) {
      switch (format) {
        case DXGI_FORMAT_R32_FLOAT:
        case DXGI_FORMAT_R32_UINT:
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
          return 4;
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
          return 8;
        default:
          return 0;
      }
    }

    std::string format_name(const DXGI_FORMAT format) {
      switch (format) {
        case DXGI_FORMAT_R32_FLOAT:
          return "DXGI_FORMAT_R32_FLOAT";
        case DXGI_FORMAT_R32_UINT:
          return "DXGI_FORMAT_R32_UINT";
        case DXGI_FORMAT_B8G8R8A8_UNORM:
          return "DXGI_FORMAT_B8G8R8A8_UNORM";
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
          return "DXGI_FORMAT_B8G8R8A8_UNORM_SRGB";
        case DXGI_FORMAT_R8G8B8A8_UNORM:
          return "DXGI_FORMAT_R8G8B8A8_UNORM";
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
          return "DXGI_FORMAT_R8G8B8A8_UNORM_SRGB";
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
          return "DXGI_FORMAT_R16G16B16A16_FLOAT";
        default:
          return "DXGI_FORMAT_UNKNOWN_" + std::to_string(static_cast<unsigned>(format));
      }
    }

    nlohmann::json texture_description(const texture_snapshot &snapshot) {
      return {
        {"width", snapshot.desc.Width},
        {"height", snapshot.desc.Height},
        {"format", format_name(snapshot.desc.Format)},
        {"format_value", static_cast<unsigned>(snapshot.desc.Format)},
      };
    }

    inline float half_to_float(const std::uint16_t h) {
      const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16u;
      int exp = (h & 0x7C00u) >> 10u;
      std::uint32_t mant = h & 0x03FFu;
      std::uint32_t bits;
      if (exp == 0) {
        if (mant == 0) {
          bits = sign;
        } else {
          exp = 127 - 15 + 1;
          while (!(mant & 0x0400u)) {
            mant <<= 1u;
            --exp;
          }
          mant &= 0x03FFu;
          bits = sign | (static_cast<std::uint32_t>(exp) << 23u) | (mant << 13u);
        }
      } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13u);
      } else {
        bits = sign |
               (static_cast<std::uint32_t>(exp - 15 + 127) << 23u) |
               (mant << 13u);
      }
      return std::bit_cast<float>(bits);
    }

    inline std::uint8_t encode_srgb(float value) {
      value = std::clamp(value, 0.0f, 1.0f);
      const float encoded = value <= 0.0031308f ?
                              12.92f * value :
                              1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
      return static_cast<std::uint8_t>(std::lround(encoded * 255.0f));
    }

    inline std::uint8_t encode_unit(float value) {
      return static_cast<std::uint8_t>(
        std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f)
      );
    }

    inline void tonemap_scrgb(float &r, float &g, float &b) {
      r = std::max(r, 0.0f);
      g = std::max(g, 0.0f);
      b = std::max(b, 0.0f);
      const float luminance =
        std::max(0.2126f * r + 0.7152f * g + 0.0722f * b, 0.0f);
      const float tone_scale = 1.0f / (1.0f + luminance);
      r *= tone_scale;
      g *= tone_scale;
      b *= tone_scale;
      const float gamut_scale =
        1.0f / std::max(1.0f, std::max(r, std::max(g, b)));
      r *= gamut_scale;
      g *= gamut_scale;
      b *= gamut_scale;
    }

    inline void colormap_jet(float value, std::uint8_t &r, std::uint8_t &g, std::uint8_t &b) {
      const float t = std::clamp(value, 0.0f, 1.0f);
      const auto channel = [](const float x) {
        return static_cast<std::uint8_t>(
          std::lround(std::clamp(x, 0.0f, 1.0f) * 255.0f)
        );
      };
      r = channel(1.5f - std::fabs(4.0f * t - 3.0f));
      g = channel(1.5f - std::fabs(4.0f * t - 2.0f));
      b = channel(1.5f - std::fabs(4.0f * t - 1.0f));
    }

    bool write_bytes(
      const std::filesystem::path &path,
      const void *data,
      const std::size_t size
    ) {
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      if (!out) {
        return false;
      }
      if (size != 0) {
        out.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
      }
      out.flush();
      return out.good();
    }

    bool write_text(const std::filesystem::path &path, const std::string &text) {
      return write_bytes(path, text.data(), text.size());
    }

    bool write_json(const std::filesystem::path &path, const nlohmann::json &value) {
      try {
        return write_text(path, value.dump(2) + "\n");
      } catch (const std::exception &error) {
        BOOST_LOG(warning) << "SBS debug dump: JSON serialization failed for "sv
                           << path.string() << ": " << error.what();
        return false;
      }
    }

    bool write_png(
      const std::filesystem::path &path,
      const std::uint32_t width,
      const std::uint32_t height,
      const std::vector<std::uint8_t> &rgb
    ) {
      if (
        width == 0 || height == 0 ||
        static_cast<std::uint64_t>(width) * height >
          std::numeric_limits<std::size_t>::max() / 3u ||
        rgb.size() != static_cast<std::size_t>(width) * height * 3u
      ) {
        return false;
      }

      const std::size_t scanline = 1u + static_cast<std::size_t>(width) * 3u;
      if (static_cast<std::size_t>(height) > SIZE_MAX / scanline) {
        return false;
      }
      std::vector<std::uint8_t> raw(static_cast<std::size_t>(height) * scanline);
      for (std::uint32_t y = 0; y < height; ++y) {
        const std::size_t output_offset = static_cast<std::size_t>(y) * scanline;
        raw[output_offset] = 0;
        std::memcpy(
          raw.data() + output_offset + 1u,
          rgb.data() + static_cast<std::size_t>(y) * width * 3u,
          static_cast<std::size_t>(width) * 3u
        );
      }

      if (
        raw.size() > std::numeric_limits<uLong>::max() ||
        raw.size() > std::numeric_limits<uLongf>::max()
      ) {
        return false;
      }
      uLongf compressed_size = compressBound(static_cast<uLong>(raw.size()));
      std::vector<std::uint8_t> compressed(compressed_size);
      if (
        compress2(
          compressed.data(),
          &compressed_size,
          raw.data(),
          static_cast<uLong>(raw.size()),
          Z_BEST_SPEED
        ) != Z_OK
      ) {
        return false;
      }

      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      if (!out) {
        return false;
      }
      const auto write_be32 = [](const std::uint32_t value, std::uint8_t *bytes) {
        bytes[0] = static_cast<std::uint8_t>(value >> 24u);
        bytes[1] = static_cast<std::uint8_t>(value >> 16u);
        bytes[2] = static_cast<std::uint8_t>(value >> 8u);
        bytes[3] = static_cast<std::uint8_t>(value);
      };
      const auto write_chunk = [&](const char *type, const std::uint8_t *data, const std::uint32_t size) {
        std::uint8_t encoded_size[4];
        write_be32(size, encoded_size);
        out.write(reinterpret_cast<const char *>(encoded_size), sizeof(encoded_size));
        out.write(type, 4);
        if (size != 0) {
          out.write(reinterpret_cast<const char *>(data), size);
        }
        uLong crc = crc32(0, reinterpret_cast<const Bytef *>(type), 4);
        if (size != 0) {
          crc = crc32(crc, data, size);
        }
        std::uint8_t encoded_crc[4];
        write_be32(static_cast<std::uint32_t>(crc), encoded_crc);
        out.write(reinterpret_cast<const char *>(encoded_crc), sizeof(encoded_crc));
        return out.good();
      };

      static constexpr std::uint8_t signature[8] {
        0x89,
        'P',
        'N',
        'G',
        0x0D,
        0x0A,
        0x1A,
        0x0A
      };
      out.write(reinterpret_cast<const char *>(signature), sizeof(signature));
      std::uint8_t header[13] {};
      write_be32(width, header);
      write_be32(height, header + 4);
      header[8] = 8;
      header[9] = 2;
      const bool ok =
        out.good() &&
        write_chunk("IHDR", header, sizeof(header)) &&
        write_chunk(
          "IDAT",
          compressed.data(),
          static_cast<std::uint32_t>(compressed_size)
        ) &&
        write_chunk("IEND", nullptr, 0);
      out.flush();
      return ok && out.good();
    }

    bool read_buffer(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv,
      const std::size_t required_bytes,
      std::vector<std::uint8_t> &bytes
    ) {
      if (!device || !ctx || !srv || required_bytes == 0) {
        return false;
      }
      Microsoft::WRL::ComPtr<ID3D11Resource> resource;
      srv->GetResource(&resource);
      Microsoft::WRL::ComPtr<ID3D11Buffer> buffer;
      if (!resource || FAILED(resource.As(&buffer))) {
        return false;
      }

      D3D11_BUFFER_DESC source_desc {};
      buffer->GetDesc(&source_desc);
      if (required_bytes > source_desc.ByteWidth) {
        return false;
      }
      D3D11_BUFFER_DESC staging_desc = source_desc;
      staging_desc.Usage = D3D11_USAGE_STAGING;
      staging_desc.BindFlags = 0;
      staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      staging_desc.MiscFlags = 0;
      staging_desc.StructureByteStride = 0;
      Microsoft::WRL::ComPtr<ID3D11Buffer> staging;
      if (FAILED(device->CreateBuffer(&staging_desc, nullptr, &staging))) {
        return false;
      }

      ctx->CopyResource(staging.Get(), buffer.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        return false;
      }
      bytes.resize(required_bytes);
      std::memcpy(bytes.data(), mapped.pData, required_bytes);
      ctx->Unmap(staging.Get(), 0);
      return true;
    }

    bool read_float_buffer(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv,
      const std::size_t value_count,
      std::vector<float> &values
    ) {
      if (value_count > SIZE_MAX / sizeof(float)) {
        return false;
      }
      std::vector<std::uint8_t> bytes;
      if (!read_buffer(device, ctx, srv, value_count * sizeof(float), bytes)) {
        return false;
      }
      values.resize(value_count);
      std::memcpy(values.data(), bytes.data(), bytes.size());
      return true;
    }

    bool read_texture(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv,
      texture_snapshot &snapshot
    ) {
      if (!device || !ctx || !srv) {
        return false;
      }
      Microsoft::WRL::ComPtr<ID3D11Resource> resource;
      srv->GetResource(&resource);
      Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
      if (!resource || FAILED(resource.As(&texture))) {
        return false;
      }
      texture->GetDesc(&snapshot.desc);
      const std::uint32_t bytes_per_pixel = format_bytes_per_pixel(snapshot.desc.Format);
      if (
        bytes_per_pixel == 0 || snapshot.desc.Width == 0 || snapshot.desc.Height == 0 ||
        snapshot.desc.ArraySize != 1 || snapshot.desc.MipLevels != 1 ||
        snapshot.desc.SampleDesc.Count != 1
      ) {
        return false;
      }
      snapshot.row_bytes =
        static_cast<std::size_t>(snapshot.desc.Width) * bytes_per_pixel;
      if (
        static_cast<std::size_t>(snapshot.desc.Height) >
        SIZE_MAX / snapshot.row_bytes
      ) {
        return false;
      }

      D3D11_TEXTURE2D_DESC staging_desc = snapshot.desc;
      staging_desc.Usage = D3D11_USAGE_STAGING;
      staging_desc.BindFlags = 0;
      staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      staging_desc.MiscFlags = 0;
      Microsoft::WRL::ComPtr<ID3D11Texture2D> staging;
      if (FAILED(device->CreateTexture2D(&staging_desc, nullptr, &staging))) {
        return false;
      }
      ctx->CopyResource(staging.Get(), texture.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        return false;
      }
      if (mapped.RowPitch < snapshot.row_bytes) {
        ctx->Unmap(staging.Get(), 0);
        return false;
      }
      snapshot.bytes.resize(
        static_cast<std::size_t>(snapshot.desc.Height) * snapshot.row_bytes
      );
      for (std::uint32_t y = 0; y < snapshot.desc.Height; ++y) {
        std::memcpy(
          snapshot.bytes.data() + static_cast<std::size_t>(y) * snapshot.row_bytes,
          static_cast<const std::uint8_t *>(mapped.pData) +
            static_cast<std::size_t>(y) * mapped.RowPitch,
          snapshot.row_bytes
        );
      }
      ctx->Unmap(staging.Get(), 0);
      return true;
    }

    bool texture_to_rgb(
      const texture_snapshot &snapshot,
      const models::input_color_space color_space,
      std::vector<std::uint8_t> &rgb
    ) {
      const auto width = snapshot.desc.Width;
      const auto height = snapshot.desc.Height;
      if (
        static_cast<std::uint64_t>(width) * height >
        SIZE_MAX / 3u
      ) {
        return false;
      }
      rgb.resize(static_cast<std::size_t>(width) * height * 3u);
      for (std::uint32_t y = 0; y < height; ++y) {
        const std::uint8_t *input =
          snapshot.bytes.data() + static_cast<std::size_t>(y) * snapshot.row_bytes;
        std::uint8_t *output =
          rgb.data() + static_cast<std::size_t>(y) * width * 3u;
        for (std::uint32_t x = 0; x < width; ++x) {
          std::uint8_t r;
          std::uint8_t g;
          std::uint8_t b;
          switch (snapshot.desc.Format) {
            case DXGI_FORMAT_R16G16B16A16_FLOAT:
              {
                const auto *pixel =
                  reinterpret_cast<const std::uint16_t *>(input + static_cast<std::size_t>(x) * 8u);
                float rf = half_to_float(pixel[0]);
                float gf = half_to_float(pixel[1]);
                float bf = half_to_float(pixel[2]);
                if (!std::isfinite(rf) || !std::isfinite(gf) || !std::isfinite(bf)) {
                  r = 255;
                  g = 0;
                  b = 255;
                } else {
                  if (color_space == models::input_color_space::scrgb_hdr) {
                    tonemap_scrgb(rf, gf, bf);
                  }
                  if (color_space == models::input_color_space::srgb) {
                    // Some capture paths conservatively retain FP16 storage for sRGB code
                    // values. The warp copies those values unchanged, so applying an OETF here
                    // would double-gamma both source.png and sbs.png.
                    r = encode_unit(rf);
                    g = encode_unit(gf);
                    b = encode_unit(bf);
                  } else {
                    r = encode_srgb(rf);
                    g = encode_srgb(gf);
                    b = encode_srgb(bf);
                  }
                }
                break;
              }
            case DXGI_FORMAT_B8G8R8A8_UNORM:
            case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
              {
                const std::uint8_t *pixel = input + static_cast<std::size_t>(x) * 4u;
                b = pixel[0];
                g = pixel[1];
                r = pixel[2];
                break;
              }
            case DXGI_FORMAT_R8G8B8A8_UNORM:
            case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
              {
                const std::uint8_t *pixel = input + static_cast<std::size_t>(x) * 4u;
                r = pixel[0];
                g = pixel[1];
                b = pixel[2];
                break;
              }
            default:
              return false;
          }
          output[static_cast<std::size_t>(x) * 3u + 0u] = r;
          output[static_cast<std::size_t>(x) * 3u + 1u] = g;
          output[static_cast<std::size_t>(x) * 3u + 2u] = b;
        }
      }
      return true;
    }

    bool write_color_preview(
      const std::filesystem::path &path,
      const texture_snapshot &snapshot,
      const models::input_color_space color_space
    ) {
      std::vector<std::uint8_t> rgb;
      return texture_to_rgb(snapshot, color_space, rgb) &&
             write_png(path, snapshot.desc.Width, snapshot.desc.Height, rgb);
    }

    bool texture_float_values(
      const texture_snapshot &snapshot,
      std::vector<float> &values
    ) {
      if (snapshot.desc.Format != DXGI_FORMAT_R32_FLOAT) {
        return false;
      }
      const std::size_t count =
        static_cast<std::size_t>(snapshot.desc.Width) * snapshot.desc.Height;
      if (snapshot.bytes.size() != count * sizeof(float)) {
        return false;
      }
      values.resize(count);
      std::memcpy(values.data(), snapshot.bytes.data(), snapshot.bytes.size());
      return true;
    }

    scalar_stats calculate_scalar_stats(const std::vector<float> &values);

    bool write_scalar_previews(
      const std::filesystem::path &gray_path,
      const std::filesystem::path &heat_path,
      const texture_snapshot &snapshot
    ) {
      std::vector<float> values;
      if (!texture_float_values(snapshot, values)) {
        return false;
      }
      std::vector<std::uint8_t> gray(values.size() * 3u);
      std::vector<std::uint8_t> heat(values.size() * 3u);
      for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
          gray[index * 3u + 0u] = heat[index * 3u + 0u] = 255;
          gray[index * 3u + 1u] = heat[index * 3u + 1u] = 0;
          gray[index * 3u + 2u] = heat[index * 3u + 2u] = 255;
          continue;
        }
        const float value = std::clamp(values[index], 0.0f, 1.0f);
        const std::uint8_t encoded = encode_unit(value);
        gray[index * 3u + 0u] = encoded;
        gray[index * 3u + 1u] = encoded;
        gray[index * 3u + 2u] = encoded;
        colormap_jet(
          value,
          heat[index * 3u + 0u],
          heat[index * 3u + 1u],
          heat[index * 3u + 2u]
        );
      }
      return write_png(
               gray_path,
               snapshot.desc.Width,
               snapshot.desc.Height,
               gray
             ) &&
             write_png(
               heat_path,
               snapshot.desc.Width,
               snapshot.desc.Height,
               heat
             );
    }

    bool write_float_texture_artifacts(
      const std::filesystem::path &data_path,
      const std::filesystem::path &shape_path,
      const texture_snapshot &snapshot,
      const std::string_view stage
    ) {
      std::vector<float> values;
      if (!texture_float_values(snapshot, values)) {
        return false;
      }
      const scalar_stats stats = calculate_scalar_stats(values);
      if (
        !write_bytes(
          data_path,
          values.data(),
          values.size() * sizeof(float)
        )
      ) {
        return false;
      }
      nlohmann::json shape {
        {"schema", 1},
        {"width", snapshot.desc.Width},
        {"height", snapshot.desc.Height},
        {"dtype", "float32-le"},
        {"layout", "row-major"},
        {"stage", std::string(stage)},
        {"finite_count", stats.finite_count},
        {"sample_count", values.size()},
      };
      if (stats.finite_count != 0) {
        shape["minimum"] = stats.minimum;
        shape["maximum"] = stats.maximum;
      }
      return write_json(shape_path, shape);
    }

    scalar_stats calculate_scalar_stats(const std::vector<float> &values) {
      scalar_stats stats;
      std::vector<float> finite;
      finite.reserve(values.size());
      for (const float value : values) {
        if (std::isfinite(value)) {
          finite.push_back(value);
        }
      }
      stats.finite_count = finite.size();
      if (finite.empty()) {
        return stats;
      }
      std::sort(finite.begin(), finite.end());
      const auto percentile = [&](const double fraction) {
        const std::size_t index = static_cast<std::size_t>(
          std::lround(fraction * static_cast<double>(finite.size() - 1u))
        );
        return finite[std::min(index, finite.size() - 1u)];
      };
      stats.minimum = finite.front();
      stats.maximum = finite.back();
      stats.preview_low = percentile(0.02);
      stats.preview_high = percentile(0.98);
      const float scale = std::max(
        1.0f,
        std::max(std::fabs(stats.preview_low), std::fabs(stats.preview_high))
      );
      if (
        !(stats.preview_high - stats.preview_low >
          std::numeric_limits<float>::epsilon() * scale)
      ) {
        stats.preview_low = stats.minimum;
        stats.preview_high = stats.maximum;
      }
      return stats;
    }

    bool write_percentile_previews(
      const std::filesystem::path &gray_path,
      const std::filesystem::path &heat_path,
      const std::uint32_t width,
      const std::uint32_t height,
      const std::vector<float> &values,
      const scalar_stats &stats
    ) {
      if (stats.finite_count == 0) {
        return false;
      }
      const float span = stats.preview_high - stats.preview_low;
      const float scale = std::max(
        1.0f,
        std::max(std::fabs(stats.preview_low), std::fabs(stats.preview_high))
      );
      const bool collapsed =
        !(span > std::numeric_limits<float>::epsilon() * scale);
      std::vector<std::uint8_t> gray(values.size() * 3u);
      std::vector<std::uint8_t> heat(values.size() * 3u);
      for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
          gray[index * 3u + 0u] = heat[index * 3u + 0u] = 255;
          gray[index * 3u + 1u] = heat[index * 3u + 1u] = 0;
          gray[index * 3u + 2u] = heat[index * 3u + 2u] = 255;
          continue;
        }
        const float normalized = collapsed ?
                                   0.5f :
                                   std::clamp(
                                     (values[index] - stats.preview_low) / span,
                                     0.0f,
                                     1.0f
                                   );
        const std::uint8_t encoded = encode_unit(normalized);
        gray[index * 3u + 0u] = encoded;
        gray[index * 3u + 1u] = encoded;
        gray[index * 3u + 2u] = encoded;
        colormap_jet(
          normalized,
          heat[index * 3u + 0u],
          heat[index * 3u + 1u],
          heat[index * 3u + 2u]
        );
      }
      return write_png(gray_path, width, height, gray) &&
             write_png(heat_path, width, height, heat);
    }

    bool dump_model_input(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv,
      const int width,
      const int height,
      const std::filesystem::path &dir
    ) {
      if (width <= 0 || height <= 0) {
        return false;
      }
      const std::uint64_t pixel_count =
        static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
      if (pixel_count > SIZE_MAX / (3u * sizeof(float))) {
        return false;
      }
      std::vector<float> values;
      if (
        !read_float_buffer(
          device,
          ctx,
          srv,
          static_cast<std::size_t>(pixel_count) * 3u,
          values
        ) ||
        !write_bytes(
          dir / "model_input.f32",
          values.data(),
          values.size() * sizeof(float)
        )
      ) {
        return false;
      }

      std::vector<std::uint8_t> rgb(static_cast<std::size_t>(pixel_count) * 3u);
      const std::size_t plane_size = static_cast<std::size_t>(pixel_count);
      for (std::size_t pixel = 0; pixel < plane_size; ++pixel) {
        for (std::size_t channel = 0; channel < 3u; ++channel) {
          const float normalized = values[channel * plane_size + pixel];
          if (!std::isfinite(normalized)) {
            rgb[pixel * 3u + 0u] = 255;
            rgb[pixel * 3u + 1u] = 0;
            rgb[pixel * 3u + 2u] = 255;
            break;
          }
          // rgb_to_nchw_cs stores already-sRGB model values after ImageNet normalization.
          // Reverse only mean/std here; applying the OETF again would corrupt the preview.
          rgb[pixel * 3u + channel] =
            encode_unit(normalized * imagenet_std[channel] + imagenet_mean[channel]);
        }
      }
      const nlohmann::json shape {
        {"schema", 1},
        {"width", width},
        {"height", height},
        {"dtype", "float32-le"},
        {"layout", "NCHW"},
        {"channels", {"R", "G", "B"}},
        {"stage", "exact model input after area resize, HDR tone mapping, sRGB conversion, and ImageNet normalization"},
        {"imagenet_mean", imagenet_mean},
        {"imagenet_std", imagenet_std},
        {"preview", {
                      {"file", "model_input.png"},
                      {"operation", "channel * std + mean, clamped to [0,1]"},
                      {"extra_srgb_oetf", false},
                      {"nonfinite_color", "magenta"},
                    }},
      };
      return write_png(
               dir / "model_input.png",
               static_cast<std::uint32_t>(width),
               static_cast<std::uint32_t>(height),
               rgb
             ) &&
             write_json(dir / "model_input_shape.json", shape);
    }

    bool dump_raw_depth(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv,
      const int width,
      const int height,
      const std::filesystem::path &dir,
      raw_depth_dump_stats &stats
    ) {
      if (width <= 0 || height <= 0) {
        return false;
      }
      const std::uint64_t value_count =
        static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
      if (value_count > SIZE_MAX / sizeof(float)) {
        return false;
      }
      std::vector<float> values;
      if (
        !read_float_buffer(
          device,
          ctx,
          srv,
          static_cast<std::size_t>(value_count),
          values
        )
      ) {
        return false;
      }
      const scalar_stats scalar = calculate_scalar_stats(values);
      static_cast<scalar_stats &>(stats) = scalar;
      stats.width = static_cast<std::uint32_t>(width);
      stats.height = static_cast<std::uint32_t>(height);
      if (
        scalar.finite_count == 0 ||
        !write_bytes(
          dir / "raw_depth.f32",
          values.data(),
          values.size() * sizeof(float)
        ) ||
        !write_percentile_previews(
          dir / "raw_depth.png",
          dir / "raw_depth_heat.png",
          stats.width,
          stats.height,
          values,
          stats
        )
      ) {
        return false;
      }
      const nlohmann::json shape {
        {"schema", 1},
        {"width", stats.width},
        {"height", stats.height},
        {"dtype", "float32-le"},
        {"layout", "row-major"},
        {"stage", "raw model output before transform, robust normalization, temporal EMA, or curvature"},
        {"finite_count", stats.finite_count},
        {"sample_count", values.size()},
        {"minimum", stats.minimum},
        {"maximum", stats.maximum},
        {"preview_normalization", "finite p2-p98"},
        {"preview_low", stats.preview_low},
        {"preview_high", stats.preview_high},
        {"nonfinite_color", "magenta"},
      };
      return write_json(dir / "raw_shape.json", shape);
    }

    depth_dumpability read_normalization_state(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv,
      normalization_state &state
    ) {
      std::vector<float> values;
      if (!read_float_buffer(device, ctx, srv, 4u, values)) {
        return depth_dumpability::unreadable;
      }
      for (const float value : values) {
        if (!std::isfinite(value)) {
          return depth_dumpability::unreadable;
        }
      }
      state = {values[0], values[1], values[2], values[3]};
      return state.frame_state >= 0.5f ?
               depth_dumpability::valid :
               depth_dumpability::invalid;
    }

    const char *encoding_name(const sbs_adaptive_state::gpu_encoding_e encoding) {
      switch (encoding) {
        case sbs_adaptive_state::gpu_encoding_e::float_value:
          return "float_value";
        case sbs_adaptive_state::gpu_encoding_e::uint_bits:
          return "uint_bits";
        case sbs_adaptive_state::gpu_encoding_e::uint_valued_float:
          return "uint_valued_float";
      }
      return "unknown";
    }

    bool dump_adaptive_state(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv,
      const normalization_state &normalization,
      const frame &completed,
      const config::video_t::sbs_t &cfg,
      const std::filesystem::path &dir,
      nlohmann::json &adaptive_summary
    ) {
      std::vector<std::uint8_t> bytes;
      if (
        !read_buffer(
          device,
          ctx,
          srv,
          sbs_adaptive_state::word_count * sizeof(std::uint32_t),
          bytes
        )
      ) {
        return false;
      }
      sbs_adaptive_state::words_t words {};
      std::memcpy(words.data(), bytes.data(), bytes.size());
      nlohmann::json fields = nlohmann::json::array();
      nlohmann::json values = nlohmann::json::array();
      nlohmann::json named_values = nlohmann::json::object();
      std::array<float, sbs_adaptive_state::word_count> scalars {};
      for (const auto &descriptor : sbs_adaptive_state::fields) {
        const std::size_t index = sbs_adaptive_state::index(descriptor.word);
        const std::uint32_t raw_word = words[index];
        nlohmann::json value;
        if (
          descriptor.gpu_encoding ==
          sbs_adaptive_state::gpu_encoding_e::uint_bits
        ) {
          value = raw_word;
        } else {
          const float scalar = std::bit_cast<float>(raw_word);
          if (!std::isfinite(scalar)) {
            return false;
          }
          scalars[index] = scalar;
          if (
            descriptor.gpu_encoding ==
            sbs_adaptive_state::gpu_encoding_e::uint_valued_float
          ) {
            if (
              scalar < 0.0f ||
              scalar > static_cast<float>(std::numeric_limits<std::uint32_t>::max()) ||
              std::trunc(scalar) != scalar
            ) {
              return false;
            }
            value = static_cast<std::uint32_t>(scalar);
          } else {
            value = scalar;
          }
        }
        values.push_back(value);
        named_values[std::string(descriptor.name)] = value;
        fields.push_back({
          {"index", index},
          {"name", std::string(descriptor.name)},
          {"json_type", std::string(descriptor.json_type)},
          {"gpu_encoding", encoding_name(descriptor.gpu_encoding)},
          {"raw_word", raw_word},
          {"value", std::move(value)},
        });
      }

      using sbs_adaptive_state::word_e;
      const auto scalar = [&](const word_e word) {
        return scalars[sbs_adaptive_state::index(word)];
      };
      const float cut_flags_value = scalar(word_e::cut_flags);
      const float analysis_flags_value = scalar(word_e::analysis_flags);
      if (
        cut_flags_value < 0.0f ||
        cut_flags_value >
          static_cast<float>(sbs_adaptive_state::known_cut_flag_mask) ||
        std::trunc(cut_flags_value) != cut_flags_value ||
        analysis_flags_value < 0.0f ||
        analysis_flags_value >
          static_cast<float>(sbs_adaptive_state::known_analysis_flag_mask) ||
        std::trunc(analysis_flags_value) != analysis_flags_value
      ) {
        return false;
      }
      const auto cut_flags = static_cast<std::uint32_t>(cut_flags_value);
      const auto analysis_flags =
        static_cast<std::uint32_t>(analysis_flags_value);
      nlohmann::json decoded_cut_flags = nlohmann::json::object();
      for (const auto &bit : sbs_adaptive_state::cut_flag_bits) {
        decoded_cut_flags[std::string(bit.name)] = (cut_flags & bit.mask) != 0u;
      }
      nlohmann::json decoded_analysis_flags = nlohmann::json::object();
      for (const auto &bit : sbs_adaptive_state::analysis_flag_bits) {
        decoded_analysis_flags[std::string(bit.name)] =
          (analysis_flags & bit.mask) != 0u;
      }
      const float effective_ratio = cfg.adaptive_pop ?
                                      std::max(
                                        scalar(word_e::adaptive_pop_ratio),
                                        1.0f
                                      ) :
                                      1.0f;
      const double absolute_effective_pop = cfg.pop_strength * effective_ratio;
      if (!std::isfinite(absolute_effective_pop)) {
        return false;
      }

      adaptive_summary = {
        {"schema", sbs_adaptive_state::schema_version},
        {"source", std::string(sbs_adaptive_state::source)},
        {"capture", std::string(sbs_adaptive_state::capture)},
        {"matched_frame_id", completed.matched_frame_id},
        {"depth_model", completed.depth_model},
        {"fields", std::move(fields)},
        {"values", std::move(values)},
        {"named_values", std::move(named_values)},
        {"decoded", {
                      {"cut_flags", {
                                      {"value", cut_flags},
                                      {"known_mask", sbs_adaptive_state::known_cut_flag_mask},
                                      {"bits", std::move(decoded_cut_flags)},
                                    }},
                      {"analysis_flags", {
                                           {"value", analysis_flags},
                                           {"known_mask", sbs_adaptive_state::known_analysis_flag_mask},
                                           {"bits", std::move(decoded_analysis_flags)},
                                         }},
                      {"hard_cut_pulse", scalar(word_e::hard_cut_pulse) > 0.5f},
                      {"hard_cut_count", words[sbs_adaptive_state::index(word_e::hard_cut_count)]},
                      {"external_cut_count", words[sbs_adaptive_state::index(word_e::external_cut_count)]},
                      {"empty_raw_count", words[sbs_adaptive_state::index(word_e::empty_raw_count)]},
                      {"collapsed_raw_count", words[sbs_adaptive_state::index(word_e::collapsed_raw_count)]},
                      {"absolute_effective_pop", absolute_effective_pop},
                      {"resolved_zero_anchor_shift_px", scalar(word_e::zero_anchor_shift_px)},
                    }},
        {"normalization", {
                            {"effective_lower", normalization.lower},
                            {"effective_upper", normalization.upper},
                            {"initialized", normalization.initialized > 0.5f},
                            {"initialized_value", normalization.initialized},
                            {"frame_state", normalization_frame_state_name(normalization.frame_state)},
                            {"frame_state_value", normalization.frame_state},
                          }},
      };
      return write_json(dir / "adaptive_state.json", adaptive_summary);
    }

    struct scene_controller_dump_result {
      bool authoritative_output_valid = false;
      std::uint32_t gpu_backend_generation = 0;
      std::uint32_t gpu_roi_generation = 0;
      nlohmann::json summary = nlohmann::json::object();
    };

    nlohmann::json finite_float_json(const float value) {
      return std::isfinite(value) ? nlohmann::json(value) :
                                    nlohmann::json(nullptr);
    }

    struct roi_transform_dump_result {
      bool unbound_zero = false;
      bool contract_valid = false;
      nlohmann::json summary = nlohmann::json::object();
    };

    bool dump_roi_transform(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv,
      const frame &completed,
      const std::uint32_t source_width,
      const std::uint32_t source_height,
      const std::filesystem::path &dir,
      roi_transform_dump_result &result
    ) {
      std::vector<std::uint8_t> bytes;
      if (
        !read_buffer(
          device,
          ctx,
          srv,
          sbs_scene_cache::roi_transform_word_count *
            sizeof(std::uint32_t),
          bytes
        )
      ) {
        return false;
      }
      sbs_scene_cache::roi_transform_words_t words {};
      std::memcpy(words.data(), bytes.data(), bytes.size());
      if (
        !write_bytes(
          dir / "roi_transform.words",
          words.data(),
          sizeof(words)
        )
      ) {
        return false;
      }

      const auto join_u64 = [](const std::uint32_t low,
                               const std::uint32_t high) {
        return static_cast<std::uint64_t>(low) |
               (static_cast<std::uint64_t>(high) << 32u);
      };
      const auto float4_json = [&](const std::size_t offset) {
        nlohmann::json value = nlohmann::json::array();
        for (std::size_t index = 0; index < 4u; ++index) {
          value.push_back(
            finite_float_json(
              std::bit_cast<float>(words[offset + index])
            )
          );
        }
        return value;
      };
      const auto uint4_json = [&](const std::size_t offset) {
        return nlohmann::json::array({
          words[offset + 0u],
          words[offset + 1u],
          words[offset + 2u],
          words[offset + 3u],
        });
      };

      result.unbound_zero =
        sbs_scene_cache::transform_is_unbound_zero(words);
      result.contract_valid =
        result.unbound_zero ||
        sbs_scene_cache::valid_roi_transform(
          words,
          completed.matched_frame_id,
          source_width,
          source_height,
          static_cast<std::uint32_t>(completed.model_width),
          static_cast<std::uint32_t>(completed.model_height)
        );

      nlohmann::json vectors = nlohmann::json::array();
      for (
        std::size_t vector = 0;
        vector < models::frame_roi_transform_vector_count;
        ++vector
      ) {
        vectors.push_back(uint4_json(vector * 4u));
      }
      const std::uint32_t flags = words[1u];
      const std::uint32_t fallback_reason =
        models::frame_roi_fallback_reason_from_words(words);
      result.summary = {
        {"schema", 1},
        {"gpu_contract_schema", words[0u]},
        {"gpu_vector_count", models::frame_roi_transform_vector_count},
        {"capture", "exact StructuredBuffer<uint4> sampled by the matched Host-SBS frame"},
        {"unbound_zero_legacy_transform", result.unbound_zero},
        {"contract_valid_for_matched_frame", result.contract_valid},
        {"matched_frame_id", completed.matched_frame_id},
        {"raw_vectors_uint4", std::move(vectors)},
        {"identity", {
                       {"flags", flags},
                       {"decoded_flags", {
                                           {"valid", (flags & (1u << 0u)) != 0u},
                                           {"full_frame", (flags & (1u << 1u)) != 0u},
                                           {"active_roi", (flags & (1u << 2u)) != 0u},
                                           {"reset_debt", (flags & (1u << 3u)) != 0u},
                                         }},
                       {"source_frame_id", join_u64(words[2u], words[3u])},
                       {"roi_generation", words[4u]},
                       {"backend_generation", words[5u]},
                       {"source_width", words[6u]},
                       {"source_height", words[7u]},
                       {"model_width", words[8u]},
                       {"model_height", words[9u]},
                       {"accepted_focus_pixel_count", words[10u]},
                       {"shape_request_id", words[11u]},
                       {"transform_version", join_u64(words[28u], words[29u])},
                       {"gpu_bank_identity", words[30u]},
                       {"fallback_reason", fallback_reason},
                       {
                         "fallback_reason_name",
                         models::frame_roi_fallback_reason_name_from_words(
                           words
                         )
                       },
                     }},
        {"geometry", {
                       {"focus_source_uv", float4_json(12u)},
                       {"crop_source_uv", float4_json(16u)},
                       {"accepted_model_bounds_half_open", uint4_json(20u)},
                       {"feather_source_uv_inward", float4_json(24u)},
                     }},
      };
      return write_json(dir / "roi_transform.json", result.summary);
    }

    nlohmann::json scalar_stats_json(
      const std::vector<float> &values
    ) {
      const scalar_stats stats = calculate_scalar_stats(values);
      nlohmann::json result {
        {"sample_count", values.size()},
        {"finite_count", stats.finite_count},
      };
      if (stats.finite_count != 0) {
        result["minimum"] = stats.minimum;
        result["maximum"] = stats.maximum;
        result["preview_low_p02"] = stats.preview_low;
        result["preview_high_p98"] = stats.preview_high;
      }
      return result;
    }

    template<std::size_t Count>
    nlohmann::json string_array_json(
      const std::array<std::string_view, Count> &names
    ) {
      nlohmann::json result = nlohmann::json::array();
      for (const auto name : names) {
        result.push_back(std::string(name));
      }
      return result;
    }

    template<std::size_t Count>
    nlohmann::json named_float_values_json(
      const std::vector<float> &values,
      const std::array<std::string_view, Count> &names
    ) {
      nlohmann::json result = nlohmann::json::object();
      if (values.size() != Count) {
        return result;
      }
      for (std::size_t index = 0; index < Count; ++index) {
        result[std::string(names[index])] =
          finite_float_json(values[index]);
      }
      return result;
    }

    template<typename Bits>
    nlohmann::json decode_named_bits(
      const std::uint32_t value,
      const Bits &bits
    ) {
      nlohmann::json decoded = nlohmann::json::object();
      for (const auto &bit : bits) {
        decoded[std::string(bit.name)] = (value & bit.mask) != 0u;
      }
      return decoded;
    }

    bool write_scene_tensor(
      const std::filesystem::path &dir,
      const std::string_view tensor_name,
      const std::string_view filename,
      const std::vector<float> &values,
      const std::size_t channel_count,
      const std::size_t samples_per_channel,
      nlohmann::json shape,
      nlohmann::json channels,
      const std::string_view stage,
      nlohmann::json &tensor_manifest
    ) {
      if (
        channel_count == 0 || samples_per_channel == 0 ||
        channel_count > SIZE_MAX / samples_per_channel ||
        values.size() != channel_count * samples_per_channel
      ) {
        return false;
      }
      if (
        !write_bytes(
          dir / std::string(filename),
          values.data(),
          values.size() * sizeof(float)
        )
      ) {
        return false;
      }

      nlohmann::json channel_stats = nlohmann::json::array();
      for (std::size_t channel = 0; channel < channel_count; ++channel) {
        const auto begin =
          values.begin() + static_cast<std::ptrdiff_t>(
                             channel * samples_per_channel
                           );
        std::vector<float> channel_values(
          begin,
          begin + static_cast<std::ptrdiff_t>(samples_per_channel)
        );
        auto stats = scalar_stats_json(channel_values);
        stats["index"] = channel;
        if (channels.is_array() && channel < channels.size()) {
          stats["name"] = channels[channel];
        }
        channel_stats.push_back(std::move(stats));
      }

      const std::string layout =
        shape.is_array() && shape.size() == 4u ? "NCHW" : "row-major";
      tensor_manifest[std::string(tensor_name)] = {
        {"file", std::string(filename)},
        {"dtype", "float32-le"},
        {"layout", layout},
        {"shape", std::move(shape)},
        {"channels", std::move(channels)},
        {"stage", std::string(stage)},
        {"statistics", scalar_stats_json(values)},
        {"channel_statistics", std::move(channel_stats)},
      };
      return true;
    }

    bool write_scene_rgb_preview(
      const std::filesystem::path &path,
      const std::vector<float> &values
    ) {
      constexpr std::size_t canvas =
        sbs_scene_controller::appearance_canvas_size;
      constexpr std::size_t plane = canvas * canvas;
      if (values.size() != 3u * plane) {
        return false;
      }
      std::vector<std::uint8_t> rgb(3u * plane);
      for (std::size_t index = 0; index < plane; ++index) {
        const float red = values[index];
        const float green = values[index + plane];
        const float blue = values[index + 2u * plane];
        if (
          !std::isfinite(red) || !std::isfinite(green) ||
          !std::isfinite(blue)
        ) {
          rgb[index * 3u + 0u] = 255u;
          rgb[index * 3u + 1u] = 0u;
          rgb[index * 3u + 2u] = 255u;
          continue;
        }
        // scene_rgb is already display-referred sRGB code, not linear light.
        rgb[index * 3u + 0u] = encode_unit(red);
        rgb[index * 3u + 1u] = encode_unit(green);
        rgb[index * 3u + 2u] = encode_unit(blue);
      }
      return write_png(
        path,
        static_cast<std::uint32_t>(canvas),
        static_cast<std::uint32_t>(canvas),
        rgb
      );
    }

    const char *scene_encoding_name(
      const sbs_scene_controller::gpu_encoding_e encoding
    ) {
      switch (encoding) {
        case sbs_scene_controller::gpu_encoding_e::float_value:
          return "float_value";
        case sbs_scene_controller::gpu_encoding_e::uint_bits:
          return "uint_bits";
        case sbs_scene_controller::gpu_encoding_e::uint_valued_float:
          return "uint_valued_float";
      }
      return "unknown";
    }

    bool dump_scene_controller(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      const frame &completed,
      const std::filesystem::path &dir,
      scene_controller_dump_result &result
    ) {
      using namespace sbs_scene_controller;

      const std::array<ID3D11ShaderResourceView *, 9> resources {{
        completed.scene_controller_scene_rgb,
        completed.scene_controller_analysis_grid,
        completed.scene_controller_dense_output,
        completed.scene_controller_global_output,
        completed.scene_controller_layout_history,
        completed.scene_controller_depth_history,
        completed.scene_controller_hidden_output,
        completed.scene_controller_meta,
        completed.scene_controller_rule_state,
      }};
      if (
        !completed.scene_controller_snapshot_available ||
        completed.scene_controller_frame_id != completed.matched_frame_id ||
        std::any_of(
          resources.begin(),
          resources.end(),
          [](const auto *resource) {
            return resource == nullptr;
          }
        )
      ) {
        return false;
      }

      constexpr std::size_t appearance_pixels =
        appearance_canvas_size * appearance_canvas_size;
      constexpr std::size_t analysis_pixels =
        analysis_canvas_size * analysis_canvas_size;
      constexpr std::size_t recurrent_pixels =
        recurrent_canvas_size * recurrent_canvas_size;
      std::vector<float> scene_rgb;
      std::vector<float> analysis_grid;
      std::vector<float> dense_output;
      std::vector<float> global_output;
      std::vector<float> layout_history;
      std::vector<float> depth_history;
      std::vector<float> hidden_output;
      std::vector<float> meta;
      std::vector<std::uint8_t> rule_state_bytes;
      if (
        !read_float_buffer(
          device,
          ctx,
          completed.scene_controller_scene_rgb,
          3u * appearance_pixels,
          scene_rgb
        ) ||
        !read_float_buffer(
          device,
          ctx,
          completed.scene_controller_analysis_grid,
          analysis_grid_channel_count * analysis_pixels,
          analysis_grid
        ) ||
        !read_float_buffer(
          device,
          ctx,
          completed.scene_controller_dense_output,
          dense_out_channel_count * analysis_pixels,
          dense_output
        ) ||
        !read_float_buffer(
          device,
          ctx,
          completed.scene_controller_global_output,
          global_out_word_count,
          global_output
        ) ||
        !read_float_buffer(
          device,
          ctx,
          completed.scene_controller_layout_history,
          layout_history_channel_count * analysis_pixels,
          layout_history
        ) ||
        !read_float_buffer(
          device,
          ctx,
          completed.scene_controller_depth_history,
          depth_history_channel_count * analysis_pixels,
          depth_history
        ) ||
        !read_float_buffer(
          device,
          ctx,
          completed.scene_controller_hidden_output,
          hidden_channel_count * recurrent_pixels,
          hidden_output
        ) ||
        !read_float_buffer(
          device,
          ctx,
          completed.scene_controller_meta,
          meta_word_count,
          meta
        ) ||
        !read_buffer(
          device,
          ctx,
          completed.scene_controller_rule_state,
          rule_state_word_count * sizeof(std::uint32_t),
          rule_state_bytes
        )
      ) {
        return false;
      }

      std::array<std::uint32_t, rule_state_word_count> rule_words {};
      std::memcpy(
        rule_words.data(),
        rule_state_bytes.data(),
        rule_state_bytes.size()
      );
      std::array<float, rule_state_word_count> rule_scalars {};
      std::array<bool, rule_state_word_count> rule_scalar_valid {};
      bool encodings_valid = true;
      nlohmann::json rule_fields = nlohmann::json::array();
      nlohmann::json rule_values = nlohmann::json::array();
      nlohmann::json rule_named_values = nlohmann::json::object();
      for (const auto &descriptor : rule_state_fields) {
        const std::size_t word_index = index(descriptor.word);
        const std::uint32_t raw_word = rule_words[word_index];
        bool valid_encoding = true;
        nlohmann::json value;
        if (descriptor.gpu_encoding == gpu_encoding_e::uint_bits) {
          value = raw_word;
        } else {
          const float scalar = std::bit_cast<float>(raw_word);
          rule_scalars[word_index] = scalar;
          rule_scalar_valid[word_index] = std::isfinite(scalar);
          valid_encoding = rule_scalar_valid[word_index];
          if (
            valid_encoding &&
            descriptor.gpu_encoding ==
              gpu_encoding_e::uint_valued_float
          ) {
            valid_encoding =
              scalar >= 0.0f &&
              static_cast<double>(scalar) <=
                static_cast<double>(
                  std::numeric_limits<std::uint32_t>::max()
                ) &&
              std::trunc(scalar) == scalar;
          }
          if (!valid_encoding) {
            value = nullptr;
          } else if (
            descriptor.gpu_encoding ==
            gpu_encoding_e::uint_valued_float
          ) {
            value = static_cast<std::uint32_t>(scalar);
          } else {
            value = scalar;
          }
        }
        encodings_valid = encodings_valid && valid_encoding;
        rule_values.push_back(value);
        rule_named_values[std::string(descriptor.name)] = value;
        rule_fields.push_back({
          {"index", word_index},
          {"name", std::string(descriptor.name)},
          {"json_type", std::string(descriptor.json_type)},
          {"gpu_encoding", scene_encoding_name(descriptor.gpu_encoding)},
          {"required_zero", descriptor.required_zero},
          {"valid_encoding", valid_encoding},
          {"raw_word", raw_word},
          {"value", std::move(value)},
        });
      }

      const auto state_scalar = [&](const rule_state_word_e word) {
        const auto word_index = index(word);
        return rule_scalar_valid[word_index] ?
                 rule_scalars[word_index] :
                 std::numeric_limits<float>::quiet_NaN();
      };
      const auto state_uint = [&](const rule_state_word_e word) {
        return rule_words[index(word)];
      };
      const auto float_flag = [](const float value) {
        return std::isfinite(value) && value > 0.5f;
      };

      const float global_valid_value =
        global_output[static_cast<std::size_t>(
          global_out_word_e::backend_output_valid
        )];
      const bool global_output_valid = float_flag(global_valid_value);
      const bool rule_output_valid =
        float_flag(state_scalar(rule_state_word_e::output_valid));
      const float rule_schema =
        state_scalar(rule_state_word_e::schema_version);
      const bool schema_valid =
        std::isfinite(rule_schema) &&
        rule_schema == static_cast<float>(schema_version);
      result.gpu_backend_generation =
        state_uint(rule_state_word_e::backend_generation);
      result.gpu_roi_generation =
        state_uint(rule_state_word_e::roi_generation);
      const bool backend_generation_valid =
        result.gpu_backend_generation ==
        completed.scene_controller_backend_generation;

      bool reserved_zero = true;
      for (
        std::size_t word =
          static_cast<std::size_t>(global_out_word_e::reserved_35);
        word <= static_cast<std::size_t>(
                  global_out_word_e::reserved_40
                );
        ++word
      ) {
        reserved_zero =
          reserved_zero && std::isfinite(global_output[word]) &&
          global_output[word] == 0.0f;
      }
      for (
        std::size_t word =
          static_cast<std::size_t>(meta_word_e::reserved_28);
        word <= static_cast<std::size_t>(meta_word_e::reserved_31);
        ++word
      ) {
        reserved_zero =
          reserved_zero && std::isfinite(meta[word]) &&
          meta[word] == 0.0f;
      }
      for (const auto &descriptor : rule_state_fields) {
        if (descriptor.required_zero) {
          const auto word = index(descriptor.word);
          reserved_zero =
            reserved_zero && rule_scalar_valid[word] &&
            rule_scalars[word] == 0.0f;
        }
      }

      result.authoritative_output_valid =
        global_output_valid && rule_output_valid && schema_valid &&
        backend_generation_valid && encodings_valid && reserved_zero;

      const std::uint32_t state_flags =
        state_uint(rule_state_word_e::state_flags);
      const std::uint32_t reset_flags =
        state_uint(rule_state_word_e::reset_flags);
      const std::uint32_t promotion_flags =
        state_uint(rule_state_word_e::promotion_flags);
      const std::uint32_t history_flags =
        state_uint(rule_state_word_e::history_flags);
      const std::uint32_t diagnostic_flags =
        state_uint(rule_state_word_e::diagnostic_flags);
      const nlohmann::json validation {
        {"producer_snapshot_available",
         completed.scene_controller_snapshot_available},
        {"frame_identity_valid",
         completed.scene_controller_frame_id ==
           completed.matched_frame_id},
        {"global_output_valid", global_output_valid},
        {"rule_state_output_valid", rule_output_valid},
        {"schema_valid", schema_valid},
        {"backend_generation_valid", backend_generation_valid},
        {"encodings_valid", encodings_valid},
        {"reserved_words_zero", reserved_zero},
        {"authoritative_output_valid",
         result.authoritative_output_valid},
      };

      nlohmann::json controller_state {
        {"schema", schema_version},
        {"rule_revision", std::string(rule_revision)},
        {"ordered_abi_hash", std::string(ordered_abi_hash)},
        {"capture", "one matched, completed Host-SBS scene-controller update"},
        {"sidecar", {
                      {"source_frame_id",
                       completed.scene_controller_frame_id},
                      {"completed_depth_frame_id",
                       completed.matched_frame_id},
                      {"roi_frame_id", completed.matched_frame_id},
                      {"roi_generation_gpu_state",
                       result.gpu_roi_generation},
                      {"backend_generation_reported_by_producer",
                       completed.scene_controller_backend_generation},
                      {"backend_generation_gpu_state",
                       result.gpu_backend_generation},
                      {"shadow", completed.scene_controller_shadow},
                    }},
        {"validation", validation},
        {"global_out", {
                         {"values", global_output},
                         {"named_values",
                          named_float_values_json(
                            global_output,
                            global_out_names
                          )},
                       }},
        {"meta", {
                   {"values", meta},
                   {"named_values",
                    named_float_values_json(meta, meta_names)},
                 }},
        {"rule_state", {
                         {"stage", "post-resolve committed rule state"},
                         {"fields", std::move(rule_fields)},
                         {"values", std::move(rule_values)},
                         {"named_values", std::move(rule_named_values)},
                         {"decoded_flags", {
                                             {"state_flags", {
                                                {"value", state_flags},
                                                {"bits", decode_named_bits(
                                                  state_flags,
                                                  state_flags_bits
                                                )},
                                              }},
                                             {"reset_flags", {
                                                {"value", reset_flags},
                                                {"bits", decode_named_bits(
                                                  reset_flags,
                                                  reset_flags_bits
                                                )},
                                              }},
                                             {"promotion_flags", {
                                                {"value", promotion_flags},
                                                {"bits", decode_named_bits(
                                                  promotion_flags,
                                                  promotion_flags_bits
                                                )},
                                              }},
                                             {"history_flags", {
                                                {"value", history_flags},
                                                {"bits", decode_named_bits(
                                                  history_flags,
                                                  history_flags_bits
                                                )},
                                              }},
                                             {"diagnostic_flags", {
                                                {"value", diagnostic_flags},
                                                {"bits", decode_named_bits(
                                                  diagnostic_flags,
                                                  diagnostic_flags_bits
                                                )},
                                              }},
                                           }},
                       }},
      };

      const auto controller_dir = dir / "scene_controller";
      std::error_code error;
      if (
        !std::filesystem::create_directories(controller_dir, error) &&
        error
      ) {
        return false;
      }
      const auto cleanup = [&]() {
        std::error_code cleanup_error;
        std::filesystem::remove_all(controller_dir, cleanup_error);
      };

      nlohmann::json tensors = nlohmann::json::object();
      constexpr std::array<std::string_view, 3> rgb_names {{
        "red",
        "green",
        "blue",
      }};
      const bool files_written =
        write_scene_tensor(
          controller_dir,
          "scene_rgb",
          "scene_rgb.f32",
          scene_rgb,
          3u,
          appearance_pixels,
          {1u, 3u, appearance_canvas_size, appearance_canvas_size},
          string_array_json(rgb_names),
          "matched display-referred, exact-area scene input",
          tensors
        ) &&
        write_scene_rgb_preview(
          controller_dir / "scene_rgb.png",
          scene_rgb
        ) &&
        write_scene_tensor(
          controller_dir,
          "analysis_grid",
          "analysis_grid.f32",
          analysis_grid,
          analysis_grid_channel_count,
          analysis_pixels,
          {
            1u,
            analysis_grid_channel_count,
            analysis_canvas_size,
            analysis_canvas_size,
          },
          string_array_json(analysis_grid_names),
          "matched deterministic scene-analysis input",
          tensors
        ) &&
        write_scene_tensor(
          controller_dir,
          "dense_out",
          "dense_out.f32",
          dense_output,
          dense_out_channel_count,
          analysis_pixels,
          {
            1u,
            dense_out_channel_count,
            analysis_canvas_size,
            analysis_canvas_size,
          },
          string_array_json(dense_out_names),
          "matched dense rule evidence output",
          tensors
        ) &&
        write_scene_tensor(
          controller_dir,
          "global_out",
          "global_out.f32",
          global_output,
          global_out_word_count,
          1u,
          {1u, global_out_word_count},
          string_array_json(global_out_names),
          "matched global rule evidence output",
          tensors
        ) &&
        write_scene_tensor(
          controller_dir,
          "layout_history_post",
          "layout_history_post.f32",
          layout_history,
          layout_history_channel_count,
          analysis_pixels,
          {
            1u,
            layout_history_channel_count,
            analysis_canvas_size,
            analysis_canvas_size,
          },
          string_array_json(layout_history_names),
          "post-resolve promoted layout history; not the pre-resolve input bank",
          tensors
        ) &&
        write_scene_tensor(
          controller_dir,
          "depth_history_post",
          "depth_history_post.f32",
          depth_history,
          depth_history_channel_count,
          analysis_pixels,
          {
            1u,
            depth_history_channel_count,
            analysis_canvas_size,
            analysis_canvas_size,
          },
          string_array_json(depth_history_names),
          "post-resolve promoted depth history; not the pre-resolve input bank",
          tensors
        ) &&
        write_scene_tensor(
          controller_dir,
          "hidden_out",
          "hidden_out.f32",
          hidden_output,
          hidden_channel_count,
          recurrent_pixels,
          {
            1u,
            hidden_channel_count,
            recurrent_canvas_size,
            recurrent_canvas_size,
          },
          nlohmann::json::array(),
          "matched recurrent output; rules_v1 defines this tensor as zero",
          tensors
        ) &&
        write_scene_tensor(
          controller_dir,
          "meta",
          "meta.f32",
          meta,
          meta_word_count,
          1u,
          {1u, meta_word_count},
          string_array_json(meta_names),
          "matched controller metadata",
          tensors
        ) &&
        write_bytes(
          controller_dir / "rule_state.words",
          rule_state_bytes.data(),
          rule_state_bytes.size()
        ) &&
        write_json(
          controller_dir / "tensor_manifest.json",
          {
            {"schema", schema_version},
            {"rule_revision", std::string(rule_revision)},
            {"ordered_abi_hash", std::string(ordered_abi_hash)},
            {"capture", "matched scene-controller tensors"},
            {"endianness", "little"},
            {"rule_state", {
                              {"file", "rule_state.words"},
                              {"stage", "post-resolve committed rule state"},
                              {"word_count", rule_state_word_count},
                              {"word_bytes", sizeof(std::uint32_t)},
                              {"field_contract", "state.json fields"},
                            }},
            {"tensors", tensors},
          }
        ) &&
        write_json(controller_dir / "state.json", controller_state);
      if (!files_written) {
        cleanup();
        return false;
      }

      result.summary = {
        {"status", "captured"},
        {"schema", schema_version},
        {"rule_revision", std::string(rule_revision)},
        {"ordered_abi_hash", std::string(ordered_abi_hash)},
        {"frame_id", completed.scene_controller_frame_id},
        {"shadow", completed.scene_controller_shadow},
        {"gpu_backend_generation", result.gpu_backend_generation},
        {"gpu_roi_generation", result.gpu_roi_generation},
        {"authoritative_output_valid",
         result.authoritative_output_valid},
        {"state_kind",
         finite_float_json(
           state_scalar(rule_state_word_e::state_kind)
         )},
        {"committed_roi", {
                            finite_float_json(
                              state_scalar(
                                rule_state_word_e::committed_roi_x0
                              )
                            ),
                            finite_float_json(
                              state_scalar(
                                rule_state_word_e::committed_roi_y0
                              )
                            ),
                            finite_float_json(
                              state_scalar(
                                rule_state_word_e::committed_roi_x1
                              )
                            ),
                            finite_float_json(
                              state_scalar(
                                rule_state_word_e::committed_roi_y1
                              )
                            ),
                          }},
        {"layout_decision",
         finite_float_json(
           state_scalar(rule_state_word_e::layout_decision)
         )},
        {"event_decision",
         finite_float_json(
           state_scalar(rule_state_word_e::event_decision)
         )},
        {"pop_strength",
         finite_float_json(
           state_scalar(rule_state_word_e::pop_strength)
         )},
        {"zero_plane_decision",
         finite_float_json(
           state_scalar(rule_state_word_e::zero_plane_decision)
         )},
        {"rejection_reason",
         finite_float_json(
           state_scalar(rule_state_word_e::rejection_reason)
         )},
      };
      return true;
    }

    bool dump_warp_map(
      const texture_snapshot &mapping,
      const std::uint32_t source_width,
      const std::uint32_t source_height,
      const std::filesystem::path &dir,
      warp_map_dump_stats &stats
    ) {
      std::vector<float> map;
      if (
        !texture_float_values(mapping, map) ||
        mapping.desc.Width < 2u || (mapping.desc.Width & 1u) != 0u ||
        source_width == 0u || source_height == 0u
      ) {
        return false;
      }
      stats.width = mapping.desc.Width;
      stats.height = mapping.desc.Height;
      stats.eye_width = mapping.desc.Width / 2u;
      stats.eye_height = mapping.desc.Height;
      const float source_aspect =
        static_cast<float>(source_width) / static_cast<float>(source_height);
      const float eye_aspect =
        static_cast<float>(stats.eye_width) / static_cast<float>(stats.eye_height);
      stats.content_scale_x =
        eye_aspect > source_aspect ? source_aspect / eye_aspect : 1.0f;
      stats.content_scale_y =
        eye_aspect < source_aspect ? eye_aspect / source_aspect : 1.0f;
      static_cast<scalar_stats &>(stats) = calculate_scalar_stats(map);

      std::vector<float> displacement(map.size(), 0.0f);
      std::vector<float> finite_absolute_displacement;
      finite_absolute_displacement.reserve(map.size());
      std::vector<std::uint8_t> content_valid(map.size(), 0u);
      const float content_lo_x = 0.5f * (1.0f - stats.content_scale_x);
      const float content_hi_x = content_lo_x + stats.content_scale_x;
      const float content_lo_y = 0.5f * (1.0f - stats.content_scale_y);
      const float content_hi_y = content_lo_y + stats.content_scale_y;
      for (std::uint32_t y = 0; y < stats.height; ++y) {
        const float output_v =
          (static_cast<float>(y) + 0.5f) / static_cast<float>(stats.height);
        for (std::uint32_t x = 0; x < stats.width; ++x) {
          const std::size_t index =
            static_cast<std::size_t>(y) * stats.width + x;
          const std::uint32_t eye_x = x % stats.eye_width;
          const float output_u =
            (static_cast<float>(eye_x) + 0.5f) /
            static_cast<float>(stats.eye_width);
          if (
            output_u < content_lo_x || output_u > content_hi_x ||
            output_v < content_lo_y || output_v > content_hi_y ||
            !std::isfinite(map[index])
          ) {
            continue;
          }
          const float unwarped_source_u =
            (output_u - content_lo_x) / stats.content_scale_x;
          displacement[index] =
            (map[index] - unwarped_source_u) *
            stats.content_scale_x *
            static_cast<float>(stats.eye_width);
          if (std::isfinite(displacement[index])) {
            content_valid[index] = 1u;
            finite_absolute_displacement.push_back(std::fabs(displacement[index]));
          }
        }
      }
      if (finite_absolute_displacement.empty()) {
        return false;
      }
      std::sort(
        finite_absolute_displacement.begin(),
        finite_absolute_displacement.end()
      );
      const std::size_t p98_index = static_cast<std::size_t>(
        std::lround(
          0.98 * static_cast<double>(finite_absolute_displacement.size() - 1u)
        )
      );
      stats.displacement_preview_abs_px =
        finite_absolute_displacement[std::min(
          p98_index,
          finite_absolute_displacement.size() - 1u
        )];
      if (!(stats.displacement_preview_abs_px > 1.0e-6f)) {
        stats.displacement_preview_abs_px =
          finite_absolute_displacement.back();
      }
      if (!(stats.displacement_preview_abs_px > 1.0e-6f)) {
        stats.displacement_preview_abs_px = 1.0f;
      }

      std::vector<std::uint8_t> heat(map.size() * 3u, 0u);
      for (std::size_t index = 0; index < map.size(); ++index) {
        if (!content_valid[index]) {
          if (!std::isfinite(map[index])) {
            heat[index * 3u + 0u] = 255;
            heat[index * 3u + 1u] = 0;
            heat[index * 3u + 2u] = 255;
          }
          continue;
        }
        const float normalized = std::clamp(
          0.5f +
            0.5f * displacement[index] / stats.displacement_preview_abs_px,
          0.0f,
          1.0f
        );
        colormap_jet(
          normalized,
          heat[index * 3u + 0u],
          heat[index * 3u + 1u],
          heat[index * 3u + 2u]
        );
      }
      const nlohmann::json shape {
        {"schema", 1},
        {"width", stats.width},
        {"height", stats.height},
        {"eye_width", stats.eye_width},
        {"eye_height", stats.eye_height},
        {"source_width", source_width},
        {"source_height", source_height},
        {"content_scale_x", stats.content_scale_x},
        {"content_scale_y", stats.content_scale_y},
        {"dtype", "float32-le"},
        {"layout", "row-major"},
        {"channels", {"raw_reproject_source_u_normalized"}},
        {"validity", {
                       {"content", "derive from content_scale_x/content_scale_y and packed output coordinate"},
                       {"forward_coverage", "warp_mask.png red == 0 inside content"},
                     }},
        {"live_sample_source_u_normalized", "clamp(raw_reproject_source_u_normalized, 0, 1)"},
        {"derived_inverse_displacement_output_eye_px", "(raw_reproject_source_u_normalized - aspect_fitted_unwarped_source_u) * content_scale_x * eye_width"},
        {"derived_signed_binocular_disparity_px", "invert both eye maps at common source-U samples; x_right - x_left"},
        {"displacement_preview", {
                                   {"file", "warp_displacement_heat.png"},
                                   {"range_px", {
                                                  -stats.displacement_preview_abs_px,
                                                  stats.displacement_preview_abs_px,
                                                }},
                                   {"normalization", "symmetric finite-content p98 absolute displacement"},
                                   {"negative", "blue"},
                                   {"zero", "green"},
                                   {"positive", "red"},
                                   {"bars", "black"},
                                   {"nonfinite", "magenta"},
                                 }},
      };
      return write_bytes(
               dir / "warp_map.f32",
               map.data(),
               map.size() * sizeof(float)
             ) &&
             write_png(
               dir / "warp_displacement_heat.png",
               stats.width,
               stats.height,
               heat
             ) &&
             write_json(dir / "warp_map_shape.json", shape);
    }

    nlohmann::json config_json(
      const config::video_t::sbs_t &cfg,
      const std::string &model_name
    ) {
      return {
        {"profile", cfg.profile},
        {"pop_strength", cfg.pop_strength},
        {"adaptive_pop", cfg.adaptive_pop},
        {"adaptive_pop_max", cfg.adaptive_pop_max},
        {"ema", cfg.ema},
        {"ema_edge_change", cfg.ema_edge_change},
        {"ema_edge_gradient", cfg.ema_edge_gradient},
        {"ema_edge_strength", cfg.ema_edge_strength},
        {"depth_short_side", cfg.depth_short_side},
        {"depth_max_aspect", cfg.depth_max_aspect},
        {"minmax_ema", cfg.minmax_ema},
        {"subject_recenter", cfg.subject_recenter},
        {"subject_stretch", cfg.subject_stretch},
        {"zero_plane", cfg.zero_plane},
        {"depth_model", model_name},
        {"configured_depth_model", cfg.depth_model},
        {"depth_model_url", cfg.depth_model_url},
        {"max_encode_width", cfg.max_encode_width},
        {"cuda_graph", cfg.cuda_graph},
        {"scene_controller",
         std::string(config::to_string(cfg.scene_controller))},
      };
    }

    nlohmann::json artifact_description(
      const bool available,
      const bool required,
      const std::string_view stage,
      const std::string_view description
    ) {
      return {
        {"available", available},
        {"required", required},
        {"stage", std::string(stage)},
        {"description", std::string(description)},
      };
    }

    std::string timestamp_string() {
      char text[32] = "unknown";
      const std::time_t now = std::time(nullptr);
      std::tm local {};
      if (localtime_s(&local, &now) == 0) {
        std::strftime(text, sizeof(text), "%Y%m%d_%H%M%S", &local);
      }
      return text;
    }

    struct output_paths {
      std::filesystem::path temporary;
      std::filesystem::path final;
    };

    bool make_output_paths(
      const std::filesystem::path &root,
      output_paths &paths,
      std::error_code &error
    ) {
      static std::atomic<std::uint64_t> sequence {0};
      const auto ticks = static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
      );
      for (unsigned attempt = 0; attempt < 128u; ++attempt) {
        const std::uint64_t serial =
          sequence.fetch_add(1u, std::memory_order_relaxed);
        std::ostringstream suffix;
        suffix << timestamp_string() << '_' << GetCurrentProcessId() << '_'
               << std::hex << ticks << '_' << serial;
        paths.final = root / ("dump_" + suffix.str());
        paths.temporary = root / (".partial_" + suffix.str());
        error.clear();
        if (
          !std::filesystem::exists(paths.final, error) && !error &&
          std::filesystem::create_directory(paths.temporary, error) && !error
        ) {
          return true;
        }
        if (error) {
          return false;
        }
      }
      return false;
    }

  }  // namespace

  dumper::dumper() {
    if (const char *override_dir = std::getenv("APOLLO_SBS_DUMP"); override_dir && *override_dir) {
      dir_ = override_dir;
      file_trigger_enabled_ = config::sunshine.diagnostics_enabled;
    } else if (!config::sunshine.log_file.empty()) {
      dir_ =
        std::filesystem::path(config::sunshine.log_file).parent_path() /
        "sbs_dump";
    } else {
      dir_ = "sbs_dump";
    }
  }

  void dumper::set_button_request(std::shared_ptr<std::atomic<bool>> request) {
    button_request_ = std::move(request);
  }

  void dumper::cancel_pending_request() noexcept {
    const bool remove_file_trigger =
      file_trigger_enabled_ || file_trigger_pending_;
    snapshot_armed_for_dump_ = false;
    prepared_normalization_valid_ = false;
    retry_backoff_frames_ = 0;
    if (auto *button = button_request_.get()) {
      button->store(false, std::memory_order_relaxed);
    }
    file_trigger_pending_ = false;
    file_trigger_enabled_ = false;
    if (!remove_file_trigger) {
      return;
    }
    try {
      std::error_code error;
      std::filesystem::remove(dir_ / "dump.trigger", error);
    } catch (...) {
      // Permanent estimator failure already makes this request impossible. Cleanup is best-effort
      // and must never turn the encode-thread failure handling into another exception path.
    }
  }

  bool dumper::snapshot_requested() {
    snapshot_armed_for_dump_ = false;
    prepared_normalization_valid_ = false;
    if (retry_backoff_frames_ != 0u) {
      --retry_backoff_frames_;
      return false;
    }
    const auto *button = button_request_.get();
    if (button && button->load(std::memory_order_relaxed)) {
      snapshot_armed_for_dump_ = true;
      return true;
    }
    if (!file_trigger_enabled_) {
      return false;
    }
    if (file_trigger_pending_) {
      snapshot_armed_for_dump_ = true;
      return true;
    }
    if ((poll_counter_++ & 63u) != 0u) {
      return false;
    }
    std::error_code error;
    file_trigger_pending_ =
      std::filesystem::exists(dir_ / "dump.trigger", error) && !error;
    snapshot_armed_for_dump_ = file_trigger_pending_;
    return file_trigger_pending_;
  }

  bool dumper::preflight_requested_frame(
    ID3D11Device *device,
    ID3D11DeviceContext *ctx,
    ID3D11ShaderResourceView *depth_frame_state,
    const std::uint64_t matched_frame_id
  ) noexcept {
    auto *button = button_request_.get();
    const bool requested =
      (button && button->load(std::memory_order_relaxed)) ||
      file_trigger_pending_;
    if (!requested || !snapshot_armed_for_dump_) {
      return false;
    }

    normalization_state normalization;
    depth_dumpability dumpability = depth_dumpability::unreadable;
    try {
      dumpability =
        read_normalization_state(device, ctx, depth_frame_state, normalization);
    } catch (const std::exception &error) {
      try {
        BOOST_LOG(warning)
          << "SBS debug dump normalization preflight failed transiently: "
          << error.what();
      } catch (...) {
      }
    } catch (...) {
      try {
        BOOST_LOG(warning)
          << "SBS debug dump normalization preflight failed transiently."sv;
      } catch (...) {
      }
    }

    if (dumpability == depth_dumpability::valid) {
      prepared_frame_id_ = matched_frame_id;
      prepared_normalization_ = {
        normalization.lower,
        normalization.upper,
        normalization.initialized,
        normalization.frame_state,
      };
      prepared_normalization_valid_ = true;
      return true;
    }

    // Do not run full-resolution mapping/coverage for an invalid or temporarily unreadable depth
    // completion. Retain the request, but avoid retrying diagnostic allocations every render frame.
    snapshot_armed_for_dump_ = false;
    prepared_normalization_valid_ = false;
    retry_backoff_frames_ = retry_backoff_frames;
    if (dumpability == depth_dumpability::unreadable) {
      try {
        BOOST_LOG(warning)
          << "SBS debug dump normalization state is temporarily unreadable; request retained for a rate-limited retry."sv;
      } catch (...) {
      }
    }
    return false;
  }

  bool dumper::maybe_dump(
    ID3D11Device *device,
    ID3D11DeviceContext *ctx,
    const frame &completed,
    const config::video_t::sbs_t &cfg
  ) {
    const bool hdr =
      completed.color_space == models::input_color_space::scrgb_hdr;
    auto *button = button_request_.get();
    const bool by_button =
      button && button->load(std::memory_order_relaxed);
    const bool by_file = file_trigger_pending_;
    if ((!by_button && !by_file) || !snapshot_armed_for_dump_) {
      return false;
    }
    snapshot_armed_for_dump_ = false;

    if (
      !device || !ctx || !completed.source || !completed.model_input ||
      !completed.raw_depth || !completed.depth || !completed.warp_depth ||
      !completed.adaptive_state || !completed.depth_frame_state ||
      !completed.depth_roi_transform || !completed.sbs ||
      completed.model_width <= 0 ||
      completed.model_height <= 0 || completed.raw_width <= 0 ||
      completed.raw_height <= 0
    ) {
      return false;
    }

    std::filesystem::path trigger;
    std::error_code error;
    output_paths paths;
    bool success = false;
    try {
      trigger = dir_ / "dump.trigger";
      normalization_state normalization;
      depth_dumpability dumpability = depth_dumpability::unreadable;
      if (
        prepared_normalization_valid_ &&
        prepared_frame_id_ == completed.matched_frame_id
      ) {
        normalization = {
          prepared_normalization_[0],
          prepared_normalization_[1],
          prepared_normalization_[2],
          prepared_normalization_[3],
        };
        prepared_normalization_valid_ = false;
        dumpability = depth_dumpability::valid;
      } else {
        dumpability = read_normalization_state(
          device,
          ctx,
          completed.depth_frame_state,
          normalization
        );
      }
      if (dumpability == depth_dumpability::invalid) {
        retry_backoff_frames_ = retry_backoff_frames;
        return false;
      }
      if (dumpability == depth_dumpability::unreadable) {
        retry_backoff_frames_ = retry_backoff_frames;
        BOOST_LOG(warning)
          << "SBS debug dump validity state is temporarily unreadable; request retained for a rate-limited retry."sv;
        return false;
      }

      if (
        !std::filesystem::create_directories(dir_, error) && error
      ) {
        retry_backoff_frames_ = retry_backoff_frames;
        BOOST_LOG(warning) << "SBS debug dump: cannot create root "sv
                           << dir_.string() << ": " << error.message();
        return false;
      }
      if (!make_output_paths(dir_, paths, error)) {
        retry_backoff_frames_ = retry_backoff_frames;
        BOOST_LOG(warning) << "SBS debug dump: cannot reserve a unique output folder in "sv
                           << dir_.string()
                           << (error ? ": " + error.message() : "."s);
        return false;
      }

      do {
        texture_snapshot source;
        texture_snapshot depth;
        texture_snapshot warp_depth;
        texture_snapshot sbs;
        if (
          !read_texture(device, ctx, completed.source, source) ||
          !read_texture(device, ctx, completed.depth, depth) ||
          !read_texture(device, ctx, completed.warp_depth, warp_depth) ||
          !read_texture(device, ctx, completed.sbs, sbs) ||
          !write_color_preview(
            paths.temporary / "source.png",
            source,
            completed.color_space
          ) ||
          !dump_model_input(
            device,
            ctx,
            completed.model_input,
            completed.model_width,
            completed.model_height,
            paths.temporary
          )
        ) {
          break;
        }

        raw_depth_dump_stats raw_stats;
        if (
          !dump_raw_depth(
            device,
            ctx,
            completed.raw_depth,
            completed.raw_width,
            completed.raw_height,
            paths.temporary,
            raw_stats
          ) ||
          !write_float_texture_artifacts(
            paths.temporary / "depth.f32",
            paths.temporary / "depth_shape.json",
            depth,
            "normalized and temporally filtered estimator depth before warp prefilter"
          ) ||
          !write_scalar_previews(
            paths.temporary / "depth.png",
            paths.temporary / "depth_heat.png",
            depth
          ) ||
          !write_float_texture_artifacts(
            paths.temporary / "warp_depth.f32",
            paths.temporary / "warp_depth_shape.json",
            warp_depth,
            "exact depth texture sampled by Host-SBS reprojection"
          ) ||
          !write_scalar_previews(
            paths.temporary / "warp_depth.png",
            paths.temporary / "warp_depth_heat.png",
            warp_depth
          ) ||
          !write_color_preview(
            paths.temporary / "sbs.png",
            sbs,
            completed.color_space
          )
        ) {
          break;
        }

        nlohmann::json adaptive;
        if (
          !dump_adaptive_state(
            device,
            ctx,
            completed.adaptive_state,
            normalization,
            completed,
            cfg,
            paths.temporary,
            adaptive
          )
        ) {
          break;
        }

        roi_transform_dump_result roi_transform;
        if (
          !dump_roi_transform(
            device,
            ctx,
            completed.depth_roi_transform,
            completed,
            source.desc.Width,
            source.desc.Height,
            paths.temporary,
            roi_transform
          )
        ) {
          break;
        }

        const bool scene_controller_configured =
          cfg.scene_controller != config::sbs_scene_controller_e::off;
        scene_controller_dump_result scene_controller;
        bool scene_controller_available = false;
        if (scene_controller_configured) {
          try {
            scene_controller_available = dump_scene_controller(
              device,
              ctx,
              completed,
              paths.temporary,
              scene_controller
            );
          } catch (const std::exception &exception) {
            std::error_code cleanup_error;
            std::filesystem::remove_all(
              paths.temporary / "scene_controller",
              cleanup_error
            );
            BOOST_LOG(warning)
              << "SBS debug dump: optional scene-controller package threw: "
              << exception.what();
          } catch (...) {
            std::error_code cleanup_error;
            std::filesystem::remove_all(
              paths.temporary / "scene_controller",
              cleanup_error
            );
            BOOST_LOG(warning)
              << "SBS debug dump: optional scene-controller package threw an "
                 "unknown exception.";
          }
          if (!scene_controller_available) {
            scene_controller.summary = {
              {"status", "unavailable"},
              {"configured_backend",
               std::string(config::to_string(cfg.scene_controller))},
              {"reason",
               completed.scene_controller_snapshot_available ?
                 "matched controller resources could not be read or written" :
                 "no matched completed controller output"},
            };
            BOOST_LOG(warning)
              << "SBS debug dump: optional scene-controller package is unavailable; "
                 "publishing the core dump without it.";
          }
        } else {
          scene_controller.summary = {
            {"status", "disabled"},
            {"configured_backend",
             std::string(config::to_string(cfg.scene_controller))},
          };
        }

        const bool warp_map_available = completed.warp_map != nullptr;
        const bool warp_mask_available = completed.warp_mask != nullptr;
        warp_map_dump_stats warp_map_stats;
        texture_snapshot warp_map;
        if (
          warp_map_available &&
          (!read_texture(device, ctx, completed.warp_map, warp_map) ||
           !dump_warp_map(
             warp_map,
             source.desc.Width,
             source.desc.Height,
             paths.temporary,
             warp_map_stats
           ))
        ) {
          break;
        }
        texture_snapshot warp_mask;
        if (
          warp_mask_available &&
          (!read_texture(device, ctx, completed.warp_mask, warp_mask) ||
           !write_color_preview(
             paths.temporary / "warp_mask.png",
             warp_mask,
             models::input_color_space::srgb
           ))
        ) {
          break;
        }

        if (
          sbs.desc.Width < 2u || (sbs.desc.Width & 1u) != 0u ||
          source.desc.Width == 0u || source.desc.Height == 0u ||
          sbs.desc.Height == 0u
        ) {
          break;
        }
        const std::uint32_t eye_width = sbs.desc.Width / 2u;
        const std::uint32_t eye_height = sbs.desc.Height;
        const float source_aspect =
          static_cast<float>(source.desc.Width) /
          static_cast<float>(source.desc.Height);
        const float eye_aspect =
          static_cast<float>(eye_width) / static_cast<float>(eye_height);
        const float content_scale_x =
          eye_aspect > source_aspect ? source_aspect / eye_aspect : 1.0f;
        const float content_scale_y =
          eye_aspect < source_aspect ? eye_aspect / source_aspect : 1.0f;

        nlohmann::json artifacts = nlohmann::json::object();
        artifacts["source.png"] = artifact_description(
          true,
          true,
          "captured color supplied to Host SBS",
          "Preview decoded according to the matched frame's declared transfer: direct sRGB, linear-SDR OETF, or HDR tone map plus OETF."
        );
        artifacts["model_input.f32"] = artifact_description(
          true,
          true,
          "exact neural-network input",
          "Float32-le NCHW tensor after preprocessing and ImageNet normalization."
        );
        artifacts["model_input.png"] = artifact_description(
          true,
          true,
          "neural-network input preview",
          "ImageNet mean/std reversed without a second sRGB transfer function."
        );
        artifacts["model_input_shape.json"] = artifact_description(
          true,
          true,
          "model-input contract",
          "Dimensions, layout, normalization, and preview semantics."
        );
        artifacts["raw_depth.f32"] = artifact_description(
          true,
          true,
          "exact model output",
          "Float32-le raw depth before normalization or temporal processing."
        );
        artifacts["raw_depth.png"] = artifact_description(
          true,
          true,
          "raw model output preview",
          "Finite p2-p98 grayscale preview; not the tensor's numeric contract."
        );
        artifacts["raw_depth_heat.png"] = artifact_description(
          true,
          true,
          "raw model output preview",
          "Finite p2-p98 jet preview."
        );
        artifacts["raw_shape.json"] = artifact_description(
          true,
          true,
          "raw-depth contract",
          "Dimensions, scalar statistics, and preview bounds."
        );
        artifacts["depth.png"] = artifact_description(
          true,
          true,
          "normalized temporal depth",
          "Grayscale preview of the estimator output before warp prefiltering."
        );
        artifacts["depth.f32"] = artifact_description(
          true,
          true,
          "normalized temporal depth",
          "Exact float32-le estimator output before warp prefiltering."
        );
        artifacts["depth_shape.json"] = artifact_description(
          true,
          true,
          "normalized temporal depth contract",
          "Dimensions, layout, and scalar range for depth.f32."
        );
        artifacts["depth_heat.png"] = artifact_description(
          true,
          true,
          "normalized temporal depth",
          "Jet preview of the estimator output before warp prefiltering."
        );
        artifacts["warp_depth.png"] = artifact_description(
          true,
          true,
          "actual reprojection depth",
          "Grayscale preview of the exact depth texture sampled by the warp."
        );
        artifacts["warp_depth.f32"] = artifact_description(
          true,
          true,
          "actual reprojection depth",
          "Exact float32-le depth texture sampled by the warp."
        );
        artifacts["warp_depth_shape.json"] = artifact_description(
          true,
          true,
          "actual reprojection depth contract",
          "Dimensions, layout, and scalar range for warp_depth.f32."
        );
        artifacts["warp_depth_heat.png"] = artifact_description(
          true,
          true,
          "actual reprojection depth",
          "Jet preview of the exact depth texture sampled by the warp."
        );
        artifacts["adaptive_state.json"] = artifact_description(
          true,
          true,
          "adaptive controller state",
          "Generated schema-v3 typed state, decoded flags, counters, and normalization float4."
        );
        artifacts["roi_transform.words"] = artifact_description(
          true,
          true,
          "exact frame-owned ROI transform",
          "Eight little-endian uint4 vectors exactly as sampled by every ROI-aware Host-SBS stage."
        );
        artifacts["roi_transform.json"] = artifact_description(
          true,
          true,
          "decoded frame-owned ROI transform",
          "Raw vectors plus matched identity, flags, focus/crop rectangles, accepted model bounds, feathering, and fallback reason."
        );
        artifacts["scene_controller/scene_rgb.f32"] = artifact_description(
          scene_controller_available,
          false,
          "scene-controller matched appearance input",
          "Exact float32-le NCHW scene_rgb tensor on the aspect-preserving 256-square canvas."
        );
        artifacts["scene_controller/scene_rgb.png"] = artifact_description(
          scene_controller_available,
          false,
          "scene-controller matched appearance preview",
          "Direct preview of display-referred sRGB scene_rgb; invalid values are magenta."
        );
        artifacts["scene_controller/analysis_grid.f32"] = artifact_description(
          scene_controller_available,
          false,
          "scene-controller matched analysis input",
          "Exact float32-le analysis_grid tensor with generated ABI channel names."
        );
        artifacts["scene_controller/dense_out.f32"] = artifact_description(
          scene_controller_available,
          false,
          "scene-controller dense evidence output",
          "Exact dense layout, mask, motion, and event evidence for the matched frame."
        );
        artifacts["scene_controller/global_out.f32"] = artifact_description(
          scene_controller_available,
          false,
          "scene-controller global evidence output",
          "Exact generated-schema layout/event/pop/zero-plane logits, confidences, and validity."
        );
        artifacts["scene_controller/layout_history_post.f32"] =
          artifact_description(
            scene_controller_available,
            false,
            "scene-controller promoted layout history",
            "Post-resolve layout history; this is not the pre-resolve input bank."
          );
        artifacts["scene_controller/depth_history_post.f32"] =
          artifact_description(
            scene_controller_available,
            false,
            "scene-controller promoted depth history",
            "Post-resolve depth history; this is not the pre-resolve input bank."
          );
        artifacts["scene_controller/hidden_out.f32"] = artifact_description(
          scene_controller_available,
          false,
          "scene-controller recurrent output",
          "Exact hidden_out tensor; rules_v1 defines it as zero while preserving the future model ABI."
        );
        artifacts["scene_controller/meta.f32"] = artifact_description(
          scene_controller_available,
          false,
          "scene-controller matched metadata",
          "Exact float32-le generated-schema metadata words."
        );
        artifacts["scene_controller/rule_state.words"] =
          artifact_description(
            scene_controller_available,
            false,
            "scene-controller committed rule state",
            "Exact little-endian mixed-encoding 32-bit words after the matched resolve."
          );
        artifacts["scene_controller/tensor_manifest.json"] =
          artifact_description(
            scene_controller_available,
            false,
            "scene-controller tensor contract",
            "Shapes, generated channel names, stages, finite statistics, ABI hash, and rule revision."
          );
        artifacts["scene_controller/state.json"] = artifact_description(
          scene_controller_available,
          false,
          "scene-controller decoded state",
          "Generated typed fields, named values, flags, sidecars, and authoritative GPU validity checks."
        );
        artifacts["warp_map.f32"] = artifact_description(
          warp_map_available,
          false,
          "exact inverse-warp mapping",
          warp_map_available ?
            "Raw normalized source-U selected by the production Reproject function." :
            "Unavailable because the caller could not create the dump-only mapping pass."
        );
        artifacts["warp_map_shape.json"] = artifact_description(
          warp_map_available,
          false,
          "inverse-warp mapping contract",
          warp_map_available ?
            "Dimensions, content fit, validity rules, and displacement derivation." :
            "Unavailable with the dump-only mapping pass."
        );
        artifacts["warp_displacement_heat.png"] = artifact_description(
          warp_map_available,
          false,
          "derived inverse displacement",
          warp_map_available ?
            "Signed output-eye-pixel displacement derived from the exact inverse map." :
            "Unavailable with the dump-only mapping pass."
        );
        artifacts["warp_mask.png"] = artifact_description(
          warp_mask_available,
          false,
          "forward-coverage mask",
          warp_mask_available ?
            "Red marks exact pre-fill disocclusion; aspect bars remain unmarked." :
            "Unavailable because the caller could not create the dump-only coverage/mask pass."
        );
      artifacts["sbs.png"] = artifact_description(
        true,
        true,
        "packed Host-SBS output",
        "Final packed stereo preview using the same matched-frame transfer handling as source.png."
      );
        artifacts["meta.txt"] = artifact_description(
          true,
          true,
          "human-readable summary",
          "Compact compatibility summary; dump_manifest.json is authoritative."
        );
        artifacts["dump_manifest.json"] = artifact_description(
          true,
          true,
          "package contract",
          "Authoritative dimensions, formats, settings, stage descriptions, and availability."
        );

        nlohmann::json dimensions {
          {"source", texture_description(source)},
          {"model_input", {
                            {"width", completed.model_width},
                            {"height", completed.model_height},
                            {"channels", 3},
                            {"layout", "NCHW"},
                            {"dtype", "float32-le"},
                          }},
          {"raw_depth", {
                          {"width", completed.raw_width},
                          {"height", completed.raw_height},
                          {"format", "float32-le structured buffer"},
                        }},
          {"normalized_depth", texture_description(depth)},
          {"warp_depth", texture_description(warp_depth)},
          {"packed_sbs", texture_description(sbs)},
          {"eye", {
                    {"width", eye_width},
                    {"height", eye_height},
                  }},
          {"content_fit", {
                            {"scale_x", content_scale_x},
                            {"scale_y", content_scale_y},
                          }},
        };
        if (warp_map_available) {
          dimensions["warp_map"] = texture_description(warp_map);
        } else {
          dimensions["warp_map"] = nullptr;
        }
        if (warp_mask_available) {
          dimensions["warp_mask"] = texture_description(warp_mask);
        } else {
          dimensions["warp_mask"] = nullptr;
        }

        const std::string color_mode =
          completed.color_space == models::input_color_space::scrgb_hdr  ? "scrgb_hdr" :
          completed.color_space == models::input_color_space::linear_sdr ? "linear_sdr" :
                                                                           "srgb";
        const std::string color_preview_transform =
          completed.color_space == models::input_color_space::scrgb_hdr ?
            "luminance-preserving diagnostic tone map then sRGB OETF" :
          completed.color_space == models::input_color_space::linear_sdr ?
            "sRGB OETF" :
            "none; values are already sRGB code values";
        const std::string trigger_source =
          by_button && by_file ? "button+file" : by_button ? "button" :
                                                             "file";
        const nlohmann::json manifest {
          {"schema", 1},
          {"capture", "one matched, completed Host-SBS frame"},
          {"published_atomically", true},
          {"host_sbs_mode", "ai"},
          {"trigger", trigger_source},
          {"matched_frame_id", completed.matched_frame_id},
          {"depth_model", completed.depth_model},
          {"color_mode", color_mode},
          {"color_preview_transform", color_preview_transform},
          {"hdr_preview", hdr ? color_preview_transform : "not applied"},
          {"cuda_graph_active", completed.cuda_graph_active},
          {"warp_depth_prefilter_applied", completed.warp_depth_prefilter_applied},
          {"dimensions", std::move(dimensions)},
          {"normalization", {
                              {"effective_lower", normalization.lower},
                              {"effective_upper", normalization.upper},
                              {"initialized", normalization.initialized > 0.5f},
                              {"initialized_value", normalization.initialized},
                              {"frame_state", normalization_frame_state_name(normalization.frame_state)},
                              {"frame_state_value", normalization.frame_state},
                            }},
          {"raw_depth_statistics", {
                                     {"finite_count", raw_stats.finite_count},
                                     {"sample_count", static_cast<std::uint64_t>(raw_stats.width) * raw_stats.height},
                                     {"finite_fraction", static_cast<double>(raw_stats.finite_count) / (static_cast<double>(raw_stats.width) * raw_stats.height)},
                                     {"minimum", raw_stats.minimum},
                                     {"maximum", raw_stats.maximum},
                                     {"preview_low_p02", raw_stats.preview_low},
                                     {"preview_high_p98", raw_stats.preview_high},
                                   }},
          {"adaptive_summary", adaptive["decoded"]},
          {"roi_transform", roi_transform.summary},
          {"scene_controller", scene_controller.summary},
          {"config", config_json(cfg, completed.depth_model)},
          {"artifacts", std::move(artifacts)},
        };

        std::ostringstream meta;
        meta.imbue(std::locale::classic());
        meta << std::setprecision(std::numeric_limits<float>::max_digits10)
             << "depth_model=" << completed.depth_model << '\n'
             << "color_mode=" << color_mode << '\n'
             << "trigger=" << trigger_source << '\n'
             << "matched_frame_id=" << completed.matched_frame_id << '\n'
             << "source_width=" << source.desc.Width << '\n'
             << "source_height=" << source.desc.Height << '\n'
             << "packed_sbs_width=" << sbs.desc.Width << '\n'
             << "packed_sbs_height=" << sbs.desc.Height << '\n'
             << "eye_width=" << eye_width << '\n'
             << "eye_height=" << eye_height << '\n'
             << "model_input_width=" << completed.model_width << '\n'
             << "model_input_height=" << completed.model_height << '\n'
             << "raw_depth_width=" << raw_stats.width << '\n'
             << "raw_depth_height=" << raw_stats.height << '\n'
             << "raw_depth_finite_fraction="
             << static_cast<double>(raw_stats.finite_count) /
                  (static_cast<double>(raw_stats.width) * raw_stats.height)
             << '\n'
             << "raw_depth_min=" << raw_stats.minimum << '\n'
             << "raw_depth_max=" << raw_stats.maximum << '\n'
             << "raw_depth_preview_low_p02=" << raw_stats.preview_low << '\n'
             << "raw_depth_preview_high_p98=" << raw_stats.preview_high << '\n'
             << "normalization_effective_lower=" << normalization.lower << '\n'
             << "normalization_effective_upper=" << normalization.upper << '\n'
             << "normalization_initialized=" << normalization.initialized << '\n'
             << "normalization_frame_state=" << normalization.frame_state << '\n'
             << "roi_transform_unbound_zero="
             << (roi_transform.unbound_zero ? "true" : "false") << '\n'
             << "roi_transform_contract_valid="
             << (roi_transform.contract_valid ? "true" : "false") << '\n'
             << "cuda_graph_active="
             << (completed.cuda_graph_active ? "true" : "false") << '\n'
             << "warp_depth_prefilter_applied="
             << (completed.warp_depth_prefilter_applied ? "true" : "false") << '\n'
             << "scene_controller_backend="
             << config::to_string(cfg.scene_controller) << '\n'
             << "scene_controller_available="
             << (scene_controller_available ? "true" : "false") << '\n'
             << "scene_controller_frame_id="
             << completed.scene_controller_frame_id << '\n'
             << "scene_controller_authoritative_output_valid="
             << (scene_controller.authoritative_output_valid ? "true" : "false")
             << '\n'
             << "scene_controller_gpu_backend_generation="
             << scene_controller.gpu_backend_generation << '\n'
             << "scene_controller_gpu_roi_generation="
             << scene_controller.gpu_roi_generation << '\n'
             << "warp_map_available=" << (warp_map_available ? "true" : "false")
             << '\n'
             << "warp_mask_available=" << (warp_mask_available ? "true" : "false")
             << '\n';
        if (
          !write_text(paths.temporary / "meta.txt", meta.str()) ||
          !write_json(paths.temporary / "dump_manifest.json", manifest)
        ) {
          break;
        }

        error.clear();
        std::filesystem::rename(paths.temporary, paths.final, error);
        if (error) {
          BOOST_LOG(warning) << "SBS debug dump: atomic directory publication failed: "
                             << error.message();
          break;
        }
        success = true;
      } while (false);

      if (!success) {
        error.clear();
        std::filesystem::remove_all(paths.temporary, error);
        retry_backoff_frames_ = retry_backoff_frames;
        BOOST_LOG(warning)
          << "SBS debug dump failed; request retained for a rate-limited retry."sv;
        return false;
      }
    } catch (const std::exception &exception) {
      try {
        if (!paths.temporary.empty()) {
          std::error_code cleanup_error;
          std::filesystem::remove_all(paths.temporary, cleanup_error);
        }
      } catch (...) {
      }
      retry_backoff_frames_ = retry_backoff_frames;
      try {
        BOOST_LOG(warning)
          << "SBS debug dump transaction threw; partial output removed and request retained: "
          << exception.what();
      } catch (...) {
      }
      return false;
    } catch (...) {
      try {
        if (!paths.temporary.empty()) {
          std::error_code cleanup_error;
          std::filesystem::remove_all(paths.temporary, cleanup_error);
        }
      } catch (...) {
      }
      retry_backoff_frames_ = retry_backoff_frames;
      try {
        BOOST_LOG(warning)
          << "SBS debug dump transaction threw an unknown exception; partial output removed and request retained."sv;
      } catch (...) {
      }
      return false;
    }

    if (by_button && button) {
      button->store(false, std::memory_order_relaxed);
    }
    if (by_file) {
      try {
        error.clear();
        std::filesystem::remove(trigger, error);
      } catch (...) {
        error = std::make_error_code(std::errc::not_enough_memory);
      }
      file_trigger_pending_ = false;
      if (error) {
        file_trigger_enabled_ = false;
        try {
          BOOST_LOG(warning)
            << "SBS debug dump: could not remove dump.trigger; file polling disabled for this session: "
            << error.message();
        } catch (...) {
        }
      }
    }
    try {
      BOOST_LOG(info) << "SBS debug dump written to "sv << paths.final.string()
                      << " (model "sv << completed.depth_model << ", frame "
                      << completed.matched_frame_id << ')';
    } catch (...) {
    }
    return true;
  }

}  // namespace platf::sbs_debug
