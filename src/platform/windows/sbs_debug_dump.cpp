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
#include "src/depth_coordinate_v2.h"
#include "src/host_sbs_shader_cache.h"
#include "src/logging.h"
#include "src/model_manager.h"
#include "src/video_depth_estimator.h"

namespace platf::sbs_debug {

  using namespace std::literals;

  namespace {

    static_assert(
      std::endian::native == std::endian::little,
      "Dump contracts explicitly use little-endian float and integer words."
    );

    constexpr unsigned retry_backoff_frames = 60;

    bool parallax_v2_shader_identity_matches_contract(
      const std::shared_ptr<const models::parallax_v2_shader_provenance_t> &identity
    ) {
      using namespace models::depth_coordinate_v2;
      return identity &&
             identity->source_closure_schema == shader_source_closure_schema &&
             identity->source_compile_flags == shader_source_compile_flags &&
             identity->source_macro_count == shader_source_macro_count &&
             identity->source_closure_sha256 == shader_source_closure_sha256;
    }

    nlohmann::json parallax_v2_shader_identity_json(
      const models::parallax_v2_shader_provenance_t &identity
    ) {
      return {
        {"source_closure_schema", identity.source_closure_schema},
        {"source_compile_flags", identity.source_compile_flags},
        {"source_macro_count", identity.source_macro_count},
        {"source_closure_sha256", identity.source_closure_sha256},
      };
    }

    nlohmann::json parallax_v2_coordinate_binding(
      const models::parallax_v2_shader_provenance_t &identity,
      const char *word_count_key,
      const std::size_t word_count
    ) {
      using namespace models::depth_coordinate_v2;
      auto binding = parallax_v2_shader_identity_json(identity);
      binding["schema"] = contract_schema;
      binding["tag"] = contract_tag;
      binding[word_count_key] = word_count;
      return binding;
    }

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
      const std::filesystem::path &dir,
      const models::depth_coordinate_v2::model_preprocess_contract_t &preprocess
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
            encode_unit(
              normalized * preprocess.imagenet_std[channel] +
              preprocess.imagenet_mean[channel]
            );
        }
      }
      const nlohmann::json shape {
        {"schema", preprocess.model_input_schema},
        {"width", width},
        {"height", height},
        {"dtype", std::string {preprocess.dtype}},
        {"layout", std::string {preprocess.layout}},
        {"channels", {
                       std::string {preprocess.channels[0]},
                       std::string {preprocess.channels[1]},
                       std::string {preprocess.channels[2]},
                     }},
        {"stage", std::string {preprocess.stage}},
        {"imagenet_mean", preprocess.imagenet_mean},
        {"imagenet_std", preprocess.imagenet_std},
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

    bool dump_shadow_float_texture(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv,
      const std::filesystem::path &dir,
      const std::string_view stem,
      const std::string_view stage,
      texture_snapshot &snapshot
    ) {
      std::vector<float> values;
      if (!read_texture(device, ctx, srv, snapshot) ||
          !texture_float_values(snapshot, values)) {
        return false;
      }
      const scalar_stats stats = calculate_scalar_stats(values);
      if (stats.finite_count != values.size()) {
        return false;
      }
      const std::string name(stem);
      return write_float_texture_artifacts(
               dir / (name + ".f32"),
               dir / (name + "_shape.json"),
               snapshot,
               stage
             ) &&
             write_percentile_previews(
               dir / (name + ".png"),
               dir / (name + "_heat.png"),
               snapshot.desc.Width,
               snapshot.desc.Height,
               values,
               stats
             );
    }

    bool dump_parallax_v2_state(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      const frame &completed,
      const std::filesystem::path &dir,
      nlohmann::json &summary
    ) {
      using namespace models::depth_coordinate_v2;
      if (!parallax_v2_shader_identity_matches_contract(
            completed.parallax_v2_shader_provenance
          )) {
        return false;
      }
      const auto &shader_identity = *completed.parallax_v2_shader_provenance;
      std::vector<float> state;
      std::vector<float> frame_stats;
      if (!read_float_buffer(
            device,
            ctx,
            completed.shadow_state,
            state_float_count,
            state
          ) ||
          !read_float_buffer(
            device,
            ctx,
            completed.shadow_frame_stats,
            frame_stats_float_count,
            frame_stats
          )) {
        return false;
      }
      bool state_finite = true;
      for (std::size_t index = 0; index < state.size(); ++index) {
        if (state_fields[index].gpu_encoding == state_gpu_encoding_e::float_value &&
            !std::isfinite(state[index])) {
          state_finite = false;
          break;
        }
      }
      if (!state_finite || !std::all_of(frame_stats.begin(), frame_stats.end(), [](float value) {
            return std::isfinite(value);
          })) {
        return false;
      }
      if (std::bit_cast<std::uint32_t>(state[contract_tag_bits]) != contract_tag) {
        return false;
      }
      const bool runtime_constants_valid =
        std::isfinite(completed.parallax_v2_raw_coordinate_scale) &&
        completed.parallax_v2_raw_coordinate_scale > 0.0f &&
        std::isfinite(completed.parallax_v2_requested_pop_strength) &&
        completed.parallax_v2_requested_pop_strength >= 0.0f &&
        std::isfinite(completed.parallax_v2_requested_gain) &&
        completed.parallax_v2_requested_gain >= 0.0f &&
        std::abs(completed.parallax_v2_requested_gain -
                 completed.parallax_v2_requested_pop_strength * gain_per_pop) <= 1.0e-7f;
      const bool state_frame_valid = state[frame_valid] > 0.5f;
      const auto calibration_revision_value =
        std::bit_cast<std::uint32_t>(state[calibration_revision]);
      const float latched_near_tail_coverage_value = state[latched_near_tail_coverage];
      const float effective_near_log_tau_value = state[effective_near_log_tau];
      const auto latched_near_tail_count_value =
        std::bit_cast<std::uint32_t>(state[latched_near_tail_count]);
      const auto camera_center_integrity_value =
        std::bit_cast<std::uint32_t>(state[camera_center_integrity_bits]);
      const bool camera_center_integrity_valid = camera_center_integrity_is_valid(
        std::bit_cast<std::uint32_t>(state[center]),
        std::bit_cast<std::uint32_t>(state[inverse_scale]),
        calibration_revision_value,
        camera_center_integrity_value
      );
      const float expected_near_log_tau =
        near_tail_effective_tau_for_coverage(latched_near_tail_coverage_value);
      const bool near_shoulder_semantics_valid =
        std::isfinite(latched_near_tail_coverage_value) &&
        latched_near_tail_coverage_value >= 0.0f &&
        latched_near_tail_coverage_value <= 1.0f &&
        std::isfinite(effective_near_log_tau_value) &&
        effective_near_log_tau_value >= near_log_tau_dense &&
        effective_near_log_tau_value <= near_log_tau &&
        std::abs(effective_near_log_tau_value - expected_near_log_tau) <= 1.0e-6f;
      const bool camera_initialized =
        camera_center_integrity_valid && near_shoulder_semantics_valid &&
        state[inverse_scale] > 0.0f &&
        calibration_revision_value > 0u &&
        calibration_revision_is_valid(calibration_revision_value);
      const bool camera_empty =
        state[center] == 0.0f && state[inverse_scale] == 0.0f;
      const bool near_shoulder_empty =
        latched_near_tail_coverage_value == 0.0f &&
        effective_near_log_tau_value == near_log_tau &&
        latched_near_tail_count_value == 0u;
      const bool state_semantics_valid =
        camera_center_integrity_valid &&
        calibration_revision_is_valid(calibration_revision_value) &&
        (state[frame_valid] == 0.0f || state[frame_valid] == 1.0f) &&
        state[container_scale] >= 0.0f && state[container_scale] <= 1.0f &&
        state[convergence_curve] == convergence_curve_default &&
        (state_frame_valid ?
           (camera_initialized && state[container_scale] > 0.0f) :
           ((camera_initialized || (camera_empty && near_shoulder_empty)) &&
            state[container_scale] == 1.0f)) &&
        (!camera_initialized ||
         std::abs(1.0f / state[inverse_scale] -
                  completed.parallax_v2_raw_coordinate_scale) <= 1.0e-6f);
      const float valid_count = frame_stats[frame_stat_valid_count];
      const float texel_count = frame_stats[frame_stat_texel_count];
      const bool frame_is_valid = frame_stats[frame_stat_valid] > 0.5f;
      const bool expected_state_frame_valid =
        frame_is_valid &&
        frame_stats[frame_stat_population_std] > collapse_abs_epsilon;
      const bool frame_counts_valid =
        valid_count >= 0.0f && texel_count > 0.0f && valid_count <= texel_count &&
        valid_count == std::floor(valid_count) && texel_count == std::floor(texel_count) &&
        texel_count <= static_cast<float>(std::numeric_limits<std::uint32_t>::max());
      const bool near_tail_count_valid = frame_counts_valid &&
        latched_near_tail_count_value <= static_cast<std::uint32_t>(texel_count) &&
        std::abs(
          latched_near_tail_coverage_value -
          static_cast<float>(latched_near_tail_count_value) / texel_count
        ) <= 1.0e-6f;
      const bool frame_semantics_valid =
        (frame_stats[frame_stat_valid] == 0.0f ||
         frame_stats[frame_stat_valid] == 1.0f) &&
        frame_counts_valid && frame_stats[frame_stat_population_std] >= 0.0f &&
        (frame_is_valid ?
           (valid_count == texel_count &&
            frame_stats[frame_stat_maximum] >= frame_stats[frame_stat_minimum]) :
           (frame_stats[frame_stat_mean] == 0.0f &&
            frame_stats[frame_stat_population_std] == 0.0f &&
            frame_stats[frame_stat_minimum] == 0.0f &&
            frame_stats[frame_stat_maximum] == 0.0f));
      if (!runtime_constants_valid || !state_semantics_valid || !near_tail_count_valid ||
          !frame_semantics_valid || state_frame_valid != expected_state_frame_valid) {
        return false;
      }

      nlohmann::json fields = nlohmann::json::array();
      nlohmann::json named_values = nlohmann::json::object();
      for (std::size_t index = 0; index < state.size(); ++index) {
        const auto &descriptor = state_fields[index];
        const std::string name {descriptor.name};
        if (descriptor.gpu_encoding == state_gpu_encoding_e::uint_bits) {
          const auto bits = std::bit_cast<std::uint32_t>(state[index]);
          fields.push_back({
            {"index", index},
            {"name", name},
            {"type", "uint32-bitcast"},
            {"value", bits},
          });
          named_values[name] = bits;
          continue;
        }
        fields.push_back({
          {"index", index},
          {"name", name},
          {"type", "float32"},
          {"value", state[index]},
        });
        named_values[name] = state[index];
      }

      const float effective_gain_value = state_frame_valid ?
        completed.parallax_v2_requested_gain * state[container_scale] : 0.0f;
      const float latched_scale_value = camera_initialized ?
        1.0f / state[inverse_scale] : 0.0f;
      const auto confirmed_cut_count_value =
        std::bit_cast<std::uint32_t>(state[confirmed_cut_count]);
      const nlohmann::json decoded {
        {"frame_valid", state_frame_valid},
        {"camera_valid", camera_initialized},
        {"calibration_revision", calibration_revision_value},
        {"confirmed_cut_count", confirmed_cut_count_value},
        {"contract_tag", contract_tag},
        {"requested_gain", completed.parallax_v2_requested_gain},
        {"requested_pop_strength", completed.parallax_v2_requested_pop_strength},
        {"latched_scale", latched_scale_value},
        {"convergence_curve", state[convergence_curve]},
        {"container_scale", state[container_scale]},
        {"effective_gain", effective_gain_value},
        {"latched_near_tail_coverage", latched_near_tail_coverage_value},
        {"effective_near_log_tau", effective_near_log_tau_value},
        {"latched_near_tail_count", latched_near_tail_count_value},
        {"camera_center_integrity_bits", camera_center_integrity_value},
      };
      const nlohmann::json state_json {
        {"schema", shadow_state_dump_schema},
        {"coordinate_contract", parallax_v2_coordinate_binding(
                                  shader_identity,
                                  "state_word_count",
                                  state_float_count
                                )},
        {"source", std::string {shadow_state_source}},
        {"capture", std::string {shadow_state_capture}},
        {"rendered_output_selected", true},
        {"wire_contract", "authenticated live Host-SBS renderer input; not a client wire contract"},
        {"units", {
                    {"coordinate", "dimensionless canonical coordinate derived from raw depth"},
                    {"gain", "one-eye source-U per curve unit"},
                    {"parallax", "signed one-eye source-U"},
                  }},
        {"constants", {
                        {"raw_coordinate_scale", completed.parallax_v2_raw_coordinate_scale},
                        {"collapse_abs_epsilon", collapse_abs_epsilon},
                        {"far_tau", far_tau},
                        {"near_log_tau", near_log_tau},
                        {"near_tail_probe_u", near_tail_probe_u},
                        {"near_tail_coverage_low", near_tail_coverage_low},
                        {"near_tail_coverage_high", near_tail_coverage_high},
                        {"near_log_tau_dense", near_log_tau_dense},
                        {"gain_per_pop", gain_per_pop},
                        {"reference_pop_strength", reference_pop_strength},
                        {"reference_gain_at_pop_2", parallax_gain},
                        {"requested_gain", completed.parallax_v2_requested_gain},
                        {"requested_pop_strength", completed.parallax_v2_requested_pop_strength},
                        {"direct_container_limit", direct_container_limit},
                        {"max_horizontal_slope", max_horizontal_slope},
                        {"max_vertical_shear", max_vertical_shear},
                        {"convergence_curve_default", convergence_curve_default},
                      }},
        {"fields", std::move(fields)},
        {"named_values", std::move(named_values)},
        {"decoded", decoded},
        {"adaptation_semantics", {
                                    {"coordinate", "center-latched-until-cut-fixed-authenticated-scale-retained-across-unusable"},
                                    {"convergence_curve", "separately-scene-latched-curve-offset-currently-zero"},
                                    {"requested_gain", "immutable-cfg-pop-strength"},
                                    {"container_scale", "frame-local-hard-direct-parallax-attenuation-recoverable-next-frame"},
                                    {"near_shoulder", "shot-latched-near-tail-coverage-and-effective-tau-reset-on-confirmed-cut"},
                                  }},
      };

      nlohmann::json frame_named = nlohmann::json::object();
      for (std::size_t index = 0; index < frame_stats.size(); ++index) {
        frame_named[frame_stat_names[index]] = frame_stats[index];
      }
      const nlohmann::json frame_json {
        {"schema", shadow_frame_stats_dump_schema},
        {"coordinate_contract", parallax_v2_coordinate_binding(
                                  shader_identity,
                                  "frame_stats_word_count",
                                  frame_stats_float_count
                                )},
        {"source", std::string {frame_stats_source}},
        {"named_values", std::move(frame_named)},
      };
      summary = decoded;
      summary["raw_coordinate_scale"] = completed.parallax_v2_raw_coordinate_scale;
      summary["rendered_output_selected"] = true;
      return write_json(dir / "shadow_state.json", state_json) &&
             write_json(dir / "shadow_frame_stats.json", frame_json);
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
      adaptive_summary = {
        {"schema", sbs_adaptive_state::schema_version},
        {"source", std::string(sbs_adaptive_state::source)},
        {"capture", std::string(sbs_adaptive_state::capture)},
        {"role", "comparison-only scene-cut bridge evidence; no live V2 geometry authority"},
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
                      {"geometry_authority", false},
                    }},
        {"normalization", {
                            {"role", "comparison-only scene-cut bridge evidence"},
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
        {"schema", 2},
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
          {"inverse", "12-step contractive fixed-point solution of the signed final-parallax field"},
          {"mask", "warp_mask.png red marks finite-source boundary extrapolation; V2 has no internal owner or synthetic-fill path"},
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
      const std::string &model_name,
      const std::string &effective_model_url
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
        {"depth_model_url", effective_model_url},
        {"configured_depth_model_url", cfg.depth_model_url},
        {"max_encode_width", cfg.max_encode_width},
        {"cuda_graph", cfg.cuda_graph},
        {"parallax_v2_shadow", false},
        {"parallax_v2_render", true},
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
    prepared_frame_id_ = 0;
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

  void dumper::reject_pending_request() noexcept {
    snapshot_armed_for_dump_ = false;
    prepared_frame_id_ = 0;
    retry_backoff_frames_ = 0;
    if (auto *button = button_request_.get()) {
      button->store(false, std::memory_order_relaxed);
    }
    if (file_trigger_pending_) {
      try {
        std::error_code error;
        std::filesystem::remove(dir_ / "dump.trigger", error);
      } catch (...) {
      }
    }
    file_trigger_pending_ = false;
  }

  bool dumper::snapshot_requested() {
    snapshot_armed_for_dump_ = false;
    prepared_frame_id_ = 0;
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

  bool dumper::preflight_requested_v2_frame(
    ID3D11Device *device,
    ID3D11DeviceContext *ctx,
    ID3D11ShaderResourceView *shadow_state,
    const std::uint64_t matched_frame_id
  ) noexcept {
    auto *button = button_request_.get();
    const bool requested =
      (button && button->load(std::memory_order_relaxed)) ||
      file_trigger_pending_;
    if (!requested || !snapshot_armed_for_dump_) {
      return false;
    }

    try {
      std::vector<float> state;
      if (read_float_buffer(
            device,
            ctx,
            shadow_state,
            models::depth_coordinate_v2::state_float_count,
            state
          ) &&
          std::bit_cast<std::uint32_t>(
            state[models::depth_coordinate_v2::contract_tag_bits]
          ) == models::depth_coordinate_v2::contract_tag &&
          state[models::depth_coordinate_v2::frame_valid] > 0.5f &&
          std::isfinite(state[models::depth_coordinate_v2::center]) &&
          std::isfinite(state[models::depth_coordinate_v2::inverse_scale]) &&
          std::isfinite(state[models::depth_coordinate_v2::convergence_curve]) &&
          state[models::depth_coordinate_v2::inverse_scale] > 0.0f &&
          std::bit_cast<std::uint32_t>(
            state[models::depth_coordinate_v2::calibration_revision]
          ) > 0u) {
        prepared_frame_id_ = matched_frame_id;
        return true;
      }
    } catch (...) {
    }

    // This completion did not render an attributable current V2 geometry pair (it may have held
    // the prior target, drawn identity without a camera, or exposed unreadable state). Retain the
    // trigger and try another completed frame after a bounded delay; do not mix current source/raw
    // tensors with an unrelated packed SBS target.
    snapshot_armed_for_dump_ = false;
    retry_backoff_frames_ = retry_backoff_frames;
    try {
      BOOST_LOG(warning)
        << "SBS debug dump deferred: selected parallax-v2 completion is not a current valid "sv
           "camera/output pair."sv;
    } catch (...) {
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
      !completed.raw_depth || !completed.warp_depth || !completed.sbs ||
      completed.model_width <= 0 ||
      completed.model_height <= 0 || completed.raw_width <= 0 ||
      completed.raw_height <= 0 || !completed.raw_model_provenance ||
      completed.raw_model_provenance->depth_model.empty() ||
      completed.raw_model_provenance->depth_model != completed.depth_model ||
      completed.raw_model_provenance->onnx_sha256.empty() ||
      completed.raw_model_provenance->preprocess_source_closure_sha256.empty() ||
      completed.raw_width != completed.model_width ||
      completed.raw_height != completed.model_height ||
      prepared_frame_id_ != completed.matched_frame_id
    ) {
      return false;
    }
    if (!completed.parallax_v2_render_selected ||
        !completed.parallax_v2_producer_active ||
        !completed.shadow_candidate_parallax ||
        !completed.shadow_vertical_majorant ||
        !completed.shadow_final_parallax || !completed.shadow_state ||
        !completed.shadow_frame_stats) {
      BOOST_LOG(warning)
        << "SBS debug dump: production V2 renderer is not selected or has an incomplete "sv
           "authenticated resource set; dump rejected (legacy live rendering is unsupported)."sv;
      return false;
    }
    if (!completed.shadow_coordinate) {
      BOOST_LOG(warning)
        << "SBS debug dump: the explicit Dump 3D canonical-coordinate snapshot is "sv
           "unavailable; live V2 rendering remains authenticated and unaffected."sv;
      return false;
    }
    if (completed.parallax_v2_live_renderer_source_closure_sha256 !=
          models::host_sbs_shader_cache::
            parallax_v2_live_renderer_source_closure_sha256) {
      BOOST_LOG(warning)
        << "SBS debug dump: production V2 renderer source closure is missing or "sv
           "mismatched; dump rejected."sv;
      return false;
    }
    const auto &model_identity = *completed.raw_model_provenance;
    const auto *capture_calibration =
      models::depth_coordinate_v2::find_capture_calibration(
        model_identity.depth_model,
        model_identity.depth_model_url,
        model_identity.onnx_sha256,
        model_identity.preprocess_profile,
        model_identity.preprocess_source_closure_sha256,
        static_cast<std::uint32_t>(completed.model_width),
        static_cast<std::uint32_t>(completed.model_height)
      );
    if (!capture_calibration) {
      BOOST_LOG(warning)
        << "SBS debug dump: production V2 resources do not resolve to exactly one "sv
           "authenticated model/preprocess/shape calibration; dump rejected."sv;
      return false;
    }
    if (!parallax_v2_shader_identity_matches_contract(
          completed.parallax_v2_shader_provenance
        )) {
      BOOST_LOG(warning)
        << "SBS debug dump: production V2 resources have missing or mismatched "sv
           "shader-source provenance; dump rejected."sv;
      return false;
    }
    const auto &capture_preprocess = capture_calibration->preprocess;

    std::filesystem::path trigger;
    std::error_code error;
    output_paths paths;
    bool success = false;
    try {
      trigger = dir_ / "dump.trigger";
      normalization_state normalization {};
      bool scene_cut_bridge_state_available = false;
      if (completed.adaptive_state && completed.depth_frame_state) {
        try {
          scene_cut_bridge_state_available =
            read_normalization_state(
              device,
              ctx,
              completed.depth_frame_state,
              normalization
            ) == depth_dumpability::valid;
        } catch (...) {
          // This comparison-only bridge must never gate an authenticated production-V2 dump.
        }
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
        texture_snapshot warp_depth;
        texture_snapshot sbs;
        if (
          !read_texture(device, ctx, completed.source, source) ||
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
            paths.temporary,
            capture_preprocess
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
            paths.temporary / "warp_depth.f32",
            paths.temporary / "warp_depth_shape.json",
            warp_depth,
            "exact one-eye source-U from the least near-preserving anisotropic 2D Lipschitz majorant sampled by live Host-SBS V2 reprojection"
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

        nlohmann::json adaptive = nullptr;
        bool adaptive_available = false;
        if (scene_cut_bridge_state_available) {
          try {
            adaptive_available = dump_adaptive_state(
              device,
              ctx,
              completed.adaptive_state,
              normalization,
              completed,
              paths.temporary,
              adaptive
            );
          } catch (...) {
            // Scene-cut bridge evidence is optional and has no live V2 geometry authority.
          }
        }

        // Bind provenance to the exact bytes that were successfully written into this
        // transaction directory. Hashing the files (rather than reconstructed vectors) also
        // covers byte order and the canonical JSON serialization of the input-shape contract.
        const std::string raw_depth_sha256 = models::file_sha256_hex(
          paths.temporary / "raw_depth.f32"
        );
        const std::string model_input_sha256 = models::file_sha256_hex(
          paths.temporary / "model_input.f32"
        );
        const std::string model_input_shape_sha256 = models::file_sha256_hex(
          paths.temporary / "model_input_shape.json"
        );
        if (raw_depth_sha256.empty() || model_input_sha256.empty() ||
            model_input_shape_sha256.empty()) {
          BOOST_LOG(warning)
            << "SBS debug dump: cannot hash the exact raw-model artifacts; publication aborted."sv;
          break;
        }

        texture_snapshot shadow_coordinate;
        texture_snapshot shadow_candidate;
        texture_snapshot shadow_vertical;
        texture_snapshot shadow_final;
        nlohmann::json shadow_summary = nullptr;
        if (
          !dump_shadow_float_texture(
             device,
             ctx,
             completed.shadow_coordinate,
             paths.temporary,
             "shadow_coordinate",
             "parallax-v2 canonical unbounded coordinate u; diagnostic only",
             shadow_coordinate
           ) ||
           !dump_shadow_float_texture(
             device,
             ctx,
             completed.shadow_candidate_parallax,
             paths.temporary,
             "shadow_candidate_parallax",
             "parallax-v2 immutable signed pre-limiter candidate one-eye source-U; diagnostic only",
             shadow_candidate
           ) ||
           !dump_shadow_float_texture(
             device,
             ctx,
             completed.shadow_vertical_majorant,
             paths.temporary,
             "shadow_vertical_majorant",
             "parallax-v2 least column-wise near-preserving shear-2 majorant; explicit intermediate consumed by the row limiter",
             shadow_vertical
           ) ||
           !dump_shadow_float_texture(
             device,
             ctx,
             completed.shadow_final_parallax,
             paths.temporary,
             "shadow_final_parallax",
             "parallax-v2 least anisotropic 2D near-preserving Lipschitz majorant; live render position authority",
             shadow_final
           ) ||
           !dump_parallax_v2_state(
             device,
             ctx,
             completed,
             paths.temporary,
             shadow_summary
           )
        ) {
          break;
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

        const std::string warp_scalar_stage =
          "actual near-preserving anisotropic 2D Lipschitz majorant sampled by live V2 reprojection";
        const std::string warp_scalar_description =
          "Exact signed one-eye source-U after the vertical shear-2 majorant and row majorant, sampled by the live V2 12-step contractive inverse.";
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
          false,
          false,
          "removed V1 normalized-depth renderer artifact",
          "Not captured: production Host SBS consumes the authenticated V2 parallax field directly."
        );
        artifacts["depth.f32"] = artifact_description(
          false,
          false,
          "removed V1 normalized-depth renderer artifact",
          "Not captured: production Host SBS consumes the authenticated V2 parallax field directly."
        );
        artifacts["depth_shape.json"] = artifact_description(
          false,
          false,
          "removed V1 normalized-depth renderer artifact",
          "Not captured: the V1 normalized-depth live renderer has been removed."
        );
        artifacts["depth_heat.png"] = artifact_description(
          false,
          false,
          "removed V1 normalized-depth renderer artifact",
          "Not captured: the V1 normalized-depth live renderer has been removed."
        );
        artifacts["warp_depth.png"] = artifact_description(
          true,
          true,
          warp_scalar_stage,
          "Grayscale preview of the exact near-preserving anisotropic 2D majorant sampled by the warp."
        );
        artifacts["warp_depth.f32"] = artifact_description(
          true,
          true,
          warp_scalar_stage,
          warp_scalar_description
        );
        artifacts["warp_depth_shape.json"] = artifact_description(
          true,
          true,
          "actual near-preserving anisotropic 2D majorant contract",
          "Dimensions, layout, units, and scalar range for warp_depth.f32."
        );
        artifacts["warp_depth_heat.png"] = artifact_description(
          true,
          true,
          warp_scalar_stage,
          "Jet preview of the exact near-preserving anisotropic 2D majorant sampled by the warp."
        );
        artifacts["adaptive_state.json"] = artifact_description(
          adaptive_available,
          false,
          "scene-cut bridge comparison state",
          adaptive_available ?
            "Comparison-only cut flags, counters, and normalization state; no live V2 geometry authority." :
            "Unavailable; scene-cut bridge evidence never gates an authenticated live V2 dump."
        );
        artifacts["shadow_coordinate.f32"] = artifact_description(
          true,
          true,
          "parallax-v2 canonical coordinate diagnostic",
          "Exact float32-le unbounded canonical coordinate u; diagnostic only and never used by the live renderer."
        );
        artifacts["shadow_coordinate_shape.json"] = artifact_description(
          true,
          false,
          "parallax-v2 canonical coordinate contract",
          "Dimensions, units, and finite scalar range."
        );
        artifacts["shadow_coordinate.png"] = artifact_description(
          true,
          false,
          "parallax-v2 canonical coordinate preview",
          "Finite p2-p98 grayscale preview; not the numeric contract."
        );
        artifacts["shadow_coordinate_heat.png"] = artifact_description(
          true,
          false,
          "parallax-v2 canonical coordinate preview",
          "Finite p2-p98 jet preview."
        );
        artifacts["shadow_candidate_parallax.f32"] = artifact_description(
          true,
          false,
          "parallax-v2 pre-limiter candidate displacement",
          "Exact immutable signed one-eye source-U before the spatial limiter; diagnostic only."
        );
        artifacts["shadow_candidate_parallax_shape.json"] = artifact_description(
          true,
          false,
          "parallax-v2 pre-limiter candidate displacement contract",
          "Dimensions, units, and finite scalar range."
        );
        artifacts["shadow_candidate_parallax.png"] = artifact_description(
          true,
          false,
          "parallax-v2 pre-limiter candidate displacement preview",
          "Finite p2-p98 grayscale preview."
        );
        artifacts["shadow_candidate_parallax_heat.png"] = artifact_description(
          true,
          false,
          "parallax-v2 pre-limiter candidate displacement preview",
          "Finite p2-p98 jet preview."
        );
        artifacts["shadow_vertical_majorant.f32"] = artifact_description(
          true,
          false,
          "parallax-v2 vertical shear-limiter intermediate",
          "Exact signed one-eye source-U after the least column-wise majorant v >= candidate with |dv/dy| <= max_vertical_shear/target_width; diagnostic intermediate consumed by the row limiter."
        );
        artifacts["shadow_vertical_majorant_shape.json"] = artifact_description(
          true,
          false,
          "parallax-v2 vertical shear-limiter intermediate contract",
          "Dimensions, units, finite scalar range, v >= candidate, and the generated max_vertical_shear bound; not the live renderer position authority."
        );
        artifacts["shadow_vertical_majorant.png"] = artifact_description(
          true,
          false,
          "parallax-v2 vertical shear-limiter intermediate preview",
          "Finite p2-p98 grayscale preview of the exact column-wise majorant."
        );
        artifacts["shadow_vertical_majorant_heat.png"] = artifact_description(
          true,
          false,
          "parallax-v2 vertical shear-limiter intermediate preview",
          "Finite p2-p98 jet preview of the exact column-wise majorant."
        );
        artifacts["shadow_final_parallax.f32"] = artifact_description(
          true,
          true,
          "parallax-v2 final near-preserving anisotropic 2D majorant",
          "Exact signed one-eye source-U after the row majorant of shadow_vertical_majorant; q >= vertical >= candidate, |dq/dx| <= max_horizontal_slope/target_width, and |dq/dy| <= max_vertical_shear/target_width. Live V2 render position authority."
        );
        artifacts["shadow_final_parallax_shape.json"] = artifact_description(
          true,
          false,
          "parallax-v2 final near-preserving anisotropic 2D majorant contract",
          "Dimensions, units, finite scalar range, no-lowering chain, horizontal slope bound, and vertical shear bound; live renderer authority."
        );
        artifacts["shadow_final_parallax.png"] = artifact_description(
          true,
          false,
          "parallax-v2 final near-preserving anisotropic 2D majorant preview",
          "Finite p2-p98 grayscale preview of the live V2 position field."
        );
        artifacts["shadow_final_parallax_heat.png"] = artifact_description(
          true,
          false,
          "parallax-v2 final near-preserving anisotropic 2D majorant preview",
          "Finite p2-p98 jet preview of the live V2 position field."
        );
        artifacts["shadow_state.json"] = artifact_description(
          true,
          false,
          "parallax-v2 shot calibration and attenuation state",
          "Independent typed state bound to the exact coordinate contract tag."
        );
        artifacts["shadow_frame_stats.json"] = artifact_description(
          true,
          false,
          "parallax-v2 current-frame moments",
          "Independent mean/std/min/max state bound to the exact coordinate contract tag."
        );
        artifacts["warp_map.f32"] = artifact_description(
          warp_map_available,
          false,
          "exact inverse-warp mapping",
          warp_map_available ?
            "Raw normalized source-U selected by the production Reproject function." :
            "Unavailable because the matching dump-only mapping pass could not be created."
        );
        artifacts["warp_map_shape.json"] = artifact_description(
          warp_map_available,
          false,
          "inverse-warp mapping contract",
          warp_map_available ?
            "Dimensions, content fit, validity rules, and displacement derivation." :
            "Unavailable with the matching dump-only mapping pass."
        );
        artifacts["warp_displacement_heat.png"] = artifact_description(
          warp_map_available,
          false,
          "derived inverse displacement",
          warp_map_available ?
            "Signed output-eye-pixel displacement derived from the exact inverse map." :
            "Unavailable with the matching dump-only mapping pass."
        );
        artifacts["warp_mask.png"] = artifact_description(
          warp_mask_available,
          false,
          "V2 boundary-extrapolation mask",
          warp_mask_available ?
            "Red marks inverse samples outside the finite source interval that the live renderer clamps to the nearest boundary column; V2 has no internal owner selection or synthetic fill." :
            "Unavailable because the matching dump-only mask pass could not be created."
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
          {"normalized_depth", nullptr},
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
        dimensions["shadow_coordinate"] = texture_description(shadow_coordinate);
        dimensions["shadow_candidate_parallax"] =
          texture_description(shadow_candidate);
        dimensions["shadow_vertical_majorant"] =
          texture_description(shadow_vertical);
        dimensions["shadow_final_parallax"] = texture_description(shadow_final);

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
        const nlohmann::json shadow_shader_source =
          parallax_v2_shader_identity_json(
            *completed.parallax_v2_shader_provenance
          );
        nlohmann::json manifest {
          {"schema", 7},
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
          {"warp_depth_prefilter_applied", false},
          {"renderer", {
                         {"authority", "authenticated-parallax-v2-final-near-preserving-anisotropic-2d-majorant"},
                         {"parallax_v2_render_requested", true},
                         {"parallax_v2_render_selected", true},
                         {"mapping_artifacts_match_selected_renderer", warp_map_available && warp_mask_available},
                         {"parallax_v2_position_field", "shadow_final_parallax"},
                         {"parallax_v2_coordinate_role", "shadow_coordinate is diagnostic only; it has no renderer authority"},
                         {"parallax_v2_vertical_majorant_role", "least column-wise v >= candidate with adjacent-row source-U change <= max_vertical_shear/target_width; diagnostic intermediate consumed by the row limiter"},
                         {"parallax_v2_majorant_role", "least row-wise q >= shadow_vertical_majorant; final q >= vertical >= candidate with horizontal slope <= max_horizontal_slope and vertical shear <= max_vertical_shear; live position authority"},
                         {"parallax_v2_inverse", "12-step contractive fixed point; no owner pass or synthetic fill"},
                         {"live_shader_source", nlohmann::json {
                              {"source_closure_schema", models::host_sbs_shader_cache::source_closure_schema},
                              {"source_compile_flags", models::host_sbs_shader_cache::shader_compile_flags},
                              {"source_macro_count", 0u},
                              {"source_closure_sha256", completed.parallax_v2_live_renderer_source_closure_sha256},
                              {"source_file", std::string {models::host_sbs_shader_cache::parallax_v2_live_renderer.filename}},
                              {"entrypoint", std::string {models::host_sbs_shader_cache::parallax_v2_live_renderer.entrypoint}},
                              {"target", std::string {models::host_sbs_shader_cache::parallax_v2_live_renderer.target}},
                              {"diagnostic_source_closure_sha256", std::string {models::host_sbs_shader_cache::parallax_v2_diagnostic_source_closure_sha256}},
                              {"mapping_source_file", std::string {models::host_sbs_shader_cache::parallax_v2_live_mapping.filename}},
                              {"mapping_entrypoint", std::string {models::host_sbs_shader_cache::parallax_v2_live_mapping.entrypoint}},
                              {"mask_source_file", std::string {models::host_sbs_shader_cache::parallax_v2_live_mask.filename}},
                              {"mask_entrypoint", std::string {models::host_sbs_shader_cache::parallax_v2_live_mask.entrypoint}},
                            }},
                       }},
          {"dimensions", std::move(dimensions)},
          {"normalization", adaptive_available ?
             nlohmann::json {
               {"role", "comparison-only scene-cut bridge evidence; no live V2 geometry authority"},
               {"effective_lower", normalization.lower},
               {"effective_upper", normalization.upper},
               {"initialized", normalization.initialized > 0.5f},
               {"initialized_value", normalization.initialized},
               {"frame_state", normalization_frame_state_name(normalization.frame_state)},
               {"frame_state_value", normalization.frame_state},
             } : nlohmann::json {nullptr}},
          {"raw_depth_statistics", {
                                     {"finite_count", raw_stats.finite_count},
                                     {"sample_count", static_cast<std::uint64_t>(raw_stats.width) * raw_stats.height},
                                     {"finite_fraction", static_cast<double>(raw_stats.finite_count) / (static_cast<double>(raw_stats.width) * raw_stats.height)},
                                     {"minimum", raw_stats.minimum},
                                     {"maximum", raw_stats.maximum},
                                     {"preview_low_p02", raw_stats.preview_low},
                                     {"preview_high_p98", raw_stats.preview_high},
                                   }},
          {"adaptive_summary", adaptive_available ? adaptive["decoded"] :
                                                     nlohmann::json {nullptr}},
          {"parallax_v2_shadow", {
                                    {"requested", false},
                                    {"active", true},
                                    {"rendered_output_selected", true},
                                    {"shader_source", shadow_shader_source},
                                    {"state", std::move(shadow_summary)},
                                  }},
          {"config", config_json(
                       cfg,
                       completed.depth_model,
                       completed.raw_model_provenance->depth_model_url
                     )},
          {"artifacts", std::move(artifacts)},
        };
        manifest[std::string {
            models::depth_coordinate_v2::capture_provenance_manifest_key
          }] = {
            {"schema", models::depth_coordinate_v2::capture_provenance_schema},
            {"binding", std::string {
                          models::depth_coordinate_v2::capture_provenance_binding
                        }},
            {"depth_model", model_identity.depth_model},
            {"depth_model_url", model_identity.depth_model_url},
            {"onnx_sha256", model_identity.onnx_sha256},
            {"preprocess_profile", model_identity.preprocess_profile},
            {"preprocess_source_closure_sha256",
             model_identity.preprocess_source_closure_sha256},
            {"raw_depth_sha256", raw_depth_sha256},
            {"model_input_sha256", model_input_sha256},
            {"model_input_shape_sha256", model_input_shape_sha256},
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
             << "legacy_normalization_available="
             << (adaptive_available ? "true" : "false") << '\n'
             << "normalization_effective_lower=" << normalization.lower << '\n'
             << "normalization_effective_upper=" << normalization.upper << '\n'
             << "normalization_initialized=" << normalization.initialized << '\n'
             << "normalization_frame_state=" << normalization.frame_state << '\n'
             << "cuda_graph_active="
             << (completed.cuda_graph_active ? "true" : "false") << '\n'
             << "warp_depth_prefilter_applied=false\n"
             << "warp_map_available=" << (warp_map_available ? "true" : "false")
             << '\n'
             << "warp_mask_available=" << (warp_mask_available ? "true" : "false")
             << '\n'
             << "parallax_v2_shadow_requested="
             << "false\n"
             << "parallax_v2_shadow_active=true\n"
             << "parallax_v2_render_requested=true\n"
             << "parallax_v2_render_selected=true\n"
             << "renderer_authority=authenticated-parallax-v2-final-near-preserving-anisotropic-2d-majorant\n"
             << "parallax_v2_live_renderer_source_closure_sha256="
             << completed.parallax_v2_live_renderer_source_closure_sha256 << '\n'
             << "raw_model_provenance=authoritative\n"
             << "raw_model_preprocess_profile="
             << model_identity.preprocess_profile
             << '\n'
             << "raw_model_preprocess_source_closure_sha256="
             << model_identity.preprocess_source_closure_sha256
             << '\n';
        const auto &shader_identity = *completed.parallax_v2_shader_provenance;
        meta << "parallax_v2_shader_source_closure_schema="
             << shader_identity.source_closure_schema << '\n'
             << "parallax_v2_shader_source_compile_flags="
             << shader_identity.source_compile_flags << '\n'
             << "parallax_v2_shader_source_macro_count="
             << shader_identity.source_macro_count << '\n'
             << "parallax_v2_shader_source_closure_sha256="
             << shader_identity.source_closure_sha256 << '\n';
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
