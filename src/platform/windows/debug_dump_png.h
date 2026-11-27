/**
 * @file src/platform/windows/debug_dump_png.h
 * @brief The PNG encoder shared by the host's Dump 3D and Game 3D dump previews.
 *
 * Header-only so the stand-alone Game 3D preview tools need only zlib. The Game 3D add-on's own
 * dump (tools/reshade) is a separate DLL and does not use this header.
 */
#pragma once

// standard includes
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <vector>

// lib includes
#include <zlib.h>

namespace platf::debug_dump {

  /** Fills the `width * 3` RGB bytes of PNG row `y`; rows are requested top to bottom. */
  using png_row_writer = std::function<void(std::uint32_t y, std::uint8_t *rgb)>;

  /**
   * Write an 8-bit RGB PNG: filter 0 on every row and one IDAT from compress2 at Z_BEST_SPEED,
   * so the same pixels always give the same bytes. Returns false for an empty or oversized image
   * or a failed write.
   */
  inline bool write_png(
    const std::filesystem::path &path,
    const std::uint32_t width,
    const std::uint32_t height,
    const png_row_writer &write_row
  ) {
    if (
      width == 0 || height == 0 ||
      static_cast<std::uint64_t>(width) * height >
        std::numeric_limits<std::size_t>::max() / 3u
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
      write_row(y, raw.data() + output_offset + 1u);
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

  /** write_png() from tightly packed, top-to-bottom RGB bytes (`width * height * 3`). */
  inline bool write_png(
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
    const std::size_t row_bytes = static_cast<std::size_t>(width) * 3u;
    return write_png(path, width, height, [&](const std::uint32_t y, std::uint8_t *row) {
      std::memcpy(row, rgb.data() + static_cast<std::size_t>(y) * row_bytes, row_bytes);
    });
  }

}  // namespace platf::debug_dump
