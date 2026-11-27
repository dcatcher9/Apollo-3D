// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file tests/unit/platform/test_debug_dump_png.cpp
 * @brief The PNG encoder shared by Dump 3D and the Game 3D dump previews.
 */
#include "src/platform/windows/debug_dump_png.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <wincodec.h>
#include <windows.h>
#include <wrl/client.h>
#include <zlib.h>

namespace {
  namespace debug_dump = platf::debug_dump;

  struct temporary_directory {
    std::filesystem::path path = std::filesystem::temp_directory_path() / ("sunshine-dump-png-test-" + std::to_string(GetCurrentProcessId()) + "-" + std::to_string(GetTickCount64()));

    temporary_directory() {
      std::filesystem::create_directories(path);
    }

    ~temporary_directory() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  };

  std::vector<std::uint8_t> read_file(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), {}};
  }

  // WIC shares no code with zlib or the encoder; it returns tightly packed RGB.
  std::vector<std::uint8_t> decode_rgb(const std::filesystem::path &path, unsigned &width, unsigned &height) {
    const auto initialized = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    Microsoft::WRL::ComPtr<IWICImagingFactory> factory;
    Microsoft::WRL::ComPtr<IWICBitmapDecoder> decoder;
    Microsoft::WRL::ComPtr<IWICBitmapFrameDecode> frame;
    Microsoft::WRL::ComPtr<IWICFormatConverter> converter;
    width = height = 0;
    const bool ok = SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory))) &&
                    SUCCEEDED(factory->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnLoad, &decoder)) &&
                    SUCCEEDED(decoder->GetFrame(0, &frame)) && SUCCEEDED(frame->GetSize(&width, &height)) &&
                    SUCCEEDED(factory->CreateFormatConverter(&converter)) &&
                    SUCCEEDED(converter->Initialize(frame.Get(), GUID_WICPixelFormat24bppRGB, WICBitmapDitherTypeNone, nullptr, 0, WICBitmapPaletteTypeCustom));
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * 3);
    const bool copied = ok && SUCCEEDED(converter->CopyPixels(nullptr, width * 3, static_cast<UINT>(pixels.size()), pixels.data()));
    converter.Reset();
    frame.Reset();
    decoder.Reset();
    factory.Reset();
    if (SUCCEEDED(initialized)) {
      CoUninitialize();
    }
    if (!copied) {
      throw std::runtime_error("Cannot decode test PNG");
    }
    return pixels;
  }

  std::uint32_t be32(const std::uint8_t *bytes) {
    return (std::uint32_t(bytes[0]) << 24) | (std::uint32_t(bytes[1]) << 16) | (std::uint32_t(bytes[2]) << 8) | bytes[3];
  }
}  // namespace

TEST(DebugDumpPng, EveryByteValueRoundTripsThroughAnIndependentDecoder) {
  temporary_directory directory;
  // An odd width leaves rows unaligned; the first row walks all 256 values in each channel.
  constexpr std::uint32_t width = 259, height = 7;
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3);
  std::mt19937 random(20261127);
  for (std::size_t i = 0; i < rgb.size(); ++i) {
    rgb[i] = i < width * 3 ? static_cast<std::uint8_t>(i / 3 + i % 3 * 85) : static_cast<std::uint8_t>(random());
  }
  const auto packed = directory.path / "packed.png";
  ASSERT_TRUE(debug_dump::write_png(packed, width, height, rgb));
  unsigned decoded_width = 0, decoded_height = 0;
  const auto decoded = decode_rgb(packed, decoded_width, decoded_height);
  EXPECT_EQ(decoded_width, width);
  EXPECT_EQ(decoded_height, height);
  ASSERT_EQ(decoded.size(), rgb.size());
  for (std::size_t i = 0; i < rgb.size(); ++i) {
    ASSERT_EQ(decoded[i], rgb[i]) << "byte " << i << " (x " << i / 3 % width << ", y " << i / 3 / width << ", channel " << i % 3 << ")";
  }

  // The row writer is the same encoder: rows requested top to bottom, the same file bytes.
  const auto rows = directory.path / "rows.png";
  std::uint32_t next_row = 0;
  ASSERT_TRUE(debug_dump::write_png(rows, width, height, [&](std::uint32_t y, std::uint8_t *row) {
    EXPECT_EQ(y, next_row++);
    std::copy_n(rgb.begin() + static_cast<std::ptrdiff_t>(y) * width * 3, width * 3, row);
  }));
  EXPECT_EQ(next_row, height);
  EXPECT_EQ(read_file(rows), read_file(packed));
}

TEST(DebugDumpPng, WritesOneFilterZeroIdatFromBestSpeedCompress2) {
  temporary_directory directory;
  constexpr std::uint32_t width = 3, height = 2;
  const std::vector<std::uint8_t> rgb {0, 1, 2, 250, 251, 252, 9, 8, 7, 255, 0, 255, 128, 64, 32, 17, 34, 51};
  const auto path = directory.path / "small.png";
  ASSERT_TRUE(debug_dump::write_png(path, width, height, rgb));
  const auto file = read_file(path);

  constexpr std::array<std::uint8_t, 8> signature {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
  ASSERT_GE(file.size(), signature.size());
  EXPECT_TRUE(std::equal(signature.begin(), signature.end(), file.begin()));
  std::vector<std::string> types;
  std::vector<std::vector<std::uint8_t>> payloads;
  for (std::size_t offset = signature.size(); offset < file.size();) {
    ASSERT_LE(offset + 12, file.size());
    const auto size = be32(file.data() + offset);
    ASSERT_LE(offset + 12 + size, file.size());
    const auto *type = file.data() + offset + 4;
    const auto crc = crc32(crc32(0, type, 4), type + 4, size);
    EXPECT_EQ(be32(type + 4 + size), static_cast<std::uint32_t>(crc));
    types.emplace_back(reinterpret_cast<const char *>(type), 4);
    payloads.emplace_back(type + 4, type + 4 + size);
    offset += 12 + size;
  }
  ASSERT_EQ(types, (std::vector<std::string> {"IHDR", "IDAT", "IEND"}));
  // 8-bit truecolor RGB, deflate, adaptive filtering method 0, no interlace.
  const std::vector<std::uint8_t> header {0, 0, 0, width, 0, 0, 0, height, 8, 2, 0, 0, 0};
  EXPECT_EQ(payloads[0], header);
  EXPECT_TRUE(payloads[2].empty());

  // One zlib stream of filter-0 scanlines at Z_BEST_SPEED, so equal pixels give equal bytes.
  std::vector<std::uint8_t> raw;
  for (std::uint32_t y = 0; y < height; ++y) {
    raw.push_back(0);
    raw.insert(raw.end(), rgb.begin() + y * width * 3, rgb.begin() + (y + 1) * width * 3);
  }
  uLongf compressed_size = compressBound(static_cast<uLong>(raw.size()));
  std::vector<std::uint8_t> compressed(compressed_size);
  ASSERT_EQ(compress2(compressed.data(), &compressed_size, raw.data(), static_cast<uLong>(raw.size()), Z_BEST_SPEED), Z_OK);
  compressed.resize(compressed_size);
  EXPECT_EQ(payloads[1], compressed);

  ASSERT_TRUE(debug_dump::write_png(path, width, height, rgb));
  EXPECT_EQ(read_file(path), file);
}

TEST(DebugDumpPng, RefusesEmptyOrMismatchedImagesWithoutWritingAFile) {
  temporary_directory directory;
  const auto path = directory.path / "refused.png";
  EXPECT_FALSE(debug_dump::write_png(path, 0, 1, std::vector<std::uint8_t> {}));
  EXPECT_FALSE(debug_dump::write_png(path, 1, 0, std::vector<std::uint8_t> {}));
  EXPECT_FALSE(debug_dump::write_png(path, 2, 2, std::vector<std::uint8_t>(11)));
  bool called = false;
  EXPECT_FALSE(debug_dump::write_png(path, 4, 0, [&](std::uint32_t, std::uint8_t *) {
    called = true;
  }));
  EXPECT_FALSE(called);
  EXPECT_FALSE(std::filesystem::exists(path));
  EXPECT_FALSE(debug_dump::write_png(directory.path / "missing" / "refused.png", 1, 1, std::vector<std::uint8_t>(3)));
}
