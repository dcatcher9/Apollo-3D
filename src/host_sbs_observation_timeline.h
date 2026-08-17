/**
 * @file src/host_sbs_observation_timeline.h
 * @brief Exact source-observation timestamps shared by offline Host SBS callers.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace models::host_sbs_observation_timeline {
  inline constexpr std::array<std::uint8_t, 8> magic {
    'S', 'B', 'S', 'O', 'T', 'L', '1', 0,
  };
  inline constexpr std::uint32_t schema = 1u;
  inline constexpr std::uint32_t header_bytes = 24u;

  inline void append_u32_le(std::vector<std::uint8_t> &bytes, const std::uint32_t value) {
    for (unsigned shift = 0u; shift < 32u; shift += 8u) {
      bytes.push_back(static_cast<std::uint8_t>(value >> shift));
    }
  }

  inline void append_u64_le(std::vector<std::uint8_t> &bytes, const std::uint64_t value) {
    for (unsigned shift = 0u; shift < 64u; shift += 8u) {
      bytes.push_back(static_cast<std::uint8_t>(value >> shift));
    }
  }

  inline std::uint32_t read_u32_le(const std::uint8_t *bytes) {
    std::uint32_t value = 0u;
    for (unsigned index = 0u; index < 4u; ++index) {
      value |= static_cast<std::uint32_t>(bytes[index]) << (index * 8u);
    }
    return value;
  }

  inline std::uint64_t read_u64_le(const std::uint8_t *bytes) {
    std::uint64_t value = 0u;
    for (unsigned index = 0u; index < 8u; ++index) {
      value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8u);
    }
    return value;
  }

  inline bool valid_timestamps(const std::span<const std::uint64_t> timestamps) {
    if (timestamps.empty() || timestamps.front() == 0u) {
      return false;
    }
    for (std::size_t index = 1u; index < timestamps.size(); ++index) {
      if (timestamps[index] == 0u || timestamps[index] < timestamps[index - 1u]) {
        return false;
      }
    }
    return true;
  }

  inline bool write(
    const std::filesystem::path &path,
    const std::span<const std::uint64_t> timestamps,
    std::string &error
  ) {
    if (!valid_timestamps(timestamps) ||
        timestamps.size() >
          (std::numeric_limits<std::size_t>::max() - header_bytes) / sizeof(std::uint64_t)) {
      error = "observation timeline timestamps are empty, zero, regressed, or too large";
      return false;
    }
    std::vector<std::uint8_t> bytes;
    bytes.reserve(header_bytes + timestamps.size() * sizeof(std::uint64_t));
    bytes.insert(bytes.end(), magic.begin(), magic.end());
    append_u32_le(bytes, schema);
    append_u32_le(bytes, header_bytes);
    append_u64_le(bytes, static_cast<std::uint64_t>(timestamps.size()));
    for (const auto timestamp : timestamps) {
      append_u64_le(bytes, timestamp);
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(
      reinterpret_cast<const char *>(bytes.data()),
      static_cast<std::streamsize>(bytes.size())
    );
    stream.flush();
    if (!stream.good()) {
      error = "could not write observation timeline";
      return false;
    }
    stream.close();
    if (!stream.good()) {
      error = "could not close observation timeline";
      return false;
    }
    return true;
  }

  inline bool read(
    const std::filesystem::path &path,
    std::vector<std::uint64_t> &timestamps,
    std::string &error
  ) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
      error = "could not open observation timeline";
      return false;
    }
    stream.seekg(0, std::ios::end);
    const auto end = stream.tellg();
    if (end < static_cast<std::streamoff>(header_bytes)) {
      error = "observation timeline is shorter than its header";
      return false;
    }
    const auto byte_count = static_cast<std::uint64_t>(end);
    if (byte_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      error = "observation timeline is too large";
      return false;
    }
    stream.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(byte_count));
    stream.read(
      reinterpret_cast<char *>(bytes.data()),
      static_cast<std::streamsize>(bytes.size())
    );
    if (!stream || !std::equal(magic.begin(), magic.end(), bytes.begin()) ||
        read_u32_le(bytes.data() + 8u) != schema ||
        read_u32_le(bytes.data() + 12u) != header_bytes) {
      error = "observation timeline header is invalid";
      return false;
    }
    const auto count = read_u64_le(bytes.data() + 16u);
    if (count == 0u || count >
          (std::numeric_limits<std::uint64_t>::max() - header_bytes) /
            sizeof(std::uint64_t) ||
        header_bytes + count * sizeof(std::uint64_t) != byte_count ||
        count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      error = "observation timeline length is invalid";
      return false;
    }
    timestamps.resize(static_cast<std::size_t>(count));
    for (std::size_t index = 0u; index < timestamps.size(); ++index) {
      timestamps[index] = read_u64_le(
        bytes.data() + header_bytes + index * sizeof(std::uint64_t)
      );
    }
    if (!valid_timestamps(timestamps)) {
      error = "observation timeline timestamps are zero or regressed";
      timestamps.clear();
      return false;
    }
    return true;
  }
}
