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
  // The native whole-clip worker already caps retained timing evidence at 128 MiB. Apply the
  // same bound at this standalone file boundary so a malformed or sparse sidecar cannot make the
  // harness allocate an attacker-controlled amount of memory before it validates the header.
  inline constexpr std::uint64_t max_payload_bytes = 128ull * 1024ull * 1024ull;
  inline constexpr std::uint64_t max_timestamp_count =
    max_payload_bytes / sizeof(std::uint64_t);
  inline constexpr std::size_t io_buffer_bytes = 64u * 1024u;
  static_assert(io_buffer_bytes % sizeof(std::uint64_t) == 0u);

  inline void write_u32_le(std::uint8_t *bytes, const std::uint32_t value) {
    for (unsigned shift = 0u; shift < 32u; shift += 8u) {
      bytes[shift / 8u] = static_cast<std::uint8_t>(value >> shift);
    }
  }

  inline void write_u64_le(std::uint8_t *bytes, const std::uint64_t value) {
    for (unsigned shift = 0u; shift < 64u; shift += 8u) {
      bytes[shift / 8u] = static_cast<std::uint8_t>(value >> shift);
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
    error.clear();
    if (!valid_timestamps(timestamps) ||
        timestamps.size() > max_timestamp_count) {
      error = "observation timeline timestamps are empty, zero, regressed, or too large";
      return false;
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
      error = "could not open observation timeline for writing";
      return false;
    }
    std::array<std::uint8_t, header_bytes> header {};
    std::copy(magic.begin(), magic.end(), header.begin());
    write_u32_le(header.data() + 8u, schema);
    write_u32_le(header.data() + 12u, header_bytes);
    write_u64_le(header.data() + 16u, static_cast<std::uint64_t>(timestamps.size()));
    stream.write(
      reinterpret_cast<const char *>(header.data()),
      static_cast<std::streamsize>(header.size())
    );
    std::array<std::uint8_t, io_buffer_bytes> buffer {};
    for (std::size_t begin = 0u; begin < timestamps.size() && stream; ) {
      const std::size_t count = std::min(
        timestamps.size() - begin,
        buffer.size() / sizeof(std::uint64_t)
      );
      for (std::size_t index = 0u; index < count; ++index) {
        write_u64_le(
          buffer.data() + index * sizeof(std::uint64_t),
          timestamps[begin + index]
        );
      }
      stream.write(
        reinterpret_cast<const char *>(buffer.data()),
        static_cast<std::streamsize>(count * sizeof(std::uint64_t))
      );
      begin += count;
    }
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
    error.clear();
    timestamps.clear();
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
    const std::uint64_t maximum_file_bytes =
      header_bytes + max_timestamp_count * sizeof(std::uint64_t);
    if (byte_count > maximum_file_bytes) {
      error = "observation timeline is too large";
      return false;
    }
    stream.seekg(0, std::ios::beg);
    std::array<std::uint8_t, header_bytes> header {};
    stream.read(
      reinterpret_cast<char *>(header.data()),
      static_cast<std::streamsize>(header.size())
    );
    if (!stream || !std::equal(magic.begin(), magic.end(), header.begin()) ||
        read_u32_le(header.data() + 8u) != schema ||
        read_u32_le(header.data() + 12u) != header_bytes) {
      error = "observation timeline header is invalid";
      return false;
    }
    const auto count = read_u64_le(header.data() + 16u);
    if (count == 0u || count > max_timestamp_count ||
        header_bytes + count * sizeof(std::uint64_t) != byte_count ||
        count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      error = "observation timeline length is invalid";
      return false;
    }
    timestamps.resize(static_cast<std::size_t>(count));
    std::array<std::uint8_t, io_buffer_bytes> buffer {};
    for (std::size_t begin = 0u; begin < timestamps.size(); ) {
      const std::size_t chunk_count = std::min(
        timestamps.size() - begin,
        buffer.size() / sizeof(std::uint64_t)
      );
      const std::size_t chunk_bytes = chunk_count * sizeof(std::uint64_t);
      stream.read(
        reinterpret_cast<char *>(buffer.data()),
        static_cast<std::streamsize>(chunk_bytes)
      );
      if (!stream) {
        error = "could not read observation timeline payload";
        timestamps.clear();
        return false;
      }
      for (std::size_t index = 0u; index < chunk_count; ++index) {
        timestamps[begin + index] = read_u64_le(
          buffer.data() + index * sizeof(std::uint64_t)
        );
      }
      begin += chunk_count;
    }
    if (!valid_timestamps(timestamps)) {
      error = "observation timeline timestamps are zero or regressed";
      timestamps.clear();
      return false;
    }
    return true;
  }
}
