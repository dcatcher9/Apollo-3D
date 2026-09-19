// SPDX-License-Identifier: GPL-3.0-only
#pragma once

// Private fixture IPC, unrelated to the production ReShade handoff protocol.
// Pixels here have already passed through the child's unchanged production receiver.
#include <cstddef>
#include <cstdint>
#include <windows.h>

namespace receiver_process_test {
  constexpr std::uint64_t magic = 0x5353425354505243ull;
  constexpr std::uint32_t version = 1;
  constexpr std::size_t pixel_offset = 4096;
  constexpr std::uint32_t max_source_dimension = 1024;
  constexpr DWORD command_timeout_ms = 3000;

  enum class command : std::uint32_t {
    poll = 1,
    focus = 2,
    quit = 3
  };

  struct alignas(64) mailbox {
    std::uint64_t signature = magic;
    std::uint32_t protocol_version = version;
    DWORD parent_pid = 0;
    std::uint64_t parent_creation_time = 0;
    std::uint64_t nonce[2] {};
    std::uint64_t window = 0;
    std::uint64_t adapter_luid = 0;
    std::uint64_t mapping_bytes = 0;
    RECT source {};
    DWORD child_pid = 0;
    std::uint64_t request_id = 0;
    std::uint64_t response_id = 0;
    command operation = command::poll;
    BOOL focused = TRUE;
    int result = -1;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t format = 0;
    std::uint32_t row_pitch = 0;
    std::uint32_t linear = 0;
    std::uint64_t sequence = 0;
    std::int64_t timestamp_ns = 0;
    char error[256] {};
  };

  static_assert(sizeof(mailbox) <= pixel_offset);
}  // namespace receiver_process_test
