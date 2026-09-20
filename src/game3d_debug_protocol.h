// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

// Optional, independent of the streaming SBS ABI. One immutable diagnostic batch may
// be outstanding. Publish request_id/response_id/released_id with Interlocked operations.
namespace game3d_debug {
  inline constexpr std::uint64_t magic = 0x31504D5544334753ULL;
  inline constexpr std::uint32_t version = 3;
  inline constexpr wchar_t mapping_prefix[] = L"Local\\Sunshine3D.ReShade.Dump.";
  inline constexpr std::uint32_t max_textures = 40;
  inline constexpr std::uint32_t max_json_bytes = 256 * 1024;
  inline constexpr std::uint32_t max_dimension = 16384;
  inline constexpr std::uint64_t max_capture_bytes = 768ULL * 1024 * 1024;
  // Out-of-band reporting survives even an already-full primary JSON envelope.
  inline constexpr std::uint32_t optional_metadata_omitted = 1u;

  enum class status : std::uint32_t {
    complete = 1,
    unavailable = 2,
    failed = 3
  };
  enum class artifact : std::uint32_t {
    source_color = 1,
    raw_depth = 2,
    candidate = 3,
    vertical_majorant = 4,
    vertical_field = 5,
    final_field = 6,
    sbs = 7,
    linear_color = 8,
    // Exact RGBA input consumed for UI alpha; optional catalog occupies 9..32.
    ui_source_color = 33,
  };

  struct texture_t {
    artifact kind {};
    std::uint32_t width = 0, height = 0, dxgi_format = 0;
    std::uint64_t handle = 0;
  };

  struct response_t {
    std::uint64_t consumer_nonce = 0;
    std::uint64_t capture_id = 0, capture_qpc = 0;
    std::uint64_t runtime_epoch = 0, export_generation = 0, export_sequence = 0;
    status result = status::failed;
    std::uint32_t texture_count = 0, json_bytes = 0, flags = 0;
    texture_t textures[max_textures] {};
  };

  struct alignas(64) shared_state_t {
    std::uint64_t signature = magic;
    std::uint32_t protocol_version = version, shared_bytes = sizeof(shared_state_t);
    std::uint32_t producer_pid = 0, consumer_pid = 0;
    std::uint64_t producer_creation_time = 0, consumer_creation_time = 0;
    std::uint64_t consumer_nonce = 0;
    // Consumer fills its identity, then publishes a fresh nonzero request_id.
    std::uint64_t request_id = 0;
    // Producer publishes only after ALL snapshot GPU writes have completed; no CPU wait.
    // Handles/JSON remain immutable until released_id acknowledges this response.
    std::uint64_t response_id = 0;
    // Consumer may release after opening and retaining every shared texture. Textures
    // are never overwritten/reused, so its COM references preserve in-flight copies.
    // Releasing before a response cancels that request; producer still retires GPU work safely.
    std::uint64_t released_id = 0;
    response_t response;
    char json[max_json_bytes] {};
  };

  static_assert(std::is_standard_layout_v<shared_state_t> && std::is_trivially_copyable_v<shared_state_t>);
  static_assert(offsetof(shared_state_t, request_id) % 8 == 0);
  static_assert(offsetof(shared_state_t, response_id) % 8 == 0);
  static_assert(offsetof(shared_state_t, released_id) % 8 == 0);
}  // namespace game3d_debug
