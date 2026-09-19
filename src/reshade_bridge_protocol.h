// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

// Shared by the ReShade add-on and the Windows receiver. This is a versioned Windows IPC
// layout, not a C++ object ABI: use Interlocked operations for ownership/control fields.
namespace reshade_bridge {
  inline constexpr std::uint64_t magic = 0x3153425353485353ULL;
  inline constexpr std::uint32_t version = 1;
  inline constexpr std::uint32_t slot_count = 3;
  inline constexpr wchar_t mapping_prefix[] = L"Local\\Sunshine3D.ReShade.SBS.";
  inline constexpr std::uint32_t max_source_width = 8192;
  inline constexpr std::uint32_t max_source_height = 8192;
  inline constexpr std::uint32_t max_packed_width = 16384;

  enum class slot_state : std::uint32_t {
    free = 0,
    writing = 1,
    ready = 2,
    reading = 3,
  };

  inline constexpr std::uint64_t max_generation = UINT64_MAX >> 2;

  [[nodiscard]] constexpr std::uint64_t slot_control(std::uint64_t generation, slot_state state) {
    return (generation << 2) | static_cast<std::uint64_t>(state);
  }

  [[nodiscard]] constexpr slot_state control_state(std::uint64_t control) {
    return static_cast<slot_state>(control & 3);
  }

  [[nodiscard]] constexpr std::uint64_t control_generation(std::uint64_t control) {
    return control >> 2;
  }

  enum class transfer : std::uint32_t {
    srgb = 1,
    scrgb = 2,  // Linear Rec.709 primaries; 1.0 = 80 cd/m2.
  };

  enum class layout : std::uint32_t {
    full_sbs_left_first = 1,
  };

  struct metadata_t {
    std::uint64_t signature = magic;
    std::uint32_t protocol_version = version;
    std::uint32_t metadata_bytes = sizeof(metadata_t);
    std::uint32_t producer_pid = 0;
    std::uint32_t source_width = 0;
    std::uint32_t source_height = 0;
    std::uint32_t packed_width = 0;
    std::uint32_t packed_height = 0;
    std::uint32_t dxgi_format = 0;
    transfer color_transfer = transfer::srgb;
    layout image_layout = layout::full_sbs_left_first;
    std::uint64_t producer_creation_time = 0;
    std::uint64_t window = 0;
    std::uint64_t adapter_luid = 0;
    std::uint64_t generation = 0;
    std::uint64_t accepted_consumer_nonce = 0;
    std::uint64_t texture_handles[slot_count] {};
    std::uint64_t ready_fence_handle = 0;
  };

  struct alignas(64) slot_t {
    // Generation and state change in one CAS. Retiring consumers cannot unlock a new ring.
    std::uint64_t control = 0;
    std::uint64_t sequence = 0;
    std::uint64_t qpc = 0;
  };

  struct alignas(64) shared_state_t {
    // Producer is the single metadata writer: odd during replacement, even when readable.
    std::uint32_t metadata_sequence = 0;
    std::uint32_t shared_bytes = sizeof(shared_state_t);
    // A new receiver writes a nonzero nonce once. The producer creates NEW resources for it;
    // resources with abandoned reading slots are never reused across consumer lifetimes.
    std::uint64_t consumer_nonce = 0;
    metadata_t metadata;
    slot_t slots[slot_count];
  };

  [[nodiscard]] constexpr bool supported_format(std::uint32_t format, transfer color) {
    // DXGI_FORMAT_R8G8B8A8_UNORM, B8G8R8A8_UNORM, R10G10B10A2_UNORM, R16G16B16A16_FLOAT.
    return color == transfer::srgb ? (format == 28 || format == 87 || format == 24 || format == 10) :
                                     color == transfer::scrgb && format == 10;
  }

  [[nodiscard]] constexpr bool valid_metadata(const metadata_t &m) {
    if (m.signature != magic || m.protocol_version != version || m.metadata_bytes != sizeof(metadata_t) || m.producer_pid == 0 || m.producer_creation_time == 0 || m.window == 0 || m.generation == 0 || m.generation > max_generation || m.accepted_consumer_nonce == 0 || m.source_width == 0 || m.source_width > max_source_width || m.source_height == 0 || m.source_height > max_source_height || m.packed_width != m.source_width * 2 || m.packed_width > max_packed_width || m.packed_height != m.source_height || m.packed_width % 4 != 0 || m.packed_height % 2 != 0 || m.image_layout != layout::full_sbs_left_first || !supported_format(m.dxgi_format, m.color_transfer) || m.ready_fence_handle == 0) {
      return false;
    }
    for (auto handle : m.texture_handles) {
      if (handle == 0) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] constexpr bool matches_output(const metadata_t &m, std::uint32_t width, std::uint32_t height) {
    // Source color is explicit metadata, independent of the display/stream's HDR setting.
    // The final presenter or encoder converts this source to its negotiated output color.
    return valid_metadata(m) && m.packed_width == width && m.packed_height == height;
  }

  static_assert(std::is_standard_layout_v<shared_state_t> && std::is_trivially_copyable_v<shared_state_t>);
  static_assert(sizeof(slot_t) == 64);
  static_assert(offsetof(shared_state_t, consumer_nonce) % 8 == 0);
  static_assert(offsetof(shared_state_t, slots) % 64 == 0);
}  // namespace reshade_bridge
