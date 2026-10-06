// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

// Shared by the ReShade add-on and the Windows receiver. This is a versioned Windows IPC
// layout, not a C++ object ABI: use Interlocked operations for ownership/control fields.
namespace reshade_bridge {
  inline constexpr std::uint64_t magic = 0x3153425353485353ULL;
  // Protocol 4: four export slots (a 448-byte shared state with 128-byte metadata) and every
  // transfer, PQ included. Protocols 1-3 had three slots (384 and 120 bytes). Producer and
  // consumer accept exactly this version and layout and refuse any other by name: the add-on and
  // the host are installed together, and neither may read the other's layout of another release.
  inline constexpr std::uint32_t version = 4;
  inline constexpr std::uint32_t slot_count = 4;
  inline constexpr wchar_t mapping_prefix[] = L"Local\\Sunshine3D.ReShade.SBS.";
  inline constexpr std::uint32_t max_source_width = 8192;
  inline constexpr std::uint32_t max_source_height = 8192;
  inline constexpr std::uint32_t max_packed_width = 16384;
  // Signed displacement of one eye in source-eye UV. The producer publishes the actual
  // rendered UI displacement; the receiver never recomputes it from scene parameters.
  inline constexpr float maximum_ui_parallax_uv = 0.04f;
  inline constexpr std::uint32_t cursor_plane_present = 1u;
  // Consumer capability bits, valid only for the consumer nonce they were written with.
  inline constexpr std::uint32_t consumer_accepts_pq = 1u;
  // The consumer encodes HDR10 PQ: pack any HDR source, scRGB included, as R10G10B10A2 PQ.
  inline constexpr std::uint32_t consumer_stream_pq = 2u;

  [[nodiscard]] constexpr bool valid_ui_parallax(float value) {
    // Ordered comparisons also reject NaN and either infinity.
    return value >= -maximum_ui_parallax_uv && value <= maximum_ui_parallax_uv;
  }

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
    // Rec.2020 primaries with SMPTE ST 2084 code values; code 1.0 = 10000 cd/m2.
    // Valid only as R10G10B10A2_UNORM, and only for a consumer that accepts it.
    pq = 3,
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
    // The cursor plane of this exact texture, with the same writing/ready/reading ownership as
    // the texture and sequence.
    std::uint32_t cursor_plane_flags = 0;
    float ui_parallax_uv = 0.0f;
  };

  [[nodiscard]] constexpr bool read_ui_parallax(std::uint32_t protocol_version, const slot_t &slot, float &value) {
    value = 0.0f;
    if (protocol_version != version || (slot.cursor_plane_flags != 0u && slot.cursor_plane_flags != cursor_plane_present) || !valid_ui_parallax(slot.ui_parallax_uv) || (slot.cursor_plane_flags == 0u && slot.ui_parallax_uv != 0.0f)) {
      return false;
    }
    if (slot.cursor_plane_flags == cursor_plane_present) {
      value = slot.ui_parallax_uv;
    }
    return true;
  }

  struct alignas(64) shared_state_t {
    // Producer is the single metadata writer: odd during replacement, even when readable.
    std::uint32_t metadata_sequence = 0;
    std::uint32_t shared_bytes = sizeof(shared_state_t);
    // A new receiver writes a nonzero nonce once and the producer answers with a new
    // generation. Its textures are new unless the previous ring has been idle (no receiver and
    // no producer GPU work) for at least 500 ms, long after any read an abandoned receiver left
    // in flight; the receiver then reopens the same shared handles under the new generation.
    // A detaching receiver resets its own nonce to zero; zero means no receiver.
    std::uint64_t consumer_nonce = 0;
    metadata_t metadata;
    // A consumer's declaration: it writes consumer_capabilities and consumer_protocol, then
    // capability_nonce = its nonce, then consumer_nonce (each a full barrier). A producer honours
    // them only when capability_nonce equals the nonce it answers, so a replaced consumer's
    // declaration never applies to another consumer's request.
    std::uint64_t capability_nonce = 0;
    std::uint32_t consumer_capabilities = 0;
    // The protocol the consumer speaks. A producer answers only a consumer that declared its
    // own `version`, and names any other.
    std::uint32_t consumer_protocol = 0;
    slot_t slots[slot_count];
  };

  // The capabilities a producer may use while answering `answered_nonce`.
  [[nodiscard]] constexpr std::uint32_t answered_capabilities(std::uint64_t answered_nonce, std::uint64_t capability_nonce, std::uint32_t capabilities) {
    return answered_nonce != 0 && capability_nonce == answered_nonce ? capabilities : 0u;
  }

  // The protocol the consumer of `answered_nonce` declared, zero when none was declared for it.
  [[nodiscard]] constexpr std::uint32_t answered_protocol(std::uint64_t answered_nonce, std::uint64_t capability_nonce, std::uint32_t protocol) {
    return answered_nonce != 0 && capability_nonce == answered_nonce ? protocol : 0u;
  }

  // Every protocol, 1 to 4, begins with the same header: shared_bytes, then the metadata's
  // signature, protocol_version and metadata_bytes. A consumer reads it before anything else and
  // refuses any other layout or version by name.
  [[nodiscard]] constexpr bool same_protocol(std::uint32_t shared_bytes, std::uint64_t signature, std::uint32_t protocol, std::uint32_t metadata_bytes) {
    return shared_bytes == sizeof(shared_state_t) && signature == magic && protocol == version && metadata_bytes == sizeof(metadata_t);
  }

  [[nodiscard]] constexpr bool supported_format(std::uint32_t format, transfer color) {
    // DXGI_FORMAT_R8G8B8A8_UNORM, B8G8R8A8_UNORM, R10G10B10A2_UNORM, R16G16B16A16_FLOAT.
    return color == transfer::srgb  ? (format == 28 || format == 87 || format == 24 || format == 10) :
           color == transfer::scrgb ? format == 10 :
                                      color == transfer::pq && format == 24;
  }

  // The source a producer describes whether or not a consumer is attached: its identity, eye
  // raster, format and transfer. Without a consumer the producer publishes exactly this, with
  // generation, nonce and handles zero, and allocates no ring (a status-only observer).
  [[nodiscard]] constexpr bool valid_source_metadata(const metadata_t &m) {
    return m.signature == magic && m.protocol_version == version && m.metadata_bytes == sizeof(metadata_t) && m.producer_pid != 0 && m.producer_creation_time != 0 && m.window != 0 && m.generation <= max_generation && m.source_width != 0 && m.source_width <= max_source_width && m.source_height != 0 && m.source_height <= max_source_height && m.packed_width == m.source_width * 2 && m.packed_width <= max_packed_width && m.packed_height == m.source_height && m.packed_width % 4 == 0 && m.packed_height % 2 == 0 && m.image_layout == layout::full_sbs_left_first && supported_format(m.dxgi_format, m.color_transfer);
  }

  [[nodiscard]] constexpr bool valid_metadata(const metadata_t &m) {
    if (!valid_source_metadata(m) || m.generation == 0 || m.accepted_consumer_nonce == 0 || m.ready_fence_handle == 0) {
      return false;
    }
    for (auto handle : m.texture_handles) {
      if (handle == 0) {
        return false;
      }
    }
    return true;
  }

  // How the authored eyes map onto a requested packed output. Both halves scale by the same
  // factor, so a matching aspect ratio (within 0.5%) keeps each eye undistorted; a different
  // aspect would stretch disparity and is never scaled.
  enum class output_fit { exact, scaled, aspect_mismatch };
  [[nodiscard]] constexpr output_fit fit_output(const metadata_t &m, std::uint32_t width, std::uint32_t height) {
    if (m.packed_width == width && m.packed_height == height) {
      return output_fit::exact;
    }
    if (width < 2 || width % 2 || height == 0 || m.packed_width == 0 || m.packed_height == 0) {
      return output_fit::aspect_mismatch;
    }
    const std::uint64_t authored = std::uint64_t(m.packed_width) * height;
    const std::uint64_t requested = std::uint64_t(width) * m.packed_height;
    const std::uint64_t difference = authored > requested ? authored - requested : requested - authored;
    return difference * 200 <= (authored < requested ? authored : requested) ? output_fit::scaled : output_fit::aspect_mismatch;
  }

  static_assert(std::is_standard_layout_v<shared_state_t> && std::is_trivially_copyable_v<shared_state_t>);
  // The header every protocol shares (same_protocol) never moves.
  static_assert(offsetof(shared_state_t, shared_bytes) == 4 && offsetof(shared_state_t, metadata) == 16);
  static_assert(offsetof(metadata_t, signature) == 0 && offsetof(metadata_t, protocol_version) == 8 && offsetof(metadata_t, metadata_bytes) == 12);
  static_assert(sizeof(metadata_t) == 128);
  static_assert(sizeof(slot_t) == 64);
  static_assert(sizeof(shared_state_t) == 448);
  static_assert(offsetof(slot_t, cursor_plane_flags) == 24);
  static_assert(offsetof(slot_t, ui_parallax_uv) == 28);
  static_assert(offsetof(shared_state_t, consumer_nonce) % 8 == 0);
  static_assert(offsetof(shared_state_t, capability_nonce) == 144);
  static_assert(offsetof(shared_state_t, consumer_capabilities) == 152);
  static_assert(offsetof(shared_state_t, consumer_protocol) == 156);
  static_assert(offsetof(shared_state_t, slots) == 192);
}  // namespace reshade_bridge
