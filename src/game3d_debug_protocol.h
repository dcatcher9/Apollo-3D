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
    // Exact RGBA input consumed for UI alpha; optional IDs are in the resource catalog.
    ui_source_color = 33,
    // Optional: output-resolution targets the game cleared to transparent black,
    // copied before a clear. Candidate offscreen UI layers, not verified UI.
    ui_layer_0 = 40,
    ui_layer_1 = 41,
    ui_layer_2 = 42,
    // Optional: the presented colors the renderer retained for the pre-UI
    // change set's pairing (fix 3), one and two Presents before the dumped
    // render, and the offscreen UI layer copy UI detection consumed (t7).
    retained_present_1 = 43,
    retained_present_2 = 44,
    ui_layer_detected = 45,
  };
  inline constexpr unsigned ui_layer_count = 3;
  inline constexpr const char *ui_layer_names[ui_layer_count] {"ui_layer_candidate_0", "ui_layer_candidate_1", "ui_layer_candidate_2"};
  inline constexpr bool ui_layer_artifact(unsigned id) noexcept {
    return id >= static_cast<unsigned>(artifact::ui_layer_0) && id < static_cast<unsigned>(artifact::ui_layer_0) + ui_layer_count;
  }
  inline constexpr const char *ui_layer_name(unsigned id) noexcept {
    return ui_layer_artifact(id) ? ui_layer_names[id - static_cast<unsigned>(artifact::ui_layer_0)] : nullptr;
  }
  // The change-set artifacts (43-45), optional like every ID from 9: a host
  // without these names drops them with an optional_capture_errors entry.
  inline constexpr unsigned change_set_artifact_count = 3;
  inline constexpr const char *change_set_names[change_set_artifact_count] {"retained_present_1", "retained_present_2", "ui_layer_detected"};
  inline constexpr bool change_set_artifact(unsigned id) noexcept {
    return id >= static_cast<unsigned>(artifact::retained_present_1) &&
           id < static_cast<unsigned>(artifact::retained_present_1) + change_set_artifact_count;
  }
  inline constexpr const char *change_set_name(unsigned id) noexcept {
    return change_set_artifact(id) ? change_set_names[id - static_cast<unsigned>(artifact::retained_present_1)] : nullptr;
  }
  // The ID of a change-set artifact name, zero for any other name.
  inline unsigned change_set_artifact_id(const char *name) noexcept {
    if (name)
      for (unsigned i = 0; i < change_set_artifact_count; ++i) {
        const char *a = change_set_names[i], *b = name;
        while (*a && *a == *b) {
          ++a;
          ++b;
        }
        if (!*a && !*b) return static_cast<unsigned>(artifact::retained_present_1) + i;
      }
    return 0;
  }
  static_assert(static_cast<unsigned>(artifact::ui_layer_2) + 1 == static_cast<unsigned>(artifact::retained_present_1) &&
                static_cast<unsigned>(artifact::ui_layer_detected) + 1 ==
                  static_cast<unsigned>(artifact::retained_present_1) + change_set_artifact_count);

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
