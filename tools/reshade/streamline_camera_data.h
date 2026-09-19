// SPDX-License-Identifier: MIT
// Narrow ABI declarations adapted from NVIDIA Streamline, Copyright (c) 2022-2024 NVIDIA CORPORATION.
// Pinned sources: v1.1.1 (5bac43f464f53bc0583bab8df506b788d8d14c3c),
// v2.7.30 (4bcd1fbe8d4f38b50ec86817f1f7cecc55db9779), include/sl_consts.h,
// sl_core_types.h, sl_struct.h, sl.h, sl_core_api.h and sl_pcl.h. See streamline_camera_license.txt.
// Windows x64 observation ABI. Only the documented feature-function lookup is
// called for hook discovery; observers never enable or evaluate SDK features.
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace sunshine_streamline {
  struct matrix4 { float m[4][4]{}; };
  struct extent { std::uint32_t top{}, left{}, width{}, height{}; };
  struct guid { std::uint32_t a; std::uint16_t b, c; std::uint8_t d[8]; };
  struct base_structure { const void *next{}; guid type{}; std::uint64_t version{}; };
  inline constexpr guid constants_guid{0xdcd35ad7, 0x4e4a, 0x4bad, {0xa9,0x0c,0xe0,0xc4,0x9e,0xb2,0x3a,0xfe}};
  inline constexpr guid resource_guid{0x3a9d70cf, 0x2418, 0x4b72, {0x83,0x91,0x13,0xf8,0x72,0x1c,0x72,0x61}};
  inline constexpr guid tag_guid{0x4c6a5aad, 0xb445, 0x496c, {0x87,0xff,0x1a,0xf3,0x84,0x5b,0xe6,0x53}};
  inline constexpr guid viewport_guid{0x171b6435, 0x9b3c, 0x4fc8, {0x99,0x94,0xfb,0xe5,0x25,0x69,0xaa,0xa4}};
  struct common_constants {
    matrix4 camera_view_to_clip, clip_to_camera_view, clip_to_lens_clip, clip_to_prev_clip, prev_clip_to_clip;
    float jitter_offset[2]{}, motion_vector_scale[2]{}, camera_pinhole_offset[2]{};
    float camera_position[3]{}, camera_up[3]{}, camera_right[3]{}, camera_forward[3]{};
    float camera_near{}, camera_far{}, camera_fov{}, camera_aspect{}, invalid_motion_vector{};
  };
  namespace abi_v1 {
    struct constants {
      common_constants common;
      std::uint8_t depth_inverted{}, camera_motion_included{}, motion_vectors_3d{}, reset{};
      std::uint8_t not_rendering_game_frames{}, orthographic_projection{}, motion_vectors_dilated{}, motion_vectors_jittered{};
      const void *ext{};
    };
    struct resource { std::uint8_t type{}; void *native{}, *memory{}, *view{}; std::uint32_t state{}; void *ext{}; };
    using set_constants = bool (*)(const constants &, std::uint32_t frame, std::uint32_t viewport);
    using set_tag = bool (*)(const resource *, std::uint32_t type, std::uint32_t viewport, const extent *);
    using evaluate_feature = bool (*)(void *commands, std::uint32_t feature, std::uint32_t frame, std::uint32_t viewport);
  }
  namespace abi_v2 {
    struct constants {
      base_structure base;
      common_constants common;
      std::uint8_t depth_inverted{}, camera_motion_included{}, motion_vectors_3d{}, reset{};
      std::uint8_t orthographic_projection{}, motion_vectors_dilated{}, motion_vectors_jittered{};
      float min_relative_linear_depth_object_separation{};
    };
    struct resource {
      base_structure base;
      std::uint8_t type{}; void *native{}, *memory{}, *view{};
      std::uint32_t state{}, width{}, height{}, native_format{}, mip_levels{}, array_layers{};
      std::uint64_t gpu_virtual_address{};
      std::uint32_t flags{}, usage{}, reserved{};
    };
    struct resource_tag { base_structure base; const resource *resource_ptr{}; std::uint32_t type{}, lifecycle{}; extent area; };
    struct viewport { base_structure base; std::uint32_t value{}; };
    // FrameToken has a virtual conversion operator. Do not declare or call that operator across
    // compilers: keep its address opaque. A reused token address is not a frame number.
    struct frame_token;
    using set_constants = std::int32_t (*)(const constants &, const frame_token &, const viewport &);
    using set_tag = std::int32_t (*)(const viewport &, const resource_tag *, std::uint32_t count, void *commands);
    using set_tag_for_frame = std::int32_t (*)(const frame_token &, const viewport &, const resource_tag *, std::uint32_t count, void *commands);
    using evaluate_feature = std::int32_t (*)(std::uint32_t feature, const frame_token &, const base_structure **inputs, std::uint32_t count, void *commands);
    // Pinned sl_core_api.h: observe the returned opaque pointer and optional
    // caller-supplied index; never invoke FrameToken's virtual conversion.
    using get_new_frame_token = std::int32_t (*)(frame_token *&, const std::uint32_t *frame_index);
    using get_feature_function = std::int32_t (*)(std::uint32_t feature, const char *name, void *&function);
    using pcl_set_marker = std::int32_t (*)(std::uint32_t marker, const frame_token &frame);
    inline constexpr std::uint32_t feature_pcl = 4;
  }
  static_assert(sizeof(void *) == 8, "Streamline observer supports Windows x64 only");
  static_assert(sizeof(base_structure) == 32 && sizeof(common_constants) == 412);
  static_assert(sizeof(abi_v1::constants) == 432 && offsetof(abi_v1::constants, ext) == 424);
  static_assert(sizeof(abi_v1::resource) == 48 && offsetof(abi_v1::resource, native) == 8);
  static_assert(sizeof(abi_v2::constants) == 456 && offsetof(abi_v2::constants, depth_inverted) == 444);
  static_assert(sizeof(abi_v2::resource) == 112 && offsetof(abi_v2::resource, native) == 40);
  static_assert(sizeof(abi_v2::resource_tag) == 64 && sizeof(abi_v2::viewport) == 40);
  static_assert(std::is_trivially_copyable_v<abi_v1::constants> && std::is_trivially_copyable_v<abi_v2::constants>);
  static_assert(std::is_trivially_copyable_v<abi_v2::viewport>);
  // Verified against independently compiled, unmodified headers at the two pinned revisions.
  static_assert(offsetof(common_constants, clip_to_camera_view) == 64 && offsetof(common_constants, jitter_offset) == 320);
  static_assert(offsetof(common_constants, camera_near) == 392);
  static_assert(offsetof(abi_v1::constants, not_rendering_game_frames) == 416 && offsetof(abi_v1::constants, orthographic_projection) == 417);
  static_assert(offsetof(abi_v2::constants, common) == 32 && offsetof(abi_v2::constants, orthographic_projection) == 448);
  static_assert(offsetof(abi_v2::constants, min_relative_linear_depth_object_separation) == 452);
  static_assert(offsetof(abi_v2::resource_tag, resource_ptr) == 32 && offsetof(abi_v2::resource_tag, area) == 48);

  enum class decode_status { ok, short_buffer, wrong_type, unknown_version, extension_present };
  struct camera_data {
    matrix4 projection, inverse_projection;
    float near_plane{}, far_plane{}, fov{}, aspect{};
    std::uint8_t depth_inverted{2}, orthographic{2}, reset{2}, not_rendering_game_frames{2};
    // Optional render-pixel jitter belongs to this constants frame. Invalid
    // jitter does not invalidate independently usable projection metadata.
    float jitter_offset[2]{};
    bool jitter_supplied{};
  };
  // Input must be a readable snapshot of 'size' bytes, not an unchecked in-game pointer.
  decode_status decode_v1_constants(const void *bytes, std::size_t size, camera_data &out);
  decode_status decode_v2_constants(const void *bytes, std::size_t size, camera_data &out);
  decode_status decode_v2_viewport(const void *bytes, std::size_t size, std::uint32_t &out);
  const char *name(decode_status value);

  enum class validation_status {
    valid, valid_matrix_scalar_mismatch, non_finite, invalid_flags, inactive, orthographic, unsupported_projection,
    inverse_mismatch, invalid_planes, depth_direction_mismatch, invalid_fov_aspect
  };
  struct camera_validation {
    validation_status status{validation_status::non_finite};
    // raw_depth = depth_offset + depth_scale / positive_view_distance. Units remain game units.
    double depth_offset{}, depth_scale{}, near_from_matrix{}, far_from_matrix{};
    double inverse_error{}, fov_from_matrix{}, aspect_from_matrix{};
    bool infinite_far{}, reversed{}, right_handed{};
    // The checked matrix pair owns depth reconstruction. Redundant scalar
    // camera metadata may use another unit/FOV convention; disagreement is
    // diagnostic and never changes the matrix coefficients or invents units.
    bool scalar_planes_match{}, scalar_fov_aspect_match{};
    bool valid() const {
      return status == validation_status::valid || status == validation_status::valid_matrix_scalar_mismatch;
    }
  };
  camera_validation validate(const camera_data &value);
  const char *name(validation_status value);

  struct depth_tag {
    std::uintptr_t native_resource{};
    std::uint32_t viewport{}, type{}, lifecycle{0xffffffffu};
    extent area;
    std::uint32_t resource_width{}, resource_height{};
    std::uint64_t observation_epoch{}, frame_key{};
    bool explicit_frame{}; // v1 numeric frame, or observer-owned numeric frame mapping; never an opaque address.
  };
  struct camera_key { std::uint32_t viewport{}; std::uint64_t observation_epoch{}, frame_key{}; bool explicit_frame{}; };
  enum class match_status { exact_frame, same_epoch_only, missing_resource, not_depth, wrong_viewport, stale_epoch, wrong_frame, invalid_extent, unsupported_lifecycle };
  match_status match(const camera_key &camera, const depth_tag &tag);
  const char *name(match_status value);
}
