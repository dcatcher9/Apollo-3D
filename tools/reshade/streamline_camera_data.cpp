// SPDX-License-Identifier: GPL-3.0-only
#include "streamline_camera_data.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace sunshine_streamline {
  namespace {
    bool same_type(const guid &a, const guid &b) { return std::memcmp(&a, &b, sizeof(guid)) == 0; }
    bool ordinary(float v) { return std::isfinite(v) && std::abs(v) < 1e30f; }
    void copy_common(const common_constants &in, camera_data &out) {
      out.projection = in.camera_view_to_clip;
      out.inverse_projection = in.clip_to_camera_view;
      out.near_plane = in.camera_near;
      out.far_plane = in.camera_far;
      out.fov = in.camera_fov;
      out.aspect = in.camera_aspect;
      out.jitter_supplied = ordinary(in.jitter_offset[0]) && ordinary(in.jitter_offset[1]);
      if (out.jitter_supplied) {
        out.jitter_offset[0] = in.jitter_offset[0];
        out.jitter_offset[1] = in.jitter_offset[1];
      }
    }
    decode_status check_base(const base_structure &base, const guid &expected, std::uint64_t max_version) {
      if (!same_type(base.type, expected)) return decode_status::wrong_type;
      if (base.version < 1 || base.version > max_version) return decode_status::unknown_version;
      // Extensions can change resource precision or interpretation. Never chase in-game pointers.
      if (base.next) return decode_status::extension_present;
      return decode_status::ok;
    }
  }

  decode_status decode_v1_constants(const void *bytes, std::size_t size, camera_data &out) {
    out = {};
    if (!bytes || size < sizeof(abi_v1::constants)) return decode_status::short_buffer;
    abi_v1::constants data{};
    std::memcpy(&data, bytes, sizeof(data));
    if (data.ext) return decode_status::extension_present;
    copy_common(data.common, out);
    out.depth_inverted = data.depth_inverted;
    out.orthographic = data.orthographic_projection;
    out.reset = data.reset;
    out.not_rendering_game_frames = data.not_rendering_game_frames;
    return decode_status::ok;
  }

  decode_status decode_v2_constants(const void *bytes, std::size_t size, camera_data &out) {
    out = {};
    if (!bytes || size < sizeof(base_structure)) return decode_status::short_buffer;
    base_structure base{};
    std::memcpy(&base, bytes, sizeof(base));
    auto status = check_base(base, constants_guid, 2);
    if (status != decode_status::ok) return status;
    // V1 and V2 share this prefix. Do not read the optional V2 tail from a V1 allocation.
    constexpr auto prefix_size = offsetof(abi_v2::constants, motion_vectors_jittered) + 1;
    if (size < prefix_size) return decode_status::short_buffer;
    abi_v2::constants data{};
    std::memcpy(static_cast<void *>(&data), bytes, prefix_size);
    copy_common(data.common, out);
    out.depth_inverted = data.depth_inverted;
    out.orthographic = data.orthographic_projection;
    out.reset = data.reset;
    return decode_status::ok;
  }

  decode_status decode_v2_viewport(const void *bytes, std::size_t size, std::uint32_t &out) {
    out = 0xffffffffu;
    constexpr auto prefix_size = offsetof(abi_v2::viewport, value) + sizeof(std::uint32_t);
    if (!bytes || size < prefix_size) return decode_status::short_buffer;
    abi_v2::viewport data{};
    std::memcpy(static_cast<void *>(&data), bytes, prefix_size);
    auto status = check_base(data.base, viewport_guid, 1);
    if (status == decode_status::ok) out = data.value;
    return status;
  }

  camera_validation validate(const camera_data &v) {
    camera_validation result;
    const auto fail = [&](validation_status s) { result.status = s; return result; };
    for (unsigned r = 0; r != 4; ++r)
      for (unsigned c = 0; c != 4; ++c)
        if (!ordinary(v.projection.m[r][c]) || !ordinary(v.inverse_projection.m[r][c]))
          return fail(validation_status::non_finite);
    if (v.depth_inverted > 1 || v.orthographic > 1) return fail(validation_status::invalid_flags);
    if (v.not_rendering_game_frames == 1) return fail(validation_status::inactive);
    if (v.orthographic) return fail(validation_status::orthographic);
    const auto &p = v.projection.m;
    const double h = p[2][3];
    // D3D [0,1] row-vector perspective, including off-axis projections and either handedness.
    // Reject oblique, orthographic, transposed and lens-distorted forms rather than guessing.
    if (std::abs(h) < 1e-12 || std::abs(p[0][0]) < 1e-12 || std::abs(p[1][1]) < 1e-12 || p[3][2] == 0)
      return fail(validation_status::unsupported_projection);
    constexpr unsigned zeros[][2] = {{0,1},{0,2},{0,3},{1,0},{1,2},{1,3},{3,0},{3,1},{3,3}};
    for (const auto &position : zeros)
      if (std::abs(p[position[0]][position[1]] / h) > 1e-5)
        return fail(validation_status::unsupported_projection);
    // Check both orders: a valid-looking depth row must not hide a stale or unrelated inverse.
    for (unsigned order = 0; order != 2; ++order) {
      const auto &a = order ? v.inverse_projection.m : v.projection.m;
      const auto &b = order ? v.projection.m : v.inverse_projection.m;
      for (unsigned row = 0; row != 4; ++row)
        for (unsigned col = 0; col != 4; ++col) {
          double dot = 0;
          for (unsigned k = 0; k != 4; ++k) dot += static_cast<double>(a[row][k]) * b[k][col];
          result.inverse_error = std::max(result.inverse_error, std::abs(dot - (row == col ? 1.0 : 0.0)));
        }
    }
    if (result.inverse_error > 2e-3) return fail(validation_status::inverse_mismatch);
    result.right_handed = h < 0;
    result.depth_offset = p[2][2] / h;
    result.depth_scale = p[3][2] / std::abs(h);
    result.reversed = result.depth_scale > 0;
    if (result.reversed != (v.depth_inverted == 1)) return fail(validation_status::depth_direction_mismatch);
    const double near_depth = result.reversed ? 1.0 : 0.0;
    const double far_depth = 1.0 - near_depth;
    const double near_inverse = (near_depth - result.depth_offset) / result.depth_scale;
    const double far_inverse = (far_depth - result.depth_offset) / result.depth_scale;
    if (!(near_inverse > 0) || far_inverse < 0 || far_inverse >= near_inverse)
      return fail(validation_status::invalid_planes);
    result.near_from_matrix = 1.0 / near_inverse;
    result.infinite_far = far_inverse == 0;
    result.far_from_matrix = result.infinite_far ? std::numeric_limits<double>::infinity() : 1.0 / far_inverse;
    const bool explicit_infinity = std::isinf(v.far_plane) && v.far_plane > 0;
    if (!ordinary(v.near_plane) || !(v.near_plane > 0) ||
        (!explicit_infinity && (!ordinary(v.far_plane) || !(v.far_plane > v.near_plane))))
      return fail(validation_status::invalid_planes);
    // Scalar planes are a redundant cross-check, not coefficients of the
    // tagged depth encoding. Some integrations supply those distances in a
    // different unit convention, or report finite gameplay culling planes
    // alongside an infinite projection. Keep the self-consistent matrix pair
    // authoritative; never replace B with the scalar near plane. Recovering
    // far from float coefficients is also ill-conditioned when far/near is
    // large, so compare metadata endpoints in clip space for diagnostics.
    const double metadata_near = result.depth_offset + result.depth_scale / v.near_plane;
    const double metadata_far = explicit_infinity ? result.depth_offset :
      result.depth_offset + result.depth_scale / v.far_plane;
    result.scalar_planes_match = std::abs(metadata_near - near_depth) <= 2e-4 &&
      std::abs(metadata_far - far_depth) <= 2e-4;
    result.fov_from_matrix = 2 * std::atan(std::abs(h / p[1][1]));
    result.aspect_from_matrix = std::abs(static_cast<double>(p[1][1]) / p[0][0]);
    if (!ordinary(v.fov) || !ordinary(v.aspect) || !(v.fov > 0 && v.fov < 3.141592654f) || !(v.aspect > 0))
      return fail(validation_status::invalid_fov_aspect);
    // cameraFOV does not specify a horizontal/vertical convention in the
    // pinned SL headers. Rendering uses the matrix, so disagreement with this
    // scalar must not discard an otherwise checked depth encoding.
    result.scalar_fov_aspect_match = std::abs(result.fov_from_matrix - v.fov) <= 0.03 &&
      std::abs(result.aspect_from_matrix / v.aspect - 1) <= 0.03;
    return fail(result.scalar_planes_match && result.scalar_fov_aspect_match ?
      validation_status::valid : validation_status::valid_matrix_scalar_mismatch);
  }

  match_status match(const camera_key &camera, const depth_tag &tag) {
    if (!tag.native_resource) return match_status::missing_resource;
    if (tag.type != 0 && tag.type != 48) return match_status::not_depth;
    if (camera.viewport != tag.viewport) return match_status::wrong_viewport;
    if (!camera.observation_epoch || camera.observation_epoch != tag.observation_epoch)
      return match_status::stale_epoch;
    if (camera.explicit_frame && tag.explicit_frame && camera.frame_key != tag.frame_key)
      return match_status::wrong_frame;
    if (tag.lifecycle != 1 && tag.lifecycle != 0xffffffffu) return match_status::unsupported_lifecycle;
    const auto &a = tag.area;
    const bool whole_resource = a.top == 0 && a.left == 0 && a.width == 0 && a.height == 0;
    if (!whole_resource) {
      const auto right = static_cast<std::uint64_t>(a.left) + a.width;
      const auto bottom = static_cast<std::uint64_t>(a.top) + a.height;
      if (!a.width || !a.height || right > 0xffffffffu || bottom > 0xffffffffu ||
          (tag.resource_width && right > tag.resource_width) ||
          (tag.resource_height && bottom > tag.resource_height)) return match_status::invalid_extent;
    }
    return camera.explicit_frame && tag.explicit_frame ? match_status::exact_frame : match_status::same_epoch_only;
  }

  const char *name(decode_status v) {
    switch (v) {
      case decode_status::ok: return "ok";
      case decode_status::short_buffer: return "short-buffer";
      case decode_status::wrong_type: return "wrong-type";
      case decode_status::unknown_version: return "unknown-version";
      case decode_status::extension_present: return "extension-present";
    }
    return "unknown";
  }
  const char *name(validation_status v) {
    switch (v) {
      case validation_status::valid: return "valid";
      case validation_status::valid_matrix_scalar_mismatch: return "valid-matrix-scalar-mismatch";
      case validation_status::non_finite: return "non-finite-or-unset";
      case validation_status::invalid_flags: return "invalid-flags";
      case validation_status::inactive: return "not-rendering-game";
      case validation_status::orthographic: return "orthographic";
      case validation_status::unsupported_projection: return "unsupported-projection";
      case validation_status::inverse_mismatch: return "inverse-mismatch";
      case validation_status::invalid_planes: return "invalid-planes";
      case validation_status::depth_direction_mismatch: return "depth-direction-mismatch";
      case validation_status::invalid_fov_aspect: return "invalid-fov-aspect";
    }
    return "unknown";
  }
  const char *name(match_status v) {
    switch (v) {
      case match_status::exact_frame: return "explicit-frame-match";
      case match_status::same_epoch_only: return "same-observation-epoch-frame-unverified";
      case match_status::missing_resource: return "missing-resource";
      case match_status::not_depth: return "not-projection-depth";
      case match_status::wrong_viewport: return "wrong-viewport";
      case match_status::stale_epoch: return "stale-observation-epoch";
      case match_status::wrong_frame: return "wrong-frame";
      case match_status::invalid_extent: return "invalid-extent";
      case match_status::unsupported_lifecycle: return "not-valid-until-present";
    }
    return "unknown";
  }
}
