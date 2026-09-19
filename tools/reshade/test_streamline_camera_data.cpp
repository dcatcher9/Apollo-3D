// SPDX-License-Identifier: GPL-3.0-only
#include "streamline_camera_data.h"
#include "projection_depth_scale.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {
  using namespace sunshine_streamline;
  void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
  }
  void close(double a, double b, double tolerance, const char *message) {
    require(std::abs(a - b) <= tolerance, message);
  }
  matrix4 inverse(const matrix4 &value) {
    // Independent double Gaussian elimination fixture, not the production matrix checks.
    double work[4][8]{};
    for (unsigned row = 0; row != 4; ++row)
      for (unsigned col = 0; col != 4; ++col) {
        work[row][col] = value.m[row][col];
        work[row][4 + col] = row == col ? 1 : 0;
      }
    for (unsigned col = 0; col != 4; ++col) {
      unsigned pivot = col;
      for (unsigned row = col + 1; row != 4; ++row)
        if (std::abs(work[row][col]) > std::abs(work[pivot][col])) pivot = row;
      require(std::abs(work[pivot][col]) > 1e-15, "fixture matrix singular");
      for (unsigned k = 0; k != 8; ++k) std::swap(work[col][k], work[pivot][k]);
      const auto divisor = work[col][col];
      for (double &entry : work[col]) entry /= divisor;
      for (unsigned row = 0; row != 4; ++row) {
        if (row == col) continue;
        const auto multiplier = work[row][col];
        for (unsigned k = 0; k != 8; ++k) work[row][k] -= multiplier * work[col][k];
      }
    }
    matrix4 result;
    for (unsigned row = 0; row != 4; ++row)
      for (unsigned col = 0; col != 4; ++col) result.m[row][col] = static_cast<float>(work[row][4 + col]);
    return result;
  }
  camera_data perspective(bool reverse, bool right_handed, bool infinite = false, float near = .25f, float far = 200) {
    camera_data v;
    v.near_plane = near;
    v.far_plane = infinite ? std::numeric_limits<float>::infinity() : far;
    v.fov = 1.1f;
    v.aspect = 16.0f / 9;
    const float sy = 1 / std::tan(v.fov / 2);
    v.projection.m[0][0] = sy / v.aspect;
    v.projection.m[1][1] = sy;
    float a = infinite ? 1 : far / (far - near);
    float b = infinite ? -near : -near * far / (far - near);
    if (reverse) { a = infinite ? 0 : -near / (far - near); b = -b; }
    v.projection.m[2][2] = right_handed ? -a : a;
    v.projection.m[2][3] = right_handed ? -1 : 1;
    v.projection.m[3][2] = b;
    v.inverse_projection = inverse(v.projection);
    v.depth_inverted = reverse ? 1 : 0;
    v.orthographic = 0;
    v.reset = 0;
    v.not_rendering_game_frames = 0;
    return v;
  }
  void test_supported_projections() {
    for (bool reversed : {false, true})
      for (bool right_handed : {false, true})
        for (bool infinite : {false, true}) {
          auto v = perspective(reversed, right_handed, infinite);
          const auto r = validate(v);
          require(r.valid(), "standard perspective rejected");
          require(r.reversed == reversed && r.right_handed == right_handed && r.infinite_far == infinite,
            "projection convention not recovered");
          close(r.near_from_matrix, .25, 1e-6, "incorrect recovered near");
          if (!infinite) close(r.far_from_matrix, 200, .02, "incorrect recovered far");
          close(r.depth_offset + r.depth_scale / v.near_plane, reversed ? 1 : 0, 1e-6, "near endpoint incorrect");
          const double d1 = r.depth_offset + r.depth_scale / 3;
          const double d2 = r.depth_offset + r.depth_scale / 10;
          require(reversed ? d1 > d2 : d1 < d2, "reconstructed raw depth orientation incorrect");
          close(r.depth_scale / (d1 - r.depth_offset), 3, 1e-10, "inverse depth coefficients fail reconstruction");
          close(r.fov_from_matrix, v.fov, 1e-6, "FOV mismatch");
          close(r.aspect_from_matrix, v.aspect, 1e-6, "aspect mismatch");
        }
    auto off_axis = perspective(true, false);
    off_axis.projection.m[2][0] = .12f;
    off_axis.projection.m[2][1] = -.07f;
    off_axis.inverse_projection = inverse(off_axis.projection);
    require(validate(off_axis).valid(), "off-axis perspective rejected");
    for (auto &row : off_axis.projection.m) for (float &entry : row) entry *= 3.5f;
    off_axis.inverse_projection = inverse(off_axis.projection);
    require(validate(off_axis).valid(), "homogeneously scaled perspective rejected");
    // A large finite far can round to the infinite projection in float32. Compare in clip space.
    require(validate(perspective(false, false, false, .25f, 1e10f)).valid(), "quantized infinite far rejected");
    require(validate(perspective(true, true, false, 25, 100000)).valid(), "game units assumed to be meters");
  }
  void test_invalid_metadata() {
    auto v = perspective(false, false);
    v.inverse_projection.m[0][0] *= 2;
    require(validate(v).status == validation_status::inverse_mismatch, "stale inverse accepted");
    v = perspective(false, false);
    v.depth_inverted = 1;
    require(validate(v).status == validation_status::depth_direction_mismatch, "wrong reversed flag accepted");
    v = perspective(false, false);
    v.depth_inverted = 2;
    require(validate(v).status == validation_status::invalid_flags, "unset reversed flag accepted");
    v = perspective(false, false);
    v.orthographic = 1;
    require(validate(v).status == validation_status::orthographic, "orthographic flag accepted as perspective");
    v = perspective(false, false);
    v.not_rendering_game_frames = 1;
    require(validate(v).status == validation_status::inactive, "menu metadata accepted as gameplay");
    v = perspective(false, false);
    v.projection.m[0][2] = .1f;
    v.inverse_projection = inverse(v.projection);
    require(validate(v).status == validation_status::unsupported_projection, "oblique depth accepted");
    v = perspective(false, false);
    for (unsigned r = 0; r != 4; ++r)
      for (unsigned c = r + 1; c != 4; ++c) std::swap(v.projection.m[r][c], v.projection.m[c][r]);
    v.inverse_projection = inverse(v.projection);
    require(!validate(v).valid(), "transposed matrix accepted");
    v = perspective(false, false);
    v.near_plane *= 10;
    require(validate(v).status == validation_status::valid_matrix_scalar_mismatch &&
        !validate(v).scalar_planes_match, "near metadata disagreement discarded the checked matrix or went unreported");
    v = perspective(false, false);
    v.far_plane = -1;
    require(validate(v).status == validation_status::invalid_planes, "negative far accepted");
    v = perspective(false, false);
    v.far_plane = std::numeric_limits<float>::max();
    require(validate(v).status == validation_status::invalid_planes, "Streamline unset far sentinel accepted");
    v = perspective(false, false);
    v.fov += .2f;
    require(validate(v).status == validation_status::valid_matrix_scalar_mismatch &&
        !validate(v).scalar_fov_aspect_match, "FOV disagreement discarded the checked matrix or went unreported");
    v = perspective(false, false);
    v.aspect *= 2;
    require(validate(v).status == validation_status::valid_matrix_scalar_mismatch &&
        !validate(v).scalar_fov_aspect_match, "aspect disagreement discarded the checked matrix or went unreported");
    v = perspective(false, false);
    v.inverse_projection.m[1][2] = std::numeric_limits<float>::max();
    require(validate(v).status == validation_status::non_finite, "unset matrix sentinel accepted");
  }
  void test_frame_generation_matrix_conventions() {
    // Recorded Expedition 33 FG matrix shape: reversed infinite depth in
    // matrix units, independently reported near=.01/far=75000 and a different
    // FOV scalar. The pair includes off-axis jitter; it does not alter depth.
    auto v = perspective(true, false, true, 1);
    v.projection.m[0][0] = .975587189f;
    v.projection.m[1][1] = 1.73437715f;
    v.projection.m[2][0] = .0002f;
    v.projection.m[2][1] = -.0003f;
    v.inverse_projection = inverse(v.projection);
    v.near_plane = .01f;
    v.far_plane = 75000;
    v.fov = 1.48352969f;
    v.aspect = 1.77813244f;
    const auto checked = validate(v);
    require(checked.valid() && checked.status == validation_status::valid_matrix_scalar_mismatch &&
        !checked.scalar_planes_match && !checked.scalar_fov_aspect_match,
      "FG matrix discarded because of conflicting scalar conventions");
    close(checked.depth_offset, 0, 0, "scalar metadata changed FG offset");
    close(checked.depth_scale, 1, 0, "scalar near plane changed FG depth units");
    close(checked.near_from_matrix, 1, 0, "FG matrix near was replaced by scalar near");
    require(checked.infinite_far, "scalar far truncated infinite FG projection");
    const auto scale = sunshine_projection_depth::make(checked.depth_offset, checked.depth_scale);
    require(scale.valid(), "checked FG matrix not representable by production scale");
    close(scale.near_plane, 1, 0, "FG reconstruction guessed scalar units");
    for (float z : {1.f, 2.f, 17.f, 1000.f}) {
      // Project a real view-space point with the recorded matrix; reconstruct
      // with production coefficients independently of any endpoint metadata.
      const float clip_z = z*v.projection.m[2][2]+v.projection.m[3][2];
      const float clip_w = z*v.projection.m[2][3]+v.projection.m[3][3];
      float q = 0;
      require(sunshine_projection_depth::inverse_distance(scale, clip_z/clip_w, q),
        "FG matrix failed inverse-depth reconstruction");
      close(q, 1.0/z, 1e-8, "FG matrix recovered incorrect inverse distance");
    }

    // Camera reconstruction preserves units; independently normalizing the
    // same physical center makes K*q invariant. Scalar near metadata must not
    // supply an artistic multiplier.
    const auto centimeter = sunshine_projection_depth::make(0, 1);
    const auto meter = sunshine_projection_depth::make(0, .01);
    float center_centimeter = 0, center_meter = 0;
    require(sunshine_projection_depth::inverse_distance(centimeter, .01f, center_centimeter) &&
        sunshine_projection_depth::inverse_distance(meter, .01f, center_meter), "unit center fixture rejected");
    for (float raw : {.001f, .01f, .1f, .5f}) {
      float q_centimeter = 0, q_meter = 0;
      require(sunshine_projection_depth::inverse_distance(centimeter, raw, q_centimeter) &&
          sunshine_projection_depth::inverse_distance(meter, raw, q_meter), "unit fixture rejected");
      close(q_centimeter/center_centimeter, q_meter/center_meter, 1e-5,
        "matrix distance-unit convention changed normalized depth strength");
    }

    // Redundant metadata mismatch cannot waive matrix and orientation checks.
    v.inverse_projection.m[0][0] *= 2;
    require(validate(v).status == validation_status::inverse_mismatch,
      "FG scalar mismatch hid an unrelated inverse matrix");
    v.inverse_projection = inverse(v.projection);
    v.depth_inverted = 0;
    require(validate(v).status == validation_status::depth_direction_mismatch,
      "FG scalar mismatch hid an incompatible depth direction");
    v.depth_inverted = 1;
    v.near_plane = std::numeric_limits<float>::max();
    require(validate(v).status == validation_status::invalid_planes,
      "FG scalar mismatch admitted unset metadata sentinel");
  }
  void fill_common(common_constants &out, const camera_data &v) {
    out.camera_view_to_clip = v.projection;
    out.clip_to_camera_view = v.inverse_projection;
    out.camera_near = v.near_plane;
    out.camera_far = v.far_plane;
    out.camera_fov = v.fov;
    out.camera_aspect = v.aspect;
    // These SDK-layout fields do not participate in depth reconstruction. Their
    // missing/nonfinite values must not veto either version's valid projection.
    out.jitter_offset[0] = std::numeric_limits<float>::quiet_NaN();
    out.jitter_offset[1] = std::numeric_limits<float>::infinity();
    out.motion_vector_scale[0] = std::numeric_limits<float>::infinity();
    out.motion_vector_scale[1] = std::numeric_limits<float>::quiet_NaN();
    for (unsigned axis=0; axis<3; ++axis) {
      out.camera_position[axis] = out.camera_up[axis] = out.camera_right[axis] = out.camera_forward[axis] =
        std::numeric_limits<float>::quiet_NaN();
    }
  }
  void test_bounded_abi() {
    const auto v = perspective(true, true);
    abi_v1::constants old{};
    fill_common(old.common, v);
    old.depth_inverted = 1;
    old.not_rendering_game_frames = 0;
    old.orthographic_projection = 0;
    camera_data decoded;
    require(decode_v1_constants(&old, sizeof(old), decoded) == decode_status::ok && validate(decoded).valid(),
      "v1 valid projection was rejected because of unused SDK metadata");
    old.not_rendering_game_frames = 1;
    require(decode_v1_constants(&old, sizeof(old), decoded) == decode_status::ok && decoded.not_rendering_game_frames == 1 && decoded.orthographic == 0,
      "v1 extra Boolean incorrectly interpreted as orthographic");
    require(decode_v1_constants(&old, sizeof(old) - 1, decoded) == decode_status::short_buffer, "v1 short payload accepted");
    old.ext = &old;
    require(decode_v1_constants(&old, sizeof(old), decoded) == decode_status::extension_present, "v1 extension blindly accepted");
    abi_v2::constants modern{};
    modern.base.type = constants_guid;
    modern.base.version = 2;
    fill_common(modern.common, v);
    modern.depth_inverted = 1;
    require(decode_v2_constants(&modern, sizeof(modern), decoded) == decode_status::ok && validate(decoded).valid(),
      "v2 valid projection was rejected because of unused SDK metadata");
    require(decoded.not_rendering_game_frames == 2, "v2 invented v1 gameplay flag");
    modern.base.version = 1;
    constexpr auto prefix_size = offsetof(abi_v2::constants, motion_vectors_jittered) + 1;
    require(decode_v2_constants(&modern, prefix_size, decoded) == decode_status::ok, "v2 historical V1 prefix requires V2 tail");
    require(decode_v2_constants(&modern, prefix_size - 1, decoded) == decode_status::short_buffer, "truncated v2 flags accepted");
    modern.base.version = 99;
    require(decode_v2_constants(&modern, sizeof(base_structure), decoded) == decode_status::unknown_version, "unknown version interpreted");
    modern.base.version = 2;
    modern.base.type = resource_guid;
    require(decode_v2_constants(&modern, sizeof(modern), decoded) == decode_status::wrong_type, "wrong typed structure accepted");
    modern.base.type = constants_guid;
    modern.base.next = &modern;
    require(decode_v2_constants(&modern, sizeof(modern), decoded) == decode_status::extension_present, "v2 unknown extension accepted");
    require(decode_v2_constants(nullptr, 999, decoded) == decode_status::short_buffer, "null accepted");
    abi_v2::viewport port{{nullptr, viewport_guid, 1}, 123};
    std::uint32_t id = 0;
    require(decode_v2_viewport(&port, sizeof(port), id) == decode_status::ok && id == 123, "viewport layout decode failed");
    port.base.version = 3;
    require(decode_v2_viewport(&port, sizeof(port), id) == decode_status::unknown_version && id == 0xffffffffu, "unknown viewport version accepted");
  }

  void test_tag_matching() {
    camera_key camera{5, 9, 77, true};
    depth_tag depth;
    depth.native_resource = 0x1234;
    depth.viewport = 5;
    depth.observation_epoch = 9;
    depth.frame_key = 77;
    depth.explicit_frame = true;
    depth.lifecycle = 1;
    depth.area = {4, 8, 1280, 720};
    depth.resource_width = 1920;
    depth.resource_height = 1080;
    require(match(camera, depth) == match_status::exact_frame, "explicit frame match rejected");
    depth.explicit_frame = false;
    require(match(camera, depth) == match_status::same_epoch_only, "opaque identity promoted to proven frame match");
    depth.explicit_frame = true;
    depth.frame_key = 78;
    require(match(camera, depth) == match_status::wrong_frame, "mismatched frame accepted");
    depth.frame_key = 77;
    depth.observation_epoch = 8;
    require(match(camera, depth) == match_status::stale_epoch, "stale frame accepted");
    depth.observation_epoch = 9;
    depth.viewport = 6;
    require(match(camera, depth) == match_status::wrong_viewport, "another viewport accepted");
    depth.viewport = 5;
    depth.area.left = 1000;
    require(match(camera, depth) == match_status::invalid_extent, "out of bounds extent accepted");
    depth.area = {0, 0xffffffffu, 1, 1};
    require(match(camera, depth) == match_status::invalid_extent, "integer overflow extent accepted");
    depth.area = {};
    require(match(camera, depth) == match_status::exact_frame, "whole-resource extent rejected");
    depth.lifecycle = 0;
    require(match(camera, depth) == match_status::unsupported_lifecycle, "valid-now resource claimed safe at present");
    depth.lifecycle = 1;
    depth.type = 49;
    require(match(camera, depth) == match_status::not_depth, "linear depth mistaken for projection depth");
    depth.type = 48;
    require(match(camera, depth) == match_status::exact_frame, "high resolution depth rejected");
    depth.native_resource = 0;
    require(match(camera, depth) == match_status::missing_resource, "null native resource accepted");
  }

  void test_optional_jitter() {
    for (const bool modern : {false, true}) {
      abi_v1::constants old{};
      abi_v2::constants current{};
      current.base = {nullptr, constants_guid, 2};
      auto &common = modern ? current.common : old.common;
      fill_common(common, perspective(true, true));
      old.depth_inverted = current.depth_inverted = 1;
      camera_data decoded;
      const auto decode = [&] {
        return modern ? decode_v2_constants(&current, sizeof(current), decoded) :
          decode_v1_constants(&old, sizeof(old), decoded);
      };
      require(decode() == decode_status::ok && validate(decoded).valid() && !decoded.jitter_supplied,
        "Nonfinite jitter rejected usable projection or was marked available");
      common.jitter_offset[0] = .25f; common.jitter_offset[1] = -.375f;
      require(decode() == decode_status::ok && decoded.jitter_supplied &&
          decoded.jitter_offset[0] == .25f && decoded.jitter_offset[1] == -.375f,
        "Signed render-pixel jitter was not decoded exactly");
      common.clip_to_camera_view.m[0][0] *= 2;
      require(decode() == decode_status::ok && !validate(decoded).valid() && decoded.jitter_supplied,
        "Invalid projection erased independent, finite jitter");
      common.jitter_offset[0] = std::numeric_limits<float>::quiet_NaN();
      require(decode() == decode_status::ok && !decoded.jitter_supplied &&
          decoded.jitter_offset[0] == 0 && decoded.jitter_offset[1] == 0,
        "Missing jitter retained a previous frame's offsets");
      common.jitter_offset[0] = common.jitter_offset[1] = 0;
      require(decode() == decode_status::ok && decoded.jitter_supplied,
        "Valid zero jitter was treated as missing metadata");
    }
  }
}
int main() {
  try {
    test_supported_projections();
    test_invalid_metadata();
    test_frame_generation_matrix_conventions();
    test_bounded_abi();
    test_optional_jitter();
    test_tag_matching();
    std::puts("PASS Streamline ABI, projection validation, and conservative tag correlation");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
