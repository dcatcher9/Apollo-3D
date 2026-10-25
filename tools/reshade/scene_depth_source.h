// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <cstdint>
#include "scene_feedback.h"
#include "depth_encoding.h"

// Provider adapters validate their own ABI/frame/camera evidence before emitting
// this small value snapshot. It contains no vendor structures or pixel owner.
namespace sunshine_scene_depth {
  enum class provider_kind { streamline, ngx };
  // The API that reports a source. Depth sources come from Streamline, NGX
  // and Generic; the frame record's UI rows (game3d_frame_record.h) also
  // name the UI layer and the Present's own color. A later API (FFX, XeSS)
  // reports an existing role; it is not a new role.
  enum class source_api : std::uint8_t { none, streamline, ngx, generic, ui_layer, present };
  // What a depth source is, whichever API reports it: a frame-generation
  // input (one per viewport), scene (upscaler) depth, or the Generic
  // depth-stencil choice. The frame record's rungs D1, D2 and D3.
  enum class source_role : std::uint8_t { fg_depth, scene_depth, generic };
  // One logical depth source, independent of rotating resources.
  struct source_key {
    source_role role{source_role::scene_depth};
    source_api api{source_api::none};
    std::uint32_t viewport{};
    // The source within its role, API and viewport: NGX's feature
    // generation, Generic's resource identity; 0 for Streamline scene depth
    // and FG depth.
    std::uint64_t id{};
  };
  constexpr bool operator==(const source_key &a, const source_key &b) {
    return a.role == b.role && a.api == b.api && a.viewport == b.viewport && a.id == b.id;
  }
  constexpr bool operator!=(const source_key &a, const source_key &b) { return !(a == b); }
  // The API of a frame's provider.
  constexpr source_api reporting_api(provider_kind provider) {
    return provider == provider_kind::ngx ? source_api::ngx : source_api::streamline;
  }
  // A viewport's FG depth. Streamline's FG options scope it; DLSS-G's own
  // NGX evaluation supplies the same source.
  constexpr source_key fg_depth_key(std::uint32_t viewport) {
    return {source_role::fg_depth, source_api::streamline, viewport};
  }
  // frame::source_id, the legacy value that logs and Dump 3D packages carry,
  // unchanged: FG depth is (1 << 63) | viewport, any other key its id. A
  // value names one source only together with its provider and viewport
  // (test_scene_depth_source.cpp inverts it for every key a frame carries).
  constexpr std::uint64_t encode_source_id(const source_key &key) {
    return key.role == source_role::fg_depth ? (std::uint64_t{1} << 63) | key.viewport : key.id;
  }
  enum class resource_kind { raw_depth, display_depth };
  // declared: the caller stated this resource's state for this call.
  // declared_for_use: the caller stated the state at a later use of the
  // resource, not at this copy point (a longer-lived Streamline tag copied
  // inside its tag call): the state observed on the recording proves the copy
  // there, and a declaration that disagrees with it does not refuse it.
  // observed_nonzero: only a state observed on the recording is acceptable (a
  // COMMON only on a depth-stencil texture: capture_state_policy.h
  // common_is_state).
  // sdk_contract: the SDK requires this state at the capture boundary for every
  // input (NGX: NON_PIXEL_SHADER_RESOURCE). A known state observed on the same
  // recording takes precedence; the contract applies only when none was observed.
  // clear_contract: D3D12 requires this state of a depth clear's subresource
  // (DEPTH_WRITE); an observed state must agree, as with a declaration.
  // last_proven: the state the source was last proven to be in by the work
  // executed on the copy's queue (the capture owner's last_proven_state),
  // which stands in for an observation the recording lacks.
  enum class state_proof { unavailable, declared, observed_nonzero, sdk_contract, clear_contract, last_proven, declared_for_use };
  enum class lifetime { unsupported, until_present, until_evaluation, at_call };
  struct extent { std::uint32_t left{}, top{}, width{}, height{}; };
  struct jitter_offset {
    // NVIDIA projection jitter in active render pixels, not allocation/output
    // pixels. This snapshot travels with the captured depth, including FG reuse.
    float x{}, y{};
    std::uint32_t width{}, height{};
    bool supplied{};
  };
  struct projection_basis {
    // Camera clip depth = depth_offset + depth_scale / positive_view_distance.
    // Distances remain provider/game units; no metric-scale claim.
    double depth_offset{}, depth_scale{};
    // Direction may be declared without a projection (for example NGX create
    // flags). Numeric coefficients have meaning only when supplied is true.
    bool supplied{}, reversed{};
    // A decoded provider flag or the shared transport's established clear
    // evidence can supply direction without projection coefficients. Without
    // either, consumers must not assume normal.
    bool direction_supplied{};
    // Optional texture encoding: decoded = raw * scale + bias. Decoded is
    // camera clip depth or positive view distance according to encoding.
    // Keep this separate from the camera coefficients and their comfort gain.
    double raw_scale{1.0}, raw_bias{};
    // Linear inputs carry positive view distance, independently of whether
    // paired camera matrices are available. supplied still means camera data.
    depth_encoding encoding{depth_encoding::device};
  };
  struct resource_description {
    std::uint64_t native{};
    std::uint32_t width{}, height{};
    extent area;
    resource_kind kind{resource_kind::raw_depth};
    // The provider's declared typed DXGI format of the resource (Streamline
    // Resource.nativeFormat), or 0 when it declared none.
    std::uint32_t native_format{};
  };
  struct frame {
    // An adapter observation sequence is not a native presentation number.
    std::uint64_t epoch{}, sequence{}, tick{};
    std::uint32_t viewport{};
    projection_basis projection;
    jitter_offset jitter;
    resource_description resource;
    std::uint32_t native_state{0xffffffffu};
    state_proof proof{state_proof::unavailable};
    lifetime valid_until{lifetime::unsupported};
    provider_kind provider{provider_kind::streamline};
    // The source's key, encoded (encode_source_id). Streamline SR is zero
    // here (its viewport and epoch name it); FG is fg_depth_key(viewport), so
    // disabling FG can release only that source; NGX is its feature generation.
    std::uint64_t source_id{};
    // Middleware identity only; never a native Present number. Explicit frame
    // tags/markers can associate camera data without guessing from cadence.
    std::uint64_t source_frame_generation{}, source_frame_token{}, source_frame_numeric{};
    bool source_frame_has_numeric{}, source_frame_explicit{};
    bool frame_generation_input{};
    // Source-adapter observation generation captured with this frame:
    // Streamline's observation loss counter (NGX leaves it zero). Display
    // reuse never consults it (decide_display, depth_cache_update.h); the UI
    // qualification scope and the UI mask's capture matching compare it, and
    // the provider's status lines report it.
    std::uint64_t observation_revision{};
    sunshine_scene_feedback::sample feedback;
  };
}
