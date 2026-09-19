// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <cstdint>
#include "scene_feedback.h"

// Provider adapters validate their own ABI/frame/camera evidence before emitting
// this small value snapshot. It contains no vendor structures or pixel owner.
namespace sunshine_scene_depth {
  // Nomination and completed pixels share one freshness limit. Multi-frame
  // generation can present while two real frames are still in flight.
  inline constexpr std::uint64_t maximum_source_age_ms = 250;
  enum class provider_kind { streamline, ngx };
  enum class resource_kind { raw_depth, display_depth };
  enum class state_proof { unavailable, declared, observed_nonzero };
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
    // Optional texture encoding: camera clip depth = raw * scale + bias.
    // Keep this separate from the camera coefficients and their comfort gain.
    double raw_scale{1.0}, raw_bias{};
  };
  struct resource_description {
    std::uint64_t native{};
    std::uint32_t width{}, height{};
    extent area;
    resource_kind kind{resource_kind::raw_depth};
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
    // Stable logical feature generation, independent of rotating resources.
    // Streamline SR uses viewport/epoch with zero here; FG uses its own
    // viewport-scoped identity so disabling FG can release only that source.
    std::uint64_t source_id{};
    // Middleware identity only; never a native Present number. Explicit frame
    // tags/markers can associate camera data without guessing from cadence.
    std::uint64_t source_frame_generation{}, source_frame_token{}, source_frame_numeric{};
    bool source_frame_has_numeric{}, source_frame_explicit{};
    bool frame_generation_input{};
    // Source-adapter observation generation captured with this frame. Streamline
    // uses its existing loss counter so a known camera reset/drop immediately
    // revokes approximate reuse, even before the next depth tag is submitted.
    std::uint64_t observation_revision{};
    sunshine_scene_feedback::sample feedback;
  };
}
