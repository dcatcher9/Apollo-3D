// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The API depth provider's status lines (streamline_depth_provider.cpp): the
// values each line prints and the function that renders it. The provider
// gathers the values where it decides and logs the rendered text; the log
// line goldens (test_provider_log_lines.cpp) render fixed values, so a change
// of how the values are gathered keeps every line byte for byte.
#include "depth_addon.h"
#include "streamline_camera_probe.h"
#include "streamline_depth_capture.h"
#include "streamline_native_observer.h"
#include "depth_cache_update.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace sunshine_streamline::provider {
  enum class display_status { not_attempted, ready, reused_depth, texture_creation_failed, view_creation_failed, waiting_capture };
  const char *name(display_status value);
  const char *name(depth_capture::display_action action);
  struct observation_loss_lookup {
    std::uint64_t requested_revision{};
    loss_diagnostics::event value;
    bool found{};
  };
  struct observation_loss_episode {
    std::uint64_t present{}, sampled_tick{};
    observation_loss_lookup sampled;
  };
  // What a Present's depth decision rested on, for the readiness trace.
  struct readiness_evidence {
    sunshine_depth::frame_depth prior;
    depth_capture::capture_diagnostic capture;
    depth_capture::consumer_diagnostic consumer;
    frame_generation_snapshot fg;
    depth_capture::selection_policy selection;
    frame_generation_query_status fg_query{frame_generation_query_status::unavailable};
    display_status display{display_status::not_attempted};
    sunshine_scene_depth::provider_kind prior_provider{sunshine_scene_depth::provider_kind::streamline};
    std::uint64_t present{}, tick{}, prior_capture{};
    bool prior_provider_known{}, prior_output_ready{}, available{}, copied{}, snapshot_ready{}, fg_silent{};
    depth_capture::display_action action{depth_capture::display_action::invalidate};
    const char *source_reason{"not_checked"}, *cache_reason{"not_checked"};
  };
  // Dynamic resolution of one logical source: the last fresh copy's identity,
  // storage and size, and the size changes since the last render size line
  // (diagnostic only; no decision reads it).
  struct render_size_state {
    std::array<std::uint64_t, 4> source{}; // provider, epoch, source_id, viewport
    std::uint32_t width{}, height{}, format{}, view_format{};
    sunshine_scene_depth::extent area;
    std::uint64_t changes{}, held{}, total{}, next_line{};
    std::uint32_t min_width{}, min_height{}, max_width{}, max_height{};
  };

  // "Sunshine depth readiness": a lost or recovered episode of ready depth.
  // The provider's own values beside the evidence: its runtime and swapchain,
  // the trace clock and the episode's first loss, the output's reuse, this
  // Present's capture metadata, the display cache's reason and the video
  // memory suffix.
  struct readiness_line {
    bool ready{};
    const char *why{""};
    std::uint64_t runtime{}, native_swapchain{}, now{}, loss_tick{}, suppressed_episodes{};
    bool output_reused{};
    sunshine_scene_depth::frame current;
    const char *retained_clear{""}, *memory{""};
  };
  void format_readiness(char *text, std::size_t size, const readiness_evidence &e, const readiness_line &line);
  // "Sunshine depth observation evidence": the episode's first-loss evidence.
  void format_observation_evidence(char *text, std::size_t size, bool ready, std::uint64_t runtime,
    std::uint64_t loss_tick, const observation_loss_episode &loss);
  // "Sunshine depth render size": the size changes of one source.
  void format_render_size(char *text, std::size_t size, const render_size_state &value);
  // "Sunshine Streamline frame generation depth scope": a required scope
  // turned silent or live again.
  void format_frame_generation_scope(char *text, std::size_t size, bool silent, std::uint32_t viewport,
    std::uint64_t epoch);
  // "Sunshine API depth": no API source is associated with the queue; with
  // hold, ownership is held in mono for a provider handoff of at most hold_ms.
  void format_unassociated(char *text, std::size_t size, depth_capture::status status,
    const depth_capture::capture_diagnostic &capture, bool hold, std::uint64_t hold_ms);
  // "Sunshine <path> depth" with a selected source: the copied frame.
  struct selected_line {
    const char *path{""};
    depth_capture::status status{depth_capture::status::inactive};
    bool output_ready{}, output_projection_supplied{};
    display_status display{display_status::not_attempted};
    std::uint64_t present{}, command{};
  };
  void format_selected(char *text, std::size_t size, const selected_line &line, const depth_capture::packet &frame,
    const depth_capture::consumer_diagnostic &consumer, const depth_capture::capture_diagnostic &capture);
  // "Sunshine <path> depth" without one: the provider holds the queue.
  void format_held(char *text, std::size_t size, const char *path, depth_capture::status status, bool output_ready,
    display_status display, const depth_capture::capture_diagnostic &capture, const native_observer::counters &observer);
  // Both status lines end with the selection and display decision, then the
  // newest observation loss (unknown without one; age_ms is its age).
  void append_selection(char *text, std::size_t size, const depth_capture::capture_diagnostic &capture,
    const depth_capture::display_decision &update, std::uint64_t prior_depth_age,
    const sunshine_scene_depth::jitter_offset &jitter);
  void append_loss(char *text, std::size_t size, bool known, const loss_diagnostics::event &loss, std::uint64_t age_ms,
    std::uint64_t losses_since);
}
