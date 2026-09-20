// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "camera_scene_policy.h"
#include "depth_moments.h"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

// Immutable submission/readback transport. This establishes metadata association,
// NOT complete native mutation coverage, final-color/depth registration, GPU
// ordering, projection correctness, or permission to drive stereo geometry.
namespace sunshine_camera_binding {
  namespace scene = sunshine_camera_scene;
  inline constexpr std::uint64_t maximum_age_ms = scene::target_expiry_ms;

  struct rectangle {
    std::uint32_t x{}, y{}, width{}, height{};
    bool operator==(const rectangle &r) const noexcept {
      return x == r.x && y == r.y && width == r.width && height == r.height;
    }
  };
  struct resource_key {
    std::uint64_t native{}, lifetime{};
    bool operator==(const resource_key &r) const noexcept {
      return native == r.native && lifetime == r.lifetime;
    }
  };
  struct scope_key {
    std::uint64_t runtime{}, runtime_epoch{}, device{}, device_epoch{};
    bool operator==(const scope_key &r) const noexcept {
      return runtime == r.runtime && runtime_epoch == r.runtime_epoch &&
        device == r.device && device_epoch == r.device_epoch;
    }
  };
  struct selection_key {
    scope_key scope;
    scene::source_key original;
    resource_key sampled; // Actual resource passed to the sampler (depth backup).
    std::uint32_t sample_format{}, sample_width{}, sample_height{};
    rectangle crop; // Exact requested crop, already normalized to the source.
    std::uint64_t layout_epoch{};
    bool operator==(const selection_key &r) const noexcept {
      return scope == r.scope && original == r.original && sampled == r.sampled &&
        sample_format == r.sample_format && sample_width == r.sample_width && sample_height == r.sample_height &&
        crop == r.crop && layout_epoch == r.layout_epoch;
    }
  };
  struct depth_copy_key {
    // Opaque captured provenance; zero means unknown. Association never promotes
    // these fields into proof. Later copies of the same backup must not rewrite
    // this submission's key, camera or frame metadata.
    std::uint64_t copy_id{}, ledger_epoch{}, recording_command{}, recording_generation{};
    std::uint64_t source_content_version{}, backup_registration{};
  };
  enum class observation_level { unavailable, source, command, content };
  enum class observation_binding_status { unavailable, captured, selection_mismatch, copy_mismatch, rejected_camera, rejected_frame };
  inline const char *name(observation_level value) noexcept {
    switch (value) {
      case observation_level::source: return "source";
      case observation_level::command: return "command";
      case observation_level::content: return "content-coverage-incomplete";
      default: return "unavailable";
    }
  }
  inline const char *name(observation_binding_status value) noexcept {
    switch (value) {
      case observation_binding_status::captured: return "captured";
      case observation_binding_status::selection_mismatch: return "selection-mismatch";
      case observation_binding_status::copy_mismatch: return "copy-mismatch";
      case observation_binding_status::rejected_camera: return "rejected-camera";
      case observation_binding_status::rejected_frame: return "rejected-frame";
      default: return "unavailable";
    }
  }
  struct camera_observation {
    // Opaque Streamline evidence/frame discriminants, retained as values without
    // importing native hooks into this transport. Their meanings come from the
    // probe enums. A source match is explicitly weaker than a content match.
    std::uint32_t evidence_status{}, frame_kind{}, viewport{}, tag_type{};
    std::uint64_t epoch{}, sequence{}, camera_sequence{}, tick{}, camera_tick{}, loss_revision{};
    std::uint64_t frame_generation{}, frame_numeric{};
    std::uintptr_t frame_token{};
    bool frame_has_numeric{};
    observation_level level{observation_level::unavailable};
    observation_binding_status binding{observation_binding_status::unavailable};
  };
  struct submission {
    scene::registration camera;
    selection_key selection;
    depth_copy_key copy;
    camera_observation observation;
    std::uint64_t capture_frame{}, capture_ms{};
    bool projection_associated{}; // Caller observed this camera/source pairing.
    // Freeze the requested point layout with the crop. Defaults preserve the
    // historical physical-camera fixture; live requests use tile_grid(crop).
    std::uint32_t grid_width{scene::grid_width}, grid_height{scene::grid_height};
  };
  struct readback {
    // Independent sampler request identity, never its legacy source-ID token.
    std::uint64_t capture_id{};
    // Scope and lifetime are immutable owner observations. The sampler obtains
    // native source, format, size and actual crop from its submitted resource.
    // Echoing owner metadata is not independent GPU/frame/content proof.
    scope_key scope;
    resource_key sampled;
    std::uint32_t source_format{}, source_width{}, source_height{};
    rectangle crop;
    std::uint32_t width{scene::grid_width}, height{scene::grid_height};
    const float *values{}; // Borrowed ONLY during complete(); never retained.
    std::size_t value_count{};
    bool valid{};
  };
  struct point_sample {
    std::uint64_t id{}, capture_ms{};
    scene::registration metadata;
    scene::frame_key readback_frame;
    scene::source_key readback_source;
    std::uint32_t width{}, height{};
    std::size_t count{};
    std::array<float, sunshine_depth_statistics::maximum_tiles> raw{};
  };
  struct associated_sample {
    submission captured;
    // Transport is independent of the old fixed-grid numerical policy. These
    // points remain exact; no truncation, resampling or fake padding is admitted.
    point_sample sample;
    bool association_available{}, projection_associated{};
  };
  enum class status {
    submitted, associated, busy, no_pending, wrong_capture_id, invalid_submission,
    expired, clock_went_backwards, scope_mismatch, source_mismatch, format_mismatch,
    layout_mismatch, invalid_gpu_result, selection_changed, unchanged, token_unavailable
  };
  inline const char *name(status value) noexcept {
    switch (value) {
#define SUNSHINE_BINDING_STATUS(x) case status::x: return #x
      SUNSHINE_BINDING_STATUS(submitted); SUNSHINE_BINDING_STATUS(associated);
      SUNSHINE_BINDING_STATUS(busy); SUNSHINE_BINDING_STATUS(no_pending);
      SUNSHINE_BINDING_STATUS(wrong_capture_id); SUNSHINE_BINDING_STATUS(invalid_submission);
      SUNSHINE_BINDING_STATUS(expired); SUNSHINE_BINDING_STATUS(clock_went_backwards);
      SUNSHINE_BINDING_STATUS(scope_mismatch); SUNSHINE_BINDING_STATUS(source_mismatch);
      SUNSHINE_BINDING_STATUS(format_mismatch); SUNSHINE_BINDING_STATUS(layout_mismatch);
      SUNSHINE_BINDING_STATUS(invalid_gpu_result); SUNSHINE_BINDING_STATUS(selection_changed);
      SUNSHINE_BINDING_STATUS(unchanged); SUNSHINE_BINDING_STATUS(token_unavailable);
#undef SUNSHINE_BINDING_STATUS
    }
    return "unknown";
  }

  namespace detail {
    // Shared across mailbox instances/TUs in this add-on. Do not reset this when
    // a runtime/mailbox is recreated, or late old completions could reuse an ID.
    inline std::atomic<std::uint64_t> next_capture_id{1};
    inline std::uint64_t new_capture_id() noexcept {
      auto candidate = next_capture_id.load(std::memory_order_relaxed);
      // Bounded even under unexpected concurrent callers; the owner may retry a
      // refused submission later. UINT64_MAX permanently exhausts, never wraps.
      for (unsigned attempt = 0; attempt != 8; ++attempt) {
        if (candidate == std::numeric_limits<std::uint64_t>::max()) return 0;
        if (next_capture_id.compare_exchange_strong(candidate, candidate + 1, std::memory_order_relaxed))
          return candidate;
      }
      return 0;
    }
    inline bool valid_rectangle(const rectangle &r, std::uint32_t width, std::uint32_t height) noexcept {
      return width && height && r.width && r.height && r.x < width && r.y < height &&
        r.width <= width - r.x && r.height <= height - r.y;
    }
    inline bool valid_selection(const selection_key &v) noexcept {
      const auto &s = v.original;
      return v.scope.runtime && v.scope.runtime_epoch && v.scope.device && v.scope.device_epoch &&
        s.native && s.lifetime && v.sampled.native && v.sampled.lifetime && v.sample_format &&
        valid_rectangle({s.left, s.top, s.extent_width, s.extent_height}, s.width, s.height) &&
        valid_rectangle(v.crop, v.sample_width, v.sample_height);
    }
  }

  // One pending request, matching the current global-busy GPU sampler. The owner
  // serializes methods with its existing state lock; only ID allocation is
  // atomic. No method allocates or calls GPU/OS/probe APIs.
  class mailbox {
  public:
    mailbox() = default;
    mailbox(const mailbox &) = delete;
    mailbox &operator=(const mailbox &) = delete;
    status begin(const submission &value, std::uint64_t now_ms, std::uint64_t &capture_id) noexcept {
      capture_id = 0;
      if (pending_) return status::busy;
      if (!detail::valid_selection(value.selection) || !value.grid_width || !value.grid_height ||
          value.grid_width > value.selection.crop.width || value.grid_height > value.selection.crop.height ||
          std::uint64_t(value.grid_width) * value.grid_height > sunshine_depth_statistics::maximum_tiles ||
          value.capture_ms > now_ms ||
          (value.projection_associated && (!value.camera.unit_epoch || !(value.camera.source == value.selection.original))))
        return status::invalid_submission;
      if (now_ms - value.capture_ms >= maximum_age_ms) return status::expired;
      const auto id = detail::new_capture_id();
      if (!id) return status::token_unavailable;
      captured_ = value;
      // This mailbox currently carries a weak observation, never a complete
      // registration proof, including when an upstream caller passes true.
      captured_.camera.proof_admitted = false;
      capture_id_ = capture_id = id;
      submit_ms_ = now_ms;
      pending_ = true;
      return status::submitted;
    }

    status complete(const readback &value, std::uint64_t now_ms, associated_sample &out) noexcept {
      out = {};
      if (!pending_) return status::no_pending;
      // An old canceled request may complete while a new one is pending. It
      // cannot consume or poison the newer binding.
      if (!value.capture_id || value.capture_id != capture_id_) return status::wrong_capture_id;
      pending_ = false; // Every matching completion is terminal, even failure.
      if (now_ms < submit_ms_) return status::clock_went_backwards;
      if (now_ms - captured_.capture_ms >= maximum_age_ms) return status::expired;
      const auto &expected = captured_.selection;
      if (!(value.scope == expected.scope)) return status::scope_mismatch;
      if (!(value.sampled == expected.sampled)) return status::source_mismatch;
      if (value.source_format != expected.sample_format) return status::format_mismatch;
      if (value.source_width != expected.sample_width || value.source_height != expected.sample_height ||
          !(value.crop == expected.crop) || value.width != captured_.grid_width || value.height != captured_.grid_height)
        return status::layout_mismatch;
      if (!value.valid || !value.values || value.value_count != std::size_t(captured_.grid_width) * captured_.grid_height)
        return status::invalid_gpu_result;
      out.captured = captured_;
      out.sample.id = capture_id_;
      out.sample.capture_ms = captured_.capture_ms;
      out.sample.metadata = captured_.camera;
      out.sample.readback_frame = captured_.camera.frame;
      out.sample.readback_source = captured_.camera.source;
      out.sample.width = value.width;
      out.sample.height = value.height;
      out.sample.count = value.value_count;
      for (std::size_t i = 0; i != value.value_count; ++i) out.sample.raw[i] = value.values[i];
      out.sample.metadata.proof_admitted = false;
      out.association_available = true;
      out.projection_associated = captured_.projection_associated;
      return status::associated;
    }

    // Call when selection/runtime/device identity or layout changes. Copy IDs
    // and latest projection/frame intentionally are NOT part of this check:
    // completing a correctly submitted older frame must return its old metadata.
    status invalidate_if_changed(const selection_key &current) noexcept {
      if (!pending_) return status::no_pending;
      if (current == captured_.selection) return status::unchanged;
      cancel();
      return status::selection_changed;
    }
    void cancel() noexcept {
      pending_ = false;
      capture_id_ = 0;
      captured_ = {};
      submit_ms_ = 0;
    }
    bool busy() const noexcept { return pending_; }

  private:
    submission captured_;
    std::uint64_t capture_id_{}, submit_ms_{};
    bool pending_{};
  };
}
