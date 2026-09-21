// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "depth_addon.h"
#include "depth_cache_update.h"

namespace sunshine_streamline::provider {
  struct display_cache_context {
    std::uint32_t color_width{}, color_height{};
    bool immediate_commands{};
    std::uint64_t queue{}, texture{}, view{}, now{};
  };

  // One successful private display copy. Source admission belongs to capture;
  // this owner only checks whether its presentation storage can still be used.
  // It owns no game resource or capture lease and performs no runtime/GPU calls.
  class display_depth_cache {
  public:
    const sunshine_depth::frame_depth &depth() const { return depth_; }
    bool matches_color(std::uint32_t width, std::uint32_t height) const {
      return width == color_width_ && height == color_height_;
    }

    depth_capture::retained_depth reference() const {
      if (!depth_.ready) return {};
      depth_capture::retained_depth out;
      out.metadata = depth_.provided;
      out.capture_id = capture_id_;
      out.real_present = depth_.frame_index;
      out.width = depth_.width;
      out.height = depth_.height;
      out.format = format_;
      out.area = {depth_.x, depth_.y, depth_.active_width, depth_.active_height};
      return out;
    }

    // Only the caller's successful fresh copy can arm the cache. In particular,
    // feeding a held output back here cannot refresh its age or NGX present limit.
    void commit(const sunshine_depth::frame_depth &value, std::uint64_t capture_id,
        std::uint32_t format, std::uint32_t color_width, std::uint32_t color_height) {
      if (!value.ready || value.reused_depth || !capture_id) {
        invalidate("invalid_fresh_copy");
        return;
      }
      depth_ = value;
      capture_id_ = capture_id;
      format_ = format;
      color_width_ = color_width;
      color_height_ = color_height;
      reason_ = "none";
    }

    // The reason must remain valid until the next commit/invalidation; production
    // callers use static diagnostic strings. Invalidated pixels cannot revive.
    void invalidate(const char *reason) {
      depth_ = {};
      capture_id_ = 0;
      format_ = color_width_ = color_height_ = 0;
      reason_ = reason ? reason : "invalidated";
    }
    const char *reason() const { return reason_; }

    bool reuse(const depth_capture::display_decision &decision, const display_cache_context &context,
        std::uint64_t present, sunshine_depth::frame_depth &out) {
      out = {};
      if (decision.action != depth_capture::display_action::hold) {
        invalidate(decision.reason);
        return false;
      }
      if (!depth_.ready) return false;
      if (!matches_color(context.color_width, context.color_height))
        return reject("color_shape_changed");
      if (!context.immediate_commands) return reject("consumer_command_changed");
      if (!context.queue || context.queue != depth_.command_queue)
        return reject("consumer_queue_changed");
      if (!context.texture || !context.view || context.texture != depth_.resource.handle ||
          context.view != depth_.shader_resource.handle)
        return reject("display_storage_changed");
      if (!depth_.provided.tick || context.now < depth_.provided.tick ||
          context.now - depth_.provided.tick >= sunshine_scene_depth::maximum_source_age_ms)
        return reject("retained_depth_expired");
      out = depth_;
      out.frame_index = present;
      out.reused_depth = true;
      return true;
    }

  private:
    bool reject(const char *reason) { invalidate(reason); return false; }
    sunshine_depth::frame_depth depth_;
    std::uint64_t capture_id_{};
    std::uint32_t format_{}, color_width_{}, color_height_{};
    const char *reason_{"no_retained_depth"};
  };
}
