/**
 * Pure parsing, geometry, selection, and machine-acquisition policy for video-dom-info.
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace video_dom_probe {
  enum class census_checkpoint_e {
    continue_walk,
    service,
    exhausted,
  };

  // A large tree retains traversal progress while the scan thread services liveness and
  // independently retained authority. Neither a checkpoint nor elapsed time completes a census.
  [[nodiscard]] inline constexpr census_checkpoint_e census_checkpoint_policy(
    std::chrono::steady_clock::duration elapsed,
    std::chrono::steady_clock::duration since_service,
    std::size_t visited_nodes,
    std::size_t maximum_nodes,
    std::chrono::steady_clock::duration lifetime,
    std::chrono::steady_clock::duration service_interval
  ) noexcept {
    if (elapsed >= lifetime || visited_nodes >= maximum_nodes) {
      return census_checkpoint_e::exhausted;
    }
    return since_service >= service_interval ? census_checkpoint_e::service :
                                              census_checkpoint_e::continue_walk;
  }

  // Refresh may enter cross-process COM and deliver a foreground transition. Authenticate
  // again after it returns, immediately before publishing any tentative heartbeat record.
  template<typename Refresh, typename Validate, typename Publish>
  [[nodiscard]] bool service_census_checkpoint(Refresh refresh, Validate validate, Publish publish) {
    const auto tentative = refresh();
    if (!validate()) {
      return false;
    }
    publish(tentative);
    return true;
  }

  struct machine_event_generations_t {
    std::uint64_t foreground {};
    std::uint64_t object {};
  };

  struct machine_event_policy_t {
    bool discard_completed_scan {};
    bool invalidate_semantic_state {};
    bool request_follow_up_audit {};
  };

  // Object WinEvents are deliberately advisory: dynamic pages can continuously create,
  // relocate, or destroy unrelated accessible objects while a complete census is running.
  // Preserve an independently validated semantic observation and schedule another audit.
  // A foreground transition changes the authority boundary and must invalidate the result.
  [[nodiscard]] inline machine_event_policy_t machine_event_policy(
    machine_event_generations_t before,
    machine_event_generations_t after
  ) noexcept {
    if (after.foreground != before.foreground) {
      return {
        .discard_completed_scan = true,
        .invalidate_semantic_state = true,
        .request_follow_up_audit = true,
      };
    }

    return {
      .request_follow_up_audit = after.object != before.object,
    };
  }

  enum class cached_selection_phase_e {
    provisional,
    established,
  };

  struct cached_refresh_policy_t {
    bool retain_selection {};
    cached_selection_phase_e next_phase {cached_selection_phase_e::provisional};
    bool publish_ok {};
    bool request_follow_up_audit {};
  };

  // A complete census authorizes only a provisional selection. The following machine tick
  // must authenticate the retained document/video objects again before the selection becomes
  // externally authoritative. Unrelated object WinEvents remain advisory, matching established
  // cache behavior, while a foreground transition is always a hard authority-boundary veto.
  [[nodiscard]] inline cached_refresh_policy_t cached_refresh_policy(
    cached_selection_phase_e current_phase,
    bool exact_refresh_succeeded,
    machine_event_policy_t events
  ) noexcept {
    if (!exact_refresh_succeeded || events.discard_completed_scan ||
        events.invalidate_semantic_state) {
      return {
        .request_follow_up_audit = events.request_follow_up_audit,
      };
    }

    return {
      .retain_selection = true,
      .next_phase = current_phase == cached_selection_phase_e::provisional ?
                      cached_selection_phase_e::established : current_phase,
      .publish_ok = true,
      .request_follow_up_audit = events.request_follow_up_audit,
    };
  }

  enum class uncached_scan_outcome_e {
    warming_up,
    traversal_incomplete,
    accessibility_unavailable,
    other,
  };

  // Retry an interrupted tree walk promptly, but retain the deliberate long backoff after the
  // cold-accessibility limit so a Chromium provider without ExtendedProperties is not hammered.
  [[nodiscard]] inline constexpr std::chrono::milliseconds uncached_scan_retry_delay(
    uncached_scan_outcome_e outcome
  ) noexcept {
    using namespace std::chrono_literals;
    switch (outcome) {
      case uncached_scan_outcome_e::warming_up:
      case uncached_scan_outcome_e::traversal_incomplete:
        return 1s;
      case uncached_scan_outcome_e::accessibility_unavailable:
      case uncached_scan_outcome_e::other:
        return 15s;
    }
    return 15s;
  }

  struct rect_t {
    long left {};
    long top {};
    long right {};
    long bottom {};

    [[nodiscard]] bool operator==(const rect_t &) const = default;
  };

  [[nodiscard]] inline bool rect_valid(const rect_t &rect) noexcept {
    return rect.right > rect.left && rect.bottom > rect.top;
  }

  [[nodiscard]] inline std::uint64_t rect_area(const rect_t &rect) noexcept {
    if (!rect_valid(rect)) {
      return 0;
    }
    const auto width = static_cast<std::uint64_t>(
      static_cast<std::int64_t>(rect.right) - static_cast<std::int64_t>(rect.left)
    );
    const auto height = static_cast<std::uint64_t>(
      static_cast<std::int64_t>(rect.bottom) - static_cast<std::int64_t>(rect.top)
    );
    return width * height;
  }

  [[nodiscard]] inline bool rect_contains(const rect_t &outer, const rect_t &inner) noexcept {
    return rect_valid(outer) && rect_valid(inner) &&
           inner.left >= outer.left && inner.top >= outer.top &&
           inner.right <= outer.right && inner.bottom <= outer.bottom;
  }

  [[nodiscard]] inline bool rect_contains_with_tolerance(
    const rect_t &outer,
    const rect_t &inner,
    long tolerance
  ) noexcept {
    if (!rect_valid(outer) || !rect_valid(inner) || tolerance < 0) {
      return false;
    }
    return static_cast<std::int64_t>(inner.left) >=
             static_cast<std::int64_t>(outer.left) - tolerance &&
           static_cast<std::int64_t>(inner.top) >=
             static_cast<std::int64_t>(outer.top) - tolerance &&
           static_cast<std::int64_t>(inner.right) <=
             static_cast<std::int64_t>(outer.right) + tolerance &&
           static_cast<std::int64_t>(inner.bottom) <=
             static_cast<std::int64_t>(outer.bottom) + tolerance;
  }

  [[nodiscard]] inline std::optional<rect_t> rect_intersection(
    const rect_t &left,
    const rect_t &right
  ) noexcept {
    rect_t intersection {
      std::max(left.left, right.left),
      std::max(left.top, right.top),
      std::min(left.right, right.right),
      std::min(left.bottom, right.bottom),
    };
    if (!rect_valid(intersection)) {
      return std::nullopt;
    }
    return intersection;
  }

  [[nodiscard]] inline std::optional<rect_t> windowed_video_viewport(
    const rect_t &client,
    const rect_t &document,
    const rect_t &video
  ) noexcept {
    if (!rect_valid(client) || !rect_valid(document) || !rect_valid(video)) {
      return std::nullopt;
    }
    const auto in_client = rect_intersection(client, video);
    return in_client ? rect_intersection(*in_client, document) : std::nullopt;
  }

  [[nodiscard]] inline bool credible_video_extent(const rect_t &video, const rect_t &client) noexcept {
    const auto client_area = rect_area(client);
    return rect_valid(video) && client_area > 0u && rect_area(video) >= client_area / 20u;
  }

  [[nodiscard]] inline bool client_fills_monitor(const rect_t &client, const rect_t &monitor) noexcept {
    return rect_valid(client) && rect_valid(monitor) && client == monitor;
  }

  // Chromium occasionally reports a fullscreen IA2 rectangle one physical pixel
  // beyond the Win32 client rectangle. Accept only that bounded rounding error and
  // always return geometry clipped into the trusted outer coordinate space.
  [[nodiscard]] inline std::optional<rect_t> clip_if_within_tolerance(
    const rect_t &outer,
    const rect_t &inner,
    long tolerance = 1
  ) noexcept {
    if (!rect_contains_with_tolerance(outer, inner, tolerance)) {
      return std::nullopt;
    }
    return rect_intersection(outer, inner);
  }

  struct attribute_match_t {
    bool valid {};
    bool has_tag {};
    bool is_video {};
  };

  [[nodiscard]] inline attribute_match_t match_video_tag(std::wstring_view attributes) {
    constexpr std::size_t maximum_attributes_length = 64 * 1024;
    if (attributes.empty() || attributes.size() > maximum_attributes_length || attributes.find(L'\0') != std::wstring_view::npos) {
      return {};
    }

    auto valid_escape = [](wchar_t character) {
      return character == L'\\' || character == L':' || character == L',' ||
             character == L'=' || character == L';';
    };

    auto decode = [&](std::wstring_view encoded, std::wstring &decoded) {
      decoded.clear();
      decoded.reserve(encoded.size());
      bool escaped = false;
      for (const wchar_t character : encoded) {
        if (escaped) {
          if (!valid_escape(character)) {
            return false;
          }
          decoded.push_back(character);
          escaped = false;
        } else if (character == L'\\') {
          escaped = true;
        } else {
          decoded.push_back(character);
        }
      }
      return !escaped;
    };

    auto ascii_equal_folded = [](std::wstring_view left, std::wstring_view right) {
      if (left.size() != right.size()) {
        return false;
      }
      auto fold = [](wchar_t character) {
        if (character >= L'A' && character <= L'Z') {
          return static_cast<wchar_t>(character - L'A' + L'a');
        }
        return character;
      };
      for (std::size_t index = 0; index < left.size(); ++index) {
        if (fold(left[index]) != fold(right[index])) {
          return false;
        }
      }
      return true;
    };

    bool saw_video_tag = false;
    bool saw_any_tag = false;
    std::size_t token_start = 0;
    while (token_start < attributes.size()) {
      std::size_t token_end = token_start;
      bool escaped = false;
      for (; token_end < attributes.size(); ++token_end) {
        const wchar_t character = attributes[token_end];
        if (escaped) {
          escaped = false;
        } else if (character == L'\\') {
          if (token_end + 1 >= attributes.size() || !valid_escape(attributes[token_end + 1])) {
            return {};
          }
          escaped = true;
        } else if (character == L';') {
          break;
        }
      }
      if (escaped) {
        return {};
      }

      const auto token = attributes.substr(token_start, token_end - token_start);
      if (!token.empty()) {
        std::size_t separator = std::wstring_view::npos;
        escaped = false;
        for (std::size_t index = 0; index < token.size(); ++index) {
          const wchar_t character = token[index];
          if (escaped) {
            escaped = false;
          } else if (character == L'\\') {
            if (index + 1 >= token.size() || !valid_escape(token[index + 1])) {
              return {};
            }
            escaped = true;
          } else if (character == L':') {
            separator = index;
            break;
          }
        }
        if (escaped || separator == std::wstring_view::npos || separator == 0) {
          return {};
        }

        std::wstring key;
        std::wstring value;
        if (!decode(token.substr(0, separator), key) || !decode(token.substr(separator + 1), value)) {
          return {};
        }
        if (ascii_equal_folded(key, L"tag")) {
          if (saw_any_tag) {
            return {};
          }
          saw_any_tag = true;
          saw_video_tag = ascii_equal_folded(value, L"video");
        }
      }

      if (token_end == attributes.size()) {
        break;
      }
      token_start = token_end + 1;
    }

    return {.valid = true, .has_tag = saw_any_tag, .is_video = saw_video_tag};
  }

  struct video_candidate_t {
    long unique_id {};
    rect_t element_rect {};
    rect_t visible_rect {};
    // Windowed availability requires current video/document states and a visible
    // intersection. Containment remains diagnostic only. Fullscreen keeps its
    // separate availability/coverage policy, independent of document clipping.
    bool available {};
    bool fully_contained {};
    bool credible_size {};
    bool fullscreen_available {};
  };

  [[nodiscard]] inline constexpr bool fullscreen_semantic_available(
    bool video_state_available,
    bool has_owning_document,
    bool document_state_available
  ) noexcept {
    return video_state_available && has_owning_document && document_state_available;
  }

  enum class selection_reason_e {
    none,
    single,
    largest,
    fullscreen,
    ambiguous,
  };

  struct selection_t {
    std::optional<std::size_t> index;
    selection_reason_e reason {selection_reason_e::none};
  };

  [[nodiscard]] inline bool selection_can_stage_provisional(
    const selection_t &selection
  ) noexcept {
    return selection.index.has_value() &&
           (selection.reason == selection_reason_e::single ||
            selection.reason == selection_reason_e::largest ||
            selection.reason == selection_reason_e::fullscreen);
  }

  struct complete_census_policy_t {
    bool stage_selection {};
    bool revoke_cached_selection {};
    cached_selection_phase_e phase {cached_selection_phase_e::provisional};
    bool publish_ok {};
  };

  // A matching established cache was independently refreshed immediately before the census, so
  // replacing its retained handles does not reduce its authority. Every new unique selection is
  // provisional and must stay unpublished until cached_refresh_policy() runs on the next tick.
  [[nodiscard]] inline complete_census_policy_t complete_census_policy(
    const selection_t &selection,
    bool matches_established_selection
  ) noexcept {
    if (!selection_can_stage_provisional(selection)) {
      return {.revoke_cached_selection = true};
    }
    if (matches_established_selection) {
      return {
        .stage_selection = true,
        .phase = cached_selection_phase_e::established,
        .publish_ok = true,
      };
    }
    return {
      .stage_selection = true,
      .phase = cached_selection_phase_e::provisional,
      .publish_ok = false,
    };
  }

  [[nodiscard]] inline selection_t select_candidate(
    std::span<const video_candidate_t> candidates,
    const rect_t &client_rect
  ) {
    std::vector<std::size_t> eligible;
    eligible.reserve(candidates.size());
    for (std::size_t index = 0; index < candidates.size(); ++index) {
      const auto &candidate = candidates[index];
      if (!candidate.available || !candidate.credible_size || !rect_valid(candidate.element_rect) || !rect_valid(candidate.visible_rect)) {
        continue;
      }
      eligible.push_back(index);
    }

    if (eligible.empty()) {
      return {};
    }

    std::size_t largest_index = eligible.front();
    auto largest_area = rect_area(candidates[largest_index].element_rect);
    bool tied = false;
    for (const auto index : std::span(eligible).subspan(1)) {
      const auto area = rect_area(candidates[index].element_rect);
      if (area > largest_area) {
        largest_index = index;
        largest_area = area;
        tied = false;
      } else if (area == largest_area) {
        tied = true;
      }
    }
    if (tied) {
      return {.reason = selection_reason_e::ambiguous};
    }
    // Rank the complete semantic element before testing its visible viewport. Cropping the
    // main video must not promote a smaller advert, even when only a tiny sliver remains.
    const auto &largest = candidates[largest_index];
    if (!credible_video_extent(largest.visible_rect, client_rect) || !rect_contains(client_rect, largest.visible_rect) || !rect_contains(largest.element_rect, largest.visible_rect)) {
      return {};
    }
    return {.index = largest_index, .reason = eligible.size() == 1u ? selection_reason_e::single : selection_reason_e::largest};
  }

  // Prefer the independently verified visible viewport for an ordinary browser client. Only a
  // currently verified full-monitor client preserves relaxed fullscreen priority, so harmless
  // element overscan/document clipping cannot tighten its established refresh requirements.
  // The relaxed fallback can still accept full-cover clones or clipped document geometry. Pick
  // a stable IA2 identity there so traversal-order churn does not change the retained object.
  [[nodiscard]] inline selection_t select_authority_candidate(
    std::span<const video_candidate_t> candidates,
    const rect_t &client_rect,
    long coordinate_tolerance = 1,
    bool verified_full_monitor_client = false
  ) {
    const auto windowed = select_candidate(candidates, client_rect);
    if (windowed.index && !verified_full_monitor_client) {
      return windowed;
    }
    std::optional<std::size_t> fullscreen_index;
    if (rect_valid(client_rect) && coordinate_tolerance >= 0) {
      for (std::size_t index = 0; index < candidates.size(); ++index) {
        const auto &candidate = candidates[index];
        if (!candidate.fullscreen_available || !rect_contains_with_tolerance(candidate.element_rect, client_rect, coordinate_tolerance)) {
          continue;
        }
        if (!fullscreen_index || candidate.unique_id < candidates[*fullscreen_index].unique_id) {
          fullscreen_index = index;
        }
      }
    }
    if (fullscreen_index) {
      return {
        .index = *fullscreen_index,
        .reason = selection_reason_e::fullscreen,
      };
    }
    return windowed;
  }

  [[nodiscard]] inline std::optional<rect_t> selection_authority_rect(
    std::span<const video_candidate_t> candidates,
    const selection_t &selection,
    const rect_t &client_rect,
    long coordinate_tolerance = 1
  ) noexcept {
    if (!selection.index || *selection.index >= candidates.size()) {
      return std::nullopt;
    }
    const auto &candidate = candidates[*selection.index];
    if (selection.reason == selection_reason_e::fullscreen) {
      if (!candidate.fullscreen_available ||
          !rect_contains_with_tolerance(
            candidate.element_rect,
            client_rect,
            coordinate_tolerance
          )) {
        return std::nullopt;
      }
      return client_rect;
    }
    if (!candidate.available || !credible_video_extent(candidate.visible_rect, client_rect) || !rect_contains(client_rect, candidate.visible_rect) || !rect_contains(candidate.element_rect, candidate.visible_rect)) {
      return std::nullopt;
    }
    return candidate.visible_rect;
  }

  // The retained document/video identities are authenticated by the caller. An unrelated
  // document layout change does not revoke a video whose exact geometry and visible authority
  // remain unchanged. Recompute the exact visible intersection of the CURRENT document/client.
  [[nodiscard]] inline bool windowed_cache_refresh_matches(
    const rect_t &cached_element_rect,
    const rect_t &cached_authority_rect,
    const rect_t &current_client_rect,
    const rect_t &current_document_rect,
    const rect_t &current_video_rect
  ) noexcept {
    if (current_video_rect != cached_element_rect) {
      return false;
    }
    const auto visible = windowed_video_viewport(
      current_client_rect,
      current_document_rect,
      current_video_rect
    );
    return visible && credible_video_extent(*visible, current_client_rect) &&
           *visible == cached_authority_rect;
  }

  // Fullscreen cache refresh deliberately has no document-rectangle input. The authority is the
  // unchanged foreground client plus the retained semantic video identity; Chromium may clip or
  // relocate its document rectangle and may change harmless video overscan between full censuses.
  [[nodiscard]] inline bool fullscreen_cache_refresh_matches(
    long expected_video_unique_id,
    long observed_video_unique_id,
    const rect_t &cached_authority_rect,
    const rect_t &current_client_rect,
    const rect_t &current_video_rect,
    long coordinate_tolerance = 1
  ) noexcept {
    return expected_video_unique_id != 0 &&
           observed_video_unique_id == expected_video_unique_id &&
           cached_authority_rect == current_client_rect &&
           rect_contains_with_tolerance(
             current_video_rect,
             current_client_rect,
             coordinate_tolerance
           );
  }

  [[nodiscard]] inline const char *selection_reason_name(selection_reason_e reason) noexcept {
    switch (reason) {
      case selection_reason_e::none:
        return "none";
      case selection_reason_e::single:
        return "single";
      case selection_reason_e::largest:
        return "largest";
      case selection_reason_e::fullscreen:
        return "fullscreen";
      case selection_reason_e::ambiguous:
        return "ambiguous";
    }
    return "unknown";
  }
}  // namespace video_dom_probe
