/**
 * Pure parsing, geometry, and selection helpers for video-dom-info.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace video_dom_probe {
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
    bool available {};
    bool fully_contained {};
    bool credible_size {};
  };

  enum class selection_reason_e {
    none,
    single,
    largest,
    ambiguous,
  };

  struct selection_t {
    std::optional<std::size_t> index;
    selection_reason_e reason {selection_reason_e::none};
  };

  [[nodiscard]] inline selection_t select_candidate(std::span<const video_candidate_t> candidates) {
    std::vector<std::size_t> eligible;
    eligible.reserve(candidates.size());
    bool has_partial_candidate = false;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
      const auto &candidate = candidates[index];
      if (!candidate.credible_size || !rect_valid(candidate.visible_rect)) {
        continue;
      }
      // A credible clipped video can be the real main player even when Chromium also marks it
      // OFFSCREEN/UNAVAILABLE. It must poison selection before availability filtering; otherwise
      // a smaller fully visible advert could be promoted merely because the main player is clipped.
      if (!candidate.fully_contained) {
        has_partial_candidate = true;
      } else if (candidate.available) {
        eligible.push_back(index);
      }
    }

    if (has_partial_candidate) {
      return {.reason = selection_reason_e::ambiguous};
    }

    if (eligible.empty()) {
      return {};
    }

    if (eligible.size() == 1) {
      return {.index = eligible.front(), .reason = selection_reason_e::single};
    }

    std::size_t largest_index = eligible.front();
    auto largest_area = rect_area(candidates[largest_index].visible_rect);
    bool tied = false;
    for (const auto index : std::span(eligible).subspan(1)) {
      const auto area = rect_area(candidates[index].visible_rect);
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
    return {.index = largest_index, .reason = selection_reason_e::largest};
  }

  [[nodiscard]] inline const char *selection_reason_name(selection_reason_e reason) noexcept {
    switch (reason) {
      case selection_reason_e::none:
        return "none";
      case selection_reason_e::single:
        return "single";
      case selection_reason_e::largest:
        return "largest";
      case selection_reason_e::ambiguous:
        return "ambiguous";
    }
    return "unknown";
  }
}  // namespace video_dom_probe
