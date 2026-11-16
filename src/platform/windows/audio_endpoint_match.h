/**
 * @file src/platform/windows/audio_endpoint_match.h
 * @brief Ranked audio endpoint selection with demand-driven property reads.
 */
#pragma once

#include <cstddef>
#include <optional>
#include <propidl.h>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace platf::audio::detail {
  enum class endpoint_match_field_e {
    device_id,
    device_friendly_name,
    adapter_friendly_name,
    device_description,
  };

  using endpoint_match_t = std::pair<endpoint_match_field_e, std::wstring>;
  using endpoint_match_list_t = std::vector<endpoint_match_t>;

  inline endpoint_match_list_t match_steam_speakers() {
    return {{endpoint_match_field_e::adapter_friendly_name, L"Steam Streaming Speakers"}};
  }

  inline endpoint_match_list_t match_all_fields(const std::wstring &name) {
    return {
      {endpoint_match_field_e::device_id, name},
      {endpoint_match_field_e::device_friendly_name, name},
      {endpoint_match_field_e::device_description, name},
      {endpoint_match_field_e::adapter_friendly_name, name},
    };
  }

  inline std::optional<std::wstring_view> endpoint_property_string(HRESULT status, const PROPVARIANT &value) {
    if (FAILED(status) || value.vt != VT_LPWSTR || !value.pwszVal) {
      return std::nullopt;
    }
    return value.pwszVal;
  }

  // read_device(index) returns an optional reader with id() and matches(field, value).
  // Enumeration order breaks same-field ties; the match-list order outranks it.
  template<class ReadDevice>
  std::optional<std::wstring> find_endpoint_id(
    std::span<const endpoint_match_t> match_list,
    std::size_t count,
    ReadDevice &&read_device
  ) {
    std::optional<std::wstring> matched;
    auto best = match_list.size();
    for (std::size_t index = 0; index < count && best != 0; ++index) {
      auto device = read_device(index);
      if (!device) {
        continue;
      }
      const auto id = device->id();
      if (id.empty()) {
        continue;
      }
      for (std::size_t field = 0; field < best; ++field) {
        const auto &[kind, value] = match_list[field];
        const std::wstring_view expected {value.c_str()};
        const bool matches = kind == endpoint_match_field_e::device_id ?
                               id == expected :
                               device->matches(kind, expected);
        if (matches) {
          best = field;
          matched.emplace(id);
          break;
        }
      }
    }
    return matched;
  }
}  // namespace platf::audio::detail
