#pragma once

#include "../src_assets/windows/assets/shaders/directx/include/host_sbs_subtitle_cohort_contract.h"

#include <array>
#include <cstdint>
#include <type_traits>

namespace host_sbs_subtitle_cohort {
  // Analysis-local native coordinates. Owner is a typed known-birth ID or an UNKNOWN
  // seed tagged with its line context ID; UNKNOWN has no acquisition anchor or age.
  using state_words_t = std::array<std::uint32_t, SBC_STATE_WORDS>;

  // Exact previous unfiltered baseline S, native R32_UINT 0/1, paired with the
  // previous RGB/P/SCT snapshot. Filtered support or filled M is not a substitute.
  constexpr std::uint32_t previous_baseline_support_srv_slot = SBC_PREVIOUS_BASELINE_SUPPORT_SRV_SLOT;

  constexpr bool owner_is_unknown(std::uint32_t owner) {
    return (owner & SBC_OWNER_UNKNOWN_BIT) != 0u;
  }

  constexpr std::uint32_t owner_context(std::uint32_t owner) {
    return owner & SBC_OWNER_ID_MASK;
  }

  struct alignas(16) constants_t {
    std::array<std::uint32_t, 4u> source;       // W,H,crop top,height
    std::array<std::uint32_t, 4u> field;        // W,H,OCR ROI top,bottom
    std::array<std::uint32_t, 4u> content;      // exact half-open field rectangle
    std::array<std::uint32_t, 4u> frame;        // source frame lo/hi,generation lo/hi
    std::array<std::uint32_t, 4u> observation;  // source us lo/hi,domain reset,reserved zero; cut comes from current UIP
    std::array<std::uint32_t, 4u> control;      // enabled,schema,two reserved zeros
  };

  static_assert(sizeof(constants_t) == SBC_CONSTANT_WORDS * sizeof(std::uint32_t));
  static_assert(std::is_standard_layout_v<constants_t>);
  static_assert(SBC_CAPACITY == 8u);
  static_assert(SBC_STATE_WORDS == 288u);

  constexpr std::uint32_t entry_offset(std::uint32_t slot) {
    return SBC_HEADER_WORDS + slot * SBC_ENTRY_WORDS;
  }

  constexpr state_words_t initial_state() {
    state_words_t result {};
    result[SBC_H_TAG] = SBC_TAG;
    result[SBC_H_SCHEMA] = SBC_SCHEMA;
    result[SBC_H_NEXT_ID] = 1u;
    return result;
  }
}
