/** Session provenance stamped immediately before native mouse/keyboard injection. */
#pragma once

#include "src/platform/common.h"

#include <Windows.h>

namespace platf::detail {
  inline void apply_input_tag(INPUT &input) noexcept {
    const auto tag = static_cast<ULONG_PTR>(input_tag_scope_t::current());
    if (input.type == INPUT_MOUSE) {
      input.mi.dwExtraInfo = tag;
    } else if (input.type == INPUT_KEYBOARD) {
      input.ki.dwExtraInfo = tag;
    }
  }
}  // namespace platf::detail
