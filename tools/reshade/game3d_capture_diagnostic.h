// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "streamline_depth_capture.h"

#include <nlohmann/json.hpp>
#include <sstream>

namespace sunshine_game3d {
  // A bounded copy of one native record attempt, never a fresh state query.
  // Fields not reached by its stage retain the native diagnostic's defaults;
  // in particular observed_state is evidence only when observed is true.
  inline nlohmann::json capture_diagnostic_json(const sunshine_streamline::depth_capture::record_diagnostic &value) {
    namespace capture = sunshine_streamline::depth_capture;
    const auto hex = [](std::uint64_t identity) {
      std::ostringstream out;
      out << "0x" << std::hex << identity;
      return out.str();
    };
    return {
      {"result", capture::name(value.result)}, {"stage", capture::name(value.stage)}, {"loss", capture::name(value.loss)},
      {"command", hex(value.command)}, {"resource", hex(value.resource)}, {"recording_cookie", value.recording_cookie},
      {"device_identity", value.device_identity}, {"expected_device_identity", value.expected_device_identity},
      {"expected_present_generation", value.expected_generation}, {"current_present_generation", value.current_generation},
      {"native_state", value.native_state}, {"observed_state", value.observed_state}, {"observed", value.observed}, {"blocked", value.blocked},
      {"copy_state", value.copy_state_known ? nlohmann::json(value.copy_state) : nlohmann::json(nullptr)},
      {"copy_state_known", value.copy_state_known}, {"used_observed_state", value.used_observed_state},
      {"recording_closed", value.recording_closed}, {"recording_invalid", value.recording_invalid}, {"render_pass", value.render_pass},
      {"command_type", value.command_type}, {"width", value.width}, {"height", value.height}, {"format", value.format}, {"flags", value.flags},
      {"dimension", value.dimension}, {"mip_levels", value.mip_levels}, {"array_size", value.array_size}, {"samples", value.samples}
    };
  }
}
