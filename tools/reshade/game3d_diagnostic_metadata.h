// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "streamline_camera_data.h"
#include "../../src/game3d_debug_ui_resources.h"

#include <array>
#include <cstdint>
#include <string>
#include <Windows.h>

namespace sunshine_game3d {
  // Diagnostics never select a source, retain a game texture or wait for GPU work.
  // Arming starts a fresh bounded observation window; disarming preserves it for
  // serialization. The JSON contains observations, not a color/depth pairing claim.
  void arm_diagnostic_metadata(bool enabled);
  std::uint64_t diagnostic_metadata_generation() noexcept;
  // True only after this armed window observes an SL frame call or NGX named
  // parameters. Cached modules/options and stale completion callbacks do not
  // qualify; callers may use a bounded deadline when no provider is available.
  bool has_diagnostic_frame_observation() noexcept;
  std::string diagnostic_metadata_json();

  namespace diagnostic {
    struct stamp {
      std::uint64_t session {}, epoch {}, sequence {}, tick {}, frame_token {}, frame_numeric {}, command {};
      std::uint32_t viewport {UINT32_MAX};
      bool numeric_frame {};
    };
    enum class tag_scope : unsigned {
      global,
      frame,
      evaluation
    };
    // Invocation-scoped borrowed native identity. The observer may validate and
    // copy a resource synchronously here; it must not later follow this address
    // from metadata. No callback runs while the metadata storage lock is held.
    struct resource_observation {
      stamp observation;
      unsigned artifact_id {};
      std::uint64_t native {}, owner {}, source_id {};
      unsigned feature {UINT32_MAX}, tag_type {UINT32_MAX}, lifecycle {UINT32_MAX}, native_state {};
      sunshine_streamline::extent area;
      bool readable {}, descriptor_supported {}, state_declared {}, version_one {};
    };
    struct resource_callbacks {
      void (*resource)(const resource_observation &) noexcept {};
      void (*finish)(ui_resources::provider, const stamp &, std::uint64_t source_id, bool successful) noexcept {};
    };
    // Registered tables must outlive all in-flight callbacks (normally static
    // add-on lifetime). nullptr disables subsequent callback dispatch.
    void set_resource_callbacks(const resource_callbacks *) noexcept;
    // Module inspection runs during existing SDK discovery, never a vendor hook.
    void observe_module(HMODULE module, const char *provider, const char *abi) noexcept;
    void observe_sl_constants(const stamp &, const sunshine_streamline::abi_v1::constants &, bool readable, sunshine_streamline::decode_status decoded) noexcept;
    void observe_sl_constants(const stamp &, const sunshine_streamline::abi_v2::constants &, bool readable, sunshine_streamline::decode_status decoded, bool minimum_separation_available = false) noexcept;
    // Call synchronously before the original SDK function, while its arguments
    // are valid. Only bounded readable value prefixes are copied; no saved pointer
    // is subsequently followed and no texture is retained or read back.
    void observe_sl_tag_v1(const stamp &, const sunshine_streamline::abi_v1::resource *, std::uint32_t type, const sunshine_streamline::extent *) noexcept;
    void observe_sl_tags_v2(const stamp &, const sunshine_streamline::abi_v2::viewport &, const sunshine_streamline::abi_v2::resource_tag *, unsigned count, tag_scope) noexcept;
    void observe_sl_inputs_v2(const stamp &, const sunshine_streamline::base_structure **, unsigned count) noexcept;
    void observe_sl_evaluation(const stamp &, unsigned feature, bool successful) noexcept;
    void finish_sl_call(const stamp &, bool successful) noexcept;
    void observe_sl_fg(const stamp &, unsigned mode, unsigned generated_frames, bool supported, bool successful) noexcept;

    enum class parameter_type : unsigned {
      resource,
      pointer,
      unsigned_integer,
      signed_integer,
      floating
    };

    struct parameter {
      char name[64] {};
      parameter_type type {};
      bool getter_available {}, successful {};
      std::uint32_t result {};
      std::uint64_t integer {};
      double number {};
    };

    inline constexpr unsigned maximum_parameters = 48;

    struct ngx_evaluation {
      stamp observation;
      std::uint64_t owner {}, handle {}, source_id {}, scene_revision {};
      unsigned feature {}, create_width {}, create_height {};
      int create_flags {};
      std::array<parameter, maximum_parameters> parameters {};
      unsigned parameter_count {};
      bool truncated {}, feature_known {}, result_known {}, successful {};
    };

    void observe_ngx(const ngx_evaluation &) noexcept;
    void finish_ngx(const stamp &, std::uint64_t source_id, bool successful) noexcept;

  }  // namespace diagnostic
}  // namespace sunshine_game3d
