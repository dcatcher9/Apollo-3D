// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "streamline_depth_capture.h"

#include <d3d12.h>
#include <cstdint>

// The one owner of how an injected copy chooses its source's pre-copy state.
// Adapters say where their state claim comes from (state_proof); the capture
// owner says what the evaluating recording observed. A proof/policy pair
// selects exactly one rule, and every rule rejects blocked or incomplete
// observations. Nothing here guesses a state from texture type or format.
namespace sunshine_streamline::depth_capture::copy_state {
  inline constexpr std::uint32_t write_states = D3D12_RESOURCE_STATE_DEPTH_WRITE | D3D12_RESOURCE_STATE_COPY_DEST |
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS | D3D12_RESOURCE_STATE_RENDER_TARGET;

  // Ordinary legacy texture states only; a write state cannot be combined.
  // Enhanced, split and unknown states are never converted to a legacy state.
  constexpr bool supported(std::uint32_t value) {
    constexpr std::uint32_t allowed = D3D12_RESOURCE_STATE_DEPTH_WRITE | D3D12_RESOURCE_STATE_DEPTH_READ |
      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
      D3D12_RESOURCE_STATE_COPY_SOURCE | D3D12_RESOURCE_STATE_COPY_DEST |
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS | D3D12_RESOURCE_STATE_RENDER_TARGET;
    if (value & ~allowed) return false;
    return !(value & write_states) || ((value & (value - 1)) == 0);
  }

  enum class rule {
    none,                   // No usable state source.
    declared_must_match,    // The per-call declaration; an observed state must agree.
    observed_only,          // A nonzero state observed on this recording.
    observed_else_declared, // Observed wins; no entry permits a flag-compatible declaration.
    observed_else_contract, // Observed wins; no entry permits the SDK's input contract.
  };

  //                   | source_contract        | prefer_observed_recording
  //  unavailable      | none                   | none
  //  declared         | declared_must_match    | observed_else_declared
  //  observed_nonzero | observed_only          | observed_only
  //  sdk_contract     | observed_else_contract | observed_else_contract
  constexpr rule select(sunshine_scene_depth::state_proof proof, texture_state_policy policy) {
    using proof_kind = sunshine_scene_depth::state_proof;
    switch (proof) {
      case proof_kind::declared:
        return policy == texture_state_policy::prefer_observed_recording ? rule::observed_else_declared : rule::declared_must_match;
      case proof_kind::observed_nonzero: return rule::observed_only;
      case proof_kind::sdk_contract: return rule::observed_else_contract;
      default: return rule::none;
    }
  }

  // State evidence for the source on the exact recording that will carry the copy.
  struct observation {
    bool present{}, known{}, blocked{};
    std::uint32_t value{};
  };
  struct decision {
    bool admitted{};
    status result{status::unsupported_state};
    record_stage stage{record_stage::unsupported_proof};
    std::uint32_t state{}; // Transition and restore basis when admitted.
    bool used_observed{}, used_contract{};
  };

  constexpr decision reject(status result, record_stage stage) { return {false, result, stage}; }

  constexpr bool usable_claim(std::uint32_t value) { return value != 0 && value != UINT32_MAX; }

  // A declaration naming a bind state the allocation cannot have is not a claim
  // about this resource, so it cannot stand in for missing evidence.
  constexpr bool compatible_with(std::uint32_t state, std::uint32_t resource_flags) {
    return !((state & D3D12_RESOURCE_STATE_RENDER_TARGET) && !(resource_flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)) &&
      !((state & D3D12_RESOURCE_STATE_UNORDERED_ACCESS) && !(resource_flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) &&
      !((state & (D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_DEPTH_WRITE)) &&
        !(resource_flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL));
  }

  constexpr decision resolve(rule selected, std::uint32_t claimed, const observation &observed, std::uint32_t resource_flags) {
    if (selected == rule::none) return reject(status::unsupported_state, record_stage::unsupported_proof);
    // An in-progress split barrier or unrepresentable transition overrides every claim.
    if (observed.present && (observed.blocked || !observed.known))
      return reject(status::incomplete_state, record_stage::incomplete_state);
    decision out{true, status::recorded, record_stage::recorded, claimed};
    switch (selected) {
      case rule::declared_must_match:
        if (observed.present && observed.value != claimed)
          return reject(status::conflicting_state, record_stage::conflicting_state);
        break;
      case rule::observed_only:
      case rule::observed_else_declared:
      case rule::observed_else_contract:
        if (observed.present || selected == rule::observed_only) {
          // Observed COMMON/zero is evidence, not an absent entry.
          if (!observed.present || observed.value == 0) return reject(status::missing_state, record_stage::missing_state);
          out.state = observed.value;
          out.used_observed = true;
        } else if (!usable_claim(claimed)) {
          return reject(status::missing_state, record_stage::missing_state);
        } else if (selected == rule::observed_else_declared) {
          if (!compatible_with(claimed, resource_flags)) return reject(status::unsupported_state, record_stage::unsupported_state);
        } else {
          // The SDK contract describes a shader-readable input.
          if (resource_flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)
            return reject(status::missing_state, record_stage::missing_state);
          out.used_contract = true;
        }
        break;
      default: break;
    }
    if (!supported(out.state)) return reject(status::unsupported_state, record_stage::unsupported_state);
    return out;
  }
}
