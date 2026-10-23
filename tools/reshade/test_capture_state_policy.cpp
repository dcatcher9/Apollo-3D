// SPDX-License-Identifier: GPL-3.0-only
// CPU regression for the capture owner's pre-copy state table: its proof and
// policy selection, and named behaviours. The contract leaves the copy's own
// layouts to the add-on; the other examples pin current behaviour, and those
// that fall short of the contract name their known-gap row. The API contract
// matrix judges every legal state cell (reshade_contract_matrix_cpu,
// test_game3d_api_contract_cpu.cpp).
#include "capture_state_policy.h"

#include <cstdio>
#include <stdexcept>

namespace {
  namespace capture = sunshine_streamline::depth_capture;
  namespace copy_state = capture::copy_state;
  using proof = sunshine_scene_depth::state_proof;
  using policy = capture::texture_state_policy;

  void require(bool value, const char *message) {
    if (!value) {
      throw std::runtime_error(message);
    }
  }
}  // namespace

int main() try {
  using copy_state::rule;
  require(copy_state::select(proof::unavailable, policy::source_contract) == rule::observed_only &&
      copy_state::select(proof::unavailable, policy::prefer_observed_recording) == rule::observed_only &&
      copy_state::select(static_cast<proof>(99), policy::source_contract) == rule::none &&
      copy_state::select(proof::declared, policy::source_contract) == rule::declared_must_match &&
      copy_state::select(proof::declared, policy::prefer_observed_recording) == rule::observed_else_declared &&
      copy_state::select(proof::observed_nonzero, policy::source_contract) == rule::observed_only &&
      copy_state::select(proof::observed_nonzero, policy::prefer_observed_recording) == rule::observed_only &&
      copy_state::select(proof::sdk_contract, policy::source_contract) == rule::observed_else_contract &&
      copy_state::select(proof::sdk_contract, policy::prefer_observed_recording) == rule::observed_else_contract,
    "proof/policy table changed");

  // Named behaviors: pinned examples and how the copy builds its layouts
  // (copy_source, equivalent), which the contract leaves to the add-on.
  const auto contract = copy_state::select(proof::sdk_contract, policy::source_contract);
  const auto readable = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  auto chosen = copy_state::resolve(contract, readable, {}, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
  require(chosen.admitted && chosen.state == readable && chosen.used_contract && !chosen.used_observed,
    "absent evidence did not use the SDK input contract");
  chosen = copy_state::resolve(contract, readable, {true, true, false, D3D12_RESOURCE_STATE_DEPTH_READ | readable},
    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
  require(chosen.admitted && chosen.state == (D3D12_RESOURCE_STATE_DEPTH_READ | readable) && chosen.used_observed &&
    !chosen.used_contract, "observed state did not override the SDK contract");
  chosen = copy_state::resolve(contract, readable, {true, true, false, 0}, D3D12_RESOURCE_FLAG_NONE);
  require(!chosen.admitted && chosen.result == capture::status::missing_state, "observed COMMON satisfied the contract");
  chosen = copy_state::resolve(contract, readable, {},
    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE);
  require(!chosen.admitted && chosen.result == capture::status::missing_state, "contract accepted an unreadable allocation");
  const auto declared = copy_state::select(proof::declared, policy::source_contract);
  chosen = copy_state::resolve(declared, readable, {true, true, false, D3D12_RESOURCE_STATE_COPY_SOURCE}, 0);
  require(!chosen.admitted && chosen.result == capture::status::conflicting_state, "declaration overrode a different observation");
  // Enhanced layouts (D3D12 Enhanced Barriers): the copy keeps the exact
  // layout and access; a declaration agrees only through the contract's
  // equivalent layout; COMMON is an explicit layout, not missing evidence.
  namespace layouts = copy_state::layouts;
  for (std::uint32_t layout = 0; layout != 0x20; ++layout) {
    const bool direct = layout >= D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COMMON && layout <= D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_DEST;
    const bool generic = layout <= D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE;
    require(layouts::admissible(layout) == (direct || generic), "an enhanced layout's admission changed");
  }
  require(!layouts::admissible(D3D12_BARRIER_LAYOUT_UNDEFINED) && !layouts::admissible(D3D12_BARRIER_LAYOUT_RESOLVE_DEST) &&
      !layouts::admissible(D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_SHADER_RESOURCE) &&
      !layouts::admissible(D3D12_BARRIER_LAYOUT_VIDEO_DECODE_READ) && !layouts::admissible(D3D12_BARRIER_LAYOUT_VIDEO_QUEUE_COMMON),
    "an undefined, resolve-destination, compute-queue or video layout was admitted");
  require(layouts::copy_source(D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE) == D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_SOURCE &&
      layouts::copy_source(D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COMMON) == D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COMMON &&
      layouts::copy_source(D3D12_BARRIER_LAYOUT_COMMON) == D3D12_BARRIER_LAYOUT_COMMON &&
      layouts::copy_source(D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_GENERIC_READ) == D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_SOURCE &&
      layouts::copy_source(D3D12_BARRIER_LAYOUT_SHADER_RESOURCE) == D3D12_BARRIER_LAYOUT_COPY_SOURCE &&
      layouts::copy_source(D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ) == D3D12_BARRIER_LAYOUT_COPY_SOURCE,
    "the copy's source layout left its queue family");
  constexpr auto shader_states = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  require(layouts::equivalent(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) == D3D12_BARRIER_LAYOUT_SHADER_RESOURCE &&
      layouts::equivalent(shader_states) == D3D12_BARRIER_LAYOUT_SHADER_RESOURCE &&
      layouts::equivalent(D3D12_RESOURCE_STATE_DEPTH_READ | shader_states) == D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ &&
      layouts::equivalent(D3D12_RESOURCE_STATE_GENERIC_READ) == D3D12_BARRIER_LAYOUT_GENERIC_READ &&
      layouts::equivalent(0) == D3D12_BARRIER_LAYOUT_COMMON &&
      layouts::equivalent(D3D12_RESOURCE_STATE_COPY_SOURCE | shader_states) == D3D12_BARRIER_LAYOUT_UNDEFINED &&
      layouts::equivalent(UINT32_MAX) == D3D12_BARRIER_LAYOUT_UNDEFINED,
    "a legacy state's equivalent layout changed");
  const auto enhanced = [](std::uint32_t layout, bool known = true) {
    copy_state::observation value{true, known, false, 0, false, true, layout, D3D12_BARRIER_ACCESS_SHADER_RESOURCE};
    return value;
  };
  for (const auto layout : {D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_GENERIC_READ,
      D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COMMON, D3D12_BARRIER_LAYOUT_COMMON, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE}) {
    chosen = copy_state::resolve(rule::observed_only, UINT32_MAX, enhanced(layout), D3D12_RESOURCE_FLAG_NONE);
    require(chosen.admitted && chosen.enhanced && chosen.layout == std::uint32_t(layout) &&
        chosen.access == D3D12_BARRIER_ACCESS_SHADER_RESOURCE && chosen.used_observed,
      "an admissible enhanced layout was not kept exactly for the copy");
  }
  chosen = copy_state::resolve(rule::observed_only, 0, enhanced(D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_SHADER_RESOURCE, false), 0);
  require(!chosen.admitted && chosen.result == capture::status::incomplete_state, "an unknown enhanced layout was admitted");
  chosen = copy_state::resolve(declared, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    enhanced(D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE), 0);
  require(chosen.admitted && chosen.enhanced && chosen.layout == D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE &&
      !chosen.used_observed, "a declaration equivalent to the observed layout did not agree");
  chosen = copy_state::resolve(declared, D3D12_RESOURCE_STATE_DEPTH_READ, enhanced(D3D12_BARRIER_LAYOUT_SHADER_RESOURCE), 0);
  require(!chosen.admitted && chosen.result == capture::status::conflicting_state, "a different declaration agreed with a layout");
  chosen = copy_state::resolve(declared, D3D12_RESOURCE_STATE_COPY_SOURCE | shader_states,
    enhanced(D3D12_BARRIER_LAYOUT_SHADER_RESOURCE), 0);
  require(!chosen.admitted && chosen.result == capture::status::conflicting_state,
    "a declaration without an equivalent layout agreed with one");
  // A split barrier in transition rejects every rule, even beside a known value.
  for (const auto supplied : {proof::unavailable, proof::declared, proof::observed_nonzero, proof::sdk_contract}) {
    copy_state::observation pending{true, true, false, D3D12_RESOURCE_STATE_DEPTH_READ, true};
    chosen = copy_state::resolve(copy_state::select(supplied, policy::source_contract), D3D12_RESOURCE_STATE_DEPTH_READ, pending,
      D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    require(!chosen.admitted && chosen.result == capture::status::incomplete_state &&
      chosen.stage == capture::record_stage::incomplete_state, "a pending split barrier was admitted");
  }
  // Current behaviour, a recorded contract gap (D3b): a declared COMMON on a
  // promotable texture is admitted.
  chosen = copy_state::resolve(declared, 0, {}, 0);
  require(chosen.admitted && chosen.state == 0, "typed declared COMMON was not retained");
  // Assassin's Creed Shadows 10-22: depth observed in GENERIC_READ (0xac3), a
  // read-only legacy combination a copy can leave and restore.
  const auto observed_only = copy_state::select(proof::observed_nonzero, policy::source_contract);
  chosen = copy_state::resolve(observed_only, 0, {true, true, false, D3D12_RESOURCE_STATE_GENERIC_READ},
    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
  require(chosen.admitted && chosen.state == D3D12_RESOURCE_STATE_GENERIC_READ && chosen.used_observed,
    "an observed GENERIC_READ depth was not admitted");
  // An undeclared Streamline state (UINT_MAX) under either policy: only a
  // supported nonzero state observed on this recording is a proof.
  for (const auto chosen_policy : {policy::source_contract, policy::prefer_observed_recording}) {
    const auto undeclared = copy_state::select(proof::unavailable, chosen_policy);
    chosen = copy_state::resolve(undeclared, UINT32_MAX, {true, true, false, D3D12_RESOURCE_STATE_DEPTH_READ},
      D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    require(chosen.admitted && chosen.state == D3D12_RESOURCE_STATE_DEPTH_READ && chosen.used_observed,
      "an undeclared state was not taken from this recording's observation");
    chosen = copy_state::resolve(undeclared, UINT32_MAX, {}, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    require(!chosen.admitted && chosen.result == capture::status::missing_state, "an undeclared state was admitted without an observation");
    chosen = copy_state::resolve(undeclared, UINT32_MAX, {true, true, true, D3D12_RESOURCE_STATE_DEPTH_READ},
      D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    require(!chosen.admitted && chosen.result == capture::status::incomplete_state, "an undeclared state took a blocked observation");
    // Current behaviour, a recorded contract gap (D3, owner ruling R1).
    chosen = copy_state::resolve(undeclared, UINT32_MAX, {true, true, false, 0}, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    require(!chosen.admitted && chosen.result == capture::status::missing_state, "an undeclared state took an observed COMMON");
  }
  // RESOLVE_DEST's refusal is a recorded contract gap (D2).
  require(copy_state::supported(D3D12_RESOURCE_STATE_GENERIC_READ) &&
      copy_state::supported(D3D12_RESOURCE_STATE_RESOLVE_SOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE) &&
      !copy_state::supported(D3D12_RESOURCE_STATE_GENERIC_READ | D3D12_RESOURCE_STATE_DEPTH_WRITE) &&
      !copy_state::supported(D3D12_RESOURCE_STATE_GENERIC_READ | D3D12_RESOURCE_STATE_COPY_DEST) &&
      !copy_state::supported(D3D12_RESOURCE_STATE_RESOLVE_DEST) &&
      !copy_state::supported(D3D12_RESOURCE_STATE_VIDEO_DECODE_READ),
    "read-only legacy combinations and write or non-texture states were classified wrongly");
  std::printf("PASS capture state table: proof/policy table, copy layouts and named rules\n");
  return 0;
} catch (const std::exception &error) {
  std::fprintf(stderr, "FAIL capture state table: %s\n", error.what());
  return 1;
}
