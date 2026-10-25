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
#include <string>

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
      copy_state::select(proof::sdk_contract, policy::prefer_observed_recording) == rule::observed_else_contract &&
      copy_state::select(proof::clear_contract, policy::source_contract) == rule::declared_must_match &&
      copy_state::select(proof::clear_contract, policy::prefer_observed_recording) == rule::declared_must_match &&
      copy_state::select(proof::last_proven, policy::source_contract) == rule::observed_only &&
      copy_state::select(proof::last_proven, policy::prefer_observed_recording) == rule::observed_only &&
      copy_state::select(proof::declared_for_use, policy::source_contract) == rule::observed_over_declared &&
      copy_state::select(proof::declared_for_use, policy::prefer_observed_recording) == rule::observed_over_declared,
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
  // An observed COMMON is the state of a depth-stencil texture, which is never
  // promoted (D3, owner ruling R1); elsewhere it proves nothing (D3b).
  chosen = copy_state::resolve(contract, readable, {true, true, false, 0}, D3D12_RESOURCE_FLAG_NONE);
  require(!chosen.admitted && chosen.result == capture::status::missing_state, "observed COMMON satisfied the contract");
  chosen = copy_state::resolve(contract, readable, {true, true, false, 0}, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
  require(chosen.admitted && chosen.state == 0 && chosen.used_observed && !chosen.used_contract,
    "an observed COMMON on a depth-stencil texture did not override the contract");
  chosen = copy_state::resolve(contract, readable, {},
    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE);
  require(!chosen.admitted && chosen.result == capture::status::missing_state, "contract accepted an unreadable allocation");
  const auto declared = copy_state::select(proof::declared, policy::source_contract);
  chosen = copy_state::resolve(declared, readable, {true, true, false, D3D12_RESOURCE_STATE_COPY_SOURCE}, 0);
  require(!chosen.admitted && chosen.result == capture::status::conflicting_state, "declaration overrode a different observation");
  // A declaration of a later use (a longer-lived tag at its frame-generation
  // tag call, the matrix's S11): this recording's state is required and wins
  // over it; one that agrees stands; the observation rules still apply.
  const auto for_use = copy_state::select(proof::declared_for_use, policy::source_contract);
  chosen = copy_state::resolve(for_use, readable, {true, true, false, D3D12_RESOURCE_STATE_COPY_SOURCE}, 0);
  require(chosen.admitted && chosen.state == D3D12_RESOURCE_STATE_COPY_SOURCE && chosen.used_observed && !chosen.enhanced,
    "a declaration of a later use refused the state its recording observed");
  chosen = copy_state::resolve(for_use, readable, {true, true, false, D3D12_RESOURCE_STATE_DEPTH_WRITE},
    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
  require(chosen.admitted && chosen.state == D3D12_RESOURCE_STATE_DEPTH_WRITE && chosen.used_observed,
    "a declaration of a later use refused the observed write state");
  chosen = copy_state::resolve(for_use, readable, {true, true, false, readable}, 0);
  require(chosen.admitted && chosen.state == readable && !chosen.used_observed, "an agreeing declaration of a later use did not stand");
  chosen = copy_state::resolve(for_use, 0, {true, true, false, 0}, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
  require(chosen.admitted && chosen.state == 0, "an agreeing COMMON declaration of a later use did not stand");
  chosen = copy_state::resolve(for_use, readable, {}, 0);
  require(!chosen.admitted && chosen.result == capture::status::missing_state, "a declaration of a later use proved an unrecorded state");
  // A recorded COMMON wins there only on a depth-stencil texture (D3, S11).
  chosen = copy_state::resolve(for_use, readable, {true, true, false, 0}, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
  require(chosen.admitted && chosen.state == 0 && chosen.used_observed,
    "an observed COMMON on a depth-stencil texture did not override a declaration of a later use");
  chosen = copy_state::resolve(for_use, readable, {true, true, false, 0}, D3D12_RESOURCE_FLAG_NONE);
  require(!chosen.admitted && chosen.result == capture::status::missing_state,
    "an observed COMMON on a promotable texture overrode a declaration of a later use");
  // RESOLVE_DEST is a write state a direct list's copy may leave (D2, S11).
  chosen = copy_state::resolve(for_use, readable, {true, true, false, D3D12_RESOURCE_STATE_RESOLVE_DEST}, 0);
  require(chosen.admitted && chosen.state == D3D12_RESOURCE_STATE_RESOLVE_DEST && chosen.used_observed,
    "an observed RESOLVE_DEST did not override a declaration of a later use");
  chosen = copy_state::resolve(declared, D3D12_RESOURCE_STATE_RESOLVE_DEST, {}, 0);
  require(chosen.admitted && chosen.state == D3D12_RESOURCE_STATE_RESOLVE_DEST && !chosen.used_observed,
    "a declared RESOLVE_DEST was refused");
  chosen = copy_state::resolve(for_use, readable, {true, true, true, D3D12_RESOURCE_STATE_COPY_SOURCE}, 0);
  require(!chosen.admitted && chosen.result == capture::status::incomplete_state, "a declaration of a later use took a blocked observation");
  // Enhanced layouts (D3D12 Enhanced Barriers): the copy keeps the exact
  // layout and access; a declaration agrees only through the contract's
  // equivalent layout; COMMON is an explicit layout, not missing evidence.
  namespace layouts = copy_state::layouts;
  for (std::uint32_t layout = 0; layout != 0x20; ++layout) {
    const bool direct = layout >= D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COMMON && layout <= D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_DEST;
    const bool generic = layout <= D3D12_BARRIER_LAYOUT_RESOLVE_DEST;
    require(layouts::admissible(layout) == (direct || generic), "an enhanced layout's admission changed");
  }
  require(!layouts::admissible(D3D12_BARRIER_LAYOUT_UNDEFINED) && layouts::admissible(D3D12_BARRIER_LAYOUT_RESOLVE_DEST) &&
      !layouts::admissible(D3D12_BARRIER_LAYOUT_SHADING_RATE_SOURCE) &&
      !layouts::admissible(D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_SHADER_RESOURCE) &&
      !layouts::admissible(D3D12_BARRIER_LAYOUT_VIDEO_DECODE_READ) && !layouts::admissible(D3D12_BARRIER_LAYOUT_VIDEO_QUEUE_COMMON),
    "an undefined, shading-rate, compute-queue or video layout was admitted, or resolve-destination refused");
  require(layouts::copy_source(D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE) == D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_SOURCE &&
      layouts::copy_source(D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COMMON) == D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COMMON &&
      layouts::copy_source(D3D12_BARRIER_LAYOUT_COMMON) == D3D12_BARRIER_LAYOUT_COMMON &&
      layouts::copy_source(D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_GENERIC_READ) == D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_SOURCE &&
      layouts::copy_source(D3D12_BARRIER_LAYOUT_SHADER_RESOURCE) == D3D12_BARRIER_LAYOUT_COPY_SOURCE &&
      layouts::copy_source(D3D12_BARRIER_LAYOUT_RESOLVE_DEST) == D3D12_BARRIER_LAYOUT_COPY_SOURCE &&
      layouts::copy_source(D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ) == D3D12_BARRIER_LAYOUT_COPY_SOURCE &&
      layouts::copy_source(D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_SHADER_RESOURCE) == D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COPY_SOURCE &&
      layouts::copy_source(D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COMMON) == D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COMMON,
    "the copy's source layout left its queue family");
  // A COMPUTE list (S22, S22b, N7): the states and layouts its queue
  // supports, and nothing a direct list alone may name.
  for (std::uint32_t layout = 0; layout != 0x20; ++layout) {
    const bool compute = layout >= D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COMMON && layout <= D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COPY_DEST;
    require(layouts::compute_queue(layout) == compute, "a compute queue's own layout changed");
    const bool generic = layout == D3D12_BARRIER_LAYOUT_COMMON || layout == D3D12_BARRIER_LAYOUT_GENERIC_READ ||
      layout == D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS || layout == D3D12_BARRIER_LAYOUT_SHADER_RESOURCE ||
      layout == D3D12_BARRIER_LAYOUT_COPY_SOURCE || layout == D3D12_BARRIER_LAYOUT_COPY_DEST;
    require(layouts::compute_list(layout) == (compute || generic), "a compute list's layouts changed");
  }
  require(!layouts::compute_list(D3D12_BARRIER_LAYOUT_UNDEFINED), "a compute list admitted an undefined layout");
  for (const std::uint32_t state : {0u, 0x8u, 0x40u, 0x400u, 0x800u, 0x40u | 0x800u}) {
    require(copy_state::compute_list_state(state), "a compute list refused a state its queue supports");
  }
  for (const std::uint32_t state : {0x10u, 0x20u, 0x80u, 0x4u, 0x1000u, 0x2000u, 0x40u | 0x80u, 0x20u | 0x40u,
                                    static_cast<std::uint32_t>(D3D12_RESOURCE_STATE_GENERIC_READ)}) {
    require(!copy_state::compute_list_state(state), "a compute list admitted a direct queue's state");
  }
  {
    copy_state::decision legacy{true, capture::status::recorded, capture::record_stage::recorded, 0x40};
    copy_state::decision depth{true, capture::status::recorded, capture::record_stage::recorded, 0x20 | 0x40};
    copy_state::decision compute_layout{true, capture::status::recorded, capture::record_stage::recorded, 0};
    compute_layout.enhanced = true; compute_layout.layout = D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_SHADER_RESOURCE;
    copy_state::decision direct_layout = compute_layout;
    direct_layout.layout = D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE;
    copy_state::decision depth_layout = compute_layout;
    depth_layout.layout = D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ;
    require(copy_state::list_basis(legacy, true) && copy_state::list_basis(legacy, false) &&
        !copy_state::list_basis(depth, true) && copy_state::list_basis(depth, false) &&
        copy_state::list_basis(compute_layout, true) && !copy_state::list_basis(compute_layout, false) &&
        !copy_state::list_basis(direct_layout, true) && copy_state::list_basis(direct_layout, false) &&
        !copy_state::list_basis(depth_layout, true) && copy_state::list_basis(depth_layout, false),
      "a copy basis was judged against the wrong list type");
  }
  constexpr auto shader_states = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  require(layouts::equivalent(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) == D3D12_BARRIER_LAYOUT_SHADER_RESOURCE &&
      layouts::equivalent(shader_states) == D3D12_BARRIER_LAYOUT_SHADER_RESOURCE &&
      layouts::equivalent(D3D12_RESOURCE_STATE_DEPTH_READ | shader_states) == D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ &&
      layouts::equivalent(D3D12_RESOURCE_STATE_GENERIC_READ) == D3D12_BARRIER_LAYOUT_GENERIC_READ &&
      layouts::equivalent(D3D12_RESOURCE_STATE_RESOLVE_DEST) == D3D12_BARRIER_LAYOUT_RESOLVE_DEST &&
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
  chosen = copy_state::resolve(for_use, D3D12_RESOURCE_STATE_DEPTH_READ, enhanced(D3D12_BARRIER_LAYOUT_SHADER_RESOURCE), 0);
  require(chosen.admitted && chosen.enhanced && chosen.layout == D3D12_BARRIER_LAYOUT_SHADER_RESOURCE &&
      chosen.access == D3D12_BARRIER_ACCESS_SHADER_RESOURCE && chosen.used_observed,
    "a declaration of a later use refused the layout its recording observed");
  chosen = copy_state::resolve(declared, D3D12_RESOURCE_STATE_COPY_SOURCE | shader_states,
    enhanced(D3D12_BARRIER_LAYOUT_SHADER_RESOURCE), 0);
  require(!chosen.admitted && chosen.result == capture::status::conflicting_state,
    "a declaration without an equivalent layout agreed with one");
  // A layout an enhanced barrier set on an earlier list (D6, N8; owner ruling
  // D6-relax): whatever layout that list proved is the copy's, with enhanced
  // barriers, never a legacy barrier from the claim; the claim's rule decides
  // against it as against an observation, and the recording's list type must
  // name it (list_basis).
  const auto earlier = [&](std::uint32_t layout, bool proven = true, std::uint32_t access = D3D12_BARRIER_ACCESS_SHADER_RESOURCE) {
    auto value = enhanced(layout);
    value.present = value.known = value.enhanced = proven;
    value.access = access;
    value.earlier = true;
    return value;
  };
  const auto at_call_ui = copy_state::select(proof::declared, policy::prefer_observed_recording);
  struct earlier_copy {
    std::uint32_t claim, layout, access;
  };
  for (const auto selected : {declared, contract, at_call_ui}) {
    for (const auto &value : {
           earlier_copy{D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, D3D12_BARRIER_ACCESS_SHADER_RESOURCE},
           earlier_copy{shader_states, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, D3D12_BARRIER_ACCESS_SHADER_RESOURCE},
           earlier_copy{D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE, D3D12_BARRIER_ACCESS_SHADER_RESOURCE},
           earlier_copy{D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_SHADER_RESOURCE, D3D12_BARRIER_ACCESS_SHADER_RESOURCE},
           earlier_copy{D3D12_RESOURCE_STATE_DEPTH_READ | shader_states, D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ, D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ},
           earlier_copy{D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE, D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE},
           earlier_copy{D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_BARRIER_LAYOUT_COPY_SOURCE, D3D12_BARRIER_ACCESS_COPY_SOURCE},
         }) {
      chosen = copy_state::resolve(selected, value.claim, earlier(value.layout, true, value.access), D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
      require(chosen.admitted && chosen.enhanced && chosen.layout == value.layout && chosen.access == value.access,
        "a claim that agrees with its earlier proof was not copied from the proven layout");
    }
    chosen = copy_state::resolve(selected, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, earlier(D3D12_BARRIER_LAYOUT_SHADER_RESOURCE),
      D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE);
    require(!chosen.admitted && chosen.result == capture::status::unsupported_state, "an earlier layout of a texture shaders may not read was copied");
  }
  // A layout of one queue type is no LayoutBefore on another's list.
  chosen = copy_state::resolve(declared, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    earlier(D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_SHADER_RESOURCE), D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
  require(chosen.admitted && !copy_state::list_basis(chosen, false) && copy_state::list_basis(chosen, true),
    "a compute queue's earlier layout was not refused on a direct list, or not admitted on a compute list");
  chosen = copy_state::resolve(declared, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    earlier(D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE), D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
  require(chosen.admitted && copy_state::list_basis(chosen, false) && !copy_state::list_basis(chosen, true),
    "a direct queue's earlier layout was not refused on a compute list, or not admitted on a direct list");
  // A claim that names another layout than the proof: a declaration must
  // agree, while an SDK contract and an at-call UI declaration give way to it
  // as to an observation.
  chosen = copy_state::resolve(declared, D3D12_RESOURCE_STATE_DEPTH_WRITE, earlier(D3D12_BARRIER_LAYOUT_SHADER_RESOURCE),
    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
  require(!chosen.admitted && chosen.result == capture::status::conflicting_state, "a declaration that names another layout than its earlier proof was copied");
  for (const auto selected : {contract, at_call_ui}) {
    chosen = copy_state::resolve(selected, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
      earlier(D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ, true, D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ), D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    require(chosen.admitted && chosen.enhanced && chosen.layout == D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ && chosen.used_observed,
      "an earlier proof did not stand over a claim that gives way to an observation");
  }
  struct earlier_case {
    std::uint32_t claim;
    copy_state::observation observed;
    const char *what;
  };
  for (const auto &value : {
         earlier_case{D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, earlier(D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, false), "an unproven latched texture"},
         earlier_case{D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, earlier(D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, true, D3D12_BARRIER_ACCESS_NO_ACCESS), "an earlier layout without access"},
       }) {
    chosen = copy_state::resolve(declared, value.claim, value.observed, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    require(!chosen.admitted && chosen.result == capture::status::unsupported_state && chosen.stage == capture::record_stage::unsupported_state,
      (std::string(value.what) + " was copied from a claim").c_str());
  }
  // A source that needs a state recorded on its own list takes none from an
  // earlier one; a clear's contract is checked against it like a declaration.
  for (const auto supplied : {proof::unavailable, proof::observed_nonzero, proof::last_proven, proof::declared_for_use}) {
    chosen = copy_state::resolve(copy_state::select(supplied, policy::source_contract), D3D12_RESOURCE_STATE_DEPTH_WRITE,
      earlier(D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE, true, D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE), D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    require(!chosen.admitted && chosen.result == capture::status::unsupported_state,
      "a source needing an observation of its own was copied from an earlier layout");
  }
  chosen = copy_state::resolve(copy_state::select(proof::clear_contract, policy::source_contract), D3D12_RESOURCE_STATE_DEPTH_WRITE,
    earlier(D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE, true, D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE), D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
  require(chosen.admitted && chosen.enhanced && chosen.layout == D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE,
    "a clear's DEPTH_WRITE was not copied from the depth-stencil-write layout an earlier list proved");
  chosen = copy_state::resolve(copy_state::select(proof::clear_contract, policy::source_contract), D3D12_RESOURCE_STATE_DEPTH_WRITE,
    earlier(D3D12_BARRIER_LAYOUT_SHADER_RESOURCE), D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
  require(!chosen.admitted && chosen.result == capture::status::conflicting_state, "a clear's DEPTH_WRITE agreed with another earlier layout");
  // A split barrier in transition rejects every rule, even beside a known value.
  for (const auto supplied : {proof::unavailable, proof::declared, proof::observed_nonzero, proof::sdk_contract, proof::declared_for_use}) {
    copy_state::observation pending{true, true, false, D3D12_RESOURCE_STATE_DEPTH_READ, true};
    chosen = copy_state::resolve(copy_state::select(supplied, policy::source_contract), D3D12_RESOURCE_STATE_DEPTH_READ, pending,
      D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    require(!chosen.admitted && chosen.result == capture::status::incomplete_state &&
      chosen.stage == capture::record_stage::incomplete_state, "a pending split barrier was admitted");
  }
  // A declared COMMON is the state of a depth-stencil texture (D3) and of no
  // texture that COMMON may have been promoted out of (D3b): a colour or
  // linear-depth texture, simultaneous-access or not.
  chosen = copy_state::resolve(declared, 0, {}, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
  require(chosen.admitted && chosen.state == 0 && !chosen.used_observed, "a declared COMMON on a depth-stencil texture was refused");
  for (const std::uint32_t flags : {std::uint32_t(D3D12_RESOURCE_FLAG_NONE), std::uint32_t(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
         std::uint32_t(D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)}) {
    for (const auto &observed : {copy_state::observation{}, copy_state::observation{true, true, false, 0}}) {
      chosen = copy_state::resolve(declared, 0, observed, flags);
      require(!chosen.admitted && chosen.result == capture::status::missing_state && chosen.stage == capture::record_stage::missing_state,
        "a declared COMMON on a promotable texture was copied from");
    }
  }
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
    // An observed COMMON proves a depth-stencil texture's state (D3) and
    // nothing else's (D3b).
    chosen = copy_state::resolve(undeclared, UINT32_MAX, {true, true, false, 0}, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    require(chosen.admitted && chosen.state == 0 && chosen.used_observed, "an undeclared state refused an observed COMMON on depth-stencil");
    chosen = copy_state::resolve(undeclared, UINT32_MAX, {true, true, false, 0}, D3D12_RESOURCE_FLAG_NONE);
    require(!chosen.admitted && chosen.result == capture::status::missing_state, "an undeclared state took an observed COMMON on a promotable texture");
  }
  // RESOLVE_DEST is a write state like the others (D2). A buffer read bit is
  // a texture state only as part of GENERIC_READ itself (X2).
  require(copy_state::supported(D3D12_RESOURCE_STATE_GENERIC_READ) &&
      !copy_state::supported(D3D12_RESOURCE_STATE_GENERIC_READ | D3D12_RESOURCE_STATE_DEPTH_READ) &&
      !copy_state::supported(D3D12_RESOURCE_STATE_GENERIC_READ | D3D12_RESOURCE_STATE_RESOLVE_SOURCE) &&
      !copy_state::supported(D3D12_RESOURCE_STATE_INDEX_BUFFER) &&
      !copy_state::supported(D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT | D3D12_RESOURCE_STATE_COPY_SOURCE) &&
      !copy_state::supported(D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) &&
      copy_state::supported(D3D12_RESOURCE_STATE_RESOLVE_SOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE) &&
      !copy_state::supported(D3D12_RESOURCE_STATE_GENERIC_READ | D3D12_RESOURCE_STATE_DEPTH_WRITE) &&
      !copy_state::supported(D3D12_RESOURCE_STATE_GENERIC_READ | D3D12_RESOURCE_STATE_COPY_DEST) &&
      copy_state::supported(D3D12_RESOURCE_STATE_RESOLVE_DEST) &&
      !copy_state::supported(D3D12_RESOURCE_STATE_RESOLVE_DEST | D3D12_RESOURCE_STATE_COPY_SOURCE) &&
      !copy_state::supported(D3D12_RESOURCE_STATE_VIDEO_DECODE_READ),
    "read-only legacy combinations and write or non-texture states were classified wrongly");
  // Every basis must be one the allocation's flags allow (X1): an observed
  // state or layout, a declaration or a recorded state over a declaration
  // of a later use, whatever the rule.
  constexpr std::uint32_t ds = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
  constexpr std::uint32_t unreadable = ds | std::uint32_t(D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE);
  struct flags_case {
    rule selected;
    std::uint32_t claim;
    copy_state::observation observed;
    std::uint32_t flags;
    const char *what;
  };
  for (const auto &value : {
         flags_case{observed_only, 0, {true, true, false, D3D12_RESOURCE_STATE_DEPTH_WRITE}, 0, "an observed DEPTH_WRITE without ALLOW_DEPTH_STENCIL"},
         flags_case{observed_only, 0, {true, true, false, D3D12_RESOURCE_STATE_RENDER_TARGET}, ds, "an observed RENDER_TARGET without ALLOW_RENDER_TARGET"},
         flags_case{observed_only, 0, {true, true, false, D3D12_RESOURCE_STATE_UNORDERED_ACCESS}, 0, "an observed UNORDERED_ACCESS without ALLOW_UNORDERED_ACCESS"},
         flags_case{declared, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, {}, unreadable, "a declared shader read with DENY_SHADER_RESOURCE"},
         flags_case{declared, D3D12_RESOURCE_STATE_GENERIC_READ, {}, unreadable, "a declared GENERIC_READ with DENY_SHADER_RESOURCE"},
         flags_case{declared, D3D12_RESOURCE_STATE_DEPTH_WRITE, {true, true, false, D3D12_RESOURCE_STATE_DEPTH_WRITE}, 0, "an agreeing DEPTH_WRITE without ALLOW_DEPTH_STENCIL"},
         flags_case{for_use, readable, {true, true, false, D3D12_RESOURCE_STATE_RENDER_TARGET}, 0, "a recorded RENDER_TARGET without ALLOW_RENDER_TARGET"},
         flags_case{at_call_ui, D3D12_RESOURCE_STATE_RENDER_TARGET, {}, 0, "a UI declaration of RENDER_TARGET without ALLOW_RENDER_TARGET"},
         flags_case{observed_only, 0, enhanced(D3D12_BARRIER_LAYOUT_RENDER_TARGET), ds, "an observed RENDER_TARGET layout without ALLOW_RENDER_TARGET"},
         flags_case{observed_only, 0, enhanced(D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS), 0, "an observed direct-queue UAV layout without ALLOW_UNORDERED_ACCESS"},
         flags_case{observed_only, 0, enhanced(D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ), 0, "an observed depth-read layout without ALLOW_DEPTH_STENCIL"},
         flags_case{contract, readable, enhanced(D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE), unreadable, "an observed shader-resource layout with DENY_SHADER_RESOURCE"},
       }) {
    chosen = copy_state::resolve(value.selected, value.claim, value.observed, value.flags);
    require(!chosen.admitted && chosen.result == capture::status::unsupported_state && chosen.stage == capture::record_stage::unsupported_state,
      (std::string(value.what) + " was copied from").c_str());
  }
  // The flags the contract leaves free: GENERIC_READ's layout needs none.
  chosen = copy_state::resolve(observed_only, 0, enhanced(D3D12_BARRIER_LAYOUT_GENERIC_READ), unreadable);
  require(chosen.admitted && chosen.enhanced && chosen.layout == D3D12_BARRIER_LAYOUT_GENERIC_READ,
    "a GENERIC_READ layout was refused on a texture shaders may not read");
  std::printf("PASS capture state table: proof/policy table, copy layouts and named rules\n");
  return 0;
} catch (const std::exception &error) {
  std::fprintf(stderr, "FAIL capture state table: %s\n", error.what());
  return 1;
}
