// SPDX-License-Identifier: GPL-3.0-only
// CPU regression for the capture owner's pre-copy state table. The previous
// inline decision is kept here as a reference so the table stays equivalent.
#include "capture_state_policy.h"

#include <cstdio>
#include <stdexcept>

namespace {
  namespace capture = sunshine_streamline::depth_capture;
  namespace copy_state = capture::copy_state;
  using proof = sunshine_scene_depth::state_proof;
  using policy = capture::texture_state_policy;
  void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }

  // Former record_impl logic, in its original order.
  copy_state::decision reference(proof supplied, policy chosen, std::uint32_t claimed,
      const copy_state::observation &observed, std::uint32_t flags) {
    const bool prefer = chosen == policy::prefer_observed_recording;
    if (supplied != proof::declared && supplied != proof::observed_nonzero && supplied != proof::sdk_contract)
      return copy_state::reject(capture::status::unsupported_state, capture::record_stage::unsupported_proof);
    auto before = claimed;
    if (observed.present && (observed.blocked || !observed.known))
      return copy_state::reject(capture::status::incomplete_state, capture::record_stage::incomplete_state);
    const bool contract = supplied == proof::sdk_contract;
    const bool use_observed = ((prefer || contract) && observed.present) || supplied == proof::observed_nonzero;
    if (contract && !observed.present) {
      if (before == 0 || before == UINT32_MAX || (flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE))
        return copy_state::reject(capture::status::missing_state, capture::record_stage::missing_state);
    } else if (use_observed) {
      if (!observed.present || observed.value == 0)
        return copy_state::reject(capture::status::missing_state, capture::record_stage::missing_state);
      before = observed.value;
    } else if (observed.present && observed.value != before) {
      return copy_state::reject(capture::status::conflicting_state, capture::record_stage::conflicting_state);
    }
    if (prefer && !observed.present) {
      if (supplied != proof::declared || before == 0 || before == UINT32_MAX)
        return copy_state::reject(capture::status::missing_state, capture::record_stage::missing_state);
      if (((before & D3D12_RESOURCE_STATE_RENDER_TARGET) && !(flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)) ||
          ((before & D3D12_RESOURCE_STATE_UNORDERED_ACCESS) && !(flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) ||
          ((before & (D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_DEPTH_WRITE)) &&
            !(flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)))
        return copy_state::reject(capture::status::unsupported_state, capture::record_stage::unsupported_state);
    }
    if (!copy_state::supported(before))
      return copy_state::reject(capture::status::unsupported_state, capture::record_stage::unsupported_state);
    return {true, capture::status::recorded, capture::record_stage::recorded, before, use_observed, contract && !observed.present};
  }

  bool same(const copy_state::decision &a, const copy_state::decision &b) {
    if (a.admitted != b.admitted || a.result != b.result || a.stage != b.stage) return false;
    return !a.admitted || (a.state == b.state && a.used_observed == b.used_observed && a.used_contract == b.used_contract);
  }
}

int main() try {
  using copy_state::rule;
  require(copy_state::select(proof::unavailable, policy::source_contract) == rule::none &&
      copy_state::select(proof::unavailable, policy::prefer_observed_recording) == rule::none &&
      copy_state::select(proof::declared, policy::source_contract) == rule::declared_must_match &&
      copy_state::select(proof::declared, policy::prefer_observed_recording) == rule::observed_else_declared &&
      copy_state::select(proof::observed_nonzero, policy::source_contract) == rule::observed_only &&
      copy_state::select(proof::observed_nonzero, policy::prefer_observed_recording) == rule::observed_only &&
      copy_state::select(proof::sdk_contract, policy::source_contract) == rule::observed_else_contract &&
      copy_state::select(proof::sdk_contract, policy::prefer_observed_recording) == rule::observed_else_contract,
    "proof/policy table changed");

  constexpr std::uint32_t states[]{0, UINT32_MAX, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE,
    D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_DEPTH_READ,
    D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RESOLVE_DEST,
    D3D12_RESOURCE_STATE_RENDER_TARGET | D3D12_RESOURCE_STATE_DEPTH_WRITE};
  constexpr std::uint32_t flag_sets[]{D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE,
    D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
  unsigned compared = 0;
  for (const auto supplied : {proof::unavailable, proof::declared, proof::observed_nonzero, proof::sdk_contract})
    for (const auto chosen : {policy::source_contract, policy::prefer_observed_recording})
      for (const auto claimed : states)
        for (const auto flags : flag_sets)
          for (unsigned kind = 0; kind != 3 + std::size(states); ++kind) {
            copy_state::observation observed;
            if (kind == 1) observed = {true, true, true, D3D12_RESOURCE_STATE_COPY_SOURCE}; // Pending split barrier.
            else if (kind == 2) observed = {true, false, false, D3D12_RESOURCE_STATE_RESOLVE_DEST}; // Unrepresentable.
            else if (kind >= 3) observed = {true, true, false, states[kind - 3]};
            const auto actual = copy_state::resolve(copy_state::select(supplied, chosen), claimed, observed, flags);
            if (supplied == proof::sdk_contract && chosen == policy::prefer_observed_recording && !observed.present) {
              // Unreachable in production (that policy is Streamline-only); the
              // table now states the contract rule instead of rejecting.
              require(same(actual, reference(supplied, policy::source_contract, claimed, observed, flags)),
                "SDK contract ignored the policy-independent table");
              continue;
            }
            if (!same(actual, reference(supplied, chosen, claimed, observed, flags))) {
              std::fprintf(stderr, "proof=%u policy=%u claimed=0x%x flags=0x%x observed=%u/%u/%u/0x%x\n",
                unsigned(supplied), unsigned(chosen), claimed, flags, observed.present, observed.known,
                observed.blocked, observed.value);
              throw std::runtime_error("state table diverged from the former inline decision");
            }
            ++compared;
          }

  // Named behaviors, independent of the reference.
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
  chosen = copy_state::resolve(declared, 0, {}, 0);
  require(chosen.admitted && chosen.state == 0, "typed declared COMMON was not retained");
  std::printf("PASS capture state table: %u proof/policy/claim/observation/flag combinations match the former decision; contract, observation, COMMON and flag rules\n", compared);
  return 0;
} catch (const std::exception &error) {
  std::fprintf(stderr, "FAIL capture state table: %s\n", error.what());
  return 1;
}
