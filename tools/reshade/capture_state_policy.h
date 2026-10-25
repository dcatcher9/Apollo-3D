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
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS | D3D12_RESOURCE_STATE_RENDER_TARGET | D3D12_RESOURCE_STATE_RESOLVE_DEST;

  // Ordinary legacy texture states only: D3D12 permits each on a texture and
  // a copy may transition from it to COPY_SOURCE and back. Any combination of
  // the texture read states is one, and so is GENERIC_READ itself (Assassin's
  // Creed Shadows 10-22 kept its depth in it, so every capture refused the
  // depth as an unknown state). GENERIC_READ's other bits are buffer states
  // (VERTEX_AND_CONSTANT_BUFFER, INDEX_BUFFER, INDIRECT_ARGUMENT): the D3D12
  // runtime refuses a texture barrier into any other state that holds one,
  // and a copy's barrier from it makes the game's own Close fail
  // (E_INVALIDARG; the contract matrix's X2). Each write state alone is one
  // too, RESOLVE_DEST included (a direct list's resolve target; D2). Enhanced,
  // split and unknown states are never converted to a legacy state.
  constexpr bool supported(std::uint32_t value) {
    constexpr std::uint32_t texture_reads = D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
      D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_COPY_SOURCE | D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
    if (value == D3D12_RESOURCE_STATE_GENERIC_READ) return true;
    if (value & ~(write_states | texture_reads)) return false;
    return !(value & write_states) || ((value & (value - 1)) == 0);
  }

  // Whether the allocation's flags let a texture be in a legacy state (the
  // contract matrix's X1; D3D12_RESOURCE_FLAGS): ALLOW_RENDER_TARGET,
  // ALLOW_UNORDERED_ACCESS and ALLOW_DEPTH_STENCIL each enable the resource
  // to transition into their states, and DENY_SHADER_RESOURCE disables the
  // shader-resource ones. A state the allocation cannot be in is illegal
  // input, never a copy's basis.
  constexpr bool allowed(std::uint32_t state, std::uint32_t resource_flags) {
    constexpr std::uint32_t shader = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    return !((state & D3D12_RESOURCE_STATE_RENDER_TARGET) && !(resource_flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)) &&
      !((state & D3D12_RESOURCE_STATE_UNORDERED_ACCESS) && !(resource_flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) &&
      !((state & (D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_DEPTH_WRITE)) &&
        !(resource_flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)) &&
      !((state & shader) && (resource_flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE));
  }

  // Enhanced (CommandList7) layouts. The D3D12 Enhanced Barriers contract
  // forbids a legacy barrier on a texture in an enhanced layout other than
  // COMMON and defines legacy equivalents only for the generic layouts, so a
  // texture whose latest barrier on the recording was an enhanced Barrier is
  // copied with enhanced barriers from, and back to, its exact layout and
  // access. Legacy states are compared with a layout only through the
  // contract's equivalence table.
  namespace layouts {
    // Admissible for a copy on a direct command list (the only list type the
    // capture records on): the generic read/write layouts a legacy copy
    // admits, resolve destination among them (Enhanced Barriers, Command
    // Queue Layout Compatibility, lists it for DIRECT; D2), and the
    // direct-queue forms. Compute- and video-queue, video, shading-rate,
    // undefined and unknown layouts are not.
    constexpr bool admissible(std::uint32_t layout) {
      switch (static_cast<D3D12_BARRIER_LAYOUT>(layout)) {
        case D3D12_BARRIER_LAYOUT_COMMON: case D3D12_BARRIER_LAYOUT_GENERIC_READ:
        case D3D12_BARRIER_LAYOUT_RENDER_TARGET: case D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS:
        case D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE: case D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ:
        case D3D12_BARRIER_LAYOUT_SHADER_RESOURCE: case D3D12_BARRIER_LAYOUT_COPY_SOURCE:
        case D3D12_BARRIER_LAYOUT_COPY_DEST: case D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE:
        case D3D12_BARRIER_LAYOUT_RESOLVE_DEST:
        case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COMMON: case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_GENERIC_READ:
        case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS: case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE:
        case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_SOURCE: case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_DEST:
          return true;
        default: return false;
      }
    }
    constexpr bool direct_queue(std::uint32_t layout) {
      return layout >= D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COMMON && layout <= D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_DEST;
    }
    // The generic layout a queue-specific layout specializes; itself otherwise.
    constexpr std::uint32_t generic(std::uint32_t layout) {
      switch (static_cast<D3D12_BARRIER_LAYOUT>(layout)) {
        case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COMMON: case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COMMON:
          return D3D12_BARRIER_LAYOUT_COMMON;
        case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_GENERIC_READ: case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_GENERIC_READ:
          return D3D12_BARRIER_LAYOUT_GENERIC_READ;
        case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS: case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_UNORDERED_ACCESS:
          return D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS;
        case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE: case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_SHADER_RESOURCE:
          return D3D12_BARRIER_LAYOUT_SHADER_RESOURCE;
        case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_SOURCE: case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COPY_SOURCE:
          return D3D12_BARRIER_LAYOUT_COPY_SOURCE;
        case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_DEST: case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COPY_DEST:
          return D3D12_BARRIER_LAYOUT_COPY_DEST;
        default: return layout;
      }
    }
    // The contract's layout for a legacy state (its "equivalent layout" table),
    // or UNDEFINED where it names none. Shader-resource bits with DEPTH_READ
    // are the depth-read layout, which shaders may also read.
    constexpr std::uint32_t equivalent(std::uint32_t state) {
      constexpr std::uint32_t shader = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
      if (state == D3D12_RESOURCE_STATE_COMMON) return D3D12_BARRIER_LAYOUT_COMMON;
      if (state == D3D12_RESOURCE_STATE_GENERIC_READ) return D3D12_BARRIER_LAYOUT_GENERIC_READ;
      if ((state & ~shader) == D3D12_RESOURCE_STATE_DEPTH_READ) return D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ;
      if (!(state & ~shader)) return D3D12_BARRIER_LAYOUT_SHADER_RESOURCE;
      switch (state) {
        case D3D12_RESOURCE_STATE_RENDER_TARGET: return D3D12_BARRIER_LAYOUT_RENDER_TARGET;
        case D3D12_RESOURCE_STATE_UNORDERED_ACCESS: return D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS;
        case D3D12_RESOURCE_STATE_DEPTH_WRITE: return D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE;
        case D3D12_RESOURCE_STATE_COPY_SOURCE: return D3D12_BARRIER_LAYOUT_COPY_SOURCE;
        case D3D12_RESOURCE_STATE_COPY_DEST: return D3D12_BARRIER_LAYOUT_COPY_DEST;
        case D3D12_RESOURCE_STATE_RESOLVE_SOURCE: return D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE;
        case D3D12_RESOURCE_STATE_RESOLVE_DEST: return D3D12_BARRIER_LAYOUT_RESOLVE_DEST;
        default: return D3D12_BARRIER_LAYOUT_UNDEFINED;
      }
    }
    // Whether the allocation's flags let a texture be in a layout: those its
    // legacy equivalent needs (allowed above; X1).
    constexpr bool allowed(std::uint32_t layout, std::uint32_t resource_flags) {
      switch (static_cast<D3D12_BARRIER_LAYOUT>(generic(layout))) {
        case D3D12_BARRIER_LAYOUT_RENDER_TARGET: return resource_flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        case D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS: return resource_flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        case D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE: case D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ:
          return resource_flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        case D3D12_BARRIER_LAYOUT_SHADER_RESOURCE: return !(resource_flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE);
        default: return true;
      }
    }
    // The copy's own source layout. A common layout already admits COPY_SOURCE
    // access, so the copy changes only access (a simultaneous-access texture
    // never changes layout); any other leaves for COPY_SOURCE, in its
    // direct-queue form inside that family.
    constexpr std::uint32_t copy_source(std::uint32_t layout) {
      if (layout == D3D12_BARRIER_LAYOUT_COMMON || layout == D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COMMON) return layout;
      return direct_queue(layout) ? D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_SOURCE : D3D12_BARRIER_LAYOUT_COPY_SOURCE;
    }
  }

  enum class rule {
    none,                   // No usable state source.
    declared_must_match,    // The per-call declaration; an observed state must agree.
    observed_only,          // A nonzero state observed on this recording.
    observed_else_declared, // Observed wins; no entry permits a flag-compatible declaration.
    observed_else_contract, // Observed wins; no entry permits the SDK's input contract.
    observed_over_declared, // A declaration of a later use: an observed state is required and wins.
  };

  //                   | source_contract        | prefer_observed_recording
  //  unavailable      | observed_only          | observed_only
  //  declared         | declared_must_match    | observed_else_declared
  //  declared_for_use | observed_over_declared | observed_over_declared
  //  observed_nonzero | observed_only          | observed_only
  //  sdk_contract     | observed_else_contract | observed_else_contract
  //  clear_contract   | declared_must_match    | declared_must_match
  //  last_proven      | observed_only          | observed_only
  // An undeclared state (Streamline's UINT_MAX default) claims nothing, so
  // only a state observed on this recording can supply it, exactly as for a
  // source that requires observation. A clear's contract state is checked
  // against an observation like a declaration. A declaration made for a later
  // use describes another point than this copy (the contract matrix's S11), so
  // only this recording's observation proves the copy: a declaration that
  // agrees with it stands as declared_must_match's does, and one that
  // disagrees gives way to it under the observation rules. A last proven state
  // is an observation the capture owner supplies when the recording has none
  // (record_impl), so it follows the observation rules. An unknown proof
  // value selects none.
  constexpr rule select(sunshine_scene_depth::state_proof proof, texture_state_policy policy) {
    using proof_kind = sunshine_scene_depth::state_proof;
    switch (proof) {
      case proof_kind::declared:
        return policy == texture_state_policy::prefer_observed_recording ? rule::observed_else_declared : rule::declared_must_match;
      case proof_kind::declared_for_use: return rule::observed_over_declared;
      case proof_kind::unavailable:
      case proof_kind::observed_nonzero:
      case proof_kind::last_proven: return rule::observed_only;
      case proof_kind::sdk_contract: return rule::observed_else_contract;
      case proof_kind::clear_contract: return rule::declared_must_match;
      default: return rule::none;
    }
  }

  // State evidence for the source on the exact recording that will carry the copy.
  // pending: a split barrier began and has not ended (the state is in transition).
  // enhanced: the latest barrier was an enhanced Barrier into layout with
  // access (known only when both are admissible); value is not used then.
  // earlier: the recording names nothing, and an enhanced barrier on an
  // earlier executed list latched the source's layout (proven_state.h); the
  // other fields are then its proof from that list's queue, present only
  // while one stands.
  struct observation {
    bool present{}, known{}, blocked{};
    std::uint32_t value{};
    bool pending{}, enhanced{};
    std::uint32_t layout{}, access{};
    bool earlier{};
  };
  struct decision {
    bool admitted{};
    status result{status::unsupported_state};
    record_stage stage{record_stage::unsupported_proof};
    std::uint32_t state{}; // Transition and restore basis when admitted.
    bool used_observed{}, used_contract{};
    // The copy leaves and restores this enhanced layout and access instead.
    bool enhanced{};
    std::uint32_t layout{}, access{};
  };

  constexpr decision reject(status result, record_stage stage) { return {false, result, stage}; }

  constexpr bool usable_claim(std::uint32_t value) { return value != 0 && value != UINT32_MAX; }

  // Whether a legacy COMMON, observed or declared, is the source's state (the
  // contract matrix's D3 and D3b, owner ruling R1). A depth-stencil texture
  // cannot allow simultaneous access, so it is never implicitly promoted out
  // of COMMON (MS Learn, Using Resource Barriers, Common state promotion):
  // COMMON there is its state. Any other texture may have been promoted to a
  // read state or COPY_DEST by an earlier access on the same execution, so
  // COMMON names nothing about it there.
  constexpr bool common_is_state(std::uint32_t resource_flags) {
    return (resource_flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) &&
      !(resource_flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS);
  }

  // A declaration agrees with an observation: the same legacy state, or for
  // an enhanced layout the contract's equivalent of the declared state.
  constexpr bool agrees(std::uint32_t claimed, const observation &observed) {
    if (!observed.enhanced) return observed.value == claimed;
    const auto equivalent = layouts::equivalent(claimed);
    return equivalent != std::uint32_t(D3D12_BARRIER_LAYOUT_UNDEFINED) && equivalent == layouts::generic(observed.layout);
  }

  // A claim against a layout an enhanced barrier set on an earlier list (the
  // contract matrix's D6 and N8): a legacy barrier from the claim is illegal
  // there (Enhanced Barriers, legacy interop), and the claim describes the
  // layout only through the equivalence table. The matrix admits one such
  // basis: the shader-resource layout that a shader-read declaration and
  // NGX's contract name, as its proof from that list states it, on a texture
  // shaders may read. A queue-specific layout (DIRECT_QUEUE_SHADER_RESOURCE)
  // is another layout no claim names; any other layout, an unproven latched
  // texture and a source that needs an observation (a declaration of a later
  // use included) are refused.
  constexpr bool earlier_layout_copy(rule selected, std::uint32_t claimed, const observation &observed, std::uint32_t resource_flags) {
    return selected != rule::observed_only && selected != rule::observed_over_declared && usable_claim(claimed) &&
      layouts::equivalent(claimed) == std::uint32_t(D3D12_BARRIER_LAYOUT_SHADER_RESOURCE) && observed.present && observed.known &&
      !observed.blocked && !observed.pending && observed.enhanced &&
      observed.layout == std::uint32_t(D3D12_BARRIER_LAYOUT_SHADER_RESOURCE) &&
      observed.access != std::uint32_t(D3D12_BARRIER_ACCESS_NO_ACCESS) && layouts::allowed(observed.layout, resource_flags);
  }

  constexpr decision resolve(rule selected, std::uint32_t claimed, const observation &observed, std::uint32_t resource_flags) {
    if (selected == rule::none) return reject(status::unsupported_state, record_stage::unsupported_proof);
    if (observed.earlier) {
      if (!earlier_layout_copy(selected, claimed, observed, resource_flags))
        return reject(status::unsupported_state, record_stage::unsupported_state);
      decision out{true, status::recorded, record_stage::recorded, claimed};
      out.enhanced = true; out.layout = observed.layout; out.access = observed.access;
      return out;
    }
    // An in-progress split barrier or unrepresentable transition overrides every claim.
    if (observed.present && (observed.blocked || observed.pending || !observed.known))
      return reject(status::incomplete_state, record_stage::incomplete_state);
    decision out{true, status::recorded, record_stage::recorded, claimed};
    switch (selected) {
      case rule::declared_must_match:
        if (observed.present && !agrees(claimed, observed))
          return reject(status::conflicting_state, record_stage::conflicting_state);
        break;
      case rule::observed_over_declared:
        // The declaration describes a later use, so this recording must name
        // the state: where the two agree the declaration stands, otherwise
        // the observation does, by observed_only's rules.
        if (!observed.present) return reject(status::missing_state, record_stage::missing_state);
        if (agrees(claimed, observed)) break;
        out.state = observed.value;
        out.used_observed = true;
        break;
      case rule::observed_only:
      case rule::observed_else_declared:
      case rule::observed_else_contract:
        if (observed.present || selected == rule::observed_only) {
          // An observed legacy COMMON is evidence, not an absent entry: no
          // claim stands in for it, and it is the basis only where COMMON is
          // the state (below). An enhanced COMMON layout is an explicit layout.
          if (!observed.present) return reject(status::missing_state, record_stage::missing_state);
          out.state = observed.value;
          out.used_observed = true;
        } else if (!usable_claim(claimed)) {
          return reject(status::missing_state, record_stage::missing_state);
        } else if (selected != rule::observed_else_declared) {
          // The SDK contract describes a shader-readable input.
          if (resource_flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)
            return reject(status::missing_state, record_stage::missing_state);
          out.used_contract = true;
        }
        break;
      default: break;
    }
    // Every basis, observed, claimed or contracted, must be one the
    // allocation's flags allow (X1).
    if (observed.present && observed.enhanced) {
      if (!layouts::allowed(observed.layout, resource_flags))
        return reject(status::unsupported_state, record_stage::unsupported_state);
      out.enhanced = true; out.layout = observed.layout; out.access = observed.access;
      return out;
    }
    // A legacy COMMON, observed or claimed, is a basis only where it is the
    // source's state (common_is_state).
    if (out.state == D3D12_RESOURCE_STATE_COMMON && !common_is_state(resource_flags))
      return reject(status::missing_state, record_stage::missing_state);
    if (!supported(out.state) || !allowed(out.state, resource_flags))
      return reject(status::unsupported_state, record_stage::unsupported_state);
    return out;
  }
}
