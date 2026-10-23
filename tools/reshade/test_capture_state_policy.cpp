// SPDX-License-Identifier: GPL-3.0-only
// CPU regression for the capture owner's pre-copy state table, judged by the
// Game 3D API contract oracle (test_game3d_api_contract_oracle.h) rather than
// by a reference derived from the table itself. The matrix crosses every legal
// allocation, adapter claim, consumer, declared state and recorded evidence
// the table can see, plus deliberately illegal rows that must refuse. Cells
// where the table does not meet the contract are rows of
// test_game3d_api_contract_known_gaps.h; a listed cell that starts passing
// fails this test until its row is removed.
#include "capture_state_policy.h"
#include "test_game3d_api_contract_known_gaps.h"

#include <cstdio>
#include <iterator>
#include <stdexcept>

namespace {
  namespace capture = sunshine_streamline::depth_capture;
  namespace copy_state = capture::copy_state;
  namespace matrix = game3d_contract;
  namespace gaps = matrix::known_gaps;
  using proof = sunshine_scene_depth::state_proof;
  using policy = capture::texture_state_policy;

  void require(bool value, const char *message) {
    if (!value) {
      throw std::runtime_error(message);
    }
  }

  // The copy-state table's view of one case: the proof and policy the adapters
  // choose (streamline_camera_probe.cpp, ngx_depth_source.cpp and
  // auxiliary_state_policy for an at-call UI copy) and the recording's entry
  // for the source. The entry mirrors the barrier hooks
  // (streamline_depth_capture.cpp barriers() and enhanced_textures()), which
  // classify a recorded state with these same copy_state predicates; the
  // device matrix replaces this mirror with real barriers.
  copy_state::decision production(const matrix::state_case &value) {
    auto supplied = proof::observed_nonzero;
    std::uint32_t claimed {};
    switch (value.source) {
      case matrix::claim::sl_declared:
        supplied = proof::declared;
        claimed = value.declared;
        break;
      case matrix::claim::sl_undeclared:
        supplied = proof::unavailable;
        claimed = UINT32_MAX;
        break;
      case matrix::claim::ngx_contract:
        supplied = proof::sdk_contract;
        claimed = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        break;
      default:
        break;
    }
    auto chosen = policy::source_contract;
    if (value.role == matrix::consumer::ui) {
      capture::input input;
      input.provider = sunshine_scene_depth::provider_kind::streamline;
      input.valid_until = sunshine_scene_depth::lifetime::at_call;
      chosen = capture::auxiliary_state_policy(input);
    }
    const auto &source = value.resource;
    copy_state::observation observed;
    switch (source.seen) {
      case matrix::evidence::transition:
      case matrix::evidence::end_without_begin:
        observed = {true, copy_state::supported(source.observed), false, source.observed};
        break;
      case matrix::evidence::split_open:
        observed = {true, false, false, source.observed, true};
        break;
      case matrix::evidence::begin_and_end:
      case matrix::evidence::mixed_split:
        observed = {true, false, true, source.observed};
        break;
      case matrix::evidence::enhanced:
      case matrix::evidence::enhanced_split_open:
        {
          const bool pending = source.seen == matrix::evidence::enhanced_split_open;
          observed = {true, !pending && copy_state::layouts::admissible(source.layout) && source.access != D3D12_BARRIER_ACCESS_NO_ACCESS, false, UINT32_MAX, pending, true, source.layout, source.access};
          break;
        }
      default:
        break;
    }
    return copy_state::resolve(copy_state::select(supplied, chosen), claimed, observed, source.flags);
  }

  matrix::result judged(const copy_state::decision &decision) {
    matrix::result out;
    if (decision.admitted) {
      out.kind = matrix::outcome_kind::capture;
      out.basis = {decision.enhanced, decision.enhanced ? 0 : decision.state, decision.layout, decision.access};
      out.basis_known = true;
      return out;
    }
    switch (decision.result) {
      case capture::status::missing_state:
      case capture::status::incomplete_state:
      case capture::status::unsupported_state:
      case capture::status::conflicting_state:
        out.why = matrix::refusal::state_unprovable;
        break;
      case capture::status::malformed:
        out.why = matrix::refusal::malformed;
        break;
      default:
        break;
    }
    return out;
  }

  // The legacy states a texture can be in (the dimension's legal values):
  // COMMON, each write state alone, every read-only combination and the
  // GENERIC_READ forms.
  template<class Visit>
  void each_texture_state(Visit &&visit) {
    visit(std::uint32_t(D3D12_RESOURCE_STATE_COMMON));
    for (const auto write : matrix::texture_write_states) {
      visit(write);
    }
    for (std::uint32_t bits = 1; bits != 0x20; ++bits) {
      std::uint32_t state {};
      constexpr std::uint32_t read[] {D3D12_RESOURCE_STATE_DEPTH_READ, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RESOLVE_SOURCE};
      for (unsigned bit = 0; bit != std::size(read); ++bit) {
        if (bits & (1u << bit)) {
          state |= read[bit];
        }
      }
      visit(state);
    }
    for (const std::uint32_t extra : {0u, std::uint32_t(D3D12_RESOURCE_STATE_DEPTH_READ), std::uint32_t(D3D12_RESOURCE_STATE_RESOLVE_SOURCE), std::uint32_t(D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_RESOLVE_SOURCE)}) {
      visit(D3D12_RESOURCE_STATE_GENERIC_READ | extra);
    }
  }

  // Every legal cell of the state dimensions the table sees: allocation
  // flags, consumer, claim, declared state and the recording's evidence (none,
  // a completed or pending legacy transition, a completed or pending enhanced
  // barrier into each layout with a compatible access or NO_ACCESS).
  template<class Visit>
  void each_legal_case(Visit &&visit) {
    for (std::uint32_t flags = 0; flags <= matrix::allocation_flag_bits; ++flags) {
      if ((flags & ~matrix::allocation_flag_bits) || !matrix::legal_flags(flags)) {
        continue;
      }
      std::uint32_t allowed[64] {};
      unsigned allowed_count {};
      each_texture_state([&](std::uint32_t state) {
        if (matrix::state_allowed(state, flags)) {
          allowed[allowed_count++] = state;
        }
      });
      std::uint32_t layouts[0x20 * 2] {}, accesses[0x20 * 2] {};
      unsigned layout_count {};
      for (std::uint32_t layout = 0; layout <= 0x1f; ++layout) {
        const auto value = layout == 0x1f ? std::uint32_t(D3D12_BARRIER_LAYOUT_UNDEFINED) : layout;
        for (const auto access : {matrix::compatible_access(value), std::uint32_t(D3D12_BARRIER_ACCESS_NO_ACCESS)}) {
          matrix::resource_state probe {flags, matrix::evidence::enhanced, 0, value, access};
          if (!matrix::legal_evidence(probe, matrix::list_type::direct)) {
            continue;
          }
          bool repeated {};
          for (unsigned i = 0; i != layout_count; ++i) {
            repeated |= layouts[i] == value && accesses[i] == access;
          }
          if (!repeated) {
            layouts[layout_count] = value;
            accesses[layout_count++] = access;
          }
        }
      }
      for (const auto role : {matrix::consumer::depth, matrix::consumer::ui}) {
        if (role == matrix::consumer::ui && matrix::depth_stencil(flags)) {
          continue;
        }
        const auto with_evidence = [&](matrix::state_case value) {
          value.resource = {flags, matrix::evidence::none};
          visit(value);
          for (const auto seen : {matrix::evidence::transition, matrix::evidence::split_open}) {
            for (unsigned i = 0; i != allowed_count; ++i) {
              value.resource = {flags, seen, allowed[i]};
              visit(value);
            }
          }
          for (const auto seen : {matrix::evidence::enhanced, matrix::evidence::enhanced_split_open}) {
            for (unsigned i = 0; i != layout_count; ++i) {
              value.resource = {flags, seen, 0, layouts[i], accesses[i]};
              visit(value);
            }
          }
        };
        matrix::state_case value;
        value.role = role;
        for (const auto source : {matrix::claim::none, matrix::claim::sl_undeclared, matrix::claim::ngx_contract, matrix::claim::sl_declared}) {
          if (source == matrix::claim::ngx_contract && role == matrix::consumer::ui) {
            continue;
          }
          value.source = source;
          if (source != matrix::claim::sl_declared) {
            value.declared = 0;
            with_evidence(value);
            continue;
          }
          for (unsigned i = 0; i != allowed_count; ++i) {
            value.declared = allowed[i];
            with_evidence(value);
          }
        }
      }
    }
  }

  // Deliberately illegal input the table can see (the rows the legality
  // filter drops): each must end in a safe refusal.
  std::size_t illegal_cases(matrix::state_case *out) {
    using matrix::claim;
    using matrix::consumer;
    using matrix::evidence;
    constexpr std::uint32_t ds = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, none = D3D12_RESOURCE_FLAG_NONE;
    const auto row = [](std::uint32_t flags, claim source, std::uint32_t declared, evidence seen, std::uint32_t observed = 0, std::uint32_t layout = 0, std::uint32_t access = 0, consumer role = consumer::depth) {
      matrix::state_case value;
      value.resource = {flags, seen, observed, layout, access};
      value.role = role;
      value.source = source;
      value.declared = declared;
      return value;
    };
    const matrix::state_case rows[] {
      // States and layouts the allocation's flags do not allow (X1).
      row(none, claim::sl_declared, D3D12_RESOURCE_STATE_DEPTH_WRITE, evidence::none),
      row(ds, claim::sl_declared, D3D12_RESOURCE_STATE_RENDER_TARGET, evidence::none),
      row(ds | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE, claim::sl_declared, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, evidence::none),
      row(ds | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE, claim::sl_declared, D3D12_RESOURCE_STATE_GENERIC_READ, evidence::none),
      row(none, claim::sl_declared, D3D12_RESOURCE_STATE_DEPTH_WRITE, evidence::transition, D3D12_RESOURCE_STATE_DEPTH_WRITE),
      row(none, claim::none, 0, evidence::transition, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
      row(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, claim::sl_undeclared, UINT32_MAX, evidence::transition, D3D12_RESOURCE_STATE_DEPTH_READ),
      row(ds, claim::none, 0, evidence::enhanced, 0, D3D12_BARRIER_LAYOUT_RENDER_TARGET, D3D12_BARRIER_ACCESS_RENDER_TARGET),
      row(ds | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE, claim::ngx_contract, 0, evidence::enhanced, 0, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, D3D12_BARRIER_ACCESS_SHADER_RESOURCE),
      row(none, claim::sl_declared, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, evidence::transition, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, 0, 0, consumer::ui),
      row(none, claim::sl_declared, D3D12_RESOURCE_STATE_RENDER_TARGET, evidence::none, 0, 0, 0, consumer::ui),
      // Not a texture state (X2).
      row(ds, claim::sl_declared, D3D12_RESOURCE_STATE_RENDER_TARGET | D3D12_RESOURCE_STATE_DEPTH_WRITE, evidence::none),
      row(none, claim::sl_declared, D3D12_RESOURCE_STATE_COPY_DEST | D3D12_RESOURCE_STATE_GENERIC_READ, evidence::none),
      row(ds, claim::none, 0, evidence::transition, D3D12_RESOURCE_STATE_DEPTH_WRITE | D3D12_RESOURCE_STATE_DEPTH_READ),
      row(ds, claim::sl_declared, 0x4000, evidence::none),
      row(none, claim::none, 0, evidence::transition, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE),
      row(none, claim::none, 0, evidence::enhanced, 0, D3D12_BARRIER_LAYOUT_SHADING_RATE_SOURCE, D3D12_BARRIER_ACCESS_SHADING_RATE_SOURCE),
      // Barriers outside the contract (D4, D5b).
      row(ds, claim::sl_declared, D3D12_RESOURCE_STATE_DEPTH_READ, evidence::begin_and_end, D3D12_RESOURCE_STATE_DEPTH_READ),
      row(ds, claim::ngx_contract, 0, evidence::mixed_split, D3D12_RESOURCE_STATE_DEPTH_READ),
      row(ds, claim::none, 0, evidence::enhanced, 0, D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_SHADER_RESOURCE, D3D12_BARRIER_ACCESS_SHADER_RESOURCE),
      row(none, claim::sl_undeclared, UINT32_MAX, evidence::enhanced, 0, D3D12_BARRIER_LAYOUT_VIDEO_DECODE_READ, D3D12_BARRIER_ACCESS_VIDEO_DECODE_READ),
    };
    for (const auto &value : rows) {
      *out++ = value;
    }
    return std::size(rows);
  }

  const char *name(matrix::outcome_kind kind) {
    constexpr const char *names[] {"capture", "refuse", "hold", "continuous", "offered", "ignored"};
    return names[unsigned(kind)];
  }

  void describe(const char *what, const matrix::state_case &value, const matrix::outcome &expected, const copy_state::decision &actual) {
    const auto &source = value.resource;
    std::fprintf(stderr,
                 "%s: flags=0x%x role=%u claim=%u declared=0x%x evidence=%u observed=0x%x layout=0x%x access=0x%x -> "
                 "expected %s%s%s rule=%s refusal=%u basis=%u/0x%x/0x%x/0x%x; table admitted=%u status=%u stage=%u state=0x%x enhanced=%u/0x%x/0x%x\n",
                 what,
                 source.flags,
                 unsigned(value.role),
                 unsigned(value.source),
                 value.declared,
                 unsigned(source.seen),
                 source.observed,
                 source.layout,
                 source.access,
                 name(expected.kind),
                 matrix::either(expected) ? " or " : "",
                 matrix::either(expected) ? name(expected.alternative) : "",
                 expected.by ? expected.by->id : "?",
                 unsigned(expected.why),
                 expected.basis.enhanced,
                 expected.basis.state,
                 expected.basis.layout,
                 expected.basis.access,
                 actual.admitted,
                 unsigned(actual.result),
                 unsigned(actual.stage),
                 actual.state,
                 actual.enhanced,
                 actual.layout,
                 actual.access);
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

  // The contract matrix. Every cell's expectation comes from expect(); the
  // known-gap table says which cells are recorded as failing.
  constexpr auto row_count = std::size(gaps::state_rows);
  unsigned legal {}, illegal {}, met {}, unmet {}, closed {}, known[row_count] {};
  const auto judge = [&](const matrix::state_case &value, bool legal_row) {
    require(matrix::legal(value) == legal_row, legal_row ? "a generated cell is illegal" : "an illegal row passed the legality filter");
    ++(legal_row ? legal : illegal);
    const auto expected = matrix::expect(value);
    require(expected.by != nullptr, "the oracle named no rule");
    // Illegal input must end in a safe refusal.
    require(legal_row || expected.kind == matrix::outcome_kind::refuse, "the oracle admitted an illegal row");
    const auto actual = production(value);
    int index {};
    switch (gaps::judge(gaps::state_rows, value, expected, matrix::satisfies(expected, judged(actual)), &index)) {
      case gaps::verdict::meets:
        ++met;
        break;
      case gaps::verdict::known_failure:
        ++known[index];
        break;
      case gaps::verdict::gap_closed:
        if (closed++ < 8) {
          describe("known gap now passes; remove its row", value, expected, actual);
        }
        break;
      case gaps::verdict::misattributed:
        if (unmet++ < 8) {
          describe("known-gap row matched a cell another rule decides", value, expected, actual);
        }
        break;
      case gaps::verdict::overlapping:
        if (unmet++ < 8) {
          describe("more than one known-gap row matches", value, expected, actual);
        }
        break;
      default:
        if (unmet++ < 8) {
          describe("contract violated", value, expected, actual);
        }
        break;
    }
  };
  each_legal_case([&](const matrix::state_case &value) {
    judge(value, true);
  });
  matrix::state_case rows[32];
  const auto count = illegal_cases(rows);
  for (std::size_t i = 0; i != count; ++i) {
    judge(rows[i], false);
  }
  if (unmet || closed) {
    std::fprintf(stderr, "%u cells violate the contract outside the known-gap table; %u recorded gaps now pass\n", unmet, closed);
    throw std::runtime_error("the capture state table diverged from the API contract matrix");
  }
  for (std::size_t i = 0; i != row_count; ++i) {
    if (gaps::state_rows[i].status == gaps::kind::expected_failure && !known[i]) {
      std::fprintf(stderr, "known-gap row %s matches no failing cell\n", gaps::state_rows[i].id);
      throw std::runtime_error("a stale known-gap row");
    }
  }

  // Named behaviors, independent of the matrix: pinned examples and how the
  // copy builds its layouts (copy_source, equivalent), which the contract
  // leaves to the add-on.
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
  std::printf("PASS capture state table: %u contract cells (%u legal, %u illegal); %u meet the API contract oracle, %u are recorded gaps (", legal + illegal, legal, illegal, met, legal + illegal - met);
  for (std::size_t i = 0; i != row_count; ++i) {
    std::printf("%s%s %u", i ? ", " : "", gaps::state_rows[i].id, known[i]);
  }
  std::printf("); proof/policy table, copy layouts and named rules\n");
  return 0;
} catch (const std::exception &error) {
  std::fprintf(stderr, "FAIL capture state table: %s\n", error.what());
  return 1;
}
