// SPDX-License-Identifier: GPL-3.0-only
// The API contract matrix's pure layer (L0): production tables judged by the
// Game 3D API contract oracle (test_game3d_api_contract_oracle.h), never by a
// reference derived from the tables themselves. Cells where the add-on does
// not meet the contract are rows of test_game3d_api_contract_known_gaps.h; a
// listed cell that starts passing fails this test until its row is removed.
//
// - Streamline buffer types: each surveyed release's interposer version
//   resource, identified as the probe identifies it, and every raw tag 0-80
//   read through the selected buffer contract, against that release's own
//   header (test_sl_survey_tables.generated.h).
// - Copy state: every legal allocation, adapter claim, consumer, declared
//   state and recorded evidence the capture state table sees, plus
//   deliberately illegal rows that must refuse.
// - Depth formats: every DXGI format as allocation and as declared format,
//   for hardware and linear depth, through depth_view_format.h.
//
// The executable is built with the probe tests' sources and defines, so the
// hook layer (L1) can join it.
#include "capture_state_policy.h"
#include "depth_view_format.h"
#include "streamline_buffer_contract.h"
#include "streamline_camera_version.h"
#include "test_game3d_api_contract_known_gaps.h"

#include <cstdio>
#include <iterator>
#include <stdexcept>

namespace {
  namespace buffers = sunshine_streamline::buffers;
  namespace versioning = sunshine_streamline::versioning;
  namespace capture = sunshine_streamline::depth_capture;
  namespace copy_state = capture::copy_state;
  namespace matrix = game3d_contract;
  namespace survey = matrix::survey;
  namespace gaps = matrix::known_gaps;
  using proof = sunshine_scene_depth::state_proof;
  using policy = capture::texture_state_policy;

  void require(bool value, const char *message) {
    if (!value) {
      throw std::runtime_error(message);
    }
  }

  const char *name(matrix::outcome_kind kind) {
    constexpr const char *names[] {"capture", "refuse", "hold", "continuous", "offered", "ignored"};
    return names[unsigned(kind)];
  }

  // One family's verdicts against its known-gap rows. Up to eight failing
  // cells per family are described.
  template<unsigned N>
  struct tally {
    unsigned cells {}, met {}, unmet {}, closed {}, known[N] {};

    // The reason to describe a failing cell, or null.
    const char *add(gaps::verdict verdict, int index) {
      ++cells;
      switch (verdict) {
        case gaps::verdict::meets:
          ++met;
          return nullptr;
        case gaps::verdict::known_failure:
          ++known[index];
          return nullptr;
        case gaps::verdict::gap_closed:
          return closed++ < 8 ? "known gap now passes; remove its row" : nullptr;
        case gaps::verdict::misattributed:
          return unmet++ < 8 ? "known-gap row matched a cell another rule decides" : nullptr;
        case gaps::verdict::overlapping:
          return unmet++ < 8 ? "more than one known-gap row matches" : nullptr;
        default:
          return unmet++ < 8 ? "contract violated" : nullptr;
      }
    }

    template<class Case>
    void finish(const gaps::row<Case> (&rows)[N], const char *family) const {
      if (unmet || closed) {
        std::fprintf(stderr, "%s: %u cells violate the contract outside the known-gap table; %u recorded gaps now pass\n", family, unmet, closed);
        throw std::runtime_error("a production table diverged from the API contract matrix");
      }
      for (unsigned i = 0; i != N; ++i) {
        if (rows[i].status == gaps::kind::expected_failure && !known[i]) {
          std::fprintf(stderr, "%s: known-gap row %s matches no failing cell\n", family, rows[i].id);
          throw std::runtime_error("a stale known-gap row");
        }
      }
    }

    template<class Case>
    void print(const gaps::row<Case> (&rows)[N]) const {
      std::printf("%u meet the oracle, %u are recorded gaps (", met, cells - met);
      for (unsigned i = 0; i != N; ++i) {
        std::printf("%s%s %u", i ? ", " : "", rows[i].id, known[i]);
      }
      std::printf(")");
    }
  };

  // Streamline buffer types.

  // A release DLL's version resource as versioning::read_file reports it.
  versioning::information version_resource(const std::uint32_t (&fixed)[4], const char *file, const char *product) {
    versioning::information value;
    value.fixed_valid = true;
    value.file_ms = value.product_ms = MAKELONG(fixed[1], fixed[0]);
    value.file_ls = value.product_ls = MAKELONG(fixed[3], fixed[2]);
    for (std::size_t i = 0; file[i] && i + 1 < value.file_text.size(); ++i) {
      value.file_text[i] = wchar_t(file[i]);
    }
    for (std::size_t i = 0; product[i] && i + 1 < value.product_text.size(); ++i) {
      value.product_text[i] = wchar_t(product[i]);
    }
    value.strings_complete = file[0] && product[0];
    return value;
  }

  // The interposer as the probe identifies it (discover() in
  // streamline_camera_probe.cpp): its ABI family, and the buffer contract of
  // its exact header or of every candidate of an inexact 2.x family.
  struct identified {
    versioning::abi abi {};
    bool exact {};
    versioning::number number;
    versioning::candidates family;
    buffers::contract contract;
  };

  identified identify(const versioning::information &resource) {
    identified out;
    auto classified = resource;
    out.abi = versioning::classify(classified);
    auto exact = resource;
    out.exact = versioning::identity(exact, out.number);
    out.family = versioning::family(resource);
    out.contract = out.abi == versioning::abi::unsupported ? buffers::contract {} :
                   out.exact                               ? buffers::for_version(out.number) :
                                                             buffers::for_versions(out.family);
    return out;
  }

  matrix::sl_abi judged_abi(versioning::abi value) {
    switch (value) {
      case versioning::abi::legacy_no_state:
        return matrix::sl_abi::legacy_no_state;
      case versioning::abi::legacy:
        return matrix::sl_abi::legacy;
      case versioning::abi::v2:
        return matrix::sl_abi::typed;
      default:
        return matrix::sl_abi::none;
    }
  }

  // The meaning consumers give a canonical buffer type (streamline_buffer_types.h).
  matrix::buffer_meaning judged_meaning(std::uint32_t canonical) {
    switch (canonical) {
      case buffers::depth:
        return matrix::buffer_meaning::depth;
      case buffers::high_resolution_depth:
        return matrix::buffer_meaning::high_resolution_depth;
      case buffers::linear_depth:
        return matrix::buffer_meaning::linear_depth;
      case buffers::hudless_color:
        return matrix::buffer_meaning::hudless_color;
      case buffers::scaling_input_color:
        return matrix::buffer_meaning::scaling_input;
      case buffers::scaling_output_color:
        return matrix::buffer_meaning::scaling_output;
      case buffers::ui_color_and_alpha:
        return matrix::buffer_meaning::ui_color_and_alpha;
      case buffers::backbuffer:
        return matrix::buffer_meaning::backbuffer;
      case buffers::ui_alpha:
        return matrix::buffer_meaning::ui_alpha;
      case buffers::unknown:
        return matrix::buffer_meaning::undeclared;
      default:
        return matrix::buffer_meaning::other;
    }
  }

  // A raw tag meets its expectation when the contract gives it the role the
  // header does, and the same meaning where that role is a depth source or a
  // UI input. An unknown major must give no tag a meaning.
  bool meets(const matrix::outcome &expected, matrix::buffer_meaning wanted, matrix::buffer_meaning actual) {
    if (expected.kind == matrix::outcome_kind::refuse) {
      return actual == matrix::buffer_meaning::undeclared;
    }
    const auto role = matrix::buffer_role(actual).kind;
    return matrix::permits(expected, role) && (role == matrix::outcome_kind::ignored || actual == wanted);
  }

  struct buffer_counts {
    tally<std::size(gaps::buffer_rows)> tags;
    unsigned identities {}, canonical {};
  };

  void describe(const char *what, const matrix::buffer_case &value, const matrix::outcome &expected, matrix::buffer_meaning wanted, matrix::buffer_meaning actual) {
    std::fprintf(stderr, "%s: release %u.%u.%u raw %u -> expected %s%s%s rule=%s meaning=%u; contract meaning=%u\n", what, value.version.major, value.version.minor, value.version.patch, value.raw, name(expected.kind), matrix::either(expected) ? " or " : "", matrix::either(expected) ? name(expected.alternative) : "", expected.by ? expected.by->id : "?", unsigned(wanted), unsigned(actual));
  }

  // Every raw tag 0-80 at one identified release.
  void judge_tags(buffer_counts &counts, const matrix::sl_version &version, const identified &found) {
    for (std::uint32_t raw = 0; raw <= 80; ++raw) {
      const matrix::buffer_case value {version, raw};
      const auto expected = matrix::expect(value);
      require(expected.by != nullptr, "the oracle named no rule for a buffer type");
      const auto wanted = matrix::declared_meaning(value);
      const auto actual = judged_meaning(buffers::canonical(found.contract, raw));
      int index {};
      const auto verdict = gaps::judge(gaps::buffer_rows, value, expected, meets(expected, wanted, actual), &index);
      if (const auto *what = counts.tags.add(verdict, index)) {
        describe(what, value, expected, wanted, actual);
      }
    }
  }

  buffer_counts buffer_types() {
    buffer_counts counts;
    for (const auto &release : survey::releases) {
      const matrix::sl_version version {release.major, release.minor, release.patch};
      const auto found = identify(version_resource(release.dll_fixed, release.dll_file_version, release.dll_product_version));
      // S17b: the release DLL selects a contract covering its own header, and
      // the structure ABI that header defines.
      bool covered = found.exact && found.number.major == version.major && found.number.minor == version.minor && found.number.patch == version.patch;
      for (unsigned i = 0; !found.exact && i != found.family.count; ++i) {
        const auto &candidate = found.family.values[i];
        covered |= candidate.major == version.major && candidate.minor == version.minor && candidate.patch == version.patch;
      }
      if (!covered || !found.contract.known || judged_abi(found.abi) != matrix::declared_abi(release) || buffers::ui_alpha_authenticated(found.contract) != matrix::ui_alpha_automatic(version)) {
        std::fprintf(stderr, "release %s: covered=%u known=%u abi=%u (header %u) ui-alpha automatic=%u (header %u)\n", release.tag, covered, found.contract.known, unsigned(judged_abi(found.abi)), unsigned(matrix::declared_abi(release)), buffers::ui_alpha_authenticated(found.contract), matrix::ui_alpha_automatic(version));
        throw std::runtime_error("a surveyed release DLL selected the wrong buffer contract or ABI (S17b)");
      }
      ++counts.identities;
      judge_tags(counts, version, found);
    }
    // Outside the survey: an unknown major has no contract and gives no tag a
    // meaning (S24); a newer 2.x keeps UI alpha manual (U4).
    const std::uint32_t major3[] {3, 0, 0, 0}, newer[] {2, 15, 0, 0};
    const auto unknown = identify(version_resource(major3, "3.0.0.0", "3.0.0.0"));
    require(unknown.abi == versioning::abi::unsupported && !unknown.contract.known, "an unknown Streamline major selected a contract (S24)");
    ++counts.identities;
    judge_tags(counts, {3, 0, 0}, unknown);
    const auto later = identify(version_resource(newer, "2.15.0.0", "2.15.0.0"));
    require(later.contract.known && buffers::ui_alpha_authenticated(later.contract) == matrix::ui_alpha_automatic({2, 15, 0}), "a release newer than the survey used UI alpha automatically (U4)");
    ++counts.identities;
    // Consumers compare canonical numbers: the newest surveyed header's.
    for (const auto canonical : {buffers::depth, buffers::high_resolution_depth, buffers::linear_depth, buffers::hudless_color, buffers::scaling_input_color, buffers::scaling_output_color, buffers::ui_color_and_alpha, buffers::backbuffer, buffers::responsivity_mask, buffers::ui_alpha}) {
      require(matrix::declared_meaning({matrix::newest_surveyed, canonical}) == judged_meaning(canonical), "a canonical buffer type is not the newest surveyed header's");
      ++counts.canonical;
    }
    counts.tags.finish(gaps::buffer_rows, "buffer types");
    return counts;
  }

  // Copy state.

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

  struct state_counts {
    tally<std::size(gaps::state_rows)> table;
    unsigned legal {}, illegal {};
  };

  state_counts copy_states() {
    state_counts counts;
    const auto judge = [&](const matrix::state_case &value, bool legal_row) {
      require(matrix::legal(value) == legal_row, legal_row ? "a generated cell is illegal" : "an illegal row passed the legality filter");
      ++(legal_row ? counts.legal : counts.illegal);
      const auto expected = matrix::expect(value);
      require(expected.by != nullptr, "the oracle named no rule");
      // Illegal input must end in a safe refusal.
      require(legal_row || expected.kind == matrix::outcome_kind::refuse, "the oracle admitted an illegal row");
      const auto actual = production(value);
      int index {};
      const auto verdict = gaps::judge(gaps::state_rows, value, expected, matrix::satisfies(expected, judged(actual)), &index);
      if (const auto *what = counts.table.add(verdict, index)) {
        describe(what, value, expected, actual);
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
    counts.table.finish(gaps::state_rows, "copy state");
    return counts;
  }

  // Depth formats.

  struct format_counts {
    unsigned cells {}, captured {}, either {};
  };

  // Every DXGI format value through the newest this build's dxgiformat.h
  // declares, as allocation and as declared format, for hardware and linear
  // depth.
  format_counts depth_formats() {
    format_counts counts;
    unsigned unmet {};
    for (std::uint32_t allocation = 0; allocation <= DXGI_FORMAT_V408; ++allocation) {
      for (std::uint32_t declared = 0; declared <= DXGI_FORMAT_V408; ++declared) {
        for (const bool device_depth : {true, false}) {
          const matrix::format_case value {allocation, declared, device_depth};
          const auto expected = matrix::expect(value);
          const auto storage = sunshine_depth_format::storage(DXGI_FORMAT(allocation));
          const auto view = sunshine_depth_format::view(DXGI_FORMAT(allocation), DXGI_FORMAT(declared), device_depth);
          const bool captured = view != DXGI_FORMAT_UNKNOWN;
          const auto kind = captured ? matrix::outcome_kind::capture : matrix::outcome_kind::refuse;
          const bool satisfied = (expected.kind == kind || expected.alternative == kind) &&
                                 (!captured || (storage == expected.storage && view == expected.view && sunshine_depth_format::readable(storage, view)));
          ++counts.cells;
          counts.captured += captured;
          counts.either += expected.kind != expected.alternative;
          if (!satisfied && unmet++ < 8) {
            std::fprintf(stderr, "contract violated: allocation=%u declared=%u device_depth=%u -> expected %s%s%s rule=%s storage=%u view=%u; table storage=%u view=%u\n", allocation, declared, device_depth, name(expected.kind), expected.kind != expected.alternative ? " or " : "", expected.kind != expected.alternative ? name(expected.alternative) : "", expected.by ? expected.by->id : "?", expected.storage, expected.view, unsigned(storage), unsigned(view));
          }
        }
      }
    }
    if (unmet) {
      std::fprintf(stderr, "depth formats: %u cells violate the contract\n", unmet);
      throw std::runtime_error("depth_view_format.h diverged from the API contract matrix");
    }
    return counts;
  }
}  // namespace

int main() try {
  const auto streamline = buffer_types();
  const auto state = copy_states();
  const auto formats = depth_formats();
  std::printf("PASS API contract matrix L0: %u buffer-type cells over %u release identities; ", streamline.tags.cells, streamline.identities);
  streamline.tags.print(gaps::buffer_rows);
  std::printf("; %u canonical numbers. %u copy-state cells (%u legal, %u illegal); ", streamline.canonical, state.table.cells, state.legal, state.illegal);
  state.table.print(gaps::state_rows);
  std::printf(". %u depth-format cells (%u captured, %u EITHER) meet the oracle\n", formats.cells, formats.captured, formats.either);
  return 0;
} catch (const std::exception &error) {
  std::fprintf(stderr, "FAIL API contract matrix L0: %s\n", error.what());
  return 1;
}
