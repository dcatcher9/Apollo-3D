// SPDX-License-Identifier: GPL-3.0-only
// The API contract matrix's CPU layers: the add-on judged by the Game 3D API
// contract oracle (test_game3d_api_contract_oracle.h), never by a reference
// derived from the add-on itself. Cells where the add-on does not meet the
// contract are rows of test_game3d_api_contract_known_gaps.h; a listed cell
// that starts passing fails this test until its row is removed.
//
// The pure layer (L0) reads production tables exhaustively:
// - Streamline buffer types: each surveyed release's interposer version
//   resource, identified as the probe identifies it (identify() mirrors
//   discover()'s selection), and every raw tag 0-80 read through the selected
//   buffer contract, against that release's own header
//   (test_sl_survey_tables.generated.h).
// - Copy state: every legal allocation, adapter claim, consumer, declared
//   state and recorded evidence the capture state table sees, plus
//   deliberately illegal rows that must refuse.
// - Depth formats: every DXGI format as allocation and as declared format,
//   for hardware and linear depth, through depth_view_format.h.
// - Refusal classes: every reason the adapters and the capture owner name
//   classifies (depth_refusal.h), and a reason produced by an input the
//   oracle has a case for carries that case's REFUSE class (or HOLD).
//
// The hook layer (L1) drives the real adapters without a device:
// - Streamline: the probe's MinHook hooks on fixture entry points
//   (streamline_camera_probe.h testing::install, as the probe tests do),
//   over the full product of tag entry point, lifecycle, command list, copy
//   point and generated Presents; each other value alone; a pairwise
//   covering array over every dimension; and named seed rows
//   (test_contract_matrix_generator.h). The hooks stay installed across
//   consecutive cases of one release; each case starts a fresh observation
//   epoch (initialize). A changing extent runs two frames of different
//   render sizes and requires the same logical source for both (R3).
// - NGX: the evaluation adapter (ngx_depth_source.h) with this executable's
//   parameter getters and testing::set_callbacks, over every combination of
//   its dimensions, DLSS-G's own evaluation (feature 11, DLSSG.*) included.
// A cell's result is what the adapter nominates at the copy point, admitted
// by the capture layer's rules: copy_state::select and resolve, the
// production predicates, and two record_impl checks repeated here (a copy
// needs a command list; a copy outside its tag call needs no device Present
// since the tag, source_lifetime_current). Without a device the harness
// counts the Presents a case puts between tag and copy, and admits DLSS-G's
// inputs as observe_frame_generation_inputs nominates them (a state observed
// on the evaluation's own list); the device and runtime layers replace these
// repeats. A known gap these repeats decide (S7) closes only when its fix
// changes the repeat too. L1 judges the kind, deferral and copy basis of each
// result, and that a recorded gap still fails the way its row records;
// refusal classes wait for the runtime layer, which reads them from the test
// add-on's SunshineDepthTestLastRefusal.
//
// Command line (test_contract_matrix_generator.h): --list, --case
// <id|family:name=value,...>, --shard i/N. Without one, both layers run;
// shard 0 also runs the pure layer. Only an unsharded run finds a stale
// known-gap row of a hook family.
#include "capture_state_policy.h"
#include "depth_refusal.h"
#include "depth_view_format.h"
#include "ngx_depth_source.h"
#include "streamline_buffer_contract.h"
#include "streamline_camera_data.h"
#include "streamline_camera_probe.h"
#include "streamline_camera_version.h"
#include "test_contract_matrix_generator.h"
#include "test_game3d_api_contract_tally.h"

#include <cstdio>
#include <cstring>
#include <iterator>
#include <reshade.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__GNUC__) && !defined(__clang__)
  #define FIXTURE_NOINLINE __attribute__((noinline, noipa))
#else
  #define FIXTURE_NOINLINE __declspec(noinline)
#endif

// The probe finds ReShade's log and configuration through these exports
// (reshade.hpp). Lines are written synchronously (SUNSHINE_SYNCHRONOUS_LOG)
// and dropped: the matrix judges published observations, not log text.
extern "C" __declspec(dllexport) bool ReShadeRegisterAddon(void *, std::uint32_t) {
  return false;
}

extern "C" __declspec(dllexport) void ReShadeUnregisterAddon(void *) {}

extern "C" __declspec(dllexport) void ReShadeLogMessage(void *, int, const char *) {}

// Streamline and NGX depth sources as a game sees them by default; the
// opt-in probe diagnostics and call trace stay off.
extern "C" __declspec(dllexport) bool ReShadeGetConfigValue(void *, reshade::api::effect_runtime *, const char *section, const char *key, char *value, size_t *size) {
  if (std::strcmp(section, "SUNSHINE_DEPTH") != 0 || !size) {
    return false;
  }
  const char *setting = std::strcmp(key, "StreamlineCameraProbe") == 0 ? "0" :
                        std::strcmp(key, "StreamlineDepthSource") == 0 ? "1" :
                        std::strcmp(key, "NGXDepthSource") == 0        ? "0" :
                                                                         nullptr;
  if (!setting) {
    return false;
  }
  const auto length = std::strlen(setting);
  if (!value || *size <= length) {
    *size = length + 1;
    return value == nullptr;
  }
  std::memcpy(value, setting, length + 1);
  *size = length;
  return true;
}

// One NGX parameter map per case, answered through the named C getters an
// NGX SDK owner exports (ngx_depth_source.h resolve_parameter_api).
#include "test_ngx_stand_in.h"

namespace {
  namespace buffers = sunshine_streamline::buffers;
  namespace versioning = sunshine_streamline::versioning;
  namespace capture = sunshine_streamline::depth_capture;
  namespace copy_state = capture::copy_state;
  namespace matrix = game3d_contract;
  namespace survey = matrix::survey;
  namespace gaps = matrix::known_gaps;
  namespace generator = matrix::generator;
  using proof = sunshine_scene_depth::state_proof;
  using policy = capture::texture_state_policy;
  using namespace matrix::judging;

  void require(bool value, const char *message) {
    if (!value) {
      throw std::runtime_error(message);
    }
  }

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

  // The interposer as the probe identifies it: its ABI family, and the buffer
  // contract of its exact header or of every candidate of an inexact 2.x
  // family. A mirror of discover()'s selection (streamline_camera_probe.cpp),
  // whose testing::install repeats only the exact branch: a change to that
  // selection is repeated here until one shared helper replaces all three.
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

  // The recording's entry for the source. It mirrors the barrier hooks
  // (streamline_depth_capture.cpp barriers() and enhanced_textures()), which
  // classify a recorded state with these same copy_state predicates; the
  // device matrix replaces this mirror with real barriers.
  copy_state::observation observation_of(const matrix::resource_state &source) {
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
    return observed;
  }

  // The copy-state table's view of one case: the proof and policy the adapters
  // choose (streamline_camera_probe.cpp, ngx_depth_source.cpp and
  // auxiliary_state_policy for an at-call UI copy) and the recording's entry
  // for the source.
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
    return copy_state::resolve(copy_state::select(supplied, chosen), claimed, observation_of(value.resource), value.resource.flags);
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

  using matrix::each_texture_state;

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
  std::vector<matrix::state_case> illegal_cases() {
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
      // Buffer read bits on a texture other than GENERIC_READ itself, which
      // the D3D12 runtime refuses (X2).
      row(ds, claim::sl_declared, D3D12_RESOURCE_STATE_GENERIC_READ | D3D12_RESOURCE_STATE_DEPTH_READ, evidence::none),
      row(none, claim::sl_declared, D3D12_RESOURCE_STATE_GENERIC_READ | D3D12_RESOURCE_STATE_RESOLVE_SOURCE, evidence::none),
      row(none, claim::sl_declared, D3D12_RESOURCE_STATE_INDEX_BUFFER, evidence::none),
      row(none, claim::none, 0, evidence::transition, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
      row(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, claim::sl_declared, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, evidence::none, 0, 0, 0, consumer::ui),
      // Barriers outside the contract (D4, D5b).
      row(ds, claim::sl_declared, D3D12_RESOURCE_STATE_DEPTH_READ, evidence::begin_and_end, D3D12_RESOURCE_STATE_DEPTH_READ),
      row(ds, claim::ngx_contract, 0, evidence::mixed_split, D3D12_RESOURCE_STATE_DEPTH_READ),
      row(ds, claim::none, 0, evidence::enhanced, 0, D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_SHADER_RESOURCE, D3D12_BARRIER_ACCESS_SHADER_RESOURCE),
      row(none, claim::sl_undeclared, UINT32_MAX, evidence::enhanced, 0, D3D12_BARRIER_LAYOUT_VIDEO_DECODE_READ, D3D12_BARRIER_ACCESS_VIDEO_DECODE_READ),
    };
    return {std::begin(rows), std::end(rows)};
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
    for (const auto &value : illegal_cases()) {
      judge(value, false);
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

  // Refusal classes.

  namespace refusal = sunshine_depth_refusal;

  constexpr bool oracle_classes() {
    using contract = matrix::refusal;
    using production = refusal::reason_class;
    constexpr std::pair<contract, production> same[] {{contract::none, production::none}, {contract::state_unprovable, production::state_unprovable}, {contract::no_copy_point, production::no_copy_point}, {contract::lifetime, production::lifetime}, {contract::not_depth, production::not_depth}, {contract::malformed, production::malformed}, {contract::unknown_abi, production::unknown_abi}, {contract::ambiguous_view, production::ambiguous_view}, {contract::fg_scope, production::fg_scope}};
    for (const auto &[oracle, reported] : same) {
      if (unsigned(oracle) != unsigned(reported)) {
        return false;
      }
    }
    return true;
  }

  static_assert(oracle_classes(), "depth_refusal.h's first classes are not the oracle's REFUSE classes in its order");

  // A reported class meets the oracle's outcome for the input that produces
  // it: its REFUSE class (a branch of an EITHER included), or its HOLD.
  bool agrees(refusal::reason_class reported, const matrix::outcome &expected) {
    if (reported == refusal::reason_class::hold) {
      return matrix::permits(expected, matrix::outcome_kind::hold);
    }
    return unsigned(reported) <= unsigned(refusal::reason_class::fg_scope) && matrix::permits(expected, matrix::outcome_kind::refuse) && unsigned(expected.why) == unsigned(reported);
  }

  struct refusal_counts {
    unsigned named {}, stages {}, outcomes {}, paired {};
  };

  // Every reason the adapters name classifies. A reason that an input the
  // oracle has a case for produces is checked against that case; the others
  // are outside the contract (internal, absent) or illegal input the
  // dimensions do not hold (an unreadable tag).
  refusal_counts refusal_classes() {
    using namespace matrix::pinned;
    using capture::record_stage;
    using matrix::claim;
    using matrix::copy_point;
    using matrix::evidence;
    using matrix::extent_form;
    using matrix::fg_mode;
    using matrix::lifecycle;
    using matrix::ngx_case;
    using matrix::ngx_depth_set;
    using matrix::ngx_feature;
    using matrix::ngx_hw_depth;
    using matrix::ngx_subrect;
    using matrix::precision_form;
    using matrix::render_pass;
    using matrix::resource_form;
    using matrix::resource_shape;
    using matrix::sl_case;
    using matrix::sl_result;
    using matrix::struct_form;
    using matrix::subresource_barrier;
    using matrix::tag_commands;
    using matrix::viewports;
    using depth_outcome = sunshine_streamline::frame_generation::depth_outcome;
    refusal_counts counts;
    for (const auto &entry : refusal::named_reasons) {
      require(entry.reason != refusal::reason_class::none && entry.reason != refusal::reason_class::unclassified, "a named refusal reason has no class");
      unsigned repeats {};
      for (const auto &other : refusal::named_reasons) {
        repeats += !std::strcmp(entry.name, other.name);
      }
      require(repeats == 1 && refusal::classify(entry.name) == entry.reason, "a named refusal reason is listed twice");
      ++counts.named;
    }
    require(refusal::classify("") == refusal::reason_class::unclassified && refusal::classify(nullptr) == refusal::reason_class::unclassified, "an unknown reason was classified");
    // The source gate returns these evidence statuses' names; the rest
    // admit the call.
    using sunshine_streamline::evidence_status;
    for (auto status = unsigned(evidence_status::inactive); status <= unsigned(evidence_status::tracked_content_evaluation); ++status) {
      const auto value = evidence_status(status);
      const bool admitted = value == evidence_status::source_associated_evaluation || value == evidence_status::missing_constants || value == evidence_status::rejected_constants || value == evidence_status::invalid_camera || value == evidence_status::camera_reset || value == evidence_status::untracked_frame;
      require(admitted || refusal::classify(sunshine_streamline::name(value)) != refusal::reason_class::unclassified, "an evidence status the source gate names has no refusal class");
    }
    // The tag gate returns these match statuses' names; same_epoch_only admits.
    using sunshine_streamline::match_status;
    for (auto status = unsigned(match_status::exact_frame); status <= unsigned(match_status::unsupported_lifecycle); ++status) {
      const auto value = match_status(status);
      require(value == match_status::same_epoch_only || value == match_status::exact_frame || refusal::classify(sunshine_streamline::name(value)) != refusal::reason_class::unclassified, "a match status the tag gate names has no refusal class");
    }
    for (auto stage = unsigned(record_stage::not_attempted); stage <= unsigned(record_stage::retained); ++stage) {
      const auto value = record_stage(stage);
      const auto reported = refusal::classify(value);
      const bool admitted = value == record_stage::not_attempted || value == record_stage::recorded || value == record_stage::retained;
      require(reported != refusal::reason_class::unclassified && (reported == refusal::reason_class::none) == admitted, "a capture record stage has no refusal class");
      ++counts.stages;
    }
    for (auto value = unsigned(depth_outcome::not_offered); value <= unsigned(depth_outcome::capture); ++value) {
      const auto depth = depth_outcome(value);
      const auto reported = refusal::classify(depth, record_stage::missing_state);
      require(reported != refusal::reason_class::unclassified && (reported == refusal::reason_class::none) == (refusal::gate_of(depth) == refusal::gate::none), "a DLSSG.Depth outcome has no refusal class or gate");
      ++counts.outcomes;
    }

    const auto fg = with(ngx_case {}, &ngx_case::feature, ngx_feature::frame_generation);
    const auto at_evaluate = tag(present, evaluate);
    const auto msaa = on_subresource(subresource_barrier::all, resource_shape::msaa);
    const auto outside = with(at_evaluate, &sl_case::extent, extent_form::outside_allocation);
    const auto after_present = tag(present, copy_point::after_present);
    const std::pair<const char *, matrix::outcome> named[] {
      {"boundary_without_commands", expect(tag(now, copy_point::tag_call, tag_commands::none))},
      {"unsupported_extension", expect(with(at_evaluate, &sl_case::tag_header, struct_form::unknown_chained))},
      {"unsupported_precision", expect(with(at_evaluate, &sl_case::precision, precision_form::zero_scale))},
      {"unsupported_precision", expect(with(at_evaluate, &sl_case::precision, precision_form::unknown_formula))},
      {"unsupported_resource_type", expect(with(at_evaluate, &sl_case::resource, resource_form::type_buffer))},
      {"unsupported_lifecycle", expect(tag(lifecycle::undeclared, evaluate))},
      {"lifecycle_outlives_boundary", expect(with(tag(until_evaluate, copy_point::tag_call), &sl_case::fg, fg_mode::off))},
      {"invalid-extent", expect(outside)},
      {"not-valid-until-present", expect(tag(now, evaluate))},
      {"missing-resource", expect(with(at_evaluate, &sl_case::resource, resource_form::null_native))},
      {"evaluation-failed", expect(with(at_evaluate, &sl_case::evaluate_result, sl_result::error))},
      {"ambiguous-viewport", expect(with(at_evaluate, &sl_case::views, viewports::two_active))},
      {"no_retained_source", expect(msaa)},
      {"missing_parameters", expect(with(ngx_case {}, &ngx_case::hw_depth, ngx_hw_depth::other))},
      {"missing_parameters", expect(with(ngx_case {}, &ngx_case::subrect, ngx_subrect::outside))},
      {"missing_parameters", expect(with(ngx_case {}, &ngx_case::depth_set, ngx_depth_set::missing))},
    };
    const std::pair<record_stage, matrix::outcome> stages[] {
      {record_stage::missing_state, expect(state(recorded(ds, evidence::none)))},
      {record_stage::missing_state, expect(fg)},
      {record_stage::incomplete_state, expect(state(recorded(ds, evidence::split_open, depth_read)))},
      {record_stage::conflicting_state, expect(state(recorded(ds, evidence::transition, depth_read), claim::sl_declared, nps))},
      {record_stage::unsupported_state, expect(state(recorded(ds, evidence::enhanced, 0, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, D3D12_BARRIER_ACCESS_NO_ACCESS)))},
      {record_stage::unsupported_state, expect(state(recorded(0, evidence::transition, D3D12_RESOURCE_STATE_COMMON)))},
      {record_stage::unsupported_proof, expect(state(recorded(ds, evidence::none), claim::sl_undeclared, UINT32_MAX))},
      {record_stage::unsupported_lifetime, expect(after_present)},
      {record_stage::source_lifetime, expect(after_present)},
      {record_stage::resource_region, expect(outside)},
      {record_stage::source_description, expect(msaa)},
      {record_stage::recording_render_pass, expect(with(at_evaluate, &sl_case::pass, render_pass::open))},
    };
    const std::pair<depth_outcome, matrix::outcome> outcomes[] {
      {depth_outcome::absent, expect(with(fg, &ngx_case::depth_set, ngx_depth_set::missing))},
      {depth_outcome::null_resource, expect(with(fg, &ngx_case::depth_set, ngx_depth_set::null))},
      {depth_outcome::subrect, expect(with(fg, &ngx_case::subrect, ngx_subrect::one_side))},
      {depth_outcome::capture, expect(fg)},  // No state recorded on DLSS-G's list: missing_state.
    };
    const auto check = [&](bool agreed, const char *what, const matrix::outcome &expected) {
      if (!agreed) {
        std::fprintf(stderr, "refusal class of %s disagrees with the oracle: expected %s%s%s rule=%s refusal=%u\n", what, name(expected.kind), matrix::either(expected) ? " or " : "", matrix::either(expected) ? name(expected.alternative) : "", expected.by ? expected.by->id : "?", unsigned(expected.why));
        throw std::runtime_error("depth_refusal.h diverged from the API contract matrix");
      }
      ++counts.paired;
    };
    for (const auto &[name, expected] : named) {
      check(agrees(refusal::classify(name), expected), name, expected);
    }
    for (const auto &[stage, expected] : stages) {
      check(agrees(refusal::classify(stage), expected), capture::name(stage), expected);
    }
    for (const auto &[depth, expected] : outcomes) {
      check(agrees(refusal::classify(depth, record_stage::missing_state), expected), sunshine_streamline::frame_generation::name(depth), expected);
    }
    return counts;
  }

  // The hook layer (L1).

  // A hook cell meets the oracle by its kind, deferral and copy basis.
  bool hook_satisfies(const matrix::outcome &expected, const matrix::result &actual) {
    if (actual.kind != matrix::outcome_kind::capture) {
      // Nothing was copied at this copy point: a refusal, this layer's part
      // of a hold (the hold itself belongs to the capture owner), or an
      // ignored input.
      return matrix::permits(expected, matrix::outcome_kind::refuse) || matrix::permits(expected, matrix::outcome_kind::hold) ||
             matrix::permits(expected, matrix::outcome_kind::ignored);
    }
    if (!matrix::permits(expected, matrix::outcome_kind::capture) && !matrix::permits(expected, matrix::outcome_kind::continuous)) {
      return false;
    }
    return actual.deferred == expected.deferred && (!expected.exact_basis || matrix::same_basis(expected.basis, actual.basis));
  }

  // The Streamline fixture SDK: noinline entry points the probe hooks, with
  // the volatile dispatch and ReShade stubs test_streamline_camera_probe.cpp
  // also defines. Its entry points only return each boundary's set result;
  // the probe tests' also record every call's arguments and can block or
  // lose observation inside the call, so each executable keeps its own.
  namespace sl_fixture {
    using namespace sunshine_streamline;
    constexpr DWORD outgoing_error = 0xabcdef01;
    constexpr guid fg_options_type {0xfac5f1cb, 0x2dfd, 0x4f36, {0xa1, 0xe6, 0x3a, 0x9e, 0x86, 0x52, 0x56, 0xc5}};
    constexpr guid precision_type {0x98f6e9ba, 0x8d16, 0x4831, {0xa8, 0x02, 0x4d, 0x3b, 0x52, 0xff, 0x26, 0xbf}};
    constexpr guid unknown_type {0x12345678, 1, 2, {3, 4, 5, 6, 7, 8, 9, 10}};

    struct fg_options {
      base_structure base {nullptr, fg_options_type, 5};
      std::uint32_t mode {}, generated_frames {};
    };

    using options_function = std::int32_t (*)(const abi_v2::viewport &, const fg_options &);

    struct precision_info {
      base_structure base {nullptr, precision_type, 1};
      std::uint32_t formula {1};
      float bias {.25f}, scale {.5f};
    };

    // FrameToken storage the observer never dereferences (an opaque address).
    struct opaque_token {
      alignas(16) std::uint64_t identity[2] {};

      const abi_v2::frame_token &ref() const {
        return *reinterpret_cast<const abi_v2::frame_token *>(identity);
      }
    };

    // The sl::Result each call returns, and the token the getter hands out.
    std::int32_t tag_result {}, evaluate_result {}, options_result {};
    abi_v2::frame_token *minted {};

    FIXTURE_NOINLINE std::int32_t set_constants(const abi_v2::constants &, const abi_v2::frame_token &, const abi_v2::viewport &) {
      SetLastError(outgoing_error);
      return 0;
    }

    FIXTURE_NOINLINE std::int32_t set_tag(const abi_v2::viewport &, const abi_v2::resource_tag *, std::uint32_t, void *) {
      SetLastError(outgoing_error);
      return tag_result;
    }

    FIXTURE_NOINLINE std::int32_t set_tag_for_frame(const abi_v2::frame_token &, const abi_v2::viewport &, const abi_v2::resource_tag *, std::uint32_t, void *) {
      SetLastError(outgoing_error);
      return tag_result;
    }

    FIXTURE_NOINLINE std::int32_t evaluate(std::uint32_t, const abi_v2::frame_token &, const base_structure **, std::uint32_t, void *) {
      SetLastError(outgoing_error);
      return evaluate_result;
    }

    FIXTURE_NOINLINE std::int32_t new_frame_token(abi_v2::frame_token *&token, const std::uint32_t *) {
      token = minted;
      SetLastError(outgoing_error);
      return 0;
    }

    FIXTURE_NOINLINE std::int32_t set_options(const abi_v2::viewport &, const fg_options &) {
      SetLastError(outgoing_error);
      return options_result;
    }

    FIXTURE_NOINLINE std::int32_t get_feature_function(std::uint32_t, const char *, void *&function) {
      function = reinterpret_cast<void *>(&set_options);
      SetLastError(outgoing_error);
      return 0;
    }

    // Volatile dispatch keeps the compiler from calling a fixture directly,
    // past the entry MinHook patches.
    abi_v2::set_constants volatile call_constants = &set_constants;
    abi_v2::set_tag volatile call_tag = &set_tag;
    abi_v2::set_tag_for_frame volatile call_frame_tag = &set_tag_for_frame;
    abi_v2::evaluate_feature volatile call_evaluate = &evaluate;
    abi_v2::get_new_frame_token volatile call_new_frame_token = &new_frame_token;
    abi_v2::get_feature_function volatile call_feature_function = &get_feature_function;

    const testing::targets targets(bool frame_tags) {
      return {reinterpret_cast<void *>(&set_constants), reinterpret_cast<void *>(&set_tag), frame_tags ? reinterpret_cast<void *>(&set_tag_for_frame) : nullptr, reinterpret_cast<void *>(&evaluate), reinterpret_cast<void *>(&new_frame_token), reinterpret_cast<void *>(&get_feature_function)};
    }

    // An exact reversed infinite-far D3D perspective and its analytic inverse.
    abi_v2::constants camera(std::uint64_t version) {
      abi_v2::constants value {};
      value.base = {nullptr, constants_guid, version};
      auto &common = value.common;
      common.camera_view_to_clip.m[0][0] = common.camera_view_to_clip.m[1][1] = 1;
      common.camera_view_to_clip.m[2][3] = 1;
      common.camera_view_to_clip.m[3][2] = .5f;
      common.clip_to_camera_view.m[0][0] = common.clip_to_camera_view.m[1][1] = 1;
      common.clip_to_camera_view.m[2][3] = 2;
      common.clip_to_camera_view.m[3][2] = 1;
      common.camera_near = .5f;
      common.camera_far = std::numeric_limits<float>::infinity();
      common.camera_fov = 1.5707963267948966f;
      common.camera_aspect = 1;
      value.depth_inverted = 1;
      return value;
    }

    abi_v2::viewport viewport(std::uint32_t id, std::uint64_t version) {
      abi_v2::viewport value {};
      value.base = {nullptr, viewport_guid, version};
      value.value = id;
      return value;
    }
  }  // namespace sl_fixture

  // The Streamline hook family: one dimension per sl_case field the hooks
  // can see, each value list starting with the oracle's default.
  namespace sl_cells {
    enum dimension : unsigned {
      version,
      entry,
      life,
      commands,
      at,
      generated,
      resource,
      extent,
      precision,
      tags,
      tag_header,
      resource_header,
      constants,
      viewport_handle,
      fg_options,
      views,
      fg,
      frames,
      tag_result,
      evaluate_result,
      options_result,
      untag,
      declared,
      observed
    };

    // The typed values behind the dimensions that are not one enumeration in
    // order; family() names them in the same order (check_sizes).
    constexpr matrix::sl_version versions[] {{2, 14, 1}, {2, 6, 10}, {2, 7, 30}, {2, 12, 0}, {2, 15, 0}, {3, 0, 0}};
    constexpr matrix::lifecycle lives[] {matrix::lifecycle::valid_until_present, matrix::lifecycle::only_valid_now, matrix::lifecycle::valid_until_evaluate, matrix::lifecycle::undeclared};
    constexpr matrix::copy_point points[] {matrix::copy_point::same_frame_evaluate, matrix::copy_point::tag_call, matrix::copy_point::after_present};
    constexpr matrix::struct_form tag_forms[] {matrix::struct_form::newest_known, matrix::struct_form::above_known, matrix::struct_form::unknown_chained};
    constexpr matrix::struct_form version_forms[] {matrix::struct_form::newest_known, matrix::struct_form::first, matrix::struct_form::above_known};
    constexpr std::uint32_t states[] {D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, UINT32_MAX, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_READ, D3D12_RESOURCE_STATE_GENERIC_READ};
    constexpr int tag_call_point = 1, frame_generation_on = 1;

    bool local(const generator::row &value) {
      return value[entry] >= int(matrix::tag_entry::local_array);
    }

    matrix::sl_case to_case(const generator::row &value) {
      matrix::sl_case out;
      out.version = versions[value[version]];
      out.entry = matrix::tag_entry(value[entry]);
      out.life = lives[value[life]];
      out.commands = matrix::tag_commands(value[commands]);
      out.at = points[value[at]];
      out.generated_presents = matrix::generated_presents[value[generated]];
      out.resource = matrix::resource_form(value[resource]);
      out.extent = matrix::extent_form(value[extent]);
      out.precision = matrix::precision_form(value[precision]);
      out.tags_in_call = matrix::tags_per_call[value[tags]];
      out.tag_header = tag_forms[value[tag_header]];
      out.resource_header = tag_forms[value[resource_header]];
      out.constants = version_forms[value[constants]];
      out.viewport_handle = value[viewport_handle] ? matrix::struct_form::above_known : matrix::struct_form::newest_known;
      out.frame_generation_options = version_forms[value[fg_options]];
      out.views = value[views] ? matrix::viewports::five_rotating : matrix::viewports::one;
      out.fg = matrix::fg_mode(value[fg]);
      out.frames_generated = matrix::frames_generated[value[frames]];
      out.tag_result = matrix::sl_result(value[tag_result]);
      out.evaluate_result = matrix::sl_result(value[evaluate_result]);
      out.options_result = matrix::sl_result(value[options_result]);
      out.untag = matrix::untag_pattern(value[untag]);
      out.declared_state = states[value[declared]];
      // The state the copy's recording names: the declared one, another one
      // (DEPTH_WRITE), or an aliasing barrier, which names none.
      out.depth = {D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL};
      switch (value[observed]) {
        case 1:
          out.depth.seen = matrix::evidence::transition;
          out.depth.observed = out.declared_state == UINT32_MAX ? std::uint32_t(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) : out.declared_state;
          break;
        case 2:
          out.depth.seen = matrix::evidence::transition;
          out.depth.observed = D3D12_RESOURCE_STATE_DEPTH_WRITE;
          break;
        case 3:
          out.depth.seen = matrix::evidence::alias_wildcard;
          break;
        default:
          break;
      }
      return out;
    }

    // Partial rows: each check runs once its dimensions are assigned.
    bool legal(const generator::row &value) {
      const auto assigned = [&](std::initializer_list<unsigned> dimensions) {
        for (const auto d : dimensions) {
          if (value[d] < 0) {
            return false;
          }
        }
        return true;
      };
      const auto untags = [&] {
        return value[resource] == int(matrix::resource_form::null_resource) || value[resource] == int(matrix::resource_form::null_native);
      };
      if (assigned({entry, version}) && value[entry] == int(matrix::tag_entry::set_tag_for_frame) && versions[value[version]] < matrix::first_frame_tag) {
        return false;
      }
      if (assigned({entry, at}) && local(value) && value[at] == tag_call_point) {
        return false;
      }
      if (assigned({commands, life, resource}) && value[commands] && value[life] && !untags()) {
        return false;
      }
      // An evaluate's command buffer is not optional (SLA:253).
      if (assigned({entry, commands}) && local(value) && value[commands]) {
        return false;
      }
      // A local input has no tag call to untag or to return a result, and a
      // zero-header Resource has no header to version or extend.
      if ((assigned({entry, untag}) && local(value) && value[untag]) || (assigned({entry, tag_result}) && local(value) && value[tag_result])) {
        return false;
      }
      if (assigned({resource, resource_header}) && value[resource] == int(matrix::resource_form::zero_header) && value[resource_header]) {
        return false;
      }
      return !generator::complete(value) || matrix::legal(to_case(value));
    }

    generator::family family() {
      return {
        "sl",
        {
          {"version", {"2.14.1", "2.6.10", "2.7.30", "2.12.0", "2.15.0", "3.0.0"}},
          {"entry", {"set_tag", "set_tag_for_frame", "local_array", "local_chain_not_last"}},
          {"life", {"until_present", "only_valid_now", "until_evaluate", "undeclared"}},
          {"commands", {"game_list", "none"}},
          {"at", {"same_frame_evaluate", "tag_call", "after_present"}},
          {"generated", {"0", "1", "3"}},
          {"resource", {"native", "null_resource", "null_native", "zero_header", "type_buffer", "type_unknown"}},
          {"extent", {"null", "whole", "padded", "offset", "one_dimension_zero", "outside_allocation", "changing", "declared_size_differs"}},
          {"precision", {"none", "scale_bias", "zero_scale", "unknown_formula"}},
          {"tags", {"1", "32", "33", "73"}},
          {"tag_header", {"newest_known", "above_known", "unknown_chained"}},
          {"resource_header", {"newest_known", "above_known", "unknown_chained"}},
          {"constants", {"newest_known", "first", "above_known"}},
          {"viewport_handle", {"newest_known", "above_known"}},
          {"fg_options", {"newest_known", "first", "above_known"}},
          {"views", {"one", "five_rotating"}},
          {"fg", {"off", "on", "automatic", "dynamic", "undeclared"}},
          {"frames", {"1", "0", "3", "5"}},
          {"tag_result", {"ok", "warn_out_of_vram", "error"}},
          {"evaluate_result", {"ok", "warn_out_of_vram", "error"}},
          {"options_result", {"ok", "warn_out_of_vram", "error"}},
          {"untag", {"none", "null_after_each", "before_tag"}},
          {"declared", {"non_pixel_shader_resource", "undeclared", "common", "depth_read", "generic_read"}},
          {"observed", {"none", "agrees", "disagrees", "alias"}},
        },
        legal,
      };
    }

    // Each dimension's names against the values to_case gives them: a typed
    // table's size, or an enumeration's count.
    void check_sizes(const generator::family &owner) {
      require(owner.dimensions.size() == observed + 1, "the Streamline hook family and its dimension enumeration differ");
      const auto sized = [&](unsigned d, std::size_t count) {
        require(owner.dimensions[d].values.size() == count, "a Streamline hook dimension's names and values differ in number");
      };
      sized(version, std::size(versions));
      sized(entry, std::size_t(matrix::tag_entry::local_chain_not_last) + 1);
      sized(life, std::size(lives));
      sized(commands, std::size_t(matrix::tag_commands::none) + 1);
      sized(at, std::size(points));
      sized(generated, std::size(matrix::generated_presents));
      sized(resource, std::size_t(matrix::resource_form::type_unknown) + 1);
      sized(extent, std::size_t(matrix::extent_form::declared_size_differs) + 1);
      sized(precision, std::size_t(matrix::precision_form::unknown_formula) + 1);
      sized(tags, std::size(matrix::tags_per_call));
      sized(tag_header, std::size(tag_forms));
      sized(resource_header, std::size(tag_forms));
      sized(constants, std::size(version_forms));
      sized(viewport_handle, 2);
      sized(fg_options, std::size(version_forms));
      sized(views, 2);
      sized(fg, std::size_t(matrix::fg_mode::undeclared) + 1);
      sized(frames, std::size(matrix::frames_generated));
      for (const auto d : {tag_result, evaluate_result, options_result}) {
        sized(d, std::size_t(matrix::sl_result::error) + 1);
      }
      sized(untag, std::size_t(matrix::untag_pattern::before_tag) + 1);
      sized(declared, std::size(states));
      sized(observed, 4);
    }

    // Rows from games the add-on was tuned on (design 3.2); the oracle still
    // decides each one. Witcher 3's token contention needs concurrent calls:
    // test_streamline_camera_probe.cpp's token tests cover it.
    struct seed {
      const char *name, *assignment;
    };

    constexpr seed seeds[] {
      {"shadows", "commands=none,at=tag_call,fg=on,untag=null_after_each,declared=generic_read"},
      {"shadows-dynamic-resolution", "commands=none,at=tag_call,extent=changing,fg=on,untag=null_after_each,declared=generic_read"},
      {"bf6", "fg=on,evaluate_result=warn_out_of_vram,options_result=warn_out_of_vram"},
      {"re9", "observed=alias"},
      {"e33", "extent=padded"},  // A 2228x1256 crop of a 3840x2160 allocation.
    };

    // One row per open owner question (a pending known-gap row), so the run
    // shows what the add-on does there today.
    constexpr seed questions[] {
      {"S24-question", "version=2.15.0,tags=32"},
      {"S14-question", "resource=zero_header,declared=depth_read"},
    };

    struct census {
      unsigned product {}, constructible {}, legal_product {}, seeded {}, questions {}, one_factor {}, pairwise {};
      std::size_t pairs {}, covered {}, unfilled {};
    };

    std::vector<generator::row> cells(const generator::family &owner, census &out) {
      std::vector<generator::row> rows;
      const auto base = generator::defaults(owner);
      // The full product of the copy-point dimensions. Frame generation is On
      // at tag-call copy points, the only cells where it makes a longer-lived
      // tag's call a copy point; its modes are paired with every dimension below.
      for (auto value : generator::product(owner, {entry, life, commands, at, generated}, base)) {
        ++out.product;
        if (local(value) && value[at] == tag_call_point) {
          continue;  // A local input has no tag call.
        }
        if (value[at] == tag_call_point) {
          value[fg] = frame_generation_on;
        }
        ++out.constructible;
        out.legal_product += matrix::legal(to_case(value));
        rows.push_back(value);
      }
      std::vector<generator::row> seeded;
      const auto parse_all = [&](const auto &named) {
        for (const auto &entry : named) {
          generator::row value;
          std::string error;
          require(generator::parse(owner, entry.assignment, value, error) && owner.legal(value), "a hook seed row is malformed or illegal");
          seeded.push_back(value);
        }
      };
      auto count = rows.size();
      parse_all(seeds);
      generator::append_unique(owner, rows, seeded);
      out.seeded = unsigned(rows.size() - count);
      count = rows.size();
      parse_all(questions);
      generator::append_unique(owner, rows, seeded);
      out.questions = unsigned(rows.size() - count);
      // Each value alone, at a same-frame evaluate and at frame-generation
      // tag calls of a longer-lived and of an OnlyValidNow tag.
      auto tag_call = base;
      tag_call[at] = tag_call_point;
      tag_call[fg] = frame_generation_on;
      auto only_valid_now = tag_call;
      only_valid_now[life] = int(1);
      count = rows.size();
      for (const auto &start : {base, tag_call, only_valid_now}) {
        generator::append_unique(owner, rows, generator::one_factor(owner, start));
      }
      out.one_factor = unsigned(rows.size() - count);
      const auto covering = generator::pairwise(owner, seeded);
      count = rows.size();
      generator::append_unique(owner, rows, covering.rows);
      out.pairwise = unsigned(rows.size() - count);
      out.pairs = covering.pairs;
      out.covered = covering.covered;
      out.unfilled = covering.unfilled;
      return rows;
    }

    const char *seed_name(const generator::family &owner, const generator::row &value) {
      const auto find = [&](const auto &named) -> const char * {
        for (const auto &entry : named) {
          generator::row parsed;
          std::string error;
          if (generator::parse(owner, entry.assignment, parsed, error) && parsed == value) {
            return entry.name;
          }
        }
        return nullptr;
      };
      const auto *name = find(seeds);
      return name ? name : find(questions);
    }

    // What the probe published at the copy point.
    struct observation {
      bool installed {}, published {}, nominated {}, forwarded {true};
      sunshine_streamline::evaluation_snapshot snapshot;
      sunshine_scene_depth::frame frame;
      std::uint32_t presents {};  // Device Presents between the tag and the copy.
      sunshine_streamline::extent area {};  // The depth tag's extent.
    };

    // The release whose hooks are installed. Consecutive cases of one release
    // keep them and start a fresh observation epoch instead (initialize, as
    // when ReShade re-registers the add-on), which keeps the hook layer fast:
    // installing patches and unpatches every hooked entry point.
    struct installation {
      bool valid {}, installed {};
      matrix::sl_version version;
    };

    installation installed;

    bool install(const matrix::sl_version &version) {
      if (installed.valid && installed.version == version) {
        sunshine_streamline::initialize(nullptr);
        return installed.installed;
      }
      sunshine_streamline::testing::clear();
      const bool frame_tags = !(version < matrix::first_frame_tag);
      installed = {true, sunshine_streamline::testing::install({version.major, version.minor, version.patch, 0}, sl_fixture::targets(frame_tags)), version};
      return installed.installed;
    }

    void uninstall() {
      sunshine_streamline::testing::clear();
      installed = {};
    }

    std::int32_t sdk_result(matrix::sl_result value) {
      switch (value) {
        case matrix::sl_result::ok:
          return 0;
        case matrix::sl_result::warn_out_of_vram:
          return sunshine_streamline::abi_v2::result_warn_out_of_vram;
        default:
          return 3;  // eErrorOSOutOfDate, one of the eError* values (sl_result.h).
      }
    }

    // Extents {top, left, width, height} of a 3840x2160 allocation.
    sunshine_streamline::extent area(matrix::extent_form form) {
      switch (form) {
        case matrix::extent_form::whole:
          return {0, 0, 3840, 2160};
        case matrix::extent_form::padded:
        case matrix::extent_form::declared_size_differs:  // Beyond a declared 1920x1080.
          return {0, 0, 2228, 1256};
        case matrix::extent_form::offset:
          return {120, 200, 2228, 1256};
        case matrix::extent_form::one_dimension_zero:
          return {0, 0, 3840, 0};
        case matrix::extent_form::outside_allocation:
          return {0, 0, 3968, 2160};
        case matrix::extent_form::changing:  // After a 2206x1240 frame (changing_from).
          return {0, 0, 2658, 1494};
        default:
          return {};
      }
    }

    // Dynamic resolution's previous render size (Assassin's Creed Shadows,
    // e31c9d2c).
    constexpr sunshine_streamline::extent changing_from {0, 0, 2206, 1240};

    std::uint64_t latest_sequence(std::uint32_t viewport) {
      sunshine_streamline::evaluation_snapshot snapshot;
      return sunshine_streamline::testing::latest_snapshot(viewport, snapshot) ? snapshot.sequence : 0;
    }

    void observe(observation &out, std::uint32_t viewport, std::uint64_t before) {
      out.published = sunshine_streamline::testing::latest_snapshot(viewport, out.snapshot) && out.snapshot.sequence > before;
      out.nominated = out.published && sunshine_streamline::testing::normalized_source(viewport, 0, out.frame);
    }

    // One case through the real hooks: frame-generation options, then four
    // other viewports' frames when they rotate, then the case's frame, and
    // for a copy after Present the next frame's evaluation without a new tag.
    // A changing extent first runs the same frame at another render size,
    // observed into previous.
    observation run(const matrix::sl_case &value, observation *previous = nullptr) {
      using namespace sunshine_streamline;
      namespace fx = sl_fixture;
      observation out;
      fx::tag_result = fx::evaluate_result = fx::options_result = 0;
      out.installed = install(value.version);
      const auto forward = [&](std::int32_t result, std::int32_t expected) {
        out.forwarded = out.forwarded && result == expected && GetLastError() == fx::outgoing_error;
      };
      std::uint32_t index {};
      const auto mint = [&](const fx::opaque_token &token) {
        fx::minted = const_cast<abi_v2::frame_token *>(&token.ref());
        abi_v2::frame_token *issued {};
        ++index;
        forward(fx::call_new_frame_token(issued, &index), 0);
      };
      int list {}, native {}, other_native {};
      void *const game_list = &list;
      constexpr std::uint32_t id = 7;
      // The newest structure versions a reader knows are the newest surveyed
      // release's (survey tables); above_known sends one more.
      const auto &newest = matrix::newest_known();
      const auto version_of = [](matrix::struct_form form, std::uint64_t known) -> std::uint64_t {
        if (form == matrix::struct_form::first) {
          return 1;
        }
        return form == matrix::struct_form::above_known ? known + 1 : known;
      };
      auto view = fx::viewport(id, version_of(value.viewport_handle, newest.viewport_handle));
      const auto camera = fx::camera(version_of(value.constants, newest.constants));
      abi_v2::resource other {};
      other.base = {nullptr, resource_guid, newest.resource};
      other.native = &other_native;
      other.state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
      other.width = 3840;
      other.height = 2160;
      if (value.views == matrix::viewports::five_rotating) {
        fx::opaque_token rotating_token;
        for (std::uint32_t step = 1; step <= 4; ++step) {
          auto rotating = fx::viewport(id + step, newest.viewport_handle);
          mint(rotating_token);
          forward(fx::call_constants(fx::camera(newest.constants), rotating_token.ref(), rotating), 0);
          abi_v2::resource_tag tag {{nullptr, tag_guid, newest.resource_tag}, &other, 0, 1, {}};
          forward(fx::call_tag(rotating, &tag, 1, game_list), 0);
          const base_structure *inputs[] {&rotating.base};
          forward(fx::call_evaluate(0, rotating_token.ref(), inputs, 1, game_list), 0);
        }
      }
      fx::tag_result = sdk_result(value.tag_result);
      fx::evaluate_result = sdk_result(value.evaluate_result);
      fx::options_result = sdk_result(value.options_result);
      if (value.fg != matrix::fg_mode::off) {
        void *function {};
        forward(fx::call_feature_function(1000, "slDLSSGSetOptions", function), 0);
        fx::fg_options options;
        options.base.version = version_of(value.frame_generation_options, newest.dlssg_options);
        options.mode = std::uint32_t(value.fg);
        options.generated_frames = value.frames_generated;
        if (function) {
          forward(reinterpret_cast<fx::options_function>(function)(view, options), fx::options_result);
        }
        // At a later copy point the frame-generation scope is silent (no FG
        // copy for a second), the case where an evaluation supplies the
        // viewport's depth (docs/reshade-sbs.md, frame-generation ownership).
        if (out.installed && value.at != matrix::copy_point::tag_call) {
          testing::age_frame_generation(1001);
        }
      }
      // The depth Resource and its tag, then tags_in_call - 1 other tags.
      abi_v2::resource resource {};
      base_structure resource_extension {nullptr, fx::unknown_type, 1};
      resource.base = {nullptr, resource_guid, version_of(value.resource_header, newest.resource)};
      if (value.resource_header == matrix::struct_form::unknown_chained) {
        resource.base.next = &resource_extension;
      }
      if (value.resource == matrix::resource_form::zero_header) {
        resource.base = {};
      }
      resource.type = 0;  // ResourceType eTex2d, eBuffer 1, eUnknown 8.
      if (value.resource == matrix::resource_form::type_buffer || value.resource == matrix::resource_form::type_unknown) {
        resource.type = value.resource == matrix::resource_form::type_buffer ? 1 : 8;
      }
      resource.native = value.resource == matrix::resource_form::null_native ? nullptr : &native;
      resource.state = value.declared_state;
      const bool declared_small = value.extent == matrix::extent_form::declared_size_differs;
      resource.width = declared_small ? 1920 : 3840;
      resource.height = declared_small ? 1080 : 2160;
      const auto tag_version = version_of(value.tag_header, newest.resource_tag);
      std::vector<abi_v2::resource_tag> tags(value.tags_in_call);
      for (auto &tag : tags) {
        tag = {{nullptr, tag_guid, tag_version}, &other, 1, std::uint32_t(value.life), {}};  // Motion vectors.
      }
      auto &depth = tags[0];
      depth.resource_ptr = value.resource == matrix::resource_form::null_resource ? nullptr : &resource;
      depth.type = 0;
      // The depth tag's next chain: PrecisionInfo, an unknown structure, and
      // in a chained input list the call's other inputs and its viewport.
      fx::precision_info precision;
      precision.formula = value.precision == matrix::precision_form::unknown_formula ? 2 : 1;
      precision.scale = value.precision == matrix::precision_form::zero_scale ? 0.f : .5f;
      base_structure tag_extension {nullptr, fx::unknown_type, 1};
      std::vector<base_structure *> links {&depth.base};
      if (value.precision != matrix::precision_form::none) {
        links.push_back(&precision.base);
      }
      if (value.tag_header == matrix::struct_form::unknown_chained) {
        links.push_back(&tag_extension);
      }
      std::vector<const base_structure *> inputs {&view.base};
      if (value.entry == matrix::tag_entry::local_chain_not_last) {
        for (std::size_t i = 1; i < tags.size(); ++i) {
          links.push_back(&tags[i].base);
        }
        links.push_back(&view.base);
        inputs = {&depth.base};
      } else if (value.entry == matrix::tag_entry::local_array) {
        for (const auto &tag : tags) {
          inputs.push_back(&tag.base);
        }
      }
      for (std::size_t i = 0; i + 1 < links.size(); ++i) {
        links[i]->next = links[i + 1];
      }
      const abi_v2::resource_tag untag {{nullptr, tag_guid, newest.resource_tag}, nullptr, 0, std::uint32_t(value.life), {}};
      const bool local = matrix::known_gaps::streamline::local(value.entry);
      void *const tag_commands = value.commands == matrix::tag_commands::game_list ? game_list : nullptr;
      fx::opaque_token tokens[4];
      const auto frame = [&](const sunshine_streamline::extent &tagged, const fx::opaque_token *token, observation &seen) {
        const auto tag_call = [&](const abi_v2::resource_tag *values, std::uint32_t count) {
          if (value.entry == matrix::tag_entry::set_tag_for_frame) {
            forward(fx::call_frame_tag(token[0].ref(), view, values, count, tag_commands), fx::tag_result);
          } else {
            forward(fx::call_tag(view, values, count, tag_commands), fx::tag_result);
          }
        };
        depth.area = seen.area = tagged;
        mint(token[0]);
        forward(fx::call_constants(camera, token[0].ref(), view), 0);
        if (!local) {
          if (value.untag == matrix::untag_pattern::before_tag) {
            tag_call(&untag, 1);
          }
          const auto before = latest_sequence(id);
          tag_call(tags.data(), value.tags_in_call);
          if (value.at == matrix::copy_point::tag_call) {
            observe(seen, id, before);
            return;
          }
          if (value.untag == matrix::untag_pattern::null_after_each) {
            tag_call(&untag, 1);
          }
        }
        auto before = latest_sequence(id);
        forward(fx::call_evaluate(0, token[0].ref(), inputs.data(), std::uint32_t(inputs.size()), local ? tag_commands : game_list), fx::evaluate_result);
        if (value.at == matrix::copy_point::same_frame_evaluate) {
          // A local input is tagged at this evaluate: no Present comes between.
          seen.presents = local ? 0 : value.generated_presents;
          observe(seen, id, before);
          return;
        }
        mint(token[1]);
        forward(fx::call_constants(camera, token[1].ref(), view), 0);
        const base_structure *next_inputs[] {&view.base};
        before = latest_sequence(id);
        forward(fx::call_evaluate(0, token[1].ref(), next_inputs, 1, game_list), fx::evaluate_result);
        seen.presents = 1 + value.generated_presents;
        observe(seen, id, before);
      };
      observation earlier;
      if (value.extent == matrix::extent_form::changing) {
        frame(changing_from, tokens, previous ? *previous : earlier);
      }
      frame(area(value.extent), tokens + 2, out);
      return out;
    }

    // The capture owner's admission of the probe's nomination.
    matrix::result admission(const matrix::sl_case &value, const observation &run) {
      using sunshine_scene_depth::lifetime;
      matrix::result out;
      if (!run.nominated) {
        return out;
      }
      const auto &frame = run.frame;
      const bool at_call = run.snapshot.tag_boundary;
      const bool forced = at_call && frame.valid_until == lifetime::at_call;
      // Repeats of record_impl (streamline_depth_capture.cpp), which needs a
      // device: a command list, a lifetime it keeps, and
      // source_lifetime_current: inside the tag call, or no Present since.
      // A known gap these decide (S7) closes only with a change here.
      if (!run.snapshot.command_buffer) {
        return out;
      }
      if (frame.valid_until != lifetime::until_present && frame.valid_until != lifetime::until_evaluation && !forced) {
        return out;
      }
      if (!at_call && !forced && run.presents) {
        return out;
      }
      return judged(copy_state::resolve(copy_state::select(frame.proof, policy::source_contract), frame.native_state, observation_of(value.depth), value.depth.flags));
    }

    bool newer_struct(const matrix::sl_case &value) {
      for (const auto form : {value.tag_header, value.resource_header, value.constants, value.viewport_handle, value.frame_generation_options}) {
        if (form == matrix::struct_form::above_known) {
          return true;
        }
      }
      return false;
    }

    matrix::sl_case newest_twin(matrix::sl_case value) {
      for (auto *form : {&value.tag_header, &value.resource_header, &value.constants, &value.viewport_handle, &value.frame_generation_options}) {
        if (*form == matrix::struct_form::above_known) {
          *form = matrix::struct_form::newest_known;
        }
      }
      return value;
    }

    // S13: a newer structure version reads as the newest known one.
    bool same_nomination(const observation &a, const observation &b) {
      if (a.nominated != b.nominated) {
        return false;
      }
      return !a.nominated || (a.frame.valid_until == b.frame.valid_until && a.frame.proof == b.frame.proof && a.frame.native_state == b.frame.native_state &&
                              a.frame.frame_generation_input == b.frame.frame_generation_input && a.frame.projection.supplied == b.frame.projection.supplied);
    }

    // R3: two render sizes of one tag are one logical source (provider,
    // epoch, viewport, source id, frame-generation role and resource), each
    // copy with its own extent.
    bool same_source(const observation &a, const observation &b) {
      const auto keeps_area = [](const observation &value) {
        const auto &area = value.frame.resource.area;
        return area.left == value.area.left && area.top == value.area.top && area.width == value.area.width && area.height == value.area.height;
      };
      return a.frame.provider == b.frame.provider && a.frame.epoch == b.frame.epoch && a.frame.viewport == b.frame.viewport &&
             a.frame.source_id == b.frame.source_id && a.frame.frame_generation_input == b.frame.frame_generation_input &&
             a.frame.resource.native == b.frame.resource.native && a.frame.resource.kind == b.frame.resource.kind && keeps_area(a) && keeps_area(b);
    }

    constexpr unsigned row_count = std::size(gaps::sl_rows);

    struct judgement {
      matrix::sl_case value;
      matrix::outcome expected;
      matrix::result actual;
      observation observed, previous;
      bool legal {}, twin_checked {}, twin_same {true}, continuity_checked {}, continuous {true}, satisfied {};
      gaps::verdict verdict {};
      bool matched[row_count] {};
    };

    // Any complete assignment, legal or not: illegal input must be refused.
    judgement judge(const generator::row &assignment) {
      judgement out;
      out.value = to_case(assignment);
      out.legal = legal(assignment);
      out.expected = matrix::expect(out.value);
      require(out.expected.by != nullptr, "the oracle named no rule for a Streamline hook cell");
      out.observed = run(out.value, &out.previous);
      require(out.observed.forwarded, "the probe changed a Streamline result or LastError");
      require(out.observed.installed || out.value.version.major != 2, "the probe did not hook a 2.x fixture SDK");
      out.actual = admission(out.value, out.observed);
      if (out.value.extent == matrix::extent_form::changing && out.actual.kind == matrix::outcome_kind::capture) {
        out.continuity_checked = true;
        out.continuous = admission(out.value, out.previous).kind == matrix::outcome_kind::capture && same_source(out.previous, out.observed);
      }
      if (out.legal && matrix::permits(out.expected, matrix::outcome_kind::capture) && newer_struct(out.value)) {
        const auto twin = newest_twin(out.value);
        const auto newest = run(twin);
        out.twin_checked = true;
        out.twin_same = same_nomination(out.observed, newest) && admission(twin, newest).kind == out.actual.kind;
      }
      // Illegal input must end in a refusal; its class is diagnostic.
      out.satisfied = out.legal ? hook_satisfies(out.expected, out.actual) && out.twin_same && out.continuous : out.actual.kind != matrix::outcome_kind::capture;
      out.verdict = gaps::judge_each(gaps::sl_rows, out.value, out.satisfied, {out.actual, out.twin_same && out.continuous}, out.matched);
      return out;
    }

    void describe(const char *what, const generator::family &owner, const generator::row &assignment, const judgement &value) {
      const auto &seen = value.observed;
      std::string rows;
      const char *ids[row_count] {};
      for (unsigned i = 0; i != row_count; ++i) {
        ids[i] = gaps::sl_rows[i].id;
      }
      std::fprintf(stderr, "%s: case sl %s %s%s\n", what, generator::id_text(generator::case_id(owner, assignment)).c_str(), generator::describe(owner, assignment).c_str(), value.legal ? "" : " (illegal input)");
      describe_expected(value.expected);
      std::fprintf(stderr, "  probe: installed=%u published=%u nominated=%u boundary=%u command=%u fg_input=%u valid_until=%u proof=%u state=0x%x projection=%u presents=%u; status=%s tag present=%u supported=%u refusal=%u\n", seen.installed, seen.published, seen.nominated, seen.snapshot.tag_boundary, seen.snapshot.command_buffer != 0, seen.frame.frame_generation_input, unsigned(seen.frame.valid_until), unsigned(seen.frame.proof), seen.frame.native_state, seen.frame.projection.supplied, seen.presents, sunshine_streamline::name(seen.snapshot.status), seen.snapshot.tags[0].present, seen.snapshot.tags[0].supported, unsigned(seen.snapshot.tags[0].value.refusal));
      const char *twin = !value.twin_checked ? "not checked" : value.twin_same ? "same" :
                                                                                 "differs";
      const char *continuity = !value.continuity_checked ? "not checked" : value.continuous ? "same source" :
                                                                                              "not continuous";
      std::fprintf(stderr, "  admission: %s basis=0x%x/0x%x/0x%x enhanced=%u; newest-version twin %s; previous render size %s; rows %s; %s\n", name(value.actual.kind), value.actual.basis.state, value.actual.basis.layout, value.actual.basis.access, value.actual.basis.enhanced, twin, continuity, row_names(value.matched, ids, row_count, rows), name(value.verdict));
    }
  }  // namespace sl_cells

  // The NGX hook family: the evaluation adapter's dimensions, all combined.
  namespace ngx_cells {
    enum dimension : unsigned {
      feature,
      hardware_depth,
      inverted,
      depth_set,
      subrect,
      result,
      lifetime,
      state
    };

    constexpr matrix::ngx_feature features[] {matrix::ngx_feature::super_resolution, matrix::ngx_feature::ray_reconstruction, matrix::ngx_feature::frame_generation, matrix::ngx_feature::unknown};

    matrix::ngx_case to_case(const generator::row &value) {
      matrix::ngx_case out;
      out.feature = features[value[feature]];
      out.hw_depth = matrix::ngx_hw_depth(value[hardware_depth]);
      out.depth_inverted = value[inverted] != 0;
      out.depth_set = matrix::ngx_depth_set(value[depth_set]);
      out.subrect = matrix::ngx_subrect(value[subrect]);
      out.result = matrix::ngx_result(value[result]);
      out.lifetime = matrix::ngx_lifetime(value[lifetime]);
      out.depth = {D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL};
      if (value[state] == 1) {
        out.depth.flags |= D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
      } else if (value[state] == 2) {
        out.depth.seen = matrix::evidence::transition;
        out.depth.observed = D3D12_RESOURCE_STATE_DEPTH_READ;
      }
      return out;
    }

    bool legal(const generator::row &value) {
      return !generator::complete(value) || matrix::legal(to_case(value));
    }

    // A subrect outside the allocation needs the allocation's size, which
    // only the capture owner reads (source_region): the device layer's.
    generator::family family();

    // Each dimension's names against the values to_case gives them.
    void check_sizes(const generator::family &owner) {
      const std::size_t counts[] {std::size(features), std::size_t(matrix::ngx_hw_depth::other) + 1, 2, std::size_t(matrix::ngx_depth_set::null) + 1, std::size_t(matrix::ngx_subrect::inside) + 1, std::size_t(matrix::ngx_result::failure) + 1, std::size_t(matrix::ngx_lifetime::nested_in_streamline) + 1, 3};
      require(owner.dimensions.size() == std::size(counts), "the NGX hook family and its dimension enumeration differ");
      for (unsigned d = 0; d != std::size(counts); ++d) {
        require(owner.dimensions[d].values.size() == counts[d], "an NGX hook dimension's names and values differ in number");
      }
    }

    generator::family family() {
      return {
        "ngx",
        {
          {"feature", {"super_resolution", "ray_reconstruction", "frame_generation", "unknown"}},
          {"hardware_depth", {"absent", "linear", "device", "other"}},
          {"inverted", {"no", "yes"}},
          {"depth_set", {"d3d12_resource", "void_pointer", "missing", "null"}},
          {"subrect", {"missing", "zero", "one_side", "inside"}},
          {"result", {"success", "failure"}},
          {"lifetime", {"steady", "release_recreate", "nested_in_streamline"}},
          {"state", {"contract", "deny_shader_resource", "observed_depth_read"}},
        },
        legal,
      };
    }

    // Hogwarts Legacy's NGX depth had no provable state (design 3.2).
    constexpr const char *hogwarts = "state=deny_shader_resource";

    std::vector<generator::row> cells(const generator::family &owner) {
      std::vector<unsigned> every;
      for (unsigned i = 0; i != owner.dimensions.size(); ++i) {
        every.push_back(i);
      }
      std::vector<generator::row> rows;
      for (const auto &value : generator::product(owner, every, generator::defaults(owner))) {
        if (owner.legal(value)) {
          rows.push_back(value);
        }
      }
      return rows;
    }

    namespace dlssg = sunshine_streamline::frame_generation;

    struct observation {
      bool recorded {}, finished {}, succeeded {}, suppressed {};
      sunshine_scene_depth::frame frame;
      // DLSS-G's own inputs as the adapter read them, and their completion.
      bool offered {}, completed {}, completed_successfully {};
      dlssg::inputs inputs;
    };

    observation *current {};
    constexpr std::uint64_t command = 0x1000;

    // The parameter map as the SDK helpers fill it: SR's keys
    // (nvsdk_ngx_helpers_d3d.h), or DLSS-G's (nvsdk_ngx_helpers_dlssg_d3d.h)
    // for feature 11. A subrect with one side leaves its height unset.
    ngx_fixture::parameters parameters_of(const matrix::ngx_case &value, void *depth) {
      ngx_fixture::parameters out;
      const bool frame_generation = value.feature == matrix::ngx_feature::frame_generation;
      out.set_i("DLSS.Feature.Create.Flags", value.depth_inverted ? 8 : 0);  // NVSDK_NGX_DLSS_Feature_Flags_DepthInverted.
      out.set_ui("Width", 1920);
      out.set_ui("Height", 1080);
      const char *depth_key = frame_generation ? "DLSSG.Depth" : "Depth";
      if (value.depth_set == matrix::ngx_depth_set::void_pointer) {
        out.set_pointer(depth_key, depth);
      } else if (value.depth_set == matrix::ngx_depth_set::d3d12_resource || value.depth_set == matrix::ngx_depth_set::null) {
        out.set_resource(depth_key, value.depth_set == matrix::ngx_depth_set::null ? nullptr : depth);
      }
      const bool zero = value.subrect == matrix::ngx_subrect::zero, inside = value.subrect == matrix::ngx_subrect::inside;
      const unsigned width = zero ? 0 : 1280, height = zero ? 0 : 720;
      if (frame_generation) {
        out.set_ui("DLSSG.DepthInverted", value.depth_inverted);
        if (value.subrect != matrix::ngx_subrect::missing) {
          out.set_ui("DLSSG.DepthSubrectBaseX", zero ? 0 : 3);
          out.set_ui("DLSSG.DepthSubrectBaseY", zero ? 0 : 2);
          out.set_ui("DLSSG.DepthSubrectWidth", width);
          if (zero || inside) {
            out.set_ui("DLSSG.DepthSubrectHeight", height);
          }
        }
        return out;
      }
      if (value.hw_depth != matrix::ngx_hw_depth::absent) {
        const bool device = value.hw_depth == matrix::ngx_hw_depth::device;
        out.set_ui("DLSS.Use.HW.Depth", value.hw_depth == matrix::ngx_hw_depth::other ? 2 : device ? 1 :
                                                                                                     0);
      }
      if (value.subrect != matrix::ngx_subrect::missing) {
        out.set_ui("DLSS.Render.Subrect.Dimensions.Width", width);
        if (zero || inside) {
          out.set_ui("DLSS.Render.Subrect.Dimensions.Height", height);
        }
      }
      if (inside) {
        out.set_ui("DLSS.Input.Depth.Subrect.Base.X", 3);
        out.set_ui("DLSS.Input.Depth.Subrect.Base.Y", 2);
      }
      return out;
    }

    // One feature's creation (and release and recreation) and evaluation,
    // through the adapter's own calls in hook order. DLSS-G's inputs stop
    // where the adapter hands them to the probe: placing them in a
    // Streamline FG scope and retaining the resource need a device (the
    // runtime layer's).
    observation run(const matrix::ngx_case &value) {
      observation out;
      current = &out;
      sunshine_ngx::initialize(true);
      sunshine_ngx::testing::callbacks callbacks;
      callbacks.record = [](std::uint64_t, const sunshine_scene_depth::frame &frame) -> std::uint64_t {
        current->recorded = true;
        current->frame = frame;
        return 91;
      };
      callbacks.finish = [](std::uint64_t, bool successful) {
        current->finished = true;
        current->succeeded = successful;
      };
      callbacks.frame_generation = [](const dlssg::inputs &inputs) {
        current->offered = true;
        current->inputs = inputs;
        dlssg::attempt attempt;
        attempt.observation = 1;
        return attempt;
      };
      callbacks.finish_frame_generation = [](const dlssg::attempt &attempt, bool successful) {
        if (attempt.observation) {
          current->completed = true;
          current->completed_successfully = successful;
        }
      };
      sunshine_ngx::testing::set_callbacks(callbacks);
      int depth {};
      auto parameters = parameters_of(value, &depth);
      const auto owner = GetModuleHandleW(nullptr);
      const auto api = sunshine_ngx::resolve_parameter_api(owner);
      const auto id = std::uint32_t(value.feature);
      auto *handle = reinterpret_cast<void *>(0x5a5a0000);
      sunshine_ngx::after_create(owner, handle, sunshine_ngx::before_create(api, id, &parameters), true);
      if (value.lifetime == matrix::ngx_lifetime::release_recreate) {
        sunshine_ngx::after_evaluate(sunshine_ngx::before_evaluate(owner, api, command, handle, &parameters), true);
        sunshine_ngx::after_release(owner, handle, sunshine_ngx::epoch(), true);
        handle = reinterpret_cast<void *>(0x5a5b0000);
        sunshine_ngx::after_create(owner, handle, sunshine_ngx::before_create(api, id, &parameters), true);
        out = {};
      }
      // A Streamline evaluation that holds capture authority encloses a
      // nested NGX call (upscaler_call_trace.h streamline_scope).
      const auto attempt = sunshine_ngx::before_evaluate(owner, api, command, handle, &parameters, value.lifetime != matrix::ngx_lifetime::nested_in_streamline);
      sunshine_ngx::after_evaluate(attempt, value.result == matrix::ngx_result::success);
      out.suppressed = value.lifetime == matrix::ngx_lifetime::nested_in_streamline;
      sunshine_ngx::testing::set_callbacks({});
      current = nullptr;
      return out;
    }

    matrix::result admission(const matrix::ngx_case &value, const observation &run) {
      matrix::result out;
      if (value.feature == matrix::ngx_feature::frame_generation) {
        // observe_frame_generation_inputs nominates DLSSG.Depth with a valid
        // subrect on the evaluation's own command list, provable only by a
        // state observed there (observed_nonzero, no declared state), and
        // keeps the copy when the evaluation succeeds.
        const auto &depth = run.inputs.depth;
        if (!run.offered || depth.state != dlssg::parameter::present || !depth.area_valid || run.inputs.command != command || !run.completed || !run.completed_successfully) {
          return out;
        }
        return judged(copy_state::resolve(copy_state::select(proof::observed_nonzero, policy::source_contract), UINT32_MAX, observation_of(value.depth), value.depth.flags));
      }
      // Nothing recorded, or the copy discarded with its failed evaluation.
      if (!run.recorded || !run.finished || !run.succeeded) {
        return out;
      }
      return judged(copy_state::resolve(copy_state::select(run.frame.proof, policy::source_contract), run.frame.native_state, observation_of(value.depth), value.depth.flags));
    }

    constexpr unsigned row_count = std::size(gaps::ngx_rows);

    struct judgement {
      matrix::ngx_case value;
      matrix::outcome expected;
      matrix::result actual;
      observation observed;
      bool satisfied {};
      gaps::verdict verdict {};
      bool matched[row_count] {};
    };

    judgement judge(const generator::row &assignment) {
      judgement out;
      out.value = to_case(assignment);
      out.expected = matrix::expect(out.value);
      require(out.expected.by != nullptr, "the oracle named no rule for an NGX hook cell");
      out.observed = run(out.value);
      require(!out.observed.recorded || out.observed.finished, "an NGX capture ticket was never finished");
      require(!out.observed.offered || out.observed.completed, "DLSS-G's inputs were never finished");
      out.actual = admission(out.value, out.observed);
      // Nested inside a Streamline evaluation that holds capture authority,
      // NGX records and offers nothing: the frame has one source (N9b).
      out.satisfied = out.value.lifetime == matrix::ngx_lifetime::nested_in_streamline ? !out.observed.recorded && !out.observed.offered :
                                                                                         hook_satisfies(out.expected, out.actual);
      out.verdict = gaps::judge_each(gaps::ngx_rows, out.value, out.satisfied, {out.actual}, out.matched);
      return out;
    }

    void describe(const char *what, const generator::family &owner, const generator::row &assignment, const judgement &value) {
      std::string rows;
      const char *ids[row_count] {};
      for (unsigned i = 0; i != row_count; ++i) {
        ids[i] = gaps::ngx_rows[i].id;
      }
      std::fprintf(stderr, "%s: case ngx %s %s\n", what, generator::id_text(generator::case_id(owner, assignment)).c_str(), generator::describe(owner, assignment).c_str());
      describe_expected(value.expected);
      const auto &depth = value.observed.inputs.depth;
      std::fprintf(stderr, "  adapter: recorded=%u finished=%u succeeded=%u proof=%u state=0x%x valid_until=%u; DLSS-G offered=%u depth=%u area_valid=%u completed=%u/%u; admission %s basis=0x%x; rows %s; %s\n", value.observed.recorded, value.observed.finished, value.observed.succeeded, unsigned(value.observed.frame.proof), value.observed.frame.native_state, unsigned(value.observed.frame.valid_until), value.observed.offered, unsigned(depth.state), depth.area_valid, value.observed.completed, value.observed.completed_successfully, name(value.actual.kind), value.actual.basis.state, row_names(value.matched, ids, row_count, rows), name(value.verdict));
    }
  }  // namespace ngx_cells

  struct hook_counts {
    sl_cells::census census;
    each_tally<sl_cells::row_count> streamline;
    each_tally<ngx_cells::row_count> ngx;
    unsigned ngx_frame_generation {};
  };

  // Every selected case of both hook families.
  hook_counts hook_layer(const generator::options &options) {
    hook_counts counts;
    const auto sl = sl_cells::family();
    sl_cells::check_sizes(sl);
    const auto streamline = sl_cells::cells(sl, counts.census);
    require(counts.census.covered == counts.census.pairs && !counts.census.unfilled, "the Streamline covering array left a legal pair uncovered");
    for (std::size_t i = 0; i != streamline.size(); ++i) {
      if (!options.selected(i)) {
        continue;
      }
      const auto value = sl_cells::judge(streamline[i]);
      if (const auto *what = counts.streamline.add(value.verdict, value.matched, value.satisfied)) {
        sl_cells::describe(what, sl, streamline[i], value);
      }
    }
    sl_cells::uninstall();
    const auto ngx = ngx_cells::family();
    ngx_cells::check_sizes(ngx);
    const auto evaluations = ngx_cells::cells(ngx);
    for (std::size_t i = 0; i != evaluations.size(); ++i) {
      if (!options.selected(i)) {
        continue;
      }
      const auto value = ngx_cells::judge(evaluations[i]);
      counts.ngx_frame_generation += value.value.feature == matrix::ngx_feature::frame_generation;
      if (const auto *what = counts.ngx.add(value.verdict, value.matched, value.satisfied)) {
        ngx_cells::describe(what, ngx, evaluations[i], value);
      }
    }
    sunshine_ngx::shutdown();
    counts.streamline.finish(gaps::sl_rows, "Streamline hooks", !options.partial());
    counts.ngx.finish(gaps::ngx_rows, "NGX hooks", !options.partial());
    return counts;
  }

  // --list: every hook case; --case: one, described.
  int list_cases() {
    sl_cells::census census;
    const auto sl = sl_cells::family();
    for (const auto &value : sl_cells::cells(sl, census)) {
      const auto *seed = sl_cells::seed_name(sl, value);
      std::printf("sl %s %s%s%s\n", generator::id_text(generator::case_id(sl, value)).c_str(), generator::describe(sl, value).c_str(), seed ? " seed=" : "", seed ? seed : "");
    }
    const auto ngx = ngx_cells::family();
    generator::row hogwarts;
    std::string error;
    generator::parse(ngx, ngx_cells::hogwarts, hogwarts, error);
    for (const auto &value : ngx_cells::cells(ngx)) {
      std::printf("ngx %s %s%s\n", generator::id_text(generator::case_id(ngx, value)).c_str(), generator::describe(ngx, value).c_str(), value == hogwarts ? " seed=hogwarts" : "");
    }
    return 0;
  }

  int run_case(const std::string &text) {
    const auto sl = sl_cells::family();
    const auto ngx = ngx_cells::family();
    generator::row value;
    std::string error;
    bool streamline {}, found {};
    const auto colon = text.find(':');
    if (colon != std::string::npos) {
      const auto family = text.substr(0, colon);
      streamline = family == "sl";
      require(streamline || family == "ngx", "--case: the family is sl or ngx");
      if (!generator::parse(streamline ? sl : ngx, text.substr(colon + 1), value, error)) {
        std::fprintf(stderr, "--case: %s\n", error.c_str());
        return 2;
      }
      found = true;
    } else {
      sl_cells::census census;
      for (const auto &candidate : sl_cells::cells(sl, census)) {
        if (!found && generator::id_text(generator::case_id(sl, candidate)) == text) {
          value = candidate;
          streamline = found = true;
        }
      }
      for (const auto &candidate : ngx_cells::cells(ngx)) {
        if (!found && generator::id_text(generator::case_id(ngx, candidate)) == text) {
          value = candidate;
          found = true;
        }
      }
    }
    if (!found) {
      std::fprintf(stderr, "--case: no hook case has id %s (--list prints them)\n", text.c_str());
      return 2;
    }
    gaps::verdict verdict;
    if (streamline) {
      // An illegal assignment is judged as the full run judges it: it must
      // end in a refusal.
      const auto judged_case = sl_cells::judge(value);
      sl_cells::describe("case", sl, value, judged_case);
      verdict = judged_case.verdict;
      sl_cells::uninstall();
    } else {
      if (!ngx.legal(value)) {
        std::fprintf(stderr, "--case: the NGX family holds no illegal cell\n");
        return 2;
      }
      const auto judged_case = ngx_cells::judge(value);
      ngx_cells::describe("case", ngx, value, judged_case);
      verdict = judged_case.verdict;
      sunshine_ngx::shutdown();
    }
    return verdict == gaps::verdict::violates || verdict == gaps::verdict::gap_closed || verdict == gaps::verdict::misattributed || verdict == gaps::verdict::overlapping ? 1 : 0;
  }
}  // namespace

int main(int argc, char **argv) try {
  generator::options options;
  std::string error;
  if (!generator::parse_options(argc, argv, options, error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 2;
  }
  if (options.list) {
    return list_cases();
  }
  if (!options.single.empty()) {
    return run_case(options.single);
  }
  // The pure layer runs whole in an unsharded run and in shard 0.
  if (!options.partial() || (options.shards != 1 && options.shard == 0)) {
    const auto streamline = buffer_types();
    const auto state = copy_states();
    const auto formats = depth_formats();
    const auto refusals = refusal_classes();
    std::printf("PASS API contract matrix L0: %u buffer-type cells over %u release identities; ", streamline.tags.cells, streamline.identities);
    streamline.tags.print(gaps::buffer_rows);
    std::printf("; %u canonical numbers. %u copy-state cells (%u legal, %u illegal); ", streamline.canonical, state.table.cells, state.legal, state.illegal);
    state.table.print(gaps::state_rows);
    std::printf(". %u depth-format cells (%u captured, %u EITHER) meet the oracle. %u named refusal reasons, %u capture stages and %u DLSSG.Depth outcomes classify; %u meet the oracle's class for the input that produces them\n", formats.cells, formats.captured, formats.either, refusals.named, refusals.stages, refusals.outcomes, refusals.paired);
  }
  const auto hooks = hook_layer(options);
  const auto &census = hooks.census;
  std::printf("PASS API contract matrix L1%s: %u Streamline hook cells: the %u-row copy-point product (%u constructible, %u legal), %u seed rows, %u open-question rows, %u one-factor rows and %u pairwise rows covering all %zu legal value pairs; ", options.partial() ? " shard" : "", hooks.streamline.cells, census.product, census.constructible, census.legal_product, census.seeded, census.questions, census.one_factor, census.pairwise, census.pairs);
  hooks.streamline.print(gaps::sl_rows);
  std::printf(". %u NGX evaluation cells (%u of DLSS-G's own evaluation): ", hooks.ngx.cells, hooks.ngx_frame_generation);
  hooks.ngx.print(gaps::ngx_rows);
  std::printf("\n");
  return 0;
} catch (const std::exception &error) {
  std::fprintf(stderr, "FAIL API contract matrix: %s\n", error.what());
  return 1;
}
