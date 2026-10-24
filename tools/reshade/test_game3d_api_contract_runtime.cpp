// SPDX-License-Identifier: GPL-3.0-only
// The API contract matrix's runtime layer, L3 (README, API contract matrix).
// Cases run through the native provider fixture: the official ReShade
// runtime, the test add-on and a real D3D12 device, with the game's SDKs as
// the add-on's own discovery finds and hooks them (never the
// SunshineStreamlineTestCapture bypass). Each case is a frame script judged
// at its Presents against the contract oracle
// (test_game3d_api_contract_oracle.h): whether a Present shows fresh depth of
// the case's source with its extent (a UI case: whether UI detection is
// offered the case's input), and the refusal classes the test add-on records
// (SunshineDepthTestLastRefusal), never a log line.
//
// Families, each its own ctest and process:
// - sl26, sl27, sl212_214, sl215_30: a synthetic Streamline release (2.6.10,
//   without frame tags; 2.7.30; 2.14.1, the newest surveyed; 2.15.0, newer
//   than the survey, and 3.0.0, a major no header defines, each in a process
//   of its own) whose public entry points the add-on's MinHook discovery
//   hooks. sl27 runs every cell its design names: the state class x
//   lifecycle x command-list product, seed rows, each value alone and a
//   pairwise array. sl26 and sl212_214, releases of other contract tiers,
//   run their seed rows and a pairwise array; 2.15.0 and 3.0.0 their seed
//   rows, which cover what the survey cannot vouch for.
// - ngx: the NGX stand-in the executable exports. Super Resolution, Ray
//   Reconstruction and an unknown feature through Depth, and DLSS-G's own
//   evaluation (feature 11) through DLSSG.* on a viewport whose frame
//   generation the 2.7.30 interposer's options enabled.
// - generic: no API source (Streamline and NGX depth sources off); real
//   depth-stencil views drawn, cleared and unbound on the game's list.
// - ui: 2.14.1 UI tags beside a captured depth tag, and an offscreen UI
//   layer, judged by the candidates Dump3D lists for UI detection.
//
// A frame can hold two Streamline copy points: its tag call and its evaluate
// (the next frame's, for a copy after Present). Its Present shows whichever
// copied, so a case is judged at the earliest point whose outcome requires a
// capture, else at its own point, which then permits the tag call's capture
// where the tag call permits one. A capture judged at a frame-generation tag
// call is deferred when nothing was nominated inside that call (the add-on's
// nomination gate, which frame-generation inputs pass). A Generic case's
// frame holds its own boundary and no other before the end of frame. A
// generated Present (one with no SDK call of its own, as multi frame
// generation presents while the game records the next frame) may show the
// case's source only as the capture the last real Present showed: a HOLD,
// so invalidated depth never returns. A changing extent (dynamic resolution,
// R3) is one logical source: each Present shows fresh depth of the same
// allocation and identity with its own frame's extent.
//
// Isolation: every case has its own viewport and fresh resources.
// Afterwards frame generation is turned off, the depth untagged (an NGX
// feature released) and the case's viewport retired, the fixture presents
// past maximum_source_age_ms with no SDK call or draw, the case's depth must
// be gone, and SunshineGame3DTestRecalibrate resets calibration. A
// known-good control case runs first and after every ten cases and must
// capture (be offered); 3.0.0 has none, since nothing it tags may be shown.
//
// Command line: [--family <name>] [--list | --case <id|family:name=value,...> |
// --shard i/N] official.dll frozenShaders test.addon64 freshOutput
// [interposer.dll...], one interposer per release (--list needs no other
// argument). Every judged case appends one JSON line to
// <freshOutput>/contract-<family>.jsonl: {layer, family, id, seed, scope,
// dimensions, legal, rule, expected, observed, verdict, rows, evidence}. A
// family of several releases runs each in a child process and collects their
// lines there.
#define SUNSHINE_NATIVE_PROVIDER_FIXTURE_ONLY
// The fixture defines its Windows macros before any standard header.
// clang-format off
#include "test_game3d_native_provider_runtime.cpp"
// clang-format on
#include "test_contract_matrix_generator.h"
#include "test_game3d_api_contract_jsonl.h"
#include "test_game3d_api_contract_tally.h"

namespace {
  namespace matrix = game3d_contract;
  namespace generator = game3d_contract::generator;
  namespace gaps = game3d_contract::known_gaps;
  namespace judging = game3d_contract::judging;
  namespace refusals = sunshine_depth_refusal;

  // The cells a family judges: every cell its design names (whole, where a
  // known-gap row that no failing cell needs is stale), its seed rows and a
  // pairwise array (sampled), or its seed rows (seeded).
  enum class scope {
    whole,
    sampled,
    seeded
  };

  const char *scope_name(scope value) {
    constexpr const char *names[] {"whole", "sampled", "seeded"};
    return names[unsigned(value)];
  }

  struct seed {
    const char *name, *assignment;
  };

  // The seed rows a family can carry, and the name of a row that is one.
  std::vector<generator::row> parse_seeds(const generator::family &owner, const std::vector<seed> &seeds) {
    std::vector<generator::row> out;
    for (const auto &entry : seeds) {
      generator::row value;
      std::string error;
      require(generator::parse(owner, entry.assignment, value, error), ("a runtime seed row is malformed: " + error).c_str());
      if (owner.legal(value)) {
        out.push_back(value);
      }
    }
    return out;
  }

  const char *seed_name(const generator::family &owner, const generator::row &value, const std::vector<seed> &seeds) {
    for (const auto &entry : seeds) {
      generator::row parsed;
      std::string error;
      if (generator::parse(owner, entry.assignment, parsed, error) && parsed == value) {
        return entry.name;
      }
    }
    return nullptr;
  }

  struct census {
    unsigned product {}, legal_product {}, seeded {}, one_factor {}, pairwise {};
    std::size_t pairs {}, covered {}, unfilled {};
  };

  // Seed rows, then a pairwise covering array over every dimension on top
  // of them (sampled and whole scope).
  void cover_pairs(const generator::family &owner, std::vector<generator::row> &rows, const std::vector<generator::row> &seeded, census &out) {
    const auto covering = generator::pairwise(owner, seeded);
    const auto before = rows.size();
    generator::append_unique(owner, rows, covering.rows);
    out.pairwise = unsigned(rows.size() - before);
    out.pairs = covering.pairs;
    out.covered = covering.covered;
    out.unfilled = covering.unfilled;
  }

  constexpr bool requires_capture(const matrix::outcome &value) {
    return (value.kind == matrix::outcome_kind::capture || value.kind == matrix::outcome_kind::continuous) && !matrix::either(value);
  }

  constexpr bool permits_capture(const matrix::outcome &value) {
    return matrix::permits(value, matrix::outcome_kind::capture) || matrix::permits(value, matrix::outcome_kind::continuous);
  }

  // What a case's Presents must show: its outcome, whether a capture is
  // judged at a frame-generation tag call (deferred unless nominated inside
  // it), and the refusal classes a Present that shows nothing may record.
  struct expectation {
    matrix::outcome expected;
    bool at_tag_call {}, nomination {};
    matrix::refusal accepted[2] {};
    unsigned classes {};

    void accept(const matrix::outcome &outcome) {
      if (matrix::permits(outcome, matrix::outcome_kind::refuse) && outcome.why != matrix::refusal::none && classes < std::size(accepted)) {
        accepted[classes++] = outcome.why;
      }
    }

    bool accepts(matrix::refusal why) const {
      for (unsigned i = 0; i != classes; ++i) {
        if (accepted[i] == why) {
          return true;
        }
      }
      return false;
    }
  };

  // A Streamline family: one release's interposer and the cells its public
  // entry points carry. The D3D12 state shrinks to five classes (design 3.1)
  // that the game's own list puts the tagged R32 allocation
  // (ALLOW_UNORDERED_ACCESS) in before the copy point.
  namespace sl_runtime {
    struct release {
      const char *family;
      matrix::sl_version version;
      scope cells;
    };

    constexpr release releases[] {
      {"sl26", {2, 6, 10}, scope::sampled},
      {"sl27", {2, 7, 30}, scope::whole},
      {"sl212_214", {2, 14, 1}, scope::sampled},
      {"sl215", {2, 15, 0}, scope::seeded},
      {"sl30", {3, 0, 0}, scope::seeded},
    };

    enum dimension : unsigned {
      state,
      life,
      commands,
      entry,
      at,
      generated,
      fg,
      frames,
      extent,
      untag,
      tags,
      tag_result,
      evaluate_result,
      options_result
    };

    enum class state_class {
      proven_read,  // Declared NON_PIXEL_SHADER_RESOURCE; the copy's recording transitions into it.
      generic_read,  // Declared GENERIC_READ; the copy's recording transitions into it.
      write,  // Declared UNORDERED_ACCESS, the producer's own state; only its UAV barrier names the resource.
      unprovable,  // Declared UINT_MAX; only the producer's UAV barrier names the resource.
      enhanced  // Declared NON_PIXEL_SHADER_RESOURCE; an enhanced barrier on the recording sets SHADER_RESOURCE.
    };

    constexpr matrix::lifecycle lives[] {matrix::lifecycle::valid_until_present, matrix::lifecycle::only_valid_now, matrix::lifecycle::valid_until_evaluate};
    constexpr matrix::copy_point points[] {matrix::copy_point::same_frame_evaluate, matrix::copy_point::tag_call, matrix::copy_point::after_present};
    constexpr matrix::fg_mode modes[] {matrix::fg_mode::off, matrix::fg_mode::on, matrix::fg_mode::automatic, matrix::fg_mode::dynamic};
    // numFramesToGenerate: 2x, and 4x multi frame generation.
    constexpr std::uint32_t frame_counts[] {1, 3};
    constexpr matrix::extent_form extents[] {matrix::extent_form::null, matrix::extent_form::padded, matrix::extent_form::offset, matrix::extent_form::one_dimension_zero, matrix::extent_form::outside_allocation, matrix::extent_form::declared_size_differs, matrix::extent_form::changing};
    // A tag array: the depth tag alone, or beside a second tag.
    constexpr std::uint32_t tag_counts[] {1, 2};
    constexpr int tag_call_point = 1, after_present_point = 2, frame_generation_on = 1;
    static_assert(points[tag_call_point] == matrix::copy_point::tag_call && points[after_present_point] == matrix::copy_point::after_present);

    bool local(const generator::row &value) {
      return value[entry] >= int(matrix::tag_entry::local_array);
    }

    // The options the release's interposer accepts
    // (test_frame_generation_interposer.cpp): what the newest surveyed
    // release at or below it declares. It refuses the rest with eErrorIO.
    constexpr std::int32_t options_refused = 1;

    bool options_accepted(const matrix::sl_version &version, matrix::fg_mode mode, std::uint32_t count) {
      const matrix::survey::release *declared {};
      for (const auto &value : matrix::survey::releases) {
        const auto at = matrix::detail::version_of(value);
        if (!(version < at) && (!declared || matrix::detail::version_of(*declared) < at)) {
          declared = &value;
        }
      }
      return declared && std::uint32_t(mode) < declared->dlssg_modes && (mode == matrix::fg_mode::off || count <= (declared->dlssg_multi_frame ? 5u : 1u));
    }

    // The fixture sends every structure at version 1, which is the newest
    // known one where the newest surveyed release declares no later.
    constexpr matrix::struct_form first_version(std::uint32_t newest) {
      return newest > 1 ? matrix::struct_form::first : matrix::struct_form::newest_known;
    }

    matrix::sl_case to_case(const generator::row &value, const matrix::sl_version &version) {
      matrix::sl_case out;
      out.version = version;
      out.entry = matrix::tag_entry(value[entry]);
      out.life = lives[value[life]];
      out.commands = matrix::tag_commands(value[commands]);
      out.at = points[value[at]];
      out.generated_presents = matrix::generated_presents[value[generated]];
      out.extent = extents[value[extent]];
      out.tags_in_call = tag_counts[value[tags]];
      const auto &newest = matrix::newest_known();
      out.tag_header = first_version(newest.resource_tag);
      out.resource_header = first_version(newest.resource);
      out.constants = first_version(newest.constants);
      out.viewport_handle = first_version(newest.viewport_handle);
      out.frame_generation_options = first_version(newest.dlssg_options);
      out.fg = modes[value[fg]];
      out.frames_generated = frame_counts[value[frames]];
      out.untag = matrix::untag_pattern(value[untag]);
      out.tag_result = matrix::sl_result(value[tag_result]);
      out.evaluate_result = matrix::sl_result(value[evaluate_result]);
      out.options_result = matrix::sl_result(value[options_result]);
      if (out.fg != matrix::fg_mode::off && !options_accepted(version, out.fg, out.frames_generated)) {
        out.options_result = matrix::sl_result::error;  // The SDK refused the options.
      }
      out.depth = {D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
      switch (state_class(value[state])) {
        case state_class::proven_read:
          out.declared_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
          out.depth.seen = matrix::evidence::transition;
          out.depth.observed = out.declared_state;
          break;
        case state_class::generic_read:
          out.declared_state = D3D12_RESOURCE_STATE_GENERIC_READ;
          out.depth.seen = matrix::evidence::transition;
          out.depth.observed = out.declared_state;
          break;
        case state_class::write:
          out.declared_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
          break;
        case state_class::unprovable:
          out.declared_state = UINT32_MAX;
          break;
        case state_class::enhanced:
          out.declared_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
          out.depth.seen = matrix::evidence::enhanced;
          out.depth.layout = D3D12_BARRIER_LAYOUT_SHADER_RESOURCE;
          out.depth.access = D3D12_BARRIER_ACCESS_SHADER_RESOURCE;
          break;
      }
      return out;
    }

    // What the fixture can carry (partial rows): nothing comes between a tag
    // call and itself, a tag-call copy point has no evaluate, and a local
    // input has no tag call to untag or to return a result. Nor has a local
    // input a copy point after its frame's Present here: its own evaluate
    // may copy it (S4, S5), and that copy shows at the same Present a later
    // evaluate would be judged at. The hook layer carries that point (a
    // later evaluate without the input).
    bool constructible(const generator::row &value) {
      const auto assigned = [&](std::initializer_list<unsigned> dimensions) {
        for (const auto d : dimensions) {
          if (value[d] < 0) {
            return false;
          }
        }
        return true;
      };
      if (assigned({entry, at}) && local(value) && (value[at] == tag_call_point || value[at] == after_present_point)) {
        return false;
      }
      if ((assigned({at, generated}) && value[at] == tag_call_point && value[generated]) || (assigned({at, evaluate_result}) && value[at] == tag_call_point && value[evaluate_result])) {
        return false;
      }
      return !((assigned({entry, untag}) && local(value) && value[untag]) || (assigned({entry, tag_result}) && local(value) && value[tag_result]));
    }

    // Partial rows, as the hook layer's: each check runs once its dimensions
    // are assigned.
    bool legal(const generator::row &value, const matrix::sl_version &version) {
      if (!constructible(value)) {
        return false;
      }
      if (value[commands] > 0 && value[life] > 0) {
        return false;  // SLA:145, 162: only ValidUntilPresent tags may come without a command list.
      }
      if (value[entry] >= 0 && value[commands] > 0 && local(value)) {
        return false;  // An evaluate's command buffer is not optional (SLA:253).
      }
      if (value[entry] == int(matrix::tag_entry::set_tag_for_frame) && version < matrix::first_frame_tag) {
        return false;  // No slSetTagForFrame before 2.7.30.
      }
      return !generator::complete(value) || matrix::legal(to_case(value, version));
    }

    generator::family family(const release &value) {
      const auto version = value.version;
      return {
        value.family,
        {
          {"state", {"proven_read", "generic_read", "write", "unprovable", "enhanced"}},
          {"life", {"until_present", "only_valid_now", "until_evaluate"}},
          {"commands", {"game_list", "none"}},
          {"entry", {"set_tag", "set_tag_for_frame", "local_array", "local_chain_not_last"}},
          {"at", {"same_frame_evaluate", "tag_call", "after_present"}},
          {"generated", {"0", "1", "3"}},
          {"fg", {"off", "on", "automatic", "dynamic"}},
          {"frames", {"1", "3"}},
          {"extent", {"null", "padded", "offset", "one_dimension_zero", "outside_allocation", "declared_size_differs", "changing"}},
          {"untag", {"none", "null_after_each"}},
          {"tags", {"1", "2"}},
          {"tag_result", {"ok", "warn_out_of_vram", "error"}},
          {"evaluate_result", {"ok", "warn_out_of_vram", "error"}},
          {"options_result", {"ok", "warn_out_of_vram", "error"}},
        },
        [version](const generator::row &row) {
          return legal(row, version);
        },
      };
    }

    // Each dimension's names against the values to_case gives them.
    void check_sizes(const generator::family &owner) {
      require(owner.dimensions.size() == options_result + 1, "the Streamline runtime family and its dimension enumeration differ");
      const auto sized = [&](unsigned d, std::size_t count) {
        require(owner.dimensions[d].values.size() == count, "a Streamline runtime dimension's names and values differ in number");
      };
      sized(state, std::size_t(state_class::enhanced) + 1);
      sized(life, std::size(lives));
      sized(commands, std::size_t(matrix::tag_commands::none) + 1);
      sized(entry, std::size_t(matrix::tag_entry::local_chain_not_last) + 1);
      sized(at, std::size(points));
      sized(generated, std::size(matrix::generated_presents));
      sized(fg, std::size(modes));
      sized(frames, std::size(frame_counts));
      sized(extent, std::size(extents));
      sized(untag, std::size_t(matrix::untag_pattern::null_after_each) + 1);
      sized(tags, std::size(tag_counts));
      for (const auto d : {tag_result, evaluate_result, options_result}) {
        sized(d, std::size_t(matrix::sl_result::error) + 1);
      }
    }

    // Rows from games the add-on was tuned on (design 3.2), where a release
    // can carry them; the oracle still decides each one. Witcher 3's token
    // contention and RE9's aliasing barrier are other layers'; Hogwarts' NGX
    // depth is the ngx family's.
    constexpr seed game_seeds[] {
      {"shadows", "state=generic_read,commands=none,at=tag_call,fg=on,untag=null_after_each"},
      // Shadows' dynamic resolution with FG on: one source while its extent changes (R3).
      {"shadows-dynamic-resolution", "state=generic_read,fg=on,extent=changing"},
      {"bf6", "fg=on,evaluate_result=warn_out_of_vram,options_result=warn_out_of_vram"},
      {"e33", "extent=padded"},  // Expedition 33's crop of its allocation.
      // 4x multi frame generation: three generated Presents of earlier frames before the evaluate.
      {"multi-frame-generation", "fg=on,frames=3,generated=3"},
    };

    // A release outside the survey also runs each copy point and entry point
    // it has, with frame generation, and a tag array (S24).
    constexpr seed unsurveyed_seeds[] {
      {"evaluate", "at=same_frame_evaluate"},
      {"global-tag-fg", "life=only_valid_now,at=tag_call,fg=on"},
      {"frame-tag-fg", "entry=set_tag_for_frame,life=only_valid_now,at=tag_call,fg=on"},
      {"local", "entry=local_array"},
      {"dynamic-mode", "at=tag_call,fg=dynamic"},
      {"tag-array", "tags=2"},
      {"tag-array-fg", "life=only_valid_now,at=tag_call,fg=on,tags=2"},
    };

    // A cell that keeps a recorded gap's own failing cell in every release:
    // added after the pairwise array, which it leaves unchanged.
    constexpr seed gap_seeds[] {
      {"local-now-undeclared", "state=unprovable,life=only_valid_now,entry=local_array"},  // S4-refusal.
      {"local-now", "life=only_valid_now,entry=local_array"},  // S4.
      {"generated-undeclared", "state=unprovable,commands=none,generated=3"},  // S7-refusal.
      {"declared-size-undeclared", "state=unprovable,entry=local_array,extent=declared_size_differs"},  // S16d-refusal.
    };

    // The rows that seed the pairwise array, and with gaps the gap seeds.
    std::vector<seed> seeds(const release &value, bool gaps = true) {
      std::vector<seed> out(std::begin(game_seeds), std::end(game_seeds));
      if (matrix::newest_surveyed < value.version) {
        out.insert(out.end(), std::begin(unsurveyed_seeds), std::end(unsurveyed_seeds));
      }
      if (gaps) {
        out.insert(out.end(), std::begin(gap_seeds), std::end(gap_seeds));
      }
      return out;
    }

    // The known-good control: an OnlyValidNow tag copied at its
    // frame-generation tag call from a state its recording proves (S1), a
    // frame tag where the release has one. A major the add-on does not know
    // has none: nothing it tags may be shown (S24).
    const char *control(const release &value) {
      if (value.version.major != 2) {
        return nullptr;
      }
      return value.version < matrix::first_frame_tag ? "entry=set_tag,life=only_valid_now,at=tag_call,fg=on" : "entry=set_tag_for_frame,life=only_valid_now,at=tag_call,fg=on";
    }

    // whole: the three-way product of state class, lifecycle and command list
    // (design 3.2) at a same-frame evaluate and at a frame-generation tag
    // call, illegal rows included; the seed rows; each value alone; then a
    // pairwise covering array over every dimension on top of them. sampled:
    // the seed rows and a pairwise array over them. seeded: the seed rows.
    std::vector<generator::row> cells(const generator::family &owner, const release &value, census &out) {
      std::vector<generator::row> rows;
      if (value.cells == scope::whole) {
        auto tag_call = generator::defaults(owner);
        tag_call[at] = tag_call_point;
        tag_call[fg] = frame_generation_on;
        for (const auto &base : {generator::defaults(owner), tag_call}) {
          for (const auto &row : generator::product(owner, {state, life, commands}, base)) {
            ++out.product;
            out.legal_product += matrix::legal(to_case(row, value.version));
            rows.push_back(row);
          }
        }
      }
      std::vector<generator::row> seeded = rows;
      const auto count = rows.size();
      const auto parsed = parse_seeds(owner, seeds(value, false));
      generator::append_unique(owner, rows, parsed);
      seeded.insert(seeded.end(), parsed.begin(), parsed.end());
      out.seeded = unsigned(rows.size() - count);
      if (value.cells == scope::whole) {
        // Each value alone on the default row (a same-frame evaluate outside
        // frame-generation scope), so every value is judged on its own.
        const auto alone = generator::one_factor(owner, generator::defaults(owner));
        const auto before = rows.size();
        generator::append_unique(owner, rows, alone);
        out.one_factor = unsigned(rows.size() - before);
        seeded.insert(seeded.end(), alone.begin(), alone.end());
      }
      if (value.cells != scope::seeded) {
        cover_pairs(owner, rows, seeded, out);
      }
      const auto before = rows.size();
      generator::append_unique(owner, rows, parse_seeds(owner, {std::begin(gap_seeds), std::end(gap_seeds)}));
      out.seeded += unsigned(rows.size() - before);
      return rows;
    }

    // The case at the copy point that decides what its Presents show.
    struct decision {
      matrix::sl_case value;
      expectation shown;
    };

    decision at_present(const matrix::sl_case &value) {
      decision out {value, {matrix::expect(value), value.at == matrix::copy_point::tag_call}};
      if (!matrix::known_gaps::streamline::local(value.entry) && !out.shown.at_tag_call) {
        auto tag = value;
        tag.at = matrix::copy_point::tag_call;
        tag.generated_presents = 0;
        const auto at_tag = matrix::expect(tag);
        if (requires_capture(at_tag)) {
          out.value = tag;
          out.shown.expected = at_tag;
          out.shown.at_tag_call = true;
        } else {
          out.shown.accept(out.shown.expected);
          if (!requires_capture(out.shown.expected) && permits_capture(at_tag) && !permits_capture(out.shown.expected)) {
            // The tag call may copy; otherwise the case's own point decides.
            out.shown.expected = matrix::detail::either(at_tag, out.shown.expected);
            out.shown.accept(at_tag);
          }
          return out;
        }
      }
      out.shown.accept(out.shown.expected);
      out.shown.nomination = out.shown.at_tag_call && matrix::detail::frame_generation_scope(out.value);
      return out;
    }
  }  // namespace sl_runtime

  // The NGX family: the stand-in's D3D12 feature entry points on the game's
  // list. Super Resolution, Ray Reconstruction and an unknown feature name
  // their depth through Depth and the DLSS keys; DLSS-G (feature 11) through
  // DLSSG.Depth and DLSSG.DepthSubrect*. Its frame-generation scope is the
  // case's viewport, whose options the 2.7.30 interposer enabled, with
  // camera constants every frame. The allocation is the R32 UAV depth.
  namespace ngx_runtime {
    enum dimension : unsigned {
      feature,
      hw_depth,
      depth_set,
      subrect,
      result,
      state,
      lifetime
    };

    constexpr matrix::ngx_feature features[] {matrix::ngx_feature::super_resolution, matrix::ngx_feature::ray_reconstruction, matrix::ngx_feature::frame_generation, matrix::ngx_feature::unknown};
    constexpr matrix::ngx_hw_depth hw_depths[] {matrix::ngx_hw_depth::absent, matrix::ngx_hw_depth::linear, matrix::ngx_hw_depth::device, matrix::ngx_hw_depth::other};
    constexpr matrix::ngx_depth_set depth_sets[] {matrix::ngx_depth_set::d3d12_resource, matrix::ngx_depth_set::void_pointer, matrix::ngx_depth_set::missing, matrix::ngx_depth_set::null};
    constexpr matrix::ngx_subrect subrects[] {matrix::ngx_subrect::missing, matrix::ngx_subrect::zero, matrix::ngx_subrect::inside, matrix::ngx_subrect::one_side, matrix::ngx_subrect::outside};
    constexpr matrix::ngx_lifetime lifetimes[] {matrix::ngx_lifetime::steady, matrix::ngx_lifetime::release_recreate};

    // observed: the evaluation's own list transitions the written depth into
    // NON_PIXEL_SHADER_RESOURCE; unobserved: an earlier, executed list did.
    matrix::ngx_case to_case(const generator::row &value) {
      matrix::ngx_case out;
      out.feature = features[value[feature]];
      out.hw_depth = hw_depths[value[hw_depth]];
      out.depth_inverted = true;
      out.depth_set = depth_sets[value[depth_set]];
      out.subrect = subrects[value[subrect]];
      out.result = matrix::ngx_result(value[result]);
      out.lifetime = lifetimes[value[lifetime]];
      out.depth = {D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
      if (!value[state]) {
        out.depth.seen = matrix::evidence::transition;
        out.depth.observed = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
      }
      return out;
    }

    generator::family family() {
      return {
        "ngx",
        {
          {"feature", {"super_resolution", "ray_reconstruction", "frame_generation", "unknown"}},
          {"hw_depth", {"absent", "linear", "device", "other"}},
          {"depth_set", {"d3d12_resource", "void_pointer", "missing", "null"}},
          {"subrect", {"missing", "zero", "inside", "one_side", "outside"}},
          {"result", {"success", "failure"}},
          {"state", {"observed", "unobserved"}},
          {"lifetime", {"steady", "release_recreate"}},
        },
        [](const generator::row &row) {
          return !generator::complete(row) || matrix::legal(to_case(row));
        },
      };
    }

    void check_sizes(const generator::family &owner) {
      const std::size_t counts[] {std::size(features), std::size(hw_depths), std::size(depth_sets), std::size(subrects), std::size_t(matrix::ngx_result::failure) + 1, 2, std::size(lifetimes)};
      require(owner.dimensions.size() == std::size(counts), "the NGX runtime family and its dimension enumeration differ");
      for (std::size_t d = 0; d != std::size(counts); ++d) {
        require(owner.dimensions[d].values.size() == counts[d], "an NGX runtime dimension's names and values differ in number");
      }
    }

    // Hogwarts Legacy's SR depth has no state on its evaluation's list (N1's
    // contract state); Shadows supplies its frame-generation depth through
    // DLSS-G's own evaluation; a feature no guide covers and Ray
    // Reconstruction need a recorded state. The pairwise array mostly pairs
    // malformed inputs, so each rule a well-formed evaluation reaches has a
    // seed of its own as well: a recreated feature (N9), a failed evaluation
    // (N3), Depth as a void pointer (N5), linear and device encodings, a
    // crop of SR's and of DLSS-G's depth, and RR with a recorded state.
    const std::vector<seed> seeds {
      {"hogwarts", "state=unobserved"},
      {"shadows-dlssg", "feature=frame_generation"},
      {"dlssg-unrecorded", "feature=frame_generation,state=unobserved"},
      {"unknown-feature", "feature=unknown,hw_depth=device,state=unobserved"},
      {"ray-reconstruction", "feature=ray_reconstruction,hw_depth=device,state=unobserved"},
      {"recreated", "lifetime=release_recreate"},
      {"failed", "result=failure"},
      {"void-pointer", "depth_set=void_pointer"},
      {"linear", "hw_depth=linear"},
      {"device", "hw_depth=device,subrect=inside"},
      {"dlssg-crop", "feature=frame_generation,subrect=inside"},
      {"dlssg-void-pointer", "feature=frame_generation,depth_set=void_pointer"},
      {"ray-reconstruction-recorded", "feature=ray_reconstruction,hw_depth=device"},
      {"unknown-feature-recorded", "feature=unknown,hw_depth=linear,lifetime=release_recreate"},
    };

    // A Super Resolution evaluation whose list records the state (N1).
    constexpr const char *control = "state=observed";

    std::vector<generator::row> cells(const generator::family &owner, census &out) {
      std::vector<generator::row> rows;
      const auto seeded = parse_seeds(owner, seeds);
      generator::append_unique(owner, rows, seeded);
      out.seeded = unsigned(rows.size());
      cover_pairs(owner, rows, seeded, out);
      return rows;
    }
  }  // namespace ngx_runtime

  // The Generic family: one depth-stencil view of the swapchain's size on
  // the game's list, its boundary, then the end of frame. Clears clear the
  // whole view; the view stays bound to the end of frame unless the boundary
  // unbinds it, so no boundary but the case's comes first. A clear or
  // transition the add-on would copy from a state the recording does not
  // prove (D10: the end of frame after a transition away, or a read-only
  // DSV) is the device layer's: it records that copy without executing it.
  namespace generic_runtime {
    enum dimension : unsigned {
      boundary,
      cleared,
      format
    };

    constexpr matrix::generic_boundary boundaries[] {matrix::generic_boundary::clear_after_draws, matrix::generic_boundary::clear_before_draws, matrix::generic_boundary::render_pass_clear, matrix::generic_boundary::unbind_writable_dsv, matrix::generic_boundary::end_of_frame};
    constexpr matrix::clear_target clears[] {matrix::clear_target::depth_of_subresource_zero, matrix::clear_target::stencil_only};
    constexpr matrix::depth_format formats[] {matrix::depth_format::d32, matrix::depth_format::d32s8};

    matrix::generic_case to_case(const generator::row &value) {
      matrix::generic_case out;
      out.boundary = boundaries[value[boundary]];
      out.depth_written = true;
      out.depth.format = formats[value[format]];
      out.cleared = clears[value[cleared]];
      return out;
    }

    // A stencil clear needs a stencil plane, and the fixture's render pass
    // clears depth only.
    bool legal(const generator::row &value) {
      if (value[cleared] == 1 && ((value[format] == 0) || value[boundary] == 2)) {
        return false;
      }
      return !generator::complete(value) || matrix::legal(to_case(value));
    }

    generator::family family() {
      return {
        "generic",
        {
          {"boundary", {"clear_after_draws", "clear_before_draws", "render_pass_clear", "unbind_writable_dsv", "end_of_frame"}},
          {"cleared", {"depth", "stencil_only"}},
          {"format", {"d32", "d32s8"}},
        },
        [](const generator::row &row) {
          return generic_runtime::legal(row);
        },
      };
    }

    constexpr const char *control = "boundary=clear_after_draws";

    // Every legal cell.
    std::vector<generator::row> cells(const generator::family &owner, census &out) {
      std::vector<generator::row> rows;
      for (const auto &row : generator::product(owner, {boundary, cleared, format}, generator::defaults(owner))) {
        ++out.product;
        if (owner.legal(row)) {
          ++out.legal_product;
          rows.push_back(row);
        }
      }
      return rows;
    }
  }  // namespace generic_runtime

  // The UI family on the 2.14.1 interposer: each frame tags its depth
  // (ValidUntilPresent, a state its recording proves, copied at the same
  // frame's evaluate, S5) so the renderer asks for UI inputs on the case's
  // viewport, and the case's UI input: a full-size tag of its kind at a tag
  // call with the game's list, without a command list (ValidUntilPresent),
  // or as a local evaluate input; or an offscreen UI layer cleared to
  // transparent every frame (D12, legacy RENDER_TARGET). The layer cleared
  // in an enhanced layout or as a simultaneous-access target is the device
  // layer's: the add-on records legacy barriers there, which must not run.
  namespace ui_runtime {
    enum dimension : unsigned {
      kind,
      how
    };

    struct input_kind {
      const char *name;
      matrix::buffer_meaning meaning;
      std::uint32_t type;  // The 2.14.1 BufferType value.
    };

    constexpr input_kind kinds[] {
      {"ui_color_and_alpha", matrix::buffer_meaning::ui_color_and_alpha, 23},
      {"hudless", matrix::buffer_meaning::hudless_color, 2},
      {"backbuffer", matrix::buffer_meaning::backbuffer, 53},
      {"ui_alpha", matrix::buffer_meaning::ui_alpha, 69},
      {"layer", matrix::buffer_meaning::other, 0},
    };
    constexpr int layer = 4;
    constexpr matrix::ui_capture captures[] {matrix::ui_capture::tag_call, matrix::ui_capture::no_command_list, matrix::ui_capture::local};
    constexpr matrix::sl_version version {2, 14, 1};

    matrix::ui_case to_case(const generator::row &value) {
      matrix::ui_case out;
      out.version = version;
      out.how = captures[value[how]];
      if (value[kind] == layer) {
        out.clear = matrix::ui_clear_boundary::legacy_render_target;
        return out;
      }
      out.meaning = kinds[value[kind]].meaning;
      out.life = out.how == matrix::ui_capture::no_command_list ? matrix::lifecycle::valid_until_present : matrix::lifecycle::only_valid_now;
      return out;
    }

    // The layer is no tag: its one cell is the clear boundary.
    bool legal(const generator::row &value) {
      if (value[kind] == layer && value[how] > 0) {
        return false;
      }
      return !generator::complete(value) || matrix::legal(to_case(value));
    }

    generator::family family() {
      std::vector<const char *> names;
      for (const auto &value : kinds) {
        names.push_back(value.name);
      }
      return {
        "ui",
        {
          {"kind", names},
          {"how", {"tag_call", "no_command_list", "local"}},
        },
        [](const generator::row &row) {
          return ui_runtime::legal(row);
        },
      };
    }

    constexpr const char *control = "kind=ui_color_and_alpha";

    std::vector<generator::row> cells(const generator::family &owner, census &out) {
      std::vector<generator::row> rows;
      for (const auto &row : generator::product(owner, {kind, how}, generator::defaults(owner))) {
        ++out.product;
        if (owner.legal(row)) {
          ++out.legal_product;
          rows.push_back(row);
        }
      }
      return rows;
    }
  }  // namespace ui_runtime

  // A recorded reason's class in the oracle's order, through fg_scope; hold,
  // absent, internal and unclassified are outside its REFUSE classes.
  bool contract_class(refusals::reason_class value) {
    return value >= refusals::reason_class::state_unprovable && value <= refusals::reason_class::fg_scope;
  }

  const char *class_name(refusals::reason_class value) {
    constexpr const char *outside[] {"hold", "absent", "internal", "unclassified"};
    if (value <= refusals::reason_class::fg_scope) {
      return game3d_contract::jsonl::name(matrix::refusal(unsigned(value)));
    }
    const auto index = unsigned(value) - unsigned(refusals::reason_class::hold);
    return index < std::size(outside) ? outside[index] : "?";
  }

  // What one Present showed of the case's source.
  struct sample {
    bool shows {}, reused {}, frame_generation_input {};
    std::uint64_t resource {}, capture {}, identity {};
    std::uint32_t width {}, height {};
  };

  // What a case's Presents showed.
  struct presented {
    unsigned presents {}, judged {}, fresh {}, late {}, generated {}, held {};
    bool used {}, steady {}, area {true}, same_source {true}, generated_new {}, nominated_at_call {}, offered {};
    std::uint64_t first_fresh_ms {}, elapsed_ms {}, shown_capture {};
    std::vector<refusals::record> refused;  // Distinct reasons for the case, oldest first.
    sample last;
    nlohmann::json candidates = nlohmann::json::array();  // UI: the Dump3D candidates.
  };

  template<unsigned N>
  struct judged_case {
    expectation decided;
    matrix::result actual;
    presented observed;
    bool legal {}, satisfied {};
    gaps::verdict verdict {};
    bool matched[N] {};
    std::uint64_t elapsed_ms {};
  };

  // At Present: a source shown steadily (with its extent, one logical
  // source, no copy after its tag's Present) is a capture, deferred at a
  // frame-generation tag call when nothing was nominated inside it; one
  // shown otherwise is a hold. A source never shown is a recorded contract
  // class, one a copy point gives where there is one (each copy point may
  // refuse), else a hold no gate recorded.
  matrix::result result_of(const expectation &decided, const presented &value) {
    matrix::result out;
    if (value.used) {
      out.kind = value.steady && value.area && value.same_source && !value.late ? matrix::outcome_kind::capture : matrix::outcome_kind::hold;
      out.deferred = decided.nomination && !value.nominated_at_call;
      return out;
    }
    out.kind = matrix::outcome_kind::hold;
    for (const auto &record : value.refused) {
      if (contract_class(record.reason) && (out.kind == matrix::outcome_kind::hold || decided.accepts(matrix::refusal(unsigned(record.reason))))) {
        out.kind = matrix::outcome_kind::refuse;
        out.why = matrix::refusal(unsigned(record.reason));
      }
    }
    return out;
  }

  // Illegal input must show nothing. A capture must be steady, with its
  // extent, one logical source, no copy after its tag's Present and the
  // expected deferral. A Present that shows nothing meets a refusal or a
  // hold; a recorded contract class must be one a copy point gives, unless
  // a hold is permitted (a hold with no earlier capture shows nothing
  // either). No generated Present may show a capture its real Present did
  // not.
  bool satisfies(const expectation &decided, const matrix::result &actual, const presented &value, bool legal) {
    if (value.generated_new) {
      return false;
    }
    if (!legal) {
      return !value.used && !value.late;
    }
    const auto &expected = decided.expected;
    if (value.used) {
      return permits_capture(expected) && value.steady && value.area && value.same_source && !value.late && actual.deferred == expected.deferred;
    }
    if (value.late || !(matrix::permits(expected, matrix::outcome_kind::refuse) || matrix::permits(expected, matrix::outcome_kind::hold) || matrix::permits(expected, matrix::outcome_kind::ignored))) {
      return false;
    }
    return matrix::permits(expected, matrix::outcome_kind::hold) || actual.kind != matrix::outcome_kind::refuse || decided.accepts(actual.why);
  }

  // The adapter a family drives.
  enum class family_kind {
    streamline,
    ngx,
    generic,
    ui
  };

  // The contract runtime fixture: the native provider fixture with one case
  // at a time on its own viewport and resources.
  struct contract_fixture: native_provider_fixture {
    using action_t = BOOL (*)(api::effect_runtime *);
    using gate_t = void (*)(HANDLE, HANDLE);
    using results_t = void (*)(std::int32_t, std::int32_t, std::int32_t, std::int32_t, std::int32_t);
    using last_render_t = BOOL (*)(api::effect_runtime *, sunshine_game3d::test::last_render *);
    // Judged Presents a capture needs, from its first fresh one (design 2.1:
    // every real frame after warm-up has fresh depth).
    static constexpr unsigned steady_presents = 6;
    // How long a case waits for its first fresh capture; frame generation
    // owns its viewport's depth for a second after it is enabled
    // (association_timeout_ms), so an evaluate copy of an FG case waits past
    // it, and the Generic selector settles on a new view first.
    static constexpr std::uint64_t window_ms = 600, frame_generation_window_ms = 1600, generic_window_ms = 2500;
    // Isolation: past maximum_source_age_ms with no SDK call.
    static constexpr std::uint64_t isolation_ms = sunshine_scene_depth::maximum_source_age_ms + 10;
    static constexpr unsigned retired_viewports = 4;
    // UI: how long the renderer's UI request has to take the case's input
    // before Dump3D lists what detection is offered.
    static constexpr std::uint64_t ui_settle_ms = 500;

    action_t recalibrate {};
    gate_t nomination_gate {};
    results_t set_results {};
    last_render_t last_render {};
    decltype(&NVSDK_NGX_D3D12_CreateFeature) ngx_create {};
    decltype(&NVSDK_NGX_D3D12_EvaluateFeature) ngx_evaluate {};
    decltype(&NVSDK_NGX_D3D12_ReleaseFeature) ngx_release {};
    HANDLE nominated {};
    matrix::sl_version version;
    bool streamline {};  // A release's interposer is loaded.
    std::unique_ptr<uav_target> case_target;
    std::unique_ptr<target_t> depth_stencil;  // Generic.
    com_ptr<ID3D12Resource> ui_input, ui_upload;  // UI.
    com_ptr<ID3D12DescriptorHeap> ui_heap;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT ui_footprint {};
    std::uint32_t next_viewport {1000}, next_scratch {900000};
    std::function<sample()> sample_present;
    // UI inputs added to a Streamline case's frame before its evaluate.
    std::function<void(sunshine_streamline::abi_v2::frame_token &, std::vector<const sunshine_streamline::base_structure *> &)> extra_inputs;

    // The case being run.
    struct plan {
      family_kind kind {};
      matrix::sl_case value;
      sl_runtime::state_class state {};
      sunshine_streamline::abi_v2::viewport view {};
      sunshine_streamline::extent area {}, areas[3] {};
      unsigned changes {};
      std::uint32_t shown_width {}, shown_height {};
      std::int32_t tag_result {}, evaluate_result {}, options_result {};
      bool entered_layout {};  // enhanced: the allocation left legacy COMMON.
      bool frame_generation {};  // Options enabled on the viewport.
      // NGX.
      matrix::ngx_case ngx;
      ngx_fixture::parameters parameters;
      void *feature {};
      std::uint32_t feature_id {}, ngx_result {};
      unsigned ngx_frames {};
      // Generic.
      matrix::generic_case generic;
      // UI.
      matrix::ui_case ui;
      std::uint32_t ui_type {};
      D3D12_RESOURCE_STATES ui_state {};
      sunshine_streamline::abi_v2::resource ui_resource {};
      sunshine_streamline::abi_v2::resource_tag ui_tag {};
    } current;

    presented seen;
    std::uint64_t refusal_serial {};

    ~contract_fixture() {
      if (nominated) {
        CloseHandle(nominated);
      }
    }

    // The test add-on's actions and the NGX stand-in's entry points; with
    // sdk, the release's public Streamline entry points as the game holds
    // them, after discovery hooked them (hooked) or refused the release.
    void bind(HMODULE sdk, bool hooked) {
      using namespace sunshine_streamline;
      const auto addon = GetModuleHandleW(L"SunshineSBSTest.addon64");
      set_foreground = reinterpret_cast<void (*)(HWND)>(GetProcAddress(addon, "SunshineSbsTestSetForeground"));
      recalibrate = reinterpret_cast<action_t>(GetProcAddress(addon, "SunshineGame3DTestRecalibrate"));
      nomination_gate = reinterpret_cast<gate_t>(GetProcAddress(addon, "SunshineStreamlineTestNominationGate"));
      last_render = reinterpret_cast<last_render_t>(GetProcAddress(addon, "SunshineGame3DTestLastRender"));
      require(set_foreground && recalibrate && nomination_gate && last_render, "Test add-on lacks the foreground observer, recalibration, nomination gate or last-render query");
      // The export stop at a focus loss would end FG reuse: select the
      // hidden game window in the test observer without moving focus.
      set_foreground(window);
      const auto exe = GetModuleHandleW(nullptr);
      ngx_create = reinterpret_cast<decltype(ngx_create)>(GetProcAddress(exe, "NVSDK_NGX_D3D12_CreateFeature"));
      ngx_evaluate = reinterpret_cast<decltype(ngx_evaluate)>(GetProcAddress(exe, "NVSDK_NGX_D3D12_EvaluateFeature"));
      ngx_release = reinterpret_cast<decltype(ngx_release)>(GetProcAddress(exe, "NVSDK_NGX_D3D12_ReleaseFeature"));
      require(ngx_create && ngx_evaluate && ngx_release, "The contract runtime lacks its exported NGX entry points");
      nominated = CreateEventW(nullptr, TRUE, FALSE, nullptr);
      require(nominated, "Create the nomination gate event");
      streamline = sdk != nullptr;
      if (streamline) {
        const auto getter = bind_streamline(sdk);
        set_results = reinterpret_cast<results_t>(GetProcAddress(sdk, "SunshineFixtureSetResults"));
        require(getter && sl.new_token && sl.constants && sl.global_tag && sl.evaluate && set_results, "Contract interposer lacks public SL exports or settable results");
        require(!(version < matrix::first_frame_tag) == (sl.frame_tag != nullptr), "Contract interposer's frame-tag export differs from its release");
        // Discovery is deferred to a Present: only an options function fetched
        // through the hooked getter afterwards is observed. A major no header
        // defines is refused, leaving the game's calls untouched (S24).
        require(reshade_log_contains({hooked ? "Sunshine Streamline: observing 2.x" : "Sunshine Streamline: camera hooks disabled because the version identity or ABI major above is unsupported"}, 15000), hooked ? "Production discovery did not hook the contract interposer" : "Production discovery did not refuse the unknown major");
        void *function {};
        require(getter(1000, "slDLSSGSetOptions", function) == 0 && function, "Contract interposer did not expose FG options");
        sl.set_options = reinterpret_cast<set_options_t>(function);
        D3D12_FEATURE_DATA_D3D12_OPTIONS12 options {};
        require(SUCCEEDED(game->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &options, sizeof(options))) && options.EnhancedBarriersSupported, "The contract runtime's enhanced state class needs a device with enhanced barriers");
      }
      for (unsigned frame = 0; frame < 4; ++frame) {
        step();
      }
    }

    static std::int32_t sdk_result(matrix::sl_result value) {
      switch (value) {
        case matrix::sl_result::ok:
          return 0;
        case matrix::sl_result::warn_out_of_vram:
          return sunshine_streamline::abi_v2::result_warn_out_of_vram;
        default:
          return 3;  // eErrorOSOutOfDate, one of the eError* values (sl_result.h).
      }
    }

    // The global invariant: the add-on returns the SDK's result unchanged.
    static void returned(std::int32_t value, std::int32_t expected, const char *call) {
      if (value != expected) {
        throw std::runtime_error(std::string("The add-on changed the SDK result of ") + call);
      }
    }

    // A refusal for the case: one on its viewport, or any NGX refusal (an
    // NGX evaluation has no Streamline viewport).
    void collect_refusals() {
      const auto value = last_refusal();
      if (value.serial <= refusal_serial) {
        return;
      }
      refusal_serial = value.serial;
      if (value.viewport != current.view.value && !(current.kind == family_kind::ngx && value.provider == refusals::adapter::ngx)) {
        return;
      }
      for (const auto &known : seen.refused) {
        if (!std::strcmp(known.name, value.name) && known.where == value.where) {
          return;
        }
      }
      seen.refused.push_back(value);
    }

    // The API source the provider reports, as a sample of the case's
    // allocation (provider: the API it must come from).
    sample api_sample(sunshine_scene_depth::provider_kind provider) {
      const auto source = inspect().source;
      sample out;
      out.shows = source.selected && source.ready && source.provider == provider && case_target && source.current.resource == native(*case_target);
      out.reused = source.reused_depth;
      out.frame_generation_input = source.frame_generation_input;
      out.resource = source.current.resource;
      out.capture = source.current.capture;
      out.identity = source.current.identity;
      out.width = source.current.width;
      out.height = source.current.height;
      return out;
    }

    // The Generic depth the last native render consumed, as a sample of the
    // case's view: a capture is the source's last use (its frame index).
    sample generic_sample() {
      sunshine_game3d::test::last_render render;
      sample out;
      require(last_render(observed.runtime, &render), "Native Game 3D last-render observation failed");
      const auto &frame = render.depth;
      if (render.rendered && frame.ready && depth_stencil) {
        out.shows = frame.source_resource.handle == reinterpret_cast<std::uint64_t>(depth_stencil->texture.p);
        out.reused = frame.reused_depth;
        out.resource = frame.source_resource.handle;
        out.capture = frame.frame_index;
        out.identity = frame.source_id;
        out.width = frame.active_width;
        out.height = frame.active_height;
      }
      return out;
    }

    void layout_barrier(D3D12_BARRIER_SYNC sync_before, D3D12_BARRIER_ACCESS access_before, D3D12_BARRIER_LAYOUT before, D3D12_BARRIER_SYNC sync_after, D3D12_BARRIER_ACCESS access_after, D3D12_BARRIER_LAYOUT after) {
      com_ptr<ID3D12GraphicsCommandList7> list7;
      checked(commands->QueryInterface(IID_PPV_ARGS(list7.put())), "The game's list lacks enhanced barriers");
      D3D12_TEXTURE_BARRIER value {};
      value.SyncBefore = sync_before;
      value.SyncAfter = sync_after;
      value.AccessBefore = access_before;
      value.AccessAfter = access_after;
      value.LayoutBefore = before;
      value.LayoutAfter = after;
      value.pResource = case_target->resource.p;
      value.Subresources.IndexOrFirstMipLevel = UINT32_MAX;
      D3D12_BARRIER_GROUP group {};
      group.Type = D3D12_BARRIER_TYPE_TEXTURE;
      group.NumBarriers = 1;
      group.pTextureBarriers = &value;
      list7->Barrier(1, &group);
    }

    // The game's producer work on its own list: the depth written as UAV,
    // then left in the case's state class for the copy point.
    void produce() {
      using sl_runtime::state_class;
      auto &target = *case_target;
      switch (current.state) {
        case state_class::proven_read:
        case state_class::generic_read:
          {
            const auto rest = current.state == state_class::proven_read ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE : D3D12_RESOURCE_STATE_GENERIC_READ;
            if (target.state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
              transition(commands.p, target.resource.p, target.state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
            write_depth_pixels();
            transition(commands.p, target.resource.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, rest);
            target.state = rest;
            break;
          }
        case state_class::write:
        case state_class::unprovable:
          write_depth_pixels();
          break;
        case state_class::enhanced:
          // From legacy COMMON (set on an earlier executed list), as the
          // device layer enters a layout, then every barrier enhanced.
          if (current.entered_layout) {
            layout_barrier(D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, D3D12_BARRIER_SYNC_COMPUTE_SHADING, D3D12_BARRIER_ACCESS_UNORDERED_ACCESS, D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS);
          } else {
            layout_barrier(D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_COMMON, D3D12_BARRIER_LAYOUT_COMMON, D3D12_BARRIER_SYNC_COMPUTE_SHADING, D3D12_BARRIER_ACCESS_UNORDERED_ACCESS, D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS);
            current.entered_layout = true;
          }
          write_depth_pixels(false);
          layout_barrier(D3D12_BARRIER_SYNC_COMPUTE_SHADING, D3D12_BARRIER_ACCESS_UNORDERED_ACCESS, D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS, D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE);
          break;
      }
    }

    // A generated Present of an earlier frame while the game records this
    // one (multi frame generation presents on its own). It may show the
    // case's source only as the capture the last real Present showed.
    void present_generated() {
      checked(swapchain->Present(0, 0), "Generated Present between the tag and its copy point");
      ++seen.presents;
      ++seen.generated;
      const auto shown = sample_present();
      if (shown.shows) {
        if (shown.capture == seen.shown_capture) {
          ++seen.held;
        } else {
          seen.generated_new = true;
        }
      }
    }

    // Constants for the case's viewport on a fresh frame token.
    sunshine_streamline::abi_v2::frame_token &case_token() {
      sunshine_streamline::abi_v2::frame_token *token {};
      const auto frame = ++fg_frame;
      require(sl.new_token(token, &frame) == 0 && token, "Contract interposer did not supply a frame token");
      returned(sl.constants(camera_constants(false), *token, current.view), 0, "slSetConstants");
      return *token;
    }

    // One real frame of a Streamline case: the producer work, constants,
    // then the tag call (tag_frame) and the evaluate the case's copy point
    // needs. A changing extent tags its own extent every frame.
    void case_frame(bool tag_frame) {
      using namespace sunshine_streamline;
      const auto &value = current.value;
      const bool local = matrix::known_gaps::streamline::local(value.entry);
      if (value.extent == matrix::extent_form::changing && (tag_frame || local)) {
        current.area = current.areas[current.changes++ % std::size(current.areas)];
        current.shown_width = current.area.width;
        current.shown_height = current.area.height;
      }
      produce();
      auto &token = case_token();
      abi_v2::resource resource {};
      resource.base = {nullptr, resource_guid, 1};
      resource.type = 0;  // ResourceType eTex2d.
      resource.native = case_target->resource.p;
      resource.state = value.declared_state;
      const bool declared_small = value.extent == matrix::extent_form::declared_size_differs;
      resource.width = declared_small ? case_target->width / 2 : case_target->width;
      resource.height = declared_small ? case_target->height / 2 : case_target->height;
      resource.native_format = DXGI_FORMAT_R32_FLOAT;
      resource.mip_levels = resource.array_layers = 1;
      // A tag array's second tag is a motion-vector untag (a null Resource).
      abi_v2::resource_tag tags[2] {{{nullptr, tag_guid, 1}, &resource, 0, std::uint32_t(value.life), current.area}, {{nullptr, tag_guid, 1}, nullptr, 1, std::uint32_t(value.life), {}}};
      const auto count = value.tags_in_call;
      const auto command = reinterpret_cast<void *>(game_native_command);
      void *const tag_commands = value.commands == matrix::tag_commands::game_list ? command : nullptr;
      const auto tag_call = [&](const abi_v2::resource_tag *values, std::uint32_t n) {
        const auto result = value.entry == matrix::tag_entry::set_tag_for_frame ? sl.frame_tag(token, current.view, values, n, tag_commands) :
                                                                                  sl.global_tag(current.view, values, n, tag_commands);
        returned(result, current.tag_result, "the tag call");
        collect_refusals();
      };
      if (!local && tag_frame) {
        ResetEvent(nominated);
        nomination_gate(nominated, nullptr);
        tag_call(tags, count);
        nomination_gate(nullptr, nullptr);
        seen.nominated_at_call = seen.nominated_at_call || WaitForSingleObject(nominated, 0) == WAIT_OBJECT_0;
        if (value.untag == matrix::untag_pattern::null_after_each) {
          const abi_v2::resource_tag untag {{nullptr, tag_guid, 1}, nullptr, 0, std::uint32_t(value.life), {}};
          tag_call(&untag, 1);
        }
      }
      if (value.at == matrix::copy_point::tag_call || (value.at == matrix::copy_point::after_present && tag_frame)) {
        return;
      }
      if (!local) {
        for (std::uint32_t i = 0; i != value.generated_presents; ++i) {
          present_generated();
        }
      }
      std::vector<const base_structure *> inputs {&current.view.base};
      if (value.entry == matrix::tag_entry::local_array) {
        for (std::uint32_t i = 0; i != count; ++i) {
          inputs.push_back(&tags[i].base);
        }
      } else if (value.entry == matrix::tag_entry::local_chain_not_last) {
        tags[0].base.next = count > 1 ? &tags[1].base : &current.view.base;
        tags[1].base.next = &current.view.base;
        inputs = {&tags[0].base};
      }
      if (extra_inputs) {
        extra_inputs(token, inputs);
      }
      returned(sl.evaluate(0, token, inputs.data(), std::uint32_t(inputs.size()), command), current.evaluate_result, "slEvaluateFeature");
      collect_refusals();
    }

    // Frame-generation options on the case's viewport (Off with no frames).
    void set_options(matrix::fg_mode mode, std::uint32_t frames, std::int32_t expected) {
      const fg_options value {{nullptr, fg_options_type, 1}, std::uint32_t(mode), mode == matrix::fg_mode::off ? 0u : frames};
      returned(sl.set_options(current.view, value), expected, "slDLSSGSetOptions");
      current.frame_generation = mode != matrix::fg_mode::off;
    }

    // What every case starts from: its own viewport and refusal serial.
    void begin_case(family_kind kind) {
      current = {};
      current.kind = kind;
      current.view = {{nullptr, sunshine_streamline::viewport_guid, 1}, next_viewport++};
      seen = {};
      refusal_serial = last_refusal().serial;
    }

    // Warm-up of a Streamline case: the case's own viewport, allocation,
    // results and options.
    void prepare(const matrix::sl_case &value, sl_runtime::state_class state) {
      begin_case(family_kind::streamline);
      current.value = value;
      current.state = state;
      current.tag_result = sdk_result(value.tag_result);
      current.evaluate_result = sdk_result(value.evaluate_result);
      current.options_result = sdk_result(value.options_result);
      case_target = target_uav(width / 2, height / 2, 0);
      selected = case_target.get();
      sample_present = [this] {
        return api_sample(sunshine_scene_depth::provider_kind::streamline);
      };
      const auto w = case_target->width, h = case_target->height;
      switch (value.extent) {
        case matrix::extent_form::padded:
        case matrix::extent_form::declared_size_differs:  // Beyond the declared half size.
          current.area = {0, 0, w, h - 3};
          break;
        case matrix::extent_form::offset:
          current.area = {8, 16, w - 32, h - 16};
          break;
        case matrix::extent_form::one_dimension_zero:
          current.area = {0, 0, w, 0};
          break;
        case matrix::extent_form::outside_allocation:
          current.area = {0, 0, w + 64, h};
          break;
        case matrix::extent_form::changing:
          // Dynamic render sizes of the same allocation, as Shadows' with FG on.
          current.areas[0] = {0, 0, w, h - 3};
          current.areas[1] = {0, 0, w * 7 / 8, h * 7 / 8};
          current.areas[2] = {0, 0, w * 3 / 4, h * 3 / 4};
          current.area = current.areas[0];
          break;
        default:
          break;
      }
      const bool whole = !(current.area.width && current.area.height);
      current.shown_width = whole ? w : current.area.width;
      current.shown_height = whole ? h : current.area.height;
      if (state == sl_runtime::state_class::enhanced) {
        begin_commands();
        transition(commands.p, case_target->resource.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        submit(false);
        case_target->state = D3D12_RESOURCE_STATE_COMMON;
      }
      set_results(0, current.tag_result, current.evaluate_result, current.options_result, 0);
      if (value.fg != matrix::fg_mode::off) {
        const bool accepted = sl_runtime::options_accepted(version, value.fg, value.frames_generated);
        set_options(value.fg, value.frames_generated, accepted ? current.options_result : sl_runtime::options_refused);
      }
    }

    // Stimulus: real frames until the first fresh capture of the case's
    // source and steady_presents judged Presents from it, or the window.
    void observe(const std::function<void(bool)> &frame, bool alternate, std::uint64_t window) {
      const auto started = GetTickCount64();
      std::uint64_t last_capture {}, identity {};
      unsigned since_first {}, steady_fresh {};
      for (bool tag_frame = true;; tag_frame = alternate ? !tag_frame : true) {
        render_tracked_depth = [&frame, tag_frame] {
          frame(tag_frame);
        };
        present_frame();
        ++seen.presents;
        collect_refusals();
        const auto shown = sample_present();
        seen.last = shown;
        const bool fresh = shown.shows && !shown.reused && shown.capture && shown.capture != last_capture;
        const bool area = shown.width == current.shown_width && shown.height == current.shown_height;
        if (shown.shows) {
          seen.shown_capture = shown.capture;
          identity = identity ? identity : shown.identity;
          seen.same_source = seen.same_source && shown.identity == identity;
        }
        if (fresh) {
          last_capture = shown.capture;
        }
        const auto elapsed = GetTickCount64() - started;
        if (!tag_frame) {
          seen.late += fresh;  // An evaluate after its tag's Present made a copy.
        } else {
          ++seen.judged;
          seen.used = seen.used || shown.shows;
          seen.fresh += fresh;
          if (since_first) {
            ++since_first;
            steady_fresh += fresh && area;
          } else if (fresh) {
            since_first = 1;
            steady_fresh = area;
            seen.first_fresh_ms = elapsed;
          }
          if (shown.shows && !area) {
            seen.area = false;
          }
        }
        if (since_first >= steady_presents) {
          seen.steady = steady_fresh == steady_presents;
          break;
        }
        if ((!since_first && elapsed >= window) || elapsed >= window + 1000) {
          break;
        }
      }
      render_tracked_depth = {};
      seen.elapsed_ms = GetTickCount64() - started;
    }

    // Recovery: frame generation off, the depth untagged and any NGX feature
    // released, the case's viewport retired, frames with no SDK call or draw
    // past the source age; the case's depth must be gone.
    void isolate() {
      using namespace sunshine_streamline;
      render_tracked_depth = {};
      extra_inputs = {};
      ngx_fixture::results.evaluate = ngx_fixture::success;
      if (current.feature) {
        returned(std::int32_t(ngx_release(current.feature)), std::int32_t(ngx_fixture::success), "NVSDK_NGX_D3D12_ReleaseFeature");
        current.feature = nullptr;
      }
      if (streamline) {
        set_results(0, 0, 0, 0, 0);
        current.tag_result = current.evaluate_result = 0;
        if (current.frame_generation) {
          set_options(matrix::fg_mode::off, 0, 0);
        }
        const abi_v2::resource_tag untag {{nullptr, tag_guid, 1}, nullptr, 0, 1, {}};
        returned(sl.global_tag(current.view, &untag, 1, nullptr), 0, "the isolating untag");
        // FG authority is the one viewport whose options the probe holds, and
        // a viewport turned Off still holds them, so the next case's FG would
        // be ambiguous (query_frame_generation). The probe keeps the newest
        // retired_viewports viewports (streamline_camera_probe.cpp slot_count):
        // one frame's constants on as many scratch viewports retire the case's.
        render_tracked_depth = [this] {
          abi_v2::frame_token *token {};
          const auto frame = ++fg_frame;
          require(sl.new_token(token, &frame) == 0 && token, "Contract interposer did not supply a frame token");
          for (unsigned i = 0; i != retired_viewports; ++i) {
            const abi_v2::viewport scratch {{nullptr, viewport_guid, 1}, next_scratch++};
            returned(sl.constants(camera_constants(false), *token, scratch), 0, "slSetConstants");
          }
        };
      }
      present_frame();
      render_tracked_depth = {};
      const auto until = GetTickCount64() + isolation_ms;
      do {
        present_frame();
      } while (GetTickCount64() < until);
      present_frame();
      require(!sample_present().shows, "Isolation: a case's depth outlived its source age with no SDK call or draw");
      require(recalibrate(observed.runtime), "Isolation: SunshineGame3DTestRecalibrate failed");
      selected = first.get();
      case_target.reset();
      depth_stencil.reset();
      ui_input.reset();
      ui_upload.reset();
      ui_heap.reset();
    }

    // Warm-up of an NGX case: its own allocation and parameter map, and for
    // DLSS-G its own viewport's frame generation.
    void prepare_ngx(const matrix::ngx_case &value) {
      begin_case(family_kind::ngx);
      current.ngx = value;
      case_target = target_uav(width / 2, height / 2, 0);
      selected = case_target.get();
      const bool frame_generation = value.feature == matrix::ngx_feature::frame_generation;
      sample_present = [this, frame_generation] {
        return api_sample(frame_generation ? sunshine_scene_depth::provider_kind::streamline : sunshine_scene_depth::provider_kind::ngx);
      };
      const auto w = case_target->width, h = case_target->height;
      auto &p = current.parameters;
      // The keys the SR and RR create helpers (nvsdk_ngx_helpers_d3d.h,
      // nvsdk_ngx_helpers_dlssd_d3d.h) and the DLSS-G helpers
      // (nvsdk_ngx_helpers_dlssg_d3d.h) write: inverted depth, one real frame.
      if (frame_generation) {
        p.set_ui("DLSSG.DepthInverted", 1);
        p.set_ui("DLSSG.MultiFrameIndex", 1);
        p.set_ui("DLSSG.MultiFrameCount", 1);
      } else {
        p.set_ui("Width", w);
        p.set_ui("Height", h);
        p.set_i("DLSS.Feature.Create.Flags", 8);
        p.set_i("Reset", 0);
        if (value.hw_depth != matrix::ngx_hw_depth::absent) {
          p.set_ui("DLSS.Use.HW.Depth", value.hw_depth == matrix::ngx_hw_depth::linear ? 0 : value.hw_depth == matrix::ngx_hw_depth::device ? 1 :
                                                                                                                                              2);
        }
      }
      const char *depth = frame_generation ? "DLSSG.Depth" : "Depth";
      switch (value.depth_set) {
        case matrix::ngx_depth_set::d3d12_resource:
          p.set_resource(depth, case_target->resource.p);
          break;
        case matrix::ngx_depth_set::void_pointer:
          p.set_pointer(depth, case_target->resource.p);
          break;
        case matrix::ngx_depth_set::null:
          p.set_resource(depth, nullptr);
          break;
        default:
          break;
      }
      const char *base_x = frame_generation ? "DLSSG.DepthSubrectBaseX" : "DLSS.Input.Depth.Subrect.Base.X";
      const char *base_y = frame_generation ? "DLSSG.DepthSubrectBaseY" : "DLSS.Input.Depth.Subrect.Base.Y";
      const char *area_width = frame_generation ? "DLSSG.DepthSubrectWidth" : "DLSS.Render.Subrect.Dimensions.Width";
      const char *area_height = frame_generation ? "DLSSG.DepthSubrectHeight" : "DLSS.Render.Subrect.Dimensions.Height";
      const auto set_area = [&](unsigned left, unsigned top, unsigned area_w, unsigned area_h) {
        p.set_ui(base_x, left);
        p.set_ui(base_y, top);
        p.set_ui(area_width, area_w);
        p.set_ui(area_height, area_h);
      };
      current.shown_width = w;
      current.shown_height = h;
      switch (value.subrect) {
        case matrix::ngx_subrect::zero:
          set_area(0, 0, 0, 0);
          break;
        case matrix::ngx_subrect::inside:
          set_area(8, 16, w - 32, h - 16);
          current.shown_width = w - 32;
          current.shown_height = h - 16;
          break;
        case matrix::ngx_subrect::one_side:
          p.set_ui(area_width, w);
          break;
        case matrix::ngx_subrect::outside:
          set_area(0, 0, w + 64, h);
          break;
        default:
          break;
      }
      current.feature_id = std::uint32_t(value.feature);
      current.ngx_result = value.result == matrix::ngx_result::success ? ngx_fixture::success : ngx_fixture::failure;
      ngx_fixture::results.evaluate = current.ngx_result;
      if (frame_generation) {
        set_results(0, 0, 0, 0, 0);
        set_options(matrix::fg_mode::on, 1, 0);
      }
    }

    // One real frame of an NGX case: the depth written and moved into
    // NON_PIXEL_SHADER_RESOURCE (on the evaluation's list, or an earlier
    // executed one), DLSS-G's viewport constants, then the evaluation.
    void ngx_frame() {
      const auto &value = current.ngx;
      auto &target = *case_target;
      if (target.state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
        transition(commands.p, target.resource.p, target.state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      }
      write_depth_pixels();
      transition(commands.p, target.resource.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      target.state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
      if (value.depth.seen == matrix::evidence::none) {
        submit(false);
        begin_commands();
      }
      if (value.feature == matrix::ngx_feature::frame_generation) {
        case_token();
      }
      const auto list = reinterpret_cast<void *>(game_native_command);
      const auto create = [&] {
        require(ngx_create(list, current.feature_id, &current.parameters, &current.feature) == ngx_fixture::success && current.feature, "The NGX stand-in did not create the case's feature");
      };
      if (!current.feature) {
        create();
      } else if (value.lifetime == matrix::ngx_lifetime::release_recreate && current.ngx_frames == 1) {
        returned(std::int32_t(ngx_release(current.feature)), std::int32_t(ngx_fixture::success), "NVSDK_NGX_D3D12_ReleaseFeature");
        current.feature = nullptr;
        create();
      }
      ++current.ngx_frames;
      returned(std::int32_t(ngx_evaluate(list, current.feature, &current.parameters, nullptr)), std::int32_t(current.ngx_result), "NVSDK_NGX_D3D12_EvaluateFeature");
      collect_refusals();
    }

    // Warm-up of a Generic case: its own depth-stencil view of the
    // swapchain's size, drawn four times a frame.
    void prepare_generic(const matrix::generic_case &value) {
      begin_case(family_kind::generic);
      current.generic = value;
      depth_stencil = target(width, height, 4, 1, value.depth.format == matrix::depth_format::d32s8);
      current.shown_width = width;
      current.shown_height = height;
      sample_present = [this] {
        return generic_sample();
      };
    }

    // One frame of a Generic case: the view bound, drawn and cleared around
    // the case's boundary.
    void generic_frame() {
      auto &view = *depth_stencil;
      const auto dsv = view.heap->GetCPUDescriptorHandleForHeapStart();
      const auto &value = current.generic;
      const auto clear = value.cleared == matrix::clear_target::stencil_only ? D3D12_CLEAR_FLAG_STENCIL : D3D12_CLEAR_FLAG_DEPTH;
      const auto draws = [&] {
        commands->SetGraphicsRootSignature(root.p);
        commands->SetPipelineState(view.format == DXGI_FORMAT_D32_FLOAT ? depth32_pipeline.p : depth32s8_pipeline.p);
        commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        const D3D12_VIEWPORT viewport {0, 0, float(view.width), float(view.height), 0, 1};
        const D3D12_RECT scissor {0, 0, LONG(view.width), LONG(view.height)};
        commands->RSSetViewports(1, &viewport);
        commands->RSSetScissorRects(1, &scissor);
        struct {
          float w, h;
          unsigned pattern;
        } constants {float(view.width), float(view.height), view.pattern};
        commands->SetGraphicsRoot32BitConstants(0, 3, &constants, 0);
        for (unsigned i = 0; i < view.draws; ++i) {
          commands->DrawInstanced(3, 1, 0, 0);
        }
      };
      commands->OMSetRenderTargets(0, nullptr, FALSE, &dsv);
      switch (value.boundary) {
        case matrix::generic_boundary::clear_before_draws:
          commands->ClearDepthStencilView(dsv, clear, 0.f, 0, 0, nullptr);
          draws();
          break;
        case matrix::generic_boundary::clear_after_draws:
          draws();
          commands->ClearDepthStencilView(dsv, clear, 0.f, 0, 0, nullptr);
          commands->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
          break;
        case matrix::generic_boundary::render_pass_clear:
          {
            draws();
            com_ptr<ID3D12GraphicsCommandList4> list4;
            checked(commands->QueryInterface(IID_PPV_ARGS(list4.put())), "The game's list lacks render passes");
            D3D12_RENDER_PASS_DEPTH_STENCIL_DESC pass {};
            pass.cpuDescriptor = dsv;
            pass.DepthBeginningAccess.Type = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR;
            pass.DepthBeginningAccess.Clear.ClearValue.Format = view.format;
            pass.DepthEndingAccess.Type = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE;
            pass.StencilBeginningAccess.Type = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_NO_ACCESS;
            pass.StencilEndingAccess.Type = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_NO_ACCESS;
            list4->BeginRenderPass(0, nullptr, &pass, D3D12_RENDER_PASS_FLAG_NONE);
            list4->EndRenderPass();
            break;
          }
        case matrix::generic_boundary::unbind_writable_dsv:
          draws();
          commands->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
          break;
        default:  // The end of frame: still bound.
          draws();
          break;
      }
    }

    // Warm-up of a UI case: a Streamline case whose depth its own frames
    // capture, and the case's UI input.
    void prepare_ui(const matrix::ui_case &value, std::uint32_t type) {
      prepare(matrix::sl_case {ui_runtime::version}, sl_runtime::state_class::proven_read);
      current.kind = family_kind::ui;
      current.ui = value;
      current.ui_type = type;
      const auto heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
      D3D12_RESOURCE_DESC desc {};
      desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      desc.Width = width;
      desc.Height = height;
      desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
      const bool layer = value.clear != matrix::ui_clear_boundary::none;
      // UI colour and alpha as Unreal's, the HUD-less colour and Backbuffer in
      // the swapchain's format, UI alpha in one channel, the layer as RGBA.
      desc.Format = layer ? DXGI_FORMAT_R8G8B8A8_UNORM : value.meaning == matrix::buffer_meaning::ui_alpha         ? DXGI_FORMAT_R8_UNORM :
                                                       value.meaning == matrix::buffer_meaning::ui_color_and_alpha ? DXGI_FORMAT_B8G8R8A8_UNORM :
                                                                                                                     backbuffers[0]->GetDesc().Format;
      desc.Flags = layer ? D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET : D3D12_RESOURCE_FLAG_NONE;
      current.ui_state = layer ? D3D12_RESOURCE_STATE_RENDER_TARGET : D3D12_RESOURCE_STATE_COPY_DEST;
      checked(game->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, current.ui_state, nullptr, IID_PPV_ARGS(ui_input.put())), "Create the case's UI input");
      // UI over a fifth of the frame: translucent marks on transparent black.
      const auto pixel_bytes = bytes_per_pixel(desc.Format);
      std::vector<std::uint8_t> pattern(std::size_t(width) * height * pixel_bytes);
      for (unsigned y = height / 8; y < height / 4; ++y) {
        for (unsigned x = width / 8; x < width / 4; ++x) {
          auto *pixel = pattern.data() + (std::size_t(y) * width + x) * pixel_bytes;
          for (unsigned c = 0; c != pixel_bytes; ++c) {
            pixel[c] = std::uint8_t(c + 1 == pixel_bytes ? 160 : (x * 5 + y * (c + 1)) & 127);
          }
        }
      }
      fill_upload(ui_upload, desc, pattern.data(), ui_footprint);
      if (layer) {
        D3D12_DESCRIPTOR_HEAP_DESC heap_desc {};
        heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        heap_desc.NumDescriptors = 1;
        checked(game->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(ui_heap.put())), "Create the UI layer's RTV heap");
        game->CreateRenderTargetView(ui_input.p, nullptr, ui_heap->GetCPUDescriptorHandleForHeapStart());
      }
      auto &resource = current.ui_resource;
      resource.base = {nullptr, sunshine_streamline::resource_guid, 1};
      resource.native = ui_input.p;
      resource.state = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
      resource.width = width;
      resource.height = height;
      resource.native_format = desc.Format;
      resource.mip_levels = resource.array_layers = 1;
      current.ui_tag = {{nullptr, sunshine_streamline::tag_guid, 1}, &current.ui_resource, type, std::uint32_t(value.life), {0, 0, width, height}};
      extra_inputs = [this](sunshine_streamline::abi_v2::frame_token &, std::vector<const sunshine_streamline::base_structure *> &inputs) {
        ui_frame(inputs);
      };
    }

    // The case's UI input in the frame: the layer cleared and drawn, or the
    // tagged input written, put in its declared state on the game's list and
    // tagged.
    void ui_frame(std::vector<const sunshine_streamline::base_structure *> &inputs) {
      const auto &value = current.ui;
      D3D12_TEXTURE_COPY_LOCATION from {}, to {};
      from.pResource = ui_upload.p;
      from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
      from.PlacedFootprint = ui_footprint;
      to.pResource = ui_input.p;
      to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      if (value.clear != matrix::ui_clear_boundary::none) {
        // Through ReShade's command-list proxy, so its clear event fires.
        const float transparent[4] {};
        commands->ClearRenderTargetView(ui_heap->GetCPUDescriptorHandleForHeapStart(), transparent, 0, nullptr);
        transition(commands.p, ui_input.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST);
        commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        transition(commands.p, ui_input.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET);
        return;
      }
      if (current.ui_state != D3D12_RESOURCE_STATE_COPY_DEST) {
        transition(commands.p, ui_input.p, current.ui_state, D3D12_RESOURCE_STATE_COPY_DEST);
      }
      commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
      transition(commands.p, ui_input.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
      current.ui_state = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
      if (value.how == matrix::ui_capture::local) {
        inputs.push_back(&current.ui_tag.base);
        return;
      }
      void *const list = value.how == matrix::ui_capture::tag_call ? reinterpret_cast<void *>(game_native_command) : nullptr;
      returned(sl.global_tag(current.view, &current.ui_tag, 1, list), 0, "the UI tag call");
    }

    // Whether Dump3D lists the case's input among the candidates UI
    // detection is offered: its tag by native resource, or the live layer.
    void judge_ui_candidates(const nlohmann::json &metadata) {
      const auto native_input = reinterpret_cast<std::uint64_t>(ui_input.p);
      const bool layer = current.ui.clear != matrix::ui_clear_boundary::none;
      seen.candidates = metadata.at("ui_source").value("candidates", nlohmann::json::array());
      for (const auto &candidate : seen.candidates) {
        const bool own = layer ? candidate.value("source", "") == "ui_layer" : candidate.value("source_native", std::uint64_t {}) == native_input;
        seen.offered = seen.offered || (own && candidate.value("available_for_detection", false));
      }
    }

    template<unsigned N, class Case>
    void judge_rows(judged_case<N> &out, const gaps::row<Case> (&rows)[N], const Case &value) {
      out.observed = seen;
      out.actual = result_of(out.decided, out.observed);
      out.satisfied = satisfies(out.decided, out.actual, out.observed, out.legal);
      out.verdict = gaps::judge_each(rows, value, out.satisfied, {out.actual}, out.matched);
    }

    static constexpr unsigned sl_rows = std::size(gaps::sl_runtime_rows), ngx_rows = std::size(gaps::ngx_runtime_rows), generic_rows = std::size(gaps::generic_runtime_rows), ui_rows = std::size(gaps::ui_runtime_rows);

    judged_case<sl_rows> judge_streamline(const generator::family &owner, const generator::row &assignment, const char *name) {
      judged_case<sl_rows> out;
      const auto value = sl_runtime::to_case(assignment, version);
      out.legal = owner.legal(assignment);
      const auto decided = sl_runtime::at_present(value);
      out.decided = decided.shown;
      require(out.decided.expected.by != nullptr, "the oracle named no rule for a Streamline runtime cell");
      const auto state = sl_runtime::state_class(assignment[sl_runtime::state]);
      // A copy at an evaluate waits past frame generation's first second.
      const bool frame_generation_wait = value.fg != matrix::fg_mode::off && value.at != matrix::copy_point::tag_call;
      const bool alternate = value.at == matrix::copy_point::after_present && !matrix::known_gaps::streamline::local(value.entry);
      const auto started = GetTickCount64();
      run_script({name, 8000, [&] {
                    prepare(value, state);
                  },
                  [&] {
                    observe(
                      [this](bool tag_frame) {
                        case_frame(tag_frame);
                      },
                      alternate,
                      frame_generation_wait ? frame_generation_window_ms : window_ms
                    );
                  },
                  [&] {
                    isolate();
                  }});
      out.elapsed_ms = GetTickCount64() - started;
      out.observed = seen;
      out.actual = result_of(out.decided, out.observed);
      out.satisfied = satisfies(out.decided, out.actual, out.observed, out.legal);
      out.verdict = gaps::judge_each(gaps::sl_runtime_rows, gaps::present_case {value, decided.value}, out.satisfied, {out.actual}, out.matched);
      return out;
    }

    judged_case<ngx_rows> judge_ngx(const generator::family &owner, const generator::row &assignment, const char *name) {
      judged_case<ngx_rows> out;
      const auto value = ngx_runtime::to_case(assignment);
      out.legal = owner.legal(assignment);
      out.decided.expected = matrix::expect(value);
      require(out.decided.expected.by != nullptr, "the oracle named no rule for an NGX runtime cell");
      out.decided.accept(out.decided.expected);
      const bool frame_generation = value.feature == matrix::ngx_feature::frame_generation;
      const auto started = GetTickCount64();
      run_script({name, 8000, [&] {
                    prepare_ngx(value);
                  },
                  [&] {
                    observe(
                      [this](bool) {
                        ngx_frame();
                      },
                      false,
                      frame_generation ? frame_generation_window_ms : window_ms
                    );
                  },
                  [&] {
                    isolate();
                  }});
      out.elapsed_ms = GetTickCount64() - started;
      judge_rows(out, gaps::ngx_runtime_rows, value);
      return out;
    }

    judged_case<generic_rows> judge_generic(const generator::family &owner, const generator::row &assignment, const char *name) {
      judged_case<generic_rows> out;
      const auto value = generic_runtime::to_case(assignment);
      out.legal = owner.legal(assignment);
      out.decided.expected = matrix::expect(value);
      require(out.decided.expected.by != nullptr, "the oracle named no rule for a Generic runtime cell");
      out.decided.accept(out.decided.expected);
      const auto started = GetTickCount64();
      run_script({name, 10000, [&] {
                    prepare_generic(value);
                  },
                  [&] {
                    observe(
                      [this](bool) {
                        generic_frame();
                      },
                      false,
                      generic_window_ms
                    );
                  },
                  [&] {
                    isolate();
                  }});
      out.elapsed_ms = GetTickCount64() - started;
      judge_rows(out, gaps::generic_runtime_rows, value);
      return out;
    }

    // A UI case: its depth must capture (the renderer asks for UI inputs of
    // a viewport whose depth it renders), then the request has ui_settle_ms
    // to take the input before Dump3D lists the candidates.
    judged_case<ui_rows> judge_ui(const generator::family &owner, const generator::row &assignment, const char *name) {
      judged_case<ui_rows> out;
      const auto value = ui_runtime::to_case(assignment);
      out.legal = owner.legal(assignment);
      out.decided.expected = matrix::expect(value);
      require(out.decided.expected.by != nullptr, "the oracle named no rule for a UI runtime cell");
      const auto started = GetTickCount64();
      run_script({name, 20000, [&] {
                    prepare_ui(value, ui_runtime::kinds[assignment[ui_runtime::kind]].type);
                  },
                  [&] {
                    const auto frame = [this](bool tag_frame) {
                      case_frame(tag_frame);
                    };
                    observe(frame, false, window_ms);
                    require(seen.used && seen.steady, "A UI case's depth did not capture, so no UI request names its viewport");
                    render_tracked_depth = [&frame] {
                      frame(true);
                    };
                    for (const auto until = GetTickCount64() + ui_settle_ms; GetTickCount64() < until;) {
                      present_frame();
                    }
                    dump_mailbox box(*this, 0x5549434f4e54ULL);
                    judge_ui_candidates(box.request("Production Dump3D did not list the UI case's candidates"));
                    render_tracked_depth = {};
                  },
                  [&] {
                    isolate();
                  }});
      out.elapsed_ms = GetTickCount64() - started;
      out.observed = seen;
      out.actual.kind = seen.offered ? matrix::outcome_kind::offered : matrix::outcome_kind::ignored;
      out.satisfied = matrix::permits(out.decided.expected, out.actual.kind);
      out.verdict = gaps::judge_each(gaps::ui_runtime_rows, value, out.satisfied, {out.actual}, out.matched);
      return out;
    }
  };

  template<unsigned N, class Case>
  void describe(const char *what, const generator::family &owner, const generator::row &assignment, const judged_case<N> &value, const gaps::row<Case> (&rows)[N]) {
    const auto &seen = value.observed;
    std::string names;
    const char *ids[N] {};
    for (unsigned i = 0; i != N; ++i) {
      ids[i] = rows[i].id;
    }
    std::fprintf(stderr, "%s: case %s %s %s%s\n", what, owner.name, generator::id_text(generator::case_id(owner, assignment)).c_str(), generator::describe(owner, assignment).c_str(), value.legal ? "" : " (illegal input)");
    judging::describe_expected(value.decided.expected);
    std::fprintf(stderr, "  decided at %s; Presents %u (judged %u, fresh %u, after-Present copies %u, generated %u held %u new=%u); used=%u steady=%u area=%u same_source=%u nominated_at_call=%u offered=%u first_fresh_ms=%llu; shown resource=0x%llx %ux%u reused=%u fg_input=%u\n", value.decided.at_tag_call ? "the tag call" : "the case's copy point", seen.presents, seen.judged, seen.fresh, seen.late, seen.generated, seen.held, seen.generated_new, seen.used, seen.steady, seen.area, seen.same_source, seen.nominated_at_call, seen.offered, static_cast<unsigned long long>(seen.first_fresh_ms), static_cast<unsigned long long>(seen.last.resource), seen.last.width, seen.last.height, seen.last.reused, seen.last.frame_generation_input);
    for (const auto &record : seen.refused) {
      std::fprintf(stderr, "  refused: %s (%s) adapter=%u gate=%u stage=%u status=%u fg=%u\n", record.name, class_name(record.reason), unsigned(record.provider), unsigned(record.where), unsigned(record.stage), unsigned(record.status), record.frame_generation);
    }
    if (!seen.candidates.empty()) {
      std::fprintf(stderr, "  UI candidates: %s\n", seen.candidates.dump().c_str());
    }
    std::fprintf(stderr, "  result: %s why=%s deferred=%u; rows %s; %s\n", judging::name(value.actual.kind), game3d_contract::jsonl::name(value.actual.why), value.actual.deferred, judging::row_names(value.matched, ids, N, names), judging::name(value.verdict));
  }

  template<unsigned N, class Case>
  nlohmann::json json_line(const generator::family &owner, const generator::row &assignment, const judged_case<N> &value, const gaps::row<Case> (&rows)[N], const char *seed, scope cells) {
    const auto &seen = value.observed;
    auto line = game3d_contract::jsonl::line("L3", owner, assignment, seed, value.legal, value.decided.expected, value.actual, value.verdict, rows, value.matched);
    line["scope"] = scope_name(cells);
    nlohmann::json refused = nlohmann::json::array();
    for (const auto &record : seen.refused) {
      refused.push_back({{"name", record.name}, {"class", class_name(record.reason)}, {"adapter", unsigned(record.provider)}, {"gate", unsigned(record.where)}, {"stage", unsigned(record.stage)}, {"status", unsigned(record.status)}});
    }
    auto &observed = line["observed"];
    observed["used"] = seen.used;
    observed["steady"] = seen.steady;
    observed["area"] = seen.area;
    observed["same_source"] = seen.same_source;
    observed["late_copies"] = seen.late;
    observed["generated_new"] = seen.generated_new;
    observed["offered"] = seen.offered;
    observed["refused"] = refused;
    line["evidence"] = {{"elapsed_ms", value.elapsed_ms}, {"presents", seen.presents}, {"judged", seen.judged}, {"fresh", seen.fresh}, {"generated", seen.generated}, {"held", seen.held}, {"first_fresh_ms", seen.first_fresh_ms}, {"nominated_at_call", seen.nominated_at_call}, {"shown", {{"width", seen.last.width}, {"height", seen.last.height}, {"reused", seen.last.reused}, {"fg_input", seen.last.frame_generation_input}}}};
    if (!seen.candidates.empty()) {
      line["evidence"]["ui_candidates"] = seen.candidates;
    }
    return line;
  }

  // The cells options select: one case by id or assignment, a shard, or all
  // (a whole run requires its pairwise array to cover every legal pair).
  bool select_cells(const generator::family &owner, const std::vector<generator::row> &all, const census &counts, const generator::options &options, const std::function<bool(const generator::row &)> &constructible, std::vector<generator::row> &rows, std::vector<std::size_t> &positions) {
    if (!options.single.empty()) {
      generator::row value;
      std::string error;
      const auto colon = options.single.find(':');
      if (colon != std::string::npos) {
        require(options.single.substr(0, colon) == owner.name, "--case: the family differs from --family");
        if (!generator::parse(owner, options.single.substr(colon + 1), value, error)) {
          std::fprintf(stderr, "--case: %s\n", error.c_str());
          return false;
        }
        require(!constructible || constructible(value), "--case: the fixture cannot carry that assignment");
      } else {
        bool found {};
        for (const auto &candidate : all) {
          if (!found && generator::id_text(generator::case_id(owner, candidate)) == options.single) {
            value = candidate;
            found = true;
          }
        }
        if (!found) {
          std::fprintf(stderr, "--case: no %s case has id %s (--list prints them)\n", owner.name, options.single.c_str());
          return false;
        }
      }
      rows.push_back(value);
      positions.push_back(0);
      return true;
    }
    require(counts.covered == counts.pairs && !counts.unfilled, "the runtime covering array left a legal pair uncovered");
    for (std::size_t i = 0; i != all.size(); ++i) {
      if (options.selected(i)) {
        rows.push_back(all[i]);
        positions.push_back(i);
      }
    }
    return true;
  }

  // Runs a family's selected cells, a known-good control first and after
  // every ten (none where control is null), one JSON line each, and checks
  // their verdicts against the family's known-gap rows.
  template<unsigned N, class Case, class Judge>
  void run_cells(const generator::family &owner, const std::vector<generator::row> &rows, const std::vector<std::size_t> &positions, const generator::options &options, scope cells, const std::vector<seed> &seeds, const char *control, const gaps::row<Case> (&gap_rows)[N], std::ofstream &lines, Judge judge, const std::function<bool(const judged_case<N> &)> &controlled, judging::each_tally<N> &tally) {
    unsigned controls {};
    std::uint64_t total_ms {}, slowest_ms {};
    generator::row control_row;
    if (control) {
      std::string error;
      require(generator::parse(owner, control, control_row, error), "the runtime control row is malformed");
    }
    const auto run_control = [&](std::size_t before) {
      if (!control) {
        return;
      }
      const auto name = std::string(owner.name) + "-control-" + std::to_string(controls++);
      const auto value = judge(owner, control_row, name.c_str());
      if (value.verdict != gaps::verdict::meets || !controlled(value)) {
        describe("control failed", owner, control_row, value, gap_rows);
        throw std::runtime_error("the known-good control failed before case " + std::to_string(before) + ": isolation lost");
      }
    };
    for (std::size_t i = 0; i != rows.size(); ++i) {
      if (i % 10 == 0) {
        run_control(positions[i]);
      }
      const auto id = generator::id_text(generator::case_id(owner, rows[i]));
      const auto name = std::string(owner.name) + '-' + id;
      const auto value = judge(owner, rows[i], name.c_str());
      total_ms += value.elapsed_ms;
      slowest_ms = std::max(slowest_ms, value.elapsed_ms);
      std::printf("MEASURE contract-runtime family=%s case=%s elapsed_ms=%llu verdict=%s\n", owner.name, id.c_str(), static_cast<unsigned long long>(value.elapsed_ms), game3d_contract::jsonl::name(value.verdict));
      lines << json_line(owner, rows[i], value, gap_rows, seed_name(owner, rows[i], seeds), cells).dump() << '\n';
      const auto *what = tally.add(value.verdict, value.matched, value.satisfied);
      if (what || !options.single.empty()) {
        describe(what ? what : "case", owner, rows[i], value, gap_rows);
      }
    }
    run_control(rows.size());
    require(lines.good(), "Could not write the contract runtime's JSON lines");
    std::printf("MEASURE contract-runtime family=%s cases=%zu controls=%u mean_ms=%llu max_ms=%llu total_s=%.1f estimate_ms=1300\n", owner.name, rows.size(), controls, static_cast<unsigned long long>(rows.empty() ? 0 : total_ms / rows.size()), static_cast<unsigned long long>(slowest_ms), double(total_ms) / 1000.);
    // Only a whole family shows a row stale: the others sample the cells.
    tally.finish(gap_rows, owner.name, !options.partial() && cells == scope::whole);
  }

  const sl_runtime::release *find_release(const std::string &name) {
    for (const auto &value : sl_runtime::releases) {
      if (name == value.family) {
        return &value;
      }
    }
    return nullptr;
  }

  // A family of several releases: each runs in its own process, since
  // discovery hooks the one sl.interposer.dll a process holds.
  struct group {
    const char *family;
    const char *members[2];
  };

  constexpr group groups[] {{"sl215_30", {"sl215", "sl30"}}};

  const group *find_group(const std::string &name) {
    for (const auto &value : groups) {
      if (name == value.family) {
        return &value;
      }
    }
    return nullptr;
  }

  // The families this executable runs in one process, and the interposers
  // each needs (one per release; none for generic).
  bool find_family(const std::string &name, family_kind &kind) {
    if (find_release(name)) {
      kind = family_kind::streamline;
    } else if (name == "ngx") {
      kind = family_kind::ngx;
    } else if (name == "generic") {
      kind = family_kind::generic;
    } else if (name == "ui") {
      kind = family_kind::ui;
    } else {
      return false;
    }
    return true;
  }

  unsigned interposers(const std::string &name) {
    if (const auto *value = find_group(name)) {
      return unsigned(std::size(value->members));
    }
    family_kind kind {};
    return find_family(name, kind) && kind == family_kind::generic ? 0u : 1u;
  }

  void print_cells(const generator::family &owner, const std::vector<generator::row> &rows, const std::vector<seed> &seeds, const std::function<std::pair<matrix::outcome, bool>(const generator::row &)> &decide) {
    for (const auto &value : rows) {
      const auto decided = decide(value);
      const auto &expected = decided.first;
      const auto *seed = seed_name(owner, value, seeds);
      std::printf("%s %s %s%s expect=%s%s%s rule=%s at=%s%s%s\n", owner.name, generator::id_text(generator::case_id(owner, value)).c_str(), generator::describe(owner, value).c_str(), owner.legal(value) ? "" : " (illegal)", judging::name(expected.kind), matrix::either(expected) ? "|" : "", matrix::either(expected) ? judging::name(expected.alternative) : "", expected.by ? expected.by->id : "?", decided.second ? "tag_call" : "own", seed ? " seed=" : "", seed ? seed : "");
    }
  }

  void print_census(const char *family, std::size_t cells, const census &value, scope kind) {
    std::printf("%s: %zu cells (%s): %u-row product (%u legal), %u seed rows, %u one-factor rows, %u pairwise rows covering %zu of %zu legal pairs (%zu unfilled)\n", family, cells, scope_name(kind), value.product, value.legal_product, value.seeded, value.one_factor, value.pairwise, value.covered, value.pairs, value.unfilled);
  }

  // A family's cells, as list and run take them.
  struct family_cells {
    generator::family owner;
    std::vector<generator::row> rows;
    census counts;
    scope cells {scope::whole};
    std::vector<seed> seeds;
    const char *control {};
  };

  family_cells cells_of(const std::string &name) {
    family_kind kind {};
    require(find_family(name, kind), "unknown family");
    switch (kind) {
      case family_kind::streamline:
        {
          const auto &release = *find_release(name);
          family_cells out {sl_runtime::family(release)};
          sl_runtime::check_sizes(out.owner);
          out.rows = sl_runtime::cells(out.owner, release, out.counts);
          out.cells = release.cells;
          out.seeds = sl_runtime::seeds(release);
          out.control = sl_runtime::control(release);
          return out;
        }
      case family_kind::ngx:
        {
          family_cells out {ngx_runtime::family()};
          ngx_runtime::check_sizes(out.owner);
          out.rows = ngx_runtime::cells(out.owner, out.counts);
          out.seeds = ngx_runtime::seeds;
          out.control = ngx_runtime::control;
          return out;
        }
      case family_kind::generic:
        {
          family_cells out {generic_runtime::family()};
          out.rows = generic_runtime::cells(out.owner, out.counts);
          out.control = generic_runtime::control;
          return out;
        }
      default:
        {
          family_cells out {ui_runtime::family()};
          out.rows = ui_runtime::cells(out.owner, out.counts);
          out.control = ui_runtime::control;
          return out;
        }
    }
  }

  int list_cases(const std::string &name) {
    if (const auto *value = find_group(name)) {
      for (const auto *member : value->members) {
        list_cases(member);
      }
      return 0;
    }
    family_kind kind {};
    find_family(name, kind);
    const auto family = cells_of(name);
    switch (kind) {
      case family_kind::streamline:
        {
          const auto version = find_release(name)->version;
          print_cells(family.owner, family.rows, family.seeds, [&](const generator::row &value) {
            const auto decided = sl_runtime::at_present(sl_runtime::to_case(value, version));
            return std::make_pair(decided.shown.expected, decided.shown.at_tag_call);
          });
          break;
        }
      case family_kind::ngx:
        print_cells(family.owner, family.rows, family.seeds, [](const generator::row &value) {
          return std::make_pair(matrix::expect(ngx_runtime::to_case(value)), false);
        });
        break;
      case family_kind::generic:
        print_cells(family.owner, family.rows, family.seeds, [](const generator::row &value) {
          return std::make_pair(matrix::expect(generic_runtime::to_case(value)), false);
        });
        break;
      default:
        print_cells(family.owner, family.rows, family.seeds, [](const generator::row &value) {
          return std::make_pair(matrix::expect(ui_runtime::to_case(value)), false);
        });
        break;
    }
    print_census(family.owner.name, family.rows.size(), family.counts, family.cells);
    return 0;
  }

  // The cells selected by options, each judged after the control as the
  // design's isolation requires.
  int run_family(contract_fixture &fixture, const std::string &name, const generator::options &options, const fs::path &jsonl) {
    family_kind kind {};
    find_family(name, kind);
    const auto family = cells_of(name);
    const auto &owner = family.owner;
    std::vector<generator::row> rows;
    std::vector<std::size_t> positions;
    const std::function<bool(const generator::row &)> constructible = kind == family_kind::streamline ? std::function<bool(const generator::row &)>(sl_runtime::constructible) : std::function<bool(const generator::row &)>(owner.legal);
    if (!select_cells(owner, family.rows, family.counts, options, constructible, rows, positions)) {
      return 2;
    }
    std::ofstream lines(jsonl, std::ios::binary);
    require(lines.good(), "Could not open the contract runtime's JSON lines");
    const auto captures = [](const auto &value) {
      return value.actual.kind == matrix::outcome_kind::capture;
    };
    const auto summary = [&](auto &tally, const auto &gap_rows, const char *what) {
      std::printf("PASS API contract matrix L3 %s%s: %zu cells (%s) %s, judged at Present; ", owner.name, options.partial() ? " (partial)" : "", rows.size(), scope_name(family.cells), what);
      tally.print(gap_rows);
      std::printf("\n");
    };
    switch (kind) {
      case family_kind::streamline:
        {
          judging::each_tally<contract_fixture::sl_rows> tally;
          const auto judge = [&](const generator::family &family_owner, const generator::row &value, const char *case_name) {
            return fixture.judge_streamline(family_owner, value, case_name);
          };
          run_cells(owner, rows, positions, options, family.cells, family.seeds, family.control, gaps::sl_runtime_rows, lines, judge, std::function<bool(const judged_case<contract_fixture::sl_rows> &)>(captures), tally);
          char what[160] {};
          std::snprintf(what, sizeof(what), "through the %u.%u.%u interposer and the production hooks", fixture.version.major, fixture.version.minor, fixture.version.patch);
          summary(tally, gaps::sl_runtime_rows, what);
          break;
        }
      case family_kind::ngx:
        {
          judging::each_tally<contract_fixture::ngx_rows> tally;
          const auto judge = [&](const generator::family &family_owner, const generator::row &value, const char *case_name) {
            return fixture.judge_ngx(family_owner, value, case_name);
          };
          run_cells(owner, rows, positions, options, family.cells, family.seeds, family.control, gaps::ngx_runtime_rows, lines, judge, std::function<bool(const judged_case<contract_fixture::ngx_rows> &)>(captures), tally);
          summary(tally, gaps::ngx_runtime_rows, "through the NGX stand-in and the production NGX hooks");
          break;
        }
      case family_kind::generic:
        {
          judging::each_tally<contract_fixture::generic_rows> tally;
          const auto judge = [&](const generator::family &family_owner, const generator::row &value, const char *case_name) {
            return fixture.judge_generic(family_owner, value, case_name);
          };
          run_cells(owner, rows, positions, options, family.cells, family.seeds, family.control, gaps::generic_runtime_rows, lines, judge, std::function<bool(const judged_case<contract_fixture::generic_rows> &)>(captures), tally);
          summary(tally, gaps::generic_runtime_rows, "on real depth-stencil views with no API source");
          break;
        }
      default:
        {
          judging::each_tally<contract_fixture::ui_rows> tally;
          const auto judge = [&](const generator::family &family_owner, const generator::row &value, const char *case_name) {
            return fixture.judge_ui(family_owner, value, case_name);
          };
          const auto offered = [](const judged_case<contract_fixture::ui_rows> &value) {
            return value.actual.kind == matrix::outcome_kind::offered;
          };
          run_cells(owner, rows, positions, options, family.cells, family.seeds, family.control, gaps::ui_runtime_rows, lines, judge, std::function<bool(const judged_case<contract_fixture::ui_rows> &)>(offered), tally);
          summary(tally, gaps::ui_runtime_rows, "through the 2.14.1 interposer, judged by Dump3D's UI candidates");
          break;
        }
    }
    return 0;
  }

  std::wstring widen(const std::string &value) {
    return std::wstring(value.begin(), value.end());
  }

  // A child process of a family of several releases: this executable with
  // the member family, the same options and that release's interposer; its
  // output goes to the parent's console.
  int run_member(const std::wstring &arguments) {
    wchar_t executable[32768] {};
    require(GetModuleFileNameW(nullptr, executable, 32768) != 0, "Cannot name the contract runtime executable");
    auto command = L"\"" + std::wstring(executable) + L"\" " + arguments;
    STARTUPINFOW startup {};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION process {};
    require(CreateProcessW(executable, command.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &startup, &process), "Cannot start a release's contract runtime process");
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD code {};
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return int(code);
  }

  int run_group(const group &value, const std::vector<std::string> &flags, const std::vector<std::string> &positional) {
    const auto directory = fs::absolute(positional[3]);
    require(!fs::exists(directory), "The contract runtime requires a fresh isolated output directory");
    fs::create_directories(directory);
    std::ofstream lines(directory / ("contract-" + std::string(value.family) + ".jsonl"), std::ios::binary);
    int result {};
    for (std::size_t i = 0; i != std::size(value.members); ++i) {
      const std::string member = value.members[i];
      std::wstring arguments = L"--family " + widen(member);
      for (const auto &flag : flags) {
        arguments += L" \"" + widen(flag) + L"\"";
      }
      for (std::size_t p = 0; p != 3; ++p) {
        arguments += L" \"" + fs::absolute(positional[p]).wstring() + L"\"";
      }
      const auto output = directory / member;
      arguments += L" \"" + output.wstring() + L"\" \"" + fs::absolute(positional[4 + i]).wstring() + L"\"";
      const auto code = run_member(arguments);
      std::printf("MEASURE contract-runtime group=%s member=%s exit=%d\n", value.family, member.c_str(), code);
      result = result ? result : code;
      std::ifstream member_lines(output / ("contract-" + member + ".jsonl"), std::ios::binary);
      lines << member_lines.rdbuf();
    }
    require(lines.good(), "Could not collect the releases' JSON lines");
    return result;
  }
}  // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::vector<char *> flags {argv[0]};
  std::vector<std::string> forwarded, positional;
  std::string family = "sl27";
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--family" && i + 1 < argc) {
      family = argv[++i];
    } else if (argument == "--list") {
      flags.push_back(argv[i]);
      forwarded.push_back(argument);
    } else if ((argument == "--case" || argument == "--shard") && i + 1 < argc) {
      flags.push_back(argv[i]);
      flags.push_back(argv[++i]);
      forwarded.push_back(argument);
      forwarded.push_back(argv[i]);
    } else if (argument.rfind("--", 0) == 0) {
      positional.clear();
      break;
    } else {
      positional.push_back(argument);
    }
  }
  generator::options options;
  std::string error;
  family_kind kind {};
  const auto *members = find_group(family);
  const bool known = members || find_family(family, kind);
  if (!known || !generator::parse_options(int(flags.size()), flags.data(), options, error) || (!options.list && positional.size() != 4 + interposers(family))) {
    std::fprintf(stderr, "%s%susage: reshade_contract_matrix_runtime_tests [--family sl26|sl27|sl212_214|sl215_30|sl215|sl30|ngx|generic|ui] [--list | --case <id|family:name=value,...> | --shard i/N] official.dll frozenShaders test.addon64 freshOutput [interposer.dll per release]\n", error.c_str(), error.empty() ? "" : "\n");
    return 2;
  }
  if (options.list) {
    return list_cases(family);
  }
  try {
    if (members) {
      return run_group(*members, forwarded, positional);
    }
  } catch (const std::exception &failure) {
    std::fprintf(stderr, "FAIL API contract matrix L3: %s\n", failure.what());
    return 1;
  }
  // Below ctest's 240 s.
  std::thread([] {
    Sleep(230000);
    std::fputs("FAIL API contract matrix L3 watchdog\n", stderr);
    TerminateProcess(GetCurrentProcess(), 124);
  }).detach();
  try {
    const bool generic = kind == family_kind::generic;
    require(_putenv_s("SUNSHINE_DEPTH3D_EFFECT", "SunshineGame3D") == 0 && _putenv_s("SUNSHINE_GAME3D_AUTOMATIC", "1") == 0 && _putenv_s("SUNSHINE_GAME3D_AUTOMATIC_ACTIONS_TEST", "1") == 0 && _putenv_s("SUNSHINE_DEPTH_GENERIC_ONLY_TEST", generic ? "1" : "0") == 0, "Could not configure the contract runtime");
    width = 1280;
    height = 720;
    const auto directory = fs::absolute(positional[3]);
    require(!fs::exists(directory), "The contract runtime requires a fresh isolated output directory");
    fs::create_directories(directory);
    HMODULE sdk {};
    const auto *release = find_release(family);
    if (!generic) {
      const auto interposer = fs::absolute(positional[4]);
      require(fs::is_regular_file(interposer), "The release's interposer DLL is missing");
      const auto isolated = directory / "sl.interposer.dll";
      fs::copy_file(interposer, isolated);
      sdk = LoadLibraryW(isolated.c_str());
      require(sdk, "Could not load the isolated interposer");
    }
    contract_fixture fixture;
    fixture.runtime_directory = directory;
    fixture.version = release ? release->version : kind == family_kind::ui ? ui_runtime::version :
                                                                             matrix::sl_version {2, 7, 30};
    fixture.initialize(fs::absolute(positional[0]), fs::absolute(positional[1]), directory, 2, 0, fs::absolute(positional[2]));
    fixture.load_native(directory);
    fixture.bind(sdk, fixture.version.major == 2);

    struct foreground_scope {
      contract_fixture &fixture;

      ~foreground_scope() {
        fixture.set_foreground(nullptr);
      }
    } foreground {fixture};

    const auto result = run_family(fixture, family, options, directory / ("contract-" + family + ".jsonl"));
    fixture.finish();
    return result;
  } catch (const std::exception &failure) {
    std::fprintf(stderr, "FAIL API contract matrix L3: %s\n", failure.what());
    return 1;
  }
}
