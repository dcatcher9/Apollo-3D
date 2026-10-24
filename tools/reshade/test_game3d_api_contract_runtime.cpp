// SPDX-License-Identifier: GPL-3.0-only
// The API contract matrix's runtime layer, L3 (README, API contract matrix).
// Cases run through the native provider fixture: the official ReShade
// runtime, the test add-on and a real D3D12 device, with a synthetic
// Streamline release whose public entry points the add-on's own MinHook
// discovery finds and hooks (never the SunshineStreamlineTestCapture
// bypass). Each case is a frame script judged at its Presents against the
// contract oracle (test_game3d_api_contract_oracle.h): whether the provider
// shows fresh depth of the case's source with its tagged extent, and the
// refusal classes the test add-on records (SunshineDepthTestLastRefusal),
// never a log line.
//
// A frame can hold two copy points: its tag call and its evaluate (the next
// frame's, for a copy after Present). Its Present shows whichever copied, so
// a case is judged at the earliest point whose outcome requires a capture,
// else at its own point, which then permits the tag call's capture where
// the tag call permits one. A capture judged at a frame-generation tag call
// is deferred when nothing was nominated inside that call (the add-on's
// nomination gate, which frame-generation inputs pass).
//
// Isolation: every case has its own viewport and a fresh depth allocation.
// Afterwards frame generation is turned off, the depth untagged and the
// case's viewport retired, the fixture presents past maximum_source_age_ms
// with no SDK call, the case's depth must be gone, and
// SunshineGame3DTestRecalibrate resets calibration.
// A known-good control case runs first and after every ten cases; it must
// capture. Each family is its own ctest, so each runs in its own process.
//
// Families: sl27, the 2.7.30 interposer (frame tags, DLSS-G options).
//
// Command line: [--family sl27] [--list | --case <id|sl27:name=value,...> |
// --shard i/N] official.dll frozenShaders test.addon64 freshOutput
// interposer.dll (--list needs no other argument). Every judged case appends
// one JSON line to <freshOutput>/contract-<family>.jsonl:
// {layer, family, id, seed, dimensions, legal, rule, expected, observed,
// verdict, rows, evidence}.
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

  // A Streamline family: one release's interposer and the cells its public
  // entry points carry. The D3D12 state shrinks to five classes (design 3.1)
  // that the game's own list puts the tagged R32 allocation
  // (ALLOW_UNORDERED_ACCESS) in before the copy point.
  namespace sl_runtime {
    struct release {
      const char *family;
      matrix::sl_version version;
    };

    constexpr release releases[] {{"sl27", {2, 7, 30}}};

    enum dimension : unsigned {
      state,
      life,
      commands,
      entry,
      at,
      generated,
      fg,
      extent,
      untag,
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
    constexpr matrix::extent_form extents[] {matrix::extent_form::null, matrix::extent_form::padded, matrix::extent_form::offset, matrix::extent_form::one_dimension_zero, matrix::extent_form::outside_allocation, matrix::extent_form::declared_size_differs};
    constexpr int tag_call_point = 1, frame_generation_on = 1;

    generator::family family(const release &value);

    bool local(const generator::row &value) {
      return value[entry] >= int(matrix::tag_entry::local_array);
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
      // The fixture sends every structure at version 1.
      out.tag_header = out.resource_header = out.constants = out.viewport_handle = out.frame_generation_options = matrix::struct_form::first;
      out.fg = value[fg] ? matrix::fg_mode::on : matrix::fg_mode::off;
      out.untag = matrix::untag_pattern(value[untag]);
      out.tag_result = matrix::sl_result(value[tag_result]);
      out.evaluate_result = matrix::sl_result(value[evaluate_result]);
      out.options_result = matrix::sl_result(value[options_result]);
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
    // input has no tag call to untag or to return a result.
    bool constructible(const generator::row &value) {
      const auto assigned = [&](std::initializer_list<unsigned> dimensions) {
        for (const auto d : dimensions) {
          if (value[d] < 0) {
            return false;
          }
        }
        return true;
      };
      if (assigned({entry, at}) && local(value) && value[at] == tag_call_point) {
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
          {"fg", {"off", "on"}},
          {"extent", {"null", "padded", "offset", "one_dimension_zero", "outside_allocation", "declared_size_differs"}},
          {"untag", {"none", "null_after_each"}},
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
      sized(fg, 2);
      sized(extent, std::size(extents));
      sized(untag, std::size_t(matrix::untag_pattern::null_after_each) + 1);
      for (const auto d : {tag_result, evaluate_result, options_result}) {
        sized(d, std::size_t(matrix::sl_result::error) + 1);
      }
    }

    // Rows from games the add-on was tuned on (design 3.2), where a release
    // of this family can carry them; the oracle still decides each one.
    // Witcher 3's token contention, RE9's aliasing barrier, Shadows' dynamic
    // resolution and Hogwarts' NGX depth are other families' or layers'.
    struct seed {
      const char *name, *assignment;
    };

    constexpr seed seeds[] {
      {"shadows", "state=generic_read,commands=none,at=tag_call,fg=on,untag=null_after_each"},
      {"bf6", "fg=on,evaluate_result=warn_out_of_vram,options_result=warn_out_of_vram"},
      {"e33", "extent=padded"},  // Expedition 33's crop of its allocation.
    };

    // The known-good control: an OnlyValidNow frame tag copied at its
    // frame-generation tag call from a state its recording proves (S1).
    constexpr const char *control = "entry=set_tag_for_frame,life=only_valid_now,at=tag_call,fg=on";

    struct census {
      unsigned product {}, legal_product {}, seeded {}, one_factor {}, pairwise {};
      std::size_t pairs {}, covered {}, unfilled {};
    };

    // The three-way product of state class, lifecycle and command list
    // (design 3.2) at a same-frame evaluate and at a frame-generation tag
    // call, illegal rows included; the seed rows; each value alone; then a
    // pairwise covering array over every dimension on top of them.
    std::vector<generator::row> cells(const generator::family &owner, const matrix::sl_version &version, census &out) {
      std::vector<generator::row> rows;
      auto tag_call = generator::defaults(owner);
      tag_call[at] = tag_call_point;
      tag_call[fg] = frame_generation_on;
      for (const auto &base : {generator::defaults(owner), tag_call}) {
        for (const auto &value : generator::product(owner, {state, life, commands}, base)) {
          ++out.product;
          out.legal_product += matrix::legal(to_case(value, version));
          rows.push_back(value);
        }
      }
      std::vector<generator::row> seeded = rows;
      const auto count = rows.size();
      for (const auto &entry : seeds) {
        generator::row value;
        std::string error;
        require(generator::parse(owner, entry.assignment, value, error) && owner.legal(value), "a runtime seed row is malformed or illegal");
        generator::append_unique(owner, rows, {value});
        seeded.push_back(value);
      }
      out.seeded = unsigned(rows.size() - count);
      // Each value alone on the default row (a same-frame evaluate outside
      // frame-generation scope), so every value is judged on its own.
      const auto alone = generator::one_factor(owner, generator::defaults(owner));
      const auto before_alone = rows.size();
      generator::append_unique(owner, rows, alone);
      out.one_factor = unsigned(rows.size() - before_alone);
      seeded.insert(seeded.end(), alone.begin(), alone.end());
      const auto covering = generator::pairwise(owner, seeded);
      const auto before = rows.size();
      generator::append_unique(owner, rows, covering.rows);
      out.pairwise = unsigned(rows.size() - before);
      out.pairs = covering.pairs;
      out.covered = covering.covered;
      out.unfilled = covering.unfilled;
      return rows;
    }

    const char *seed_name(const generator::family &owner, const generator::row &value) {
      for (const auto &entry : seeds) {
        generator::row parsed;
        std::string error;
        if (generator::parse(owner, entry.assignment, parsed, error) && parsed == value) {
          return entry.name;
        }
      }
      return nullptr;
    }

    constexpr bool requires_capture(const matrix::outcome &value) {
      return (value.kind == matrix::outcome_kind::capture || value.kind == matrix::outcome_kind::continuous) && !matrix::either(value);
    }

    constexpr bool permits_capture(const matrix::outcome &value) {
      return matrix::permits(value, matrix::outcome_kind::capture) || matrix::permits(value, matrix::outcome_kind::continuous);
    }

    // What a case's Presents must show: the case at the copy point that
    // decides, its outcome, and the refusal classes a Present may show.
    struct expectation {
      matrix::sl_case value;
      matrix::outcome expected;
      bool at_tag_call {};
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

    expectation at_present(const matrix::sl_case &value) {
      expectation out {value, matrix::expect(value), value.at == matrix::copy_point::tag_call};
      if (matrix::known_gaps::streamline::local(value.entry) || out.at_tag_call) {
        out.accept(out.expected);
        return out;
      }
      auto tag = value;
      tag.at = matrix::copy_point::tag_call;
      tag.generated_presents = 0;
      const auto at_tag = matrix::expect(tag);
      if (requires_capture(at_tag)) {
        out.value = tag;
        out.expected = at_tag;
        out.at_tag_call = true;
        return out;
      }
      out.accept(out.expected);
      if (!requires_capture(out.expected) && permits_capture(at_tag) && !permits_capture(out.expected)) {
        // The tag call may copy; otherwise the case's own point decides.
        out.expected = matrix::detail::either(at_tag, out.expected);
        out.accept(at_tag);
      }
      return out;
    }
  }  // namespace sl_runtime

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

  // The contract runtime fixture: the native provider fixture with one case
  // at a time on its own viewport and allocation.
  struct contract_fixture: native_provider_fixture {
    using action_t = BOOL (*)(api::effect_runtime *);
    using gate_t = void (*)(HANDLE, HANDLE);
    using results_t = void (*)(std::int32_t, std::int32_t, std::int32_t, std::int32_t, std::int32_t);
    // Judged Presents a capture needs, from its first fresh one (design 2.1:
    // every real frame after warm-up has fresh depth).
    static constexpr unsigned steady_presents = 6;
    // How long a case waits for its first fresh capture; frame generation
    // owns its viewport's depth for a second after it is enabled
    // (association_timeout_ms), so an evaluate copy of an FG case waits past it.
    static constexpr std::uint64_t window_ms = 600, frame_generation_window_ms = 1600;
    // Isolation: past maximum_source_age_ms with no SDK call.
    static constexpr std::uint64_t isolation_ms = sunshine_scene_depth::maximum_source_age_ms + 10;
    static constexpr unsigned retired_viewports = 4;

    action_t recalibrate {};
    gate_t nomination_gate {};
    results_t set_results {};
    HANDLE nominated {};
    matrix::sl_version version;
    std::unique_ptr<uav_target> case_target;
    std::uint32_t next_viewport {1000}, next_scratch {900000};

    // The case being run.
    struct plan {
      matrix::sl_case value;
      sl_runtime::state_class state {};
      sunshine_streamline::abi_v2::viewport view {};
      sunshine_streamline::extent area {};
      std::uint32_t shown_width {}, shown_height {};
      std::int32_t tag_result {}, evaluate_result {}, options_result {};
      bool entered_layout {};  // enhanced: the allocation left legacy COMMON.
    } current;

    // What the case's Presents showed.
    struct presented {
      unsigned presents {}, judged {}, fresh {}, late {};
      bool used {}, steady {}, area {true}, nominated_at_call {};
      std::uint64_t first_fresh_ms {}, elapsed_ms {};
      std::vector<refusals::record> refused;  // Distinct reasons for the case's viewport, oldest first.
      sunshine_streamline::provider::source_status last;
    } seen;

    std::uint64_t refusal_serial {};

    ~contract_fixture() {
      if (nominated) {
        CloseHandle(nominated);
      }
    }

    void bind(HMODULE sdk) {
      using namespace sunshine_streamline;
      const auto addon = GetModuleHandleW(L"SunshineSBSTest.addon64");
      set_foreground = reinterpret_cast<void (*)(HWND)>(GetProcAddress(addon, "SunshineSbsTestSetForeground"));
      recalibrate = reinterpret_cast<action_t>(GetProcAddress(addon, "SunshineGame3DTestRecalibrate"));
      nomination_gate = reinterpret_cast<gate_t>(GetProcAddress(addon, "SunshineStreamlineTestNominationGate"));
      require(set_foreground && recalibrate && nomination_gate, "Test add-on lacks the foreground observer, recalibration or nomination gate");
      // The export stop at a focus loss would end FG reuse: select the
      // hidden game window in the test observer without moving focus.
      set_foreground(window);
      const auto getter = reinterpret_cast<abi_v2::get_feature_function>(GetProcAddress(sdk, "slGetFeatureFunction"));
      sl.new_token = reinterpret_cast<abi_v2::get_new_frame_token>(GetProcAddress(sdk, "slGetNewFrameToken"));
      sl.constants = reinterpret_cast<abi_v2::set_constants>(GetProcAddress(sdk, "slSetConstants"));
      sl.frame_tag = reinterpret_cast<abi_v2::set_tag_for_frame>(GetProcAddress(sdk, "slSetTagForFrame"));
      sl.global_tag = reinterpret_cast<abi_v2::set_tag>(GetProcAddress(sdk, "slSetTag"));
      sl.evaluate = reinterpret_cast<abi_v2::evaluate_feature>(GetProcAddress(sdk, "slEvaluateFeature"));
      set_results = reinterpret_cast<results_t>(GetProcAddress(sdk, "SunshineFixtureSetResults"));
      require(getter && sl.new_token && sl.constants && sl.global_tag && sl.evaluate && set_results, "Contract interposer lacks public SL exports or settable results");
      require(!(version < matrix::first_frame_tag) == (sl.frame_tag != nullptr), "Contract interposer's frame-tag export differs from its release");
      // Discovery is deferred to a Present: only an options function fetched
      // through the hooked getter afterwards is observed.
      require(reshade_log_contains({"Sunshine Streamline: observing 2.x"}, 15000), "Production discovery did not hook the contract interposer");
      void *function {};
      require(getter(1000, "slDLSSGSetOptions", function) == 0 && function, "Contract interposer did not expose FG options");
      sl.set_options = reinterpret_cast<set_options_t>(function);
      nominated = CreateEventW(nullptr, TRUE, FALSE, nullptr);
      require(nominated, "Create the nomination gate event");
      D3D12_FEATURE_DATA_D3D12_OPTIONS12 options {};
      require(SUCCEEDED(game->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &options, sizeof(options))) && options.EnhancedBarriersSupported, "The contract runtime's enhanced state class needs a device with enhanced barriers");
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

    void collect_refusals() {
      const auto value = last_refusal();
      if (value.serial <= refusal_serial) {
        return;
      }
      refusal_serial = value.serial;
      if (value.viewport != current.view.value) {
        return;
      }
      for (const auto &known : seen.refused) {
        if (!std::strcmp(known.name, value.name) && known.where == value.where) {
          return;
        }
      }
      seen.refused.push_back(value);
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
    // one (multi frame generation presents on its own).
    void present_generated() {
      checked(swapchain->Present(0, 0), "Generated Present between the tag and its copy point");
      ++seen.presents;
    }

    // One real frame of the case: the producer work, constants, then the
    // tag call (tag_frame) and the evaluate the case's copy point needs.
    void case_frame(bool tag_frame) {
      using namespace sunshine_streamline;
      const auto &value = current.value;
      produce();
      abi_v2::frame_token *token {};
      const auto frame = ++fg_frame;
      require(sl.new_token(token, &frame) == 0 && token, "Contract interposer did not supply a frame token");
      returned(sl.constants(camera_constants(false), *token, current.view), 0, "slSetConstants");
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
      abi_v2::resource_tag tag {{nullptr, tag_guid, 1}, &resource, 0, std::uint32_t(value.life), current.area};
      const bool local = matrix::known_gaps::streamline::local(value.entry);
      const auto command = reinterpret_cast<void *>(game_native_command);
      void *const tag_commands = value.commands == matrix::tag_commands::game_list ? command : nullptr;
      const auto tag_call = [&](const abi_v2::resource_tag *values) {
        const auto result = value.entry == matrix::tag_entry::set_tag_for_frame ? sl.frame_tag(*token, current.view, values, 1, tag_commands) :
                                                                                  sl.global_tag(current.view, values, 1, tag_commands);
        returned(result, current.tag_result, "the tag call");
        collect_refusals();
      };
      if (!local && tag_frame) {
        ResetEvent(nominated);
        nomination_gate(nominated, nullptr);
        tag_call(&tag);
        nomination_gate(nullptr, nullptr);
        seen.nominated_at_call = seen.nominated_at_call || WaitForSingleObject(nominated, 0) == WAIT_OBJECT_0;
        if (value.untag == matrix::untag_pattern::null_after_each) {
          const abi_v2::resource_tag untag {{nullptr, tag_guid, 1}, nullptr, 0, std::uint32_t(value.life), {}};
          tag_call(&untag);
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
        inputs.push_back(&tag.base);
      } else if (value.entry == matrix::tag_entry::local_chain_not_last) {
        tag.base.next = &current.view.base;
        inputs = {&tag.base};
      }
      returned(sl.evaluate(0, *token, inputs.data(), std::uint32_t(inputs.size()), command), current.evaluate_result, "slEvaluateFeature");
      collect_refusals();
    }

    void set_options(bool enabled, std::int32_t expected) {
      const fg_options value {{nullptr, fg_options_type, 1}, enabled ? 1u : 0u, enabled ? 1u : 0u};
      returned(sl.set_options(current.view, value), expected, "slDLSSGSetOptions");
    }

    // Warm-up: the case's own viewport, allocation, results and options.
    void prepare(const matrix::sl_case &value, sl_runtime::state_class state) {
      current = {};
      current.value = value;
      current.state = state;
      current.view = {{nullptr, sunshine_streamline::viewport_guid, 1}, next_viewport++};
      current.tag_result = sdk_result(value.tag_result);
      current.evaluate_result = sdk_result(value.evaluate_result);
      current.options_result = sdk_result(value.options_result);
      seen = {};
      refusal_serial = last_refusal().serial;
      case_target = target_uav(width / 2, height / 2, 0);
      selected = case_target.get();
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
        set_options(true, current.options_result);
      }
    }

    // Stimulus: real frames until the first fresh capture of the case's
    // source and steady_presents judged Presents from it, or the window.
    void observe(bool frame_generation_wait) {
      const auto &value = current.value;
      const bool alternate = value.at == matrix::copy_point::after_present && !matrix::known_gaps::streamline::local(value.entry);
      const auto window = frame_generation_wait ? frame_generation_window_ms : window_ms;
      const auto started = GetTickCount64();
      const auto native_source = native(*case_target);
      std::uint64_t last_capture {};
      unsigned since_first {}, steady_fresh {};
      for (bool tag_frame = true;; tag_frame = alternate ? !tag_frame : true) {
        render_tracked_depth = [this, tag_frame] {
          case_frame(tag_frame);
        };
        present_frame();
        ++seen.presents;
        collect_refusals();
        const auto s = inspect();
        seen.last = s.source;
        const auto &source = s.source;
        const bool shows = source.selected && source.ready && source.provider == sunshine_scene_depth::provider_kind::streamline && source.current.resource == native_source;
        const bool fresh = shows && !source.reused_depth && source.current.capture && source.current.capture != last_capture;
        const bool area = source.current.width == current.shown_width && source.current.height == current.shown_height;
        if (fresh) {
          last_capture = source.current.capture;
        }
        const auto elapsed = GetTickCount64() - started;
        if (!tag_frame) {
          seen.late += fresh;  // An evaluate after its tag's Present made a copy.
        } else {
          ++seen.judged;
          seen.used = seen.used || shows;
          seen.fresh += fresh;
          if (since_first) {
            ++since_first;
            steady_fresh += fresh && area;
          } else if (fresh) {
            since_first = 1;
            steady_fresh = area;
            seen.first_fresh_ms = elapsed;
          }
          if (shows && !area) {
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

    // Recovery: frame generation off, the depth untagged, the case's viewport
    // retired, Presents with no SDK call past the source age; the case's
    // depth must be gone.
    void isolate() {
      using namespace sunshine_streamline;
      render_tracked_depth = {};
      set_results(0, 0, 0, 0, 0);
      current.tag_result = current.evaluate_result = 0;
      if (current.value.fg != matrix::fg_mode::off) {
        set_options(false, 0);
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
      present_frame();
      render_tracked_depth = {};
      const auto until = GetTickCount64() + isolation_ms;
      do {
        present_frame();
      } while (GetTickCount64() < until);
      present_frame();
      const auto s = inspect();
      require(!(s.source.ready && s.source.current.resource == native(*case_target)), "Isolation: a case's depth outlived its source age with no SDK call");
      require(recalibrate(observed.runtime), "Isolation: SunshineGame3DTestRecalibrate failed");
      selected = first.get();
      case_target.reset();
    }

    struct judged_case {
      matrix::sl_case value;
      sl_runtime::expectation decided;
      matrix::result actual;
      presented observed;
      bool legal {}, satisfied {};
      gaps::verdict verdict {};
      bool matched[std::size(gaps::sl_runtime_rows)] {};
      std::uint64_t elapsed_ms {};
    };

    // At Present: a source shown steadily is a capture (deferred at a
    // frame-generation tag call when nothing was nominated inside it: the
    // nomination gate fires for frame-generation inputs only), and one shown
    // otherwise a hold; a source never shown is a recorded contract class,
    // one the copy points give where there is one (each copy point may
    // refuse), else a hold no gate recorded.
    static matrix::result result_of(const sl_runtime::expectation &decided, const presented &value) {
      matrix::result out;
      if (value.used) {
        out.kind = value.steady && value.area && !value.late ? matrix::outcome_kind::capture : matrix::outcome_kind::hold;
        out.deferred = decided.at_tag_call && matrix::detail::frame_generation_scope(decided.value) && !value.nominated_at_call;
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
    // extent, no copy after its tag's Present and the expected deferral. A
    // Present that shows nothing meets a refusal or a hold; a recorded
    // contract class must be one a copy point gives, unless a hold is
    // permitted (a hold with no earlier capture shows nothing either).
    static bool satisfies(const sl_runtime::expectation &decided, const matrix::result &actual, const presented &value, bool legal) {
      if (!legal) {
        return !value.used && !value.late;
      }
      const auto &expected = decided.expected;
      if (value.used) {
        return sl_runtime::permits_capture(expected) && value.steady && value.area && !value.late && actual.deferred == expected.deferred;
      }
      if (value.late || !(matrix::permits(expected, matrix::outcome_kind::refuse) || matrix::permits(expected, matrix::outcome_kind::hold) || matrix::permits(expected, matrix::outcome_kind::ignored))) {
        return false;
      }
      return matrix::permits(expected, matrix::outcome_kind::hold) || actual.kind != matrix::outcome_kind::refuse || decided.accepts(actual.why);
    }

    judged_case judge(const generator::family &owner, const generator::row &assignment, const char *name) {
      judged_case out;
      out.value = sl_runtime::to_case(assignment, version);
      out.legal = owner.legal(assignment);
      out.decided = sl_runtime::at_present(out.value);
      require(out.decided.expected.by != nullptr, "the oracle named no rule for a Streamline runtime cell");
      const auto state = sl_runtime::state_class(assignment[sl_runtime::state]);
      // A copy at an evaluate waits past frame generation's first second.
      const bool frame_generation_wait = out.value.fg != matrix::fg_mode::off && out.value.at != matrix::copy_point::tag_call;
      const auto started = GetTickCount64();
      run_script({name, 8000, [&] {
                    prepare(out.value, state);
                  },
                  [&] {
                    observe(frame_generation_wait);
                  },
                  [&] {
                    isolate();
                  }});
      out.elapsed_ms = GetTickCount64() - started;
      out.observed = seen;
      out.actual = result_of(out.decided, out.observed);
      out.satisfied = satisfies(out.decided, out.actual, out.observed, out.legal);
      out.verdict = gaps::judge_each(gaps::sl_runtime_rows, gaps::present_case {out.value, out.decided.value}, out.satisfied, {out.actual}, out.matched);
      return out;
    }
  };

  constexpr unsigned runtime_rows = std::size(gaps::sl_runtime_rows);

  void describe(const char *what, const generator::family &owner, const generator::row &assignment, const contract_fixture::judged_case &value) {
    const auto &seen = value.observed;
    std::string rows;
    const char *ids[runtime_rows] {};
    for (unsigned i = 0; i != runtime_rows; ++i) {
      ids[i] = gaps::sl_runtime_rows[i].id;
    }
    std::fprintf(stderr, "%s: case %s %s %s%s\n", what, owner.name, generator::id_text(generator::case_id(owner, assignment)).c_str(), generator::describe(owner, assignment).c_str(), value.legal ? "" : " (illegal input)");
    judging::describe_expected(value.decided.expected);
    std::fprintf(stderr, "  decided at %s; Presents %u (judged %u, fresh %u, after-Present copies %u); used=%u steady=%u area=%u nominated_at_call=%u first_fresh_ms=%llu; shown resource=0x%llx %ux%u reused=%u fg_input=%u\n", value.decided.at_tag_call ? "the tag call" : "the case's copy point", seen.presents, seen.judged, seen.fresh, seen.late, seen.used, seen.steady, seen.area, seen.nominated_at_call, static_cast<unsigned long long>(seen.first_fresh_ms), static_cast<unsigned long long>(seen.last.current.resource), seen.last.current.width, seen.last.current.height, seen.last.reused_depth, seen.last.frame_generation_input);
    for (const auto &record : seen.refused) {
      std::fprintf(stderr, "  refused: %s (%s) adapter=%u gate=%u stage=%u status=%u fg=%u\n", record.name, class_name(record.reason), unsigned(record.provider), unsigned(record.where), unsigned(record.stage), unsigned(record.status), record.frame_generation);
    }
    std::fprintf(stderr, "  result: %s why=%s deferred=%u; rows %s; %s\n", judging::name(value.actual.kind), game3d_contract::jsonl::name(value.actual.why), value.actual.deferred, judging::row_names(value.matched, ids, runtime_rows, rows), judging::name(value.verdict));
  }

  nlohmann::json json_line(const generator::family &owner, const generator::row &assignment, const contract_fixture::judged_case &value, const char *seed) {
    const auto &seen = value.observed;
    auto line = game3d_contract::jsonl::line("L3", owner, assignment, seed, value.legal, value.decided.expected, value.actual, value.verdict, gaps::sl_runtime_rows, value.matched);
    line["expected"]["decided_at"] = value.decided.at_tag_call ? "tag_call" : owner.dimensions[sl_runtime::at].values[assignment[sl_runtime::at]];
    nlohmann::json refused = nlohmann::json::array();
    for (const auto &record : seen.refused) {
      refused.push_back({{"name", record.name}, {"class", class_name(record.reason)}, {"adapter", unsigned(record.provider)}, {"gate", unsigned(record.where)}, {"stage", unsigned(record.stage)}, {"status", unsigned(record.status)}});
    }
    auto &observed = line["observed"];
    observed["used"] = seen.used;
    observed["steady"] = seen.steady;
    observed["area"] = seen.area;
    observed["late_copies"] = seen.late;
    observed["refused"] = refused;
    line["evidence"] = {{"elapsed_ms", value.elapsed_ms}, {"presents", seen.presents}, {"judged", seen.judged}, {"fresh", seen.fresh}, {"first_fresh_ms", seen.first_fresh_ms}, {"nominated_at_call", seen.nominated_at_call}, {"shown", {{"width", seen.last.current.width}, {"height", seen.last.current.height}, {"reused", seen.last.reused_depth}, {"fg_input", seen.last.frame_generation_input}}}};
    return line;
  }

  const sl_runtime::release *find_release(const std::string &name) {
    for (const auto &value : sl_runtime::releases) {
      if (name == value.family) {
        return &value;
      }
    }
    return nullptr;
  }

  int list_cases(const sl_runtime::release &release) {
    const auto owner = sl_runtime::family(release);
    sl_runtime::check_sizes(owner);
    sl_runtime::census census;
    for (const auto &value : sl_runtime::cells(owner, release.version, census)) {
      const auto decided = sl_runtime::at_present(sl_runtime::to_case(value, release.version));
      const auto *seed = sl_runtime::seed_name(owner, value);
      std::printf("%s %s %s%s expect=%s%s%s rule=%s at=%s%s%s\n", owner.name, generator::id_text(generator::case_id(owner, value)).c_str(), generator::describe(owner, value).c_str(), owner.legal(value) ? "" : " (illegal)", judging::name(decided.expected.kind), matrix::either(decided.expected) ? "|" : "", matrix::either(decided.expected) ? judging::name(decided.expected.alternative) : "", decided.expected.by->id, decided.at_tag_call ? "tag_call" : "own", seed ? " seed=" : "", seed ? seed : "");
    }
    std::printf("%u cells: %u-row state x lifecycle x command-list product (%u legal), %u seed rows, %u one-factor rows, %u pairwise rows covering %zu of %zu legal pairs (%zu unfilled)\n", census.product + census.seeded + census.one_factor + census.pairwise, census.product, census.legal_product, census.seeded, census.one_factor, census.pairwise, census.covered, census.pairs, census.unfilled);
    return 0;
  }

  // The cells selected by options, each judged after the control as the
  // design's isolation requires.
  int run_family(contract_fixture &fixture, const sl_runtime::release &release, const generator::options &options, const fs::path &jsonl) {
    const auto owner = sl_runtime::family(release);
    sl_runtime::check_sizes(owner);
    sl_runtime::census census;
    std::vector<generator::row> rows;
    std::vector<std::size_t> positions;
    if (!options.single.empty()) {
      generator::row value;
      std::string error;
      const auto colon = options.single.find(':');
      if (colon != std::string::npos) {
        require(options.single.substr(0, colon) == owner.name, "--case: the family differs from --family");
        if (!generator::parse(owner, options.single.substr(colon + 1), value, error)) {
          std::fprintf(stderr, "--case: %s\n", error.c_str());
          return 2;
        }
        require(sl_runtime::constructible(value), "--case: the fixture cannot carry that assignment");
      } else {
        bool found {};
        for (const auto &candidate : sl_runtime::cells(owner, release.version, census)) {
          if (!found && generator::id_text(generator::case_id(owner, candidate)) == options.single) {
            value = candidate;
            found = true;
          }
        }
        if (!found) {
          std::fprintf(stderr, "--case: no %s case has id %s (--list prints them)\n", owner.name, options.single.c_str());
          return 2;
        }
      }
      rows.push_back(value);
      positions.push_back(0);
    } else {
      const auto all = sl_runtime::cells(owner, release.version, census);
      require(census.covered == census.pairs && !census.unfilled, "the runtime covering array left a legal pair uncovered");
      for (std::size_t i = 0; i != all.size(); ++i) {
        if (options.selected(i)) {
          rows.push_back(all[i]);
          positions.push_back(i);
        }
      }
    }
    std::ofstream lines(jsonl, std::ios::binary);
    require(lines.good(), "Could not open the contract runtime's JSON lines");
    generator::row control;
    std::string error;
    require(generator::parse(owner, sl_runtime::control, control, error), "the runtime control row is malformed");
    judging::each_tally<runtime_rows> tally;
    unsigned controls {};
    std::uint64_t total_ms {}, slowest_ms {};
    const auto run_control = [&](std::size_t before) {
      const auto name = std::string(owner.name) + "-control-" + std::to_string(controls++);
      const auto value = fixture.judge(owner, control, name.c_str());
      if (value.verdict != gaps::verdict::meets || value.actual.kind != matrix::outcome_kind::capture) {
        describe("control failed", owner, control, value);
        throw std::runtime_error("the known-good control did not capture before case " + std::to_string(before) + ": isolation lost");
      }
    };
    for (std::size_t i = 0; i != rows.size(); ++i) {
      if (i % 10 == 0) {
        run_control(positions[i]);
      }
      const auto id = generator::id_text(generator::case_id(owner, rows[i]));
      const auto name = std::string(owner.name) + '-' + id;
      const auto value = fixture.judge(owner, rows[i], name.c_str());
      total_ms += value.elapsed_ms;
      slowest_ms = std::max(slowest_ms, value.elapsed_ms);
      std::printf("MEASURE contract-runtime family=%s case=%s elapsed_ms=%llu verdict=%s\n", owner.name, id.c_str(), static_cast<unsigned long long>(value.elapsed_ms), game3d_contract::jsonl::name(value.verdict));
      lines << json_line(owner, rows[i], value, sl_runtime::seed_name(owner, rows[i])).dump() << '\n';
      const auto *what = tally.add(value.verdict, value.matched, value.satisfied);
      if (what || !options.single.empty()) {
        describe(what ? what : "case", owner, rows[i], value);
      }
    }
    run_control(rows.size());
    require(lines.good(), "Could not write the contract runtime's JSON lines");
    std::printf("MEASURE contract-runtime family=%s cases=%zu controls=%u mean_ms=%llu max_ms=%llu total_s=%.1f estimate_ms=1300\n", owner.name, rows.size(), controls, static_cast<unsigned long long>(rows.empty() ? 0 : total_ms / rows.size()), static_cast<unsigned long long>(slowest_ms), double(total_ms) / 1000.);
    tally.finish(gaps::sl_runtime_rows, owner.name, !options.partial());
    std::printf("PASS API contract matrix L3 %s%s: %zu cells through the %u.%u.%u interposer and the production hooks, judged at Present (%u-row product, %u seed rows, %u one-factor rows, %u pairwise rows covering all %zu legal pairs), %u controls; ", owner.name, options.partial() ? " (partial)" : "", rows.size(), release.version.major, release.version.minor, release.version.patch, census.product, census.seeded, census.one_factor, census.pairwise, census.pairs, controls);
    tally.print(gaps::sl_runtime_rows);
    std::printf("\n");
    return 0;
  }
}  // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::vector<char *> flags {argv[0]};
  std::vector<std::string> positional;
  std::string family = "sl27";
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--family" && i + 1 < argc) {
      family = argv[++i];
    } else if (argument == "--list") {
      flags.push_back(argv[i]);
    } else if ((argument == "--case" || argument == "--shard") && i + 1 < argc) {
      flags.push_back(argv[i]);
      flags.push_back(argv[++i]);
    } else if (argument.rfind("--", 0) == 0) {
      positional.clear();
      break;
    } else {
      positional.push_back(argument);
    }
  }
  generator::options options;
  std::string error;
  const auto *release = find_release(family);
  if (!release || !generator::parse_options(int(flags.size()), flags.data(), options, error) || (!options.list && positional.size() != 5)) {
    std::fprintf(stderr, "%s%susage: reshade_contract_matrix_runtime_tests [--family sl27] [--list | --case <id|family:name=value,...> | --shard i/N] official.dll frozenShaders test.addon64 freshOutput interposer.dll\n", error.c_str(), error.empty() ? "" : "\n");
    return 2;
  }
  if (options.list) {
    return list_cases(*release);
  }
  // Below ctest's 240 s.
  std::thread([] {
    Sleep(230000);
    std::fputs("FAIL API contract matrix L3 watchdog\n", stderr);
    TerminateProcess(GetCurrentProcess(), 124);
  }).detach();
  try {
    require(_putenv_s("SUNSHINE_DEPTH3D_EFFECT", "SunshineGame3D") == 0 && _putenv_s("SUNSHINE_GAME3D_AUTOMATIC", "1") == 0 && _putenv_s("SUNSHINE_GAME3D_AUTOMATIC_ACTIONS_TEST", "1") == 0 && _putenv_s("SUNSHINE_DEPTH_GENERIC_ONLY_TEST", "0") == 0, "Could not configure the contract runtime");
    width = 1280;
    height = 720;
    const auto directory = fs::absolute(positional[3]);
    require(!fs::exists(directory), "The contract runtime requires a fresh isolated output directory");
    const auto interposer = fs::absolute(positional[4]);
    require(fs::is_regular_file(interposer), "The release's interposer DLL is missing");
    fs::create_directories(directory);
    const auto isolated = directory / "sl.interposer.dll";
    fs::copy_file(interposer, isolated);
    const auto sdk = LoadLibraryW(isolated.c_str());
    require(sdk, "Could not load the isolated interposer");
    contract_fixture fixture;
    fixture.runtime_directory = directory;
    fixture.version = release->version;
    fixture.initialize(fs::absolute(positional[0]), fs::absolute(positional[1]), directory, 2, 0, fs::absolute(positional[2]));
    fixture.load_native(directory);
    fixture.bind(sdk);

    struct foreground_scope {
      contract_fixture &fixture;

      ~foreground_scope() {
        fixture.set_foreground(nullptr);
      }
    } foreground {fixture};

    const auto result = run_family(fixture, *release, options, directory / ("contract-" + family + ".jsonl"));
    fixture.finish();
    return result;
  } catch (const std::exception &failure) {
    std::fprintf(stderr, "FAIL API contract matrix L3: %s\n", failure.what());
    return 1;
  }
}
