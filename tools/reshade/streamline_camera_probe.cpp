// SPDX-License-Identifier: GPL-3.0-only
#include "streamline_camera_probe.h"
#include "addon_lifetime.h"
#include "addon_config.h"
#include "hook_host.h"
#include "api_contract.h"
#include "streamline_camera_data.h"
#include "streamline_camera_version.h"
#include "streamline_buffer_contract.h"
#include "streamline_v1_resource_state.h"
#include "upscaler_call_trace.h"
#include "game3d_diagnostic_metadata.h"
#include "game3d_ui_mask.h"
#include "observer_snapshot.h"
#include "streamline_camera_probe_test_access.h"
#include "streamline_native_observer.h"
#include <MinHook.h>
#include <reshade.hpp>
#include "async_log.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
#include <functional>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#define SUNSHINE_UPSCALER_CALLER reinterpret_cast<std::uintptr_t>(_ReturnAddress())
#else
#define SUNSHINE_UPSCALER_CALLER reinterpret_cast<std::uintptr_t>(__builtin_extract_return_addr(__builtin_return_address(0)))
#endif

namespace sunshine_streamline {
  // A Streamline tag's contract: its state proof, lifetime and the lifecycle
  // match() judges (api_contract.h).
  namespace sl_contract = sunshine_api_contract::streamline;
  SUNSHINE_PROBE_INTERNAL {
    namespace dump_metadata = sunshine_game3d::diagnostic;
    dump_metadata::stamp dump_stamp(std::uint64_t epoch, std::uint64_t sequence, std::uint64_t tick,
        std::uint32_t viewport = UINT32_MAX, std::uint64_t token = 0, std::uint64_t frame = 0,
        bool numeric = false, std::uint64_t command = 0) {
      return {sunshine_game3d::diagnostic_metadata_generation(), epoch, sequence, tick, token, frame, command, viewport, numeric};
    }
    using abi = versioning::abi;
    enum class origin { global, frame, local };
    // sl_core_api.h sets no limit on numTags or numInputs (the contract
    // matrix's S18): a tag array or input list is read tag_chunk elements at
    // a time, on the stack.
    constexpr unsigned slot_count = frame_generation::viewport_slots, tag_chunk = 32, frame_count = 32;
    // Canonical buffer types this observer keeps, depth kinds first. Every
    // tag array below is indexed by position in this list.
    constexpr std::array<std::uint32_t, 9> observed_tag_types{buffers::depth, buffers::high_resolution_depth,
      buffers::linear_depth, buffers::hudless_color, buffers::scaling_input_color, buffers::scaling_output_color,
      buffers::backbuffer, buffers::ui_color_and_alpha, buffers::ui_alpha};
    constexpr unsigned slot_of(std::uint32_t type) {
      for (unsigned i = 0; i != observed_tag_types.size(); ++i) if (observed_tag_types[i] == type) return i;
      return static_cast<unsigned>(observed_tag_types.size());
    }
    static_assert(slot_of(buffers::depth) == 0 && slot_of(buffers::high_resolution_depth) == 1 &&
      slot_of(buffers::linear_depth) == 2, "depth kinds occupy the first three tag slots");
    // Explicit display-resolution depth first, then render depth, then linear distance.
    constexpr std::array<unsigned, 3> depth_priority{slot_of(buffers::high_resolution_depth), slot_of(buffers::depth),
      slot_of(buffers::linear_depth)};
    // The UI kinds, in the order a batch offers them.
    constexpr std::array<unsigned, 4> ui_priority{slot_of(buffers::hudless_color), slot_of(buffers::ui_alpha),
      slot_of(buffers::ui_color_and_alpha), slot_of(buffers::backbuffer)};
    // Defined in sl_observation.inl.
    bool known_renderer_contract(std::uint32_t feature);
    unsigned tag_index(std::uint32_t type);
    struct tag_record {
      depth_tag value;
      std::uint64_t tick{};
      std::uintptr_t token{};
      origin source{};
      bool present{}, supported{};
      std::uint64_t serial{}, loss{};
      frame_identity frame;
      std::uint32_t native_state{0xffffffffu};
      bool state_requires_observation{};
      // Its call supplied a command list (an evaluate input's evaluation always does).
      bool commanded{};
      depth_capture::source_ref direct_source;
      std::uint64_t source_present_generation{};
      double precision_scale{1.0}, precision_bias{};
    };
    struct viewport_record {
      camera_data camera;
      camera_key key;
      decode_status decoded{decode_status::short_buffer};
      std::array<tag_record, observed_tag_types.size()> global_tags{}, local_tags{};
      std::array<tag_record, observed_tag_types.size()> active_tags{};
      // Per depth kind, the viewport's newest global or frame tag call that
      // named a resource. An untag stored after it leaves it in place, so an
      // evaluate that reads that untag judges the call it ended
      // (ended_depth_tag, tag_call_decided).
      std::array<tag_record, depth_priority.size()> depth_calls{};
      frame_identity camera_frame;
      std::uintptr_t camera_token{};
      std::uint64_t camera_tick{}, camera_serial{}, changed{};
      // The viewport's newest frame: its key (frame_key; 0
      // unknown, and a repeated key is the same frame) and the observation
      // sequence of the constants call that began it.
      std::uint64_t frame_key{}, frame_serial{};
      // Observed camera resets of this viewport; each starts a new temporal history.
      std::uint64_t history{};
      // The newest valid camera's depth coefficients and how many consecutive
      // valid cameras of this viewport repeated them exactly.
      camera_validation steady_projection;
      std::uint32_t steady_frames{};
      std::uint32_t viewport{};
      bool used{}, has_camera{};
    };
    // Consecutive identical valid cameras before a viewport's depth
    // coefficients apply to depth without its own frame's camera.
    constexpr std::uint32_t steady_projection_frames = 8;
    struct targets { void *constants{}, *tag{}, *tag_for_frame{}, *evaluate{}, *new_frame_token{}, *get_feature_function{}; };
    struct batch {
      std::array<tag_record, observed_tag_types.size()> tags{}; // Three depth and six independently tagged UI/color kinds.
      unsigned count{};
      std::uint32_t viewport{};
      std::uint64_t observation{};
      bool valid_viewport{};
      std::uint64_t serial{}, loss{}, tick{};
      frame_identity frame;
    };
    struct frame_record {
      frame_identity frame;
      camera_data camera;
      decode_status decoded{decode_status::short_buffer};
      std::array<tag_record, observed_tag_types.size()> tags{};
      std::uint64_t camera_serial{}, camera_tick{}, loss{}, changed{}, history{};
      std::uint32_t viewport{};
      bool used{}, has_camera{};
    };
    struct token_record { frame_identity frame; };

    // POD/atomic storage remains live in the pinned add-on after AddonUninit. In
    // particular there is no destructible C++ mutex or worker used by detours.
    SRWLOCK poll_lock = SRWLOCK_INIT;
    struct metadata_state {
      viewport_record records[slot_count]{};
      frame_record frames[frame_count]{};
      std::uint64_t epoch{}, observation{}, revision{};
    };
    using metadata_owner = observation::snapshot_owner<metadata_state, 8>;
    metadata_owner metadata;
#if defined(SUNSHINE_STREAMLINE_PROBE_TEST) || defined(SUNSHINE_SBS_RUNTIME_TEST_ADDON)
    // Test seam: the next count metadata reads (read_metadata) fail, as a
    // read fails that lost the race with the writers on every attempt
    // (observer_snapshot.h), which no fixture produces by itself
    // (testing::fail_metadata_reads, SunshineStreamlineTestFailMetadataReads).
    std::atomic<unsigned> metadata_read_failures{};
#endif
    struct token_state {
      token_record tokens[frame_count]{};
      std::uint64_t observation{};
    };
    using token_owner = observation::snapshot_owner<token_state, 8>;
    token_owner token_metadata;
    // Defined in sl_observation.inl.
    token_owner::transaction write_tokens_briefly();
    std::atomic<bool> observing{};
    std::atomic<bool> source_requested{};
    std::atomic<std::uint64_t> observation_generation{};
    std::atomic<std::uint64_t> constant_calls{}, tag_calls{}, evaluation_calls{}, dropped{}, invalid{};
    // Depth tags naming no resource (Streamline's untag): a null Resource
    // pointer, or a Resource whose native is null.
    std::atomic<std::uint64_t> depth_untags{};
    // FG tag boundaries that named no capturable depth.
    std::atomic<std::uint64_t> fg_refusals{};
    // Next tick each bounded tag line may be written (5 s apart).
    std::atomic<std::uint64_t> next_compatibility_report{}, next_untag_report{}, next_fg_refusal_report{};
#include "presentation_scope_supply.inl"
#include "presentation_scope_mode.inl"
    // SDK calls answered with eWarnOutOfVRAM: accepted, counted for the log.
    std::atomic<std::uint64_t> vram_warnings{};
    // One verdict per 2.x call (abi_v2::accepted): the warning is a success.
    bool sdk_accepted(std::int32_t result) {
      if (result == abi_v2::result_warn_out_of_vram) vram_warnings.fetch_add(1, std::memory_order_relaxed);
      return abi_v2::accepted(result);
    }
    std::atomic<std::uint64_t> call_sequence{}, loss_revision{};
    // Lost constants observations (each may have hidden a camera reset) and
    // observation restarts. With a viewport's own resets this alone bounds
    // temporal history (feedback.revision); other transient losses (a busy
    // lock, an SDK failure) cannot hide a reset and leave history intact.
    std::atomic<std::uint64_t> history_loss{};
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
    thread_local bool entry_policy_armed{}, entry_policy_observed{}, entry_path_selected{}, entry_source_admitted{};
#endif
    loss_diagnostics::journal loss_journal;
    thread_local loss_diagnostics::context loss_context;
    // Nested middleware calls restore the outer callback's metadata. This
    // context is diagnostic only and never participates in source admission.
    struct loss_context_scope {
      loss_diagnostics::context previous{loss_context};
      explicit loss_context_scope(std::uint64_t sequence = 0, std::uint32_t viewport = UINT32_MAX,
          std::uint32_t feature = UINT32_MAX) { loss_context = {sequence, viewport, feature}; }
      ~loss_context_scope() { loss_context = previous; }
    };
    std::atomic<std::uint64_t> epoch{};
    std::uint64_t next_discovery{};
    HMODULE addon_module{};
    abi installed_abi{};
    // Header numbering of the hooked interposer; raw tags are canonicalized once.
    buffers::contract installed_contract{};
    bool hooks_installed{}, permanently_rejected{}, reported_missing{}, minhook_initialized{};
    std::atomic<bool> v1_private_state_ready{};
    bool v1_private_state_checked{};
    targets installed_targets{};
    abi_v1::set_constants original_constants_v1{};
    abi_v1::set_tag original_tag_v1{};
    abi_v1::evaluate_feature original_evaluate_v1{};
    abi_v2::set_constants original_constants_v2{};
    abi_v2::set_tag original_tag_v2{};
    abi_v2::set_tag_for_frame original_frame_tag_v2{};
    abi_v2::evaluate_feature original_evaluate_v2{};
    abi_v2::get_new_frame_token original_new_token_v2{};
    abi_v2::get_feature_function original_feature_function_v2{};

    // Surveyed 2.x sl_core_types.h/sl_dlss_g.h prefixes. These are copied,
    // never called as SDK objects or followed past the supported version.
    constexpr guid precision_guid{0x98f6e9ba, 0x8d16, 0x4831, {0xa8,0x02,0x4d,0x3b,0x52,0xff,0x26,0xbf}};
    constexpr guid fg_options_guid{0xfac5f1cb, 0x2dfd, 0x4f36, {0xa1,0xe6,0x3a,0x9e,0x86,0x52,0x56,0xc5}};
    struct precision_info { base_structure base; std::uint32_t formula{}; float bias{}, scale{1}; };
    using fg_options_prefix = abi_v2::fg_options_prefix;
    using fg_set_options = abi_v2::fg_set_options;
    struct fg_target {
      void *entry{};
      HMODULE owner{};
      std::atomic<fg_set_options> original{};
      std::atomic<unsigned> state{}; // 0 wrapper/pending detour, 1 installed, 2 rejected
    };
    std::array<fg_target, 4> fg_targets;
    unsigned fg_target_count{};
    SRWLOCK fg_lock = SRWLOCK_INIT;
    std::atomic<void *> current_fg_target{};
    std::atomic<bool> fg_available{};
    std::atomic<std::uint64_t> fg_loss_revision{};
    thread_local unsigned fg_forwarding_depth{};

    // Defined in sl_observation.inl.
    void release_camera_records();
    bool same(const guid &a, const guid &b);
    bool read_bytes(const void *source, void *destination, std::size_t size);
    template<class T> bool read(const T *source, T &destination);
    void message(const char *text);
    std::uint64_t lose(loss_diagnostics::reason cause, const char *site, std::uint32_t line,
        loss_diagnostics::context details = loss_context);
    void invalid_observation(loss_diagnostics::reason cause, const char *site, std::uint32_t line);
    void dropped_observation(loss_diagnostics::reason cause, const char *site, std::uint32_t line,
        loss_diagnostics::context details = loss_context);
    bool middleware_requested() {
      return !sunshine_addon_lifetime::stopping() && source_requested.load(std::memory_order_acquire);
    }
    unsigned tag_index(std::uint32_t type) {
      // Scaling input/output colors are not observed.
      if (type == buffers::scaling_input_color || type == buffers::scaling_output_color)
        return static_cast<unsigned>(observed_tag_types.size());
      return slot_of(type);
    }
    // Defined in sl_observation.inl.
    std::uint64_t observation_ticket();
    bool current(std::uint64_t ticket);
    metadata_owner::read_pin read_metadata(bool briefly = false);
    metadata_owner::transaction write_metadata(std::uint64_t ticket, bool briefly = false);
    void metadata_write_lost(const metadata_owner::transaction &update, const char *site,
        std::uint32_t line, loss_diagnostics::context details = loss_context);
    viewport_record &slot(metadata_state &state, std::uint32_t viewport);
    bool same_frame(const frame_identity &a, const frame_identity &b);
    frame_identity numeric_frame(std::uint32_t frame);
    bool token_current(const metadata_state &state, const frame_identity &frame);
    frame_identity tag_frame(const metadata_state &state, const viewport_record &record, const frame_identity &call,
        bool global, std::uint64_t serial);
    frame_generation::frame_mark evaluated_frame(std::uint32_t viewport);
    frame_identity observed_frame(std::uintptr_t address);
    bool viewport_value(const abi_v2::viewport &value, std::uint32_t &output);
    // Defined in sl_hooks.inl.
    unsigned retain_fg_target(void *entry, std::uint64_t ticket);
    // Defined in sl_frame_generation.inl.
    template<unsigned Index> std::int32_t hook_fg_options(const abi_v2::viewport &viewport, const fg_options_prefix &options);
    constexpr std::array<fg_set_options, 4> fg_hooks{hook_fg_options<0>, hook_fg_options<1>, hook_fg_options<2>, hook_fg_options<3>};
    // Defined in sl_hooks.inl.
    bool install_fg_hooks();
    // Defined in sl_hooks.inl.
    void discover_fg_options();
    // FG-only lookup fragments are defined in sl_hooks.inl.
    bool frame_generation_function_wanted(std::uint64_t ticket, std::uint32_t feature, const char *function_name);
    void offer_frame_generation_function(std::uint64_t ticket, bool fg_wanted, bool sdk_ok, void *&function);
    std::int32_t hook_get_feature_function(std::uint32_t feature, const char *function_name, void *&function) {
      const DWORD incoming = GetLastError();
      const auto ticket = observation_ticket();
      // A PCL marker lookup is still a game SDK call whose eWarnOutOfVRAM
      // reply vram_warnings counts; only its marker target is no longer kept.
      constexpr char expected[] = "slPCLSetMarker";
      char copied_name[sizeof(expected)]{};
      const bool pcl_lookup = ticket && feature == abi_v2::feature_pcl &&
        read_bytes(function_name, copied_name, sizeof(copied_name)) && std::memcmp(copied_name, expected, sizeof(expected)) == 0;
      const bool fg_wanted = frame_generation_function_wanted(ticket, feature, function_name);
      SetLastError(incoming);
      const auto result = original_feature_function_v2(feature, function_name, function);
      const DWORD outgoing = GetLastError();
      const bool sdk_ok = (pcl_lookup || fg_wanted) && sdk_accepted(result);
      offer_frame_generation_function(ticket, fg_wanted, sdk_ok, function);
      SetLastError(outgoing);
      return result;
    }
    // Defined in sl_observation.inl.
    frame_record *frame_slot(metadata_state &state, std::uint32_t viewport, const frame_identity &identity, std::uint64_t serial, bool create);
    const frame_record *find_frame(const metadata_state &state, std::uint32_t viewport, const frame_identity &identity);
    void store_camera(std::uint32_t viewport, const camera_data &camera, decode_status decoded,
        std::uint64_t frame, bool explicit_frame, std::uintptr_t token, std::uint64_t ticket,
        const frame_identity &identity, std::uint64_t serial, std::uint64_t loss, std::uint64_t entry_tick);
    void store_tags(batch &values);
    evaluation_snapshot capture_evaluation(std::uint32_t viewport, std::uint32_t feature,
        const frame_identity &identity, void *commands, std::uint64_t ticket, std::uint64_t serial,
        std::uint64_t loss, const batch *local = nullptr, const metadata_state *selected_state = nullptr,
        bool tag_boundary = false);
    void finish_evaluation(evaluation_snapshot value, bool success, std::uint64_t ticket);
    bool evaluate_input(const void *node);
    tag_refusal decode_precision(const void *extension, double &scale, double &bias, bool input_chain);
    void append_tag(batch &out, const abi_v2::resource_tag &tag, origin source, std::uintptr_t token, bool report_invalid = true);
    batch read_tags(const abi_v2::viewport &viewport, const abi_v2::resource_tag *tags,
        std::uint32_t count, origin source, std::uintptr_t token);

    bool hook_constants_v1(const abi_v1::constants &values, std::uint32_t frame, std::uint32_t viewport) {
      const DWORD incoming = GetLastError();
      const auto ticket = observation_ticket();
      const bool sample = ticket != 0;
      const auto serial = sample ? ++call_sequence : 0;
      const loss_context_scope loss_scope(serial, viewport);
      const auto loss = loss_revision.load(std::memory_order_acquire);
      const auto entry_tick = sample ? GetTickCount64() : 0;
      const auto diagnostic = dump_stamp(ticket, serial, entry_tick, viewport, 0, frame, true);
      camera_data camera;
      decode_status decoded = decode_status::short_buffer;
      if (sample) {
        ++constant_calls;
        abi_v1::constants copy{};
        const bool readable = read(&values, copy);
        if (readable) decoded = decode_v1_constants(&copy, sizeof(copy), camera);
        if (diagnostic.session) dump_metadata::observe_sl_constants(diagnostic, copy, readable, decoded);
      }
      SetLastError(incoming);
      const bool result = original_constants_v1(values, frame, viewport);
      const DWORD outgoing = GetLastError();
      loss_context.sdk_result = result ? 1 : 0; loss_context.has_sdk_result = true;
      if (diagnostic.session) dump_metadata::finish_sl_call(diagnostic, result);
      if (sample && result) store_camera(viewport, camera, decoded, frame, true, 0, ticket, numeric_frame(frame), serial, loss, entry_tick);
      else if (sample && current(ticket)) lose(loss_diagnostics::reason::sdk_failure, __func__, 1743);
      SetLastError(outgoing);
      return result;
    }
    bool hook_tag_v1(const abi_v1::resource *resource, std::uint32_t type, std::uint32_t viewport, const extent *area) {
      const DWORD incoming = GetLastError();
      const auto ticket = observation_ticket();
      const bool sample = ticket != 0;
      batch out;
      out.observation = ticket;
      out.serial = sample ? ++call_sequence : 0;
      const loss_context_scope loss_scope(out.serial, viewport);
      out.loss = loss_revision.load(std::memory_order_acquire);
      out.tick = sample ? GetTickCount64() : 0;
      const auto diagnostic = dump_stamp(ticket, out.serial, out.tick, viewport);
      const bool resource_has_state = installed_abi != abi::legacy_no_state;
      if (sample && diagnostic.session) dump_metadata::observe_sl_tag_v1(diagnostic, resource, type, area, resource_has_state);
      if (sample) {
        ++tag_calls;
        out.viewport = viewport;
        out.valid_viewport = true;
        const auto canonical_type = buffers::canonical(installed_contract, type);
        const unsigned index = tag_index(canonical_type);
        if (index < 6) {
          abi_v1::resource copy{};
          extent copied_area{};
          if (abi_v1::read_resource(resource, resource_has_state, copy, read_bytes) && (!area || read(area, copied_area))) {
            out.count = index + 1;
            auto &tag = out.tags[index];
            tag.present = true;
            tag.supported = copy.ext == nullptr && copy.type == 0;
            tag.value.refusal = copy.ext ? tag_refusal::extension : copy.type ? tag_refusal::resource_type : tag_refusal::none;
            tag.value.native_resource = tag.supported ? reinterpret_cast<std::uintptr_t>(copy.native) : 0;
            tag.value.type = canonical_type;
            tag.value.viewport = viewport;
            tag.value.area = copied_area;
            tag.native_state = copy.state;
            if (tag.supported && index < 3)
              tag.direct_source = depth_capture::retain_source(tag.value.native_resource);
            tag.source_present_generation = depth_capture::source_present_generation(tag.direct_source);
          } else {
            invalid_observation(loss_diagnostics::reason::invalid_input, __func__, 1787);
            out.count = index + 1;
            out.tags[index].present = true;
            out.tags[index].value.type = canonical_type;
            out.tags[index].value.viewport = viewport;
          }
        }
      }
      SetLastError(incoming);
      const bool result = original_tag_v1(resource, type, viewport, area);
      const DWORD outgoing = GetLastError();
      loss_context.sdk_result = result ? 1 : 0; loss_context.has_sdk_result = true;
      if (diagnostic.session) dump_metadata::finish_sl_call(diagnostic, result);
      if (sample && result && out.count) store_tags(out);
      else if (sample && !result && current(ticket)) lose(loss_diagnostics::reason::sdk_failure, __func__, 1801);
      SetLastError(outgoing);
      return result;
    }
    // Defined in sl_nomination.inl.
    bool unframed_tag_source(const evaluation_snapshot &snapshot);
    const char *source_rejection(const evaluation_snapshot &snapshot);
    bool direct_source_admissible(const evaluation_snapshot &snapshot);
    sl_contract::copy_point copy_point(const evaluation_snapshot &snapshot, const evaluated_depth_tag &tag);
    template<class Tag> sl_contract::tag_declaration declaration(const Tag &tag, bool version_one);
    const char *tag_rejection(const evaluation_snapshot &snapshot, const evaluated_depth_tag &tag,
        match_status *matched = nullptr);
    bool direct_tag_admissible(const evaluation_snapshot &snapshot, const evaluated_depth_tag &tag);
    bool depth_untag(const evaluation_snapshot &snapshot);
    sunshine_scene_depth::jitter_offset depth_frame_jitter(const evaluation_snapshot &snapshot,
        const evaluated_depth_tag &tag);
    // Defined in sl_report.inl; calls keep the original guarded reporting points.
    void report_tag_resource(const batch &out, const abi_v2::resource_tag &tag, origin source,
        const tag_record &value, const abi_v2::resource &resource, unsigned index, bool zero_resource_header);
    void report_rejected_tag(const batch &out, const abi_v2::resource_tag &tag, const tag_record &value,
        const abi_v2::resource &resource, bool precision_ok, bool resource_ok, std::atomic<std::uint64_t> &next_tag_report);
    void report_v1_state_lookup(const evaluation_snapshot &snapshot, const evaluated_depth_tag &tag,
        const depth_capture::input &value, const v1_resource_state::read_details &state_details,
        std::atomic<std::uint64_t> &next_state_report);
    // Defined in sl_report.inl.
    void report_declared_size(const evaluated_depth_tag &tag);
    // Defined in sl_nomination.inl.
    sunshine_scene_depth::frame normalize_depth_frame(const evaluation_snapshot &snapshot,
        const evaluated_depth_tag &tag, bool version_one);
    // A viewport's depth route under FG: none (FG not enabled), owned by its
    // live FG scope, or silent (enabled, but no FG copy accepted within a
    // second: depth_capture::frame_generation_scope_live). A busy metadata
    // read keeps an existing publication's FG ownership.
    // disabled (optional): the viewport's FG options were observed off in
    // this epoch (a known mode 0 under the current options revision).
    using fg_route = presentation_scope::fg_route;
    // Defined in sl_nomination.inl.
    fg_route frame_generation_route(std::uint32_t viewport, bool *disabled = nullptr);
    bool source_path_selected(const evaluation_snapshot &snapshot);
    bool owns_nested_depth_capture(const evaluation_snapshot &snapshot, bool copy_recorded);
    // Defined in sl_report.inl.
    const char *name(frame_identity_kind value);
    void report_fg_refusal(const evaluation_snapshot &snapshot, const char *reason, bool withdrawn, bool live,
        bool supplied = false);
    void count_tag_call_copy(const depth_capture::input *candidates, std::size_t count,
        const depth_capture::record_diagnostic &diagnostic);
    // Defined in sl_nomination.inl.
    const char *candidate_rejection(const evaluation_snapshot &snapshot);
    bool tag_call_decided(const evaluation_snapshot &snapshot, bool version_one);
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
    // The nomination tap (testing::latest_nomination). Only the thread that
    // armed it records, so an unarmed fixture runs exactly as before. Each
    // nominate_depth_source call evaluates its gates as it begins (pending),
    // and its finish_capture keeps the result, while the observation is
    // current, as its viewport's newest nomination, never over a newer one.
    struct tapped_nomination {
      sunshine_streamline::testing::nomination value;
      std::uint64_t epoch{}, observation{};
      bool used{};
    };
    thread_local bool nomination_tap_armed{};
    bool nomination_tap_pending{};
    sunshine_streamline::testing::nomination nomination_tap_value;
    std::array<tapped_nomination, 8> nomination_tap{};
    bool nomination_tap_current(const tapped_nomination &entry) {
      return entry.used && entry.epoch == epoch.load(std::memory_order_acquire) &&
        entry.observation == observation_generation.load(std::memory_order_acquire);
    }
    void tap_nomination(const evaluation_snapshot &snapshot, bool version_one) {
      auto &out = nomination_tap_value;
      out = {};
      out.path_selected = source_path_selected(snapshot);
      out.source_admissible = direct_source_admissible(snapshot);
      for (unsigned i = 0; i != snapshot.tags.size(); ++i) {
        out.tag_admissible[i] = direct_tag_admissible(snapshot, snapshot.tags[i]);
        if (out.path_selected && out.source_admissible && out.tag_admissible[i])
          out.frames[i] = normalize_depth_frame(snapshot, snapshot.tags[i], version_one);
      }
      out.snapshot = snapshot;
      for (auto &tag : out.snapshot.tags) { tag.direct_source.reset(); tag.ended.direct_source.reset(); }
      nomination_tap_pending = true;
    }
    void finish_tap(bool successful, std::uint64_t ticket) {
      if (!nomination_tap_pending) return;
      nomination_tap_pending = false;
      if (!current(ticket)) return;
      auto &value = nomination_tap_value;
      value.successful = successful;
      tapped_nomination *chosen = nullptr;
      for (auto &entry : nomination_tap) if (entry.used && entry.value.snapshot.viewport == value.snapshot.viewport) {
        chosen = &entry; break;
      }
      if (chosen && nomination_tap_current(*chosen) && chosen->value.snapshot.sequence >= value.snapshot.sequence) return;
      for (auto &entry : nomination_tap) if (!chosen && !nomination_tap_current(entry)) chosen = &entry;
      if (!chosen) {
        chosen = &nomination_tap[0];
        for (auto &entry : nomination_tap) if (entry.value.snapshot.sequence < chosen->value.snapshot.sequence) chosen = &entry;
      }
      *chosen = {std::move(value), epoch.load(std::memory_order_acquire), observation_generation.load(std::memory_order_acquire), true};
    }
#endif
    // Defined in sl_nomination.inl.
    std::uint64_t nominate_depth_source(const evaluation_snapshot &snapshot, bool version_one, bool *copy_recorded = nullptr,
        deferred_depth *deferred = nullptr);
    void finish_capture(std::uint64_t captured, bool successful, std::uint64_t ticket);
    bool hook_evaluate_v1(void *commands, std::uint32_t feature, std::uint32_t frame, std::uint32_t viewport);
    // Defined in sl_observation.inl.
    std::int32_t hook_constants_v2(const abi_v2::constants &values, const abi_v2::frame_token &frame, const abi_v2::viewport &viewport);
    // Defined in sl_ui_capture.inl.
    sunshine_game3d::ui_mask::attempt capture_ui_tag(const batch &out, void *commands,
        const tag_record &tag, std::uint64_t source_epoch, bool fg_enabled, deferred_ui *later = nullptr);
    using alpha_attempts = std::array<sunshine_game3d::ui_mask::attempt, 4>;
    using deferred_uis = std::array<deferred_ui, 4>;
    // Defined in sl_ui_capture.inl.
    bool carries_ui(const batch &out);
    alpha_attempts capture_ui_inputs(const batch &out, void *commands, deferred_uis *later = nullptr);
    void finish_ui_inputs(const alpha_attempts &attempts, bool kept);
    void initialize_frame_generation_ui(const frame_generation::inputs &value, frame_generation::attempt &out);
    void capture_frame_generation_ui(const frame_generation::inputs &value, frame_generation::attempt &out,
        std::uint64_t source_epoch, std::uint64_t loss, const frame_generation_snapshot &fg,
        std::uint64_t serial, std::uint64_t tick, std::uint64_t source_id);
    void finish_frame_generation_ui(const frame_generation::attempt &value, bool successful, bool still_current);
    // Defined in sl_observation.inl.
    evaluation_snapshot capture_depth_tag(const batch &out, void *commands);
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    // The deferral line's totals of depth copies deferred to their frame's
    // Present (R2), of those copied, and of the deferrals an untag cancelled
    // (R7): the session timelines judge each frame's own untag by them.
    extern "C" __declspec(dllexport) void SunshineStreamlineTestDeferrals(std::uint64_t *depth_deferred, std::uint64_t *depth_copied, std::uint64_t *cancelled) {
      if (depth_deferred) *depth_deferred = deferrals.depth.load(std::memory_order_relaxed);
      if (depth_copied) *depth_copied = deferrals.depth_copied.load(std::memory_order_relaxed);
      if (cancelled) *cancelled = deferrals.cancelled.load(std::memory_order_relaxed);
    }
    // The next count metadata reads fail (metadata_read_failures); 0 ends a
    // run of failures. The session timelines' reads=lost_at_present.
    extern "C" __declspec(dllexport) void SunshineStreamlineTestFailMetadataReads(unsigned count) {
      metadata_read_failures.store(count, std::memory_order_release);
    }
#endif
    // Defined in sl_frame_generation.inl.
    void report_deferrals();
    void count_deferred();
    unsigned deferred_kinds(const deferred_frame &entry);
    unsigned kind_count(unsigned kinds);
    void hand_over(std::uint64_t scope_epoch, std::uint32_t viewport, unsigned kinds, std::uint64_t sequence);
    void defer_tag_call(const batch &out, bool tokened, deferred_depth &depth, deferred_uis &ui, bool ui_current);
    // Defined in sl_observation.inl.
    std::int32_t hook_tag_v2(const abi_v2::viewport &viewport, const abi_v2::resource_tag *tags, std::uint32_t count, void *commands);
    std::int32_t hook_frame_tag_v2(const abi_v2::frame_token &frame, const abi_v2::viewport &viewport,
        const abi_v2::resource_tag *tags, std::uint32_t count, void *commands);
    batch read_local_inputs(const base_structure **inputs, std::uint32_t count, std::uintptr_t token, bool report_invalid);
    // Defined in sl_nomination.inl.
    std::int32_t hook_evaluate_v2(std::uint32_t feature, const abi_v2::frame_token &frame,
        const base_structure **inputs, std::uint32_t count, void *commands);

    // Defined in sl_observation.inl.
    std::int32_t hook_new_token_v2(abi_v2::frame_token *&token, const std::uint32_t *frame_index);

    // Defined in sl_hooks.inl.
    bool install_hooks(abi version, const targets &functions, const buffers::contract &contract);
    // Defined in sl_hooks.inl.
    versioning::information module_version(HMODULE module);
    // Defined in sl_hooks.inl.
    void discover_v1_private_state();
    // Defined in sl_hooks.inl.
    void discover();
  }

  // MinHook snapshots every thread on the system (~50 ms with ~10k threads)
  // for each enable or apply. Streamline hook discovery and newly registered
  // FG targets are therefore installed on a pool thread that holds
  // poll_lock; a present-path poll meanwhile skips, as on contention. The
  // add-on is pinned, so a pending callback never outlives its code.
  namespace {
    std::atomic<bool> install_scheduled{false};
    // Defined in sl_hooks.inl.
    bool fg_targets_pending();
    // Caller holds poll_lock.
    // Defined in sl_hooks.inl.
    void install_hook_work();
    // Defined in sl_hooks.inl.
    void CALLBACK install_on_pool(PTP_CALLBACK_INSTANCE, void *);
    // Caller holds poll_lock.
    // Defined in sl_hooks.inl.
    void schedule_install();
  }

  void initialize(HMODULE addon) {
    const bool source_enabled = sunshine_game3d::addon_config::streamline_depth_source();
    const bool call_trace_enabled = sunshine_game3d::addon_config::upscaler_call_trace();
    const bool ngx_source_enabled = sunshine_game3d::addon_config::ngx_depth_source();
    const bool queue_fence_order = sunshine_game3d::addon_config::queue_fence_order();
    native_observer::set_queue_order(queue_fence_order);
    sunshine_upscaler_trace::initialize(addon, call_trace_enabled, ngx_source_enabled);
    // A pool install may still be discovering after device recreation; wait
    // for it before resetting the state it reads and writes.
    AcquireSRWLockExclusive(&poll_lock);
    source_requested.store(false, std::memory_order_release);
    observing.store(false, std::memory_order_release);
    observation_generation.fetch_add(1, std::memory_order_acq_rel);
    addon_module = addon;
    // ReShade may re-register this process-pinned add-on after device recreation.
    // Keep the existing hooks and only start a fresh observation epoch.
    while (!token_metadata.clear()) SwitchToThread();
    release_camera_records();
    lose(loss_diagnostics::reason::lifecycle, __func__, 3382, {});
    ++history_loss; // Camera records were released: every viewport's history restarts.
    ++epoch;
    next_discovery = 0;
    source_requested.store(source_enabled, std::memory_order_release);
    if (hooks_installed) observing.store(source_enabled, std::memory_order_release);
    ReleaseSRWLockExclusive(&poll_lock);
  }
  bool source_enabled() { return source_requested.load(std::memory_order_acquire); }
  std::uint64_t depth_observation_revision() { return loss_revision.load(std::memory_order_acquire); }
  bool query_depth_observation_loss(std::uint64_t revision, loss_diagnostics::event &out) {
    return loss_journal.query(revision, out);
  }
  void shutdown() {
    sunshine_upscaler_trace::shutdown();
    source_requested.store(false, std::memory_order_release);
    observing.store(false, std::memory_order_release);
    observation_generation.fetch_add(1, std::memory_order_acq_rel);
    // The shared capture owner retains GPU completion/retirement leases.
    sunshine_game3d::ui_mask::invalidate_all();
    release_camera_records();
  }

  // present_deferred and the shared DLSS-G input wrappers are defined in sl_frame_generation.inl.

  const char *name(evidence_status value) {
    switch (value) {
      case evidence_status::inactive: return "inactive";
      case evidence_status::busy: return "snapshot-busy";
      case evidence_status::missing_evaluation: return "missing-evaluation";
      case evidence_status::observation_lost: return "observation-lost";
      case evidence_status::untracked_frame: return "untracked-frame";
      case evidence_status::missing_constants: return "missing-frame-constants";
      case evidence_status::rejected_constants: return "rejected-constants";
      case evidence_status::invalid_camera: return "invalid-camera";
      case evidence_status::camera_reset: return "camera-reset";
      case evidence_status::missing_depth: return "missing-depth";
      case evidence_status::stale: return "stale-evaluation";
      case evidence_status::source_associated_evaluation: return "source-associated-evaluation-reshade-frame-unverified";
    }
    return "unknown";
  }

  namespace {
    // Diagnostic only: whether this game also sends Streamline camera constants,
    // reported beside NGX capture. Never a cross-API frame match.
    // Defined in sl_report.inl in this same anonymous namespace.
    void report_camera_availability(std::uint64_t now);
  }

  // poll_guard::~poll_guard() is defined in sl_hooks.inl; its declaration stays in the public header.

  // try_poll() is defined in sl_hooks.inl; its declaration stays in the public header.

  // middleware_requested_for_poll() is defined in sl_hooks.inl; its declaration stays in the public header.

  // discovery_step is defined in sl_hooks.inl; its declaration stays in the public header.

  // coverage_step() is defined in sl_hooks.inl; its declaration stays in the public header.

  // camera_availability_step is defined in sl_hooks.inl; its declaration stays in the public header.

  // depth_capture_step() is defined in sl_hooks.inl; its declaration stays in the public header.

  // queue_order_step is defined in sl_hooks.inl; its declaration stays in the public header.

  // pending_installs_step() is defined in sl_hooks.inl; its declaration stays in the public header.

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  // The test add-on's exports (test_streamline_camera_probe_exports.cpp) read
  // camera data through this instantiation.
  SUNSHINE_PROBE_INTERNAL {
    template bool read(const camera_data *, camera_data &);
  }
#endif

#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
  namespace testing {
    void arm_nomination_tap(bool armed) {
      nomination_tap_armed = armed;
      nomination_tap_pending = false;
    }
    bool latest_nomination(std::uint32_t viewport, nomination &out) {
      for (const auto &entry : nomination_tap) if (nomination_tap_current(entry) && entry.value.snapshot.viewport == viewport) {
        out = entry.value; return true;
      }
      return false;
    }
    thread_local metadata_owner::transaction held_metadata_writer;
    thread_local metadata_owner::read_pin held_metadata_reader;
    void lock_records() { while (!(held_metadata_writer = metadata.try_write())) SwitchToThread(); }
    void unlock_records() { held_metadata_writer.reset(); }
    void lock_records_shared() { held_metadata_reader = read_metadata(); }
    void unlock_records_shared() { held_metadata_reader.reset(); }
    thread_local token_owner::transaction held_token_writer;
    thread_local std::array<token_owner::read_pin, 8> held_token_readers;
    void lock_tokens() { while (!(held_token_writer = token_metadata.try_write())) SwitchToThread(); }
    void unlock_tokens() { held_token_writer.reset(); }
    bool pin_tokens(unsigned index) {
      if (index >= held_token_readers.size()) return false;
      held_token_readers[index] = token_metadata.read();
      return bool(held_token_readers[index]);
    }
    void unpin_tokens() { for (auto &pin : held_token_readers) pin.reset(); }
    void fail_metadata_reads(unsigned count) { metadata_read_failures.store(count, std::memory_order_release); }
    void hold_evaluation_supply() { AcquireSRWLockExclusive(&evaluation_supply_lock); }
    void release_evaluation_supply() { ReleaseSRWLockExclusive(&evaluation_supply_lock); }
    void supply_evaluation(std::uint64_t scope_epoch, std::uint32_t viewport, std::uint64_t tick) {
      remember_evaluation_supply(scope_epoch, viewport, tick);
    }
    bool evaluation_supplied(std::uint64_t scope_epoch, std::uint32_t viewport) { return evaluation_supplies(scope_epoch, viewport); }
    presentation_scope::fg_route route_of(std::uint32_t viewport) { return frame_generation_route(viewport); }
    void age_frame_generation(std::uint64_t ms) {
      AcquireSRWLockExclusive(&fg_mode_lock);
      for (auto &slot : fg_modes) {
        auto &tick = slot.state.enabled_tick;
        if (tick) tick = tick > ms ? tick - ms : 1;
      }
      ReleaseSRWLockExclusive(&fg_mode_lock);
    }
    void set_fg_store_read_seam(std::function<void()> seam) { fg_store_read_seam = std::move(seam); }
    void set_fg_transition_seam(std::function<void()> seam) { fg_transition_seam = std::move(seam); }
    void defer_depth(std::uint32_t viewport, const depth_capture::input &candidate) {
      AcquireSRWLockExclusive(&evaluation_supply_lock);
      for (auto &entry : deferred_frames) if (!entry.used) {
        entry = {};
        entry.used = true; entry.observation = observation_ticket(); entry.epoch = epoch.load(std::memory_order_acquire);
        entry.serial = ++call_sequence; entry.tick = GetTickCount64(); entry.executions = depth_capture::game_executions();
        entry.viewport = viewport;
        entry.depth.candidates[0] = candidate; entry.depth.count = 1;
        break;
      }
      count_deferred();
      ReleaseSRWLockExclusive(&evaluation_supply_lock);
    }
    void arm_entry_policy() { entry_policy_armed = true; entry_policy_observed = false; }
    bool entry_policy(bool &selected, bool &admissible) {
      selected = entry_path_selected; admissible = entry_source_admitted;
      return entry_policy_observed;
    }
    bool discover_frame_generation_hooks() { discover_fg_options(); return install_fg_hooks(); }
    bool install(const versioning::number &version, const testing::targets &functions) {
      if (!middleware_requested() && !sunshine_upscaler_trace::enabled()) return false;
      // discover()'s selection for a version resource naming exactly this version.
      versioning::information resource;
      resource.fixed_valid = true;
      resource.file_ms = resource.product_ms = MAKELONG(version.minor, version.major);
      resource.file_ls = resource.product_ls = MAKELONG(version.build, version.patch);
      const abi native_version = versioning::classify(resource);
      const auto contract = native_version == abi::unsupported ? buffers::contract{} : buffers::for_version(version);
      const bool success = install_hooks(native_version, {functions.constants, functions.tag, functions.tag_for_frame, functions.evaluate, functions.new_frame_token, functions.get_feature_function}, contract);
      if (success) observing.store(middleware_requested(), std::memory_order_release);
      sunshine_upscaler_trace::set_streamline_coverage(success);
      return success;
    }
    counters counts() {
      return {constant_calls.load(), tag_calls.load(), evaluation_calls.load(), dropped.load(), invalid.load(), vram_warnings.load(),
        depth_untags.load()};
    }
    void lose_observation() { dropped_observation(loss_diagnostics::reason::test_injected, __func__, 4488, {}); }
    bool latest(std::uint32_t viewport, std::uint64_t &resource, bool &has_camera, bool &explicit_same_frame, bool &same_token_address) {
      auto pin = read_metadata();
      if (!pin) return false;
      bool found = false;
      for (const auto &record : pin->records) if (record.used && record.viewport == viewport) {
        const tag_record *chosen = nullptr;
        for (const auto &tag : record.global_tags) if (!chosen && tag.present) chosen = &tag;
        for (const auto &tag : record.local_tags) if (tag.present) { chosen = &tag; break; }
        const tag_record empty;
        const auto &tag = chosen ? *chosen : empty;
        resource = tag.value.native_resource;
        has_camera = record.has_camera;
        explicit_same_frame = tag.present && match(record.key, tag.value) == match_status::exact_frame;
        same_token_address = record.camera_token && record.camera_token == tag.token;
        found = tag.present;
      }
      return found;
    }
    void clear() {
      sunshine_game3d::ui_mask::invalidate_all();
      observing.store(false, std::memory_order_release);
      if (minhook_initialized) {
        // No other thread exists in the fixture; production never does this.
        MH_DisableHook(MH_ALL_HOOKS);
        MH_Uninitialize();
      }
      for (unsigned i = 0; i != fg_target_count; ++i) {
        if (fg_targets[i].owner) FreeLibrary(fg_targets[i].owner);
        fg_targets[i].entry = nullptr; fg_targets[i].owner = nullptr;
        fg_targets[i].original = nullptr; fg_targets[i].state = 0;
      }
      fg_target_count = 0; current_fg_target = nullptr; fg_available = false;
      fg_loss_revision = 0;
      // The process's FG view and this thread's Present snapshot: each
      // fixture case begins as a new process does.
      fg_state = {};
      present_snapshot = {};
      fg_store_read_seam = fg_transition_seam = nullptr;
      original_feature_function_v2 = nullptr;
      hooks_installed = minhook_initialized = false;
      v1_private_state_ready.store(false, std::memory_order_release);
      v1_private_state_checked = false;
      installed_abi = sunshine_streamline::abi::unsupported;
      installed_targets = {};
      constant_calls = tag_calls = evaluation_calls = dropped = invalid = vram_warnings = depth_untags = fg_refusals = 0;
      next_compatibility_report = next_untag_report = next_fg_refusal_report = 0;
      initialize(nullptr);
    }
  }
#endif
}

#include "sl_observation.inl"
#include "sl_nomination.inl"
#include "sl_frame_generation.inl"
#include "sl_ui_capture.inl"
#include "sl_report.inl"

#include "sl_hooks.inl"
