// SPDX-License-Identifier: GPL-3.0-only
#include "ngx_depth_source.h"
#include "addon_lifetime.h"
#ifndef SUNSHINE_UPSCALER_TRACE_TEST
#include "streamline_depth_capture.h"
#include "streamline_native_observer.h"
#endif
#include <reshade.hpp>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>

namespace sunshine_ngx {
  namespace {
    // Independent minimal ABI adapter; authority is NVIDIA/DLSS commit
    // 374959484e79a640feaba44c93ac8cfb0a03f5b5, include/nvsdk_ngx_params.h,
    // nvsdk_ngx_defs.h and nvsdk_ngx_helpers_d3d.h. The public C Get* wrappers
    // dispatch through the owning SDK's own C++ ABI. We never decode that ABI.
    constexpr std::uint32_t super_sampling = 1, inverted_depth = 1u << 3;
    constexpr unsigned max_features = 64;
    enum class availability { missing_getter, failed, empty, present };
    struct optional_pointer {
      availability state{availability::missing_getter};
      std::uintptr_t address{}; // Identity only, never retained or dereferenced.
    };
    struct calibration_observation {
      std::uint64_t sequence{}, depth{}, source_id{}, revision{};
      sunshine_scene_depth::extent area;
      bool reversed{};
      optional_pointer position, world_to_view, view_to_clip, inv_view_projection, clip_to_prev_clip;
    };
    struct feature {
      HMODULE owner{};
      const void *handle{};
      creation parameters;
      std::uint64_t generation{}, scene_revision{1};
    };
    SRWLOCK feature_lock = SRWLOCK_INIT;
    std::array<feature, max_features> features;
    std::atomic<std::uint64_t> active_epoch{}, next_epoch{}, sequence{}, next_feature{};
    std::atomic<std::uint64_t> evaluations{}, recorded{}, unknown{}, missing_parameters{}, failed{}, feature_overflow{};
    std::uint64_t next_report{};
    bool calibration_probe_requested{}; // Protected by feature_lock, like the feature records.
    struct calibration_slot {
      std::uint64_t source_id{}, next_tick{}, reported_sequence{};
      calibration_observation observation;
    };
    // Diagnostics must not introduce exclusive feature-lock ownership: capture
    // completion uses a shared try-lock and would mistake contention for loss.
    SRWLOCK calibration_lock = SRWLOCK_INIT;
    std::array<calibration_slot, max_features> calibration_slots;
#ifndef SUNSHINE_UPSCALER_TRACE_TEST
    struct capture_rejection {
      sunshine_streamline::depth_capture::record_diagnostic diagnostic;
      sunshine_scene_depth::extent area;
      std::uint64_t source_id{}, sequence{};
      bool retaining{};
    };
    SRWLOCK rejection_lock = SRWLOCK_INIT;
    capture_rejection last_rejection;
    std::atomic<std::uint64_t> rejected_captures{};
    std::uint64_t reported_rejections{};
    void remember_rejection(const sunshine_scene_depth::frame &value,
        const sunshine_streamline::depth_capture::record_diagnostic &diagnostic, bool retaining) {
      if (value.epoch != active_epoch.load(std::memory_order_acquire)) return;
      ++rejected_captures;
      if (!TryAcquireSRWLockExclusive(&rejection_lock)) return;
      if (value.epoch == active_epoch.load(std::memory_order_acquire))
        last_rejection = {diagnostic, value.resource.area, value.source_id, value.sequence, retaining};
      ReleaseSRWLockExclusive(&rejection_lock);
    }
#endif
#ifdef SUNSHINE_UPSCALER_TRACE_TEST
    testing::callbacks backend;
#endif
    struct preserve_error {
      DWORD value{GetLastError()};
      ~preserve_error() { SetLastError(value); }
    };
    bool success(result value) { return (value & 0xfff00000u) != 0xbad00000u; }
    const char *name(availability state) {
      switch (state) {
      case availability::missing_getter: return "missing-getter";
      case availability::failed: return "failed";
      case availability::empty: return "null";
      default: return "present";
      }
    }
    optional_pointer probe_pointer(decltype(parameter_api::resource) getter, const void *parameters, const char *key) {
      if (!getter) return {};
      void *pointer{};
      // A failed getter can leave garbage in its out parameter. Success with a
      // null pointer is a separate result: the game may populate optional keys
      // with null. Neither proves that this SDK can never supply the parameter.
      if (!success(getter(const_cast<void *>(parameters), key, &pointer))) return {availability::failed, 0};
      return {pointer ? availability::present : availability::empty, reinterpret_cast<std::uintptr_t>(pointer)};
    }
    calibration_observation probe_calibration(const parameter_api &api, const void *parameters,
        const sunshine_scene_depth::frame &value) {
      calibration_observation observation;
      observation.sequence = value.sequence;
      observation.depth = value.resource.native;
      observation.source_id = value.source_id;
      observation.revision = value.feedback.revision;
      observation.area = value.resource.area;
      observation.reversed = value.projection.reversed;
      // Position.ViewSpace is an optional SR research ID3D12Resource*. RR's
      // matrices are void-pointer parameters. Probe only their presence: no
      // assumed payload layout, GPU copy, lifetime extension or camera binding.
      observation.position = probe_pointer(api.resource, parameters, "Position.ViewSpace");
      observation.world_to_view = probe_pointer(api.void_pointer, parameters, "WorldToViewMatrix");
      observation.view_to_clip = probe_pointer(api.void_pointer, parameters, "ViewToClipMatrix");
      observation.inv_view_projection = probe_pointer(api.void_pointer, parameters, "InvViewProjectionMatrix");
      observation.clip_to_prev_clip = probe_pointer(api.void_pointer, parameters, "ClipToPrevClipMatrix");
      return observation;
    }
    void report_calibration(const calibration_observation &value) {
      char message[1100]{};
      std::snprintf(message, sizeof(message),
        "Sunshine NGX calibration probe: source=%llu sequence=%llu revision=%llu Depth=0x%llx area=%u,%u,%u,%u inverted=%u "
        "Position.ViewSpace=%s(0x%llx) WorldToViewMatrix=%s ViewToClipMatrix=%s InvViewProjectionMatrix=%s ClipToPrevClipMatrix=%s; "
        "presence only, optional pointers not read, no cross-API camera/depth association established",
        static_cast<unsigned long long>(value.source_id), static_cast<unsigned long long>(value.sequence),
        static_cast<unsigned long long>(value.revision), static_cast<unsigned long long>(value.depth),
        value.area.left, value.area.top, value.area.width, value.area.height, value.reversed ? 1 : 0,
        name(value.position.state), static_cast<unsigned long long>(value.position.address),
        name(value.world_to_view.state), name(value.view_to_clip.state),
        name(value.inv_view_projection.state), name(value.clip_to_prev_clip.state));
      reshade::log::message(reshade::log::level::info, message);
    }
    bool valid_dimension(unsigned value) { return value && value <= 16384; }
    bool get_uint(const parameter_api &api, const void *parameters, const char *key, unsigned &out) {
      return api.unsigned_integer && success(api.unsigned_integer(const_cast<void *>(parameters), key, &out));
    }
    bool read_pair(const parameter_api &api, const void *parameters, const char *key_x, const char *key_y,
        unsigned &x, unsigned &y, unsigned default_x, unsigned default_y) {
      x = y = 0;
      const bool have_x = get_uint(api, parameters, key_x, x), have_y = get_uint(api, parameters, key_y, y);
      if (!have_x && !have_y) { x = default_x; y = default_y; return true; }
      return have_x && have_y;
    }
    feature *find_feature(HMODULE owner, const void *handle, bool create) {
      feature *empty = nullptr;
      for (auto &value : features) {
        if (!value.handle) { if (!empty) empty = &value; continue; }
        if (value.owner == owner && value.handle == handle) return &value;
      }
      return create ? empty : nullptr;
    }
    void *export_pointer(HMODULE owner, const char *name) {
      auto *entry = reinterpret_cast<void *>(GetProcAddress(owner, name));
      MEMORY_BASIC_INFORMATION info{};
      HMODULE pinned{};
      if (!entry || !VirtualQuery(entry, &info, sizeof(info)) || info.Type != MEM_IMAGE ||
          info.State != MEM_COMMIT || (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) ||
          !(info.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)) ||
          !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
            reinterpret_cast<LPCWSTR>(entry), &pinned)) return nullptr;
      return entry;
    }
  }
  parameter_api resolve_parameter_api(HMODULE owner) {
    preserve_error error;
    parameter_api value;
    if (sunshine_addon_lifetime::stopping() || !owner) return value;
    value.integer = reinterpret_cast<decltype(value.integer)>(export_pointer(owner, "NVSDK_NGX_Parameter_GetI"));
    value.unsigned_integer = reinterpret_cast<decltype(value.unsigned_integer)>(export_pointer(owner, "NVSDK_NGX_Parameter_GetUI"));
    value.resource = reinterpret_cast<decltype(value.resource)>(export_pointer(owner, "NVSDK_NGX_Parameter_GetD3d12Resource"));
    value.void_pointer = reinterpret_cast<decltype(value.void_pointer)>(export_pointer(owner, "NVSDK_NGX_Parameter_GetVoidPointer"));
    value.floating = reinterpret_cast<decltype(value.floating)>(export_pointer(owner, "NVSDK_NGX_Parameter_GetF"));
    return value;
  }
  void initialize(bool requested, bool calibration_probe) {
    if (sunshine_addon_lifetime::stopping()) return;
    preserve_error error;
    active_epoch.store(0, std::memory_order_release);
    AcquireSRWLockExclusive(&feature_lock);
    features = {};
    calibration_probe_requested = calibration_probe;
    AcquireSRWLockExclusive(&calibration_lock);
    calibration_slots = {};
    ReleaseSRWLockExclusive(&calibration_lock);
    evaluations = recorded = unknown = missing_parameters = failed = feature_overflow = 0;
    next_report = 0;
#ifndef SUNSHINE_UPSCALER_TRACE_TEST
    rejected_captures = reported_rejections = 0;
    AcquireSRWLockExclusive(&rejection_lock);
    last_rejection = {};
    ReleaseSRWLockExclusive(&rejection_lock);
#endif
    if (requested) active_epoch.store(++next_epoch, std::memory_order_release);
    ReleaseSRWLockExclusive(&feature_lock);
  }
  void shutdown() { active_epoch.store(0, std::memory_order_release); }
  bool enabled() { return !sunshine_addon_lifetime::stopping() && active_epoch.load(std::memory_order_acquire) != 0; }
  std::uint64_t epoch() { return sunshine_addon_lifetime::stopping() ? 0 : active_epoch.load(std::memory_order_acquire); }
  creation before_create(const parameter_api &api, std::uint32_t feature_id, const void *parameters) {
    preserve_error error;
    creation value;
    value.epoch = epoch();
    value.feature = feature_id;
    if (!value.epoch || feature_id != super_sampling || !api || !parameters) return value;
    value.valid = success(api.integer(const_cast<void *>(parameters), "DLSS.Feature.Create.Flags", &value.flags)) &&
      get_uint(api, parameters, "Width", value.width) && get_uint(api, parameters, "Height", value.height) &&
      valid_dimension(value.width) && valid_dimension(value.height);
    return value;
  }
  void after_create(HMODULE owner, const void *handle, const creation &value, bool successful) {
    preserve_error error;
    if (sunshine_addon_lifetime::stopping() || !value.epoch || value.epoch != active_epoch.load(std::memory_order_acquire) || !successful) return;
    // Rare lifecycle mutations cannot be dropped: one missed successful create
    // would leave the feature unavailable until the game recreates DLSS. This
    // lock covers only bounded POD metadata, never a getter, COM call or GPU work.
    AcquireSRWLockExclusive(&feature_lock);
    if (value.epoch == active_epoch.load(std::memory_order_acquire)) {
      if (!handle) features = {}; // Unreadable output cannot preserve old identities.
      else if (auto *entry = find_feature(owner, handle, value.valid)) {
        if (value.valid) *entry = {owner, handle, value, ++next_feature, 1};
        else *entry = {};
      } else if (value.valid) ++feature_overflow;
    }
    ReleaseSRWLockExclusive(&feature_lock);
  }
  void after_release(HMODULE owner, const void *handle, std::uint64_t epoch, bool successful) {
    preserve_error error;
    if (sunshine_addon_lifetime::stopping() || !epoch || epoch != active_epoch.load(std::memory_order_acquire) || !successful) return;
    std::uint64_t generation{};
    AcquireSRWLockExclusive(&feature_lock);
    if (epoch == active_epoch.load(std::memory_order_acquire)) {
      if (auto *entry = find_feature(owner, handle, false)) {
        generation = entry->generation;
        *entry = {};
      }
    }
    ReleaseSRWLockExclusive(&feature_lock);
    if (!generation) return;
#ifdef SUNSHINE_UPSCALER_TRACE_TEST
    if (backend.retire) backend.retire(epoch, generation);
#else
    sunshine_streamline::depth_capture::retire_source(sunshine_scene_depth::provider_kind::ngx, epoch, generation);
#endif
  }
  evaluation before_evaluate(HMODULE owner, const parameter_api &api, std::uint64_t command,
      const void *handle, const void *parameters) {
    preserve_error error;
    evaluation attempt;
    attempt.epoch = epoch();
    if (!attempt.epoch || !api || !parameters || !command || !handle) return attempt;
    feature selected;
    unsigned selected_slot{};
    // Missing a read is one missed frame, not evidence that feature lifetimes
    // changed. In particular UI polling must never erase valid feature IDs.
    if (!TryAcquireSRWLockShared(&feature_lock)) return attempt;
    if (const auto *entry = find_feature(owner, handle, false)) {
      selected = *entry;
      selected_slot = static_cast<unsigned>(entry - features.data());
    }
    ReleaseSRWLockShared(&feature_lock);
    if (!selected.handle || selected.parameters.epoch != attempt.epoch) { ++unknown; return attempt; }
    attempt.observed = true;
    attempt.source_id = selected.generation;
    ++evaluations;
    sunshine_scene_depth::frame value;
    value.provider = sunshine_scene_depth::provider_kind::ngx;
    value.source_id = selected.generation;
    value.epoch = attempt.epoch;
    int reset{};
    const bool reset_available = success(api.integer(const_cast<void *>(parameters), "Reset", &reset));
    bool probe_requested = false;
    // A reset pulse must survive frames that produce no depth/readback. Commit
    // it and the observation sequence together after calling the game getter:
    // concurrent getter completion must not stamp an older scene revision onto
    // a newer sequence. This lock contains POD only, never a getter or GPU call.
    AcquireSRWLockExclusive(&feature_lock);
    if (auto *entry = find_feature(owner, handle, false); entry &&
        entry->generation == selected.generation && entry->parameters.epoch == attempt.epoch &&
        attempt.epoch == active_epoch.load(std::memory_order_acquire)) {
      if (reset_available && reset != 0) ++entry->scene_revision;
      value.sequence = ++sequence;
      value.feedback.revision = entry->scene_revision;
      value.feedback.reset = reset_available && reset != 0;
      probe_requested = calibration_probe_requested;
    }
    ReleaseSRWLockExclusive(&feature_lock);
    if (!value.sequence) return attempt;
    value.tick = GetTickCount64();
    value.projection.reversed = (selected.parameters.flags & inverted_depth) != 0;
    value.projection.direction_supplied = true;
    // NGX names the resource, but does not supply its current D3D12 state.
    // The shared copy owner decides whether its own state evidence is usable.
    value.valid_until = sunshine_scene_depth::lifetime::until_evaluation;
#ifndef SUNSHINE_UPSCALER_TRACE_TEST
    using namespace sunshine_streamline;
    depth_capture::begin_evaluation(value.epoch, value.sequence, value.viewport, value.provider, value.source_id);
#endif
    void *resource{};
    auto &area = value.resource.area;
    const bool valid = success(api.resource(const_cast<void *>(parameters), "Depth", &resource)) && resource &&
      read_pair(api, parameters, "DLSS.Render.Subrect.Dimensions.Width", "DLSS.Render.Subrect.Dimensions.Height",
        area.width, area.height, selected.parameters.width, selected.parameters.height) &&
      read_pair(api, parameters, "DLSS.Input.Depth.Subrect.Base.X", "DLSS.Input.Depth.Subrect.Base.Y", area.left, area.top, 0, 0) &&
      valid_dimension(area.width) && valid_dimension(area.height) && area.left < 16384 && area.top < 16384;
    if (!valid) { ++missing_parameters; return attempt; }
    value.resource.native = reinterpret_cast<std::uint64_t>(resource);
    if (api.floating) {
      float jitter_x{}, jitter_y{};
      const bool have_x = success(api.floating(const_cast<void *>(parameters), "Jitter.Offset.X", &jitter_x));
      const bool have_y = success(api.floating(const_cast<void *>(parameters), "Jitter.Offset.Y", &jitter_y));
      // NVIDIA's offsets are input/render pixels, not output pixels or resource-allocation
      // dimensions. Copy only this evaluation's complete finite pair alongside its depth.
      // Missing jitter does not invalidate a usable depth nomination.
      if (have_x && have_y && std::isfinite(jitter_x) && std::isfinite(jitter_y))
        value.jitter = {jitter_x, jitter_y, area.width, area.height, true};
    }
    if (probe_requested) {
      // Only a usable depth input consumes a probe slot. A loading frame with
      // no depth must not delay the first valid gameplay observation.
      probe_requested = false;
      if (TryAcquireSRWLockExclusive(&calibration_lock)) {
        auto &slot = calibration_slots[selected_slot];
        if (value.epoch == active_epoch.load(std::memory_order_acquire) && slot.source_id <= value.source_id) {
          if (slot.source_id != value.source_id) slot = {value.source_id, 0, 0, {}};
          if (value.tick >= slot.next_tick) {
            slot.next_tick = value.tick + 5000;
            probe_requested = true;
          }
        }
        ReleaseSRWLockExclusive(&calibration_lock);
      }
    }
    if (probe_requested) {
      const auto observation = probe_calibration(api, parameters, value);
      // Getter calls run outside locks. Revalidate the epoch and generation
      // before publishing so release/recreate cannot inherit stale diagnostics.
      if (TryAcquireSRWLockShared(&feature_lock)) {
        const auto &entry = features[selected_slot];
        if (entry.generation == selected.generation && value.epoch == active_epoch.load(std::memory_order_acquire) &&
            TryAcquireSRWLockExclusive(&calibration_lock)) {
          auto &slot = calibration_slots[selected_slot];
          if (slot.source_id == value.source_id && observation.sequence > slot.observation.sequence)
            slot.observation = observation;
          ReleaseSRWLockExclusive(&calibration_lock);
        }
        ReleaseSRWLockShared(&feature_lock);
      }
    }
#ifdef SUNSHINE_UPSCALER_TRACE_TEST
    if (backend.record) attempt.ticket = backend.record(command, value);
#else
    depth_capture::input input;
    static_cast<sunshine_scene_depth::frame &>(input) = value;
    depth_capture::record_diagnostic diagnostic;
    input.source = depth_capture::retain_source(value.resource.native, &diagnostic);
    input.source_present_generation = depth_capture::source_present_generation(input.source);
    if (input.source) {
      attempt.ticket = depth_capture::nominate(command, input, &diagnostic);
      if (!attempt.ticket) remember_rejection(value, diagnostic, false);
    } else {
      // Preserve the existing admission/status path; retain_source's own
      // diagnostic explains why the subsequent missing-source record fails.
      attempt.ticket = depth_capture::nominate(command, input);
      remember_rejection(value, diagnostic, true);
    }
#endif
    if (attempt.ticket) ++recorded;
    return attempt;
  }
  void after_evaluate(const evaluation &value, bool successful) {
    preserve_error error;
    // An original vendor call can return after detach began. Its old ticket
    // must not reenter the capture owner's already destructing containers.
    if (sunshine_addon_lifetime::stopping() || !value.observed) return;
    bool current = false;
    if (value.epoch == active_epoch.load(std::memory_order_acquire) && TryAcquireSRWLockShared(&feature_lock)) {
      for (const auto &entry : features) {
        if (entry.handle && entry.generation == value.source_id && entry.parameters.epoch == value.epoch) {
          current = true;
          break;
        }
      }
      ReleaseSRWLockShared(&feature_lock);
    }
    if (!successful || !current) ++failed;
    if (!value.ticket) return;
#ifdef SUNSHINE_UPSCALER_TRACE_TEST
    if (backend.finish) backend.finish(value.ticket, successful && current);
#else
    sunshine_streamline::depth_capture::finish(value.ticket, successful && current);
#endif
  }
  void poll() {
    preserve_error error;
    if (!enabled()) return;
    const auto now = GetTickCount64();
    if (now < next_report) return;
    next_report = now + 5000;
    unsigned count = 0;
    std::array<calibration_observation, max_features> probes;
    unsigned probe_count = 0;
    AcquireSRWLockShared(&feature_lock);
    const bool read_probes = calibration_probe_requested && TryAcquireSRWLockExclusive(&calibration_lock);
    for (unsigned i = 0; i < features.size(); ++i) {
      const auto &value = features[i];
      count += value.handle != nullptr;
      if (read_probes) {
        auto &slot = calibration_slots[i];
        if (value.handle && slot.source_id == value.generation && slot.observation.sequence > slot.reported_sequence) {
          probes[probe_count++] = slot.observation;
          slot.reported_sequence = slot.observation.sequence;
        }
      }
    }
    if (read_probes) ReleaseSRWLockExclusive(&calibration_lock);
    ReleaseSRWLockShared(&feature_lock);
    for (unsigned i = 0; i < probe_count; ++i) report_calibration(probes[i]);
    char message[400]{};
    std::snprintf(message, sizeof(message),
      "Sunshine NGX depth: confirmed_features=%u evaluations=%llu recorded=%llu unknown_feature=%llu missing_parameters=%llu failed=%llu feature_capacity_loss=%llu; D3D12 DLSS, named parameter exports, shared copy owner",
      count, static_cast<unsigned long long>(evaluations.load()), static_cast<unsigned long long>(recorded.load()),
      static_cast<unsigned long long>(unknown.load()), static_cast<unsigned long long>(missing_parameters.load()),
      static_cast<unsigned long long>(failed.load()), static_cast<unsigned long long>(feature_overflow.load()));
    reshade::log::message(reshade::log::level::info, message);
#ifndef SUNSHINE_UPSCALER_TRACE_TEST
    const auto rejected_count = rejected_captures.load(std::memory_order_relaxed);
    if (rejected_count != reported_rejections && TryAcquireSRWLockShared(&rejection_lock)) {
      const auto snapshot = last_rejection;
      ReleaseSRWLockShared(&rejection_lock);
      if (snapshot.sequence) {
        reported_rejections = rejected_count;
        const auto &d = snapshot.diagnostic;
        const auto observer = sunshine_streamline::native_observer::counts();
        char rejected[1600]{};
        std::snprintf(rejected, sizeof(rejected),
          "Sunshine NGX capture rejection: count=%llu operation=%s result=%s stage=%s loss=%s source=%llu sequence=%llu command=0x%llx cookie=%llu type=%u resource=0x%llx dimensions=%ux%u format=%u flags=0x%x dimension=%u mips=%u layers=%u samples=%u area=%u,%u,%u,%u state=0x%x observed_state=0x%x observed=%u blocked=%u closed=%u invalid=%u render_pass=%u generation=%llu/%llu device=%llu/%llu; observer calls=%llu observed=%llu unreadable=%llu dropped=%llu installed=%llu/%llu rejected=%llu barrier_overflow=%llu submission_overflow=%llu discovery_contention=%llu",
          static_cast<unsigned long long>(rejected_count), snapshot.retaining ? "retain_source" : "nominate",
          sunshine_streamline::depth_capture::name(d.result), sunshine_streamline::depth_capture::name(d.stage),
          sunshine_streamline::depth_capture::name(d.loss), static_cast<unsigned long long>(snapshot.source_id),
          static_cast<unsigned long long>(snapshot.sequence), static_cast<unsigned long long>(d.command),
          static_cast<unsigned long long>(d.recording_cookie), d.command_type, static_cast<unsigned long long>(d.resource),
          d.width, d.height, d.format, d.flags, d.dimension, d.mip_levels, d.array_size, d.samples,
          snapshot.area.left, snapshot.area.top, snapshot.area.width, snapshot.area.height,
          d.native_state, d.observed_state, d.observed ? 1 : 0, d.blocked ? 1 : 0, d.recording_closed ? 1 : 0,
          d.recording_invalid ? 1 : 0, d.render_pass ? 1 : 0,
          static_cast<unsigned long long>(d.expected_generation), static_cast<unsigned long long>(d.current_generation),
          static_cast<unsigned long long>(d.expected_device_identity), static_cast<unsigned long long>(d.device_identity),
          static_cast<unsigned long long>(observer.calls), static_cast<unsigned long long>(observer.observed),
          static_cast<unsigned long long>(observer.unreadable), static_cast<unsigned long long>(observer.dropped),
          static_cast<unsigned long long>(observer.installed), static_cast<unsigned long long>(observer.targets),
          static_cast<unsigned long long>(observer.rejected), static_cast<unsigned long long>(observer.barrier_overflow),
          static_cast<unsigned long long>(observer.submission_overflow), static_cast<unsigned long long>(observer.discovery_contention));
        reshade::log::message(reshade::log::level::info, rejected);
      }
    }
#endif
  }
#ifdef SUNSHINE_UPSCALER_TRACE_TEST
  namespace testing { void set_callbacks(callbacks value) { backend = value; } }
#endif
}
