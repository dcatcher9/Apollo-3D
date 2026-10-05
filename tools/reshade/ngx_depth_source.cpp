// SPDX-License-Identifier: GPL-3.0-only
#include "ngx_depth_source.h"
#include "addon_lifetime.h"
#include "game3d_diagnostic_metadata.h"
#include "streamline_depth_capture.h"
#ifndef SUNSHINE_UPSCALER_TRACE_TEST
#include "streamline_native_observer.h"
#endif
#include <reshade.hpp>
#include "async_log.h"
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace sunshine_ngx {
  namespace {
    // Independent minimal ABI adapter; authority is NVIDIA/DLSS commit
    // 374959484e79a640feaba44c93ac8cfb0a03f5b5, include/nvsdk_ngx_params.h,
    // nvsdk_ngx_defs.h and nvsdk_ngx_helpers_d3d.h. The public C Get* wrappers
    // dispatch through the owning SDK's own C++ ABI. We never decode that ABI.
    constexpr std::uint32_t super_sampling = 1, inverted_depth = 1u << 3;
    // D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE. NVIDIA's DLSS Programming
    // Guide, section 3.4 "Resource States": every D3D12 input must be in this
    // state at the evaluation call, and DLSS always restores it afterwards.
    constexpr std::uint32_t evaluation_input_state = 0x40;
    // Feature ID for a registration recovered at evaluation: CreateFeature ran
    // before these hooks or before the add-on's current epoch.
    constexpr std::uint32_t recovered_feature = 0xfffffffeu;
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
    // Handles released in this epoch; evaluation-time recovery never revives
    // one. A later successful CreateFeature for the same address clears it.
    struct released_handle { HMODULE owner{}; const void *handle{}; };
    std::array<released_handle, 16> released_handles;
    unsigned next_released{};
    bool released(HMODULE owner, const void *handle) {
      for (const auto &value : released_handles) if (value.handle == handle && value.owner == owner) return true;
      return false;
    }
    void forget_release(HMODULE owner, const void *handle) {
      for (auto &value : released_handles) if (value.handle == handle && value.owner == owner) value = {};
    }
    std::atomic<std::uint64_t> active_epoch{}, next_epoch{}, sequence{}, next_feature{};
    // Dump-only observations must not advance the capture/evaluation ordering
    // counter when the feature was never admitted or depth belongs to SL.
    std::atomic<std::uint64_t> unknown_diagnostic_sequence{};
    std::atomic<std::uint64_t> evaluations{}, nominations{}, copy_recorded{}, metadata_only{},
      unknown{}, missing_parameters{}, failed{}, feature_overflow{}, recovered{};
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
    struct capture_rejection {
      sunshine_streamline::depth_capture::record_diagnostic diagnostic;
      sunshine_scene_depth::extent area;
      std::uint64_t source_id{}, sequence{}, nomination_ticket{};
      bool retaining{};
    };
    SRWLOCK rejection_lock = SRWLOCK_INIT;
    capture_rejection last_rejection;
    std::atomic<std::uint64_t> rejected_captures{};
#ifndef SUNSHINE_UPSCALER_TRACE_TEST
    std::uint64_t reported_rejections{};
#endif
    void remember_rejection(const sunshine_scene_depth::frame &value,
        const sunshine_streamline::depth_capture::record_diagnostic &diagnostic, bool retaining,
        std::uint64_t nomination_ticket) {
      if (value.epoch != active_epoch.load(std::memory_order_acquire)) return;
      ++rejected_captures;
      if (!TryAcquireSRWLockExclusive(&rejection_lock)) return;
      if (value.epoch == active_epoch.load(std::memory_order_acquire))
        last_rejection = {diagnostic, value.resource.area, value.source_id, value.sequence, nomination_ticket, retaining};
      ReleaseSRWLockExclusive(&rejection_lock);
    }
#ifdef SUNSHINE_UPSCALER_TRACE_TEST
    testing::callbacks backend;
#endif
    struct preserve_error {
      DWORD value{GetLastError()};
      ~preserve_error() { SetLastError(value); }
    };
    bool success(result value) { return (value & 0xfff00000u) != 0xbad00000u; }
    sunshine_game3d::diagnostic::stamp observe_dump_parameters(const parameter_api &api, const void *parameters,
        const sunshine_scene_depth::frame &frame, const feature &selected,
        const void *handle, std::uint64_t command, std::uint64_t diagnostic_session,
        bool capture_suppressed_by_depth_owner = false) {
      namespace diagnostic = sunshine_game3d::diagnostic;
      using kind = diagnostic::parameter_type;
      diagnostic::ngx_evaluation out;
      out.observation = {diagnostic_session, frame.epoch, frame.sequence, frame.tick, 0, 0, command, UINT32_MAX, false};
      out.owner = reinterpret_cast<std::uint64_t>(selected.owner); out.handle = reinterpret_cast<std::uint64_t>(handle);
      out.source_id = selected.parameters.valid && !capture_suppressed_by_depth_owner ? selected.generation : 0;
      out.scene_revision = frame.feedback.revision; out.feature_known = selected.handle != nullptr;
      out.feature = selected.parameters.feature; out.create_width = selected.parameters.width;
      out.create_height = selected.parameters.height; out.create_flags = selected.parameters.flags;
      out.metadata_unavailable = selected.handle && !selected.parameters.valid;
      out.capture_suppressed_by_depth_owner = capture_suppressed_by_depth_owner;
      const auto probe = [&](const char *name, kind type) {
        if (out.parameter_count == out.parameters.size()) { out.truncated = true; return; }
        auto &p = out.parameters[out.parameter_count++];
        std::snprintf(p.name, sizeof(p.name), "%s", name); p.type = type;
        void *pointer{}; unsigned uint_value{}; int int_value{}; float float_value{};
        switch (type) {
          case kind::resource: case kind::pointer: {
            const auto getter = type == kind::resource ? api.resource : api.void_pointer;
            p.getter_available = getter != nullptr;
            if (getter) p.result = getter(const_cast<void *>(parameters), name, &pointer);
            p.integer = reinterpret_cast<std::uint64_t>(pointer); break;
          }
          case kind::unsigned_integer:
            p.getter_available = api.unsigned_integer != nullptr;
            if (api.unsigned_integer) p.result = api.unsigned_integer(const_cast<void *>(parameters), name, &uint_value);
            p.integer = uint_value; break;
          case kind::signed_integer:
            p.getter_available = api.integer != nullptr;
            if (api.integer) p.result = api.integer(const_cast<void *>(parameters), name, &int_value);
            p.integer = static_cast<std::int64_t>(int_value); break;
          case kind::floating:
            p.getter_available = api.floating != nullptr;
            if (api.floating) p.result = api.floating(const_cast<void *>(parameters), name, &float_value);
            p.number = float_value; break;
        }
        p.successful = p.getter_available && success(p.result);
        // Never publish an out parameter left untouched/poisoned by failure.
        if (!p.successful) { p.integer = 0; p.number = 0; }
      };
      // Bounded public named-parameter list, independently checked against
      // NVIDIA/DLSS nvsdk_ngx_defs.h. Availability is a getter result, never an
      // inference from a loaded DLL. Optional pointer payloads remain opaque.
      for (const char *name : {"Color", "Output", "Depth", "DepthHighRes", "MotionVectors", "ExposureTexture", "Position.ViewSpace"}) probe(name, kind::resource);
      for (const auto &entry : sunshine_game3d::ui_resources::catalog) {
        if (entry.source == sunshine_game3d::ui_resources::provider::ngx) probe(entry.parameter_key, kind::resource);
      }
      for (const auto &entry : sunshine_game3d::ui_resources::catalog) {
        if (!entry.parameter_subrect_prefix) continue;
        for (const auto *suffix : {"BaseX", "BaseY", "Width", "Height"}) {
          char key[64]{};
          std::snprintf(key, sizeof(key), "%s%s", entry.parameter_subrect_prefix, suffix);
          probe(key, kind::unsigned_integer);
        }
      }
      // A boolean HDR flag alone does not identify PQ vs scRGB; preserve it as
      // evidence without assigning a transfer function to the captured pixels.
      probe("DLSSG.ColorBuffersHDR", kind::unsigned_integer);
      for (const char *name : {"WorldToViewMatrix", "ViewToClipMatrix", "InvViewProjectionMatrix", "ClipToPrevClipMatrix"}) probe(name, kind::pointer);
      for (const char *name : {"DLSS.Use.HW.Depth", "Width", "Height", "OutWidth", "OutHeight", "Color.Format", "Output.Format",
          "DLSS.Render.Subrect.Dimensions.Width", "DLSS.Render.Subrect.Dimensions.Height",
          "DLSS.Input.Color.Subrect.Base.X", "DLSS.Input.Color.Subrect.Base.Y", "DLSS.Input.Depth.Subrect.Base.X", "DLSS.Input.Depth.Subrect.Base.Y",
          "DLSS.Input.MV.Subrect.Base.X", "DLSS.Input.MV.Subrect.Base.Y", "DLSS.Output.Subrect.Base.X", "DLSS.Output.Subrect.Base.Y"}) probe(name, kind::unsigned_integer);
      for (const char *name : {"Reset", "DLSS.Feature.Create.Flags", "PerfQualityValue"}) probe(name, kind::signed_integer);
      for (const char *name : {"Jitter.Offset.X", "Jitter.Offset.Y", "MV.Scale.X", "MV.Scale.Y", "MV.Offset.X", "MV.Offset.Y",
          "DLSS.Pre.Exposure", "DLSS.Exposure.Scale", "FrameTimeDeltaInMsec"}) probe(name, kind::floating);
      diagnostic::observe_ngx(out);
      return out.observation;
    }
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
      sunshine_log::message(reshade::log::level::info, message);
    }
    bool valid_dimension(unsigned value) { return value && value <= 16384; }
    bool get_uint(const parameter_api &api, const void *parameters, const char *key, unsigned &out) {
      return api.unsigned_integer && success(api.unsigned_integer(const_cast<void *>(parameters), key, &out));
    }
    // zero_unset: a pair reported as 0,0 means "not set", exactly like absent
    // keys (NGX's evaluation helpers always write the render subrect, zero
    // when the game renders to the whole input).
    bool read_pair(const parameter_api &api, const void *parameters, const char *key_x, const char *key_y,
        unsigned &x, unsigned &y, unsigned default_x, unsigned default_y, bool zero_unset = false) {
      x = y = 0;
      const bool have_x = get_uint(api, parameters, key_x, x), have_y = get_uint(api, parameters, key_y, y);
      if ((!have_x && !have_y) || (zero_unset && have_x && have_y && !x && !y)) { x = default_x; y = default_y; return true; }
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
    sunshine_game3d::diagnostic::observe_module(owner, "ngx", "named C parameter exports");
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
    released_handles = {}; next_released = 0;
    calibration_probe_requested = calibration_probe;
    AcquireSRWLockExclusive(&calibration_lock);
    calibration_slots = {};
    ReleaseSRWLockExclusive(&calibration_lock);
    evaluations = nominations = copy_recorded = metadata_only = unknown = missing_parameters = failed = feature_overflow = recovered = 0;
    next_report = 0;
    rejected_captures = 0;
#ifndef SUNSHINE_UPSCALER_TRACE_TEST
    reported_rejections = 0;
#endif
    AcquireSRWLockExclusive(&rejection_lock);
    last_rejection = {};
    ReleaseSRWLockExclusive(&rejection_lock);
    if (requested) active_epoch.store(++next_epoch, std::memory_order_release);
    ReleaseSRWLockExclusive(&feature_lock);
  }
  void shutdown() { active_epoch.store(0, std::memory_order_release); }
  bool enabled() { return !sunshine_addon_lifetime::stopping() && active_epoch.load(std::memory_order_acquire) != 0; }
  std::uint64_t epoch() { return sunshine_addon_lifetime::stopping() ? 0 : active_epoch.load(std::memory_order_acquire); }
  namespace {
    // device_default: the feature's documented depth input is device depth when
    // DLSS.Use.HW.Depth is absent. That holds for Super Resolution/DLAA only;
    // Ray Reconstruction's helpers always write the explicit key.
    creation parse_creation(const parameter_api &api, std::uint32_t feature_id, const void *parameters, bool device_default) {
      creation value;
      value.epoch = epoch();
      value.feature = feature_id;
      if (!value.epoch || !api || !parameters) return value;
      unsigned hardware_depth{};
      const bool explicit_encoding = get_uint(api, parameters, "DLSS.Use.HW.Depth", hardware_depth);
      if (explicit_encoding) {
        if (hardware_depth > 1) return value;
        value.encoding = hardware_depth ? sunshine_scene_depth::depth_encoding::device :
          sunshine_scene_depth::depth_encoding::linear_distance;
      } else if (!device_default) {
        // Every feature is registered. SR's documented input is device depth;
        // other features need an explicit encoding before pixels can be used.
        return value;
      }
      value.valid = success(api.integer(const_cast<void *>(parameters), "DLSS.Feature.Create.Flags", &value.flags)) &&
        get_uint(api, parameters, "Width", value.width) && get_uint(api, parameters, "Height", value.height) &&
        valid_dimension(value.width) && valid_dimension(value.height);
      return value;
    }
    // NGX's create helpers write Width/Height/Create.Flags (and RR's explicit
    // depth encoding) into the parameter map that later evaluations reuse. When
    // CreateFeature was missed, register the live handle from that map once.
    // Only SR/DLAA evaluates a "Depth" input without the explicit encoding key,
    // so the SR default is applied; a map without creation values stays unknown.
    feature recover_feature(HMODULE owner, const void *handle, const parameter_api &api, const void *parameters,
        std::uint64_t epoch, unsigned &slot) {
      auto value = parse_creation(api, recovered_feature, parameters, true);
      if (!value.valid || value.epoch != epoch) return {};
      feature result;
      AcquireSRWLockExclusive(&feature_lock);
      if (epoch == active_epoch.load(std::memory_order_acquire) && !released(owner, handle)) {
        if (auto *entry = find_feature(owner, handle, true)) {
          // A concurrent create or recovery for this handle keeps its registration.
          if (entry->handle != handle || entry->parameters.epoch != epoch) {
            *entry = {owner, handle, value, ++next_feature, 1};
            ++recovered;
          }
          result = *entry;
          slot = static_cast<unsigned>(entry - features.data());
        } else ++feature_overflow;
      }
      ReleaseSRWLockExclusive(&feature_lock);
      return result;
    }
  }
  creation before_create(const parameter_api &api, std::uint32_t feature_id, const void *parameters) {
    preserve_error error;
    return parse_creation(api, feature_id, parameters, feature_id == super_sampling);
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
      else if (auto *entry = find_feature(owner, handle, true))
        *entry = {owner, handle, value, ++next_feature, 1};
      else ++feature_overflow;
      if (handle) forget_release(owner, handle);
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
      if (handle && !released(owner, handle)) {
        released_handles[next_released] = {owner, handle};
        next_released = (next_released + 1) % released_handles.size();
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
      const void *handle, const void *parameters, bool capture_depth, bool observe_diagnostics) {
    preserve_error error;
    evaluation attempt;
    attempt.epoch = epoch();
    if (!attempt.epoch || !api || !parameters || !command || !handle) return attempt;
    const auto diagnostic_session = observe_diagnostics ? sunshine_game3d::diagnostic_metadata_generation() : 0;
    // An enclosing SL owner still owns all depth work. Only an explicitly armed
    // dump may query this nested NGX call, without committing resets, advancing
    // source ordering or invoking optional-resource copy callbacks.
    if (!capture_depth && !diagnostic_session) return attempt;
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
    // Dump-only observation of a call that yields no capture: its own sequence
    // domain, the observed owner/handle, and no invented capture metadata.
    const auto observe_without_capture = [&](feature observed, bool suppressed) {
      observed.owner = owner;
      sunshine_scene_depth::frame diagnostic_frame;
      diagnostic_frame.epoch = attempt.epoch;
      diagnostic_frame.sequence = ++unknown_diagnostic_sequence;
      diagnostic_frame.tick = GetTickCount64();
      attempt.diagnostic_observation = observe_dump_parameters(api, parameters, diagnostic_frame, observed, handle, command,
        diagnostic_session, suppressed);
    };
    if (!capture_depth) {
      if (selected.parameters.epoch != attempt.epoch) selected = {};
      attempt.capture_suppressed_by_depth_owner = true;
      observe_without_capture(selected, true);
      return attempt;
    }
    if (!selected.handle || selected.parameters.epoch != attempt.epoch)
      selected = recover_feature(owner, handle, api, parameters, attempt.epoch, selected_slot);
    if (!selected.handle || selected.parameters.epoch != attempt.epoch || !selected.parameters.valid) {
      if (!selected.handle || selected.parameters.epoch != attempt.epoch) { ++unknown; selected = {}; }
      else ++missing_parameters;
      // An unrecoverable or incomplete feature still reports its named parameters.
      if (diagnostic_session) observe_without_capture(selected, false);
      return attempt;
    }
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
    if (diagnostic_session)
      attempt.diagnostic_observation = observe_dump_parameters(api, parameters, value, selected, handle, command, diagnostic_session);
    value.projection.encoding = selected.parameters.encoding;
    value.projection.reversed = value.projection.encoding == sunshine_scene_depth::depth_encoding::device &&
      (selected.parameters.flags & inverted_depth) != 0;
    value.projection.direction_supplied = true;
    // NGX names the resource but no per-call state. Its SDK contract fixes the
    // input state at this boundary; a state observed on the same recording
    // still takes precedence in the shared copy owner. Engines commonly record
    // that transition on an earlier command list, where it cannot be observed.
    value.native_state = evaluation_input_state;
    value.proof = sunshine_scene_depth::state_proof::sdk_contract;
    value.valid_until = sunshine_scene_depth::lifetime::until_evaluation;
#ifndef SUNSHINE_UPSCALER_TRACE_TEST
    using namespace sunshine_streamline;
    depth_capture::begin_evaluation(value.epoch, value.sequence, value.viewport, value.provider, value.source_id);
#endif
    void *resource{};
    auto &area = value.resource.area;
    const bool valid = success(api.resource(const_cast<void *>(parameters), "Depth", &resource)) && resource &&
      read_pair(api, parameters, "DLSS.Render.Subrect.Dimensions.Width", "DLSS.Render.Subrect.Dimensions.Height",
        area.width, area.height, selected.parameters.width, selected.parameters.height, true) &&
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
    sunshine_streamline::depth_capture::record_diagnostic diagnostic;
    bool retaining = false;
#ifdef SUNSHINE_UPSCALER_TRACE_TEST
    if (backend.record) attempt.ticket = backend.record(command, value);
    attempt.capture_authority = attempt.ticket && (!backend.capture_recorded || backend.capture_recorded(attempt.ticket));
    diagnostic.result = attempt.capture_authority ? sunshine_streamline::depth_capture::status::recorded :
      sunshine_streamline::depth_capture::status::unavailable;
    if (backend.describe_record) {
      backend.describe_record(attempt.ticket, diagnostic);
      attempt.capture_authority = attempt.ticket && diagnostic.result == sunshine_streamline::depth_capture::status::recorded;
    }
#else
    depth_capture::input input;
    static_cast<sunshine_scene_depth::frame &>(input) = value;
    input.source = depth_capture::retain_source(value.resource.native, &diagnostic);
    input.source_present_generation = depth_capture::source_present_generation(input.source);
    if (input.source) {
      attempt.ticket = depth_capture::nominate(command, input, &diagnostic);
      attempt.capture_authority = attempt.ticket && diagnostic.result == depth_capture::status::recorded;
    } else {
      // Preserve the existing admission/status path; retain_source's own
      // diagnostic explains why the subsequent missing-source record fails.
      attempt.ticket = depth_capture::nominate(command, input);
      retaining = true;
    }
#endif
    // A failed pixel copy can still yield a valid metadata-only nomination.
    // Preserve that admission failure even though its SDK/submission ticket
    // must continue through after_evaluate; ticket presence is not pixel proof.
    if (diagnostic.result != sunshine_streamline::depth_capture::status::recorded)
      remember_rejection(value, diagnostic, retaining, attempt.ticket);
    // Keep completing metadata-only nominations, but only a recorded copy or
    // authenticated shared preservation may suppress a nested usable input.
    if (attempt.ticket) {
      ++nominations;
      if (attempt.capture_authority) ++copy_recorded;
      else ++metadata_only;
    }
    return attempt;
  }
  void after_evaluate(const evaluation &value, bool successful) {
    preserve_error error;
    // An original vendor call can return after detach began. Its old ticket
    // must not reenter the capture owner's already destructing containers.
    if (sunshine_addon_lifetime::stopping()) return;
    if (value.diagnostic_observation.session)
      sunshine_game3d::diagnostic::finish_ngx(value.diagnostic_observation, value.source_id, successful,
        value.capture_suppressed_by_depth_owner);
    if (!value.observed) return;
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
    unsigned count = 0, capture_eligible = 0;
    std::array<calibration_observation, max_features> probes;
    unsigned probe_count = 0;
    AcquireSRWLockShared(&feature_lock);
    const bool read_probes = calibration_probe_requested && TryAcquireSRWLockExclusive(&calibration_lock);
    for (unsigned i = 0; i < features.size(); ++i) {
      const auto &value = features[i];
      count += value.handle != nullptr;
      capture_eligible += value.handle && value.parameters.valid;
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
    char message[512]{};
    std::snprintf(message, sizeof(message),
      "Sunshine NGX depth: confirmed_features=%u capture_eligible=%u recovered_features=%llu evaluations=%llu nominations=%llu copy_recorded=%llu metadata_only=%llu unknown_feature=%llu missing_parameters=%llu failed=%llu feature_capacity_loss=%llu; D3D12 NGX, named parameter exports, shared copy owner; recorded work is not completed pixels",
      count, capture_eligible, static_cast<unsigned long long>(recovered.load()),
      static_cast<unsigned long long>(evaluations.load()), static_cast<unsigned long long>(nominations.load()),
      static_cast<unsigned long long>(copy_recorded.load()), static_cast<unsigned long long>(metadata_only.load()),
      static_cast<unsigned long long>(unknown.load()), static_cast<unsigned long long>(missing_parameters.load()),
      static_cast<unsigned long long>(failed.load()), static_cast<unsigned long long>(feature_overflow.load()));
    sunshine_log::message(reshade::log::level::info, message);
#ifndef SUNSHINE_UPSCALER_TRACE_TEST
    const auto rejected_count = rejected_captures.load(std::memory_order_relaxed);
    if (rejected_count != reported_rejections && TryAcquireSRWLockShared(&rejection_lock)) {
      const auto snapshot = last_rejection;
      ReleaseSRWLockShared(&rejection_lock);
      if (snapshot.sequence) {
        reported_rejections = rejected_count;
        const auto &d = snapshot.diagnostic;
        const auto observer = sunshine_streamline::native_observer::counts();
        char rejected[1900]{};
        std::snprintf(rejected, sizeof(rejected),
          "Sunshine NGX capture rejection: count=%llu operation=%s result=%s stage=%s loss=%s source=%llu sequence=%llu nomination_ticket=%llu command=0x%llx cookie=%llu type=%u resource=0x%llx dimensions=%ux%u format=%u flags=0x%x dimension=%u mips=%u layers=%u samples=%u area=%u,%u,%u,%u state=0x%x observed_state=0x%x observed=%u blocked=%u closed=%u invalid=%u render_pass=%u generation=%llu/%llu device=%llu/%llu; recording barriers=%llu transitions=%llu tracked_sources=%llu last_barrier_command=0x%llx source_cookie=%llu; observer calls=%llu observed=%llu unreadable=%llu dropped=%llu installed=%llu/%llu rejected=%llu barrier_overflow=%llu submission_overflow=%llu discovery_contention=%llu",
          static_cast<unsigned long long>(rejected_count), snapshot.retaining ? "retain_source" : "nominate",
          sunshine_streamline::depth_capture::name(d.result), sunshine_streamline::depth_capture::name(d.stage),
          sunshine_streamline::depth_capture::name(d.loss), static_cast<unsigned long long>(snapshot.source_id),
          static_cast<unsigned long long>(snapshot.sequence), static_cast<unsigned long long>(snapshot.nomination_ticket),
          static_cast<unsigned long long>(d.command),
          static_cast<unsigned long long>(d.recording_cookie), d.command_type, static_cast<unsigned long long>(d.resource),
          d.width, d.height, d.format, d.flags, d.dimension, d.mip_levels, d.array_size, d.samples,
          snapshot.area.left, snapshot.area.top, snapshot.area.width, snapshot.area.height,
          d.native_state, d.observed_state, d.observed ? 1 : 0, d.blocked ? 1 : 0, d.recording_closed ? 1 : 0,
          d.recording_invalid ? 1 : 0, d.render_pass ? 1 : 0,
          static_cast<unsigned long long>(d.expected_generation), static_cast<unsigned long long>(d.current_generation),
          static_cast<unsigned long long>(d.expected_device_identity), static_cast<unsigned long long>(d.device_identity),
          static_cast<unsigned long long>(d.native_barrier_calls), static_cast<unsigned long long>(d.native_transition_count),
          static_cast<unsigned long long>(d.tracked_source_count), static_cast<unsigned long long>(d.last_barrier_command),
          static_cast<unsigned long long>(d.source_cookie),
          static_cast<unsigned long long>(observer.calls), static_cast<unsigned long long>(observer.observed),
          static_cast<unsigned long long>(observer.unreadable), static_cast<unsigned long long>(observer.dropped),
          static_cast<unsigned long long>(observer.installed), static_cast<unsigned long long>(observer.targets),
          static_cast<unsigned long long>(observer.rejected), static_cast<unsigned long long>(observer.barrier_overflow),
          static_cast<unsigned long long>(observer.submission_overflow), static_cast<unsigned long long>(observer.discovery_contention));
        sunshine_log::message(reshade::log::level::info, rejected);
      }
    }
#endif
  }
#ifdef SUNSHINE_UPSCALER_TRACE_TEST
  namespace testing {
    void set_callbacks(callbacks value) { backend = value; }
    bool last_capture_rejection(sunshine_streamline::depth_capture::record_diagnostic &out,
        std::uint64_t &nomination_ticket) {
      AcquireSRWLockShared(&rejection_lock);
      out = last_rejection.diagnostic;
      nomination_ticket = last_rejection.nomination_ticket;
      const bool available = last_rejection.sequence != 0;
      ReleaseSRWLockShared(&rejection_lock);
      return available;
    }
  }
#endif
}
