// SPDX-License-Identifier: GPL-3.0-only
#include "upscaler_call_trace.h"
#include "addon_lifetime.h"
#include "game3d_pool_submit.h"
#include "hook_host.h"
#include "ngx_depth_source.h"

#include <MinHook.h>
#include <TlHelp32.h>
#include <reshade.hpp>
#include "async_log.h"
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <cwctype>
#include <string>
#include <utility>
#include "game3d_clock.h"
#if defined(_MSC_VER)
#include <intrin.h>
#define SUNSHINE_TRACE_CALLER reinterpret_cast<std::uintptr_t>(_ReturnAddress())
#define SUNSHINE_TRACE_NOINLINE __declspec(noinline)
#else
#define SUNSHINE_TRACE_CALLER reinterpret_cast<std::uintptr_t>(__builtin_return_address(0))
#define SUNSHINE_TRACE_NOINLINE __attribute__((noinline))
#endif

namespace sunshine_upscaler_trace {
  namespace {
    // ABI authority: NVIDIA/DLSS include/nvsdk_ngx.h (CreateFeature,
    // EvaluateFeature, ReleaseFeature) and nvsdk_ngx_defs.h, inspected 2026-09-16:
    // https://github.com/NVIDIA/DLSS/blob/main/include/nvsdk_ngx.h
    // https://github.com/NVIDIA/DLSS/blob/main/include/nvsdk_ngx_defs.h
    // Win64 C ABI; enum/result are 32 bits, all other arguments are opaque
    // pointers. The SDK/core and snippet variants differ only in constness for
    // these three functions. We never inspect the parameter object's C++ ABI.
    using result = std::uint32_t;
    using create_function = result (__cdecl *)(void *, std::uint32_t, void *, void **);
    using progress_function = void (__cdecl *)(float, bool &);
    using evaluate_function = result (__cdecl *)(void *, const void *, const void *, progress_function);
    using release_function = result (__cdecl *)(void *);
    enum class operation : unsigned { create, evaluate, release, streamline };
    constexpr unsigned target_limit = 64, event_limit = 64, handle_limit = 64, stack_limit = 8;
    constexpr std::uint32_t unknown_feature = 0xffffffffu;
    constexpr unsigned no_target = target_limit;
    struct target {
      void *entry{};
      HMODULE owner{};
      operation op{};
      bool d3d11{};
      sunshine_ngx::parameter_api parameters;
      std::atomic<void *> original{};
      std::atomic<unsigned> state{}; // 1 created, 2 enabled, 3 rejected; never removed.
      std::atomic<std::uint64_t> sampled_epoch{};
    };
    struct event {
      std::uintptr_t caller{};
      std::uint32_t feature{};
      unsigned target_index{}, stack_count{};
      operation op{};
      bool inside_sl{}, used{};
      std::uint64_t calls{}, succeeded{}, failed{};
      // The newest NGX failure code (NVSDK_NGX_Result); zero before any NGX failure.
      std::uint32_t last_failure{};
      std::array<void *, stack_limit> stack{};
    };
    struct handle_record {
      HMODULE owner{};
      const void *handle{};
      std::uint32_t feature{};
      bool d3d11{};
    };
    struct invocation {
      std::uint64_t epoch{};
      std::uintptr_t caller{};
      bool inside_sl{};
      unsigned stack_count{};
      std::array<void *, stack_limit> stack{};
    };
    struct last_error_guard {
      DWORD value{GetLastError()};
      ~last_error_guard() { SetLastError(value); }
    };
    // No destructible locks/containers or workers: detours stay valid after the
    // add-on's normal shutdown and continue calling their immutable originals.
    SRWLOCK records_lock = SRWLOCK_INIT, poll_lock = SRWLOCK_INIT;
    std::array<target, target_limit> targets;
    std::array<event, event_limit> events;
    std::array<event, event_limit> reported_events; // Poll-owned; never read by hooks.
    std::array<unsigned, target_limit> reported_coverage{};
    std::array<handle_record, handle_limit> handles;
    std::atomic<std::uint64_t> active_epoch{}, dropped{}, stale{};
    std::atomic<bool> streamline_covered{};
    std::atomic<bool> handle_observation_gap{};
    std::atomic<HMODULE> addon_module{};
    std::uint64_t next_epoch{}, next_poll{}, next_report{}, began{}, reports{};
    unsigned target_count{}, rejected{}, found_modules{}, found_exports{}, skipped_frame_generation{};
    bool minhook_ready{};
    // A full export scan of ~200 game modules costs a few ms, so rescan only
    // after the loader reports a new module. The notification only counts
    // loads; poll schedules discovery, which runs outside the loader lock.
    // Without notifications, keep the periodic rescan.
    using loader_notification = void (NTAPI *)(ULONG reason, const void *data, void *context);
    using loader_register = LONG (NTAPI *)(ULONG flags, loader_notification callback, void *context, void **cookie);
    using loader_unregister = LONG (NTAPI *)(void *cookie);
    constexpr ULONG module_loaded_reason = 1; // LDR_DLL_NOTIFICATION_REASON_LOADED
    std::atomic<std::uint64_t> module_loads{1};
    std::uint64_t scanned_loads{}, scans{}; // poll-owned
    // One scan examines at most this many modules: a bound against a
    // pathological enumeration, not a module count any game reaches. A scan
    // that hits it reports truncated=1 and the next poll rescans from the
    // first module it did not examine. Written by discovery under poll_lock.
    constexpr unsigned module_scan_limit = 65536;
    std::atomic<bool> scan_truncated{};
    unsigned scan_resume{}, scan_truncations{};
#ifdef SUNSHINE_UPSCALER_TRACE_TEST
    unsigned synthetic_modules{}; // Enumeration entries ahead of the real modules (no module).
#endif
    std::atomic<void *> loader_cookie{}; // Written by initialize/shutdown; poll reads it.
    void NTAPI module_loaded(ULONG reason, const void *, void *);
    void watch_modules();
    void unwatch_modules();
    thread_local std::uint64_t tls_epoch{};
    thread_local unsigned tls_depth{};
    thread_local unsigned tls_capture_depth{};
    thread_local unsigned ngx_capture_depth{};
    thread_local unsigned ngx_diagnostic_depth{};

    class capture_scope; // Definition is in ngx_hooks.inl.

    bool succeeded(result value) { return (value & 0xfff00000u) != 0xbad00000u; }
    const char *op_name(operation op) {
      switch (op) {
      case operation::create: return "create";
      case operation::evaluate: return "evaluate";
      case operation::release: return "release";
      default: return "SL-evaluate";
      }
    }
    void message(const char *text) { sunshine_log::message(reshade::log::level::info, text); }
    invocation begin(unsigned index, std::uintptr_t caller) {
      invocation value;
      if (sunshine_addon_lifetime::stopping()) return value;
      value.epoch = active_epoch.load(std::memory_order_acquire);
      value.caller = caller;
      value.inside_sl = value.epoch && tls_epoch == value.epoch && tls_depth != 0;
      if (value.epoch && targets[index].sampled_epoch.exchange(value.epoch, std::memory_order_relaxed) != value.epoch) {
        // Exactly one bounded sample per export and observation epoch. No
        // module lookup, symbol engine, allocation, logging or GPU work here.
        value.stack_count = CaptureStackBackTrace(1, stack_limit, value.stack.data(), nullptr);
      }
      return value;
    }
    event *find_event(unsigned index, operation op, std::uintptr_t caller, std::uint32_t feature, bool inside) {
      event *empty = nullptr;
      for (auto &value : events) {
        if (!value.used) { if (!empty) empty = &value; continue; }
        if (value.target_index == index && value.op == op && value.caller == caller &&
            value.feature == feature && value.inside_sl == inside) return &value;
      }
      if (empty) {
        empty->used = true;
        empty->target_index = index;
        empty->op = op;
        empty->caller = caller;
        empty->feature = feature;
        empty->inside_sl = inside;
      }
      return empty;
    }
    // Caller holds the nonblocking records lock. Handle namespaces belong to
    // the export's owning module and API; a core and snippet may reuse pointers.
    handle_record *find_handle(const target &hook, const void *handle, bool create) {
      handle_record *empty = nullptr;
      for (auto &value : handles) {
        if (!value.handle) { if (!empty) empty = &value; continue; }
        if (value.owner == hook.owner && value.d3d11 == hook.d3d11 && value.handle == handle) return &value;
      }
      return create ? empty : nullptr;
    }
    void record(unsigned index, operation op, const invocation &call, std::uint32_t feature,
        const void *handle, bool success, std::uint32_t failure = 0) {
      if (sunshine_addon_lifetime::stopping() || !call.epoch) return;
      if (call.epoch != active_epoch.load(std::memory_order_acquire)) { ++stale; return; }
      if (!TryAcquireSRWLockExclusive(&records_lock)) {
        handle_observation_gap.store(true, std::memory_order_release);
        ++dropped;
        return;
      }
      if (call.epoch != active_epoch.load(std::memory_order_acquire)) {
        ++stale;
        ReleaseSRWLockExclusive(&records_lock);
        return;
      }
      if (index < target_limit) {
        const auto &hook = targets[index];
        // A missed create/release can reuse an address. Never retain a feature
        // identity across missing evidence; subsequent evaluations are unknown
        // until a new successful create is observed.
        if (handle_observation_gap.exchange(false, std::memory_order_acq_rel)) handles = {};
        if (op == operation::create && success && !handle) handles = {};
        if (op == operation::create && success && handle) {
          if (auto *value = find_handle(hook, handle, true))
            *value = {hook.owner, handle, feature, hook.d3d11};
          else ++dropped;
        } else if (op != operation::create) {
          if (auto *value = find_handle(hook, handle, false)) {
            feature = value->feature;
            if (op == operation::release && success) *value = {};
          }
        }
      }
      if (auto *value = find_event(index, op, call.caller, feature, call.inside_sl)) {
        ++value->calls;
        if (success) ++value->succeeded; else { ++value->failed; if (failure) value->last_failure = failure; }
        if (call.stack_count) { value->stack = call.stack; value->stack_count = call.stack_count; }
      } else ++dropped;
      ReleaseSRWLockExclusive(&records_lock);
    }
    template<unsigned Index> SUNSHINE_TRACE_NOINLINE result __cdecl create_hook(void *commands,
        std::uint32_t feature, void *parameters, void **output);
    template<unsigned Index> SUNSHINE_TRACE_NOINLINE result __cdecl evaluate_hook(void *commands,
        const void *handle, const void *parameters, progress_function callback);
    template<unsigned Index> SUNSHINE_TRACE_NOINLINE result __cdecl release_hook(void *handle);
    template<std::size_t... Indices> auto create_entries(std::index_sequence<Indices...>) {
      return std::array<void *, sizeof...(Indices)>{reinterpret_cast<void *>(&create_hook<Indices>)...};
    }
    template<std::size_t... Indices> auto evaluate_entries(std::index_sequence<Indices...>) {
      return std::array<void *, sizeof...(Indices)>{reinterpret_cast<void *>(&evaluate_hook<Indices>)...};
    }
    template<std::size_t... Indices> auto release_entries(std::index_sequence<Indices...>) {
      return std::array<void *, sizeof...(Indices)>{reinterpret_cast<void *>(&release_hook<Indices>)...};
    }
    const auto create_detours = create_entries(std::make_index_sequence<target_limit>{});
    const auto evaluate_detours = evaluate_entries(std::make_index_sequence<target_limit>{});
    const auto release_detours = release_entries(std::make_index_sequence<target_limit>{});

    bool queue(void *entry, operation op, bool d3d11);
    void apply_queued(unsigned first);
    bool frame_generation_module(const wchar_t *path);
    void discover();
    // Discovery runs on a pool thread, so the present thread neither scans
    // modules nor waits for MinHook's system-wide thread snapshot; it is
    // suspended only while new hooks are written. The pool thread holds
    // poll_lock, and a present-path poll meanwhile skips as on contention.
    // The add-on is pinned, so a pending callback never outlives its code.
    std::atomic<bool> discovery_pending{};
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    // TEST add-on only: the add-on lifecycle fixture holds the next pool
    // discovery, before it scans or hooks anything, across ReShade's unload
    // (SunshineSbsTestHoldDiscovery).
    std::atomic<HANDLE> discovery_gate_entered {}, discovery_gate_release {};
#endif
    void CALLBACK discover_on_pool(PTP_CALLBACK_INSTANCE, void *);
    void location(std::uintptr_t address, char *text, std::size_t size) {
      HMODULE module{};
      if (!address || !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
          reinterpret_cast<LPCWSTR>(address), &module)) {
        std::snprintf(text, size, "unresolved@0x%llx", static_cast<unsigned long long>(address));
        return;
      }
      char path[MAX_PATH]{};
      const auto length = GetModuleFileNameA(module, path, MAX_PATH);
      const char *name = length && length < MAX_PATH ? std::strrchr(path, '\\') : nullptr;
      if (name) ++name; else name = length && length < MAX_PATH ? path : "unknown-module";
      std::snprintf(text, size, "%s+0x%llx", name,
        static_cast<unsigned long long>(address - reinterpret_cast<std::uintptr_t>(module)));
      FreeLibrary(module);
    }
    void report() {
      std::array<event, event_limit> snapshot;
      AcquireSRWLockShared(&records_lock);
      snapshot = events;
      ReleaseSRWLockShared(&records_lock);
      std::array<std::uint64_t, 4> counts{};
      std::uint64_t inside{}, outside{}, unknown{};
      for (const auto &value : snapshot) if (value.used) {
        counts[static_cast<unsigned>(value.op)] += value.calls;
        if (value.op != operation::streamline) {
          (value.inside_sl ? inside : outside) += value.calls;
          if (value.feature == unknown_feature) unknown += value.calls;
        }
      }
      unsigned installed = 0;
      for (unsigned i = 0; i != target_count; ++i) installed += targets[i].state.load() == 2;
      char text[1200]{};
      std::snprintf(text, sizeof(text),
        "Sunshine upscaler trace: epoch=%llu elapsed_ms=%llu SL_hook=%u modules=%u truncated=%u truncations=%u skipped_frame_generation=%u exports=%u NGX_hooks=%u/%u rejected=%u SL_evaluate=%llu NGX_create=%llu NGX_evaluate=%llu NGX_release=%llu inside_observed_SL=%llu outside_observed_SL=%llu unknown_feature=%llu dropped=%llu stale=%llu; zero means no observed calls in this window, not API absence; outside is not proof of a direct path",
        static_cast<unsigned long long>(active_epoch.load()), static_cast<unsigned long long>(sunshine_game3d::clock::game_now() - began),
        streamline_covered.load() ? 1 : 0, found_modules, scan_truncated.load(std::memory_order_acquire) ? 1u : 0u,
        scan_truncations, skipped_frame_generation, found_exports, installed, target_count, rejected,
        static_cast<unsigned long long>(counts[3]), static_cast<unsigned long long>(counts[0]),
        static_cast<unsigned long long>(counts[1]), static_cast<unsigned long long>(counts[2]),
        static_cast<unsigned long long>(inside), static_cast<unsigned long long>(outside),
        static_cast<unsigned long long>(unknown), static_cast<unsigned long long>(dropped.load()), static_cast<unsigned long long>(stale.load()));
      message(text);
      for (unsigned i = 0; i != target_count; ++i) {
        const unsigned state = targets[i].state.load();
        if (reported_coverage[i] == state) continue;
        reported_coverage[i] = state;
        char owner[320]{};
        location(reinterpret_cast<std::uintptr_t>(targets[i].entry), owner, sizeof(owner));
        std::snprintf(text, sizeof(text), "Sunshine upscaler trace coverage: target=%u D3D%u %s %s state=%s",
          i, targets[i].d3d11 ? 11 : 12, op_name(targets[i].op), owner,
          targets[i].state.load() == 2 ? "hooked" : "unavailable");
        message(text);
      }
      for (unsigned event_index = 0; event_index != event_limit; ++event_index) {
        const auto &value = snapshot[event_index];
        auto &previous = reported_events[event_index];
        if (!value.used || value.calls == previous.calls) continue;
        char caller[320]{}, feature[40]{};
        location(value.caller, caller, sizeof(caller));
        if (value.feature == unknown_feature) std::snprintf(feature, sizeof(feature), "unknown(create not observed)");
        else std::snprintf(feature, sizeof(feature), "%u", value.feature);
        std::snprintf(text, sizeof(text),
          "Sunshine upscaler trace call: target=%u api=%s feature=%s caller=%s context=%s calls=%llu succeeded=%llu failed=%llu last_failure=0x%08x",
          value.target_index, op_name(value.op), feature, caller,
          value.op == operation::streamline ? "SL" : value.inside_sl ? "inside-observed-SL" : "outside-observed-SL",
          static_cast<unsigned long long>(value.calls), static_cast<unsigned long long>(value.succeeded), static_cast<unsigned long long>(value.failed),
          value.last_failure);
        message(text);
        for (unsigned frame = previous.stack_count; frame < value.stack_count; ++frame) {
          char entry[320]{};
          location(reinterpret_cast<std::uintptr_t>(value.stack[frame]), entry, sizeof(entry));
          std::snprintf(text, sizeof(text), "Sunshine upscaler trace first-call stack: target=%u frame=%u %s", value.target_index, frame, entry);
          message(text);
        }
        previous = value;
      }
      ++reports;
    }
  }


#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  // TEST add-on only: the next pool discovery signals entered, then waits for
  // release (at most 60 s) before it scans modules or installs a hook, which
  // pins the add-on as a hook's owner. The add-on lifecycle fixture arms it
  // as ReShade initializes a startup device, so that device's unload comes
  // while its discovery is certainly in flight and unpinned by any hook
  // (Assassin's Creed Shadows' crash in discover(), fixed by 5148a6da).
  extern "C" __declspec(dllexport) BOOL SunshineSbsTestHoldDiscovery(HANDLE entered, HANDLE release) {
    if (!entered || !release) {
      return FALSE;
    }
    discovery_gate_release.store(release, std::memory_order_release);
    discovery_gate_entered.store(entered, std::memory_order_release);
    return TRUE;
  }
#endif

#ifdef SUNSHINE_UPSCALER_TRACE_TEST
  namespace testing {
    bool frame_generation_module(const wchar_t *path) { return sunshine_upscaler_trace::frame_generation_module(path); }
    summary counts() {
      summary out;
      out.epoch = active_epoch.load();
      out.enabled = out.epoch != 0;
      out.streamline_covered = streamline_covered.load();
      out.dropped = dropped.load();
      out.stale = stale.load();
      out.reports = reports;
      out.discovered = target_count;
      out.rejected = rejected;
      out.scans = scans;
      out.modules = found_modules;
      out.truncated = scan_truncated.load();
      out.truncations = scan_truncations;
      out.load_notifications = loader_cookie.load() != nullptr;
      for (unsigned i = 0; i != target_count; ++i) out.installed += targets[i].state.load() == 2;
      AcquireSRWLockShared(&records_lock);
      for (const auto &value : events) if (value.used) {
        switch (value.op) {
        case sunshine_upscaler_trace::operation::create: out.ngx_create += value.calls; break;
        case sunshine_upscaler_trace::operation::evaluate: out.ngx_evaluate += value.calls; break;
        case sunshine_upscaler_trace::operation::release: out.ngx_release += value.calls; break;
        default: out.streamline += value.calls; break;
        }
        out.succeeded += value.succeeded;
        out.failed += value.failed;
        if (value.op != sunshine_upscaler_trace::operation::streamline) {
          (value.inside_sl ? out.inside_streamline : out.outside_streamline) += value.calls;
          if (value.feature == unknown_feature) out.unknown_feature += value.calls;
        }
      }
      ReleaseSRWLockShared(&records_lock);
      return out;
    }
    bool install(void *target, operation op, bool d3d11) {
      last_error_guard error;
      const auto kind = static_cast<sunshine_upscaler_trace::operation>(op);
      const unsigned first = target_count;
      queue(target, kind, d3d11);
      apply_queued(first);
      for (unsigned i = 0; i != target_count; ++i)
        if (targets[i].entry == target) return targets[i].op == kind && targets[i].d3d11 == d3d11 && targets[i].state.load() == 2;
      return false;
    }
    void report_now() { last_error_guard error; report(); }
    void wait_discovery() {
      while (discovery_pending.load(std::memory_order_acquire)) Sleep(1);
      AcquireSRWLockExclusive(&poll_lock); // The pool thread releases it after clearing pending.
      ReleaseSRWLockExclusive(&poll_lock);
    }
    void set_synthetic_modules(unsigned count) {
      AcquireSRWLockExclusive(&poll_lock);
      synthetic_modules = count;
      ReleaseSRWLockExclusive(&poll_lock);
    }
  }
#endif
}

#include "ngx_hooks.inl"
