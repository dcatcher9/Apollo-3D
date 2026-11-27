// SPDX-License-Identifier: GPL-3.0-only
// The add-on's one NGX hook host (ngx_hooks.h): the NGX entry detours, which
// run production NGX depth capture (ngx_depth_source.h) and feed the call
// trace (upscaler_call_trace.h), their discovery and installation, loader
// notifications, and the Streamline scope both nest in.
#include "ngx_hooks.h"
#include "upscaler_call_trace.h"
#include "addon_lifetime.h"
#include "addon_win32.h"
#include "game3d_pool_submit.h"
#include "hook_host.h"
#include "ngx_depth_source.h"

#include <MinHook.h>
#include <TlHelp32.h>
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

namespace sunshine_ngx_hooks {
  namespace {
    using namespace sunshine_upscaler_trace::feed;
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
    struct target {
      void *entry{};
      HMODULE owner{};
      operation op{};
      bool d3d11{};
      sunshine_ngx::parameter_api parameters;
      std::atomic<void *> original{};
      std::atomic<unsigned> state{}; // 1 created, 2 enabled, 3 rejected; never removed.
    };
    using sunshine_addon_win32::last_error_guard;
    // No destructible locks/containers or workers: detours stay valid after the
    // add-on's normal shutdown and continue calling their immutable originals.
    SRWLOCK poll_lock = SRWLOCK_INIT;
    std::array<target, target_limit> targets;
    std::atomic<HMODULE> addon_module{};
    std::uint64_t next_poll{}, next_report{};
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
    // The calling thread's Streamline evaluation (streamline_scope): its trace
    // epoch and nesting, and whether it claimed depth capture.
    thread_local std::uint64_t tls_epoch{};
    thread_local unsigned tls_depth{};
    thread_local unsigned tls_capture_depth{};
    thread_local unsigned ngx_capture_depth{};
    thread_local unsigned ngx_diagnostic_depth{};

    bool succeeded(result value) { return (value & 0xfff00000u) != 0xbad00000u; }
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
    // The targets and discovery as the trace's report reads them.
    coverage hook_coverage() {
      coverage out;
      out.count = target_count;
      out.rejected = rejected;
      out.modules = found_modules;
      out.exports = found_exports;
      out.skipped_frame_generation = skipped_frame_generation;
      out.truncated = scan_truncated.load(std::memory_order_acquire);
      out.truncations = scan_truncations;
      for (unsigned i = 0; i != target_count; ++i)
        out.targets[i] = {targets[i].entry, targets[i].op, targets[i].d3d11, targets[i].state.load()};
      return out;
    }
    class capture_scope {
    public:
      sunshine_ngx::evaluation value;
      bool finished{};
      bool diagnostic_owner{};
      capture_scope(const target &hook, std::uint64_t command, const void *handle, const void *parameters) {
        if (hook.d3d11 || ngx_capture_depth || !sunshine_ngx::enabled()) return;
        // SDK/core wrappers can evaluate recursively even after an outer depth
        // nomination was rejected. Query/copy the dump resources once while
        // still allowing the inner wrapper to supply a usable depth input.
        diagnostic_owner = !ngx_diagnostic_depth && hook.parameters && command && handle && parameters &&
          sunshine_game3d::diagnostic_metadata_generation();
        if (diagnostic_owner) ++ngx_diagnostic_depth;
        value = sunshine_ngx::before_evaluate(hook.owner, hook.parameters, command, handle, parameters,
          tls_capture_depth == 0, diagnostic_owner);
        if (value.capture_authority) ++ngx_capture_depth;
      }
      void finish(bool success) {
        if (finished) return;
        finished = true;
        sunshine_ngx::after_evaluate(value, success);
      }
      ~capture_scope() {
        if (!finished) finish(false);
        if (value.capture_authority) --ngx_capture_depth;
        if (diagnostic_owner) --ngx_diagnostic_depth;
      }
    };

    void NTAPI module_loaded(ULONG reason, const void *, void *) {
      if (reason == module_loaded_reason) module_loads.fetch_add(1, std::memory_order_release);
    }

    void watch_modules() {
      if (loader_cookie.load(std::memory_order_acquire)) return;
      const auto ntdll = GetModuleHandleW(L"ntdll.dll");
      const auto watch = ntdll ? reinterpret_cast<loader_register>(reinterpret_cast<void *>(GetProcAddress(ntdll, "LdrRegisterDllNotification"))) : nullptr;
      void *cookie{};
      if (watch && watch(0, &module_loaded, nullptr, &cookie) == 0) loader_cookie.store(cookie, std::memory_order_release);
    }

    void unwatch_modules() {
      const auto cookie = loader_cookie.exchange(nullptr, std::memory_order_acq_rel);
      if (!cookie) return;
      const auto ntdll = GetModuleHandleW(L"ntdll.dll");
      const auto unwatch = ntdll ? reinterpret_cast<loader_unregister>(reinterpret_cast<void *>(GetProcAddress(ntdll, "LdrUnregisterDllNotification"))) : nullptr;
      if (unwatch) unwatch(cookie);
    }

    template<unsigned Index> SUNSHINE_TRACE_NOINLINE result __cdecl create_hook(void *commands,
        std::uint32_t feature, void *parameters, void **output) {
      const DWORD incoming = GetLastError();
      const auto call = begin(Index, SUNSHINE_TRACE_CALLER, tls_epoch, tls_depth);
      const auto creation = targets[Index].d3d11 ? sunshine_ngx::creation{} :
        sunshine_ngx::before_create(targets[Index].parameters, feature, parameters);
      const auto original = reinterpret_cast<create_function>(targets[Index].original.load(std::memory_order_acquire));
      SetLastError(incoming);
      const result value = original(commands, feature, parameters, output);
      last_error_guard outgoing;
      void *handle = nullptr;
      SIZE_T copied = 0;
      if (!sunshine_addon_lifetime::stopping() && (call.epoch || creation.epoch) && succeeded(value) && output &&
          (!ReadProcessMemory(GetCurrentProcess(), output, &handle, sizeof(handle), &copied) || copied != sizeof(handle)))
        handle = nullptr;
      sunshine_ngx::after_create(targets[Index].owner, handle, creation, succeeded(value));
      record(Index, targets[Index].owner, targets[Index].d3d11, operation::create, call, feature, handle, succeeded(value), value);
      return value;
    }

    template<unsigned Index> SUNSHINE_TRACE_NOINLINE result __cdecl evaluate_hook(void *commands,
        const void *handle, const void *parameters, progress_function callback) {
      const DWORD incoming = GetLastError();
      const auto call = begin(Index, SUNSHINE_TRACE_CALLER, tls_epoch, tls_depth);
      capture_scope capture(targets[Index], reinterpret_cast<std::uint64_t>(commands), handle, parameters);
      const auto original = reinterpret_cast<evaluate_function>(targets[Index].original.load(std::memory_order_acquire));
      SetLastError(incoming);
      const result value = original(commands, handle, parameters, callback);
      last_error_guard outgoing;
      capture.finish(succeeded(value));
      record(Index, targets[Index].owner, targets[Index].d3d11, operation::evaluate, call, unknown_feature, handle, succeeded(value), value);
      return value;
    }

    template<unsigned Index> SUNSHINE_TRACE_NOINLINE result __cdecl release_hook(void *handle) {
      const DWORD incoming = GetLastError();
      const auto call = begin(Index, SUNSHINE_TRACE_CALLER, tls_epoch, tls_depth);
      const auto capture_epoch = targets[Index].d3d11 ? 0 : sunshine_ngx::epoch();
      const auto original = reinterpret_cast<release_function>(targets[Index].original.load(std::memory_order_acquire));
      SetLastError(incoming);
      const result value = original(handle);
      last_error_guard outgoing;
      sunshine_ngx::after_release(targets[Index].owner, handle, capture_epoch, succeeded(value));
      record(Index, targets[Index].owner, targets[Index].d3d11, operation::release, call, unknown_feature, handle, succeeded(value), value);
      return value;
    }

    // Creates a new target's detour and queues its enable. Every MinHook
    // enable or apply first snapshots all threads on the system (~50 ms with
    // ~10k threads) and then suspends this process's, so a scan commits all
    // of its new targets in one apply.
    bool queue(void *entry, operation op, bool d3d11) {
      for (unsigned i = 0; i != target_count; ++i)
        if (targets[i].entry == entry) return false;
      if (target_count == target_limit) { ++rejected; return false; }
      MEMORY_BASIC_INFORMATION memory{};
      HMODULE owner{}, addon{};
      const auto executable = VirtualQuery(entry, &memory, sizeof(memory)) &&
        sunshine_game3d::hook_host::executable_image(memory);
      if (!executable || !sunshine_addon_win32::pin_module_of(entry, &owner) ||
          !sunshine_addon_win32::pin_module_of(&initialize, &addon) || addon != addon_module.load()) {
        ++rejected;
        return false;
      }
      if (!sunshine_game3d::hook_host::minhook_ready(minhook_ready)) { ++rejected; return false; }
      const unsigned index = target_count++;
      auto &hook = targets[index];
      hook.entry = entry;
      hook.owner = owner;
      hook.op = op;
      hook.d3d11 = d3d11;
      hook.parameters = sunshine_ngx::resolve_parameter_api(owner);
      void *original{};
      void *detour = op == operation::create ? create_detours[index] :
        op == operation::evaluate ? evaluate_detours[index] : release_detours[index];
      if (MH_CreateHook(entry, detour, &original) != MH_OK || !original) {
        hook.state.store(3);
        ++rejected;
        return false;
      }
      hook.original.store(original, std::memory_order_release);
      if (MH_QueueEnableHook(entry) != MH_OK) {
        hook.state.store(3);
        ++rejected;
        return false;
      }
      hook.state.store(1, std::memory_order_release);
      return true;
    }

    // A failed apply leaves the queued targets reported unavailable; any that
    // were enabled stay pass-through through their retained trampolines.
    void apply_queued(unsigned first) {
      if (first == target_count) return;
      const bool applied = MH_ApplyQueued() == MH_OK;
      for (unsigned i = first; i != target_count; ++i) {
        if (targets[i].state.load() != 1) continue;
        if (applied) targets[i].state.store(2, std::memory_order_release);
        else ++rejected;
      }
    }

    // nvngx_dlssg*.dll, or a driver model under an NGX models\dlssg directory;
    // never a folder that merely contains "dlssg" in its name.
    bool frame_generation_module(const wchar_t *path) {
      std::wstring lower(path);
      for (auto &c : lower) c = static_cast<wchar_t>(towlower(c));
      const auto slash = lower.find_last_of(L"\\/");
      const auto name = slash == std::wstring::npos ? lower : lower.substr(slash + 1);
      return name.rfind(L"nvngx_dlssg", 0) == 0 || lower.find(L"\\ngx\\models\\dlssg\\") != std::wstring::npos;
    }

    void discover() {
      found_modules = found_exports = skipped_frame_generation = 0;
      ++scans;
      const unsigned first = target_count;
      struct commit {
        unsigned first;
        ~commit() { apply_queued(first); }
      } queued{first};
      HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
      if (snapshot == INVALID_HANDLE_VALUE) return;
      MODULEENTRY32W value{};
      value.dwSize = sizeof(value);
      // Entries before scan_resume were examined by the truncated scan before.
      const unsigned skip = std::exchange(scan_resume, 0u);
      unsigned position = 0, examined = 0;
      bool truncated = false;
      const auto admit = [&] {
        if (position++ < skip) return false;
        if (examined == module_scan_limit) { truncated = true; return false; }
        ++examined;
        return true;
      };
#ifdef SUNSHINE_UPSCALER_TRACE_TEST
      for (unsigned i = 0; i != synthetic_modules && !truncated; ++i) admit();
#endif
      if (!truncated && Module32FirstW(snapshot, &value)) do {
        if (!admit()) {
          if (truncated) break;
          continue;
        }
        // DLSS Frame Generation refuses to create its feature once its module's
        // NGX entry points are patched: The Witcher 3 got NVSDK_NGX_Result
        // FAIL_PlatformError (0xBAD00002) on every attempt while hooked, and
        // generated frames once the hooks were off. Frame generation supplies no
        // depth through NGX (Streamline tags carry it), so never patch its module,
        // shipped as nvngx_dlssg.dll or as a driver model under NGX\models\dlssg.
        if (frame_generation_module(value.szExePath)) { ++skipped_frame_generation; continue; }
        // Static SDKs can export NGX from any game/plugin module. Discover by
        // exact public exports, independently of DLL filename or SDK version.
        HMODULE retained{};
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
            reinterpret_cast<LPCWSTR>(value.hModule), &retained)) continue;
        ++found_modules;
        // D3D11 NGX supplies no depth capture; its calls only feed the trace.
        const unsigned apis = epoch() ? 2 : 1;
        for (unsigned api = 0; api != apis; ++api) {
          for (unsigned op = 0; op != 4; ++op) {
            char name[80]{};
            const char *suffix = op == 0 ? "CreateFeature" : op == 1 ? "EvaluateFeature" :
              op == 2 ? "ReleaseFeature" : "EvaluateFeature_C";
            std::snprintf(name, sizeof(name), "NVSDK_NGX_D3D%u_%s", api == 0 ? 12 : 11, suffix);
            if (auto entry = GetProcAddress(retained, name)) {
              ++found_exports;
              // The _C form changes only the progress callback's pointee
              // signature (bool* vs bool&); our pass-through never invokes it.
              queue(reinterpret_cast<void *>(entry), op == 3 ? operation::evaluate : static_cast<operation>(op), api == 1);
            }
          }
        }
        FreeLibrary(retained);
      } while (Module32NextW(snapshot, &value));
      CloseHandle(snapshot);
      if (truncated) { scan_resume = position - 1; ++scan_truncations; }
      scan_truncated.store(truncated, std::memory_order_release);
    }

    void CALLBACK discover_on_pool(PTP_CALLBACK_INSTANCE, void *) {
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      if (const auto entered = discovery_gate_entered.exchange(nullptr, std::memory_order_acq_rel)) {
        const auto release = discovery_gate_release.exchange(nullptr, std::memory_order_acq_rel);
        SetEvent(entered);
        if (release) {
          WaitForSingleObject(release, 60000);
        }
      }
#endif
      AcquireSRWLockExclusive(&poll_lock);
      if (!sunshine_addon_lifetime::stopping()) discover();
      discovery_pending.store(false, std::memory_order_release);
      ReleaseSRWLockExclusive(&poll_lock);
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
    bool frame_generation_module(const wchar_t *path) { return sunshine_ngx_hooks::frame_generation_module(path); }
    summary counts() {
      const auto trace = sunshine_upscaler_trace::feed::test_counts();
      summary out;
      out.epoch = trace.epoch;
      out.enabled = out.epoch != 0;
      out.streamline_covered = trace.streamline_covered;
      out.dropped = trace.dropped;
      out.stale = trace.stale;
      out.reports = trace.reports;
      out.discovered = target_count;
      out.rejected = rejected;
      out.scans = scans;
      out.modules = found_modules;
      out.truncated = scan_truncated.load();
      out.truncations = scan_truncations;
      out.load_notifications = loader_cookie.load() != nullptr;
      for (unsigned i = 0; i != target_count; ++i) out.installed += targets[i].state.load() == 2;
      out.ngx_create = trace.ngx_create;
      out.ngx_evaluate = trace.ngx_evaluate;
      out.ngx_release = trace.ngx_release;
      out.streamline = trace.streamline;
      out.succeeded = trace.succeeded;
      out.failed = trace.failed;
      out.inside_streamline = trace.inside_streamline;
      out.outside_streamline = trace.outside_streamline;
      out.unknown_feature = trace.unknown_feature;
      return out;
    }
    bool install(void *target, operation op, bool d3d11) {
      last_error_guard error;
      const auto kind = static_cast<sunshine_upscaler_trace::feed::operation>(op);
      const unsigned first = target_count;
      queue(target, kind, d3d11);
      apply_queued(first);
      for (unsigned i = 0; i != target_count; ++i)
        if (targets[i].entry == target) return targets[i].op == kind && targets[i].d3d11 == d3d11 && targets[i].state.load() == 2;
      return false;
    }
    void report_now() { last_error_guard error; sunshine_upscaler_trace::feed::report(hook_coverage()); }
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

  void initialize(HMODULE addon, bool requested, bool capture) {
    if (sunshine_addon_lifetime::stopping()) return;
    last_error_guard error;
    next_poll = next_report = 0;
    scanned_loads = 0;
    addon_module.store(addon);
    sunshine_upscaler_trace::feed::restart(requested);
    sunshine_ngx::initialize(capture);
    if (requested || capture) watch_modules();
  }

  void shutdown() {
    sunshine_upscaler_trace::feed::stop();
    sunshine_ngx::shutdown();
    unwatch_modules();
  }

  bool capture_enabled() { return sunshine_ngx::enabled(); }

  void poll() {
    last_error_guard error;
    if ((!sunshine_upscaler_trace::enabled() && !capture_enabled()) || !TryAcquireSRWLockExclusive(&poll_lock)) return;
    const auto now = sunshine_game3d::clock::game_now();
    const auto loads = module_loads.load(std::memory_order_acquire);
    const bool notified = loader_cookie.load(std::memory_order_acquire) != nullptr;
    const bool pending = discovery_pending.load(std::memory_order_acquire);
    if (!pending && now >= next_poll &&
        (!notified || loads != scanned_loads || scan_truncated.load(std::memory_order_acquire))) {
      // Loads observed during this scan leave the count changed for the next one.
      scanned_loads = loads;
      next_poll = now + (notified ? 250 : 1000);
      discovery_pending.store(true, std::memory_order_release);
      if (!sunshine_game3d::pool::submit(sunshine_game3d::pool::family::ngx_discover, discover_on_pool, nullptr)) {
        discover();
        discovery_pending.store(false, std::memory_order_release);
      }
    }
    // Coverage is reported after a scan completes, never from its starting state.
    if (sunshine_upscaler_trace::enabled() && now >= next_report && !discovery_pending.load(std::memory_order_acquire)) {
      next_report = now + 5000;
      sunshine_upscaler_trace::feed::report(hook_coverage());
    }
    sunshine_ngx::poll();
    ReleaseSRWLockExclusive(&poll_lock);
  }

  streamline_scope::streamline_scope(std::uint32_t feature, std::uintptr_t caller) :
      epoch_(sunshine_addon_lifetime::stopping() ? 0 : sunshine_upscaler_trace::feed::epoch()), previous_epoch_(tls_epoch),
      feature_(feature), previous_depth_(tls_depth), previous_capture_depth_(tls_capture_depth), caller_(caller) {
    tls_depth = tls_depth + 1;
    tls_epoch = epoch_;
  }

  void streamline_scope::claim_depth_capture(bool owns) {
    if (owns && !capture_claimed_) {
      ++tls_capture_depth;
      capture_claimed_ = true;
    }
  }

  void streamline_scope::finish(bool success) {
    last_error_guard error;
    if (finished_) return;
    finished_ = true;
    sunshine_upscaler_trace::feed::invocation call;
    call.epoch = epoch_;
    call.caller = caller_;
    sunshine_upscaler_trace::feed::record(sunshine_upscaler_trace::feed::no_target, nullptr, false,
      sunshine_upscaler_trace::feed::operation::streamline, call, feature_, nullptr, success);
  }

  streamline_scope::~streamline_scope() {
    tls_epoch = previous_epoch_;
    tls_depth = previous_depth_;
    tls_capture_depth = previous_capture_depth_;
  }
}
