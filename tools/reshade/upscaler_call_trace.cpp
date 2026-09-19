// SPDX-License-Identifier: GPL-3.0-only
#include "upscaler_call_trace.h"
#include "addon_lifetime.h"
#include "ngx_depth_source.h"

#include <MinHook.h>
#include <TlHelp32.h>
#include <reshade.hpp>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <cwctype>
#include <utility>
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
    unsigned target_count{}, rejected{}, found_modules{}, found_exports{};
    bool minhook_ready{};
    thread_local std::uint64_t tls_epoch{};
    thread_local unsigned tls_depth{};
    thread_local unsigned ngx_capture_depth{};

    class capture_scope {
    public:
      sunshine_ngx::evaluation value;
      bool finished{};
      capture_scope(const target &hook, std::uint64_t command, const void *handle, const void *parameters) {
        if (hook.d3d11 || tls_depth || ngx_capture_depth || !sunshine_ngx::enabled()) return;
        value = sunshine_ngx::before_evaluate(hook.owner, hook.parameters, command, handle, parameters);
        if (value.observed) ++ngx_capture_depth;
      }
      void finish(bool success) {
        if (finished) return;
        finished = true;
        sunshine_ngx::after_evaluate(value, success);
      }
      ~capture_scope() {
        if (!finished) finish(false);
        if (value.observed) --ngx_capture_depth;
      }
    };

    bool succeeded(result value) { return (value & 0xfff00000u) != 0xbad00000u; }
    const char *op_name(operation op) {
      switch (op) {
      case operation::create: return "create";
      case operation::evaluate: return "evaluate";
      case operation::release: return "release";
      default: return "SL-evaluate";
      }
    }
    void message(const char *text) { reshade::log::message(reshade::log::level::info, text); }
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
        const void *handle, bool success) {
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
        if (success) ++value->succeeded; else ++value->failed;
        if (call.stack_count) { value->stack = call.stack; value->stack_count = call.stack_count; }
      } else ++dropped;
      ReleaseSRWLockExclusive(&records_lock);
    }
    template<unsigned Index> SUNSHINE_TRACE_NOINLINE result __cdecl create_hook(void *commands,
        std::uint32_t feature, void *parameters, void **output) {
      const DWORD incoming = GetLastError();
      const auto call = begin(Index, SUNSHINE_TRACE_CALLER);
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
      record(Index, operation::create, call, feature, handle, succeeded(value));
      return value;
    }
    template<unsigned Index> SUNSHINE_TRACE_NOINLINE result __cdecl evaluate_hook(void *commands,
        const void *handle, const void *parameters, progress_function callback) {
      const DWORD incoming = GetLastError();
      const auto call = begin(Index, SUNSHINE_TRACE_CALLER);
      capture_scope capture(targets[Index], reinterpret_cast<std::uint64_t>(commands), handle, parameters);
      const auto original = reinterpret_cast<evaluate_function>(targets[Index].original.load(std::memory_order_acquire));
      SetLastError(incoming);
      const result value = original(commands, handle, parameters, callback);
      last_error_guard outgoing;
      capture.finish(succeeded(value));
      record(Index, operation::evaluate, call, unknown_feature, handle, succeeded(value));
      return value;
    }
    template<unsigned Index> SUNSHINE_TRACE_NOINLINE result __cdecl release_hook(void *handle) {
      const DWORD incoming = GetLastError();
      const auto call = begin(Index, SUNSHINE_TRACE_CALLER);
      const auto capture_epoch = targets[Index].d3d11 ? 0 : sunshine_ngx::epoch();
      const auto original = reinterpret_cast<release_function>(targets[Index].original.load(std::memory_order_acquire));
      SetLastError(incoming);
      const result value = original(handle);
      last_error_guard outgoing;
      sunshine_ngx::after_release(targets[Index].owner, handle, capture_epoch, succeeded(value));
      record(Index, operation::release, call, unknown_feature, handle, succeeded(value));
      return value;
    }
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

    bool install(void *entry, operation op, bool d3d11) {
      for (unsigned i = 0; i != target_count; ++i)
        if (targets[i].entry == entry) return targets[i].op == op && targets[i].d3d11 == d3d11 && targets[i].state.load() == 2;
      if (target_count == target_limit) { ++rejected; return false; }
      MEMORY_BASIC_INFORMATION memory{};
      HMODULE owner{}, addon{};
      const auto executable = VirtualQuery(entry, &memory, sizeof(memory)) && memory.Type == MEM_IMAGE &&
        memory.State == MEM_COMMIT && !(memory.Protect & (PAGE_GUARD | PAGE_NOACCESS)) &&
        (memory.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY));
      if (!executable || !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
            reinterpret_cast<LPCWSTR>(entry), &owner) ||
          !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
            reinterpret_cast<LPCWSTR>(&initialize), &addon) || addon != addon_module.load()) {
        ++rejected;
        return false;
      }
      if (!minhook_ready) {
        const auto status = MH_Initialize();
        minhook_ready = status == MH_OK || status == MH_ERROR_ALREADY_INITIALIZED;
      }
      if (!minhook_ready) { ++rejected; return false; }
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
      hook.state.store(1, std::memory_order_release);
      if (MH_EnableHook(entry) != MH_OK) { ++rejected; return false; }
      hook.state.store(2, std::memory_order_release);
      return true;
    }
    void discover() {
      found_modules = found_exports = 0;
      HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
      if (snapshot == INVALID_HANDLE_VALUE) return;
      MODULEENTRY32W value{};
      value.dwSize = sizeof(value);
      unsigned examined = 0;
      if (Module32FirstW(snapshot, &value)) do {
        if (++examined > 1024) break;
        wchar_t lower[MAX_PATH]{};
        for (unsigned i = 0; i + 1 < MAX_PATH && value.szModule[i]; ++i) lower[i] = std::towlower(value.szModule[i]);
        const bool ngx_module = std::wcscmp(lower, L"nvngx.dll") == 0 || std::wcscmp(lower, L"_nvngx.dll") == 0 ||
          std::wcscmp(lower, L"nvngx_dlss.dll") == 0 || std::wcscmp(lower, L"nvngx_dlssd.dll") == 0 ||
          std::wcscmp(lower, L"nvngx_dlssg.dll") == 0 || std::wcscmp(lower, L"nvngx_deepdvc.dll") == 0;
        if (value.hModule != GetModuleHandleW(nullptr) && !ngx_module) continue;
        HMODULE retained{};
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
            reinterpret_cast<LPCWSTR>(value.hModule), &retained)) continue;
        ++found_modules;
        for (unsigned api = 0; api != 2; ++api) {
          for (unsigned op = 0; op != 4; ++op) {
            char name[80]{};
            const char *suffix = op == 0 ? "CreateFeature" : op == 1 ? "EvaluateFeature" :
              op == 2 ? "ReleaseFeature" : "EvaluateFeature_C";
            std::snprintf(name, sizeof(name), "NVSDK_NGX_D3D%u_%s", api == 0 ? 12 : 11, suffix);
            if (auto entry = GetProcAddress(retained, name)) {
              ++found_exports;
              // The _C form changes only the progress callback's pointee
              // signature (bool* vs bool&); our pass-through never invokes it.
              install(reinterpret_cast<void *>(entry), op == 3 ? operation::evaluate : static_cast<operation>(op), api == 1);
            }
          }
        }
        FreeLibrary(retained);
      } while (Module32NextW(snapshot, &value));
      CloseHandle(snapshot);
    }
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
        "Sunshine upscaler trace: epoch=%llu elapsed_ms=%llu SL_hook=%u modules=%u exports=%u NGX_hooks=%u/%u rejected=%u SL_evaluate=%llu NGX_create=%llu NGX_evaluate=%llu NGX_release=%llu inside_observed_SL=%llu outside_observed_SL=%llu unknown_feature=%llu dropped=%llu stale=%llu; zero means no observed calls in this window, not API absence; outside is not proof of a direct path",
        static_cast<unsigned long long>(active_epoch.load()), static_cast<unsigned long long>(GetTickCount64() - began),
        streamline_covered.load() ? 1 : 0, found_modules, found_exports, installed, target_count, rejected,
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
          "Sunshine upscaler trace call: target=%u api=%s feature=%s caller=%s context=%s calls=%llu succeeded=%llu failed=%llu",
          value.target_index, op_name(value.op), feature, caller,
          value.op == operation::streamline ? "SL" : value.inside_sl ? "inside-observed-SL" : "outside-observed-SL",
          static_cast<unsigned long long>(value.calls), static_cast<unsigned long long>(value.succeeded), static_cast<unsigned long long>(value.failed));
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

  void initialize(HMODULE addon, bool requested, bool capture, bool calibration_probe) {
    if (sunshine_addon_lifetime::stopping()) return;
    last_error_guard error;
    active_epoch.store(0, std::memory_order_release);
    AcquireSRWLockExclusive(&records_lock);
    events = {};
    reported_events = {};
    reported_coverage = {};
    handles = {};
    handle_observation_gap.store(false);
    dropped = stale = 0;
    reports = 0;
    next_poll = next_report = 0;
    began = GetTickCount64();
    addon_module.store(addon);
    if (requested) active_epoch.store(++next_epoch, std::memory_order_release);
    ReleaseSRWLockExclusive(&records_lock);
    sunshine_ngx::initialize(capture, calibration_probe);
  }
  void shutdown() { active_epoch.store(0, std::memory_order_release); sunshine_ngx::shutdown(); }
  bool enabled() { return !sunshine_addon_lifetime::stopping() && active_epoch.load(std::memory_order_acquire) != 0; }
  bool capture_enabled() { return sunshine_ngx::enabled(); }
  void set_streamline_coverage(bool installed) { streamline_covered.store(installed, std::memory_order_release); }
  void poll() {
    last_error_guard error;
    if ((!enabled() && !capture_enabled()) || !TryAcquireSRWLockExclusive(&poll_lock)) return;
    const auto now = GetTickCount64();
    if (now >= next_poll) { next_poll = now + 1000; discover(); }
    if (enabled() && now >= next_report) { next_report = now + 5000; report(); }
    sunshine_ngx::poll();
    ReleaseSRWLockExclusive(&poll_lock);
  }
  streamline_scope::streamline_scope(std::uint32_t feature, std::uintptr_t caller) :
      epoch_(sunshine_addon_lifetime::stopping() ? 0 : active_epoch.load(std::memory_order_acquire)), previous_epoch_(tls_epoch),
      feature_(feature), previous_depth_(tls_depth), caller_(caller) {
    // Native NGX calls made underneath SL must not produce a second capture,
    // including when diagnostic tracing is disabled.
    tls_depth = tls_depth + 1;
    tls_epoch = epoch_;
  }
  void streamline_scope::finish(bool success) {
    last_error_guard error;
    if (finished_) return;
    finished_ = true;
    invocation call;
    call.epoch = epoch_;
    call.caller = caller_;
    record(no_target, operation::streamline, call, feature_, nullptr, success);
  }
  streamline_scope::~streamline_scope() {
    tls_epoch = previous_epoch_;
    tls_depth = previous_depth_;
  }

#ifdef SUNSHINE_UPSCALER_TRACE_TEST
  namespace testing {
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
      return sunshine_upscaler_trace::install(target, static_cast<sunshine_upscaler_trace::operation>(op), d3d11);
    }
    void report_now() { last_error_guard error; report(); }
  }
#endif
}
