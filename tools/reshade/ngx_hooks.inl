// SPDX-License-Identifier: GPL-3.0-only
// Live NGX hook host, included at the end of upscaler_call_trace.cpp.
// Targets, detour table initialization, locks and TLS retain their shell positions.

namespace sunshine_upscaler_trace {
  namespace {
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
      record(Index, operation::create, call, feature, handle, succeeded(value), value);
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
      record(Index, operation::evaluate, call, unknown_feature, handle, succeeded(value), value);
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
      record(Index, operation::release, call, unknown_feature, handle, succeeded(value), value);
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
        const unsigned apis = active_epoch.load(std::memory_order_acquire) ? 2 : 1;
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

  void initialize(HMODULE addon, bool requested, bool capture) {
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
    scanned_loads = 0;
    began = GetTickCount64();
    addon_module.store(addon);
    if (requested) active_epoch.store(++next_epoch, std::memory_order_release);
    ReleaseSRWLockExclusive(&records_lock);
    sunshine_ngx::initialize(capture);
    if (requested || capture) watch_modules();
  }

  void shutdown() {
    active_epoch.store(0, std::memory_order_release);
    sunshine_ngx::shutdown();
    unwatch_modules();
  }

  bool enabled() { return !sunshine_addon_lifetime::stopping() && active_epoch.load(std::memory_order_acquire) != 0; }

  bool capture_enabled() { return sunshine_ngx::enabled(); }

  void set_streamline_coverage(bool installed) { streamline_covered.store(installed, std::memory_order_release); }

  void poll() {
    last_error_guard error;
    if ((!enabled() && !capture_enabled()) || !TryAcquireSRWLockExclusive(&poll_lock)) return;
    const auto now = GetTickCount64();
    const auto loads = module_loads.load(std::memory_order_acquire);
    const bool notified = loader_cookie.load(std::memory_order_acquire) != nullptr;
    const bool pending = discovery_pending.load(std::memory_order_acquire);
    if (!pending && now >= next_poll &&
        (!notified || loads != scanned_loads || scan_truncated.load(std::memory_order_acquire))) {
      // Loads observed during this scan leave the count changed for the next one.
      scanned_loads = loads;
      next_poll = now + (notified ? 250 : 1000);
      discovery_pending.store(true, std::memory_order_release);
      if (!TrySubmitThreadpoolCallback(discover_on_pool, nullptr, nullptr)) {
        discover();
        discovery_pending.store(false, std::memory_order_release);
      }
    }
    // Coverage is reported after a scan completes, never from its starting state.
    if (enabled() && now >= next_report && !discovery_pending.load(std::memory_order_acquire)) {
      next_report = now + 5000;
      report();
    }
    sunshine_ngx::poll();
    ReleaseSRWLockExclusive(&poll_lock);
  }

  streamline_scope::streamline_scope(std::uint32_t feature, std::uintptr_t caller) :
      epoch_(sunshine_addon_lifetime::stopping() ? 0 : active_epoch.load(std::memory_order_acquire)), previous_epoch_(tls_epoch),
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
    invocation call;
    call.epoch = epoch_;
    call.caller = caller_;
    record(no_target, operation::streamline, call, feature_, nullptr, success);
  }

  streamline_scope::~streamline_scope() {
    tls_epoch = previous_epoch_;
    tls_depth = previous_depth_;
    tls_capture_depth = previous_capture_depth_;
  }
}
