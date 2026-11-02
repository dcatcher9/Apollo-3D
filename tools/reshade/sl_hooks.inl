// SPDX-License-Identifier: GPL-3.0-only
// Part of streamline_camera_probe.cpp: SL hook retention, installation,
// discovery and poll steps. Function definitions only: targets, flags, locks,
// tables, TLS and worker state stay at their original shell positions.
// FG lookup helpers run the old fragments at their old call points.
// The ordinary worker anonymous namespace and public guard APIs stay separate
// from the runtime-test private view. Included after the existing parts.

namespace sunshine_streamline {
  SUNSHINE_PROBE_INTERNAL {
    unsigned retain_fg_target(void *entry, std::uint64_t ticket) {
      if (!entry || !TryAcquireSRWLockExclusive(&fg_lock)) { fg_available = false; return UINT32_MAX; }
      if (!current(ticket)) { ReleaseSRWLockExclusive(&fg_lock); return UINT32_MAX; }
      if (current_fg_target.exchange(entry, std::memory_order_acq_rel) != entry) {
        fg_available = false;
        ++fg_loss_revision;
      }
      for (unsigned i = 0; i != fg_target_count; ++i) if (fg_targets[i].entry == entry) {
        fg_available.store(fg_targets[i].state.load(std::memory_order_acquire) != 2, std::memory_order_release);
        ReleaseSRWLockExclusive(&fg_lock); return i;
      }
      MEMORY_BASIC_INFORMATION memory{};
      HMODULE owner{};
      const bool executable = VirtualQuery(entry, &memory, sizeof(memory)) == sizeof(memory) &&
        sunshine_game3d::hook_host::executable_image(memory);
      if (fg_target_count == fg_targets.size() || !executable ||
          !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, reinterpret_cast<LPCWSTR>(entry), &owner)) {
        fg_available = false; ReleaseSRWLockExclusive(&fg_lock); return UINT32_MAX;
      }
      const auto index = fg_target_count++;
      auto &target = fg_targets[index];
      target.entry = entry; target.owner = owner;
      target.original.store(reinterpret_cast<fg_set_options>(entry), std::memory_order_release);
      target.state.store(0, std::memory_order_release);
      fg_available.store(true, std::memory_order_release);
      ReleaseSRWLockExclusive(&fg_lock);
      return index;
    }
    bool install_fg_hooks() {
      if (!observing.load(std::memory_order_acquire) || !middleware_requested()) return false;
      std::array<unsigned, 4> pending{}; unsigned count{};
      if (!TryAcquireSRWLockShared(&fg_lock)) return false;
      for (unsigned i = 0; i != fg_target_count; ++i)
        if (fg_targets[i].state.load(std::memory_order_acquire) == 0) pending[count++] = i;
      ReleaseSRWLockShared(&fg_lock);
      bool okay = true;
      for (unsigned i = 0; i != count; ++i) {
        const auto index = pending[i]; auto &slot = fg_targets[index];
        dump_metadata::observe_module(slot.owner, "streamline", "DLSS Frame Generation options");
        void *trampoline{};
        // retain_fg_target already holds the target module, and discovery pins
        // this add-on. Hooks/trampolines remain pass-through after shutdown.
        if (MH_CreateHook(slot.entry, reinterpret_cast<void *>(fg_hooks[index]), &trampoline) != MH_OK) {
          slot.state = 2; okay = false;
        } else {
          slot.original.store(reinterpret_cast<fg_set_options>(trampoline), std::memory_order_release);
          if (MH_EnableHook(slot.entry) == MH_OK) {
            slot.state.store(1, std::memory_order_release);
            message("Sunshine Streamline FG options observer: native entry hooked; cached and new function pointers are observed; mode remains game controlled");
          }
          else { slot.state = 2; okay = false; }
        }
        if (current_fg_target.load(std::memory_order_acquire) == slot.entry)
          fg_available.store(slot.state.load(std::memory_order_acquire) == 1, std::memory_order_release);
      }
      return okay;
    }
    void discover_fg_options() {
      const auto ticket = observation_ticket();
      if (!ticket || installed_abi != abi::v2 || !original_feature_function_v2) return;
      // Engines may cache this public function before the first ReShade poll.
      // Resolve its real entry at discovery so those cached pointers also pass
      // through our deferred detour. A lookup never establishes enabled state.
      // Same verdict as every 2.x call (eWarnOutOfVRAM is a success), but the
      // add-on's own lookup is not a game call, so vram_warnings ignores it.
      const DWORD saved = GetLastError();
      void *entry{};
      if (abi_v2::accepted(original_feature_function_v2(1000, "slDLSSGSetOptions", entry)) && current(ticket) && entry)
        retain_fg_target(entry, ticket);
      SetLastError(saved);
    }
    bool frame_generation_function_wanted(std::uint64_t ticket, std::uint32_t feature, const char *function_name) {
      constexpr char fg_expected[] = "slDLSSGSetOptions";
      char copied_fg_name[sizeof(fg_expected)]{};
      const bool fg_wanted = ticket && feature == 1000 &&
        read_bytes(function_name, copied_fg_name, sizeof(copied_fg_name)) && std::memcmp(copied_fg_name, fg_expected, sizeof(fg_expected)) == 0;
      return fg_wanted;
    }
    void offer_frame_generation_function(std::uint64_t ticket, bool fg_wanted, bool sdk_ok, void *&function) {
      if (fg_wanted && current(ticket)) {
        void *entry{};
        if (sdk_ok && read_bytes(&function, &entry, sizeof(entry))) {
          // The SDK helper caches this pointer and may set On only once. A
          // returned wrapper observes that first call without installing an
          // inline hook or suspending renderer threads in this callback. Poll
          // later detours the native entry to catch previously cached pointers.
          const auto index = retain_fg_target(entry, ticket);
          if (index < fg_hooks.size()) function = reinterpret_cast<void *>(fg_hooks[index]);
        }
        else { current_fg_target = nullptr; fg_available = false; }
      }
    }
    bool install_hooks(abi version, const targets &functions, const buffers::contract &contract) {
      if (version == abi::unsupported || !functions.constants || !functions.tag || !functions.evaluate) return false;
      if (!contract.known) return false;
      if (!sunshine_game3d::hook_host::minhook_ready(minhook_initialized)) return false;
      struct entry { void *target, *hook; void **original; };
      std::array<entry, 6> entries{};
      unsigned count{};
      if (versioning::is_legacy(version)) {
        entries[count++] = {functions.constants, reinterpret_cast<void *>(hook_constants_v1), reinterpret_cast<void **>(&original_constants_v1)};
        entries[count++] = {functions.tag, reinterpret_cast<void *>(hook_tag_v1), reinterpret_cast<void **>(&original_tag_v1)};
        entries[count++] = {functions.evaluate, reinterpret_cast<void *>(hook_evaluate_v1), reinterpret_cast<void **>(&original_evaluate_v1)};
      } else {
        entries[count++] = {functions.constants, reinterpret_cast<void *>(hook_constants_v2), reinterpret_cast<void **>(&original_constants_v2)};
        entries[count++] = {functions.tag, reinterpret_cast<void *>(hook_tag_v2), reinterpret_cast<void **>(&original_tag_v2)};
        if (functions.tag_for_frame) entries[count++] = {functions.tag_for_frame, reinterpret_cast<void *>(hook_frame_tag_v2), reinterpret_cast<void **>(&original_frame_tag_v2)};
        entries[count++] = {functions.evaluate, reinterpret_cast<void *>(hook_evaluate_v2), reinterpret_cast<void **>(&original_evaluate_v2)};
        if (functions.new_frame_token) entries[count++] = {functions.new_frame_token, reinterpret_cast<void *>(hook_new_token_v2), reinterpret_cast<void **>(&original_new_token_v2)};
        if (functions.get_feature_function) entries[count++] = {functions.get_feature_function, reinterpret_cast<void *>(hook_get_feature_function), reinterpret_cast<void **>(&original_feature_function_v2)};
      }
      unsigned made = 0;
      for (; made < count; ++made) {
        if (MH_CreateHook(entries[made].target, entries[made].hook, entries[made].original) != MH_OK) break;
      }
      if (made != count) {
        // None was enabled, so these trampolines cannot be running.
        for (unsigned i = 0; i < made; ++i) MH_RemoveHook(entries[i].target);
        return false;
      }
      // Detours canonicalize raw tags from their first call.
      installed_contract = contract;
      buffers::set_active_contract(contract);
      for (unsigned i = 0; i < count; ++i) {
        if (MH_QueueEnableHook(entries[i].target) != MH_OK) return false;
      }
      // On a partial enable failure retain all trampolines in pinned code and
      // disable observation; do not assume no thread entered a detour.
      if (MH_ApplyQueued() != MH_OK) return false;
      installed_targets = functions;
      installed_abi = version;
      hooks_installed = true;
      return true;
    }
    versioning::information module_version(HMODULE module) {
      wchar_t path[32768]{};
      const auto size = GetModuleFileNameW(module, path, static_cast<DWORD>(std::size(path)));
      if (!size || size == std::size(path)) {
        versioning::information out;
        out.reason = "module path unavailable or truncated";
        return out;
      }
      return versioning::read_file(path);
    }
    void discover_v1_private_state() {
      if (!versioning::is_legacy(installed_abi) || v1_private_state_checked) return;
      // The encoding is owned by sl.common, whose version can differ from the
      // interposer in a mixed installation. Resolve it once outside callbacks;
      // a delayed load remains eligible at the next existing discovery poll.
      HMODULE module{};
      if (!GetModuleHandleExW(0, L"sl.common.dll", &module)) return;
      dump_metadata::observe_module(module, "streamline", "1.1.1 resource-state adapter");
      auto info = module_version(module);
      const bool supported = versioning::private_state_v1_1_1(info);
      HMODULE pinned{};
      const bool ready = supported && GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
        reinterpret_cast<LPCWSTR>(module), &pinned);
      v1_private_state_checked = true;
      v1_private_state_ready.store(ready, std::memory_order_release);
      char text[640]{};
      std::snprintf(text, sizeof(text),
        "Sunshine Streamline: V1 private resource-state adapter=%s; sl.common fixed=%u.%u.%u.%u strings='%ls'/'%ls'; %s",
        ready ? "enabled" : "unsupported", HIWORD(info.file_ms), LOWORD(info.file_ms), HIWORD(info.file_ls), LOWORD(info.file_ls),
        info.file_text.data(), info.product_text.data(), supported && !ready ? "module lifetime could not be secured" : info.reason);
      message(text);
      FreeLibrary(module);
    }
    void discover() {
      // GetModuleHandleEx takes a temporary reference atomically, preventing
      // unload while exports/version are inspected. It never loads middleware.
      HMODULE module{};
      if (!GetModuleHandleExW(0, L"sl.interposer.dll", &module)) {
        if (!reported_missing) { message("Sunshine Streamline: waiting for an already-loaded sl.interposer.dll; waiting for supported Streamline observation"); reported_missing = true; }
        return;
      }
      auto info = module_version(module);
      const abi version = versioning::classify(info);
      // An exact identity's header, or every candidate of an inexact 2.x family.
      versioning::number number{};
      auto exact = info;
      const auto contract = version == abi::unsupported ? buffers::contract{} :
        versioning::identity(exact, number) ? buffers::for_version(number) : buffers::for_versions(versioning::family(info));
      dump_metadata::observe_module(module, "streamline", versioning::name(version));
      targets functions{
        reinterpret_cast<void *>(GetProcAddress(module, "slSetConstants")),
        reinterpret_cast<void *>(GetProcAddress(module, "slSetTag")),
        reinterpret_cast<void *>(GetProcAddress(module, "slSetTagForFrame")),
        reinterpret_cast<void *>(GetProcAddress(module, "slEvaluateFeature")),
        reinterpret_cast<void *>(GetProcAddress(module, "slGetNewFrameToken")),
        reinterpret_cast<void *>(GetProcAddress(module, "slGetFeatureFunction"))};
      // 1.0.0/1.0.1 shipped namespace exports; 1.0.2 kept that resource layout
      // but introduced the C names. These are the official Windows x64 names.
      if (version == abi::legacy_no_state) {
        if (!functions.constants) functions.constants = reinterpret_cast<void *>(GetProcAddress(module, "?setConstants@sl@@YA_NAEBUConstants@1@II@Z"));
        if (!functions.tag) functions.tag = reinterpret_cast<void *>(GetProcAddress(module, "?setTag@sl@@YA_NPEBUResource@1@W4BufferType@1@IPEBUExtent@1@@Z"));
        if (!functions.evaluate) functions.evaluate = reinterpret_cast<void *>(GetProcAddress(module, "?evaluateFeature@sl@@YA_NPEAXW4Feature@1@II@Z"));
      }
      char identity[768]{};
      std::snprintf(identity, sizeof(identity),
        "Sunshine Streamline: version fixed-file=%u.%u.%u.%u fixed-product=%u.%u.%u.%u strings='%ls'/'%ls' result=%s; adapter=%s; exports constants=%d tag=%d frame-tag=%d evaluate=%d new-frame-token=%d feature-function=%d",
        HIWORD(info.file_ms), LOWORD(info.file_ms), HIWORD(info.file_ls), LOWORD(info.file_ls),
        HIWORD(info.product_ms), LOWORD(info.product_ms), HIWORD(info.product_ls), LOWORD(info.product_ls),
        info.file_text.data(), info.product_text.data(), info.reason, versioning::name(version),
        functions.constants != nullptr, functions.tag != nullptr, functions.tag_for_frame != nullptr, functions.evaluate != nullptr,
        functions.new_frame_token != nullptr, functions.get_feature_function != nullptr);
      message(identity);
      if (version == abi::unsupported || !functions.constants || !functions.tag || !functions.evaluate) {
        message(version == abi::unsupported ?
          "Sunshine Streamline: camera hooks disabled because the version identity or ABI major above is unsupported" :
          "Sunshine Streamline: camera hooks disabled because a required export above is missing");
        permanently_rejected = true;
        FreeLibrary(module);
        return;
      }
      HMODULE pinned_addon{}, pinned_interposer{}, own_module{};
      // Never pin the fixture process executable or another caller's module.
      const bool owned = addon_module && addon_module != GetModuleHandleW(nullptr) &&
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          reinterpret_cast<LPCWSTR>(&discover), &own_module) && own_module == addon_module;
      if (!owned || !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
            reinterpret_cast<LPCWSTR>(&discover), &pinned_addon) ||
          !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
            reinterpret_cast<LPCWSTR>(functions.constants), &pinned_interposer)) {
        message("Sunshine Streamline: module lifetime could not be secured; camera hooks disabled");
        permanently_rejected = true;
        FreeLibrary(module);
        return;
      }
      const bool installed = install_hooks(version, functions, contract);
      FreeLibrary(module);
      if (!installed) {
        observing.store(false, std::memory_order_release);
        permanently_rejected = true;
        message("Sunshine Streamline: inline-hook setup failed; retained code stays pass-through and camera observation is disabled");
        return;
      }
      observing.store(middleware_requested(), std::memory_order_release);
      message(versioning::is_legacy(version) ?
        "Sunshine Streamline: observing 1.x constants, global depth tags and feature evaluation; direct depth capture enabled when supported" :
        "Sunshine Streamline: observing 2.x constants, global/local depth tags and feature evaluation; direct depth capture enabled when supported");
      char candidates_text[128]{};
      for (unsigned i = 0, at = 0; i != contract.candidate_count && at < sizeof(candidates_text); ++i) {
        const auto &v = contract.candidates[i];
        at += unsigned(std::snprintf(candidates_text + at, sizeof(candidates_text) - at, "%s%u.%u.%u.%u", i ? "," : " candidates=",
          v.major, v.minor, v.patch, v.build));
      }
      char buffers_text[448]{};
      std::snprintf(buffers_text, sizeof(buffers_text),
        "Sunshine Streamline: buffer contract %u.%u.%u%s%s last_type=%u ui_alpha_tag=%d ui_color_alpha_tag=%s surveyed=%u ui_alpha_auto=%u",
        contract.version.major, contract.version.minor, contract.version.patch,
        contract.candidate_count > 1 ? " (v2 family, inexact version; a tag is known only where every candidate agrees)" : "",
        candidates_text, contract.last_type,
        contract.raw_ui_alpha == buffers::unknown ? -1 : int(contract.raw_ui_alpha),
        contract.ui_color_and_alpha ? "23" : "none", contract.surveyed ? 1u : 0u,
        buffers::ui_alpha_authenticated(contract) ? 1u : 0u);
      message(buffers_text);
      if (version == abi::v2 && !functions.tag_for_frame)
        message("Sunshine Streamline: optional slSetTagForFrame unavailable; global/evaluation-local tags remain enabled");
      if (version == abi::v2 && !functions.new_frame_token)
        message("Sunshine Streamline: slGetNewFrameToken unavailable; v2 evaluation frame evidence stays untracked");
    }
  }

  namespace {
    bool fg_targets_pending() {
      if (!TryAcquireSRWLockShared(&fg_lock)) return false;
      bool pending = false;
      for (unsigned i = 0; i != fg_target_count && !pending; ++i)
        pending = fg_targets[i].state.load(std::memory_order_acquire) == 0;
      ReleaseSRWLockShared(&fg_lock);
      return pending;
    }
    void install_hook_work() {
      if (!hooks_installed && !permanently_rejected) discover();
      if (middleware_requested() && observing.load(std::memory_order_acquire)) {
        install_fg_hooks();
      }
    }
    void CALLBACK install_on_pool(PTP_CALLBACK_INSTANCE, void *) {
      AcquireSRWLockExclusive(&poll_lock);
      if (!sunshine_addon_lifetime::stopping()) install_hook_work();
      install_scheduled.store(false, std::memory_order_release);
      ReleaseSRWLockExclusive(&poll_lock);
    }
    void schedule_install() {
      if (install_scheduled.exchange(true, std::memory_order_acq_rel)) return;
      if (sunshine_game3d::pool::submit(sunshine_game3d::pool::family::sl_install, install_on_pool, nullptr)) return;
      install_hook_work(); // Without a pool thread, install here as before.
      install_scheduled.store(false, std::memory_order_release);
    }
  }

  poll_guard::~poll_guard() {
    if (acquired_) ReleaseSRWLockExclusive(&poll_lock);
  }

  poll_guard try_poll() {
    return poll_guard(TryAcquireSRWLockExclusive(&poll_lock) != FALSE);
  }

  bool middleware_requested_for_poll() {
    return middleware_requested();
  }

  void discovery_step(bool api_requested, std::uint64_t now) {
    if (api_requested && now >= next_discovery) {
      next_discovery = now + 1000;
      if (!hooks_installed && !permanently_rejected) schedule_install();
      if (hooks_installed) discover_v1_private_state();
      if (hooks_installed) discover_fg_options();
    }
  }

  void coverage_step() {
    // Call routing is independent of per-draw diagnostics and source selection.
    // A zero count is only useful together with actual hook coverage.
    sunshine_upscaler_trace::set_streamline_coverage(hooks_installed);
  }

  void camera_availability_step(std::uint64_t now) {
    report_camera_availability(now);
  }

  void depth_capture_step() {
    if (sunshine_game3d::hook_host::minhook_ready(minhook_initialized)) depth_capture::poll();
  }

  void queue_order_step(std::uint64_t now) {
    char order_line[1024];
    if (depth_capture::queue_order_line(now, order_line, sizeof(order_line)))
      sunshine_log::message(reshade::log::level::info, order_line);
  }

  void pending_installs_step() {
    if (middleware_requested() && observing.load(std::memory_order_acquire) && fg_targets_pending())
      schedule_install();
  }

}
