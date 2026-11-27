// SPDX-License-Identifier: GPL-3.0-only
#include "upscaler_call_trace.h"
#include "addon_lifetime.h"

#include <reshade.hpp>
#include "async_log.h"
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include "game3d_clock.h"

namespace sunshine_upscaler_trace {
  namespace {
    using namespace feed;
    constexpr unsigned event_limit = 64, handle_limit = 64;
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
    // No destructible locks/containers or workers: detours stay valid after the
    // add-on's normal shutdown and continue calling their immutable originals.
    SRWLOCK records_lock = SRWLOCK_INIT;
    std::array<event, event_limit> events;
    std::array<event, event_limit> reported_events; // Report-owned; never read by hooks.
    std::array<unsigned, target_limit> reported_coverage{};
    std::array<handle_record, handle_limit> handles;
    // Exactly one bounded stack sample per target and observation epoch.
    std::array<std::atomic<std::uint64_t>, target_limit> sampled_epoch{};
    std::atomic<std::uint64_t> active_epoch{}, dropped{}, stale{};
    std::atomic<bool> streamline_covered{};
    std::atomic<bool> handle_observation_gap{};
    std::uint64_t next_epoch{}, began{}, reports{};

    const char *op_name(operation op) {
      switch (op) {
      case operation::create: return "create";
      case operation::evaluate: return "evaluate";
      case operation::release: return "release";
      default: return "SL-evaluate";
      }
    }
    void message(const char *text) { sunshine_log::message(reshade::log::level::info, text); }
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
    handle_record *find_handle(HMODULE owner, bool d3d11, const void *handle, bool create) {
      handle_record *empty = nullptr;
      for (auto &value : handles) {
        if (!value.handle) { if (!empty) empty = &value; continue; }
        if (value.owner == owner && value.d3d11 == d3d11 && value.handle == handle) return &value;
      }
      return create ? empty : nullptr;
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
  }

  bool enabled() { return !sunshine_addon_lifetime::stopping() && active_epoch.load(std::memory_order_acquire) != 0; }

  void set_streamline_coverage(bool installed) { streamline_covered.store(installed, std::memory_order_release); }

  namespace feed {
    std::uint64_t epoch() { return active_epoch.load(std::memory_order_acquire); }

    void restart(bool enabled) {
      active_epoch.store(0, std::memory_order_release);
      AcquireSRWLockExclusive(&records_lock);
      events = {};
      reported_events = {};
      reported_coverage = {};
      handles = {};
      handle_observation_gap.store(false);
      dropped = stale = 0;
      reports = 0;
      began = sunshine_game3d::clock::game_now();
      if (enabled) active_epoch.store(++next_epoch, std::memory_order_release);
      ReleaseSRWLockExclusive(&records_lock);
    }

    void stop() { active_epoch.store(0, std::memory_order_release); }

    invocation begin(unsigned target, std::uintptr_t caller, std::uint64_t sl_epoch, unsigned sl_depth) {
      invocation value;
      if (sunshine_addon_lifetime::stopping()) return value;
      value.epoch = active_epoch.load(std::memory_order_acquire);
      value.caller = caller;
      value.inside_sl = value.epoch && sl_epoch == value.epoch && sl_depth != 0;
      if (value.epoch && sampled_epoch[target].exchange(value.epoch, std::memory_order_relaxed) != value.epoch) {
        // Exactly one bounded sample per export and observation epoch. No
        // module lookup, symbol engine, allocation, logging or GPU work here.
        value.stack_count = CaptureStackBackTrace(1, stack_limit, value.stack.data(), nullptr);
      }
      return value;
    }

    void record(unsigned index, HMODULE owner, bool d3d11, operation op, const invocation &call,
        std::uint32_t feature, const void *handle, bool success, std::uint32_t failure) {
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
        // A missed create/release can reuse an address. Never retain a feature
        // identity across missing evidence; subsequent evaluations are unknown
        // until a new successful create is observed.
        if (handle_observation_gap.exchange(false, std::memory_order_acq_rel)) handles = {};
        if (op == operation::create && success && !handle) handles = {};
        if (op == operation::create && success && handle) {
          if (auto *value = find_handle(owner, d3d11, handle, true))
            *value = {owner, handle, feature, d3d11};
          else ++dropped;
        } else if (op != operation::create) {
          if (auto *value = find_handle(owner, d3d11, handle, false)) {
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

    void report(const coverage &hooks) {
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
      for (unsigned i = 0; i != hooks.count; ++i) installed += hooks.targets[i].state == 2;
      char text[1200]{};
      std::snprintf(text, sizeof(text),
        "Sunshine upscaler trace: epoch=%llu elapsed_ms=%llu SL_hook=%u modules=%u truncated=%u truncations=%u skipped_frame_generation=%u exports=%u NGX_hooks=%u/%u rejected=%u SL_evaluate=%llu NGX_create=%llu NGX_evaluate=%llu NGX_release=%llu inside_observed_SL=%llu outside_observed_SL=%llu unknown_feature=%llu dropped=%llu stale=%llu; zero means no observed calls in this window, not API absence; outside is not proof of a direct path",
        static_cast<unsigned long long>(active_epoch.load()), static_cast<unsigned long long>(sunshine_game3d::clock::game_now() - began),
        streamline_covered.load() ? 1 : 0, hooks.modules, hooks.truncated ? 1u : 0u,
        hooks.truncations, hooks.skipped_frame_generation, hooks.exports, installed, hooks.count, hooks.rejected,
        static_cast<unsigned long long>(counts[3]), static_cast<unsigned long long>(counts[0]),
        static_cast<unsigned long long>(counts[1]), static_cast<unsigned long long>(counts[2]),
        static_cast<unsigned long long>(inside), static_cast<unsigned long long>(outside),
        static_cast<unsigned long long>(unknown), static_cast<unsigned long long>(dropped.load()), static_cast<unsigned long long>(stale.load()));
      message(text);
      for (unsigned i = 0; i != hooks.count; ++i) {
        const auto &target = hooks.targets[i];
        if (reported_coverage[i] == target.state) continue;
        reported_coverage[i] = target.state;
        char owner[320]{};
        location(reinterpret_cast<std::uintptr_t>(target.entry), owner, sizeof(owner));
        std::snprintf(text, sizeof(text), "Sunshine upscaler trace coverage: target=%u D3D%u %s %s state=%s",
          i, target.d3d11 ? 11 : 12, op_name(target.op), owner,
          target.state == 2 ? "hooked" : "unavailable");
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

#ifdef SUNSHINE_UPSCALER_TRACE_TEST
    counts test_counts() {
      counts out;
      out.epoch = active_epoch.load();
      out.streamline_covered = streamline_covered.load();
      out.dropped = dropped.load();
      out.stale = stale.load();
      out.reports = reports;
      AcquireSRWLockShared(&records_lock);
      for (const auto &value : events) if (value.used) {
        switch (value.op) {
        case operation::create: out.ngx_create += value.calls; break;
        case operation::evaluate: out.ngx_evaluate += value.calls; break;
        case operation::release: out.ngx_release += value.calls; break;
        default: out.streamline += value.calls; break;
        }
        out.succeeded += value.succeeded;
        out.failed += value.failed;
        if (value.op != operation::streamline) {
          (value.inside_sl ? out.inside_streamline : out.outside_streamline) += value.calls;
          if (value.feature == unknown_feature) out.unknown_feature += value.calls;
        }
      }
      ReleaseSRWLockShared(&records_lock);
      return out;
    }
#endif
  }
}
