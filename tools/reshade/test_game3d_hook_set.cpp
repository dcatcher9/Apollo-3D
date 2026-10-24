// SPDX-License-Identifier: GPL-3.0-only
// The shipping add-on's installed hook set (refactor plan P4's baseline):
// which ReShade events, in which order, its AddonInit registers and its
// AddonUninit unregisters, its overlay, and the MinHook hooks it creates in a
// process without Streamline or NGX. This binary is that ReShade: it exports
// the add-on API that reshade.hpp looks up and records each call. A change
// here is a change of the add-on's hooks; only a commit that means to make
// one updates these tables. Each MinHook family's hooks on fixture exports are
// pinned where they are installed: the Streamline probe and NGX call-trace
// tests.
#include "test_minhook_ledger.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <reshade.hpp>
#include <string>
#include <vector>

extern "C" {
  bool AddonInit(HMODULE addon, HMODULE reshade);
  void AddonUninit(HMODULE addon, HMODULE reshade);
}

namespace {
  enum class call {
    register_event,
    unregister_event,
    register_overlay,
    unregister_overlay,
    unregister_addon
  };

  struct recorded {
    call kind;
    std::uint32_t event;
    const void *callback;
  };

  std::vector<recorded> calls;
  std::string overlay;
  unsigned registered_addons {}, failures {};

  void fail(const std::string &text) {
    ++failures;
    std::fprintf(stderr, "FAIL %s\n", text.c_str());
  }
}  // namespace

// The add-on API of ReShade 6.8 (reshade.hpp), as this process's ReShade.
extern "C" __declspec(dllexport) bool ReShadeRegisterAddon(void *, std::uint32_t version) {
  ++registered_addons;
  return version == RESHADE_API_VERSION;
}

extern "C" __declspec(dllexport) void ReShadeUnregisterAddon(void *) {
  calls.push_back({call::unregister_addon, 0, nullptr});
}

// register_addon requires Dear ImGui's function table. The add-on draws
// nothing here, so no entry is ever called.
extern "C" __declspec(dllexport) const void *ReShadeGetImGuiFunctionTable(std::uint32_t) {
  static std::array<void *, 1024> table {};
  return table.data();
}

extern "C" __declspec(dllexport) void ReShadeRegisterEvent(reshade::addon_event event, void *callback) {
  calls.push_back({call::register_event, static_cast<std::uint32_t>(event), callback});
}

extern "C" __declspec(dllexport) void ReShadeUnregisterEvent(reshade::addon_event event, void *callback) {
  calls.push_back({call::unregister_event, static_cast<std::uint32_t>(event), callback});
}

extern "C" __declspec(dllexport) void ReShadeRegisterOverlay(const char *title, void (*callback)(reshade::api::effect_runtime *)) {
  overlay = title ? title : "";
  calls.push_back({call::register_overlay, 0, reinterpret_cast<const void *>(callback)});
}

extern "C" __declspec(dllexport) void ReShadeUnregisterOverlay(const char *, void (*callback)(reshade::api::effect_runtime *)) {
  calls.push_back({call::unregister_overlay, 0, reinterpret_cast<const void *>(callback)});
}

extern "C" __declspec(dllexport) void ReShadeLogMessage(void *, int, const char *) {}

extern "C" __declspec(dllexport) bool ReShadeGetBasePath(char *, size_t *) {
  return false;
}

// A configuration where the depth selector registers: the built-in Generic
// Depth is disabled, as the add-on's own first launch arranges. Every other
// setting is absent, so each keeps its default. Like ReShade, a size query
// counts the array's final terminator, and a read returns the array's size.
extern "C" __declspec(dllexport) bool ReShadeGetConfigValue(void *, reshade::api::effect_runtime *, const char *section, const char *key, char *value, size_t *size) {
  static constexpr char disabled[] = "Generic Depth\0Sunshine Depth";  // Each name ends in a NUL.
  if (!size || std::strcmp(section, "ADDON") != 0 || std::strcmp(key, "DisabledAddons") != 0) {
    return false;
  }
  if (!value || *size <= sizeof(disabled)) {
    *size = sizeof(disabled) + 1;
    return value == nullptr;
  }
  std::memcpy(value, disabled, sizeof(disabled));
  value[sizeof(disabled)] = '\0';
  *size = sizeof(disabled);
  return true;
}

extern "C" __declspec(dllexport) void ReShadeSetConfigValue(void *, reshade::api::effect_runtime *, const char *, const char *, const char *) {}

extern "C" __declspec(dllexport) void ReShadeSetConfigArray(void *, reshade::api::effect_runtime *, const char *, const char *, const char *, size_t) {}

namespace {
  using event = reshade::addon_event;

  struct expected {
    call kind;
    event value;
    const char *name;
  };

#define SUNSHINE_HOOK(kind, name) {call::kind, event::name, #name}

  // AddonInit: the Game 3D controls, the depth selector (only when Generic
  // Depth is disabled), the exporter, then the UI layer.
  constexpr expected registered[] {
    SUNSHINE_HOOK(register_event, destroy_effect_runtime),
    SUNSHINE_HOOK(register_event, init_device),
    SUNSHINE_HOOK(register_event, init_command_list),
    SUNSHINE_HOOK(register_event, init_command_queue),
    SUNSHINE_HOOK(register_event, init_effect_runtime),
    SUNSHINE_HOOK(register_event, destroy_device),
    SUNSHINE_HOOK(register_event, destroy_command_list),
    SUNSHINE_HOOK(register_event, destroy_command_queue),
    SUNSHINE_HOOK(register_event, destroy_effect_runtime),
    SUNSHINE_HOOK(register_event, create_resource),
    SUNSHINE_HOOK(register_event, create_resource_view),
    SUNSHINE_HOOK(register_event, init_resource),
    SUNSHINE_HOOK(register_event, destroy_resource),
    SUNSHINE_HOOK(register_event, draw),
    SUNSHINE_HOOK(register_event, draw_indexed),
    SUNSHINE_HOOK(register_event, draw_or_dispatch_indirect),
    SUNSHINE_HOOK(register_event, dispatch_mesh),
    SUNSHINE_HOOK(register_event, copy_resource),
    SUNSHINE_HOOK(register_event, copy_texture_region),
    SUNSHINE_HOOK(register_event, copy_buffer_to_texture),
    SUNSHINE_HOOK(register_event, resolve_texture_region),
    SUNSHINE_HOOK(register_event, barrier),
    SUNSHINE_HOOK(register_event, end_render_pass),
    SUNSHINE_HOOK(register_event, close_command_list),
    SUNSHINE_HOOK(register_event, bind_viewports),
    SUNSHINE_HOOK(register_event, begin_render_pass),
    SUNSHINE_HOOK(register_event, bind_render_targets_and_depth_stencil),
    SUNSHINE_HOOK(register_event, clear_depth_stencil_view),
    SUNSHINE_HOOK(register_event, reset_command_list),
    SUNSHINE_HOOK(register_event, execute_command_list),
    SUNSHINE_HOOK(register_event, execute_secondary_command_list),
    SUNSHINE_HOOK(register_event, present),
    SUNSHINE_HOOK(register_event, finish_present),
    SUNSHINE_HOOK(register_event, reshade_begin_effects),
    SUNSHINE_HOOK(register_event, reshade_finish_effects),
    SUNSHINE_HOOK(register_event, reshade_reloaded_effects),
    SUNSHINE_HOOK(register_event, reshade_set_technique_state),
    SUNSHINE_HOOK(register_event, init_effect_runtime),
    SUNSHINE_HOOK(register_event, reshade_begin_effects),
    SUNSHINE_HOOK(register_event, reshade_render_technique),
    SUNSHINE_HOOK(register_event, reshade_reloaded_effects),
    SUNSHINE_HOOK(register_event, destroy_effect_runtime),
    SUNSHINE_HOOK(register_event, destroy_swapchain),
    SUNSHINE_HOOK(register_event, destroy_device),
    SUNSHINE_HOOK(register_event, reshade_present),
    SUNSHINE_HOOK(register_event, present),
    SUNSHINE_HOOK(register_event, finish_present),
    SUNSHINE_HOOK(register_event, reshade_open_overlay),
    SUNSHINE_HOOK(register_event, reshade_overlay),
    SUNSHINE_HOOK(register_event, clear_render_target_view),
    SUNSHINE_HOOK(register_event, begin_render_pass),
    SUNSHINE_HOOK(register_event, destroy_device),
    SUNSHINE_HOOK(register_event, execute_command_list),
    SUNSHINE_HOOK(register_event, execute_secondary_command_list),
    SUNSHINE_HOOK(register_event, reset_command_list),
    SUNSHINE_HOOK(register_event, init_swapchain),
    SUNSHINE_HOOK(register_event, destroy_swapchain),
  };

  // AddonUninit: what the add-on unregisters itself (the UI layer, the Game
  // 3D controls, the depth selector) before ReShade drops the exporter's
  // events with the add-on.
  constexpr expected unregistered[] {
    SUNSHINE_HOOK(unregister_event, clear_render_target_view),
    SUNSHINE_HOOK(unregister_event, begin_render_pass),
    SUNSHINE_HOOK(unregister_event, destroy_device),
    SUNSHINE_HOOK(unregister_event, execute_command_list),
    SUNSHINE_HOOK(unregister_event, execute_secondary_command_list),
    SUNSHINE_HOOK(unregister_event, reset_command_list),
    SUNSHINE_HOOK(unregister_event, init_swapchain),
    SUNSHINE_HOOK(unregister_event, destroy_swapchain),
    SUNSHINE_HOOK(unregister_event, destroy_effect_runtime),
    SUNSHINE_HOOK(unregister_event, init_device),
    SUNSHINE_HOOK(unregister_event, init_command_list),
    SUNSHINE_HOOK(unregister_event, init_command_queue),
    SUNSHINE_HOOK(unregister_event, init_effect_runtime),
    SUNSHINE_HOOK(unregister_event, destroy_device),
    SUNSHINE_HOOK(unregister_event, destroy_command_list),
    SUNSHINE_HOOK(unregister_event, destroy_command_queue),
    SUNSHINE_HOOK(unregister_event, destroy_effect_runtime),
    SUNSHINE_HOOK(unregister_event, create_resource),
    SUNSHINE_HOOK(unregister_event, create_resource_view),
    SUNSHINE_HOOK(unregister_event, init_resource),
    SUNSHINE_HOOK(unregister_event, destroy_resource),
    SUNSHINE_HOOK(unregister_event, draw),
    SUNSHINE_HOOK(unregister_event, draw_indexed),
    SUNSHINE_HOOK(unregister_event, draw_or_dispatch_indirect),
    SUNSHINE_HOOK(unregister_event, dispatch_mesh),
    SUNSHINE_HOOK(unregister_event, copy_resource),
    SUNSHINE_HOOK(unregister_event, copy_texture_region),
    SUNSHINE_HOOK(unregister_event, copy_buffer_to_texture),
    SUNSHINE_HOOK(unregister_event, resolve_texture_region),
    SUNSHINE_HOOK(unregister_event, barrier),
    SUNSHINE_HOOK(unregister_event, end_render_pass),
    SUNSHINE_HOOK(unregister_event, close_command_list),
    SUNSHINE_HOOK(unregister_event, bind_viewports),
    SUNSHINE_HOOK(unregister_event, begin_render_pass),
    SUNSHINE_HOOK(unregister_event, bind_render_targets_and_depth_stencil),
    SUNSHINE_HOOK(unregister_event, clear_depth_stencil_view),
    SUNSHINE_HOOK(unregister_event, reset_command_list),
    SUNSHINE_HOOK(unregister_event, execute_command_list),
    SUNSHINE_HOOK(unregister_event, execute_secondary_command_list),
    SUNSHINE_HOOK(unregister_event, present),
    SUNSHINE_HOOK(unregister_event, finish_present),
    SUNSHINE_HOOK(unregister_event, reshade_begin_effects),
    SUNSHINE_HOOK(unregister_event, reshade_finish_effects),
    SUNSHINE_HOOK(unregister_event, reshade_reloaded_effects),
    SUNSHINE_HOOK(unregister_event, reshade_set_technique_state),
  };

#undef SUNSHINE_HOOK

  const char *kind_name(call kind) {
    switch (kind) {
      case call::register_event:
        return "register_event";
      case call::unregister_event:
        return "unregister_event";
      case call::register_overlay:
        return "register_overlay";
      case call::unregister_overlay:
        return "unregister_overlay";
      case call::unregister_addon:
        return "unregister_addon";
    }
    return "?";
  }

  // The event calls in order against a table; the overlay and add-on calls
  // are checked by the caller.
  void compare(const char *phase, const std::vector<recorded> &got, const expected *table, std::size_t count) {
    std::vector<recorded> events;
    for (const auto &value : got) {
      if (value.kind == call::register_event || value.kind == call::unregister_event) {
        events.push_back(value);
      }
    }
    for (std::size_t i = 0; i < std::max(events.size(), count); ++i) {
      const bool have = i < events.size(), want = i < count;
      if (have && want && events[i].kind == table[i].kind && events[i].event == static_cast<std::uint32_t>(table[i].value) && events[i].callback) {
        continue;
      }
      char text[256];
      std::snprintf(text, sizeof(text), "%s call %zu: %s event %u, expected %s %s", phase, i, have ? kind_name(events[i].kind) : "none", have ? events[i].event : 0u, want ? kind_name(table[i].kind) : "none", want ? table[i].name : "-");
      fail(text);
    }
  }
}  // namespace

int main() {
  const auto self = GetModuleHandleW(nullptr);
  const auto hooks = minhook_ledger::mark();
  if (!AddonInit(self, self)) {
    std::fputs("FAIL AddonInit refused the add-on\n", stderr);
    return 1;
  }
  const auto initialized = calls;
  compare("AddonInit", initialized, registered, std::size(registered));
  unsigned overlays = 0;
  for (const auto &value : initialized) {
    overlays += value.kind == call::register_overlay;
  }
  if (registered_addons != 1 || overlays != 1 || overlay != "Sunshine 3D") {
    fail("AddonInit did not register one add-on and its one Sunshine 3D overlay");
  }
  // Two callbacks of one event are two different functions.
  for (std::size_t i = 0; i < initialized.size(); ++i) {
    for (std::size_t j = i + 1; j < initialized.size(); ++j) {
      if (initialized[i].kind == call::register_event && initialized[j].kind == call::register_event && initialized[i].event == initialized[j].event && initialized[i].callback == initialized[j].callback) {
        fail("AddonInit registered one callback twice for event " + std::to_string(initialized[i].event));
      }
    }
  }
  const auto created = minhook_ledger::created_since(hooks);
  if (minhook_ledger::available() && !created.empty()) {
    fail("AddonInit created " + std::to_string(created.size()) + " MinHook hooks without Streamline or NGX");
  }

  calls.clear();
  AddonUninit(self, self);
  compare("AddonUninit", calls, unregistered, std::size(unregistered));
  // Every event the add-on unregisters itself is one it registered, with the same callback.
  unsigned overlay_removed = 0, addon_removed = 0;
  for (const auto &value : calls) {
    overlay_removed += value.kind == call::unregister_overlay;
    addon_removed += value.kind == call::unregister_addon;
    if (value.kind != call::unregister_event) {
      continue;
    }
    bool found = false;
    for (const auto &before : initialized) {
      found = found || (before.kind == call::register_event && before.event == value.event && before.callback == value.callback);
    }
    if (!found) {
      fail("AddonUninit unregistered a callback it never registered for event " + std::to_string(value.event));
    }
  }
  if (overlay_removed != 1 || addon_removed != 1 || !calls.empty() && calls.back().kind != call::unregister_addon) {
    fail("AddonUninit did not remove its overlay, then unregister the add-on last");
  }
  if (failures) {
    std::fprintf(stderr, "Hook set: %u differences\n", failures);
    return 1;
  }
  std::printf("PASS hook set: %zu event registrations, %zu unregistrations, one overlay, %s MinHook hooks\n", std::size(registered), std::size(unregistered), minhook_ledger::available() ? "no" : "unrecorded");
  return 0;
}
