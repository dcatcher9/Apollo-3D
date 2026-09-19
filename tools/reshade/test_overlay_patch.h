// SPDX-License-Identifier: GPL-3.0-only
#pragma once

// Included only by the distinct runtime-test add-on, after imgui.h and reshade.hpp.
// These primitives go through ReShade's actual GUI renderer and its normal alpha blend.
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  #include <atomic>

// Calls the very same registered production panel from a real ImGui frame.
// The fixture must render it, not only exercise its model through API adapters.
void sunshine_game3d_test_draw_production_panel(reshade::api::effect_runtime *runtime, bool expanded);

namespace sunshine_sbs_test_overlay {
  inline std::atomic<bool> enabled {false};
  inline std::atomic<unsigned> production_panel_mode {0};
  inline std::atomic<unsigned> production_panel_draws {0};

  inline void draw(reshade::api::effect_runtime *runtime) {
    if (const unsigned mode = production_panel_mode.load(std::memory_order_acquire)) {
      // Keep the fixture's independent right-hand HDR/alpha probes uncovered.
      ImGui::SetNextWindowPos(ImVec2(8, 48), ImGuiCond_Always);
      ImGui::SetNextWindowSize(ImVec2(630, 440), ImGuiCond_Always);
      if (ImGui::Begin("Sunshine production panel regression", nullptr,
          ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDocking)) {
        sunshine_game3d_test_draw_production_panel(runtime, mode == 2);
        production_panel_draws.fetch_add(1, std::memory_order_release);
      }
      ImGui::End();
    }
    if (!enabled.load(std::memory_order_acquire)) {
      return;
    }
    const auto &size = ImGui::GetIO().DisplaySize;
    auto *list = ImGui::GetForegroundDrawList(nullptr);
    const float x = size.x - 120.0f;
    list->AddRectFilled(ImVec2(x, size.y - 160.0f), ImVec2(x + 48.0f, size.y - 128.0f), IM_COL32(255, 255, 255, 255));
    list->AddRectFilled(ImVec2(x, size.y - 96.0f), ImVec2(x + 48.0f, size.y - 64.0f), IM_COL32(255, 255, 255, 128));
  }
}  // namespace sunshine_sbs_test_overlay

extern "C" __declspec(dllexport) void SunshineSbsTestSetOverlayPatch(BOOL enabled) {
  sunshine_sbs_test_overlay::enabled.store(enabled != FALSE, std::memory_order_release);
}

extern "C" __declspec(dllexport) void SunshineSbsTestSetProductionPanel(unsigned mode) {
  sunshine_sbs_test_overlay::production_panel_mode.store(mode <= 2 ? mode : 0, std::memory_order_release);
}

extern "C" __declspec(dllexport) unsigned SunshineSbsTestProductionPanelDraws() {
  return sunshine_sbs_test_overlay::production_panel_draws.load(std::memory_order_acquire);
}
#endif
