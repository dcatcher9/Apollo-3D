#pragma once

#include <memory>
#include <optional>
#include <string_view>
#include <windows.h>

namespace platf {
  namespace monitor_desktop_detail {
    // Exact known security identities; ordinary Explorer and UWP application windows remain valid.
    bool is_security_surface(std::wstring_view window_class, std::wstring_view process_image);
    bool valid_rect(RECT rect);
    bool contains(RECT rect, POINT point);
    bool contains_center(RECT monitor, RECT window);
    std::optional<RECT> map_rect(RECT window, RECT from, RECT to);
    std::optional<POINT> map_point(POINT point, RECT from, RECT to);
    // Minimized windows' screen rectangles can be the shell's off-screen icon rectangle.
    std::optional<RECT> window_ownership_rect(RECT screen_rect, RECT normal_rect, bool minimized);
    std::optional<WINDOWPLACEMENT> placement_for_move(WINDOWPLACEMENT placement, RECT normal_workspace_rect, bool minimized, bool visible = true);
  }  // namespace monitor_desktop_detail

  /** Moves the selected physical desktop's applications onto the private capture desktop.
   * Own this beside the virtual-display session. Capture before changing topology, activate only
   * after the source rectangle is final, then restore after physical topology/cursor confinement
   * has been restored but before retiring the virtual source. Presenter-only restarts retain it.
   */
  class monitor_desktop_handoff_t {
  public:
    monitor_desktop_handoff_t();
    ~monitor_desktop_handoff_t();
    monitor_desktop_handoff_t(const monitor_desktop_handoff_t &) = delete;
    monitor_desktop_handoff_t &operator=(const monitor_desktop_handoff_t &) = delete;

    bool capture(RECT original_monitor_rect);
    bool activate(RECT virtual_source_rect);
    // Controller-only reconciliation for applications/dialogs opening on the parked physical
    // sink. Nonblocking: false retains pending migration/restoration ownership for the next poll.
    bool refresh(RECT physical_sink_rect, RECT virtual_source_rect);
    // Freeze source ownership immediately before topology restoration can remap its windows.
    void prepare_restore();
    // False retains ownership/snapshot for retry; do not retire the source on restoration failure.
    bool restore(RECT restored_physical_rect);

  private:
    struct impl_t;
    std::unique_ptr<impl_t> impl_;
  };
}  // namespace platf
