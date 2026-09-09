/** Bounded, read-only UI Automation qualification of native shell pointer targets. */
#pragma once

#include <memory>
#include <optional>
#include <windows.h>

namespace desktop_launcher {
  /** 0 verifies real private-desktop UIA; 77 explicitly reports unsupported, other codes fail. */
  int native_shell_item_self_test();

  /**
   * Queries run on one MTA worker, never a hook or router thread. A hung provider
   * cannot admit a gesture or block shutdown; its worker dies with this helper.
   */
  class native_shell_item_query_t {
  public:
    native_shell_item_query_t();
    ~native_shell_item_query_t();
    native_shell_item_query_t(const native_shell_item_query_t &) = delete;
    native_shell_item_query_t &operator=(const native_shell_item_query_t &) = delete;

    /** Prefetch an exact shell root/point during hover or button-down. Never blocks on UIA. */
    void request(HWND source_root, POINT point);
    /** A completed answer is valid for 500 ms and only the same live root/process/point. */
    [[nodiscard]] std::optional<bool> result(HWND source_root, POINT point) const;
    void clear();

  private:
    struct state_t;
    static void worker(std::shared_ptr<state_t> state);
    std::shared_ptr<state_t> state_;
  };
}  // namespace desktop_launcher
