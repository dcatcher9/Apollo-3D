// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <windows.h>

// Which presented window a Game 3D export belongs to, shared by the ReShade add-on and the Windows
// receiver. A swapchain presented to the foreground top-level window owns the foreground as before.
// One presented to a child window of it (at any depth) owns the foreground only while that child is
// what the user sees in the root's client area: every window from the child up to the root
// belongs to the producer and has exactly the expected client area in screen coordinates (so no
// ancestor clips it), the child and its ancestors are visible, and no other visible window with the
// same parent as any of them overlaps that area. Two child swapchains that both cover the root
// overlap each other, so neither owns it. The published metadata names the swapchain's own window.
namespace reshade_bridge::window_owner {
  struct rect_t {
    std::int32_t left = 0, top = 0, right = 0, bottom = 0;
  };

  [[nodiscard]] constexpr bool same(const rect_t &a, const rect_t &b) {
    return a.left == b.left && a.top == b.top && a.right == b.right && a.bottom == b.bottom;
  }

  [[nodiscard]] constexpr bool intersects(const rect_t &a, const rect_t &b) {
    return a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
  }

  // One window of the chain from the swapchain's window up to its root, as USER32 reports it.
  struct link_t {
    std::uintptr_t window = 0;
    std::uint32_t process_id = 0;
    // The client area in screen coordinates of the querying thread's DPI awareness.
    rect_t client;
    // Every query of this window succeeded.
    bool queried = false;
    // Another visible window with the same parent intersects `client` (never set for the root,
    // whose siblings are other top-level windows).
    bool overlapped = false;
  };

  // The swapchain's window, up to six intermediate parents, and the root.
  inline constexpr std::size_t max_links = 8;
  // Siblings examined at one level; more leave the chain incomplete.
  inline constexpr std::size_t max_siblings = 256;

  struct chain_t {
    std::array<link_t, max_links> links {};
    std::size_t count = 0;
    // The links end at GetAncestor(window, GA_ROOT) and every sibling census finished.
    bool complete = false;
    // IsWindowVisible(window): the window and all of its ancestors are visible.
    bool visible = false;
  };

  enum class verdict {
    owner,
    not_child,  ///< A top-level window, or no window at all: the caller's own rule applies.
    incomplete,  ///< A query failed, the chain is deeper than max_links, or a census overflowed.
    not_foreground,  ///< The chain's root is not the foreground root.
    other_process,  ///< A window of the chain belongs to another process.
    not_visible,
    partial,  ///< A window of the chain does not have exactly the expected client area.
    overlapped,  ///< Another visible window overlaps the chain below the root.
  };

  [[nodiscard]] constexpr const char *verdict_name(verdict value) {
    switch (value) {
      case verdict::owner:
        return "owner";
      case verdict::not_child:
        return "not_child";
      case verdict::incomplete:
        return "incomplete";
      case verdict::not_foreground:
        return "not_foreground";
      case verdict::other_process:
        return "other_process";
      case verdict::not_visible:
        return "not_visible";
      case verdict::partial:
        return "partial";
      case verdict::overlapped:
        return "overlapped";
    }
    return "unknown";
  }

  // The pure rule over one chain. `expected` is the client area the export must cover exactly:
  // the captured source rectangle on the host, the root's own client area in the add-on (the host
  // separately requires the root's client area to equal the captured source).
  [[nodiscard]] constexpr verdict classify(const chain_t &chain, std::uintptr_t foreground_root, std::uint32_t producer_pid, const rect_t &expected) {
    if (chain.count == 0 || (chain.count == 1 && chain.complete)) {
      return verdict::not_child;
    }
    if (chain.count > max_links || !chain.complete) {
      return verdict::incomplete;
    }
    for (std::size_t i = 0; i < chain.count; ++i) {
      if (!chain.links[i].queried || !chain.links[i].window) {
        return verdict::incomplete;
      }
    }
    if (chain.links[chain.count - 1].window != foreground_root) {
      return verdict::not_foreground;
    }
    for (std::size_t i = 0; i < chain.count; ++i) {
      if (chain.links[i].process_id != producer_pid) {
        return verdict::other_process;
      }
    }
    if (!chain.visible) {
      return verdict::not_visible;
    }
    if (expected.right <= expected.left || expected.bottom <= expected.top) {
      return verdict::partial;
    }
    for (std::size_t i = 0; i < chain.count; ++i) {
      if (!same(chain.links[i].client, expected)) {
        return verdict::partial;
      }
    }
    for (std::size_t i = 0; i + 1 < chain.count; ++i) {
      if (chain.links[i].overlapped) {
        return verdict::overlapped;
      }
    }
    return verdict::owner;
  }

  namespace detail {
    inline bool client_rect(HWND window, rect_t &client) {
      RECT rect {};
      if (GetClientRect(window, &rect) == FALSE) {
        return false;
      }
      SetLastError(ERROR_SUCCESS);
      if (MapWindowPoints(window, nullptr, reinterpret_cast<POINT *>(&rect), 2) == 0 && GetLastError() != ERROR_SUCCESS) {
        return false;
      }
      client = {rect.left, rect.top, rect.right, rect.bottom};
      return true;
    }

    // Whether another visible window at `window`'s level intersects `client`. Clears `complete`
    // when the level has more than max_siblings windows or a query fails.
    inline bool sibling_overlaps(HWND window, const rect_t &client, bool &complete) {
      std::size_t examined = 0;
      for (HWND sibling = GetWindow(window, GW_HWNDFIRST); sibling; sibling = GetWindow(sibling, GW_HWNDNEXT)) {
        if (++examined > max_siblings) {
          complete = false;
          return false;
        }
        if (sibling == window || IsWindowVisible(sibling) == FALSE) {
          continue;
        }
        RECT bounds {};
        if (GetWindowRect(sibling, &bounds) == FALSE) {
          complete = false;
          return false;
        }
        if (intersects({bounds.left, bounds.top, bounds.right, bounds.bottom}, client)) {
          return true;
        }
      }
      return false;
    }
  }  // namespace detail

  // Queries the chain from `window` up to its root. None of these calls sends a window message,
  // so a render thread may query a window that another, busy thread owns. Coordinates follow the
  // calling thread's DPI awareness: the host queries under per-monitor awareness (physical
  // pixels); the add-on compares a child with its own root in one coordinate space.
  [[nodiscard]] inline chain_t query(HWND window) {
    chain_t chain;
    const HWND root = window && IsWindow(window) != FALSE ? GetAncestor(window, GA_ROOT) : nullptr;
    if (!root) {
      return chain;
    }
    chain.visible = IsWindowVisible(window) != FALSE;
    bool complete = true;
    for (HWND current = window; current && chain.count < max_links;) {
      auto &link = chain.links[chain.count++];
      link.window = reinterpret_cast<std::uintptr_t>(current);
      DWORD process_id = 0;
      link.queried = IsWindow(current) != FALSE && GetWindowThreadProcessId(current, &process_id) != 0 && detail::client_rect(current, link.client);
      link.process_id = process_id;
      if (current == root) {
        chain.complete = complete;
        return chain;
      }
      link.overlapped = detail::sibling_overlaps(current, link.client, complete);
      current = GetAncestor(current, GA_PARENT);
    }
    return chain;
  }
}  // namespace reshade_bridge::window_owner
