// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <windows.h>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <utility>
#include <reshade.hpp>
#include "addon_lifetime.h"

// ReShade writes every log line through to disk (FILE_FLAG_WRITE_THROUGH). On a
// game streaming assets from the same disk one line took up to ~50 ms, and
// periodic diagnostics were written from the game's Present. Lines are queued
// in order and written by one thread-pool drain; the caller only formats and
// enqueues. The add-on is pinned, so a pending drain never outlives its code,
// and process exit ends pool threads before any DLL detach.
namespace sunshine_log {
  namespace detail {
    struct queue_t {
      std::mutex mutex;
      std::deque<std::pair<reshade::log::level, std::string>> lines;
      bool draining = false;
      unsigned dropped = 0;
    };
    // Deliberately leaked: a late drain must not touch a destroyed mutex.
    inline queue_t &queue() {
      static queue_t *value = new queue_t;
      return *value;
    }
    inline constexpr std::size_t capacity = 512;
    inline void drain() {
      auto &q = queue();
      for (;;) {
        std::pair<reshade::log::level, std::string> line;
        unsigned dropped = 0;
        {
          std::lock_guard<std::mutex> lock(q.mutex);
          if (q.lines.empty()) {
            q.draining = false;
            return;
          }
          line = std::move(q.lines.front());
          q.lines.pop_front();
          dropped = std::exchange(q.dropped, 0u);
        }
        if (sunshine_addon_lifetime::stopping()) continue;
        if (dropped) {
          char text[96];
          std::snprintf(text, sizeof(text), "Sunshine log: %u lines dropped while the log writer was behind", dropped);
          reshade::log::message(reshade::log::level::warning, text);
        }
        reshade::log::message(line.first, line.second.c_str());
      }
    }
    inline void CALLBACK drain_callback(PTP_CALLBACK_INSTANCE, void *) { drain(); }
  }

  inline void message(reshade::log::level level, const char *text) {
#ifdef SUNSHINE_SYNCHRONOUS_LOG
    // Unit tests that intercept ReShade's log check emission synchronously.
    reshade::log::message(level, text);
#else
    auto &q = detail::queue();
    bool submit = false;
    {
      std::lock_guard<std::mutex> lock(q.mutex);
      if (q.lines.size() >= detail::capacity) {
        ++q.dropped;
        return;
      }
      q.lines.emplace_back(level, text ? text : "");
      submit = !std::exchange(q.draining, true);
    }
    // Without a pool thread, write on the caller rather than lose the line.
    if (submit && !TrySubmitThreadpoolCallback(detail::drain_callback, nullptr, nullptr)) detail::drain();
#endif
  }
}
