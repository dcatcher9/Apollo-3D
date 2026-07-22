/**
 * @file src/sbs_perf.cpp
 * @brief Implementation of the SBS per-stage performance collector (see sbs_perf.h).
 */
#include "sbs_perf.h"

// standard includes
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// local includes
#include "src/logging.h"

namespace sbs_perf {

  using namespace std::literals;

  namespace {

    // How many recent samples to keep per stage. Live summaries aggregate all active SBS streams;
    // the offline harness has one stream and explicitly resets before each run.
    constexpr size_t kWindow = 512;
    // Emit a summary line every this many aggregate convert() ticks.
    constexpr int kSummaryInterval = 300;

    struct stage_stat {
      std::vector<float> ring;  ///< Ring buffer of the last kWindow samples (ms).
      size_t next = 0;          ///< Next write index into `ring`.
      bool full = false;        ///< Has `ring` wrapped at least once.
      uint64_t total = 0;       ///< Lifetime sample count.

      void push(float ms) {
        if (ring.size() < kWindow) {
          ring.push_back(ms);
        } else {
          ring[next] = ms;
          next = (next + 1) % kWindow;
          full = true;
        }
        ++total;
      }

      size_t window_count() const {
        return full ? kWindow : ring.size();
      }
    };

    struct summary {
      double p50 = 0, p95 = 0, max = 0, mean = 0;
      size_t n = 0;
    };

    summary summarize(const stage_stat &s) {
      summary out;
      out.n = s.window_count();
      if (out.n == 0) {
        return out;
      }
      std::vector<float> v(s.ring.begin(), s.ring.begin() + out.n);
      std::sort(v.begin(), v.end());
      double sum = 0;
      for (float x : v) {
        sum += x;
      }
      out.mean = sum / (double) out.n;
      out.max = v.back();
      auto pick = [&](double q) {
        size_t idx = (size_t) (q * (double) (out.n - 1) + 0.5);
        return (double) v[std::min(idx, out.n - 1)];
      };
      out.p50 = pick(0.50);
      out.p95 = pick(0.95);
      return out;
    }

    std::atomic<bool> g_enabled {false};
    std::atomic<std::uint64_t> g_generation {1};
    std::mutex g_mutex;
    // Insertion-ordered would be nicer for the log, but std::map keyed by the literal keeps it
    // simple and the stage set is tiny; the log prints a stable (alphabetical) order.
    std::map<std::string, stage_stat> g_stages;
    int g_frame = 0;

    // Assumes g_mutex is held.
    bool write_json_locked(const std::string &path) {
      std::ofstream f(path, std::ios::trunc);
      if (!f) {
        return false;
      }
      f << "{\n  \"stages\": {\n";
      bool first = true;
      for (auto &[name, st] : g_stages) {
        auto s = summarize(st);
        if (!first) {
          f << ",\n";
        }
        first = false;
        f << "    \"" << name << "\": {"
          << "\"p50_ms\": " << s.p50 << ", "
          << "\"p95_ms\": " << s.p95 << ", "
          << "\"max_ms\": " << s.max << ", "
          << "\"mean_ms\": " << s.mean << ", "
          << "\"n\": " << s.n << ", "
          << "\"total\": " << st.total << "}";
      }
      f << "\n  }\n}\n";
      return true;
    }

  }  // namespace

  void set_enabled(bool on) {
    bool was = g_enabled.exchange(on, std::memory_order_relaxed);
    if (on && !was) {
      BOOST_LOG(info) << "[sbs-perf] process-wide per-stage timing enabled (summary every "sv
                      << kSummaryInterval << " frames; JSON snapshots are explicit)"sv;
    }
  }

  bool enabled() {
    return g_enabled.load(std::memory_order_relaxed);
  }

  void add_sample_ms(const char *stage, double ms) {
    if (!g_enabled.load(std::memory_order_relaxed) || !stage) {
      return;
    }
    std::lock_guard<std::mutex> lk(g_mutex);
    g_stages[stage].push((float) ms);
  }

  std::uint64_t generation() {
    return g_generation.load(std::memory_order_acquire);
  }

  void add_sample_ms_if_current(const char *stage, double ms, std::uint64_t expected_generation) {
    if (!g_enabled.load(std::memory_order_relaxed) || !stage ||
        expected_generation != g_generation.load(std::memory_order_acquire)) {
      return;
    }
    std::lock_guard<std::mutex> lk(g_mutex);
    if (expected_generation == g_generation.load(std::memory_order_relaxed)) {
      g_stages[stage].push((float) ms);
    }
  }

  void tick() {
    if (!g_enabled.load(std::memory_order_relaxed)) {
      return;
    }
    std::lock_guard<std::mutex> lk(g_mutex);
    if (++g_frame < kSummaryInterval) {
      return;
    }
    g_frame = 0;
    if (g_stages.empty()) {
      return;
    }

    std::string line = "[sbs-perf aggregate]";
    for (auto &[name, st] : g_stages) {
      auto s = summarize(st);
      if (s.n == 0) {
        continue;
      }
      char buf[160];
      snprintf(buf, sizeof(buf), " %s p50=%.2f p95=%.2f max=%.2f (n=%zu) |",
        name.c_str(), s.p50, s.p95, s.max, s.n);
      line += buf;
    }
    if (!line.empty() && line.back() == '|') {
      line.pop_back();
    }
    BOOST_LOG(info) << line;
  }

  void reset() {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_stages.clear();
    g_frame = 0;
    g_generation.fetch_add(1, std::memory_order_release);
  }

  bool dump_json(const std::string &path) {
    std::lock_guard<std::mutex> lk(g_mutex);
    return write_json_locked(path);
  }

}  // namespace sbs_perf
