// SPDX-License-Identifier: GPL-3.0-only
#pragma once

// The per-Present decision trace of the runtime fixtures (T1 of the add-on
// structure refactor): one normalized line per Present a fixture drives and
// observes, kept in a buffer reserved up front and written to its file once,
// when the fixture or session timeline ends. No Present allocates or writes a
// file, so tracing adds no file latency between the Presents a fixture
// drives (the 250 ms source age, fg_silent_ms). Fixtures write it, never the
// add-on; a trace never fails a run or changes what a fixture judges.
//
// Lines are space-separated key=value tokens. Normalized: no tick, time,
// handle, pointer, loss counter or revision is written, and ids (capture
// ids, source ids, epochs, identities, sequences, resources, effect
// runtimes) are written as their rank in first-appearance order within their
// own space, 0 staying 0, so that two runs of one fixture can compare equal.
// The test add-on line is test_game3d_native_observation.h's
// (native_present_trace); the production add-on's selection fixtures write
// their own observations. "teardown <unload> <events>" lines are
// addon_unload's teardown sequences (T7), and a "!" line says the buffer was
// full.
#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>

namespace game3d_present_trace {
  // The id spaces ranks are kept in, each its own first-appearance order.
  // runtimes: effect runtime addresses (the T8 binding events).
  enum space : unsigned { capture, source, epoch, identity, sequence, resource, runtimes, space_count };

  class writer {
  public:
    // The buffer of one trace: about 90 000 lines of the test add-on's.
    static constexpr std::size_t capacity = std::size_t(64) << 20;
    // An id seen after its space's table filled.
    static constexpr std::uint32_t unranked = 0xffffffffu;

    writer() = default;
    writer(const writer &) = delete;
    writer &operator=(const writer &) = delete;
    ~writer() { close(); }

    bool active() const { return active_; }
    // Counts open calls: a caller's per-trace state belongs to one trace.
    std::uint64_t generation() const { return generation_; }

    // Starts a trace to path (written by close). Reserves the buffer once and
    // restarts every rank space and the Present base.
    void open(const std::filesystem::path &path, const char *kind) {
      close();
      if (text_.capacity() < capacity) text_.reserve(capacity);
      text_.clear();
      for (auto &table : spaces_) table.clear();
      path_ = path;
      dropped_ = 0;
      based_ = sequence_based_ = false;
      present_base_ = sequence_base_ = 0;
      ++generation_;
      active_ = true;
      begin();
      add("# game3d-present-trace 1 %s", kind);
      end();
    }

    // Ends the trace and writes it: the one file write of a fixture or
    // timeline. A trace that cannot be written is reported, not judged.
    void close() {
      if (!active_) return;
      active_ = false;
      if (dropped_) {
        char line[96];
        const int length = std::snprintf(line, sizeof(line), "! overflow dropped=%llu\n", static_cast<unsigned long long>(dropped_));
        text_.append(line, std::size_t(length));
      }
      std::FILE *file = _wfopen(path_.c_str(), L"wb");
      const bool written = file && std::fwrite(text_.data(), 1, text_.size(), file) == text_.size();
      if (file) std::fclose(file);
      if (!written) std::fprintf(stderr, "T1 trace: could not write %ls\n", path_.c_str());
      text_.clear();
    }

    // The rank of id in its space: 0 for 0, else its first-appearance order.
    std::uint32_t rank(space where, std::uint64_t id) { return spaces_[where].rank(id); }

    // A Present index or render sequence relative to the trace's first line
    // (set by the first call of each). Signed: a fixture's ordinals never step
    // back, but a reload's could.
    long long present(std::uint64_t value) {
      if (!based_) { based_ = true; present_base_ = value; }
      return static_cast<long long>(value - present_base_);
    }
    long long sequence(std::uint64_t value) {
      if (!sequence_based_) { sequence_based_ = true; sequence_base_ = value; }
      return static_cast<long long>(value - sequence_base_);
    }

    // One line, built in a fixed buffer: begin, add tokens, end. A line that
    // does not fit the trace's buffer is counted as dropped instead.
    void begin() { used_ = 0; }
    [[gnu::format(printf, 2, 3)]] void add(const char *format, ...) {
      if (used_ + 1 >= line_.size()) return;
      std::va_list arguments;
      va_start(arguments, format);
      const int length = std::vsnprintf(line_.data() + used_, line_.size() - used_, format, arguments);
      va_end(arguments);
      if (length > 0) used_ = std::min(line_.size() - 1, used_ + std::size_t(length));
    }
    // A static token (a reason or a name) as one value: anything that would
    // split a token becomes '_'.
    void token(const char *value) {
      if (!value) value = "null";
      for (; *value && used_ + 1 < line_.size(); ++value) {
        const char c = *value;
        line_[used_++] = (c == ' ' || c == '=' || c == ',' || c == '\t' || c == '\r' || c == '\n') ? '_' : c;
      }
      line_[used_] = '\0';
    }
    // Text kept as written but for white space, which becomes '_' (a list's
    // commas and parentheses stay).
    void text(const char *value) {
      if (!value) value = "null";
      for (; *value && used_ + 1 < line_.size(); ++value) {
        const char c = *value;
        line_[used_++] = (c == ' ' || c == '\t' || c == '\r' || c == '\n') ? '_' : c;
      }
      line_[used_] = '\0';
    }
    void end() {
      if (!active_) return;
      if (text_.size() + used_ + 1 > capacity) { ++dropped_; return; }
      text_.append(line_.data(), used_);
      text_.push_back('\n');
    }

    // T7 (refactor plan 3.4): addon_unload's teardown sequence of one unload
    // (SunshineGame3DTestTeardownOrder, as the fixture names its events), one
    // "teardown <unload> <events>" line per unload in the order the unloads
    // happen. The unload's label is one token (its spaces become '_').
    void teardown(const char *unload, const char *events) {
      if (!active_) return;
      begin();
      add("teardown ");
      token(unload);
      add(" ");
      text(events);
      end();
    }

  private:
    // Open addressing in a fixed table: no Present allocates.
    class rank_table {
    public:
      void clear() { keys_.fill(0); count_ = 0; }
      std::uint32_t rank(std::uint64_t id) {
        if (!id) return 0;
        std::size_t slot = mix(id) & (size - 1);
        for (std::size_t probe = 0; probe != size; ++probe, slot = (slot + 1) & (size - 1)) {
          if (keys_[slot] == id) return ranks_[slot];
          if (!keys_[slot]) {
            if (count_ >= size / 2) return unranked;
            keys_[slot] = id;
            return ranks_[slot] = ++count_;
          }
        }
        return unranked;
      }

    private:
      static constexpr std::size_t size = 8192;
      static std::uint64_t mix(std::uint64_t value) {
        value ^= value >> 33;
        value *= 0xff51afd7ed558ccdULL;
        return value ^ (value >> 33);
      }
      std::array<std::uint64_t, size> keys_ {};
      std::array<std::uint32_t, size> ranks_ {};
      std::uint32_t count_ {};
    };

    std::array<rank_table, space_count> spaces_ {};
    std::array<char, 4096> line_ {};
    std::size_t used_ {};
    std::string text_;
    std::filesystem::path path_;
    std::uint64_t dropped_ {}, present_base_ {}, sequence_base_ {}, generation_ {};
    bool active_ {}, based_ {}, sequence_based_ {};
  };

  // The fixture's trace: one open file at a time.
  inline writer out;

  // A label as a file-name part.
  inline std::string file_part(const std::string &label) {
    std::string part = label;
    for (auto &c : part)
      if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.')) c = '_';
    return part;
  }
}
