// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// An INDEPENDENT happens-before oracle over a fixture's own queue calls
// (README, Session-level tests, unordered_read): the native provider fixture
// and the session timelines log every ExecuteCommandLists, Signal and Wait
// they make on the game's queues and fences, the CPU Signals they make, and
// each Present. The oracle never reads the add-on's model or report: it
// computes vector clocks from that script alone, as D3D12 defines the order.
//
// A queue's call follows every earlier call on its queue. A Wait(G, w) on a
// queue orders its later calls after a signal only when every signal of G
// that reaches w anywhere in the script (CPU ones included) could satisfy it
// no earlier: the wait follows the pointwise earliest of those signals'
// clocks, and nothing when G's initial value already reaches w, a CPU signal
// reaches it, or no signal does. An Execute's copy-out (the add-on's list in
// that call's post-call, on the same thread) precedes the queue's next
// logged call, so a reader is ordered after the copy-out once its clock
// passes the Execute.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace queue_order_script {
  enum class kind : std::uint8_t {
    execute,
    signal,
    wait,
    cpu_signal,
    present
  };

  struct event {
    kind what {};
    const void *queue {}, *fence {};
    std::uint64_t value {};
    // execute: the add-on's queue-order position for the call (its copy-out
    // list's when it made one), as a frame record names a producer.
    std::uint64_t order_queue {}, order_ticket {};
  };

  // A Present whose record says its fresh copy was read because the model
  // ordered it (depth_read_order=ordered), and the producer it named.
  struct ordered_read {
    std::size_t present {};
    std::uint64_t present_index {}, order_queue {}, order_ticket {};
    // The producer Execute's index when the script knows it; else it is the
    // Execute the add-on positioned at (order_queue, order_ticket).
    std::size_t made {SIZE_MAX};
  };

  struct log {
    std::vector<event> events;
    std::map<const void *, std::uint64_t> initial;  // A fence's value at creation (0 when absent).
    std::vector<ordered_read> reads;
    bool enabled {};

    void execute(const void *queue, std::uint64_t order_queue, std::uint64_t order_ticket) {
      if (enabled) {
        events.push_back({kind::execute, queue, nullptr, 0, order_queue, order_ticket});
      }
    }

    void signal(const void *queue, const void *fence, std::uint64_t value) {
      if (enabled) {
        events.push_back({kind::signal, queue, fence, value});
      }
    }

    void wait(const void *queue, const void *fence, std::uint64_t value) {
      if (enabled) {
        events.push_back({kind::wait, queue, fence, value});
      }
    }

    void cpu_signal(const void *fence, std::uint64_t value) {
      if (enabled) {
        events.push_back({kind::cpu_signal, nullptr, fence, value});
      }
    }

    // A Present on queue; read: its record read its copy ordered after the
    // producer position (order_queue, order_ticket).
    void present(const void *queue, std::uint64_t present_index, bool read = false, std::uint64_t order_queue = 0, std::uint64_t order_ticket = 0) {
      if (!enabled) {
        return;
      }
      events.push_back({kind::present, queue});
      if (read) {
        reads.push_back({events.size() - 1, present_index, order_queue, order_ticket});
      }
    }

    // The latest event of what on queue, or events.size().
    std::size_t last(kind what, const void *queue) const {
      for (auto i = events.size(); i-- != 0;) {
        if (events[i].what == what && events[i].queue == queue) {
          return i;
        }
      }
      return events.size();
    }

    // A read the script knows was in flight at the Present at index present
    // (its copy's work had not run yet): it must be ordered after the
    // Execute at index made as any read the record claims ordered.
    void in_flight(std::size_t present, std::uint64_t present_index, std::size_t made) {
      if (present < events.size() && made < events.size()) {
        reads.push_back({present, present_index, events[made].order_queue, events[made].order_ticket, made});
      }
    }

    void clear() {
      events.clear();
      reads.clear();
    }

    // Each queue's clock: for every queue, one past the index of the latest
    // of its calls it is ordered after (0: none).
    using clock = std::map<const void *, std::size_t>;

    // The clock of each event's queue just after it, for the events at
    // (sorted), computed in one pass over the script.
    std::map<std::size_t, clock> clocks(const std::vector<std::size_t> &at) const {
      // Per fence, its signals by value, with each suffix's earliest signal
      // per queue and whether a CPU signal is among them: a wait for w is
      // satisfied no earlier than the earliest signal reaching w of each
      // queue (a queue's later signal follows its earlier one).
      struct reaching {
        std::vector<std::pair<std::uint64_t, std::size_t>> by_value;
        std::vector<std::map<const void *, std::size_t>> earliest;
        std::vector<bool> cpu;
      };

      std::map<const void *, reaching> signals;
      for (std::size_t i = 0; i != events.size(); ++i) {
        if (events[i].what == kind::signal || events[i].what == kind::cpu_signal) {
          signals[events[i].fence].by_value.emplace_back(events[i].value, i);
        }
      }
      for (auto &[fence, value] : signals) {
        std::sort(value.by_value.begin(), value.by_value.end());
        const auto count = value.by_value.size();
        value.earliest.resize(count + 1);
        value.cpu.assign(count + 1, false);
        for (std::size_t k = count; k-- != 0;) {
          const auto j = value.by_value[k].second;
          value.earliest[k] = value.earliest[k + 1];
          value.cpu[k] = value.cpu[k + 1] || events[j].what == kind::cpu_signal;
          if (events[j].what == kind::signal) {
            auto &first = value.earliest[k][events[j].queue];
            first = first ? std::min(first, j + 1) : j + 1;
          }
        }
      }
      std::map<const void *, clock> current;
      std::map<std::size_t, clock> signal_clock, out;
      auto next = at.begin();
      for (std::size_t i = 0; i != events.size() && next != at.end(); ++i) {
        const auto &e = events[i];
        if (e.what != kind::cpu_signal) {
          auto &own = current[e.queue];
          own[e.queue] = i + 1;
          if (e.what == kind::signal) {
            signal_clock[i] = own;
          } else if (e.what == kind::wait) {
            const auto found = initial.find(e.fence);
            const bool initially = found != initial.end() && found->second >= e.value;
            const auto fence = signals.find(e.fence);
            if (!initially && fence != signals.end()) {
              const auto &value = fence->second;
              const auto k = static_cast<std::size_t>(std::lower_bound(value.by_value.begin(), value.by_value.end(), std::make_pair(e.value, std::size_t {})) - value.by_value.begin());
              if (k != value.by_value.size() && !value.cpu[k] && !value.earliest[k].empty()) {
                clock meet;
                bool first = true;
                for (const auto &[queue, one_past] : value.earliest[k]) {
                  const auto j = one_past - 1;
                  // A signal not yet made: no earlier than its queue's clock
                  // now and its own call (an underestimate, so never more).
                  clock reached = j < i ? signal_clock[j] : current[queue];
                  reached[queue] = j + 1;
                  if (first) {
                    meet = reached;
                    first = false;
                    continue;
                  }
                  clock both;
                  for (const auto &[q, position] : meet) {
                    const auto other = reached.find(q);
                    if (other != reached.end()) {
                      both[q] = std::min(position, other->second);
                    }
                  }
                  meet = both;
                }
                for (const auto &[q, position] : meet) {
                  own[q] = std::max(own[q], position);
                }
              }
            }
          }
        }
        if (*next == i) {
          out[i] = e.queue ? current[e.queue] : clock {};
          ++next;
        }
      }
      return out;
    }

    // The Execute the add-on positioned at (order_queue, order_ticket), or
    // events.size() when the script made no such call.
    std::size_t producer(std::uint64_t order_queue, std::uint64_t order_ticket) const {
      for (std::size_t i = 0; i != events.size(); ++i) {
        if (events[i].what == kind::execute && events[i].order_queue == order_queue && events[i].order_ticket == order_ticket) {
          return i;
        }
      }
      return events.size();
    }

    // unordered_read: every ordered read the add-on recorded must name an
    // Execute of the script that the oracle orders the Present after. The
    // failures, described (empty when every read is ordered). Checked reads
    // are dropped; the script stays for later reads.
    std::vector<std::string> unordered_reads() {
      std::vector<std::string> out;
      if (reads.empty()) {
        return out;
      }
      std::vector<std::size_t> at;
      for (const auto &read : reads) {
        at.push_back(read.present);
      }
      std::sort(at.begin(), at.end());
      at.erase(std::unique(at.begin(), at.end()), at.end());
      const auto all = clocks(at);
      for (const auto &read : reads) {
        const auto made = read.made != SIZE_MAX ? read.made : producer(read.order_queue, read.order_ticket);
        if (made == events.size()) {
          out.push_back("Present " + std::to_string(read.present_index) + " read in flight after a position (" + std::to_string(read.order_queue) + "," + std::to_string(read.order_ticket) + ") no logged Execute of the script took");
          continue;
        }
        bool ordered = made < read.present;
        if (ordered && events[made].queue != events[read.present].queue) {
          const auto &reader = all.at(read.present);
          const auto found = reader.find(events[made].queue);
          ordered = found != reader.end() && found->second > made + 1;
        }
        if (!ordered) {
          out.push_back("Present " + std::to_string(read.present_index) + " read in flight after Execute #" + std::to_string(made) + " that the script's own fences do not order it after");
        }
      }
      reads.clear();
      return out;
    }
  };
}  // namespace queue_order_script
