// SPDX-License-Identifier: GPL-3.0-only
// The queue-order model (queue_order.h): one cell per fail-closed rule, the
// positive chains, and generated scripts judged by an INDEPENDENT brute-force
// oracle that explores every GPU interleaving of the script (README,
// Queue order). No device, no hooks.
#include "queue_order.h"
#include "test_queue_order_script.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
  namespace qo = sunshine_queue_order;
  using model = qo::model<>;

  void require(bool value, const std::string &message) {
    if (!value) throw std::runtime_error(message);
  }

  // A tiny host: tickets per queue, ids from the model, as the observer
  // keeps them; every call is quiet unless a cell says otherwise.
  struct host {
    model m;
    std::map<int, std::uint64_t> queue_ids, fence_ids;
    std::map<int, std::uint64_t> entered;
    std::map<int, std::uint64_t> fence_values;  // What GetCompletedValue reads now.
    std::uint64_t now = 1000;
    std::uint64_t device = 0xd1;

    std::uint64_t queue(int q, std::uint64_t on = 0) {
      auto &id = queue_ids[q];
      if (!id || !m.known_queue(id)) id = m.add_queue(on ? on : device, 100 + q);
      return id;
    }
    std::uint64_t fence(int f, std::uint64_t on = 0) {
      auto &id = fence_ids[f];
      if (!id || !m.known_fence(id)) id = m.add_fence(on ? on : device, fence_values[f], m.memory(id));
      return id;
    }
    qo::position execute(int q, bool quiet = true) {
      const auto id = queue(q);
      return {id, ++entered[q], quiet};
    }
    void signal(int q, int f, std::uint64_t value) {
      const auto id = queue(q);
      const auto ticket = ++entered[q];
      m.signal(id, fence(f), value, ticket, fence_values[f], now);
    }
    void signal_unrecorded(int f, std::uint64_t value) { m.signal(0, fence(f), value, 0, fence_values[f], now); }
    void wait(int q, int f, std::uint64_t value, bool quiet = true) {
      const auto id = queue(q);
      const auto ticket = ++entered[q];
      m.wait(id, fence(f), value, ticket, quiet, fence_values[f], now);
    }
    void cpu(int f) { m.cpu_signal(fence(f), now); }
    // The GPU ran: the fence reads value, seen by a hooked call.
    void complete(int f, std::uint64_t value) {
      fence_values[f] = value;
      m.observed_completed(fence(f), value, now);
    }
    qo::verdict ordered(int reader, qo::position p) { return m.ordered(queue(reader), p); }
  };

  // Settled fence 0, signalled by queue 0: a first signal completed.
  void settle(host &h, int q = 0, int f = 0) {
    h.signal(q, f, 1);
    h.complete(f, 1);
  }

  void cells() {
    // A queue that only signals (an async producer) is in use: a full table
    // of stale records evicts one of those, never it.
    {
      host h;
      settle(h);
      for (int q = 1; q != 32; ++q) h.queue(q);
      h.signal(0, 0, 2);
      h.queue(40);
      require(h.m.known_queue(h.queue_ids[0]) && !h.m.known_queue(h.queue_ids[1]),
        "a signalling queue was the least recently used record");
    }
    // Positive: the reader waits after the producer's signal.
    {
      host h;
      settle(h);
      const auto p = h.execute(0);
      h.signal(0, 0, 2);
      h.wait(1, 0, 2);
      const auto v = h.ordered(1, p);
      require(v.ordered && v.fences[0] == h.fence_ids[0], "after-signal edge");
    }
    // Positive: wait logged before the signal resolves when the signal is logged.
    {
      host h;
      settle(h);
      const auto p = h.execute(0);
      h.wait(1, 0, 2);
      require(!h.ordered(1, p).ordered && h.ordered(1, p).why == qo::refusal::unresolved, "pending wait refuses");
      h.signal(0, 0, 2);
      require(h.ordered(1, p).ordered, "wait-before-signal resolves at the signal");
    }
    // A wait resolves to the FIRST signal reaching its value, not a later one.
    {
      host h;
      settle(h);
      h.signal(0, 0, 2);
      const auto p = h.execute(0);
      h.signal(0, 0, 3);
      h.wait(1, 0, 2);
      require(!h.ordered(1, p).ordered, "a wait satisfied before the producer is no edge");
    }
    // The producer's own later signals do not order a reader with no wait.
    {
      host h;
      settle(h);
      const auto p = h.execute(0);
      h.signal(0, 0, 2);
      require(!h.ordered(1, p).ordered, "no wait, no edge");
    }
    // Same queue: queue order.
    {
      host h;
      const auto p = h.execute(0, false);
      require(h.ordered(0, p).ordered, "same queue is ordered");
    }
    // Overlap: a producer call that was not quiet is never the earlier side.
    {
      host h;
      settle(h);
      const auto p = h.execute(0, false);
      h.signal(0, 0, 2);
      h.wait(1, 0, 2);
      require(h.ordered(1, p).why == qo::refusal::overlap, "overlap refuses");
    }
    // Two hops: render 0 -> compute 1 -> present 2.
    {
      host h;
      settle(h, 0, 0);
      settle(h, 1, 1);
      const auto p = h.execute(0);
      h.signal(0, 0, 2);
      h.wait(1, 0, 2);
      h.execute(1);
      h.signal(1, 1, 2);
      h.wait(2, 1, 2);
      require(h.ordered(2, p).ordered, "two-hop chain");
      // A hop wait that was not quiet does not carry the chain.
      host g;
      settle(g, 0, 0);
      settle(g, 1, 1);
      const auto q = g.execute(0);
      g.signal(0, 0, 2);
      g.wait(1, 0, 2, false);
      g.signal(1, 1, 2);
      g.wait(2, 1, 2);
      require(!g.ordered(2, q).ordered, "a non-quiet hop wait carries no edge");
      // A hop wait after the hop's signal does not carry it either.
      host k;
      settle(k, 0, 0);
      settle(k, 1, 1);
      const auto r = k.execute(0);
      k.signal(0, 0, 2);
      k.signal(1, 1, 2);
      k.wait(1, 0, 2);
      k.wait(2, 1, 2);
      require(!k.ordered(2, r).ordered, "a hop wait after the hop's signal carries no edge");
    }
    // Three hops.
    {
      host h;
      for (int i = 0; i != 3; ++i) settle(h, i, i);
      const auto p = h.execute(0);
      h.signal(0, 0, 2);
      h.wait(1, 0, 2);
      h.signal(1, 1, 2);
      h.wait(2, 1, 2);
      h.signal(2, 2, 2);
      h.wait(3, 2, 2);
      require(h.ordered(3, p).ordered, "three-hop chain");
    }
    // Baseline: a wait at or below the value at first sight may already pass.
    {
      host g;
      g.fence_values[0] = 5;
      g.signal(0, 0, 6);
      g.complete(0, 6);
      const auto p = g.execute(0);
      g.signal(0, 0, 7);
      g.wait(1, 0, 5);
      require(g.ordered(1, p).why != qo::refusal::none && !g.ordered(1, p).ordered, "baseline refuses");
    }
    // Unsettled: a wait logged before the first post-sight signal completed.
    {
      host h;
      h.signal(0, 0, 1);  // Not complete yet.
      const auto p = h.execute(0);
      h.signal(0, 0, 2);
      h.wait(1, 0, 2);
      require(h.ordered(1, p).why == qo::refusal::unsettled, "unsettled refuses");
      h.complete(0, 1);
      h.wait(1, 0, 2);
      require(h.ordered(1, p).ordered, "a wait logged after settling proves");
    }
    // Settling needs a value above every value seen: a first signal at the
    // baseline cannot mark the queue's pre-sight work complete.
    {
      host h;
      h.fence_values[0] = 3;
      h.signal(0, 0, 3);
      h.complete(0, 3);
      const auto p = h.execute(0);
      h.signal(0, 0, 4);
      h.wait(1, 0, 4);
      require(h.ordered(1, p).why == qo::refusal::unsettled, "a first signal at the seen value does not settle");
    }
    // CPU signal: sticky poison, even before the decision.
    {
      host h;
      settle(h);
      const auto p = h.execute(0);
      h.signal(0, 0, 2);
      h.wait(1, 0, 2);
      h.cpu(0);
      require(h.ordered(1, p).why == qo::refusal::poisoned, "CPU signal poisons");
    }
    // Second signaller.
    {
      host h;
      settle(h);
      const auto p = h.execute(0);
      h.signal(2, 0, 2);
      h.signal(0, 0, 3);
      h.wait(1, 0, 3);
      require(!h.ordered(1, p).ordered, "a second signaller poisons");
    }
    // Non-monotonic and UINT64_MAX.
    {
      host h;
      settle(h);
      h.signal(0, 0, 5);
      const auto p = h.execute(0);
      h.signal(0, 0, 3);
      h.wait(1, 0, 3);
      require(!h.ordered(1, p).ordered, "a decreasing value poisons");
      host g;
      settle(g);
      const auto q = g.execute(0);
      g.signal(0, 0, UINT64_MAX);
      g.wait(1, 0, 2);
      require(!g.ordered(1, q).ordered, "UINT64_MAX poisons");
    }
    // Unobserved: a completed value no logged signal set.
    {
      host h;
      settle(h);
      h.signal(0, 0, 2);
      h.signal(0, 0, 4);
      const auto p = h.execute(0);
      h.signal(0, 0, 5);
      h.complete(0, 3);  // 3 was never signalled.
      h.wait(1, 0, 5);
      require(h.ordered(1, p).why == qo::refusal::poisoned, "a completed value no signal set poisons");
      host g;
      settle(g);
      g.complete(0, 9);  // Above anything signalled.
      const auto q = g.execute(0);
      g.signal(0, 0, 10);
      g.wait(1, 0, 10);
      require(!g.ordered(1, q).ordered, "a completed value above every signal poisons");
      // An in-flight value is a logged value (finding 16): the fast GPU.
      host k;
      settle(k);
      const auto r = k.execute(0);
      k.signal(0, 0, 2);
      k.complete(0, 2);
      k.wait(1, 0, 2);
      require(k.ordered(1, r).ordered, "a signal logged before forwarding is not unobserved");
    }
    // A hooked Signal from a queue without a record.
    {
      host h;
      settle(h);
      const auto p = h.execute(0);
      h.signal(0, 0, 2);
      h.signal_unrecorded(0, 3);
      h.wait(1, 0, 2);
      require(!h.ordered(1, p).ordered, "an unrecorded signaller poisons");
    }
    // Device removed: every fence of the device.
    {
      host h;
      settle(h, 0, 0);
      settle(h, 0, 1);
      const auto p = h.execute(0);
      h.signal(0, 1, 2);
      h.wait(1, 1, 2);
      h.complete(0, UINT64_MAX);
      require(h.ordered(1, p).why == qo::refusal::poisoned, "a removed device poisons its fences");
    }
    // Foreign device and cross device.
    {
      host h;
      h.queue(5, 0xd2);
      settle(h);
      const auto p = h.execute(0);
      require(h.ordered(5, p).why == qo::refusal::cross_device, "cross device refuses");
      h.signal(0, 0, 2);
      h.wait(5, 0, 2);
      require(!h.ordered(5, p).ordered, "a fence waited on another device's queue");
    }
    // Eviction: a signal ring that dropped the deciding entry.
    {
      host h;
      settle(h);
      const auto p = h.execute(0);
      for (std::uint64_t v = 2; v != 40; ++v) h.signal(0, 0, v);
      h.wait(1, 0, 3);
      require(h.ordered(1, p).why == qo::refusal::evicted, "an evicted signal refuses");
      // Within the ring, the eager resolution still decides.
      h.wait(1, 0, 39);
      require(h.ordered(1, p).ordered, "a signal still in the ring proves");
    }
    // Model reset: positions and records refuse; private-data memory keeps
    // the poison of a re-sighted fence (finding 16).
    {
      host h;
      settle(h);
      h.cpu(0);
      const auto memory = h.m.memory(h.fence_ids[0]);
      h.m.reset();
      const auto id = h.m.add_fence(h.device, 1, memory);
      require(h.m.memory(id).why == qo::poison::cpu_signal, "a re-sighted fence keeps its poison");
      const auto p = qo::position{h.queue_ids[0], 3, true};
      require(h.m.ordered(h.queue(1), p).why == qo::refusal::untracked, "a reset forgets positions");
      // The signaller persists too, by its queue's stable cookie: the same
      // queue signals on after the re-sight, another one is a second
      // signaller.
      host g;
      settle(g);
      const auto kept = g.m.memory(g.fence_ids[0]);
      g.m.reset();
      g.queue_ids.clear();
      const auto same = g.m.add_fence(g.device, 1, kept);
      g.m.signal(g.queue(0), same, 2, 1, 1, g.now);
      require(g.m.memory(same).why == qo::poison::none, "the persisted signaller signals on after a reset");
      g.m.signal(g.queue(7), same, 3, 1, 1, g.now);
      require(g.m.memory(same).why == qo::poison::second_signaller, "a persisted signaller stays the only one");
    }
    // Late conflict: a poison of a fence a live read relied on.
    {
      host h;
      settle(h);
      const auto p = h.execute(0);
      h.signal(0, 0, 2);
      h.wait(1, 0, 2);
      const auto v = h.ordered(1, p);
      h.m.rely(v, h.now, 41);
      h.m.rely(v, h.now);  // A read without a tag counts, and marks nothing.
      require(!h.m.conflicted(41), "a relied read is conflicted before any late signal");
      h.cpu(0);
      require(h.m.snapshot().late_conflict == 2, "a poison after a relied read is a late conflict");
      // O1: the read's presentation copy is not held any further, also
      // after a model reset; another read is not conflicted.
      require(h.m.conflicted(41) && !h.m.conflicted(42) && !h.m.conflicted(0), "a late conflict did not mark its read");
      h.m.reset();
      require(h.m.conflicted(41), "a model reset forgot a conflicted read");
      host g;
      settle(g);
      const auto q = g.execute(0);
      g.signal(0, 0, 2);
      g.wait(1, 0, 2);
      g.m.rely(g.ordered(1, q), g.now, 43);
      g.now += model::live_read_ms + 1;
      g.cpu(0);
      require(g.m.snapshot().late_conflict == 0 && !g.m.conflicted(43), "an expired read has no late conflict");
      // A refused verdict relies on nothing.
      host r;
      settle(r);
      r.m.rely(r.ordered(1, r.execute(0)), r.now, 44);
      r.cpu(0);
      require(r.m.snapshot().late_conflict == 0 && !r.m.conflicted(44), "a refused verdict was relied on");
    }
    // O1 fail-closed: the decisive fence is recorded however many other
    // clean fences the reader waited on first (until 10-27 the verdict kept
    // the first four of its log, so a late signal of a fifth conflicted
    // nothing); a fence off the chain conflicts nothing.
    {
      host h;
      for (int f = 1; f != 6; ++f) {
        settle(h, f + 1, f);
        h.signal(f + 1, f, 2);
        h.wait(1, f, 2);
      }
      settle(h);
      const auto p = h.execute(0);
      h.signal(0, 0, 2);
      h.wait(1, 0, 2);
      const auto v = h.ordered(1, p);
      require(v.ordered && !v.unbounded && v.fences[0] == h.fence_ids[0] && !v.fences[1],
        "a verdict did not rest on the fence that ordered the producer's queue");
      h.m.rely(v, h.now, 51);
      h.cpu(1);
      require(!h.m.conflicted(51), "a late signal of a fence off the chain conflicted a read");
      h.cpu(0);
      require(h.m.conflicted(51), "a late signal of the decisive fence after four other waited fences conflicted nothing");
    }
    // A chain longer than a verdict holds rests on every fence.
    {
      host h;
      for (int f = 0; f != 5; ++f) settle(h, f, f);
      const auto p = h.execute(0);
      for (int f = 0; f != 5; ++f) {
        h.signal(f, f, 2);
        h.wait(f + 1, f, 2);
      }
      const auto v = h.ordered(5, p);
      require(v.ordered && v.unbounded, "a chain of five fences was not unbounded");
      h.m.rely(v, h.now, 55);
      h.cpu(9);
      require(h.m.conflicted(55), "a poison did not conflict an unbounded read");
    }
    // A model reset (a coverage change) conflicts every live relied read,
    // and an expired one not. Each conflict is counted by its cause: only a
    // poison is a late conflict (the R1 residual); a reset, an evicted fence
    // record or an evicted live read count apart (11-01, F57: each counted
    // as a late conflict, which the readiness report calls R1).
    {
      host h;
      settle(h);
      const auto p = h.execute(0);
      h.signal(0, 0, 2);
      h.wait(1, 0, 2);
      const auto v = h.ordered(1, p);
      h.m.rely(v, h.now, 52);
      h.m.rely(v, h.now - model::live_read_ms - 1, 56);
      h.m.reset(h.now);
      require(h.m.conflicted(52) && !h.m.conflicted(56), "a model reset left a live relied read unconflicted");
      require(h.m.snapshot().late_conflict == 0, "a model reset counted a late conflict");
      require(h.m.snapshot().reset_conflicts == 1 && h.m.snapshot().evicted_reads == 0, "a model reset's conflict was not counted as one");
    }
    // A reset between the verdict and rely forgets the verdict's fence: the
    // read is conflicted at rely, counted as evicted, not late.
    {
      host h;
      settle(h);
      const auto p = h.execute(0);
      h.signal(0, 0, 2);
      h.wait(1, 0, 2);
      const auto v = h.ordered(1, p);
      h.m.reset(h.now);
      require(!h.m.rely(v, h.now, 59) && h.m.conflicted(59), "a read relied on a fence a reset forgot");
      require(h.m.snapshot().late_conflict == 0, "a forgotten fence counted a late conflict");
      require(h.m.snapshot().evicted_reads == 1 && h.m.snapshot().reset_conflicts == 0, "a forgotten fence's conflict was not counted as evicted");
    }
    // An evicted fence record conflicts the live reads that relied on it:
    // its later signals are never logged against it.
    {
      host h;
      settle(h);
      const auto p = h.execute(0);
      h.signal(0, 0, 2);
      h.wait(1, 0, 2);
      h.m.rely(h.ordered(1, p), h.now, 53);
      for (unsigned i = 0; i != 256; ++i) h.m.add_fence(h.device, 0, {}, qo::poison::none, h.now);
      require(!h.m.known_fence(h.fence_ids[0]) && h.m.conflicted(53), "an evicted fence record left its relied read unconflicted");
      require(h.m.snapshot().late_conflict == 0, "an evicted fence record counted a late conflict");
      require(h.m.snapshot().evicted_reads == 1, "an evicted fence record's conflict was not counted as evicted");
    }
    // A fence poisoned between the verdict and rely (another lock hold)
    // conflicts the read at once.
    {
      host h;
      settle(h);
      const auto p = h.execute(0);
      h.signal(0, 0, 2);
      h.wait(1, 0, 2);
      const auto v = h.ordered(1, p);
      h.cpu(0);
      // The caller has not copied yet: rely says the read no longer stands,
      // so it is never made (consume_owned refuses it).
      require(!h.m.rely(v, h.now, 54), "rely let a read stand on a fence poisoned since its verdict");
      require(h.m.conflicted(54), "a read relied on a fence poisoned since its verdict");
      require(h.m.snapshot().late_conflict == 1, "a fence poisoned since its verdict did not count a late conflict");
      const auto w = h.ordered(1, p);
      require(!w.ordered, "a poisoned fence still ordered a read");
    }
    // A read whose fences are all clean at rely stands.
    {
      host h;
      settle(h);
      const auto p = h.execute(0);
      h.signal(0, 0, 2);
      h.wait(1, 0, 2);
      require(h.m.rely(h.ordered(1, p), h.now, 57) && !h.m.conflicted(57), "a clean relied read did not stand");
      require(!h.m.rely({}, h.now, 58), "a refused verdict was relied on");
    }
    // A live read evicted by later reads is conflicted (fail-closed).
    {
      host h;
      settle(h);
      const auto p = h.execute(0);
      h.signal(0, 0, 2);
      h.wait(1, 0, 2);
      const auto v = h.ordered(1, p);
      for (std::uint64_t tag = 100; tag != 165; ++tag) h.m.rely(v, h.now, tag);
      require(h.m.conflicted(100) && !h.m.conflicted(101), "a live relied read was evicted unconflicted");
      require(h.m.snapshot().late_conflict == 0, "an evicted live read counted a late conflict");
      require(h.m.snapshot().evicted_reads == 1, "an evicted live read's conflict was not counted as evicted");
    }
    // Keyed by id, not by pointer: a forgotten device's records are gone.
    {
      host h;
      settle(h);
      const auto p = h.execute(0);
      h.m.forget_device(h.device);
      require(h.m.ordered(h.queue(1), p).why == qo::refusal::untracked, "a forgotten device refuses");
    }
  }

  // -- The independent oracle ----------------------------------------------
  // Ops in CPU submission order. A queue op becomes runnable on the GPU at its
  // submission; a CPU signal applies at its submission. The oracle explores
  // every interleaving of the CPU program with GPU execution and asks whether
  // a read appended to the reader after the last op can run before the
  // producer's execute op completed.
  enum class kind { execute, signal, wait, cpu };
  struct op {
    kind k;
    int queue, fence;
    std::uint64_t value;
    bool quiet = true;
  };
  struct script {
    int queues, fences;
    std::vector<op> ops;
    int producer;  // Index of an execute op.
    int reader;
  };

  bool oracle_ordered(const script &s) {
    // Per queue, the indices of its ops in CPU order, then the read.
    std::vector<std::vector<int>> lanes(s.queues);
    for (int i = 0; i != int(s.ops.size()); ++i)
      if (s.ops[i].k != kind::cpu) lanes[s.ops[i].queue].push_back(i);
    const int read = -1;
    lanes[s.reader].push_back(read);
    const int n = int(s.ops.size());
    int producer_lane_index = 0;
    for (int i = 0; i != int(lanes[s.ops[s.producer].queue].size()); ++i)
      if (lanes[s.ops[s.producer].queue][i] == s.producer) producer_lane_index = i;
    using state = std::tuple<int, std::vector<int>, std::vector<std::uint64_t>>;
    std::set<state> seen;
    std::vector<state> work{{0, std::vector<int>(s.queues), std::vector<std::uint64_t>(s.fences)}};
    while (!work.empty()) {
      auto [cpu, pcs, values] = work.back();
      work.pop_back();
      if (!seen.insert({cpu, pcs, values}).second) continue;
      // The CPU submits its next op (the read is submitted after all of them).
      if (cpu < n) {
        auto next = values;
        if (s.ops[cpu].k == kind::cpu) next[s.ops[cpu].fence] = s.ops[cpu].value;
        work.push_back({cpu + 1, pcs, next});
      }
      for (int q = 0; q != s.queues; ++q) {
        const auto &lane = lanes[q];
        if (pcs[q] >= int(lane.size())) continue;
        const int index = lane[pcs[q]];
        if (index == read) {
          if (cpu < n) continue;  // Not yet submitted.
          const int pq = s.ops[s.producer].queue;
          if (pcs[pq] <= producer_lane_index) return false;  // Read before the producer completed.
          continue;
        }
        if (index >= cpu) continue;  // Not yet submitted.
        const auto &o = s.ops[index];
        auto next = values;
        if (o.k == kind::wait && values[o.fence] < o.value) continue;
        if (o.k == kind::signal) next[o.fence] = o.value;
        auto advanced = pcs;
        ++advanced[q];
        work.push_back({cpu, advanced, next});
      }
    }
    return true;
  }

  // Feeds the script to the model as the hooks would, with the completed
  // values of one eager GPU schedule (each op runs as soon as it can).
  qo::verdict model_verdict(const script &s) {
    host h;
    std::vector<std::uint64_t> values(s.fences);
    std::vector<std::vector<int>> lanes(s.queues);
    std::vector<int> pcs(s.queues);
    qo::position producer;
    const auto run_gpu = [&](int submitted) {
      for (bool progress = true; progress;) {
        progress = false;
        for (int q = 0; q != s.queues; ++q) {
          if (pcs[q] >= int(lanes[q].size())) continue;
          const int index = lanes[q][pcs[q]];
          if (index >= submitted) continue;
          const auto &o = s.ops[index];
          if (o.k == kind::wait && values[o.fence] < o.value) continue;
          if (o.k == kind::signal) values[o.fence] = o.value;
          ++pcs[q];
          progress = true;
        }
      }
    };
    for (int i = 0; i != int(s.ops.size()); ++i) {
      const auto &o = s.ops[i];
      if (o.k != kind::cpu) lanes[o.queue].push_back(i);
      for (int f = 0; f != s.fences; ++f) h.fence_values[f] = values[f];
      switch (o.k) {
        case kind::execute: {
          const auto at = h.execute(o.queue, o.quiet);
          if (i == s.producer) producer = at;
          break;
        }
        case kind::signal: h.signal(o.queue, o.fence, o.value); break;
        case kind::wait: break;  // Logged after its native call returned (below).
        case kind::cpu: h.cpu(o.fence); values[o.fence] = o.value; break;
      }
      run_gpu(i + 1);
      for (int f = 0; f != s.fences; ++f) h.fence_values[f] = values[f];
      if (o.k == kind::wait) h.wait(o.queue, o.fence, o.value, o.quiet);
      // A hooked call on each fence samples it now and then.
      if (o.k != kind::cpu && (i & 1)) h.complete(o.fence, values[o.fence]);
    }
    return h.ordered(s.reader, producer);
  }

  script generate(std::mt19937_64 &random) {
    script s;
    s.queues = 2 + int(random() % 2);
    s.fences = 1 + int(random() % 2);
    const int count = 4 + int(random() % 7);
    // Mostly increasing values per fence (as games signal), sometimes any
    // value, so that every fail-closed rule is met as well as real edges.
    std::vector<std::uint64_t> next(s.fences);
    for (int i = 0; i != count; ++i) {
      const auto pick = random() % 20;
      op o{};
      o.queue = int(random() % s.queues);
      o.fence = int(random() % s.fences);
      o.quiet = random() % 8 != 0;
      o.k = pick < 6 ? kind::execute : pick < 12 ? kind::signal : pick < 19 ? kind::wait : kind::cpu;
      auto &top = next[o.fence];
      if (o.k == kind::signal) o.value = random() % 6 ? ++top : 1 + random() % (top + 1);
      else o.value = 1 + random() % (top + 2);
      s.ops.push_back(o);
    }
    std::vector<int> executes;
    for (int i = 0; i != count; ++i)
      if (s.ops[i].k == kind::execute) executes.push_back(i);
    if (executes.empty()) {
      s.ops.insert(s.ops.begin(), op{kind::execute, 0, 0, 0, true});
      executes.push_back(0);
    }
    s.producer = executes[random() % executes.size()];
    s.reader = (s.ops[s.producer].queue + 1 + int(random() % (s.queues - 1))) % s.queues;
    return s;
  }

  std::string describe(const script &s) {
    std::string out;
    for (const auto &o : s.ops) {
      char line[96];
      const char *k = o.k == kind::execute ? "E" : o.k == kind::signal ? "S" : o.k == kind::wait ? "W" : "C";
      std::snprintf(line, sizeof line, "%s q%d f%d v%llu%s; ", k, o.queue, o.fence,
        static_cast<unsigned long long>(o.value), o.quiet ? "" : " (not quiet)");
      out += line;
    }
    return out + "producer #" + std::to_string(s.producer) + " reader q" + std::to_string(s.reader);
  }

  void generated() {
    unsigned ordered = 0, oracle_true = 0;
    for (std::uint64_t seed = 1; seed <= 20000; ++seed) {
      std::mt19937_64 random(seed * 0x9e3779b97f4a7c15ull);
      const auto s = generate(random);
      const auto v = model_verdict(s);
      const bool truth = oracle_ordered(s);
      oracle_true += truth;
      if (v.ordered) {
        ++ordered;
        require(truth, "UNSOUND at seed " + std::to_string(seed) + ": " + describe(s));
      }
    }
    std::printf("generated: 20000 scripts, model ordered %u, oracle ordered %u\n", ordered, oracle_true);
    require(ordered >= 200, "too few ordered verdicts for the soundness check to mean anything");
  }

  // Completeness: fixed scripts the model must order.
  void positives() {
    const std::vector<script> fixed{
      // Producer, signal, settle sample, reader waits.
      {2, 1, {{kind::signal, 0, 0, 1}, {kind::execute, 1, 0, 0}, {kind::execute, 0, 0, 0}, {kind::signal, 0, 0, 2},
        {kind::wait, 1, 0, 2}}, 2, 1},
      // Wait before signal.
      {2, 1, {{kind::signal, 0, 0, 1}, {kind::execute, 1, 0, 0}, {kind::execute, 0, 0, 0}, {kind::wait, 1, 0, 2},
        {kind::signal, 0, 0, 2}}, 2, 1},
      // Two hops.
      {3, 2, {{kind::signal, 0, 0, 1}, {kind::signal, 1, 1, 1}, {kind::execute, 2, 0, 0}, {kind::execute, 0, 0, 0},
        {kind::signal, 0, 0, 2}, {kind::wait, 1, 0, 2}, {kind::signal, 1, 1, 2}, {kind::wait, 2, 1, 2}}, 3, 2},
    };
    for (const auto &s : fixed) {
      require(oracle_ordered(s), "a fixed positive is not ordered by the oracle: " + describe(s));
      require(model_verdict(s).ordered, "a fixed positive is not ordered by the model: " + describe(s));
    }
  }
}

namespace {
  // The fixtures' script oracle (test_queue_order_script.h), which judges
  // their in-flight reads (unordered_read): one cell per rule.
  void script_oracle() {
    using script = queue_order_script::log;
    int p = 0, q = 0, r = 0, g = 0, h = 0;
    const auto judged = [](script &value) { return value.unordered_reads().empty(); };
    {
      script s; s.enabled = true;
      s.execute(&p, 1, 1); s.signal(&p, &g, 1); s.wait(&r, &g, 1); s.present(&r, 10, true, 1, 1);
      require(judged(s), "oracle: a wait for the producer's later signal does not order the read");
    }
    {
      script s; s.enabled = true;
      s.signal(&p, &g, 1); s.wait(&r, &g, 1); s.execute(&p, 1, 1); s.signal(&p, &g, 2); s.present(&r, 10, true, 1, 1);
      require(!judged(s), "oracle: a wait for a signal before the producer orders the read");
    }
    {
      script s; s.enabled = true;
      s.execute(&p, 1, 1); s.signal(&p, &g, 1); s.cpu_signal(&g, 1); s.wait(&r, &g, 1); s.present(&r, 10, true, 1, 1);
      require(!judged(s), "oracle: a CPU signal reaching the waited value orders the read");
    }
    {
      script s; s.enabled = true;
      s.execute(&p, 1, 1); s.signal(&p, &g, 1); s.signal(&q, &g, 1); s.wait(&r, &g, 1); s.present(&r, 10, true, 1, 1);
      require(!judged(s), "oracle: a second signaller reaching the waited value orders the read");
    }
    // A signal first submitted after the read (O1's residual): the read is
    // unordered, and late (R1-late-signaller), which unordered_reads excuses
    // and counts; a signal before it is no late one.
    {
      script s; s.enabled = true;
      s.execute(&p, 1, 1); s.signal(&p, &g, 1); s.wait(&r, &g, 1); s.present(&r, 10, true, 1, 1); s.signal(&q, &g, 1);
      auto copy = s;
      const auto judged_reads = copy.judge_reads();
      require(judged_reads.size() == 1 && judged_reads[0].late && judged_reads[0].present_index == 10,
        "oracle: a second signaller submitted after the read is no late signaller");
      require(judged(s) && s.late_reads == 1, "oracle: a late second signaller was not excused and counted");
    }
    {
      script s; s.enabled = true;
      s.execute(&p, 1, 1); s.signal(&p, &g, 1); s.wait(&r, &g, 1); s.present(&r, 10, true, 1, 1); s.cpu_signal(&g, 2);
      require(judged(s) && s.late_reads == 1, "oracle: a late CPU signal was not excused and counted");
      s.clear();
      s.execute(&p, 1, 1); s.signal(&p, &g, 1); s.cpu_signal(&g, 1); s.wait(&r, &g, 1); s.present(&r, 10, true, 1, 1);
      require(!judged(s) && s.late_reads == 0, "oracle: a CPU signal before the read counted as late");
    }
    // Wait before signal: the presenting queue waits for a value its
    // producer queue signals only afterwards (legal): ordered.
    {
      script s; s.enabled = true;
      s.wait(&r, &g, 1); s.execute(&p, 1, 1); s.signal(&p, &g, 1); s.present(&r, 10, true, 1, 1);
      require(judged(s) && s.late_reads == 0, "oracle: a wait before its signal does not order the read");
    }
    {
      script s; s.enabled = true; s.initial[&g] = 5;
      s.execute(&p, 1, 1); s.signal(&p, &g, 6); s.wait(&r, &g, 5); s.present(&r, 10, true, 1, 1);
      require(!judged(s), "oracle: a wait at the fence's initial value orders the read");
      s.wait(&r, &g, 6); s.present(&r, 11, true, 1, 1);
      require(judged(s), "oracle: a wait above the initial value does not order the read");
    }
    {
      script s; s.enabled = true;
      s.execute(&p, 1, 1); s.signal(&p, &g, 1); s.wait(&q, &g, 1); s.signal(&q, &h, 1); s.wait(&r, &h, 1);
      s.present(&r, 10, true, 1, 1);
      require(judged(s), "oracle: a two-hop chain does not order the read");
    }
    {
      script s; s.enabled = true;
      s.execute(&p, 1, 1); s.signal(&p, &g, 1); s.wait(&r, &g, 1); s.present(&r, 10, true, 1, 2);
      require(!judged(s), "oracle: a read naming no logged Execute passed");
    }
    {
      script s; s.enabled = true;
      s.execute(&p, 1, 1); s.signal(&p, &g, 1); s.present(&r, 10);
      s.in_flight(s.last(queue_order_script::kind::present, &r), 10, s.last(queue_order_script::kind::execute, &p));
      require(!judged(s), "oracle: an in-flight read without a wait passed");
    }
    std::puts("queue order script oracle: PASS");
  }
}

int main() {
  try {
    script_oracle();
    cells();
    positives();
    generated();
    std::puts("queue order model: PASS");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "queue order model: FAIL: %s\n", error.what());
    return 1;
  }
}
