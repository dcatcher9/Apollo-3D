// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The queue-order model (Step 4, docs/reshade-sbs.md, Queue order): which
// game queue's work is ordered after a given submission on another queue,
// from the queue Signal and Wait calls the native observer hooks see. Pure:
// it holds no COM pointer and never calls D3D12; the caller locks (the
// observer's order_lock: exclusive for the mutating calls, shared for
// ordered()).
//
// An edge "reader queue R is ordered after position p on queue P" rests on
// D3D12 queue order (a queue starts a call only after the calls before it,
// and Signal(G, v) updates G only after all earlier work on its queue), on
// Wait semantics (work on R after Wait(G, w) starts only once G >= w), and on
// every source that can raise G being observed. A fence that ever has a
// second source, a CPU signal, a value the logged signals never set, a
// decreasing or UINT64_MAX value, or that is shared or of another device
// never proves (poison, sticky; the host persists it in the fence's own
// private data so that a model reset or an eviction never turns a fence the
// add-on has seen into a clean record). What no observation can exclude is
// the residual of section 1 of the design (R1-late-signaller).
//
// Positions are tickets: each hooked Execute/Signal/Wait call on a tracked
// queue takes ticket t = ++entered at entry; quiet means no later ticket of
// that queue was taken before its native call returned. A quiet call
// natively precedes every call with a larger ticket on its queue.
#include <algorithm>
#include <array>
#include <cstdint>

namespace sunshine_queue_order {
  struct position {
    std::uint64_t queue{}, ticket{};
    bool quiet{};
    explicit operator bool() const { return queue && ticket; }
  };
  enum class poison : std::uint8_t {
    none,
    shared,  // A shared, cross-adapter or non-monitored fence (or no ID3D12Fence1).
    foreign_device,  // Signalled or waited on a queue of another device.
    cpu_signal,  // ID3D12Fence::Signal, whatever its result.
    second_signaller,  // A queue other than the first post-sight signaller.
    non_monotonic,  // A value below one already signalled.
    unobserved_signal,  // A completed value no logged signal set.
    removed,  // UINT64_MAX: a signalled or read device-removed value.
    unrecorded_signaller,  // A hooked Signal from a queue the model has no record of.
    count
  };
  enum class refusal : std::uint8_t {
    none,
    disabled,  // QueueFenceOrder=0, or the model was just reset.
    pending_hooks,  // An order slot is discovered but not yet installed.
    uncovered,  // The reader's or producer's queue lacks an installed Execute/Signal/Wait slot.
    untracked,  // A position or queue the model has no record of.
    overlap,  // The producer call was not quiet.
    unresolved,  // A wait logged before any signal reaching its value.
    baseline,  // A wait value at or below the fence's value at first sight.
    unsettled,  // A wait logged before the fence's pre-sight signals completed.
    poisoned,
    evicted,  // A ring dropped the entry that decides.
    cross_device,
    count
  };
  inline const char *name(poison value) {
    switch (value) {
      case poison::none: return "none";
      case poison::shared: return "shared";
      case poison::foreign_device: return "foreign";
      case poison::cpu_signal: return "cpu";
      case poison::second_signaller: return "second";
      case poison::non_monotonic: return "monotonic";
      case poison::unobserved_signal: return "unobserved";
      case poison::removed: return "removed";
      case poison::unrecorded_signaller: return "unrecorded";
      default: return "unknown";
    }
  }
  inline const char *name(refusal value) {
    switch (value) {
      case refusal::none: return "none";
      case refusal::disabled: return "disabled";
      case refusal::pending_hooks: return "pending_hooks";
      case refusal::uncovered: return "uncovered";
      case refusal::untracked: return "untracked";
      case refusal::overlap: return "overlap";
      case refusal::unresolved: return "unresolved";
      case refusal::baseline: return "baseline";
      case refusal::unsettled: return "unsettled";
      case refusal::poisoned: return "poisoned";
      case refusal::evicted: return "evicted";
      case refusal::cross_device: return "cross_device";
      default: return "unknown";
    }
  }
  struct verdict {
    bool ordered{};
    refusal why{};
    // The fences the edge rests on (late_conflict): those of the wait chain
    // that ordered the producer's queue, from it back to the reader. A chain
    // of more fences, or one the walk cannot follow, rests on every fence
    // (unbounded: any poison within the source age conflicts its read).
    std::array<std::uint64_t, 4> fences{};
    bool unbounded{};
  };
  // What the host persists in a fence's private data (finding 16): its
  // record id, its signaller's stable queue cookie (the queue's own private
  // data, which a model reset never changes) and its poison.
  struct fence_memory {
    std::uint64_t id{}, signaller{};
    poison why{};
  };
  // late_conflict: relied reads a poison of one of their fences conflicted
  // within the source age (the R1 residual). reset_conflicts and
  // evicted_reads: those conflicted because their protection lapsed, at a
  // model reset, or when a fence record they rested on was evicted or
  // forgotten or the read itself was evicted while live (fail-closed; until
  // 11-01 all three counted as late_conflict, F57).
  struct counts {
    std::uint64_t queues{}, fences{}, settled{}, signals{}, waits{}, cpu_signals{}, late_conflict{}, evicted_waits{};
    std::uint64_t reset_conflicts{}, evicted_reads{};
    std::array<std::uint64_t, static_cast<unsigned>(poison::count)> poisoned{};
  };

  // LiveReads: the relied reads kept for the source age, at least one per
  // Present at 250 Presents a second (a read evicted while live is
  // conflicted at once, fail-closed); also the conflicted tags kept.
  template<unsigned Queues = 32, unsigned Fences = 256, unsigned WaitLog = 64, unsigned SignalLog = 32,
    unsigned Pending = 8, unsigned LiveReads = 64>
  class model {
    static_assert(Queues < 255 && Fences < 65535, "ids keep the slot in their low bits");

  public:
    static constexpr std::uint64_t live_read_ms = 250;  // The source age (freshness.h).

    // A new queue record (the least recently used one is evicted when the
    // table is full). cookie: the queue's stable identity across records.
    std::uint64_t add_queue(std::uint64_t device, std::uint64_t cookie) {
      unsigned slot = Queues;
      for (unsigned i = 0; i != Queues; ++i)
        if (!queues_[i].id) { slot = i; break; }
      if (slot == Queues) {
        slot = 0;
        for (unsigned i = 1; i != Queues; ++i)
          if (queues_[i].last_use < queues_[slot].last_use) slot = i;
      }
      auto &q = queues_[slot];
      q = {};
      q.id = (slot + 1) | (++serial_ << 8);
      q.device = device;
      q.cookie = cookie;
      q.last_use = ++clock_;
      return q.id;
    }
    bool known_queue(std::uint64_t id) const { return queue_of(id) != nullptr; }
    std::uint64_t queue_device(std::uint64_t id) const {
      const auto *q = queue_of(id);
      return q ? q->device : 0;
    }
    // A fence record at its (re-)sight: memory restores what an earlier
    // record learned (the private data), baseline is its completed value now,
    // and fresh is a poison found at this sight (shared, foreign device).
    // An evicted record's later signals are never logged against it, so a
    // live read that relied on it is conflicted now (fail-closed).
    std::uint64_t add_fence(std::uint64_t device, std::uint64_t baseline, fence_memory memory = {}, poison fresh = poison::none,
        std::uint64_t now = 0) {
      unsigned slot = Fences;
      for (unsigned i = 0; i != Fences; ++i)
        if (!fences_[i].id) { slot = i; break; }
      if (slot == Fences) {
        slot = 0;
        for (unsigned i = 1; i != Fences; ++i)
          if (fences_[i].last_use < fences_[slot].last_use) slot = i;
        conflict_reads(fences_[slot].id, now, &counts::evicted_reads);
      }
      auto &f = fences_[slot];
      f = {};
      f.id = (slot + 1) | (++serial_ << 16);
      f.device = device;
      f.baseline = f.seen_max = baseline;
      f.signaller = memory.signaller;
      f.why = memory.why;
      f.last_use = ++clock_;
      if (fresh != poison::none) set_poison(f, fresh, now);
      if (baseline == UINT64_MAX) set_poison(f, poison::removed, now);
      return f.id;
    }
    bool known_fence(std::uint64_t id) const { return fence_of(id) != nullptr; }
    fence_memory memory(std::uint64_t id) const {
      const auto *f = fence_of(id);
      return f ? fence_memory{f->id, f->signaller, f->why} : fence_memory{};
    }
    void poison_fence(std::uint64_t id, poison why, std::uint64_t now) {
      if (auto *f = fence_of(id)) set_poison(*f, why, now);
    }

    // A hooked queue Signal, logged BEFORE it is forwarded (finding 7):
    // queue 0 is a signalling queue without a record. completed is the
    // fence's value read in the same call.
    void signal(std::uint64_t queue, std::uint64_t fence, std::uint64_t value, std::uint64_t ticket,
        std::uint64_t completed, std::uint64_t now) {
      auto *f = fence_of(fence);
      if (!f) return;
      f->last_use = ++clock_;
      ++counts_.signals;
      if (completed != UINT64_MAX) f->seen_max = std::max(f->seen_max, completed);
      // An in-flight value is a logged value for the unobserved check
      // (finding 16): append before checking what completed.
      auto *q = queue_of(queue);
      // A queue that only signals (an async producer) is in use: never the
      // least recently used record while stale ones of destroyed queues stay.
      if (q) q->last_use = ++clock_;
      if (f->why == poison::none) {
        if (!q) set_poison(*f, poison::unrecorded_signaller, now);
        else if (value == UINT64_MAX) set_poison(*f, poison::removed, now);
        else if (f->signaller && f->signaller != q->cookie) set_poison(*f, poison::second_signaller, now);
        else if (f->signalled && value < f->max_signalled) set_poison(*f, poison::non_monotonic, now);
        else if (q->device != f->device) set_poison(*f, poison::foreign_device, now);
      }
      if (f->why == poison::none) {
        if (!f->signaller) f->signaller = q->cookie;
        // The settle marker: the first logged signal above every value the
        // fence was seen at, so that reading it complete means this signal
        // (or a later one of its queue) executed, after every pre-sight
        // signal of that queue.
        if (!f->settle_value && value > f->seen_max) f->settle_value = value;
        f->signalled = true;
        f->max_signalled = std::max(f->max_signalled, value);
        if (f->size == SignalLog) {
          f->evicted_max = std::max(f->evicted_max, f->ring[f->head].value);
          f->evicted = true;
          f->head = (f->head + 1) % SignalLog;
          --f->size;
        }
        f->ring[(f->head + f->size++) % SignalLog] = {value, ticket, queue};
        // The first logged signal reaching a pending wait's value satisfies it.
        for (unsigned i = 0; i < f->pending_count;) {
          auto &p = f->pending[i];
          auto *w = pending_wait(p);
          if (w && w->value <= value) {
            w->state = wait_state::resolved;
            w->signaller = queue;
            w->signal_ticket = ticket;
          }
          if (!w || w->value <= value) p = f->pending[--f->pending_count];
          else ++i;
        }
      }
      observed_completed(fence, completed, now);
    }
    // A hooked queue Wait, logged AFTER its native call returned success.
    void wait(std::uint64_t queue, std::uint64_t fence, std::uint64_t value, std::uint64_t ticket, bool quiet,
        std::uint64_t completed, std::uint64_t now) {
      auto *q = queue_of(queue);
      if (!q) return;
      q->last_use = ++clock_;
      ++counts_.waits;
      if (fence_of(fence)) observed_completed(fence, completed, now);
      if (q->size == WaitLog) { q->head = (q->head + 1) % WaitLog; --q->size; }
      auto &w = q->waits[(q->head + q->size++) % WaitLog];
      w = {};
      w.fence = fence; w.value = value; w.ticket = ticket; w.quiet = quiet; w.serial = ++serial_;
      w.state = wait_state::refused;
      auto *f = fence_of(fence);
      if (!f) { w.why = refusal::untracked; return; }
      f->last_use = ++clock_;
      if (f->device != q->device) set_poison(*f, poison::foreign_device, now);
      if (f->why != poison::none) { w.why = refusal::poisoned; return; }
      if (!f->settled) { w.why = refusal::unsettled; return; }
      if (value <= f->baseline) { w.why = refusal::baseline; return; }
      if (f->evicted && f->evicted_max >= value) { w.why = refusal::evicted; ++counts_.evicted_waits; return; }
      for (unsigned i = 0; i != f->size; ++i) {
        const auto &s = f->ring[(f->head + i) % SignalLog];
        if (s.value >= value) {
          w.state = wait_state::resolved;
          w.signaller = s.queue;
          w.signal_ticket = s.ticket;
          return;
        }
      }
      if (f->pending_count == Pending) { w.why = refusal::evicted; ++counts_.evicted_waits; return; }
      w.state = wait_state::pending;
      w.why = refusal::unresolved;
      f->pending[f->pending_count++] = {static_cast<unsigned>(q - queues_.data()),
        static_cast<unsigned>(&w - q->waits.data()), w.serial};
    }
    // A hooked ID3D12Fence::Signal (CPU), logged before forwarding.
    void cpu_signal(std::uint64_t fence, std::uint64_t now) {
      ++counts_.cpu_signals;
      if (auto *f = fence_of(fence)) set_poison(*f, poison::cpu_signal, now);
    }
    // A completed value read in a hooked call on the fence.
    void observed_completed(std::uint64_t fence, std::uint64_t completed, std::uint64_t now) {
      auto *f = fence_of(fence);
      if (!f) return;
      if (completed == UINT64_MAX) {
        for (auto &other : fences_)
          if (other.id && other.device == f->device) set_poison(other, poison::removed, now);
        return;
      }
      if (f->why != poison::none) return;
      f->seen_max = std::max(f->seen_max, completed);
      if (completed > std::max(f->baseline, f->max_signalled)) {
        set_poison(*f, poison::unobserved_signal, now);
        return;
      }
      // After settling, every value the fence takes is a logged one.
      if (f->settled && completed != f->baseline && completed > f->evicted_max) {
        bool logged = false;
        for (unsigned i = 0; i != f->size && !logged; ++i) logged = f->ring[(f->head + i) % SignalLog].value == completed;
        if (!logged) { set_poison(*f, poison::unobserved_signal, now); return; }
      }
      // The first post-sight signal executed: every pre-sight signal of its
      // queue executed before it.
      if (!f->settled && f->settle_value && completed >= f->settle_value) {
        f->settled = true;
        ++counts_.settled;
      }
    }
    void forget_device(std::uint64_t device) {
      for (auto &q : queues_)
        if (q.id && q.device == device) q = {};
      for (auto &f : fences_)
        if (f.id && f.device == device) f = {};
    }
    // A coverage change: a signal may have bypassed the hooks, so every live
    // relied read is conflicted before its fences are forgotten.
    void reset(std::uint64_t now = 0) {
      for (auto &read : reads_)
        if (read.expires > now) conflict(read, &counts::reset_conflicts);
      for (auto &q : queues_) q = {};
      for (auto &f : fences_) f = {};
      reads_ = {};
    }

    // Is every call reader submits from now on ordered after the producer's
    // call at p? Every logged wait of the reader counts (it returned before
    // this query); a wait of another queue X counts only if it was quiet and
    // before the signal of X the frontier already reached.
    verdict ordered(std::uint64_t reader, position p) const {
      verdict out;
      const auto *r = queue_of(reader);
      const auto *producer = p ? queue_of(p.queue) : nullptr;
      if (!r || !producer) { out.why = refusal::untracked; return out; }
      if (r->device != producer->device) { out.why = refusal::cross_device; return out; }
      if (reader == p.queue) { out.ordered = true; return out; }
      if (!p.quiet) { out.why = refusal::overlap; return out; }
      std::array<std::uint64_t, Queues> frontier{}, via_fence{};
      std::array<unsigned, Queues> via_queue{};
      std::array<bool, Queues> listed{};
      std::array<unsigned, Queues> stack{};
      unsigned depth = 0;
      const auto rslot = slot_of(*r), pslot = slot_of(*producer);
      frontier[rslot] = UINT64_MAX;
      stack[depth++] = rslot; listed[rslot] = true;
      refusal first = refusal::none;
      const auto note = [&](refusal why) { if (first == refusal::none) first = why; };
      while (depth) {
        const auto x = stack[--depth];
        listed[x] = false;
        const auto &q = queues_[x];
        for (unsigned i = 0; i != q.size; ++i) {
          const auto &w = q.waits[(q.head + i) % WaitLog];
          if (w.ticket >= frontier[x] || (x != rslot && !w.quiet)) continue;
          if (w.state != wait_state::resolved) { note(w.why); continue; }
          const auto *f = fence_of(w.fence);
          if (!f) { note(refusal::evicted); continue; }
          if (f->why != poison::none) { note(refusal::poisoned); continue; }
          const auto *s = queue_of(w.signaller);
          if (!s) { note(refusal::untracked); continue; }
          if (s->device != r->device) { note(refusal::cross_device); continue; }
          const auto sslot = slot_of(*s);
          if (w.signal_ticket <= frontier[sslot]) continue;
          frontier[sslot] = w.signal_ticket;
          // The wait that raised this frontier last, and the queue it is on.
          via_fence[sslot] = w.fence; via_queue[sslot] = x;
          if (!listed[sslot]) { stack[depth++] = sslot; listed[sslot] = true; }
        }
      }
      out.ordered = frontier[pslot] > p.ticket;
      if (!out.ordered) { out.why = first; return out; }
      // The chain that raised the producer queue's frontier, back to the
      // reader: each link's frontier is proven by the links after it. A walk
      // that loops (a frontier raised later through a queue it ordered) or
      // needs more fences than a verdict holds rests on every fence.
      unsigned used = 0;
      for (unsigned at = pslot, steps = 0; at != rslot; at = via_queue[at]) {
        if (++steps > Queues || used == out.fences.size()) { out.unbounded = true; break; }
        if (std::find(out.fences.begin(), out.fences.begin() + used, via_fence[at]) == out.fences.begin() + used)
          out.fences[used++] = via_fence[at];
      }
      return out;
    }
    // A read that relied on an ordered verdict: a poison of one of its
    // fences within the source age counts as late_conflict, and marks the
    // read's tag (the capture it read, 0 none) conflicted. A fence poisoned
    // (late_conflict) or forgotten (evicted_reads) since the verdict (a
    // signal, a reset or an eviction between the verdict and this call,
    // under another lock hold) conflicts it at once, as does evicting a read
    // that is still live (evicted_reads). Returns whether the read still
    // stands: one conflicted here must not be made (the caller has not
    // copied yet).
    bool rely(const verdict &value, std::uint64_t now, std::uint64_t tag = 0) {
      if (!value.ordered) return false;
      live_read read{value.fences, now + live_read_ms, tag, value.unbounded};
      for (const auto id : value.fences) {
        const auto *f = id ? fence_of(id) : nullptr;
        if (id && (!f || f->why != poison::none)) {
          conflict(read, f ? &counts::late_conflict : &counts::evicted_reads);
          return false;
        }
      }
      auto *oldest = &reads_[0];
      for (auto &entry : reads_)
        if (entry.expires < oldest->expires) oldest = &entry;
      if (oldest->expires > now) conflict(*oldest, &counts::evicted_reads);
      *oldest = read;
      return true;
    }
    // O1 (fail-safe): whether a read of tag relied on a fence that a late
    // conflict poisoned; its presentation copy is not held any further. The
    // newest LiveReads conflicted tags are kept, across a model reset.
    bool conflicted(std::uint64_t tag) const {
      return tag && std::find(conflicted_.begin(), conflicted_.end(), tag) != conflicted_.end();
    }
    counts snapshot() const {
      auto out = counts_;
      out.queues = 0; out.fences = 0;
      for (const auto &q : queues_) out.queues += q.id != 0;
      for (const auto &f : fences_) out.fences += f.id != 0;
      return out;
    }

  private:
    enum class wait_state : std::uint8_t { pending, resolved, refused };
    struct wait_entry {
      std::uint64_t fence{}, value{}, ticket{}, serial{}, signaller{}, signal_ticket{};
      bool quiet{};
      wait_state state{};
      refusal why{};
    };
    struct queue_record {
      std::uint64_t id{}, device{}, cookie{}, last_use{};
      std::array<wait_entry, WaitLog> waits{};
      unsigned head{}, size{};
    };
    struct signal_entry {
      std::uint64_t value{}, ticket{}, queue{};
    };
    struct pending_entry {
      unsigned queue{}, index{};
      std::uint64_t serial{};
    };
    struct fence_record {
      std::uint64_t id{}, device{}, baseline{}, seen_max{}, signaller{}, settle_value{}, max_signalled{}, evicted_max{}, last_use{};
      bool signalled{}, settled{}, evicted{};
      poison why{};
      std::array<signal_entry, SignalLog> ring{};
      unsigned head{}, size{};
      std::array<pending_entry, Pending> pending{};
      unsigned pending_count{};
    };
    struct live_read {
      std::array<std::uint64_t, 4> fences{};
      std::uint64_t expires{}, tag{};
      bool unbounded{};  // verdict::unbounded: every poison conflicts it.
    };

    static unsigned slot_of(const queue_record &q) { return static_cast<unsigned>((q.id & 0xff) - 1); }
    const queue_record *queue_of(std::uint64_t id) const {
      const auto slot = id & 0xff;
      return id && slot && slot <= Queues && queues_[slot - 1].id == id ? &queues_[slot - 1] : nullptr;
    }
    queue_record *queue_of(std::uint64_t id) {
      return const_cast<queue_record *>(static_cast<const model *>(this)->queue_of(id));
    }
    const fence_record *fence_of(std::uint64_t id) const {
      const auto slot = id & 0xffff;
      return id && slot && slot <= Fences && fences_[slot - 1].id == id ? &fences_[slot - 1] : nullptr;
    }
    fence_record *fence_of(std::uint64_t id) {
      return const_cast<fence_record *>(static_cast<const model *>(this)->fence_of(id));
    }
    wait_entry *pending_wait(const pending_entry &p) {
      if (p.queue >= Queues || !queues_[p.queue].id) return nullptr;
      auto &w = queues_[p.queue].waits[p.index];
      return w.serial == p.serial && w.state == wait_state::pending ? &w : nullptr;
    }
    void set_poison(fence_record &f, poison why, std::uint64_t now) {
      if (f.why != poison::none) return;
      f.why = why;
      ++counts_.poisoned[static_cast<unsigned>(why)];
      conflict_reads(f.id, now, &counts::late_conflict);
    }
    // A live read that relied on fence (0: none) or on every fence, counted
    // in cause (a counts member).
    void conflict_reads(std::uint64_t fence, std::uint64_t now, std::uint64_t counts::*cause) {
      for (auto &read : reads_)
        if (read.expires > now &&
            (read.unbounded || (fence && std::find(read.fences.begin(), read.fences.end(), fence) != read.fences.end())))
          conflict(read, cause);
    }
    void conflict(live_read &read, std::uint64_t counts::*cause) {
      ++(counts_.*cause);
      read.expires = 0;
      if (read.tag) conflicted_[conflicted_next_++ % LiveReads] = read.tag;
    }

    std::array<queue_record, Queues> queues_{};
    std::array<fence_record, Fences> fences_{};
    std::array<live_read, LiveReads> reads_{};
    std::array<std::uint64_t, LiveReads> conflicted_{};
    unsigned conflicted_next_{};
    counts counts_{};
    std::uint64_t serial_{}, clock_{};
  };
}
