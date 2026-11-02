// SPDX-License-Identifier: GPL-3.0-only
// Capture and provider selection; functions-only part of streamline_depth_capture.cpp.
// Types, namespace-scope state and test variables stay in the shell.

namespace sunshine_streamline::depth_capture {
  SUNSHINE_CAPTURE_INTERNAL {
    status diagnostic_admission(const slot &value, const queue_progress &progress) {
      if (value.diagnostic_released) return status::stale;
      if (value.invalid || (value.finished && !value.success)) return status::failed;
      if (!value.finished || !value.producer_submitted) return status::recorded;
      if (!progress.valid() || !value.producer_fence || value.retirement_unknown) return status::failed;
      if (!producer_recording_retired(value) || progress.completed < value.producer_fence) return status::submitted;
      return status::ready;
    }

    status submission_status(const slot &value, std::uint64_t native,
        std::uint64_t consumer_device, const queue_progress &progress) {
      if (!submitted_success(value, value.queue))
        return value.invalid || (value.finished && !value.success) ? status::failed : status::recorded;
      if (value.nomination_only) return value.pixel_failure;
      // Consumers temporarily transition the owned copy for their own copies.
      // Bind a slot to one consuming timeline instead of permitting unordered
      // state transitions on multiple queues. Actual retirement stays separate.
      if (value.consumer_queue && value.consumer_queue != native) return status::unsupported_queue;
      if (value.queue == native) return status::ready; // Existing queue ordering, no completion query required.
      if (!consumer_device || progress.device_identity != consumer_device) return status::unsupported_queue;
      if (!progress.valid()) return status::failed;
      // A completed but still-closed game recording may legally replay its
      // injected copy, and post-Execute replay detection would be too late.
      // Its copy-out cannot be written again; without one, only the
      // recording's Reset/destruction makes the copy immutable for a foreign
      // queue (immutable_for_other_queues).
      if (!immutable_for_other_queues(value)) return status::submitted;
      // The game may already order this producer after future consumer work.
      // Adding a reverse GPU Wait here can form a cycle, even after flushing
      // the consumer's immediate list. Poll completion, or read the game's own
      // queue order (U-C1), without adding an edge to the game's queue graph;
      // pending source metadata remains admissible.
      return readable_elsewhere(value, native, progress) ? status::ready : status::submitted;
    }

    // Nomination authority can survive pixel failure; it does not require completed_pixels.
    status nomination_status(const slot &value, std::uint64_t consumer_device) {
      if (!value.source_nominated || !value.id || !value.finished || !value.success || !value.producer_submitted)
        return value.invalid || (value.finished && !value.success) ? status::failed : status::recorded;
      // A failed snapshot fence/consumer concerns pixels, not the already
      // observed successful evaluation and real producer submission.
      if (value.nomination_invalid) return status::failed;
      return value.metadata.source && consumer_device && value.metadata.source->device_identity == consumer_device ?
        status::ready : status::unsupported_queue;
    }

    status local_texture_admission(const slot &value, std::uint64_t consumer, std::uint64_t device,
        const queue_progress &progress) {
      if (!value.diagnostic_only || !value.diagnostic_reusable) return status::unsupported_resource;
      if (value.diagnostic_released) return status::stale;
      if (!value.texture || value.texture->device_identity != device) return status::unsupported_queue;
      if (value.retirement_unknown) return status::failed;
      // Reuse the depth owner's existing same-queue ordering contract. A later
      // producer replay on this queue follows an already submitted consumer;
      // it still invalidates future acquisition and retains every GPU lease.
      return submission_status(value, consumer, device, progress);
    }

    void bind_pixel_consumer(slot &value, std::uint64_t consumer, status pixels) {
      // Source authority may be admitted even when this queue cannot read the
      // pixels. Such a nomination must never steal an existing read timeline.
      if (pixels == status::ready && !value.nomination_only &&
          (!value.consumer_queue || value.consumer_queue == consumer)) value.consumer_queue = consumer;
    }

    // Requires mutex. An active view of provider's namespace in epoch (the
    // namespace's current one, or at FG Off the options call's): its newest
    // evaluation younger than the source age or the frame the Present being
    // decided presents (frame_presenting), by its call or by its newest
    // copy's submission (view_presenting). The one activity rule: selection
    // counts these views (select_latest_capture), the pool keeps their heads
    // (selectable_head), FG Off reads another viewport's (retire_source) and
    // the acquisition's diagnostic counts them (active_view_count).
    bool view_active(std::uint64_t epoch, const evaluation_head &value, provider_kind provider, std::uint64_t now) {
      return value.epoch == epoch && value.sequence &&
        (sunshine_freshness::fresh_lt(value.tick, now, sunshine_freshness::source_age_ms) || view_presenting(value, provider));
    }

    capture_pick select_latest_capture(const queue_state &owner, std::uint64_t native,
        std::uint64_t present, provider_kind provider, std::uint64_t now, selection_policy policy) {
      capture_pick out;
      out.provider = provider;
      const auto &current = evaluations[provider_index(provider)];
      const auto active = [&](const evaluation_head &value) { return view_active(current.epoch, value, provider, now); };
      unsigned active_views = 0;
      for (const auto &value : current.views) active_views += active(value);
      // Two or more active views tie (S19). A confirmed FG scope's viewport
      // breaks the tie (owner ruling R5): the policy's tie_epoch and
      // tie_viewport, a required scope's or a silent one's
      // (select_frame_generation_scope keeps them). When exactly one active
      // view is that viewport's, the Present selects as if it were the one
      // active view: its newest nomination is the head, whose identity every
      // look below matches (the slots of that nomination, its stand-ins:
      // completed_snapshot and lineage_snapshot of its viewport), so no
      // other viewport's copy is picked. Without such a scope (FG off,
      // unknown, ambiguous or unavailable: exclude_unconfirmed_fg; or enabled
      // FG without a declared generated count, Dynamic mode with
      // numFramesToGenerate 0: fg_view::update names no tie viewport), or
      // with no single active view of its viewport, the tie stays
      // (ambiguous). One active view needs no tie-break and is selected as
      // before, whichever viewport FG names. Until 11-01 the tie always
      // stayed: two active SR viewports were mono, also with FG bound to one
      // of them.
      const evaluation_head *bound = nullptr;
      if (active_views > 1 && provider == provider_kind::streamline && policy.tie_epoch && policy.tie_epoch == current.epoch) {
        unsigned bound_views = 0;
        for (const auto &value : current.views)
          if (value.viewport == policy.tie_viewport && active(value)) { bound = &value; ++bound_views; }
        if (bound_views == 1) active_views = 1;
        else bound = nullptr;
      }
      // The head's identity: the namespace's newest nomination, or the bound
      // viewport's newest where it broke the tie.
      const auto epoch = bound ? bound->epoch : current.epoch;
      const auto sequence = bound ? bound->sequence : current.sequence;
      const auto source_id = bound ? bound->source_id : current.source_id;
      // The head's view, active or not (capture_pick::named).
      for (const auto &value : current.views)
        if (epoch && sequence && value.epoch == epoch && value.sequence == sequence &&
            value.source_id == source_id) out.named = &value;
      const auto is_latest = [&](const slot &value) {
        return value.id && !value.preservation_only && value.metadata.provider == provider && value.metadata.epoch == epoch &&
          value.metadata.sequence == sequence && value.metadata.source_id == source_id;
      };
      for (const auto &value : capture_pool.slots) if (is_latest(value) && (!out.latest || value.id > out.latest->id)) out.latest = &value;
      out.selection = out.latest ? selection_reason::no_admissible_capture : selection_reason::no_current_nomination;
      if (out.latest && out.latest->queue != native) out.progress = producer_progress(*out.latest);
      if (active_views != 1) {
        out.result = active_views ? status::ambiguous : status::stale;
        out.selection = active_views ? selection_reason::ambiguous_views : selection_reason::inactive_views;
        return out;
      }
      // The head's view, when it is the one active view or the bound one:
      // its viewport names the head's display lineage (select_capture).
      for (const auto &value : current.views)
        if (epoch && sequence && value.epoch == epoch && value.sequence == sequence &&
            value.source_id == source_id && active(value))
          out.head = &value;
      // The namespace has one nomination in progress, its newest's
      // (mark_pending): a bound viewport's nomination in progress after
      // another viewport's evaluation began has none, and reads as a missing
      // frame (select_capture: the same stand-in or hold, with no watermark
      // moved).
      const auto &pending = current.pending_nomination;
      if (pending.sequence && pending.epoch == epoch && pending.sequence == sequence &&
          pending.source_id == source_id &&
          (sunshine_freshness::fresh_lt(pending.tick, now, sunshine_freshness::source_age_ms) || frame_presenting(pending.present)) &&
          owner.provider_established && owner.provider == provider && owner.source_id == pending.source_id &&
          owner.viewport == pending.viewport && owner.last_epoch == pending.epoch && pending.sequence > owner.last_sequence)
        out.pending_nomination = &pending;
      // Intermediate metadata-only tickets do not finish a multi-tag attempt.
      // Its remaining candidate may still produce current pixels. Keep the
      // bounded pending identity until the single batch scope has resolved.
      if (out.pending_nomination) {
        // No candidate is final yet. Do not let a rejected intermediate slot
        // overwrite the clean pending identity with its failure diagnostics.
        out.latest = nullptr;
        out.result = status::recorded;
        out.selection = selection_reason::pending_nomination;
        return out;
      }
      for (auto &value : capture_pool.slots) {
        if (!is_latest(value) || value.id <= owner.admission_after) continue;
        const auto progress = value.queue == native ? queue_progress{} : producer_progress(value);
        if ((value.source_nominated ? nomination_status(value, owner.device_identity) :
            submission_status(value, native, owner.device_identity, progress)) != status::ready) continue;
        if (owner.present == present && owner.present_capture) {
          if (value.id == owner.present_capture) { out.value = &value; out.progress = progress; break; }
          continue;
        }
        // The consumed frame, also of an FG role FG Off retired (retire_source).
        if (owner.provider == provider && owner.last_epoch == value.metadata.epoch &&
            value.metadata.sequence <= owner.last_sequence) continue;
        if (!out.value || value.id > out.value->id) { out.value = &value; out.progress = progress; }
      }
      if (!out.value) {
        // The provider's last attempt (attempt_status): a newer evaluation's
        // begin resets it, and the record of the namespace's newest
        // nomination usually wrote it, but a record of an older evaluation
        // that finishes later on another thread may have. A bound viewport's
        // older nomination reports none (unavailable). Diagnostic only: every
        // value it holds is a missing frame alike (select_capture,
        // interrupts_admission).
        const bool newest = sequence == current.sequence && source_id == current.source_id;
        out.result = out.pending_nomination ? status::recorded :
          newest ? attempt_status[provider_index(provider)].load() : status::unavailable;
        if (out.latest) {
          if (out.latest->invalid || (out.latest->finished && !out.latest->success)) out.result = status::failed;
          else if (out.latest->producer_submitted) {
            out.result = submission_status(*out.latest, native, owner.device_identity, out.progress);
            // Already used by a prior presentation: readiness does not grant
            // permission to reuse an old sequence for the next presentation.
            if (out.result == status::ready) {
              out.result = status::submitted;
              out.selection = selection_reason::current_already_consumed;
            }
          }
          else out.result = status::recorded;
        }
        return out;
      }
      if (!sunshine_freshness::fresh_lt(out.value->metadata.tick, now, sunshine_freshness::source_age_ms) &&
          !frame_presenting(presented_by(*out.value))) {
        out.latest = out.value; out.value = nullptr; out.result = status::stale;
        return out;
      }
      out.result = submission_status(*out.value, native, owner.device_identity, out.progress);
      out.selection = selection_reason::current_capture;
      return out;
    }

    bool pending_frame(const capture_pick &chosen, const queue_state &owner) {
      if (chosen.pending_nomination) return true;
      const auto *value = chosen.latest;
      return value && pending_snapshot_core(*value, chosen.result) &&
        value->metadata.frame_generation_input && owner.provider_established &&
        owner.provider == value->metadata.provider && owner.source_id == value->metadata.source_id &&
        owner.last_epoch == value->metadata.epoch && value->metadata.sequence > owner.last_sequence;
    }

    acquisition_decision classify_acquisition(const capture_pick &chosen, const queue_state &owner,
        selection_policy policy) {
      acquisition_decision out;
      out.source_selected = chosen.value != nullptr;
      out.pending_frame = pending_frame(chosen, owner);
      const auto *value = chosen.value ? chosen.value : chosen.latest;
      if (value) {
        const auto &metadata = value->metadata;
        out.provider = metadata.provider; out.epoch = metadata.epoch;
        out.sequence = metadata.sequence; out.source_id = metadata.source_id;
        out.viewport = metadata.viewport; out.capture_id = value->id;
        out.source_valid = !value->invalid && value->failure == capture_failure::none &&
          (!value->finished || value->success);
        out.repeated_frame = !chosen.value && chosen.result != status::failed && chosen.result != status::stale &&
          chosen.result != status::ambiguous && value->finished && value->success &&
          !value->nomination_invalid && !value->invalid &&
          owner.last_epoch == metadata.epoch && owner.last_sequence == metadata.sequence;
      }
      else if (chosen.pending_nomination) {
        const auto &pending = *chosen.pending_nomination;
        out.provider = policy.require_frame_generation ? provider_kind::streamline : owner.provider;
        out.epoch = pending.epoch; out.sequence = pending.sequence;
        out.source_id = pending.source_id; out.viewport = pending.viewport;
        out.source_valid = true; // The selector already validated this bounded nomination.
      }
      // A head without a capture (refused before its slot existed), also
      // beside ambiguous or inactive views: its view names the source, so the
      // display holds only a copy of that source or its FG/SR lineage and
      // drops any other (source_changed). Until 10-28 such a pick named none
      // (epoch zero), and the display held a copy of the previous viewport or
      // NGX feature generation for up to the source age.
      else if (chosen.named) {
        const auto &head = *chosen.named;
        out.provider = chosen.provider; out.epoch = head.epoch; out.sequence = head.sequence;
        out.source_id = head.source_id; out.viewport = head.viewport;
      }
      else if (policy.require_frame_generation) {
        out.epoch = policy.epoch; out.viewport = policy.viewport;
        out.source_id = sunshine_scene_depth::fg_source_id(policy.viewport);
      }
      return out;
    }

    // The selection's outcome as depth_choice records it. Describes, never decides.
    depth_choice describe_choice(const capture_pick &chosen, const acquisition_decision &decision,
        const queue_state &owner, selection_policy policy, std::uint64_t present, std::uint64_t tick, bool pixel_ready) {
      depth_choice out;
      out.present = present; out.tick_ms = tick;
      out.decision = decision;
      out.result = chosen.result; out.selection = chosen.selection;
      const auto *source = chosen.value ? chosen.value : chosen.latest;
      if (source) out.failure = source->failure;
      const bool fg = source ? source->metadata.frame_generation_input : policy.require_frame_generation;
      // A head refused before its capture existed names its source too
      // (classify_acquisition, capture_pick::named).
      if (source || chosen.pending_nomination || chosen.named || policy.require_frame_generation)
        out.rung = fg ? sunshine_game3d::source_slot::fg_depth : sunshine_game3d::source_slot::scene_depth;
      out.pixel_ready = pixel_ready;
      out.fg_required = policy.require_frame_generation; out.fg_excluded = policy.exclude_unconfirmed_fg;
      out.owner_established = owner.provider_established;
      out.owner = owner.provider; out.owner_source = owner.source_id;
      return out;
    }

    // Keep one completed API snapshot while the CPU records its successor.
    // Completion and nomination advance independently for SR as well as FG.
    // Requiring the latest nomination to finish can starve a pipelined game.
    // Finished pixels stay valid through later metadata events (camera reset,
    // tag changes, observation loss); only source identity and age bound them.
    // consumer: the reading queue. A snapshot already submitted on it is
    // ordered before the read by queue order, exactly as a current capture is
    // (submission_status); another queue needs its copy-out or a retired
    // recording (immutable_for_other_queues) and a completed producer fence.
    slot *completed_snapshot(const input &current, std::uint64_t now, std::uint64_t consumer) {
      if (!valid_provider(current.provider) || !current.epoch) return nullptr;
      slot *best = nullptr;
      for (auto &value : capture_pool.slots) {
        const auto &metadata = value.metadata;
        if (!completed_pixels(value) ||
            metadata.provider != current.provider || metadata.frame_generation_input != current.frame_generation_input ||
            metadata.epoch != current.epoch || metadata.source_id != current.source_id || metadata.viewport != current.viewport ||
            metadata.sequence > current.sequence ||
            !(within_source_age(metadata.tick, now) || frame_presenting(presented_by(value)))) continue;
        if (!consumer || value.queue != consumer) {
          if (!immutable_for_other_queues(value)) continue;
          if (!readable_elsewhere(value, consumer, producer_progress(value))) continue;
        }
        if (!best || metadata.sequence > best->metadata.sequence ||
            (metadata.sequence == best->metadata.sequence && value.id > best->id)) best = &value;
      }
      return best;
    }

    // Two snapshots of one logical source read the same way: one storage
    // family through one view, of one resource kind. The allocation size and
    // the tag's active rectangle are not part of it: dynamic resolution gives
    // the same source a new allocation or extent every few frames (Assassin's
    // Creed Shadows with FG, 10-23: 2206x1240, 2442x1374, 2658x1494), and each
    // snapshot carries its own allocation and rectangle to its consumer.
    bool same_depth_storage(const input &a, const input &b) {
      if (!a.source || !b.source) return false;
      const auto left = a.source->desc.Format, right = b.source->desc.Format;
      return typeless(left) == typeless(right) && depth_view(left, a) == depth_view(right, b) &&
        a.resource.kind == b.resource.kind;
    }

    // head: the newest nomination the snapshot would stand in for. Its
    // storage is compared when it has a capture slot (compare_storage); a
    // head refused before its slot existed names no storage.
    selection_reason completed_fallback_reason(const queue_state &owner, std::uint64_t present,
        const input &head, const slot &completed, bool compare_storage) {
      if (completed.id <= owner.admission_after) return selection_reason::completed_before_gap;
      if (completed.metadata.sequence >= head.sequence) return selection_reason::completed_not_older;
      if (compare_storage && !same_depth_storage(completed.metadata, head)) return selection_reason::format_changed;
      // The consumed watermark belongs to the owner's provider; sequences of
      // another provider's namespace are not comparable with it.
      if (owner.provider == completed.metadata.provider && owner.last_epoch == completed.metadata.epoch &&
          completed.metadata.sequence <= owner.last_sequence)
        return selection_reason::completed_already_consumed;
      if (owner.present == present && owner.present_capture && owner.present_capture != completed.id)
        return selection_reason::presentation_already_selected;
      return selection_reason::completed_snapshot;
    }

    // A viewport's Streamline FG input (head) is pending, or a missing frame
    // (capture_pick::missing_frame: a refused DLSS-G attempt), and none of its
    // snapshots completed: FG On's first frames, or DLSS-G's compute copy
    // still in flight at its own Present in a GPU-bound game. The viewport's
    // ordinary SR/RR source of the same epoch is one display lineage with it
    // (depth_hold.h, lineage_successor), and its newest snapshot that
    // this queue can read now and has not consumed (the queue's watermark,
    // shared by the viewport's sources) stands in, within the source age: at
    // FG On that is the frame's own SR copy. Until 10-27 only the display's
    // retained copy was held there (FG_copy_pending), older by the frames
    // without depth before FG On, and mono once that passed the source age
    // while the frame's own SR copy was readable (the session tests' review
    // of seed c28965dca8d7c76c). The FG input takes the viewport at its first
    // readable frame, as before; a silent scope's held FG copy is still held
    // over it (frame_generation_holds), and a repeated effect of one
    // presentation keeps its one tuple.
    // A finished, valid SR/RR snapshot of the FG input fg's display lineage:
    // the viewport's ordinary Streamline source of the same epoch, no newer
    // than fg, within the source age or presented by this Present. Its
    // admission and readability are the caller's (lineage_snapshot,
    // lineage_fallback).
    bool lineage_candidate(const slot &value, const input &fg, std::uint64_t now) {
      const auto &metadata = value.metadata;
      return completed_pixels(value) && metadata.provider == provider_kind::streamline && !metadata.frame_generation_input &&
        metadata.epoch == fg.epoch && metadata.viewport == fg.viewport && metadata.sequence <= fg.sequence &&
        (within_source_age(metadata.tick, now) || frame_presenting(presented_by(value)));
    }

    bool lineage_head(const input &fg) {
      return fg.provider == provider_kind::streamline && fg.frame_generation_input && fg.epoch;
    }

    slot *lineage_snapshot(const queue_state &owner, std::uint64_t native, std::uint64_t present, const input &head,
        std::uint64_t now) {
      const auto &fg = head;
      if (!lineage_head(fg) || (owner.present == present && owner.present_capture)) return nullptr;
      slot *best = nullptr;
      for (auto &value : capture_pool.slots) {
        const auto &metadata = value.metadata;
        if (!lineage_candidate(value, fg, now) || value.id <= owner.admission_after) continue;
        if (owner.provider == provider_kind::streamline && owner.last_epoch == metadata.epoch &&
            metadata.sequence <= owner.last_sequence) continue;
        if (best && (metadata.sequence < best->metadata.sequence ||
            (metadata.sequence == best->metadata.sequence && value.id < best->id))) continue;
        const auto progress = value.queue == native ? queue_progress{} : producer_progress(value);
        if (submission_status(value, native, owner.device_identity, progress) != status::ready) continue;
        best = &value;
      }
      return best;
    }

    // The capture pool's view of lineage_snapshot, with no queue's state
    // (completed_snapshot's consumer 0): the newest lineage candidate of the
    // FG input fg that any queue can read once its work completed. Every
    // stand-in lineage_snapshot can still pick is this one: a newer candidate
    // is unfinished GPU work or an open recording, which is never
    // reclaimable, and an older one is consumed, or below the gap, wherever
    // this one is.
    const slot *lineage_fallback(const input &fg, std::uint64_t now) {
      if (!lineage_head(fg)) return nullptr;
      const slot *best = nullptr;
      for (const auto &value : capture_pool.slots) {
        if (!lineage_candidate(value, fg, now) || !immutable_for_other_queues(value) ||
            !readable_elsewhere(value, 0, producer_progress(value))) continue;
        if (!best || value.metadata.sequence > best->metadata.sequence ||
            (value.metadata.sequence == best->metadata.sequence && value.id > best->id)) best = &value;
      }
      return best;
    }

    // A head refused before its capture slot existed: the identity its view
    // names. A Streamline view of a viewport's FG key is its FG input.
    input head_identity(const evaluation_head &head, provider_kind provider) {
      input out;
      out.provider = provider; out.epoch = head.epoch; out.sequence = head.sequence;
      out.source_id = head.source_id; out.viewport = head.viewport; out.tick = head.tick;
      out.frame_generation_input = provider == provider_kind::streamline &&
        sunshine_scene_depth::is_fg_source(head.source_id, head.viewport);
      return out;
    }

    capture_pick select_capture(const queue_state &owner, std::uint64_t native,
        std::uint64_t present, provider_kind provider, std::uint64_t now, selection_policy policy) {
      auto chosen = select_latest_capture(owner, native, present, provider, now, policy);
      const auto *latest = chosen.latest;
      // This live check is source age or this frame being presented; unlike
      // select_provider's live, it does not interpret status::stale or a pending nomination.
      const bool live = latest && (within_source_age(latest->metadata.tick, now) || frame_presenting(presented_by(*latest)));
      // A live source whose newest snapshot is pending may use its own newest
      // completed one; completed_snapshot matches its identity. Ownership does
      // not gate this: a pipelined source is otherwise never readable, so it
      // could never (re)establish itself after a reset, a source change or
      // another provider's expired capture.
      const bool pending = live && pending_snapshot_core(*latest, chosen.result);
      // A missing frame takes the same path: the one active view's fresh head
      // failed, was refused (before its capture slot existed too), is invalid,
      // or names its source without pixels this queue can read (a nomination-
      // only slot's pixel failure). That is no evidence against the finished
      // pixels of its display lineage, so its lineage's newest legal copy
      // stands in under the pending path's checks (identity, finished,
      // readable here, within the source age, unconsumed), and it moves no
      // admission watermark (interrupts_admission). Until 10-28 it revoked
      // depth: acquire's watermark made the newer copy completed_before_gap
      // while the display held an older one, and the Present went mono once
      // that passed the source age with a readable copy (Assassin's Creed
      // Shadows' refused SR inputs, 10-25). FG Off is one too: retire_source
      // passes the retired FG role's newest nomination to the viewport's
      // ordinary source as such a head, so the lineage's newest SR copy
      // stands in at once (10-28). Ambiguous or inactive views, a head past
      // the source age that this Present does not present, the FG scope's
      // policy picks (select_provider), an NGX feature's release
      // (retire_source) and expire_association still revoke.
      chosen.missing_frame = !pending && chosen.head && !chosen.pending_nomination && (!latest || live) &&
        chosen.result != status::ready && chosen.result != status::stale && chosen.result != status::ambiguous;
      // A nomination in progress (pending_nomination: the owner's exact
      // source, newer than its last shown frame) takes the same path too: it
      // ends recorded (the pending path above) or refused (a missing frame),
      // and either picks this same copy, so a Present meanwhile shows it now
      // rather than one Present late. It keeps its pending_nomination, which
      // still names the owner's live source (select_provider), and moves no
      // watermark. Until 10-28 it held the retained copy, one frame behind
      // the readable copy (Assassin's Creed Shadows' FG Presents racing its SR
      // evaluates), because a refusal then moved acquire's gap watermark.
      // Without a stand-in it keeps its pending result.
      const bool in_progress = chosen.pending_nomination != nullptr;
      if (!pending && !chosen.missing_frame && !in_progress) return chosen;
      const auto identity = latest ? latest->metadata :
        head_identity(in_progress ? *chosen.pending_nomination : *chosen.head, provider);
      // A missing frame or a nomination in progress without a stand-in keeps
      // its own result and reason.
      auto *completed = completed_snapshot(identity, now, native);
      // The pending head's own copy completed between select_latest_capture's
      // look and this one (each reads its producer fence anew): the frame's
      // newest copy is readable now, so it is decided again, as the current
      // capture. Until 10-28 it read completed_not_older here and the Present
      // held an older copy beside it: at FG Off, the FG copy beside the
      // frame's own readable tag copy, while the frame's SR copy on the
      // game's queue completed during that Present's selection (the 10-28
      // review's J2, once in a few fixed-set runs).
      if (pending && completed && completed == latest) {
        auto again = select_latest_capture(owner, native, present, provider, now, policy);
        if (again.value == completed && again.result == status::ready) return again;
      }
      if (!completed) {
        if (pending) chosen.selection = selection_reason::no_completed_snapshot;
        if (auto *standin = lineage_snapshot(owner, native, present, identity, now)) {
          chosen.value = standin;
          chosen.progress = standin->queue == native ? queue_progress{} : producer_progress(*standin);
          chosen.result = status::ready;
          chosen.selection = selection_reason::lineage_snapshot;
        }
        return chosen;
      }
      // An FG head whose own completed copy is refused (published before a
      // gap, consumed, of another storage) or not readable here: its
      // lineage's newest legal SR copy stands in, as where it has none
      // (10-28; until then the Present held the display's older copy beside
      // that readable SR copy: the extra shard's seed d0dd7845ac141ad2 at
      // frame 113, under the lineage judge).
      const auto lineage_stand_in = [&] {
        auto *standin = lineage_snapshot(owner, native, present, identity, now);
        if (!standin) return false;
        chosen.value = standin;
        chosen.progress = standin->queue == native ? queue_progress{} : producer_progress(*standin);
        chosen.result = status::ready;
        chosen.selection = selection_reason::lineage_snapshot;
        return true;
      };
      const auto reason = completed_fallback_reason(owner, present, identity, *completed, latest != nullptr);
      if (reason != selection_reason::completed_snapshot) {
        if (pending) chosen.selection = reason;
        if (lineage_stand_in()) return chosen;
        if (pending && reason == selection_reason::completed_already_consumed && completed->metadata.sequence == owner.last_sequence)
          chosen.consumed_completed_capture = completed->id;
        return chosen;
      }
      const auto progress = producer_progress(*completed);
      if (submission_status(*completed, native, owner.device_identity, progress) != status::ready) {
        if (pending) chosen.selection = selection_reason::completed_not_readable;
        lineage_stand_in();
        return chosen;
      }
      // An FG copy of a frame before FG Off retired its role (its sequence
      // at or below the retiring options call's serial), recorded after the
      // Off by a race past the retirement (frame_generation_retired), then
      // FG On within the source age: the newer frame of the lineage shows,
      // whichever path names it (until 10-28 that copy stood in for FG On's
      // pending head before the lineage's newer readable SR copy of a later
      // frame was tried, the 10-28 review's P5 and J8). Any other completed
      // FG copy keeps its precedence over the SR stand-in, as the scope's
      // copies own their viewport (a live compute scope's copy in flight at
      // 4K holds the scope's previous copy, not SR).
      const auto *retired = completed->metadata.frame_generation_input ?
        fg_retirement_of(completed->metadata.epoch, completed->metadata.viewport) : nullptr;
      if (auto *standin = retired && completed->metadata.sequence <= retired->serial ?
            lineage_snapshot(owner, native, present, identity, now) : nullptr;
          standin && standin->metadata.epoch == completed->metadata.epoch &&
          standin->metadata.sequence > completed->metadata.sequence) {
        chosen.value = standin;
        chosen.progress = standin->queue == native ? queue_progress{} : producer_progress(*standin);
        chosen.result = status::ready;
        chosen.selection = selection_reason::lineage_snapshot;
        return chosen;
      }
      chosen.value = completed;
      chosen.progress = progress;
      chosen.result = status::ready;
      chosen.selection = selection_reason::completed_snapshot;
      return chosen;
    }

    // A genuine source interruption revokes the queue's snapshots from before
    // it (admission_after). A pending, repeated or missing frame
    // (capture_pick::missing_frame; FG Off's retired FG head too, which
    // retire_source passes to the viewport's ordinary source) does not, nor
    // does a newest copy that this Present still presents
    // (frame_presenting): its frame's copies stay admissible.
    bool interrupts_admission(const queue_state &owner, const capture_pick &chosen) {
      return owner.provider_established && !chosen.missing_frame &&
        chosen.result != status::ready && chosen.result != status::recorded && chosen.result != status::submitted &&
        !(chosen.result == status::stale && chosen.latest && frame_presenting(presented_by(*chosen.latest)));
    }

    capture_pick select_provider(const queue_state &owner, std::uint64_t native,
        std::uint64_t present, std::uint64_t now, selection_policy policy) {
      const auto fg_head = [](const evaluation_head *head) {
        return head && sunshine_scene_depth::is_fg_source(head->source_id, head->viewport);
      };
      const auto select_sl = [&] {
        auto chosen = select_capture(owner, native, present, provider_kind::streamline, now, policy);
        const auto *source = chosen.value ? chosen.value : chosen.latest;
        // Without a capture, a nomination in progress or a head refused before
        // its slot existed names the source: an FG input's is excluded as one.
        if (policy.exclude_unconfirmed_fg &&
            (source ? source->metadata.frame_generation_input :
              fg_head(chosen.pending_nomination ? chosen.pending_nomination : chosen.head))) {
          chosen = {};
          chosen.result = status::stale;
          chosen.selection = selection_reason::fg_scope_missing;
          return chosen;
        }
        return chosen;
      };
      if (policy.require_frame_generation) {
        if (!policy.epoch) { capture_pick missing; missing.selection = selection_reason::fg_scope_missing; return missing; }
        // FG owns the source choice before pixels are ready. Do not silently
        // substitute SR depth while its matching camera belongs to SL FG.
        auto fg = select_sl();
        const auto fg_source = sunshine_scene_depth::fg_source_id(policy.viewport);
        const auto matches = [&](const sunshine_scene_depth::frame &value) {
          return value.provider == provider_kind::streamline && value.frame_generation_input &&
            value.epoch == policy.epoch && value.viewport == policy.viewport &&
            value.source_id == fg_source;
        };
        // The lineage's ordinary snapshot stands in for its pending input
        // (lineage_snapshot), of this scope's viewport and epoch.
        const bool standin = fg.value && fg.selection == selection_reason::lineage_snapshot &&
          fg.value->metadata.epoch == policy.epoch && fg.value->metadata.viewport == policy.viewport;
        // Without a capture, a nomination in progress or a head refused before
        // its slot existed names the source it must match.
        const auto *named = fg.latest ? nullptr : fg.pending_nomination ? fg.pending_nomination : fg.head;
        if ((fg.value && !standin && !matches(fg.value->metadata)) || (fg.latest && !matches(fg.latest->metadata)) ||
            (named && (named->epoch != policy.epoch || named->viewport != policy.viewport ||
              named->source_id != fg_source))) {
          capture_pick mismatch; mismatch.selection = selection_reason::fg_scope_mismatch; return mismatch;
        }
        return fg;
      }
      // Repeated effects in one presentation keep one immutable provider tuple.
      if (owner.provider_established && owner.present == present && owner.present_capture)
        return owner.provider == provider_kind::streamline ? select_sl() : select_capture(owner, native, present, owner.provider, now, policy);
      auto sl = select_sl();
      auto ngx = select_capture(owner, native, present, provider_kind::ngx, now, policy);
      // Ownership never moves to depth older than the owner's last selected
      // copy (present_capture). Capture ids follow publication order across
      // both providers, while GPU readability need not: the other provider's
      // copy of an earlier frame can turn readable after the owner's newer
      // one was shown. Such a pick is no readable source, only a live one,
      // so the owner keeps selection and the display holds its copy until
      // the other provider's next copy is readable (one frame at most). Until
      // 10-28 a failing owner handed over to it, older depth after newer, and
      // the owner's shown copy then stood in again as a missing frame's,
      // flipping ownership back (the consumed watermark restarts at a
      // handover and cannot compare the two namespaces).
      const auto older_than_shown = [&](capture_pick &value, provider_kind provider) {
        if (!owner.provider_established || owner.provider == provider || !owner.present_capture || !value.value ||
            value.value->id > owner.present_capture) return;
        value.value = nullptr;
        value.result = status::recorded;
        value.selection = selection_reason::completed_older_than_shown;
      };
      older_than_shown(sl, provider_kind::streamline);
      older_than_shown(ngx, provider_kind::ngx);
      const auto readable = [](const capture_pick &value) { return value.value && value.result == status::ready; };
      // Expired: no active view, or a newest nomination older than the source
      // age. A pick without any slot (an attempt rejected before its capture
      // existed) is not expired while its view is active.
      const auto expired = [&](const capture_pick &value) {
        return value.result == status::stale || (value.latest && !within_source_age(value.latest->metadata.tick, now) &&
          !frame_presenting(presented_by(*value.latest)));
      };
      // A source is live while its newest nomination is within the source age.
      // Readable implies live; liveness never extends that age.
      // Unlike select_capture's live, this includes pending_nomination and
      // uses expired, which also refuses status::stale. Keep these two rules distinct.
      const auto live = [&](const capture_pick &value) {
        return value.pending_nomination || (value.latest && !expired(value));
      };
      // A live source whose newest valid snapshot is still on its way, so its
      // completed snapshot can become readable. A failed or refused head
      // (capture_pick::missing_frame) moves no gap watermark either, but has
      // no pixels of its own to wait for: its lineage's completed copy, when
      // one is legal, already stood in (select_capture).
      const auto pending = [&](const capture_pick &value) {
        const auto *latest = value.latest;
        return value.pending_nomination || (live(value) && pending_snapshot_core(*latest, value.result));
      };
      // Ownership only arbitrates between live sources. Only a delivering owner
      // (one frame copied within the source age) keeps its own live source:
      // while that source is readable, or while its real successor is pending,
      // so a short pipeline delay does not alternate providers every frame. Its
      // already copied frame is not a successor. This preserves selection only;
      // display-cache rules still govern old pixels. Failed, unsupported,
      // missing, replaced, expired or undelivered owner evidence never holds
      // against a readable source; a failed head's completed copy that stood
      // in (select_capture) is the owner's readable source.
      const auto protected_owner = [&](const capture_pick &value) {
        const auto *source = value.value ? value.value : value.latest;
        if (source) {
          if (source->metadata.epoch != owner.nominated_epoch || source->metadata.source_id != owner.source_id ||
              source->metadata.viewport != owner.viewport) return false;
        } else if (!value.pending_nomination) return false; // Already bound to the owner's exact source.
        if (!within_source_age(owner.last_source_tick, now)) return false;
        if (readable(value) || value.pending_nomination) return true;
        return pending(value) && (value.latest->metadata.epoch != owner.last_epoch ||
          value.latest->metadata.sequence > owner.last_sequence);
      };
      const auto *own = owner.provider_established ? owner.provider == provider_kind::streamline ? &sl : &ngx : nullptr;
      if (own && protected_owner(*own)) return *own;
      // No protected owner: readable SL is preferred over readable NGX. This
      // includes an established SL owner whose readable frames are never copied
      // (failed display preparation or copy): it is not delivering, yet keeps
      // that preference until its source is revoked or stops being readable.
      if (readable(sl)) return sl;
      if (readable(ngx)) return ngx;
      // Neither provider is readable. A delivering owner keeps any pick that
      // has not expired: after one rejected, failed or ambiguous attempt, also
      // one rejected before its capture slot existed, the display may hold its
      // last copied frame, and the other provider's pixel-less pending capture,
      // such as a nested NGX evaluation still in flight, takes no authority. A
      // persistently failing owner stops delivering within the source age and
      // so cannot lock out a pending source. Otherwise a pending live pick ranks
      // above a failing live one, whose lineage had no completed copy to stand
      // in (a failing pick no longer marks a gap, interrupts_admission, but a
      // pending one may still become readable), and any live pick ranks above
      // an expired one. An owner that has stopped
      // delivering keeps its provider only while the other provider does not
      // rank higher.
      const auto rank = [&](const capture_pick &value) { return pending(value) ? 2 : live(value) ? 1 : 0; };
      if (own) {
        if (!expired(*own) && within_source_age(owner.last_source_tick, now)) return *own;
        const auto &other = own == &sl ? ngx : sl;
        return rank(other) > rank(*own) ? other : *own;
      }
      if (rank(sl) != rank(ngx)) return rank(sl) > rank(ngx) ? sl : ngx;
      // Preserve pending/failure diagnostics when neither provider is ready.
      // Pending SL cannot block a subsequently readable NGX source.
      if (sl.value || (!ngx.value && (sl.latest || !evaluations[provider_index(provider_kind::ngx)].sequence))) return sl;
      return ngx;
    }

    void establish_provider(queue_state &owner, const slot &value, std::uint64_t present) {
      // Ownership moving to the other provider ends the previous provider's
      // presentation: every copy recorded before the new owner's (its own
      // older ones too) is historical. The consumed watermark below restarts
      // and cannot compare the two namespaces' sequences, so without this the
      // previous provider's copy already shown, or one older than the new
      // owner's, could be selected again and flip ownership back (a missing
      // frame no longer revokes them, select_capture). Copies recorded after
      // it, of either provider, stay admissible. The previous owner's last
      // selected copy is historical however the new one was chosen (also a
      // required FG scope's pick, which select_provider's older_than_shown
      // does not judge). So is it after FG Off retired the owner's FG role
      // (retire_source keeps its present_capture): an NGX copy taking the
      // queue then makes the viewport's SR copies before it historical, as
      // any handover does, so FG On's lineage stand-in never steps back to
      // one of them.
      if ((owner.provider_established || owner.present_capture) && owner.provider != value.metadata.provider && value.id)
        owner.admission_after = std::max({owner.admission_after, value.id - 1, owner.present_capture});
      if (!owner.provider_established || owner.provider != value.metadata.provider ||
          owner.source_id != value.metadata.source_id || owner.nominated_epoch != value.metadata.epoch)
        owner.last_epoch = owner.last_sequence = owner.last_source_tick = 0;
      owner.provider_established = true;
      owner.provider = value.metadata.provider;
      owner.source_id = value.metadata.source_id;
      owner.viewport = value.metadata.viewport;
      owner.present = present; owner.present_capture = value.id;
      owner.nominated_epoch = value.metadata.epoch;
    }

    // After the association's silence (freshness.h, association_ms).
    // Requires mutex.
    void expire_association(queue_state &owner, std::uint64_t now) {
      if (!owner.provider_established || !valid_provider(owner.provider)) return;
      const auto &current = evaluations[provider_index(owner.provider)];
      // The owner's own evaluations: its source and epoch, so the malformed-
      // input marker (epoch 0, source 0) never keeps a Streamline SR owner
      // (source 0) associated.
      const auto within = [&](std::uint64_t tick) {
        return tick && sunshine_freshness::fresh_le(tick, now, sunshine_freshness::association_ms);
      };
      const auto recent = [&](const evaluation_head &head) {
        return head.source_id == owner.source_id && head.epoch == owner.nominated_epoch && within(head.tick) &&
          (!head.rejected_since || within(head.rejected_since));
      };
      if (recent(current.pending_nomination) || std::any_of(current.views.begin(), current.views.end(), recent)) return;
      owner.provider_established = false;
      owner.source_id = owner.last_epoch = owner.last_sequence = owner.last_source_tick = owner.nominated_epoch = owner.present_capture = 0;
      // Snapshots from before the silence never re-establish it.
      owner.admission_after = std::max(owner.admission_after, serial.load(std::memory_order_relaxed));
    }
  }

  bool provider_active(std::uint64_t native) {
    if (!requested.load() || !native) return false;
    std::lock_guard lock(mutex);
    auto *owner = queue(native);
    if (!requested.load() || !owner || owner->retiring) return false;
    expire_association(*owner, GetTickCount64());
    return owner->provider_established;
  }

  bool evaluation_live(std::uint64_t now_ms, std::uint64_t window_ms) {
    if (!requested.load()) return false;
    std::lock_guard lock(mutex);
    // Any real evaluation: Streamline SR uses source 0 by design, and only the
    // malformed-input marker begins one with epoch 0.
    const auto recent = [&](const evaluation_head &head) {
      return head.epoch && head.tick && sunshine_freshness::fresh_le(head.tick, now_ms, window_ms);
    };
    for (const auto &space : evaluations) {
      if (recent(space.pending_nomination)) return true;
      for (const auto &view : space.views) if (recent(view)) return true;
    }
    return false;
  }

  bool provider_identity(std::uint64_t native, provider_kind &provider, std::uint64_t &source_id) {
    source_id = 0;
    if (!requested.load() || !native) return false;
    std::lock_guard lock(mutex);
    auto *owner = queue(native);
    if (!owner || owner->retiring) return false;
    expire_association(*owner, GetTickCount64());
    if (!owner->provider_established) return false;
    provider = owner->provider;
    source_id = owner->source_id;
    return true;
  }

  bool acquire(std::uint64_t native, std::uint64_t present, packet &out, capture_diagnostic *diagnostic,
      selection_policy policy) {
    acquisition_decision ignored;
    return acquire(native, present, out, ignored, diagnostic, policy);
  }

  bool acquire(std::uint64_t native, std::uint64_t present, packet &out, acquisition_decision &decision,
      capture_diagnostic *diagnostic, selection_policy policy) {
    out = {};
    decision = {};
    if (policy.require_frame_generation) {
      decision.epoch = policy.epoch; decision.viewport = policy.viewport;
      decision.source_id = sunshine_scene_depth::fg_source_id(policy.viewport);
    }
    if (diagnostic) {
      *diagnostic = {}; diagnostic->requested_queue = native;
    }
    queue_progress progress;
    bool recording_retired = false;
    const auto result = [diagnostic, &decision, &progress, &recording_retired, &policy](status why,
        const slot *value = nullptr) {
      if (diagnostic) {
        diagnostic->result = why;
        diagnostic->frame_generation_input = value ? value->metadata.frame_generation_input :
          policy.require_frame_generation;
        diagnostic->capture_id = decision.capture_id; diagnostic->sequence = decision.sequence;
        diagnostic->epoch = decision.epoch; diagnostic->source_id = decision.source_id;
        diagnostic->viewport = decision.viewport; diagnostic->provider = decision.provider;
        diagnostic->repeated_frame = decision.repeated_frame; diagnostic->pending_frame = decision.pending_frame;
        if (value) {
          diagnostic->failure = value->failure;
          diagnostic->source_tick_ms = value->metadata.tick;
          diagnostic->command = value->command;
          diagnostic->queue = value->queue; diagnostic->producer_fence = value->producer_fence;
          diagnostic->producer_completed = progress.completed;
          diagnostic->producer_completion_valid = progress.valid();
          diagnostic->producer_recording_retired = recording_retired;
          for (const auto &point : value->retirements)
            if (point.queue == diagnostic->consumer_queue) diagnostic->retire_fence = point.value;
          diagnostic->finished = value->finished; diagnostic->success = value->success;
          diagnostic->submitted = value->producer_submitted; diagnostic->invalid = value->invalid;
        }
      }
      reason = why;
      return why == status::ready;
    };
    if (!requested.load() || !native || !native_observer::queue_ready(native)) return result(status::inactive);
    std::lock_guard lock(mutex);
    auto *owner = queue(native);
    if (!owner || owner->retiring) return result(status::unsupported_queue);
    native = reinterpret_cast<std::uint64_t>(owner->queue.p);
    if (diagnostic) diagnostic->consumer_queue = native;
    const auto selection_tick = GetTickCount64();
    const auto chosen = select_provider(*owner, native, present, selection_tick, policy);
    progress = chosen.progress;
    const auto *selected = chosen.value ? chosen.value : chosen.latest;
    if (selected) {
      // Diagnostic completion facts only; never changes admission or GPU ordering.
      if (selected->queue == native) progress = producer_progress(*selected);
      recording_retired = producer_recording_retired(*selected);
    }
    decision = classify_acquisition(chosen, *owner, policy);
    if (diagnostic) {
      diagnostic->selection = chosen.selection;
      diagnostic->consumed_completed_capture = chosen.consumed_completed_capture;
      diagnostic->selection_tick_ms = selection_tick;
      if (selected || policy.require_frame_generation || owner->provider_established) {
        const auto provider = selected ? selected->metadata.provider :
          policy.require_frame_generation ? provider_kind::streamline : owner->provider;
        const auto &current = evaluations[provider_index(provider)];
        for (const auto &view : current.views) if (view.epoch == current.epoch && view.sequence) {
          diagnostic->newest_view_tick_ms = std::max(diagnostic->newest_view_tick_ms, view.tick);
          // As select_latest_capture counts them (frame_presenting too),
          // before a confirmed FG scope's viewport breaks their tie.
          if (view_active(current.epoch, view, provider, selection_tick)) ++diagnostic->active_view_count;
        }
      }
    }
    // A genuine source interruption revokes snapshots from before the gap;
    // pending, repeated and missing frames do not (interrupts_admission).
    if (interrupts_admission(*owner, chosen)) owner->admission_after = serial.load(std::memory_order_relaxed);
    // The display may hold the newest evaluated frame's own copy at this
    // Present, whatever its age (depth_hold.h, own_frame).
    decision.presenting_floor = presenting_floor();
    // A head refused before its capture existed names its sequence too, so a
    // missing frame's stand-in reads as sequence < newest_sequence.
    if (diagnostic) diagnostic->newest_sequence = chosen.latest ? chosen.latest->metadata.sequence :
      chosen.pending_nomination ? chosen.pending_nomination->sequence : chosen.head ? chosen.head->sequence : 0;
    if (diagnostic && chosen.pending_nomination) {
      const auto &pending = *chosen.pending_nomination;
      diagnostic->source_tick_ms = pending.tick;
    }
    if (!chosen.value) {
      owner->choice = describe_choice(chosen, decision, *owner, policy, present, selection_tick, false);
      owner->chosen = true; read_on(owner->cookie, false);
      return result(chosen.result, chosen.latest);
    }
    auto *best = chosen.value;
    // Borrowed CPU ownership alone does not imply a GPU consumer. Only a
    // successful pre-read mark_consumer establishes that retirement obligation.
    bind_pixel_consumer(*best, native, chosen.result);
    establish_provider(*owner, *best, present);
    describe_packet(*best, native, out);
    out.pixel_ready = out.pixel_ready && chosen.result == status::ready;
    owner->choice = describe_choice(chosen, decision, *owner, policy, present, selection_tick, out.pixel_ready);
    owner->chosen = true; read_on(owner->cookie, out.pixel_ready);
    result(chosen.result, best);
    return true; // Source authority and current pixels are separate outcomes.
  }

  bool choice(std::uint64_t native, depth_choice &out) {
    out = {};
    if (!native) return false;
    std::lock_guard lock(mutex);
    const auto *owner = queue(native);
    if (!owner || !owner->chosen) return false;
    out = owner->choice;
    return true;
  }

  void complete_frame(const packet &value, std::uint64_t present) {
    std::lock_guard lock(mutex);
    auto *owner = queue(value.queue);
    // The presentation this Present selected and its consumer copied. FG Off
    // on the game's thread between that selection and this copy (DLSS-G
    // presenting on its own) retires the owner's FG role but keeps that
    // presentation tuple (retire_source), so the copied frame is still the
    // consumed one: until 10-28 it was not, and an SR copy of an earlier
    // frame published after it could show at the next Present, older depth
    // after newer (the 10-28 review's P4). Every other end of an owner clears
    // present_capture.
    if (!owner || owner->present != present || owner->present_capture != value.capture_id ||
        owner->provider != value.metadata.provider ||
        (owner->provider_established && owner->nominated_epoch != value.metadata.epoch)) return;
    owner->last_epoch = value.metadata.epoch;
    owner->last_sequence = value.metadata.sequence;
    owner->last_source_tick = value.metadata.tick;
    owner->shown_capture = value.capture_id;
  }
}
