// SPDX-License-Identifier: GPL-3.0-only
// Provider status gathering, log rendering and diagnostics.
// Same TU as streamline_depth_provider.cpp; namespace-scope state and types stay there.
  namespace {
    void emit_fg_switch(fg_switch_line &line) {
      char text[512]{};
      format_fg_switch(text, sizeof(text), line);
      sunshine_log::message(reshade::log::level::info, text);
      line.active = false;
    }

    // A runtime's teardown hands its FG mode and its FG switch line to the
    // runtime whose first Present comes next, which adopts both
    // (track_fg_switch; a resize tears down before the new runtime exists,
    // a recreation may create the new swapchain's runtime before it
    // releases the old one): a reset with FG unchanged logs nothing, and a switch
    // the game applied across one, in either order, logs one line, counted
    // to the first fresh copy or a second after the change over the Presents
    // of both runtimes. A runtime torn down before its first Present (a
    // second ResizeBuffers with no Present between, as at a display-settings
    // Apply) takes over the handover first and passes it on unchanged: it saw
    // no Present, so its mode is only the options' and its line is empty.
    // A handed line no runtime adopted before another runtime's teardown
    // (one that already presented: another swapchain) is written there,
    // counted to that point and marked ended=runtime_reset. Until 11-01 the
    // teardown wrote the line so marked, so no line counted the Presents to
    // fresh depth after the reset and the report read the marked line as no
    // fresh depth within a second (the fix review's X1, F100-b); and the new
    // runtime started from the options' mode, so an options call just
    // before the reset logged no switch at all (F100-a). Until the second
    // fix review a runtime that never presented handed over its own empty
    // line and seeded mode, which wrote the earlier runtime's line at the
    // second of two resets and lost an FG On made just before them. Before
    // F100 the line was dropped and the new runtime started at mode 0.
    void hand_over_fg_switch(state &data) {
      if (!data.fg_switch_first_present) {
        data.fg_switch_first_present = true;
        adopt_fg_switch(data);
      }
      AcquireSRWLockExclusive(&fg_switch_handover_lock);
      auto unadopted = std::exchange(fg_switch_handed, {true, data.fg_mode, data.fg_switch});
      ReleaseSRWLockExclusive(&fg_switch_handover_lock);
      if (unadopted.valid && unadopted.line.active) {
        unadopted.line.runtime_reset = true;
        emit_fg_switch(unadopted.line);
      }
    }

    void adopt_fg_switch(state &data) {
      AcquireSRWLockExclusive(&fg_switch_handover_lock);
      const auto handed = std::exchange(fg_switch_handed, {});
      ReleaseSRWLockExclusive(&fg_switch_handover_lock);
      if (!handed.valid) return;
      data.fg_mode = handed.fg_mode;
      data.fg_switch = handed.line;
    }

    // One "Sunshine depth FG switch" line per FG mode change (fg_switch_line),
    // from every Present's readiness evidence. ready: the Present shows depth
    // (fresh when copied, else a held copy). Diagnostic only. A runtime starts
    // from the FG mode the game's options stand at (initialize), and its
    // first Present takes over a torn-down runtime's mode and line.
    void track_fg_switch(state &data, const readiness_evidence &e, bool ready, const char *why) {
      if (!data.fg_switch_first_present) {
        data.fg_switch_first_present = true;
        adopt_fg_switch(data);
      }
      auto &line = data.fg_switch;
      const auto now = e.tick ? e.tick : GetTickCount64();
      const auto emit = [&] { emit_fg_switch(line); };
      if (e.fg_query == frame_generation_query_status::observed && e.fg.known && e.fg.mode != data.fg_mode) {
        if (line.active) emit();
        line = {};
        line.active = true;
        line.viewport = e.fg.viewport; line.from = data.fg_mode; line.to = e.fg.mode; line.tick = now;
        data.fg_mode = e.fg.mode;
      }
      if (!line.active) return;
      ++line.presents;
      if (ready && e.copied) {
        line.fresh_ms = now >= line.tick ? now - line.tick : 0;
        emit();
        return;
      }
      if (!line.reasoned) {
        line.reasoned = true;
        // The display decision's own reason where it decided (a hold names
        // why it held), else the Present's.
        line.first_reason = e.source_reason && std::strcmp(e.source_reason, "not_checked") ? e.source_reason : why;
        line.selection = e.capture.selection;
      }
      if (ready) {
        ++line.held;
        const auto tick = data.output.provided.tick;
        if (tick && now >= tick) line.max_held_age = std::max(line.max_held_age, now - tick);
      } else ++line.mono;
      if (now - line.tick >= 1000) emit();
    }

    // presenting: a Present's decision (not an invalidation between them).
    void trace_readiness(api::effect_runtime *runtime, state &data, const readiness_evidence &e,
        bool ready, const char *why, bool presenting) {
      const sunshine_game3d::slow_step step("depth readiness trace");
      if (presenting) track_fg_switch(data, e, ready, why);
      const bool changed = data.traced_ready != ready;
      data.traced_ready = ready;
      if (!changed) return;
      const auto now = e.tick ? e.tick : GetTickCount64();
      if (!ready) {
        data.trace_loss_tick = now;
        data.traced_loss = now >= data.next_trace_episode;
        if (!data.traced_loss) { ++data.suppressed_trace_episodes; return; }
        // At most four admitted loss episodes per second, each with its paired
        // recovery. No ready/unavailable decision depends on this diagnostic.
        data.next_trace_episode = now + 250;
        const auto lookup = [](std::uint64_t revision) {
          observation_loss_lookup out;
          out.requested_revision = revision;
          out.found = query_depth_observation_loss(revision, out.value);
          return out;
        };
        auto &loss = data.trace_observation_loss;
        loss.present = e.present;
        loss.sampled_tick = GetTickCount64();
        // Contemporaneous evidence only: depth decisions never consult it.
        loss.sampled = lookup(depth_observation_revision());
      } else if (!data.traced_loss) return; // No startup/recovery-only noise.
      // The game process's latest video-memory sample (game3d_vram.h), which
      // the present thread took before this Present's depth preparation.
      const sunshine_game3d::vram::latest_suffix memory;
      readiness_line line;
      line.ready = ready; line.why = why;
      line.runtime = reinterpret_cast<std::uint64_t>(runtime); line.native_swapchain = runtime->get_native();
      line.now = now; line.loss_tick = data.trace_loss_tick; line.suppressed_episodes = data.suppressed_trace_episodes;
      line.output_reused = data.output.reused_depth; line.current = data.frame.metadata;
      line.retained_clear = data.cache.reason(); line.memory = memory.text;
      char text[3072]{};
      format_readiness(text, sizeof(text), e, line);
      sunshine_log::message(reshade::log::level::info, text);
      // Repeat the frozen first-loss evidence on recovery, without querying a
      // newer revision or retroactively filling a missing/busy journal entry.
      format_observation_evidence(text, sizeof(text), ready, reinterpret_cast<std::uint64_t>(runtime), data.trace_loss_tick,
        data.trace_observation_loss);
      sunshine_log::message(reshade::log::level::info, text);
      data.suppressed_trace_episodes = 0;
      if (ready) data.traced_loss = false;
    }

    void log_render_size(render_size_state &size, std::uint64_t now) {
      size.next_line = now + 5000;
      char text[640]{};
      format_render_size(text, sizeof(text), size);
      sunshine_log::message(reshade::log::level::info, text);
      size.changes = size.held = 0;
    }

    // Counts render-size changes of one logical source (dynamic resolution),
    // which keep depth ready: a fresh copy of a new size is used at once, and
    // a Present whose newer size is still pending holds the previous copy
    // (sunshine_game3d::depth_output::decide_display). At most one line every 5 s while the
    // size changes. A new source, storage format or view first reports the
    // previous source's unreported changes, then restarts the count.
    void track_render_size(state &data, bool copied, bool held, bool selected) {
      auto &size = data.render_size;
      if (copied) {
        const auto &packet = data.frame;
        const auto &metadata = packet.metadata;
        const std::array<std::uint64_t, 4> source{static_cast<std::uint64_t>(metadata.provider), metadata.epoch,
          metadata.source_id, metadata.viewport};
        const bool same = size.width && source == size.source && packet.format == size.format &&
          packet.srv_format == size.view_format;
        const bool resized = same && (packet.width != size.width || packet.height != size.height ||
          packet.area.left != size.area.left || packet.area.top != size.area.top ||
          packet.area.width != size.area.width || packet.area.height != size.area.height);
        if (!same) {
          if (size.changes) log_render_size(size, GetTickCount64());
          size = {source, 0, 0, 0, 0, {}, 0, 0, 0, size.next_line};
        }
        if (resized) {
          if (!size.changes) {
            size.min_width = size.max_width = size.area.width;
            size.min_height = size.max_height = size.area.height;
          }
          ++size.changes; ++size.total;
          size.min_width = std::min(size.min_width, packet.area.width);
          size.min_height = std::min(size.min_height, packet.area.height);
          size.max_width = std::max(size.max_width, packet.area.width);
          size.max_height = std::max(size.max_height, packet.area.height);
        }
        size.width = packet.width; size.height = packet.height; size.area = packet.area;
        size.format = packet.format; size.view_format = packet.srv_format;
      } else if (held && selected && size.width &&
          (data.frame.width != data.output.width || data.frame.height != data.output.height ||
           data.frame.area.width != data.output.active_width || data.frame.area.height != data.output.active_height)) {
        // The held copy is the previous size; the selected newer one is pending.
        ++size.held;
      }
      if (!size.changes) return;
      const auto now = GetTickCount64();
      if (now >= size.next_line) log_render_size(size, now);
    }
  }

  // The status lines (streamline_depth_provider_log.h).
  const char *name(display_status value) {
    switch (value) {
      case display_status::not_attempted: return "not_attempted";
      case display_status::ready: return "ready";
      case display_status::reused_depth: return "reused_depth";
      case display_status::texture_creation_failed: return "texture_creation_failed";
      case display_status::view_creation_failed: return "view_creation_failed";
      case display_status::waiting_capture: return "waiting_capture";
    }
    return "unknown";
  }

  const char *name(sunshine_game3d::depth_output::display_action action) {
    switch (action) {
      case sunshine_game3d::depth_output::display_action::copy_fresh: return "copy_fresh";
      case sunshine_game3d::depth_output::display_action::hold: return "hold";
      case sunshine_game3d::depth_output::display_action::invalidate: return "invalidate";
    }
    return "unknown";
  }

  void format_readiness(char *text, std::size_t size, const readiness_evidence &e, const readiness_line &line) {
    const auto now = line.now;
    const auto age = [now](std::uint64_t tick) -> unsigned long long {
      return tick && now >= tick ? now - tick : UINT64_MAX;
    };
    const auto u = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
    const auto &c = e.capture;
    const auto selection_age = [&c](std::uint64_t tick) -> unsigned long long {
      return c.selection_tick_ms && tick && c.selection_tick_ms >= tick ? c.selection_tick_ms - tick : UINT64_MAX;
    };
    const auto &p = e.prior.provided;
    const auto &current = line.current;
    const auto provider_name = [](sunshine_scene_depth::provider_kind provider) {
      return provider == sunshine_scene_depth::provider_kind::ngx ? "NGX" : "Streamline";
    };
    // An empty acquisition diagnostic's default enum is not source evidence.
    // FG policy can still establish Streamline authority before a tag exists.
    const char *current_provider = c.capture_id || c.sequence ? provider_name(c.provider) :
      e.selection.require_frame_generation ? "Streamline" : "unknown";
    std::snprintf(text, size,
      "Sunshine depth readiness: %s reason=%s runtime=0x%llx native_swapchain=0x%llx present=%llu unavailable_ms=%llu suppressed_episodes=%llu; "
      "provider=%s prior_provider=%s selected=%u snapshot_ready=%u copied=%u output_reused=%u display=%s selection=%s status=%s failure=%s "
      "capture=%llu sequence=%llu newest_sequence=%llu epoch=%llu source=%llu viewport=%u "
      "selection_tick_ms=%llu source_age_ms=%llu newest_view_age_ms=%llu active_views=%u "
      "finished=%u success=%u submitted=%u invalid=%u capture_pending=%u capture_repeated=%u "
      "fence=%llu completed=%llu completion_valid=%u producer_retired=%u; "
      "FG_query=%u FG_known=%u FG_enabled=%u FG_mode=%u FG_generated=%u FG_epoch=%llu FG_viewport=%u require_FG=%u FG_silent=%u policy_epoch=%llu; "
      "prior_output_ready=%u prior_retained=%u prior_age_ms=%llu prior_present=%llu prior_capture=%llu prior_sequence=%llu prior_epoch=%llu prior_source=%llu "
      "prior_viewport=%u prior_observation=%llu current_observation=%llu prior_feedback=%llu current_feedback=%llu "
      "prior_reset=%u current_reset=%u retained_clear=%s; "
      "source_action=%s source_reason=%s cache_reason=%s "
      "consumer=%s consumer_loss=%s cookie=%llu slot_invalid=%u%s",
      line.ready ? "recovered" : "lost", line.why, u(line.runtime), u(line.native_swapchain), u(e.present),
      line.ready && now >= line.loss_tick ? u(now - line.loss_tick) : 0ull, u(line.suppressed_episodes),
      current_provider, e.prior_provider_known ? provider_name(e.prior_provider) : "unknown", unsigned(e.available),
      unsigned(e.snapshot_ready), unsigned(e.copied), unsigned(line.output_reused), name(e.display),
      depth_capture::name(c.selection), depth_capture::name(c.result), depth_capture::name(c.failure),
      u(c.capture_id), u(c.sequence), u(c.newest_sequence), u(c.epoch), u(c.source_id), c.viewport,
      u(c.selection_tick_ms), selection_age(c.source_tick_ms), selection_age(c.newest_view_tick_ms), c.active_view_count,
      unsigned(c.finished), unsigned(c.success), unsigned(c.submitted), unsigned(c.invalid),
      unsigned(c.pending_frame), unsigned(c.repeated_frame), u(c.producer_fence),
      u(c.producer_completed), unsigned(c.producer_completion_valid), unsigned(c.producer_recording_retired),
      unsigned(e.fg_query), unsigned(e.fg.known), unsigned(e.fg.enabled), e.fg.mode, e.fg.generated_frames,
      u(e.fg.epoch), e.fg.viewport, unsigned(e.selection.require_frame_generation), unsigned(e.fg_silent), u(e.selection.epoch),
      unsigned(e.prior_output_ready), unsigned(e.prior.ready), age(p.tick), u(e.prior.frame_index), u(e.prior_capture), u(p.sequence), u(p.epoch),
      u(p.source_id), p.viewport, u(p.observation_revision), u(current.observation_revision),
      u(p.feedback.revision), u(current.feedback.revision), unsigned(p.feedback.reset), unsigned(current.feedback.reset),
      line.retained_clear, name(e.action), e.source_reason, e.cache_reason,
      depth_capture::name(e.consumer.result), depth_capture::name(e.consumer.invalidation),
      u(e.consumer.cookie), unsigned(e.consumer.slot_invalid), line.memory);
  }

  void format_observation_evidence(char *text, std::size_t size, bool ready, std::uint64_t runtime,
      std::uint64_t loss_tick, const observation_loss_episode &loss) {
    const auto u = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
    const auto format_loss =[](const observation_loss_lookup &lookup, char *out, std::size_t size) {
      const auto &event = lookup.value;
      const auto &context = event.details;
      std::snprintf(out, size,
        "revision=%llu found=%u cause=%s site=%s line=%u tick_ms=%llu thread=%u sequence=%llu viewport=%u feature=%u sdk_known=%u sdk_result=%d reset=%u",
        static_cast<unsigned long long>(lookup.requested_revision), unsigned(lookup.found),
        lookup.found ? loss_diagnostics::name(event.cause) : "unavailable",
        lookup.found && event.site ? event.site : "unavailable", event.line,
        static_cast<unsigned long long>(event.tick), event.thread_id,
        static_cast<unsigned long long>(context.sequence), context.viewport, context.feature,
        unsigned(context.has_sdk_result), context.sdk_result, context.reset);
    };
    char sampled_loss[512]{};
    format_loss(loss.sampled, sampled_loss, sizeof(sampled_loss));
    std::snprintf(text, size,
      "Sunshine depth observation evidence: %s runtime=0x%llx loss_present=%llu loss_tick_ms=%llu sampled_tick_ms=%llu origin=episode_start; "
      "sampled_only={%s}",
      ready ? "recovered" : "lost", u(runtime), u(loss.present),
      u(loss_tick), u(loss.sampled_tick), sampled_loss);
  }

  void format_render_size(char *text, std::size_t size, const render_size_state &value) {
    const auto provider = static_cast<sunshine_scene_depth::provider_kind>(value.source[0]);
    std::snprintf(text, size,
      "Sunshine depth render size: changed=%llu held=%llu total=%llu active=%ux%u..%ux%u now allocation=%ux%u "
      "active=%u,%u,%u,%u; provider=%s epoch=%llu source=%llu viewport=%llu; dynamic resolution of one source: "
      "each copy keeps its own size, a new size is used at once and a pending one holds the previous copy, so depth "
      "stays ready",
      static_cast<unsigned long long>(value.changes), static_cast<unsigned long long>(value.held),
      static_cast<unsigned long long>(value.total), value.min_width, value.min_height, value.max_width, value.max_height,
      value.width, value.height, value.area.left, value.area.top, value.area.width, value.area.height,
      provider == sunshine_scene_depth::provider_kind::ngx ? "NGX" : "Streamline",
      static_cast<unsigned long long>(value.source[1]), static_cast<unsigned long long>(value.source[2]),
      static_cast<unsigned long long>(value.source[3]));
  }

  void format_frame_generation_scope(char *text, std::size_t size, bool silent, std::uint32_t viewport,
      std::uint64_t epoch, std::uint64_t changes) {
    std::snprintf(text, size, silent ?
      "Sunshine Streamline frame generation depth scope: silent viewport=%u epoch=%llu changes=%llu; its newest frame has "
      "no accepted FG depth copy (or none came for 1000 ms), so the viewport's ordinary SR/RR depth may supply depth until "
      "the next FG copy" :
      "Sunshine Streamline frame generation depth scope: live viewport=%u epoch=%llu changes=%llu; an accepted FG copy "
      "owns the viewport's depth again",
      viewport, static_cast<unsigned long long>(epoch), static_cast<unsigned long long>(changes));
  }

  void format_fg_switch(char *text, std::size_t size, const fg_switch_line &line) {
    char fresh[24] = "none";
    if (line.fresh_ms != UINT64_MAX) std::snprintf(fresh, sizeof(fresh), "%llu", static_cast<unsigned long long>(line.fresh_ms));
    std::snprintf(text, size,
      "Sunshine depth FG switch: viewport=%u mode=%u->%u presents=%llu first_fresh_ms=%s held=%llu max_held_age_ms=%llu "
      "mono=%llu first_reason=%s selection=%s%s",
      line.viewport, line.from, line.to, static_cast<unsigned long long>(line.presents), fresh,
      static_cast<unsigned long long>(line.held), static_cast<unsigned long long>(line.max_held_age),
      static_cast<unsigned long long>(line.mono), line.first_reason, depth_capture::name(line.selection),
      line.runtime_reset ? " ended=runtime_reset" : "");
  }

  void format_unassociated(char *text, std::size_t size, depth_capture::status status,
      const depth_capture::capture_diagnostic &capture, bool hold, std::uint64_t hold_ms) {
    std::snprintf(text, size, "Sunshine API depth: %s; capture=%llu sequence=%llu epoch=%llu producer_queue=0x%llx consumer_queue=0x%llx requested_queue=0x%llx producer_fence=%llu producer_completed=%llu completion_valid=%u producer_retired=%u finished=%u success=%u submitted=%u invalid=%u cause=%s; using generic depth fallback",
      depth_capture::name(status), static_cast<unsigned long long>(capture.capture_id),
      static_cast<unsigned long long>(capture.sequence), static_cast<unsigned long long>(capture.epoch),
      static_cast<unsigned long long>(capture.queue), static_cast<unsigned long long>(capture.consumer_queue),
      static_cast<unsigned long long>(capture.requested_queue), static_cast<unsigned long long>(capture.producer_fence),
      static_cast<unsigned long long>(capture.producer_completed), unsigned(capture.producer_completion_valid),
      unsigned(capture.producer_recording_retired),
      unsigned(capture.finished), unsigned(capture.success), unsigned(capture.submitted),
      unsigned(capture.invalid), depth_capture::name(capture.failure));
    if (hold) {
      // The same evidence; ownership is held (mono) for the handoff.
      const auto suffix = std::strstr(text, "; using generic depth fallback");
      if (suffix) std::snprintf(suffix, size - std::size_t(suffix - text),
        "; holding API ownership (mono) for a provider handoff, at most %llu ms",
        static_cast<unsigned long long>(hold_ms));
    }
  }

  void format_selected(char *text, std::size_t size, const selected_line &line, const depth_capture::packet &frame,
      const depth_capture::consumer_diagnostic &consumer, const depth_capture::capture_diagnostic &capture) {
    std::snprintf(text, size, "Sunshine %s depth: %s; source_selected=1 snapshot_ready=%u final_ready=%u display=%s consumer=%s loss=%s; allocation=%ux%u active=%u,%u,%u,%u source=%u viewport=%u; metadata_projection=%u output_projection=%u; state=0x%x proof=%u; capture=%llu sequence=%llu present=%llu command=0x%llx cookie=%llu device=%llu expected_device=%llu slot_invalid=%u tracked_commands=%u; producer_queue=0x%llx consumer_queue=0x%llx producer_fence=%llu producer_completed=%llu completion_valid=%u producer_retired=%u; resource_id=%llu",
      line.path, depth_capture::name(line.status), unsigned(frame.pixel_ready), unsigned(line.output_ready), name(line.display), depth_capture::name(consumer.result),
      depth_capture::name(consumer.invalidation),
      frame.width, frame.height, frame.area.left, frame.area.top, frame.area.width, frame.area.height,
      unsigned(frame.metadata.resource.kind), frame.metadata.viewport,
      unsigned(frame.metadata.projection.supplied), unsigned(line.output_projection_supplied),
      frame.metadata.native_state, unsigned(frame.metadata.proof),
      static_cast<unsigned long long>(frame.capture_id), static_cast<unsigned long long>(frame.metadata.sequence),
      static_cast<unsigned long long>(line.present), static_cast<unsigned long long>(line.command),
      static_cast<unsigned long long>(consumer.cookie), static_cast<unsigned long long>(consumer.device_identity),
      static_cast<unsigned long long>(consumer.expected_device_identity), unsigned(consumer.slot_invalid), consumer.tracked_commands,
      static_cast<unsigned long long>(capture.queue), static_cast<unsigned long long>(capture.consumer_queue),
      static_cast<unsigned long long>(capture.producer_fence), static_cast<unsigned long long>(capture.producer_completed),
      unsigned(capture.producer_completion_valid), unsigned(capture.producer_recording_retired),
      static_cast<unsigned long long>(frame.resource_id));
  }

  void format_held(char *text, std::size_t size, const char *path, depth_capture::status status, bool output_ready,
      display_status display, const depth_capture::capture_diagnostic &capture, const native_observer::counters &observer) {
    std::snprintf(text, size, "Sunshine %s depth: %s; source_selected=0 provider_held=1 final_ready=%u display=%s cause=%s; capture=%llu sequence=%llu epoch=%llu producer_recording=%llu queue=0x%llx consumer_queue=0x%llx producer_fence=%llu producer_completed=%llu completion_valid=%u producer_retired=%u retire_fence=%llu finished=%u success=%u submitted=%u invalid=%u; observer_dropped=%llu unreadable=%llu rejected=%llu barrier_overflow=%llu submission_overflow=%llu discovery_contention=%llu",
      path, depth_capture::name(status), unsigned(output_ready), name(display), depth_capture::name(capture.failure),
      static_cast<unsigned long long>(capture.capture_id), static_cast<unsigned long long>(capture.sequence),
      static_cast<unsigned long long>(capture.epoch), static_cast<unsigned long long>(capture.command),
      static_cast<unsigned long long>(capture.queue), static_cast<unsigned long long>(capture.consumer_queue),
      static_cast<unsigned long long>(capture.producer_fence), static_cast<unsigned long long>(capture.producer_completed),
      unsigned(capture.producer_completion_valid), unsigned(capture.producer_recording_retired),
      static_cast<unsigned long long>(capture.retire_fence), unsigned(capture.finished), unsigned(capture.success),
      unsigned(capture.submitted), unsigned(capture.invalid), static_cast<unsigned long long>(observer.dropped),
      static_cast<unsigned long long>(observer.unreadable), static_cast<unsigned long long>(observer.rejected),
      static_cast<unsigned long long>(observer.barrier_overflow), static_cast<unsigned long long>(observer.submission_overflow),
      static_cast<unsigned long long>(observer.discovery_contention));
  }

  void append_selection(char *text, std::size_t size, const depth_capture::capture_diagnostic &capture,
      const sunshine_game3d::depth_output::display_decision &update, std::uint64_t prior_depth_age,
      const sunshine_scene_depth::jitter_offset &jitter) {
    const auto used = std::strlen(text);
    std::snprintf(text + used, size - used, "; selection=%s source_action=%s source_reason=%s consumed_completed_capture=%llu newest_sequence=%llu prior_depth_age_ms=%llu jitter_supplied=%u jitter_px=(%.6g,%.6g) jitter_render=%ux%u",
      depth_capture::name(capture.selection), name(update.action), update.reason,
      static_cast<unsigned long long>(capture.consumed_completed_capture),
      static_cast<unsigned long long>(capture.newest_sequence),
      static_cast<unsigned long long>(prior_depth_age), unsigned(jitter.supplied),
      double(jitter.x), double(jitter.y), jitter.width, jitter.height);
  }

  void append_loss(char *text, std::size_t size, bool known, const loss_diagnostics::event &loss, std::uint64_t age_ms,
      std::uint64_t losses_since) {
    const auto tail = std::strlen(text);
    std::snprintf(text + tail, size - tail, "; last_loss=%s site=%s line=%u age_ms=%llu losses_since_last_line=%llu",
      known ? sunshine_streamline::loss_diagnostics::name(loss.cause) : "unknown", known && loss.site ? loss.site : "unknown",
      known ? loss.line : 0u, static_cast<unsigned long long>(age_ms),
      static_cast<unsigned long long>(losses_since));
  }
