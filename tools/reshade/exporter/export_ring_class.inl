// SPDX-License-Identifier: GPL-3.0-only
// Part of exporter.cpp, the same translation unit (tools/reshade/README.md, Source map),
// included inside its anonymous namespace immediately before publisher_t: the export ring,
// the producer side of the wire (docs/reshade-sbs.md, GPU handoff contract). It owns the
// shared state's view, the published metadata, the active, retired and cached generations,
// their slots and the ring counters. export_ring.inl holds its types and wire helpers.
//
// publisher_t holds it as ring_, at the former position of the active generation, and calls
// it only under publisher_t's mutex_. It has no user-declared destructor: the unload steps
// run from publisher_t's destructor (unload, unmap), and its generations are destroyed with
// ring_ (cached, retired, then active), after parked_ and before overlays_, as before.
// What it needs of the publisher comes as callables: release_views(texture) drops the views
// renderers cached of a released ring's slot, and count_output(runtime, sample, now) adds a
// published slot's sample to its runtime's SBS output counters.
  class export_ring {
    friend class publisher_t;
#ifdef SUNSHINE_SBS_TEST
    friend struct publisher_tests;
#endif

  public:
    // Add-on unload (publisher_t's destructor, unless the process is
    // stopping): hosts see the identity alone, and each generation with
    // unfinished GPU work keeps its native allocations until process exit.
    void unload() {
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      namespace teardown = sunshine_game3d::test::teardown;
#endif
      if (shared_) {
        publish(identity_);
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        teardown::note(teardown::step::identity_published);
#endif
      }
      if (generation_) {
        generation_->preserve_inflight_on_unload();
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        teardown::note(teardown::step::preserved, teardown::ring_owner::active);
#endif
      }
      if (retired_) {
        retired_->preserve_inflight_on_unload();
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        teardown::note(teardown::step::preserved, teardown::ring_owner::retired);
#endif
      }
      if (cached_ring_) {
        cached_ring_->preserve_inflight_on_unload();
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        teardown::note(teardown::step::preserved, teardown::ring_owner::cached);
#endif
      }
    }

    // The shared state's view ends with the publisher (its destructor); the
    // mapping is the publisher's.
    void unmap() {
      if (shared_) {
        UnmapViewOfFile(shared_);
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        namespace teardown = sunshine_game3d::test::teardown;
        teardown::note(teardown::step::unmapped);
#endif
      }
    }

    // The ring part of frame(): gives the consumer `nonce` this runtime's ring
    // for `source`. The current generation serves while it matches; otherwise
    // (ring_replaced) the idle ring is renewed, the cached ring of this
    // transfer reused or a new ring created, and its metadata and handles
    // published. False: this Present exports nothing.
    template<class Views>
    bool serve(api::effect_runtime *runtime, const source_t &source, std::uint64_t nonce, api::device_api backend,
        const Views &release_views) {
      const bool replace = !generation_ || generation_->runtime != runtime || generation_->nonce != nonce ||
                           !same_ring(generation_->source, source);
      if (replace) {
        if (generation_) {
          deactivate(generation_->runtime, "ring_replaced");
        }
        retire_destroyed_generation(release_views);
        if (generation_ && !generation_->finished()) {
          return false;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now < retry_at_) {
          return false;
        }
        if (next_generation_ >= wire::max_generation) {
          deactivate(runtime, "generation_limit");
          return false;
        }
        // An abandoned consumer read of a reused slot finishes within frames;
        // a ring idle this long only ever served consumers that are gone. A
        // consumer that restarts while exporting still gets fresh resources.
        // finished() is also true for a removed device; such a ring is never reused.
        const bool reuse = generation_ && !generation_->owner_destroyed && generation_->owner_runtime == runtime &&
          generation_->completed() != UINT64_MAX &&
          generation_->native_swapchain == runtime->get_native() && same_ring(generation_->source, source) &&
          generation_->inactive_since != std::chrono::steady_clock::time_point{} && now - generation_->inactive_since >= idle_ring_reuse;
        // The cached ring of the other colour transfer (item 6: one entry,
        // released after cached_ring_idle) serves a toggle back the same way.
        const bool cached = !reuse && cached_ring_ && !cached_ring_->owner_destroyed && cached_ring_->owner_runtime == runtime &&
          cached_ring_->completed() != UINT64_MAX && cached_ring_->finished() &&
          cached_ring_->native_swapchain == runtime->get_native() && same_ring(cached_ring_->source, source) &&
          now - cached_ring_->inactive_since >= idle_ring_reuse;
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        note_rings();  // Names a ring the assignments below destroy.
#endif
        // A finished ring that is not reused now stays cached for a toggle back.
        const auto keep_outgoing = [&] {
          if (generation_ && generation_->finished() && !generation_->owner_destroyed && generation_->completed() != UINT64_MAX) {
            if (generation_->inactive_since == std::chrono::steady_clock::time_point{}) generation_->inactive_since = now;
            cached_ring_ = std::move(generation_);
            cached_ring_since_ = now;
          }
        };
        if (reuse) {
          generation_->renew(runtime, nonce, ++next_generation_);
        } else if (cached) {
          auto previous = std::move(cached_ring_);
          keep_outgoing();
          generation_ = std::move(previous);
          generation_->renew(runtime, nonce, ++next_generation_);
        } else {
          auto next = std::make_unique<generation_t>();
          if (!next->create(runtime, source, nonce, ++next_generation_)) {
            deactivate(runtime, "ring_creation_failed");
            retry_at_ = now + std::chrono::seconds(2);
            return false;
          }
          keep_outgoing();
          generation_ = std::move(next);
        }
        wire::metadata_t metadata = identity_;
        set_source(metadata, source);
        metadata.generation = generation_->id;
        metadata.accepted_consumer_nonce = nonce;
        for (std::uint32_t index = 0; index < wire::slot_count; ++index) {
          metadata.texture_handles[index] = reinterpret_cast<std::uint64_t>(generation_->texture_handles[index].get());
        }
        metadata.ready_fence_handle = reinterpret_cast<std::uint64_t>(generation_->fence_handle.get());
        publish(metadata, true, runtime);
        char message[256];
        std::snprintf(message, sizeof(message), "Sunshine SBS: generation %llu, %ux%u full SBS, DXGI %u, D3D%u, %s, protocol %u (source color %u)%s", static_cast<unsigned long long>(metadata.generation), metadata.packed_width, metadata.packed_height, metadata.dxgi_format, backend == api::device_api::d3d11 ? 11u : 12u, metadata.color_transfer == wire::transfer::scrgb ? "scRGB" : metadata.color_transfer == wire::transfer::pq ? "PQ HDR10" : "sRGB", metadata.protocol_version, static_cast<unsigned>(source.color.input), reuse ? ", reusing the idle export ring" : cached ? ", reusing the cached export ring of this transfer" : "");
        log(reshade::log::level::info, message);
      }
      return true;
    }

    // Only a consumer that declared this export protocol is answered
    // (docs/reshade-sbs.md, GPU handoff contract): a host of another release
    // is refused by name, once per request, and gets no ring. Zero is a
    // request that declared nothing, or one being replaced (its successor's
    // declaration was read).
    bool consumer_speaks_protocol(api::effect_runtime *runtime, std::uint64_t nonce) {
      const auto protocol = read_consumer_protocol(*shared_, nonce);
      if (protocol == wire::version) return true;
      if (protocol && refused_nonce_ != nonce) {
        refused_nonce_ = nonce;
        char message[256];
        std::snprintf(message, sizeof(message), "Sunshine SBS: refusing a host that speaks export protocol %u; this add-on speaks protocol %u (%u export slots). Install the Sunshine 3D host and add-on of the same release; no stereo export to this host.", protocol, wire::version, wire::slot_count);
        log(reshade::log::level::warning, message);
      }
      deactivate(runtime, protocol ? "consumer_protocol" : "consumer_undeclared");
      return false;
    }

    // Claims a slot for this Present's export (docs/reshade-sbs.md, GPU handoff
    // contract): a free slot first, else the oldest ready slot that a newer
    // completed frame supersedes. The consumer claims the newest ready frame
    // whose fence passed, so the newest completed frame of the ready and
    // reading slots is the one it claims next (or holds) and is never written
    // over, even while a newer write is still queued: a consumer that comes
    // late still finds it, instead of nothing until the queued write lands. A
    // slot is reusable once its previous GPU write completed: a ready slot
    // whose write is still queued is never reused either. A consumer may
    // discard a frame without reading it, so a free slot also waits for its
    // write to complete (every free slot of the current generation has:
    // reset, or freed by a consumer after its fence passed).
    // overwrote: a ready slot was claimed.
    std::uint32_t acquire_slot(std::uint64_t completed, bool *overwrote = nullptr) {
      if (overwrote) *overwrote = false;
      // A recorded copy must be fenced before admitting the next. Submitted
      // work is bounded by the existing ring, not a global one-copy gate.
      if (!generation_ || generation_->signal_failed || completed == UINT64_MAX ||
          generation_->pending_slot != wire::slot_count) return wire::slot_count;
      const auto id = generation_->id;
      std::array<wire::slot_state, wire::slot_count> states{};
      std::array<bool, wire::slot_count> current{};
      std::uint64_t newest_completed = 0;
      for (std::uint32_t index = 0; index < wire::slot_count; ++index) {
        const auto control = static_cast<std::uint64_t>(InterlockedCompareExchange64(
          reinterpret_cast<volatile LONG64 *>(&shared_->slots[index].control), 0, 0));
        current[index] = wire::control_generation(control) == id;
        states[index] = wire::control_state(control);
        if (current[index] && (states[index] == wire::slot_state::ready || states[index] == wire::slot_state::reading) &&
            shared_->slots[index].sequence <= completed)
          newest_completed = std::max(newest_completed, shared_->slots[index].sequence);
      }
      const auto written = [&](std::uint32_t index) {
        return current[index] && completed >= shared_->slots[index].sequence;
      };
      for (std::uint32_t offset = 0; offset < wire::slot_count; ++offset) {
        const auto index = (next_slot_ + offset) % wire::slot_count;
        if (states[index] == wire::slot_state::free && written(index) &&
            exchange_state(shared_->slots[index], id, wire::slot_state::writing, wire::slot_state::free)) return index;
      }
      // Completed ready slots oldest first; a claim the consumer won meanwhile moves on.
      std::array<std::uint32_t, wire::slot_count> order{};
      for (std::uint32_t index = 0; index < wire::slot_count; ++index) order[index] = index;
      std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
        return shared_->slots[a].sequence < shared_->slots[b].sequence;
      });
      for (const auto index : order) {
        if (states[index] != wire::slot_state::ready || !written(index) || shared_->slots[index].sequence >= newest_completed) continue;
        if (exchange_state(shared_->slots[index], id, wire::slot_state::writing, wire::slot_state::ready)) {
          if (overwrote) *overwrote = true;
          return index;
        }
      }
      return wire::slot_count;
    }

    template<class Views>
    void retire_destroyed_generation(const Views &release_views) {
      if (retired_ && retired_->finished()) {
        release_ring(retired_, release_views);
      }
      if (!generation_ || !generation_->owner_destroyed) {
        return;
      }
      if (generation_->finished()) {
        release_ring(generation_, release_views);
        return;
      }
      if (!retired_) {
        retired_ = std::move(generation_);
        log(reshade::log::level::warning, "Sunshine SBS: retaining a destroyed runtime's pending generation; successor runtime may export");
      } else if (!retirement_limit_logged_) {
        retirement_limit_logged_ = true;
        log(reshade::log::level::error, "Sunshine SBS: two runtime generations have unconfirmed GPU work; restart the game to resume export");
      }
    }

    // One-shot invalidation also covers identity metadata published before a host connects.
    // The reason is logged once, when the export actually stops; only then
    // does this return true.
    bool deactivate(api::effect_runtime *runtime, const char *reason = "stopped") {
      if (!runtime) {
        return false;
      }
      const bool stopped = published_runtime_ == runtime;
      if (stopped) {
        publish(identity_);
        char message[160];
        std::snprintf(message, sizeof(message), "Sunshine SBS: export inactive (%s, %s); waiting for a valid focused technique", reason,
          export_pause_class(reason));
        log(reshade::log::level::info, message);
      }
      if (generation_ && generation_->runtime == runtime) {
        for (auto &overlay : generation_->overlays) if (overlay) overlay->cancel();
        generation_->runtime = nullptr;
        generation_->inactive_since = std::chrono::steady_clock::now();
      }
      return stopped;
    }

    // Publishes the pending export, then queues its fence signal behind the
    // pack and overlay already submitted on the generation's queue (D3D11
    // flushes after it). Ready first: the consumer claims a slot only once the
    // fence reached its sequence, so a ready slot is never read early and
    // every fence wake finds its slot ready. A recording retired by reload,
    // focus or nonce changes, or whose overlay composite was never recorded,
    // is still fenced but never published. A failed signal withdraws the slot
    // and deactivates the export (runtime: the reshade_present caller, null
    // from finish_present).
    template<class Counts>
    void signal_pending(api::effect_runtime *runtime, const Counts &count_output) {
      auto &g = *generation_;
      const auto index = std::exchange(g.pending_slot, wire::slot_count);
      const bool composed = !std::exchange(g.pending_overlay, false);
      if (!composed && g.overlays[index]) g.overlays[index]->cancel();
      const bool publish = composed && g.runtime && published_runtime_ == g.runtime && read_nonce(*shared_) == g.nonce;
      if (publish) complete_slot(index, g.pending_qpc, count_output);
      HRESULT hr = E_FAIL;
      if (g.backend == api::device_api::d3d11) {
        hr = g.context11->Signal(g.fence11.get(), g.last_submitted);
        g.context11->Flush();
      } else {
        // The add-on's own fence (signal_own): never a game Signal to the
        // order hooks.
        hr = sunshine_streamline::native_observer::signal_own(g.queue12.get(), g.fence12.get(), g.last_submitted);
      }
      if (SUCCEEDED(hr)) return;
      g.signal_failed = true;
      if (publish) exchange_state(shared_->slots[index], g.id, wire::slot_state::writing, wire::slot_state::ready);
      log_hr(g.backend == api::device_api::d3d11 ? "D3D11 publication fence signal" : "D3D12 publication fence signal", hr);
      deactivate(runtime ? runtime : g.runtime, "fence_signal_failed");
    }

    // A ring that stopped exporting (deactivated: focus loss, no consumer)
    // and has no unfinished work is released after export_ring_idle, so a
    // stereo-to-mono switch frees its four 2W x H slots (265-530 MB at 4K); a
    // later export allocates a new ring. The cached ring of the other colour
    // transfer keeps its own cached_ring_idle.
    template<class Views>
    void release_idle_rings(const Views &release_views) {
      const auto now = std::chrono::steady_clock::now();
      if (generation_ && !generation_->runtime && generation_->finished() &&
          generation_->inactive_since != std::chrono::steady_clock::time_point{} &&
          now - generation_->inactive_since >= export_ring_idle) {
        log(reshade::log::level::info, "Sunshine SBS: released the idle export ring");
        release_ring(generation_, release_views);
      }
      if (cached_ring_ && cached_ring_->finished() && now - cached_ring_since_ >= cached_ring_idle) release_ring(cached_ring_, release_views);
    }

    // Releases a ring that no pack follows. Renderers cache render-target
    // views of the slots they pack into (renderer::pack) and drop them only
    // at their next pack; a D3D11 view holds its texture, so they are dropped
    // here first or the slots' VRAM would stay allocated.
    template<class Views>
    void release_ring(std::unique_ptr<generation_t> &ring, const Views &release_views) {
      if (!ring) return;
      for (std::uint32_t index = 0; index < wire::slot_count; ++index) {
        const auto texture = ring->texture(index);
        if (!texture.handle) continue;
        release_views(texture);
      }
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      note_rings();
#endif
      ring.reset();
    }

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    // SunshineGame3DTestTeardownOrder: the member that holds each ring now,
    // which names the ring when it is destroyed.
    void note_rings() const {
      sunshine_game3d::test::teardown::note_rings(generation_.get(), retired_.get(), cached_ring_.get());
    }
#endif

    template<class Counts>
    void complete_slot(std::uint32_t index, std::uint64_t qpc, const Counts &count_output) {
      const auto now = GetTickCount64();
      auto &slot = shared_->slots[index];
      slot.sequence = generation_->last_submitted;
      slot.qpc = qpc;
      slot.cursor_plane_flags = wire::cursor_plane_present;
      slot.ui_parallax_uv = generation_->pending_ui_parallax_uv;
      store_state(slot, generation_->id, wire::slot_state::ready);
      if (const auto sample = std::exchange(generation_->pending_output, std::nullopt))
        count_output(generation_->runtime, *sample, now);
    }

    void publish(const wire::metadata_t &metadata, bool reset_slots = false, api::effect_runtime *owner = nullptr) {
      InterlockedIncrement(reinterpret_cast<volatile LONG *>(&shared_->metadata_sequence));
      published_metadata_ = metadata;
      published_runtime_ = owner;
      MemoryBarrier();
      shared_->metadata = metadata;
      if (reset_slots) {
        for (auto &slot : shared_->slots) {
          slot.sequence = 0;
          slot.qpc = 0;
          slot.cursor_plane_flags = 0;
          slot.ui_parallax_uv = 0.f;
          store_state(slot, metadata.generation, wire::slot_state::free);
        }
      }
      MemoryBarrier();
      InterlockedIncrement(reinterpret_cast<volatile LONG *>(&shared_->metadata_sequence));
    }

    static void set_source(wire::metadata_t &metadata, const source_t &source) {
      metadata.window = reinterpret_cast<std::uint64_t>(source.window);
      metadata.adapter_luid = source.adapter;
      metadata.source_width = source.width;
      metadata.source_height = source.height;
      metadata.packed_width = source.width * 2;
      metadata.packed_height = source.height;
      metadata.dxgi_format = static_cast<std::uint32_t>(source.format);
      metadata.color_transfer = source.color.output;
      metadata.protocol_version = wire::version;
    }

  private:
    wire::shared_state_t *shared_ = nullptr;
    wire::metadata_t identity_;
    wire::metadata_t published_metadata_;
    api::effect_runtime *published_runtime_ = nullptr;
    std::unique_ptr<generation_t> generation_;
    // A missing finish_present after runtime destruction must not wedge the next runtime.
    // Bound this fallback to one retired ring plus the active ring; never allocate indefinitely.
    std::unique_ptr<generation_t> retired_;
    // The finished ring of the previous colour transfer or extent, reused on a
    // toggle back (item 6) and released once cached this long.
    std::unique_ptr<generation_t> cached_ring_;
    std::chrono::steady_clock::time_point cached_ring_since_{};
    static constexpr auto cached_ring_idle = std::chrono::seconds(60);
    static constexpr auto export_ring_idle = std::chrono::seconds(10);
    bool retirement_limit_logged_ = false;
    std::uint64_t next_generation_ = 0;
    std::uint32_t next_slot_ = 0;
    // The last request refused for another export protocol, named once.
    std::uint64_t refused_nonce_ = 0;
    std::chrono::steady_clock::time_point retry_at_ {};
  };
