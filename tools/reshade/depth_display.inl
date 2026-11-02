// SPDX-License-Identifier: GPL-3.0-only
// Provider display binding, storage and fenced retirement.
// Same TU as streamline_depth_provider.cpp; namespace-scope state and types stay there.
  namespace {
    void bind_api(api::effect_runtime *runtime, state &data, api::resource_view view, bool ready) {
      // ReShade 6.8 waits for the whole queue on every update, even if the
      // handle is unchanged. Keep the descriptor stable while pixels rotate.
      if (view != data.bound_view) {
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        depth_events::binding(runtime, depth_events::site_id::provider_bind, view.handle != 0);
#endif
        const sunshine_game3d::slow_step step("API depth binding");
        runtime->update_texture_bindings("DEPTH", view, view);
        data.bound_view = view;
      }
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      depth_events::readiness(runtime, depth_events::site_id::provider_bind_ready, ready);
#endif
      set_depth_ready(runtime, ready);
    }

    // Requires an idle runtime queue (teardown only).
    void destroy_display(api::effect_runtime *runtime, state &data) {
      data.cache.invalidate("display_recreated");
      auto *device = runtime->get_device();
      for (auto &entry : data.displays) {
        if (entry.view.handle) device->destroy_resource_view(entry.view);
        if (entry.texture.handle) device->destroy_resource(entry.texture);
        entry = {};
      }
      for (auto &entry : data.retired_displays) {
        if (entry.view.handle) device->destroy_resource_view(entry.view);
        if (entry.texture.handle) device->destroy_resource(entry.texture);
      }
      data.retired_displays.clear();
      if (data.retire_fence.handle) device->destroy_fence(data.retire_fence);
      data.retire_fence = {};
      data.display_view = {}; data.display_texture = {}; data.display_view_format = 0;
    }

    // pooled: on a pool thread, which logs its own slow release (on the
    // present thread "depth display retirement" names it). Leaves the count
    // last: destroy() may return, and the device go, as soon as it is zero.
    void release_displays(display_release &job, bool pooled) {
      LARGE_INTEGER start{}, end{}, frequency{};
      QueryPerformanceCounter(&start);
      std::uint64_t bytes = 0;
      // At process exit the device may be gone: nothing is released then.
      if (!sunshine_addon_lifetime::stopping())
        for (const auto &entry : job.entries) {
          if (entry.view.handle) job.device->destroy_resource_view(entry.view);
          if (entry.texture.handle) job.device->destroy_resource(entry.texture);
          bytes += entry.bytes;
        }
      QueryPerformanceCounter(&end);
      QueryPerformanceFrequency(&frequency);
      const double ms = double(end.QuadPart - start.QuadPart) * 1000.0 / double(frequency.QuadPart);
      static std::atomic<std::uint64_t> next_line{};
      const auto now = sunshine_game3d::clock::game_now();
      auto due = next_line.load(std::memory_order_relaxed);
      if (pooled && ms >= 8.0 && now >= due && next_line.compare_exchange_strong(due, now + 1000)) {
        const sunshine_game3d::vram::latest_suffix memory;
        char text[256];
        std::snprintf(text, sizeof(text), "Sunshine Game 3D: depth display release took %.1f ms off the present thread "
          "(%u textures, %.1f MiB)%s", ms, static_cast<unsigned>(job.entries.size()), double(bytes) / 1048576.0, memory.text);
        sunshine_log::message(reshade::log::level::info, text);
      }
      const auto owner = job.owner;
      {
        std::lock_guard lock(owner->mutex);
        --owner->outstanding;
      }
      owner->idle.notify_all();
    }

    void CALLBACK release_task(PTP_CALLBACK_INSTANCE, void *context) {
      const std::unique_ptr<display_release> job(static_cast<display_release *>(context));
      release_displays(*job, true);
    }

    // Without a pool thread, the entries go here, as before 10-27.
    void submit_release(std::unique_ptr<display_release> job) {
      {
        std::lock_guard lock(job->owner->mutex);
        ++job->owner->outstanding;
      }
      if (TrySubmitThreadpoolCallback(release_task, job.get(), nullptr)) {
        job.release();
        return;
      }
      release_displays(*job, false);
    }

    // Teardown. Once ExitProcess has begun, every other thread (the pool's
    // too) has ended: a job that has not run never will, and none runs.
    void wait_releases(state::release_jobs &jobs) {
      using shutdown_query = BOOLEAN(NTAPI *)();
      static const auto shutting_down = reinterpret_cast<shutdown_query>(
        reinterpret_cast<void *>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlDllShutdownInProgress")));
      if (sunshine_addon_lifetime::stopping() || (shutting_down && shutting_down())) return;
      std::unique_lock lock(jobs.mutex);
      jobs.idle.wait(lock, [&jobs] { return !jobs.outstanding; });
    }

    void retire_display(state &data, state::display_entry &entry, std::uint64_t present) {
      const auto format = static_cast<api::format>(entry.format);
      data.retired_displays.push_back({entry.texture, entry.view, present, 0,
        std::uint64_t(api::format_slice_pitch(format, api::format_row_pitch(format, entry.width), entry.height))});
      entry = {};
    }

    // A display unused this long is retired, so its storage does not outlive
    // a resolution drop or a provider flip (until 10-31 it stayed until a
    // third shape evicted it: after 4K then 1080p both stayed allocated), nor
    // its API source (until 11-01 the active display stayed for the rest of
    // the session once Generic took DEPTH over). A shape that returns sooner
    // reuses its entry: dynamic resolution alternating between two render
    // sizes, an FG toggle and back, or an API source that resumes.
    constexpr std::uint64_t display_idle_ms = 2000;

    // The provider's binding (bound_view) is in use at this Present: the
    // active display while the provider owns the pass. Any other entry,
    // the active one too once the provider stopped binding it (its API
    // source gone: Generic rebound DEPTH), unused for display_idle_ms is
    // retired as an evicted one is (fenced on a later Present, released off
    // the present thread). No binding changes here.
    void expire_idle_displays(state &data, std::uint64_t present, std::uint64_t now) {
      for (auto &entry : data.displays) {
        if (!entry.texture.handle) continue;
        if (entry.view == data.bound_view) entry.used = now;
        else if (now - entry.used >= display_idle_ms) {
          if (entry.texture == data.display_texture) { data.display_texture = {}; data.display_view = {}; data.display_view_format = 0; }
          retire_display(data, entry, present);
        }
      }
    }

    // Once per Present. A retired display is fenced on a Present after the one
    // that replaced it, when every earlier read of it (effects, renderer,
    // sampler) is already submitted on this queue; once that fence completed
    // it is handed to a pool thread (display_release). Signaled natively, so
    // ReShade's immediate list is not flushed early, and as the add-on's own
    // (signal_own), so it never waits on the queue-order lock. The present
    // thread only reads the fence and signals it.
    void collect_displays(api::effect_runtime *runtime, state &data, std::uint64_t present) {
      expire_idle_displays(data, present, sunshine_game3d::clock::game_now());
      if (data.retired_displays.empty()) return;
      auto *device = runtime->get_device();
      // Only a fence initialize() failed to create is created here.
      if (!data.retire_fence.handle && !device->create_fence(0, api::fence_flags::none, &data.retire_fence)) return;
      auto *fence = reinterpret_cast<ID3D12Fence *>(data.retire_fence.handle);
      auto *queue = reinterpret_cast<ID3D12CommandQueue *>(runtime->get_command_queue()->get_native());
      const auto completed = fence->GetCompletedValue();
      std::unique_ptr<display_release> released;
      auto keep = data.retired_displays.begin();
      for (auto &entry : data.retired_displays) {
        if (!entry.fence_value && present > entry.present) {
          const sunshine_game3d::slow_step step("depth display retire signal");
          if (SUCCEEDED(native_observer::signal_own(queue, fence, data.retire_value + 1))) entry.fence_value = ++data.retire_value;
        }
        if (entry.fence_value && completed != UINT64_MAX && completed >= entry.fence_value) {
          if (!released) released.reset(new display_release{device, data.releases, {}});
          released->entries.push_back(entry);
          continue;
        }
        *keep++ = entry;
      }
      data.retired_displays.erase(keep, data.retired_displays.end());
      if (released) submit_release(std::move(released));
    }

    void activate_display(state &data, const state::display_entry &entry) {
      data.display_texture = entry.texture; data.display_view = entry.view; data.display_view_format = entry.view_format;
    }

    // The entry a fresh copy of packet's shape and format is made into
    // (target): the active display, the other retained entry or a new one.
    // It becomes the active display only once that copy succeeded
    // (activate_display, depth_display_copy.inl), so a failed copy leaves
    // the active display, which holds the retained copy, and the display
    // cache as they were, and decide_failed_copy holds that copy (until
    // 11-01 a new render size or format switched the active display and
    // dropped the cache before the copy: a failed copy there was a mono
    // Present, F64). A successful copy's commit replaces the retained copy.
    display_status prepare_display(api::effect_runtime *runtime, state &data, const depth_capture::packet &packet,
        std::uint64_t present, state::display_entry *&target) {
      target = nullptr;
      // A new shape or format allocates here, on the present thread, and
      // retires the inactive entry (released off it, collect_displays).
      // Dynamic resolution makes that frequent, not rare (ACS 10-27b: 316
      // render-size changes in one session). Ordinary rotation at one shape
      // never allocates a texture or changes descriptors. A provider flip
      // (NGX and Streamline usually differ in shape or format) copies into
      // the other retained entry; the cache never reuses that entry's
      // earlier pixels.
      for (auto &entry : data.displays)
        if (entry.texture.handle && entry.width == packet.width && entry.height == packet.height &&
            entry.format == packet.format && entry.view_format == packet.srv_format) {
          target = &entry;
          return display_status::ready;
        }
      // An empty entry, else the inactive one. The active entry is retained.
      auto *victim = &data.displays[0];
      for (auto &entry : data.displays) {
        if (!entry.texture.handle) { victim = &entry; break; }
        if (entry.texture != data.display_texture) victim = &entry;
      }
      // Retiring the inactive entry changes no binding: the victim is never
      // the active entry, and the provider's binding (bound_view) is none or
      // the active entry's view, since display_view changes only when a copy
      // succeeded (activate_display), at an idle release and at teardown, and
      // begin() binds display_view (bind_api) before it returns. (Until 11-01
      // an unbind for an evicted bound view and a clear of an evicted active
      // entry stood here; neither could run.)
      if (victim->texture.handle) retire_display(data, *victim, present);
      api::resource_desc desc(packet.width, packet.height, 1, 1, static_cast<api::format>(packet.format), 1,
        api::memory_heap::default_, api::resource_usage::shader_resource | api::resource_usage::copy_dest);
      state::display_entry created;
      if (!runtime->get_device()->create_resource(desc, nullptr, api::resource_usage::shader_resource, &created.texture))
        return display_status::texture_creation_failed;
      api::resource_view_desc view(static_cast<api::format>(packet.srv_format));
      view.type = api::resource_view_type::texture_2d;
      if (!runtime->get_device()->create_resource_view(created.texture, api::resource_usage::shader_resource, view, &created.view)) {
        // Never read or bound: no GPU work references it yet.
        runtime->get_device()->destroy_resource(created.texture);
        return display_status::view_creation_failed;
      }
      created.width = packet.width; created.height = packet.height; created.format = packet.format;
      created.view_format = packet.srv_format;
      // A new entry counts as used from its creation (expire_idle_displays),
      // also while a failed copy left it inactive.
      created.used = sunshine_game3d::clock::game_now();
      *victim = created;
      target = victim;
      return display_status::ready;
    }
  }

  void initialize(api::effect_runtime *runtime) {
    auto *data = runtime->create_private_data<state>();
    // The retire fence comes with the state, never from a Present's first
    // retirement (a driver create on the present thread).
    if (data && runtime->get_device()->get_api() == api::device_api::d3d12)
      runtime->get_device()->create_fence(0, api::fence_flags::none, &data->retire_fence);
    // A runtime starts from the mode the game's options stand at, as
    // track_fg_switch reads it, so FG On set before the process's first
    // runtime is no switch (until 11-01 every runtime started at 0 and
    // logged mode=0->N with FG on, F100). At its first Present it takes
    // over the mode and FG switch line a torn-down runtime handed over (a
    // swapchain resize or recreation; adopt_fg_switch), so a reset with FG
    // unchanged is no switch and one across a switch keeps its line.
    frame_generation_snapshot fg;
    auto read = frame_generation_query_status::unavailable;
    query_frame_generation(UINT32_MAX, fg, &read);
    if (data && read == frame_generation_query_status::observed && fg.known) data->fg_mode = fg.mode;
  }

  void destroy(api::effect_runtime *runtime) {
    if (auto *data = runtime->get_private_data<state>()) {
      if (!sunshine_addon_lifetime::stopping()) hand_over_fg_switch(*data);
      // Every release job of this runtime finished first: none outlives it.
      wait_releases(*data->releases);
      const bool owned = std::any_of(data->displays.begin(), data->displays.end(),
        [](const auto &entry) { return entry.texture.handle != 0; }) || !data->retired_displays.empty();
      if (owned) runtime->get_command_queue()->wait_idle();
      destroy_display(runtime, *data);
    }
    runtime->destroy_private_data<state>();
  }
