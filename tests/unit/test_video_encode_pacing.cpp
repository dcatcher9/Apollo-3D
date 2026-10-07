#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <src/platform/windows/capture_timing.h>
#include <src/platform/windows/display.h>
#include <src/reshade_bridge_protocol.h>
#include <src/video.h>
#include <src/video_encode_pacing.h>
#include <src/video_encode_pipeline.h>

namespace {
  using namespace std::chrono_literals;

  // Model the image-event boundary with a deterministic clock. A depth completion does not
  // signal this event: only a new captured image can interrupt its wait, as in encode_run().
  struct image_event {
    std::chrono::nanoseconds now {};
    std::optional<std::chrono::nanoseconds> next_image;
    bool running = true;
    int polls = 0;

    std::optional<int> pop(std::chrono::nanoseconds wait) {
      ++polls;
      if (!running) {
        return std::nullopt;
      }
      if (next_image && *next_image <= now + wait) {
        now = std::max(now, *next_image);
        next_image.reset();
        return 42;
      }
      now += wait;
      return std::nullopt;
    }
  };

  TEST(RemoteEncodeCompletionTest, LastFrameCompletionDoesNotWaitForIdleHeartbeat) {
    image_event images;
    constexpr auto frame_interval = 16666666ns;
    constexpr auto idle_interval = 83333333ns;
    constexpr auto gpu_ready = 20ms;
    bool pending = true;
    while (pending) {
      ASSERT_FALSE(video::detail::wait_for_encode_image(
        images, idle_interval, frame_interval, true, false, pending
      ));
      pending = images.now < gpu_ready;
      ASSERT_LE(images.polls, 2);
    }
    EXPECT_EQ(images.now, 2 * frame_interval);
    EXPECT_LT(images.now, idle_interval);

    // Once the completion is consumed, there is no longer a reason to encode at stream cadence.
    video::detail::wait_for_encode_image(images, idle_interval, frame_interval, true, false, pending);
    EXPECT_EQ(images.now, 2 * frame_interval + idle_interval);
  }

  TEST(RemoteEncodeCompletionTest, NewCaptureInterruptsPendingWait) {
    image_event images;
    images.next_image = 3ms;
    EXPECT_EQ(video::detail::wait_for_encode_image(images, 83ms, 16ms, true, false, true), 42);
    EXPECT_EQ(images.now, 3ms);
  }

  TEST(RemoteEncodeCompletionTest, ReadyPipelineNeedsARealRetainedSource) {
    image_event images;
    video::detail::wait_for_encode_image(images, 83ms, 16ms, false, true, true);
    EXPECT_EQ(images.now, 83ms);
    video::detail::wait_for_encode_image(images, 83ms, 16ms, true, true, true);
    EXPECT_EQ(images.now, 83ms);
  }

  TEST(RemoteEncodeCompletionTest, BusyInferenceCannotSpinOrSuppressFasterKeepalives) {
    image_event images;
    for (int poll = 0; poll < 8; ++poll) {
      video::detail::wait_for_encode_image(images, 83ms, 0ns, true, false, true);
    }
    EXPECT_EQ(images.now, 8ms);
    video::detail::wait_for_encode_image(images, 2ms, 16ms, true, false, true);
    EXPECT_EQ(images.now, 10ms);
  }

  TEST(RemoteEncodeCompletionTest, CaptureShutdownReturnsWithoutWaitingOrProducingAnImage) {
    image_event images;
    images.running = false;
    EXPECT_FALSE(video::detail::wait_for_encode_image(images, 83ms, 16ms, true, false, true));
    EXPECT_EQ(images.now, 0ns);
  }

  struct captured_source {
    int pixels;
    std::chrono::steady_clock::time_point captured_at;
  };
  using source_owner = video::detail::latest_encode_source_t<std::shared_ptr<captured_source>>;

  std::chrono::steady_clock::time_point at(std::chrono::nanoseconds elapsed) {
    return std::chrono::steady_clock::time_point {elapsed};
  }

  TEST(RemoteEncodeProviderPacingTest, EncodeCostIsIncludedInThePresentationInterval) {
    source_owner source;
    source.observe(std::make_shared<captured_source>(1, at(0ns)));
    source.converted();
    image_event images;
    constexpr auto interval = 13888888ns; // 72 Hz
    constexpr auto encode_cost = 6700us;
    auto next_target = at(interval);
    images.now = encode_cost;
    for (int frame = 1; frame <= 6; ++frame) {
      ASSERT_FALSE(video::detail::wait_for_encode_image(
        images, 69ms, interval, true, false, true,
        source.remaining_wait(at(images.now), next_target, true)
      ));
      EXPECT_EQ(images.now, frame * interval);
      ASSERT_TRUE(source.due(at(images.now), next_target, false, true));
      const auto schedule = video::detail::select_encode_frame_schedule(
        at(images.now), next_target, interval, interval / 4
      );
      EXPECT_EQ(schedule.presentation_timestamp, at(frame * interval));
      next_target = schedule.next_encode_target;
      source.converted();
      images.now += encode_cost;
      EXPECT_EQ(source.remaining_wait(at(images.now), next_target, true), interval - encode_cost);
      EXPECT_FALSE(source.due(at(images.now), next_target, false, true));
    }
    EXPECT_EQ(images.polls, 6);
    EXPECT_EQ(source.latest()->captured_at, at(0ns)); // No invented source timestamp.
  }

  TEST(RemoteEncodeProviderPacingTest, EarlyCaptureSharesTheRetainedProviderDeadline) {
    source_owner source;
    source.observe(std::make_shared<captured_source>(1, at(0ns)));
    source.converted();
    image_event images;
    images.now = 6ms;
    images.next_image = 8ms;
    auto next_target = at(16ms);
    ASSERT_EQ(video::detail::wait_for_encode_image(
      images, 80ms, 16ms, true, false, true,
      source.remaining_wait(at(images.now), next_target, true)
    ), 42);
    source.observe(std::make_shared<captured_source>(2, at(images.now)));
    EXPECT_FALSE(source.due(at(images.now), next_target));
    ASSERT_FALSE(video::detail::wait_for_encode_image(
      images, 80ms, 16ms, true, false, true,
      source.remaining_wait(at(images.now), next_target, true)
    ));
    EXPECT_EQ(images.now, 16ms); // Not 8+16: no second full-frame wait.
    ASSERT_TRUE(source.due(at(images.now), next_target));
    const auto schedule = video::detail::select_encode_frame_schedule(at(images.now), next_target, 16ms, 4ms);
    next_target = schedule.next_encode_target;
    source.converted();
    images.now += 6ms;
    EXPECT_EQ(source.remaining_wait(at(images.now), next_target, true), 10ms);
    EXPECT_FALSE(source.pending());
  }

  TEST(RemoteEncodeProviderPacingTest, OverduePollRebasesAndCannotBusyLoop) {
    source_owner source;
    source.observe(std::make_shared<captured_source>(1, at(0ns)));
    source.converted();
    image_event images;
    images.now = 40ms; // Previous encode exceeded the interval.
    auto next_target = at(16ms);
    EXPECT_EQ(source.remaining_wait(at(images.now), next_target, true), 0ns);
    EXPECT_TRUE(source.due(at(images.now), next_target, false, true));
    const auto schedule = video::detail::select_encode_frame_schedule(at(images.now), next_target, 16ms, 4ms);
    EXPECT_EQ(schedule.presentation_timestamp, at(40ms));
    next_target = schedule.next_encode_target;
    source.converted();
    EXPECT_FALSE(source.due(at(images.now), next_target, false, true));
    EXPECT_EQ(source.remaining_wait(at(images.now), next_target, true), 16ms);
  }

  TEST(RemoteEncodeProviderPacingTest, FasterHeartbeatDoesNotAdvanceProviderAndOptOutIsUnchanged) {
    source_owner source;
    EXPECT_FALSE(source.remaining_wait(at(0ns), at(16ms), true));
    EXPECT_FALSE(source.due(at(20ms), at(16ms), true, true));
    source.observe(std::make_shared<captured_source>(1, at(0ns)));
    source.converted();
    image_event images;
    images.now = 6ms;
    const auto next_target = at(16ms);
    ASSERT_FALSE(video::detail::wait_for_encode_image(
      images, 2ms, 16ms, true, false, true,
      source.remaining_wait(at(images.now), next_target, true)
    ));
    EXPECT_EQ(images.now, 8ms);
    EXPECT_FALSE(source.due(at(images.now), next_target, false, true));
    EXPECT_EQ(source.remaining_wait(at(images.now), next_target, true), 8ms);
    EXPECT_TRUE(source.due(at(images.now), next_target, true, true)); // IDR may service now.
    EXPECT_FALSE(source.remaining_wait(at(images.now), next_target));
    EXPECT_FALSE(source.due(at(20ms), next_target));
    video::detail::wait_for_encode_image(images, 80ms, 16ms, true, false, true);
    EXPECT_EQ(images.now, 24ms); // Existing Host AI completion wait remains cadence-sized.
  }

  TEST(RemoteEncodeProviderPacingTest, NonpollingBackendDoesNotSpinAtExpiredProviderDeadline) {
    source_owner source;
    source.observe(std::make_shared<captured_source>(1, at(0ns)));
    source.converted();
    image_event images;
    images.now = 40ms;
    const auto expired_target = at(16ms);
    constexpr bool independent_provider = true;
    constexpr bool conversion_poll_pending = false;
    for (int heartbeat = 0; heartbeat < 3; ++heartbeat) {
      const auto wait = source.remaining_wait(
        at(images.now),
        expired_target,
        independent_provider && conversion_poll_pending
      );
      EXPECT_FALSE(wait);
      EXPECT_FALSE(video::detail::wait_for_encode_image(images, 80ms, 16ms, true, false, conversion_poll_pending, wait));
    }
    EXPECT_EQ(images.now, 280ms);

    // Ordinary pending capture work still has its deadline, even without provider polling.
    source.observe(std::make_shared<captured_source>(2, at(images.now)));
    EXPECT_EQ(source.remaining_wait(at(images.now), expired_target, independent_provider && conversion_poll_pending), 0ns);
    EXPECT_TRUE(source.due(at(images.now), expired_target));
  }

  TEST(RemoteEncodeProviderPacingTest, UnchangedMetadataCaptureNeitherSchedulesNorCancelsConversion) {
    source_owner source;
    source.observe(std::make_shared<captured_source>(1, at(0ns)));
    source.converted();
    // Metadata for pixels a live export owns, with an unchanged cursor: retained, not converted.
    source.observe(std::make_shared<captured_source>(2, at(1ms)), false);
    EXPECT_EQ(source.latest()->pixels, 2);
    EXPECT_FALSE(source.pending());
    EXPECT_FALSE(source.remaining_wait(at(1ms), at(16ms)));
    EXPECT_FALSE(source.due(at(20ms), at(16ms)));
    // A conversion still owed for an earlier capture survives a later unchanged one, which then
    // supplies the newest cursor metadata.
    source.observe(std::make_shared<captured_source>(3, at(2ms)));
    source.observe(std::make_shared<captured_source>(4, at(3ms)), false);
    EXPECT_TRUE(source.pending());
    EXPECT_EQ(source.latest()->pixels, 4);
  }

  TEST(RemoteEncodeProviderPacingTest, KeepaliveIsMeasuredFromTheLastEncode) {
    EXPECT_EQ(video::detail::provider_keepalive_wait(at(10ms), at(0ns), 55ms), 45ms);
    EXPECT_EQ(video::detail::provider_keepalive_wait(at(55ms), at(0ns), 55ms), 0ns);
    EXPECT_EQ(video::detail::provider_keepalive_wait(at(80ms), at(0ns), 55ms), 0ns);
  }

  // The provider wake path of encode_run(): an export is converted when it is ready, or once the
  // time reaches its target minus the variation threshold, and a newer export supersedes one that
  // is still waiting. Returns the wait of each converted export.
  std::vector<std::chrono::nanoseconds> woken_provider_waits(
    std::chrono::nanoseconds game_interval,
    std::chrono::nanoseconds stream_interval,
    std::chrono::nanoseconds phase,
    int exports
  ) {
    const auto threshold = stream_interval / 4;
    source_owner source;
    source.observe(std::make_shared<captured_source>(1, at(0ns)));
    source.converted();
    auto target = at(0ns);
    std::chrono::nanoseconds now {};
    std::optional<std::chrono::nanoseconds> ready;
    std::vector<std::chrono::nanoseconds> waits;
    int next = 0;
    while (next < exports || ready) {
      const auto arrival = phase + next * game_interval;
      if (!ready) {
        now = arrival;
        ready = arrival;
        ++next;
        continue;
      }
      const auto poll_target = target - threshold;
      const auto due_at = std::max(now, poll_target.time_since_epoch());
      if (next < exports && arrival <= due_at) {
        now = arrival;
        ready = arrival;  // The waiting export is superseded, never queued.
        ++next;
        continue;
      }
      now = due_at;
      EXPECT_TRUE(source.due(at(now), poll_target, false, true));
      EXPECT_EQ(source.remaining_wait(at(now), poll_target, true), 0ns);
      const auto schedule = video::detail::select_encode_frame_schedule(at(now), target, stream_interval, threshold);
      EXPECT_GE(schedule.next_encode_target - target, stream_interval);  // Never above stream rate.
      target = schedule.next_encode_target;
      source.converted();
      waits.push_back(now - *ready);
      ready.reset();
    }
    return waits;
  }

  TEST(RemoteEncodeProviderPacingTest, WokenProviderFramesConvertWhenReadyAtOrBelowStreamRate) {
    constexpr auto stream = 11111111ns;  // 90 Hz
    for (const std::chrono::nanoseconds phase : {0ns, 3000000ns, 7000000ns, 10000000ns}) {
      // A game at or below the stream rate is converted the moment each export is ready, in any
      // phase relative to the stream's earlier deadlines.
      for (const std::chrono::nanoseconds game : {stream, 16666666ns, 33333333ns}) {
        const auto waits = woken_provider_waits(game, stream, phase, 60);
        ASSERT_EQ(waits.size(), 60u);
        for (const auto wait : waits) {
          EXPECT_EQ(wait, 0ns);
        }
      }
    }
  }

  TEST(RemoteEncodeProviderPacingTest, FasterGameIsCappedToTheStreamRateWithBoundedWaits) {
    constexpr auto stream = 11111111ns;  // 90 Hz
    constexpr auto game = 8333333ns;  // 120 fps
    const auto waits = woken_provider_waits(game, stream, 1ms, 120);  // 1 s of exports
    // At most one conversion per stream interval on average. A converted export never waited a
    // whole game frame: a newer export would have superseded it.
    EXPECT_LE(waits.size(), 91u);
    EXPECT_GE(waits.size(), 85u);
    std::chrono::nanoseconds total {};
    for (const auto wait : waits) {
      EXPECT_LT(wait, game);
      total += wait;
    }
    EXPECT_LT(total / static_cast<int>(waits.size()), stream / 2);
  }

  // Live Game 3D evidence (Stellar Blade 4K HDR, 90 fps stream, 10-06): the add-on published
  // 77-171 frames/s into its three-slot export ring, yet the encode loop claimed only 56-62 new
  // frames/s while the add-on replaced 25-109 finished frames/s that the host never claimed. The
  // export's fence wakes arrived on time (90-130/s, and almost no claim found its frame before its
  // wake). The loop itself was held to the Windows scheduler tick: every iteration started with a
  // zero-timeout pop of an empty control queue, and on this toolchain such a condition-variable wait
  // ends only at the next 15.625 ms tick (safe::no_wait()), as does any timed image wait that
  // nothing notifies. This model runs the loop's own wait and schedule helpers against the export
  // ring's rules (docs/reshade-sbs.md, GPU handoff contract) with that arrival pattern and with
  // waits that take as long as they really do.
  enum class game_frames_e {
    fg_off,  ///< ~115 fps.
    fg_2x,  ///< ~72 real fps, ~145 Presents/s.
    fg_4x,  ///< ~55 real fps, ~220 Presents/s.
    slow,  ///< ~40 fps, below the stream rate.
    /** FG 4x GPU-bound (Stellar Blade 10-06 15:39:44-15:40:00): ~24 real fps, ~95 Presents/s, each
     *  pack's fence completing 15-35 ms after its Present on the busy queue. */
    fg_4x_gpu_bound,
  };

  // GetTickCount64's period. winpthreads (no clock-based condition-variable wait in this libstdc++)
  // re-waits until that count passes a timed wait's deadline, so even a zero timeout ends only at
  // the next tick; timeBeginPeriod and NtSetTimerResolution do not change it.
  constexpr auto windows_tick = 15625us;
  // A high-resolution waitable timer ends a 1 ms sleep after 1.4-1.55 ms on average.
  constexpr auto hold_timer_overshoot = 500us;

  // A condition-variable wait that nothing notifies ends at the first Windows scheduler tick at or
  // after its deadline, counted from the tick before it started (winpthreads without a clock-based
  // wait), whatever the timer resolution. The models' ticks run at this phase of their clock.
  constexpr auto tick_phase = 3100us;

  std::chrono::nanoseconds last_tick(std::chrono::nanoseconds t) {
    constexpr auto offset = windows_tick - tick_phase;
    return ((t + offset) / windows_tick) * windows_tick - offset;
  }

  std::chrono::nanoseconds tick_wait_end(std::chrono::nanoseconds start, std::chrono::nanoseconds wait) {
    const auto deadline = last_tick(start) + std::chrono::ceil<std::chrono::milliseconds>(std::max(wait, 0ns));
    auto end = last_tick(start) + windows_tick;
    while (end < deadline) {
      end += windows_tick;
    }
    return end;
  }

  enum class fence_wake_e {
    prompt,
    late,  ///< 0-11 ms after the completion.
    lost,
  };

  // Which encode loop runs.
  enum class loop_rules_e {
    /** Until 10-06: each iteration began with stream_gamma_requests->pop(0ms), a wait to the next
     *  tick; a pending frame waited on the image event for its poll target, and a held export was
     *  re-checked every millisecond from there (export_recheck_wait), both tick-bound. */
    head,
    /** Zero-timeout pops only check; a pending frame is held exactly to its poll target
     *  (provider_hold()); every other wait ends at a wake, capture or its keepalive bound. */
    production,
  };

  // The removed re-check bound of a held export (58449a17), for the head rules only.
  std::chrono::nanoseconds head_export_recheck_wait(std::chrono::nanoseconds now, std::chrono::nanoseconds poll_target) {
    return std::max(poll_target - now, std::chrono::nanoseconds {1ms});
  }

  // Fixed pseudo-random jitter: the same sequence with every standard library.
  struct sim_jitter_t {
    std::uint32_t state = 12345;

    std::chrono::nanoseconds between(double low_ms, double high_ms) {
      state = state * 1664525u + 1013904223u;
      const double unit = static_cast<double>(state >> 8) / 16777216.0;
      return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double, std::milli>(low_ms + (high_ms - low_ms) * unit));
    }
  };

  // Generated Presents are spaced evenly within each real frame.
  std::vector<std::chrono::nanoseconds> game_presents(game_frames_e mode, std::chrono::nanoseconds duration, sim_jitter_t &jitter) {
    std::vector<std::chrono::nanoseconds> presents;
    std::chrono::nanoseconds real {};
    while (real < duration) {
      if (mode == game_frames_e::fg_off || mode == game_frames_e::slow) {
        real += mode == game_frames_e::slow ? jitter.between(23.0, 27.0) : jitter.between(7.5, 10.0);
        presents.push_back(real);
        continue;
      }
      const int generated = mode == game_frames_e::fg_2x ? 2 : 4;
      const auto length = mode == game_frames_e::fg_2x ? jitter.between(12.5, 15.0) :
                          mode == game_frames_e::fg_4x ? jitter.between(16.0, 20.5) :
                                                         jitter.between(38.0, 46.0);
      for (int i = 0; i < generated; ++i) {
        presents.push_back(real + length * i / generated);
      }
      real += length;
    }
    return presents;
  }

  struct export_ring_result_t {
    double presents = 0, published = 0, dropped = 0, overwritten = 0, new_frames = 0;  ///< Per second.
    int repeats = 0;
    std::chrono::nanoseconds max_gap {};  ///< Between new frames.
    std::chrono::nanoseconds mean_age {};  ///< From a claimed frame's Present to its claim.
    std::chrono::nanoseconds mean_claim_delay {}, max_claim_delay {};  ///< From its fence completion.
    std::chrono::nanoseconds mean_encode {};  ///< From a new frame's claim to its packet.
    std::chrono::nanoseconds mean_present_to_packet {};  ///< From a new frame's Present to its packet.
    unsigned max_in_flight = 0;
  };

  // The encoder behind encode_run(). Unset: the encode loop blocks for conversion and NVENC
  // (8.4-10.4 ms), the model the rules above were tuned against. Set: the conversion and picture
  // submission cost the loop `cpu`; the picture completes `latency_low_ms`-`latency_high_ms` after
  // its submission (the conversion's GPU work queued behind the game's, then NVENC), in order and
  // at least `engine` after the previous one; the conversion's GPU work, and so its reads of the
  // export slot, completes `engine` before that latency ends. With `depth` 1 the loop waits for the
  // completion (nvenc_base::encode_frame()); with more, the loop submits another picture at once
  // and waits only when `depth` are in flight (encode_pipeline_t), and the packet leaves when its
  // picture completes.
  struct encoder_model_t {
    unsigned depth = 1;
    double latency_low_ms = 0, latency_high_ms = 0;
    std::chrono::nanoseconds cpu {400us};
    std::chrono::nanoseconds engine {5ms};
    /** The tail: every `tail_every`-th picture takes `latency_high_ms`-`tail_high_ms` instead. */
    unsigned tail_every = 0;
    double tail_high_ms = 0;

    /** When a slot replaced while its conversion's reads still ran returns to the producer. */
    enum class retire_e {
      reads_complete,  ///< As the GPU completes them: the receiver's read-fence wait.
      encode_thread,  ///< At the encode thread's first check after that: the next submission or conversion (5798c45f).
    } retire = retire_e::reads_complete;

    /** With every picture in flight, the loop waits for the oldest before it claims, instead of
     *  claiming and converting at its poll target and then waiting (encode_run()). */
    bool wait_before_claim = false;
  };

  // The export ring and encode_run() while an independent provider's export is live, stepped every
  // 10 us. Producer, per Present: a free slot whose write completed, else the oldest ready slot
  // whose write completed and that a newer completed frame supersedes, else the Present drops.
  // Host: claims the newest ready frame past its held one whose fence passed and returns the
  // replaced slot once its conversion's reads completed (with `encoder`, by its `retire` rule;
  // without, they completed during the previous encode). Its wait composes
  // remaining_wait(), provider_keepalive_wait() and, by `rules`, provider_hold() as encode_run()
  // does; a fence wake or desktop capture ends an image wait early, and one that arrives while the
  // loop is busy ends the next image wait at once (event_t). After the wait a pending frame
  // converts once due, a due keepalive repeats the input, and anything else encodes nothing.
  // Waits take as long as they really do with this toolchain: a condition-variable wait that
  // nothing notifies ends at a Windows scheduler tick (windows_tick). The ring has `slots` slots:
  // the protocol's, or three for the ring before protocol 4.
  export_ring_result_t simulate_export_ring(game_frames_e mode, fence_wake_e wake, loop_rules_e rules, std::chrono::nanoseconds duration = 4s, std::optional<encoder_model_t> encoder = std::nullopt, int slots = static_cast<int>(::reshade_bridge::slot_count)) {
    constexpr auto stream = 11111111ns;  // 90 fps
    constexpr auto threshold = stream / 4;
    constexpr auto keepalive = 55555555ns;  // The 18 fps minimum of a 90 fps stream.
    constexpr auto step = 10us;

    enum class slot_e {
      free,
      ready,
      reading,
    };

    struct slot_t {
      slot_e state = slot_e::free;
      std::size_t sequence = 0;  ///< 1-based publication index; also its fence value.
    };

    sim_jitter_t jitter;
    sim_jitter_t encode_jitter {54321};  // Its own sequence: the game's arrivals stay identical.
    std::deque<std::chrono::nanoseconds> in_flight;  // Completion times, oldest first.
    std::chrono::nanoseconds last_completion {}, total_encode {}, total_present_to_packet {};
    const auto presents = game_presents(mode, duration, jitter);
    std::vector<std::chrono::nanoseconds> captures;
    for (auto capture = jitter.between(0.0, 11.1); capture < duration; capture += jitter.between(10.1, 12.1)) {
      captures.push_back(capture);
    }
    std::vector<slot_t> ring(static_cast<std::size_t>(slots));
    std::vector<std::chrono::nanoseconds> present_of {0ns}, completion_of {0ns}, wakes;
    std::size_t next_present = 0, next_capture = 0, completed = 0, held = 0;
    int held_slot = -1, next_slot = 0, published = 0, dropped = 0, overwritten = 0, new_frames = 0, encodes = 0;
    export_ring_result_t result;
    std::chrono::nanoseconds total_age {}, total_claim_delay {};
    std::optional<std::chrono::nanoseconds> last_new;
    // Slots replaced while the reads of their last conversion still ran, and when those completed.
    struct retiring_slot_t {
      int slot;
      std::size_t sequence;
      std::chrono::nanoseconds reads_done;
    };

    std::vector<retiring_slot_t> retiring;
    std::chrono::nanoseconds held_reads_done {};  // The last conversion that read the held slot.
    std::deque<std::chrono::nanoseconds> submissions;  // Encode-thread checks still to come.
    const auto return_read_slots = [&](std::chrono::nanoseconds at) {
      std::erase_if(retiring, [&](const retiring_slot_t &replaced) {
        if (replaced.reads_done > at) {
          return false;
        }
        if (ring[replaced.slot].state == slot_e::reading && ring[replaced.slot].sequence == replaced.sequence) {
          ring[replaced.slot].state = slot_e::free;
        }
        return true;
      });
    };

    source_owner source;
    source.observe(std::make_shared<captured_source>(1, at(0ns)));
    source.converted();
    auto target = at(0ns);
    std::chrono::nanoseconds last_encode {}, blocked_until {}, wait_until {};
    // The loop's phase once blocked_until passed (an encode, a hold, or a wait no wake can end):
    // an iteration starting, its wait being planned, the wait on the image event, or a hold ended.
    enum class loop_e {
      start,
      planning,
      waiting,
      held,
    } loop = loop_e::start;
    bool woken = false;
    const auto frame_pending = [&]() {
      for (const auto &slot : ring) {
        if (slot.state == slot_e::ready && slot.sequence > held && slot.sequence <= completed) {
          return true;
        }
      }
      return false;
    };

    for (std::chrono::nanoseconds now {}; now < duration; now += step) {
      // Replaced slots whose reads completed: at once by the read-fence wait, else at the encode
      // thread's checks (each submission here; each conversion below).
      if (!encoder || encoder->retire == encoder_model_t::retire_e::reads_complete) {
        return_read_slots(now);
      }
      for (; !submissions.empty() && submissions.front() <= now; submissions.pop_front()) {
        return_read_slots(submissions.front());
      }
      // Producer.
      for (; next_present < presents.size() && presents[next_present] <= now; ++next_present) {
        std::size_t newest_completed = 0;
        for (const auto &slot : ring) {
          if (slot.state != slot_e::free && slot.sequence <= completed) {
            newest_completed = std::max(newest_completed, slot.sequence);
          }
        }
        int index = -1;
        for (int offset = 0; offset < slots && index < 0; ++offset) {
          const int i = (next_slot + offset) % slots;
          if (ring[i].state == slot_e::free && ring[i].sequence <= completed) {
            index = i;
          }
        }
        for (int i = 0; i < slots && index < 0; ++i) {
          if (ring[i].state == slot_e::ready && ring[i].sequence < newest_completed) {
            index = i;
            for (int other = 0; other < slots; ++other) {
              if (ring[other].state == slot_e::ready && ring[other].sequence < ring[index].sequence) {
                index = other;
              }
            }
            ++overwritten;
          }
        }
        if (index < 0) {
          ++dropped;
          continue;
        }
        ring[index] = {slot_e::ready, static_cast<std::size_t>(++published)};
        next_slot = (index + 1) % slots;
        present_of.push_back(presents[next_present]);
        const auto pack = mode == game_frames_e::fg_4x_gpu_bound ? jitter.between(15.0, 35.0) : jitter.between(5.0, 15.0);
        completion_of.push_back(std::max(completion_of.back(), presents[next_present] + pack));
        if (wake != fence_wake_e::lost) {
          wakes.push_back(completion_of.back() + (wake == fence_wake_e::late ? jitter.between(0.0, 11.1) : 0ns));
        }
      }
      // Fence and wakes.
      while (completed + 1 < completion_of.size() && completion_of[completed + 1] <= now) {
        ++completed;
      }
      for (; next_capture < captures.size() && captures[next_capture] <= now; ++next_capture) {
        woken = true;
      }
      for (auto wake_at = wakes.begin(); wake_at != wakes.end();) {
        if (*wake_at <= now) {
          woken = true;
          wake_at = wakes.erase(wake_at);
        } else {
          ++wake_at;
        }
      }
      // Encode loop.
      if (now < blocked_until) {
        continue;
      }
      if (loop == loop_e::start) {
        if (encoder && encoder->depth > 1 && encoder->wait_before_claim) {
          // Every picture in flight: wait for the oldest before planning the claim.
          while (!in_flight.empty() && in_flight.front() <= now) {
            in_flight.pop_front();
          }
          if (in_flight.size() >= encoder->depth) {
            blocked_until = in_flight.front();
            continue;
          }
        }
        loop = loop_e::planning;
        if (rules == loop_rules_e::head) {
          // stream_gamma_requests->pop(0ms) on its empty queue: a wait that nothing notifies.
          blocked_until = tick_wait_end(now, 0ns);
          continue;
        }
      }
      if (loop == loop_e::planning) {
        const auto poll_target = target - threshold;
        const auto pending_wait = source.remaining_wait(at(now), poll_target, frame_pending());
        const auto keepalive_wait = video::detail::provider_keepalive_wait(at(now), at(last_encode), keepalive);
        if (rules == loop_rules_e::head) {
          const auto bound = std::min(keepalive_wait, head_export_recheck_wait(now, poll_target.time_since_epoch()));
          wait_until = tick_wait_end(now, pending_wait ? std::min(*pending_wait, bound) : bound);
          loop = loop_e::waiting;
        } else if (const auto hold = video::detail::provider_hold(pending_wait, keepalive_wait, stream)) {
          // No wake ends the hold; the zero-timeout image pop after it consumes any that came.
          blocked_until = now + *hold + hold_timer_overshoot;
          loop = loop_e::held;
          continue;
        } else {
          const auto wait = pending_wait ? std::min(*pending_wait, keepalive_wait) : keepalive_wait;
          wait_until = wait > 0ns ? tick_wait_end(now, wait) : now;
          loop = loop_e::waiting;
        }
      }
      if (loop == loop_e::waiting && !woken && now < wait_until) {
        continue;
      }
      woken = false;
      loop = loop_e::start;
      const auto poll_target = target - threshold;
      int selected = -1;
      for (int i = 0; i < slots; ++i) {
        if (ring[i].state == slot_e::ready && ring[i].sequence > held && ring[i].sequence <= completed && (selected < 0 || ring[i].sequence > ring[selected].sequence)) {
          selected = i;
        }
      }
      const bool keepalive_due = now >= last_encode + keepalive;
      if ((selected >= 0 || keepalive_due) && source.due(at(now), poll_target, false, true)) {
        target = video::detail::select_encode_frame_schedule(at(now), target, stream, threshold).next_encode_target;
        if (selected >= 0) {
          // The conversion's poll first returns replaced slots whose reads completed; the claim
          // then returns the held slot at once if its reads completed, else once they do.
          return_read_slots(now);
          if (held_slot >= 0) {
            if (held_reads_done <= now) {
              ring[held_slot].state = slot_e::free;
            } else {
              retiring.push_back({held_slot, held, held_reads_done});
            }
          }
          ring[selected].state = slot_e::reading;
          held_slot = selected;
          held = ring[selected].sequence;
          total_age += now - present_of[held];
          total_claim_delay += now - completion_of[held];
          result.max_claim_delay = std::max(result.max_claim_delay, now - completion_of[held]);
          if (last_new) {
            result.max_gap = std::max(result.max_gap, now - *last_new);
          }
          last_new = now;
          ++new_frames;
        } else {
          ++result.repeats;
        }
        if (!encoder) {
          blocked_until = now + 8400us + 1ms * (encodes++ % 3);  // Conversion and NVENC: 8.4-10.4 ms.
          last_encode = blocked_until;
          held_reads_done = blocked_until;
          continue;
        }
        // Retired pictures have left as packets.
        while (!in_flight.empty() && in_flight.front() <= now) {
          in_flight.pop_front();
        }
        auto submitted = now + encoder->cpu;
        if (in_flight.size() >= encoder->depth) {
          submitted = std::max(submitted, in_flight.front());  // acquire(): wait for the oldest.
          in_flight.pop_front();
        }
        const bool tail = encoder->tail_every && encodes % encoder->tail_every == encoder->tail_every - 1;
        const auto latency = tail ? encode_jitter.between(encoder->latency_high_ms, encoder->tail_high_ms) : encode_jitter.between(encoder->latency_low_ms, encoder->latency_high_ms);
        const auto completion = std::max(submitted + latency, last_completion + encoder->engine);
        last_completion = completion;
        held_reads_done = submitted + std::max(latency - encoder->engine, 0ns);
        submissions.push_back(submitted);
        in_flight.push_back(completion);
        result.max_in_flight = std::max(result.max_in_flight, static_cast<unsigned>(in_flight.size()));
        if (selected >= 0) {
          total_encode += completion - now;
          total_present_to_packet += completion - present_of[held];
        }
        // One picture at a time: encode_frame() returns once it completed.
        blocked_until = encoder->depth == 1 ? completion : submitted;
        last_encode = blocked_until;
        ++encodes;
      }
    }
    const double seconds = std::chrono::duration<double>(duration).count();
    result.presents = static_cast<double>(presents.size()) / seconds;
    result.published = published / seconds;
    result.dropped = dropped / seconds;
    result.overwritten = overwritten / seconds;
    result.new_frames = new_frames / seconds;
    result.mean_age = new_frames ? total_age / new_frames : 0ns;
    result.mean_claim_delay = new_frames ? total_claim_delay / new_frames : 0ns;
    result.mean_encode = new_frames ? total_encode / new_frames : 0ns;
    result.mean_present_to_packet = new_frames ? total_present_to_packet / new_frames : 0ns;
    return result;
  }

  void print_export_ring(const char *label, game_frames_e mode, fence_wake_e wake, const export_ring_result_t &result) {
    constexpr const char *modes[] {"FG off", "FG 2x", "FG 4x", "40 fps", "FG 4x GPU-bound"};
    constexpr const char *wakes[] {"prompt", "late", "lost"};
    std::printf(
      "[ MEASURE  ] %s, %s, %s fence wake: presents %.0f/s published %.0f/s dropped %.0f/s overwritten %.0f/s new %.1f/s repeats %d, "
      "mean claim age %.1f ms, completion to claim mean %.2f max %.2f ms, max gap %.1f ms\n",
      label,
      modes[static_cast<int>(mode)],
      wakes[static_cast<int>(wake)],
      result.presents,
      result.published,
      result.dropped,
      result.overwritten,
      result.new_frames,
      result.repeats,
      std::chrono::duration<double, std::milli>(result.mean_age).count(),
      std::chrono::duration<double, std::milli>(result.mean_claim_delay).count(),
      std::chrono::duration<double, std::milli>(result.max_claim_delay).count(),
      std::chrono::duration<double, std::milli>(result.max_gap).count()
    );
  }

  TEST(RemoteEncodeProviderPacingTest, TickBoundWaitsReproduceTheLiveSixtyFrameCap) {
    // The head rules against the live arrival patterns: the model must land where the live host
    // did (56-62 new frames/s), which is what makes the production result below evidence. Those
    // runs used the three-slot ring of export protocols 1-3, so the calibration runs on three.
    for (const auto mode : {game_frames_e::fg_off, game_frames_e::fg_2x, game_frames_e::fg_4x}) {
      const auto result = simulate_export_ring(mode, fence_wake_e::prompt, loop_rules_e::head, 4s, std::nullopt, 3);
      print_export_ring("head", mode, fence_wake_e::prompt, result);
      EXPECT_GE(result.new_frames, 54.0);
      EXPECT_LE(result.new_frames, 64.0);  // At most one iteration per 15.625 ms tick.
      EXPECT_GE(result.max_gap, 2 * windows_tick);
    }
  }

  TEST(RemoteEncodeProviderPacingTest, LiveExportDeliversNewFramesAtStreamRate) {
    for (const auto mode : {game_frames_e::fg_off, game_frames_e::fg_2x, game_frames_e::fg_4x}) {
      for (const auto wake : {fence_wake_e::prompt, fence_wake_e::late, fence_wake_e::lost}) {
        const auto result = simulate_export_ring(mode, wake, loop_rules_e::production);
        print_export_ring("production", mode, wake, result);
        EXPECT_LE(result.new_frames, 90.5);
        EXPECT_EQ(result.repeats, 0);
        if (wake == fence_wake_e::prompt) {
          // Every mode presents faster than the stream, so min(Present rate, 90) is 90. The live
          // wakes were prompt (10-06: almost no claim found its frame before its wake).
          EXPECT_GE(result.new_frames, 87.5);
          EXPECT_LT(result.max_gap, 2 * 11111111ns);  // Never a skipped stream frame.
        } else {
          // Nothing re-checks for a missing wake: a frame that completed during the previous
          // encode is still found by the next iteration's poll, and the others by the next
          // ~90 Hz desktop capture, so the stream degrades by at most a frame now and then.
          EXPECT_GE(result.new_frames, 80.0);
          EXPECT_LT(result.max_gap, 3 * 11111111ns);
        }
      }
    }
  }

  TEST(RemoteEncodeProviderPacingTest, SlowerGameIsClaimedWhenItsFenceWakes) {
    // Below the stream rate every frame is new and is claimed as its wake arrives: no poll, tick
    // or re-check stands between a finished frame and its conversion.
    const auto result = simulate_export_ring(game_frames_e::slow, fence_wake_e::prompt, loop_rules_e::production);
    print_export_ring("production", game_frames_e::slow, fence_wake_e::prompt, result);
    EXPECT_GE(result.new_frames, result.presents - 1.0);
    EXPECT_EQ(result.repeats, 0);
    EXPECT_LT(result.max_claim_delay, 1ms);
    const auto head = simulate_export_ring(game_frames_e::slow, fence_wake_e::prompt, loop_rules_e::head);
    print_export_ring("head", game_frames_e::slow, fence_wake_e::prompt, head);
    EXPECT_GT(head.mean_claim_delay, 4ms);  // Each claim waited for the control poll's tick.
  }

  void print_encoder_model(const char *label, game_frames_e mode, const export_ring_result_t &result) {
    constexpr const char *modes[] {"FG off", "FG 2x", "FG 4x", "40 fps", "FG 4x GPU-bound"};
    const auto ms = [](std::chrono::nanoseconds value) {
      return std::chrono::duration<double, std::milli>(value).count();
    };
    std::printf(
      "[ MEASURE  ] %s, %s: new %.1f/s repeats %d, dropped Presents %.0f/s, at most %u in flight, claim to packet mean %.2f ms, "
      "Present to packet mean %.2f ms, mean claim age %.1f ms, max gap %.1f ms\n",
      label,
      modes[static_cast<int>(mode)],
      result.new_frames,
      result.repeats,
      result.dropped,
      result.max_in_flight,
      ms(result.mean_encode),
      ms(result.mean_present_to_packet),
      ms(result.mean_age),
      ms(result.max_gap)
    );
  }

  TEST(RemoteEncodeProviderPacingTest, PicturesInFlightTakeEveryStreamFrameAtFrameGenerationEncodeTimes) {
    // Live 10-06 (Stellar Blade 4K HDR, 90 fps stream), after the loop's waits became exact: with
    // FG off the host took 80-90 new frames/s, with FG on only 55-71/s, while encoding filled 10.4
    // to 17.2 s of every 20 s. NVENC's completion wait averaged 8.5-12.2 ms per picture (about 5 ms
    // uncontended at 7680x2160 10-bit HEVC; the rest is the conversion's GPU work queued behind the
    // game's), 5.4 ms at least and with a long tail. encode_frame() submitted a picture and waited
    // for it, so the next frame could not even be converted meanwhile: one picture per encode.
    struct encode_times_t {
      const char *label;
      double low_ms, high_ms;
      double serial_low, serial_high;  ///< New frames/s one picture at a time, three slots.
    };

    for (const auto &times : {
           encode_times_t {"5.5-22 ms (the live FG-on spread)", 5.5, 22.0, 60.0, 72.0},
           encode_times_t {"12 ms", 12.0, 12.0, 76.0, 81.0},  // At most one per 12.4 ms.
         }) {
      for (const auto mode : {game_frames_e::fg_2x, game_frames_e::fg_4x}) {
        SCOPED_TRACE(times.label);
        // The one-picture band was fitted to live runs on the three-slot ring of export protocols
        // 1-3, so it is checked there; the comparisons below run both models on this protocol's ring.
        const auto live_ring = simulate_export_ring(mode, fence_wake_e::prompt, loop_rules_e::production, 4s, encoder_model_t {1, times.low_ms, times.high_ms}, 3);
        const auto serial = simulate_export_ring(mode, fence_wake_e::prompt, loop_rules_e::production, 4s, encoder_model_t {1, times.low_ms, times.high_ms});
        const auto pipelined = simulate_export_ring(mode, fence_wake_e::prompt, loop_rules_e::production, 4s, encoder_model_t {2, times.low_ms, times.high_ms});
        std::printf("[ MEASURE  ] encodes of %s:\n", times.label);
        print_encoder_model("  one picture at a time, three slots", mode, live_ring);
        print_encoder_model("  one picture at a time", mode, serial);
        print_encoder_model("  two pictures in flight", mode, pipelined);
        EXPECT_GE(live_ring.new_frames, times.serial_low);
        EXPECT_LE(live_ring.new_frames, times.serial_high);
        EXPECT_GE(pipelined.new_frames, 87.5);  // Every stream frame, as with FG off.
        EXPECT_LE(pipelined.new_frames, 90.5);
        EXPECT_EQ(pipelined.repeats, 0);
        EXPECT_LT(pipelined.max_gap, 2 * 11111111ns);
        EXPECT_EQ(pipelined.max_in_flight, 2u);
        // A frame starts encoding as soon as it is converted and its packet leaves as soon as its
        // picture completes. Only NVENC's own order remains: a quick picture right behind a slow
        // one completes after it, a fraction of a millisecond on average.
        EXPECT_LE(pipelined.mean_encode, serial.mean_encode + 1ms);
        EXPECT_LE(pipelined.mean_present_to_packet, serial.mean_present_to_packet + 500us);
      }
    }
  }

  TEST(RemoteEncodeProviderPacingTest, PicturesInFlightChangeNothingWhenEncodesFitTheStreamInterval) {
    for (const auto mode : {game_frames_e::fg_off, game_frames_e::slow}) {
      const auto serial = simulate_export_ring(mode, fence_wake_e::prompt, loop_rules_e::production, 4s, encoder_model_t {1, 5.0, 8.0});
      const auto pipelined = simulate_export_ring(mode, fence_wake_e::prompt, loop_rules_e::production, 4s, encoder_model_t {2, 5.0, 8.0});
      print_encoder_model("one picture at a time, 5-8 ms encodes", mode, serial);
      print_encoder_model("two pictures in flight, 5-8 ms encodes", mode, pipelined);
      EXPECT_GE(pipelined.new_frames, serial.new_frames - 0.5);
      EXPECT_EQ(pipelined.repeats, 0);
      EXPECT_LE(pipelined.mean_encode, serial.mean_encode + 500us);
    }
  }

  TEST(RemoteEncodeProviderPacingTest, AReplacedSlotReturnsAsItsReadsCompleteWithPicturesInFlight) {
    // With two pictures in flight a claim can come before the previous conversion's GPU work (queued
    // behind the game's) completed, so the slot it replaces is still being read. 5798c45f returned
    // such a slot only at the encode thread's next check (a submission or the next conversion), and
    // none runs during an exact hold: the producer meanwhile had one slot. The receiver's read-fence
    // wait returns it as the GPU completes those reads. That decision was made on the three-slot
    // ring, where it also bought new frames; with the protocol's four the stream rate is reached
    // either way and the read-fence wait still drops fewer Presents.
    for (const auto mode : {game_frames_e::fg_2x, game_frames_e::fg_4x}) {
      for (const int slots : {3, static_cast<int>(::reshade_bridge::slot_count)}) {
        encoder_model_t checks {2, 5.5, 22.0};
        checks.retire = encoder_model_t::retire_e::encode_thread;
        const auto at_checks = simulate_export_ring(mode, fence_wake_e::prompt, loop_rules_e::production, 4s, checks, slots);
        const auto at_completion = simulate_export_ring(mode, fence_wake_e::prompt, loop_rules_e::production, 4s, encoder_model_t {2, 5.5, 22.0}, slots);
        std::printf("[ MEASURE  ] %d export slots:\n", slots);
        print_encoder_model("  two in flight, slot returned at the encode thread's next check", mode, at_checks);
        print_encoder_model("  two in flight, slot returned as its reads complete", mode, at_completion);
        EXPECT_GE(at_completion.new_frames, 87.5);
        if (slots == 3) {
          EXPECT_GE(at_completion.new_frames, at_checks.new_frames + 1.5);
        }
        EXPECT_EQ(at_completion.repeats, 0);
        EXPECT_LT(at_completion.max_gap, 2 * 11111111ns);
        EXPECT_LT(at_completion.dropped, at_checks.dropped);
      }
    }
  }

  TEST(RemoteEncodeProviderPacingTest, AFourthExportSlotTakesThePresentsAGpuBoundGameDroppedWithThree) {
    // Live 10-06 15:39:44-15:40:00 (Stellar Blade 4K, FG 4x GPU-bound at ~24 real fps): the host
    // claimed nearly every published frame (overwritten_unconsumed ~0), yet the game dropped 27-35 of
    // its ~95 Presents/s and the stream had 57-65 new frames/s. Of three slots one was held, one
    // replaced and still being read, and the last carried the host's next frame or a write still
    // queued on the busy GPU. The production loop with two pictures in flight and the live FG-on
    // encode times reproduces that with three slots; the protocol's fourth must take most of them.
    const encoder_model_t live {2, 5.5, 22.0};
    const auto three = simulate_export_ring(game_frames_e::fg_4x_gpu_bound, fence_wake_e::prompt, loop_rules_e::production, 4s, live, 3);
    const auto four = simulate_export_ring(game_frames_e::fg_4x_gpu_bound, fence_wake_e::prompt, loop_rules_e::production, 4s, live);
    print_export_ring("three export slots", game_frames_e::fg_4x_gpu_bound, fence_wake_e::prompt, three);
    print_export_ring("four export slots", game_frames_e::fg_4x_gpu_bound, fence_wake_e::prompt, four);
    EXPECT_GE(three.dropped, 25.0);
    EXPECT_LE(three.dropped, 40.0);
    EXPECT_GE(three.new_frames, 52.0);
    EXPECT_LE(three.new_frames, 68.0);
    EXPECT_LE(four.dropped, three.dropped / 2);
    EXPECT_GE(four.new_frames, three.new_frames + 10.0);
    EXPECT_EQ(four.repeats, 0);
  }

  TEST(RemoteEncodeProviderPacingTest, ClaimingAtThePollTargetKeepsTheCadenceWhenEveryPictureIsInFlight) {
    // In the encode times' long tail both pictures are still in flight when the next frame is due.
    // encode_run() claims and converts the newest frame at its poll target and then waits for the
    // oldest picture, so a frame completed during that wait goes to the next claim. Waiting for the
    // slot first would claim that newer frame, but the claim would then come after its target: the
    // schedule rebases to it (select_encode_frame_schedule()), which skips stream frames after every
    // slow picture for a few tenths of a millisecond of Present to packet on the three-slot ring.
    // With four slots more of the newer frames are published, and waiting first would gain 1.3-1.7
    // ms of Present to packet, still at the cost of those skipped stream frames: smoothness first.
    // Live (10-06, FG on) completion waits averaged 6.4-12.2 ms with maxima of 92-691 ms: here
    // 5.5-14 ms, and every eighth picture 14-45 ms (12.2 ms on average).
    encoder_model_t production {2, 5.5, 14.0};
    production.tail_every = 8;
    production.tail_high_ms = 45.0;
    for (const auto mode : {game_frames_e::fg_2x, game_frames_e::fg_4x}) {
      for (const int slots : {3, static_cast<int>(::reshade_bridge::slot_count)}) {
        auto slot_first = production;
        slot_first.wait_before_claim = true;
        const auto at_target = simulate_export_ring(mode, fence_wake_e::prompt, loop_rules_e::production, 4s, production, slots);
        const auto after_slot = simulate_export_ring(mode, fence_wake_e::prompt, loop_rules_e::production, 4s, slot_first, slots);
        std::printf("[ MEASURE  ] %d export slots:\n", slots);
        print_encoder_model("  two in flight, claimed at the poll target, then waited for a slot", mode, at_target);
        print_encoder_model("  two in flight, waited for a slot, then claimed", mode, after_slot);
        EXPECT_GE(at_target.new_frames, 87.0);
        EXPECT_GE(at_target.new_frames, after_slot.new_frames + 3.0);
        EXPECT_EQ(at_target.repeats, 0);
        EXPECT_LT(at_target.max_gap, after_slot.max_gap);
        EXPECT_LE(at_target.mean_present_to_packet, after_slot.mean_present_to_packet + std::chrono::microseconds {slots == 3 ? 500 : 2000});
      }
    }
  }

  // Desktop Host SBS (AI depth on desktop capture, a 7680x2160 picture at 4K) through encode_run()
  // and its encoder. It is not an independent provider: it converts at capture cadence, holds
  // nothing, and an image wait that runs out encodes the input again (a repeat). The desktop
  // presents on the display's vblank grid (+-0.5 ms) and capture hands each change over `handoff`
  // after its Present (09-06 live: 1.6 ms). A conversion costs the loop `admission` (DDup and
  // adaptive admission, the TensorRT enqueue), then the same-frame poll waits for that capture's own
  // inference (`infer_*`, one at a time on its stream) within its budget
  // (host_sbs_same_frame_poll_plan(): at most 8 ms, ending 3 ms before the next encode target), then
  // `output` (postprocess, warp and output draw recorded): 3-6 ms in all by default. An inference
  // that misses the budget, or a capture admitted while the previous inference still runs
  // (backpressure), is drawn with older depth (`depth_misses`). Without `same_frame_budget` the
  // loop always waits for its inference: the conversion is a fixed 3-6 ms. The picture completes
  // `encode_*` after its submission (the warp and output GPU work, then NVENC), in order and at least
  // `engine` after the previous one; with two in flight the copy into the picture's own input comes
  // first. With depth 1 the loop waits for that completion (nvenc_base::encode_frame()); with 2 it
  // submits and waits only while both pictures are in flight (encode_pipeline_t), and each packet
  // leaves when its picture completes.
  struct host_sbs_desktop_model_t {
    unsigned depth = 1;
    double stream_fps = 90, content_fps = 90;
    double infer_low_ms = 1.5, infer_high_ms = 4.5;
    std::chrono::nanoseconds admission {500us}, output {1ms};
    bool same_frame_budget = true;
    double encode_low_ms = 6.0, encode_high_ms = 12.0;
    std::chrono::nanoseconds submit {400us};
    std::chrono::nanoseconds engine {5ms};
    std::chrono::nanoseconds copy {300us};
    std::chrono::nanoseconds handoff {1600us};
  };

  struct host_sbs_desktop_result_t {
    double presents = 0, new_frames = 0, depth_misses = 0;  ///< Per second.
    int repeats = 0;
    std::chrono::nanoseconds mean_latency {}, max_latency {};  ///< From a new frame's Present to its packet.
    std::chrono::nanoseconds max_gap {};  ///< Between new frames' packets.
    unsigned max_in_flight = 0;
  };

  host_sbs_desktop_result_t simulate_host_sbs_desktop(const host_sbs_desktop_model_t &model, std::chrono::nanoseconds duration = 4s) {
    using ns = std::chrono::nanoseconds;
    const auto seconds = [](double value) {
      return std::chrono::duration_cast<ns>(std::chrono::duration<double>(value));
    };
    const auto stream = seconds(1.0 / model.stream_fps);
    const auto threshold = stream / 4;
    // max_frametime: the minimum FPS target, a fifth of the stream rate and at least 10 fps.
    const auto idle = seconds(1.0 / std::max(model.stream_fps / 5.0, 10.0));
    const auto content = seconds(1.0 / model.content_fps);
    constexpr auto step = 10us;

    sim_jitter_t jitter;
    sim_jitter_t work_jitter {54321};  // Its own sequence: the desktop's Presents stay identical.
    std::vector<ns> presents;
    const auto phase = jitter.between(1.0, std::chrono::duration<double, std::milli>(content).count());
    for (auto vblank = phase; vblank < duration; vblank += content) {
      presents.push_back(vblank - 500us + jitter.between(0.0, 1.0));
    }

    source_owner source;
    source.observe(std::make_shared<captured_source>(-1, at(0ns)));  // The desktop before the run.
    source.converted();
    auto target = at(0ns);
    std::optional<std::size_t> mailbox;  // The newest capture handed over and not yet taken.
    std::size_t next_present = 0;
    std::deque<ns> in_flight;  // Completion times, oldest first.
    ns blocked_until {}, wait_until {}, last_completion {}, infer_free {}, total_latency {};
    std::optional<ns> last_new_packet;
    bool waiting = false;
    int new_frames = 0, misses = 0;
    host_sbs_desktop_result_t result;

    for (ns now {}; now < duration; now += step) {
      for (; next_present < presents.size() && presents[next_present] + model.handoff <= now; ++next_present) {
        mailbox = next_present;  // The image event keeps only the newest capture.
      }
      if (now < blocked_until) {
        continue;
      }
      if (!waiting) {
        // A capture already handed over ends the wait at once; otherwise only a capture or the
        // bound (at a scheduler tick, as nothing else notifies) ends it.
        const auto pending_wait = source.remaining_wait(at(now), target);
        const auto wait = video::detail::encode_image_wait(idle, stream, true, false, false, pending_wait);
        wait_until = wait > 0ns ? tick_wait_end(now, wait) : now;
        waiting = true;
      }
      if (!mailbox && now < wait_until) {
        continue;
      }
      waiting = false;

      std::optional<video::detail::encode_frame_schedule_t> schedule;
      if (mailbox) {
        const auto captured = *mailbox;
        mailbox.reset();
        source.observe(std::make_shared<captured_source>(static_cast<int>(captured), at(presents[captured])));
        auto current = at(presents[captured]);
        if (current - target < -threshold) {
          if (!source.due(at(now), target)) {
            continue;  // Early: deferred to its target.
          }
          current = at(now);
        }
        schedule = video::detail::select_encode_frame_schedule(current, target, stream, threshold);
      } else if (source.pending() && source.due(at(now), target)) {
        schedule = video::detail::select_encode_frame_schedule(at(now), target, stream, threshold);  // The deferred capture.
      }

      ns converted = now;
      std::optional<std::size_t> new_frame;
      if (schedule) {
        const auto enqueued = now + model.admission;
        ns polled_until = enqueued;
        if (infer_free > enqueued) {
          ++misses;  // The previous inference still runs: no new admission, older depth.
        } else {
          const auto inferred = enqueued + work_jitter.between(model.infer_low_ms, model.infer_high_ms);
          infer_free = inferred;
          const auto plan = platf::dxgi::detail::host_sbs_same_frame_poll_plan(true, false, schedule->next_encode_target, at(enqueued));
          if (!model.same_frame_budget) {
            polled_until = inferred;
          } else if (plan.eligible) {
            polled_until = std::min(inferred, plan.deadline.time_since_epoch());
          }
          if (inferred > polled_until) {
            ++misses;
          }
        }
        converted = polled_until + model.output;
        target = schedule->next_encode_target;
        new_frame = static_cast<std::size_t>(source.latest()->pixels);
        source.converted();
      } else {
        ++result.repeats;  // The wait ran out: encode the input again.
      }

      // acquire(): with every picture in flight, wait for the oldest.
      auto acquired = converted;
      while (!in_flight.empty() && in_flight.front() <= acquired) {
        in_flight.pop_front();
      }
      if (in_flight.size() >= model.depth) {
        acquired = in_flight.front();
        in_flight.pop_front();
      }
      const auto submitted = acquired + model.submit;
      const auto latency = work_jitter.between(model.encode_low_ms, model.encode_high_ms) + (model.depth > 1 ? model.copy : 0ns);
      const auto completion = std::max(submitted + latency, last_completion + model.engine);
      last_completion = completion;
      in_flight.push_back(completion);
      result.max_in_flight = std::max(result.max_in_flight, static_cast<unsigned>(in_flight.size()));
      // One picture at a time: encode_frame() returns once it completed.
      blocked_until = model.depth == 1 ? completion : submitted;
      if (new_frame) {
        ++new_frames;
        const auto latency_to_packet = completion - presents[*new_frame];
        total_latency += latency_to_packet;
        result.max_latency = std::max(result.max_latency, latency_to_packet);
        if (last_new_packet) {
          result.max_gap = std::max(result.max_gap, completion - *last_new_packet);
        }
        last_new_packet = completion;
      }
    }
    const double elapsed = std::chrono::duration<double>(duration).count();
    result.presents = static_cast<double>(presents.size()) / elapsed;
    result.new_frames = new_frames / elapsed;
    result.depth_misses = misses / elapsed;
    result.mean_latency = new_frames ? total_latency / new_frames : 0ns;
    return result;
  }

  void print_host_sbs_desktop(const char *label, const host_sbs_desktop_model_t &model, const host_sbs_desktop_result_t &result) {
    const auto ms = [](std::chrono::nanoseconds value) {
      return std::chrono::duration<double, std::milli>(value).count();
    };
    std::printf(
      "[ MEASURE  ] %s, stream %.0f fps, desktop %.0f fps, %u in flight: presents %.0f/s new %.1f/s older-depth %.1f/s repeats %d, "
      "Present to packet mean %.1f max %.1f ms, max gap %.1f ms, max in flight %u\n",
      label,
      model.stream_fps,
      model.content_fps,
      model.depth,
      result.presents,
      result.new_frames,
      result.depth_misses,
      result.repeats,
      ms(result.mean_latency),
      ms(result.max_latency),
      ms(result.max_gap),
      result.max_in_flight
    );
  }

  TEST(RemoteEncodeHostSbsPipelineTest, TwoPicturesInFlightKeepEachCapturesOwnDepthAtNinetyFps) {
    // Conversion 3-6 ms (the inference the same-frame poll waits for, 1.5-4.5 ms) and 6-12 ms from
    // submission to packet, 90 fps desktop. One picture at a time the loop spends 9-18 ms per
    // capture, so the next capture waits behind the encode and reaches its conversion with too little
    // budget left: the loop keeps up by drawing almost every capture with older depth, and Present to
    // packet grows by the wait. With two in flight the capture converts while the previous picture
    // encodes, on time and with its own depth.
    host_sbs_desktop_model_t serial;
    auto pipelined = serial;
    pipelined.depth = 2;
    const auto before = simulate_host_sbs_desktop(serial);
    const auto after = simulate_host_sbs_desktop(pipelined);
    print_host_sbs_desktop("one at a time", serial, before);
    print_host_sbs_desktop("two in flight", pipelined, after);
    EXPECT_GE(before.depth_misses, 60.0);
    EXPECT_GE(after.new_frames, after.presents - 0.5);  // Every capture.
    EXPECT_GE(after.new_frames, before.new_frames);
    EXPECT_LE(after.depth_misses, 1.0);
    EXPECT_LE(after.mean_latency, before.mean_latency - 3ms);
    EXPECT_LT(after.max_latency, before.max_latency);
    EXPECT_LT(after.max_gap, 2 * 11111111ns);  // No stream frame skipped.
    EXPECT_EQ(after.repeats, 0);
    EXPECT_EQ(after.max_in_flight, 2u);
    EXPECT_EQ(before.max_in_flight, 1u);
  }

  TEST(RemoteEncodeHostSbsPipelineTest, TwoPicturesInFlightTakeEveryCaptureWhenTheConversionCannotShrink) {
    // The same times with a conversion that always costs the loop 3-6 ms (no same-frame budget to
    // give up): one picture at a time drops about a fifth of the 90 fps captures.
    host_sbs_desktop_model_t serial;
    serial.same_frame_budget = false;
    auto pipelined = serial;
    pipelined.depth = 2;
    const auto before = simulate_host_sbs_desktop(serial);
    const auto after = simulate_host_sbs_desktop(pipelined);
    print_host_sbs_desktop("fixed conversion, one at a time", serial, before);
    print_host_sbs_desktop("fixed conversion, two in flight", pipelined, after);
    EXPECT_LE(before.new_frames, 80.0);
    EXPECT_GE(after.new_frames, after.presents - 0.5);
    EXPECT_LE(after.mean_latency, before.mean_latency - 4ms);
    EXPECT_LT(after.max_latency, before.max_latency);
    EXPECT_LT(after.max_gap, 2 * 11111111ns);
    EXPECT_EQ(after.repeats, 0);
  }

  TEST(RemoteEncodeHostSbsPipelineTest, SixtyFpsCapturesNeverOverlapAndPayOnlyTheInputCopy) {
    // At 60 fps (a 90 or 60 fps stream) a picture completes before the next capture converts: both
    // depths take every capture, no picture is ever submitted behind another, and two in flight cost
    // the copy into the picture's own input.
    for (const bool budget : {true, false}) {
      for (const double stream_fps : {90.0, 60.0}) {
        host_sbs_desktop_model_t serial;
        serial.stream_fps = stream_fps;
        serial.content_fps = 60.0;
        serial.same_frame_budget = budget;
        auto pipelined = serial;
        pipelined.depth = 2;
        const auto before = simulate_host_sbs_desktop(serial);
        const auto after = simulate_host_sbs_desktop(pipelined);
        print_host_sbs_desktop(budget ? "one at a time" : "fixed conversion, one at a time", serial, before);
        print_host_sbs_desktop(budget ? "two in flight" : "fixed conversion, two in flight", pipelined, after);
        EXPECT_GE(before.new_frames, before.presents - 0.5);
        EXPECT_GE(after.new_frames, after.presents - 0.5);
        EXPECT_EQ(after.depth_misses, before.depth_misses);
        EXPECT_LE(after.mean_latency, before.mean_latency + pipelined.copy);
        EXPECT_EQ(after.max_in_flight, 1u);
        EXPECT_EQ(after.repeats, 0);
      }
    }
  }

  TEST(RemoteEncodeHostSbsPipelineTest, TwoPicturesInFlightChangeNothingWhenHostSbsFitsTheInterval) {
    // 09-06 live Host SBS at 90 fps: conversion call 2.29 ms and NVENC encode and retrieve 6.05 ms on
    // average. Both depths take every capture, no picture is submitted behind another, and two in
    // flight cost only the input copy.
    for (const double content_fps : {90.0, 60.0}) {
      host_sbs_desktop_model_t serial;
      serial.content_fps = content_fps;
      serial.infer_low_ms = 0.8;
      serial.infer_high_ms = 1.8;
      serial.output = 500us;
      serial.encode_low_ms = 5.0;
      serial.encode_high_ms = 7.0;
      auto pipelined = serial;
      pipelined.depth = 2;
      const auto before = simulate_host_sbs_desktop(serial);
      const auto after = simulate_host_sbs_desktop(pipelined);
      print_host_sbs_desktop("09-06 times, one at a time", serial, before);
      print_host_sbs_desktop("09-06 times, two in flight", pipelined, after);
      EXPECT_GE(before.new_frames, before.presents - 1.0);
      EXPECT_GE(after.new_frames, after.presents - 1.0);
      EXPECT_EQ(after.depth_misses, 0.0);
      EXPECT_LE(after.mean_latency, before.mean_latency + pipelined.copy);
      EXPECT_EQ(after.max_in_flight, 1u);
      EXPECT_EQ(after.repeats, 0);
    }
  }

  // The desktop then stays static, and the final capture's inference missed the same-frame budget:
  // it completes at `inference_done`. While that poll is pending encode_run() caps its image wait at
  // the stream interval (encode_image_wait(); it ends at a scheduler tick, as no capture notifies),
  // and each timeout reconverts the retained source. Its nonblocking query keeps the target while
  // the inference is busy (a packed repeat) and draws the frame with its own depth once complete;
  // the idle heartbeat then resumes. The final capture's picture was submitted at 4.4 ms.
  struct pending_inference_result_t {
    std::vector<std::chrono::nanoseconds> polls;  ///< When each retained-source poll converted.
    std::chrono::nanoseconds upgraded_packet {};  ///< When the frame drawn with its own depth left.
    std::chrono::nanoseconds heartbeat {};  ///< The next encode after that.
    unsigned max_in_flight = 0;
  };

  pending_inference_result_t simulate_pending_inference_on_static_desktop(unsigned depth, std::chrono::nanoseconds inference_done) {
    using ns = std::chrono::nanoseconds;
    constexpr ns stream {11111111ns}, idle {55555555ns};
    constexpr ns query {300us}, output {1ms}, submit {400us}, encode {9ms}, engine {5ms};
    constexpr ns final_submitted {4400us};
    pending_inference_result_t result;
    std::deque<ns> in_flight {final_submitted + encode};
    auto last_completion = in_flight.front();
    auto now = depth == 1 ? last_completion : final_submitted;
    bool poll_pending = true;
    while (true) {
      const auto wait = video::detail::encode_image_wait(idle, stream, true, false, poll_pending, std::nullopt);
      now = tick_wait_end(now, wait);
      if (!poll_pending) {
        result.heartbeat = now;
        return result;
      }
      result.polls.push_back(now);
      const bool upgraded = now >= inference_done;
      poll_pending = !upgraded;
      auto acquired = now + query + (upgraded ? output : 0ns);
      while (!in_flight.empty() && in_flight.front() <= acquired) {
        in_flight.pop_front();
      }
      if (in_flight.size() >= depth) {
        acquired = in_flight.front();
        in_flight.pop_front();
      }
      const auto submitted = acquired + submit;
      const auto completion = std::max(submitted + encode, last_completion + engine);
      last_completion = completion;
      in_flight.push_back(completion);
      result.max_in_flight = std::max(result.max_in_flight, static_cast<unsigned>(in_flight.size()));
      if (upgraded) {
        result.upgraded_packet = completion;
      }
      now = depth == 1 ? completion : submitted;
    }
  }

  TEST(RemoteEncodeHostSbsPipelineTest, PendingInferencePollsKeepTheirCadenceWithPicturesInFlight) {
    constexpr std::chrono::nanoseconds stream {11111111ns}, idle {55555555ns};
    for (const std::chrono::nanoseconds inference_done : {6ms, 20ms, 30ms, 47ms, 80ms}) {
      const auto serial = simulate_pending_inference_on_static_desktop(1, inference_done);
      const auto pipelined = simulate_pending_inference_on_static_desktop(2, inference_done);
      std::printf(
        "[ MEASURE  ] inference complete at %.0f ms: one at a time %zu polls, own depth's packet at %.1f ms; two in flight %zu polls, packet at %.1f ms\n",
        std::chrono::duration<double, std::milli>(inference_done).count(),
        serial.polls.size(),
        std::chrono::duration<double, std::milli>(serial.upgraded_packet).count(),
        pipelined.polls.size(),
        std::chrono::duration<double, std::milli>(pipelined.upgraded_packet).count()
      );
      for (const auto *result : {&serial, &pipelined}) {
        ASSERT_FALSE(result->polls.empty());
        // A busy poll never repeats the input faster than the stream rate...
        for (std::size_t poll = 1; poll < result->polls.size(); ++poll) {
          EXPECT_GE(result->polls[poll] - result->polls[poll - 1], stream);
        }
        // ...the first poll after the completion draws the frame with its own depth...
        EXPECT_GE(result->polls.back(), inference_done);
        EXPECT_LT(result->polls.back() - inference_done, windows_tick);
        // ...and the idle heartbeat then resumes.
        EXPECT_GE(result->heartbeat - result->polls.back(), idle);
      }
      EXPECT_EQ(serial.max_in_flight, 1u);
      EXPECT_LE(pipelined.max_in_flight, 2u);
      EXPECT_LE(pipelined.polls.size(), serial.polls.size() + 1);
      EXPECT_LE(pipelined.upgraded_packet, serial.upgraded_packet);
    }
  }

  // A recovery IDR on a desktop source (desktop Host SBS, or plain desktop one picture at a time),
  // each encode_run() iteration with fixed times and no startup hold. The full encoded queue drops
  // the delta picture of capture `dropped` and requests the IDR (encoded_queue_drop_report_t), or
  // the client requests one at `client_idr`, which wakes nothing for a desktop source. One picture
  // at a time, encode_frame() publishes on the encode thread, so the next iteration reads that IDR at
  // its top. With pictures in flight the retrieving thread publishes while the loop converts or waits
  // for the next capture, and recovery_gate_t discards the deltas published after the drop until
  // the IDR. Captures arrive at the stream cadence, so none is early.
  enum class recovery_rules_e {
    top_only,  ///< cf0330ef: the loop reads an IDR only at the top of an iteration.
    /** The retrieving thread wakes the image wait; the loop reads an IDR raised during the wait
     *  before converting, and a recovery that skips the wait discards a pending wake. */
    read_after_wait,
  };

  struct recovery_model_t {
    unsigned depth = 2;
    recovery_rules_e rules = recovery_rules_e::read_after_wait;
    std::vector<std::chrono::nanoseconds> captures;  ///< When each capture reaches the image event.
    std::optional<int> dropped;  ///< The capture whose delta picture the full queue drops.
    std::optional<std::chrono::nanoseconds> client_idr;  ///< When the client requests an IDR.
    std::chrono::nanoseconds conversion {3ms}, submit {400us}, encode {6ms}, engine {5ms};
  };

  struct recovery_result_t {
    /** Each picture in submission order: I (IDR), P (a new capture) or R (a repeat) and the capture
     *  it shows; then "d" if the full queue dropped it, "x" if recovery_gate_t discarded it. */
    std::vector<std::string> pictures;
    std::chrono::nanoseconds requested {}, idr_submitted {};
  };

  recovery_result_t simulate_recovery_idr(const recovery_model_t &model, std::chrono::nanoseconds duration = 60ms) {
    using ns = std::chrono::nanoseconds;
    constexpr ns stream {11111111ns}, idle {55555555ns};
    constexpr ns threshold = stream / 4;
    constexpr auto step = 10us;
    const bool reads_after_wait = model.rules == recovery_rules_e::read_after_wait;
    const bool wakes = reads_after_wait && model.depth > 1;  // Only a retrieving thread wakes.

    struct picture_t {
      ns completion;
      bool idr;
      int content;
      std::size_t entry;
    };

    source_owner source;
    source.observe(std::make_shared<captured_source>(-1, at(0ns)));  // The desktop before the run.
    source.converted();
    int target_content = -1;  // What the conversion target shows.
    auto target = at(0ns);
    std::optional<std::size_t> mailbox;
    std::size_t next_capture = 0;
    std::deque<picture_t> in_flight;  // Oldest first; completions are in submission order.
    video::detail::recovery_gate_t gate;
    bool idr_raised = false, woken = false, idr_latched = false, recovery = false;
    bool dropped = false, client_requested = false;
    enum class phase_e {
      top,
      waiting,
      busy,
    } phase = phase_e::top;
    ns busy_until {}, wait_until {}, last_completion {};
    recovery_result_t result;

    for (ns now {}; now < duration; now += step) {
      for (; next_capture < model.captures.size() && model.captures[next_capture] <= now; ++next_capture) {
        mailbox = next_capture;  // The image event keeps only the newest capture.
      }
      if (model.client_idr && !client_requested && *model.client_idr <= now) {
        client_requested = true;
        idr_raised = true;
        result.requested = now;
      }
      // Each completed picture is published, by the retrieving thread or by encode_frame().
      while (!in_flight.empty() && in_flight.front().completion <= now) {
        const auto picture = in_flight.front();
        in_flight.pop_front();
        if (!gate.admits(picture.idr)) {
          result.pictures[picture.entry] += "x";
          continue;
        }
        const bool drop = !dropped && model.dropped && picture.content == *model.dropped && !picture.idr;
        gate.published(drop);
        if (drop) {
          dropped = true;
          result.pictures[picture.entry] += "d";
          result.requested = now;
          idr_raised = true;  // idr_events->try_raise()
          woken = woken || wakes;  // images->wake()
        }
      }
      if (phase == phase_e::busy) {
        if (now < busy_until) {
          continue;
        }
        phase = phase_e::top;
      }

      for (bool again = true; again;) {
        again = false;
        if (phase == phase_e::top) {
          // The session latches a requested IDR until it encodes a picture.
          recovery = idr_raised;
          idr_latched = idr_latched || idr_raised;
          idr_raised = false;
          if (!recovery || mailbox) {
            const auto wait = video::detail::encode_image_wait(idle, stream, true, false, false, source.remaining_wait(at(now), target));
            wait_until = wait > 0ns ? tick_wait_end(now, wait) : now;
            phase = phase_e::waiting;
          } else if (reads_after_wait) {
            woken = false;  // images->discard_wake()
          }
        }
        bool captured = false;
        if (phase == phase_e::waiting) {
          if (!mailbox && !woken && now < wait_until) {
            break;  // Still waiting.
          }
          woken = false;  // Every timed pop consumes a wake.
          if (mailbox) {
            source.observe(std::make_shared<captured_source>(static_cast<int>(*mailbox), at(model.captures[*mailbox])));
            mailbox.reset();
            captured = source.pending();
          }
          if (reads_after_wait && idr_raised) {
            phase = phase_e::top;  // `continue`: the capture stays pending in the source.
            again = true;
            continue;
          }
        }

        std::optional<video::detail::encode_frame_schedule_t> schedule;
        if (captured) {
          schedule = video::detail::select_encode_frame_schedule(source.latest()->captured_at, target, stream, threshold);
        } else if (source.pending() && source.due(at(now), target, recovery)) {
          schedule = video::detail::select_encode_frame_schedule(at(now), target, stream, threshold);
        }
        auto converted = now;
        if (schedule) {
          converted += model.conversion;
          target = schedule->next_encode_target;
          target_content = source.latest()->pixels;
          source.converted();
        }
        // acquire(): with every picture in flight, wait for the oldest.
        auto acquired = converted;
        const auto busy = static_cast<std::size_t>(std::count_if(in_flight.begin(), in_flight.end(), [&](const picture_t &picture) {
          return picture.completion > converted;
        }));
        if (busy >= model.depth) {
          acquired = in_flight[in_flight.size() - busy].completion;
        }
        const auto submitted = acquired + model.submit;
        const auto completion = std::max(submitted + model.encode, last_completion + model.engine);
        last_completion = completion;
        const bool idr = idr_latched;
        idr_latched = false;
        result.pictures.push_back(std::string {idr ? "I" : schedule ? "P" : "R"} + std::to_string(target_content));
        if (idr && result.idr_submitted == 0ns) {
          result.idr_submitted = submitted;
        }
        in_flight.push_back({completion, idr, target_content, result.pictures.size() - 1});
        busy_until = model.depth == 1 ? completion : submitted;
        phase = phase_e::busy;
      }
    }
    return result;
  }

  void print_recovery(const char *label, const recovery_model_t &model, const recovery_result_t &result) {
    std::string pictures;
    for (const auto &picture : result.pictures) {
      pictures += (pictures.empty() ? "" : " ") + picture;
    }
    std::printf(
      "[ MEASURE  ] %s, %u in flight, %s: IDR submitted %.1f ms after the request; %s\n",
      label,
      model.depth,
      model.rules == recovery_rules_e::top_only ? "read at the top" : "read after the wait",
      std::chrono::duration<double, std::milli>(result.idr_submitted - result.requested).count(),
      pictures.c_str()
    );
  }

  std::vector<recovery_result_t> simulate_recovery_rules(recovery_model_t model, const char *label, std::chrono::nanoseconds duration = 60ms) {
    std::vector<recovery_result_t> results;
    for (const auto &[depth, rules] : {std::pair {1u, recovery_rules_e::top_only}, std::pair {2u, recovery_rules_e::top_only}, std::pair {2u, recovery_rules_e::read_after_wait}}) {
      model.depth = depth;
      model.rules = rules;
      results.push_back(simulate_recovery_idr(model, duration));
      print_recovery(label, model, results.back());
    }
    return results;
  }

  // 90 fps captures handed over 2 ms into each stream interval.
  std::vector<std::chrono::nanoseconds> ninety_fps_captures(int count) {
    std::vector<std::chrono::nanoseconds> captures;
    for (int capture = 0; capture < count; ++capture) {
      captures.push_back(2ms + capture * 11110us);
    }
    return captures;
  }

  TEST(RemoteEncodeHostSbsPipelineTest, AQueueDropOnAStaticDesktopGetsItsRecoveryIdrAtOnce) {
    // The desktop's last capture's picture is dropped while the loop waits with nothing to convert.
    // Read only at the top, the IDR waited for that idle wait (the minimum-FPS bound, 55.6 ms at
    // 90 fps) behind a repeat the gate discards; woken, it follows the drop as one at a time, and
    // the idle repeat comes at its usual bound.
    recovery_model_t model;
    model.captures = {2ms};
    model.dropped = 0;
    const auto results = simulate_recovery_rules(model, "static desktop, queue drop", 100ms);
    const auto &serial = results[0], &before = results[1], &after = results[2];
    EXPECT_EQ(serial.idr_submitted - serial.requested, model.submit);
    EXPECT_GE(before.idr_submitted - before.requested, 40ms);
    EXPECT_EQ(before.pictures, (std::vector<std::string> {"P0d", "R0x", "I0"}));
    EXPECT_EQ(after.idr_submitted - after.requested, model.submit);
    EXPECT_EQ(after.pictures, serial.pictures);
    EXPECT_EQ(after.pictures, (std::vector<std::string> {"P0d", "I0", "R0"}));
  }

  TEST(RemoteEncodeHostSbsPipelineTest, AQueueDropDuringTheCaptureWaitGetsItsRecoveryIdrBeforeTheNextCapture) {
    // 90 fps desktop, conversion 3 ms, 6 ms to the packet: capture 0's picture is dropped while the
    // loop waits for capture 1. Read only at the top, capture 1 converted and encoded as a delta the
    // gate discards, and its IDR followed. Woken, the loop encodes the IDR at once, as one at a time,
    // and capture 1 follows as a delta the client can decode.
    recovery_model_t model;
    model.captures = ninety_fps_captures(4);
    model.dropped = 0;
    const auto results = simulate_recovery_rules(model, "90 fps desktop, drop during the wait");
    const auto &serial = results[0], &before = results[1], &after = results[2];
    EXPECT_EQ(serial.idr_submitted - serial.requested, model.submit);
    EXPECT_EQ(before.pictures, (std::vector<std::string> {"P0d", "P1x", "I1", "P2", "P3"}));
    EXPECT_GE(before.idr_submitted - before.requested, model.conversion + model.submit);
    EXPECT_EQ(after.idr_submitted - after.requested, model.submit);
    EXPECT_EQ(after.pictures, (std::vector<std::string> {"P0d", "I0", "P1", "P2", "P3"}));
  }

  TEST(RemoteEncodeHostSbsPipelineTest, AQueueDropDuringAConversionLeavesNoRepeatBehindItsRecoveryIdr) {
    // Conversion 4 ms, 9 ms to the packet: capture 0's picture is dropped while capture 1 converts,
    // so capture 1 is a delta the gate discards whatever the loop does, and the IDR follows at the
    // next iteration, which skips its wait. The retrieving thread's wake came while nothing waited:
    // discarded then, it cannot end the next capture wait at once and encode a repeat.
    recovery_model_t model;
    model.captures = ninety_fps_captures(4);
    model.dropped = 0;
    model.conversion = 4ms;
    model.encode = 9ms;
    const auto results = simulate_recovery_rules(model, "90 fps desktop, drop during a conversion");
    const auto &before = results[1], &after = results[2];
    EXPECT_EQ(before.pictures, (std::vector<std::string> {"P0d", "P1x", "I1", "P2", "P3"}));
    EXPECT_EQ(after.pictures, before.pictures);
    EXPECT_EQ(after.idr_submitted, before.idr_submitted);
  }

  TEST(RemoteEncodeHostSbsPipelineTest, AClientIdrDuringTheCaptureWaitMakesTheCaptureTheIdr) {
    // A client's IDR request wakes nothing for a desktop source: the next capture ends the wait.
    // Read only at the top, that capture was encoded as a delta first and the IDR repeated it; read
    // after the wait, the capture itself is the IDR, one picture earlier at either depth.
    recovery_model_t model;
    model.captures = ninety_fps_captures(4);
    model.client_idr = 12ms;
    const auto results = simulate_recovery_rules(model, "90 fps desktop, client IDR");
    const auto &serial = results[0], &before = results[1], &after = results[2];
    EXPECT_EQ(serial.pictures, (std::vector<std::string> {"P0", "P1", "I1", "P2", "P3"}));
    EXPECT_EQ(before.pictures, serial.pictures);
    EXPECT_EQ(after.pictures, (std::vector<std::string> {"P0", "I1", "P2", "P3"}));
    EXPECT_EQ(after.idr_submitted, model.captures[1] + model.conversion + model.submit);
    EXPECT_LT(after.idr_submitted, before.idr_submitted);

    model.depth = 1;  // Plain desktop: the same rule, one picture at a time.
    const auto plain = simulate_recovery_idr(model);
    print_recovery("90 fps desktop, client IDR", model, plain);
    EXPECT_EQ(plain.pictures, after.pictures);
    EXPECT_EQ(plain.idr_submitted, after.idr_submitted);
    // One at a time, the delta's whole encode no longer precedes the IDR.
    EXPECT_EQ(serial.idr_submitted - plain.idr_submitted, model.encode + model.submit);
  }

  TEST(RemoteEncodeProviderPacingTest, ProviderHoldSleepsExactlyToAPendingFramesPollTarget) {
    constexpr std::chrono::nanoseconds stream {11111111ns};
    constexpr std::chrono::nanoseconds keepalive {55555555ns};
    EXPECT_FALSE(video::detail::provider_hold(std::nullopt, keepalive, stream));  // Nothing pending.
    EXPECT_FALSE(video::detail::provider_hold(0ns, keepalive, stream));  // Due: convert now.
    EXPECT_FALSE(video::detail::provider_hold(30ms, 20ms, stream));  // The keepalive comes first.
    // Whole 100 ns timer units, rounded up: the timer must not end before the target.
    EXPECT_EQ(video::detail::provider_hold(1234567ns, keepalive, stream), 1234600ns);
    EXPECT_EQ(video::detail::provider_hold(30ms, keepalive, stream), 11111200ns);  // At most a frame.
  }

  TEST(RemoteEncodeLoopStatsTest, AccountsHoldsWaitsAndEncodesForOneWindow) {
    video::detail::encode_loop_stats_t stats;
    stats.iteration();
    stats.iteration();
    stats.held(1500us, 2000us);
    stats.waited(55ms, 3ms);  // Woken by the fence: no overshoot.
    stats.waited(1ms, 15600us);  // Ran past its bound to the scheduler tick.
    stats.converted(170us);
    stats.encoded(true, 9600us, 300us, 9100us);
    stats.pictures(0, 1, 9600us, 9600us);
    stats.encoded(false, 5ms, 200us, 4600us);
    stats.pictures(0, 1, 5ms, 5ms);
    const auto line = stats.report(1s);
    EXPECT_NE(line.find("2 iterations in 1.0 s; 1 new-content and 1 repeated-content encodes"), std::string::npos) << line;
    EXPECT_NE(line.find("holding 2.0 ms in 1 exact holds (requested 1.5 ms, overshoot avg 0.50 max 0.50 ms)"), std::string::npos) << line;
    EXPECT_NE(line.find("waiting 18.6 ms in 2 image waits (requested 56.0 ms; 1 ran to their bound, overshoot avg 14.60 max 14.60 ms)"), std::string::npos) << line;
    EXPECT_NE(line.find("converting 0.2 ms in 1 conversions; encoding 14.6 ms (NVENC submit 0.5 ms, completion wait 13.7 ms; "
                        "up to 1 picture in flight, 0 submitted behind another, submission to packet avg 7.30 max 9.60 ms)"),
              std::string::npos)
      << line;
    EXPECT_NE(line.find("loop work 964.6 ms"), std::string::npos) << line;
    stats.reset();
    EXPECT_EQ(stats.iterations(), 0u);
  }

  TEST(RemoteEncodeLoopStatsTest, ReportsPicturesInFlightAndTheirSubmissionToPacketTime) {
    // With two pictures in flight the loop spends only the submission (and any wait for the oldest
    // picture) in its encode step; the retrieving thread reports each picture's whole encode.
    video::detail::encode_loop_stats_t stats;
    stats.set_pipeline_depth(2);
    stats.encoded(true, 500us, 300us, 0us);
    stats.encoded(true, 2500us, 300us, 2000us);
    stats.pictures(1, 2, 24ms, 13ms);
    const auto line = stats.report(1s);
    EXPECT_NE(line.find("encoding 3.0 ms (NVENC submit 0.6 ms, completion wait 2.0 ms; up to 2 pictures in flight, 1 submitted behind another, "
                        "submission to packet avg 12.00 max 13.00 ms)"),
              std::string::npos)
      << line;
    stats.reset();
    EXPECT_NE(stats.report(1s).find("up to 2 pictures in flight, 0 submitted behind another"), std::string::npos);
  }

  TEST(RemoteEncodeLoopStatsTest, AReconvertedUnchangedExportIsARepeatedContentEncode) {
    // As encode_run() classifies them, like the packets: by the encoded content's identity. A
    // keepalive or poll that re-renders the cached export converts again but repeats its content.
    video::detail::diagnostic_content_tracker_t content;
    video::detail::encode_loop_stats_t stats;
    const auto encode = [&](video::detail::diagnostic_timestamp_t content_timestamp, bool converted_frame) {
      const bool new_content = content.observe(content_timestamp, !converted_frame) == video::detail::diagnostic_content_e::new_content;
      stats.encoded(new_content, 9ms, 300us, 8ms);
    };
    const auto export_frame = std::chrono::steady_clock::time_point {} + 1s;
    encode(export_frame, true);  // A new export frame.
    encode(export_frame, true);  // The minimum-FPS keepalive re-renders it.
    encode(export_frame, false);  // A kept input.
    encode(std::nullopt, true);  // No content identity: not counted as new.
    encode(export_frame + 11ms, true);  // The next export frame.
    const auto line = stats.report(1s);
    EXPECT_NE(line.find("2 new-content and 3 repeated-content encodes"), std::string::npos) << line;
  }

  TEST(RemoteEncodePendingSourceTest, FinalEarlySourceSurvivesUntilItsPresentationDeadline) {
    source_owner source;
    image_event images;
    images.now = 5ms;
    const auto next_target = at(16666666ns);
    auto final_image = std::make_shared<captured_source>(17, at(5ms));
    const std::weak_ptr<captured_source> retained = final_image;
    source.observe(std::move(final_image));
    ASSERT_FALSE(final_image);
    ASSERT_FALSE(retained.expired());
    ASSERT_TRUE(source.pending());
    EXPECT_FALSE(source.due(at(images.now), next_target));

    // No subsequent capture arrives. The real image-event wait reaches the source deadline,
    // rather than waiting 83 ms and re-encoding the older persistent encoder input.
    EXPECT_FALSE(video::detail::wait_for_encode_image(
      images, 83333333ns, 16666666ns, true, false, false,
      source.remaining_wait(at(images.now), next_target)
    ));
    EXPECT_EQ(images.now, 16666666ns);
    ASSERT_TRUE(source.due(at(images.now), next_target));
    ASSERT_EQ(source.latest()->pixels, 17);
    EXPECT_EQ(source.latest()->captured_at, at(5ms));

    source.converted();
    EXPECT_FALSE(source.pending());
    EXPECT_FALSE(source.due(at(images.now), next_target));
    EXPECT_FALSE(source.remaining_wait(at(images.now), next_target));
    EXPECT_FALSE(retained.expired());  // Still available to depth completion / same-display rebuild.

    video::detail::wait_for_encode_image(
      images, 83333333ns, 16666666ns, true, false, false,
      source.remaining_wait(at(images.now), next_target)
    );
    EXPECT_EQ(images.now, 99999999ns);
    EXPECT_EQ(images.polls, 2);  // One due-source wait, then the ordinary idle heartbeat.
  }

  TEST(RemoteEncodePendingSourceTest, ANewCaptureReplacesTheDeferredOwnerWithoutGrowingAQueue) {
    source_owner source;
    source.observe(std::make_shared<captured_source>(1, at(3ms)));
    const std::weak_ptr<captured_source> superseded = source.latest();
    image_event images;
    images.now = 3ms;
    images.next_image = 8ms;
    EXPECT_EQ(video::detail::wait_for_encode_image(
      images, 83ms, 16ms, true, false, false, source.remaining_wait(at(images.now), at(16ms))
    ), 42);
    EXPECT_EQ(images.now, 8ms);
    source.observe(std::make_shared<captured_source>(2, at(8ms)));
    EXPECT_TRUE(superseded.expired());
    EXPECT_EQ(source.latest()->pixels, 2);
    EXPECT_EQ(source.remaining_wait(at(images.now), at(16ms)), 8ms);
    EXPECT_TRUE(source.due(at(16ms), at(16ms)));
  }

  TEST(RemoteEncodePendingSourceTest, StaleSourceTimestampCannotStarveDuePresentation) {
    source_owner source;
    source.observe(std::make_shared<captured_source>(9, at(1ms)));
    EXPECT_TRUE(source.due(at(40ms), at(16ms)));
    EXPECT_EQ(source.remaining_wait(at(40ms), at(16ms)), 0ns);
    EXPECT_EQ(source.latest()->captured_at, at(1ms));
    source.converted();
    EXPECT_FALSE(source.due(at(41ms), at(16ms)));
  }

  TEST(RemoteEncodePendingSourceTest, IdrUsesPendingSourceImmediatelyAndDoesNotInventOne) {
    source_owner source;
    EXPECT_FALSE(source.due(at(2ms), at(16ms), true));
    source.observe(std::make_shared<captured_source>(3, at(2ms)));
    EXPECT_FALSE(source.due(at(2ms), at(16ms)));
    EXPECT_TRUE(source.due(at(2ms), at(16ms), true));
    source.converted();
    EXPECT_FALSE(source.due(at(2ms), at(16ms), true));
    ASSERT_TRUE(source.latest());  // An ordinary IDR may still encode the existing input surface.
  }

  TEST(RemoteEncodePendingSourceTest, FasterConfiguredHeartbeatDoesNotRetirePendingSource) {
    source_owner source;
    source.observe(std::make_shared<captured_source>(4, at(5ms)));
    image_event images;
    images.now = 5ms;
    video::detail::wait_for_encode_image(
      images, 2ms, 16ms, true, false, false, source.remaining_wait(at(images.now), at(16ms))
    );
    EXPECT_EQ(images.now, 7ms);
    EXPECT_FALSE(source.due(at(images.now), at(16ms)));
    EXPECT_TRUE(source.pending());
    EXPECT_EQ(source.remaining_wait(at(images.now), at(16ms)), 9ms);
  }

  TEST(RemoteEncodePendingSourceTest, SameDisplayHandoffTransfersNewestPendingOwnerExactlyOnce) {
    source_owner retiring;
    retiring.observe(std::make_shared<captured_source>(5, at(5ms)));
    const std::weak_ptr<captured_source> image = retiring.latest();
    source_owner replacement;
    replacement.observe(retiring.release());
    EXPECT_FALSE(retiring.latest());
    EXPECT_FALSE(retiring.pending());
    EXPECT_TRUE(replacement.pending());
    EXPECT_EQ(replacement.latest()->pixels, 5);
    EXPECT_FALSE(image.expired());
    EXPECT_FALSE(retiring.release());
  }

  TEST(RemoteEncodePendingSourceTest, ShutdownDoesNotWaitOrConvertAndDropsRetainedOwnerOnExit) {
    std::weak_ptr<captured_source> image;
    {
      source_owner source;
      source.observe(std::make_shared<captured_source>(6, at(5ms)));
      image = source.latest();
      image_event images;
      images.now = 5ms;
      images.running = false;
      EXPECT_FALSE(video::detail::wait_for_encode_image(
        images, 83ms, 16ms, true, false, false, source.remaining_wait(at(images.now), at(16ms))
      ));
      EXPECT_EQ(images.now, 5ms);
      EXPECT_TRUE(source.pending());
      EXPECT_FALSE(image.expired());
    }
    EXPECT_TRUE(image.expired());
  }

  TEST(RemoteEncodePendingSourceTest, FailedEncoderReturnsStaticDesktopToReplacement) {
    safe::event_t<std::shared_ptr<captured_source>> images;
    source_owner retiring;
    retiring.observe(std::make_shared<captured_source>(7, at(5ms)));
    retiring.converted();  // No more desktop changes arrive before the encoder times out.
    retiring.return_for_rebuild(images, false, false);
    EXPECT_FALSE(retiring.latest());

    source_owner replacement;
    replacement.observe(images.pop(0ms));
    ASSERT_TRUE(replacement.latest());
    EXPECT_EQ(replacement.latest()->pixels, 7);
    EXPECT_EQ(replacement.latest()->captured_at, at(5ms));
    EXPECT_TRUE(replacement.pending());
    EXPECT_FALSE(images.peek());
  }

  TEST(RemoteEncodePendingSourceTest, MetadataOnlySourceIsNeverHandedToAReplacement) {
    // A live export's capture carries no pixels. Returned to a replacement encoder, it would only
    // keep that encoder's black startup input; capture copies the whole desktop again instead.
    safe::event_t<std::shared_ptr<captured_source>> images;
    std::weak_ptr<captured_source> skipped;
    {
      source_owner retiring;
      retiring.observe(std::make_shared<captured_source>(7, at(5ms)));
      skipped = retiring.latest();
      retiring.return_for_rebuild(images, false, false, false);
      EXPECT_FALSE(images.try_pop());
    }
    EXPECT_TRUE(skipped.expired());
  }

  TEST(RemoteEncodeStartupInputHoldTest, BoundIsOneIdleCaptureWaitAndItsBackoff) {
    // Capture re-reads the desktop for a replacement no later than this after it starts.
    const platf::dxgi::detail::capture_wait_policy_t remote {};
    EXPECT_EQ(video::detail::desktop_after_export_wait, remote.source_timeout() + remote.idle_backoff());
  }

  TEST(RemoteEncodeStartupInputHoldTest, OnlyAPixelLessPredecessorHoldsTheStartupInput) {
    video::detail::startup_input_hold_t hold {at(0ns), false};
    EXPECT_FALSE(hold.active(at(0ns)));
    EXPECT_EQ(hold.wait(at(0ns), std::chrono::nanoseconds {55ms}), 55ms);
  }

  TEST(RemoteEncodeStartupInputHoldTest, HeldInputWaitsOnlyForCaptureUntilTheDesktopArrives) {
    // The predecessor's input had no pixels. The keepalive (55 ms) is overdue from the first wait
    // and an IDR may be pending, yet nothing encodes the black startup input and no due deadline
    // shortens the wait: it ends at each idle interval or with the desktop capture.
    constexpr std::chrono::nanoseconds idle {55ms};
    video::detail::startup_input_hold_t hold {at(0ns), true};
    image_event images;
    images.next_image = 150ms;
    std::optional<int> captured;
    while (!captured) {
      ASSERT_TRUE(hold.active(at(images.now)));
      captured = images.pop(hold.wait(at(images.now), idle));
    }
    EXPECT_EQ(images.now, 150ms);
    EXPECT_EQ(images.polls, 3);  // 55 ms, 110 ms, then the capture.
    hold.replaced();  // Converting the capture replaced the startup input; it is encoded now.
    EXPECT_FALSE(hold.active(at(images.now)));
    EXPECT_EQ(hold.wait(at(images.now), idle), idle);
  }

  TEST(RemoteEncodeStartupInputHoldTest, WithoutTheDesktopTheHoldEndsAtItsBound) {
    constexpr std::chrono::nanoseconds idle {55ms};
    video::detail::startup_input_hold_t hold {at(0ns), true};
    image_event images;
    while (hold.active(at(images.now))) {
      ASSERT_FALSE(images.pop(hold.wait(at(images.now), idle)));
      ASSERT_LE(images.polls, 4);
    }
    // 55, 110, 165 and the 210 ms bound; then the startup input may be encoded as before.
    EXPECT_EQ(images.now, video::detail::desktop_after_export_wait);
    EXPECT_EQ(images.polls, 4);
  }

  TEST(RemoteEncodeProviderPacingTest, KeptInputGammaConversionPollsAtStreamCadence) {
    // A pending stream gamma converts the retained source, but the conversion keeps the encoder
    // input (metadata for pixels a released export owned), so pending_gamma stays armed. Its
    // branch advances the provider schedule like a keepalive; left at the past target, the next
    // wait would be zero and the same input would convert again at once.
    constexpr std::chrono::nanoseconds interval {16ms};
    constexpr auto threshold = interval / 4;
    source_owner source;
    source.observe(std::make_shared<captured_source>(1, at(0ns)));
    source.converted();
    image_event images;
    images.now = 100ms;
    auto target = at(40ms);
    EXPECT_EQ(source.remaining_wait(at(images.now), target - threshold, true), 0ns);

    std::vector<std::chrono::nanoseconds> waits;
    for (int poll = 0; poll < 3; ++poll) {
      target = video::detail::select_encode_frame_schedule(at(images.now), target, interval, threshold).next_encode_target;
      const auto wait = source.remaining_wait(at(images.now), target - threshold, true);
      ASSERT_TRUE(wait);
      waits.push_back(*wait);
      ASSERT_FALSE(video::detail::wait_for_encode_image(images, 55ms, interval, true, false, true, wait));
    }
    // The first poll rebases the overdue schedule; later ones keep exact stream cadence.
    EXPECT_EQ(waits, (std::vector<std::chrono::nanoseconds> {interval - threshold, interval, interval}));
    EXPECT_EQ(images.now, 100ms + interval - threshold + 2 * interval);
    EXPECT_EQ(images.polls, 3);
  }

  TEST(RemoteEncodeWakeTest, ControlRequestsWakeOnlyAnInstalledEncodeLoop) {
    video::encode_wake_t wake;
    wake.wake();  // Nothing installed: no effect.

    safe::event_t<int> images;
    wake.install([&images]() {
      images.wake();
    });
    wake.wake();
    const auto started = std::chrono::steady_clock::now();
    EXPECT_FALSE(images.pop(10s));  // Ends at once, without a value.
    EXPECT_LT(std::chrono::steady_clock::now() - started, 5s);

    wake.install({});
    wake.wake();
    const auto idle = std::chrono::steady_clock::now();
    EXPECT_FALSE(images.pop(20ms));  // A full wait again.
    EXPECT_GE(std::chrono::steady_clock::now() - idle, 15ms);
  }

  TEST(RemoteEncodePendingSourceTest, FailureHandoffPreservesNewerQueuedCapture) {
    safe::event_t<std::shared_ptr<captured_source>> images;
    source_owner retiring;
    retiring.observe(std::make_shared<captured_source>(7, at(5ms)));
    const std::weak_ptr<captured_source> old_image = retiring.latest();
    images.raise(std::make_shared<captured_source>(8, at(6ms)));

    retiring.return_for_rebuild(images, false, false);
    EXPECT_TRUE(old_image.expired());
    const auto next = images.pop(0ms);
    ASSERT_TRUE(next);
    EXPECT_EQ(next->pixels, 8);
    EXPECT_EQ(next->captured_at, at(6ms));
  }

  TEST(RemoteEncodePendingSourceTest, FailureHandoffDoesNotKeepAnOldDisplayAliveAfterCancellation) {
    for (int cancellation = 0; cancellation < 3; ++cancellation) {
      safe::event_t<std::shared_ptr<captured_source>> images;
      std::weak_ptr<captured_source> old_image;
      {
        source_owner retiring;
        retiring.observe(std::make_shared<captured_source>(7, at(5ms)));
        old_image = retiring.latest();
        if (cancellation == 0) {
          images.stop();
        }
        retiring.return_for_rebuild(images, cancellation == 1, cancellation == 2);
        EXPECT_FALSE(images.try_pop());
      }
      EXPECT_TRUE(old_image.expired());
    }
  }
}  // namespace
