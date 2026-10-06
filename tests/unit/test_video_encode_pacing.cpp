#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <vector>
#include <src/platform/windows/capture_timing.h>
#include <src/video.h>
#include <src/video_encode_pacing.h>

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

  // Live Game 3D evidence (Stellar Blade 4K HDR, 90 fps stream): the add-on published 65-127
  // frames/s into its three-slot export ring, yet the encode loop claimed only 49-64 new frames/s
  // while the add-on dropped 22-110 Presents/s for want of a slot and replaced 2-79 finished
  // frames/s that the host never claimed. Each frame's fence completes one GPU lag (5-15 ms) after
  // its Present, and the completion wake that should end the loop's wait came late or not at all,
  // so the loop noticed finished frames only on ~90 Hz desktop metadata captures or its keepalive.
  // This model runs the loop's own wait and schedule helpers against the export ring's rules
  // (docs/reshade-sbs.md, GPU handoff contract) with that arrival pattern.
  enum class game_frames_e {
    fg_off,  ///< ~115 fps.
    fg_2x,  ///< ~72 real fps, ~145 Presents/s.
    fg_4x,  ///< ~55 real fps, ~220 Presents/s.
  };

  enum class fence_wake_e {
    prompt,
    late,  ///< 0-11 ms after the completion.
    lost,
  };

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
      if (mode == game_frames_e::fg_off) {
        real += jitter.between(7.5, 10.0);
        presents.push_back(real);
        continue;
      }
      const int generated = mode == game_frames_e::fg_2x ? 2 : 4;
      const auto length = mode == game_frames_e::fg_2x ? jitter.between(12.5, 15.0) : jitter.between(16.0, 20.5);
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
  };

  // The export ring and encode_run() while an independent provider's export is live, stepped every
  // 10 us. Producer, per Present: a free slot whose write completed, else the oldest ready slot
  // whose write completed and that a newer completed frame supersedes, else the Present drops.
  // Host: claims the newest ready frame past its held one whose fence passed and returns the
  // replaced slot (its reads completed during the previous encode). Its wait composes
  // remaining_wait(), provider_keepalive_wait() and export_recheck_wait() as encode_run() does; a
  // fence wake or desktop capture ends it early, and one that arrives while the loop is busy ends
  // the next wait at once (event_t). After the wait a pending frame converts once due, a due
  // keepalive repeats the input, and anything else encodes nothing.
  export_ring_result_t simulate_export_ring(game_frames_e mode, fence_wake_e wake, std::chrono::nanoseconds duration = 4s) {
    constexpr auto stream = 11111111ns;  // 90 fps
    constexpr auto threshold = stream / 4;
    constexpr auto keepalive = 55555555ns;  // The 18 fps minimum of a 90 fps stream.
    constexpr auto step = 10us;
    constexpr int slots = 3;

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
    const auto presents = game_presents(mode, duration, jitter);
    std::vector<std::chrono::nanoseconds> captures;
    for (auto capture = jitter.between(0.0, 11.1); capture < duration; capture += jitter.between(10.1, 12.1)) {
      captures.push_back(capture);
    }
    std::array<slot_t, slots> ring {};
    std::vector<std::chrono::nanoseconds> present_of {0ns}, completion_of {0ns}, wakes;
    std::size_t next_present = 0, next_capture = 0, completed = 0, held = 0;
    int held_slot = -1, next_slot = 0, published = 0, dropped = 0, overwritten = 0, new_frames = 0, encodes = 0;
    export_ring_result_t result;
    std::chrono::nanoseconds total_age {};
    std::optional<std::chrono::nanoseconds> last_new;

    source_owner source;
    source.observe(std::make_shared<captured_source>(1, at(0ns)));
    source.converted();
    auto target = at(0ns);
    std::chrono::nanoseconds last_encode {}, busy_until {}, wait_until {};
    bool busy = false, woken = false;
    const auto start_wait = [&](std::chrono::nanoseconds now) {
      const auto poll_target = target - threshold;
      bool pending = false;
      for (const auto &slot : ring) {
        pending = pending || (slot.state == slot_e::ready && slot.sequence > held && slot.sequence <= completed);
      }
      const auto pending_wait = source.remaining_wait(at(now), poll_target, pending);
      const auto bound = std::min(
        video::detail::provider_keepalive_wait(at(now), at(last_encode), keepalive),
        video::detail::export_recheck_wait(at(now), poll_target)
      );
      wait_until = now + (pending_wait ? std::min(*pending_wait, bound) : bound);
    };
    start_wait(0ns);

    for (std::chrono::nanoseconds now {}; now < duration; now += step) {
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
        completion_of.push_back(std::max(completion_of.back(), presents[next_present] + jitter.between(5.0, 15.0)));
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
      if (busy) {
        if (now < busy_until) {
          continue;
        }
        busy = false;
        start_wait(now);
      }
      if (!woken && now < wait_until) {
        continue;
      }
      woken = false;
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
          if (held_slot >= 0) {
            ring[held_slot].state = slot_e::free;
          }
          ring[selected].state = slot_e::reading;
          held_slot = selected;
          held = ring[selected].sequence;
          total_age += now - present_of[held];
          if (last_new) {
            result.max_gap = std::max(result.max_gap, now - *last_new);
          }
          last_new = now;
          ++new_frames;
        } else {
          ++result.repeats;
        }
        busy = true;
        busy_until = now + 8400us + 1ms * (encodes++ % 3);  // Conversion and NVENC: 8.4-10.4 ms.
        last_encode = busy_until;
        continue;
      }
      start_wait(now);
    }
    const double seconds = std::chrono::duration<double>(duration).count();
    result.presents = static_cast<double>(presents.size()) / seconds;
    result.published = published / seconds;
    result.dropped = dropped / seconds;
    result.overwritten = overwritten / seconds;
    result.new_frames = new_frames / seconds;
    result.mean_age = new_frames ? total_age / new_frames : 0ns;
    return result;
  }

  TEST(RemoteEncodeProviderPacingTest, LiveExportDeliversNewFramesAtStreamRateWithoutItsFenceWake) {
    constexpr const char *modes[] {"FG off", "FG 2x", "FG 4x"};
    constexpr const char *wakes[] {"prompt", "late", "lost"};
    for (const auto mode : {game_frames_e::fg_off, game_frames_e::fg_2x, game_frames_e::fg_4x}) {
      for (const auto wake : {fence_wake_e::prompt, fence_wake_e::late, fence_wake_e::lost}) {
        const auto result = simulate_export_ring(mode, wake);
        std::printf(
          "[ MEASURE  ] %s, %s fence wake: presents %.0f/s published %.0f/s dropped %.0f/s overwritten %.0f/s new %.1f/s repeats %d, mean claim age %.1f ms, max gap %.1f ms\n",
          modes[static_cast<int>(mode)],
          wakes[static_cast<int>(wake)],
          result.presents,
          result.published,
          result.dropped,
          result.overwritten,
          result.new_frames,
          result.repeats,
          std::chrono::duration<double, std::milli>(result.mean_age).count(),
          std::chrono::duration<double, std::milli>(result.max_gap).count()
        );
        // Every mode presents faster than the stream, so min(Present rate, 90) is 90.
        EXPECT_GE(result.new_frames, 85.5);
        EXPECT_LE(result.new_frames, 90.5);
        EXPECT_EQ(result.repeats, 0);  // A re-check that finds nothing encodes nothing.
        EXPECT_LT(result.max_gap, 25ms);  // At most one skipped stream frame.
      }
    }
  }

  TEST(RemoteEncodeProviderPacingTest, HeldExportBelowStreamRateIsClaimedWithinOneRecheck) {
    // A 40 fps game, no fence wake and no desktop capture: only the loop's own re-checks can find
    // each finished frame, at most one re-check interval after its completion.
    constexpr auto stream = 11111111ns;
    constexpr auto threshold = stream / 4;
    constexpr auto game = 25ms;
    EXPECT_EQ(video::detail::export_recheck_wait(at(10ms), at(4ms)), video::detail::export_recheck_interval);
    EXPECT_EQ(video::detail::export_recheck_wait(at(10ms), at(15ms)), 5ms);  // Not before the target.
    source_owner source;
    source.observe(std::make_shared<captured_source>(1, at(0ns)));
    source.converted();
    std::chrono::nanoseconds now {}, last_encode {};
    auto target = at(0ns);
    int claimed = 0, waits = 0;
    while (claimed < 20) {
      const auto poll_target = target - threshold;
      const int completed = static_cast<int>(now / game);
      const auto pending_wait = source.remaining_wait(at(now), poll_target, completed > claimed);
      const auto bound = std::min(
        video::detail::provider_keepalive_wait(at(now), at(last_encode), 55555555ns),
        video::detail::export_recheck_wait(at(now), poll_target)
      );
      now += pending_wait ? std::min(*pending_wait, bound) : bound;
      ++waits;
      const int newest = static_cast<int>(now / game);
      if (newest > claimed && source.due(at(now), poll_target, false, true)) {
        EXPECT_LE(now - newest * game, video::detail::export_recheck_interval);
        target = video::detail::select_encode_frame_schedule(at(now), target, stream, threshold).next_encode_target;
        claimed = newest;
        now += 9400us;
        last_encode = now;
      }
    }
    // One wait to the poll target, then about one per millisecond until the 25 ms frame is done.
    EXPECT_LE(waits, 20 * 18);
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
