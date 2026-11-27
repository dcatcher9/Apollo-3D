/**
 * @file src/platform/windows/local_shutter_session.cpp
 * @brief The shutter monitor's presentation thread: emitter connection, clock acquisition and
 *        recovery, the timed command of each eye, and their diagnostics.
 *
 * One thread does all of it, in this order for every eye: take the pair, wait for a
 * presentation slot, present the eye, wait for its feedback, check it against the clock, hold
 * the command to its time and send it. The thread's stack owns a connection and a session; the
 * session borrows the connection and is destroyed first.
 */
// standard includes
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

// platform includes
#include <d3d11_4.h>
#include <dwmapi.h>
#include <wrl/client.h>

// local includes
#include "local_shutter_session.h"
#include "local_shutter_slot_timing.h"
#include "misc.h"
#include "open3d_emitter.h"
#include "shutter_presentation_clock.h"
#include "src/logging.h"
#include "src/platform/common.h"
#include "src/thread_safe.h"
#include "src/utility.h"

namespace platf::dxgi::local_presenter {
  using namespace std::literals;

  namespace {
    using switch_request_t = local_monitor_source_switch_t::request_t;
    using switch_state_e = local_monitor_source_switch_t::state_e;

    std::int64_t qpc_now() {
      LARGE_INTEGER counter {};
      QueryPerformanceCounter(&counter);
      return counter.QuadPart;
    }

    // Cold DWM/MPO transitions can publish stale IDs or switch refresh-counter domains
    // without a disjoint HRESULT. Establish twelve consecutive intervals before any IR.
    constexpr unsigned acquisition_samples = 13;
    constexpr std::int64_t cadence_tolerance_us = 100;

    /**
     * The profile's timing, settled before the emitter is configured: the eye interval, the
     * split of the configured delay between the host's hold and the emitter, how old an eye may
     * be when its command leaves, and the tolerances of the feedback checks.
     */
    struct local_shutter_timing_t {
      int eye_period_us = 0;
      detail::shutter_command_plan_t command {};
      platf::open3d::settings_t emitter_settings {};
      int host_age_budget_us = 0;
      std::int64_t armed_cadence_tolerance_us = 0;
      std::int64_t armed_future_tolerance_us = 0;
      std::chrono::milliseconds acquisition_wait {};
    };

    /**
     * The emitter connection of one shutter thread, with the high-resolution waiter and the
     * cancellation that ends the waits of both. The thread's stack owns it for the whole thread.
     * Its members are destroyed in the order the thread has always destroyed them: the emitter
     * first, then the stop callbacks, then the stop source and the waiter those callbacks reach.
     */
    class local_shutter_connection_t {
    public:
      local_shutter_connection_t(const local_shutter_presenter_t &presenter, std::stop_token shutter_stop):
          presenter_(presenter),
          config_(presenter.config),
          shutter_stop_(std::move(shutter_stop)),
          feedback_waiter(platf::create_deadline_waiter()) {}

      local_shutter_connection_t(const local_shutter_connection_t &) = delete;
      local_shutter_connection_t &operator=(const local_shutter_connection_t &) = delete;

      // Each failure is reported once through fail(); a connection cancelled while it is being
      // configured returns false without a report.
      bool connect() {
        if (!feedback_waiter) {
          fail("high-resolution presentation feedback waits are unavailable on this system");
          return false;
        }
        cancel_session_setup.emplace(presenter_.session_stop, cancel_waits_t {this});
        cancel_capture_setup.emplace(shutter_stop_, cancel_waits_t {this});
        if (!plan_timing()) {
          return false;
        }
        BOOST_LOG(info) << "Local shutter command timing: lens opens "sv << config_.monitor_config.frame_delay_us
                        << " us after the eye's refresh boundary for "sv << config_.monitor_config.frame_duration_us
                        << " us; host holds the command "sv << timing.command.hold_us << " us, emitter delay "sv
                        << timing.emitter_settings.frame_delay_us << " us, eye interval "sv << timing.eye_period_us
                        << " us, window "sv << (timing.command.late_window ? "reaches into the next eye interval"sv : "inside the eye's own interval"sv)
                        << '.';
        if (!emitter.connect(timing.emitter_settings, emitter_setup_stop.get_token())) {
          if (!stopped()) {
            fail("could not configure and verify the selected USB emitter");
          }
          return false;
        }
        LARGE_INTEGER frequency {};
        if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart <= 0) {
          fail("the Windows presentation clock is unavailable");
          return false;
        }
        qpc_frequency = frequency.QuadPart;
        return true;
      }

      bool stopped() const {
        return shutter_stop_.stop_requested() || presenter_.session_stop.stop_requested() || !IsWindow(presenter_.window);
      }

      void fail(std::string_view reason) const {
        BOOST_LOG(error) << "Local shutter monitor stopped: "sv << reason;
        if (config_.on_failure) {
          config_.on_failure(std::string(reason));
        }
        presenter_.failed.store(true, std::memory_order_release);
      }

    private:
      struct cancel_waits_t {
        local_shutter_connection_t *connection;

        void operator()() const noexcept {
          connection->emitter_setup_stop.request_stop();
          connection->feedback_waiter->notify();
        }
      };

      bool plan_timing() {
        const auto &config = config_;
        const auto eye_repeats = presenter_.eye_repeats;
        const auto emitter_eye_period = platf::open3d::eye_period_us(config.target_refresh_millihz, eye_repeats);
        if (!emitter_eye_period) {
          fail("the selected display refresh cannot represent a safe shutter-eye interval");
          return false;
        }
        const auto emitter_limits = platf::open3d::timing_limits(config.monitor_config.ir_protocol, config.monitor_config.signal_spacing_us);
        // The configured delay runs from the eye's refresh boundary to the lens opening. The
        // host holds the command for part of it and the emitter counts the rest.
        const auto command_plan = emitter_limits ?
                                    detail::plan_shutter_command(
                                      config.monitor_config.frame_delay_us,
                                      config.monitor_config.frame_duration_us,
                                      config.monitor_config.signal_spacing_us + emitter_limits->min_period_tail_base_us,
                                      emitter_limits->min_delay_us,
                                      *emitter_eye_period
                                    ) :
                                    std::nullopt;
        if (!command_plan) {
          fail("the shutter delay and duration do not fit the selected display refresh");
          return false;
        }
        const platf::open3d::settings_t emitter_settings {
          .port = config.monitor_config.emitter_port,
          .protocol = config.monitor_config.ir_protocol,
          .frame_delay_us = command_plan->emitter_delay_us,
          .frame_duration_us = config.monitor_config.frame_duration_us,
          .signal_spacing_us = config.monitor_config.signal_spacing_us,
          .flip_eyes = config.monitor_config.flip_eyes,
          .target_frametime_us = *emitter_eye_period,
          .averaging = config.monitor_config.emitter_averaging,
        };
        const auto feedback_budget = platf::open3d::detail::host_feedback_budget_us(emitter_settings);
        if (!feedback_budget) {
          fail("the shutter profile leaves no time for current-eye presentation feedback");
          return false;
        }
        // The budget is how late an eye command may leave the host and still open its own
        // window in time. An averaging emitter opens on its own rhythm and ignores a late
        // command, so a long window does not shorten the time the host has to confirm an eye.
        // Half the eye interval stays well short of the lateness the firmware would mistake
        // for a dropped frame, which is a whole interval less a millisecond.
        // A window that reaches into the next eye interval is held back by the host first, so
        // the eye is older when its command leaves; the plan's budget includes that hold.
        const auto host_age_budget_us = emitter_settings.averaging ? std::max(command_plan->host_age_budget_us, *emitter_eye_period / 2) :
                                                                     command_plan->host_age_budget_us;
        // Armed images are identified by their exact refresh-count difference. Their timestamp
        // only has to rule out a neighbouring refresh, so allow a quarter of one refresh period
        // of timestamp error, at most 750 us. Measured errors at 330 Hz reach about 620 us.
        const std::int64_t armed_cadence_tolerance_us = std::clamp<std::int64_t>(
          250000000LL / config.target_refresh_millihz,
          cadence_tolerance_us,
          750
        );
        timing = {
          .eye_period_us = *emitter_eye_period,
          .command = *command_plan,
          .emitter_settings = emitter_settings,
          .host_age_budget_us = host_age_budget_us,
          .armed_cadence_tolerance_us = armed_cadence_tolerance_us,
          // Measured refresh timestamps lead the query clock by up to about 140 us.
          .armed_future_tolerance_us = std::min<std::int64_t>(armed_cadence_tolerance_us, 250),
          .acquisition_wait = std::chrono::milliseconds {
            std::clamp<std::int64_t>((static_cast<std::int64_t>(*emitter_eye_period) * 3 + 999) / 1000, 25, 250)
          },
        };
        return true;
      }

      const local_shutter_presenter_t &presenter_;
      const local_presenter_config_t &config_;
      std::stop_token shutter_stop_;

    public:
      // The session borrows these. Each stop callback ends the emitter's setup and drain waits
      // and the current feedback or hold wait; connect() registers them once the waiter exists.
      std::unique_ptr<safe::deadline_waiter_t> feedback_waiter;
      std::stop_source emitter_setup_stop;
      std::optional<std::stop_callback<cancel_waits_t>> cancel_session_setup;
      std::optional<std::stop_callback<cancel_waits_t>> cancel_capture_setup;
      platf::open3d::emitter_t emitter;
      local_shutter_timing_t timing;  ///< Valid once connect() succeeds.
      std::int64_t qpc_frequency = 0;  ///< Valid once connect() succeeds.
    };

    /**
     * Reads the swapchain's presentation statistics for the eye being confirmed. It keeps that
     * eye's first sample and its last flush for the failure log.
     */
    class local_shutter_feedback_t {
    public:
      local_shutter_feedback_t(const local_shutter_presenter_t &presenter, const local_shutter_connection_t &connection):
          swapchain_(presenter.swapchain),
          io_ {
            .now = []() {
              return std::chrono::steady_clock::now();
            },
            .cancelled = [connection = &connection]() {
              return connection->stopped();
            },
            .query = [this]() {
              DXGI_FRAME_STATISTICS statistics {};
              const auto queried = swapchain_->GetFrameStatistics(&statistics);
              const detail::shutter_feedback_t feedback {
                queried == S_OK                                 ? detail::shutter_feedback_status_e::available :
                queried == DXGI_ERROR_FRAME_STATISTICS_DISJOINT ? detail::shutter_feedback_status_e::disjoint :
                                                                  detail::shutter_feedback_status_e::unsupported,
                statistics.PresentCount,
                statistics.PresentRefreshCount,
                statistics.SyncRefreshCount,
                statistics.SyncQPCTime.QuadPart,
                static_cast<std::int32_t>(queried),
              };
              disjoint |= feedback.status == detail::shutter_feedback_status_e::disjoint;
              if (!first_feedback) {
                first_feedback = feedback;
              }
              return feedback;
            },
            // Hardware Flip Queue may require a flush to publish statistics. The wait policy
            // permits this synchronous call only while acquiring with IR quiet; armed frames
            // must never spend their current shutter window blocked inside DwmFlush.
            .flush = [this]() {
              flush_start_qpc = qpc_now();
              flush_result = DwmFlush();
              flush_end_qpc = qpc_now();
              return SUCCEEDED(flush_result);
            },
            .wait_until = [connection = &connection](auto deadline) {
              return connection->feedback_waiter->wait_until(deadline);
            },
          } {}

      // The lambdas above refer to this object.
      local_shutter_feedback_t(const local_shutter_feedback_t &) = delete;
      local_shutter_feedback_t &operator=(const local_shutter_feedback_t &) = delete;

      void begin_eye() {
        first_feedback.reset();
        disjoint = false;
        flush_result = S_FALSE;
        flush_start_qpc = flush_end_qpc = 0;
      }

      detail::shutter_feedback_wait_result_t await(std::uint32_t requested_present, bool acquiring, std::chrono::milliseconds timeout) const {
        return detail::await_shutter_present(io_, requested_present, acquiring, timeout);
      }

      std::optional<detail::shutter_feedback_t> first_feedback;
      bool disjoint = false;
      HRESULT flush_result = S_FALSE;
      std::int64_t flush_start_qpc = 0;
      std::int64_t flush_end_qpc = 0;

    private:
      const Microsoft::WRL::ComPtr<IDXGISwapChain1> &swapchain_;
      const detail::shutter_feedback_wait_io_t io_;
    };

    /** Draws one eye into the back buffer and presents it once for each of its refreshes. */
    class local_shutter_eye_writer_t {
    public:
      struct submission_t {
        // The eye's first present: its identity and when it was handed to DXGI.
        std::uint32_t present_id = 0;
        std::int64_t present_begin_qpc = 0, present_end_qpc = 0;
        HRESULT result = S_FALSE, count_result = S_FALSE;
      };

      explicit local_shutter_eye_writer_t(const local_shutter_presenter_t &presenter):
          presenter_(presenter) {}

      // One eye is `presents` presents of the same image, each for exactly one refresh. The
      // first identifies the eye and marks its boundary; the others keep it on screen. Every
      // present after a flip-discard needs the image drawn again into the new back buffer.
      // An eye that repays a late image submits fewer presents.
      submission_t submit(bool left_eye, unsigned presents, ID3D11Texture2D *pair) {
        const auto &swapchain = presenter_.swapchain;
        submission_t submission;
        draw(left_eye, pair);
        DXGI_PRESENT_PARAMETERS parameters {};
        submission.present_begin_qpc = qpc_now();
        submission.result = swapchain->Present1(1, DXGI_PRESENT_RESTRICT_TO_OUTPUT, &parameters);
        submission.present_end_qpc = qpc_now();
        if (submission.result == S_OK) {
          submission.count_result = swapchain->GetLastPresentCount(&submission.present_id);
          for (unsigned repeat = 1; repeat < presents && submission.result == S_OK; ++repeat) {
            // The queue is two eyes deep, so a slot is normally free at once. A slot that
            // does not appear means the output stopped retiring images.
            if (WaitForSingleObject(presenter_.frame_latency_waitable, 100) != WAIT_OBJECT_0) {
              submission.result = HRESULT_FROM_WIN32(ERROR_TIMEOUT);
              break;
            }
            draw(left_eye, pair);
            const auto repeated = swapchain->Present1(1, DXGI_PRESENT_RESTRICT_TO_OUTPUT, &parameters);
            if (repeated != S_OK) {
              // Occlusion is a success code; an eye that is only partly queued is a failure.
              submission.result = FAILED(repeated) ? repeated : E_FAIL;
            }
          }
          if (presents > 1 && submission.result == S_OK && SUCCEEDED(submission.count_result)) {
            UINT last_present = 0;
            const auto last_result = swapchain->GetLastPresentCount(&last_present);
            if (FAILED(last_result) || last_present != static_cast<UINT>(submission.present_id + static_cast<UINT>(presents) - 1u)) {
              // Another present entered the sequence, so the eye's identity stride is unknown.
              submission.count_result = FAILED(last_result) ? last_result : E_UNEXPECTED;
            }
          }
        }
        return submission;
      }

    private:
      void draw(bool left_eye, ID3D11Texture2D *pair) {
        const auto &config = presenter_.config;
        const auto &shutter_context = presenter_.context;
        const auto &backbuffer_rtv = presenter_.backbuffer_rtv;
        const auto output_width = presenter_.output_width;
        const auto output_height = presenter_.output_height;
        if (presenter_.ghost_settings_file) {
          presenter_.ghost_settings_file->take(ghost_settings_);
        }
        // The settings file's own patterns replace the red and blue one. A picture is
        // corrected only when there is a pair to take the other eye from; if the pass cannot
        // draw, the plain copy below still shows the eye.
        if ((ghost_settings_.pattern != 0 || (!presenter_.test_pattern && ghost_settings_.corrects() && pair)) &&
            presenter_.ghost_pass.draw(
              shutter_context.Get(),
              backbuffer_rtv.Get(),
              pair,
              left_eye,
              static_cast<std::uint32_t>(output_width),
              static_cast<std::uint32_t>(output_height),
              config.hdr,
              ghost_settings_
            )) {
          return;
        }
        if (presenter_.test_pattern) {
          // In an HDR swapchain 1.0 is only 80 nits, too dim to judge through a lens that is
          // open for a small part of the time; 6.0 is 480 nits.
          const FLOAT level = config.hdr ? 6.0f : 1.0f;
          const FLOAT red[4] {level, 0.0f, 0.0f, 1.0f};
          const FLOAT blue[4] {0.0f, 0.0f, level, 1.0f};
          const FLOAT white[4] {level, level, level, 1.0f};
          const auto &labels = presenter_.test_pattern_labels;
          const auto &glyph = labels.glyphs[left_eye ? 0 : 1];
          labels.context->ClearView(backbuffer_rtv.Get(), left_eye ? red : blue, nullptr, 0);
          labels.context->ClearView(backbuffer_rtv.Get(), white, glyph.rectangles.data(), glyph.count);
        } else {
          const UINT left = left_eye ? 0 : static_cast<UINT>(output_width);
          const D3D11_BOX eye_box {left, 0, 0, left + static_cast<UINT>(output_width), static_cast<UINT>(output_height), 1};
          shutter_context->CopySubresourceRegion(presenter_.backbuffer.Get(), 0, 0, 0, 0, pair, 0, &eye_box);
        }
      }

      const local_shutter_presenter_t &presenter_;
      detail::shutter_ghost_settings_t ghost_settings_;
    };

    /**
     * Everything a lost clock takes with it. A recovery replaces all of it at once, so no present
     * identity, refresh count, eye phase or owed refresh carries over to the new clock.
     */
    struct local_shutter_clock_t {
      struct acquisition_sample_t {
        UINT requested = 0;
        detail::shutter_feedback_wait_status_e status {};
        detail::shutter_feedback_t feedback {};
        std::int64_t present_begin = 0, present_end = 0, query = 0, flush_begin = 0, flush_end = 0;
        unsigned streak = 0;
      };

      local_shutter_clock_t(const detail::shutter_feedback_tracker_t &tracker, int eye_repeats):
          feedback_tracker(tracker),
          eye_clock(static_cast<unsigned>(eye_repeats)),
          eye_presents(static_cast<unsigned>(eye_repeats)),
          previous_eye_presents(static_cast<unsigned>(eye_repeats)) {}

      detail::shutter_feedback_tracker_t feedback_tracker;
      detail::shutter_presentation_clock_t eye_clock;

      // Cold acquisition, with no eye command sent.
      std::optional<std::chrono::steady_clock::time_point> acquisition_deadline;
      unsigned acquisition_presents = 0;
      std::array<acquisition_sample_t, 8> acquisition_history {};
      std::array<unsigned, 6> acquisition_wait_counts {};
      unsigned acquisition_maximum_streak = 0;
      bool recovering_clock = false;

      // Armed eyes.
      bool has_displayed = false;
      // An averaging emitter keeps its own rhythm, so a late image leaves the picture whole
      // refreshes behind it. The presenter repays them by showing following eyes for fewer
      // refreshes, which needs no pause. These count the refreshes still owed, the presents
      // of the eye now being shown, and those of the eye before it.
      unsigned owed_refreshes = 0;
      unsigned eye_presents;
      unsigned previous_eye_presents;
      std::optional<std::uint32_t> previous_displayed_refresh;
      std::shared_ptr<switch_request_t> confirmed_left_source;
    };

    /** The counts the presentation summary reports when the thread ends. */
    struct local_shutter_summary_t {
      std::uint64_t confirmed_eye_commands = 0, held_eye_images = 0, extra_held_refreshes = 0;
      std::uint64_t absorbed_late_images = 0;
      std::uint64_t clock_recoveries = 0;
      std::uint64_t muted_eye_commands = 0;
      std::uint32_t maximum_confirmed_refresh_delta = 0;

      void log(int eye_repeats, const std::optional<unsigned> &muted_slot) const {
        BOOST_LOG(info) << "Local shutter presentation summary: confirmed_eye_commands="sv << confirmed_eye_commands
                        << " held_eye_images="sv << held_eye_images << " extra_held_refreshes="sv << extra_held_refreshes
                        << " absorbed_late_images="sv << absorbed_late_images
                        << " clock_recoveries="sv << clock_recoveries
                        << " maximum_refresh_delta="sv << maximum_confirmed_refresh_delta << " repeats="sv << eye_repeats << '.';
        if (muted_slot) {
          BOOST_LOG(info) << "Local shutter single-slot diagnostic summary: muted_logical_slot="sv << (*muted_slot == 0u ? 'L' : 'R')
                          << " skipped_usb_commands="sv << muted_eye_commands
                          << " sent_usb_commands="sv << confirmed_eye_commands - muted_eye_commands << '.';
        }
      }
    };

    /**
     * One shutter thread's presentation once its emitter is connected: the clock, the pair and
     * eye being shown, the timed command of each eye, and the diagnostics of all of it. It
     * borrows the connection, which outlives it.
     */
    class local_shutter_session_t {
    public:
      local_shutter_session_t(const local_shutter_presenter_t &presenter, local_shutter_connection_t &connection):
          presenter_(presenter),
          config_(presenter.config),
          connection_(connection),
          timing_(connection.timing),
          qpc_frequency_(connection.qpc_frequency),
          slot_timing_(presenter.slot_timing ? std::make_unique<local_shutter_slot_timing_t>(qpc_frequency_, static_cast<unsigned>(presenter.eye_repeats)) : nullptr),
          clock_(make_feedback_tracker(), presenter.eye_repeats),
          feedback_(presenter, connection),
          writer_(presenter) {}

      local_shutter_session_t(const local_shutter_session_t &) = delete;
      local_shutter_session_t &operator=(const local_shutter_session_t &) = delete;

      void run() {
        first_pair_deadline_ = std::chrono::steady_clock::now() + 5s;
        while (!connection_.stopped()) {
          if (!connection_.emitter.healthy() || presenter_.pairs.failed()) {
            connection_.fail("USB emitter or shared stereo packet became unavailable");
            break;
          }
          const bool acquiring = !clock_.feedback_tracker.armed();
          const bool left_eye = clock_.eye_clock.left_eye();
          const auto pair = take_pair(acquiring, left_eye);
          if (pair == pair_e::waiting) {
            continue;
          }
          if (pair == pair_e::failed || !acquisition_has_time(acquiring) || !wait_for_slot(acquiring) ||
              !present_eye(acquiring, left_eye)) {
            break;
          }
        }
        finish();
      }

    private:
      enum class pair_e {
        ready,
        waiting,  ///< No pair yet at startup; the loop tries again.
        failed,
      };

      // How a step left the eye: carry on with the eye, go on to the next one, or stop.
      enum class next_e {
        proceed,
        next_eye,
        stop,
      };

      // One eye's facts as its steps learn them, with its slot-timing sample.
      struct eye_t {
        bool acquiring;
        bool left_eye;
        unsigned logical_slot;
        std::optional<local_shutter_slot_timing_t::sample_t> timing_sample;
        local_shutter_eye_writer_t::submission_t submission {};
        detail::shutter_feedback_wait_result_t confirmation {};
        std::int64_t confirmation_qpc = 0;
        std::uint32_t confirmed_refresh_delta = 0;

        void set_timing_outcome(local_shutter_slot_timing_t::outcome_e outcome) {
          if (timing_sample) {
            timing_sample->outcome = outcome;
          }
        }
      };

      detail::shutter_feedback_tracker_t make_feedback_tracker() const {
        const auto eye_repeats = presenter_.eye_repeats;
        return detail::shutter_feedback_tracker_t {
          static_cast<unsigned>(eye_repeats), config_.target_refresh_millihz, qpc_frequency_,
          timing_.host_age_budget_us, cadence_tolerance_us, acquisition_samples, timing_.armed_cadence_tolerance_us,
          timing_.armed_future_tolerance_us, static_cast<unsigned>(eye_repeats),
        };
      }

      bool stopped() const {
        return connection_.stopped();
      }

      void fail(std::string_view reason) const {
        connection_.fail(reason);
      }

      RECT live_target_rect() const {
        std::lock_guard lock(config_.live_target->mutex);
        return config_.live_target->rect;
      }

      // The current packet stays keyed-locked through both eyes, including all repeated
      // refreshes. Publishing another capture can never change the second eye of this pair.
      pair_e take_pair(bool acquiring, bool left_eye) {
        if (!pair_ || (clock_.has_displayed && clock_.eye_clock.starts_pair())) {
          pair_ = presenter_.pairs.acquire_pair([&](const std::shared_ptr<switch_request_t> &request) {
            if (acquiring || !left_eye || !presenter_.switch_is_current(request)) {
              return false;
            }
            auto expected = switch_state_e::requested;
            return request->state.compare_exchange_strong(expected, switch_state_e::adopted);
          });
        }
        if (!pair_) {
          if (clock_.has_displayed) {
            fail("the shared stereo frame became unavailable during presentation");
            return pair_e::failed;
          }
          if (std::chrono::steady_clock::now() >= first_pair_deadline_) {
            fail("no stereo frame became available for monitor startup");
            return pair_e::failed;
          }
          std::this_thread::sleep_for(1ms);
          return pair_e::waiting;
        }
        return pair_e::ready;
      }

      // Cold acquisition, a recovery's included, has one budget for all of its presents.
      bool acquisition_has_time(bool acquiring) {
        if (acquiring) {
          if (!clock_.acquisition_deadline) {
            clock_.acquisition_deadline = std::chrono::steady_clock::now() + detail::shutter_acquisition_timeout;
          }
          if (std::chrono::steady_clock::now() >= *clock_.acquisition_deadline) {
            log_acquisition_history();
            BOOST_LOG(error) << "Local shutter clock acquisition expired: presents="sv << clock_.acquisition_presents
                             << " stable_samples="sv << clock_.feedback_tracker.streak() << '.';
            fail("DXGI could not establish a stable monitor clock before starting the glasses; returning to 2D");
            return false;
          }
        }
        return true;
      }

      bool wait_for_slot(bool acquiring) {
        const auto slot_deadline = std::min(
          std::chrono::steady_clock::now() + std::max(50ms, timing_.acquisition_wait),
          acquiring ? *clock_.acquisition_deadline : std::chrono::steady_clock::time_point::max()
        );
        DWORD latency = WAIT_TIMEOUT;
        while (!stopped() && std::chrono::steady_clock::now() < slot_deadline) {
          const auto remaining = std::chrono::ceil<std::chrono::milliseconds>(slot_deadline - std::chrono::steady_clock::now());
          latency = WaitForSingleObject(presenter_.frame_latency_waitable, static_cast<DWORD>(std::clamp<std::int64_t>(remaining.count(), 1, 50)));
          if (latency != WAIT_TIMEOUT) {
            break;
          }
        }
        if (latency != WAIT_OBJECT_0) {
          if (!stopped()) {
            if (acquiring) {
              // The acquisition deadline also bounds this wait, so an expired acquisition
              // can end here. Record what the output reported before giving up.
              log_acquisition_history();
              BOOST_LOG(error) << "Local shutter clock acquisition ended waiting for a presentation slot: presents="sv
                               << clock_.acquisition_presents << " stable_samples="sv << clock_.feedback_tracker.streak()
                               << " wait_result="sv << latency << '.';
            }
            fail("the physical output did not admit a bounded presentation slot");
          }
          return false;
        }
        return true;
      }

      // One eye in an admitted presentation slot: present it, confirm it on screen, and send
      // its command at its time. Returns false when presentation stops. The eye's slot-timing
      // sample is recorded when this returns, however the eye ended.
      bool present_eye(bool acquiring, bool left_eye) {
        eye_t eye {acquiring, left_eye, left_eye ? 0u : 1u};
        if (slot_timing_) {
          eye.timing_sample.emplace();
          eye.timing_sample->logical_slot = eye.logical_slot;
          eye.timing_sample->acquiring = acquiring;
          eye.timing_sample->segment = slot_timing_->segment;
          eye.timing_sample->previous = slot_timing_->previous;
        }
        const auto record_slot_timing = util::fail_guard([&]() {
          if (eye.timing_sample) {
            slot_timing_->finish(*eye.timing_sample);
          }
        });
        feedback_.begin_eye();
        if (stopped()) {
          return false;
        }
        repay_owed_refreshes();
        if (!submit_eye(eye)) {
          return false;
        }
        confirm_eye(eye);
        if (eye.confirmation.status == detail::shutter_feedback_wait_status_e::cancelled) {
          eye.set_timing_outcome(local_shutter_slot_timing_t::outcome_e::feedback_cancelled);
          return false;
        }
        if (acquiring) {
          return acquire_clock(eye);
        }
        if (const auto next = check_displayed_eye(eye); next != next_e::proceed) {
          return next == next_e::next_eye;
        }
        eye.set_timing_outcome(local_shutter_slot_timing_t::outcome_e::expired_before_emit);
        hold_command();
        if (!clock_.feedback_tracker.fresh_for_emit(qpc_now())) {
          log_feedback_failure(eye);
          return recover_or_fail("displayed eye expired before the USB command"sv, "the displayed eye became too old before its USB shutter command") != next_e::stop;
        }
        if (!signal_eye(eye)) {
          return false;
        }
        confirm_presented_pair(left_eye);
        return account_eye(eye);
      }

      void repay_owed_refreshes() {
        clock_.eye_presents = static_cast<unsigned>(presenter_.eye_repeats);
        if (clock_.owed_refreshes != 0 && clock_.has_displayed && clock_.eye_presents > 1) {
          // Keep at least one present, which identifies the eye.
          const unsigned repaid = std::min(clock_.owed_refreshes, clock_.eye_presents - 1);
          clock_.eye_presents -= repaid;
          clock_.owed_refreshes -= repaid;
        }
      }

      bool submit_eye(eye_t &eye) {
        eye.submission = writer_.submit(eye.left_eye, clock_.eye_presents, pair_);
        const auto &submission = eye.submission;
        if (slot_timing_) {
          eye.timing_sample->present_begin = submission.present_begin_qpc;
          eye.timing_sample->present_end = submission.present_end_qpc;
        }
        if (submission.result != S_OK) {
          // Occlusion is not a successful scanout. Never continue emitting an alternating eye
          // sequence while Windows is showing some other image on the selected monitor.
          fail("the restricted swapchain was occluded, stale, or could not present");
          return false;
        }
        eye.set_timing_outcome(local_shutter_slot_timing_t::outcome_e::present_id_failed);
        if (slot_timing_) {
          eye.timing_sample->requested = submission.present_id;
        }
        if (FAILED(submission.count_result)) {
          BOOST_LOG(error) << "Local shutter present identity failure: Present1="sv << util::log_hex(submission.result)
                           << " GetLastPresentCount="sv << util::log_hex(submission.count_result) << '.';
          fail("DXGI could not identify the submitted eye frame");
          return false;
        }
        return true;
      }

      void confirm_eye(eye_t &eye) {
        const auto feedback_timeout = eye.acquiring ?
                                        std::min(timing_.acquisition_wait, std::max(0ms, std::chrono::duration_cast<std::chrono::milliseconds>(*clock_.acquisition_deadline - std::chrono::steady_clock::now()))) :
                                        250ms;
        const auto confirmation = feedback_.await(eye.submission.present_id, eye.acquiring, feedback_timeout);
        const auto confirmation_qpc = qpc_now();
        eye.confirmation = confirmation;
        eye.confirmation_qpc = confirmation_qpc;
        if (slot_timing_) {
          eye.timing_sample->receipt = confirmation_qpc;
          eye.timing_sample->feedback = confirmation.feedback;
          eye.timing_sample->outcome = local_shutter_slot_timing_t::outcome_e::feedback_unconfirmed;
        }
      }

      // Cold acquisition: count the sample toward a stable clock and arm it once it is ready.
      // Hold the same left image and immutable pair while acquiring. The first eye must then
      // continue this exact clock; startup never sends an emitter eye command.
      bool acquire_clock(eye_t &eye) {
        const auto &confirmation = eye.confirmation;
        const auto requested_present = eye.submission.present_id;
        auto &feedback_tracker = clock_.feedback_tracker;
        eye.set_timing_outcome(local_shutter_slot_timing_t::outcome_e::cold_unsignalled);
        ++clock_.acquisition_presents;
        if (clock_.recovering_clock && confirmation.status != detail::shutter_feedback_wait_status_e::displayed &&
            !detail::shutter_feedback_is_timing_interruption(confirmation) &&
            !detail::shutter_feedback_superseded_within_eye(confirmation, requested_present, clock_.eye_presents)) {
          log_feedback_failure(eye);
          fail("presentation feedback failed while reacquiring the shutter clock");
          return false;
        }
        if (confirmation.status != detail::shutter_feedback_wait_status_e::displayed || feedback_.disjoint) {
          // A timeout's last raw sample is diagnostic data, never display authorization.
          feedback_tracker.observe(requested_present, detail::shutter_feedback_t {}, eye.confirmation_qpc);
        }
        const auto acquisition_status = confirmation.status == detail::shutter_feedback_wait_status_e::displayed ?
                                          feedback_tracker.observe(requested_present, confirmation.feedback, eye.confirmation_qpc) :
                                          detail::shutter_feedback_tracker_status_e::acquiring;
        ++clock_.acquisition_wait_counts[static_cast<std::size_t>(confirmation.status)];
        clock_.acquisition_maximum_streak = std::max(clock_.acquisition_maximum_streak, feedback_tracker.streak());
        clock_.acquisition_history[(clock_.acquisition_presents - 1) % clock_.acquisition_history.size()] = {
          requested_present, confirmation.status, confirmation.feedback,
          eye.submission.present_begin_qpc, eye.submission.present_end_qpc, eye.confirmation_qpc,
          feedback_.flush_start_qpc, feedback_.flush_end_qpc, feedback_tracker.streak(),
        };
        if (acquisition_status == detail::shutter_feedback_tracker_status_e::ready) {
          if (clock_.recovering_clock && !recovery_context_valid()) {
            if (!stopped()) {
              fail("the display or Windows desktop changed while reacquiring the shutter clock");
            }
            return false;
          }
          if (!feedback_tracker.arm()) {
            fail("the acquired shutter clock could not be armed");
            return false;
          }
          clock_.recovering_clock = false;
          BOOST_LOG(info) << "Local shutter monitor clock acquired: unsignalled_presents="sv << clock_.acquisition_presents
                          << " stable_intervals="sv << acquisition_samples - 1
                          << " host_age_budget_us="sv << timing_.host_age_budget_us
                          << " cadence_tolerance_us="sv << cadence_tolerance_us
                          << " armed_cadence_tolerance_us="sv << timing_.armed_cadence_tolerance_us
                          << " armed_future_tolerance_us="sv << timing_.armed_future_tolerance_us << '.';
        }
        return true;
      }

      // An armed eye must be on screen, continue the acquired clock exactly, and keep the eye
      // cadence before its command may be sent.
      next_e check_displayed_eye(eye_t &eye) {
        const auto &confirmation = eye.confirmation;
        const auto requested_present = eye.submission.present_id;
        auto &feedback_tracker = clock_.feedback_tracker;
        if (confirmation.status != detail::shutter_feedback_wait_status_e::displayed) {
          log_feedback_failure(eye);
          if (detail::shutter_feedback_is_timing_interruption(confirmation) &&
              recover_clock("presentation feedback interrupted"sv)) {
            return next_e::next_eye;
          }
          if (detail::shutter_feedback_superseded_within_eye(confirmation, requested_present, clock_.eye_presents) &&
              recover_clock("the eye's first present was superseded before its feedback was read"sv)) {
            return next_e::next_eye;
          }
          if (stopped()) {
            return next_e::stop;
          }
          fail("DXGI scanout feedback is unavailable, late, or invalid on this display configuration; returning to 2D");
          return next_e::stop;
        }
        eye.set_timing_outcome(local_shutter_slot_timing_t::outcome_e::tracker_rejected);
        if (feedback_tracker.observe(requested_present, confirmation.feedback, eye.confirmation_qpc, clock_.previous_eye_presents) != detail::shutter_feedback_tracker_status_e::accepted) {
          log_feedback_failure(eye);
          return recover_or_fail("displayed-eye identity, cadence or freshness changed"sv, "the displayed eye lost its acquired refresh clock or arrived too late for the shutter window");
        }
        if (slot_timing_) {
          eye.timing_sample->boundary = feedback_tracker.last_present_qpc().value_or(0);
        }
        eye.set_timing_outcome(local_shutter_slot_timing_t::outcome_e::eye_clock_rejected);
        if (!clock_.eye_clock.observe_displayed(confirmation.feedback.refresh_count, clock_.previous_eye_presents)) {
          log_feedback_failure(eye);
          return recover_or_fail("eye scanout cadence interrupted"sv, "DXGI could not prove an uninterrupted eye scanout cadence; restart after checking display timing");
        }
        eye.confirmed_refresh_delta = clock_.previous_displayed_refresh ?
                                        static_cast<std::uint32_t>(confirmation.feedback.refresh_count - *clock_.previous_displayed_refresh) :
                                        static_cast<std::uint32_t>(presenter_.eye_repeats);
        if (slot_timing_) {
          eye.timing_sample->delta_known = clock_.previous_displayed_refresh.has_value();
          eye.timing_sample->refresh_delta = eye.confirmed_refresh_delta;
        }
        clock_.previous_displayed_refresh = confirmation.feedback.refresh_count;
        return next_e::proceed;
      }

      // Hold the command until the planned time after the eye's refresh boundary.
      void hold_command() {
        if (const auto boundary = clock_.feedback_tracker.last_present_qpc(); boundary && timing_.command.hold_us > 0) {
          const auto send_at = *boundary + timing_.command.hold_us * qpc_frequency_ / 1000000;
          // The usual hold is too short to sleep: the scheduler's granularity is longer than
          // the whole of it. A hold of milliseconds, for a window that reaches into the next
          // eye interval, sleeps on the high-resolution timer and spins only at the end.
          const auto spin_from = send_at - 1000 * qpc_frequency_ / 1000000;
          while (!stopped()) {
            const auto remaining = std::chrono::microseconds {(spin_from - qpc_now()) * 1000000 / qpc_frequency_};
            if (remaining < 300us || !connection_.feedback_waiter->wait_until(std::chrono::steady_clock::now() + remaining)) {
              break;
            }
          }
          while (qpc_now() < send_at) {
            YieldProcessor();
          }
        }
      }

      bool signal_eye(eye_t &eye) {
        eye.set_timing_outcome(local_shutter_slot_timing_t::outcome_e::cancelled_before_emit);
        if (stopped()) {
          return false;
        }
        eye.set_timing_outcome(local_shutter_slot_timing_t::outcome_e::signal_failed);
        if (slot_timing_) {
          eye.timing_sample->signal_begin = qpc_now();
        }
        auto *write_timing = eye.timing_sample ? &eye.timing_sample->write : nullptr;
        const bool slot_muted = presenter_.muted_slot && *presenter_.muted_slot == eye.logical_slot;
        if (slot_muted) {
          ++summary_.muted_eye_commands;
          if (eye.timing_sample) {
            // Nothing is written for a muted slot, so there is no write to time.
            eye.timing_sample->signal_begin = 0;
          }
        }
        const bool signalled = slot_muted || connection_.emitter.signal(eye.left_eye, write_timing);
        if (slot_timing_) {
          eye.timing_sample->signal_end = qpc_now();
        }
        if (!signalled) {
          fail("the USB emitter missed an eye boundary");
          return false;
        }
        eye.set_timing_outcome(local_shutter_slot_timing_t::outcome_e::signalled);
        return true;
      }

      // Capture learns of the first whole pair, and a switched source is accepted once both of
      // its eyes have been shown.
      void confirm_presented_pair(bool left_eye) {
        if (!left_eye && clock_.has_displayed && !presenter_.pair_presented.load(std::memory_order_relaxed)) {
          presenter_.pair_presented.store(true, std::memory_order_release);
        }
        if (const auto &request = presenter_.pairs.current_request(); request && request->state.load() == switch_state_e::adopted) {
          if (left_eye) {
            clock_.confirmed_left_source = request;
          } else if (request == clock_.confirmed_left_source && presenter_.switch_is_current(request)) {
            auto expected = switch_state_e::adopted;
            const auto previous_mode = presenter_.source_switch->presented_mode.exchange(request->sbs_mode);
            if (!request->state.compare_exchange_strong(expected, switch_state_e::presented)) {
              // Cancellation won the acknowledgement race. A later attempt starts from
              // the last confirmed mode after this thread has joined.
              presenter_.source_switch->presented_mode.store(previous_mode);
            }
            clock_.confirmed_left_source.reset();
          }
        }
      }

      // Counts the eye and repays, or restarts the rhythm for, an image shown late.
      bool account_eye(const eye_t &eye) {
        ++summary_.confirmed_eye_commands;
        summary_.maximum_confirmed_refresh_delta = std::max(summary_.maximum_confirmed_refresh_delta, eye.confirmed_refresh_delta);
        // The eye before this one was meant to last previous_eye_presents refreshes. Elapsed
        // time decides how long it did last; Windows' refresh counter can trail it by one.
        const auto elapsed_refresh_delta = std::max(eye.confirmed_refresh_delta, clock_.feedback_tracker.interval_refreshes());
        const bool late_image = clock_.has_displayed && elapsed_refresh_delta > clock_.previous_eye_presents;
        const unsigned late_refreshes = late_image ? elapsed_refresh_delta - clock_.previous_eye_presents : 0;
        if (late_image) {
          ++summary_.held_eye_images;
          summary_.extra_held_refreshes += late_refreshes;
        }
        clock_.has_displayed = true;
        clock_.previous_eye_presents = clock_.eye_presents;
        config_.presented_frames->fetch_add(clock_.eye_presents, std::memory_order_relaxed);
        // A late image moves every later eye by whole refreshes. An averaging emitter keeps
        // its earlier rhythm and its firmware ignores so large a correction, so the lenses
        // would stay out of step with the picture. Repay the refreshes by shortening the
        // following eyes; if that is not possible or too many are owed, start the rhythm again.
        if (late_image && timing_.emitter_settings.averaging && presenter_.eye_repeats > 1 &&
            clock_.owed_refreshes + late_refreshes <= 2u * static_cast<unsigned>(presenter_.eye_repeats)) {
          clock_.owed_refreshes += late_refreshes;
          ++summary_.absorbed_late_images;
        } else if (late_image && timing_.emitter_settings.averaging) {
          return recover_or_fail("a late image moved the eye rhythm away from the averaging emitter"sv, "the averaging emitter could not be restarted after a late image") != next_e::stop;
        }
        return true;
      }

      // A recoverable fault: the next eye after a quiet recovery, or the end of presentation.
      next_e recover_or_fail(std::string_view recovery_reason, std::string_view failure) {
        if (recover_clock(recovery_reason)) {
          return next_e::next_eye;
        }
        if (stopped()) {
          return next_e::stop;
        }
        fail(failure);
        return next_e::stop;
      }

      bool recovery_context_valid() const {
        if (stopped() || presenter_.pairs.failed() || !config_.shutter_recovery_allowed ||
            !config_.shutter_recovery_allowed() || platf::display_change_epoch() != presenter_.display_epoch_at_open) {
          return false;
        }
        const auto source = query_local_presenter_display_identity(presenter_.source_name);
        const auto target = query_local_presenter_display_identity(presenter_.target_name);
        DXGI_OUTPUT_DESC output {};
        Microsoft::WRL::ComPtr<IDXGIOutput6> output6;
        DXGI_OUTPUT_DESC1 color {};
        if (!source || !target || !display_config::equal_device_paths(source->device_path, presenter_.source_identity_at_start->device_path) ||
            !display_config::equal_device_paths(target->device_path, presenter_.target_identity_at_start->device_path) ||
            !local_presenter_identity_is_coherent(config_.live_target, *source, *target, config_.target_adapter_id) ||
            FAILED(presenter_.target_output->GetDesc(&output)) || output.Monitor != presenter_.target_output_desc.Monitor ||
            !EqualRect(&output.DesktopCoordinates, &presenter_.target_output_desc.DesktopCoordinates) ||
            FAILED(presenter_.target_output.As(&output6)) || FAILED(output6->GetDesc1(&color)) ||
            color.ColorSpace != presenter_.target_desc1.ColorSpace || color.BitsPerColor != presenter_.target_desc1.BitsPerColor ||
            FAILED(presenter_.device->GetDeviceRemovedReason())) {
          return false;
        }
        return !stopped();
      }

      bool recover_clock(std::string_view reason) {
        if (!recovery_context_valid()) {
          BOOST_LOG(warning) << "Local shutter clock recovery rejected a changed display, device or desktop context."sv;
          return false;
        }
        ++summary_.clock_recoveries;
        BOOST_LOG(warning) << "Local shutter clock lost; holding presentation while the emitter drains: "sv << reason;
        if (slot_timing_) {
          slot_timing_->recovering();
        }
        // Present1 may have displayed the new eye even when its metadata was rejected. Freeze
        // NEW image submissions here, rather than repainting while an old USB write or finite
        // OPEN/CLOSE timer may still be active. The presents already queued for the current
        // eye still appear during this drain.
        if (!connection_.emitter.quiesce(connection_.emitter_setup_stop.get_token())) {
          BOOST_LOG(warning) << "Local shutter clock recovery could not verify a quiet USB emitter."sv;
          return false;
        }
        if (!recovery_context_valid()) {
          BOOST_LOG(warning) << "Local shutter clock recovery context changed while the emitter drained."sv;
          return false;
        }
        // A failed armed tracker is sticky. Only a successful same-handle FIFO barrier and
        // firmware drain authorize a wholly new clock; never repair IDs or infer eye parity.
        clock_ = local_shutter_clock_t {make_feedback_tracker(), presenter_.eye_repeats};
        clock_.recovering_clock = true;
        clock_.acquisition_deadline = std::chrono::steady_clock::now() + detail::shutter_acquisition_timeout;
        BOOST_LOG(info) << "Local shutter emitter quiet; reacquiring an unsignalled left-eye clock without changing the desktop."sv;
        return true;
      }

      void log_acquisition_history() const {
        log_local_shutter_window_snapshot(presenter_.window, live_target_rect(), presenter_.target_output_desc.Monitor);
        const auto acquisition_presents = clock_.acquisition_presents;
        const auto &acquisition_wait_counts = clock_.acquisition_wait_counts;
        const auto &acquisition_history = clock_.acquisition_history;
        BOOST_LOG(error) << "Local shutter acquisition outcomes: presents="sv << acquisition_presents
                         << " maximum_streak="sv << clock_.acquisition_maximum_streak
                         << " displayed="sv << acquisition_wait_counts[0] << " cancelled="sv << acquisition_wait_counts[1]
                         << " timeout="sv << acquisition_wait_counts[2] << " unsupported="sv << acquisition_wait_counts[3]
                         << " flush_failed="sv << acquisition_wait_counts[4] << " wait_failed="sv << acquisition_wait_counts[5]
                         << " qpc_frequency="sv << qpc_frequency_ << " host_age_budget_us="sv << timing_.host_age_budget_us << '.';
        const auto count = std::min<std::size_t>(acquisition_presents, acquisition_history.size());
        for (std::size_t i = acquisition_presents - count; i < acquisition_presents; ++i) {
          const auto &sample = acquisition_history[i % acquisition_history.size()];
          BOOST_LOG(error) << "Local shutter acquisition sample: requested="sv << sample.requested
                           << " wait_status="sv << static_cast<unsigned>(sample.status)
                           << " raw_status="sv << static_cast<unsigned>(sample.feedback.status)
                           << " raw_hr="sv << util::log_hex(sample.feedback.result_code)
                           << " present="sv << sample.feedback.present_count << " refresh="sv << sample.feedback.refresh_count
                           << " sync_refresh="sv << sample.feedback.sync_refresh_count << " sync_qpc="sv << sample.feedback.sync_qpc
                           << " query_qpc="sv << sample.query << " present_begin_qpc="sv << sample.present_begin
                           << " present_end_qpc="sv << sample.present_end << " flush_begin_qpc="sv << sample.flush_begin
                           << " flush_end_qpc="sv << sample.flush_end << " streak="sv << sample.streak << '.';
        }
      }

      void log_feedback_failure(const eye_t &eye) const {
        const auto &swapchain = presenter_.swapchain;
        const auto &swapchain_desc = presenter_.swapchain_desc;
        const auto &confirmation = eye.confirmation;
        // Capture all HWND/session evidence before notifying the controller to restore.
        log_local_shutter_window_snapshot(presenter_.window, live_target_rect(), presenter_.target_output_desc.Monitor);
        const auto first = feedback_.first_feedback.value_or(detail::shutter_feedback_t {});
        const auto &last = confirmation.feedback;
        UINT latest_submitted = 0;
        const auto latest_count_result = swapchain->GetLastPresentCount(&latest_submitted);
        constexpr std::array status_names {"displayed", "cancelled", "timeout", "unsupported", "flush_failed", "wait_failed"};
        BOOST_LOG(error) << "Local shutter scanout feedback: result="sv
                         << status_names[static_cast<std::size_t>(confirmation.status)]
                         << " requested="sv << eye.submission.present_id << " latest_submitted="sv << latest_submitted
                         << " first_eye="sv << !clock_.has_displayed << " left="sv << eye.left_eye << " repeats="sv << presenter_.eye_repeats
                         << " Present1="sv << util::log_hex(eye.submission.result)
                         << " count_hr="sv << util::log_hex(eye.submission.count_result)
                         << " latest_count_hr="sv << util::log_hex(latest_count_result)
                         << " queries="sv << confirmation.queries << " flushes="sv << confirmation.flushes << " waits="sv << confirmation.waits
                         << " first_hr="sv << util::log_hex(first.result_code)
                         << " first_present="sv << first.present_count << " first_refresh="sv << first.refresh_count
                         << " first_sync_refresh="sv << first.sync_refresh_count << " first_sync_qpc="sv << first.sync_qpc
                         << " last_hr="sv << util::log_hex(last.result_code)
                         << " last_present="sv << last.present_count << " last_refresh="sv << last.refresh_count
                         << " last_sync_refresh="sv << last.sync_refresh_count << " last_sync_qpc="sv << last.sync_qpc
                         << " previous_displayed_refresh="sv << clock_.previous_displayed_refresh.value_or(0)
                         << " qpc_frequency="sv << qpc_frequency_
                         << " present_start_qpc="sv << eye.submission.present_begin_qpc << " present_end_qpc="sv << eye.submission.present_end_qpc
                         << " confirmation_qpc="sv << eye.confirmation_qpc
                         << " acquiring="sv << eye.acquiring << " stable_samples="sv << clock_.feedback_tracker.streak()
                         << " host_age_budget_us="sv << timing_.host_age_budget_us
                         << " flush_hr="sv << util::log_hex(feedback_.flush_result)
                         << " flush_start_qpc="sv << feedback_.flush_start_qpc << " flush_end_qpc="sv << feedback_.flush_end_qpc << '.';

        // Read-only, failure-only probes distinguish composition feedback from a stale or
        // wrong output. They never authorize an eye or replace the confirmed-present gate.
        BOOL fullscreen = FALSE;
        const auto fullscreen_result = swapchain->GetFullscreenState(&fullscreen, nullptr);
        Microsoft::WRL::ComPtr<IDXGIOutput> containing_output;
        DXGI_OUTPUT_DESC containing_desc {};
        auto containing_result = swapchain->GetContainingOutput(&containing_output);
        if (SUCCEEDED(containing_result)) {
          containing_result = containing_output->GetDesc(&containing_desc);
        }
        Microsoft::WRL::ComPtr<IDXGISwapChainMedia> media;
        DXGI_FRAME_STATISTICS_MEDIA media_stats {};
        auto media_result = swapchain.As(&media);
        if (SUCCEEDED(media_result)) {
          media_result = media->GetFrameStatisticsMedia(&media_stats);
        }
        BOOST_LOG(error) << "Local shutter scanout context: target="sv << presenter_.target_display_name
                         << " containing="sv << platf::to_utf8(containing_desc.DeviceName)
                         << " containing_hr="sv << util::log_hex(containing_result)
                         << " fullscreen="sv << fullscreen << " fullscreen_hr="sv << util::log_hex(fullscreen_result)
                         << " format="sv << swapchain_desc.Format << " swap_effect="sv << swapchain_desc.SwapEffect
                         << " flags="sv << swapchain_desc.Flags << " present_flags="sv << DXGI_PRESENT_RESTRICT_TO_OUTPUT
                         << " hdr="sv << config_.hdr << " media_hr="sv << util::log_hex(media_result)
                         << " media_present="sv << media_stats.PresentCount << " media_refresh="sv << media_stats.PresentRefreshCount
                         << " media_sync_refresh="sv << media_stats.SyncRefreshCount << " media_sync_qpc="sv << media_stats.SyncQPCTime.QuadPart
                         << " composition_mode="sv << media_stats.CompositionMode << '.';
      }

      void finish() {
        summary_.log(presenter_.eye_repeats, presenter_.muted_slot);
        // Closing the transport first stops an averaging emitter's own rhythm. Without
        // averaging, finite pulses leave no autonomous eye-toggle source running after the last
        // one expires.
        connection_.emitter.close();
        presenter_.context->Flush();
        if (slot_timing_) {
          slot_timing_->log_summary(config_.monitor_config.flip_eyes, timing_.host_age_budget_us);
        }
      }

      const local_shutter_presenter_t &presenter_;
      const local_presenter_config_t &config_;
      local_shutter_connection_t &connection_;
      const local_shutter_timing_t &timing_;
      const std::int64_t qpc_frequency_;
      std::unique_ptr<local_shutter_slot_timing_t> slot_timing_;
      local_shutter_clock_t clock_;
      local_shutter_feedback_t feedback_;
      local_shutter_eye_writer_t writer_;
      ID3D11Texture2D *pair_ = nullptr;
      std::chrono::steady_clock::time_point first_pair_deadline_ {};
      local_shutter_summary_t summary_;
    };
  }  // namespace

  void run_local_shutter_presentation(const local_shutter_presenter_t &presenter, std::stop_token shutter_stop) {
    auto presentation_scheduling = platf::acquire_presentation_scheduling();
    platf::adjust_thread_priority(platf::thread_priority_e::high);
    local_shutter_connection_t connection {presenter, std::move(shutter_stop)};
    if (!connection.connect()) {
      return;
    }
    local_shutter_session_t session {presenter, connection};
    session.run();
  }
}  // namespace platf::dxgi::local_presenter
