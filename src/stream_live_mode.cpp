/**
 * @file src/stream_live_mode.cpp
 * @brief Definitions for the serialized 0x3007 live presentation-mode worker.
 */

// standard includes
#include <algorithm>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

extern "C" {
  // clang-format off
#include <moonlight-common-c/src/Limelight-internal.h>
  // clang-format on
}

// local includes
#include "atomic_presentation.h"
#include "globals.h"
#include "host_sbs_resolution.h"
#include "logging.h"
#include "process.h"
#include "stream.h"
#include "stream_internal.h"
#include "video.h"
#ifdef SUNSHINE_TESTS
  #include "atomic_presentation_test.h"
#endif

using namespace std::literals;

namespace stream {
  namespace {
    /**
     * Queue one acknowledgement for the control thread to encrypt and send.
     *
     * This is the only step any off-control thread performs. The control thread owns outgoing
     * control encryption and every ENet call, and it is also the thread that clears
     * session_t::control.peer and the server's session slot, so handing it a plain value keeps
     * the peer and GCM state from ever being touched by a worker. session_t::control retains the
     * queue's strong owner; mail_raw_t itself stores only a weak reference, so a producer-only
     * temporary would otherwise destroy the acknowledgement before the control thread could drain
     * it. If the session has already gone away, the queue dies with it and the client's own
     * timeout covers the missing reply.
     */
    void queue_live_video_mode_ack(const safe::mail_t &mail, const live_video_mode_ack_t &ack) {
      mail->queue<live_video_mode_ack_t>(mail::live_video_mode_ack, LIVE_VIDEO_MODE_ACK_QUEUE_LIMIT)->raise(ack);
    }

    /** Hand one serialized request to the encode loop. */
    void commit_live_video_mode(const live_video_mode_request_t &request) {
      // Do not publish pacing here. The requested encoder may fail, so video.cpp publishes the
      // coherent effective bitrate/cadence pair only after init_encoder() succeeds.
      request.mail->queue<video::video_mode_change_t>(mail::video_mode)->raise(request.change);
      request.video->wake_encoder();
    }
  }  // namespace

  /**
   * Answer a request that never reached the encode loop. Only the encoder knows the geometry it
   * installed, so a refusal reports the last mode it published instead.
   */
  void reject_live_video_mode(
    const safe::mail_t &mail,
    const std::shared_ptr<video::effective_video_mode_publisher_t> &effective_mode,
    std::uint32_t request_id,
    live_video_mode_ack_e status
  ) {
    const auto mode = effective_mode ?
                        effective_mode->current() :
                        video::effective_video_mode_t {};
    queue_live_video_mode_ack(
      mail,
      make_live_video_mode_ack(request_id, status, mode)
    );
  }

  /**
   * Serialize a live request through display apply, encoder creation and (on failure) display
   * rollback. Only an unstarted pending request may be coalesced.
   */
  void dispatch_live_video_mode(session_t &session, live_video_mode_request_t request) {
    auto &worker_state = session.live_video_mode_worker;
    const auto submitted = worker_state.transactions->submit(std::move(request));
    const auto &superseded = submitted.superseded;
    const auto &rejected = submitted.rejected;

    if (rejected) {
      reject_live_video_mode(
        rejected->mail,
        rejected->effective_mode,
        rejected->change.request_id,
        live_video_mode_ack_e::failed
      );
      return;
    }

    // A queued request that a newer one displaced was never applied, so it is answered as a
    // retryable failure carrying the mode still in effect. Without this a client that changed
    // its mind would be left waiting out a timeout for a reply that never comes.
    if (superseded) {
      reject_live_video_mode(
        superseded->mail,
        superseded->effective_mode,
        superseded->change.request_id,
        live_video_mode_ack_e::failed
      );
    }

    // Stop closes admission before waiting on this publication lock. A disconnect therefore
    // cannot miss a thread whose constructor passed the stopped check while it was racing.
    std::unique_lock worker_lock(worker_state.thread_mutex);
    if (worker_state.transactions->stopping()) {
      return;
    }
    if (worker_state.thread.joinable()) {
      worker_lock.unlock();
      worker_state.transactions->notify_pending();
      return;
    }
    worker_state.thread = std::thread([&session]() {
      auto &state = session.live_video_mode_worker;
      auto &transactions = *state.transactions;
      while (true) {
        auto job = transactions.wait_next();
        if (!job) {
          return;
        }
        const auto transaction_id = job->change.transaction_id;

        // The session can end while a display transition is in flight. Never resize a desktop
        // for a stream that is already gone. The acknowledgement is dropped with it; the client
        // is disconnected at that point and its own timeout covers the missing reply.
        if (job->mail->event<bool>(mail::shutdown)->peek()) {
          (void) transactions.finish(transaction_id);
          continue;
        }

        const auto host_sbs_rejects_geometry = [&]() {
          const int desired_sbs_mode = job->change.requested_sbs_mode;
          return desired_sbs_mode == ::video::SBS_AI &&
                 !models::host_sbs_v2_source_resolution_is_supported(
                   static_cast<std::uint32_t>(job->change.width),
                   static_cast<std::uint32_t>(job->change.height)
                 );
        };
        // Recheck the immutable requested mode's geometry before touching the desktop. This
        // preflight and the encoder's final checks retain their own authority.
        if (host_sbs_rejects_geometry()) {
          BOOST_LOG(warning)
            << "Rejecting live video mode "sv << job->change.width << 'x'
            << job->change.height
            << " because the requested Host SBS V2 geometry is unsupported."sv;
          reject_live_video_mode(
            job->mail,
            job->effective_mode,
            job->change.request_id,
            live_video_mode_ack_e::rejected_invalid
          );
          (void) transactions.finish(transaction_id);
          continue;
        }

        auto result = proc::live_video_mode_result_e::unchanged;
        if (
#ifdef SUNSHINE_TESTS
          state.process_hooks ? state.process_hooks->needs_display_change(job->change.width, job->change.height, job->change.encodingFramerate) :
#endif
                                proc::proc.live_video_mode_needs_display_change(job->change.width, job->change.height, job->change.encodingFramerate)
        ) {
          result =
#ifdef SUNSHINE_TESTS
            state.process_hooks ? state.process_hooks->apply(
                                    job->change.width,
                                    job->change.height,
                                    job->change.encodingFramerate,
                                    job->launch_session_id
                                  ) :
#endif
                                  proc::proc.apply_live_video_mode(
                                    job->change.width,
                                    job->change.height,
                                    job->change.encodingFramerate,
                                    job->launch_session_id
                                  );
        }
        const bool display_changed =
          result == proc::live_video_mode_result_e::applied;

        if (transactions.stopping()) {
          return;
        }
        switch (result) {
          case proc::live_video_mode_result_e::applied:
          case proc::live_video_mode_result_e::unchanged:
            if (job->mail->event<bool>(mail::shutdown)->peek()) {
              (void) transactions.finish(transaction_id);
              continue;
            }
            // The desktop is ready. Keep the transaction active until the encode loop proves
            // which mode exists and, on failure, the worker restores the prior desktop.
            commit_live_video_mode(*job);
            break;
          case proc::live_video_mode_result_e::needs_reconnect:
            BOOST_LOG(warning) << "type [IDX_SET_VIDEO_MODE]: "sv << job->change.width << 'x'
                               << job->change.height << " cannot be applied without reconnecting for ["sv
                               << job->client_name
                               << "]; the client must reconnect before trusting presentation state."sv;
            reject_live_video_mode(
              job->mail,
              job->effective_mode,
              job->change.request_id,
              live_video_mode_ack_status(result)
            );
            (void) transactions.finish(transaction_id);
            break;
          case proc::live_video_mode_result_e::failed:
            BOOST_LOG(warning) << "type [IDX_SET_VIDEO_MODE]: the desktop could not be resized to "sv
                               << job->change.width << 'x' << job->change.height << " for ["sv
                               << job->client_name << "]; the stream keeps its current mode."sv;
            reject_live_video_mode(
              job->mail,
              job->effective_mode,
              job->change.request_id,
              live_video_mode_ack_status(result)
            );
            (void) transactions.finish(transaction_id);
            break;
        }

        if (result != proc::live_video_mode_result_e::applied && result != proc::live_video_mode_result_e::unchanged) {
          continue;
        }

        auto completion = transactions.wait_completion(transaction_id);
        if (!completion) {
          return;
        }

        const auto matching_request = std::find_if(
          completion->requests.begin(),
          completion->requests.end(),
          [&](const video::video_mode_request_ref_t &request) {
            return request.transaction_id == transaction_id;
          }
        );
        if (matching_request == completion->requests.end()) {
          BOOST_LOG(error) << "Live video-mode completion lost its active transaction."sv;
        } else if (matching_request->request_id != job->change.request_id) {
          BOOST_LOG(error) << "Live video-mode completion matched a token but not its client request id."sv;
        }

        bool rollback_restored_desktop = true;
        if (!completion->applied && display_changed) {
          BOOST_LOG(warning) << "Restoring the last encodable virtual desktop mode "
                             << completion->desktop_mode.width << 'x'
                             << completion->desktop_mode.height << " after the encoder rejected "
                             << job->change.width << 'x' << job->change.height << '.';
#ifdef SUNSHINE_TESTS
          const auto rollback_result = state.process_hooks ?
                                         state.process_hooks->apply(
                                           completion->desktop_mode.width,
                                           completion->desktop_mode.height,
                                           completion->desktop_mode.framerate_millihz,
                                           job->launch_session_id
                                         ) :
                                         proc::proc.apply_live_video_mode(
                                           completion->desktop_mode.width,
                                           completion->desktop_mode.height,
                                           completion->desktop_mode.framerate_millihz,
                                           job->launch_session_id
                                         );
#else
          const auto rollback_result = proc::proc.apply_live_video_mode(
            completion->desktop_mode.width,
            completion->desktop_mode.height,
            completion->desktop_mode.framerate_millihz,
            job->launch_session_id
          );
#endif
          if (rollback_result != proc::live_video_mode_result_e::applied && rollback_result != proc::live_video_mode_result_e::unchanged) {
            rollback_restored_desktop = false;
            BOOST_LOG(error) << "The virtual desktop could not be restored after a live encoder "
                                "rollback; the client will be told to reconnect."sv;
          }
        }

        // Re-arm diagnostics only after the effective pacing pair is known. Resetting at request
        // enqueue lets the old encoder consume the latch before the new encoder is proven.
        job->video->pacing_plan_logged.store(false, std::memory_order_release);
        job->video->idr_pacing_plan_logged.store(false, std::memory_order_release);
        queue_live_video_mode_ack(
          job->mail,
          make_live_video_mode_ack(
            job->change.request_id,
            completion->applied ?
              live_video_mode_ack_e::applied :
              (rollback_restored_desktop ?
                 live_video_mode_ack_e::failed :
                 live_video_mode_ack_e::rejected_needs_reconnect),
            completion->mode
          )
        );

        if (!transactions.finish(transaction_id)) {
          BOOST_LOG(error) << "Live video-mode transaction gate rejected its active completion."sv;
        }
      }
    });
    worker_lock.unlock();
    worker_state.transactions->notify_pending();
  }

  void stop_live_video_mode_worker(session_t &session) {
    session.live_video_mode_worker.stop();
  }

  void join_live_video_mode_worker(session_t &session) {
    session.live_video_mode_worker.join();
  }

#ifdef SUNSHINE_TESTS
  namespace testing {
    struct atomic_presentation_fixture_t::impl_t {
      impl_t(video::effective_video_mode_t initial, atomic_presentation_process_hooks_t hooks) {
        if (!hooks.needs_display_change || !hooks.apply) {
          throw std::invalid_argument("The presentation fixture requires both inert process edges");
        }
        session.mail = std::make_shared<safe::mail_raw_t>();
        session.shutdown_event = session.mail->event<bool>(mail::shutdown);
        session.video = std::make_shared<video_channel_t>();
        session.config.monitor.effective_mode = std::make_shared<video::effective_video_mode_publisher_t>(initial);
        session.video->effective_mode = session.config.monitor.effective_mode;
        session.config.monitor.atomic_presentation = session.live_video_mode_worker.transactions;
        session.video->pacing_plan_logged.store(true);
        session.video->idr_pacing_plan_logged.store(true);
        session.control.video_mode_queue = session.mail->queue<video::video_mode_change_t>(mail::video_mode);
        session.control.live_video_mode_ack_queue = session.mail->queue<live_video_mode_ack_t>(mail::live_video_mode_ack, LIVE_VIDEO_MODE_ACK_QUEUE_LIMIT);
        session.launch_session_id = 0x70000001U;
        session.device_name = "atomic presentation fixture";
        session.live_video_mode_worker.process_hooks = std::make_shared<const atomic_presentation_process_hooks_t>(std::move(hooks));
        session.video->encode_wake->install([this] {
          std::unique_lock lock(wake_mutex);
          if (!wake_armed) {
            return;
          }
          wake_armed = false;
          wake_entered = true;
          wake_changed.notify_all();
          if (!wake_changed.wait_for(lock, 10s, [this] {
                return wake_released;
              })) {
            wake_timed_out = true;
          }
        });
      }

      ~impl_t() {
        release_wake();
        stop_live_video_mode_worker(session);
        join_live_video_mode_worker(session);
      }

      void release_wake() {
        std::lock_guard lock(wake_mutex);
        wake_armed = false;
        wake_released = true;
        wake_changed.notify_all();
      }

      mutable std::mutex wake_mutex;
      std::condition_variable wake_changed;
      bool wake_armed = false;
      bool wake_entered = false;
      bool wake_released = true;
      bool wake_timed_out = false;
      // Its real worker is joined before the gate or any captured session state can die.
      session_t session {};
    };

    atomic_presentation_fixture_t::atomic_presentation_fixture_t(video::effective_video_mode_t initial, atomic_presentation_process_hooks_t hooks):
        impl_(std::make_unique<impl_t>(initial, std::move(hooks))) {}

    atomic_presentation_fixture_t::~atomic_presentation_fixture_t() = default;

    void atomic_presentation_fixture_t::submit(video::video_mode_change_t change) {
      auto &session = impl_->session;
      dispatch_live_video_mode(session, {session.mail, session.video, session.config.monitor.effective_mode, change, session.launch_session_id, session.device_name});
    }

    std::optional<video::video_mode_change_t> atomic_presentation_fixture_t::pop_encode_request(std::chrono::milliseconds timeout) {
      return impl_->session.control.video_mode_queue->pop(timeout);
    }

    void atomic_presentation_fixture_t::publish_effective_mode(video::effective_video_mode_t mode) {
      impl_->session.config.monitor.effective_mode->publish(mode);
    }

    void atomic_presentation_fixture_t::deliver_completion(video::video_mode_applied_t completion) {
      (void) impl_->session.config.monitor.atomic_presentation->complete(std::move(completion));
    }

    std::optional<live_video_mode_ack_t> atomic_presentation_fixture_t::pop_ack(std::chrono::milliseconds timeout) {
      return impl_->session.control.live_video_mode_ack_queue->pop(timeout);
    }

    void atomic_presentation_fixture_t::gate_next_encode_wake() {
      std::lock_guard lock(impl_->wake_mutex);
      impl_->wake_armed = true;
      impl_->wake_entered = false;
      impl_->wake_released = false;
    }

    bool atomic_presentation_fixture_t::wait_encode_wake(std::chrono::milliseconds timeout) {
      std::unique_lock lock(impl_->wake_mutex);
      return impl_->wake_changed.wait_for(lock, timeout, [this] {
        return impl_->wake_entered;
      });
    }

    void atomic_presentation_fixture_t::release_encode_wake() {
      impl_->release_wake();
    }

    bool atomic_presentation_fixture_t::gate_timed_out() const {
      std::lock_guard lock(impl_->wake_mutex);
      return impl_->wake_timed_out;
    }

    void atomic_presentation_fixture_t::stop() {
      stop_live_video_mode_worker(impl_->session);
      impl_->session.shutdown_event->raise(true);
    }

    void atomic_presentation_fixture_t::join() {
      join_live_video_mode_worker(impl_->session);
    }

    bool atomic_presentation_fixture_t::stopping() const {
      return impl_->session.live_video_mode_worker.transactions->stopping();
    }

    std::uint32_t atomic_presentation_fixture_t::launch_session_id() const {
      return impl_->session.launch_session_id;
    }

    bool atomic_presentation_fixture_t::pacing_diagnostics_rearmed() const {
      const auto &video = impl_->session.video;
      return !video->pacing_plan_logged.load(std::memory_order_acquire) && !video->idr_pacing_plan_logged.load(std::memory_order_acquire);
    }
  }  // namespace testing
#endif
}  // namespace stream
