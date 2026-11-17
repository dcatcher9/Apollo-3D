/**
 * @file src/atomic_presentation.h
 * @brief Private per-session ownership of atomic-presentation requests and encoder completion.
 */
#pragma once

#include "thread_safe.h"
#include "video.h"

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace stream {
  struct video_channel_t;
}

namespace stream::detail {
  /** Plain transaction state only: no session pointer, thread, network peer or callback. */
  class atomic_presentation_t {
  public:
    struct request_t {
      safe::mail_t mail;
      std::shared_ptr<video_channel_t> video;
      // The last encoder proof remains a rollback target after an unprovable desktop change.
      std::shared_ptr<video::effective_video_mode_publisher_t> effective_mode;
      video::video_mode_change_t change;
      std::uint32_t launch_session_id;
      std::string client_name;
    };

    using request_ptr = std::shared_ptr<const request_t>;

    struct submission_t {
      request_ptr superseded;
      request_ptr rejected;
    };

    atomic_presentation_t() = default;
    ~atomic_presentation_t();
    atomic_presentation_t(const atomic_presentation_t &) = delete;
    atomic_presentation_t &operator=(const atomic_presentation_t &) = delete;

    /** Assign identity to the owned payload. The caller signals after queuing superseded ACKs. */
    submission_t submit(request_t request);
    void notify_pending();
    /** One worker owns the returned immutable request through display/encoder work and rollback. */
    request_ptr wait_next();
    /** Accept one matching receipt per active token, including before the worker starts waiting. */
    bool complete(video::video_mode_applied_t completion);
    std::unique_ptr<video::video_mode_applied_t> wait_completion(std::uint64_t transaction_id);
    /** Release active admission only after rollback and plain ACK enqueue have completed. */
    bool finish(std::uint64_t transaction_id);
    bool stopping() const;
    /** Wake both waits, discard pending work, and leave native work already running alone. */
    void stop();
    /** Called after the worker joins so retained endpoint copies keep no session dependencies. */
    void clear_stopped();

  private:
    enum class receipt_phase_e {
      awaiting,
      received,
    };

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    bool stopping_ = false;
    std::uint64_t last_transaction_id_ = 0;
    request_ptr pending_;
    request_ptr active_;
    receipt_phase_e receipt_phase_ = receipt_phase_e::awaiting;
    std::unique_ptr<video::video_mode_applied_t> completion_;
  };
}  // namespace stream::detail
