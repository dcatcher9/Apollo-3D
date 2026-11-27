/**
 * @file src/stream.cpp
 * @brief Definitions for stream sessions and the shared broadcast sockets.
 */

// standard includes
#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>

extern "C" {
  // clang-format off
#include <moonlight-common-c/src/Limelight-internal.h>
  // clang-format on
}

// local includes
#include "config.h"
#include "crypto.h"
#include "globals.h"
#include "input.h"
#include "logging.h"
#include "network.h"
#include "platform/common.h"
#include "rtsp.h"
#include "session_join_watchdog.h"
#include "stream.h"
#include "stream_internal.h"
#include "sync.h"
#include "thread_safe.h"
#include "utility.h"
#include "video.h"

namespace asio = boost::asio;
namespace sys = boost::system;

using asio::ip::tcp;
using asio::ip::udp;

using namespace std::literals;

namespace stream {
  int start_broadcast(broadcast_ctx_t &ctx);
  void end_broadcast(broadcast_ctx_t &ctx);

  static auto broadcast = safe::make_shared<broadcast_ctx_t>(start_broadcast, end_broadcast);

  void recvThread(broadcast_ctx_t &ctx) {
    std::array<std::optional<av_ping_route_t>, 2> routes;

    auto &video_sock = ctx.video_sock;
    auto &audio_sock = ctx.audio_sock;

    auto &route_updates = ctx.av_ping_route_updates;
    auto broadcast_shutdown_event = mail::man->event<bool>(mail::broadcast_shutdown);

    auto &io = ctx.io_context;

    std::array<udp::endpoint, 2> peers;
    std::array<std::array<char, 2048>, 2> buffers;
    std::array<std::function<void(const boost::system::error_code, size_t)>, 2> recv_func;

    auto route_index = [](socket_e type) {
      return type == socket_e::video ? 0U : 1U;
    };

    auto apply_route_updates = [&]() {
      while (route_updates->peek()) {
        auto update = route_updates->pop();
        if (!update) {
          break;
        }

        auto &route = routes[route_index(update->socket_type)];
        if (update->route) {
          route = std::move(update->route);
        } else if (route && route->id == update->id) {
          // A stopped session may publish removal after its successor has registered. Remove only
          // the generation that issued this update, never the newer route occupying the slot.
          route.reset();
        }
      }
    };

    auto recv_func_init = [&](udp::socket &sock, std::size_t index) {
      recv_func[index] = [&, index](const boost::system::error_code &ec, size_t bytes) {
        auto fg = util::fail_guard([&]() {
          sock.async_receive_from(asio::buffer(buffers[index]), peers[index], 0, recv_func[index]);
        });

        apply_route_updates();

        // No data, yet no error
        if (ec == boost::system::errc::connection_refused || ec == boost::system::errc::connection_reset) {
          return;
        }

        if (ec || !bytes) {
          BOOST_LOG(error) << "Couldn't receive data from udp socket: "sv << ec.message();
          return;
        }

        auto &route = routes[index];
        if (!route) {
          return;
        }

        if (bytes == sizeof(SS_PING)) {
          auto ping = reinterpret_cast<PSS_PING>(buffers[index].data());
          if (route->payload == std::string_view {ping->payload, sizeof(ping->payload)}) {
            route->messages->raise(
              peers[index],
              std::string {buffers[index].data(), bytes}
            );
          }
        }
      };
    };

    recv_func_init(video_sock, 0);
    recv_func_init(audio_sock, 1);

    video_sock.async_receive_from(asio::buffer(buffers[0]), peers[0], 0, recv_func[0]);
    audio_sock.async_receive_from(asio::buffer(buffers[1]), peers[1], 0, recv_func[1]);

    while (!broadcast_shutdown_event->peek()) {
      io.run();
    }
  }

  int start_broadcast(broadcast_ctx_t &ctx) {
    auto address_family = net::af_from_enum_string(config::sunshine.address_family);
    boost::system::error_code ec;
    const auto bind_address_string = net::get_bind_address(address_family);
    if (!bind_address_string) {
      BOOST_LOG(fatal) << "Stream servers refused invalid bind_address ["sv << config::sunshine.bind_address << ']';
      return -1;
    }
    const auto bind_address = boost::asio::ip::make_address(*bind_address_string, ec);
    if (ec) {
      BOOST_LOG(fatal) << "Invalid stream bind address ["sv << *bind_address_string << "]: "sv << ec.message();
      return -1;
    }
    auto protocol = bind_address.is_v4() ? udp::v4() : udp::v6();
    auto control_port = net::map_port(CONTROL_PORT);
    auto video_port = net::map_port(VIDEO_STREAM_PORT);
    auto audio_port = net::map_port(AUDIO_STREAM_PORT);

    if (ctx.control_server.bind(address_family, control_port)) {
      BOOST_LOG(error) << "Couldn't bind Control server to port ["sv << control_port << "], likely another process already bound to the port"sv;

      return -1;
    }

    ctx.video_sock.open(protocol, ec);
    if (ec) {
      BOOST_LOG(fatal) << "Couldn't open socket for Video server: "sv << ec.message();

      return -1;
    }

    // Set video socket send buffer size (SO_SENDBUF) to 1MB
    try {
      ctx.video_sock.set_option(boost::asio::socket_base::send_buffer_size(1024 * 1024));
    } catch (...) {
      BOOST_LOG(error) << "Failed to set video socket send buffer size (SO_SENDBUF)";
    }

    ctx.video_sock.bind(udp::endpoint(bind_address, video_port), ec);
    if (ec) {
      BOOST_LOG(fatal) << "Couldn't bind Video server to port ["sv << video_port << "]: "sv << ec.message();

      return -1;
    }

    ctx.audio_sock.open(protocol, ec);
    if (ec) {
      BOOST_LOG(fatal) << "Couldn't open socket for Audio server: "sv << ec.message();

      return -1;
    }

    ctx.audio_sock.bind(udp::endpoint(bind_address, audio_port), ec);
    if (ec) {
      BOOST_LOG(fatal) << "Couldn't bind Audio server to port ["sv << audio_port << "]: "sv << ec.message();

      return -1;
    }

    ctx.av_ping_route_updates = std::make_shared<av_ping_route_queue_t::element_type>(30);

    // Thread construction can throw after one or more workers have started. Roll back every
    // partially initialized worker before shared_t destroys and retries this broadcast object.
    auto rollback = util::fail_guard([&ctx] {
      end_broadcast(ctx);
    });

    ctx.video_epoch = std::chrono::steady_clock::now();
    ctx.control_thread = std::thread {controlBroadcastThread, &ctx.control_server};

    ctx.recv_thread = std::thread {recvThread, std::ref(ctx)};

    rollback.disable();
    return 0;
  }

  void end_broadcast(broadcast_ctx_t &ctx) {
    auto broadcast_shutdown_event = mail::man->event<bool>(mail::broadcast_shutdown);

    broadcast_shutdown_event->raise(true);

    if (ctx.av_ping_route_updates) {
      ctx.av_ping_route_updates->stop();
    }
    ctx.io_context.stop();

    boost::system::error_code ec;
    ctx.video_sock.close(ec);
    ctx.audio_sock.close(ec);

    BOOST_LOG(debug) << "Waiting for main listening thread to end..."sv;
    if (ctx.recv_thread.joinable()) {
      ctx.recv_thread.join();
    }
    BOOST_LOG(debug) << "Waiting for main control thread to end..."sv;
    if (ctx.control_thread.joinable()) {
      ctx.control_thread.join();
    }
    BOOST_LOG(debug) << "All broadcasting threads ended"sv;

    broadcast_shutdown_event->reset();
  }

  int recv_ping(session_t *session, decltype(broadcast)::ptr_t ref, socket_e type, std::string_view expected_payload, udp::endpoint &peer, std::chrono::milliseconds timeout) {
    if (expected_payload.size() != sizeof(SS_PING::payload)) {
      BOOST_LOG(error) << "Refusing invalid A/V ping identity length: "sv << expected_payload.size();
      return -1;
    }

    auto messages = std::make_shared<message_queue_t::element_type>(30);
    ref->av_ping_route_updates->raise(av_ping_route_update_t {
      type,
      session->launch_session_id,
      av_ping_route_t {
        session->launch_session_id,
        std::string {expected_payload},
        messages,
      },
    });

    auto fg = util::fail_guard([&]() {
      messages->stop();
      ref->av_ping_route_updates->raise(av_ping_route_update_t {
        type,
        session->launch_session_id,
        std::nullopt,
      });
    });

    // Nothing raises this queue when the session stops, so wait in short slices and look at
    // the session's own stop event between them: a session ended before its ping arrived
    // must not hold its teardown for the whole ping timeout.
    constexpr auto stop_poll_interval = 50ms;
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;) {
      if (session->shutdown_event->peek()) {
        BOOST_LOG(info) << "Session stopped before its initial "sv
                        << (type == socket_e::video ? "video"sv : "audio"sv) << " ping."sv;
        return -1;
      }
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        break;
      }
      const auto slice = std::min(
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now) + 1ms,
        std::chrono::milliseconds {stop_poll_interval}
      );
      auto msg_opt = messages->pop(slice);
      if (!msg_opt) {
        if (!messages->running()) {
          break;
        }
        continue;
      }

      auto &recv_peer = msg_opt->first;
      BOOST_LOG(debug) << "Received initial "sv << (type == socket_e::video ? "video"sv : "audio"sv)
                       << " SS ping from "sv << recv_peer.address() << ':' << recv_peer.port();

      // Update connection details.
      peer = recv_peer;
      return 0;
    }

    BOOST_LOG(error) << "Initial Ping Timeout"sv;
    return -1;
  }

  void videoThread(session_t *session) {
    auto fg = util::fail_guard([&]() {
      session::stop(*session);
    });

    auto ref = broadcast.ref();
    auto error = recv_ping(session, ref, socket_e::video, session->video->ping_payload, session->video->peer, config::stream.ping_timeout);
    if (error < 0) {
      return;
    }

    // Register for the UDP ping immediately so an early client ping is not lost, then wait until
    // platform resume and the control connection have both completed before emitting any frames.
    if (!session->startup_ready.view() || !session->control_ready.view() || session->state.load(std::memory_order_acquire) != session::state_e::RUNNING) {
      return;
    }

    // Enable local prioritization and QoS tagging on video traffic if requested by the client
    auto address = session->video->peer.address();
    session->video->qos = platf::enable_socket_qos(ref->video_sock.native_handle(), address, session->video->peer.port(), platf::qos_data_type_e::video, session->config.videoQosType != 0);

    BOOST_LOG(debug) << "Start capturing Video"sv;
    video::capture(session->mail, session->config.monitor, session->media_senders.video_packets());
  }

  void audioThread(session_t *session) {
    auto fg = util::fail_guard([&]() {
      session::stop(*session);
    });

    auto ref = broadcast.ref();
    auto error = recv_ping(session, ref, socket_e::audio, session->audio->ping_payload, session->audio->peer, config::stream.ping_timeout);
    if (error < 0) {
      return;
    }

    // Audio pings commonly arrive before a resumed virtual display is ready. Holding capture here
    // prevents packets from being sent with an uninitialized local address (WSAEINVAL/10022).
    if (!session->startup_ready.view() || !session->control_ready.view() || session->state.load(std::memory_order_acquire) != session::state_e::RUNNING) {
      return;
    }

    // Enable local prioritization and QoS tagging on audio traffic if requested by the client
    auto address = session->audio->peer.address();
    session->audio->qos = platf::enable_socket_qos(ref->audio_sock.native_handle(), address, session->audio->peer.port(), platf::qos_data_type_e::audio, session->config.audioQosType != 0);

    BOOST_LOG(debug) << "Start capturing Audio"sv;
    audio::capture(session->mail, session->config.audio, session->media_senders.audio_packets());
  }

  namespace session {
    namespace {
      // Construct all four per-session workers behind a shared gate. If creating any std::thread
      // throws, already-created workers are released as no-ops and joined before their thread
      // objects are destroyed. This keeps a resource-exhaustion failure from terminating the host.
      class pending_session_workers_t {
      public:
        ~pending_session_workers_t() {
          cancel();
        }

        pending_session_workers_t(const pending_session_workers_t &) = delete;
        pending_session_workers_t &operator=(const pending_session_workers_t &) = delete;

        pending_session_workers_t() = default;

        bool prepare(
          std::function<void()> audio_entry,
          std::function<void()> video_entry,
          std::function<void()> video_sender_entry,
          std::function<void()> audio_sender_entry,
          std::size_t fail_before_worker_for_test = std::numeric_limits<std::size_t>::max(),
          std::function<void()> failure_handler = {}
        ) noexcept {
          try {
            gate_ = std::make_shared<std::promise<bool>>();
            auto ready = gate_->get_future().share();
            auto start_worker = [ready, failure_handler, fail_before_worker_for_test](std::size_t index, std::string_view name, std::function<void()> entry) {
#ifdef SUNSHINE_TESTS
              if (fail_before_worker_for_test == index) {
                throw std::runtime_error {"injected session-worker start failure"};
              }
#else
              (void) index;
              (void) fail_before_worker_for_test;
#endif
              return std::thread {[ready, failure_handler, name, entry = std::move(entry)]() mutable {
                try {
                  if (!ready.get()) {
                    return;
                  }
                  entry();
                } catch (const std::future_error &) {
                  // A broken startup gate is equivalent to a cancelled session.
                  return;
                } catch (const std::exception &exception) {
                  BOOST_LOG(error) << name << " session worker failed: "sv << exception.what();
                } catch (...) {
                  BOOST_LOG(error) << name << " session worker failed with an unknown error."sv;
                }

                if (failure_handler) {
                  failure_handler();
                }
              }};
            };
            audio_ = start_worker(0, "Audio"sv, std::move(audio_entry));
            video_ = start_worker(1, "Video"sv, std::move(video_entry));
            video_sender_ = start_worker(2, "Video sender"sv, std::move(video_sender_entry));
            audio_sender_ = start_worker(3, "Audio sender"sv, std::move(audio_sender_entry));
            return true;
          } catch (const std::exception &exception) {
            if (fail_before_worker_for_test == std::numeric_limits<std::size_t>::max()) {
              BOOST_LOG(error) << "Failed to create streaming session workers: "sv << exception.what();
            }
          } catch (...) {
            if (fail_before_worker_for_test == std::numeric_limits<std::size_t>::max()) {
              BOOST_LOG(error) << "Failed to create streaming session workers with an unknown error."sv;
            }
          }

          cancel();
          return false;
        }

        void commit(session_t &session) noexcept {
          session.audioThread = std::move(audio_);
          session.videoThread = std::move(video_);
          session.media_senders.adopt_video(std::move(video_sender_));
          session.media_senders.adopt_audio(std::move(audio_sender_));
          resolve(true);
        }

        void cancel() noexcept {
          resolve(false);
          if (audio_sender_.joinable()) {
            audio_sender_.join();
          }
          if (video_sender_.joinable()) {
            video_sender_.join();
          }
          if (video_.joinable()) {
            video_.join();
          }
          if (audio_.joinable()) {
            audio_.join();
          }
        }

        [[nodiscard]] bool has_joinable_worker() const noexcept {
          return audio_.joinable() || video_.joinable() || video_sender_.joinable() || audio_sender_.joinable();
        }

      private:
        void resolve(bool run) noexcept {
          if (!gate_) {
            return;
          }

          try {
            gate_->set_value(run);
          } catch (const std::future_error &) {
            // Resolution is intentionally idempotent for cleanup paths.
          }
          gate_.reset();
        }

        std::shared_ptr<std::promise<bool>> gate_;
        std::thread audio_;
        std::thread video_;
        std::thread video_sender_;
        std::thread audio_sender_;
      };

      template<class Clock>
      void publish_control_registration(
        session_t &session,
        control_server_t::session_slot_t &slot,
        pending_session_workers_t &workers,
        std::chrono::milliseconds ping_timeout,
        Clock &&now
      ) {
        // The caller holds the control slot lock. Platform/input setup and worker construction
        // must not consume the client's connection window before the session is visible.
        session.pingTimeout = now() + ping_timeout;
        session.state.store(state_e::RUNNING, std::memory_order_relaxed);
        session.startup_ready.raise(true);
        slot.session = &session;
        slot.peer = nullptr;
        workers.commit(session);
      }
    }  // namespace

    state_e state(session_t &session) {
      return session.state.load(std::memory_order_relaxed);
    }

#ifdef SUNSHINE_TESTS
    void set_state_for_test(session_t &session, state_e state) {
      session.state.store(state, std::memory_order_relaxed);
    }

    bool worker_start_rollback_for_test(std::size_t fail_before_worker) {
      std::atomic_int executed {};
      pending_session_workers_t workers;
      const auto entry = [&executed]() {
        ++executed;
      };
      const bool prepared = workers.prepare(
        entry,
        entry,
        entry,
        entry,
        fail_before_worker
      );
      return !prepared && executed.load() == 0 && !workers.has_joinable_worker();
    }

    control_registration_test_result_t control_registration_after_preparation_for_test(
      std::chrono::milliseconds preparation,
      std::chrono::milliseconds ping_timeout
    ) {
      control_registration_test_result_t result;
      session_t session;
      sync_util::sync_t<control_server_t::session_slot_t> slot;
      auto now = std::chrono::steady_clock::time_point {1s};
      // Seed the old startup-era deadline, then model platform/input setup taking longer than it.
      session.pingTimeout = now + ping_timeout;
      session.state.store(state_e::STOPPED, std::memory_order_relaxed);
      now += preparation;
      result.expired_before_publication = now > session.pingTimeout;
      std::atomic_int ready_workers {};
      pending_session_workers_t workers;
      const auto observe_registration = [&]() {
        auto lock = slot.lock();
        if (slot->session == &session && !slot->peer && session.state.load(std::memory_order_relaxed) == state_e::RUNNING && session.startup_ready.peek() && session.pingTimeout == now + ping_timeout) {
          ++ready_workers;
        }
      };
      result.workers_prepared = workers.prepare(observe_registration, observe_registration, observe_registration, observe_registration);
      if (!result.workers_prepared) {
        return result;
      }
      {
        auto lock = slot.lock();
        publish_control_registration(session, *slot, workers, ping_timeout, [&]() {
          ++result.clock_samples;
          return now;
        });
        result.remaining_ping_budget = session.pingTimeout - now;
      }
      session.audioThread.join();
      session.videoThread.join();
      session.media_senders.stop_and_join();
      result.ready_workers = ready_workers.load();
      return result;
    }
#endif

    std::string uuid(const session_t &session) {
      return session.device_uuid;
    }

    bool uuid_match(const session_t &session, const std::string_view &uuid) {
      return session.device_uuid == uuid;
    }

    std::string client_name(const session_t &session) {
      std::lock_guard lock(session.device_info_mutex);
      return session.device_name;
    }

    crypto::PERM permissions(const session_t &session) {
      return session.permission.load(std::memory_order_acquire);
    }

    client_policy_result_e update_client_policy(
      session_t &session,
      std::uint64_t generation,
      std::string_view name,
      crypto::PERM new_permissions,
      bool revoked
    ) {
      std::string previous_name;
      bool should_stop;
      {
        // Serialize policy publications per session. Merely making the permission atomic is not
        // enough: two administrator updates can otherwise publish out of generation order.
        std::lock_guard lock(session.device_info_mutex);
        if (generation <= session.client_policy_generation) {
          return client_policy_result_e::ignored;
        }

        session.client_policy_generation = generation;
        if (session.input) {
          input::update_permissions(session.input, new_permissions);
        }
        session.permission.store(new_permissions, std::memory_order_release);
        if (!(new_permissions & crypto::PERM::input_microphone) && session.microphone_receiver) {
          session.microphone_receiver->stop();
        }
        previous_name = session.device_name;
        session.device_name = name;
        should_stop = revoked || !(new_permissions & crypto::PERM::_allow_view);
      }

      if (should_stop) {
        BOOST_LOG(debug) << "Session: Client authorization revoked for [" << previous_name << "], disconnecting...";
        return client_policy_result_e::disconnect;
      }

      BOOST_LOG(debug) << "Session: Permission updated for [" << previous_name << "]";
      if (previous_name != name) {
        BOOST_LOG(debug) << "Session: Device name changed from [" << previous_name << "] to [" << name << "]";
      }
      return client_policy_result_e::updated;
    }

    static bool transition_to_stopping(session_t &session, bool graceful) {
      std::lock_guard lock(session.stop_mutex);
      if (session.state.load(std::memory_order_relaxed) != state_e::RUNNING) {
        return false;
      }

      session.video->active.store(false, std::memory_order_release);
      session.audio->active.store(false, std::memory_order_release);
      if (session.microphone_receiver) {
        session.microphone_receiver->stop();
      }
      if (session.control.sbs_telemetry_subscription) {
        session.control.sbs_telemetry_subscription->update(
          false,
          false,
          SBS_TELEMETRY_INTERVAL_BACKGROUND_MS
        );
      }
      session.startup_ready.stop();
      session.control_ready.stop();
      session.graceful_stop_requested.store(graceful, std::memory_order_relaxed);
      session.state.store(state_e::STOPPING, std::memory_order_release);
      stop_live_video_mode_worker(session);
      return true;
    }

    void stop(session_t &session) {
      if (!transition_to_stopping(session, false)) {
        return;
      }

      session.shutdown_event->raise(true);
      session.video->wake_encoder();
      rtsp_stream::notify_session_stopping();
    }

    void graceful_stop(session_t &session) {
      if (!transition_to_stopping(session, true)) {
        return;
      }

      session.shutdown_event->raise(true);
      session.video->wake_encoder();
      rtsp_stream::notify_session_stopping();
    }

    bool stop_if_client_policy_current(session_t &session, std::uint64_t generation, bool graceful) {
      {
        // Make validation and the RUNNING -> STOPPING claim one transaction with policy updates.
        // A delayed revoke can no longer stop a session after a newer allow policy was applied.
        std::lock_guard lock(session.device_info_mutex);
        if (generation != session.client_policy_generation || !transition_to_stopping(session, graceful)) {
          return false;
        }
      }

      session.shutdown_event->raise(true);
      session.video->wake_encoder();
      rtsp_stream::notify_session_stopping();
      return true;
    }

    void join(session_t &session) {
      if (session.microphone_receiver) {
        session.microphone_receiver->stop();
      }
      // A live desktop resize owns process/display state and can outlive the control thread.
      // Reap it before arming the NVENC hang watchdog and before releasing the active-session
      // slot, so a successor can never overlap stale work from this launch.
      {
        const detail::session_join_watchdog_t mode_worker_watchdog {
          [](std::function<void()> expired) {
            return task_pool.pushDelayed(std::move(expired), 30s);
          },
          [](auto task_id) {
            task_pool.cancel(task_id);
          },
          []() {
            BOOST_LOG(fatal) << "Hang detected! Live video-mode worker failed to terminate in 30 seconds."sv;
            logging::log_flush();
            lifetime::debug_trap();
          },
        };
        join_live_video_mode_worker(session);
      }

      // Current Nvidia drivers have a bug where NVENC can deadlock the encoder thread with hardware-accelerated
      // GPU scheduling enabled. If this happens, we will terminate ourselves and the service can restart.
      // The alternative is that Sunshine can never start another session until it's manually restarted.
      detail::join_workers_before_session_cleanup(
        []() {
          return detail::session_join_watchdog_t {
            [](std::function<void()> expired) {
              return task_pool.pushDelayed(std::move(expired), 10s);
            },
            [](auto task_id) {
              task_pool.cancel(task_id);
            },
            []() {
              BOOST_LOG(fatal) << "Hang detected! Session failed to terminate in 10 seconds."sv;
              logging::log_flush();
              lifetime::debug_trap();
            },
          };
        },
        [&session]() {
          BOOST_LOG(debug) << "Waiting for video to end..."sv;
          session.videoThread.join();
          BOOST_LOG(debug) << "Waiting for audio to end..."sv;
          session.audioThread.join();
          BOOST_LOG(debug) << "Waiting for control to end..."sv;
          session.controlEnd.view();
          // Reset input on session stop to avoid stuck repeated keys.
          BOOST_LOG(debug) << "Resetting Input..."sv;
          input::reset(session.input).wait();
        },
        [&session]() {
          // Worker shutdown is complete and its watchdog is disarmed. App exit, undo commands
          // and display restoration have separate bounds and may legitimately take longer.
          session.media_senders.stop_and_join();
          // Keep the active slot until this lock is held so a successor cannot overlap teardown.
          std::lock_guard lifecycle_lock(platform_lifecycle_mutex);
          if (!remote_session_active) {
            BOOST_LOG(error) << "Streaming session ended without owning the active-session slot."sv;
          } else {
            remote_session_active = false;
            retain_or_stop_session_locked();
          }
        }
      );

      BOOST_LOG(debug) << "Session ended"sv;
    }

    int start(session_t &session) {
      session.broadcast_ref = broadcast.ref();
      if (!session.broadcast_ref) {
        return -1;
      }

      // Claim the sole session under the same lifecycle lock used by launch-time display
      // mutation and encoder probing. A stale RTSP handshake therefore cannot create a second
      // capture/encoder/input stack behind the HTTP admission check.
      {
        std::lock_guard lifecycle_lock(platform_lifecycle_mutex);
        if (remote_session_active) {
          BOOST_LOG(warning) << "Rejecting a second active streaming session."sv;
          return -1;
        }
        remote_session_active = true;
        if (!activate_remote_session_locked(session.launch_session_id)) {
          remote_session_active = false;
          return -1;
        }
        // Capture input ownership only after admission, under the same lock that protects
        // launch/replacement. A stale handshake must not inject its delayed startup mouse nudge.
        session.input = input::alloc(
          session.mail,
          permissions(session),
          session.config.client_supports_authored_pcm
        );
      }

      auto rollback_active_session = []() {
        std::lock_guard lifecycle_lock(platform_lifecycle_mutex);
        if (remote_session_active) {
          remote_session_active = false;
          retain_or_stop_session_locked();
        }
      };

      pending_session_workers_t workers;
      if (!workers.prepare(
            [&session]() {
              audioThread(&session);
            },
            [&session]() {
              videoThread(&session);
            },
            [&session]() {
              run_video_sender(
                session.broadcast_ref->video_sock,
                session.media_senders.video_packets(),
                session.video,
                session.shutdown_event,
                session.broadcast_ref->video_epoch
              );
            },
            [&session]() {
              run_audio_sender(
                session.broadcast_ref->audio_sock,
                session.media_senders.audio_packets(),
                session.audio,
                session.shutdown_event
              );
            },
            std::numeric_limits<std::size_t>::max(),
            [&session]() {
              stop(session);
            }
          )) {
        rollback_active_session();
        return -1;
      }

      // Publish the raw control registration only after platform startup and all worker-thread
      // constructions succeed. Its owner remains the RTSP slot; controlEnd is the lifetime barrier.
      {
        auto slot_lock = session.broadcast_ref->control_server._session.lock();
        if (session.broadcast_ref->control_server._session->session) {
          BOOST_LOG(error) << "Control server still has a registered session while the active slot is empty."sv;
          workers.cancel();
          rollback_active_session();
          return -1;
        }

        // A/V capture may already have received its UDP ping. Release it only after display/audio
        // platform state has finished resuming; control_ready independently protects source routing.
        publish_control_registration(
          session,
          *session.broadcast_ref->control_server._session,
          workers,
          config::stream.ping_timeout,
          std::chrono::steady_clock::now
        );
        if (session.microphone_receiver) {
          if (!!(permissions(session) & crypto::PERM::input_microphone)) {
            session.microphone_receiver->activate();
          } else {
            session.microphone_receiver->stop();
          }
        }
      }

      return 0;
    }

    std::shared_ptr<session_t> alloc(config_t &config, rtsp_stream::launch_session_t &launch_session) {
      auto session = std::make_shared<session_t>();

      auto mail = std::make_shared<safe::mail_raw_t>();

      session->shutdown_event = mail->event<bool>(mail::shutdown);
      session->launch_session_id = launch_session.id;
      session->device_name = launch_session.display_context->device_name;
      session->device_uuid = launch_session.display_context->unique_id;
      session->client_policy_generation = 0;
      session->permission.store(launch_session.perm, std::memory_order_relaxed);

      session->config = config;
      session->microphone_receiver = launch_session.take_microphone();

      // Seed the effective-mode view with the launch mode so a live request refused before the
      // encode loop has published anything still reports a real mode rather than zeros. The video
      // thread receives a copy of session->config.monitor, so both sides share this publisher.
      session->config.monitor.effective_mode = std::make_shared<video::effective_video_mode_publisher_t>(
        video::effective_video_mode_t {
          config.monitor.width,
          config.monitor.height,
          config.monitor.framerateX100 > 0 ? config.monitor.framerateX100 : config.monitor.framerate * 100,
          config.monitor.bitrate,
          config.monitor.width,
          config.monitor.height,
          0,
          0,
          config.monitor.sbs_mode,
          0,
        }
      );
      session->config.monitor.sbs_telemetry_subscription =
        std::make_shared<video::sbs_telemetry_subscription_t>();
      session->config.monitor.stream_gamma_state = std::make_shared<video::stream_gamma_publisher_t>(
        session->config.monitor.stream_gamma
      );

      session->video = std::make_shared<video_channel_t>();
      session->video->packet_size = config.packetsize;
      session->video->min_required_fec_packets = config.minRequiredFecPackets;
      session->video->source_frame_id_negotiated = config.client_supports_source_frame_id_v1;
      session->video->effective_mode = session->config.monitor.effective_mode;
      session->config.monitor.encode_wake = session->video->encode_wake;
      session->config.monitor.atomic_presentation = session->live_video_mode_worker.transactions;
      session->config.monitor.sbs_debug_dump_pending =
        session->video->sbs_debug_dump_pending;
      session->requested_sbs_mode->store(
        session->config.monitor.sbs_mode,
        std::memory_order_relaxed
      );
      session->config.monitor.requested_sbs_mode = session->requested_sbs_mode;

      session->audio = std::make_shared<audio_channel_t>();
      session->audio->packet_duration = config.audio.packetDuration;

      session->control.connect_data = launch_session.control_connect_data;
      session->control.feedback_queue = mail->queue<platf::gamepad_feedback_msg_t>(mail::gamepad_feedback);
      session->control.haptics_feedback_queue =
        mail->queue<platf::gamepad_feedback_msg_t>(mail::ds5_haptics_feedback, 64);
      session->control.hdr_queue = mail->event<video::hdr_info_t>(mail::hdr);
      session->control.depth_status_event = mail->event<int>(mail::sbs_depth_status);
      session->control.live_video_mode_ack_queue = mail->queue<live_video_mode_ack_t>(
        mail::live_video_mode_ack,
        LIVE_VIDEO_MODE_ACK_QUEUE_LIMIT
      );
      session->control.video_mode_queue =
        mail->queue<video::video_mode_change_t>(mail::video_mode);
      session->control.stream_gamma_queue = mail->queue<video::stream_gamma_request_t>(
        mail::stream_gamma, STREAM_GAMMA_QUEUE_LIMIT
      );
      session->control.stream_gamma_ack_queue = mail->queue<video::stream_gamma_ack_t>(
        mail::stream_gamma_ack, 2 * STREAM_GAMMA_QUEUE_LIMIT
      );
      session->control.sbs_telemetry_event =
        mail->event<video::sbs_telemetry_snapshot_t>(mail::sbs_telemetry);
      session->control.game_source_status_event =
        mail->event<video::game_source_state_t>(mail::game_source_status);
      session->config.monitor.game_source_status_event = session->control.game_source_status_event;
      session->control.sbs_telemetry_subscription =
        session->config.monitor.sbs_telemetry_subscription;
      session->control.cipher = crypto::cipher::gcm_t {
        launch_session.gcm_key,
        false
      };

      session->video->idr_events = mail->event<bool>(mail::idr);
      session->video->invalidate_ref_frames_events = mail->event<std::pair<int64_t, int64_t>>(mail::invalidate_ref_frames);
      session->video->lowseq = 0;
      session->video->ping_payload = launch_session.av_ping_payload;
      session->video->cipher = crypto::cipher::gcm_t {
        launch_session.gcm_key,
        false
      };
      session->video->gcm_iv_counter = 0;

      constexpr auto max_block_size = crypto::cipher::round_to_pkcs7_padded(2048);

      util::buffer_t<char> shards {RTPA_TOTAL_SHARDS * max_block_size};
      util::buffer_t<uint8_t *> shards_p {RTPA_TOTAL_SHARDS};

      for (auto x = 0; x < RTPA_TOTAL_SHARDS; ++x) {
        shards_p[x] = (uint8_t *) &shards[x * max_block_size];
      }

      // Audio FEC spans multiple audio packets,
      // therefore its session specific
      session->audio->shards = std::move(shards);
      session->audio->shard_ptrs = std::move(shards_p);

      session->audio->fec_packet.rtp.header = 0x80;
      session->audio->fec_packet.rtp.packetType = 127;
      session->audio->fec_packet.rtp.timestamp = 0;
      session->audio->fec_packet.rtp.ssrc = 0;

      session->audio->fec_packet.fecHeader.payloadType = 97;
      session->audio->fec_packet.fecHeader.ssrc = 0;

      session->audio->cipher = crypto::cipher::cbc_t {
        launch_session.gcm_key,
        true
      };

      session->audio->ping_payload = launch_session.av_ping_payload;
      std::uint32_t av_ri_key_id;
      std::memcpy(&av_ri_key_id, launch_session.iv.data(), sizeof(av_ri_key_id));
      session->audio->av_ri_key_id = util::endian::big(av_ri_key_id);
      session->audio->sequence_number = 0;
      session->audio->timestamp = 0;

      session->control.peer = nullptr;
      session->state.store(state_e::STOPPED, std::memory_order_relaxed);

      session->mail = std::move(mail);

      return session;
    }
  }  // namespace session
}  // namespace stream
