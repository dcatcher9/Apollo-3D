/**
 * @file src/stream_media_test.inc
 * @brief Private, per-instance loopback fixture for the actual media sender bodies.
 */

namespace testing {
  struct media_sender_fixture_t::impl_t {
    struct lane_t {
      std::mutex mutex;
      std::condition_variable changed;
      bool ready = false;
      bool exited = false;
      bool admission_armed = false;
      bool admission_reached = false;
      bool admission_released = false;
      std::size_t processed = 0;
      std::string error;
      media_sender_test_hooks_t hooks;

      void mark_ready() {
        std::lock_guard lock {mutex};
        ready = true;
        changed.notify_all();
      }

      void before_admission() {
        std::unique_lock lock {mutex};
        if (!admission_armed) {
          return;
        }
        admission_reached = true;
        changed.notify_all();
        if (!changed.wait_for(lock, 10s, [this] {
              return admission_released;
            })) {
          error = "media admission gate timed out";
        }
        admission_armed = false;
      }

      void mark_processed() {
        std::lock_guard lock {mutex};
        ++processed;
        changed.notify_all();
      }

      void release_admission() {
        std::lock_guard lock {mutex};
        admission_released = true;
        changed.notify_all();
      }
    };

    explicit impl_t(media_sender_config_t config):
        config {std::move(config)},
        local_mail {std::make_shared<safe::mail_raw_t>()},
        video_shutdown {local_mail->event<bool>("video-shutdown")},
        audio_shutdown {local_mail->event<bool>("audio-shutdown")},
        video_channel {std::make_shared<video_channel_t>()},
        audio_channel {std::make_shared<audio_channel_t>()},
        frame_pool {std::make_shared<::video::encoded_frame_buffer_pool_t>()},
        video_socket {io_context},
        audio_socket {io_context},
        video_receiver {io_context},
        audio_receiver {io_context} {
      const auto loopback = asio::ip::address_v4::loopback();
      for (auto *socket : {&video_socket, &audio_socket, &video_receiver, &audio_receiver}) {
        socket->open(udp::v4());
        socket->bind(udp::endpoint {loopback, 0});
      }
      video_receiver.non_blocking(true);
      audio_receiver.non_blocking(true);

      crypto::aes_t key {this->config.key.begin(), this->config.key.end()};
      video_channel->packet_size = this->config.video_packet_size;
      video_channel->min_required_fec_packets = this->config.video_min_fec_packets;
      video_channel->source_frame_id_negotiated = true;
      video_channel->effective_mode = std::make_shared<::video::effective_video_mode_publisher_t>(
        ::video::effective_video_mode_t {1920, 1080, 6000, 50000}
      );
      video_channel->local_address = loopback;
      video_channel->peer = video_receiver.local_endpoint();
      video_channel->lowseq = this->config.video_sequence;
      video_channel->gcm_iv_counter = this->config.video_iv_counter;
      video_channel->cipher = crypto::cipher::gcm_t {key, false};
      video_channel->idr_events = local_mail->event<bool>("video-idr");
      video_channel->invalidate_ref_frames_events = local_mail->event<std::pair<std::int64_t, std::int64_t>>("video-invalidate");

      audio_channel->packet_duration = this->config.audio_packet_duration;
      audio_channel->local_address = loopback;
      audio_channel->peer = audio_receiver.local_endpoint();
      audio_channel->cipher = crypto::cipher::cbc_t {key, true};
      audio_channel->av_ri_key_id = this->config.audio_key_id;
      audio_channel->sequence_number = this->config.audio_sequence;
      audio_channel->timestamp = this->config.audio_timestamp;
      constexpr auto max_block_size = crypto::cipher::round_to_pkcs7_padded(2048);
      audio_channel->shards = util::buffer_t<char> {RTPA_TOTAL_SHARDS * max_block_size};
      audio_channel->shard_ptrs = util::buffer_t<std::uint8_t *> {RTPA_TOTAL_SHARDS};
      for (auto index = 0; index < RTPA_TOTAL_SHARDS; ++index) {
        audio_channel->shard_ptrs[index] = reinterpret_cast<std::uint8_t *>(&audio_channel->shards[index * max_block_size]);
      }
      audio_channel->fec_packet.rtp.header = 0x80;
      audio_channel->fec_packet.rtp.packetType = 127;
      audio_channel->fec_packet.rtp.timestamp = 0;
      audio_channel->fec_packet.rtp.ssrc = 0;
      audio_channel->fec_packet.fecHeader.payloadType = 97;
      audio_channel->fec_packet.fecHeader.ssrc = 0;

      for (auto *lane : {&video_lane, &audio_lane}) {
        lane->hooks.ready = [lane] {
          lane->mark_ready();
        };
        lane->hooks.before_admission = [lane] {
          lane->before_admission();
        };
        lane->hooks.processed = [lane] {
          lane->mark_processed();
        };
      }
      video_lane.hooks.video_fec_percentage = this->config.video_fec_percentage;
    }

    ~impl_t() {
      close();
    }

    lane_t &lane(media_lane_e which) {
      return which == media_lane_e::video ? video_lane : audio_lane;
    }

    void run(media_lane_e which) {
      auto &state = lane(which);
      try {
        if (which == media_lane_e::video) {
          run_video_sender(video_socket, media_senders.video_packets(), video_channel, video_shutdown, config.epoch, &state.hooks);
        } else {
          run_audio_sender(audio_socket, media_senders.audio_packets(), audio_channel, audio_shutdown, &state.hooks);
        }
      } catch (const std::exception &exception) {
        std::lock_guard lock {state.mutex};
        state.error = exception.what();
      } catch (...) {
        std::lock_guard lock {state.mutex};
        state.error = "unknown sender exception";
      }
      std::lock_guard lock {state.mutex};
      state.exited = true;
      state.changed.notify_all();
    }

    void start() {
      if (started || stopped || closed) {
        throw std::logic_error {"media sender fixture can only start once"};
      }
      started = true;
      try {
        reed_solomon_init();
        media_senders.adopt_video(std::thread {[this] {
          run(media_lane_e::video);
        }});
        media_senders.adopt_audio(std::thread {[this] {
          run(media_lane_e::audio);
        }});
      } catch (...) {
        stop();
        throw;
      }
    }

    void finish(bool signal_shutdown) {
      if (stopped) {
        return;
      }
      if (signal_shutdown) {
        video_shutdown->raise(true);
        audio_shutdown->raise(true);
      }
      // Stop admission before opening the fixture's held after-pop gates. The actual owner
      // remains the authority for joining workers and releasing the stopped queues.
      if (media_senders.video_packets()) {
        media_senders.video_packets()->stop();
      }
      if (media_senders.audio_packets()) {
        media_senders.audio_packets()->stop();
      }
      video_lane.release_admission();
      audio_lane.release_admission();
      media_senders.stop_and_join();
      final_state.video_sequence = video_channel->lowseq;
      final_state.video_iv_counter = video_channel->gcm_iv_counter;
      final_state.audio_sequence = audio_channel->sequence_number;
      final_state.audio_timestamp = audio_channel->timestamp;
      final_state.video_shutdown = video_shutdown->peek();
      final_state.audio_shutdown = audio_shutdown->peek();
      final_state.video_error = std::move(video_lane.error);
      final_state.audio_error = std::move(audio_lane.error);
      stopped = true;
    }

    void stop() {
      finish(true);
    }

    void close() {
      if (closed) {
        return;
      }
      stop();
      video_channel.reset();
      audio_channel.reset();
      frame_pool.reset();
      sys::error_code error;
      video_socket.close(error);
      audio_socket.close(error);
      video_receiver.close(error);
      audio_receiver.close(error);
      closed = true;
    }

    media_sender_config_t config;
    safe::mail_t local_mail;
    safe::mail_raw_t::event_t<bool> video_shutdown;
    safe::mail_raw_t::event_t<bool> audio_shutdown;
    std::shared_ptr<video_channel_t> video_channel;
    std::shared_ptr<audio_channel_t> audio_channel;
    std::shared_ptr<::video::encoded_frame_buffer_pool_t> frame_pool;
    asio::io_context io_context;
    udp::socket video_socket;
    udp::socket audio_socket;
    udp::socket video_receiver;
    udp::socket audio_receiver;
    lane_t video_lane;
    lane_t audio_lane;
    bool started = false;
    bool stopped = false;
    bool closed = false;
    media_sender_state_t final_state;
    session_media_senders_t media_senders;
  };

  media_sender_fixture_t::media_sender_fixture_t(media_sender_config_t config):
      impl_ {std::make_unique<impl_t>(std::move(config))} {
  }

  media_sender_fixture_t::~media_sender_fixture_t() = default;

  void media_sender_fixture_t::start() {
    impl_->start();
  }

  bool media_sender_fixture_t::wait_ready(media_lane_e which, std::chrono::milliseconds timeout) {
    auto &lane = impl_->lane(which);
    std::unique_lock lock {lane.mutex};
    lane.changed.wait_for(lock, timeout, [&lane] {
      return lane.ready || lane.exited;
    });
    return lane.ready;
  }

  void media_sender_fixture_t::gate_next_admission(media_lane_e which) {
    auto &lane = impl_->lane(which);
    std::lock_guard lock {lane.mutex};
    if (impl_->stopped || lane.exited || lane.admission_armed) {
      throw std::logic_error {"media admission gate is unavailable"};
    }
    lane.admission_armed = true;
    lane.admission_reached = false;
    lane.admission_released = false;
  }

  bool media_sender_fixture_t::wait_admission(media_lane_e which, std::chrono::milliseconds timeout) {
    auto &lane = impl_->lane(which);
    std::unique_lock lock {lane.mutex};
    lane.changed.wait_for(lock, timeout, [&lane] {
      return lane.admission_reached || lane.exited;
    });
    return lane.admission_reached;
  }

  void media_sender_fixture_t::release_admission(media_lane_e which) {
    impl_->lane(which).release_admission();
  }

  bool media_sender_fixture_t::wait_processed(media_lane_e which, std::size_t count, std::chrono::milliseconds timeout) {
    auto &lane = impl_->lane(which);
    std::unique_lock lock {lane.mutex};
    lane.changed.wait_for(lock, timeout, [&lane, count] {
      return lane.processed >= count || lane.exited;
    });
    return lane.processed >= count;
  }

  bool media_sender_fixture_t::enqueue_video(video_packet_t packet) {
    if (!impl_->media_senders.video_packets()) {
      return false;
    }
    auto encoded = std::make_unique<::video::packet_raw_generic>(
      std::move(packet.bytes),
      packet.index,
      packet.idr,
      impl_->frame_pool
    );
    encoded->encoder_input_retained = packet.retained;
    encoded->after_ref_frame_invalidation = packet.after_invalidation;
    encoded->frame_timestamp = impl_->config.epoch + packet.pts;
    {
      std::lock_guard lock {impl_->video_lane.mutex};
      if (impl_->video_lane.admission_armed) {
        // Held-admission fixtures test cancellation, not stale-backlog rejection.
        encoded->encoded_timestamp = std::chrono::steady_clock::now() + 1h;
      }
    }
    return impl_->media_senders.video_packets()->raise_with_overflow_policy(true, std::move(encoded)).queued;
  }

  bool media_sender_fixture_t::enqueue_audio(std::vector<std::uint8_t> bytes) {
    if (!impl_->media_senders.audio_packets()) {
      return false;
    }
    if (bytes.size() >= 2048) {
      throw std::length_error {"fixture audio payload exceeds its native shard storage"};
    }
    ::audio::buffer_t payload {bytes.size()};
    std::copy(bytes.begin(), bytes.end(), std::begin(payload));
    return impl_->media_senders.audio_packets()->raise_with_overflow_policy(true, std::move(payload)).queued;
  }

  void media_sender_fixture_t::deactivate(media_lane_e which) {
    if (impl_->closed) {
      return;
    }
    if (which == media_lane_e::video) {
      impl_->video_channel->active.store(false, std::memory_order_release);
    } else {
      impl_->audio_channel->active.store(false, std::memory_order_release);
    }
  }

  void media_sender_fixture_t::request_shutdown(media_lane_e which) {
    if (which == media_lane_e::video) {
      impl_->video_shutdown->raise(true);
    } else {
      impl_->audio_shutdown->raise(true);
    }
  }

  std::vector<std::vector<std::uint8_t>> media_sender_fixture_t::receive(
    media_lane_e which,
    std::size_t count,
    std::chrono::milliseconds timeout
  ) {
    auto &socket = which == media_lane_e::video ? impl_->video_receiver : impl_->audio_receiver;
    std::vector<std::vector<std::uint8_t>> datagrams;
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::array<std::uint8_t, 65536> buffer;
    while (datagrams.size() < count) {
      udp::endpoint peer;
      sys::error_code error;
      const auto bytes = socket.receive_from(asio::buffer(buffer), peer, 0, error);
      if (!error) {
        datagrams.emplace_back(buffer.begin(), buffer.begin() + bytes);
      } else if (error == asio::error::would_block || error == asio::error::try_again) {
        if (std::chrono::steady_clock::now() >= deadline) {
          break;
        }
        std::this_thread::sleep_for(1ms);
      } else {
        throw sys::system_error {error};
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        break;
      }
    }
    return datagrams;
  }

  void media_sender_fixture_t::stop() {
    impl_->stop();
  }

  void media_sender_fixture_t::finish_senders_without_shutdown() {
    impl_->finish(false);
  }

  void media_sender_fixture_t::close() {
    impl_->close();
  }

  media_sender_state_t media_sender_fixture_t::state_after_stop() const {
    if (!impl_->stopped) {
      throw std::logic_error {"sender state is only available after stop"};
    }
    return impl_->final_state;
  }

  std::weak_ptr<void> media_sender_fixture_t::weak_channel(media_lane_e which) const {
    if (which == media_lane_e::video) {
      return impl_->video_channel;
    }
    return impl_->audio_channel;
  }

  std::weak_ptr<void> media_sender_fixture_t::weak_frame_pool() const {
    return impl_->frame_pool;
  }

  std::weak_ptr<void> media_sender_fixture_t::weak_queue(media_lane_e which) const {
    if (which == media_lane_e::video) {
      return impl_->media_senders.video_packets();
    }
    return impl_->media_senders.audio_packets();
  }
}  // namespace testing
