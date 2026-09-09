/**
 * @file src/microphone.cpp
 * @brief Bounded legacy VoidLink Opus microphone receiver.
 */
#include "microphone.h"

#include <algorithm>
#include <atomic>
#include <boost/asio.hpp>
#include <chrono>
#include <condition_variable>
#include <future>
#include <map>
#include <mutex>
#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <optional>
#include <opus/opus.h>
#include <thread>
#include <utility>

namespace microphone {
  namespace {
    using clock_t = std::chrono::steady_clock;
    using udp = boost::asio::ip::udp;
    constexpr std::size_t max_queued_packets = 4;
    constexpr std::size_t max_pcm_samples = frame_samples * 4;

    struct key_cleanup_t {
      std::array<std::uint8_t, 16> &key;

      ~key_cleanup_t() {
        OPENSSL_cleanse(key.data(), key.size());
      }
    };

    std::uint16_t read_le16(std::span<const std::uint8_t> data, std::size_t offset) {
      return data[offset] | (static_cast<std::uint16_t>(data[offset + 1]) << 8);
    }

    std::uint32_t read_le32(std::span<const std::uint8_t> data, std::size_t offset) {
      return read_le16(data, offset) | (static_cast<std::uint32_t>(read_le16(data, offset + 2)) << 16);
    }

    std::string normalized_address(boost::asio::ip::address address) {
      if (address.is_v6() && address.to_v6().is_v4_mapped()) {
        return boost::asio::ip::make_address_v4(boost::asio::ip::v4_mapped, address.to_v6()).to_string();
      }
      return address.to_string();
    }
  }  // namespace

  struct packet_decoder_t::impl_t {
    options_t options;
    OpusDecoder *decoder {};
    EVP_CIPHER_CTX *cipher {};
    std::optional<std::uint16_t> peer_port;
    std::optional<std::uint16_t> last_sequence;
    std::int64_t highest_sequence {};
    std::int64_t next_sequence {};
    unsigned startup_slots {};
    unsigned missing_slots {};
    std::map<std::int64_t, std::vector<std::uint8_t>> packets;

    explicit impl_t(const options_t &value):
        options(value) {
      int error = OPUS_OK;
      decoder = opus_decoder_create(sample_rate, 1, &error);
      if (error != OPUS_OK) {
        decoder = nullptr;
      }
      if (options.encrypted) {
        cipher = EVP_CIPHER_CTX_new();
      }
    }

    ~impl_t() {
      opus_decoder_destroy(decoder);
      EVP_CIPHER_CTX_free(cipher);
      OPENSSL_cleanse(options.aes_key.data(), options.aes_key.size());
    }

    bool decrypt(std::span<const std::uint8_t> input, std::uint16_t sequence, std::vector<std::uint8_t> &output) {
      if (!options.encrypted) {
        output.assign(input.begin(), input.end());
        return true;
      }
      // Shared common-C deliberately preserves the reference's manual rounding
      // followed by EVP PKCS7 finalization. Remove the final PKCS7 layer only:
      // remaining bytes are passed to Opus exactly as the legacy host does.
      if (input.empty() || input.size() % 16 != 0) {
        return false;
      }
      std::array<std::uint8_t, 16> iv {};
      const auto value = options.key_id + sequence;
      for (unsigned index = 0; index < 4; ++index) {
        iv[index] = static_cast<std::uint8_t>(value >> (24 - 8 * index));
      }
      output.resize(input.size());
      int count = 0;
      int tail = 0;
      if (EVP_DecryptInit_ex(cipher, EVP_aes_128_cbc(), nullptr, options.aes_key.data(), iv.data()) != 1 || EVP_DecryptUpdate(cipher, output.data(), &count, input.data(), static_cast<int>(input.size())) != 1 || EVP_DecryptFinal_ex(cipher, output.data() + count, &tail) != 1) {
        output.clear();
        return false;
      }
      output.resize(count + tail);
      return true;
    }
  };

  packet_decoder_t::packet_decoder_t(const options_t &options):
      impl_(std::make_unique<impl_t>(options)) {}

  packet_decoder_t::~packet_decoder_t() = default;

  bool packet_decoder_t::valid() const {
    return impl_->decoder && (!impl_->options.encrypted || impl_->cipher) && impl_->options.ping_payload[0] != 0;
  }

  bool packet_decoder_t::receive(std::span<const std::uint8_t> packet, const std::string &remote_address, std::uint16_t remote_port) {
    auto &state = *impl_;
    if (!valid() || remote_address != state.options.remote_address || remote_port == 0) {
      return false;
    }
    if (packet.size() == 20 && CRYPTO_memcmp(packet.data(), state.options.ping_payload.data(), 16) == 0) {
      // A session nonce binds the source port. Never permit another endpoint to
      // steal an active stream, even if it is behind the same NAT address.
      if (!state.peer_port) {
        state.peer_port = remote_port;
      }
      return *state.peer_port == remote_port;
    }
    if (!state.peer_port || *state.peer_port != remote_port || packet.size() <= 12 || packet.size() > max_datagram_size || packet[0] != 0 || packet[1] != 0x61 || read_le32(packet, 8) != 0x12345678) {
      return false;
    }
    const auto sequence = read_le16(packet, 2);
    const auto raw_distance = state.last_sequence ? static_cast<std::uint16_t>(sequence - *state.last_sequence) : 0;
    const auto distance = raw_distance < 0x8000 ? static_cast<int>(raw_distance) : static_cast<int>(raw_distance) - 0x10000;
    const auto extended_sequence = state.highest_sequence + distance;
    if (state.last_sequence && (extended_sequence < state.next_sequence || state.packets.contains(extended_sequence))) {
      return false;
    }
    std::vector<std::uint8_t> payload;
    if (!state.decrypt(packet.subspan(12), sequence, payload) || payload.empty() || opus_packet_get_nb_samples(payload.data(), static_cast<opus_int32>(payload.size()), sample_rate) != frame_samples) {
      return false;
    }
    const unsigned char *frames[48];
    opus_int16 sizes[48];
    unsigned char toc;
    int payload_offset;
    if (opus_packet_parse(payload.data(), static_cast<opus_int32>(payload.size()), &toc, frames, sizes, &payload_offset) < 0) {
      return false;
    }
    if (!state.last_sequence) {
      state.next_sequence = extended_sequence;
      state.startup_slots = 2;
    } else if (extended_sequence > state.next_sequence + 8) {
      // A long loss or pause must not queue seconds of stale microphone audio.
      state.packets.clear();
      state.next_sequence = extended_sequence;
      state.startup_slots = 2;
      (void) opus_decoder_ctl(state.decoder, OPUS_RESET_STATE);
    }
    if (!state.last_sequence || distance > 0) {
      state.last_sequence = sequence;
      state.highest_sequence = extended_sequence;
    }
    state.packets.emplace(extended_sequence, std::move(payload));
    if (state.packets.size() > max_queued_packets) {
      const auto furthest = std::prev(state.packets.end());
      const bool retained = furthest->first != extended_sequence;
      state.packets.erase(furthest);
      return retained;
    }
    return true;
  }

  std::vector<std::int16_t> packet_decoder_t::playout() {
    auto &state = *impl_;
    if (!valid() || !state.last_sequence) {
      return {};
    }
    if (state.startup_slots > 0) {
      --state.startup_slots;
      return {};
    }
    auto packet = state.packets.find(state.next_sequence);
    std::vector<std::int16_t> pcm(frame_samples);
    int result;
    if (packet == state.packets.end()) {
      if (++state.missing_slots > 2) {
        // After a longer gap, advance only toward a received future packet.
        // When capture pauses its sequence does not advance with our clock.
        if (!state.packets.empty()) {
          ++state.next_sequence;
        }
        return {};
      }
      if (!state.packets.empty()) {
        ++state.next_sequence;
      }
      result = opus_decode(state.decoder, nullptr, 0, pcm.data(), frame_samples, 0);
    } else {
      ++state.next_sequence;
      state.missing_slots = 0;
      result = opus_decode(state.decoder, packet->second.data(), static_cast<opus_int32>(packet->second.size()), pcm.data(), frame_samples, 0);
      state.packets.erase(packet);
    }
    if (result != frame_samples) {
      return {};
    }
    return pcm;
  }

  std::size_t packet_decoder_t::queued_packets() const {
    return impl_->packets.size();
  }

  struct session_t::impl_t {
    std::atomic<bool> stopping {false};
    std::atomic<bool> active {false};
    std::atomic<bool> activated {false};
    std::uint16_t port {};
    std::thread worker;
    std::mutex mutex;
    std::mutex join_mutex;
    std::condition_variable wake;

    void run(options_t options, const sink_factory_t &factory, std::promise<std::string> ready) {
      const key_cleanup_t key_cleanup {options.aes_key};
      bool reported = false;
      try {
        boost::system::error_code ec;
        const auto bind = boost::asio::ip::make_address(options.bind_address, ec);
        if (ec) {
          ready.set_value("Invalid microphone bind address");
          return;
        }
        const auto remote = boost::asio::ip::make_address(options.remote_address, ec);
        if (ec) {
          ready.set_value("Invalid microphone client address");
          return;
        }
        const bool remote_is_v4 = remote.is_v4() || remote.to_v6().is_v4_mapped();
        if ((bind.is_v4() && !remote_is_v4) || (bind.is_v6() && bind.to_v6().is_v4_mapped() && !remote_is_v4) || (bind.is_v6() && remote_is_v4 && !bind.is_unspecified() && !bind.to_v6().is_v4_mapped())) {
          ready.set_value("Microphone bind address cannot receive the client's address family");
          return;
        }
        options.remote_address = normalized_address(remote);
        packet_decoder_t decoder(options);
        if (!decoder.valid()) {
          ready.set_value("Microphone decoder or session ping is unavailable");
          return;
        }
        boost::asio::io_context io;
        udp::socket socket(io);
        socket.open(bind.is_v4() ? udp::v4() : udp::v6());
        if (bind.is_v6()) {
          // Windows defaults an IPv6 UDP socket to IPV6_V6ONLY. The host's
          // unrestricted BOTH-family bind is ::, which must also accept the
          // IPv4/mapped IPv4 peer connected to the dual-stack RTSP listener.
          socket.set_option(boost::asio::ip::v6_only(false));
        }
        socket.bind(udp::endpoint(bind, options.port));
        socket.non_blocking(true);
        socket.set_option(boost::asio::socket_base::receive_buffer_size(64 * 1024));
        auto sink = factory();
        std::string error;
        if (!sink || !sink->open(options.sink_endpoint, error)) {
          ready.set_value(error.empty() ? "Microphone sink is unavailable" : error);
          return;
        }
        port = socket.local_endpoint().port();
        active.store(true);
        ready.set_value({});
        reported = true;
        std::array<std::uint8_t, max_datagram_size + 1> buffer;
        std::vector<std::int16_t> pending_pcm;
        pending_pcm.reserve(max_pcm_samples);
        auto next_playout = clock_t::now() + std::chrono::milliseconds(20);
        while (!stopping.load()) {
          // A flooded socket cannot starve playout, device loss, or teardown.
          for (unsigned count = 0; count < 32 && !stopping.load(); ++count) {
            udp::endpoint peer;
            const auto received = socket.receive_from(boost::asio::buffer(buffer), peer, 0, ec);
            if (ec == boost::asio::error::would_block || ec == boost::asio::error::try_again) {
              break;
            }
            if (ec) {
              if (ec == boost::asio::error::message_size || ec == boost::asio::error::connection_reset) {
                continue;
              }
              stopping.store(true);
              break;
            }
            if (activated.load() || (received == 20 && CRYPTO_memcmp(buffer.data(), options.ping_payload.data(), 16) == 0)) {
              decoder.receive(std::span(buffer).first(received), normalized_address(peer.address()), peer.port());
            }
          }
          const auto now = clock_t::now();
          if (activated.load() && now >= next_playout) {
            // Catch up by dropping skipped slots; never burst delayed speech.
            if (now - next_playout >= std::chrono::milliseconds(40)) {
              pending_pcm.clear();
            }
            unsigned skipped = 0;
            while (now - next_playout >= std::chrono::milliseconds(20) && skipped++ < 8) {
              (void) decoder.playout();
              next_playout += std::chrono::milliseconds(20);
            }
            auto pcm = decoder.playout();
            if (pending_pcm.size() + pcm.size() > max_pcm_samples) {
              pending_pcm.clear();
            }
            pending_pcm.insert(pending_pcm.end(), pcm.begin(), pcm.end());
            next_playout = now + std::chrono::milliseconds(20);
          }
          if (!pending_pcm.empty()) {
            const int written = sink->write(pending_pcm);
            if (written < 0 || static_cast<std::size_t>(written) > pending_pcm.size()) {
              break;
            }
            pending_pcm.erase(pending_pcm.begin(), pending_pcm.begin() + written);
          }
          std::unique_lock lock(mutex);
          wake.wait_for(lock, std::chrono::milliseconds(2), [&] {
            return stopping.load();
          });
        }
      } catch (const std::exception &e) {
        if (!reported) {
          ready.set_value(e.what());
        }
      }
      active.store(false);
    }
  };

  session_t::session_t():
      impl_(std::make_unique<impl_t>()) {}

  session_t::~session_t() {
    stop();
  }

  std::unique_ptr<session_t> session_t::start(const options_t &options, std::string &error, sink_factory_t factory) {
    auto session = std::unique_ptr<session_t>(new session_t());
    std::promise<std::string> ready;
    auto result = ready.get_future();
    try {
      session->impl_->worker = std::thread(&impl_t::run, session->impl_.get(), options, std::move(factory), std::move(ready));
      error = result.get();
      if (!error.empty()) {
        return nullptr;
      }
      return session;
    } catch (const std::exception &e) {
      error = e.what();
      return nullptr;
    }
  }

  void session_t::stop() {
    impl_->stopping.store(true);
    impl_->wake.notify_all();
    std::lock_guard lock(impl_->join_mutex);
    if (impl_->worker.joinable()) {
      impl_->worker.join();
    }
  }

  void session_t::activate() {
    impl_->activated.store(true);
  }

  bool session_t::running() const {
    return impl_->active.load();
  }

  std::uint16_t session_t::bound_port() const {
    return impl_->port;
  }

  bool sink_available(const std::string &endpoint, std::string &error) {
    if (endpoint.empty()) {
      error = "Configure a virtual microphone cable render endpoint first";
      return false;
    }
    // COM objects are opened/released on a dedicated thread, regardless of the
    // apartment (or lack of one) used by the RTSP/HTTP caller.
    auto result = std::async(std::launch::async, [&] {
      auto sink = make_wasapi_sink();
      return sink && sink->open(endpoint, error);
    });
    return result.get();
  }
}  // namespace microphone
