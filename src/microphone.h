/**
 * @file src/microphone.h
 * @brief Session-owned client microphone transport and Windows input routing.
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace microphone {
  constexpr unsigned sample_rate = 48000;
  constexpr unsigned frame_samples = 960;  // Legacy clients encode mono 20 ms Opus frames.
  constexpr unsigned max_datagram_size = 1400;

  struct options_t {
    std::string bind_address;
    std::string remote_address;
    std::string sink_endpoint;  // Exact render endpoint name or ID of a VB-Audio microphone cable.
    std::uint16_t port {};
    // Exactly the 16 bytes advertised in X-SS-Ping-Payload. The shared sender
    // uses byte zero as a presence sentinel; the host's ASCII hex nonce is safe.
    std::array<std::uint8_t, 16> ping_payload {};
    std::array<std::uint8_t, 16> aes_key {};
    std::uint32_t key_id {};
    bool encrypted {};
  };

  /** All methods, including destruction, run on the microphone worker thread. */
  class pcm_sink_t {
  public:
    virtual ~pcm_sink_t() = default;
    virtual bool open(const std::string &endpoint, std::string &error) = 0;
    /** Nonblocking mono S16/48 kHz write; return frames consumed, or -1 on device loss. */
    virtual int write(std::span<const std::int16_t> samples) = 0;
  };

  using sink_factory_t = std::function<std::unique_ptr<pcm_sink_t>()>;
  std::unique_ptr<pcm_sink_t> make_wasapi_sink();
  /** Never changes defaults, plays sound, downloads, or installs a device. */
  bool sink_available(const std::string &endpoint, std::string &error);

  /**
   * Bounded single-thread receiver/decoder. Public to exercise the real wire parser
   * independently of devices. Its owning session supplies a normalized peer address.
   */
  class packet_decoder_t {
  public:
    explicit packet_decoder_t(const options_t &options);
    ~packet_decoder_t();
    packet_decoder_t(const packet_decoder_t &) = delete;
    packet_decoder_t &operator=(const packet_decoder_t &) = delete;
    bool valid() const;
    bool receive(std::span<const std::uint8_t> packet, const std::string &remote_address, std::uint16_t remote_port);
    /** Advance one 20 ms playout slot. Empty means no microphone audio is available. */
    std::vector<std::int16_t> playout();
    std::size_t queued_packets() const;

  private:
    struct impl_t;
    std::unique_ptr<impl_t> impl_;
  };

  class session_t {
  public:
    /** Succeeds only after the UDP socket, decoder, and configured sink are ready. */
    static std::unique_ptr<session_t> start(const options_t &options, std::string &error, sink_factory_t sink_factory = make_wasapi_sink);
    ~session_t();
    session_t(const session_t &) = delete;
    session_t &operator=(const session_t &) = delete;
    /** Arm after ANNOUNCE has accepted the microphone encryption negotiation. */
    void activate();
    void stop();
    bool running() const;
    std::uint16_t bound_port() const;

  private:
    session_t();
    struct impl_t;
    std::unique_ptr<impl_t> impl_;
  };
}  // namespace microphone
