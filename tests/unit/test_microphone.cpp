/** @file tests/unit/test_microphone.cpp */
#include "src/microphone.h"

#include <algorithm>
#include <atomic>
#include <boost/asio.hpp>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <gtest/gtest.h>
#include <mutex>
#include <openssl/evp.h>
#include <opus/opus.h>
#include <thread>

namespace {
  microphone::options_t options(bool encrypted = false) {
    microphone::options_t value;
    value.bind_address = "127.0.0.1";
    value.remote_address = "127.0.0.1";
    value.sink_endpoint = "injected-test-cable";
    value.key_id = 0x01020304;
    value.encrypted = encrypted;
    for (unsigned index = 0; index < 16; ++index) {
      value.ping_payload[index] = static_cast<std::uint8_t>('A' + index);
      value.aes_key[index] = static_cast<std::uint8_t>(index);
    }
    return value;
  }

  std::vector<std::uint8_t> ping(const microphone::options_t &value) {
    std::vector<std::uint8_t> result(value.ping_payload.begin(), value.ping_payload.end());
    result.insert(result.end(), {0, 0, 0, 1});
    return result;
  }

  std::vector<std::uint8_t> opus_packet(int frame_samples = microphone::frame_samples) {
    int error;
    OpusEncoder *encoder = opus_encoder_create(microphone::sample_rate, 1, OPUS_APPLICATION_VOIP, &error);
    if (!encoder || error != OPUS_OK) {
      return {};
    }
    std::vector<std::int16_t> pcm(frame_samples);
    for (int index = 0; index < frame_samples; ++index) {
      pcm[index] = static_cast<std::int16_t>(6000 * std::sin(index * 0.09));
    }
    std::vector<std::uint8_t> result(1000);
    const int size = opus_encode(encoder, pcm.data(), frame_samples, result.data(), static_cast<opus_int32>(result.size()));
    opus_encoder_destroy(encoder);
    if (size <= 0) {
      return {};
    }
    result.resize(size);
    return result;
  }

  std::vector<std::uint8_t> packet(const microphone::options_t &value, std::uint16_t sequence, std::vector<std::uint8_t> payload) {
    std::vector<std::uint8_t> result {0, 0x61, static_cast<std::uint8_t>(sequence), static_cast<std::uint8_t>(sequence >> 8), 0x12, 0x34, 0x56, 0x78, 0x78, 0x56, 0x34, 0x12};
    if (value.encrypted) {
      const auto rounded = (payload.size() + 15) / 16 * 16;
      if (rounded != payload.size()) {
        const auto padding = static_cast<std::uint8_t>(rounded - payload.size());
        payload.resize(rounded, padding);
      }
      std::array<std::uint8_t, 16> iv {};
      const auto iv_sequence = value.key_id + sequence;
      for (unsigned index = 0; index < 4; ++index) {
        iv[index] = static_cast<std::uint8_t>(iv_sequence >> (24 - index * 8));
      }
      std::vector<std::uint8_t> encrypted(payload.size() + 16);
      auto *cipher = EVP_CIPHER_CTX_new();
      int first = 0;
      int final = 0;
      if (!cipher || EVP_EncryptInit_ex(cipher, EVP_aes_128_cbc(), nullptr, value.aes_key.data(), iv.data()) != 1 || EVP_EncryptUpdate(cipher, encrypted.data(), &first, payload.data(), static_cast<int>(payload.size())) != 1 || EVP_EncryptFinal_ex(cipher, encrypted.data() + first, &final) != 1) {
        EVP_CIPHER_CTX_free(cipher);
        return {};
      }
      EVP_CIPHER_CTX_free(cipher);
      encrypted.resize(first + final);
      result.insert(result.end(), encrypted.begin(), encrypted.end());
    } else {
      result.insert(result.end(), payload.begin(), payload.end());
    }
    return result;
  }

  TEST(Microphone, RequiresSessionNonceAddressAndFixedSourcePort) {
    const auto value = options();
    microphone::packet_decoder_t decoder(value);
    const auto audio = packet(value, 0, opus_packet());
    EXPECT_FALSE(decoder.receive(audio, value.remote_address, 1234));
    EXPECT_FALSE(decoder.receive(ping(value), "127.0.0.2", 1234));
    auto old_ping = ping(value);
    old_ping[0] ^= 1;
    EXPECT_FALSE(decoder.receive(old_ping, value.remote_address, 1234));
    EXPECT_FALSE(decoder.receive(std::vector<std::uint8_t> {'P', 'I', 'N', 'G'}, value.remote_address, 1234));
    ASSERT_TRUE(decoder.receive(ping(value), value.remote_address, 1234));
    EXPECT_FALSE(decoder.receive(ping(value), value.remote_address, 4321));
    EXPECT_FALSE(decoder.receive(audio, value.remote_address, 4321));
    EXPECT_TRUE(decoder.receive(audio, value.remote_address, 1234));
  }

  TEST(Microphone, RejectsAbsentNonceSentinelBeforeOpeningSink) {
    auto value = options();
    value.ping_payload[0] = 0;
    microphone::packet_decoder_t decoder(value);
    EXPECT_FALSE(decoder.valid());
    EXPECT_FALSE(decoder.receive(ping(value), value.remote_address, 1234));
    unsigned sink_creations = 0;
    std::string error;
    EXPECT_FALSE(microphone::session_t::start(value, error, [&]() -> std::unique_ptr<microphone::pcm_sink_t> {
      ++sink_creations;
      return nullptr;
    }));
    EXPECT_EQ(sink_creations, 0u);
    EXPECT_FALSE(error.empty());
  }

  TEST(Microphone, DecodesActualOpusAndRetainsLegacyEncryptedPaddingBehavior) {
    for (const bool encrypted : {false, true}) {
      const auto value = options(encrypted);
      microphone::packet_decoder_t decoder(value);
      ASSERT_TRUE(decoder.receive(ping(value), value.remote_address, 1234));
      auto encoded = opus_packet();
      ASSERT_FALSE(encoded.empty());
      ASSERT_TRUE(decoder.receive(packet(value, 0, encoded), value.remote_address, 1234));
      EXPECT_TRUE(decoder.playout().empty());
      EXPECT_TRUE(decoder.playout().empty());
      auto pcm = decoder.playout();
      ASSERT_EQ(pcm.size(), microphone::frame_samples);
      if (encrypted && encoded.size() % 16 != 0) {
        encoded.resize((encoded.size() + 15) / 16 * 16, static_cast<std::uint8_t>(16 - encoded.size() % 16));
      }
      int error;
      auto *reference_decoder = opus_decoder_create(microphone::sample_rate, 1, &error);
      ASSERT_EQ(error, OPUS_OK);
      std::vector<std::int16_t> reference(microphone::frame_samples);
      ASSERT_EQ(opus_decode(reference_decoder, encoded.data(), static_cast<opus_int32>(encoded.size()), reference.data(), microphone::frame_samples, 0), static_cast<int>(microphone::frame_samples));
      opus_decoder_destroy(reference_decoder);
      EXPECT_EQ(pcm, reference);
      EXPECT_TRUE(std::any_of(pcm.begin(), pcm.end(), [](auto sample) {
        return sample != 0;
      }));
    }
  }

  TEST(Microphone, RejectsMalformedHeaderCiphertextAndWrongOpusDuration) {
    const auto value = options(true);
    microphone::packet_decoder_t decoder(value);
    ASSERT_TRUE(decoder.receive(ping(value), value.remote_address, 1234));
    const auto audio = packet(value, 0, opus_packet());
    for (std::size_t offset : {0, 1, 8, 9, 10, 11}) {
      auto malformed = audio;
      malformed[offset] ^= 0xff;
      EXPECT_FALSE(decoder.receive(malformed, value.remote_address, 1234));
    }
    auto truncated = audio;
    truncated.pop_back();
    EXPECT_FALSE(decoder.receive(truncated, value.remote_address, 1234));
    auto corrupt = audio;
    corrupt.back() ^= 0x7f;
    EXPECT_FALSE(decoder.receive(corrupt, value.remote_address, 1234));
    auto oversized = audio;
    oversized.resize(microphone::max_datagram_size + 1);
    EXPECT_FALSE(decoder.receive(oversized, value.remote_address, 1234));
    EXPECT_FALSE(decoder.receive(packet(value, 0, opus_packet(480)), value.remote_address, 1234));
    EXPECT_EQ(decoder.queued_packets(), 0u);
    EXPECT_TRUE(decoder.receive(audio, value.remote_address, 1234));
  }

  TEST(Microphone, OrdersPacketsAcrossSequenceWrapAndDropsReplays) {
    const auto value = options();
    microphone::packet_decoder_t decoder(value);
    const auto encoded = opus_packet();
    ASSERT_TRUE(decoder.receive(ping(value), value.remote_address, 1234));
    ASSERT_TRUE(decoder.receive(packet(value, 65534, encoded), value.remote_address, 1234));
    ASSERT_TRUE(decoder.receive(packet(value, 0, encoded), value.remote_address, 1234));
    ASSERT_TRUE(decoder.receive(packet(value, 65535, encoded), value.remote_address, 1234));
    EXPECT_FALSE(decoder.receive(packet(value, 65535, encoded), value.remote_address, 1234));
    decoder.playout();
    decoder.playout();
    for (unsigned index = 0; index < 3; ++index) {
      EXPECT_EQ(decoder.playout().size(), microphone::frame_samples);
    }
    EXPECT_EQ(decoder.queued_packets(), 0u);
    EXPECT_FALSE(decoder.receive(packet(value, 65534, encoded), value.remote_address, 1234));
    EXPECT_TRUE(decoder.receive(packet(value, 1, encoded), value.remote_address, 1234));
    EXPECT_EQ(decoder.playout().size(), microphone::frame_samples);
  }

  TEST(Microphone, BoundsJitterBufferAndResumesAfterCapturePause) {
    const auto value = options();
    microphone::packet_decoder_t decoder(value);
    const auto encoded = opus_packet();
    ASSERT_TRUE(decoder.receive(ping(value), value.remote_address, 1234));
    for (std::uint16_t index = 0; index < 200; ++index) {
      decoder.receive(packet(value, index, encoded), value.remote_address, 1234);
      EXPECT_LE(decoder.queued_packets(), 4u);
    }
    microphone::packet_decoder_t resumed(value);
    ASSERT_TRUE(resumed.receive(ping(value), value.remote_address, 1234));
    ASSERT_TRUE(resumed.receive(packet(value, 0, encoded), value.remote_address, 1234));
    resumed.playout();
    resumed.playout();
    ASSERT_EQ(resumed.playout().size(), microphone::frame_samples);
    for (unsigned index = 0; index < 100; ++index) {
      resumed.playout();
    }
    EXPECT_TRUE(resumed.receive(packet(value, 1, encoded), value.remote_address, 1234));
    EXPECT_EQ(resumed.playout().size(), microphone::frame_samples);
  }

  struct sink_state_t {
    std::mutex mutex;
    std::condition_variable ready;
    std::vector<std::int16_t> pcm;
    std::atomic<unsigned> destroyed {};
    bool fail_open {};
    bool fail_write {};
  };

  class test_sink_t: public microphone::pcm_sink_t {
  public:
    explicit test_sink_t(std::shared_ptr<sink_state_t> state):
        state_(std::move(state)) {}

    ~test_sink_t() override {
      ++state_->destroyed;
    }

    bool open(const std::string &, std::string &error) override {
      if (state_->fail_open) {
        error = "injected unavailable endpoint";
        return false;
      }
      return true;
    }

    int write(std::span<const std::int16_t> pcm) override {
      std::lock_guard lock(state_->mutex);
      if (state_->fail_write) {
        return -1;
      }
      const auto count = std::min<std::size_t>(pcm.size(), 240);
      state_->pcm.insert(state_->pcm.end(), pcm.begin(), pcm.begin() + count);
      state_->ready.notify_all();
      return static_cast<int>(count);
    }

  private:
    std::shared_ptr<sink_state_t> state_;
  };

  TEST(Microphone, RealUdpRoutesDecodedAudioOnlyAfterActivationAndJoinsOnStop) {
    auto value = options(true);
    auto state = std::make_shared<sink_state_t>();
    std::string error;
    auto session = microphone::session_t::start(value, error, [state] {
      return std::make_unique<test_sink_t>(state);
    });
    ASSERT_TRUE(session) << error;
    ASSERT_TRUE(session->running());
    boost::asio::io_context io;
    boost::asio::ip::udp::socket sender(io, boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), 0));
    const boost::asio::ip::udp::endpoint receiver(boost::asio::ip::make_address("127.0.0.1"), session->bound_port());
    sender.send_to(boost::asio::buffer(ping(value)), receiver);
    sender.send_to(boost::asio::buffer(packet(value, 0, opus_packet())), receiver);
    {
      std::unique_lock lock(state->mutex);
      EXPECT_FALSE(state->ready.wait_for(lock, std::chrono::milliseconds(80), [&] {
        return !state->pcm.empty();
      }));
    }
    session->activate();
    sender.send_to(boost::asio::buffer(packet(value, 1, opus_packet())), receiver);
    {
      std::unique_lock lock(state->mutex);
      ASSERT_TRUE(state->ready.wait_for(lock, std::chrono::seconds(2), [&] {
        return state->pcm.size() >= microphone::frame_samples;
      }));
      EXPECT_TRUE(std::any_of(state->pcm.begin(), state->pcm.end(), [](auto sample) {
        return sample != 0;
      }));
    }
    std::thread stop_one([&] {
      session->stop();
    });
    std::thread stop_two([&] {
      session->stop();
    });
    stop_one.join();
    stop_two.join();
    EXPECT_FALSE(session->running());
    EXPECT_EQ(state->destroyed.load(), 1u);
  }

  TEST(Microphone, UnavailableSinkFailsSetupAndReleasesPortForNextSession) {
    auto state = std::make_shared<sink_state_t>();
    state->fail_open = true;
    std::string error;
    EXPECT_FALSE(microphone::session_t::start(options(), error, [state] {
      return std::make_unique<test_sink_t>(state);
    }));
    EXPECT_EQ(error, "injected unavailable endpoint");
    EXPECT_EQ(state->destroyed.load(), 1u);
    state->fail_open = false;
    for (unsigned count = 0; count < 8; ++count) {
      auto session = microphone::session_t::start(options(), error, [state] {
        return std::make_unique<test_sink_t>(state);
      });
      ASSERT_TRUE(session) << error;
      session->stop();
    }
    EXPECT_EQ(state->destroyed.load(), 9u);
  }

  TEST(Microphone, DualStackAndMappedIpv4RouteRealUdpToPcm) {
    struct address_case_t {
      std::string bind;
      std::string remote;
      std::string destination;
    };

    for (const auto &addresses : std::vector<address_case_t> {
           {"::", "127.0.0.1", "127.0.0.1"},
           {"::", "::ffff:127.0.0.1", "127.0.0.1"},
           {"::", "::1", "::1"},
           {"127.0.0.1", "::ffff:127.0.0.1", "127.0.0.1"}
         }) {
      SCOPED_TRACE(addresses.bind + " -> " + addresses.remote);
      auto value = options(true);
      value.bind_address = addresses.bind;
      value.remote_address = addresses.remote;
      auto state = std::make_shared<sink_state_t>();
      std::string error;
      auto session = microphone::session_t::start(value, error, [state] {
        return std::make_unique<test_sink_t>(state);
      });
      ASSERT_TRUE(session) << error;
      session->activate();
      boost::asio::io_context io;
      const auto destination = boost::asio::ip::make_address(addresses.destination);
      boost::asio::ip::udp::socket sender(io, boost::asio::ip::udp::endpoint(destination.is_v4() ? boost::asio::ip::udp::v4() : boost::asio::ip::udp::v6(), 0));
      const boost::asio::ip::udp::endpoint receiver(destination, session->bound_port());
      sender.send_to(boost::asio::buffer(ping(value)), receiver);
      sender.send_to(boost::asio::buffer(packet(value, 0, opus_packet())), receiver);
      {
        std::unique_lock lock(state->mutex);
        ASSERT_TRUE(state->ready.wait_for(lock, std::chrono::seconds(2), [&] {
          return state->pcm.size() >= microphone::frame_samples;
        }));
        EXPECT_TRUE(std::any_of(state->pcm.begin(), state->pcm.end(), [](auto sample) {
          return sample != 0;
        }));
      }
      session->stop();
    }
  }

  TEST(Microphone, RejectsIncompatibleBindFamilyBeforeOpeningSink) {
    unsigned sink_creations = 0;
    for (const auto &addresses : std::vector<std::pair<std::string, std::string>> {
           {"127.0.0.1", "::1"},
           {"::1", "127.0.0.1"},
           {"::ffff:127.0.0.1", "::1"}
         }) {
      auto value = options();
      value.bind_address = addresses.first;
      value.remote_address = addresses.second;
      std::string error;
      EXPECT_FALSE(microphone::session_t::start(value, error, [&]() -> std::unique_ptr<microphone::pcm_sink_t> {
        ++sink_creations;
        return nullptr;
      }));
      EXPECT_EQ(error, "Microphone bind address cannot receive the client's address family");
    }
    EXPECT_EQ(sink_creations, 0u);
  }

  TEST(Microphone, NoConfiguredCableNeverFallsBackToDefaultSpeakers) {
    std::string error;
    EXPECT_FALSE(microphone::sink_available("", error));
    EXPECT_FALSE(error.empty());
    EXPECT_FALSE(microphone::sink_available("nonexistent-microphone-endpoint", error));
    EXPECT_FALSE(error.empty());
  }
}  // namespace
