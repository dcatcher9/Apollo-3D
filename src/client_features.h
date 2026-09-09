/** Shared microphone and authored DualSense wire contracts. */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace client_features {
  constexpr std::uint32_t encryption_microphone = 0x08;
  constexpr std::uint32_t client_authored_pcm = 0x20;
  constexpr std::uint32_t client_authored_ir_v2 = 0x40;
  constexpr std::uint16_t pcm_control_type = 0x550a;
  constexpr std::size_t pcm_header_size = 28;
  constexpr std::size_t pcm_max_frames = 240;
  constexpr std::size_t pcm_max_control_size = 4 + pcm_header_size + pcm_max_frames * 4;

  struct pcm_frame_t {
    std::uint16_t controller;
    std::uint8_t flags;
    std::uint16_t frame_count;
    std::uint32_t sequence;
    std::uint64_t presentation_time_us;
    std::span<const std::uint8_t> samples;
  };

  struct pcm_control_packet_t {
    std::array<std::uint8_t, pcm_max_control_size> bytes {};
    std::size_t size {};
  };

  // Includes the four-byte encrypted-control plaintext header. The caller sends
  // exactly size bytes, including no unused PCM capacity or C struct padding.
  [[nodiscard]] inline std::optional<pcm_control_packet_t> encode_pcm(const pcm_frame_t &frame) {
    if (frame.controller >= 16 || (frame.flags & ~0x07u) || frame.frame_count > pcm_max_frames || frame.samples.size() != frame.frame_count * 4u) {
      return std::nullopt;
    }
    pcm_control_packet_t packet;
    auto put = [&](std::size_t offset, std::uint64_t value, std::size_t size) {
      for (std::size_t i = 0; i < size; ++i) {
        packet.bytes[offset + i] = static_cast<std::uint8_t>(value >> (i * 8));
      }
    };
    packet.size = 4 + pcm_header_size + frame.samples.size();
    put(0, pcm_control_type, 2);
    put(2, packet.size - 4, 2);
    put(4, 1, 1);
    put(5, frame.flags, 1);
    put(6, pcm_header_size, 2);
    put(8, frame.controller, 2);
    put(10, frame.frame_count, 2);
    put(12, frame.sequence, 4);
    put(16, frame.presentation_time_us, 8);
    put(24, 48000, 4);
    put(28, 2, 1);
    put(29, 16, 1);
    for (std::size_t i = 0; i < frame.samples.size(); ++i) {
      packet.bytes[4 + pcm_header_size + i] = frame.samples[i];
    }
    return packet;
  }
}  // namespace client_features
