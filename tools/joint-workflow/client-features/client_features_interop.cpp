/** Real host encoder -> real shared common-C parser interoperability. */
#include "src/client_features.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
extern "C" {
#include "Ds5HapticsStream.h"
}

namespace {
  void require(bool condition, const char *message) {
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  struct capture_t {
    LI_DS5_HAPTICS_PCM_FRAME frame {};
    std::array<std::uint8_t, 960> pcm {};
    unsigned calls {};
  };

  void capture(const LI_DS5_HAPTICS_PCM_FRAME *frame, void *opaque) {
    auto &result = *static_cast<capture_t *>(opaque);
    result.frame = *frame;
    require(frame->pcmDataLength <= result.pcm.size(), "Parser PCM exceeds owned callback storage");
    std::copy_n(frame->pcmData, frame->pcmDataLength, result.pcm.begin());
    ++result.calls;
  }

  void verify_capabilities() {
    constexpr std::uint32_t host = LI_FF_HOST_SBS_TELEMETRY_V2 | LI_FF_ATOMIC_PRESENTATION_MODE_V2 |
                                   LI_FF_SOURCE_FRAME_ID_V1 | LI_FF_DS5_HAPTICS_PCM | LI_FF_DS5_HAPTICS_CAPABILITIES_V2;
    require(!usesLegacyDs5HapticsCapabilities(host), "Shared SBS host selected legacy haptics profile");
    require(supportsDs5HapticsPcm(host), "Shared host PCM capability was not recognized");
    require(!supportsDs5HapticsIrV2(host), "Unimplemented host IR capability was enabled");
    require(getDs5HapticsClientFeatureFlags(host, true, false) == client_features::client_authored_pcm, "Client and host authored PCM bits disagree");
    require(getDs5HapticsClientFeatureFlags(host, false, false) == 0, "Authored haptics enabled without a callback");
    require(getDs5HapticsClientFeatureFlags(host, false, true) == 0, "Client negotiated IR against a PCM-only host");
  }

  unsigned verify_packets() {
    std::array<std::uint8_t, 960> samples;
    for (std::size_t i = 0; i < samples.size(); ++i) {
      samples[i] = static_cast<std::uint8_t>(i ^ 0xa5);
    }
    unsigned cases = 0;
    for (std::uint16_t count : {0, 1, 2, 240}) {
      for (std::uint8_t flags = 0; flags <= 7; ++flags) {
        const client_features::pcm_frame_t input {
          15,
          flags,
          count,
          0x87654321,
          0xfedcba9876543210,
          std::span(samples).first(count * 4),
        };
        auto packet = client_features::encode_pcm(input);
        require(packet.has_value(), "Host rejected a valid PCM boundary packet");
        capture_t result;
        require(processDs5HapticsStreamPacket(packet->bytes.data() + 4, static_cast<int>(packet->size - 4), capture, &result), "Shared parser rejected the host PCM packet");
        require(result.calls == 1 && result.frame.controllerNumber == 15 && result.frame.flags == flags, "Controller or stream flags changed across the wire");
        require(result.frame.frameCount == count && result.frame.sequenceNumber == input.sequence, "Frame count or sequence changed across the wire");
        require(result.frame.presentationTimeUs == input.presentation_time_us, "64-bit presentation time changed across the wire");
        require(result.frame.sampleRate == 48000 && result.frame.channelCount == 2 && result.frame.bitsPerSample == 16, "PCM format changed across the wire");
        require(result.frame.pcmDataLength == count * 4, "PCM byte count changed across the wire");
        require(std::equal(samples.begin(), samples.begin() + count * 4, result.pcm.begin()), "Actuator samples changed across the wire");
        packet->bytes[30] = 1;
        require(!processDs5HapticsStreamPacket(packet->bytes.data() + 4, static_cast<int>(packet->size - 4), capture, &result), "Shared parser accepted nonzero reserved wire bits");
        require(result.calls == 1, "Malformed packet reached the client callback");
        // The callback copied borrowed bytes before the control packet was released.
        packet->bytes.fill(0);
        require(std::equal(samples.begin(), samples.begin() + count * 4, result.pcm.begin()), "Client callback retained borrowed packet storage");
        ++cases;
      }
    }
    return cases;
  }
}  // namespace

int main() {
  try {
    verify_capabilities();
    const auto cases = verify_packets();
    std::printf("PASS %u actual host/shared-C PCM cases and versioned capability negotiation\n", cases);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
