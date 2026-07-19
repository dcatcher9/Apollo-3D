/**
 * @file tests/unit/test_input.cpp
 * @brief Tests for input packet validation.
 */

#include <array>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

extern "C" {
#include <moonlight-common-c/src/Input.h>
}

#include <src/input.h>
#include <src/utility.h>

#include "../tests_common.h"

namespace {
  std::vector<std::uint8_t> make_input_packet(std::uint32_t magic, std::size_t packet_size) {
    std::vector<std::uint8_t> packet(packet_size);
    const auto declared_size = util::endian::big<std::uint32_t>(packet_size - sizeof(std::uint32_t));
    const auto wire_magic = util::endian::little(magic);
    std::memcpy(packet.data(), &declared_size, sizeof(declared_size));
    std::memcpy(packet.data() + sizeof(declared_size), &wire_magic, sizeof(wire_magic));
    return packet;
  }
}  // namespace

TEST(InputPacketValidationTests, AcceptsEveryHandledFixedPacketAtItsExactSize) {
  const std::array packets {
    std::pair {ENABLE_HAPTICS_MAGIC, sizeof(NV_HAPTICS_PACKET)},
    std::pair {KEY_DOWN_EVENT_MAGIC, sizeof(NV_KEYBOARD_PACKET)},
    std::pair {KEY_UP_EVENT_MAGIC, sizeof(NV_KEYBOARD_PACKET)},
    std::pair {MOUSE_MOVE_REL_MAGIC_GEN5, sizeof(NV_REL_MOUSE_MOVE_PACKET)},
    std::pair {MOUSE_MOVE_ABS_MAGIC, sizeof(NV_ABS_MOUSE_MOVE_PACKET)},
    std::pair {MOUSE_BUTTON_DOWN_EVENT_MAGIC_GEN5, sizeof(NV_MOUSE_BUTTON_PACKET)},
    std::pair {MOUSE_BUTTON_UP_EVENT_MAGIC_GEN5, sizeof(NV_MOUSE_BUTTON_PACKET)},
    std::pair {MULTI_CONTROLLER_MAGIC_GEN5, sizeof(NV_MULTI_CONTROLLER_PACKET)},
    std::pair {SCROLL_MAGIC_GEN5, sizeof(NV_SCROLL_PACKET)},
    std::pair {SS_HSCROLL_MAGIC, sizeof(SS_HSCROLL_PACKET)},
    std::pair {SS_TOUCH_MAGIC, sizeof(SS_TOUCH_PACKET)},
    std::pair {SS_PEN_MAGIC, sizeof(SS_PEN_PACKET)},
    std::pair {SS_CONTROLLER_ARRIVAL_MAGIC, sizeof(SS_CONTROLLER_ARRIVAL_PACKET)},
    std::pair {SS_CONTROLLER_TOUCH_MAGIC, sizeof(SS_CONTROLLER_TOUCH_PACKET)},
    std::pair {SS_CONTROLLER_MOTION_MAGIC, sizeof(SS_CONTROLLER_MOTION_PACKET)},
    std::pair {SS_CONTROLLER_BATTERY_MAGIC, sizeof(SS_CONTROLLER_BATTERY_PACKET)},
  };

  for (const auto &[magic, packet_size] : packets) {
    SCOPED_TRACE(testing::Message() << "magic=" << magic << " size=" << packet_size);
    EXPECT_EQ(input::validated_packet_magic(make_input_packet(magic, packet_size)), magic);
    EXPECT_FALSE(input::validated_packet_magic(make_input_packet(magic, packet_size - 1)));
    EXPECT_FALSE(input::validated_packet_magic(make_input_packet(magic, packet_size + 1)));
  }
}

TEST(InputPacketValidationTests, AcceptsOnlyBoundedVariableUnicodePackets) {
  for (const auto packet_size : {sizeof(NV_INPUT_HEADER) + 1, sizeof(NV_INPUT_HEADER) + 4, sizeof(NV_UNICODE_PACKET)}) {
    EXPECT_EQ(input::validated_packet_magic(make_input_packet(UTF8_TEXT_EVENT_MAGIC, packet_size)), UTF8_TEXT_EVENT_MAGIC);
  }

  EXPECT_FALSE(input::validated_packet_magic(make_input_packet(
    UTF8_TEXT_EVENT_MAGIC,
    sizeof(NV_INPUT_HEADER)
  )));
  EXPECT_FALSE(input::validated_packet_magic(make_input_packet(
    UTF8_TEXT_EVENT_MAGIC,
    sizeof(NV_UNICODE_PACKET) + 1
  )));
}

TEST(InputPacketValidationTests, RejectsRuntOversizedUnknownAndMismatchedPackets) {
  for (std::size_t packet_size = 0; packet_size < sizeof(NV_INPUT_HEADER); ++packet_size) {
    std::vector<std::uint8_t> packet(packet_size);
    EXPECT_FALSE(input::validated_packet_magic(packet));
  }

  EXPECT_FALSE(input::validated_packet_magic(make_input_packet(0xDEADBEEF, sizeof(NV_INPUT_HEADER))));
  EXPECT_FALSE(input::validated_packet_magic(make_input_packet(
    UTF8_TEXT_EVENT_MAGIC,
    input::INPUT_PACKET_SIZE_MAX + 1
  )));

  auto mismatched = make_input_packet(KEY_DOWN_EVENT_MAGIC, sizeof(NV_KEYBOARD_PACKET));
  const auto wrong_size = util::endian::big<std::uint32_t>(1);
  std::memcpy(mismatched.data(), &wrong_size, sizeof(wrong_size));
  EXPECT_FALSE(input::validated_packet_magic(mismatched));
}
