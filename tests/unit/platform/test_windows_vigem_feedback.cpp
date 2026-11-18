/**
 * @file tests/unit/platform/test_windows_vigem_feedback.cpp
 * @brief Exercise actual ViGEm feedback methods without creating a driver client or target.
 */

#include "src/config.h"
#include "src/platform/windows/input_feedback_test.h"

#include <chrono>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <string>

namespace {
  using namespace std::chrono_literals;
  using platf::detail::vigem_feedback_test_t;
  using platf::feedback_queue_t;
  using platf::gamepad_feedback_e;
  using platf::gamepad_feedback_msg_t;

  class WindowsVigemFeedbackTest: public testing::Test {
  protected:
    void SetUp() override {
      config::input.forward_rumble = true;
      feedback = std::make_unique<vigem_feedback_test_t>(queue, client_index);
    }

    ~WindowsVigemFeedbackTest() override {
      feedback.reset();
      config::input.forward_rumble = original_forward_rumble;
    }

    static void expect_rumble(const feedback_queue_t &source, std::uint16_t id, std::uint16_t low, std::uint16_t high) {
      const auto message = source->pop(0ms);
      ASSERT_TRUE(message);
      ASSERT_EQ(message->type, gamepad_feedback_e::rumble);
      EXPECT_EQ(message->id, id);
      EXPECT_EQ(message->data.rumble.lowfreq, low);
      EXPECT_EQ(message->data.rumble.highfreq, high);
    }

    static void expect_rgb(const feedback_queue_t &source, std::uint16_t id, std::uint8_t r, std::uint8_t g, std::uint8_t b) {
      const auto message = source->pop(0ms);
      ASSERT_TRUE(message);
      ASSERT_EQ(message->type, gamepad_feedback_e::set_rgb_led);
      EXPECT_EQ(message->id, id);
      EXPECT_EQ(message->data.rgb_led.r, r);
      EXPECT_EQ(message->data.rgb_led.g, g);
      EXPECT_EQ(message->data.rgb_led.b, b);
    }

    static constexpr std::uint8_t client_index = 7;
    const bool original_forward_rumble = config::input.forward_rumble;
    const safe::mail_t mail = std::make_shared<safe::mail_raw_t>();
    const feedback_queue_t queue = mail->queue<gamepad_feedback_msg_t>("vigem-feedback", 32);
    std::unique_ptr<vigem_feedback_test_t> feedback;
  };

  TEST_F(WindowsVigemFeedbackTest, SuppressesInitialZeroAndDuplicateRumbleButForwardsEachMotorChange) {
    const auto sizes = vigem_feedback_test_t::cache_sizes();
    EXPECT_EQ(sizes.rumble, sizeof(decltype(gamepad_feedback_msg_t {}.data.rumble)));
    EXPECT_EQ(sizes.rgb, sizeof(decltype(gamepad_feedback_msg_t {}.data.rgb_led)));
    EXPECT_EQ(sizes.message, sizeof(gamepad_feedback_msg_t));
    EXPECT_EQ(sizes.slot_count, platf::MAX_GAMEPADS);
    RecordProperty("rumble_cache_bytes", std::to_string(sizes.rumble));
    RecordProperty("rgb_cache_bytes", std::to_string(sizes.rgb));
    RecordProperty("gamepad_context_bytes", std::to_string(sizes.context));
    RecordProperty("feedback_message_bytes", std::to_string(sizes.message));
    RecordProperty("configured_slot_count", std::to_string(sizes.slot_count));

    feedback->rumble(0, 0);
    EXPECT_FALSE(queue->peek());

    feedback->rumble(0x12, 0xfe);
    expect_rumble(queue, client_index, 0x1200, 0xfe00);
    feedback->rumble(0x12, 0xfe);
    EXPECT_FALSE(queue->peek());

    feedback->rumble(0x13, 0xfe);
    expect_rumble(queue, client_index, 0x1300, 0xfe00);
    feedback->rumble(0x13, 0xff);
    expect_rumble(queue, client_index, 0x1300, 0xff00);
    feedback->rumble(0, 0);
    expect_rumble(queue, client_index, 0, 0);
    feedback->rumble(0, 0);
    EXPECT_FALSE(queue->peek());
  }

  TEST_F(WindowsVigemFeedbackTest, RgbChannelsAndInterleavedRumbleKeepIndependentDuplicateState) {
    feedback->set_rgb_led(0, 0, 0);
    EXPECT_FALSE(queue->peek());
    feedback->set_rgb_led(1, 2, 3);
    expect_rgb(queue, client_index, 1, 2, 3);
    feedback->rumble(4, 5);
    expect_rumble(queue, client_index, 0x0400, 0x0500);
    feedback->set_rgb_led(1, 2, 3);
    feedback->rumble(4, 5);
    EXPECT_FALSE(queue->peek());

    feedback->set_rgb_led(6, 2, 3);
    expect_rgb(queue, client_index, 6, 2, 3);
    feedback->set_rgb_led(6, 7, 3);
    expect_rgb(queue, client_index, 6, 7, 3);
    feedback->set_rgb_led(6, 7, 8);
    expect_rgb(queue, client_index, 6, 7, 8);
    feedback->rumble(4, 5);
    EXPECT_FALSE(queue->peek());
    feedback->set_rgb_led(0, 0, 0);
    expect_rgb(queue, client_index, 0, 0, 0);
    feedback->set_rgb_led(0, 0, 0);
    EXPECT_FALSE(queue->peek());
  }

  TEST_F(WindowsVigemFeedbackTest, DisabledRumbleDoesNotAdvanceItsCacheOrSuppressRgb) {
    feedback->rumble(1, 2);
    expect_rumble(queue, client_index, 0x0100, 0x0200);
    config::input.forward_rumble = false;
    feedback->rumble(3, 4);
    feedback->rumble(3, 4);
    EXPECT_FALSE(queue->peek());
    feedback->set_rgb_led(5, 6, 7);
    expect_rgb(queue, client_index, 5, 6, 7);

    config::input.forward_rumble = true;
    feedback->rumble(3, 4);
    expect_rumble(queue, client_index, 0x0300, 0x0400);
    feedback->rumble(3, 4);
    EXPECT_FALSE(queue->peek());
    config::input.forward_rumble = false;
    feedback->rumble(0, 0);
    config::input.forward_rumble = true;
    feedback->rumble(3, 4);
    EXPECT_FALSE(queue->peek());
    feedback->rumble(0, 0);
    expect_rumble(queue, client_index, 0, 0);
    EXPECT_FALSE(queue->peek());
  }

  TEST_F(WindowsVigemFeedbackTest, QueuedMessagesOwnTheirValuesAfterLaterUpdatesAndOwnerDestruction) {
    feedback->rumble(1, 2);
    feedback->set_rgb_led(3, 4, 5);
    feedback->rumble(6, 7);
    feedback->set_rgb_led(8, 9, 10);
    feedback.reset();

    expect_rumble(queue, client_index, 0x0100, 0x0200);
    expect_rgb(queue, client_index, 3, 4, 5);
    expect_rumble(queue, client_index, 0x0600, 0x0700);
    expect_rgb(queue, client_index, 8, 9, 10);
    EXPECT_FALSE(queue->peek());
  }

  TEST_F(WindowsVigemFeedbackTest, StoppedQueueStillAdvancesCachesAndDestinationChangesPreserveThem) {
    feedback->rumble(1, 2);
    feedback->set_rgb_led(3, 4, 5);
    expect_rumble(queue, client_index, 0x0100, 0x0200);
    expect_rgb(queue, client_index, 3, 4, 5);
    queue->stop();
    feedback->rumble(6, 7);
    feedback->set_rgb_led(8, 9, 10);

    const auto rebound = mail->queue<gamepad_feedback_msg_t>("rebound-feedback", 32);
    constexpr std::uint8_t rebound_index = 12;
    feedback->set_feedback_destination(rebound, rebound_index);
    feedback->rumble(6, 7);
    feedback->set_rgb_led(8, 9, 10);
    EXPECT_FALSE(rebound->peek());
    feedback->rumble(6, 11);
    feedback->set_rgb_led(8, 9, 12);
    expect_rumble(rebound, rebound_index, 0x0600, 0x0b00);
    expect_rgb(rebound, rebound_index, 8, 9, 12);
    EXPECT_FALSE(rebound->peek());
  }
}  // namespace
