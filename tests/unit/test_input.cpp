/**
 * @file tests/unit/test_input.cpp
 * @brief Tests for input packet validation.
 */

#include <array>
#include <cstdint>
#include <cstring>
#include <list>
#include <map>
#include <utility>
#include <vector>

extern "C" {
#include <moonlight-common-c/src/Input.h>
}

#include "../tests_common.h"

#include <src/input.h>
#include <src/platform/windows/input_cursor.h>
#include <src/platform/windows/misc.h>
#include <src/utility.h>

namespace {
  std::vector<std::uint8_t> make_input_packet(std::uint32_t magic, std::size_t packet_size) {
    std::vector<std::uint8_t> packet(packet_size);
    const auto declared_size = util::endian::big<std::uint32_t>(packet_size - sizeof(std::uint32_t));
    const auto wire_magic = util::endian::little(magic);
    std::memcpy(packet.data(), &declared_size, sizeof(declared_size));
    std::memcpy(packet.data() + sizeof(declared_size), &wire_magic, sizeof(wire_magic));
    return packet;
  }

  std::vector<std::uint8_t> make_relative_mouse_packet(std::int16_t x, std::int16_t y) {
    auto packet = make_input_packet(MOUSE_MOVE_REL_MAGIC_GEN5, sizeof(NV_REL_MOUSE_MOVE_PACKET));
    auto *move = reinterpret_cast<PNV_REL_MOUSE_MOVE_PACKET>(packet.data());
    move->deltaX = util::endian::big(x);
    move->deltaY = util::endian::big(y);
    return packet;
  }

  std::vector<std::uint8_t> make_vertical_scroll_packet(std::int16_t amount) {
    auto packet = make_input_packet(SCROLL_MAGIC_GEN5, sizeof(NV_SCROLL_PACKET));
    auto *scroll = reinterpret_cast<PNV_SCROLL_PACKET>(packet.data());
    scroll->scrollAmt1 = util::endian::big(amount);
    scroll->scrollAmt2 = util::endian::big(amount);
    return packet;
  }

  std::vector<std::uint8_t> make_horizontal_scroll_packet(std::int16_t amount) {
    auto packet = make_input_packet(SS_HSCROLL_MAGIC, sizeof(SS_HSCROLL_PACKET));
    auto *scroll = reinterpret_cast<PSS_HSCROLL_PACKET>(packet.data());
    scroll->scrollAmount = util::endian::big(amount);
    return packet;
  }

  std::vector<std::uint8_t> make_mouse_button_packet(std::uint32_t magic, std::uint8_t button) {
    auto packet = make_input_packet(magic, sizeof(NV_MOUSE_BUTTON_PACKET));
    reinterpret_cast<PNV_MOUSE_BUTTON_PACKET>(packet.data())->button = button;
    return packet;
  }

  std::pair<std::int16_t, std::int16_t> relative_mouse_delta(const std::vector<std::uint8_t> &packet) {
    const auto *move = reinterpret_cast<const NV_REL_MOUSE_MOVE_PACKET *>(packet.data());
    return {util::endian::big(move->deltaX), util::endian::big(move->deltaY)};
  }
}  // namespace

TEST(InputCursorConfinementTests, RelativeMovementStopsAtEveryEdgeWithNegativeOrigin) {
  platf::detail::confined_cursor_state_t cursor;
  const platf::detail::cursor_bounds_t bounds {-1920, -1080, 1920, 1080};
  EXPECT_EQ(cursor.move(bounds, std::pair {-100, -100}, 32767, 32767), (std::pair {-1, -1}));
  cursor.observe_injected_cursor({-1, -1});
  EXPECT_EQ(cursor.move(bounds, std::nullopt, -32768, -32768), (std::pair {-1920, -1080}));
  cursor.observe_injected_cursor({-1920, -1080});
  EXPECT_EQ(cursor.move(bounds, std::nullopt, 10, 20), (std::pair {-1910, -1060}));
}

TEST(InputCursorConfinementTests, AbsoluteRightBottomAndOutOfRangeStayOnLastPixel) {
  platf::detail::confined_cursor_state_t cursor;
  const platf::detail::cursor_bounds_t bounds {3840, 100, 1920, 1080};
  EXPECT_EQ(cursor.absolute(bounds, 1.0f, 1.0f), (std::pair {5759, 1179}));
  EXPECT_EQ(cursor.absolute(bounds, 2.0f, -1.0f), (std::pair {5759, 100}));
  EXPECT_FALSE(cursor.absolute(bounds, std::numeric_limits<float>::quiet_NaN(), 0));
}

TEST(InputCursorConfinementTests, RemotePositionSurvivesPhysicalMovementAndResize) {
  platf::detail::confined_cursor_state_t cursor;
  const platf::detail::cursor_bounds_t first {1920, 0, 1920, 1080};
  EXPECT_EQ(cursor.move(first, std::pair {50, 50}, 0, 0), (std::pair {2880, 540}));
  cursor.observe_injected_cursor({2880, 540});
  EXPECT_EQ(cursor.move(first, std::pair {50, 50}, 10, 0), (std::pair {2890, 540}));
  cursor.observe_injected_cursor({2890, 540});
  // Even stale GetCursorPos observations cannot discard successive queued relative deltas.
  EXPECT_EQ(cursor.move(first, std::pair {2880, 540}, 10, 0), (std::pair {2900, 540}));
  cursor.observe_injected_cursor({2900, 540});
  EXPECT_EQ(cursor.move(platf::detail::cursor_bounds_t {-800, -600, 800, 600}, std::nullopt, 0, 0), (std::pair {-1, -1}));
}

TEST(InputCursorConfinementTests, MissingTargetAndResetNeverFallBackToPhysicalDisplay) {
  platf::detail::confined_cursor_state_t old_session;
  const platf::detail::cursor_bounds_t bounds {1920, 0, 1920, 1080};
  ASSERT_TRUE(old_session.absolute(bounds, 0.5f, 0.5f));
  EXPECT_FALSE(old_session.move(std::nullopt, std::pair {50, 50}, 1, 1));
  EXPECT_FALSE(old_session.absolute(std::nullopt, 0.5f, 0.5f));
  EXPECT_FALSE(old_session.move(platf::detail::cursor_bounds_t {0, 0, 0, 0}, std::nullopt, 1, 1));
  old_session.reset();
  EXPECT_FALSE(old_session.move(bounds, std::nullopt, 1, 1));
  EXPECT_FALSE(old_session.absolute(bounds, 0.5f, 0.5f));
  platf::detail::confined_cursor_state_t replacement_session;
  EXPECT_EQ(replacement_session.move(bounds, std::nullopt, 0, 0), (std::pair {2880, 540}));
}

TEST(InputCursorConfinementTests, AbsoluteEncodingKeepsInteriorDisplayEdgesInsideTheirPixel) {
  for (int pixel : {-3840, -1921, -1920, -1, 0, 1919, 3839}) {
    const auto normalized = platf::detail::cursor_pixel_to_absolute(pixel, -3840, 7680);
    const auto decoded = static_cast<int>((static_cast<std::int64_t>(normalized) * 7680) / 65536) - 3840;
    EXPECT_EQ(decoded, pixel);
  }
}

TEST(InputCursorConfinementTests, RepeatedLeftInputStaysInsideAfterAppReleasesItsCursorClip) {
  using platf::detail::cursor_bounds_t;
  for (const auto &[monitor, desktop] : std::array {
         std::pair {cursor_bounds_t {0, 0, 1920, 1080}, cursor_bounds_t {-2560, 0, 4480, 1440}},
         std::pair {cursor_bounds_t {2560, 0, 1920, 1080}, cursor_bounds_t {0, 0, 4480, 1440}},
         std::pair {cursor_bounds_t {-1920, 0, 1920, 1080}, cursor_bounds_t {-1920, 0, 4480, 1440}},
         std::pair {cursor_bounds_t {0, 0, 1920, 1080}, cursor_bounds_t {0, 0, 1920, 1080}},
       }) {
    platf::detail::confined_cursor_state_t cursor;
    auto position = cursor.move(monitor, std::nullopt, -32768, 0);
    ASSERT_TRUE(position);
    auto injection = platf::detail::make_confined_cursor_injection(monitor, monitor, desktop, *position);
    ASSERT_TRUE(injection);
    cursor.observe_injected_cursor(injection->pixel);
    for (int packet = 0; packet < 20; ++packet) {
      // A game releases ClipCursor, and Windows/physical input reports an unrelated position.
      // Every subsequent remote event must still encode the virtual monitor's leftmost pixel.
      position = cursor.move(monitor, std::pair {desktop.x, desktop.y}, -32768, 0);
      ASSERT_TRUE(position);
      injection = platf::detail::make_confined_cursor_injection(monitor, desktop, desktop, *position);
      ASSERT_TRUE(injection);
      EXPECT_EQ(injection->pixel.first, monitor.x);
      const auto decoded_x = desktop.x + static_cast<std::int64_t>(injection->absolute.first) * desktop.width / 65536;
      const auto decoded_y = desktop.y + static_cast<std::int64_t>(injection->absolute.second) * desktop.height / 65536;
      EXPECT_EQ(decoded_x, monitor.x);
      EXPECT_TRUE(monitor.contains({static_cast<int>(decoded_x), static_cast<int>(decoded_y)}));
      cursor.observe_injected_cursor(injection->pixel);
    }
  }
}

TEST(InputCursorConfinementTests, IncoherentDesktopGeometryAndOutsideAppClipCannotProduceInjection) {
  const platf::detail::cursor_bounds_t monitor {0, 0, 1920, 1080};
  const platf::detail::cursor_bounds_t desktop {-2560, 0, 4480, 1440};
  EXPECT_FALSE(platf::detail::make_confined_cursor_injection(monitor, desktop, {-2560, 0, 2560, 1440}, {0, 500}));
  EXPECT_FALSE(platf::detail::make_confined_cursor_injection(monitor, desktop, {0, 0, 1280, 720}, {0, 500}));
  EXPECT_FALSE(platf::detail::make_confined_cursor_injection(monitor, desktop, {0, 0, 0, 0}, {0, 500}));
  EXPECT_FALSE(platf::detail::make_confined_cursor_injection(monitor, {-2560, 0, 2560, 1440}, desktop, {0, 500}));
}

TEST(InputCursorConfinementTests, FailedCandidatesDoNotAccumulateBeforeInputRecovers) {
  platf::detail::confined_cursor_state_t cursor;
  const platf::detail::cursor_bounds_t bounds {0, 0, 1920, 1080};
  const auto accepted = cursor.move(bounds, std::pair {900, 500}, 10, 0);
  ASSERT_TRUE(accepted);
  cursor.observe_injected_cursor(*accepted);
  EXPECT_EQ(*accepted, (std::pair {910, 500}));

  // Neither blocked relative input nor a rejected absolute placement changes the saved cursor.
  for (int packet = 0; packet < 10; ++packet) {
    EXPECT_EQ(cursor.move(bounds, std::pair {50, 50}, -2000, 0), (std::pair {0, 500}));
    EXPECT_EQ(cursor.absolute(bounds, 1.0f, 1.0f), (std::pair {1919, 1079}));
  }
  const auto recovered = cursor.move(bounds, std::pair {50, 50}, 10, 0);
  ASSERT_TRUE(recovered);
  EXPECT_EQ(*recovered, (std::pair {920, 500}));
  cursor.observe_injected_cursor(*recovered);
  // Successful queued movement still accumulates even before GetCursorPos catches up.
  EXPECT_EQ(cursor.move(bounds, std::pair {910, 500}, 10, 0), (std::pair {930, 500}));
}

TEST(InputCursorConfinementTests, DesktopRetryRejectsRebasedResizedOrMissingBounds) {
  const platf::detail::cursor_bounds_t original {1920, 100, 1920, 1080};
  EXPECT_TRUE(platf::detail::cursor_retry_bounds_match(original, original));
  EXPECT_FALSE(platf::detail::cursor_retry_bounds_match(original, platf::detail::cursor_bounds_t {0, 100, 1920, 1080}));
  EXPECT_FALSE(platf::detail::cursor_retry_bounds_match(original, platf::detail::cursor_bounds_t {1920, 0, 1920, 1080}));
  EXPECT_FALSE(platf::detail::cursor_retry_bounds_match(original, platf::detail::cursor_bounds_t {1920, 100, 1280, 720}));
  EXPECT_FALSE(platf::detail::cursor_retry_bounds_match(original, std::nullopt));
  EXPECT_FALSE(platf::detail::cursor_retry_bounds_match(std::nullopt, std::nullopt));
  EXPECT_FALSE(platf::detail::cursor_retry_bounds_match(platf::detail::cursor_bounds_t {0, 0, 0, 0}, platf::detail::cursor_bounds_t {0, 0, 0, 0}));
}

TEST(InputCursorConfinementTests, PhysicalAppClipCannotRedirectRemoteInputOntoPhysicalMonitor) {
  const platf::detail::cursor_bounds_t virtual_monitor {1920, 0, 1920, 1080};
  EXPECT_FALSE(platf::detail::cursor_bounds_intersection(virtual_monitor, {0, 0, 1920, 1080}));
  const auto partial = platf::detail::cursor_bounds_intersection(virtual_monitor, {1800, 100, 400, 500});
  ASSERT_TRUE(partial);
  EXPECT_EQ(partial->clamp(3000, 900), (std::pair {2199, 599}));
  EXPECT_EQ(partial->clamp(0, 0), (std::pair {1920, 100}));
}

TEST(InputCursorConfinementTests, ClonedAndIncompleteSourcesCannotBeAdmittedAsPrivateTargets) {
  platf::detail::cursor_source_identity_t private_source;
  private_source.observe(true, true);
  private_source.observe(false, false);  // A disconnected monitor is not a clone.
  EXPECT_TRUE(private_source.private_target(true));
  EXPECT_FALSE(private_source.private_target(false));
  private_source.observe(true, false);  // Physical output cloned onto the virtual source.
  EXPECT_FALSE(private_source.private_target(true));
  platf::detail::cursor_source_identity_t physical_only;
  physical_only.observe(true, false);
  EXPECT_FALSE(physical_only.private_target(true));
  physical_only.observe(true, true);  // Enumeration order cannot admit the same cloned source.
  EXPECT_FALSE(physical_only.private_target(true));
}

TEST(InputCursorConfinementTests, NativeMonitorIdentityResolverMatchesActiveInterfacesWithoutInjectingInput) {
  const auto previous_dpi = SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
  ASSERT_NE(previous_dpi, nullptr);
  const auto restore_dpi = util::fail_guard([previous_dpi]() {
    SetThreadDpiAwarenessContext(previous_dpi);
  });

  struct monitor_t {
    std::string path;
    RECT bounds;
    bool private_source;
  };

  std::vector<monitor_t> monitors;
  EnumDisplayMonitors(nullptr, nullptr, [](HMONITOR monitor, HDC, LPRECT, LPARAM context) -> BOOL {
    auto &targets = *reinterpret_cast<std::vector<monitor_t> *>(context);
    MONITORINFOEXW info {};
    info.cbSize = sizeof(info);
    if (!GetMonitorInfoW(monitor, &info)) {
      return TRUE;
    }
    const auto first = targets.size();
    unsigned active_interfaces = 0;
    bool enumeration_complete = false;
    for (DWORD index = 0; index < 16; ++index) {
      DISPLAY_DEVICEW device {};
      device.cb = sizeof(device);
      if (!EnumDisplayDevicesW(info.szDevice, index, &device, EDD_GET_DEVICE_INTERFACE_NAME)) {
        enumeration_complete = true;
        break;
      }
      if (device.StateFlags & DISPLAY_DEVICE_ACTIVE) {
        ++active_interfaces;
        if (device.DeviceID[0]) {
          targets.push_back({platf::to_utf8(device.DeviceID), info.rcMonitor, false});
        }
      }
    }
    for (auto index = first; index < targets.size(); ++index) {
      targets[index].private_source = enumeration_complete && active_interfaces == 1;
    }
    return TRUE;
  },
                      reinterpret_cast<LPARAM>(&monitors));
  if (monitors.empty()) {
    GTEST_SKIP() << "No active monitor interface is available on this test desktop";
  }
  for (const auto &monitor : monitors) {
    const auto resolved = platf::cursor_confinement_bounds_for_test(monitor.path);
    if (!monitor.private_source) {
      EXPECT_FALSE(resolved);
      continue;
    }
    ASSERT_TRUE(resolved);
    EXPECT_EQ(resolved->offset_x, monitor.bounds.left);
    EXPECT_EQ(resolved->offset_y, monitor.bounds.top);
    EXPECT_EQ(resolved->width, monitor.bounds.right - monitor.bounds.left);
    EXPECT_EQ(resolved->height, monitor.bounds.bottom - monitor.bounds.top);
    // A replacement identity must never resolve through a previous source/GDI monitor name.
    EXPECT_FALSE(platf::cursor_confinement_bounds_for_test(monitor.path + "-retired-session"));
  }
  EXPECT_FALSE(platf::cursor_confinement_bounds_for_test(""));
}

TEST(InputCursorConfinementTests, DesktopSyncRecoversClipQueryFailureOnlyWithOriginalBounds) {
  const platf::detail::cursor_bounds_t original {1920, 0, 1920, 1080};
  for (const bool rebased : {false, true}) {
    platf::detail::confined_cursor_state_t cursor;
    cursor.observe_injected_cursor({2500, 500});
    const auto candidate = cursor.move(original, std::nullopt, 100, 0);
    ASSERT_TRUE(candidate);
    auto resolved = original;
    bool clip_query_available = false;
    unsigned clip_queries = 0, accepted = 0, synchronizations = 0;
    const auto sent = platf::detail::run_with_desktop_retry(
      [&]() {
        if (!platf::detail::cursor_retry_bounds_match(original, resolved)) {
          return false;
        }
        ++clip_queries;
        if (!clip_query_available) {
          return false;
        }
        ++accepted;
        cursor.observe_injected_cursor(*candidate);
        return true;
      },
      [&]() {
        EXPECT_EQ(accepted, 0u);  // GetClipCursor failed before reaching SendInput.
        ++synchronizations;
        clip_query_available = true;
        if (rebased) {
          resolved.x = 0;
        }
        return true;
      }
    );
    EXPECT_EQ(sent, !rebased);
    EXPECT_EQ(synchronizations, 1u);
    EXPECT_EQ(clip_queries, rebased ? 1u : 2u);
    EXPECT_EQ(accepted, rebased ? 0u : 1u);
    EXPECT_EQ(cursor.move(original, std::nullopt, 0, 0), (std::pair {rebased ? 2500 : 2600, 500}));
  }
}

TEST(InputPermissionTests, RevocationPurgesQueuedPressesAndReleasesHeldCategoryBeforeRegrant) {
  std::map<unsigned, bool> keys {{65, true}, {66, false}};
  std::array<bool, 6> buttons {false, true, false, false, false, false};
  std::list<std::vector<std::uint8_t>> packets;
  packets.push_back(make_input_packet(KEY_DOWN_EVENT_MAGIC, sizeof(NV_KEYBOARD_PACKET)));
  const auto allowed_move = make_relative_mouse_packet(4, 5);
  packets.push_back(allowed_move);
  input::detail::queue_permission_release(packets, crypto::PERM::input_kbd);
  // An immediate regrant's new press must remain after the release of the old hold.
  const auto regranted_press = make_input_packet(KEY_DOWN_EVENT_MAGIC, sizeof(NV_KEYBOARD_PACKET));
  packets.push_back(regranted_press);

  std::vector<unsigned> released_keys;
  std::vector<unsigned> released_buttons;
  auto barrier = input::detail::pop_next_batched_packet(packets);
  ASSERT_TRUE(barrier);
  const auto revoked = input::detail::release_barrier_permissions(*barrier);
  ASSERT_TRUE(revoked);
  EXPECT_FALSE(input::validated_packet_magic(*barrier));
  input::detail::release_pressed_states(keys, buttons, *revoked,
    [&](unsigned key) { released_keys.push_back(key); },
    [&](unsigned button) { released_buttons.push_back(button); });
  EXPECT_EQ(released_keys, (std::vector<unsigned> {65}));
  EXPECT_TRUE(released_buttons.empty());
  EXPECT_FALSE(keys[65]);
  EXPECT_TRUE(buttons[1]);
  EXPECT_EQ(input::detail::pop_next_batched_packet(packets), allowed_move);
  EXPECT_EQ(input::detail::pop_next_batched_packet(packets), regranted_press);
  EXPECT_TRUE(packets.empty());

  input::detail::queue_permission_release(packets, crypto::PERM::input_mouse);
  barrier = input::detail::pop_next_batched_packet(packets);
  input::detail::release_pressed_states(keys, buttons, *input::detail::release_barrier_permissions(*barrier),
    [&](unsigned key) { released_keys.push_back(key); },
    [&](unsigned button) { released_buttons.push_back(button); });
  EXPECT_EQ(released_buttons, (std::vector<unsigned> {1}));
  EXPECT_FALSE(buttons[1]);
  EXPECT_EQ(released_keys.size(), 1U);
}

TEST(InputPermissionTests, ReleaseBarriersPreserveOtherRevocationsAndResetOrdering) {
  std::list<std::vector<std::uint8_t>> packets;
  packets.push_back(make_input_packet(SS_TOUCH_MAGIC, sizeof(SS_TOUCH_PACKET)));
  packets.push_back(make_input_packet(SS_PEN_MAGIC, sizeof(SS_PEN_PACKET)));
  packets.push_back(make_input_packet(MULTI_CONTROLLER_MAGIC_GEN5, sizeof(NV_MULTI_CONTROLLER_PACKET)));
  input::detail::queue_permission_release(packets, crypto::PERM::input_touch);
  input::detail::queue_permission_release(packets, crypto::PERM::input_pen);
  auto first = input::detail::pop_next_batched_packet(packets);
  auto second = input::detail::pop_next_batched_packet(packets);
  EXPECT_EQ(input::detail::release_barrier_permissions(*first), crypto::PERM::input_pen);
  EXPECT_EQ(input::detail::release_barrier_permissions(*second), crypto::PERM::input_touch);
  const auto remaining = input::detail::pop_next_batched_packet(packets);
  EXPECT_EQ(input::detail::packet_permission(*remaining), crypto::PERM::input_controller);
  packets.emplace_back();
  EXPECT_TRUE(input::detail::pop_next_batched_packet(packets)->empty());
}

TEST(InputPermissionTests, RevocationKeepsUnrelatedPoppedKeyUpAndInvalidatesOldMouseAfterRegrant) {
  input::detail::dispatch_generations_t generations;
  std::mutex dispatch_mutex;
  const auto popped_key_up = generations.current(crypto::PERM::input_kbd);
  const auto popped_mouse_down = generations.current(crypto::PERM::input_mouse);
  generations.invalidate(crypto::PERM::input_mouse);
  // Regrant does not rewind generations: the old press remains stale after a quick regrant.
  bool key_released = false;
  EXPECT_TRUE(input::detail::dispatch_if_current(dispatch_mutex, popped_key_up,
    generations.current(crypto::PERM::input_kbd), [&]() { key_released = true; }));
  EXPECT_TRUE(key_released);
  EXPECT_FALSE(input::detail::dispatch_if_current(dispatch_mutex, popped_mouse_down,
    generations.current(crypto::PERM::input_mouse), []() { ADD_FAILURE(); }));
  const auto new_mouse = generations.current(crypto::PERM::input_mouse);
  EXPECT_TRUE(input::detail::dispatch_if_current(dispatch_mutex, new_mouse,
    generations.current(crypto::PERM::input_mouse), []() {}));
  generations.invalidate(crypto::PERM::_all_inputs);
  EXPECT_FALSE(input::detail::dispatch_if_current(dispatch_mutex, popped_key_up,
    generations.current(crypto::PERM::input_kbd), []() { ADD_FAILURE(); }));
  EXPECT_FALSE(input::detail::dispatch_if_current(dispatch_mutex, new_mouse,
    generations.current(crypto::PERM::input_mouse), []() { ADD_FAILURE(); }));
}

TEST(InputPermissionTests, ControllerRevocationNeutralizesBeforeRegrantWithoutLosingArrivalIdentity) {
  struct gamepad_t {
    int id;
    platf::gamepad_state_t gamepad_state;
  };
  std::array<gamepad_t, 2> gamepads {{{2, {}}, {-1, {}}}};
  gamepads[0].gamepad_state.buttonFlags = platf::HOME | platf::BACK;
  gamepads[0].gamepad_state.lt = 255;
  std::vector<int> updates;
  input::detail::neutralize_gamepads(gamepads, [&](int id, const auto &state) {
    EXPECT_EQ(state.buttonFlags, 0U);
    EXPECT_EQ(state.lt, 0U);
    updates.push_back(id);
  });
  EXPECT_EQ(updates, (std::vector<int> {2}));
  EXPECT_EQ(gamepads[0].id, 2);  // A later authorized state report still has its arrival device.
  EXPECT_EQ(gamepads[1].id, -1);
  EXPECT_EQ(gamepads[0].gamepad_state.buttonFlags, 0U);
  input::detail::free_gamepads_before_completion(gamepads, [&](int id) { updates.push_back(id); }, []() {});
  EXPECT_EQ(updates, (std::vector<int> {2, 2}));
  EXPECT_EQ(gamepads[0].id, -1);  // Full teardown still frees before completing.
}

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

  EXPECT_FALSE(input::validated_packet_magic(make_input_packet(UTF8_TEXT_EVENT_MAGIC, sizeof(NV_INPUT_HEADER))));
  EXPECT_FALSE(input::validated_packet_magic(make_input_packet(UTF8_TEXT_EVENT_MAGIC, sizeof(NV_UNICODE_PACKET) + 1)));
}

TEST(InputPacketValidationTests, RejectsRuntOversizedUnknownAndMismatchedPackets) {
  for (std::size_t packet_size = 0; packet_size < sizeof(NV_INPUT_HEADER); ++packet_size) {
    std::vector<std::uint8_t> packet(packet_size);
    EXPECT_FALSE(input::validated_packet_magic(packet));
  }

  EXPECT_FALSE(input::validated_packet_magic(make_input_packet(0xDEADBEEF, sizeof(NV_INPUT_HEADER))));
  EXPECT_FALSE(input::validated_packet_magic(make_input_packet(UTF8_TEXT_EVENT_MAGIC, input::INPUT_PACKET_SIZE_MAX + 1)));

  auto mismatched = make_input_packet(KEY_DOWN_EVENT_MAGIC, sizeof(NV_KEYBOARD_PACKET));
  const auto wrong_size = util::endian::big<std::uint32_t>(1);
  std::memcpy(mismatched.data(), &wrong_size, sizeof(wrong_size));
  EXPECT_FALSE(input::validated_packet_magic(mismatched));
}

TEST(InputBatchingTests, AddsOrdinarySignedDeltasExactly) {
  EXPECT_EQ(input::detail::checked_add_i16(12, 7), 19);
  EXPECT_EQ(input::detail::checked_add_i16(-12, -7), -19);
  EXPECT_EQ(input::detail::checked_add_i16(12, -7), 5);
  EXPECT_EQ(input::detail::checked_add_i16(-12, 7), -5);
}

TEST(InputBatchingTests, AcceptsRepresentableBoundarySums) {
  EXPECT_EQ(input::detail::checked_add_i16(32760, 7), 32767);
  EXPECT_EQ(input::detail::checked_add_i16(-32760, -8), -32768);
}

TEST(InputBatchingTests, RejectsPositiveAndNegativeOverflow) {
  EXPECT_FALSE(input::detail::checked_add_i16(32760, 8));
  EXPECT_FALSE(input::detail::checked_add_i16(-32760, -9));
}

TEST(InputBatchingTests, ProductionCoalescerBatchesMouseAndBothScrollAxes) {
  auto mouse = make_relative_mouse_packet(12, -7);
  const auto later_mouse = make_relative_mouse_packet(7, 12);
  EXPECT_EQ(
    input::detail::batch_packets(mouse, later_mouse),
    input::detail::batch_result_e::batched
  );
  EXPECT_EQ(relative_mouse_delta(mouse), (std::pair<std::int16_t, std::int16_t> {19, 5}));

  auto vertical = make_vertical_scroll_packet(90);
  const auto later_vertical = make_vertical_scroll_packet(-30);
  EXPECT_EQ(
    input::detail::batch_packets(vertical, later_vertical),
    input::detail::batch_result_e::batched
  );
  const auto *vertical_result = reinterpret_cast<const NV_SCROLL_PACKET *>(vertical.data());
  EXPECT_EQ(util::endian::big(vertical_result->scrollAmt1), 60);
  EXPECT_EQ(util::endian::big(vertical_result->scrollAmt2), 60);

  auto horizontal = make_horizontal_scroll_packet(-90);
  const auto later_horizontal = make_horizontal_scroll_packet(30);
  EXPECT_EQ(
    input::detail::batch_packets(horizontal, later_horizontal),
    input::detail::batch_result_e::batched
  );
  EXPECT_EQ(
    util::endian::big(reinterpret_cast<const SS_HSCROLL_PACKET *>(horizontal.data())->scrollAmount),
    -60
  );
}

TEST(InputBatchingTests, ProductionCoalescerIncludesBothInt16Boundaries) {
  auto mouse = make_relative_mouse_packet(32760, -32760);
  const auto later_mouse = make_relative_mouse_packet(7, -8);
  ASSERT_EQ(
    input::detail::batch_packets(mouse, later_mouse),
    input::detail::batch_result_e::batched
  );
  EXPECT_EQ(
    relative_mouse_delta(mouse),
    (std::pair<std::int16_t, std::int16_t> {32767, -32768})
  );

  auto vertical = make_vertical_scroll_packet(32760);
  const auto later_vertical = make_vertical_scroll_packet(7);
  ASSERT_EQ(
    input::detail::batch_packets(vertical, later_vertical),
    input::detail::batch_result_e::batched
  );
  EXPECT_EQ(
    util::endian::big(reinterpret_cast<const NV_SCROLL_PACKET *>(vertical.data())->scrollAmt1),
    32767
  );

  auto horizontal = make_horizontal_scroll_packet(-32760);
  const auto later_horizontal = make_horizontal_scroll_packet(-8);
  ASSERT_EQ(
    input::detail::batch_packets(horizontal, later_horizontal),
    input::detail::batch_result_e::batched
  );
  EXPECT_EQ(
    util::endian::big(reinterpret_cast<const SS_HSCROLL_PACKET *>(horizontal.data())->scrollAmount),
    -32768
  );
}

TEST(InputBatchingTests, SingleAxisOverflowLeavesDestinationByteIdentical) {
  {
    auto mouse = make_relative_mouse_packet(32760, 12);
    const auto before = mouse;
    const auto later_mouse = make_relative_mouse_packet(8, 7);
    EXPECT_EQ(
      input::detail::batch_packets(mouse, later_mouse),
      input::detail::batch_result_e::terminate_batch
    );
    EXPECT_EQ(mouse, before);
  }

  {
    auto mouse = make_relative_mouse_packet(12, -32760);
    const auto before = mouse;
    const auto later_mouse = make_relative_mouse_packet(7, -9);
    EXPECT_EQ(
      input::detail::batch_packets(mouse, later_mouse),
      input::detail::batch_result_e::terminate_batch
    );
    EXPECT_EQ(mouse, before);
  }
}

TEST(InputBatchingTests, ButtonPacketsRemainOrderingBoundaries) {
  std::list<std::vector<std::uint8_t>> queue;
  queue.push_back(make_relative_mouse_packet(2, 3));
  queue.push_back(make_mouse_button_packet(MOUSE_BUTTON_DOWN_EVENT_MAGIC_GEN5, 1));
  queue.push_back(make_relative_mouse_packet(5, 7));
  queue.push_back(make_mouse_button_packet(MOUSE_BUTTON_UP_EVENT_MAGIC_GEN5, 1));

  const auto before_down = input::detail::pop_next_batched_packet(queue);
  const auto down = input::detail::pop_next_batched_packet(queue);
  const auto before_up = input::detail::pop_next_batched_packet(queue);
  const auto up = input::detail::pop_next_batched_packet(queue);

  ASSERT_TRUE(before_down);
  ASSERT_TRUE(down);
  ASSERT_TRUE(before_up);
  ASSERT_TRUE(up);
  EXPECT_EQ(relative_mouse_delta(*before_down), (std::pair<std::int16_t, std::int16_t> {2, 3}));
  EXPECT_EQ(input::validated_packet_magic(*down), MOUSE_BUTTON_DOWN_EVENT_MAGIC_GEN5);
  EXPECT_EQ(relative_mouse_delta(*before_up), (std::pair<std::int16_t, std::int16_t> {5, 7}));
  EXPECT_EQ(input::validated_packet_magic(*up), MOUSE_BUTTON_UP_EVENT_MAGIC_GEN5);
  EXPECT_TRUE(queue.empty());
}

TEST(InputBatchingTests, ResetBarrierPreservesPacketsOnBothSides) {
  std::list<std::vector<std::uint8_t>> queue;
  queue.push_back(make_relative_mouse_packet(2, 3));
  queue.emplace_back();
  queue.push_back(make_relative_mouse_packet(5, 7));

  const auto before_reset = input::detail::pop_next_batched_packet(queue);
  const auto reset = input::detail::pop_next_batched_packet(queue);
  const auto after_reset = input::detail::pop_next_batched_packet(queue);

  ASSERT_TRUE(before_reset);
  ASSERT_TRUE(reset);
  ASSERT_TRUE(after_reset);
  EXPECT_EQ(relative_mouse_delta(*before_reset), (std::pair<std::int16_t, std::int16_t> {2, 3}));
  EXPECT_TRUE(reset->empty());
  EXPECT_EQ(relative_mouse_delta(*after_reset), (std::pair<std::int16_t, std::int16_t> {5, 7}));
  EXPECT_TRUE(queue.empty());
}

TEST(InputDrainGateTests, CoalescesWakeupsAndHandsOffWithoutLosingAnArrival) {
  input::detail::drain_gate_t gate;

  EXPECT_TRUE(gate.request());
  EXPECT_TRUE(gate.active());
  EXPECT_FALSE(gate.request());
  EXPECT_FALSE(gate.request());

  // An arrival before the drain releases ownership is covered by the active owner.
  EXPECT_TRUE(input::detail::release_if_empty(gate, false));
  EXPECT_TRUE(gate.active());

  // The drain's empty-to-idle transition occurs atomically under the same queue mutex used by
  // arrivals, so the next arrival becomes exactly one new owner.
  EXPECT_FALSE(input::detail::release_if_empty(gate, true));
  EXPECT_FALSE(gate.active());

  EXPECT_TRUE(gate.request());
  EXPECT_FALSE(gate.request());
}

TEST(InputDrainGateTests, QuantumYieldsOneContinuationThatRetainsOwnership) {
  input::detail::drain_gate_t gate;
  ASSERT_TRUE(gate.request());

  EXPECT_FALSE(input::detail::drain_turn_exhausted(0));
  EXPECT_FALSE(input::detail::drain_turn_exhausted(input::detail::INPUT_DRAIN_QUANTUM - 1));
  EXPECT_TRUE(input::detail::drain_turn_exhausted(input::detail::INPUT_DRAIN_QUANTUM));
  EXPECT_TRUE(input::detail::drain_continuation_uses_timer_queue());

  // The continuation retains ownership, including an arrival racing the task handoff.
  EXPECT_TRUE(gate.active());
  EXPECT_FALSE(gate.request());

  // Once the continuation observes an empty queue, the next arrival owns one new drain.
  gate.release();
  EXPECT_TRUE(gate.request());
  EXPECT_FALSE(gate.request());
}

TEST(InputDrainGateTests, ResetGenerationInvalidatesOnlyPacketsAlreadyRemovedByAnOldDrain) {
  constexpr std::uint64_t before_reset = 7;
  constexpr std::uint64_t after_reset = before_reset + 1;

  EXPECT_TRUE(input::detail::generation_is_current(before_reset, before_reset));
  EXPECT_FALSE(input::detail::generation_is_current(before_reset, after_reset));
  EXPECT_TRUE(input::detail::generation_is_current(after_reset, after_reset));
}

TEST(InputDrainGateTests, DispatchAdmissionIsAtomicWithResetGenerationChange) {
  std::mutex dispatch_mutex;
  std::uint64_t generation = 7;
  bool dispatched = false;

  EXPECT_TRUE(input::detail::dispatch_if_current(
    dispatch_mutex,
    generation,
    generation,
    [&]() {
      dispatched = true;
    }
  ));
  EXPECT_TRUE(dispatched);

  ++generation;  // Reset wins the same serialization mutex before stale dispatch admission.
  dispatched = false;
  EXPECT_FALSE(input::detail::dispatch_if_current(
    dispatch_mutex,
    generation - 1,
    generation,
    [&]() {
      dispatched = true;
    }
  ));
  EXPECT_FALSE(dispatched);
}

TEST(InputDelayedActionTests, ResetAndSupersedingTimersInvalidateStaleContinuations) {
  EXPECT_TRUE(input::detail::delayed_action_is_current(false, 11, 11));
  EXPECT_FALSE(input::detail::delayed_action_is_current(true, 11, 11));
  EXPECT_FALSE(input::detail::delayed_action_is_current(false, 11, 12));
}

TEST(InputDelayedActionTests, RapidRescheduleRejectsOlderCallbackBeforeItCanClearNewState) {
  constexpr std::uint64_t stale_action = 11;
  constexpr std::uint64_t current_action = 12;
  bool current_timer_present = true;

  if (input::detail::delayed_action_is_current(false, stale_action, current_action)) {
    current_timer_present = false;
  }
  EXPECT_TRUE(current_timer_present);

  if (input::detail::delayed_action_is_current(false, current_action, current_action)) {
    current_timer_present = false;
  }
  EXPECT_FALSE(current_timer_present);
}

TEST(InputDelayedActionTests, SecondBackTimerDoesNotInvalidateActiveHomeRelease) {
  std::uint64_t back_generation = 11;
  const std::uint64_t home_generation = 21;

  ++back_generation;  // A second BACK press schedules another long-press timeout.
  EXPECT_TRUE(input::detail::delayed_action_is_current(false, home_generation, 21));
  EXPECT_FALSE(input::detail::delayed_action_is_current(false, 11, back_generation));
}

TEST(InputDelayedActionTests, ControllerRemovalInvalidatesPendingBackAndHomeCallbacks) {
  EXPECT_TRUE(input::detail::controller_action_is_current(false, 3, 11, 11));
  EXPECT_FALSE(input::detail::controller_action_is_current(false, -1, 11, 11));
  EXPECT_FALSE(input::detail::controller_action_is_current(false, -1, 21, 22));
}

TEST(InputDelayedActionTests, HomeRemainsLatchedAcrossReportsUntilReleaseContinuation) {
  constexpr std::uint32_t home = 0x400;
  constexpr std::uint32_t ordinary_report = 0x20;

  EXPECT_EQ(
    input::detail::latch_button_while_active(ordinary_report, home, true),
    ordinary_report | home
  );
  EXPECT_EQ(
    input::detail::latch_button_while_active(ordinary_report, home, false),
    ordinary_report
  );
}

TEST(InputDrainGateTests, ResetStateIsAnAdmissionBarrierForNewPackets) {
  bool queued = false;
  EXPECT_TRUE(input::detail::admit_if_live(false, [&]() {
    queued = true;
  }));
  EXPECT_TRUE(queued);

  queued = false;
  EXPECT_FALSE(input::detail::admit_if_live(true, [&]() {
    queued = true;
  }));
  EXPECT_FALSE(queued);
}

TEST(InputDelayedActionTests, ObsoleteAbsoluteLeftReleaseClearsCompletedTimerToken) {
  auto token = reinterpret_cast<void *>(0x1234);
  token = nullptr;  // Callback clears its completed task token before checking button state.
  EXPECT_FALSE(input::detail::delayed_action_needed(true));
  EXPECT_EQ(token, nullptr);

  // A subsequent LEFT-up sees the empty token and may publish a fresh delayed task.
  token = reinterpret_cast<void *>(0x5678);
  EXPECT_NE(token, nullptr);
  EXPECT_TRUE(input::detail::delayed_action_needed(false));
}

TEST(InputDelayedActionTests, RelativeModeInvalidatesOldAbsoluteReleaseWithoutClearingSentinel) {
  std::atomic<std::uint64_t> generation {7};
  auto timeout_state = reinterpret_cast<void *>(0x1);  // Relative-mode sentinel.

  ++generation;  // Relative movement wins before the old generation-7 callback.
  EXPECT_FALSE(input::detail::claim_delayed_generation(generation, 7));
  EXPECT_EQ(timeout_state, reinterpret_cast<void *>(0x1));

  EXPECT_TRUE(input::detail::claim_delayed_generation(generation, 8));
}

TEST(InputDelayedActionTests, TracksAllFiveProtocolMouseButtonsForResetRelease) {
  EXPECT_FALSE(input::detail::valid_mouse_button(0));
  for (std::uint8_t button = 1; button <= 5; ++button) {
    EXPECT_TRUE(input::detail::valid_mouse_button(button));
  }
  EXPECT_FALSE(input::detail::valid_mouse_button(6));
}

TEST(InputDelayedActionTests, PendingAbsoluteLeftReleaseIsFlushedOnRelativeModeOrReset) {
  auto timer = reinterpret_cast<void *>(0x1234);
  const auto sentinel = reinterpret_cast<void *>(0x1);
  EXPECT_TRUE(input::detail::should_flush_pending_left_release(timer, sentinel, false));
  EXPECT_FALSE(input::detail::should_flush_pending_left_release(timer, sentinel, true));
  EXPECT_FALSE(input::detail::should_flush_pending_left_release(static_cast<void *>(nullptr), sentinel, false));
  EXPECT_FALSE(input::detail::should_flush_pending_left_release(sentinel, sentinel, false));
}

TEST(InputDelayedActionTests, KeyRepeatRequiresLiveMatchingPressedGeneration) {
  EXPECT_TRUE(input::detail::key_repeat_is_current(false, 7, 7, true));
  EXPECT_FALSE(input::detail::key_repeat_is_current(true, 7, 7, true));
  EXPECT_FALSE(input::detail::key_repeat_is_current(false, 6, 7, true));
  EXPECT_FALSE(input::detail::key_repeat_is_current(false, 7, 7, false));
}

TEST(InputResetBarrierTests, FreesVirtualGamepadsSynchronouslyBeforeCompletion) {
  struct fake_gamepad_t {
    int id;
  };
  std::array<fake_gamepad_t, 3> gamepads {{{2}, {-1}, {5}}};
  std::vector<int> events;

  input::detail::free_gamepads_before_completion(
    gamepads,
    [&](int id) {
      events.push_back(id);
    },
    [&]() {
      for (const auto &gamepad : gamepads) {
        EXPECT_EQ(gamepad.id, -1);
      }
      events.push_back(99);
    }
  );

  ASSERT_EQ(events.size(), 3);
  EXPECT_EQ(events[0], 2);
  EXPECT_EQ(events[1], 5);
  EXPECT_EQ(events[2], 99);
}
