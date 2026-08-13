/**
 * @file src/input.cpp
 * @brief Definitions for gamepad, keyboard, and mouse input handling.
 */
#include <cstdint>
extern "C" {
#include <moonlight-common-c/src/Input.h>
#include <moonlight-common-c/src/Limelight.h>
}

// standard includes
#include <bitset>
#include <chrono>
#include <cmath>
#include <cstring>
#include <list>
#include <thread>
#include <unordered_map>

// lib includes
#include <boost/endian/buffers.hpp>

// local includes
#include "config.h"
#include "globals.h"
#include "input.h"
#include "logging.h"
#include "platform/common.h"
#include "thread_pool.h"
#include "utility.h"

// Win32 WHEEL_DELTA constant
#ifndef WHEEL_DELTA
  #define WHEEL_DELTA 120
#endif

using namespace std::literals;

namespace input {

  constexpr auto MAX_GAMEPADS = std::min((std::size_t) platf::MAX_GAMEPADS, sizeof(std::int16_t) * 8);
#define DISABLE_LEFT_BUTTON_DELAY ((thread_pool_util::ThreadPool::task_id_t) 0x01)
#define ENABLE_LEFT_BUTTON_DELAY nullptr

  constexpr auto VKEY_SHIFT = 0x10;
  constexpr auto VKEY_LSHIFT = 0xA0;
  constexpr auto VKEY_RSHIFT = 0xA1;
  constexpr auto VKEY_CONTROL = 0x11;
  constexpr auto VKEY_LCONTROL = 0xA2;
  constexpr auto VKEY_RCONTROL = 0xA3;
  constexpr auto VKEY_MENU = 0x12;
  constexpr auto VKEY_LMENU = 0xA4;
  constexpr auto VKEY_RMENU = 0xA5;

  std::optional<std::uint32_t> validated_packet_magic(std::span<const std::uint8_t> input_data) noexcept {
    if (input_data.size() < sizeof(NV_INPUT_HEADER) || input_data.size() > INPUT_PACKET_SIZE_MAX) {
      return std::nullopt;
    }

    NV_INPUT_HEADER header;
    std::memcpy(&header, input_data.data(), sizeof(header));

    const auto declared_size = util::endian::big(header.size);
    const auto magic = util::endian::little(header.magic);
    if (declared_size != input_data.size() - sizeof(header.size)) {
      return std::nullopt;
    }

    std::size_t expected_size;
    switch (magic) {
      case ENABLE_HAPTICS_MAGIC:
        expected_size = sizeof(NV_HAPTICS_PACKET);
        break;
      case KEY_DOWN_EVENT_MAGIC:
      case KEY_UP_EVENT_MAGIC:
        expected_size = sizeof(NV_KEYBOARD_PACKET);
        break;
      case UTF8_TEXT_EVENT_MAGIC:
        if (input_data.size() > sizeof(NV_INPUT_HEADER) && input_data.size() <= sizeof(NV_UNICODE_PACKET)) {
          return magic;
        }
        return std::nullopt;
      case MOUSE_MOVE_REL_MAGIC_GEN5:
        expected_size = sizeof(NV_REL_MOUSE_MOVE_PACKET);
        break;
      case MOUSE_MOVE_ABS_MAGIC:
        expected_size = sizeof(NV_ABS_MOUSE_MOVE_PACKET);
        break;
      case MOUSE_BUTTON_DOWN_EVENT_MAGIC_GEN5:
      case MOUSE_BUTTON_UP_EVENT_MAGIC_GEN5:
        expected_size = sizeof(NV_MOUSE_BUTTON_PACKET);
        break;
      case MULTI_CONTROLLER_MAGIC_GEN5:
        expected_size = sizeof(NV_MULTI_CONTROLLER_PACKET);
        break;
      case SCROLL_MAGIC_GEN5:
        expected_size = sizeof(NV_SCROLL_PACKET);
        break;
      case SS_HSCROLL_MAGIC:
        expected_size = sizeof(SS_HSCROLL_PACKET);
        break;
      case SS_TOUCH_MAGIC:
        expected_size = sizeof(SS_TOUCH_PACKET);
        break;
      case SS_PEN_MAGIC:
        expected_size = sizeof(SS_PEN_PACKET);
        break;
      case SS_CONTROLLER_ARRIVAL_MAGIC:
        expected_size = sizeof(SS_CONTROLLER_ARRIVAL_PACKET);
        break;
      case SS_CONTROLLER_TOUCH_MAGIC:
        expected_size = sizeof(SS_CONTROLLER_TOUCH_PACKET);
        break;
      case SS_CONTROLLER_MOTION_MAGIC:
        expected_size = sizeof(SS_CONTROLLER_MOTION_PACKET);
        break;
      case SS_CONTROLLER_BATTERY_MAGIC:
        expected_size = sizeof(SS_CONTROLLER_BATTERY_PACKET);
        break;
      default:
        return std::nullopt;
    }

    if (input_data.size() == expected_size) {
      return magic;
    }
    return std::nullopt;
  }

  enum class button_state_e {
    NONE,  ///< No button state
    DOWN,  ///< Button is down
    UP  ///< Button is up
  };

  template<std::size_t N>
  int alloc_id(std::bitset<N> &gamepad_mask) {
    for (int x = 0; x < gamepad_mask.size(); ++x) {
      if (!gamepad_mask[x]) {
        gamepad_mask[x] = true;
        return x;
      }
    }

    return -1;
  }

  template<std::size_t N>
  void free_id(std::bitset<N> &gamepad_mask, int id) {
    gamepad_mask[id] = false;
  }

  typedef uint32_t key_press_id_t;

  key_press_id_t make_kpid(uint16_t vk, uint8_t flags) {
    return (key_press_id_t) vk << 8 | flags;
  }

  uint16_t vk_from_kpid(key_press_id_t kpid) {
    return kpid >> 8;
  }

  uint8_t flags_from_kpid(key_press_id_t kpid) {
    return kpid & 0xFF;
  }

  /**
   * @brief Convert a little-endian netfloat to a native endianness float.
   * @param f Netfloat value.
   * @return The native endianness float value.
   */
  float from_netfloat(netfloat f) {
    return boost::endian::endian_load<float, sizeof(float), boost::endian::order::little>(f);
  }

  /**
   * @brief Convert a little-endian netfloat to a native endianness float and clamps it.
   * @param f Netfloat value.
   * @param min The minimium value for clamping.
   * @param max The maximum value for clamping.
   * @return Clamped native endianess float value.
   */
  float from_clamped_netfloat(netfloat f, float min, float max) {
    return std::clamp(from_netfloat(f), min, max);
  }

  static std::unordered_map<key_press_id_t, bool> key_press {};
  // Protocol mouse buttons are numbered 1 through 5; index 0 is intentionally unused.
  static std::array<std::uint8_t, 6> mouse_press {};

  static platf::input_t platf_input;
  static std::bitset<platf::MAX_GAMEPADS> gamepadMask {};

  void free_gamepad(platf::input_t &platf_input, int id) {
    platf::gamepad_update(platf_input, id, platf::gamepad_state_t {});
    platf::free_gamepad(platf_input, id);

    free_id(gamepadMask, id);
  }

  struct gamepad_t {
    gamepad_t():
        gamepad_state {},
        back_timeout_id {},
        home_release_id {},
        back_action_generation {},
        home_action_generation {},
        id {-1},
        back_button_state {button_state_e::NONE} {
    }

    ~gamepad_t() {
      if (id >= 0) {
        task_pool.push([id = this->id]() {
          free_gamepad(platf_input, id);
        });
      }
    }

    platf::gamepad_state_t gamepad_state;

    thread_pool_util::ThreadPool::task_id_t back_timeout_id;
    thread_pool_util::ThreadPool::task_id_t home_release_id;
    std::uint64_t back_action_generation;
    std::uint64_t home_action_generation;

    int id;

    // When emulating the HOME button, we may need to artificially release the back button.
    // Afterwards, the gamepad state on Apollo won't match the state on Artemis.
    // To prevent Sunshine from sending erroneous input data to the active application,
    // Sunshine forces the button to be in a specific state until the gamepad state matches that of
    // Artemis once more.
    button_state_e back_button_state;
  };

  struct input_t {
    enum modifier_e {
      CTRL = 0x1,  ///< Control key
      ALT = 0x2,  ///< Alt key
      SHIFT = 0x4  ///< Shift key
    };

    input_t(
      safe::mail_raw_t::event_t<input::touch_port_t> touch_port_event,
      platf::feedback_queue_t feedback_queue
    ):
        pressed_modifiers {},
        gamepads(MAX_GAMEPADS),
        client_context {platf::allocate_client_input_context(platf_input)},
        touch_port_event {std::move(touch_port_event)},
        feedback_queue {std::move(feedback_queue)},
        key_repeat_id {},
        key_repeat_generation {},
        mouse_left_button_timeout {},
        mouse_left_button_generation {},
        touch_port {{0, 0, 0, 0}, 0, 0, 1.0f},
        accumulated_vscroll_delta {},
        accumulated_hscroll_delta {} {
    }

    int pressed_modifiers;

    bool left_alt_pressed = false;
    bool right_alt_pressed = false;

    std::vector<gamepad_t> gamepads;
    std::unique_ptr<platf::client_input_t> client_context;

    safe::mail_raw_t::event_t<input::touch_port_t> touch_port_event;
    platf::feedback_queue_t feedback_queue;

    std::list<std::vector<uint8_t>> input_queue;
    std::mutex input_queue_lock;
    std::mutex input_dispatch_lock;
    detail::drain_gate_t input_drain_gate;
    std::uint64_t input_generation = 0;
    bool input_reset = false;
    std::shared_ptr<std::promise<void>> reset_completion;

    thread_pool_util::ThreadPool::task_id_t key_repeat_id;
    std::uint64_t key_repeat_generation;

    std::atomic<thread_pool_util::ThreadPool::task_id_t> mouse_left_button_timeout;
    std::atomic<std::uint64_t> mouse_left_button_generation;

    input::touch_port_t touch_port;

    int32_t accumulated_vscroll_delta;
    int32_t accumulated_hscroll_delta;
  };

  void passthrough(std::shared_ptr<input_t> &input, PNV_REL_MOUSE_MOVE_PACKET packet) {
    ++input->mouse_left_button_generation;
    const auto old_timer = input->mouse_left_button_timeout.exchange(DISABLE_LEFT_BUTTON_DELAY);
    if (old_timer && old_timer != DISABLE_LEFT_BUTTON_DELAY) {
      task_pool.cancel(old_timer);
      // Absolute LEFT-up has already changed logical state but was waiting for injection. Flush it
      // before entering relative mode so cancelling the delayed task cannot strand OS LEFT down.
      if (detail::should_flush_pending_left_release(
            old_timer,
            DISABLE_LEFT_BUTTON_DELAY,
            mouse_press[BUTTON_LEFT]
          )) {
        platf::button_mouse(platf_input, BUTTON_LEFT, true);
      }
    }
    platf::move_mouse(platf_input, util::endian::big(packet->deltaX), util::endian::big(packet->deltaY));
  }

  /**
   * @brief Converts client coordinates on the specified surface into screen coordinates.
   * @param input The input context.
   * @param val The cartesian coordinate pair to convert.
   * @param size The size of the client's surface containing the value.
   * @return The host-relative coordinate pair if a touchport is available.
   */
  std::optional<std::pair<float, float>> client_to_touchport(std::shared_ptr<input_t> &input, const std::pair<float, float> &val, const std::pair<float, float> &size) {
    auto &touch_port_event = input->touch_port_event;
    auto &touch_port = input->touch_port;
    if (touch_port_event->peek()) {
      touch_port = *touch_port_event->pop();
    }
    if (!touch_port) {
      return std::nullopt;
    }

    auto scalarX = touch_port.width / size.first;
    auto scalarY = touch_port.height / size.second;

    float x = std::clamp(val.first, 0.0f, size.first) * scalarX;
    float y = std::clamp(val.second, 0.0f, size.second) * scalarY;

    auto offsetX = touch_port.client_offsetX;
    auto offsetY = touch_port.client_offsetY;

    x = std::clamp(x, offsetX, (size.first * scalarX) - offsetX);
    y = std::clamp(y, offsetY, (size.second * scalarY) - offsetY);

    return std::pair {(x - offsetX) * touch_port.scalar_inv, (y - offsetY) * touch_port.scalar_inv};
  }

  /**
   * @brief Multiply a polar coordinate pair by a cartesian scaling factor.
   * @param r The radial coordinate.
   * @param angle The angular coordinate (radians).
   * @param scalar The scalar cartesian coordinate pair.
   * @return The scaled radial coordinate.
   */
  float multiply_polar_by_cartesian_scalar(float r, float angle, const std::pair<float, float> &scalar) {
    // Convert polar to cartesian coordinates
    float x = r * std::cos(angle);
    float y = r * std::sin(angle);

    // Scale the values
    x *= scalar.first;
    y *= scalar.second;

    // Convert the result back to a polar radial coordinate
    return std::sqrt(std::pow(x, 2) + std::pow(y, 2));
  }

  std::pair<float, float> scale_client_contact_area(const std::pair<float, float> &val, uint16_t rotation, const std::pair<float, float> &scalar) {
    // If the rotation is unknown, we'll just scale both axes equally by using
    // a 45-degree angle for our scaling calculations
    float angle = rotation == LI_ROT_UNKNOWN ? (M_PI / 4) : (rotation * (M_PI / 180));

    // If we have a major but not a minor axis, treat the touch as circular
    float major = val.first;
    float minor = val.second != 0.0f ? val.second : val.first;

    // The minor axis is perpendicular to major axis so the angle must be rotated by 90 degrees
    return {multiply_polar_by_cartesian_scalar(major, angle, scalar), multiply_polar_by_cartesian_scalar(minor, angle + (M_PI / 2), scalar)};
  }

  void passthrough(std::shared_ptr<input_t> &input, PNV_ABS_MOUSE_MOVE_PACKET packet) {
    if (input->mouse_left_button_timeout.load() == DISABLE_LEFT_BUTTON_DELAY) {
      input->mouse_left_button_timeout = ENABLE_LEFT_BUTTON_DELAY;
    }

    float x = util::endian::big(packet->x);
    float y = util::endian::big(packet->y);

    // Prevent divide by zero
    // Don't expect it to happen, but just in case
    if (!packet->width || !packet->height) {
      BOOST_LOG(warning) << "Artemis passed invalid dimensions"sv;

      return;
    }

    auto width = (float) util::endian::big(packet->width);
    auto height = (float) util::endian::big(packet->height);

    auto tpcoords = client_to_touchport(input, {x, y}, {width, height});
    if (!tpcoords) {
      return;
    }

    auto &touch_port = input->touch_port;
    platf::touch_port_t abs_port {
      touch_port.offset_x,
      touch_port.offset_y,
      touch_port.env_width,
      touch_port.env_height
    };

    platf::abs_mouse(platf_input, abs_port, tpcoords->first, tpcoords->second);
  }

  void passthrough(std::shared_ptr<input_t> &input, PNV_MOUSE_BUTTON_PACKET packet) {
    auto release = util::endian::little(packet->header.magic) == MOUSE_BUTTON_UP_EVENT_MAGIC_GEN5;
    auto button = util::endian::big(packet->button);
    if (!detail::valid_mouse_button(button)) {
      return;
    }
    if (mouse_press[button] != release) {
      // button state is already what we want
      return;
    }

    mouse_press[button] = !release;
    /**
     * When Artemis sends mouse input through absolute coordinates,
     * it's possible that BUTTON_RIGHT is pressed down immediately after releasing BUTTON_LEFT.
     * As a result, Sunshine will left-click on hyperlinks in the browser before right-clicking
     *
     * This can be solved by delaying BUTTON_LEFT, however, any delay on input is undesirable during gaming
     * As a compromise, Sunshine will only put delays on BUTTON_LEFT when
     * absolute mouse coordinates have been sent.
     *
     * Try to make sure BUTTON_RIGHT gets called before BUTTON_LEFT is released.
     *
     * input->mouse_left_button_timeout can only be nullptr
     * when the last mouse coordinates were absolute
     */
    if (button == BUTTON_LEFT && release && !input->mouse_left_button_timeout) {
      const auto generation = input->mouse_left_button_generation.fetch_add(1) + 1;
      auto f = [=]() {
        std::lock_guard dispatch_guard {input->input_dispatch_lock};
        // Clear the completed task ID on every callback path, including a left re-press that makes
        // this delayed release obsolete. Keeping it would leave a dangling task pointer.
        auto expected_generation = generation;
        if (!input->mouse_left_button_generation.compare_exchange_strong(
              expected_generation,
              generation + 1
            )) {
          return;
        }
        auto left_released = mouse_press[BUTTON_LEFT];
        input->mouse_left_button_timeout = nullptr;
        if (!detail::delayed_action_needed(left_released)) {
          // Already released left button
          return;
        }
        platf::button_mouse(platf_input, BUTTON_LEFT, release);

        mouse_press[BUTTON_LEFT] = false;
      };

    input->mouse_left_button_timeout = task_pool.pushDelayed(std::move(f), 10ms).task_id;

      return;
    }
    const auto left_release_timer = input->mouse_left_button_timeout.load();
    if (
      button == BUTTON_RIGHT && !release &&
      left_release_timer && left_release_timer != DISABLE_LEFT_BUTTON_DELAY
    ) {
      platf::button_mouse(platf_input, BUTTON_RIGHT, false);
      platf::button_mouse(platf_input, BUTTON_RIGHT, true);

      mouse_press[BUTTON_RIGHT] = false;

      return;
    }

    platf::button_mouse(platf_input, button, release);
  }

  short map_keycode(short keycode) {
    auto it = config::input.keybindings.find(keycode);
    if (it != std::end(config::input.keybindings)) {
      return it->second;
    }

    return keycode;
  }

  /**
   * @brief Update flags for keyboard shortcut combo's
   */
  inline void update_modifier_flags(int *flags, short keyCode, bool release) {
    switch (keyCode) {
      case VKEY_SHIFT:
      case VKEY_LSHIFT:
      case VKEY_RSHIFT:
        if (release) {
          *flags &= ~input_t::SHIFT;
        } else {
          *flags |= input_t::SHIFT;
        }
        break;
      case VKEY_CONTROL:
      case VKEY_LCONTROL:
      case VKEY_RCONTROL:
        if (release) {
          *flags &= ~input_t::CTRL;
        } else {
          *flags |= input_t::CTRL;
        }
        break;
      case VKEY_MENU:
      case VKEY_LMENU:
      case VKEY_RMENU:
        if (release) {
          *flags &= ~input_t::ALT;
        } else {
          *flags |= input_t::ALT;
        }
        break;
    }
  }

  bool is_modifier(uint16_t keyCode) {
    switch (keyCode) {
      case VKEY_SHIFT:
      case VKEY_LSHIFT:
      case VKEY_RSHIFT:
      case VKEY_CONTROL:
      case VKEY_LCONTROL:
      case VKEY_RCONTROL:
      case VKEY_MENU:
      case VKEY_LMENU:
      case VKEY_RMENU:
        return true;
      default:
        return false;
    }
  }

  void send_key_and_modifiers(uint16_t key_code, bool release, uint8_t flags, uint8_t synthetic_modifiers) {
    if (!release) {
      // Press any synthetic modifiers required for this key
      if (synthetic_modifiers & MODIFIER_SHIFT) {
        platf::keyboard_update(platf_input, VKEY_SHIFT, false, flags);
      }
      if (synthetic_modifiers & MODIFIER_CTRL) {
        platf::keyboard_update(platf_input, VKEY_CONTROL, false, flags);
      }
      if (synthetic_modifiers & MODIFIER_ALT) {
        platf::keyboard_update(platf_input, VKEY_MENU, false, flags);
      }
    }

    platf::keyboard_update(platf_input, map_keycode(key_code), release, flags);

    if (!release) {
      // Raise any synthetic modifier keys we pressed
      if (synthetic_modifiers & MODIFIER_SHIFT) {
        platf::keyboard_update(platf_input, VKEY_SHIFT, true, flags);
      }
      if (synthetic_modifiers & MODIFIER_CTRL) {
        platf::keyboard_update(platf_input, VKEY_CONTROL, true, flags);
      }
      if (synthetic_modifiers & MODIFIER_ALT) {
        platf::keyboard_update(platf_input, VKEY_MENU, true, flags);
      }
    }
  }

  void repeat_key(
    std::shared_ptr<input_t> input,
    uint16_t key_code,
    uint8_t flags,
    uint8_t synthetic_modifiers,
    std::uint64_t generation
  ) {
    std::lock_guard dispatch_guard {input->input_dispatch_lock};
    // If key no longer pressed, stop repeating
    if (!detail::key_repeat_is_current(
          input->input_reset,
          generation,
          input->key_repeat_generation,
          key_press[make_kpid(key_code, flags)]
        )) {
      if (generation == input->key_repeat_generation) {
        input->key_repeat_id = nullptr;
      }
      return;
    }

    send_key_and_modifiers(key_code, false, flags, synthetic_modifiers);

    input->key_repeat_id = task_pool.pushDelayed(
      repeat_key,
      config::input.key_repeat_period,
      input,
      key_code,
      flags,
      synthetic_modifiers,
      generation
    ).task_id;
  }

  void passthrough(std::shared_ptr<input_t> &input, PNV_KEYBOARD_PACKET packet) {
    auto release = util::endian::little(packet->header.magic) == KEY_UP_EVENT_MAGIC;
    auto keyCode = packet->keyCode & 0x00FF;

    if (keyCode == VKEY_LMENU) {
      input->left_alt_pressed = !release;
    } else if (keyCode == VKEY_RMENU) {
      input->right_alt_pressed = !release;
    }

    // A right Alt remapped through keybindings must not also synthesize generic Alt on every
    // non-modifier packet. Derive this from the effective mapping.
    auto modifiers = packet->modifiers;
    if (detail::suppress_synthetic_alt(map_keycode(VKEY_RMENU), input->left_alt_pressed, input->right_alt_pressed)) {
      modifiers &= ~MODIFIER_ALT;
    }

    // Set synthetic modifier flags if the keyboard packet is requesting modifier
    // keys that are not current pressed.
    uint8_t synthetic_modifiers = 0;
    if (!release && !is_modifier(keyCode)) {
      if (!(input->pressed_modifiers & input_t::SHIFT) && (modifiers & MODIFIER_SHIFT)) {
        synthetic_modifiers |= MODIFIER_SHIFT;
      }
      if (!(input->pressed_modifiers & input_t::CTRL) && (modifiers & MODIFIER_CTRL)) {
        synthetic_modifiers |= MODIFIER_CTRL;
      }
      if (!(input->pressed_modifiers & input_t::ALT) && (modifiers & MODIFIER_ALT)) {
        synthetic_modifiers |= MODIFIER_ALT;
      }
    }

    auto &pressed = key_press[make_kpid(keyCode, packet->flags)];
    if (!pressed) {
      if (!release) {
        if (input->key_repeat_id) {
          task_pool.cancel(input->key_repeat_id);
          input->key_repeat_id = nullptr;
        }
        const auto repeat_generation = ++input->key_repeat_generation;

        if (config::input.key_repeat_delay.count() > 0) {
          input->key_repeat_id = task_pool.pushDelayed(
            repeat_key,
            config::input.key_repeat_delay,
            input,
            keyCode,
            packet->flags,
            synthetic_modifiers,
            repeat_generation
          ).task_id;
        }
      } else {
        // Already released
        return;
      }
    } else if (!release) {
      // Already pressed down key
      return;
    }

    pressed = !release;

    send_key_and_modifiers(keyCode, release, packet->flags, synthetic_modifiers);

    update_modifier_flags(&input->pressed_modifiers, map_keycode(keyCode), release);
  }

  /**
   * @brief Called to pass a vertical scroll message the platform backend.
   * @param input The input context pointer.
   * @param packet The scroll packet.
   */
  void passthrough(std::shared_ptr<input_t> &input, PNV_SCROLL_PACKET packet) {
    if (config::input.high_resolution_scrolling) {
      platf::scroll(platf_input, util::endian::big(packet->scrollAmt1));
    } else {
      input->accumulated_vscroll_delta += util::endian::big(packet->scrollAmt1);
      auto full_ticks = input->accumulated_vscroll_delta / WHEEL_DELTA;
      if (full_ticks) {
        // Send any full ticks that have accumulated and store the rest
        platf::scroll(platf_input, full_ticks * WHEEL_DELTA);
        input->accumulated_vscroll_delta -= full_ticks * WHEEL_DELTA;
      }
    }
  }

  /**
   * @brief Called to pass a horizontal scroll message the platform backend.
   * @param input The input context pointer.
   * @param packet The scroll packet.
   */
  void passthrough(std::shared_ptr<input_t> &input, PSS_HSCROLL_PACKET packet) {
    if (config::input.high_resolution_scrolling) {
      platf::hscroll(platf_input, util::endian::big(packet->scrollAmount));
    } else {
      input->accumulated_hscroll_delta += util::endian::big(packet->scrollAmount);
      auto full_ticks = input->accumulated_hscroll_delta / WHEEL_DELTA;
      if (full_ticks) {
        // Send any full ticks that have accumulated and store the rest
        platf::hscroll(platf_input, full_ticks * WHEEL_DELTA);
        input->accumulated_hscroll_delta -= full_ticks * WHEEL_DELTA;
      }
    }
  }

  void passthrough(PNV_UNICODE_PACKET packet) {
    auto size = util::endian::big(packet->header.size) - sizeof(packet->header.magic);
    platf::unicode(platf_input, packet->text, size);
  }

  /**
   * @brief Called to pass a controller arrival message to the platform backend.
   * @param input The input context pointer.
   * @param packet The controller arrival packet.
   */
  void passthrough(std::shared_ptr<input_t> &input, PSS_CONTROLLER_ARRIVAL_PACKET packet) {
    if (packet->controllerNumber < 0 || packet->controllerNumber >= input->gamepads.size()) {
      BOOST_LOG(warning) << "ControllerNumber out of range ["sv << packet->controllerNumber << ']';
      return;
    }

    if (input->gamepads[packet->controllerNumber].id >= 0) {
      BOOST_LOG(warning) << "ControllerNumber already allocated ["sv << packet->controllerNumber << ']';
      return;
    }

    platf::gamepad_arrival_t arrival {
      packet->type,
      util::endian::little(packet->capabilities),
      util::endian::little(packet->supportedButtonFlags),
    };

    auto id = alloc_id(gamepadMask);
    if (id < 0) {
      return;
    }

    // Allocate a new gamepad
    if (platf::alloc_gamepad(platf_input, {id, packet->controllerNumber}, arrival, input->feedback_queue)) {
      free_id(gamepadMask, id);
      return;
    }

    input->gamepads[packet->controllerNumber].id = id;
  }

  /**
   * @brief Called to pass a touch message to the platform backend.
   * @param input The input context pointer.
   * @param packet The touch packet.
   */
  void passthrough(std::shared_ptr<input_t> &input, PSS_TOUCH_PACKET packet) {
    // Convert the client normalized coordinates to touchport coordinates
    auto coords = client_to_touchport(input, {from_clamped_netfloat(packet->x, 0.0f, 1.0f) * 65535.f, from_clamped_netfloat(packet->y, 0.0f, 1.0f) * 65535.f}, {65535.f, 65535.f});
    if (!coords) {
      return;
    }

    auto &touch_port = input->touch_port;
    platf::touch_port_t abs_port {
      touch_port.offset_x,
      touch_port.offset_y,
      touch_port.env_width,
      touch_port.env_height
    };

    // Renormalize the coordinates
    coords->first /= abs_port.width;
    coords->second /= abs_port.height;

    // Normalize rotation value to 0-359 degree range
    auto rotation = util::endian::little(packet->rotation);
    if (rotation != LI_ROT_UNKNOWN) {
      rotation %= 360;
    }

    // Normalize the contact area based on the touchport
    auto contact_area = scale_client_contact_area(
      {from_clamped_netfloat(packet->contactAreaMajor, 0.0f, 1.0f) * 65535.f,
       from_clamped_netfloat(packet->contactAreaMinor, 0.0f, 1.0f) * 65535.f},
      rotation,
      {abs_port.width / 65535.f, abs_port.height / 65535.f}
    );

    platf::touch_input_t touch {
      packet->eventType,
      rotation,
      util::endian::little(packet->pointerId),
      coords->first,
      coords->second,
      from_clamped_netfloat(packet->pressureOrDistance, 0.0f, 1.0f),
      contact_area.first,
      contact_area.second,
    };

    platf::touch_update(input->client_context.get(), abs_port, touch);
  }

  /**
   * @brief Called to pass a pen message to the platform backend.
   * @param input The input context pointer.
   * @param packet The pen packet.
   */
  void passthrough(std::shared_ptr<input_t> &input, PSS_PEN_PACKET packet) {
    // Convert the client normalized coordinates to touchport coordinates
    auto coords = client_to_touchport(input, {from_clamped_netfloat(packet->x, 0.0f, 1.0f) * 65535.f, from_clamped_netfloat(packet->y, 0.0f, 1.0f) * 65535.f}, {65535.f, 65535.f});
    if (!coords) {
      return;
    }

    auto &touch_port = input->touch_port;
    platf::touch_port_t abs_port {
      touch_port.offset_x,
      touch_port.offset_y,
      touch_port.env_width,
      touch_port.env_height
    };

    // Renormalize the coordinates
    coords->first /= abs_port.width;
    coords->second /= abs_port.height;

    // Normalize rotation value to 0-359 degree range
    auto rotation = util::endian::little(packet->rotation);
    if (rotation != LI_ROT_UNKNOWN) {
      rotation %= 360;
    }

    // Normalize the contact area based on the touchport
    auto contact_area = scale_client_contact_area(
      {from_clamped_netfloat(packet->contactAreaMajor, 0.0f, 1.0f) * 65535.f,
       from_clamped_netfloat(packet->contactAreaMinor, 0.0f, 1.0f) * 65535.f},
      rotation,
      {abs_port.width / 65535.f, abs_port.height / 65535.f}
    );

    platf::pen_input_t pen {
      packet->eventType,
      packet->toolType,
      packet->penButtons,
      packet->tilt,
      rotation,
      coords->first,
      coords->second,
      from_clamped_netfloat(packet->pressureOrDistance, 0.0f, 1.0f),
      contact_area.first,
      contact_area.second,
    };

    platf::pen_update(input->client_context.get(), abs_port, pen);
  }

  /**
   * @brief Called to pass a controller touch message to the platform backend.
   * @param input The input context pointer.
   * @param packet The controller touch packet.
   */
  void passthrough(std::shared_ptr<input_t> &input, PSS_CONTROLLER_TOUCH_PACKET packet) {
    if (packet->controllerNumber < 0 || packet->controllerNumber >= input->gamepads.size()) {
      BOOST_LOG(warning) << "ControllerNumber out of range ["sv << packet->controllerNumber << ']';
      return;
    }

    auto &gamepad = input->gamepads[packet->controllerNumber];
    if (gamepad.id < 0) {
      BOOST_LOG(warning) << "ControllerNumber ["sv << packet->controllerNumber << "] not allocated"sv;
      return;
    }

    platf::gamepad_touch_t touch {
      {gamepad.id, packet->controllerNumber},
      packet->eventType,
      util::endian::little(packet->pointerId),
      from_clamped_netfloat(packet->x, 0.0f, 1.0f),
      from_clamped_netfloat(packet->y, 0.0f, 1.0f),
      from_clamped_netfloat(packet->pressure, 0.0f, 1.0f),
    };

    platf::gamepad_touch(platf_input, touch);
  }

  /**
   * @brief Called to pass a controller motion message to the platform backend.
   * @param input The input context pointer.
   * @param packet The controller motion packet.
   */
  void passthrough(std::shared_ptr<input_t> &input, PSS_CONTROLLER_MOTION_PACKET packet) {
    if (packet->controllerNumber < 0 || packet->controllerNumber >= input->gamepads.size()) {
      BOOST_LOG(warning) << "ControllerNumber out of range ["sv << packet->controllerNumber << ']';
      return;
    }

    auto &gamepad = input->gamepads[packet->controllerNumber];
    if (gamepad.id < 0) {
      BOOST_LOG(warning) << "ControllerNumber ["sv << packet->controllerNumber << "] not allocated"sv;
      return;
    }

    platf::gamepad_motion_t motion {
      {gamepad.id, packet->controllerNumber},
      packet->motionType,
      from_netfloat(packet->x),
      from_netfloat(packet->y),
      from_netfloat(packet->z),
    };

    platf::gamepad_motion(platf_input, motion);
  }

  /**
   * @brief Called to pass a controller battery message to the platform backend.
   * @param input The input context pointer.
   * @param packet The controller battery packet.
   */
  void passthrough(std::shared_ptr<input_t> &input, PSS_CONTROLLER_BATTERY_PACKET packet) {
    if (packet->controllerNumber < 0 || packet->controllerNumber >= input->gamepads.size()) {
      BOOST_LOG(warning) << "ControllerNumber out of range ["sv << packet->controllerNumber << ']';
      return;
    }

    auto &gamepad = input->gamepads[packet->controllerNumber];
    if (gamepad.id < 0) {
      BOOST_LOG(warning) << "ControllerNumber ["sv << packet->controllerNumber << "] not allocated"sv;
      return;
    }

    platf::gamepad_battery_t battery {
      {gamepad.id, packet->controllerNumber},
      packet->batteryState,
      packet->batteryPercentage
    };

    platf::gamepad_battery(platf_input, battery);
  }

  void passthrough(std::shared_ptr<input_t> &input, PNV_MULTI_CONTROLLER_PACKET packet) {
    if (packet->controllerNumber < 0 || packet->controllerNumber >= input->gamepads.size()) {
      BOOST_LOG(warning) << "ControllerNumber out of range ["sv << packet->controllerNumber << ']';

      return;
    }

    auto &gamepad = input->gamepads[packet->controllerNumber];

    // Modern Artemis announces each controller before sending state. The state packet remains
    // authoritative for removal, but it must not silently synthesize an unannounced device.
    if (!(packet->activeGamepadMask & (1 << packet->controllerNumber)) && gamepad.id >= 0) {
      // If this is the final event for a gamepad being removed, free the gamepad and return.
      const auto back_timer = gamepad.back_timeout_id;
      const auto home_timer = gamepad.home_release_id;
      gamepad.back_timeout_id = nullptr;
      gamepad.home_release_id = nullptr;
      ++gamepad.back_action_generation;
      ++gamepad.home_action_generation;
      gamepad.gamepad_state.buttonFlags &= ~(platf::BACK | platf::HOME);
      gamepad.back_button_state = button_state_e::NONE;
      free_gamepad(platf_input, gamepad.id);
      gamepad.id = -1;
      task_pool.cancel(back_timer);
      task_pool.cancel(home_timer);
      return;
    }

    // If this gamepad has not been initialized, ignore it.
    // This could happen when platf::alloc_gamepad fails
    if (gamepad.id < 0) {
      BOOST_LOG(warning) << "ControllerNumber ["sv << packet->controllerNumber << "] not allocated"sv;
      return;
    }

    std::uint16_t bf = packet->buttonFlags;
    std::uint32_t bf2 = packet->buttonFlags2;
    platf::gamepad_state_t gamepad_state {
      bf | (bf2 << 16),
      packet->leftTrigger,
      packet->rightTrigger,
      packet->leftStickX,
      packet->leftStickY,
      packet->rightStickX,
      packet->rightStickY
    };

    auto bf_new = gamepad_state.buttonFlags;
    switch (gamepad.back_button_state) {
      case button_state_e::UP:
        if (!(platf::BACK & bf_new)) {
          gamepad.back_button_state = button_state_e::NONE;
        }
        gamepad_state.buttonFlags &= ~platf::BACK;
        break;
      case button_state_e::DOWN:
        if (platf::BACK & bf_new) {
          gamepad.back_button_state = button_state_e::NONE;
        }
        gamepad_state.buttonFlags |= platf::BACK;
        break;
      case button_state_e::NONE:
        break;
    }

    bf = gamepad_state.buttonFlags ^ gamepad.gamepad_state.buttonFlags;
    bf_new = gamepad_state.buttonFlags;

    if (platf::BACK & bf) {
      if (platf::BACK & bf_new) {
        // Don't emulate home button if timeout < 0
        if (config::input.back_button_timeout >= 0ms) {
          const auto controller = packet->controllerNumber;
          const auto back_generation = ++gamepad.back_action_generation;
          auto f = [input, controller, back_generation]() {
            std::lock_guard dispatch_guard {input->input_dispatch_lock};
            auto &gamepad = input->gamepads[controller];
            if (!detail::controller_action_is_current(
                  input->input_reset,
                  gamepad.id,
                  back_generation,
                  gamepad.back_action_generation
                )) {
              return;
            }
            gamepad.back_timeout_id = nullptr;

            auto &state = gamepad.gamepad_state;

            // Force the back button up
            gamepad.back_button_state = button_state_e::UP;
            state.buttonFlags &= ~platf::BACK;
            platf::gamepad_update(platf_input, gamepad.id, state);

            // Press Home button
            state.buttonFlags |= platf::HOME;
            platf::gamepad_update(platf_input, gamepad.id, state);

            const auto home_generation = ++gamepad.home_action_generation;
            auto release_home = [input, controller, home_generation]() {
              std::lock_guard dispatch_guard {input->input_dispatch_lock};
              auto &gamepad = input->gamepads[controller];
              if (!detail::controller_action_is_current(
                    input->input_reset,
                    gamepad.id,
                    home_generation,
                    gamepad.home_action_generation
                  )) {
                return;
              }
              gamepad.home_release_id = nullptr;
              gamepad.gamepad_state.buttonFlags &= ~platf::HOME;
              platf::gamepad_update(platf_input, gamepad.id, gamepad.gamepad_state);
            };
            gamepad.home_release_id = task_pool.pushDelayed(std::move(release_home), 100ms).task_id;
          };

          gamepad.back_timeout_id = task_pool.pushDelayed(std::move(f), config::input.back_button_timeout).task_id;
        }
      } else if (gamepad.back_timeout_id || gamepad.home_release_id) {
        task_pool.cancel(gamepad.back_timeout_id);
        task_pool.cancel(gamepad.home_release_id);
        gamepad.back_timeout_id = nullptr;
        gamepad.home_release_id = nullptr;
        ++gamepad.back_action_generation;
        ++gamepad.home_action_generation;
        gamepad.gamepad_state.buttonFlags &= ~platf::HOME;
      }
    }

    gamepad_state.buttonFlags = detail::latch_button_while_active(
      gamepad_state.buttonFlags,
      static_cast<decltype(gamepad_state.buttonFlags)>(platf::HOME),
      gamepad.home_release_id != nullptr
    );

    platf::gamepad_update(platf_input, gamepad.id, gamepad_state);

    gamepad.gamepad_state = gamepad_state;
  }

  using detail::batch_result_e;

  /**
   * @brief Batch two relative mouse messages.
   * @param dest The original packet to batch into.
   * @param src A later packet to attempt to batch.
   * @return The status of the batching operation.
   */
  batch_result_e batch(NV_REL_MOUSE_MOVE_PACKET *dest, const NV_REL_MOUSE_MOVE_PACKET *src) {
    // Batching is safe as long as the result doesn't overflow a 16-bit integer
    const auto deltaX = detail::checked_add_i16(
      util::endian::big(dest->deltaX),
      util::endian::big(src->deltaX)
    );
    const auto deltaY = detail::checked_add_i16(
      util::endian::big(dest->deltaY),
      util::endian::big(src->deltaY)
    );
    if (!deltaX || !deltaY) {
      return batch_result_e::terminate_batch;
    }

    // Take the sum of deltas
    dest->deltaX = util::endian::big(*deltaX);
    dest->deltaY = util::endian::big(*deltaY);
    return batch_result_e::batched;
  }

  /**
   * @brief Batch two absolute mouse messages.
   * @param dest The original packet to batch into.
   * @param src A later packet to attempt to batch.
   * @return The status of the batching operation.
   */
  batch_result_e batch(NV_ABS_MOUSE_MOVE_PACKET *dest, const NV_ABS_MOUSE_MOVE_PACKET *src) {
    // Batching must only happen if the reference width and height don't change
    if (dest->width != src->width || dest->height != src->height) {
      return batch_result_e::terminate_batch;
    }

    // Take the latest absolute position
    *dest = *src;
    return batch_result_e::batched;
  }

  /**
   * @brief Batch two vertical scroll messages.
   * @param dest The original packet to batch into.
   * @param src A later packet to attempt to batch.
   * @return The status of the batching operation.
   */
  batch_result_e batch(NV_SCROLL_PACKET *dest, const NV_SCROLL_PACKET *src) {
    // Batching is safe as long as the result doesn't overflow a 16-bit integer
    const auto scrollAmt = detail::checked_add_i16(
      util::endian::big(dest->scrollAmt1),
      util::endian::big(src->scrollAmt1)
    );
    if (!scrollAmt) {
      return batch_result_e::terminate_batch;
    }

    // Take the sum of delta
    dest->scrollAmt1 = util::endian::big(*scrollAmt);
    dest->scrollAmt2 = util::endian::big(*scrollAmt);
    return batch_result_e::batched;
  }

  /**
   * @brief Batch two horizontal scroll messages.
   * @param dest The original packet to batch into.
   * @param src A later packet to attempt to batch.
   * @return The status of the batching operation.
   */
  batch_result_e batch(SS_HSCROLL_PACKET *dest, const SS_HSCROLL_PACKET *src) {
    // Batching is safe as long as the result doesn't overflow a 16-bit integer
    const auto scrollAmt = detail::checked_add_i16(
      util::endian::big(dest->scrollAmount),
      util::endian::big(src->scrollAmount)
    );
    if (!scrollAmt) {
      return batch_result_e::terminate_batch;
    }

    // Take the sum of delta
    dest->scrollAmount = util::endian::big(*scrollAmt);
    return batch_result_e::batched;
  }

  /**
   * @brief Batch two controller state messages.
   * @param dest The original packet to batch into.
   * @param src A later packet to attempt to batch.
   * @return The status of the batching operation.
   */
  batch_result_e batch(NV_MULTI_CONTROLLER_PACKET *dest, const NV_MULTI_CONTROLLER_PACKET *src) {
    // Do not allow batching if the active controllers change
    if (dest->activeGamepadMask != src->activeGamepadMask) {
      return batch_result_e::terminate_batch;
    }

    // We can only batch entries for the same controller, but allow batching attempts to continue
    // in case we have more packets for this controller later in the queue.
    if (dest->controllerNumber != src->controllerNumber) {
      return batch_result_e::not_batchable;
    }

    // Do not allow batching if the button state changes on this controller
    if (dest->buttonFlags != src->buttonFlags || dest->buttonFlags2 != src->buttonFlags2) {
      return batch_result_e::terminate_batch;
    }

    // Take the latest state
    *dest = *src;
    return batch_result_e::batched;
  }

  /**
   * @brief Batch two touch messages.
   * @param dest The original packet to batch into.
   * @param src A later packet to attempt to batch.
   * @return The status of the batching operation.
   */
  batch_result_e batch(SS_TOUCH_PACKET *dest, const SS_TOUCH_PACKET *src) {
    // Only batch hover or move events
    if (dest->eventType != LI_TOUCH_EVENT_MOVE && dest->eventType != LI_TOUCH_EVENT_HOVER) {
      return batch_result_e::terminate_batch;
    }

    // Don't batch beyond state changing events
    if (src->eventType != LI_TOUCH_EVENT_MOVE && src->eventType != LI_TOUCH_EVENT_HOVER) {
      return batch_result_e::terminate_batch;
    }

    // Batched events must be the same pointer ID
    if (dest->pointerId != src->pointerId) {
      return batch_result_e::not_batchable;
    }

    // The pointer must be in the same state
    if (dest->eventType != src->eventType) {
      return batch_result_e::terminate_batch;
    }

    // Take the latest state
    *dest = *src;
    return batch_result_e::batched;
  }

  /**
   * @brief Batch two pen messages.
   * @param dest The original packet to batch into.
   * @param src A later packet to attempt to batch.
   * @return The status of the batching operation.
   */
  batch_result_e batch(SS_PEN_PACKET *dest, const SS_PEN_PACKET *src) {
    // Only batch hover or move events
    if (dest->eventType != LI_TOUCH_EVENT_MOVE && dest->eventType != LI_TOUCH_EVENT_HOVER) {
      return batch_result_e::terminate_batch;
    }

    // Batched events must be the same type
    if (dest->eventType != src->eventType) {
      return batch_result_e::terminate_batch;
    }

    // Do not allow batching if the button state changes
    if (dest->penButtons != src->penButtons) {
      return batch_result_e::terminate_batch;
    }

    // Do not batch beyond tool changes
    if (dest->toolType != src->toolType) {
      return batch_result_e::terminate_batch;
    }

    // Take the latest state
    *dest = *src;
    return batch_result_e::batched;
  }

  /**
   * @brief Batch two controller touch messages.
   * @param dest The original packet to batch into.
   * @param src A later packet to attempt to batch.
   * @return The status of the batching operation.
   */
  batch_result_e batch(SS_CONTROLLER_TOUCH_PACKET *dest, const SS_CONTROLLER_TOUCH_PACKET *src) {
    // Only batch hover or move events
    if (dest->eventType != LI_TOUCH_EVENT_MOVE && dest->eventType != LI_TOUCH_EVENT_HOVER) {
      return batch_result_e::terminate_batch;
    }

    // We can only batch entries for the same controller, but allow batching attempts to continue
    // in case we have more packets for this controller later in the queue.
    if (dest->controllerNumber != src->controllerNumber) {
      return batch_result_e::not_batchable;
    }

    // Don't batch beyond state changing events
    if (src->eventType != LI_TOUCH_EVENT_MOVE && src->eventType != LI_TOUCH_EVENT_HOVER) {
      return batch_result_e::terminate_batch;
    }

    // Batched events must be the same pointer ID
    if (dest->pointerId != src->pointerId) {
      return batch_result_e::not_batchable;
    }

    // The pointer must be in the same state
    if (dest->eventType != src->eventType) {
      return batch_result_e::terminate_batch;
    }

    // Take the latest state
    *dest = *src;
    return batch_result_e::batched;
  }

  /**
   * @brief Batch two controller motion messages.
   * @param dest The original packet to batch into.
   * @param src A later packet to attempt to batch.
   * @return The status of the batching operation.
   */
  batch_result_e batch(SS_CONTROLLER_MOTION_PACKET *dest, const SS_CONTROLLER_MOTION_PACKET *src) {
    // We can only batch entries for the same controller, but allow batching attempts to continue
    // in case we have more packets for this controller later in the queue.
    if (dest->controllerNumber != src->controllerNumber) {
      return batch_result_e::not_batchable;
    }

    // Batched events must be the same sensor
    if (dest->motionType != src->motionType) {
      return batch_result_e::not_batchable;
    }

    // Take the latest state
    *dest = *src;
    return batch_result_e::batched;
  }

  /**
   * @brief Batch two input messages.
   * @param dest The original packet to batch into.
   * @param src A later packet to attempt to batch.
   * @return The status of the batching operation.
   */
  batch_result_e batch(NV_INPUT_HEADER *dest, const NV_INPUT_HEADER *src) {
    // We can only batch if the packet types are the same
    if (dest->magic != src->magic) {
      return batch_result_e::terminate_batch;
    }

    // We can only batch certain message types
    switch (util::endian::little(dest->magic)) {
      case MOUSE_MOVE_REL_MAGIC_GEN5:
        return batch((PNV_REL_MOUSE_MOVE_PACKET) dest, (const NV_REL_MOUSE_MOVE_PACKET *) src);
      case MOUSE_MOVE_ABS_MAGIC:
        return batch((PNV_ABS_MOUSE_MOVE_PACKET) dest, (const NV_ABS_MOUSE_MOVE_PACKET *) src);
      case SCROLL_MAGIC_GEN5:
        return batch((PNV_SCROLL_PACKET) dest, (const NV_SCROLL_PACKET *) src);
      case SS_HSCROLL_MAGIC:
        return batch((PSS_HSCROLL_PACKET) dest, (const SS_HSCROLL_PACKET *) src);
      case MULTI_CONTROLLER_MAGIC_GEN5:
        return batch((PNV_MULTI_CONTROLLER_PACKET) dest, (const NV_MULTI_CONTROLLER_PACKET *) src);
      case SS_TOUCH_MAGIC:
        return batch((PSS_TOUCH_PACKET) dest, (const SS_TOUCH_PACKET *) src);
      case SS_PEN_MAGIC:
        return batch((PSS_PEN_PACKET) dest, (const SS_PEN_PACKET *) src);
      case SS_CONTROLLER_TOUCH_MAGIC:
        return batch((PSS_CONTROLLER_TOUCH_PACKET) dest, (const SS_CONTROLLER_TOUCH_PACKET *) src);
      case SS_CONTROLLER_MOTION_MAGIC:
        return batch((PSS_CONTROLLER_MOTION_PACKET) dest, (const SS_CONTROLLER_MOTION_PACKET *) src);
      default:
        // Not a batchable message type
        return batch_result_e::terminate_batch;
    }
  }

  batch_result_e detail::batch_packets(
    std::span<std::uint8_t> destination,
    std::span<const std::uint8_t> source
  ) noexcept {
    if (!validated_packet_magic(destination) || !validated_packet_magic(source)) {
      return batch_result_e::terminate_batch;
    }

    return batch(
      reinterpret_cast<NV_INPUT_HEADER *>(destination.data()),
      reinterpret_cast<const NV_INPUT_HEADER *>(source.data())
    );
  }

  std::optional<std::vector<std::uint8_t>> detail::pop_next_batched_packet(
    std::list<std::vector<std::uint8_t>> &packets
  ) {
    if (packets.empty()) {
      return std::nullopt;
    }

    auto entry = std::move(packets.front());
    packets.pop_front();

    // Empty entries are reset barriers. They deliberately stop packet coalescing and preserve
    // reset ordering relative to input arriving before and after the barrier.
    if (entry.empty()) {
      return entry;
    }

    auto candidate = packets.begin();
    while (candidate != packets.end() && !candidate->empty()) {
      const auto result = batch_packets(entry, *candidate);
      if (result == batch_result_e::terminate_batch) {
        break;
      }
      if (result == batch_result_e::batched) {
        candidate = packets.erase(candidate);
      } else {
        ++candidate;
      }
    }
    return entry;
  }

  /**
   * @brief Called on a thread pool thread to process an input message.
   * @param input The input context pointer.
   */
  void dispatch_input_packet(std::shared_ptr<input_t> &input, std::vector<uint8_t> &entry) {
    auto payload = (PNV_INPUT_HEADER) entry.data();
    switch (util::endian::little(payload->magic)) {
      case MOUSE_MOVE_REL_MAGIC_GEN5:
        passthrough(input, (PNV_REL_MOUSE_MOVE_PACKET) payload);
        break;
      case MOUSE_MOVE_ABS_MAGIC:
        passthrough(input, (PNV_ABS_MOUSE_MOVE_PACKET) payload);
        break;
      case MOUSE_BUTTON_DOWN_EVENT_MAGIC_GEN5:
      case MOUSE_BUTTON_UP_EVENT_MAGIC_GEN5:
        passthrough(input, (PNV_MOUSE_BUTTON_PACKET) payload);
        break;
      case SCROLL_MAGIC_GEN5:
        passthrough(input, (PNV_SCROLL_PACKET) payload);
        break;
      case SS_HSCROLL_MAGIC:
        passthrough(input, (PSS_HSCROLL_PACKET) payload);
        break;
      case KEY_DOWN_EVENT_MAGIC:
      case KEY_UP_EVENT_MAGIC:
        passthrough(input, (PNV_KEYBOARD_PACKET) payload);
        break;
      case UTF8_TEXT_EVENT_MAGIC:
        passthrough((PNV_UNICODE_PACKET) payload);
        break;
      case MULTI_CONTROLLER_MAGIC_GEN5:
        passthrough(input, (PNV_MULTI_CONTROLLER_PACKET) payload);
        break;
      case SS_TOUCH_MAGIC:
        passthrough(input, (PSS_TOUCH_PACKET) payload);
        break;
      case SS_PEN_MAGIC:
        passthrough(input, (PSS_PEN_PACKET) payload);
        break;
      case SS_CONTROLLER_ARRIVAL_MAGIC:
        passthrough(input, (PSS_CONTROLLER_ARRIVAL_PACKET) payload);
        break;
      case SS_CONTROLLER_TOUCH_MAGIC:
        passthrough(input, (PSS_CONTROLLER_TOUCH_PACKET) payload);
        break;
      case SS_CONTROLLER_MOTION_MAGIC:
        passthrough(input, (PSS_CONTROLLER_MOTION_PACKET) payload);
        break;
      case SS_CONTROLLER_BATTERY_MAGIC:
        passthrough(input, (PSS_CONTROLLER_BATTERY_PACKET) payload);
        break;
    }
  }

  void release_all_input_state() {
    for (int x = 0; x < mouse_press.size(); ++x) {
      if (mouse_press[x]) {
        platf::button_mouse(platf_input, x, true);
        mouse_press[x] = false;
      }
    }

    for (auto &kp : key_press) {
      if (!kp.second) {
        continue;
      }
      platf::keyboard_update(platf_input, map_keycode(vk_from_kpid(kp.first) & 0x00FF), true, flags_from_kpid(kp.first));
      key_press[kp.first] = false;
    }
  }

  /**
   * @brief Own and drain one input context's queue on the global worker.
   * @details The gate transition to idle happens while holding the queue lock, so an arrival
   * racing the end of a drain either remains owned by this drain or schedules the next one.
   */
  void drain_input_queue(std::shared_ptr<input_t> input) {
    std::size_t dispatched = 0;
    while (!detail::drain_turn_exhausted(dispatched)) {
      std::optional<std::vector<uint8_t>> entry;
      std::uint64_t generation;
      {
        std::lock_guard<std::mutex> lg(input->input_queue_lock);
        entry = detail::pop_next_batched_packet(input->input_queue);
        if (!entry) {
          detail::release_if_empty(input->input_drain_gate, true);
          return;
        }
        generation = input->input_generation;
      }

      if (entry->empty()) {
        std::lock_guard dispatch_guard {input->input_dispatch_lock};
        release_all_input_state();
        // Complete virtual-controller teardown before allowing the sole active-session slot to be
        // released. gamepad_t destructors then see id == -1 and do not enqueue stale frees.
        detail::free_gamepads_before_completion(
          input->gamepads,
          [&](int id) {
            free_gamepad(platf_input, id);
          },
          [&]() {
            if (input->reset_completion) {
              input->reset_completion->set_value();
              input->reset_completion.reset();
            }
          }
        );
      } else {
        detail::dispatch_if_current(
          input->input_dispatch_lock,
          generation,
          input->input_generation,
          [&]() {
          dispatch_input_packet(input, *entry);
          }
        );
      }
      ++dispatched;
    }

    // Retain gate ownership across exactly one continuation. Arrivals during the handoff see an
    // active owner and need no task of their own; the continuation either consumes them or makes
    // the empty-to-idle transition under the queue lock.
    task_pool.pushDelayed(drain_input_queue, 0ms, input);
  }

  void schedule_input_drain_if_needed(std::shared_ptr<input_t> &input) {
    bool schedule_drain;
    {
      std::lock_guard<std::mutex> lg(input->input_queue_lock);
      schedule_drain = input->input_drain_gate.request();
    }
    if (schedule_drain) {
      task_pool.push(drain_input_queue, input);
    }
  }

  /**
   * @brief Called on the control stream thread to queue an input message.
   * @param input The input context pointer.
   * @param input_data The input message.
   */
  void passthrough(std::shared_ptr<input_t> &input, std::vector<std::uint8_t> &&input_data, const crypto::PERM &permission) {
    // No input permissions at all
    if (!(permission & crypto::PERM::_all_inputs)) {
      return;
    }

    const auto magic = validated_packet_magic(input_data);
    if (!magic) {
      BOOST_LOG(warning) << "Dropping malformed or unsupported input packet of "sv << input_data.size() << " bytes"sv;
      return;
    }

    // Artemis sends this capability handshake when input starts. Apollo does not need to
    // act on it, but accepting it avoids treating every valid client connection as malformed.
    if (*magic == ENABLE_HAPTICS_MAGIC) {
      return;
    }

    // Have some input permission
    // Otherwise have all input permission
    if ((permission & crypto::PERM::_all_inputs) != crypto::PERM::_all_inputs) {
      // Check permission
      switch (*magic) {
        case MULTI_CONTROLLER_MAGIC_GEN5:
        case SS_CONTROLLER_ARRIVAL_MAGIC:
        case SS_CONTROLLER_TOUCH_MAGIC:
        case SS_CONTROLLER_MOTION_MAGIC:
        case SS_CONTROLLER_BATTERY_MAGIC:
          if (!(permission & crypto::PERM::input_controller)) {
            return;
          } else {
            break;
          }
        case MOUSE_MOVE_REL_MAGIC_GEN5:
        case MOUSE_MOVE_ABS_MAGIC:
        case MOUSE_BUTTON_DOWN_EVENT_MAGIC_GEN5:
        case MOUSE_BUTTON_UP_EVENT_MAGIC_GEN5:
        case SCROLL_MAGIC_GEN5:
        case SS_HSCROLL_MAGIC:
          if (!(permission & crypto::PERM::input_mouse)) {
            return;
          } else {
            break;
          }
        case KEY_DOWN_EVENT_MAGIC:
        case KEY_UP_EVENT_MAGIC:
        case UTF8_TEXT_EVENT_MAGIC:
          if (!(permission & crypto::PERM::input_kbd)) {
            return;
          } else {
            break;
          }
        case SS_TOUCH_MAGIC:
          if (!(permission & crypto::PERM::input_touch)) {
            return;
          } else {
            break;
          }
        case SS_PEN_MAGIC:
          if (!(permission & crypto::PERM::input_pen)) {
            return;
          } else {
            break;
          }
        default:
          // Unknown input event
          return;
      }
    }

    bool schedule_drain;
    {
      std::lock_guard dispatch_guard {input->input_dispatch_lock};
      const auto admitted = detail::admit_if_live(input->input_reset, [&]() {
        std::lock_guard<std::mutex> lg(input->input_queue_lock);
        input->input_queue.push_back(std::move(input_data));
        schedule_drain = input->input_drain_gate.request();
      });
      if (!admitted) {
        return;
      }
    }
    if (schedule_drain) {
      task_pool.push(drain_input_queue, input);
    }
  }

  std::future<void> reset(std::shared_ptr<input_t> &input) {
    auto completion = std::make_shared<std::promise<void>>();
    auto future = completion->get_future();
    {
      std::lock_guard dispatch_guard {input->input_dispatch_lock};
      input->input_reset = true;
      input->reset_completion = completion;
      task_pool.cancel(input->key_repeat_id);
      input->key_repeat_id = nullptr;
      ++input->key_repeat_generation;
      ++input->mouse_left_button_generation;
      const auto pending_left_release = input->mouse_left_button_timeout.exchange(nullptr);
      // The mouse callback uses input_dispatch_lock before clearing its raw ID, so cancelling here
      // cannot hit a destroyed/reused task address and cannot race a post-reset publication.
      task_pool.cancel(pending_left_release);
      for (auto &gamepad : input->gamepads) {
        // The callbacks take input_dispatch_lock before clearing their raw task IDs. Cancel while
        // holding that lock so an executed task cannot be destroyed/reused between snapshot and
        // cancellation (the task pool never holds its mutex while running a callback).
        task_pool.cancel(gamepad.back_timeout_id);
        task_pool.cancel(gamepad.home_release_id);
        gamepad.back_timeout_id = nullptr;
        gamepad.home_release_id = nullptr;
        ++gamepad.back_action_generation;
        ++gamepad.home_action_generation;
        gamepad.gamepad_state.buttonFlags &= ~platf::HOME;
      }
      // An empty queue entry is an ordered reset barrier. Invalidate the current drain generation,
      // clear queued input from the ended session, then enqueue exactly one release operation. A
      // stale drain that is already in flight observes the generation change before dispatching any
      // packet it removed before reset. Input arriving after reset remains behind the barrier.
      std::lock_guard queue_guard {input->input_queue_lock};
      ++input->input_generation;
      input->input_queue.clear();
      input->input_queue.emplace_back();
      if (detail::should_flush_pending_left_release(
            pending_left_release,
            DISABLE_LEFT_BUTTON_DELAY,
            mouse_press[BUTTON_LEFT]
          )) {
        // The logical LEFT state is already up, so release_all_input_state() cannot discover this
        // outstanding delayed OS release. Flush it into the ordered reset critical section.
        platf::button_mouse(platf_input, BUTTON_LEFT, true);
      }
    }
    schedule_input_drain_if_needed(input);
    return future;
  }

  class deinit_t: public platf::deinit_t {
  public:
    ~deinit_t() override {
      platf_input.reset();
    }
  };

  [[nodiscard]] std::unique_ptr<platf::deinit_t> init() {
    platf_input = platf::input();

    return std::make_unique<deinit_t>();
  }

  bool probe_gamepads() {
    auto input = static_cast<platf::input_t *>(platf_input.get());
    const auto gamepads = platf::supported_gamepads(input);
    for (auto &gamepad : gamepads) {
      if (gamepad.is_enabled && gamepad.name != "auto") {
        return false;
      }
    }
    return true;
  }

  std::shared_ptr<input_t> alloc(safe::mail_t mail) {
    auto input = std::make_shared<input_t>(
      mail->event<input::touch_port_t>(mail::touch_port),
      mail->queue<platf::gamepad_feedback_msg_t>(mail::gamepad_feedback)
    );

    // Workaround to ensure new frames will be captured when a client connects
    task_pool.pushDelayed([]() {
      platf::move_mouse(platf_input, 1, 1);
      platf::move_mouse(platf_input, -1, -1);
    },
                          100ms);

    return input;
  }
}  // namespace input
