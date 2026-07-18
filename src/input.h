/**
 * @file src/input.h
 * @brief Declarations for gamepad, keyboard, and mouse input handling.
 */
#pragma once

// standard includes
#include <functional>

// local includes
#include "platform/common.h"
#include "thread_safe.h"
#include "crypto.h"

namespace input {
  struct input_t;

  namespace detail {
    /**
     * @brief Determine whether a held, remapped right Alt should be removed from packet modifiers.
     * @param mapped_right_alt Effective virtual-key mapping for right Alt.
     * @param left_alt_pressed Whether physical left Alt is currently held.
     * @param right_alt_pressed Whether physical right Alt is currently held.
     */
    constexpr bool suppress_synthetic_alt(uint16_t mapped_right_alt, bool left_alt_pressed, bool right_alt_pressed) noexcept {
      constexpr uint16_t VKEY_LWIN = 0x5B;
      constexpr uint16_t VKEY_RWIN = 0x5C;
      const bool right_alt_maps_to_meta = mapped_right_alt == VKEY_LWIN || mapped_right_alt == VKEY_RWIN;
      return right_alt_maps_to_meta && right_alt_pressed && !left_alt_pressed;
    }
  }  // namespace detail

  void print(void *input);
  void reset(std::shared_ptr<input_t> &input);
  void passthrough(std::shared_ptr<input_t> &input, std::vector<std::uint8_t> &&input_data, const crypto::PERM& permission);

  [[nodiscard]] std::unique_ptr<platf::deinit_t> init();

  bool probe_gamepads();

  std::shared_ptr<input_t> alloc(safe::mail_t mail);

  struct touch_port_t: public platf::touch_port_t {
    int env_width, env_height;

    // Offset x and y coordinates of the client
    float client_offsetX, client_offsetY;

    float scalar_inv;

    explicit operator bool() const {
      return width != 0 && height != 0 && env_width != 0 && env_height != 0;
    }
  };

  /**
   * @brief Scale the ellipse axes according to the provided size.
   * @param val The major and minor axis pair.
   * @param rotation The rotation value from the touch/pen event.
   * @param scalar The scalar cartesian coordinate pair.
   * @return The major and minor axis pair.
   */
  std::pair<float, float> scale_client_contact_area(const std::pair<float, float> &val, uint16_t rotation, const std::pair<float, float> &scalar);
}  // namespace input
