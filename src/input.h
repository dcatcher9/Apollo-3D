/**
 * @file src/input.h
 * @brief Declarations for gamepad, keyboard, and mouse input handling.
 */
#pragma once

// standard includes
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
#include <list>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

// local includes
#include "crypto.h"
#include "platform/common.h"
#include "thread_safe.h"

namespace input {
  constexpr std::size_t INPUT_PACKET_SIZE_MAX = 128;

  struct input_t;

  namespace detail {
    constexpr std::size_t INPUT_DRAIN_QUANTUM = 32;

    enum class batch_result_e {
      batched,  ///< The later packet was folded into the first packet.
      not_batchable,  ///< Skip this packet and continue looking for compatible packets.
      terminate_batch,  ///< Preserve this ordering boundary and stop batching.
    };

    /**
     * @brief Add two signed 16-bit input deltas without allowing protocol-field wraparound.
     * @return The exact sum, or std::nullopt when it is not representable as int16_t.
     */
    constexpr std::optional<std::int16_t> checked_add_i16(std::int16_t left, std::int16_t right) noexcept {
      const auto sum = static_cast<std::int32_t>(left) + static_cast<std::int32_t>(right);
      if (sum < std::numeric_limits<std::int16_t>::min() || sum > std::numeric_limits<std::int16_t>::max()) {
        return std::nullopt;
      }
      return static_cast<std::int16_t>(sum);
    }

    /**
     * @brief Batch two validated wire-format input packets using the production coalescer.
     * @details The destination is mutated only when the result is `batched`.
     */
    [[nodiscard]] batch_result_e batch_packets(
      std::span<std::uint8_t> destination,
      std::span<const std::uint8_t> source
    ) noexcept;

    /**
     * @brief Remove and coalesce the next packet from an input queue.
     * @details Empty packets are internal reset barriers and are never coalesced across.
     * The caller owns synchronization for the queue.
     */
    [[nodiscard]] std::optional<std::vector<std::uint8_t>> pop_next_batched_packet(
      std::list<std::vector<std::uint8_t>> &packets
    );

    /**
     * @brief Edge-trigger state for a single input-queue drain owner.
     * @details All calls must be serialized by the queue's mutex.
     */
    class drain_gate_t {
    public:
      [[nodiscard]] bool request() noexcept {
        if (active_) {
          return false;
        }
        active_ = true;
        return true;
      }

      void release() noexcept {
        active_ = false;
      }

      [[nodiscard]] bool active() const noexcept {
        return active_;
      }

    private:
      bool active_ = false;
    };

    /**
     * @brief Decide whether a drain turn has exhausted its fairness quantum.
     * @details The production drain and deterministic tests share this exact policy seam.
     */
    [[nodiscard]] constexpr bool drain_turn_exhausted(std::size_t dispatched) noexcept {
      return dispatched >= INPUT_DRAIN_QUANTUM;
    }

    /** Continuations yield through the timer queue so already-due timers run before more input. */
    [[nodiscard]] constexpr bool drain_continuation_uses_timer_queue() noexcept {
      return true;
    }

    /**
     * @brief Reject a packet removed by a drain before a concurrent reset generation change.
     */
    [[nodiscard]] constexpr bool generation_is_current(
      std::uint64_t packet_generation,
      std::uint64_t current_generation
    ) noexcept {
      return packet_generation == current_generation;
    }

    /**
     * @brief Model the atomic empty-to-idle transition used by an input drain owner.
     * @return true when the owner must continue because an arrival is already queued.
     */
    inline bool release_if_empty(drain_gate_t &gate, bool queue_empty) noexcept {
      if (!queue_empty) {
        return true;
      }
      gate.release();
      return false;
    }

    /** @brief Guard delayed controller continuations against reset or a superseding timer. */
    [[nodiscard]] constexpr bool delayed_action_is_current(
      bool input_reset,
      std::uint64_t scheduled_action,
      std::uint64_t current_action
    ) noexcept {
      return !input_reset && scheduled_action == current_action;
    }

    [[nodiscard]] constexpr bool controller_action_is_current(
      bool input_reset,
      int controller_id,
      std::uint64_t scheduled_action,
      std::uint64_t current_action
    ) noexcept {
      return controller_id >= 0 && delayed_action_is_current(input_reset, scheduled_action, current_action);
    }

    /** A key-repeat callback is valid only for the live, still-pressed scheduling generation. */
    [[nodiscard]] constexpr bool key_repeat_is_current(
      bool input_reset,
      std::uint64_t scheduled_generation,
      std::uint64_t current_generation,
      bool key_is_pressed
    ) noexcept {
      return delayed_action_is_current(input_reset, scheduled_generation, current_generation) && key_is_pressed;
    }

    /** Keep HOME asserted across ordinary controller reports until the release continuation. */
    template<class ButtonFlags>
    constexpr ButtonFlags latch_button_while_active(
      ButtonFlags reported_flags,
      ButtonFlags held_button,
      bool hold_active
    ) noexcept {
      return hold_active ? static_cast<ButtonFlags>(reported_flags | held_button) : reported_flags;
    }

    /**
     * @brief Admit work only while the input context is live, under the caller's dispatch lock.
     */
    template<class Admit>
    bool admit_if_live(bool input_reset, Admit &&admit) {
      if (input_reset) {
        return false;
      }
      std::invoke(std::forward<Admit>(admit));
      return true;
    }

    /** Decide whether an obsolete delayed callback should still perform its action. */
    [[nodiscard]] constexpr bool delayed_action_needed(bool state_already_changed) noexcept {
      return !state_already_changed;
    }

    /** A delayed mouse action may only replace state from the generation that scheduled it. */
    [[nodiscard]] inline bool claim_delayed_generation(
      std::atomic<std::uint64_t> &current,
      std::uint64_t scheduled
    ) noexcept {
      return current.compare_exchange_strong(scheduled, scheduled + 1);
    }

    [[nodiscard]] constexpr bool valid_mouse_button(std::uint8_t button) noexcept {
      return button >= 1 && button <= 5;
    }

    template<class Token>
    [[nodiscard]] constexpr bool should_flush_pending_left_release(
      Token token,
      Token relative_sentinel,
      bool left_is_pressed
    ) noexcept {
      return token != nullptr && token != relative_sentinel && !left_is_pressed;
    }

    /** Complete reset only after every allocated virtual gamepad has been freed synchronously. */
    template<class GamepadRange, class FreeGamepad, class Complete>
    void free_gamepads_before_completion(
      GamepadRange &gamepads,
      FreeGamepad &&free_gamepad,
      Complete &&complete
    ) {
      for (auto &gamepad : gamepads) {
        if (gamepad.id >= 0) {
          std::invoke(free_gamepad, gamepad.id);
          gamepad.id = -1;
        }
      }
      std::invoke(std::forward<Complete>(complete));
    }

    /**
     * Serialize packet admission/dispatch with reset without holding the producer queue mutex
     * across OS injection. A reset that wins this mutex invalidates the old generation first;
     * a dispatch that wins completes before reset's release barrier.
     */
    template<class Dispatch>
    bool dispatch_if_current(
      std::mutex &dispatch_mutex,
      const std::uint64_t packet_generation,
      const std::uint64_t &current_generation,
      Dispatch &&dispatch
    ) {
      std::lock_guard lock {dispatch_mutex};
      if (!generation_is_current(packet_generation, current_generation)) {
        return false;
      }
      std::invoke(std::forward<Dispatch>(dispatch));
      return true;
    }

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

  /** Reset all input and return a completion that becomes ready after the ordered release barrier. */
  [[nodiscard]] std::future<void> reset(std::shared_ptr<input_t> &input);
  void passthrough(std::shared_ptr<input_t> &input, std::vector<std::uint8_t> &&input_data, const crypto::PERM &permission);

  /**
   * @brief Validate a Gen 5+ Artemis input packet before any typed access.
   * @return The host-endian packet magic, or std::nullopt for malformed/unsupported input.
   */
  [[nodiscard]] std::optional<std::uint32_t> validated_packet_magic(std::span<const std::uint8_t> input_data) noexcept;

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
