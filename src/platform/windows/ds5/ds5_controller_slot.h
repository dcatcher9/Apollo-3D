/**
 * @file src/platform/windows/ds5/ds5_controller_slot.h
 * @brief Serialized input routing from a failed DualSense helper to ViGEm.
 */
#pragma once

#include "ds5_sidecar_client.h"

#include <chrono>
#include <optional>

namespace platf::ds5 {
  // Accessed only by the platform input path. The sidecar reader reports final
  // recovery failure through owns(); it never mutates the platform's slots.
  class controller_slot_t {
  public:
    using clock_t = std::chrono::steady_clock;

    int alloc(const gamepad_id_t &id, const gamepad_arrival_t &metadata, feedback_queue_t feedback, bool audio_haptics) {
      reset();
      _sidecar = std::make_unique<sidecar_client_t>();
      if (_sidecar->alloc(id, feedback, audio_haptics, false, metadata.haptics_feedback_queue) != 0) {
        reset();
        return -1;
      }
      _fallback.emplace(id, metadata, std::move(feedback));
      return 0;
    }

    // A null result lets the caller deliver the current event through ViGEm.
    // Retain the original allocation if ViGEm is temporarily unavailable, and
    // limit retries so motion packets cannot repeatedly enumerate devices.
    template<class Fallback>
    sidecar_client_t *sidecar_for_input(Fallback &&allocate_fallback, clock_t::time_point now = clock_t::now()) {
      if (!_fallback) {
        return nullptr;
      }
      if (_sidecar && _sidecar->owns(_fallback->id.globalIndex)) {
        return _sidecar.get();
      }

      // Join the finished reader and close all helper handles before attaching
      // a replacement controller with the same host/client indices.
      _sidecar.reset();
      if (now >= _next_retry) {
        if (allocate_fallback(_fallback->id, _fallback->metadata, _fallback->feedback) == 0) {
          _fallback.reset();
        } else {
          _next_retry = now + std::chrono::seconds(1);
        }
      }
      return nullptr;
    }

    void reset() {
      _sidecar.reset();
      _fallback.reset();
      _next_retry = {};
    }

  private:
    struct fallback_t {
      gamepad_id_t id;
      gamepad_arrival_t metadata;
      feedback_queue_t feedback;
    };

    std::unique_ptr<sidecar_client_t> _sidecar;
    std::optional<fallback_t> _fallback;
    clock_t::time_point _next_retry {};
  };
}  // namespace platf::ds5
