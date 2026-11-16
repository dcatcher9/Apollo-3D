/**
 * @file src/platform/windows/input_feedback_test.h
 * @brief Private, device-free access to the real Windows ViGEm feedback methods.
 */
#pragma once

#ifdef SUNSHINE_TESTS
  #include "src/platform/common.h"

  #include <cstddef>
  #include <cstdint>
  #include <memory>

namespace platf::detail {
  struct vigem_feedback_cache_sizes_t {
    std::size_t rumble;
    std::size_t rgb;
    std::size_t context;
    std::size_t message;
    std::size_t slot_count;
  };

  class vigem_feedback_test_t {
  public:
    vigem_feedback_test_t(feedback_queue_t queue, std::uint8_t relative_index);
    ~vigem_feedback_test_t();

    void rumble(std::uint8_t large_motor, std::uint8_t small_motor);
    void set_rgb_led(std::uint8_t r, std::uint8_t g, std::uint8_t b);
    void set_feedback_destination(feedback_queue_t queue, std::uint8_t relative_index);
    static vigem_feedback_cache_sizes_t cache_sizes();

  private:
    struct impl_t;
    std::unique_ptr<impl_t> impl_;
  };
}  // namespace platf::detail
#endif
