/**
 * @file src/native_encode_owner.cpp
 * @brief Definitions for tracked native encoder ownership.
 */
#include "native_encode_owner.h"

#include "platform/common.h"
#include "video.h"

#include <utility>

#ifdef SUNSHINE_TESTS
  #include "tracked_async_worker.h"
#endif

namespace platf {
  native_encode_owner_t::native_encode_owner_t() noexcept = default;

  native_encode_owner_t::native_encode_owner_t(std::unique_ptr<nvenc_encode_device_t> device) noexcept:
      device_(std::move(device)) {
  }

  native_encode_owner_t::~native_encode_owner_t() {
    (void) std::move(*this).retire();
  }

  native_encode_owner_t::native_encode_owner_t(native_encode_owner_t &&other) noexcept:
      device_(std::move(other.device_)) {
#ifdef SUNSHINE_TESTS
    test_workers_ = std::exchange(other.test_workers_, nullptr);
#endif
  }

  native_encode_owner_t &native_encode_owner_t::operator=(native_encode_owner_t &&other) noexcept {
    if (this != &other) {
      (void) std::move(*this).retire();
      device_ = std::move(other.device_);
#ifdef SUNSHINE_TESTS
      test_workers_ = std::exchange(other.test_workers_, nullptr);
#endif
    }
    return *this;
  }

  native_encode_owner_t::operator bool() const noexcept {
    return static_cast<bool>(device_);
  }

  nvenc_encode_device_t *native_encode_owner_t::get() const noexcept {
    return device_.get();
  }

  nvenc_encode_device_t *native_encode_owner_t::operator->() const noexcept {
    return device_.get();
  }

  std::future<void> native_encode_owner_t::retire() && {
    if (!device_) {
      return {};
    }
    std::packaged_task<void()> teardown([device = std::move(device_)]() mutable {
      // Captured state can outlive invocation. Destroy here so completion proves cleanup finished.
      device.reset();
    });
    auto completion = teardown.get_future();
#ifdef SUNSHINE_TESTS
    if (auto *workers = std::exchange(test_workers_, nullptr)) {
      workers->launch(std::move(teardown));
    } else
#endif
    {
      video::launch_async_teardown_worker(std::move(teardown));
    }
    return completion;
  }

#ifdef SUNSHINE_TESTS
  native_encode_owner_t native_encode_owner_test_access::adopt(
    std::unique_ptr<nvenc_encode_device_t> device,
    video::detail::tracked_async_worker_pool_t &workers
  ) {
    native_encode_owner_t owner {std::move(device)};
    owner.test_workers_ = &workers;
    return owner;
  }
#endif
}  // namespace platf
