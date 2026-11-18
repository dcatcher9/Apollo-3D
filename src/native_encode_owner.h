/**
 * @file src/native_encode_owner.h
 * @brief Move-only native encoder ownership with tracked retirement.
 */
#pragma once

#include <future>
#include <memory>

namespace video::detail {
  class tracked_async_worker_pool_t;
}

namespace platf {
  struct nvenc_encode_device_t;

  /** Keep driver-sensitive backend destruction on the process-owned teardown workers. */
  class native_encode_owner_t {
  public:
    native_encode_owner_t() noexcept;
    explicit native_encode_owner_t(std::unique_ptr<nvenc_encode_device_t> device) noexcept;
    ~native_encode_owner_t();

    native_encode_owner_t(const native_encode_owner_t &) = delete;
    native_encode_owner_t &operator=(const native_encode_owner_t &) = delete;
    native_encode_owner_t(native_encode_owner_t &&other) noexcept;
    native_encode_owner_t &operator=(native_encode_owner_t &&other) noexcept;

    explicit operator bool() const noexcept;
    nvenc_encode_device_t *get() const noexcept;
    nvenc_encode_device_t *operator->() const noexcept;

    /** Consume this owner. A valid completion becomes ready only after backend destruction. */
    [[nodiscard]] std::future<void> retire() &&;

  private:
    std::unique_ptr<nvenc_encode_device_t> device_;
#ifdef SUNSHINE_TESTS
    friend struct native_encode_owner_test_access;
    video::detail::tracked_async_worker_pool_t *test_workers_ = nullptr;
#endif
  };

#ifdef SUNSHINE_TESTS
  struct native_encode_owner_test_access {
    static native_encode_owner_t adopt(
      std::unique_ptr<nvenc_encode_device_t> device,
      video::detail::tracked_async_worker_pool_t &workers
    );
  };
#endif
}  // namespace platf
