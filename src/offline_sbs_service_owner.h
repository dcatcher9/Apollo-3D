/** Private process-facade lifetime for the native offline job service. */
#pragma once

#include "offline_sbs_job.h"

#include <mutex>
#include <optional>
#include <thread>

#ifdef SUNSHINE_TESTS
namespace offline_sbs::testing {
  struct service_initialization_hooks_t;
  struct service_owner_test_access;
}  // namespace offline_sbs::testing
#endif

namespace offline_sbs::detail {
  class service_owner_t {
  public:
    service_owner_t() = default;
    ~service_owner_t();
    service_owner_t(const service_owner_t &) = delete;
    service_owner_t &operator=(const service_owner_t &) = delete;
    service_owner_t(service_owner_t &&) = delete;
    service_owner_t &operator=(service_owner_t &&) = delete;

    // Retain the immutable startup snapshot; request() alone starts initialization.
    void configure(service_config_t config);
    void configuration_failed(std::string error);
    void request();
    // The process lifetime serializes shutdown; concurrent join callers are unsupported.
    void shutdown();
    [[nodiscard]] std::shared_ptr<job_service_t> ready_service() const;
    [[nodiscard]] nlohmann::json capabilities() const;

  private:
    enum class phase_e {
      unconfigured,
      dormant,
      initializing,
      ready,
      failed,
      closing,
      closed
    };
    void initialize_worker(service_config_t config);

    mutable std::mutex mutex_;
    phase_e phase_ = phase_e::unconfigured;
    std::optional<service_config_t> config_;
    std::shared_ptr<job_service_t> service_;
    std::string error_;
#ifdef SUNSHINE_TESTS
    friend struct testing::service_owner_test_access;
    std::shared_ptr<const testing::service_initialization_hooks_t> test_hooks_;
#endif
    std::jthread initializer_;
  };
}  // namespace offline_sbs::detail
