/** Finite gates around the actual offline facade owner and job-service startup. */
#pragma once

#ifdef SUNSHINE_TESTS
  #include "offline_sbs_service_owner.h"

  #include <functional>
  #include <memory>

namespace offline_sbs::testing {
  enum class service_initialization_stage_e {
    worker_enter,
    service_start_finished,
    worker_exit,
  };

  struct service_initialization_hooks_t {
    // Observers must not throw. Gates must be bounded and released before owner destruction.
    std::function<void(service_initialization_stage_e)> observe;
    std::function<void()> before_thread_launch;
    std::function<void()> before_service_allocation;
  };

  struct service_owner_test_access {
    static std::unique_ptr<detail::service_owner_t> create(service_config_t config, service_initialization_hooks_t hooks = {});
    // Common worker-lifecycle cases explicitly request startup even when configure is eager.
    static std::unique_ptr<detail::service_owner_t> start(service_config_t config, service_initialization_hooks_t hooks = {});
    // Observational join only; serialize this with shutdown and request calls.
    static void join(detail::service_owner_t &owner);
  };

  // Install on the actual handler's thread. The local owner must outlive this scope.
  class scoped_service_owner_override_t {
  public:
    explicit scoped_service_owner_override_t(detail::service_owner_t &owner);
    ~scoped_service_owner_override_t();
    scoped_service_owner_override_t(const scoped_service_owner_override_t &) = delete;
    scoped_service_owner_override_t &operator=(const scoped_service_owner_override_t &) = delete;

  private:
    detail::service_owner_t *previous_;
  };
}  // namespace offline_sbs::testing
#endif
