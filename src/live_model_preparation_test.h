#pragma once

#ifdef SUNSHINE_TESTS
  #include "live_model_preparation.h"
  #include "host_sbs_conversion_work.h"

  #include <filesystem>
  #include <functional>
  #include <memory>
  #include <string>

namespace models::testing {
  enum class preparation_stage_e {
    worker_enter,
    depth_building,
    depth_finished,
    ocr_finished,
    worker_exit,
  };

  struct preparation_hooks_t {
    /** Observer must not throw. Captured state must remain alive until the real worker joins. */
    std::function<void(preparation_stage_e)> observe;
    /** Empty together selects the real native calls; CPU substitutes must supply both. */
    std::function<bool(const std::filesystem::path &, const config::depth_model_info &, const std::string &)> prepare_depth;
    std::function<bool(const std::filesystem::path &, const std::string &)> prepare_ocr;
    /** Optional finite CPU-prewarm override; absent with paired CPU operations skips compilation. */
    std::function<bool(const std::filesystem::path &)> prewarm_shaders;
    /** Runs at the real launch edge, after once-slot and building publication. It may throw. */
    std::function<void()> before_gpu_thread_launch;
  };

  struct live_model_preparation_test_access {
    /** Construct the actual owner without adding an explicit demand in either phase. */
    static std::unique_ptr<live_model_preparation_t> create(config::depth_model_info model, std::string adapter_name, preparation_hooks_t hooks);
    /** Worker-semantic adapter: construct the owner and issue one explicit process request. */
    static std::unique_ptr<live_model_preparation_t> start(config::depth_model_info model, std::string adapter_name, preparation_hooks_t hooks);
    /** Observational join; caller serializes it against requests. Does not unregister or request. */
    static void join(live_model_preparation_t &owner);
    /** Actual unregister-then-join path, without deleting the owner during a concurrent test. */
    static void close(live_model_preparation_t &owner);
    /** Pure registry observation; unlike request, this cannot start GPU work. */
    static bool registered(const live_model_preparation_t &owner);
  };

  /** Explicit controller-only simulated completion for unique noncanonical CPU test identities. */
  void publish_model_preparation_status_for_test(const config::depth_model_info &model, engine_build_status status);

  /** No-op outside an observed worker; the real model calls this after publishing building. */
  void observe_preparation_stage_for_test(preparation_stage_e stage) noexcept;
}  // namespace models::testing
#endif
