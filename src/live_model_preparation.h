#pragma once

#include <memory>
#include <string>

namespace config {
  struct depth_model_info;
}

namespace models {
#ifdef SUNSHINE_TESTS
  namespace testing {
    struct preparation_hooks_t;
    struct live_model_preparation_test_access;
  }  // namespace testing
#endif

  /** Request the process-owned GPU preparation once. False means no owner or denied admission. */
  bool request_live_model_preparation();

  namespace detail {
    void begin_live_model_preparation(const config::depth_model_info &model);
    void fail_live_model_preparation(const config::depth_model_info &model);
  }  // namespace detail

  /** Process-scoped CPU prewarm and demand-owned GPU preparation, joined before logging ends. */
  class live_model_preparation_t {
  public:
    live_model_preparation_t(config::depth_model_info model, std::string adapter_name);
    ~live_model_preparation_t();
    live_model_preparation_t(const live_model_preparation_t &) = delete;
    live_model_preparation_t &operator=(const live_model_preparation_t &) = delete;
    live_model_preparation_t(live_model_preparation_t &&) = delete;
    live_model_preparation_t &operator=(live_model_preparation_t &&) = delete;

  private:
    class impl_t;
    std::unique_ptr<impl_t> impl_;
    friend bool request_live_model_preparation();
    void register_owner();
    void close();

#ifdef SUNSHINE_TESTS
    friend struct testing::live_model_preparation_test_access;
    live_model_preparation_t(config::depth_model_info model, std::string adapter_name, testing::preparation_hooks_t hooks);
#endif
  };
}  // namespace models
