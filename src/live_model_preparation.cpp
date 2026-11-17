#include "live_model_preparation.h"

#include "config.h"
#include "gpu_workload_arbiter.h"
#include "host_sbs_shader_cache.h"
#include "logging.h"
#include "video_depth_estimator.h"

#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

#ifdef SUNSHINE_TESTS
  #include "live_model_preparation_test.h"
  #include "utility.h"
#endif

namespace models {
  using namespace std::literals;

  namespace {
    std::mutex preparation_registry_mutex;
    live_model_preparation_t *registered_preparation = nullptr;

#ifdef SUNSHINE_TESTS
    thread_local const testing::preparation_hooks_t *active_preparation_hooks = nullptr;

    testing::preparation_hooks_t checked_hooks(testing::preparation_hooks_t hooks) {
      if (static_cast<bool>(hooks.prepare_depth) != static_cast<bool>(hooks.prepare_ocr)) {
        throw std::invalid_argument("preparation CPU substitutes must supply both native operations");
      }
      return hooks;
    }
#endif
  }  // namespace

#ifdef SUNSHINE_TESTS
  void testing::observe_preparation_stage_for_test(preparation_stage_e stage) noexcept {
    if (active_preparation_hooks && active_preparation_hooks->observe) {
      active_preparation_hooks->observe(stage);
    }
  }
#endif

  class live_model_preparation_t::impl_t {
  public:
    impl_t(config::depth_model_info model, std::string adapter_name
#ifdef SUNSHINE_TESTS
           ,
           testing::preparation_hooks_t hooks = {}
#endif
    ):
        model_(std::move(model)),
        adapter_name_(std::move(adapter_name)),
#ifdef SUNSHINE_TESTS
        hooks_(checked_hooks(std::move(hooks))),
#endif
        cpu_prewarm_([
#ifdef SUNSHINE_TESTS
                       hooks = hooks_
#endif
    ]() {
          try {
#ifdef SUNSHINE_TESTS
            if (hooks.prewarm_shaders) {
              if (!hooks.prewarm_shaders(SUNSHINE_ASSETS_DIR)) {
                BOOST_LOG(error) << "One or more fixed-shape Host SBS shaders could not be precompiled at startup.";
              }
              return;
            }
            if (hooks.prepare_depth) {
              // Common finite CPU fixtures replace both GPU operations and need no compiler.
              return;
            }
#endif
            if (!host_sbs_shader_cache::prewarm(SUNSHINE_ASSETS_DIR)) {
              BOOST_LOG(error) << "One or more fixed-shape Host SBS shaders could not be precompiled at startup.";
            }
          } catch (const std::exception &exception) {
            BOOST_LOG(error) << "Host SBS startup shader preparation failed: " << exception.what();
          } catch (...) {
            BOOST_LOG(error) << "Host SBS startup shader preparation failed with an unknown exception.";
          }
        }) {
    }

    // The registry lock serializes this publication with every request and unregister.
    bool request_locked() {
      if (gpu_worker_) {
        return true;
      }
      auto lease = gpu_workload::try_acquire(gpu_workload::kind_e::live_stream);
      if (!lease) {
        return false;
      }
      // Engaged even after join or launch failure: one owned attempt, without a parallel flag.
      gpu_worker_.emplace();
      try {
        detail::begin_live_model_preparation(model_);
#ifdef SUNSHINE_TESTS
        if (hooks_.before_gpu_thread_launch) {
          hooks_.before_gpu_thread_launch();
        }
#endif
        *gpu_worker_ = std::jthread([model = model_, adapter_name = adapter_name_, lease = std::move(*lease)
#ifdef SUNSHINE_TESTS
                                                                                     ,
                                     hooks = hooks_
#endif
        ]() {
#ifdef SUNSHINE_TESTS
          const auto *previous = std::exchange(active_preparation_hooks, &hooks);
          auto restore_observer = util::fail_guard([previous]() {
            active_preparation_hooks = previous;
          });
          testing::observe_preparation_stage_for_test(testing::preparation_stage_e::worker_enter);
#endif
          try {
            BOOST_LOG(info) << "Preparing authenticated live depth model '"sv << model.name << "'..."sv;
            try {
#ifdef SUNSHINE_TESTS
              const bool prepared = hooks.prepare_depth ? hooks.prepare_depth(SUNSHINE_ASSETS_DIR, model, adapter_name) : models::prepare_tensorrt_model(SUNSHINE_ASSETS_DIR, model, adapter_name);
#else
              const bool prepared = models::prepare_tensorrt_model(SUNSHINE_ASSETS_DIR, model, adapter_name);
#endif
              if (!prepared) {
                detail::fail_live_model_preparation(model);
                BOOST_LOG(error) << "Startup depth-model preparation failed for '"sv << model.name << "'."sv;
              }
            } catch (const std::exception &exception) {
              detail::fail_live_model_preparation(model);
              BOOST_LOG(error) << "Live depth-model preparation failed for '" << model.name << "': " << exception.what();
            } catch (...) {
              detail::fail_live_model_preparation(model);
              BOOST_LOG(error) << "Live depth-model preparation failed for '" << model.name << "' with an unknown exception.";
            }
#ifdef SUNSHINE_TESTS
            testing::observe_preparation_stage_for_test(testing::preparation_stage_e::depth_finished);
#endif
            try {
#ifdef SUNSHINE_TESTS
              const bool prepared = hooks.prepare_ocr ? hooks.prepare_ocr(SUNSHINE_ASSETS_DIR, adapter_name) : models::prepare_ocr_tensorrt_model(SUNSHINE_ASSETS_DIR, adapter_name);
#else
              const bool prepared = models::prepare_ocr_tensorrt_model(SUNSHINE_ASSETS_DIR, adapter_name);
#endif
              if (!prepared) {
                BOOST_LOG(warning)
                  << "Startup PP-OCRv6 tiny preparation failed; Host SBS will run without subtitle "sv
                     "conditioning."sv;
              }
            } catch (const std::exception &exception) {
              BOOST_LOG(warning) << "Optional live PP-OCRv6 tiny preparation failed: " << exception.what();
            } catch (...) {
              BOOST_LOG(warning) << "Optional live PP-OCRv6 tiny preparation failed with an unknown exception.";
            }
#ifdef SUNSHINE_TESTS
            testing::observe_preparation_stage_for_test(testing::preparation_stage_e::ocr_finished);
#endif
          } catch (const std::exception &exception) {
            if (tensorrt_model_prepare_status(model) != engine_build_status::ready) {
              detail::fail_live_model_preparation(model);
            }
            BOOST_LOG(error) << "Live model preparation worker failed: " << exception.what();
          } catch (...) {
            if (tensorrt_model_prepare_status(model) != engine_build_status::ready) {
              detail::fail_live_model_preparation(model);
            }
            BOOST_LOG(error) << "Live model preparation worker failed with an unknown exception.";
          }
#ifdef SUNSHINE_TESTS
          testing::observe_preparation_stage_for_test(testing::preparation_stage_e::worker_exit);
#endif
          // The captured counted live lease survives the complete depth -> optional OCR sequence.
          // Joining the actual worker also proves its closure (and therefore the lease) retired.
        });
      } catch (const std::exception &exception) {
        detail::fail_live_model_preparation(model_);
        BOOST_LOG(error) << "Could not start live model preparation: " << exception.what();
      } catch (...) {
        detail::fail_live_model_preparation(model_);
        BOOST_LOG(error) << "Could not start live model preparation: unknown exception.";
      }
      return true;
    }

    void join() {
      if (cpu_prewarm_.joinable()) {
        cpu_prewarm_.join();
      }
      if (gpu_worker_ && gpu_worker_->joinable()) {
        gpu_worker_->join();
      }
    }

  private:
    const config::depth_model_info model_;
    const std::string adapter_name_;
#ifdef SUNSHINE_TESTS
    const testing::preparation_hooks_t hooks_;
#endif
    std::optional<std::jthread> gpu_worker_;
    std::jthread cpu_prewarm_;
  };

  void live_model_preparation_t::register_owner() {
    std::lock_guard lock(preparation_registry_mutex);
    if (registered_preparation) {
      throw std::logic_error("a process live model preparation owner is already registered");
    }
    registered_preparation = this;
  }

  live_model_preparation_t::live_model_preparation_t(config::depth_model_info model, std::string adapter_name):
      impl_(std::make_unique<impl_t>(std::move(model), std::move(adapter_name))) {
    register_owner();
  }

  void live_model_preparation_t::close() {
    {
      std::lock_guard lock(preparation_registry_mutex);
      if (registered_preparation == this) {
        registered_preparation = nullptr;
      }
    }
    // Every request that could publish a handle completed before unregister released the lock.
    impl_->join();
  }

  live_model_preparation_t::~live_model_preparation_t() {
    close();
  }

  bool request_live_model_preparation() {
    std::lock_guard lock(preparation_registry_mutex);
    return registered_preparation && registered_preparation->impl_->request_locked();
  }

#ifdef SUNSHINE_TESTS
  live_model_preparation_t::live_model_preparation_t(config::depth_model_info model, std::string adapter_name, testing::preparation_hooks_t hooks):
      impl_(std::make_unique<impl_t>(std::move(model), std::move(adapter_name), std::move(hooks))) {
    register_owner();
  }

  std::unique_ptr<live_model_preparation_t> testing::live_model_preparation_test_access::create(config::depth_model_info model, std::string adapter_name, preparation_hooks_t hooks) {
    return std::unique_ptr<live_model_preparation_t>(new live_model_preparation_t(std::move(model), std::move(adapter_name), std::move(hooks)));
  }

  std::unique_ptr<live_model_preparation_t> testing::live_model_preparation_test_access::start(config::depth_model_info model, std::string adapter_name, preparation_hooks_t hooks) {
    auto owner = create(std::move(model), std::move(adapter_name), std::move(hooks));
    (void) request_live_model_preparation();
    return owner;
  }

  void testing::live_model_preparation_test_access::join(live_model_preparation_t &owner) {
    owner.impl_->join();
  }

  void testing::live_model_preparation_test_access::close(live_model_preparation_t &owner) {
    owner.close();
  }

  bool testing::live_model_preparation_test_access::registered(const live_model_preparation_t &owner) {
    std::lock_guard lock(preparation_registry_mutex);
    return registered_preparation == &owner;
  }
#endif
}  // namespace models
