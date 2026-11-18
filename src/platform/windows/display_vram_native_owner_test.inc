// Test-only facade included inside platf::dxgi after d3d_nvenc_encode_device_t.
// All headers, including tracked_async_worker.h, are included at file scope.
namespace {
  struct native_capture_release_state_t {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false;
    bool released = false;
    bool timed_out = false;
    std::uint64_t claim_after_release = 0;
    std::atomic<std::uint64_t> final_claim {0};
    std::atomic<bool> display_destroyed {false};

    void release() {
      {
        std::lock_guard lock {mutex};
        released = true;
      }
      changed.notify_all();
    }
  };

  class native_capture_display_t final: public display_base_t {
  public:
    explicit native_capture_display_t(std::shared_ptr<native_capture_release_state_t> state):
        state_(std::move(state)) {
    }

    ~native_capture_display_t() override {
      state_->final_claim.store(external_pixels_owner.load(std::memory_order_acquire), std::memory_order_release);
      state_->display_destroyed.store(true, std::memory_order_release);
    }

    std::shared_ptr<platf::img_t> alloc_img() override {
      return {};
    }

    int dummy_img(platf::img_t *) override {
      return -1;
    }

    std::vector<DXGI_FORMAT> get_supported_capture_formats() override {
      return {};
    }

  protected:
    capture_e snapshot(const pull_free_image_cb_t &, std::shared_ptr<platf::img_t> &, std::chrono::milliseconds, bool) override {
      return capture_e::error;
    }

    capture_e release_snapshot() override {
      return capture_e::ok;
    }

    int complete_img(platf::img_t *, bool) override {
      return -1;
    }

  private:
    std::shared_ptr<native_capture_release_state_t> state_;
  };
}  // namespace

struct native_capture_retirement_state_t {
  std::mutex mutex;
  std::condition_variable changed;
  bool entered = false;
  bool released = false;
  bool timed_out = false;

  void release() {
    {
      std::lock_guard lock {mutex};
      released = true;
    }
    changed.notify_all();
  }
};

struct native_capture_release_test_access {
  static void prepare_retirement(
    d3d_nvenc_encode_device_t &device,
    std::shared_ptr<display_base_t> display,
    const std::shared_ptr<native_capture_retirement_state_t> &state
  ) {
    device.base.display = std::move(display);
    device.base.claim_capture_pixels(true);
    device.nvenc_d3d = std::make_unique<nvenc::nvenc_d3d11_native>(nullptr);
    device.test_after_capture_release_ = [state]() {
      std::unique_lock lock {state->mutex};
      state->entered = true;
      state->changed.notify_all();
      state->timed_out = !state->changed.wait_for(lock, std::chrono::seconds {10}, [&]() {
        return state->released;
      });
    };
  }

  static void prepare(
    d3d_nvenc_encode_device_t &device,
    const std::shared_ptr<display_base_t> &display,
    const std::shared_ptr<native_capture_release_state_t> &state
  ) {
    device.base.display = display;
    device.base.claim_capture_pixels(true);
    // Its constructor creates only the CPU completion event. Neither init_device(),
    // init_encoder(), nor the NVIDIA loader is called by this facade.
    device.nvenc_d3d = std::make_unique<nvenc::nvenc_d3d11_native>(nullptr);
    device.test_after_capture_release_ = [weak_display = std::weak_ptr<display_base_t> {display}, state]() {
      std::uint64_t claim = state->final_claim.load(std::memory_order_acquire);
      {
        // Do not retain this temporary display reference across the destruction gate.
        const auto display = weak_display.lock();
        if (display) {
          claim = display->external_pixels_owner.load(std::memory_order_acquire);
        }
      }
      std::unique_lock lock {state->mutex};
      state->claim_after_release = claim;
      state->entered = true;
      state->changed.notify_all();
      state->timed_out = !state->changed.wait_for(lock, std::chrono::seconds {2}, [&]() {
        return state->released;
      });
    };
  }

  static std::uint64_t own_token(const d3d_nvenc_encode_device_t &device) {
    return device.base.capture_pixels_token;
  }
};

native_capture_release_report_t native_capture_release_for_test(const bool foreign) {
  native_capture_release_report_t report;
  try {
    ::video::detail::tracked_async_worker_pool_t workers;
    const auto state = std::make_shared<native_capture_release_state_t>();
    // Release before the local pool drains, including when an allocation or wait fails.
    auto release_guard = util::fail_guard([&]() {
      state->release();
    });
    auto display = std::make_shared<native_capture_display_t>(state);
    const auto weak_display = std::weak_ptr<display_base_t> {display};
    auto candidate = std::make_unique<d3d_nvenc_encode_device_t>();
    auto &device = *candidate;
    auto owner = platf::native_encode_owner_test_access::adopt(std::move(candidate), workers);
    native_capture_release_test_access::prepare(device, display, state);
    report.own_token = native_capture_release_test_access::own_token(device);

    std::unique_ptr<d3d_base_encode_device> foreign_owner;
    if (foreign) {
      foreign_owner = std::make_unique<d3d_base_encode_device>();
      foreign_owner->display = display;
      foreign_owner->claim_capture_pixels(true);
      report.foreign_token = foreign_owner->capture_pixels_token;
    }
    report.valid = report.own_token != 0 && (!foreign || (report.foreign_token != 0 && report.foreign_token != report.own_token));
    display.reset();

    auto completion = std::move(owner).retire();
    report.owner_consumed = !owner;
    {
      std::unique_lock lock {state->mutex};
      report.gate_entered = state->changed.wait_for(lock, std::chrono::seconds {1}, [&]() {
        return state->entered;
      });
      report.claim_after_capture_release = state->claim_after_release;
    }
    report.retirement_pending_before_release = completion.valid() && completion.wait_for(std::chrono::milliseconds {0}) == std::future_status::timeout;
    if (foreign_owner) {
      foreign_owner->release_capture_display();
      foreign_owner.reset();
    }
    report.display_expired_before_release = weak_display.expired() && state->display_destroyed.load(std::memory_order_acquire);
    report.final_claim = state->final_claim.load(std::memory_order_acquire);

    state->release();
    report.completed_after_release = completion.valid() && completion.wait_for(std::chrono::seconds {2}) == std::future_status::ready;
    if (report.completed_after_release) {
      completion.get();
    }
    {
      std::lock_guard lock {state->mutex};
      report.gate_timed_out = state->timed_out;
    }
  } catch (...) {
    report.valid = false;
  }
  return report;
}

struct native_capture_retirement_test_t::impl_t {
  impl_t(
    std::shared_ptr<display_base_t> display,
    ::video::detail::tracked_async_worker_pool_t &workers
  ):
      state(std::make_shared<native_capture_retirement_state_t>()) {
    try {
      auto candidate = std::make_unique<d3d_nvenc_encode_device_t>();
      auto &device = *candidate;
      owner = platf::native_encode_owner_test_access::adopt(std::move(candidate), workers);
      native_capture_release_test_access::prepare_retirement(device, std::move(display), state);
    } catch (...) {
      state->release();
      throw;
    }
  }

  ~impl_t() {
    state->release();
    retire();
    if (completion.valid()) {
      completion.wait();
    }
  }

  void retire() {
    if (owner) {
      completion = std::move(owner).retire().share();
    }
  }

  std::shared_ptr<native_capture_retirement_state_t> state;
  platf::native_encode_owner_t owner;
  std::shared_future<void> completion;
};

native_capture_retirement_test_t::native_capture_retirement_test_t(
  std::shared_ptr<display_base_t> display,
  ::video::detail::tracked_async_worker_pool_t &workers
):
    impl_(std::make_unique<impl_t>(std::move(display), workers)) {
}

native_capture_retirement_test_t::~native_capture_retirement_test_t() = default;

void native_capture_retirement_test_t::retire() {
  impl_->retire();
}

bool native_capture_retirement_test_t::wait_after_capture_release(const std::chrono::milliseconds timeout) {
  std::unique_lock lock {impl_->state->mutex};
  return impl_->state->changed.wait_for(lock, timeout, [&]() {
    return impl_->state->entered;
  });
}

void native_capture_retirement_test_t::release_native_teardown() {
  impl_->state->release();
}

bool native_capture_retirement_test_t::wait_retired(const std::chrono::milliseconds timeout) {
  return impl_->completion.valid() && impl_->completion.wait_for(timeout) == std::future_status::ready;
}

bool native_capture_retirement_test_t::gate_timed_out() const {
  std::lock_guard lock {impl_->state->mutex};
  return impl_->state->timed_out;
}
