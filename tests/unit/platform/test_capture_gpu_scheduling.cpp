#include <gtest/gtest.h>
#ifdef _WIN32
  #include "src/platform/windows/capture_gpu_scheduling.h"

  #include <vector>

namespace {
  namespace d = platf::dxgi::detail;

  constexpr d::status_t failure = static_cast<d::status_t>(0xc0000001u);

  enum class operation_e {
    open,
    query,
    close,
    set_priority,
  };

  class CaptureGpuSchedulingTest: public ::testing::Test {
  protected:
    void SetUp() override {
      current_ = this;
      adapter_desc.AdapterLuid.LowPart = 0x12345678;
      adapter_desc.AdapterLuid.HighPart = -17;
      adapter_desc.VendorId = 0x10DE;
    }

    void TearDown() override {
      current_ = nullptr;
    }

    static d::status_t WINAPI open_adapter(const d::D3DKMT_OPENADAPTERFROMLUID *request) {
      auto &state = *current_;
      state.operations.push_back(operation_e::open);
      EXPECT_EQ(request->AdapterLuid.LowPart, state.adapter_desc.AdapterLuid.LowPart);
      EXPECT_EQ(request->AdapterLuid.HighPart, state.adapter_desc.AdapterLuid.HighPart);
      if (SUCCEEDED(state.open_result)) {
        // The SDK declares the outer pointer const even though hAdapter is an output.
        const_cast<d::D3DKMT_OPENADAPTERFROMLUID *>(request)->hAdapter = adapter_handle;
      }
      return state.open_result;
    }

    static d::status_t WINAPI query_adapter_info(const d::D3DKMT_QUERYADAPTERINFO *request) {
      auto &state = *current_;
      state.operations.push_back(operation_e::query);
      EXPECT_EQ(request->hAdapter, adapter_handle);
      EXPECT_EQ(request->Type, 70u);
      EXPECT_EQ(request->PrivateDriverDataSize, sizeof(d::D3DKMT_WDDM_2_7_CAPS));
      auto &caps = *static_cast<d::D3DKMT_WDDM_2_7_CAPS *>(request->pPrivateDriverData);
      EXPECT_EQ(caps.Value, 0u);
      // Also set data on failure to prove that a failed query never supplies HAGS policy.
      caps.HwSchEnabled = state.hags_enabled;
      return state.query_result;
    }

    static d::status_t WINAPI close_adapter(const d::D3DKMT_CLOSEADAPTER *request) {
      auto &state = *current_;
      state.operations.push_back(operation_e::close);
      EXPECT_EQ(request->hAdapter, adapter_handle);
      return state.close_result;
    }

    static d::status_t WINAPI set_process_priority(HANDLE process, d::D3DKMT_SCHEDULINGPRIORITYCLASS priority) {
      auto &state = *current_;
      state.operations.push_back(operation_e::set_priority);
      EXPECT_EQ(process, GetCurrentProcess());
      state.requested_priority = priority;
      return state.priority_result;
    }

    void configure(bool nv_realtime_hags = false) {
      d::configure_capture_process_gpu_priority(adapter_desc, nv_realtime_hags, api);
    }

    void expect_complete_order() const {
      EXPECT_EQ(operations, (std::vector<operation_e> {
                              operation_e::open,
                              operation_e::query,
                              operation_e::close,
                              operation_e::set_priority,
                            }));
    }

    DXGI_ADAPTER_DESC adapter_desc {};
    bool hags_enabled = true;
    d::status_t open_result = 0, query_result = 0, close_result = 0, priority_result = 0;
    d::D3DKMT_SCHEDULINGPRIORITYCLASS requested_priority = d::D3DKMT_SCHEDULINGPRIORITYCLASS_IDLE;
    std::vector<operation_e> operations;
    const d::capture_gpu_scheduling_api_t api {
      open_adapter,
      query_adapter_info,
      close_adapter,
      set_process_priority,
    };

  private:
    static constexpr UINT adapter_handle = 0x76543210;
    static inline CaptureGpuSchedulingTest *current_ = nullptr;
  };

  TEST_F(CaptureGpuSchedulingTest, PreservesNvidiaHagsPriorityPolicy) {
    struct policy_t {
      UINT vendor;
      bool hags;
      bool nv_realtime_hags;
      d::D3DKMT_SCHEDULINGPRIORITYCLASS expected;
    };

    const policy_t policies[] = {
      {0x10DE, true, false, d::D3DKMT_SCHEDULINGPRIORITYCLASS_HIGH},
      {0x10DE, false, false, d::D3DKMT_SCHEDULINGPRIORITYCLASS_REALTIME},
      {0x10DE, true, true, d::D3DKMT_SCHEDULINGPRIORITYCLASS_REALTIME},
      {0x8086, true, false, d::D3DKMT_SCHEDULINGPRIORITYCLASS_REALTIME},
    };
    for (const auto &policy : policies) {
      SCOPED_TRACE(::testing::Message() << "vendor=" << policy.vendor << ", hags=" << policy.hags << ", nv_realtime_hags=" << policy.nv_realtime_hags);
      operations.clear();
      adapter_desc.VendorId = policy.vendor;
      hags_enabled = policy.hags;

      configure(policy.nv_realtime_hags);

      expect_complete_order();
      EXPECT_EQ(requested_priority, policy.expected);
    }
  }

  TEST_F(CaptureGpuSchedulingTest, FailedOpenStillSetsRealtimeWithoutQueryOrClose) {
    open_result = failure;

    configure();

    EXPECT_EQ(operations, (std::vector<operation_e> {operation_e::open, operation_e::set_priority}));
    EXPECT_EQ(requested_priority, d::D3DKMT_SCHEDULINGPRIORITYCLASS_REALTIME);
  }

  TEST_F(CaptureGpuSchedulingTest, FailedQueryStillClosesAndSetsRealtime) {
    query_result = failure;

    configure();

    expect_complete_order();
    EXPECT_EQ(requested_priority, d::D3DKMT_SCHEDULINGPRIORITYCLASS_REALTIME);
  }

  TEST_F(CaptureGpuSchedulingTest, FailedCloseRetainsHagsPolicyAndStillSetsPriority) {
    close_result = failure;

    configure();

    expect_complete_order();
    EXPECT_EQ(requested_priority, d::D3DKMT_SCHEDULINGPRIORITYCLASS_HIGH);
  }

  TEST_F(CaptureGpuSchedulingTest, FailedPriorityRequestRemainsNonfatal) {
    priority_result = failure;

    EXPECT_NO_THROW(configure());

    expect_complete_order();
    EXPECT_EQ(requested_priority, d::D3DKMT_SCHEDULINGPRIORITYCLASS_HIGH);
  }

  TEST_F(CaptureGpuSchedulingTest, AcceptsNonnegativeNtstatusSuccess) {
    open_result = query_result = close_result = priority_result = 1;

    configure();

    expect_complete_order();
    EXPECT_EQ(requested_priority, d::D3DKMT_SCHEDULINGPRIORITYCLASS_HIGH);
  }
}  // namespace
#endif
