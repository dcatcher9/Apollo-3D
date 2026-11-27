#include <gtest/gtest.h>
#include "src/platform/windows/capture_dxgi_device.h"

#include <dxgi.h>
#include <utility>
#include <vector>

namespace {
  namespace d = platf::dxgi::detail;

  enum class operation_e {
    query,
    priority,
    latency,
    release,
  };

  class capture_dxgi_device_t final: public IDXGIDevice1 {
  public:
    HRESULT query_result = S_OK;
    HRESULT priority_result = S_OK;
    HRESULT latency_result = S_OK;
    GUID requested_iid {};
    std::vector<operation_e> operations;
    ULONG references = 1;
    unsigned add_ref_calls = 0;
    INT priority = 0;
    UINT latency = 0;

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **out) override {
      operations.push_back(operation_e::query);
      requested_iid = iid;
      *out = nullptr;
      if (FAILED(query_result)) {
        return query_result;
      }
      if (!IsEqualIID(iid, IID_IUnknown) && !IsEqualIID(iid, IID_IDXGIObject) && !IsEqualIID(iid, IID_IDXGIDevice) && !IsEqualIID(iid, IID_IDXGIDevice1)) {
        return E_NOINTERFACE;
      }
      *out = static_cast<IDXGIDevice1 *>(this);
      AddRef();
      return query_result;
    }

    ULONG STDMETHODCALLTYPE AddRef() override {
      ++add_ref_calls;
      return ++references;
    }

    ULONG STDMETHODCALLTYPE Release() override {
      operations.push_back(operation_e::release);
      return --references;
    }

    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void *) override {
      return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown *) override {
      return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT *, void *) override {
      return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE GetParent(REFIID, void **out) override {
      *out = nullptr;
      return E_NOINTERFACE;
    }

    HRESULT STDMETHODCALLTYPE GetAdapter(IDXGIAdapter **) override {
      return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE CreateSurface(const DXGI_SURFACE_DESC *, UINT, DXGI_USAGE, const DXGI_SHARED_RESOURCE *, IDXGISurface **) override {
      return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE QueryResourceResidency(IUnknown *const *, DXGI_RESIDENCY *, UINT) override {
      return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE SetGPUThreadPriority(INT value) override {
      operations.push_back(operation_e::priority);
      priority = value;
      return priority_result;
    }

    HRESULT STDMETHODCALLTYPE GetGPUThreadPriority(INT *) override {
      return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE SetMaximumFrameLatency(UINT value) override {
      operations.push_back(operation_e::latency);
      latency = value;
      return latency_result;
    }

    HRESULT STDMETHODCALLTYPE GetMaximumFrameLatency(UINT *) override {
      return E_NOTIMPL;
    }
  };

  void expect_configured_and_released(const capture_dxgi_device_t &device) {
    EXPECT_TRUE(IsEqualIID(device.requested_iid, IID_IDXGIDevice1));
    EXPECT_EQ(device.operations, (std::vector<operation_e> {
                                   operation_e::query,
                                   operation_e::priority,
                                   operation_e::latency,
                                   operation_e::release,
                                 }));
    EXPECT_EQ(device.priority, 7);
    EXPECT_EQ(device.latency, 1u);
    EXPECT_EQ(device.add_ref_calls, 1u);
    EXPECT_EQ(device.references, 1u);
  }
}  // namespace

TEST(CaptureDxgiDeviceTest, QueriesDevice1OnceAndSetsPriorityThenLatency) {
  capture_dxgi_device_t device;

  EXPECT_TRUE(d::configure_capture_dxgi_device(&device));

  expect_configured_and_released(device);
}

TEST(CaptureDxgiDeviceTest, QueryFailureStopsBeforeSetters) {
  capture_dxgi_device_t device;
  device.query_result = E_NOINTERFACE;

  EXPECT_FALSE(d::configure_capture_dxgi_device(&device));

  EXPECT_TRUE(IsEqualIID(device.requested_iid, IID_IDXGIDevice1));
  EXPECT_EQ(device.operations, (std::vector<operation_e> {operation_e::query}));
  EXPECT_EQ(device.add_ref_calls, 0u);
  EXPECT_EQ(device.references, 1u);
}

TEST(CaptureDxgiDeviceTest, SetterFailuresContinueAndReleaseTheInterface) {
  const std::pair<HRESULT, HRESULT> failures[] = {
    {E_ACCESSDENIED, S_OK},
    {S_OK, DXGI_ERROR_DEVICE_REMOVED},
    {E_FAIL, E_FAIL},
  };
  for (const auto &[priority_result, latency_result] : failures) {
    SCOPED_TRACE(::testing::Message() << "priority=" << priority_result << ", latency=" << latency_result);
    capture_dxgi_device_t device;
    device.priority_result = priority_result;
    device.latency_result = latency_result;

    EXPECT_TRUE(d::configure_capture_dxgi_device(&device));

    expect_configured_and_released(device);
  }
}
