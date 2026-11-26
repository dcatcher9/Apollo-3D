#pragma once

#include <d3d11.h>
#include <wrl/client.h>

namespace d3d11_device_release_fixture {

  /**
   * D3D11 destroys a device only after the state its immediate context holds is cleared and its
   * deferred destruction is flushed after the last release. A test that draws on its own device
   * and just drops it keeps that device, its VRAM and about thirty driver threads for the rest of
   * the process. Declare this before every other D3D11 object of the test, so it is destroyed
   * last, and give it the immediate context once the device exists.
   */
  class device_release_t {
  public:
    device_release_t() = default;
    device_release_t(const device_release_t &) = delete;
    device_release_t &operator=(const device_release_t &) = delete;

    ~device_release_t() {
      if (context_) {
        context_->ClearState();
        context_->Flush();
      }
    }

    void watch(ID3D11DeviceContext *context) {
      context_ = context;
    }

  private:
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context_;
  };

}  // namespace d3d11_device_release_fixture
