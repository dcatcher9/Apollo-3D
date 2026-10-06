/**
 * @file src/nvenc/nvenc_d3d11.h
 * @brief Declarations for abstract Direct3D11 NVENC encoder.
 */
#pragma once
#ifdef _WIN32

  // standard includes
  #include <array>
  #include <comdef.h>
  #include <d3d11_4.h>

  // local includes
  #include "nvenc_base.h"
  #include "src/utility.h"

namespace nvenc {

  _COM_SMARTPTR_TYPEDEF(ID3D11Device, IID_ID3D11Device);
  _COM_SMARTPTR_TYPEDEF(ID3D11Device5, IID_ID3D11Device5);
  _COM_SMARTPTR_TYPEDEF(ID3D11Texture2D, IID_ID3D11Texture2D);
  _COM_SMARTPTR_TYPEDEF(ID3D11DeviceContext, IID_ID3D11DeviceContext);
  _COM_SMARTPTR_TYPEDEF(ID3D11DeviceContext4, IID_ID3D11DeviceContext4);
  _COM_SMARTPTR_TYPEDEF(ID3D11Fence, IID_ID3D11Fence);

  /**
   * @brief Abstract Direct3D11 NVENC encoder.
   *        Encapsulates common code used by native and interop implementations.
   */
  class nvenc_d3d11: public nvenc_base {
  public:
    explicit nvenc_d3d11(NV_ENC_DEVICE_TYPE device_type);
    ~nvenc_d3d11();

    /**
     * @brief Get input surface texture.
     * @return Input surface texture.
     */
    virtual ID3D11Texture2D *get_input_texture() = 0;

  protected:
    bool init_library() override;
    void *create_completion_event(unsigned slot) override;
    nvenc_event_wait_result wait_for_async_event(void *event, uint32_t timeout_ms) override;
    void *create_flush_event() override;
    nvenc_event_wait_result wait_for_flush_event(uint32_t timeout_ms) override;
    void release_async_event() override;
    void mark_input_producer_end(unsigned slot) override;
    input_producer_state poll_input_producer(unsigned slot) override;

  private:
    nvenc_event_wait_result wait_for_event(void *event, uint32_t timeout_ms);

    // Fence signaled on the encoder device's immediate context after the work that writes a
    // picture's input (conversion, and its copy into the picture's own input). The submitting
    // encode thread signals; a retrieving thread may read the fence, never the context.
    ID3D11DeviceContext4Ptr producer_context;
    ID3D11FencePtr producer_fence;
    std::uint64_t producer_value = 0;
    std::array<std::uint64_t, max_pipeline_depth> producer_marks {};
    bool producer_fence_unavailable = false;

    util::safe_ptr_v2<void, BOOL, CloseHandle> owned_async_event;
    std::array<util::safe_ptr_v2<void, BOOL, CloseHandle>, max_pipeline_depth> owned_completion_events;
    util::safe_ptr_v2<void, BOOL, CloseHandle> owned_flush_event;
    HMODULE dll = nullptr;
  };

}  // namespace nvenc
#endif
