/**
 * @file src/nvenc/nvenc_d3d11.h
 * @brief Declarations for abstract Direct3D11 NVENC encoder.
 */
#pragma once
#ifdef _WIN32

  // standard includes
  #include <comdef.h>
  #include <d3d11.h>

  // local includes
  #include "nvenc_base.h"
  #include "src/utility.h"

namespace nvenc {

  _COM_SMARTPTR_TYPEDEF(ID3D11Device, IID_ID3D11Device);
  _COM_SMARTPTR_TYPEDEF(ID3D11Texture2D, IID_ID3D11Texture2D);
  _COM_SMARTPTR_TYPEDEF(ID3D11DeviceContext, IID_ID3D11DeviceContext);
  _COM_SMARTPTR_TYPEDEF(ID3D11Query, IID_ID3D11Query);

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
    nvenc_event_wait_result wait_for_async_event(uint32_t timeout_ms) override;
    void *create_flush_event() override;
    nvenc_event_wait_result wait_for_flush_event(uint32_t timeout_ms) override;
    void release_async_event() override;
    void mark_input_producer_end() override;
    input_producer_state poll_input_producer() override;

  private:
    nvenc_event_wait_result wait_for_event(void *event, uint32_t timeout_ms);

    // Event query ended on the encoder device's immediate context after conversion has written
    // the input surface. Conversion and encode share that context on the encode thread.
    ID3D11DeviceContextPtr producer_context;
    ID3D11QueryPtr producer_query;
    bool producer_marked = false;
    bool producer_query_unavailable = false;

    util::safe_ptr_v2<void, BOOL, CloseHandle> owned_async_event;
    util::safe_ptr_v2<void, BOOL, CloseHandle> owned_flush_event;
    HMODULE dll = nullptr;
  };

}  // namespace nvenc
#endif
