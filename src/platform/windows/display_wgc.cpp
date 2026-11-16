/**
 * @file src/platform/windows/display_wgc.cpp
 * @brief Definitions for WinRT Windows.Graphics.Capture API
 */
// platform includes
#include <dxgi1_2.h>

// Gross hack to work around MINGW-packages#22160
#define ____FIReference_1_boolean_INTERFACE_DEFINED__

#include <Windows.Graphics.Capture.Interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <winrt/windows.foundation.h>
#include <winrt/windows.foundation.metadata.h>
#include <winrt/windows.graphics.directx.direct3d11.h>

// local includes
#include "display.h"
#include "misc.h"
#include "src/logging.h"

namespace platf {
  using namespace std::literals;
}

namespace winrt {
  using namespace Windows::Foundation;
  using namespace Windows::Foundation::Metadata;
  using namespace Windows::Graphics::Capture;
  using namespace Windows::Graphics::DirectX::Direct3D11;
}  // namespace winrt

namespace platf::dxgi {
  namespace d3d11_interop = ::Windows::Graphics::DirectX::Direct3D11;

  bool test_wgc_capture(output_t &output) {
    try {
      if (!winrt::GraphicsCaptureSession::IsSupported()) {
        return false;
      }

      DXGI_OUTPUT_DESC output_desc {};
      if (FAILED(output->GetDesc(&output_desc))) {
        return false;
      }

      auto monitor_factory = winrt::get_activation_factory<winrt::GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
      winrt::GraphicsCaptureItem item {nullptr};
      return monitor_factory &&
             SUCCEEDED(monitor_factory->CreateForMonitor(
               output_desc.Monitor,
               winrt::guid_of<winrt::IGraphicsCaptureItem>(),
               winrt::put_abi(item)
             ));
    } catch (const winrt::hresult_error &) {
      return false;
    }
  }

  wgc_capture_t::wgc_capture_t() {
    InitializeConditionVariable(&frame_present_cv);
  }

  wgc_capture_t::~wgc_capture_t() {
    drop_kept_frame();
    // Close() can throw (for example after device loss); a destructor must not propagate it.
    try {
      if (frame_pool && frame_arrived_token) {
        // The handler captures a raw `this`. Stop new deliveries before tearing anything down.
        frame_pool.FrameArrived(frame_arrived_token);
        frame_arrived_token = {};
      }
      if (capture_session) {
        capture_session.Close();
      }
      if (frame_pool) {
        frame_pool.Close();
      }
    } catch (winrt::hresult_error &e) {
      BOOST_LOG(warning) << "Failed to close the capture session cleanly: [0x"sv << util::hex(e.code()).to_string_view() << ']';
    }
    // A delivery that was already running finishes under the lock before this clears its frame.
    AcquireSRWLockExclusive(&frame_lock);
    produced_frame = nullptr;
    ReleaseSRWLockExclusive(&frame_lock);
    item = nullptr;
    capture_session = nullptr;
    frame_pool = nullptr;
  }

  /**
   * @brief Initialize the Windows.Graphics.Capture backend.
   * @return 0 on success, -1 on failure.
   */
  int wgc_capture_t::init(display_base_t *display, const ::video::config_t &config) {
    HRESULT status;
    dxgi::dxgi_t dxgi;
    winrt::com_ptr<::IInspectable> d3d_comhandle;
    try {
      if (!winrt::GraphicsCaptureSession::IsSupported()) {
        BOOST_LOG(error) << "Screen capture is not supported on this device for this release of Windows!"sv;
        return -1;
      }
      if (FAILED(status = display->device->QueryInterface(IID_IDXGIDevice, (void **) &dxgi))) {
        BOOST_LOG(error) << "Failed to query DXGI interface from device [0x"sv << util::hex(status).to_string_view() << ']';
        return -1;
      }
      if (FAILED(status = ::CreateDirect3D11DeviceFromDXGIDevice(*&dxgi, d3d_comhandle.put()))) {
        BOOST_LOG(error) << "Failed to query WinRT DirectX interface from device [0x"sv << util::hex(status).to_string_view() << ']';
        return -1;
      }
    } catch (winrt::hresult_error &e) {
      BOOST_LOG(error) << "Screen capture is not supported on this device for this release of Windows: failed to acquire device: [0x"sv << util::hex(e.code()).to_string_view() << ']';
      return -1;
    }

    DXGI_OUTPUT_DESC output_desc {};
    uwp_device = d3d_comhandle.as<winrt::IDirect3DDevice>();
    status = display->output->GetDesc(&output_desc);
    if (FAILED(status)) {
      BOOST_LOG(error) << "Screen capture is not supported on this device for this release of Windows: failed to query display description [0x"sv
                       << util::hex(status).to_string_view() << ']';
      return -1;
    }

    auto monitor_factory = winrt::get_activation_factory<winrt::GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
    if (monitor_factory == nullptr || FAILED(status = monitor_factory->CreateForMonitor(output_desc.Monitor, winrt::guid_of<winrt::IGraphicsCaptureItem>(), winrt::put_abi(item)))) {
      BOOST_LOG(error) << "Screen capture is not supported on this device for this release of Windows: failed to acquire display: [0x"sv << util::hex(status).to_string_view() << ']';
      return -1;
    }

    if (config.dynamicRange) {
      display->capture_format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    } else {
      display->capture_format = DXGI_FORMAT_B8G8R8A8_UNORM;
    }

    try {
      // Three buffers: while an export owns the output, capture keeps one frame (keep_frame())
      // and produced_frame may hold another between ticks. The third is free for the newest
      // composition, so the frame copied when the export releases the output is never stale.
      frame_pool = winrt::Direct3D11CaptureFramePool::CreateFreeThreaded(uwp_device, static_cast<winrt::Windows::Graphics::DirectX::DirectXPixelFormat>(display->capture_format), 3, item.Size());
      capture_session = frame_pool.CreateCaptureSession(item);
      frame_arrived_token = frame_pool.FrameArrived({this, &wgc_capture_t::on_frame_arrived});
    } catch (winrt::hresult_error &e) {
      BOOST_LOG(error) << "Screen capture is not supported on this device for this release of Windows: failed to create capture session: [0x"sv << util::hex(e.code()).to_string_view() << ']';
      return -1;
    }
    try {
      if (winrt::ApiInformation::IsPropertyPresent(L"Windows.Graphics.Capture.GraphicsCaptureSession", L"IsBorderRequired")) {
        capture_session.IsBorderRequired(false);
      } else {
        BOOST_LOG(warning) << "Can't disable colored border around capture area on this version of Windows";
      }
    } catch (winrt::hresult_error &e) {
      BOOST_LOG(warning) << "Screen capture may not be fully supported on this device for this release of Windows: failed to disable border around capture area: [0x"sv << util::hex(e.code()).to_string_view() << ']';
    }
    try {
      if (winrt::ApiInformation::IsPropertyPresent(L"Windows.Graphics.Capture.GraphicsCaptureSession", L"MinUpdateInterval")) {
        capture_session.MinUpdateInterval(winrt::TimeSpan {10000000 / (config.framerate * 2)});
      } else {
        BOOST_LOG(warning) << "Can't set MinUpdateInterval";
      }
    } catch (winrt::hresult_error &e) {
      BOOST_LOG(warning) << "Screen capture may not be fully supported on this device for this release of Windows: failed to set MinUpdateInterval: [0x"sv << util::hex(e.code()).to_string_view() << ']';
    }
    try {
      capture_session.StartCapture();
    } catch (winrt::hresult_error &e) {
      BOOST_LOG(error) << "Screen capture is not supported on this device for this release of Windows: failed to start capture: [0x"sv << util::hex(e.code()).to_string_view() << ']';
      return -1;
    }
    return 0;
  }

  /**
   * This function runs in a separate thread spawned by the frame pool and is a producer of frames.
   * To maintain parity with the original display interface, this frame will be consumed by the capture thread.
   * Acquire a read-write lock, make the produced frame available to the capture thread, then wake the capture thread.
   */
  void wgc_capture_t::on_frame_arrived(winrt::Direct3D11CaptureFramePool const &sender, winrt::IInspectable const &) {
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame frame {nullptr};
    try {
      frame = sender.TryGetNextFrame();
    } catch (winrt::hresult_error &e) {
      BOOST_LOG(warning) << "Failed to capture frame: "sv << e.code();
      return;
    }
    if (frame != nullptr) {
      AcquireSRWLockExclusive(&frame_lock);
      if (produced_frame) {
        produced_frame.Close();
      }

      produced_frame = frame;
      ReleaseSRWLockExclusive(&frame_lock);
      WakeConditionVariable(&frame_present_cv);
    }
  }

  /**
   * @brief Get the next frame from the producer thread.
   * If not available, the capture thread blocks until one is, or the wait times out.
   * @param timeout how long to wait for the next frame
   * @param out a texture containing the frame just captured
   * @param out_time the timestamp of the frame just captured
   */
  capture_e wgc_capture_t::next_frame(std::chrono::milliseconds timeout, ID3D11Texture2D **out, detail::wgc_timestamp_t &out_time) {
    // this CONSUMER runs in the capture thread
    release_frame();

    AcquireSRWLockExclusive(&frame_lock);
    if (produced_frame == nullptr && SleepConditionVariableSRW(&frame_present_cv, &frame_lock, timeout.count(), 0) == 0) {
      ReleaseSRWLockExclusive(&frame_lock);
      if (GetLastError() == ERROR_TIMEOUT) {
        return capture_e::timeout;
      } else {
        return capture_e::error;
      }
    }
    if (produced_frame) {
      consumed_frame = produced_frame;
      produced_frame = nullptr;
    }
    ReleaseSRWLockExclusive(&frame_lock);
    if (consumed_frame == nullptr) {  // spurious wakeup
      return capture_e::timeout;
    }

    // These WinRT calls throw on failure (a lost device, for example). Report that as a
    // capture error so the caller reinitializes, instead of terminating the process.
    try {
      auto capture_access = consumed_frame.Surface().try_as<d3d11_interop::IDirect3DDxgiInterfaceAccess>();
      if (capture_access == nullptr || FAILED(capture_access->GetInterface(IID_ID3D11Texture2D, (void **) out)) || !*out) {
        return capture_e::error;
      }
      out_time = consumed_frame.SystemRelativeTime();
    } catch (winrt::hresult_error &e) {
      BOOST_LOG(warning) << "Failed to read the captured frame: [0x"sv << util::hex(e.code()).to_string_view() << ']';
      return capture_e::error;
    }
    return capture_e::ok;
  }

  capture_e wgc_capture_t::release_frame() {
    if (consumed_frame != nullptr) {
      try {
        consumed_frame.Close();
      } catch (winrt::hresult_error &) {
        // Nothing more can be released; the frame object is dropped below either way.
      }
      consumed_frame = nullptr;
    }
    return capture_e::ok;
  }

  void wgc_capture_t::keep_frame() {
    if (consumed_frame == nullptr) {
      return;
    }
    drop_kept_frame();
    kept_frame = consumed_frame;
    consumed_frame = nullptr;
  }

  bool wgc_capture_t::kept_frame_texture(ID3D11Texture2D **out, detail::wgc_timestamp_t &out_time) {
    if (kept_frame == nullptr) {
      return false;
    }
    try {
      auto capture_access = kept_frame.Surface().try_as<d3d11_interop::IDirect3DDxgiInterfaceAccess>();
      if (capture_access == nullptr || FAILED(capture_access->GetInterface(IID_ID3D11Texture2D, (void **) out))) {
        drop_kept_frame();
        return false;
      }
      out_time = kept_frame.SystemRelativeTime();
    } catch (winrt::hresult_error &) {
      drop_kept_frame();
      return false;
    }
    return true;
  }

  void wgc_capture_t::drop_kept_frame() {
    if (kept_frame != nullptr) {
      try {
        kept_frame.Close();
      } catch (winrt::hresult_error &) {
        // As in release_frame(): the frame object is dropped below either way.
      }
      kept_frame = nullptr;
    }
  }

  int wgc_capture_t::set_cursor_visible(bool x) {
    try {
      if (capture_session.IsCursorCaptureEnabled() != x) {
        capture_session.IsCursorCaptureEnabled(x);
      }
      return 0;
    } catch (winrt::hresult_error &) {
      return -1;
    }
  }

}  // namespace platf::dxgi
