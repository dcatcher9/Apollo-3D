/**
 * @file src/platform/windows/microphone.cpp
 * @brief WASAPI render into an explicitly selected virtual microphone cable.
 */
#define INITGUID
#include "src/microphone.h"

#include <algorithm>
#include <audioclient.h>
#include <avrt.h>
#include <cstring>
#include <functiondiscoverykeys_devpkey.h>
#include <memory>
#include <mmdeviceapi.h>
#include <string>

namespace microphone {
  namespace {
    template<class T>
    struct release_t {
      void operator()(T *value) const {
        if (value) {
          value->Release();
        }
      }
    };
    template<class T>
    using com_ptr_t = std::unique_ptr<T, release_t<T>>;

    std::wstring wide(const std::string &value) {
      const auto size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), nullptr, 0);
      if (size <= 0) {
        return {};
      }
      std::wstring result(size, L'\0');
      MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), result.data(), size);
      return result;
    }

    std::wstring device_property(IMMDevice *device, REFPROPERTYKEY key) {
      IPropertyStore *raw = nullptr;
      if (FAILED(device->OpenPropertyStore(STGM_READ, &raw))) {
        return {};
      }
      com_ptr_t<IPropertyStore> properties(raw);
      PROPVARIANT value;
      PropVariantInit(&value);
      std::wstring result;
      if (SUCCEEDED(properties->GetValue(key, &value)) && value.vt == VT_LPWSTR && value.pwszVal) {
        result = value.pwszVal;
      }
      PropVariantClear(&value);
      return result;
    }

    std::wstring adapter_name(IMMDevice *device) {
      return device_property(device, PKEY_DeviceInterface_FriendlyName);
    }

    com_ptr_t<IMMDevice> resolve_endpoint(IMMDeviceEnumerator *enumerator, const std::wstring &name_or_id, std::string &error) {
      IMMDevice *raw = nullptr;
      if (!name_or_id.empty() && SUCCEEDED(enumerator->GetDevice(name_or_id.c_str(), &raw))) {
        return com_ptr_t<IMMDevice>(raw);
      }
      IMMDeviceCollection *collection_raw = nullptr;
      if (name_or_id.empty() || FAILED(enumerator->EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE, &collection_raw))) {
        error = "The configured microphone cable endpoint does not exist";
        return {};
      }
      com_ptr_t<IMMDeviceCollection> collection(collection_raw);
      UINT count = 0;
      if (FAILED(collection->GetCount(&count))) {
        error = "Could not enumerate microphone cable endpoints";
        return {};
      }
      com_ptr_t<IMMDevice> match;
      for (UINT index = 0; index < count; ++index) {
        raw = nullptr;
        if (FAILED(collection->Item(index, &raw))) {
          continue;
        }
        com_ptr_t<IMMDevice> candidate(raw);
        if (device_property(candidate.get(), PKEY_Device_FriendlyName) != name_or_id) {
          continue;
        }
        if (match) {
          error = "Multiple active audio endpoints have that exact name; configure the microphone cable endpoint ID";
          return {};
        }
        match = std::move(candidate);
      }
      if (!match) {
        error = "The configured microphone cable endpoint does not exist";
      }
      return match;
    }

    bool virtual_cable_pair(IMMDeviceEnumerator *enumerator, IMMDevice *render) {
      const auto adapter = adapter_name(render);
      // Supported and identifiable microphone routes only. A selected physical
      // speaker must never become a fallback for missing microphone hardware.
      if (adapter.find(L"VB-Audio") == std::wstring::npos) {
        return false;
      }
      IMMDeviceCollection *raw = nullptr;
      if (FAILED(enumerator->EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE, &raw))) {
        return false;
      }
      com_ptr_t<IMMDeviceCollection> devices(raw);
      UINT count = 0;
      if (FAILED(devices->GetCount(&count))) {
        return false;
      }
      for (UINT index = 0; index < count; ++index) {
        IMMDevice *capture_raw = nullptr;
        if (SUCCEEDED(devices->Item(index, &capture_raw))) {
          com_ptr_t<IMMDevice> capture(capture_raw);
          if (adapter_name(capture.get()) == adapter) {
            return true;
          }
        }
      }
      return false;
    }

    class wasapi_sink_t final: public pcm_sink_t {
    public:
      ~wasapi_sink_t() override {
        if (client_) {
          client_->Stop();
        }
        render_.reset();
        client_.reset();
        if (mmcss_) {
          AvRevertMmThreadCharacteristics(mmcss_);
        }
        if (com_initialized_) {
          CoUninitialize();
        }
      }

      bool open(const std::string &endpoint, std::string &error) override {
        if (endpoint.empty()) {
          error = "Configure the VB-Audio virtual cable render endpoint name or ID for client microphone forwarding";
          return false;
        }
        const auto initialized = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (FAILED(initialized)) {
          error = "Could not initialize COM on microphone worker";
          return false;
        }
        com_initialized_ = true;
        IMMDeviceEnumerator *enumerator_raw = nullptr;
        if (FAILED(CoCreateInstance(CLSID_MMDeviceEnumerator, nullptr, CLSCTX_ALL, IID_IMMDeviceEnumerator, reinterpret_cast<void **>(&enumerator_raw)))) {
          error = "Windows audio endpoint enumeration is unavailable";
          return false;
        }
        com_ptr_t<IMMDeviceEnumerator> enumerator(enumerator_raw);
        auto device = resolve_endpoint(enumerator.get(), wide(endpoint), error);
        if (!device) {
          return false;
        }
        DWORD state = 0;
        IMMEndpoint *flow_raw = nullptr;
        if (FAILED(device->GetState(&state)) || state != DEVICE_STATE_ACTIVE || FAILED(device->QueryInterface(IID_IMMEndpoint, reinterpret_cast<void **>(&flow_raw)))) {
          error = "The configured microphone cable endpoint is not active";
          return false;
        }
        com_ptr_t<IMMEndpoint> flow(flow_raw);
        EDataFlow direction;
        if (FAILED(flow->GetDataFlow(&direction)) || direction != eRender || !virtual_cable_pair(enumerator.get(), device.get())) {
          error = "Microphone forwarding requires an active VB-Audio render/capture cable pair; physical speakers are not supported";
          return false;
        }
        IAudioClient *client_raw = nullptr;
        if (FAILED(device->Activate(IID_IAudioClient, CLSCTX_ALL, nullptr, reinterpret_cast<void **>(&client_raw)))) {
          error = "Could not activate the microphone cable render endpoint";
          return false;
        }
        client_.reset(client_raw);
        // WASAPI performs channel/rate conversion to the endpoint's shared mix
        // format. Do not modify its format, volume, or Windows default role.
        const WAVEFORMATEX format {WAVE_FORMAT_PCM, 1, sample_rate, sample_rate * 2, 2, 16, 0};
        if (FAILED(client_->Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY, 1000000, 0, &format, nullptr)) || FAILED(client_->GetBufferSize(&buffer_frames_)) || buffer_frames_ == 0) {
          error = "The microphone cable cannot accept shared 48 kHz PCM audio";
          return false;
        }
        IAudioRenderClient *render_raw = nullptr;
        if (FAILED(client_->GetService(IID_IAudioRenderClient, reinterpret_cast<void **>(&render_raw)))) {
          error = "The microphone cable render service is unavailable";
          return false;
        }
        render_.reset(render_raw);
        if (FAILED(client_->Start())) {
          error = "Could not start the microphone cable audio stream";
          return false;
        }
        DWORD task_index = 0;
        mmcss_ = AvSetMmThreadCharacteristicsW(L"Audio", &task_index);
        error.clear();
        return true;
      }

      int write(std::span<const std::int16_t> samples) override {
        UINT32 padding = 0;
        if (!client_ || !render_ || FAILED(client_->GetCurrentPadding(&padding)) || padding > buffer_frames_) {
          return -1;
        }
        const auto count = static_cast<UINT32>(std::min<std::size_t>(samples.size(), buffer_frames_ - padding));
        if (count == 0) {
          return 0;
        }
        BYTE *buffer = nullptr;
        const auto status = render_->GetBuffer(count, &buffer);
        if (status == AUDCLNT_E_BUFFER_TOO_LARGE) {
          return 0;
        }
        if (FAILED(status)) {
          return -1;
        }
        std::memcpy(buffer, samples.data(), count * sizeof(std::int16_t));
        return SUCCEEDED(render_->ReleaseBuffer(count, 0)) ? static_cast<int>(count) : -1;
      }

    private:
      bool com_initialized_ {};
      com_ptr_t<IAudioClient> client_;
      com_ptr_t<IAudioRenderClient> render_;
      UINT32 buffer_frames_ {};
      HANDLE mmcss_ {};
    };
  }  // namespace

  std::unique_ptr<pcm_sink_t> make_wasapi_sink() {
    return std::make_unique<wasapi_sink_t>();
  }
}  // namespace microphone
