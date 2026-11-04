#pragma once
#ifdef _WIN32
  #include <d3d11.h>

namespace models::current_outline_stage_observer {
  enum class stage_e { source5_end, cbc_end, f2_end };
  struct observer_t {
    using callback_t = void (*)(void *, ID3D11DeviceContext *, stage_e) noexcept;
    void *user = nullptr;
    callback_t callback = nullptr;
    void mark(ID3D11DeviceContext *context, stage_e stage) const noexcept {
      if (callback) callback(user, context, stage);
    }
  };
}
#endif
