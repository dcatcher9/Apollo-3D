#pragma once
#ifdef _WIN32
  #include <array>
  #include <cstddef>
  #include <d3d11.h>
  #include <utility>
  #include <wrl/client.h>

namespace models::host_sbs_compute_program_lease {
  // Shader objects and their device are immutable. Retain a validated tuple so
  // repeated public raw-pointer calls can reuse those facts, never pixel facts.
  // A different/missing/foreign tuple still takes the complete validation path.
  template<std::size_t N>
  class group_t {
    Microsoft::WRL::ComPtr<ID3D11Device> device_;
    std::array<Microsoft::WRL::ComPtr<ID3D11ComputeShader>, N> shaders_;

  public:
    bool acquire(ID3D11Device *device, const std::array<ID3D11ComputeShader *, N> &requested) {
      bool same = device && device_.Get() == device;
      for (std::size_t i = 0u; i < N; ++i) same = same && requested[i] && requested[i] == shaders_[i].Get();
      if (same) return true;
      if (!device) return false;
      group_t next;
      for (std::size_t i = 0u; i < N; ++i) {
        if (!requested[i]) return false;
        Microsoft::WRL::ComPtr<ID3D11Device> actual;
        requested[i]->GetDevice(&actual);
        if (actual.Get() != device) return false;
        next.shaders_[i] = requested[i];
      }
      next.device_ = device;
      *this = std::move(next);
      return true;
    }
  };
}
#endif
