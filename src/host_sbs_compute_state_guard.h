/**
 * @file src/host_sbs_compute_state_guard.h
 * @brief Scoped save and restore of the compute-stage bindings a Host SBS pass touches.
 */
#pragma once
#include <array>
#include <d3d11.h>
#include <wrl/client.h>

namespace models::host_sbs_compute_state {
  /**
   * Saves compute SRVs [0, Srvs), UAVs [0, Uavs) and constant buffers [0, Constants), the
   * compute shader and the predicate, then clears the predicate so the
   * pass is never skipped. Restoring first unbinds those slots, so a saved view is not refused
   * because the pass still holds its resource on another slot. Slots past the counts are never
   * read or written. No Host SBS shader uses class linkage, so none is saved.
   */
  template<UINT Srvs, UINT Uavs, UINT Constants>
  class guard_t {
  public:
    explicit guard_t(ID3D11DeviceContext *context):
        context_(context) {
      std::array<ID3D11ShaderResourceView *, Srvs> srvs {};
      std::array<ID3D11UnorderedAccessView *, Uavs> uavs {};
      std::array<ID3D11Buffer *, Constants> constants {};
      ID3D11ComputeShader *shader = nullptr;
      ID3D11Predicate *predicate = nullptr;
      context_->CSGetShaderResources(0u, Srvs, srvs.data());
      context_->CSGetUnorderedAccessViews(0u, Uavs, uavs.data());
      context_->CSGetConstantBuffers(0u, Constants, constants.data());
      context_->CSGetShader(&shader, nullptr, nullptr);
      context_->GetPredication(&predicate, &predicate_value_);
      context_->SetPredication(nullptr, FALSE);
      for (UINT i = 0u; i < Srvs; ++i) {
        srvs_[i].Attach(srvs[i]);
      }
      for (UINT i = 0u; i < Uavs; ++i) {
        uavs_[i].Attach(uavs[i]);
      }
      for (UINT i = 0u; i < Constants; ++i) {
        constants_[i].Attach(constants[i]);
      }
      shader_.Attach(shader);
      predicate_.Attach(predicate);
    }

    ~guard_t() {
      unbind(context_);
      std::array<ID3D11ShaderResourceView *, Srvs> srvs {};
      std::array<ID3D11UnorderedAccessView *, Uavs> uavs {};
      std::array<ID3D11Buffer *, Constants> constants {};
      for (UINT i = 0u; i < Srvs; ++i) {
        srvs[i] = srvs_[i].Get();
      }
      for (UINT i = 0u; i < Uavs; ++i) {
        uavs[i] = uavs_[i].Get();
      }
      for (UINT i = 0u; i < Constants; ++i) {
        constants[i] = constants_[i].Get();
      }
      context_->CSSetShaderResources(0u, Srvs, srvs.data());
      context_->CSSetUnorderedAccessViews(0u, Uavs, uavs.data(), nullptr);
      context_->CSSetConstantBuffers(0u, Constants, constants.data());
      context_->CSSetShader(shader_.Get(), nullptr, 0u);
      context_->SetPredication(predicate_.Get(), predicate_value_);
    }

    guard_t(const guard_t &) = delete;
    guard_t &operator=(const guard_t &) = delete;

    /** Clear the guarded SRV, UAV and constant-buffer slots. */
    static void unbind(ID3D11DeviceContext *context) {
      std::array<ID3D11ShaderResourceView *, Srvs> srvs {};
      std::array<ID3D11UnorderedAccessView *, Uavs> uavs {};
      std::array<ID3D11Buffer *, Constants> constants {};
      context->CSSetShaderResources(0u, Srvs, srvs.data());
      context->CSSetUnorderedAccessViews(0u, Uavs, uavs.data(), nullptr);
      context->CSSetConstantBuffers(0u, Constants, constants.data());
    }

  private:
    ID3D11DeviceContext *context_;
    std::array<Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>, Srvs> srvs_;
    std::array<Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView>, Uavs> uavs_;
    std::array<Microsoft::WRL::ComPtr<ID3D11Buffer>, Constants> constants_;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> shader_;
    Microsoft::WRL::ComPtr<ID3D11Predicate> predicate_;
    BOOL predicate_value_ = FALSE;
  };
}  // namespace models::host_sbs_compute_state
