#pragma once
// DEFAULT-OFF private NEGATIVE render permission. Never positive/EMPTY authority.
#include "host_sbs_current_source_snapshot_gpu.h"
#include "host_sbs_gpu_completion_receipt.h"

#include <array>
#include <cstddef>
#include <cstdint>
#ifdef _WIN32
  #include <d3d11.h>
  #include <wrl/client.h>
#endif
namespace models::research_source_pattern_retirement_v1 {
  inline constexpr std::uint32_t tag = 0x31524753u;
  inline constexpr std::uint32_t schema = 1u;
  enum class decision_e : std::uint32_t {
    unknown = 0u,
    not_retired = 1u,
    retired = 2u
  };

  struct alignas(16) expected_t {
    std::array<std::uint32_t, 4> control {}, current {}, scope {}, domain {}, source_rect {}, band_field {}, content {}, owner0 {}, owner1 {}, owner2 {};
    bool operator==(const expected_t &) const = default;
  };

  using receipt_t = std::array<std::uint32_t, 64>;
  static_assert(sizeof(expected_t) == 160u && offsetof(expected_t, owner0) == 112u && offsetof(expected_t, owner2) == 144u);

  // SERIALIZATION ONLY. Caller authenticates actual current bundle, exact completed
  // owner publication and source/domain continuity. GPU checks literal CSP/SCT joins.
  inline expected_t make_expected(const research_current_source_snapshot_v1::constants_t &current, const host_sbs_gpu_completion_receipt::expected_t &owner) noexcept {
    const auto split = [](std::uint64_t v) {
      return std::array<std::uint32_t, 2> {std::uint32_t(v), std::uint32_t(v >> 32)};
    };
    expected_t e;
    e.control = {1u, tag, schema, 0u};
    e.current = current.current;
    e.scope = current.scope;
    e.domain = current.domain;
    e.source_rect = current.source_rect;
    e.band_field = current.band_field;
    e.content = current.content;
    const auto gen = split(owner.estimator_generation), pub = split(owner.publication_sequence), frame = split(owner.frame_id), clock = split(owner.observation_timestamp_us), analysis = split(owner.analysis_generation);
    e.owner0 = {gen[0], gen[1], pub[0], pub[1]};
    e.owner1 = {frame[0], frame[1], clock[0], clock[1]};
    e.owner2 = {analysis[0], analysis[1], 0u, 0u};
    return e;
  }
#ifdef _WIN32
  class scoped_ps_negative_t {
    ID3D11DeviceContext *context_ = nullptr;
    Microsoft::WRL::ComPtr<ID3D11Buffer> saved_cb_;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> saved_srv_;

  public:
    scoped_ps_negative_t() = default;
    scoped_ps_negative_t(const scoped_ps_negative_t &) = delete;
    scoped_ps_negative_t &operator=(const scoped_ps_negative_t &) = delete;

    bool bind(ID3D11DeviceContext *c, ID3D11Buffer *expected, ID3D11ShaderResourceView *receipt) noexcept {
      if (context_ || !c || c->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE) {
        return false;
      }
      Microsoft::WRL::ComPtr<ID3D11Predicate> predicate;
      BOOL value = FALSE;
      c->GetPredication(&predicate, &value);
      if (predicate) {
        return false;
      }
      c->PSGetConstantBuffers(5u, 1u, saved_cb_.GetAddressOf());
      c->PSGetShaderResources(10u, 1u, saved_srv_.GetAddressOf());
      context_ = c;
      c->PSSetConstantBuffers(5u, 1u, &expected);
      c->PSSetShaderResources(10u, 1u, &receipt);
      return true;
    }

    ~scoped_ps_negative_t() {
      if (context_) {
        auto *cb = saved_cb_.Get();
        auto *srv = saved_srv_.Get();
        context_->PSSetShaderResources(10u, 1u, &srv);
        context_->PSSetConstantBuffers(5u, 1u, &cb);
      }
    }
  };
#endif
}  // namespace models::research_source_pattern_retirement_v1
