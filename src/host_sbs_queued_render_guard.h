#pragma once
// PRIVATE queued-render binding only. This is not completed owner/semantic authority.
#include <array>
#include <cstddef>
#include <cstdint>
#ifdef _WIN32
  #include <d3d11.h>
  #include <wrl/client.h>
#endif

namespace models::research_queued_render_guard_v1 {
  inline constexpr std::uint32_t tag = 0x31475251u;  // QRG1
  inline constexpr std::uint32_t schema = 1u;
  inline constexpr std::uint32_t work_cookie = 0x6f435257u;

  struct alignas(16) expected_t {
    std::array<std::uint32_t, 4> control {}, root {}, observation {};
    bool operator==(const expected_t &) const = default;
  };

  static_assert(sizeof(expected_t) == 48u && offsetof(expected_t, root) == 16u && offsetof(expected_t, observation) == 32u);

  // Serialization only. The host must own the exact pending source and transaction contents.
  // No queued draw is authorized by an absent/disabled CB, even though legacy mode remains intact.
  inline expected_t make_expected(std::uint64_t token, std::uint32_t expected_work, std::uint64_t observation) noexcept {
    expected_t result;
    result.control = {1u, tag, schema, 0u};
    result.root = {std::uint32_t(token), std::uint32_t(token >> 32u), expected_work, expected_work == 0u ? 0u : expected_work ^ work_cookie};
    result.observation = {std::uint32_t(observation), std::uint32_t(observation >> 32u), 0u, 0u};
    return result;
  }

#ifdef _WIN32
  class scoped_binding_t {
    ID3D11DeviceContext *context_ = nullptr;
    Microsoft::WRL::ComPtr<ID3D11Buffer> previous_cb_;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> previous_srv_;

  public:
    scoped_binding_t() = default;
    scoped_binding_t(const scoped_binding_t &) = delete;
    scoped_binding_t &operator=(const scoped_binding_t &) = delete;

    bool bind(ID3D11DeviceContext *context, ID3D11Buffer *expected, ID3D11ShaderResourceView *transaction) noexcept {
      if (context_ || !context || context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE) {
        return false;
      }
      Microsoft::WRL::ComPtr<ID3D11Predicate> predicate;
      BOOL value = FALSE;
      context->GetPredication(&predicate, &value);
      if (predicate) {
        return false;
      }
      context->PSGetConstantBuffers(5u, 1u, previous_cb_.GetAddressOf());
      context->PSGetShaderResources(10u, 1u, previous_srv_.GetAddressOf());
      context_ = context;
      context->PSSetConstantBuffers(5u, 1u, &expected);
      context->PSSetShaderResources(10u, 1u, &transaction);
      return true;
    }

    ~scoped_binding_t() {
      if (context_) {
        auto *srv = previous_srv_.Get();
        auto *cb = previous_cb_.Get();
        context_->PSSetShaderResources(10u, 1u, &srv);
        context_->PSSetConstantBuffers(5u, 1u, &cb);
      }
    }
  };
#endif
}  // namespace models::research_queued_render_guard_v1
