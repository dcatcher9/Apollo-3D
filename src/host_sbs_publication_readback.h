/**
 * @file src/host_sbs_publication_readback.h
 * @brief Nonblocking publication receipt readback and the inference completion wake.
 */
#pragma once

#include "cuda_driver_api.h"
#include "host_sbs_gpu_completion_receipt.h"
#include "thread_safe.h"
#include "video_depth_estimator.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <d3d11.h>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <wrl/client.h>

namespace models::host_sbs_publication_readback {
  namespace publication_receipt = host_sbs_gpu_completion_receipt;

  class inference_completion_notification_t {
    using host_function_t = void(__stdcall *)(void *);
    using launch_host_function_t = CUresult(__stdcall *)(CUstream, host_function_t, void *);
    using stream_wait_event_t = CUresult(__stdcall *)(CUstream, CUevent, unsigned);

    struct callback_state_t {
      std::weak_ptr<safe::deadline_waiter_t> waiter;
      std::atomic<unsigned> references {1};
      std::atomic<bool> completed {false};

      void release() noexcept {
        if (references.fetch_sub(1, std::memory_order_acq_rel) == 1) {
          delete this;
        }
      }
    };

    static void __stdcall notify(void *opaque) noexcept;

  public:
    inference_completion_notification_t() = default;
    inference_completion_notification_t(const inference_completion_notification_t &) = delete;
    inference_completion_notification_t &operator=(const inference_completion_notification_t &) = delete;

    ~inference_completion_notification_t();

    bool destroy_stream(cuda_driver_api &cuda) noexcept;

    bool callback_pending() const noexcept;

    std::optional<CUresult> arm(cuda_driver_api &cuda, CUevent event);

    bool wait_until(std::chrono::steady_clock::time_point deadline) noexcept;

  private:
    CUresult create_stream(cuda_driver_api &cuda);

    std::shared_ptr<safe::deadline_waiter_t> waiter_;
    callback_state_t *state_ = nullptr;
    CUstream stream_ = nullptr;
    stream_wait_event_t wait_for_event_ = nullptr;
    launch_host_function_t launch_host_function_ = nullptr;
    bool setup_attempted_ = false;
  };

  // The offsets describe existing GPU records, copied without another shader or CUDA node.
  inline constexpr std::array<std::pair<std::size_t, std::size_t>, 8> publication_copy_ranges {{
    {publication_receipt::transaction_begin, publication_receipt::transaction_word_count},
    {publication_receipt::history_owner_begin, publication_receipt::history_owner_word_count},
    {publication_receipt::subtitle_locator_begin, publication_receipt::subtitle_locator_word_count},
    {publication_receipt::subtitle_condition_begin, publication_receipt::subtitle_condition_word_count},
    {publication_receipt::parallax_state_begin, publication_receipt::parallax_state_word_count},
    {publication_receipt::depth_frame_state_begin, publication_receipt::depth_frame_state_word_count},
    {publication_receipt::depth_health_state_begin, publication_receipt::depth_health_state_word_count},
    {publication_receipt::outcome_counts_begin, publication_receipt::outcome_counts_word_count},
  }};

  class publication_readback_t {
  public:
    publication_readback_t(ID3D11Device *device, ID3D11DeviceContext *context);

    bool submit(
      const std::array<ID3D11Buffer *, publication_copy_ranges.size()> &sources,
      const publication_receipt::expected_t &expected
    );

    publication_receipt_poll_result poll(const std::uint64_t estimator_generation);

    [[nodiscard]] std::size_t pending_count() const noexcept;

  private:
    bool ensure_resources();

    void disable(const std::string_view reason);

    struct slot_t {
      Microsoft::WRL::ComPtr<ID3D11Buffer> staging;
      Microsoft::WRL::ComPtr<ID3D11Query> completion;
      publication_receipt::slot_state_t state;
      std::chrono::steady_clock::time_point sampled_at {};
      bool depth_health_present = false;
      bool outcomes_present = false;
    };

    ID3D11Device *device_;
    ID3D11DeviceContext *context_;
    std::array<slot_t, publication_receipt::slot_count> slots_;
    bool ready_ = false;
    bool disabled_ = false;
    bool failure_pending_ = false;
    std::uint64_t submitted_ = 0u;
    std::uint64_t skipped_ = 0u;
    std::uint64_t decoded_ = 0u;
    std::uint64_t decode_failed_ = 0u;
  };
}  // namespace models::host_sbs_publication_readback
