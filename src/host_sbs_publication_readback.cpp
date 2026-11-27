/**
 * @file src/host_sbs_publication_readback.cpp
 * @brief Nonblocking publication receipt readback and the inference completion wake.
 */
#include "host_sbs_publication_readback.h"

#include "logging.h"
#include "platform/common.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <string>
#include <thread>
#include <windows.h>

using namespace std::literals;

namespace models::host_sbs_publication_readback {
  void __stdcall inference_completion_notification_t::notify(void *opaque) noexcept {
    auto *state = static_cast<callback_state_t *>(opaque);
    state->completed.store(true, std::memory_order_release);
    if (const auto waiter = state->waiter.lock()) {
      waiter->notify();
    }
    state->release();
  }

  inference_completion_notification_t::~inference_completion_notification_t() {
    waiter_.reset();
    if (state_) {
      state_->release();
    }
  }

  bool inference_completion_notification_t::destroy_stream(cuda_driver_api &cuda) noexcept {
    return !stream_ || cuda.cuStreamDestroy(stream_) == CUDA_SUCCESS;
  }

  bool inference_completion_notification_t::callback_pending() const noexcept {
    return state_ && state_->references.load(std::memory_order_acquire) > 1;
  }

  std::optional<CUresult> inference_completion_notification_t::arm(cuda_driver_api &cuda, CUevent event) {
    if (!setup_attempted_) {
      setup_attempted_ = true;
      const auto created = create_stream(cuda);
      if (created != CUDA_SUCCESS) {
        return created;
      }
    }
    if (!stream_ || callback_pending()) {
      return std::nullopt;
    }
    // The notification stream waits for the recorded root. Future root inference can
    // advance independently of CPU dispatch of this host function.
    const auto waiting = wait_for_event_(stream_, event, 0u);
    if (waiting != CUDA_SUCCESS) {
      return waiting;
    }
    // The previous callback released its reference after signalling, so it can no
    // longer publish completion for this arm.
    state_->completed.store(false, std::memory_order_relaxed);
    // CUDA can suppress host functions after a context error. Preserve the reference
    // after an ambiguous launch failure as well as during ordinary delayed dispatch.
    state_->references.fetch_add(1, std::memory_order_relaxed);
    return launch_host_function_(stream_, &notify, state_);
  }

  bool inference_completion_notification_t::wait_until(std::chrono::steady_clock::time_point deadline) noexcept {
    // A callback can signal after its wait has already reached the cutoff.
    // Reuse the same cutoff if a wake precedes this arm's callback.
    do {
      if (!waiter_->wait_until(deadline)) {
        return false;
      }
    } while (!state_->completed.load(std::memory_order_acquire) &&
             std::chrono::steady_clock::now() < deadline);
    return true;
  }

  CUresult inference_completion_notification_t::create_stream(cuda_driver_api &cuda) {
    wait_for_event_ = std::bit_cast<stream_wait_event_t>(
      GetProcAddress(cuda.hMod, "cuStreamWaitEvent")
    );
    launch_host_function_ = std::bit_cast<launch_host_function_t>(
      GetProcAddress(cuda.hMod, "cuLaunchHostFunc")
    );
    if (!wait_for_event_ || !launch_host_function_) {
      return CUDA_SUCCESS;
    }
    waiter_ = platf::create_deadline_waiter();
    if (!waiter_) {
      return CUDA_SUCCESS;
    }
    // Lazy creation keeps this stream outside the inference graph capture.
    const auto created = cuda.cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING);
    if (created == CUDA_SUCCESS) {
      state_ = new callback_state_t {waiter_};
    }
    return created;
  }

  publication_readback_t::publication_readback_t(ID3D11Device *device, ID3D11DeviceContext *context):
      device_(device), context_(context) {}

  bool publication_readback_t::submit(
    const std::array<ID3D11Buffer *, publication_copy_ranges.size()> &sources,
    const publication_receipt::expected_t &expected
  ) {
    if (!publication_receipt::valid_expected(expected) ||
        std::any_of(sources.begin(), sources.begin() + 6u, [](auto *source) { return !source; })) {
      ++skipped_;
      return false;
    }
    if (!ensure_resources()) {
      ++skipped_;
      return false;
    }
    for (auto &slot : slots_) {
      if (!slot.state.reserve(expected)) {
        continue;
      }
      slot.sampled_at = std::chrono::steady_clock::now();
      slot.depth_health_present = sources[6u] != nullptr;
      slot.outcomes_present = sources[7u] != nullptr;
      for (std::size_t index = 0u; index < sources.size(); ++index) {
        if (!sources[index]) {
          continue;
        }
        const auto [begin, count] = publication_copy_ranges[index];
        const D3D11_BOX range {
          0u, 0u, 0u, static_cast<UINT>(count * sizeof(std::uint32_t)), 1u, 1u,
        };
        context_->CopySubresourceRegion(
          slot.staging.Get(), 0u, static_cast<UINT>(begin * sizeof(std::uint32_t)),
          0u, 0u, sources[index], 0u, &range
        );
      }
      // This proves the complete D3D publication and its immutable snapshot, not merely the
      // CUDA branch/unmap event. The next transaction may now reuse every source record.
      context_->End(slot.completion.Get());
      ++submitted_;
      return true;
    }
    ++skipped_;
    return false;
  }

  publication_receipt_poll_result publication_readback_t::poll(const std::uint64_t estimator_generation) {
    publication_receipt_poll_result result;
    for (auto &slot : slots_) {
      if (!slot.state.pending()) {
        continue;
      }
      BOOL complete = FALSE;
      const auto ready = context_->GetData(
        slot.completion.Get(), &complete, sizeof(complete), D3D11_ASYNC_GETDATA_DONOTFLUSH
      );
      if (ready == S_FALSE || (ready == S_OK && !complete)) {
        continue;
      }
      if (FAILED(ready)) {
        disable("nonblocking completion query failed");
        break;
      }
      D3D11_MAPPED_SUBRESOURCE mapped {};
      const auto mapped_result = context_->Map(
        slot.staging.Get(), 0u, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped
      );
      if (mapped_result == DXGI_ERROR_WAS_STILL_DRAWING) {
        continue;
      }
      if (FAILED(mapped_result)) {
        disable("nonblocking snapshot map failed");
        break;
      }
      if (!mapped.pData) {
        context_->Unmap(slot.staging.Get(), 0u);
        disable("snapshot map returned no data");
        break;
      }
      publication_receipt::snapshot_words_t words {};
      std::memcpy(words.data(), mapped.pData, sizeof(words));
      context_->Unmap(slot.staging.Get(), 0u);
      // Diagnostics remain independently useful when publication authentication fails.
      // Presence bits belong to this slot; absent sources can never expose stale bytes.
      const auto expected = *slot.state.expected();
      if (expected.estimator_generation == estimator_generation) {
        result.diagnostics[result.diagnostic_count++] = publication_receipt::decode_diagnostics(
          words, expected, slot.depth_health_present, slot.outcomes_present, slot.sampled_at
        );
      }
      if (auto receipt = slot.state.retire(words, estimator_generation)) {
        result.receipts[result.count++] = *receipt;
        ++decoded_;
      } else {
        result.failed = true;
        ++decode_failed_;
      }
    }
    // Recycled slots need not retire in submission order. Cumulative diagnostic consumers
    // require monotonic sequence order, independently of rendering's exact-current check.
    for (std::size_t index = 1u; index < result.diagnostic_count; ++index) {
      for (auto cursor = index; cursor > 0u &&
           result.diagnostics[cursor].expected.publication_sequence <
             result.diagnostics[cursor - 1u].expected.publication_sequence; --cursor) {
        std::swap(result.diagnostics[cursor], result.diagnostics[cursor - 1u]);
      }
    }
    result.transport_failed = std::exchange(failure_pending_, false);
    result.failed = result.transport_failed || result.failed;
    result.submitted = submitted_;
    result.skipped = skipped_;
    result.decoded = decoded_;
    result.decode_failed = decode_failed_;
    return result;
  }

  std::size_t publication_readback_t::pending_count() const noexcept {
    return static_cast<std::size_t>(std::count_if(
      slots_.begin(), slots_.end(), [](const auto &slot) { return slot.state.pending(); }
    ));
  }

  bool publication_readback_t::ensure_resources() {
    if (disabled_) {
      return false;
    }
    if (ready_) {
      return true;
    }
    if (!device_ || !context_) {
      disable("D3D context is unavailable");
      return false;
    }
    D3D11_BUFFER_DESC buffer_desc {};
    buffer_desc.ByteWidth = publication_receipt::snapshot_byte_count;
    buffer_desc.Usage = D3D11_USAGE_STAGING;
    buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    const D3D11_QUERY_DESC query_desc {D3D11_QUERY_EVENT, 0u};
    for (auto &slot : slots_) {
      if (FAILED(device_->CreateBuffer(&buffer_desc, nullptr, slot.staging.ReleaseAndGetAddressOf())) ||
          FAILED(device_->CreateQuery(&query_desc, slot.completion.ReleaseAndGetAddressOf()))) {
        disable("staging/query allocation failed");
        return false;
      }
    }
    ready_ = true;
    return true;
  }

  void publication_readback_t::disable(const std::string_view reason) {
    disabled_ = true;
    ready_ = false;
    failure_pending_ = true;
    for (auto &slot : slots_) {
      slot.completion.Reset();
      slot.staging.Reset();
      slot.state.reset();
    }
    BOOST_LOG(warning) << "Host SBS asynchronous publication receipts are unavailable ("
                       << reason << "); adaptive reuse is unavailable; using force inference.";
  }
}  // namespace models::host_sbs_publication_readback

namespace models {
#ifdef SUNSHINE_TESTS
  detail::publication_receipt_readback_test_result detail::publication_receipt_readback_for_test(
    ID3D11Device *device,
    ID3D11DeviceContext *context,
    const host_sbs_gpu_completion_receipt::snapshot_words_t &snapshot,
    const host_sbs_gpu_completion_receipt::expected_t &expected,
    const std::uint64_t poll_estimator_generation,
    const bool depth_health_present,
    const bool outcomes_present
  ) {
    using namespace host_sbs_publication_readback;
    publication_receipt_readback_test_result result;
    if (!device || !context) {
      return result;
    }
    std::array<Microsoft::WRL::ComPtr<ID3D11Buffer>, publication_copy_ranges.size()> buffers;
    std::array<ID3D11Buffer *, publication_copy_ranges.size()> sources {};
    for (std::size_t index = 0u; index < buffers.size(); ++index) {
      const auto [begin, count] = publication_copy_ranges[index];
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = static_cast<UINT>(count * sizeof(std::uint32_t));
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      desc.StructureByteStride = sizeof(std::uint32_t);
      const D3D11_SUBRESOURCE_DATA initial {snapshot.data() + begin, 0u, 0u};
      if (FAILED(device->CreateBuffer(&desc, &initial, buffers[index].ReleaseAndGetAddressOf()))) {
        return result;
      }
      sources[index] = buffers[index].Get();
    }
    result.resources_created = true;
    if (!depth_health_present) {
      sources[6u] = nullptr;
    }
    if (!outcomes_present) {
      sources[7u] = nullptr;
    }
    publication_readback_t readback {device, context};
    for (std::size_t index = 0u; index <= publication_receipt::slot_count; ++index) {
      auto publication = expected;
      publication.publication_sequence += index;
      readback.submit(sources, publication);
    }
    // These writes are ordered after the three immutable snapshots. No later map may observe
    // the poisoned mutable source records, and the fourth submission must not replace a slot.
    const publication_receipt::snapshot_words_t poison {};
    for (auto *source : sources) {
      if (source) {
        context->UpdateSubresource(source, 0u, nullptr, poison.data(), 0u, 0u);
      }
    }
    const auto generation = poll_estimator_generation == 0u ?
                              expected.estimator_generation : poll_estimator_generation;
    const auto poll = [&] {
      const auto batch = readback.poll(generation);
      ++result.poll_count;
      result.result.failed = result.result.failed || batch.failed;
      result.result.transport_failed = result.result.transport_failed || batch.transport_failed;
      result.result.submitted = batch.submitted;
      result.result.skipped = batch.skipped;
      result.result.decoded = batch.decoded;
      result.result.decode_failed = batch.decode_failed;
      for (std::size_t index = 0u; index < batch.diagnostic_count; ++index) {
        if (result.result.diagnostic_count == result.result.diagnostics.size()) {
          result.result.failed = true;
          break;
        }
        result.result.diagnostics[result.result.diagnostic_count++] = batch.diagnostics[index];
      }
      for (std::size_t index = 0u; index < batch.count; ++index) {
        if (result.result.count == result.result.receipts.size()) {
          result.result.failed = true;
          break;
        }
        result.result.receipts[result.result.count++] = batch.receipts[index];
      }
    };
    poll();
    result.pending_before_flush = readback.pending_count() != 0u;
    // Production never flushes or waits. A WARP unit test has no Present/encode owner to drive
    // submission, so this test-only boundary explicitly drives its bounded completion loop.
    context->Flush();
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (readback.pending_count() != 0u && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(1ms);
      poll();
    }
    result.completed = readback.pending_count() == 0u;
    return result;
  }
#endif
}  // namespace models
