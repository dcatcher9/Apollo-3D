// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The capture owner's slot and evaluation fixtures, shared by its white-box
// regressions (test_streamline_depth_capture.cpp) and the seam
// characterization (test_game3d_seam_characterization.cpp). They write the
// owner's globals through its test-access header, with no device. Their
// stamps are game time: the probe view's test clock once a fixture engaged
// it (test_game3d_clock.h), else the real tick.
#ifndef SUNSHINE_STREAMLINE_PROBE_TEST
  #error "Built only into the Streamline probe tests' view of the production sources"
#endif
#include "streamline_depth_capture_test_access.h"
#include "test_game3d_clock.h"

#include <cstdint>

namespace sunshine_streamline::depth_capture::fixture {
  // No capture slots, no API evaluations and an empty presentation scope.
  // Each owner's state is cleared by its reset in the test-access header.
  inline void clear_captures() {
    reset_pool();
    reset_evaluations();
    reset_scope();
  }

  // clear_captures, every provider's last attempt unavailable, no queue
  // owners, and capture requested.
  inline void reset_owner() {
    clear_captures();
    reset_attempts();
    reset_queues();
    requested = true;
  }

  // Counts count Presents (on_present's present(), counted before each
  // Present's selection): an evaluation's view or snapshot stamped before
  // them is the frame the Present being decided presents only for a count
  // of one (presented_frame).
  inline void presents_since(unsigned count) {
    scope.presents.fetch_add(count);
  }

  // A queue's private fence as selection reads it: only GetCompletedValue,
  // which returns completed (after reads reads, completed_later when set:
  // the GPU completes the work between two looks of one selection).
  struct completion_fence final : ID3D12Fence {
    UINT64 completed{}, completed_later{};
    unsigned reads{}, later_after{};
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void **out) override { if (out) *out = nullptr; return E_NOINTERFACE; }
    ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
    ULONG STDMETHODCALLTYPE Release() override { return 1; }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT *, void *) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void *) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown *) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetName(LPCWSTR) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID, void **out) override { if (out) *out = nullptr; return E_NOINTERFACE; }
    UINT64 STDMETHODCALLTYPE GetCompletedValue() override {
      return completed_later && reads++ >= later_after ? completed_later : completed;
    }
    HRESULT STDMETHODCALLTYPE SetEventOnCompletion(UINT64, HANDLE) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE Signal(UINT64) override { return E_NOTIMPL; }
  };

  // One submitted, successful capture recorded on queue, and the API
  // evaluation that nominated it (begin_evaluation, which superseded may make
  // replace another source of the same viewport).
  inline slot &make_capture(unsigned index, std::uint64_t queue, provider_kind provider, std::uint64_t epoch, std::uint64_t sequence, std::uint64_t source, std::uint32_t viewport = 0, std::uint64_t superseded = UINT64_MAX) {
    auto &value = capture_pool.slots[index];
    value = {};
    value.id = ++serial;
    value.queue = queue;
    value.producer_fence = 1;
    value.finished = value.success = value.producer_submitted = true;
    value.metadata.provider = provider;
    value.metadata.source_id = source;
    value.metadata.viewport = viewport;
    value.metadata.epoch = epoch;
    value.metadata.sequence = sequence;
    value.metadata.tick = sunshine_game3d_fixture::fixture_clock().now();
    begin_evaluation(epoch, sequence, viewport, provider, source, superseded);
    return value;
  }
}  // namespace sunshine_streamline::depth_capture::fixture
