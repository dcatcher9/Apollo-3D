#include <gtest/gtest.h>
#ifdef _WIN32
#include <cstring>
#include <initializer_list>
#include <optional>

// Match the existing Windows platform fixture prerequisite before display.h.
typedef long NTSTATUS;
#include "src/platform/windows/display.h"

namespace {
  namespace d = platf::dxgi::detail;
  DXGI_OUTDUPL_FRAME_INFO present(UINT count) {
    DXGI_OUTDUPL_FRAME_INFO info {};
    info.LastPresentTime.QuadPart = 100;
    info.AccumulatedFrames = count;
    return info;
  }
  auto history() { return std::make_shared<d::ddup_damage_history_t>(); }

  class metadata_duplication_t final: public IDXGIOutputDuplication {
  public:
    std::vector<DXGI_OUTDUPL_MOVE_RECT> moves;
    std::vector<RECT> dirty;
    std::optional<UINT> move_report, dirty_report;
    HRESULT move_status = S_OK, dirty_status = S_OK;
    UINT move_capacity = 0u, dirty_capacity = 0u;
    unsigned move_calls = 0u, dirty_calls = 0u;
    void *move_buffer = nullptr;

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void **out) override {
      *out = nullptr;
      return E_NOINTERFACE;
    }

    ULONG STDMETHODCALLTYPE AddRef() override {
      return ++references_;
    }

    ULONG STDMETHODCALLTYPE Release() override {
      const auto references = --references_;
      if (!references) {
        delete this;
      }
      return references;
    }

    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void *) override {
      return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown *) override {
      return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT *, void *) override {
      return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE GetParent(REFIID, void **out) override {
      *out = nullptr;
      return E_NOINTERFACE;
    }

    void STDMETHODCALLTYPE GetDesc(DXGI_OUTDUPL_DESC *) override {}

    HRESULT STDMETHODCALLTYPE AcquireNextFrame(UINT, DXGI_OUTDUPL_FRAME_INFO *, IDXGIResource **) override {
      return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE GetFrameDirtyRects(UINT capacity, RECT *out, UINT *required) override {
      ++dirty_calls;
      dirty_capacity = capacity;
      *required = dirty_report.value_or(static_cast<UINT>(dirty.size() * sizeof(RECT)));
      if (dirty_status != S_OK) {
        return dirty_status;
      }
      if (!out || capacity < dirty.size() * sizeof(RECT)) {
        return DXGI_ERROR_MORE_DATA;
      }
      if (!dirty.empty()) {
        std::memcpy(out, dirty.data(), dirty.size() * sizeof(RECT));
      }
      return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetFrameMoveRects(UINT capacity, DXGI_OUTDUPL_MOVE_RECT *out, UINT *required) override {
      ++move_calls;
      move_capacity = capacity;
      move_buffer = out;
      *required = move_report.value_or(static_cast<UINT>(moves.size() * sizeof(DXGI_OUTDUPL_MOVE_RECT)));
      if (move_status != S_OK) {
        return move_status;
      }
      if (!out || capacity < moves.size() * sizeof(DXGI_OUTDUPL_MOVE_RECT)) {
        return DXGI_ERROR_MORE_DATA;
      }
      if (!moves.empty()) {
        std::memcpy(out, moves.data(), moves.size() * sizeof(DXGI_OUTDUPL_MOVE_RECT));
      }
      return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetFramePointerShape(UINT, void *, UINT *, DXGI_OUTDUPL_POINTER_SHAPE_INFO *) override {
      return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE MapDesktopSurface(DXGI_MAPPED_RECT *) override {
      return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE UnMapDesktopSurface() override {
      return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE ReleaseFrame() override {
      return S_OK;
    }

  private:
    ULONG references_ = 1u;
  };

  void expect_rects(const d::ddup_damage_update_t &update, std::initializer_list<RECT> expected) {
    ASSERT_TRUE(update.known);
    ASSERT_EQ(update.rects.size(), expected.size());
    std::size_t index = 0u;
    for (const auto &rect : expected) {
      const auto &actual = update.rects[index++];
      EXPECT_EQ(actual.left, rect.left);
      EXPECT_EQ(actual.top, rect.top);
      EXPECT_EQ(actual.right, rect.right);
      EXPECT_EQ(actual.bottom, rect.bottom);
    }
  }
}  // namespace

TEST(WindowsDdupCaptureIntervalTest, ActualOnePresentIsSeparateFromDamageMetadata) {
  const auto info = present(1u);
  EXPECT_TRUE(d::ddup_frame_has_one_desktop_present(info));
  EXPECT_FALSE(d::ddup_frame_damage_metadata_available(info));
}
TEST(WindowsDdupCaptureIntervalTest, CoalescedAndMissingPresentCountsAreUnknown) {
  EXPECT_FALSE(d::ddup_frame_has_one_desktop_present(present(0u)));
  EXPECT_FALSE(d::ddup_frame_has_one_desktop_present(present(2u)));
  EXPECT_FALSE(d::ddup_frame_has_one_desktop_present(present(UINT_MAX)));
}
TEST(WindowsDdupCaptureIntervalTest, CursorOnlyAndInvalidPresentTimesAreUnknown) {
  auto info = present(1u);
  info.LastPresentTime.QuadPart = 0;
  EXPECT_FALSE(d::ddup_frame_has_one_desktop_present(info));
  info.LastPresentTime.QuadPart = -1;
  EXPECT_FALSE(d::ddup_frame_has_one_desktop_present(info));
}
TEST(WindowsDdupCaptureIntervalTest, ProtectedContentCannotSupplyContinuity) {
  auto info = present(1u);
  info.ProtectedContentMaskedOut = TRUE;
  EXPECT_FALSE(d::ddup_frame_has_one_desktop_present(info));
}
TEST(WindowsDdupCaptureIntervalTest, DamageOnlyAuthoredUpdatesDoNotGainPresentEvidence) {
  auto h = history();
  const auto a = h->commit({true, {}});
  const auto b = h->commit({true, {}});
  EXPECT_FALSE(b.single_desktop_present);
  EXPECT_FALSE(d::ddup_adjacent_desktop_present_observed(a, b));
}
TEST(WindowsDdupCaptureIntervalTest, OneObservedPresentDoesNotRequireKnownDirtyRects) {
  auto h = history();
  const auto a = h->commit({false, {}, false});
  const auto b = h->commit({false, {}, true});
  EXPECT_TRUE(d::ddup_adjacent_desktop_present_observed(a, b));
  EXPECT_EQ(d::query_ddup_damage_between(a, b, RECT {0, 0, 100, 100}),
            d::ddup_damage_intersection_e::unknown);
}
TEST(WindowsDdupCaptureIntervalTest, SkippedCommittedSurfacesNeverProveCompleteObservation) {
  auto h = history();
  const auto a = h->commit({true, {}, true});
  (void)h->commit({true, {}, true});
  const auto c = h->commit({true, {}, true});
  EXPECT_FALSE(d::ddup_adjacent_desktop_present_observed(a, c));
}
TEST(WindowsDdupCaptureIntervalTest, SameCaptureRetryIsNotANewPresent) {
  auto h = history();
  const auto a = h->commit({true, {}, true});
  EXPECT_FALSE(d::ddup_adjacent_desktop_present_observed(a, a));
}
TEST(WindowsDdupCaptureIntervalTest, RegressedAndCrossNamespaceSnapshotsAreUnknown) {
  auto h = history();
  const auto a = h->commit({true, {}, true});
  const auto b = h->commit({true, {}, true});
  auto other = history();
  const auto c = other->commit({true, {}, true});
  EXPECT_FALSE(d::ddup_adjacent_desktop_present_observed(b, a));
  EXPECT_FALSE(d::ddup_adjacent_desktop_present_observed(a, c));
}
TEST(WindowsDdupCaptureIntervalTest, MissingAndOverflowedIdentitiesAreUnknown) {
  auto h = history();
  d::ddup_damage_snapshot_t a {h, UINT64_MAX, true};
  d::ddup_damage_snapshot_t b {h, 0u, true};
  EXPECT_FALSE(d::ddup_adjacent_desktop_present_observed(a, b));
  EXPECT_FALSE(d::ddup_adjacent_desktop_present_observed(std::nullopt, a));
  EXPECT_FALSE(d::ddup_adjacent_desktop_present_observed(b, a));
}

TEST(WindowsDdupMetadataTest, ReusesScratchWhileCoverageAndHistoryKeepTheirOwnStorage) {
  platf::dxgi::duplication_t duplication;
  auto *native = new metadata_duplication_t;
  duplication.dup.reset(native);
  native->moves = {{{5, 5}, {20, 20, 30, 30}}};
  native->dirty = {{70, 70, 75, 75}};
  auto info = present(1u);
  info.TotalMetadataBufferSize = d::ddup_damage_metadata_byte_budget;

  const auto first = duplication.damage_update(info, DXGI_MODE_ROTATION_IDENTITY, 100, 100);
  expect_rects(first, {{70, 70, 75, 75}, {5, 5, 15, 15}, {20, 20, 30, 30}});
  const auto scratch = native->move_buffer;
  EXPECT_EQ(native->move_capacity, info.TotalMetadataBufferSize);
  EXPECT_EQ(native->dirty_capacity, info.TotalMetadataBufferSize - sizeof(DXGI_OUTDUPL_MOVE_RECT));
  auto retained = history();
  const auto baseline = retained->commit({true, {}});
  const auto published = retained->commit(first);

  native->moves.clear();
  native->dirty = {{1, 1, 2, 2}};
  info.TotalMetadataBufferSize = sizeof(RECT);
  const auto second = duplication.damage_update(info, DXGI_MODE_ROTATION_IDENTITY, 100, 100);
  expect_rects(second, {{1, 1, 2, 2}});
  EXPECT_EQ(native->move_buffer, scratch);
  EXPECT_EQ(native->move_capacity, sizeof(RECT));
  EXPECT_EQ(native->dirty_capacity, sizeof(RECT));

  // No records are written on this call: old moves/dirty bytes must never become current damage.
  native->dirty.clear();
  info.TotalMetadataBufferSize = d::ddup_damage_metadata_byte_budget;
  expect_rects(duplication.damage_update(info, DXGI_MODE_ROTATION_IDENTITY, 100, 100), {});
  EXPECT_EQ(native->move_buffer, scratch);
  expect_rects(first, {{70, 70, 75, 75}, {5, 5, 15, 15}, {20, 20, 30, 30}});
  expect_rects(second, {{1, 1, 2, 2}});
  EXPECT_EQ(d::query_ddup_damage_between(baseline, published, RECT {20, 20, 30, 30}), d::ddup_damage_intersection_e::changed);
  EXPECT_EQ(d::query_ddup_damage_between(baseline, published, RECT {0, 0, 3, 3}), d::ddup_damage_intersection_e::unchanged);
}

TEST(WindowsDdupMetadataTest, RejectsMalformedCurrentCountsFailuresAndGeometryAfterScratchReuse) {
  platf::dxgi::duplication_t duplication;
  auto *native = new metadata_duplication_t;
  duplication.dup.reset(native);
  native->moves = {{{5, 5}, {20, 20, 30, 30}}};
  native->dirty = {{70, 70, 75, 75}};
  auto info = present(1u);
  info.TotalMetadataBufferSize = sizeof(DXGI_OUTDUPL_MOVE_RECT) + sizeof(RECT);
  ASSERT_TRUE(duplication.damage_update(info, DXGI_MODE_ROTATION_IDENTITY, 100, 100).known);
  const auto expect_unknown = [&]() {
    const auto update = duplication.damage_update(info, DXGI_MODE_ROTATION_IDENTITY, 100, 100);
    EXPECT_FALSE(update.known);
    EXPECT_TRUE(update.rects.empty());
  };

  native->move_report = info.TotalMetadataBufferSize + 1u;
  expect_unknown();
  native->move_report = 1u;
  expect_unknown();
  native->move_report.reset();
  native->dirty_report = sizeof(RECT) + 1u;
  expect_unknown();
  native->dirty_report = 1u;
  expect_unknown();
  native->dirty_report.reset();
  for (const auto status : {E_FAIL, S_FALSE, DXGI_ERROR_MORE_DATA}) {
    native->move_status = status;
    expect_unknown();
    native->move_status = S_OK;
    native->dirty_status = status;
    expect_unknown();
    native->dirty_status = S_OK;
  }
  native->dirty = {{0, 0, 101, 1}};
  expect_unknown();
  native->dirty = {{70, 70, 75, 75}};
  native->moves.front().SourcePoint.x = -1;
  expect_unknown();
  native->moves.front().SourcePoint.x = 5;
  EXPECT_TRUE(duplication.damage_update(info, DXGI_MODE_ROTATION_IDENTITY, 100, 100).known);
}

TEST(WindowsDdupMetadataTest, UnavailableOrOverBudgetMetadataNeverReachesTheDriver) {
  platf::dxgi::duplication_t duplication;
  auto *native = new metadata_duplication_t;
  duplication.dup.reset(native);
  auto info = present(1u);
  info.TotalMetadataBufferSize = d::ddup_damage_metadata_byte_budget + 1u;
  EXPECT_FALSE(duplication.damage_update(info, DXGI_MODE_ROTATION_IDENTITY, 100, 100).known);
  info.TotalMetadataBufferSize = 0u;
  EXPECT_FALSE(duplication.damage_update(info, DXGI_MODE_ROTATION_IDENTITY, 100, 100).known);
  info.TotalMetadataBufferSize = sizeof(RECT);
  info.ProtectedContentMaskedOut = TRUE;
  EXPECT_FALSE(duplication.damage_update(info, DXGI_MODE_ROTATION_IDENTITY, 100, 100).known);
  info.ProtectedContentMaskedOut = FALSE;
  EXPECT_FALSE(duplication.damage_update(info, DXGI_MODE_ROTATION_ROTATE90, 100, 100).known);
  EXPECT_EQ(native->move_calls, 0u);
  EXPECT_EQ(native->dirty_calls, 0u);
}

#endif // _WIN32
