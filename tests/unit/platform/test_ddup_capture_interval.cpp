#include <gtest/gtest.h>
#ifdef _WIN32
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <optional>
#include <utility>
#include <wrl/client.h>

// Match the existing Windows platform fixture prerequisite before display.h.
typedef long NTSTATUS;
#include "src/platform/windows/display.h"

namespace platf::dxgi {
  // Reuse the capture backend's production shader compiler and cursor blend state.
  blob_t compile_vertex_shader(LPCSTR file);
  blob_t compile_pixel_shader(LPCSTR file);
  blend_t make_blend(device_t::pointer device, bool enable, bool invert);
}  // namespace platf::dxgi

namespace {
  namespace dxgi = platf::dxgi;
  namespace d = platf::dxgi::detail;
  using Microsoft::WRL::ComPtr;

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
    std::optional<DXGI_OUTDUPL_FRAME_INFO> next_frame;
    ComPtr<IDXGIResource> next_surface;
    static constexpr std::uint32_t cursor_bgra = 0xffe1b643u;
    std::uint32_t pointer_shape_bgra = cursor_bgra;

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

    HRESULT STDMETHODCALLTYPE AcquireNextFrame(UINT, DXGI_OUTDUPL_FRAME_INFO *info, IDXGIResource **surface) override {
      if (!next_frame) {
        return DXGI_ERROR_WAIT_TIMEOUT;
      }
      *info = *next_frame;
      next_frame.reset();
      *surface = next_surface.Detach();
      return S_OK;
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

    HRESULT STDMETHODCALLTYPE GetFramePointerShape(UINT capacity, void *data, UINT *required, DXGI_OUTDUPL_POINTER_SHAPE_INFO *shape) override {
      *required = sizeof(cursor_bgra);
      if (capacity < *required) {
        return DXGI_ERROR_MORE_DATA;
      }
      *shape = {DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR, 1u, 1u, sizeof(cursor_bgra), {0, 0}};
      std::memcpy(data, &pointer_shape_bgra, sizeof(pointer_shape_bgra));
      return S_OK;
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

  class WindowsDdupSnapshotTest: public testing::Test {
  protected:
    static constexpr UINT width = 16u;
    static constexpr UINT height = 8u;
    using pixels_t = std::array<std::uint32_t, width * height>;
    using img_t = std::shared_ptr<platf::img_t>;

    dxgi::display_ddup_vram_t display;
    metadata_duplication_t *native = nullptr;  // owned by display.dup

    unsigned image_pulls = 0u;
    bool allow_image = true;
    LONGLONG qpc_stamp = 0;
    LONGLONG qpc_step = 0;

    void SetUp() override {
      constexpr auto level = D3D_FEATURE_LEVEL_11_0;
      const auto status = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT, &level, 1, D3D11_SDK_VERSION,
        &display.device, nullptr, &display.device_ctx);
      if (FAILED(status)) {
        GTEST_SKIP() << "D3D11 WARP unavailable";
      }
      display.width = display.width_before_rotation = width;
      display.height = display.height_before_rotation = height;
      display.display_rotation = DXGI_MODE_ROTATION_IDENTITY;
      display.capture_format = DXGI_FORMAT_UNKNOWN;
      display.add_damage_observer();
      display.next_image_id = 1u;
      native = new metadata_duplication_t;
      native->dirty = {{1, 0, 2, 1}};
      display.dup.dup.reset(native);

      auto vertex = dxgi::compile_vertex_shader(SUNSHINE_SHADERS_DIR "/cursor_vs.hlsl");
      auto pixel = dxgi::compile_pixel_shader(SUNSHINE_SHADERS_DIR "/cursor_ps.hlsl");
      ASSERT_TRUE(vertex);
      ASSERT_TRUE(pixel);
      ASSERT_EQ(display.device->CreateVertexShader(vertex->GetBufferPointer(), vertex->GetBufferSize(), nullptr, &display.cursor_vs), S_OK);
      ASSERT_EQ(display.device->CreatePixelShader(pixel->GetBufferPointer(), pixel->GetBufferSize(), nullptr, &display.cursor_ps), S_OK);
      const std::array<std::int32_t, 4u> rotation {};
      D3D11_BUFFER_DESC rotation_desc {};
      rotation_desc.ByteWidth = sizeof(rotation);
      rotation_desc.Usage = D3D11_USAGE_IMMUTABLE;
      rotation_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      const D3D11_SUBRESOURCE_DATA rotation_data {rotation.data(), 0u, 0u};
      ComPtr<ID3D11Buffer> rotation_buffer;
      ASSERT_EQ(display.device->CreateBuffer(&rotation_desc, &rotation_data, &rotation_buffer), S_OK);
      display.device_ctx->VSSetConstantBuffers(2u, 1u, rotation_buffer.GetAddressOf());
      display.blend_alpha = dxgi::make_blend(display.device.get(), true, false);
      display.blend_invert = dxgi::make_blend(display.device.get(), true, true);
      display.blend_disable = dxgi::make_blend(display.device.get(), false, false);
      ASSERT_TRUE(display.blend_alpha && display.blend_invert && display.blend_disable);

      D3D11_SAMPLER_DESC sampler {};
      sampler.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
      sampler.AddressU = sampler.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
      sampler.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
      sampler.ComparisonFunc = D3D11_COMPARISON_NEVER;
      sampler.MaxLOD = D3D11_FLOAT32_MAX;
      ASSERT_EQ(display.device->CreateSamplerState(&sampler, &display.sampler_linear), S_OK);
      display.device_ctx->PSSetSamplers(0u, 1u, &display.sampler_linear);
      display.device_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

      // Well-separated QPC times give deterministic timestamp ordering even when all desktop
      // acquisitions are issued in rapid succession on the same thread.
      LARGE_INTEGER qpc {}, frequency {};
      ASSERT_TRUE(QueryPerformanceCounter(&qpc));
      ASSERT_TRUE(QueryPerformanceFrequency(&frequency));
      qpc_stamp = qpc.QuadPart - frequency.QuadPart * 2;
      qpc_step = frequency.QuadPart / 20;
    }

    static pixels_t desktop_pixels(std::uint32_t first) {
      pixels_t result {};
      for (auto &pixel : result) {
        pixel = first++;
      }
      return result;
    }

    ComPtr<IDXGIResource> desktop_surface(const pixels_t &pixels, UINT surface_width = width, DXGI_FORMAT format = DXGI_FORMAT_B8G8R8A8_UNORM) {
      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = surface_width;
      desc.Height = height;
      desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1u;
      desc.Format = format;
      desc.Usage = D3D11_USAGE_IMMUTABLE;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      const D3D11_SUBRESOURCE_DATA initial {pixels.data(), width * sizeof(pixels_t::value_type), 0u};
      ComPtr<ID3D11Texture2D> texture;
      ComPtr<IDXGIResource> resource;
      if (SUCCEEDED(display.device->CreateTexture2D(&desc, &initial, &texture))) {
        EXPECT_EQ(texture.As(&resource), S_OK);
      }
      return resource;
    }

    DXGI_OUTDUPL_FRAME_INFO desktop_present() {
      auto info = present(1u);
      info.LastPresentTime.QuadPart = qpc_stamp += qpc_step;
      info.TotalMetadataBufferSize = sizeof(RECT);
      return info;
    }

    void pointer_update(DXGI_OUTDUPL_FRAME_INFO &info, POINT position, bool visible, bool new_shape = false) {
      info.LastMouseUpdateTime.QuadPart = qpc_stamp += qpc_step;
      info.PointerPosition.Position = position;
      info.PointerPosition.Visible = visible;
      info.PointerShapeBufferSize = new_shape ? sizeof(metadata_duplication_t::cursor_bgra) : 0u;
    }

    platf::capture_e snapshot(DXGI_OUTDUPL_FRAME_INFO info, const ComPtr<IDXGIResource> &surface, img_t &image) {
      native->next_frame = info;
      native->next_surface = surface;
      return display.snapshot([&](img_t &slot) {
        ++image_pulls;
        if (!allow_image) {
          return false;
        }
        slot = display.alloc_img();
        return true;
      }, image, std::chrono::milliseconds {0}, true);
    }

    std::optional<pixels_t> read_image(const img_t &image) {
      if (!image || !image->data) {
        return {};
      }
      auto *texture = reinterpret_cast<ID3D11Texture2D *>(image->data);
      D3D11_TEXTURE2D_DESC desc {};
      texture->GetDesc(&desc);
      if (desc.Width != width || desc.Height != height || desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM) {
        return {};
      }
      desc.Usage = D3D11_USAGE_STAGING;
      desc.BindFlags = desc.MiscFlags = 0u;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Texture2D> staging;
      ComPtr<IDXGIKeyedMutex> lock;
      if (FAILED(display.device->CreateTexture2D(&desc, nullptr, &staging)) ||
          FAILED(texture->QueryInterface(IID_PPV_ARGS(&lock))) || lock->AcquireSync(0u, 1000u) != S_OK) {
        return {};
      }
      display.device_ctx->CopyResource(staging.Get(), texture);
      EXPECT_EQ(lock->ReleaseSync(0u), S_OK);
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(display.device_ctx->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) {
        return {};
      }
      pixels_t pixels {};
      for (UINT y = 0u; y < height; ++y) {
        std::memcpy(pixels.data() + y * width, static_cast<const std::uint8_t *>(mapped.pData) + y * mapped.RowPitch,
          width * sizeof(pixels_t::value_type));
      }
      display.device_ctx->Unmap(staging.Get(), 0u);
      return pixels;
    }

    static pixels_t with_cursor(pixels_t pixels, POINT point, std::uint32_t cursor_bgra = metadata_duplication_t::cursor_bgra) {
      // The production straight-alpha blend writes zero alpha and the opaque cursor RGB.
      pixels[point.y * width + point.x] = cursor_bgra & 0x00ffffffu;
      return pixels;
    }
  };
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

TEST_F(WindowsDdupSnapshotTest, DamageCollectionFollowsActiveConvertersWithoutChangingCapturedPixels) {
  display.remove_damage_observer();
  const auto pixels = desktop_pixels(0xff123400u);
  const auto surface = desktop_surface(pixels);
  ASSERT_TRUE(surface);

  img_t ordinary;
  ASSERT_EQ(snapshot(desktop_present(), surface, ordinary), platf::capture_e::ok);
  ASSERT_TRUE(ordinary && ordinary->content_timestamp);
  EXPECT_EQ(read_image(ordinary), pixels);
  EXPECT_FALSE(display.damage_history);
  EXPECT_FALSE(display.last_ddup_damage);
  EXPECT_EQ(native->move_calls, 0u);
  EXPECT_EQ(native->dirty_calls, 0u);

  display.add_damage_observer();
  img_t first;
  ASSERT_EQ(snapshot(desktop_present(), surface, first), platf::capture_e::ok);
  ASSERT_TRUE(first && first->content_timestamp && display.last_ddup_damage);
  EXPECT_GT(*first->content_timestamp, *ordinary->content_timestamp);
  EXPECT_EQ(read_image(first), pixels);
  EXPECT_EQ(native->move_calls, 0u);
  EXPECT_EQ(native->dirty_calls, 0u);
  const auto baseline = display.last_ddup_damage;

  // During a converter handoff, one remaining observer keeps the existing history useful.
  display.add_damage_observer();
  display.remove_damage_observer();
  img_t next;
  ASSERT_EQ(snapshot(desktop_present(), surface, next), platf::capture_e::ok);
  ASSERT_TRUE(next && next->content_timestamp && display.last_ddup_damage);
  EXPECT_EQ(read_image(next), pixels);
  EXPECT_TRUE(d::ddup_adjacent_desktop_present_observed(baseline, display.last_ddup_damage));
  EXPECT_EQ(d::query_ddup_damage_between(baseline, display.last_ddup_damage, RECT {1, 0, 2, 1}),
    d::ddup_damage_intersection_e::changed);
  EXPECT_EQ(native->move_calls, 1u);
  EXPECT_EQ(native->dirty_calls, 1u);
  const auto old_damage = display.last_ddup_damage;

  display.remove_damage_observer();
  img_t unobserved;
  ASSERT_EQ(snapshot(desktop_present(), surface, unobserved), platf::capture_e::ok);
  ASSERT_TRUE(unobserved && unobserved->content_timestamp);
  EXPECT_GT(*unobserved->content_timestamp, *next->content_timestamp);
  EXPECT_EQ(read_image(unobserved), pixels);
  EXPECT_FALSE(display.damage_history);
  EXPECT_FALSE(display.last_ddup_damage);
  EXPECT_EQ(native->move_calls, 1u);
  EXPECT_EQ(native->dirty_calls, 1u);

  display.add_damage_observer();
  img_t resumed;
  ASSERT_EQ(snapshot(desktop_present(), surface, resumed), platf::capture_e::ok);
  ASSERT_TRUE(resumed && resumed->content_timestamp && display.last_ddup_damage);
  EXPECT_GT(*resumed->content_timestamp, *unobserved->content_timestamp);
  EXPECT_EQ(read_image(resumed), pixels);
  EXPECT_NE(display.last_ddup_damage->history, old_damage->history);
  EXPECT_EQ(d::query_ddup_damage_between(old_damage, display.last_ddup_damage, RECT {1, 0, 2, 1}),
    d::ddup_damage_intersection_e::unknown);
  EXPECT_EQ(native->move_calls, 1u);
  EXPECT_EQ(native->dirty_calls, 1u);
}

TEST_F(WindowsDdupSnapshotTest, VisiblePointerMovesAndHidesWithoutChangingDesktopIdentity) {
  constexpr POINT first_position {2, 3};
  constexpr POINT next_position {9, 4};
  const auto pixels = desktop_pixels(0xff123400u);
  const auto surface = desktop_surface(pixels);
  ASSERT_TRUE(surface);

  auto info = desktop_present();
  pointer_update(info, first_position, true, true);
  img_t first;
  ASSERT_EQ(snapshot(info, surface, first), platf::capture_e::ok);
  ASSERT_TRUE(first && first->frame_timestamp && first->content_timestamp);
  EXPECT_EQ(image_pulls, 1u);
  EXPECT_EQ(read_image(first), with_cursor(pixels, first_position));
  EXPECT_GT(*first->frame_timestamp, *first->content_timestamp);
  ASSERT_TRUE(display.last_ddup_damage);
  const auto damage = display.last_ddup_damage;
  const auto content_time = first->content_timestamp;

  info = {};
  pointer_update(info, next_position, true);
  img_t moved;
  ASSERT_EQ(snapshot(info, surface, moved), platf::capture_e::ok);
  ASSERT_TRUE(moved && moved->frame_timestamp);
  EXPECT_EQ(image_pulls, 2u);
  EXPECT_EQ(read_image(moved), with_cursor(pixels, next_position));
  EXPECT_EQ(moved->content_timestamp, content_time);
  EXPECT_GT(*moved->frame_timestamp, *first->frame_timestamp);
  ASSERT_TRUE(display.last_ddup_damage);
  EXPECT_EQ(display.last_ddup_damage->token, damage->token);

  info = {};
  pointer_update(info, next_position, false);
  img_t hidden;
  ASSERT_EQ(snapshot(info, surface, hidden), platf::capture_e::ok);
  ASSERT_TRUE(hidden && hidden->frame_timestamp);
  EXPECT_EQ(image_pulls, 3u);
  EXPECT_EQ(read_image(hidden), pixels);
  EXPECT_EQ(hidden->content_timestamp, content_time);
  EXPECT_GT(*hidden->frame_timestamp, *moved->frame_timestamp);
  ASSERT_TRUE(display.last_ddup_damage);
  EXPECT_EQ(display.last_ddup_damage->token, damage->token);

  const auto hidden_delivery_time = hidden->frame_timestamp;
  const auto diagnostic_time = std::chrono::steady_clock::now();
  hidden->diagnostic_capture_ready_timestamp = diagnostic_time;
  const auto held_image = hidden;
  info = {};
  pointer_update(info, first_position, false);
  EXPECT_EQ(snapshot(info, surface, hidden), platf::capture_e::timeout);
  EXPECT_FALSE(hidden);
  EXPECT_EQ(image_pulls, 3u);
  EXPECT_EQ(held_image->frame_timestamp, hidden_delivery_time);
  EXPECT_EQ(held_image->content_timestamp, content_time);
  EXPECT_EQ(held_image->diagnostic_capture_ready_timestamp, diagnostic_time);
  EXPECT_EQ(read_image(first), with_cursor(pixels, first_position));
  EXPECT_EQ(read_image(held_image), pixels);

  img_t next;
  ASSERT_EQ(snapshot(desktop_present(), surface, next), platf::capture_e::ok);
  ASSERT_TRUE(next && next->content_timestamp);
  EXPECT_EQ(image_pulls, 4u);
  EXPECT_GT(next->content_timestamp, content_time);
  EXPECT_TRUE(d::ddup_adjacent_desktop_present_observed(damage, display.last_ddup_damage));
  EXPECT_EQ(d::query_ddup_damage_between(damage, display.last_ddup_damage, RECT {1, 0, 2, 1}),
    d::ddup_damage_intersection_e::changed);
}

TEST_F(WindowsDdupSnapshotTest, DuplicateVisibleCursorOnlyFramesSkipCopyAfterBitmapValidation) {
  constexpr POINT position {2, 3};
  const auto pixels = desktop_pixels(0xff123400u);
  const auto surface = desktop_surface(pixels);
  const auto wrong_size = desktop_surface(pixels, width / 2u);
  const auto wrong_format = desktop_surface(pixels, width, DXGI_FORMAT_R8G8B8A8_UNORM);
  ASSERT_TRUE(surface && wrong_size && wrong_format);

  auto info = desktop_present();
  pointer_update(info, position, true, true);
  img_t first;
  ASSERT_EQ(snapshot(info, surface, first), platf::capture_e::ok);
  ASSERT_TRUE(first && first->frame_timestamp && first->content_timestamp && display.last_ddup_damage);
  const auto first_presentation = first->frame_timestamp;
  const auto content_time = first->content_timestamp;
  const auto damage = display.last_ddup_damage;
  ASSERT_EQ(read_image(first), with_cursor(pixels, position));

  info = {};
  pointer_update(info, position, true);
  img_t repeated = first;
  EXPECT_EQ(snapshot(info, surface, repeated), platf::capture_e::timeout);
  EXPECT_FALSE(repeated);
  EXPECT_EQ(image_pulls, 1u);
  EXPECT_EQ(first->frame_timestamp, first_presentation);
  EXPECT_EQ(first->content_timestamp, content_time);
  EXPECT_EQ(read_image(first), with_cursor(pixels, position));

  for (const ComPtr<IDXGIResource> &invalid : {ComPtr<IDXGIResource> {}, wrong_size, wrong_format}) {
    info = {};
    pointer_update(info, position, true);
    EXPECT_EQ(snapshot(info, invalid, repeated), platf::capture_e::reinit);
    EXPECT_FALSE(repeated);
    EXPECT_EQ(image_pulls, 1u);
  }

  native->pointer_shape_bgra = 0xff2143c5u;
  info = {};
  pointer_update(info, position, true, true);
  img_t changed_shape;
  ASSERT_EQ(snapshot(info, surface, changed_shape), platf::capture_e::ok);
  ASSERT_TRUE(changed_shape && changed_shape->frame_timestamp);
  EXPECT_EQ(image_pulls, 2u);
  EXPECT_GT(changed_shape->frame_timestamp, first_presentation);
  EXPECT_EQ(changed_shape->content_timestamp, content_time);
  EXPECT_EQ(read_image(changed_shape), with_cursor(pixels, position, native->pointer_shape_bgra));

  display.external_pixels_owner = 123u;
  info = {};
  pointer_update(info, {3, 3}, true);
  img_t metadata;
  ASSERT_EQ(snapshot(info, {}, metadata), platf::capture_e::ok);
  ASSERT_TRUE(metadata && metadata->pixels_skipped);
  info = {};
  pointer_update(info, position, true);
  ASSERT_EQ(snapshot(info, {}, metadata), platf::capture_e::ok);
  ASSERT_TRUE(metadata && metadata->pixels_skipped && metadata->frame_timestamp);
  EXPECT_FALSE(display.dropped_desktop.dropped());
  EXPECT_EQ(image_pulls, 4u);

  display.external_pixels_owner = 0u;
  info = {};
  pointer_update(info, position, true);
  img_t restored;
  ASSERT_EQ(snapshot(info, surface, restored), platf::capture_e::ok);
  ASSERT_TRUE(restored && restored->frame_timestamp && display.last_ddup_damage);
  EXPECT_FALSE(restored->pixels_skipped);
  EXPECT_GT(restored->frame_timestamp, metadata->frame_timestamp);
  EXPECT_EQ(restored->content_timestamp, content_time);
  EXPECT_EQ(display.last_ddup_damage->token, damage->token);
  EXPECT_EQ(read_image(restored), with_cursor(pixels, position, native->pointer_shape_bgra));
  EXPECT_EQ(image_pulls, 5u);

  info = {};
  pointer_update(info, position, true);
  EXPECT_EQ(snapshot(info, surface, repeated), platf::capture_e::timeout);
  EXPECT_FALSE(repeated);
  EXPECT_EQ(image_pulls, 5u);

  const auto next_pixels = desktop_pixels(0xff562300u);
  const auto next_surface = desktop_surface(next_pixels);
  ASSERT_TRUE(next_surface);
  img_t updated;
  ASSERT_EQ(snapshot(desktop_present(), next_surface, updated), platf::capture_e::ok);
  ASSERT_TRUE(updated && updated->content_timestamp);
  EXPECT_GT(updated->content_timestamp, content_time);
  EXPECT_EQ(read_image(updated), with_cursor(next_pixels, position, native->pointer_shape_bgra));
  EXPECT_EQ(image_pulls, 6u);

  EXPECT_EQ(snapshot(desktop_present(), {}, repeated), platf::capture_e::reinit);
  EXPECT_FALSE(display.damage_chain_valid);
  info = {};
  pointer_update(info, position, true);
  img_t resynchronized;
  ASSERT_EQ(snapshot(info, surface, resynchronized), platf::capture_e::ok);
  ASSERT_TRUE(resynchronized && resynchronized->content_timestamp);
  EXPECT_GT(resynchronized->content_timestamp, updated->content_timestamp);
  EXPECT_EQ(read_image(resynchronized), with_cursor(pixels, position, native->pointer_shape_bgra));
  EXPECT_EQ(image_pulls, 7u);
}

TEST_F(WindowsDdupSnapshotTest, ExportMetadataKeepsBakedCursorIdentityAndDroppedDesktopRecoversOnPointerUpdate) {
  const auto first_pixels = desktop_pixels(0xff123400u);
  const auto first_surface = desktop_surface(first_pixels);
  const auto next_pixels = desktop_pixels(0xff562300u);
  const auto next_surface = desktop_surface(next_pixels);
  ASSERT_TRUE(first_surface && next_surface);

  auto info = desktop_present();
  pointer_update(info, {2, 3}, true, true);
  img_t first;
  ASSERT_EQ(snapshot(info, first_surface, first), platf::capture_e::ok);
  ASSERT_TRUE(first && display.last_ddup_damage);
  const auto first_damage = display.last_ddup_damage;
  const auto first_content_time = first->content_timestamp;

  display.external_pixels_owner = 123u;
  info = {};
  pointer_update(info, {2, 3}, false);
  img_t metadata;
  ASSERT_EQ(snapshot(info, {}, metadata), platf::capture_e::ok);
  ASSERT_TRUE(metadata);
  EXPECT_TRUE(metadata->pixels_skipped);
  EXPECT_TRUE(metadata->conversion_needed);
  EXPECT_FALSE(metadata->data);
  EXPECT_EQ(metadata->content_timestamp, first_content_time);
  EXPECT_FALSE(display.dropped_desktop.dropped());

  display.external_pixels_owner = 0u;
  info = {};
  pointer_update(info, {3, 3}, false);
  img_t cursor_removed;
  ASSERT_EQ(snapshot(info, first_surface, cursor_removed), platf::capture_e::ok);
  ASSERT_TRUE(cursor_removed);
  EXPECT_FALSE(cursor_removed->pixels_skipped);
  EXPECT_EQ(read_image(cursor_removed), first_pixels);
  EXPECT_EQ(cursor_removed->content_timestamp, first_content_time);
  ASSERT_TRUE(display.last_ddup_damage);
  EXPECT_EQ(display.last_ddup_damage->token, first_damage->token);

  display.external_pixels_owner = 123u;
  img_t dropped;
  ASSERT_EQ(snapshot(desktop_present(), next_surface, dropped), platf::capture_e::ok);
  ASSERT_TRUE(dropped);
  EXPECT_TRUE(dropped->pixels_skipped);
  EXPECT_FALSE(dropped->conversion_needed);
  EXPECT_FALSE(dropped->data);
  EXPECT_EQ(dropped->content_timestamp, first_content_time);
  EXPECT_TRUE(display.dropped_desktop.dropped());
  EXPECT_EQ(image_pulls, 4u);

  display.external_pixels_owner = 0u;
  info = {};
  pointer_update(info, {5, 4}, false);
  img_t missing_recovery;
  EXPECT_EQ(snapshot(info, {}, missing_recovery), platf::capture_e::timeout);
  EXPECT_FALSE(missing_recovery);
  EXPECT_TRUE(display.dropped_desktop.dropped());
  EXPECT_EQ(image_pulls, 4u);

  info = {};
  constexpr POINT recovered_pointer {5, 5};
  pointer_update(info, recovered_pointer, true);
  img_t recovered;
  ASSERT_EQ(snapshot(info, next_surface, recovered), platf::capture_e::ok);
  ASSERT_TRUE(recovered && recovered->content_timestamp);
  EXPECT_FALSE(recovered->pixels_skipped);
  EXPECT_EQ(read_image(recovered), with_cursor(next_pixels, recovered_pointer));
  EXPECT_GT(recovered->content_timestamp, first_content_time);
  EXPECT_FALSE(display.dropped_desktop.dropped());
  EXPECT_EQ(image_pulls, 5u);
  ASSERT_TRUE(display.last_ddup_damage);
  EXPECT_EQ(display.last_ddup_damage->token, first_damage->token + 1u);
  EXPECT_FALSE(d::ddup_adjacent_desktop_present_observed(first_damage, display.last_ddup_damage));
  EXPECT_EQ(d::query_ddup_damage_between(first_damage, display.last_ddup_damage, RECT {1, 0, 2, 1}),
    d::ddup_damage_intersection_e::unknown);
}

TEST_F(WindowsDdupSnapshotTest, FailedDesktopCopiesNeverAdvanceIdentityAndNextSurfaceResynchronizesDamage) {
  const auto first_pixels = desktop_pixels(0xff123400u);
  const auto first_surface = desktop_surface(first_pixels);
  const auto next_pixels = desktop_pixels(0xff562300u);
  const auto next_surface = desktop_surface(next_pixels);
  ASSERT_TRUE(first_surface && next_surface);
  img_t first;
  ASSERT_EQ(snapshot(desktop_present(), first_surface, first), platf::capture_e::ok);
  ASSERT_TRUE(first && first->content_timestamp && display.last_ddup_damage);
  const auto first_damage = display.last_ddup_damage;
  const auto first_content_time = first->content_timestamp;

  DXGI_OUTDUPL_FRAME_INFO info {};
  pointer_update(info, {2, 3}, true, true);
  img_t missing;
  EXPECT_EQ(snapshot(info, {}, missing), platf::capture_e::reinit);
  EXPECT_FALSE(missing);
  EXPECT_TRUE(display.damage_chain_valid);
  EXPECT_EQ(image_pulls, 1u);
  EXPECT_EQ(display.last_content_timestamp, first_content_time);
  ASSERT_TRUE(display.last_ddup_damage);
  EXPECT_EQ(display.last_ddup_damage->token, first_damage->token);

  EXPECT_EQ(snapshot(desktop_present(), {}, missing), platf::capture_e::reinit);
  EXPECT_FALSE(missing);
  EXPECT_FALSE(display.damage_chain_valid);
  EXPECT_EQ(image_pulls, 1u);
  EXPECT_EQ(display.last_content_timestamp, first_content_time);
  ASSERT_TRUE(display.last_ddup_damage);
  EXPECT_EQ(display.last_ddup_damage->token, first_damage->token);

  allow_image = false;
  EXPECT_EQ(snapshot(desktop_present(), next_surface, missing), platf::capture_e::interrupted);
  EXPECT_FALSE(missing);
  EXPECT_FALSE(display.damage_chain_valid);
  EXPECT_EQ(image_pulls, 2u);
  EXPECT_EQ(display.last_content_timestamp, first_content_time);
  ASSERT_TRUE(display.last_ddup_damage);
  EXPECT_EQ(display.last_ddup_damage->token, first_damage->token);

  // The fixture deliberately keeps the same object after failures. Its next acquired full
  // texture holds new pixels even though the only fresh DXGI notification is a pointer update.
  allow_image = true;
  info = {};
  pointer_update(info, {2, 3}, false);
  img_t resynchronized;
  ASSERT_EQ(snapshot(info, next_surface, resynchronized), platf::capture_e::ok);
  ASSERT_TRUE(resynchronized && resynchronized->content_timestamp);
  EXPECT_EQ(read_image(resynchronized), next_pixels);
  EXPECT_GT(resynchronized->content_timestamp, first_content_time);
  EXPECT_TRUE(display.damage_chain_valid);
  EXPECT_EQ(image_pulls, 3u);
  ASSERT_TRUE(display.last_ddup_damage);
  EXPECT_EQ(display.last_ddup_damage->token, first_damage->token + 1u);
  EXPECT_FALSE(d::ddup_adjacent_desktop_present_observed(first_damage, display.last_ddup_damage));
  EXPECT_EQ(d::query_ddup_damage_between(first_damage, display.last_ddup_damage, RECT {1, 0, 2, 1}),
    d::ddup_damage_intersection_e::unknown);
}

#endif // _WIN32
