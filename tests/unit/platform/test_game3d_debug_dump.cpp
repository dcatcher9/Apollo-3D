// SPDX-License-Identifier: GPL-3.0-only
#include "src/platform/windows/game3d_debug_dump.h"
#include "src/game3d_debug_ui_resources.h"
#include "src/platform/windows/sbs_debug_dump_async.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <d3d11_1.h>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <iterator>
#include <nlohmann/json.hpp>
#include <thread>
#include <vector>
#include <wrl/client.h>

namespace {
  namespace wire = ::game3d_debug;
  namespace dump = platf::game3d_debug;
  using Microsoft::WRL::ComPtr;
  using namespace std::chrono_literals;

  wire::response_t valid_response() {
    wire::response_t value;
    value.consumer_nonce = 123;
    value.capture_id = 1;
    value.result = wire::status::complete;
    value.texture_count = 1;
    value.textures[0] = {wire::artifact::raw_depth, 8, 4, DXGI_FORMAT_R32_FLOAT, 1234};
    return value;
  }

  bool wait_for_publications() {
    auto drained = std::make_shared<std::promise<void>>();
    auto finished = drained->get_future();
    return platf::sbs_debug::detail::process_publication_queue().enqueue([drained] {
      drained->set_value();
    }) && finished.wait_for(5s) == std::future_status::ready;
  }
}  // namespace

TEST(Game3DDumpPublication, ConverterReplacementCannotQueueAnotherLargePackage) {
  // Hold the existing CPU worker instead of depending on GPU or disk timing.
  struct release_guard {
    std::promise<void> promise;
    bool released = false;
    void release() {
      if (!released) {
        released = true;
        promise.set_value();
      }
    }
    ~release_guard() { release(); }
  } blocker;
  const auto release = blocker.promise.get_future().share();
  ASSERT_TRUE(platf::sbs_debug::detail::process_publication_queue().enqueue([release] {
    release.wait();
  }));

  LARGE_INTEGER qpc {};
  QueryPerformanceCounter(&qpc);
  const auto directory = std::filesystem::temp_directory_path() /
    ("sunshine_game_dump_publication_test_" + std::to_string(qpc.QuadPart));
  auto button = std::make_shared<std::atomic<bool>>(true);
  {
    dump::dumper original({}, directory);
    original.set_button_request(button);
    // An invalid context produces a CPU-only unavailable package, using the
    // same publication ownership as a complete full-resolution GPU package.
    original.poll(nullptr, nullptr, {}, 8, 4, 2);
    EXPECT_FALSE(button->load());
  }
  auto next_button = std::make_shared<std::atomic<bool>>(true);
  dump::dumper replacement({}, directory);
  replacement.set_button_request(next_button);
  EXPECT_FALSE(replacement.needs_conversion_poll());
  replacement.poll(nullptr, nullptr, {}, 8, 4, 2);
  EXPECT_TRUE(next_button->load()) << "The replacement must leave its request pending until the old publication retires";

  blocker.release();
  // Drain a sentinel after the publication before inspecting or removing files.
  ASSERT_TRUE(wait_for_publications());
  EXPECT_TRUE(replacement.needs_conversion_poll());
  replacement.cancel();
  std::error_code ignored;
  std::filesystem::remove_all(directory, ignored);
}

TEST(Game3DDumpProtocol, RejectsInvalidIdentityLengthsFormatsDimensionsAndMemory) {
  auto response = valid_response();
  std::string reason;
  EXPECT_TRUE(dump::detail::validate_response(response, 123, reason));
  for (const auto mutate : std::array<void (*)(wire::response_t &), 10> {
         [](auto &r) {
           ++r.consumer_nonce;
         },
         [](auto &r) {
           r.result = static_cast<wire::status>(0);
         },
         [](auto &r) {
           r.texture_count = wire::max_textures + 1;
         },
         [](auto &r) {
           r.json_bytes = wire::max_json_bytes + 1;
         },
         [](auto &r) {
           r.textures[0].width = 0;
         },
         [](auto &r) {
           r.textures[0].height = wire::max_dimension + 1;
         },
         [](auto &r) {
           r.textures[0].dxgi_format = DXGI_FORMAT_R32_TYPELESS;
         },
         [](auto &r) {
           r.textures[0].handle = 0;
         },
         [](auto &r) {
           r.textures[0].kind = static_cast<wire::artifact>(0);
         },
         [](auto &r) {
           r.textures[0].width = r.textures[0].height = wire::max_dimension;
         },
       }) {
    auto invalid = response;
    mutate(invalid);
    EXPECT_FALSE(dump::detail::validate_response(invalid, 123, reason));
    EXPECT_FALSE(reason.empty());
  }
  EXPECT_FALSE(dump::detail::validate_response(response, 0, reason));
  response.texture_count = 2;
  response.textures[1] = response.textures[0];
  EXPECT_FALSE(dump::detail::validate_response(response, 123, reason));
  response.textures[1].kind = wire::artifact::sbs;
  EXPECT_FALSE(dump::detail::validate_response(response, 123, reason));
}

TEST(Game3DDumpProtocol, MissingDepthIsUnavailableRatherThanAnEmptySuccessfulCapture) {
  auto response = valid_response();
  response.texture_count = 0;
  std::string reason;
  EXPECT_FALSE(dump::detail::validate_response(response, 123, reason));
  response.result = wire::status::unavailable;
  EXPECT_TRUE(dump::detail::validate_response(response, 123, reason));
  EXPECT_EQ(dump::detail::bytes_per_pixel(DXGI_FORMAT_R16G16B16A16_FLOAT), 8u);
  EXPECT_EQ(dump::detail::bytes_per_pixel(DXGI_FORMAT_R32_FLOAT), 4u);
  EXPECT_EQ(dump::detail::bytes_per_pixel(DXGI_FORMAT_R8_UINT), 1u);
  EXPECT_EQ(dump::detail::bytes_per_pixel(DXGI_FORMAT_R16G16_UINT), 4u);
  EXPECT_EQ(dump::detail::bytes_per_pixel(DXGI_FORMAT_R32G32B32A32_SINT), 16u);
  EXPECT_EQ(dump::detail::bytes_per_pixel(DXGI_FORMAT_R9G9B9E5_SHAREDEXP), 4u);
  EXPECT_EQ(dump::detail::bytes_per_pixel(DXGI_FORMAT_BC1_UNORM), 0u);
}

TEST(Game3DDumpProtocol, OptionalCatalogCannotInvalidatePrimaryReplayDescriptors) {
  auto response = valid_response();
  std::string reason;
  for (const auto &entry : sunshine_game3d::ui_resources::catalog) {
    const auto index = response.texture_count++;
    response.textures[index] = {static_cast<wire::artifact>(entry.artifact_id), 8, 4, DXGI_FORMAT_R8_UNORM, 2000 + index};
  }
  ASSERT_GT(response.texture_count, 10u);
  ASSERT_LE(response.texture_count, wire::max_textures);
  EXPECT_TRUE(dump::detail::validate_response(response, 123, reason));
  // Optional descriptor failures are handled per artifact by the receiver.
  response.textures[1].dxgi_format = DXGI_FORMAT_UNKNOWN;
  response.textures[2].handle = 0;
  response.textures[3].kind = static_cast<wire::artifact>(999);
  EXPECT_TRUE(dump::detail::validate_response(response, 123, reason));
  response.textures[0].dxgi_format = DXGI_FORMAT_UNKNOWN;
  EXPECT_FALSE(dump::detail::validate_response(response, 123, reason));
}

TEST(Game3DDumpProtocol, ConsumedUiAndEntireCatalogFitV3Envelope) {
  auto response = valid_response();
  response.texture_count = 0;
  for (unsigned id = 1; id <= static_cast<unsigned>(wire::artifact::linear_color); ++id) {
    const auto index = response.texture_count++;
    response.textures[index] = {static_cast<wire::artifact>(id), 8, 4, DXGI_FORMAT_R32_FLOAT, 1000 + index};
  }
  for (const auto &entry : sunshine_game3d::ui_resources::catalog) {
    ASSERT_LT(response.texture_count, wire::max_textures);
    const auto index = response.texture_count++;
    response.textures[index] = {static_cast<wire::artifact>(entry.artifact_id), 8, 4, DXGI_FORMAT_R10G10B10A2_UNORM, 2000 + index};
  }
  response.textures[response.texture_count++] = {wire::artifact::ui_source_color, 8, 4, DXGI_FORMAT_R10G10B10A2_UNORM, 9000};
  EXPECT_EQ(wire::version, 3u);
  EXPECT_EQ(response.texture_count, 33u);
  EXPECT_LE(response.texture_count, wire::max_textures);
  ASSERT_NE(sunshine_game3d::ui_resources::find_sl(53), nullptr);
  EXPECT_EQ(sunshine_game3d::ui_resources::find_sl(53)->artifact_id, 32u);
  std::string reason;
  EXPECT_TRUE(dump::detail::validate_response(response, 123, reason)) << reason;
  response.textures[response.texture_count - 1].handle = 0;
  EXPECT_FALSE(dump::detail::validate_response(response, 123, reason)) << "Consumed UI is a mandatory replay input, not best-effort diagnostics";
}

namespace {
  class Game3DDumpGpu: public testing::Test {
  protected:
    static constexpr UINT width = 8, height = 4;
    const RECT fullscreen {100, 200, 108, 204};

    void SetUp() override {
      D3D_FEATURE_LEVEL level {};
      const D3D_FEATURE_LEVEL requested = D3D_FEATURE_LEVEL_11_0;
      const auto result = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &requested, 1, D3D11_SDK_VERSION, &producer, &level, &producer_context);
      if (FAILED(result)) {
        GTEST_SKIP() << "Hardware D3D11 unavailable: " << std::hex << result;
      }
      ComPtr<IDXGIDevice> dxgi;
      ASSERT_EQ(producer.As(&dxgi), S_OK);
      ComPtr<IDXGIAdapter> adapter;
      ASSERT_EQ(dxgi->GetAdapter(&adapter), S_OK);
      ASSERT_EQ(D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, &requested, 1, D3D11_SDK_VERSION, &consumer, &level, &consumer_context), S_OK);
      const auto name = std::wstring(wire::mapping_prefix) + std::to_wstring(GetCurrentProcessId());
      mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(wire::shared_state_t), name.c_str());
      ASSERT_NE(mapping, nullptr);
      ASSERT_NE(GetLastError(), ERROR_ALREADY_EXISTS);
      shared = static_cast<wire::shared_state_t *>(MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(wire::shared_state_t)));
      ASSERT_NE(shared, nullptr);
      *shared = wire::shared_state_t {};
      shared->producer_pid = GetCurrentProcessId();
      FILETIME creation {}, exit {}, kernel {}, user {};
      ASSERT_TRUE(GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user));
      shared->producer_creation_time = (static_cast<std::uint64_t>(creation.dwHighDateTime) << 32) | creation.dwLowDateTime;
      observed.status = platf::foreground_window::status_e::ok;
      observed.process_id = GetCurrentProcessId();
      observed.window = 42;
      observed.client_screen_rect = {fullscreen.left, fullscreen.top, fullscreen.right, fullscreen.bottom};
      LARGE_INTEGER qpc {};
      QueryPerformanceCounter(&qpc);
      directory = std::filesystem::temp_directory_path() / ("sunshine_game_dump_test_" + std::to_string(qpc.QuadPart));
      consumer_dump = std::make_unique<dump::dumper>([this] {
        return observed;
      },
                                                     directory);
      button = std::make_shared<std::atomic<bool>>(false);
      consumer_dump->set_button_request(button);
    }

    void TearDown() override {
      consumer_dump.reset();
      close_producer_textures();
      if (shared) {
        UnmapViewOfFile(shared);
      }
      if (mapping) {
        CloseHandle(mapping);
      }
      // Rename precedes the worker clearing its shared single-flight state.
      // Drain that last callback before the next fixture submits a new request.
      EXPECT_TRUE(wait_for_publications());
      if (!directory.empty()) {
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
      }
    }

    void poll() {
      consumer_dump->poll(consumer.Get(), consumer_context.Get(), fullscreen, width, height, 2);
    }

    void request() {
      button->store(true);
      poll();
    }

    void close_producer_textures() {
      for (HANDLE handle : handles) {
        CloseHandle(handle);
      }
      handles.clear();
      textures.clear();
    }

    bool texture(wire::artifact kind, DXGI_FORMAT format, const std::vector<std::uint8_t> &bytes) {
      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = width;
      desc.Height = height;
      desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
      desc.Format = format;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
      desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
      D3D11_SUBRESOURCE_DATA data {bytes.data(), width * dump::detail::bytes_per_pixel(format), 0};
      ComPtr<ID3D11Texture2D> texture;
      if (FAILED(producer->CreateTexture2D(&desc, &data, &texture))) {
        return false;
      }
      ComPtr<IDXGIResource1> resource;
      HANDLE handle = nullptr;
      if (FAILED(texture.As(&resource)) || FAILED(resource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr, &handle))) {
        return false;
      }
      const auto index = shared->response.texture_count++;
      shared->response.textures[index] = {kind, width, height, static_cast<std::uint32_t>(format), reinterpret_cast<std::uintptr_t>(handle)};
      textures.push_back(texture);
      handles.push_back(handle);
      return true;
    }

    void respond(wire::status status, std::string metadata = "{\"provider\":\"test\",\"source_transfer\":\"scRGB\"}") {
      ComPtr<ID3D11Query> done;
      D3D11_QUERY_DESC desc {D3D11_QUERY_EVENT, 0};
      ASSERT_EQ(producer->CreateQuery(&desc, &done), S_OK);
      producer_context->End(done.Get());
      producer_context->Flush();
      BOOL completed = FALSE;
      const auto until = std::chrono::steady_clock::now() + 5s;
      while (producer_context->GetData(done.Get(), &completed, sizeof(completed), D3D11_ASYNC_GETDATA_DONOTFLUSH) == S_FALSE &&
             std::chrono::steady_clock::now() < until) {
        std::this_thread::sleep_for(1ms);
      }
      ASSERT_TRUE(completed);
      shared->response.consumer_nonce = shared->consumer_nonce;
      shared->response.capture_id = 99;
      shared->response.result = status;
      shared->response.json_bytes = metadata.size();
      std::memcpy(shared->json, metadata.data(), metadata.size());
      InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&shared->response_id), static_cast<LONG64>(shared->request_id));
    }

    std::filesystem::path await_package() {
      const auto until = std::chrono::steady_clock::now() + 5s;
      do {
        poll();
        if (std::filesystem::exists(directory)) {
          for (const auto &entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_directory() && entry.path().extension() != ".tmp" && std::filesystem::exists(entry.path() / "manifest.json")) {
              return entry.path();
            }
          }
        }
        std::this_thread::sleep_for(1ms);
      } while (std::chrono::steady_clock::now() < until);
      ADD_FAILURE() << "Timed out waiting for diagnostic publication";
      return {};
    }

    nlohmann::json read_manifest(const std::filesystem::path &path) {
      std::ifstream file(path / "manifest.json");
      return nlohmann::json::parse(file);
    }

    static std::vector<std::uint8_t> read_bytes(const std::filesystem::path &path) {
      std::ifstream stream(path, std::ios::binary);
      return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    }

    ComPtr<ID3D11Device> producer, consumer;
    ComPtr<ID3D11DeviceContext> producer_context, consumer_context;
    std::vector<ComPtr<ID3D11Texture2D>> textures;
    std::vector<HANDLE> handles;
    HANDLE mapping = nullptr;
    wire::shared_state_t *shared = nullptr;
    std::filesystem::path directory;
    platf::foreground_window::observation_t observed;
    std::shared_ptr<std::atomic<bool>> button;
    std::unique_ptr<dump::dumper> consumer_dump;
  };
}  // namespace

TEST_F(Game3DDumpGpu, LosslessHdrAndRawDepthSurviveProducerReleaseAfterOpening) {
  request();
  ASSERT_NE(shared->request_id, 0u);
  const auto request_id = shared->request_id;
  std::vector<std::uint8_t> hdr(width * height * 8), depth(width * height * 4);
  // Includes negative/large FP16 and R32 payload bit patterns; dumping never normalizes them.
  const std::array<std::uint16_t, 4> pixel {0xBC00, 0x4400, 0x3555, 0x3C00};
  for (std::size_t i = 0; i < hdr.size(); i += sizeof(pixel)) {
    std::memcpy(hdr.data() + i, pixel.data(), sizeof(pixel));
  }
  for (std::size_t i = 0; i < depth.size(); ++i) {
    depth[i] = static_cast<std::uint8_t>(i * 13);
  }
  ASSERT_TRUE(texture(wire::artifact::source_color, DXGI_FORMAT_R16G16B16A16_FLOAT, hdr));
  ASSERT_TRUE(texture(wire::artifact::raw_depth, DXGI_FORMAT_R32_FLOAT, depth));
  respond(wire::status::complete, "{\"color_space\":2,\"consumed_depth\":{\"orientation\":2}}");
  poll();
  ASSERT_EQ(shared->released_id, request_id);
  close_producer_textures();
  const auto path = await_package();
  ASSERT_FALSE(path.empty());
  const auto manifest = read_manifest(path);
  EXPECT_EQ(manifest["status"], "complete");
  EXPECT_EQ(manifest["request_id"], request_id);
  EXPECT_EQ(manifest["artifacts"].size(), 2u);
  EXPECT_EQ(read_bytes(path / "0_source_color.bin"), hdr);
  EXPECT_EQ(read_bytes(path / "1_raw_depth.bin"), depth);
  EXPECT_EQ(manifest["artifacts"][0]["row_bytes"], width * 8);
  EXPECT_EQ(manifest["artifacts"][1]["dxgi_format"], DXGI_FORMAT_R32_FLOAT);
  // Human views are generated by the same publication, with no extra client
  // command, and cannot replace or normalize the exact native artifacts.
  ASSERT_TRUE(manifest.contains("visualizations"));
  const std::array<std::uint8_t, 8> png_signature {137, 80, 78, 71, 13, 10, 26, 10};
  for (const auto *name : {"color.png", "raw_depth.png"}) {
    const auto png = read_bytes(path / name);
    ASSERT_GE(png.size(), png_signature.size()) << name;
    EXPECT_TRUE(std::equal(png_signature.begin(), png_signature.end(), png.begin())) << name;
  }
  EXPECT_TRUE(std::filesystem::exists(path / "index.html"));
  EXPECT_FALSE(consumer_dump->needs_conversion_poll());
}

TEST_F(Game3DDumpGpu, MissingDepthPublishesExplicitUnavailableMetadata) {
  request();
  respond(wire::status::unavailable, "{\"reason\":\"waiting for depth\"}");
  const auto path = await_package();
  ASSERT_FALSE(path.empty());
  const auto manifest = read_manifest(path);
  EXPECT_EQ(manifest["status"], "unavailable");
  EXPECT_EQ(manifest["producer_metadata"]["reason"], "waiting for depth");
  EXPECT_TRUE(manifest["artifacts"].empty());
  EXPECT_TRUE(manifest.contains("visualizations"));
  EXPECT_TRUE(std::filesystem::exists(path / "index.html"));
}

TEST_F(Game3DDumpGpu, ConsumedUiPrecedesOptionalSnapshotsAndSurvivesProducerRelease) {
  request();
  const std::vector<std::uint8_t> source(width * height * 4, 255);
  const std::vector<std::uint8_t> latest(width * height * 4, 73);
  std::vector<std::uint8_t> consumed(width * height * 4);
  for (std::size_t i = 0; i < consumed.size(); ++i) consumed[i] = static_cast<std::uint8_t>(i * 17);
  // Deliberately publish the unrelated optional snapshot first: the consumer
  // must reserve/read/write mandatory replay inputs before best-effort data.
  ASSERT_TRUE(texture(static_cast<wire::artifact>(32), DXGI_FORMAT_R8G8B8A8_UNORM, latest));
  ASSERT_TRUE(texture(wire::artifact::source_color, DXGI_FORMAT_R8G8B8A8_UNORM, source));
  ASSERT_TRUE(texture(wire::artifact::ui_source_color, DXGI_FORMAT_R8G8B8A8_UNORM, consumed));
  respond(wire::status::complete, R"({"color_space":1,"replay":{"source_alpha_ui":true,"ui_alpha_source":"ui_source_color"},"ui_source":{"sequence":42,"same_game_frame":"unverified"}})");
  poll();
  close_producer_textures();
  const auto path = await_package();
  ASSERT_FALSE(path.empty());
  const auto manifest = read_manifest(path);
  ASSERT_EQ(manifest["artifacts"].size(), 3u);
  EXPECT_EQ(manifest["status"], "complete");
  EXPECT_EQ(manifest["artifacts"][1]["artifact_id"], 33);
  EXPECT_EQ(manifest["producer_metadata"]["ui_source"]["sequence"], 42);
  EXPECT_EQ(read_bytes(path / "0_source_color.bin"), source);
  EXPECT_EQ(read_bytes(path / "1_ui_source_color.bin"), consumed);
  EXPECT_EQ(read_bytes(path / "2_sl_backbuffer.bin"), latest);
  EXPECT_TRUE(std::filesystem::exists(path / "ui_source_alpha.png"));
  EXPECT_TRUE(std::filesystem::exists(path / "ui_source_color.png"));
}

TEST_F(Game3DDumpGpu, OptionalMasksAndAlphaPreserveBytesAndBadOptionalDoesNotLosePrimary) {
  request();
  const std::vector<std::uint8_t> color(width * height * 4, 73);
  const std::vector<std::uint8_t> mask(width * height, 255);
  std::vector<std::uint8_t> ui(width * height * 4);
  std::vector<std::uint8_t> integers(width * height);
  for (std::size_t i = 0; i < ui.size(); ++i) ui[i] = static_cast<std::uint8_t>(i * 17);
  for (std::size_t i = 0; i < integers.size(); ++i) integers[i] = static_cast<std::uint8_t>(i * 7);
  ASSERT_TRUE(texture(wire::artifact::source_color, DXGI_FORMAT_R8G8B8A8_UNORM, color));
  ASSERT_TRUE(texture(static_cast<wire::artifact>(11), DXGI_FORMAT_R8_UNORM, mask));
  ASSERT_TRUE(texture(static_cast<wire::artifact>(10), DXGI_FORMAT_R8G8B8A8_UNORM, ui));
  ASSERT_TRUE(texture(static_cast<wire::artifact>(13), DXGI_FORMAT_R8_UINT, integers));
  shared->response.textures[shared->response.texture_count++] = {static_cast<wire::artifact>(999), 1, 1, DXGI_FORMAT_R8_UNORM, 1};
  shared->response.textures[shared->response.texture_count++] = {static_cast<wire::artifact>(15), 1, 1, DXGI_FORMAT_UNKNOWN, 2};
  shared->response.textures[shared->response.texture_count++] = {static_cast<wire::artifact>(19), 1, 1, DXGI_FORMAT_R8_UNORM, 3};
  respond(wire::status::complete, "{\"color_space\":2,\"optional_captures\":[{\"artifact_id\":10,\"transfer_status\":\"unknown\"}]}");
  poll();
  close_producer_textures();
  const auto path = await_package();
  ASSERT_FALSE(path.empty());
  const auto manifest = read_manifest(path);
  EXPECT_EQ(manifest["status"], "complete");
  ASSERT_EQ(manifest["artifacts"].size(), 4u);
  EXPECT_EQ(manifest["optional_capture_errors"].size(), 3u);
  EXPECT_EQ(read_bytes(path / "0_source_color.bin"), color);
  EXPECT_EQ(read_bytes(path / "1_sl_no_warp_mask.bin"), mask);
  EXPECT_EQ(read_bytes(path / "2_sl_ui_color_alpha.bin"), ui);
  EXPECT_EQ(read_bytes(path / "3_sl_alpha.bin"), integers);
  EXPECT_EQ(manifest["artifacts"][2]["artifact_id"], 10);
  for (const auto *name : {"color.png", "sl_no_warp_mask.png", "sl_ui_color_alpha.png", "sl_ui_color_alpha_alpha.png", "sl_alpha.png"})
    EXPECT_TRUE(std::filesystem::exists(path / name));
}

TEST_F(Game3DDumpGpu, PreviewFailurePreservesLosslessCaptureAndManifest) {
  request();
  const std::vector<std::uint8_t> color(width * height * 4, 73);
  ASSERT_TRUE(texture(wire::artifact::source_color, DXGI_FORMAT_R8G8B8A8_UNORM, color));
  // Producer metadata is independent from validated native texture bytes. A
  // malformed preview-only value must not make reproduction inputs disappear.
  respond(wire::status::complete, "{\"color_space\":\"invalid\"}");
  const auto path = await_package();
  ASSERT_FALSE(path.empty());
  const auto manifest = read_manifest(path);
  EXPECT_EQ(manifest["status"], "complete");
  EXPECT_EQ(read_bytes(path / "0_source_color.bin"), color);
  ASSERT_EQ(manifest["artifacts"].size(), 1u);
  EXPECT_EQ(manifest["visualizations"]["status"], "failed");
  EXPECT_EQ(manifest["visualizations"]["raw_artifacts_preserved"], true);
}

TEST_F(Game3DDumpGpu, OptionalMetadataOmissionFlagPreservesPrimaryPixelsAndFullMetadata) {
  request();
  std::vector<std::uint8_t> depth(width * height * 4);
  for (std::size_t i = 0; i < depth.size(); ++i) depth[i] = static_cast<std::uint8_t>(i * 11);
  ASSERT_TRUE(texture(wire::artifact::raw_depth, DXGI_FORMAT_R32_FLOAT, depth));
  shared->response.flags = wire::optional_metadata_omitted;
  const nlohmann::json original {{"color_space", 1}, {"primary_marker", "exact primary metadata"},
    {"ui_resources", {{{"file_stem", "sl_no_warp_mask"}, {"state", "null"}}}},
    {"padding", std::string(wire::max_json_bytes - 1024, 'x')}};
  const auto encoded = original.dump();
  ASSERT_LT(encoded.size(), wire::max_json_bytes);
  respond(wire::status::complete, encoded);
  poll();
  close_producer_textures();
  const auto path = await_package();
  ASSERT_FALSE(path.empty());
  const auto manifest = read_manifest(path);
  EXPECT_EQ(manifest["status"], "complete");
  EXPECT_EQ(manifest["producer_flags"], wire::optional_metadata_omitted);
  EXPECT_EQ(manifest["producer_metadata"], original);
  ASSERT_EQ(manifest["artifacts"].size(), 1u);
  EXPECT_EQ(read_bytes(path / "0_raw_depth.bin"), depth);
  ASSERT_EQ(manifest["optional_capture_errors"].size(), 1u);
  EXPECT_EQ(manifest["optional_capture_errors"][0]["stage"], "producer_metadata");
  EXPECT_FALSE(manifest["optional_capture_errors"][0]["reason"].get<std::string>().empty());
  EXPECT_TRUE(std::filesystem::exists(path / "raw_depth.png"));
}

TEST_F(Game3DDumpGpu, ValidHandleWithLyingDimensionsIsRejectedBeforeCopy) {
  request();
  std::vector<std::uint8_t> depth(width * height * 4, 17);
  ASSERT_TRUE(texture(wire::artifact::raw_depth, DXGI_FORMAT_R32_FLOAT, depth));
  ++shared->response.textures[0].width;
  respond(wire::status::complete);
  const auto path = await_package();
  ASSERT_FALSE(path.empty());
  const auto manifest = read_manifest(path);
  EXPECT_EQ(manifest["status"], "failed");
  EXPECT_EQ(manifest["reason"], "shared texture does not match snapshot descriptor");
  EXPECT_TRUE(manifest["artifacts"].empty());
}

TEST_F(Game3DDumpGpu, WrongProcessCreationTimeCannotAuthorizeARequest) {
  ++shared->producer_creation_time;
  request();
  EXPECT_EQ(shared->request_id, 0u);
  const auto path = await_package();
  ASSERT_FALSE(path.empty());
  EXPECT_EQ(read_manifest(path)["status"], "unavailable");
}

TEST_F(Game3DDumpGpu, CancellationReleasesOnlyItsOwnRequestAndRejectsOldResponse) {
  request();
  const auto old_id = shared->request_id;
  const auto old_nonce = shared->consumer_nonce;
  consumer_dump->cancel();
  EXPECT_EQ(shared->released_id, old_id);
  EXPECT_FALSE(consumer_dump->needs_conversion_poll());
  request();
  ASSERT_NE(shared->request_id, old_id);
  ASSERT_NE(shared->consumer_nonce, old_nonce);
  shared->response.consumer_nonce = old_nonce;
  shared->response_id = old_id;
  poll();
  EXPECT_TRUE(consumer_dump->needs_conversion_poll());
  EXPECT_EQ(shared->released_id, old_id);
  respond(wire::status::unavailable, "{\"reason\":\"second request\"}");
  const auto path = await_package();
  ASSERT_FALSE(path.empty());
  EXPECT_EQ(read_manifest(path)["producer_metadata"]["reason"], "second request");
}

TEST_F(Game3DDumpGpu, RepeatedClickCannotOverwriteAnOutstandingProducerRequest) {
  request();
  const auto request_id = shared->request_id;
  const auto nonce = shared->consumer_nonce;
  button->store(true);
  poll();
  poll();
  EXPECT_EQ(shared->request_id, request_id);
  EXPECT_EQ(shared->consumer_nonce, nonce);
  EXPECT_TRUE(button->load());
  consumer_dump->cancel();
  EXPECT_EQ(shared->released_id, request_id);
  EXPECT_FALSE(button->load());
  EXPECT_FALSE(consumer_dump->needs_conversion_poll());
}
