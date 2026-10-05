// SPDX-License-Identifier: GPL-3.0-only
#include "src/reshade_bridge_protocol.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {
  ::reshade_bridge::metadata_t valid_bridge_metadata() {
    ::reshade_bridge::metadata_t metadata;
    metadata.producer_pid = 42;
    metadata.producer_creation_time = 123;
    metadata.window = 456;
    metadata.adapter_luid = 789;
    metadata.generation = 1;
    metadata.accepted_consumer_nonce = 1;
    metadata.source_width = 1920;
    metadata.source_height = 1080;
    metadata.packed_width = 3840;
    metadata.packed_height = 1080;
    metadata.dxgi_format = 28;
    metadata.texture_handles[0] = 1;
    metadata.texture_handles[1] = 2;
    metadata.texture_handles[2] = 3;
    metadata.ready_fence_handle = 4;
    return metadata;
  }
}  // namespace

TEST(ReShadeBridgeProtocol, RequiresVersionedIdentityAndCompleteSynchronizationHandles) {
  using namespace ::reshade_bridge;
  const auto valid = valid_bridge_metadata();
  ASSERT_TRUE(valid_metadata(valid));
  for (const auto mutate : std::array<void (*)(metadata_t &), 12> {
         [](metadata_t &m) {
           m.signature ^= 1;
         },
         [](metadata_t &m) {
           m.protocol_version = pq_version + 1;
         },
         [](metadata_t &m) {
           --m.metadata_bytes;
         },
         [](metadata_t &m) {
           m.producer_pid = 0;
         },
         [](metadata_t &m) {
           m.producer_creation_time = 0;
         },
         [](metadata_t &m) {
           m.window = 0;
         },
         [](metadata_t &m) {
           m.generation = 0;
         },
         [](metadata_t &m) {
           m.generation = max_generation + 1;
         },
         [](metadata_t &m) {
           m.accepted_consumer_nonce = 0;
         },
         [](metadata_t &m) {
           m.ready_fence_handle = 0;
         },
         [](metadata_t &m) {
           m.texture_handles[1] = 0;
         },
         [](metadata_t &m) {
           m.image_layout = static_cast<layout>(2);
         },
       }) {
    auto candidate = valid;
    mutate(candidate);
    EXPECT_FALSE(valid_metadata(candidate));
  }
}

TEST(ReShadeBridgeProtocol, SourceDescriptionNeedsNoConsumerRingButTheSameSourceRules) {
  using namespace ::reshade_bridge;
  // What a producer publishes while no consumer is attached: its source, without a ring.
  auto source = valid_bridge_metadata();
  source.generation = 0;
  source.accepted_consumer_nonce = 0;
  source.ready_fence_handle = 0;
  source.texture_handles[0] = source.texture_handles[1] = source.texture_handles[2] = 0;
  EXPECT_TRUE(valid_source_metadata(source));
  EXPECT_FALSE(valid_metadata(source));
  EXPECT_TRUE(valid_source_metadata(valid_bridge_metadata()));

  // The bare identity published after the producer stops is no source.
  auto identity = source;
  identity.source_width = identity.source_height = identity.packed_width = identity.packed_height = 0;
  identity.dxgi_format = 0;
  EXPECT_FALSE(valid_source_metadata(identity));
  for (const auto mutate : std::array<void (*)(metadata_t &), 6> {
         [](metadata_t &m) {
           m.window = 0;
         },
         [](metadata_t &m) {
           m.packed_width = m.source_width;
         },
         [](metadata_t &m) {
           m.generation = max_generation + 1;
         },
         [](metadata_t &m) {
           m.color_transfer = transfer::scrgb;
         },
         [](metadata_t &m) {
           m.dxgi_format = 24;
           m.color_transfer = transfer::pq;
         },
         [](metadata_t &m) {
           m.protocol_version = pq_version + 1;
         },
       }) {
    auto candidate = source;
    mutate(candidate);
    EXPECT_FALSE(valid_source_metadata(candidate));
  }
}

TEST(ReShadeBridgeProtocol, RejectsHalfSbsOversizedAndColorIncoherentMetadata) {
  using namespace ::reshade_bridge;
  auto metadata = valid_bridge_metadata();
  EXPECT_TRUE(valid_metadata(metadata));
  metadata.packed_width = metadata.source_width;
  EXPECT_FALSE(valid_metadata(metadata));
  metadata = valid_bridge_metadata();
  metadata.source_width = std::numeric_limits<std::uint32_t>::max();
  EXPECT_FALSE(valid_metadata(metadata));
  metadata = valid_bridge_metadata();
  --metadata.source_height;
  --metadata.packed_height;
  EXPECT_FALSE(valid_metadata(metadata));
  metadata = valid_bridge_metadata();
  metadata.color_transfer = transfer::scrgb;
  EXPECT_FALSE(valid_metadata(metadata));
  metadata.dxgi_format = 10;
  EXPECT_TRUE(valid_metadata(metadata));
  metadata.color_transfer = static_cast<transfer>(3);
  EXPECT_FALSE(valid_metadata(metadata));
  metadata.color_transfer = static_cast<transfer>(4);
  metadata.dxgi_format = 24;
  metadata.protocol_version = pq_version;
  EXPECT_FALSE(valid_metadata(metadata));
}

TEST(ReShadeBridgeProtocol, PqTransferRequiresTenBitStorageAndProtocolThree) {
  using namespace ::reshade_bridge;
  auto metadata = valid_bridge_metadata();
  metadata.color_transfer = transfer::pq;
  metadata.dxgi_format = 24;  // R10G10B10A2_UNORM
  metadata.protocol_version = pq_version;
  EXPECT_TRUE(valid_metadata(metadata));
  // Older consumers know protocol 2 only; a PQ declaration there is malformed.
  for (const auto protocol : {screen_plane_version, version}) {
    metadata.protocol_version = protocol;
    EXPECT_FALSE(valid_metadata(metadata)) << protocol;
  }
  metadata.protocol_version = pq_version;
  for (const auto format : {10u, 28u, 87u, 0u}) {
    metadata.dxgi_format = format;
    EXPECT_FALSE(valid_metadata(metadata)) << format;
  }
  // Protocol 3 carries the other transfers unchanged.
  metadata = valid_bridge_metadata();
  metadata.protocol_version = pq_version;
  EXPECT_TRUE(valid_metadata(metadata));
  metadata.color_transfer = transfer::scrgb;
  metadata.dxgi_format = 10;
  EXPECT_TRUE(valid_metadata(metadata));
  metadata.dxgi_format = 24;
  EXPECT_FALSE(valid_metadata(metadata));
  EXPECT_TRUE(supported_version(pq_version));
  EXPECT_FALSE(supported_version(pq_version + 1));
}

TEST(ReShadeBridgeProtocol, CapabilitiesBindToTheNonceTheyWereWrittenWith) {
  using namespace ::reshade_bridge;
  EXPECT_EQ(offsetof(shared_state_t, capability_nonce), 136u);
  EXPECT_EQ(offsetof(shared_state_t, consumer_capabilities), 144u);
  EXPECT_EQ(offsetof(shared_state_t, slots), 192u);
  EXPECT_EQ(sizeof(shared_state_t), 384u);
  EXPECT_EQ(answered_capabilities(7, 7, consumer_accepts_pq), consumer_accepts_pq);
  // A replaced consumer's (stale) capabilities never apply to the nonce being answered, and
  // an old consumer that never writes them reads as zero.
  EXPECT_EQ(answered_capabilities(8, 7, consumer_accepts_pq), 0u);
  EXPECT_EQ(answered_capabilities(8, 0, 0), 0u);
  EXPECT_EQ(answered_capabilities(0, 0, consumer_accepts_pq), 0u);
}

TEST(ReShadeBridgeProtocol, ScalesSameAspectEyesAndRejectsDistortingOutputs) {
  using namespace ::reshade_bridge;
  const auto metadata = valid_bridge_metadata(); // 1920x1080 eyes.
  EXPECT_EQ(fit_output(metadata, 3840, 1080), output_fit::exact);
  EXPECT_EQ(fit_output(metadata, 7680, 2160), output_fit::scaled);
  EXPECT_EQ(fit_output(metadata, 5120, 1440), output_fit::scaled);
  EXPECT_EQ(fit_output(metadata, 3838, 1080), output_fit::scaled); // Rounding-sized differences.
  EXPECT_EQ(fit_output(metadata, 1920, 1080), output_fit::aspect_mismatch); // Half-width SBS.
  EXPECT_EQ(fit_output(metadata, 6880, 1440), output_fit::aspect_mismatch); // 21:9 eyes.
  EXPECT_EQ(fit_output(metadata, 3841, 1080), output_fit::aspect_mismatch); // Odd packed width.
  EXPECT_EQ(fit_output(metadata, 3840, 0), output_fit::aspect_mismatch);
}

TEST(ReShadeBridgeProtocol, PreservesLegacyLayoutAndTreatsLegacyPaddingAsScreenPlane) {
  using namespace ::reshade_bridge;
  EXPECT_EQ(sizeof(metadata_t), 120u);
  EXPECT_EQ(sizeof(slot_t), 64u);
  EXPECT_EQ(sizeof(shared_state_t), 384u);
  EXPECT_EQ(offsetof(shared_state_t, slots), 192u);
  auto metadata = valid_bridge_metadata();
  metadata.protocol_version = screen_plane_version;
  ASSERT_TRUE(valid_metadata(metadata));
  ASSERT_EQ(fit_output(metadata, 3840, 1080), output_fit::exact);

  slot_t legacy;
  legacy.cursor_plane_flags = UINT32_MAX;
  legacy.ui_parallax_uv = std::numeric_limits<float>::quiet_NaN();
  float value = 1.0f;
  ASSERT_TRUE(read_ui_parallax(metadata.protocol_version, legacy, value));
  EXPECT_EQ(value, 0.0f);
}

TEST(ReShadeBridgeProtocol, CarriesResolvedSignedEyeUvWithoutRescalingOrClamping) {
  using namespace ::reshade_bridge;
  slot_t slot;
  float value = 1.0f;
  ASSERT_TRUE(read_ui_parallax(version, slot, value));
  EXPECT_EQ(value, 0.0f);

  slot.cursor_plane_flags = cursor_plane_present;
  for (const auto expected : {-maximum_ui_parallax_uv, -0.00375f, 0.0f, 0.00375f, maximum_ui_parallax_uv}) {
    slot.ui_parallax_uv = expected;
    ASSERT_TRUE(read_ui_parallax(version, slot, value));
    EXPECT_EQ(value, expected);
  }
}

TEST(ReShadeBridgeProtocol, RejectsMalformedPlaneExtensionsAndUnknownVersions) {
  using namespace ::reshade_bridge;
  slot_t slot;
  slot.cursor_plane_flags = cursor_plane_present;
  for (const auto invalid : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), 0.040001f, -0.040001f}) {
    slot.ui_parallax_uv = invalid;
    float value = 1.0f;
    EXPECT_FALSE(read_ui_parallax(version, slot, value));
    EXPECT_EQ(value, 0.0f);
  }
  slot.ui_parallax_uv = 0.005f;
  for (const auto flags : {0u, 2u, cursor_plane_present | 2u, UINT32_MAX}) {
    slot.cursor_plane_flags = flags;
    float value = 1.0f;
    EXPECT_FALSE(read_ui_parallax(version, slot, value));
    EXPECT_EQ(value, 0.0f);
  }
  slot.cursor_plane_flags = cursor_plane_present;
  slot.ui_parallax_uv = 0.005f;
  float pq_value = 0.0f;
  EXPECT_TRUE(read_ui_parallax(pq_version, slot, pq_value));
  EXPECT_EQ(pq_value, 0.005f);
  for (const auto invalid_version : {0u, pq_version + 1u, UINT32_MAX}) {
    float value = 1.0f;
    EXPECT_FALSE(read_ui_parallax(invalid_version, slot, value));
    EXPECT_EQ(value, 0.0f);
  }
}

#ifdef _WIN32

  #include "src/platform/windows/reshade_bridge.h"

  #include <d3d11_4.h>
  #include <d3d12.h>
  #include <dxgi1_2.h>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;
  namespace protocol = ::reshade_bridge;
  namespace receiver = platf::reshade_bridge;
  using bridge_clock_t = std::chrono::steady_clock;

  struct shared_ring_t {
    std::array<ComPtr<ID3D11Texture2D>, protocol::slot_count> textures;
    std::array<HANDLE, protocol::slot_count> handles {};
    ComPtr<ID3D11Fence> ready_fence;
    HANDLE fence_handle = nullptr;

    ~shared_ring_t() {
      for (auto handle : handles) {
        if (handle) {
          CloseHandle(handle);
        }
      }
      if (fence_handle) {
        CloseHandle(fence_handle);
      }
    }
  };

  class ReShadeBridgeGpu: public testing::Test {
  protected:
    static constexpr int source_width = 4;
    static constexpr int packed_width = source_width * 2;
    static constexpr int height = 2;
    const RECT source_rect {100, 200, 100 + source_width, 200 + height};

    void SetUp() override {
      constexpr D3D_FEATURE_LEVEL requested = D3D_FEATURE_LEVEL_11_0;
      D3D_FEATURE_LEVEL actual {};
      const auto created = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &requested, 1, D3D11_SDK_VERSION, &producer, &actual, &producer_context);
      if (FAILED(created)) {
        GTEST_SKIP() << "Hardware D3D11 is unavailable: " << std::hex << created;
      }
      if (FAILED(producer.As(&producer5)) || FAILED(producer_context.As(&producer_context4))) {
        GTEST_SKIP() << "D3D11 shared-fence interfaces are unavailable";
      }
      ComPtr<IDXGIDevice> dxgi_device;
      ASSERT_EQ(producer.As(&dxgi_device), S_OK);
      ComPtr<IDXGIAdapter> adapter;
      ASSERT_EQ(dxgi_device->GetAdapter(&adapter), S_OK);
      DXGI_ADAPTER_DESC adapter_desc {};
      ASSERT_EQ(adapter->GetDesc(&adapter_desc), S_OK);
      std::memcpy(&adapter_luid, &adapter_desc.AdapterLuid, sizeof(adapter_luid));
      ASSERT_EQ(D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, &requested, 1, D3D11_SDK_VERSION, &consumer, &actual, &consumer_context), S_OK);

      const auto name = std::wstring(protocol::mapping_prefix) + std::to_wstring(GetCurrentProcessId());
      mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(protocol::shared_state_t), name.c_str());
      ASSERT_NE(mapping, nullptr);
      ASSERT_NE(GetLastError(), ERROR_ALREADY_EXISTS);
      state = static_cast<protocol::shared_state_t *>(MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(protocol::shared_state_t)));
      ASSERT_NE(state, nullptr);
      *state = protocol::shared_state_t {};
      FILETIME creation {}, exit {}, kernel {}, user {};
      ASSERT_TRUE(GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user));
      std::memcpy(&creation_time, &creation, sizeof(creation_time));
      observed.status = platf::foreground_window::status_e::ok;
      observed.process_id = GetCurrentProcessId();
      observed.window = 0x12345;
      observed.client_screen_rect = {source_rect.left, source_rect.top, source_rect.right, source_rect.bottom};
      protocol::metadata_t identity;
      identity.producer_pid = GetCurrentProcessId();
      identity.producer_creation_time = creation_time;
      identity.window = observed.window;
      write_metadata(identity);
      bridge = make_receiver();
      EXPECT_FALSE(poll());
      ASSERT_NE(state->consumer_nonce, 0u);
      ASSERT_TRUE(new_generation());
    }

    void TearDown() override {
      if (copy_gate_active) {
        signal_ready(copy_gate_value);
      }
      bridge.reset();
      rings.clear();
      if (state) {
        UnmapViewOfFile(state);
      }
      if (mapping) {
        CloseHandle(mapping);
      }
    }

    std::unique_ptr<receiver::receiver_t> make_receiver() {
      return std::make_unique<receiver::receiver_t>(consumer.Get(), consumer_context.Get(), [this]() {
        return observed;
      });
    }

    std::optional<receiver::frame_t> poll() {
      return bridge->poll(source_rect, packed_width, height);
    }

    void write_metadata(const protocol::metadata_t &metadata) {
      InterlockedIncrement(reinterpret_cast<volatile LONG *>(&state->metadata_sequence));
      MemoryBarrier();
      state->metadata = metadata;
      MemoryBarrier();
      InterlockedIncrement(reinterpret_cast<volatile LONG *>(&state->metadata_sequence));
    }

    bool new_generation(
      DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM,
      protocol::transfer transfer = protocol::transfer::srgb,
      int render_width = source_width,
      int render_height = height,
      std::uint32_t protocol_version = protocol::version
    ) {
      auto ring = std::make_unique<shared_ring_t>();
      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = render_width * 2;
      desc.Height = render_height;
      desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
      desc.Format = format;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
      desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
      for (std::size_t i = 0; i < protocol::slot_count; ++i) {
        if (FAILED(producer->CreateTexture2D(&desc, nullptr, &ring->textures[i]))) {
          return false;
        }
        ComPtr<IDXGIResource1> resource;
        if (FAILED(ring->textures[i].As(&resource)) || FAILED(resource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr, &ring->handles[i]))) {
          return false;
        }
      }
      if (FAILED(producer5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&ring->ready_fence))) || FAILED(ring->ready_fence->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &ring->fence_handle))) {
        return false;
      }

      auto metadata = valid_bridge_metadata();
      metadata.producer_pid = GetCurrentProcessId();
      metadata.producer_creation_time = creation_time;
      metadata.window = observed.window;
      metadata.adapter_luid = adapter_luid;
      metadata.generation = rings.size() + 1;
      metadata.accepted_consumer_nonce = state->consumer_nonce;
      metadata.source_width = render_width;
      metadata.source_height = render_height;
      metadata.packed_width = render_width * 2;
      metadata.packed_height = render_height;
      metadata.dxgi_format = format;
      metadata.color_transfer = transfer;
      metadata.protocol_version = protocol_version;
      for (std::size_t i = 0; i < protocol::slot_count; ++i) {
        metadata.texture_handles[i] = reinterpret_cast<std::uintptr_t>(ring->handles[i]);
        state->slots[i] = protocol::slot_t {};
        state->slots[i].control = protocol::slot_control(metadata.generation, protocol::slot_state::free);
      }
      metadata.ready_fence_handle = reinterpret_cast<std::uintptr_t>(ring->fence_handle);
      rings.push_back(std::move(ring));
      write_metadata(metadata);
      return true;
    }

    std::array<std::uint32_t, packed_width * height> pixels(std::uint32_t left, std::uint32_t right) {
      std::array<std::uint32_t, packed_width * height> result {};
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < packed_width; ++x) {
          result[y * packed_width + x] = x < source_width ? left : right;
        }
      }
      return result;
    }

    void publish(std::uint64_t sequence, bool signal = true, float ui_parallax_uv = 0.0f, std::uint32_t cursor_plane_flags = protocol::cursor_plane_present) {
      const auto source_pixels = pixels(0xFF0000FF, 0xFFFF0000);
      publish_pixels(sequence, source_pixels.data(), packed_width * sizeof(std::uint32_t), signal, ui_parallax_uv, cursor_plane_flags);
    }

    // Like the producer, write any free slot of the current generation. The receiver holds the
    // slot of its newest frame and returns a replaced one only after its reads complete, so a
    // free slot may take a few polls to appear.
    void publish_pixels(std::uint64_t sequence, const void *source_pixels, UINT row_pitch, bool signal = true, float ui_parallax_uv = 0.0f, std::uint32_t cursor_plane_flags = protocol::cursor_plane_present) {
      const auto generation = state->metadata.generation;
      const auto free_control = static_cast<LONG64>(protocol::slot_control(generation, protocol::slot_state::free));
      const auto writing_control = static_cast<LONG64>(protocol::slot_control(generation, protocol::slot_state::writing));
      int index = -1;
      const auto deadline = bridge_clock_t::now() + std::chrono::seconds(2);
      while (index < 0 && bridge_clock_t::now() < deadline) {
        for (int i = 0; i < static_cast<int>(protocol::slot_count) && index < 0; ++i) {
          if (InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&state->slots[i].control), writing_control, free_control) == free_control) {
            index = i;
          }
        }
        if (index < 0) {
          poll();
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      }
      ASSERT_GE(index, 0) << "No free export slot";
      last_slot = index;
      auto &slot = state->slots[index];
      producer_context->UpdateSubresource(rings.back()->textures[index].Get(), 0, nullptr, source_pixels, row_pitch, 0);
      slot.sequence = sequence;
      LARGE_INTEGER qpc {};
      ASSERT_TRUE(QueryPerformanceCounter(&qpc));
      slot.qpc = qpc.QuadPart;
      slot.cursor_plane_flags = cursor_plane_flags;
      slot.ui_parallax_uv = ui_parallax_uv;
      if (signal) {
        signal_ready(sequence);
      }
      MemoryBarrier();
      InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&slot.control), static_cast<LONG64>(protocol::slot_control(generation, protocol::slot_state::ready)));
    }

    void signal_ready(std::uint64_t sequence) {
      ASSERT_EQ(producer_context4->Signal(rings.back()->ready_fence.Get(), sequence), S_OK);
      producer_context->Flush();
    }

    std::optional<receiver::frame_t> await_frame() {
      const auto deadline = bridge_clock_t::now() + std::chrono::seconds(2);
      do {
        if (auto frame = poll()) {
          return frame;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } while (bridge_clock_t::now() < deadline);
      return std::nullopt;
    }

    std::optional<receiver::frame_t> await_frame_after(std::uint64_t sequence) {
      const auto deadline = bridge_clock_t::now() + std::chrono::seconds(2);
      do {
        if (auto frame = poll(); frame && frame->sequence > sequence) {
          return frame;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } while (bridge_clock_t::now() < deadline);
      return std::nullopt;
    }

    bool await_slot(int index, protocol::slot_state expected) {
      const auto deadline = bridge_clock_t::now() + std::chrono::seconds(2);
      do {
        poll();
        if (protocol::control_state(state->slots[index].control) == expected) {
          return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } while (bridge_clock_t::now() < deadline);
      return false;
    }

    // The receiver consumed the last published slot: it holds a valid frame (`reading`) or
    // returned a malformed one (`free`).
    bool await_consumed() {
      const auto deadline = bridge_clock_t::now() + std::chrono::seconds(2);
      do {
        poll();
        if (protocol::control_state(state->slots[last_slot].control) != protocol::slot_state::ready) {
          return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } while (bridge_clock_t::now() < deadline);
      return false;
    }

    // Blocks the consumer's GPU queue behind a producer fence value, so reads recorded after
    // this point cannot complete until the test signals it.
    void hold_consumer_reads(std::uint64_t gate) {
      ComPtr<ID3D11Device5> consumer5;
      ComPtr<ID3D11DeviceContext4> consumer_context4;
      ASSERT_EQ(consumer.As(&consumer5), S_OK);
      ASSERT_EQ(consumer_context.As(&consumer_context4), S_OK);
      ASSERT_EQ(consumer5->OpenSharedFence(rings.back()->fence_handle, IID_PPV_ARGS(&consumer_fence)), S_OK);
      ASSERT_EQ(consumer_context4->Wait(consumer_fence.Get(), gate), S_OK);
      copy_gate_value = gate;
      copy_gate_active = true;
    }

    void expect_pixels(ID3D11Texture2D *texture, DXGI_FORMAT expected_format = DXGI_FORMAT_R8G8B8A8_UNORM, std::uint32_t left = 0xFF0000FF, std::uint32_t right = 0xFFFF0000) {
      const auto expected = pixels(left, right);
      expect_pixel_bytes(texture, expected_format, expected.data(), packed_width * sizeof(std::uint32_t));
    }

    void expect_pixel_bytes(ID3D11Texture2D *texture, DXGI_FORMAT expected_format, const void *expected, UINT expected_pitch) {
      D3D11_TEXTURE2D_DESC desc {};
      texture->GetDesc(&desc);
      ASSERT_EQ(desc.Width, packed_width);
      ASSERT_EQ(desc.Height, height);
      ASSERT_EQ(desc.Format, expected_format);
      desc.Usage = D3D11_USAGE_STAGING;
      desc.BindFlags = desc.MiscFlags = 0;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Texture2D> staging;
      ASSERT_EQ(consumer->CreateTexture2D(&desc, nullptr, &staging), S_OK);
      consumer_context->CopyResource(staging.Get(), texture);
      D3D11_MAPPED_SUBRESOURCE mapped {};
      ASSERT_EQ(consumer_context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped), S_OK);
      for (int row = 0; row < height; ++row) {
        EXPECT_EQ(std::memcmp(static_cast<std::uint8_t *>(mapped.pData) + row * mapped.RowPitch, static_cast<const std::uint8_t *>(expected) + row * expected_pitch, expected_pitch), 0);
      }
      consumer_context->Unmap(staging.Get(), 0);
    }

    ComPtr<ID3D11Device> producer, consumer;
    ComPtr<ID3D11Device5> producer5;
    ComPtr<ID3D11DeviceContext> producer_context, consumer_context;
    ComPtr<ID3D11DeviceContext4> producer_context4;
    std::vector<std::unique_ptr<shared_ring_t>> rings;
    HANDLE mapping = nullptr;
    protocol::shared_state_t *state = nullptr;
    std::uint64_t creation_time = 0, adapter_luid = 0;
    platf::foreground_window::observation_t observed;
    std::unique_ptr<receiver::receiver_t> bridge;
    bool copy_gate_active = false;
    std::uint64_t copy_gate_value = 0;
    ComPtr<ID3D11Fence> consumer_fence;
    int last_slot = 0;
  };
}  // namespace

TEST_F(ReShadeBridgeGpu, AdvertisesPqBeforeItsNonceAndImportsAPqGeneration) {
  // SetUp's first poll wrote the request: capabilities, their nonce, then consumer_nonce.
  EXPECT_EQ(state->consumer_capabilities, protocol::consumer_accepts_pq);
  EXPECT_EQ(state->capability_nonce, state->consumer_nonce);
  EXPECT_EQ(protocol::answered_capabilities(state->consumer_nonce, state->capability_nonce, state->consumer_capabilities), protocol::consumer_accepts_pq);

  // A PQ declaration at protocol 2 is malformed and never presented.
  ASSERT_TRUE(new_generation(DXGI_FORMAT_R10G10B10A2_UNORM, protocol::transfer::pq));
  const auto codes = pixels(0xC00003FF, 0xFFF00000);
  publish_pixels(1, codes.data(), packed_width * sizeof(std::uint32_t));
  for (int i = 0; i < 20; ++i) {
    EXPECT_FALSE(poll());
  }

  ASSERT_TRUE(new_generation(DXGI_FORMAT_R10G10B10A2_UNORM, protocol::transfer::pq, source_width, height, protocol::pq_version));
  publish_pixels(1, codes.data(), packed_width * sizeof(std::uint32_t));
  const auto frame = await_frame();
  ASSERT_TRUE(frame);
  EXPECT_EQ(frame->transfer, receiver::transfer_e::pq);
  EXPECT_FALSE(frame->linear);
  expect_pixels(frame->texture, DXGI_FORMAT_R10G10B10A2_UNORM, 0xC00003FF, 0xFFF00000);
}

TEST_F(ReShadeBridgeGpu, ConvertsFromTheHeldSlotAndReturnsItOnlyAfterItsReadsComplete) {
  publish(1, false, 0.007f);
  const int first_slot = last_slot;
  const auto started = bridge_clock_t::now();
  for (int i = 0; i < 20; ++i) {
    EXPECT_FALSE(poll());
  }
  EXPECT_LT(bridge_clock_t::now() - started, std::chrono::milliseconds(500));
  EXPECT_EQ(protocol::control_state(state->slots[first_slot].control), protocol::slot_state::ready);
  signal_ready(1);
  const auto frame = await_frame();
  ASSERT_TRUE(frame);
  EXPECT_EQ(frame->sequence, 1u);
  EXPECT_EQ(frame->ui_parallax_uv, 0.007f);
  EXPECT_FALSE(frame->linear);
  EXPECT_EQ(frame->transfer, receiver::transfer_e::srgb);
  EXPECT_NE(frame->timestamp, bridge_clock_t::time_point {});
  // No private copy: the frame is the shared slot itself, held while it is the newest frame.
  // Repeat conversions keep reading it, and the producer cannot reclaim it.
  for (int i = 0; i < 20; ++i) {
    const auto retained = poll();
    ASSERT_TRUE(retained);
    EXPECT_EQ(retained->sequence, frame->sequence);
    EXPECT_EQ(retained->texture, frame->texture);
    EXPECT_EQ(retained->view, frame->view);
    EXPECT_EQ(retained->ui_parallax_uv, frame->ui_parallax_uv);
    EXPECT_EQ(protocol::control_state(state->slots[first_slot].control), protocol::slot_state::reading);
  }
  expect_pixels(frame->texture);

  // A read recorded before the replacement is gated on the consumer GPU. The replaced slot must
  // stay `reading` until that read has completed, then return to the producer without a wait.
  hold_consumer_reads(100);
  const auto gated = pixels(0xFF00FF00, 0xFF00FF00);
  publish_pixels(2, gated.data(), packed_width * sizeof(std::uint32_t), false);
  const int second_slot = last_slot;
  EXPECT_NE(second_slot, first_slot);
  signal_ready(2);
  const auto replacing = bridge_clock_t::now();
  auto second = await_frame_after(1);
  ASSERT_TRUE(second);
  for (int i = 0; i < 20; ++i) {
    second = poll();
    ASSERT_TRUE(second);
    EXPECT_EQ(second->sequence, 2u);
    EXPECT_EQ(protocol::control_state(state->slots[first_slot].control), protocol::slot_state::reading);
  }
  EXPECT_LT(bridge_clock_t::now() - replacing, std::chrono::milliseconds(500));
  EXPECT_NE(second->texture, frame->texture);
  EXPECT_EQ(protocol::control_state(state->slots[second_slot].control), protocol::slot_state::reading);
  signal_ready(100);
  copy_gate_active = false;
  ASSERT_TRUE(await_slot(first_slot, protocol::slot_state::free));
  EXPECT_EQ(protocol::control_state(state->slots[second_slot].control), protocol::slot_state::reading);
  expect_pixels(second->texture, DXGI_FORMAT_R8G8B8A8_UNORM, 0xFF00FF00, 0xFF00FF00);

  // A frame dropped by a transient metadata replacement is not presented again, but its slot
  // stays held until a newer frame replaces it.
  InterlockedIncrement(reinterpret_cast<volatile LONG *>(&state->metadata_sequence));
  EXPECT_FALSE(poll());
  InterlockedIncrement(reinterpret_cast<volatile LONG *>(&state->metadata_sequence));
  EXPECT_FALSE(poll());
  EXPECT_EQ(protocol::control_state(state->slots[second_slot].control), protocol::slot_state::reading);
  publish(3, false);  // The gate already completed fence value 100.
  const auto third = await_frame();
  ASSERT_TRUE(third);
  EXPECT_EQ(third->sequence, 3u);
  ASSERT_TRUE(await_slot(second_slot, protocol::slot_state::free));
}

TEST_F(ReShadeBridgeGpu, WakesWhenTheExportFenceCompletesAndRetiresWithoutAnotherPoll) {
  struct event_t {
    HANDLE value = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    ~event_t() {
      if (value) {
        CloseHandle(value);
      }
    }
  } woken;
  ASSERT_NE(woken.value, nullptr);
  bridge->set_frame_wake([event = woken.value]() {
    SetEvent(event);
  });
  EXPECT_FALSE(bridge->frame_wake_active());
  EXPECT_TRUE(bridge->frame_pending());  // Not live yet: the owner keeps polling.

  publish(1);
  const auto first = await_frame();
  ASSERT_TRUE(first);
  const int first_slot = last_slot;
  EXPECT_TRUE(bridge->frame_wake_active());
  EXPECT_FALSE(bridge->frame_pending());
  WaitForSingleObject(woken.value, 0);  // Drop a wake for the frame just polled, if any.

  // A ready slot whose GPU work has not completed neither wakes nor reads as pending.
  publish(2, false);
  EXPECT_EQ(WaitForSingleObject(woken.value, 50), static_cast<DWORD>(WAIT_TIMEOUT));
  EXPECT_FALSE(bridge->frame_pending());
  signal_ready(2);
  ASSERT_EQ(WaitForSingleObject(woken.value, 2000), static_cast<DWORD>(WAIT_OBJECT_0));
  EXPECT_TRUE(bridge->frame_pending());
  const auto second = poll();
  ASSERT_TRUE(second);
  EXPECT_EQ(second->sequence, 2u);
  EXPECT_FALSE(bridge->frame_pending());

  // The replaced slot returns once its reads complete, without waiting for another poll.
  EXPECT_EQ(protocol::control_state(state->slots[first_slot].control), protocol::slot_state::reading);
  const auto deadline = bridge_clock_t::now() + std::chrono::seconds(2);
  while (protocol::control_state(state->slots[first_slot].control) != protocol::slot_state::free && bridge_clock_t::now() < deadline) {
    bridge->retire();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_EQ(protocol::control_state(state->slots[first_slot].control), protocol::slot_state::free);

  // Each completion re-arms the wake for the next frame.
  publish(3);
  ASSERT_EQ(WaitForSingleObject(woken.value, 2000), static_cast<DWORD>(WAIT_OBJECT_0));
  EXPECT_TRUE(bridge->frame_pending());
  const auto third = poll();
  ASSERT_TRUE(third);
  EXPECT_EQ(third->sequence, 3u);
  EXPECT_FALSE(bridge->frame_pending());

  // A metadata replacement is a pending change even without a fence advance.
  InterlockedIncrement(reinterpret_cast<volatile LONG *>(&state->metadata_sequence));
  InterlockedIncrement(reinterpret_cast<volatile LONG *>(&state->metadata_sequence));
  EXPECT_TRUE(bridge->frame_pending());
  ASSERT_TRUE(poll());
  EXPECT_FALSE(bridge->frame_pending());

  // A new generation re-arms on its own fence.
  ASSERT_TRUE(new_generation());
  EXPECT_TRUE(bridge->frame_pending());
  publish(1);
  const auto replaced = await_frame();
  ASSERT_TRUE(replaced);
  EXPECT_TRUE(bridge->frame_wake_active());
  WaitForSingleObject(woken.value, 0);
  publish(2);
  ASSERT_EQ(WaitForSingleObject(woken.value, 2000), static_cast<DWORD>(WAIT_OBJECT_0));
  EXPECT_TRUE(bridge->frame_pending());  // Not a leftover signal of the released fence.
}

TEST_F(ReShadeBridgeGpu, StatusObservesTheForegroundSourceWithoutAttaching) {
  // A status observer withdraws an earlier consumer connection instead of keeping the ring alive.
  ASSERT_NE(state->consumer_nonce, 0u);
  ASSERT_TRUE(bridge->status(source_rect, packed_width, height));
  EXPECT_EQ(state->consumer_nonce, 0u);

  // Without a consumer the producer publishes only its source description, and no ring.
  bridge = make_receiver();
  state->capability_nonce = 0;
  state->consumer_capabilities = 0;
  auto source = state->metadata;
  source.generation = 0;
  source.accepted_consumer_nonce = 0;
  source.ready_fence_handle = 0;
  for (auto &handle : source.texture_handles) {
    handle = 0;
  }
  write_metadata(source);
  for (int i = 0; i < 5; ++i) {
    const auto observed_source = bridge->status(source_rect, packed_width, height);
    ASSERT_TRUE(observed_source);
    EXPECT_EQ(observed_source->producer_process_id, GetCurrentProcessId());
    EXPECT_EQ(observed_source->producer_creation_time, creation_time);
    EXPECT_EQ(observed_source->resource_generation, 0u);
  }
  EXPECT_EQ(state->consumer_nonce, 0u);
  EXPECT_EQ(state->capability_nonce, 0u);
  EXPECT_EQ(state->consumer_capabilities, 0u);

  // The eyes fit under poll()'s rules: same aspect scales, another aspect does not.
  EXPECT_TRUE(bridge->status(source_rect, packed_width * 2, height * 2));
  EXPECT_FALSE(bridge->status(source_rect, packed_width, height * 2));

  // Another adapter, another foreground window, or the bare identity of a stopped producer.
  auto elsewhere = source;
  elsewhere.adapter_luid ^= 1;
  write_metadata(elsewhere);
  EXPECT_FALSE(bridge->status(source_rect, packed_width, height));
  write_metadata(source);
  observed.window = 0x777;
  EXPECT_FALSE(bridge->status(source_rect, packed_width, height));
  observed.window = source.window;
  EXPECT_TRUE(bridge->status(source_rect, packed_width, height));
  protocol::metadata_t identity;
  identity.producer_pid = source.producer_pid;
  identity.producer_creation_time = source.producer_creation_time;
  identity.window = source.window;
  write_metadata(identity);
  EXPECT_FALSE(bridge->status(source_rect, packed_width, height));
  EXPECT_EQ(state->consumer_nonce, 0u);
}

TEST_F(ReShadeBridgeGpu, KeepsCursorPlaneWithItsFrameAcrossPendingInvalidAndReplacedExports) {
  publish(1, true, 0.006f);
  const auto first = await_frame();
  ASSERT_TRUE(first);
  ASSERT_TRUE(await_consumed());
  EXPECT_EQ(first->sequence, 1u);
  EXPECT_EQ(first->ui_parallax_uv, 0.006f);

  publish(2, false, 0.012f);
  for (int i = 0; i < 20; ++i) {
    const auto retained = poll();
    ASSERT_TRUE(retained);
    EXPECT_EQ(retained->sequence, first->sequence);
    EXPECT_EQ(retained->ui_parallax_uv, first->ui_parallax_uv);
  }
  EXPECT_EQ(protocol::control_state(state->slots[last_slot].control), protocol::slot_state::ready);
  signal_ready(2);
  ASSERT_TRUE(await_consumed());
  const auto second = poll();
  ASSERT_TRUE(second);
  EXPECT_EQ(second->sequence, 2u);
  EXPECT_EQ(second->ui_parallax_uv, 0.012f);

  // Invalid metadata must not advance either the retained pixels or their cursor plane.
  const auto invalid_pixels = pixels(0xFF00FF00, 0xFF00FF00);
  publish_pixels(3, invalid_pixels.data(), packed_width * sizeof(std::uint32_t), true, std::numeric_limits<float>::quiet_NaN());
  ASSERT_TRUE(await_consumed());
  EXPECT_EQ(protocol::control_state(state->slots[last_slot].control), protocol::slot_state::free);
  const auto after_invalid = poll();
  ASSERT_TRUE(after_invalid);
  EXPECT_EQ(after_invalid->sequence, second->sequence);
  EXPECT_EQ(after_invalid->ui_parallax_uv, second->ui_parallax_uv);
  expect_pixels(after_invalid->texture);

  publish(4, true, -0.004f);
  ASSERT_TRUE(await_consumed());
  const auto recovered = poll();
  ASSERT_TRUE(recovered);
  EXPECT_EQ(recovered->sequence, 4u);
  EXPECT_EQ(recovered->ui_parallax_uv, -0.004f);

  ASSERT_TRUE(new_generation());
  publish(1);
  const auto replacement = await_frame();
  ASSERT_TRUE(replacement);
  EXPECT_NE(replacement->resource_generation, recovered->resource_generation);
  EXPECT_EQ(replacement->sequence, 1u);
  EXPECT_EQ(replacement->ui_parallax_uv, 0.0f);
}

TEST_F(ReShadeBridgeGpu, ImportsLegacyPublisherWithUnspecifiedSlotPaddingAtScreenPlane) {
  auto metadata = state->metadata;
  metadata.protocol_version = protocol::screen_plane_version;
  write_metadata(metadata);
  publish(1, true, std::numeric_limits<float>::quiet_NaN(), UINT32_MAX);
  const auto legacy = await_frame();
  ASSERT_TRUE(legacy);
  EXPECT_EQ(legacy->sequence, 1u);
  EXPECT_EQ(legacy->ui_parallax_uv, 0.0f);
  ASSERT_TRUE(await_consumed());
  expect_pixels(legacy->texture);

  ASSERT_TRUE(new_generation());
  publish(1, true, 0.009f);
  const auto current = await_frame();
  ASSERT_TRUE(current);
  EXPECT_NE(current->resource_generation, legacy->resource_generation);
  EXPECT_EQ(current->ui_parallax_uv, 0.009f);
}

TEST_F(ReShadeBridgeGpu, PublisherIdentityStaysStableAcrossFramesAndTracksResourceReplacement) {
  publish(1);
  const auto first = await_frame();
  ASSERT_TRUE(first);
  EXPECT_EQ(first->producer_process_id, GetCurrentProcessId());
  EXPECT_EQ(first->producer_creation_time, creation_time);
  EXPECT_EQ(first->resource_generation, state->metadata.generation);
  ASSERT_TRUE(await_consumed());

  publish(2);
  ASSERT_TRUE(await_consumed());
  const auto second = poll();
  ASSERT_TRUE(second);
  EXPECT_EQ(second->sequence, 2u);
  EXPECT_EQ(second->producer_process_id, first->producer_process_id);
  EXPECT_EQ(second->producer_creation_time, first->producer_creation_time);
  EXPECT_EQ(second->resource_generation, first->resource_generation);

  ASSERT_TRUE(new_generation());
  publish(1);
  const auto replacement = await_frame();
  ASSERT_TRUE(replacement);
  // Frame sequence can restart at the same value; readiness must track resource identity.
  EXPECT_EQ(replacement->sequence, first->sequence);
  EXPECT_EQ(replacement->producer_process_id, first->producer_process_id);
  EXPECT_EQ(replacement->producer_creation_time, first->producer_creation_time);
  EXPECT_NE(replacement->resource_generation, first->resource_generation);
  EXPECT_EQ(replacement->resource_generation, state->metadata.generation);

  observed.status = platf::foreground_window::status_e::no_foreground;
  EXPECT_FALSE(poll());
}

TEST_F(ReShadeBridgeGpu, RejectsFocusSizeAndProcessIdentityChanges) {
  publish(1);
  ASSERT_TRUE(await_frame());
  // Exercise output admission while this exact producer still owns an acknowledged nonce.
  EXPECT_FALSE(bridge->poll(source_rect, packed_width / 2, height));
  observed.status = platf::foreground_window::status_e::no_foreground;
  EXPECT_FALSE(poll());
  observed.status = platf::foreground_window::status_e::ok;
  observed.client_screen_rect.right -= 1;
  EXPECT_FALSE(poll());
  observed.client_screen_rect.right += 1;
  EXPECT_FALSE(poll());
  ASSERT_TRUE(new_generation());
  auto metadata = state->metadata;
  ++metadata.producer_creation_time;
  write_metadata(metadata);
  publish(1);
  EXPECT_FALSE(poll());
}

TEST_F(ReShadeBridgeGpu, KeepsExactGameEyesAcrossFullscreenDisplayScalingAndGenerationRecovery) {
  constexpr int render_width = 3840;
  constexpr int render_height = 2160;
  constexpr int authored_width = render_width * 2;
  ASSERT_TRUE(new_generation(DXGI_FORMAT_R8G8B8A8_UNORM, protocol::transfer::srgb, render_width, render_height));
  const auto initial_generation = state->metadata.generation;
  RECT capture {100, 200, 100 + 4800, 200 + 2700};
  observed.client_screen_rect = {capture.left, capture.top, capture.right, capture.bottom};
  std::vector<std::uint32_t> authored_pixels(static_cast<std::size_t>(authored_width) * render_height);
  for (int y = 0; y < render_height; ++y) {
    for (int x = 0; x < authored_width; ++x) {
      authored_pixels[static_cast<std::size_t>(y) * authored_width + x] = x < render_width ? 0xFF0000FF : 0xFFFF0000;
    }
  }
  publish_pixels(1, authored_pixels.data(), authored_width * sizeof(std::uint32_t));

  // Eyes with a different aspect ratio would distort disparity when scaled; they stay unavailable.
  EXPECT_FALSE(bridge->poll(capture, authored_width, render_height * 2));
  EXPECT_FALSE(bridge->poll(capture, render_width, render_height));

  const auto await_frame_for = [&](int width, int height) {
    const auto deadline = bridge_clock_t::now() + std::chrono::seconds(2);
    do {
      if (auto frame = bridge->poll(capture, width, height)) {
        return frame;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (bridge_clock_t::now() < deadline);
    return std::optional<receiver::frame_t> {};
  };
  const auto await_game_frame = [&]() {
    return await_frame_for(authored_width, render_height);
  };
  const auto expect_native_eyes = [&](ID3D11Texture2D *texture) {
    D3D11_TEXTURE2D_DESC desc {};
    texture->GetDesc(&desc);
    ASSERT_EQ(desc.Width, authored_width);
    ASSERT_EQ(desc.Height, render_height);
    ASSERT_EQ(desc.Format, DXGI_FORMAT_R8G8B8A8_UNORM);
    desc.Width = 4;
    desc.Height = 1;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = desc.MiscFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> staging;
    ASSERT_EQ(consumer->CreateTexture2D(&desc, nullptr, &staging), S_OK);
    const std::array<UINT, 4> sample_x {0, render_width - 1, render_width, authored_width - 1};
    for (UINT i = 0; i < sample_x.size(); ++i) {
      const D3D11_BOX texel {sample_x[i], render_height / 2, 0, sample_x[i] + 1, render_height / 2 + 1, 1};
      consumer_context->CopySubresourceRegion(staging.Get(), 0, i, 0, 0, texture, 0, &texel);
    }
    D3D11_MAPPED_SUBRESOURCE mapped {};
    ASSERT_EQ(consumer_context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped), S_OK);
    const std::array<std::uint32_t, 4> expected {0xFF0000FF, 0xFF0000FF, 0xFFFF0000, 0xFFFF0000};
    EXPECT_EQ(std::memcmp(mapped.pData, expected.data(), sizeof(expected)), 0);
    consumer_context->Unmap(staging.Get(), 0);
  };
  const auto scaled_fullscreen = await_game_frame();
  ASSERT_TRUE(scaled_fullscreen);
  expect_native_eyes(scaled_fullscreen->texture);

  // A same-aspect stream of another size receives the native authored eyes; the consumer's
  // linear sampling scales both halves identically. Rounding-sized differences also scale.
  for (const auto &[width, height] : std::array<std::pair<int, int>, 3> {{{4800 * 2, 2700}, {authored_width - 2, render_height}, {authored_width, render_height - 2}}}) {
    const auto scaled = await_frame_for(width, height);
    ASSERT_TRUE(scaled);
    expect_native_eyes(scaled->texture);
  }

  // A different desktop aspect still owns the same exact authored 4K eyes; do not stretch them.
  capture.right = capture.left + 5120;
  capture.bottom = capture.top + 2160;
  observed.client_screen_rect = {capture.left, capture.top, capture.right, capture.bottom};
  const auto wide_fullscreen = await_game_frame();
  ASSERT_TRUE(wide_fullscreen);
  EXPECT_EQ(wide_fullscreen->resource_generation, initial_generation);
  expect_native_eyes(wide_fullscreen->texture);

  // A matching export does not authorize a partial-desktop or unfocused publisher.
  const auto previous_nonce = state->consumer_nonce;
  --observed.client_screen_rect.right;
  EXPECT_FALSE(bridge->poll(capture, authored_width, render_height));
  // Detaching withdraws the receiver, so the producer cannot build for a stale nonce.
  EXPECT_EQ(state->consumer_nonce, 0u);
  observed.client_screen_rect.right = capture.right;
  EXPECT_FALSE(bridge->poll(capture, authored_width, render_height));
  EXPECT_NE(state->consumer_nonce, 0u);
  EXPECT_NE(state->consumer_nonce, previous_nonce);
  EXPECT_FALSE(bridge->poll(capture, authored_width, render_height));
  ASSERT_TRUE(new_generation(DXGI_FORMAT_R8G8B8A8_UNORM, protocol::transfer::srgb, render_width, render_height));
  publish_pixels(1, authored_pixels.data(), authored_width * sizeof(std::uint32_t));
  const auto recovered = await_game_frame();
  ASSERT_TRUE(recovered);
  EXPECT_NE(recovered->resource_generation, initial_generation);
  EXPECT_EQ(recovered->producer_process_id, GetCurrentProcessId());
  expect_native_eyes(recovered->texture);
  observed.status = platf::foreground_window::status_e::no_foreground;
  EXPECT_FALSE(bridge->poll(capture, authored_width, render_height));
}

TEST_F(ReShadeBridgeGpu, PreservesScRgbHighlightsAndUsesDeclaredTransferAcrossGenerations) {
  ASSERT_TRUE(new_generation(DXGI_FORMAT_R16G16B16A16_FLOAT, protocol::transfer::scrgb));
  std::array<std::uint64_t, packed_width * height> source_pixels {};
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < packed_width; ++x) {
      // Exact IEEE half components: left (4, 0.5, -0.25, 1), right (12.5, 2, 0, 1).
      // scRGB must preserve both HDR highlights and negative wide-gamut components.
      source_pixels[y * packed_width + x] = x < source_width ? 0x3C00B40038004400ULL : 0x3C00000040004A40ULL;
    }
  }
  const auto row_pitch = packed_width * sizeof(std::uint64_t);
  publish_pixels(1, source_pixels.data(), row_pitch);
  const auto hdr_frame = await_frame();
  ASSERT_TRUE(hdr_frame);
  EXPECT_TRUE(hdr_frame->linear);
  ASSERT_TRUE(await_consumed());
  expect_pixel_bytes(hdr_frame->texture, DXGI_FORMAT_R16G16B16A16_FLOAT, source_pixels.data(), row_pitch);

  // Color declarations cannot be changed underneath an existing resource generation.
  auto metadata = state->metadata;
  metadata.color_transfer = protocol::transfer::srgb;
  write_metadata(metadata);
  EXPECT_FALSE(poll());

  // A float format alone is not proof of HDR. A replacement SDR publisher is admitted
  // without consulting the capture display or the client's HDR output setting.
  ASSERT_TRUE(new_generation(DXGI_FORMAT_R16G16B16A16_FLOAT, protocol::transfer::srgb));
  source_pixels.fill(0x3C00380038003800ULL);  // (0.5, 0.5, 0.5, 1), encoded sRGB.
  publish_pixels(1, source_pixels.data(), row_pitch);
  const auto sdr_frame = await_frame();
  ASSERT_TRUE(sdr_frame);
  EXPECT_FALSE(sdr_frame->linear);
  ASSERT_TRUE(await_consumed());
  expect_pixel_bytes(sdr_frame->texture, DXGI_FORMAT_R16G16B16A16_FLOAT, source_pixels.data(), row_pitch);
}

TEST_F(ReShadeBridgeGpu, ConsumerRestartRequiresNewNonceAndResourceGeneration) {
  publish(1);
  const auto previous = await_frame();
  ASSERT_TRUE(previous);
  const auto previous_nonce = state->consumer_nonce;
  bridge.reset();
  EXPECT_EQ(state->consumer_nonce, 0u);
  bridge = make_receiver();
  EXPECT_FALSE(poll());
  ASSERT_NE(state->consumer_nonce, 0u);
  EXPECT_NE(state->consumer_nonce, previous_nonce);
  EXPECT_NE(state->metadata.accepted_consumer_nonce, state->consumer_nonce);
  ASSERT_TRUE(new_generation());
  publish(1);
  const auto frame = await_frame();
  ASSERT_TRUE(frame);
  EXPECT_EQ(frame->producer_process_id, previous->producer_process_id);
  EXPECT_EQ(frame->producer_creation_time, previous->producer_creation_time);
  EXPECT_NE(frame->resource_generation, previous->resource_generation);
  expect_pixels(frame->texture);

  // Detaching never withdraws the nonce of a replacement receiver.
  constexpr std::uint64_t replacement_nonce = 0x5245504c41434521ULL;
  InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&state->consumer_nonce), static_cast<LONG64>(replacement_nonce));
  bridge.reset();
  EXPECT_EQ(state->consumer_nonce, replacement_nonce);
}

TEST_F(ReShadeBridgeGpu, RejectsUnacknowledgedNonceAndMetadataBeingReplaced) {
  auto metadata = state->metadata;
  metadata.accepted_consumer_nonce ^= 0x10;
  write_metadata(metadata);
  publish(1);
  EXPECT_FALSE(poll());
  metadata.accepted_consumer_nonce = state->consumer_nonce;
  write_metadata(metadata);
  InterlockedIncrement(reinterpret_cast<volatile LONG *>(&state->metadata_sequence));
  EXPECT_FALSE(poll());
  InterlockedIncrement(reinterpret_cast<volatile LONG *>(&state->metadata_sequence));
  EXPECT_TRUE(await_frame());
}

TEST_F(ReShadeBridgeGpu, ImportsD3D12Rgb10A2BothEyesWithoutWaitingForProducerFence) {
  // Load the test-only producer API without adding a D3D12 dependency to the host.
  struct module_t {
    HMODULE handle = LoadLibraryExW(L"d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);

    ~module_t() {
      if (handle) {
        FreeLibrary(handle);
      }
    }
  } module;

  if (!module.handle) {
    GTEST_SKIP() << "D3D12 runtime is unavailable";
  }
  const auto create_device = reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(GetProcAddress(module.handle, "D3D12CreateDevice"));
  ASSERT_NE(create_device, nullptr);
  ComPtr<IDXGIDevice> dxgi_device;
  ComPtr<IDXGIAdapter> adapter;
  ASSERT_EQ(producer.As(&dxgi_device), S_OK);
  ASSERT_EQ(dxgi_device->GetAdapter(&adapter), S_OK);
  ComPtr<ID3D12Device> device12;
  const auto created = create_device(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device12));
  if (FAILED(created)) {
    GTEST_SKIP() << "D3D12 is unavailable on the capture adapter: " << std::hex << created;
  }

  D3D12_COMMAND_QUEUE_DESC queue_desc {};
  queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  ComPtr<ID3D12CommandQueue> queue;
  ComPtr<ID3D12CommandAllocator> allocator;
  ComPtr<ID3D12GraphicsCommandList> commands;
  ASSERT_EQ(device12->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)), S_OK);
  ASSERT_EQ(device12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), S_OK);
  ASSERT_EQ(device12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&commands)), S_OK);

  shared_ring_t shared_handles;
  std::array<ComPtr<ID3D12Resource>, protocol::slot_count> textures;
  ComPtr<ID3D12Fence> ready_fence;
  ASSERT_EQ(device12->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&ready_fence)), S_OK);
  ASSERT_EQ(device12->CreateSharedHandle(ready_fence.Get(), nullptr, GENERIC_ALL, nullptr, &shared_handles.fence_handle), S_OK);
  D3D12_HEAP_PROPERTIES heap {};
  heap.Type = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC desc {};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  desc.Width = packed_width;
  desc.Height = height;
  desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
  desc.Format = DXGI_FORMAT_R10G10B10A2_UNORM;
  desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
  auto metadata = state->metadata;
  ++metadata.generation;
  metadata.dxgi_format = desc.Format;
  metadata.ready_fence_handle = reinterpret_cast<std::uintptr_t>(shared_handles.fence_handle);
  for (std::uint32_t i = 0; i < protocol::slot_count; ++i) {
    ASSERT_EQ(device12->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&textures[i])), S_OK);
    ASSERT_EQ(device12->CreateSharedHandle(textures[i].Get(), nullptr, GENERIC_ALL, nullptr, &shared_handles.handles[i]), S_OK);
    metadata.texture_handles[i] = reinterpret_cast<std::uintptr_t>(shared_handles.handles[i]);
    state->slots[i] = protocol::slot_t {};
    state->slots[i].control = protocol::slot_control(metadata.generation, protocol::slot_state::free);
  }
  write_metadata(metadata);

  D3D12_DESCRIPTOR_HEAP_DESC rtv_desc {};
  rtv_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
  rtv_desc.NumDescriptors = 1;
  ComPtr<ID3D12DescriptorHeap> rtvs;
  ASSERT_EQ(device12->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtvs)), S_OK);
  const auto target = rtvs->GetCPUDescriptorHandleForHeapStart();
  device12->CreateRenderTargetView(textures[0].Get(), nullptr, target);
  D3D12_RESOURCE_BARRIER transition {};
  transition.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  transition.Transition.pResource = textures[0].Get();
  transition.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  transition.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
  transition.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
  commands->ResourceBarrier(1, &transition);
  const float red[4] {1.0f, 0.0f, 0.0f, 1.0f};
  const float blue[4] {0.0f, 0.0f, 1.0f, 1.0f};
  const D3D12_RECT left {0, 0, source_width, height}, right {source_width, 0, packed_width, height};
  commands->ClearRenderTargetView(target, red, 1, &left);
  commands->ClearRenderTargetView(target, blue, 1, &right);
  std::swap(transition.Transition.StateBefore, transition.Transition.StateAfter);
  commands->ResourceBarrier(1, &transition);
  ASSERT_EQ(commands->Close(), S_OK);
  ID3D12CommandList *submission = commands.Get();
  queue->ExecuteCommandLists(1, &submission);

  auto &slot = state->slots[0];
  slot.sequence = 1;
  LARGE_INTEGER qpc {};
  ASSERT_TRUE(QueryPerformanceCounter(&qpc));
  slot.qpc = qpc.QuadPart;
  MemoryBarrier();
  InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&slot.control), static_cast<LONG64>(protocol::slot_control(metadata.generation, protocol::slot_state::ready)));
  const auto started = bridge_clock_t::now();
  for (int i = 0; i < 20; ++i) {
    EXPECT_FALSE(poll());
  }
  EXPECT_LT(bridge_clock_t::now() - started, std::chrono::milliseconds(500));
  EXPECT_EQ(protocol::control_state(slot.control), protocol::slot_state::ready);
  ASSERT_EQ(queue->Signal(ready_fence.Get(), 1), S_OK);
  const auto frame = await_frame();
  ASSERT_TRUE(frame);
  EXPECT_EQ(frame->sequence, 1u);
  EXPECT_FALSE(frame->linear);
  EXPECT_EQ(protocol::control_state(slot.control), protocol::slot_state::reading);
  expect_pixels(frame->texture, DXGI_FORMAT_R10G10B10A2_UNORM, 0xC00003FF, 0xFFF00000);
}

#endif
