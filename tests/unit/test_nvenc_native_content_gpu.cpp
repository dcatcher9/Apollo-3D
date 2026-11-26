/**
 * @file tests/unit/test_nvenc_native_content_gpu.cpp
 * @brief Opt-in offscreen native NVENC artifact producer; the integration runner decodes its output.
 */
#include "../tests_common.h"

#include "src/nvenc/nvenc_d3d11_native.h"
#include "src/platform/windows/misc.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dxgi1_2.h>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;
  using json = nlohmann::json;
  constexpr UINT width = 480;
  constexpr UINT height = 256;
  constexpr unsigned frame_count = 12;

  void require(bool condition, const std::string &message) {
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  struct profile_t {
    const char *name;
    int codec;
    bool ten_bit;
    unsigned depth;
  };

  struct authored_frame_t {
    std::array<std::vector<std::uint16_t>, 3> planes;
    std::vector<std::uint8_t> canonical;
  };

  authored_frame_t author(bool pattern_b, bool ten_bit) {
    // Four constant interiors in every plane expose stale images, plane swaps and pitch errors.
    constexpr std::uint16_t values[3][4] {{48, 96, 144, 208}, {64, 96, 160, 192}, {192, 160, 96, 64}};
    authored_frame_t result;
    for (unsigned plane = 0; plane < 3; ++plane) {
      const auto plane_width = plane == 0 ? width : width / 2;
      const auto plane_height = plane == 0 ? height : height / 2;
      auto &samples = result.planes[plane];
      samples.reserve(plane_width * plane_height);
      for (UINT y = 0; y < plane_height; ++y) {
        for (UINT x = 0; x < plane_width; ++x) {
          const unsigned quadrant = (y >= plane_height / 2 ? 2 : 0) + (x >= plane_width / 2 ? 1 : 0);
          const unsigned source = pattern_b ? 3 - quadrant : quadrant;
          const auto sample = static_cast<std::uint16_t>(ten_bit ? values[plane][source] * 4 + (plane + quadrant + 1) % 4 : values[plane][source]);
          samples.push_back(sample);
          result.canonical.push_back(static_cast<std::uint8_t>(sample));
          if (ten_bit) {
            result.canonical.push_back(static_cast<std::uint8_t>(sample >> 8));
          }
        }
      }
    }
    return result;
  }

  struct device_t {
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    D3D_FEATURE_LEVEL feature_level {};
    json description;
  };

  device_t create_device() {
    device_t result;
    ComPtr<IDXGIFactory1> factory;
    require(SUCCEEDED(CreateDXGIFactory1(IID_IDXGIFactory1, reinterpret_cast<void **>(factory.GetAddressOf()))), "Cannot create the offscreen DXGI factory");
    DXGI_ADAPTER_DESC1 description {};
    for (UINT index = 0;; ++index) {
      ComPtr<IDXGIAdapter1> candidate;
      const auto status = factory->EnumAdapters1(index, candidate.GetAddressOf());
      if (status == DXGI_ERROR_NOT_FOUND) {
        break;
      }
      require(SUCCEEDED(status), "Cannot enumerate the offscreen NVIDIA adapter");
      require(SUCCEEDED(candidate->GetDesc1(&description)), "Cannot describe the offscreen adapter");
      if (description.VendorId == 0x10de && !(description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
        result.adapter = std::move(candidate);
        break;
      }
    }
    require(result.adapter != nullptr, "An opted-in native content run requires a hardware NVIDIA adapter");
    constexpr D3D_FEATURE_LEVEL levels[] {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    require(SUCCEEDED(D3D11CreateDevice(result.adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, levels, std::size(levels), D3D11_SDK_VERSION, result.device.GetAddressOf(), &result.feature_level, result.context.GetAddressOf())), "Cannot create the offscreen NVIDIA D3D11 device");
    result.description = {
      {"vendor_id", description.VendorId},
      {"device_id", description.DeviceId},
      {"subsystem_id", description.SubSysId},
      {"revision", description.Revision},
      {"luid_low", description.AdapterLuid.LowPart},
      {"luid_high", description.AdapterLuid.HighPart},
      {"description", platf::to_utf8(description.Description)}
    };
    return result;
  }

  ComPtr<ID3D11Texture2D> create_upload(device_t &device, DXGI_FORMAT format, const authored_frame_t &frame, UINT &row_pitch) {
    D3D11_TEXTURE2D_DESC description {};
    description.Width = width;
    description.Height = height;
    description.MipLevels = 1;
    description.ArraySize = 1;
    description.Format = format;
    description.SampleDesc.Count = 1;
    description.Usage = D3D11_USAGE_STAGING;
    description.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    ComPtr<ID3D11Texture2D> texture;
    require(SUCCEEDED(device.device->CreateTexture2D(&description, nullptr, texture.GetAddressOf())), "Cannot create a planar staging upload");
    D3D11_MAPPED_SUBRESOURCE mapped {};
    require(SUCCEEDED(device.context->Map(texture.Get(), 0, D3D11_MAP_WRITE, 0, &mapped)), "Cannot map a planar staging upload");
    auto unmap = util::fail_guard([&]() {
      device.context->Unmap(texture.Get(), 0);
    });
    const bool ten_bit = format == DXGI_FORMAT_P010;
    const unsigned bytes_per_sample = ten_bit ? 2 : 1;
    require(mapped.RowPitch >= width * bytes_per_sample, "Mapped planar RowPitch is smaller than a logical row");
    row_pitch = mapped.RowPitch;
    // NV12/P010 staging storage is RowPitch * (height + height / 2), including row padding.
    std::memset(mapped.pData, 0, static_cast<std::size_t>(mapped.RowPitch) * (height + height / 2));
    const auto store = [&](std::uint8_t *destination, std::uint16_t sample) {
      if (ten_bit) {
        const auto word = static_cast<std::uint16_t>(sample << 6);
        destination[0] = static_cast<std::uint8_t>(word);
        destination[1] = static_cast<std::uint8_t>(word >> 8);
      } else {
        destination[0] = static_cast<std::uint8_t>(sample);
      }
    };
    for (UINT y = 0; y < height; ++y) {
      auto *row = static_cast<std::uint8_t *>(mapped.pData) + static_cast<std::size_t>(y) * mapped.RowPitch;
      for (UINT x = 0; x < width; ++x) {
        store(row + x * bytes_per_sample, frame.planes[0][y * width + x]);
      }
    }
    for (UINT y = 0; y < height / 2; ++y) {
      auto *row = static_cast<std::uint8_t *>(mapped.pData) + static_cast<std::size_t>(height + y) * mapped.RowPitch;
      for (UINT x = 0; x < width / 2; ++x) {
        store(row + (2 * x) * bytes_per_sample, frame.planes[1][y * (width / 2) + x]);
        store(row + (2 * x + 1) * bytes_per_sample, frame.planes[2][y * (width / 2) + x]);
      }
    }
    device.context->Unmap(texture.Get(), 0);
    unmap.disable();
    return texture;
  }

  void append(std::ofstream &file, const std::vector<std::uint8_t> &bytes) {
    file.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    require(file.good(), "Cannot write a native content artifact");
  }

  json run_owner(device_t &device, const profile_t &profile, const std::filesystem::path &root, bool rebuilt) {
    const auto name = rebuilt ? "rebuilt" : "first";
    const auto origin = rebuilt ? 2000ULL : 1000ULL;
    const auto stream_name = std::string(name) + (profile.codec == 1 ? ".hevc" : ".obu");
    const auto expected_name = std::string(name) + ".expected.yuv";
    std::ofstream stream(root / stream_name, std::ios::binary);
    std::ofstream expected(root / expected_name, std::ios::binary);
    require(stream.good() && expected.good(), "Cannot open the native content artifacts");
    const auto frame_a = author(false, profile.ten_bit);
    const auto frame_b = author(true, profile.ten_bit);
    const auto format = profile.ten_bit ? DXGI_FORMAT_P010 : DXGI_FORMAT_NV12;
    UINT pitch_a = 0, pitch_b = 0;
    auto upload_a = create_upload(device, format, frame_a, pitch_a);
    auto upload_b = create_upload(device, format, frame_b, pitch_b);
    auto encoder = std::make_unique<nvenc::nvenc_d3d11_native>(device.device.Get());
    video::config_t client {};
    client.width = width;
    client.height = height;
    client.framerate = 30;
    client.framerateX100 = 3000;
    client.encodingFramerate = 30000;
    client.bitrate = 20000;
    client.slicesPerFrame = 1;
    client.numRefFrames = 1;
    client.encoderCscMode = 2;
    client.videoFormat = profile.codec;
    client.dynamicRange = profile.ten_bit ? 1 : 0;
    // Only selects native pipeline depth; this fixture never constructs a renderer/provider/model.
    client.sbs_mode = profile.depth == 2 ? video::SBS_GAME_MONO : video::SBS_OFF;
    nvenc::nvenc_config config;
    config.quality_preset = 1;
    config.two_pass = nvenc::nvenc_two_pass::disabled;
    config.vbv_percentage_increase = 100;
    config.adaptive_quantization = false;
    config.hevc_unidirectional_b = false;
    config.split_encode = false;
    const nvenc::nvenc_colorspace_t colorspace {NV_ENC_VUI_COLOR_PRIMARIES_BT709, NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709, NV_ENC_VUI_MATRIX_COEFFS_BT709, false};
    require(encoder->create_encoder(config, client, colorspace, profile.ten_bit ? NV_ENC_BUFFER_FORMAT_YUV420_10BIT : NV_ENC_BUFFER_FORMAT_NV12, std::nullopt), "The actual native NVENC encoder did not initialize");
    require(encoder->pipeline_depth() == profile.depth, "Native NVENC fell back from the requested pipeline depth");
    const auto surface_count = encoder->input_surface_count();
    require(surface_count == (profile.depth == 1 ? 1u : 3u), "Native NVENC did not allocate the requested registered surface ring");
    std::vector<ID3D11Texture2D *> textures;
    for (unsigned surface = 0; surface < surface_count; ++surface) {
      auto *texture = encoder->get_input_texture(surface);
      require(texture != nullptr && std::find(textures.begin(), textures.end(), texture) == textures.end(), "Native input surfaces are missing or alias the same allocation");
      D3D11_TEXTURE2D_DESC description {};
      texture->GetDesc(&description);
      require(description.Width == width && description.Height == height && description.Format == format && description.ArraySize == 1 && description.MipLevels == 1, "Native input surface dimensions/format differ from the requested profile");
      textures.push_back(texture);
    }
    require(encoder->get_input_texture() == textures.front(), "The initial view target is not native surface zero");

    using mappings_t = std::array<std::optional<unsigned>, nvenc::nvenc_base::max_pipeline_depth>;
    const auto mappings = [&]() {
      mappings_t result {};
      for (unsigned slot = 0; slot < profile.depth; ++slot) {
        result[slot] = encoder->input_surface_for_picture_for_test(slot);
        require(!result[slot] || *result[slot] < surface_count, "A picture mapped a surface outside its native ring");
      }
      require(!result[0] || !result[1] || result[0] != result[1], "Two pictures mapped the same input surface");
      return result;
    };
    const auto mappings_json = [&](const mappings_t &values) {
      auto result = json::array();
      for (unsigned slot = 0; slot < profile.depth; ++slot) {
        result.push_back(values[slot] ? json(*values[slot]) : json(nullptr));
      }
      return result;
    };
    require(mappings() == mappings_t {}, "A fresh native owner inherited mapped picture surfaces");

    json result {
      {"name", name},
      {"initial_pattern", rebuilt ? "B" : "A"},
      {"frame_index_origin", origin},
      {"pipeline_depth", encoder->pipeline_depth()},
      {"input_storage", "registered-surface-ring-v1"},
      {"input_surface_count", surface_count},
      {"copy_path_removed", true},
      {"stream_file", stream_name},
      {"expected_file", expected_name},
      {"upload_row_pitches", {{{"pattern", "A"}, {"row_pitch", pitch_a}}, {{"pattern", "B"}, {"row_pitch", pitch_b}}}},
      {"returned_indices", json::array()},
      {"returned_idr", json::array()},
      {"timeline", json::array()}
    };

    struct pending_t {
      std::uint64_t index;
      bool pattern_b;
      bool idr;
      unsigned slot;
      unsigned surface;
    };

    std::deque<pending_t> pending;
    unsigned peak = 0, refusals = 0, repeat_refusals = 0, submitted = 0;
    std::uint64_t revision = 0;
    std::optional<unsigned> converted_surface;
    const auto check_pending_mappings = [&]() {
      const auto observed = mappings();
      for (const auto &picture : pending) {
        require(observed[picture.slot] == picture.surface, "An in-flight picture lost its actual input-surface mapping");
      }
    };
    const auto write_target = [&](bool pattern_b) {
      require(profile.depth > 1 || encoder->frames_in_flight() == 0, "A depth-one target rewrite would race a submitted picture");
      const auto before = mappings();
      const auto surface = encoder->begin_input_conversion();
      require(surface && *surface < surface_count, "No unmapped native surface was available for the next conversion");
      require(std::find(before.begin(), before.end(), surface) == before.end(), "The producer selected a still-mapped native input");
      auto *target = encoder->get_input_texture(*surface);
      require(target == textures[*surface], "A native surface allocation changed during its owner lifetime");
      device.context->CopyResource(target, pattern_b ? upload_b.Get() : upload_a.Get());
      encoder->set_converted_input(*surface);
      converted_surface = surface;
      ++revision;
      require(!encoder->input_waits_for_retirement(), "A newly written unmapped surface still waits for an old picture");
      check_pending_mappings();
      result["timeline"].push_back({{"event", "write"}, {"pattern", pattern_b ? "B" : "A"}, {"authored_revision", revision}, {"input_surface", *surface}, {"before_in_flight", encoder->frames_in_flight()}, {"mapped_surfaces_before", mappings_json(before)}, {"mapped_surfaces_after", mappings_json(mappings())}});
    };
    const auto submit_one = [&](unsigned frame, bool pattern_b, bool retained, bool accept) {
      const auto index = origin + frame;
      const bool idr = frame == 0 || frame == 10;
      const auto before = encoder->frames_in_flight();
      require(converted_surface.has_value(), "A native submission has no committed producer input");
      const auto mapped_before = mappings();
      const bool waits_before = encoder->input_waits_for_retirement();
      const bool accepted = encoder->submit_frame(index, idr);
      const auto after = encoder->frames_in_flight();
      const auto mapped_after = mappings();
      result["timeline"].push_back({{"event", "submit"}, {"frame_index", index}, {"retained", retained}, {"authored_revision", revision}, {"input_surface", *converted_surface}, {"picture_slot", submitted % profile.depth}, {"force_idr", idr}, {"accepted", accepted}, {"before_in_flight", before}, {"after_in_flight", after}, {"input_waits_before", waits_before}, {"mapped_surfaces_before", mappings_json(mapped_before)}, {"mapped_surfaces_after", mappings_json(mapped_after)}});
      require(accepted == accept, "Actual native submission returned an unexpected result (NEED_MORE_INPUT remains public false)");
      if (accept) {
        require(after == before + 1, "Accepted native submission did not acquire exactly one slot");
        const auto slot = submitted++ % profile.depth;
        require(!waits_before, "A native input was accepted while its prior picture was still in flight");
        require(mapped_after[slot] == converted_surface, "Native submission did not map the exact committed producer surface");
        pending.push_back({index, pattern_b, idr, slot, *converted_surface});
        peak = std::max(peak, after);
      } else {
        ++refusals;
        repeat_refusals += retained;
        require(after == before && (before == profile.depth || waits_before), "A refused native submission changed ownership or had no busy input");
      }
      check_pending_mappings();
    };
    const auto retrieve_one = [&]() {
      require(!pending.empty(), "Native fixture lost its pending metadata");
      const auto oldest = pending.front();
      const auto mapped_before = mappings();
      require(mapped_before[oldest.slot] == oldest.surface, "Native retrieval lost the oldest picture's input surface");
      auto packet = encoder->retrieve_frame();
      require(!packet.data.empty(), "Actual native retrieval failed; target will not be rewritten after empty retrieval");
      require(packet.frame_index == oldest.index, "Actual native output order/timestamp differs from its submission");
      require(!oldest.idr || packet.idr, "A forced native IDR was not returned as IDR");
      append(stream, packet.data);
      append(expected, oldest.pattern_b ? frame_b.canonical : frame_a.canonical);
      pending.pop_front();
      require(encoder->frames_in_flight() == pending.size(), "Native output and fixture ownership ledgers diverged");
      check_pending_mappings();
      result["returned_indices"].push_back(packet.frame_index);
      result["returned_idr"].push_back(packet.idr);
      result["timeline"].push_back({{"event", "retrieve"}, {"frame_index", packet.frame_index}, {"expected_pattern", oldest.pattern_b ? "B" : "A"}, {"picture_slot", oldest.slot}, {"input_surface", oldest.surface}, {"idr", packet.idr}, {"bytes", packet.data.size()}, {"after_in_flight", encoder->frames_in_flight()}, {"input_waits_after", encoder->input_waits_for_retirement()}, {"mapped_surfaces_before", mappings_json(mapped_before)}, {"mapped_surfaces_after", mappings_json(mappings())}});
    };
    for (unsigned frame = 0; frame < frame_count; ++frame) {
      const bool pattern_b = ((frame / 4) % 2 != 0) != rebuilt;
      // New conversions can fill two pictures and their spare even when their authored pixels
      // match. Repeats deliberately retain one committed surface and must retire its exact owner.
      const bool changed = frame <= 4 || frame == 8 || frame == 9;
      if (changed) {
        write_target(pattern_b);
      }
      if (profile.depth == 2 && frame == 4) {
        // The writable third surface is separate from both still-owned picture inputs. This is
        // logical fullness, not a claim that the GPU has not already completed those pictures.
        require(encoder->frames_in_flight() == 2, "The authored busy transition was not reached with two owned pictures");
        submit_one(frame, pattern_b, false, false);
      }
      if (profile.depth == 2 && !changed) {
        require(encoder->input_waits_for_retirement(), "The authored repeat did not retain an in-flight input");
        unsigned retired = 0;
        while (encoder->input_waits_for_retirement()) {
          require(retired < profile.depth && !pending.empty(), "Retiring the committed input exceeded the bounded pending ledger");
          submit_one(frame, pattern_b, true, false);
          retrieve_one();
          ++retired;
        }
      }
      if (encoder->frames_in_flight() == profile.depth) {
        retrieve_one();
      }
      submit_one(frame, pattern_b, !changed || (profile.depth == 2 && frame == 4), true);
      if (profile.depth == 1) {
        retrieve_one();
      }
    }
    while (!pending.empty()) {
      retrieve_one();
    }
    require(encoder->frames_in_flight() == 0, "Native content run did not drain every picture");
    require(result["returned_indices"].size() == frame_count, "Native content run did not return every authored frame");
    require(revision == 7, "Native content run lost a committed conversion write");
    require(refusals == (profile.depth == 2 ? 8u : 0u) && repeat_refusals == (profile.depth == 2 ? 7u : 0u), "Native content run did not exercise the exact capacity and repeat-retirement refusals");
    encoder->destroy_encoder();
    require(mappings() == mappings_t {}, "Native destruction left a registered input mapped");
    encoder.reset();
    stream.close();
    expected.close();
    require(!stream.fail() && !expected.fail(), "Closing a native content artifact failed");
    result["conversion_writes"] = revision;
    result["peak_in_flight"] = peak;
    result["busy_refusals"] = refusals;
    result["repeat_refusals"] = repeat_refusals;
    result["stream_bytes"] = std::filesystem::file_size(root / stream_name);
    result["expected_bytes"] = std::filesystem::file_size(root / expected_name);
    result["destroy_completed"] = true;
    return result;
  }

  void run_profile(const profile_t &profile) {
    const auto *enabled = std::getenv("SUNSHINE_RUN_NATIVE_NVENC_TESTS");
    if (!enabled || std::string_view {enabled} != "1") {
      GTEST_SKIP() << "Set SUNSHINE_RUN_NATIVE_NVENC_TESTS=1 and use Invoke-NvencNativeContentBaseline.ps1 for decoded-content qualification.";
    }
    const auto *output = std::getenv("SUNSHINE_NATIVE_NVENC_OUTPUT_DIR");
    require(output && *output, "An opted-in native run requires a new artifact directory");
    const std::filesystem::path root(output);
    require(root.is_absolute() && std::filesystem::create_directory(root), "Native artifact directory must be absolute and new");
    auto device = create_device();
    json result {
      {"schema", 2},
      {"input_storage", "registered-surface-ring-v1"},
      {"case", profile.name},
      {"codec", profile.codec == 1 ? "hevc" : "av1"},
      {"input_format", profile.ten_bit ? "P010" : "NV12"},
      {"pixel_format", profile.ten_bit ? "yuv420p10le" : "yuv420p"},
      {"width", width},
      {"height", height},
      {"frames_per_owner", frame_count},
      {"requested_depth", profile.depth},
      {"adapter", device.description},
      {"nvenc_api", {{"major", NVENCAPI_MAJOR_VERSION}, {"minor", NVENCAPI_MINOR_VERSION}}},
      {"settings", {{"nvenc", {{"quality_preset", 1}, {"two_pass", "disabled"}, {"vbv_percentage_increase", 100}, {"adaptive_quantization", false}, {"hevc_unidirectional_b", false}, {"split_encode", false}}}, {"client", {{"width", width}, {"height", height}, {"framerate", 30}, {"framerateX100", 3000}, {"encodingFramerate", 30000}, {"bitrate", 20000}, {"slicesPerFrame", 1}, {"numRefFrames", 1}, {"encoderCscMode", 2}, {"videoFormat", profile.codec}, {"dynamicRange", profile.ten_bit ? 1 : 0}, {"sbs_mode", profile.depth == 2 ? video::SBS_GAME_MONO : video::SBS_OFF}}}, {"colorspace", {{"primaries", 1}, {"transfer", 1}, {"matrix", 1}, {"full_range", false}}}, {"d3d", {{"create_flags", 0}, {"driver_type", "unknown"}, {"feature_level", static_cast<unsigned>(device.feature_level)}}}}},
      {"owners", json::array()}
    };
    result["owners"].push_back(run_owner(device, profile, root, false));
    result["owners"].push_back(run_owner(device, profile, root, true));
    result["all_destroyed"] = true;
    std::ofstream file(root / "result.json", std::ios::binary);
    file << result.dump(2) << '\n';
    file.close();
    require(!file.fail(), "Cannot publish the completed native content manifest");
    testing::Test::RecordProperty("artifact_directory", root.string());
    testing::Test::RecordProperty("encoded_frames", 2 * frame_count);
  }

  TEST(NativeNvencContentGpuTest, HevcNv12Depth1) {
    run_profile({"HevcNv12Depth1", 1, false, 1});
  }

  TEST(NativeNvencContentGpuTest, HevcNv12Depth2) {
    run_profile({"HevcNv12Depth2", 1, false, 2});
  }

  TEST(NativeNvencContentGpuTest, HevcP010Depth1) {
    run_profile({"HevcP010Depth1", 1, true, 1});
  }

  TEST(NativeNvencContentGpuTest, HevcP010Depth2) {
    run_profile({"HevcP010Depth2", 1, true, 2});
  }

  TEST(NativeNvencContentGpuTest, Av1Nv12Depth1) {
    run_profile({"Av1Nv12Depth1", 2, false, 1});
  }

  TEST(NativeNvencContentGpuTest, Av1Nv12Depth2) {
    run_profile({"Av1Nv12Depth2", 2, false, 2});
  }

  TEST(NativeNvencContentGpuTest, Av1P010Depth1) {
    run_profile({"Av1P010Depth1", 2, true, 1});
  }

  TEST(NativeNvencContentGpuTest, Av1P010Depth2) {
    run_profile({"Av1P010Depth2", 2, true, 2});
  }
}  // namespace
