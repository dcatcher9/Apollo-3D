// SPDX-License-Identifier: GPL-3.0-only
// Offline check of Game 3D automatic UI detection. Runs the production
// SunshineUIDetection{Tiles,Reduce,Mask}CS passes of a shader file on the
// candidate textures saved in Dump 3D packages and compares each decision with
// a labelled expectation. It replaces live trial and error when a detection
// rule changes: every labelled screen of every game is judged at once.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
  namespace fs = std::filesystem;
  using Microsoft::WRL::ComPtr;
  using json = nlohmann::json;

  void checked(HRESULT result, const std::string &message) {
    if (FAILED(result)) {
      char code[16];
      std::snprintf(code, sizeof(code), "0x%08lx", static_cast<unsigned long>(result));
      throw std::runtime_error(message + " failed: " + code);
    }
  }

  std::string read_text(const fs::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  }

  // Candidate slots in shader order: t11 UI alpha (R), t12 UI color (A),
  // t13 Backbuffer (A), current color alpha (t0 A), t14 HUD-less. An offscreen
  // UI layer from the census (ui_layer_candidate_N) takes the UI color slot
  // with the renderer's premultiplied check.
  const std::map<std::string, unsigned> candidate_bits{
    {"sl_ui_alpha", 1u}, {"sl_ui_color_alpha", 2u}, {"sl_backbuffer", 4u}, {"current", 8u}, {"sl_hudless_color", 16u}};
  bool ui_layer_kind(const std::string &name) { return name.rfind("ui_layer_candidate_", 0) == 0; }

  struct device_t {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
  };

  struct texture_t {
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> srv;
    ComPtr<ID3D11UnorderedAccessView> uav;
  };

  texture_t upload(device_t &gpu, const fs::path &file, const json &artifact) {
    const auto width = artifact.at("width").get<UINT>(), height = artifact.at("height").get<UINT>();
    const auto row = artifact.at("row_bytes").get<UINT>();
    const auto bytes = read_text(file);
    if (bytes.size() != size_t(row) * height) throw std::runtime_error("unexpected size of " + file.string());
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = width; desc.Height = height; desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
    desc.Format = static_cast<DXGI_FORMAT>(artifact.at("dxgi_format").get<unsigned>());
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    const D3D11_SUBRESOURCE_DATA data{bytes.data(), row, 0};
    texture_t result;
    checked(gpu.device->CreateTexture2D(&desc, &data, &result.texture), "upload " + file.filename().string());
    checked(gpu.device->CreateShaderResourceView(result.texture.Get(), nullptr, &result.srv), "view " + file.filename().string());
    return result;
  }

  texture_t target(device_t &gpu, UINT width, UINT height, DXGI_FORMAT format) {
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = width; desc.Height = height; desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
    desc.Format = format;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    texture_t result;
    checked(gpu.device->CreateTexture2D(&desc, nullptr, &result.texture), "target");
    checked(gpu.device->CreateShaderResourceView(result.texture.Get(), nullptr, &result.srv), "target view");
    checked(gpu.device->CreateUnorderedAccessView(result.texture.Get(), nullptr, &result.uav), "target UAV");
    return result;
  }

  template<class T> std::vector<T> download(device_t &gpu, const texture_t &source) {
    D3D11_TEXTURE2D_DESC desc{};
    source.texture->GetDesc(&desc);
    desc.BindFlags = 0; desc.Usage = D3D11_USAGE_STAGING; desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> staging;
    checked(gpu.device->CreateTexture2D(&desc, nullptr, &staging), "staging");
    gpu.context->CopyResource(staging.Get(), source.texture.Get());
    D3D11_MAPPED_SUBRESOURCE mapped{};
    checked(gpu.context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped), "map");
    const size_t per_row = desc.Width * (sizeof(T) == 4 && desc.Format == DXGI_FORMAT_R32_FLOAT ? 1 : 4);
    std::vector<T> values(per_row * desc.Height);
    for (UINT y = 0; y < desc.Height; ++y)
      std::memcpy(values.data() + y * per_row, static_cast<const char *>(mapped.pData) + size_t(y) * mapped.RowPitch, per_row * sizeof(T));
    gpu.context->Unmap(staging.Get(), 0);
    return values;
  }

  struct shaders_t {
    ComPtr<ID3D11ComputeShader> tiles, reduce, mask;
  };

  ComPtr<ID3D11ComputeShader> compile(device_t &gpu, const std::string &source, const char *entry, UINT width, UINT height, unsigned color) {
    const auto w = std::to_string(width), h = std::to_string(height), c = std::to_string(color);
    const D3D_SHADER_MACRO defines[]{{"BUFFER_WIDTH", w.c_str()}, {"BUFFER_HEIGHT", h.c_str()}, {"BUFFER_COLOR_SPACE", c.c_str()}, {nullptr, nullptr}};
    ComPtr<ID3DBlob> code, errors;
    const HRESULT result = D3DCompile(source.data(), source.size(), "game3d_native.hlsl", defines, nullptr, entry, "cs_5_0",
      D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
    if (FAILED(result)) throw std::runtime_error(std::string(entry) + ": " + (errors ? static_cast<const char *>(errors->GetBufferPointer()) : "compile failed"));
    ComPtr<ID3D11ComputeShader> shader;
    checked(gpu.device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &shader), entry);
    return shader;
  }

  struct outcome {
    std::array<std::uint32_t, 24> decision{};
    std::uint64_t ui_pixels = 0, pixels = 0;
  };

  outcome run_case(device_t &gpu, const std::string &shader_source, const fs::path &dump, const json &label) {
    const auto manifest = json::parse(read_text(dump / "manifest.json"));
    std::map<std::string, json> artifacts;
    for (const auto &artifact : manifest.at("artifacts")) artifacts[artifact.at("kind").get<std::string>()] = artifact;
    const auto color = manifest.at("producer_metadata").at("color_space").get<unsigned>();
    const auto load = [&](const std::string &kind) {
      const auto found = artifacts.find(kind);
      if (found == artifacts.end()) throw std::runtime_error(dump.filename().string() + " has no " + kind);
      return upload(gpu, dump / found->second.at("file").get<std::string>(), found->second);
    };
    // The color HUD-less is compared with, whose alpha is the current-color
    // candidate: the tagged Backbuffer of a batch pair, else the presented color.
    const auto paired = label.value("paired", std::string("source_color"));
    const auto paired_color = load(paired);
    // The game's frame size; source_width/height is the host output, which
    // differs when the host scales the eyes.
    const auto width = artifacts.at(paired).at("width").get<UINT>(), height = artifacts.at(paired).at("height").get<UINT>();
    std::array<texture_t, 4> inputs{}; // t11..t14
    unsigned bits = 0, flags = 0;
    for (const auto &kind : label.at("candidates")) {
      const auto name = kind.get<std::string>();
      const bool layer = ui_layer_kind(name);
      const auto bit = layer ? 2u : candidate_bits.at(name);
      bits |= bit;
      if (name == "current") continue;
      const unsigned slot = name == "sl_ui_alpha" ? 0 : name == "sl_ui_color_alpha" || layer ? 1 : name == "sl_backbuffer" ? 2 : 3;
      inputs[slot] = load(name);
      if (layer) {
        const auto format = artifacts.at(name).at("dxgi_format").get<unsigned>();
        flags = format == DXGI_FORMAT_R16G16B16A16_FLOAT || format == DXGI_FORMAT_R32G32B32A32_FLOAT ? 3u : 1u;
      }
    }
    if (label.value("exact", false) && (bits & 16u)) bits |= 32u;
    unsigned trusted = 0;
    for (const auto &index : label.value("trusted", json::array())) trusted |= 1u << index.get<unsigned>();

    shaders_t shaders{compile(gpu, shader_source, "SunshineUIDetectionTilesCS", width, height, color),
      compile(gpu, shader_source, "SunshineUIDetectionReduceCS", width, height, color),
      compile(gpu, shader_source, "SunshineUIDetectionMaskCS", width, height, color)};
    // Sized for every shader revision's statistics rows and decision texels.
    auto statistics = target(gpu, 16, 80, DXGI_FORMAT_R32G32B32A32_UINT);
    auto decision = target(gpu, 6, 1, DXGI_FORMAT_R32G32B32A32_UINT);
    auto mask = target(gpu, width, height, DXGI_FORMAT_R32_FLOAT);
    // Detection constants b2: candidate bits, difference threshold, trusted
    // channels, flags. The threshold matches the renderer's per-format choice.
    const float threshold = paired_color.srv && artifacts.at(paired).at("dxgi_format").get<unsigned>() == DXGI_FORMAT_R10G10B10A2_UNORM ?
      4.f / 1023.f : color == 2 ? .005f : 2.f / 255.f;
    struct { std::uint32_t bits; float threshold; std::uint32_t trusted, flags; } constants{bits, threshold, trusted, flags};
    D3D11_BUFFER_DESC buffer{};
    buffer.ByteWidth = sizeof(constants); buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    const D3D11_SUBRESOURCE_DATA initial{&constants, 0, 0};
    ComPtr<ID3D11Buffer> cb;
    checked(gpu.device->CreateBuffer(&buffer, &initial, &cb), "constants");

    auto &context = *gpu.context.Get();
    const auto stage = [&](ID3D11ComputeShader *shader, ID3D11ShaderResourceView *t10, UINT uav_slot, ID3D11UnorderedAccessView *uav, UINT x, UINT y) {
      std::array<ID3D11ShaderResourceView *, 15> views{};
      views[0] = paired_color.srv.Get();
      views[10] = t10;
      for (unsigned i = 0; i < 4; ++i) views[11 + i] = inputs[i].srv.Get();
      context.CSSetShader(shader, nullptr, 0);
      context.CSSetShaderResources(0, UINT(views.size()), views.data());
      ID3D11Buffer *buffers[3]{nullptr, nullptr, cb.Get()};
      context.CSSetConstantBuffers(0, 3, buffers);
      context.CSSetUnorderedAccessViews(uav_slot, 1, &uav, nullptr);
      context.Dispatch(x, y, 1);
      ID3D11UnorderedAccessView *none = nullptr;
      context.CSSetUnorderedAccessViews(uav_slot, 1, &none, nullptr);
      std::array<ID3D11ShaderResourceView *, 15> cleared{};
      context.CSSetShaderResources(0, UINT(cleared.size()), cleared.data());
    };
    stage(shaders.tiles.Get(), nullptr, 6, statistics.uav.Get(), 16, 16);
    stage(shaders.reduce.Get(), statistics.srv.Get(), 6, decision.uav.Get(), 1, 1);
    stage(shaders.mask.Get(), decision.srv.Get(), 0, mask.uav.Get(), (width + 7) / 8, (height + 7) / 8);

    outcome result;
    const auto words = download<std::uint32_t>(gpu, decision);
    std::copy_n(words.begin(), result.decision.size(), result.decision.begin());
    const auto values = download<float>(gpu, mask);
    result.pixels = values.size();
    for (const float value : values) result.ui_pixels += value > 0.f ? 1u : 0u;
    return result;
  }

  std::string mask_class(const outcome &o) {
    return o.ui_pixels == 0 ? "empty" : o.ui_pixels == o.pixels ? "flat" : "partial";
  }
}

int main(int argc, char **argv) {
  try {
    if (argc != 3) {
      std::fprintf(stderr, "Usage: ui_detection_replay <game3d_native.hlsl> <cases.json>\n"
        "cases.json: {\"dump_root\": dir, \"cases\": [{\"dump\", \"label\", \"candidates\": [kinds|\"current\"], "
        "\"paired\": kind, \"exact\": bool, \"trusted\": [candidate indices], \"expect\": {\"mask\", \"source\": [ids]}}]}\n");
      return 2;
    }
    const auto shader_source = read_text(argv[1]);
    const auto cases_path = fs::absolute(argv[2]);
    const auto document = json::parse(read_text(cases_path));
    fs::path root = document.value("dump_root", cases_path.parent_path().string());
    device_t gpu;
    const D3D_FEATURE_LEVEL level = D3D_FEATURE_LEVEL_11_0;
    checked(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &level, 1, D3D11_SDK_VERSION,
      &gpu.device, nullptr, &gpu.context), "D3D11CreateDevice");
    unsigned passed = 0, failed = 0;
    for (const auto &label : document.at("cases")) {
      const auto dump = root / label.at("dump").get<std::string>();
      const auto name = label.value("label", dump.filename().string());
      outcome result;
      try {
        result = run_case(gpu, shader_source, dump, label);
      } catch (const std::exception &error) {
        ++failed;
        std::printf("FAIL %-44s %s\n", name.c_str(), error.what());
        continue;
      }
      const auto &d = result.decision;
      const auto expect = label.at("expect");
      // "mask" is one class or a list of acceptable classes.
      std::string wanted_mask;
      bool okay = false;
      for (const auto &wanted : expect.at("mask").is_array() ? expect.at("mask") : json::array({expect.at("mask")})) {
        wanted_mask += (wanted_mask.empty() ? "" : "|") + wanted.get<std::string>();
        okay = okay || mask_class(result) == wanted.get<std::string>();
      }
      if (expect.contains("source")) {
        bool any = false;
        for (const auto &source : expect.at("source")) any = any || source.get<std::uint32_t>() == d[0];
        okay = okay && any;
      }
      (okay ? passed : failed) += 1;
      std::printf("%s %-44s source=%u covered=%u/%u ui=%.2f%% mask=%s (want %s) candidates=0x%x trusted=0x%x "
        "alpha_covered=%u/%u/%u/%u hudless={changed=%u unchanged=%u invalid=%u tiles=%u lit=%u}\n",
        okay ? "PASS" : "FAIL", name.c_str(), d[0], d[1], d[2], 100.0 * double(result.ui_pixels) / double(result.pixels),
        mask_class(result).c_str(), wanted_mask.c_str(), d[4], d[17], d[8], d[9], d[10], d[11], d[5], d[6], d[7], d[3], d[16]);
    }
    std::printf("%s UI detection replay: %u passed, %u failed\n", failed ? "FAIL" : "PASS", passed, failed);
    return failed ? 1 : 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 2;
  }
}
