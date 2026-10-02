// SPDX-License-Identifier: GPL-3.0-only
// Offline check of Game 3D automatic UI detection. Runs the production
// SunshineUIDetection{Tiles,Reduce,Mask}CS passes of a shader file, and its
// SunshineScene{Cells,Compare,Evidence}CS hidden-scene evidence when it has them, on
// the candidate textures saved in Dump 3D packages and compares each decision
// with a labelled expectation. It replaces live trial and error when a detection
// rule changes: every labelled screen of every game is judged at once.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <nlohmann/json.hpp>

#include "game3d_ui_detection_contract.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
  namespace fs = std::filesystem;
  namespace contract = sunshine_game3d::ui_detection;
  namespace word = contract::decision_word;
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

  void write_bytes(const fs::path &path, const void *data, size_t size) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(static_cast<const char *>(data), std::streamsize(size));
    output.close();
    if (!output) throw std::runtime_error("cannot write " + path.string());
  }

  // Candidate slots in shader order: t11 UI alpha (R), t12 UI color (A),
  // t13 Backbuffer (A), current color alpha (t0 A), t14 HUD-less. An offscreen
  // UI layer from the census (ui_layer_candidate_N) takes the UI color slot
  // with the renderer's layer flags.
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

  // An uploaded artifact and its native bytes, kept for the CPU mask reference.
  struct artifact_t {
    texture_t gpu;
    json descriptor;
    std::string bytes;
  };

  artifact_t upload(device_t &gpu, const fs::path &file, const json &artifact) {
    const auto width = artifact.at("width").get<UINT>(), height = artifact.at("height").get<UINT>();
    const auto row = artifact.at("row_bytes").get<UINT>();
    artifact_t result{{}, artifact, read_text(file)};
    if (result.bytes.size() != size_t(row) * height) throw std::runtime_error("unexpected size of " + file.string());
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = width; desc.Height = height; desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
    desc.Format = static_cast<DXGI_FORMAT>(artifact.at("dxgi_format").get<unsigned>());
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    const D3D11_SUBRESOURCE_DATA data{result.bytes.data(), row, 0};
    checked(gpu.device->CreateTexture2D(&desc, &data, &result.gpu.texture), "upload " + file.filename().string());
    checked(gpu.device->CreateShaderResourceView(result.gpu.texture.Get(), nullptr, &result.gpu.srv), "view " + file.filename().string());
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

  float half_to_float(std::uint16_t half) {
    const std::uint32_t sign = std::uint32_t(half & 0x8000u) << 16, exponent = (half >> 10) & 31u, fraction = half & 1023u;
    std::uint32_t bits;
    if (exponent == 31u) bits = sign | 0x7f800000u | (fraction << 13);
    else if (exponent) bits = sign | ((exponent + 112u) << 23) | (fraction << 13);
    else if (!fraction) bits = sign;
    else {
      // Subnormal half: normalize into a float32 exponent.
      std::uint32_t shift = 0, value = fraction;
      while (!(value & 1024u)) { value <<= 1; ++shift; }
      bits = sign | ((113u - shift) << 23) | ((value & 1023u) << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
  }

  // One component of every texel as Texture2D<float4>.Load returns it (absent
  // components read 0, absent alpha 1): the CPU reference for mask_exact.
  // Empty for a format without an exact reference here.
  std::vector<float> component(const artifact_t &a, unsigned channel) {
    const auto format = a.descriptor.at("dxgi_format").get<unsigned>();
    const auto width = a.descriptor.at("width").get<size_t>(), height = a.descriptor.at("height").get<size_t>();
    const auto row = a.descriptor.at("row_bytes").get<size_t>();
    const auto *bytes = reinterpret_cast<const unsigned char *>(a.bytes.data());
    std::vector<float> values(width * height);
    const auto texel = [&](size_t x, size_t y, size_t size) { return bytes + y * row + x * size; };
    for (size_t y = 0; y < height; ++y) for (size_t x = 0; x < width; ++x) {
      float &value = values[y * width + x];
      switch (format) {
        case DXGI_FORMAT_R32G32B32A32_FLOAT: std::memcpy(&value, texel(x, y, 16) + 4 * channel, 4); break;
        case DXGI_FORMAT_R16G16B16A16_FLOAT: {
          std::uint16_t half; std::memcpy(&half, texel(x, y, 8) + 2 * channel, 2); value = half_to_float(half); break;
        }
        case DXGI_FORMAT_R10G10B10A2_UNORM: {
          std::uint32_t packed; std::memcpy(&packed, texel(x, y, 4), 4);
          value = channel == 3 ? float(packed >> 30) / 3.f : float((packed >> (10 * channel)) & 1023u) / 1023.f; break;
        }
        case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
          if (format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB && channel != 3) return {}; // sRGB color is decoded on load.
          value = float(texel(x, y, 4)[channel]) / 255.f; break;
        case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
          if (format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB && channel != 3) return {};
          value = float(texel(x, y, 4)[channel == 3 ? 3 : 2 - channel]) / 255.f; break;
        case DXGI_FORMAT_R32_FLOAT:
          if (channel) value = channel == 3 ? 1.f : 0.f; else std::memcpy(&value, texel(x, y, 4), 4);
          break;
        case DXGI_FORMAT_R16_FLOAT: {
          std::uint16_t half; std::memcpy(&half, texel(x, y, 2), 2);
          value = channel ? (channel == 3 ? 1.f : 0.f) : half_to_float(half); break;
        }
        case DXGI_FORMAT_R16_UNORM: {
          std::uint16_t unorm; std::memcpy(&unorm, texel(x, y, 2), 2);
          value = channel ? (channel == 3 ? 1.f : 0.f) : float(unorm) / 65535.f; break;
        }
        case DXGI_FORMAT_R8_UNORM:
          value = channel ? (channel == 3 ? 1.f : 0.f) : float(*texel(x, y, 1)) / 255.f; break;
        default: return {};
      }
    }
    return values;
  }

  struct shaders_t {
    ComPtr<ID3D11ComputeShader> tiles, reduce, mask, cells, compare, evidence;
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

  // Resource sizes from the shader's markers. Retired revisions without them
  // wrote opaque counts to statistics rows 64-79 and decision texel 5, so
  // every revision fits 80 rows and 6 texels. A shader with scene evidence
  // measures both images and writes decision texels 5 and 6.
  struct sizes_t {
    UINT statistics_rows, decision_texels;
    bool scene;
  };
  sizes_t detection_sizes(const std::string &source) {
    auto texels = sunshine_game3d::shader_marker(source, contract::decision_texels_marker);
    const auto images = sunshine_game3d::shader_marker(source, contract::scene_evidence_images_marker);
    if (!texels) texels = contract::default_decision_texels;
    const bool scene = images == contract::max_scene_evidence_images && texels >= contract::scene_decision_texels;
    if (texels > contract::max_decision_texels || (images && !scene))
      throw std::runtime_error("unsupported detection size markers in the shader");
    return {std::max(80u, contract::statistics_rows(images)), std::max(6u, texels), scene};
  }

  struct outcome {
    std::vector<std::uint32_t> decision;
    std::uint64_t ui_pixels = 0, pixels = 0, mask_hash = 1469598103934665603ull;
    std::string mask_exact; // Empty unless the label asks for it.
    std::vector<float> mask;
    UINT width = 0, height = 0;
    std::uint32_t flags = 0; // The pushed detection flags.
  };

  std::string mask_exact(const outcome &result, const std::array<const artifact_t *, 4> &inputs, const artifact_t &paired) {
    // Raw selected alpha, or a whole-frame flat (sources 6, 8 and 9); a
    // HUD-less difference has no CPU reference here.
    const auto source = result.decision.at(word::source);
    std::vector<float> reference;
    if (!source) reference.assign(result.mask.size(), 0.f);
    else if (source == 6u || source == 8u || source == 9u) reference.assign(result.mask.size(), 1.f);
    else if (source <= 4u) {
      const auto *input = source == 4u ? &paired : inputs[source - 1];
      if (!input) return "no-reference(missing candidate)";
      reference = component(*input, source == 1u ? 0 : 3);
      if (reference.empty()) return "no-reference(format " + input->descriptor.at("dxgi_format").dump() + ")";
    } else return "no-reference(source " + std::to_string(source) + ")";
    if (reference.size() != result.mask.size()) return "differs(extent)";
    std::uint64_t different = 0;
    size_t first = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
      std::uint32_t a, b;
      std::memcpy(&a, &result.mask[i], 4); std::memcpy(&b, &reference[i], 4);
      if (a == b || (std::isnan(result.mask[i]) && std::isnan(reference[i]))) continue;
      if (!different++) first = i;
    }
    if (!different) return "match";
    char text[160];
    std::snprintf(text, sizeof(text), "differs(%llu pixels, first x=%zu y=%zu gpu=%.9g reference=%.9g)",
      static_cast<unsigned long long>(different), first % result.width, first / result.width,
      double(result.mask[first]), double(reference[first]));
    return text;
  }

  outcome run_case(device_t &gpu, const std::string &shader_source, const fs::path &dump, const json &label) {
    const auto manifest = json::parse(read_text(dump / "manifest.json"));
    std::map<std::string, json> artifacts;
    for (const auto &artifact : manifest.at("artifacts")) artifacts[artifact.at("kind").get<std::string>()] = artifact;
    const auto &metadata = manifest.at("producer_metadata");
    const auto color = metadata.at("color_space").get<unsigned>();
    std::map<std::string, artifact_t> loaded;
    const auto load = [&](const std::string &kind) -> const artifact_t & {
      if (const auto found = loaded.find(kind); found != loaded.end()) return found->second;
      const auto found = artifacts.find(kind);
      if (found == artifacts.end()) throw std::runtime_error(dump.filename().string() + " has no " + kind);
      return loaded.emplace(kind, upload(gpu, dump / found->second.at("file").get<std::string>(), found->second)).first->second;
    };
    // The color HUD-less is compared with, whose alpha is the current-color
    // candidate: the tagged Backbuffer of a batch pair, else the presented color.
    const auto paired = label.value("paired", std::string("source_color"));
    const auto &paired_color = load(paired);
    // The presented color, whatever HUD-less is paired with.
    const auto &presented = load("source_color");
    // The game's frame size; source_width/height is the host output, which
    // differs when the host scales the eyes.
    const auto width = artifacts.at(paired).at("width").get<UINT>(), height = artifacts.at(paired).at("height").get<UINT>();
    std::array<const artifact_t *, 4> inputs{}; // t11..t14
    unsigned bits = 0, flags = 0;
    for (const auto &kind : label.at("candidates")) {
      const auto name = kind.get<std::string>();
      const bool layer = ui_layer_kind(name);
      const auto bit = layer ? 2u : candidate_bits.at(name);
      bits |= bit;
      if (name == "current") continue;
      const unsigned slot = name == "sl_ui_alpha" ? 0 : name == "sl_ui_color_alpha" || layer ? 1 : name == "sl_backbuffer" ? 2 : 3;
      inputs[slot] = &load(name);
      if (layer) flags = contract::layer_detection_flags(contract::float_layer_format(artifacts.at(name).at("dxgi_format").get<unsigned>()));
    }
    if (label.value("exact", false) && (bits & 16u)) bits |= 32u;
    unsigned trusted = 0;
    for (const auto &index : label.value("trusted", json::array())) trusted |= 1u << index.get<unsigned>();
    // Per-frame bits a label names; they never select a candidate.
    if (label.value("scene_hold", false)) flags |= contract::per_frame_scene_hold;
    if (label.value("sample", false)) flags |= contract::per_frame_sample;
    if (label.value("depth_not_current", false)) flags |= contract::per_frame_depth_not_current;
    if (label.value("scene_hold_hudless", false)) flags |= contract::per_frame_scene_hold_hudless;

    // b0: the exact 80 bytes the render consumed. Without a raw depth
    // artifact, t1 is a 1x1 placeholder and depth is not ready, as in render().
    std::array<std::uint8_t, 80> parameters{};
    const auto hex = metadata.at("replay").at("parameter_hex").get<std::string>();
    if (hex.size() != parameters.size() * 2) throw std::runtime_error("parameter_hex is not 80 bytes");
    for (size_t i = 0; i < parameters.size(); ++i) parameters[i] = std::uint8_t(std::stoul(hex.substr(i * 2, 2), nullptr, 16));
    texture_t depth;
    if (artifacts.count("raw_depth")) depth = load("raw_depth").gpu;
    else {
      const float zero = 0.f;
      D3D11_TEXTURE2D_DESC desc{};
      desc.Width = desc.Height = desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
      desc.Format = DXGI_FORMAT_R32_FLOAT; desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      const D3D11_SUBRESOURCE_DATA data{&zero, sizeof(zero), 0};
      checked(gpu.device->CreateTexture2D(&desc, &data, &depth.texture), "depth placeholder");
      checked(gpu.device->CreateShaderResourceView(depth.texture.Get(), nullptr, &depth.srv), "depth placeholder view");
      std::memset(parameters.data() + 8, 0, 8); // depth_ready, camera_ready
    }

    const auto sizes = detection_sizes(shader_source);
    shaders_t shaders{compile(gpu, shader_source, "SunshineUIDetectionTilesCS", width, height, color),
      compile(gpu, shader_source, "SunshineUIDetectionReduceCS", width, height, color),
      compile(gpu, shader_source, "SunshineUIDetectionMaskCS", width, height, color)};
    if (sizes.scene) {
      shaders.cells = compile(gpu, shader_source, "SunshineSceneCellsCS", width, height, color);
      shaders.compare = compile(gpu, shader_source, "SunshineSceneCompareCS", width, height, color);
      shaders.evidence = compile(gpu, shader_source, "SunshineSceneEvidenceCS", width, height, color);
    }
    auto statistics = target(gpu, 16, sizes.statistics_rows, DXGI_FORMAT_R32G32B32A32_UINT);
    texture_t cells;
    if (sizes.scene) cells = target(gpu, contract::scene::cells_x, contract::scene::cells_y, DXGI_FORMAT_R32G32B32A32_UINT);
    auto decision = target(gpu, sizes.decision_texels, 1, DXGI_FORMAT_R32G32B32A32_UINT);
    auto mask = target(gpu, width, height, DXGI_FORMAT_R32_FLOAT);
    // Detection constants b2: candidate bits, difference threshold, trusted
    // channels, flags. The threshold matches the renderer's per-format choice.
    const float threshold = artifacts.at(paired).at("dxgi_format").get<unsigned>() == DXGI_FORMAT_R10G10B10A2_UNORM ?
      4.f / 1023.f : color == 2 ? .005f : 2.f / 255.f;
    struct { std::uint32_t bits; float threshold; std::uint32_t trusted, flags; } constants{bits, threshold, trusted, flags};
    const auto constant_buffer = [&](const void *bytes, UINT size) {
      D3D11_BUFFER_DESC buffer{};
      buffer.ByteWidth = size; buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      const D3D11_SUBRESOURCE_DATA initial{bytes, 0, 0};
      ComPtr<ID3D11Buffer> result;
      checked(gpu.device->CreateBuffer(&buffer, &initial, &result), "constants");
      return result;
    };
    const auto geometry = constant_buffer(parameters.data(), UINT(parameters.size()));
    const auto cb = constant_buffer(&constants, sizeof(constants));

    auto &context = *gpu.context.Get();
    const auto stage = [&](ID3D11ComputeShader *shader, ID3D11ShaderResourceView *t10, UINT uav_slot, ID3D11UnorderedAccessView *uav, UINT x, UINT y) {
      std::array<ID3D11ShaderResourceView *, 15> views{};
      views[0] = paired_color.gpu.srv.Get();
      views[1] = depth.srv.Get();
      views[6] = presented.gpu.srv.Get();
      views[10] = t10;
      for (unsigned i = 0; i < 4; ++i) views[11 + i] = inputs[i] ? inputs[i]->gpu.srv.Get() : nullptr;
      context.CSSetShader(shader, nullptr, 0);
      context.CSSetShaderResources(0, UINT(views.size()), views.data());
      ID3D11Buffer *buffers[3]{geometry.Get(), nullptr, cb.Get()};
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
    // The hidden-scene evidence of a sample frame: measured after the
    // decision, it writes decision texels 5 and 6 only.
    if (sizes.scene) {
      stage(shaders.cells.Get(), nullptr, 6, cells.uav.Get(), contract::scene::cells_x / 16, contract::scene::cells_y);
      stage(shaders.compare.Get(), cells.srv.Get(), 6, statistics.uav.Get(), contract::scene::cells_x / 16, contract::scene::cells_y / 16);
      stage(shaders.evidence.Get(), statistics.srv.Get(), 6, decision.uav.Get(), 1, 1);
    }

    outcome result;
    result.decision = download<std::uint32_t>(gpu, decision);
    result.mask = download<float>(gpu, mask);
    result.width = width; result.height = height;
    result.flags = flags;
    result.pixels = result.mask.size();
    for (const float value : result.mask) {
      result.ui_pixels += value > 0.f ? 1u : 0u;
      std::uint32_t word;
      std::memcpy(&word, &value, sizeof(word));
      result.mask_hash = (result.mask_hash ^ word) * 1099511628211ull;
    }
    if (label.at("expect").value("mask_exact", false)) result.mask_exact = mask_exact(result, inputs, paired_color);
    return result;
  }

  std::string mask_class(const outcome &o) {
    return o.ui_pixels == 0 ? "empty" : o.ui_pixels == o.pixels ? "flat" : "partial";
  }

  // expect.scene: the presented image's hidden-scene evidence in decision
  // texel 5; expect.hudless_scene: the HUD-less image's in texel 6. "verdict"
  // is one name or a list (the HUD-less texel's verdict is visible when its
  // valid D reaches the visible bound, else none); "d_min" and "d_max" bound D
  // inclusively. Evidence that did not run, as from a shader without scene
  // evidence, fails the check.
  bool scene_matches(const std::vector<std::uint32_t> &d, const json &expected, bool hudless, std::string &text) {
    if (d.size() <= word::hudless_scene_state) {
      text = "no-reference(" + std::to_string(d.size() / 4) + " decision texels)";
      return false;
    }
    const auto n = d[hudless ? word::hudless_scene_n : word::scene_n], state = d[hudless ? word::hudless_scene_state : word::scene_state];
    float value;
    std::memcpy(&value, &d[hudless ? word::hudless_scene_d : word::scene_d], 4);
    const bool valid = contract::scene_state_valid(state), ran = contract::scene_state_ran(state);
    const auto *verdict = hudless ? (valid && contract::scene_visible(value) ? "visible" : "none") :
      contract::name(contract::scene_state_verdict(state));
    bool okay = ran;
    if (expected.contains("verdict")) {
      bool any = false;
      for (const auto &wanted : expected.at("verdict").is_array() ? expected.at("verdict") : json::array({expected.at("verdict")}))
        any = any || wanted.get<std::string>() == verdict;
      okay = okay && any;
    }
    if (expected.contains("d_min")) okay = okay && value >= expected.at("d_min").get<float>();
    if (expected.contains("d_max")) okay = okay && value <= expected.at("d_max").get<float>();
    char buffer[128];
    std::snprintf(buffer, sizeof(buffer), "%s(%s n=%u d=%.4f valid=%u ran=%u)", okay ? "match" : "differs", verdict, n,
      double(value), unsigned(valid), unsigned(ran));
    text = buffer;
    return okay;
  }

  // A copy of the dump whose consumed mask is the one this replay resolved,
  // for replay_game3d_dump --shader. The captured package is never modified.
  void write_mask(const fs::path &directory, const fs::path &dump, const std::string &name, const outcome &result,
      const fs::path &shader) {
    auto manifest = json::parse(read_text(dump / "manifest.json"));
    const auto &record = manifest.at("producer_metadata").at("replay");
    json *consumed = nullptr;
    for (auto &artifact : manifest.at("artifacts")) if (artifact.at("kind") == "ui_source_color") consumed = &artifact;
    if (!consumed || consumed->at("dxgi_format") != DXGI_FORMAT_R32_FLOAT || consumed->at("width") != result.width ||
        consumed->at("height") != result.height || consumed->at("row_bytes") != result.width * 4 ||
        record.value("ui_alpha_source", std::string{}) != "ui_source_color" ||
        record.at("ui_constant_binding").value("mask_channel", std::string{}) != "red")
      throw std::runtime_error("--write-mask needs a dump that consumed an automatic R32 mask of the detection extent");
    if (fs::exists(directory)) throw std::runtime_error(directory.string() + " already exists");
    fs::create_directories(directory);
    for (const auto &artifact : manifest.at("artifacts")) {
      const auto file = artifact.at("file").get<std::string>();
      if (fs::path(file).filename().string() != file) throw std::runtime_error("artifact outside the dump: " + file);
      if (artifact.at("kind") == "ui_source_color") write_bytes(directory / file, result.mask.data(), result.mask.size() * sizeof(float));
      else fs::copy_file(dump / file, directory / file);
    }
    manifest["ui_detection_replay_mask"] = {{"source_dump", dump.string()}, {"label", name}, {"shader", shader.string()},
      {"decision", result.decision}, {"flags", result.flags},
      {"meaning", "ui_source_color is the R32 mask ui_detection_replay resolved with this shader; every other artifact and the "
        "metadata are the captured ones. flags are the detection flags it pushed."}};
    const auto text = manifest.dump(2) + '\n';
    write_bytes(directory / "manifest.json", text.data(), text.size());
  }
}

int main(int argc, char **argv) {
  try {
    std::optional<fs::path> write_masks;
    bool verbose = false;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--write-mask" && i + 1 < argc && !write_masks) write_masks = fs::absolute(argv[++i]);
      else if (argument == "--verbose") verbose = true;
      else positional.push_back(argument);
    }
    if (positional.size() != 2) {
      std::fprintf(stderr, "Usage: ui_detection_replay <game3d_native.hlsl> <cases.json> [--write-mask <new-dir>] [--verbose]\n"
        "cases.json: {\"dump_root\": dir, \"cases\": [{\"dump\", \"label\", \"candidates\": [kinds|\"current\"], "
        "\"paired\": kind, \"exact\": bool, \"trusted\": [candidate indices], \"scene_hold\": bool, \"scene_hold_hudless\": bool, "
        "\"sample\": bool, \"depth_not_current\": bool, \"expect\": {\"mask\", \"source\": [ids], \"mask_exact\": bool, "
        "\"scene\": {\"verdict\", \"d_min\", \"d_max\"}, \"hudless_scene\": {\"verdict\", \"d_min\", \"d_max\"}}}]}\n"
        "Binds each dump's candidates (t0 paired color, t11-t14), raw depth (t1; a 1x1 placeholder without it), presented\n"
        "color (t6) and its exact 80-byte b0, and runs the shader's scene evidence passes as on a sample frame. A case\n"
        "whose dump directory is gone is skipped; the run fails when no case ran.\n"
        "mask_exact compares the resolved mask with the selected raw alpha, all zeros or all ones; scene and hudless_scene\n"
        "check the hidden-scene evidence that the shader's evidence passes write to decision texels 5 and 6. --write-mask\n"
        "copies each dump that consumed an automatic R32 mask into <new-dir>/<NN>_<dump> (NN: the case's position in\n"
        "cases.json) with ui_source_color replaced by the resolved mask, for replay_game3d_dump --shader; any other dump\n"
        "fails its case. --verbose prints every decision word.\n");
      return 2;
    }
    const auto shader_path = fs::absolute(positional[0]);
    const auto shader_source = read_text(shader_path);
    const auto cases_path = fs::absolute(positional[1]);
    const auto document = json::parse(read_text(cases_path));
    fs::path root = document.value("dump_root", cases_path.parent_path().string());
    // A wrong or unmounted root would otherwise skip every case.
    if (!fs::is_directory(root)) throw std::runtime_error("dump_root " + root.string() + " is not a directory");
    if (write_masks && fs::exists(*write_masks)) throw std::runtime_error("--write-mask needs a new directory");
    device_t gpu;
    const D3D_FEATURE_LEVEL level = D3D_FEATURE_LEVEL_11_0;
    checked(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &level, 1, D3D11_SDK_VERSION,
      &gpu.device, nullptr, &gpu.context), "D3D11CreateDevice");
    unsigned passed = 0, failed = 0, skipped = 0, index = 0;
    for (const auto &label : document.at("cases")) {
      ++index;
      const auto dump = root / label.at("dump").get<std::string>();
      const auto name = label.value("label", dump.filename().string());
      // Dumps live outside the repository and may be deleted; that is not a
      // detection failure. A directory without its manifest still fails.
      if (!fs::exists(dump)) {
        ++skipped;
        std::printf("SKIP %-44s dump missing: %s\n", name.c_str(), dump.string().c_str());
        continue;
      }
      outcome result;
      try {
        result = run_case(gpu, shader_source, dump, label);
      } catch (const std::exception &error) {
        ++failed;
        std::printf("FAIL %-44s %s\n", name.c_str(), error.what());
        continue;
      }
      const auto &d = result.decision;
      const auto &expect = label.at("expect");
      // "mask" is one class or a list of acceptable classes.
      std::string wanted_mask;
      bool okay = false;
      for (const auto &wanted : expect.at("mask").is_array() ? expect.at("mask") : json::array({expect.at("mask")})) {
        wanted_mask += (wanted_mask.empty() ? "" : "|") + wanted.get<std::string>();
        okay = okay || mask_class(result) == wanted.get<std::string>();
      }
      if (expect.contains("source")) {
        bool any = false;
        for (const auto &source : expect.at("source")) any = any || source.get<std::uint32_t>() == d[word::source];
        okay = okay && any;
      }
      okay = okay && (result.mask_exact.empty() || result.mask_exact == "match");
      std::string scene, hudless_scene;
      if (expect.contains("scene")) okay = scene_matches(d, expect.at("scene"), false, scene) && okay;
      if (expect.contains("hudless_scene")) okay = scene_matches(d, expect.at("hudless_scene"), true, hudless_scene) && okay;
      // A mask that cannot be written fails its case once.
      std::string written;
      if (write_masks) {
        char prefix[16];
        std::snprintf(prefix, sizeof(prefix), "%02u_", index);
        const auto directory = *write_masks / (prefix + dump.filename().string());
        try {
          write_mask(directory, dump, name, result, shader_path);
          written = "  mask written: " + directory.string();
        } catch (const std::exception &error) {
          okay = false;
          written = std::string("  --write-mask failed: ") + error.what();
        }
      }
      (okay ? passed : failed) += 1;
      std::printf("%s %-44s source=%u covered=%u/%u ui=%.2f%% mask=%s (want %s) candidates=0x%x trusted=0x%x "
        "alpha_covered=%u/%u/%u/%u hudless={changed=%u unchanged=%u invalid=%u tiles=%u lit=%u}%s%s%s%s%s%s\n",
        okay ? "PASS" : "FAIL", name.c_str(), d[word::source], d[word::covered], d[word::pixels],
        100.0 * double(result.ui_pixels) / double(result.pixels), mask_class(result).c_str(), wanted_mask.c_str(),
        d[word::candidates], d[word::trusted], d[word::alpha_covered], d[word::alpha_covered + 1], d[word::alpha_covered + 2],
        d[word::alpha_covered + 3], d[word::hudless_changed], d[word::hudless_unchanged], d[word::hudless_invalid],
        d[word::matching_tiles], d[word::hudless_lit], result.mask_exact.empty() ? "" : " mask_exact=", result.mask_exact.c_str(),
        scene.empty() ? "" : " scene=", scene.c_str(), hudless_scene.empty() ? "" : " hudless_scene=", hudless_scene.c_str());
      if (verbose) {
        std::printf("  words=");
        for (size_t i = 0; i < d.size(); ++i) std::printf("%u%s", d[i], i + 1 < d.size() ? "," : "");
        std::printf(" mask_fnv=%016llx\n", static_cast<unsigned long long>(result.mask_hash));
      }
      if (!written.empty()) std::printf("%s\n", written.c_str());
    }
    // Every case skipped means nothing was checked, never a pass.
    const bool none = passed + failed == 0;
    std::printf("%s UI detection replay: %u passed, %u failed, %u skipped%s\n", failed || none ? "FAIL" : "PASS", passed, failed,
      skipped, none ? " (no case ran)" : "");
    return failed || none ? 1 : 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 2;
  }
}
