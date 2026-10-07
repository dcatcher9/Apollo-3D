// SPDX-License-Identifier: GPL-3.0-only
// Offline check of Game 3D automatic UI detection. Runs the production
// SunshineUIDetection{Tiles,Reduce,Mask}CS passes of a shader file, and its
// SunshineScene{Cells,Compare,Evidence}CS hidden-scene evidence when it has them, on
// the candidate textures saved in Dump 3D packages and compares each decision
// with a labelled expectation. It replaces live trial and error when a detection
// rule changes: every labelled screen of every game is judged at once. With a
// shader of the current candidate layout and selection revision, each decision
// is also checked against ui_selection::decide (mirror=match). A replay is a
// single real frame without a previous decision: nothing is bound at the T1
// hold store (u5), so the reduce reads none and never reuses, and a dump taken
// on a reused frame replays as its own decision. A label pushes the
// hidden-scene guard's per-frame bits (H1) itself, or has the guard
// (game3d_scene_guard.h) derive them from the frame's own measured evidence;
// pre_ui_proven stands for the acceptance ledger's pre-UI proof of the offered
// layer's signature (H1 d). A dump's still_bits and S3 identity fields
// (selection revisions 5 to 8) are ignored.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <nlohmann/json.hpp>

#include "game3d_scene_guard.h"
#include "game3d_ui_detection_contract.h"
#include "game3d_ui_selection.h"
#include "game3d_ui_temporal.h"

#include <algorithm>
#include <array>
#include <cctype>
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
  namespace selection = sunshine_game3d::ui_selection;
  namespace candidate = contract::candidate;
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

  // Candidate kinds by artifact name and their bits in candidate layout 2:
  // t11 UI alpha (R), t12 UI color tag (A), t13 Backbuffer (A), current color
  // alpha (the presented colour's, t6 A), t14 HUD-less, and an offscreen UI layer from the census
  // (ui_layer_candidate_N) at t7 with the renderer's layer flags. Layout 1
  // shaders (no SUNSHINE_UI_CANDIDATE_LAYOUT marker) take the layer in the UI
  // color slot (t12, bit 0x2) and trusted slot indices in b2 word 2.
  const std::map<std::string, unsigned> candidate_bits{{"sl_ui_alpha", candidate::ui_alpha},
    {"sl_ui_color_alpha", candidate::ui_color}, {"sl_backbuffer", candidate::backbuffer}, {"current", candidate::current},
    {"sl_hudless_color", candidate::hudless}};
  bool ui_layer_kind(const std::string &name) { return name.rfind("ui_layer_candidate_", 0) == 0; }
  // A candidate name's bit in the given layout; zero when unknown.
  unsigned candidate_bit(const std::string &name, std::uint32_t layout) {
    if (ui_layer_kind(name)) return layout >= contract::candidate_layout ? candidate::layer : candidate::ui_color;
    const auto found = candidate_bits.find(name);
    return found == candidate_bits.end() ? 0u : found->second;
  }
  // An accepted name's bit in b2 word 2: its candidate bit in layout 2, its
  // trusted alpha slot in layout 1 (where a HUD-less pair had no trust).
  unsigned accepted_bit(const std::string &name, std::uint32_t layout) {
    const auto bit = candidate_bit(name, layout);
    return layout >= contract::candidate_layout ? bit : bit & 15u;
  }
  // A candidate name's kind (candidate layout 2), for its signature.
  std::optional<selection::kind> candidate_kind(const std::string &name) {
    if (ui_layer_kind(name)) return selection::kind::ui_layer;
    if (name == "sl_ui_alpha") return selection::kind::ui_alpha;
    if (name == "sl_ui_color_alpha") return selection::kind::ui_color;
    if (name == "sl_backbuffer") return selection::kind::backbuffer;
    if (name == "current") return selection::kind::current;
    if (name == "sl_hudless_color") return selection::kind::hudless;
    return std::nullopt;
  }

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
  // measures both images and writes decision texels 5 and 6; candidate layout
  // 2 adds the layer's statistics rows 64-79 and decision texel 7,
  // selection revision 2 the one-way judgment rows 80-111 (the scene rows
  // follow from 112, 121 rows in all) and decision texels 8 and 9, and
  // selection revision 4 (12 decision texels) the pre-UI pixel rows 128-143
  // (144 rows in all) and decision texel 11; selection revisions 5 to 8 (13
  // decision texels) added H2's stillness rows from 144 (160 rows in all)
  // and decision texel 12, both removed by revision 9 (12 texels again). A
  // shader with the tile-parts marker counts each tile in
  // that many groups (the dispatch's z) at columns 16 apart.
  // Selection revision 10 (17 decision texels) adds the declared alphas'
  // one-way counts in texel 16 and the layer's pixels beyond the
  // premultiplied bound in rows 208-223 (224 rows in all). The live headers
  // read the current revision only: this replay alone sizes older shaders
  // and pads their decision words with zeros to the current count for the
  // decoders (as their missing texels read before).
  struct sizes_t {
    UINT statistics_rows, decision_texels;
    bool scene;
    UINT tile_parts;
  };
  // A shader without the markers: 5 decision texels and no scene evidence.
  constexpr std::uint32_t default_decision_texels = 5, default_scene_evidence_images = 0;
  // The statistics rows of a shader with these markers.
  constexpr std::uint32_t statistics_rows_of(std::uint32_t images, std::uint32_t texels) {
    return texels >= contract::declared_judgment_decision_texels ? contract::statistics_row_count :
      texels >= contract::pre_ui_decision_texels ? contract::pre_ui_statistics_row + 16u :
      contract::scene_partial_row + (images ? contract::scene::cells_y / 16u : 0u);
  }
  static_assert(statistics_rows_of(0, contract::h1_decision_texels) == 112u && statistics_rows_of(2, contract::h1_decision_texels) == 121u &&
    statistics_rows_of(2, contract::pre_ui_decision_texels) == 144u && statistics_rows_of(2, 13) == 144u &&
    statistics_rows_of(2, contract::decision_texels) == 224u);
  sizes_t detection_sizes(const std::string &source) {
    auto texels = sunshine_game3d::shader_marker(source, contract::decision_texels_marker);
    auto images = sunshine_game3d::shader_marker(source, contract::scene_evidence_images_marker);
    if (!texels) texels = default_decision_texels;
    if (!images) images = default_scene_evidence_images;
    const bool scene = images == contract::max_scene_evidence_images && texels >= contract::scene_decision_texels;
    const auto parts = sunshine_game3d::shader_marker(source, contract::tile_parts_marker);
    if (texels > contract::max_decision_texels || (images && !scene) || parts > contract::max_tile_parts)
      throw std::runtime_error("unsupported detection size markers in the shader");
    return {std::max(80u, statistics_rows_of(images, texels)), std::max(6u, texels), scene, std::max(parts, 1u)};
  }
  // Decision words padded with zeros to the current revision's count, for
  // the live decoders.
  std::vector<std::uint32_t> padded(std::vector<std::uint32_t> words) {
    if (words.size() < 4u * contract::decision_texels) words.resize(4u * contract::decision_texels);
    return words;
  }

  struct outcome {
    std::vector<std::uint32_t> decision;
    std::uint64_t ui_pixels = 0, pixels = 0, mask_hash = 1469598103934665603ull;
    std::string mask_exact; // Empty unless the label asks for it.
    std::vector<float> mask;
    UINT width = 0, height = 0;
    // The candidate layout, and the pushed candidate bits, accepted mask and
    // detection flags (of the second pass when measured).
    std::uint32_t layout = 0, offered = 0, accepted = 0, flags = 0;
    // scene_hold "measured": the per-frame bits the scene guard derived.
    bool measured = false;
    std::uint32_t guard_bits = 0;
  };

  std::string mask_exact(const outcome &result, const std::array<const artifact_t *, 4> &inputs, const artifact_t *layer,
      const artifact_t &paired) {
    // Raw selected alpha, or a whole-frame flat (sources 6 and 8); a
    // HUD-less difference has no CPU reference here.
    const auto source = result.decision.at(word::source);
    std::vector<float> reference;
    if (!source) reference.assign(result.mask.size(), 0.f);
    else if (source == 6u || source == 8u) reference.assign(result.mask.size(), 1.f);
    else if (source <= 4u || source == contract::source_layer) {
      const auto *input = source == 4u ? &paired : source == contract::source_layer ? layer : inputs[source - 1];
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
    // The color HUD-less is compared with (t0): the tagged Backbuffer of a
    // batch pair, else the presented color.
    const auto paired = label.value("paired", std::string("source_color"));
    const auto &paired_color = load(paired);
    // The presented color (t6), whatever HUD-less is paired with; its alpha
    // is the current-color candidate.
    const auto &presented = load("source_color");
    // The game's frame size; source_width/height is the host output, which
    // differs when the host scales the eyes.
    const auto width = artifacts.at(paired).at("width").get<UINT>(), height = artifacts.at(paired).at("height").get<UINT>();
    auto layout = sunshine_game3d::shader_marker(shader_source, contract::candidate_layout_marker);
    if (!layout) layout = contract::legacy_candidate_layout;
    std::array<const artifact_t *, 4> inputs{}; // t11..t14
    const artifact_t *layer_input = nullptr;    // t7 (layout 2)
    unsigned bits = 0, flags = 0;
    // Selection revisions 2-9 read the one-frame-late layer bit 0x4, which
    // every layer copy carried through 180f1842 (now reserved): a shader that
    // still defines it gets it with the layer's flags, so it replays as before.
    const bool late_layer_bit = shader_source.find("#define SUNSHINE_UI_STORED_LATE_LAYER ") != std::string::npos;
    for (const auto &kind : label.at("candidates")) {
      const auto name = kind.get<std::string>();
      const bool layer = ui_layer_kind(name);
      const auto bit = candidate_bit(name, layout);
      if (!bit) throw std::runtime_error("unknown candidate " + name);
      if (bits & bit)
        throw std::runtime_error("two candidates share bit " + std::to_string(bit) + " in candidate layout " + std::to_string(layout));
      bits |= bit;
      if (name == "current") continue;
      if (layer && layout >= contract::candidate_layout) layer_input = &load(name);
      else {
        const unsigned slot = name == "sl_ui_alpha" ? 0 : bit == candidate::ui_color ? 1 : name == "sl_backbuffer" ? 2 : 3;
        inputs[slot] = &load(name);
      }
      if (layer) {
        flags = contract::layer_detection_flags(contract::float_layer_format(artifacts.at(name).at("dxgi_format").get<unsigned>())) |
          (late_layer_bit ? 0x4u : 0u);
      }
    }
    // V2: the threshold from the pair's own encodings, and a HUD-less image
    // that is not comparable with its pair is not offered. Without one, the
    // paired color's own encoding sizes the difference tests, as before.
    const auto encoding_of = [&](const std::string &kind) {
      return selection::encoding{artifacts.at(kind).at("dxgi_format").get<std::uint32_t>(), color};
    };
    std::optional<float> pair_threshold;
    if (bits & candidate::hudless) {
      pair_threshold = selection::comparable(encoding_of("sl_hudless_color"), encoding_of(paired));
      if (!pair_threshold) {
        bits &= ~candidate::hudless;
        inputs[3] = nullptr;
      }
    }
    if (label.value("exact", false) && (bits & candidate::hudless)) bits |= candidate::exact;
    unsigned accepted = 0;
    for (const auto &kind : label.value("accepted", json::array())) {
      const auto name = kind.get<std::string>();
      if (!candidate_bit(name, layout)) throw std::runtime_error("unknown accepted kind " + name);
      accepted |= accepted_bit(name, layout);
    }
    // The acceptance signature of every offered kind (A1): its artifact's
    // typed format (the current alpha's is the paired color's) and the
    // manifest color space, as the scene guard keys refutations.
    sunshine_game3d::scene_guard::kind_signatures signatures{};
    for (const auto &kind : label.at("candidates")) {
      const auto name = kind.get<std::string>();
      if (const auto k = candidate_kind(name))
        signatures[std::size_t(*k)] = {*k, artifacts.at(name == "current" ? paired : name).at("dxgi_format").get<std::uint32_t>(),
          color};
    }
    // Per-frame bits a label names; they never select a candidate. scene_hold
    // true: the CPU holds a hidden verdict (H1); "measured": the scene guard
    // derives the bits from this frame's own evidence. pre_ui_visible: the
    // held samples read the pre-UI image visible. refuted: candidate kinds
    // whose signatures a visible verdict refuted. pre_ui_proven: the
    // acceptance ledger proved the offered layer's signature the pre-UI scene
    // image (per_frame_pre_ui_proven, pushed on every pass as the renderer
    // does, and the guard's layer_proven when measured).
    bool measured = false;
    const bool pre_ui_proven = label.value("pre_ui_proven", false);
    if (pre_ui_proven && !(bits & contract::candidate::layer))
      throw std::runtime_error("pre_ui_proven needs an offered layer");
    if (pre_ui_proven) flags |= contract::per_frame_pre_ui_proven;
    if (label.contains("scene_hold")) {
      const auto &hold = label.at("scene_hold");
      if (hold.is_string() && hold.get<std::string>() == "measured") measured = true;
      else if (hold.is_boolean()) flags |= hold.get<bool>() ? contract::per_frame_scene_hidden : 0u;
      else throw std::runtime_error("scene_hold must be true, false or \"measured\"");
    }
    if (label.value("pre_ui_visible", false)) flags |= contract::per_frame_pre_ui_visible;
    for (const auto &kind : label.value("refuted", json::array())) {
      const auto name = kind.get<std::string>();
      const auto bit = candidate_bit(name, layout);
      if (!bit || layout < contract::candidate_layout) throw std::runtime_error("refuted names an unknown candidate kind " + name);
      flags |= bit << contract::per_frame_refuted_shift;
    }
    // A "sample" key is accepted and ignored: its flag 0x20000 is reserved and
    // the shader never read it. Every replay is a status sample, so it pushes
    // per_frame_sample (selection revision 10: the one-way judgment and the
    // pre-UI pixels are counted on samples only; older shaders read that bit
    // as an unused stored bit).
    flags |= contract::per_frame_sample;
    if (label.value("depth_not_current", false)) flags |= contract::per_frame_depth_not_current;

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
    if (measured && !sizes.scene) throw std::runtime_error("scene_hold \"measured\" needs a shader with scene evidence");
    shaders_t shaders{compile(gpu, shader_source, "SunshineUIDetectionTilesCS", width, height, color),
      compile(gpu, shader_source, "SunshineUIDetectionReduceCS", width, height, color),
      compile(gpu, shader_source, "SunshineUIDetectionMaskCS", width, height, color)};
    if (sizes.scene) {
      shaders.cells = compile(gpu, shader_source, "SunshineSceneCellsCS", width, height, color);
      shaders.compare = compile(gpu, shader_source, "SunshineSceneCompareCS", width, height, color);
      shaders.evidence = compile(gpu, shader_source, "SunshineSceneEvidenceCS", width, height, color);
    }
    auto statistics = target(gpu, contract::statistics_columns(sizes.tile_parts), sizes.statistics_rows,
      DXGI_FORMAT_R32G32B32A32_UINT);
    texture_t cells;
    if (sizes.scene) cells = target(gpu, contract::scene::cells_x, contract::scene::cells_y, DXGI_FORMAT_R32G32B32A32_UINT);
    auto decision = target(gpu, sizes.decision_texels, 1, DXGI_FORMAT_R32G32B32A32_UINT);
    auto mask = target(gpu, width, height, DXGI_FORMAT_R32_FLOAT);
    // Detection constants b2: candidate bits, difference threshold, accepted
    // candidates (layout 1: trusted slots), flags and (selection revision 4)
    // the offscreen UI layer's pair threshold with the presented color, from
    // the two artifacts' own encodings (zero without a layer or when not
    // comparable), padded with zeros to the 16-byte constant buffer
    // granularity: shaders of selection revisions 5 to 9 read their word 5
    // there as zero, as the renderer pushed it.
    const float threshold = pair_threshold ? *pair_threshold :
      selection::comparable(encoding_of(paired), encoding_of(paired)).value_or(2.f / 255.f);
    float pre_ui_threshold = 0.f;
    for (const auto &kind : label.at("candidates"))
      if (ui_layer_kind(kind.get<std::string>()) && layer_input)
        pre_ui_threshold =
          selection::comparable(encoding_of(kind.get<std::string>()), encoding_of("source_color")).value_or(0.f);
    struct {
      std::uint32_t bits;
      float threshold;
      std::uint32_t accepted, flags;
      float pre_ui_threshold;
      std::uint32_t padding[3];
    } constants{bits, threshold, accepted, flags, pre_ui_threshold, {}};
    static_assert(sizeof(constants) == 32 && contract::b2_words * 4 <= sizeof(constants));
    const auto constant_buffer = [&](const void *bytes, UINT size) {
      D3D11_BUFFER_DESC buffer{};
      buffer.ByteWidth = size; buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      const D3D11_SUBRESOURCE_DATA initial{bytes, 0, 0};
      ComPtr<ID3D11Buffer> result;
      checked(gpu.device->CreateBuffer(&buffer, &initial, &result), "constants");
      return result;
    };
    const auto geometry = constant_buffer(parameters.data(), UINT(parameters.size()));
    auto cb = constant_buffer(&constants, sizeof(constants));

    auto &context = *gpu.context.Get();
    const auto stage = [&](ID3D11ComputeShader *shader, ID3D11ShaderResourceView *t10, UINT uav_slot, ID3D11UnorderedAccessView *uav, UINT x, UINT y,
        UINT z = 1) {
      std::array<ID3D11ShaderResourceView *, 15> views{};
      views[0] = paired_color.gpu.srv.Get();
      views[1] = depth.srv.Get();
      views[6] = presented.gpu.srv.Get();
      views[7] = layer_input ? layer_input->gpu.srv.Get() : nullptr;
      views[10] = t10;
      for (unsigned i = 0; i < 4; ++i) views[11 + i] = inputs[i] ? inputs[i]->gpu.srv.Get() : nullptr;
      context.CSSetShader(shader, nullptr, 0);
      context.CSSetShaderResources(0, UINT(views.size()), views.data());
      ID3D11Buffer *buffers[3]{geometry.Get(), nullptr, cb.Get()};
      context.CSSetConstantBuffers(0, 3, buffers);
      context.CSSetUnorderedAccessViews(uav_slot, 1, &uav, nullptr);
      context.Dispatch(x, y, z);
      ID3D11UnorderedAccessView *none = nullptr;
      context.CSSetUnorderedAccessViews(uav_slot, 1, &none, nullptr);
      std::array<ID3D11ShaderResourceView *, 15> cleared{};
      context.CSSetShaderResources(0, UINT(cleared.size()), cleared.data());
    };
    stage(shaders.tiles.Get(), nullptr, 6, statistics.uav.Get(), 16, 16, sizes.tile_parts);
    stage(shaders.reduce.Get(), statistics.srv.Get(), 6, decision.uav.Get(), 1, 1);
    stage(shaders.mask.Get(), decision.srv.Get(), 0, mask.uav.Get(), (width + 7) / 8, (height + 7) / 8);
    // The hidden-scene evidence of a sample frame: measured after the
    // decision, it writes decision texels 5 and 6 only.
    if (sizes.scene) {
      stage(shaders.cells.Get(), nullptr, 6, cells.uav.Get(), contract::scene::cells_x / 16, contract::scene::cells_y);
      stage(shaders.compare.Get(), cells.srv.Get(), 6, statistics.uav.Get(), contract::scene::cells_x / 16, contract::scene::cells_y / 16);
      stage(shaders.evidence.Get(), statistics.srv.Get(), 6, decision.uav.Get(), 1, 1);
    }
    // scene_hold "measured": the scene guard observes this frame's sample as
    // the renderer would three consecutive samples of a static screen: at
    // tick 900 (not actionable unless a proven layer is the offer's pre-UI
    // image: otherwise the sample that shows a claim first opens the gate),
    // then 1000 and 1100, each actionable as measure() says. The bits it
    // pushes at 1100 drive a second reduce, mask and evidence pass over the
    // same statistics. It proves the single-sample relation only; the
    // sequence replay owns temporal behaviour.
    std::uint32_t guard_bits = 0;
    if (measured) {
      const auto words = padded(download<std::uint32_t>(gpu, decision));
      sunshine_game3d::scene_guard::state guard;
      guard.enter_scope(1, 0);
      const bool proven_image = pre_ui_proven && selection::pre_ui_image_of(bits) == contract::pre_ui_image::layer;
      for (const std::uint64_t tick : {900u, 1000u, 1100u})
        guard.observe(sunshine_game3d::ui_temporal::guard_sample(
          sunshine_game3d::ui_temporal::decode_detection_sample(words.data(), words.size(), tick, 1)),
          guard.measure(tick, proven_image), signatures);
      guard_bits = guard.per_frame(1100, bits, signatures, pre_ui_proven);
      flags |= guard_bits;
      constants.flags = flags;
      cb = constant_buffer(&constants, sizeof(constants));
      stage(shaders.reduce.Get(), statistics.srv.Get(), 6, decision.uav.Get(), 1, 1);
      stage(shaders.mask.Get(), decision.srv.Get(), 0, mask.uav.Get(), (width + 7) / 8, (height + 7) / 8);
      stage(shaders.evidence.Get(), statistics.srv.Get(), 6, decision.uav.Get(), 1, 1);
    }

    outcome result;
    result.measured = measured;
    result.guard_bits = guard_bits;
    result.decision = download<std::uint32_t>(gpu, decision);
    result.mask = download<float>(gpu, mask);
    result.width = width; result.height = height;
    result.layout = layout; result.offered = bits; result.accepted = accepted; result.flags = flags;
    result.pixels = result.mask.size();
    for (const float value : result.mask) {
      result.ui_pixels += value > 0.f ? 1u : 0u;
      std::uint32_t word;
      std::memcpy(&word, &value, sizeof(word));
      result.mask_hash = (result.mask_hash ^ word) * 1099511628211ull;
    }
    const bool exact_today = label.contains("xfail") && label.at("xfail").is_object() && label.at("xfail").contains("today") &&
      label.at("xfail").at("today").is_object() && label.at("xfail").at("today").value("mask_exact", false);
    if (label.at("expect").value("mask_exact", false) || exact_today)
      result.mask_exact = mask_exact(result, inputs, layer_input, paired_color);
    return result;
  }

  std::string mask_class(const outcome &o) {
    return o.ui_pixels == 0 ? "empty" : o.ui_pixels == o.pixels ? "flat" : "partial";
  }

  // expect.scene: the presented image's hidden-scene evidence in decision
  // texel 5; expect.pre_ui_scene: the pre-UI scene image's in texel 6, with
  // "image" ("hudless" or "layer") the image it measured. "verdict" is one
  // name or a list (the pre-UI texel's verdict is read from its valid D with
  // the same bounds, else none); "d_min" and "d_max" bound D inclusively.
  // Evidence that did not run, as from a shader without scene evidence, fails
  // the check.
  constexpr std::array<const char *, 3> pre_ui_image_names{"none", "hudless", "layer"};
  bool scene_matches(const std::vector<std::uint32_t> &d, const json &expected, bool pre_ui, std::string &text) {
    if (d.size() <= word::pre_ui_scene_image) {
      text = "no-reference(" + std::to_string(d.size() / 4) + " decision texels)";
      return false;
    }
    const auto n = d[pre_ui ? word::pre_ui_scene_n : word::scene_n], state = d[pre_ui ? word::pre_ui_scene_state : word::scene_state];
    float value;
    std::memcpy(&value, &d[pre_ui ? word::pre_ui_scene_d : word::scene_d], 4);
    const bool valid = contract::scene_state_valid(state), ran = contract::scene_state_ran(state);
    const auto *verdict = pre_ui ? (valid ? contract::name(contract::scene_verdict_of(value)) : "none") :
      contract::name(contract::scene_state_verdict(state));
    const auto image = d[word::pre_ui_scene_image];
    const char *image_name = image < pre_ui_image_names.size() ? pre_ui_image_names[image] : "other";
    bool okay = ran;
    if (pre_ui && expected.contains("image")) okay = okay && expected.at("image").get<std::string>() == image_name;
    if (expected.contains("verdict")) {
      bool any = false;
      for (const auto &wanted : expected.at("verdict").is_array() ? expected.at("verdict") : json::array({expected.at("verdict")}))
        any = any || wanted.get<std::string>() == verdict;
      okay = okay && any;
    }
    if (expected.contains("d_min")) okay = okay && value >= expected.at("d_min").get<float>();
    if (expected.contains("d_max")) okay = okay && value <= expected.at("d_max").get<float>();
    char buffer[160];
    std::snprintf(buffer, sizeof(buffer), "%s(%s%s%s%s n=%u d=%.4f valid=%u ran=%u)", okay ? "match" : "differs",
      pre_ui ? "image " : "", pre_ui ? image_name : "", pre_ui ? " " : "", verdict, n, double(value), unsigned(valid),
      unsigned(ran));
    text = buffer;
    return okay;
  }

  // expect.pre_ui_match: whether this sample would count toward the
  // acceptance ledger's pre-UI proof of the offered layer (H1 d): the layer
  // without coverage and ui_selection::pre_ui_match on decision texel 11.
  // A shader without texel 11 fails the check.
  bool pre_ui_match_matches(const std::vector<std::uint32_t> &d, bool expected, std::string &text) {
    if (d.size() < 4 * contract::pre_ui_decision_texels) {
      text = "no-reference(" + std::to_string(d.size() / 4) + " decision texels)";
      return false;
    }
    const bool match = !d[word::layer_covered] &&
      selection::pre_ui_match(d[word::pre_ui_match], d[word::pre_ui_image_lit], d[word::pixels]);
    char buffer[96];
    std::snprintf(buffer, sizeof(buffer), "%s(%s)", match == expected ? "match" : "differs", match ? "true" : "false");
    text = buffer;
    return match == expected;
  }

  // One expectation: a case's "expect", or the "today" outcome of its xfail.
  struct judged_t {
    bool okay = false;
    std::string wanted_mask, scene, pre_ui_scene, pre_ui_match;
  };
  judged_t judge(const outcome &result, const json &expect) {
    const auto &d = result.decision;
    judged_t judged;
    // "mask" is one class or a list of acceptable classes.
    for (const auto &wanted : expect.at("mask").is_array() ? expect.at("mask") : json::array({expect.at("mask")})) {
      judged.wanted_mask += (judged.wanted_mask.empty() ? "" : "|") + wanted.get<std::string>();
      judged.okay = judged.okay || mask_class(result) == wanted.get<std::string>();
    }
    if (expect.contains("source")) {
      bool any = false;
      for (const auto &source : expect.at("source")) any = any || source.get<std::uint32_t>() == d[word::source];
      judged.okay = judged.okay && any;
    }
    if (expect.value("mask_exact", false)) judged.okay = judged.okay && result.mask_exact == "match";
    if (expect.contains("scene")) judged.okay = scene_matches(d, expect.at("scene"), false, judged.scene) && judged.okay;
    if (expect.contains("pre_ui_scene"))
      judged.okay = scene_matches(d, expect.at("pre_ui_scene"), true, judged.pre_ui_scene) && judged.okay;
    if (expect.contains("pre_ui_match"))
      judged.okay = pre_ui_match_matches(d, expect.at("pre_ui_match").get<bool>(), judged.pre_ui_match) && judged.okay;
    return judged;
  }

  // A known-wrong cell: "expect" holds the target outcome and "xfail" the
  // roadmap stage that reaches it, the rule it applies and today's outcome.
  // Returns the schema error, empty when the case is well formed.
  std::string xfail_error(const json &label) {
    if (label.contains("needs_dump") && (!label.at("needs_dump").is_string() || label.at("needs_dump").get<std::string>().empty()))
      return "needs_dump must name what the dump is missing";
    // Acceptance is named by candidate kind (UI framework S1); slot indices
    // meant different sources in different layouts.
    if (label.contains("trusted")) return "trusted is retired: use accepted with candidate kind names";
    // The hidden-scene routes merged into H1 (UI framework S2b).
    if (label.contains("scene_hold_hudless"))
      return "scene_hold_hudless is retired: use scene_hold with pre_ui_visible (H1)";
    const auto retired_scene = [](const json &expect) { return expect.is_object() && expect.contains("hudless_scene"); };
    if (retired_scene(label.at("expect")) ||
        (label.contains("xfail") && label.at("xfail").is_object() && label.at("xfail").contains("today") &&
         retired_scene(label.at("xfail").at("today"))))
      return "hudless_scene is retired: use pre_ui_scene with image \"hudless\" (H1)";
    if (label.contains("refuted")) {
      if (!label.at("refuted").is_array()) return "refuted must be a list of candidate kind names";
      for (const auto &kind : label.at("refuted"))
        if (!kind.is_string() || !candidate_bit(kind.get<std::string>(), contract::candidate_layout))
          return "refuted names an unknown candidate kind";
    }
    if (label.contains("accepted")) {
      if (!label.at("accepted").is_array()) return "accepted must be a list of candidate kind names";
      for (const auto &kind : label.at("accepted"))
        if (!kind.is_string() || !candidate_bit(kind.get<std::string>(), contract::candidate_layout))
          return "accepted names an unknown candidate kind";
    }
    if (!label.contains("xfail")) return {};
    const auto &xfail = label.at("xfail");
    if (!xfail.is_object()) return "xfail must be an object";
    static const std::array<const char *, 6> stages{"S1", "S2a", "S2b", "S4", "S5", "S6"};
    const auto stage = xfail.value("stage", std::string{});
    // Roadmap stage S3 (the frame identity) was removed.
    if (stage == "S3") return "xfail.stage S3 was removed: name the stage that now owns the cell";
    if (std::none_of(stages.begin(), stages.end(), [&](const char *name) { return stage == name; }))
      return "xfail.stage must be one of S1, S2a, S2b, S4, S5, S6";
    const auto reason = xfail.contains("reason") && xfail.at("reason").is_string() ? xfail.at("reason").get<std::string>() : "";
    // The reason names a framework rule the stage applies, as a standalone ID
    // (stages and rules: docs/reshade-sbs.md, UI decision framework).
    static const std::array<const char *, 13> rules{"E1", "E2", "V1", "V2", "A1", "A2", "A3", "S1", "S2", "H1", "P1", "T1", "F1"};
    bool rule = false;
    for (size_t i = 0; i < reason.size() && !rule; ++i) {
      if (i && std::isalnum(static_cast<unsigned char>(reason[i - 1]))) continue;
      size_t end = i;
      while (end < reason.size() && std::isalnum(static_cast<unsigned char>(reason[end]))) ++end;
      const auto word = reason.substr(i, end - i);
      rule = std::any_of(rules.begin(), rules.end(), [&](const char *id) { return word == id; });
    }
    if (!rule) return "xfail.reason must name a rule (E1, E2, V1, V2, A1-A3, S1, S2, H1, P1, T1 or F1) that fixes the cell";
    if (!xfail.contains("today") || !xfail.at("today").is_object()) return "xfail.today must be an object";
    const auto &today = xfail.at("today");
    if (!today.contains("mask") || !today.contains("source") || !today.at("source").is_array() || today.at("source").empty())
      return "xfail.today needs a mask and a non-empty source list";
    return {};
  }

  // ui_selection::decide on the counts the GPU wrote, with the pushed
  // candidate bits, accepted mask and flags and no previous decision (the
  // hold store unbound): "match" when the GPU's
  // source, coverage, accepted
  // word, valid bits, refused candidate, frame reason, claims and h1 word
  // agree, "n/a" for a shader of another candidate layout or selection
  // revision.
  std::string mirror_of(const outcome &result, bool mirrored) {
    const auto &d = result.decision;
    if (!mirrored || d.size() <= word::h1) return "n/a";
    const auto expected = selection::decide(selection::counts_from_words(d.data(), d.size()), result.offered, result.accepted,
      result.flags, selection::hold_state{});
    if (expected.source == d[word::source] && expected.covered == d[word::covered] && expected.valid_bits == d[word::valid_bits] &&
        d[word::accepted] == result.accepted && d[word::candidates] == result.offered && expected.refused == d[word::refused] &&
        selection::frame_reason_word(expected) == d[word::frame_reason] && expected.claims == d[word::claims] &&
        selection::h1_word(expected) == d[word::h1])
      return "match";
    char text[240];
    std::snprintf(text, sizeof(text),
      "differs(decide source=%u covered=%u valid=0x%x refused=0x%x frame_reason=0x%x claims=0x%x h1=0x%x)", expected.source,
      expected.covered, expected.valid_bits, expected.refused, selection::frame_reason_word(expected), expected.claims,
      selection::h1_word(expected));
    return text;
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
    bool verbose = false, strict = false;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--write-mask" && i + 1 < argc && !write_masks) write_masks = fs::absolute(argv[++i]);
      else if (argument == "--verbose") verbose = true;
      else if (argument == "--strict") strict = true;
      else positional.push_back(argument);
    }
    if (positional.size() != 2) {
      std::fprintf(stderr, "Usage: ui_detection_replay <game3d_native.hlsl> <cases.json> [--write-mask <new-dir>] [--verbose] "
        "[--strict]\n"
        "cases.json: {\"dump_root\": dir, \"cases\": [{\"dump\", \"label\", \"candidates\": [kinds|\"current\"], "
        "\"paired\": kind, \"exact\": bool, \"accepted\": [kinds], \"scene_hold\": bool|\"measured\", "
        "\"pre_ui_visible\": bool, \"pre_ui_proven\": bool, \"refuted\": [kinds], \"depth_not_current\": bool, "
        "\"expect\": {\"mask\", "
        "\"source\": [ids], \"mask_exact\": bool, \"scene\": {\"verdict\", \"d_min\", \"d_max\"}, "
        "\"pre_ui_scene\": {\"image\", \"verdict\", \"d_min\", \"d_max\"}, \"pre_ui_match\": bool}, "
        "\"xfail\": {\"stage\", \"reason\", \"today\": {expect fields}}, \"needs_dump\": text}]}\n"
        "Binds each dump's candidates (t0 paired color, t11-t14, and a census ui_layer_candidate_N at t7 with the layer\n"
        "flags; a shader without SUNSHINE_UI_CANDIDATE_LAYOUT takes the layer at t12), raw depth (t1; a 1x1 placeholder\n"
        "without it), presented color (t6) and its exact 80-byte b0, and runs the shader's scene evidence passes as on a\n"
        "sample frame. accepted names the candidate kinds the game session accepts (A1); the retired trusted key fails\n"
        "its case. The hidden-scene guard's per-frame bits (H1): scene_hold true pushes its held hidden verdict,\n"
        "pre_ui_visible its held visible pre-UI image, refuted the named kinds' refuted signatures; scene_hold \"measured\"\n"
        "has game3d_scene_guard.h observe this frame's own sample at ticks 900, 1000 and 1100 (signatures from the\n"
        "artifact formats and the manifest color_space) and reruns the reduce, mask and evidence passes with the bits it\n"
        "pushes. pre_ui_proven stands for the acceptance ledger's pre-UI proof of the offered layer's signature (H1 d):\n"
        "it pushes per_frame_pre_ui_proven with any scene_hold form and, when measured, is the guard's layer_proven.\n"
        "The retired scene_hold_hudless and hudless_scene keys fail their case.\n"
        "A HUD-less image not comparable with its pair (ui_selection::comparable, from the two artifact formats\n"
        "and the manifest color_space) is not offered, and the pair's threshold comes from the same function. With a\n"
        "shader of the current layout and selection revision, every decision is checked against ui_selection::decide on\n"
        "the GPU's counts with no previous decision, as nothing is bound at the T1 hold store (mirror=match); a\n"
        "mirror that differs fails its case. A case\n"
        "whose dump directory is gone, or whose needs_dump names what its dump lacks, is skipped; the run fails when no\n"
        "case ran. A \"sample\" key is accepted and ignored: its flag 0x20000 is reserved.\n"
        "A case with xfail is a known-wrong cell: expect holds the target outcome, xfail.stage the roadmap stage (S1, S2a,\n"
        "S2b, S4-S6; the removed S3 fails the case) that reaches it, xfail.reason the rule it applies (E1, E2, V1, V2,\n"
        "A1-A3, S1, S2, H1, P1, T1, F1) and xfail.today the outcome it has now\n"
        "(mask and source required); docs/reshade-sbs.md (UI decision framework) defines the stages and rules.\n"
        "Meeting the target is XPASS (remove the xfail), else meeting today is XFAIL, else\n"
        "FAIL; a malformed xfail fails. The run fails on any FAIL, and with --strict also on any XPASS.\n"
        "mask_exact compares the resolved mask with the selected raw alpha, all zeros or all ones (sources 6 and 8); scene\n"
        "and pre_ui_scene check the hidden-scene evidence that the shader's evidence passes write to decision texels 5 and\n"
        "6 (the pre-UI image: HUD-less when offered, else the UI layer); pre_ui_match checks whether the sample would\n"
        "count toward the layer's pre-UI proof (a layer without coverage and ui_selection::pre_ui_match on texel 11: its\n"
        "colour within 8 times the layer/presented pair threshold on 90% of pixels, lit on half). --write-mask\n"
        "copies each dump that consumed an automatic R32 mask into <new-dir>/<NN>_<dump> (NN: the case's position in\n"
        "cases.json) with ui_source_color replaced by the resolved mask, for replay_game3d_dump --shader; any other dump\n"
        "fails its case. Each line shows the frame reason, the refused candidate and the one-way judgment counts\n"
        "(strong/contradicted pixels of the layer, Backbuffer and current alpha) from selection revision 2, from\n"
        "revision 10 (every case is a status sample: per_frame_sample) the layer's as reserved zeros and those of\n"
        "UIAlpha and the UI color tag (declared_one_way), and the H1\n"
        "claims and h1 word (applied, S1 winner) from selection revision 3, and the layer's pre-UI pixel counts\n"
        "(texel 11: match and image_lit) from selection revision 4.\n"
        "--verbose prints every decision word.\n");
      return 2;
    }
    const auto shader_path = fs::absolute(positional[0]);
    const auto shader_source = read_text(shader_path);
    // Shaders of the current candidate layout and selection revision are
    // checked against ui_selection::decide.
    const bool mirrored =
      sunshine_game3d::shader_marker(shader_source, contract::candidate_layout_marker) == contract::candidate_layout &&
      sunshine_game3d::shader_marker(shader_source, selection::revision_marker) == selection::revision;
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
    unsigned passed = 0, xfailed = 0, xpassed = 0, failed = 0, skipped = 0, index = 0;
    for (const auto &label : document.at("cases")) {
      ++index;
      const auto dump = root / label.at("dump").get<std::string>();
      const auto name = label.value("label", dump.filename().string());
      if (const auto error = xfail_error(label); !error.empty()) {
        ++failed;
        std::printf("FAIL %-44s %s\n", name.c_str(), error.c_str());
        continue;
      }
      // A dump known to lack an artifact the case needs (a re-dump is due).
      if (label.contains("needs_dump")) {
        ++skipped;
        std::printf("SKIP %-44s needs a dump with %s\n", name.c_str(), label.at("needs_dump").get<std::string>().c_str());
        continue;
      }
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
      const auto target = judge(result, label.at("expect"));
      const json *xfail = label.contains("xfail") ? &label.at("xfail") : nullptr;
      // A known-wrong cell is XPASS at its target, XFAIL at today's outcome
      // and FAIL at anything else: an unexpected change is never green.
      const auto today = xfail ? judge(result, xfail->at("today")) : judged_t{};
      const char *status = target.okay ? (xfail ? "XPASS" : "PASS") : xfail && today.okay ? "XFAIL" : "FAIL";
      bool okay = target.okay || (xfail && today.okay);
      // The GPU decision against ui_selection::decide on the GPU's own counts.
      const auto mirror = mirror_of(result, mirrored);
      if (mirror.rfind("differs", 0) == 0) {
        okay = false;
        status = "FAIL";
      }
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
          status = "FAIL";
          written = std::string("  --write-mask failed: ") + error.what();
        }
      }
      if (!okay) ++failed;
      else if (!xfail) ++passed;
      else (target.okay ? xpassed : xfailed) += 1;
      // The scene evidence shown is the target's check, else today's.
      const auto &scene = target.scene.empty() ? today.scene : target.scene;
      const auto &pre_ui_scene = target.pre_ui_scene.empty() ? today.pre_ui_scene : target.pre_ui_scene;
      const auto &pre_ui_match = target.pre_ui_match.empty() ? today.pre_ui_match : target.pre_ui_match;
      auto wanted = target.wanted_mask;
      if (xfail) {
        wanted += ", today " + today.wanted_mask + "; " + xfail->at("stage").get<std::string>() + ": " +
          xfail->at("reason").get<std::string>() + (target.okay ? "; remove xfail" : "");
      }
      // The layer's own counts exist from candidate layout 2 (texel 7).
      char layer[96] = "";
      if (d.size() > word::valid_bits)
        std::snprintf(layer, sizeof(layer), " layer={covered=%u invalid=%u opaque=%u} valid=0x%x", d[word::layer_covered],
          d[word::layer_invalid], d[word::layer_opaque], d[word::valid_bits]);
      // The frame reason, refused candidate and one-way counts from
      // selection revision 2 (texels 8 and 9: the layer's, a reserved zero
      // since revision 10, Backbuffer and current alpha), and from revision
      // 10 those of UIAlpha and the UI color tag (texel 16).
      char judgment[280] = "";
      if (d.size() > word::frame_reason) {
        const auto reason = selection::frame_reason_name(d[word::frame_reason]);
        const auto refused = selection::candidate_name(d[word::refused]);
        char declared[96] = "";
        if (d.size() > word::contradicted_ui_color)
          std::snprintf(declared, sizeof(declared), " declared_one_way={strong=%u/%u contradicted=%u/%u}", d[word::strong_ui_alpha],
            d[word::strong_ui_color], d[word::contradicted_ui_alpha], d[word::contradicted_ui_color]);
        std::snprintf(judgment, sizeof(judgment), " reason=%.*s%s refused=%.*s one_way={strong=%u/%u/%u contradicted=%u/%u/%u}%s",
          int(reason.size()), reason.data(), (d[word::frame_reason] & contract::frame_reason_reused) ? "(reused)" : "",
          int(refused.size()), refused.data(), d[word::strong], d[word::strong + 1], d[word::strong + 2], d[word::contradicted],
          d[word::contradicted + 1], d[word::contradicted + 2], declared);
      }
      // H1 from selection revision 3 (texel 10): the raw claims, whether H1
      // applied and the S1 winner; a measured case also shows the bits the
      // scene guard pushed.
      char h1[128] = "";
      if (d.size() > word::h1)
        std::snprintf(h1, sizeof(h1), " claims=0x%x h1={applied=%d,winner=%u}", d[word::claims],
          (d[word::h1] & contract::h1_applied) ? 1 : 0, d[word::h1] & contract::h1_winner_mask);
      char guard[64] = "";
      if (result.measured) std::snprintf(guard, sizeof(guard), " measured(per_frame=0x%x)", result.guard_bits);
      // H1 (d) from selection revision 4 (texel 11): the layer against the
      // presented frame.
      char pre_ui_pixels[128] = "";
      if (d.size() >= 4 * contract::pre_ui_decision_texels)
        std::snprintf(pre_ui_pixels, sizeof(pre_ui_pixels), " pre_ui_pixels={match=%u image_lit=%u}", d[word::pre_ui_match],
          d[word::pre_ui_image_lit]);
      std::printf("%s %-44s source=%u covered=%u/%u ui=%.2f%% mask=%s (want %s) candidates=0x%x accepted=0x%x "
        "alpha_covered=%u/%u/%u/%u%s%s%s%s%s hudless={changed=%u unchanged=%u invalid=%u tiles=%u lit=%u} "
        "mirror=%s%s%s%s%s%s%s%s%s\n",
        status, name.c_str(), d[word::source], d[word::covered], d[word::pixels],
        100.0 * double(result.ui_pixels) / double(result.pixels), mask_class(result).c_str(), wanted.c_str(),
        d[word::candidates], d[word::accepted], d[word::alpha_covered], d[word::alpha_covered + 1], d[word::alpha_covered + 2],
        d[word::alpha_covered + 3], layer, judgment, h1, guard, pre_ui_pixels, d[word::hudless_changed],
        d[word::hudless_unchanged], d[word::hudless_invalid], d[word::matching_tiles], d[word::hudless_lit], mirror.c_str(),
        result.mask_exact.empty() ? "" : " mask_exact=", result.mask_exact.c_str(), scene.empty() ? "" : " scene=", scene.c_str(),
        pre_ui_scene.empty() ? "" : " pre_ui_scene=", pre_ui_scene.c_str(), pre_ui_match.empty() ? "" : " pre_ui_match=",
        pre_ui_match.c_str());
      if (verbose) {
        std::printf("  words=");
        for (size_t i = 0; i < d.size(); ++i) std::printf("%u%s", d[i], i + 1 < d.size() ? "," : "");
        std::printf(" mask_fnv=%016llx\n", static_cast<unsigned long long>(result.mask_hash));
      }
      if (!written.empty()) std::printf("%s\n", written.c_str());
    }
    // Every case skipped means nothing was checked, never a pass.
    const bool none = passed + xfailed + xpassed + failed == 0, bad = failed || none || (strict && xpassed);
    std::printf("%s UI detection replay: %u passed, %u xfailed, %u xpassed, %u failed, %u skipped%s%s\n", bad ? "FAIL" : "PASS",
      passed, xfailed, xpassed, failed, skipped, none ? " (no case ran)" : "",
      strict && xpassed ? " (--strict: remove the xfail of each XPASS)" : "");
    return bad ? 1 : 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 2;
  }
}
