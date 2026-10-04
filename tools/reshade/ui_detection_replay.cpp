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
// layer's signature (H1 d). H2's still-screen flag (b2 word 5) is the dump's
// own (replay.ui_detection.rules_bits, else still_bits, zero when absent); a
// single frame has no previous cell means, so its stillness counts read
// nothing compared. Fix 3: a label's refine pushes the refine rule bit, and
// the layer pairs with a retained Present the dump carries (layer_pair) or,
// absent one, is late; the pre-UI change set is offered as the renderer
// offers it (change_set::offered), and every line shows the change-set
// shadow that change_set::measure_shadow reads from texels 13-15. Fix 4
// (pin only UI, rule P2): a shader with the darkening planes runs its
// darkening passes after the reduce as the renderer does on a sample frame
// (b2 word 5 always has rules::darkening_measured, and rules::tag_linear
// with a float UI color tag; refine, or its alias pin_only_ui, adds
// rules::pin_only_ui, so the mask pass applies it), and every line of an
// eligible source shows words 62-63 against game3d_ui_darkening.h's CPU
// reference on the same inputs.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <nlohmann/json.hpp>

#include "game3d_scene_guard.h"
#include "game3d_ui_change_set.h"
#include "game3d_ui_darkening.h"
#include "game3d_ui_detection_contract.h"
#include "game3d_ui_selection.h"

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
  namespace ui_darkening = sunshine_game3d::ui_darkening;
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
  // alpha (t0 A), t14 HUD-less, and an offscreen UI layer from the census
  // (ui_layer_candidate_N) or the copy detection consumed (ui_layer_detected,
  // fix 3) at t7 with the renderer's layer flags. Layout 1
  // shaders (no SUNSHINE_UI_CANDIDATE_LAYOUT marker) take the layer in the UI
  // color slot (t12, bit 0x2) and trusted slot indices in b2 word 2.
  const std::map<std::string, unsigned> candidate_bits{{"sl_ui_alpha", candidate::ui_alpha},
    {"sl_ui_color_alpha", candidate::ui_color}, {"sl_backbuffer", candidate::backbuffer}, {"current", candidate::current},
    {"sl_hudless_color", candidate::hudless}};
  bool ui_layer_kind(const std::string &name) {
    return name.rfind("ui_layer_candidate_", 0) == 0 || name == "ui_layer_detected";
  }
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
    const size_t per_row =
      desc.Width * (sizeof(T) == 4 && (desc.Format == DXGI_FORMAT_R32_FLOAT || desc.Format == DXGI_FORMAT_R32_UINT) ? 1 : 4);
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
  // Empty for a format without an exact reference here, sRGB colour
  // included unless srgb_decoded: the darkening reference (fix 4) decodes it
  // with the sRGB EOTF in double, rounded to float, as the load does to well
  // within its 4/255 colour tolerance.
  std::vector<float> component(const artifact_t &a, unsigned channel, bool srgb_decoded = false) {
    const auto format = a.descriptor.at("dxgi_format").get<unsigned>();
    const auto width = a.descriptor.at("width").get<size_t>(), height = a.descriptor.at("height").get<size_t>();
    const auto row = a.descriptor.at("row_bytes").get<size_t>();
    const auto *bytes = reinterpret_cast<const unsigned char *>(a.bytes.data());
    std::vector<float> values(width * height);
    const auto texel = [&](size_t x, size_t y, size_t size) { return bytes + y * row + x * size; };
    const auto decoded = [](unsigned char code) {
      const double c = code / 255.;
      return float(c <= .04045 ? c / 12.92 : std::pow((c + .055) / 1.055, 2.4));
    };
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
          if (format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB && channel != 3) { // sRGB color is decoded on load.
            if (!srgb_decoded) return {};
            value = decoded(texel(x, y, 4)[channel]);
            break;
          }
          value = float(texel(x, y, 4)[channel]) / 255.f; break;
        case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
          if (format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB && channel != 3) {
            if (!srgb_decoded) return {};
            value = decoded(texel(x, y, 4)[2 - channel]);
            break;
          }
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
    ComPtr<ID3D11ComputeShader> tiles, reduce, mask, cells, compare, evidence, bits, count;
    // Fix 4: the darkening passes.
    ComPtr<ID3D11ComputeShader> dark_bits, dark_tiles, dark_region, dark_count, dark_finish;
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
  // (144 rows in all) and decision texel 11, and selection revision 5 (13
  // decision texels) H2's stillness rows from 144 (160 rows in all) and
  // decision texel 12, and (fix 4, the darkening planes) the darkening rows
  // from 192 (208 rows in all).
  struct sizes_t {
    UINT statistics_rows, decision_texels;
    bool scene;
  };
  sizes_t detection_sizes(const std::string &source) {
    auto texels = sunshine_game3d::shader_marker(source, contract::decision_texels_marker);
    const auto images = sunshine_game3d::shader_marker(source, contract::scene_evidence_images_marker);
    const auto planes = sunshine_game3d::shader_marker(source, contract::change_set::planes_marker);
    if (!texels) texels = contract::default_decision_texels;
    const bool scene = images == contract::max_scene_evidence_images && texels >= contract::scene_decision_texels;
    if (texels > contract::max_decision_texels || (images && !scene))
      throw std::runtime_error("unsupported detection size markers in the shader");
    return {std::max(80u, contract::statistics_rows(images, texels, planes)), std::max(6u, texels), scene};
  }

  // Fix 4: the darkening of the decided source (decision words 62-63)
  // against ui_darkening's CPU reference on the same inputs. The GPU's own
  // classes come from its planes (DARK, STRUCT and TILE after the region
  // pass) by the mask pass's rule; GPU and CPU may round apart where a
  // pixel's residual, fit or opacity step, or a tile's effect, lies at its
  // bound, so a differing unpinned pixel is excused within two pixels of
  // such a pixel, in the 3x3 tile ring of such a tile, or in a darkening
  // region with such a tile in the ring of one of its tiles (at_bound: one
  // tile can turn a whole region), and any other fails the case. The
  // colourless bit of word 63 must equal the CPU reference's.
  struct darkening_t {
    bool measured = false;
    std::string reference; // Why there is no CPU reference; empty with one.
    ui_darkening::counts cpu;
    std::uint64_t gpu_unpinned = 0, gpu_kept = 0, differ = 0, at_bound = 0;
    std::size_t first = 0;
    // The CPU reference's unpinned pixels (empty without one), which the
    // mask references zero when the mask pass applied the rule.
    std::vector<std::uint8_t> unpinned;
    bool applied = false, ok = true;
  };

  struct outcome {
    std::vector<std::uint32_t> decision;
    std::uint64_t ui_pixels = 0, pixels = 0, mask_hash = 1469598103934665603ull;
    std::string mask_exact; // Empty unless the label asks for it.
    std::vector<float> mask;
    UINT width = 0, height = 0;
    // The candidate layout, and the pushed candidate bits, accepted mask,
    // detection flags (of the second pass when measured) and b2 word 5's
    // rule bits.
    std::uint32_t layout = 0, offered = 0, accepted = 0, flags = 0, rules = 0;
    // scene_hold "measured": the per-frame bits the scene guard derived.
    bool measured = false;
    std::uint32_t guard_bits = 0;
    // Fix 3: the layer's pairing, the label's refine and pre_ui_proven, and
    // the change-set shadow; mask_reference's verdict when asked for.
    sunshine_game3d::change_set::layer_pairing pairing;
    bool refine = false, pre_ui_proven = false;
    sunshine_game3d::change_set::shadow_sample shadow;
    std::string mask_reference;
    // Fix 4: the decided source's darkening against its CPU reference.
    darkening_t darkening;
    // S3: the stamps and proposals pushed (b2 words 6-9 and t9), whether the
    // package or the label carried any (a legacy package replays with none
    // proposed and no gate, as the renderer did before S3), and whether the
    // shader verifies identity (texel 12 .z/.w).
    selection::identity_input identity;
    bool identity_labelled = false, identity_supported = false;
  };

  // S3: the identity a case replays. The package's replay.ui_detection
  // carries the renderer's b2 words 6-9 (expected_layer_present,
  // expected_layer_token, expected_hudless_present, identity_bits) and its
  // stamp buffer read back (stamps: one [C_P, C_T] per ui_ticket slot, the
  // layer at 4 and the HUD-less image at 5); a label's identity object
  // (the same names, plus layer_present, layer_token and hudless_present for
  // the stamp reads) overrides any of them.
  selection::identity_input identity_of(const json &metadata, const json &label, bool &labelled) {
    selection::identity_input in;
    labelled = false;
    const auto read = [&](const json &object, const char *name, std::uint32_t &field) {
      if (!object.is_object() || !object.contains(name)) return;
      field = object.at(name).get<std::uint32_t>();
      labelled = true;
    };
    if (metadata.contains("replay") && metadata.at("replay").contains("ui_detection")) {
      const auto &recorded = metadata.at("replay").at("ui_detection");
      read(recorded, "expected_layer_present", in.expected_layer_present);
      read(recorded, "expected_layer_token", in.expected_layer_token);
      read(recorded, "expected_hudless_present", in.expected_hudless_present);
      read(recorded, "identity_bits", in.bits);
      if (recorded.contains("stamps") && recorded.at("stamps").is_array()) {
        const auto &stamps = recorded.at("stamps");
        const auto entry = [&](std::size_t slot, std::size_t component) {
          return slot < stamps.size() && stamps[slot].is_array() && component < stamps[slot].size() ?
            stamps[slot][component].get<std::uint32_t>() : 0u;
        };
        in.layer_present = entry(contract::identity::stamp_layer, 0);
        in.layer_token = entry(contract::identity::stamp_layer, 1);
        in.hudless_present = entry(contract::identity::stamp_hudless, 0);
        labelled = true;
      }
    }
    if (label.contains("identity")) {
      const auto &override_ = label.at("identity");
      if (!override_.is_object()) throw std::runtime_error("identity must be an object");
      read(override_, "expected_layer_present", in.expected_layer_present);
      read(override_, "expected_layer_token", in.expected_layer_token);
      read(override_, "expected_hudless_present", in.expected_hudless_present);
      read(override_, "bits", in.bits);
      read(override_, "layer_present", in.layer_present);
      read(override_, "layer_token", in.layer_token);
      read(override_, "hudless_present", in.hudless_present);
    }
    return in;
  }

  // The " identity={...}" text of --identity: each offered pair's GPU verdict,
  // label space and delta (texel 12 .z/.w), whether the CPU's verify_identity
  // agrees, and whether the package or label carried stamps.
  std::string identity_text(const outcome &result) {
    const auto &d = result.decision;
    if (!result.identity_supported || d.size() <= word::id_deltas) return " identity=n/a";
    const auto gpu = selection::identity_of_words(d[word::id_verdicts], d[word::id_deltas]);
    const auto cpu = selection::verify_identity(result.identity, result.offered);
    static constexpr const char *verdicts[]{"none", "exact", "mismatch", "unstamped", "unproposed"};
    static constexpr const char *spaces[]{"none", "token", "present", "?"};
    const auto verdict = [](std::uint32_t v) { return v < 5u ? verdicts[v] : "?"; };
    char text[256];
    std::snprintf(text, sizeof(text), " identity={layer=%s/%s delta=%d hudless=%s/%s delta=%d labelled=%d cpu=%s}",
      verdict(gpu.layer), spaces[gpu.layer_space & 3u], gpu.layer_delta, verdict(gpu.hudless), spaces[gpu.hudless_space & 3u],
      gpu.hudless_delta, result.identity_labelled ? 1 : 0, gpu == cpu ? "match" : "differs");
    return text;
  }

  std::string mask_exact(const outcome &result, const std::array<const artifact_t *, 4> &inputs, const artifact_t *layer,
      const artifact_t &paired) {
    // Raw selected alpha, or a whole-frame flat (sources 6, 8 and 11); a
    // HUD-less difference has no CPU reference here. Where the mask pass
    // applied the darkening rule (fix 4), the CPU reference's unpinned
    // pixels read zero.
    const auto source = result.decision.at(word::source);
    std::vector<float> reference;
    if (!source) reference.assign(result.mask.size(), 0.f);
    else if (source == 6u || source == 8u || source == contract::source_still) reference.assign(result.mask.size(), 1.f);
    else if (source <= 4u || source == contract::source_layer) {
      const auto *input = source == 4u ? &paired : source == contract::source_layer ? layer : inputs[source - 1];
      if (!input) return "no-reference(missing candidate)";
      reference = component(*input, source == 1u ? 0 : 3);
      if (reference.empty()) return "no-reference(format " + input->descriptor.at("dxgi_format").dump() + ")";
    } else return "no-reference(source " + std::to_string(source) + ")";
    if (reference.size() != result.mask.size()) return "differs(extent)";
    if (result.darkening.applied) {
      if (result.darkening.unpinned.size() != reference.size())
        return "no-reference(darkening applied without its CPU reference: " + result.darkening.reference + ")";
      for (size_t i = 0; i < reference.size(); ++i)
        if (result.darkening.unpinned[i]) reference[i] = 0.f;
    }
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

  // mask_reference (fix 3): the pre-UI change set's mask (source 12) against
  // a CPU reference of the same rule, the layer against its pair's Present
  // changed beyond change_set::inferred_scale times b2 word 1 and kept by the
  // 3x3 rule, and (fix 4) without the darkening reference's unpinned pixels
  // where the mask pass applied the rule. GPU and CPU arithmetic may round a
  // pixel on the bound apart, so at most 0.01% of the pixels may differ.
  std::string mask_reference(const outcome &result, const artifact_t *layer, const artifact_t *pair, float threshold,
      unsigned color) {
    if (result.decision.at(word::source) != contract::source_pre_ui) return "no-reference(source not 12)";
    if (!layer || !pair) return "no-reference(no layer pair)";
    if (result.darkening.applied && result.darkening.unpinned.size() != result.mask.size())
      return "no-reference(darkening applied without its CPU reference: " + result.darkening.reference + ")";
    std::array<std::vector<float>, 3> final, pre;
    for (unsigned c = 0; c != 3; ++c) {
      final[c] = component(*pair, c);
      pre[c] = component(*layer, c);
      if (final[c].size() != result.mask.size() || pre[c].size() != result.mask.size())
        return "no-reference(format or extent)";
    }
    const float bound = threshold * float(contract::change_set::inferred_scale);
    const std::size_t width = result.width, height = result.height;
    std::vector<std::uint8_t> changed(width * height);
    for (std::size_t i = 0; i != changed.size(); ++i) {
      bool finite = true;
      float peak = 0.f, delta = 0.f;
      for (unsigned c = 0; c != 3; ++c) {
        finite = finite && std::isfinite(final[c][i]) && std::isfinite(pre[c][i]);
        peak = std::max(peak, std::abs(final[c][i]));
        delta = std::max(delta, std::abs(final[c][i] - pre[c][i]));
      }
      changed[i] = finite && delta / (color == 2 ? std::max(1.f, peak) : 1.f) > bound;
    }
    std::uint64_t different = 0, reference_pixels = 0;
    for (std::size_t y = 0; y != height; ++y)
      for (std::size_t x = 0; x != width; ++x) {
        std::uint32_t count = 0;
        if (changed[y * width + x])
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
              const auto nx = std::ptrdiff_t(x) + dx, ny = std::ptrdiff_t(y) + dy;
              if (nx >= 0 && ny >= 0 && nx < std::ptrdiff_t(width) && ny < std::ptrdiff_t(height))
                count += changed[std::size_t(ny) * width + std::size_t(nx)];
            }
        const bool unpinned = result.darkening.applied && result.darkening.unpinned[y * width + x];
        const float reference = count >= contract::change_set::min_neighbourhood && !unpinned ? 1.f : 0.f;
        reference_pixels += reference > 0.f;
        different += result.mask[y * width + x] != reference;
      }
    char text[160];
    std::snprintf(text, sizeof(text), "%s(%llu of %llu pixels differ, reference %llu UI pixels)",
      different * 10000u <= std::uint64_t(width) * height ? "match" : "differs", static_cast<unsigned long long>(different),
      static_cast<unsigned long long>(width * height), static_cast<unsigned long long>(reference_pixels));
    return text;
  }

  // Fix 4: the darkening mirror of an eligible decided source (not reused,
  // as no replay is). planes are the GPU's plane words (DARK, STRUCT and
  // TILE); the CPU reference reads the source's own artifacts: the UI color
  // tag (2) and the layer (10) by opacity, the paired color against the
  // HUD-less image (5, k 1) and the layer pair's Present against the layer
  // (12, k change_set::inferred_scale, with the 3x3 rule) by change set, at
  // the change-set slot's threshold t.
  darkening_t darkening_of(const std::vector<std::uint32_t> &decision, const std::vector<std::uint32_t> &planes, UINT width,
      UINT height, std::uint32_t rules, std::uint32_t flags, unsigned color, float t, const artifact_t *tag, const artifact_t *layer,
      const artifact_t *paired, const artifact_t *hudless, const artifact_t *pair) {
    darkening_t r;
    if (decision.size() <= word::dk_kept) return r;
    const auto source = decision[word::source];
    if (!contract::darkening_dispatched(rules, source, (decision[word::frame_reason] & contract::frame_reason_reused) != 0u))
      return r;
    r.measured = true;
    r.applied = contract::darkening_applied(rules, source);
    namespace plane = contract::change_set::plane;
    const std::size_t count = std::size_t(width) * height, words = contract::change_set::plane_words(width),
      row = contract::change_set::planes * words;
    if (planes.size() != row * height) {
      r.reference = "no planes";
      r.ok = false;
      return r;
    }
    const auto bit = [&](std::uint32_t p, std::uint32_t x, std::uint32_t y) {
      return ((planes[std::size_t(y) * row + p * words + x / 32u] >> (x % 32u)) & 1u) != 0u;
    };
    const bool opacity = ui_darkening::opacity_source(source);
    // The GPU's classes, by the mask pass's rule: 1 kept, 2 unpinned. An
    // opacity source unpins only on a frame with colour (TILE row 1, word 0,
    // bit 0), a change set only in a tile whose region unpins (row 3 ty + 2).
    const bool coloured = opacity && height > 1u && bit(plane::tile, 0u, 1u);
    std::vector<std::uint8_t> gpu(count);
    for (std::uint32_t y = 0; y != height; ++y)
      for (std::uint32_t x = 0; x != width; ++x) {
        if (!bit(plane::dark, x, y)) continue;
        bool kept = false;
        if (bit(plane::structure, x, y)) {
          unsigned around = 0;
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
              const int nx = int(x) + dx, ny = int(y) + dy;
              if (nx >= 0 && ny >= 0 && nx < int(width) && ny < int(height))
                around += bit(plane::structure, std::uint32_t(nx), std::uint32_t(ny)) ? 1u : 0u;
            }
          kept = around >= contract::change_set::min_neighbourhood;
        }
        const std::uint32_t tx = x / ui_darkening::tile, ty = y / ui_darkening::tile;
        const std::uint32_t region_row = contract::darkening::tile_rows * ty + 2u;
        const bool proven = opacity ? coloured : region_row < height && bit(plane::tile, tx, region_row);
        const bool unpinned = !kept && proven;
        gpu[std::size_t(y) * width + x] = unpinned ? 2u : 1u;
        (unpinned ? r.gpu_unpinned : r.gpu_kept) += 1u;
      }
    // The count and finish passes sum exactly these classes.
    const bool colourless = (decision[word::dk_kept] & contract::darkening::colourless) != 0u;
    if (r.gpu_unpinned != decision[word::dk_unpinned] || r.gpu_kept != (decision[word::dk_kept] & ~contract::darkening::colourless) ||
        colourless != (opacity && !coloured)) {
      r.reference = "words 62-63 differ from the GPU planes";
      r.ok = false;
      return r;
    }
    // The CPU reference.
    const auto channels = [&](const artifact_t *a) {
      std::array<std::vector<float>, 4> values;
      if (!a) return values;
      for (unsigned c = 0; c != 4; ++c) values[c] = component(*a, c, true);
      return values;
    };
    const auto usable = [&](const std::array<std::vector<float>, 4> &values) {
      return std::all_of(values.begin(), values.end(), [&](const std::vector<float> &v) { return v.size() == count; });
    };
    const auto rgb_of = [&](const std::array<std::vector<float>, 4> &values) {
      std::vector<ui_darkening::rgb> image(count);
      for (std::size_t i = 0; i != count; ++i) image[i] = {values[0][i], values[1][i], values[2][i]};
      return image;
    };
    ui_darkening::result reference;
    std::vector<std::uint8_t> fragile(count), fragile_tiles;
    const std::uint32_t tiles_x = ui_darkening::tiles(width), tiles_y = ui_darkening::tiles(height);
    const auto tile_of = [&](std::size_t i) {
      return (i / width / ui_darkening::tile) * tiles_x + (i % width) / ui_darkening::tile;
    };
    if (opacity) {
      const auto values = channels(source == 2u ? tag : layer);
      if (!usable(values)) {
        r.reference = "no CPU reference for the source's format";
        return r;
      }
      // A float source stores linear colour: the tag by its own format
      // (rules::tag_linear), the layer by its flags (stored_hdr_headroom).
      const bool linear = source == 2u ? (rules & contract::rules::tag_linear) != 0u : (flags & contract::stored_hdr_headroom) != 0u;
      reference = ui_darkening::reference_opacity(width, height, rgb_of(values), values[3], linear);
      if (reference.colourless != colourless) {
        r.reference = std::string("the colourless bit differs from the CPU reference's (GPU ") + (colourless ? "1" : "0") + ")";
        r.ok = false;
        return r;
      }
    } else {
      const bool pre_ui = source == contract::source_pre_ui;
      const auto final_values = channels(pre_ui ? pair : paired), pre_values = channels(pre_ui ? layer : hudless);
      if (!usable(final_values) || !usable(pre_values)) {
        r.reference = "no CPU reference for the pair's formats";
        return r;
      }
      const auto presented = rgb_of(final_values), pre = rgb_of(pre_values);
      const float k = pre_ui ? float(contract::change_set::inferred_scale) : 1.f;
      const auto mask = ui_darkening::change_set_mask(width, height, presented, pre, t, k, color, pre_ui);
      reference = ui_darkening::reference_change_set(width, height, presented, pre, mask, t, k, color);
      // Pixels at a bound: the residual within 1% of its bound, the fit at 1,
      // the changed bound, and opacity steps at the step bound (PQ's pow
      // rounds most); then tiles with such a pixel or an effect within 1% of
      // the tile bound.
      const float margin = color == ui_darkening::color_space::pq ? 1e-3f : 1e-5f;
      const float residual_bound = ui_darkening::change_set_tolerance_scale * t;
      std::vector<ui_darkening::evidence> e(count);
      std::vector<ui_darkening::tile_sums> sums(std::size_t(tiles_x) * tiles_y);
      for (std::size_t i = 0; i != count; ++i) {
        if (!ui_darkening::finite(presented[i]) || !ui_darkening::finite(pre[i])) continue;
        e[i] = ui_darkening::change_set(presented[i], pre[i], t, k, mask[i] != 0u, color);
        const auto fit = ui_darkening::fit_darkening(presented[i], pre[i], color);
        const float delta = ui_darkening::color_difference(presented[i], pre[i], color);
        if (std::abs(fit.residual - residual_bound) <= .01f * residual_bound || std::abs(fit.s_raw - 1.f) <= margin ||
            std::abs(delta - k * t) <= margin * k * t)
          fragile[i] = 1u;
        if (e[i].fits) sums[tile_of(i)].add(presented[i], pre[i], color);
      }
      for (std::uint32_t y = 0; y != height; ++y)
        for (std::uint32_t x = 0; x != width; ++x) {
          const std::size_t i = std::size_t(y) * width + x;
          for (const std::size_t n : {x + 1 < width ? i + 1 : i, y + 1 < height ? i + width : i}) {
            if (n == i || !e[i].finite || !e[n].finite) continue;
            const float step = std::abs(e[i].a - e[n].a) - e[i].sigma - e[n].sigma;
            if (std::abs(step - ui_darkening::step) <= std::max(margin, 1e-4f)) fragile[i] = fragile[n] = 1u;
          }
        }
      fragile_tiles.assign(sums.size(), 0u);
      for (std::size_t i = 0; i != sums.size(); ++i) {
        const double effect = ui_darkening::tile_effect(sums[i], t, color);
        if (effect >= 0. && std::abs(effect - ui_darkening::tile_effect_scale) <= .01 * ui_darkening::tile_effect_scale)
          fragile_tiles[i] = 1u;
      }
      for (std::size_t i = 0; i != count; ++i)
        if (fragile[i]) fragile_tiles[tile_of(i)] = 1u;
    }
    r.cpu = reference.n;
    // Fragile pixels reach two pixels (STRUCT reads 4-neighbours, kept a 3x3
    // window); fragile tiles their 3x3 ring.
    std::vector<std::uint8_t> close_to_bound(count);
    for (std::uint32_t y = 0; y != height; ++y)
      for (std::uint32_t x = 0; x != width; ++x) {
        if (!fragile[std::size_t(y) * width + x]) continue;
        for (int dy = -2; dy <= 2; ++dy)
          for (int dx = -2; dx <= 2; ++dx) {
            const int nx = int(x) + dx, ny = int(y) + dy;
            if (nx >= 0 && ny >= 0 && nx < int(width) && ny < int(height)) close_to_bound[std::size_t(ny) * width + std::size_t(nx)] = 1u;
          }
      }
    // Regions (change sets) with a fragile tile in the ring of a member.
    std::vector<std::uint8_t> fragile_region;
    if (!fragile_tiles.empty() && !reference.component.empty()) {
      fragile_region.assign(std::size_t(reference.n.regions), 0u);
      for (std::uint32_t ty = 0; ty != tiles_y; ++ty)
        for (std::uint32_t tx = 0; tx != tiles_x; ++tx) {
          const auto c = reference.component[std::size_t(ty) * tiles_x + tx];
          if (c < 0) continue;
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
              const int nx = int(tx) + dx, ny = int(ty) + dy;
              if (nx >= 0 && ny >= 0 && nx < int(tiles_x) && ny < int(tiles_y) &&
                  fragile_tiles[std::size_t(ny) * tiles_x + std::size_t(nx)])
                fragile_region[std::size_t(c)] = 1u;
            }
        }
    }
    r.unpinned.assign(count, 0u);
    for (std::uint32_t y = 0; y != height; ++y)
      for (std::uint32_t x = 0; x != width; ++x) {
        const std::size_t i = std::size_t(y) * width + x;
        const bool cpu = reference.classes[i] == ui_darkening::pixel_class::unpinned;
        r.unpinned[i] = cpu ? 1u : 0u;
        if (cpu == (gpu[i] == 2u)) continue;
        bool excused = close_to_bound[i] != 0u;
        if (!excused && !fragile_region.empty()) {
          const auto c = reference.component[std::size_t(y / ui_darkening::tile) * tiles_x + x / ui_darkening::tile];
          excused = c >= 0 && fragile_region[std::size_t(c)];
        }
        if (!excused && !fragile_tiles.empty()) {
          const int tx = int(x / ui_darkening::tile), ty = int(y / ui_darkening::tile);
          for (int dy = -1; dy <= 1 && !excused; ++dy)
            for (int dx = -1; dx <= 1 && !excused; ++dx) {
              const int nx = tx + dx, ny = ty + dy;
              excused = nx >= 0 && ny >= 0 && nx < int(tiles_x) && ny < int(tiles_y) &&
                fragile_tiles[std::size_t(ny) * tiles_x + std::size_t(nx)];
            }
        }
        if (excused) ++r.at_bound;
        else if (!r.differ++) r.first = i;
      }
    r.ok = r.differ == 0u;
    return r;
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
    auto layout = sunshine_game3d::shader_marker(shader_source, contract::candidate_layout_marker);
    if (!layout) layout = contract::legacy_candidate_layout;
    std::array<const artifact_t *, 4> inputs{}; // t11..t14
    const artifact_t *layer_input = nullptr;    // t7 (layout 2)
    unsigned bits = 0, flags = 0;
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
      if (layer) flags = contract::layer_detection_flags(contract::float_layer_format(artifacts.at(name).at("dxgi_format").get<unsigned>()));
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
    // Fix 3: the layer's pairing. layer_pair names the retained Present the
    // dump carries (retained_present_1 at t2, retained_present_2 at t3);
    // without one the copy is the one-frame-late layer, paired with nothing
    // retained (late). layer_pair_exact, a test-only override, binds the
    // presented color at t2 as the retained Present one back, to exercise
    // the decision path on dumps taken before retained Presents existed.
    namespace change_set = sunshine_game3d::change_set;
    change_set::layer_pairing pairing;
    const artifact_t *pair_input = nullptr; // t2 or t3
    std::string pair_kind;
    bool retained_1 = false, retained_2 = false;
    // Fix 4: pin_only_ui is the same switch under its UIPinOnlyUI name.
    const bool refine = label.value("refine", false) || label.value("pin_only_ui", false),
      pair_exact = label.value("layer_pair_exact", false);
    if (pair_exact && label.contains("layer_pair")) throw std::runtime_error("layer_pair and layer_pair_exact exclude each other");
    if (layer_input) {
      pairing = {change_set::pair_class::late, 0u};
      if (pair_exact) pair_kind = "source_color";
      else if (label.contains("layer_pair")) {
        pair_kind = label.at("layer_pair").get<std::string>();
        if (pair_kind != "retained_present_1" && pair_kind != "retained_present_2")
          throw std::runtime_error("layer_pair must be retained_present_1 or retained_present_2");
      }
      if (!pair_kind.empty()) {
        pair_input = &load(pair_kind);
        retained_2 = pair_kind == "retained_present_2";
        retained_1 = !retained_2;
        pairing = {change_set::pair_class::retained, retained_2 ? 2u : 1u};
      }
    } else if (pair_exact || label.contains("layer_pair")) {
      throw std::runtime_error("layer_pair needs an offered layer in candidate layout 2");
    }
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
    // The pre-UI change set as the renderer offers it: Auto with the switch
    // (refine), the proven layer, no HUD-less image and an exact retained
    // pairing whose two images are comparable; its acceptance is the proof
    // (the pre_ui key of the layer's signature).
    std::optional<float> pre_ui_pair_threshold;
    std::string layer_name;
    for (const auto &kind : label.at("candidates"))
      if (ui_layer_kind(kind.get<std::string>())) layer_name = kind.get<std::string>();
    if (pair_input && !layer_name.empty())
      pre_ui_pair_threshold = selection::comparable(encoding_of(layer_name), encoding_of(pair_kind));
    if (pre_ui_pair_threshold &&
        change_set::offered(true, refine, pre_ui_proven, (bits & candidate::hudless) != 0u, pairing)) {
      bits |= candidate::pre_ui;
      accepted |= candidate::pre_ui;
      signatures[std::size_t(selection::kind::pre_ui)] = selection::pre_ui_key(signatures[std::size_t(selection::kind::ui_layer)]);
    }
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
    // the shader never read it.
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
    // Fix 3: a shader with the change-set shadow's bit planes measures it in
    // passes of its own after the tiles pass, as the renderer dispatches them.
    // Fix 4: its planes marker also says it has the darkening passes.
    const bool planes = sunshine_game3d::shader_marker(shader_source, contract::change_set::planes_marker) ==
      contract::change_set::planes;
    texture_t plane_texture;
    if (planes) {
      shaders.bits = compile(gpu, shader_source, "SunshineUIDetectionChangeSetBitsCS", width, height, color);
      shaders.count = compile(gpu, shader_source, "SunshineUIDetectionChangeSetCountCS", width, height, color);
      shaders.dark_bits = compile(gpu, shader_source, "SunshineUIDarkeningBitsCS", width, height, color);
      shaders.dark_tiles = compile(gpu, shader_source, "SunshineUIDarkeningTilesCS", width, height, color);
      shaders.dark_region = compile(gpu, shader_source, "SunshineUIDarkeningRegionCS", width, height, color);
      shaders.dark_count = compile(gpu, shader_source, "SunshineUIDarkeningCountCS", width, height, color);
      shaders.dark_finish = compile(gpu, shader_source, "SunshineUIDarkeningFinishCS", width, height, color);
      plane_texture = target(gpu, contract::change_set::planes * contract::change_set::plane_words(width), height,
        DXGI_FORMAT_R32_UINT);
    }
    auto statistics = target(gpu, 16, sizes.statistics_rows, DXGI_FORMAT_R32G32B32A32_UINT);
    texture_t cells;
    if (sizes.scene) cells = target(gpu, contract::scene::cells_x, contract::scene::cells_y, DXGI_FORMAT_R32G32B32A32_UINT);
    auto decision = target(gpu, sizes.decision_texels, 1, DXGI_FORMAT_R32G32B32A32_UINT);
    auto mask = target(gpu, width, height, DXGI_FORMAT_R32_FLOAT);
    // Detection constants b2: candidate bits, difference threshold (the
    // change-set slot's pair: the HUD-less pair, or the pre-UI layer's when
    // its change set is offered), accepted candidates (layout 1: trusted
    // slots), flags, (selection revision 4) the offscreen UI layer's pair
    // threshold with the presented color, from the two artifacts' own
    // encodings (zero without a layer or when not comparable), and
    // (selection revision 5) b2 word 5: H2's still-screen flag as the dump
    // recorded it and (revision 6) the label's refine and the pairing's rule
    // bits, padded to the 16-byte constant buffer granularity.
    const float threshold = pair_threshold ? *pair_threshold : (bits & candidate::pre_ui) ? *pre_ui_pair_threshold :
      selection::comparable(encoding_of(paired), encoding_of(paired)).value_or(2.f / 255.f);
    float pre_ui_threshold = 0.f;
    for (const auto &kind : label.at("candidates"))
      if (ui_layer_kind(kind.get<std::string>()) && layer_input)
        pre_ui_threshold =
          selection::comparable(encoding_of(kind.get<std::string>()), encoding_of("source_color")).value_or(0.f);
    std::uint32_t still = 0;
    if (const auto &replay = metadata.at("replay"); replay.contains("ui_detection")) {
      const auto &recorded = replay.at("ui_detection");
      if (recorded.contains("rules_bits")) still = recorded.at("rules_bits").get<std::uint32_t>() & contract::still::flatten;
      else if (recorded.contains("still_bits")) still = recorded.at("still_bits").get<std::uint32_t>();
    }
    // The shadow bit as the renderer pushes it: the offered layer proven the
    // pre-UI scene image (pre_ui_proven); a single frame has no T1 gap. Fix
    // 4: a replay frame is a sample, so the darkening passes always measure,
    // with rules::tag_linear when the UI color tag has a float format.
    const bool tag_linear = inputs[1] && contract::float_layer_format(inputs[1]->descriptor.at("dxgi_format").get<std::uint32_t>());
    const std::uint32_t rules = still |
      change_set::rule_bits(false, refine, pre_ui_proven, false, pairing, retained_1, retained_2) |
      (planes ? contract::rules::darkening_measured | (tag_linear ? contract::rules::tag_linear : 0u) : 0u);
    // S3: b2 words 6-9 and the stamp buffer at t9 (identity_of; a legacy
    // package proposes nothing and pushes no gate).
    bool identity_labelled = false;
    const auto identity = identity_of(metadata, label, identity_labelled);
    struct {
      std::uint32_t bits;
      float threshold;
      std::uint32_t accepted, flags;
      float pre_ui_threshold;
      std::uint32_t rules;
      std::uint32_t expected_layer_present, expected_layer_token, expected_hudless_present, identity_bits;
      std::uint32_t padding[2];
    } constants{bits, threshold, accepted, flags, pre_ui_threshold, rules, identity.expected_layer_present,
      identity.expected_layer_token, identity.expected_hudless_present, identity.bits, {}};
    static_assert(sizeof(constants) == 48);
    ComPtr<ID3D11Buffer> stamp_buffer;
    ComPtr<ID3D11ShaderResourceView> stamp_view;
    {
      std::array<std::array<std::uint32_t, 4>, contract::identity::stamp_entries> entries{};
      entries[contract::identity::stamp_layer] = {identity.layer_present, identity.layer_token, 0u, 0u};
      entries[contract::identity::stamp_hudless] = {identity.hudless_present, 0u, 0u, 0u};
      D3D11_BUFFER_DESC buffer{};
      buffer.ByteWidth = UINT(sizeof(entries));
      buffer.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      const D3D11_SUBRESOURCE_DATA initial{entries.data(), 0, 0};
      checked(gpu.device->CreateBuffer(&buffer, &initial, &stamp_buffer), "stamps");
      D3D11_SHADER_RESOURCE_VIEW_DESC view{};
      view.Format = DXGI_FORMAT_R32G32B32A32_UINT;
      view.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
      view.Buffer.NumElements = contract::identity::stamp_entries;
      checked(gpu.device->CreateShaderResourceView(stamp_buffer.Get(), &view, &stamp_view), "stamps view");
    }
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
    // The count pass's bit planes (t8).
    ID3D11ShaderResourceView *t8 = nullptr;
    const auto stage = [&](ID3D11ComputeShader *shader, ID3D11ShaderResourceView *t10, UINT uav_slot, ID3D11UnorderedAccessView *uav, UINT x, UINT y) {
      std::array<ID3D11ShaderResourceView *, 15> views{};
      views[0] = paired_color.gpu.srv.Get();
      views[1] = depth.srv.Get();
      // Fix 3: the retained Present of the layer pair (t2 one back, t3 two).
      if (pair_input) views[retained_2 ? 3 : 2] = pair_input->gpu.srv.Get();
      views[6] = presented.gpu.srv.Get();
      views[7] = layer_input ? layer_input->gpu.srv.Get() : nullptr;
      views[8] = t8;
      views[9] = stamp_view.Get();
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
    if (planes && contract::change_set::shadow_dispatched(bits, pre_ui_threshold, rules)) {
      stage(shaders.bits.Get(), nullptr, 4, plane_texture.uav.Get(), contract::change_set::plane_words(width),
        (height + contract::change_set::bits_group_rows - 1) / contract::change_set::bits_group_rows);
      t8 = plane_texture.srv.Get();
      stage(shaders.count.Get(), nullptr, 6, statistics.uav.Get(), 16, 16);
      t8 = nullptr;
    }
    // The reduce, then (fix 4) the darkening passes as the renderer
    // dispatches them on a sample frame: the bits, tiles and region passes
    // into the planes, the mask pass reading them at t8, and the count and
    // finish passes into decision words 62-63.
    const auto decide_and_mask = [&]() {
      stage(shaders.reduce.Get(), statistics.srv.Get(), 6, decision.uav.Get(), 1, 1);
      if (planes) {
        stage(shaders.dark_bits.Get(), decision.srv.Get(), 4, plane_texture.uav.Get(), contract::change_set::plane_words(width),
          (height + contract::change_set::bits_group_rows - 1) / contract::change_set::bits_group_rows);
        stage(shaders.dark_tiles.Get(), decision.srv.Get(), 4, plane_texture.uav.Get(), contract::darkening::tiles(width),
          contract::darkening::tiles(height));
        stage(shaders.dark_region.Get(), decision.srv.Get(), 4, plane_texture.uav.Get(), 1, 1);
        t8 = plane_texture.srv.Get();
      }
      stage(shaders.mask.Get(), decision.srv.Get(), 0, mask.uav.Get(), (width + 7) / 8, (height + 7) / 8);
      if (planes) {
        stage(shaders.dark_count.Get(), decision.srv.Get(), 6, statistics.uav.Get(), 16, 16);
        t8 = nullptr;
        stage(shaders.dark_finish.Get(), statistics.srv.Get(), 6, decision.uav.Get(), 1, 1);
      }
    };
    decide_and_mask();
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
      const auto words = download<std::uint32_t>(gpu, decision);
      sunshine_game3d::scene_guard::state guard;
      guard.enter_scope(1, 0);
      const bool proven_image = pre_ui_proven && selection::pre_ui_image_of(bits) == contract::pre_ui_image::layer;
      for (const std::uint64_t tick : {900u, 1000u, 1100u})
        guard.observe(sunshine_game3d::scene_guard::sample_of(words.data(), words.size(), tick, true),
          guard.measure(tick, false, false, proven_image).actionable, signatures);
      guard_bits = guard.per_frame(1100, bits, signatures, pre_ui_proven);
      flags |= guard_bits;
      constants.flags = flags;
      cb = constant_buffer(&constants, sizeof(constants));
      decide_and_mask();
      stage(shaders.evidence.Get(), statistics.srv.Get(), 6, decision.uav.Get(), 1, 1);
    }

    outcome result;
    result.measured = measured;
    result.guard_bits = guard_bits;
    result.decision = download<std::uint32_t>(gpu, decision);
    result.mask = download<float>(gpu, mask);
    result.width = width; result.height = height;
    result.layout = layout; result.offered = bits; result.accepted = accepted; result.flags = flags; result.rules = rules;
    result.identity = identity;
    result.identity_labelled = identity_labelled;
    result.identity_supported = sunshine_game3d::shader_marker(shader_source, contract::identity::marker) ==
      contract::identity::version;
    result.pairing = pairing;
    result.refine = refine;
    result.pre_ui_proven = pre_ui_proven;
    if (result.decision.size() >= 4u * contract::change_set_decision_texels)
      result.shadow = change_set::measure_shadow(selection::counts_from_words(result.decision.data(), result.decision.size()),
        bits, accepted, flags, rules, pairing, pre_ui_proven, (bits & candidate::hudless) != 0u, true, refine);
    if (planes)
      result.darkening = darkening_of(result.decision, download<std::uint32_t>(gpu, plane_texture), width, height, rules, flags, color,
        threshold, inputs[1], layer_input, &paired_color, inputs[3], pair_input);
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
    if (label.at("expect").value("mask_reference", false))
      result.mask_reference = mask_reference(result, layer_input, pair_input, threshold, color);
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
    if (d.size() <= word::presented_lit_differs) {
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

  // expect.change_set (fix 3): the change-set shadow as
  // change_set::measure_shadow reads it: "class" (the pairing), "valid",
  // "would_refine", and inclusive bounds changed_min/max, filtered_min/max,
  // matching_tiles_min/max, judge_precision_min/max and judge_recall_min/max
  // (a bound on a ratio without a basis fails). A shader without texels
  // 13-15 fails the check.
  bool change_set_matches(const outcome &result, const json &expected, std::string &text) {
    namespace change_set = sunshine_game3d::change_set;
    if (result.decision.size() < 4u * contract::change_set_decision_texels) {
      text = "no-reference(" + std::to_string(result.decision.size() / 4) + " decision texels)";
      return false;
    }
    const auto &s = result.shadow;
    bool okay = true;
    if (expected.contains("class")) okay = okay && expected.at("class").get<std::string>() == change_set::name(s.pairing.kind);
    if (expected.contains("valid")) okay = okay && expected.at("valid").get<bool>() == s.valid;
    if (expected.contains("would_refine")) okay = okay && expected.at("would_refine").get<bool>() == s.would_refine;
    const auto bounded = [&](const char *key, double value) {
      const auto low = std::string(key) + "_min", high = std::string(key) + "_max";
      if (expected.contains(low)) okay = okay && value >= expected.at(low).get<double>();
      if (expected.contains(high)) okay = okay && value <= expected.at(high).get<double>();
    };
    bounded("changed", s.counts.changed);
    bounded("filtered", s.counts.filtered);
    bounded("matching_tiles", s.counts.matching_tiles);
    bounded("judge_precision", s.precision);
    bounded("judge_recall", s.recall);
    text = okay ? "match" : "differs";
    return okay;
  }

  // One expectation: a case's "expect", or the "today" outcome of its xfail.
  struct judged_t {
    bool okay = false;
    std::string wanted_mask, scene, pre_ui_scene, pre_ui_match, change_set, darkening;
  };

  // expect.darkening (fix 4): inclusive bounds unpinned_min/max and
  // kept_min/max on decision words 62-63 (the GPU's darkening of the decided
  // source). A shader without them fails the check.
  bool darkening_matches(const outcome &result, const json &expected, std::string &text) {
    if (result.decision.size() <= word::dk_kept) {
      text = "no-reference(" + std::to_string(result.decision.size() / 4) + " decision texels)";
      return false;
    }
    bool okay = true;
    const auto bounded = [&](const char *key, double value) {
      const auto low = std::string(key) + "_min", high = std::string(key) + "_max";
      if (expected.contains(low)) okay = okay && value >= expected.at(low).get<double>();
      if (expected.contains(high)) okay = okay && value <= expected.at(high).get<double>();
    };
    bounded("unpinned", result.decision[word::dk_unpinned]);
    bounded("kept", result.decision[word::dk_kept] & ~contract::darkening::colourless);
    if (expected.contains("colourless"))
      okay = okay && ((result.decision[word::dk_kept] & contract::darkening::colourless) != 0u) ==
        expected.at("colourless").get<bool>();
    text = okay ? "match" : "differs";
    return okay;
  }
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
    if (expect.contains("change_set"))
      judged.okay = change_set_matches(result, expect.at("change_set"), judged.change_set) && judged.okay;
    if (expect.contains("darkening"))
      judged.okay = darkening_matches(result, expect.at("darkening"), judged.darkening) && judged.okay;
    if (expect.value("mask_reference", false)) judged.okay = judged.okay && result.mask_reference.rfind("match", 0) == 0;
    if (expect.contains("refined"))
      judged.okay = judged.okay && d.size() > word::h1 &&
        ((d[word::h1] & contract::h1_refined) != 0u) == expect.at("refined").get<bool>();
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
    static const std::array<const char *, 7> stages{"S1", "S2a", "S2b", "S3", "S4", "S5", "S6"};
    const auto stage = xfail.value("stage", std::string{});
    if (std::none_of(stages.begin(), stages.end(), [&](const char *name) { return stage == name; }))
      return "xfail.stage must be one of S1, S2a, S2b, S3, S4, S5, S6";
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
  // candidate bits, accepted mask, flags and rule bits and no previous
  // decision (the hold store unbound): "match" when the GPU's source,
  // coverage, accepted word, valid bits, refused candidate, frame reason,
  // claims and h1 word (its applied and refined bits included) agree, "n/a"
  // for a shader of another candidate layout or selection revision.
  std::string mirror_of(const outcome &result, bool mirrored) {
    const auto &d = result.decision;
    if (!mirrored || d.size() <= word::h1) return "n/a";
    const auto expected = selection::decide(selection::counts_from_words(d.data(), d.size()), result.offered, result.accepted,
      result.flags, selection::hold_state{}, result.rules, result.identity);
    // S3: a shader that verifies identity also writes its verdicts (texel 12).
    const bool identity = !result.identity_supported || (d.size() > word::id_deltas &&
      d[word::id_verdicts] == selection::identity_verdict_word(expected.identity) &&
      d[word::id_deltas] == selection::identity_delta_word(expected.identity));
    if (expected.source == d[word::source] && expected.covered == d[word::covered] && expected.valid_bits == d[word::valid_bits] &&
        d[word::accepted] == result.accepted && d[word::candidates] == result.offered && expected.refused == d[word::refused] &&
        selection::frame_reason_word(expected) == d[word::frame_reason] && expected.claims == d[word::claims] &&
        selection::h1_word(expected) == d[word::h1] && identity)
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
    bool verbose = false, strict = false, identity = false;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--write-mask" && i + 1 < argc && !write_masks) write_masks = fs::absolute(argv[++i]);
      else if (argument == "--verbose") verbose = true;
      else if (argument == "--strict") strict = true;
      else if (argument == "--identity") identity = true;
      else positional.push_back(argument);
    }
    if (positional.size() != 2) {
      std::fprintf(stderr, "Usage: ui_detection_replay <game3d_native.hlsl> <cases.json> [--write-mask <new-dir>] [--verbose] "
        "[--strict] [--identity]\n"
        "cases.json: {\"dump_root\": dir, \"cases\": [{\"dump\", \"label\", \"candidates\": [kinds|\"current\"], "
        "\"paired\": kind, \"exact\": bool, \"accepted\": [kinds], \"scene_hold\": bool|\"measured\", "
        "\"pre_ui_visible\": bool, \"pre_ui_proven\": bool, \"refuted\": [kinds], \"depth_not_current\": bool, "
        "\"refine\"|\"pin_only_ui\": bool, \"layer_pair\": \"retained_present_1\"|\"retained_present_2\", "
        "\"layer_pair_exact\": bool, "
        "\"expect\": {\"mask\", "
        "\"source\": [ids], \"mask_exact\": bool, \"scene\": {\"verdict\", \"d_min\", \"d_max\"}, "
        "\"pre_ui_scene\": {\"image\", \"verdict\", \"d_min\", \"d_max\"}, \"pre_ui_match\": bool, "
        "\"change_set\": {\"class\", \"valid\", \"would_refine\", \"<count>_min\", \"<count>_max\"}, "
        "\"mask_reference\": bool, \"refined\": bool, \"darkening\": {\"<unpinned|kept>_min\", \"<unpinned|kept>_max\"}}, "
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
        "S2b, S3-S6) that reaches it, xfail.reason the rule it applies (E1, E2, V1, V2, A1-A3, S1, S2, H1, P1, T1, F1)\n"
        "and xfail.today the outcome it has now\n"
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
        "(strong/contradicted pixels of the layer, Backbuffer and current alpha) from selection revision 2, and the H1\n"
        "claims and h1 word (applied, S1 winner) from selection revision 3, and the layer's pre-UI pixel counts\n"
        "(texel 11: match, image_lit, presented_lit, presented_lit_differs) from selection revision 4.\n"
        "Fix 3 (selection revision 6): refine pushes b2 word 5's refine bit (UIPinOnlyUI=1). The layer (a census\n"
        "ui_layer_candidate_N or ui_layer_detected) pairs with the retained Present layer_pair names (its artifact at t2\n"
        "or t3, offset 1 or 2), else it is late (offset 0); layer_pair_exact, a test-only override, binds source_color at\n"
        "t2 as the Present one back to exercise the decision on a late dump. The pre-UI change set (candidate 0x100,\n"
        "source 12) is offered as the renderer offers it (game3d_ui_change_set.h: refine, pre_ui_proven, no HUD-less\n"
        "image, a retained pairing of comparable images), accepted by the proof, at the pair's threshold. Every line shows\n"
        "change_set={...} (texels 13-15 through change_set::measure_shadow: pairing class and offset, changed, unchanged,\n"
        "matching tiles, lit, 3x3-filtered, changed against the Presents 0/1/2 back, the pair verdict, the judge's\n"
        "precision, recall and IoU, validity and what the switch would do) and refined= (the h1 word's bit). change_set\n"
        "checks those fields (class, valid, would_refine, and _min/_max bounds of changed, filtered, matching_tiles,\n"
        "judge_precision, judge_recall); mask_reference compares a source-12 mask with a CPU reference of the same rule\n"
        "(at most 0.01%% of pixels may differ); refined checks the h1 word's refined bit. As in the renderer, the shadow\n"
        "measures only a layer proven the pre-UI scene image (pre_ui_proven pushes b2 word 5's shadow bit); without the\n"
        "proof its counts are zero.\n"
        "Fix 4 (pin only UI, rule P2): a shader whose SUNSHINE_UI_CHANGE_SET_PLANES is 9 runs its darkening passes after the\n"
        "reduce as on a sample frame (b2 word 5 always has 0x100; refine, or its alias pin_only_ui, adds 0x2, so the mask\n"
        "pass applies the rule: UIPinOnlyUI=1). A line whose decided source is eligible (2, 5, 10, 12) shows\n"
        "darkening={unpinned kept colourless applied reference regions mirror}: decision words 62-63 (word 63's\n"
        "colourless bit apart), the CPU reference's counts and its change-set regions that unpin of all\n"
        "(game3d_ui_darkening.h on the source's own artifacts) and the per-pixel mirror of the GPU planes' unpinned pixels\n"
        "against it (match, match(N px near a bound) where a residual, fit, step or tile effect lies at its bound, or\n"
        "differs, which fails the case, as do words 62-63 that are not the planes' sum or are set for another source).\n"
        "With the rule applied, mask_exact and mask_reference zero the CPU reference's unpinned pixels. expect.darkening\n"
        "checks _min/_max bounds of unpinned and kept (words 62-63) and colourless (word 63's top bit).\n"
        "--verbose prints every decision word (62-63 the darkening).\n"
        "S3 (frame identity, shadow): a shader with SUNSHINE_UI_IDENTITY binds the package's stamp buffer read back at t9\n"
        "(replay.ui_detection.stamps: [C_P, C_T] per ticket slot, the layer at 4 and the HUD-less image at 5) and pushes\n"
        "its b2 words 6-9 (replay.ui_detection expected_layer_present, expected_layer_token, expected_hudless_present,\n"
        "identity_bits); a label's \"identity\" object (those names with bits, and layer_present, layer_token and\n"
        "hudless_present for the stamp reads) overrides them. A legacy package proposes nothing and pushes no gate, so\n"
        "it replays as before S3. The mirror also compares texel 12's identity words with ui_selection::verify_identity.\n"
        "--identity adds identity={layer=<verdict>/<space> delta hudless=<verdict>/<space> delta labelled cpu} to every\n"
        "line.\n");
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
      // Fix 4: the darkening against its CPU reference; a pixel that differs
      // away from every bound, or words 62-63 that are not the planes' sum,
      // fail the case.
      std::string darkening;
      if (result.darkening.measured) {
        const auto &k = result.darkening;
        char text[320];
        if (!k.ok && k.differ) {
          std::snprintf(text, sizeof(text), "differs(%llu px, first x=%zu y=%zu; %llu near a bound)",
            static_cast<unsigned long long>(k.differ), k.first % result.width, k.first / result.width,
            static_cast<unsigned long long>(k.at_bound));
        } else if (!k.ok) {
          std::snprintf(text, sizeof(text), "differs(%s)", k.reference.c_str());
        } else if (!k.reference.empty()) {
          std::snprintf(text, sizeof(text), "n/a(%s)", k.reference.c_str());
        } else if (k.at_bound) {
          std::snprintf(text, sizeof(text), "match(%llu px near a bound)", static_cast<unsigned long long>(k.at_bound));
        } else {
          std::snprintf(text, sizeof(text), "match");
        }
        char line[512];
        std::snprintf(line, sizeof(line),
          " darkening={unpinned=%u kept=%u colourless=%d applied=%d reference=%llu/%llu regions=%llu/%llu mirror=%s}",
          d[word::dk_unpinned], d[word::dk_kept] & ~contract::darkening::colourless,
          (d[word::dk_kept] & contract::darkening::colourless) ? 1 : 0, k.applied ? 1 : 0,
          static_cast<unsigned long long>(k.cpu.unpinned), static_cast<unsigned long long>(k.cpu.kept),
          static_cast<unsigned long long>(k.cpu.regions_unpinned), static_cast<unsigned long long>(k.cpu.regions), text);
        darkening = line;
        if (!k.ok) {
          okay = false;
          status = "FAIL";
        }
      } else if (d.size() > word::dk_kept && (d[word::dk_unpinned] || d[word::dk_kept])) {
        // Words 62-63 are zero unless the passes measured an eligible source.
        darkening = " darkening={words 62-63 set without a measured eligible source}";
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
      const auto &change_set_check = target.change_set.empty() ? today.change_set : target.change_set;
      const auto &darkening_check = target.darkening.empty() ? today.darkening : target.darkening;
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
      // selection revision 2 (texels 8 and 9).
      char judgment[200] = "";
      if (d.size() > word::frame_reason) {
        const auto reason = selection::frame_reason_name(d[word::frame_reason]);
        const auto refused = selection::candidate_name(d[word::refused]);
        std::snprintf(judgment, sizeof(judgment), " reason=%.*s%s refused=%.*s one_way={strong=%u/%u/%u contradicted=%u/%u/%u}",
          int(reason.size()), reason.data(), (d[word::frame_reason] & contract::frame_reason_reused) ? "(reused)" : "",
          int(refused.size()), refused.data(), d[word::strong], d[word::strong + 1], d[word::strong + 2], d[word::contradicted],
          d[word::contradicted + 1], d[word::contradicted + 2]);
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
      if (d.size() > word::presented_lit_differs)
        std::snprintf(pre_ui_pixels, sizeof(pre_ui_pixels),
          " pre_ui_pixels={match=%u image_lit=%u presented_lit=%u presented_lit_differs=%u}", d[word::pre_ui_match],
          d[word::pre_ui_image_lit], d[word::presented_lit], d[word::presented_lit_differs]);
      // Fix 3, selection revision 6 (texels 13-15): the change-set shadow as
      // change_set::measure_shadow reads it, and the h1 word's refined bit.
      std::string shadow;
      if (d.size() >= 4u * contract::change_set_decision_texels) {
        namespace change_set = sunshine_game3d::change_set;
        const auto &s = result.shadow;
        const auto ratio = [](double value) {
          char text[16] = "-";
          if (value >= 0.) std::snprintf(text, sizeof(text), "%.3f", value);
          return std::string(text);
        };
        const auto offset = [&](std::size_t i) { return s.measured[i] ? std::to_string(s.offsets[i]) : std::string("-"); };
        shadow = " change_set={class=" + std::string(change_set::name(s.pairing.kind)) + " offset=" +
          std::to_string(s.pairing.offset) + " changed=" + std::to_string(s.counts.changed) + " unchanged=" +
          std::to_string(s.counts.unchanged) + " tiles=" + std::to_string(s.counts.matching_tiles) + " lit=" +
          std::to_string(s.lit) + " filtered=" + std::to_string(s.counts.filtered) + " offsets=" + offset(0) + '/' + offset(1) +
          '/' + offset(2) + " pair=" + std::string(change_set::name(s.verdict)) + " judge=" +
          std::string(change_set::judge_name(s.counts.judge_kind)) + " precision=" + ratio(s.precision) + " recall=" +
          ratio(s.recall) + " iou=" + ratio(s.iou) + " valid=" + (s.valid ? "1" : "0") + " would_refine=" +
          (s.would_refine ? "1" : "0") + " would_source=" + std::to_string(s.would_source) + "} refined=" +
          ((d[word::h1] & contract::h1_refined) ? "1" : "0");
      }
      std::printf("%s %-44s source=%u covered=%u/%u ui=%.2f%% mask=%s (want %s) candidates=0x%x accepted=0x%x "
        "alpha_covered=%u/%u/%u/%u%s%s%s%s%s hudless={changed=%u unchanged=%u invalid=%u tiles=%u lit=%u}%s "
        "mirror=%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s\n",
        status, name.c_str(), d[word::source], d[word::covered], d[word::pixels],
        100.0 * double(result.ui_pixels) / double(result.pixels), mask_class(result).c_str(), wanted.c_str(),
        d[word::candidates], d[word::accepted], d[word::alpha_covered], d[word::alpha_covered + 1], d[word::alpha_covered + 2],
        d[word::alpha_covered + 3], layer, judgment, h1, guard, pre_ui_pixels, d[word::hudless_changed],
        d[word::hudless_unchanged], d[word::hudless_invalid], d[word::matching_tiles], d[word::hudless_lit], shadow.c_str(),
        mirror.c_str(), result.mask_exact.empty() ? "" : " mask_exact=", result.mask_exact.c_str(), scene.empty() ? "" : " scene=",
        scene.c_str(), pre_ui_scene.empty() ? "" : " pre_ui_scene=", pre_ui_scene.c_str(),
        pre_ui_match.empty() ? "" : " pre_ui_match=", pre_ui_match.c_str(), change_set_check.empty() ? "" : " change_set_check=",
        change_set_check.c_str(), result.mask_reference.empty() ? "" : " mask_reference=", result.mask_reference.c_str(),
        darkening.c_str(), darkening_check.empty() ? "" : " darkening_check=", darkening_check.c_str());
      if (identity) std::printf("  %s\n", identity_text(result).c_str() + 1);
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
