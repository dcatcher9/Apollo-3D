// SPDX-License-Identifier: GPL-3.0-only
// CPU-only compile/reflection validation of the native Game 3D shader ABI.
#include "game3d_ui_counters.h"
#include "game3d_ui_detection_contract.h"
#include "game3d_ui_selection.h"

#include <d3d11shader.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {
  void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
  }

  struct binding {
    const char *name;
    D3D_SHADER_INPUT_TYPE type;
    unsigned slot;
  };
  constexpr binding bindings[] = {
    {"SunshineGame3DConstants", D3D_SIT_CBUFFER, 0},
    {"SunshineUIConstants", D3D_SIT_CBUFFER, 1},
    {"SunshineUIDetectionConstants", D3D_SIT_CBUFFER, 2},
    {"SunshineSourceSampler", D3D_SIT_TEXTURE, 0},
    {"DepthBuffer", D3D_SIT_TEXTURE, 1},
    {"SunshineLinearClamp", D3D_SIT_TEXTURE, 2},
    {"SunshineHostCandidateSampler", D3D_SIT_TEXTURE, 3},
    {"SunshineHostVerticalConditionedSampler", D3D_SIT_TEXTURE, 4},
    {"SunshineHostFinalSampler", D3D_SIT_TEXTURE, 5},
    {"SunshineEyeLeftSampler", D3D_SIT_TEXTURE, 6},
    {"SunshinePresentedColor", D3D_SIT_TEXTURE, 6},
    {"SunshineEyeRightSampler", D3D_SIT_TEXTURE, 7},
    {"SunshineUILayer", D3D_SIT_TEXTURE, 7},
    {"SunshineUIPlaneTilesSampler", D3D_SIT_TEXTURE, 8},
    {"SunshineUIPlaneResolvedSampler", D3D_SIT_TEXTURE, 9},
    {"SunshineUIDetectionSampler", D3D_SIT_TEXTURE, 10},
    {"SunshineUIDedicatedAlpha", D3D_SIT_TEXTURE, 11},
    {"SunshineUIColorAlpha", D3D_SIT_TEXTURE, 12},
    {"SunshineUIBackbufferAlpha", D3D_SIT_TEXTURE, 13},
    {"SunshineHUDless", D3D_SIT_TEXTURE, 14},
    {"SunshineHostCandidateStore", D3D_SIT_UAV_RWTYPED, 0},
    {"SunshineHostVerticalMajorantStore", D3D_SIT_UAV_RWTYPED, 1},
    {"SunshineHostVerticalConditionedStore", D3D_SIT_UAV_RWTYPED, 2},
    {"SunshineHostFinalStore", D3D_SIT_UAV_RWTYPED, 3},
    {"SunshineUIPlaneTilesStore", D3D_SIT_UAV_RWTYPED, 4},
    {"SunshineUIPlaneResolvedStore", D3D_SIT_UAV_RWTYPED, 5},
    {"SunshineAlphaCoverageStore", D3D_SIT_UAV_RWTYPED, 6},
    {"SunshineUIConflictStore", D3D_SIT_UAV_RWTYPED, 7},
    {"SunshineUICountersStore", D3D_SIT_UAV_RWTYPED, 7}, // Detection reduce only.
    {"SunshineUIHoldStore", D3D_SIT_UAV_RWTYPED, 5}, // Detection reduce only.
    {"SunshinePointClamp", D3D_SIT_SAMPLER, 0},
    {"SunshineLinearClampState", D3D_SIT_SAMPLER, 1},
    {"SunshinePointBorder", D3D_SIT_SAMPLER, 2},
  };

  struct constant {
    const char *name;
    unsigned offset, size;
    D3D_SHADER_VARIABLE_TYPE type;
  };
  constexpr constant constants[] = {
    {"Depth_Adjustment", 0, 4, D3D_SVT_FLOAT},
    {"Depth_Map_View", 4, 4, D3D_SVT_INT},
    {"Sunshine_DepthReady", 8, 4, D3D_SVT_UINT},
    {"Sunshine_CameraDepthReady", 12, 4, D3D_SVT_UINT},
    {"Sunshine_CameraCoordinateBasis", 16, 4, D3D_SVT_INT},
    {"Sunshine_CameraDepthScale", 20, 4, D3D_SVT_FLOAT},
    {"Sunshine_CameraStrengthBlend", 24, 4, D3D_SVT_FLOAT},
    {"Sunshine_DisparityLimitUv", 28, 4, D3D_SVT_FLOAT},
    {"Sunshine_CameraProjection", 32, 8, D3D_SVT_FLOAT},
    {"Sunshine_CameraRawDepthRange", 40, 8, D3D_SVT_FLOAT},
    {"Sunshine_CameraConvergence", 48, 8, D3D_SVT_FLOAT},
    {"Sunshine_DepthJitter", 56, 8, D3D_SVT_FLOAT},
    {"Sunshine_CameraDepthRect", 64, 16, D3D_SVT_FLOAT},
  };
  constexpr constant ui_constants[] = {
    {"Sunshine_SourceAlphaUI", 0, 4, D3D_SVT_UINT},
    {"Sunshine_UIPlaneMode", 4, 4, D3D_SVT_UINT},
    {"Sunshine_UIPlaneInverseDepth", 8, 4, D3D_SVT_FLOAT},
    {"Sunshine_UIMaskChannel", 12, 4, D3D_SVT_UINT},
  };
  constexpr constant detection_constants[] = {
    {"Sunshine_UICandidates", 0, 4, D3D_SVT_UINT},
    {"Sunshine_UIDifferenceThreshold", 4, 4, D3D_SVT_FLOAT},
    {"Sunshine_UIAcceptedCandidates", 8, 4, D3D_SVT_UINT},
    {"Sunshine_UIDetectionFlags", 12, 4, D3D_SVT_UINT},
    // Selection revision 4: the layer's pair threshold with the presented color.
    {"Sunshine_UIPreUIThreshold", 16, 4, D3D_SVT_FLOAT},
    // Reserved and pushed as zero since selection revision 9 (H2's still-screen
    // flag and fix 3 and fix 4's bits, all removed); S3's words 6-9 are gone.
    {"Sunshine_UIReserved", 20, 4, D3D_SVT_UINT},
  };

  struct entry_point {
    const char *name, *profile;
    unsigned x = 0, y = 0, z = 0;
  };

  std::string disassembly(ID3DBlob *bytecode) {
    ComPtr<ID3DBlob> text;
    require(SUCCEEDED(D3DDisassemble(bytecode->GetBufferPointer(), bytecode->GetBufferSize(), 0, nullptr, &text)),
      "Shader disassembly failed");
    return std::string(static_cast<const char *>(text->GetBufferPointer()), text->GetBufferSize());
  }

  // Group-shared bytes the compiled shader declares, from its disassembly.
  unsigned groupshared_bytes(ID3DBlob *bytecode) {
    const auto listing = disassembly(bytecode);
    unsigned total = 0;
    for (size_t at = listing.find("dcl_tgsm_"); at != std::string::npos; at = listing.find("dcl_tgsm_", at + 1)) {
      unsigned slot = 0, stride = 0, count = 0;
      if (std::sscanf(listing.c_str() + at, "dcl_tgsm_structured g%u, %u, %u", &slot, &stride, &count) == 3) total += stride * count;
      else if (std::sscanf(listing.c_str() + at, "dcl_tgsm_raw g%u, %u", &slot, &stride) == 2) total += stride;
      else throw std::runtime_error("Unrecognized group-shared declaration");
    }
    return total;
  }

  // The pin bound of a texel without slack must stay the UI plane -/+
  // distance * 0.5/W, each one non-precise multiply-add, as in the binary
  // distance rule the soft band replaced: binary masks then keep their fields
  // bit for bit on any device. FXC marks arithmetic precise once a precise
  // result feeds it, and a precise multiply-add rounds twice; reassociating
  // the slack in, or dropping the zero-slack clamp, also changes the
  // rounding. Returns the -/+ pairs found, one per scan direction.
  unsigned pin_bound_pairs(ID3DBlob *bytecode, unsigned width) {
    const auto listing = disassembly(bytecode);
    // FXC prints a literal operand as l(x) with six decimals.
    const auto ramp = [width](const std::string &operand) {
      double value = 0;
      return std::sscanf(operand.c_str(), "l(%lf)", &value) == 1 && std::abs(value - .5 / width) <= 5.01e-7;
    };
    struct multiply_add { std::string factor, addend; };
    std::vector<multiply_add> bounds;
    size_t ramp_uses = 0;
    for (size_t begin = 0, end; begin < listing.size(); begin = end + 1) {
      end = std::min(listing.find('\n', begin), listing.size());
      std::string line = listing.substr(begin, end - begin);
      const auto first = line.find_first_not_of(" \t");
      if (first == std::string::npos || !line.compare(first, 2, "//")) continue;
      auto at = line.find(' ', first);
      if (at == std::string::npos) continue;
      const auto op = line.substr(first, at - first);
      at = line.find_first_not_of(' ', at);
      const bool precise = at != std::string::npos && !line.compare(at, 8, "[precise");
      if (precise) at = line.find(']', at) + 1;
      // Operands, split at top-level commas: l(a, b, c, d) is one operand.
      std::vector<std::string> operands(1);
      for (int depth = 0; at < line.size(); ++at) {
        const char c = line[at];
        depth += c == '(' ? 1 : c == ')' ? -1 : 0;
        if (c == ',' && !depth) operands.emplace_back();
        else if (c != ' ' && c != '\r') operands.back() += c;
      }
      bool uses_ramp = false;
      for (size_t i = 1; i < operands.size(); ++i) uses_ramp = uses_ramp || ramp(operands[i]);
      if (!uses_ramp) continue;
      ++ramp_uses;
      require(!precise, "SunshineApplyUICS computes a pin ramp bound with precise arithmetic: " + line);
      if (op == "mad" && operands.size() == 4 && ramp(operands[2])) bounds.push_back({operands[1], operands[3]});
    }
    unsigned pairs = 0;
    for (const auto &low : bounds)
      for (const auto &high : bounds)
        pairs += low.factor == '-' + high.factor && low.addend == high.addend;
    // Precise arithmetic divides by W instead of multiplying by this literal.
    require(ramp_uses > 0, "SunshineApplyUICS no longer multiplies distances by the 0.5/W pin ramp literal");
    return pairs;
  }

  // Returns how many typed stores to u1 (the vertical majorant) the compiled
  // entry contains.
  unsigned compile(const std::string &source, const std::string &source_name,
      const std::filesystem::path &output, unsigned width, unsigned height,
      unsigned color, const entry_point &entry, std::ostream &manifest) {
    const auto width_text = std::to_string(width);
    const auto height_text = std::to_string(height);
    const auto color_text = std::to_string(color);
    const D3D_SHADER_MACRO macros[] = {
      {"BUFFER_WIDTH", width_text.c_str()}, {"BUFFER_HEIGHT", height_text.c_str()},
      {"BUFFER_COLOR_SPACE", color_text.c_str()}, {nullptr, nullptr},
    };
    ComPtr<ID3DBlob> bytecode, errors;
    const HRESULT result = D3DCompile(source.data(), source.size(), source_name.c_str(),
      macros, nullptr, entry.name, entry.profile,
      D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0,
      &bytecode, &errors);
    if (FAILED(result)) {
      const std::string diagnostic = errors ?
        std::string(static_cast<const char *>(errors->GetBufferPointer()), errors->GetBufferSize()) : "no compiler diagnostic";
      throw std::runtime_error(std::string(entry.name) + ": " + diagnostic);
    }
    ComPtr<ID3D11ShaderReflection> reflection;
    require(SUCCEEDED(D3DReflect(bytecode->GetBufferPointer(), bytecode->GetBufferSize(),
      IID_ID3D11ShaderReflection, reinterpret_cast<void **>(reflection.GetAddressOf()))), "D3DReflect failed");
    D3D11_SHADER_DESC shader {};
    require(SUCCEEDED(reflection->GetDesc(&shader)), "Shader reflection description failed");
    require(((shader.Version >> 4) & 0xfu) == 5 && (shader.Version & 0xfu) == 0,
      "Native shader is not Shader Model 5.0");
    manifest << "entry " << entry.name << ' ' << entry.profile << '\n';
    bool has_ui_binding = false, reads_layer = false, reads_hudless = false;
    for (unsigned index = 0; index < shader.BoundResources; ++index) {
      D3D11_SHADER_INPUT_BIND_DESC actual {};
      require(SUCCEEDED(reflection->GetResourceBindingDesc(index, &actual)), "Binding reflection failed");
      has_ui_binding |= std::string(actual.Name) == "SunshineUIConstants";
      reads_layer |= std::string(actual.Name) == "SunshineUILayer";
      reads_hudless |= std::string(actual.Name) == "SunshineHUDless";
      bool known = false;
      for (const auto &expected : bindings) {
        if (std::string(actual.Name) != expected.name) continue;
        require(actual.Type == expected.type && actual.BindPoint == expected.slot && actual.BindCount == 1,
          std::string("Register ABI mismatch: ") + expected.name);
        known = true;
        break;
      }
      require(known, std::string("Unrecognized native shader resource: ") + actual.Name);
      manifest << "  binding " << actual.Name << ' ' << actual.Type << ' ' << actual.BindPoint << '\n';
    }
    if (std::string(entry.name) == "SunshineApplyUICS" ||
        std::string(entry.name) == "SunshineUINearestTilesCS" || std::string(entry.name) == "SunshineUINearestReduceCS")
      require(has_ui_binding, "UI plane pass lost its independent b1 binding");
    // The scene cells read the pre-UI scene image: the HUD-less image (t14),
    // else the offscreen UI layer's colour (t7), on frames the 256 x 144
    // grid fits (smaller ones compile the evidence away).
    if (std::string(entry.name) == "SunshineSceneCellsCS" && width >= 256 && height >= 144)
      require(reads_layer && reads_hudless, "The scene cells pass must read the pre-UI images at t7 and t14");
    // UI pinning is its own pass after the scene field; the limiter never reads UI state.
    if (std::string(entry.name) == "SunshineHostHorizontalCS")
      require(!has_ui_binding, "Scene limiter must not pin UI");
    for (unsigned index = 0; index < shader.ConstantBuffers; ++index) {
      auto *buffer = reflection->GetConstantBufferByIndex(index);
      D3D11_SHADER_BUFFER_DESC description {};
      require(SUCCEEDED(buffer->GetDesc(&description)), "Constant-buffer reflection failed");
      const bool ui = std::string(description.Name) == "SunshineUIConstants";
      const bool detection = std::string(description.Name) == "SunshineUIDetectionConstants";
      // b2 is ui_detection::b2_words (six since selection revision 9), padded
      // to the 16-byte constant buffer granularity.
      require((ui && description.Size == 16) ||
          (detection && description.Size == (sunshine_game3d::ui_detection::b2_words * 4u + 15u) / 16u * 16u) ||
          (std::string(description.Name) == "SunshineGame3DConstants" && description.Size == 80),
        "Native constants must retain the 80-byte b0, the 16-byte b1 and the six-word (32-byte) b2");
      const constant *begin = detection ? std::begin(detection_constants) : ui ? std::begin(ui_constants) : std::begin(constants);
      const constant *end = detection ? std::end(detection_constants) : ui ? std::end(ui_constants) : std::end(constants);
      for (auto item = begin; item != end; ++item) {
        const auto &expected = *item;
        auto *variable = buffer->GetVariableByName(expected.name);
        D3D11_SHADER_VARIABLE_DESC actual {};
        D3D11_SHADER_TYPE_DESC type {};
        require(SUCCEEDED(variable->GetDesc(&actual)) && SUCCEEDED(variable->GetType()->GetDesc(&type)) &&
          actual.StartOffset == expected.offset && actual.Size == expected.size && type.Type == expected.type,
          std::string("Constant ABI mismatch: ") + expected.name);
      }
      manifest << "  constant_buffer " << description.Name << " bytes " << description.Size << '\n';
    }
    if (entry.x != 0) {
      unsigned x, y, z;
      reflection->GetThreadGroupSize(&x, &y, &z);
      require(x == entry.x && y == entry.y && z == entry.z, "Native compute group shape changed");
      manifest << "  threads " << x << ' ' << y << ' ' << z << '\n';
      // Shader Model 5.0 compute shaders have 32 KiB of group-shared memory.
      const auto shared = groupshared_bytes(bytecode.Get());
      require(shared <= 32768, std::string(entry.name) + " exceeds 32 KiB of group-shared memory");
      manifest << "  groupshared " << shared << '\n';
    }
    // The exact UI counters share u7 with the conflict probe's statistics, and
    // the T1 hold store shares u5 with the resolved UI plane: only the
    // detection reduce binds the counters and the hold store, and no entry
    // binds both resources of one register.
    {
      bool counters = false, conflict = false, hold = false, plane = false;
      for (unsigned index = 0; index < shader.BoundResources; ++index) {
        D3D11_SHADER_INPUT_BIND_DESC actual {};
        require(SUCCEEDED(reflection->GetResourceBindingDesc(index, &actual)), "Binding reflection failed");
        counters |= std::string(actual.Name) == "SunshineUICountersStore";
        conflict |= std::string(actual.Name) == "SunshineUIConflictStore";
        hold |= std::string(actual.Name) == "SunshineUIHoldStore";
        plane |= std::string(actual.Name) == "SunshineUIPlaneResolvedStore";
      }
      const bool reduce = std::string(entry.name) == "SunshineUIDetectionReduceCS";
      require(counters == reduce && !(counters && conflict),
        std::string(entry.name) + ": only the detection reduce may add to the UI counters at u7");
      require(hold == reduce && !(hold && plane),
        std::string(entry.name) + ": only the detection reduce may bind the UI hold store at u5");
    }
    if (std::string(entry.name) == "SunshineApplyUICS") {
      const auto pairs = pin_bound_pairs(bytecode.Get(), width);
      require(pairs >= 2, "SunshineApplyUICS lost the binary pin bound's multiply-adds in a scan direction");
      manifest << "  pin_bound_multiply_add_pairs " << pairs << '\n';
    }
    if (std::string(entry.name) == "SunshineRenderPackedPS") {
      require(shader.OutputParameters == 1, "Packed eye pass must write exactly the side-by-side target");
      D3D11_SIGNATURE_PARAMETER_DESC parameter {};
      require(SUCCEEDED(reflection->GetOutputParameterDesc(0, &parameter)) &&
        parameter.SystemValueType == D3D_NAME_TARGET && parameter.SemanticIndex == 0 && parameter.Mask == 15,
        "Packed eye pass target signature changed");
    }
    std::ofstream binary(output / (std::string(entry.name) + ".cso"), std::ios::binary);
    binary.write(static_cast<const char *>(bytecode->GetBufferPointer()), bytecode->GetBufferSize());
    require(binary.good(), "Could not write compiled shader evidence");
    const auto listing = disassembly(bytecode.Get());
    unsigned majorant_stores = 0;
    for (size_t at = listing.find("store_uav_typed u1."); at != std::string::npos; at = listing.find("store_uav_typed u1.", at + 1))
      ++majorant_stores;
    return majorant_stores;
  }
}

int main(int argc, char **argv) {
  try {
    require(argc == 3, "Usage: test_game3d_native_shader source.hlsl fresh-output-directory");
    const std::filesystem::path source_path = std::filesystem::absolute(argv[1]);
    const std::filesystem::path output = std::filesystem::absolute(argv[2]);
    require(!std::filesystem::exists(output), "Use a fresh output directory");
    std::ifstream input(source_path, std::ios::binary);
    require(input.good(), "Cannot read native shader source");
    const std::string source((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    std::filesystem::create_directories(output);
    // The renderer sizes limiter and UI pinning dispatches from these markers.
    const auto marker = [&source](const std::string &name) {
      const auto key = "#define " + name + ' ';
      const auto at = source.find(key);
      require(at != std::string::npos, "Missing shader marker " + name);
      return unsigned(std::stoul(source.substr(at + key.size())));
    };
    const unsigned limiter_lines = marker("SUNSHINE_LIMITER_LINE_GROUPS"), pin_lines = marker("SUNSHINE_UI_PIN_LINE_GROUPS");
    // Dump 3D records the soft pin gain. Every automatic mask, the late
    // offscreen layer's included, is its source's raw alpha: no margin marker.
    require(marker("SUNSHINE_UI_SOFT_PIN_GAIN") >= 1, "UI soft pin gain must be at least one");
    require(source.find("SUNSHINE_UI_LATE_MARGIN") == std::string::npos, "UI late-layer margin must stay removed");
    // UI detection sizes its decision texels and statistics rows from these
    // markers, and its flag bits mirror game3d_ui_detection_contract.h.
    namespace detection = sunshine_game3d::ui_detection;
    const unsigned decision_texels = marker(std::string(detection::decision_texels_marker)),
      evidence_images = marker(std::string(detection::scene_evidence_images_marker));
    require(decision_texels >= detection::min_decision_texels && decision_texels <= detection::max_decision_texels &&
        evidence_images <= detection::max_scene_evidence_images, "UI detection size markers out of range");
    const auto mirrored = [&source](const auto &defines) {
      for (const auto &[name, value] : defines) {
        const auto key = "#define " + std::string(name) + ' ';
        const auto at = source.find(key);
        require(at != std::string::npos, "Missing UI detection define " + std::string(name));
        require(std::stoul(source.substr(at + key.size(), 16), nullptr, 0) == value,
          std::string(name) + " differs from game3d_ui_detection_contract.h");
      }
    };
    mirrored(detection::hlsl_flag_defines);
    mirrored(detection::hlsl_scene_defines);
    mirrored(sunshine_game3d::hlsl_counter_defines);
    mirrored(detection::hlsl_candidate_defines);
    mirrored(detection::hlsl_hold_defines);
    mirrored(detection::hlsl_h1_defines);
    // The tiles pass counts each statistics tile in this many parts; the
    // renderer and the replay size the statistics texture from it.
    mirrored(detection::hlsl_tile_defines);
    require(detection::tile_parts >= 1 && detection::tile_parts <= detection::max_tile_parts,
      "UI detection tile parts out of range");
    // The renderer may skip conditioning for a pack that shows the source mono.
    require(marker("SUNSHINE_MONO_SKIPS_CONDITIONING") == 1, "The mono pack must keep reading no conditioning");
    // The renderer packs native scRGB as PQ and runs the live vertical pass
    // only when the shader declares them.
    require(marker("SUNSHINE_SCRGB_PQ_PACK") == 1 && marker("SUNSHINE_VERTICAL_LIVE_ENTRY") == 1,
      "The scRGB PQ pack and the live vertical pass must stay declared");
    // The reduce ports ui_selection::decide of this revision.
    require(marker(std::string(sunshine_game3d::ui_selection::revision_marker)) == sunshine_game3d::ui_selection::revision &&
        sunshine_game3d::ui_selection::revision == 10u,
      "SUNSHINE_UI_SELECTION_REVISION differs from ui_selection::revision 10");
    require(decision_texels == detection::decision_texels && decision_texels == 17u,
      "Selection revision 10 (texels 12-15 reserved, the declared alphas' one-way counts in texel 16) writes 17 decision texels");
    // Hidden-scene evidence writes decision texels 5 and 6 from cells of both images.
    require(evidence_images == detection::max_scene_evidence_images && decision_texels >= detection::scene_decision_texels,
      "The native shader lost its hidden-scene evidence markers");
    unsigned compiled = 0;
    for (const auto [width, height] : std::array<std::array<unsigned, 2>, 6> {{
      {16, 8}, {64, 36}, {1920, 1080}, {3840, 2160}, {2160, 3840}, {4800, 2700},
    }}) {
      for (unsigned color = 1; color <= 3; ++color) {
        const auto name = std::to_string(width) + 'x' + std::to_string(height) + "-color" + std::to_string(color);
        const auto directory = output / name;
        std::filesystem::create_directories(directory);
        std::ofstream manifest(directory / "bindings.txt");
        std::vector<entry_point> entries {
          {"PostProcessVS", "vs_5_0"}, {"SunshineRenderPackedPS", "ps_5_0"},
        };
        // The 10-bit PQ export: HDR10, and native scRGB for an HDR10 stream.
        if (color >= 2) entries.push_back({"SunshineRenderPackedPQPS", "ps_5_0"});
        if (width <= 3840 && height <= 3840) {
          entries.push_back({"SunshineHostCandidateCS", "cs_5_0", 8, 8, 1});
          // Limiter groups: the marked number of adjacent columns (rows) by eight chunks.
          entries.push_back({"SunshineHostVerticalCS", "cs_5_0", limiter_lines, 8, 1});
          entries.push_back({"SunshineHostVerticalLiveCS", "cs_5_0", limiter_lines, 8, 1});
          entries.push_back({"SunshineHostHorizontalCS", "cs_5_0", limiter_lines, 8, 1});
          entries.push_back({"SunshineUINearestTilesCS", "cs_5_0", 16, 16, 1});
          entries.push_back({"SunshineUINearestReduceCS", "cs_5_0", 256, 1, 1});
          entries.push_back({"SunshineUIConflictCS", "cs_5_0", 8, 8, 1});
          entries.push_back({"SunshineApplyUICS", "cs_5_0", pin_lines, 8, 1});
          entries.push_back({"SunshineUIDetectionTilesCS", "cs_5_0", 16, 16, 1});
          entries.push_back({"SunshineUIDetectionReduceCS", "cs_5_0", 256, 1, 1});
          entries.push_back({"SunshineUIDetectionMaskCS", "cs_5_0", 8, 8, 1});
          entries.push_back({"SunshineSceneCellsCS", "cs_5_0", 16, 16, 1});
          entries.push_back({"SunshineSceneCompareCS", "cs_5_0", 16, 16, 1});
          entries.push_back({"SunshineSceneEvidenceCS", "cs_5_0", 16, 16, 1});
        }
        unsigned full_majorant = 0, live_majorant = 0;
        for (const auto &entry : entries) {
          const auto stores = compile(source, source_path.string(), directory, width, height, color, entry, manifest);
          if (std::string(entry.name) == "SunshineHostVerticalCS") full_majorant = stores;
          if (std::string(entry.name) == "SunshineHostVerticalLiveCS") live_majorant = stores;
          ++compiled;
        }
        // The live vertical pass leaves out only the final majorant write-back
        // (frames of at most 32 rows keep the serial diagnostic path in both).
        if (width <= 3840 && height <= 3840 && height > 32)
          require(live_majorant != 0 && live_majorant < full_majorant,
            name + ": the live vertical pass must store fewer majorant texels than the full pass");
        std::cout << "PASS " << name << ": " << entries.size() << " entries, explicit registers and 80-byte constants\n";
      }
    }
    std::cout << "PASS native Game 3D: " << compiled << " compiled/reflected entries across 18 configurations; no GPU execution.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
