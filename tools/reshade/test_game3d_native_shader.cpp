// SPDX-License-Identifier: GPL-3.0-only
// CPU-only compile/reflection validation of the native Game 3D shader ABI.
#include <d3d11shader.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <array>
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
    {"SunshineSourceSampler", D3D_SIT_TEXTURE, 0},
    {"DepthBuffer", D3D_SIT_TEXTURE, 1},
    {"SunshineLinearClamp", D3D_SIT_TEXTURE, 2},
    {"SunshineHostCandidateSampler", D3D_SIT_TEXTURE, 3},
    {"SunshineHostVerticalConditionedSampler", D3D_SIT_TEXTURE, 4},
    {"SunshineHostFinalSampler", D3D_SIT_TEXTURE, 5},
    {"SunshineEyeLeftSampler", D3D_SIT_TEXTURE, 6},
    {"SunshineEyeRightSampler", D3D_SIT_TEXTURE, 7},
    {"SunshineHostCandidateStore", D3D_SIT_UAV_RWTYPED, 0},
    {"SunshineHostVerticalMajorantStore", D3D_SIT_UAV_RWTYPED, 1},
    {"SunshineHostVerticalConditionedStore", D3D_SIT_UAV_RWTYPED, 2},
    {"SunshineHostFinalStore", D3D_SIT_UAV_RWTYPED, 3},
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
    {"Sunshine_Reserved", 28, 4, D3D_SVT_FLOAT},
    {"Sunshine_CameraProjection", 32, 8, D3D_SVT_FLOAT},
    {"Sunshine_CameraRawDepthRange", 40, 8, D3D_SVT_FLOAT},
    {"Sunshine_CameraConvergence", 48, 8, D3D_SVT_FLOAT},
    {"Sunshine_DepthJitter", 56, 8, D3D_SVT_FLOAT},
    {"Sunshine_CameraDepthRect", 64, 16, D3D_SVT_FLOAT},
  };

  struct entry_point {
    const char *name, *profile;
    unsigned x = 0, y = 0, z = 0;
  };

  void compile(const std::string &source, const std::string &source_name,
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
    for (unsigned index = 0; index < shader.BoundResources; ++index) {
      D3D11_SHADER_INPUT_BIND_DESC actual {};
      require(SUCCEEDED(reflection->GetResourceBindingDesc(index, &actual)), "Binding reflection failed");
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
    for (unsigned index = 0; index < shader.ConstantBuffers; ++index) {
      auto *buffer = reflection->GetConstantBufferByIndex(index);
      D3D11_SHADER_BUFFER_DESC description {};
      require(SUCCEEDED(buffer->GetDesc(&description)), "Constant-buffer reflection failed");
      require(std::string(description.Name) == "SunshineGame3DConstants" && description.Size == 80,
        "Native constants must remain an 80-byte b0");
      for (const auto &expected : constants) {
        auto *variable = buffer->GetVariableByName(expected.name);
        D3D11_SHADER_VARIABLE_DESC actual {};
        D3D11_SHADER_TYPE_DESC type {};
        require(SUCCEEDED(variable->GetDesc(&actual)) && SUCCEEDED(variable->GetType()->GetDesc(&type)) &&
          actual.StartOffset == expected.offset && actual.Size == expected.size && type.Type == expected.type,
          std::string("Constant ABI mismatch: ") + expected.name);
      }
      manifest << "  constant_buffer_bytes " << description.Size << '\n';
    }
    if (entry.x != 0) {
      unsigned x, y, z;
      reflection->GetThreadGroupSize(&x, &y, &z);
      require(x == entry.x && y == entry.y && z == entry.z, "Native compute group shape changed");
      manifest << "  threads " << x << ' ' << y << ' ' << z << '\n';
    }
    if (std::string(entry.name) == "SunshineRenderEyesPS") {
      require(shader.OutputParameters == 2, "Eye pass must retain its two render targets");
      for (unsigned index = 0; index < shader.OutputParameters; ++index) {
        D3D11_SIGNATURE_PARAMETER_DESC parameter {};
        require(SUCCEEDED(reflection->GetOutputParameterDesc(index, &parameter)) &&
          parameter.SystemValueType == D3D_NAME_TARGET && parameter.SemanticIndex == index && parameter.Mask == 15,
          "Eye pass MRT signature changed");
      }
    }
    std::ofstream binary(output / (std::string(entry.name) + ".cso"), std::ios::binary);
    binary.write(static_cast<const char *>(bytecode->GetBufferPointer()), bytecode->GetBufferSize());
    require(binary.good(), "Could not write compiled shader evidence");
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
          {"PostProcessVS", "vs_5_0"}, {"SunshineRenderEyesPS", "ps_5_0"}, {"SunshinePackEyesPS", "ps_5_0"},
        };
        if (color == 3) entries.push_back({"SunshinePreparePQPS", "ps_5_0"});
        if (width <= 3840 && height <= 3840) {
          entries.push_back({"SunshineHostCandidateCS", "cs_5_0", 8, 8, 1});
          entries.push_back({"SunshineHostVerticalCS", "cs_5_0", 32, 1, 1});
          entries.push_back({"SunshineHostHorizontalCS", "cs_5_0", 32, 1, 1});
        }
        for (const auto &entry : entries) {
          compile(source, source_path.string(), directory, width, height, color, entry, manifest);
          ++compiled;
        }
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
