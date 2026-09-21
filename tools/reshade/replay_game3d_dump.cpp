// SPDX-License-Identifier: GPL-3.0-only
// A frozen, single-frame replay of the PRODUCTION add-on renderer. No effects,
// estimator, alternative warp, host scaling, cursor, overlay or encoder runs.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "game3d_renderer.h"
#include "replay_game3d_package.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <d3d11.h>
#include <dxgi1_4.h>
#include <filesystem>
#include <fstream>
#include <optional>
#include <reshade.hpp>
#include <vector>
#include <wrl/client.h>

namespace {
  namespace api = reshade::api;
  namespace fs = std::filesystem;
  namespace replay = sunshine_game3d::replay;
  using Microsoft::WRL::ComPtr;
  using replay::require;
  using json = nlohmann::json;

  void checked(HRESULT result, const char *message) {
    if (FAILED(result)) {
      char error[320];
      std::snprintf(error, sizeof(error), "%s (0x%08lx)", message, static_cast<unsigned long>(result));
      throw std::runtime_error(error);
    }
  }

  std::vector<std::uint8_t> read_file(const fs::path &file, std::uint64_t maximum, std::optional<std::uint64_t> exact = {}) {
    require(fs::is_regular_file(file), "Missing file: " + file.string());
    const auto size = fs::file_size(file);
    require(size <= maximum && (!exact || size == *exact), "Unexpected file size: " + file.string());
    std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
    std::ifstream stream(file, std::ios::binary);
    require(bool(stream), "Cannot open file: " + file.string());
    if (size) {
      stream.read(reinterpret_cast<char *>(result.data()), std::streamsize(size));
    }
    require(bool(stream) && stream.peek() == std::char_traits<char>::eof(), "File changed or could not be read: " + file.string());
    return result;
  }

  std::vector<std::uint8_t> read_artifact(const fs::path &directory, const replay::artifact &a) {
    const auto path = directory / a.file;
    const auto attributes = GetFileAttributesW(path.c_str());
    require(attributes != INVALID_FILE_ATTRIBUTES && !(attributes & FILE_ATTRIBUTE_REPARSE_POINT) && fs::canonical(path).parent_path() == directory, "Artifact must be a regular file inside the dump: " + a.file);
    return read_file(path, replay::max_package_bytes, a.byte_count);
  }

  void write_file(const fs::path &path, const void *data, std::size_t size) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    require(bool(stream), "Cannot create output: " + path.string());
    stream.write(static_cast<const char *>(data), std::streamsize(size));
    stream.close();
    require(bool(stream), "Cannot write output: " + path.string());
  }

  void write_json(const fs::path &path, const json &value) {
    const auto text = value.dump(2);
    write_file(path, text.data(), text.size());
  }

  bool inside(const fs::path &directory, const fs::path &path) {
    // Compare normalized Windows paths case-insensitively, including junctions.
    auto base = directory.wstring(), candidate = path.wstring();
    if (candidate.size() < base.size() || _wcsnicmp(base.c_str(), candidate.c_str(), base.size()) != 0) {
      return false;
    }
    return candidate.size() == base.size() || base.back() == L'\\' || base.back() == L'/' ||
           candidate[base.size()] == L'\\' || candidate[base.size()] == L'/';
  }

  api::effect_runtime *runtime_seen = nullptr;
  unsigned reloads = 0;

  void on_init(api::effect_runtime *runtime) {
    runtime_seen = runtime;
  }

  void on_reload(api::effect_runtime *) {
    ++reloads;
  }

  struct runtime_fixture {
    HMODULE module = nullptr;
    HWND window = nullptr;
    bool registered = false;
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<IDXGISwapChain> swapchain;
    ComPtr<ID3D11Texture2D> backbuffer;

    ~runtime_fixture() {
      if (registered) {
        reshade::unregister_event<reshade::addon_event::init_effect_runtime>(on_init);
        reshade::unregister_event<reshade::addon_event::reshade_reloaded_effects>(on_reload);
        reshade::unregister_addon(GetModuleHandleW(nullptr), module);
      }
      backbuffer.Reset();
      swapchain.Reset();
      context.Reset();
      device.Reset();
      runtime_seen = nullptr;
      if (window) {
        DestroyWindow(window);
      }
      // ReShade installs process-wide hooks. As in the official-runtime tests,
      // leave the module loaded until process exit rather than unsafe FreeLibrary.
    }

    void initialize(const fs::path &dll, const fs::path &directory, const replay::package &p) {
      fs::create_directories(directory / "effects");
      fs::create_directories(directory / "addons");
      fs::copy_file(dll, directory / "dxgi.dll");
      const std::string config = "[ADDON]\nAddonPath=.\\addons\n[GENERAL]\nEffectSearchPaths=.\\effects\nTextureSearchPaths=.\\effects\nPresetPath=.\\preset.ini\nEffectCachePath=.\\cache\nPerformanceMode=0\nSkipLoadingDisabledEffects=0\n[OVERLAY]\nTutorialProgress=4\nShowFPS=0\nShowClock=0\nShowPresetName=0\n";
      write_file(directory / "ReShade.ini", config.data(), config.size());
      write_file(directory / "preset.ini", "Techniques=\n", 12);
      require(SetEnvironmentVariableW(L"RESHADE_BASE_PATH_OVERRIDE", directory.c_str()), "Cannot isolate ReShade configuration");
      wchar_t system[MAX_PATH] {};
      require(GetSystemDirectoryW(system, MAX_PATH), "Cannot find Windows system directory");
      require(LoadLibraryW((fs::path(system) / "d3d11.dll").c_str()) != nullptr, "Cannot preload native D3D11");
      module = LoadLibraryW((directory / "dxgi.dll").c_str());
      require(module != nullptr, "Cannot load supplied official ReShade64.dll");
      registered = reshade::register_addon(GetModuleHandleW(nullptr), module);
      require(registered, "Official ReShade runtime must support the add-on API");
      reshade::register_event<reshade::addon_event::init_effect_runtime>(on_init);
      reshade::register_event<reshade::addon_event::reshade_reloaded_effects>(on_reload);
      WNDCLASSW wc {};
      wc.lpfnWndProc = DefWindowProcW;
      wc.hInstance = GetModuleHandleW(nullptr);
      wc.lpszClassName = L"SunshineGame3DReplay";
      require(RegisterClassW(&wc), "Cannot register replay window");
      window = CreateWindowW(wc.lpszClassName, L"Hidden Game 3D replay", WS_OVERLAPPEDWINDOW, 0, 0, int(p.width), int(p.height), nullptr, nullptr, wc.hInstance, nullptr);
      require(window != nullptr, "Cannot create hidden replay window");
      DXGI_SWAP_CHAIN_DESC desc {};
      desc.BufferDesc.Width = p.width;
      desc.BufferDesc.Height = p.height;
      desc.BufferDesc.Format = static_cast<DXGI_FORMAT>(p.artifacts.at("source_color").format);
      desc.SampleDesc.Count = 1;
      desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
      desc.BufferCount = 2;
      desc.OutputWindow = window;
      desc.Windowed = TRUE;
      desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
      const auto create = reinterpret_cast<decltype(&D3D11CreateDeviceAndSwapChain)>(GetProcAddress(module, "D3D11CreateDeviceAndSwapChain"));
      require(create != nullptr, "Supplied runtime lacks the official D3D11 proxy export");
      checked(create(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &desc, &swapchain, &device, nullptr, &context), "Create wrapped D3D11 swapchain");
      ComPtr<IDXGISwapChain3> chain3;
      checked(swapchain.As(&chain3), "Query swapchain color interface");
      checked(chain3->SetColorSpace1(p.color == 2 ? DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709 : p.color == 3 ? DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 :
                                                                                                             DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709),
              "Set captured source color space");
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
      do {
        MSG message;
        while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
          TranslateMessage(&message);
          DispatchMessageW(&message);
        }
        checked(swapchain->Present(0, 0), "Initialize hidden official runtime");
        if (runtime_seen && reloads) {
          break;
        }
        Sleep(5);
      } while (std::chrono::steady_clock::now() < deadline);
      require(runtime_seen && reloads && runtime_seen->get_device()->get_api() == api::device_api::d3d11, "Official D3D11 runtime did not initialize; inspect output/ReShade.log");
      unsigned techniques = 0;
      runtime_seen->enumerate_techniques(nullptr, [&](api::effect_runtime *, api::effect_technique) {
        ++techniques;
      });
      require(techniques == 0, "Replay unexpectedly loaded an FX effect");
      checked(swapchain->GetBuffer(0, IID_PPV_ARGS(&backbuffer)), "Get replay backbuffer");
    }

    ComPtr<ID3D11Texture2D> upload(const replay::artifact &a, const std::vector<std::uint8_t> &bytes) {
      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = a.width;
      desc.Height = a.height;
      desc.ArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
      desc.Format = static_cast<DXGI_FORMAT>(a.format);
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA data {bytes.data(), a.row_bytes, 0};
      ComPtr<ID3D11Texture2D> result;
      checked(device->CreateTexture2D(&desc, &data, &result), "Upload captured native pixels");
      return result;
    }

    std::vector<std::uint8_t> read(ID3D11Texture2D *texture, const replay::artifact &expected) {
      D3D11_TEXTURE2D_DESC desc {};
      texture->GetDesc(&desc);
      require(desc.Width == expected.width && desc.Height == expected.height && desc.Format == static_cast<DXGI_FORMAT>(expected.format) && desc.MipLevels == 1 && desc.ArraySize == 1 && desc.SampleDesc.Count == 1, "Replayed artifact descriptor differs from capture: " + expected.kind);
      desc.Usage = D3D11_USAGE_STAGING;
      desc.BindFlags = desc.MiscFlags = 0;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Texture2D> staging;
      checked(device->CreateTexture2D(&desc, nullptr, &staging), "Create replay readback");
      context->CopyResource(staging.Get(), texture);
      D3D11_QUERY_DESC q {D3D11_QUERY_EVENT, 0};
      ComPtr<ID3D11Query> query;
      checked(device->CreateQuery(&q, &query), "Create readback fence");
      context->End(query.Get());
      context->Flush();
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
      HRESULT ready;
      while ((ready = context->GetData(query.Get(), nullptr, 0, D3D11_ASYNC_GETDATA_DONOTFLUSH)) == S_FALSE && std::chrono::steady_clock::now() < deadline) {
        Sleep(1);
      }
      require(ready == S_OK, "Replay GPU readback failed or timed out");
      std::vector<std::uint8_t> result(static_cast<std::size_t>(expected.byte_count));
      D3D11_MAPPED_SUBRESOURCE mapped {};
      checked(context->Map(staging.Get(), 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped), "Map completed replay pixels");
      if (!mapped.pData || mapped.RowPitch < expected.row_bytes) {
        context->Unmap(staging.Get(), 0);
        throw std::runtime_error("Invalid readback pitch");
      }
      for (unsigned y = 0; y < expected.height; ++y) {
        std::memcpy(result.data() + std::size_t(y) * expected.row_bytes, static_cast<const std::uint8_t *>(mapped.pData) + std::size_t(y) * mapped.RowPitch, expected.row_bytes);
      }
      context->Unmap(staging.Get(), 0);
      return result;
    }
  };

  double half_float(unsigned value) {
    const double sign = (value & 0x8000) ? -1 : 1;
    const unsigned exponent = (value >> 10) & 31, fraction = value & 1023;
    if (!exponent) {
      return sign * std::ldexp(double(fraction), -24);
    }
    if (exponent == 31) {
      return fraction ? NAN : sign * INFINITY;
    }
    return sign * std::ldexp(1. + double(fraction) / 1024., int(exponent) - 15);
  }

  double component(const std::vector<std::uint8_t> &bytes, unsigned format, std::size_t index) {
    if (format == DXGI_FORMAT_R32_FLOAT) {
      float f;
      std::memcpy(&f, bytes.data() + index * 4, 4);
      return f;
    }
    if (format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
      std::uint16_t h;
      std::memcpy(&h, bytes.data() + index * 2, 2);
      return half_float(h);
    }
    if (format == DXGI_FORMAT_R10G10B10A2_UNORM) {
      std::uint32_t v;
      std::memcpy(&v, bytes.data() + (index / 4) * 4, 4);
      return index % 4 == 3 ? double(v >> 30) / 3. : double((v >> (10 * (index % 4))) & 1023) / 1023.;
    }
    return double(bytes[index]) / 255.;  // Native BGRA channel order stays explicit in DXGI format.
  }

  json compare(const replay::artifact &a, const std::vector<std::uint8_t> &captured, const std::vector<std::uint8_t> &actual) {
    require(captured.size() == actual.size(), "Comparison size mismatch");
    std::uint64_t different = 0, nonfinite = 0, above = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
      different += captured[i] != actual[i];
    }
    const auto channels = a.format == DXGI_FORMAT_R32_FLOAT ? 1u : 4u;
    const auto count = std::size_t(a.width) * a.height * channels;
    double max_abs = 0, max_relative = 0;
    long double squared = 0;
    for (std::size_t i = 0; i < count; ++i) {
      const auto x = component(captured, a.format, i), y = component(actual, a.format, i);
      if (!std::isfinite(x) || !std::isfinite(y)) {
        ++nonfinite;
        ++above;
        continue;
      }
      const auto delta = std::abs(x - y), magnitude = std::max(std::abs(x), std::abs(y));
      max_abs = std::max(max_abs, delta);
      max_relative = std::max(max_relative, delta / std::max(1., magnitude));
      squared += delta * delta;
      // Verification tolerates one output quantization step; float32 fields get
      // 1e-7 absolute + 1e-5 relative. The exact byte verdict remains separate.
      double tolerance = 0;
      if (a.format == DXGI_FORMAT_R32_FLOAT) {
        tolerance = 1.e-7 + magnitude * 1.e-5;
      } else if (a.format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
        int e = 0;
        std::frexp(magnitude, &e);
        tolerance = std::ldexp(1., std::max(-24, e - 11));
      } else if (a.format == DXGI_FORMAT_R10G10B10A2_UNORM) {
        tolerance = i % 4 == 3 ? 1. / 3. : 1. / 1023.;
      } else {
        tolerance = 1. / 255.;
      }
      if (delta > tolerance * 1.000001) {
        ++above;
      }
    }
    const bool source = a.kind == "source_color" || a.kind == "ui_source_color";
    return {{"byte_exact", different == 0}, {"different_bytes", different}, {"sample_count", count}, {"nonfinite_samples", nonfinite}, {"max_absolute_error", max_abs}, {"max_relative_error", max_relative}, {"rms_error", std::sqrt(double(squared / std::max<std::size_t>(1, count - nonfinite)))}, {"samples_above_tolerance", above}, {"within_tolerance", above == 0 && (!source || different == 0)}, {"verification_contract", source ? "Native source copy must be byte exact" : "Native output quantization tolerance"}};
  }

  api::resource diagnostic(const sunshine_game3d::diagnostic_resources &d, const std::string &kind) {
    if (kind == "ui_source_color") return d.ui_source;
    if (kind == "source_color") {
      return d.source;
    }
    if (kind == "linear_color") {
      return d.linear_color;
    }
    if (kind == "candidate") {
      return d.candidate;
    }
    if (kind == "vertical_majorant") {
      return d.vertical_majorant;
    }
    if (kind == "vertical_field") {
      return d.vertical_field;
    }
    if (kind == "final_field") {
      return d.final_field;
    }
    if (kind == "sbs") {
      return d.sbs;
    }
    return {};
  }
}  // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: replay_game3d_dump <dump-dir> <official-ReShade64.dll> <fresh-output-dir> [--shader file.hlsl] [--strength 0..100] [--zero-inverse-depth nonnegative-float] [--ui-inverse-depth nonnegative-float | --ui-nearest-floor nonnegative-float | --ui-plane screen|front-limit] [--depth-gain positive-float] [--disparity-limit-uv (0,.04]] [--source-alpha-ui on|off] [--verify]\n");
    return 2;
  }
  fs::path output;
  bool created = false;
  try {
    std::optional<float> strength;
    std::optional<float> zero_inverse_depth;
    std::optional<float> depth_gain;
    std::optional<float> disparity_limit_uv;
    std::optional<bool> source_alpha_ui;
    std::optional<sunshine_game3d::ui_plane_parameters> ui_plane_override;
    std::string ui_plane_argument;
    std::string zero_inverse_depth_argument;
    fs::path shader_override;
    bool verify = false;
    for (int i = 4; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--verify") {
        require(!verify, "Duplicate --verify");
        verify = true;
      } else if (option == "--source-alpha-ui") {
        require(!source_alpha_ui.has_value() && i + 1 < argc, "Expected one source-alpha UI selection");
        const std::string value = argv[++i];
        require(value == "on" || value == "off", "Source-alpha UI must be on or off");
        source_alpha_ui = value == "on";
      } else if (option == "--shader") {
        require(shader_override.empty() && i + 1 < argc, "Expected one shader filename");
        shader_override = fs::canonical(fs::u8path(argv[++i]));
      } else if (option == "--strength") {
        require(!strength && i + 1 < argc, "Expected one strength value");
        std::size_t end = 0;
        const std::string value = argv[++i];
        strength = std::stof(value, &end);
        require(end == value.size() && std::isfinite(*strength) && *strength >= 0 && *strength <= 100, "Strength must be finite and within 0..100");
      } else if (option == "--zero-inverse-depth") {
        require(!zero_inverse_depth && i + 1 < argc, "Expected one zero inverse-depth value");
        std::size_t end = 0;
        zero_inverse_depth_argument = argv[++i];
        zero_inverse_depth = std::stof(zero_inverse_depth_argument, &end);
        require(end == zero_inverse_depth_argument.size() && *zero_inverse_depth >= 0.f && std::isfinite(*zero_inverse_depth),
          "Zero inverse depth must be finite and nonnegative");
      } else if (option == "--ui-inverse-depth" || option == "--ui-nearest-floor") {
        require(!ui_plane_override && i + 1 < argc, "Expected one independent UI plane override");
        std::size_t end = 0;
        ui_plane_argument = argv[++i];
        const float inverse = std::stof(ui_plane_argument, &end);
        require(end == ui_plane_argument.size() && inverse >= 0.f && std::isfinite(inverse),
          "UI inverse depth must be finite and nonnegative");
        ui_plane_override = sunshine_game3d::ui_plane_parameters {
          option == "--ui-nearest-floor" ? sunshine_game3d::ui_plane_mode::depth_midpoint_nearest_ui :
            sunshine_game3d::ui_plane_mode::depth_midpoint, inverse};
      } else if (option == "--ui-plane") {
        require(!ui_plane_override && i + 1 < argc, "Expected one independent UI plane override");
        ui_plane_argument = argv[++i];
        require(ui_plane_argument == "screen" || ui_plane_argument == "front-limit", "UI plane must be screen or front-limit; use --ui-inverse-depth to supply an independent depth");
        ui_plane_override = sunshine_game3d::ui_plane_parameters {
          ui_plane_argument == "front-limit" ? sunshine_game3d::ui_plane_mode::front_limit : sunshine_game3d::ui_plane_mode::screen, 0.f};
      } else if (option == "--depth-gain") {
        require(!depth_gain && i + 1 < argc, "Expected one depth gain");
        std::size_t end = 0;
        const std::string value = argv[++i];
        depth_gain = std::stof(value, &end);
        require(end == value.size() && *depth_gain > 0.f && std::isnormal(*depth_gain), "Depth gain must be positive, finite and normal");
      } else if (option == "--disparity-limit-uv") {
        require(!disparity_limit_uv && i + 1 < argc, "Expected one disparity limit");
        std::size_t end = 0;
        const std::string value = argv[++i];
        disparity_limit_uv = std::stof(value, &end);
        require(end == value.size() && std::isfinite(*disparity_limit_uv) && *disparity_limit_uv > 0.f && *disparity_limit_uv <= .04f,
          "Disparity limit must be finite and within (0,.04] of per-eye width");
      } else {
        throw std::runtime_error("Unknown option: " + option);
      }
    }
    const auto directory = fs::canonical(fs::u8path(argv[1]));
    const auto dll = fs::canonical(fs::u8path(argv[2]));
    require(fs::is_directory(directory) && fs::is_regular_file(dll), "Expected dump directory and official runtime file");
    output = fs::weakly_canonical(fs::absolute(fs::u8path(argv[3])));
    require(!fs::exists(output) && !inside(directory, output), "Output directory must be new and outside the captured dump");
    const auto manifest_bytes = read_file(directory / "manifest.json", 512 * 1024);
    const auto manifest = json::parse(manifest_bytes.begin(), manifest_bytes.end());
    const auto package = replay::parse(manifest);
    // Validate every artifact before loading a DLL or issuing GPU work. Each is
    // re-read just before comparison, preserving a bounded CPU working set.
    for (const auto &[kind, a] : package.artifacts) {
      (void) read_artifact(directory, a);
    }
    std::string shader = package.shader;
    if (!shader_override.empty()) {
      const auto bytes = read_file(shader_override, 256 * 1024);
      shader.assign(bytes.begin(), bytes.end());
      require(!shader.empty() && shader.find('\0') == std::string::npos, "Invalid experimental shader source");
      require(!replay::needs_disparity_limit_override(package, shader) || disparity_limit_uv.has_value(),
        "A v1 dump replayed with the bounded shader requires explicit --disparity-limit-uv; v1 offset 28 was reserved padding");
    }
    sunshine_game3d::render_parameters parameters;
    static_assert(sizeof(parameters) == replay::parameter_bytes);
    std::memcpy(&parameters, package.parameters.data(), sizeof(parameters));
    if (strength) {
      parameters.strength = *strength;
    }
    if (zero_inverse_depth) {
      // Convergence and gain are independent production controls.
      parameters.convergence[1] = *zero_inverse_depth;
    }
    if (depth_gain) parameters.depth_scale = *depth_gain;
    if (disparity_limit_uv) parameters.disparity_limit_uv = *disparity_limit_uv;
    const bool protect_ui = source_alpha_ui.value_or(package.source_alpha_ui);
    const auto ui_plane = ui_plane_override.value_or(package.ui_plane);
    const std::string ui_alpha_source = !protect_ui ? "none" : package.ui_alpha_source == "ui_source_color" ? "ui_source_color" : "source_color";
    require(!protect_ui || shader.find("Sunshine_SourceAlphaUI") != std::string::npos,
      "Source-alpha UI requires a shader that consumes the UI selection");
    require(ui_plane.mode == sunshine_game3d::ui_plane_mode::screen || replay::supports_ui_plane(shader),
      "Independent UI plane requires a shader that consumes its parameters; supply --shader for a legacy captured shader");
    require(ui_plane.mode != sunshine_game3d::ui_plane_mode::depth_midpoint_nearest_ui || replay::supports_nearest_ui_plane(shader),
      "Nearest-covered UI plane requires its GPU reduction shader; supply --shader for a legacy captured shader");
    require(ui_plane.mode != sunshine_game3d::ui_plane_mode::front_limit || replay::supports_front_ui_plane(shader),
      "Fixed front-limit UI plane requires its shader contract; supply --shader for a legacy captured shader");
    const bool experiment = ui_plane_override.has_value() || source_alpha_ui.has_value() || strength.has_value() || zero_inverse_depth.has_value() || depth_gain.has_value() || disparity_limit_uv.has_value() || !shader_override.empty();
    require(fs::create_directories(output), "Cannot create fresh output directory");
    created = true;
    write_file(output / "shader_used.hlsl", shader.data(), shader.size());
    const auto recorded_parameters = replay::hex(&parameters, sizeof(parameters));
    json report {{"schema", "sunshine.game3d.replay-result.v1"}, {"scope", "Frozen single-frame production Game 3D renderer, before host scaling, overlay, Windows cursor composition and encoder; saved constants freeze temporal scene policy."}, {"input_directory", directory.string()}, {"baseline", !experiment}, {"experiment", experiment}, {"runtime_dll", dll.string()}, {"renderer_api", "D3D11 via official ReShade"}, {"shader_source", shader_override.empty() ? "captured embedded HLSL" : shader_override.string()}, {"render_parameter_hex", recorded_parameters}, {"captured_parameter_hex", replay::hex(package.parameters.data(), package.parameters.size())}, {"capture_id", manifest.value("capture_id", std::uint64_t(0))}, {"source_width", package.width}, {"source_height", package.height}, {"source_color_space", package.color}, {"tolerance", "float32 fields: abs <= 1e-7 + 1e-5*max(abs(a),abs(b)); FP16/color: one native quantization step. Byte exactness is separately reported; nonfinite samples fail verification."}, {"artifacts", json::array()}};
    if (zero_inverse_depth) {
      report["zero_inverse_depth_override"] = {{"scope", "offline experiment only"}, {"requested_argument", zero_inverse_depth_argument},
        {"convergence_1", parameters.convergence[1]}, {"depth_scale", parameters.depth_scale},
        {"rule", "zero plane only; gain is independent"}};
    }
    if (depth_gain) report["depth_gain_override"] = *depth_gain;
    report["source_alpha_ui"] = protect_ui;
    report["captured_source_alpha_ui"] = package.source_alpha_ui;
    const auto describe_ui = [](bool enabled, const sunshine_game3d::ui_plane_parameters &plane) {
      const auto words = sunshine_game3d::ui_parameter_words(enabled, plane);
      return json {{"mode", static_cast<std::uint32_t>(plane.mode)},
        {"mode_name", plane.mode == sunshine_game3d::ui_plane_mode::screen ? "screen" :
          plane.mode == sunshine_game3d::ui_plane_mode::depth_midpoint ? "depth_midpoint" :
          plane.mode == sunshine_game3d::ui_plane_mode::depth_midpoint_nearest_ui ? "depth_midpoint_nearest_ui" :
          plane.mode == sunshine_game3d::ui_plane_mode::front_limit ? "front_limit" : "unknown"},
        {"inverse_depth", std::isfinite(plane.inverse_depth) ? json(plane.inverse_depth) : json(nullptr)},
        {"inverse_depth_role", plane.mode == sunshine_game3d::ui_plane_mode::front_limit ||
          plane.mode == sunshine_game3d::ui_plane_mode::screen ? "unused" :
          plane.mode == sunshine_game3d::ui_plane_mode::depth_midpoint_nearest_ui ? "midpoint_floor" : "explicit_plane"},
        {"inverse_depth_bits", words[2]}, {"uint32", words}};
    };
    report["captured_ui_parameter_abi"] = package.ui_parameter_abi;
    report["captured_ui_parameter_hex"] = replay::hex(package.ui_parameters.data(), package.ui_parameters.size());
    report["captured_ui_plane"] = describe_ui(package.source_alpha_ui, package.ui_plane);
    report["ui_parameter_abi"] = ui_plane.mode == sunshine_game3d::ui_plane_mode::front_limit ? "sunshine_game3d.ui_parameters.v4" :
      ui_plane.mode == sunshine_game3d::ui_plane_mode::depth_midpoint_nearest_ui ?
      "sunshine_game3d.ui_parameters.v3" : "sunshine_game3d.ui_parameters.v2";
    report["ui_parameter_bytes"] = replay::ui_parameter_bytes;
    if (ui_plane_override) report["ui_plane_override"] = {{"scope", "offline experiment only"},
      {"requested_argument", ui_plane_argument},
      {"rule", "Only the UI plane changes. Front-limit mode uses the current positive display bound scaled by strength and stereo blend, independent of scene depth/gain/zero. Depth modes map qUI through current geometry. No frozen displacement is supplied."}};
    report["ui_alpha_source"] = ui_alpha_source;
    report["captured_ui_alpha_source"] = package.ui_alpha_source;
    if (package.ui_alpha_source == "ui_source_color") report["captured_ui_source_provenance"] = manifest.at("producer_metadata").value("ui_source", json::object());
    report["validated_optional_artifacts_not_used_by_renderer"] = package.ignored_optional_artifacts;
    if (source_alpha_ui) report["source_alpha_ui_override"] = *source_alpha_ui;
    report["captured_parameter_abi"] = package.parameter_abi;
    if (disparity_limit_uv) report["disparity_limit_uv_override"] = {{"scope", "offline experiment only"},
      {"per_eye_width_fraction_at_full_strength", *disparity_limit_uv}, {"parameter_byte_offset", 28}};
    runtime_fixture fixture;
    fixture.initialize(dll, output, package);
    ComPtr<IDXGIDevice> dxgi_device;
    ComPtr<IDXGIAdapter> adapter;
    checked(fixture.device.As(&dxgi_device), "Query replay GPU identity");
    checked(dxgi_device->GetAdapter(&adapter), "Query replay adapter");
    DXGI_ADAPTER_DESC adapter_desc {};
    checked(adapter->GetDesc(&adapter_desc), "Read replay adapter identity");
    report["adapter"] = {{"vendor_id", adapter_desc.VendorId}, {"device_id", adapter_desc.DeviceId}, {"subsystem_id", adapter_desc.SubSysId}, {"revision", adapter_desc.Revision}, {"luid_low", adapter_desc.AdapterLuid.LowPart}, {"luid_high", adapter_desc.AdapterLuid.HighPart}};
    auto *queue = runtime_seen->get_command_queue();
    sunshine_game3d::renderer renderer;

    struct renderer_drain {
      sunshine_game3d::renderer &renderer;
      api::command_queue *queue;

      ~renderer_drain() {
        queue->flush_immediate_command_list();
        queue->wait_idle();
        renderer.reset_after_runtime_drain();
      }
    } drain {renderer, queue};

    const api::resource backbuffer {reinterpret_cast<std::uint64_t>(fixture.backbuffer.Get())};
    require(renderer.configure(runtime_seen, backbuffer, static_cast<api::color_space>(package.color), shader), "Production renderer cannot configure captured frame");
    const auto source = fixture.upload(package.artifacts.at("source_color"), read_artifact(directory, package.artifacts.at("source_color")));
    fixture.context->CopyResource(fixture.backbuffer.Get(), source.Get());
    ComPtr<ID3D11Texture2D> depth;
    ComPtr<ID3D11ShaderResourceView> depth_view;
    if (const auto found = package.artifacts.find("raw_depth"); found != package.artifacts.end()) {
      depth = fixture.upload(found->second, read_artifact(directory, found->second));
      checked(fixture.device->CreateShaderResourceView(depth.Get(), nullptr, &depth_view), "Create full-allocation R32 depth SRV");
    }
    ComPtr<ID3D11Texture2D> ui_source;
    ComPtr<ID3D11ShaderResourceView> ui_source_view;
    if (ui_alpha_source == "ui_source_color") {
      const auto &a = package.artifacts.at("ui_source_color");
      ui_source = fixture.upload(a, read_artifact(directory, a));
      checked(fixture.device->CreateShaderResourceView(ui_source.Get(), nullptr, &ui_source_view), "Create exact consumed UI-alpha SRV");
    }
    require(renderer.render(queue->get_immediate_command_list(), backbuffer, {reinterpret_cast<std::uint64_t>(depth_view.Get())}, parameters, protect_ui, {reinterpret_cast<std::uint64_t>(ui_source_view.Get())}, ui_plane), "Production renderer rejected replay");
    const auto consumed_ui = renderer.consumed_ui_plane();
    const auto consumed_ui_words = sunshine_game3d::ui_parameter_words(renderer.consumed_source_alpha_ui(), consumed_ui);
    require(consumed_ui_words == sunshine_game3d::ui_parameter_words(protect_ui, ui_plane), "Renderer changed the submitted UI constants");
    report["ui_parameter_hex"] = replay::hex(consumed_ui_words.data(), sizeof(consumed_ui_words));
    report["applied_ui_plane"] = describe_ui(renderer.consumed_source_alpha_ui(), consumed_ui);
    queue->flush_immediate_command_list();
    renderer.finish_present();
    queue->wait_idle();
    const auto resources = renderer.diagnostics();
    if (resources.ui_plane_resolved.handle) {
      const replay::artifact scalar {"ui_plane_resolved", "ui_plane_resolved.bin", 1, 1, DXGI_FORMAT_R32_FLOAT, 4, 4};
      const auto bytes = fixture.read(reinterpret_cast<ID3D11Texture2D *>(resources.ui_plane_resolved.handle), scalar);
      require(bytes.size() == sizeof(float), "Invalid resolved UI plane readback");
      float resolved;
      std::memcpy(&resolved, bytes.data(), sizeof(resolved));
      require(std::isfinite(resolved) && resolved >= 0.f, "Invalid GPU-resolved UI inverse depth");
      write_file(output / scalar.file, bytes.data(), bytes.size());
      report["applied_ui_plane"]["resolved_inverse_depth"] = resolved;
      report["applied_ui_plane"]["resolved_inverse_depth_file"] = scalar.file;
      report["applied_ui_plane"]["resolution"] = "Same-render GPU maximum over covered depth with the submitted midpoint floor";
    }
    bool all_exact = true, all_within = true;
    for (const auto &[kind, a] : package.artifacts) {
      if (kind == "raw_depth") {
        continue;  // An input, never a separately reconstructed depth algorithm.
      }
      if (kind == "ui_source_color" && ui_alpha_source != "ui_source_color") continue;  // Explicit offline protection-off override.
      const auto resource = diagnostic(resources, kind);
      require(resource.handle != 0, "Production renderer did not produce captured artifact: " + kind);
      const auto pixels = fixture.read(reinterpret_cast<ID3D11Texture2D *>(resource.handle), a);
      const auto comparison = compare(a, read_artifact(directory, a), pixels);
      all_exact &= comparison.at("byte_exact").get<bool>();
      all_within &= comparison.at("within_tolerance").get<bool>();
      const auto filename = kind + ".bin";
      write_file(output / filename, pixels.data(), pixels.size());
      report["artifacts"].push_back({{"kind", kind}, {"file", filename}, {"width", a.width}, {"height", a.height}, {"dxgi_format", a.format}, {"row_bytes", a.row_bytes}, {"byte_count", a.byte_count}, {"comparison", comparison}});
      std::printf("%s: %s, max error %.9g, above tolerance %llu\n", kind.c_str(), comparison.at("byte_exact").get<bool>() ? "byte exact" : "differs", comparison.at("max_absolute_error").get<double>(), static_cast<unsigned long long>(comparison.at("samples_above_tolerance").get<std::uint64_t>()));
    }
    report["byte_exact"] = all_exact;
    report["within_tolerance"] = all_within;
    report["status"] = "complete";
    report["verification_requested"] = verify;
    write_json(output / "replay.json", report);
    std::printf("%s replay complete: %s\n", experiment ? "Experimental" : "Captured baseline", (output / "replay.json").string().c_str());
    return verify && !all_within ? 1 : 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "Game 3D replay failed: %s\n", e.what());
    if (created) {
      try {
        write_json(output / "failure.json", {{"status", "failed"}, {"reason", e.what()}});
      } catch (...) {}
    }
    return 2;
  }
}
