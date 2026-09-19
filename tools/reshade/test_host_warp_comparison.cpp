// SPDX-License-Identifier: GPL-3.0-only
// Opt-in hardware GPU comparison: Game3D source convention -> unchanged Host V2
// conditioning/inverse. This is not the DAV2 camera/calibration or live authority path.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi.h>
#include <bcrypt.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/generated/depth_coordinate_v2_contract.h"

namespace fs = std::filesystem;
namespace v2 = models::depth_coordinate_v2;
using Microsoft::WRL::ComPtr;

namespace {
  void need(bool ok, const std::string &message) {
    if (!ok) throw std::runtime_error(message);
  }
  void check(HRESULT result, const char *message) {
    if (FAILED(result)) {
      std::ostringstream text;
      text << message << " HRESULT=0x" << std::hex << std::uint32_t(result);
      throw std::runtime_error(text.str());
    }
  }
  std::vector<std::uint8_t> read(const fs::path &path) {
    std::ifstream file(path, std::ios::binary);
    need(file.good(), "Cannot read " + path.string());
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  }
  void write(const fs::path &path, const void *data, size_t bytes) {
    fs::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::binary);
    file.write(static_cast<const char *>(data), std::streamsize(bytes));
    need(file.good(), "Cannot write " + path.string());
  }
  template<class T> void write(const fs::path &path, const std::vector<T> &data) {
    write(path, data.data(), data.size() * sizeof(T));
  }
  std::string sha256(const void *data, size_t bytes) {
    need(bytes <= std::numeric_limits<ULONG>::max(), "Hash input too large");
    BCRYPT_ALG_HANDLE algorithm {};
    need(BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) >= 0,
      "Cannot open SHA256 provider");
    std::array<UCHAR, 32> hash {};
    const auto status = BCryptHash(algorithm, nullptr, 0,
      const_cast<PUCHAR>(static_cast<const UCHAR *>(data)), ULONG(bytes), hash.data(), ULONG(hash.size()));
    BCryptCloseAlgorithmProvider(algorithm, 0);
    need(status >= 0, "Cannot hash bytes");
    std::ostringstream text;
    text << std::hex << std::setfill('0');
    for (auto value : hash) text << std::setw(2) << unsigned(value);
    return text.str();
  }
  template<class T> std::string sha256(const std::vector<T> &data) {
    return sha256(data.data(), data.size() * sizeof(T));
  }

  // All compile roots and includes are immutable snapshots, retained with their hashes.
  struct includes_t final : ID3DInclude {
    fs::path root;
    std::map<std::string, std::vector<std::uint8_t>> sources;
    explicit includes_t(fs::path path) : root(std::move(path)) {}
    const std::vector<std::uint8_t> &source(const std::string &name) {
      const fs::path relative(name);
      need(!relative.is_absolute(), "Absolute shader include rejected");
      for (const auto &part : relative) need(part != "..", "Parent shader include rejected");
      auto [item, inserted] = sources.try_emplace(relative.generic_string());
      if (inserted) item->second = read(root / relative);
      return item->second;
    }
    HRESULT STDMETHODCALLTYPE Open(D3D_INCLUDE_TYPE, LPCSTR name, LPCVOID, LPCVOID *data, UINT *bytes) override {
      try {
        const auto &snapshot = source(name);
        need(snapshot.size() <= UINT_MAX, "Shader include too large");
        *data = snapshot.data();
        *bytes = UINT(snapshot.size());
        return S_OK;
      } catch (...) { return E_FAIL; }
    }
    HRESULT STDMETHODCALLTYPE Close(LPCVOID) override { return S_OK; }
  };

  struct texture_t {
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> srv;
    ComPtr<ID3D11UnorderedAccessView> uav;
    ComPtr<ID3D11RenderTargetView> rtv;
  };

  struct fixture_t {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    includes_t includes;
    fs::path output;
    std::ofstream manifest;
    ComPtr<ID3D11ComputeShader> adapter, vertical, horizontal;
    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> ps;
    ComPtr<ID3D11Buffer> depth_cb, coordinate_cb, adapter_cb, geometry_cb, state;
    ComPtr<ID3D11ShaderResourceView> state_srv;
    ComPtr<ID3D11SamplerState> sampler;
    ComPtr<ID3D11RasterizerState> rasterizer;
    unsigned width, height;

    fixture_t(fs::path shader_root, fs::path output_path, unsigned w, unsigned h,
              float gain, float zero, float reference, float strength) :
      includes(std::move(shader_root)), output(std::move(output_path)), width(w), height(h) {
      need(!fs::exists(output), "Output directory already exists; use fresh evidence path");
      fs::create_directories(output);
      manifest.open(output / "manifest.txt");
      manifest << std::setprecision(10)
        << "schema=game3d-host-warp-comparison-1\n"
        << "scope=Game3D-coordinate-adapter + unmodified HostV2 vertical/horizontal/live inverse\n"
        << "camera_calibration=not_under_test\nDAV2_inference=not_executed\n"
        << "renderer_authorization=test_only_explicit_tag_not_live_provenance\n"
        << "numeric_warp=GPU_only\ncolor_input=RGBA16F_linear_scRGB\n"
        << "color_output=RGBA16F_linear_scRGB_decoded_to_RGBA32F\n"
        << "width=" << width << "\nheight=" << height << "\nH=" << gain << "\nq0=" << zero
        << "\ng=" << reference << "\nstrength=" << strength << '\n'
        << "candidate=-clamp(g*H*(q0-q)*strength/100,-1.5,2.5)*(height/2160*100)/width\n"
        << "compile_flags=" << models::host_sbs_shader_cache::shader_compile_flags << '\n'
        << "contract_schema=" << v2::contract_schema << '\n'
        << "contract_sha256=" << v2::contract_canonical_sha256 << '\n';
      const D3D_FEATURE_LEVEL requested[] {D3D_FEATURE_LEVEL_11_0};
      D3D_FEATURE_LEVEL actual {};
      check(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
        requested, 1, D3D11_SDK_VERSION, &device, &actual, &context), "Hardware D3D11 device");
      ComPtr<IDXGIDevice> dxgi_device;
      ComPtr<IDXGIAdapter> gpu;
      check(device.As(&dxgi_device), "Get DXGI device");
      check(dxgi_device->GetAdapter(&gpu), "Get hardware adapter");
      DXGI_ADAPTER_DESC description {};
      check(gpu->GetDesc(&description), "Get adapter description");
      const auto adapter_name = fs::path(description.Description).u8string();
      manifest << "adapter=" << std::string(adapter_name.begin(), adapter_name.end())
        << "\nvendor=" << description.VendorId << "\ndevice=" << description.DeviceId
        << "\nluid_high=" << description.AdapterLuid.HighPart
        << "\nluid_low=" << description.AdapterLuid.LowPart << '\n';

      const auto adapter_source = std::string(R"(
Texture2D<float> RawDepth : register(t0);
RWTexture2D<float> Candidate : register(u0);
cbuffer Adapter : register(b0) {
  uint width; uint height; float H; float q0;
  float g; float strength; float pixel_to_u; float reserved;
};
[numthreads(8,8,1)] void main(uint3 id : SV_DispatchThreadID) {
  if (id.x >= width || id.y >= height) return;
  precise float displacement = g * H * (q0 - RawDepth[id.xy]) * strength / 100.0;
  Candidate[id.xy] = -clamp(displacement, -1.5, 2.5) * pixel_to_u;
}
)");
      write(output / "adapter.hlsl", adapter_source.data(), adapter_source.size());
      auto blob = compile(adapter_source.data(), adapter_source.size(), "adapter.hlsl", "main", "cs_5_0");
      check(device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &adapter), "Create adapter shader");
      blob = compile_file("depth_coordinate_v2_vertical_limit_cs.hlsl", "main", "cs_5_0");
      check(device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &vertical), "Create vertical shader");
      blob = compile_file("depth_coordinate_v2_limit_cs.hlsl", "main", "cs_5_0");
      check(device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &horizontal), "Create horizontal shader");
      blob = compile_file("sbs_reprojection_vs.hlsl", "main_vs", "vs_5_0");
      check(device->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &vs), "Create vertex shader");
      blob = compile_file("sbs_reprojection_v2_live_ps.hlsl", "main_ps", "ps_5_0");
      check(device->CreatePixelShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &ps), "Create live pixel shader");
      std::ostringstream closure;
      for (const auto &[name, bytes] : includes.sources) {
        const auto digest = sha256(bytes);
        manifest << "shader_source " << name << ' ' << digest << '\n';
        closure << name << ' ' << digest << '\n';
        write(output / "shader-sources" / name, bytes);
      }
      const auto closure_text = closure.str();
      manifest << "source_closure_format=sorted-path-space-sha256-LF\nsource_closure_sha256="
        << sha256(closure_text.data(), closure_text.size()) << '\n';
      write(output / "source-closure.txt", closure_text.data(), closure_text.size());
      std::array<std::uint32_t, 16> dimensions {};
      dimensions[0] = width; dimensions[1] = height;
      dimensions[11] = width; dimensions[12] = height;
      depth_cb = buffer(dimensions.data(), sizeof(dimensions));
      const v2::constants_t constants {
        v2::model_calibrations.front().raw_coordinate_scale, v2::collapse_abs_epsilon,
        v2::far_tau, v2::near_log_tau, v2::gain_per_pop,
        v2::max_horizontal_slope, v2::direct_container_limit, v2::convergence_curve_default
      };
      coordinate_cb = buffer(&constants, sizeof(constants));
      struct adapter_constants_t { UINT w, h; float H, q0, g, strength, pixel_to_u, pad; };
      const adapter_constants_t adapter_constants {width, height, gain, zero, reference, strength,
        (float(height) / 2160.f * 100.f) / float(width), 0};
      adapter_cb = buffer(&adapter_constants, sizeof(adapter_constants));
      std::array<float, 12> geometry {};
      geometry[0] = geometry[1] = 1;
      geometry_cb = buffer(geometry.data(), sizeof(geometry));
      auto state_words = v2::state_initial_words;
      // Explicit test authority for the renderer, not a forged live producer receipt.
      state_words[v2::renderer_authorization_bits] = v2::contract_tag;
      D3D11_BUFFER_DESC state_desc {};
      state_desc.ByteWidth = sizeof(state_words);
      state_desc.Usage = D3D11_USAGE_IMMUTABLE;
      state_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      state_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      state_desc.StructureByteStride = 16;
      D3D11_SUBRESOURCE_DATA state_data {state_words.data(), 0, 0};
      check(device->CreateBuffer(&state_desc, &state_data, &state), "Create test state");
      check(device->CreateShaderResourceView(state.Get(), nullptr, &state_srv), "Create test state SRV");
      write(output / "test-state.u32", state_words.data(), sizeof(state_words));
      D3D11_SAMPLER_DESC sampling {};
      sampling.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
      sampling.AddressU = sampling.AddressV = sampling.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
      sampling.MaxLOD = D3D11_FLOAT32_MAX;
      check(device->CreateSamplerState(&sampling, &sampler), "Create linear clamp sampler");
      D3D11_RASTERIZER_DESC raster {};
      raster.FillMode = D3D11_FILL_SOLID;
      raster.CullMode = D3D11_CULL_NONE;
      raster.DepthClipEnable = TRUE;
      check(device->CreateRasterizerState(&raster, &rasterizer), "Create rasterizer");
    }

    ComPtr<ID3DBlob> compile(const void *text, size_t size, const char *name, const char *entry, const char *target) {
      ComPtr<ID3DBlob> blob, error;
      const auto result = D3DCompile(text, size, name, nullptr, &includes, entry, target,
        models::host_sbs_shader_cache::shader_compile_flags, 0, &blob, &error);
      if (FAILED(result) && error)
        std::cerr.write(static_cast<const char *>(error->GetBufferPointer()), error->GetBufferSize());
      check(result, name);
      const auto cso = std::string(name) + ".cso";
      write(output / cso, blob->GetBufferPointer(), blob->GetBufferSize());
      manifest << "shader_cso " << cso << ' ' << sha256(blob->GetBufferPointer(), blob->GetBufferSize()) << '\n';
      return blob;
    }
    ComPtr<ID3DBlob> compile_file(const char *name, const char *entry, const char *target) {
      const auto &bytes = includes.source(name);
      return compile(bytes.data(), bytes.size(), name, entry, target);
    }
    ComPtr<ID3D11Buffer> buffer(const void *bytes, UINT size) {
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = size; desc.Usage = D3D11_USAGE_IMMUTABLE;
      desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      D3D11_SUBRESOURCE_DATA data {bytes, 0, 0};
      ComPtr<ID3D11Buffer> value;
      check(device->CreateBuffer(&desc, &data, &value), "Create constants");
      return value;
    }
    texture_t texture(UINT w, UINT h, DXGI_FORMAT format, UINT flags, const void *data = nullptr, UINT pitch = 0) {
      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = w; desc.Height = h; desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
      desc.Format = format; desc.Usage = D3D11_USAGE_DEFAULT; desc.BindFlags = flags;
      D3D11_SUBRESOURCE_DATA initial {data, pitch, 0};
      texture_t value;
      check(device->CreateTexture2D(&desc, data ? &initial : nullptr, &value.texture), "Create texture");
      if (flags & D3D11_BIND_SHADER_RESOURCE)
        check(device->CreateShaderResourceView(value.texture.Get(), nullptr, &value.srv), "Create texture SRV");
      if (flags & D3D11_BIND_UNORDERED_ACCESS)
        check(device->CreateUnorderedAccessView(value.texture.Get(), nullptr, &value.uav), "Create texture UAV");
      if (flags & D3D11_BIND_RENDER_TARGET)
        check(device->CreateRenderTargetView(value.texture.Get(), nullptr, &value.rtv), "Create texture RTV");
      return value;
    }
    std::vector<std::uint8_t> readback(ID3D11Texture2D *texture, unsigned pixel_bytes) {
      D3D11_TEXTURE2D_DESC desc {};
      texture->GetDesc(&desc);
      desc.Usage = D3D11_USAGE_STAGING; desc.BindFlags = desc.MiscFlags = 0;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Texture2D> staging;
      check(device->CreateTexture2D(&desc, nullptr, &staging), "Create readback");
      context->CopyResource(staging.Get(), texture);
      D3D11_MAPPED_SUBRESOURCE mapped {};
      check(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped), "Read GPU result");
      std::vector<std::uint8_t> bytes(size_t(desc.Width) * desc.Height * pixel_bytes);
      for (unsigned y = 0; y < desc.Height; ++y)
        std::memcpy(bytes.data() + size_t(y) * desc.Width * pixel_bytes,
          static_cast<const std::uint8_t *>(mapped.pData) + size_t(y) * mapped.RowPitch,
          size_t(desc.Width) * pixel_bytes);
      context->Unmap(staging.Get(), 0);
      return bytes;
    }
    void dispatch(ID3D11ComputeShader *shader, ID3D11ShaderResourceView *input,
                  ID3D11UnorderedAccessView *first, ID3D11UnorderedAccessView *second,
                  ID3D11Buffer *cb0, ID3D11Buffer *cb1, UINT x, UINT y) {
      ID3D11UnorderedAccessView *uavs[] {first, second};
      ID3D11Buffer *cbs[] {cb0, cb1};
      context->CSSetShader(shader, nullptr, 0);
      context->CSSetShaderResources(0, 1, &input);
      context->CSSetUnorderedAccessViews(0, second ? 2 : 1, uavs, nullptr);
      context->CSSetConstantBuffers(0, 2, cbs);
      context->Dispatch(x, y, 1);
      ID3D11ShaderResourceView *null_srv = nullptr;
      ID3D11UnorderedAccessView *null_uavs[] {nullptr, nullptr};
      context->CSSetShaderResources(0, 1, &null_srv);
      context->CSSetUnorderedAccessViews(0, 2, null_uavs, nullptr);
      context->CSSetShader(nullptr, nullptr, 0);
    }
    std::vector<float> field(const texture_t &texture, const std::string &path) {
      const auto bytes = readback(texture.texture.Get(), 4);
      std::vector<float> values(size_t(width) * height);
      std::memcpy(values.data(), bytes.data(), bytes.size());
      for (auto value : values) need(std::isfinite(value), "Nonfinite field " + path);
      write(output / path, bytes);
      manifest << "field " << path << ' ' << sha256(bytes) << '\n';
      return values;
    }
    static float half_float(std::uint16_t value) {
      const int exponent = (value >> 10) & 31;
      const int fraction = value & 1023;
      need(exponent != 31, "Nonfinite FP16 result");
      const float magnitude = exponent ? std::ldexp(float(1024 + fraction), exponent - 25) :
        std::ldexp(float(fraction), -24);
      return value & 0x8000 ? -magnitude : magnitude;
    }
    void run(const fs::path &input, const std::string &name) {
      const auto color = read(input / (name + ".source.bin"));
      const auto depth = read(input / (name + ".depth.f32"));
      need(color.size() == size_t(width) * height * 8 && depth.size() == size_t(width) * height * 4,
        "Input dimensions/format mismatch for " + name);
      for (size_t i = 0; i < color.size(); i += 2) {
        std::uint16_t value; std::memcpy(&value, color.data() + i, 2);
        need((value & 0x7c00) != 0x7c00, "Nonfinite source color");
      }
      for (size_t i = 0; i < depth.size(); i += 4) {
        float value; std::memcpy(&value, depth.data() + i, 4);
        need(std::isfinite(value) && value >= 0 && value <= 1, "Invalid raw source depth");
      }
      write(output / (name + ".source.bin"), color);
      write(output / (name + ".depth.f32"), depth);
      manifest << "source " << name << ' ' << sha256(color) << ' ' << sha256(depth) << '\n';
      auto source = texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D11_BIND_SHADER_RESOURCE, color.data(), width * 8);
      auto raw = texture(width, height, DXGI_FORMAT_R32_FLOAT, D3D11_BIND_SHADER_RESOURCE, depth.data(), width * 4);
      constexpr UINT flags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      auto candidate = texture(width, height, DXGI_FORMAT_R32_FLOAT, flags);
      auto majorant = texture(width, height, DXGI_FORMAT_R32_FLOAT, flags);
      auto conditioned = texture(width, height, DXGI_FORMAT_R32_FLOAT, flags);
      auto final = texture(width, height, DXGI_FORMAT_R32_FLOAT, flags);
      auto packed = texture(width * 2, height, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D11_BIND_RENDER_TARGET);
      const float invalid[] {NAN, NAN, NAN, NAN};
      for (const auto *value : {&candidate, &majorant, &conditioned, &final})
        context->ClearUnorderedAccessViewFloat(value->uav.Get(), invalid);
      // The FP16 readback rejects nonfinite channels, including any pixel a
      // malformed fullscreen draw leaves unwritten.
      context->ClearRenderTargetView(packed.rtv.Get(), invalid);
      dispatch(adapter.Get(), raw.srv.Get(), candidate.uav.Get(), nullptr,
        adapter_cb.Get(), nullptr, (width + 7) / 8, (height + 7) / 8);
      dispatch(vertical.Get(), candidate.srv.Get(), majorant.uav.Get(), conditioned.uav.Get(),
        depth_cb.Get(), coordinate_cb.Get(), width, 1);
      dispatch(horizontal.Get(), conditioned.srv.Get(), final.uav.Get(), nullptr,
        depth_cb.Get(), coordinate_cb.Get(), height, 1);
      const auto candidates = field(candidate, name + ".candidate.f32");
      for (auto value : candidates)
        need(std::abs(value) <= v2::direct_container_limit, "Candidate would be silently clipped by Host container");
      field(majorant, name + ".vertical-majorant.f32");
      field(conditioned, name + ".conditioned.f32");
      const auto values = field(final, name + ".final.f32");
      double max_abs = 0, max_horizontal = 0, max_vertical = 0;
      for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
        const size_t i = size_t(y) * width + x;
        max_abs = std::max(max_abs, std::abs(double(values[i])));
        if (x) max_horizontal = std::max(max_horizontal, std::abs(double(values[i]) - values[i - 1]) * width);
        if (y) max_vertical = std::max(max_vertical, std::abs(double(values[i]) - values[i - width]) * width);
      }
      need(max_abs <= double(v2::direct_container_limit) + 2e-7, "Final field exceeds container");
      need(max_horizontal <= double(v2::max_horizontal_slope) + 2e-5, "Final horizontal slope exceeds contract");
      need(max_vertical <= double(v2::max_vertical_shear) + 2e-5, "Final vertical shear exceeds contract");
      manifest << "validation " << name << " max_abs=" << max_abs
        << " max_horizontal_slope=" << max_horizontal << " max_vertical_shear=" << max_vertical << '\n';
      D3D11_VIEWPORT viewport {0, 0, float(width * 2), float(height), 0, 1};
      context->RSSetViewports(1, &viewport);
      context->RSSetState(rasterizer.Get());
      ID3D11RenderTargetView *rtv = packed.rtv.Get();
      context->OMSetRenderTargets(1, &rtv, nullptr);
      context->IASetInputLayout(nullptr);
      context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      context->VSSetShader(vs.Get(), nullptr, 0);
      context->PSSetShader(ps.Get(), nullptr, 0);
      ID3D11ShaderResourceView *srvs[] {source.srv.Get(), final.srv.Get(), state_srv.Get()};
      context->PSSetShaderResources(0, 3, srvs);
      context->PSSetConstantBuffers(2, 1, geometry_cb.GetAddressOf());
      context->PSSetSamplers(0, 1, sampler.GetAddressOf());
      context->Draw(3, 0);
      ID3D11ShaderResourceView *null_srvs[] {nullptr, nullptr, nullptr};
      context->PSSetShaderResources(0, 3, null_srvs);
      context->OMSetRenderTargets(0, nullptr, nullptr);
      const auto packed_bytes = readback(packed.texture.Get(), 8);
      std::vector<float> rgba(packed_bytes.size() / 2);
      for (size_t i = 0; i < rgba.size(); ++i) {
        std::uint16_t value; std::memcpy(&value, packed_bytes.data() + i * 2, 2);
        rgba[i] = half_float(value);
      }
      const auto label = name == "calibration" || name.starts_with("motion-") ? name : name + "-aa0";
      write(output / (label + ".rgba.f32"), rgba);
      manifest << "render " << label << ' ' << sha256(rgba) << '\n';
      manifest.flush();
      need(manifest.good(), "Could not publish manifest");
      std::cout << "PASS " << name << " max_horizontal_slope=" << max_horizontal << '\n';
    }
  };
}

int main(int argc, char **argv) {
  try {
    need(argc == 10, "usage: host_warp_comparison <shader-root> <input-parity> <fresh-output> <width> <height> <H> <q0> <g> <strength>");
    auto integer = [](const char *text) {
      size_t used = 0; const auto value = std::stoul(text, &used);
      need(used == std::strlen(text) && value > 0 && value <= UINT_MAX, "Invalid dimension");
      return unsigned(value);
    };
    auto scalar = [](const char *text) {
      size_t used = 0; const float value = std::stof(text, &used);
      need(used == std::strlen(text) && std::isfinite(value), "Invalid scalar");
      return value;
    };
    const unsigned width = integer(argv[4]), height = integer(argv[5]);
    constexpr unsigned maximum_dimension = 2u * v2::model_calibrations.front().preprocess.maximum_dimension;
    need(width <= maximum_dimension && height <= maximum_dimension,
      "Source exceeds unchanged Host limiter maximum dimension");
    const float gain = scalar(argv[6]), zero = scalar(argv[7]), reference = scalar(argv[8]), strength = scalar(argv[9]);
    // Match the existing fixed-camera fixture's H domain. Combined with the
    // per-pixel q validation and these scalar bounds, every multiplication in
    // the pre-clamp GPU displacement is finite (magnitude at most 1.6e6).
    need(gain > 0 && gain <= 16000 && zero >= 0 && zero <= 1 && reference > 0 && reference <= 1 && strength >= 0 && strength <= 100,
      "Camera/strength outside test adapter domain");
    const auto input = fs::absolute(argv[2]);
    fixture_t fixture(fs::absolute(argv[1]), fs::absolute(argv[3]), width, height, gain, zero, reference, strength);
    for (const auto *name : {"calibration", "step", "detail", "fringe"}) fixture.run(input, name);
    std::set<std::string> motion;
    for (const auto &entry : fs::directory_iterator(input)) {
      const auto name = entry.path().filename().string();
      if (name.starts_with("motion-") && name.ends_with(".source.bin"))
        motion.insert(name.substr(0, name.size() - std::strlen(".source.bin")));
    }
    for (const auto &name : motion) fixture.run(input, name);
    fixture.manifest << "status=complete\n";
    fixture.manifest.flush();
    need(fixture.manifest.good(), "Cannot finish evidence manifest");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
