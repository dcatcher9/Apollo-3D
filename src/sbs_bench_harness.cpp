/**
 * @file src/sbs_bench_harness.cpp
 * @brief Headless frame-fed SBS benchmark harness (see sbs_bench_harness.h).
 *
 * Duplicates the minimal SBS composite from platform/windows/display_vram.cpp convert()
 * (which lives in an anonymous-namespace class and can't be called directly) but drives it
 * with the REAL video_depth_estimator on a fixed directory of frames. Output PNGs are scored
 * by tools/sbsbench/sbsbench.py. Windows-only (the estimator + shaders are D3D11/TensorRT).
 */
#include "sbs_bench_harness.h"

#ifdef _WIN32

  // standard includes
  #include <algorithm>
  #include <cctype>
  #include <chrono>
  #include <cmath>
  #include <cstring>
  #include <filesystem>
  #include <fstream>
  #include <iomanip>
  #include <limits>
  #include <locale>
  #include <string>
  #include <string_view>
  #include <thread>
  #include <vector>

  // platform includes
  #include <d3d11.h>
  #include <d3dcompiler.h>
  #include <wincodec.h>
  #include <wrl/client.h>

  // local includes
  #include "config.h"
  #include "logging.h"
  #include "sbs_perf.h"
  #include "video.h"
  #include "video_depth_estimator.h"

  #ifndef SUNSHINE_SHADERS_DIR
    #define SUNSHINE_SHADERS_DIR SUNSHINE_ASSETS_DIR "/shaders/directx"
  #endif

using Microsoft::WRL::ComPtr;
using namespace std::literals;

namespace sbs_bench {

  namespace fs = std::filesystem;

  namespace {

    struct rgba_image {
      UINT w = 0, h = 0;
      std::vector<uint8_t> bgra;  // tightly packed B,G,R,A rows, top-to-bottom
    };

    std::string json_string(std::string_view value) {
      static constexpr char hex[] = "0123456789abcdef";
      std::string escaped;
      escaped.reserve(value.size() + 2);
      escaped.push_back('"');
      for (unsigned char c : value) {
        switch (c) {
          case '"':
            escaped += "\\\"";
            break;
          case '\\':
            escaped += "\\\\";
            break;
          case '\b':
            escaped += "\\b";
            break;
          case '\f':
            escaped += "\\f";
            break;
          case '\n':
            escaped += "\\n";
            break;
          case '\r':
            escaped += "\\r";
            break;
          case '\t':
            escaped += "\\t";
            break;
          default:
            if (c < 0x20) {
              escaped += "\\u00";
              escaped.push_back(hex[c >> 4]);
              escaped.push_back(hex[c & 0x0f]);
            } else {
              escaped.push_back((char) c);
            }
            break;
        }
      }
      escaped.push_back('"');
      return escaped;
    }

    uint16_t float_to_half(float value) {
      uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      const uint32_t sign = (bits >> 16) & 0x8000u;
      int exp = (int) ((bits >> 23) & 0xffu) - 127 + 15;
      uint32_t mant = bits & 0x7fffffu;
      if (exp <= 0) {
        if (exp < -10) {
          return (uint16_t) sign;
        }
        mant = (mant | 0x800000u) >> (1 - exp);
        return (uint16_t) (sign | ((mant + 0x1000u) >> 13));
      }
      if (exp >= 31) {
        return (uint16_t) (sign | 0x7c00u);
      }
      mant += 0x1000u;
      if (mant & 0x800000u) {
        mant = 0;
        if (++exp >= 31) {
          return (uint16_t) (sign | 0x7c00u);
        }
      }
      return (uint16_t) (sign | ((uint32_t) exp << 10) | (mant >> 13));
    }

    float half_to_float(uint16_t h) {
      const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
      int exp = (h >> 10) & 0x1f;
      uint32_t mant = h & 0x3ffu;
      uint32_t bits;
      if (exp == 0) {
        if (mant == 0) {
          bits = sign;
        } else {
          exp = 127 - 15 + 1;
          while (!(mant & 0x400u)) {
            mant <<= 1;
            --exp;
          }
          bits = sign | ((uint32_t) exp << 23) | ((mant & 0x3ffu) << 13);
        }
      } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
      } else {
        bits = sign | ((uint32_t) (exp - 15 + 127) << 23) | (mant << 13);
      }
      float value;
      std::memcpy(&value, &bits, sizeof(value));
      return value;
    }

    float srgb_to_linear(float value) {
      return value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
    }

    void hdr_preview_bgra(float r, float g, float b, uint8_t *out) {
      r = std::max(r, 0.0f);
      g = std::max(g, 0.0f);
      b = std::max(b, 0.0f);
      const float y = std::max(0.2126f * r + 0.7152f * g + 0.0722f * b, 0.0f);
      r /= 1.0f + y;
      g /= 1.0f + y;
      b /= 1.0f + y;
      const float peak = std::max(1.0f, std::max(r, std::max(g, b)));
      r /= peak;
      g /= peak;
      b /= peak;
      auto encode = [](float c) {
        c = std::clamp(c, 0.0f, 1.0f);
        c = c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
        return (uint8_t) std::lround(std::clamp(c, 0.0f, 1.0f) * 255.0f);
      };
      out[0] = encode(b);
      out[1] = encode(g);
      out[2] = encode(r);
      out[3] = 255;
    }

    // ---- WIC PNG load/save (32bpp BGRA, matching the SDR B8G8R8A8_UNORM pipeline) ----

    ComPtr<IWICImagingFactory> g_wic;

    bool wic_init() {
      if (g_wic) {
        return true;
      }
      if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) {
        // Already initialized on this thread with another mode is fine.
      }
      return SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&g_wic)));
    }

    bool load_png(const fs::path &path, rgba_image &out) {
      ComPtr<IWICBitmapDecoder> dec;
      if (FAILED(g_wic->CreateDecoderFromFilename(path.wstring().c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &dec))) {
        return false;
      }
      ComPtr<IWICBitmapFrameDecode> frame;
      if (FAILED(dec->GetFrame(0, &frame))) {
        return false;
      }
      ComPtr<IWICFormatConverter> conv;
      if (FAILED(g_wic->CreateFormatConverter(&conv))) {
        return false;
      }
      if (FAILED(conv->Initialize(frame.Get(), GUID_WICPixelFormat32bppBGRA, WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom))) {
        return false;
      }
      if (FAILED(conv->GetSize(&out.w, &out.h))) {
        return false;
      }
      out.bgra.resize((size_t) out.w * out.h * 4);
      return SUCCEEDED(conv->CopyPixels(nullptr, out.w * 4, (UINT) out.bgra.size(), out.bgra.data()));
    }

    bool load_depth_texture(ID3D11Device *dev, const fs::path &path,
                            ComPtr<ID3D11Texture2D> &texture,
                            ComPtr<ID3D11ShaderResourceView> &srv) {
      ComPtr<IWICBitmapDecoder> dec;
      if (FAILED(g_wic->CreateDecoderFromFilename(path.wstring().c_str(), nullptr, GENERIC_READ,
                                                   WICDecodeMetadataCacheOnDemand, &dec))) {
        return false;
      }
      ComPtr<IWICBitmapFrameDecode> frame;
      ComPtr<IWICFormatConverter> conv;
      if (FAILED(dec->GetFrame(0, &frame)) || FAILED(g_wic->CreateFormatConverter(&conv)) ||
          FAILED(conv->Initialize(frame.Get(), GUID_WICPixelFormat16bppGray,
                                  WICBitmapDitherTypeNone, nullptr, 0.0,
                                  WICBitmapPaletteTypeCustom))) {
        return false;
      }
      UINT width = 0, height = 0;
      if (FAILED(conv->GetSize(&width, &height)) || !width || !height) {
        return false;
      }
      std::vector<uint16_t> gray((size_t) width * height);
      if (FAILED(conv->CopyPixels(nullptr, width * sizeof(uint16_t),
                                  (UINT) (gray.size() * sizeof(uint16_t)),
                                  (BYTE *) gray.data()))) {
        return false;
      }
      std::vector<float> depth(gray.size());
      std::transform(gray.begin(), gray.end(), depth.begin(), [](uint16_t value) {
        return value / 65535.0f;
      });
      D3D11_TEXTURE2D_DESC desc = {};
      desc.Width = width;
      desc.Height = height;
      desc.MipLevels = 1;
      desc.ArraySize = 1;
      desc.Format = DXGI_FORMAT_R32_FLOAT;
      desc.SampleDesc.Count = 1;
      desc.Usage = D3D11_USAGE_IMMUTABLE;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA data = {depth.data(), (UINT) (width * sizeof(float)), 0};
      return SUCCEEDED(dev->CreateTexture2D(&desc, &data, &texture)) &&
             SUCCEEDED(dev->CreateShaderResourceView(texture.Get(), nullptr, &srv));
    }

    bool save_png(const fs::path &path, UINT w, UINT h, const std::vector<uint8_t> &bgra) {
      ComPtr<IWICStream> stream;
      if (FAILED(g_wic->CreateStream(&stream))) {
        return false;
      }
      if (FAILED(stream->InitializeFromFilename(path.wstring().c_str(), GENERIC_WRITE))) {
        return false;
      }
      ComPtr<IWICBitmapEncoder> enc;
      if (FAILED(g_wic->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc))) {
        return false;
      }
      if (FAILED(enc->Initialize(stream.Get(), WICBitmapEncoderNoCache))) {
        return false;
      }
      ComPtr<IWICBitmapFrameEncode> fe;
      ComPtr<IPropertyBag2> props;
      if (FAILED(enc->CreateNewFrame(&fe, &props))) {
        return false;
      }
      if (FAILED(fe->Initialize(props.Get()))) {
        return false;
      }
      fe->SetSize(w, h);
      WICPixelFormatGUID fmt = GUID_WICPixelFormat32bppBGRA;
      fe->SetPixelFormat(&fmt);
      if (FAILED(fe->WritePixels(h, w * 4, (UINT) bgra.size(), const_cast<uint8_t *>(bgra.data())))) {
        return false;
      }
      return SUCCEEDED(fe->Commit()) && SUCCEEDED(enc->Commit());
    }

    bool save_gray16_png(const fs::path &path, UINT w, UINT h, const std::vector<uint16_t> &gray) {
      ComPtr<IWICStream> stream;
      if (FAILED(g_wic->CreateStream(&stream))) {
        return false;
      }
      if (FAILED(stream->InitializeFromFilename(path.wstring().c_str(), GENERIC_WRITE))) {
        return false;
      }
      ComPtr<IWICBitmapEncoder> enc;
      if (FAILED(g_wic->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc))) {
        return false;
      }
      if (FAILED(enc->Initialize(stream.Get(), WICBitmapEncoderNoCache))) {
        return false;
      }
      ComPtr<IWICBitmapFrameEncode> fe;
      ComPtr<IPropertyBag2> props;
      if (FAILED(enc->CreateNewFrame(&fe, &props))) {
        return false;
      }
      if (FAILED(fe->Initialize(props.Get()))) {
        return false;
      }
      fe->SetSize(w, h);
      WICPixelFormatGUID fmt = GUID_WICPixelFormat16bppGray;
      fe->SetPixelFormat(&fmt);
      if (FAILED(fe->WritePixels(h, w * 2, (UINT) (gray.size() * 2), (BYTE *) const_cast<uint16_t *>(gray.data())))) {
        return false;
      }
      return SUCCEEDED(fe->Commit()) && SUCCEEDED(enc->Commit());
    }

    // Read back an R32_FLOAT depth SRV and save it as a 16-bit grayscale PNG (values clamped to
    // [0,1] scaled to 0-65535). 16-bit matters: the swim metric measures frame-to-frame depth
    // deltas that sit below 1/255. The staging texture is cached across frames (constant size).
    void dump_depth(ID3D11Device *dev, ID3D11DeviceContext *ctx, ID3D11ShaderResourceView *srv, const fs::path &path, ComPtr<ID3D11Texture2D> &stage_cache) {
      if (!srv) {
        return;
      }
      ComPtr<ID3D11Resource> res;
      srv->GetResource(&res);
      ComPtr<ID3D11Texture2D> tex;
      if (FAILED(res.As(&tex))) {
        return;
      }
      D3D11_TEXTURE2D_DESC d = {};
      tex->GetDesc(&d);
      if (!stage_cache) {
        D3D11_TEXTURE2D_DESC sd = d;
        sd.Usage = D3D11_USAGE_STAGING;
        sd.BindFlags = 0;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        sd.MiscFlags = 0;
        if (FAILED(dev->CreateTexture2D(&sd, nullptr, &stage_cache))) {
          return;
        }
      }
      ctx->CopyResource(stage_cache.Get(), tex.Get());
      D3D11_MAPPED_SUBRESOURCE m = {};
      if (FAILED(ctx->Map(stage_cache.Get(), 0, D3D11_MAP_READ, 0, &m))) {
        return;
      }
      std::vector<uint16_t> gray((size_t) d.Width * d.Height);
      for (UINT y = 0; y < d.Height; y++) {
        const float *row = (const float *) ((const uint8_t *) m.pData + (size_t) y * m.RowPitch);
        for (UINT x = 0; x < d.Width; x++) {
          float v = row[x];
          v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
          gray[(size_t) y * d.Width + x] = (uint16_t) (v * 65535.0f + 0.5f);
        }
      }
      ctx->Unmap(stage_cache.Get(), 0);
      save_gray16_png(path, d.Width, d.Height, gray);
    }

    void dump_uint_mask(ID3D11Device *dev, ID3D11DeviceContext *ctx,
                        ID3D11ShaderResourceView *srv, const fs::path &path,
                        ComPtr<ID3D11Texture2D> &stage_cache) {
      if (!srv) {
        return;
      }
      ComPtr<ID3D11Resource> resource;
      srv->GetResource(&resource);
      ComPtr<ID3D11Texture2D> texture;
      if (FAILED(resource.As(&texture))) {
        return;
      }
      D3D11_TEXTURE2D_DESC desc {};
      texture->GetDesc(&desc);
      if (!stage_cache) {
        auto stage_desc = desc;
        stage_desc.Usage = D3D11_USAGE_STAGING;
        stage_desc.BindFlags = 0;
        stage_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        stage_desc.MiscFlags = 0;
        if (FAILED(dev->CreateTexture2D(&stage_desc, nullptr, &stage_cache))) {
          return;
        }
      }
      ctx->CopyResource(stage_cache.Get(), texture.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(ctx->Map(stage_cache.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        return;
      }
      std::vector<uint16_t> gray((size_t) desc.Width * desc.Height);
      for (UINT y = 0; y < desc.Height; ++y) {
        const auto *row = (const uint32_t *) ((const uint8_t *) mapped.pData +
                                              (size_t) y * mapped.RowPitch);
        for (UINT x = 0; x < desc.Width; ++x) {
          gray[(size_t) y * desc.Width + x] = row[x] ? 65535u : 0u;
        }
      }
      ctx->Unmap(stage_cache.Get(), 0);
      save_gray16_png(path, desc.Width, desc.Height, gray);
    }

    // Read back a harness-only B8G8R8A8 diagnostic target without coupling it to the stream's
    // SDR/HDR format. Disocclusion masks use R=pre-fill hole and G=still unresolved after fill.
    void dump_bgra8_texture(ID3D11Device *dev, ID3D11DeviceContext *ctx,
                            ID3D11Texture2D *texture, const fs::path &path,
                            ComPtr<ID3D11Texture2D> &stage_cache) {
      if (!texture) {
        return;
      }
      D3D11_TEXTURE2D_DESC d = {};
      texture->GetDesc(&d);
      if (!stage_cache) {
        D3D11_TEXTURE2D_DESC sd = d;
        sd.Usage = D3D11_USAGE_STAGING;
        sd.BindFlags = 0;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        sd.MiscFlags = 0;
        if (FAILED(dev->CreateTexture2D(&sd, nullptr, &stage_cache))) {
          return;
        }
      }
      ctx->CopyResource(stage_cache.Get(), texture);
      D3D11_MAPPED_SUBRESOURCE mapped = {};
      if (FAILED(ctx->Map(stage_cache.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        return;
      }
      std::vector<uint8_t> pixels((size_t) d.Width * d.Height * 4);
      for (UINT y = 0; y < d.Height; ++y) {
        std::memcpy(pixels.data() + (size_t) y * d.Width * 4,
                    (const uint8_t *) mapped.pData + (size_t) y * mapped.RowPitch,
                    (size_t) d.Width * 4);
      }
      ctx->Unmap(stage_cache.Get(), 0);
      save_png(path, d.Width, d.Height, pixels);
    }

    // Preserve the harness-only R32_FLOAT mapping target without quantization. Rows are
    // written tightly packed even though D3D11 staging resources may have a padded RowPitch.
    // Windows/D3D11 targets are little-endian, matching the sidecar's declared float32-le dtype.
    bool dump_float_texture(ID3D11Device *dev, ID3D11DeviceContext *ctx,
                            ID3D11Texture2D *texture, const fs::path &path,
                            ComPtr<ID3D11Texture2D> &stage_cache) {
      if (!texture) {
        return false;
      }
      D3D11_TEXTURE2D_DESC desc {};
      texture->GetDesc(&desc);
      if (desc.Format != DXGI_FORMAT_R32_FLOAT) {
        return false;
      }
      if (!stage_cache) {
        auto stage_desc = desc;
        stage_desc.Usage = D3D11_USAGE_STAGING;
        stage_desc.BindFlags = 0;
        stage_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        stage_desc.MiscFlags = 0;
        if (FAILED(dev->CreateTexture2D(&stage_desc, nullptr, &stage_cache))) {
          return false;
        }
      }
      ctx->CopyResource(stage_cache.Get(), texture);
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(ctx->Map(stage_cache.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        return false;
      }
      std::ofstream out(path, std::ios::binary);
      const std::streamsize row_bytes = (std::streamsize) desc.Width * sizeof(float);
      if (out) {
        for (UINT y = 0; y < desc.Height; ++y) {
          out.write((const char *) mapped.pData + (size_t) y * mapped.RowPitch, row_bytes);
        }
      }
      out.flush();
      const bool succeeded = out.good();
      ctx->Unmap(stage_cache.Get(), 0);
      return succeeded;
    }

    // Preserve the exact raw model output for stage-by-stage parity checks. Unlike the display
    // PNG, this is not clamped or normalized: it is row-major float32, width*height values.
    void dump_raw_model_depth(ID3D11Device *dev, ID3D11DeviceContext *ctx, ID3D11ShaderResourceView *srv, int width, int height, const fs::path &path, ComPtr<ID3D11Buffer> &stage_cache) {
      if (!srv || width <= 0 || height <= 0) {
        return;
      }
      ComPtr<ID3D11Resource> res;
      srv->GetResource(&res);
      ComPtr<ID3D11Buffer> buf;
      if (FAILED(res.As(&buf))) {
        return;
      }
      D3D11_BUFFER_DESC d = {};
      buf->GetDesc(&d);
      if (!stage_cache) {
        D3D11_BUFFER_DESC sd = d;
        sd.Usage = D3D11_USAGE_STAGING;
        sd.BindFlags = 0;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        sd.MiscFlags = 0;
        if (FAILED(dev->CreateBuffer(&sd, nullptr, &stage_cache))) {
          return;
        }
      }
      ctx->CopyResource(stage_cache.Get(), buf.Get());
      D3D11_MAPPED_SUBRESOURCE m = {};
      if (FAILED(ctx->Map(stage_cache.Get(), 0, D3D11_MAP_READ, 0, &m))) {
        return;
      }
      std::ofstream out(path, std::ios::binary);
      if (out) {
        out.write((const char *) m.pData, (std::streamsize) width * height * sizeof(float));
      }
      ctx->Unmap(stage_cache.Get(), 0);
    }

    // Keep output identities tied to source identities. Positional renumbering made a dropped
    // source frame silently shift every depth/SBS/source comparison by one.
    std::string source_frame_id(const fs::path &path) {
      std::string stem = path.stem().string();
      size_t split = stem.find_last_of('_');
      std::string id = split == std::string::npos ? "" : stem.substr(split + 1);
      if (!id.empty() && std::all_of(id.begin(), id.end(), [](unsigned char c) {
            return std::isdigit(c);
          })) {
        return id;
      }
      return {};
    }

    bool numeric_frame_less(const fs::path &left, const fs::path &right) {
      auto left_id = source_frame_id(left);
      auto right_id = source_frame_id(right);
      if (!left_id.empty() && !right_id.empty()) {
        auto trim_zeroes = [](const std::string &value) {
          const auto first = value.find_first_not_of('0');
          return first == std::string::npos ? std::string_view(value).substr(value.size() - 1) :
                                             std::string_view(value).substr(first);
        };
        const auto left_number = trim_zeroes(left_id);
        const auto right_number = trim_zeroes(right_id);
        if (left_number.size() != right_number.size()) {
          return left_number.size() < right_number.size();
        }
        if (left_number != right_number) {
          return left_number < right_number;
        }
      } else if (left_id.empty() != right_id.empty()) {
        return !left_id.empty();
      }
      return left.filename().string() < right.filename().string();
    }

    std::string frame_id(const fs::path &path, size_t fallback) {
      auto id = source_frame_id(path);
      if (!id.empty()) {
        return id;
      }
      char buf[16];
      snprintf(buf, sizeof(buf), "%05zu", fallback);
      return buf;
    }

    // ---- D3D helpers ----

    ComPtr<ID3DBlob> compile(const char *file, const char *entry, const char *model) {
      std::wstring wfile(file, file + strlen(file));
      ComPtr<ID3DBlob> blob, err;
      HRESULT hr = D3DCompileFromFile(wfile.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, entry, model, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &err);
      if (FAILED(hr)) {
        BOOST_LOG(error) << "sbs-bench: shader compile failed [" << file << "]: "
                         << (err ? (const char *) err->GetBufferPointer() : "?");
        return nullptr;
      }
      return blob;
    }

    template<int N>
    ComPtr<ID3D11Buffer> const_buffer(ID3D11Device *dev, const float (&params)[N]) {
      static_assert(N % 4 == 0, "cbuffer must be 16-byte aligned");
      D3D11_BUFFER_DESC bd = {};
      bd.ByteWidth = N * 4;  // 16-byte aligned
      bd.Usage = D3D11_USAGE_IMMUTABLE;
      bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      D3D11_SUBRESOURCE_DATA sd = {params, 0, 0};
      ComPtr<ID3D11Buffer> b;
      dev->CreateBuffer(&bd, &sd, &b);
      return b;
    }

    // ---- argument parsing ----

    struct opts {
      std::string frames, out, model, depth_override_root;
      int eye_w = 0;  // 0 -> derive from source aspect; set with eye_h to test letterboxing
      int eye_h = 0;  // 0 -> match/derive from the input frame
      double output_scale = 1.0;  // per-eye linear scale vs source; preserves source aspect
      double pop_strength = -1.0;  // final shared stereo-parallax multiplier; <0 = conf
      double adaptive_pop_max = -1.0;  // absolute ceiling; <0 = conf
      std::string zero_plane;  // empty = conf; legacy, subject, median, or background
      bool simulate_hdr = false;  // decode sRGB frames into linear scRGB FP16 and use HDR paths
      double hdr_scale = 4.0;  // scRGB multiplier after sRGB EOTF (4.0 = 320-nit diffuse white)
      int max_width = 0;  // 0 -> use config max_encode_width
      int limit = 0;  // 0 -> all
      int output_every = 1;  // process every input for temporal state; dump only every Nth
      int depth_every = 1;  // infer every Nth source frame; reuse depth between updates
      // Apollo depth-pipeline A/B levers; <0 / false -> use the conf's value.
      double subject_lock = -1.0;  // subject anchor strength override (e.g. 0.95)
      double subject_recenter = -1.0;  // global subject recenter override
      int depth_short_side = 0;  // depth inference short-side override (0 = conf)
      double ema = -1.0;  // per-pixel depth EMA override (1.0 = off)
      double ema_edge_change = -1.0;
      double ema_edge_gradient = -1.0;
      double ema_edge_strength = -1.0;
      int subject_stretch = -1;  // -1 = conf, 0 = off, 1 = on
      double minmax_ema = -1.0;  // range-bounds EMA new-weight; <0 = conf
      int cuda_graph = -1;  // -1 = conf, 0 = ordinary enqueue, 1 = CUDA graph replay
      int adaptive_pop = -1;  // -1 = conf, 0 = off, 1 = on
      bool literal_bestv2 = false;  // reference-only: disable production resolution/pop scaling
      bool depth_override_all = false;  // reference-only: replace every inferred depth frame
    };

    bool parse_opts(int argc, char **argv, opts &o) {
      for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char *n) -> std::string {
          if (i + 1 >= argc) {
            BOOST_LOG(error) << "sbs-bench: " << n << " needs a value";
            return "";
          }
          return argv[++i];
        };
        if (a == "--frames") {
          o.frames = next("--frames");
        } else if (a == "--out") {
          o.out = next("--out");
        } else if (a == "--model") {
          o.model = next("--model");
        } else if (a == "--eye-w") {
          o.eye_w = std::stoi(next("--eye-w"));
        } else if (a == "--eye-h") {
          o.eye_h = std::stoi(next("--eye-h"));
        } else if (a == "--output-scale") {
          o.output_scale = std::stod(next("--output-scale"));
        } else if (a == "--pop-strength") {
          o.pop_strength = std::stod(next("--pop-strength"));
        } else if (a == "--adaptive-pop") {
          o.adaptive_pop = 1;
        } else if (a == "--no-adaptive-pop") {
          o.adaptive_pop = 0;
        } else if (a == "--adaptive-pop-max") {
          o.adaptive_pop_max = std::stod(next("--adaptive-pop-max"));
        } else if (a == "--zero-plane") {
          o.zero_plane = next("--zero-plane");
        } else if (a == "--simulate-hdr") {
          o.simulate_hdr = true;
        } else if (a == "--hdr-scale") {
          o.hdr_scale = std::stod(next("--hdr-scale"));
        } else if (a == "--max-width") {
          o.max_width = std::stoi(next("--max-width"));
        } else if (a == "--limit") {
          o.limit = std::stoi(next("--limit"));
        } else if (a == "--output-every") {
          o.output_every = std::max(1, std::stoi(next("--output-every")));
        } else if (a == "--depth-every") {
          o.depth_every = std::stoi(next("--depth-every"));
        } else if (a == "--depth-override-root") {
          o.depth_override_root = next("--depth-override-root");
        } else if (a == "--depth-override-all") {
          o.depth_override_all = true;
        } else if (a == "--subject-lock") {
          o.subject_lock = std::stod(next("--subject-lock"));
        } else if (a == "--subject-recenter") {
          o.subject_recenter = std::stod(next("--subject-recenter"));
        } else if (a == "--depth-short-side") {
          o.depth_short_side = std::stoi(next("--depth-short-side"));
        } else if (a == "--subject-stretch") {
          o.subject_stretch = 1;
        } else if (a == "--no-subject-stretch") {
          o.subject_stretch = 0;
        } else if (a == "--ema") {
          o.ema = std::stod(next("--ema"));
        } else if (a == "--ema-edge-change") {
          o.ema_edge_change = std::stod(next("--ema-edge-change"));
        } else if (a == "--ema-edge-gradient") {
          o.ema_edge_gradient = std::stod(next("--ema-edge-gradient"));
        } else if (a == "--ema-edge-strength") {
          o.ema_edge_strength = std::stod(next("--ema-edge-strength"));
        } else if (a == "--minmax-ema") {
          o.minmax_ema = std::stod(next("--minmax-ema"));
        } else if (a == "--cuda-graph") {
          std::string v = next("--cuda-graph");
          if (v == "on" || v == "1" || v == "true") {
            o.cuda_graph = 1;
          } else if (v == "off" || v == "0" || v == "false") {
            o.cuda_graph = 0;
          } else {
            BOOST_LOG(error) << "sbs-bench: --cuda-graph must be on or off";
            return false;
          }
        } else if (a == "--literal-bestv2") {
          o.literal_bestv2 = true;
        } else {
          BOOST_LOG(error) << "sbs-bench: unknown arg '" << a << "'";
          return false;
        }
      }
      if (o.frames.empty() || o.out.empty()) {
        BOOST_LOG(error) << "sbs-bench: --frames DIR and --out DIR are required";
        return false;
      }
      if (!(o.output_scale > 0.0 && o.output_scale <= 4.0)) {
        BOOST_LOG(error) << "sbs-bench: --output-scale must be greater than 0 and at most 4";
        return false;
      }
      if (o.pop_strength >= 0.0 && !(o.pop_strength >= 0.25 && o.pop_strength <= 2.0)) {
        BOOST_LOG(error) << "sbs-bench: --pop-strength must be between 0.25 and 2";
        return false;
      }
      if (o.adaptive_pop_max >= 0.0 &&
          !(o.adaptive_pop_max >= 0.25 && o.adaptive_pop_max <= 2.0)) {
        BOOST_LOG(error) << "sbs-bench: --adaptive-pop-max must be between 0.25 and 2";
        return false;
      }
      if (!o.zero_plane.empty() && o.zero_plane != "legacy" &&
          o.zero_plane != "subject" && o.zero_plane != "median" &&
          o.zero_plane != "background") {
        BOOST_LOG(error) << "sbs-bench: --zero-plane must be legacy, subject, median, or background";
        return false;
      }
      if (!(o.hdr_scale > 0.0 && o.hdr_scale <= 64.0)) {
        BOOST_LOG(error) << "sbs-bench: --hdr-scale must be greater than 0 and at most 64";
        return false;
      }
      if (o.depth_every < 1 || o.depth_every > 8) {
        BOOST_LOG(error) << "sbs-bench: --depth-every must be between 1 and 8";
        return false;
      }
      if (o.ema_edge_change > 1.0 || o.ema_edge_gradient > 1.0 || o.ema_edge_strength > 1.0) {
        BOOST_LOG(error) << "sbs-bench: EMA edge thresholds and strength must be <=1";
        return false;
      }
      if (!o.depth_override_root.empty() && !fs::is_directory(o.depth_override_root)) {
        BOOST_LOG(error) << "sbs-bench: --depth-override-root is not a directory";
        return false;
      }
      if (o.depth_override_all && o.depth_override_root.empty()) {
        BOOST_LOG(error) << "sbs-bench: --depth-override-all requires --depth-override-root";
        return false;
      }
      if (o.depth_override_all && o.depth_every != 1) {
        BOOST_LOG(error) << "sbs-bench: --depth-override-all requires --depth-every 1";
        return false;
      }
      return true;
    }

    config::depth_model_info pick_model(const opts &o) {
      const auto &reg = config::depth_model_registry();
      std::string want = o.model;
      if (!want.empty()) {
        for (const auto &m : reg) {
          if (m.name == want) {
            return m;
          }
        }
        BOOST_LOG(warning) << "sbs-bench: model '" << want << "' not in registry; using active model";
      }
      return video::active_depth_model();
    }

  }  // namespace

  int run(int argc, char **argv) {
    opts o;
    if (!parse_opts(argc, argv, o)) {
      return 2;
    }
    if (!wic_init()) {
      BOOST_LOG(error) << "sbs-bench: WIC init failed";
      return 3;
    }

    // Collect + sort input frames (png/jpg; WIC decodes both). Small pre-resized JPEG clips keep
    // the repo light; the harness never resizes them -- the SBS output tracks the input size.
    std::vector<fs::path> frames;
    std::error_code ec;
    for (auto &e : fs::directory_iterator(o.frames, ec)) {
      if (!e.is_regular_file()) {
        continue;
      }
      auto ext = e.path().extension().string();
      for (auto &ch : ext) {
        ch = (char) tolower((unsigned char) ch);
      }
      if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
        frames.push_back(e.path());
      }
    }
    std::sort(frames.begin(), frames.end(), numeric_frame_less);
    if (o.limit > 0 && (int) frames.size() > o.limit) {
      frames.resize(o.limit);
    }
    if (frames.empty()) {
      BOOST_LOG(error) << "sbs-bench: no png/jpg frames in " << o.frames;
      return 4;
    }
    fs::create_directories(o.out, ec);

    // Inherit the loaded config, then pin one depth update per source frame. The benchmark is
    // frame-driven rather than wall-clock-driven, so cadence throttling would make the result
    // depend on machine speed.
    auto sbs_cfg = config::video.sbs;
    if (o.pop_strength >= 0.0) {
      sbs_cfg.pop_strength = o.pop_strength;
    }
    if (o.adaptive_pop >= 0) {
      sbs_cfg.adaptive_pop = (o.adaptive_pop != 0);
    }
    if (o.adaptive_pop_max >= 0.0) {
      sbs_cfg.adaptive_pop_max = o.adaptive_pop_max;
    }
    if (!o.zero_plane.empty()) {
      sbs_cfg.zero_plane = o.zero_plane;
    }
    sbs_cfg.adaptive_pop_max = std::max(sbs_cfg.adaptive_pop_max, sbs_cfg.pop_strength);
    if (o.subject_lock >= 0.0) {
      sbs_cfg.subject_lock = o.subject_lock;
    }
    if (o.subject_recenter >= 0.0) {
      sbs_cfg.subject_recenter = o.subject_recenter;
    }
    if (o.depth_short_side > 0) {
      sbs_cfg.depth_short_side = o.depth_short_side;
    }
    if (o.subject_stretch >= 0) {
      sbs_cfg.subject_stretch = (o.subject_stretch != 0);
    }
    if (o.ema > 0.0) {
      sbs_cfg.ema = o.ema;  // A/B lever: depth EMA (1.0 = off)
    }
    if (o.ema_edge_change >= 0.0) {
      sbs_cfg.ema_edge_change = o.ema_edge_change;
    }
    if (o.ema_edge_gradient >= 0.0) {
      sbs_cfg.ema_edge_gradient = o.ema_edge_gradient;
    }
    if (o.ema_edge_strength >= 0.0) {
      sbs_cfg.ema_edge_strength = o.ema_edge_strength;
    }
    if (o.minmax_ema >= 0.0) {
      sbs_cfg.minmax_ema = o.minmax_ema;
    }
    if (o.cuda_graph >= 0) {
      sbs_cfg.cuda_graph = (o.cuda_graph != 0);
    }
    sbs_cfg.perf_stats = true;  // the harness always measures
    sbs_perf::set_enabled(true);
    sbs_perf::reset();
    auto model = pick_model(o);
    const int max_width = o.max_width > 0 ? o.max_width : config::video.sbs.max_encode_width;

    BOOST_LOG(info) << "sbs-bench: " << frames.size() << " frames, model '" << model.name
                    << "', eye " << (o.eye_w > 0 ? std::to_string(o.eye_w) : "auto") << 'x'
                    << (o.eye_h > 0 ? std::to_string(o.eye_h) : "auto")
                    << ", depth_step "
                    << (o.depth_every == 1 ? std::string("current-once") :
                                             "reuse-" + std::to_string(o.depth_every))
                    << ", profile " << sbs_cfg.profile
                    << ", literal_bestv2 " << (o.literal_bestv2 ? "on" : "off")
                    << ", depth_every " << o.depth_every
                    << " -> " << o.out;

    // ---- D3D device + shaders ----
    ComPtr<ID3D11Device> dev;
    ComPtr<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL fl;
    const D3D_FEATURE_LEVEL want_fl[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT, want_fl, 2, D3D11_SDK_VERSION, &dev, &fl, &ctx))) {
      BOOST_LOG(error) << "sbs-bench: D3D11CreateDevice failed";
      return 5;
    }

    // Harness-only GPU timestamps. CPU submission time is not a useful warp-cost measurement.
    ComPtr<ID3D11Query> warp_disjoint, warp_start, warp_end;
    D3D11_QUERY_DESC qd = {D3D11_QUERY_TIMESTAMP_DISJOINT, 0};
    dev->CreateQuery(&qd, &warp_disjoint);
    qd.Query = D3D11_QUERY_TIMESTAMP;
    dev->CreateQuery(&qd, &warp_start);
    dev->CreateQuery(&qd, &warp_end);

    auto vs_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_reprojection_vs.hlsl", "main_vs", "vs_5_0");
    auto ps_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_reprojection_ps.hlsl", "main_ps", "ps_5_0");
    auto mask_ps_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_reprojection_ps.hlsl", "mask_ps", "ps_5_0");
    auto mapping_ps_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_reprojection_ps.hlsl", "mapping_ps", "ps_5_0");
    auto coverage_cs_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_forward_coverage_cs.hlsl", "main", "cs_5_0");
    auto warp_prefilter_cs_blob = compile(SUNSHINE_SHADERS_DIR "/depth_warp_prefilter_cs.hlsl", "main", "cs_5_0");
    if (!vs_blob || !ps_blob || !mask_ps_blob || !mapping_ps_blob || !coverage_cs_blob ||
        !warp_prefilter_cs_blob) {
      return 6;
    }
    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> ps, mask_ps, mapping_ps;
    ComPtr<ID3D11ComputeShader> coverage_cs, warp_prefilter_cs;
    if (FAILED(dev->CreateVertexShader(
          vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr, &vs)) ||
        FAILED(dev->CreatePixelShader(
          ps_blob->GetBufferPointer(), ps_blob->GetBufferSize(), nullptr, &ps)) ||
        FAILED(dev->CreatePixelShader(
          mask_ps_blob->GetBufferPointer(), mask_ps_blob->GetBufferSize(), nullptr, &mask_ps)) ||
        FAILED(dev->CreatePixelShader(mapping_ps_blob->GetBufferPointer(),
          mapping_ps_blob->GetBufferSize(), nullptr, &mapping_ps)) ||
        FAILED(dev->CreateComputeShader(coverage_cs_blob->GetBufferPointer(),
          coverage_cs_blob->GetBufferSize(), nullptr, &coverage_cs)) ||
        FAILED(dev->CreateComputeShader(warp_prefilter_cs_blob->GetBufferPointer(),
          warp_prefilter_cs_blob->GetBufferSize(), nullptr, &warp_prefilter_cs)) ||
        !vs || !ps || !mask_ps || !mapping_ps || !coverage_cs || !warp_prefilter_cs) {
      BOOST_LOG(error) << "sbs-bench: D3D11 shader creation failed";
      return 6;
    }

    ComPtr<ID3D11SamplerState> sampler;
    {
      D3D11_SAMPLER_DESC sd = {};
      sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
      sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
      sd.ComparisonFunc = D3D11_COMPARISON_NEVER;
      sd.MaxLOD = D3D11_FLOAT32_MAX;
      dev->CreateSamplerState(&sd, &sampler);
    }

    // Built after the first source frame reveals the source/output aspect relationship.
    ComPtr<ID3D11Buffer> repro_cb;

    // ---- estimator ----
    models::video_depth_estimator estimator(dev, ctx, fs::path(SUNSHINE_ASSETS_DIR), sbs_cfg, model);

    // Per-run state built lazily on the first frame (once we know the input size).
    ComPtr<ID3D11Texture2D> sbs_tex, sbs_stage;
    ComPtr<ID3D11RenderTargetView> sbs_rtv;
    ComPtr<ID3D11ShaderResourceView> sbs_srv;
    ComPtr<ID3D11Texture2D> warp_mask_tex, warp_mask_stage;
    ComPtr<ID3D11RenderTargetView> warp_mask_rtv;
    ComPtr<ID3D11Texture2D> warp_mapping_tex, warp_mapping_stage;
    ComPtr<ID3D11RenderTargetView> warp_mapping_rtv;
    ComPtr<ID3D11Texture2D> coverage_tex;
    ComPtr<ID3D11UnorderedAccessView> coverage_uav;
    ComPtr<ID3D11ShaderResourceView> coverage_srv;
    D3D11_VIEWPORT vp = {};
    UINT sbs_w = 0, sbs_h = 0;
    ComPtr<ID3D11Texture2D> depth_stage;  // dump_depth staging cache (depth size is constant)
    ComPtr<ID3D11Texture2D> warp_depth_tex;
    ComPtr<ID3D11UnorderedAccessView> warp_depth_uav;
    ComPtr<ID3D11ShaderResourceView> warp_depth_srv;
    ComPtr<ID3D11Texture2D> ema_mask_stage;
    ComPtr<ID3D11Buffer> raw_depth_stage;
    bool raw_shape_written = false;
    bool warp_mapping_shape_written = false;
    float hdr_output_min = std::numeric_limits<float>::infinity();
    float hdr_output_max = -std::numeric_limits<float>::infinity();
    uint64_t hdr_nonfinite = 0;

    int written = 0;
    models::estimate_result est;
    bool cuda_graph_captured = false;
    bool have_depth_result = false;
    UINT source_width = 0;
    UINT source_height = 0;
    const fs::path depth_override_dir = o.depth_override_root.empty() ? fs::path() :
                                          fs::path(o.depth_override_root) / fs::path(o.frames).filename();
    size_t expected_depth_override_frames = 0;
    size_t applied_depth_override_frames = 0;
    if (!depth_override_dir.empty()) {
      if (!fs::is_directory(depth_override_dir)) {
        BOOST_LOG(error) << "sbs-bench: depth-override clip directory is missing: "
                         << depth_override_dir;
        return 7;
      }
      for (size_t fi = 0; fi < frames.size(); ++fi) {
        expected_depth_override_frames += o.depth_override_all ||
                                          (fi % (size_t) o.depth_every) != 0;
      }
      size_t actual_depth_override_frames = 0;
      for (const auto &entry : fs::directory_iterator(depth_override_dir)) {
        const auto filename = entry.path().filename().string();
        if (entry.is_regular_file() && filename.starts_with("depth_") &&
            entry.path().extension() == ".png") {
          ++actual_depth_override_frames;
        }
      }
      if (actual_depth_override_frames != expected_depth_override_frames) {
        BOOST_LOG(error) << "sbs-bench: expected " << expected_depth_override_frames
                         << " depth overrides in " << depth_override_dir << ", found "
                         << actual_depth_override_frames;
        return 7;
      }
    }
    for (size_t fi = 0; fi < frames.size(); fi++) {
      rgba_image img;
      if (!load_png(frames[fi], img)) {
        BOOST_LOG(warning) << "sbs-bench: skip " << frames[fi];
        continue;
      }
      if (source_width == 0) {
        source_width = img.w;
        source_height = img.h;
      } else if (img.w != source_width || img.h != source_height) {
        BOOST_LOG(error) << "sbs-bench: mixed source dimensions are not a valid clip: first frame "
                         << source_width << "x" << source_height << ", " << frames[fi]
                         << " is " << img.w << "x" << img.h;
        return 9;
      }
      const std::string output_id = frame_id(frames[fi], fi);

      // Input texture + SRV.
      D3D11_TEXTURE2D_DESC id = {};
      id.Width = img.w;
      id.Height = img.h;
      id.MipLevels = 1;
      id.ArraySize = 1;
      id.Format = o.simulate_hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_B8G8R8A8_UNORM;
      id.SampleDesc.Count = 1;
      id.Usage = D3D11_USAGE_IMMUTABLE;
      id.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      std::vector<uint16_t> hdr_rgba;
      const void *input_pixels = img.bgra.data();
      UINT input_pitch = img.w * 4;
      if (o.simulate_hdr) {
        hdr_rgba.resize((size_t) img.w * img.h * 4);
        for (size_t p = 0; p < (size_t) img.w * img.h; ++p) {
          const float b = srgb_to_linear(img.bgra[p * 4 + 0] / 255.0f) * (float) o.hdr_scale;
          const float g = srgb_to_linear(img.bgra[p * 4 + 1] / 255.0f) * (float) o.hdr_scale;
          const float r = srgb_to_linear(img.bgra[p * 4 + 2] / 255.0f) * (float) o.hdr_scale;
          hdr_rgba[p * 4 + 0] = float_to_half(r);
          hdr_rgba[p * 4 + 1] = float_to_half(g);
          hdr_rgba[p * 4 + 2] = float_to_half(b);
          hdr_rgba[p * 4 + 3] = float_to_half(1.0f);
        }
        input_pixels = hdr_rgba.data();
        input_pitch = img.w * 8;
      }
      D3D11_SUBRESOURCE_DATA isd = {input_pixels, input_pitch, 0};
      ComPtr<ID3D11Texture2D> in_tex;
      if (FAILED(dev->CreateTexture2D(&id, &isd, &in_tex))) {
        BOOST_LOG(error) << "sbs-bench: input tex fail";
        continue;
      }
      ComPtr<ID3D11ShaderResourceView> in_srv;
      dev->CreateShaderResourceView(in_tex.Get(), nullptr, &in_srv);

      // First frame: size the SBS target. Per eye = the input resolution by default (so the clip
      // size, not a fixed constant, drives eval cost); --eye-h pins a specific output height.
      // The width is still capped at max_encode_width like the live path.
      if (!sbs_tex) {
        int eh_target = o.eye_h > 0 ? o.eye_h :
                                      std::max(2, (int) std::lround((double) img.h * o.output_scale));
        float aspect = (float) img.w / (float) img.h;
        int eye_w = o.eye_w > 0 ? o.eye_w : (o.eye_h > 0 ? std::max(1, (int) std::lround(eh_target * aspect)) : std::max(1, (int) std::lround((double) img.w * o.output_scale)));
        int eye_h = eh_target;
        if (o.eye_w > 0 && o.eye_h <= 0) {
          eye_h = std::max(1, (int) std::lround(eye_w / aspect));
        }
        if (2 * eye_w > max_width) {
          const double scale = (double) max_width / (double) (2 * eye_w);
          eye_w = std::max(1, max_width / 2);
          eye_h = std::max(2, ((int) std::lround(eh_target * scale)) & ~1);
        }
        sbs_w = (UINT) (2 * eye_w);
        sbs_h = (UINT) eye_h;
        const float eye_aspect = (float) eye_w / (float) eye_h;
        const float content_scale_x = eye_aspect > aspect ? aspect / eye_aspect : 1.0f;
        const float content_scale_y = eye_aspect < aspect ? eye_aspect / aspect : 1.0f;
        float repro_params[8] = {
          (float) sbs_cfg.subject_lock,
          sbs_cfg.subject_stretch ? 1.0f : 0.0f,
          content_scale_x,
          content_scale_y,
          (float) sbs_cfg.pop_strength,
          o.literal_bestv2 ? 1.0f : 0.0f,
          sbs_cfg.adaptive_pop ? 1.0f : 0.0f,
          (float) sbs_cfg.adaptive_pop_max
        };
        repro_cb = const_buffer(dev.Get(), repro_params);
        D3D11_TEXTURE2D_DESC td = {};
        td.Width = sbs_w;
        td.Height = sbs_h;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = o.simulate_hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        dev->CreateTexture2D(&td, nullptr, &sbs_tex);
        dev->CreateRenderTargetView(sbs_tex.Get(), nullptr, &sbs_rtv);
        dev->CreateShaderResourceView(sbs_tex.Get(), nullptr, &sbs_srv);
        D3D11_TEXTURE2D_DESC mask_desc = td;
        mask_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        mask_desc.BindFlags = D3D11_BIND_RENDER_TARGET;
        if (FAILED(dev->CreateTexture2D(&mask_desc, nullptr, &warp_mask_tex)) ||
            FAILED(dev->CreateRenderTargetView(warp_mask_tex.Get(), nullptr, &warp_mask_rtv))) {
          BOOST_LOG(error) << "sbs-bench: warp-mask texture creation failed";
          return 6;
        }
        D3D11_TEXTURE2D_DESC mapping_desc = td;
        mapping_desc.Format = DXGI_FORMAT_R32_FLOAT;
        mapping_desc.BindFlags = D3D11_BIND_RENDER_TARGET;
        if (FAILED(dev->CreateTexture2D(&mapping_desc, nullptr, &warp_mapping_tex)) ||
            FAILED(dev->CreateRenderTargetView(warp_mapping_tex.Get(), nullptr,
                                               &warp_mapping_rtv))) {
          BOOST_LOG(error) << "sbs-bench: warp-mapping texture creation failed";
          return 6;
        }
        {
          D3D11_TEXTURE2D_DESC wd = td;
          wd.Format = DXGI_FORMAT_R32_UINT;
          wd.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
          if (FAILED(dev->CreateTexture2D(&wd, nullptr, &coverage_tex)) || FAILED(dev->CreateUnorderedAccessView(coverage_tex.Get(), nullptr, &coverage_uav)) || FAILED(dev->CreateShaderResourceView(coverage_tex.Get(), nullptr, &coverage_srv))) {
            BOOST_LOG(error) << "sbs-bench: forward-coverage texture creation failed";
            return 6;
          }
        }
        D3D11_TEXTURE2D_DESC sd2 = td;
        sd2.Usage = D3D11_USAGE_STAGING;
        sd2.BindFlags = 0;
        sd2.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        dev->CreateTexture2D(&sd2, nullptr, &sbs_stage);
        vp = {0, 0, (float) sbs_w, (float) sbs_h, 0, 1};
        std::ofstream mapping_shape(fs::path(o.out) / "warp_map_shape.json");
        if (!mapping_shape) {
          BOOST_LOG(error) << "sbs-bench: cannot write warp_map_shape.json";
          return 6;
        }
        mapping_shape.imbue(std::locale::classic());
        mapping_shape << std::setprecision(std::numeric_limits<float>::max_digits10);
        mapping_shape
          << "{\n"
          << "  \"schema\": 1,\n"
          << "  \"width\": " << sbs_w << ",\n"
          << "  \"height\": " << sbs_h << ",\n"
          << "  \"eye_width\": " << eye_w << ",\n"
          << "  \"eye_height\": " << eye_h << ",\n"
          << "  \"source_width\": " << img.w << ",\n"
          << "  \"source_height\": " << img.h << ",\n"
          << "  \"content_scale_x\": " << content_scale_x << ",\n"
          << "  \"content_scale_y\": " << content_scale_y << ",\n"
          << "  \"dtype\": \"float32-le\",\n"
          << "  \"layout\": \"row-major\",\n"
          << "  \"channels\": [\n"
          << "    \"raw_reproject_source_u_normalized\"\n"
          << "  ],\n"
          << "  \"validity\": {\"content\": \"derive from content_scale_x/content_scale_y and packed output coordinate\", \"forward_coverage\": \"warp_mask_<frame-id>.png red == 0 inside content\"},\n"
          << "  \"live_sample_source_u_normalized\": \"clamp(raw_reproject_source_u_normalized, 0, 1)\",\n"
          << "  \"derived_inverse_displacement_output_eye_px\": \"(raw_reproject_source_u_normalized - aspect_fitted_unwarped_source_u) * content_scale_x * eye_width\",\n"
          << "  \"derived_signed_binocular_disparity_px\": \"invert both eye maps at common source-U samples; x_right - x_left\"\n"
          << "}\n";
        mapping_shape.flush();
        if (!mapping_shape.good()) {
          BOOST_LOG(error) << "sbs-bench: failed writing warp_map_shape.json";
          return 6;
        }
        warp_mapping_shape_written = true;
        BOOST_LOG(info) << "sbs-bench: input " << img.w << "x" << img.h << " -> SBS "
                        << sbs_w << "x" << sbs_h
                        << (o.simulate_hdr ? " (linear scRGB FP16 HDR simulation)" : " (sRGB SDR)");
      }

      // Submit and consume exactly one inference for this source frame.
      const auto input_color = o.simulate_hdr ? models::input_color_space::scrgb_hdr : models::input_color_space::srgb;
      if (!have_depth_result || (fi % (size_t) o.depth_every) == 0) {
        estimator.estimate_depth(in_srv.Get(), input_color, (std::uint64_t) fi);
        est = estimator.finish_pending_depth_for_evaluation(input_color);
        cuda_graph_captured = cuda_graph_captured || est.cuda_graph_active;
        have_depth_result = true;
      } else {
        // Match the live stream between depth ticks: color advances while all depth-derived
        // geometry remains the last completed result. The views remain owned by the estimator
        // and are valid until the next inference overwrites their backing resources.
      }

      // Offline motion-compensation reference: replace only explicitly supplied held-depth
      // frames, while retaining the estimator's real subject/convergence state and production
      // warp. This keeps the experiment on the actual shader path without pretending the Python
      // flow prototype is production code.
      ComPtr<ID3D11Texture2D> override_depth_texture;
      ComPtr<ID3D11ShaderResourceView> override_depth_srv;
      if (!depth_override_dir.empty() &&
          (o.depth_override_all || (fi % (size_t) o.depth_every) != 0)) {
        const fs::path override_path = depth_override_dir / ("depth_" + output_id + ".png");
        if (!fs::exists(override_path) ||
            !load_depth_texture(dev.Get(), override_path, override_depth_texture,
                                override_depth_srv)) {
          BOOST_LOG(error) << "sbs-bench: missing or invalid depth override " << override_path;
          return 7;
        }
        est.depth = override_depth_srv;
        ++applied_depth_override_frames;
      }

      // Sampling output must never sample the depth/EMA/subject pipeline itself. Every source
      // frame above was inferred and consumed; only expensive composite/readback is skipped.
      if ((fi % (size_t) o.output_every) != 0) {
        continue;
      }

      // Composite (mirrors display_vram::convert()'s SBS block): probe reprojection.
      const auto comp_t0 = std::chrono::steady_clock::now();
      const bool time_warp = warp_disjoint && warp_start && warp_end;
      if (time_warp) {
        ctx->Begin(warp_disjoint.Get());
        ctx->End(warp_start.Get());
      }
      ID3D11ShaderResourceView *warp_depth = est.depth.Get();
      if (est.depth) {
        ComPtr<ID3D11Resource> depth_resource;
        est.depth->GetResource(&depth_resource);
        ComPtr<ID3D11Texture2D> depth_texture;
        if (FAILED(depth_resource.As(&depth_texture))) {
          BOOST_LOG(error) << "sbs-bench: warp prefilter input is not a texture";
          return 6;
        }
        D3D11_TEXTURE2D_DESC depth_desc {};
        depth_texture->GetDesc(&depth_desc);
        bool recreate_warp_depth = !warp_depth_tex || !warp_depth_uav || !warp_depth_srv;
        if (!recreate_warp_depth) {
          D3D11_TEXTURE2D_DESC current_desc {};
          warp_depth_tex->GetDesc(&current_desc);
          recreate_warp_depth = current_desc.Width != depth_desc.Width ||
                                current_desc.Height != depth_desc.Height ||
                                current_desc.Format != depth_desc.Format;
        }
        if (recreate_warp_depth) {
          depth_desc.Usage = D3D11_USAGE_DEFAULT;
          depth_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
          depth_desc.CPUAccessFlags = 0;
          depth_desc.MiscFlags = 0;
          warp_depth_tex.Reset();
          warp_depth_uav.Reset();
          warp_depth_srv.Reset();
          if (FAILED(dev->CreateTexture2D(&depth_desc, nullptr, &warp_depth_tex)) || FAILED(dev->CreateUnorderedAccessView(warp_depth_tex.Get(), nullptr, &warp_depth_uav)) || FAILED(dev->CreateShaderResourceView(warp_depth_tex.Get(), nullptr, &warp_depth_srv))) {
            BOOST_LOG(error) << "sbs-bench: warp prefilter resource creation failed";
            return 6;
          }
        }
        ctx->CSSetShader(warp_prefilter_cs.Get(), nullptr, 0);
        ctx->CSSetShaderResources(0, 1, est.depth.GetAddressOf());
        ctx->CSSetUnorderedAccessViews(0, 1, warp_depth_uav.GetAddressOf(), nullptr);
        ctx->Dispatch((depth_desc.Width + 15u) / 16u, (depth_desc.Height + 15u) / 16u, 1u);
        ID3D11UnorderedAccessView *null_prefilter_uav = nullptr;
        ID3D11ShaderResourceView *null_prefilter_srv = nullptr;
        ctx->CSSetUnorderedAccessViews(0, 1, &null_prefilter_uav, nullptr);
        ctx->CSSetShaderResources(0, 1, &null_prefilter_srv);
        warp_depth = warp_depth_srv.Get();
      }
      auto dispatch_coverage = [&](ID3D11ComputeShader *shader,
                                   ID3D11UnorderedAccessView *coverage_view) {
        const UINT clear_winner[4] = {0, 0, 0, 0};
        ctx->ClearUnorderedAccessViewUint(coverage_view, clear_winner);
        ctx->CSSetShader(shader, nullptr, 0);
        ctx->CSSetSamplers(0, 1, sampler.GetAddressOf());
        ID3D11ShaderResourceView *cs_srvs[] = {in_srv.Get(), warp_depth, est.subject.Get()};
        ctx->CSSetShaderResources(0, 3, cs_srvs);
        ctx->CSSetUnorderedAccessViews(0, 1, &coverage_view, nullptr);
        ctx->CSSetConstantBuffers(2, 1, repro_cb.GetAddressOf());
        ctx->Dispatch(((sbs_w / 2u) + 15u) / 16u, (sbs_h + 15u) / 16u, 1u);
        ID3D11UnorderedAccessView *null_uav[] = {nullptr};
        ID3D11ShaderResourceView *null_cs_srvs[] = {nullptr, nullptr, nullptr};
        ctx->CSSetUnorderedAccessViews(0, 1, null_uav, nullptr);
        ctx->CSSetShaderResources(0, 3, null_cs_srvs);
      };
      ctx->OMSetRenderTargets(1, sbs_rtv.GetAddressOf(), nullptr);
      ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      ctx->VSSetShader(vs.Get(), nullptr, 0);
      ctx->PSSetShader(ps.Get(), nullptr, 0);
      ctx->RSSetViewports(1, &vp);
      ctx->PSSetSamplers(0, 1, sampler.GetAddressOf());

      ID3D11ShaderResourceView *srvs[] = {in_srv.Get(), warp_depth, est.subject.Get()};
      ctx->PSSetShaderResources(0, 3, srvs);
      ID3D11Buffer *cb = repro_cb.Get();
      ctx->PSSetConstantBuffers(2, 1, &cb);
      ctx->Draw(3, 0);

      ID3D11RenderTargetView *null_rtv[] = {nullptr};
      ctx->OMSetRenderTargets(1, null_rtv, nullptr);
      ID3D11ShaderResourceView *null_srv[] = {nullptr, nullptr, nullptr, nullptr};
      ctx->PSSetShaderResources(0, 3, null_srv);
      ID3D11Texture2D *final_sbs_tex = sbs_tex.Get();
      if (time_warp) {
        ctx->End(warp_end.Get());
        ctx->End(warp_disjoint.Get());
      }

      // Real composite-submission CPU cost. GPU warp time is captured separately below with D3D
      // timestamp queries; tick() advances the perf window.
      sbs_perf::add_sample_ms("sbs_composite_cpu", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - comp_t0).count());
      if (time_warp) {
        D3D11_QUERY_DATA_TIMESTAMP_DISJOINT timing = {};
        UINT64 start_tick = 0, end_tick = 0;
        ctx->Flush();
        while (ctx->GetData(warp_disjoint.Get(), &timing, sizeof(timing), 0) == S_FALSE) {
          std::this_thread::yield();
        }
        HRESULT hs = ctx->GetData(warp_start.Get(), &start_tick, sizeof(start_tick), 0);
        HRESULT he = ctx->GetData(warp_end.Get(), &end_tick, sizeof(end_tick), 0);
        if (SUCCEEDED(hs) && SUCCEEDED(he) && !timing.Disjoint && timing.Frequency > 0 && end_tick >= start_tick) {
          sbs_perf::add_sample_ms("warp_infer", (double) (end_tick - start_tick) * 1000.0 / (double) timing.Frequency);
        }
      }
      sbs_perf::tick();

      // Offline-only mask pass, deliberately outside the production warp timestamp/CPU sample.
      // It exports R=pre-fill disocclusion. This evidence must not perturb perf conclusions.
      if (est.depth) {
        dispatch_coverage(coverage_cs.Get(), coverage_uav.Get());
      }
      ctx->OMSetRenderTargets(1, warp_mask_rtv.GetAddressOf(), nullptr);
      ctx->VSSetShader(vs.Get(), nullptr, 0);
      ctx->PSSetShader(mask_ps.Get(), nullptr, 0);
      ctx->RSSetViewports(1, &vp);
      ctx->PSSetSamplers(0, 1, sampler.GetAddressOf());
      ID3D11ShaderResourceView *mask_srvs[] = {
        in_srv.Get(),
        warp_depth,
        est.subject.Get(),
        est.depth ? coverage_srv.Get() : nullptr
      };
      ctx->PSSetShaderResources(0, 4, mask_srvs);
      ctx->PSSetConstantBuffers(2, 1, &cb);
      ctx->Draw(3, 0);
      ctx->OMSetRenderTargets(1, null_rtv, nullptr);
      ctx->PSSetShaderResources(0, 4, null_srv);

      // Offline-only exact inverse-warp mapping. This deliberately repeats the production
      // Reproject path after timing has ended: metric labels receive the shader's actual sampled
      // source coordinate instead of estimating correspondence again from the rendered colors.
      ctx->OMSetRenderTargets(1, warp_mapping_rtv.GetAddressOf(), nullptr);
      ctx->VSSetShader(vs.Get(), nullptr, 0);
      ctx->PSSetShader(mapping_ps.Get(), nullptr, 0);
      ctx->RSSetViewports(1, &vp);
      ctx->PSSetSamplers(0, 1, sampler.GetAddressOf());
      ID3D11ShaderResourceView *mapping_srvs[] = {
        in_srv.Get(),
        warp_depth,
        est.subject.Get()
      };
      ctx->PSSetShaderResources(0, 3, mapping_srvs);
      ctx->PSSetConstantBuffers(2, 1, &cb);
      ctx->Draw(3, 0);
      ctx->OMSetRenderTargets(1, null_rtv, nullptr);
      ctx->PSSetShaderResources(0, 3, null_srv);

      char mapping_name[64];
      snprintf(mapping_name, sizeof(mapping_name), "warp_map_%s.f32", output_id.c_str());
      if (!dump_float_texture(dev.Get(), ctx.Get(), warp_mapping_tex.Get(),
                              fs::path(o.out) / mapping_name, warp_mapping_stage)) {
        BOOST_LOG(error) << "sbs-bench: failed writing " << mapping_name;
        return 6;
      }

      // Readback -> PNG.
      char mask_name[64];
      snprintf(mask_name, sizeof(mask_name), "warp_mask_%s.png", output_id.c_str());
      dump_bgra8_texture(dev.Get(), ctx.Get(), warp_mask_tex.Get(),
                         fs::path(o.out) / mask_name, warp_mask_stage);
      ctx->CopyResource(sbs_stage.Get(), final_sbs_tex);
      D3D11_MAPPED_SUBRESOURCE m = {};
      if (SUCCEEDED(ctx->Map(sbs_stage.Get(), 0, D3D11_MAP_READ, 0, &m))) {
        std::vector<uint8_t> buf((size_t) sbs_w * sbs_h * 4);
        if (o.simulate_hdr) {
          for (UINT y = 0; y < sbs_h; ++y) {
            const uint16_t *row = (const uint16_t *) ((const uint8_t *) m.pData + (size_t) y * m.RowPitch);
            for (UINT x = 0; x < sbs_w; ++x) {
              const float r = half_to_float(row[x * 4 + 0]);
              const float g = half_to_float(row[x * 4 + 1]);
              const float b = half_to_float(row[x * 4 + 2]);
              for (float value : {r, g, b}) {
                if (std::isfinite(value)) {
                  hdr_output_min = std::min(hdr_output_min, value);
                  hdr_output_max = std::max(hdr_output_max, value);
                } else {
                  ++hdr_nonfinite;
                }
              }
              hdr_preview_bgra(r, g, b, &buf[((size_t) y * sbs_w + x) * 4]);
            }
          }
        } else {
          for (UINT y = 0; y < sbs_h; y++) {
            memcpy(&buf[(size_t) y * sbs_w * 4], (uint8_t *) m.pData + (size_t) y * m.RowPitch, sbs_w * 4);
          }
        }
        ctx->Unmap(sbs_stage.Get(), 0);
        char name[64];
        snprintf(name, sizeof(name), "sbs_%s.png", output_id.c_str());
        if (save_png(fs::path(o.out) / name, sbs_w, sbs_h, buf)) {
          written++;
        }
        char dname[64];
        snprintf(dname, sizeof(dname), "depth_%s.png", output_id.c_str());
        dump_depth(dev.Get(), ctx.Get(), est.depth.Get(), fs::path(o.out) / dname, depth_stage);
        if (sbs_cfg.ema_edge_change > 0.0) {
          char mname[64];
          snprintf(mname, sizeof(mname), "ema_mask_%s.png", output_id.c_str());
          dump_uint_mask(dev.Get(), ctx.Get(), est.ema_motion_mask.Get(),
                         fs::path(o.out) / mname, ema_mask_stage);
        }
        char rname[64];
        snprintf(rname, sizeof(rname), "raw_%s.f32", output_id.c_str());
        dump_raw_model_depth(dev.Get(), ctx.Get(), est.raw_model_depth.Get(), est.raw_width, est.raw_height, fs::path(o.out) / rname, raw_depth_stage);
        if (!raw_shape_written && est.raw_width > 0 && est.raw_height > 0) {
          std::ofstream shape(fs::path(o.out) / "raw_shape.json");
          if (shape) {
            shape << "{\n  \"width\": " << est.raw_width << ",\n  \"height\": "
                  << est.raw_height << ",\n  \"dtype\": \"float32-le\",\n"
                                       "  \"stage\": \"raw model output before transform/normalization/EMA/curvature\"\n}\n";
            raw_shape_written = true;
          }
        }
      }
      if (((fi + 1) % 20) == 0) {
        BOOST_LOG(info) << "sbs-bench: " << (fi + 1) << "/" << frames.size();
      }
    }

    if (applied_depth_override_frames != expected_depth_override_frames) {
      BOOST_LOG(error) << "sbs-bench: applied " << applied_depth_override_frames
                       << " of " << expected_depth_override_frames << " expected depth overrides";
      return 7;
    }
    if (!warp_mapping_shape_written) {
      BOOST_LOG(error) << "sbs-bench: no warp mapping shape contract was written";
      return 8;
    }

    sbs_perf::dump_json((fs::path(o.out) / "sbs_perf.json").string());
    {
      // Machine-readable execution contract. Evaluation must not scrape human log prose: custom
      // profile names are case-sensitive and fidelity runs must prove literal Bestv2 was active.
      std::ofstream contract(fs::path(o.out) / "contract.json");
      if (contract) {
        contract << "{\n"
                 << "  \"schema\": 16,\n"
                 << "  \"model\": " << json_string(model.name) << ",\n"
                 << "  \"profile\": " << json_string(sbs_cfg.profile) << ",\n"
                 << "  \"depth_step\": "
                 << json_string(o.depth_every == 1 ? std::string("current-once") : "reuse-" + std::to_string(o.depth_every))
                 << ",\n"
                 << "  \"depth_reuse_interval\": " << o.depth_every << ",\n"
                 << "  \"depth_compensation\": "
                 << json_string(o.depth_override_root.empty() ? "none" :
                                (o.depth_override_all ? "external-treatment" :
                                                        "external-reference"))
                 << ",\n"
                 << "  \"depth_override_frames\": " << applied_depth_override_frames << ",\n"
                 << "  \"ema\": " << sbs_cfg.ema << ",\n"
                 << "  \"ema_edge_change\": " << sbs_cfg.ema_edge_change << ",\n"
                 << "  \"ema_edge_gradient\": " << sbs_cfg.ema_edge_gradient << ",\n"
                 << "  \"ema_edge_strength\": " << sbs_cfg.ema_edge_strength << ",\n"
                 << "  \"adaptive_pop\": " << (sbs_cfg.adaptive_pop ? "true" : "false") << ",\n"
                 << "  \"adaptive_pop_max\": " << sbs_cfg.adaptive_pop_max << ",\n"
                 << "  \"zero_plane\": " << json_string(sbs_cfg.zero_plane) << ",\n"
                 << "  \"literal_bestv2\": " << (o.literal_bestv2 ? "true" : "false") << ",\n"
                 << "  \"cuda_graph\": " << (sbs_cfg.cuda_graph ? "true" : "false") << ",\n"
                 << "  \"cuda_graph_captured\": " << (cuda_graph_captured ? "true" : "false") << ",\n"
                 << "  \"warp_mask\": {\"red\": \"forward_disocclusion_before_fill\"},\n"
                 << "  \"warp_mapping\": {\n"
                 << "    \"file_pattern\": \"warp_map_<frame-id>.f32\",\n"
                 << "    \"shape_contract\": \"warp_map_shape.json\",\n"
                 << "    \"dtype\": \"float32-le\",\n"
                 << "    \"layout\": \"row-major\",\n"
                 << "    \"channels\": [\"raw_reproject_source_u_normalized\"],\n"
                 << "    \"live_sample_transform\": \"clamp(raw_reproject_source_u_normalized, 0, 1)\",\n"
                 << "    \"validity_companion\": \"warp_mask_<frame-id>.png:red=forward_disocclusion_before_fill; content validity derives from warp_map_shape.json\"\n"
                 << "  }\n"
                 << "}\n";
      }
    }
    if (o.simulate_hdr) {
      std::ofstream stats(fs::path(o.out) / "hdr_output_stats.json");
      if (stats) {
        stats << "{\n  \"format\": \"linear-scRGB-fp16\",\n"
              << "  \"input_scale\": " << o.hdr_scale << ",\n"
              << "  \"output_min\": " << hdr_output_min << ",\n"
              << "  \"output_max\": " << hdr_output_max << ",\n"
              << "  \"nonfinite_components\": " << hdr_nonfinite << "\n}\n";
      }
    }
    BOOST_LOG(info) << "sbs-bench: wrote " << written << " SBS frames + sbs_perf.json to " << o.out;
    return written > 0 ? 0 : 8;
  }

}  // namespace sbs_bench

#else  // !_WIN32
namespace sbs_bench {
  int run(int, char **) {
    return 1;
  }
}  // namespace sbs_bench
#endif
