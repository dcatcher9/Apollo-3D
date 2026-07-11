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
#include <string>
#include <thread>
#include <vector>
#include <limits>

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

    uint16_t float_to_half(float value) {
      uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      const uint32_t sign = (bits >> 16) & 0x8000u;
      int exp = (int)((bits >> 23) & 0xffu) - 127 + 15;
      uint32_t mant = bits & 0x7fffffu;
      if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x800000u) >> (1 - exp);
        return (uint16_t)(sign | ((mant + 0x1000u) >> 13));
      }
      if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
      mant += 0x1000u;
      if (mant & 0x800000u) {
        mant = 0;
        if (++exp >= 31) return (uint16_t)(sign | 0x7c00u);
      }
      return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    }

    float half_to_float(uint16_t h) {
      const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
      int exp = (h >> 10) & 0x1f;
      uint32_t mant = h & 0x3ffu;
      uint32_t bits;
      if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
          exp = 127 - 15 + 1;
          while (!(mant & 0x400u)) { mant <<= 1; --exp; }
          bits = sign | ((uint32_t)exp << 23) | ((mant & 0x3ffu) << 13);
        }
      } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
      } else {
        bits = sign | ((uint32_t)(exp - 15 + 127) << 23) | (mant << 13);
      }
      float value;
      std::memcpy(&value, &bits, sizeof(value));
      return value;
    }

    float srgb_to_linear(float value) {
      return value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
    }

    void hdr_preview_bgra(float r, float g, float b, uint8_t *out) {
      r = std::max(r, 0.0f); g = std::max(g, 0.0f); b = std::max(b, 0.0f);
      const float y = std::max(0.2126f * r + 0.7152f * g + 0.0722f * b, 0.0f);
      r /= 1.0f + y; g /= 1.0f + y; b /= 1.0f + y;
      const float peak = std::max(1.0f, std::max(r, std::max(g, b)));
      r /= peak; g /= peak; b /= peak;
      auto encode = [](float c) {
        c = std::clamp(c, 0.0f, 1.0f);
        c = c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
        return (uint8_t)std::lround(std::clamp(c, 0.0f, 1.0f) * 255.0f);
      };
      out[0] = encode(b); out[1] = encode(g); out[2] = encode(r); out[3] = 255;
    }

    // ---- WIC PNG load/save (32bpp BGRA, matching the SDR B8G8R8A8_UNORM pipeline) ----

    ComPtr<IWICImagingFactory> g_wic;

    bool wic_init() {
      if (g_wic) return true;
      if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) {
        // Already initialized on this thread with another mode is fine.
      }
      return SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&g_wic)));
    }

    bool load_png(const fs::path &path, rgba_image &out) {
      ComPtr<IWICBitmapDecoder> dec;
      if (FAILED(g_wic->CreateDecoderFromFilename(path.wstring().c_str(), nullptr, GENERIC_READ,
            WICDecodeMetadataCacheOnDemand, &dec)))
        return false;
      ComPtr<IWICBitmapFrameDecode> frame;
      if (FAILED(dec->GetFrame(0, &frame))) return false;
      ComPtr<IWICFormatConverter> conv;
      if (FAILED(g_wic->CreateFormatConverter(&conv))) return false;
      if (FAILED(conv->Initialize(frame.Get(), GUID_WICPixelFormat32bppBGRA,
            WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom)))
        return false;
      if (FAILED(conv->GetSize(&out.w, &out.h))) return false;
      out.bgra.resize((size_t) out.w * out.h * 4);
      return SUCCEEDED(conv->CopyPixels(nullptr, out.w * 4, (UINT) out.bgra.size(), out.bgra.data()));
    }

    bool save_png(const fs::path &path, UINT w, UINT h, const std::vector<uint8_t> &bgra) {
      ComPtr<IWICStream> stream;
      if (FAILED(g_wic->CreateStream(&stream))) return false;
      if (FAILED(stream->InitializeFromFilename(path.wstring().c_str(), GENERIC_WRITE))) return false;
      ComPtr<IWICBitmapEncoder> enc;
      if (FAILED(g_wic->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc))) return false;
      if (FAILED(enc->Initialize(stream.Get(), WICBitmapEncoderNoCache))) return false;
      ComPtr<IWICBitmapFrameEncode> fe;
      ComPtr<IPropertyBag2> props;
      if (FAILED(enc->CreateNewFrame(&fe, &props))) return false;
      if (FAILED(fe->Initialize(props.Get()))) return false;
      fe->SetSize(w, h);
      WICPixelFormatGUID fmt = GUID_WICPixelFormat32bppBGRA;
      fe->SetPixelFormat(&fmt);
      if (FAILED(fe->WritePixels(h, w * 4, (UINT) bgra.size(), const_cast<uint8_t *>(bgra.data()))))
        return false;
      return SUCCEEDED(fe->Commit()) && SUCCEEDED(enc->Commit());
    }

    bool save_gray16_png(const fs::path &path, UINT w, UINT h, const std::vector<uint16_t> &gray) {
      ComPtr<IWICStream> stream;
      if (FAILED(g_wic->CreateStream(&stream))) return false;
      if (FAILED(stream->InitializeFromFilename(path.wstring().c_str(), GENERIC_WRITE))) return false;
      ComPtr<IWICBitmapEncoder> enc;
      if (FAILED(g_wic->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc))) return false;
      if (FAILED(enc->Initialize(stream.Get(), WICBitmapEncoderNoCache))) return false;
      ComPtr<IWICBitmapFrameEncode> fe;
      ComPtr<IPropertyBag2> props;
      if (FAILED(enc->CreateNewFrame(&fe, &props))) return false;
      if (FAILED(fe->Initialize(props.Get()))) return false;
      fe->SetSize(w, h);
      WICPixelFormatGUID fmt = GUID_WICPixelFormat16bppGray;
      fe->SetPixelFormat(&fmt);
      if (FAILED(fe->WritePixels(h, w * 2, (UINT) (gray.size() * 2), (BYTE *) const_cast<uint16_t *>(gray.data()))))
        return false;
      return SUCCEEDED(fe->Commit()) && SUCCEEDED(enc->Commit());
    }

    // Read back an R32_FLOAT depth SRV and save it as a 16-bit grayscale PNG (values clamped to
    // [0,1] scaled to 0-65535). 16-bit matters: the swim metric measures frame-to-frame depth
    // deltas that sit below 1/255. The staging texture is cached across frames (constant size).
    void dump_depth(ID3D11Device *dev, ID3D11DeviceContext *ctx, ID3D11ShaderResourceView *srv,
      const fs::path &path, ComPtr<ID3D11Texture2D> &stage_cache) {
      if (!srv) return;
      ComPtr<ID3D11Resource> res;
      srv->GetResource(&res);
      ComPtr<ID3D11Texture2D> tex;
      if (FAILED(res.As(&tex))) return;
      D3D11_TEXTURE2D_DESC d = {};
      tex->GetDesc(&d);
      if (!stage_cache) {
        D3D11_TEXTURE2D_DESC sd = d;
        sd.Usage = D3D11_USAGE_STAGING;
        sd.BindFlags = 0;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        sd.MiscFlags = 0;
        if (FAILED(dev->CreateTexture2D(&sd, nullptr, &stage_cache))) return;
      }
      ctx->CopyResource(stage_cache.Get(), tex.Get());
      D3D11_MAPPED_SUBRESOURCE m = {};
      if (FAILED(ctx->Map(stage_cache.Get(), 0, D3D11_MAP_READ, 0, &m))) return;
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

    // Preserve the exact raw model output for stage-by-stage parity checks. Unlike the display
    // PNG, this is not clamped or normalized: it is row-major float32, width*height values.
    void dump_raw_model_depth(ID3D11Device *dev, ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv, int width, int height, const fs::path &path,
      ComPtr<ID3D11Buffer> &stage_cache) {
      if (!srv || width <= 0 || height <= 0) return;
      ComPtr<ID3D11Resource> res;
      srv->GetResource(&res);
      ComPtr<ID3D11Buffer> buf;
      if (FAILED(res.As(&buf))) return;
      D3D11_BUFFER_DESC d = {};
      buf->GetDesc(&d);
      if (!stage_cache) {
        D3D11_BUFFER_DESC sd = d;
        sd.Usage = D3D11_USAGE_STAGING;
        sd.BindFlags = 0;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        sd.MiscFlags = 0;
        if (FAILED(dev->CreateBuffer(&sd, nullptr, &stage_cache))) return;
      }
      ctx->CopyResource(stage_cache.Get(), buf.Get());
      D3D11_MAPPED_SUBRESOURCE m = {};
      if (FAILED(ctx->Map(stage_cache.Get(), 0, D3D11_MAP_READ, 0, &m))) return;
      std::ofstream out(path, std::ios::binary);
      if (out) {
        out.write((const char *) m.pData, (std::streamsize) width * height * sizeof(float));
      }
      ctx->Unmap(stage_cache.Get(), 0);
    }

    // Keep output identities tied to source identities. Positional renumbering made a dropped
    // source frame silently shift every depth/SBS/source comparison by one.
    std::string frame_id(const fs::path &path, size_t fallback) {
      std::string stem = path.stem().string();
      size_t split = stem.find_last_of('_');
      std::string id = split == std::string::npos ? "" : stem.substr(split + 1);
      if (!id.empty() && std::all_of(id.begin(), id.end(), [](unsigned char c) { return std::isdigit(c); })) {
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
      HRESULT hr = D3DCompileFromFile(wfile.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        entry, model, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &err);
      if (FAILED(hr)) {
        BOOST_LOG(error) << "sbs-bench: shader compile failed [" << file << "]: "
                         << (err ? (const char *) err->GetBufferPointer() : "?");
        return nullptr;
      }
      return blob;
    }

    template <int N>
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
      std::string frames, out, model, warp, shift_profile;
      int eye_w = 0;       // 0 -> derive from source aspect; set with eye_h to test letterboxing
      int eye_h = 0;       // 0 -> match/derive from the input frame
      double output_scale = 1.0;  // per-eye linear scale vs source; preserves source aspect
      bool simulate_hdr = false;  // decode sRGB frames into linear scRGB FP16 and use HDR paths
      double hdr_scale = 4.0;     // scRGB multiplier after sRGB EOTF (4.0 = 320-nit diffuse white)
      int max_width = 0;   // 0 -> use config max_encode_width
      int limit = 0;       // 0 -> all
      int output_every = 1;  // process every input for temporal state; dump only every Nth
      double divergence = -1.0;  // <0 -> use the conf's value; else override (parallax/disocclusion size)
      // VD3D-pipeline A/B levers; <0 / false -> use the conf's value.
      double pct_lo = -1.0;      // robust normalization low percentile (e.g. 1.0)
      double pct_hi = -1.0;      // robust normalization high percentile (e.g. 99.0)
      int subject_track = -1;      // -1 = conf, 0 = off, 1 = on
      double subject_lock = -1.0;  // subject anchor strength override (e.g. 0.95)
      double subject_recenter = -1.0;  // global subject recenter override
      int depth_short_side = 0;    // depth inference short-side override (0 = conf; VD3D uses 432)
      double ema = -1.0;         // per-pixel depth EMA override (1.0 = off)
      int subject_stretch = -1;     // -1 = conf, 0 = off, 1 = on
      double subject_plane_lock = -1.0;  // local subject-band flatten (e.g. 0.28); <0 = conf
      double curvature = -1.0;     // foreground curvature strength (e.g. 0.07); <0 = conf
      double vd3d_forward_blend = -1.0;  // VD3D backward/forward blend override; <0 = conf
      double minmax_ema = -1.0;    // range-bounds EMA new-weight (VD3D DepthPercentileEMA a=0.82 -> 0.18); <0 = conf
      int guided = -1;             // guided upsample: -1 = conf, 0 = off, 1 = on
      int pixel_ema_first = -1;     // -1 = conf, 0 = range->pixel, 1 = pixel->range
      double minmax_snap = -1.0;    // scene-cut normalization snap ratio; 0 = off
      double range_floor = -1.0;    // flat-scene range-floor fraction; 0 = off
      double depth_floor = -1.0;    // warp depth floor; 0 = off
      double border_fade = -1.0;    // warp border fade; 0 = off
      int bestv2_sharpen = -1;      // -1 = conf, 0 = off, 1 = on
    };

    bool parse_opts(int argc, char **argv, opts &o) {
      for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char *n) -> std::string {
          if (i + 1 >= argc) { BOOST_LOG(error) << "sbs-bench: " << n << " needs a value"; return ""; }
          return argv[++i];
        };
        if (a == "--frames") o.frames = next("--frames");
        else if (a == "--out") o.out = next("--out");
        else if (a == "--model") o.model = next("--model");
        else if (a == "--warp") o.warp = next("--warp");
        else if (a == "--shift-profile") o.shift_profile = next("--shift-profile");
        else if (a == "--eye-w") o.eye_w = std::stoi(next("--eye-w"));
        else if (a == "--eye-h") o.eye_h = std::stoi(next("--eye-h"));
        else if (a == "--output-scale") o.output_scale = std::stod(next("--output-scale"));
        else if (a == "--simulate-hdr") o.simulate_hdr = true;
        else if (a == "--hdr-scale") o.hdr_scale = std::stod(next("--hdr-scale"));
        else if (a == "--max-width") o.max_width = std::stoi(next("--max-width"));
        else if (a == "--limit") o.limit = std::stoi(next("--limit"));
        else if (a == "--output-every") o.output_every = std::max(1, std::stoi(next("--output-every")));
        else if (a == "--divergence") o.divergence = std::stod(next("--divergence"));
        else if (a == "--pct-lo") o.pct_lo = std::stod(next("--pct-lo"));
        else if (a == "--pct-hi") o.pct_hi = std::stod(next("--pct-hi"));
        else if (a == "--subject-track") o.subject_track = 1;
        else if (a == "--no-subject-track") o.subject_track = 0;
        else if (a == "--subject-lock") o.subject_lock = std::stod(next("--subject-lock"));
        else if (a == "--subject-recenter") o.subject_recenter = std::stod(next("--subject-recenter"));
        else if (a == "--depth-short-side") o.depth_short_side = std::stoi(next("--depth-short-side"));
        else if (a == "--subject-stretch") o.subject_stretch = 1;
        else if (a == "--no-subject-stretch") o.subject_stretch = 0;
        else if (a == "--subject-plane-lock") o.subject_plane_lock = std::stod(next("--subject-plane-lock"));
        else if (a == "--curvature") o.curvature = std::stod(next("--curvature"));
        else if (a == "--vd3d-forward-blend") o.vd3d_forward_blend = std::stod(next("--vd3d-forward-blend"));
        else if (a == "--ema") o.ema = std::stod(next("--ema"));
        else if (a == "--minmax-ema") o.minmax_ema = std::stod(next("--minmax-ema"));
        else if (a == "--guided-upsample") {
          std::string v = next("--guided-upsample");
          o.guided = (v == "off" || v == "0" || v == "false") ? 0 : 1;
        }
        else if (a == "--pixel-ema-first") o.pixel_ema_first = 1;
        else if (a == "--no-pixel-ema-first") o.pixel_ema_first = 0;
        else if (a == "--minmax-snap") o.minmax_snap = std::stod(next("--minmax-snap"));
        else if (a == "--range-floor") o.range_floor = std::stod(next("--range-floor"));
        else if (a == "--depth-floor") o.depth_floor = std::stod(next("--depth-floor"));
        else if (a == "--border-fade") o.border_fade = std::stod(next("--border-fade"));
        else if (a == "--bestv2-sharpen") {
          std::string v = next("--bestv2-sharpen");
          o.bestv2_sharpen = (v == "off" || v == "0" || v == "false") ? 0 : 1;
        }
        else { BOOST_LOG(error) << "sbs-bench: unknown arg '" << a << "'"; return false; }
      }
      if (o.frames.empty() || o.out.empty()) {
        BOOST_LOG(error) << "sbs-bench: --frames DIR and --out DIR are required";
        return false;
      }
      if (!(o.output_scale > 0.0 && o.output_scale <= 4.0)) {
        BOOST_LOG(error) << "sbs-bench: --output-scale must be greater than 0 and at most 4";
        return false;
      }
      if (!(o.hdr_scale > 0.0 && o.hdr_scale <= 64.0)) {
        BOOST_LOG(error) << "sbs-bench: --hdr-scale must be greater than 0 and at most 64";
        return false;
      }
      return true;
    }

    config::depth_model_info pick_model(const opts &o) {
      const auto &reg = config::depth_model_registry();
      std::string want = o.model;
      if (!want.empty()) {
        for (const auto &m : reg)
          if (m.name == want) return m;
        BOOST_LOG(warning) << "sbs-bench: model '" << want << "' not in registry; using active model";
      }
      return video::active_depth_model();
    }

  }  // namespace

  int run(int argc, char **argv) {
    opts o;
    if (!parse_opts(argc, argv, o)) return 2;
    if (!wic_init()) { BOOST_LOG(error) << "sbs-bench: WIC init failed"; return 3; }

    // Collect + sort input frames (png/jpg; WIC decodes both). Small pre-resized JPEG clips keep
    // the repo light; the harness never resizes them -- the SBS output tracks the input size.
    std::vector<fs::path> frames;
    std::error_code ec;
    for (auto &e : fs::directory_iterator(o.frames, ec)) {
      if (!e.is_regular_file()) continue;
      auto ext = e.path().extension().string();
      for (auto &ch : ext) ch = (char) tolower((unsigned char) ch);
      if (ext == ".png" || ext == ".jpg" || ext == ".jpeg")
        frames.push_back(e.path());
    }
    std::sort(frames.begin(), frames.end());
    if (o.limit > 0 && (int) frames.size() > o.limit) frames.resize(o.limit);
    if (frames.empty()) { BOOST_LOG(error) << "sbs-bench: no png/jpg frames in " << o.frames; return 4; }
    fs::create_directories(o.out, ec);

    // Inherit the loaded config, then pin one depth update per source frame. The benchmark is
    // frame-driven rather than wall-clock-driven, so cadence throttling would make the result
    // depend on machine speed.
    auto sbs_cfg = config::video.sbs;
    sbs_cfg.depth_fps = 0.0;
    if (!o.warp.empty()) {
      if (o.warp != "apollo" && o.warp != "vd3d") {
        BOOST_LOG(error) << "sbs-bench: --warp must be 'apollo' or 'vd3d'";
        return 2;
      }
      sbs_cfg.warp = o.warp;
    }
    if (!o.shift_profile.empty()) {
      if (o.shift_profile != "apollo" && o.shift_profile != "bestv2") {
        BOOST_LOG(error) << "sbs-bench: --shift-profile must be 'apollo' or 'bestv2'";
        return 2;
      }
      sbs_cfg.shift_profile = o.shift_profile;
    }
    if (o.divergence >= 0.0) sbs_cfg.divergence = o.divergence;  // A/B lever: parallax/disocclusion size
    if (o.pct_lo >= 0.0) sbs_cfg.norm_pct_lo = o.pct_lo;         // A/B lever: robust normalization
    if (o.pct_hi >= 0.0) sbs_cfg.norm_pct_hi = o.pct_hi;
    if (o.subject_track >= 0) sbs_cfg.subject_track = (o.subject_track != 0);
    if (o.subject_lock >= 0.0) sbs_cfg.subject_lock = o.subject_lock;
    if (o.subject_recenter >= 0.0) sbs_cfg.subject_recenter = o.subject_recenter;
    if (o.depth_short_side > 0) sbs_cfg.depth_short_side = o.depth_short_side;  // VD3D uses 432
    if (o.subject_stretch >= 0) sbs_cfg.subject_stretch = (o.subject_stretch != 0);
    if (o.subject_plane_lock >= 0.0) sbs_cfg.subject_plane_lock = o.subject_plane_lock;
    if (o.curvature >= 0.0) sbs_cfg.foreground_curvature = o.curvature;
    if (o.vd3d_forward_blend >= 0.0) sbs_cfg.vd3d_forward_blend = o.vd3d_forward_blend;
    if (o.ema > 0.0) sbs_cfg.ema = o.ema;                        // A/B lever: depth EMA (1.0 = off)
    if (o.minmax_ema >= 0.0) sbs_cfg.minmax_ema = o.minmax_ema;  // A/B lever: range-bounds EMA (VD3D 0.18)
    if (o.guided >= 0) sbs_cfg.guided_upsample = (o.guided != 0);  // A/B lever: guided upsample on/off
    if (o.pixel_ema_first >= 0) sbs_cfg.ema_pixel_first = (o.pixel_ema_first != 0);
    if (o.minmax_snap >= 0.0) sbs_cfg.minmax_snap = o.minmax_snap;
    if (o.range_floor >= 0.0) sbs_cfg.range_floor = o.range_floor;
    if (o.depth_floor >= 0.0) sbs_cfg.depth_floor = o.depth_floor;
    if (o.border_fade >= 0.0) sbs_cfg.border_fade = o.border_fade;
    if (o.bestv2_sharpen >= 0) sbs_cfg.bestv2_sharpen = (o.bestv2_sharpen != 0);
    sbs_cfg.perf_stats = true;  // the harness always measures
    sbs_perf::set_enabled(true);
    sbs_perf::reset();
    auto model = pick_model(o);
    const int max_width = o.max_width > 0 ? o.max_width : config::video.sbs.max_encode_width;

    BOOST_LOG(info) << "sbs-bench: " << frames.size() << " frames, model '" << model.name
                    << "', eye " << (o.eye_w > 0 ? std::to_string(o.eye_w) : "auto") << 'x'
                    << (o.eye_h > 0 ? std::to_string(o.eye_h) : "auto")
                    << ", depth_step current-once, warp " << sbs_cfg.warp
                    << ", shift_profile " << sbs_cfg.shift_profile << " -> " << o.out;

    // ---- D3D device + shaders ----
    ComPtr<ID3D11Device> dev;
    ComPtr<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL fl;
    const D3D_FEATURE_LEVEL want_fl[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
          D3D11_CREATE_DEVICE_BGRA_SUPPORT, want_fl, 2, D3D11_SDK_VERSION, &dev, &fl, &ctx))) {
      BOOST_LOG(error) << "sbs-bench: D3D11CreateDevice failed";
      return 5;
    }

    // Harness-only GPU timestamps. CPU submission time is not a useful comparison between the
    // probe draw and VD3D's compute+splat+resolve sequence.
    ComPtr<ID3D11Query> warp_disjoint, warp_start, warp_end;
    D3D11_QUERY_DESC qd = {D3D11_QUERY_TIMESTAMP_DISJOINT, 0};
    dev->CreateQuery(&qd, &warp_disjoint);
    qd.Query = D3D11_QUERY_TIMESTAMP;
    dev->CreateQuery(&qd, &warp_start);
    dev->CreateQuery(&qd, &warp_end);

    auto vs_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_reprojection_vs.hlsl", "main_vs", "vs_5_0");
    auto ps_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_reprojection_ps.hlsl", "main_ps", "ps_5_0");
    auto sharpen_ps_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_sharpen_ps.hlsl", "main_ps", "ps_5_0");
    auto vd3d_cs_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_vd3d_forward_cs.hlsl", "main", "cs_5_0");
    auto vd3d_ps_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_vd3d_reprojection_ps.hlsl", "main_ps", "ps_5_0");
    if (!vs_blob || !ps_blob || !sharpen_ps_blob || !vd3d_cs_blob || !vd3d_ps_blob) return 6;
    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> ps, sharpen_ps, vd3d_ps;
    ComPtr<ID3D11ComputeShader> vd3d_cs;
    dev->CreateVertexShader(vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr, &vs);
    dev->CreatePixelShader(ps_blob->GetBufferPointer(), ps_blob->GetBufferSize(), nullptr, &ps);
    dev->CreatePixelShader(sharpen_ps_blob->GetBufferPointer(), sharpen_ps_blob->GetBufferSize(), nullptr, &sharpen_ps);
    dev->CreateComputeShader(vd3d_cs_blob->GetBufferPointer(), vd3d_cs_blob->GetBufferSize(), nullptr, &vd3d_cs);
    dev->CreatePixelShader(vd3d_ps_blob->GetBufferPointer(), vd3d_ps_blob->GetBufferSize(), nullptr, &vd3d_ps);

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
    ComPtr<ID3D11Buffer> repro_cb, pass_cb;

    // ---- estimator ----
    models::video_depth_estimator estimator(dev, ctx, fs::path(SUNSHINE_ASSETS_DIR), sbs_cfg, model);

    // Per-run state built lazily on the first frame (once we know the input size).
    ComPtr<ID3D11Texture2D> sbs_tex, sbs_stage;
    ComPtr<ID3D11RenderTargetView> sbs_rtv;
    ComPtr<ID3D11ShaderResourceView> sbs_srv;
    ComPtr<ID3D11Texture2D> sharpen_tex;
    ComPtr<ID3D11RenderTargetView> sharpen_rtv;
    ComPtr<ID3D11ShaderResourceView> sharpen_srv;
    ComPtr<ID3D11Texture2D> vd3d_winner_tex;
    ComPtr<ID3D11UnorderedAccessView> vd3d_winner_uav;
    ComPtr<ID3D11ShaderResourceView> vd3d_winner_srv;
    D3D11_VIEWPORT vp = {};
    UINT sbs_w = 0, sbs_h = 0;
    ComPtr<ID3D11Texture2D> depth_stage;  // dump_depth staging cache (depth size is constant)
    ComPtr<ID3D11Buffer> raw_depth_stage;
    bool raw_shape_written = false;
    float hdr_output_min = std::numeric_limits<float>::infinity();
    float hdr_output_max = -std::numeric_limits<float>::infinity();
    uint64_t hdr_nonfinite = 0;

    int written = 0;
    for (size_t fi = 0; fi < frames.size(); fi++) {
      rgba_image img;
      if (!load_png(frames[fi], img)) { BOOST_LOG(warning) << "sbs-bench: skip " << frames[fi]; continue; }

      // Input texture + SRV.
      D3D11_TEXTURE2D_DESC id = {};
      id.Width = img.w; id.Height = img.h; id.MipLevels = 1; id.ArraySize = 1;
      id.Format = o.simulate_hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_B8G8R8A8_UNORM;
      id.SampleDesc.Count = 1;
      id.Usage = D3D11_USAGE_IMMUTABLE; id.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      std::vector<uint16_t> hdr_rgba;
      const void *input_pixels = img.bgra.data();
      UINT input_pitch = img.w * 4;
      if (o.simulate_hdr) {
        hdr_rgba.resize((size_t)img.w * img.h * 4);
        for (size_t p = 0; p < (size_t)img.w * img.h; ++p) {
          const float b = srgb_to_linear(img.bgra[p * 4 + 0] / 255.0f) * (float)o.hdr_scale;
          const float g = srgb_to_linear(img.bgra[p * 4 + 1] / 255.0f) * (float)o.hdr_scale;
          const float r = srgb_to_linear(img.bgra[p * 4 + 2] / 255.0f) * (float)o.hdr_scale;
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
      if (FAILED(dev->CreateTexture2D(&id, &isd, &in_tex))) { BOOST_LOG(error) << "sbs-bench: input tex fail"; continue; }
      ComPtr<ID3D11ShaderResourceView> in_srv;
      dev->CreateShaderResourceView(in_tex.Get(), nullptr, &in_srv);

      // First frame: size the SBS target. Per eye = the input resolution by default (so the clip
      // size, not a fixed constant, drives eval cost); --eye-h pins a specific output height.
      // The width is still capped at max_encode_width like the live path.
      if (!sbs_tex) {
        int eh_target = o.eye_h > 0 ? o.eye_h :
          std::max(2, (int)std::lround((double)img.h * o.output_scale));
        float aspect = (float) img.w / (float) img.h;
        int eye_w = o.eye_w > 0 ? o.eye_w : (o.eye_h > 0
          ? std::max(1, (int)std::lround(eh_target * aspect))
          : std::max(1, (int)std::lround((double)img.w * o.output_scale)));
        int eye_h = eh_target;
        if (o.eye_w > 0 && o.eye_h <= 0) eye_h = std::max(1, (int)std::lround(eye_w / aspect));
        if (2 * eye_w > max_width) {
          const double scale = (double) max_width / (double) (2 * eye_w);
          eye_w = std::max(1, max_width / 2);
          eye_h = std::max(2, ((int) std::lround(eh_target * scale)) & ~1);
        }
        sbs_w = (UINT) (2 * eye_w); sbs_h = (UINT) eye_h;
        const float eye_aspect = (float)eye_w / (float)eye_h;
        const float content_scale_x = eye_aspect > aspect ? aspect / eye_aspect : 1.0f;
        const float content_scale_y = eye_aspect < aspect ? eye_aspect / aspect : 1.0f;
        const float source_to_output = (float)eye_w * content_scale_x / (float)img.w;
        float repro_params[16] = {(float) sbs_cfg.divergence, (float) sbs_cfg.focal_plane,
          (float) sbs_cfg.parallax_steps, (float) sbs_cfg.border_fade, (float) sbs_cfg.depth_floor,
          sbs_cfg.subject_track ? 1.0f : 0.0f, (float) sbs_cfg.subject_lock,
          sbs_cfg.subject_stretch ? 1.0f : 0.0f, (float) sbs_cfg.subject_plane_lock,
          (float) sbs_cfg.subject_plane_width, content_scale_x, content_scale_y,
          (float) sbs_cfg.vd3d_forward_blend, (float) sbs_cfg.vd3d_fill_radius,
          sbs_cfg.shift_profile == "bestv2" ? 1.0f : 0.0f, source_to_output};
        repro_cb = const_buffer(dev.Get(), repro_params);
        float pass_params[16] = {0, (float) sbs_cfg.focal_plane, (float) sbs_cfg.parallax_steps,
          (float) sbs_cfg.border_fade, (float) sbs_cfg.depth_floor, 0, 0, 0, 0, 0,
          content_scale_x, content_scale_y, (float) sbs_cfg.vd3d_forward_blend,
          (float) sbs_cfg.vd3d_fill_radius, 0, source_to_output};
        pass_cb = const_buffer(dev.Get(), pass_params);
        D3D11_TEXTURE2D_DESC td = {};
        td.Width = sbs_w; td.Height = sbs_h; td.MipLevels = 1; td.ArraySize = 1;
        td.Format = o.simulate_hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        dev->CreateTexture2D(&td, nullptr, &sbs_tex);
        dev->CreateRenderTargetView(sbs_tex.Get(), nullptr, &sbs_rtv);
        dev->CreateShaderResourceView(sbs_tex.Get(), nullptr, &sbs_srv);
        if (!o.simulate_hdr && sbs_cfg.shift_profile == "bestv2" && sbs_cfg.bestv2_sharpen) {
          if (FAILED(dev->CreateTexture2D(&td, nullptr, &sharpen_tex)) ||
              FAILED(dev->CreateRenderTargetView(sharpen_tex.Get(), nullptr, &sharpen_rtv)) ||
              FAILED(dev->CreateShaderResourceView(sharpen_tex.Get(), nullptr, &sharpen_srv))) {
            BOOST_LOG(error) << "sbs-bench: sharpen texture creation failed";
            return 6;
          }
        }
        if (sbs_cfg.warp == "vd3d") {
          D3D11_TEXTURE2D_DESC wd = td;
          wd.Format = DXGI_FORMAT_R32_UINT;
          wd.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
          if (FAILED(dev->CreateTexture2D(&wd, nullptr, &vd3d_winner_tex)) ||
              FAILED(dev->CreateUnorderedAccessView(vd3d_winner_tex.Get(), nullptr, &vd3d_winner_uav)) ||
              FAILED(dev->CreateShaderResourceView(vd3d_winner_tex.Get(), nullptr, &vd3d_winner_srv))) {
            BOOST_LOG(error) << "sbs-bench: VD3D winner texture creation failed";
            return 6;
          }
        }
        D3D11_TEXTURE2D_DESC sd2 = td;
        sd2.Usage = D3D11_USAGE_STAGING; sd2.BindFlags = 0; sd2.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        dev->CreateTexture2D(&sd2, nullptr, &sbs_stage);
        vp = {0, 0, (float) sbs_w, (float) sbs_h, 0, 1};
        BOOST_LOG(info) << "sbs-bench: input " << img.w << "x" << img.h << " -> SBS "
                        << sbs_w << "x" << sbs_h
                        << (o.simulate_hdr ? " (linear scRGB FP16 HDR simulation)" : " (sRGB SDR)");
      }

      // Submit and consume exactly one inference for this source frame.
      const auto input_color = o.simulate_hdr ? models::input_color_space::scrgb_hdr
                                               : models::input_color_space::srgb;
      estimator.estimate_depth(in_srv.Get(), input_color);
      auto est = estimator.finish_pending_depth_for_benchmark(in_srv.Get(), input_color);

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
      if (sbs_cfg.warp == "vd3d" && est.depth) {
        const UINT clear_winner[4] = {0, 0, 0, 0};
        ctx->ClearUnorderedAccessViewUint(vd3d_winner_uav.Get(), clear_winner);
        ctx->CSSetShader(vd3d_cs.Get(), nullptr, 0);
        ctx->CSSetSamplers(0, 1, sampler.GetAddressOf());
        ID3D11ShaderResourceView *cs_srvs[] = {in_srv.Get(), est.depth.Get(), est.subject.Get(),
          nullptr, est.plane_lock.Get()};
        ctx->CSSetShaderResources(0, 5, cs_srvs);
        ctx->CSSetUnorderedAccessViews(0, 1, vd3d_winner_uav.GetAddressOf(), nullptr);
        ctx->CSSetConstantBuffers(2, 1, repro_cb.GetAddressOf());
        ctx->Dispatch(((sbs_w / 2u) + 15u) / 16u, (sbs_h + 15u) / 16u, 1u);
        ID3D11UnorderedAccessView *null_uav[] = {nullptr};
        ID3D11ShaderResourceView *null_cs_srvs[] = {nullptr, nullptr, nullptr, nullptr, nullptr};
        ctx->CSSetUnorderedAccessViews(0, 1, null_uav, nullptr);
        ctx->CSSetShaderResources(0, 5, null_cs_srvs);
      }
      ctx->OMSetRenderTargets(1, sbs_rtv.GetAddressOf(), nullptr);
      ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      ctx->VSSetShader(vs.Get(), nullptr, 0);
      ctx->PSSetShader(sbs_cfg.warp == "vd3d" && est.depth ? vd3d_ps.Get() : ps.Get(), nullptr, 0);
      ctx->RSSetViewports(1, &vp);
      ctx->PSSetSamplers(0, 1, sampler.GetAddressOf());

      ID3D11ShaderResourceView *srvs[] = {in_srv.Get(), est.depth.Get(), est.subject.Get(),
        sbs_cfg.warp == "vd3d" && est.depth ? vd3d_winner_srv.Get() : nullptr, est.plane_lock.Get()};
      ctx->PSSetShaderResources(0, 5, srvs);
      ID3D11Buffer *cb = est.depth ? repro_cb.Get() : pass_cb.Get();
      ctx->PSSetConstantBuffers(2, 1, &cb);
      ctx->Draw(3, 0);

      ID3D11RenderTargetView *null_rtv[] = {nullptr};
      ctx->OMSetRenderTargets(1, null_rtv, nullptr);
      ID3D11ShaderResourceView *null_srv[] = {nullptr, nullptr, nullptr, nullptr, nullptr};
      ctx->PSSetShaderResources(0, 5, null_srv);
      ID3D11Texture2D *final_sbs_tex = sbs_tex.Get();
      ID3D11ShaderResourceView *post_input_srv = sbs_srv.Get();
      if (est.depth && sharpen_rtv && sharpen_srv) {
        ctx->OMSetRenderTargets(1, sharpen_rtv.GetAddressOf(), nullptr);
        ctx->VSSetShader(vs.Get(), nullptr, 0);
        ctx->PSSetShader(sharpen_ps.Get(), nullptr, 0);
        ctx->RSSetViewports(1, &vp);
        ctx->PSSetShaderResources(0, 1, &post_input_srv);
        ctx->Draw(3, 0);
        ctx->OMSetRenderTargets(1, null_rtv, nullptr);
        ctx->PSSetShaderResources(0, 1, null_srv);
        final_sbs_tex = sharpen_tex.Get();
      }
      if (time_warp) {
        ctx->End(warp_end.Get());
        ctx->End(warp_disjoint.Get());
      }

      // Real composite-submission CPU cost. GPU warp time is captured separately below with D3D
      // timestamp queries; tick() advances the perf window.
      sbs_perf::add_sample_ms("sbs_composite_cpu",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - comp_t0).count());
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
          sbs_perf::add_sample_ms("warp_infer",
            (double)(end_tick - start_tick) * 1000.0 / (double)timing.Frequency);
        }
      }
      sbs_perf::tick();

      // Readback -> PNG.
      ctx->CopyResource(sbs_stage.Get(), final_sbs_tex);
      D3D11_MAPPED_SUBRESOURCE m = {};
      if (SUCCEEDED(ctx->Map(sbs_stage.Get(), 0, D3D11_MAP_READ, 0, &m))) {
        std::vector<uint8_t> buf((size_t) sbs_w * sbs_h * 4);
        if (o.simulate_hdr) {
          for (UINT y = 0; y < sbs_h; ++y) {
            const uint16_t *row = (const uint16_t *)((const uint8_t *)m.pData + (size_t)y * m.RowPitch);
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
              hdr_preview_bgra(r, g, b, &buf[((size_t)y * sbs_w + x) * 4]);
            }
          }
        } else {
          for (UINT y = 0; y < sbs_h; y++)
            memcpy(&buf[(size_t) y * sbs_w * 4], (uint8_t *) m.pData + (size_t) y * m.RowPitch, sbs_w * 4);
        }
        ctx->Unmap(sbs_stage.Get(), 0);
        const std::string id = frame_id(frames[fi], fi);
        char name[64];
        snprintf(name, sizeof(name), "sbs_%s.png", id.c_str());
        if (save_png(fs::path(o.out) / name, sbs_w, sbs_h, buf)) written++;
        char dname[64];
        snprintf(dname, sizeof(dname), "depth_%s.png", id.c_str());
        dump_depth(dev.Get(), ctx.Get(), est.depth.Get(), fs::path(o.out) / dname, depth_stage);
        char rname[64];
        snprintf(rname, sizeof(rname), "raw_%s.f32", id.c_str());
        dump_raw_model_depth(dev.Get(), ctx.Get(), est.raw_model_depth.Get(), est.raw_width,
          est.raw_height, fs::path(o.out) / rname, raw_depth_stage);
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
      if (((fi + 1) % 20) == 0)
        BOOST_LOG(info) << "sbs-bench: " << (fi + 1) << "/" << frames.size();
    }

    sbs_perf::dump_json((fs::path(o.out) / "sbs_perf.json").string());
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
  int run(int, char **) { return 1; }
}
#endif
