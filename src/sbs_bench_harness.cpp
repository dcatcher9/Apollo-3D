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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
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
      int eye_h = 0;       // 0 -> match the input frame height (so the input size controls eval
                           // resolution/speed; a bigger clip = a bigger eval). Override to pin one.
      int max_width = 0;   // 0 -> use config max_encode_width
      int limit = 0;       // 0 -> all
      int output_every = 1;  // process every input for temporal state; dump only every Nth
      double divergence = -1.0;  // <0 -> use the conf's value; else override (parallax/disocclusion size)
      // VD3D-pipeline A/B levers; <0 / false -> use the conf's value.
      double pct_lo = -1.0;      // robust normalization low percentile (e.g. 1.0)
      double pct_hi = -1.0;      // robust normalization high percentile (e.g. 99.0)
      bool subject_track = false;  // VD3D-style shaped disparity + subject anchoring
      double subject_lock = -1.0;  // subject anchor strength override (e.g. 0.95)
      int depth_short_side = 0;    // depth inference short-side override (0 = conf; VD3D uses 432)
      double ema = -1.0;         // per-pixel depth EMA override (1.0 = off)
      bool subject_stretch = false;  // VD3D shape_depth_for_pop 5/95 disparity stretch
      double subject_plane_lock = -1.0;  // local subject-band flatten (e.g. 0.28); <0 = conf
      double curvature = -1.0;     // foreground curvature strength (e.g. 0.07); <0 = conf
      double dof = -1.0;           // depth-of-field strength (e.g. 0.3); <0 = conf
      double vd3d_forward_blend = -1.0;  // VD3D backward/forward blend override; <0 = conf
      double minmax_ema = -1.0;    // range-bounds EMA new-weight (VD3D DepthPercentileEMA a=0.82 -> 0.18); <0 = conf
      int guided = -1;             // guided upsample: -1 = conf, 0 = off, 1 = on
      bool pixel_ema_first = false;  // VD3D EMA order: per-pixel smooth raw BEFORE normalizing
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
        else if (a == "--eye-h") o.eye_h = std::stoi(next("--eye-h"));
        else if (a == "--max-width") o.max_width = std::stoi(next("--max-width"));
        else if (a == "--limit") o.limit = std::stoi(next("--limit"));
        else if (a == "--output-every") o.output_every = std::max(1, std::stoi(next("--output-every")));
        else if (a == "--divergence") o.divergence = std::stod(next("--divergence"));
        else if (a == "--pct-lo") o.pct_lo = std::stod(next("--pct-lo"));
        else if (a == "--pct-hi") o.pct_hi = std::stod(next("--pct-hi"));
        else if (a == "--subject-track") o.subject_track = true;
        else if (a == "--subject-lock") o.subject_lock = std::stod(next("--subject-lock"));
        else if (a == "--depth-short-side") o.depth_short_side = std::stoi(next("--depth-short-side"));
        else if (a == "--subject-stretch") o.subject_stretch = true;
        else if (a == "--subject-plane-lock") o.subject_plane_lock = std::stod(next("--subject-plane-lock"));
        else if (a == "--curvature") o.curvature = std::stod(next("--curvature"));
        else if (a == "--dof") o.dof = std::stod(next("--dof"));
        else if (a == "--vd3d-forward-blend") o.vd3d_forward_blend = std::stod(next("--vd3d-forward-blend"));
        else if (a == "--ema") o.ema = std::stod(next("--ema"));
        else if (a == "--minmax-ema") o.minmax_ema = std::stod(next("--minmax-ema"));
        else if (a == "--guided-upsample") {
          std::string v = next("--guided-upsample");
          o.guided = (v == "off" || v == "0" || v == "false") ? 0 : 1;
        }
        else if (a == "--pixel-ema-first") o.pixel_ema_first = true;
        else { BOOST_LOG(error) << "sbs-bench: unknown arg '" << a << "'"; return false; }
      }
      if (o.frames.empty() || o.out.empty()) {
        BOOST_LOG(error) << "sbs-bench: --frames DIR and --out DIR are required";
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
    if (o.subject_track) sbs_cfg.subject_track = true;          // A/B lever: shaped disparity
    if (o.subject_lock >= 0.0) sbs_cfg.subject_lock = o.subject_lock;
    if (o.depth_short_side > 0) sbs_cfg.depth_short_side = o.depth_short_side;  // VD3D uses 432
    if (o.subject_stretch) sbs_cfg.subject_stretch = true;      // A/B lever: shape_depth_for_pop stretch
    if (o.subject_plane_lock >= 0.0) sbs_cfg.subject_plane_lock = o.subject_plane_lock;
    if (o.curvature >= 0.0) sbs_cfg.foreground_curvature = o.curvature;
    if (o.dof >= 0.0) sbs_cfg.dof_strength = o.dof;
    if (o.vd3d_forward_blend >= 0.0) sbs_cfg.vd3d_forward_blend = o.vd3d_forward_blend;
    if (o.ema > 0.0) sbs_cfg.ema = o.ema;                        // A/B lever: depth EMA (1.0 = off)
    if (o.minmax_ema >= 0.0) sbs_cfg.minmax_ema = o.minmax_ema;  // A/B lever: range-bounds EMA (VD3D 0.18)
    if (o.guided >= 0) sbs_cfg.guided_upsample = (o.guided != 0);  // A/B lever: guided upsample on/off
    if (o.pixel_ema_first) sbs_cfg.ema_pixel_first = true;         // A/B lever: VD3D pixel->range EMA order
    sbs_cfg.perf_stats = true;  // the harness always measures
    sbs_perf::set_enabled(true);
    sbs_perf::reset();
    auto model = pick_model(o);
    const int max_width = o.max_width > 0 ? o.max_width : config::video.sbs.max_encode_width;

    BOOST_LOG(info) << "sbs-bench: " << frames.size() << " frames, model '" << model.name
                    << "', eye_h " << (o.eye_h > 0 ? std::to_string(o.eye_h) : "match-input")
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
    auto vd3d_cs_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_vd3d_forward_cs.hlsl", "main", "cs_5_0");
    auto vd3d_ps_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_vd3d_reprojection_ps.hlsl", "main_ps", "ps_5_0");
    if (!vs_blob || !ps_blob || !vd3d_cs_blob || !vd3d_ps_blob) return 6;
    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> ps, vd3d_ps;
    ComPtr<ID3D11ComputeShader> vd3d_cs;
    dev->CreateVertexShader(vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr, &vs);
    dev->CreatePixelShader(ps_blob->GetBufferPointer(), ps_blob->GetBufferSize(), nullptr, &ps);
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

    // Shared disparity constants. The last register carries VD3D's hybrid/fill parameters.
    float repro_params[16] = {(float) sbs_cfg.divergence, (float) sbs_cfg.focal_plane,
      (float) sbs_cfg.parallax_steps, (float) sbs_cfg.border_fade, (float) sbs_cfg.depth_floor,
      sbs_cfg.subject_track ? 1.0f : 0.0f, (float) sbs_cfg.subject_lock,
      sbs_cfg.subject_stretch ? 1.0f : 0.0f, (float) sbs_cfg.subject_plane_lock,
      (float) sbs_cfg.subject_plane_width, (float) sbs_cfg.dof_strength,
      (float) sbs_cfg.dof_focus_width, (float) sbs_cfg.vd3d_forward_blend,
      (float) sbs_cfg.vd3d_fill_radius, sbs_cfg.shift_profile == "bestv2" ? 1.0f : 0.0f, 0.0f};
    auto repro_cb = const_buffer(dev.Get(), repro_params);
    float pass_params[16] = {0, (float) sbs_cfg.focal_plane, (float) sbs_cfg.parallax_steps,
      (float) sbs_cfg.border_fade, (float) sbs_cfg.depth_floor, 0, 0, 0, 0, 0, 0, 0,
      (float) sbs_cfg.vd3d_forward_blend, (float) sbs_cfg.vd3d_fill_radius, 0, 0};
    auto pass_cb = const_buffer(dev.Get(), pass_params);

    // ---- estimator ----
    models::video_depth_estimator estimator(dev, ctx, fs::path(SUNSHINE_ASSETS_DIR), sbs_cfg, model);

    // Per-run state built lazily on the first frame (once we know the input size).
    ComPtr<ID3D11Texture2D> sbs_tex, sbs_stage;
    ComPtr<ID3D11RenderTargetView> sbs_rtv;
    ComPtr<ID3D11Texture2D> vd3d_winner_tex;
    ComPtr<ID3D11UnorderedAccessView> vd3d_winner_uav;
    ComPtr<ID3D11ShaderResourceView> vd3d_winner_srv;
    D3D11_VIEWPORT vp = {};
    UINT sbs_w = 0, sbs_h = 0;
    ComPtr<ID3D11Texture2D> depth_stage;  // dump_depth staging cache (depth size is constant)
    ComPtr<ID3D11Buffer> raw_depth_stage;
    bool raw_shape_written = false;

    int written = 0;
    for (size_t fi = 0; fi < frames.size(); fi++) {
      rgba_image img;
      if (!load_png(frames[fi], img)) { BOOST_LOG(warning) << "sbs-bench: skip " << frames[fi]; continue; }

      // Input texture + SRV.
      D3D11_TEXTURE2D_DESC id = {};
      id.Width = img.w; id.Height = img.h; id.MipLevels = 1; id.ArraySize = 1;
      id.Format = DXGI_FORMAT_B8G8R8A8_UNORM; id.SampleDesc.Count = 1;
      id.Usage = D3D11_USAGE_IMMUTABLE; id.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA isd = {img.bgra.data(), img.w * 4, 0};
      ComPtr<ID3D11Texture2D> in_tex;
      if (FAILED(dev->CreateTexture2D(&id, &isd, &in_tex))) { BOOST_LOG(error) << "sbs-bench: input tex fail"; continue; }
      ComPtr<ID3D11ShaderResourceView> in_srv;
      dev->CreateShaderResourceView(in_tex.Get(), nullptr, &in_srv);

      // First frame: size the SBS target. Per eye = the input resolution by default (so the clip
      // size, not a fixed constant, drives eval cost); --eye-h pins a specific output height.
      // The width is still capped at max_encode_width like the live path.
      if (!sbs_tex) {
        int eh_target = o.eye_h > 0 ? o.eye_h : (int) img.h;
        float aspect = (float) img.w / (float) img.h;
        int eye_w = (int) std::lround(eh_target * aspect);
        if (2 * eye_w > max_width) eye_w = max_width / 2;
        sbs_w = (UINT) (2 * eye_w); sbs_h = (UINT) eh_target;
        D3D11_TEXTURE2D_DESC td = {};
        td.Width = sbs_w; td.Height = sbs_h; td.MipLevels = 1; td.ArraySize = 1;
        td.Format = DXGI_FORMAT_B8G8R8A8_UNORM; td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        dev->CreateTexture2D(&td, nullptr, &sbs_tex);
        dev->CreateRenderTargetView(sbs_tex.Get(), nullptr, &sbs_rtv);
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
        BOOST_LOG(info) << "sbs-bench: input " << img.w << "x" << img.h << " -> SBS " << sbs_w << "x" << sbs_h;
      }

      // Submit and consume exactly one inference for this source frame.
      estimator.estimate_depth(in_srv.Get(), /*is_hdr=*/false);
      auto est = estimator.finish_pending_depth_for_benchmark(in_srv.Get(), /*is_hdr=*/false);

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
        ID3D11ShaderResourceView *cs_srvs[] = {est.depth.Get(), est.subject.Get(), nullptr, est.plane_lock.Get()};
        ctx->CSSetShaderResources(1, 4, cs_srvs);
        ctx->CSSetUnorderedAccessViews(0, 1, vd3d_winner_uav.GetAddressOf(), nullptr);
        ctx->CSSetConstantBuffers(2, 1, repro_cb.GetAddressOf());
        ctx->Dispatch(((sbs_w / 2u) + 15u) / 16u, (sbs_h + 15u) / 16u, 1u);
        ID3D11UnorderedAccessView *null_uav[] = {nullptr};
        ID3D11ShaderResourceView *null_cs_srvs[] = {nullptr, nullptr, nullptr, nullptr};
        ctx->CSSetUnorderedAccessViews(0, 1, null_uav, nullptr);
        ctx->CSSetShaderResources(1, 4, null_cs_srvs);
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
      ctx->CopyResource(sbs_stage.Get(), sbs_tex.Get());
      D3D11_MAPPED_SUBRESOURCE m = {};
      if (SUCCEEDED(ctx->Map(sbs_stage.Get(), 0, D3D11_MAP_READ, 0, &m))) {
        std::vector<uint8_t> buf((size_t) sbs_w * sbs_h * 4);
        for (UINT y = 0; y < sbs_h; y++)
          memcpy(&buf[(size_t) y * sbs_w * 4], (uint8_t *) m.pData + (size_t) y * m.RowPitch, sbs_w * 4);
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
    BOOST_LOG(info) << "sbs-bench: wrote " << written << " SBS frames + sbs_perf.json to " << o.out;
    return written > 0 ? 0 : 8;
  }

}  // namespace sbs_bench

#else  // !_WIN32
namespace sbs_bench {
  int run(int, char **) { return 1; }
}
#endif
