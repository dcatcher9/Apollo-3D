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
#include <chrono>
#include <cstring>
#include <filesystem>
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

    ComPtr<ID3D11Buffer> const_buffer(ID3D11Device *dev, const float params[8]) {
      D3D11_BUFFER_DESC bd = {};
      bd.ByteWidth = 32;  // 8 floats, 16-byte aligned
      bd.Usage = D3D11_USAGE_IMMUTABLE;
      bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      D3D11_SUBRESOURCE_DATA sd = {params, 0, 0};
      ComPtr<ID3D11Buffer> b;
      dev->CreateBuffer(&bd, &sd, &b);
      return b;
    }

    // ---- argument parsing ----

    struct opts {
      std::string frames, out, model;
      bool movie = false;
      int eye_h = 1728;
      int max_width = 0;   // 0 -> use config max_encode_width
      int settle = 3;      // estimate passes per frame so async depth/warp catch up
      int settle_ms = 40;  // sleep between passes so the CUDA streams finish
      int limit = 0;       // 0 -> all
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
        else if (a == "--movie") o.movie = true;
        else if (a == "--eye-h") o.eye_h = std::stoi(next("--eye-h"));
        else if (a == "--max-width") o.max_width = std::stoi(next("--max-width"));
        else if (a == "--settle") o.settle = std::max(1, std::stoi(next("--settle")));
        else if (a == "--settle-ms") o.settle_ms = std::stoi(next("--settle-ms"));
        else if (a == "--limit") o.limit = std::stoi(next("--limit"));
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
      if (want.empty() && o.movie) want = "da3mono_large_fp16";
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

    // Collect + sort input frames (any *.png in the folder).
    std::vector<fs::path> frames;
    std::error_code ec;
    for (auto &e : fs::directory_iterator(o.frames, ec))
      if (e.is_regular_file() && e.path().extension() == ".png")
        frames.push_back(e.path());
    std::sort(frames.begin(), frames.end());
    if (o.limit > 0 && (int) frames.size() > o.limit) frames.resize(o.limit);
    if (frames.empty()) { BOOST_LOG(error) << "sbs-bench: no .png frames in " << o.frames; return 4; }
    fs::create_directories(o.out, ec);

    // Config: inherit whatever the loaded conf set (learned_warp / warp models / divergence...),
    // then apply the movie-mode warp/depth overrides exactly like display_vram::ensure_depth_estimator.
    auto sbs_cfg = config::video.sbs;
    if (o.movie) {
      if (!sbs_cfg.warp_model_movie.empty()) sbs_cfg.warp_model = sbs_cfg.warp_model_movie;
      if (sbs_cfg.movie_depth_fps > 0.0) sbs_cfg.depth_fps = sbs_cfg.movie_depth_fps;
    }
    sbs_cfg.perf_stats = true;  // the harness always measures
    sbs_perf::set_enabled(true);
    sbs_perf::reset();
    auto model = pick_model(o);
    const int max_width = o.max_width > 0 ? o.max_width : config::video.sbs.max_encode_width;

    BOOST_LOG(info) << "sbs-bench: " << frames.size() << " frames, model '" << model.name
                    << "', warp '" << (sbs_cfg.learned_warp ? sbs_cfg.warp_model : std::string("probe"))
                    << "', eye_h " << o.eye_h << ", settle " << o.settle << " -> " << o.out;

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

    auto vs_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_reprojection_vs.hlsl", "main_vs", "vs_5_0");
    auto ps_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_reprojection_ps.hlsl", "main_ps", "ps_5_0");
    auto mlbw_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_mlbw_composite_ps.hlsl", "main_ps", "ps_5_0");
    if (!vs_blob || !ps_blob || !mlbw_blob) return 6;
    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> ps, mlbw_ps;
    dev->CreateVertexShader(vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr, &vs);
    dev->CreatePixelShader(ps_blob->GetBufferPointer(), ps_blob->GetBufferSize(), nullptr, &ps);
    dev->CreatePixelShader(mlbw_blob->GetBufferPointer(), mlbw_blob->GetBufferSize(), nullptr, &mlbw_ps);

    ComPtr<ID3D11SamplerState> sampler;
    {
      D3D11_SAMPLER_DESC sd = {};
      sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
      sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
      sd.ComparisonFunc = D3D11_COMPARISON_NEVER;
      sd.MaxLOD = D3D11_FLOAT32_MAX;
      dev->CreateSamplerState(&sd, &sampler);
    }

    // Reprojection constants: {divergence, focal, parallax_steps, border_fade, depth_floor, 0,0,0}.
    float repro_params[8] = {(float) sbs_cfg.divergence, (float) sbs_cfg.focal_plane,
      (float) sbs_cfg.parallax_steps, (float) sbs_cfg.border_fade, (float) sbs_cfg.depth_floor, 0, 0, 0};
    auto repro_cb = const_buffer(dev.Get(), repro_params);
    float pass_params[8] = {0, (float) sbs_cfg.focal_plane, (float) sbs_cfg.parallax_steps,
      (float) sbs_cfg.border_fade, (float) sbs_cfg.depth_floor, 0, 0, 0};
    auto pass_cb = const_buffer(dev.Get(), pass_params);

    // ---- estimator ----
    models::video_depth_estimator estimator(dev, ctx, fs::path(SUNSHINE_ASSETS_DIR), sbs_cfg, model);

    // Per-run state built lazily on the first frame (once we know the input size).
    ComPtr<ID3D11Texture2D> sbs_tex, sbs_stage;
    ComPtr<ID3D11RenderTargetView> sbs_rtv;
    ComPtr<ID3D11ShaderResourceView> sbs_srv;
    D3D11_VIEWPORT vp = {};
    ComPtr<ID3D11Buffer> mlbw_cb;
    int mlbw_fw = 0, mlbw_fh = 0, mlbw_layers = 0;
    UINT sbs_w = 0, sbs_h = 0;

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

      // First frame: size the SBS target from the input aspect (mirrors the live capping).
      if (!sbs_tex) {
        float aspect = (float) img.w / (float) img.h;
        int eye_w = (int) std::lround(o.eye_h * aspect);
        if (2 * eye_w > max_width) eye_w = max_width / 2;
        sbs_w = (UINT) (2 * eye_w); sbs_h = (UINT) o.eye_h;
        D3D11_TEXTURE2D_DESC td = {};
        td.Width = sbs_w; td.Height = sbs_h; td.MipLevels = 1; td.ArraySize = 1;
        td.Format = DXGI_FORMAT_B8G8R8A8_UNORM; td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        dev->CreateTexture2D(&td, nullptr, &sbs_tex);
        dev->CreateRenderTargetView(sbs_tex.Get(), nullptr, &sbs_rtv);
        dev->CreateShaderResourceView(sbs_tex.Get(), nullptr, &sbs_srv);
        D3D11_TEXTURE2D_DESC sd2 = td;
        sd2.Usage = D3D11_USAGE_STAGING; sd2.BindFlags = 0; sd2.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        dev->CreateTexture2D(&sd2, nullptr, &sbs_stage);
        vp = {0, 0, (float) sbs_w, (float) sbs_h, 0, 1};
        BOOST_LOG(info) << "sbs-bench: input " << img.w << "x" << img.h << " -> SBS " << sbs_w << "x" << sbs_h;
      }

      // Settle: run the estimator a few times so the ASYNC depth (and MLBW warp fields, which
      // land one call after their depth update) catch up to THIS frame instead of lagging by one.
      models::estimate_result est;
      for (int s = 0; s < o.settle; s++) {
        est = estimator.estimate_depth(in_srv.Get(), /*is_hdr=*/false);
        if (s + 1 < o.settle) std::this_thread::sleep_for(std::chrono::milliseconds(o.settle_ms));
      }

      const bool use_mlbw = est.delta_left && est.weight_left && est.delta_right && est.weight_right;

      // Composite (mirrors display_vram::convert()'s SBS block).
      ctx->OMSetRenderTargets(1, sbs_rtv.GetAddressOf(), nullptr);
      ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      ctx->VSSetShader(vs.Get(), nullptr, 0);
      ctx->PSSetShader(use_mlbw ? mlbw_ps.Get() : ps.Get(), nullptr, 0);
      ctx->RSSetViewports(1, &vp);
      ctx->PSSetSamplers(0, 1, sampler.GetAddressOf());

      if (use_mlbw) {
        if (!mlbw_cb || mlbw_fw != est.field_w || mlbw_fh != est.field_h || mlbw_layers != est.layers) {
          const float eye_w = sbs_w / 2.0f, eye_h = (float) sbs_h;
          const float delta_to_u = (eye_w - 1.0f) / (2.0f * (float) (est.field_w / 2 - 1)) / eye_w;
          float p[8] = {eye_w, eye_h, (float) est.field_w, (float) est.field_h, delta_to_u,
            (float) est.layers, 0, 0};
          mlbw_cb = const_buffer(dev.Get(), p);
          mlbw_fw = est.field_w; mlbw_fh = est.field_h; mlbw_layers = est.layers;
        }
        ID3D11ShaderResourceView *srvs[] = {in_srv.Get(), est.delta_left.Get(), est.weight_left.Get(),
          est.delta_right.Get(), est.weight_right.Get()};
        ctx->PSSetShaderResources(0, 5, srvs);
        ctx->PSSetConstantBuffers(2, 1, mlbw_cb.GetAddressOf());
      } else {
        ID3D11ShaderResourceView *srvs[] = {in_srv.Get(), est.depth.Get()};
        ctx->PSSetShaderResources(0, 2, srvs);
        ID3D11Buffer *cb = est.depth ? repro_cb.Get() : pass_cb.Get();
        ctx->PSSetConstantBuffers(2, 1, &cb);
      }
      ctx->Draw(3, 0);

      ID3D11RenderTargetView *null_rtv[] = {nullptr};
      ctx->OMSetRenderTargets(1, null_rtv, nullptr);
      ID3D11ShaderResourceView *null_srv[] = {nullptr, nullptr, nullptr, nullptr, nullptr};
      ctx->PSSetShaderResources(0, 5, null_srv);

      sbs_perf::add_sample_ms("sbs_convert_cpu", 0.0);  // keep the frame counter moving for tick()
      sbs_perf::tick();

      // Readback -> PNG.
      ctx->CopyResource(sbs_stage.Get(), sbs_tex.Get());
      D3D11_MAPPED_SUBRESOURCE m = {};
      if (SUCCEEDED(ctx->Map(sbs_stage.Get(), 0, D3D11_MAP_READ, 0, &m))) {
        std::vector<uint8_t> buf((size_t) sbs_w * sbs_h * 4);
        for (UINT y = 0; y < sbs_h; y++)
          memcpy(&buf[(size_t) y * sbs_w * 4], (uint8_t *) m.pData + (size_t) y * m.RowPitch, sbs_w * 4);
        ctx->Unmap(sbs_stage.Get(), 0);
        char name[32];
        snprintf(name, sizeof(name), "sbs_%05zu.png", fi);
        if (save_png(fs::path(o.out) / name, sbs_w, sbs_h, buf)) written++;
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
