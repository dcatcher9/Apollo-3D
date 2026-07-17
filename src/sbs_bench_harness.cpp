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
  #include <bit>
  #include <cctype>
  #include <chrono>
  #include <cmath>
  #include <condition_variable>
  #include <cstdint>
  #include <cstring>
  #include <deque>
  #include <filesystem>
  #include <fstream>
  #include <iomanip>
  #include <iterator>
  #include <limits>
  #include <memory>
  #include <mutex>
  #include <set>
  #include <string>
  #include <string_view>
  #include <thread>
  #include <utility>
  #include <vector>

  // platform includes
  #include <d3d11.h>
  #include <d3dcompiler.h>
  #include <nlohmann/json.hpp>
  #include <wincodec.h>
  #include <wrl/client.h>

  // local includes
  #include "config.h"
  #include "crypto.h"
  #include "logging.h"
  #include "sbs_perf.h"
  #include "sbs_depth_state_sequence.h"
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

    void simulated_hdr_metric_bgra(float r, float g, float b, float input_scale, uint8_t *out) {
      // --simulate-hdr starts from an SDR source and represents Windows' SDR-in-HDR
      // compositor by multiplying linear Rec.709 by the configured scRGB white scale.  The
      // SBS texture must retain those absolute scRGB values, but evaluator PNGs are compared
      // with the original SDR frame.  Undo only that known compositor scale before the sRGB
      // encoding.  Tone-mapping the absolute scRGB output here makes source-relative coverage,
      // halo, stretch, and temporal metrics change with Windows' white slider even when the
      // warp itself is identical.
      const float inverse_scale =
        std::isfinite(input_scale) && input_scale > 0.0f ? 1.0f / input_scale : 1.0f;
      r = std::isfinite(r) ? std::max(r * inverse_scale, 0.0f) : 0.0f;
      g = std::isfinite(g) ? std::max(g * inverse_scale, 0.0f) : 0.0f;
      b = std::isfinite(b) ? std::max(b * inverse_scale, 0.0f) : 0.0f;
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

    void native_hdr_metric_bgra(float r, float g, float b, uint8_t *out) {
      // Match DepthHdrScRgbToSrgb() exactly for finite source values. Native-PQ sources have
      // already been decoded into absolute linear Rec.709 scRGB (1.0 = 80 nits), so unlike the
      // simulated SDR-in-HDR path there is no source-white multiplier to undo.
      r = std::isfinite(r) ? std::max(r, 0.0f) : 0.0f;
      g = std::isfinite(g) ? std::max(g, 0.0f) : 0.0f;
      b = std::isfinite(b) ? std::max(b, 0.0f) : 0.0f;
      const float luminance = std::max(
        0.2126f * r + 0.7152f * g + 0.0722f * b,
        0.0f
      );
      r /= 1.0f + luminance;
      g /= 1.0f + luminance;
      b /= 1.0f + luminance;
      const float peak = std::max(r, std::max(g, b));
      const float peak_scale = std::max(peak, 1.0f);
      r /= peak_scale;
      g /= peak_scale;
      b /= peak_scale;
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

    bool load_scrgb16(const fs::path &path, UINT width, UINT height,
                      std::vector<uint16_t> &rgba) {
      const std::uint64_t expected_size =
        static_cast<std::uint64_t>(width) * height * 4u * sizeof(uint16_t);
      std::error_code file_error;
      const std::uint64_t actual_size = fs::file_size(path, file_error);
      if (file_error || actual_size != expected_size ||
          expected_size > static_cast<std::uint64_t>(
                            std::numeric_limits<std::streamsize>::max())) {
        BOOST_LOG(error) << "sbs-bench: native HDR sidecar size differs for " << path
                         << "; expected " << expected_size << " bytes for " << width << 'x'
                         << height << " RGBA16F, found "
                         << (file_error ? std::string("unavailable") :
                                          std::to_string(actual_size));
        return false;
      }
      std::ifstream stream(path, std::ios::binary);
      if (!stream) {
        BOOST_LOG(error) << "sbs-bench: cannot open native HDR sidecar " << path;
        return false;
      }
      rgba.resize(static_cast<size_t>(width) * height * 4u);
      stream.read(
        reinterpret_cast<char *>(rgba.data()),
        static_cast<std::streamsize>(expected_size)
      );
      if (!stream || stream.gcount() != static_cast<std::streamsize>(expected_size)) {
        BOOST_LOG(error) << "sbs-bench: cannot read complete native HDR sidecar " << path;
        return false;
      }
      return true;
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

    bool save_png_with_factory(IWICImagingFactory *factory, const fs::path &path,
                               UINT w, UINT h, const std::vector<uint8_t> &bgra) {
      if (!factory) {
        return false;
      }
      ComPtr<IWICStream> stream;
      if (FAILED(factory->CreateStream(&stream))) {
        return false;
      }
      if (FAILED(stream->InitializeFromFilename(path.wstring().c_str(), GENERIC_WRITE))) {
        return false;
      }
      ComPtr<IWICBitmapEncoder> enc;
      if (FAILED(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc))) {
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

    bool save_png(const fs::path &path, UINT w, UINT h,
                  const std::vector<uint8_t> &bgra) {
      return save_png_with_factory(g_wic.Get(), path, w, h, bgra);
    }

    constexpr size_t multiscale_artifact_worker_count = 4;
    constexpr size_t multiscale_artifact_queue_capacity =
      2 * multiscale_artifact_worker_count;
    constexpr std::string_view multiscale_artifact_writer_contract =
      "apollo-bounded-multiscale-png-writer-v1";

    enum class artifact_pixel_encoding {
      bgra8,
      native_scrgb16,
      simulated_scrgb16,
    };

    struct artifact_write_job {
      fs::path path;
      UINT width = 0;
      UINT height = 0;
      artifact_pixel_encoding encoding = artifact_pixel_encoding::bgra8;
      float hdr_input_scale = 1.0f;
      bool count_sbs = false;
      std::vector<uint8_t> bgra;
      std::vector<uint16_t> scrgb;
    };

    struct artifact_writer_result {
      bool success = true;
      std::string error;
      size_t worker_count = 0;
      size_t queue_capacity = 0;
      size_t submitted_jobs = 0;
      size_t completed_jobs = 0;
      size_t written_sbs = 0;
      float hdr_min = std::numeric_limits<float>::infinity();
      float hdr_max = -std::numeric_limits<float>::infinity();
      std::uint64_t hdr_nonfinite = 0;
    };

    // Offline multiscale runs spend most of their wall time converting FP16 metric previews and
    // compressing PNGs. D3D11 immediate-context work and staging readback remain on the harness
    // thread; this bounded queue owns only tightly packed CPU buffers. Every worker initializes
    // its own COM apartment and WIC factory, avoiding unsafe cross-thread use of g_wic.
    class bounded_multiscale_artifact_writer {
     public:
      bounded_multiscale_artifact_writer(size_t worker_count, size_t queue_capacity) :
          queue_capacity_(queue_capacity) {
        result_.worker_count = worker_count;
        result_.queue_capacity = queue_capacity;
        if (worker_count == 0 || queue_capacity == 0) {
          result_.success = false;
          result_.error = "artifact writer requires positive worker and queue counts";
          closing_ = true;
          return;
        }
        workers_.reserve(worker_count);
        try {
          for (size_t index = 0; index < worker_count; ++index) {
            workers_.emplace_back([this]() {
              worker_loop();
            });
          }
        } catch (const std::exception &error) {
          {
            std::lock_guard lock(mutex_);
            closing_ = true;
            set_failure_locked(
              "artifact writer could not start every worker: " + std::string(error.what())
            );
          }
          work_available_.notify_all();
          space_available_.notify_all();
          for (auto &worker : workers_) {
            if (worker.joinable()) {
              worker.join();
            }
          }
        }
      }

      bounded_multiscale_artifact_writer(const bounded_multiscale_artifact_writer &) = delete;
      bounded_multiscale_artifact_writer &operator=(
        const bounded_multiscale_artifact_writer &
      ) = delete;

      ~bounded_multiscale_artifact_writer() {
        finish();
      }

      bool submit(artifact_write_job job) {
        std::unique_lock lock(mutex_);
        space_available_.wait(lock, [this]() {
          return !result_.success || closing_ || jobs_.size() < queue_capacity_;
        });
        if (!result_.success || closing_) {
          return false;
        }
        if (!submitted_paths_.insert(job.path).second) {
          set_failure_locked("artifact writer received a duplicate output path");
          return false;
        }
        jobs_.push_back(std::move(job));
        ++result_.submitted_jobs;
        work_available_.notify_one();
        return true;
      }

      artifact_writer_result finish() {
        {
          std::lock_guard finish_lock(finish_mutex_);
          if (!joined_) {
            {
              std::lock_guard lock(mutex_);
              closing_ = true;
            }
            work_available_.notify_all();
            space_available_.notify_all();
            for (auto &worker : workers_) {
              if (worker.joinable()) {
                worker.join();
              }
            }
            joined_ = true;
          }
        }
        std::lock_guard lock(mutex_);
        if (result_.success &&
            result_.completed_jobs != result_.submitted_jobs) {
          set_failure_locked("artifact writer drained an incomplete job set");
        }
        return result_;
      }

     private:
      void set_failure_locked(std::string error) {
        if (result_.success) {
          result_.success = false;
          result_.error = std::move(error);
        }
        work_available_.notify_all();
        space_available_.notify_all();
      }

      void set_failure(std::string error) {
        std::lock_guard lock(mutex_);
        set_failure_locked(std::move(error));
      }

      void worker_loop() {
        const HRESULT com_result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (FAILED(com_result)) {
          set_failure("artifact writer worker could not initialize COM");
          return;
        }
        ComPtr<IWICImagingFactory> factory;
        if (FAILED(CoCreateInstance(
              CLSID_WICImagingFactory,
              nullptr,
              CLSCTX_INPROC_SERVER,
              IID_PPV_ARGS(&factory)
            )) || !factory) {
          set_failure("artifact writer worker could not create a WIC factory");
          CoUninitialize();
          return;
        }

        while (true) {
          artifact_write_job job;
          {
            std::unique_lock lock(mutex_);
            work_available_.wait(lock, [this]() {
              return !result_.success || closing_ || !jobs_.empty();
            });
            if (!result_.success) {
              break;
            }
            if (jobs_.empty()) {
              if (closing_) {
                break;
              }
              continue;
            }
            job = std::move(jobs_.front());
            jobs_.pop_front();
            space_available_.notify_one();
          }

          float local_hdr_min = std::numeric_limits<float>::infinity();
          float local_hdr_max = -std::numeric_limits<float>::infinity();
          std::uint64_t local_hdr_nonfinite = 0;
          if (job.encoding != artifact_pixel_encoding::bgra8) {
            const size_t pixel_count = static_cast<size_t>(job.width) * job.height;
            if (job.scrgb.size() != pixel_count * 4u) {
              set_failure("artifact writer received an invalid FP16 pixel buffer");
              break;
            }
            job.bgra.resize(pixel_count * 4u);
            for (size_t pixel = 0; pixel < pixel_count; ++pixel) {
              const float r = half_to_float(job.scrgb[pixel * 4u + 0u]);
              const float g = half_to_float(job.scrgb[pixel * 4u + 1u]);
              const float b = half_to_float(job.scrgb[pixel * 4u + 2u]);
              for (float value : {r, g, b}) {
                if (std::isfinite(value)) {
                  local_hdr_min = std::min(local_hdr_min, value);
                  local_hdr_max = std::max(local_hdr_max, value);
                } else {
                  ++local_hdr_nonfinite;
                }
              }
              uint8_t *preview = &job.bgra[pixel * 4u];
              if (job.encoding == artifact_pixel_encoding::native_scrgb16) {
                native_hdr_metric_bgra(r, g, b, preview);
              } else {
                simulated_hdr_metric_bgra(r, g, b, job.hdr_input_scale, preview);
              }
            }
            job.scrgb.clear();
          } else if (job.bgra.size() !=
                     static_cast<size_t>(job.width) * job.height * 4u) {
            set_failure("artifact writer received an invalid BGRA8 pixel buffer");
            break;
          }

          if (!save_png_with_factory(
                factory.Get(), job.path, job.width, job.height, job.bgra
              )) {
            set_failure("artifact writer could not encode " + job.path.string());
            break;
          }

          {
            std::lock_guard lock(mutex_);
            ++result_.completed_jobs;
            if (job.count_sbs) {
              ++result_.written_sbs;
            }
            if (job.encoding != artifact_pixel_encoding::bgra8) {
              result_.hdr_min = std::min(result_.hdr_min, local_hdr_min);
              result_.hdr_max = std::max(result_.hdr_max, local_hdr_max);
              result_.hdr_nonfinite += local_hdr_nonfinite;
            }
          }
        }
        factory.Reset();
        CoUninitialize();
      }

      const size_t queue_capacity_;
      std::mutex mutex_;
      std::mutex finish_mutex_;
      std::condition_variable work_available_;
      std::condition_variable space_available_;
      std::deque<artifact_write_job> jobs_;
      std::set<fs::path> submitted_paths_;
      std::vector<std::thread> workers_;
      artifact_writer_result result_;
      bool closing_ = false;
      bool joined_ = false;
    };

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
    bool dump_depth(ID3D11Device *dev, ID3D11DeviceContext *ctx, ID3D11ShaderResourceView *srv, const fs::path &path, ComPtr<ID3D11Texture2D> &stage_cache) {
      if (!srv) {
        return false;
      }
      ComPtr<ID3D11Resource> res;
      srv->GetResource(&res);
      ComPtr<ID3D11Texture2D> tex;
      if (FAILED(res.As(&tex))) {
        return false;
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
          return false;
        }
      }
      ctx->CopyResource(stage_cache.Get(), tex.Get());
      D3D11_MAPPED_SUBRESOURCE m = {};
      if (FAILED(ctx->Map(stage_cache.Get(), 0, D3D11_MAP_READ, 0, &m))) {
        return false;
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
      return save_gray16_png(path, d.Width, d.Height, gray);
    }

    // Little-endian uint32 width/height followed by tightly packed row-major float32 values.
    // Unlike the PNG depth artifact, disparity is signed and must remain lossless for fitting.
    bool dump_float_texture(ID3D11Device *dev, ID3D11DeviceContext *ctx,
                            ID3D11Texture2D *texture, const fs::path &path,
                            ComPtr<ID3D11Texture2D> &stage_cache) {
      if (!texture) {
        return false;
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
          return false;
        }
      }
      ctx->CopyResource(stage_cache.Get(), texture);
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(ctx->Map(stage_cache.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        return false;
      }
      std::ofstream stream(path, std::ios::binary);
      stream.write((const char *) &desc.Width, sizeof(desc.Width));
      stream.write((const char *) &desc.Height, sizeof(desc.Height));
      for (UINT y = 0; y < desc.Height && stream; ++y) {
        stream.write((const char *) mapped.pData + (size_t) y * mapped.RowPitch,
                     (std::streamsize) desc.Width * sizeof(float));
      }
      ctx->Unmap(stage_cache.Get(), 0);
      return (bool) stream;
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
    // SDR/HDR format. D3D11 access stays on the caller thread; the returned vector can be moved
    // safely to an offline artifact worker.
    bool read_bgra8_texture(ID3D11Device *dev, ID3D11DeviceContext *ctx,
                            ID3D11Texture2D *texture,
                            ComPtr<ID3D11Texture2D> &stage_cache,
                            std::vector<uint8_t> &pixels) {
      if (!texture) {
        return false;
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
          return false;
        }
      }
      ctx->CopyResource(stage_cache.Get(), texture);
      D3D11_MAPPED_SUBRESOURCE mapped = {};
      if (FAILED(ctx->Map(stage_cache.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        return false;
      }
      pixels.resize((size_t) d.Width * d.Height * 4);
      for (UINT y = 0; y < d.Height; ++y) {
        std::memcpy(pixels.data() + (size_t) y * d.Width * 4,
                    (const uint8_t *) mapped.pData + (size_t) y * mapped.RowPitch,
                    (size_t) d.Width * 4);
      }
      ctx->Unmap(stage_cache.Get(), 0);
      return true;
    }

    void dump_bgra8_texture(ID3D11Device *dev, ID3D11DeviceContext *ctx,
                            ID3D11Texture2D *texture, const fs::path &path,
                            ComPtr<ID3D11Texture2D> &stage_cache) {
      std::vector<uint8_t> pixels;
      if (read_bgra8_texture(dev, ctx, texture, stage_cache, pixels)) {
        D3D11_TEXTURE2D_DESC desc {};
        texture->GetDesc(&desc);
        save_png(path, desc.Width, desc.Height, pixels);
      }
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

    bool numeric_frame_id(const fs::path &path, std::uint64_t &result) {
      const std::string stem = path.stem().string();
      if (stem.rfind("frame_", 0) != 0) {
        return false;
      }
      const std::string value = stem.substr(6);
      if (value.empty() || !std::all_of(value.begin(), value.end(), [](unsigned char character) {
            return std::isdigit(character);
          })) {
        return false;
      }
      try {
        size_t consumed = 0;
        result = std::stoull(value, &consumed);
        return consumed == value.size();
      } catch (const std::exception &) {
        return false;
      }
    }

    std::string sha256_hex(std::string_view value) {
      static constexpr char hex[] = "0123456789abcdef";
      const auto digest = crypto::hash(value);
      std::string result;
      result.reserve(digest.size() * 2);
      for (std::uint8_t byte : digest) {
        result.push_back(hex[byte >> 4]);
        result.push_back(hex[byte & 0x0f]);
      }
      return result;
    }

    bool load_label_frame_ids(const fs::path &path, std::vector<std::uint64_t> &frame_ids, std::string &manifest_sha256) {
      std::error_code file_error;
      const auto size = fs::file_size(path, file_error);
      if (file_error || size > 1024 * 1024) {
        BOOST_LOG(error) << "sbs-bench: missing or oversized label-frame manifest " << path;
        return false;
      }
      std::ifstream stream(path, std::ios::binary);
      if (!stream) {
        BOOST_LOG(error) << "sbs-bench: cannot open label-frame manifest " << path;
        return false;
      }
      const std::string contents(
        (std::istreambuf_iterator<char>(stream)),
        std::istreambuf_iterator<char>()
      );
      manifest_sha256 = sha256_hex(contents);
      try {
        const auto payload = nlohmann::json::parse(contents);
        if (!payload.is_object() || payload.size() != 2 || !payload.contains("schema") || !payload.contains("frame_ids") || !payload["schema"].is_number_integer() || payload["schema"].get<int>() != 1 || !payload["frame_ids"].is_array() || payload["frame_ids"].empty()) {
          BOOST_LOG(error) << "sbs-bench: label_frames.json must be schema 1 with a non-empty "
                              "frame_ids array and no unversioned fields";
          return false;
        }
        bool first = true;
        std::uint64_t previous = 0;
        for (const auto &item : payload["frame_ids"]) {
          std::uint64_t value = 0;
          if (item.is_number_unsigned()) {
            value = item.get<std::uint64_t>();
          } else if (item.is_number_integer()) {
            const auto signed_value = item.get<std::int64_t>();
            if (signed_value < 0) {
              BOOST_LOG(error) << "sbs-bench: label frame IDs must be nonnegative";
              return false;
            }
            value = static_cast<std::uint64_t>(signed_value);
          } else {
            BOOST_LOG(error) << "sbs-bench: label frame IDs must be integers";
            return false;
          }
          if (!first && value <= previous) {
            BOOST_LOG(error) << "sbs-bench: label frame IDs must be unique and strictly increasing";
            return false;
          }
          frame_ids.push_back(value);
          previous = value;
          first = false;
        }
      } catch (const std::exception &exception) {
        BOOST_LOG(error) << "sbs-bench: invalid label-frame manifest " << path << ": "
                         << exception.what();
        return false;
      }
      return true;
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
      if (FAILED(dev->CreateBuffer(&bd, &sd, &b))) {
        return nullptr;
      }
      return b;
    }

    // ---- argument parsing ----

    struct opts {
      std::string frames, out, model, depth_override_root;
      std::string depth_state_export_root, depth_state_replay_root, depth_state_cache_key;
      std::string depth_state_manifest_sha256;
      int eye_w = 0;  // 0 -> derive from source aspect; set with eye_h to test letterboxing
      int eye_h = 0;  // 0 -> match/derive from the input frame
      double output_scale = 1.0;  // per-eye linear scale vs source; preserves source aspect
      double pop_strength = -1.0;  // final shared stereo-parallax multiplier; <0 = conf
      double adaptive_pop_max = -1.0;  // absolute ceiling; <0 = conf
      std::string zero_plane;  // empty = conf; legacy, subject, median, or background
      bool simulate_hdr = false;  // decode sRGB frames into linear scRGB FP16 and use HDR paths
      bool native_hdr_scrgb = false;  // consume matching model_source/frame_*.scrgb16
      double hdr_scale = 4.0;  // scRGB multiplier after sRGB EOTF (4.0 = 320-nit diffuse white)
      int sdr_white_level_raw = 4000;  // Windows raw value; scRGB multiplier = raw / 1000
      int max_width = 0;  // 0 -> use config max_encode_width
      int limit = 0;  // 0 -> all
      int output_every = 1;  // process every input for temporal state; dump only every Nth
      bool output_gt_right_only = false;  // emit only identities with authored right-eye labels
      bool output_label_frames = false;  // emit exact identities from label_frames.json
      bool runtime_scene_evidence = false;  // read SubjectState after every completed depth frame
      int depth_every = 1;  // infer every Nth source frame; reuse depth between updates
      bool depth_only = false;  // training-data export: skip warp/SBS/raw artifacts
      bool artistic_policy = true;  // consume optional model control output; false is an ablation
      double artistic_scale_override = 0.0;  // exact post-baseline multiplier; 0 = model/off
      std::vector<float> artistic_scale_grid;  // harness-only: render many exact scales per depth
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
        } else if (a == "--native-hdr-scrgb") {
          o.native_hdr_scrgb = true;
        } else if (a == "--hdr-scale") {
          o.hdr_scale = std::stod(next("--hdr-scale"));
          o.sdr_white_level_raw = (int) std::llround(o.hdr_scale * 1000.0);
        } else if (a == "--sdr-white-level-raw") {
          o.sdr_white_level_raw = std::stoi(next("--sdr-white-level-raw"));
          o.hdr_scale = o.sdr_white_level_raw / 1000.0;
        } else if (a == "--max-width") {
          o.max_width = std::stoi(next("--max-width"));
        } else if (a == "--limit") {
          o.limit = std::stoi(next("--limit"));
        } else if (a == "--output-every") {
          o.output_every = std::stoi(next("--output-every"));
        } else if (a == "--output-gt-right-only") {
          o.output_gt_right_only = true;
        } else if (a == "--output-label-frames") {
          o.output_label_frames = true;
        } else if (a == "--runtime-scene-evidence") {
          o.runtime_scene_evidence = true;
        } else if (a == "--depth-every") {
          o.depth_every = std::stoi(next("--depth-every"));
        } else if (a == "--depth-only") {
          o.depth_only = true;
        } else if (a == "--artistic-policy") {
          o.artistic_policy = true;
        } else if (a == "--no-artistic-policy") {
          o.artistic_policy = false;
        } else if (a == "--artistic-scale-override") {
          o.artistic_scale_override = std::stod(next("--artistic-scale-override"));
        } else if (a == "--artistic-scale-grid") {
          const std::string value = next("--artistic-scale-grid");
          size_t begin = 0;
          while (begin <= value.size()) {
            const size_t end = value.find(',', begin);
            const std::string token = value.substr(
              begin, end == std::string::npos ? std::string::npos : end - begin
            );
            if (token.empty()) {
              BOOST_LOG(error) << "sbs-bench: --artistic-scale-grid contains an empty scale";
              return false;
            }
            try {
              o.artistic_scale_grid.push_back(std::stof(token));
            } catch (const std::exception &) {
              BOOST_LOG(error) << "sbs-bench: invalid artistic scale " << token;
              return false;
            }
            if (end == std::string::npos) {
              break;
            }
            begin = end + 1;
          }
        } else if (a == "--depth-override-root") {
          o.depth_override_root = next("--depth-override-root");
        } else if (a == "--depth-override-all") {
          o.depth_override_all = true;
        } else if (a == "--depth-state-export-root") {
          o.depth_state_export_root = next("--depth-state-export-root");
        } else if (a == "--depth-state-replay-root") {
          o.depth_state_replay_root = next("--depth-state-replay-root");
        } else if (a == "--depth-state-cache-key") {
          o.depth_state_cache_key = next("--depth-state-cache-key");
        } else if (a == "--depth-state-manifest-sha256") {
          o.depth_state_manifest_sha256 = next("--depth-state-manifest-sha256");
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
      if (o.simulate_hdr && o.native_hdr_scrgb) {
        BOOST_LOG(error) << "sbs-bench: --simulate-hdr and --native-hdr-scrgb are mutually "
                            "exclusive";
        return false;
      }
      if (!(o.hdr_scale > 0.0 && o.hdr_scale <= 64.0)) {
        BOOST_LOG(error) << "sbs-bench: --hdr-scale must be greater than 0 and at most 64";
        return false;
      }
      if (o.sdr_white_level_raw < 1 || o.sdr_white_level_raw > 64000) {
        BOOST_LOG(error) << "sbs-bench: --sdr-white-level-raw must be between 1 and 64000";
        return false;
      }
      if (o.depth_every < 1 || o.depth_every > 8) {
        BOOST_LOG(error) << "sbs-bench: --depth-every must be between 1 and 8";
        return false;
      }
      if (o.output_every < 1) {
        BOOST_LOG(error) << "sbs-bench: --output-every must be at least 1";
        return false;
      }
      if (o.ema_edge_change > 1.0 || o.ema_edge_gradient > 1.0 || o.ema_edge_strength > 1.0) {
        BOOST_LOG(error) << "sbs-bench: EMA edge thresholds and strength must be <=1";
        return false;
      }
      if (o.artistic_scale_override != 0.0 &&
          !(o.artistic_scale_override >= 0.5 && o.artistic_scale_override <= 1.5)) {
        BOOST_LOG(error) << "sbs-bench: --artistic-scale-override must be between 0.5 and 1.5";
        return false;
      }
      if (!o.artistic_scale_grid.empty()) {
        if (o.artistic_scale_override != 0.0 || o.artistic_policy || o.literal_bestv2 ||
            o.depth_only || o.depth_every != 1 || o.output_every != 1 ||
            o.output_gt_right_only || !o.depth_override_root.empty()) {
          BOOST_LOG(error) << "sbs-bench: --artistic-scale-grid requires full-cadence depth and "
                              "temporal processing with --no-artistic-policy and no depth "
                              "overrides; --output-label-frames may select emitted artifacts";
          return false;
        }
        if (!o.runtime_scene_evidence) {
          BOOST_LOG(error) << "sbs-bench: --artistic-scale-grid requires "
                              "--runtime-scene-evidence so the exact completed SubjectState "
                              "snapshot is authenticated and reused";
          return false;
        }
        if (o.artistic_scale_grid.size() > 64) {
          BOOST_LOG(error) << "sbs-bench: --artistic-scale-grid supports at most 64 scales";
          return false;
        }
        std::set<std::uint32_t> scale_bits;
        std::set<int> scale_slugs;
        float previous = 0.0f;
        for (size_t index = 0; index < o.artistic_scale_grid.size(); ++index) {
          const float scale = o.artistic_scale_grid[index];
          const int slug = (int) std::lround(scale * 100.0f);
          if (!std::isfinite(scale) || scale < 0.5f || scale > 1.5f ||
              std::abs(scale * 100.0f - (float) slug) > 1e-4f ||
              (index > 0 && !(scale > previous)) ||
              !scale_bits.insert(std::bit_cast<std::uint32_t>(scale)).second ||
              !scale_slugs.insert(slug).second) {
            BOOST_LOG(error) << "sbs-bench: --artistic-scale-grid must be strictly increasing, "
                                "unique, finite 0.01 lattice values in [0.5, 1.5]";
            return false;
          }
          previous = scale;
        }
      }
      if (o.literal_bestv2 && o.artistic_scale_override > 0.0) {
        BOOST_LOG(error) << "sbs-bench: --literal-bestv2 cannot be combined with "
                            "--artistic-scale-override; literal mode intentionally bypasses "
                            "production artistic scaling.";
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
      const bool depth_state_export = !o.depth_state_export_root.empty();
      const bool depth_state_replay = !o.depth_state_replay_root.empty();
      const auto valid_sha256 = [](const std::string &value) {
        return value.size() == 64u &&
          std::all_of(
            value.begin(), value.end(), [](const unsigned char character) {
              return std::isdigit(character) ||
                     (character >= static_cast<unsigned char>('a') &&
                      character <= static_cast<unsigned char>('f'));
            }
          );
      };
      if (depth_state_export && depth_state_replay) {
        BOOST_LOG(error) << "sbs-bench: depth-state export and replay are mutually exclusive";
        return false;
      }
      if (depth_state_export || depth_state_replay) {
        if (!valid_sha256(o.depth_state_cache_key) ||
            o.artistic_scale_grid.empty() || o.artistic_policy ||
            o.depth_every != 1 || o.output_every != 1 || !o.runtime_scene_evidence ||
            !o.output_label_frames || o.depth_only || !o.depth_override_root.empty()) {
          BOOST_LOG(error) << "sbs-bench: depth-state caching requires an authenticated key, "
                              "a multiscale sparse-label run, full-cadence depth/runtime scene "
                              "evidence, and no policy/depth overrides";
          return false;
        }
      } else if (!o.depth_state_cache_key.empty()) {
        BOOST_LOG(error) << "sbs-bench: --depth-state-cache-key requires export or replay";
        return false;
      }
      if (depth_state_replay && !valid_sha256(o.depth_state_manifest_sha256)) {
        BOOST_LOG(error) << "sbs-bench: depth-state replay requires the authenticated "
                            "outer-CAS inner-manifest receipt";
        return false;
      }
      if (!depth_state_replay && !o.depth_state_manifest_sha256.empty()) {
        BOOST_LOG(error) << "sbs-bench: --depth-state-manifest-sha256 requires replay";
        return false;
      }
      if (depth_state_replay && !fs::is_directory(o.depth_state_replay_root)) {
        BOOST_LOG(error) << "sbs-bench: depth-state replay root is not a directory";
        return false;
      }
      if (o.output_label_frames && o.output_gt_right_only) {
        BOOST_LOG(error) << "sbs-bench: --output-label-frames and --output-gt-right-only "
                            "are mutually exclusive";
        return false;
      }
      if (o.output_label_frames && o.output_every != 1) {
        BOOST_LOG(error) << "sbs-bench: --output-label-frames requires --output-every 1";
        return false;
      }
      if (!o.depth_override_root.empty() &&
          (o.output_label_frames || o.output_gt_right_only || o.depth_only)) {
        BOOST_LOG(error) << "sbs-bench: --depth-override-root cannot be combined with "
                            "--output-label-frames, --output-gt-right-only, or --depth-only";
        return false;
      }
      return true;
    }

    std::string artistic_scale_slug(float scale) {
      char value[16];
      snprintf(value, sizeof(value), "s%03d", (int) std::lround(scale * 100.0f));
      return value;
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
        // Evaluator experiments intentionally use local, unregistered ONNX stems. Returning the
        // active profile here would silently evaluate the wrong model while contract.json claimed
        // the requested name. Keep custom harness models local-only; missing artifacts fail in
        // ensure_model_available() instead of falling back.
        return config::depth_model_info {want, ""};
      }
      return video::active_depth_model();
    }

  }  // namespace

  int run(int argc, char **argv) {
    opts o;
    if (!parse_opts(argc, argv, o)) {
      return 2;
    }
    const bool hdr_input = o.simulate_hdr || o.native_hdr_scrgb;
    const char *hdr_source_kind = o.native_hdr_scrgb ? "native-pq-in-windows-hdr" :
                                  o.simulate_hdr ? "sdr-in-windows-hdr" : "native-sdr";
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
    std::sort(frames.begin(), frames.end());
    if (o.limit > 0 && (int) frames.size() > o.limit) {
      frames.resize(o.limit);
    }
    if (frames.empty()) {
      BOOST_LOG(error) << "sbs-bench: no png/jpg frames in " << o.frames;
      return 4;
    }

    std::vector<std::uint64_t> source_frame_ids;
    std::vector<std::string> source_frame_suffixes;
    std::set<std::uint64_t> source_frame_id_set;
    source_frame_ids.reserve(frames.size());
    source_frame_suffixes.reserve(frames.size());
    for (size_t index = 0; index < frames.size(); ++index) {
      std::uint64_t id = 0;
      if (!numeric_frame_id(frames[index], id)) {
        BOOST_LOG(error) << "sbs-bench: source frame has an invalid numeric identity: "
                         << frames[index];
        return 4;
      }
      if (!source_frame_id_set.insert(id).second) {
        BOOST_LOG(error) << "sbs-bench: duplicate numeric source-frame identity " << id;
        return 4;
      }
      if (!source_frame_ids.empty() && id <= source_frame_ids.back()) {
        BOOST_LOG(error) << "sbs-bench: source-frame identities are not in increasing order; "
                            "use consistently zero-padded frame names";
        return 4;
      }
      source_frame_ids.push_back(id);
      // numeric_frame_id() has already authenticated the exact "frame_<digits>" shape.
      // Retain those source digits (including zero padding) as the artifact identity.
      source_frame_suffixes.push_back(frames[index].stem().string().substr(6));
    }

    std::string output_selection_mode = "interval";
    std::string output_label_frames_sha256;
    std::vector<std::uint64_t> label_frame_ids;
    std::vector<std::uint64_t> output_selected_frame_ids;
    if (o.output_label_frames) {
      output_selection_mode = "label-frames";
      const fs::path manifest_path = fs::path(o.frames) / "label_frames.json";
      if (!load_label_frame_ids(
            manifest_path,
            label_frame_ids,
            output_label_frames_sha256
          )) {
        return 4;
      }
      for (std::uint64_t id : label_frame_ids) {
        const auto source = std::lower_bound(source_frame_ids.begin(), source_frame_ids.end(), id);
        if (source == source_frame_ids.end() || *source != id) {
          BOOST_LOG(error) << "sbs-bench: label_frames.json references missing source frame "
                           << id;
          return 4;
        }
      }
      output_selected_frame_ids = label_frame_ids;
    } else {
      std::set<std::uint64_t> gt_right_frame_ids;
      if (o.output_gt_right_only) {
        output_selection_mode = "gt-right";
        const fs::path gt_root = fs::path(o.frames) / "gt_right";
        std::error_code gt_error;
        for (const auto &entry : fs::directory_iterator(gt_root, gt_error)) {
          if (!entry.is_regular_file()) {
            continue;
          }
          auto extension = entry.path().extension().string();
          std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
          });
          if (extension != ".png" && extension != ".jpg" && extension != ".jpeg") {
            continue;
          }
          std::uint64_t id = 0;
          if (!numeric_frame_id(entry.path(), id)) {
            BOOST_LOG(error) << "sbs-bench: invalid gt_right frame identity " << entry.path();
            return 4;
          }
          if (!gt_right_frame_ids.insert(id).second) {
            BOOST_LOG(error) << "sbs-bench: invalid or duplicate gt_right frame identity in "
                             << gt_root;
            return 4;
          }
        }
        if (gt_error) {
          BOOST_LOG(error) << "sbs-bench: cannot enumerate gt_right identities in " << gt_root;
          return 4;
        }
      }
      for (size_t index = 0; index < frames.size();
           index += static_cast<size_t>(o.output_every)) {
        const bool selected = !o.output_gt_right_only ||
                              gt_right_frame_ids.find(source_frame_ids[index]) !=
                                gt_right_frame_ids.end();
        if (selected) {
          output_selected_frame_ids.push_back(source_frame_ids[index]);
        }
      }
    }
    if (output_selected_frame_ids.empty()) {
      BOOST_LOG(error) << "sbs-bench: output selection contains no source frames";
      return 4;
    }
    const std::set<std::uint64_t> output_selected_frame_id_set(
      output_selected_frame_ids.begin(),
      output_selected_frame_ids.end()
    );
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
    HRESULT query_hr = dev->CreateQuery(&qd, &warp_disjoint);
    qd.Query = D3D11_QUERY_TIMESTAMP;
    query_hr = FAILED(query_hr) ? query_hr : dev->CreateQuery(&qd, &warp_start);
    query_hr = FAILED(query_hr) ? query_hr : dev->CreateQuery(&qd, &warp_end);
    if (FAILED(query_hr) || !warp_disjoint || !warp_start || !warp_end) {
      BOOST_LOG(error) << "sbs-bench: failed to create GPU timestamp queries";
      return 6;
    }

    auto vs_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_reprojection_vs.hlsl", "main_vs", "vs_5_0");
    auto ps_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_reprojection_ps.hlsl", "main_ps", "ps_5_0");
    auto mask_ps_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_reprojection_ps.hlsl", "mask_ps", "ps_5_0");
    auto coverage_cs_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_forward_coverage_cs.hlsl", "main", "cs_5_0");
    auto parallax_cs_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_forward_coverage_cs.hlsl", "parallax_main", "cs_5_0");
    auto raw_parallax_cs_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_forward_coverage_cs.hlsl", "raw_parallax_main", "cs_5_0");
    auto warp_prefilter_cs_blob = compile(SUNSHINE_SHADERS_DIR "/depth_warp_prefilter_cs.hlsl", "main", "cs_5_0");
    if (!vs_blob || !ps_blob || !mask_ps_blob || !coverage_cs_blob || !parallax_cs_blob || !raw_parallax_cs_blob || !warp_prefilter_cs_blob) {
      return 6;
    }
    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> ps, mask_ps;
    ComPtr<ID3D11ComputeShader> coverage_cs, parallax_cs, raw_parallax_cs, warp_prefilter_cs;
    const HRESULT vs_hr = dev->CreateVertexShader(
      vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr, &vs
    );
    const HRESULT ps_hr = dev->CreatePixelShader(
      ps_blob->GetBufferPointer(), ps_blob->GetBufferSize(), nullptr, &ps
    );
    const HRESULT mask_ps_hr = dev->CreatePixelShader(
      mask_ps_blob->GetBufferPointer(), mask_ps_blob->GetBufferSize(), nullptr, &mask_ps
    );
    const HRESULT coverage_hr = dev->CreateComputeShader(
      coverage_cs_blob->GetBufferPointer(), coverage_cs_blob->GetBufferSize(), nullptr, &coverage_cs
    );
    const HRESULT parallax_hr = dev->CreateComputeShader(
      parallax_cs_blob->GetBufferPointer(), parallax_cs_blob->GetBufferSize(), nullptr, &parallax_cs
    );
    const HRESULT raw_parallax_hr = dev->CreateComputeShader(
      raw_parallax_cs_blob->GetBufferPointer(), raw_parallax_cs_blob->GetBufferSize(), nullptr, &raw_parallax_cs
    );
    const HRESULT prefilter_hr = dev->CreateComputeShader(
      warp_prefilter_cs_blob->GetBufferPointer(), warp_prefilter_cs_blob->GetBufferSize(), nullptr, &warp_prefilter_cs
    );
    if (FAILED(vs_hr) || FAILED(ps_hr) || FAILED(mask_ps_hr) || FAILED(coverage_hr) ||
        FAILED(parallax_hr) || FAILED(raw_parallax_hr) || FAILED(prefilter_hr) ||
        !vs || !ps || !mask_ps || !coverage_cs || !parallax_cs || !raw_parallax_cs ||
        !warp_prefilter_cs) {
      BOOST_LOG(error) << "sbs-bench: failed to create one or more evaluation shaders";
      return 6;
    }

    ComPtr<ID3D11SamplerState> sampler;
    {
      D3D11_SAMPLER_DESC sd = {};
      sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
      sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
      sd.ComparisonFunc = D3D11_COMPARISON_NEVER;
      sd.MaxLOD = D3D11_FLOAT32_MAX;
      if (FAILED(dev->CreateSamplerState(&sd, &sampler)) || !sampler) {
        BOOST_LOG(error) << "sbs-bench: failed to create warp sampler";
        return 6;
      }
    }

    // Built after the first source frame reveals the source/output aspect relationship.
    ComPtr<ID3D11Buffer> repro_cb;

    // ---- estimator / authenticated geometry-independent depth-state replay ----
    const bool replay_depth_state = !o.depth_state_replay_root.empty();
    std::unique_ptr<models::video_depth_estimator> estimator;
    if (!replay_depth_state) {
      estimator = std::make_unique<models::video_depth_estimator>(
        dev, ctx, fs::path(SUNSHINE_ASSETS_DIR), sbs_cfg, model,
        o.artistic_policy,
        o.artistic_scale_grid.empty() ? (float) o.artistic_scale_override : 1.0f,
        o.artistic_policy ?
          models::artistic_policy_authorization::candidate_evaluation :
          models::artistic_policy_authorization::deployment
      );
    }
    std::unique_ptr<depth_state::sequence_writer> depth_state_writer;
    std::unique_ptr<depth_state::sequence_reader> depth_state_reader;
    if (!o.depth_state_export_root.empty()) {
      depth_state_writer = std::make_unique<depth_state::sequence_writer>(
        dev, ctx, fs::path(o.depth_state_export_root), o.depth_state_cache_key,
        source_frame_ids, output_selected_frame_ids
      );
      if (!depth_state_writer->valid()) {
        BOOST_LOG(error) << "sbs-bench: depth-state exporter initialization failed: "
                         << depth_state_writer->error();
        return 6;
      }
    } else if (replay_depth_state) {
      depth_state_reader = std::make_unique<depth_state::sequence_reader>(
        dev, fs::path(o.depth_state_replay_root), o.depth_state_cache_key,
        o.depth_state_manifest_sha256, source_frame_ids, output_selected_frame_ids
      );
      if (!depth_state_reader->valid()) {
        BOOST_LOG(error) << "sbs-bench: depth-state replay validation failed: "
                         << depth_state_reader->error();
        return 7;
      }
    }

    const bool multiscale = !o.artistic_scale_grid.empty();
    const fs::path batch_common_dir = fs::path(o.out) / "common";
    if (multiscale) {
      std::error_code directory_error;
      fs::create_directories(batch_common_dir, directory_error);
      if (directory_error) {
        BOOST_LOG(error) << "sbs-bench: cannot create multiscale common directory: "
                         << directory_error.message();
        return 6;
      }
      for (const float scale : o.artistic_scale_grid) {
        fs::create_directories(
          fs::path(o.out) / "scales" / artistic_scale_slug(scale), directory_error
        );
        if (directory_error) {
          BOOST_LOG(error) << "sbs-bench: cannot create multiscale output directory: "
                           << directory_error.message();
          return 6;
        }
      }
    }
    std::unique_ptr<bounded_multiscale_artifact_writer> multiscale_artifact_writer;
    artifact_writer_result multiscale_artifact_writer_result;
    if (multiscale) {
      multiscale_artifact_writer =
        std::make_unique<bounded_multiscale_artifact_writer>(
          multiscale_artifact_worker_count,
          multiscale_artifact_queue_capacity
        );
    }

    // Per-run state built lazily on the first frame (once we know the input size).
    ComPtr<ID3D11Texture2D> sbs_tex, sbs_stage;
    ComPtr<ID3D11RenderTargetView> sbs_rtv;
    ComPtr<ID3D11ShaderResourceView> sbs_srv;
    ComPtr<ID3D11Texture2D> warp_mask_tex, warp_mask_stage;
    ComPtr<ID3D11RenderTargetView> warp_mask_rtv;
    ComPtr<ID3D11Texture2D> coverage_tex;
    ComPtr<ID3D11UnorderedAccessView> coverage_uav;
    ComPtr<ID3D11ShaderResourceView> coverage_srv;
    D3D11_VIEWPORT vp = {};
    UINT sbs_w = 0, sbs_h = 0;
    UINT source_w_contract = 0, source_h_contract = 0;
    UINT eye_w_contract = 0, eye_h_contract = 0;
    float content_scale_x_contract = 1.0f;
    float content_scale_y_contract = 1.0f;
    UINT disparity_raster_w_contract = 0, disparity_raster_h_contract = 0;
    UINT model_input_w_contract = 0, model_input_h_contract = 0;
    bool artistic_output_geometry_set = false;
    ComPtr<ID3D11Texture2D> depth_stage;  // dump_depth staging cache (depth size is constant)
    ComPtr<ID3D11Texture2D> parallax_tex, parallax_stage;
    ComPtr<ID3D11UnorderedAccessView> parallax_uav;
    ComPtr<ID3D11Texture2D> warp_depth_tex;
    ComPtr<ID3D11UnorderedAccessView> warp_depth_uav;
    ComPtr<ID3D11ShaderResourceView> warp_depth_srv;
    ComPtr<ID3D11Texture2D> ema_mask_stage;
    ComPtr<ID3D11Buffer> raw_depth_stage;
    ComPtr<ID3D11Buffer> multiscale_subject_buf;
    ComPtr<ID3D11ShaderResourceView> multiscale_subject_srv;
    bool raw_shape_written = false;
    float hdr_output_min = std::numeric_limits<float>::infinity();
    float hdr_output_max = -std::numeric_limits<float>::infinity();
    uint64_t hdr_nonfinite = 0;
    float hdr_input_min = std::numeric_limits<float>::infinity();
    float hdr_input_max = -std::numeric_limits<float>::infinity();
    uint64_t hdr_input_nonfinite = 0;

    int written = 0;
    models::estimate_result est;
    bool cuda_graph_captured = false;
    bool have_depth_result = false;
    std::array<float, 12> completed_subject_state {};
    bool completed_subject_state_valid = false;
    nlohmann::json runtime_scene_frames = nlohmann::json::array();
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
      if (source_w_contract == 0) {
        source_w_contract = img.w;
        source_h_contract = img.h;
      } else if (source_w_contract != img.w || source_h_contract != img.h) {
        BOOST_LOG(error) << "sbs-bench: mixed source resolutions are unsupported in one clip: "
                         << source_w_contract << 'x' << source_h_contract << " then "
                         << img.w << 'x' << img.h;
        return 7;
      }
      const std::string &output_id = source_frame_suffixes[fi];

      // Input texture + SRV.
      D3D11_TEXTURE2D_DESC id = {};
      id.Width = img.w;
      id.Height = img.h;
      id.MipLevels = 1;
      id.ArraySize = 1;
      id.Format = hdr_input ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_B8G8R8A8_UNORM;
      id.SampleDesc.Count = 1;
      id.Usage = D3D11_USAGE_IMMUTABLE;
      id.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      std::vector<uint16_t> hdr_rgba;
      const void *input_pixels = img.bgra.data();
      UINT input_pitch = img.w * 4;
      if (o.native_hdr_scrgb) {
        const fs::path sidecar = fs::path(o.frames) / "model_source" /
                                 ("frame_" + output_id + ".scrgb16");
        if (!load_scrgb16(sidecar, img.w, img.h, hdr_rgba)) {
          return 7;
        }
        input_pixels = hdr_rgba.data();
        input_pitch = img.w * 8;
      } else if (o.simulate_hdr) {
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
      if (hdr_input) {
        for (size_t p = 0; p < (size_t) img.w * img.h; ++p) {
          for (size_t channel = 0; channel < 3; ++channel) {
            const float value = half_to_float(hdr_rgba[p * 4 + channel]);
            if (std::isfinite(value)) {
              hdr_input_min = std::min(hdr_input_min, value);
              hdr_input_max = std::max(hdr_input_max, value);
            } else {
              ++hdr_input_nonfinite;
            }
          }
        }
      }
      D3D11_SUBRESOURCE_DATA isd = {input_pixels, input_pitch, 0};
      ComPtr<ID3D11Texture2D> in_tex;
      if (FAILED(dev->CreateTexture2D(&id, &isd, &in_tex))) {
        BOOST_LOG(error) << "sbs-bench: input tex fail";
        continue;
      }
      ComPtr<ID3D11ShaderResourceView> in_srv;
      if (FAILED(dev->CreateShaderResourceView(in_tex.Get(), nullptr, &in_srv)) || !in_srv) {
        BOOST_LOG(error) << "sbs-bench: input SRV creation failed";
        return 6;
      }

      // First frame: size the SBS target. Per eye = the input resolution by default (so the clip
      // size, not a fixed constant, drives eval cost); --eye-h pins a specific output height.
      // The width is still capped at max_encode_width like the live path.
      if (!o.depth_only && !sbs_tex) {
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
        eye_w_contract = (UINT) eye_w;
        eye_h_contract = (UINT) eye_h;
        const float eye_aspect = (float) eye_w / (float) eye_h;
        const float content_scale_x = eye_aspect > aspect ? aspect / eye_aspect : 1.0f;
        const float content_scale_y = eye_aspect < aspect ? eye_aspect / aspect : 1.0f;
        content_scale_x_contract = content_scale_x;
        content_scale_y_contract = content_scale_y;
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
        if (!repro_cb) {
          BOOST_LOG(error) << "sbs-bench: reprojection constant-buffer creation failed";
          return 6;
        }
        D3D11_TEXTURE2D_DESC td = {};
        td.Width = sbs_w;
        td.Height = sbs_h;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = hdr_input ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(dev->CreateTexture2D(&td, nullptr, &sbs_tex)) || !sbs_tex ||
            FAILED(dev->CreateRenderTargetView(sbs_tex.Get(), nullptr, &sbs_rtv)) || !sbs_rtv ||
            FAILED(dev->CreateShaderResourceView(sbs_tex.Get(), nullptr, &sbs_srv)) || !sbs_srv) {
          BOOST_LOG(error) << "sbs-bench: SBS render-target creation failed";
          return 6;
        }
        D3D11_TEXTURE2D_DESC mask_desc = td;
        mask_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        mask_desc.BindFlags = D3D11_BIND_RENDER_TARGET;
        if (FAILED(dev->CreateTexture2D(&mask_desc, nullptr, &warp_mask_tex)) ||
            FAILED(dev->CreateRenderTargetView(warp_mask_tex.Get(), nullptr, &warp_mask_rtv))) {
          BOOST_LOG(error) << "sbs-bench: warp-mask texture creation failed";
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
        if (FAILED(dev->CreateTexture2D(&sd2, nullptr, &sbs_stage)) || !sbs_stage) {
          BOOST_LOG(error) << "sbs-bench: SBS staging-texture creation failed";
          return 6;
        }
        vp = {0, 0, (float) sbs_w, (float) sbs_h, 0, 1};
        BOOST_LOG(info) << "sbs-bench: input " << img.w << "x" << img.h << " -> SBS "
                        << sbs_w << "x" << sbs_h
                        << (o.native_hdr_scrgb ? " (native linear scRGB FP16 HDR)" :
                            o.simulate_hdr ? " (linear scRGB FP16 HDR simulation)" :
                                             " (sRGB SDR)");
      }

      if (!artistic_output_geometry_set) {
        if (o.depth_only) {
          eye_w_contract = img.w;
          eye_h_contract = img.h;
          content_scale_x_contract = 1.0f;
          content_scale_y_contract = 1.0f;
        }
        if (estimator) {
          estimator->set_artistic_output_geometry(
            eye_w_contract,
            eye_h_contract,
            content_scale_x_contract,
            content_scale_y_contract
          );
        }
        artistic_output_geometry_set = true;
      }

      const bool emit_frame = output_selected_frame_id_set.find(source_frame_ids[fi]) !=
                              output_selected_frame_id_set.end();
      // Submit and consume exactly one inference for this source frame, or restore the exact
      // selected state after an up-front authenticated whole-sequence cache validation.
      const auto input_color = hdr_input ? models::input_color_space::scrgb_hdr :
                                           models::input_color_space::srgb;
      if (replay_depth_state) {
        models::runtime_scene_evidence evidence;
        if (!depth_state_reader->runtime_scene(fi, source_frame_ids[fi], evidence)) {
          BOOST_LOG(error) << "sbs-bench: cached runtime-scene row differs at source ordinal "
                           << fi;
          return 7;
        }
        if (emit_frame &&
            !depth_state_reader->load_selected_state(fi, source_frame_ids[fi], est)) {
          BOOST_LOG(error) << "sbs-bench: cannot restore selected depth state at source ordinal "
                           << fi;
          return 7;
        }
        if (emit_frame) {
          completed_subject_state = evidence.subject_state;
          completed_subject_state_valid = true;
          model_input_w_contract = (UINT) est.raw_width;
          model_input_h_contract = (UINT) est.raw_height;
          have_depth_result = true;
        }
        cuda_graph_captured = depth_state_reader->cuda_graph_captured();
        runtime_scene_frames.push_back({
          {"source_frame_ordinal", evidence.completed_frame_id},
          {"source_frame_id", source_frame_ids[fi]},
          {"runtime_scene_id", evidence.runtime_scene_id},
          {"scene_age", evidence.scene_age},
          {"subject_initialized", evidence.subject_initialized},
          {"hard_cut", evidence.hard_cut},
          {"scene_start", evidence.scene_start},
        });
      } else if (!have_depth_result || (fi % (size_t) o.depth_every) == 0) {
        estimator->estimate_depth(in_srv.Get(), input_color, (std::uint64_t) fi);
        est = estimator->finish_pending_depth_for_evaluation(input_color);
        if (o.runtime_scene_evidence) {
          if (!est.completed_frame_valid || est.completed_frame_id >= source_frame_ids.size()) {
            BOOST_LOG(error) << "sbs-bench: runtime scene evidence has no authenticated "
                                "completed depth-frame identity";
            return 7;
          }
          const auto evidence = estimator->read_runtime_scene_evidence_for_evaluation(
            est.completed_frame_id
          );
          if (!evidence.valid || evidence.completed_frame_id != est.completed_frame_id) {
            BOOST_LOG(error) << "sbs-bench: SubjectState runtime scene readback failed for "
                             << est.completed_frame_id;
            return 7;
          }
          completed_subject_state = evidence.subject_state;
          completed_subject_state_valid = true;
          runtime_scene_frames.push_back({
            {"source_frame_ordinal", evidence.completed_frame_id},
            {"source_frame_id", source_frame_ids[(size_t) evidence.completed_frame_id]},
            {"runtime_scene_id", evidence.runtime_scene_id},
            {"scene_age", evidence.scene_age},
            {"subject_initialized", evidence.subject_initialized},
            {"hard_cut", evidence.hard_cut},
            {"scene_start", evidence.scene_start},
          });
          if (depth_state_writer &&
              (!depth_state_writer->capture_runtime_scene(
                 fi, source_frame_ids[fi], evidence
               ) ||
               (emit_frame && !depth_state_writer->capture_selected_state(
                 fi, source_frame_ids[fi], est
               )))) {
            BOOST_LOG(error) << "sbs-bench: depth-state export failed at source ordinal "
                             << fi << ": " << depth_state_writer->error();
            return 7;
          }
        }
        if (est.raw_width > 0 && est.raw_height > 0) {
          if ((model_input_w_contract && model_input_w_contract != (UINT) est.raw_width) ||
              (model_input_h_contract && model_input_h_contract != (UINT) est.raw_height)) {
            BOOST_LOG(error) << "sbs-bench: model input geometry changed within one clip";
            return 7;
          }
          model_input_w_contract = (UINT) est.raw_width;
          model_input_h_contract = (UINT) est.raw_height;
        }
        cuda_graph_captured = cuda_graph_captured || est.cuda_graph_active;
        have_depth_result = true;
      } else {
        // Match the live stream between depth ticks: color advances while all depth-derived
        // geometry remains the last completed result. The views remain owned by the estimator
        // and are valid until the next inference overwrites their backing resources.
      }

      // Sampling must never sample the depth/EMA/subject pipeline itself. Every source frame
      // above was inferred and consumed; only expensive artifact generation/readback is skipped.
      // Depth-only exports use the same contract so data preparation can retain full-cadence
      // temporal state while materializing only the authenticated selected frame identities.
      // Sparse training/reference selections do not need to execute the stateless warp for a
      // frame whose artifacts will be discarded. Keep ordinary --output-every sampling on the
      // full warp path so its production-performance samples remain comparable.
      const bool sparse_output_selection = o.output_label_frames || o.output_gt_right_only;
      if (!emit_frame && (o.depth_only || sparse_output_selection)) {
        continue;
      }

      if (o.depth_only) {
        if (!repro_cb) {
          float repro_params[8] = {
            (float) sbs_cfg.subject_lock,
            sbs_cfg.subject_stretch ? 1.0f : 0.0f,
            1.0f,
            1.0f,
            (float) sbs_cfg.pop_strength,
            o.literal_bestv2 ? 1.0f : 0.0f,
            sbs_cfg.adaptive_pop ? 1.0f : 0.0f,
            (float) sbs_cfg.adaptive_pop_max
          };
          repro_cb = const_buffer(dev.Get(), repro_params);
        }
        if (!est.depth || !est.subject || !repro_cb) {
          BOOST_LOG(error) << "sbs-bench: depth-only export has no complete warp state";
          return 6;
        }
        // The baseline must use the exact texture fed into the shipping warp, including the
        // completed-depth silhouette prefilter. Exporting est.depth directly would fit labels
        // against a different disparity field from the one Apollo actually renders.
        ComPtr<ID3D11Resource> depth_resource;
        est.depth->GetResource(&depth_resource);
        ComPtr<ID3D11Texture2D> depth_texture;
        if (FAILED(depth_resource.As(&depth_texture))) {
          return 6;
        }
        D3D11_TEXTURE2D_DESC depth_desc {};
        depth_texture->GetDesc(&depth_desc);
        if (!warp_depth_tex) {
          auto warp_desc = depth_desc;
          warp_desc.Usage = D3D11_USAGE_DEFAULT;
          warp_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS |
                                D3D11_BIND_SHADER_RESOURCE;
          warp_desc.CPUAccessFlags = 0;
          warp_desc.MiscFlags = 0;
          if (FAILED(dev->CreateTexture2D(&warp_desc, nullptr, &warp_depth_tex)) ||
              FAILED(dev->CreateUnorderedAccessView(warp_depth_tex.Get(), nullptr,
                                                    &warp_depth_uav)) ||
              FAILED(dev->CreateShaderResourceView(warp_depth_tex.Get(), nullptr,
                                                   &warp_depth_srv))) {
            return 6;
          }
        }
        ctx->CSSetShader(warp_prefilter_cs.Get(), nullptr, 0);
        ctx->CSSetShaderResources(0, 1, est.depth.GetAddressOf());
        ctx->CSSetUnorderedAccessViews(0, 1, warp_depth_uav.GetAddressOf(), nullptr);
        ctx->Dispatch((depth_desc.Width + 15u) / 16u,
                      (depth_desc.Height + 15u) / 16u, 1u);
        ID3D11UnorderedAccessView *null_prefilter_uav = nullptr;
        ID3D11ShaderResourceView *null_prefilter_srv = nullptr;
        ctx->CSSetUnorderedAccessViews(0, 1, &null_prefilter_uav, nullptr);
        ctx->CSSetShaderResources(0, 1, &null_prefilter_srv);
        if (!parallax_tex) {
          D3D11_TEXTURE2D_DESC desc {};
          depth_texture->GetDesc(&desc);
          // Depth-only data preparation has no SBS target. Sample the disparity field at the
          // complete source-content raster, matching the pixel centers a same-resolution eye
          // would feed into the shipping bilinear depth lookup.
          desc.Width = img.w;
          desc.Height = img.h;
          desc.Format = DXGI_FORMAT_R32_FLOAT;
          desc.Usage = D3D11_USAGE_DEFAULT;
          desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
          desc.CPUAccessFlags = 0;
          desc.MiscFlags = 0;
          if (FAILED(dev->CreateTexture2D(&desc, nullptr, &parallax_tex)) ||
              FAILED(dev->CreateUnorderedAccessView(parallax_tex.Get(), nullptr,
                                                    &parallax_uav))) {
            BOOST_LOG(error) << "sbs-bench: baseline-disparity texture creation failed";
            return 6;
          }
          disparity_raster_w_contract = desc.Width;
          disparity_raster_h_contract = desc.Height;
          eye_w_contract = desc.Width;
          eye_h_contract = desc.Height;
        }
        D3D11_TEXTURE2D_DESC parallax_desc {};
        parallax_tex->GetDesc(&parallax_desc);
        ctx->CSSetShader(parallax_cs.Get(), nullptr, 0);
        ctx->CSSetSamplers(0, 1, sampler.GetAddressOf());
        ID3D11ShaderResourceView *parallax_srvs[] = {
          in_srv.Get(), warp_depth_srv.Get(), est.subject.Get()
        };
        ctx->CSSetShaderResources(0, 3, parallax_srvs);
        ID3D11UnorderedAccessView *parallax_uavs[] = {nullptr, parallax_uav.Get()};
        ctx->CSSetUnorderedAccessViews(0, 2, parallax_uavs, nullptr);
        ctx->CSSetConstantBuffers(2, 1, repro_cb.GetAddressOf());
        ctx->Dispatch((parallax_desc.Width + 15u) / 16u,
                      (parallax_desc.Height + 15u) / 16u, 1u);
        ID3D11UnorderedAccessView *null_uavs[] = {nullptr, nullptr};
        ID3D11ShaderResourceView *null_srvs[] = {nullptr, nullptr, nullptr};
        ctx->CSSetUnorderedAccessViews(0, 2, null_uavs, nullptr);
        ctx->CSSetShaderResources(0, 3, null_srvs);
        char dname[64];
        snprintf(dname, sizeof(dname), "depth_%s.png", output_id.c_str());
        char pname[64];
        snprintf(pname, sizeof(pname), "baseline_disparity_%s.f32", output_id.c_str());
        const bool depth_written = dump_depth(
          dev.Get(), ctx.Get(), est.depth.Get(), fs::path(o.out) / dname, depth_stage
        );
        const bool disparity_written = dump_float_texture(
          dev.Get(), ctx.Get(), parallax_tex.Get(), fs::path(o.out) / pname, parallax_stage
        );

        ctx->CSSetShader(raw_parallax_cs.Get(), nullptr, 0);
        ctx->CSSetSamplers(0, 1, sampler.GetAddressOf());
        ctx->CSSetShaderResources(0, 3, parallax_srvs);
        ctx->CSSetUnorderedAccessViews(0, 2, parallax_uavs, nullptr);
        ctx->CSSetConstantBuffers(2, 1, repro_cb.GetAddressOf());
        ctx->Dispatch((parallax_desc.Width + 15u) / 16u,
                      (parallax_desc.Height + 15u) / 16u, 1u);
        ctx->CSSetUnorderedAccessViews(0, 2, null_uavs, nullptr);
        ctx->CSSetShaderResources(0, 3, null_srvs);
        char raw_pname[80];
        snprintf(raw_pname, sizeof(raw_pname),
                 "baseline_unclamped_disparity_%s.f32", output_id.c_str());
        const bool raw_disparity_written = dump_float_texture(
          dev.Get(), ctx.Get(), parallax_tex.Get(),
          fs::path(o.out) / raw_pname, parallax_stage
        );
        if (depth_written && disparity_written && raw_disparity_written) {
          ++written;
        }
        if (((fi + 1) % 20) == 0) {
          BOOST_LOG(info) << "sbs-bench: " << (fi + 1) << "/" << frames.size();
        }
        continue;
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

      // The ordinal scale grid is harness-only.  Clone the completed shipping SubjectState on
      // the GPU and patch only SubjectState[2].w in the clone for each render.  The estimator's
      // permanent buffer is never modified, so DA-V2, EMA, subject tracking, zero-plane state,
      // adaptive-pop state, and cut detection advance exactly once per source frame.
      if (multiscale) {
        if (!est.subject || !completed_subject_state_valid) {
          BOOST_LOG(error) << "sbs-bench: multiscale render has no authenticated completed "
                              "SubjectState snapshot";
          return 6;
        }
        ComPtr<ID3D11Resource> subject_resource;
        est.subject->GetResource(&subject_resource);
        ComPtr<ID3D11Buffer> subject_buffer;
        if (FAILED(subject_resource.As(&subject_buffer)) || !subject_buffer) {
          BOOST_LOG(error) << "sbs-bench: multiscale SubjectState is not a structured buffer";
          return 6;
        }
        if (!multiscale_subject_buf || !multiscale_subject_srv) {
          D3D11_BUFFER_DESC subject_desc {};
          subject_buffer->GetDesc(&subject_desc);
          if (subject_desc.ByteWidth != 3u * 4u * sizeof(float) ||
              subject_desc.StructureByteStride != 4u * sizeof(float) ||
              !(subject_desc.BindFlags & D3D11_BIND_SHADER_RESOURCE)) {
            BOOST_LOG(error) << "sbs-bench: unexpected SubjectState buffer contract";
            return 6;
          }
          subject_desc.Usage = D3D11_USAGE_DEFAULT;
          subject_desc.CPUAccessFlags = 0;
          if (FAILED(dev->CreateBuffer(&subject_desc, nullptr, &multiscale_subject_buf)) ||
              FAILED(dev->CreateShaderResourceView(
                multiscale_subject_buf.Get(), nullptr, &multiscale_subject_srv
              )) || !multiscale_subject_srv) {
            BOOST_LOG(error) << "sbs-bench: cannot create multiscale SubjectState clone";
            return 6;
          }
        }
      }

      // The completed depth and its silhouette prefilter are invariant to the
      // final artistic camera multiplier.  Produce the shipping warp input
      // once per source frame, then reuse it for every exact-scale render.
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
          depth_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS |
                                 D3D11_BIND_SHADER_RESOURCE;
          depth_desc.CPUAccessFlags = 0;
          depth_desc.MiscFlags = 0;
          warp_depth_tex.Reset();
          warp_depth_uav.Reset();
          warp_depth_srv.Reset();
          if (FAILED(dev->CreateTexture2D(&depth_desc, nullptr, &warp_depth_tex)) ||
              FAILED(dev->CreateUnorderedAccessView(
                warp_depth_tex.Get(), nullptr, &warp_depth_uav
              )) ||
              FAILED(dev->CreateShaderResourceView(
                warp_depth_tex.Get(), nullptr, &warp_depth_srv
              ))) {
            BOOST_LOG(error) << "sbs-bench: warp prefilter resource creation failed";
            return 6;
          }
        }
        ctx->CSSetShader(warp_prefilter_cs.Get(), nullptr, 0);
        ctx->CSSetShaderResources(0, 1, est.depth.GetAddressOf());
        ctx->CSSetUnorderedAccessViews(0, 1, warp_depth_uav.GetAddressOf(), nullptr);
        ctx->Dispatch((depth_desc.Width + 15u) / 16u,
                      (depth_desc.Height + 15u) / 16u, 1u);
        ID3D11UnorderedAccessView *null_prefilter_uav = nullptr;
        ID3D11ShaderResourceView *null_prefilter_srv = nullptr;
        ctx->CSSetUnorderedAccessViews(0, 1, &null_prefilter_uav, nullptr);
        ctx->CSSetShaderResources(0, 1, &null_prefilter_srv);
        warp_depth = warp_depth_srv.Get();
      }

      const size_t render_scale_count = multiscale ? o.artistic_scale_grid.size() : 1u;
      for (size_t render_scale_index = 0; render_scale_index < render_scale_count;
           ++render_scale_index) {
        const float render_scale = multiscale ?
          o.artistic_scale_grid[render_scale_index] :
          (float) o.artistic_scale_override;
        const fs::path render_output_dir = multiscale ?
          fs::path(o.out) / "scales" / artistic_scale_slug(render_scale) :
          fs::path(o.out);
        ID3D11ShaderResourceView *render_subject = est.subject.Get();
        if (multiscale) {
          auto render_subject_state = completed_subject_state;
          render_subject_state[11] = render_scale;
          ctx->UpdateSubresource(
            multiscale_subject_buf.Get(), 0, nullptr, render_subject_state.data(), 0, 0
          );
          render_subject = multiscale_subject_srv.Get();
        }

      // Composite (mirrors display_vram::convert()'s SBS block): probe reprojection.
      const auto comp_t0 = std::chrono::steady_clock::now();
      const bool time_warp = warp_disjoint && warp_start && warp_end;
      if (time_warp) {
        ctx->Begin(warp_disjoint.Get());
        ctx->End(warp_start.Get());
      }
      // Export the exact full-binocular disparity consumed by this warp. Image phase correlation
      // remains useful for perceived volume, but repetitive textures can alias to an impossible
      // shift and must never drive the hard comfort gate.
      if (emit_frame && est.depth) {
        ComPtr<ID3D11Resource> depth_resource;
        est.depth->GetResource(&depth_resource);
        ComPtr<ID3D11Texture2D> depth_texture;
        if (FAILED(depth_resource.As(&depth_texture))) {
          return 6;
        }
        D3D11_TEXTURE2D_DESC depth_desc {};
        depth_texture->GetDesc(&depth_desc);
        // Preserve the exact pixel centers visited by the shipping output shader. Compacting to
        // a rounded content rectangle shifts those centers whenever a bar boundary is fractional.
        // The diagnostic shader writes zero in bars; scoring excludes them with the same centered
        // ContentToSourceUV validity rule.
        const UINT disparity_w = sbs_w / 2u;
        const UINT disparity_h = sbs_h;
        bool recreate_parallax = !parallax_tex || !parallax_uav;
        if (!recreate_parallax) {
          D3D11_TEXTURE2D_DESC current_desc {};
          parallax_tex->GetDesc(&current_desc);
          recreate_parallax = current_desc.Width != disparity_w ||
                              current_desc.Height != disparity_h;
        }
        if (recreate_parallax) {
          D3D11_TEXTURE2D_DESC parallax_desc = depth_desc;
          parallax_desc.Width = disparity_w;
          parallax_desc.Height = disparity_h;
          parallax_desc.Format = DXGI_FORMAT_R32_FLOAT;
          parallax_desc.Usage = D3D11_USAGE_DEFAULT;
          parallax_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
          parallax_desc.CPUAccessFlags = 0;
          parallax_desc.MiscFlags = 0;
          parallax_tex.Reset();
          parallax_uav.Reset();
          parallax_stage.Reset();
          if (FAILED(dev->CreateTexture2D(&parallax_desc, nullptr, &parallax_tex)) ||
              FAILED(dev->CreateUnorderedAccessView(parallax_tex.Get(), nullptr,
                                                    &parallax_uav))) {
            BOOST_LOG(error) << "sbs-bench: warp-disparity texture creation failed";
            return 6;
          }
          disparity_raster_w_contract = parallax_desc.Width;
          disparity_raster_h_contract = parallax_desc.Height;
        }
        D3D11_TEXTURE2D_DESC parallax_desc {};
        parallax_tex->GetDesc(&parallax_desc);
        ctx->CSSetShader(parallax_cs.Get(), nullptr, 0);
        ctx->CSSetSamplers(0, 1, sampler.GetAddressOf());
        ID3D11ShaderResourceView *parallax_srvs[] = {
          in_srv.Get(), warp_depth, render_subject
        };
        ctx->CSSetShaderResources(0, 3, parallax_srvs);
        ID3D11UnorderedAccessView *parallax_uavs[] = {nullptr, parallax_uav.Get()};
        ctx->CSSetUnorderedAccessViews(0, 2, parallax_uavs, nullptr);
        ctx->CSSetConstantBuffers(2, 1, repro_cb.GetAddressOf());
        ctx->Dispatch((parallax_desc.Width + 15u) / 16u,
                      (parallax_desc.Height + 15u) / 16u, 1u);
        ID3D11UnorderedAccessView *null_uavs[] = {nullptr, nullptr};
        ID3D11ShaderResourceView *null_srvs[] = {nullptr, nullptr, nullptr};
        ctx->CSSetUnorderedAccessViews(0, 2, null_uavs, nullptr);
        ctx->CSSetShaderResources(0, 3, null_srvs);
        char disparity_name[64];
        snprintf(disparity_name, sizeof(disparity_name),
                 "warp_disparity_%s.f32", output_id.c_str());
        if (!dump_float_texture(dev.Get(), ctx.Get(), parallax_tex.Get(),
                                render_output_dir / disparity_name, parallax_stage)) {
          BOOST_LOG(error) << "sbs-bench: exact warp-disparity readback failed";
          return 6;
        }

        // Also preserve the unclamped, scale-1 baseline. Multiplying an already-clamped identity
        // field is not equivalent to the shipping clamp(raw * scale) contract at high pop. The
        // raw shader explicitly divides out the artistic multiplier, so a multiscale sequence
        // dispatches/readbacks it once and gives every scale an identical hard link (copy fallback).
        char raw_disparity_name[72];
        snprintf(raw_disparity_name, sizeof(raw_disparity_name),
                 "warp_unclamped_disparity_%s.f32", output_id.c_str());
        const fs::path raw_disparity_path = render_output_dir / raw_disparity_name;
        if (!multiscale || render_scale_index == 0) {
          ctx->CSSetShader(raw_parallax_cs.Get(), nullptr, 0);
          ctx->CSSetSamplers(0, 1, sampler.GetAddressOf());
          ctx->CSSetShaderResources(0, 3, parallax_srvs);
          ctx->CSSetUnorderedAccessViews(0, 2, parallax_uavs, nullptr);
          ctx->CSSetConstantBuffers(2, 1, repro_cb.GetAddressOf());
          ctx->Dispatch((parallax_desc.Width + 15u) / 16u,
                        (parallax_desc.Height + 15u) / 16u, 1u);
          ctx->CSSetUnorderedAccessViews(0, 2, null_uavs, nullptr);
          ctx->CSSetShaderResources(0, 3, null_srvs);
          if (!dump_float_texture(dev.Get(), ctx.Get(), parallax_tex.Get(),
                                  raw_disparity_path, parallax_stage)) {
            BOOST_LOG(error) << "sbs-bench: unclamped warp-disparity readback failed";
            return 6;
          }
        } else {
          const fs::path first_raw_path =
            fs::path(o.out) / "scales" / artistic_scale_slug(o.artistic_scale_grid.front()) /
            raw_disparity_name;
          std::error_code link_error;
          fs::create_hard_link(first_raw_path, raw_disparity_path, link_error);
          if (link_error) {
            link_error.clear();
            fs::copy_file(
              first_raw_path, raw_disparity_path,
              fs::copy_options::overwrite_existing, link_error
            );
          }
          if (link_error) {
            BOOST_LOG(error) << "sbs-bench: cannot publish shared unclamped disparity: "
                             << link_error.message();
            return 6;
          }
        }
      }
      auto dispatch_coverage = [&](ID3D11ComputeShader *shader,
                                   ID3D11UnorderedAccessView *coverage_view) {
        const UINT clear_winner[4] = {0, 0, 0, 0};
        ctx->ClearUnorderedAccessViewUint(coverage_view, clear_winner);
        ctx->CSSetShader(shader, nullptr, 0);
        ctx->CSSetSamplers(0, 1, sampler.GetAddressOf());
        ID3D11ShaderResourceView *cs_srvs[] = {in_srv.Get(), warp_depth, render_subject};
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

      ID3D11ShaderResourceView *srvs[] = {in_srv.Get(), warp_depth, render_subject};
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

      if (emit_frame) {
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
          render_subject,
          est.depth ? coverage_srv.Get() : nullptr
        };
        ctx->PSSetShaderResources(0, 4, mask_srvs);
        ctx->PSSetConstantBuffers(2, 1, &cb);
        ctx->Draw(3, 0);
        ctx->OMSetRenderTargets(1, null_rtv, nullptr);
        ctx->PSSetShaderResources(0, 4, null_srv);

        // Readback -> PNG.
        char mask_name[64];
        snprintf(mask_name, sizeof(mask_name), "warp_mask_%s.png", output_id.c_str());
        if (multiscale) {
          artifact_write_job mask_job;
          mask_job.path = render_output_dir / mask_name;
          mask_job.width = sbs_w;
          mask_job.height = sbs_h;
          if (!read_bgra8_texture(
                dev.Get(), ctx.Get(), warp_mask_tex.Get(), warp_mask_stage, mask_job.bgra
              ) || !multiscale_artifact_writer->submit(std::move(mask_job))) {
            BOOST_LOG(error) << "sbs-bench: cannot queue multiscale warp-mask artifact";
            return 6;
          }
        } else {
          dump_bgra8_texture(
            dev.Get(), ctx.Get(), warp_mask_tex.Get(),
            render_output_dir / mask_name, warp_mask_stage
          );
        }
        ctx->CopyResource(sbs_stage.Get(), final_sbs_tex);
        D3D11_MAPPED_SUBRESOURCE m = {};
        if (FAILED(ctx->Map(sbs_stage.Get(), 0, D3D11_MAP_READ, 0, &m))) {
          BOOST_LOG(error) << "sbs-bench: SBS staging readback failed";
          return 6;
        }
        char name[64];
        snprintf(name, sizeof(name), "sbs_%s.png", output_id.c_str());
        if (multiscale) {
          artifact_write_job sbs_job;
          sbs_job.path = render_output_dir / name;
          sbs_job.width = sbs_w;
          sbs_job.height = sbs_h;
          sbs_job.count_sbs = true;
          if (hdr_input) {
            sbs_job.encoding = o.native_hdr_scrgb ?
              artifact_pixel_encoding::native_scrgb16 :
              artifact_pixel_encoding::simulated_scrgb16;
            sbs_job.hdr_input_scale = (float) o.hdr_scale;
            sbs_job.scrgb.resize((size_t) sbs_w * sbs_h * 4u);
            for (UINT y = 0; y < sbs_h; ++y) {
              std::memcpy(
                sbs_job.scrgb.data() + (size_t) y * sbs_w * 4u,
                (const uint8_t *) m.pData + (size_t) y * m.RowPitch,
                (size_t) sbs_w * 4u * sizeof(uint16_t)
              );
            }
          } else {
            sbs_job.bgra.resize((size_t) sbs_w * sbs_h * 4u);
            for (UINT y = 0; y < sbs_h; ++y) {
              std::memcpy(
                sbs_job.bgra.data() + (size_t) y * sbs_w * 4u,
                (const uint8_t *) m.pData + (size_t) y * m.RowPitch,
                (size_t) sbs_w * 4u
              );
            }
          }
          ctx->Unmap(sbs_stage.Get(), 0);
          if (!multiscale_artifact_writer->submit(std::move(sbs_job))) {
            BOOST_LOG(error) << "sbs-bench: cannot queue multiscale SBS artifact";
            return 6;
          }
        } else {
          std::vector<uint8_t> buf((size_t) sbs_w * sbs_h * 4u);
          if (hdr_input) {
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
                uint8_t *preview = &buf[((size_t) y * sbs_w + x) * 4];
                if (o.native_hdr_scrgb) {
                  native_hdr_metric_bgra(r, g, b, preview);
                } else {
                  simulated_hdr_metric_bgra(r, g, b, (float) o.hdr_scale, preview);
                }
              }
            }
          } else {
            for (UINT y = 0; y < sbs_h; y++) {
              memcpy(&buf[(size_t) y * sbs_w * 4], (uint8_t *) m.pData + (size_t) y * m.RowPitch, sbs_w * 4);
            }
          }
          ctx->Unmap(sbs_stage.Get(), 0);
          if (save_png(render_output_dir / name, sbs_w, sbs_h, buf)) {
            written++;
          }
        }
          // These artifacts are invariant to the final camera multiplier.  A multiscale run
          // writes one authenticated common copy instead of 26 redundant copies.
          if (!multiscale || render_scale_index == 0) {
            const fs::path common_output_dir = multiscale ? batch_common_dir : fs::path(o.out);
            char dname[64];
            snprintf(dname, sizeof(dname), "depth_%s.png", output_id.c_str());
            dump_depth(
              dev.Get(), ctx.Get(), est.depth.Get(), common_output_dir / dname, depth_stage
            );
            if (sbs_cfg.ema_edge_change > 0.0) {
              char mname[64];
              snprintf(mname, sizeof(mname), "ema_mask_%s.png", output_id.c_str());
              dump_uint_mask(
                dev.Get(), ctx.Get(), est.ema_motion_mask.Get(), common_output_dir / mname,
                ema_mask_stage
              );
            }
            char rname[64];
            snprintf(rname, sizeof(rname), "raw_%s.f32", output_id.c_str());
            dump_raw_model_depth(
              dev.Get(), ctx.Get(), est.raw_model_depth.Get(), est.raw_width, est.raw_height,
              common_output_dir / rname, raw_depth_stage
            );
          }
          if (!raw_shape_written && est.raw_width > 0 && est.raw_height > 0) {
            const fs::path common_output_dir = multiscale ? batch_common_dir : fs::path(o.out);
            std::ofstream shape(common_output_dir / "raw_shape.json");
            if (shape) {
              shape << "{\n  \"width\": " << est.raw_width << ",\n  \"height\": "
                    << est.raw_height << ",\n  \"dtype\": \"float32-le\",\n"
                                         "  \"stage\": \"raw model output before transform/normalization/EMA/curvature\"\n}\n";
              raw_shape_written = true;
            }
          }
      }
      }
      if (((fi + 1) % 20) == 0) {
        BOOST_LOG(info) << "sbs-bench: " << (fi + 1) << "/" << frames.size();
      }
    }

    if (multiscale_artifact_writer) {
      multiscale_artifact_writer_result = multiscale_artifact_writer->finish();
      const size_t expected_async_jobs =
        output_selected_frame_ids.size() * o.artistic_scale_grid.size() * 2u;
      const size_t expected_async_sbs =
        output_selected_frame_ids.size() * o.artistic_scale_grid.size();
      if (!multiscale_artifact_writer_result.success ||
          multiscale_artifact_writer_result.submitted_jobs != expected_async_jobs ||
          multiscale_artifact_writer_result.completed_jobs != expected_async_jobs ||
          multiscale_artifact_writer_result.written_sbs != expected_async_sbs) {
        BOOST_LOG(error) << "sbs-bench: bounded multiscale artifact writer failed: "
                         << multiscale_artifact_writer_result.error << " (submitted "
                         << multiscale_artifact_writer_result.submitted_jobs << ", completed "
                         << multiscale_artifact_writer_result.completed_jobs << ", SBS "
                         << multiscale_artifact_writer_result.written_sbs << ')';
        return 6;
      }
      written += (int) multiscale_artifact_writer_result.written_sbs;
      hdr_output_min = std::min(
        hdr_output_min,
        multiscale_artifact_writer_result.hdr_min
      );
      hdr_output_max = std::max(
        hdr_output_max,
        multiscale_artifact_writer_result.hdr_max
      );
      hdr_nonfinite += multiscale_artifact_writer_result.hdr_nonfinite;
    }

    if (applied_depth_override_frames != expected_depth_override_frames) {
      BOOST_LOG(error) << "sbs-bench: applied " << applied_depth_override_frames
                       << " of " << expected_depth_override_frames << " expected depth overrides";
      return 7;
    }

    if (depth_state_writer && !depth_state_writer->finish(cuda_graph_captured)) {
      BOOST_LOG(error) << "sbs-bench: cannot publish complete depth-state export: "
                       << depth_state_writer->error();
      return 7;
    }
    const std::string depth_state_cache_mode = depth_state_writer ? "cold-export" :
                                               depth_state_reader ? "authenticated-replay" :
                                                                    "disabled";
    const std::string depth_state_manifest_sha256 = depth_state_writer ?
      depth_state_writer->manifest_sha256() :
      depth_state_reader ? depth_state_reader->manifest_sha256() : "";

    sbs_perf::dump_json(((multiscale ? batch_common_dir : fs::path(o.out)) /
                         "sbs_perf.json").string());
    if (o.runtime_scene_evidence) {
      const size_t expected_completed_frames =
        (frames.size() + (size_t) o.depth_every - 1u) / (size_t) o.depth_every;
      if (runtime_scene_frames.size() != expected_completed_frames) {
        BOOST_LOG(error) << "sbs-bench: runtime scene evidence captured "
                         << runtime_scene_frames.size() << " of " << expected_completed_frames
                         << " expected completed depth frames";
        return 7;
      }
      nlohmann::json completed_source_frame_ids = nlohmann::json::array();
      for (const auto &frame : runtime_scene_frames) {
        completed_source_frame_ids.push_back(frame.at("source_frame_id"));
      }
      const nlohmann::json scene_contract = {
        {"schema", 1},
        {"contract", "apollo-subject-state-runtime-scenes-v1"},
        {"evidence_source", "SubjectState[0].y after completed depth postprocess"},
        {"cut_rule", "prior_scene_age_gte_7_and_current_scene_age_eq_0"},
        {"cadence", "completed-depth-frames-only"},
        {"completion_sequence_contract",
         "exact for this synchronous harness sequence; live busy-drop cadence is not replayed"},
        {"depth_reuse_interval", o.depth_every},
        {"source_frame_ids", source_frame_ids},
        {"completed_source_frame_ids", completed_source_frame_ids},
        {"completed_depth_frame_count", runtime_scene_frames.size()},
        {"frames", runtime_scene_frames},
      };
      std::ofstream scene_output(
        (multiscale ? batch_common_dir : fs::path(o.out)) /
        "runtime_scene_evidence.json"
      );
      if (!scene_output) {
        BOOST_LOG(error) << "sbs-bench: cannot create runtime_scene_evidence.json";
        return 7;
      }
      scene_output << scene_contract.dump(2) << '\n';
      if (!scene_output) {
        BOOST_LOG(error) << "sbs-bench: cannot write runtime_scene_evidence.json";
        return 7;
      }
    }
    {
      // Machine-readable execution contract. Evaluation must not scrape human log prose: custom
      // profile names are case-sensitive and fidelity runs must prove literal Bestv2 was active.
      std::vector<std::pair<fs::path, double>> contract_targets;
      if (multiscale) {
        for (const float scale : o.artistic_scale_grid) {
          contract_targets.emplace_back(
            fs::path(o.out) / "scales" / artistic_scale_slug(scale),
            (double) std::lround(scale * 100.0f) / 100.0
          );
        }
      } else {
        // Preserve the scalar harness's historical double-valued CLI contract.  Only the
        // multiscale grid originates as float32 because it is patched into SubjectState.
        contract_targets.emplace_back(fs::path(o.out), o.artistic_scale_override);
      }
      const auto artistic_policy_status = estimator ?
        estimator->artistic_policy_status() : models::artistic_policy_provenance {};
      // Preserve the HLSL float32 operation order so the advertised raw->clamped identity is
      // exact even at a saturated pixel.
      const float source_aspect = (float) source_w_contract /
                                  (float) std::max(1u, source_h_contract);
      const float artistic_aspect_scale = o.literal_bestv2 ? 1.0f :
        std::clamp((5120.0f / 2160.0f) / source_aspect, 0.5f, 3.0f);
      const float artistic_full_clamp_abs =
        2.0f * content_scale_x_contract * 0.071f * artistic_aspect_scale;
      for (size_t target_index = 0; target_index < contract_targets.size(); ++target_index) {
        const auto &[target_directory, target_scale] = contract_targets[target_index];
        const float target_render_scale = (float) target_scale;
        std::ofstream contract(target_directory / "contract.json");
        if (!contract) {
          BOOST_LOG(error) << "sbs-bench: cannot write contract for " << target_directory;
          return 7;
        }
        contract << std::setprecision(std::numeric_limits<float>::max_digits10)
                 << "{\n"
                 << "  \"schema\": 28,\n"
                 << "  \"model\": " << json_string(model.name) << ",\n"
                 << "  \"profile\": " << json_string(sbs_cfg.profile) << ",\n"
                 << "  \"source_width\": " << source_w_contract << ",\n"
                 << "  \"source_height\": " << source_h_contract << ",\n"
                 << "  \"model_input_width\": " << model_input_w_contract << ",\n"
                 << "  \"model_input_height\": " << model_input_h_contract << ",\n"
                 << "  \"eye_width\": " << eye_w_contract << ",\n"
                 << "  \"eye_height\": " << eye_h_contract << ",\n"
                 << "  \"content_scale_x\": " << content_scale_x_contract << ",\n"
                 << "  \"content_scale_y\": " << content_scale_y_contract << ",\n"
                 << "  \"disparity_raster_width\": " << disparity_raster_w_contract << ",\n"
                 << "  \"disparity_raster_height\": " << disparity_raster_h_contract << ",\n"
                 << "  \"policy_warp_source_sha256\": \""
                 << APOLLO_ARTISTIC_WARP_CONTRACT_SHA256 << "\",\n"
                 << "  \"metric_sha256\": \""
                 << APOLLO_ARTISTIC_METRIC_CONTRACT_SHA256 << "\",\n"
                 << "  \"artistic_full_clamp_abs\": "
                 << artistic_full_clamp_abs << ",\n"
                 << "  \"depth_step\": "
                 << json_string(o.depth_every == 1 ? std::string("current-once") : "reuse-" + std::to_string(o.depth_every))
                 << ",\n"
                 << "  \"depth_reuse_interval\": " << o.depth_every << ",\n"
                 << "  \"depth_compensation\": "
                 << json_string(o.depth_override_root.empty() ? "none" : (o.depth_override_all ? "external-treatment" : "external-reference"))
                 << ",\n"
                 << "  \"depth_override_frames\": " << applied_depth_override_frames << ",\n"
                 << "  \"ema\": " << sbs_cfg.ema << ",\n"
                 << "  \"ema_edge_change\": " << sbs_cfg.ema_edge_change << ",\n"
                 << "  \"ema_edge_gradient\": " << sbs_cfg.ema_edge_gradient << ",\n"
                 << "  \"ema_edge_strength\": " << sbs_cfg.ema_edge_strength << ",\n"
                 << "  \"minmax_ema\": " << sbs_cfg.minmax_ema << ",\n"
                 << "  \"subject_lock\": " << sbs_cfg.subject_lock << ",\n"
                 << "  \"subject_recenter\": " << sbs_cfg.subject_recenter << ",\n"
                 << "  \"subject_stretch\": "
                 << (sbs_cfg.subject_stretch ? "true" : "false") << ",\n"
                 << "  \"depth_short_side\": " << sbs_cfg.depth_short_side << ",\n"
                 << "  \"depth_max_aspect\": " << sbs_cfg.depth_max_aspect << ",\n"
                 << "  \"pop_strength\": " << sbs_cfg.pop_strength << ",\n"
                 << "  \"adaptive_pop\": " << (sbs_cfg.adaptive_pop ? "true" : "false") << ",\n"
                 << "  \"adaptive_pop_max\": " << sbs_cfg.adaptive_pop_max << ",\n"
                 << "  \"zero_plane\": " << json_string(sbs_cfg.zero_plane) << ",\n"
                 << "  \"artistic_style\": " << json_string(sbs_cfg.artistic_style) << ",\n"
                 << "  \"artistic_policy\": " << (o.artistic_policy ? "true" : "false") << ",\n"
                 << "  \"artistic_policy_consumed\": "
                 << (artistic_policy_status.consumed ? "true" : "false") << ",\n"
                 << "  \"artistic_policy_authorization\": "
                 << json_string(artistic_policy_status.authorization) << ",\n"
                 << "  \"model_onnx_sha256\": "
                 << json_string(artistic_policy_status.model_onnx_sha256) << ",\n"
                 << "  \"policy_metadata_sha256\": "
                 << json_string(artistic_policy_status.policy_metadata_sha256) << ",\n"
                 << "  \"deployment_geometry_allowlist_sha256\": "
                 << json_string(artistic_policy_status.deployment_geometry_allowlist_sha256)
                 << ",\n"
                 << "  \"color_mode\": "
                 << json_string(hdr_input ? "hdr-scrgb-fp16" : "sdr-srgb-8bit")
                 << ",\n"
                 << "  \"hdr_source_kind\": " << json_string(hdr_source_kind) << ",\n"
                 << "  \"metric_preview_encoding\": "
                 << json_string(
                      o.native_hdr_scrgb ?
                        "perceptual-srgb-from-native-scrgb-reinhard-v1" :
                        o.simulate_hdr ?
                          "source-relative-srgb-from-scrgb-white-normalized-v1" :
                          "native-srgb-v1"
                    )
                 << ",\n"
                 << "  \"hdr_input_scale\": " << (o.simulate_hdr ? o.hdr_scale : 0.0)
                 << ",\n"
                 << "  \"sdr_white_level_raw\": "
                 << (o.simulate_hdr ? o.sdr_white_level_raw : 0) << ",\n"
                 << "  \"artistic_scale_override\": " << target_scale << ",\n";
        if (multiscale) {
          contract << "  \"multiscale_batch\": true,\n"
                   << "  \"multiscale_batch_contract\": "
                   << json_string("apollo-harness-artistic-multiscale-v5") << ",\n"
                   << "  \"multiscale_scale_index\": " << target_index << ",\n"
                   << "  \"multiscale_scale_float32_bits\": "
                   << std::bit_cast<std::uint32_t>(target_render_scale) << ",\n"
                   << "  \"multiscale_common_artifact_directory\": \"../../common\",\n";
        }
        contract << "  \"output_interval\": " << o.output_every << ",\n"
                 << "  \"output_gt_right_only\": "
                 << (o.output_gt_right_only ? "true" : "false") << ",\n"
                 << "  \"output_selection_mode\": "
                 << json_string(output_selection_mode) << ",\n"
                 << "  \"label_frame_ids\": "
                 << nlohmann::json(label_frame_ids).dump() << ",\n"
                 << "  \"output_selected_frame_ids\": "
                 << nlohmann::json(output_selected_frame_ids).dump() << ",\n"
                 << "  \"output_label_frames_sha256\": "
                 << json_string(output_label_frames_sha256) << ",\n"
                 << "  \"literal_bestv2\": " << (o.literal_bestv2 ? "true" : "false") << ",\n"
                 << "  \"cuda_graph\": " << (sbs_cfg.cuda_graph ? "true" : "false") << ",\n"
                 << "  \"cuda_graph_captured\": " << (cuda_graph_captured ? "true" : "false") << ",\n"
                 << "  \"depth_state_cache_mode\": "
                 << json_string(depth_state_cache_mode) << ",\n"
                 << "  \"depth_state_cache_key_sha256\": "
                 << json_string(o.depth_state_cache_key) << ",\n"
                 << "  \"depth_state_manifest_sha256\": "
                 << json_string(depth_state_manifest_sha256) << ",\n"
                 << "  \"artifact_mode\": "
                 << json_string(o.depth_only ? "depth+baseline-disparity" : "full") << ",\n"
                 << "  \"warp_mask\": {\"red\": \"forward_disocclusion_before_fill\"},\n"
                 << "  \"warp_disparity\": "
                    "\"exact_clamped_full_binocular_normalized_at_output_eye_raster_zero_bars\",\n"
                 << "  \"warp_unclamped_disparity\": "
                    "\"unclamped_full_binocular_normalized_at_artistic_scale_1_output_eye_raster_zero_bars\",\n"
                 << "  \"artistic_disparity_contract\": "
                    "\"clamp(raw_baseline_times_scale_to_plus_or_minus_0.142_times_aspect_scale_times_content_scale_x)\"\n"
                 << "}\n";
        if (!contract) {
          BOOST_LOG(error) << "sbs-bench: incomplete contract for " << target_directory;
          return 7;
        }
      }
    }
    if (multiscale) {
      nlohmann::json scale_rows = nlohmann::json::array();
      for (size_t index = 0; index < o.artistic_scale_grid.size(); ++index) {
        const float scale = o.artistic_scale_grid[index];
        const double canonical_scale = (double) std::lround(scale * 100.0f) / 100.0;
        scale_rows.push_back({
          {"index", index},
          {"scale", canonical_scale},
          {"float32_bits", std::bit_cast<std::uint32_t>(scale)},
          {"directory", std::string("scales/") + artistic_scale_slug(scale)},
        });
      }
      const nlohmann::json batch_contract = {
        {"schema", 5},
        {"contract", "apollo-harness-artistic-multiscale-v5"},
        {"scope", "offline-sbs-bench-only"},
        {"shipping_estimator_calls_per_source_frame", replay_depth_state ? 0 : 1},
        {"depth_state_cache", {
          {"mode", depth_state_cache_mode},
          {"key_sha256", o.depth_state_cache_key},
          {"manifest_sha256", depth_state_manifest_sha256},
          {"boundary", "completed-production-depth-state-before-warp-prefilter"},
          {"selected_state_frame_count", output_selected_frame_ids.size()},
          {"runtime_scene_frame_count", source_frame_ids.size()},
        }},
        {"subject_state_source",
         "one authenticated shipping SubjectState snapshot patched only at float[11]"},
        {"estimator_artistic_scale_override", 1.0f},
        {"common_directory", "common"},
        {"scale_rows", scale_rows},
        {"source_frame_ids", source_frame_ids},
        {"output_selection_mode", output_selection_mode},
        {"label_frame_ids", label_frame_ids},
        {"output_selected_frame_ids", output_selected_frame_ids},
        {"output_label_frames_sha256", output_label_frames_sha256},
        {"source_frame_count", source_frame_ids.size()},
        {"output_frame_count_per_scale", output_selected_frame_ids.size()},
        {"artifact_writer", {
          {"contract", std::string(multiscale_artifact_writer_contract)},
          {"mode", "bounded-async-worker-owned-buffers"},
          {"d3d_readback_thread", "harness-main"},
          {"png_factory_scope", "per-worker-com-mta"},
          {"worker_count", multiscale_artifact_writer_result.worker_count},
          {"queue_capacity", multiscale_artifact_writer_result.queue_capacity},
          {"maximum_inflight_job_bound",
           multiscale_artifact_writer_result.worker_count +
             multiscale_artifact_writer_result.queue_capacity + 1u},
          {"submitted_jobs", multiscale_artifact_writer_result.submitted_jobs},
          {"completed_jobs", multiscale_artifact_writer_result.completed_jobs},
          {"sbs_png_jobs", multiscale_artifact_writer_result.written_sbs},
          {"mask_png_jobs",
           multiscale_artifact_writer_result.completed_jobs -
             multiscale_artifact_writer_result.written_sbs},
          {"deterministic_unique_output_paths", true},
          {"drained_before_publication", true},
        }},
        {"scale_invariant_artifacts", nlohmann::json::array({
          "depth_*.png", "raw_*.f32", "raw_shape.json", "ema_mask_*.png",
          "runtime_scene_evidence.json"
        })},
        {"scale_variant_artifacts", nlohmann::json::array({
          "sbs_*.png", "warp_mask_*.png", "warp_disparity_*.f32",
          "warp_unclamped_disparity_*.f32", "contract.json"
        })},
      };
      std::ofstream batch_output(fs::path(o.out) / "multiscale_contract.json");
      if (!batch_output) {
        BOOST_LOG(error) << "sbs-bench: cannot create multiscale_contract.json";
        return 7;
      }
      batch_output << batch_contract.dump(2) << '\n';
      if (!batch_output) {
        BOOST_LOG(error) << "sbs-bench: cannot write multiscale_contract.json";
        return 7;
      }
    }
    if (hdr_input) {
      std::ofstream stats((multiscale ? batch_common_dir : fs::path(o.out)) /
                          "hdr_output_stats.json");
      if (stats) {
        stats << "{\n  \"format\": \"linear-scRGB-fp16\",\n"
              << "  \"hdr_source_kind\": " << json_string(hdr_source_kind) << ",\n"
              << "  \"input_scale\": " << (o.simulate_hdr ? o.hdr_scale : 0.0) << ",\n"
              << "  \"sdr_white_level_raw\": "
              << (o.simulate_hdr ? o.sdr_white_level_raw : 0) << ",\n"
              << "  \"input_min\": ";
        if (std::isfinite(hdr_input_min)) {
          stats << hdr_input_min;
        } else {
          stats << "null";
        }
        stats << ",\n  \"input_max\": ";
        if (std::isfinite(hdr_input_max)) {
          stats << hdr_input_max;
        } else {
          stats << "null";
        }
        stats << ",\n  \"input_nonfinite_components\": " << hdr_input_nonfinite << ",\n"
              << "  \"output_min\": ";
        if (std::isfinite(hdr_output_min)) {
          stats << hdr_output_min;
        } else {
          stats << "null";
        }
        stats << ",\n  \"output_max\": ";
        if (std::isfinite(hdr_output_max)) {
          stats << hdr_output_max;
        } else {
          stats << "null";
        }
        stats << ",\n"
              << "  \"nonfinite_components\": " << hdr_nonfinite << "\n}\n";
      }
    }
    BOOST_LOG(info) << "sbs-bench: wrote " << written << ' '
                    << (o.depth_only ? "depth frames" : "SBS frames")
                    << " + sbs_perf.json to " << o.out;
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
