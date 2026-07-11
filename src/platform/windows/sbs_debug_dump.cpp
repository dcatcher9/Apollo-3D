/**
 * @file src/platform/windows/sbs_debug_dump.cpp
 * @brief Implementation of the debug SBS frame dumper (see sbs_debug_dump.h).
 */
#include "sbs_debug_dump.h"

// standard includes
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <mutex>
#include <system_error>
#include <vector>

// platform includes
#include <wrl/client.h>

// lib includes
#include <zlib.h>

// local includes
#include "src/config.h"
#include "src/logging.h"

// The debug-dump request flag (set by the 0x3004 control-message handler in stream.cpp). Declared
// in video.h; forward-declared here to avoid pulling the full video pipeline header into a debug TU.
// active_depth_model() (also declared in video.h) is used to attribute each dump to the model that
// produced it, so A/B crops across models are traceable.
namespace video {
  extern std::atomic<bool> sbs_debug_dump_pending;
  config::depth_model_info active_depth_model();
}

namespace platf::sbs_debug {

  using namespace std::literals;

  namespace {

    // IEEE half -> float.
    inline float half_to_float(uint16_t h) {
      uint32_t sign = (uint32_t) (h & 0x8000) << 16;
      int exp = (h & 0x7C00) >> 10;
      uint32_t mant = h & 0x03FF;
      uint32_t f;
      if (exp == 0) {
        if (mant == 0) {
          f = sign;
        } else {
          exp = 127 - 15 + 1;
          while (!(mant & 0x0400)) {
            mant <<= 1;
            exp--;
          }
          mant &= 0x03FF;
          f = sign | ((uint32_t) exp << 23) | (mant << 13);
        }
      } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (mant << 13);
      } else {
        f = sign | ((uint32_t) (exp - 15 + 127) << 23) | (mant << 13);
      }
      float out;
      std::memcpy(&out, &f, sizeof(out));
      return out;
    }

    inline uint8_t encode_srgb(float c) {
      c = c < 0.0f ? 0.0f : (c > 1.0f ? 1.0f : c);
      float s = c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
      return (uint8_t) std::lround(s * 255.0f);
    }

    // Diagnostic scRGB preview. Compress luminance uniformly so hue is retained, then apply a
    // second uniform scale only when a wide-gamut component remains above the PNG gamut.
    inline void tonemap_scrgb(float &r, float &g, float &b) {
      r = std::max(r, 0.0f);
      g = std::max(g, 0.0f);
      b = std::max(b, 0.0f);
      const float luminance = std::max(0.2126f * r + 0.7152f * g + 0.0722f * b, 0.0f);
      const float tone_scale = 1.0f / (1.0f + luminance);
      r *= tone_scale;
      g *= tone_scale;
      b *= tone_scale;
      const float gamut_scale = 1.0f / std::max(1.0f, std::max(r, std::max(g, b)));
      r *= gamut_scale;
      g *= gamut_scale;
      b *= gamut_scale;
    }

    // Write a tightly-packed 8-bit RGB buffer (w*h*3, top-to-bottom) as a PNG (color type 2)
    // using zlib deflate. Minimal encoder: signature + IHDR + one IDAT + IEND.
    void write_png(const std::filesystem::path &path, uint32_t w, uint32_t h,
      const std::vector<uint8_t> &rgb) {
      // PNG scanlines are prefixed with a per-row filter byte (0 = none).
      std::vector<uint8_t> raw((size_t) h * (1 + (size_t) w * 3));
      for (uint32_t y = 0; y < h; y++) {
        raw[(size_t) y * (1 + (size_t) w * 3)] = 0;
        std::memcpy(&raw[(size_t) y * (1 + (size_t) w * 3) + 1], &rgb[(size_t) y * w * 3], (size_t) w * 3);
      }

      uLongf comp_len = compressBound((uLong) raw.size());
      std::vector<uint8_t> comp(comp_len);
      if (compress2(comp.data(), &comp_len, raw.data(), (uLong) raw.size(), Z_BEST_SPEED) != Z_OK) {
        BOOST_LOG(warning) << "SBS debug dump: zlib compress failed for "sv << path.string();
        return;
      }

      std::ofstream f(path, std::ios::binary);
      if (!f) {
        BOOST_LOG(warning) << "SBS debug dump: cannot open "sv << path.string();
        return;
      }

      auto be32 = [](uint32_t v, uint8_t *p) {
        p[0] = (uint8_t) (v >> 24);
        p[1] = (uint8_t) (v >> 16);
        p[2] = (uint8_t) (v >> 8);
        p[3] = (uint8_t) v;
      };
      auto write_chunk = [&](const char *type, const uint8_t *data, uint32_t len) {
        uint8_t buf[4];
        be32(len, buf);
        f.write((const char *) buf, 4);
        f.write(type, 4);
        if (len) {
          f.write((const char *) data, len);
        }
        uLong crc = crc32(0, (const Bytef *) type, 4);
        if (len) {
          crc = crc32(crc, (const Bytef *) data, len);
        }
        be32((uint32_t) crc, buf);
        f.write((const char *) buf, 4);
      };

      static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
      f.write((const char *) sig, 8);

      uint8_t ihdr[13] = {};
      be32(w, ihdr);
      be32(h, ihdr + 4);
      ihdr[8] = 8;  // bit depth
      ihdr[9] = 2;  // color type: truecolor RGB
      // ihdr[10..12] = compression/filter/interlace = 0
      write_chunk("IHDR", ihdr, 13);
      write_chunk("IDAT", comp.data(), (uint32_t) comp_len);
      write_chunk("IEND", nullptr, 0);
    }

    // Jet colormap (t: 0=far/blue -> 1=near/red); vivid enough to reveal subtle depth structure
    // (e.g. a depth model hallucinating parallax on flat desktop/page content).
    inline void colormap_jet(float t, uint8_t &r, uint8_t &g, uint8_t &b) {
      t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
      auto ch = [](float x) {
        x = x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
        return (uint8_t) std::lround(x * 255.0f);
      };
      r = ch(1.5f - std::fabs(4.0f * t - 3.0f));
      g = ch(1.5f - std::fabs(4.0f * t - 2.0f));
      b = ch(1.5f - std::fabs(4.0f * t - 1.0f));
    }

    // Save an SRV's texture to <tag>.png inside dir (grayscale for R32_FLOAT depth, or a jet
    // heatmap when heatmap=true). Linear FP16 SDR receives only the sRGB OETF; HDR scRGB receives
    // a luminance-preserving diagnostic tone map followed by the OETF. B8G8R8A8 is passed through.
    void dump_srv(ID3D11Device *device, ID3D11DeviceContext *ctx, ID3D11ShaderResourceView *srv,
      const std::filesystem::path &dir, const char *tag, bool heatmap = false, bool hdr = false) {
      if (!srv) {
        return;
      }
      Microsoft::WRL::ComPtr<ID3D11Resource> res;
      srv->GetResource(&res);
      Microsoft::WRL::ComPtr<ID3D11Texture2D> tex;
      if (FAILED(res.As(&tex))) {
        return;
      }
      D3D11_TEXTURE2D_DESC desc = {};
      tex->GetDesc(&desc);

      D3D11_TEXTURE2D_DESC sdesc = desc;
      sdesc.Usage = D3D11_USAGE_STAGING;
      sdesc.BindFlags = 0;
      sdesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      sdesc.MiscFlags = 0;
      Microsoft::WRL::ComPtr<ID3D11Texture2D> staging;
      if (FAILED(device->CreateTexture2D(&sdesc, nullptr, &staging))) {
        return;
      }
      ctx->CopyResource(staging.Get(), tex.Get());

      D3D11_MAPPED_SUBRESOURCE mapped = {};
      if (FAILED(ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        return;
      }

      const UINT w = desc.Width, h = desc.Height;
      std::vector<uint8_t> rgb((size_t) w * h * 3);  // tightly packed 8-bit RGB
      const uint8_t *base = (const uint8_t *) mapped.pData;
      for (UINT y = 0; y < h; y++) {
        const uint8_t *in_row = base + (size_t) y * mapped.RowPitch;
        uint8_t *out_row = rgb.data() + (size_t) y * w * 3;
        for (UINT x = 0; x < w; x++) {
          uint8_t r, g, b;
          if (desc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
            const uint16_t *px = (const uint16_t *) (in_row + (size_t) x * 8);
            float rf = half_to_float(px[0]);
            float gf = half_to_float(px[1]);
            float bf = half_to_float(px[2]);
            if (hdr) {
              tonemap_scrgb(rf, gf, bf);
            }
            r = encode_srgb(rf);
            g = encode_srgb(gf);
            b = encode_srgb(bf);
          } else if (desc.Format == DXGI_FORMAT_R32_FLOAT) {
            float v = *(const float *) (in_row + (size_t) x * 4);
            if (heatmap) {
              colormap_jet(v, r, g, b);
            } else {
              v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
              r = g = b = (uint8_t) std::lround(v * 255.0f);
            }
          } else {  // assume 8-bit BGRA (B8G8R8A8_UNORM)
            const uint8_t *px = in_row + (size_t) x * 4;
            b = px[0];
            g = px[1];
            r = px[2];
          }
          out_row[x * 3 + 0] = r;
          out_row[x * 3 + 1] = g;
          out_row[x * 3 + 2] = b;
        }
      }
      ctx->Unmap(staging.Get(), 0);

      char name[64];
      snprintf(name, sizeof(name), "%s.png", tag);
      write_png(dir / name, w, h, rgb);
    }

  }  // namespace

  dumper::dumper() {
    // Prefer the APOLLO_SBS_DUMP override; otherwise fall back to an "sbs_dump" subfolder next to
    // the sunshine log so the feature works with no configuration. The per-dump subfolders (and
    // this root) are created on demand at dump time.
    if (const char *d = std::getenv("APOLLO_SBS_DUMP")) {
      dir_ = d;
    } else if (!config::sunshine.log_file.empty()) {
      dir_ = std::filesystem::path(config::sunshine.log_file).parent_path() / "sbs_dump";
    } else {
      dir_ = "sbs_dump";
    }
    // Encode devices (and with them this dumper) are recreated on every SBS toggle / HDR /
    // resolution change; log the resolved dir once per process, not once per device.
    static std::once_flag log_once;
    std::call_once(log_once, [this]() {
      BOOST_LOG(info) << "SBS debug frame dump dir: "sv << dir_.string();
    });
  }

  void dumper::maybe_dump(ID3D11Device *device, ID3D11DeviceContext *ctx,
    ID3D11ShaderResourceView *source, ID3D11ShaderResourceView *depth,
    ID3D11ShaderResourceView *sbs, bool hdr) {
    if (dir_.empty()) {
      return;
    }
    std::error_code ec;
    auto trigger = dir_ / "dump.trigger";
    bool by_button = ::video::sbs_debug_dump_pending.exchange(false, std::memory_order_relaxed);
    bool by_file = false;
    if (by_button) {
      // Also consume a trigger file if one happens to be present, so it doesn't fire again.
      by_file = std::filesystem::exists(trigger, ec);
    } else {
      // The manual trigger file is a dev fallback; polling the filesystem every frame on the
      // encode thread is wasted syscalls, so only stat it about once a second.
      if ((poll_counter_++ & 63) != 0) {
        return;
      }
      by_file = std::filesystem::exists(trigger, ec);
      if (!by_file) {
        return;
      }
    }

    // One timestamped subfolder per dump so successive dumps never overwrite each other and the
    // three images of a frame stay grouped: <dump dir>/dump_YYYYMMDD_HHMMSS_NN/. The counter
    // suffix disambiguates same-second dumps. The whole tree is created on demand.
    char ts[32] = "unknown";
    std::time_t t = std::time(nullptr);
    if (const std::tm *lt = std::localtime(&t)) {
      std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", lt);
    }
    char sub[64];
    snprintf(sub, sizeof(sub), "dump_%s_%02d", ts, counter_);
    auto out_dir = dir_ / sub;
    std::filesystem::create_directories(out_dir, ec);

    dump_srv(device, ctx, source, out_dir, "source", /*heatmap=*/false, hdr);
    dump_srv(device, ctx, depth, out_dir, "depth");
    dump_srv(device, ctx, depth, out_dir, "depth_heat", /*heatmap=*/true);  // A4: jet colormap
    dump_srv(device, ctx, sbs, out_dir, "sbs", /*heatmap=*/false, hdr);

    // Attribute the dump to the model that produced it (for cross-model A/B).
    auto model_name = ::video::active_depth_model().name;
    if (std::ofstream meta(out_dir / "meta.txt"); meta) {
      meta << "depth_model=" << model_name << "\n";
      meta << "color_mode=" << (hdr ? "linear-scRGB-HDR" : "SDR") << "\n";
    }

    counter_++;
    BOOST_LOG(info) << "SBS debug dump written to "sv << out_dir.string()
                    << " (model "sv << model_name << ')';

    if (by_file) {
      std::filesystem::remove(trigger, ec);
    }
  }

}  // namespace platf::sbs_debug
