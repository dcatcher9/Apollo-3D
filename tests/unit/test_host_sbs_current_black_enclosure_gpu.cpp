/** Private code1 enclosure mechanics and authentic saved-current native parity. No model/warp/GT. */
#include <gtest/gtest.h>
#ifdef _WIN32
  #include "src/crypto.h"
  #include "src/host_sbs_current_black_enclosure_gpu.h"
  #include "src/host_sbs_shader_cache.h"
  #include "subtitle_helper_source_pins.h"

  #include <algorithm>
  #include <array>
  #include <bit>
  #include <cstdint>
  #include <cstdlib>
  #include <cstring>
  #include <dxgi.h>
  #include <filesystem>
  #include <fstream>
  #include <map>
  #include <nlohmann/json.hpp>
  #include <stdexcept>
  #include <string>
  #include <string_view>
  #include <vector>

namespace {
  using Microsoft::WRL::ComPtr;
  namespace enclosure = models::host_sbs_current_black_enclosure_gpu_v1;
  namespace baseline = models::host_sbs_subtitle_inner_colour_gpu_global_validation_v2;
  namespace cache = models::host_sbs_shader_cache;
  using pixel16 = std::array<std::uint16_t, 4u>;
  using pixel8 = std::array<std::uint8_t, 4u>;
  constexpr const char *compiled_enclosure_helper_pin = subtitle_test_source_pins::current_black_enclosure;
  constexpr const char *compiled_baseline_helper_pin = subtitle_test_source_pins::global_colour_validation;

  struct texture_t {
    ComPtr<ID3D11Texture2D> resource;
    ComPtr<ID3D11ShaderResourceView> srv;
  };

  struct buffer_t {
    ComPtr<ID3D11Buffer> resource;
    ComPtr<ID3D11ShaderResourceView> srv;
  };

  std::vector<char> file_bytes(const std::filesystem::path &p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() < 0) {
      throw std::runtime_error("Cannot read native enclosure artifact");
    }
    std::vector<char> b(static_cast<std::size_t>(f.tellg()));
    f.seekg(0);
    if (!b.empty()) {
      f.read(b.data(), static_cast<std::streamsize>(b.size()));
    }
    if (!f) {
      throw std::runtime_error("Incomplete native enclosure artifact");
    }
    return b;
  }

  std::string hash_bytes(const std::vector<char> &b) {
    const auto h = crypto::hash(std::string_view(b.data(), b.size()));
    constexpr char d[] = "0123456789abcdef";
    std::string s;
    for (auto c : h) {
      s.push_back(d[c >> 4u]);
      s.push_back(d[c & 15u]);
    }
    return s;
  }

  std::vector<char> bound_bytes(const nlohmann::json &b, std::size_t expected) {
    auto v = file_bytes(b.at("path").get<std::string>());
    const auto declared = b.contains("bytes") ? b.at("bytes").get<std::size_t>() : b.at("byte_count").get<std::size_t>();
    if (b.contains("bytes") && b.contains("byte_count") && b.at("bytes") != b.at("byte_count")) {
      throw std::runtime_error("Conflicting native byte count");
    }
    if (v.size() != expected || v.size() != declared || hash_bytes(v) != b.at("sha256").get<std::string>()) {
      throw std::runtime_error("Native enclosure binding drift");
    }
    return v;
  }

  template<class T>
  std::vector<char> raw(const T &v) {
    std::vector<char> b(v.size() * sizeof(typename T::value_type));
    if (!b.empty()) {
      std::memcpy(b.data(), v.data(), b.size());
    }
    return b;
  }

  std::vector<UINT> words(const std::vector<char> &v) {
    if (v.size() % 4u) {
      throw std::runtime_error("Nonword diagnostic");
    }
    std::vector<UINT> w(v.size() / 4u);
    if (!v.empty()) {
      std::memcpy(w.data(), v.data(), v.size());
    }
    return w;
  }

  class HostSbsCurrentBlackEnclosureGpuTest: public ::testing::Test {
  protected:
    void require(bool b) {
      if (!b) {
        throw std::runtime_error(error_.empty() ? "Native enclosure D3D operation failed" : error_);
      }
    }

    void SetUp() override {
      const auto *ep = std::getenv("SUNSHINE_TEST_BLACK_ENCLOSURE_SHADER_PIN"), *er = std::getenv("SUNSHINE_TEST_BLACK_ENCLOSURE_SHADER_ROOT");
      const auto *ip = std::getenv("SUNSHINE_TEST_SUBTITLE_INNER_SHADER_PIN"), *ir = std::getenv("SUNSHINE_TEST_SUBTITLE_INNER_SHADER_ROOT");
      const auto *gp = std::getenv("SUNSHINE_TEST_SUBTITLE_INNER_COMPONENT_PIN"), *gr = std::getenv("SUNSHINE_TEST_SUBTITLE_INNER_COMPONENT_ROOT");
      if (!ep || !er || !ip || !ir || !gp || !gr) {
        GTEST_SKIP() << "Explicit frozen private source roots and pins required";
      }
      const auto *driver = std::getenv("SUNSHINE_TEST_SUBTITLE_INNER_DRIVER");
      require(!driver || std::string(driver) == "warp" || std::string(driver) == "hardware");
      hardware_ = driver && std::string(driver) == "hardware";
      D3D_FEATURE_LEVEL got {}, requested[] = {D3D_FEATURE_LEVEL_11_0};
      require(SUCCEEDED(D3D11CreateDevice(nullptr, hardware_ ? D3D_DRIVER_TYPE_HARDWARE : D3D_DRIVER_TYPE_WARP, nullptr, 0u, requested, 1u, D3D11_SDK_VERSION, &device_, &got, &context_)));
      require(got == D3D_FEATURE_LEVEL_11_0);
      ComPtr<IDXGIDevice> xd;
      ComPtr<IDXGIAdapter> adapter;
      require(SUCCEEDED(device_.As(&xd)));
      require(SUCCEEDED(xd->GetAdapter(&adapter)));
      DXGI_ADAPTER_DESC desc {};
      require(SUCCEEDED(adapter->GetDesc(&desc)));
      if (hardware_) {
        require(desc.VendorId == 4318u);
      }
      adapter_ = {{"vendor_id", desc.VendorId}, {"device_id", desc.DeviceId}, {"luid_low", desc.AdapterLuid.LowPart}, {"luid_high", desc.AdapterLuid.HighPart}};
      const std::array cspec {
        cache::shader_spec {"host_sbs_subtitle_inner_colour_cs.hlsl", "palette_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_inner_colour_cs.hlsl", "emit_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_inner_colour_cs.hlsl", "clear_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_inner_colour_cs.hlsl", "accumulate_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_inner_colour_cs.hlsl", "validate_main", "cs_5_0"}
      };
      const std::array gspec {
        cache::shader_spec {"host_sbs_subtitle_seeded_components_cs.hlsl", "init_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_seeded_components_cs.hlsl", "union_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_seeded_components_cs.hlsl", "reduce_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_seeded_components_cs.hlsl", "resolve_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_seeded_components_cs.hlsl", "dilate_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_seeded_components_cs.hlsl", "dilate_box_main", "cs_5_0"}
      };
      const std::array espec {
        cache::shader_spec {"host_sbs_current_black_enclosure_cs.hlsl", "begin_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_black_enclosure_cs.hlsl", "clear_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_black_enclosure_cs.hlsl", "validate_encoding_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_black_enclosure_cs.hlsl", "unit_meta_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_black_enclosure_cs.hlsl", "collect_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_black_enclosure_cs.hlsl", "clear_scratch_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_black_enclosure_cs.hlsl", "prepare_unit_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_black_enclosure_cs.hlsl", "emit_exterior_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_black_enclosure_cs.hlsl", "resolve_exterior_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_black_enclosure_cs.hlsl", "count_enclosed_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_black_enclosure_cs.hlsl", "select_unit_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_black_enclosure_cs.hlsl", "publish_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_seeded_components_roi_cs.hlsl", "init_roi_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_seeded_components_roi_cs.hlsl", "union_roi_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_seeded_components_roi_cs.hlsl", "reduce_roi_main", "cs_5_0"}
      };
      load_shaders(ir, ip, cspec, colour_);
      load_shaders(gr, gp, gspec, graph_);
      const auto snapshot = cache::snapshot_sources(er, espec);
      enclosure_closure_ = cache::source_closure_sha256(snapshot);
      require(enclosure_closure_ == ep);
      for (std::size_t i = 0; i < espec.size(); ++i) {
        const auto code = cache::get(snapshot, espec[i]);
        require(code && !code->empty());
        if (i < 12u) {
          require(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &enclosure_[i])));
        } else {
          require(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &roi_[i - 12u])));
        }
      }
      inner_closure_ = ip;
      graph_closure_ = gp;
      RecordProperty("enclosure_source_closure", ep);
      RecordProperty("helper_source_pin", compiled_enclosure_helper_pin);
      RecordProperty("semantic_status", "UNKNOWN");
      RecordProperty("CBC_output_schema", 5);
      if (const auto *hp = std::getenv("SUNSHINE_TEST_BLACK_ENCLOSURE_HEADER_PIN")) {
        require(std::string(hp) == compiled_enclosure_helper_pin);
      }
      for (std::size_t i = 0; i < 12u; ++i) {
        program_.shaders[i] = enclosure_[i].Get();
      }
      for (std::size_t i = 0; i < 3u; ++i) {
        program_.roi_graph[i] = roi_[i].Get();
      }
      author(true, 1u);
    }

    void TearDown() override {
      if (!context_) {
        return;
      }
      std::array<ID3D11ShaderResourceView *, 16u> s {};
      std::array<ID3D11UnorderedAccessView *, 8u> u {};
      std::array<ID3D11Buffer *, 3u> b {};
      context_->CSSetShaderResources(0u, 16u, s.data());
      context_->CSSetUnorderedAccessViews(0u, 8u, u.data(), nullptr);
      context_->CSSetConstantBuffers(0u, 3u, b.data());
      context_->CSSetShader(nullptr, nullptr, 0u);
    }

    template<std::size_t N>
    void load_shaders(const char *root, const char *pin, const std::array<cache::shader_spec, N> &spec, std::array<ComPtr<ID3D11ComputeShader>, N> &out) {
      const auto snap = cache::snapshot_sources(root, spec);
      require(cache::source_closure_sha256(snap) == pin);
      for (std::size_t i = 0; i < N; ++i) {
        const auto code = cache::get(snap, spec[i]);
        require(code && !code->empty());
        require(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &out[i])));
      }
    }

    texture_t texture(UINT w, UINT h, DXGI_FORMAT format) {
      texture_t t;
      D3D11_TEXTURE2D_DESC d {};
      d.Width = w;
      d.Height = h;
      d.MipLevels = d.ArraySize = 1u;
      d.Format = format;
      d.SampleDesc.Count = 1u;
      d.Usage = D3D11_USAGE_DEFAULT;
      d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      require(SUCCEEDED(device_->CreateTexture2D(&d, nullptr, &t.resource)));
      require(SUCCEEDED(device_->CreateShaderResourceView(t.resource.Get(), nullptr, &t.srv)));
      return t;
    }

    buffer_t buffer(UINT bytes, bool constant = false, bool indirect = false) {
      buffer_t b;
      D3D11_BUFFER_DESC d {};
      d.ByteWidth = bytes;
      d.Usage = D3D11_USAGE_DEFAULT;
      d.BindFlags = constant ? D3D11_BIND_CONSTANT_BUFFER : indirect ? 0u :
                                                                       D3D11_BIND_SHADER_RESOURCE;
      if (indirect) {
        d.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      } else if (!constant) {
        d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        d.StructureByteStride = 4u;
      }
      require(SUCCEEDED(device_->CreateBuffer(&d, nullptr, &b.resource)));
      if (!constant && !indirect) {
        require(SUCCEEDED(device_->CreateShaderResourceView(b.resource.Get(), nullptr, &b.srv)));
      }
      return b;
    }

    static std::uint16_t code_half(std::uint8_t c) {
      if (!c) {
        return 0u;
      }
      const float v = static_cast<float>(c) / 255.0f;
      const UINT b = std::bit_cast<UINT>(v);
      return static_cast<std::uint16_t>((((b >> 23u) & 255u) - 112u) * 1024u + ((b >> 13u) & 1023u));
    }

    void allocate_inputs(UINT w, UINT h, UINT fullh) {
      W_ = w;
      H_ = h;
      fullH_ = fullh;
      top_ = fullh - h;
      require(helper_.ensure(device_.Get(), w, h, error_));
      rgb_ = texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
      support_ = texture(w, h, DXGI_FORMAT_R32_UINT);
      origin_ = texture(w, fullh, DXGI_FORMAT_R8G8B8A8_UNORM);
      probability_ = buffer(614400u);
      ocr_ = buffer(1024u);
      sct_ = buffer(128u);
      uip_ = buffer(160u);
      cb_ = buffer(96u, true);
      root_ = buffer(256u, false, true);
    }

    void author(bool caption, UINT frame, UINT code = 1u, bool open = false, bool partial = false, bool overlap = false, UINT w = 32u, UINT h = 16u, UINT stress = 0u) {
      allocate_inputs(w, h, partial ? h * 6u : h);
      origin_pixels_.assign(static_cast<std::size_t>(w) * fullH_, {32u, 32u, 32u, 255u});
      auto put = [&](UINT x, UINT y, std::uint8_t c) {
        origin_pixels_[(y + top_) * w + x] = {c, c, c, 255u};
      };
      if (stress == 1u) {
        for (UINT y = 0; y < h; ++y) {
          for (UINT x = 0; x < w; ++x) {
            put(x, y, 255u);
          }
        }
      }
      if (stress == 2u) {
        for (UINT y = 0; y < h; ++y) {
          for (UINT x = 0; x < w; ++x) {
            put(x, y, 0u);
          }
        }
        for (UINT y = 3u * h / 16u; y < 14u * h / 16u; y += 2u) {
          for (UINT x = w / 8u; x < 7u * w / 8u; ++x) {
            put(x, y, 2u);
          }
          if (y + 1u < 14u * h / 16u) {
            put(((y / (2u)) & 1u) ? w / 8u : 7u * w / 8u - 1u, y + 1u, 2u);
          }
        }
      }
      if (caption) {
        for (UINT y = h / 4u; y < 13u * h / 16u; ++y) {
          for (UINT x = w / 4u; x < 3u * w / 4u; ++x) {
            if (!open || x < w / 4u + 2u) {
              put(x, y, static_cast<std::uint8_t>(code));
            }
          }
        }
        for (UINT y = 3u * h / 8u; y < 11u * h / 16u; ++y) {
          for (UINT x = 5u * w / 16u; x < 11u * w / 16u; ++x) {
            put(x, y, 255u);
          }
        }
        if (!open && !stress) {
          for (UINT y = 5u * h / 16u; y < 12u * h / 16u; ++y) {
            for (UINT x = 5u * w / 32u; x < w / 4u; ++x) {
              put(x, y, 255u);
            }
          }
        }
      }
      pixels_.resize(static_cast<std::size_t>(w) * h);
      for (UINT y = 0; y < h; ++y) {
        for (UINT x = 0; x < w; ++x) {
          const auto p = origin_pixels_[(y + top_) * w + x];
          pixels_[y * w + x] = {code_half(p[0]), code_half(p[1]), code_half(p[2]), 0x3c00u};
        }
      }
      support_words_.assign(static_cast<std::size_t>(w) * h, 0u);
      probability_words_.assign(153600u, 1.0f);
      cb_words_ = {w, fullH_, top_, h, 16u, 16u, 0u, 16u, 0u, 0u, 16u, 16u, frame, 0u, 7u, 0u, 1000000u + frame, 0u, 0u, 0u, 1u, 27u, 0u, 0u};
      ocr_words_ = {};
      ocr_words_[0] = 4u;
      ocr_words_[1] = 0x3852434fu;
      ocr_words_[2] = 1u;
      ocr_words_[3] = ocr_words_[4] = caption ? (overlap ? 2u : 1u) : 0u;
      std::copy_n(cb_words_.begin() + 12u, 4u, ocr_words_.begin() + 5u);
      ocr_words_[9] = w;
      ocr_words_[10] = fullH_;
      std::copy_n(cb_words_.begin() + 4u, 4u, ocr_words_.begin() + 11u);
      if (caption) {
        for (UINT u = 0; u < ocr_words_[4]; ++u) {
          const std::array<UINT, 4u> core = partial ? std::array<UINT, 4u> {4u, 14u, 12u, 16u} : std::array<UINT, 4u> {4u, 5u, 12u, 12u};
          const std::array<UINT, 4u> cover = partial ? std::array<UINT, 4u> {2u, 12u, 14u, 16u} : std::array<UINT, 4u> {2u, 3u, 14u, 14u};
          std::copy(core.begin(), core.end(), ocr_words_.begin() + 16u + 8u * u);
          std::copy(cover.begin(), cover.end(), ocr_words_.begin() + 144u + 8u * u);
        }
      }
      sct_words_ = {};
      sct_words_[0] = 5u;
      sct_words_[1] = 894714707u;
      sct_words_[2] = caption ? 1u : 0u;
      std::copy_n(cb_words_.begin() + 12u, 4u, sct_words_.begin() + 4u);
      std::copy_n(cb_words_.begin(), 12u, sct_words_.begin() + 8u);
      sct_words_[20] = cb_words_[16];
      sct_words_[21] = cb_words_[17];
      sct_words_[22] = 27u;
      sct_words_[23] = 1u;
      uip_words_ = {};
      uip_words_[0] = 4u;
      uip_words_[1] = 0x34504955u;
      uip_words_[2] = 1u;
      uip_words_[3] = (1u << ocr_words_[4]) - 1u;
      std::copy_n(cb_words_.begin() + 12u, 4u, uip_words_.begin() + 4u);
      uip_words_[8] = cb_words_[16];
      uip_words_[9] = cb_words_[17];
      std::copy_n(cb_words_.begin(), 2u, uip_words_.begin() + 10u);
      std::copy_n(cb_words_.begin() + 4u, 4u, uip_words_.begin() + 12u);
      std::copy_n(cb_words_.begin() + 8u, 4u, uip_words_.begin() + 16u);
      program_.current_sct = sct_.srv.Get();
      program_.current_uip = uip_.srv.Get();
      program_.producer = {};
      program_.producer.original_source = origin_.srv.Get();
      program_.producer.transfer = enclosure::transfer_t::encoded_sdr;
      program_.producer.input_known = true;
      program_.producer.direct_encoded_recipe = 1u;
      program_.producer.postprocess_mode = 2u;
      program_.producer.expected_work = 1u;
      program_.producer.transaction_token = 1000u + frame;
      program_.producer.model_frame = frame;
      program_.producer.model_observation_us = 1000000u + frame;
      program_.producer.analysis_generation = 7u;
      program_.producer.analysis_roi = {0u, 0u, w, fullH_};
      root_for(2u);
      upload();
    }

    void paint_native(UINT left, UINT top, UINT right, UINT bottom, std::uint8_t code) {
      for (UINT y = top; y < bottom; ++y) {
        for (UINT x = left; x < right; ++x) {
          origin_pixels_[(y + top_) * W_ + x] = {code, code, code, 255u};
        }
      }
    }

    void upload_canonical_origin() {
      for (UINT y = 0u; y < H_; ++y) {
        for (UINT x = 0u; x < W_; ++x) {
          const auto p = origin_pixels_[(y + top_) * W_ + x];
          pixels_[y * W_ + x] = {code_half(p[0]), code_half(p[1]), code_half(p[2]), 0x3c00u};
        }
      }
      upload();
    }

    void probability_native_rect(UINT left, UINT top, UINT right, UINT bottom, float value) {
      constexpr UINT width = models::depth_coordinate_v2::subtitle_ocr_output_width;
      constexpr UINT height = models::depth_coordinate_v2::subtitle_ocr_output_height;
      // Integral fixture ratios keep all four native-center contributors inside
      // the assigned detector rectangle. This authors P; it does not score RGB.
      require(width % W_ == 0u && height % H_ == 0u);
      for (UINT y = top * height / H_; y < bottom * height / H_; ++y) {
        std::fill(probability_words_.begin() + y * width + left * width / W_,
                  probability_words_.begin() + y * width + right * width / W_, value);
      }
    }

    void weak_exterior_scene() {
      // The authored exterior white component occupies native x=[5,8).
      // Current S remains set while every P contributor for this component is 0.
      std::fill(support_words_.begin(), support_words_.end(), 1u);
      probability_native_rect(0u, 0u, 8u, H_, 0.0f);
      upload();
    }

    void root_for(UINT mode) {
      root_words_ = {};
      const auto t = program_.producer.transaction_token;
      root_words_[0] = 1u;
      root_words_[1] = 1u ^ 0xd1ec15a5u ^ 0x52434f4fu;
      root_words_[2] = UINT(t);
      root_words_[3] = UINT(t >> 32u);
      root_words_[4] = root_words_[2] ^ 0xa3756c91u;
      root_words_[5] = root_words_[3] ^ 0x5c8a936eu;
      root_words_[6] = 0x47524243u;
      root_words_[7] = 0x52434f4fu;
      std::copy_n(root_words_.begin() + 2u, 4u, root_words_.begin() + 8u);
      root_words_[12] = 0x54535152u;
      root_words_[13] = 1u;
      root_words_[14] = 1u ^ 0x6f435257u;
      if (mode != 2u) {
        root_words_[20] = root_words_[21] = root_words_[22] = 1u;
      }
      root_words_[24] = root_words_[25] = root_words_[26] = 1u;
      root_words_[52] = root_words_[53] = root_words_[54] = 1u;
      root_words_[60] = root_words_[61] = root_words_[62] = 1u;
      program_.producer.postprocess_mode = mode;
    }

    void upload() {
      context_->UpdateSubresource(rgb_.resource.Get(), 0u, nullptr, pixels_.data(), W_ * 8u, 0u);
      context_->UpdateSubresource(support_.resource.Get(), 0u, nullptr, support_words_.data(), W_ * 4u, 0u);
      context_->UpdateSubresource(origin_.resource.Get(), 0u, nullptr, origin_pixels_.data(), W_ * 4u, 0u);
      context_->UpdateSubresource(probability_.resource.Get(), 0u, nullptr, probability_words_.data(), 0u, 0u);
      context_->UpdateSubresource(ocr_.resource.Get(), 0u, nullptr, ocr_words_.data(), 0u, 0u);
      context_->UpdateSubresource(cb_.resource.Get(), 0u, nullptr, cb_words_.data(), 0u, 0u);
      context_->UpdateSubresource(root_.resource.Get(), 0u, nullptr, root_words_.data(), 0u, 0u);
      context_->UpdateSubresource(sct_.resource.Get(), 0u, nullptr, sct_words_.data(), 0u, 0u);
      context_->UpdateSubresource(uip_.resource.Get(), 0u, nullptr, uip_words_.data(), 0u, 0u);
    }

    std::array<ID3D11ComputeShader *, 5u> colour_ptrs() const {
      std::array<ID3D11ComputeShader *, 5u> p {};
      for (UINT i = 0; i < 5u; ++i) {
        p[i] = colour_[i].Get();
      }
      return p;
    }

    std::array<ID3D11ComputeShader *, 6u> graph_ptrs() const {
      std::array<ID3D11ComputeShader *, 6u> p {};
      for (UINT i = 0; i < 6u; ++i) {
        p[i] = graph_[i].Get();
      }
      return p;
    }

    bool run(bool enabled = true, bool recorded = true) {
      helper_.set_recorded_execution(recorded);
      return helper_.dispatch(context_.Get(), colour_ptrs(), graph_ptrs(), cb_.resource.Get(), rgb_.srv.Get(), probability_.srv.Get(), support_.srv.Get(), ocr_.srv.Get(), root_.resource.Get(), 240u, error_, enabled ? &program_ : nullptr);
    }

    std::vector<char> read(ID3D11Resource *r) {
      ComPtr<ID3D11Resource> stage;
      D3D11_RESOURCE_DIMENSION dim;
      r->GetType(&dim);
      UINT row = 0u, rows = 1u;
      if (dim == D3D11_RESOURCE_DIMENSION_TEXTURE2D) {
        ComPtr<ID3D11Texture2D> t;
        require(SUCCEEDED(r->QueryInterface(IID_PPV_ARGS(&t))));
        D3D11_TEXTURE2D_DESC d {};
        t->GetDesc(&d);
        row = d.Width * (d.Format == DXGI_FORMAT_R16G16B16A16_FLOAT ? 8u : 4u);
        rows = d.Height;
        require(d.ArraySize == 1u && d.MipLevels == 1u);
        d.Usage = D3D11_USAGE_STAGING;
        d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        d.BindFlags = d.MiscFlags = 0u;
        ComPtr<ID3D11Texture2D> x;
        require(SUCCEEDED(device_->CreateTexture2D(&d, nullptr, &x)));
        stage = x;
      } else {
        ComPtr<ID3D11Buffer> b;
        require(SUCCEEDED(r->QueryInterface(IID_PPV_ARGS(&b))));
        D3D11_BUFFER_DESC d {};
        b->GetDesc(&d);
        row = d.ByteWidth;
        d.Usage = D3D11_USAGE_STAGING;
        d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        d.BindFlags = d.MiscFlags = d.StructureByteStride = 0u;
        ComPtr<ID3D11Buffer> x;
        require(SUCCEEDED(device_->CreateBuffer(&d, nullptr, &x)));
        stage = x;
      }
      context_->CopyResource(stage.Get(), r);
      D3D11_MAPPED_SUBRESOURCE m {};
      require(SUCCEEDED(context_->Map(stage.Get(), 0u, D3D11_MAP_READ, 0u, &m)));
      std::vector<char> b(static_cast<std::size_t>(row) * rows);
      for (UINT y = 0; y < rows; ++y) {
        std::memcpy(b.data() + static_cast<std::size_t>(y) * row, static_cast<const char *>(m.pData) + static_cast<std::size_t>(y) * (dim == D3D11_RESOURCE_DIMENSION_TEXTURE2D ? m.RowPitch : row), row);
      }
      context_->Unmap(stage.Get(), 0u);
      return b;
    }

    std::vector<char> read(ID3D11ShaderResourceView *s) {
      ComPtr<ID3D11Resource> r;
      s->GetResource(&r);
      return read(r.Get());
    }

    std::vector<UINT> receipt() {
      return words(read(helper_.mutable_views().receipt.Get()));
    }

    std::vector<char> body() {
      return read(helper_.mutable_views().body.Get());
    }

    std::vector<char> matte() {
      return read(helper_.mutable_views().matte.Get());
    }

    void expect_baseline() {
      const auto b = helper_.baseline_views();
      EXPECT_EQ(body(), read(b.body.Get()));
      EXPECT_EQ(matte(), read(b.matte.Get()));
    }

    void expect_ready() {
      const auto r = receipt();
      ASSERT_EQ(r.size(), 512u);
      EXPECT_EQ(r[0], 0x31434e45u);
      EXPECT_EQ(r[1], 5u);
      EXPECT_EQ(r[2], 1u);
      EXPECT_EQ(r[3], 0u);
      EXPECT_EQ(r[81u], 0x31424649u);EXPECT_EQ(r[82u], 2u);EXPECT_EQ(r[83u], 17u);EXPECT_EQ(r[84u], 8u);
      EXPECT_EQ(r[87u] & ~255u, 0u); // Restored seen includes rejected/weak candidates.
      for (UINT unit = 0u; unit < 8u; ++unit) {
        const UINT u = 128u + 32u * unit;
        EXPECT_LE(r[u + 30u], r[u + 3u]); // OLD-only closure cannot exceed combined closure.
        EXPECT_LE(r[u + 3u], r[u + 4u]);
        EXPECT_LE(r[u + 5u], r[u + 4u]);
      }
      const auto v = helper_.mutable_views();
      const auto original = words(read(v.all_body_bits.Get())), selected = words(read(v.selected_bits.Get()));
      ASSERT_EQ(original.size(), selected.size());
      for (std::size_t i = 0u; i < selected.size(); ++i) {
        EXPECT_EQ(original[i] & ~65535u, 0u); // No phase-local restore/pending bits exported.
        EXPECT_EQ(selected[i] & ~65535u, 0u);
        EXPECT_EQ(((selected[i] >> 8u) & 255u) & ~(original[i] & 255u), 0u);
      }
      EXPECT_EQ(std::vector<UINT>(r.begin() + 16u, r.begin() + 40u), std::vector<UINT>(cb_words_.begin(), cb_words_.end()));
    }


    struct mode_graph_observation_t {
      UINT mode_count = 0u; // Actual reissued unit0 palette, paired with observed graph.
      std::vector<UINT> eligible, parent, flags;
    };

    UINT native_graph_root(const std::vector<UINT> &parent, UINT id) {
      for (UINT hops=0u;hops<parent.size();++hops) {
        require(id>0u&&id<parent.size());const UINT next=parent[id];
        require(next!=0u&&next!=0xffffffffu&&next<=id);
        if(next==id)return id;id=next;
      }
      throw std::runtime_error("Non-convergent native graph");
    }

    void author_local_support_pair(bool supported, UINT frame) {
      author(true,frame,1u,false,false,false,64u,32u);
      paint_native(0u,0u,W_,H_,32u);
      // Known-K calibration mixture; this stripe touches the colour-search boundary.
      paint_native(18u,0u,28u,19u,1u);paint_native(20u,0u,26u,H_,255u);
      // A genuine independently accepted current white component confirms F.
      paint_native(28u,8u,36u,22u,1u);paint_native(30u,10u,34u,20u,255u);
      // B has no colour seed. Its non-white S witness is in the same closed region,
      // below white eligibility and outside learned K; S never directly seeds B.
      paint_native(38u,14u,50u,26u,1u);paint_native(40u,16u,48u,24u,255u);
      paint_native(41u,17u,42u,18u,96u);
      std::fill(support_words_.begin(),support_words_.end(),0u);
      support_words_[17u*W_+41u]=supported?1u:0u;
      std::fill(probability_words_.begin(),probability_words_.end(),0.0f);
      probability_native_rect(20u,0u,26u,H_,1.0f);
      probability_native_rect(30u,10u,34u,20u,1.0f);
      upload_canonical_origin();
    }

    void advance_local_support_observation(UINT frame, bool empty) {
      cb_words_[12u]=frame;cb_words_[13u]=0u;cb_words_[16u]=1000000u+frame;cb_words_[17u]=0u;
      std::copy_n(cb_words_.begin()+12u,4u,ocr_words_.begin()+5u);
      std::copy_n(cb_words_.begin()+12u,4u,sct_words_.begin()+4u);
      std::copy_n(cb_words_.begin()+12u,4u,uip_words_.begin()+4u);
      sct_words_[20u]=uip_words_[8u]=cb_words_[16u];sct_words_[21u]=uip_words_[9u]=0u;
      program_.producer.transaction_token=1000u+frame;program_.producer.model_frame=frame;
      program_.producer.model_observation_us=1000000u+frame;
      std::fill(support_words_.begin(),support_words_.end(),0u);
      if(empty) {
        paint_native(0u,0u,W_,H_,32u);std::fill(probability_words_.begin(),probability_words_.end(),0.0f);
        ocr_words_[3u]=ocr_words_[4u]=0u;std::fill(ocr_words_.begin()+16u,ocr_words_.end(),0u);
        sct_words_[2u]=0u;uip_words_[3u]=0u;
      }
      root_for(2u);upload_canonical_origin();
    }

    void expect_local_phase_clear() {
      expect_ready();const auto r=receipt();
      for(UINT unit=0u;unit<8u;++unit)EXPECT_EQ(r[128u+32u*unit+31u],0u);
    }

    std::vector<char> emit_mode_seeds(const std::array<ID3D11ComputeShader *, 5u> &colour, UINT mode, mode_graph_observation_t *observation = nullptr) {
      baseline::resources_t reference;
      require(reference.ensure(device_.Get(), W_, H_, error_));
      reference.set_recorded_execution(false);
      require(reference.dispatch(context_.Get(), colour, graph_ptrs(), cb_.resource.Get(), rgb_.srv.Get(), probability_.srv.Get(), support_.srv.Get(), ocr_.srv.Get(), root_.resource.Get(), 240u, error_));
      const auto v = reference.mutable_views();
      auto request = buffer(16u, true);
      const std::array<UINT, 4u> req {0u, mode, 0u, 0u}, zero {};
      context_->UpdateSubresource(request.resource.Get(), 0u, nullptr, req.data(), 0u, 0u);
      auto palette = uav(v.palette.Get()), eligible = uav(v.eligible.Get()), seeds = uav(v.seeds.Get()), domain = uav(v.domain_tag.Get()), cover = uav(v.cover_clip.Get());
      ComPtr<ID3D11Resource> args_resource;v.mode_args->GetResource(&args_resource);
      D3D11_UNORDERED_ACCESS_VIEW_DESC desc {};desc.Format = DXGI_FORMAT_R32_TYPELESS;desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;desc.Buffer.NumElements = 24u;desc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
      ComPtr<ID3D11UnorderedAccessView> args;
      require(SUCCEEDED(device_->CreateUnorderedAccessView(args_resource.Get(), &desc, &args)));
      context_->ClearUnorderedAccessViewUint(args.Get(), zero.data());
      std::array<ID3D11ShaderResourceView *, 9u> in {rgb_.srv.Get(), probability_.srv.Get(), support_.srv.Get(), ocr_.srv.Get(), nullptr, nullptr, nullptr, nullptr, v.validation_receipt.Get()};
      std::array<ID3D11UnorderedAccessView *, 8u> out {};out[0] = palette.Get();out[7] = args.Get();
      ID3D11Buffer *constants[2] {cb_.resource.Get(), request.resource.Get()};
      context_->CSSetConstantBuffers(0u, 2u, constants);context_->CSSetShaderResources(0u, 9u, in.data());context_->CSSetUnorderedAccessViews(0u, 8u, out.data(), nullptr);context_->CSSetShader(colour[0], nullptr, 0u);context_->DispatchIndirect(root_.resource.Get(), 240u);
      out.fill(nullptr);context_->CSSetUnorderedAccessViews(0u, 8u, out.data(), nullptr);
      const auto actual_palette = words(read(v.palette.Get()));require(actual_palette[4u] > mode);
      in[4] = v.palette.Get();out[1] = eligible.Get();out[2] = seeds.Get();out[3] = domain.Get();out[4] = cover.Get();
      context_->CSSetShaderResources(0u, 9u, in.data());context_->CSSetUnorderedAccessViews(0u, 8u, out.data(), nullptr);context_->CSSetShader(colour[1], nullptr, 0u);context_->DispatchIndirect(v.current_mode_args.Get(), mode * 12u);
      out.fill(nullptr);in.fill(nullptr);context_->CSSetUnorderedAccessViews(0u, 8u, out.data(), nullptr);context_->CSSetShaderResources(0u, 9u, in.data());
      if(observation) {
        observation->mode_count=actual_palette[4u];
        observation->eligible=words(read(v.eligible.Get()));observation->parent=words(read(v.parent.Get()));
        observation->flags=words(read(v.root_flags.Get()));
      }
      return read(v.seeds.Get());
    }

    ComPtr<ID3D11UnorderedAccessView> uav(ID3D11ShaderResourceView *s) {
      ComPtr<ID3D11Resource> r;
      s->GetResource(&r);
      ComPtr<ID3D11UnorderedAccessView> u;
      require(SUCCEEDED(device_->CreateUnorderedAccessView(r.Get(), nullptr, &u)));
      return u;
    }

    void inject_late_parent_fault() {
      // Authored stage fault only: after real full extraction/ROIgraph, force actual Parent[0],
      // rerun the actual exterior resolve and publish. This is not an inference input or host verdict.
      const auto b = helper_.baseline_views();
      const auto v = helper_.mutable_views();
      ComPtr<ID3D11Resource> p;
      b.parent->GetResource(&p);
      const UINT invalid = 0xffffffffu;
      const D3D11_BOX box {0u, 0u, 0u, 4u, 1u, 1u};
      context_->UpdateSubresource(p.Get(), 0u, &box, &invalid, 0u, 0u);
      const std::array<UINT, 4u> hook {};
      context_->UpdateSubresource(v.hook_constants.Get(), 0u, nullptr, hook.data(), 0u, 0u);
      ID3D11Buffer *cb[3] {cb_.resource.Get(), v.hook_constants.Get(), v.producer_constants.Get()};
      std::array<ID3D11ShaderResourceView *, 16u> s {rgb_.srv.Get(), ocr_.srv.Get(), b.validation_receipt.Get(), b.palette.Get(), b.eligible.Get(), b.cover_clip.Get(), b.parent.Get(), b.root_flags.Get(), v.all_body_bits.Get(), v.cover_bits.Get(), sct_.srv.Get(), uip_.srv.Get(), nullptr, origin_.srv.Get(), b.body.Get(), b.matte.Get()};
      ComPtr<ID3D11ShaderResourceView> mirror;
      D3D11_SHADER_RESOURCE_VIEW_DESC d {};
      d.Format = DXGI_FORMAT_R32_TYPELESS;
      d.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
      d.BufferEx.NumElements = 64u;
      d.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
      require(SUCCEEDED(device_->CreateShaderResourceView(v.root_mirror.Get(), &d, &mirror)));
      s[12] = mirror.Get();
      std::array<ID3D11UnorderedAccessView *, 8u> u {};
      auto meta = uav(v.receipt.Get()), ex = uav(b.mode_body.Get());
      u[0] = meta.Get();
      u[6] = ex.Get();
      context_->CSSetConstantBuffers(0u, 3u, cb);
      context_->CSSetShaderResources(0u, 16u, s.data());
      context_->CSSetUnorderedAccessViews(0u, 8u, u.data(), nullptr);
      context_->CSSetShader(enclosure_[8].Get(), nullptr, 0u);
      context_->Dispatch((W_ + 15u) / 16u, (H_ + 15u) / 16u, 1u);
      u.fill(nullptr);
      s[4] = nullptr;
      context_->CSSetUnorderedAccessViews(0u, 8u, u.data(), nullptr);
      context_->CSSetShaderResources(0u, 16u, s.data());
      auto selected = uav(v.selected_bits.Get()), bo = uav(v.body.Get()), ma = uav(v.matte.Get());
      u[0] = meta.Get();
      u[3] = selected.Get();
      u[4] = bo.Get();
      u[5] = ma.Get();
      context_->CSSetUnorderedAccessViews(0u, 8u, u.data(), nullptr);
      context_->CSSetShader(enclosure_[11].Get(), nullptr, 0u);
      context_->Dispatch((W_ + 15u) / 16u, (H_ + 15u) / 16u, 1u);
      u.fill(nullptr);
      s.fill(nullptr);
      context_->CSSetUnorderedAccessViews(0u, 8u, u.data(), nullptr);
      context_->CSSetShaderResources(0u, 16u, s.data());
    }

    void saved();
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    enclosure::resources_t helper_;
    enclosure::program_t program_;
    std::array<ComPtr<ID3D11ComputeShader>, 5u> colour_;
    std::array<ComPtr<ID3D11ComputeShader>, 6u> graph_;
    std::array<ComPtr<ID3D11ComputeShader>, 12u> enclosure_;
    std::array<ComPtr<ID3D11ComputeShader>, 3u> roi_;
    texture_t rgb_, support_, origin_;
    buffer_t probability_, ocr_, sct_, uip_, cb_, root_;
    std::array<UINT, 24u> cb_words_ {};
    std::array<UINT, 256u> ocr_words_ {};
    std::array<UINT, 32u> sct_words_ {};
    std::array<UINT, 40u> uip_words_ {};
    std::array<UINT, 64u> root_words_ {};
    std::vector<pixel16> pixels_;
    std::vector<pixel8> origin_pixels_;
    std::vector<float> probability_words_;
    std::vector<UINT> support_words_;
    UINT W_ = 0u, H_ = 0u, fullH_ = 0u, top_ = 0u;
    bool hardware_ = false;
    std::string error_, inner_closure_, graph_closure_, enclosure_closure_;
    nlohmann::json adapter_;
  };

  TEST_F(HostSbsCurrentBlackEnclosureGpuTest, NormalRolesAndRestoreTemporaryBitsAreNormalized) {
    ASSERT_TRUE(run(true, false)) << error_;
    expect_ready();
    const auto r = receipt();
    EXPECT_EQ(r[87u], 0u) << "This authored ordinary fixture has no clipped colour component";
    for (UINT unit = 0u; unit < 8u; ++unit) {
      const UINT u = 128u + 32u * unit;
      EXPECT_EQ(r[u + 30u], r[u + 3u]);
    }
    const auto a = words(read(helper_.mutable_views().all_body_bits.Get()));
    EXPECT_TRUE(std::all_of(a.begin(), a.end(), [](UINT x) { return (x & 0xffff0000u) == 0u; }));
    const auto before_body = body(), before_matte = matte();
    ASSERT_TRUE(run(true, true)) << error_;
    EXPECT_EQ(body(), before_body);EXPECT_EQ(matte(), before_matte);
  }

  TEST_F(HostSbsCurrentBlackEnclosureGpuTest, GrowthTopRestorationKeepsExactReferenceSeedMap) {
    const auto *root = std::getenv("SUNSHINE_TEST_SEED_REFERENCE_SHADER_ROOT"), *pin = std::getenv("SUNSHINE_TEST_SEED_REFERENCE_SHADER_PIN");
    if (!root || !pin) GTEST_SKIP() << "Explicit frozen old130 Colour reference required; not a PASS";
    ASSERT_STREQ(pin, "d09f347ef5030cb01106a70312efee067a8a87e280c1b5696851d46f3b61eef1");
    const std::array spec {
      cache::shader_spec {"host_sbs_subtitle_inner_colour_cs.hlsl", "palette_main", "cs_5_0"},
      cache::shader_spec {"host_sbs_subtitle_inner_colour_cs.hlsl", "emit_main", "cs_5_0"},
      cache::shader_spec {"host_sbs_subtitle_inner_colour_cs.hlsl", "clear_main", "cs_5_0"},
      cache::shader_spec {"host_sbs_subtitle_inner_colour_cs.hlsl", "accumulate_main", "cs_5_0"},
      cache::shader_spec {"host_sbs_subtitle_inner_colour_cs.hlsl", "validate_main", "cs_5_0"}
    };
    std::array<ComPtr<ID3D11ComputeShader>, 5u> reference;
    load_shaders(root, pin, spec, reference);
    std::array<ID3D11ComputeShader *, 5u> refs {};
    for (UINT i = 0u; i < 5u; ++i) refs[i] = reference[i].Get();
    // Same authored RGB/P/S; this core makes the old seed top differ from Cover top.
    ocr_words_[17u] = 7u;upload();
    const auto expected = emit_mode_seeds(refs, 0u), actual = emit_mode_seeds(colour_ptrs(), 0u);
    EXPECT_EQ(actual, expected);
    const auto seed_words = words(actual);
    EXPECT_GT(std::count_if(seed_words.begin(), seed_words.end(), [](UINT x) { return x != 0u; }), 0u);
  }


  TEST_F(HostSbsCurrentBlackEnclosureGpuTest, LocalCurrentSupportDoesNotBorrowAnotherComponent) {
    for(bool supported:{false,true}) {
      SCOPED_TRACE(supported);author_local_support_pair(supported,801u+(supported?1u:0u));
      ASSERT_TRUE(run())<<error_;expect_local_phase_clear();
      const auto r=receipt();const auto b=helper_.baseline_views();const auto v=helper_.mutable_views();
      const auto original=words(read(b.body.Get())),selected=words(read(v.selected_bits.Get()));
      const auto actualBody=words(body()),actualM=words(matte()),family=words(read(v.family_models.Get()));
      const auto parent=words(read(b.parent.Get())),flags=words(read(b.root_flags.Get()));
      constexpr UINT target=20u*64u+44u,witness=17u*64u+41u,accepted=14u*64u+30u;
      ASSERT_EQ(r[386u],1u);ASSERT_EQ(r[135u],1u);ASSERT_NE(family[0u]&0x100u,0u);
      ASSERT_EQ(original[accepted],1u);ASSERT_EQ(original[target],0u);ASSERT_EQ(original[witness],0u);
      const UINT root=native_graph_root(parent,target+1u);
      ASSERT_EQ(native_graph_root(parent,witness+1u),root);
      ASSERT_EQ(flags[root],supported?4u:0u)<<"Actual closed CBC region carries only its own current S";
      mode_graph_observation_t graph;const auto seeds=words(emit_mode_seeds(colour_ptrs(),0u,&graph));
      ASSERT_EQ(graph.mode_count,1u)<<"Mode0 seeds and graph must share the single actual colour mode";
      ASSERT_EQ(graph.eligible[target],1u);ASSERT_EQ(seeds[target],0u);
      ASSERT_EQ(graph.flags[native_graph_root(graph.parent,target+1u)],0u);
      ASSERT_EQ(graph.eligible[witness],0u);ASSERT_EQ(seeds[witness],0u);
      ASSERT_EQ(seeds[accepted],3u);ASSERT_EQ(graph.flags[native_graph_root(graph.parent,accepted+1u)],5u);
      EXPECT_EQ(selected[target]&0x101u,supported?0x101u:0u);
      EXPECT_EQ(r[87u],supported?1u:0u);
      for(UINT y=16u;y<24u;++y)for(UINT x=40u;x<48u;++x)if(x!=41u||y!=17u) {
        EXPECT_EQ(actualBody[y*W_+x],supported?1u:0u);EXPECT_EQ(actualM[y*W_+x],supported?1u:0u);
      }
      for(UINT y=10u;y<20u;++y)for(UINT x=30u;x<34u;++x)EXPECT_EQ(actualBody[y*W_+x],1u);
    }
  }

  TEST_F(HostSbsCurrentBlackEnclosureGpuTest, LocalSupportAndRestorationClearOnFirstCurrentS0AndEmpty) {
    author_local_support_pair(true,811u);ASSERT_TRUE(run())<<error_;expect_local_phase_clear();
    constexpr UINT target=20u*64u+44u;ASSERT_EQ(words(body())[target],1u);ASSERT_EQ(receipt()[87u],1u);
    const auto *rgb=rgb_.resource.Get(),*support=support_.resource.Get();
    const auto *allBody=helper_.mutable_views().all_body_bits.Get();
    for(bool empty:{false,true}) {
      advance_local_support_observation(empty?813u:812u,empty);
      ASSERT_EQ(rgb_.resource.Get(),rgb);ASSERT_EQ(support_.resource.Get(),support);
      ASSERT_TRUE(run())<<error_;expect_local_phase_clear();
      ASSERT_EQ(helper_.mutable_views().all_body_bits.Get(),allBody);
      const auto actualBody=words(body()),actualM=words(matte());const auto r=receipt();
      EXPECT_EQ(actualBody[target],0u);EXPECT_EQ(actualM[target],0u);EXPECT_EQ(r[87u],0u);
      if(!empty) {
        const auto b=helper_.baseline_views();const auto parent=words(read(b.parent.Get())),flags=words(read(b.root_flags.Get()));
        EXPECT_EQ(flags[native_graph_root(parent,target+1u)]&4u,0u);
        for(UINT y=10u;y<20u;++y)for(UINT x=30u;x<34u;++x)EXPECT_EQ(actualBody[y*W_+x],1u);
      }else {
        EXPECT_TRUE(std::all_of(actualBody.begin(),actualBody.end(),[](UINT x){return x==0u;}));
        EXPECT_TRUE(std::all_of(actualM.begin(),actualM.end(),[](UINT x){return x==0u;}));
        EXPECT_EQ(r[7u],0u);
      }
    }
  }

  TEST_F(HostSbsCurrentBlackEnclosureGpuTest, NullProgramLiteralBaselineBodyAndMatte) {
    ASSERT_TRUE(run(false));
    const auto v = helper_.baseline_views();
    const auto b = read(v.body.Get()), m = read(v.matte.Get());
    baseline::resources_t reference;
    ASSERT_TRUE(reference.ensure(device_.Get(), W_, H_, error_));
    ASSERT_TRUE(reference.dispatch(context_.Get(), colour_ptrs(), graph_ptrs(), cb_.resource.Get(), rgb_.srv.Get(), probability_.srv.Get(), support_.srv.Get(), ocr_.srv.Get(), root_.resource.Get(), 240u, error_));
    EXPECT_EQ(b, read(reference.mutable_views().body.Get()));
    EXPECT_EQ(m, read(reference.mutable_views().matte.Get()));
    EXPECT_FALSE(helper_.last_commands_enabled());
  }

  TEST_F(HostSbsCurrentBlackEnclosureGpuTest, ClosedCode1RefinesWeakExteriorSceneAndDirectRecordedMatch) {
    weak_exterior_scene();
    ASSERT_TRUE(run(true, false)) << error_;
    expect_ready();
    const auto b = body(), m = matte();
    const auto r = receipt();
    ASSERT_EQ(r[10], 2u);
    EXPECT_NE(b, read(helper_.baseline_views().body.Get()));
    ASSERT_TRUE(run(true, true)) << error_;
    EXPECT_EQ(body(), b);
    EXPECT_EQ(matte(), m);
    EXPECT_EQ(receipt(), r);
    EXPECT_EQ(helper_.diagnostics().recordings, 1u);
    ASSERT_TRUE(run());
    EXPECT_EQ(helper_.diagnostics().recordings, 1u);
    EXPECT_EQ(helper_.diagnostics().replays, 2u);
  }

  TEST_F(HostSbsCurrentBlackEnclosureGpuTest, RootZeroPoisonHoldsReceiptOutputsAndFreshEmptyClears) {
    ASSERT_TRUE(run());
    const auto b = body(), m = matte();
    const auto r = receipt();
    std::array<UINT, 24u> poison;
    poison.fill(1u);
    context_->UpdateSubresource(helper_.baseline_views().current_mode_args.Get(), 0u, nullptr, poison.data(), 0u, 0u);
    author(false, 2u);
    root_words_[60] = 0u;
    upload();
    ASSERT_TRUE(run());
    EXPECT_EQ(body(), b);
    EXPECT_EQ(matte(), m);
    EXPECT_EQ(receipt(), r);
    EXPECT_EQ(words(read(helper_.mutable_views().unit_arguments.Get())), std::vector<UINT>(16u, 0u));
    root_words_[60] = 1u;
    upload();
    ASSERT_TRUE(run());
    expect_ready();
    EXPECT_EQ(body(), std::vector<char>(W_ * H_ * 4u, 0));
    EXPECT_EQ(matte(), std::vector<char>(W_ * H_ * 4u, 0));
  }

  TEST_F(HostSbsCurrentBlackEnclosureGpuTest, ActualClass1Class2Class4AndInferArgumentsJoin) {
    for (UINT mode : {1u, 2u, 4u}) {
      root_for(mode);
      upload();
      ASSERT_TRUE(run());
      expect_ready();
      EXPECT_EQ(receipt()[11], mode);
      root_words_[20] = mode == 2u ? 1u : 0u;
      upload();
      ASSERT_TRUE(run());
      EXPECT_EQ(receipt()[2], 0u);
      EXPECT_NE(receipt()[3], 0u);
      expect_baseline();
    }
  }

  TEST_F(HostSbsCurrentBlackEnclosureGpuTest, StaleTokenRequestCookiesAndOptionalWorkReject) {
    const auto good = root_words_;
    for (UINT word : {2u, 10u, 12u, 14u, 15u, 52u}) {
      root_words_ = good;
      root_words_[word] ^= 1u;
      upload();
      ASSERT_TRUE(run());
      EXPECT_EQ(receipt()[2], 0u);
      EXPECT_NE(receipt()[3], 0u);
      expect_baseline();
    }
    root_words_ = good;
    upload();
    ASSERT_TRUE(run());
    expect_ready();
  }

  TEST_F(HostSbsCurrentBlackEnclosureGpuTest, OwnSctUipCbModelAndObservationMustJoin) {
    const auto gs = sct_words_;
    const auto gu = uip_words_;
    const auto gc = cb_words_;
    for (UINT k : {4u, 6u, 20u, 22u, 23u}) {
      sct_words_ = gs;
      sct_words_[k] ^= 1u;
      upload();
      ASSERT_TRUE(run());
      EXPECT_EQ(receipt()[2], 0u);
      expect_baseline();
    }
    sct_words_ = gs;
    uip_words_ = gu;
    uip_words_[20] ^= 1u;
    upload();
    ASSERT_TRUE(run());
    EXPECT_EQ(receipt()[2], 0u);
    uip_words_ = gu;
    cb_words_ = gc;
    program_.producer.model_observation_us++;
    ASSERT_TRUE(run());
    EXPECT_EQ(receipt()[2], 0u);
    expect_baseline();
    program_.producer.model_observation_us--;
    upload();
    ASSERT_TRUE(run());
    expect_ready();
    const auto private_views = helper_.mutable_views();
    const auto private_body = uav(private_views.body.Get()), private_matte = uav(private_views.matte.Get());
    constexpr std::array<UINT, 4u> poison {1u, 1u, 1u, 1u};
    for (const auto [word, value] : std::array<std::pair<UINT, UINT>, 3u> {{{0u, 0u}, {3u, 0u}, {3u, H_ - 1u}}}) {
      context_->ClearUnorderedAccessViewUint(private_body.Get(), poison.data());
      context_->ClearUnorderedAccessViewUint(private_matte.Get(), poison.data());
      cb_words_ = gc;
      cb_words_[word] = value;
      upload();
      ASSERT_TRUE(run());
      EXPECT_EQ(receipt()[2], 0u);
      EXPECT_NE(receipt()[3], 0u);
      EXPECT_EQ(body(), std::vector<char>(W_ * H_ * 4u, 0));
      EXPECT_EQ(matte(), std::vector<char>(W_ * H_ * 4u, 0));
    }
    cb_words_ = gc;
    upload();
    ASSERT_TRUE(run());
    expect_ready();
  }

  TEST_F(HostSbsCurrentBlackEnclosureGpuTest, UnsupportedTransferViewAndGpuSourceMismatchKeepWholeBaseline) {
    program_.producer.transfer = enclosure::transfer_t::linear;
    ASSERT_TRUE(run());
    expect_ready();
    EXPECT_EQ(receipt()[10], 1u);
    expect_baseline();
    program_.producer.transfer = enclosure::transfer_t::encoded_sdr;
    const auto origin = program_.producer.original_source;
    program_.producer.original_source = probability_.srv.Get();
    ASSERT_TRUE(run());
    expect_ready();
    EXPECT_EQ(receipt()[4], 0u);
    EXPECT_EQ(receipt()[10], 1u);
    expect_baseline();
    program_.producer.original_source = origin;
    origin_pixels_[0][0] ^= 1u;
    context_->UpdateSubresource(origin_.resource.Get(), 0u, nullptr, origin_pixels_.data(), W_ * 4u, 0u);
    ASSERT_TRUE(run());
    expect_ready();
    EXPECT_EQ(receipt()[5], 0u);
    EXPECT_EQ(receipt()[10], 1u);
    expect_baseline();
    origin_pixels_[0][0] ^= 1u;
    probability_words_.back() = std::bit_cast<float>(0x7fc00000u);
    upload();
    ASSERT_TRUE(run());
    EXPECT_EQ(receipt()[2], 0u);
    EXPECT_NE(receipt()[3], 0u);
    expect_baseline();
    probability_words_.back() = 1.0f;
    support_words_.back() = 2u;
    upload();
    ASSERT_TRUE(run());
    EXPECT_EQ(receipt()[2], 0u);
    EXPECT_NE(receipt()[3], 0u);
    expect_baseline();
  }

  TEST_F(HostSbsCurrentBlackEnclosureGpuTest, OpenUnitNoEnclosedAndPartialCoverFallbackWithoutGlyphLoss) {
    author(true, 2u, 1u, true);
    ASSERT_TRUE(run());
    expect_ready();
    expect_baseline();
    EXPECT_NE(receipt()[8] & 1u, 0u);
    author(true, 3u, 1u, false, true);
    ASSERT_TRUE(run());
    expect_ready();
    expect_baseline();
    EXPECT_NE(receipt()[8] & 1u, 0u);
    // This exterior component is deliberately S-only without a strong current-P
    // seed. The separate main fill has strong P and a code1 closed rim.
    author(true, 4u);
    weak_exterior_scene();
    for (UINT y = 4u; y < 13u; ++y) {
      for (UINT x = 4u; x < 8u; ++x) {
        if (x >= 5u && y >= 5u && y < 12u) {
          continue;
        }
        origin_pixels_[y * W_ + x] = {2u, 2u, 2u, 255u};
      }
    }
    for (UINT y = 0u; y < H_; ++y) {
      for (UINT x = 0u; x < W_; ++x) {
        const auto p = origin_pixels_[y * W_ + x];
        pixels_[y * W_ + x] = {code_half(p[0]), code_half(p[1]), code_half(p[2]), 0x3c00u};
      }
    }
    upload();
    ASSERT_TRUE(run());
    expect_ready();
    const auto original = words(read(helper_.baseline_views().body.Get()));
    const auto selected = words(body());
    EXPECT_EQ(original[6u * W_ + 6u], 1u);
    EXPECT_EQ(selected[6u * W_ + 6u] & 255u, 0u);
    EXPECT_EQ(selected[8u * W_ + 16u] & 255u, 1u);
  }

  TEST_F(HostSbsCurrentBlackEnclosureGpuTest, OpenStrongGlyphSurvivesSmallAndWideGapsBesideClosedWeakPeer) {
    for (UINT gap_width : {1u, 5u}) {
      SCOPED_TRACE(gap_width);
      author(true, gap_width + 10u, 1u, false, false, false, 64u, 32u);
      paint_native(0u, 0u, W_, H_, 32u);
      paint_native(12u, 10u, 26u, 24u, 1u);
      paint_native(14u, 12u, 24u, 22u, 255u);
      paint_native(36u, 10u, 50u, 24u, 1u);
      paint_native(38u, 12u, 48u, 22u, 255u);
      // An actual channel through the first rim, without changing its white fill.
      paint_native(18u, 10u, 18u + gap_width, 12u, 32u);
      std::fill(support_words_.begin(), support_words_.end(), 1u);
      probability_native_rect(32u, 0u, W_, H_, 0.0f);
      upload_canonical_origin();
      ASSERT_TRUE(run()) << error_;
      expect_ready();
      const auto r = receipt();
      ASSERT_EQ(r[128u + 29u], 1u) << "Fixture needs an admitted enclosure graph";
      ASSERT_GT(r[128u + 3u], 0u) << "Closed S-only peer must supply enclosed evidence";
      ASSERT_NE(r[7u] & 1u, 0u) << "Refinement must run, not hide behind fallback";
      const auto original = words(read(helper_.baseline_views().body.Get()));
      const auto selected = words(read(helper_.mutable_views().selected_bits.Get()));
      ASSERT_EQ(original[16u * W_ + 18u], 1u);
      ASSERT_EQ(original[16u * W_ + 42u], 1u);
      EXPECT_EQ(selected[16u * W_ + 18u] & 255u, 1u);
      EXPECT_EQ(selected[16u * W_ + 42u] & 255u, 1u);
      expect_baseline();
    }
  }

  TEST_F(HostSbsCurrentBlackEnclosureGpuTest, ExteriorSourceSeedCannotGrantTrustedStyleButConnectedInkSurvives) {
    // ONE fresh observation: no temporal confirmation/dwell is part of this case.
    for (UINT gap_width : {1u, 5u}) {
      SCOPED_TRACE(gap_width);
      author(true, 40u + gap_width, 1u, false, false, false, 64u, 32u);
      paint_native(0u, 0u, W_, H_, 32u);
      paint_native(12u, 10u, 26u, 24u, 1u);
      paint_native(14u, 12u, 24u, 22u, 255u);
      paint_native(18u, 10u, 18u + gap_width, 12u, 32u);
      paint_native(36u, 10u, 50u, 24u, 1u);
      paint_native(38u, 12u, 48u, 22u, 255u);
      // Connected gray ink/AA shares the white component and is one native step
      // from current modal-white witnesses. Its 95-code jump deliberately needs
      // the style-modal reach channel, rather than the 0.08 RGB-coherent channel.
      paint_native(18u, 14u, 20u, 20u, 160u);
      // A separate scene patch has an L/open black witness, not an enclosing rim.
      // q=(30,18) is three pixels from black x27, two steps from source seed x28,
      // and at least seven steps from every white/modal pixel.
      paint_native(26u, 15u, 28u, 21u, 1u);
      paint_native(28u, 15u, 34u, 21u, 160u);
      std::fill(support_words_.begin(), support_words_.end(), 1u);
      probability_native_rect(32u, 0u, W_, H_, 0.0f); // weak closed peer
      probability_native_rect(18u, 14u, 20u, 20u, 0.0f);
      probability_native_rect(28u, 15u, 34u, 21u, 0.0f);
      probability_native_rect(28u, 15u, 29u, 21u, 1.0f); // one genuine strong seed edge
      upload_canonical_origin();
      ASSERT_TRUE(run()) << error_;
      expect_ready();
      const auto r = receipt();
      ASSERT_EQ(r[384u + 2u], 1u) << "The actual physical black model must be KNOWN";
      ASSERT_EQ(r[128u + 29u], 1u);
      ASSERT_GT(r[128u + 30u], 0u) << "Closed peer must activate ordinary refinement";
      ASSERT_NE(r[7u] & 1u, 0u);
      const auto b = helper_.baseline_views();
      const auto v = helper_.mutable_views();
      const auto original = words(read(b.body.Get()));
      const auto exterior = words(read(b.mode_body.Get())); // last active unit0 Exterior
      const auto selected = words(read(v.selected_bits.Get()));
      const auto family = words(read(v.family_models.Get()));
      constexpr UINT scene = 18u * 64u + 30u, source_seed = 18u * 64u + 28u;
      constexpr UINT connected_aa = 17u * 64u + 19u, white = 16u * 64u + 21u;
      ASSERT_EQ(r[128u + 7u], 1u) << "Exactly one de-duplicated white family/mode is required";
      ASSERT_NE(family[0u] & 0x100u, 0u) << "Mode0 must be repeat-confident";
      ASSERT_GT(family[1u] & 255u, 160u) << "Gray160 must be outside the actual modal F box";
      ASSERT_EQ(original[scene], 1u) << "Scene must be accepted upstream OriginalBODY";
      ASSERT_EQ(exterior[scene], 1u) << "Scene must really connect to physical exterior";
      ASSERT_EQ(exterior[connected_aa], 1u) << "Positive cannot hide behind closure promotion";
      const auto seeds = words(emit_mode_seeds(colour_ptrs(), 0u));
      ASSERT_EQ(seeds[scene], 0u) << "Flat scene center is not an original source seed";
      ASSERT_EQ(seeds[source_seed], 3u) << "Contrast-bearing edge must be a genuine strong-P seed";
      ASSERT_EQ(selected[scene] & 1u, 1u) << "This fix must preserve original LOW geometry";
      EXPECT_EQ(selected[scene] & 0x100u, 0u) << "Source-only reach plus style is not trusted ink";
      EXPECT_NE(selected[connected_aa] & 0x100u, 0u) << "Actual connected style-modal AA must survive";
      EXPECT_NE(selected[white] & 0x100u, 0u);
      const auto kept = words(body());
      for (UINT y = 12u; y < 22u; ++y) {
        for (UINT x = 14u; x < 24u; ++x) EXPECT_EQ(kept[y * W_ + x], 1u);
        for (UINT x = 38u; x < 48u; ++x) EXPECT_EQ(kept[y * W_ + x], 1u);
      }
    }
  }

  TEST_F(HostSbsCurrentBlackEnclosureGpuTest, ClosedSupportOnlyPunctuationRetainsBodyAndMatte) {
    author(true, 20u);
    paint_native(4u, 4u, 9u, 13u, 1u);
    paint_native(5u, 5u, 8u, 12u, 255u);
    weak_exterior_scene();
    upload_canonical_origin();
    ASSERT_TRUE(run()) << error_;
    expect_ready();
    const auto r = receipt();
    ASSERT_EQ(r[128u + 29u], 1u);
    ASSERT_GT(r[128u + 3u], 0u);
    ASSERT_NE(r[7u] & 1u, 0u);
    const auto original = words(read(helper_.baseline_views().body.Get()));
    const auto selected = words(read(helper_.mutable_views().selected_bits.Get()));
    ASSERT_EQ(original[6u * W_ + 6u], 1u) << "Weak current-S component must exist upstream";
    EXPECT_EQ(selected[6u * W_ + 6u] & 255u, 1u);
    EXPECT_EQ(words(body())[6u * W_ + 6u], 1u);
    expect_baseline();
  }

  TEST_F(HostSbsCurrentBlackEnclosureGpuTest, OpenSupportOnlySceneIsRejectedWithoutDroppingStrongBody) {
    author(true, 21u);
    weak_exterior_scene();
    ASSERT_TRUE(run()) << error_;
    expect_ready();
    const auto r = receipt();
    ASSERT_EQ(r[128u + 29u], 1u);
    ASSERT_GT(r[128u + 3u], 0u);
    ASSERT_NE(r[7u] & 1u, 0u);
    const auto original = words(read(helper_.baseline_views().body.Get()));
    const auto selected = words(read(helper_.mutable_views().selected_bits.Get()));
    ASSERT_EQ(original[6u * W_ + 6u], 1u) << "S-only scene must remain present in baseline Body";
    ASSERT_EQ(original[8u * W_ + 16u], 1u);
    EXPECT_EQ(selected[6u * W_ + 6u] & 255u, 0u);
    EXPECT_EQ(selected[8u * W_ + 16u] & 255u, 1u);
    EXPECT_EQ(words(body())[6u * W_ + 6u], 0u);
    EXPECT_EQ(words(body())[8u * W_ + 16u], 1u);
    EXPECT_EQ(words(matte())[6u * W_ + 6u], 0u);
  }

  TEST_F(HostSbsCurrentBlackEnclosureGpuTest, StrongProbabilitySeedPreservesCompleteMatchingLowPEdges) {
    author(true, 22u);
    std::fill(probability_words_.begin(), probability_words_.end(), 0.0f);
    // This strip contains contrast-bearing white P anchors. The remaining fill
    // has P=0 and S=0, but shares the same four-connected strong-P component.
    probability_native_rect(14u, 6u, 18u, 8u, 1.0f);
    upload();
    ASSERT_TRUE(run()) << error_;
    expect_ready();
    std::vector<UINT> expected(W_ * H_, 0u);
    for (UINT y = 6u; y < 11u; ++y) {
      std::fill(expected.begin() + y * W_ + 10u, expected.begin() + y * W_ + 22u, 1u);
    }
    // Exact authored colour component, including every P-low edge. The separate
    // same-colour component has neither kind of seed and stays absent upstream.
    EXPECT_EQ(words(read(helper_.baseline_views().body.Get())), expected);
    EXPECT_EQ(words(body()), expected);
    expect_baseline();
  }

  TEST_F(HostSbsCurrentBlackEnclosureGpuTest, UnknownBlackModelRetainsOpenSupportOnlyBodyAndMatte) {
    author(true, 23u);
    paint_native(0u, 0u, W_, H_, 22u);
    paint_native(5u, 5u, 8u, 12u, 255u);
    paint_native(10u, 6u, 22u, 11u, 255u);
    weak_exterior_scene();
    upload_canonical_origin();
    ASSERT_TRUE(run()) << error_;
    expect_ready();
    const auto r = receipt();
    ASSERT_EQ(r[384u + 2u], 0u) << "Uniform nonliteral dark witnesses must remain UNKNOWN";
    ASSERT_EQ(r[384u + 3u], 5u);
    ASSERT_EQ(r[128u + 29u], 0u);
    const auto original = words(read(helper_.baseline_views().body.Get()));
    ASSERT_EQ(original[6u * W_ + 6u], 1u);
    ASSERT_EQ(original[8u * W_ + 16u], 1u);
    expect_baseline();
  }

  TEST_F(HostSbsCurrentBlackEnclosureGpuTest, OverlappingUnitsPreservePerUnitBitsAndBox3Union) {
    author(true, 2u, 1u, false, false, true, 64u, 32u);
    origin_pixels_.assign(W_ * H_, {32u, 32u, 32u, 255u});
    const auto paint = [&](UINT l, UINT t, UINT r, UINT b, std::uint8_t code) {
      for (UINT y = t; y < b; ++y) {
        for (UINT x = l; x < r; ++x) {
          origin_pixels_[y * W_ + x] = {code, code, code, 255u};
        }
      }
    };
    paint(22u, 10u, 33u, 23u, 1u);
    paint(24u, 12u, 31u, 21u, 255u);
    paint(42u, 10u, 55u, 23u, 1u);
    paint(44u, 12u, 53u, 21u, 255u);
    const std::array<UINT, 4u> core0 {3u, 5u, 8u, 12u}, core1 {8u, 5u, 14u, 12u};
    const std::array<UINT, 4u> cover0 {2u, 3u, 8u, 14u}, cover1 {7u, 3u, 15u, 14u};
    std::copy(core0.begin(), core0.end(), ocr_words_.begin() + 16u);
    std::copy(core1.begin(), core1.end(), ocr_words_.begin() + 24u);
    std::copy(cover0.begin(), cover0.end(), ocr_words_.begin() + 144u);
    std::copy(cover1.begin(), cover1.end(), ocr_words_.begin() + 152u);
    for (UINT y = 0; y < H_; ++y) {
      for (UINT x = 0; x < W_; ++x) {
        const auto p = origin_pixels_[y * W_ + x];
        pixels_[y * W_ + x] = {code_half(p[0]), code_half(p[1]), code_half(p[2]), 0x3c00u};
      }
    }
    upload();
    ASSERT_TRUE(run());
    expect_ready();
    const auto r = receipt();
    EXPECT_EQ(r[6] & 3u, 3u);
    EXPECT_EQ(r[7] & 3u, 3u);
    const auto selected = words(read(helper_.mutable_views().selected_bits.Get()));
    const auto covers = words(read(helper_.mutable_views().cover_bits.Get()));
    EXPECT_EQ(selected[16u * W_ + 27u] & 255u, 1u);
    EXPECT_EQ(selected[16u * W_ + 48u] & 255u, 2u);
    EXPECT_TRUE(std::any_of(covers.begin(), covers.end(), [](UINT v) {
      return (v & 3u) == 3u;
    }));
    std::vector<UINT> expected(W_ * H_), erroneous_union(W_ * H_);
    for (UINT y = 0; y < H_; ++y) {
      for (UINT x = 0; x < W_; ++x) {
        for (int dy = -3; dy <= 3; ++dy) {
          for (int dx = -3; dx <= 3; ++dx) {
            const int qx = static_cast<int>(x) + dx, qy = static_cast<int>(y) + dy;
            if (qx < 0 || qy < 0 || qx >= static_cast<int>(W_) || qy >= static_cast<int>(H_)) {
              continue;
            }
            const auto value = selected[static_cast<UINT>(qy) * W_ + static_cast<UINT>(qx)] & 255u;
            expected[y * W_ + x] |= (value & covers[y * W_ + x]) != 0u ? 1u : 0u;
            erroneous_union[y * W_ + x] |= value != 0u && covers[y * W_ + x] != 0u ? 1u : 0u;
          }
        }
      }
    }
    ASSERT_NE(expected, erroneous_union) << "Authored covers must discriminate per-unit clipping from union-cover dilation";
    EXPECT_EQ(words(matte()), expected);
  }

  TEST_F(HostSbsCurrentBlackEnclosureGpuTest, EnabledRejectsPredicationBadRootAndDeferredBeforeCommandsAndRestoresCs) {
    ASSERT_TRUE(run());
    const auto b = body(), m = matte();
    const auto r = receipt();
    D3D11_QUERY_DESC q {D3D11_QUERY_OCCLUSION_PREDICATE, 0u};
    ComPtr<ID3D11Predicate> predicate;
    require(SUCCEEDED(device_->CreatePredicate(&q, &predicate)));
    context_->SetPredication(predicate.Get(), FALSE);
    EXPECT_FALSE(run());
    context_->SetPredication(nullptr, FALSE);
    EXPECT_EQ(body(), b);
    EXPECT_EQ(matte(), m);
    EXPECT_EQ(receipt(), r);
    const auto c = colour_ptrs();
    const auto g = graph_ptrs();
    EXPECT_FALSE(helper_.dispatch(context_.Get(), c, g, cb_.resource.Get(), rgb_.srv.Get(), probability_.srv.Get(), support_.srv.Get(), ocr_.srv.Get(), root_.resource.Get(), 0u, error_, &program_));
    auto short_root = buffer(12u, false, true);
    EXPECT_FALSE(helper_.dispatch(context_.Get(), c, g, cb_.resource.Get(), rgb_.srv.Get(), probability_.srv.Get(), support_.srv.Get(), ocr_.srv.Get(), short_root.resource.Get(), 0u, error_, &program_));
    ComPtr<ID3D11DeviceContext> d;
    require(SUCCEEDED(device_->CreateDeferredContext(0u, &d)));
    EXPECT_FALSE(helper_.dispatch(d.Get(), c, g, cb_.resource.Get(), rgb_.srv.Get(), probability_.srv.Get(), support_.srv.Get(), ocr_.srv.Get(), root_.resource.Get(), 240u, error_, &program_));
    std::array<ID3D11ShaderResourceView *, 16u> s {};
    s.fill(rgb_.srv.Get());
    std::array<ID3D11Buffer *, 3u> bc {cb_.resource.Get(), cb_.resource.Get(), cb_.resource.Get()};
    std::array<texture_t, 8u> dummy;
    std::array<ComPtr<ID3D11UnorderedAccessView>, 8u> held_uav;
    std::array<ID3D11UnorderedAccessView *, 8u> expected_uav {};
    for (UINT i = 0u; i < 8u; ++i) {
      dummy[i] = texture(W_, H_, DXGI_FORMAT_R32_UINT);
      held_uav[i] = uav(dummy[i].srv.Get());
      expected_uav[i] = held_uav[i].Get();
    }
    context_->CSSetShaderResources(0u, 16u, s.data());
    context_->CSSetUnorderedAccessViews(0u, 8u, expected_uav.data(), nullptr);
    context_->CSSetConstantBuffers(0u, 3u, bc.data());
    context_->CSSetShader(colour_[0].Get(), nullptr, 0u);
    ASSERT_TRUE(run(true, false));
    std::array<ID3D11ShaderResourceView *, 16u> after {};
    context_->CSGetShaderResources(0u, 16u, after.data());
    for (UINT i = 0; i < 16u; ++i) {
      EXPECT_EQ(after[i], s[i]);
      if (after[i]) {
        after[i]->Release();
      }
    }
    std::array<ID3D11Buffer *, 3u> ab {};
    context_->CSGetConstantBuffers(0u, 3u, ab.data());
    for (UINT i = 0; i < 3u; ++i) {
      EXPECT_EQ(ab[i], bc[i]);
      if (ab[i]) {
        ab[i]->Release();
      }
    }
    std::array<ID3D11UnorderedAccessView *, 8u> actual_uav {};
    context_->CSGetUnorderedAccessViews(0u, 8u, actual_uav.data());
    for (UINT i = 0u; i < 8u; ++i) {
      EXPECT_EQ(actual_uav[i], expected_uav[i]);
      if (actual_uav[i]) {
        actual_uav[i]->Release();
      }
    }
    ID3D11ComputeShader *actual_shader = nullptr;
    UINT classes = 0u;
    context_->CSGetShader(&actual_shader, nullptr, &classes);
    EXPECT_EQ(actual_shader, colour_[0].Get());
    EXPECT_EQ(classes, 0u);
    if (actual_shader) {
      actual_shader->Release();
    }
  }

  TEST_F(HostSbsCurrentBlackEnclosureGpuTest, DenseAndSnakeCoverStressLateGraphFaultAndExtentCaps) {
    for (UINT stress : {1u, 2u}) {
      author(true, stress + 1u, 1u, false, false, false, 256u, 128u, stress);
      ASSERT_TRUE(run()) << error_;
      expect_ready();
      const auto stress_body = words(body());
      EXPECT_TRUE(std::any_of(stress_body.begin(), stress_body.end(), [](UINT v) {
        return v == 1u;
      }));
    }
    author(true, 9u);
    ASSERT_TRUE(run());
    inject_late_parent_fault();
    EXPECT_EQ(receipt()[2], 0u);
    EXPECT_NE(receipt()[3], 0u);
    expect_baseline();
    enclosure::resources_t invalid;
    EXPECT_FALSE(invalid.ensure(device_.Get(), 5121u, 16u, error_));
    EXPECT_FALSE(invalid.ensure(device_.Get(), 0u, 16u, error_));
  }

  TEST_F(HostSbsCurrentBlackEnclosureGpuTest, Saved28AuthenticFullCurrentInputsParity) {
    if (!std::getenv("SUNSHINE_TEST_BLACK_ENCLOSURE_CASES") || !std::getenv("SUNSHINE_TEST_BLACK_ENCLOSURE_OUTPUT")) {
      GTEST_SKIP() << "Explicit immutable28 cases/output required";
    }
    saved();
  }

  void HostSbsCurrentBlackEnclosureGpuTest::saved() {
    const std::filesystem::path input = std::getenv("SUNSHINE_TEST_BLACK_ENCLOSURE_CASES"), output = std::getenv("SUNSHINE_TEST_BLACK_ENCLOSURE_OUTPUT");
    const auto bytes = file_bytes(input);
    const auto spec = nlohmann::json::parse(bytes.begin(), bytes.end());
    ASSERT_EQ(spec.at("schema"), 1u);
    ASSERT_EQ(spec.at("inner_source_closure_sha256"), inner_closure_);
    ASSERT_EQ(spec.at("component_source_closure_sha256"), graph_closure_);
    ASSERT_EQ(spec.at("enclosure_source_closure_sha256"), enclosure_closure_);
    ASSERT_EQ(spec.at("driver"), hardware_ ? "hardware" : "warp");
    ASSERT_EQ(spec.at("cases").size(), 28u);
    ASSERT_FALSE(std::filesystem::exists(output));
    ASSERT_TRUE(std::filesystem::create_directories(output));
    std::array<wchar_t, 32768u> path {};
    const auto len = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
    ASSERT_GT(len, 0u);
    ASSERT_LT(len, path.size());
    const std::filesystem::path exe(std::wstring(path.data(), len));
    const auto exeBytes = file_bytes(exe);
    const nlohmann::json exeBinding {{"path", exe.string()}, {"sha256", hash_bytes(exeBytes)}, {"bytes", exeBytes.size()}};
    nlohmann::json report {{"schema", 1u}, {"driver", hardware_ ? "hardware" : "warp"}, {"adapter", adapter_}, {"test_executable", exeBinding}, {"input_specification_sha256", hash_bytes(bytes)}, {"helper_source_pin", compiled_enclosure_helper_pin}, {"baseline_helper_source_pin", compiled_baseline_helper_pin}, {"inner_source_closure_sha256", inner_closure_}, {"component_source_closure_sha256", graph_closure_}, {"enclosure_source_closure_sha256", enclosure_closure_}, {"all_semantic_status_UNKNOWN", true}, {"performance_claim", false}, {"cases", nlohmann::json::array()}};
    for (std::size_t k = 0; k < 28u; ++k) {
      const auto &item = spec.at("cases").at(k);
      SCOPED_TRACE(item.at("name").get<std::string>());
      const auto &a = item.at("artifacts");
      const auto cb = bound_bytes(a.at("constants"), 96u);
      std::memcpy(cb_words_.data(), cb.data(), 96u);
      const auto &producer = item.at("producer");
      const UINT ow = producer.at("origin_width"), oh = producer.at("origin_height");
      allocate_inputs(cb_words_[0], cb_words_[3], cb_words_[1]);
      const std::size_t n = std::size_t(W_) * H_;
      const auto source = bound_bytes(a.at("source_band"), n * 8u), p = bound_bytes(a.at("probability"), 614400u), s = bound_bytes(a.at("current_support"), n * 4u), o = bound_bytes(a.at("ocr_record"), 1024u), root = bound_bytes(a.at("root_arguments"), 256u), sct = bound_bytes(a.at("current_sct"), 128u), uip = bound_bytes(a.at("current_uip"), 160u), origin = bound_bytes(a.at("original_source_rgba8"), std::size_t(ow) * oh * 4u);
      ASSERT_EQ(producer.at("origin_format"), 28u);
      origin_ = texture(ow, oh, DXGI_FORMAT_R8G8B8A8_UNORM);
      context_->UpdateSubresource(rgb_.resource.Get(), 0u, nullptr, source.data(), W_ * 8u, 0u);
      context_->UpdateSubresource(support_.resource.Get(), 0u, nullptr, s.data(), W_ * 4u, 0u);
      context_->UpdateSubresource(probability_.resource.Get(), 0u, nullptr, p.data(), 0u, 0u);
      context_->UpdateSubresource(ocr_.resource.Get(), 0u, nullptr, o.data(), 0u, 0u);
      context_->UpdateSubresource(cb_.resource.Get(), 0u, nullptr, cb.data(), 0u, 0u);
      context_->UpdateSubresource(root_.resource.Get(), 0u, nullptr, root.data(), 0u, 0u);
      context_->UpdateSubresource(sct_.resource.Get(), 0u, nullptr, sct.data(), 0u, 0u);
      context_->UpdateSubresource(uip_.resource.Get(), 0u, nullptr, uip.data(), 0u, 0u);
      context_->UpdateSubresource(origin_.resource.Get(), 0u, nullptr, origin.data(), ow * 4u, 0u);
      program_.current_sct = sct_.srv.Get();
      program_.current_uip = uip_.srv.Get();
      auto &pr = program_.producer;
      pr.original_source = origin_.srv.Get();
      pr.transfer = static_cast<enclosure::transfer_t>(producer.at("transfer").get<UINT>());
      pr.input_known = producer.at("input_known");
      pr.direct_encoded_recipe = producer.at("direct_encoded_recipe");
      pr.postprocess_mode = producer.at("postprocess_mode");
      pr.expected_work = producer.at("expected_work");
      pr.transaction_token = producer.at("transaction_token");
      pr.model_frame = producer.at("model_frame");
      pr.model_observation_us = producer.at("model_observation_us");
      pr.analysis_generation = producer.at("analysis_generation");
      pr.analysis_roi = producer.at("analysis_roi").get<std::array<UINT, 4u>>();
      ASSERT_TRUE(run()) << error_;
      const auto v = helper_.mutable_views();
      const auto b = helper_.baseline_views();
      std::map<std::string, std::vector<char>> arrays {{"uploaded_source.rgba16f", read(rgb_.resource.Get())}, {"uploaded_probability.f32", read(probability_.resource.Get())}, {"uploaded_current_support.u32", read(support_.resource.Get())}, {"uploaded_ocr_record.u32", read(ocr_.resource.Get())}, {"uploaded_constants.u32", read(cb_.resource.Get())}, {"uploaded_root.u32", read(root_.resource.Get())}, {"uploaded_SCT.u32", read(sct_.resource.Get())}, {"uploaded_UIP.u32", read(uip_.resource.Get())}, {"uploaded_origin.rgba8", read(origin_.resource.Get())}, {"producer_context.u32", read(v.producer_constants.Get())}, {"root_mirror.u32", read(v.root_mirror.Get())}, {"baseline_body.u32", read(b.body.Get())}, {"baseline_final.u32", read(b.matte.Get())}, {"body.u32", body()}, {"final.u32", matte()}, {"receipt.u32", read(v.receipt.Get())}, {"all_body_unit_bits.u32", read(v.all_body_bits.Get())}, {"cover_unit_bits.u32", read(v.cover_bits.Get())}, {"selected_unit_bits.u32", read(v.selected_bits.Get())}, {"numeric_validation.u32", read(b.validation_receipt.Get())}, {"unit_arguments.u32", read(v.unit_arguments.Get())}};
      EXPECT_EQ(arrays.at("uploaded_source.rgba16f"), source);
      EXPECT_EQ(arrays.at("uploaded_probability.f32"), p);
      EXPECT_EQ(arrays.at("uploaded_current_support.u32"), s);
      EXPECT_EQ(arrays.at("uploaded_ocr_record.u32"), o);
      EXPECT_EQ(arrays.at("uploaded_constants.u32"), cb);
      EXPECT_EQ(arrays.at("uploaded_root.u32"), root);
      EXPECT_EQ(arrays.at("uploaded_SCT.u32"), sct);
      EXPECT_EQ(arrays.at("uploaded_UIP.u32"), uip);
      EXPECT_EQ(arrays.at("uploaded_origin.rgba8"), origin);
      EXPECT_EQ(arrays.at("root_mirror.u32"), root);
      bool expected_masks_exact = true;
      for (const auto &[name, key] : std::array<std::pair<const char *, const char *>, 4u> {{{"baseline_body.u32", "baseline_body"}, {"baseline_final.u32", "baseline_final"}, {"body.u32", "body"}, {"final.u32", "final"}}}) {
        const auto expected = bound_bytes(item.at("expected").at(key), n * 4u);
        expected_masks_exact = expected_masks_exact && arrays.at(name) == expected;
        EXPECT_EQ(arrays.at(name), expected);
      }
      const auto rc = words(arrays.at("receipt.u32"));
      ASSERT_EQ(rc.size(), 512u);
      EXPECT_EQ(rc[0], 0x31434e45u);
      EXPECT_EQ(rc[1], 5u);
      EXPECT_EQ(rc[2], 1u);
      EXPECT_EQ(rc[3], 0u);
      EXPECT_EQ(std::vector<UINT>(rc.begin() + 16u, rc.begin() + 40u), words(cb));
      bool expected_receipt_exact = false;
      if (item.at("expected").contains("receipt")) {
        const auto expected = bound_bytes(item.at("expected").at("receipt"), 2048u);
        expected_receipt_exact = arrays.at("receipt.u32") == expected;
        EXPECT_EQ(arrays.at("receipt.u32"), expected);
      }
      const auto folder = output / ("case_" + std::to_string(k + 1u));
      ASSERT_TRUE(std::filesystem::create_directory(folder));
      nlohmann::json bindings = nlohmann::json::object();
      for (const auto &[name, data] : arrays) {
        std::ofstream f(folder / name, std::ios::binary);
        f.write(data.data(), static_cast<std::streamsize>(data.size()));
        require(f.good());
        bindings[name] = {{"path", (folder / name).string()}, {"sha256", hash_bytes(data)}, {"bytes", data.size()}};
      }
      const auto dg = helper_.diagnostics();
      nlohmann::json identity {{"schema", 1u}, {"case", item}, {"driver", hardware_ ? "hardware" : "warp"}, {"adapter", adapter_}, {"test_executable", exeBinding}, {"input_specification_sha256", hash_bytes(bytes)}, {"helper_source_pin", compiled_enclosure_helper_pin}, {"baseline_helper_source_pin", compiled_baseline_helper_pin}, {"inner_source_closure_sha256", inner_closure_}, {"component_source_closure_sha256", graph_closure_}, {"enclosure_source_closure_sha256", enclosure_closure_}, {"artifacts", bindings}, {"recordings", dg.recordings}, {"replays", dg.replays}, {"direct_emissions", dg.direct_emissions}, {"expected_baseline_and_candidate_byte_exact", expected_masks_exact}, {"expected_receipt_present", item.at("expected").contains("receipt")}, {"expected_receipt_byte_exact", expected_receipt_exact}, {"all_semantic_status_UNKNOWN", true}, {"performance_claim", false}};
      identity["CBC_output_schema"] = 5u;
      identity["typed_contract"] = "CBC5_LOW8_GEOMETRY_HIGH8_TRUSTED_COLOR_OR_CLOSED_BODY";
      const auto text = identity.dump(2) + '\n';
      std::ofstream f(folder / "current_black_enclosure_identity.json", std::ios::binary);
      f << text;
      require(f.good());
      report["cases"].push_back({{"index", k + 1u}, {"name", item.at("name")}, {"identity_sha256", hash_bytes(std::vector<char>(text.begin(), text.end()))}});
    }
    report["CBC_output_schema"] = 5u;
    report["typed_contract"] = "CBC5_LOW8_GEOMETRY_HIGH8_TRUSTED_COLOR_OR_CLOSED_BODY";
    std::ofstream f(output / "current_black_enclosure_replay_identity.json", std::ios::binary);
    f << report.dump(2) << '\n';
    require(f.good());
  }
}  // namespace
#include "current_black_enclosure_cost_cases.inc"
#endif
