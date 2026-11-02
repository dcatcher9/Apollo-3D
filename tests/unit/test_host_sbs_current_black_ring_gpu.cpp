/** Private code1 enclosure mechanics and authentic saved-current native parity. No model/warp/GT. */
#include <gtest/gtest.h>
#ifdef _WIN32
  #include "src/crypto.h"
  #include "src/generated/depth_coordinate_v2_contract.h"
  #include "src/host_sbs_current_black_ring_gpu.h"
  #include "src/host_sbs_shader_cache.h"

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
  namespace ring = models::host_sbs_current_black_ring_gpu_v1;
  namespace baseline = models::host_sbs_subtitle_inner_colour_gpu_global_validation_v2;
  namespace cache = models::host_sbs_shader_cache;
  using pixel16 = std::array<std::uint16_t, 4u>;
  using pixel8 = std::array<std::uint8_t, 4u>;
  constexpr UINT probability_width = models::depth_coordinate_v2::subtitle_ocr_output_width;
  constexpr UINT probability_height = models::depth_coordinate_v2::subtitle_ocr_output_height;
  constexpr UINT probability_word_count = probability_width * probability_height;
  static_assert(probability_width == 960u && probability_height == 160u);
  constexpr const char *compiled_ring_helper_pin = "0bda8b195e38e3fc3b5712061448a981e21546838d7f4242a2b41d1d6aca811b";
  constexpr const char *compiled_enclosure_helper_pin = "7923142f4094410b4d96d56e0d964db5948d5af48361d66f9a9fb3576da4b617";
  constexpr const char *compiled_baseline_helper_pin = "5cadff739dafce03fdc0b840c42249ca490fa573b85bec276c4fea8798b6942f";

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

  class HostSbsCurrentBlackRingGpuTest: public ::testing::Test {
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
      if (const auto *hp = std::getenv("SUNSHINE_TEST_BLACK_ENCLOSURE_HEADER_PIN")) {
        require(std::string(hp) == compiled_enclosure_helper_pin);
      }
      for (std::size_t i = 0; i < 12u; ++i) {
        program_.enclosure.shaders[i] = enclosure_[i].Get();
      }
      for (std::size_t i = 0; i < 3u; ++i) {
        program_.enclosure.roi_graph[i] = roi_[i].Get();
      }

      const auto *rp = std::getenv("SUNSHINE_TEST_BLACK_RING_SHADER_PIN"), *rr = std::getenv("SUNSHINE_TEST_BLACK_RING_SHADER_ROOT");
      require(rp && rr);
      const std::array rspec {
        cache::shader_spec {"host_sbs_current_black_ring_cs.hlsl", "begin_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_black_ring_cs.hlsl", "reset_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_black_ring_cs.hlsl", "prepare_unit_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_black_ring_cs.hlsl", "emit_passable_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_black_ring_cs.hlsl", "resolve_inner_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_black_ring_cs.hlsl", "emit_black_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_black_ring_cs.hlsl", "append_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_black_ring_cs.hlsl", "publish_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_black_ring_roi_cs.hlsl", "init_roi_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_black_ring_roi_cs.hlsl", "union_roi_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_black_ring_roi_cs.hlsl", "reduce_roi_main", "cs_5_0"}
      };
      const auto rsnapshot = cache::snapshot_sources(rr, rspec);
      ring_closure_ = cache::source_closure_sha256(rsnapshot);
      require(ring_closure_ == rp);
      for (std::size_t i = 0; i < rspec.size(); ++i) {
        const auto code = cache::get(rsnapshot, rspec[i]); require(code && !code->empty());
        if (i < 8u) {
          require(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &ring_shaders_[i])));
          program_.shaders[i] = ring_shaders_[i].Get();
        } else {
          require(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &ring_roi_[i-8u])));
          program_.roi_graph[i-8u] = ring_roi_[i-8u].Get();
        }
      }
      const auto *hp = std::getenv("SUNSHINE_TEST_BLACK_RING_HEADER_PIN");
      require(hp && std::string(hp) == compiled_ring_helper_pin);
      RecordProperty("ring_source_closure", ring_closure_);
      RecordProperty("ring_helper_source_pin", compiled_ring_helper_pin);
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
      probability_ = buffer(probability_word_count * sizeof(float));
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
      probability_words_.assign(probability_word_count, 1.0f);
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
      program_.enclosure.current_sct = sct_.srv.Get();
      program_.enclosure.current_uip = uip_.srv.Get();
      program_.enclosure.producer = {};
      program_.enclosure.producer.original_source = origin_.srv.Get();
      program_.enclosure.producer.transfer = enclosure::transfer_t::encoded_sdr;
      program_.enclosure.producer.input_known = true;
      program_.enclosure.producer.direct_encoded_recipe = 1u;
      program_.enclosure.producer.postprocess_mode = 2u;
      program_.enclosure.producer.expected_work = 1u;
      program_.enclosure.producer.transaction_token = 1000u + frame;
      program_.enclosure.producer.model_frame = frame;
      program_.enclosure.producer.model_observation_us = 1000000u + frame;
      program_.enclosure.producer.analysis_generation = 7u;
      program_.enclosure.producer.analysis_roi = {0u, 0u, w, fullH_};
      root_for(2u);
      upload();
    }

    void root_for(UINT mode) {
      root_words_ = {};
      const auto t = program_.enclosure.producer.transaction_token;
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
      program_.enclosure.producer.postprocess_mode = mode;
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
      return words(read(helper_.enclosure_views().receipt.Get()));
    }

    std::vector<char> body() {
      return read(helper_.enclosure_views().body.Get());
    }

    std::vector<char> matte() {
      return read(helper_.enclosure_views().matte.Get());
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
      EXPECT_EQ(r[1], 1u);
      EXPECT_EQ(r[2], 1u);
      EXPECT_EQ(r[3], 0u);
      EXPECT_EQ(std::vector<UINT>(r.begin() + 16u, r.begin() + 40u), std::vector<UINT>(cb_words_.begin(), cb_words_.end()));
    }

    ComPtr<ID3D11UnorderedAccessView> uav(ID3D11ShaderResourceView *s) {
      ComPtr<ID3D11Resource> r;
      s->GetResource(&r);
      ComPtr<ID3D11UnorderedAccessView> u;
      require(SUCCEEDED(device_->CreateUnorderedAccessView(r.Get(), nullptr, &u)));
      return u;
    }


    std::vector<char> ring_matte() { return read(helper_.mutable_views().matte.Get()); }
    std::vector<UINT> ring_receipt() { return words(read(helper_.mutable_views().receipt.Get())); }
    void expect_ring_ready() {
      const auto r = ring_receipt(); ASSERT_EQ(r.size(), 512u);
      EXPECT_EQ(r[0], 0x31474e52u); EXPECT_EQ(r[1], 1u); EXPECT_EQ(r[2], 1u); EXPECT_EQ(r[3], 0u);
    }
    void expect_ring_parent() { EXPECT_EQ(ring_matte(), matte()); }
    void upload_pixels_from_origin() {
      for (UINT y=0u;y<H_;++y) for (UINT x=0u;x<W_;++x) {
        const auto p=origin_pixels_[(y+top_)*W_+x];
        pixels_[y*W_+x]={code_half(p[0]),code_half(p[1]),code_half(p[2]),code_half(p[3])};
      }
      upload();
    }
    void inject_late_ring_parent_fault() {
      // Fault injection after real parent extraction and both ring component graphs.
      // No source/model evidence is fabricated; rerun only actual append/publish stages.
      const auto v=helper_.mutable_views();const auto e=helper_.enclosure_views();const auto b=helper_.baseline_views();
      ComPtr<ID3D11Resource> parent_resource;b.parent->GetResource(&parent_resource);
      const UINT invalid=0xffffffffu;const D3D11_BOX box{0u,0u,0u,4u,1u,1u};
      context_->UpdateSubresource(parent_resource.Get(),0u,&box,&invalid,0u,0u);
      const std::array<UINT,4u> hook{};context_->UpdateSubresource(v.hook_constants.Get(),0u,nullptr,hook.data(),0u,0u);
      auto shape=buffer(16u,true);const std::array<UINT,4u> dims{W_,H_,W_*H_,0u};
      context_->UpdateSubresource(shape.resource.Get(),0u,nullptr,dims.data(),0u,0u);
      ID3D11Buffer *constants[3]{cb_.resource.Get(),v.hook_constants.Get(),shape.resource.Get()};
      const std::array<ID3D11ShaderResourceView*,10u> in{rgb_.srv.Get(),e.all_body_bits.Get(),e.selected_bits.Get(),e.cover_bits.Get(),e.receipt.Get(),b.eligible.Get(),b.parent.Get(),b.root_flags.Get(),v.eligible_inner.Get(),e.matte.Get()};
      std::array<ID3D11UnorderedAccessView*,8u> out{};auto meta=uav(v.receipt.Get()),m=uav(v.matte.Get());out[0]=meta.Get();out[6]=m.Get();
      context_->CSSetConstantBuffers(0u,3u,constants);context_->CSSetShaderResources(0u,10u,in.data());context_->CSSetUnorderedAccessViews(0u,8u,out.data(),nullptr);
      context_->CSSetShader(ring_shaders_[6].Get(),nullptr,0u);context_->Dispatch((W_+15u)/16u,(H_+15u)/16u,1u);
      context_->CSSetShader(ring_shaders_[7].Get(),nullptr,0u);context_->Dispatch((W_+15u)/16u,(H_+15u)/16u,1u);
      std::array<ID3D11ShaderResourceView*,16u> clear{};out.fill(nullptr);
      context_->CSSetUnorderedAccessViews(0u,8u,out.data(),nullptr);context_->CSSetShaderResources(0u,16u,clear.data());
    }

    void saved();
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ring::resources_t helper_;
    ring::program_t program_;
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
    std::string error_, inner_closure_, graph_closure_, enclosure_closure_, ring_closure_;
    std::array<ComPtr<ID3D11ComputeShader>, 8u> ring_shaders_;
    std::array<ComPtr<ID3D11ComputeShader>, 3u> ring_roi_;
    nlohmann::json adapter_;
  };


  TEST_F(HostSbsCurrentBlackRingGpuTest, NullProgramDelegatesLiteral5cad) {
    ASSERT_TRUE(run(false));
    baseline::resources_t reference; ASSERT_TRUE(reference.ensure(device_.Get(), W_, H_, error_));
    ASSERT_TRUE(reference.dispatch(context_.Get(), colour_ptrs(), graph_ptrs(), cb_.resource.Get(), rgb_.srv.Get(), probability_.srv.Get(), support_.srv.Get(), ocr_.srv.Get(), root_.resource.Get(), 240u, error_));
    EXPECT_EQ(read(helper_.baseline_views().body.Get()),read(reference.mutable_views().body.Get()));
    EXPECT_EQ(read(helper_.baseline_views().matte.Get()),read(reference.mutable_views().matte.Get()));
  }
  TEST_F(HostSbsCurrentBlackRingGpuTest, ClosedCode01RawRingAndDirectRecordedMatch) {
    for (UINT code : {0u,1u}) {
      author(true,code+2u,code,false,false,false,64u,32u);
      ASSERT_TRUE(run(true,false)) << error_; expect_ring_ready();
      const auto b=body(),m=ring_matte(),r=raw(ring_receipt());
      const auto old=words(matte()),now=words(m);
      bool added=false;
      for (std::size_t i=0;i<now.size();++i) {
        EXPECT_LE(old[i],now[i]);
        if(now[i]&&!old[i]) {added=true;EXPECT_LE(std::max({origin_pixels_[i][0],origin_pixels_[i][1],origin_pixels_[i][2]}),1u);}
      }
      EXPECT_TRUE(added);
      ASSERT_TRUE(run(true,true)); EXPECT_EQ(body(),b);EXPECT_EQ(ring_matte(),m);EXPECT_EQ(raw(ring_receipt()),r);
      const auto record=helper_.diagnostics().recordings;
      ASSERT_TRUE(run()); EXPECT_EQ(helper_.diagnostics().recordings,record);
    }
  }
  TEST_F(HostSbsCurrentBlackRingGpuTest, RootZeroHoldsAllOutputsAndFreshEmptyClearsImmediately) {
    author(true,2u,1u,false,false,false,64u,32u);ASSERT_TRUE(run());expect_ring_ready();
    const auto b=body(),m=matte(),n=ring_matte();
    const auto r=receipt(),rr=ring_receipt();
    author(false,3u,1u,false,false,false,64u,32u);root_words_[60]=0u;upload();
    ASSERT_TRUE(run());EXPECT_EQ(body(),b);EXPECT_EQ(matte(),m);EXPECT_EQ(ring_matte(),n);EXPECT_EQ(receipt(),r);EXPECT_EQ(ring_receipt(),rr);
    root_words_[60]=1u;upload();ASSERT_TRUE(run());expect_ring_ready();
    EXPECT_EQ(body(),std::vector<char>(W_*H_*4u,0));EXPECT_EQ(matte(),std::vector<char>(W_*H_*4u,0));EXPECT_EQ(ring_matte(),std::vector<char>(W_*H_*4u,0));
  }
  TEST_F(HostSbsCurrentBlackRingGpuTest, OpenPartialAndUnsupportedTransferKeepParentMatte) {
    author(true,2u,1u,true,false,false,64u,32u);ASSERT_TRUE(run());expect_ring_ready();expect_ring_parent();
    author(true,3u,1u,false,true,false,64u,32u);ASSERT_TRUE(run());expect_ring_ready();expect_ring_parent();
    author(true,4u,1u,false,false,false,64u,32u);program_.enclosure.producer.transfer=enclosure::transfer_t::linear;
    ASSERT_TRUE(run());expect_ring_ready();expect_ring_parent();
    program_.enclosure.producer.transfer=enclosure::transfer_t::encoded_sdr;
    program_.enclosure.producer.original_source=probability_.srv.Get();ASSERT_TRUE(run());expect_ring_ready();expect_ring_parent();
  }
  TEST_F(HostSbsCurrentBlackRingGpuTest, CurrentTypedAndNumericFaultFallsBackWithoutStaleRing) {
    author(true,2u,1u,false,false,false,64u,32u);ASSERT_TRUE(run());expect_ring_ready();
    uip_words_[4]^=1u;upload();ASSERT_TRUE(run());EXPECT_EQ(ring_receipt()[2],0u);expect_ring_parent();
    uip_words_[4]^=1u;probability_words_.back()=std::bit_cast<float>(0x7fc00000u);upload();
    ASSERT_TRUE(run());EXPECT_EQ(ring_receipt()[2],0u);expect_ring_parent();
    probability_words_.back()=1.0f;upload();ASSERT_TRUE(run());expect_ring_ready();
  }

  TEST_F(HostSbsCurrentBlackRingGpuTest, TrueCoverBorderBlackAttachmentRejectedWholeComponent) {
    author(true,2u,1u,false,false,false,64u,32u);
    for(UINT x=48u;x<56u;++x) origin_pixels_[16u*W_+x]={1u,1u,1u,255u};
    upload_pixels_from_origin();ASSERT_TRUE(run());expect_ring_ready();expect_ring_parent();
  }

  TEST_F(HostSbsCurrentBlackRingGpuTest, ExpectedLimitationBoundedConnectedSceneBlackBranchIncluded) {
    author(true,2u,1u,false,false,false,64u,32u);
    for(UINT x=48u;x<52u;++x) origin_pixels_[16u*W_+x]={1u,1u,1u,255u};
    upload_pixels_from_origin();ASSERT_TRUE(run());expect_ring_ready();
    const auto old=words(matte()),now=words(ring_matte());
    EXPECT_EQ(old[16u*W_+51u],0u);EXPECT_EQ(now[16u*W_+51u],1u);
    // Port-conformance counterexample, never a claim that arbitrary scene-black is subtitle.
  }

  TEST_F(HostSbsCurrentBlackRingGpuTest, DenseSnakeCoverAndInvalidExtentStayBounded) {
    for(UINT stress : {1u,2u}) {
      author(true,stress+2u,1u,false,false,false,256u,128u,stress);
      ASSERT_TRUE(run()) << error_;expect_ring_ready();
      const auto old=words(matte()),now=words(ring_matte());
      ASSERT_EQ(old.size(),now.size());
      for(std::size_t i=0;i<now.size();++i) {EXPECT_LE(old[i],now[i]);EXPECT_LE(now[i],1u);}
    }
    ring::resources_t invalid;
    EXPECT_FALSE(invalid.ensure(device_.Get(),0u,16u,error_));
    EXPECT_FALSE(invalid.ensure(device_.Get(),5121u,16u,error_));
  }

  TEST_F(HostSbsCurrentBlackRingGpuTest, LateActualGraphFaultDiscardsAllAddedRingPixels) {
    author(true,2u,1u,false,false,false,64u,32u);ASSERT_TRUE(run());expect_ring_ready();
    const auto before=ring_receipt();ASSERT_GT(before[7],0u);
    const auto b=body();inject_late_ring_parent_fault();
    const auto after=ring_receipt();EXPECT_EQ(after[2],0u);EXPECT_NE(after[3]&4u,0u);EXPECT_EQ(after[7],0u);
    EXPECT_EQ(body(),b);expect_ring_parent();
  }

  TEST_F(HostSbsCurrentBlackRingGpuTest, EnabledRejectsPredicationBadRootAndDeferredBeforeCommandsAndRestoresCs) {
    ASSERT_TRUE(run());
    const auto b = body(), m = matte();
    const auto r = receipt();
    const auto old_ring_matte = ring_matte();
    const auto old_ring_receipt = ring_receipt();
    D3D11_QUERY_DESC q {D3D11_QUERY_OCCLUSION_PREDICATE, 0u};
    ComPtr<ID3D11Predicate> predicate;
    require(SUCCEEDED(device_->CreatePredicate(&q, &predicate)));
    context_->SetPredication(predicate.Get(), FALSE);
    EXPECT_FALSE(run());
    context_->SetPredication(nullptr, FALSE);
    EXPECT_EQ(body(), b);
    EXPECT_EQ(matte(), m);
    EXPECT_EQ(receipt(), r);
    EXPECT_EQ(ring_matte(), old_ring_matte);
    EXPECT_EQ(ring_receipt(), old_ring_receipt);
    const auto c = colour_ptrs();
    const auto g = graph_ptrs();
    EXPECT_FALSE(helper_.dispatch(context_.Get(), c, g, cb_.resource.Get(), rgb_.srv.Get(), probability_.srv.Get(), support_.srv.Get(), ocr_.srv.Get(), root_.resource.Get(), 0u, error_, &program_));
    auto short_root = buffer(12u, false, true);
    EXPECT_FALSE(helper_.dispatch(context_.Get(), c, g, cb_.resource.Get(), rgb_.srv.Get(), probability_.srv.Get(), support_.srv.Get(), ocr_.srv.Get(), short_root.resource.Get(), 0u, error_, &program_));
    ComPtr<ID3D11DeviceContext> d;
    require(SUCCEEDED(device_->CreateDeferredContext(0u, &d)));
    EXPECT_FALSE(helper_.dispatch(d.Get(), c, g, cb_.resource.Get(), rgb_.srv.Get(), probability_.srv.Get(), support_.srv.Get(), ocr_.srv.Get(), root_.resource.Get(), 240u, error_, &program_));
    EXPECT_EQ(ring_matte(), old_ring_matte);
    EXPECT_EQ(ring_receipt(), old_ring_receipt);
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

  TEST_F(HostSbsCurrentBlackRingGpuTest, Saved28AuthenticFullCurrentInputsParity) {
    if (!std::getenv("SUNSHINE_TEST_BLACK_RING_CASES") || !std::getenv("SUNSHINE_TEST_BLACK_RING_OUTPUT")) {
      GTEST_SKIP() << "Explicit immutable28 cases/output required";
    }
    saved();
  }

  void HostSbsCurrentBlackRingGpuTest::saved() {
    const std::filesystem::path input = std::getenv("SUNSHINE_TEST_BLACK_RING_CASES"), output = std::getenv("SUNSHINE_TEST_BLACK_RING_OUTPUT");
    const auto bytes = file_bytes(input);
    const auto spec = nlohmann::json::parse(bytes.begin(), bytes.end());
    ASSERT_EQ(spec.at("schema"), 1u);
    ASSERT_EQ(spec.at("inner_source_closure_sha256"), inner_closure_);
    ASSERT_EQ(spec.at("component_source_closure_sha256"), graph_closure_);
    ASSERT_EQ(spec.at("enclosure_source_closure_sha256"), enclosure_closure_);
    ASSERT_EQ(spec.at("driver"), hardware_ ? "hardware" : "warp");
    ASSERT_EQ(spec.at("ring_source_closure_sha256"), ring_closure_);
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
    nlohmann::json report {{"schema", 1u}, {"driver", hardware_ ? "hardware" : "warp"}, {"adapter", adapter_}, {"probability_shape", {probability_width, probability_height}}, {"test_executable", exeBinding}, {"input_specification_sha256", hash_bytes(bytes)}, {"helper_source_pin", compiled_enclosure_helper_pin}, {"baseline_helper_source_pin", compiled_baseline_helper_pin}, {"inner_source_closure_sha256", inner_closure_}, {"component_source_closure_sha256", graph_closure_}, {"enclosure_source_closure_sha256", enclosure_closure_}, {"ring_helper_source_pin", compiled_ring_helper_pin}, {"ring_source_closure_sha256", ring_closure_}, {"all_semantic_status_UNKNOWN", true}, {"performance_claim", false}, {"cases", nlohmann::json::array()}};
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
      const auto source = bound_bytes(a.at("source_band"), n * 8u), p = bound_bytes(a.at("probability"), probability_word_count * sizeof(float)), s = bound_bytes(a.at("current_support"), n * 4u), o = bound_bytes(a.at("ocr_record"), 1024u), root = bound_bytes(a.at("root_arguments"), 256u), sct = bound_bytes(a.at("current_sct"), 128u), uip = bound_bytes(a.at("current_uip"), 160u), origin = bound_bytes(a.at("original_source_rgba8"), std::size_t(ow) * oh * 4u);
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
      program_.enclosure.current_sct = sct_.srv.Get();
      program_.enclosure.current_uip = uip_.srv.Get();
      auto &pr = program_.enclosure.producer;
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
      const auto v = helper_.enclosure_views();
      const auto b = helper_.baseline_views();
      std::map<std::string, std::vector<char>> arrays {{"uploaded_source.rgba16f", read(rgb_.resource.Get())}, {"uploaded_probability.f32", read(probability_.resource.Get())}, {"uploaded_current_support.u32", read(support_.resource.Get())}, {"uploaded_ocr_record.u32", read(ocr_.resource.Get())}, {"uploaded_constants.u32", read(cb_.resource.Get())}, {"uploaded_root.u32", read(root_.resource.Get())}, {"uploaded_SCT.u32", read(sct_.resource.Get())}, {"uploaded_UIP.u32", read(uip_.resource.Get())}, {"uploaded_origin.rgba8", read(origin_.resource.Get())}, {"producer_context.u32", read(v.producer_constants.Get())}, {"root_mirror.u32", read(v.root_mirror.Get())}, {"baseline_body.u32", read(b.body.Get())}, {"baseline_final.u32", read(b.matte.Get())}, {"body.u32", body()}, {"final.u32", matte()}, {"receipt.u32", read(v.receipt.Get())}, {"all_body_unit_bits.u32", read(v.all_body_bits.Get())}, {"cover_unit_bits.u32", read(v.cover_bits.Get())}, {"selected_unit_bits.u32", read(v.selected_bits.Get())}, {"numeric_validation.u32", read(b.validation_receipt.Get())}, {"unit_arguments.u32", read(v.unit_arguments.Get())}};
      const auto rv = helper_.mutable_views();
      arrays.emplace("ring_final.u32", ring_matte());
      arrays.emplace("ring_receipt.u32", read(rv.receipt.Get()));
      arrays.emplace("ring_eligible_inner.u32", read(rv.eligible_inner.Get()));
      arrays.emplace("ring_unit_arguments.u32", read(rv.unit_arguments.Get()));
      arrays.emplace("ring_native_arguments.u32", read(rv.native_arguments.Get()));
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
      const auto expected_ring = bound_bytes(item.at("expected_ring").at("final"), n * 4u);
      const bool expected_ring_exact = arrays.at("ring_final.u32") == expected_ring;
      EXPECT_EQ(arrays.at("ring_final.u32"), expected_ring);
      EXPECT_EQ(arrays.at("body.u32"), bound_bytes(item.at("expected_ring").at("body"), n * 4u));
      const auto rr = words(arrays.at("ring_receipt.u32"));
      ASSERT_EQ(rr.size(), 512u); EXPECT_EQ(rr[0], 0x31474e52u); EXPECT_EQ(rr[1], 1u); EXPECT_EQ(rr[2], 1u); EXPECT_EQ(rr[3], 0u);
      const auto rc = words(arrays.at("receipt.u32"));
      ASSERT_EQ(rc.size(), 512u);
      EXPECT_EQ(rc[0], 0x31434e45u);
      EXPECT_EQ(rc[1], 1u);
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
      const auto dg = helper_.parent_diagnostics();
      const auto rd = helper_.diagnostics();
      nlohmann::json identity {{"schema", 1u}, {"case", item}, {"driver", hardware_ ? "hardware" : "warp"}, {"adapter", adapter_}, {"probability_shape", {probability_width, probability_height}}, {"test_executable", exeBinding}, {"input_specification_sha256", hash_bytes(bytes)}, {"helper_source_pin", compiled_enclosure_helper_pin}, {"baseline_helper_source_pin", compiled_baseline_helper_pin}, {"inner_source_closure_sha256", inner_closure_}, {"component_source_closure_sha256", graph_closure_}, {"enclosure_source_closure_sha256", enclosure_closure_}, {"artifacts", bindings}, {"expected_ring_byte_exact", expected_ring_exact}, {"ring_recordings", rd.recordings}, {"ring_replays", rd.replays}, {"ring_direct_emissions", rd.direct_emissions}, {"recordings", dg.recordings}, {"replays", dg.replays}, {"direct_emissions", dg.direct_emissions}, {"expected_baseline_and_candidate_byte_exact", expected_masks_exact}, {"expected_receipt_present", item.at("expected").contains("receipt")}, {"expected_receipt_byte_exact", expected_receipt_exact}, {"ring_helper_source_pin", compiled_ring_helper_pin}, {"ring_source_closure_sha256", ring_closure_}, {"all_semantic_status_UNKNOWN", true}, {"performance_claim", false}};
      const auto text = identity.dump(2) + '\n';
      std::ofstream f(folder / "current_black_ring_identity.json", std::ios::binary);
      f << text;
      require(f.good());
      report["cases"].push_back({{"index", k + 1u}, {"name", item.at("name")}, {"identity_sha256", hash_bytes(std::vector<char>(text.begin(), text.end()))}});
    }
    std::ofstream f(output / "current_black_ring_replay_identity.json", std::ios::binary);
    f << report.dump(2) << '\n';
    require(f.good());
  }
}  // namespace
#endif
