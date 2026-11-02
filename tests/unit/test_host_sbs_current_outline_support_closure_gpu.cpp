/** Private f2a9 outline-support closure port. Private source fixture; no runtime activation or GT inputs. */
#include <gtest/gtest.h>
#ifdef _WIN32
  #include "src/crypto.h"
  #include "src/generated/depth_coordinate_v2_contract.h"
  #include "src/host_sbs_current_outline_support_closure_gpu.h"
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

// The scalar contract is compiled from the same source by C++ and HLSL.
// Keep model confidence and width calibration out of the adoption API.
namespace cbc_unit_state_contract_test {
  using uint = std::uint32_t;
  #include "src_assets/windows/assets/shaders/directx/current_outline/outline/cbc_unit_state_shared.hlsl"
}

TEST(CurrentCbcUnitStateContract, PhysicalRefinementAdoptionTruthTable) {
  constexpr std::array<std::uint32_t, 4u> values {0u, 1u, 2u, 0xffffffffu};
  constexpr std::array<std::uint32_t, 7u> statuses {0u, 1u, 2u, 3u, 4u, 5u, 0xffffffffu};
  for (const auto evaluated : values) for (const auto observed : values)
    for (const auto enclosed : values) for (const auto status : statuses)
      for (const auto ready : values) {
        const bool original_raw_guard = !(evaluated == 0u || observed != 1u ||
            enclosed == 0u || status != 1u || ready != 1u);
        const bool original_band_guard = evaluated != 0u && observed == 1u &&
            enclosed > 0u && status == 1u && ready == 1u;
        const bool adopted = cbc_unit_state_contract_test::CurrentCBCPhysicalRefinementAdopted(
            evaluated, observed, enclosed, status, ready);
        EXPECT_EQ(adopted, original_raw_guard);
        EXPECT_EQ(adopted, original_band_guard);
      }
}

namespace {
  using Microsoft::WRL::ComPtr;
  namespace enclosure = models::host_sbs_current_black_enclosure_gpu_v1;
  namespace ring = models::host_sbs_current_outline_support_closure_gpu_v2;
  namespace baseline = models::host_sbs_subtitle_inner_colour_gpu_global_validation_v2;
  namespace cache = models::host_sbs_shader_cache;
  using pixel16 = std::array<std::uint16_t, 4u>;
  using pixel8 = std::array<std::uint8_t, 4u>;
  constexpr UINT probability_width = models::depth_coordinate_v2::subtitle_ocr_output_width;
  constexpr UINT probability_height = models::depth_coordinate_v2::subtitle_ocr_output_height;
  constexpr UINT probability_word_count = probability_width * probability_height;
  static_assert(probability_width == 960u && probability_height == 160u);
  constexpr const char *compiled_ring_helper_pin = subtitle_test_source_pins::current_outline_support;
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

  class HostSbsCurrentOutlineSupportClosureGpuTest: public ::testing::Test {
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
      const auto *roi_root = std::getenv("SUNSHINE_TEST_CURRENT_ROI_SHADER_ROOT"), *roi_pin = std::getenv("SUNSHINE_TEST_CURRENT_ROI_SHADER_PIN");require(roi_root&&roi_pin);
      const std::array roi_specs {cache::shader_spec{"host_sbs_current_components_roi_cs.hlsl","init_roi_main","cs_5_0"},cache::shader_spec{"host_sbs_current_components_roi_cs.hlsl","union_roi_main","cs_5_0"},cache::shader_spec{"host_sbs_current_components_roi_cs.hlsl","reduce_roi_main","cs_5_0"}};
      load_shaders(roi_root,roi_pin,roi_specs,roi_);ring_roi_=roi_;for(UINT k=0u;k<3u;++k)program_.roi_graph[k]=roi_[k].Get();
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

      const auto *rp = std::getenv("SUNSHINE_TEST_OUTLINE_CLOSURE_SHADER_PIN"), *rr = std::getenv("SUNSHINE_TEST_OUTLINE_CLOSURE_SHADER_ROOT");
      require(rp && rr);
      const std::array rspec {
        cache::shader_spec {"host_sbs_current_outline_support_raw_cs.hlsl", "begin_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_outline_support_raw_cs.hlsl", "reset_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_outline_support_raw_cs.hlsl", "prepare_unit_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_outline_support_raw_cs.hlsl", "emit_passable_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_outline_support_raw_cs.hlsl", "resolve_inner_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_outline_support_raw_cs.hlsl", "emit_black_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_outline_support_raw_cs.hlsl", "append_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_outline_support_raw_cs.hlsl", "publish_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_outline_support_band_cs.hlsl", "prepare_profiles_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_outline_support_band_cs.hlsl", "profile_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_outline_support_band_cs.hlsl", "calibrate_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_outline_support_band_cs.hlsl", "grow_tile_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_outline_support_band_cs.hlsl", "publish_final_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_outline_support_band_cs.hlsl", "aggregate_support_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_outline_support_band_cs.hlsl", "publish_closure_main", "cs_5_0"}
      };
      const auto rsnapshot = cache::snapshot_sources(rr, rspec);
      ring_closure_ = cache::source_closure_sha256(rsnapshot);
      require(ring_closure_ == rp);
      for (std::size_t i = 0; i < rspec.size(); ++i) {
        const auto code = cache::get(rsnapshot, rspec[i]);
        require(code && !code->empty());
        if (i < 8u) {
          require(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &ring_shaders_[i])));
          program_.shaders[i] = ring_shaders_[i].Get();
        } else {
          require(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &band_shaders_[i - 8u])));
          program_.band_shaders[i - 8u] = band_shaders_[i - 8u].Get();
        }
      }
      const auto *inside_root = std::getenv("SUNSHINE_TEST_INSIDE_COMPLETION_SHADER_ROOT");
      const auto *inside_pin = std::getenv("SUNSHINE_TEST_INSIDE_COMPLETION_SHADER_PIN");
      require(inside_root && inside_pin);
      const std::array ispec {
        cache::shader_spec {"host_sbs_current_inside_completion_cs.hlsl", "prepare_unit_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_inside_completion_cs.hlsl", "emit_init_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_inside_completion_cs.hlsl", "publish_inside_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_current_inside_completion_cs.hlsl", "finalize_main", "cs_5_0"}
      };
      load_shaders(inside_root, inside_pin, ispec, inside_shaders_);
      inside_closure_ = inside_pin;
      for (UINT i = 0u; i < 4u; ++i) program_.inside_shaders[i] = inside_shaders_[i].Get();
      RecordProperty("inside_source_closure", inside_closure_);
      const auto *hp = std::getenv("SUNSHINE_TEST_OUTLINE_CLOSURE_HEADER_PIN");
      require(hp && std::string(hp) == compiled_ring_helper_pin);
      RecordProperty("outline_source_closure", ring_closure_);
      RecordProperty("outline_helper_source_pin", compiled_ring_helper_pin);
      RecordProperty("outline_calibration_tag", ring::calibration_tag);
      RecordProperty("outline_calibration_schema", ring::calibration_schema);
      RecordProperty("outline_profile_measurement", ring::profile_measurement);
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
      packed_snapshot_ = texture(w, h, DXGI_FORMAT_R32_UINT);
      cache_snapshot_ = texture(w, h, DXGI_FORMAT_R32_UINT);
      pre_inside_snapshot_ = texture(w, h, DXGI_FORMAT_R32_UINT);
      raw_parent_snapshot_ = buffer((w * h + 1u) * 4u);
      raw_flags_snapshot_ = buffer((w * h + 1u) * 4u);
      program_.pre_inside_final_snapshot = pre_inside_snapshot_.resource.Get();
      program_.raw_parent_snapshot = raw_parent_snapshot_.resource.Get();
      program_.raw_flags_snapshot = raw_flags_snapshot_.resource.Get();
      program_.packed_temp_snapshot = packed_snapshot_.resource.Get();
      program_.passable_cache_snapshot = cache_snapshot_.resource.Get();
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

    void rect(UINT left, UINT top, UINT right, UINT bottom, pixel8 colour) {
      require(left < right && top < bottom && right <= W_ && bottom <= H_);
      for (UINT y = top; y < bottom; ++y) {
        for (UINT x = left; x < right; ++x) {
          origin_pixels_[(y + top_) * W_ + x] = colour;
        }
      }
    }

    // Width is present only in CURRENT source raster. There is no width/hist GPU input.
    void author_band(UINT thickness, UINT frame, bool target_black_background = false) {
      author(true, frame, 0u, false, false, false, 256u, 128u, 3u);
      rect(0u, 0u, W_, H_, {128u, 128u, 128u, 255u});
      if (target_black_background) {
        rect(32u, 24u, 127u, 112u, {0u, 0u, 0u, 255u});
      } else {
        rect(79u - thickness, 48u - thickness, 111u + thickness, 88u + thickness, {0u, 0u, 0u, 255u});
      }
      rect(79u, 48u, 111u, 88u, {255u, 255u, 255u, 255u});
      rect(160u - thickness, 48u - thickness, 208u + thickness, 88u + thickness, {0u, 0u, 0u, 255u});
      rect(160u, 48u, 208u, 88u, {255u, 255u, 255u, 255u});
      upload_pixels_from_origin();
    }

    void expect_outline_accounting() {
      const auto r = words(ring_matte()), v = words(outline_matte()), b = words(body());
      const auto code1 = words(matte()), packed = words(read(packed_snapshot_.resource.Get()));
      const auto d = calibration();
      ASSERT_EQ(r.size(), v.size());
      ASSERT_EQ(b.size(), v.size());
      ASSERT_EQ(code1.size(), v.size());
      ASSERT_EQ(packed.size(), v.size());
      ASSERT_EQ(d.size(), ring::data_word_count);
      for (std::size_t i = 0; i < v.size(); ++i) {
        EXPECT_LE(r[i], 1u);
        EXPECT_LE(v[i], 1u);
        EXPECT_LE(b[i], 1u);
        EXPECT_LE(code1[i], 1u);
      }
      if (d[2u] == 0u || d[3u] != 0u) {
        EXPECT_EQ(v, r);  // Invalid publication still requires literal Raw fallback.
        EXPECT_EQ(d[7u], 0u);
        return;
      }
      ASSERT_EQ(d[2u], 1u);
      EXPECT_LE(d[83u], 1u);  // Inside may retain the pre-Inside result on unavailable evidence.
      std::size_t raw_added = 0u, raw_removed = 0u, reference_added = 0u, code1_removed = 0u;
      for (std::size_t i = 0; i < v.size(); ++i) {
        const bool reference = r[i] != 0u || (packed[i] & (255u | 256u)) != 0u;
        raw_added += v[i] != 0u && r[i] == 0u;
        raw_removed += r[i] != 0u && v[i] == 0u;
        reference_added += v[i] != 0u && !reference;
        code1_removed += code1[i] != 0u && v[i] == 0u;
      }
      // CFB7 permits current evidence to prune geometric Raw/LOW8 candidates.
      // Every final addition/removal must remain exactly accounted after Inside.
      EXPECT_EQ(d[7u], raw_added);
      EXPECT_EQ(d[73u], raw_removed);
      EXPECT_EQ(d[70u], reference_added);
      EXPECT_EQ(d[76u], code1_removed);
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
      EXPECT_EQ(r[1], 5u);
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

    std::vector<char> ring_matte() {
      return read(helper_.raw_views().matte.Get());
    }

    std::vector<UINT> ring_receipt() {
      return words(read(helper_.raw_views().receipt.Get()));
    }

    void expect_ring_ready() {
      const auto r = ring_receipt();
      ASSERT_EQ(r.size(), 512u);
      EXPECT_EQ(r[0], 0x31474e52u);
      EXPECT_EQ(r[1], 1u);
      EXPECT_EQ(r[2], 1u);
      EXPECT_EQ(r[3], 0u);
    }

    void expect_ring_parent() {
      EXPECT_EQ(ring_matte(), matte());
    }

    void upload_pixels_from_origin() {
      for (UINT y = 0u; y < H_; ++y) {
        for (UINT x = 0u; x < W_; ++x) {
          const auto p = origin_pixels_[(y + top_) * W_ + x];
          pixels_[y * W_ + x] = {code_half(p[0]), code_half(p[1]), code_half(p[2]), code_half(p[3])};
        }
      }
      upload();
    }

    std::vector<UINT> calibrate_width_histogram(const std::vector<UINT> &modes, bool source_mismatch = false) {
      // Kernel-level regression: receipts/CB come from run(); calibration inputs are controlled.
      auto data = calibration();
      require(data.size() == 68608u && modes.size() <= 64u);
      require(data[0u] == 0x37424643u && data[1u] == 7u && data[2u] == 1u && data[3u] == 0u);
      require(std::equal(cb_words_.begin(), cb_words_.end(), data.begin() + 8u));
      constexpr UINT unit = 128u;
      std::fill(data.begin() + 1024u, data.end(), 0u);
      std::fill(data.begin() + unit + 6u, data.begin() + unit + 25u, 0u);
      data[3u] = 0u;data[5u] &= ~1u;data[6u] &= ~1u;
      data[unit + 6u] = 64u;data[unit + 7u] = static_cast<UINT>(modes.size());
      data[unit + 13u] = 11u;data[unit + 14u] = 1u;
      for (UINT band = 0u; band < modes.size(); ++band) {
        require(modes[band] < 64u);
        if (modes[band] != 0u) {
          data[1024u + band * 128u + modes[band]] = 10u;
          data[unit + 8u] += 10u;
        }
      }
      if (source_mismatch) data[8u] ^= 1u;
      const auto v = helper_.mutable_views();
      ComPtr<ID3D11Resource> resource;
      v.calibration_data->GetResource(&resource);
      context_->UpdateSubresource(resource.Get(), 0u, nullptr, data.data(), 0u, 0u);
      D3D11_UNORDERED_ACCESS_VIEW_DESC desc {};
      desc.Format = DXGI_FORMAT_R32_TYPELESS;desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
      desc.Buffer.NumElements = 68608u;desc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
      ComPtr<ID3D11UnorderedAccessView> output;
      require(SUCCEEDED(device_->CreateUnorderedAccessView(resource.Get(), &desc, &output)));
      const std::array<UINT, 4u> hook {}, dims {W_, H_, W_ * H_, 0u};
      context_->UpdateSubresource(v.hook_constants.Get(), 0u, nullptr, hook.data(), 0u, 0u);
      auto shape = buffer(16u, true);
      context_->UpdateSubresource(shape.resource.Get(), 0u, nullptr, dims.data(), 0u, 0u);
      ID3D11Buffer *constants[3] {cb_.resource.Get(), v.hook_constants.Get(), shape.resource.Get()};
      std::array<ID3D11ShaderResourceView *, 12u> inputs {};
      inputs[5u] = helper_.enclosure_views().receipt.Get();inputs[11u] = helper_.raw_views().receipt.Get();
      ID3D11UnorderedAccessView *target = output.Get();
      context_->CSSetConstantBuffers(0u, 3u, constants);
      context_->CSSetShaderResources(0u, 12u, inputs.data());
      context_->CSSetUnorderedAccessViews(0u, 1u, &target, nullptr);
      context_->CSSetShader(program_.band_shaders[2u], nullptr, 0u);
      context_->Dispatch(1u, 1u, 1u);
      target = nullptr;inputs.fill(nullptr);
      context_->CSSetUnorderedAccessViews(0u, 1u, &target, nullptr);
      context_->CSSetShaderResources(0u, 12u, inputs.data());
      return calibration();
    }

    void inject_late_actual_fault(bool graph_shape) {
      // Explicit fault shim after real current-source GPU inference. It never supplies
      // Body, support, histogram or t; the actual aggregate/publish shaders decide fallback.
      const auto v = helper_.mutable_views();
      const auto e = helper_.enclosure_views();
      const auto rv = helper_.raw_views();
      const auto b = helper_.baseline_views();
      auto short_parent = buffer(4u);
      if (!graph_shape) {
        const auto current = root_cache();
        const auto i = std::find_if(current.begin(), current.end(), [](UINT x) {
          return (x & 0x80000000u) != 0u;
        });
        require(i != current.end());
        const UINT root = *i & 0x3fffffffu;
        require(root > 0u && root <= W_ * H_);
        ComPtr<ID3D11Resource> cache;
        v.passable_root_cache->GetResource(&cache);
        const UINT x = (root - 1u) % W_, y = (root - 1u) / W_;
        const UINT invalid = 0xbfffffffu;
        const D3D11_BOX box {x, y, 0u, x + 1u, y + 1u, 1u};
        context_->UpdateSubresource(cache.Get(), 0u, &box, &invalid, 4u, 0u);
      }
      const std::array<UINT, 4u> hook {};
      context_->UpdateSubresource(v.hook_constants.Get(), 0u, nullptr, hook.data(), 0u, 0u);
      auto shape = buffer(16u, true);
      const std::array<UINT, 4u> dims {W_, H_, W_ * H_, 0u};
      context_->UpdateSubresource(shape.resource.Get(), 0u, nullptr, dims.data(), 0u, 0u);
      ID3D11Buffer *constants[3] {cb_.resource.Get(), v.hook_constants.Get(), shape.resource.Get()};
      const std::array<ID3D11ShaderResourceView *, 16u> in {rgb_.srv.Get(), ocr_.srv.Get(), e.selected_bits.Get(), e.body.Get(), e.cover_bits.Get(), e.receipt.Get(), graph_shape ? short_parent.srv.Get() : b.parent.Get(), b.root_flags.Get(), b.eligible.Get(), rv.eligible_inner.Get(), rv.matte.Get(), rv.receipt.Get(), e.matte.Get(), nullptr, nullptr, v.inside_bits.Get()};
      ComPtr<ID3D11Resource> data_resource;
      v.calibration_data->GetResource(&data_resource);
      D3D11_UNORDERED_ACCESS_VIEW_DESC dd {};
      dd.Format = DXGI_FORMAT_R32_TYPELESS;
      dd.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
      dd.Buffer.NumElements = 68608u;
      dd.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
      ComPtr<ID3D11UnorderedAccessView> data;
      require(SUCCEEDED(device_->CreateUnorderedAccessView(data_resource.Get(), &dd, &data)));
      auto m = uav(v.matte.Get()), cache = uav(v.passable_root_cache.Get());
      std::array<ID3D11UnorderedAccessView *, 8u> out {};
      out[0] = data.Get();
      out[1] = m.Get();
      out[2] = cache.Get();
      context_->CSSetConstantBuffers(0u, 3u, constants);
      context_->CSSetShaderResources(0u, 16u, in.data());
      context_->CSSetUnorderedAccessViews(0u, 8u, out.data(), nullptr);
      // Current source CBC unit0 cover is [32,24,224,112] for author_band.
      context_->CSSetShader(band_shaders_[5].Get(), nullptr, 0u);
      context_->Dispatch(12u, 6u, 1u);
      context_->CSSetShader(band_shaders_[4].Get(), nullptr, 0u);
      context_->Dispatch((W_ + 15u) / 16u, (H_ + 15u) / 16u, 1u);
      std::array<ID3D11ShaderResourceView *, 16u> clear {};
      out.fill(nullptr);
      context_->CSSetUnorderedAccessViews(0u, 8u, out.data(), nullptr);
      context_->CSSetShaderResources(0u, 16u, clear.data());
    }

    std::vector<char> outline_matte() {
      return read(helper_.mutable_views().matte.Get());
    }

    std::vector<UINT> calibration() {
      return words(read(helper_.mutable_views().calibration_data.Get()));
    }

    std::vector<UINT> root_cache() {
      return words(read(helper_.mutable_views().passable_root_cache.Get()));
    }

    struct aa_history_snapshot_t { std::vector<char> history,header,rgb; };
    aa_history_snapshot_t aa_history_snapshot(const char *label) {
      const auto v=helper_.mutable_views();const auto aa=helper_.aa_history_view();
      require(aa&&v.stability_image&&v.stability_header);
      aa_history_snapshot_t p{read(aa.Get()),read(v.stability_header.Get()),read(v.stability_image.Get())};
      const auto flags=words(p.history),h=words(p.header),packed=words(read(packed_snapshot_.resource.Get()));
      EXPECT_EQ(flags.size(),std::size_t(W_)*H_);EXPECT_EQ(h.size(),160u);
      for(auto x:flags)EXPECT_EQ(x&~3u,0u);
      EXPECT_EQ(h[153u],0u);
      if(h[0u]==0x31435342u){EXPECT_EQ(h[105u],16826368u);EXPECT_EQ(h[106u],16842751u);}
      // This is a bitmap join to the actual prepublish snapshot, not a donor-policy replica.
      // The hook holds its last FULL-dispatch image during the standalone fault shims below.
      if(h[146u]==1u){
        EXPECT_EQ(h[147u],W_*H_);EXPECT_EQ(flags.size(),packed.size());
        for(std::size_t i=0;i<flags.size();++i){const UINT normal=packed[i]&0x3fffu;
          EXPECT_EQ(flags[i],UINT(normal!=0u)|UINT(((normal&768u)!=0u||(packed[i]&(1u<<24u))!=0u)?2u:0u));}
      }
      const auto construction=std::count_if(flags.begin(),flags.end(),[](UINT x){return(x&1u)!=0u;});
      const auto acceptedAA=std::count_if(flags.begin(),flags.end(),[](UINT x){return(x&2u)!=0u;});
      RecordProperty(std::string(label)+"_construction_pixels",construction);RecordProperty(std::string(label)+"_acceptedAA_pixels",acceptedAA);
      return p;
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
    texture_t rgb_, support_, origin_, packed_snapshot_, cache_snapshot_, pre_inside_snapshot_;
    buffer_t probability_, ocr_, sct_, uip_, cb_, root_, raw_parent_snapshot_, raw_flags_snapshot_;
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
    std::string error_, inner_closure_, graph_closure_, enclosure_closure_, ring_closure_, inside_closure_;
    std::array<ComPtr<ID3D11ComputeShader>, 8u> ring_shaders_;
    std::array<ComPtr<ID3D11ComputeShader>, 3u> ring_roi_;
    std::array<ComPtr<ID3D11ComputeShader>, 7u> band_shaders_;
    std::array<ComPtr<ID3D11ComputeShader>, 4u> inside_shaders_;
    nlohmann::json adapter_;
  };

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, InsideProgramIsRequiredAndFreshFinalizerPublishesVersionedReady) {
    for (auto *shader : program_.inside_shaders) ASSERT_NE(shader, nullptr);
    author(true, 0x131u, 1u, false, false, false, 64u, 32u);
    const auto saved = program_.inside_shaders[0u];
    program_.inside_shaders[0u] = nullptr;
    EXPECT_FALSE(run(true, false)) << "Missing Inside program must reject before recording";
    program_.inside_shaders[0u] = saved;
    ASSERT_TRUE(run(true, false)) << error_;
    expect_ready();expect_ring_ready();
    const auto d = calibration();
    ASSERT_EQ(d.size(), ring::data_word_count);
    EXPECT_EQ(d[0u], ring::calibration_tag);EXPECT_EQ(d[1u], ring::calibration_schema);
    EXPECT_EQ(d[77u], 0x31444e49u);EXPECT_EQ(d[78u], 1u);EXPECT_EQ(d[83u], 1u);
    const auto before = words(read(pre_inside_snapshot_.resource.Get()));
    const auto inside = words(read(helper_.mutable_views().inside_bits.Get()));
    const auto final = words(outline_matte());
    ASSERT_EQ(before.size(), W_ * H_);ASSERT_EQ(inside.size(), before.size());ASSERT_EQ(final.size(), before.size());
    const UINT good = d[79u] & ~d[80u] & 255u;
    std::size_t applied = 0u, added = 0u;
    for (std::size_t i = 0u; i < final.size(); ++i) {
      const bool use_inside = (inside[i] & good) != 0u;
      EXPECT_EQ(final[i], UINT(before[i] != 0u || use_inside));
      applied += use_inside;added += use_inside && before[i] == 0u;
    }
    EXPECT_EQ(d[81u], applied);EXPECT_EQ(d[82u], added);
    const auto models = words(read(helper_.enclosure_views().family_models.Get()));
    ASSERT_EQ(models.size(), 17u * 8u);
    ASSERT_TRUE(run(true, true)) << error_;
    EXPECT_EQ(words(outline_matte()), final);
    EXPECT_EQ(words(read(helper_.mutable_views().inside_bits.Get())), inside);
    EXPECT_EQ(calibration(), d);
  }


  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, AdjacentWidthWindowsRetainSharedExtentAndRejectConflicts) {
    author_band(5u, 1320u);
    ASSERT_TRUE(run()) << error_;
    expect_ready();expect_ring_ready();
    constexpr UINT known = 1u, no_information = 4u, tied = 5u, majority = 7u, cap = 9u, fault = 10u;
    struct Case {const char *name;std::vector<UINT> modes;UINT status, width;bool additive = false, source_mismatch = false;};
    const std::vector<Case> cases {
      {"before_one_band_outlier", {8u,6u,8u,7u,10u,9u,7u}, known, 8u},
      {"after_one_band_outlier", {8u,6u,8u,40u,10u,9u,7u}, known, 8u, true},
      {"old_common_majority", {7u,8u,8u,8u,8u,9u,30u}, known, 8u},
      {"separated_equal_cohorts", {8u,8u,8u,20u,20u,20u,40u}, tied, 0u},
      {"only_one_common_band", {7u,7u,8u,9u,9u,20u,30u}, tied, 0u},
      {"union_exactly_half", {7u,8u,8u,9u,20u,30u,40u,50u}, tied, 0u},
      {"unopposed_without_majority", {8u,8u,20u,30u,40u,50u,60u}, majority, 0u},
      {"extent_exceeds_cap", {17u,17u,17u,17u,4u,30u,40u}, cap, 0u},
      {"no_profile_information", {0u,0u,0u,0u,0u,0u,0u}, no_information, 0u},
      {"mismatched_current_source", {8u,6u,8u,40u,10u,9u,7u}, fault, 0u, false, true}
    };
    for (const auto &c : cases) {
      SCOPED_TRACE(c.name);
      const auto data = calibrate_width_histogram(c.modes, c.source_mismatch);
      ASSERT_EQ(data.size(), 68608u);
      EXPECT_EQ(data[3u], 0u);
      EXPECT_EQ(data[128u + 13u], c.status);
      EXPECT_EQ(data[128u + 11u], c.width);
      EXPECT_EQ(data[128u + 12u], c.width == 0u ? 0u : c.width - 1u);
      EXPECT_EQ(data[5u] & 1u, c.status == known ? 1u : 0u);
      EXPECT_EQ(data[128u + 36u], c.additive ? 1u : 0u);
    }
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, SeparateSeedlessRecoveryPreservesExistingLowOnlyGlyph) {
    author_band(5u,821u);
    rect(0u,0u,W_,H_,{40u,60u,100u,255u});
    rect(74u,43u,116u,93u,{1u,1u,1u,255u});rect(79u,48u,111u,88u,{255u,255u,255u,255u});
    rect(155u,43u,213u,93u,{1u,1u,1u,255u});rect(160u,48u,208u,88u,{255u,255u,255u,255u});
    // The minority gray rim makes K genuinely KNOWN rather than unanimously black.
    rect(118u,46u,126u,58u,{32u,32u,32u,255u});rect(120u,48u,124u,56u,{255u,255u,255u,255u});
    // Open both CBC and Inside topology all the way to producer Cover's boundary.
    // This hue fails white eligibility but is not a relative F/K barrier.
    rect(95u,88u,96u,113u,{255u,160u,255u,255u});rect(183u,88u,184u,113u,{255u,160u,255u,255u});
    rect(122u,56u,123u,113u,{255u,160u,255u,255u});
    rect(36u,42u,62u,68u,{1u,1u,1u,255u});rect(41u,47u,57u,63u,{40u,60u,100u,255u});
    rect(45u,51u,53u,59u,{255u,255u,255u,255u});
    std::fill(probability_words_.begin(),probability_words_.end(),0.0f);
    auto detectorRect=[&](UINT l,UINT t,UINT r,UINT b){for(UINT y=t;y<b;++y)std::fill(probability_words_.begin()+y*probability_width+l,probability_words_.begin()+y*probability_width+r,1.0f);};
    detectorRect(285u,55u,425u,115u);detectorRect(600u,55u,800u,115u);detectorRect(450u,55u,470u,74u);
    std::fill(support_words_.begin(),support_words_.end(),0u);support_words_[55u*W_+43u]=1u;
    upload_pixels_from_origin();ASSERT_TRUE(run())<<error_;expect_ready();
    const auto r=receipt();const auto selected=words(read(helper_.enclosure_views().selected_bits.Get()));
    const auto original=words(read(helper_.baseline_views().body.Get())),m=words(outline_matte());
    const auto packed=words(read(packed_snapshot_.resource.Get())),inside=words(read(helper_.mutable_views().inside_bits.Get()));
    constexpr UINT q=70u*256u+206u,target=55u*256u+49u,witness=55u*256u+43u;
    ASSERT_EQ(r[386u],1u);ASSERT_EQ(r[157u],1u);ASSERT_EQ(r[131u],0u);ASSERT_EQ(r[158u],0u);ASSERT_EQ(r[134u],2u);
    ASSERT_EQ(r[7u],0u);ASSERT_EQ(r[8u],1u);ASSERT_EQ(r[87u],1u);
    ASSERT_EQ(original[q],1u);ASSERT_EQ(selected[q]&255u,1u);ASSERT_EQ(selected[q]&0xff00u,0u);
    ASSERT_EQ(ocr_words_[18u],12u);ASSERT_GT(206u,12u*W_/16u);
    ASSERT_EQ(packed[q],0u);ASSERT_EQ(inside[q],0u);ASSERT_EQ(ring_receipt()[4u],0u);
    for(UINT y=69u;y<=71u;++y)for(UINT x=205u;x<=207u;++x)ASSERT_EQ(origin_pixels_[y*W_+x],(pixel8{255u,255u,255u,255u}));
    ASSERT_LT((r[396u]>>16u)&255u,255u); // Actual K box excludes q's all-white N8.
    auto root=[&](const std::vector<UINT>& parent,UINT id){for(UINT h=0u;h<parent.size();++h){require(id>0u&&id<parent.size());UINT next=parent[id];require(next!=0u&&next!=0xffffffffu&&next<=id);if(next==id)return id;id=next;}throw std::runtime_error("Non-convergent native graph");};
    // The typed current CBC mode count proves the last ACTIVE graph is mode0.
    // Palette scratch itself belongs to a later inactive unit after dispatch.
    baseline::resources_t colour;ASSERT_TRUE(colour.ensure(device_.Get(),W_,H_,error_));
    ASSERT_TRUE(colour.dispatch(context_.Get(),colour_ptrs(),graph_ptrs(),cb_.resource.Get(),rgb_.srv.Get(),probability_.srv.Get(),support_.srv.Get(),ocr_.srv.Get(),root_.resource.Get(),240u,error_));
    const auto cv=colour.mutable_views();ASSERT_EQ(r[135u],1u);
    const auto eligible=words(read(cv.eligible.Get())),seeds=words(read(cv.seeds.Get())),parent=words(read(cv.parent.Get())),flags=words(read(cv.root_flags.Get()));
    ASSERT_EQ(eligible[target],1u);ASSERT_EQ(seeds[target],0u);ASSERT_EQ(flags[root(parent,target+1u)],0u);
    ASSERT_EQ(eligible[witness],0u);ASSERT_EQ(seeds[witness],0u);ASSERT_EQ(original[target],0u);
    // CBC's own graph is inspected through its public API, before Raw scratch reuse.
    enclosure::resources_t physical;ASSERT_TRUE(physical.ensure(device_.Get(),W_,H_,error_));
    ASSERT_TRUE(physical.dispatch(context_.Get(),colour_ptrs(),graph_ptrs(),cb_.resource.Get(),rgb_.srv.Get(),probability_.srv.Get(),support_.srv.Get(),ocr_.srv.Get(),root_.resource.Get(),240u,error_,&program_.enclosure));
    const auto pv=physical.baseline_views();const auto pp=words(read(pv.parent.Get())),pf=words(read(pv.root_flags.Get()));
    const UINT physicalRoot=root(pp,target+1u);ASSERT_EQ(root(pp,witness+1u),physicalRoot);ASSERT_EQ(pf[physicalRoot],4u);
    ASSERT_EQ(selected[target]&0x101u,0x101u);
    EXPECT_EQ(m[q],1u);
    for(UINT y=48u;y<88u;++y){for(UINT x=79u;x<111u;++x)EXPECT_EQ(m[y*W_+x],1u);for(UINT x=160u;x<208u;++x)EXPECT_EQ(m[y*W_+x],1u);}
    for(UINT y=51u;y<59u;++y)for(UINT x=45u;x<53u;++x)EXPECT_EQ(m[y*W_+x],1u);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, NullProgramDelegatesLiteral5cad) {
    ASSERT_TRUE(run(false));
    baseline::resources_t reference;
    ASSERT_TRUE(reference.ensure(device_.Get(), W_, H_, error_));
    ASSERT_TRUE(reference.dispatch(context_.Get(), colour_ptrs(), graph_ptrs(), cb_.resource.Get(), rgb_.srv.Get(), probability_.srv.Get(), support_.srv.Get(), ocr_.srv.Get(), root_.resource.Get(), 240u, error_));
    EXPECT_EQ(read(helper_.baseline_views().body.Get()), read(reference.mutable_views().body.Get()));
    EXPECT_EQ(read(helper_.baseline_views().matte.Get()), read(reference.mutable_views().matte.Get()));
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, ClosedCode01RawRingAndDirectRecordedMatch) {
    for (UINT code : {0u, 1u}) {
      author(true, code + 2u, code, false, false, false, 64u, 32u);
      ASSERT_TRUE(run(true, false)) << error_;
      expect_ring_ready();
      const auto b = body(), m = ring_matte(), r = raw(ring_receipt());
      const auto old = words(matte()), now = words(m);
      bool added = false;
      for (std::size_t i = 0; i < now.size(); ++i) {
        EXPECT_LE(old[i], now[i]);
        if (now[i] && !old[i]) {
          added = true;
          EXPECT_LE(std::max({origin_pixels_[i][0], origin_pixels_[i][1], origin_pixels_[i][2]}), 1u);
        }
      }
      EXPECT_TRUE(added);
      ASSERT_TRUE(run(true, true));
      EXPECT_EQ(body(), b);
      EXPECT_EQ(ring_matte(), m);
      EXPECT_EQ(raw(ring_receipt()), r);
      const auto record = helper_.diagnostics().recordings;
      ASSERT_TRUE(run());
      EXPECT_EQ(helper_.diagnostics().recordings, record);
    }
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, RootZeroHoldsAllOutputsAndFreshEmptyClearsImmediately) {
    author(true, 2u, 1u, false, false, false, 64u, 32u);
    ASSERT_TRUE(run());
    expect_ring_ready();
    const auto b = body(), m = matte(), n = ring_matte(), out = outline_matte();
    const auto r = receipt(), rr = ring_receipt(), cd = calibration(), pc = root_cache();
    author(false, 3u, 1u, false, false, false, 64u, 32u);
    root_words_[60] = 0u;
    upload();
    ASSERT_TRUE(run());
    EXPECT_EQ(body(), b);
    EXPECT_EQ(matte(), m);
    EXPECT_EQ(ring_matte(), n);
    EXPECT_EQ(receipt(), r);
    EXPECT_EQ(ring_receipt(), rr);
    EXPECT_EQ(outline_matte(), out);
    EXPECT_EQ(calibration(), cd);
    EXPECT_EQ(root_cache(), pc);
    root_words_[60] = 1u;
    upload();
    ASSERT_TRUE(run());
    expect_ring_ready();
    EXPECT_EQ(body(), std::vector<char>(W_ * H_ * 4u, 0));
    EXPECT_EQ(matte(), std::vector<char>(W_ * H_ * 4u, 0));
    EXPECT_EQ(ring_matte(), std::vector<char>(W_ * H_ * 4u, 0));
    EXPECT_EQ(outline_matte(), std::vector<char>(W_ * H_ * 4u, 0));
    const auto fresh = calibration();
    EXPECT_EQ(fresh.size(), 68608u);
    EXPECT_TRUE(std::all_of(fresh.begin() + 1024u, fresh.end(), [](UINT v) {
      return v == 0u;
    }));
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, OpenPartialAndUnsupportedTransferKeepRawParentAndAccountFinal) {
    author(true, 2u, 1u, true, false, false, 64u, 32u);
    ASSERT_TRUE(run());
    expect_ring_ready();
    expect_ring_parent();
    expect_outline_accounting();
    author(true, 3u, 1u, false, true, false, 64u, 32u);
    ASSERT_TRUE(run());
    expect_ring_ready();
    expect_ring_parent();
    expect_outline_accounting();
    author(true, 4u, 1u, false, false, false, 64u, 32u);
    program_.enclosure.producer.transfer = enclosure::transfer_t::linear;
    ASSERT_TRUE(run());
    expect_ring_ready();
    expect_ring_parent();
    expect_outline_accounting();
    program_.enclosure.producer.transfer = enclosure::transfer_t::encoded_sdr;
    program_.enclosure.producer.original_source = probability_.srv.Get();
    ASSERT_TRUE(run());
    expect_ring_ready();
    expect_ring_parent();
    expect_outline_accounting();
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, CurrentTypedAndNumericFaultFallsBackWithoutStaleRing) {
    author(true, 2u, 1u, false, false, false, 64u, 32u);
    ASSERT_TRUE(run());
    expect_ring_ready();
    uip_words_[4] ^= 1u;
    upload();
    ASSERT_TRUE(run());
    EXPECT_EQ(ring_receipt()[2], 0u);
    expect_ring_parent();
    EXPECT_EQ(calibration()[2], 0u);
    EXPECT_EQ(outline_matte(), ring_matte());
    uip_words_[4] ^= 1u;
    probability_words_.back() = std::bit_cast<float>(0x7fc00000u);
    upload();
    ASSERT_TRUE(run());
    EXPECT_EQ(ring_receipt()[2], 0u);
    expect_ring_parent();
    EXPECT_EQ(calibration()[2], 0u);
    EXPECT_EQ(outline_matte(), ring_matte());
    probability_words_.back() = 1.0f;
    upload();
    ASSERT_TRUE(run());
    expect_ring_ready();
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, TrueCoverBorderBlackAttachmentRejectedWholeComponent) {
    author(true, 2u, 1u, false, false, false, 64u, 32u);
    for (UINT x = 48u; x < 56u; ++x) {
      origin_pixels_[16u * W_ + x] = {1u, 1u, 1u, 255u};
    }
    upload_pixels_from_origin();
    ASSERT_TRUE(run());
    expect_ring_ready();
    expect_ring_parent();
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, ExpectedLimitationBoundedConnectedSceneBlackBranchIncluded) {
    author(true, 2u, 1u, false, false, false, 64u, 32u);
    for (UINT x = 48u; x < 52u; ++x) {
      origin_pixels_[16u * W_ + x] = {1u, 1u, 1u, 255u};
    }
    upload_pixels_from_origin();
    ASSERT_TRUE(run());
    expect_ring_ready();
    const auto old = words(matte()), now = words(ring_matte());
    EXPECT_EQ(old[16u * W_ + 51u], 0u);
    EXPECT_EQ(now[16u * W_ + 51u], 1u);
    // Port-conformance counterexample, never a claim that arbitrary scene-black is subtitle.
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, DenseSnakeCoverAndInvalidExtentStayBounded) {
    for (UINT stress : {1u, 2u}) {
      author(true, stress + 2u, 1u, false, false, false, 256u, 128u, stress);
      ASSERT_TRUE(run()) << error_;
      expect_ring_ready();
      const auto old = words(matte()), now = words(ring_matte());
      ASSERT_EQ(old.size(), now.size());
      for (std::size_t i = 0; i < now.size(); ++i) {
        EXPECT_LE(old[i], now[i]);
        EXPECT_LE(now[i], 1u);
      }
    }
    ring::resources_t invalid;
    EXPECT_FALSE(invalid.ensure(device_.Get(), 0u, 16u, error_));
    EXPECT_FALSE(invalid.ensure(device_.Get(), 5121u, 16u, error_));
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, EnabledRejectsPredicationBadRootAndDeferredBeforeCommandsAndRestoresCs) {
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

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, CurrentGpuProfilesChooseObservedThicknessWithoutWidthUpload) {
    for (UINT t : {4u, 6u}) {
      author_band(t, 20u + t);
      ASSERT_TRUE(run(true, false)) << error_;
      expect_outline_accounting();
      const auto d = calibration();
      ASSERT_EQ(d.size(), 68608u);
      EXPECT_EQ(d[128u + 11u], t);
      EXPECT_EQ(d[128u + 12u], t - 1u);
      EXPECT_GT(d[128u + 8u], 0u);
      EXPECT_EQ(d[128u + 22u], t);
      EXPECT_GE(d[128u + 24u], 2u);
      const auto direct_m = outline_matte(), direct_packed = read(packed_snapshot_.resource.Get());
      const auto direct_cache = root_cache();
      ASSERT_TRUE(run(true, true)) << error_;
      EXPECT_EQ(outline_matte(), direct_m);
      EXPECT_EQ(calibration(), d);
      EXPECT_EQ(root_cache(), direct_cache);
      EXPECT_EQ(read(packed_snapshot_.resource.Get()), direct_packed);
    }
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, TileBoundaryT16UsesGlobalInnerNeighbourBeyondSharedHalo) {
    author_band(16u, 42u, true);
    ASSERT_TRUE(run()) << error_;
    expect_outline_accounting();
    const auto d = calibration();
    ASSERT_EQ(d.size(), 68608u);
    ASSERT_EQ(d[139u], 16u);
    const auto r = words(ring_matte()), m = words(outline_matte());
    EXPECT_EQ(r[64u * W_ + 63u], 0u);
    EXPECT_EQ(m[64u * W_ + 63u], 1u);
    const auto packed = words(read(packed_snapshot_.resource.Get()));
    EXPECT_NE(packed[64u * W_ + 63u] & 1u, 0u);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, ThicknessResourceBoundFallsBackWithoutClamping) {
    author_band(17u, 43u);
    ASSERT_TRUE(run()) << error_;
    expect_outline_accounting();
    const auto d = calibration();
    ASSERT_EQ(d.size(), 68608u);
    EXPECT_EQ(d[139u], 0u);
    EXPECT_EQ(d[150u], 17u);
    EXPECT_NE(d[151u] & 4u, 0u);
    EXPECT_EQ(outline_matte(), ring_matte());
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, NearBlackEnclosedIslandUsesWholeVerifiedBoundary) {
    author_band(5u, 44u);
    rect(75u, 60u, 76u, 61u, {2u, 2u, 2u, 255u});
    upload_pixels_from_origin();
    ASSERT_TRUE(run()) << error_;
    expect_outline_accounting();
    const auto d = calibration();
    ASSERT_EQ(d[139u], 5u);
    EXPECT_GE(d[128u + 26u], 1u);
    const auto r = words(ring_matte()), m = words(outline_matte()), packed = words(read(packed_snapshot_.resource.Get()));
    const auto q = 60u * W_ + 75u;
    EXPECT_EQ(r[q], 0u);
    EXPECT_EQ(packed[q] & 511u, 0u);
    EXPECT_NE(packed[q] & 1024u, 0u);
    EXPECT_EQ(m[q], 1u);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, MeasuredPhysicalBlackBoundarySupportsIslandWithoutSelectedBody) {
    author_band(5u, 45u);
    rect(75u, 58u, 78u, 61u, {2u, 2u, 2u, 255u});
    rect(76u, 59u, 77u, 60u, {0u, 0u, 0u, 255u});
    upload_pixels_from_origin();
    ASSERT_TRUE(run()) << error_;
    expect_outline_accounting();
    const auto d = calibration();
    ASSERT_EQ(d[139u], 5u);
    EXPECT_EQ(d[128u + 27u], 0u);
    const auto r = words(ring_matte()), m = words(outline_matte()), packed = words(read(packed_snapshot_.resource.Get()));
    const auto selected = words(read(helper_.enclosure_views().selected_bits.Get()));
    const auto q = 59u * W_ + 75u;
    const auto black = 59u * W_ + 76u;
    ASSERT_EQ(r[black], 1u);
    ASSERT_EQ(selected[black] & 1u, 0u);
    ASSERT_NE(packed[black] & 1u, 0u);  // Learned physical BAND supports the boundary without selected BODY.
    EXPECT_GE(d[128u + 26u], 1u);
    EXPECT_NE(packed[q] & 1024u, 0u);
    EXPECT_EQ(r[q], 0u);
    EXPECT_EQ(m[q], 1u);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, OpenedIslandIsExteriorAndNeverGetsTopologyBit) {
    author_band(5u, 46u);
    rect(74u, 58u, 78u, 61u, {2u, 2u, 2u, 255u});
    upload_pixels_from_origin();
    ASSERT_TRUE(run()) << error_;
    expect_outline_accounting();
    const auto d = calibration();
    ASSERT_EQ(d[139u], 5u);
    const auto packed = words(read(packed_snapshot_.resource.Get()));
    EXPECT_EQ(packed[59u * W_ + 75u] & 1024u, 0u);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, ZeroColourChannelsOneHopAndNoRecursiveHalo) {
    author_band(5u, 47u);
    rect(73u, 60u, 74u, 61u, {100u, 0u, 0u, 255u});
    rect(72u, 60u, 73u, 61u, {200u, 0u, 0u, 255u});
    upload_pixels_from_origin();
    ASSERT_TRUE(run()) << error_;
    expect_outline_accounting();
    const auto m = words(outline_matte()), packed = words(read(packed_snapshot_.resource.Get()));
    EXPECT_NE(packed[60u * W_ + 73u] & 256u, 0u);
    EXPECT_NE(packed[60u * W_ + 73u] & 512u, 0u);
    EXPECT_EQ(m[60u * W_ + 73u], 1u);
    EXPECT_EQ(m[60u * W_ + 72u], 0u);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, OtherUnitBodyBlocksOwnOutlineBandWhileCode1CanSupportValidatedAA) {
    author_band(5u, 48u);
    rect(53u, 47u, 79u, 89u, {0u, 0u, 0u, 255u});
    rect(58u, 52u, 74u, 84u, {100u, 100u, 100u, 255u});
    rect(72u, 60u, 73u, 61u, {120u, 120u, 120u, 255u});  // Below WHITE's half-peak eligibility floor.
    // A second pair has Code1 candidate support without accepted BODY. Its
    // literal 25/50 gray edge can still earn a current one-hop AA certificate.
    rect(64u, 50u, 74u, 51u, {2u, 2u, 2u, 255u});
    rect(73u, 50u, 74u, 51u, {25u, 25u, 25u, 255u});
    rect(72u, 50u, 73u, 51u, {50u, 50u, 50u, 255u});
    // Unit0 cover begins x64, so both test rays are genuine unit0 exterior.
    // Unit1 keeps the full closed gray glyph but its core excludes WHITE x79.
    ocr_words_[144u] = 4u;
    ocr_words_[3] = ocr_words_[4] = 2u;
    uip_words_[3] = 3u;
    const std::array<UINT, 4u> core {3u, 6u, 4u, 11u}, cover {2u, 3u, 6u, 14u};
    std::copy(core.begin(), core.end(), ocr_words_.begin() + 24u);
    std::copy(cover.begin(), cover.end(), ocr_words_.begin() + 152u);
    upload_pixels_from_origin();
    ASSERT_TRUE(run()) << error_;
    expect_outline_accounting();
    const auto b = words(body()), packed = words(read(packed_snapshot_.resource.Get()));
    const auto bits = words(read(helper_.enclosure_views().selected_bits.Get()));
    const auto original_m = words(matte());
    const auto own_box3 = [&](UINT x, UINT y) {
      for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
          const int xx = int(x) + dx, yy = int(y) + dy;
          if (xx >= 0 && yy >= 0 && xx < int(W_) && yy < int(H_) && (bits[UINT(yy) * W_ + UINT(xx)] & 1u)) {
            return true;
          }
        }
      }
      return false;
    };
    ASSERT_EQ(calibration()[139u], 5u);
    const auto q = 60u * W_ + 73u, r = 60u * W_ + 72u;
    ASSERT_EQ(b[q], 1u);
    ASSERT_EQ(bits[q] & 3u, 2u);  // The blocking Body really belongs to the OTHER unit.
    ASSERT_EQ(bits[r] & 1u, 0u);
    ASSERT_FALSE(own_box3(73u, 60u));
    ASSERT_FALSE(own_box3(72u, 60u));
    ASSERT_NE(packed[60u * W_ + 74u] & 1u, 0u);  // Current unit0 physical seed exists.
    EXPECT_EQ(packed[q] & 1u, 0u);
    EXPECT_EQ(packed[r] & 1u, 0u);
    EXPECT_EQ(packed[q] & (256u | 512u), 0u);
    const auto qm = 50u * W_ + 73u, rm = 50u * W_ + 72u;
    ASSERT_EQ(b[qm], 0u);
    ASSERT_EQ(b[rm], 0u);
    ASSERT_EQ(original_m[qm], 1u);
    ASSERT_EQ(original_m[rm], 1u);
    ASSERT_FALSE(own_box3(73u, 50u));
    ASSERT_FALSE(own_box3(72u, 50u));
    ASSERT_NE(packed[50u * W_ + 74u] & 1u, 0u);
    EXPECT_EQ(packed[qm] & 1u, 0u);
    EXPECT_EQ(packed[rm] & 1u, 0u);
    // AA flags are global, not unit-owned BAND bits; Code1 is a candidate,
    // rather than an exclusion from this independently validated certificate.
    EXPECT_EQ(packed[qm] & (256u | 512u), 256u | 512u);
    EXPECT_EQ(words(outline_matte())[qm], 1u);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, InvalidOrAliasedOptionalSnapshotsRejectBeforeCommands) {
    author_band(5u, 49u);
    ASSERT_TRUE(run()) << error_;
    const auto m = outline_matte(), r = ring_matte();
    const auto d = calibration();
    program_.packed_temp_snapshot = rgb_.resource.Get();
    EXPECT_FALSE(run());
    EXPECT_EQ(outline_matte(), m);
    EXPECT_EQ(ring_matte(), r);
    EXPECT_EQ(calibration(), d);
    program_.packed_temp_snapshot = cache_snapshot_.resource.Get();
    EXPECT_FALSE(run());
    EXPECT_EQ(outline_matte(), m);
    EXPECT_EQ(calibration(), d);
    program_.packed_temp_snapshot = packed_snapshot_.resource.Get();
    ComPtr<ID3D11Resource> alias;
    helper_.mutable_views().matte->GetResource(&alias);
    ComPtr<ID3D11Texture2D> alias_texture;
    require(SUCCEEDED(alias.As(&alias_texture)));
    program_.packed_temp_snapshot = alias_texture.Get();
    EXPECT_FALSE(run());
    EXPECT_EQ(outline_matte(), m);
    program_.packed_temp_snapshot = packed_snapshot_.resource.Get();
    ASSERT_TRUE(run());
    EXPECT_EQ(outline_matte(), m);
    program_.packed_temp_snapshot = nullptr;
    program_.passable_cache_snapshot = nullptr;
    ASSERT_TRUE(run());
    EXPECT_EQ(outline_matte(), m);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, ActualLateCacheRootFaultPublishesLiteralRawFallback) {
    author_band(5u, 50u);
    rect(75u, 60u, 76u, 61u, {3u, 3u, 3u, 255u});
    upload_pixels_from_origin();
    ASSERT_TRUE(run()) << error_;
    const auto b = body(), r = ring_matte();
    ASSERT_NE(outline_matte(), r);
    inject_late_actual_fault(false);
    const auto d = calibration();
    EXPECT_EQ(d[2], 0u);
    EXPECT_NE(d[3] & 8u, 0u);
    EXPECT_EQ(d[7], 0u);
    EXPECT_EQ(outline_matte(), r);
    EXPECT_EQ(body(), b);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, ActualLateGraphShapeFaultPublishesLiteralRawFallback) {
    author_band(5u, 51u);
    rect(75u, 60u, 76u, 61u, {3u, 3u, 3u, 255u});
    upload_pixels_from_origin();
    ASSERT_TRUE(run()) << error_;
    const auto b = body(), r = ring_matte();
    ASSERT_NE(outline_matte(), r);
    inject_late_actual_fault(true);
    const auto d = calibration();
    EXPECT_EQ(d[2], 0u);
    EXPECT_NE(d[3] & 1u, 0u);
    EXPECT_EQ(d[7], 0u);
    EXPECT_EQ(outline_matte(), r);
    EXPECT_EQ(body(), b);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, N8MiterAndUncappedSelectedRawSupportRemainDistinct) {
    author_band(5u, 52u, true);
    ASSERT_TRUE(run()) << error_;
    expect_outline_accounting();
    const auto d = calibration();
    ASSERT_EQ(d[139u], 5u);
    const auto packed = words(read(packed_snapshot_.resource.Get()));
    EXPECT_NE(packed[43u * W_ + 74u] & 1u, 0u);  // True N8 corner, with black-background RawM rejected at cover edge.
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, BandAndNativeCoreResourceBoundsKeepCurrentRawFallback) {
    author(true, 53u, 0u, false, false, false, 1024u, 128u, 3u);
    rect(0u, 0u, W_, H_, {128u, 128u, 128u, 255u});
    rect(155u, 51u, 869u, 65u, {0u, 0u, 0u, 255u});
    rect(160u, 56u, 864u, 60u, {255u, 255u, 255u, 255u});
    upload_pixels_from_origin();
    ASSERT_TRUE(run()) << error_;
    expect_outline_accounting();
    auto d = calibration();
    ASSERT_EQ(d[128u], 704u * 4u);  // White stripe is now sampled at native palette row57.
    EXPECT_EQ(d[133u], 4u);
    EXPECT_EQ(d[135u], 176u);
    EXPECT_GT(d[135u], 64u);
    EXPECT_EQ(d[139u], 0u);
    EXPECT_NE(d[151u] & 1u, 0u);
    EXPECT_EQ(outline_matte(), ring_matte());
    author(true, 54u, 0u, false, false, false, 256u, 512u, 3u);
    ASSERT_TRUE(run()) << error_;
    expect_outline_accounting();
    d = calibration();
    ASSERT_GT(d[128u], 0u);
    EXPECT_GT(d[134u], 128u);
    EXPECT_EQ(d[139u], 0u);
    EXPECT_NE(d[151u] & 2u, 0u);
    EXPECT_EQ(outline_matte(), ring_matte());
  }

  // Genuine current-source regression; every prerequisite is checked on GPU outputs.
  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, ResidualCollarRequiresVisibleBlackForUntrustedLowEndpoint) {
    for (const bool recorded : {false, true}) {
      author_band(5u, 61u);
      // Distinct-hue scene background cannot join WHITE's broad half-peak E.
      rect(0u, 0u, W_, H_, {40u, 60u, 100u, 255u});
      // Real closed WHITE peers supply known black and adopted current topology.
      rect(74u, 43u, 116u, 93u, {1u, 1u, 1u, 255u});
      rect(79u, 48u, 111u, 88u, {255u, 255u, 255u, 255u});
      rect(155u, 43u, 213u, 93u, {1u, 1u, 1u, 255u});
      rect(160u, 48u, 208u, 88u, {255u, 255u, 255u, 255u});
      // A minority GRAY32 outline supplies an actual second dark-anchor
      // population; a unanimous K1 histogram intentionally remains UNKNOWN.
      rect(118u, 46u, 126u, 58u, {32u, 32u, 32u, 255u});
      rect(120u, 48u, 124u, 56u, {255u, 255u, 255u, 255u});
      // These short rows lie BETWEEN the native palette-grid rows. They must
      // remain source-seeded LOW only, not create a modal F family themselves.
      for (const UINT y : {60u, 72u, 80u}) {
        rect(126u, y, 128u, y + 4u, {1u, 1u, 1u, 255u});
        rect(128u, y, 132u, y + 4u, {160u, 140u, 160u, 255u});
      }
      // q lies3 pixels from F, with two noneligible scene pixels between.
      // K is opposite q at distance3; no N8 black/OuterRays shortcut is present.
      rect(134u, 62u, 135u, 63u, {157u, 137u, 157u, 255u});
      rect(137u, 62u, 138u, 63u, {1u, 1u, 1u, 255u});
      rect(134u, 74u, 135u, 75u, {80u, 70u, 80u, 255u});
      rect(137u, 74u, 138u, 75u, {1u, 1u, 1u, 255u});
      // A physical K barrier at distance2 must stop the search before F.
      rect(132u, 80u, 133u, 84u, {1u, 1u, 1u, 255u});
      rect(134u, 82u, 135u, 83u, {80u, 70u, 80u, 255u});
      rect(137u, 82u, 138u, 83u, {1u, 1u, 1u, 255u});
      // A separate trusted modal WHITE endpoint preserves the old near-F path.
      rect(144u, 60u, 146u, 64u, {1u, 1u, 1u, 255u});
      rect(146u, 60u, 150u, 64u, {255u, 255u, 255u, 255u});
      rect(152u, 62u, 153u, 63u, {252u, 252u, 252u, 255u});

      // Authored probability rectangles, not a CPU segmentation result. There
      // is no support anywhere. Target q cells stay P0/S0 and component-isolated.
      std::fill(probability_words_.begin(), probability_words_.end(), 0.0f);
      const auto probability_rect = [&](UINT l, UINT t, UINT r, UINT b) {
        ASSERT_LT(l, r); ASSERT_LT(t, b);
        ASSERT_LE(r, probability_width); ASSERT_LE(b, probability_height);
        for (UINT y = t; y < b; ++y) for (UINT x = l; x < r; ++x) {
          probability_words_[y * probability_width + x] = 1.0f;
        }
      };
      probability_rect(285u, 55u, 425u, 115u);
      probability_rect(600u, 55u, 800u, 115u);
      probability_rect(450u, 55u, 470u, 74u);
      probability_rect(470u, 72u, 495u, 80u);
      probability_rect(470u, 87u, 495u, 96u);
      probability_rect(470u, 99u, 495u, 106u);
      probability_rect(545u, 72u, 568u, 80u);
      upload_pixels_from_origin();
      ASSERT_TRUE(run(true, recorded)) << error_;
      expect_ready(); expect_ring_ready(); expect_outline_accounting();

      const auto meta = receipt();
      ASSERT_EQ(meta[386u], 1u) << "native K record384..399=" << nlohmann::json(std::vector<UINT>(meta.begin() + 384u, meta.begin() + 400u)).dump();
      ASSERT_TRUE(cbc_unit_state_contract_test::CurrentCBCPhysicalRefinementAdopted(
          meta[128u], meta[129u], meta[131u], meta[134u], meta[157u]));
      const auto selected = words(read(helper_.enclosure_views().selected_bits.Get()));
      const auto code1 = words(matte()), raw_m = words(ring_matte());
      const auto packed = words(read(packed_snapshot_.resource.Get()));
      const auto before_inside = words(read(pre_inside_snapshot_.resource.Get()));
      const auto final_m = words(outline_matte());
      const auto inside = words(read(helper_.mutable_views().inside_bits.Get()));
      const auto at = [&](UINT x, UINT y) { return y * W_ + x; };
      for (const UINT y : {62u, 74u, 82u}) {
        ASSERT_NE(selected[at(131u, y)] & 1u, 0u);   // First reached F is real own LOW.
        ASSERT_EQ(selected[at(131u, y)] & 256u, 0u); // It is deliberately untrusted.
      }
      ASSERT_NE(selected[at(149u, 62u)] & 256u, 0u); // Independent trusted-HIGH control.
      for (const auto q : {at(134u, 62u), at(134u, 74u), at(134u, 82u), at(152u, 62u)}) {
        ASSERT_EQ(selected[q] & 0xffffu, 0u); // Target was not admitted as any BODY/HIGH.
        ASSERT_EQ(raw_m[q], 1u);               // Raw starts from literal Code1; valid final publication may prune it.
        ASSERT_EQ(packed[q] & 0x3fffu, 0u);   // No BAND/AA/closure/unknown fallback path.
        ASSERT_EQ(inside[q], 0u);
        ASSERT_EQ(code1[q], 1u);              // Actual existing BOX3 candidate only.
      }
      // New assertion fails on old control ONLY at the false q if all premises
      // above hold. The visible black mix and trusted-HIGH paths must survive.
      EXPECT_EQ(before_inside[at(134u, 62u)], 0u);
      EXPECT_EQ(final_m[at(134u, 62u)], 0u);
      EXPECT_EQ(before_inside[at(134u, 74u)], 1u);
      EXPECT_EQ(final_m[at(134u, 74u)], 1u);
      EXPECT_EQ(before_inside[at(152u, 62u)], 1u);
      EXPECT_EQ(final_m[at(152u, 62u)], 1u);
      EXPECT_EQ(before_inside[at(134u, 82u)], 0u);
      EXPECT_EQ(final_m[at(134u, 82u)], 0u);
      for (UINT y = 48u; y < 88u; ++y) for (UINT x = 79u; x < 111u; ++x) {
        EXPECT_EQ(final_m[at(x, y)], 1u);       // All actual opaque WHITE peer ink retained.
      }
    }
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, EarlyInsideMovedShrunkCoverReplacesRejectedAndClearsOutside) {
    author_band(5u, 100u);
    ASSERT_TRUE(run()) << error_;
    const auto firstMeta=receipt(), firstData=calibration();
    ASSERT_EQ(firstMeta[386u],1u) << "author_band must establish a real Known K model";
    ASSERT_EQ(firstData[79u]&1u,1u); ASSERT_EQ(firstData[80u],0u);
    const auto oldInside=words(read(helper_.mutable_views().inside_bits.Get()));
    ASSERT_GT(std::count_if(oldInside.begin(),oldInside.end(),[](UINT v){return(v&1u)!=0u;}),0);
    // Fresh authored source/clock, same shape; retain the first enclosed glyph
    // while clipping the second through the new cover boundary.
    author_band(5u, 101u);
    const std::array<UINT,4u> core{5u,5u,11u,11u},cover{4u,4u,12u,13u};
    std::copy(core.begin(),core.end(),ocr_words_.begin()+16u);
    std::copy(cover.begin(),cover.end(),ocr_words_.begin()+144u);
    upload(); ASSERT_TRUE(run()) << error_;
    const auto meta=receipt(),data=calibration();
    ASSERT_EQ(meta[386u],1u); ASSERT_EQ(data[79u]&1u,1u); ASSERT_EQ(data[80u],0u);
    const auto now=words(read(helper_.mutable_views().inside_bits.Get()));
    const auto covers=words(read(helper_.enclosure_views().cover_bits.Get()));
    ASSERT_EQ(now.size(),oldInside.size());
    std::size_t currentPositive=0u,oldOutside=0u,rejectedInCurrentROI=0u;
    for(UINT y=0u;y<H_;++y)for(UINT x=0u;x<W_;++x){
      const auto i=y*W_+x;const bool inROI=x>=meta[152u]&&y>=meta[153u]&&x<meta[154u]&&y<meta[155u];
      currentPositive+=(now[i]&1u)!=0u;
      if((covers[i]&1u)==0u){EXPECT_EQ(now[i]&1u,0u);oldOutside+=(oldInside[i]&1u)!=0u;}
      if(inROI&&(covers[i]&1u)!=0u&&(oldInside[i]&1u)!=0u&&(now[i]&1u)==0u)++rejectedInCurrentROI;
    }
    ASSERT_GT(currentPositive,0u); ASSERT_GT(oldOutside,0u); ASSERT_GT(rejectedInCurrentROI,0u);
    RecordProperty("positive_current_Inside_pixels",currentPositive);
    RecordProperty("old_Inside_outside_currentCover",oldOutside);
    RecordProperty("rejected_old_Inside_inside_currentROI",rejectedInCurrentROI);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, EarlyInsideOverlapPublicationFaultRetainsOtherUnit) {
    author_band(5u, 110u);
    ocr_words_[3u]=ocr_words_[4u]=2u;
    std::copy_n(ocr_words_.begin()+16u,4u,ocr_words_.begin()+24u);
    std::copy_n(ocr_words_.begin()+144u,4u,ocr_words_.begin()+152u);
    uip_words_[3u]=3u; upload(); ASSERT_TRUE(run()) << error_;
    const auto meta=receipt(),initialData=calibration();
    ASSERT_EQ(meta[386u],1u); ASSERT_EQ(meta[402u],1u);
    ASSERT_EQ(initialData[79u]&3u,3u); ASSERT_EQ(initialData[80u],0u);
    const auto v=helper_.mutable_views();const auto e=helper_.enclosure_views();
    const auto rv=helper_.raw_views();const auto b=helper_.baseline_views();
    const auto before=words(read(v.inside_bits.Get()));
    ASSERT_GT(std::count(before.begin(),before.end(),3u),0);
    // Only test fault injection is Data80 bit0, after real native positive
    // inference. No foreground/mask/graph/packed/history pixels are authored.
    ComPtr<ID3D11Resource> dataResource;v.calibration_data->GetResource(&dataResource);
    UINT fault=initialData[80u]|1u;const D3D11_BOX faultBox{80u*4u,0u,0u,81u*4u,1u,1u};
    context_->UpdateSubresource(dataResource.Get(),0u,&faultBox,&fault,0u,0u);
    D3D11_UNORDERED_ACCESS_VIEW_DESC dd{};dd.Format=DXGI_FORMAT_R32_TYPELESS;dd.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;dd.Buffer.NumElements=68608u;dd.Buffer.Flags=D3D11_BUFFER_UAV_FLAG_RAW;
    ComPtr<ID3D11UnorderedAccessView> dataUAV;ASSERT_TRUE(SUCCEEDED(device_->CreateUnorderedAccessView(dataResource.Get(),&dd,&dataUAV)));
    auto shape=buffer(16u,true);const std::array<UINT,4u> dims{W_,H_,W_*H_,0u};context_->UpdateSubresource(shape.resource.Get(),0u,nullptr,dims.data(),0u,0u);
    const std::array<UINT,4u> hook{};context_->UpdateSubresource(v.hook_constants.Get(),0u,nullptr,hook.data(),0u,0u);
    std::array<ID3D11ShaderResourceView*,16u> clear{};std::array<ID3D11UnorderedAccessView*,8u> out{};
    context_->CSSetShaderResources(0u,16u,clear.data());context_->CSSetUnorderedAccessViews(0u,8u,out.data(),nullptr);
    ID3D11Buffer* constants[3]{cb_.resource.Get(),v.hook_constants.Get(),shape.resource.Get()};
    const std::array<ID3D11ShaderResourceView*,16u> in{rgb_.srv.Get(),ocr_.srv.Get(),e.selected_bits.Get(),e.body.Get(),e.cover_bits.Get(),e.receipt.Get(),b.parent.Get(),b.root_flags.Get(),b.eligible.Get(),rv.eligible_inner.Get(),rv.matte.Get(),rv.receipt.Get(),e.matte.Get(),nullptr,nullptr,v.inside_bits.Get()};
    auto matteUAV=uav(v.matte.Get()),insideUAV=uav(v.inside_bits.Get());out[0]=dataUAV.Get();out[1]=matteUAV.Get();
    context_->CSSetConstantBuffers(0u,3u,constants);context_->CSSetShaderResources(0u,16u,in.data());context_->CSSetUnorderedAccessViews(0u,8u,out.data(),nullptr);
    context_->CSSetShader(band_shaders_[4].Get(),nullptr,0u);context_->Dispatch((W_+15u)/16u,(H_+15u)/16u,1u);
    out.fill(nullptr);context_->CSSetUnorderedAccessViews(0u,8u,out.data(),nullptr);context_->CSSetShaderResources(0u,16u,clear.data());
    // Existing final Inside phase now owns pointwise canonicalization.
    const std::array<ID3D11ShaderResourceView*,15u> insideIn{rgb_.srv.Get(),e.all_body_bits.Get(),nullptr,e.cover_bits.Get(),e.receipt.Get(),nullptr,nullptr,nullptr,rv.matte.Get(),e.matte.Get(),nullptr,nullptr,nullptr,e.family_models.Get(),nullptr};
    out[0]=matteUAV.Get();out[6]=insideUAV.Get();out[7]=dataUAV.Get();
    context_->CSSetShaderResources(0u,15u,insideIn.data());context_->CSSetUnorderedAccessViews(0u,8u,out.data(),nullptr);
    context_->CSSetShader(inside_shaders_[3].Get(),nullptr,0u);context_->Dispatch((W_+15u)/16u,(H_+15u)/16u,1u);
    out.fill(nullptr);context_->CSSetUnorderedAccessViews(0u,8u,out.data(),nullptr);context_->CSSetShaderResources(0u,16u,clear.data());
    const auto after=words(read(v.inside_bits.Get())),data=calibration();
    ASSERT_EQ(data[80u],1u); ASSERT_EQ(data[2u],1u); ASSERT_EQ(data[3u],0u);
    ASSERT_EQ(after.size(),before.size());std::size_t preservedOverlap=0u;
    for(std::size_t i=0u;i<after.size();++i){EXPECT_EQ(after[i],before[i]&2u);if(before[i]==3u){EXPECT_EQ(after[i],2u);++preservedOverlap;}}
    ASSERT_GT(preservedOverlap,0u);RecordProperty("native_overlap3_preserved_as2",preservedOverlap);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, EarlyInsideRootZeroHoldsThenFreshEmptyClears) {
    author_band(5u,120u);ASSERT_TRUE(run()) << error_;
    const auto meta=receipt(),d=calibration();ASSERT_EQ(meta[386u],1u);ASSERT_EQ(d[79u]&1u,1u);ASSERT_EQ(d[80u],0u);
    const auto held=read(helper_.mutable_views().inside_bits.Get());const auto native=words(held);
    ASSERT_GT(std::count_if(native.begin(),native.end(),[](UINT v){return(v&1u)!=0u;}),0);
    author(false,121u,0u,false,false,false,256u,128u,3u);
    root_words_[60u]=0u;upload();ASSERT_TRUE(run()) << error_;
    EXPECT_EQ(read(helper_.mutable_views().inside_bits.Get()),held);
    root_words_[60u]=1u;upload();ASSERT_TRUE(run()) << error_;
    EXPECT_EQ(read(helper_.mutable_views().inside_bits.Get()),std::vector<char>(W_*H_*4u,0));
    EXPECT_EQ(outline_matte(),std::vector<char>(W_*H_*4u,0));
    const auto empty=calibration();EXPECT_EQ(empty[79u],0u);EXPECT_EQ(empty[80u],0u);
    EXPECT_EQ(words(read(helper_.mutable_views().inside_arguments.Get())),std::vector<UINT>({0u,0u,0u,0u}));
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, LiteralSingleton012ModelsPreserveVisibleWhiteBody) {
    std::vector<UINT> zeroMask;
    for(UINT code : {0u,1u,2u}) {
      author_band(5u,200u+code);
      if(code!=0u) {
        // Separate authored input scene. Existing0595 raster/tests are untouched.
        for(auto& p:origin_pixels_)if(p[0]==0u&&p[1]==0u&&p[2]==0u) {
          p[0]=p[1]=p[2]=static_cast<std::uint8_t>(code);
        }
        upload_pixels_from_origin();
      }
      ASSERT_TRUE(run()) << error_;
      const auto meta=receipt(),data=calibration();
      const std::string label="literal"+std::to_string(code);
      RecordProperty(label+"_actual_K16",nlohmann::json(std::vector<UINT>(meta.begin()+384u,meta.begin()+400u)).dump());
      ASSERT_EQ(meta[384u],0x314b4c42u);
      ASSERT_GE(meta[388u],2u);ASSERT_GE(meta[391u],2u);
      ASSERT_EQ(meta[390u],meta[388u]);ASSERT_EQ(meta[392u],meta[391u]);
      ASSERT_EQ(meta[389u],code*0x010101u);
      if(code<=1u) {
        EXPECT_EQ(meta[385u],3u);EXPECT_EQ(meta[386u],1u);EXPECT_EQ(meta[387u],1u);
        EXPECT_EQ(meta[393u],meta[388u]);EXPECT_EQ(meta[394u],code);
        EXPECT_EQ(meta[395u],meta[389u]);EXPECT_EQ(meta[396u],meta[389u]);
      } else {
        EXPECT_EQ(meta[385u],1u);EXPECT_EQ(meta[386u],0u);EXPECT_EQ(meta[387u],5u);
        EXPECT_EQ(meta[393u],0u);EXPECT_EQ(meta[394u],0u);EXPECT_EQ(meta[395u],0u);EXPECT_EQ(meta[396u],0u);
      }
      const auto b=words(body()),m=words(outline_matte());
      std::size_t whiteBody=0u,whiteProtected=0u;
      // Authored opaque white interiors, independent of inferred model/mask.
      for(UINT y=48u;y<88u;++y)for(UINT x=0u;x<W_;++x)if((x>=79u&&x<111u)||(x>=160u&&x<208u)) {
        const auto i=y*W_+x;EXPECT_EQ(b[i],1u);EXPECT_EQ(m[i],1u);
        whiteBody+=b[i]!=0u;whiteProtected+=m[i]!=0u;
      }
      EXPECT_EQ(whiteBody,3200u);EXPECT_EQ(whiteProtected,3200u);
      RecordProperty(label+"_white_BODY_covered",whiteBody);RecordProperty(label+"_white_M_covered",whiteProtected);
      RecordProperty(label+"_Inside_good",data[79u]);RecordProperty(label+"_Inside_fault",data[80u]);
      if(code==0u)zeroMask=m;
      if(code==1u)EXPECT_EQ(m,zeroMask) << "Literal0/1 same physical mask semantics; preserve any difference as failure";
    }
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, UniformSceneWithoutContrastHasNoSingletonKnownModel) {
    author_band(5u,210u);
    // Fresh active OCR/P envelope, but no RGB contrast anywhere in the scene.
    for(auto& p:origin_pixels_)p={128u,128u,128u,255u};
    upload_pixels_from_origin();ASSERT_TRUE(run()) << error_;
    const auto meta=receipt();ASSERT_EQ(ocr_words_[4u],1u);
    RecordProperty("uniform_actual_K16",nlohmann::json(std::vector<UINT>(meta.begin()+384u,meta.begin()+400u)).dump());
    EXPECT_EQ(meta[386u],0u);EXPECT_NE(meta[385u],3u);
    const auto b=words(body()),m=words(outline_matte());
    EXPECT_EQ(std::count_if(b.begin(),b.end(),[](UINT v){return v!=0u;}),0);
    EXPECT_EQ(std::count_if(m.begin(),m.end(),[](UINT v){return v!=0u;}),0);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, Saved28AuthenticFullCurrentInputsParity) {
    if (!std::getenv("SUNSHINE_TEST_OUTLINE_CLOSURE_CASES") || !std::getenv("SUNSHINE_TEST_OUTLINE_CLOSURE_OUTPUT")) {
      GTEST_SKIP() << "Explicit immutable28 cases/output required";
    }
    saved();
  }

  void HostSbsCurrentOutlineSupportClosureGpuTest::saved() {
    const std::filesystem::path input = std::getenv("SUNSHINE_TEST_OUTLINE_CLOSURE_CASES"), output = std::getenv("SUNSHINE_TEST_OUTLINE_CLOSURE_OUTPUT");
    const auto bytes = file_bytes(input);
    const auto spec = nlohmann::json::parse(bytes.begin(), bytes.end());
    ASSERT_EQ(spec.at("schema"), 1u);
    ASSERT_EQ(spec.at("inner_source_closure_sha256"), inner_closure_);
    ASSERT_EQ(spec.at("component_source_closure_sha256"), graph_closure_);
    ASSERT_EQ(spec.at("enclosure_source_closure_sha256"), enclosure_closure_);
    ASSERT_EQ(spec.at("driver"), hardware_ ? "hardware" : "warp");
    ASSERT_EQ(spec.at("outline_source_closure_sha256"), ring_closure_);
    ASSERT_EQ(spec.at("inside_source_closure_sha256"), inside_closure_);
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
    nlohmann::json report {{"schema", 1u}, {"driver", hardware_ ? "hardware" : "warp"}, {"adapter", adapter_}, {"probability_shape", {probability_width, probability_height}}, {"test_executable", exeBinding}, {"input_specification_sha256", hash_bytes(bytes)}, {"helper_source_pin", compiled_enclosure_helper_pin}, {"baseline_helper_source_pin", compiled_baseline_helper_pin}, {"inner_source_closure_sha256", inner_closure_}, {"component_source_closure_sha256", graph_closure_}, {"enclosure_source_closure_sha256", enclosure_closure_}, {"outline_helper_source_pin", compiled_ring_helper_pin}, {"outline_source_closure_sha256", ring_closure_}, {"outline_calibration_tag", ring::calibration_tag}, {"outline_calibration_schema", ring::calibration_schema}, {"outline_profile_measurement", ring::profile_measurement}, {"all_semantic_status_UNKNOWN", true}, {"performance_claim", false}, {"cases", nlohmann::json::array()}};
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
      const auto rv = helper_.raw_views();
      arrays.emplace("ring_final.u32", ring_matte());
      arrays.emplace("ring_receipt.u32", read(rv.receipt.Get()));
      arrays.emplace("ring_eligible_inner.u32", read(rv.eligible_inner.Get()));
      arrays.emplace("ring_unit_arguments.u32", read(rv.unit_arguments.Get()));
      arrays.emplace("ring_native_arguments.u32", read(rv.native_arguments.Get()));
      arrays.emplace("outline_final.u32", outline_matte());
      arrays.emplace("calibration_data.u32", read(helper_.mutable_views().calibration_data.Get()));
      arrays.emplace("passable_root_cache.u32", read(cache_snapshot_.resource.Get()));
      arrays.emplace("packed_temp_snapshot.u32", read(packed_snapshot_.resource.Get()));
      arrays.emplace("pre_inside_141_M.u32", read(pre_inside_snapshot_.resource.Get()));
      arrays.emplace("inside_bits.u32", read(helper_.mutable_views().inside_bits.Get()));
      arrays.emplace("family_models_17x8.u32", read(helper_.enclosure_views().family_models.Get()));
      arrays.emplace("inside_arguments.u32", read(helper_.mutable_views().inside_arguments.Get()));
      arrays.emplace("raw_parent_snapshot.u32", read(raw_parent_snapshot_.resource.Get()));
      arrays.emplace("raw_flags_snapshot.u32", read(raw_flags_snapshot_.resource.Get()));
      const bool snapshot_cache_exact_live = arrays.at("passable_root_cache.u32") == read(helper_.mutable_views().passable_root_cache.Get());
      EXPECT_TRUE(snapshot_cache_exact_live);
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
      const auto expected_outline = bound_bytes(item.at("expected_outline").at("final"), n * 4u);
      const bool expected_outline_exact = arrays.at("outline_final.u32") == expected_outline;
      EXPECT_EQ(arrays.at("outline_final.u32"), expected_outline);
      EXPECT_EQ(arrays.at("body.u32"), bound_bytes(item.at("expected_outline").at("body"), n * 4u));
      const auto expected_histogram = words(bound_bytes(item.at("expected_profiles").at("histogram"), 65536u * 4u));
      const auto expected_bands = words(bound_bytes(item.at("expected_profiles").at("bands"), 2048u * 4u));
      const auto data = words(arrays.at("calibration_data.u32"));
      ASSERT_EQ(data.size(), 68608u);
      EXPECT_EQ(std::vector<UINT>(data.begin() + 1024u, data.begin() + 66560u), expected_histogram);
      EXPECT_EQ(std::vector<UINT>(data.begin() + 66560u, data.end()), expected_bands);
      const auto packed = words(arrays.at("packed_temp_snapshot.u32"));
      const auto raw_words = words(arrays.at("ring_final.u32"));
      const auto v3_words = words(bound_bytes(item.at("expected_outline").at("v3_final"), n * 4u));
      const auto e881_words = words(bound_bytes(item.at("expected_outline").at("e881_final"), n * 4u));
      const auto final_words = words(arrays.at("outline_final.u32"));
      const auto code1_words = words(arrays.at("final.u32"));
      const auto before_inside_words = words(arrays.at("pre_inside_141_M.u32")),inside_words = words(arrays.at("inside_bits.u32"));
      const auto selected_body_words = words(arrays.at("selected_unit_bits.u32"));
      const auto candidate_cbc_words = words(arrays.at("receipt.u32"));
      std::size_t measured_rim_pixels = 0u, unknown_raw_pixels = 0u, withdrawn_raw_pixels = 0u, raw_zero_additions = 0u, certified_collar_pixels = 0u, withdrawn_code1_pixels = 0u, packed_certificate_pixels = 0u, residual_certificate_candidates = 0u;
      for (std::size_t q = 0; q < n; ++q) {
        EXPECT_EQ(packed[q] & ~16383u, 0u);
        if ((packed[q] & 8192u) != 0u) { ++measured_rim_pixels; EXPECT_NE(raw_words[q], 0u); }
        const UINT good_inside = data[79u] & ~data[80u] & 255u;
        EXPECT_EQ(final_words[q], UINT(before_inside_words[q] != 0u || (inside_words[q] & good_inside) != 0u));
        const UINT actual_v3 = raw_words[q] || ((packed[q] & 255u) != 0u);
        const UINT actual_e881 = actual_v3 || ((packed[q] & 256u) != 0u);
        EXPECT_EQ(actual_v3, v3_words[q]);
        EXPECT_EQ(actual_e881, e881_words[q]);
        const bool local_certificate = before_inside_words[q] != 0u && (selected_body_words[q] & 255u) == 0u && packed[q] == 0u && (selected_body_words[q] & 65280u) == 0u;
        // CBC5: LOW8 geometry; HIGH8 trusted colour OR adopted known-black-closed nonblack BODY.
        EXPECT_EQ(selected_body_words[q] & ~65535u, 0u);
        const UINT legacy_selected = selected_body_words[q] & 255u;
        const UINT foreground_selected = (selected_body_words[q] >> 8u) & 255u;
        bool guaranteed_support = false;
        for (UINT unit = 0u; unit < 8u; ++unit) {
          const UINT bit = 1u << unit, u = 128u + 32u * unit;
          const bool known_black = candidate_cbc_words[384u + 16u * unit + 2u] == 1u;
          const bool adopted_geometry = candidate_cbc_words[u] != 0u && candidate_cbc_words[u + 1u] == 1u &&
              candidate_cbc_words[u + 3u] > 0u && candidate_cbc_words[u + 6u] == 1u && candidate_cbc_words[u + 29u] == 1u;
          guaranteed_support = guaranteed_support || ((foreground_selected & bit) != 0u && known_black && adopted_geometry) ||
              ((legacy_selected & bit) != 0u && (!known_black || !adopted_geometry));
        }
        if (guaranteed_support || packed[q]) { EXPECT_EQ(final_words[q], 1u); }
        (void)legacy_selected;
        if (local_certificate) { EXPECT_NE(code1_words[q], 0u); }
        packed_certificate_pixels += (packed[q] & 4096u) != 0u;
        residual_certificate_candidates += (selected_body_words[q] & 255u) == 0u && packed[q] == 0u && code1_words[q] != 0u;
        unknown_raw_pixels += (packed[q] & 2048u) != 0u;
        withdrawn_raw_pixels += raw_words[q] != 0u && final_words[q] == 0u;
        raw_zero_additions += raw_words[q] == 0u && final_words[q] != 0u;
        certified_collar_pixels += (packed[q] & 4096u) != 0u || local_certificate;
        withdrawn_code1_pixels += code1_words[q] != 0u && final_words[q] == 0u;
      }
      EXPECT_EQ(data[84u], measured_rim_pixels);
      for (UINT word = 85u; word < 128u; ++word) EXPECT_EQ(data[word], 0u);
      EXPECT_EQ(data[71u], 2048u);
      EXPECT_EQ(data[74u], 4096u);
      // HIGH8 foreground can overlap a local collar certificate. FinalM alone no
      // longer identifies that private localbool. Keep the old comparison labelled,
      // bound the current count, and retain exact ray/counter verification in the
      // source-bound qualification reader; do not silently call the old formula PASS.
      const bool historical_counter75_formula_exact = data[75u] == certified_collar_pixels;
      EXPECT_GE(data[75u], packed_certificate_pixels);
      EXPECT_LE(data[75u], packed_certificate_pixels + residual_certificate_candidates);
      EXPECT_EQ(data[76u], withdrawn_code1_pixels);
      EXPECT_EQ(data[72u], unknown_raw_pixels);
      EXPECT_EQ(data[73u], withdrawn_raw_pixels);
      EXPECT_EQ(data[7u], raw_zero_additions);
      const auto rr = words(arrays.at("ring_receipt.u32"));
      ASSERT_EQ(rr.size(), 512u);
      EXPECT_EQ(rr[0], 0x31474e52u);
      EXPECT_EQ(rr[1], 1u);
      EXPECT_EQ(rr[2], 1u);
      EXPECT_EQ(rr[3], 0u);
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
      const auto dg = helper_.parent_diagnostics();
      const auto rd = helper_.diagnostics();
      nlohmann::json identity {{"schema", 1u}, {"case", item}, {"driver", hardware_ ? "hardware" : "warp"}, {"adapter", adapter_}, {"probability_shape", {probability_width, probability_height}}, {"test_executable", exeBinding}, {"input_specification_sha256", hash_bytes(bytes)}, {"helper_source_pin", compiled_enclosure_helper_pin}, {"baseline_helper_source_pin", compiled_baseline_helper_pin}, {"inner_source_closure_sha256", inner_closure_}, {"component_source_closure_sha256", graph_closure_}, {"enclosure_source_closure_sha256", enclosure_closure_}, {"artifacts", bindings}, {"expected_ring_byte_exact", expected_ring_exact}, {"expected_outline_byte_exact", expected_outline_exact}, {"snapshot_cache_exact_live", snapshot_cache_exact_live}, {"appendix_recordings", rd.recordings}, {"appendix_replays", rd.replays}, {"appendix_direct_emissions", rd.direct_emissions}, {"recordings", dg.recordings}, {"replays", dg.replays}, {"direct_emissions", dg.direct_emissions}, {"expected_baseline_and_candidate_byte_exact", expected_masks_exact}, {"expected_receipt_present", item.at("expected").contains("receipt")}, {"expected_receipt_byte_exact", expected_receipt_exact}, {"outline_helper_source_pin", compiled_ring_helper_pin}, {"outline_source_closure_sha256", ring_closure_}, {"outline_calibration_tag", ring::calibration_tag}, {"outline_calibration_schema", ring::calibration_schema}, {"outline_profile_measurement", ring::profile_measurement}, {"all_semantic_status_UNKNOWN", true}, {"performance_claim", false}};
      identity["typed_contract"] = "INSIDE145_CBC5_CFB7_LOW8_GEOMETRY_HIGH8_TRUSTED_COLOR_OR_CLOSED_BODY_N4";
      identity["CBC_output_schema"] = 5u;
      identity["F2_packed_allowed"] = 16383u;identity["measured_rim_bit"] = 8192u;identity["measured_rim_counter_word"] = 84u;
      identity["inside_source_closure_sha256"] = inside_closure_;
      identity["inside_snapshots_fresh_active_only"] = true;
      identity["historical_expected_arrays_unchanged"] = true;
      identity["historical_counter75_formula_exact"] = historical_counter75_formula_exact;
      identity["historical_masks_are_comparison_only_for_new_policy"] = true;
      const auto text = identity.dump(2) + '\n';
      std::ofstream f(folder / "current_outline_support_closure_identity.json", std::ios::binary);
      f << text;
      require(f.good());
      report["cases"].push_back({{"index", k + 1u}, {"name", item.at("name")}, {"identity_sha256", hash_bytes(std::vector<char>(text.begin(), text.end()))}});
    }
    report["typed_contract"] = "INSIDE145_CBC5_CFB7_LOW8_GEOMETRY_HIGH8_TRUSTED_COLOR_OR_CLOSED_BODY_N4";
    report["CBC_output_schema"] = 5u;
    report["inside_source_closure_sha256"] = inside_closure_;
    std::ofstream f(output / "current_outline_support_closure_replay_identity.json", std::ios::binary);
    f << report.dump(2) << '\n';
    require(f.good());
  }
  // One ordinary-input native AA lifecycle. No models, masks or history pixels are uploaded.
  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, NativeAASeedRootZeroDuplicateStaleResetEmptyLifecycle) {
    // The canonical fixture leaves the optional stability program disabled;
    // this lifecycle test explicitly owns its three shaders for the whole TEST.
    const char *history_stability_root=std::getenv("SUNSHINE_TEST_BLACK_STABILITY_SHADER_ROOT");
    const char *history_stability_pin=std::getenv("SUNSHINE_TEST_BLACK_STABILITY_SHADER_PIN");
    ASSERT_NE(history_stability_root,nullptr);ASSERT_NE(history_stability_pin,nullptr);
    std::array<ComPtr<ID3D11ComputeShader>,3u> history_stability_shaders;
    const std::array history_stability_specs{
      cache::shader_spec{"host_sbs_current_outline_support_band_cs.hlsl","private_cache_begin_main","cs_5_0"},
      cache::shader_spec{"host_sbs_current_outline_support_band_cs.hlsl","private_cache_validate_main","cs_5_0"},
      cache::shader_spec{"host_sbs_current_outline_support_band_cs.hlsl","private_cache_commit_main","cs_5_0"}};
    load_shaders(history_stability_root,history_stability_pin,history_stability_specs,history_stability_shaders);
    for(UINT i=0u;i<3u;++i)program_.stability_shaders[i]=history_stability_shaders[i].Get();
    RecordProperty("AA_history_stability_source_closure_sha256",history_stability_pin);
    author_band(5u,600u);
    rect(73u,60u,74u,61u,{64u,64u,64u,255u}); // one visible AA fringe: K0 -> 64 -> background128
    upload_pixels_from_origin();ASSERT_TRUE(run()) << error_;
    const auto seeded=aa_history_snapshot("AA-seeded");const auto sh=words(seeded.header),sf=words(seeded.history),sp=words(read(packed_snapshot_.resource.Get()));
    const auto meta=receipt(),data=calibration();const std::size_t q=60u*W_+73u,p=60u*W_+74u;
    ASSERT_EQ(meta[386u],1u);ASSERT_EQ(data[141u],1u);ASSERT_EQ(data[139u],5u);
    ASSERT_EQ(sh[2u],1u);ASSERT_EQ(sh[146u],1u);
    ASSERT_GT(std::count_if(sf.begin(),sf.end(),[](UINT x){return(x&2u)!=0u;}),0);
    ASSERT_NE(sp[q]&768u,0u);ASSERT_NE(sf[q]&2u,0u);ASSERT_NE(sp[p]&1u,0u);
    const auto initialM=outline_matte();
    ASSERT_TRUE(run()) << error_;
    const auto duplicate=aa_history_snapshot("exact-duplicate");const auto dh=words(duplicate.header);
    EXPECT_EQ(duplicate.history,seeded.history);EXPECT_EQ(duplicate.rgb,seeded.rgb);EXPECT_EQ(outline_matte(),initialM);
    EXPECT_EQ(dh[104u],sh[104u]);EXPECT_EQ(dh[2u],1u);EXPECT_EQ(dh[150u],0u);EXPECT_EQ(dh[151u],1u);
    author(false,601u,0u,false,false,false,256u,128u,3u);root_words_[60u]=0u;upload();ASSERT_TRUE(run()) << error_;
    const auto held=aa_history_snapshot("root0-new-input");EXPECT_EQ(held.history,duplicate.history);EXPECT_EQ(held.header,duplicate.header);EXPECT_EQ(held.rgb,duplicate.rgb);EXPECT_EQ(outline_matte(),initialM);
    // A valid stale EMPTY cannot revoke the newer committed pair or its high-water identity.
    author(false,599u,0u,false,false,false,256u,128u,3u);ASSERT_TRUE(run()) << error_;
    const auto stale=aa_history_snapshot("stale-EMPTY");const auto th=words(stale.header);
    EXPECT_EQ(stale.history,seeded.history);EXPECT_EQ(stale.rgb,seeded.rgb);EXPECT_EQ(th[104u],sh[104u]);EXPECT_EQ(th[2u],1u);
    EXPECT_NE(th[103u]&4u,0u);EXPECT_EQ(th[56u],600u);EXPECT_EQ(th[60u],1000600u);EXPECT_EQ(th[153u],0u);
    // Fresh nonempty reset replaces the pair with CURRENT source/geometry, without the fringe.
    author_band(5u,602u);cb_words_[18u]=1u;upload();ASSERT_TRUE(run()) << error_;
    const auto reset=aa_history_snapshot("fresh-nonempty-reset");const auto rh=words(reset.header),rf=words(reset.history);
    EXPECT_EQ(rh[2u],1u);EXPECT_EQ(rh[146u],1u);EXPECT_EQ(rh[104u],sh[104u]+1u);EXPECT_NE(rh[103u]&16u,0u);EXPECT_EQ(rh[153u],0u);
    EXPECT_EQ(rf[q]&2u,0u);
    author(false,603u,0u,false,false,false,256u,128u,3u);ASSERT_TRUE(run()) << error_;
    const auto empty=aa_history_snapshot("fresh-EMPTY");const auto eh=words(empty.header);
    EXPECT_EQ(eh[2u],0u);EXPECT_EQ(eh[153u],0u);EXPECT_EQ(eh[104u],rh[104u]);EXPECT_EQ(empty.history,reset.history);EXPECT_EQ(empty.rgb,reset.rgb);
    EXPECT_EQ(outline_matte(),std::vector<char>(W_*H_*4u,0));
  }

}  // namespace
#endif

#ifdef _WIN32
#include "current_outline_support_cost_cases.inc"
#endif
