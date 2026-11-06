/** Private f2a9 outline-support closure port. Private source fixture; no runtime activation or GT inputs. */
#include <gtest/gtest.h>
#ifdef _WIN32
  #include "src/crypto.h"
  #include "src/generated/depth_coordinate_v2_contract.h"
  #include "src/host_sbs_current_outline_support_closure_gpu.h"
  #include "src/host_sbs_current_outline_runtime_programs.h"
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
  #include <limits>
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
  constexpr const char *compiled_hole_helper_pin = subtitle_test_source_pins::current_mask_holes;
  constexpr const char *compiled_foreground_helper_pin = subtitle_test_source_pins::current_foreground_edge;
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
      const auto *hole_root = std::getenv("SUNSHINE_TEST_CURRENT_HOLE_SHADER_ROOT");
      const auto *hole_pin = std::getenv("SUNSHINE_TEST_CURRENT_HOLE_SHADER_PIN");
      if (hole_root || hole_pin) {
        require(hole_root && hole_pin);
        load_shaders(hole_root, hole_pin, models::current_outline_runtime::hole_specs, hole_shaders_);
        hole_closure_ = hole_pin;
        for (UINT i = 0u; i < hole_shaders_.size(); ++i) program_.hole_shaders[i] = hole_shaders_[i].Get();
        if (const auto *pin = std::getenv("SUNSHINE_TEST_CURRENT_HOLE_HEADER_PIN")) require(std::string(pin) == compiled_hole_helper_pin);
        RecordProperty("hole_source_closure", hole_closure_);
        RecordProperty("hole_helper_source_pin", compiled_hole_helper_pin);
      }
      const auto *hp = std::getenv("SUNSHINE_TEST_OUTLINE_CLOSURE_HEADER_PIN");
      const auto *coverage_root = std::getenv("SUNSHINE_TEST_STROKE_COVERAGE_SHADER_ROOT");
      const auto *coverage_pin = std::getenv("SUNSHINE_TEST_STROKE_COVERAGE_SHADER_PIN");
      if (coverage_root || coverage_pin) {
        require(coverage_root && coverage_pin);
        load_shaders(coverage_root, coverage_pin, models::current_outline_runtime::coverage_specs, coverage_shaders_);
        coverage_closure_ = coverage_pin;
        for (UINT i = 0u; i < coverage_shaders_.size(); ++i) program_.coverage_shaders[i] = coverage_shaders_[i].Get();
        program_.coverage_diagnostics = true; // Native truth/export fixtures opt in; production stays off.
        RecordProperty("stroke_coverage_source_closure", coverage_closure_);
      }
      const auto *foreground_root=std::getenv("SUNSHINE_TEST_FOREGROUND_EDGE_SHADER_ROOT");
      const auto *foreground_pin=std::getenv("SUNSHINE_TEST_FOREGROUND_EDGE_SHADER_PIN");
      if(foreground_root||foreground_pin) {
        require(foreground_root&&foreground_pin);
        load_shaders(foreground_root,foreground_pin,models::current_outline_runtime::foreground_edge_specs,foreground_edge_shaders_);
        foreground_edge_closure_=foreground_pin;
        RecordProperty("foreground_edge_source_closure",foreground_edge_closure_);
        RecordProperty("foreground_edge_helper_source_pin",compiled_foreground_helper_pin);
        if(const auto *pin=std::getenv("SUNSHINE_TEST_FOREGROUND_EDGE_HEADER_PIN"))require(std::string(pin)==compiled_foreground_helper_pin);
        // Loading makes the optional shader available; old fixtures stay OFF.
        // New owner tests and the explicit Saved FG replay opt in separately.
      }
      const auto *trusted_root = std::getenv("SUNSHINE_TEST_TRUSTED_INNER_SHADER_ROOT");
      const auto *trusted_pin = std::getenv("SUNSHINE_TEST_TRUSTED_INNER_SHADER_PIN");
      if (trusted_root || trusted_pin) {
        require(trusted_root && trusted_pin);
        load_shaders(trusted_root, trusted_pin, models::current_outline_runtime::trusted_inner_specs, trusted_inner_shaders_);
        RecordProperty("trusted_inner_source_closure", trusted_pin);
        // Availability alone keeps every older native fixture on its exact old route.
      }
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

    // Author INPUT layers, never a replica of production mask/render decisions.
    // The same scene and black/white caption geometry yield two independently
    // declared truths: an open transparent scene notch, or faint black ink.
    enum class authored_scene_pattern {flat128, affine_gray, affine_rgb, checker};
    static const char *authored_scene_pattern_name(authored_scene_pattern pattern) {
      switch (pattern) {
        case authored_scene_pattern::flat128: return "flat128";
        case authored_scene_pattern::affine_gray: return "affine_gray";
        case authored_scene_pattern::affine_rgb: return "affine_rgb";
        case authored_scene_pattern::checker: return "checker";
      }
      throw std::runtime_error("Unknown predeclared scene challenge");
    }
    static pixel8 authored_scene_pixel(authored_scene_pattern pattern, UINT x, UINT y) {
      // Independently authored scene input: no OCR/BODY/Raw/result dependency.
      // Quantized affine surfaces and the fixed4px checker stay within RGB8.
      switch (pattern) {
        case authored_scene_pattern::flat128: return {128u, 128u, 128u, 255u};
        case authored_scene_pattern::affine_gray: {
          const auto value = static_cast<std::uint8_t>(112u + (x + 2u * y) / 16u);
          return {value, value, value, 255u};
        }
        case authored_scene_pattern::affine_rgb:
          return {static_cast<std::uint8_t>(112u + (x + 2u * y) / 16u),
            static_cast<std::uint8_t>(120u + (2u * x + y) / 24u),
            static_cast<std::uint8_t>(128u + (x + y) / 32u), 255u};
        case authored_scene_pattern::checker: {
          const auto value = static_cast<std::uint8_t>(128u + 16u * (((x / 4u) + (y / 4u)) & 1u));
          return {value, value, value, 255u};
        }
      }
      throw std::runtime_error("Unknown predeclared scene challenge");
    }
    void expect_authored_open_notch_alpha(UINT alpha, UINT thickness,
      authored_scene_pattern scene_pattern = authored_scene_pattern::flat128,
      UINT challenge_frame = 0u, const char *challenge_prefix = nullptr) {
      ASSERT_TRUE(alpha == 0u || alpha == 8u || alpha == 128u);
      ASSERT_TRUE(thickness == 3u || thickness == 4u);
      const UINT frame = challenge_frame != 0u ? challenge_frame : 1400u + thickness * 16u + alpha;
      author(true, frame, 0u, false, false, false, 256u, 128u, 3u);
      const std::size_t count = static_cast<std::size_t>(W_) * H_;
      const pixel8 scene_colour {128u, 128u, 128u, 255u};
      std::vector<pixel8> scene(count, scene_colour), ink(count, {0u, 0u, 0u, 255u});
      std::vector<std::uint8_t> ink_alpha(count, 0u);
      // Default flat128 bypasses this branch: existing authored input bytes stay exact.
      if (scene_pattern != authored_scene_pattern::flat128) {
        for (UINT y = 0u; y < H_; ++y) for (UINT x = 0u; x < W_; ++x)
          scene[y * W_ + x] = authored_scene_pixel(scene_pattern, x, y);
      }
      auto layer_rect = [&](UINT l, UINT t, UINT r, UINT b, pixel8 colour, UINT a) {
        require(l < r && t < b && r <= W_ && b <= H_ && a <= 255u);
        for (UINT y = t; y < b; ++y) for (UINT x = l; x < r; ++x) {
          ink[y * W_ + x] = colour;ink_alpha[y * W_ + x] = static_cast<std::uint8_t>(a);
        }
      };
      const pixel8 black {0u, 0u, 0u, 255u}, white {255u, 255u, 255u, 255u};
      layer_rect(79u - thickness, 48u - thickness, 111u + thickness, 88u + thickness, black, 255u);
      layer_rect(79u, 48u, 111u, 88u, white, 255u);
      layer_rect(160u - thickness, 48u - thickness, 208u + thickness, 88u + thickness, black, 255u);
      layer_rect(160u, 48u, 208u, 88u, white, 255u);
      const UINT px = thickness == 4u ? 73u : 74u, rx = 77u, qx = px + 1u;
      layer_rect(px, 59u, rx, 62u, black, 255u);
      layer_rect(px, 58u, rx, 59u, black, 255u); // Remote crown joins the black endpoints.
      layer_rect(qx, 59u, rx, 62u, black, alpha);
      // This is the declared encoded-SDR INPUT authoring convention. Keeping the
      // exact layers makes alpha truth independent of detector/topology output.
      origin_pixels_ = scene;
      for (std::size_t i = 0u; i < count; ++i) for (UINT c = 0u; c < 3u; ++c) {
        const UINT a = ink_alpha[i];
        origin_pixels_[i][c] = static_cast<std::uint8_t>((a * ink[i][c] + (255u - a) * scene[i][c] + 127u) / 255u);
      }
      const auto label = (challenge_prefix ? std::string(challenge_prefix) + "_" + authored_scene_pattern_name(scene_pattern) :
        std::string("authored_open_notch")) + "_t" + std::to_string(thickness) + "_a" + std::to_string(alpha);
      RecordProperty(label + "_scene_rgba8_sha256", hash_bytes(raw(scene)));
      RecordProperty(label + "_ink_rgba8_sha256", hash_bytes(raw(ink)));
      RecordProperty(label + "_ink_alpha8_sha256", hash_bytes(raw(ink_alpha)));
      RecordProperty(label + "_source_rgba8_sha256", hash_bytes(raw(origin_pixels_)));
      upload_pixels_from_origin();
      ASSERT_TRUE(run()) << error_;
      expect_ready();expect_ring_ready();expect_outline_accounting();
      const auto cbc = receipt(), raw_meta = ring_receipt(), f2 = calibration();
      ASSERT_EQ(cbc.size(), 512u);ASSERT_EQ(raw_meta.size(), 512u);ASSERT_EQ(f2.size(), 68608u);
      const auto b = helper_.baseline_views();const auto rv = helper_.raw_views();
      const auto original = words(read(b.body.Get())), selected = words(body());
      const auto code1 = words(matte()), raw_m = words(ring_matte()), final_m = words(outline_matte());
      const auto packed = words(read(packed_snapshot_.resource.Get()));
      const auto parent = words(read(b.parent.Get())), flags = words(read(b.root_flags.Get()));
      const auto passable = words(read(b.eligible.Get())), inner = words(read(rv.eligible_inner.Get()));
      const auto cache_words = root_cache(), cover = words(read(helper_.enclosure_views().cover_bits.Get()));
      // Optional whole-array evidence is captured BEFORE semantic/geometry
      // prerequisites, so a failed control is diagnosable without changing it.
      if (const char *output = std::getenv("SUNSHINE_TEST_AUTHORED_ALPHA_OUTPUT_DIR")) {
        const auto folder = std::filesystem::path(output) / label;
        require(!std::filesystem::exists(folder));
        require(std::filesystem::create_directories(folder));
        nlohmann::json artifacts = nlohmann::json::object();
        auto emit = [&](const std::string &name, const auto &values) {
          const auto bytes = raw(values);const auto path = folder / name;
          std::ofstream stream(path, std::ios::binary);
          stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));stream.close();require(bool(stream));
          artifacts[name] = {{"path", path.string()}, {"bytes", bytes.size()}, {"sha256", hash_bytes(bytes)}};
        };
        emit("scene.rgba8", scene);emit("ink.rgba8", ink);emit("ink_alpha.u8", ink_alpha);emit("source.rgba8", origin_pixels_);
        emit("original_BODY.u32", original);emit("selected_BODY.u32", selected);emit("Code1.u32", code1);
        emit("RawM.u32", raw_m);emit("final_M.u32", final_m);emit("packed.u32", packed);
        emit("RawParent.u32", parent);emit("RawFlags.u32", flags);emit("RawPassable.u32", passable);
        emit("InnerEligible.u32", inner);emit("PassableCache.u32", cache_words);emit("CoverBits.u32", cover);
        emit("CBCMeta.u32", cbc);emit("RawMeta.u32", raw_meta);emit("F2Data.u32", f2);
        nlohmann::json manifest {{"schema", 1u}, {"kind", "authored_input_layers_and_actual_gpu_diagnostics"},
          {"native_shape", {W_, H_}}, {"frame", cb_words_[12u]}, {"thickness", thickness}, {"target_alpha_u8", alpha},
          {"source_authoring", "encoded SDR integer round((alpha*ink+(255-alpha)*scene)/255), +127 then /255"},
          {"target_rect", {qx, 59u, rx, 62u}}, {"artifacts", artifacts}};
        if (challenge_prefix) {
          manifest["scene_pattern"] = authored_scene_pattern_name(scene_pattern);
          manifest["development_challenge_only"] = true;
          manifest["predeclared_matrix_sha256"] = "ddc51ed195d87c44a9e3c69ef9dfb64818f22d6fec0bd559f3ee1e731310d0aa";
          manifest["truth_not_producer_derived"] = true;
        }
        std::ofstream stream(folder / "manifest.json", std::ios::binary);
        stream << manifest.dump(2) << '\n';stream.close();require(bool(stream));
        RecordProperty(label + "_artifact_manifest", (folder / "manifest.json").string());
      }
      record_uniform_black_state(label);
      ASSERT_EQ(original.size(), count);ASSERT_EQ(selected.size(), count);
      for (const auto *v : {&code1, &raw_m, &final_m, &packed, &passable, &inner, &cache_words, &cover}) {
        ASSERT_EQ(v->size(), count);
      }
      ASSERT_EQ(parent.size(), count + 1u);ASSERT_EQ(flags.size(), parent.size());
      expect_uniform_known(cbc, 0u, black, 3u);
      ASSERT_EQ(cbc[128u], 1u);ASSERT_EQ(cbc[129u], 1u);ASSERT_GT(cbc[131u], 0u);
      ASSERT_EQ(cbc[134u], 1u);ASSERT_EQ(cbc[157u], 1u);
      ASSERT_EQ(raw_meta[6u], 1u);ASSERT_EQ(raw_meta[128u], 1u);ASSERT_EQ(raw_meta[129u], 1u);
      ASSERT_EQ(f2[139u], thickness);ASSERT_EQ(f2[141u], 1u);ASSERT_EQ(f2[164u], 0u);
      EXPECT_EQ(selected, original);
      // Keep BOTH white interiors intact and independently authored as opaque ink.
      for (std::size_t i = 0u; i < count; ++i) if (ink[i] == white && ink_alpha[i] == 255u) {
        ASSERT_EQ(selected[i], 1u) << i % W_ << ',' << i / W_;
        EXPECT_EQ(final_m[i], 1u) << i % W_ << ',' << i / W_;
      }
      auto root = [&](UINT pixel) {
        UINT id = pixel + 1u;
        for (std::size_t hops = 0u; hops < parent.size(); ++hops) {
          require(id > 0u && id < parent.size());
          const UINT next = parent[id];require(next > 0u && next <= id);
          if (next == id) return id;
          id = next;
        }
        throw std::runtime_error("Non-convergent authored-alpha Raw graph");
      };
      const UINT p = 60u * W_ + px, r = 60u * W_ + rx;
      const UINT p_root = root(p), r_root = root(r);
      ASSERT_EQ(p_root, r_root);ASSERT_EQ(flags[p_root] & 3u, 1u);
      ASSERT_EQ(passable[p], 1u);ASSERT_EQ(passable[r], 1u);
      ASSERT_EQ(raw_m[p], 1u);ASSERT_EQ(raw_m[r], 1u);
      ASSERT_EQ(selected[p], 0u);ASSERT_EQ(code1[p], 0u);ASSERT_EQ(packed[p] & 1u, 0u);
      ASSERT_EQ(packed[r] & 1u, 1u); // Genuine opposite measured BAND, never an AA-created donor.
      constexpr UINT root_mask = 0x3fffffffu, enclosed = 0x80000000u;
      const UINT exterior = 63u * W_ + qx;
      ASSERT_EQ(ink_alpha[exterior], 0u);ASSERT_EQ(origin_pixels_[exterior], scene[exterior]);
      ASSERT_EQ(selected[exterior], 0u);ASSERT_EQ(inner[exterior], 0u);
      const UINT exterior_tag = cache_words[exterior] & (root_mask | enclosed);
      ASSERT_GT(exterior_tag & root_mask, 0u);ASSERT_EQ(exterior_tag & enclosed, 0u);
      nlohmann::json targets = nlohmann::json::array();
      for (UINT x = qx; x < rx; ++x) {
        const UINT q = 60u * W_ + x;
        ASSERT_EQ(scene[q], authored_scene_pixel(scene_pattern, x, 60u));
        ASSERT_EQ(ink[q], black);ASSERT_EQ(ink_alpha[q], alpha);
        pixel8 expected_source = scene[q];
        for (UINT c = 0u; c < 3u; ++c) expected_source[c] =
          static_cast<std::uint8_t>(((255u - alpha) * scene[q][c] + 127u) / 255u);
        ASSERT_EQ(origin_pixels_[q], expected_source);
        ASSERT_EQ(selected[q], 0u);ASSERT_EQ(passable[q], 0u);ASSERT_EQ(inner[q], 0u);
        ASSERT_EQ(packed[q] & 255u, 0u);ASSERT_EQ(cover[q] & 1u, 1u);
        ASSERT_EQ(code1[q], x == 76u ? 1u : 0u);ASSERT_EQ(raw_m[q], code1[q]);
        ASSERT_EQ(cache_words[q] & (root_mask | enclosed), exterior_tag);
        const UINT background_root = cache_words[q] & root_mask;
        ASSERT_LE(background_root, count);ASSERT_EQ(cover[background_root - 1u] & 1u, 1u);
        ASSERT_EQ(cache_words[background_root - 1u] & (root_mask | enclosed), exterior_tag);
        // A physical route certificate must not turn declared alpha-zero scene
        // into glyph coverage; genuinely faint authored ink must remain active.
        EXPECT_EQ(final_m[q], alpha == 0u ? 0u : 1u) << x << ",60 alpha=" << alpha;
        targets.push_back({{"native", {x, 60u}}, {"authored_alpha_u8", alpha},
          {"source_RGB8", {expected_source[0], expected_source[1], expected_source[2]}}, {"actual_Code1", code1[q]},
          {"actual_RawM", raw_m[q]}, {"actual_packed", packed[q]}, {"actual_M", final_m[q]}});
      }
      RecordProperty(label + "_final_M_sha256", hash_bytes(raw(final_m)));
      RecordProperty(label + "_targets", targets.dump());
    }

    template<class Emit>
    void emit_current_hole_evidence(Emit &&emit, bool include_family = true) {
      if (include_family) emit("FamilyModels17x8.u32", words(read(helper_.enclosure_views().family_models.Get())));
      if (helper_.mask_holes_enabled()) {
        emit("HoleProof.u32", words(read(helper_.mask_hole_proof())));
        emit("HoleParent.u32", words(read(helper_.mask_hole_parents())));
        emit("HoleFlags.u32", words(read(helper_.mask_hole_flags())));
      }
      if (helper_.stroke_coverage_enabled()) {
        emit("CoverageOffers.u32", words(read(helper_.stroke_coverage_offers())));
      }
    }

    nlohmann::json current_hole_policy() const {
      return {{"source_closure",hole_closure_},{"compiled_helper_pin",compiled_hole_helper_pin},
        {"phase_resources_enabled",helper_.mask_holes_enabled()},{"shader_count",hole_shaders_.size()},
        {"coverage_source_closure",coverage_closure_},{"coverage_enabled",helper_.stroke_coverage_enabled()}};
    }

    enum class authored_stroke_hole_role {opaque_tint, O_counter, blue_scene_cutout};
    void expect_authored_stroke_hole(authored_stroke_hole_role role, UINT thickness, pixel8 tint,
      authored_scene_pattern pattern, UINT frame) {
      const bool positive = role == authored_stroke_hole_role::opaque_tint;
      const bool counter = role == authored_stroke_hole_role::O_counter;
      const auto role_name = positive ? "opaque_tint" : counter ? "O_counter" : "blue_scene_cutout";
      const auto label = std::string("stroke_hole_v1_") + role_name + "_" + authored_scene_pattern_name(pattern) +
        "_t" + std::to_string(thickness) + "_rgb" + std::to_string(tint[0]) + "_" + std::to_string(tint[1]) + "_" + std::to_string(tint[2]);
      const char *matrix_pin = role == authored_stroke_hole_role::blue_scene_cutout ?
        "ed085c3a6b03ad04a13eb950c69c8fc5890b03a00d142fdb687f3c97c3d74a66" :
        "4574f802a71066cca0d3dfc2fe0ff23d34c1e6f0c5da0830bc13a551b87c28ab";
      author(true, frame, 0u, false, false, false, 256u, 128u, 3u);
      const std::size_t count = std::size_t(W_) * H_;
      const pixel8 black {3u, 4u, 8u, 255u}, white {245u, 245u, 245u, 255u};
      std::vector<pixel8> scene(count), ink(count, black);
      std::vector<std::uint8_t> alpha(count, 0u);
      for (UINT y = 0u; y < H_; ++y) for (UINT x = 0u; x < W_; ++x) scene[y * W_ + x] = authored_scene_pixel(pattern, x, y);
      const auto paint = [&](UINT l, UINT t, UINT r, UINT b, pixel8 rgb, UINT a) {
        require(l < r && t < b && r <= W_ && b <= H_ && a <= 255u);
        for (UINT y = t; y < b; ++y) for (UINT x = l; x < r; ++x) {ink[y * W_ + x] = rgb;alpha[y * W_ + x] = std::uint8_t(a);}
      };
      paint(79u-thickness,48u-thickness,111u+thickness,88u+thickness,black,255u);
      paint(79u,48u,111u,88u,white,255u);
      paint(160u-thickness,48u-thickness,208u+thickness,88u+thickness,black,255u);
      paint(160u,48u,208u,88u,white,255u);
      // Real minority dark anchors; a uniform nonliteral palette would be UNKNOWN.
      // This witness and its alpha are recorded, not injected model metadata.
      paint(118u,46u,126u,58u,{32u,32u,32u,255u},255u);
      paint(120u,48u,124u,56u,white,255u);
      const std::array<UINT,4u> target = counter ? std::array<UINT,4u>{94u,64u,97u,67u} : std::array<UINT,4u>{75u,63u,78u,66u};
      std::vector<UINT> primary, corridor, outside;
      const auto enumerate = [&](UINT l,UINT t,UINT r,UINT b,auto &points) {for(UINT y=t;y<b;++y)for(UINT x=l;x<r;++x)points.push_back(y*W_+x);};
      if (counter) {
        // A genuine O counter: original inner black border has the SAME width.
        // Every transparent target is farther than t from its white ring.
        paint(94u-thickness,64u-thickness,97u+thickness,67u+thickness,black,255u);
      } else {
        // A one-pixel-THICK corridor reaches the outer source background for both t5/t6.
        paint(79u-thickness,64u,75u,65u,tint,positive ? 255u : 0u);
        enumerate(79u-thickness,64u,75u,65u,corridor);
      }
      paint(target[0],target[1],target[2],target[3],tint,positive ? 255u : 0u);
      enumerate(target[0],target[1],target[2],target[3],primary);
      if (!positive) for (UINT q : primary) scene[q] = tint;
      if (role == authored_stroke_hole_role::blue_scene_cutout) for (UINT q : corridor) scene[q] = tint;
      enumerate(56u,63u,59u,66u,outside);
      for (UINT q : outside) scene[q] = tint;
      origin_pixels_ = scene;
      for (std::size_t q=0u;q<count;++q)for(UINT c=0u;c<3u;++c) {
        const UINT a=alpha[q];origin_pixels_[q][c]=std::uint8_t((a*ink[q][c]+(255u-a)*scene[q][c]+127u)/255u);
      }
      RecordProperty(label+"_matrix_sha256",matrix_pin);
      RecordProperty(label+"_scene_sha256",hash_bytes(raw(scene)));RecordProperty(label+"_ink_sha256",hash_bytes(raw(ink)));
      RecordProperty(label+"_alpha_sha256",hash_bytes(raw(alpha)));RecordProperty(label+"_source_sha256",hash_bytes(raw(origin_pixels_)));
      std::filesystem::path folder;
      nlohmann::json artifacts=nlohmann::json::object();
      if(const auto *output=std::getenv("SUNSHINE_TEST_AUTHORED_STROKE_HOLE_OUTPUT_DIR")) {
        folder=std::filesystem::path(output)/label;require(!std::filesystem::exists(folder));require(std::filesystem::create_directories(folder));
      }
      const auto emit=[&](const char *name,const auto &values) {
        if(folder.empty())return;const auto bytes=raw(values);const auto path=folder/name;
        std::ofstream f(path,std::ios::binary);f.write(bytes.data(),std::streamsize(bytes.size()));f.close();require(bool(f));
        artifacts[name]={{"path",path.string()},{"bytes",bytes.size()},{"sha256",hash_bytes(bytes)}};
      };
      emit("scene.rgba8",scene);emit("ink.rgba8",ink);emit("ink_alpha.u8",alpha);emit("source.rgba8",origin_pixels_);
      emit("P.f32",probability_words_);emit("S.u32",support_words_);emit("C24.u32",cb_words_);emit("OCR.u32",ocr_words_);
      emit("SCT.u32",sct_words_);emit("UIP.u32",uip_words_);emit("root.u32",root_words_);
      upload_pixels_from_origin();
      const bool dispatched=run();
      nlohmann::json manifest{{"schema",1u},{"kind","independently_authored_stroke_hole"},{"matrix_sha256",matrix_pin},
        {"role",role_name},{"scene_pattern",authored_scene_pattern_name(pattern)},{"frame",frame},{"thickness",thickness},
        {"native_shape",{W_,H_}},{"source_authoring","integer encoded-SDR round((alpha*ink+(255-alpha)*scene)/255), +127 then/255"},
        {"primary_target_rect",target},{"primary_points",primary},{"corridor_points",corridor},{"outside_scene_points",outside},
        {"actual_dispatch_queued",dispatched},{"dispatch_error",error_},{"calibration_witness_is_not_primary_target",true}};
      const auto write_manifest=[&]() {if(!folder.empty()){manifest["artifacts"]=artifacts;std::ofstream f(folder/"manifest.json",std::ios::binary);f<<manifest.dump(2)<<'\n';f.close();require(bool(f));}};
      if(!dispatched){write_manifest();ASSERT_TRUE(dispatched)<<error_;return;}
      const auto cbc=receipt(),raw_meta=ring_receipt(),f2=calibration();
      const auto b=helper_.baseline_views();const auto rv=helper_.raw_views();
      const auto original=words(read(b.body.Get())),selected=words(body()),code1=words(matte()),raw_m=words(ring_matte()),final_m=words(outline_matte());
      const auto packed=words(read(packed_snapshot_.resource.Get())),parent=words(read(b.parent.Get())),flags=words(read(b.root_flags.Get()));
      const auto passable=words(read(b.eligible.Get())),inner=words(read(rv.eligible_inner.Get())),cache_words=root_cache();
      const auto e=helper_.enclosure_views();const auto cover=words(read(e.cover_bits.Get())),selected_bits=words(read(e.selected_bits.Get()));
      emit("CBCMeta.u32",cbc);emit("RawMeta.u32",raw_meta);emit("F2Data.u32",f2);emit("original_BODY.u32",original);
      emit("selected_BODY.u32",selected);emit("Code1.u32",code1);emit("RawM.u32",raw_m);emit("final_M.u32",final_m);
      emit("packed.u32",packed);emit("RawParent.u32",parent);emit("RawFlags.u32",flags);emit("RawPassable.u32",passable);
      emit("InnerEligible.u32",inner);emit("PassableCache.u32",cache_words);emit("CoverBits.u32",cover);emit("SelectedBits.u32",selected_bits);
      emit_current_hole_evidence(emit);manifest["hole_policy"]=current_hole_policy();
      // Capture every actual map before publication, model or geometry prerequisites.
      write_manifest();
      ASSERT_EQ(cbc.size(),512u);ASSERT_EQ(raw_meta.size(),512u);ASSERT_EQ(f2.size(),68608u);
      for(const auto *v:{&original,&selected,&code1,&raw_m,&final_m,&packed,&passable,&inner,&cache_words,&cover,&selected_bits})ASSERT_EQ(v->size(),count);
      ASSERT_EQ(parent.size(),count+1u);ASSERT_EQ(flags.size(),parent.size());
      nlohmann::json summaries=nlohmann::json::object();
      const auto truth_counts=[&](const auto &points) {
        UINT a0_m1=0u,a255_m0=0u;for(UINT q:points){a0_m1+=alpha[q]==0u&&final_m[q]!=0u;a255_m0+=alpha[q]==255u&&final_m[q]==0u;}
        return nlohmann::json{{"points",points.size()},{"alpha0_M1",a0_m1},{"alpha255_M0",a255_m0}};
      };
      summaries["primary"]=truth_counts(primary);summaries["corridor"]=truth_counts(corridor);summaries["outside"]=truth_counts(outside);
      manifest["all_attempt_literal_truth_counts"]=summaries;manifest["prerequisites_not_yet_asserted"]=true;write_manifest();
      RecordProperty(label+"_literal_truth_counts",summaries.dump());record_uniform_black_state(label);
      expect_ready();expect_ring_ready();expect_outline_accounting();EXPECT_EQ(selected,original);
      ASSERT_EQ(cbc[385u],1u);ASSERT_EQ(cbc[386u],1u);ASSERT_EQ(cbc[387u],1u);
      ASSERT_EQ(cbc[128u],1u);ASSERT_EQ(cbc[129u],1u);ASSERT_GT(cbc[131u],0u);ASSERT_EQ(cbc[134u],1u);ASSERT_EQ(cbc[157u],1u);
      ASSERT_EQ(raw_meta[6u],1u);ASSERT_EQ(raw_meta[128u],1u);ASSERT_EQ(raw_meta[129u],1u);
      ASSERT_EQ(f2[139u],thickness);ASSERT_EQ(f2[141u],1u);ASSERT_EQ(f2[164u],0u);
      const auto unpack=[](UINT value){return pixel8{std::uint8_t(value),std::uint8_t(value>>8u),std::uint8_t(value>>16u),255u};};
      const auto low=unpack(cbc[395u]),high=unpack(cbc[396u]);
      for(UINT c=0u;c<3u;++c){ASSERT_LE(low[c],black[c]);ASSERT_GE(high[c],black[c]);}
      ASSERT_TRUE(tint[0]<low[0]||tint[0]>high[0]||tint[1]<low[1]||tint[1]>high[1]||tint[2]<low[2]||tint[2]>high[2]);
      const auto distance_to_original=[&](UINT q) {
        UINT best=W_+H_;UINT nearest=0u;
        for(UINT p=0u;p<count;++p)if((cover[p]&1u)!=0u&&((selected_bits[p]&1u)!=0u||inner[p]==1u)) {
          const UINT dx=UINT(std::abs(int(p%W_)-int(q%W_))),dy=UINT(std::abs(int(p/W_)-int(q/W_)));
          const UINT d=std::max(dx,dy);if(d<best){best=d;nearest=p;}
        }
        return std::pair{best,nearest};
      };
      constexpr UINT root_mask=0x3fffffffu,enclosed=0x80000000u;
      const UINT exterior=64u*W_+78u-thickness;
      const UINT exterior_tag=cache_words[exterior]&(root_mask|enclosed);
      if(!counter){ASSERT_GT(exterior_tag&root_mask,0u);ASSERT_EQ(exterior_tag&enclosed,0u);}
      nlohmann::json diagnostics=nlohmann::json::array();
      for(UINT q:primary) {
        ASSERT_EQ(origin_pixels_[q],tint);ASSERT_EQ(alpha[q],positive?255u:0u);ASSERT_EQ(ink[q],tint);
        ASSERT_EQ(selected[q],0u);ASSERT_EQ(inner[q],0u);ASSERT_EQ(passable[q],0u);ASSERT_EQ(cover[q]&1u,1u);
        const auto [distance,nearest]=distance_to_original(q);
        if(counter){ASSERT_GT(distance,thickness);}else{ASSERT_LE(distance,thickness);}
        ASSERT_EQ(alpha[nearest],255u);ASSERT_EQ(ink[nearest],white);
        const UINT tag=cache_words[q]&(root_mask|enclosed);
        ASSERT_GT(tag&root_mask,0u);
        if(counter){ASSERT_NE(tag&enclosed,0u);}else{ASSERT_EQ(tag,exterior_tag);}
        diagnostics.push_back({{"native",{q%W_,q/W_}},{"source_RGB8",{tint[0],tint[1],tint[2]}},{"alpha",alpha[q]},
          {"actual_M",final_m[q]},{"packed",packed[q]},{"Code1",code1[q]},{"RawM",raw_m[q]},
          {"distance_to_actual_LOW_or_Inner",distance},{"nearest_original",{nearest%W_,nearest/W_}},{"first_graph_tag",tag}});
        EXPECT_EQ(final_m[q],positive?1u:0u)<<label<<" q="<<q%W_<<','<<q/W_;
      }
      for(UINT q:corridor){ASSERT_EQ(alpha[q],positive?255u:0u);ASSERT_EQ(origin_pixels_[q],tint);EXPECT_EQ(final_m[q],positive?1u:0u)<<label<<" corridor="<<q%W_<<','<<q/W_;}
      for(UINT q:outside){ASSERT_EQ(alpha[q],0u);ASSERT_EQ(origin_pixels_[q],tint);ASSERT_GT(distance_to_original(q).first,thickness);EXPECT_EQ(final_m[q],0u)<<label<<" outside="<<q%W_<<','<<q/W_;}
      for(UINT y=48u;y<88u;++y)for(UINT x=160u;x<208u;++x){const UINT q=y*W_+x;ASSERT_EQ(alpha[q],255u);ASSERT_EQ(ink[q],white);EXPECT_EQ(final_m[q],1u);}
      manifest["target_diagnostics"]=diagnostics;manifest["prerequisites_not_yet_asserted"]=false;write_manifest();
      RecordProperty(label+"_target_diagnostics",diagnostics.dump());
    }

    void expect_authored_low_contrast_aa(bool affine, UINT coverage, UINT thickness, UINT frame, const char *label_prefix = "") {
      require(coverage==0u||coverage==32u||coverage==64u||coverage==96u);
      const auto label=std::string(label_prefix)+"low_contrast_aa_v1_"+(affine?"affine":"flat")+"_t"+std::to_string(thickness)+"_alpha"+std::to_string(coverage);
      const char *matrix_pin=(coverage==0u||coverage==32u)?"b56db2e9cfd08e364785d4a7ebbaf0727a5ca627de521e6268c41d2981acd9f1":"8cf94e7b6ed878d70e48334cca00066c977cff82725689f5966c3f5e29fe5209";
      author(true,frame,0u,false,false,false,256u,128u,3u);
      const std::size_t count=std::size_t(W_)*H_;
      const pixel8 black{3u,4u,8u,255u},white{245u,245u,245u,255u};
      std::vector<pixel8> scene(count,pixel8{128u,128u,128u,255u}),ink(count,black);
      std::vector<std::uint8_t> alpha(count,0u);
      const auto paint=[&](UINT l,UINT t,UINT r,UINT b,pixel8 c,UINT a) {
        require(l<r&&t<b&&r<=W_&&b<=H_&&a<=255u);
        for(UINT y=t;y<b;++y)for(UINT x=l;x<r;++x){ink[y*W_+x]=c;alpha[y*W_+x]=std::uint8_t(a);}
      };
      paint(79u-thickness,48u-thickness,111u+thickness,88u+thickness,black,255u);
      paint(79u,48u,111u,88u,white,255u);
      paint(160u-thickness,48u-thickness,208u+thickness,88u+thickness,black,255u);
      paint(160u,48u,208u,88u,white,255u);
      paint(118u,46u,126u,58u,{32u,32u,32u,255u},255u);
      paint(120u,48u,124u,56u,white,255u);
      std::vector<UINT> cutout;
      const auto transparent=[&](UINT l,UINT t,UINT r,UINT b) {
        for(UINT y=t;y<b;++y)for(UINT x=l;x<r;++x) {
          const UINT i=y*W_+x;alpha[i]=0u;cutout.push_back(i);const int d=2*(int(y)-63);
          scene[i]=affine?pixel8{std::uint8_t(13+d),std::uint8_t(18+d),std::uint8_t(25+d),255u}:pixel8{15u,20u,27u,255u};
        }
      };
      // Independent scene and caption alpha: only q has original AA ink in the
      // positive. A transparent scene cutout connects to the outer background.
      transparent(75u,63u,78u,66u);transparent(79u-thickness,64u,75u,65u);
      const UINT q=63u*W_+75u,p=62u*W_+75u,r=64u*W_+75u;
      alpha[q]=std::uint8_t(coverage);ink[q]=black;origin_pixels_=scene;
      for(std::size_t i=0u;i<count;++i)for(UINT c=0u;c<3u;++c)
        origin_pixels_[i][c]=std::uint8_t((UINT(alpha[i])*ink[i][c]+(255u-alpha[i])*scene[i][c]+127u)/255u);
      RecordProperty(label+"_matrix_sha256",matrix_pin);RecordProperty(label+"_source_sha256",hash_bytes(raw(origin_pixels_)));
      std::filesystem::path folder;nlohmann::json artifacts=nlohmann::json::object();
      if(const auto *out=std::getenv("SUNSHINE_TEST_AUTHORED_LOW_CONTRAST_OUTPUT_DIR")) {
        folder=std::filesystem::path(out)/label;require(!std::filesystem::exists(folder));require(std::filesystem::create_directories(folder));
      }
      const auto emit=[&](const char *name,const auto &values) {
        if(folder.empty())return;const auto bytes=raw(values);const auto path=folder/name;
        std::ofstream f(path,std::ios::binary);f.write(bytes.data(),std::streamsize(bytes.size()));f.close();require(bool(f));
        artifacts[name]={{"path",path.string()},{"bytes",bytes.size()},{"sha256",hash_bytes(bytes)}};
      };
      emit("scene.rgba8",scene);emit("ink.rgba8",ink);emit("ink_alpha.u8",alpha);emit("source.rgba8",origin_pixels_);
      emit("P.f32",probability_words_);emit("S.u32",support_words_);emit("C24.u32",cb_words_);emit("OCR.u32",ocr_words_);
      emit("SCT.u32",sct_words_);emit("UIP.u32",uip_words_);emit("root.u32",root_words_);
      upload_pixels_from_origin();const bool dispatched=run();
      nlohmann::json manifest{{"schema",1u},{"kind","independently_authored_low_contrast_aa"},{"matrix_sha256",matrix_pin},
        {"frame",frame},{"scene",affine?"affine":"flat"},{"coverage",coverage},{"thickness",thickness},{"native_shape",{W_,H_}},
        {"target",{75u,63u}},{"original_donor",{75u,62u}},{"source_background_reference",{75u,64u}},{"cutout_points",cutout},
        {"source_authoring","integer(alpha*ink+(255-alpha)*scene+127)/255; affine scene RGB(13,18,25)+2*(y-63)"},
        {"actual_dispatch_queued",dispatched},{"dispatch_error",error_},{"prerequisites_not_yet_asserted",true}};
      const auto save=[&](){if(folder.empty())return;manifest["artifacts"]=artifacts;std::ofstream f(folder/"manifest.json",std::ios::binary);f<<manifest.dump(2)<<'\n';f.close();require(bool(f));};
      if(!dispatched){save();ASSERT_TRUE(dispatched)<<error_;return;}
      const auto cbc=receipt(),rm=ring_receipt(),f2=calibration();const auto b=helper_.baseline_views();const auto rv=helper_.raw_views();
      const auto original=words(read(b.body.Get())),selected=words(body()),code1=words(matte()),raw_m=words(ring_matte()),m=words(outline_matte());
      const auto packed=words(read(packed_snapshot_.resource.Get())),parent=words(read(b.parent.Get())),flags=words(read(b.root_flags.Get()));
      const auto passable=words(read(b.eligible.Get())),inner=words(read(rv.eligible_inner.Get())),cache=root_cache();
      const auto e=helper_.enclosure_views();const auto cover=words(read(e.cover_bits.Get())),low_bits=words(read(e.selected_bits.Get()));
      emit("CBCMeta.u32",cbc);emit("RawMeta.u32",rm);emit("F2Data.u32",f2);emit("original_BODY.u32",original);emit("selected_BODY.u32",selected);
      emit("Code1.u32",code1);emit("RawM.u32",raw_m);emit("final_M.u32",m);emit("packed.u32",packed);emit("RawParent.u32",parent);
      emit("RawFlags.u32",flags);emit("RawPassable.u32",passable);emit("InnerEligible.u32",inner);emit("PassableCache.u32",cache);
      emit("CoverBits.u32",cover);emit("SelectedBits.u32",low_bits);
      emit_current_hole_evidence(emit);manifest["hole_policy"]=current_hole_policy();
      if(m.size()>q)manifest["target_literal_truth"]={{"alpha",alpha[q]},{"M",m[q]},{"FP",coverage==0u&&m[q]!=0u},{"FN",coverage!=0u&&m[q]==0u}};
      else manifest["target_literal_truth"]={{"available",false},{"actual_words",m.size()}};
      save();RecordProperty(label+"_literal_truth",manifest["target_literal_truth"].dump());
      ASSERT_EQ(cbc.size(),512u);ASSERT_EQ(rm.size(),512u);ASSERT_EQ(f2.size(),68608u);
      for(const auto *v:{&original,&selected,&code1,&raw_m,&m,&packed,&passable,&inner,&cache,&cover,&low_bits})ASSERT_EQ(v->size(),count);
      ASSERT_EQ(parent.size(),count+1u);ASSERT_EQ(flags.size(),parent.size());
      expect_ready();expect_ring_ready();expect_outline_accounting();EXPECT_EQ(original,selected);
      ASSERT_EQ(cbc[385u],1u);ASSERT_EQ(cbc[386u],1u);ASSERT_EQ(cbc[387u],1u);ASSERT_EQ(cbc[134u],1u);ASSERT_EQ(cbc[157u],1u);
      ASSERT_EQ(rm[6u],1u);ASSERT_EQ(rm[128u],1u);ASSERT_EQ(rm[129u],1u);ASSERT_EQ(f2[139u],thickness);ASSERT_EQ(f2[141u],1u);ASSERT_EQ(f2[164u],0u);
      const auto unpack=[](UINT v){return pixel8{std::uint8_t(v),std::uint8_t(v>>8u),std::uint8_t(v>>16u),255u};};
      const auto black_low=unpack(cbc[395u]),black_high=unpack(cbc[396u]);ASSERT_EQ(black_low,black);ASSERT_EQ(black_high,black);
      const auto physical=[&](UINT i){for(UINT c=0u;c<3u;++c)if(origin_pixels_[i][c]<black_low[c]||origin_pixels_[i][c]>black_high[c])return false;return true;};
      const auto root=[&](UINT i){UINT n=i+1u;for(std::size_t k=0u;k<parent.size();++k){if(n>=parent.size())return 0u;const UINT next=parent[n];if(next==n)return n;if(next==0u||next>n)return 0u;n=next;}return 0u;};
      constexpr UINT root_mask=0x3fffffffu,enclosed=0x80000000u;
      const auto tag=[&](UINT i){return cache[i]&(root_mask|enclosed);};
      for(UINT i:{q,r}) {
        ASSERT_EQ(cover[i],1u);ASSERT_FALSE(physical(i));ASSERT_EQ(original[i],0u);ASSERT_EQ(inner[i],0u);ASSERT_EQ(passable[i],0u);
        ASSERT_GT(tag(i)&root_mask,0u);ASSERT_EQ(tag(i)&enclosed,0u);
        const UINT ri=(tag(i)&root_mask)-1u;ASSERT_LT(ri,count);ASSERT_EQ(tag(ri),tag(i));
      }
      ASSERT_EQ(tag(q),tag(r));ASSERT_EQ(alpha[r],0u);ASSERT_EQ(alpha[q],coverage);
      ASSERT_EQ(code1[q],0u);ASSERT_EQ(raw_m[q],0u);ASSERT_EQ(low_bits[q],0u);
      ASSERT_EQ(origin_pixels_[p],black);ASSERT_TRUE(physical(p));ASSERT_EQ(passable[p],1u);ASSERT_NE(packed[p]&1u,0u);
      const UINT common=root(p);ASSERT_GT(common,0u);ASSERT_EQ(flags[common]&3u,1u);
      UINT contrast=0u;for(UINT c=0u;c<3u;++c)contrast=std::max(contrast,UINT(std::abs(int(origin_pixels_[r][c])-int(origin_pixels_[p][c]))));ASSERT_LT(contrast,21u);
      UINT distance=W_+H_;for(UINT i=0u;i<count;++i)if((cover[i]&1u)!=0u&&((low_bits[i]&1u)!=0u||inner[i]==1u))
        distance=std::min(distance,UINT(std::max(std::abs(int(i%W_)-75),std::abs(int(i/W_)-63))));
      ASSERT_LE(distance,thickness);
      const std::array<std::array<int,2u>,8u> dirs{{{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1}}};
      UINT directions=0u;
      for(UINT ray=0u;ray<8u;++ray)for(UINT step=1u;step<=thickness;++step) {
        const int x=75+int(step)*dirs[ray][0],y=63+int(step)*dirs[ray][1];
        if(x<0||y<0||x>=int(W_)||y>=int(H_))break;const UINT i=UINT(y)*W_+UINT(x);
        if((cover[i]&1u)==0u||original[i]!=0u)break;if(!physical(i))continue;
        ASSERT_EQ(passable[i],1u);ASSERT_EQ(root(i),common);ASSERT_EQ(flags[root(i)]&3u,1u);ASSERT_NE(packed[i]&1u,0u);
        directions|=1u<<ray;break;
      }
      ASSERT_GE(std::popcount(directions),3);
      for(UINT start=0u;start<8u;++start){UINT halfplane=0u;for(UINT step=0u;step<=4u;++step)halfplane|=1u<<((start+step)&7u);ASSERT_NE(directions&~halfplane,0u);}
      manifest["mechanism_prerequisites"]={{"p_r_contrast",contrast},{"original_envelope_distance",distance},{"actual_Raw_root",common},{"measured_direction_mask",directions},{"completed",true}};
      manifest["prerequisites_not_yet_asserted"]=false;save();
      EXPECT_EQ(m[q],coverage?1u:0u)<<label<<" q=75,63";
      for(UINT y=48u;y<88u;++y)for(UINT x=160u;x<208u;++x)EXPECT_EQ(m[y*W_+x],1u);
    }


    void expect_broad_background_stroke_coverage(UINT scene_kind, UINT coverage, UINT thickness, UINT frame) {
      require(scene_kind<4u);require(coverage==0u||coverage==64u||coverage==96u);
      const std::array<const char*,4u> scene_names{"flat","channel_affine","corner","dark_channel_affine"};
      const auto label=std::string("broad_background_v1_")+scene_names[scene_kind]+"_t"+std::to_string(thickness)+"_alpha"+std::to_string(coverage)+"_frame"+std::to_string(frame);
      const char *matrix_pin="c7f2634ebe69a05332aaf58ac2a04f1c8a6cc1f40a3417b8b80cec74d5606609";
      author(true,frame,0u,false,false,false,256u,128u,3u);
      const std::size_t count=std::size_t(W_)*H_;
      const pixel8 black{3u,4u,8u,255u},white{245u,245u,245u,255u};
      std::vector<pixel8> scene(count,pixel8{128u,128u,128u,255u}),ink(count,black);
      std::vector<std::uint8_t> alpha(count,0u);
      const auto paint=[&](UINT l,UINT t,UINT r,UINT b,pixel8 c,UINT a) {
        require(l<r&&t<b&&r<=W_&&b<=H_&&a<=255u);
        for(UINT y=t;y<b;++y)for(UINT x=l;x<r;++x){ink[y*W_+x]=c;alpha[y*W_+x]=std::uint8_t(a);}
      };
      paint(79u-thickness,48u-thickness,111u+thickness,88u+thickness,black,255u);
      paint(79u,48u,111u,88u,white,255u);
      paint(160u-thickness,48u-thickness,208u+thickness,88u+thickness,black,255u);
      paint(160u,48u,208u,88u,white,255u);
      paint(118u,46u,126u,58u,{32u,32u,32u,255u},255u);
      paint(120u,48u,124u,56u,white,255u);
      std::vector<UINT> cutout;
      const auto transparent=[&](UINT l,UINT t,UINT r,UINT b) {
        for(UINT y=t;y<b;++y)for(UINT x=l;x<r;++x) {
          const UINT i=y*W_+x;alpha[i]=0u;cutout.push_back(i);
        }
      };
      // Independent scene and caption alpha: only q has original AA ink in the
      // positive. A transparent scene cutout connects to the outer background.
      transparent(75u,63u,78u,66u);transparent(79u-thickness,64u,75u,65u);
      // The entire9x9 field is independent scene, including every possible
      // N8 donor's signed-tangent2x2 stencil. Opaque ink still hides its scene.
      for(UINT y=59u;y<68u;++y)for(UINT x=71u;x<80u;++x) {
        const int dx=int(x)-75,dy=int(y)-63;pixel8 bg{15u,20u,27u,255u};
        if(scene_kind==1u)bg={std::uint8_t(15+dx),std::uint8_t(20+dy),std::uint8_t(27-dx),255u};
        else if(scene_kind==2u)bg={std::uint8_t(15+4*(dx>=0)),std::uint8_t(20+4*(dy>=0)),std::uint8_t(27+4*((dx>=0)!=(dy>=0))),255u};
        else if(scene_kind==3u)bg={std::uint8_t(9+dx),std::uint8_t(12+dy),std::uint8_t(17-dx),255u};
        scene[y*W_+x]=bg;
      }
      const UINT q=63u*W_+75u,p=62u*W_+75u,r=64u*W_+75u;
      alpha[q]=std::uint8_t(coverage);ink[q]=black;origin_pixels_=scene;
      for(std::size_t i=0u;i<count;++i)for(UINT c=0u;c<3u;++c)
        origin_pixels_[i][c]=std::uint8_t((UINT(alpha[i])*ink[i][c]+(255u-alpha[i])*scene[i][c]+127u)/255u);
      RecordProperty(label+"_matrix_sha256",matrix_pin);RecordProperty(label+"_source_sha256",hash_bytes(raw(origin_pixels_)));
      std::filesystem::path folder;nlohmann::json artifacts=nlohmann::json::object();
      if(const auto *out=std::getenv("SUNSHINE_TEST_BROAD_BACKGROUND_OUTPUT_DIR")) {
        folder=std::filesystem::path(out)/label;require(!std::filesystem::exists(folder));require(std::filesystem::create_directories(folder));
      }
      const auto emit=[&](const char *name,const auto &values) {
        if(folder.empty())return;const auto bytes=raw(values);const auto path=folder/name;
        std::ofstream f(path,std::ios::binary);f.write(bytes.data(),std::streamsize(bytes.size()));f.close();require(bool(f));
        artifacts[name]={{"path",path.string()},{"bytes",bytes.size()},{"sha256",hash_bytes(bytes)}};
      };
      emit("scene.rgba8",scene);emit("ink.rgba8",ink);emit("ink_alpha.u8",alpha);emit("source.rgba8",origin_pixels_);
      emit("P.f32",probability_words_);emit("S.u32",support_words_);emit("C24.u32",cb_words_);emit("OCR.u32",ocr_words_);
      emit("SCT.u32",sct_words_);emit("UIP.u32",uip_words_);emit("root.u32",root_words_);
      upload_pixels_from_origin();const bool dispatched=run();
      nlohmann::json manifest{{"schema",1u},{"kind","independently_authored_broad_background_stroke_coverage"},{"matrix_sha256",matrix_pin},
        {"frame",frame},{"scene",scene_names[scene_kind]},{"scene_kind",scene_kind},{"field_bounds_ltrb",{71u,59u,80u,68u}},{"coverage",coverage},{"thickness",thickness},{"native_shape",{W_,H_}},
        {"target",{75u,63u}},{"original_donor",{75u,62u}},{"source_background_reference",{75u,64u}},{"cutout_points",cutout},
        {"source_authoring","integer(alpha*ink+(255-alpha)*scene+127)/255; full9x9 predeclared field around75,63"},
        {"actual_dispatch_queued",dispatched},{"dispatch_error",error_},{"prerequisites_not_yet_asserted",true}};
      const auto save=[&](){if(folder.empty())return;manifest["artifacts"]=artifacts;std::ofstream f(folder/"manifest.json",std::ios::binary);f<<manifest.dump(2)<<'\n';f.close();require(bool(f));};
      if(!dispatched){save();ASSERT_TRUE(dispatched)<<error_;return;}
      const auto cbc=receipt(),rm=ring_receipt(),f2=calibration();const auto b=helper_.baseline_views();const auto rv=helper_.raw_views();
      const auto original=words(read(b.body.Get())),selected=words(body()),code1=words(matte()),raw_m=words(ring_matte()),m=words(outline_matte());
      const auto packed=words(read(packed_snapshot_.resource.Get())),parent=words(read(b.parent.Get())),flags=words(read(b.root_flags.Get()));
      const auto passable=words(read(b.eligible.Get())),inner=words(read(rv.eligible_inner.Get())),cache=root_cache();
      const auto e=helper_.enclosure_views();const auto cover=words(read(e.cover_bits.Get())),low_bits=words(read(e.selected_bits.Get()));
      emit("CBCMeta.u32",cbc);emit("RawMeta.u32",rm);emit("F2Data.u32",f2);emit("original_BODY.u32",original);emit("selected_BODY.u32",selected);
      emit("Code1.u32",code1);emit("RawM.u32",raw_m);emit("final_M.u32",m);emit("packed.u32",packed);emit("RawParent.u32",parent);
      emit("RawFlags.u32",flags);emit("RawPassable.u32",passable);emit("InnerEligible.u32",inner);emit("PassableCache.u32",cache);
      emit("CoverBits.u32",cover);emit("SelectedBits.u32",low_bits);
      emit_current_hole_evidence(emit);manifest["hole_policy"]=current_hole_policy();
      if(m.size()==count){UINT tp=0u,fp=0u,fn=0u;for(UINT i=0u;i<count;++i){tp+=m[i]!=0u&&alpha[i]!=0u;fp+=m[i]!=0u&&alpha[i]==0u;fn+=m[i]==0u&&alpha[i]!=0u;}manifest["whole_frame_literal_truth_counts"]={{"TP",tp},{"FP",fp},{"FN",fn}};}

      if(m.size()>q)manifest["target_literal_truth"]={{"alpha",alpha[q]},{"M",m[q]},{"FP",coverage==0u&&m[q]!=0u},{"FN",coverage!=0u&&m[q]==0u}};
      else manifest["target_literal_truth"]={{"available",false},{"actual_words",m.size()}};
      save();RecordProperty(label+"_literal_truth",manifest["target_literal_truth"].dump());
      ASSERT_EQ(cbc.size(),512u);ASSERT_EQ(rm.size(),512u);ASSERT_EQ(f2.size(),68608u);
      for(const auto *v:{&original,&selected,&code1,&raw_m,&m,&packed,&passable,&inner,&cache,&cover,&low_bits})ASSERT_EQ(v->size(),count);
      ASSERT_EQ(parent.size(),count+1u);ASSERT_EQ(flags.size(),parent.size());
      expect_ready();expect_ring_ready();expect_outline_accounting();EXPECT_EQ(original,selected);
      ASSERT_EQ(cbc[385u],1u);ASSERT_EQ(cbc[386u],1u);ASSERT_EQ(cbc[387u],1u);ASSERT_EQ(cbc[134u],1u);ASSERT_EQ(cbc[157u],1u);
      ASSERT_EQ(rm[6u],1u);ASSERT_EQ(rm[128u],1u);ASSERT_EQ(rm[129u],1u);ASSERT_EQ(f2[139u],thickness);ASSERT_EQ(f2[141u],1u);ASSERT_EQ(f2[164u],0u);
      const auto unpack=[](UINT v){return pixel8{std::uint8_t(v),std::uint8_t(v>>8u),std::uint8_t(v>>16u),255u};};
      const auto black_low=unpack(cbc[395u]),black_high=unpack(cbc[396u]);ASSERT_EQ(black_low,black);ASSERT_EQ(black_high,black);
      const auto physical=[&](UINT i){for(UINT c=0u;c<3u;++c)if(origin_pixels_[i][c]<black_low[c]||origin_pixels_[i][c]>black_high[c])return false;return true;};
      const auto root=[&](UINT i){UINT n=i+1u;for(std::size_t k=0u;k<parent.size();++k){if(n>=parent.size())return 0u;const UINT next=parent[n];if(next==n)return n;if(next==0u||next>n)return 0u;n=next;}return 0u;};
      constexpr UINT root_mask=0x3fffffffu,enclosed=0x80000000u;
      const auto tag=[&](UINT i){return cache[i]&(root_mask|enclosed);};
      for(UINT i:{q,r}) {
        ASSERT_EQ(cover[i],1u);ASSERT_FALSE(physical(i));ASSERT_EQ(original[i],0u);ASSERT_EQ(inner[i],0u);ASSERT_EQ(passable[i],0u);
        ASSERT_GT(tag(i)&root_mask,0u);ASSERT_EQ(tag(i)&enclosed,0u);
        const UINT ri=(tag(i)&root_mask)-1u;ASSERT_LT(ri,count);ASSERT_EQ(tag(ri),tag(i));
      }
      ASSERT_EQ(tag(q),tag(r));ASSERT_EQ(alpha[r],0u);ASSERT_EQ(alpha[q],coverage);
      ASSERT_EQ(code1[q],0u);ASSERT_EQ(raw_m[q],0u);ASSERT_EQ(low_bits[q],0u);
      ASSERT_EQ(origin_pixels_[p],black);ASSERT_TRUE(physical(p));ASSERT_EQ(passable[p],1u);ASSERT_NE(packed[p]&1u,0u);
      const UINT common=root(p);ASSERT_GT(common,0u);ASSERT_EQ(flags[common]&3u,1u);
      UINT contrast=0u;for(UINT c=0u;c<3u;++c)contrast=std::max(contrast,UINT(std::abs(int(origin_pixels_[r][c])-int(origin_pixels_[p][c]))));ASSERT_LT(contrast,21u);
      UINT distance=W_+H_;for(UINT i=0u;i<count;++i)if((cover[i]&1u)!=0u&&((low_bits[i]&1u)!=0u||inner[i]==1u))
        distance=std::min(distance,UINT(std::max(std::abs(int(i%W_)-75),std::abs(int(i/W_)-63))));
      ASSERT_LE(distance,thickness);
      const std::array<std::array<int,2u>,8u> dirs{{{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1}}};
      UINT directions=0u;
      for(UINT ray=0u;ray<8u;++ray)for(UINT step=1u;step<=thickness;++step) {
        const int x=75+int(step)*dirs[ray][0],y=63+int(step)*dirs[ray][1];
        if(x<0||y<0||x>=int(W_)||y>=int(H_))break;const UINT i=UINT(y)*W_+UINT(x);
        if((cover[i]&1u)==0u||original[i]!=0u)break;if(!physical(i))continue;
        ASSERT_EQ(passable[i],1u);ASSERT_EQ(root(i),common);ASSERT_EQ(flags[root(i)]&3u,1u);ASSERT_NE(packed[i]&1u,0u);
        directions|=1u<<ray;break;
      }
      ASSERT_GE(std::popcount(directions),3);
      for(UINT start=0u;start<8u;++start){UINT halfplane=0u;for(UINT step=0u;step<=4u;++step)halfplane|=1u<<((start+step)&7u);ASSERT_NE(directions&~halfplane,0u);}
      ASSERT_TRUE(helper_.mask_holes_enabled());
      const auto proof=words(read(helper_.mask_hole_proof()));ASSERT_EQ(proof.size(),count);
      ASSERT_EQ(proof[q]&65536u,0u);ASSERT_NE(proof[q]&(1u<<17u),0u);
      ASSERT_EQ(proof[p]&0x10100u,0x10100u);
      const auto domain=[&](int x,int y){return x>=0&&y>=0&&x<int(W_)&&y<int(H_)&&UINT(x)>=cbc[152u]&&UINT(y)>=cbc[153u]&&UINT(x)<cbc[154u]&&UINT(y)<cbc[155u]&&(cover[UINT(y)*W_+UINT(x)]&1u)!=0u;};
      const auto original_reference=[&](int x,int y){if(!domain(x,y))return false;const UINT i=UINT(y)*W_+UINT(x);return (proof[i]&0x100ffu)==0u&&(low_bits[i]&65535u)==0u&&!physical(i);};
      UINT complete_stencils=0u;
      for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
        if(dx==0&&dy==0)continue;const int px=75-dx,py=63-dy;if(!domain(px,py))continue;const UINT pi=UINT(py)*W_+UINT(px);
        if((proof[pi]&0x10100u)!=0x10100u||!physical(pi))continue;
        const int tx=-dy,ty=dx;
        for(int side:{-1,1})if(original_reference(75+dx,63+dy)&&original_reference(75+dx+side*tx,63+dy+side*ty)&&original_reference(75+2*dx,63+2*dy)&&original_reference(75+2*dx+side*tx,63+2*dy+side*ty))++complete_stencils;
      }
      ASSERT_GT(complete_stencils,0u)<<"No eligible original-M0 reference stencil; retain this attempt as unqualified";
      manifest["reference_readiness"]={{"original_complete_stencils",complete_stencils},{"captured_blacklike",true},{"own_C3_originalM1_donor",true}};
      manifest["mechanism_prerequisites"]={{"p_r_contrast",contrast},{"original_envelope_distance",distance},{"actual_Raw_root",common},{"measured_direction_mask",directions},{"completed",true}};
      manifest["prerequisites_not_yet_asserted"]=false;save();
      EXPECT_EQ(m[q],coverage?1u:0u)<<label<<" q=75,63";
      for(UINT y=48u;y<88u;++y)for(UINT x=160u;x<208u;++x)EXPECT_EQ(m[y*W_+x],1u);
    }

    // Predeclared source/alpha fixtures; GPU maps come from normal authored P/S/OCR.
    // The independent scene/ink/alpha arrays are exported for scoring, never GPU inputs.
    enum class generic_heldout_role {outer_aa, inner_aa, counter, attached_scene};
    void expect_generic_heldout(generic_heldout_role role, bool affine, UINT scale,
      UINT frame, bool alias_diagnostic = false, bool scene_darkspot = false) {
      require(scale==1u||scale==2u);
      const char *role_env=std::getenv("SUNSHINE_TEST_GENERIC_HELDOUT_ROLE");
      require(role_env&&(std::string(role_env)=="control"||std::string(role_env)=="treatment"));
      const bool control=std::string(role_env)=="control";
      const bool inner=role==generic_heldout_role::inner_aa||role==generic_heldout_role::counter;
      const bool positive=role==generic_heldout_role::outer_aa||role==generic_heldout_role::inner_aa;
      const UINT thickness=3u*scale,coverage=scene_darkspot?0u:(positive?(inner?96u:64u):0u);
      const bool warm=affine!=(scale==2u);
      const pixel8 black=warm?pixel8{8u,5u,3u,255u}:pixel8{3u,4u,8u,255u};
      const pixel8 white{245u,245u,245u,255u};
      const char *names[]={"outer_aa","inner_aa","counter","attached_scene"};
      const std::string label=std::string(alias_diagnostic?"generic_alias_v1_":"generic_heldout_v1_")+names[UINT(role)]+
        (affine?"_affine":"_flat")+"_s"+std::to_string(scale)+"_frame"+std::to_string(frame);
      constexpr const char *matrix_pin="a6b79a9797a0270e70a084764a1ac3acad25fc2edb2ac9c5fe16a732ed2e14fc";
      author(true,frame,0u,false,false,false,256u*scale,128u*scale,3u);
      const std::size_t count=std::size_t(W_)*H_;
      std::vector<pixel8> scene(count,pixel8{128u,128u,128u,255u}),ink(count,black);
      std::vector<std::uint8_t> alpha(count,0u);
      const auto paint=[&](UINT l,UINT t,UINT r,UINT b,pixel8 colour,UINT opacity) {
        require(l<r&&t<b&&r<=W_&&b<=H_&&opacity<=255u);
        for(UINT y=t;y<b;++y)for(UINT x=l;x<r;++x){ink[y*W_+x]=colour;alpha[y*W_+x]=std::uint8_t(opacity);}
      };
      paint(79u*scale-thickness,48u*scale-thickness,111u*scale+thickness,88u*scale+thickness,black,255u);
      paint(79u*scale,48u*scale,111u*scale,88u*scale,white,255u);
      paint(160u*scale-thickness,48u*scale-thickness,208u*scale+thickness,88u*scale+thickness,black,255u);
      paint(160u*scale,48u*scale,208u*scale,88u*scale,white,255u);
      paint(118u*scale,46u*scale,126u*scale,58u*scale,{32u,32u,32u,255u},255u);
      paint(120u*scale,48u*scale,124u*scale,56u*scale,white,255u);
      UINT qx=79u*scale-thickness+1u,qy=63u*scale;
      if(inner) {
        // A true O counter, with inner rim one pixel thinner than the easy/global
        // stroke. Its first scene point is inside the global original-width prior.
        const UINT local=thickness-1u;
        paint(87u*scale,56u*scale,103u*scale,80u*scale,black,255u);
        paint(87u*scale+local,56u*scale+local,103u*scale-local,80u*scale-local,black,0u);
        qx=95u*scale;qy=56u*scale+local;
      } else {
        paint(qx,qy,qx+2u,qy+3u,black,0u);
        paint(79u*scale-thickness,qy+1u,qx,qy+2u,black,0u);
      }
      // Scene is authored independently. The local field is flat/affine over all
      // declared references; its low contrast does not imply that it is caption.
      // Covers every +/-tangent reference from every N8 donor (offsets <=3),
      // rather than selecting only the favourable declared vertical stencil.
      for(UINT y=qy-4u;y<qy+5u;++y)for(UINT x=qx-4u;x<qx+5u;++x) {
        const int dx=int(x)-int(qx),dy=int(y)-int(qy);
        scene[y*W_+x]={std::uint8_t(int(black[0])+12+(affine?dx:0)),
          std::uint8_t(int(black[1])+16+(affine?dy:0)),
          std::uint8_t(int(black[2])+19-(affine?dx:0)),255u};
      }
      const UINT q=qy*W_+qx,p=q-W_;
      const std::array<UINT,4u> refs{q+W_,q+W_+1u,q+2u*W_,q+2u*W_+1u};
      alpha[q]=std::uint8_t(coverage);ink[q]=black;
      if(scene_darkspot) {
        // Deliberately unidentifiable source-layer diagnostic. All source RGB is
        // identical to the paired original-alpha64 case; truth differs only at q.
        for(UINT c=0u;c<3u;++c)scene[q][c]=std::uint8_t((64u*black[c]+191u*scene[q][c]+127u)/255u);
      }
      origin_pixels_=scene;
      for(std::size_t i=0u;i<count;++i)for(UINT c=0u;c<3u;++c)
        origin_pixels_[i][c]=std::uint8_t((UINT(alpha[i])*ink[i][c]+(255u-alpha[i])*scene[i][c]+127u)/255u);
      RecordProperty(label+"_matrix_sha256",matrix_pin);RecordProperty(label+"_source_sha256",hash_bytes(raw(origin_pixels_)));
      std::filesystem::path folder;nlohmann::json artifacts=nlohmann::json::object();
      if(const auto *out=std::getenv("SUNSHINE_TEST_GENERIC_HELDOUT_OUTPUT_DIR")) {
        folder=std::filesystem::path(out)/label;require(!std::filesystem::exists(folder));require(std::filesystem::create_directories(folder));
      }
      const auto emit=[&](const char *name,const auto &values) {
        if(folder.empty())return;const auto bytes=raw(values);const auto path=folder/name;
        std::ofstream f(path,std::ios::binary);f.write(bytes.data(),std::streamsize(bytes.size()));f.close();require(bool(f));
        artifacts[name]={{"path",path.string()},{"bytes",bytes.size()},{"sha256",hash_bytes(bytes)}};
      };
      emit("scene.rgba8",scene);emit("ink.rgba8",ink);emit("ink_alpha.u8",alpha);emit("source.rgba8",origin_pixels_);
      emit("P.f32",probability_words_);emit("S.u32",support_words_);emit("C24.u32",cb_words_);emit("OCR.u32",ocr_words_);
      emit("SCT.u32",sct_words_);emit("UIP.u32",uip_words_);emit("root.u32",root_words_);
      upload_pixels_from_origin();const bool dispatched=run();
      nlohmann::json manifest{{"schema",1u},{"kind",alias_diagnostic?"unresolvable_source_layer_diagnostic":"predeclared_generic_heldout16"},
        {"matrix_sha256",matrix_pin},{"frame",frame},{"role",names[UINT(role)]},{"scene",affine?"affine":"flat"},
        {"scale",scale},{"thickness",thickness},{"black_RGBA8",black},{"coverage",coverage},{"native_shape",{W_,H_}},
        {"target",{qx,qy}},{"original_donor",{qx,qy-1u}},{"declared_right_stencil",refs},{"source_layer_alias_diagnostic",alias_diagnostic},
        {"scene_darkspot",scene_darkspot},{"capture_role",control?"control":"treatment"},{"actual_dispatch_queued",dispatched},
        {"dispatch_error",error_},{"prerequisites_not_yet_asserted",true},{"treatment_requires_matching_sealed_control",!control}};
      const auto save=[&](){if(folder.empty())return;manifest["artifacts"]=artifacts;std::ofstream f(folder/"manifest.json",std::ios::binary);f<<manifest.dump(2)<<'\n';f.close();require(bool(f));};
      if(!dispatched){save();ASSERT_TRUE(dispatched)<<error_;return;}
      const auto cbc=receipt(),rm=ring_receipt(),f2=calibration();const auto b=helper_.baseline_views();const auto rv=helper_.raw_views();const auto e=helper_.enclosure_views();
      const auto original=words(read(b.body.Get())),selected=words(body()),code1=words(matte()),raw_m=words(ring_matte()),m=words(outline_matte());
      const auto packed=words(read(packed_snapshot_.resource.Get())),parent=words(read(b.parent.Get())),flags=words(read(b.root_flags.Get()));
      const auto passable=words(read(b.eligible.Get())),inner_bits=words(read(rv.eligible_inner.Get())),cache=root_cache();
      const auto cover=words(read(e.cover_bits.Get())),low_high=words(read(e.selected_bits.Get()));
      emit("CBCMeta.u32",cbc);emit("RawMeta.u32",rm);emit("F2Data.u32",f2);emit("original_BODY.u32",original);emit("selected_BODY.u32",selected);
      emit("Code1.u32",code1);emit("RawM.u32",raw_m);emit("final_M.u32",m);emit("packed.u32",packed);emit("RawParent.u32",parent);
      emit("RawFlags.u32",flags);emit("RawPassable.u32",passable);emit("InnerEligible.u32",inner_bits);emit("PassableCache.u32",cache);
      emit("CoverBits.u32",cover);emit("SelectedBits.u32",low_high);emit("ModeFinalScratch.u32",words(read(b.mode_final.Get())));
      emit_current_hole_evidence(emit);manifest["hole_policy"]=current_hole_policy();
      if(m.size()>q)manifest["target_literal_truth"]={{"alpha",alpha[q]},{"M",m[q]},{"FP",alpha[q]==0u&&m[q]!=0u},{"FN",alpha[q]!=0u&&m[q]==0u}};
      UINT tp=0u,fp=0u,fn=0u;if(m.size()==count)for(UINT i=0u;i<count;++i){tp+=m[i]!=0u&&alpha[i]!=0u;fp+=m[i]!=0u&&alpha[i]==0u;fn+=m[i]==0u&&alpha[i]!=0u;}
      manifest["whole_frame_literal_truth_counts"]={{"TP",tp},{"FP",fp},{"FN",fn}};save();
      ASSERT_EQ(cbc.size(),512u);ASSERT_EQ(rm.size(),512u);ASSERT_EQ(f2.size(),68608u);
      for(const auto *v:{&original,&selected,&code1,&raw_m,&m,&packed,&passable,&inner_bits,&cache,&cover,&low_high})ASSERT_EQ(v->size(),count);
      ASSERT_EQ(parent.size(),count+1u);ASSERT_EQ(flags.size(),parent.size());
      expect_ready();expect_ring_ready();expect_outline_accounting();EXPECT_EQ(original,selected);
      ASSERT_EQ(rm[6u],1u);ASSERT_EQ(rm[128u],1u);ASSERT_EQ(rm[129u],1u);
      ASSERT_EQ(f2[139u],thickness);ASSERT_EQ(f2[141u],1u);ASSERT_EQ(f2[164u],0u);
      const auto unpack=[](UINT v){return pixel8{std::uint8_t(v),std::uint8_t(v>>8u),std::uint8_t(v>>16u),255u};};
      const auto low=unpack(cbc[395u]),high=unpack(cbc[396u]);ASSERT_EQ(low,black);ASSERT_EQ(high,black);
      const auto physical=[&](UINT i){if(std::max({origin_pixels_[i][0],origin_pixels_[i][1],origin_pixels_[i][2]})<=1u)return true;
        for(UINT c=0u;c<3u;++c)if(origin_pixels_[i][c]<low[c]||origin_pixels_[i][c]>high[c])return false;return true;};
      const auto domain=[&](UINT i){return (cover[i]&1u)!=0u&&i%W_>=cbc[152u]&&i/W_>=cbc[153u]&&i%W_<cbc[154u]&&i/W_<cbc[155u];};
      ASSERT_TRUE(domain(q));ASSERT_TRUE(domain(p));ASSERT_FALSE(physical(q));ASSERT_TRUE(physical(p));
      ASSERT_EQ(low_high[q]&257u,0u);ASSERT_EQ(inner_bits[q],0u);ASSERT_EQ(original[q],0u);
      ASSERT_TRUE(helper_.mask_holes_enabled())<<"Use the published C7v2 control with its optional programs enabled";
      const auto proof=words(read(helper_.mask_hole_proof()));ASSERT_EQ(proof.size(),count);
      ASSERT_EQ(proof[p]&0x10100u,0x10100u);ASSERT_EQ(passable[p],1u);
      ASSERT_EQ(proof[q]&65536u,0u);ASSERT_NE(proof[q]&(1u<<17u),0u)<<"Recipient must reach the new model, not fail the existing black-like gate";
      UINT contrast=0u;for(UINT c=0u;c<3u;++c)contrast=std::max(contrast,UINT(std::abs(int(origin_pixels_[refs[0]][c])-int(black[c]))));
      ASSERT_LT(contrast,21u); // Existing early gate is deliberately active, not a new detector threshold.
      UINT distance=W_+H_;for(UINT i=0u;i<count;++i)if(domain(i)&&((low_high[i]&1u)!=0u||inner_bits[i]==1u))
        distance=std::min(distance,UINT(std::max(std::abs(int(i%W_)-int(qx)),std::abs(int(i/W_)-int(qy)))));
      ASSERT_LE(distance,thickness);
      for(UINT i:refs){ASSERT_TRUE(domain(i));ASSERT_EQ(alpha[i],0u);ASSERT_FALSE(physical(i));ASSERT_EQ(low_high[i]&257u,0u);
        ASSERT_EQ(inner_bits[i],0u);ASSERT_EQ(proof[i]&65536u,0u);if(control)ASSERT_EQ(m[i],0u);}
      if(control)ASSERT_EQ(m[q],0u)<<"An already protected target is diagnostic, not active repair evidence";
      manifest[control?"control_readiness":"current_source_readiness"]={{"completed",true},{"original_envelope_distance",distance},
        {"existing_p_reference_contrast",contrast},{"declared_right_stencil_original_alpha0",true},{"actual_C3_original_M1_donor",true}};
      manifest["prerequisites_not_yet_asserted"]=false;save();
      // The source-identical alias pair is intentionally not a universal-success
      // gate. Its contradictory original alpha remains in the complete report.
      if(!alias_diagnostic)EXPECT_EQ(m[q],coverage?1u:0u)<<label;
    }

    // Production GPU produces BODY, geometry and final M from authored P/S/OCR inputs.
    // Baseline geometry is measured only on the explicit control run; the paired
    // reader must join that immutable control to treatment, never inspect new M
    // as if it were original topology.
    void expect_authored_c7_boundary_case(bool positive, UINT scene_kind, UINT thickness, pixel8 tint, UINT frame) {
      require(thickness==5u||thickness==6u);require(scene_kind<4u);
      const auto *role_env=std::getenv("SUNSHINE_TEST_AUTHORED_C7_ROLE");
      require(role_env&&(std::string(role_env)=="control"||std::string(role_env)=="treatment"));
      const bool control=std::string(role_env)=="control";
      const std::array<const char*,4u> scenes{"flat128","affine_rgb","dark_flat24","dark_affine_rgb"};
      const auto label=std::string("c7_boundary_v1_")+(positive?"opaque_corner_":"C_counter_")+scenes[scene_kind]+"_t"+std::to_string(thickness)+"_frame"+std::to_string(frame);
      constexpr const char *matrix_pin="ad7e7b02ea50a32c29a75b7cca7ed887ca5aefc5b993af7189fbca8422d36d84";
      author(true,frame,0u,false,false,false,256u,128u,3u);
      // Exact native core[64,40,224,112], cover[32,24,240,128] under CField16x16.
      const std::array<UINT,4u> core{4u,5u,14u,14u},cover_rect{2u,3u,15u,16u};
      std::copy(core.begin(),core.end(),ocr_words_.begin()+16u);
      std::copy(cover_rect.begin(),cover_rect.end(),ocr_words_.begin()+144u);
      const std::size_t count=std::size_t(W_)*H_;
      const pixel8 black{3u,4u,8u,255u},white{245u,245u,245u,255u};
      std::vector<pixel8> scene(count),ink(count,black);
      std::vector<std::uint8_t> alpha(count,0u),fill(count,0u);
      for(UINT y=0u;y<H_;++y)for(UINT x=0u;x<W_;++x) {
        pixel8 value{128u,128u,128u,255u};
        if(scene_kind==1u)value={std::uint8_t(96u+x/4u),std::uint8_t(112u+y/4u),std::uint8_t(136u+(x+y)/8u),255u};
        if(scene_kind==2u)value={24u,24u,24u,255u};
        if(scene_kind==3u)value={std::uint8_t(16u+x/32u),std::uint8_t(20u+y/32u),std::uint8_t(32u+(x+y)/64u),255u};
        scene[y*W_+x]=value;
      }
      const auto fill_rect=[&](UINT l,UINT top,UINT r,UINT bottom,bool on) {
        require(l<r&&top<bottom&&r<=W_&&bottom<=H_);
        for(UINT y=top;y<bottom;++y)for(UINT x=l;x<r;++x)fill[y*W_+x]=on?1u:0u;
      };
      const UINT gap_top=68u-(thickness-1u),gap_bottom=68u+thickness;
      if(positive)fill_rect(80u,64u,112u,104u,true);
      else {
        fill_rect(80u,48u,120u,88u,true);
        fill_rect(88u,gap_top-thickness,112u,gap_bottom+thickness,false);
        fill_rect(80u,gap_top,96u,gap_bottom,false);
      }
      fill_rect(160u,48u,208u,88u,true);
      // Original fixture ink: Chebyshev outline around the independently authored
      // white fill. This is not a product-generated stroke or an output oracle.
      for(UINT y=0u;y<H_;++y)for(UINT x=0u;x<W_;++x)if(fill[y*W_+x])
        for(int dy=-int(thickness);dy<=int(thickness);++dy)for(int dx=-int(thickness);dx<=int(thickness);++dx) {
          const int qx=int(x)+dx,qy=int(y)+dy;if(qx>=0&&qy>=0&&qx<int(W_)&&qy<int(H_))alpha[UINT(qy)*W_+UINT(qx)]=255u;
        }
      for(std::size_t i=0u;i<count;++i)if(fill[i]){ink[i]=white;alpha[i]=255u;}
      const auto paint=[&](UINT l,UINT top,UINT r,UINT bottom,pixel8 colour) {
        require(l<r&&top<bottom&&r<=W_&&bottom<=H_);
        for(UINT y=top;y<bottom;++y)for(UINT x=l;x<r;++x){ink[y*W_+x]=colour;alpha[y*W_+x]=255u;}
      };
      paint(118u,46u,126u,58u,{32u,32u,32u,255u});paint(120u,48u,124u,56u,white);
      const UINT qx=positive?79u-thickness:100u,qy=positive?63u-thickness:68u,q=qy*W_+qx;
      if(positive){paint(qx-1u,qy-1u,qx+2u,qy+2u,black);ink[q]=tint;alpha[q]=255u;}
      std::vector<UINT> primary{q};bool authored_counter_closed=true;
      const std::array<std::array<int,2u>,4u> n4{{{1,0},{-1,0},{0,1},{0,-1}}};
      if(!positive) {
        // GT region comes from the independent source-alpha construction only.
        // No production M/graph is consulted to name the genuine counter.
        require(alpha[q]==0u);std::vector<std::uint8_t> seen(count,0u);primary.clear();primary.push_back(q);seen[q]=1u;
        for(std::size_t k=0u;k<primary.size();++k){const UINT p=primary[k],x=p%W_,y=p/W_;
          if(x==0u||y==0u||x+1u==W_||y+1u==H_){authored_counter_closed=false;continue;}
          for(auto d:n4){const UINT z=UINT(int(y)+d[1])*W_+UINT(int(x)+d[0]);if(!seen[z]&&alpha[z]==0u){seen[z]=1u;primary.push_back(z);}}
        }
      }
      origin_pixels_=scene;
      for(std::size_t i=0u;i<count;++i)for(UINT c=0u;c<3u;++c)
        origin_pixels_[i][c]=std::uint8_t((UINT(alpha[i])*ink[i][c]+(255u-alpha[i])*scene[i][c]+127u)/255u);
      std::filesystem::path folder;nlohmann::json artifacts=nlohmann::json::object();
      if(const auto *out=std::getenv("SUNSHINE_TEST_AUTHORED_C7_OUTPUT_DIR")) {
        folder=std::filesystem::path(out)/label;require(!std::filesystem::exists(folder));require(std::filesystem::create_directories(folder));
      }
      const auto emit=[&](const char *name,const auto &values) {
        if(folder.empty())return;const auto bytes=raw(values);std::ofstream f(folder/name,std::ios::binary);
        f.write(bytes.data(),std::streamsize(bytes.size()));f.close();require(bool(f));
        artifacts[name]={{"path",(folder/name).string()},{"bytes",bytes.size()},{"sha256",hash_bytes(bytes)}};
      };
      emit("scene.rgba8",scene);emit("ink.rgba8",ink);emit("ink_alpha.u8",alpha);emit("source.rgba8",origin_pixels_);
      emit("P.f32",probability_words_);emit("S.u32",support_words_);emit("C24.u32",cb_words_);emit("OCR.u32",ocr_words_);
      emit("SCT.u32",sct_words_);emit("UIP.u32",uip_words_);emit("root.u32",root_words_);
      upload_pixels_from_origin();const bool dispatched=run();
      nlohmann::json manifest{{"schema",1u},{"kind","independently_authored_c7_boundary_challenge"},{"matrix_sha256",matrix_pin},
        {"frame",frame},{"role",positive?"corner_opaque_original_ink_patch":"asymmetric_genuine_counter"},{"scene",scenes[scene_kind]},{"thickness",thickness},{"native_shape",{W_,H_}},
        {"capture_role",role_env},{"primary_points",primary},{"authored_counter_closed",authored_counter_closed},{"target",{qx,qy}},{"tint_RGBA8",tint},{"actual_dispatch_queued",dispatched},{"dispatch_error",error_},
        {"prerequisites_not_yet_asserted",true},{"control_readiness_inherited_only_from_matching_control",!control}};
      const auto save=[&](){if(folder.empty())return;manifest["artifacts"]=artifacts;std::ofstream f(folder/"manifest.json",std::ios::binary);f<<manifest.dump(2)<<'\n';f.close();require(bool(f));};
      if(!dispatched){save();ASSERT_TRUE(dispatched)<<error_;return;}
      const auto cbc=receipt(),rm=ring_receipt(),f2=calibration();const auto b=helper_.baseline_views();const auto rv=helper_.raw_views();const auto e=helper_.enclosure_views();
      const auto original=words(read(b.body.Get())),selected=words(body()),code1=words(matte()),raw_m=words(ring_matte()),m=words(outline_matte());
      const auto packed=words(read(packed_snapshot_.resource.Get())),parent=words(read(b.parent.Get())),flags=words(read(b.root_flags.Get()));
      const auto passable=words(read(b.eligible.Get())),inner=words(read(rv.eligible_inner.Get())),cache=root_cache();
      const auto cover=words(read(e.cover_bits.Get())),low_high=words(read(e.selected_bits.Get())),family=words(read(e.family_models.Get())),witness=words(read(b.mode_final.Get()));
      emit("CBCMeta.u32",cbc);emit("RawMeta.u32",rm);emit("F2Data.u32",f2);emit("original_BODY.u32",original);emit("selected_BODY.u32",selected);
      emit("Code1.u32",code1);emit("RawM.u32",raw_m);emit("final_M.u32",m);emit("packed.u32",packed);emit("RawParent.u32",parent);emit("RawFlags.u32",flags);
      emit("RawPassable.u32",passable);emit("InnerEligible.u32",inner);emit("PassableCache.u32",cache);emit("CoverBits.u32",cover);emit("SelectedBits.u32",low_high);
      emit("FamilyModels17x8.u32",family);emit("ModeFinalScratch.u32",witness);
      emit_current_hole_evidence(emit,false);manifest["hole_policy"]=current_hole_policy();save();
      RecordProperty(label+"_matrix_sha256",matrix_pin);RecordProperty(label+"_source_sha256",hash_bytes(raw(origin_pixels_)));
      ASSERT_EQ(cbc.size(),512u);ASSERT_EQ(rm.size(),512u);ASSERT_EQ(f2.size(),68608u);ASSERT_EQ(family.size(),17u*8u);
      for(const auto *v:{&original,&selected,&code1,&raw_m,&m,&packed,&passable,&inner,&cache,&cover,&low_high,&witness})ASSERT_EQ(v->size(),count);
      ASSERT_EQ(parent.size(),count+1u);ASSERT_EQ(flags.size(),parent.size());expect_ready();expect_ring_ready();expect_outline_accounting();EXPECT_EQ(original,selected);
      ASSERT_TRUE(authored_counter_closed);
      ASSERT_EQ(rm[6u],1u);ASSERT_EQ(rm[128u],1u);ASSERT_EQ(rm[129u],1u);ASSERT_EQ(cbc[386u],1u);ASSERT_EQ(cbc[157u],1u);
      ASSERT_EQ(f2[139u],thickness);ASSERT_EQ(f2[141u],1u);ASSERT_EQ(f2[164u],0u);
      const auto unpack=[](UINT v){return pixel8{std::uint8_t(v),std::uint8_t(v>>8u),std::uint8_t(v>>16u),255u};};
      const auto kl=unpack(cbc[395u]),kh=unpack(cbc[396u]);ASSERT_EQ(kl,black);ASSERT_EQ(kh,black);
      UINT total=0u,short_profiles=0u;const UINT lower=f2[146u],rmax=f2[134u],bands=f2[135u];
      for(UINT band=0u;band<64u;++band)for(UINT extent=0u;extent<128u;++extent){const UINT n=f2[1024u+band*128u+extent];
        if(n){ASSERT_LT(band,bands);ASSERT_GT(extent,0u);ASSERT_LT(extent,rmax);total+=n;if(extent<lower)short_profiles+=n;}}
      manifest["width_histogram"]={{"actual_lower_window",lower},{"accepted_profiles",total},{"short_profiles",short_profiles}};save();
      ASSERT_EQ(total,f2[136u]);ASSERT_EQ(short_profiles,0u);
      const auto box_distance=[&](UINT p,UINT lo,UINT hi){const auto l=unpack(lo),r=unpack(hi);UINT result=0u;for(UINT c=0u;c<3u;++c){const int v=origin_pixels_[p][c];result=std::max(result,UINT(std::max({int(l[c])-v,v-int(r[c]),0})));}return result;};
      const UINT fg_mask=(family[0u]>>8u)&255u;ASSERT_NE(fg_mask,0u);ASSERT_NE(cbc[86u]&1u,0u);
      const auto blacklike=[&](UINT p){UINT df=255u;for(UINT mode=0u;mode<8u;++mode)if(fg_mask&(1u<<mode))df=std::min(df,box_distance(p,family[1u+2u*mode],family[2u+2u*mode]));return box_distance(p,cbc[395u],cbc[396u])<=df;};
      nlohmann::json truth_counts=nlohmann::json::object();UINT tp=0u,fp=0u,fn=0u;
      for(UINT i=0u;i<count;++i){ASSERT_LE(m[i],1u);if(m[i]&&alpha[i])++tp;if(m[i]&&!alpha[i])++fp;if(!m[i]&&alpha[i])++fn;}
      truth_counts={{"TP",tp},{"FP",fp},{"FN",fn}};manifest["whole_frame_literal_truth_counts"]=truth_counts;
      nlohmann::json primary_rows=nlohmann::json::array();for(UINT p:primary)primary_rows.push_back({{"point",{p%W_,p/W_}},{"alpha",alpha[p]},{"M",m[p]},{"blacklike_from_captured_family",blacklike(p)}});
      manifest["primary_literal_truth"]=primary_rows;save();
      if(control) {
        for(UINT p:primary)ASSERT_EQ(m[p],0u)<<"Control target must really be missing before treatment";
        std::vector<UINT> component{q};std::vector<std::uint8_t> seen(count,0u);seen[q]=1u;bool closed=true;
        for(std::size_t k=0u;k<component.size();++k){const UINT p=component[k],x=p%W_,y=p/W_;
          if(x==0u||y==0u||x+1u==W_||y+1u==H_){closed=false;continue;}
          for(auto d:n4){const UINT z=UINT(int(y)+d[1])*W_+UINT(int(x)+d[0]);if(!seen[z]&&m[z]==0u){seen[z]=1u;component.push_back(z);}}
        }
        manifest["control_component_points"]=component;manifest["control_component_closed"]=closed;save();ASSERT_TRUE(closed);
        UINT min_distance=W_+H_,boundary_distance=W_+H_;bool c3_boundary=false,strict_any=false,cardinal_any=false,all_blacklike=true;
        const auto root=[&](UINT p){UINT n=p+1u;for(std::size_t k=0u;k<parent.size();++k){if(n>=parent.size())return 0u;const UINT next=parent[n];if(next==n)return n;if(next==0u||next>n)return 0u;n=next;}return 0u;};
        const std::array<UINT,4u> roi{cbc[152u],cbc[153u],cbc[154u],cbc[155u]};
        const auto in_domain=[&](UINT p){const UINT x=p%W_,y=p/W_;return (cover[p]&1u)&&x>=roi[0]&&x<roi[2]&&y>=roi[1]&&y<roi[3];};
        const auto c3=[&](UINT p){if(!in_domain(p)||passable[p]!=1u)return false;bool physical=true;for(UINT c=0u;c<3u;++c)physical=physical&&origin_pixels_[p][c]>=kl[c]&&origin_pixels_[p][c]<=kh[c];if(!physical)return false;const UINT r=root(p);return r&&((flags[r]&~3u)==0u)&&((flags[r]&3u)==1u)&&(witness[r-1u]&1u)!=0u&&(witness[r-1u]&~3u)==0u;};
        std::vector<UINT> original_anchors;for(UINT i=0u;i<count;++i)if(in_domain(i)&&((low_high[i]&1u)||inner[i]==1u))original_anchors.push_back(i);
        const auto distance=[&](UINT p){UINT result=W_+H_;for(UINT i:original_anchors)result=std::min(result,UINT(std::max(std::abs(int(i%W_)-int(p%W_)),std::abs(int(i/W_)-int(p/W_)))));return result;};
        const auto strict=[](UINT mask){if(std::popcount(mask)<3)return false;for(UINT start=0u;start<8u;++start){UINT half=0u;for(UINT step=0u;step<=4u;++step)half|=1u<<((start+step)&7u);if((mask&~half)==0u)return false;}return true;};
        const std::array<std::array<int,2u>,8u> dirs{{{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1}}};
        nlohmann::json component_rows=nlohmann::json::array();
        for(UINT p:component){ASSERT_TRUE(in_domain(p));ASSERT_EQ(cover[p]&1u,1u);ASSERT_EQ(inner[p],0u);ASSERT_EQ(original[p],0u);all_blacklike=all_blacklike&&blacklike(p);min_distance=std::min(min_distance,distance(p));
          const int x=int(p%W_),y=int(p/W_);UINT high_mask=0u;
          for(UINT ray=0u;ray<8u;++ray)for(UINT step=1u;step<=rmax;++step){const int rx=x+int(step)*dirs[ray][0],ry=y+int(step)*dirs[ray][1];if(rx<0||ry<0||rx>=int(W_)||ry>=int(H_))break;const UINT z=UINT(ry)*W_+UINT(rx);if(!(cover[z]&1u))break;if(low_high[z]&256u){high_mask|=1u<<ray;break;}}
          strict_any=strict_any||strict(high_mask);cardinal_any=cardinal_any||((high_mask&0x55u)==0x55u);
          for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx)if(dx||dy){const int rx=x+dx,ry=y+dy;if(rx<0||ry<0||rx>=int(W_)||ry>=int(H_))continue;const UINT z=UINT(ry)*W_+UINT(rx);if(!m[z]||!in_domain(z))continue;boundary_distance=std::min(boundary_distance,distance(z));if(c3(z))c3_boundary=true;}
          component_rows.push_back({{"point",{p%W_,p/W_}},{"first_original_HIGH_mask",high_mask},{"strict8_HIGH_surround",strict(high_mask)}});
        }
        manifest["control_readiness"]={{"completed",false},{"component_size",component.size()},{"minimum_member_distance",min_distance},{"minimum_N8_M1_boundary_distance",boundary_distance},{"has_N8_original_C3_boundary",c3_boundary},{"any_strict8_HIGH_surround",strict_any},{"any_all4_HIGH_counter",cardinal_any},{"all_members_blacklike",all_blacklike},{"component_HIGH_rows",component_rows}};save();
        ASSERT_GT(min_distance,thickness);ASSERT_LE(boundary_distance,thickness);ASSERT_TRUE(c3_boundary);
        if(positive){ASSERT_FALSE(strict_any);ASSERT_TRUE(all_blacklike);}else{ASSERT_TRUE(strict_any);ASSERT_FALSE(cardinal_any);if(scene_kind>=2u)ASSERT_TRUE(all_blacklike);}
        manifest["control_readiness"]["completed"]=true;
      }
      manifest["prerequisites_not_yet_asserted"]=false;save();
      for(UINT p:primary)EXPECT_EQ(m[p],positive?1u:0u)<<label<<" source alpha remains independent";
    }


    void recolour_literal_black(pixel8 colour) {
      for (auto &pixel : origin_pixels_) {
        if (pixel[0] == 0u && pixel[1] == 0u && pixel[2] == 0u) {
          pixel = colour;
        }
      }
      upload_pixels_from_origin();
    }

    void author_body_black_overlap(UINT frame, bool overlap) {
      const pixel8 black {60u, 70u, 90u, 255u}, foreground {119u, 139u, 179u, 255u};
      author_band(5u, frame);
      rect(0u, 0u, W_, H_, {200u, 0u, 0u, 255u});
      rect(74u, 43u, 116u, 93u, black);
      rect(79u, 48u, 111u, 88u, {255u, 255u, 255u, 255u});
      rect(155u, 43u, 213u, 93u, black);
      rect(160u, 48u, 208u, 88u, {255u, 255u, 255u, 255u});
      // Distinct palette bins: WHITE63, RED48, FG26. K90 is a black
      // neighbour for WHITE255, not RED200 or FG179; it is eligible dim FG.
      rect(84u, 60u, 106u, 80u, foreground);
      if (overlap) rect(94u, 65u, 98u, 72u, black);
      rect(88u, 68u, 89u, 69u, {0u, 0u, 0u, 255u});
      // All bilinear contributors throughout native [84,93)x[64,73).
      // Only S seeds FG at (90,68). Grid rows43/50/57/64/71/78/85/92
      // and their radius2 axial neighbours never sample the literal0 at y68.
      for (UINT y = 80u; y < 92u; ++y) for (UINT x = 316u; x < 348u; ++x) {
        probability_words_[y * probability_width + x] = 0.0f;
      }
      support_words_[68u * W_ + 90u] = 1u;
      upload_pixels_from_origin();
    }

    void expect_inherited_unknown_final(const std::string &label) {
      const auto original = words(read(helper_.baseline_views().body.Get()));
      const auto code1 = words(matte()), final_m = words(outline_matte());
      require(original.size() == W_ * H_ && code1.size() == original.size() && final_m.size() == original.size());
      expect_baseline();expect_ring_parent();
      std::size_t extras = 0u;
      std::map<UINT, std::size_t> colours;
      for (std::size_t i = 0u; i < original.size(); ++i) {
        EXPECT_LE(final_m[i], 1u);
        EXPECT_LE(original[i], final_m[i]);
        EXPECT_LE(final_m[i], code1[i]);
        if (final_m[i] != 0u && original[i] == 0u) {
          ++extras;const auto &p = origin_pixels_[i];
          ++colours[UINT(p[0]) | (UINT(p[1]) << 8u) | (UINT(p[2]) << 16u)];
        }
      }
      nlohmann::json histogram = nlohmann::json::array();
      for (const auto &[colour, count] : colours) {
        histogram.push_back({{"RGB8", {colour & 255u, (colour >> 8u) & 255u, (colour >> 16u) & 255u}}, {"pixels", count}});
      }
      // Valid UNKNOWN retains LOW BODY and existing current Code1 certificates.
      // This varied-palette raster deliberately does not assert BODY-only output.
      RecordProperty(label + "_inherited_Code1_extra_M", extras);
      RecordProperty(label + "_inherited_Code1_extra_RGB8", histogram.dump());
      RecordProperty(label + "_source_rgba8_sha256", hash_bytes(raw(origin_pixels_)));
      RecordProperty(label + "_P_sha256", hash_bytes(raw(probability_words_)));
      RecordProperty(label + "_S_sha256", hash_bytes(raw(support_words_)));
      RecordProperty(label + "_original_BODY_sha256", hash_bytes(raw(original)));
      RecordProperty(label + "_Code1_sha256", hash_bytes(raw(code1)));
      RecordProperty(label + "_final_M_sha256", hash_bytes(raw(final_m)));
    }

    void record_uniform_black_state(const std::string &label, UINT unit = 0u) {
      const auto cbc = receipt(), raw_meta = ring_receipt(), f2 = calibration();
      require(cbc.size() == 512u && raw_meta.size() == 512u && f2.size() == 68608u && unit < 8u);
      const auto slice = [](const auto &value, UINT first, UINT count) {
        return nlohmann::json(std::vector<UINT>(value.begin() + first, value.begin() + first + count)).dump();
      };
      RecordProperty(label + "_actual_K16", slice(cbc, 384u + 16u * unit, 16u));
      RecordProperty(label + "_actual_CBC_unit", slice(cbc, 128u + 32u * unit, 32u));
      RecordProperty(label + "_actual_Raw_unit", slice(raw_meta, 128u + 32u * unit, 32u));
      RecordProperty(label + "_actual_F2_unit", slice(f2, 128u + 64u * unit, 64u));
    }

    void expect_uniform_known(const std::vector<UINT> &meta, UINT unit, pixel8 colour, UINT version) {
      require(meta.size() == 512u && unit < 8u);
      const UINT k = 384u + 16u * unit;
      const UINT prototype = UINT(colour[0]) | (UINT(colour[1]) << 8u) | (UINT(colour[2]) << 16u);
      EXPECT_EQ(meta[k], 0x314b4c42u);
      EXPECT_EQ(meta[k + 1u], version);
      EXPECT_EQ(meta[k + 2u], 1u);
      EXPECT_EQ(meta[k + 3u], 1u);
      EXPECT_GE(meta[k + 4u], 2u);
      EXPECT_EQ(meta[k + 5u], prototype);
      EXPECT_EQ(meta[k + 6u], meta[k + 4u]);
      EXPECT_GE(meta[k + 7u], 2u);
      EXPECT_EQ(meta[k + 8u], meta[k + 7u]);
      EXPECT_EQ(meta[k + 9u], meta[k + 4u]);
      EXPECT_EQ(meta[k + 10u], std::max({UINT(colour[0]), UINT(colour[1]), UINT(colour[2])}));
      EXPECT_EQ(meta[k + 11u], prototype);
      EXPECT_EQ(meta[k + 12u], prototype);
      EXPECT_EQ(meta[k + 13u], 512u);
      EXPECT_EQ(meta[k + 14u], 0u);
      EXPECT_EQ(meta[k + 15u], 0u);
    }

    void republish_enclosure(const std::vector<UINT> &meta) {
      // Fault-injection seam after an actual complete native run. This invokes the
      // production final publisher with the real owned resources and constants; it
      // does not author BODY, selected pixels, root labels, width or histogram data.
      require(meta.size() == 512u);
      const auto e = helper_.enclosure_views();
      const auto b = helper_.baseline_views();
      ComPtr<ID3D11Resource> resource;
      e.receipt->GetResource(&resource);
      context_->UpdateSubresource(resource.Get(), 0u, nullptr, meta.data(), 0u, 0u);
      const std::array<UINT, 4u> hook {};
      context_->UpdateSubresource(e.hook_constants.Get(), 0u, nullptr, hook.data(), 0u, 0u);
      std::array<ID3D11ShaderResourceView *, 16u> inputs {};
      std::array<ID3D11UnorderedAccessView *, 8u> outputs {};
      context_->CSSetShaderResources(0u, 16u, inputs.data());
      context_->CSSetUnorderedAccessViews(0u, 8u, outputs.data(), nullptr);
      inputs = {rgb_.srv.Get(), ocr_.srv.Get(), b.validation_receipt.Get(), b.palette.Get(),
        b.mode_body.Get(), b.cover_clip.Get(), b.parent.Get(), b.root_flags.Get(),
        e.all_body_bits.Get(), e.cover_bits.Get(), sct_.srv.Get(), uip_.srv.Get(),
        root_.srv.Get(), program_.enclosure.producer.original_source, b.body.Get(), b.matte.Get()};
      auto meta_uav = uav(e.receipt.Get()), selected_uav = uav(e.selected_bits.Get());
      auto body_uav = uav(e.body.Get()), matte_uav = uav(e.matte.Get());
      outputs[0] = meta_uav.Get();outputs[3] = selected_uav.Get();
      outputs[4] = body_uav.Get();outputs[5] = matte_uav.Get();
      ID3D11Buffer *constants[3] {cb_.resource.Get(), e.hook_constants.Get(), e.producer_constants.Get()};
      context_->CSSetConstantBuffers(0u, 3u, constants);
      context_->CSSetShaderResources(0u, 16u, inputs.data());
      context_->CSSetUnorderedAccessViews(0u, 8u, outputs.data(), nullptr);
      context_->CSSetShader(enclosure_[11u].Get(), nullptr, 0u);
      context_->Dispatch((W_ + 15u) / 16u, (H_ + 15u) / 16u, 1u);
      inputs.fill(nullptr);outputs.fill(nullptr);
      context_->CSSetShaderResources(0u, 16u, inputs.data());
      context_->CSSetUnorderedAccessViews(0u, 8u, outputs.data(), nullptr);
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

    void dispatch_current_coverage_probe(bool commit) {
      // Test-only invocation of the actual support/final-commit entrypoints
      // over a completed current owner. No detector/model fields are authored.
      const auto v=helper_.mutable_views();const auto e=helper_.enclosure_views();
      const auto rv=helper_.raw_views();
      const auto cbc=receipt();require(cbc.size()==512u);
      auto hook=buffer(16u,true),shape=buffer(16u,true),mode=buffer(16u,true);
      const std::array<UINT,4u> h{commit?8u:0u,0u,0u,0u},s{W_,H_,W_*H_,0u},d{1u,0u,0u,0u};
      context_->UpdateSubresource(hook.resource.Get(),0u,nullptr,h.data(),0u,0u);
      context_->UpdateSubresource(shape.resource.Get(),0u,nullptr,s.data(),0u,0u);
      context_->UpdateSubresource(mode.resource.Get(),0u,nullptr,d.data(),0u,0u);
      ComPtr<ID3D11Resource> data_resource;v.calibration_data->GetResource(&data_resource);
      D3D11_UNORDERED_ACCESS_VIEW_DESC dd{};dd.Format=DXGI_FORMAT_R32_TYPELESS;
      dd.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;dd.Buffer.NumElements=68608u;dd.Buffer.Flags=D3D11_BUFFER_UAV_FLAG_RAW;
      ComPtr<ID3D11UnorderedAccessView> data;require(SUCCEEDED(device_->CreateUnorderedAccessView(data_resource.Get(),&dd,&data)));
      auto parent=uav(helper_.mask_hole_parents()),flags=uav(helper_.mask_hole_flags());
      auto mask=uav(v.matte.Get()),offers=uav(helper_.stroke_coverage_offers());
      std::array<ID3D11ShaderResourceView*,16u> in{rgb_.srv.Get(),ocr_.srv.Get(),e.selected_bits.Get(),e.cover_bits.Get(),e.receipt.Get(),rv.receipt.Get(),
        nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,rv.matte.Get(),helper_.mask_hole_proof(),e.matte.Get(),e.family_models.Get()};
      std::array<ID3D11UnorderedAccessView*,8u> out{};out[1]=parent.Get();out[2]=flags.Get();out[5]=data.Get();
      if(commit)out[4]=mask.Get();else out[6]=offers.Get();
      ID3D11Buffer* cb[4]{cb_.resource.Get(),hook.resource.Get(),shape.resource.Get(),mode.resource.Get()};
      context_->CSSetConstantBuffers(0u,4u,cb);context_->CSSetShaderResources(0u,16u,in.data());
      context_->CSSetUnorderedAccessViews(0u,8u,out.data(),nullptr);
      context_->CSSetShader(commit?hole_shaders_[7u].Get():coverage_shaders_[2u].Get(),nullptr,0u);
      if(commit)context_->Dispatch((W_+15u)/16u,(H_+15u)/16u,1u);
      else context_->Dispatch((cbc[154u]-cbc[152u]+15u)/16u,(cbc[155u]-cbc[153u]+15u)/16u,1u);
      in.fill(nullptr);out.fill(nullptr);
      context_->CSSetUnorderedAccessViews(0u,8u,out.data(),nullptr);context_->CSSetShaderResources(0u,16u,in.data());
      ID3D11Buffer* clear[4]{};context_->CSSetConstantBuffers(0u,4u,clear);
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
      auto raw_component_witness = uav(b.mode_final.Get());
      std::array<ID3D11UnorderedAccessView *, 8u> out {};
      out[0] = data.Get();
      out[1] = m.Get();
      out[2] = cache.Get();
      // Match the production aggregate binding. The fault under test must come
      // from the corrupted cache/parent, not an absent current witness texture.
      out[3] = raw_component_witness.Get();
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
      if(h[0u]==0x31435342u){EXPECT_EQ(h[105u],50380800u);EXPECT_EQ(h[106u],50397183u);}
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
    void saved_current(bool sdr);
    #include "current_outline_foreground_edge_fixture.inc"
    #include "current_outline_trusted_inner_authored_fixture.inc"
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
    std::array<ComPtr<ID3D11ComputeShader>, models::host_sbs_current_mask_holes_gpu::shader_count> hole_shaders_;
    std::string hole_closure_;
    std::array<ComPtr<ID3D11ComputeShader>, 3u> coverage_shaders_;
    std::array<ComPtr<ID3D11ComputeShader>, 1u> trusted_inner_shaders_;
    std::string coverage_closure_;
    nlohmann::json adapter_;
  };

  #include "current_outline_foreground_edge_tests.inc"
  #include "current_outline_trusted_inner_authored_tests.inc"

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
    for (const bool touches_cover : {false, true}) {
      SCOPED_TRACE(touches_cover);
      author(true, touches_cover ? 3u : 2u, 1u, false, false, false, 64u, 32u);
      // The closed twin stops one pixel before the true cover edge at X55. Adding that
      // one black pixel joins the same seeded black component to the domain boundary.
      const UINT end = touches_cover ? 56u : 55u;
      for (UINT x = 48u; x < end; ++x) {
        origin_pixels_[16u * W_ + x] = {1u, 1u, 1u, 255u};
      }
      upload_pixels_from_origin();
      ASSERT_TRUE(run()) << error_;
      expect_ready();
      expect_ring_ready();
      expect_outline_accounting();
      const auto cover = words(read(helper_.enclosure_views().cover_bits.Get()));
      const auto parent = words(read(raw_parent_snapshot_.resource.Get()));
      const auto flags = words(read(raw_flags_snapshot_.resource.Get()));
      const auto original = words(read(helper_.baseline_views().body.Get()));
      const auto selected = words(body()), code1 = words(matte()), raw = words(ring_matte()), final = words(outline_matte());
      ASSERT_EQ(cover.size(), W_ * H_);
      ASSERT_EQ(original.size(), cover.size());
      ASSERT_EQ(selected.size(), cover.size());
      ASSERT_EQ(code1.size(), cover.size());
      ASSERT_EQ(raw.size(), cover.size());
      ASSERT_EQ(final.size(), cover.size());
      ASSERT_EQ(parent.size(), cover.size() + 1u);
      ASSERT_EQ(flags.size(), parent.size());
      ASSERT_EQ(cover[16u * W_ + 55u] & 1u, 1u);
      ASSERT_EQ(cover[16u * W_ + 56u] & 1u, 0u);
      auto root = [&](UINT id) {
        for (std::size_t hops = 0u; hops < parent.size(); ++hops) {
          require(id != 0u && id < parent.size());
          const UINT next = parent[id];
          require(next != 0u && next != 0xffffffffu && next <= id);
          if (next == id) {
            return id;
          }
          id = next;
        }
        throw std::runtime_error("Non-convergent native black component");
      };
      constexpr UINT branch_x = 51u, branch_y = 16u;
      const UINT black_root = root(branch_y * W_ + branch_x + 1u);
      EXPECT_EQ(flags[black_root] & 3u, touches_cover ? 3u : 1u);
      // These are actual Raw graph flags: the rejection concerns extra enclosure/black
      // protection. It must not withdraw the original current white BODY glyph.
      ASSERT_EQ(code1[branch_y * W_ + branch_x], 0u);
      EXPECT_EQ(raw[branch_y * W_ + branch_x], touches_cover ? 0u : 1u);
      if (touches_cover) {
        expect_ring_parent();
      }
      for (UINT y = 12u; y < 22u; ++y) {
        for (UINT x = 20u; x < 44u; ++x) {
          const UINT pixel = y * W_ + x;
          ASSERT_EQ(origin_pixels_[pixel], (pixel8 {255u, 255u, 255u, 255u}));
          EXPECT_EQ(original[pixel], 1u);
          EXPECT_EQ(selected[pixel], 1u);
          EXPECT_EQ(final[pixel], 1u);
        }
      }
    }
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

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, OutwardRawLobeUsesIndependentBackgroundWitness) {
    author_band(3u, 880u);
    // A local outline lobe is wider than the independently measured style width.
    // Its own Raw root reaches current exterior background, outside the old BAND.
    rect(70u, 55u, 76u, 65u, {0u, 0u, 0u, 255u});
    rect(69u, 60u, 70u, 61u, {64u, 64u, 64u, 255u});
    upload_pixels_from_origin();
    ASSERT_TRUE(run()) << error_;
    expect_ready();expect_ring_ready();expect_outline_accounting();
    const auto cbc = receipt(), raw_meta = ring_receipt(), f2 = calibration();
    ASSERT_EQ(cbc.size(), 512u);ASSERT_EQ(raw_meta.size(), 512u);ASSERT_EQ(f2.size(), 68608u);
    expect_uniform_known(cbc, 0u, {0u, 0u, 0u, 255u}, 3u);
    ASSERT_EQ(cbc[128u], 1u);ASSERT_EQ(cbc[129u], 1u);
    ASSERT_GT(cbc[131u], 0u);ASSERT_EQ(cbc[134u], 1u);ASSERT_EQ(cbc[157u], 1u);
    ASSERT_EQ(raw_meta[128u], 1u);ASSERT_EQ(raw_meta[129u], 1u);
    ASSERT_EQ(f2[139u], 3u);ASSERT_EQ(f2[141u], 1u);ASSERT_EQ(f2[164u], 0u);
    const auto original = words(read(helper_.baseline_views().body.Get()));
    const auto selected = words(body()), code1 = words(matte()), raw_m = words(ring_matte());
    const auto final_m = words(outline_matte()), packed = words(read(packed_snapshot_.resource.Get()));
    ASSERT_EQ(original.size(), W_ * H_);ASSERT_EQ(selected.size(), original.size());
    ASSERT_EQ(code1.size(), original.size());ASSERT_EQ(raw_m.size(), original.size());
    ASSERT_EQ(final_m.size(), original.size());ASSERT_EQ(packed.size(), original.size());
    EXPECT_EQ(selected, original);
    constexpr UINT opaque_offer = 1u << 25u;
    const UINT black = 60u * W_ + 70u, aa = 60u * W_ + 69u, background = 60u * W_ + 68u;
    ASSERT_EQ(selected[black], 0u);ASSERT_EQ(code1[black], 0u);ASSERT_EQ(raw_m[black], 1u);
    EXPECT_EQ(packed[black] & 0x3fffu, 0u);  // No measured BAND, rim, AA or closure certificate.
    EXPECT_EQ(packed[black] & opaque_offer, opaque_offer);
    EXPECT_EQ(final_m[black], 1u);
    ASSERT_EQ(selected[aa], 0u);ASSERT_EQ(code1[aa], 0u);ASSERT_EQ(raw_m[aa], 0u);
    EXPECT_EQ(packed[aa] & opaque_offer, 0u);
    EXPECT_EQ(packed[aa] & 0x3fffu, 512u);  // Existing ordinary OuterRays AA, without OLD_AA/BAND.
    EXPECT_EQ(final_m[aa], 1u);
    EXPECT_EQ(packed[background] & (opaque_offer | 0x3fffu), 0u);
    EXPECT_EQ(final_m[background], 0u);
    record_uniform_black_state("outward_raw_background_witness");
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, ClosedBlackCounterDoesNotCreateOpaqueOffer) {
    author_band(3u, 881u);
    // The black counter shares the outline's RGB and is seeded by its surrounding
    // white BODY. Its centre is farther than the current width from any nonblack
    // sample; the nearer nonblack wall is BODY, which cannot prove background.
    rect(172u, 60u, 196u, 76u, {0u, 0u, 0u, 255u});
    upload_pixels_from_origin();
    ASSERT_TRUE(run()) << error_;
    expect_ready();expect_ring_ready();expect_outline_accounting();
    const auto cbc = receipt(), raw_meta = ring_receipt(), f2 = calibration();
    ASSERT_EQ(cbc.size(), 512u);ASSERT_EQ(raw_meta.size(), 512u);ASSERT_EQ(f2.size(), 68608u);
    expect_uniform_known(cbc, 0u, {0u, 0u, 0u, 255u}, 3u);
    ASSERT_EQ(cbc[128u], 1u);ASSERT_EQ(cbc[129u], 1u);
    ASSERT_GT(cbc[131u], 0u);ASSERT_EQ(cbc[134u], 1u);ASSERT_EQ(cbc[157u], 1u);
    ASSERT_EQ(raw_meta[128u], 1u);ASSERT_EQ(raw_meta[129u], 1u);
    ASSERT_EQ(f2[139u], 3u);ASSERT_EQ(f2[141u], 1u);ASSERT_EQ(f2[164u], 0u);
    const auto original = words(read(helper_.baseline_views().body.Get()));
    const auto selected = words(body()), code1 = words(matte()), raw_m = words(ring_matte());
    const auto final_m = words(outline_matte()), packed = words(read(packed_snapshot_.resource.Get()));
    ASSERT_EQ(original.size(), W_ * H_);ASSERT_EQ(selected.size(), original.size());
    ASSERT_EQ(code1.size(), original.size());ASSERT_EQ(raw_m.size(), original.size());
    ASSERT_EQ(final_m.size(), original.size());ASSERT_EQ(packed.size(), original.size());
    EXPECT_EQ(selected, original);
    constexpr UINT opaque_offer = 1u << 25u;
    const UINT centre = 68u * W_ + 184u;
    ASSERT_EQ(selected[centre], 0u);ASSERT_EQ(code1[centre], 0u);ASSERT_EQ(raw_m[centre], 1u);
    EXPECT_EQ(packed[centre] & 0x3fffu, 0u);
    EXPECT_EQ(final_m[centre], 0u);  // A current Raw member alone cannot fill the scene counter.
    for (UINT y = 60u; y < 76u; ++y) for (UINT x = 172u; x < 196u; ++x) {
      EXPECT_EQ(packed[y * W_ + x] & opaque_offer, 0u) << x << ',' << y;
    }
    record_uniform_black_state("closed_counter_without_background_witness");
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, ThickRawJunctionUsesComponentBackgroundWithoutFillingClosedCounter) {
    author_band(3u, 882u);
    rect(68u, 54u, 76u, 68u, {0u, 0u, 0u, 255u});  // Thick outer junction, connected to left outline.
    rect(172u, 60u, 196u, 76u, {0u, 0u, 0u, 255u});  // Independent closed scene counter.
    rect(40u, 40u, 50u, 50u, {0u, 0u, 0u, 255u});  // Independent unseeded scene island.
    upload_pixels_from_origin();
    ASSERT_TRUE(run()) << error_;
    expect_ready();expect_ring_ready();expect_outline_accounting();
    const auto cbc = receipt(), raw_meta = ring_receipt(), f2 = calibration();
    ASSERT_EQ(cbc.size(), 512u);ASSERT_EQ(raw_meta.size(), 512u);ASSERT_EQ(f2.size(), 68608u);
    expect_uniform_known(cbc, 0u, {0u, 0u, 0u, 255u}, 3u);
    ASSERT_EQ(cbc[128u], 1u);ASSERT_EQ(cbc[129u], 1u);
    ASSERT_GT(cbc[131u], 0u);ASSERT_EQ(cbc[134u], 1u);ASSERT_EQ(cbc[157u], 1u);
    ASSERT_EQ(raw_meta[6u], 1u);ASSERT_EQ(raw_meta[128u], 1u);ASSERT_EQ(raw_meta[129u], 1u);
    ASSERT_EQ(f2[139u], 3u);ASSERT_EQ(f2[141u], 1u);ASSERT_EQ(f2[164u], 0u);
    const auto b = helper_.baseline_views();const auto r = helper_.raw_views();
    const auto original = words(read(b.body.Get())), selected = words(body());
    const auto code1 = words(matte()), raw_m = words(ring_matte()), final_m = words(outline_matte());
    const auto packed = words(read(packed_snapshot_.resource.Get()));
    const auto parent = words(read(b.parent.Get())), flags = words(read(b.root_flags.Get()));
    const auto passable = words(read(b.eligible.Get())), inner = words(read(r.eligible_inner.Get()));
    const auto cache_words = root_cache(), witnessed = words(read(b.mode_final.Get()));
    ASSERT_EQ(original.size(), W_ * H_);ASSERT_EQ(selected.size(), original.size());
    for (const auto *v : {&code1, &raw_m, &final_m, &packed, &passable, &inner, &cache_words, &witnessed}) {
      ASSERT_EQ(v->size(), original.size());
    }
    ASSERT_EQ(parent.size(), original.size() + 1u);ASSERT_EQ(flags.size(), parent.size());
    EXPECT_EQ(selected, original);
    auto root = [&](UINT pixel) {
      UINT id = pixel + 1u;
      for (std::size_t hops = 0u; hops < parent.size(); ++hops) {
        require(id > 0u && id < parent.size());
        const UINT next = parent[id];require(next > 0u && next <= id);
        if (next == id) return id;
        id = next;
      }
      throw std::runtime_error("Non-convergent actual Raw parent graph");
    };
    constexpr UINT opaque_offer = 1u << 25u, root_mask = 0x3fffffffu, enclosed = 0x80000000u;
    const UINT junction = 61u * W_ + 72u, counter = 68u * W_ + 184u, scene = 45u * W_ + 45u;
    const UINT junction_root = root(junction), counter_root = root(counter), scene_root = root(scene);
    ASSERT_NE(junction_root, counter_root);ASSERT_NE(junction_root, scene_root);ASSERT_NE(counter_root, scene_root);
    ASSERT_EQ(flags[junction_root] & 3u, 1u);ASSERT_EQ(flags[counter_root] & 3u, 1u);
    ASSERT_EQ(flags[scene_root] & 3u, 0u);
    for (UINT q : {junction, counter}) {
      ASSERT_EQ(passable[q], 1u);ASSERT_EQ(selected[q], 0u);ASSERT_EQ(code1[q], 0u);ASSERT_EQ(raw_m[q], 1u);
      ASSERT_EQ(packed[q] & 0x3fffu, 0u);
    }
    // Every first-t leftward sample from this deep point is still source black.
    // Its component has independent exterior background at x67, five pixels away.
    for (UINT distance = 1u; distance <= 3u; ++distance) {
      ASSERT_EQ(origin_pixels_[61u * W_ + 72u - distance], (pixel8 {0u, 0u, 0u, 255u}));
    }
    const UINT exterior = 61u * W_ + 67u, wall = 68u * W_ + 171u;
    ASSERT_EQ(origin_pixels_[exterior], (pixel8 {128u, 128u, 128u, 255u}));
    ASSERT_EQ(selected[exterior], 0u);ASSERT_EQ(inner[exterior], 0u);
    const UINT exterior_root = cache_words[exterior] & root_mask;
    ASSERT_GT(exterior_root, 0u);ASSERT_LE(exterior_root, W_ * H_);
    ASSERT_EQ(cache_words[exterior] & enclosed, 0u);
    ASSERT_EQ(cache_words[exterior_root - 1u] & (root_mask | enclosed), exterior_root);
    ASSERT_EQ(origin_pixels_[wall], (pixel8 {255u, 255u, 255u, 255u}));
    ASSERT_EQ(selected[wall], 1u);ASSERT_EQ(inner[wall], 1u);
    ASSERT_NE(cache_words[wall] & enclosed, 0u);  // Counter wall is seeded interior, not background.
    EXPECT_EQ(witnessed[junction_root - 1u], 1u);
    EXPECT_EQ(witnessed[counter_root - 1u], 0u);
    EXPECT_EQ(witnessed[scene_root - 1u], 0u);
    EXPECT_EQ(packed[junction] & opaque_offer, opaque_offer);
    EXPECT_EQ(final_m[junction], 1u);
    EXPECT_EQ(packed[counter] & opaque_offer, 0u);EXPECT_EQ(final_m[counter], 0u);
    EXPECT_EQ(code1[scene], 0u);EXPECT_EQ(raw_m[scene], 0u);
    EXPECT_EQ(packed[scene] & opaque_offer, 0u);EXPECT_EQ(final_m[scene], 0u);
    record_uniform_black_state("thick_junction_separate_counter_and_scene");
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, PointwiseRawGapRequiresSameMeasuredRootAndCannotRecurse) {
    enum class scenario {same_root, different_roots, two_pixel_gap, two_pixel_different_roots, multiple_pixel_gap,
      three_pixel_t4_gap, three_pixel_t4_different_roots};
    UINT frame = 990u;
    for (const auto kind : {scenario::same_root, scenario::different_roots, scenario::two_pixel_gap, scenario::two_pixel_different_roots,
      scenario::multiple_pixel_gap, scenario::three_pixel_t4_gap, scenario::three_pixel_t4_different_roots}) {
      SCOPED_TRACE(static_cast<int>(kind));
      const bool t4 = kind == scenario::three_pixel_t4_gap || kind == scenario::three_pixel_t4_different_roots;
      const UINT thickness = t4 ? 4u : 3u;
      author_band(thickness, frame++);
      rect(t4 ? 73u : 74u, 59u, t4 ? 77u : 76u, 62u, {0u, 0u, 0u, 255u});
      if (kind == scenario::different_roots || kind == scenario::two_pixel_different_roots) {
        // p is attached to a separately seeded outlined peer. Clearing the
        // entire intervening column keeps both accepted black roots distinct.
        rect(47u, 45u, 73u, 91u, {0u, 0u, 0u, 255u});
        rect(50u, 48u, 70u, 88u, {255u, 255u, 255u, 255u});
        rect(73u, 59u, 75u, 62u, {0u, 0u, 0u, 255u});
        rect(75u, 24u, kind == scenario::two_pixel_different_roots ? 77u : 76u, 112u, {128u, 128u, 128u, 255u});
      }
      rect(75u, 60u, 76u, 61u, {64u, 64u, 64u, 255u});
      if (kind == scenario::two_pixel_gap || kind == scenario::two_pixel_different_roots) {
        // Both gray pixels stay source nonblack. Only the same-root twin gets
        // a remote crown; the distinct-root twin preserves its cleared columns.
        rect(75u, 59u, 77u, 62u, {64u, 64u, 64u, 255u});
        if (kind == scenario::two_pixel_gap) {
          rect(74u, 58u, 77u, 59u, {0u, 0u, 0u, 255u});
        }
      }
      if (kind == scenario::multiple_pixel_gap) {
        // The same Raw root remains connected over a remote black crown.
        // Existing AA near x77 may not propagate through gray x76 to q=x75.
        rect(75u, 59u, 78u, 62u, {64u, 64u, 64u, 255u});
        rect(74u, 58u, 78u, 59u, {0u, 0u, 0u, 255u});
        // Block the otherwise valid max2 diagonal endpoints for target q75,60.
        // Keep the old AA bracket (76,58)->(77,59)->(78,60) intact.
        rect(77u, 58u, 78u, 59u, {64u, 64u, 64u, 255u});
        rect(77u, 62u, 78u, 63u, {64u, 64u, 64u, 255u});
      }
      if (t4) {
        if (kind == scenario::three_pixel_t4_different_roots) {
          // End the independent white peer at x66: its t4 envelope must not
          // reach p73. Three cleared columns isolate the original right rim.
          rect(44u, 44u, 71u, 92u, {0u, 0u, 0u, 255u});
          rect(48u, 48u, 67u, 88u, {255u, 255u, 255u, 255u});
          rect(71u, 59u, 74u, 62u, {0u, 0u, 0u, 255u});
          rect(74u, 24u, 77u, 112u, {128u, 128u, 128u, 255u});
        }
        rect(74u, 59u, 77u, 62u, {64u, 64u, 64u, 255u});
        if (kind == scenario::three_pixel_t4_gap) {
          rect(73u, 58u, 77u, 59u, {0u, 0u, 0u, 255u});
        }
      }
      upload_pixels_from_origin();
      ASSERT_TRUE(run()) << error_;
      expect_ready();expect_ring_ready();expect_outline_accounting();
      const auto cbc = receipt(), raw_meta = ring_receipt(), f2 = calibration();
      ASSERT_EQ(cbc.size(), 512u);ASSERT_EQ(raw_meta.size(), 512u);ASSERT_EQ(f2.size(), 68608u);
      expect_uniform_known(cbc, 0u, {0u, 0u, 0u, 255u}, 3u);
      ASSERT_EQ(cbc[128u], 1u);ASSERT_EQ(cbc[129u], 1u);
      ASSERT_GT(cbc[131u], 0u);ASSERT_EQ(cbc[134u], 1u);ASSERT_EQ(cbc[157u], 1u);
      ASSERT_EQ(raw_meta[6u], 1u);ASSERT_EQ(raw_meta[128u], 1u);ASSERT_EQ(raw_meta[129u], 1u);
      ASSERT_EQ(f2[139u], thickness);ASSERT_EQ(f2[141u], 1u);ASSERT_EQ(f2[164u], 0u);
      const auto b = helper_.baseline_views();const auto rv = helper_.raw_views();
      const auto original = words(read(b.body.Get())), selected = words(body());
      const auto code1 = words(matte()), raw_m = words(ring_matte()), final_m = words(outline_matte());
      const auto packed = words(read(packed_snapshot_.resource.Get()));
      const auto parent = words(read(b.parent.Get())), flags = words(read(b.root_flags.Get()));
      const auto passable = words(read(b.eligible.Get())), inner = words(read(rv.eligible_inner.Get()));
      const auto cache_words = root_cache(), cover = words(read(helper_.enclosure_views().cover_bits.Get()));
      ASSERT_EQ(original.size(), W_ * H_);ASSERT_EQ(selected.size(), original.size());
      for (const auto *v : {&code1, &raw_m, &final_m, &packed, &passable, &inner, &cache_words, &cover}) {
        ASSERT_EQ(v->size(), original.size());
      }
      ASSERT_EQ(parent.size(), original.size() + 1u);ASSERT_EQ(flags.size(), parent.size());
      EXPECT_EQ(selected, original);
      auto root = [&](UINT pixel) {
        UINT id = pixel + 1u;
        for (std::size_t hops = 0u; hops < parent.size(); ++hops) {
          require(id > 0u && id < parent.size());
          const UINT next = parent[id];require(next > 0u && next <= id);
          if (next == id) return id;
          id = next;
        }
        throw std::runtime_error("Non-convergent actual Raw gap graph");
      };
      const UINT p = 60u * W_ + (t4 ? 73u : 74u), q = 60u * W_ + (t4 ? 74u : 75u);
      const UINT r = 60u * W_ + (kind == scenario::multiple_pixel_gap ? 78u :
        (t4 || kind == scenario::two_pixel_gap || kind == scenario::two_pixel_different_roots) ? 77u : 76u);
      const UINT p_root = root(p), r_root = root(r);
      ASSERT_EQ(flags[p_root] & 3u, 1u);ASSERT_EQ(flags[r_root] & 3u, 1u);
      ASSERT_EQ(passable[p], 1u);ASSERT_EQ(passable[r], 1u);
      ASSERT_EQ(raw_m[p], 1u);ASSERT_EQ(raw_m[r], 1u);
      ASSERT_EQ(packed[p] & 1u, 0u);ASSERT_EQ(code1[p], 0u);ASSERT_EQ(selected[p], 0u);
      ASSERT_EQ(packed[r] & 1u, 1u);  // Opposite endpoint really is measured BAND + own Raw.
      ASSERT_EQ(origin_pixels_[q], (pixel8 {64u, 64u, 64u, 255u}));
      ASSERT_EQ(selected[q], 0u);ASSERT_EQ(code1[q], 0u);ASSERT_EQ(raw_m[q], 0u);
      ASSERT_EQ(passable[q], 0u);ASSERT_EQ(packed[q] & 255u, 0u);
      // q itself is an actual first-graph background endpoint one step from p.
      // This proves p's pointwise C2 witness without borrowing the component offer.
      constexpr UINT root_mask = 0x3fffffffu, enclosed = 0x80000000u;
      const UINT bg_root = cache_words[q] & root_mask;
      ASSERT_EQ(cover[q] & 1u, 1u);ASSERT_GT(bg_root, 0u);ASSERT_LE(bg_root, W_ * H_);
      ASSERT_EQ(inner[q], 0u);ASSERT_EQ(cover[bg_root - 1u] & 1u, 1u);
      ASSERT_EQ(cache_words[bg_root - 1u] & (root_mask | enclosed), cache_words[q] & (root_mask | enclosed));
      if (kind == scenario::different_roots || kind == scenario::two_pixel_different_roots || kind == scenario::three_pixel_t4_different_roots) {
        ASSERT_NE(p_root, r_root);
      } else {
        ASSERT_EQ(p_root, r_root);
      }
      if (kind == scenario::same_root || kind == scenario::two_pixel_gap || kind == scenario::three_pixel_t4_gap) {
        // C2/C3 have the 1px hole; C4 still has this 2px target q=0. The bounded
        // 512-only acceptance excludes OLD_AA and all BAND/physical-Raw authority.
        EXPECT_EQ(packed[q] & 0x3fffu, 512u);
        EXPECT_EQ(final_m[q], 1u);
      } else {
        EXPECT_EQ(packed[q] & 0x3fffu, 0u);
        EXPECT_EQ(final_m[q], 0u);
      }
      if (kind == scenario::two_pixel_gap || kind == scenario::two_pixel_different_roots) {
        const UINT second = 60u * W_ + 76u;
        ASSERT_EQ(origin_pixels_[second], (pixel8 {64u, 64u, 64u, 255u}));
        ASSERT_EQ(selected[second], 0u);ASSERT_EQ(passable[second], 0u);
        ASSERT_EQ(packed[second] & 255u, 0u);
        ASSERT_EQ(code1[second], 1u);ASSERT_EQ(raw_m[second], code1[second]);
        ASSERT_EQ(cover[second] & 1u, 1u);ASSERT_EQ(inner[second], 0u);
        ASSERT_EQ(cache_words[second] & (root_mask | enclosed), cache_words[q] & (root_mask | enclosed));
        if (kind == scenario::two_pixel_gap) {
          EXPECT_EQ(packed[second] & 0x3fffu, 512u);EXPECT_EQ(final_m[second], 1u);
        } else {
          EXPECT_EQ(packed[second] & 0x3fffu, 0u);
          // Inherited Code1 may retain a separately certified final collar here;
          // the new gap must still contribute zero packed support at either q.
        }
      }
      if (t4) {
        for (const UINT x : {75u, 76u}) {
          const UINT pixel = 60u * W_ + x;
          ASSERT_EQ(origin_pixels_[pixel], (pixel8 {64u, 64u, 64u, 255u}));
          ASSERT_EQ(selected[pixel], 0u);ASSERT_EQ(passable[pixel], 0u);ASSERT_EQ(packed[pixel] & 255u, 0u);
          ASSERT_EQ(code1[pixel], x == 76u ? 1u : 0u);ASSERT_EQ(raw_m[pixel], code1[pixel]);
          ASSERT_EQ(cover[pixel] & 1u, 1u);ASSERT_EQ(inner[pixel], 0u);
          ASSERT_EQ(cache_words[pixel] & (root_mask | enclosed), cache_words[q] & (root_mask | enclosed));
          if (kind == scenario::three_pixel_t4_gap) {
            EXPECT_EQ(packed[pixel] & 0x3fffu, 512u);EXPECT_EQ(final_m[pixel], 1u);
          } else {
            EXPECT_EQ(packed[pixel] & 0x3fffu, 0u);
            if (x == 75u) {
              EXPECT_EQ(final_m[pixel], 0u);
            }
          }
        }
      }
      if (kind == scenario::multiple_pixel_gap) {
        const UINT intervening = 60u * W_ + 76u, old_aa = 59u * W_ + 77u;
        for (const UINT pixel : {intervening, old_aa}) {
          ASSERT_EQ(origin_pixels_[pixel], (pixel8 {64u, 64u, 64u, 255u}));
          ASSERT_EQ(selected[pixel], 0u);ASSERT_EQ(passable[pixel], 0u);
          ASSERT_EQ(packed[pixel] & 255u, 0u);
          // RawM inherits Code1's BOX3 collar here. Actual nonblack RawPassable0
          // and BAND0, rather than that global bitmap, exclude donor authority.
          ASSERT_EQ(code1[pixel], 1u);ASSERT_EQ(raw_m[pixel], code1[pixel]);
        }
        EXPECT_NE(packed[old_aa] & 512u, 0u);EXPECT_NE(packed[old_aa] & 256u, 0u);
        EXPECT_EQ(final_m[old_aa], 1u);  // A real admitted AA point cannot start a recursive fill.
        for (const UINT blocker : {58u * W_ + 77u, 62u * W_ + 77u}) {
          ASSERT_EQ(origin_pixels_[blocker], (pixel8 {64u, 64u, 64u, 255u}));
          ASSERT_EQ(passable[blocker], 0u);ASSERT_EQ(packed[blocker] & 255u, 0u);
        }
      }
      record_uniform_black_state("pointwise_raw_gap_" + std::to_string(static_cast<int>(kind)));
    }
  }


  // Known C5 precision failure, retained with independent alpha truth. Run with
  // --gtest_also_run_disabled_tests; default exclusion is not a passing gate.
  // See docs/host-sbs-review-followups.md#shared-width-admission-experiment.
  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, DISABLED_AuthoredOpenBackgroundNotchIsNotGlyphCoverage) {
    for (const UINT thickness : {3u, 4u}) {
      SCOPED_TRACE(thickness);
      expect_authored_open_notch_alpha(0u, thickness);
    }
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, AuthoredFaintBlackOutlineRetainsCoverage) {
    for (const UINT thickness : {3u, 4u}) {
      SCOPED_TRACE(thickness);
      expect_authored_open_notch_alpha(8u, thickness);
    }
  }

  // Known C5 and experimental-B failures; opt in with the standard GTest flag.
  // Fixed development challenge, not an untouched real-font/video holdout.
  // Truth and all24 attempts were fixed before inspecting candidate B results.
  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, DISABLED_AuthoredScenePatternsChallengePixelAdmission) {
    RecordProperty("predeclared_matrix_sha256", "ddc51ed195d87c44a9e3c69ef9dfb64818f22d6fec0bd559f3ee1e731310d0aa");
    RecordProperty("scope", "development_challenge_not_untouched_real_font_holdout");
    UINT frame = 2000u;
    for (const auto pattern : {authored_scene_pattern::flat128, authored_scene_pattern::affine_gray,
      authored_scene_pattern::affine_rgb, authored_scene_pattern::checker}) {
      for (const UINT thickness : {3u, 4u}) for (const UINT alpha : {0u, 8u, 128u}) {
        const auto trace = std::string(authored_scene_pattern_name(pattern)) + "_t" +
          std::to_string(thickness) + "_a" + std::to_string(alpha);
        SCOPED_TRACE(trace);
        // A fatal helper prerequisite returns to this loop: no failed attempt is
        // filtered, and all later declared cases still run and export evidence.
        expect_authored_open_notch_alpha(alpha, thickness, pattern, frame++, "authored_scene_challenge_v1");
      }
    }
    EXPECT_EQ(frame, 2024u);
    RecordProperty("predeclared_attempts", 24);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, DISABLED_AuthoredOpaqueTintInsideMeasuredStrokeRetainsCoverage) {
    UINT frame=3000u;
    for(const auto scene:{authored_scene_pattern::flat128,authored_scene_pattern::affine_rgb,authored_scene_pattern::checker})
      for(const UINT t:{5u,6u})for(const pixel8 tint:{pixel8{6u,14u,27u,255u},pixel8{12u,24u,42u,255u}}) {
        SCOPED_TRACE(frame);expect_authored_stroke_hole(authored_stroke_hole_role::opaque_tint,t,tint,scene,frame++);
      }
    EXPECT_EQ(frame,3012u);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, AuthoredWhiteOCounterBeyondMeasuredStrokeRemainsScene) {
    UINT frame=3012u;
    for(const auto scene:{authored_scene_pattern::flat128,authored_scene_pattern::affine_rgb,authored_scene_pattern::checker})
      for(const UINT t:{5u,6u})for(const pixel8 tint:{pixel8{6u,14u,27u,255u},pixel8{12u,24u,42u,255u}}) {
        SCOPED_TRACE(frame);expect_authored_stroke_hole(authored_stroke_hole_role::O_counter,t,tint,scene,frame++);
      }
    EXPECT_EQ(frame,3024u);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, DISABLED_AuthoredBlueSceneCutoutCannotBorrowStrokeCoverage) {
    UINT frame=4000u;
    for(const auto scene:{authored_scene_pattern::flat128,authored_scene_pattern::affine_rgb,authored_scene_pattern::checker})
      for(const UINT t:{5u,6u}) {
        SCOPED_TRACE(frame);expect_authored_stroke_hole(authored_stroke_hole_role::blue_scene_cutout,t,{0u,0u,80u,255u},scene,frame++);
      }
    EXPECT_EQ(frame,4006u);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, DISABLED_AuthoredLowContrastBlackAARetainsCoverage) {
    UINT frame=5000u;for(bool affine:{false,true})for(UINT t:{5u,6u}) {
      SCOPED_TRACE(frame);expect_authored_low_contrast_aa(affine,32u,t,frame++);
    }
    EXPECT_EQ(frame,5004u);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, AuthoredLowContrastTransparentSceneRejectsCoverage) {
    UINT frame=5004u;for(bool affine:{false,true})for(UINT t:{5u,6u}) {
      SCOPED_TRACE(frame);expect_authored_low_contrast_aa(affine,0u,t,frame++);
    }
    EXPECT_EQ(frame,5008u);
  }

  // Enabled G1 controls reuse the independently authored flat scene/alpha
  // geometry. They are configured-model regressions, not a new holdout.
  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, ConfiguredStrokeCoverageOffersOriginalMissingAa) {
    ASSERT_FALSE(coverage_closure_.empty());ASSERT_FALSE(hole_closure_.empty());
    UINT frame=9000u;
    for(UINT thickness:{5u,6u}) {
      ASSERT_NO_FATAL_FAILURE(expect_authored_low_contrast_aa(false,64u,thickness,frame++,"g1_enabled_positive_"));
      ASSERT_TRUE(helper_.stroke_coverage_diagnostics_enabled());
      const auto proof=words(read(helper_.mask_hole_proof()));
      const auto offers=words(read(helper_.stroke_coverage_offers()));
      const auto parent=words(read(helper_.mask_hole_parents()));
      const auto flags=words(read(helper_.mask_hole_flags()));
      const UINT q=63u*W_+75u,p=62u*W_+75u;
      ASSERT_EQ(proof.size(),std::size_t(W_)*H_);ASSERT_EQ(offers.size(),proof.size());
      ASSERT_EQ(parent.size(),proof.size()+1u);ASSERT_EQ(flags.size(),parent.size());
      ASSERT_EQ(parent[0u],0u);ASSERT_EQ(flags[0u]&0x03000001u,0x03000001u);
      ASSERT_EQ(proof[q]&0x10000u,0u);ASSERT_NE(proof[q]&0x20000u,0u);
      ASSERT_EQ(proof[p]&0x10100u,0x10100u);EXPECT_EQ(offers[q],1u);
      RecordProperty("configured_G1_positive_scope","original alpha64; actual original-M0 and current C3 donor required");
    }
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, ConfiguredStrokeCoverageRejectsAuthoredTransparentScene) {
    ASSERT_FALSE(coverage_closure_.empty());ASSERT_FALSE(hole_closure_.empty());
    UINT frame=9002u;
    for(UINT thickness:{5u,6u}) {
      ASSERT_NO_FATAL_FAILURE(expect_authored_low_contrast_aa(false,0u,thickness,frame++,"g1_enabled_negative_"));
      ASSERT_TRUE(helper_.stroke_coverage_diagnostics_enabled());
      const auto proof=words(read(helper_.mask_hole_proof()));
      const auto offers=words(read(helper_.stroke_coverage_offers()));
      const UINT q=63u*W_+75u,p=62u*W_+75u;
      ASSERT_EQ(proof.size(),std::size_t(W_)*H_);ASSERT_EQ(offers.size(),proof.size());
      // The negative reaches the new model's original-source/colour/donor gates.
      ASSERT_EQ(proof[q]&0x10000u,0u);ASSERT_NE(proof[q]&0x20000u,0u);
      ASSERT_EQ(proof[p]&0x10100u,0x10100u);EXPECT_EQ(offers[q],0u);
      EXPECT_EQ(std::count_if(offers.begin(),offers.end(),[](UINT x){return x!=0u;}),0);
    }
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, StrokeCoverageDiagnosticsDoNotChangeProtectionOrCounters) {
    ASSERT_FALSE(coverage_closure_.empty());ASSERT_FALSE(hole_closure_.empty());
    ASSERT_NO_FATAL_FAILURE(expect_authored_low_contrast_aa(false,64u,5u,9004u,"g1_diagnostic_parity_"));
    ASSERT_TRUE(helper_.stroke_coverage_diagnostics_enabled());
    const auto on_m=outline_matte(),on_proof=read(helper_.mask_hole_proof());
    const auto on_data=calibration(),on_offers=words(read(helper_.stroke_coverage_offers()));
    const UINT q=63u*W_+75u;ASSERT_EQ(on_offers[q],1u);
    program_.coverage_diagnostics=false;
    ASSERT_TRUE(run())<<error_;ASSERT_FALSE(helper_.stroke_coverage_diagnostics_enabled());
    EXPECT_EQ(outline_matte(),on_m);EXPECT_EQ(calibration(),on_data);
    EXPECT_EQ(read(helper_.mask_hole_proof()),on_proof);expect_outline_accounting();
    program_.coverage_diagnostics=true;
    ASSERT_TRUE(run())<<error_;ASSERT_TRUE(helper_.stroke_coverage_diagnostics_enabled());
    EXPECT_EQ(outline_matte(),on_m);EXPECT_EQ(calibration(),on_data);
    EXPECT_EQ(words(read(helper_.stroke_coverage_offers())),on_offers);
    // Root0 is a hold, not a new clear or a freshness claim for old diagnostics.
    author(false,9006u,0u,false,false,false,256u,128u,3u);
    root_words_[60u]=0u;upload();ASSERT_TRUE(run())<<error_;
    EXPECT_EQ(outline_matte(),on_m);EXPECT_EQ(calibration(),on_data);
    EXPECT_EQ(read(helper_.mask_hole_proof()),on_proof);
    EXPECT_EQ(words(read(helper_.stroke_coverage_offers())),on_offers);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, StrokeCoverageStaleSourceProofCannotReusePriorOffer) {
    ASSERT_FALSE(coverage_closure_.empty());ASSERT_FALSE(hole_closure_.empty());
    ASSERT_NO_FATAL_FAILURE(expect_authored_low_contrast_aa(false,64u,5u,9005u,"g1_stale_source_"));
    const auto offers=words(read(helper_.stroke_coverage_offers()));ASSERT_EQ(offers[63u*W_+75u],1u);
    // Keep actual RGB and native resources, but invalidate the current SCT frame join.
    sct_words_[4u]^=1u;upload();ASSERT_TRUE(run())<<error_;
    EXPECT_EQ(receipt()[2u],0u);EXPECT_EQ(calibration()[2u],0u);
    EXPECT_EQ(outline_matte(),matte());
    // Diagnostic bytes/availability alone cannot attest current execution.
    RecordProperty("stale_G1_scope","invalid current SCT uses ordinary fault fallback; old offer is not authority");
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, StrokeCoverageMalformedProofFaultSuppressesPendingCommit) {
    ASSERT_FALSE(coverage_closure_.empty());ASSERT_FALSE(hole_closure_.empty());
    ASSERT_NO_FATAL_FAILURE(expect_authored_low_contrast_aa(false,64u,5u,9007u,"g1_private_fault_"));
    const UINT q=63u*W_+75u;
    const auto proof=words(read(helper_.mask_hole_proof()));
    const auto flags=words(read(helper_.mask_hole_flags()));
    ASSERT_EQ(proof[q]&0x10000u,0u);ASSERT_NE(flags[q+1u]&0x80000000u,0u);
    ASSERT_EQ(words(read(helper_.stroke_coverage_offers()))[q],1u);
    // Clear only the already-proved recipient in the output to make an
    // erroneous recommit observable. Keep its actual pending proof untouched.
    const auto v=helper_.mutable_views();ComPtr<ID3D11Resource> mask,proof_resource;
    v.matte->GetResource(&mask);helper_.mask_hole_proof()->GetResource(&proof_resource);
    const D3D11_BOX box{75u,63u,0u,76u,64u,1u};const UINT zero=0u,bad=proof[q]|0x40000000u;
    context_->UpdateSubresource(mask.Get(),0u,&box,&zero,4u,0u);
    context_->UpdateSubresource(proof_resource.Get(),0u,&box,&bad,4u,0u);
    const auto before_m=outline_matte();const auto before_data=calibration();
    dispatch_current_coverage_probe(false);
    ASSERT_EQ(words(read(helper_.mask_hole_parents()))[0u],0xffffffffu);
    dispatch_current_coverage_probe(true);
    EXPECT_EQ(outline_matte(),before_m);EXPECT_EQ(calibration(),before_data);
    EXPECT_EQ(words(outline_matte())[q],0u);
    RecordProperty("G1_private_fault_scope","reserved current proof faults before cached colour/prefilter; sticky fault blocks prior pending recommit");
  }

  // Fixed before the new model/GPU attempt. Initial DISABLED status preserves
  // known-red controls and all readiness failures; opt-in runs export all16.
  // Development cohort: only original alpha changes; old0/32 inputs remain intact.
  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, DISABLED_AuthoredLowContrastHigherCoverageDevelopmentMatrix) {
    RecordProperty("higher_coverage_matrix_sha256", "8cf94e7b6ed878d70e48334cca00066c977cff82725689f5966c3f5e29fe5209");
    RecordProperty("higher_coverage_scope", "development_reuses_validated_geometry_not_unseen_holdout");
    UINT frame=8100u;
    for(UINT coverage:{64u,96u})for(bool affine:{false,true})for(UINT thickness:{5u,6u}) {
      SCOPED_TRACE(frame);expect_authored_low_contrast_aa(affine,coverage,thickness,frame++);
    }
    EXPECT_EQ(frame,8108u);
  }

  // Predeclared broader development background matrix; every attempt exports before assertions.
  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, DISABLED_BroadBackgroundStrokeCoverage24) {
    if(!std::getenv("SUNSHINE_TEST_BROAD_BACKGROUND_OUTPUT_DIR"))GTEST_SKIP()<<"Explicit full24 export path required";
    UINT frame=8200u;
    for(UINT scene_kind:{0u,1u,2u,3u})for(UINT coverage:{0u,64u,96u})for(UINT thickness:{5u,6u}) {
      SCOPED_TRACE(frame);expect_broad_background_stroke_coverage(scene_kind,coverage,thickness,frame++);
    }
    EXPECT_EQ(frame,8224u);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, DISABLED_GenericHeldoutOuterAA) {
    if(!std::getenv("SUNSHINE_TEST_GENERIC_HELDOUT_ROLE"))GTEST_SKIP()<<"Explicit control/treatment role required";
    UINT frame=8000u;for(bool affine:{false,true})for(UINT scale:{1u,2u}){SCOPED_TRACE(frame);expect_generic_heldout(generic_heldout_role::outer_aa,affine,scale,frame++);}
  }
  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, DISABLED_GenericHeldoutInnerAA) {
    if(!std::getenv("SUNSHINE_TEST_GENERIC_HELDOUT_ROLE"))GTEST_SKIP()<<"Explicit control/treatment role required";
    UINT frame=8004u;for(bool affine:{false,true})for(UINT scale:{1u,2u}){SCOPED_TRACE(frame);expect_generic_heldout(generic_heldout_role::inner_aa,affine,scale,frame++);}
  }
  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, DISABLED_GenericHeldoutGenuineCounter) {
    if(!std::getenv("SUNSHINE_TEST_GENERIC_HELDOUT_ROLE"))GTEST_SKIP()<<"Explicit control/treatment role required";
    UINT frame=8008u;for(bool affine:{false,true})for(UINT scale:{1u,2u}){SCOPED_TRACE(frame);expect_generic_heldout(generic_heldout_role::counter,affine,scale,frame++);}
  }
  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, DISABLED_GenericHeldoutOpenAttachedScene) {
    if(!std::getenv("SUNSHINE_TEST_GENERIC_HELDOUT_ROLE"))GTEST_SKIP()<<"Explicit control/treatment role required";
    UINT frame=8012u;for(bool affine:{false,true})for(UINT scale:{1u,2u}){SCOPED_TRACE(frame);expect_generic_heldout(generic_heldout_role::attached_scene,affine,scale,frame++);}
  }
  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, DISABLED_GenericSourceLayerAliasIsUnresolvableDiagnostic) {
    if(!std::getenv("SUNSHINE_TEST_GENERIC_HELDOUT_ROLE"))GTEST_SKIP()<<"Explicit control/treatment role required";
    expect_generic_heldout(generic_heldout_role::outer_aa,false,1u,8098u,true,false);
    expect_generic_heldout(generic_heldout_role::outer_aa,false,1u,8099u,true,true);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, DISABLED_AuthoredC7CornerOpaqueInkRequiresLateCompletion) {
    if(!std::getenv("SUNSHINE_TEST_AUTHORED_C7_ROLE"))GTEST_SKIP()<<"Explicit paired control/treatment role required";
    UINT frame=6000u;for(UINT t:{5u,6u})for(UINT scene:{0u,1u})for(pixel8 tint:{pixel8{16u,24u,40u,255u},pixel8{24u,24u,24u,255u}}) {
      SCOPED_TRACE(frame);expect_authored_c7_boundary_case(true,scene,t,tint,frame++);
    }
    EXPECT_EQ(frame,6008u);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, DISABLED_AuthoredC7AsymmetricGenuineCounterRemainsScene) {
    if(!std::getenv("SUNSHINE_TEST_AUTHORED_C7_ROLE"))GTEST_SKIP()<<"Explicit paired control/treatment role required";
    UINT frame=6008u;for(UINT t:{5u,6u})for(UINT scene:{0u,1u}) {
      SCOPED_TRACE(frame);expect_authored_c7_boundary_case(false,scene,t,{0u,0u,0u,255u},frame++);
    }
    EXPECT_EQ(frame,6012u);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, DISABLED_AuthoredC7DarkAsymmetricGenuineCounterRemainsScene) {
    if(!std::getenv("SUNSHINE_TEST_AUTHORED_C7_ROLE"))GTEST_SKIP()<<"Explicit paired control/treatment role required";
    UINT frame=6012u;for(UINT t:{5u,6u})for(UINT scene:{2u,3u}) {
      SCOPED_TRACE(frame);expect_authored_c7_boundary_case(false,scene,t,{0u,0u,0u,255u},frame++);
    }
    EXPECT_EQ(frame,6016u);
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
    EXPECT_EQ(packed[q] & 511u, 256u);  // Existing one-pixel black-gap AA; q still has no measured BAND.
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
        // Gray2 is still UNKNOWN in the upstream palette. Only this completed
        // closed-body CBC case deliberately extends it to private K4.
        expect_uniform_known(meta, 0u, pixel8 {2u, 2u, 2u, 255u}, 4u);
      }
      const auto b=words(body()),m=words(outline_matte());
      ASSERT_EQ(b.size(),W_*H_);ASSERT_EQ(m.size(),b.size());
      std::size_t whiteBody=0u,whiteProtected=0u;
      // Authored opaque white interiors, independent of inferred model/mask.
      for(UINT y=48u;y<88u;++y)for(UINT x=0u;x<W_;++x)if((x>=79u&&x<111u)||(x>=160u&&x<208u)) {
        const auto i=y*W_+x;EXPECT_EQ(b[i],1u);EXPECT_EQ(m[i],1u);
        whiteBody+=b[i]!=0u;whiteProtected+=m[i]!=0u;
      }
      EXPECT_EQ(whiteBody,3200u);EXPECT_EQ(whiteProtected,3200u);
      RecordProperty(label+"_white_BODY_covered",whiteBody);RecordProperty(label+"_white_M_covered",whiteProtected);
      RecordProperty(label+"_Inside_good",data[79u]);RecordProperty(label+"_Inside_fault",data[80u]);
      if(code==0u) {zeroMask=m;}
      if(code==1u) {EXPECT_EQ(m,zeroMask) << "Literal0/1 same physical mask semantics; preserve any difference as failure";}
      if(code==2u) {
        // K4 is an opaque-only trust tier. Unlike K3 it does not inherit
        // AA/closure certificates for scene colours in this authored raster.
        for(std::size_t i=0u;i<m.size();++i) {
          const auto &p=origin_pixels_[i];
          const bool opaque=p==(pixel8{255u,255u,255u,255u})||p==(pixel8{2u,2u,2u,255u});
          EXPECT_EQ(m[i],opaque?1u:0u);
        }
      }
    }
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, UniformNonliteralClosedPeersRecoverFinalOutlineWithoutChangingBody) {
    author_band(5u, 220u);
    ASSERT_TRUE(run()) << error_;
    expect_ready();expect_ring_ready();expect_outline_accounting();
    const auto literal_body = body(), literal_code1 = matte(), literal_raw = ring_matte(), literal_final = outline_matte();
    const auto literal_meta = receipt(), literal_f2 = calibration();
    ASSERT_EQ(literal_meta.size(), 512u);ASSERT_EQ(literal_f2.size(), 68608u);
    expect_uniform_known(literal_meta, 0u, pixel8 {0u, 0u, 0u, 255u}, 3u);
    ASSERT_EQ(literal_f2[139u], 5u);
    for (const auto colour : {pixel8 {7u, 8u, 11u, 255u}, pixel8 {2u, 2u, 2u, 255u}}) {
      SCOPED_TRACE(nlohmann::json(colour).dump());
      author_band(5u, 221u + colour[0]);
      recolour_literal_black(colour);
      ASSERT_TRUE(run()) << error_;
      expect_ready();expect_ring_ready();expect_outline_accounting();
      const auto meta = receipt(), raw_meta = ring_receipt(), f2 = calibration();
      ASSERT_EQ(meta.size(), 512u);ASSERT_EQ(raw_meta.size(), 512u);ASSERT_EQ(f2.size(), 68608u);
      expect_uniform_known(meta, 0u, colour, 4u);
      EXPECT_EQ(meta[130u], 0u);  // Every phase-only qualification flag is cleared before publication.
      EXPECT_GT(meta[132u], 0u);
      EXPECT_EQ(meta[131u], meta[132u]);
      EXPECT_EQ(meta[158u], meta[132u]);
      EXPECT_EQ(meta[157u], 1u);
      EXPECT_EQ(raw_meta[128u], 1u);
      EXPECT_EQ(f2[139u], 5u);
      EXPECT_EQ(f2[141u], 1u);
      EXPECT_EQ(f2[80u], 0u);
      EXPECT_EQ(body(), literal_body);
      EXPECT_EQ(matte(), literal_code1);
      EXPECT_EQ(ring_matte(), literal_raw);
      RecordProperty("closed_" + std::to_string(colour[0]) + "_literal_K3_final_M_sha256", hash_bytes(literal_final));
      const auto b = helper_.baseline_views();
      EXPECT_EQ(body(), read(b.body.Get()));
      const auto final_m = words(outline_matte()), code1 = words(matte());
      ASSERT_EQ(final_m.size(), W_ * H_);ASSERT_EQ(code1.size(), final_m.size());
      std::size_t white = 0u, recovered_black = 0u;
      for (std::size_t i = 0u; i < final_m.size(); ++i) {
        const bool opaque = origin_pixels_[i] == (pixel8 {255u, 255u, 255u, 255u}) || origin_pixels_[i] == colour;
        EXPECT_EQ(final_m[i], opaque ? 1u : 0u);
        if (origin_pixels_[i] == (pixel8 {255u, 255u, 255u, 255u})) {
          EXPECT_EQ(final_m[i], 1u);++white;
        }
        if (origin_pixels_[i] == colour && final_m[i] != 0u && code1[i] == 0u) ++recovered_black;
      }
      EXPECT_EQ(white, 3200u);
      EXPECT_GT(recovered_black, 0u);
      record_uniform_black_state("closed_" + std::to_string(colour[0]));
      RecordProperty("closed_" + std::to_string(colour[0]) + "_new_opaque_black_pixels", recovered_black);
    }
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, UniformNonliteralOpenOrBoundaryPeersKeepExactUnknownFallback) {
    enum class shape {one_open, both_open, dark_background, cover_candidate, cover_literal, native_candidate, connected_boundary};
    const std::array cases {shape::one_open, shape::both_open, shape::dark_background,
      shape::cover_candidate, shape::cover_literal, shape::native_candidate, shape::connected_boundary};
    for (UINT index = 0u; index < cases.size(); ++index) {
      SCOPED_TRACE(index);
      author_band(5u, 240u + index, cases[index] == shape::dark_background);
      const pixel8 black {7u, 8u, 11u, 255u};
      recolour_literal_black(black);
      if (cases[index] == shape::one_open || cases[index] == shape::both_open) {
        // Scene64 is outside WHITE's half-peak eligibility, so the original
        // colour BODY remains closed. It is nonblack in the provisional physical
        // graph and opens the rim by one row. A closed peer cannot certify it.
        rect(74u, 65u, 79u, 66u, {64u, 64u, 64u, 255u});
        if (cases[index] == shape::both_open) rect(155u, 65u, 160u, 66u, {64u, 64u, 64u, 255u});
      } else if (cases[index] == shape::cover_candidate) {
        rect(32u, 60u, 33u, 61u, black);
      } else if (cases[index] == shape::cover_literal) {
        rect(32u, 60u, 33u, 61u, {0u, 0u, 0u, 255u});
      } else if (cases[index] == shape::native_candidate) {
        const std::array<UINT, 4u> cover {0u, 0u, 16u, 16u};
        std::copy(cover.begin(), cover.end(), ocr_words_.begin() + 144u);
        rect(0u, 60u, 1u, 61u, black);
      } else if (cases[index] == shape::connected_boundary) {
        rect(32u, 65u, 75u, 66u, black);
      }
      upload_pixels_from_origin();
      ASSERT_TRUE(run()) << error_;
      expect_ready();expect_ring_ready();expect_outline_accounting();
      const auto meta = receipt(), f2 = calibration();
      ASSERT_EQ(meta.size(), 512u);ASSERT_EQ(f2.size(), 68608u);
      EXPECT_EQ(meta[385u], 1u);EXPECT_EQ(meta[386u], 0u);EXPECT_EQ(meta[387u], 5u);
      ASSERT_GE(meta[388u], 2u);ASSERT_GE(meta[391u], 2u);
      EXPECT_EQ(meta[390u], meta[388u]);EXPECT_EQ(meta[392u], meta[391u]);
      EXPECT_EQ(meta[130u], 0u);
      EXPECT_EQ(meta[131u], 0u);EXPECT_EQ(meta[157u], 0u);EXPECT_EQ(meta[158u], 0u);
      expect_baseline();expect_ring_parent();
      // A valid UNKNOWN publication retains original BODY in final CFB7.
      // Raw separately retains the unmodified Code1 BOX3 parent above.
      EXPECT_EQ(outline_matte(), body());
      const auto original = words(read(helper_.baseline_views().body.Get()));
      const auto selected = words(body()), final_m = words(outline_matte());
      ASSERT_EQ(original.size(), W_ * H_);ASSERT_EQ(selected.size(), original.size());ASSERT_EQ(final_m.size(), original.size());
      std::size_t original_white = 0u;
      for (std::size_t i = 0u; i < original.size(); ++i) {
        if (original[i] != 0u) {
          EXPECT_EQ(selected[i], 1u);EXPECT_EQ(final_m[i], 1u);++original_white;
        }
      }
      EXPECT_EQ(original_white, 3200u);
      EXPECT_EQ(f2[80u], 0u);
      record_uniform_black_state("unknown_shape_" + std::to_string(index));
    }
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, UniformNonliteralOpaqueOnlyRingPreservesHoleAndExteriorScene) {
    const char *stability_root = std::getenv("SUNSHINE_TEST_BLACK_STABILITY_SHADER_ROOT");
    const char *stability_pin = std::getenv("SUNSHINE_TEST_BLACK_STABILITY_SHADER_PIN");
    ASSERT_NE(stability_root, nullptr);ASSERT_NE(stability_pin, nullptr);
    std::array<ComPtr<ID3D11ComputeShader>, 3u> stability_shaders;
    const std::array stability_specs {
      cache::shader_spec {"host_sbs_current_outline_support_band_cs.hlsl", "private_cache_begin_main", "cs_5_0"},
      cache::shader_spec {"host_sbs_current_outline_support_band_cs.hlsl", "private_cache_validate_main", "cs_5_0"},
      cache::shader_spec {"host_sbs_current_outline_support_band_cs.hlsl", "private_cache_commit_main", "cs_5_0"}};
    load_shaders(stability_root, stability_pin, stability_specs, stability_shaders);
    for (UINT i = 0u; i < 3u; ++i) program_.stability_shaders[i] = stability_shaders[i].Get();
    // Exercise cache-role rejection with a real current width that otherwise
    // meets the ordinary two-band/issued-Raw prerequisites. The O below has
    // only one profile band and cannot isolate that role guard by itself.
    author_band(5u, 254u);recolour_literal_black({7u, 8u, 11u, 255u});
    ASSERT_TRUE(run()) << error_;
    const auto strong_meta = receipt(), strong_raw = ring_receipt(), strong_f2 = calibration();
    ASSERT_EQ(strong_meta.size(), 512u);ASSERT_EQ(strong_raw.size(), 512u);ASSERT_EQ(strong_f2.size(), 68608u);
    expect_uniform_known(strong_meta, 0u, {7u, 8u, 11u, 255u}, 4u);
    ASSERT_EQ(strong_raw[128u], 1u);ASSERT_EQ(strong_raw[129u], 1u);
    ASSERT_EQ(strong_f2[139u], 5u);ASSERT_EQ(strong_f2[141u], 1u);
    ASSERT_GE(strong_f2[135u], 2u);ASSERT_GT(strong_f2[136u], 0u);ASSERT_EQ(strong_f2[142u], 1u);
    ASSERT_EQ(strong_f2[164u], 0u);
    const auto strong_history = aa_history_snapshot("opaque_peers_no_snapshot");
    const auto strong_header = words(strong_history.header), strong_flags = words(strong_history.history);
    ASSERT_EQ(strong_header.size(), 160u);ASSERT_EQ(strong_flags.size(), W_ * H_);
    ASSERT_EQ(strong_header[0u], 0x31435342u);
    EXPECT_EQ(strong_header[102u], 0u);EXPECT_EQ(strong_header[146u], 0u);EXPECT_EQ(strong_header[147u], 0u);
    EXPECT_EQ(std::count_if(strong_flags.begin(), strong_flags.end(), [](UINT word) { return word != 0u; }), 0);
    record_uniform_black_state("opaque_peers_cache");
    author_band(2u, 255u);
    const pixel8 black {7u, 8u, 11u, 255u}, white {255u, 255u, 255u, 255u};
    // The external near-half blend is WHITE-eligible but reaches the true domain
    // edge. The darker nonblack counter is below WHITE's half-peak eligibility.
    rect(0u, 0u, W_, H_, {131u, 132u, 133u, 255u});
    rect(74u, 43u, 116u, 93u, black);
    rect(76u, 45u, 114u, 91u, white);
    rect(84u, 55u, 106u, 81u, black);
    rect(86u, 57u, 104u, 79u, {31u, 23u, 28u, 255u});
    upload_pixels_from_origin();ASSERT_TRUE(run()) << error_;
    expect_ready();expect_ring_ready();expect_outline_accounting();
    const auto meta = receipt(), f2 = calibration();
    ASSERT_EQ(meta.size(), 512u);ASSERT_EQ(f2.size(), 68608u);
    expect_uniform_known(meta, 0u, black, 4u);
    ASSERT_EQ(meta[131u], meta[132u]);ASSERT_EQ(meta[158u], meta[132u]);ASSERT_GT(meta[132u], 0u);
    const auto original = words(read(helper_.baseline_views().body.Get()));
    const auto selected = words(body()), code1 = words(matte()), raw_m = words(ring_matte()), final_m = words(outline_matte());
    const auto packed = words(read(packed_snapshot_.resource.Get()));
    ASSERT_EQ(original.size(), W_ * H_);ASSERT_EQ(selected.size(), original.size());
    ASSERT_EQ(code1.size(), original.size());ASSERT_EQ(raw_m.size(), original.size());
    ASSERT_EQ(final_m.size(), original.size());ASSERT_EQ(packed.size(), original.size());
    constexpr UINT opaque_offer = 1u << 25u;
    std::size_t white_count = 0u, black_count = 0u, opaque_offers = 0u;
    for (std::size_t i = 0u; i < final_m.size(); ++i) {
      const bool is_white = origin_pixels_[i] == white, is_black = origin_pixels_[i] == black;
      EXPECT_EQ(original[i], is_white ? 1u : 0u);
      EXPECT_EQ(selected[i], original[i]);
      EXPECT_EQ(final_m[i], is_white || is_black ? 1u : 0u);
      EXPECT_EQ(packed[i] & 0x3fffu, 0u);  // No K4 BAND, AA, closure, collar or measured rim.
      EXPECT_EQ(packed[i] & opaque_offer, is_black ? opaque_offer : 0u);
      if (is_black) EXPECT_EQ(raw_m[i], 1u);
      white_count += is_white;black_count += is_black;opaque_offers += (packed[i] & opaque_offer) != 0u;
    }
    EXPECT_EQ(white_count, 1176u);EXPECT_EQ(black_count, 528u);EXPECT_EQ(opaque_offers, black_count);
    for (const UINT index : {67u, 68u, 69u, 75u, 79u, 80u, 81u, 82u, 84u}) EXPECT_EQ(f2[index], 0u);
    for (const auto point : {std::array<UINT, 2u> {73u, 65u}, std::array<UINT, 2u> {86u, 65u}}) {
      const UINT i = point[1] * W_ + point[0];
      ASSERT_EQ(code1[i], 1u);ASSERT_EQ(raw_m[i], 1u);
      EXPECT_EQ(original[i], 0u);EXPECT_EQ(final_m[i], 0u);EXPECT_EQ(packed[i] & opaque_offer, 0u);
    }
    const auto history = aa_history_snapshot("opaque_O_history");
    const auto history_flags = words(history.history), history_header = words(history.header);
    ASSERT_EQ(history_flags.size(), final_m.size());ASSERT_EQ(history_header.size(), 160u);
    ASSERT_EQ(history_header[0u], 0x31435342u);
    EXPECT_EQ(history_header[102u], 0u);EXPECT_EQ(history_header[146u], 0u);
    EXPECT_EQ(std::count_if(history_flags.begin(), history_flags.end(), [](UINT word) { return word != 0u; }), 0);
    // The real optional cache program ran but declined a strong snapshot. This
    // checks no K4 snapshot plan/history birth, rather than a construction write.
    record_uniform_black_state("opaque_O");
    RecordProperty("opaque_O_source_rgba8_sha256", hash_bytes(raw(origin_pixels_)));
    RecordProperty("opaque_O_final_M_sha256", hash_bytes(raw(final_m)));
    RecordProperty("opaque_O_black_Raw_members", black_count);
    RecordProperty("opaque_O_private_offers", opaque_offers);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, UniformNonliteralOpaqueOwnerPreservesOtherKnownOwnersAaAndClosure) {
    const std::array<UINT, 4u> first_core {4u, 5u, 7u, 11u}, first_cover {2u, 3u, 15u, 14u};
    const std::array<UINT, 4u> second_core {10u, 5u, 13u, 11u}, second_cover {8u, 3u, 15u, 14u};
    std::vector<UINT> reference_final;
    std::string reference_source;
    for (const bool overlap : {false, true}) {
      SCOPED_TRACE(overlap);
      author_band(5u, overlap ? 258u : 257u);
      rect(74u, 43u, 116u, 93u, {2u, 2u, 2u, 255u});
      rect(79u, 48u, 111u, 88u, {255u, 255u, 255u, 255u});
      // Relocated existing NearBlack concavity: literal0 donors x155/x157
      // bracket this nonblack pixel four source pixels from WHITE x160.
      rect(156u, 60u, 157u, 61u, {2u, 2u, 2u, 255u});
      if (overlap) {
        ocr_words_[3u] = ocr_words_[4u] = 2u;uip_words_[3u] = 3u;
        std::copy(first_core.begin(), first_core.end(), ocr_words_.begin() + 16u);
        std::copy(first_cover.begin(), first_cover.end(), ocr_words_.begin() + 144u);
        std::copy(second_core.begin(), second_core.end(), ocr_words_.begin() + 24u);
        std::copy(second_cover.begin(), second_cover.end(), ocr_words_.begin() + 152u);
      } else {
        std::copy(second_core.begin(), second_core.end(), ocr_words_.begin() + 16u);
        std::copy(second_cover.begin(), second_cover.end(), ocr_words_.begin() + 144u);
      }
      upload_pixels_from_origin();ASSERT_TRUE(run()) << error_;
      expect_ready();expect_ring_ready();expect_outline_accounting();
      const auto meta = receipt();ASSERT_EQ(meta.size(), 512u);
      const auto original = words(read(helper_.baseline_views().body.Get()));
      const auto raw_m = words(ring_matte()), final_m = words(outline_matte());
      const auto packed = words(read(packed_snapshot_.resource.Get()));
      ASSERT_EQ(original.size(), W_ * H_);ASSERT_EQ(raw_m.size(), original.size());
      ASSERT_EQ(final_m.size(), original.size());ASSERT_EQ(packed.size(), original.size());
      const UINT q = 60u * W_ + 156u;
      ASSERT_EQ(original[q], 0u);ASSERT_EQ(raw_m[q], overlap ? 1u : 0u);
      ASSERT_EQ(packed[q] & 511u, 256u);ASSERT_NE(packed[q] & 1024u, 0u);
      EXPECT_EQ(final_m[q], 1u);
      if (!overlap) {
        expect_uniform_known(meta, 0u, {0u, 0u, 0u, 255u}, 3u);
        reference_final = final_m;reference_source = hash_bytes(raw(origin_pixels_));
      } else {
        expect_uniform_known(meta, 0u, {2u, 2u, 2u, 255u}, 4u);
        expect_uniform_known(meta, 1u, {0u, 0u, 0u, 255u}, 3u);
        ASSERT_EQ(hash_bytes(raw(origin_pixels_)), reference_source);
        ASSERT_EQ(reference_final.size(), final_m.size());
        for (std::size_t i = 0u; i < final_m.size(); ++i) EXPECT_LE(reference_final[i], final_m[i]);
        // q matches K4's exact prototype but remains nonblack for the K3
        // owner. K4's private opaque offer must not suppress that owner's
        // independent AA/closure through a global physical-black predicate.
        EXPECT_NE(packed[q] & (1u << 25u), 0u);
        record_uniform_black_state("opaque_overlap_K4", 0u);
        record_uniform_black_state("opaque_overlap_K3", 1u);
      }
    }
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, UniformNonliteralClosedUnitDoesNotCertifyOtherOpenUnit) {
    author_band(5u, 260u);
    recolour_literal_black({7u, 8u, 11u, 255u});
    rect(155u, 65u, 160u, 66u, {64u, 64u, 64u, 255u});
    ocr_words_[3u] = ocr_words_[4u] = 2u;uip_words_[3u] = 3u;
    const std::array<UINT, 4u> core0 {4u, 5u, 7u, 11u}, cover0 {2u, 3u, 8u, 14u};
    const std::array<UINT, 4u> core1 {9u, 5u, 14u, 11u}, cover1 {8u, 3u, 15u, 14u};
    std::copy(core0.begin(), core0.end(), ocr_words_.begin() + 16u);
    std::copy(cover0.begin(), cover0.end(), ocr_words_.begin() + 144u);
    std::copy(core1.begin(), core1.end(), ocr_words_.begin() + 24u);
    std::copy(cover1.begin(), cover1.end(), ocr_words_.begin() + 152u);
    upload_pixels_from_origin();
    ASSERT_TRUE(run()) << error_;
    expect_ready();expect_ring_ready();expect_outline_accounting();
    const auto meta = receipt(), raw_meta = ring_receipt(), f2 = calibration();
    ASSERT_EQ(meta.size(), 512u);ASSERT_EQ(raw_meta.size(), 512u);ASSERT_EQ(f2.size(), 68608u);
    expect_uniform_known(meta, 0u, {7u, 8u, 11u, 255u}, 4u);
    EXPECT_EQ(meta[401u], 1u);EXPECT_EQ(meta[402u], 0u);EXPECT_EQ(meta[403u], 5u);
    EXPECT_EQ(meta[130u], 0u);EXPECT_EQ(meta[162u], 0u);
    EXPECT_EQ(meta[157u], 1u);EXPECT_EQ(meta[189u], 0u);
    EXPECT_EQ(raw_meta[128u], 1u);EXPECT_EQ(raw_meta[160u], 0u);
    EXPECT_EQ(f2[80u], 0u);
    const auto original = words(read(helper_.baseline_views().body.Get()));
    const auto selected = words(body()), code1 = words(matte()), final_m = words(outline_matte());
    ASSERT_EQ(original.size(), W_ * H_);ASSERT_EQ(selected.size(), original.size());
    ASSERT_EQ(code1.size(), original.size());ASSERT_EQ(final_m.size(), original.size());
    std::size_t closed_added = 0u;
    for (UINT y = 0u; y < H_; ++y) for (UINT x = 0u; x < W_; ++x) {
      const UINT i = y * W_ + x;
      if (original[i]) {EXPECT_EQ(selected[i], 1u);EXPECT_EQ(final_m[i], 1u);}
      if (x < 128u) closed_added += final_m[i] != 0u && code1[i] == 0u;
      else EXPECT_EQ(final_m[i], original[i]);
    }
    EXPECT_GT(closed_added, 0u);
    record_uniform_black_state("closed_unit", 0u);record_uniform_black_state("open_unit", 1u);
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, UniformNonliteralOriginalBodyBlackOverlapRejectsOtherwiseClosedTwin) {
    const pixel8 black {60u, 70u, 90u, 255u};
    std::vector<char> closed_body, closed_code1;
    std::vector<UINT> closed_meta, closed_normal;
    for (const bool overlap : {false, true}) {
      SCOPED_TRACE(overlap);
      author_body_black_overlap(overlap ? 266u : 265u, overlap);
      ASSERT_TRUE(run()) << error_;
      expect_ready();expect_ring_ready();expect_outline_accounting();
      const auto meta = receipt(), f2 = calibration();
      const auto baseline_body = read(helper_.baseline_views().body.Get());
      const auto original = words(baseline_body), selected = words(body()), final_m = words(outline_matte());
      auto normal = words(read(helper_.enclosure_views().all_body_bits.Get()));
      ASSERT_EQ(meta.size(), 512u);ASSERT_EQ(f2.size(), 68608u);
      ASSERT_EQ(original.size(), W_ * H_);ASSERT_EQ(selected.size(), original.size());
      ASSERT_EQ(final_m.size(), original.size());ASSERT_EQ(normal.size(), original.size());
      for (auto &word : normal) word &= 1u;
      ASSERT_EQ(meta[389u], 0x005a463cu);
      ASSERT_GE(meta[388u], 2u);ASSERT_GE(meta[391u], 2u);
      ASSERT_EQ(meta[390u], meta[388u]);ASSERT_EQ(meta[392u], meta[391u]);
      ASSERT_GT(meta[132u], 0u);
      EXPECT_EQ(meta[130u], 0u);EXPECT_EQ(f2[80u], 0u);
      for (UINT y = 65u; y < 72u; ++y) for (UINT x = 94u; x < 98u; ++x) {
        const auto i = y * W_ + x;
        ASSERT_EQ(original[i], 1u);ASSERT_EQ(normal[i], 1u);
        EXPECT_EQ(selected[i], 1u);EXPECT_EQ(final_m[i], 1u);
      }
      if (!overlap) {
        expect_uniform_known(meta, 0u, black, 4u);
        ASSERT_EQ(meta[131u], meta[132u]);ASSERT_EQ(meta[158u], meta[132u]);
        ASSERT_EQ(meta[157u], 1u);
        closed_body = baseline_body;closed_code1 = matte();closed_meta = meta;closed_normal = normal;
      } else {
        // The positive twin measured every original BODY pixel closed. The only
        // changed source pixels are already BODY, so (!black || BODY) and its
        // true-boundary seeds are identical in both physical graphs. Actual
        // BODY/normal-bit equality and unchanged K sample counts establish the
        // other qualifications; BODY intersecting candidate black is the veto.
        EXPECT_EQ(baseline_body, closed_body);EXPECT_EQ(normal, closed_normal);
        EXPECT_EQ(matte(), closed_code1);
        EXPECT_EQ(meta[132u], closed_meta[132u]);
        EXPECT_EQ(std::vector<UINT>(meta.begin() + 388u, meta.begin() + 393u),
          std::vector<UINT>(closed_meta.begin() + 388u, closed_meta.begin() + 393u));
        EXPECT_EQ(meta[385u], 1u);EXPECT_EQ(meta[386u], 0u);EXPECT_EQ(meta[387u], 5u);
        EXPECT_EQ(meta[131u], 0u);EXPECT_EQ(meta[157u], 0u);EXPECT_EQ(meta[158u], 0u);
        expect_inherited_unknown_final("body_black_overlap");
      }
      const std::string label = overlap ? "body_black_overlap" : "body_black_closed_twin";
      record_uniform_black_state(label);
      RecordProperty(label + "_source_rgba8_sha256", hash_bytes(raw(origin_pixels_)));
      RecordProperty(label + "_P_sha256", hash_bytes(raw(probability_words_)));
      RecordProperty(label + "_S_sha256", hash_bytes(raw(support_words_)));
      RecordProperty(label + "_original_BODY_sha256", hash_bytes(baseline_body));
      RecordProperty(label + "_final_M_sha256", hash_bytes(outline_matte()));
    }
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, UniformNonliteralBodyBlackOverlapRetainsInheritedUnknownFinalMask) {
    // Standalone negative has no K4-positive prerequisite. The exact same test
    // can run against frozen pre-K4 shaders to compare complete native M hashes.
    author_body_black_overlap(266u, true);
    ASSERT_TRUE(run()) << error_;
    expect_ready();expect_ring_ready();expect_outline_accounting();
    const auto meta = receipt();ASSERT_EQ(meta.size(), 512u);
    ASSERT_EQ(meta[385u], 1u);ASSERT_EQ(meta[386u], 0u);ASSERT_EQ(meta[387u], 5u);
    ASSERT_EQ(meta[389u], 0x005a463cu);ASSERT_GE(meta[388u], 2u);ASSERT_GE(meta[391u], 2u);
    ASSERT_EQ(meta[390u], meta[388u]);ASSERT_EQ(meta[392u], meta[391u]);
    EXPECT_EQ(meta[130u], 0u);EXPECT_EQ(meta[131u], 0u);EXPECT_EQ(meta[157u], 0u);EXPECT_EQ(meta[158u], 0u);
    const auto original = words(read(helper_.baseline_views().body.Get()));
    const auto normal = words(read(helper_.enclosure_views().all_body_bits.Get()));
    ASSERT_EQ(original.size(), W_ * H_);ASSERT_EQ(normal.size(), original.size());
    for (UINT y = 65u; y < 72u; ++y) for (UINT x = 94u; x < 98u; ++x) {
      ASSERT_EQ(original[y * W_ + x], 1u);ASSERT_EQ(normal[y * W_ + x] & 1u, 1u);
    }
    expect_inherited_unknown_final("standalone_body_black_overlap");
    record_uniform_black_state("standalone_body_black_overlap");
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, UniformNonliteralPublisherRejectsMalformedCandidateAndPhaseFlags) {
    author_band(5u, 270u);recolour_literal_black({7u, 8u, 11u, 255u});
    ASSERT_TRUE(run()) << error_;
    const auto complete = receipt();ASSERT_EQ(complete.size(), 512u);
    ASSERT_EQ(complete[385u], 4u);ASSERT_EQ(complete[386u], 1u);
    ASSERT_GT(complete[132u], 0u);ASSERT_EQ(complete[131u], complete[132u]);ASSERT_EQ(complete[158u], complete[132u]);
    auto candidate = complete;
    candidate[385u] = 1u;candidate[386u] = 0u;candidate[387u] = 5u;
    candidate[393u] = candidate[394u] = candidate[395u] = candidate[396u] = 0u;
    candidate[130u] = 2u;
    republish_enclosure(candidate);
    const auto recovered = receipt();
    expect_uniform_known(recovered, 0u, {7u, 8u, 11u, 255u}, 4u);
    EXPECT_EQ(recovered[130u], 0u);
    for (UINT fault = 0u; fault < 16u; ++fault) {
      SCOPED_TRACE(fault);
      auto malformed = candidate;
      switch (fault) {
        case 0u: malformed[388u] = 0u;break;
        case 1u: --malformed[390u];break;
        case 2u: malformed[391u] = 1u;break;
        case 3u: --malformed[392u];break;
        case 4u: malformed[397u] = 511u;break;
        case 5u: malformed[398u] = 1u;break;
        case 6u: malformed[389u] |= 0x01000000u;break;
        case 7u: --malformed[158u];break;
        case 8u: malformed[130u] = 3u;break;  // Candidate/literal black touches true boundary.
        case 9u: malformed[130u] = 6u;break;  // Original BODY overlaps hypothesized black.
        case 10u: --malformed[131u];break;
        case 11u: malformed[130u] = 0u;break;  // Missing prepared marker cannot authorize the candidate.
        case 12u: malformed[130u] = 1u;break;
        case 13u: malformed[388u] = 513u;break;
        case 14u: malformed[391u] = 65u;break;
        case 15u: malformed[393u] = 1u;break;
      }
      republish_enclosure(malformed);
      const auto rejected = receipt();ASSERT_EQ(rejected.size(), 512u);
      EXPECT_EQ(std::vector<UINT>(rejected.begin() + 384u, rejected.begin() + 400u),
        std::vector<UINT>(malformed.begin() + 384u, malformed.begin() + 400u));
      EXPECT_NE(rejected[385u], 4u);EXPECT_EQ(rejected[386u], 0u);
      EXPECT_EQ(rejected[130u], 0u);EXPECT_EQ(rejected[131u], 0u);
      EXPECT_EQ(rejected[157u], 0u);EXPECT_EQ(rejected[158u], 0u);
    }
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, UniformNonliteralPrivateModelCannotEnterThroughPalette) {
    author_band(5u, 275u);recolour_literal_black({7u, 8u, 11u, 255u});
    ASSERT_TRUE(run()) << error_;
    const auto e = helper_.enclosure_views();
    const auto b = helper_.baseline_views();
    const std::array<UINT, 4u> hook {};
    context_->UpdateSubresource(e.hook_constants.Get(), 0u, nullptr, hook.data(), 0u, 0u);
    std::array<ID3D11ShaderResourceView *, 16u> inputs {};
    std::array<ID3D11UnorderedAccessView *, 8u> outputs {};
    context_->CSSetShaderResources(0u, 16u, inputs.data());
    context_->CSSetUnorderedAccessViews(0u, 8u, outputs.data(), nullptr);
    auto palette_uav = uav(b.palette.Get());
    D3D11_UNORDERED_ACCESS_VIEW_DESC args_desc {};
    args_desc.Format = DXGI_FORMAT_R32_TYPELESS;args_desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    args_desc.Buffer.NumElements = 24u;args_desc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
    ComPtr<ID3D11UnorderedAccessView> args_uav;
    ASSERT_TRUE(SUCCEEDED(device_->CreateUnorderedAccessView(b.current_mode_args.Get(), &args_desc, &args_uav)));
    // Regenerate the actual unit0 palette, rather than inventing a palette header
    // from final CBC. The full run's frozen validation receipt and source still join.
    inputs[0] = rgb_.srv.Get();inputs[1] = probability_.srv.Get();inputs[2] = support_.srv.Get();
    inputs[3] = ocr_.srv.Get();inputs[8] = b.validation_receipt.Get();
    outputs[0] = palette_uav.Get();outputs[7] = args_uav.Get();
    ID3D11Buffer *constants[3] {cb_.resource.Get(), e.hook_constants.Get(), nullptr};
    context_->CSSetConstantBuffers(0u, 3u, constants);
    context_->CSSetShaderResources(0u, 16u, inputs.data());
    context_->CSSetUnorderedAccessViews(0u, 8u, outputs.data(), nullptr);
    context_->CSSetShader(colour_[0u].Get(), nullptr, 0u);context_->Dispatch(1u, 1u, 1u);
    inputs.fill(nullptr);outputs.fill(nullptr);
    context_->CSSetShaderResources(0u, 16u, inputs.data());
    context_->CSSetUnorderedAccessViews(0u, 8u, outputs.data(), nullptr);
    auto palette = words(read(b.palette.Get()));ASSERT_EQ(palette.size(), 128u);
    ASSERT_EQ(palette[48u], 0x314b4c42u);ASSERT_EQ(palette[49u], 1u);
    ASSERT_EQ(palette[50u], 0u);ASSERT_EQ(palette[51u], 5u);
    ASSERT_GE(palette[52u], 2u);ASSERT_EQ(palette[54u], palette[52u]);
    // The only injected payload is a syntactically valid private K4 in an
    // upstream role where K4 is forbidden. Samples/prototype are actual GPU data.
    palette[49u] = 4u;palette[50u] = 1u;palette[51u] = 1u;
    palette[57u] = palette[52u];palette[58u] = 11u;
    palette[59u] = palette[60u] = palette[53u];
    ComPtr<ID3D11Resource> palette_resource;b.palette->GetResource(&palette_resource);
    context_->UpdateSubresource(palette_resource.Get(), 0u, nullptr, palette.data(), 0u, 0u);
    inputs = {rgb_.srv.Get(), ocr_.srv.Get(), b.validation_receipt.Get(), b.palette.Get(),
      nullptr, b.cover_clip.Get(), b.parent.Get(), b.root_flags.Get(),
      e.all_body_bits.Get(), e.cover_bits.Get(), sct_.srv.Get(), uip_.srv.Get(),
      root_.srv.Get(), program_.enclosure.producer.original_source, b.body.Get(), b.matte.Get()};
    auto meta_uav = uav(e.receipt.Get()), family_uav = uav(e.family_models.Get());
    outputs[0] = meta_uav.Get();outputs[6] = family_uav.Get();
    constants[2] = e.producer_constants.Get();
    context_->CSSetConstantBuffers(0u, 3u, constants);
    context_->CSSetShaderResources(0u, 16u, inputs.data());
    context_->CSSetUnorderedAccessViews(0u, 8u, outputs.data(), nullptr);
    context_->CSSetShader(enclosure_[3u].Get(), nullptr, 0u);context_->Dispatch(1u, 1u, 1u);
    inputs.fill(nullptr);outputs.fill(nullptr);
    context_->CSSetShaderResources(0u, 16u, inputs.data());
    context_->CSSetUnorderedAccessViews(0u, 8u, outputs.data(), nullptr);
    const auto rejected = receipt();ASSERT_EQ(rejected.size(), 512u);
    EXPECT_NE(rejected[3u] & 8u, 0u);EXPECT_EQ(rejected[386u], 0u);
    EXPECT_EQ(std::vector<UINT>(rejected.begin() + 384u, rejected.begin() + 400u), std::vector<UINT>(16u, 0u));
    republish_enclosure(rejected);
    EXPECT_EQ(receipt()[2u], 0u);expect_baseline();
  }

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, UniformNonliteralRootZeroEmptyAndStaleNeverReuseRecoveredAuthority) {
    author_band(5u, 280u);recolour_literal_black({7u, 8u, 11u, 255u});
    ASSERT_TRUE(run()) << error_;
    const auto held_meta = receipt(), held_raw = ring_receipt(), held_f2 = calibration();
    const auto held_body = body(), held_matte = matte(), held_final = outline_matte();
    ASSERT_EQ(held_meta.size(), 512u);ASSERT_EQ(held_meta[385u], 4u);ASSERT_EQ(held_meta[386u], 1u);
    author(false, 281u, 0u, false, false, false, 256u, 128u, 3u);
    root_words_[60u] = 0u;upload();ASSERT_TRUE(run()) << error_;
    EXPECT_EQ(receipt(), held_meta);EXPECT_EQ(ring_receipt(), held_raw);EXPECT_EQ(calibration(), held_f2);
    EXPECT_EQ(body(), held_body);EXPECT_EQ(matte(), held_matte);EXPECT_EQ(outline_matte(), held_final);
    root_words_[60u] = 1u;upload();ASSERT_TRUE(run()) << error_;
    const auto empty = receipt();ASSERT_EQ(empty.size(), 512u);
    for (UINT unit = 0u; unit < 8u; ++unit) {EXPECT_EQ(empty[386u + 16u * unit], 0u);EXPECT_EQ(empty[130u + 32u * unit], 0u);}
    EXPECT_EQ(body(), std::vector<char>(W_ * H_ * 4u, 0));
    EXPECT_EQ(outline_matte(), std::vector<char>(W_ * H_ * 4u, 0));
    author_band(5u, 282u);recolour_literal_black({7u, 8u, 11u, 255u});
    uip_words_[4u] ^= 1u;upload();ASSERT_TRUE(run()) << error_;
    const auto stale = receipt();ASSERT_EQ(stale.size(), 512u);
    EXPECT_EQ(stale[2u], 0u);EXPECT_EQ(stale[386u], 0u);EXPECT_EQ(stale[130u], 0u);
    expect_baseline();expect_ring_parent();EXPECT_EQ(outline_matte(), matte());
    EXPECT_EQ(calibration()[2u], 0u);
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

  // Opt-in live-input diagnostic. Model RGB comes exclusively from the joined display
  // snapshot; the newer render source supplies the saved parent shape/format only, as in the
  // production Source5 request. Undumped descriptor flags and private history are not restored.
  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, SavedCurrentHdrProductionCompositionReplay) { saved_current(false); }
  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, SavedCurrentSdrProductionCompositionReplay) { saved_current(true); }

  void HostSbsCurrentOutlineSupportClosureGpuTest::saved_current(bool sdr) {
    const auto *input_env = std::getenv("SUNSHINE_TEST_SAVED_CURRENT_OUTLINE_CASES");
    if (!input_env || !*input_env) {
      GTEST_SKIP() << "No authenticated saved-current outline specification selected";
    }
    const auto *output_env = std::getenv("SUNSHINE_TEST_SAVED_CURRENT_OUTLINE_OUTPUT");
    ASSERT_TRUE(output_env && *output_env);
    const auto *graph_env = std::getenv("SUNSHINE_TEST_SAVED_CURRENT_OUTLINE_GRAPH_SNAPSHOTS");
    const bool graph_diagnostics = graph_env && *graph_env;
    const auto *foreground_env=std::getenv("SUNSHINE_TEST_SAVED_CURRENT_FOREGROUND_EDGE");
    const bool foreground_replay=foreground_env&&*foreground_env;
    if(foreground_replay){ASSERT_STREQ(foreground_env,"1");ASSERT_TRUE(sdr);ASSERT_FALSE(graph_diagnostics);ASSERT_FALSE(foreground_edge_closure_.empty());}
    if (graph_diagnostics) {
      ASSERT_STREQ(graph_env, "1");
    }
    const auto *inner_env = std::getenv("SUNSHINE_TEST_SAVED_CURRENT_INNER_SOURCE_DIAGNOSTICS");
    const bool inner_diagnostics = inner_env && *inner_env;
    if (inner_diagnostics) {
      ASSERT_STREQ(inner_env, "1");
      ASSERT_FALSE(graph_diagnostics);
      ASSERT_FALSE(foreground_replay);
      ASSERT_FALSE(sdr);
    }
    namespace fs = std::filesystem;
    namespace runtime = models::current_outline_runtime;
    namespace composition = models::source5_cbc5_f2_composition_gpu_v1;
    namespace selector = models::research_current_outline_selected_matte_v2;
    using json = nlohmann::json;
    const fs::path input(input_env), output(output_env);
    const auto specification_bytes = file_bytes(input);
    const auto spec = json::parse(specification_bytes.begin(), specification_bytes.end());
    ASSERT_EQ(spec.at("schema"), 1u);
    if(sdr){ASSERT_EQ(spec.at("diagnostic_role"),"same-published-sdr-source5-inputs");if(!foreground_replay)ASSERT_TRUE(graph_diagnostics)<<"SDRdiagnostic requires graph snapshots and strict production parity";}
    ASSERT_EQ(spec.at("driver"), hardware_ ? "hardware" : "warp");
    const auto &cases = spec.at("cases");
    ASSERT_TRUE(cases.is_array());
    ASSERT_GE(cases.size(), 1u);
    ASSERT_LE(cases.size(), 8u);
    ASSERT_TRUE(!fs::exists(output) || (fs::is_directory(output) && fs::is_empty(output)));
    fs::create_directories(output);
    runtime::programs_t programs;
    const auto runtime_root = spec.contains("runtime_root") ? fs::path(spec.at("runtime_root").get<std::string>()) :
                                                            fs::path(SUNSHINE_SHADERS_DIR) / "current_outline";
    ASSERT_TRUE(programs.load(device_.Get(), runtime_root, error_)) << error_;
    ASSERT_TRUE(programs.stability_ready()) << programs.stability_error();
    auto effective_composition = programs.composition_program();
    // Historical native controls stay OFF even when the new production loader
    // makes FG available. The explicit replay below runs a second fresh ON owner.
    effective_composition.outline.foreground_edge_shaders={};
    if (inner_diagnostics) {
      ASSERT_TRUE(std::all_of(effective_composition.outline.trusted_inner_publish_shaders.begin(),
        effective_composition.outline.trusted_inner_publish_shaders.end(), [](auto *shader) { return shader == nullptr; }))
        << "Strict captured-baseline inner diagnostics require the original route; trusted-inner treatment is incompatible";
    }
    if(foreground_replay){ASSERT_TRUE(programs.foreground_edge_ready());ASSERT_EQ(foreground_edge_closure_,runtime::foreground_edge_closure);}
    // Saved/native output diagnostics explicitly request the independent offer plane.
    effective_composition.outline.coverage_diagnostics = true;
    const auto *fixture_f2_env = std::getenv("SUNSHINE_TEST_SAVED_CURRENT_OUTLINE_FIXTURE_F2");
    const bool fixture_f2 = fixture_f2_env && *fixture_f2_env;
    if(foreground_replay)ASSERT_FALSE(fixture_f2)<<"FG port parity requires canonical production arrays";
    std::array<ComPtr<ID3D11ComputeShader>, 3u> fixture_stability;
    const char *fixture_outline_root = nullptr, *fixture_stability_root = nullptr, *fixture_stability_pin = nullptr;
    if (fixture_f2) {
      ASSERT_STREQ(fixture_f2_env, "1");
      // Only F2 shader arrays change. Source5, colour, graph, CBC/ROI, Inside and
      // selector remain the production loader's authenticated owned programs.
      ASSERT_EQ(inner_closure_, runtime::colour_closure);ASSERT_EQ(graph_closure_, runtime::components_closure);
      ASSERT_EQ(enclosure_closure_, runtime::enclosure_closure);ASSERT_EQ(inside_closure_, runtime::inside_closure);
      const auto *roi_pin = std::getenv("SUNSHINE_TEST_CURRENT_ROI_SHADER_PIN");
      ASSERT_NE(roi_pin, nullptr);ASSERT_EQ(std::string_view(roi_pin), runtime::roi_closure);
      fixture_outline_root = std::getenv("SUNSHINE_TEST_OUTLINE_CLOSURE_SHADER_ROOT");
      fixture_stability_root = std::getenv("SUNSHINE_TEST_BLACK_STABILITY_SHADER_ROOT");
      fixture_stability_pin = std::getenv("SUNSHINE_TEST_BLACK_STABILITY_SHADER_PIN");
      ASSERT_TRUE(fixture_outline_root && *fixture_outline_root && fixture_stability_root && *fixture_stability_root &&
        fixture_stability_pin && *fixture_stability_pin);
      ASSERT_EQ(fs::weakly_canonical(fixture_outline_root), fs::weakly_canonical(fixture_stability_root));
      load_shaders(fixture_stability_root, fixture_stability_pin, runtime::stability_specs, fixture_stability);
      effective_composition.outline.shaders = program_.shaders;
      effective_composition.outline.band_shaders = program_.band_shaders;
      effective_composition.outline.hole_shaders = program_.hole_shaders;
      effective_composition.outline.coverage_shaders = program_.coverage_shaders;
      for (UINT i = 0u; i < fixture_stability.size(); ++i) effective_composition.outline.stability_shaders[i] = fixture_stability[i].Get();
    }
    const auto store = [&](const fs::path &path, const std::vector<char> &bytes) {
      std::ofstream stream(path, std::ios::binary);
      require(bool(stream));
      stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
      require(bool(stream));
      return json {{"path", path.string()}, {"byte_count", bytes.size()}, {"sha256", hash_bytes(bytes)}};
    };
    std::array<wchar_t, 32768u> exe_path {};
    const auto exe_size = GetModuleFileNameW(nullptr, exe_path.data(), DWORD(exe_path.size()));
    ASSERT_GT(exe_size, 0u);
    ASSERT_LT(exe_size, exe_path.size());
    const fs::path exe(std::wstring(exe_path.data(), exe_size));
    const auto executable = file_bytes(exe);
    const auto fixture_path = fs::path(SUNSHINE_SOURCE_DIR) / "tests/unit/test_host_sbs_current_outline_support_closure_gpu.cpp";
    const auto fixture = file_bytes(fixture_path);
    json report {{"schema", 1u}, {"driver", hardware_ ? "hardware" : "warp"}, {"adapter", adapter_},
      {"test_executable", {{"path", exe.string()}, {"sha256", hash_bytes(executable)}}},
      {"fixture_source", {{"path", fixture_path.string()}, {"sha256", hash_bytes(fixture)}}},
      {"input_specification", {{"path", input.string()}, {"sha256", hash_bytes(specification_bytes)}}},
      {"runtime_root", runtime_root.string()},
      {"closures", {{"source5", runtime::source5_closure}, {"colour", runtime::colour_closure},
        {"components", runtime::components_closure}, {"enclosure", runtime::enclosure_closure},
        {"roi", runtime::roi_closure}, {"outline", runtime::outline_closure},
        {"inside", runtime::inside_closure}, {"selector", runtime::selector_closure},
        {"stability", runtime::stability_closure}}},
      {"hole_policy", {{"compiled_source_closure",runtime::hole_closure},{"production_program_ready",programs.hole_ready()},
        {"compiled_helper_pin",compiled_hole_helper_pin}}},
      {"coverage_policy", {{"compiled_source_closure",runtime::coverage_closure},{"production_program_ready",programs.coverage_ready()}}},
      {"foreground_edge_policy", {{"compiled_source_closure",runtime::foreground_edge_closure},{"production_program_ready",programs.foreground_edge_ready()},
        {"requested",foreground_replay},{"baseline_arrays_foreground_edge_OFF",true},{"scope","Explicit ON arrays are prefixed foreground_edge_; baseline current arrays remain OFF"}}},
      {"trusted_inner_policy", {{"compiled_source_closure",runtime::trusted_inner_closure},{"production_program_ready",programs.trusted_inner_ready()},
        {"production_route_enabled",effective_composition.outline.trusted_inner_publish_shaders[0u]!=nullptr},
        {"scope","Current production route; captured expected-M binding and equality remain independently reported"}}},
      {"scope", "Actual production Source5/CBC/Raw/F2 and selected-matte GPU programs; no CPU mask reconstruction"},
      {"history", "New owner per case; prior live strong/weak/AA private history is unavailable and is not synthesized"},
      {"parent_descriptor_scope", "Saved full shape/format/ROI/transfer; undumped descriptor flags are not claimed byteexact, and the GPU proof describes the replay allocation"},
      {"single_unit_graph_diagnostics_requested", graph_diagnostics},
      {"inner_source_diagnostics_requested", inner_diagnostics},
      {"performance_claim", false}, {"cases", json::array()}};
    report["closures"]["hole"] = programs.hole_ready() ? std::string(runtime::hole_closure) : std::string();
    report["closures"]["coverage"] = programs.coverage_ready() ? std::string(runtime::coverage_closure) : std::string();
    if (fixture_f2) {
      report["compiled_production_closures"] = report["closures"];
      report["closures"]["outline"] = ring_closure_;
      report["closures"]["stability"] = fixture_stability_pin;
      report["closures"]["hole"] = hole_closure_;
      report["closures"]["coverage"] = coverage_closure_;
      report["effective_composition_closures"] = report["closures"];
      report["prototype_not_production_qualified"] = true;
      report["fixture_f2_override"] = {{"outline_root", fixture_outline_root}, {"outline_closure", ring_closure_},
        {"stability_root", fixture_stability_root}, {"stability_closure", fixture_stability_pin},
        {"compiled_outline_helper_pin", compiled_ring_helper_pin},
        {"replaced_arrays", {"outline.shaders[8]", "outline.band_shaders[7]", "outline.stability_shaders[3]", "outline.hole_shaders[9]"}},
        {"closures_field_scope", "Actual executed group closures; compiled_production_closures records the unchanged production-loader inventory"}};
      report["fixture_f2_override"]["hole_closure"] = hole_closure_;
      report["fixture_f2_override"]["coverage_closure"] = coverage_closure_;
      if (const auto *root = std::getenv("SUNSHINE_TEST_STROKE_COVERAGE_SHADER_ROOT")) report["fixture_f2_override"]["coverage_root"] = root;
      if (const auto *root = std::getenv("SUNSHINE_TEST_CURRENT_HOLE_SHADER_ROOT")) report["fixture_f2_override"]["hole_root"] = root;
      report["scope"] = "Production Source5/CBC/Raw/Inside/selector with explicitly pinned fixture-F2 prototype; effective_composition_closures records actual execution, not production qualification";
    }
    for (std::size_t index = 0u; index < cases.size(); ++index) {
      const auto &entry = cases.at(index);
      SCOPED_TRACE(entry.at("name").get<std::string>());
      const auto source_shape = entry.at("source").get<std::array<UINT, 2u>>();
      const auto shape = entry.at("analysis_shape").get<std::array<UINT, 4u>>();
      const auto field = entry.at("field").get<std::array<UINT, 2u>>();
      const auto roi = entry.at("roi_ltrb").get<std::array<UINT, 4u>>();
      ASSERT_GT(source_shape[0], 0u);ASSERT_GT(source_shape[1], 0u);
      ASSERT_LE(source_shape[0], 16384u);ASSERT_LE(source_shape[1], 16384u);
      ASSERT_GT(shape[0], 0u);ASSERT_GT(shape[1], 0u);ASSERT_GT(shape[3], 0u);
      ASSERT_LE(shape[0], 5120u);ASSERT_LE(shape[1], 5120u);
      ASSERT_LT(shape[2], shape[1]);ASSERT_EQ(shape[3], shape[1] - shape[2]);
      ASSERT_LT(roi[0], roi[2]);ASSERT_LT(roi[1], roi[3]);
      ASSERT_LE(roi[2], source_shape[0]);ASSERT_LE(roi[3], source_shape[1]);
      ASSERT_EQ(roi[2] - roi[0], shape[0]);ASSERT_EQ(roi[3] - roi[1], shape[1]);
      ASSERT_GT(field[0], 0u);ASSERT_GT(field[1], 0u);
      ASSERT_LE(field[0], 5120u);ASSERT_LE(field[1], 5120u);
      ASSERT_EQ(entry.at("transfer_mode"), sdr?0u:2u);
      if(sdr){ASSERT_EQ(entry.at("source_format"),87u);}
      const auto frame = entry.at("model_frame").get<std::uint64_t>();
      const auto clock = entry.at("observation_us").get<std::uint64_t>();
      const auto generation = entry.at("analysis_generation").get<std::uint64_t>();
      const auto token = entry.at("token").get<std::uint64_t>();
      const auto mode = entry.at("postprocess_mode").get<UINT>();
      ASSERT_NE(frame, 0u);ASSERT_NE(clock, 0u);ASSERT_NE(token, 0u);
      ASSERT_TRUE(mode == 1u || mode == 2u || mode == 4u);
      ASSERT_EQ(entry.at("expected_work"), 1u);
      const std::size_t pixels = std::size_t(shape[0]) * shape[3];
      json bindings = json::object();
      std::map<std::string, std::vector<char>> imported;
      const auto operand = [&](const char *key, std::size_t count) {
        const auto &binding = entry.at("artifacts").at(key);
        const auto bytes = bound_bytes(binding, count);
        bindings[key] = binding;
        imported.emplace(key, bytes);
        return bytes;
      };
      const auto source = operand(sdr?"source_bgra8":"source_rgba16f", std::size_t(source_shape[0]) * source_shape[1] * (sdr?4u:8u));
      if(sdr){const auto rgba=operand("original_source_rgba8",source.size());for(std::size_t k=0u;k<source.size();k+=4u){ASSERT_EQ(source[k],rgba[k+2u]);ASSERT_EQ(source[k+1u],rgba[k+1u]);ASSERT_EQ(source[k+2u],rgba[k]);ASSERT_EQ(source[k+3u],rgba[k+3u]);}}
      const auto display = operand("display_snapshot", pixels * 8u);
      const auto probability = operand("probability", probability_word_count * 4u);
      const auto support = operand("support", pixels * 4u);
      const auto ocr = operand("ocr_record", 1024u), uip = operand("current_uip", 160u);
      const auto sct = operand("current_sct", 128u), constants = operand("constants", 96u);
      const auto root = operand("root_arguments", 256u), legacy = operand("legacy_matte", pixels * 4u);
      const auto expected = operand("expected_selected_matte", pixels * 4u);
      const auto c = words(constants), s = words(sct), o = words(ocr), u = words(uip), r = words(root);
      ASSERT_EQ(c.size(), 24u);ASSERT_EQ(s.size(), 32u);ASSERT_EQ(o.size(), 256u);
      ASSERT_EQ(u.size(), 40u);ASSERT_EQ(r.size(), 64u);
      ASSERT_TRUE(std::equal(shape.begin(), shape.end(), c.begin()));
      ASSERT_EQ(c[4u], field[0]);ASSERT_EQ(c[5u], field[1]);
      const auto wide = [](UINT lo, UINT hi) {return std::uint64_t(lo) | (std::uint64_t(hi) << 32u);};
      ASSERT_EQ(wide(c[12u], c[13u]), frame);ASSERT_EQ(wide(c[14u], c[15u]), generation);
      ASSERT_EQ(wide(c[16u], c[17u]), clock);ASSERT_EQ(wide(r[2u], r[3u]), token);
      ASSERT_TRUE(std::equal(c.begin(), c.begin() + 12u, s.begin() + 8u));
      ASSERT_TRUE(std::equal(c.begin() + 12u, c.begin() + 16u, s.begin() + 4u));
      ASSERT_TRUE(std::equal(c.begin() + 12u, c.begin() + 16u, o.begin() + 5u));
      ASSERT_TRUE(std::equal(c.begin() + 12u, c.begin() + 16u, u.begin() + 4u));
      ASSERT_EQ(wide(u[8u], u[9u]), clock);ASSERT_EQ(wide(s[20u], s[21u]), clock);
      const auto legacy_words = words(legacy), expected_words = words(expected), support_values = words(support);
      for (const auto &values : {&legacy_words, &expected_words, &support_values}) {
        ASSERT_TRUE(std::all_of(values->begin(), values->end(), [](UINT value) {return value <= 1u;}));
      }
      texture_t source_texture;
      if(sdr){
        D3D11_TEXTURE2D_DESC desc{};desc.Width=source_shape[0];desc.Height=source_shape[1];desc.MipLevels=desc.ArraySize=1u;
        desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.SampleDesc.Count=1u;desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        ASSERT_TRUE(SUCCEEDED(device_->CreateTexture2D(&desc,nullptr,&source_texture.resource)));
        ASSERT_TRUE(SUCCEEDED(device_->CreateShaderResourceView(source_texture.resource.Get(),nullptr,&source_texture.srv)));
      }else source_texture=texture(source_shape[0],source_shape[1],DXGI_FORMAT_R16G16B16A16_FLOAT);
      auto display_texture = texture(shape[0], shape[3], DXGI_FORMAT_R16G16B16A16_FLOAT);
      auto support_texture = texture(shape[0], shape[3], DXGI_FORMAT_R32_UINT);
      auto legacy_texture = texture(shape[0], shape[3], DXGI_FORMAT_R32_UINT);
      auto p_buffer = buffer(probability_word_count * 4u), o_buffer = buffer(1024u);
      auto u_buffer = buffer(160u), s_buffer = buffer(128u), c_buffer = buffer(96u, true);
      auto r_buffer = buffer(256u, false, true);
      context_->UpdateSubresource(source_texture.resource.Get(), 0u, nullptr, source.data(), source_shape[0] * (sdr?4u:8u), 0u);
      context_->UpdateSubresource(display_texture.resource.Get(), 0u, nullptr, display.data(), shape[0] * 8u, 0u);
      context_->UpdateSubresource(support_texture.resource.Get(), 0u, nullptr, support.data(), shape[0] * 4u, 0u);
      context_->UpdateSubresource(legacy_texture.resource.Get(), 0u, nullptr, legacy.data(), shape[0] * 4u, 0u);
      for (const auto &[target, bytes] : std::array<std::pair<ID3D11Resource *, const std::vector<char> *>, 6u> {{
        {p_buffer.resource.Get(), &probability}, {o_buffer.resource.Get(), &ocr}, {u_buffer.resource.Get(), &uip},
        {s_buffer.resource.Get(), &sct}, {c_buffer.resource.Get(), &constants}, {r_buffer.resource.Get(), &root}}}) {
        context_->UpdateSubresource(target, 0u, nullptr, bytes->data(), 0u, 0u);
      }
      composition::resources_t owner;
      owner.set_recorded_execution(true);
      ASSERT_TRUE(owner.ensure(device_.Get(), shape[0], shape[1], shape[2], shape[3], error_)) << error_;
      composition::request_t request;
      request.source.display_snapshot_band = display_texture.srv.Get();
      request.source.parent_raw_source = source_texture.srv.Get();
      request.source.actual_contour_constants96 = c_buffer.resource.Get();
      request.source.actual_sct32 = s_buffer.srv.Get();request.source.actual_final256 = r_buffer.resource.Get();
      request.source.roi_ltrb = roi;request.source.frame = frame;request.source.observation_us = clock;
      request.source.analysis_generation = generation;request.source.token = token;request.source.transfer_mode = sdr?0u:2u;
      request.probability = p_buffer.srv.Get();request.support = support_texture.srv.Get();
      request.ocr_record = o_buffer.srv.Get();request.actual_uip40 = u_buffer.srv.Get();
      request.original_legacy_matte = legacy_texture.srv.Get();request.postprocess_mode = mode;request.expected_work = 1u;
      ASSERT_TRUE(owner.queue(context_.Get(), effective_composition, request, error_)) << error_;
      const auto bundle = owner.bundle();
      ASSERT_TRUE(bundle.all_stages_issued);
      selector::inputs_t in;
      in.current_cb96 = bundle.current.actual_contour_constants96.Get();in.actual_root_args = bundle.current.actual_final256.Get();
      in.legacy_matte = bundle.current.original_legacy_matte.Get();in.sct = bundle.current.sct.Get();
      in.mode = static_cast<selector::postprocess_mode_e>(mode);in.expected_work = 1u;in.transaction_token = token;
      in.current_outline_requested = true;in.baseline_matte = bundle.baseline_matte.Get();in.numeric_proof = bundle.numeric_proof.Get();
      in.source_envelope104 = bundle.source.uip_envelope.Get();in.cbc_meta512 = bundle.enclosure.receipt.Get();
      in.raw_meta512 = bundle.ring.receipt.Get();in.f2_data = bundle.outline.calibration_data.Get();in.f2_matte = bundle.outline.matte.Get();
      selector::resources_t selected;
      ASSERT_TRUE(selected.ensure(device_.Get(), shape[0], shape[3], error_)) << error_;
      ASSERT_TRUE(selected.queue(context_.Get(), programs.selector_program(), in, error_)) << error_;
      const auto selected_views = selected.views();
      // Mapping these exact outputs retires all commands before any owner/resource is reused.
      std::map<std::string, std::vector<char>> arrays {
        {"source_proof.u32", read(bundle.source.source_proof.Get())}, {"source_envelope.u32", read(bundle.source.uip_envelope.Get())},
        {"CBC.u32", read(bundle.enclosure.receipt.Get())}, {"Raw.u32", read(bundle.ring.receipt.Get())},
        {"F2.u32", read(bundle.outline.calibration_data.Get())}, {"BODY.u32", read(bundle.enclosure.body.Get())},
        {"CBC_M.u32", read(bundle.enclosure.matte.Get())}, {"Raw_M.u32", read(bundle.ring.matte.Get())},
        {"F2_M.u32", read(bundle.outline.matte.Get())}, {"selected_M.u32", read(selected_views.matte.Get())},
        {"selected_receipt.u32", read(selected_views.receipt.Get())}, {"numeric_proof.u32", read(bundle.numeric_proof.Get())}};
      const auto selected_receipt = words(arrays.at("selected_receipt.u32"));
      ASSERT_EQ(selected_receipt.size(), 64u);
      EXPECT_EQ(selected_receipt[0u], selector::tag);EXPECT_EQ(selected_receipt[1u], selector::schema);
      EXPECT_EQ(selected_receipt[2u], 3u);EXPECT_EQ(selected_receipt[3u], 0u);EXPECT_EQ(selected_receipt[62u], 2u);
      EXPECT_TRUE(std::equal(c.begin(), c.end(), selected_receipt.begin() + 4u));
      EXPECT_TRUE(std::equal(s.begin(), s.end(), selected_receipt.begin() + 28u));
      EXPECT_EQ(wide(selected_receipt[60u], selected_receipt[61u]), token);
      const auto actual = words(arrays.at("selected_M.u32"));
      ASSERT_EQ(actual.size(), pixels);
      std::size_t count = 0u, expected_count = 0u, added = 0u, removed = 0u;
      for (std::size_t pixel = 0u; pixel < pixels; ++pixel) {
        ASSERT_LE(actual[pixel], 1u);count += actual[pixel];expected_count += expected_words[pixel];
        added += actual[pixel] == 1u && expected_words[pixel] == 0u;
        removed += actual[pixel] == 0u && expected_words[pixel] == 1u;
      }
      const bool current = selected_receipt[0u] == selector::tag && selected_receipt[1u] == selector::schema &&
        selected_receipt[2u] == 3u && selected_receipt[3u] == 0u && selected_receipt[62u] == 2u &&
        selected_receipt[63u] == 0u && std::equal(c.begin(), c.end(), selected_receipt.begin() + 4u) &&
        std::equal(s.begin(), s.end(), selected_receipt.begin() + 28u) &&
        wide(selected_receipt[60u], selected_receipt[61u]) == token;
      const bool bytes_equal = arrays.at("selected_M.u32") == expected;
      const bool exact = current && bytes_equal;
      json production_parity=nullptr;
      if(sdr){
        production_parity=json::object();bool parity_ok=current;
        for(const auto *name:{"CBC.u32","Raw.u32","F2.u32","selected_M.u32"}){
          const auto &expected_binding=entry.at("expected_production").at(name);
          const auto expected_bytes=bound_bytes(expected_binding,arrays.at(name).size());
          const auto expected_key=std::string("expected_production:")+name;bindings[expected_key]=expected_binding;imported.emplace(expected_key,expected_bytes);
          const bool same=arrays.at(name)==expected_bytes;production_parity[name]=same;parity_ok=parity_ok&&same;
        }
        const auto carrier=read(bundle.source.reference_rgb8.Get());const std::size_t first=std::size_t(shape[0])*shape[2]*4u;
        ASSERT_EQ(carrier.size(),std::size_t(shape[0])*shape[1]*4u);
        arrays.emplace("source5_reference_band_RGBA8.bin",std::vector<char>(carrier.begin()+first,carrier.begin()+first+pixels*4u));
        if(!parity_ok){
          const auto failed=output/("case_"+std::to_string(index));fs::create_directory(failed);json saved=json::object();
          for(const auto &[name,bytes]:arrays)saved[name]=store(failed/name,bytes);
          report["cases"].push_back({{"name",entry.at("name")},{"input_bindings",bindings},{"artifacts",saved},{"expected_production_parity",production_parity},
            {"trace_authorized",false},{"history_restored",false},{"replay_classification","production-parity-not-established-no-graph-trace"}});
          const auto text=report.dump(2);store(output/"report.json",std::vector<char>(text.begin(),text.end()));
          ADD_FAILURE()<<"SDR replay must match current production CBC/Raw/F2/M before graph interpretation";continue;
        }
      }
      json foreground_case=nullptr;
      #include "current_outline_foreground_edge_saved.inc"
      json graph_report = nullptr;
      if (graph_diagnostics) {
        // The composition intentionally hides mutable scratch and accepts no diagnostic
        // destinations. Run its same F2 programs on its exact already-authenticated
        // Source5 operands in a fresh public test owner. Full output parity is mandatory
        // before attributing any scratch value to the production composition above.
        const auto raw_meta = words(arrays.at("Raw.u32"));
        ASSERT_EQ(raw_meta.size(), 512u);ASSERT_EQ(raw_meta[6u], 1u);
        for (UINT unit = 0u; unit < 8u; ++unit) {
          ASSERT_EQ(raw_meta[128u + 32u * unit], unit == 0u ? 1u : 0u);
          ASSERT_EQ(raw_meta[129u + 32u * unit], unit == 0u ? 1u : 0u);
        }
        const auto p = effective_composition;
        auto diagnostic_program = p.outline;
        diagnostic_program.coverage_diagnostics = true;
        diagnostic_program.enclosure.current_sct = bundle.current.sct.Get();
        diagnostic_program.enclosure.current_uip = bundle.source.uip_envelope.Get();
        auto &producer = diagnostic_program.enclosure.producer;
        producer.original_source = bundle.source.reference_rgb8.Get();
        producer.transfer = enclosure::transfer_t::encoded_sdr;
        producer.input_known = false;producer.direct_encoded_recipe = 5u;
        producer.postprocess_mode = bundle.current.postprocess_mode;
        producer.expected_work = bundle.current.expected_work;
        producer.transaction_token = bundle.current.token;
        producer.model_frame = bundle.current.frame;
        producer.model_observation_us = bundle.current.observation_us;
        producer.analysis_generation = bundle.current.analysis_generation;
        producer.analysis_roi = {0u, 0u, shape[0], shape[1]};
        auto diagnostic_packed = texture(shape[0], shape[3], DXGI_FORMAT_R32_UINT);
        diagnostic_program.packed_temp_snapshot = diagnostic_packed.resource.Get();
        ring::resources_t diagnostic;
        diagnostic.set_recorded_execution(false);
        ASSERT_TRUE(diagnostic.ensure(device_.Get(), shape[0], shape[3], error_)) << error_;
        ASSERT_TRUE(diagnostic.dispatch(context_.Get(), p.colour, p.components,
          bundle.current.actual_contour_constants96.Get(), bundle.source.analysis_band.Get(),
          bundle.current.probability.Get(), bundle.current.support.Get(), bundle.current.ocr_record.Get(),
          bundle.current.actual_final256.Get(), 240u, error_, &diagnostic_program)) << error_;
        const auto e = diagnostic.enclosure_views();const auto b = diagnostic.baseline_views();
        const auto r = diagnostic.raw_views();const auto f = diagnostic.mutable_views();
        const std::map<std::string, std::vector<char>> parity {
          {"CBC.u32", read(e.receipt.Get())}, {"Raw.u32", read(r.receipt.Get())},
          {"F2.u32", read(f.calibration_data.Get())}, {"BODY.u32", read(e.body.Get())},
          {"CBC_M.u32", read(e.matte.Get())}, {"Raw_M.u32", read(r.matte.Get())},
          {"F2_M.u32", read(f.matte.Get())}, {"numeric_proof.u32", read(b.validation_receipt.Get())}};
        json equal = json::object();
        for (const auto &[name, bytes] : parity) {
          ASSERT_EQ(bytes, arrays.at(name)) << "Source5-backed diagnostic mirror changed " << name;
          equal[name] = hash_bytes(bytes);
        }
        arrays.emplace("graph_unit0_RawParent.u32", read(b.parent.Get()));
        arrays.emplace("graph_unit0_RawRootFlags.u32", read(b.root_flags.Get()));
        arrays.emplace("graph_unit0_RawPassable.u32", read(b.eligible.Get()));
        arrays.emplace("graph_unit0_InnerEligible.u32", read(r.eligible_inner.Get()));
        arrays.emplace("graph_unit0_PassableCache.u32", read(f.passable_root_cache.Get()));
        arrays.emplace("graph_unit0_ModeFinalScratch.u32", read(b.mode_final.Get()));
        arrays.emplace("graph_unit0_CoverBits.u32", read(e.cover_bits.Get()));
        arrays.emplace("graph_unit0_SelectedBits.u32", read(e.selected_bits.Get()));
        arrays.emplace("graph_unit0_PackedBeforePublication.u32", read(diagnostic_packed.resource.Get()));
        arrays.emplace("graph_unit0_analysis_RGBA16F.bin", read(bundle.source.analysis_band.Get()));
        arrays.emplace("graph_unit0_FamilyModels17x8.u32",read(e.family_models.Get()));
        if(diagnostic.mask_holes_enabled()) {
          arrays.emplace("graph_unit0_HoleProof.u32",read(diagnostic.mask_hole_proof()));
          arrays.emplace("graph_unit0_HoleParent.u32",read(diagnostic.mask_hole_parents()));
          arrays.emplace("graph_unit0_HoleFlags.u32",read(diagnostic.mask_hole_flags()));
        }
        if(diagnostic.stroke_coverage_enabled())
          arrays.emplace("graph_unit0_CoverageOffers.u32",read(diagnostic.stroke_coverage_offers()));
        for (const auto &name : {"graph_unit0_RawParent.u32", "graph_unit0_RawRootFlags.u32"}) {
          ASSERT_EQ(arrays.at(name).size(), (pixels + 1u) * 4u);
        }
        for (const auto &name : {"graph_unit0_RawPassable.u32", "graph_unit0_InnerEligible.u32",
          "graph_unit0_PassableCache.u32", "graph_unit0_ModeFinalScratch.u32",
          "graph_unit0_CoverBits.u32", "graph_unit0_SelectedBits.u32", "graph_unit0_PackedBeforePublication.u32"}) {
          ASSERT_EQ(arrays.at(name).size(), pixels * 4u);
        }
        ASSERT_EQ(arrays.at("graph_unit0_analysis_RGBA16F.bin").size(), pixels * 8u);
        graph_report = {{"unit", 0u}, {"issued_unit_mask", 1u}, {"production_outputs_byteexact", equal},
          {"capture_phase", "Fresh Source5-backed mirror after all F2/Inside completion; exact production-output parity required"},
          {"authority_scope", "Exactly unit0 issued. Later zero-dispatch units and final Inside do not overwrite Raw Parent/Flags/Passable or InnerEligible"},
          {"cache_scope", "First-pass root/enclosure identity after aggregation; mutable BAD_RIM is not a pre-aggregate observation and is ignored by CachedAABackground"},
          {"mode_final_scope", "Literal shared mode-final scratch; semantics depend on bound producer closure, never standalone rendering authority"},
          {"hole_policy", {{"phase_resources_enabled",diagnostic.mask_holes_enabled()},
            {"compiled_helper_pin",compiled_hole_helper_pin},{"source_closure",programs.hole_ready()?std::string(runtime::hole_closure):std::string()}}},
          {"domain_scope", "Only current own CoverBits/unit ROI establishes graph membership; exported whole resources do not authorize unused pixels"},
          {"history_restored", false}, {"rendering_authority", false}};
      }
      json inner_report = nullptr;
      #include "current_outline_inner_source_saved.inc"
      const auto case_output = output / ("case_" + std::to_string(index));fs::create_directory(case_output);
      json outputs = json::object();
      for (const auto &[name, bytes] : arrays) outputs[name] = store(case_output / name, bytes);
      json result {{"name", entry.at("name")}, {"input_bindings", bindings}, {"artifacts", outputs},
        {"model_frame", frame}, {"observation_us", clock}, {"analysis_generation", generation}, {"token", token},
        {"actual_selected_pixels", count}, {"captured_selected_pixels", expected_count},
        {"added_against_captured", added}, {"removed_against_captured", removed},
        {"current_outline_authorized", current}, {"selected_matte_bytes_equal", bytes_equal}, {"selected_matte_bitexact", exact},
        {"replay_classification", !current ? "current-outline-publication-unavailable" :
          exact ? "captured-current-mask-exact-with-fresh-history" : "fresh-history-current-producer-only-mask-differs"},
        {"history_restored", false}, {"model_rgb_origin", "exact joined display_snapshot; parent raw texels never sampled"}};
      if (graph_diagnostics) result["single_unit_graph_diagnostics"] = std::move(graph_report);
      if (inner_diagnostics) result["inner_source_diagnostics"] = std::move(inner_report);
      if(foreground_replay)result["foreground_edge_production_diagnostic"]=std::move(foreground_case);
      if(sdr){result["expected_production_parity"]=production_parity;result["trace_authorized"]=true;result["source5_reference_band_rect"]={0u,shape[2],shape[0],shape[1]};}
      if (entry.contains("provenance")) result["provenance"] = entry.at("provenance");
      report["cases"].push_back(std::move(result));
      RecordProperty("saved_current_" + std::to_string(index) + "_selected_matte_bitexact", exact ? "true" : "false");
      RecordProperty("saved_current_" + std::to_string(index) + "_selected_M_sha256", hash_bytes(arrays.at("selected_M.u32")));
      for (const auto &[key, bytes] : imported) ASSERT_EQ(file_bytes(bindings.at(key).at("path").get<std::string>()), bytes);
    }
    ASSERT_EQ(file_bytes(input), specification_bytes);
    ASSERT_EQ(file_bytes(exe), executable);ASSERT_EQ(file_bytes(fixture_path), fixture);
    const auto text = report.dump(2);store(output / "report.json", std::vector<char>(text.begin(), text.end()));
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

  TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, SameDomainSubtitleResetRevokesStrongBlackBandAndAAOffers) {
    const char *stability_root = std::getenv("SUNSHINE_TEST_BLACK_STABILITY_SHADER_ROOT");
    const char *stability_pin = std::getenv("SUNSHINE_TEST_BLACK_STABILITY_SHADER_PIN");
    ASSERT_NE(stability_root, nullptr);
    ASSERT_NE(stability_pin, nullptr);
    std::array<ComPtr<ID3D11ComputeShader>, 3u> stability_shaders;
    const std::array stability_specs {
      cache::shader_spec {"host_sbs_current_outline_support_band_cs.hlsl", "private_cache_begin_main", "cs_5_0"},
      cache::shader_spec {"host_sbs_current_outline_support_band_cs.hlsl", "private_cache_validate_main", "cs_5_0"},
      cache::shader_spec {"host_sbs_current_outline_support_band_cs.hlsl", "private_cache_commit_main", "cs_5_0"}
    };
    load_shaders(stability_root, stability_pin, stability_specs, stability_shaders);
    for (UINT i = 0u; i < stability_shaders.size(); ++i) {
      program_.stability_shaders[i] = stability_shaders[i].Get();
    }
    const auto author_same_pixels = [&](UINT frame, bool reset) {
      author_band(5u, frame);
      rect(73u, 60u, 74u, 61u, {64u, 64u, 64u, 255u});
      upload_pixels_from_origin();
      cb_words_[18u] = reset ? 1u : 0u;
      upload();
    };
    author_same_pixels(700u, false);
    ASSERT_TRUE(run()) << error_;
    const auto seeded = aa_history_snapshot("context-A-seeded");
    const auto seeded_header = words(seeded.header), seeded_history = words(seeded.history);
    ASSERT_EQ(seeded_header[2u], 1u);
    ASSERT_EQ(seeded_header[146u], 1u);
    ASSERT_GT(std::count_if(seeded_history.begin(), seeded_history.end(), [](UINT x) { return (x & 2u) != 0u; }), 0);

    author_same_pixels(701u, false);
    ASSERT_TRUE(run()) << error_;
    const auto continued = aa_history_snapshot("context-A-continued");
    const auto continued_header = words(continued.header);
    ASSERT_EQ(continued_header[145u], 1u) << "An unchanged same-domain observation must authorize the seeded strong reference.";

    // This is the production owner operation used at the quiescent semantic-context boundary.
    // No source clock advance or subtitle dispatch is required to revoke the old permissions.
    const auto weak_view = helper_.weak_aa_state_view();
    ASSERT_TRUE(weak_view);
    const auto weak_before = words(read(weak_view.Get()));
    ASSERT_TRUE(helper_.invalidate_subtitle_history(context_.Get()));
    const auto invalidated = aa_history_snapshot("context-B-invalidated");
    const auto invalidated_header = words(invalidated.header);
    const auto invalidated_weak = words(read(helper_.weak_aa_state_view().Get()));
    EXPECT_EQ(invalidated_header[2u], 0u);
    EXPECT_EQ(invalidated_header[152u], 0u);
    EXPECT_EQ(invalidated_header[63u], std::numeric_limits<UINT>::max());
    EXPECT_EQ(invalidated_header[104u], continued_header[104u]);
    EXPECT_EQ(invalidated_header[56u], continued_header[56u]);
    EXPECT_EQ(invalidated_header[60u], continued_header[60u]);
    EXPECT_EQ(invalidated_weak[2u], 0u);
    EXPECT_EQ(invalidated_weak[4u], 0u);
    EXPECT_EQ(invalidated_weak[35u], std::numeric_limits<UINT>::max());
    EXPECT_EQ(invalidated_weak[11u], weak_before[11u]);
    EXPECT_EQ(invalidated_weak[28u], weak_before[28u]);
    EXPECT_EQ(invalidated_weak[32u], weak_before[32u]);
    EXPECT_EQ(invalidated.history, continued.history);
    EXPECT_EQ(invalidated.rgb, continued.rgb);

    // Even an exact same frame/clock/constant tuple cannot reconstruct duplicate permissions.
    ASSERT_TRUE(run()) << error_;
    const auto duplicate = aa_history_snapshot("old-context-exact-tuple");
    const auto duplicate_header = words(duplicate.header);
    const auto duplicate_weak = words(read(helper_.weak_aa_state_view().Get()));
    EXPECT_EQ(duplicate_header[2u], 0u);
    EXPECT_EQ(duplicate_header[145u], 0u);
    EXPECT_EQ(duplicate_header[146u], 0u);
    EXPECT_EQ(duplicate_weak[2u], 0u);
    EXPECT_EQ(duplicate_weak[4u], 0u);

    // A new frame with the same source clock also cannot reuse A's cached permissions. Keep
    // current SCT/UIP/producer proof coherent; no raw cached model or mask is uploaded.
    const auto same_domain = std::array {cb_words_[14u], cb_words_[15u], sct_words_[3u]};
    author_same_pixels(702u, false);
    cb_words_[16u] = sct_words_[20u] = uip_words_[8u] = 1000701u;
    program_.enclosure.producer.model_observation_us = 1000701u;
    upload();
    ASSERT_EQ((std::array {cb_words_[14u], cb_words_[15u], sct_words_[3u]}), same_domain);
    ASSERT_TRUE(run()) << error_;
    const auto equal_clock = aa_history_snapshot("context-B-equal-clock");
    const auto equal_header = words(equal_clock.header);
    EXPECT_EQ(equal_header[2u], 0u);
    EXPECT_EQ(equal_header[145u], 0u);
    EXPECT_EQ(equal_header[146u], 0u);
    EXPECT_EQ(equal_header[56u], continued_header[56u]);
    EXPECT_EQ(equal_header[60u], continued_header[60u]);
    EXPECT_EQ(equal_clock.history, continued.history);
    EXPECT_EQ(equal_clock.rgb, continued.rgb);
    const auto current_matte = words(outline_matte());
    EXPECT_GT(std::count_if(current_matte.begin(), current_matte.end(), [](UINT x) { return x != 0u; }), 0);

    // Only genuinely newer current evidence may create B's own strong/AA reference.
    author_same_pixels(703u, false);
    ASSERT_TRUE(run()) << error_;
    const auto reset = aa_history_snapshot("context-B-new-current-reference");
    const auto reset_header = words(reset.header);
    EXPECT_EQ(reset_header[145u], 0u) << "No old strong-reference authorization may cross the semantic reset.";
    EXPECT_EQ(reset_header[104u], continued_header[104u] + 1u);
    EXPECT_EQ(reset_header[146u], 1u) << "Fresh current evidence remains eligible to seed its own new reference.";
    EXPECT_EQ(reset_header[56u], 703u);
    EXPECT_EQ(reset_header[60u], 1000703u);
    const auto packed = words(read(packed_snapshot_.resource.Get()));
    EXPECT_TRUE(std::all_of(packed.begin(), packed.end(), [](UINT x) {
      return (x & ((1u << 24u) | 16384u | 32768u)) == 0u;
    }));
    EXPECT_EQ(reset.rgb, continued.rgb);

    // An older proof carrying reset1 cannot clear a newer context's committed pair or clock.
    author_same_pixels(701u, true);
    ASSERT_TRUE(run()) << error_;
    const auto stale = aa_history_snapshot("old-context-stale-reset");
    const auto stale_header = words(stale.header);
    EXPECT_EQ(stale_header[2u], 1u);
    EXPECT_EQ(stale_header[104u], reset_header[104u]);
    EXPECT_EQ(stale_header[56u], reset_header[56u]);
    EXPECT_EQ(stale_header[60u], reset_header[60u]);
    EXPECT_NE(stale_header[103u] & 4u, 0u);
    EXPECT_EQ(stale.history, reset.history);
    EXPECT_EQ(stale.rgb, reset.rgb);
  }

}  // namespace
#endif

#ifdef _WIN32
#include "current_outline_support_cost_cases.inc"
#endif
