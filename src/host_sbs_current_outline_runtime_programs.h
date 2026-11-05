#pragma once
#ifdef _WIN32
#include "host_sbs_source5_cbc5_f2_composition_gpu.h"
#include "host_sbs_current_outline_selected_matte_gpu.h"
#include "host_sbs_shader_cache.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <exception>
#include <filesystem>
#include <string>
#include <string_view>
#include <stdexcept>
#include <span>
#include <thread>
#include <utility>
#include <vector>

// Shader ownership/loading only. Availability is not GPU completion or draw authority.
namespace models::current_outline_runtime {
namespace cache=host_sbs_shader_cache;
namespace composition=source5_cbc5_f2_composition_gpu_v1;
namespace selector=research_current_outline_selected_matte_v2;
using Microsoft::WRL::ComPtr;
inline constexpr std::array<cache::shader_spec, 4u> source5_specs{{
  cache::shader_spec{"host_sbs_display_snapshot_rgb8_analysis_cs.hlsl","prepare_main","cs_5_0"},
  cache::shader_spec{"host_sbs_display_snapshot_rgb8_analysis_cs.hlsl","copy_main","cs_5_0"},
  cache::shader_spec{"host_sbs_display_snapshot_rgb8_analysis_cs.hlsl","finalize_main","cs_5_0"},
  cache::shader_spec{"host_sbs_display_snapshot_rgb8_analysis_cs.hlsl","envelope_main","cs_5_0"},
}};
inline constexpr std::string_view source5_closure="291ec723c8259b864e9889bbb8c9fd8b182da6795bd272a0fa97d8edae29f8bb";
inline constexpr std::array<cache::shader_spec, 5u> colour_specs{{
  cache::shader_spec{"host_sbs_subtitle_inner_colour_cs.hlsl","palette_main","cs_5_0"},
  cache::shader_spec{"host_sbs_subtitle_inner_colour_cs.hlsl","emit_main","cs_5_0"},
  cache::shader_spec{"host_sbs_subtitle_inner_colour_cs.hlsl","clear_main","cs_5_0"},
  cache::shader_spec{"host_sbs_subtitle_inner_colour_cs.hlsl","accumulate_main","cs_5_0"},
  cache::shader_spec{"host_sbs_subtitle_inner_colour_cs.hlsl","validate_main","cs_5_0"},
}};
inline constexpr std::string_view colour_closure="fff7e514f18ca71b653ac5f047fcb3716a087667d16f3fa1a993c055e9056652";
inline constexpr std::array<cache::shader_spec, 6u> components_specs{{
  cache::shader_spec{"host_sbs_subtitle_seeded_components_cs.hlsl","init_main","cs_5_0"},
  cache::shader_spec{"host_sbs_subtitle_seeded_components_cs.hlsl","union_main","cs_5_0"},
  cache::shader_spec{"host_sbs_subtitle_seeded_components_cs.hlsl","reduce_main","cs_5_0"},
  cache::shader_spec{"host_sbs_subtitle_seeded_components_cs.hlsl","resolve_main","cs_5_0"},
  cache::shader_spec{"host_sbs_subtitle_seeded_components_cs.hlsl","dilate_main","cs_5_0"},
  cache::shader_spec{"host_sbs_subtitle_seeded_components_cs.hlsl","dilate_box_main","cs_5_0"},
}};
inline constexpr std::string_view components_closure="3c5884de250262680dff4fb80c67e30335217af90d237fc7b531131ba9b967d1";
inline constexpr std::array<cache::shader_spec, 12u> enclosure_specs{{
  cache::shader_spec{"host_sbs_current_black_enclosure_cs.hlsl","begin_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_black_enclosure_cs.hlsl","clear_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_black_enclosure_cs.hlsl","validate_encoding_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_black_enclosure_cs.hlsl","unit_meta_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_black_enclosure_cs.hlsl","collect_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_black_enclosure_cs.hlsl","clear_scratch_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_black_enclosure_cs.hlsl","prepare_unit_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_black_enclosure_cs.hlsl","emit_exterior_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_black_enclosure_cs.hlsl","resolve_exterior_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_black_enclosure_cs.hlsl","count_enclosed_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_black_enclosure_cs.hlsl","select_unit_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_black_enclosure_cs.hlsl","publish_main","cs_5_0"},
}};
inline constexpr std::string_view enclosure_closure="0aee4eb848386d892fb972f5ecdae0dd0ea97a10c5b91f775b9c60257acb6c36";
// One canonical ROI owner; the historical enclosure-local file remains for its independently
// pinned enclosure/ring fixtures. Production loads this group once.
inline constexpr std::array<cache::shader_spec, 3u> roi_specs{{
  cache::shader_spec{"host_sbs_current_components_roi_cs.hlsl","init_roi_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_components_roi_cs.hlsl","union_roi_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_components_roi_cs.hlsl","reduce_roi_main","cs_5_0"},
}};
inline constexpr std::string_view roi_closure="32c11c351c23f617ec4d5ca1d6ce19bdc24c01fd47ff9e7ebf25d2300ccb9417";
inline constexpr std::array<cache::shader_spec, 15u> outline_specs{{
  cache::shader_spec{"host_sbs_current_outline_support_raw_cs.hlsl","begin_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_outline_support_raw_cs.hlsl","reset_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_outline_support_raw_cs.hlsl","prepare_unit_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_outline_support_raw_cs.hlsl","emit_passable_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_outline_support_raw_cs.hlsl","resolve_inner_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_outline_support_raw_cs.hlsl","emit_black_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_outline_support_raw_cs.hlsl","append_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_outline_support_raw_cs.hlsl","publish_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_outline_support_band_cs.hlsl","prepare_profiles_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_outline_support_band_cs.hlsl","profile_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_outline_support_band_cs.hlsl","calibrate_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_outline_support_band_cs.hlsl","grow_tile_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_outline_support_band_cs.hlsl","publish_final_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_outline_support_band_cs.hlsl","aggregate_support_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_outline_support_band_cs.hlsl","publish_closure_main","cs_5_0"},
}};
inline constexpr std::string_view outline_closure="168e96aa498fc93d01d9fb2f159b05997abb50a06d24d86909e85e037cf8d5cd";
inline constexpr std::array<cache::shader_spec, 3u> stability_specs{{
  cache::shader_spec{"host_sbs_current_outline_support_band_cs.hlsl","private_cache_begin_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_outline_support_band_cs.hlsl","private_cache_validate_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_outline_support_band_cs.hlsl","private_cache_commit_main","cs_5_0"},
}};
inline constexpr std::string_view stability_closure="3dc538f969b95ce0b4bbbc54f786b7db449135b95139178fa057de382cf1ff2a";
inline constexpr std::array<cache::shader_spec, 9u> hole_specs{{
  cache::shader_spec{"host_sbs_current_mask_holes_cs.hlsl","clear_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_mask_holes_cs.hlsl","capture_unit_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_mask_holes_cs.hlsl","prepare_holes_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_mask_holes_cs.hlsl","freeze_final_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_mask_holes_cs.hlsl","init_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_mask_holes_cs.hlsl","union_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_mask_holes_cs.hlsl","reduce_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_mask_holes_cs.hlsl","publish_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_mask_holes_cs.hlsl","counter_main","cs_5_0"},
}};
inline constexpr std::string_view hole_closure="f9bc3edaabe2680d655ba91ec834e6ab9fdd17524a24fc024963a5f67674640b";
inline constexpr std::array<cache::shader_spec, 3u> coverage_specs{{
  cache::shader_spec{"host_sbs_current_stroke_coverage_cs.hlsl","clear_offers_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_stroke_coverage_cs.hlsl","prepare_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_stroke_coverage_cs.hlsl","support_main","cs_5_0"},
}};
inline constexpr std::string_view coverage_closure="185fbaed311da1c2f55ad6dfa8fbdf922da8a973e32926cf8ab8032819eb8f8d";
inline constexpr std::array<cache::shader_spec, 1u> foreground_edge_specs{{
  cache::shader_spec{"host_sbs_current_foreground_edge_cs.hlsl","foreground_edge_main","cs_5_0"},
}};
inline constexpr std::string_view foreground_edge_closure="011e521ceca537b166a4a353fa05c08bb061395e9ef8af5253caa96f8bb493c4";
inline constexpr std::array<cache::shader_spec, 4u> inside_specs{{
  cache::shader_spec{"host_sbs_current_inside_completion_cs.hlsl","prepare_unit_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_inside_completion_cs.hlsl","emit_init_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_inside_completion_cs.hlsl","publish_inside_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_inside_completion_cs.hlsl","finalize_main","cs_5_0"},
}};
inline constexpr std::string_view inside_closure="0a97d8ec59d95af324d41d268634c68431f76fb3e1af924d8dfd14542c468672";
inline constexpr std::array<cache::shader_spec, 2u> selector_specs{{
  cache::shader_spec{"host_sbs_selected_matte_cs.hlsl","prepare_main","cs_5_0"},
  cache::shader_spec{"host_sbs_selected_matte_cs.hlsl","select_main","cs_5_0"},
}};
inline constexpr std::string_view selector_closure="a79bf728bc9f25c73bd745b4783664047bde3435d5f342d1b6ba21848ec40f63";
inline std::filesystem::path selector_root(const std::filesystem::path&p){return p/"selector";}

// The mandatory roles share one ROI implementation. Each optional role owns a fixed,
// disjoint segment so a failed optional load cannot discard another role's handles.
inline constexpr std::size_t required_compute_shader_count=source5_specs.size()+colour_specs.size()+
  components_specs.size()+enclosure_specs.size()+roi_specs.size()+outline_specs.size()+inside_specs.size()+selector_specs.size();
inline constexpr std::size_t stability_compute_shader_offset=required_compute_shader_count;
inline constexpr std::size_t hole_compute_shader_offset=stability_compute_shader_offset+stability_specs.size();
inline constexpr std::size_t coverage_compute_shader_offset=hole_compute_shader_offset+hole_specs.size();
inline constexpr std::size_t foreground_edge_compute_shader_offset=coverage_compute_shader_offset+coverage_specs.size();
inline constexpr std::size_t unique_compute_shader_count=foreground_edge_compute_shader_offset+foreground_edge_specs.size();
static_assert(required_compute_shader_count==51u&&hole_compute_shader_offset==54u&&
              coverage_compute_shader_offset==63u&&foreground_edge_compute_shader_offset==66u&&unique_compute_shader_count==67u);

namespace detail {
  inline constexpr std::size_t maximum_bytecode_workers = 4u;

  struct bytecode_request_t {
    cache::source_snapshot_t sources;
    cache::shader_spec spec;
    cache::bytecode_t code;
    std::exception_ptr error;
  };

  struct bytecode_group_t {
    std::string name;
    std::size_t first = 0u;
    std::size_t count = 0u;
  };

  inline bytecode_group_t authenticate_group(
    const std::string_view name,
    const std::filesystem::path &path,
    const std::span<const cache::shader_spec> specs,
    const std::string_view expected,
    std::vector<bytecode_request_t> &requests
  ) {
    const auto sources = cache::snapshot_sources(path, specs);
    if (!sources || expected.size() != 64u || cache::source_closure_sha256(sources) != expected) {
      throw std::runtime_error("Current outline source closure mismatch: " + std::string(name));
    }
    const bytecode_group_t group {std::string(name), requests.size(), specs.size()};
    for (const auto &spec : specs) {
      requests.push_back({sources, spec, {}, {}});
    }
    return group;
  }

  // Every input snapshot is authenticated before this function is called. Only CPU bytecode
  // resolution runs in the bounded workers; device creation remains on the caller after join.
  // Per-request errors keep an optional role from invalidating mandatory prepared bytecode.
  template<class Resolve>
  void prepare_bytecodes(const std::span<bytecode_request_t> requests, Resolve resolve) {
    for (const auto &request : requests) {
      if (!request.sources) {
        throw std::invalid_argument("Current outline bytecode snapshot unavailable");
      }
    }
    for (auto &request : requests) {
      request.code.reset();
      request.error = {};
    }
    std::atomic_size_t next {0u};
    const auto run = [&]() noexcept {
      for (;;) {
        const auto index = next.fetch_add(1u, std::memory_order_relaxed);
        if (index >= requests.size()) {
          return;
        }
        auto &request = requests[index];
        try {
          request.code = resolve(request.sources, request.spec);
        } catch (...) {
          request.error = std::current_exception();
        }
      }
    };
    // The already tracked build worker participates as worker one. Scoped jthreads join on
    // success and exceptions, including a failure to create a later worker thread.
    std::array<std::jthread, maximum_bytecode_workers - 1u> workers;
    for (std::size_t i = 0u; i < workers.size() && i + 1u < requests.size(); ++i) {
      workers[i] = std::jthread(run);
    }
    run();
    for (auto &worker : workers) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  inline void prepare_bytecodes(const std::span<bytecode_request_t> requests) {
    prepare_bytecodes(requests, [](const auto &sources, const auto &spec) {
      return cache::get(sources, spec);
    });
  }

  inline const cache::bytecode_t &prepared_bytecode(const bytecode_request_t &request, const std::string_view group) {
    const auto role = std::string(group) + "/" + std::string(request.spec.entrypoint);
    if (request.error) {
      try {
        std::rethrow_exception(request.error);
      } catch (const std::exception &error) {
        throw std::runtime_error("Current outline bytecode preparation failed: " + role + ": " + error.what());
      } catch (...) {
        throw std::runtime_error("Current outline bytecode preparation failed: " + role);
      }
    }
    if (!request.code || request.code->empty()) {
      throw std::runtime_error("Current outline bytecode unavailable: " + role);
    }
    return request.code;
  }

  template<std::size_t N, std::size_t Capacity>
  std::array<ID3D11ComputeShader *, N> create_compute_group(
    ID3D11Device *device,
    const bytecode_group_t &group,
    const std::span<const bytecode_request_t> bytecodes,
    std::array<ComPtr<ID3D11ComputeShader>, Capacity> &owned,
    std::size_t &at
  ) {
    if (!device || group.count != N || group.first > bytecodes.size() || N > bytecodes.size() - group.first) {
      throw std::runtime_error("Current outline prepared group mismatch: " + group.name);
    }
    std::array<ID3D11ComputeShader *, N> pointers {};
    for (std::size_t i = 0u; i < N; ++i) {
      const auto &prepared = bytecodes[group.first + i];
      const auto &code = prepared_bytecode(prepared, group.name);
      if (at >= owned.size() || FAILED(device->CreateComputeShader(code->data(), code->size(), nullptr, owned[at].ReleaseAndGetAddressOf()))) {
        throw std::runtime_error("Current outline shader creation failed: " + group.name + "/" + std::string(prepared.spec.entrypoint));
      }
      pointers[i] = owned[at++].Get();
    }
    return pointers;
  }

  template<std::size_t N, std::size_t Capacity>
  void create_optional_compute_group(
    ID3D11Device *device,
    const bytecode_group_t &group,
    const std::span<const bytecode_request_t> bytecodes,
    std::array<ComPtr<ID3D11ComputeShader>, Capacity> &owned,
    const std::size_t offset,
    std::array<ID3D11ComputeShader *, N> &pointers,
    std::string &error
  ) {
    if (offset > Capacity || N > Capacity - offset) {
      throw std::invalid_argument("Current outline optional storage range invalid");
    }
    try {
      if (!error.empty()) {
        throw std::runtime_error(error);
      }
      std::size_t at = offset;
      pointers = create_compute_group<N>(device, group, bytecodes, owned, at);
      if (at != offset + N) {
        throw std::runtime_error("Current outline optional shader count mismatch: " + group.name);
      }
    } catch (const std::exception &failure) {
      for (std::size_t i = offset; i < offset + N; ++i) {
        owned[i].Reset();
      }
      pointers.fill(nullptr);
      error = failure.what();
    }
  }
}  // namespace detail

class programs_t {
  std::array<ComPtr<ID3D11ComputeShader>, unique_compute_shader_count> shaders_ {};
  composition::program_t composition_ {};
  selector::program_t selector_ {};
  ComPtr<ID3D11Device> device_;
  bool ready_ = false;
  std::string stability_error_;
  std::string hole_error_;
  std::string coverage_error_;
  std::string foreground_edge_error_;

public:
  bool load(ID3D11Device *device, const std::filesystem::path &parent_root, std::string &error) {
    error.clear();
    if (!device || parent_root.empty()) {
      error = "Current outline program device/root missing";
      return false;
    }
    try {
      programs_t next;
      next.device_ = device;
      std::size_t at = 0u;
      std::vector<detail::bytecode_request_t> bytecodes;
      bytecodes.reserve(unique_compute_shader_count);
      const auto authenticate = [&](const char *name, const auto &specs, std::string_view expected) {
        return detail::authenticate_group(name, parent_root / name, specs, expected, bytecodes);
      };
      // Authenticate every mandatory closure before any potentially expensive cache lookup.
      const auto source = authenticate("source5", source5_specs, source5_closure);
      const auto colour = authenticate("colour", colour_specs, colour_closure);
      const auto components = authenticate("components", components_specs, components_closure);
      const auto enclosure = authenticate("enclosure", enclosure_specs, enclosure_closure);
      const auto roi = authenticate("roi", roi_specs, roi_closure);
      const auto outline = authenticate("outline", outline_specs, outline_closure);
      const auto inside = authenticate("inside", inside_specs, inside_closure);
      const auto selector = authenticate("selector", selector_specs, selector_closure);
      detail::bytecode_group_t stability;
      try {
        stability = authenticate("outline", stability_specs, stability_closure);
      } catch (const std::exception &e) {
        next.stability_error_ = e.what();
      }
      detail::bytecode_group_t hole;
      try {
        hole = authenticate("hole", hole_specs, hole_closure);
      } catch (const std::exception &e) {
        next.hole_error_ = e.what();
      }
      detail::bytecode_group_t coverage;
      try {
        coverage = detail::authenticate_group("coverage", parent_root / "hole", coverage_specs, coverage_closure, bytecodes);
      } catch (const std::exception &e) {
        next.coverage_error_ = e.what();
      }
      detail::bytecode_group_t foreground_edge;
      try {
        foreground_edge = detail::authenticate_group("foreground_edge", parent_root / "hole", foreground_edge_specs, foreground_edge_closure, bytecodes);
      } catch (const std::exception &e) {
        next.foreground_edge_error_ = e.what();
      }
      detail::prepare_bytecodes(bytecodes);
      auto compile = [&]<std::size_t N>(const detail::bytecode_group_t &group, const std::array<cache::shader_spec, N> &) {
        return detail::create_compute_group<N>(device, group, bytecodes, next.shaders_, at);
      };
      const auto s = compile(source, source5_specs);
      next.composition_.source = {s[0], s[1], s[2], s[3]};
      next.composition_.colour = compile(colour, colour_specs);
      next.composition_.components = compile(components, components_specs);
      const auto e = compile(enclosure, enclosure_specs);
      std::copy_n(e.begin(), 12u, next.composition_.outline.enclosure.shaders.begin());
      const auto r = compile(roi, roi_specs);
      std::copy_n(r.begin(), 3u, next.composition_.outline.enclosure.roi_graph.begin());
      std::copy_n(r.begin(), 3u, next.composition_.outline.roi_graph.begin());
      const auto f = compile(outline, outline_specs);
      std::copy_n(f.begin(), 8u, next.composition_.outline.shaders.begin());
      std::copy_n(f.begin() + 8u, 7u, next.composition_.outline.band_shaders.begin());
      next.composition_.outline.inside_shaders = compile(inside, inside_specs);
      // Output-only testing snapshots stay null in every production program.
      next.composition_.outline.packed_temp_snapshot = nullptr;
      next.composition_.outline.passable_cache_snapshot = nullptr;
      const auto q = compile(selector, selector_specs);
      next.selector_ = {q[0], q[1]};
      if (at != required_compute_shader_count) {
        throw std::runtime_error("Current outline mandatory shader count mismatch");
      }
      detail::create_optional_compute_group(device, stability, bytecodes, next.shaders_,
        stability_compute_shader_offset, next.composition_.outline.stability_shaders, next.stability_error_);
      detail::create_optional_compute_group(device, hole, bytecodes, next.shaders_,
        hole_compute_shader_offset, next.composition_.outline.hole_shaders, next.hole_error_);
      detail::create_optional_compute_group(device, coverage, bytecodes, next.shaders_,
        coverage_compute_shader_offset, next.composition_.outline.coverage_shaders, next.coverage_error_);
      detail::create_optional_compute_group(device, foreground_edge, bytecodes, next.shaders_,
        foreground_edge_compute_shader_offset, next.composition_.outline.foreground_edge_shaders, next.foreground_edge_error_);
      next.ready_ = true;
      *this = std::move(next);
      return true;
    } catch (const std::exception &e) {
      error = e.what();
      return false;
    }
  }

  [[nodiscard]] bool ready() const noexcept {
    return ready_;
  }

  [[nodiscard]] bool stability_ready() const noexcept {
    return std::all_of(composition_.outline.stability_shaders.begin(), composition_.outline.stability_shaders.end(), [](auto *s) {
      return s != nullptr;
    });
  }

  [[nodiscard]] const std::string &stability_error() const noexcept {
    return stability_error_;
  }

  [[nodiscard]] bool hole_ready() const noexcept {
    return std::all_of(composition_.outline.hole_shaders.begin(), composition_.outline.hole_shaders.end(), [](auto *s) {
      return s != nullptr;
    });
  }

  [[nodiscard]] const std::string &hole_error() const noexcept {
    return hole_error_;
  }

  [[nodiscard]] bool coverage_ready() const noexcept {
    return std::all_of(composition_.outline.coverage_shaders.begin(), composition_.outline.coverage_shaders.end(), [](auto *s) {
      return s != nullptr;
    });
  }

  [[nodiscard]] const std::string &coverage_error() const noexcept {
    return coverage_error_;
  }

  [[nodiscard]] bool foreground_edge_ready() const noexcept {
    return std::all_of(composition_.outline.foreground_edge_shaders.begin(), composition_.outline.foreground_edge_shaders.end(), [](auto *s) { return s != nullptr; });
  }

  [[nodiscard]] const std::string &foreground_edge_error() const noexcept { return foreground_edge_error_; }

  [[nodiscard]] composition::program_t composition_program() const {
    return composition_;
  }

  [[nodiscard]] selector::program_t selector_program() const {
    return selector_;
  }
};
}  // namespace models::current_outline_runtime
#endif
