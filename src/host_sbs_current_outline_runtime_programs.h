#pragma once
#ifdef _WIN32
#include "host_sbs_source5_cbc5_f2_composition_gpu.h"
#include "host_sbs_current_outline_selected_matte_gpu.h"
#include "host_sbs_shader_cache.h"
#include <algorithm>
#include <array>
#include <exception>
#include <filesystem>
#include <string>
#include <string_view>
#include <stdexcept>
#include <utility>

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
inline constexpr std::string_view colour_closure="c6c8a7b9f156fe5c7b9a842195a5c485bbbdb61b2ce28bcd91d75222e2105b70";
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
inline constexpr std::string_view enclosure_closure="5b15b480ee7b497235b2e5f859bac712f430a3715e7879d120db7e6bbc41ebe3";
// One canonical ROI owner; the two historical group-local files remain only
// for compatibility with existing fixture roots. Production loads this group once.
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
inline constexpr std::string_view outline_closure="4be151b6a8ed21dfb3567004bf0f8c5b36bddddfe6b13135fa31d4e3efd731e3";
inline constexpr std::array<cache::shader_spec, 3u> stability_specs{{
  cache::shader_spec{"host_sbs_current_outline_support_band_cs.hlsl","private_cache_begin_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_outline_support_band_cs.hlsl","private_cache_validate_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_outline_support_band_cs.hlsl","private_cache_commit_main","cs_5_0"},
}};
inline constexpr std::string_view stability_closure="4802aeee8bb844f83b1cc9e6dab6a0d56cebb0505cc1dfbb5e1b2d314a882ae3";
inline constexpr std::array<cache::shader_spec, 4u> inside_specs{{
  cache::shader_spec{"host_sbs_current_inside_completion_cs.hlsl","prepare_unit_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_inside_completion_cs.hlsl","emit_init_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_inside_completion_cs.hlsl","publish_inside_main","cs_5_0"},
  cache::shader_spec{"host_sbs_current_inside_completion_cs.hlsl","finalize_main","cs_5_0"},
}};
inline constexpr std::string_view inside_closure="f35af2f4f212394a9c219471aee9115671e5adce2d11203390003e400a186a7c";
inline constexpr std::array<cache::shader_spec, 2u> selector_specs{{
  cache::shader_spec{"host_sbs_selected_matte_cs.hlsl","prepare_main","cs_5_0"},
  cache::shader_spec{"host_sbs_selected_matte_cs.hlsl","select_main","cs_5_0"},
}};
inline constexpr std::string_view selector_closure="a79bf728bc9f25c73bd745b4783664047bde3435d5f342d1b6ba21848ec40f63";
inline constexpr std::array<cache::shader_spec, 1u> completed_specs{{
  cache::shader_spec{"sbs_reprojection_v2_live_ps.hlsl","main_ps","ps_5_0"},
}};
inline constexpr std::string_view completed_closure="4535920d549f6ca6528b7a67fa20dbe6e3350993cf172ba18daeebc8a3891d7b";
inline constexpr std::array<cache::shader_spec, 1u> queued_specs{{
  cache::shader_spec{"sbs_reprojection_v2_live_ps.hlsl","main_ps","ps_5_0"},
}};
inline constexpr std::string_view queued_closure="9ccfec959a8745ac44544a5d51ec27c0deaee0eefed7fde6226940d75a0662fa";
inline std::filesystem::path completed_root(const std::filesystem::path&p){return p/"completed";}
inline std::filesystem::path queued_root(const std::filesystem::path&p){return p/"queued";}
inline std::filesystem::path selector_root(const std::filesystem::path&p){return p/"selector";}
// The existing mandatory roles share one ROI implementation. The optional cache
// owns three additional programs without changing mandatory availability.
inline constexpr std::size_t required_compute_shader_count=source5_specs.size()+colour_specs.size()+
  components_specs.size()+enclosure_specs.size()+roi_specs.size()+outline_specs.size()+inside_specs.size()+selector_specs.size();
inline constexpr std::size_t unique_compute_shader_count=required_compute_shader_count+stability_specs.size();
static_assert(required_compute_shader_count==51u&&unique_compute_shader_count==54u);
class programs_t {
  std::array<ComPtr<ID3D11ComputeShader>,unique_compute_shader_count> shaders_{};
  composition::program_t composition_{};
  selector::program_t selector_{};
  ComPtr<ID3D11Device> device_;
  bool ready_=false;
  std::string stability_error_;
public:
  bool load(ID3D11Device* device,const std::filesystem::path& parent_root,std::string& error) {
    error.clear();
    if(!device||parent_root.empty()){error="Current outline program device/root missing";return false;}
    try {
      programs_t next;next.device_=device;std::size_t at=0u;
      auto compile=[&]<std::size_t N>(const char* name,const std::array<cache::shader_spec,N>& specs,std::string_view expected){
        const auto sources=cache::snapshot_sources(parent_root/name,specs);
        if(!sources||cache::source_closure_sha256(sources)!=expected)throw std::runtime_error(std::string("Current outline source closure mismatch: ")+name);
        std::array<ID3D11ComputeShader*,N> pointers{};
        for(std::size_t i=0u;i<N;++i){const auto code=cache::get(sources,specs[i]);
          if(!code||code->empty()||at>=next.shaders_.size()||FAILED(device->CreateComputeShader(code->data(),code->size(),nullptr,next.shaders_[at].ReleaseAndGetAddressOf())))
            throw std::runtime_error(std::string("Current outline shader creation failed: ")+name+"/"+std::string(specs[i].entrypoint));
          pointers[i]=next.shaders_[at++].Get();}
        return pointers;
      };
      const auto s=compile("source5",source5_specs,source5_closure);
      next.composition_.source={s[0],s[1],s[2],s[3]};
      next.composition_.colour=compile("colour",colour_specs,colour_closure);
      next.composition_.components=compile("components",components_specs,components_closure);
      const auto e=compile("enclosure",enclosure_specs,enclosure_closure);
      std::copy_n(e.begin(),12u,next.composition_.outline.enclosure.shaders.begin());
      const auto r=compile("roi",roi_specs,roi_closure);
      std::copy_n(r.begin(),3u,next.composition_.outline.enclosure.roi_graph.begin());
      std::copy_n(r.begin(),3u,next.composition_.outline.roi_graph.begin());
      const auto f=compile("outline",outline_specs,outline_closure);
      std::copy_n(f.begin(),8u,next.composition_.outline.shaders.begin());
      std::copy_n(f.begin()+8u,7u,next.composition_.outline.band_shaders.begin());
      next.composition_.outline.inside_shaders=compile("inside",inside_specs,inside_closure);
      // Output-only testing snapshots stay null in every production program.
      next.composition_.outline.packed_temp_snapshot=nullptr;
      next.composition_.outline.passable_cache_snapshot=nullptr;
      const auto q=compile("selector",selector_specs,selector_closure);next.selector_={q[0],q[1]};
      if(at!=required_compute_shader_count)throw std::runtime_error("Current outline mandatory shader count mismatch");
      try {
        next.composition_.outline.stability_shaders=compile("outline",stability_specs,stability_closure);
        if(at!=unique_compute_shader_count)throw std::runtime_error("Current outline optional cache shader count mismatch");
      }catch(const std::exception& e){
        for(std::size_t i=required_compute_shader_count;i<next.shaders_.size();++i)next.shaders_[i].Reset();
        next.composition_.outline.stability_shaders.fill(nullptr);
        next.stability_error_=e.what();
      }
      next.ready_=true;*this=std::move(next);return true;
    }catch(const std::exception& e){error=e.what();return false;}
  }
  [[nodiscard]] bool ready()const noexcept{return ready_;}
  [[nodiscard]] bool stability_ready()const noexcept{
    return std::all_of(composition_.outline.stability_shaders.begin(),composition_.outline.stability_shaders.end(),[](auto* s){return s!=nullptr;});
  }
  [[nodiscard]] const std::string& stability_error()const noexcept{return stability_error_;}
  [[nodiscard]] composition::program_t composition_program()const{return composition_;}
  [[nodiscard]] selector::program_t selector_program()const{return selector_;}
};
} // namespace models::current_outline_runtime
#endif
