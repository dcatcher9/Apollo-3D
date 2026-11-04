#pragma once

// Current CFB7 outline pipeline: bounded known-black support, followed by current-frame
// foreground completion. Owns its command recording and resources; selection/drawing
// authority remains with the separate current-outline selector.
#ifdef _WIN32
  #include "host_sbs_current_black_ring_gpu.h"
  #include "host_sbs_current_mask_holes_gpu.h"
  #include "host_sbs_compute_program_lease.h"
  #include "host_sbs_current_outline_stage_observer.h"

  #include <algorithm>
  #include <array>
  #include <cstdint>
  #include <exception>
  #include <limits>
  #include <string>
  #include <utility>
  #include <vector>

namespace models::host_sbs_current_outline_support_closure_gpu_v2 {
  using Microsoft::WRL::ComPtr;
  inline constexpr UINT calibration_word_count = 1024u;
  inline constexpr UINT histogram_word_offset = 1024u, histogram_word_count = 65536u;
  inline constexpr UINT band_word_offset = 66560u, band_word_count = 2048u;
  inline constexpr UINT data_word_count = 68608u;
  inline constexpr UINT maximum_bands = 64u, support_extent_bins = 128u, maximum_support_extent = 16u;
  // Protocol identifiers, not tuning controls. Layout/counts are unchanged.
  inline constexpr UINT calibration_tag = 0x37424643u, calibration_schema = 7u;
  inline constexpr const char *profile_measurement = "BODY_to_last_physical_black_end_distance_j_minus_1";
  namespace enclosure = host_sbs_current_black_enclosure_gpu_v1;
  namespace baseline = enclosure::baseline;
  namespace holes = host_sbs_current_mask_holes_gpu;

  struct program_t {
    enclosure::program_t enclosure;
    // begin, reset, prepare-unit, emit-passable, resolve-inner, emit-black, append, publish.
    std::array<ID3D11ComputeShader *, 8u> shaders {};
    // Exact original bounded init/union/reduce, run twice with different current predicates.
    std::array<ID3D11ComputeShader *, 3u> roi_graph {};
    // prepare-profiles, profile, calibrate, grow-tile, final-publish, support-reduce, closure-publish.
    std::array<ID3D11ComputeShader *, 7u> band_shaders {};
    // Optional all-or-none current-mask completion after the original final mask.
    holes::programs_t hole_shaders {};
    holes::coverage_programs_t coverage_shaders {};
    // Early current Inside: prepare, combined init, publish; final OR remains late.
    std::array<ID3D11ComputeShader *, 4u> inside_shaders {};
    // PRIVATE BBSC1 optional all-or-none begin/validate/commit.
    std::array<ID3D11ComputeShader *, 3u> stability_shaders {};
    bool coverage_diagnostics = false;
    // TEST-ONLY output destinations. Never detector inputs. Default null costs no copies/planes.
    ID3D11Texture2D *packed_temp_snapshot = nullptr;
    ID3D11Texture2D *passable_cache_snapshot = nullptr;
    // TEST-ONLY fresh-active diagnostics, copied after legacy final and before Inside overwrites scratch.
    ID3D11Texture2D *pre_inside_final_snapshot = nullptr;
    ID3D11Buffer *raw_parent_snapshot = nullptr;
    ID3D11Buffer *raw_flags_snapshot = nullptr;
    // CPU submission observer only; never recorded or included in the cache signature.
    const current_outline_stage_observer::observer_t *stage_observer = nullptr;
  };

  struct mutable_views_t {
    UINT width = 0u, height = 0u;
    ComPtr<ID3D11ShaderResourceView> body, matte, calibration_data, passable_root_cache;
    ComPtr<ID3D11Buffer> hook_constants, unit_arguments, native_arguments;
    ComPtr<ID3D11ShaderResourceView> inside_bits, family_models;
    ComPtr<ID3D11Buffer> inside_arguments;
    ComPtr<ID3D11ShaderResourceView> stability_image, stability_header;
  };

  using diagnostics_t = enclosure::diagnostics_t;

  class resources_t {
    struct texture_t {
      ComPtr<ID3D11Texture2D> resource;
      ComPtr<ID3D11ShaderResourceView> srv;
      ComPtr<ID3D11UnorderedAccessView> uav;
    };

    struct buffer_t {
      ComPtr<ID3D11Buffer> resource;
      ComPtr<ID3D11ShaderResourceView> srv;
      ComPtr<ID3D11UnorderedAccessView> uav;
    };

    struct saved_cs_t {
      ID3D11DeviceContext *c;
      std::array<ComPtr<ID3D11ShaderResourceView>, 16u> s;
      std::array<ComPtr<ID3D11UnorderedAccessView>, 8u> u;
      std::array<ComPtr<ID3D11Buffer>, 4u> b;
      ComPtr<ID3D11ComputeShader> shader;
      std::array<ComPtr<ID3D11ClassInstance>, 256u> instances;
      UINT count = 256u;

      explicit saved_cs_t(ID3D11DeviceContext *context):
          c(context) {
        std::array<ID3D11ShaderResourceView *, 16u> ss {};
        std::array<ID3D11UnorderedAccessView *, 8u> uu {};
        std::array<ID3D11Buffer *, 4u> bb {};
        std::array<ID3D11ClassInstance *, 256u> ii {};
        ID3D11ComputeShader *cs = nullptr;
        c->CSGetShaderResources(0u, 16u, ss.data());
        c->CSGetUnorderedAccessViews(0u, 8u, uu.data());
        c->CSGetConstantBuffers(0u, 4u, bb.data());
        c->CSGetShader(&cs, ii.data(), &count);
        shader.Attach(cs);
        for (UINT n = 0; n < 16u; ++n) {
          s[n].Attach(ss[n]);
        }
        for (UINT n = 0; n < 8u; ++n) {
          u[n].Attach(uu[n]);
        }
        for (UINT n = 0; n < 4u; ++n) {
          b[n].Attach(bb[n]);
        }
        for (UINT n = 0; n < count; ++n) {
          instances[n].Attach(ii[n]);
        }
      }

      saved_cs_t(const saved_cs_t &) = delete;
      saved_cs_t &operator=(const saved_cs_t &) = delete;

      ~saved_cs_t() {
        unbind(c);
        std::array<ID3D11ShaderResourceView *, 16u> ss {};
        std::array<ID3D11UnorderedAccessView *, 8u> uu {};
        std::array<ID3D11Buffer *, 4u> bb {};
        std::array<ID3D11ClassInstance *, 256u> ii {};
        for (UINT n = 0; n < 16u; ++n) {
          ss[n] = s[n].Get();
        }
        for (UINT n = 0; n < 8u; ++n) {
          uu[n] = u[n].Get();
        }
        for (UINT n = 0; n < 4u; ++n) {
          bb[n] = b[n].Get();
        }
        for (UINT n = 0; n < count; ++n) {
          ii[n] = instances[n].Get();
        }
        c->CSSetUnorderedAccessViews(0u, 8u, uu.data(), nullptr);
        c->CSSetShaderResources(0u, 16u, ss.data());
        c->CSSetConstantBuffers(0u, 4u, bb.data());
        c->CSSetShader(shader.Get(), count ? ii.data() : nullptr, count);
      }
    };

    struct signature_t {
      std::array<ComPtr<IUnknown>, 160u> values;
      UINT count = 0u, root_offset = 0u;

      bool matches(const std::array<IUnknown *, 160u> &v, UINT n, UINT offset) const noexcept {
        if (n > values.size() || count != n || root_offset != offset) {
          return false;
        }
        for (UINT i = 0; i < n; ++i) {
          if (values[i].Get() != v[i]) {
            return false;
          }
        }
        return true;
      }

      bool retain(const std::array<IUnknown *, 160u> &v, UINT n, UINT offset) {
        if (n > values.size()) return false;
        *this = signature_t {};
        count = n;
        root_offset = offset;
        for (UINT i = 0; i < n; ++i) {
          values[i] = v[i];
        }
        return true;
      }
    };

  public:
    resources_t() = default;
    resources_t(const resources_t &) = delete;
    resources_t &operator=(const resources_t &) = delete;
    resources_t(resources_t &&) = default;
    resources_t &operator=(resources_t &&) = default;

    [[nodiscard]] ID3D11ShaderResourceView *mask_hole_proof() const noexcept { return holes_.proof(); }
    [[nodiscard]] ID3D11ShaderResourceView *mask_hole_parents() const noexcept { return holes_.parents(); }
    [[nodiscard]] ID3D11ShaderResourceView *mask_hole_flags() const noexcept { return holes_.flags(); }
    [[nodiscard]] bool mask_holes_enabled() const noexcept { return holes_enabled_; }
    [[nodiscard]] bool stroke_coverage_diagnostics_enabled() const noexcept { return holes_.coverage_diagnostics_enabled(); }
    [[nodiscard]] bool stroke_coverage_enabled() const noexcept { return coverage_enabled_; }
    [[nodiscard]] ID3D11ShaderResourceView *stroke_coverage_offers() const noexcept { return holes_.coverage_offers(); }

    void set_recorded_execution(bool enabled) noexcept {
      recorded_ = enabled;
      enclosure_recorded_ = enabled;
      enclosure_.set_recorded_execution(enabled);
    }

    void set_enclosure_recorded_execution(bool enabled) noexcept {
      enclosure_recorded_ = enabled;
      enclosure_.set_recorded_execution(enabled);
    }

    void set_baseline_recorded_execution(bool enabled) noexcept {
      // Controls the null-program baseline fallback, not the enabled CBC list.
      baseline_recorded_ = enabled;
      enclosure_.set_baseline_recorded_execution(enabled);
    }

    [[nodiscard]] diagnostics_t diagnostics() const noexcept {
      return diagnostics_;
    }

    [[nodiscard]] enclosure::diagnostics_t parent_diagnostics() const noexcept {
      return enclosure_.diagnostics();
    }

    [[nodiscard]] bool last_commands_enabled() const noexcept {
      return last_enabled_;
    }

    [[nodiscard]] enclosure::mutable_views_t enclosure_views() const {
      return enclosure_.mutable_views();
    }

    [[nodiscard]] baseline::mutable_views_t baseline_views() const {
      return enclosure_.baseline_views();
    }

    [[nodiscard]] host_sbs_current_black_ring_gpu_v1::mutable_views_t raw_views() const {
      return {width_, height_, enclosure_views_.body, matte_.srv, meta_.srv, inner_.srv, hook_cb_, unit_args_.resource, native_args_};
    }

    // Private read-only diagnostics; no public event/view aggregate layout change.
    [[nodiscard]] ComPtr<ID3D11ShaderResourceView> aa_history_view() const {
      return stability_aa_history_.srv;
    }

    [[nodiscard]] ComPtr<ID3D11ShaderResourceView> weak_aa_state_view() const {
      return weak_aa_state_.srv;
    }

    [[nodiscard]] mutable_views_t mutable_views() const {
      return {width_, height_, enclosure_views_.body, colour_first_.srv, profile_data_.srv, root_cache_.srv, hook_cb_, unit_args_.resource, native_args_, inside_bits_.srv, enclosure_views_.family_models, inside_args_.resource, stability_image_.srv, stability_header_.srv};
    }

    // The serialized caller must retire the preceding publication's consumers first. Revoke
    // semantic-context authority without dropping allocations, source pixels or clock high-water.
    bool invalidate_subtitle_history(ID3D11DeviceContext *context) {
      if (!stability_header_.resource && !weak_aa_state_.resource) {
        last_enabled_ = false;
        return true;
      }
      if (!context || context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE || !same_device(context)) {
        return false;
      }
      const UINT zero = 0u;
      const UINT invalid_tuple = std::numeric_limits<UINT>::max();
      if (stability_header_.resource) {
        // Strong-valid, transient offers/plans and legacy-global/AA permission only.
        // LastSeen frame56/57, clock60/61 and seen107 deliberately survive.
        for (UINT word : {2u, 5u, 7u, 102u, 148u, 149u, 152u, 153u}) {
          const D3D11_BOX box {word, 0u, 0u, word + 1u, 1u, 1u};
          context->UpdateSubresource(stability_header_.resource.Get(), 0u, &box, &zero, sizeof(zero), 0u);
        }
        // Saved CB19 is reserved zero in every valid tuple. Break duplicate matching while
        // retaining the independent material-clock guard used to reject stale observations.
        const D3D11_BOX box {63u, 0u, 0u, 64u, 1u, 1u};
        context->UpdateSubresource(stability_header_.resource.Get(), 0u, &box, &invalid_tuple, sizeof(invalid_tuple), 0u);
      }
      if (weak_aa_state_.resource) {
        // Revoke the RGB anchor, current plan, episode and qualifications. Preserve seen11,
        // frame28/29, clock32/33 and every source/history pixel below the header.
        for (UINT word : {2u, 4u, 5u, 6u, 232u, 233u, 234u, 235u}) {
          const D3D11_BOX box {word * 4u, 0u, 0u, (word + 1u) * 4u, 1u, 1u};
          context->UpdateSubresource(weak_aa_state_.resource.Get(), 0u, &box, &zero, 0u, 0u);
        }
        const D3D11_BOX box {35u * 4u, 0u, 0u, 36u * 4u, 1u, 1u};
        context->UpdateSubresource(weak_aa_state_.resource.Get(), 0u, &box, &invalid_tuple, 0u, 0u);
      }
      last_enabled_ = false;
      return true;
    }

    bool ensure(ID3D11Device *device, UINT width, UINT height, std::string &error) {
      error.clear();
      constexpr UINT maximum = depth_coordinate_v2::subtitle_contour_source_max_extent;
      const std::uint64_t n = std::uint64_t {width} * height;
      if (!device || !width || !height || width > maximum || height > maximum || n + 1u > std::numeric_limits<UINT>::max() / 4u) {
        error = "Invalid current-outline-support-closure device/native shape.";
        return false;
      }
      if (ready_ && device_.Get() == device && width_ == width && height_ == height) {
        return true;
      }
      resources_t next;
      next.recorded_ = recorded_;
      next.enclosure_recorded_ = enclosure_recorded_;
      next.baseline_recorded_ = baseline_recorded_;
      next.diagnostics_ = diagnostics_;
      next.enclosure_.set_recorded_execution(enclosure_recorded_);
      next.enclosure_.set_baseline_recorded_execution(baseline_recorded_);
      if (device_) {
        ++next.diagnostics_.allocation_replacements;
      }
      next.device_ = device;
      next.width_ = width;
      next.height_ = height;
      next.pixels_ = static_cast<UINT>(n);
      try {
        if (!next.enclosure_.ensure(device, width, height, error) || !next.allocate(error)) {
          return false;
        }
      } catch (const std::exception &e) {
        error = std::string {"Could not allocate current-outline-support-closure storage: "} + e.what();
        return false;
      }
      next.ready_ = true;
      *this = std::move(next);
      return true;
    }

    // Serialized immediate owner leases all current sources and mutable scratch until ordered
    // completion. Scratch must be unbound outside CS0..15/UAV0..7. true = commands submitted only.
    // Root0 holds raw/final M, Inner, cache and metadata. Snapshot copies are output-only, not fresh root0 observations.
    bool dispatch(ID3D11DeviceContext *context, const std::array<ID3D11ComputeShader *, 5u> &colour, const std::array<ID3D11ComputeShader *, 6u> &graph, ID3D11Buffer *constants96, ID3D11ShaderResourceView *rgb, ID3D11ShaderResourceView *probability, ID3D11ShaderResourceView *support, ID3D11ShaderResourceView *ocr, ID3D11Buffer *root, UINT root_offset, std::string &error, const program_t *program = nullptr) {
      error.clear();
      last_enabled_ = false;
      if (!program) {
        invalidate_weak_aa();
        stability_needs_reset_ = true;
        return enclosure_.dispatch(context, colour, graph, constants96, rgb, probability, support, ocr, root, root_offset, error);
      }
      if (!ready_ || !context || context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE || !same_device(context)) {
        return fail(error, "Enabled outline-support-closure requires prepared immediate same-device owner.");
      }
      if (!raw_program_lease_.acquire(device_.Get(), program->shaders)) return fail(error, "Missing/wrong-device outline-support-closure shader.");
      if (!roi_program_lease_.acquire(device_.Get(), program->roi_graph)) return fail(error, "Missing/wrong-device outline-support-closure graph shader.");
      if (!band_program_lease_.acquire(device_.Get(), program->band_shaders)) return fail(error, "Missing/wrong-device outline-support band shader.");
      if (!inside_program_lease_.acquire(device_.Get(), program->inside_shaders)) return fail(error, "Missing/wrong-device independent Inside shader.");
      holes_enabled_ = hole_program_lease_.acquire(device_.Get(), program->hole_shaders);
      try {
        std::string optional_error;
        if (holes_enabled_ && !holes_.ensure(device_.Get(), width_, height_, optional_error)) holes_enabled_ = false;
      } catch (const std::exception &) { holes_enabled_ = false; }
      coverage_enabled_ = holes_enabled_ && coverage_program_lease_.acquire(device_.Get(), program->coverage_shaders);
      try {
        std::string optional_error;
        if (coverage_enabled_ && !holes_.ensure_coverage(optional_error, program->coverage_diagnostics)) coverage_enabled_ = false;
      } catch (const std::exception &) { coverage_enabled_ = false; }
      stability_enabled_ = stability_program_lease_.acquire(device_.Get(), program->stability_shaders);
      stability_enabled_ = stability_enabled_ && stability_source_format(rgb);
      try { if (stability_enabled_ && !ensure_stability()) stability_enabled_ = false; }
      catch (const std::exception &) { stability_enabled_ = false; } // Optional allocation cannot revoke old output.
      if (!stability_enabled_) stability_needs_reset_ = true;
      weak_aa_enabled_ = stability_enabled_;
      try { if (weak_aa_enabled_ && !ensure_weak_aa()) weak_aa_enabled_ = false; }
      catch (const std::exception &) { weak_aa_enabled_ = false; }
      if (!weak_aa_enabled_) invalidate_weak_aa();
      if (stability_header_.uav && stability_needs_reset_) {
        const saved_cs_t saved {context};unbind(context);
        constexpr std::array<UINT,4u> zero {};
        context->ClearUnorderedAccessViewUint(stability_header_.uav.Get(),zero.data());
        stability_needs_reset_ = false;
      }
      if (!validate_snapshots(*program, rgb, probability, support, ocr, error)) {
        return false;
      }
      // CBC19 performs the full CB/root/current/origin descriptor and predication preflight
      // before submitting its original cached helper. It is called exactly once per whole call.
      if (!enclosure_.dispatch(context, colour, graph, constants96, rgb, probability, support, ocr, root, root_offset, error, &program->enclosure)) {
        stability_needs_reset_ = true;
        invalidate_weak_aa();
        return false;
      }
      if (program->stage_observer) program->stage_observer->mark(context, current_outline_stage_observer::stage_e::cbc_end);
      if (!recorded_) {
        const saved_cs_t saved {context};
        emit(context, constants96, rgb, ocr, root, root_offset, *program);
        ++diagnostics_.direct_emissions;
        last_enabled_ = true;
        if (program->stage_observer) program->stage_observer->mark(context, current_outline_stage_observer::stage_e::f2_end);
        return true;
      }
      std::array<IUnknown *, 160u> objects {};
      UINT count = 0u;
      if (!signature_objects(objects, count, constants96, rgb, ocr, root, *program)) {
        return fail(error, "Inside owner recorded signature exceeds checked capacity112.");
      }
      if (!commands_ || !signature_.matches(objects, count, root_offset)) {
        commands_.Reset();
        signature_ = signature_t {};
        if (!deferred_ && FAILED(device_->CreateDeferredContext(0u, &deferred_))) {
          return fail(error, "Could not create outline-support-closure deferred context.");
        }
        deferred_->ClearState();
        deferred_->SetPredication(nullptr, FALSE);
        emit(deferred_.Get(), constants96, rgb, ocr, root, root_offset, *program);
        if (FAILED(deferred_->FinishCommandList(FALSE, &commands_))) {
          commands_.Reset();
          deferred_.Reset();
          return fail(error, "Could not finish outline-support-closure command list.");
        }
        if (!signature_.retain(objects, count, root_offset)) {
          commands_.Reset();return fail(error, "Inside owner could not retain recorded signature112.");
        }
        ++diagnostics_.recordings;
      }
      context->ExecuteCommandList(commands_.Get(), TRUE);
      ++diagnostics_.replays;
      last_enabled_ = true;
      if (program->stage_observer) program->stage_observer->mark(context, current_outline_stage_observer::stage_e::f2_end);
      return true;
    }

  private:
    static void unbind(ID3D11DeviceContext *c) {
      std::array<ID3D11ShaderResourceView *, 16u> s {};
      std::array<ID3D11UnorderedAccessView *, 8u> u {};
      c->CSSetShaderResources(0u, 16u, s.data());
      c->CSSetUnorderedAccessViews(0u, 8u, u.data(), nullptr);
    }

    template<class T>
    bool same_device(T *child) const {
      if (!child) {
        return false;
      }
      ComPtr<ID3D11Device> d;
      child->GetDevice(&d);
      return d.Get() == device_.Get();
    }

    bool fail(std::string &error, const char *why) {
      ++diagnostics_.unavailable;
      invalidate_weak_aa();
      stability_needs_reset_ = true;
      error = why;
      return false;
    }

    bool validate_snapshots(const program_t &p, ID3D11ShaderResourceView *rgb, ID3D11ShaderResourceView *probability, ID3D11ShaderResourceView *support, ID3D11ShaderResourceView *ocr, std::string &error) {
      const std::array<ID3D11Resource *, 5u> destinations {
        p.packed_temp_snapshot, p.passable_cache_snapshot, p.pre_inside_final_snapshot,
        p.raw_parent_snapshot, p.raw_flags_snapshot
      };
      for (UINT i = 0u; i < destinations.size(); ++i) {
        if (!destinations[i]) continue;
        for (UINT j = 0u; j < i; ++j) {
          if (destinations[i] == destinations[j]) return fail(error, "Diagnostic snapshot destinations must be distinct.");
        }
      }
      auto aliases_current = [&](ID3D11Resource *snapshot) {
        for (auto *view : {rgb, probability, support, ocr, p.enclosure.current_sct, p.enclosure.current_uip, p.enclosure.producer.original_source, enclosure_views_.body.Get(), enclosure_views_.matte.Get(), enclosure_views_.receipt.Get(), enclosure_views_.all_body_bits.Get(), enclosure_views_.cover_bits.Get(), enclosure_views_.selected_bits.Get(), enclosure_views_.family_models.Get(), scratch_.eligible.Get(), scratch_.seeds.Get(), scratch_.domain_tag.Get(), scratch_.cover_clip.Get(), scratch_.parent.Get(), scratch_.root_flags.Get(), scratch_.mode_body.Get(), scratch_.mode_final.Get(), scratch_.body.Get(), scratch_.matte.Get(), scratch_.palette.Get(), scratch_.validation_receipt.Get(), matte_.srv.Get(), inner_.srv.Get(), meta_.srv.Get(), profile_data_.srv.Get(), colour_first_.srv.Get(), root_cache_.srv.Get(), inside_bits_.srv.Get(), inside_args_.srv.Get(), stability_image_.srv.Get(), stability_header_.srv.Get(), stability_aa_history_.srv.Get(), weak_aa_state_.srv.Get()}) {
          if (!view) continue;
          ComPtr<ID3D11Resource> resource;view->GetResource(&resource);
          if (resource.Get() == snapshot) return true;
        }
        return false;
      };
      for (auto *snapshot : {p.packed_temp_snapshot, p.passable_cache_snapshot, p.pre_inside_final_snapshot}) {
        if (!snapshot) continue;
        if (!same_device(snapshot)) return fail(error, "Snapshot belongs to another D3D device.");
        D3D11_TEXTURE2D_DESC d {};snapshot->GetDesc(&d);
        if (d.Width != width_ || d.Height != height_ || d.Format != DXGI_FORMAT_R32_UINT || d.MipLevels != 1u || d.ArraySize != 1u || d.SampleDesc.Count != 1u || d.SampleDesc.Quality != 0u || d.Usage != D3D11_USAGE_DEFAULT || d.CPUAccessFlags != 0u || d.MiscFlags != 0u || (d.BindFlags & ~(D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS)) != 0u) {
          return fail(error, "Snapshot requires exact native R32_UINT/default/single-sample storage.");
        }
        if (aliases_current(snapshot)) return fail(error, "Snapshot aliases an owned, borrowed or input resource.");
      }
      for (auto *snapshot : {p.raw_parent_snapshot, p.raw_flags_snapshot}) {
        if (!snapshot) continue;
        if (!same_device(snapshot)) return fail(error, "Raw graph snapshot belongs to another D3D device.");
        D3D11_BUFFER_DESC d {};snapshot->GetDesc(&d);
        if (d.ByteWidth != (pixels_ + 1u) * 4u || d.Usage != D3D11_USAGE_DEFAULT || d.CPUAccessFlags != 0u || d.MiscFlags != D3D11_RESOURCE_MISC_BUFFER_STRUCTURED || d.StructureByteStride != 4u || (d.BindFlags & ~(D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS)) != 0u) {
          return fail(error, "Raw graph snapshot requires exact native (N+1) structured uint storage.");
        }
        if (aliases_current(snapshot)) return fail(error, "Raw graph snapshot aliases a working resource.");
      }
      return true;
    }

    bool stability_source_format(ID3D11ShaderResourceView *rgb) const {
      if (!rgb) return false;
      D3D11_SHADER_RESOURCE_VIEW_DESC v {};rgb->GetDesc(&v);
      if (v.Format!=DXGI_FORMAT_R16G16B16A16_FLOAT || v.ViewDimension!=D3D11_SRV_DIMENSION_TEXTURE2D ||
          v.Texture2D.MostDetailedMip!=0u || v.Texture2D.MipLevels!=1u) return false;
      ComPtr<ID3D11Resource> r;rgb->GetResource(&r);ComPtr<ID3D11Texture2D> t;
      if (!r || FAILED(r.As(&t))) return false;
      D3D11_TEXTURE2D_DESC d {};t->GetDesc(&d);
      return d.Width==width_ && d.Height==height_ && d.Format==v.Format && d.MipLevels==1u &&
             d.ArraySize==1u && d.SampleDesc.Count==1u;
    }

    void invalidate_weak_aa() {
      const bool had_resource = weak_aa_state_.resource != nullptr;
      weak_aa_enabled_ = false;
      weak_aa_state_ = buffer_t {};
      // Existing GPU command lists retain old resources until their ordered work ends.
      if (had_resource) { commands_.Reset();signature_ = signature_t {}; }
    }

    bool ensure_weak_aa() {
      if (weak_aa_state_.resource) return true;
      constexpr UINT header_words = 256u;
      if (pixels_ > std::numeric_limits<UINT>::max() / 4u - header_words) return false;
      const UINT words = pixels_ + header_words;
      const std::vector<UINT> zeros(words, 0u);
      D3D11_BUFFER_DESC d {};d.ByteWidth = words * 4u;d.Usage = D3D11_USAGE_DEFAULT;
      d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;d.StructureByteStride = 4u;
      D3D11_SUBRESOURCE_DATA initial {zeros.data(),0u,0u};buffer_t next;
      if (FAILED(device_->CreateBuffer(&d,&initial,&next.resource)) ||
          FAILED(device_->CreateShaderResourceView(next.resource.Get(),nullptr,&next.srv)) ||
          FAILED(device_->CreateUnorderedAccessView(next.resource.Get(),nullptr,&next.uav))) return false;
      weak_aa_state_ = std::move(next);return true;
    }

    bool ensure_stability() {
      if (stability_image_.resource && stability_header_.resource && stability_aa_history_.resource) return true;
      texture_t image,header,history;
      auto make = [&](texture_t &t,UINT w,UINT h) {
        D3D11_TEXTURE2D_DESC d {};d.Width=w;d.Height=h;d.MipLevels=d.ArraySize=1u;
        d.Format=DXGI_FORMAT_R32_UINT;d.SampleDesc.Count=1u;d.Usage=D3D11_USAGE_DEFAULT;
        d.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;
        const std::vector<UINT> zeros(static_cast<std::size_t>(w)*h,0u);
        D3D11_SUBRESOURCE_DATA initial {zeros.data(),w*4u,0u};
        return SUCCEEDED(device_->CreateTexture2D(&d,&initial,&t.resource)) &&
               SUCCEEDED(device_->CreateShaderResourceView(t.resource.Get(),nullptr,&t.srv)) &&
               SUCCEEDED(device_->CreateUnorderedAccessView(t.resource.Get(),nullptr,&t.uav));
      };
      if (!make(image,width_,height_) || !make(header,160u,1u) || !make(history,width_,height_)) return false;
      stability_image_=std::move(image);stability_header_=std::move(header);stability_aa_history_=std::move(history);stability_needs_reset_=false;
      return true;
    }

    bool allocate(std::string &error) {
      enclosure_views_ = enclosure_.mutable_views();
      scratch_ = enclosure_.baseline_views();
      const std::array<ID3D11ShaderResourceView *, 8u> scratch_views {
        scratch_.eligible.Get(),
        scratch_.seeds.Get(),
        scratch_.domain_tag.Get(),
        scratch_.cover_clip.Get(),
        scratch_.parent.Get(),
        scratch_.root_flags.Get(),
        scratch_.mode_body.Get(),
        scratch_.mode_final.Get()
      };
      for (UINT i = 0u; i < scratch_views.size(); ++i) {
        ComPtr<ID3D11Resource> resource;
        scratch_views[i]->GetResource(&resource);
        if (FAILED(device_->CreateUnorderedAccessView(resource.Get(), nullptr, &scratch_uavs_[i]))) {
          error = "Could not bind borrowed outline-support-closure graph scratch.";
          return false;
        }
        // Retain storage at allocation time; command recording never queries SRVs.
        if (i == 4u) raw_parent_resource_ = resource;
        if (i == 5u) raw_flags_resource_ = resource;
      }
      const std::vector<UINT> zero(std::max<std::size_t>(static_cast<std::size_t>(pixels_) + 512u, data_word_count), 0u);
      auto texture = [&](texture_t &t) {
        D3D11_TEXTURE2D_DESC d {};
        d.Width = width_;
        d.Height = height_;
        d.MipLevels = d.ArraySize = 1u;
        d.Format = DXGI_FORMAT_R32_UINT;
        d.SampleDesc.Count = 1u;
        d.Usage = D3D11_USAGE_DEFAULT;
        d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        D3D11_SUBRESOURCE_DATA z {zero.data(), width_ * 4u, 0u};
        return SUCCEEDED(device_->CreateTexture2D(&d, &z, &t.resource)) &&
               SUCCEEDED(device_->CreateShaderResourceView(t.resource.Get(), nullptr, &t.srv)) &&
               SUCCEEDED(device_->CreateUnorderedAccessView(t.resource.Get(), nullptr, &t.uav));
      };
      if (!texture(matte_) || !texture(inner_) || !texture(colour_first_) || !texture(root_cache_) || !texture(inside_bits_)) {
        error = "Could not allocate outline-support-closure planes.";
        return false;
      }
      D3D11_BUFFER_DESC d {};
      d.ByteWidth = 512u * 4u;
      d.Usage = D3D11_USAGE_DEFAULT;
      d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      d.StructureByteStride = 4u;
      D3D11_SUBRESOURCE_DATA z {zero.data(), 0u, 0u};
      if (FAILED(device_->CreateBuffer(&d, &z, &meta_.resource)) || FAILED(device_->CreateShaderResourceView(meta_.resource.Get(), nullptr, &meta_.srv)) || FAILED(device_->CreateUnorderedAccessView(meta_.resource.Get(), nullptr, &meta_.uav))) {
        error = "Could not allocate outline-support-closure Meta512.";
        return false;
      }
      d = {};
      d.ByteWidth = 64u;
      d.Usage = D3D11_USAGE_DEFAULT;
      d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS | D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      D3D11_SHADER_RESOURCE_VIEW_DESC srv {};
      srv.Format = DXGI_FORMAT_R32_TYPELESS;
      srv.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
      srv.BufferEx.NumElements = 16u;
      srv.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
      D3D11_UNORDERED_ACCESS_VIEW_DESC uav {};
      uav.Format = DXGI_FORMAT_R32_TYPELESS;
      uav.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
      uav.Buffer.NumElements = 16u;
      uav.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
      if (FAILED(device_->CreateBuffer(&d, &z, &unit_args_.resource)) || FAILED(device_->CreateShaderResourceView(unit_args_.resource.Get(), &srv, &unit_args_.srv)) || FAILED(device_->CreateUnorderedAccessView(unit_args_.resource.Get(), &uav, &unit_args_.uav))) {
        error = "Could not allocate outline-support-closure unit arguments.";
        return false;
      }
      d.ByteWidth = 16u;
      srv.BufferEx.NumElements = 4u;uav.Buffer.NumElements = 4u;
      if (FAILED(device_->CreateBuffer(&d, &z, &inside_args_.resource)) || FAILED(device_->CreateShaderResourceView(inside_args_.resource.Get(), &srv, &inside_args_.srv)) || FAILED(device_->CreateUnorderedAccessView(inside_args_.resource.Get(), &uav, &inside_args_.uav))) {
        error = "Could not allocate independent Inside16B arguments.";return false;
      }
      d.ByteWidth = data_word_count * 4u;
      d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
      srv.BufferEx.NumElements = data_word_count;
      uav.Buffer.NumElements = data_word_count;
      if (FAILED(device_->CreateBuffer(&d, &z, &profile_data_.resource)) || FAILED(device_->CreateShaderResourceView(profile_data_.resource.Get(), &srv, &profile_data_.srv)) || FAILED(device_->CreateUnorderedAccessView(profile_data_.resource.Get(), &uav, &profile_data_.uav))) {
        error = "Could not allocate GPU-only colour-first profile data.";
        return false;
      }
      auto cb = [&](ComPtr<ID3D11Buffer> &buffer, const void *data) {
        D3D11_BUFFER_DESC desc {};
        desc.ByteWidth = 16u;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA init {data, 0u, 0u};
        return SUCCEEDED(device_->CreateBuffer(&desc, &init, &buffer));
      };
      const std::array<UINT, 4u> shape {width_, height_, pixels_, 0u};
      if (!cb(shape_cb_, shape.data()) || !cb(hook_cb_, zero.data())) {
        error = "Could not allocate outline-support-closure constants.";
        return false;
      }
      for (UINT unit = 0u; unit < unit_constants_.size(); ++unit) {
        D3D11_BUFFER_DESC desc{};
        desc.ByteWidth = 16u;desc.Usage = D3D11_USAGE_IMMUTABLE;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        const std::array<UINT, 4u> value{unit, 0u, 0u, 0u};
        const D3D11_SUBRESOURCE_DATA data{value.data(), 0u, 0u};
        if (FAILED(device_->CreateBuffer(&desc, &data, &unit_constants_[unit]))) {
          error = "Could not allocate immutable outline unit constants.";return false;
        }
      }
      d = {};
      d.ByteWidth = 12u;
      d.Usage = D3D11_USAGE_DEFAULT;
      d.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      const std::array<UINT, 3u> args {(width_ + 15u) / 16u, (height_ + 15u) / 16u, 0u};
      D3D11_SUBRESOURCE_DATA init {args.data(), 0u, 0u};
      if (FAILED(device_->CreateBuffer(&d, &init, &native_args_))) {
        error = "Could not allocate outline-support-closure native arguments.";
        return false;
      }
      return true;
    }

    bool signature_objects(std::array<IUnknown *, 160u> &objects, UINT &count, ID3D11Buffer *cb, ID3D11ShaderResourceView *rgb, ID3D11ShaderResourceView *ocr, ID3D11Buffer *root, const program_t &p) const {
      bool valid = true;
      auto add = [&](IUnknown *o) {
        if (count >= objects.size()) { valid = false;return; }
        objects[count++] = o;
      };
      for (auto *s : p.shaders) {
        add(s);
      }
      for (auto *s : p.roi_graph) {
        add(s);
      }
      for (auto *s : p.band_shaders) {
        add(s);
      }
      for (auto *s : p.inside_shaders) add(s);
      for (auto *s : p.stability_shaders) add(s);
      for (auto *shader : p.hole_shaders) add(shader);
      for (auto *shader : p.coverage_shaders) add(shader);
      add(holes_enabled_ ? holes_.proof() : nullptr);
      holes_.signature_objects(add);
      add(stability_image_.resource.Get());add(stability_image_.srv.Get());add(stability_image_.uav.Get());
      add(stability_header_.resource.Get());add(stability_header_.srv.Get());add(stability_header_.uav.Get());
      add(stability_aa_history_.resource.Get());add(stability_aa_history_.srv.Get());add(stability_aa_history_.uav.Get());
      add(weak_aa_state_.resource.Get());add(weak_aa_state_.srv.Get());add(weak_aa_state_.uav.Get());
      add(root_cache_.resource.Get());
      add(root_cache_.srv.Get());
      add(root_cache_.uav.Get());
      add(p.packed_temp_snapshot);
      add(p.passable_cache_snapshot);
      add(p.pre_inside_final_snapshot);add(p.raw_parent_snapshot);add(p.raw_flags_snapshot);
      add(ocr);
      add(colour_first_.srv.Get());
      add(colour_first_.uav.Get());
      add(profile_data_.resource.Get());
      add(profile_data_.srv.Get());
      add(profile_data_.uav.Get());
      add(cb);
      add(rgb);
      add(root);
      add(device_.Get());
      for (auto *s : {enclosure_views_.body.Get(), enclosure_views_.matte.Get(), enclosure_views_.receipt.Get(), enclosure_views_.all_body_bits.Get(), enclosure_views_.cover_bits.Get(), enclosure_views_.selected_bits.Get(), scratch_.eligible.Get(), scratch_.seeds.Get(), scratch_.domain_tag.Get(), scratch_.cover_clip.Get(), scratch_.parent.Get(), scratch_.root_flags.Get(), scratch_.mode_body.Get(), scratch_.mode_final.Get(), matte_.srv.Get(), inner_.srv.Get(), meta_.srv.Get()}) {
        add(s);
      }
      for (const auto &u : scratch_uavs_) {
        add(u.Get());
      }
      add(matte_.uav.Get());
      add(inner_.uav.Get());
      add(meta_.uav.Get());
      add(unit_args_.uav.Get());
      add(hook_cb_.Get());
      // This immutable private bank is replaced only with the complete owner.
      // The command list itself retains every bound unit buffer.
      add(unit_constants_[0u].Get());
      add(shape_cb_.Get());
      add(native_args_.Get());
      add(unit_args_.resource.Get());
      add(enclosure_views_.family_models.Get());
      add(raw_parent_resource_.Get());add(raw_flags_resource_.Get());
      add(inside_bits_.resource.Get());add(inside_bits_.srv.Get());add(inside_bits_.uav.Get());
      add(inside_args_.resource.Get());add(inside_args_.srv.Get());add(inside_args_.uav.Get());
      return valid;
    }

    // Pure GPU recording. No Get*, Map, query/readback, CPU pixel verdict, or heap allocation.
    void emit(ID3D11DeviceContext *c, ID3D11Buffer *cb, ID3D11ShaderResourceView *rgb, ID3D11ShaderResourceView *ocr, ID3D11Buffer *root, UINT root_offset, const program_t &p) {
      unbind(c);
      const D3D11_BOX gate {root_offset, 0u, 0u, root_offset + 4u, 1u, 1u};
      c->CopySubresourceRegion(native_args_.Get(), 0u, 8u, 0u, 0u, root, 0u, &gate);
      ID3D11Buffer *bc[3] {cb, unit_constants_[0u].Get(), shape_cb_.Get()};
      const std::array<ID3D11ShaderResourceView *, 10u> inputs {rgb, enclosure_views_.all_body_bits.Get(), enclosure_views_.selected_bits.Get(), enclosure_views_.cover_bits.Get(), enclosure_views_.receipt.Get(), scratch_.eligible.Get(), scratch_.parent.Get(), scratch_.root_flags.Get(), inner_.srv.Get(), enclosure_views_.matte.Get()};
      auto stage = [&](UINT entry, const std::array<ID3D11UnorderedAccessView *, 8u> &out, ID3D11Buffer *args, UINT offset) {
        // emit() enters unbound; every stage/graph leaves these slots unbound.
        auto in = inputs;
        if (out[1]) {
          in[5] = nullptr;
        }
        if (out[5]) {
          in[8] = nullptr;
        }
        c->CSSetConstantBuffers(0u, 3u, bc);
        c->CSSetShaderResources(0u, 10u, in.data());
        c->CSSetUnorderedAccessViews(0u, 8u, out.data(), nullptr);
        c->CSSetShader(p.shaders[entry], nullptr, 0u);
        c->DispatchIndirect(args, offset);
        unbind(c);
      };
      auto band_stage = [&](UINT entry, ID3D11Buffer *args, UINT offset) {
        std::array<ID3D11ShaderResourceView *, 16u> in {rgb, ocr, enclosure_views_.selected_bits.Get(), enclosure_views_.body.Get(), enclosure_views_.cover_bits.Get(), enclosure_views_.receipt.Get(), scratch_.parent.Get(), scratch_.root_flags.Get(), scratch_.eligible.Get(), inner_.srv.Get(), matte_.srv.Get(), meta_.srv.Get(), enclosure_views_.matte.Get(), stability_enabled_ ? stability_image_.srv.Get() : nullptr, stability_enabled_ ? stability_header_.srv.Get() : nullptr, inside_bits_.srv.Get()};
        std::array<ID3D11UnorderedAccessView *, 8u> out {};
        out[0] = profile_data_.uav.Get();
        if (entry >= 3u) {
          out[1] = colour_first_.uav.Get();
        }
        if (entry >= 5u) {
          out[2] = root_cache_.uav.Get();
          // The second Raw graph already cleared mode-final over this full ROI.
          // Aggregate stores current-root background witnesses; ordered closure
          // consumes them before the next unit reinitializes the same scratch.
          // No mode-final SRV is bound in this stage.
          out[3] = scratch_uavs_[7].Get();
        }
        // Final publication observes current Inside read-only through t15.
        if (entry == 4u && stability_enabled_) {
          in[13]=nullptr;in[14]=nullptr; // No snapshot-image/header SRV while its UAV is bound.
          out[4]=stability_header_.uav.Get();out[5]=stability_image_.uav.Get();out[6]=stability_aa_history_.uav.Get();
          out[7]=weak_aa_enabled_?weak_aa_state_.uav.Get():nullptr;
        }
        c->CSSetConstantBuffers(0u, 3u, bc);
        c->CSSetShaderResources(0u, 16u, in.data());
        c->CSSetUnorderedAccessViews(0u, 8u, out.data(), nullptr);
        c->CSSetShader(p.band_shaders[entry], nullptr, 0u);
        c->DispatchIndirect(args, offset);
        unbind(c);
      };
      auto cache_stage = [&](UINT entry,ID3D11Buffer *args,UINT offset) {
        if (!stability_enabled_) return;
        const std::array<ID3D11ShaderResourceView *,15u> in {rgb,ocr,enclosure_views_.selected_bits.Get(),
          enclosure_views_.body.Get(),enclosure_views_.cover_bits.Get(),enclosure_views_.receipt.Get(),
          scratch_.parent.Get(),scratch_.root_flags.Get(),scratch_.eligible.Get(),inner_.srv.Get(),matte_.srv.Get(),
          meta_.srv.Get(),enclosure_views_.matte.Get(),stability_image_.srv.Get(),nullptr};
        std::array<ID3D11UnorderedAccessView *,8u> out {};
        out[0]=profile_data_.uav.Get();out[4]=stability_header_.uav.Get();out[6]=stability_aa_history_.uav.Get();
        out[7]=weak_aa_enabled_?weak_aa_state_.uav.Get():nullptr;
        if (entry==1u) out[1]=colour_first_.uav.Get();
        c->CSSetConstantBuffers(0u,3u,bc);c->CSSetShaderResources(0u,15u,in.data());
        c->CSSetUnorderedAccessViews(0u,8u,out.data(),nullptr);
        c->CSSetShader(p.stability_shaders[entry],nullptr,0u);c->DispatchIndirect(args,offset);unbind(c);
      };
      auto hole_views = [&]() {
        holes::inputs_t in;
        in.rgb = rgb; in.ocr = ocr; in.selected = enclosure_views_.selected_bits.Get();
        in.cover = enclosure_views_.cover_bits.Get(); in.cbc = enclosure_views_.receipt.Get();
        in.raw_meta = meta_.srv.Get(); in.raw_parent = scratch_.parent.Get(); in.raw_flags = scratch_.root_flags.Get();
        in.raw_passable = scratch_.eligible.Get(); in.inner = inner_.srv.Get(); in.raw_witness = scratch_.mode_final.Get();
        in.old_final = colour_first_.srv.Get(); in.raw_m = matte_.srv.Get(); in.code1 = enclosure_views_.matte.Get();
        in.models = enclosure_views_.family_models.Get(); in.final_m = colour_first_.uav.Get(); in.data = profile_data_.uav.Get();
        return in;
      };
      constexpr std::array<UINT, 4u> zero {};
      std::array<ID3D11UnorderedAccessView *, 8u> u {};
      u[0] = meta_.uav.Get();
      u[7] = profile_data_.uav.Get();
      stage(0u, u, root, root_offset);
      cache_stage(0u,root,root_offset); // Fresh-only begin; root0 HOLD cannot mutate the cache.
      u[1] = colour_first_.uav.Get();
      u[2] = root_cache_.uav.Get();
      u[3] = scratch_uavs_[2].Get();
      u[4] = scratch_uavs_[3].Get();
      u[5] = inner_.uav.Get();
      u[6] = matte_.uav.Get();
      stage(1u, u, native_args_.Get(), 0u);
      band_stage(0u, root, root_offset);
      if (holes_enabled_) holes_.begin(c, p.hole_shaders, cb, shape_cb_.Get(), hole_views(), native_args_.Get());
      if (coverage_enabled_) holes_.begin_coverage(c, p.coverage_shaders, cb, shape_cb_.Get(), hole_views(), native_args_.Get());
      auto inside_stage = [&](UINT entry, ID3D11Buffer *args) {
        std::array<ID3D11ShaderResourceView *, 15u> in {
          rgb, enclosure_views_.all_body_bits.Get(), nullptr, enclosure_views_.cover_bits.Get(),
          enclosure_views_.receipt.Get(), scratch_.eligible.Get(), scratch_.parent.Get(), scratch_.root_flags.Get(),
          matte_.srv.Get(), enclosure_views_.matte.Get(), profile_data_.srv.Get(), nullptr, nullptr,
          enclosure_views_.family_models.Get(), inside_bits_.srv.Get()
        };
        std::array<ID3D11UnorderedAccessView *, 8u> out {};
        if (entry == 0u) out[7] = inside_args_.uav.Get();
        if (entry == 1u) {
          for (UINT k = 0u; k < 6u; ++k) out[k] = scratch_uavs_[k].Get();
          in[5] = in[6] = in[7] = nullptr;
        }
        if (entry == 2u) {
          out[6] = inside_bits_.uav.Get();out[7] = profile_data_.uav.Get();
          in[10] = in[14] = nullptr;
        }
        if (entry == 3u) {
          out[0] = colour_first_.uav.Get();out[6] = inside_bits_.uav.Get();out[7] = profile_data_.uav.Get();
          in[10] = in[14] = nullptr;
        }
        c->CSSetConstantBuffers(0u, 3u, bc);
        c->CSSetShaderResources(0u, 15u, in.data());
        c->CSSetUnorderedAccessViews(0u, 8u, out.data(), nullptr);
        c->CSSetShader(p.inside_shaders[entry], nullptr, 0u);
        c->DispatchIndirect(args, entry == 0u ? root_offset : 0u);
        unbind(c);
      };
      for (UINT slot = 0u; slot < 8u; ++slot) {
        bc[1u] = unit_constants_[slot].Get();
        // Prevent stale dispatch when root0 skips prepare. Legacy64B args stay untouched.
        c->ClearUnorderedAccessViewUint(inside_args_.uav.Get(), zero.data());
        inside_stage(0u, root);
        inside_stage(1u, inside_args_.resource.Get());
        ID3D11Buffer *constants[3] {shape_cb_.Get(), bc[1u], nullptr};
        const std::array<ID3D11ShaderResourceView *, 5u> graph_in {
          scratch_.eligible.Get(), scratch_.seeds.Get(), scratch_.domain_tag.Get(),
          scratch_.cover_clip.Get(), enclosure_views_.receipt.Get()
        };
        std::array<ID3D11UnorderedAccessView *, 8u> graph_out {};
        for (UINT k = 0u; k < 4u; ++k) graph_out[k] = scratch_uavs_[4u + k].Get();
        c->CSSetConstantBuffers(0u, 3u, constants);
        c->CSSetShaderResources(0u, 5u, graph_in.data());
        c->CSSetUnorderedAccessViews(0u, 8u, graph_out.data(), nullptr);
        // emit_init already initializes Parent/Flags. Reuse only canonical union/reduce.
        for (UINT entry : {1u, 2u}) {
          c->CSSetShader(p.roi_graph[entry], nullptr, 0u);
          c->DispatchIndirect(inside_args_.resource.Get(), 0u);
        }
        unbind(c);
        inside_stage(2u, inside_args_.resource.Get());
      }
      auto component_graph = [&]() {
        ID3D11Buffer *constants[3] {shape_cb_.Get(), bc[1u], nullptr};
        const std::array<ID3D11ShaderResourceView *, 5u> in {
          scratch_.eligible.Get(),
          scratch_.seeds.Get(),
          scratch_.domain_tag.Get(),
          scratch_.cover_clip.Get(),
          enclosure_views_.receipt.Get()
        };
        std::array<ID3D11UnorderedAccessView *, 8u> out {};
        out[0] = scratch_uavs_[4].Get();
        out[1] = scratch_uavs_[5].Get();
        out[2] = scratch_uavs_[6].Get();
        out[3] = scratch_uavs_[7].Get();
        c->CSSetConstantBuffers(0u, 3u, constants);
        c->CSSetShaderResources(0u, 5u, in.data());
        c->CSSetUnorderedAccessViews(0u, 8u, out.data(), nullptr);
        for (auto *shader : p.roi_graph) {
          c->CSSetShader(shader, nullptr, 0u);
          c->DispatchIndirect(unit_args_.resource.Get(), 0u);
        }
        unbind(c);
      };
      for (UINT slot = 0u; slot < 8u; ++slot) {
        bc[1u] = unit_constants_[slot].Get();
        c->ClearUnorderedAccessViewUint(unit_args_.uav.Get(), zero.data());
        u.fill(nullptr);
        u[0] = meta_.uav.Get();
        u[7] = unit_args_.uav.Get();
        stage(2u, u, root, root_offset);
        u.fill(nullptr);
        u[0] = meta_.uav.Get();
        u[1] = scratch_uavs_[0].Get();
        u[2] = scratch_uavs_[1].Get();
        u[3] = scratch_uavs_[2].Get();
        u[4] = scratch_uavs_[3].Get();
        u[5] = inner_.uav.Get();
        stage(3u, u, unit_args_.resource.Get(), 0u);
        component_graph();
        u.fill(nullptr);
        u[0] = meta_.uav.Get();
        u[5] = inner_.uav.Get();
        u[4] = root_cache_.uav.Get();
        u[7] = profile_data_.uav.Get();
        stage(4u, u, unit_args_.resource.Get(), 0u);
        band_stage(1u, unit_args_.resource.Get(), 0u);
        band_stage(2u, root, root_offset);
        band_stage(3u, unit_args_.resource.Get(), 0u);
        u.fill(nullptr);
        u[0] = meta_.uav.Get();
        u[1] = scratch_uavs_[0].Get();
        u[2] = scratch_uavs_[1].Get();
        u[3] = scratch_uavs_[2].Get();
        u[4] = scratch_uavs_[3].Get();
        stage(5u, u, unit_args_.resource.Get(), 0u);
        component_graph();
        u.fill(nullptr);
        u[0] = meta_.uav.Get();
        u[6] = matte_.uav.Get();
        stage(6u, u, unit_args_.resource.Get(), 0u);
        band_stage(5u, unit_args_.resource.Get(), 0u);
        band_stage(6u, unit_args_.resource.Get(), 0u);
        if (holes_enabled_) holes_.capture_unit(c, slot, p.hole_shaders, cb, shape_cb_.Get(), hole_views(), unit_args_.resource.Get());
      }
      u.fill(nullptr);
      u[0] = meta_.uav.Get();
      u[6] = matte_.uav.Get();
      stage(7u, u, native_args_.Get(), 0u);
      cache_stage(1u,native_args_.Get(),0u); // Full geometry validation/invalidate before snapshot overwrite.
      // Output-only diagnostics at the real prepublish point. Root0 is not a fresh
      // packed observation: the held temp texture then contains prior final binary M.
      if (p.packed_temp_snapshot) {
        c->CopyResource(p.packed_temp_snapshot, colour_first_.resource.Get());
      }
      if (p.passable_cache_snapshot) {
        c->CopyResource(p.passable_cache_snapshot, root_cache_.resource.Get());
      }
      band_stage(4u, native_args_.Get(), 0u);
      cache_stage(2u,root,root_offset); // Ordered commit after every snapshot-image write.
      // All legacy Raw/profile/grow/packed/cache/final141 work has completed.
      // These output-only TEST copies contain fresh Raw graph data only for issued units.
      // Unissued regions retain shared CBC/Inside scratch and have no current Raw authority.
      if (p.pre_inside_final_snapshot) c->CopyResource(p.pre_inside_final_snapshot, colour_first_.resource.Get());
      if (p.raw_parent_snapshot) c->CopyResource(p.raw_parent_snapshot, raw_parent_resource_.Get());
      if (p.raw_flags_snapshot) c->CopyResource(p.raw_flags_snapshot, raw_flags_resource_.Get());
      inside_stage(3u, native_args_.Get());
      if (holes_enabled_) holes_.finish(c, p.hole_shaders, cb, shape_cb_.Get(), hole_views(), root, root_offset, native_args_.Get(),
                                       p.coverage_shaders, coverage_enabled_);
      unbind(c);
    }

    bool ready_ = false, recorded_ = true, enclosure_recorded_ = true, baseline_recorded_ = true, last_enabled_ = false;
    UINT width_ = 0u, height_ = 0u, pixels_ = 0u;
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11Resource> raw_parent_resource_, raw_flags_resource_;
    enclosure::resources_t enclosure_;
    enclosure::mutable_views_t enclosure_views_;
    baseline::mutable_views_t scratch_;
    std::array<ComPtr<ID3D11UnorderedAccessView>, 8u> scratch_uavs_;
    texture_t matte_, inner_, colour_first_, root_cache_, inside_bits_;
    texture_t stability_image_,stability_header_,stability_aa_history_;
    bool stability_enabled_=false,stability_needs_reset_=true;
    bool weak_aa_enabled_=false;
    holes::resources_t holes_;
    bool holes_enabled_ = false;
    bool coverage_enabled_ = false;
    buffer_t weak_aa_state_;
    buffer_t meta_, unit_args_, profile_data_, inside_args_;
    ComPtr<ID3D11Buffer> hook_cb_, shape_cb_, native_args_;
    std::array<ComPtr<ID3D11Buffer>, 8u> unit_constants_;
    host_sbs_compute_program_lease::group_t<8u> raw_program_lease_;
    host_sbs_compute_program_lease::group_t<3u> roi_program_lease_, stability_program_lease_, coverage_program_lease_;
    host_sbs_compute_program_lease::group_t<7u> band_program_lease_;
    host_sbs_compute_program_lease::group_t<4u> inside_program_lease_;
    host_sbs_compute_program_lease::group_t<9u> hole_program_lease_;
    ComPtr<ID3D11DeviceContext> deferred_;
    ComPtr<ID3D11CommandList> commands_;
    signature_t signature_;
    diagnostics_t diagnostics_;
  };
}  // namespace models::host_sbs_current_outline_support_closure_gpu_v2
#endif
