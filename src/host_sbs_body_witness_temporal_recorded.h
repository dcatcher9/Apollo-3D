#pragma once

// EXTERNAL default-null composition. The null path delegates unchanged baseline5cad.
// Enabled path records the same V5 palette/emit/4CC ONCE and private temporal hooks into one
// owned deferred command list. Commands/diagnostics grant no completion or UI authority.
#ifdef _WIN32
  #include "host_sbs_body_witness_temporal_gpu.h"
  #include "host_sbs_subtitle_inner_colour_gpu_global_validation.h"

  #include <array>
  #include <cstdint>
  #include <d3d11.h>
  #include <exception>
  #include <string>
  #include <vector>
  #include <wrl/client.h>

namespace models::host_sbs_body_witness_temporal_recorded {
  using Microsoft::WRL::ComPtr;
  namespace baseline = host_sbs_subtitle_inner_colour_gpu_global_validation_v2;
  namespace temporal = host_sbs_body_witness_temporal_gpu;

  struct program_t {
    temporal::resources_t *resources = nullptr;  // Caller owns C++ lifetime; signature owns all COM storage.
    std::array<ID3D11ComputeShader *, 19u> shaders {};
    temporal::frame_constants_t frame;
    // Optional GPU serializer result, never read back. Shape/caps remain the CPU precondition.
    ComPtr<ID3D11Buffer> gpu_frame256;
    // Functional comparison records original BOX3/OR from SAME CC without classification replay.
    // Runtime-cost experiments may select false as a separately declared static program variant.
    bool include_baseline_control = true;
  };

  struct diagnostics_t {
    std::uint64_t recordings = 0u;
    std::uint64_t replays = 0u;
    std::uint64_t direct_emissions = 0u;
    std::uint64_t unavailable = 0u;
  };

  class resources_t {
    struct saved_cs_t {
      ID3D11DeviceContext *context;
      std::array<ComPtr<ID3D11ShaderResourceView>, 9u> srvs;
      std::array<ComPtr<ID3D11UnorderedAccessView>, 8u> uavs;
      std::array<ComPtr<ID3D11Buffer>, 3u> constants;
      ComPtr<ID3D11ComputeShader> shader;
      std::array<ComPtr<ID3D11ClassInstance>, 256u> instances;
      UINT count = 256u;

      explicit saved_cs_t(ID3D11DeviceContext *c):
          context(c) {
        std::array<ID3D11ShaderResourceView *, 9u> s {};
        std::array<ID3D11UnorderedAccessView *, 8u> u {};
        std::array<ID3D11Buffer *, 3u> b {};
        std::array<ID3D11ClassInstance *, 256u> i {};
        ID3D11ComputeShader *cs = nullptr;
        c->CSGetShaderResources(0u, 9u, s.data());
        c->CSGetUnorderedAccessViews(0u, 8u, u.data());
        c->CSGetConstantBuffers(0u, 3u, b.data());
        c->CSGetShader(&cs, i.data(), &count);
        shader.Attach(cs);
        for (UINT n = 0u; n < 9u; ++n) {
          srvs[n].Attach(s[n]);
        }
        for (UINT n = 0u; n < 8u; ++n) {
          uavs[n].Attach(u[n]);
        }
        for (UINT n = 0u; n < 3u; ++n) {
          constants[n].Attach(b[n]);
        }
        for (UINT n = 0u; n < count; ++n) {
          instances[n].Attach(i[n]);
        }
      }

      ~saved_cs_t() {
        unbind(context);
        std::array<ID3D11ShaderResourceView *, 9u> s {};
        std::array<ID3D11UnorderedAccessView *, 8u> u {};
        std::array<ID3D11Buffer *, 3u> b {};
        std::array<ID3D11ClassInstance *, 256u> i {};
        for (UINT n = 0u; n < 9u; ++n) {
          s[n] = srvs[n].Get();
        }
        for (UINT n = 0u; n < 8u; ++n) {
          u[n] = uavs[n].Get();
        }
        for (UINT n = 0u; n < 3u; ++n) {
          b[n] = constants[n].Get();
        }
        for (UINT n = 0u; n < count; ++n) {
          i[n] = instances[n].Get();
        }
        context->CSSetUnorderedAccessViews(0u, 8u, u.data(), nullptr);
        context->CSSetShaderResources(0u, 9u, s.data());
        context->CSSetConstantBuffers(0u, 3u, b.data());
        context->CSSetShader(shader.Get(), count ? i.data() : nullptr, count);
      }

      static void unbind(ID3D11DeviceContext *c) {
        std::array<ID3D11ShaderResourceView *, 9u> s {};
        std::array<ID3D11UnorderedAccessView *, 8u> u {};
        c->CSSetShaderResources(0u, 9u, s.data());
        c->CSSetUnorderedAccessViews(0u, 8u, u.data(), nullptr);
      }
    };

    struct signature_t {
      std::vector<ComPtr<IUnknown>> objects;
      UINT offset = 0u;
      bool baseline_control = true;

      bool matches(const std::vector<IUnknown *> &v, UINT o, bool b) const {
        if (offset != o || baseline_control != b || objects.size() != v.size()) {
          return false;
        }
        for (std::size_t i = 0u; i < v.size(); ++i) {
          if (objects[i].Get() != v[i]) {
            return false;
          }
        }
        return true;
      }

      void set(const std::vector<IUnknown *> &v, UINT o, bool b) {
        objects.clear();
        objects.reserve(v.size());
        for (auto *p : v) {
          objects.emplace_back(p);
        }
        offset = o;
        baseline_control = b;
      }
    };

  public:
    void set_recorded_execution(bool enabled) {
      recorded_ = enabled;
    }

    // Explicit baseline experiment only. Null defaults retain baseline5cad's own false default.
    void set_baseline_recorded_execution(bool enabled) {
      base_.set_recorded_execution(enabled);
    }

    [[nodiscard]] diagnostics_t diagnostics() const {
      return diagnostics_;
    }

    [[nodiscard]] bool last_commands_queued() const {
      return last_queued_;
    }

    bool ensure(ID3D11Device *device, UINT width, UINT height, std::string &error) {
      if (!base_.ensure(device, width, height, error)) {
        return false;
      }
      const auto v = base_.mutable_views();
      if (ready_ && device_.Get() == device && width_ == width && height_ == height && base_body_.Get() == v.body.Get()) {
        return true;
      }
      ready_ = false;
      device_ = device;
      width_ = width;
      height_ = height;
      base_body_ = v.body;
      command_list_.Reset();
      deferred_.Reset();
      signature_ = signature_t {};
      base_views_ = v;
      const std::array<ID3D11ShaderResourceView *, 11u> srvs {v.eligible.Get(), v.seeds.Get(), v.domain_tag.Get(), v.cover_clip.Get(), v.mode_body.Get(), v.mode_final.Get(), v.body.Get(), v.matte.Get(), v.palette.Get(), v.parent.Get(), v.root_flags.Get()};
      for (UINT i = 0u; i < srvs.size(); ++i) {
        if (!make_uav(srvs[i], uavs_[i], error)) {
          return false;
        }
      }
      if (!make_raw_uav(v.current_mode_args.Get(), mode_args_uav_, error)) {
        return false;
      }
      const std::array<UINT, 4u> zero {};
      const std::array<UINT, 4u> graph {width, height, width * height, 0u};
      if (!make_cb(zero, request_) || !make_cb(graph, graph_)) {
        error = "Could not allocate recorded temporal V5 constants.";
        return false;
      }
      if (!make_uav(v.validation_receipt.Get(), validation_uav_, error)) {
        return false;
      }
      ready_ = true;
      return true;
    }

    // Null program is literal unchanged baseline dispatch. Enabled private composition requires
    // all inputs/output storage unbound outside CS0..8/UAV0..7 and immutable until queued work ends.
    bool dispatch(ID3D11DeviceContext *context, const std::array<ID3D11ComputeShader *, 5u> &colour, const std::array<ID3D11ComputeShader *, 6u> &components, ID3D11Buffer *constants96, ID3D11ShaderResourceView *rgb, ID3D11ShaderResourceView *probability, ID3D11ShaderResourceView *support, ID3D11ShaderResourceView *ocr, ID3D11Buffer *root, UINT root_offset, std::string &error, const program_t *program = nullptr) {
      last_queued_ = false;
      error.clear();
      if (!program) {
        last_queued_ = base_.dispatch(context, colour, components, constants96, rgb, probability, support, ocr, root, root_offset, error);
        return last_queued_;
      }
      if (!ready_ || !valid_context(context, error) || !program->resources) {
        return unavailable(error, "Private temporal program/context is unavailable.");
      }
      for (auto *s : colour) {
        if (!same_device(s)) {
          return unavailable(error, "Private colour shaders incomplete/wrong device.");
        }
      }
      for (auto *s : components) {
        if (!same_device(s)) {
          return unavailable(error, "Private graph shaders incomplete/wrong device.");
        }
      }
      for (auto *s : program->shaders) {
        if (!same_device(s)) {
          return unavailable(error, "Private temporal shaders incomplete/wrong device.");
        }
      }
      if (!same_device(constants96) || !same_device(rgb) || !same_device(probability) || !same_device(support) || !same_device(ocr) || !same_device(root)) {
        return unavailable(error, "Private source operands incomplete/wrong device.");
      }
      D3D11_BUFFER_DESC cb {}, args {};
      constants96->GetDesc(&cb);
      root->GetDesc(&args);
      if (cb.ByteWidth != 96u || (cb.BindFlags & D3D11_BIND_CONSTANT_BUFFER) == 0u || (args.MiscFlags & D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS) == 0u || root_offset % 4u != 0u || root_offset > args.ByteWidth || args.ByteWidth - root_offset < 12u) {
        return unavailable(error, "Private currentCB/root descriptor mismatch.");
      }
      if (program->gpu_frame256) {
        D3D11_BUFFER_DESC d {}; program->gpu_frame256->GetDesc(&d);
        if (!same_device(program->gpu_frame256.Get()) || d.ByteWidth != 256u)
          return unavailable(error, "Private GPU Frame256 descriptor mismatch.");
      }
      const auto tv = program->resources->mutable_views();
      if (program->gpu_frame256 && program->gpu_frame256.Get() == tv.frame_constants.Get())
        return unavailable(error, "Private GPU Frame256 cannot alias its destination.");
      if (tv.width != width_ || tv.height != height_) {
        return unavailable(error, "Private backend physical shape mismatch.");
      }
      const saved_cs_t saved {context};  // ONE outer immediate scope, never a deferred GetState.
      try {
        saved_cs_t::unbind(context);
        // Allocation/bank clear ONCE and mutable per-frame identityCB update OUTSIDE replay.
        if (!program->resources->prepare_live_frame_owned(context, program->frame, error)) {
          return unavailable(error, "Private live frame preparation failed.");
        }
        if (program->gpu_frame256) {
          saved_cs_t::unbind(context);
          context->CopyResource(tv.frame_constants.Get(), program->gpu_frame256.Get());
        }
        const auto objects = signature_objects(colour, components, constants96, rgb, probability, support, ocr, root, *program);
        if (!recorded_) {
          if (!emit_commands(context, colour, components, constants96, rgb, probability, support, ocr, root, root_offset, *program, error)) {
            std::string quarantine;
            program->resources->quarantine_owned(context, quarantine);
            return unavailable(error, "Private direct emission failed; scope quarantined.");
          }
          ++diagnostics_.direct_emissions;
          last_queued_ = true;
          return true;
        }
        if (!command_list_ || !signature_.matches(objects, root_offset, program->include_baseline_control)) {
          command_list_.Reset();
          signature_ = signature_t {};
          if (!deferred_ && FAILED(device_->CreateDeferredContext(0u, &deferred_))) {
            return unavailable(error, "Could not create owned deferred temporal context.");
          }
          deferred_->ClearState();
          deferred_->SetPredication(nullptr, FALSE);
          if (!emit_commands(deferred_.Get(), colour, components, constants96, rgb, probability, support, ocr, root, root_offset, *program, error) || FAILED(deferred_->FinishCommandList(FALSE, &command_list_)) || !command_list_) {
            command_list_.Reset();
            deferred_->ClearState();
            deferred_.Reset();
            std::string quarantine;
            program->resources->quarantine_owned(context, quarantine);
            return unavailable(error, "Private recording failed; half list discarded and scope quarantined.");
          }
          signature_.set(objects, root_offset, program->include_baseline_control);
          ++diagnostics_.recordings;
        }
        // The literal root/current/frame buffers are LIVE inputs; only fixed hook requests are
        // UpdateSubresource commands inside this list. No CPU bank flip/readback/decision.
        context->ExecuteCommandList(command_list_.Get(), TRUE);
        ++diagnostics_.replays;
        last_queued_ = true;
        return true;
      } catch (const std::exception &) {
        command_list_.Reset();
        deferred_.Reset();
        signature_ = signature_t {};
        std::string quarantine;
        program->resources->quarantine_owned(context, quarantine);
        return unavailable(error, "Private temporal recording allocation failed; scope quarantined.");
      } catch (...) {
        command_list_.Reset();
        deferred_.Reset();
        signature_ = signature_t {};
        std::string quarantine;
        program->resources->quarantine_owned(context, quarantine);
        return unavailable(error, "Private temporal recording failed; scope quarantined.");
      }
    }

    [[nodiscard]] baseline::mutable_views_t baseline_views() const {
      return base_.mutable_views();
    }

  private:
    bool unavailable(std::string &error, const char *fallback) {
      ++diagnostics_.unavailable;
      if (error.empty()) {
        error = fallback;
      }
      last_queued_ = false;
      return false;
    }

    bool same_device(ID3D11DeviceChild *p) const {
      if (!p) {
        return false;
      }
      ComPtr<ID3D11Device> d;
      p->GetDevice(&d);
      return d.Get() == device_.Get();
    }

    bool valid_context(ID3D11DeviceContext *c, std::string &error) const {
      if (!c || !device_ || c->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE || !same_device(c)) {
        error = "Private temporal composition requires matching immediate context.";
        return false;
      }
      ComPtr<ID3D11Predicate> predicate;
      BOOL value = FALSE;
      c->GetPredication(&predicate, &value);
      if (predicate) {
        error = "Private temporal composition rejects predication before issuing commands.";
        return false;
      }
      return true;
    }

    bool make_cb(const std::array<UINT, 4u> &v, ComPtr<ID3D11Buffer> &out) {
      D3D11_BUFFER_DESC d {};
      d.ByteWidth = 16u;
      d.Usage = D3D11_USAGE_DEFAULT;
      d.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      const D3D11_SUBRESOURCE_DATA init {v.data(), 0u, 0u};
      return SUCCEEDED(device_->CreateBuffer(&d, &init, &out));
    }

    bool make_uav(ID3D11ShaderResourceView *srv, ComPtr<ID3D11UnorderedAccessView> &out, std::string &error) {
      ComPtr<ID3D11Resource> r;
      srv->GetResource(&r);
      if (FAILED(device_->CreateUnorderedAccessView(r.Get(), nullptr, &out))) {
        error = "Could not retain frozen V5 working UAV.";
        return false;
      }
      return true;
    }

    bool make_raw_uav(ID3D11Buffer *buffer, ComPtr<ID3D11UnorderedAccessView> &out, std::string &error) {
      D3D11_BUFFER_DESC d {};
      buffer->GetDesc(&d);
      D3D11_UNORDERED_ACCESS_VIEW_DESC u {};
      u.Format = DXGI_FORMAT_R32_TYPELESS;
      u.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
      u.Buffer.NumElements = d.ByteWidth / 4u;
      u.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
      if (FAILED(device_->CreateUnorderedAccessView(buffer, &u, &out))) {
        error = "Could not retain frozen V5 raw mode argument UAV.";
        return false;
      }
      return true;
    }

    std::vector<IUnknown *> signature_objects(const std::array<ID3D11ComputeShader *, 5u> &c, const std::array<ID3D11ComputeShader *, 6u> &g, ID3D11Buffer *cb, ID3D11ShaderResourceView *rgb, ID3D11ShaderResourceView *p, ID3D11ShaderResourceView *s, ID3D11ShaderResourceView *ocr, ID3D11Buffer *root, const program_t &program) const {
      std::vector<IUnknown *> v {device_.Get(), cb, rgb, p, s, ocr, root, request_.Get(), graph_.Get(), base_views_.current_mode_args.Get(), base_views_.native_clear_args.Get()};
      for (auto *x : c) {
        v.push_back(x);
      }
      for (auto *x : g) {
        v.push_back(x);
      }
      for (auto *x : program.shaders) {
        v.push_back(x);
      }
      for (const auto &x : uavs_) {
        v.push_back(x.Get());
      }
      v.push_back(program.gpu_frame256.Get());
      v.push_back(mode_args_uav_.Get());
      v.push_back(validation_uav_.Get());
      const auto t = program.resources->mutable_views();
      v.insert(v.end(), {t.body.Get(), t.matte.Get(), t.output_array.Get(), t.state.Get(), t.records.Get(), t.members.Get(), t.bits.Get(), t.root_rgb.Get(), t.images.Get(), t.receipt.Get(), t.receipt_storage.Get(), t.end_arguments.Get(), t.hook_arguments.Get(), t.frame_constants.Get(), t.hook_constants.Get(), t.native_arguments.Get()});
      return v;
    }

    bool emit_commands(ID3D11DeviceContext *context, const std::array<ID3D11ComputeShader *, 5u> &colour, const std::array<ID3D11ComputeShader *, 6u> &graphShaders, ID3D11Buffer *cb, ID3D11ShaderResourceView *rgb, ID3D11ShaderResourceView *probability, ID3D11ShaderResourceView *support, ID3D11ShaderResourceView *ocr, ID3D11Buffer *root, UINT rootOffset, const program_t &program, std::string &error) {
      // Baseline5cad emit arithmetic, shader order and resource tuple are retained. Only hook
      // boundaries and explicit rebinds are added; no query/GetState exists on this owned context.
      saved_cs_t::unbind(context);
      const D3D11_BOX gate {rootOffset, 0u, 0u, rootOffset + 4u, 1u, 1u};
      context->CopySubresourceRegion(base_views_.native_clear_args.Get(), 0u, 8u, 0u, 0u, root, 0u, &gate);
      constexpr std::array<UINT, 4u> zero {};
      context->ClearUnorderedAccessViewUint(validation_uav_.Get(), zero.data());
      const std::array<UINT, 4u> validatorRequest {width_, height_, 0u, 0u};
      context->UpdateSubresource(request_.Get(), 0u, nullptr, validatorRequest.data(), 0u, 0u);
      ID3D11Buffer *colourCB[3] {cb, request_.Get(), nullptr};
      std::array<ID3D11UnorderedAccessView *, 8u> u {};
      const std::array<ID3D11ShaderResourceView *, 9u> colourInputs {rgb, probability, support, ocr, base_views_.palette.Get(), nullptr, nullptr, nullptr, base_views_.validation_receipt.Get()};
      auto validationInputs = colourInputs;
      validationInputs[4] = nullptr;
      validationInputs[8] = nullptr;
      context->CSSetConstantBuffers(0u, 3u, colourCB);
      context->CSSetShaderResources(0u, 9u, validationInputs.data());
      u[0] = validation_uav_.Get();
      context->CSSetUnorderedAccessViews(0u, 8u, u.data(), nullptr);
      context->CSSetShader(colour[4], nullptr, 0u);
      context->DispatchIndirect(base_views_.native_clear_args.Get(), 0u);
      saved_cs_t::unbind(context);
      if (!program.resources->begin_owned_emission(context, program.shaders, cb, rgb, base_views_.validation_receipt.Get(), root, rootOffset, error)) {
        return false;
      }
      // Rebind known original state after hook. No deferred state query/restoration.
      saved_cs_t::unbind(context);
      context->CSSetConstantBuffers(0u, 3u, colourCB);
      u.fill(nullptr);
      u[5] = uavs_[6].Get();
      u[6] = uavs_[7].Get();
      context->CSSetUnorderedAccessViews(0u, 8u, u.data(), nullptr);
      context->CSSetShader(colour[2], nullptr, 0u);
      context->DispatchIndirect(base_views_.native_clear_args.Get(), 0u);
      saved_cs_t::unbind(context);
      for (UINT slot = 0u; slot < 8u; ++slot) {
        context->ClearUnorderedAccessViewUint(mode_args_uav_.Get(), zero.data());
        const std::array<UINT, 4u> pr {slot, 0u, 0u, 0u};
        context->UpdateSubresource(request_.Get(), 0u, nullptr, pr.data(), 0u, 0u);
        auto pi = colourInputs;
        pi[4] = nullptr;
        context->CSSetConstantBuffers(0u, 3u, colourCB);
        context->CSSetShaderResources(0u, 9u, pi.data());
        u.fill(nullptr);
        u[0] = uavs_[8].Get();
        u[7] = mode_args_uav_.Get();
        context->CSSetUnorderedAccessViews(0u, 8u, u.data(), nullptr);
        context->CSSetShader(colour[0], nullptr, 0u);
        context->DispatchIndirect(root, rootOffset);
        saved_cs_t::unbind(context);
        if (!program.resources->unit_palette(context, slot, base_views_.palette.Get(), error)) {
          return false;
        }
        for (UINT mode = 0u; mode < 8u; ++mode) {
          const std::array<UINT, 4u> req {slot, mode, 0u, 0u};
          context->UpdateSubresource(request_.Get(), 0u, nullptr, req.data(), 0u, 0u);
          const UINT offset = 12u * mode;
          context->CSSetConstantBuffers(0u, 3u, colourCB);
          context->CSSetShaderResources(0u, 9u, colourInputs.data());
          u.fill(nullptr);
          for (UINT k = 0u; k < 4u; ++k) {
            u[k + 1u] = uavs_[k].Get();
          }
          context->CSSetUnorderedAccessViews(0u, 8u, u.data(), nullptr);
          context->CSSetShader(colour[1], nullptr, 0u);
          context->DispatchIndirect(base_views_.current_mode_args.Get(), offset);
          saved_cs_t::unbind(context);
          const std::array<ID3D11ShaderResourceView *, 4u> gi {base_views_.eligible.Get(), base_views_.seeds.Get(), base_views_.domain_tag.Get(), base_views_.cover_clip.Get()};
          ID3D11Buffer *graphCB[3] {graph_.Get(), nullptr, nullptr};
          context->CSSetConstantBuffers(0u, 3u, graphCB);
          context->CSSetShaderResources(0u, 4u, gi.data());
          u.fill(nullptr);
          u[0] = uavs_[9].Get();
          u[1] = uavs_[10].Get();
          u[2] = uavs_[4].Get();
          u[3] = uavs_[5].Get();
          context->CSSetUnorderedAccessViews(0u, 8u, u.data(), nullptr);
          for (UINT entry : {0u, 1u, 2u, 3u}) {
            context->CSSetShader(graphShaders[entry], nullptr, 0u);
            context->DispatchIndirect(base_views_.current_mode_args.Get(), offset);
          }
          saved_cs_t::unbind(context);
          if (!program.resources->mode_after_ccresolve(context, slot, mode, base_views_.seeds.Get(), base_views_.mode_body.Get(), base_views_.parent.Get(), base_views_.root_flags.Get(), base_views_.palette.Get(), base_views_.cover_clip.Get(), base_views_.current_mode_args.Get(), offset, error)) {
            return false;
          }
          if (program.include_baseline_control) {
            context->CSSetConstantBuffers(0u, 3u, graphCB);
            context->CSSetShaderResources(0u, 4u, gi.data());
            u.fill(nullptr);
            u[0] = uavs_[9].Get();
            u[1] = uavs_[10].Get();
            u[2] = uavs_[4].Get();
            u[3] = uavs_[5].Get();
            context->CSSetUnorderedAccessViews(0u, 8u, u.data(), nullptr);
            context->CSSetShader(graphShaders[5], nullptr, 0u);
            context->DispatchIndirect(base_views_.current_mode_args.Get(), offset);
            saved_cs_t::unbind(context);
            auto ai = colourInputs;
            ai[5] = base_views_.mode_body.Get();
            ai[6] = base_views_.mode_final.Get();
            ai[7] = base_views_.parent.Get();
            context->CSSetConstantBuffers(0u, 3u, colourCB);
            context->CSSetShaderResources(0u, 9u, ai.data());
            u.fill(nullptr);
            u[5] = uavs_[6].Get();
            u[6] = uavs_[7].Get();
            context->CSSetUnorderedAccessViews(0u, 8u, u.data(), nullptr);
            context->CSSetShader(colour[3], nullptr, 0u);
            context->DispatchIndirect(base_views_.current_mode_args.Get(), offset);
            saved_cs_t::unbind(context);
          }
        }
      }
      return program.resources->end(context, error);
    }

    baseline::resources_t base_;
    baseline::mutable_views_t base_views_;
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11ShaderResourceView> base_body_;
    UINT width_ = 0u, height_ = 0u;
    std::array<ComPtr<ID3D11UnorderedAccessView>, 11u> uavs_;
    ComPtr<ID3D11UnorderedAccessView> mode_args_uav_, validation_uav_;
    ComPtr<ID3D11Buffer> request_, graph_;
    ComPtr<ID3D11DeviceContext> deferred_;
    ComPtr<ID3D11CommandList> command_list_;
    signature_t signature_;
    diagnostics_t diagnostics_;
    bool recorded_ = true, last_queued_ = false, ready_ = false;
  };
}  // namespace models::host_sbs_body_witness_temporal_recorded
#endif
