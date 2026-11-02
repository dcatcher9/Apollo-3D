#pragma once

// Private CPU4023-equivalent current-only composition. No readback, temporal state,
// glyph membership certificate, EMPTY, or drawing authority. Default null delegates5cad.
#ifdef _WIN32
  #include "host_sbs_current_black_enclosure_gpu.h"

  #include <array>
  #include <cstdint>
  #include <exception>
  #include <limits>
  #include <string>
  #include <utility>
  #include <vector>

namespace models::host_sbs_current_black_ring_gpu_v1 {
  using Microsoft::WRL::ComPtr;
  namespace enclosure = host_sbs_current_black_enclosure_gpu_v1;
  namespace baseline = enclosure::baseline;

  struct program_t {
    enclosure::program_t enclosure;
    // begin, reset, prepare-unit, emit-passable, resolve-inner, emit-black, append, publish.
    std::array<ID3D11ComputeShader *, 8u> shaders {};
    // Exact original bounded init/union/reduce, run twice with different current predicates.
    std::array<ID3D11ComputeShader *, 3u> roi_graph {};
  };

  struct mutable_views_t {
    UINT width = 0u, height = 0u;
    ComPtr<ID3D11ShaderResourceView> body, matte, receipt, eligible_inner;
    ComPtr<ID3D11Buffer> hook_constants, unit_arguments, native_arguments;
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
      std::array<ComPtr<ID3D11Buffer>, 3u> b;
      ComPtr<ID3D11ComputeShader> shader;
      std::array<ComPtr<ID3D11ClassInstance>, 256u> instances;
      UINT count = 256u;

      explicit saved_cs_t(ID3D11DeviceContext *context):
          c(context) {
        std::array<ID3D11ShaderResourceView *, 16u> ss {};
        std::array<ID3D11UnorderedAccessView *, 8u> uu {};
        std::array<ID3D11Buffer *, 3u> bb {};
        std::array<ID3D11ClassInstance *, 256u> ii {};
        ID3D11ComputeShader *cs = nullptr;
        c->CSGetShaderResources(0u, 16u, ss.data());
        c->CSGetUnorderedAccessViews(0u, 8u, uu.data());
        c->CSGetConstantBuffers(0u, 3u, bb.data());
        c->CSGetShader(&cs, ii.data(), &count);
        shader.Attach(cs);
        for (UINT n = 0; n < 16u; ++n) {
          s[n].Attach(ss[n]);
        }
        for (UINT n = 0; n < 8u; ++n) {
          u[n].Attach(uu[n]);
        }
        for (UINT n = 0; n < 3u; ++n) {
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
        std::array<ID3D11Buffer *, 3u> bb {};
        std::array<ID3D11ClassInstance *, 256u> ii {};
        for (UINT n = 0; n < 16u; ++n) {
          ss[n] = s[n].Get();
        }
        for (UINT n = 0; n < 8u; ++n) {
          uu[n] = u[n].Get();
        }
        for (UINT n = 0; n < 3u; ++n) {
          bb[n] = b[n].Get();
        }
        for (UINT n = 0; n < count; ++n) {
          ii[n] = instances[n].Get();
        }
        c->CSSetUnorderedAccessViews(0u, 8u, uu.data(), nullptr);
        c->CSSetShaderResources(0u, 16u, ss.data());
        c->CSSetConstantBuffers(0u, 3u, bb.data());
        c->CSSetShader(shader.Get(), count ? ii.data() : nullptr, count);
      }
    };

    struct signature_t {
      std::array<ComPtr<IUnknown>, 64u> values;
      UINT count = 0u, root_offset = 0u;

      bool matches(const std::array<IUnknown *, 64u> &v, UINT n, UINT offset) const noexcept {
        if (count != n || root_offset != offset) {
          return false;
        }
        for (UINT i = 0; i < n; ++i) {
          if (values[i].Get() != v[i]) {
            return false;
          }
        }
        return true;
      }

      void retain(const std::array<IUnknown *, 64u> &v, UINT n, UINT offset) {
        *this = signature_t {};
        count = n;
        root_offset = offset;
        for (UINT i = 0; i < n; ++i) {
          values[i] = v[i];
        }
      }
    };

  public:
    resources_t() = default;
    resources_t(const resources_t &) = delete;
    resources_t &operator=(const resources_t &) = delete;
    resources_t(resources_t &&) = default;
    resources_t &operator=(resources_t &&) = default;

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

    [[nodiscard]] mutable_views_t mutable_views() const {
      return {width_, height_, enclosure_views_.body, matte_.srv, meta_.srv, inner_.srv, hook_cb_, unit_args_.resource, native_args_};
    }

    bool ensure(ID3D11Device *device, UINT width, UINT height, std::string &error) {
      error.clear();
      constexpr UINT maximum = depth_coordinate_v2::subtitle_contour_source_max_extent;
      const std::uint64_t n = std::uint64_t {width} * height;
      if (!device || !width || !height || width > maximum || height > maximum || n + 1u > std::numeric_limits<UINT>::max() / 4u) {
        error = "Invalid current-black-ring device/native shape.";
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
        error = std::string {"Could not allocate current-black-ring storage: "} + e.what();
        return false;
      }
      next.ready_ = true;
      *this = std::move(next);
      return true;
    }

    // Serialized immediate owner leases all current sources and mutable scratch until ordered
    // completion. Scratch must be unbound outside CS0..15/UAV0..7. true = commands submitted only.
    // Root0 holds ring M/Inner/Meta; current unavailable input returns literal code1 M, unavailable.
    bool dispatch(ID3D11DeviceContext *context, const std::array<ID3D11ComputeShader *, 5u> &colour, const std::array<ID3D11ComputeShader *, 6u> &graph, ID3D11Buffer *constants96, ID3D11ShaderResourceView *rgb, ID3D11ShaderResourceView *probability, ID3D11ShaderResourceView *support, ID3D11ShaderResourceView *ocr, ID3D11Buffer *root, UINT root_offset, std::string &error, const program_t *program = nullptr) {
      error.clear();
      last_enabled_ = false;
      if (!program) {
        return enclosure_.dispatch(context, colour, graph, constants96, rgb, probability, support, ocr, root, root_offset, error);
      }
      if (!ready_ || !context || context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE || !same_device(context)) {
        return fail(error, "Enabled black-ring requires prepared immediate same-device owner.");
      }
      for (auto *s : program->shaders) {
        if (!same_device(s)) {
          return fail(error, "Missing/wrong-device black-ring shader.");
        }
      }
      for (auto *s : program->roi_graph) {
        if (!same_device(s)) {
          return fail(error, "Missing/wrong-device black-ring graph shader.");
        }
      }
      // CBC19 performs the full CB/root/current/origin descriptor and predication preflight
      // before submitting its original cached helper. It is called exactly once per whole call.
      if (!enclosure_.dispatch(context, colour, graph, constants96, rgb, probability, support, ocr, root, root_offset, error, &program->enclosure)) {
        return false;
      }
      if (!recorded_) {
        const saved_cs_t saved {context};
        emit(context, constants96, rgb, root, root_offset, *program);
        ++diagnostics_.direct_emissions;
        last_enabled_ = true;
        return true;
      }
      std::array<IUnknown *, 64u> objects {};
      UINT count = 0u;
      signature_objects(objects, count, constants96, rgb, root, *program);
      if (!commands_ || !signature_.matches(objects, count, root_offset)) {
        commands_.Reset();
        signature_ = signature_t {};
        if (!deferred_ && FAILED(device_->CreateDeferredContext(0u, &deferred_))) {
          return fail(error, "Could not create black-ring deferred context.");
        }
        deferred_->ClearState();
        deferred_->SetPredication(nullptr, FALSE);
        emit(deferred_.Get(), constants96, rgb, root, root_offset, *program);
        if (FAILED(deferred_->FinishCommandList(FALSE, &commands_))) {
          commands_.Reset();
          deferred_.Reset();
          return fail(error, "Could not finish black-ring command list.");
        }
        signature_.retain(objects, count, root_offset);
        ++diagnostics_.recordings;
      }
      context->ExecuteCommandList(commands_.Get(), TRUE);
      ++diagnostics_.replays;
      last_enabled_ = true;
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
      error = why;
      return false;
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
          error = "Could not bind borrowed black-ring graph scratch.";
          return false;
        }
      }
      const std::vector<UINT> zero(static_cast<std::size_t>(pixels_) + 512u, 0u);
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
      if (!texture(matte_) || !texture(inner_)) {
        error = "Could not allocate black-ring planes.";
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
        error = "Could not allocate black-ring Meta512.";
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
        error = "Could not allocate black-ring unit arguments.";
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
        error = "Could not allocate black-ring constants.";
        return false;
      }
      d = {};
      d.ByteWidth = 12u;
      d.Usage = D3D11_USAGE_DEFAULT;
      d.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      const std::array<UINT, 3u> args {(width_ + 15u) / 16u, (height_ + 15u) / 16u, 0u};
      D3D11_SUBRESOURCE_DATA init {args.data(), 0u, 0u};
      if (FAILED(device_->CreateBuffer(&d, &init, &native_args_))) {
        error = "Could not allocate black-ring native arguments.";
        return false;
      }
      return true;
    }

    void signature_objects(std::array<IUnknown *, 64u> &objects, UINT &count, ID3D11Buffer *cb, ID3D11ShaderResourceView *rgb, ID3D11Buffer *root, const program_t &p) const {
      auto add = [&](IUnknown *o) {
        objects[count++] = o;
      };
      for (auto *s : p.shaders) {
        add(s);
      }
      for (auto *s : p.roi_graph) {
        add(s);
      }
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
      add(shape_cb_.Get());
      add(native_args_.Get());
      add(unit_args_.resource.Get());
    }

    // Pure GPU recording. No Get*, Map, query/readback, CPU pixel verdict, or heap allocation.
    void emit(ID3D11DeviceContext *c, ID3D11Buffer *cb, ID3D11ShaderResourceView *rgb, ID3D11Buffer *root, UINT root_offset, const program_t &p) {
      unbind(c);
      const D3D11_BOX gate {root_offset, 0u, 0u, root_offset + 4u, 1u, 1u};
      c->CopySubresourceRegion(native_args_.Get(), 0u, 8u, 0u, 0u, root, 0u, &gate);
      ID3D11Buffer *bc[3] {cb, hook_cb_.Get(), shape_cb_.Get()};
      const std::array<ID3D11ShaderResourceView *, 10u> inputs {rgb, enclosure_views_.all_body_bits.Get(), enclosure_views_.selected_bits.Get(), enclosure_views_.cover_bits.Get(), enclosure_views_.receipt.Get(), scratch_.eligible.Get(), scratch_.parent.Get(), scratch_.root_flags.Get(), inner_.srv.Get(), enclosure_views_.matte.Get()};
      auto stage = [&](UINT entry, const std::array<ID3D11UnorderedAccessView *, 8u> &out, ID3D11Buffer *args, UINT offset) {
        unbind(c);
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
      std::array<ID3D11UnorderedAccessView *, 8u> u {};
      u[0] = meta_.uav.Get();
      stage(0u, u, root, root_offset);
      u[1] = scratch_uavs_[0].Get();
      u[2] = scratch_uavs_[1].Get();
      u[3] = scratch_uavs_[2].Get();
      u[4] = scratch_uavs_[3].Get();
      u[5] = inner_.uav.Get();
      u[6] = matte_.uav.Get();
      stage(1u, u, native_args_.Get(), 0u);
      auto component_graph = [&]() {
        unbind(c);
        ID3D11Buffer *constants[3] {shape_cb_.Get(), hook_cb_.Get(), nullptr};
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
      constexpr std::array<UINT, 4u> zero {};
      for (UINT slot = 0u; slot < 8u; ++slot) {
        const std::array<UINT, 4u> hook {slot, 0u, 0u, 0u};
        c->UpdateSubresource(hook_cb_.Get(), 0u, nullptr, hook.data(), 0u, 0u);
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
        stage(4u, u, unit_args_.resource.Get(), 0u);
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
      }
      u.fill(nullptr);
      u[0] = meta_.uav.Get();
      u[6] = matte_.uav.Get();
      stage(7u, u, native_args_.Get(), 0u);
      unbind(c);
    }

    bool ready_ = false, recorded_ = true, enclosure_recorded_ = true, baseline_recorded_ = true, last_enabled_ = false;
    UINT width_ = 0u, height_ = 0u, pixels_ = 0u;
    ComPtr<ID3D11Device> device_;
    enclosure::resources_t enclosure_;
    enclosure::mutable_views_t enclosure_views_;
    baseline::mutable_views_t scratch_;
    std::array<ComPtr<ID3D11UnorderedAccessView>, 8u> scratch_uavs_;
    texture_t matte_, inner_;
    buffer_t meta_, unit_args_;
    ComPtr<ID3D11Buffer> hook_cb_, shape_cb_, native_args_;
    ComPtr<ID3D11DeviceContext> deferred_;
    ComPtr<ID3D11CommandList> commands_;
    signature_t signature_;
    diagnostics_t diagnostics_;
  };
}  // namespace models::host_sbs_current_black_ring_gpu_v1
#endif
