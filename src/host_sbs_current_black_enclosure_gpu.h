#pragma once

// Private default-null research composition. The null program delegates the literal5cad helper.
// Enabled commands run original current V5 exactly once, then the frozen code1 enclosure.
// No CPU pixel/palette verdict, readback, temporal state, semantic EMPTY or drawing authority.
#ifdef _WIN32
  #include "host_sbs_subtitle_inner_colour_gpu_global_validation.h"
  #include "host_sbs_compute_program_lease.h"

  #include <array>
  #include <cstddef>
  #include <cstdint>
  #include <d3d11.h>
  #include <exception>
  #include <limits>
  #include <string>
  #include <utility>
  #include <vector>
  #include <wrl/client.h>

namespace models::host_sbs_current_black_enclosure_gpu_v1 {
  using Microsoft::WRL::ComPtr;
  namespace baseline = host_sbs_subtitle_inner_colour_gpu_global_validation_v2;

  enum class transfer_t : UINT {
    unknown = 0u,
    encoded_sdr = 1u,
    linear = 2u,
    hdr = 3u
  };

  // Host producer facts, never inferred from a numeric lattice. The actual origin SRV is leased
  // with all current operands until this ordered interval completes. AddRef preserves storage only.
  // input_known includes actual full analysis-ROI opacity/producer continuity; band alpha alone
  // cannot establish those host producer facts. Unknown transfer/recipe remains baseline fallback.
  struct producer_t {
    ID3D11ShaderResourceView *original_source = nullptr;
    transfer_t transfer = transfer_t::unknown;
    bool input_known = false;
    UINT direct_encoded_recipe = 0u;
    UINT postprocess_mode = 0u;
    UINT expected_work = 1u;
    std::uint64_t transaction_token = 0u;
    std::uint64_t model_frame = 0u;
    std::uint64_t model_observation_us = 0u;
    std::uint64_t analysis_generation = 0u;
    std::array<UINT, 4u> analysis_roi {};
  };

  struct program_t {
    std::array<ID3D11ComputeShader *, 12u> shaders {};
    std::array<ID3D11ComputeShader *, 3u> roi_graph {};
    ID3D11ShaderResourceView *current_sct = nullptr;
    ID3D11ShaderResourceView *current_uip = nullptr;
    producer_t producer;
    bool trusted_inner = false;
  };

  struct mutable_views_t {
    UINT width = 0u, height = 0u;
    ComPtr<ID3D11ShaderResourceView> body, matte, receipt;
    ComPtr<ID3D11ShaderResourceView> all_body_bits, cover_bits, selected_bits;
    ComPtr<ID3D11Buffer> producer_constants, hook_constants, native_arguments;
    ComPtr<ID3D11Buffer> unit_arguments, root_mirror;
    // Current per-unit family boxes, Texture2D<uint> 17x8; never native graph scratch.
    ComPtr<ID3D11ShaderResourceView> family_models;
    ComPtr<ID3D11ShaderResourceView> trusted_inner_bits;
    ComPtr<ID3D11Buffer> trusted_publication_arguments;
  };

  struct diagnostics_t {
    std::uint64_t recordings = 0u, replays = 0u, direct_emissions = 0u;
    std::uint64_t allocation_replacements = 0u, unavailable = 0u;
  };

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
      std::array<ComPtr<IUnknown>, 96u> values;
      UINT count = 0u, root_offset = 0u;

      bool matches(const std::array<IUnknown *, 96u> &v, UINT n, UINT offset) const noexcept {
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

      bool retain(const std::array<IUnknown *, 96u> &v, UINT n, UINT offset) {
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

    void set_recorded_execution(bool enabled) noexcept {
      recorded_ = enabled;
    }

    void set_baseline_recorded_execution(bool enabled) noexcept {
      // This selects only the null-program legacy fallback. Enabled CBC records
      // the shared colour schedule inside its own complete command list.
      base_.set_recorded_execution(enabled);
    }

    [[nodiscard]] diagnostics_t diagnostics() const noexcept {
      return diagnostics_;
    }

    [[nodiscard]] bool last_commands_enabled() const noexcept {
      return last_enabled_;
    }

    [[nodiscard]] baseline::mutable_views_t baseline_views() const {
      return base_.mutable_views();
    }
    [[nodiscard]] bool trusted_inner_ready() const noexcept { return trusted_ready_; }

    [[nodiscard]] mutable_views_t mutable_views() const {
      return {width_, height_, candidate_body_.srv, candidate_matte_.srv, meta_.srv, all_body_.srv, cover_bits_.srv, selected_bits_.srv, producer_cb_, hook_cb_, native_args_, unit_args_.resource, root_mirror_.resource, family_models_.srv, trusted_inner_.srv, trusted_publication_args_.resource};
    }

    bool ensure(ID3D11Device *device, UINT width, UINT height, std::string &error) {
      error.clear();
      constexpr UINT maximum = depth_coordinate_v2::subtitle_contour_source_max_extent;
      const std::uint64_t n = std::uint64_t {width} * height;
      if (!device || !width || !height || width > maximum || height > maximum || n + 1u > std::numeric_limits<UINT>::max() / 4u) {
        error = "Invalid current-black-enclosure device/native shape.";
        return false;
      }
      if (device_.Get() == device && width_ == width && height_ == height && ready_) {
        return true;
      }
      resources_t next;
      next.recorded_ = recorded_;
      next.diagnostics_ = diagnostics_;
      next.base_.set_recorded_execution(base_.recorded_execution());
      if (device_) {
        ++next.diagnostics_.allocation_replacements;
      }
      next.device_ = device;
      next.width_ = width;
      next.height_ = height;
      next.pixels_ = static_cast<UINT>(n);
      try {
        if (!next.base_.ensure(device, width, height, error) || !next.allocate(error)) {
          return false;
        }
      } catch (const std::exception &e) {
        error = std::string {"Could not allocate current-black-enclosure storage: "} + e.what();
        return false;
      }
      next.ready_ = true;
      *this = std::move(next);
      return true;
    }

    // Input/output storage must be unbound outside CS0..15/UAV0..7. Serialized immediate owner
    // holds every source/packet/CB/shader immutable until ordered GPU completion; views are mutable.
    // A true return proves command submission only. Enabled root0 holds private output+Meta.
    // Root is the literal complete256B GPU dispatch twin and root_offset is observationOne240.
    bool dispatch(ID3D11DeviceContext *context, const std::array<ID3D11ComputeShader *, 5u> &colour, const std::array<ID3D11ComputeShader *, 6u> &graph, ID3D11Buffer *constants96, ID3D11ShaderResourceView *rgb, ID3D11ShaderResourceView *probability, ID3D11ShaderResourceView *support, ID3D11ShaderResourceView *ocr, ID3D11Buffer *root, UINT root_offset, std::string &error, const program_t *program = nullptr) {
      error.clear();
      last_enabled_ = false;
      if (!program) {
        return base_.dispatch(context, colour, graph, constants96, rgb, probability, support, ocr, root, root_offset, error);
      }
      if (!ready_ || !context || context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE || !same_device(context)) {
        return fail(error, "Enabled enclosure requires prepared immediate same-device owner.");
      }
      ComPtr<ID3D11Predicate> predicate;
      BOOL predicate_value = FALSE;
      context->GetPredication(&predicate, &predicate_value);
      if (predicate) {
        return fail(error, "Enabled enclosure rejects active predication before commands.");
      }
      if (!colour_program_lease_.acquire(device_.Get(), colour)) return fail(error, "Missing/wrong-device colour shader.");
      if (!graph_program_lease_.acquire(device_.Get(), graph)) return fail(error, "Missing/wrong-device component shader.");
      if (!enclosure_program_lease_.acquire(device_.Get(), program->shaders)) return fail(error, "Missing/wrong-device enclosure shader.");
      if (!roi_program_lease_.acquire(device_.Get(), program->roi_graph)) return fail(error, "Missing/wrong-device ROI graph shader.");
      for (auto *s : {rgb, probability, support, ocr, program->current_sct, program->current_uip}) {
        if (!same_device(s)) {
          return fail(error, "Missing/wrong-device current input.");
        }
      }
      if (!same_device(constants96) || !same_device(root) || (program->producer.original_source && !same_device(program->producer.original_source))) {
        return fail(error, "Wrong-device CB/root/origin.");
      }
      D3D11_BUFFER_DESC cb {}, args {};
      constants96->GetDesc(&cb);
      root->GetDesc(&args);
      if (cb.ByteWidth != 96u || !(cb.BindFlags & D3D11_BIND_CONSTANT_BUFFER) || args.ByteWidth < 256u || !(args.MiscFlags & D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS) || root_offset != 240u) {
        return fail(error, "Enabled enclosure requires CB96 and complete root256/observationOne240.");
      }
      const auto producer = producer_words(program->producer);
      // Preserve the literal unsupported descriptor in b2, but never bind a buffer/array/MS/
      // integer/SRGB view to the private Texture2D<float4> register. Null is format fallback,
      // not an encoded-SDR assertion. The shader must reject support before any origin load.
      program_t safe_program = *program;
      if (safe_program.trusted_inner && !trusted_ready_ && !trusted_allocation_attempted_) {
        trusted_allocation_attempted_ = true;
        try { trusted_ready_ = prepare_trusted_inner(); }
        catch (const std::exception &) { trusted_ready_ = false; }
      }
      safe_program.trusted_inner = safe_program.trusted_inner && trusted_ready_;
      if (producer[30] != static_cast<UINT>(D3D11_SRV_DIMENSION_TEXTURE2D) || (producer[6] != static_cast<UINT>(DXGI_FORMAT_R8G8B8A8_UNORM) && producer[6] != static_cast<UINT>(DXGI_FORMAT_B8G8R8A8_UNORM))) {
        safe_program.producer.original_source = nullptr;
      }
      context->UpdateSubresource(producer_cb_.Get(), 0u, nullptr, producer.data(), 0u, 0u);
      if (!recorded_) {
        const saved_cs_t saved {context};
        emit(context, colour, graph, constants96, rgb, probability, support, ocr, root, root_offset, safe_program);
        ++diagnostics_.direct_emissions;
        last_enabled_ = true;
        return true;
      }
      std::array<IUnknown *, 96u> objects {};
      UINT count = 0u;
      auto signature_program = *program;
      signature_program.trusted_inner = safe_program.trusted_inner;
      if (!signature_objects(objects, count, colour, graph, constants96, rgb, probability, support, ocr, root, signature_program)) {
        return fail(error, "CBC recorded signature exceeds checked capacity96.");
      }
      if (!commands_ || !signature_.matches(objects, count, root_offset)) {
        commands_.Reset();
        signature_ = signature_t {};
        if (!deferred_ && FAILED(device_->CreateDeferredContext(0u, &deferred_))) {
          return fail(error, "Could not create enclosure deferred context.");
        }
        deferred_->ClearState();
        deferred_->SetPredication(nullptr, FALSE);
        emit(deferred_.Get(), colour, graph, constants96, rgb, probability, support, ocr, root, root_offset, safe_program);
        if (FAILED(deferred_->FinishCommandList(FALSE, &commands_))) {
          commands_.Reset();
          deferred_.Reset();
          return fail(error, "Could not finish enclosure command list.");
        }
        if (!signature_.retain(objects, count, root_offset)) {
          commands_.Reset();return fail(error, "CBC could not retain recorded signature96.");
        }
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

    static void split(std::array<UINT, 40u> &v, UINT at, std::uint64_t value) {
      v[at] = static_cast<UINT>(value);
      v[at + 1u] = static_cast<UINT>(value >> 32u);
    }

    std::array<UINT, 40u> producer_words(const producer_t &p) const {
      std::array<UINT, 40u> v {};
      v[0] = 0x31434e45u;
      v[1] = 1u;
      v[2] = 1u;
      v[3] = p.postprocess_mode;
      v[4] = width_;
      v[5] = height_;
      v[7] = static_cast<UINT>(p.transfer);
      split(v, 12u, p.transaction_token);
      v[14] = p.expected_work;
      v[15] = p.expected_work ? (p.expected_work ^ 0x6f435257u) : 0u;
      split(v, 16u, p.model_frame);
      split(v, 18u, p.model_observation_us);
      split(v, 20u, p.analysis_generation);
      v[22] = p.input_known ? 1u : 0u;
      for (UINT i = 0; i < 4u; ++i) {
        v[24u + i] = p.analysis_roi[i];
      }
      v[28] = p.direct_encoded_recipe;
      if (p.original_source) {
        D3D11_SHADER_RESOURCE_VIEW_DESC s {};
        p.original_source->GetDesc(&s);
        v[6] = static_cast<UINT>(s.Format);
        v[30] = static_cast<UINT>(s.ViewDimension);
        if (s.ViewDimension == D3D11_SRV_DIMENSION_TEXTURE2D) {
          v[29] = s.Texture2D.MostDetailedMip;
        }
        ComPtr<ID3D11Resource> resource;
        p.original_source->GetResource(&resource);
        ComPtr<ID3D11Texture2D> texture;
        if (SUCCEEDED(resource.As(&texture))) {
          D3D11_TEXTURE2D_DESC d {};
          texture->GetDesc(&d);
          v[8] = d.Width;
          v[9] = d.Height;
          v[10] = static_cast<UINT>(d.Format);
          v[11] = d.SampleDesc.Count;
          v[31] = d.BindFlags;
        }
      }
      return v;
    }

    bool make_uav(ID3D11ShaderResourceView *srv, ComPtr<ID3D11UnorderedAccessView> &uav, std::string &error) {
      if (!srv) {
        error = "Missing baseline working SRV.";
        return false;
      }
      ComPtr<ID3D11Resource> r;
      srv->GetResource(&r);
      if (FAILED(device_->CreateUnorderedAccessView(r.Get(), nullptr, &uav))) {
        error = "Could not bind baseline working UAV.";
        return false;
      }
      return true;
    }

    bool allocate(std::string &error) {
      base_views_ = base_.mutable_views();
      const std::array<ID3D11ShaderResourceView *, 12u> views {base_views_.eligible.Get(), base_views_.seeds.Get(), base_views_.domain_tag.Get(), base_views_.cover_clip.Get(), base_views_.mode_body.Get(), base_views_.mode_final.Get(), base_views_.body.Get(), base_views_.matte.Get(), base_views_.palette.Get(), base_views_.parent.Get(), base_views_.root_flags.Get(), base_views_.validation_receipt.Get()};
      for (UINT i = 0; i < views.size(); ++i) {
        if (!make_uav(views[i], base_uavs_[i], error)) {
          return false;
        }
      }
      D3D11_UNORDERED_ACCESS_VIEW_DESC raw {};
      raw.Format = DXGI_FORMAT_R32_TYPELESS;
      raw.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
      raw.Buffer.NumElements = 24u;
      raw.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
      if (FAILED(device_->CreateUnorderedAccessView(base_views_.current_mode_args.Get(), &raw, &mode_args_uav_))) {
        error = "Could not bind baseline raw arguments.";
        return false;
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
        D3D11_SUBRESOURCE_DATA init {zero.data(), width_ * 4u, 0u};
        return SUCCEEDED(device_->CreateTexture2D(&d, &init, &t.resource)) && SUCCEEDED(device_->CreateShaderResourceView(t.resource.Get(), nullptr, &t.srv)) && SUCCEEDED(device_->CreateUnorderedAccessView(t.resource.Get(), nullptr, &t.uav));
      };
      for (auto *t : {&all_body_, &cover_bits_, &selected_bits_, &candidate_body_, &candidate_matte_}) {
        if (!texture(*t)) {
          error = "Could not allocate enclosure native plane.";
          return false;
        }
      }
      D3D11_BUFFER_DESC meta {};
      meta.ByteWidth = 512u * 4u;
      meta.Usage = D3D11_USAGE_DEFAULT;
      meta.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      meta.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      meta.StructureByteStride = 4u;
      D3D11_SUBRESOURCE_DATA initial {zero.data(), 0u, 0u};
      if (FAILED(device_->CreateBuffer(&meta, &initial, &meta_.resource)) || FAILED(device_->CreateShaderResourceView(meta_.resource.Get(), nullptr, &meta_.srv)) || FAILED(device_->CreateUnorderedAccessView(meta_.resource.Get(), nullptr, &meta_.uav))) {
        error = "Could not allocate enclosure Meta512.";
        return false;
      }
      // u6 StageAux is a Texture2D<uint> in every compiled CBC entry. It aliases
      // this 17x8 model table only for begin/unit_meta; other entries keep Exterior.
      D3D11_TEXTURE2D_DESC family {};
      family.Width = 17u;family.Height = 8u;family.MipLevels = family.ArraySize = 1u;
      family.Format = DXGI_FORMAT_R32_UINT;family.SampleDesc.Count = 1u;
      family.Usage = D3D11_USAGE_DEFAULT;
      family.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      const std::array<UINT, 136u> family_zero {};
      D3D11_SUBRESOURCE_DATA family_init {family_zero.data(), 17u * 4u, 0u};
      if (FAILED(device_->CreateTexture2D(&family, &family_init, &family_models_.resource)) ||
          FAILED(device_->CreateShaderResourceView(family_models_.resource.Get(), nullptr, &family_models_.srv)) ||
          FAILED(device_->CreateUnorderedAccessView(family_models_.resource.Get(), nullptr, &family_models_.uav))) {
        error = "Could not allocate current per-unit family model texture17x8.";
        return false;
      }
      auto raw_buffer = [&](UINT bytes, UINT flags, buffer_t &b) {
        D3D11_BUFFER_DESC d {};
        d.ByteWidth = bytes;
        d.Usage = D3D11_USAGE_DEFAULT;
        d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS | flags;
        D3D11_SHADER_RESOURCE_VIEW_DESC s {};
        s.Format = DXGI_FORMAT_R32_TYPELESS;
        s.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
        s.BufferEx.NumElements = bytes / 4u;
        s.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
        D3D11_UNORDERED_ACCESS_VIEW_DESC u {};
        u.Format = DXGI_FORMAT_R32_TYPELESS;
        u.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        u.Buffer.NumElements = bytes / 4u;
        u.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
        return SUCCEEDED(device_->CreateBuffer(&d, &initial, &b.resource)) && SUCCEEDED(device_->CreateShaderResourceView(b.resource.Get(), &s, &b.srv)) && SUCCEEDED(device_->CreateUnorderedAccessView(b.resource.Get(), &u, &b.uav));
      };
      if (!raw_buffer(64u, D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS, unit_args_) || !raw_buffer(256u, 0u, root_mirror_)) {
        error = "Could not allocate enclosure argument/mirror storage.";
        return false;
      }
      auto cb = [&](UINT bytes, ComPtr<ID3D11Buffer> &b, const void *data) {
        D3D11_BUFFER_DESC d {};
        d.ByteWidth = bytes;
        d.Usage = D3D11_USAGE_DEFAULT;
        d.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA z {data, 0u, 0u};
        return SUCCEEDED(device_->CreateBuffer(&d, &z, &b));
      };
      const std::array<UINT, 4u> graph {width_, height_, pixels_, 0u};
      if (!cb(16u, hook_cb_, zero.data()) || !cb(16u, graph_cb_, graph.data()) || !cb(160u, producer_cb_, zero.data())) {
        error = "Could not allocate enclosure CBs.";
        return false;
      }
      const auto immutable_hook = [&](ComPtr<ID3D11Buffer> &buffer, const std::array<UINT, 4u> &value) {
        D3D11_BUFFER_DESC desc{};
        desc.ByteWidth = 16u;desc.Usage = D3D11_USAGE_IMMUTABLE;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        const D3D11_SUBRESOURCE_DATA data{value.data(), 0u, 0u};
        return SUCCEEDED(device_->CreateBuffer(&desc, &data, &buffer));
      };
      if (!immutable_hook(validation_cb_, {width_, height_, 0u, 0u})) {
        error = "Could not allocate immutable enclosure validation constants.";return false;
      }
      for (UINT unit = 0u; unit < 8u; ++unit) for (UINT mode = 0u; mode < 8u; ++mode) {
        if (!immutable_hook(mode_constants_[unit * 8u + mode], {unit, mode, 0u, 0u})) {
          error = "Could not allocate immutable enclosure mode constants.";return false;
        }
      }
      const std::array<UINT, 3u> args {(width_ + 15u) / 16u, (height_ + 15u) / 16u, 0u};
      D3D11_BUFFER_DESC d {};
      d.ByteWidth = 12u;
      d.Usage = D3D11_USAGE_DEFAULT;
      d.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      D3D11_SUBRESOURCE_DATA z {args.data(), 0u, 0u};
      if (FAILED(device_->CreateBuffer(&d, &z, &native_args_))) {
        error = "Could not allocate enclosure native arguments.";
        return false;
      }
      return true;
    }

    // Optional storage is prepared atomically, outside recording. Failure leaves the original
    // helper and all its allocations usable; no new mask is published from partial storage.
    bool prepare_trusted_inner() {
      texture_t inner;
      buffer_t palette, modes, publication;
      std::array<ComPtr<ID3D11Buffer>, 64u> requests;
      const std::vector<UINT> zero(static_cast<std::size_t>(pixels_) + 128u, 0u);
      D3D11_TEXTURE2D_DESC t{};
      t.Width = width_;t.Height = height_;t.MipLevels = t.ArraySize = t.SampleDesc.Count = 1u;
      t.Format = DXGI_FORMAT_R32_UINT;t.Usage = D3D11_USAGE_DEFAULT;
      t.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      D3D11_SUBRESOURCE_DATA ti{zero.data(), width_ * 4u, 0u};
      if (FAILED(device_->CreateTexture2D(&t, &ti, &inner.resource)) ||
          FAILED(device_->CreateShaderResourceView(inner.resource.Get(), nullptr, &inner.srv)) ||
          FAILED(device_->CreateUnorderedAccessView(inner.resource.Get(), nullptr, &inner.uav))) return false;
      D3D11_BUFFER_DESC b{};
      b.ByteWidth = 128u * 4u;b.Usage = D3D11_USAGE_DEFAULT;
      b.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      b.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;b.StructureByteStride = 4u;
      D3D11_SUBRESOURCE_DATA init{zero.data(), 0u, 0u};
      if (FAILED(device_->CreateBuffer(&b, &init, &palette.resource)) ||
          FAILED(device_->CreateShaderResourceView(palette.resource.Get(), nullptr, &palette.srv)) ||
          FAILED(device_->CreateUnorderedAccessView(palette.resource.Get(), nullptr, &palette.uav))) return false;
      const auto raw = [&](UINT bytes, buffer_t &out) {
        b.ByteWidth = bytes;b.StructureByteStride = 0u;
        b.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS | D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
        D3D11_UNORDERED_ACCESS_VIEW_DESC u{};
        u.Format = DXGI_FORMAT_R32_TYPELESS;u.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        u.Buffer.NumElements = bytes / 4u;u.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
        return SUCCEEDED(device_->CreateBuffer(&b, &init, &out.resource)) &&
               SUCCEEDED(device_->CreateUnorderedAccessView(out.resource.Get(), &u, &out.uav));
      };
      if (!raw(96u, modes) || !raw(16u, publication)) return false;
      b.ByteWidth = 16u;b.Usage = D3D11_USAGE_IMMUTABLE;
      b.BindFlags = D3D11_BIND_CONSTANT_BUFFER;b.MiscFlags = 0u;
      for (UINT unit = 0u; unit < 8u; ++unit) for (UINT mode = 0u; mode < 8u; ++mode) {
        const std::array<UINT, 4u> value{unit, mode, 1u, 0u};
        const D3D11_SUBRESOURCE_DATA data{value.data(), 0u, 0u};
        if (FAILED(device_->CreateBuffer(&b, &data, &requests[unit * 8u + mode]))) return false;
      }
      trusted_inner_ = std::move(inner);trusted_palette_ = std::move(palette);
      trusted_mode_args_ = std::move(modes);trusted_publication_args_ = std::move(publication);
      for (UINT i = 0u; i < requests.size(); ++i) mode_constants_[64u + i] = std::move(requests[i]);
      return true;
    }

    bool signature_objects(std::array<IUnknown *, 96u> &o, UINT &n, const std::array<ID3D11ComputeShader *, 5u> &c, const std::array<ID3D11ComputeShader *, 6u> &g, ID3D11Buffer *cb, ID3D11ShaderResourceView *rgb, ID3D11ShaderResourceView *p, ID3D11ShaderResourceView *s, ID3D11ShaderResourceView *ocr, ID3D11Buffer *root, const program_t &program) const {
      bool valid = true;
      auto add = [&](IUnknown *x) {
        if (n >= o.size()) { valid = false;return; }
        o[n++] = x;
      };
      for (auto *x : c) {
        add(x);
      }
      for (auto *x : g) {
        add(x);
      }
      for (auto *x : program.shaders) {
        add(x);
      }
      for (auto *x : program.roi_graph) {
        add(x);
      }
      for (auto *x : {rgb, p, s, ocr, program.current_sct, program.current_uip, program.producer.original_source}) {
        add(x);
      }
      add(cb);
      add(root);
      add(device_.Get());
      for (const auto &u : base_uavs_) {
        add(u.Get());
      }
      add(mode_args_uav_.Get());
      for (auto *x : {all_body_.srv.Get(), cover_bits_.srv.Get(), selected_bits_.srv.Get(), candidate_body_.srv.Get(), candidate_matte_.srv.Get(), meta_.srv.Get(), root_mirror_.srv.Get()}) {
        add(x);
      }
      for (auto *x : {all_body_.uav.Get(), cover_bits_.uav.Get(), selected_bits_.uav.Get(), candidate_body_.uav.Get(), candidate_matte_.uav.Get(), meta_.uav.Get(), unit_args_.uav.Get()}) {
        add(x);
      }
      add(producer_cb_.Get());
      add(hook_cb_.Get());
      add(validation_cb_.Get());
      // The private bank is created/replaced atomically with this owner; each
      // recorded binding also retains its actual buffer. One bank identity is sufficient.
      add(mode_constants_[0u].Get());
      add(graph_cb_.Get());
      add(native_args_.Get());
      add(unit_args_.resource.Get());
      add(root_mirror_.resource.Get());
      add(base_views_.current_mode_args.Get());
      add(base_views_.native_clear_args.Get());
      add(family_models_.resource.Get());add(family_models_.srv.Get());add(family_models_.uav.Get());
      // A null first identity records the optional route's disabled state in the command key.
      add(program.trusted_inner ? trusted_inner_.uav.Get() : nullptr);
      add(trusted_palette_.srv.Get());add(trusted_palette_.uav.Get());
      add(trusted_mode_args_.resource.Get());add(trusted_mode_args_.uav.Get());
      add(trusted_publication_args_.resource.Get());add(trusted_publication_args_.uav.Get());
      return valid;
    }

    // No Get*, Map, query, CPU colour decision or heap allocation is allowed in this command body.
    void emit(ID3D11DeviceContext *c, const std::array<ID3D11ComputeShader *, 5u> &colour, const std::array<ID3D11ComputeShader *, 6u> &graph, ID3D11Buffer *cb, ID3D11ShaderResourceView *rgb, ID3D11ShaderResourceView *p, ID3D11ShaderResourceView *s, ID3D11ShaderResourceView *ocr, ID3D11Buffer *root, UINT rootOffset, const program_t &program) {
      constexpr std::array<UINT, 4u> zero {};
      std::array<ID3D11UnorderedAccessView *, 8u> u {};
      unbind(c);
      const D3D11_BOX rootBox {0u, 0u, 0u, 256u, 1u, 1u};
      c->CopySubresourceRegion(root_mirror_.resource.Get(), 0u, 0u, 0u, 0u, root, 0u, &rootBox);
      const D3D11_BOX gate {rootOffset, 0u, 0u, rootOffset + 4u, 1u, 1u};
      c->CopySubresourceRegion(native_args_.Get(), 0u, 8u, 0u, 0u, root, 0u, &gate);
      c->CopySubresourceRegion(base_views_.native_clear_args.Get(), 0u, 8u, 0u, 0u, root, 0u, &gate);
      ID3D11Buffer *bc[3] {cb, validation_cb_.Get(), nullptr};
      const std::array<ID3D11ShaderResourceView *, 9u> bi {rgb, p, s, ocr, base_views_.palette.Get(), nullptr, nullptr, nullptr, base_views_.validation_receipt.Get()};
      c->ClearUnorderedAccessViewUint(base_uavs_[11].Get(), zero.data());
      c->CSSetConstantBuffers(0u, 3u, bc);
      auto vi = bi;
      vi[4] = nullptr;
      vi[8] = nullptr;
      c->CSSetShaderResources(0u, 9u, vi.data());
      u[0] = base_uavs_[11].Get();
      c->CSSetUnorderedAccessViews(0u, 8u, u.data(), nullptr);
      c->CSSetShader(colour[4], nullptr, 0u);
      c->DispatchIndirect(base_views_.native_clear_args.Get(), 0u);
      unbind(c);
      if (program.trusted_inner) {
        const std::array<UINT, 4u> publication {(width_ + 15u) / 16u, (height_ + 15u) / 16u, 0u, 0u};
        c->UpdateSubresource(trusted_publication_args_.resource.Get(), 0u, nullptr, publication.data(), 0u, 0u);
        ID3D11Buffer *trusted_constants[3] {cb, mode_constants_[64u].Get(), nullptr};
        u.fill(nullptr);u[6u] = trusted_inner_.uav.Get();
        c->CSSetConstantBuffers(0u, 3u, trusted_constants);
        c->CSSetUnorderedAccessViews(0u, 8u, u.data(), nullptr);
        c->CSSetShader(colour[2u], nullptr, 0u);
        c->DispatchIndirect(base_views_.native_clear_args.Get(), 0u);
        unbind(c);
      }
      const std::array<ID3D11ShaderResourceView *, 16u> ei {rgb, ocr, base_views_.validation_receipt.Get(), base_views_.palette.Get(), base_views_.mode_body.Get(), base_views_.cover_clip.Get(), base_views_.parent.Get(), base_views_.root_flags.Get(), all_body_.srv.Get(), cover_bits_.srv.Get(), program.current_sct, program.current_uip, root_mirror_.srv.Get(), program.producer.original_source, base_views_.body.Get(), base_views_.matte.Get()};
      ID3D11Buffer *ec[3] {cb, validation_cb_.Get(), producer_cb_.Get()};
      auto enc = [&](UINT entry, const std::array<ID3D11UnorderedAccessView *, 8u> &out, ID3D11Buffer *args, UINT offset, bool bits_write = false, bool scratch_write = false) {
        // emit() enters unbound. Every preceding stage or shared mode hook
        // clears its writable/read slots before returning to this owner.
        auto in = ei;
        auto bound_out = out;
        if (entry == 0u || entry == 3u) bound_out[6] = family_models_.uav.Get();
        if (entry == 4u) {
          // collect_main reconstructs from this exact current per-mode Seeds plane.
          // publish_main continues to read accumulated baseline BODY at t14.
          in[14] = base_views_.seeds.Get();
        }
        if (entry == 2u) {
          // Dedicated input validation aliases: t1=actual P bits, t5=current S.
          in[1] = p;
          in[5] = s;
        }
        if (bits_write) {
          in[8] = nullptr;
          in[9] = nullptr;
        }
        if (bound_out[6]) {
          in[4] = nullptr;
        }
        if (entry == 10u) {
          // CBC4 normalizes the own-pixel pending planes in the shared emit body.
          // Remove the SRV alias before binding the same AllBody storage as u1.
          in[8] = nullptr;
          bound_out[1] = all_body_.uav.Get();
        }
        if (entry == 8u) {
          // Resolve needs passable Eligible at t4, not its Exterior output (ModeBody/u6).
          in[4] = base_views_.eligible.Get();
        }
        if (scratch_write) {
          in[5] = nullptr;
        }
        c->CSSetConstantBuffers(0u, 3u, ec);
        c->CSSetShaderResources(0u, 16u, in.data());
        c->CSSetUnorderedAccessViews(0u, 8u, bound_out.data(), nullptr);
        c->CSSetShader(program.shaders[entry], nullptr, 0u);
        c->DispatchIndirect(args, offset);
        unbind(c);
      };
      u.fill(nullptr);
      u[0] = meta_.uav.Get();
      enc(0u, u, root, rootOffset);
      u[1] = all_body_.uav.Get();
      u[2] = cover_bits_.uav.Get();
      u[3] = selected_bits_.uav.Get();
      u[4] = candidate_body_.uav.Get();
      u[5] = candidate_matte_.uav.Get();
      enc(1u, u, native_args_.Get(), 0u, true);
      u.fill(nullptr);
      u[0] = meta_.uav.Get();
      enc(2u, u, native_args_.Get(), 0u);
      // Original baseline global clear/palette/emit/4CC/BOX3/OR: no replay or changed shader.
      c->CSSetConstantBuffers(0u, 3u, bc);
      u.fill(nullptr);
      u[5] = base_uavs_[6].Get();
      u[6] = base_uavs_[7].Get();
      c->CSSetUnorderedAccessViews(0u, 8u, u.data(), nullptr);
      c->CSSetShader(colour[2], nullptr, 0u);
      c->DispatchIndirect(base_views_.native_clear_args.Get(), 0u);
      unbind(c);
      const std::array<ID3D11ShaderResourceView *, 4u> gi {base_views_.eligible.Get(), base_views_.seeds.Get(), base_views_.domain_tag.Get(), base_views_.cover_clip.Get()};
      namespace colour_commands = host_sbs_subtitle_colour_commands;
      colour_commands::bindings_t bindings;
      bindings.request_constants = hook_cb_.Get();
      for (UINT i = 0u; i < mode_constants_.size(); ++i) bindings.immutable_requests[i] = mode_constants_[i].Get();
      bindings.mode_arguments = base_views_.current_mode_args.Get();
      bindings.root = root;
      bindings.root_offset = rootOffset;
      bindings.colour_constants = {cb, hook_cb_.Get(), nullptr};
      bindings.graph_constants = {graph_cb_.Get(), nullptr, nullptr};
      bindings.colour_inputs = bi;
      bindings.component_inputs = gi;
      bindings.accumulation_views = {base_views_.mode_body.Get(), base_views_.mode_final.Get(), base_views_.parent.Get()};
      bindings.palette_outputs = {base_uavs_[8].Get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, mode_args_uav_.Get()};
      bindings.emission_outputs = {nullptr, base_uavs_[0].Get(), base_uavs_[1].Get(), base_uavs_[2].Get(), base_uavs_[3].Get(), nullptr, nullptr, nullptr};
      bindings.component_outputs = {base_uavs_[9].Get(), base_uavs_[10].Get(), base_uavs_[4].Get(), base_uavs_[5].Get(), nullptr, nullptr, nullptr, nullptr};
      bindings.accumulation_outputs = {nullptr, nullptr, nullptr, nullptr, nullptr, base_uavs_[6].Get(), base_uavs_[7].Get(), nullptr};
      if (program.trusted_inner) {
        bindings.trusted_palette = trusted_palette_.srv.Get();
        bindings.trusted_palette_output = trusted_palette_.uav.Get();
        bindings.trusted_mode_arguments = trusted_mode_args_.resource.Get();
        bindings.trusted_mode_output = trusted_mode_args_.uav.Get();
        bindings.trusted_inner_output = trusted_inner_.uav.Get();
        bindings.trusted_publication_output = trusted_publication_args_.uav.Get();
      }
      const auto collect_unit = [&](UINT unit) {
        ec[1u] = mode_constants_[unit * 8u].Get();
        std::array<ID3D11UnorderedAccessView *, 8u> outputs{};
        outputs[0] = meta_.uav.Get();
        enc(3u, outputs, root, rootOffset);
      };
      const auto collect_mode = [&](UINT unit, UINT mode, UINT argument_offset) {
        ec[1u] = mode_constants_[unit * 8u + mode].Get();
        std::array<ID3D11UnorderedAccessView *, 8u> outputs{};
        outputs[0] = meta_.uav.Get();
        outputs[1] = all_body_.uav.Get();
        outputs[2] = cover_bits_.uav.Get();
        outputs[3] = selected_bits_.uav.Get();
        enc(4u, outputs, base_views_.current_mode_args.Get(), argument_offset, true);
      };
      colour_commands::emit_modes<colour_commands::enclosure_layout_t>(
          c, colour, graph, bindings, &resources_t::unbind, collect_unit, collect_mode);
      u.fill(nullptr);
      u[1] = base_uavs_[0].Get();
      u[2] = base_uavs_[1].Get();
      u[3] = base_uavs_[2].Get();
      u[6] = base_uavs_[3].Get();
      enc(5u, u, native_args_.Get(), 0u, false, true);
      for (UINT slot = 0; slot < 8u; ++slot) {
        ec[1u] = mode_constants_[slot * 8u].Get();
        c->ClearUnorderedAccessViewUint(unit_args_.uav.Get(), zero.data());
        u.fill(nullptr);
        u[0] = meta_.uav.Get();
        u[7] = unit_args_.uav.Get();
        enc(6u, u, root, rootOffset);
        u.fill(nullptr);
        u[0] = meta_.uav.Get();
        u[1] = base_uavs_[0].Get();
        u[2] = base_uavs_[1].Get();
        u[3] = base_uavs_[2].Get();
        u[6] = base_uavs_[3].Get();
        enc(7u, u, unit_args_.resource.Get(), 0u, false, true);
        ID3D11Buffer *rc[3] {graph_cb_.Get(), mode_constants_[slot * 8u].Get(), nullptr};
        std::array<ID3D11ShaderResourceView *, 16u> ri {};
        for (UINT k = 0; k < 4u; ++k) {
          ri[k] = gi[k];
        }
        ri[4] = meta_.srv.Get();
        ri[5] = s; // Current support is evidence only for the CBC physical graph.
        c->CSSetConstantBuffers(0u, 3u, rc);
        c->CSSetShaderResources(0u, 16u, ri.data());
        u.fill(nullptr);
        u[0] = base_uavs_[9].Get();
        u[1] = base_uavs_[10].Get();
        u[2] = base_uavs_[4].Get();
        u[3] = base_uavs_[5].Get();
        c->CSSetUnorderedAccessViews(0u, 8u, u.data(), nullptr);
        for (auto *shader : program.roi_graph) {
          c->CSSetShader(shader, nullptr, 0u);
          c->DispatchIndirect(unit_args_.resource.Get(), 0u);
        }
        unbind(c);
        u.fill(nullptr);
        u[0] = meta_.uav.Get();
        u[6] = base_uavs_[4].Get();
        enc(8u, u, unit_args_.resource.Get(), 0u);
        u.fill(nullptr);
        u[0] = meta_.uav.Get();
        enc(9u, u, unit_args_.resource.Get(), 0u);
        u[3] = selected_bits_.uav.Get();
        enc(10u, u, unit_args_.resource.Get(), 16u);
      }
      u.fill(nullptr);
      u[0] = meta_.uav.Get();
      u[3] = selected_bits_.uav.Get();
      u[4] = candidate_body_.uav.Get();
      u[5] = candidate_matte_.uav.Get();
      enc(11u, u, native_args_.Get(), 0u);
      unbind(c);
    }

    bool ready_ = false, recorded_ = true, last_enabled_ = false, trusted_ready_ = false, trusted_allocation_attempted_ = false;
    UINT width_ = 0u, height_ = 0u, pixels_ = 0u;
    ComPtr<ID3D11Device> device_;
    baseline::resources_t base_;
    baseline::mutable_views_t base_views_;
    std::array<ComPtr<ID3D11UnorderedAccessView>, 12u> base_uavs_;
    ComPtr<ID3D11UnorderedAccessView> mode_args_uav_;
    texture_t all_body_, cover_bits_, selected_bits_, candidate_body_, candidate_matte_, family_models_, trusted_inner_;
    buffer_t meta_, unit_args_, root_mirror_, trusted_palette_, trusted_mode_args_, trusted_publication_args_;
    ComPtr<ID3D11Buffer> producer_cb_, hook_cb_, graph_cb_, native_args_;
    ComPtr<ID3D11Buffer> validation_cb_;
    std::array<ComPtr<ID3D11Buffer>, 128u> mode_constants_;
    host_sbs_compute_program_lease::group_t<5u> colour_program_lease_;
    host_sbs_compute_program_lease::group_t<6u> graph_program_lease_;
    host_sbs_compute_program_lease::group_t<12u> enclosure_program_lease_;
    host_sbs_compute_program_lease::group_t<3u> roi_program_lease_;
    ComPtr<ID3D11DeviceContext> deferred_;
    ComPtr<ID3D11CommandList> commands_;
    signature_t signature_;
    diagnostics_t diagnostics_;
  };
}  // namespace models::host_sbs_current_black_enclosure_gpu_v1
#endif
