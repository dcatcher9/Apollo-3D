#pragma once

// Private recipe5 analysis source. The real display snapshot is already tone mapped.
// Only its current lower band is written; no OCR, UI, EMPTY or renderer authority.
#ifdef _WIN32
  #include "host_sbs_compute_program_lease.h"
  #include <array>
  #include <cstdint>
  #include <d3d11.h>
  #include <string>
  #include <utility>
  #include <wrl/client.h>

namespace models::display_snapshot_rgb8_analysis_gpu_v1 {
  using Microsoft::WRL::ComPtr;
  inline constexpr UINT tag = 0x35534743u, schema = 1u, recipe = 5u;
  inline constexpr UINT proof_words = 64u, uip_words = 40u, envelope_words = 104u;
  inline constexpr UINT request_words_count = 36u, maximum_roi = 5120u;

  struct program_t {
    ID3D11ComputeShader *prepare = nullptr, *copy = nullptr, *finalize = nullptr;
    ID3D11ComputeShader *envelope = nullptr;
  };

  struct request_t {
    ID3D11ShaderResourceView *display_snapshot_band = nullptr;
    ID3D11ShaderResourceView *parent_raw_source = nullptr;
    ID3D11Buffer *actual_contour_constants96 = nullptr;
    ID3D11ShaderResourceView *actual_sct32 = nullptr;
    ID3D11Buffer *actual_final256 = nullptr;
    std::array<UINT, 4u> roi_ltrb {};
    std::uint64_t frame = 0u, observation_us = 0u, analysis_generation = 0u, token = 0u;
    UINT transfer_mode = 0u;  // Actual pending parent mode0/1/2; never applied a second time.
  };

  struct views_t {
    ComPtr<ID3D11ShaderResourceView> reference_rgb8, analysis_band, source_proof, uip_envelope;
    ComPtr<ID3D11Buffer> dispatch_arguments, final_root_mirror, request_constants;
  };

  class resources_t {
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11Texture2D> reference_, band_;
    ComPtr<ID3D11ShaderResourceView> reference_srv_, band_srv_, proof_srv_, root_srv_, envelope_srv_;
    ComPtr<ID3D11UnorderedAccessView> reference_uav_, band_uav_, proof_uav_, args_uav_, envelope_uav_;
    ComPtr<ID3D11Buffer> proof_, root_, args_, constants_, envelope_;
    ComPtr<ID3D11ShaderResourceView> submitted_display_, submitted_raw_, submitted_sct_;
    ComPtr<ID3D11Buffer> submitted_cb_, submitted_final_;
    std::array<UINT, request_words_count> submitted_words_ {};
    UINT width_ = 0u, full_height_ = 0u, crop_top_ = 0u, band_height_ = 0u;
    host_sbs_compute_program_lease::group_t<3u> capture_program_lease_;
    host_sbs_compute_program_lease::group_t<1u> envelope_program_lease_;

    struct saved_t {
      ID3D11DeviceContext *context;
      std::array<ComPtr<ID3D11ShaderResourceView>, 16u> srvs;
      std::array<ComPtr<ID3D11UnorderedAccessView>, 8u> uavs;
      std::array<ComPtr<ID3D11Buffer>, 3u> cbs;
      ComPtr<ID3D11ComputeShader> shader;
      std::array<ComPtr<ID3D11ClassInstance>, 256u> instances;
      UINT instance_count = 256u;

      explicit saved_t(ID3D11DeviceContext *c):
          context(c) {
        ID3D11ShaderResourceView *s[16] {};
        ID3D11UnorderedAccessView *u[8] {};
        ID3D11Buffer *b[3] {};
        ID3D11ComputeShader *sh = nullptr;
        ID3D11ClassInstance *ci[256] {};
        c->CSGetShaderResources(0u, 16u, s);
        c->CSGetUnorderedAccessViews(0u, 8u, u);
        c->CSGetConstantBuffers(0u, 3u, b);
        c->CSGetShader(&sh, ci, &instance_count);
        for (UINT i = 0u; i < 16u; ++i) {
          srvs[i].Attach(s[i]);
        }
        for (UINT i = 0u; i < 8u; ++i) {
          uavs[i].Attach(u[i]);
        }
        for (UINT i = 0u; i < 3u; ++i) {
          cbs[i].Attach(b[i]);
        }
        for (UINT i = 0u; i < instance_count; ++i) {
          instances[i].Attach(ci[i]);
        }
        shader.Attach(sh);
      }

      static void unbind(ID3D11DeviceContext *c) {
        ID3D11ShaderResourceView *s[16] {};
        ID3D11UnorderedAccessView *u[8] {};
        ID3D11Buffer *b[3] {};
        c->CSSetShaderResources(0u, 16u, s);
        c->CSSetUnorderedAccessViews(0u, 8u, u, nullptr);
        c->CSSetConstantBuffers(0u, 3u, b);
      }

      ~saved_t() {
        unbind(context);
        ID3D11ShaderResourceView *s[16] {};
        ID3D11UnorderedAccessView *u[8] {};
        ID3D11Buffer *b[3] {};
        ID3D11ClassInstance *ci[256] {};
        for (UINT i = 0u; i < 16u; ++i) {
          s[i] = srvs[i].Get();
        }
        for (UINT i = 0u; i < 8u; ++i) {
          u[i] = uavs[i].Get();
        }
        for (UINT i = 0u; i < 3u; ++i) {
          b[i] = cbs[i].Get();
        }
        for (UINT i = 0u; i < instance_count; ++i) {
          ci[i] = instances[i].Get();
        }
        context->CSSetShaderResources(0u, 16u, s);
        context->CSSetUnorderedAccessViews(0u, 8u, u, nullptr);
        context->CSSetConstantBuffers(0u, 3u, b);
        context->CSSetShader(shader.Get(), ci, instance_count);
      }
    };

    bool same_device(ID3D11DeviceChild *child) const {
      if (!child || !device_) {
        return false;
      }
      ComPtr<ID3D11Device> d;
      child->GetDevice(&d);
      return d.Get() == device_.Get();
    }

    bool context_ok(ID3D11DeviceContext *c) const {
      if (!same_device(c) || c->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE) {
        return false;
      }
      ComPtr<ID3D11Predicate> predicate;
      BOOL value = FALSE;
      c->GetPredication(&predicate, &value);
      return !predicate;
    }

    bool structured(UINT words, ComPtr<ID3D11Buffer> &buffer, ComPtr<ID3D11ShaderResourceView> &srv, ComPtr<ID3D11UnorderedAccessView> &uav) {
      D3D11_BUFFER_DESC d {};
      d.ByteWidth = words * 4u;
      d.Usage = D3D11_USAGE_DEFAULT;
      d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      d.StructureByteStride = 4u;
      std::array<UINT, envelope_words> zero {};
      D3D11_SUBRESOURCE_DATA data {zero.data(), 0u, 0u};
      return SUCCEEDED(device_->CreateBuffer(&d, &data, buffer.GetAddressOf())) &&
             SUCCEEDED(device_->CreateShaderResourceView(buffer.Get(), nullptr, srv.GetAddressOf())) &&
             SUCCEEDED(device_->CreateUnorderedAccessView(buffer.Get(), nullptr, uav.GetAddressOf()));
    }

    bool structured_view(ID3D11ShaderResourceView *view, UINT words) const {
      if (!same_device(view)) {
        return false;
      }
      D3D11_SHADER_RESOURCE_VIEW_DESC s {};
      view->GetDesc(&s);
      ComPtr<ID3D11Resource> r;
      ComPtr<ID3D11Buffer> b;
      view->GetResource(&r);
      if (FAILED(r.As(&b))) {
        return false;
      }
      D3D11_BUFFER_DESC d {};
      b->GetDesc(&d);
      return s.ViewDimension == D3D11_SRV_DIMENSION_BUFFER && s.Format == DXGI_FORMAT_UNKNOWN &&
             s.Buffer.FirstElement == 0u && s.Buffer.NumElements == words && d.ByteWidth == words * 4u &&
             d.StructureByteStride == 4u && (d.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED) != 0u;
    }

    bool source_texture(ID3D11ShaderResourceView *view, D3D11_SHADER_RESOURCE_VIEW_DESC &s, D3D11_TEXTURE2D_DESC &d, ComPtr<ID3D11Texture2D> &texture) const {
      if (!same_device(view)) {
        return false;
      }
      view->GetDesc(&s);
      ComPtr<ID3D11Resource> resource;
      view->GetResource(&resource);
      if (FAILED(resource.As(&texture))) {
        return false;
      }
      texture->GetDesc(&d);
      return s.ViewDimension == D3D11_SRV_DIMENSION_TEXTURE2D && s.Format != DXGI_FORMAT_UNKNOWN &&
             s.Texture2D.MostDetailedMip == 0u && (s.Texture2D.MipLevels == 1u || s.Texture2D.MipLevels == ~0u) &&
             d.Format != DXGI_FORMAT_UNKNOWN && d.MipLevels == 1u && d.ArraySize == 1u &&
             d.SampleDesc.Count == 1u && d.SampleDesc.Quality == 0u &&
             (d.BindFlags & D3D11_BIND_SHADER_RESOURCE) != 0u;
    }

    bool request_words(const request_t &request, std::array<UINT, request_words_count> &cb, std::string &error) const {
      if (!reference_ || !same_device(request.actual_contour_constants96) || !same_device(request.actual_final256) || !structured_view(request.actual_sct32, 32u) || !request.frame || !request.observation_us || !request.token || request.transfer_mode > 2u) {
        error = "Display-snapshot source metadata unavailable";
        return false;
      }
      D3D11_BUFFER_DESC final_desc {}, contour_desc {};
      request.actual_final256->GetDesc(&final_desc);
      request.actual_contour_constants96->GetDesc(&contour_desc);
      D3D11_SHADER_RESOURCE_VIEW_DESC raw_s {}, snapshot_s {};
      D3D11_TEXTURE2D_DESC raw_d {}, snapshot_d {};
      ComPtr<ID3D11Texture2D> raw_texture, snapshot_texture;
      if (!source_texture(request.parent_raw_source, raw_s, raw_d, raw_texture) || !source_texture(request.display_snapshot_band, snapshot_s, snapshot_d, snapshot_texture)) {
        error = "Display-snapshot inputs must be exact single-mip textures";
        return false;
      }
      const auto &r = request.roi_ltrb;
      if (final_desc.ByteWidth != 256u || !(final_desc.MiscFlags & D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS) || contour_desc.ByteWidth != 96u || !(contour_desc.BindFlags & D3D11_BIND_CONSTANT_BUFFER) || request.actual_final256 == proof_.Get() || request.actual_final256 == root_.Get() || request.actual_final256 == args_.Get() || request.actual_final256 == envelope_.Get() || raw_texture.Get() == reference_.Get() || raw_texture.Get() == band_.Get() || snapshot_texture.Get() == reference_.Get() || snapshot_texture.Get() == band_.Get() || raw_texture.Get() == snapshot_texture.Get() || snapshot_s.Format != DXGI_FORMAT_R16G16B16A16_FLOAT || snapshot_d.Format != DXGI_FORMAT_R16G16B16A16_FLOAT || snapshot_d.Width != width_ || snapshot_d.Height != band_height_ || r[0] >= r[2] || r[1] >= r[3] || r[2] > raw_d.Width || r[3] > raw_d.Height || r[2] - r[0] != width_ || r[3] - r[1] != full_height_) {
        error = "Display-snapshot descriptor, ROI or writable alias unsupported";
        return false;
      }
      cb = {
        tag,
        schema,
        recipe,
        request.transfer_mode,
        width_,
        full_height_,
        crop_top_,
        band_height_,
        r[0],
        r[1],
        r[2],
        r[3],
        static_cast<UINT>(request.frame),
        static_cast<UINT>(request.frame >> 32u),
        static_cast<UINT>(request.observation_us),
        static_cast<UINT>(request.observation_us >> 32u),
        static_cast<UINT>(request.analysis_generation),
        static_cast<UINT>(request.analysis_generation >> 32u),
        static_cast<UINT>(request.token),
        static_cast<UINT>(request.token >> 32u),
        raw_d.Width,
        raw_d.Height,
        static_cast<UINT>(raw_s.Format),
        static_cast<UINT>(raw_d.Format),
        raw_d.BindFlags,
        static_cast<UINT>(raw_s.ViewDimension),
        raw_d.MipLevels,
        raw_d.ArraySize,
        raw_d.SampleDesc.Count,
        raw_d.SampleDesc.Quality,
        static_cast<UINT>(snapshot_s.Format),
        static_cast<UINT>(snapshot_d.Format),
        0u,
        0u,
        0u,
        0u,
      };
      return true;
    }

    void emit_capture(ID3D11DeviceContext *context, const program_t &program,
                      const request_t &request, const std::array<UINT, request_words_count> &cb) {
      context->CopyResource(root_.Get(), request.actual_final256);
      context->UpdateSubresource(constants_.Get(), 0u, nullptr, cb.data(), 0u, 0u);
      ID3D11Buffer *buffers[2] {request.actual_contour_constants96, constants_.Get()};
      ID3D11ShaderResourceView *srvs[6] {request.display_snapshot_band, root_srv_.Get(), request.actual_sct32, nullptr, nullptr, request.parent_raw_source};
      ID3D11UnorderedAccessView *uavs[4] {reference_uav_.Get(), band_uav_.Get(), proof_uav_.Get(), args_uav_.Get()};
      context->CSSetConstantBuffers(0u, 2u, buffers);
      context->CSSetShaderResources(0u, 6u, srvs);
      context->CSSetUnorderedAccessViews(0u, 4u, uavs, nullptr);
      context->CSSetShader(program.prepare, nullptr, 0u);
      context->Dispatch(1u, 1u, 1u);
      ID3D11UnorderedAccessView *null_args = nullptr;
      context->CSSetUnorderedAccessViews(3u, 1u, &null_args, nullptr);
      context->CSSetShader(program.copy, nullptr, 0u);
      context->DispatchIndirect(args_.Get(), 0u);
      context->CSSetShader(program.finalize, nullptr, 0u);
      context->DispatchIndirect(request.actual_final256, 240u);
      submitted_display_ = request.display_snapshot_band;
      submitted_raw_ = request.parent_raw_source;
      submitted_sct_ = request.actual_sct32;
      submitted_cb_ = request.actual_contour_constants96;
      submitted_final_ = request.actual_final256;
      submitted_words_ = cb;
    }

    void emit_envelope(ID3D11DeviceContext *context, const program_t &program,
                       ID3D11ShaderResourceView *actual_uip40, const request_t &request,
                       const std::array<UINT, request_words_count> &cb, bool rejoin) {
      if (rejoin) {
        // Public separate consumption must rejoin bytes, not trust pointer identity.
        context->CopyResource(root_.Get(), request.actual_final256);
        context->UpdateSubresource(constants_.Get(), 0u, nullptr, cb.data(), 0u, 0u);
      }
      ID3D11Buffer *buffers[2] {request.actual_contour_constants96, constants_.Get()};
      ID3D11ShaderResourceView *srvs[6] {request.display_snapshot_band, root_srv_.Get(), request.actual_sct32, actual_uip40, proof_srv_.Get(), request.parent_raw_source};
      context->CSSetConstantBuffers(0u, 2u, buffers);
      context->CSSetShaderResources(0u, 6u, srvs);
      context->CSSetUnorderedAccessViews(4u, 1u, envelope_uav_.GetAddressOf(), nullptr);
      context->CSSetShader(program.envelope, nullptr, 0u);
      context->DispatchIndirect(request.actual_final256, 240u);
    }

  public:
    resources_t() = default;
    resources_t(const resources_t &) = delete;
    resources_t &operator=(const resources_t &) = delete;
    resources_t(resources_t &&) = default;
    resources_t &operator=(resources_t &&) = default;

    bool ensure(ID3D11Device *device, UINT width, UINT full_height, UINT crop_top, UINT band_height, std::string &error) {
      error.clear();
      if (!device || !width || !full_height || width > maximum_roi || full_height > maximum_roi || crop_top >= full_height || band_height != full_height - crop_top) {
        error = "Display-snapshot analysis shape unavailable";
        return false;
      }
      if (device_.Get() == device && width_ == width && full_height_ == full_height && crop_top_ == crop_top && band_height_ == band_height && reference_) {
        return true;
      }
      resources_t next;
      next.device_ = device;
      next.width_ = width;
      next.full_height_ = full_height;
      next.crop_top_ = crop_top;
      next.band_height_ = band_height;
      auto texture = [&](UINT h, DXGI_FORMAT format, ComPtr<ID3D11Texture2D> &t, ComPtr<ID3D11ShaderResourceView> &s, ComPtr<ID3D11UnorderedAccessView> &u) {
        D3D11_TEXTURE2D_DESC d {};
        d.Width = width;
        d.Height = h;
        d.MipLevels = d.ArraySize = d.SampleDesc.Count = 1u;
        d.Format = format;
        d.Usage = D3D11_USAGE_DEFAULT;
        d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        return SUCCEEDED(device->CreateTexture2D(&d, nullptr, t.GetAddressOf())) &&
               SUCCEEDED(device->CreateShaderResourceView(t.Get(), nullptr, s.GetAddressOf())) &&
               SUCCEEDED(device->CreateUnorderedAccessView(t.Get(), nullptr, u.GetAddressOf()));
      };
      if (!texture(full_height, DXGI_FORMAT_R8G8B8A8_UNORM, next.reference_, next.reference_srv_, next.reference_uav_) || !texture(band_height, DXGI_FORMAT_R16G16B16A16_FLOAT, next.band_, next.band_srv_, next.band_uav_) || !next.structured(proof_words, next.proof_, next.proof_srv_, next.proof_uav_) || !next.structured(envelope_words, next.envelope_, next.envelope_srv_, next.envelope_uav_)) {
        error = "Display-snapshot analysis allocation failed";
        return false;
      }
      D3D11_BUFFER_DESC d {};
      d.ByteWidth = 256u;
      d.Usage = D3D11_USAGE_DEFAULT;
      d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
      D3D11_SHADER_RESOURCE_VIEW_DESC rs {};
      rs.Format = DXGI_FORMAT_R32_TYPELESS;
      rs.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
      rs.BufferEx.NumElements = 64u;
      rs.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
      if (FAILED(device->CreateBuffer(&d, nullptr, next.root_.GetAddressOf())) || FAILED(device->CreateShaderResourceView(next.root_.Get(), &rs, next.root_srv_.GetAddressOf()))) {
        error = "Display-snapshot final-root mirror allocation failed";
        return false;
      }
      d = {};
      d.ByteWidth = 16u;
      d.Usage = D3D11_USAGE_DEFAULT;
      d.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
      d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS | D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      D3D11_UNORDERED_ACCESS_VIEW_DESC au {};
      au.Format = DXGI_FORMAT_R32_TYPELESS;
      au.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
      au.Buffer.NumElements = 4u;
      au.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
      const std::array<UINT, 4u> zero {};
      D3D11_SUBRESOURCE_DATA z {zero.data(), 0u, 0u};
      if (FAILED(device->CreateBuffer(&d, &z, next.args_.GetAddressOf())) || FAILED(device->CreateUnorderedAccessView(next.args_.Get(), &au, next.args_uav_.GetAddressOf()))) {
        error = "Display-snapshot copy arguments allocation failed";
        return false;
      }
      d = {};
      d.ByteWidth = request_words_count * 4u;
      d.Usage = D3D11_USAGE_DEFAULT;
      d.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      if (FAILED(device->CreateBuffer(&d, nullptr, next.constants_.GetAddressOf()))) {
        error = "Display-snapshot request constants allocation failed";
        return false;
      }
      *this = std::move(next);
      return true;
    }

    // Serialized immediate owner. true means submitted, not ready. Caller retains the real
    // pending source/snapshot/CB/SCT/final tuple through consumption before any next queue/ensure
    // or mutation of those resources. Operands must be unbound outside saved CS slots.
    // No CPU completion or source-pixel decisions are used.
    bool queue(ID3D11DeviceContext *context, const program_t &program, const request_t &request, std::string &error) {
      error.clear();
      if (!reference_ || !context_ok(context) || !capture_program_lease_.acquire(device_.Get(), {program.prepare, program.copy, program.finalize})) {
        error = "Display-snapshot source program unavailable";
        return false;
      }
      std::array<UINT, request_words_count> cb {};
      if (!request_words(request, cb, error)) {
        return false;
      }
      saved_t saved(context);
      saved_t::unbind(context);
      emit_capture(context, program, request, cb);
      return true;
    }

    // Actual UIP40 is copied literally. Appended proof64 only qualifies a private source;
    // no UIP, model-completion or EMPTY authority is authored. Root0 holds the whole envelope.
    bool queue_uip_envelope(ID3D11DeviceContext *context, const program_t &program, ID3D11ShaderResourceView *actual_uip40, const request_t &accepted_pending, std::string &error) {
      error.clear();
      if (!reference_ || !context_ok(context) || !envelope_program_lease_.acquire(device_.Get(), {program.envelope}) || !structured_view(actual_uip40, uip_words)) {
        error = "Display-snapshot UIP envelope unavailable";
        return false;
      }
      std::array<UINT, request_words_count> cb {};
      if (accepted_pending.display_snapshot_band != submitted_display_.Get() || accepted_pending.parent_raw_source != submitted_raw_.Get() || accepted_pending.actual_sct32 != submitted_sct_.Get() || accepted_pending.actual_contour_constants96 != submitted_cb_.Get() || accepted_pending.actual_final256 != submitted_final_.Get() || !request_words(accepted_pending, cb, error) || cb != submitted_words_) {
        error = "Display-snapshot pending tuple or source lease mismatch";
        return false;
      }
      saved_t saved(context);
      saved_t::unbind(context);
      emit_envelope(context, program, actual_uip40, accepted_pending, cb, true);
      return true;
    }

    // One caller-owned immutable queue interval. Both public operations keep
    // their independent checks above; this path validates the complete tuple
    // before submission and captures root/constants once for both consumers.
    bool queue_current(ID3D11DeviceContext *context, const program_t &program,
                       const request_t &request, ID3D11ShaderResourceView *actual_uip40,
                       std::string &error) {
      error.clear();
      if (!reference_ || !context_ok(context) ||
          !capture_program_lease_.acquire(device_.Get(), {program.prepare, program.copy, program.finalize}) ||
          !envelope_program_lease_.acquire(device_.Get(), {program.envelope}) ||
          !structured_view(actual_uip40, uip_words)) {
        error = "Display-snapshot complete source program/UIP unavailable";return false;
      }
      std::array<UINT, request_words_count> cb{};
      if (!request_words(request, cb, error)) return false;
      saved_t saved(context);
      saved_t::unbind(context);
      emit_capture(context, program, request, cb);
      saved_t::unbind(context); // proof changes from UAV producer to SRV consumer.
      emit_envelope(context, program, actual_uip40, request, cb, false);
      return true;
    }

    views_t views() const {
      return {reference_srv_, band_srv_, proof_srv_, envelope_srv_, args_, root_, constants_};
    }
  };
}  // namespace models::display_snapshot_rgb8_analysis_gpu_v1
#endif
