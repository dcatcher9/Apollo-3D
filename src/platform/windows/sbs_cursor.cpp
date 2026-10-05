// SPDX-License-Identifier: GPL-3.0-only
#include "sbs_cursor.h"

#include "src/reshade_bridge_protocol.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <d3d11_1.h>
#include <utility>
#include <wrl/client.h>

namespace platf::sbs_cursor {
  namespace {
    using Microsoft::WRL::ComPtr;

    bool valid_shape(const shape_t &shape) {
      if (!shape.width || !shape.height || shape.width > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION || shape.height > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION) {
        return false;
      }
      const auto bytes = std::size_t(shape.width) * shape.height * 4;
      return (shape.alpha_bgra.empty() || shape.alpha_bgra.size() == bytes) &&
             (shape.xor_bgra.empty() || shape.xor_bgra.size() == bytes);
    }

    struct restore_state_t {
      ID3D11DeviceContext1 *context;
      ComPtr<ID3DDeviceContextState> previous;

      ~restore_state_t() {
        context->SwapDeviceContextState(previous.Get(), nullptr);
      }
    };
  }  // namespace

  std::optional<shape_t> decode_shape(const DXGI_OUTDUPL_POINTER_SHAPE_INFO &info, const std::uint8_t *bytes, std::size_t size) {
    const bool mono = info.Type == DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MONOCHROME;
    const bool masked = info.Type == DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MASKED_COLOR;
    if (!bytes || (!mono && !masked && info.Type != DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR) || (mono && info.Height % 2 != 0)) {
      return std::nullopt;
    }
    shape_t shape;
    shape.width = info.Width;
    shape.height = mono ? info.Height / 2 : info.Height;
    if (!valid_shape(shape)) {
      return std::nullopt;
    }
    const auto row_bytes = mono ? (std::size_t(info.Width) + 7) / 8 : std::size_t(info.Width) * 4;
    // Division avoids overflow even for an invalid driver-supplied pitch or buffer length.
    if (info.Pitch < row_bytes || info.Height > size / info.Pitch) {
      return std::nullopt;
    }
    const auto packed_pitch = std::size_t(shape.width) * 4;
    shape.alpha_bgra.resize(packed_pitch * shape.height);
    if (mono || masked) {
      shape.xor_bgra.resize(shape.alpha_bgra.size());
    }
    for (std::size_t y = 0; y < shape.height; ++y) {
      const auto *row = bytes + y * info.Pitch;
      auto *alpha = shape.alpha_bgra.data() + y * packed_pitch;
      if (!mono && !masked) {
        std::memcpy(alpha, row, packed_pitch);
        continue;
      }
      auto *invert = shape.xor_bgra.data() + y * packed_pitch;
      for (std::size_t x = 0; x < shape.width; ++x) {
        if (mono) {
          const auto bit = 0x80u >> (x % 8);
          const bool and_bit = (row[x / 8] & bit) != 0;
          const bool xor_bit = (row[std::size_t(info.Pitch) * shape.height + x / 8] & bit) != 0;
          if (!and_bit) {
            alpha[4 * x] = alpha[4 * x + 1] = alpha[4 * x + 2] = xor_bit ? 255 : 0;
            alpha[4 * x + 3] = 255;
          } else if (xor_bit) {
            std::memset(invert + 4 * x, 255, 4);
          }
        } else {
          const auto mask = row[4 * x + 3];
          if (mask != 0 && mask != 255) {
            return std::nullopt;
          }
          auto *pixel = (mask ? invert : alpha) + 4 * x;
          std::memcpy(pixel, row + 4 * x, 3);
          pixel[3] = 255;
        }
      }
    }
    return shape;
  }

  std::optional<placement_t> place_cursor(const snapshot_t &cursor, std::uint32_t packed_width, std::uint32_t packed_height, float ui_parallax_uv) {
    if (!::reshade_bridge::valid_ui_parallax(ui_parallax_uv) || !cursor.capture_width || !cursor.capture_height || !packed_height || packed_width < 2 || packed_width % 2 || packed_width > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION || packed_height > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION || (cursor.rotation != DXGI_MODE_ROTATION_UNSPECIFIED && cursor.rotation != DXGI_MODE_ROTATION_IDENTITY)) {
      return std::nullopt;
    }
    const auto &v = cursor.viewport;
    if (!std::isfinite(v.TopLeftX) || !std::isfinite(v.TopLeftY) || !std::isfinite(v.Width) || !std::isfinite(v.Height) || v.Width <= 0 || v.Height <= 0) {
      return std::nullopt;
    }
    const auto eye_width = packed_width / 2;
    const float sx = float(eye_width) / cursor.capture_width;
    const float sy = float(packed_height) / cursor.capture_height;
    placement_t result {};
    for (int eye = 0; eye < 2; ++eye) {
      const float disparity_pixels = (eye == 0 ? ui_parallax_uv : -ui_parallax_uv) * eye_width;
      result.viewports[eye] = {v.TopLeftX * sx + eye * eye_width + disparity_pixels, v.TopLeftY * sy, v.Width * sx, v.Height * sy, 0, 1};
      const auto &p = result.viewports[eye];
      if (!std::isfinite(p.TopLeftX) || !std::isfinite(p.TopLeftY) || !std::isfinite(p.Width) || !std::isfinite(p.Height) || p.TopLeftX < D3D11_VIEWPORT_BOUNDS_MIN || p.TopLeftY < D3D11_VIEWPORT_BOUNDS_MIN || p.TopLeftX + p.Width > D3D11_VIEWPORT_BOUNDS_MAX || p.TopLeftY + p.Height > D3D11_VIEWPORT_BOUNDS_MAX) {
        return std::nullopt;
      }
      result.scissors[eye] = {LONG(eye * eye_width), 0, LONG((eye + 1) * eye_width), LONG(packed_height)};
    }
    return result;
  }

  std::uint32_t patch_margin(float minimum_scale) {
    // An output pixel o of a region from output_region() samples source coordinate
    // u = (o + 0.5 - x0) / s - 0.5 and reads texels floor(u) - 1 .. floor(u) + 1 at most (the
    // 4:2:0 [1,2,1] chroma taps; bilinear reads floor(u) .. floor(u) + 1). The region extends
    // two texels plus two pixels beyond `changed`, so the footprint stays within 5 + 3 / s.
    if (!std::isfinite(minimum_scale) || minimum_scale <= 0.0f) {
      return whole_frame_margin;
    }
    const double margin = std::ceil(5.0 + 3.0 / minimum_scale);
    return margin >= 4096.0 ? whole_frame_margin : static_cast<std::uint32_t>(margin);
  }

  std::optional<D3D11_RECT> output_region(const D3D11_RECT &changed, std::uint32_t source_width, std::uint32_t source_height, const D3D11_VIEWPORT &viewport) {
    if (changed.left >= changed.right || changed.top >= changed.bottom || !source_width || !source_height || !std::isfinite(viewport.TopLeftX) || !std::isfinite(viewport.TopLeftY) || !(viewport.Width > 0.0f) || !(viewport.Height > 0.0f) || !std::isfinite(viewport.Width) || !std::isfinite(viewport.Height)) {
      return std::nullopt;
    }
    // Output pixel o samples source coordinate u = (o + 0.5 - origin) / scale - 0.5. Its
    // footprint can include `changed` only for o in [origin + (first - 0.5) * scale - 0.5,
    // origin + (last + 0.5) * scale - 0.5); two extra texels and pixels keep this conservative.
    const auto axis = [](LONG first, LONG last, std::uint32_t source, float origin, float extent) {
      const double scale = double(extent) / source;
      const double begin = std::max(0.0, std::floor(double(origin)));
      const double end = std::max(begin, std::ceil(double(origin) + extent));
      const double low = std::floor(origin + (double(first) - 2.0) * scale) - 2.0;
      const double high = std::ceil(origin + (double(last) + 2.0) * scale) + 2.0;
      return std::pair<LONG, LONG> {
        LONG(std::clamp(low, begin, end)),
        LONG(std::clamp(high, begin, end)),
      };
    };
    const auto [left, right] = axis(changed.left, changed.right, source_width, viewport.TopLeftX, viewport.Width);
    const auto [top, bottom] = axis(changed.top, changed.bottom, source_height, viewport.TopLeftY, viewport.Height);
    if (left >= right || top >= bottom) {
      return std::nullopt;
    }
    return D3D11_RECT {left, top, right, bottom};
  }

  struct compositor_t::impl_t {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3DBlob> vs_blob, ps_blob, hdr_blob, pq_blob;
    ComPtr<ID3D11DeviceContext1> context1;
    ComPtr<ID3DDeviceContextState> state;
    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> ps, hdr_ps, pq_ps;
    ComPtr<ID3D11SamplerState> sampler;
    ComPtr<ID3D11RasterizerState> raster;
    ComPtr<ID3D11DepthStencilState> depth;
    ComPtr<ID3D11BlendState> alpha_blend, xor_blend;
    ComPtr<ID3D11Buffer> rotation, white;
    ComPtr<ID3D11Texture2D> scratch;
    ComPtr<ID3D11ShaderResourceView> scratch_view;
    ComPtr<ID3D11RenderTargetView> scratch_target;
    ComPtr<ID3D11ShaderResourceView> alpha_view, xor_view;
    std::shared_ptr<const shape_t> uploaded_shape;
    D3D11_TEXTURE2D_DESC scratch_desc {};

    bool initialize() {
      if (state) {
        return true;
      }
      if (!device || !context || !vs_blob || !ps_blob || !hdr_blob) {
        return false;
      }
      ComPtr<ID3D11Device1> device1;
      if (FAILED(device.As(&device1)) || FAILED(context.As(&context1))) {
        return false;
      }
      if (FAILED(device->CreateVertexShader(vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr, &vs)) || FAILED(device->CreatePixelShader(ps_blob->GetBufferPointer(), ps_blob->GetBufferSize(), nullptr, &ps)) || FAILED(device->CreatePixelShader(hdr_blob->GetBufferPointer(), hdr_blob->GetBufferSize(), nullptr, &hdr_ps))) {
        return false;
      }
      D3D11_SAMPLER_DESC sd {};
      sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
      sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
      sd.MaxLOD = D3D11_FLOAT32_MAX;
      if (FAILED(device->CreateSamplerState(&sd, &sampler))) {
        return false;
      }
      D3D11_RASTERIZER_DESC rd {};
      rd.FillMode = D3D11_FILL_SOLID;
      rd.CullMode = D3D11_CULL_NONE;
      rd.ScissorEnable = TRUE;
      rd.DepthClipEnable = TRUE;
      D3D11_DEPTH_STENCIL_DESC dd {};
      if (FAILED(device->CreateRasterizerState(&rd, &raster)) || FAILED(device->CreateDepthStencilState(&dd, &depth))) {
        return false;
      }
      for (int invert = 0; invert < 2; ++invert) {
        D3D11_BLEND_DESC bd {};
        auto &rt = bd.RenderTarget[0];
        rt.BlendEnable = TRUE;
        // Keep the authored frame's alpha intact, including at a transparent cursor pixel.
        rt.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_RED | D3D11_COLOR_WRITE_ENABLE_GREEN | D3D11_COLOR_WRITE_ENABLE_BLUE;
        rt.BlendOp = rt.BlendOpAlpha = D3D11_BLEND_OP_ADD;
        rt.SrcBlend = invert ? D3D11_BLEND_INV_DEST_COLOR : D3D11_BLEND_SRC_ALPHA;
        rt.DestBlend = invert ? D3D11_BLEND_INV_SRC_COLOR : D3D11_BLEND_INV_SRC_ALPHA;
        rt.SrcBlendAlpha = D3D11_BLEND_ZERO;
        rt.DestBlendAlpha = D3D11_BLEND_ONE;
        if (FAILED(device->CreateBlendState(&bd, invert ? &xor_blend : &alpha_blend))) {
          return false;
        }
      }
      const std::array<float, 4> zero {};
      D3D11_BUFFER_DESC cb {};
      cb.ByteWidth = sizeof(zero);
      cb.Usage = D3D11_USAGE_DEFAULT;
      cb.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      D3D11_SUBRESOURCE_DATA initial {zero.data(), 0, 0};
      if (FAILED(device->CreateBuffer(&cb, &initial, &rotation)) || FAILED(device->CreateBuffer(&cb, &initial, &white))) {
        return false;
      }
      if (pq_blob && FAILED(device->CreatePixelShader(pq_blob->GetBufferPointer(), pq_blob->GetBufferSize(), nullptr, &pq_ps))) {
        return false;
      }
      const auto feature = device->GetFeatureLevel();
      return SUCCEEDED(device1->CreateDeviceContextState(0, &feature, 1, D3D11_SDK_VERSION, __uuidof(ID3D11Device), nullptr, &state));
    }

    bool upload_shape(const std::shared_ptr<const shape_t> &shape) {
      if (uploaded_shape == shape) {
        return true;
      }
      const auto upload = [&](const std::vector<std::uint8_t> &bytes, ComPtr<ID3D11ShaderResourceView> &view) {
        if (bytes.empty()) {
          return true;
        }
        D3D11_TEXTURE2D_DESC td {};
        td.Width = shape->width;
        td.Height = shape->height;
        td.ArraySize = td.MipLevels = td.SampleDesc.Count = 1;
        td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        td.Usage = D3D11_USAGE_IMMUTABLE;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA initial {bytes.data(), shape->width * 4, 0};
        ComPtr<ID3D11Texture2D> texture;
        return SUCCEEDED(device->CreateTexture2D(&td, &initial, &texture)) && SUCCEEDED(device->CreateShaderResourceView(texture.Get(), nullptr, &view));
      };
      ComPtr<ID3D11ShaderResourceView> next_alpha, next_xor;
      if (!upload(shape->alpha_bgra, next_alpha) || !upload(shape->xor_bgra, next_xor)) {
        return false;
      }
      alpha_view = std::move(next_alpha);
      xor_view = std::move(next_xor);
      uploaded_shape = shape;
      return true;
    }

    bool ensure_scratch(const D3D11_TEXTURE2D_DESC &source, DXGI_FORMAT format) {
      if (scratch && scratch_desc.Width == source.Width && scratch_desc.Height == source.Height && scratch_desc.Format == format) {
        return true;
      }
      D3D11_TEXTURE2D_DESC td = source;
      td.Format = format;
      td.Usage = D3D11_USAGE_DEFAULT;
      td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
      td.CPUAccessFlags = td.MiscFlags = 0;
      ComPtr<ID3D11Texture2D> texture;
      ComPtr<ID3D11ShaderResourceView> view;
      ComPtr<ID3D11RenderTargetView> target;
      if (FAILED(device->CreateTexture2D(&td, nullptr, &texture)) || FAILED(device->CreateShaderResourceView(texture.Get(), nullptr, &view)) || FAILED(device->CreateRenderTargetView(texture.Get(), nullptr, &target))) {
        return false;
      }
      scratch = std::move(texture);
      scratch_view = std::move(view);
      scratch_target = std::move(target);
      scratch_desc = td;
      return true;
    }
  };

  compositor_t::compositor_t(ID3D11Device *device, ID3D11DeviceContext *context, ID3DBlob *vertex_shader, ID3DBlob *pixel_shader, ID3DBlob *hdr_pixel_shader, ID3DBlob *pq_decode_pixel_shader):
      impl_(std::make_unique<impl_t>()) {
    impl_->device = device;
    impl_->context = context;
    impl_->vs_blob = vertex_shader;
    impl_->ps_blob = pixel_shader;
    impl_->hdr_blob = hdr_pixel_shader;
    impl_->pq_blob = pq_decode_pixel_shader;
  }

  compositor_t::~compositor_t() = default;

  std::optional<result_t> compositor_t::compose(ID3D11Texture2D *texture, ID3D11ShaderResourceView *view, const snapshot_t &cursor, source_transfer_e transfer, float white_multiplier, float ui_parallax_uv, std::uint32_t margin) {
    if (!texture || !view || !::reshade_bridge::valid_ui_parallax(ui_parallax_uv)) {
      return std::nullopt;
    }
    if (!cursor.visible || !cursor.shape || (cursor.shape->alpha_bgra.empty() && cursor.shape->xor_bgra.empty())) {
      return result_t {texture, view};
    }
    const bool linear = transfer != source_transfer_e::srgb;
    const bool pq = transfer == source_transfer_e::pq;
    if (!valid_shape(*cursor.shape) || !std::isfinite(white_multiplier) || white_multiplier <= 0) {
      return std::nullopt;
    }
    D3D11_TEXTURE2D_DESC source {};
    texture->GetDesc(&source);
    const auto placement = place_cursor(cursor, source.Width, source.Height, ui_parallax_uv);
    if (!placement || source.SampleDesc.Count != 1 || source.MipLevels != 1 || source.ArraySize != 1) {
      return std::nullopt;
    }
    // Texels each eye's draw can change: its viewport's pixel bounds inside its eye scissor.
    result_t result {};
    result.linear = linear;
    const bool whole_frame = margin == whole_frame_margin;
    const auto grow = [](LONG value, std::uint32_t by, LONG low, LONG high, int sign) {
      return LONG(std::clamp(double(value) + sign * double(by), double(low), double(high)));
    };
    for (int eye = 0; eye < 2; ++eye) {
      const auto &v = placement->viewports[eye];
      const auto &clip = placement->scissors[eye];
      const auto bound = [](double value, LONG low, LONG high) {
        return LONG(std::clamp(value, double(low), double(high)));
      };
      const D3D11_RECT changed {
        bound(std::floor(double(v.TopLeftX)), clip.left, clip.right),
        bound(std::floor(double(v.TopLeftY)), clip.top, clip.bottom),
        bound(std::ceil(double(v.TopLeftX) + v.Width), clip.left, clip.right),
        bound(std::ceil(double(v.TopLeftY) + v.Height), clip.top, clip.bottom),
      };
      if (changed.left >= changed.right || changed.top >= changed.bottom) {
        continue;
      }
      const auto index = result.regions++;
      result.changed[index] = changed;
      result.valid[index] = whole_frame ? D3D11_RECT {0, 0, LONG(source.Width), LONG(source.Height)} :
                                          D3D11_RECT {
                                            grow(changed.left, margin, 0, LONG(source.Width), -1),
                                            grow(changed.top, margin, 0, LONG(source.Height), -1),
                                            grow(changed.right, margin, 0, LONG(source.Width), 1),
                                            grow(changed.bottom, margin, 0, LONG(source.Height), 1),
                                          };
    }
    if (!result.regions) {
      // Both eyes clip the cursor away; the frame is unchanged.
      return result_t {texture, view};
    }
    auto &p = *impl_;
    if (!p.initialize() || (pq && !p.pq_ps) || !p.upload_shape(cursor.shape) || !p.ensure_scratch(source, pq ? DXGI_FORMAT_R16G16B16A16_FLOAT : source.Format)) {
      return std::nullopt;
    }
    restore_state_t restore {p.context1.Get(), {}};
    p.context1->SwapDeviceContextState(p.state.Get(), &restore.previous);
    auto *ctx = p.context.Get();
    // Populate every valid rectangle before any blend: a neighbouring eye's rectangle may
    // overlap texels the other eye's cursor changes.
    const std::uint32_t copies = whole_frame ? 1u : result.regions;
    if (pq) {
      const D3D11_VIEWPORT frame {0, 0, float(source.Width), float(source.Height), 0, 1};
      ctx->OMSetRenderTargets(1, p.scratch_target.GetAddressOf(), nullptr);
      ctx->OMSetBlendState(nullptr, nullptr, UINT_MAX);
      ctx->OMSetDepthStencilState(p.depth.Get(), 0);
      ctx->RSSetState(p.raster.Get());
      ctx->RSSetViewports(1, &frame);
      ctx->IASetInputLayout(nullptr);
      ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      ctx->VSSetShader(p.vs.Get(), nullptr, 0);
      ctx->VSSetConstantBuffers(2, 1, p.rotation.GetAddressOf());
      ctx->PSSetShader(p.pq_ps.Get(), nullptr, 0);
      ctx->PSSetShaderResources(0, 1, &view);
      for (std::uint32_t i = 0; i < copies; ++i) {
        ctx->RSSetScissorRects(1, &result.valid[i]);
        ctx->Draw(3, 0);
      }
      ID3D11ShaderResourceView *none = nullptr;
      ctx->PSSetShaderResources(0, 1, &none);
    } else if (whole_frame) {
      ctx->CopyResource(p.scratch.Get(), texture);
    } else {
      for (std::uint32_t i = 0; i < copies; ++i) {
        const auto &r = result.valid[i];
        const D3D11_BOX box {UINT(r.left), UINT(r.top), 0, UINT(r.right), UINT(r.bottom), 1};
        ctx->CopySubresourceRegion(p.scratch.Get(), 0, box.left, box.top, 0, texture, 0, &box);
      }
    }
    const std::array<float, 4> white {white_multiplier, 0, 0, 0};
    ctx->UpdateSubresource(p.white.Get(), 0, nullptr, white.data(), 0, 0);
    ctx->OMSetRenderTargets(1, p.scratch_target.GetAddressOf(), nullptr);
    ctx->OMSetDepthStencilState(p.depth.Get(), 0);
    ctx->RSSetState(p.raster.Get());
    ctx->IASetInputLayout(nullptr);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->VSSetShader(p.vs.Get(), nullptr, 0);
    ctx->VSSetConstantBuffers(2, 1, p.rotation.GetAddressOf());
    ctx->PSSetShader(linear ? p.hdr_ps.Get() : p.ps.Get(), nullptr, 0);
    ctx->PSSetConstantBuffers(1, 1, p.white.GetAddressOf());
    ctx->PSSetSamplers(0, 1, p.sampler.GetAddressOf());
    for (int eye = 0; eye < 2; ++eye) {
      ctx->RSSetViewports(1, &placement->viewports[eye]);
      ctx->RSSetScissorRects(1, &placement->scissors[eye]);
      const auto draw = [&](ID3D11ShaderResourceView *cursor_view, ID3D11BlendState *blend) {
        if (cursor_view) {
          ctx->PSSetShaderResources(0, 1, &cursor_view);
          ctx->OMSetBlendState(blend, nullptr, UINT_MAX);
          ctx->Draw(3, 0);
        }
      };
      draw(p.alpha_view.Get(), p.alpha_blend.Get());
      draw(p.xor_view.Get(), p.xor_blend.Get());
    }
    ID3D11ShaderResourceView *none = nullptr;
    ctx->PSSetShaderResources(0, 1, &none);
    ctx->OMSetRenderTargets(0, nullptr, nullptr);
    result.texture = p.scratch.Get();
    result.view = p.scratch_view.Get();
    return result;
  }
}  // namespace platf::sbs_cursor
