// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file tools/text_hook/text_hook_sample.cpp
 * @brief A window that draws one line of text the way a text game may, for testing the helper.
 *
 * usage: text_hook_sample <mode>
 *   gdi-window   text drawn straight onto the window
 *   gdi-buffer   text drawn on a back buffer that is copied to the window
 *   gdi-layer    text drawn once on a keyed layer, laid over the back buffer every frame
 *   gdi-bits     text drawn on a DIB section whose pixels are sent to the window
 *   gdi-glyphs   text composed by the program itself from the glyph bitmaps GDI hands out, which
 *                no drawing call reveals: the helper sees none of it
 *   d3d11-layer  text composed the same way, loaded into a texture, drawn onto a scene target
 *                and that target drawn onto the back buffer, all with Direct3D 11
 *   d3d11-direct the same texture, written through a mapping and drawn straight onto the back
 *                buffer
 *   d3d11-own    as d3d11-layer, by a program with a font renderer of its own: the line is
 *                composed once at the start, no glyph is taken from GDI afterwards, and the
 *                text image is loaded into its texture again for every frame that shows it
 *   d3d11-parts  the same image of glyph-sized shapes, loaded once into a new texture of one
 *                size with every third frame that shows it, as a game loads the parts of a
 *                figure's face whenever it shows the figure. No glyph is taken from GDI and no
 *                texture is loaded twice, so nothing says that this is text: the helper is to
 *                see none in it
 *   d3d11-cache  as d3d11-layer, by a program that keeps the glyphs it has taken: only the first
 *                two lines it shows take any from GDI, and every line has a texture of its own
 *                size
 *   d3d11-atlas  as d3d11-own, by a program that keeps its glyphs in one texture and draws the
 *                line from a part of it: with every frame that shows the text more glyphs are
 *                set down in it and the whole of it is loaded again, until it is fuller than
 *                any text layer. Text shown again begins the cache anew
 *   d2d          text drawn with Direct2D
 *   d2d-busy     a full-size picture redrawn as fast as it can be, three lines drawn glyph by
 *                glyph with an outline, as a visual novel draws its dialogue: 810 text draws a
 *                frame. It ends after four seconds; its exit code is the number of frames drawn in
 *                the last three, so that a helper loaded in the first is counted throughout
 * The window repaints ten times a second. WM_APP + 1 with wParam 0 or 1 hides or shows the
 * text, and with wParam 2 the d2d mode draws a second line under it; WM_CLOSE ends the program.
 * Not shipped.
 */
#include <cwchar>
#include <windows.h>

#include <d2d1.h>
#include <d3d11.h>
#include <dwrite.h>

namespace {
  int client_width = 640, client_height = 360;
  constexpr int text_x = 40, text_y = 60;
  constexpr int layer_x = 30, layer_y = 200;  // where the keyed layer is laid on the buffer
  constexpr wchar_t line[] = L"Sunshine 3D sample text";
  constexpr COLORREF text_colour = RGB(240, 232, 224);
  constexpr COLORREF key_colour = RGB(255, 0, 255);

  enum class mode_e {
    gdi_window,
    gdi_buffer,
    gdi_layer,
    gdi_bits,
    gdi_glyphs,
    d3d11_layer,
    d3d11_direct,
    d3d11_own,
    d3d11_parts,
    d3d11_cache,
    d3d11_atlas,
    d2d,
    d2d_busy,
  };

  mode_e mode = mode_e::gdi_window;
  bool show_text = true, second_line = false;
  HFONT font = nullptr;
  HDC buffer_dc = nullptr, layer_dc = nullptr;
  void *buffer_bits = nullptr;
  BITMAPINFO buffer_info {};
  bool layer_drawn = false, layer_shows = false;
  ID2D1HwndRenderTarget *target = nullptr;
  ID2D1SolidColorBrush *brush = nullptr;
  IDWriteTextFormat *format = nullptr;

  void fill(HDC dc, const RECT &rect, COLORREF colour) {
    const HBRUSH solid = CreateSolidBrush(colour);
    FillRect(dc, &rect, solid);
    DeleteObject(solid);
  }

  void draw_line(HDC dc, int x, int y) {
    const HGDIOBJ previous = SelectObject(dc, font);
    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, text_colour);
    TextOutW(dc, x, y, line, static_cast<int>(std::wcslen(line)));
    SelectObject(dc, previous);
  }

  // The way an engine composes its own text: glyph bitmaps from GDI, set down by its own code
  // in an image `width` by `height`. `solid` is added to every texel of a glyph.
  void compose_line(HDC dc, unsigned int *pixels, int width, int height, int x, int y, unsigned int solid) {
    const HGDIOBJ previous = SelectObject(dc, font);
    TEXTMETRICW metrics {};
    GetTextMetricsW(dc, &metrics);
    const MAT2 identity {{0, 1}, {0, 0}, {0, 0}, {0, 1}};
    static unsigned char glyph[96 * 96];
    for (const wchar_t *character = line; *character; ++character) {
      GLYPHMETRICS box {};
      const DWORD size = GetGlyphOutlineW(dc, *character, GGO_GRAY8_BITMAP, &box, sizeof(glyph), glyph, &identity);
      if (size != GDI_ERROR && size != 0u) {
        const int pitch = static_cast<int>((box.gmBlackBoxX + 3u) & ~3u);  // rows are DWORD-aligned
        for (int row = 0; row < static_cast<int>(box.gmBlackBoxY); ++row) {
          for (int column = 0; column < static_cast<int>(box.gmBlackBoxX); ++column) {
            const int px = x + box.gmptGlyphOrigin.x + column, py = y + metrics.tmAscent - box.gmptGlyphOrigin.y + row;
            // Coverage runs from 0 to 64.
            if (glyph[row * pitch + column] > 32u && px >= 0 && px < width && py >= 0 && py < height) {
              pixels[py * width + px] = solid | (GetRValue(text_colour) << 16) | (GetGValue(text_colour) << 8) | GetBValue(text_colour);
            }
          }
        }
      }
      x += box.gmCellIncX;
    }
    SelectObject(dc, previous);
  }

  HDC memory_dc(HDC like, int width, int height, void **bits) {
    const HDC dc = CreateCompatibleDC(like);
    buffer_info.bmiHeader.biSize = sizeof(buffer_info.bmiHeader);
    buffer_info.bmiHeader.biWidth = width;
    buffer_info.bmiHeader.biHeight = -height;
    buffer_info.bmiHeader.biPlanes = 1;
    buffer_info.bmiHeader.biBitCount = 32;
    buffer_info.bmiHeader.biCompression = BI_RGB;
    SelectObject(dc, CreateDIBSection(like, &buffer_info, DIB_RGB_COLORS, bits, nullptr, 0u));
    return dc;
  }

  void paint_gdi(HWND window) {
    PAINTSTRUCT paint {};
    const HDC dc = BeginPaint(window, &paint);
    const RECT client {0, 0, client_width, client_height};
    if (mode == mode_e::gdi_window) {
      fill(dc, client, RGB(24, 32, 64));
      if (show_text) {
        draw_line(dc, text_x, text_y);
      }
      EndPaint(window, &paint);
      return;
    }
    if (!buffer_dc) {
      buffer_dc = memory_dc(dc, client_width, client_height, &buffer_bits);
    }
    fill(buffer_dc, client, RGB(24, 32, 64));
    if (mode == mode_e::gdi_layer) {
      const RECT layer {0, 0, 560, 60};
      if (!layer_dc) {
        void *unused = nullptr;
        layer_dc = memory_dc(dc, layer.right, layer.bottom, &unused);
      }
      // The layer is drawn only when its text changes, as a message layer is.
      if (!layer_drawn || layer_shows != show_text) {
        fill(layer_dc, layer, key_colour);
        if (show_text) {
          draw_line(layer_dc, 10, 10);
        }
        layer_drawn = true;
        layer_shows = show_text;
      }
      TransparentBlt(buffer_dc, layer_x, layer_y, layer.right, layer.bottom, layer_dc, 0, 0, layer.right, layer.bottom, key_colour);
    } else if (show_text && mode == mode_e::gdi_glyphs) {
      GdiFlush();
      compose_line(buffer_dc, static_cast<unsigned int *>(buffer_bits), client_width, client_height, text_x, text_y, 0u);
    } else if (show_text) {
      draw_line(buffer_dc, text_x, text_y);
    }
    if (mode == mode_e::gdi_bits) {
      GdiFlush();
      SetDIBitsToDevice(dc, 0, 0, client_width, client_height, 0, 0, 0u, client_height, buffer_bits, &buffer_info, DIB_RGB_COLORS);
    } else {
      BitBlt(dc, 0, 0, client_width, client_height, buffer_dc, 0, 0, SRCCOPY);
    }
    EndPaint(window, &paint);
  }

  // --- Direct3D 11: text the program composed, drawn from a texture ---------------------------

  constexpr int text_width = 560, text_height = 60;
  IDXGISwapChain *chain = nullptr;
  ID3D11Device *gpu = nullptr;
  ID3D11DeviceContext *gpu_context = nullptr;
  ID3D11RenderTargetView *back_target = nullptr, *scene_target = nullptr;
  ID3D11ShaderResourceView *scene_view = nullptr, *text_view = nullptr;
  ID3D11Texture2D *text_texture = nullptr;
  ID3D11Buffer *quad = nullptr;
  ID3D11BlendState *over_blend = nullptr;
  bool gpu_failed = false, text_loaded = false;
  unsigned shown_frames = 0u, lines_made = 0u;
  int text_drawn_width = 0;
  unsigned int text_pixels[text_width * text_height];
  // The glyph cache of d3d11-atlas: the line in its first band, and one more band of stand-ins
  // for glyphs with every frame, in a colour the line does not have.
  constexpr int atlas_bands = 8, atlas_height = text_height * atlas_bands;
  unsigned int atlas_pixels[text_width * atlas_height];
  int atlas_filled = 0;

  void begin_atlas() {
    ZeroMemory(atlas_pixels, sizeof(atlas_pixels));
    CopyMemory(atlas_pixels, text_pixels, sizeof(text_pixels));
    atlas_filled = 1;
  }

  void fill_atlas_band() {
    for (int y = atlas_filled * text_height; y < (atlas_filled + 1) * text_height; ++y) {
      for (int x = 0; x < text_width; ++x) {
        atlas_pixels[y * text_width + x] = x % 16 < 14 && y % 16 < 14 ? 0xff40c040u : 0u;
      }
    }
    ++atlas_filled;
  }

  void compose_text() {
    ZeroMemory(text_pixels, sizeof(text_pixels));
    const HDC dc = CreateCompatibleDC(nullptr);
    compose_line(dc, text_pixels, text_width, text_height, 0, 0, 0xff000000u);
    DeleteDC(dc);
  }

  constexpr char shaders[] =
    "cbuffer quad : register(b0) { float4 rect; float4 source; };\n"
    "struct corner_t { float4 position : SV_Position; float2 uv : TEXCOORD0; };\n"
    "corner_t vertex(uint id : SV_VertexID) {\n"
    "  corner_t corner; float2 at = float2(id & 1, id >> 1);\n"
    "  corner.position = float4(lerp(rect.x, rect.z, at.x), lerp(rect.y, rect.w, at.y), 0.0, 1.0);\n"
    "  corner.uv = lerp(source.xy, source.zw, at); return corner; }\n"
    "Texture2D image : register(t0); SamplerState smooth : register(s0);\n"
    "float4 pixel(corner_t corner) : SV_Target { return image.Sample(smooth, corner.uv); }\n";

  ID3DBlob *compile(const char *entry, const char *target) {
    using compile_fn = HRESULT(WINAPI *)(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO *, ID3DInclude *, LPCSTR, LPCSTR, UINT, UINT, ID3DBlob **, ID3DBlob **);
    const HMODULE compiler = LoadLibraryW(L"d3dcompiler_47.dll");
    const auto compile_shader = compiler ? reinterpret_cast<compile_fn>(reinterpret_cast<void *>(GetProcAddress(compiler, "D3DCompile"))) : nullptr;
    ID3DBlob *code = nullptr;
    if (compile_shader) {
      compile_shader(shaders, sizeof(shaders) - 1u, nullptr, nullptr, nullptr, entry, target, 0u, 0u, &code, nullptr);
    }
    return code;
  }

  bool start_d3d11(HWND window) {
    DXGI_SWAP_CHAIN_DESC desc {};
    desc.BufferDesc.Width = static_cast<UINT>(client_width);
    desc.BufferDesc.Height = static_cast<UINT>(client_height);
    desc.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1u;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = 1u;
    desc.OutputWindow = window;
    desc.Windowed = TRUE;
    // A machine without a graphics adapter still has the software one.
    const D3D_DRIVER_TYPE drivers[] {D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP};
    for (const D3D_DRIVER_TYPE driver : drivers) {
      if (SUCCEEDED(D3D11CreateDeviceAndSwapChain(nullptr, driver, nullptr, 0u, nullptr, 0u, D3D11_SDK_VERSION, &desc, &chain, &gpu, nullptr, &gpu_context)) && chain) {
        break;
      }
    }
    if (!chain || !gpu || !gpu_context) {
      return false;
    }
    ID3D11Texture2D *back = nullptr, *scene = nullptr;
    chain->GetBuffer(0u, __uuidof(ID3D11Texture2D), reinterpret_cast<void **>(&back));
    if (back) {
      gpu->CreateRenderTargetView(back, nullptr, &back_target);
      back->Release();
    }
    D3D11_TEXTURE2D_DESC image {};
    image.Width = static_cast<UINT>(client_width);
    image.Height = static_cast<UINT>(client_height);
    image.MipLevels = 1u;
    image.ArraySize = 1u;
    image.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    image.SampleDesc.Count = 1u;
    image.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    if (SUCCEEDED(gpu->CreateTexture2D(&image, nullptr, &scene)) && scene) {
      gpu->CreateRenderTargetView(scene, nullptr, &scene_target);
      gpu->CreateShaderResourceView(scene, nullptr, &scene_view);
      scene->Release();
    }
    image.Width = text_width;
    image.Height = mode == mode_e::d3d11_atlas ? atlas_height : text_height;
    image.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    if (mode == mode_e::d3d11_direct) {
      image.Usage = D3D11_USAGE_DYNAMIC;
      image.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    }
    if (SUCCEEDED(gpu->CreateTexture2D(&image, nullptr, &text_texture)) && text_texture) {
      gpu->CreateShaderResourceView(text_texture, nullptr, &text_view);
    }
    D3D11_BUFFER_DESC constants {};
    constants.ByteWidth = 32u;
    constants.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    gpu->CreateBuffer(&constants, nullptr, &quad);
    D3D11_BLEND_DESC blend {};
    blend.RenderTarget[0].BlendEnable = TRUE;
    blend.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    blend.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    blend.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    blend.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    blend.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
    blend.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    blend.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    gpu->CreateBlendState(&blend, &over_blend);
    ID3DBlob *vertex_code = compile("vertex", "vs_4_0"), *pixel_code = compile("pixel", "ps_4_0");
    ID3D11VertexShader *vertex = nullptr;
    ID3D11PixelShader *pixel = nullptr;
    if (vertex_code && pixel_code) {
      gpu->CreateVertexShader(vertex_code->GetBufferPointer(), vertex_code->GetBufferSize(), nullptr, &vertex);
      gpu->CreatePixelShader(pixel_code->GetBufferPointer(), pixel_code->GetBufferSize(), nullptr, &pixel);
    }
    if (!back_target || !scene_target || !scene_view || !text_view || !quad || !over_blend || !vertex || !pixel) {
      return false;
    }
    gpu_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    gpu_context->VSSetShader(vertex, nullptr, 0u);
    gpu_context->PSSetShader(pixel, nullptr, 0u);
    gpu_context->VSSetConstantBuffers(0u, 1u, &quad);
    return true;
  }

  /**
   * Draw the bound texture, from its top down to `source_bottom` of its height, on this
   * rectangle of the bound target, given in its pixels.
   */
  void draw_quad(float left, float top, float right, float bottom, float source_bottom = 1.0f) {
    const float w = static_cast<float>(client_width), h = static_cast<float>(client_height);
    const float rect[8] {left / w * 2.0f - 1.0f, 1.0f - top / h * 2.0f, right / w * 2.0f - 1.0f, 1.0f - bottom / h * 2.0f, 0.0f, 0.0f, 1.0f, source_bottom};
    gpu_context->UpdateSubresource(quad, 0u, nullptr, rect, 0u, 0u);
    gpu_context->Draw(4u, 0u);
  }

  void paint_d3d11(HWND window) {
    if (!chain && !gpu_failed) {
      gpu_failed = !start_d3d11(window);
    }
    ValidateRect(window, nullptr);
    if (gpu_failed) {
      return;
    }
    if (mode == mode_e::d3d11_parts || mode == mode_e::d3d11_cache) {
      const bool next_line = mode == mode_e::d3d11_parts ? show_text && shown_frames++ % 3u == 0u : show_text && !text_loaded;
      if (next_line) {
        D3D11_TEXTURE2D_DESC image {};
        text_texture->GetDesc(&image);
        if (mode == mode_e::d3d11_cache) {
          if (lines_made < 2u) {
            compose_text();
          }
          image.Width = static_cast<UINT>(text_width) - 16u * (++lines_made % 4u);
          text_drawn_width = static_cast<int>(image.Width);
          text_loaded = true;
        }
        const D3D11_SUBRESOURCE_DATA initial {text_pixels, text_width * 4u, 0u};
        ID3D11Texture2D *next = nullptr;
        ID3D11ShaderResourceView *next_view = nullptr;
        if (SUCCEEDED(gpu->CreateTexture2D(&image, &initial, &next)) && next && SUCCEEDED(gpu->CreateShaderResourceView(next, nullptr, &next_view)) && next_view) {
          text_view->Release();
          text_texture->Release();
          text_texture = next;
          text_view = next_view;
        } else if (next) {
          next->Release();
        }
      }
    } else if (mode == mode_e::d3d11_atlas) {
      if (show_text && (!text_loaded || atlas_filled < atlas_bands)) {
        if (!text_loaded) {
          begin_atlas();
        }
        fill_atlas_band();
        gpu_context->UpdateSubresource(text_texture, 0u, nullptr, atlas_pixels, text_width * 4u, 0u);
        text_loaded = true;
      }
    } else if (show_text && (!text_loaded || mode == mode_e::d3d11_own)) {
      if (mode != mode_e::d3d11_own) {
        compose_text();
      }
      D3D11_MAPPED_SUBRESOURCE memory {};
      if (mode != mode_e::d3d11_direct) {
        gpu_context->UpdateSubresource(text_texture, 0u, nullptr, text_pixels, text_width * 4u, 0u);
      } else if (SUCCEEDED(gpu_context->Map(text_texture, 0u, D3D11_MAP_WRITE_DISCARD, 0u, &memory))) {
        for (int row = 0; row < text_height; ++row) {
          CopyMemory(static_cast<unsigned char *>(memory.pData) + static_cast<SIZE_T>(row) * memory.RowPitch, &text_pixels[row * text_width], text_width * 4u);
        }
        gpu_context->Unmap(text_texture, 0u);
      }
      text_loaded = true;
    }
    // Text shown again is composed again, as a new line is.
    text_loaded = text_loaded && show_text;
    const float background[4] {24.0f / 255.0f, 32.0f / 255.0f, 64.0f / 255.0f, 1.0f};
    const D3D11_VIEWPORT viewport {0.0f, 0.0f, static_cast<float>(client_width), static_cast<float>(client_height), 0.0f, 1.0f};
    ID3D11ShaderResourceView *none = nullptr;
    ID3D11RenderTargetView *const first = mode == mode_e::d3d11_direct ? back_target : scene_target;
    gpu_context->RSSetViewports(1u, &viewport);
    gpu_context->OMSetRenderTargets(1u, &first, nullptr);
    gpu_context->ClearRenderTargetView(first, background);
    if (show_text) {
      gpu_context->OMSetBlendState(over_blend, nullptr, 0xffffffffu);
      gpu_context->PSSetShaderResources(0u, 1u, &text_view);
      draw_quad(static_cast<float>(text_x), static_cast<float>(text_y), static_cast<float>(text_x + (text_drawn_width ? text_drawn_width : text_width)), static_cast<float>(text_y + text_height), mode == mode_e::d3d11_atlas ? 1.0f / atlas_bands : 1.0f);
    }
    if (mode != mode_e::d3d11_direct) {
      gpu_context->PSSetShaderResources(0u, 1u, &none);
      gpu_context->OMSetRenderTargets(1u, &back_target, nullptr);
      gpu_context->OMSetBlendState(nullptr, nullptr, 0xffffffffu);
      gpu_context->PSSetShaderResources(0u, 1u, &scene_view);
      draw_quad(0.0f, 0.0f, static_cast<float>(client_width), static_cast<float>(client_height));
      gpu_context->PSSetShaderResources(0u, 1u, &none);
    }
    chain->Present(0u, 0u);
  }

  void paint_d2d(HWND window) {
    if (!target) {
      ID2D1Factory *factory = nullptr;
      IDWriteFactory *dwrite = nullptr;
      D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, __uuidof(ID2D1Factory), nullptr, reinterpret_cast<void **>(&factory));
      DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), reinterpret_cast<IUnknown **>(&dwrite));
      if (!factory || !dwrite) {
        return;
      }
      const D2D1_RENDER_TARGET_PROPERTIES properties {
        D2D1_RENDER_TARGET_TYPE_DEFAULT,
        {DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_IGNORE},
        96.0f, 96.0f, D2D1_RENDER_TARGET_USAGE_NONE, D2D1_FEATURE_LEVEL_DEFAULT,
      };
      const D2D1_HWND_RENDER_TARGET_PROPERTIES window_properties {
        window,
        {static_cast<UINT32>(client_width), static_cast<UINT32>(client_height)},
        mode == mode_e::d2d_busy ? D2D1_PRESENT_OPTIONS_IMMEDIATELY : D2D1_PRESENT_OPTIONS_NONE,
      };
      factory->CreateHwndRenderTarget(&properties, &window_properties, &target);
      dwrite->CreateTextFormat(L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 32.0f, L"en-us", &format);
      if (target) {
        target->CreateSolidColorBrush(D2D1::ColorF(240.0f / 255.0f, 232.0f / 255.0f, 224.0f / 255.0f, 1.0f), &brush);
      }
    }
    if (target && brush && format) {
      target->BeginDraw();
      target->Clear(D2D1::ColorF(24.0f / 255.0f, 32.0f / 255.0f, 64.0f / 255.0f, 1.0f));
      if (mode == mode_e::d2d_busy) {
        static ID2D1SolidColorBrush *edge = nullptr;
        if (!edge) {
          target->CreateSolidColorBrush(D2D1::ColorF(0.0f, 0.0f, 0.0f, 1.0f), &edge);
        }
        constexpr wchar_t glyphs[] = L"Sunshine3Dsampletextforabusyframe";
        constexpr float offsets[8][2] {{-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1}};
        for (int row = 0; row < 3 && edge; ++row) {
          for (int column = 0; column < 30; ++column) {
            const float x = 120.0f + 36.0f * static_cast<float>(column), y = 760.0f + 60.0f * static_cast<float>(row);
            for (const auto &offset : offsets) {
              const D2D1_RECT_F around {x + offset[0], y + offset[1], x + 60.0f, y + 60.0f};
              target->DrawText(&glyphs[column], 1u, format, &around, edge, D2D1_DRAW_TEXT_OPTIONS_NONE, DWRITE_MEASURING_MODE_NATURAL);
            }
            const D2D1_RECT_F rect {x, y, x + 60.0f, y + 60.0f};
            target->DrawText(&glyphs[column], 1u, format, &rect, brush, D2D1_DRAW_TEXT_OPTIONS_NONE, DWRITE_MEASURING_MODE_NATURAL);
          }
        }
      } else if (show_text) {
        // Exercise a slow frame: the cleared surface is not displayed between Clear and EndDraw.
        Sleep(75);
        const D2D1_RECT_F rect {static_cast<float>(text_x), static_cast<float>(text_y), 620.0f, 140.0f};
        target->DrawText(line, static_cast<UINT32>(std::wcslen(line)), format, &rect, brush, D2D1_DRAW_TEXT_OPTIONS_NONE, DWRITE_MEASURING_MODE_NATURAL);
        if (second_line) {
          const D2D1_RECT_F below {rect.left, rect.top + 80.0f, rect.right, rect.bottom + 80.0f};
          target->DrawText(line, static_cast<UINT32>(std::wcslen(line)), format, &below, brush, D2D1_DRAW_TEXT_OPTIONS_NONE, DWRITE_MEASURING_MODE_NATURAL);
        }
      }
      target->EndDraw();
    }
    ValidateRect(window, nullptr);
  }

  LRESULT CALLBACK procedure(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
    switch (message) {
      case WM_TIMER:
        InvalidateRect(window, nullptr, FALSE);
        return 0;
      case WM_APP + 1:
        show_text = wparam != 0u;
        second_line = wparam == 2u;
        InvalidateRect(window, nullptr, FALSE);
        return 0;
      case WM_ERASEBKGND:
        return 1;
      case WM_PAINT:
        if (mode == mode_e::d3d11_layer || mode == mode_e::d3d11_direct || mode == mode_e::d3d11_own || mode == mode_e::d3d11_parts || mode == mode_e::d3d11_cache || mode == mode_e::d3d11_atlas) {
          paint_d3d11(window);
        } else if (mode == mode_e::d2d || mode == mode_e::d2d_busy) {
          paint_d2d(window);
        } else {
          paint_gdi(window);
        }
        return 0;
      case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
      default:
        return DefWindowProcW(window, message, wparam, lparam);
    }
  }
}  // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR arguments, int) {
  if (std::wcscmp(arguments, L"gdi-buffer") == 0) {
    mode = mode_e::gdi_buffer;
  } else if (std::wcscmp(arguments, L"gdi-layer") == 0) {
    mode = mode_e::gdi_layer;
  } else if (std::wcscmp(arguments, L"gdi-bits") == 0) {
    mode = mode_e::gdi_bits;
  } else if (std::wcscmp(arguments, L"gdi-glyphs") == 0) {
    mode = mode_e::gdi_glyphs;
  } else if (std::wcscmp(arguments, L"d3d11-layer") == 0) {
    mode = mode_e::d3d11_layer;
  } else if (std::wcscmp(arguments, L"d3d11-direct") == 0) {
    mode = mode_e::d3d11_direct;
  } else if (std::wcscmp(arguments, L"d3d11-own") == 0) {
    mode = mode_e::d3d11_own;
  } else if (std::wcscmp(arguments, L"d3d11-parts") == 0) {
    mode = mode_e::d3d11_parts;
  } else if (std::wcscmp(arguments, L"d3d11-cache") == 0) {
    mode = mode_e::d3d11_cache;
  } else if (std::wcscmp(arguments, L"d3d11-atlas") == 0) {
    mode = mode_e::d3d11_atlas;
  } else if (std::wcscmp(arguments, L"d2d") == 0) {
    mode = mode_e::d2d;
  } else if (std::wcscmp(arguments, L"d2d-busy") == 0) {
    mode = mode_e::d2d_busy;
    client_width = 1920;
    client_height = 1080;
  } else if (std::wcscmp(arguments, L"gdi-window") != 0) {
    return 2;
  }
  font = CreateFontW(-32, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY, DEFAULT_PITCH, L"Segoe UI");
  if (mode == mode_e::d3d11_own || mode == mode_e::d3d11_parts || mode == mode_e::d3d11_atlas) {
    compose_text();
  }
  WNDCLASSW window_class {};
  window_class.lpfnWndProc = procedure;
  window_class.hInstance = instance;
  window_class.lpszClassName = L"SunshineTextHookSample";
  window_class.hCursor = LoadCursorW(nullptr, reinterpret_cast<LPCWSTR>(IDC_ARROW));
  RegisterClassW(&window_class);
  // Topmost so that it is painted, and all but transparent, click-through and never active so
  // that whoever runs the test neither sees it nor loses the focus or a click to it.
  const HWND window = CreateWindowExW(
    WS_EX_TOPMOST | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_LAYERED | WS_EX_TRANSPARENT, window_class.lpszClassName,
    L"Sunshine text helper sample", WS_POPUP, 0, 0, client_width, client_height, nullptr, nullptr, instance, nullptr
  );
  if (!window) {
    return 3;
  }
  SetLayeredWindowAttributes(window, 0u, 1u, LWA_ALPHA);
  ShowWindow(window, SW_SHOWNOACTIVATE);
  if (mode == mode_e::d2d_busy) {
    int frames = 0;
    const ULONGLONG start = GetTickCount64();
    for (ULONGLONG now = start; now < start + 4000u; now = GetTickCount64()) {
      MSG pending {};
      while (PeekMessageW(&pending, nullptr, 0u, 0u, PM_REMOVE)) {
        DispatchMessageW(&pending);
      }
      paint_d2d(window);
      frames += now >= start + 1000u ? 1 : 0;
    }
    return frames;
  }
  SetTimer(window, 1u, 100u, nullptr);
  MSG message {};
  while (GetMessageW(&message, nullptr, 0u, 0u) > 0) {
    TranslateMessage(&message);
    DispatchMessageW(&message);
  }
  return 0;
}
