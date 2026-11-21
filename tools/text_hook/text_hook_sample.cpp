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
 *   d2d          text drawn with Direct2D
 * The window repaints ten times a second. WM_APP + 1 with wParam 0 or 1 hides or shows the
 * text; WM_CLOSE ends the program. Not shipped.
 */
#include <cwchar>
#include <windows.h>

#include <d2d1.h>
#include <dwrite.h>

namespace {
  constexpr int client_width = 640, client_height = 360;
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
    d2d,
  };

  mode_e mode = mode_e::gdi_window;
  bool show_text = true;
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
      const D2D1_HWND_RENDER_TARGET_PROPERTIES window_properties {window, {client_width, client_height}, D2D1_PRESENT_OPTIONS_NONE};
      factory->CreateHwndRenderTarget(&properties, &window_properties, &target);
      dwrite->CreateTextFormat(L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 32.0f, L"en-us", &format);
      if (target) {
        target->CreateSolidColorBrush(D2D1::ColorF(240.0f / 255.0f, 232.0f / 255.0f, 224.0f / 255.0f, 1.0f), &brush);
      }
    }
    if (target && brush && format) {
      target->BeginDraw();
      target->Clear(D2D1::ColorF(24.0f / 255.0f, 32.0f / 255.0f, 64.0f / 255.0f, 1.0f));
      if (show_text) {
        const D2D1_RECT_F rect {static_cast<float>(text_x), static_cast<float>(text_y), 620.0f, 140.0f};
        target->DrawText(line, static_cast<UINT32>(std::wcslen(line)), format, &rect, brush, D2D1_DRAW_TEXT_OPTIONS_NONE, DWRITE_MEASURING_MODE_NATURAL);
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
        InvalidateRect(window, nullptr, FALSE);
        return 0;
      case WM_ERASEBKGND:
        return 1;
      case WM_PAINT:
        if (mode == mode_e::d2d) {
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
  } else if (std::wcscmp(arguments, L"d2d") == 0) {
    mode = mode_e::d2d;
  } else if (std::wcscmp(arguments, L"gdi-window") != 0) {
    return 2;
  }
  font = CreateFontW(-32, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY, DEFAULT_PITCH, L"Segoe UI");
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
  SetTimer(window, 1u, 100u, nullptr);
  MSG message {};
  while (GetMessageW(&message, nullptr, 0u, 0u) > 0) {
    TranslateMessage(&message);
    DispatchMessageW(&message);
  }
  return 0;
}
