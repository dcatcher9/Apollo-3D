// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file tools/text_hook/text_hook.cpp
 * @brief In-application observer of Direct2D / DirectWrite text drawing for Host 3D.
 *
 * Loaded only into an application its owner enabled. It patches the text-drawing slots of the
 * Direct2D render target vtables, repeats each text draw into a private bitmap of its own, and
 * forwards the call unchanged. It keeps no text content and changes nothing the application draws.
 */
#include "text_hook_shared.h"

#include <cstdarg>
#include <cstdio>
#include <cwchar>
#include <windows.h>

#include <d2d1.h>
#include <dwrite.h>

namespace {
  using namespace sunshine_text_hook;

  constexpr unsigned slot_draw_text = 27u;  // ID2D1RenderTarget::DrawText
  constexpr unsigned slot_draw_text_layout = 28u;  // ID2D1RenderTarget::DrawTextLayout
  constexpr unsigned slot_draw_glyph_run = 29u;  // ID2D1RenderTarget::DrawGlyphRun
  constexpr unsigned slot_end_draw = 49u;  // ID2D1RenderTarget::EndDraw
  constexpr unsigned max_patched_vtables = 32u;
  constexpr std::uint64_t replace_after_ms = 100u;
  constexpr std::uint32_t max_surface_extent = 8192u;

  // No C++ runtime library is linked into an application: plain arrays and min/max only.
  constexpr float min_of(const float a, const float b) { return a < b ? a : b; }
  constexpr float max_of(const float a, const float b) { return a > b ? a : b; }

  struct tracked_box_t {
    box_t box;
    std::uint64_t first_tick;
  };

  using draw_text_fn = void(STDMETHODCALLTYPE *)(ID2D1RenderTarget *, const WCHAR *, UINT32, IDWriteTextFormat *, const D2D1_RECT_F *, ID2D1Brush *, D2D1_DRAW_TEXT_OPTIONS, DWRITE_MEASURING_MODE);
  using draw_text_layout_fn = void(STDMETHODCALLTYPE *)(ID2D1RenderTarget *, D2D1_POINT_2F, IDWriteTextLayout *, ID2D1Brush *, D2D1_DRAW_TEXT_OPTIONS);
  using draw_glyph_run_fn = void(STDMETHODCALLTYPE *)(ID2D1RenderTarget *, D2D1_POINT_2F, const DWRITE_GLYPH_RUN *, ID2D1Brush *, DWRITE_MEASURING_MODE);
  using end_draw_fn = HRESULT(STDMETHODCALLTYPE *)(ID2D1RenderTarget *, D2D1_TAG *, D2D1_TAG *);

  // Unlike the text functions, EndDraw differs between render target classes, so each patched
  // vtable keeps its own original.
  struct patched_vtable_t {
    void **vtable;
    end_draw_fn end_draw;
  };
  patched_vtable_t patched_vtables[max_patched_vtables];
  unsigned patched_vtable_count = 0u;
  // The render target the application last drew text on. Compared only, never used.
  ID2D1RenderTarget *text_target = nullptr;

  const patched_vtable_t *originals_of(ID2D1RenderTarget *self) {
    void **vtable = *reinterpret_cast<void ***>(self);
    for (unsigned index = 0u; index < patched_vtable_count; ++index) {
      if (patched_vtables[index].vtable == vtable) {
        return &patched_vtables[index];
      }
    }
    return nullptr;
  }

  CRITICAL_SECTION state_lock;
  tracked_box_t boxes[max_boxes];
  std::uint32_t box_count = 0u;
  snapshot_t *shared = nullptr;
  std::uint32_t generation = 1u;
  FILE *log_file = nullptr;
  unsigned log_lines = 0u;

  draw_text_fn original_draw_text = nullptr;
  draw_text_layout_fn original_draw_text_layout = nullptr;
  draw_glyph_run_fn original_draw_glyph_run = nullptr;

  // The private bitmap: a memory DC whose pixels are the shared coverage section.
  ID2D1Factory *own_factory = nullptr;
  IDWriteFactory *own_dwrite = nullptr;
  ID2D1DCRenderTarget *own_target = nullptr;
  ID2D1SolidColorBrush *own_brush = nullptr;
  HDC own_dc = nullptr;
  HBITMAP own_bitmap = nullptr;
  HANDLE coverage_section = nullptr;
  std::uint32_t surface_width = 0u, surface_height = 0u;
  bool surface_failed = false;
  // The thread that is drawing into the private bitmap right now, under the state lock. Direct2D
  // may call a hooked function again from inside such a draw, on an object of its own; only
  // that thread's calls are the helper's. Another thread's text is the application's.
  volatile DWORD replay_thread = 0u;
  unsigned replay_depth = 0u;

  void begin_replay() {
    if (replay_depth++ == 0u) {
      replay_thread = GetCurrentThreadId();
    }
  }

  void end_replay() {
    if (--replay_depth == 0u) {
      replay_thread = 0u;
    }
  }

  bool replaying_here() {
    return replay_thread == GetCurrentThreadId();
  }
  // An application draws a line of text as hundreds of calls inside one BeginDraw and EndDraw of
  // its own. The private bitmap follows that: one BeginDraw for all of them, and one publication
  // when the application ends its drawing. Ending and publishing for every call made each call
  // cost about a millisecond, which stalled the application for the length of a line.
  bool batch_open = false;  ///< A publication is in progress: the sequence is odd.
  bool own_drawing = false;  ///< The private target is between its BeginDraw and EndDraw.
  std::uint64_t batch_tick = 0u;
  std::uint64_t last_draw_tick = 0u;  ///< The latest text draw of the open batch.
  constexpr std::uint64_t longest_batch_ms = 50u;

  void log_line(const char *format, ...) {
    if (!log_file || log_lines >= 4000u) {
      return;
    }
    ++log_lines;
    va_list arguments;
    va_start(arguments, format);
    std::vfprintf(log_file, format, arguments);
    va_end(arguments);
    std::fputc('\n', log_file);
    std::fflush(log_file);
  }

  float overlap_fraction(const box_t &a, const box_t &b) {
    const float w = min_of(a.right, b.right) - max_of(a.left, b.left);
    const float h = min_of(a.bottom, b.bottom) - max_of(a.top, b.top);
    if (w <= 0.0f || h <= 0.0f) {
      return 0.0f;
    }
    const float smaller = min_of((a.right - a.left) * (a.bottom - a.top), (b.right - b.left) * (b.bottom - b.top));
    return smaller > 0.0f ? w * h / smaller : 0.0f;
  }

  void begin_write() {
    InterlockedIncrement(reinterpret_cast<volatile LONG *>(&shared->sequence));
  }

  void end_write(const std::uint64_t tick) {
    shared->generation = generation;
    shared->change_tick_ms = tick;
    shared->surface_width = surface_width;
    shared->surface_height = surface_height;
    shared->coverage_available = own_target ? 1u : 0u;
    for (std::uint32_t index = 0u; index < box_count; ++index) {
      shared->boxes[index] = boxes[index].box;
    }
    shared->box_count = box_count;
    InterlockedIncrement(reinterpret_cast<volatile LONG *>(&shared->sequence));
  }

  /** Finish the private bitmap's drawing and publish everything recorded since the batch began. */
  void flush_batch(const std::uint64_t tick) {
    if (!batch_open) {
      return;
    }
    if (own_drawing && own_target) {
      begin_replay();
      own_target->EndDraw();
      end_replay();
    }
    own_drawing = false;
    batch_open = false;
    end_write(tick);
  }

  void release_surface() {
    if (own_brush) {
      own_brush->Release();
      own_brush = nullptr;
    }
    if (own_target) {
      own_target->Release();
      own_target = nullptr;
    }
    if (own_dc) {
      DeleteDC(own_dc);
      own_dc = nullptr;
    }
    if (own_bitmap) {
      DeleteObject(own_bitmap);
      own_bitmap = nullptr;
    }
    if (coverage_section) {
      CloseHandle(coverage_section);
      coverage_section = nullptr;
    }
  }

  /** Create the private bitmap for the application's surface size. Its pixels are shared memory. */
  bool ensure_surface(ID2D1RenderTarget *target) {
    const D2D1_SIZE_U size = target->GetPixelSize();
    if (size.width == 0u || size.height == 0u || size.width > max_surface_extent || size.height > max_surface_extent) {
      return false;
    }
    if (own_target && size.width == surface_width && size.height == surface_height) {
      return true;
    }
    if (surface_failed && size.width == surface_width && size.height == surface_height) {
      return false;
    }
    release_surface();
    surface_width = size.width;
    surface_height = size.height;
    surface_failed = true;
    box_count = 0u;
    if (++generation == 0u) {
      ++generation;
    }
    wchar_t name[128];
    std::swprintf(name, 128, L"%ls%lu-%ux%u", coverage_prefix, GetCurrentProcessId(), surface_width, surface_height);
    const DWORD bytes = surface_width * surface_height * 4u;
    coverage_section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0u, bytes, name);
    BITMAPINFO info {};
    info.bmiHeader.biSize = sizeof(info.bmiHeader);
    info.bmiHeader.biWidth = static_cast<LONG>(surface_width);
    info.bmiHeader.biHeight = -static_cast<LONG>(surface_height);  // top row first
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;
    void *bits = nullptr;
    own_dc = CreateCompatibleDC(nullptr);
    if (coverage_section && own_dc) {
      own_bitmap = CreateDIBSection(own_dc, &info, DIB_RGB_COLORS, &bits, coverage_section, 0u);
    }
    if (!own_bitmap || !bits) {
      log_line("coverage bitmap %ux%u unavailable", surface_width, surface_height);
      release_surface();
      return false;
    }
    SelectObject(own_dc, own_bitmap);
    const D2D1_RENDER_TARGET_PROPERTIES properties {
      D2D1_RENDER_TARGET_TYPE_SOFTWARE,
      {DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED},
      96.0f, 96.0f, D2D1_RENDER_TARGET_USAGE_NONE, D2D1_FEATURE_LEVEL_DEFAULT,
    };
    const RECT bounds {0, 0, static_cast<LONG>(surface_width), static_cast<LONG>(surface_height)};
    begin_replay();
    const bool created = own_factory && SUCCEEDED(own_factory->CreateDCRenderTarget(&properties, &own_target)) && own_target &&
                         SUCCEEDED(own_target->BindDC(own_dc, &bounds)) &&
                         SUCCEEDED(own_target->CreateSolidColorBrush(D2D1::ColorF(1.0f, 1.0f, 1.0f, 1.0f), &own_brush));
    if (created) {
      own_target->BeginDraw();
      own_target->Clear(D2D1::ColorF(0.0f, 0.0f, 0.0f, 0.0f));
      own_target->EndDraw();
    }
    end_replay();
    if (!created) {
      log_line("coverage render target %ux%u unavailable", surface_width, surface_height);
      release_surface();
      return false;
    }
    surface_failed = false;
    log_line("coverage surface %ux%u ready", surface_width, surface_height);
    return true;
  }

  /**
   * Record one drawn piece of text. An outline is drawn as several passes one pixel apart, which
   * overlap each other at once and are merged. A box drawn over one that has been there for a
   * while is new text replacing old text, the only signal that the previous text is gone.
   * @return true when this box starts a new generation, so the coverage bitmap must be cleared.
   */
  bool record_locked(const box_t &box, const std::uint64_t tick) {
    tracked_box_t *merge_target = nullptr;
    bool replaces = false;
    for (std::uint32_t index = 0u; index < box_count; ++index) {
      auto &existing = boxes[index];
      if (overlap_fraction(existing.box, box) < 0.5f) {
        continue;
      }
      if (tick - existing.first_tick > replace_after_ms) {
        replaces = true;
        break;
      }
      merge_target = &existing;
    }
    if (replaces) {
      box_count = 0u;
      merge_target = nullptr;
      if (++generation == 0u) {
        ++generation;
      }
    }
    if (merge_target) {
      merge_target->box.left = min_of(merge_target->box.left, box.left);
      merge_target->box.top = min_of(merge_target->box.top, box.top);
      merge_target->box.right = max_of(merge_target->box.right, box.right);
      merge_target->box.bottom = max_of(merge_target->box.bottom, box.bottom);
    } else if (box_count < max_boxes) {
      boxes[box_count++] = {box, tick};
    }
    return replaces;
  }

  /** Bounding box, in surface pixels, of a rectangle in the target's current coordinates. */
  bool surface_box(ID2D1RenderTarget *target, const float x0, const float y0, const float x1, const float y1, box_t &box) {
    D2D1_MATRIX_3X2_F transform;
    target->GetTransform(&transform);
    FLOAT dpi_x = 96.0f, dpi_y = 96.0f;
    target->GetDpi(&dpi_x, &dpi_y);
    const float xs[4] {x0, x1, x0, x1}, ys[4] {y0, y0, y1, y1};
    box = {1e30f, 1e30f, -1e30f, -1e30f};
    for (int corner = 0; corner < 4; ++corner) {
      const float x = (xs[corner] * transform._11 + ys[corner] * transform._21 + transform._31) * dpi_x / 96.0f;
      const float y = (xs[corner] * transform._12 + ys[corner] * transform._22 + transform._32) * dpi_y / 96.0f;
      box.left = min_of(box.left, x);
      box.right = max_of(box.right, x);
      box.top = min_of(box.top, y);
      box.bottom = max_of(box.bottom, y);
    }
    return box.right > box.left && box.bottom > box.top && box.right - box.left < 16384.0f && box.bottom - box.top < 16384.0f;
  }

  /**
   * Common path of the three hooks: note where the text lands, then repeat the same draw, in
   * white, into the private bitmap. `layout` describes the text; `replay` issues the draw.
   */
  template<class Replay>
  void observe(ID2D1RenderTarget *target, IDWriteTextLayout *layout, const float origin_x, const float origin_y, ID2D1Brush *brush, Replay replay) {
    if (!target || !shared || replaying_here() || target == own_target) {
      return;
    }
    EnterCriticalSection(&state_lock);
    const std::uint64_t tick = GetTickCount64();
    if (batch_open) {
      // A different surface size replaces the private bitmap, which must not be mid-draw then.
      const D2D1_SIZE_U size = target->GetPixelSize();
      if (size.width != surface_width || size.height != surface_height) {
        flush_batch(tick);
      }
    }
    const bool surface_ready = ensure_surface(target);
    box_t box {};
    bool have_box = false;
    DWRITE_TEXT_METRICS metrics {};
    if (layout && SUCCEEDED(layout->GetMetrics(&metrics)) && metrics.width > 0.0f && metrics.height > 0.0f) {
      const float left = origin_x + metrics.left, top = origin_y + metrics.top;
      have_box = surface_box(target, left, top, left + metrics.width, top + metrics.height, box);
    }
    if (!batch_open) {
      begin_write();
      batch_open = true;
      batch_tick = tick;
    }
    last_draw_tick = tick;
    text_target = target;
    const bool replaces = have_box && record_locked(box, tick);
    if (surface_ready) {
      begin_replay();
      D2D1_MATRIX_3X2_F transform;
      target->GetTransform(&transform);
      FLOAT dpi_x = 96.0f, dpi_y = 96.0f;
      target->GetDpi(&dpi_x, &dpi_y);
      own_target->SetDpi(dpi_x, dpi_y);
      own_target->SetTransform(&transform);
      own_target->SetTextAntialiasMode(target->GetTextAntialiasMode());
      // The same colour as the application's brush, so the private bitmap shows the text as the
      // application's own surface does: an outline pass dark, the fill over it light.
      D2D1_COLOR_F colour {1.0f, 1.0f, 1.0f, 1.0f};
      ID2D1SolidColorBrush *solid = nullptr;
      if (brush && SUCCEEDED(brush->QueryInterface(__uuidof(ID2D1SolidColorBrush), reinterpret_cast<void **>(&solid))) && solid) {
        colour = solid->GetColor();
        colour.a = 1.0f;
        solid->Release();
      }
      own_brush->SetColor(colour);
      if (!own_drawing) {
        own_target->BeginDraw();
        own_drawing = true;
      }
      if (replaces) {
        own_target->Clear(D2D1::ColorF(0.0f, 0.0f, 0.0f, 0.0f));
      }
      replay(own_target, own_brush);
      end_replay();
    }
    // The application's EndDraw publishes. An application that keeps one drawing open for long
    // is still published in steps.
    if (tick - batch_tick >= longest_batch_ms) {
      flush_batch(tick);
    }
    if (have_box) {
      log_line("text %.1f,%.1f-%.1f,%.1f gen %u", box.left, box.top, box.right, box.bottom, generation);
    }
    LeaveCriticalSection(&state_lock);
  }

  void STDMETHODCALLTYPE draw_text_hook(ID2D1RenderTarget *self, const WCHAR *string, UINT32 length, IDWriteTextFormat *format, const D2D1_RECT_F *rect, ID2D1Brush *brush, D2D1_DRAW_TEXT_OPTIONS options, DWRITE_MEASURING_MODE mode) {
    if (string && length != 0u && format && rect && own_dwrite && !replaying_here() && self != own_target) {
      // Direct2D lays this string out inside the rectangle; measure it the same way.
      IDWriteTextLayout *layout = nullptr;
      if (SUCCEEDED(own_dwrite->CreateTextLayout(string, length, format, max_of(rect->right - rect->left, 0.0f), max_of(rect->bottom - rect->top, 0.0f), &layout)) && layout) {
        observe(self, layout, rect->left, rect->top, brush, [&](ID2D1RenderTarget *own, ID2D1Brush *same) {
          own->DrawText(string, length, format, rect, same, options, mode);
        });
        layout->Release();
      }
    }
    original_draw_text(self, string, length, format, rect, brush, options, mode);
  }

  void STDMETHODCALLTYPE draw_text_layout_hook(ID2D1RenderTarget *self, D2D1_POINT_2F origin, IDWriteTextLayout *layout, ID2D1Brush *brush, D2D1_DRAW_TEXT_OPTIONS options) {
    if (layout && !replaying_here() && self != own_target) {
      observe(self, layout, origin.x, origin.y, brush, [&](ID2D1RenderTarget *own, ID2D1Brush *same) {
        own->DrawTextLayout(origin, layout, same, options);
      });
    }
    original_draw_text_layout(self, origin, layout, brush, options);
  }

  void STDMETHODCALLTYPE draw_glyph_run_hook(ID2D1RenderTarget *self, D2D1_POINT_2F baseline, const DWRITE_GLYPH_RUN *run, ID2D1Brush *brush, DWRITE_MEASURING_MODE mode) {
    if (run && !replaying_here() && self != own_target) {
      // No layout to measure: the coverage bitmap carries this text, without a box.
      observe(self, nullptr, baseline.x, baseline.y, brush, [&](ID2D1RenderTarget *own, ID2D1Brush *same) {
        own->DrawGlyphRun(baseline, run, same, mode);
      });
    }
    original_draw_glyph_run(self, baseline, run, brush, mode);
  }

  HRESULT STDMETHODCALLTYPE end_draw_hook(ID2D1RenderTarget *self, D2D1_TAG *tag1, D2D1_TAG *tag2) {
    const patched_vtable_t *originals = originals_of(self);
    if (shared && !replaying_here() && self != own_target) {
      EnterCriticalSection(&state_lock);
      // The application finished the drawing its text was part of: publish that text as one.
      if (batch_open && self == text_target) {
        flush_batch(GetTickCount64());
      }
      LeaveCriticalSection(&state_lock);
    }
    return originals ? originals->end_draw(self, tag1, tag2) : E_FAIL;
  }

  /**
   * Every Direct2D render target class has its own vtable, but they share the text-drawing
   * implementations. Find each vtable inside d2d1.dll whose three text slots hold those shared
   * functions and point them at the hooks.
   */
  unsigned patch_render_target_vtables(ID2D1RenderTarget *sample) {
    void **sample_vtable = *reinterpret_cast<void ***>(sample);
    void *const text = sample_vtable[slot_draw_text];
    void *const text_layout = sample_vtable[slot_draw_text_layout];
    void *const glyph_run = sample_vtable[slot_draw_glyph_run];
    original_draw_text = reinterpret_cast<draw_text_fn>(text);
    original_draw_text_layout = reinterpret_cast<draw_text_layout_fn>(text_layout);
    original_draw_glyph_run = reinterpret_cast<draw_glyph_run_fn>(glyph_run);

    const HMODULE module = GetModuleHandleW(L"d2d1.dll");
    if (!module) {
      return 0u;
    }
    const auto *base = reinterpret_cast<const std::uint8_t *>(module);
    const auto *dos = reinterpret_cast<const IMAGE_DOS_HEADER *>(base);
    const auto *nt = reinterpret_cast<const IMAGE_NT_HEADERS *>(base + dos->e_lfanew);
    const IMAGE_SECTION_HEADER *sections = IMAGE_FIRST_SECTION(nt);
    unsigned patched = 0u;
    for (unsigned index = 0u; index < nt->FileHeader.NumberOfSections; ++index) {
      const auto &header = sections[index];
      // The vtables may live in a read-only data section or merged into the code section.
      if ((header.Characteristics & IMAGE_SCN_MEM_READ) == 0u) {
        continue;
      }
      auto **begin = reinterpret_cast<void **>(const_cast<std::uint8_t *>(base) + header.VirtualAddress);
      const std::size_t count = header.Misc.VirtualSize / sizeof(void *);
      for (std::size_t word = 0u; word + 2u < count; ++word) {
        // DrawText and DrawTextLayout are adjacent and shared by every class. Some classes have
        // their own DrawGlyphRun; that slot is hooked only where it is the known shared one.
        if (begin[word] != text || begin[word + 1u] != text_layout) {
          continue;
        }
        const bool shared_glyph_run = begin[word + 2u] == glyph_run;
        // The whole vtable, from its first slot to EndDraw, must lie inside this section.
        if (word < slot_draw_text || word - slot_draw_text + slot_end_draw >= count) {
          continue;
        }
        void **vtable = &begin[word - slot_draw_text];
        DWORD old_protection = 0u;
        if (!VirtualProtect(vtable, (slot_end_draw + 1u) * sizeof(void *), PAGE_EXECUTE_READWRITE, &old_protection)) {
          continue;
        }
        vtable[slot_draw_text] = reinterpret_cast<void *>(&draw_text_hook);
        vtable[slot_draw_text_layout] = reinterpret_cast<void *>(&draw_text_layout_hook);
        if (shared_glyph_run) {
          vtable[slot_draw_glyph_run] = reinterpret_cast<void *>(&draw_glyph_run_hook);
        }
        if (patched_vtable_count < max_patched_vtables) {
          patched_vtables[patched_vtable_count++] = {
            vtable,
            reinterpret_cast<end_draw_fn>(vtable[slot_end_draw]),
          };
          vtable[slot_end_draw] = reinterpret_cast<void *>(&end_draw_hook);
        }
        VirtualProtect(vtable, (slot_end_draw + 1u) * sizeof(void *), old_protection, &old_protection);
        ++patched;
      }
    }
    return patched;
  }

  DWORD WINAPI initialize(void *) {
    wchar_t name[96];
    std::swprintf(name, 96, L"%ls%lu", section_prefix, GetCurrentProcessId());
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0u, sizeof(snapshot_t), name);
    if (section) {
      shared = static_cast<snapshot_t *>(MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0u, 0u, sizeof(snapshot_t)));
    }
    wchar_t log_path[MAX_PATH];
    if (GetEnvironmentVariableW(L"SUNSHINE_TEXT_HOOK_LOG", log_path, MAX_PATH) > 0u) {
      log_file = _wfopen(log_path, L"w");
    } else {
      // A diagnostic log is written only where its owner already put the file.
      const DWORD length = GetTempPathW(MAX_PATH, log_path);
      if (length != 0u && length + 24u < MAX_PATH) {
        std::wcscat(log_path, L"SunshineTextHook.log");
        if (GetFileAttributesW(log_path) != INVALID_FILE_ATTRIBUTES) {
          log_file = _wfopen(log_path, L"w");
        }
      }
    }
    if (!shared) {
      return 1u;
    }
    shared->magic = magic;
    shared->version = version;
    shared->generation = generation;

    if (FAILED(D2D1CreateFactory(D2D1_FACTORY_TYPE_MULTI_THREADED, __uuidof(ID2D1Factory), nullptr, reinterpret_cast<void **>(&own_factory))) || !own_factory ||
        FAILED(DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), reinterpret_cast<IUnknown **>(&own_dwrite))) || !own_dwrite) {
      log_line("Direct2D or DirectWrite unavailable");
      return 2u;
    }
    const D2D1_RENDER_TARGET_PROPERTIES properties {
      D2D1_RENDER_TARGET_TYPE_SOFTWARE,
      {DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED},
      96.0f, 96.0f, D2D1_RENDER_TARGET_USAGE_NONE, D2D1_FEATURE_LEVEL_DEFAULT,
    };
    ID2D1DCRenderTarget *sample = nullptr;
    if (FAILED(own_factory->CreateDCRenderTarget(&properties, &sample)) || !sample) {
      log_line("sample render target unavailable");
      return 3u;
    }
    EnterCriticalSection(&state_lock);
    const unsigned patched = patch_render_target_vtables(sample);
    LeaveCriticalSection(&state_lock);
    sample->Release();
    log_line("patched %u render target vtables", patched);
    if (patched == 0u) {
      return 4u;
    }
    // The application's EndDraw normally publishes a line. Where that call is not one of the
    // hooked ones, a line would stay unpublished until more text is drawn, so this thread
    // publishes any batch that has seen no text draw for a moment.
    for (;;) {
      Sleep(static_cast<DWORD>(longest_batch_ms / 2u));
      EnterCriticalSection(&state_lock);
      const std::uint64_t tick = GetTickCount64();
      if (batch_open && tick - last_draw_tick >= longest_batch_ms) {
        flush_batch(tick);
      }
      LeaveCriticalSection(&state_lock);
    }
  }
}  // namespace

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, void *) {
  if (reason == DLL_PROCESS_ATTACH) {
    DisableThreadLibraryCalls(instance);
    InitializeCriticalSection(&state_lock);
    // COM, Direct2D and DirectWrite must not be entered under the loader lock.
    if (HANDLE thread = CreateThread(nullptr, 0u, initialize, nullptr, 0u, nullptr)) {
      CloseHandle(thread);
    }
  }
  return TRUE;
}
