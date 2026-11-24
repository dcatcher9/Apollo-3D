// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file tools/text_hook/text_hook_d2d.cpp
 * @brief The Direct2D part of the text helper.
 *
 * It patches the text-drawing slots of the Direct2D render-target vtables, repeats each text
 * draw of the application on a render target of its own that draws into the published picture,
 * and forwards the call unchanged. A frame that only draws the text of the frame before again is
 * compared with it instead, and costs the application next to nothing.
 */
#include "text_hook_core.h"

#include <cstring>
#include <d2d1.h>
#include <d2d1_1.h>
#include <dwrite.h>

namespace sunshine_text_hook::helper::d2d {
  namespace {
    constexpr unsigned slot_draw_text = 27u;  // ID2D1RenderTarget::DrawText
    constexpr unsigned slot_draw_text_layout = 28u;  // ID2D1RenderTarget::DrawTextLayout
    constexpr unsigned slot_draw_glyph_run = 29u;  // ID2D1RenderTarget::DrawGlyphRun
    constexpr unsigned slot_clear = 47u;  // ID2D1RenderTarget::Clear
    constexpr unsigned slot_end_draw = 49u;  // ID2D1RenderTarget::EndDraw
    constexpr unsigned max_patched_vtables = 32u;
    constexpr unsigned max_boxes = 512u;
    /// A box drawn over one of an earlier batch that is older than this is new text replacing
    /// old text; a younger one is another pass over the same text.
    constexpr std::uint64_t replace_after_ms = 100u;
    /// A render target smaller than the picture is taken for the new picture only once the
    /// larger one has drawn no text for this long.
    constexpr std::uint64_t rebind_after_ms = 2000u;
    /// Text drawn fainter than this is not on screen: a layout pass, or the first step of a fade.
    constexpr float least_visible_alpha = 1.0f / 16.0f;
    /// How many text draws of one frame are kept for comparison, and the room for their strings
    /// and glyph runs. Past either, every draw is repeated on the picture, as all once were.
    constexpr unsigned max_frame_draws = 4096u;
    constexpr std::uint32_t frame_pool_bytes = 1u << 20u;

    /** One drawn piece of text, in pixels of the picture. Half-open. */
    struct box_t {
      float left, top, right, bottom;
    };

    struct tracked_box_t {
      box_t box;
      std::uint64_t first_tick;
      std::uint32_t batch;
    };

    enum class kind_e : std::uint8_t {
      text,
      layout,
      glyph_run,
    };

    /** One text draw of the application: what it drew, and the state it drew it in. */
    struct draw_t {
      kind_e kind;
      bool have_box;
      box_t box;  ///< Where it landed, in pixels of the picture.
      std::uint64_t signature;
      std::uint32_t pool_mark;  ///< Where the frame's pool stood before this draw's own data.
      D2D1_MATRIX_3X2_F transform;
      FLOAT dpi_x, dpi_y;
      D2D1_TEXT_ANTIALIAS_MODE antialias;
      D2D1_COLOR_F colour;
      D2D1_DRAW_TEXT_OPTIONS options;
      DWRITE_MEASURING_MODE mode;
      D2D1_POINT_2F origin;  ///< A layout's origin, or a glyph run's baseline.
      const WCHAR *string;
      UINT32 length;
      IDWriteTextFormat *format;
      D2D1_RECT_F rect;
      IDWriteTextLayout *layout;
      DWRITE_GLYPH_RUN run;
      /// For a text draw: the helper's own layout of the string as it was drawn, made when the
      /// draw was measured. A kept draw is repeated from it, since the application may change
      /// its format, or release it, at any time after the draw.
      IDWriteTextLayout *own_layout;
    };

    using draw_text_fn = void(STDMETHODCALLTYPE *)(ID2D1RenderTarget *, const WCHAR *, UINT32, IDWriteTextFormat *, const D2D1_RECT_F *, ID2D1Brush *, D2D1_DRAW_TEXT_OPTIONS, DWRITE_MEASURING_MODE);
    using draw_text_layout_fn = void(STDMETHODCALLTYPE *)(ID2D1RenderTarget *, D2D1_POINT_2F, IDWriteTextLayout *, ID2D1Brush *, D2D1_DRAW_TEXT_OPTIONS);
    using draw_glyph_run_fn = void(STDMETHODCALLTYPE *)(ID2D1RenderTarget *, D2D1_POINT_2F, const DWRITE_GLYPH_RUN *, ID2D1Brush *, DWRITE_MEASURING_MODE);
    using clear_fn = void(STDMETHODCALLTYPE *)(ID2D1RenderTarget *, const D2D1_COLOR_F *);
    using end_draw_fn = HRESULT(STDMETHODCALLTYPE *)(ID2D1RenderTarget *, D2D1_TAG *, D2D1_TAG *);

    // Unlike the text functions, Clear and EndDraw may differ between render target classes, so
    // each patched vtable keeps its own originals.
    struct patched_vtable_t {
      void **vtable;
      clear_fn clear;
      end_draw_fn end_draw;
    };
    patched_vtable_t patched_vtables[max_patched_vtables];
    unsigned patched_vtable_count = 0u;
    draw_text_fn original_draw_text = nullptr;
    draw_text_layout_fn original_draw_text_layout = nullptr;
    draw_glyph_run_fn original_draw_glyph_run = nullptr;
    bool install_settled = false;  ///< Hooked, or found impossible: nothing more to try.

    ID2D1Factory *own_factory = nullptr;
    IDWriteFactory *own_dwrite = nullptr;
    // The helper's own picture of the text, the size of the published one, and the render target
    // that draws on it.
    HDC own_dc = nullptr;
    HBITMAP own_bitmap = nullptr;
    const std::uint32_t *own_pixels = nullptr;
    ID2D1DCRenderTarget *own_target = nullptr;
    ID2D1SolidColorBrush *own_brush = nullptr;
    bool own_drawing = false;  ///< `own_target` is between its BeginDraw and EndDraw.

    bool seen = false;  ///< The application has drawn text with Direct2D.
    // The render target the application last drew text on, and the bitmap that target drew on
    // then: a device context draws on whichever bitmap is set. Compared only, never used.
    ID2D1RenderTarget *text_target = nullptr;
    const void *text_bitmap = nullptr;
    tracked_box_t boxes[max_boxes];
    std::uint32_t box_count = 0u;
    // A batch is the text published together: an application draws a line as hundreds of calls
    // inside one BeginDraw and EndDraw of its own, and the helper's render target follows that
    // with one BeginDraw, and one publication when the application ends its drawing. Ending and
    // publishing for every call cost about a millisecond each, which stalled the application
    // for the length of a line.
    std::uint32_t batch = 1u;
    bool batch_open = false;  ///< The helper's picture has changed since it was published.
    unsigned batch_draws = 0u;  ///< Text draws repeated in the open batch.
    std::uint32_t published_top = 0u, published_bottom = 0u;  ///< The text rows last published.
    std::uint64_t batch_tick = 0u;
    std::uint64_t last_draw_tick = 0u;  ///< The latest text draw on the picture.

    // The text draws the helper's picture is made of, in order, each with its own copy of what
    // it drew. Most applications wipe their surface and draw the same text again for every frame
    // they show. While the picture is exactly these draws (`frame_exact`), such a frame is only
    // compared with them, draw by draw, and nothing is drawn or published for it: repeating
    // every draw took a busy application several times its own frame time.
    draw_t frame[max_frame_draws];
    unsigned frame_count = 0u;
    std::uint8_t *frame_pool = nullptr;
    std::uint32_t frame_pool_used = 0u;
    bool frame_exact = false;
    bool matching = false;  ///< The surface was wiped: draws are being compared with the frame.
    unsigned match_index = 0u;  ///< How many of the frame's draws have been drawn again.
    bool match_ended = false;  ///< The application ended its drawing before all were.
    std::uint64_t match_ended_tick = 0u;
    // For the diagnostic log.
    unsigned draws_seen = 0u, draws_compared = 0u, frames_repeated = 0u, frames_redone = 0u;
    std::uint64_t counted_tick = 0u;

    std::uint64_t mix(std::uint64_t hash, const void *data, const std::size_t bytes) {
      const auto *byte = static_cast<const std::uint8_t *>(data);
      for (std::size_t index = 0u; index < bytes; ++index) {
        hash = (hash ^ byte[index]) * 1099511628211ull;
      }
      return hash;
    }

    template<class T>
    std::uint64_t mix_value(const std::uint64_t hash, const T &value) {
      return mix(hash, &value, sizeof(value));
    }

    void release_kept(const draw_t &draw) {
      if (draw.kind == kind_e::text) {
        if (draw.own_layout) {
          draw.own_layout->Release();
        }
        if (draw.format) {
          draw.format->Release();
        }
      } else if (draw.kind == kind_e::layout && draw.layout) {
        draw.layout->Release();
      } else if (draw.kind == kind_e::glyph_run && draw.run.fontFace) {
        draw.run.fontFace->Release();
      }
    }

    /** Keep only the first `count` draws of the frame. */
    void truncate_frame(const unsigned count) {
      if (count >= frame_count) {
        return;
      }
      for (unsigned index = count; index < frame_count; ++index) {
        release_kept(frame[index]);
      }
      frame_pool_used = frame[count].pool_mark;
      frame_count = count;
    }

    /** The picture is no longer known draw by draw. */
    void forget_frame() {
      truncate_frame(0u);
      frame_pool_used = 0u;
      frame_exact = false;
      matching = false;
    }

    /** Room in the frame's pool, aligned for anything kept there. Null when it is full. */
    void *pool_room(const std::size_t bytes) {
      const std::uint32_t start = (frame_pool_used + 7u) & ~7u;
      if (!frame_pool || bytes > frame_pool_bytes || start > frame_pool_bytes - bytes) {
        return nullptr;
      }
      frame_pool_used = start + static_cast<std::uint32_t>(bytes);
      return frame_pool + start;
    }

    /**
     * Keep a draw that was just repeated on the picture, with its own copy of what it drew.
     * False where it is not kept: the draw's own layout is then the caller's to release.
     */
    bool keep(const draw_t &draw) {
      if (!frame_exact) {
        return false;
      }
      if (!frame_pool) {
        frame_pool = static_cast<std::uint8_t *>(HeapAlloc(GetProcessHeap(), 0u, frame_pool_bytes));
      }
      draw_t kept = draw;
      kept.pool_mark = frame_pool_used;
      bool room = frame_count < max_frame_draws && frame_pool;
      if (room && draw.kind == kind_e::text && draw.own_layout) {
        // The layout holds the string and the format as they were; nothing of the application's
        // is kept.
        kept.string = nullptr;
        kept.format = nullptr;
      } else if (room && draw.kind == kind_e::text) {
        auto *string = static_cast<WCHAR *>(pool_room(static_cast<std::size_t>(draw.length) * sizeof(WCHAR)));
        room = string != nullptr;
        if (room) {
          std::memcpy(string, draw.string, static_cast<std::size_t>(draw.length) * sizeof(WCHAR));
          kept.string = string;
        }
      } else if (room && draw.kind == kind_e::glyph_run) {
        const std::size_t count = draw.run.glyphCount;
        auto *indices = draw.run.glyphIndices ? static_cast<UINT16 *>(pool_room(count * sizeof(UINT16))) : nullptr;
        auto *advances = draw.run.glyphAdvances ? static_cast<FLOAT *>(pool_room(count * sizeof(FLOAT))) : nullptr;
        auto *offsets = draw.run.glyphOffsets ? static_cast<DWRITE_GLYPH_OFFSET *>(pool_room(count * sizeof(DWRITE_GLYPH_OFFSET))) : nullptr;
        room = indices && (advances || !draw.run.glyphAdvances) && (offsets || !draw.run.glyphOffsets);
        if (room) {
          std::memcpy(indices, draw.run.glyphIndices, count * sizeof(UINT16));
          if (advances) {
            std::memcpy(advances, draw.run.glyphAdvances, count * sizeof(FLOAT));
          }
          if (offsets) {
            std::memcpy(offsets, draw.run.glyphOffsets, count * sizeof(DWRITE_GLYPH_OFFSET));
          }
          kept.run.glyphIndices = indices;
          kept.run.glyphAdvances = advances;
          kept.run.glyphOffsets = offsets;
        }
      }
      if (!room) {
        // More than is kept for one frame: every draw is repeated again from here on.
        forget_frame();
        return false;
      }
      // The application may release these at once; the kept draw may be repeated much later.
      if (kept.kind == kind_e::text) {
        if (kept.format) {
          kept.format->AddRef();
        }
      } else if (kept.kind == kind_e::layout) {
        kept.layout->AddRef();
      } else if (kept.run.fontFace) {
        kept.run.fontFace->AddRef();
      }
      frame[frame_count++] = kept;
      return true;
    }

    const patched_vtable_t *originals_of(ID2D1RenderTarget *self) {
      void **vtable = *reinterpret_cast<void ***>(self);
      for (unsigned index = 0u; index < patched_vtable_count; ++index) {
        if (patched_vtables[index].vtable == vtable) {
          return &patched_vtables[index];
        }
      }
      return nullptr;
    }

    D2D1_RENDER_TARGET_PROPERTIES own_properties() {
      return {
        D2D1_RENDER_TARGET_TYPE_SOFTWARE,
        {DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED},
        96.0f, 96.0f, D2D1_RENDER_TARGET_USAGE_NONE, D2D1_FEATURE_LEVEL_DEFAULT,
      };
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

    /** The bitmap a render target draws on now, where it is one that can be given another. */
    const void *bitmap_of(ID2D1RenderTarget *target) {
      ID2D1DeviceContext *context = nullptr;
      ID2D1Image *image = nullptr;
      if (SUCCEEDED(target->QueryInterface(__uuidof(ID2D1DeviceContext), reinterpret_cast<void **>(&context))) && context) {
        context->GetTarget(&image);
        if (image) {
          image->Release();
        }
        context->Release();
      }
      return image;
    }

    void release_own() {
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
      own_pixels = nullptr;
      own_drawing = false;
    }

    /** Finish the helper's drawing and publish everything recorded since the batch began. */
    void flush_batch() {
      if (!batch_open) {
        return;
      }
      batch_open = false;
      if (own_drawing && own_target) {
        enter_call();
        own_target->EndDraw();
        leave_call();
      }
      own_drawing = false;
      // The rows around every box, and two more: an outline pass and antialiasing reach just
      // past the measured text.
      float top = 1e30f, bottom = -1e30f;
      for (std::uint32_t index = 0u; index < box_count; ++index) {
        top = min_of(top, boxes[index].box.top);
        bottom = max_of(bottom, boxes[index].box.bottom);
      }
      std::uint32_t first = 0u, end = 0u;
      if (bottom > top && bottom > 0.0f && top < static_cast<float>(surface.height)) {
        const float clipped_top = max_of(top - 2.0f, 0.0f);
        const float clipped_end = min_of(bottom + 3.0f, static_cast<float>(surface.height));
        first = static_cast<std::uint32_t>(clipped_top);
        end = static_cast<std::uint32_t>(clipped_end);
      }
      // The rows that held text before are published too: that text may be gone from them.
      const bool text = end > first, before = published_bottom > published_top;
      publish(own_pixels, text && before ? min_of(first, published_top) : text ? first : published_top, max_of(end, published_bottom), source_direct2d, first, end, nullptr);
      published_top = text ? first : 0u;
      published_bottom = text ? end : 0u;
      ++batch;
      batch_draws = 0u;
    }

    bool fits(const D2D1_SIZE_U size) {
      return surface.pixels && size.width == surface.width && size.height == surface.height;
    }

    /** Whether text on a render target of this size belongs in the picture: see bind(). */
    bool takes(const D2D1_SIZE_U size, const std::uint64_t tick) {
      return fits(size) || !own_target || !surface.pixels ||
             static_cast<std::uint64_t>(size.width) * size.height >= static_cast<std::uint64_t>(surface.width) * surface.height ||
             tick - last_draw_tick >= rebind_after_ms;
    }

    /**
     * Make the published picture the one of this render target. An application may also draw
     * text on smaller targets of its own, which are not its picture; following each in turn
     * would rebuild the picture at every draw.
     */
    bool bind(ID2D1RenderTarget *target, const std::uint64_t tick) {
      const D2D1_SIZE_U size = target->GetPixelSize();
      if (fits(size) && own_target) {
        return true;
      }
      if (!takes(size, tick)) {
        return false;
      }
      // The helper's render target must not be mid-draw when its picture is replaced.
      flush_batch();
      forget_frame();
      release_own();
      box_count = 0u;
      if (!ensure_surface(size.width, size.height)) {
        return false;
      }
      BITMAPINFO info {};
      info.bmiHeader.biSize = sizeof(info.bmiHeader);
      info.bmiHeader.biWidth = static_cast<LONG>(surface.width);
      info.bmiHeader.biHeight = -static_cast<LONG>(surface.height);  // top row first
      info.bmiHeader.biPlanes = 1;
      info.bmiHeader.biBitCount = 32;
      info.bmiHeader.biCompression = BI_RGB;
      void *bits = nullptr;
      own_dc = CreateCompatibleDC(nullptr);
      own_bitmap = own_dc ? CreateDIBSection(own_dc, &info, DIB_RGB_COLORS, &bits, nullptr, 0u) : nullptr;
      const RECT bounds {0, 0, static_cast<LONG>(surface.width), static_cast<LONG>(surface.height)};
      const auto properties = own_properties();
      enter_call();
      const bool created = own_bitmap && bits && SelectObject(own_dc, own_bitmap) && own_factory &&
                           SUCCEEDED(own_factory->CreateDCRenderTarget(&properties, &own_target)) && own_target &&
                           SUCCEEDED(own_target->BindDC(own_dc, &bounds)) &&
                           SUCCEEDED(own_target->CreateSolidColorBrush(D2D1::ColorF(1.0f, 1.0f, 1.0f, 1.0f), &own_brush));
      leave_call();
      if (!created) {
        log_line("Direct2D render target %ux%u unavailable", surface.width, surface.height);
        release_own();
        return false;
      }
      own_pixels = static_cast<const std::uint32_t *>(bits);
      // Whatever another source left in the published picture is not this text.
      publish(own_pixels, 0u, surface.height, source_direct2d, 0u, 0u, nullptr);
      published_top = published_bottom = 0u;
      frame_exact = true;  // an empty picture and no draws
      return true;
    }

    /** True when `box` lies over text of an earlier batch that has been there for a while. */
    bool replaces_older(const box_t &box, const std::uint64_t tick) {
      for (std::uint32_t index = 0u; index < box_count; ++index) {
        const auto &existing = boxes[index];
        if (existing.batch != batch && tick - existing.first_tick > replace_after_ms && overlap_fraction(existing.box, box) >= 0.5f) {
          return true;
        }
      }
      return false;
    }

    /**
     * Record one drawn piece. An outline is drawn as several passes one pixel apart, which
     * overlap each other at once and are merged.
     */
    void record(const box_t &box, const std::uint64_t tick) {
      for (std::uint32_t index = 0u; index < box_count; ++index) {
        auto &existing = boxes[index].box;
        // Past the limit everything joins the last box: only the rows around all of them count.
        if (overlap_fraction(existing, box) >= 0.5f || (box_count == max_boxes && index + 1u == box_count)) {
          existing = {min_of(existing.left, box.left), min_of(existing.top, box.top), max_of(existing.right, box.right), max_of(existing.bottom, box.bottom)};
          return;
        }
      }
      boxes[box_count++] = {box, tick, batch};
    }

    /**
     * The text of earlier batches is gone: new text was drawn over it, the only signal there is.
     * Remove it from the picture, which is mid-draw, and keep what this batch already drew.
     */
    void retire_older() {
      std::uint32_t kept = 0u;
      own_target->SetDpi(96.0f, 96.0f);
      own_target->SetTransform(D2D1::Matrix3x2F::Identity());
      if (batch_draws == 0u) {
        own_target->Clear(D2D1::ColorF(0.0f, 0.0f, 0.0f, 0.0f));
        truncate_frame(0u);
        frame_exact = true;
      } else {
        // Parts of the picture are wiped below: it is no longer the kept draws.
        forget_frame();
      }
      for (std::uint32_t index = 0u; index < box_count; ++index) {
        const auto &existing = boxes[index];
        if (existing.batch == batch) {
          boxes[kept++] = existing;
        } else if (batch_draws != 0u) {
          const auto &old = existing.box;
          own_target->PushAxisAlignedClip(D2D1::RectF(old.left - 2.0f, old.top - 2.0f, old.right + 2.0f, old.bottom + 2.0f), D2D1_ANTIALIAS_MODE_ALIASED);
          own_target->Clear(D2D1::ColorF(0.0f, 0.0f, 0.0f, 0.0f));
          own_target->PopAxisAlignedClip();
        }
      }
      box_count = kept;
    }

    /** The rectangle of a glyph run from its font's metrics: there is no layout to measure. */
    bool glyph_run_box(const D2D1_POINT_2F baseline, const DWRITE_GLYPH_RUN &run, box_t &box) {
      if (!run.fontFace || run.glyphCount == 0u || !(run.fontEmSize > 0.0f)) {
        return false;
      }
      DWRITE_FONT_METRICS metrics {};
      run.fontFace->GetMetrics(&metrics);
      if (metrics.designUnitsPerEm == 0u) {
        return false;
      }
      const float scale = run.fontEmSize / static_cast<float>(metrics.designUnitsPerEm);
      float advance = 0.0f;
      if (run.glyphAdvances) {
        for (UINT32 index = 0u; index < run.glyphCount; ++index) {
          advance += run.glyphAdvances[index];
        }
      } else if (run.glyphIndices) {
        DWRITE_GLYPH_METRICS glyphs[64];
        for (UINT32 first = 0u; first < run.glyphCount; first += 64u) {
          const UINT32 count = min_of<UINT32>(64u, run.glyphCount - first);
          if (FAILED(run.fontFace->GetDesignGlyphMetrics(run.glyphIndices + first, count, glyphs, run.isSideways))) {
            return false;
          }
          for (UINT32 index = 0u; index < count; ++index) {
            advance += static_cast<float>(run.isSideways ? glyphs[index].advanceHeight : glyphs[index].advanceWidth) * scale;
          }
        }
      }
      const float ascent = static_cast<float>(metrics.ascent) * scale, descent = static_cast<float>(metrics.descent) * scale;
      const bool reversed = (run.bidiLevel & 1u) != 0u;
      const float start = reversed ? -advance : 0.0f, end = reversed ? 0.0f : advance;
      // A sideways run advances down the page, with its glyphs lying across the baseline.
      box = run.isSideways ? box_t {baseline.x - ascent, baseline.y + start, baseline.x + descent, baseline.y + end} :
                             box_t {baseline.x + start, baseline.y - ascent, baseline.x + end, baseline.y + descent};
      return box.right > box.left && box.bottom > box.top;
    }

    /**
     * What tells one draw from another: everything that decides what lands where. A text format
     * goes in by its properties, since an application may make a new one for every frame; a
     * layout and a font face can only go in as the objects they are, which stay the same objects
     * for as long as a kept draw holds them.
     */
    std::uint64_t sign(const draw_t &draw, const ID2D1RenderTarget *target) {
      std::uint64_t hash = 14695981039346656037ull;
      hash = mix_value(hash, draw.kind);
      hash = mix_value(hash, target);
      hash = mix_value(hash, draw.transform);
      hash = mix_value(hash, draw.dpi_x);
      hash = mix_value(hash, draw.dpi_y);
      hash = mix_value(hash, draw.antialias);
      hash = mix_value(hash, draw.colour);
      hash = mix_value(hash, draw.options);
      hash = mix_value(hash, draw.mode);
      if (draw.kind == kind_e::text) {
        hash = mix(hash, draw.string, static_cast<std::size_t>(draw.length) * sizeof(WCHAR));
        hash = mix_value(hash, draw.rect);
        WCHAR family[64] {};
        draw.format->GetFontFamilyName(family, 64u);
        hash = mix(hash, family, sizeof(family));
        hash = mix_value(hash, draw.format->GetFontSize());
        hash = mix_value(hash, draw.format->GetFontWeight());
        hash = mix_value(hash, draw.format->GetFontStyle());
        hash = mix_value(hash, draw.format->GetFontStretch());
        hash = mix_value(hash, draw.format->GetTextAlignment());
        hash = mix_value(hash, draw.format->GetParagraphAlignment());
        hash = mix_value(hash, draw.format->GetWordWrapping());
      } else if (draw.kind == kind_e::layout) {
        hash = mix_value(hash, draw.layout);
        hash = mix_value(hash, draw.origin);
        hash = mix_value(hash, draw.layout->GetMaxWidth());
        hash = mix_value(hash, draw.layout->GetMaxHeight());
        // An application may restyle a layout it keeps; what it then measures tells.
        DWRITE_TEXT_METRICS metrics {};
        if (SUCCEEDED(draw.layout->GetMetrics(&metrics))) {
          hash = mix_value(hash, metrics.width);
          hash = mix_value(hash, metrics.height);
          hash = mix_value(hash, metrics.lineCount);
        }
      } else {
        const auto &run = draw.run;
        hash = mix_value(hash, draw.origin);
        hash = mix_value(hash, run.fontFace);
        hash = mix_value(hash, run.fontEmSize);
        hash = mix_value(hash, run.glyphCount);
        hash = mix_value(hash, run.isSideways);
        hash = mix_value(hash, run.bidiLevel);
        if (run.glyphIndices) {
          hash = mix(hash, run.glyphIndices, static_cast<std::size_t>(run.glyphCount) * sizeof(UINT16));
        }
        if (run.glyphAdvances) {
          hash = mix(hash, run.glyphAdvances, static_cast<std::size_t>(run.glyphCount) * sizeof(FLOAT));
        }
        if (run.glyphOffsets) {
          hash = mix(hash, run.glyphOffsets, static_cast<std::size_t>(run.glyphCount) * sizeof(DWRITE_GLYPH_OFFSET));
        }
      }
      return hash;
    }

    /** Where a draw lands, in pixels of the picture. False where that cannot be told. */
    bool measure(draw_t &draw, box_t &box) {
      box_t local {};
      bool measured = false;
      if (draw.kind == kind_e::text) {
        // Direct2D lays this string out inside the rectangle; measure it the same way. The
        // layout is the draw's own from here on: it is what a kept draw is repeated from.
        IDWriteTextLayout *layout = nullptr;
        DWRITE_TEXT_METRICS metrics {};
        const auto &rect = draw.rect;
        if (SUCCEEDED(own_dwrite->CreateTextLayout(draw.string, draw.length, draw.format, max_of(rect.right - rect.left, 0.0f), max_of(rect.bottom - rect.top, 0.0f), &layout)) && layout) {
          measured = SUCCEEDED(layout->GetMetrics(&metrics)) && metrics.width > 0.0f && metrics.height > 0.0f;
          // Direct2D measures a GDI-compatible draw otherwise than this layout does: such a
          // draw is repeated from the application's own string and format.
          if (draw.mode == DWRITE_MEASURING_MODE_NATURAL) {
            draw.own_layout = layout;
          } else {
            layout->Release();
          }
          local = {rect.left + metrics.left, rect.top + metrics.top, rect.left + metrics.left + metrics.width, rect.top + metrics.top + metrics.height};
        }
      } else if (draw.kind == kind_e::layout) {
        DWRITE_TEXT_METRICS metrics {};
        measured = SUCCEEDED(draw.layout->GetMetrics(&metrics)) && metrics.width > 0.0f && metrics.height > 0.0f;
        local = {draw.origin.x + metrics.left, draw.origin.y + metrics.top, draw.origin.x + metrics.left + metrics.width, draw.origin.y + metrics.top + metrics.height};
      } else {
        measured = glyph_run_box(draw.origin, draw.run, local);
      }
      if (!measured) {
        return false;
      }
      box = {1e30f, 1e30f, -1e30f, -1e30f};
      const auto &transform = draw.transform;
      const float xs[4] {local.left, local.right, local.left, local.right}, ys[4] {local.top, local.top, local.bottom, local.bottom};
      for (int corner = 0; corner < 4; ++corner) {
        const float x = (xs[corner] * transform._11 + ys[corner] * transform._21 + transform._31) * draw.dpi_x / 96.0f;
        const float y = (xs[corner] * transform._12 + ys[corner] * transform._22 + transform._32) * draw.dpi_y / 96.0f;
        box = {min_of(box.left, x), min_of(box.top, y), max_of(box.right, x), max_of(box.bottom, y)};
      }
      return box.right > box.left && box.bottom > box.top && box.right - box.left < 16384.0f && box.bottom - box.top < 16384.0f;
    }

    /** The picture is about to change: open the batch and the helper's own drawing. */
    void open_batch(const std::uint64_t tick) {
      if (!batch_open) {
        batch_open = true;
        batch_tick = tick;
      }
      if (!own_drawing) {
        own_target->BeginDraw();
        own_drawing = true;
      }
    }

    /**
     * Repeat a draw on the helper's picture, which is mid-draw, in the application's colour, so
     * the picture shows the text as the application's own surface does: an outline pass dark,
     * the fill over it light.
     */
    void issue(const draw_t &draw) {
      own_target->SetDpi(draw.dpi_x, draw.dpi_y);
      own_target->SetTransform(&draw.transform);
      own_target->SetTextAntialiasMode(draw.antialias);
      own_brush->SetColor(draw.colour);
      if (draw.kind == kind_e::text && !draw.string) {
        // A kept draw: from the helper's own layout of it.
        own_target->DrawTextLayout(D2D1::Point2F(draw.rect.left, draw.rect.top), draw.own_layout, own_brush, draw.options);
      } else if (draw.kind == kind_e::text) {
        own_target->DrawText(draw.string, draw.length, draw.format, &draw.rect, own_brush, draw.options, draw.mode);
      } else if (draw.kind == kind_e::layout) {
        own_target->DrawTextLayout(draw.origin, draw.layout, own_brush, draw.options);
      } else {
        own_target->DrawGlyphRun(draw.origin, &draw.run, own_brush, draw.mode);
      }
      ++batch_draws;
    }

    /**
     * The comparison is over. Where the application drew every kept draw again, the picture
     * stands as it is. Where it stopped short, or drew something else, the picture is redone as
     * the draws that were drawn again, from their kept copies.
     */
    void settle_match(const std::uint64_t tick) {
      matching = false;
      if (match_index >= frame_count) {
        ++frames_repeated;
        // The application has just drawn all of this again: its text is of now, not text that
        // a draw laid over it would replace.
        for (std::uint32_t index = 0u; index < box_count; ++index) {
          boxes[index].first_tick = tick;
          boxes[index].batch = batch;
        }
        return;
      }
      ++frames_redone;
      truncate_frame(match_index);
      enter_call();
      open_batch(tick);
      own_target->Clear(D2D1::ColorF(0.0f, 0.0f, 0.0f, 0.0f));
      box_count = 0u;
      batch_draws = 0u;
      for (unsigned index = 0u; index < frame_count; ++index) {
        // A draw kept by the application's own objects is repeated only while they are as they
        // were. From one that has changed on, the draws are given up: the application draws
        // them again, or they are gone.
        if (!(frame[index].kind == kind_e::text && !frame[index].string) && sign(frame[index], text_target) != frame[index].signature) {
          truncate_frame(index);
          break;
        }
        if (frame[index].have_box) {
          record(frame[index].box, tick);
        }
        issue(frame[index]);
      }
      leave_call();
    }

    /**
     * Common path of the three hooks. A draw that repeats the kept one in its place is only
     * counted. Any other is repeated on the helper's render target, where it lands is noted,
     * and it is kept for the frames to come.
     */
    void observe(ID2D1RenderTarget *target, ID2D1Brush *brush, draw_t &draw) {
      EnterCriticalSection(&state_lock);
      const std::uint64_t tick = GetTickCount64();
      if (!seen) {
        seen = true;
        gdi::retire();
      }
      // The brush's alpha is left out of the colour: whether faint text is on screen is the
      // host's finding, from the picture.
      draw.colour = {1.0f, 1.0f, 1.0f, 1.0f};
      float alpha = brush ? brush->GetOpacity() : 1.0f;
      ID2D1SolidColorBrush *solid = nullptr;
      if (brush && SUCCEEDED(brush->QueryInterface(__uuidof(ID2D1SolidColorBrush), reinterpret_cast<void **>(&solid))) && solid) {
        draw.colour = solid->GetColor();
        alpha *= draw.colour.a;
        draw.colour.a = 1.0f;
        solid->Release();
      }
      if (alpha < least_visible_alpha) {
        LeaveCriticalSection(&state_lock);
        return;
      }
      target->GetTransform(&draw.transform);
      target->GetDpi(&draw.dpi_x, &draw.dpi_y);
      draw.antialias = target->GetTextAntialiasMode();
      draw.signature = sign(draw, target);
      ++draws_seen;
      if (matching) {
        if (target == text_target && match_index < frame_count && frame[match_index].signature == draw.signature) {
          ++match_index;
          ++draws_compared;
          match_ended = false;
          last_draw_tick = tick;
          LeaveCriticalSection(&state_lock);
          return;
        }
        // Text on a target that is not the picture's is no part of the frame, nor its end.
        if (!takes(target->GetPixelSize(), tick)) {
          LeaveCriticalSection(&state_lock);
          return;
        }
        settle_match(tick);
      }
      if (!bind(target, tick)) {
        LeaveCriticalSection(&state_lock);
        return;
      }
      draw.have_box = measure(draw, draw.box);
      last_draw_tick = tick;
      text_target = target;
      text_bitmap = bitmap_of(target);
      enter_call();
      open_batch(tick);
      if (draw.have_box) {
        if (replaces_older(draw.box, tick)) {
          retire_older();
        }
        record(draw.box, tick);
      }
      issue(draw);
      leave_call();
      if (!keep(draw) && draw.own_layout) {
        draw.own_layout->Release();
      }
      // The first draws show how the application draws its text; the log is for more than them.
      static unsigned logged_draws = 0u;
      if (draw.have_box && logged_draws < 600u) {
        ++logged_draws;
        log_line("text %.1f,%.1f-%.1f,%.1f batch %u kind %u", draw.box.left, draw.box.top, draw.box.right, draw.box.bottom, batch, static_cast<unsigned>(draw.kind));
      }
      // The application's EndDraw publishes. An application that keeps one drawing open for long
      // is still published in steps.
      if (tick - batch_tick >= longest_batch_ms) {
        flush_batch();
      }
      LeaveCriticalSection(&state_lock);
    }

    void STDMETHODCALLTYPE draw_text_hook(ID2D1RenderTarget *self, const WCHAR *string, UINT32 length, IDWriteTextFormat *format, const D2D1_RECT_F *rect, ID2D1Brush *brush, D2D1_DRAW_TEXT_OPTIONS options, DWRITE_MEASURING_MODE mode) {
      if (!nested() && string && length != 0u && length <= 65536u && format && rect && own_dwrite) {
        draw_t draw {};
        draw.kind = kind_e::text;
        draw.string = string;
        draw.length = length;
        draw.format = format;
        draw.rect = *rect;
        draw.options = options;
        draw.mode = mode;
        observe(self, brush, draw);
      }
      enter_call();
      original_draw_text(self, string, length, format, rect, brush, options, mode);
      leave_call();
    }

    void STDMETHODCALLTYPE draw_text_layout_hook(ID2D1RenderTarget *self, D2D1_POINT_2F origin, IDWriteTextLayout *layout, ID2D1Brush *brush, D2D1_DRAW_TEXT_OPTIONS options) {
      if (!nested() && layout) {
        draw_t draw {};
        draw.kind = kind_e::layout;
        draw.layout = layout;
        draw.origin = origin;
        draw.options = options;
        observe(self, brush, draw);
      }
      enter_call();
      original_draw_text_layout(self, origin, layout, brush, options);
      leave_call();
    }

    void STDMETHODCALLTYPE draw_glyph_run_hook(ID2D1RenderTarget *self, D2D1_POINT_2F baseline, const DWRITE_GLYPH_RUN *run, ID2D1Brush *brush, DWRITE_MEASURING_MODE mode) {
      if (!nested() && run && run->glyphCount != 0u && run->glyphCount <= 65536u && run->glyphIndices) {
        draw_t draw {};
        draw.kind = kind_e::glyph_run;
        draw.run = *run;
        draw.origin = baseline;
        draw.mode = mode;
        observe(self, brush, draw);
      }
      enter_call();
      original_draw_glyph_run(self, baseline, run, brush, mode);
      leave_call();
    }

    void STDMETHODCALLTYPE clear_hook(ID2D1RenderTarget *self, const D2D1_COLOR_F *colour) {
      const patched_vtable_t *originals = originals_of(self);
      if (!nested() && seen) {
        EnterCriticalSection(&state_lock);
        // The application wiped the surface its text is on: that text is gone, whatever it draws
        // next. This is the only erasure Direct2D names; text painted over is not noticed here.
        if (self == text_target && own_target && bitmap_of(self) == text_bitmap) {
          const std::uint64_t tick = GetTickCount64();
          // Wiped again with nothing drawn since the last wipe: still the same wipe, also where
          // the application ended a drawing in between, unless that was long enough ago for the
          // text to be gone.
          const bool same_wipe = matching && match_index == 0u && !(match_ended && tick - match_ended_tick >= longest_batch_ms);
          if (matching && !same_wipe) {
            settle_match(tick);  // the frame before ended short of its text
          }
          if (same_wipe) {
            // The comparison goes on from where it stands.
          } else if (frame_exact && frame_count != 0u && fits(self->GetPixelSize())) {
            // Most likely the same text follows: compare before wiping anything.
            matching = true;
            match_index = 0u;
            match_ended = false;
          } else if (!frame_exact || frame_count != 0u || box_count != 0u) {
            enter_call();
            open_batch(tick);
            own_target->Clear(D2D1::ColorF(0.0f, 0.0f, 0.0f, 0.0f));
            leave_call();
            box_count = 0u;
            batch_draws = 0u;
            truncate_frame(0u);
            frame_exact = true;
          }
        }
        LeaveCriticalSection(&state_lock);
      }
      if (originals) {
        originals->clear(self, colour);
      }
    }

    HRESULT STDMETHODCALLTYPE end_draw_hook(ID2D1RenderTarget *self, D2D1_TAG *tag1, D2D1_TAG *tag2) {
      const patched_vtable_t *originals = originals_of(self);
      if (!nested()) {
        EnterCriticalSection(&state_lock);
        // The application finished the drawing its text was part of: publish that text as one.
        if (seen && self == text_target) {
          if (matching && match_index < frame_count && !match_ended) {
            // It may go on drawing the rest in another step; the idle check tells if it does not.
            match_ended = true;
            match_ended_tick = GetTickCount64();
          }
          flush_batch();
        }
        LeaveCriticalSection(&state_lock);
      }
      return originals ? originals->end_draw(self, tag1, tag2) : E_FAIL;
    }

    /**
     * Every Direct2D render target class has its own vtable, but they share the text-drawing
     * implementations. Find each vtable inside d2d1.dll whose text slots hold those shared
     * functions and point them at the hooks.
     */
    unsigned patch_render_target_vtables(const HMODULE module, ID2D1RenderTarget *sample) {
      void **sample_vtable = *reinterpret_cast<void ***>(sample);
      void *const text = sample_vtable[slot_draw_text];
      void *const text_layout = sample_vtable[slot_draw_text_layout];
      void *const glyph_run = sample_vtable[slot_draw_glyph_run];
      original_draw_text = reinterpret_cast<draw_text_fn>(text);
      original_draw_text_layout = reinterpret_cast<draw_text_layout_fn>(text_layout);
      original_draw_glyph_run = reinterpret_cast<draw_glyph_run_fn>(glyph_run);

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
          if (word < slot_draw_text || word - slot_draw_text + slot_end_draw >= count || patched_vtable_count == max_patched_vtables) {
            continue;
          }
          void **vtable = &begin[word - slot_draw_text];
          DWORD old_protection = 0u;
          if (!VirtualProtect(vtable, (slot_end_draw + 1u) * sizeof(void *), PAGE_EXECUTE_READWRITE, &old_protection)) {
            continue;
          }
          // EndDraw first: once a text slot is patched, its batch needs the hook that ends it.
          patched_vtables[patched_vtable_count++] = {vtable, reinterpret_cast<clear_fn>(vtable[slot_clear]), reinterpret_cast<end_draw_fn>(vtable[slot_end_draw])};
          vtable[slot_end_draw] = reinterpret_cast<void *>(&end_draw_hook);
          vtable[slot_clear] = reinterpret_cast<void *>(&clear_hook);
          vtable[slot_draw_text] = reinterpret_cast<void *>(&draw_text_hook);
          vtable[slot_draw_text_layout] = reinterpret_cast<void *>(&draw_text_layout_hook);
          if (shared_glyph_run) {
            vtable[slot_draw_glyph_run] = reinterpret_cast<void *>(&draw_glyph_run_hook);
          }
          VirtualProtect(vtable, (slot_end_draw + 1u) * sizeof(void *), old_protection, &old_protection);
          ++patched;
        }
      }
      return patched;
    }
  }  // namespace

  bool install() {
    if (install_settled) {
      return true;
    }
    // The helper brings neither library into an application that does not use them.
    const HMODULE direct2d = GetModuleHandleW(L"d2d1.dll");
    const HMODULE directwrite = GetModuleHandleW(L"dwrite.dll");
    if (!direct2d || !directwrite) {
      return false;
    }
    install_settled = true;
    using create_d2d_fn = HRESULT(WINAPI *)(D2D1_FACTORY_TYPE, REFIID, const D2D1_FACTORY_OPTIONS *, void **);
    using create_dwrite_fn = HRESULT(WINAPI *)(DWRITE_FACTORY_TYPE, REFIID, IUnknown **);
    const auto create_d2d = reinterpret_cast<create_d2d_fn>(reinterpret_cast<void *>(GetProcAddress(direct2d, "D2D1CreateFactory")));
    const auto create_dwrite = reinterpret_cast<create_dwrite_fn>(reinterpret_cast<void *>(GetProcAddress(directwrite, "DWriteCreateFactory")));
    if (!create_d2d || !create_dwrite ||
        FAILED(create_d2d(D2D1_FACTORY_TYPE_MULTI_THREADED, __uuidof(ID2D1Factory), nullptr, reinterpret_cast<void **>(&own_factory))) || !own_factory ||
        FAILED(create_dwrite(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), reinterpret_cast<IUnknown **>(&own_dwrite))) || !own_dwrite) {
      log_line("Direct2D or DirectWrite unavailable");
      return true;
    }
    const auto properties = own_properties();
    ID2D1DCRenderTarget *sample = nullptr;
    if (FAILED(own_factory->CreateDCRenderTarget(&properties, &sample)) || !sample) {
      log_line("sample render target unavailable");
      return true;
    }
    const unsigned patched = patch_render_target_vtables(direct2d, sample);
    sample->Release();
    log_line("patched %u render target vtables", patched);
    return true;
  }

  bool active() {
    return seen;
  }

  void idle(const std::uint64_t tick) {
    // The application wiped its surface, ended its drawing and has not drawn all its text
    // again since: the rest is gone. An application that draws in several steps continues at
    // once, and a slow frame has not ended its drawing, so neither is cut short here.
    if (matching && match_ended && tick - match_ended_tick >= longest_batch_ms) {
      settle_match(tick);
      flush_batch();
    }
    // A clear may precede the same text in a slow frame. Only EndDraw can publish a clear
    // on its own. The idle timeout still bounds publication of glyphs drawn in long batches.
    if (batch_open && batch_draws != 0u && tick - last_draw_tick >= longest_batch_ms) {
      flush_batch();
    }
    counted_tick = counted_tick ? counted_tick : tick;
    if (tick - counted_tick >= 5000u) {
      if (draws_seen != 0u) {
        log_line("5 s: %u text draws, %u only compared; %u frames repeated, %u redone", draws_seen, draws_compared, frames_repeated, frames_redone);
      }
      draws_seen = draws_compared = frames_repeated = frames_redone = 0u;
      counted_tick = tick;
    }
  }
}  // namespace sunshine_text_hook::helper::d2d
