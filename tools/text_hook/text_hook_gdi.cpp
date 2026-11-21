// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file tools/text_hook/text_hook_gdi.cpp
 * @brief The GDI part of the text helper.
 *
 * A GDI application rarely draws its text straight onto its window: it draws on a bitmap and
 * copies that, often through further bitmaps, to the window. So this part follows the text
 * rather than one call. Each bitmap the application draws text on gets a layer, a picture of the
 * text on it alone. Every copy between bitmaps moves the same part of the layer with it, every
 * fill or opaque copy erases it, and what arrives in the window is the published picture.
 *
 * It sees the application's calls into GDI only. Text an application composes from glyph bitmaps
 * itself, or brings to the screen by any means other than GDI, never reaches the window layer,
 * and nothing is published for it.
 */
#include "text_hook_core.h"

#include <cstring>
#include <cwchar>
#include <psapi.h>

namespace sunshine_text_hook::helper::gdi {
  namespace {
    constexpr unsigned max_layers = 6u;
    constexpr unsigned max_modules = 512u;
    constexpr int max_extent = static_cast<int>(max_surface_extent);

    // --- rectangles ---------------------------------------------------------------------------

    constexpr RECT nothing {0, 0, 0, 0};

    bool empty(const RECT &r) {
      return r.right <= r.left || r.bottom <= r.top;
    }

    RECT intersect(const RECT &a, const RECT &b) {
      const RECT r {max_of(a.left, b.left), max_of(a.top, b.top), min_of(a.right, b.right), min_of(a.bottom, b.bottom)};
      return empty(r) ? nothing : r;
    }

    RECT unite(const RECT &a, const RECT &b) {
      if (empty(a)) {
        return b;
      }
      if (empty(b)) {
        return a;
      }
      return {min_of(a.left, b.left), min_of(a.top, b.top), max_of(a.right, b.right), max_of(a.bottom, b.bottom)};
    }

    bool contains(const RECT &outer, const RECT &inner) {
      return inner.left >= outer.left && inner.top >= outer.top && inner.right <= outer.right && inner.bottom <= outer.bottom;
    }

    // --- layers -------------------------------------------------------------------------------

    /** The text on one bitmap, or on the window: premultiplied BGRA, top row first. */
    struct layer_t {
      HBITMAP bitmap = nullptr;  ///< The bitmap this follows; null for the window's layer.
      const void *bits = nullptr;  ///< The bitmap's own pixels when it is a DIB section.
      std::size_t bits_size = 0u;
      int width = 0, height = 0;
      std::uint32_t *pixels = nullptr;
      RECT bounds = nothing;  ///< Around every text pixel; empty without text.
      bool loose = false;  ///< An erasure cut into `bounds`, which may now be larger than the text.
      std::uint64_t used = 0u;
    };

    layer_t layers[max_layers];
    layer_t window_layer;
    HWND root = nullptr;  ///< The window whose client area `window_layer` covers.
    LONG published_top = 0, published_bottom = 0;  ///< The rows last published with text.
    bool window_dirty = false;
    std::uint64_t dirty_since = 0u, last_change = 0u;
    std::uint64_t use_counter = 0u;
    bool retired = false;

    bool any_text() {
      if (!empty(window_layer.bounds)) {
        return true;
      }
      for (const auto &layer : layers) {
        if (layer.pixels && !empty(layer.bounds)) {
          return true;
        }
      }
      return false;
    }

    void free_layer(layer_t &layer) {
      if (layer.pixels) {
        VirtualFree(layer.pixels, 0u, MEM_RELEASE);
      }
      layer = {};
    }

    bool allocate(layer_t &layer, const int width, const int height) {
      free_layer(layer);
      if (width <= 0 || height <= 0 || width > max_extent || height > max_extent) {
        return false;
      }
      // Committed pages start as zero: no text.
      layer.pixels = static_cast<std::uint32_t *>(VirtualAlloc(nullptr, static_cast<std::size_t>(width) * height * 4u, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
      if (!layer.pixels) {
        return false;
      }
      layer.width = width;
      layer.height = height;
      return true;
    }

    layer_t *find_layer(const HBITMAP bitmap) {
      for (auto &layer : layers) {
        if (layer.pixels && layer.bitmap == bitmap) {
          layer.used = ++use_counter;
          return &layer;
        }
      }
      return nullptr;
    }

    /** The layer whose bitmap owns these pixels: the source of a copy made from memory. */
    layer_t *find_layer_of_bits(const void *bits) {
      for (auto &layer : layers) {
        if (layer.pixels && layer.bits && bits == layer.bits) {
          layer.used = ++use_counter;
          return &layer;
        }
      }
      return nullptr;
    }

    layer_t *make_layer(const HBITMAP bitmap, const int width, const int height) {
      layer_t *slot = &layers[0];
      for (auto &layer : layers) {
        if (!layer.pixels) {
          slot = &layer;
          break;
        }
        // Without a free slot, the bitmap whose text was touched longest ago gives way.
        if (layer.used < slot->used) {
          slot = &layer;
        }
      }
      if (!allocate(*slot, width, height)) {
        return nullptr;
      }
      slot->bitmap = bitmap;
      DIBSECTION section {};
      if (GetObjectW(bitmap, sizeof(section), &section) == sizeof(section) && section.dsBm.bmBits) {
        slot->bits = section.dsBm.bmBits;
        slot->bits_size = static_cast<std::size_t>(section.dsBm.bmWidthBytes) * static_cast<std::size_t>(height);
      }
      slot->used = ++use_counter;
      return slot;
    }

    /** Erase `rect` of a layer. Returns true when text was erased. */
    bool erase(layer_t &layer, const RECT &rect) {
      const RECT r = intersect(rect, layer.bounds);
      if (empty(r)) {
        return false;
      }
      for (LONG y = r.top; y < r.bottom; ++y) {
        std::memset(layer.pixels + static_cast<std::size_t>(y) * layer.width + r.left, 0, static_cast<std::size_t>(r.right - r.left) * 4u);
      }
      if (contains(r, layer.bounds)) {
        layer.bounds = nothing;
        layer.loose = false;
      } else {
        layer.loose = true;
      }
      return true;
    }

    /** Shrink the bounds of a layer to its text. */
    void tighten(layer_t &layer) {
      if (!layer.loose) {
        return;
      }
      layer.loose = false;
      RECT tight {layer.width, layer.height, 0, 0};
      for (LONG y = layer.bounds.top; y < layer.bounds.bottom; ++y) {
        const std::uint32_t *row = layer.pixels + static_cast<std::size_t>(y) * layer.width;
        LONG first = layer.bounds.left, last = layer.bounds.right;
        while (first < last && row[first] == 0u) {
          ++first;
        }
        while (last > first && row[last - 1] == 0u) {
          --last;
        }
        if (first < last) {
          tight = {min_of(tight.left, first), min_of(tight.top, y), max_of(tight.right, last), y + 1};
        }
      }
      layer.bounds = empty(tight) ? nothing : tight;
    }

    std::uint32_t over(const std::uint32_t source, const std::uint32_t under) {
      const std::uint32_t keep = 255u - (source >> 24u);
      if (keep == 0u || under == 0u) {
        return source;
      }
      // Each channel of `under` scaled by keep/255, two channels at a time.
      const std::uint32_t rb = (((under & 0x00ff00ffu) * keep + 0x00800080u) >> 8u) & 0x00ff00ffu;
      const std::uint32_t ag = ((((under >> 8u) & 0x00ff00ffu) * keep + 0x00800080u) >> 8u) & 0x00ff00ffu;
      const std::uint32_t scaled = rb | (ag << 8u);
      // Premultiplied channels of the two cannot sum past 255 except by rounding.
      const std::uint32_t sum_rb = (source & 0x00ff00ffu) + (scaled & 0x00ff00ffu);
      const std::uint32_t sum_ag = ((source >> 8u) & 0x00ff00ffu) + ((scaled >> 8u) & 0x00ff00ffu);
      const auto clamp = [](const std::uint32_t pair) {
        return (pair | (((pair >> 8u) & 0x00010001u) * 0xffu)) & 0x00ff00ffu;
      };
      return clamp(sum_rb) | (clamp(sum_ag) << 8u);
    }

    // --- where a DC draws -----------------------------------------------------------------------

    /** What a DC draws on, as far as the helper can follow it. */
    struct target_t {
      bool valid = false;
      bool window = false;
      HBITMAP bitmap = nullptr;
      int width = 0, height = 0;
      int dx = 0, dy = 0;  ///< From the DC's device coordinates to the layer's.
      layer_t *layer = nullptr;  ///< Null for a bitmap no text has reached yet.
    };

    void window_changed(std::uint64_t tick, bool complete);

    /** Follow the largest visible top-level window the application draws on. */
    bool adopt_root(const HWND candidate, const std::uint64_t tick) {
      RECT client {};
      if (!candidate || !GetClientRect(candidate, &client) || client.right <= 0 || client.bottom <= 0 || client.right > max_extent || client.bottom > max_extent) {
        return false;
      }
      if (candidate != root) {
        RECT current {};
        const bool usable = root && IsWindowVisible(root) && GetClientRect(root, &current);
        if (usable && static_cast<std::int64_t>(current.right) * current.bottom >= static_cast<std::int64_t>(client.right) * client.bottom) {
          return false;
        }
        root = candidate;
        free_layer(window_layer);
      }
      if (window_layer.pixels && window_layer.width == client.right && window_layer.height == client.bottom) {
        return true;
      }
      // A new window, or a new size of it: the application draws it again.
      const bool allocated = allocate(window_layer, client.right, client.bottom);
      window_changed(tick, true);
      return allocated;
    }

    target_t resolve(const HDC dc, const std::uint64_t tick) {
      target_t target;
      switch (GetObjectType(dc)) {
        case OBJ_MEMDC:
          {
            const auto bitmap = static_cast<HBITMAP>(GetCurrentObject(dc, OBJ_BITMAP));
            BITMAP info {};
            // A memory DC without a bitmap of its own holds a single placeholder pixel.
            if (!bitmap || GetObjectW(bitmap, sizeof(info), &info) == 0 || info.bmWidth <= 1 || info.bmHeight <= 1 || info.bmWidth > max_extent || info.bmHeight > max_extent) {
              return target;
            }
            target.bitmap = bitmap;
            target.width = info.bmWidth;
            target.height = info.bmHeight;
            target.layer = find_layer(bitmap);
            if (target.layer && (target.layer->width != target.width || target.layer->height != target.height)) {
              // The handle now names another bitmap.
              free_layer(*target.layer);
              target.layer = nullptr;
            }
            target.valid = true;
            return target;
          }
        case OBJ_DC:
          {
            const HWND window = WindowFromDC(dc);
            const HWND top = window ? GetAncestor(window, GA_ROOT) : nullptr;
            POINT origin {}, client {};
            if (!top || !adopt_root(top, tick) || !GetDCOrgEx(dc, &origin) || !ClientToScreen(root, &client)) {
              return target;
            }
            target.window = true;
            target.width = window_layer.width;
            target.height = window_layer.height;
            target.dx = origin.x - client.x;
            target.dy = origin.y - client.y;
            target.layer = &window_layer;
            target.valid = true;
            return target;
          }
        default:
          return target;
      }
    }

    layer_t *layer_of(target_t &target) {
      if (!target.layer && target.valid && !target.window) {
        target.layer = make_layer(target.bitmap, target.width, target.height);
      }
      return target.layer;
    }

    /** A rectangle of a DC's logical coordinates in the coordinates of its layer. */
    bool device_rect(const HDC dc, const target_t &target, int x, int y, int width, int height, RECT &rect, bool &mirrored) {
      POINT corners[2] {{x, y}, {x + width, y + height}};
      if (!LPtoDP(dc, corners, 2)) {
        return false;
      }
      mirrored = corners[1].x < corners[0].x || corners[1].y < corners[0].y;
      rect = {min_of(corners[0].x, corners[1].x) + target.dx, min_of(corners[0].y, corners[1].y) + target.dy,
              max_of(corners[0].x, corners[1].x) + target.dx, max_of(corners[0].y, corners[1].y) + target.dy};
      return !empty(rect);
    }

    /** The part of its layer a DC can draw on now. */
    RECT drawable(const HDC dc, const target_t &target) {
      RECT clip {};
      RECT whole {0, 0, target.width, target.height};
      if (GetClipBox(dc, &clip) == ERROR || !LPtoDP(dc, reinterpret_cast<POINT *>(&clip), 2)) {
        return whole;
      }
      OffsetRect(&clip, target.dx, target.dy);
      return intersect(whole, clip);
    }

    // --- the text itself ------------------------------------------------------------------------

    // A black bitmap the size of the largest layer. Each text call is repeated on it in white,
    // which gives the coverage of exactly that call; the pixels are black again afterwards.
    HDC scratch_dc = nullptr;
    HBITMAP scratch_bitmap = nullptr;
    std::uint32_t *scratch = nullptr;
    int scratch_width = 0, scratch_height = 0;

    bool ensure_scratch(const int width, const int height) {
      if (scratch && scratch_width >= width && scratch_height >= height) {
        return true;
      }
      const int new_width = max_of(width, scratch_width), new_height = max_of(height, scratch_height);
      if (!scratch_dc && !(scratch_dc = CreateCompatibleDC(nullptr))) {
        return false;
      }
      BITMAPINFO info {};
      info.bmiHeader.biSize = sizeof(info.bmiHeader);
      info.bmiHeader.biWidth = new_width;
      info.bmiHeader.biHeight = -new_height;
      info.bmiHeader.biPlanes = 1;
      info.bmiHeader.biBitCount = 32;
      info.bmiHeader.biCompression = BI_RGB;
      void *bits = nullptr;
      const HBITMAP bitmap = CreateDIBSection(scratch_dc, &info, DIB_RGB_COLORS, &bits, nullptr, 0u);
      if (!bitmap || !bits) {
        return false;
      }
      const HGDIOBJ previous = SelectObject(scratch_dc, bitmap);
      if (scratch_bitmap) {
        DeleteObject(scratch_bitmap);
      } else {
        (void) previous;
      }
      scratch_bitmap = bitmap;
      scratch = static_cast<std::uint32_t *>(bits);
      scratch_width = new_width;
      scratch_height = new_height;
      SetBkMode(scratch_dc, TRANSPARENT);
      SetTextColor(scratch_dc, RGB(255, 255, 255));
      return true;
    }

    /** Give the scratch DC the text state of the application's DC, shifted onto the layer. */
    void mirror_state(const HDC dc, const target_t &target, const POINT *position) {
      // Leaving the advanced mode requires an identity transform.
      ModifyWorldTransform(scratch_dc, nullptr, MWT_IDENTITY);
      const int graphics = GetGraphicsMode(dc);
      SetGraphicsMode(scratch_dc, graphics);
      XFORM transform {};
      if (graphics == GM_ADVANCED && GetWorldTransform(dc, &transform)) {
        SetWorldTransform(scratch_dc, &transform);
      }
      const int mapping = GetMapMode(dc);
      SetMapMode(scratch_dc, mapping);
      SIZE extent {};
      POINT origin {};
      if (mapping == MM_ISOTROPIC || mapping == MM_ANISOTROPIC) {
        GetWindowExtEx(dc, &extent);
        SetWindowExtEx(scratch_dc, extent.cx, extent.cy, nullptr);
        GetViewportExtEx(dc, &extent);
        SetViewportExtEx(scratch_dc, extent.cx, extent.cy, nullptr);
      }
      GetWindowOrgEx(dc, &origin);
      SetWindowOrgEx(scratch_dc, origin.x, origin.y, nullptr);
      GetViewportOrgEx(dc, &origin);
      SetViewportOrgEx(scratch_dc, origin.x + target.dx, origin.y + target.dy, nullptr);
      SelectObject(scratch_dc, GetCurrentObject(dc, OBJ_FONT));
      SetTextAlign(scratch_dc, GetTextAlign(dc));
      SetTextCharacterExtra(scratch_dc, GetTextCharacterExtra(dc));
      if (position) {
        MoveToEx(scratch_dc, position->x, position->y, nullptr);
      }
    }

    /**
     * Common path of the text hooks, after the application's own call succeeded. `replay` repeats
     * the call on the scratch DC. `position` is the DC's current position before the call, for
     * text aligned to it. `filled` is a rectangle the call filled first, in logical coordinates.
     */
    template<class Replay>
    void observe_text(const HDC dc, const POINT *position, const RECT *filled, Replay replay) {
      EnterCriticalSection(&state_lock);
      const std::uint64_t tick = GetTickCount64();
      target_t target = resolve(dc, tick);
      layer_t *layer = target.valid ? layer_of(target) : nullptr;
      if (!layer || !ensure_scratch(target.width, target.height)) {
        LeaveCriticalSection(&state_lock);
        return;
      }
      const RECT canvas = drawable(dc, target);
      bool changed = false;
      if (filled) {
        RECT fill {};
        bool mirrored = false;
        if (device_rect(dc, target, filled->left, filled->top, filled->right - filled->left, filled->bottom - filled->top, fill, mirrored)) {
          changed = erase(*layer, intersect(fill, canvas));
        }
      }
      mirror_state(dc, target, position);
      SetBoundsRect(scratch_dc, nullptr, DCB_ENABLE | DCB_RESET);
      replay(scratch_dc);
      RECT ink {};
      const UINT accumulated = GetBoundsRect(scratch_dc, &ink, DCB_RESET);
      SelectObject(scratch_dc, GetStockObject(SYSTEM_FONT));
      if ((accumulated & DCB_SET) != 0u && LPtoDP(scratch_dc, reinterpret_cast<POINT *>(&ink), 2)) {
        // What the call may have touched, with a pixel to spare for antialiasing.
        const RECT touched = intersect({min_of(ink.left, ink.right) - 1, min_of(ink.top, ink.bottom) - 1, max_of(ink.left, ink.right) + 1, max_of(ink.top, ink.bottom) + 1}, {0, 0, scratch_width, scratch_height});
        const RECT visible = intersect(touched, canvas);
        // An opaque background fills the text's cell before the glyphs are drawn.
        if (GetBkMode(dc) == OPAQUE) {
          changed |= erase(*layer, visible);
        }
        const COLORREF colour = GetTextColor(dc);
        const std::uint32_t red = GetRValue(colour), green = GetGValue(colour), blue = GetBValue(colour);
        RECT drawn {target.width, target.height, 0, 0};
        for (LONG y = touched.top; y < touched.bottom; ++y) {
          std::uint32_t *source = scratch + static_cast<std::size_t>(y) * scratch_width;
          const bool row_visible = y >= visible.top && y < visible.bottom;
          std::uint32_t *destination = row_visible ? layer->pixels + static_cast<std::size_t>(y) * layer->width : nullptr;
          for (LONG x = touched.left; x < touched.right; ++x) {
            const std::uint32_t white = source[x] & 0x00ffffffu;
            if (white == 0u) {
              continue;
            }
            source[x] = 0u;
            if (!destination || x < visible.left || x >= visible.right) {
              continue;
            }
            // Subpixel antialiasing colours the edges; their mean is the coverage.
            const std::uint32_t coverage = ((white & 0xffu) + ((white >> 8u) & 0xffu) + ((white >> 16u) & 0xffu) + 1u) / 3u;
            const std::uint32_t text = (coverage << 24u) | (((red * coverage + 127u) / 255u) << 16u) | (((green * coverage + 127u) / 255u) << 8u) | ((blue * coverage + 127u) / 255u);
            // The same text drawn again replaces itself; laid over itself its soft edges would
            // grow harder with every repaint.
            destination[x] = coverage >= (destination[x] >> 24u) ? text : over(text, destination[x]);
            drawn = {min_of(drawn.left, x), min_of(drawn.top, y), max_of(drawn.right, x + 1), max_of(drawn.bottom, y + 1)};
          }
        }
        if (!empty(drawn)) {
          layer->bounds = unite(layer->bounds, drawn);
          changed = true;
        }
      }
      if (changed && target.window) {
        window_changed(tick, false);
      }
      LeaveCriticalSection(&state_lock);
    }

    // --- copies and fills -----------------------------------------------------------------------

    enum class blend_e {
      replace,  ///< The destination becomes the source: text under it is gone.
      over,  ///< The source is laid over the destination, as a keyed or alpha copy does.
    };

    /** A fill of a DC, or a copy into it from something that carries no text: erase. */
    void observe_fill(const HDC dc, const int x, const int y, const int width, const int height) {
      if (retired || !any_text()) {
        return;
      }
      EnterCriticalSection(&state_lock);
      const std::uint64_t tick = GetTickCount64();
      const target_t target = resolve(dc, tick);
      RECT rect {};
      bool mirrored = false;
      if (target.layer && device_rect(dc, target, x, y, width, height, rect, mirrored) && erase(*target.layer, intersect(rect, drawable(dc, target))) && target.window) {
        window_changed(tick, false);
      }
      LeaveCriticalSection(&state_lock);
    }

    /**
     * A copy from one DC, or from a bitmap's own pixels, into another. `opacity` scales the text
     * that an alpha copy carries, out of 255.
     */
    void observe_copy(const HDC dc, const int x, const int y, const int width, const int height, const HDC source_dc, const layer_t *source_layer, const int source_x, const int source_y, const int source_width, const int source_height, const blend_e blend, const std::uint32_t opacity) {
      if (retired || !any_text()) {
        return;
      }
      EnterCriticalSection(&state_lock);
      const std::uint64_t tick = GetTickCount64();
      target_t target = resolve(dc, tick);
      RECT to {}, from {};
      bool mirrored = false, source_mirrored = false;
      if (!target.valid || !device_rect(dc, target, x, y, width, height, to, mirrored)) {
        LeaveCriticalSection(&state_lock);
        return;
      }
      if (source_dc) {
        target_t source = resolve(source_dc, tick);
        source_layer = source.layer;
        if (!source_layer || !device_rect(source_dc, source, source_x, source_y, source_width, source_height, from, source_mirrored)) {
          source_layer = nullptr;
        }
      } else if (source_layer) {
        from = {source_x, source_y, source_x + source_width, source_y + source_height};
      }
      // A mirrored copy is not followed: what it carried is not where it was.
      const bool carries = source_layer && !mirrored && !source_mirrored && !empty(intersect(from, source_layer->bounds)) && source_layer != target.layer;
      const RECT canvas = drawable(dc, target);
      bool changed = false;
      if (blend == blend_e::replace && target.layer) {
        changed = erase(*target.layer, intersect(to, canvas));
      }
      layer_t *layer = carries ? layer_of(target) : nullptr;
      if (layer) {
        const std::int64_t to_width = to.right - to.left, to_height = to.bottom - to.top;
        const std::int64_t from_width = from.right - from.left, from_height = from.bottom - from.top;
        // The destination pixels that the source's text lands on, and no others.
        const RECT text = intersect(from, source_layer->bounds);
        const RECT landing = intersect(
          {static_cast<LONG>(to.left + (text.left - from.left) * to_width / from_width) - 1, static_cast<LONG>(to.top + (text.top - from.top) * to_height / from_height) - 1,
           static_cast<LONG>(to.left + (text.right - from.left) * to_width / from_width) + 1, static_cast<LONG>(to.top + (text.bottom - from.top) * to_height / from_height) + 1},
          intersect(to, canvas)
        );
        RECT drawn {layer->width, layer->height, 0, 0};
        for (LONG row = landing.top; row < landing.bottom; ++row) {
          const std::int64_t source_row = from.top + (row - to.top) * from_height / to_height;
          if (source_row < 0 || source_row >= source_layer->height) {
            continue;
          }
          const std::uint32_t *in = source_layer->pixels + static_cast<std::size_t>(source_row) * source_layer->width;
          std::uint32_t *out = layer->pixels + static_cast<std::size_t>(row) * layer->width;
          for (LONG column = landing.left; column < landing.right; ++column) {
            const std::int64_t source_column = to_width == from_width ? from.left + (column - to.left) : from.left + (column - to.left) * from_width / to_width;
            if (source_column < 0 || source_column >= source_layer->width) {
              continue;
            }
            std::uint32_t pixel = in[source_column];
            if (pixel == 0u) {
              continue;
            }
            if (opacity < 255u) {
              pixel = ((((pixel & 0x00ff00ffu) * opacity) >> 8u) & 0x00ff00ffu) | (((((pixel >> 8u) & 0x00ff00ffu) * opacity) >> 8u & 0x00ff00ffu) << 8u);
              if (pixel == 0u) {
                continue;
              }
            }
            out[column] = blend == blend_e::replace ? pixel : over(pixel, out[column]);
            drawn = {min_of(drawn.left, column), min_of(drawn.top, row), max_of(drawn.right, column + 1), max_of(drawn.bottom, row + 1)};
          }
        }
        if (!empty(drawn)) {
          layer->bounds = unite(layer->bounds, drawn);
          changed = true;
        }
      }
      if (target.window && (changed || carries)) {
        // A copy that brings text to the window is the application presenting a finished picture.
        window_changed(tick, carries);
      }
      LeaveCriticalSection(&state_lock);
    }

    // --- publication ----------------------------------------------------------------------------

    /** Bring the published picture in line with the window's layer, if they differ. */
    void publish() {
      window_dirty = false;
      if (d2d::active() || !window_layer.pixels) {
        return;
      }
      const bool resized = !surface.pixels || surface.width != static_cast<std::uint32_t>(window_layer.width) || surface.height != static_cast<std::uint32_t>(window_layer.height);
      if (!ensure_surface(static_cast<std::uint32_t>(window_layer.width), static_cast<std::uint32_t>(window_layer.height))) {
        return;
      }
      if (resized) {
        published_top = published_bottom = 0;
      }
      tighten(window_layer);
      // Two rows more than the text: the host leaves the outermost row of a scaled picture out.
      const RECT &text = window_layer.bounds;
      const LONG top = empty(text) ? 0 : max_of<LONG>(text.top - 2, 0);
      const LONG bottom = empty(text) ? 0 : min_of<LONG>(text.bottom + 2, window_layer.height);
      // The rows that held text before are published too: that text may be gone from them.
      const bool before = published_bottom > published_top;
      const LONG first = before && bottom > top ? min_of(top, published_top) : before ? published_top : top;
      const LONG end = max_of(bottom, published_bottom);
      helper::publish(window_layer.pixels, static_cast<std::uint32_t>(first), static_cast<std::uint32_t>(end), source_gdi, static_cast<std::uint32_t>(top), static_cast<std::uint32_t>(bottom), root);
      published_top = top;
      published_bottom = bottom;
    }

    void window_changed(const std::uint64_t tick, const bool complete) {
      if (!window_dirty) {
        window_dirty = true;
        dirty_since = tick;
      }
      last_change = tick;
      // Text drawn straight onto the window arrives call by call and is published as one piece.
      if (complete || tick - dirty_since >= longest_batch_ms) {
        publish();
      }
    }

    // --- hooks ----------------------------------------------------------------------------------

    BOOL(WINAPI *original_text_out_a)(HDC, int, int, LPCSTR, int) = nullptr;
    BOOL(WINAPI *original_text_out_w)(HDC, int, int, LPCWSTR, int) = nullptr;
    BOOL(WINAPI *original_ext_text_out_a)(HDC, int, int, UINT, const RECT *, LPCSTR, UINT, const INT *) = nullptr;
    BOOL(WINAPI *original_ext_text_out_w)(HDC, int, int, UINT, const RECT *, LPCWSTR, UINT, const INT *) = nullptr;
    int(WINAPI *original_draw_text_a)(HDC, LPCSTR, int, LPRECT, UINT) = nullptr;
    int(WINAPI *original_draw_text_w)(HDC, LPCWSTR, int, LPRECT, UINT) = nullptr;
    int(WINAPI *original_draw_text_ex_a)(HDC, LPSTR, int, LPRECT, UINT, LPDRAWTEXTPARAMS) = nullptr;
    int(WINAPI *original_draw_text_ex_w)(HDC, LPWSTR, int, LPRECT, UINT, LPDRAWTEXTPARAMS) = nullptr;
    BOOL(WINAPI *original_bit_blt)(HDC, int, int, int, int, HDC, int, int, DWORD) = nullptr;
    BOOL(WINAPI *original_stretch_blt)(HDC, int, int, int, int, HDC, int, int, int, int, DWORD) = nullptr;
    BOOL(WINAPI *original_pat_blt)(HDC, int, int, int, int, DWORD) = nullptr;
    int(WINAPI *original_fill_rect)(HDC, const RECT *, HBRUSH) = nullptr;
    int(WINAPI *original_set_dibits_to_device)(HDC, int, int, DWORD, DWORD, int, int, UINT, UINT, const void *, const BITMAPINFO *, UINT) = nullptr;
    int(WINAPI *original_stretch_dibits)(HDC, int, int, int, int, int, int, int, int, const void *, const BITMAPINFO *, UINT, DWORD) = nullptr;
    BOOL(WINAPI *original_gdi_alpha_blend)(HDC, int, int, int, int, HDC, int, int, int, int, BLENDFUNCTION) = nullptr;
    BOOL(WINAPI *original_alpha_blend)(HDC, int, int, int, int, HDC, int, int, int, int, BLENDFUNCTION) = nullptr;
    BOOL(WINAPI *original_gdi_transparent_blt)(HDC, int, int, int, int, HDC, int, int, int, int, UINT) = nullptr;
    BOOL(WINAPI *original_transparent_blt)(HDC, int, int, int, int, HDC, int, int, int, int, UINT) = nullptr;
    BOOL(WINAPI *original_delete_object)(HGDIOBJ) = nullptr;
    HDC(WINAPI *original_begin_paint)(HWND, LPPAINTSTRUCT) = nullptr;

    /** The current position, where the text call is aligned to it. */
    bool position_of(const HDC dc, POINT &position) {
      return (GetTextAlign(dc) & TA_UPDATECP) != 0u && GetCurrentPositionEx(dc, &position);
    }

    template<class Call, class Replay>
    BOOL text_out(const HDC dc, Call call, Replay replay) {
      POINT position {};
      const bool positioned = !retired && position_of(dc, position);
      const BOOL result = call(dc);
      if (result && !retired) {
        observe_text(dc, positioned ? &position : nullptr, nullptr, replay);
      }
      return result;
    }

    BOOL WINAPI text_out_a_hook(HDC dc, int x, int y, LPCSTR string, int count) {
      const auto call = [&](const HDC to) {
        return original_text_out_a(to, x, y, string, count);
      };
      return text_out(dc, call, call);
    }

    BOOL WINAPI text_out_w_hook(HDC dc, int x, int y, LPCWSTR string, int count) {
      const auto call = [&](const HDC to) {
        return original_text_out_w(to, x, y, string, count);
      };
      return text_out(dc, call, call);
    }

    template<class Call>
    BOOL ext_text_out(const HDC dc, const UINT options, const RECT *rect, const UINT count, Call call) {
      POINT position {};
      const bool positioned = !retired && position_of(dc, position);
      const BOOL result = call(dc, options);
      if (result && !retired) {
        const RECT *filled = (options & ETO_OPAQUE) != 0u ? rect : nullptr;
        if (count != 0u) {
          observe_text(dc, positioned ? &position : nullptr, filled, [&](const HDC to) {
            call(to, options & ~static_cast<UINT>(ETO_OPAQUE));
          });
        } else if (filled) {
          // The common way to fill a rectangle with the background colour.
          observe_fill(dc, filled->left, filled->top, filled->right - filled->left, filled->bottom - filled->top);
        }
      }
      return result;
    }

    BOOL WINAPI ext_text_out_a_hook(HDC dc, int x, int y, UINT options, const RECT *rect, LPCSTR string, UINT count, const INT *spacing) {
      return ext_text_out(dc, options, rect, string ? count : 0u, [&](const HDC to, const UINT with) {
        return original_ext_text_out_a(to, x, y, with, rect, string, count, spacing);
      });
    }

    BOOL WINAPI ext_text_out_w_hook(HDC dc, int x, int y, UINT options, const RECT *rect, LPCWSTR string, UINT count, const INT *spacing) {
      return ext_text_out(dc, options, rect, string ? count : 0u, [&](const HDC to, const UINT with) {
        return original_ext_text_out_w(to, x, y, with, rect, string, count, spacing);
      });
    }

    template<class Call>
    int draw_text(const HDC dc, const RECT *rect, const UINT format, Call call) {
      // The call may move the rectangle it was given; the repeat needs the one it started with.
      RECT before = rect ? *rect : RECT {};
      POINT position {};
      const bool positioned = !retired && position_of(dc, position);
      const int result = call(dc, const_cast<RECT *>(rect), format);
      if (result != 0 && rect && !retired && (format & DT_CALCRECT) == 0u) {
        observe_text(dc, positioned ? &position : nullptr, nullptr, [&](const HDC to) {
          call(to, &before, format & ~static_cast<UINT>(DT_MODIFYSTRING));
        });
      }
      return result;
    }

    int WINAPI draw_text_a_hook(HDC dc, LPCSTR string, int count, LPRECT rect, UINT format) {
      return draw_text(dc, rect, format, [&](const HDC to, const LPRECT where, const UINT with) {
        return original_draw_text_a(to, string, count, where, with);
      });
    }

    int WINAPI draw_text_w_hook(HDC dc, LPCWSTR string, int count, LPRECT rect, UINT format) {
      return draw_text(dc, rect, format, [&](const HDC to, const LPRECT where, const UINT with) {
        return original_draw_text_w(to, string, count, where, with);
      });
    }

    int WINAPI draw_text_ex_a_hook(HDC dc, LPSTR string, int count, LPRECT rect, UINT format, LPDRAWTEXTPARAMS parameters) {
      return draw_text(dc, rect, format, [&](const HDC to, const LPRECT where, const UINT with) {
        return original_draw_text_ex_a(to, string, count, where, with, parameters);
      });
    }

    int WINAPI draw_text_ex_w_hook(HDC dc, LPWSTR string, int count, LPRECT rect, UINT format, LPDRAWTEXTPARAMS parameters) {
      return draw_text(dc, rect, format, [&](const HDC to, const LPRECT where, const UINT with) {
        return original_draw_text_ex_w(to, string, count, where, with, parameters);
      });
    }

    /** What a raster operation does to the text under it. */
    void observe_raster(const HDC dc, const int x, const int y, const int width, const int height, const HDC source, const int source_x, const int source_y, const int source_width, const int source_height, const DWORD operation) {
      switch (operation & 0x00ffffffu) {
        case SRCCOPY:
          if (source) {
            observe_copy(dc, x, y, width, height, source, nullptr, source_x, source_y, source_width, source_height, blend_e::replace, 255u);
            break;
          }
          [[fallthrough]];
        case PATCOPY:
        case BLACKNESS:
        case WHITENESS:
          observe_fill(dc, x, y, width, height);
          break;
        case SRCPAINT:
        case SRCAND:
          // The two halves of a masked copy: text the source carries is laid over.
          if (source) {
            observe_copy(dc, x, y, width, height, source, nullptr, source_x, source_y, source_width, source_height, blend_e::over, 255u);
          }
          break;
        default:
          break;
      }
    }

    BOOL WINAPI bit_blt_hook(HDC dc, int x, int y, int width, int height, HDC source, int source_x, int source_y, DWORD operation) {
      const BOOL result = original_bit_blt(dc, x, y, width, height, source, source_x, source_y, operation);
      if (result) {
        observe_raster(dc, x, y, width, height, source, source_x, source_y, width, height, operation);
      }
      return result;
    }

    BOOL WINAPI stretch_blt_hook(HDC dc, int x, int y, int width, int height, HDC source, int source_x, int source_y, int source_width, int source_height, DWORD operation) {
      const BOOL result = original_stretch_blt(dc, x, y, width, height, source, source_x, source_y, source_width, source_height, operation);
      if (result) {
        observe_raster(dc, x, y, width, height, source, source_x, source_y, source_width, source_height, operation);
      }
      return result;
    }

    BOOL WINAPI pat_blt_hook(HDC dc, int x, int y, int width, int height, DWORD operation) {
      const BOOL result = original_pat_blt(dc, x, y, width, height, operation);
      if (result) {
        observe_raster(dc, x, y, width, height, nullptr, 0, 0, 0, 0, operation);
      }
      return result;
    }

    int WINAPI fill_rect_hook(HDC dc, const RECT *rect, HBRUSH brush) {
      const int result = original_fill_rect(dc, rect, brush);
      if (result != 0 && rect) {
        observe_fill(dc, rect->left, rect->top, rect->right - rect->left, rect->bottom - rect->top);
      }
      return result;
    }

    /**
     * A copy from pixels in memory. It is followed when those pixels are a whole bitmap the
     * helper has a layer for, the way an application presents a back buffer it also draws text
     * on. Any other source carries no text the helper knows of, so the copy erases.
     */
    void observe_bits(const HDC dc, const int x, const int y, const int width, const int height, const void *bits, const BITMAPINFO *info, const int source_x, const int source_y, const int source_width, const int source_height) {
      if (retired || !any_text()) {
        return;
      }
      EnterCriticalSection(&state_lock);
      const layer_t *layer = bits && info ? find_layer_of_bits(bits) : nullptr;
      const bool whole = layer && source_x == 0 && source_y == 0 && source_width == layer->width && source_height == layer->height &&
                         info->bmiHeader.biWidth == layer->width && (info->bmiHeader.biHeight == layer->height || info->bmiHeader.biHeight == -layer->height);
      if (whole) {
        observe_copy(dc, x, y, width, height, nullptr, layer, 0, 0, layer->width, layer->height, blend_e::replace, 255u);
      } else {
        observe_fill(dc, x, y, width, height);
      }
      LeaveCriticalSection(&state_lock);
    }

    int WINAPI set_dibits_to_device_hook(HDC dc, int x, int y, DWORD width, DWORD height, int source_x, int source_y, UINT first_scan, UINT scans, const void *bits, const BITMAPINFO *info, UINT usage) {
      const int result = original_set_dibits_to_device(dc, x, y, width, height, source_x, source_y, first_scan, scans, bits, info, usage);
      if (result != 0) {
        // Only all scan lines at once are the whole bitmap.
        const bool all = info && first_scan == 0u && scans == static_cast<UINT>(info->bmiHeader.biHeight < 0 ? -info->bmiHeader.biHeight : info->bmiHeader.biHeight);
        observe_bits(dc, x, y, static_cast<int>(width), static_cast<int>(height), all ? bits : nullptr, info, source_x, source_y, static_cast<int>(width), static_cast<int>(height));
      }
      return result;
    }

    int WINAPI stretch_dibits_hook(HDC dc, int x, int y, int width, int height, int source_x, int source_y, int source_width, int source_height, const void *bits, const BITMAPINFO *info, UINT usage, DWORD operation) {
      const int result = original_stretch_dibits(dc, x, y, width, height, source_x, source_y, source_width, source_height, bits, info, usage, operation);
      if (result != 0 && result != static_cast<int>(GDI_ERROR)) {
        observe_bits(dc, x, y, width, height, (operation & 0x00ffffffu) == SRCCOPY ? bits : nullptr, info, source_x, source_y, source_width, source_height);
      }
      return result;
    }

    void observe_alpha(const HDC dc, const int x, const int y, const int width, const int height, const HDC source, const int source_x, const int source_y, const int source_width, const int source_height, const BLENDFUNCTION &function) {
      if (function.BlendOp == AC_SRC_OVER && function.SourceConstantAlpha != 0u) {
        observe_copy(dc, x, y, width, height, source, nullptr, source_x, source_y, source_width, source_height, blend_e::over, function.SourceConstantAlpha);
      }
    }

    BOOL WINAPI gdi_alpha_blend_hook(HDC dc, int x, int y, int width, int height, HDC source, int source_x, int source_y, int source_width, int source_height, BLENDFUNCTION function) {
      const BOOL result = original_gdi_alpha_blend(dc, x, y, width, height, source, source_x, source_y, source_width, source_height, function);
      if (result) {
        observe_alpha(dc, x, y, width, height, source, source_x, source_y, source_width, source_height, function);
      }
      return result;
    }

    BOOL WINAPI alpha_blend_hook(HDC dc, int x, int y, int width, int height, HDC source, int source_x, int source_y, int source_width, int source_height, BLENDFUNCTION function) {
      const BOOL result = original_alpha_blend(dc, x, y, width, height, source, source_x, source_y, source_width, source_height, function);
      if (result) {
        observe_alpha(dc, x, y, width, height, source, source_x, source_y, source_width, source_height, function);
      }
      return result;
    }

    BOOL WINAPI gdi_transparent_blt_hook(HDC dc, int x, int y, int width, int height, HDC source, int source_x, int source_y, int source_width, int source_height, UINT key) {
      const BOOL result = original_gdi_transparent_blt(dc, x, y, width, height, source, source_x, source_y, source_width, source_height, key);
      if (result) {
        observe_copy(dc, x, y, width, height, source, nullptr, source_x, source_y, source_width, source_height, blend_e::over, 255u);
      }
      return result;
    }

    BOOL WINAPI transparent_blt_hook(HDC dc, int x, int y, int width, int height, HDC source, int source_x, int source_y, int source_width, int source_height, UINT key) {
      const BOOL result = original_transparent_blt(dc, x, y, width, height, source, source_x, source_y, source_width, source_height, key);
      if (result) {
        observe_copy(dc, x, y, width, height, source, nullptr, source_x, source_y, source_width, source_height, blend_e::over, 255u);
      }
      return result;
    }

    HDC WINAPI begin_paint_hook(HWND window, LPPAINTSTRUCT paint) {
      const HDC dc = original_begin_paint(window, paint);
      // Windows has by now erased the part to paint with the window class's own brush, a fill
      // that passes no hook.
      if (dc && paint && !paint->fErase && GetClassLongPtrW(window, GCLP_HBRBACKGROUND) != 0) {
        observe_fill(dc, paint->rcPaint.left, paint->rcPaint.top, paint->rcPaint.right - paint->rcPaint.left, paint->rcPaint.bottom - paint->rcPaint.top);
      }
      return dc;
    }

    BOOL WINAPI delete_object_hook(HGDIOBJ object) {
      // Before the handle can name another bitmap.
      if (!retired && object) {
        EnterCriticalSection(&state_lock);
        for (auto &layer : layers) {
          if (layer.pixels && layer.bitmap == object) {
            free_layer(layer);
          }
        }
        LeaveCriticalSection(&state_lock);
      }
      return original_delete_object(object);
    }

    // --- installation ---------------------------------------------------------------------------

    struct hook_t {
      const wchar_t *module;
      const char *name;
      void *hook;
      void **original;
      void *targets[2];  ///< What an import of this function may point at before it is hooked.
    };

#define SUNSHINE_TEXT_HOOK(module, name, function) \
  {module, name, reinterpret_cast<void *>(&function##_hook), reinterpret_cast<void **>(&original_##function), {nullptr, nullptr}}
    hook_t hooks[] {
      SUNSHINE_TEXT_HOOK(L"gdi32.dll", "TextOutA", text_out_a),
      SUNSHINE_TEXT_HOOK(L"gdi32.dll", "TextOutW", text_out_w),
      SUNSHINE_TEXT_HOOK(L"gdi32.dll", "ExtTextOutA", ext_text_out_a),
      SUNSHINE_TEXT_HOOK(L"gdi32.dll", "ExtTextOutW", ext_text_out_w),
      SUNSHINE_TEXT_HOOK(L"user32.dll", "DrawTextA", draw_text_a),
      SUNSHINE_TEXT_HOOK(L"user32.dll", "DrawTextW", draw_text_w),
      SUNSHINE_TEXT_HOOK(L"user32.dll", "DrawTextExA", draw_text_ex_a),
      SUNSHINE_TEXT_HOOK(L"user32.dll", "DrawTextExW", draw_text_ex_w),
      SUNSHINE_TEXT_HOOK(L"gdi32.dll", "BitBlt", bit_blt),
      SUNSHINE_TEXT_HOOK(L"gdi32.dll", "StretchBlt", stretch_blt),
      SUNSHINE_TEXT_HOOK(L"gdi32.dll", "PatBlt", pat_blt),
      SUNSHINE_TEXT_HOOK(L"user32.dll", "FillRect", fill_rect),
      SUNSHINE_TEXT_HOOK(L"gdi32.dll", "SetDIBitsToDevice", set_dibits_to_device),
      SUNSHINE_TEXT_HOOK(L"gdi32.dll", "StretchDIBits", stretch_dibits),
      SUNSHINE_TEXT_HOOK(L"gdi32.dll", "GdiAlphaBlend", gdi_alpha_blend),
      SUNSHINE_TEXT_HOOK(L"msimg32.dll", "AlphaBlend", alpha_blend),
      SUNSHINE_TEXT_HOOK(L"gdi32.dll", "GdiTransparentBlt", gdi_transparent_blt),
      SUNSHINE_TEXT_HOOK(L"msimg32.dll", "TransparentBlt", transparent_blt),
      SUNSHINE_TEXT_HOOK(L"gdi32.dll", "DeleteObject", delete_object),
      SUNSHINE_TEXT_HOOK(L"user32.dll", "BeginPaint", begin_paint),
    };
#undef SUNSHINE_TEXT_HOOK

    HMODULE patched_modules[max_modules];
    unsigned patched_module_count = 0u;
    wchar_t windows_directory[MAX_PATH];
    unsigned windows_directory_length = 0u;

    /** Find the functions to hook. One the application has not loaded yet is found later. */
    void resolve_hooks() {
      // gdi32.dll forwards to gdi32full.dll; an import may be bound to either.
      const HMODULE full = GetModuleHandleW(L"gdi32full.dll");
      for (auto &hook : hooks) {
        if (hook.targets[0]) {
          continue;
        }
        const HMODULE module = GetModuleHandleW(hook.module);
        if (!module) {
          continue;
        }
        hook.targets[0] = reinterpret_cast<void *>(GetProcAddress(module, hook.name));
        hook.targets[1] = full ? reinterpret_cast<void *>(GetProcAddress(full, hook.name)) : nullptr;
        *hook.original = hook.targets[0];
      }
    }

    /**
     * The modules whose GDI calls are the application's own drawing: everything outside the
     * Windows directory, and the language runtimes inside it that draw on an application's
     * behalf. Windows' own libraries draw controls and themes, not the application's text.
     */
    bool applications_module(const HMODULE module) {
      wchar_t path[MAX_PATH];
      const DWORD length = GetModuleFileNameW(module, path, MAX_PATH);
      if (length == 0u || length >= MAX_PATH) {
        return false;
      }
      if (windows_directory_length == 0u || length <= windows_directory_length || _wcsnicmp(path, windows_directory, windows_directory_length) != 0 || path[windows_directory_length] != L'\\') {
        return true;
      }
      const wchar_t *name = std::wcsrchr(path, L'\\') + 1;
      return _wcsnicmp(name, L"mfc", 3u) == 0 || _wcsnicmp(name, L"msvbvm", 6u) == 0;
    }

    unsigned patch_imports(const HMODULE module) {
      const auto *base = reinterpret_cast<const std::uint8_t *>(module);
      const auto *dos = reinterpret_cast<const IMAGE_DOS_HEADER *>(base);
      if (dos->e_magic != IMAGE_DOS_SIGNATURE) {
        return 0u;
      }
      const auto *nt = reinterpret_cast<const IMAGE_NT_HEADERS *>(base + dos->e_lfanew);
      if (nt->Signature != IMAGE_NT_SIGNATURE || nt->OptionalHeader.NumberOfRvaAndSizes <= IMAGE_DIRECTORY_ENTRY_IMPORT) {
        return 0u;
      }
      const auto &directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
      if (directory.VirtualAddress == 0u || directory.Size < sizeof(IMAGE_IMPORT_DESCRIPTOR)) {
        return 0u;
      }
      unsigned patched = 0u;
      const auto *descriptor = reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR *>(base + directory.VirtualAddress);
      const auto *end = descriptor + directory.Size / sizeof(IMAGE_IMPORT_DESCRIPTOR);
      for (; descriptor < end && descriptor->Name != 0u && descriptor->FirstThunk != 0u; ++descriptor) {
        for (auto *thunk = reinterpret_cast<IMAGE_THUNK_DATA *>(const_cast<std::uint8_t *>(base) + descriptor->FirstThunk); thunk->u1.Function != 0u; ++thunk) {
          auto *slot = reinterpret_cast<void **>(&thunk->u1.Function);
          for (const auto &hook : hooks) {
            if (!hook.targets[0] || (*slot != hook.targets[0] && *slot != hook.targets[1])) {
              continue;
            }
            DWORD protection = 0u;
            if (VirtualProtect(slot, sizeof(void *), PAGE_READWRITE, &protection)) {
              *slot = hook.hook;
              VirtualProtect(slot, sizeof(void *), protection, &protection);
              ++patched;
            }
            break;
          }
        }
      }
      return patched;
    }
  }  // namespace

  void install() {
    if (retired) {
      return;
    }
    if (windows_directory_length == 0u) {
      windows_directory_length = GetWindowsDirectoryW(windows_directory, MAX_PATH);
      if (windows_directory_length >= MAX_PATH) {
        windows_directory_length = 0u;
      }
    }
    resolve_hooks();
    HMODULE own = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCWSTR>(&install), &own);
    static HMODULE modules[max_modules];
    DWORD bytes = 0u;
    if (!K32EnumProcessModules(GetCurrentProcess(), modules, sizeof(modules), &bytes)) {
      return;
    }
    const unsigned count = min_of<unsigned>(bytes / sizeof(HMODULE), max_modules);
    static HMODULE current[max_modules];
    unsigned current_count = 0u;
    for (unsigned index = 0u; index < count; ++index) {
      const HMODULE module = modules[index];
      bool known = false;
      for (unsigned seen = 0u; seen < patched_module_count && !known; ++seen) {
        known = patched_modules[seen] == module;
      }
      if (!known && module != own) {
        // Held for the duration: an application may unload a module at any time.
        HMODULE held = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, reinterpret_cast<LPCWSTR>(module), &held) || held != module) {
          if (held) {
            FreeLibrary(held);
          }
          continue;
        }
        if (applications_module(module)) {
          if (const unsigned patched = patch_imports(module)) {
            log_line("hooked %u GDI imports of module %p", patched, static_cast<void *>(module));
          }
        }
        FreeLibrary(held);
      }
      current[current_count++] = module;
    }
    // A module that has gone may give its address to another one later.
    std::memcpy(patched_modules, current, current_count * sizeof(HMODULE));
    patched_module_count = current_count;
  }

  void idle(const std::uint64_t tick) {
    if (window_dirty && tick - last_change >= longest_batch_ms / 2u) {
      publish();
    }
  }

  void retire() {
    retired = true;
    for (auto &layer : layers) {
      free_layer(layer);
    }
    free_layer(window_layer);
    window_dirty = false;
    published_top = published_bottom = 0;
  }
}  // namespace sunshine_text_hook::helper::gdi
