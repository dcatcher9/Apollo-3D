// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file tools/text_hook/text_hook_d3d11.cpp
 * @brief The Direct3D 11 part of the text helper.
 *
 * An application that composes its own text hands no text call to anyone. It takes glyph
 * bitmaps from GDI, sets them into an image in its own code, loads that image into a texture
 * and draws with it. This part follows the text from the texture on:
 *
 * - a texture whose image, loaded from memory, looks like text on nothing (text_hook_image.h)
 *   is a text texture, once there is a reason to think of text: the application took glyphs
 *   from GDI just before, or it loads such images again and again, into that texture or into
 *   new ones of its size, as the text layer of an application with a font renderer of its own
 *   is loaded line after line;
 * - every render target that text is drawn on has a mask of its own size. A draw that samples a
 *   text texture is issued once more onto the mask of its target; a draw that samples a target
 *   which has a mask is issued once more with that mask in the texture's place; any other draw
 *   onto a masked target is issued once more to wipe the mask where it covers it. The
 *   application's own shaders, geometry and state place every one of them;
 * - the mask of the back buffer, read back when the application presents, is the published
 *   picture.
 *
 * Nothing here knows an engine. What it cannot follow leaves the picture empty, and the host
 * then finds the text in the captured picture as it does for any game without a helper.
 *
 * A device context keeps its function table in memory of its own, and Direct3D rewrites the
 * entries of every call that submits work whenever it flushes: once with functions that settle
 * the flush, which the first such call then replaces with the ordinary ones. So every one of
 * those entries is hooked, including the ones this part has no other use for, and each hook
 * looks after its call whether the table still holds the hooks and puts them back.
 */
#include "text_hook_core.h"
#include "text_hook_image.h"

#include <cstring>
#include <cwchar>
#include <d3d11.h>
#include <dxgi.h>

namespace sunshine_text_hook::helper::d3d11 {
  namespace {
    // Vtable slots, as the interface headers number them.
    constexpr unsigned slot_draw_indexed = 12u;  // ID3D11DeviceContext
    constexpr unsigned slot_draw = 13u;
    constexpr unsigned slot_map = 14u;
    constexpr unsigned slot_unmap = 15u;
    constexpr unsigned slot_draw_indexed_instanced = 20u;
    constexpr unsigned slot_draw_instanced = 21u;
    constexpr unsigned slot_get_data = 29u;
    constexpr unsigned slot_draw_auto = 38u;
    constexpr unsigned slot_draw_indexed_instanced_indirect = 39u;
    constexpr unsigned slot_draw_instanced_indirect = 40u;
    constexpr unsigned slot_dispatch = 41u;
    constexpr unsigned slot_dispatch_indirect = 42u;
    constexpr unsigned slot_copy_subresource_region = 46u;
    constexpr unsigned slot_copy_resource = 47u;
    constexpr unsigned slot_update_subresource = 48u;
    constexpr unsigned slot_copy_structure_count = 49u;
    constexpr unsigned slot_clear_render_target_view = 50u;
    constexpr unsigned slot_clear_unordered_access_view_uint = 51u;
    constexpr unsigned slot_clear_unordered_access_view_float = 52u;
    constexpr unsigned slot_clear_depth_stencil_view = 53u;
    constexpr unsigned slot_generate_mips = 54u;
    constexpr unsigned slot_resolve_subresource = 57u;
    constexpr unsigned slot_execute_command_list = 58u;
    constexpr unsigned slot_flush = 111u;
    constexpr unsigned slot_create_texture_2d = 5u;  // ID3D11Device
    constexpr unsigned slot_present = 8u;  // IDXGISwapChain
    constexpr unsigned slot_present_1 = 22u;  // IDXGISwapChain1

    /// A loaded image is examined at about this many of its texels. Memory mapped for writing
    /// is slow to read, so it is examined at fewer.
    constexpr std::uint32_t examined_texels = 65536u, examined_mapped_texels = 16384u;
    /// The mask is read back when the text changed, though not in two frames running: a line
    /// that is being revealed changes in every frame, and reading costs the application time.
    /// Otherwise it is read once in this many frames: the application may move or fade the same
    /// text by means this part does not watch.
    constexpr std::uint32_t changed_read_frames = 2u, steady_read_frames = 6u;
    /// A texture whose image was plainly a picture is not looked at again for this many frames,
    /// and one whose image was not text for a few: an application that plays a film or redraws a
    /// panel loads one for every frame it shows.
    constexpr std::uint32_t picture_rest_frames = 120u, other_rest_frames = 6u;
    constexpr unsigned max_shapes = 32u, max_mapped = 16u;
    /// A texture loaded in this many frames with an image that looked like text is a text
    /// texture: a text layer or a store of glyphs is loaded line after line, or again for every
    /// frame, a sprite once. Every load before the last is a line shown without its text known.
    constexpr unsigned texture_confirmations = 2u;
    /// Its size and format are then ones that text is kept in: the next texture of that size is
    /// a text texture from its first such image, as the next store of glyphs is when one is
    /// full. New textures alone never make a size one of text, however many of them are loaded
    /// with images that look like it: the parts of a figure's face are as small as glyphs, and a
    /// game loads each into a new texture of one size whenever it shows the figure. Smaller
    /// images than `least_counted_texels` are not counted: a cursor or a spark looks like a glyph.
    constexpr std::uint32_t least_counted_texels = 4096u;
    /// A text texture is no longer one after this many whole images in a row that do not look
    /// like text, or at once after one that is plainly a picture: a line may carry a symbol or
    /// an image.
    constexpr std::uint32_t text_mark_strength = 3u;
    /// Text covers a small part of a picture: a screen full of it covers about a tenth. A mask
    /// that covers more than this has followed something that is not text. Nothing is published
    /// then, since a wrong picture would take the place of the host's own search for the text.
    constexpr float most_of_picture = 0.15f;

    // Private data the helper attaches to the application's textures. Direct3D releases the
    // attached objects with the texture they are attached to.
    const GUID text_mark_id {0x6a0c4f51u, 0x8f0du, 0x4c4eu, {0x9bu, 0x8du, 0x53u, 0x75u, 0x6eu, 0x33u, 0x44u, 0x01u}};
    const GUID mask_texture_id {0x6a0c4f51u, 0x8f0du, 0x4c4eu, {0x9bu, 0x8du, 0x53u, 0x75u, 0x6eu, 0x33u, 0x44u, 0x02u}};
    const GUID mask_target_id {0x6a0c4f51u, 0x8f0du, 0x4c4eu, {0x9bu, 0x8du, 0x53u, 0x75u, 0x6eu, 0x33u, 0x44u, 0x03u}};
    const GUID mask_view_id {0x6a0c4f51u, 0x8f0du, 0x4c4eu, {0x9bu, 0x8du, 0x53u, 0x75u, 0x6eu, 0x33u, 0x44u, 0x04u}};
    const GUID mask_active_id {0x6a0c4f51u, 0x8f0du, 0x4c4eu, {0x9bu, 0x8du, 0x53u, 0x75u, 0x6eu, 0x33u, 0x44u, 0x05u}};
    const GUID image_sum_id {0x6a0c4f51u, 0x8f0du, 0x4c4eu, {0x9bu, 0x8du, 0x53u, 0x75u, 0x6eu, 0x33u, 0x44u, 0x06u}};
    const GUID image_rest_id {0x6a0c4f51u, 0x8f0du, 0x4c4eu, {0x9bu, 0x8du, 0x53u, 0x75u, 0x6eu, 0x33u, 0x44u, 0x07u}};
    const GUID image_loads_id {0x6a0c4f51u, 0x8f0du, 0x4c4eu, {0x9bu, 0x8du, 0x53u, 0x75u, 0x6eu, 0x33u, 0x44u, 0x08u}};
    const GUID image_traced_id {0x6a0c4f51u, 0x8f0du, 0x4c4eu, {0x9bu, 0x8du, 0x53u, 0x75u, 0x6eu, 0x33u, 0x44u, 0x09u}};

    using draw_indexed_fn = void(STDMETHODCALLTYPE *)(ID3D11DeviceContext *, UINT, UINT, INT);
    using draw_fn = void(STDMETHODCALLTYPE *)(ID3D11DeviceContext *, UINT, UINT);
    using draw_indexed_instanced_fn = void(STDMETHODCALLTYPE *)(ID3D11DeviceContext *, UINT, UINT, UINT, INT, UINT);
    using draw_instanced_fn = void(STDMETHODCALLTYPE *)(ID3D11DeviceContext *, UINT, UINT, UINT, UINT);
    using map_fn = HRESULT(STDMETHODCALLTYPE *)(ID3D11DeviceContext *, ID3D11Resource *, UINT, D3D11_MAP, UINT, D3D11_MAPPED_SUBRESOURCE *);
    using unmap_fn = void(STDMETHODCALLTYPE *)(ID3D11DeviceContext *, ID3D11Resource *, UINT);
    using copy_region_fn = void(STDMETHODCALLTYPE *)(ID3D11DeviceContext *, ID3D11Resource *, UINT, UINT, UINT, UINT, ID3D11Resource *, UINT, const D3D11_BOX *);
    using copy_fn = void(STDMETHODCALLTYPE *)(ID3D11DeviceContext *, ID3D11Resource *, ID3D11Resource *);
    using update_fn = void(STDMETHODCALLTYPE *)(ID3D11DeviceContext *, ID3D11Resource *, UINT, const D3D11_BOX *, const void *, UINT, UINT);
    using clear_fn = void(STDMETHODCALLTYPE *)(ID3D11DeviceContext *, ID3D11RenderTargetView *, const FLOAT *);
    using resolve_fn = void(STDMETHODCALLTYPE *)(ID3D11DeviceContext *, ID3D11Resource *, UINT, ID3D11Resource *, UINT, DXGI_FORMAT);
    using plain_fn = void(STDMETHODCALLTYPE *)(ID3D11DeviceContext *);
    using indirect_fn = void(STDMETHODCALLTYPE *)(ID3D11DeviceContext *, ID3D11Buffer *, UINT);
    using dispatch_fn = void(STDMETHODCALLTYPE *)(ID3D11DeviceContext *, UINT, UINT, UINT);
    using copy_count_fn = void(STDMETHODCALLTYPE *)(ID3D11DeviceContext *, ID3D11Buffer *, UINT, ID3D11UnorderedAccessView *);
    using clear_uint_fn = void(STDMETHODCALLTYPE *)(ID3D11DeviceContext *, ID3D11UnorderedAccessView *, const UINT *);
    using clear_float_fn = void(STDMETHODCALLTYPE *)(ID3D11DeviceContext *, ID3D11UnorderedAccessView *, const FLOAT *);
    using clear_depth_fn = void(STDMETHODCALLTYPE *)(ID3D11DeviceContext *, ID3D11DepthStencilView *, UINT, FLOAT, UINT8);
    using generate_mips_fn = void(STDMETHODCALLTYPE *)(ID3D11DeviceContext *, ID3D11ShaderResourceView *);
    using get_data_fn = HRESULT(STDMETHODCALLTYPE *)(ID3D11DeviceContext *, ID3D11Asynchronous *, void *, UINT, UINT);
    using execute_fn = void(STDMETHODCALLTYPE *)(ID3D11DeviceContext *, ID3D11CommandList *, BOOL);
    using create_texture_fn = HRESULT(STDMETHODCALLTYPE *)(ID3D11Device *, const D3D11_TEXTURE2D_DESC *, const D3D11_SUBRESOURCE_DATA *, ID3D11Texture2D **);
    using present_fn = HRESULT(STDMETHODCALLTYPE *)(IDXGISwapChain *, UINT, UINT);
    using present_1_fn = HRESULT(STDMETHODCALLTYPE *)(IDXGISwapChain *, UINT, UINT, const void *);

    draw_indexed_fn original_draw_indexed = nullptr;
    draw_fn original_draw = nullptr;
    draw_indexed_instanced_fn original_draw_indexed_instanced = nullptr;
    draw_instanced_fn original_draw_instanced = nullptr;
    map_fn original_map = nullptr;
    unmap_fn original_unmap = nullptr;
    copy_region_fn original_copy_region = nullptr;
    copy_fn original_copy = nullptr;
    update_fn original_update = nullptr;
    clear_fn original_clear = nullptr;
    resolve_fn original_resolve = nullptr;
    plain_fn original_draw_auto = nullptr, original_flush = nullptr;
    indirect_fn original_draw_indexed_instanced_indirect = nullptr, original_draw_instanced_indirect = nullptr, original_dispatch_indirect = nullptr;
    dispatch_fn original_dispatch = nullptr;
    copy_count_fn original_copy_structure_count = nullptr;
    clear_uint_fn original_clear_unordered_access_view_uint = nullptr;
    clear_float_fn original_clear_unordered_access_view_float = nullptr;
    clear_depth_fn original_clear_depth_stencil_view = nullptr;
    generate_mips_fn original_generate_mips = nullptr;
    get_data_fn original_get_data = nullptr;
    execute_fn original_execute_command_list = nullptr;
    create_texture_fn original_create_texture = nullptr;
    present_fn original_present = nullptr;
    present_1_fn original_present_1 = nullptr;
    bool install_settled = false;  ///< Hooked, or found impossible: nothing more to try.

    // The device the application presents with, and its immediate context: the only ones
    // followed. Held, with what the helper made on that device, until another one presents.
    ID3D11Device *device = nullptr;
    ID3D11DeviceContext *context = nullptr;
    ID3D11BlendState *blend_add = nullptr;  ///< Straight colour laid over the mask by its alpha.
    ID3D11BlendState *blend_add_premultiplied = nullptr;
    ID3D11BlendState *blend_erase = nullptr;  ///< The mask kept only where the draw is clear.
    ID3D11BlendState *blend_erase_all = nullptr;
    ID3D11Texture2D *staging = nullptr;  ///< The back buffer's mask on its way to memory.
    std::uint32_t staging_width = 0u, staging_height = 0u;
    DXGI_FORMAT staging_format = DXGI_FORMAT_UNKNOWN;
    bool copy_pending = false;
    // The swap chain whose picture is followed, known by its address alone, when it last
    // presented and how large its picture is.
    const IDXGISwapChain *followed_chain = nullptr;
    std::uint64_t followed_tick = 0u, followed_texels = 0u;

    std::uint32_t frame = 1u;  ///< Counts the application's presents.
    volatile std::uint32_t glyph_frame = 0u;  ///< The frame the application last took glyphs in.
    bool marked_ever = false;  ///< A text texture was found: draws are worth looking at.
    /// The application loaded an image that looked like text right after taking glyphs from GDI:
    /// it composes its text from them. It takes a glyph once and keeps it, so a later line of
    /// characters it already has is loaded without any glyph being taken.
    bool composes_from_glyphs = false;
    bool text_changed = false;  ///< A text texture was loaded since the mask was last read.
    std::uint32_t last_copy_frame = 0u;
    unsigned frame_text_draws = 0u, last_frame_text_draws = 0u;

    /** A size and format of textures that a text texture had. */
    struct shape_t {
      std::uint32_t width, height;
      DXGI_FORMAT format;
    };

    shape_t shapes[max_shapes];
    unsigned shape_count = 0u, shape_turn = 0u;

    /** A texture the application has mapped for writing. */
    struct mapped_t {
      ID3D11Resource *resource;
      const void *data;
      std::uint32_t pitch;
    };

    mapped_t mapped[max_mapped];

    // The helper's own copy of the picture, the size of the published one.
    std::uint32_t *own_pixels = nullptr;
    std::uint32_t own_width = 0u, own_height = 0u;
    std::uint32_t published_top = 0u, published_bottom = 0u;

    // For the diagnostic log.
    unsigned counted_draws = 0u, counted_text_draws = 0u, counted_wipes = 0u, counted_reads = 0u, counted_frames = 0u;
    unsigned counted_all_draws = 0u, counted_images = 0u;
    volatile unsigned counted_glyphs = 0u;
    unsigned logged_images = 0u;
    const void *logged_texture = nullptr;
    int logged_verdict = -1;
    unsigned dumped_candidates = 0u;

    struct dumped_t {
      const void *texture;
      std::uint32_t frame;
    };

    dumped_t dumped[24];
    std::uint32_t traced_frame = 0u;  ///< Every draw of this frame goes into the log.
    unsigned dumped_textures = 0u, dumped_masks = 0u, dumped_repeated = 0u;
    std::uint64_t dumped_tick = 0u;

    /** For the log's owner: a picture beside the log, 32-bit BGRA, top row first. */
    void dump(const char *name, const unsigned index, const void *data, const std::uint32_t pitch, const std::uint32_t width, const std::uint32_t height) {
      wchar_t path[MAX_PATH];
      const DWORD length = GetTempPathW(MAX_PATH, path);
      if (length == 0u || length + 64u >= MAX_PATH || !data || width == 0u || height == 0u) {
        return;
      }
      std::swprintf(path + length, MAX_PATH - length, L"SunshineTextHook-%hs-%u.bmp", name, index);
      const HANDLE file = CreateFileW(path, GENERIC_WRITE, 0u, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
      if (file == INVALID_HANDLE_VALUE) {
        return;
      }
      BITMAPFILEHEADER header {};
      BITMAPINFOHEADER info {};
      info.biSize = sizeof(info);
      info.biWidth = static_cast<LONG>(width);
      info.biHeight = -static_cast<LONG>(height);
      info.biPlanes = 1;
      info.biBitCount = 32;
      header.bfType = 0x4d42;
      header.bfOffBits = sizeof(header) + sizeof(info);
      header.bfSize = header.bfOffBits + width * height * 4u;
      DWORD written = 0u;
      WriteFile(file, &header, sizeof(header), &written, nullptr);
      WriteFile(file, &info, sizeof(info), &written, nullptr);
      for (std::uint32_t y = 0u; y < height; ++y) {
        WriteFile(file, static_cast<const std::uint8_t *>(data) + static_cast<std::size_t>(y) * pitch, width * 4u, &written, nullptr);
      }
      CloseHandle(file);
    }

    std::uint64_t counted_tick = 0u;

    template<class T>
    void release(T *&object) {
      if (object) {
        object->Release();
        object = nullptr;
      }
    }

    bool flagged(ID3D11DeviceChild *object, const GUID &id) {
      std::uint32_t value = 0u;
      UINT size = sizeof(value);
      return SUCCEEDED(object->GetPrivateData(id, &size, &value)) && size == sizeof(value) && value != 0u;
    }

    void flag(ID3D11DeviceChild *object, const GUID &id, const bool on) {
      const std::uint32_t value = on ? 1u : 0u;
      object->SetPrivateData(id, sizeof(value), &value);
    }

    /** An object the helper attached under this id, with a reference of the caller's. */
    template<class T>
    T *attached(ID3D11DeviceChild *object, const GUID &id) {
      IUnknown *unknown = nullptr;
      UINT size = sizeof(unknown);
      if (FAILED(object->GetPrivateData(id, &size, &unknown))) {
        return nullptr;
      }
      // Only the helper sets these ids, and always to an object of this very interface.
      return static_cast<T *>(unknown);
    }

    ID3D11Texture2D *texture_of(ID3D11Resource *resource) {
      ID3D11Texture2D *texture = nullptr;
      if (resource) {
        resource->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void **>(&texture));
      }
      return texture;
    }

    ID3D11Texture2D *texture_of(ID3D11View *view) {
      ID3D11Resource *resource = nullptr;
      view->GetResource(&resource);
      ID3D11Texture2D *texture = texture_of(resource);
      release(resource);
      return texture;
    }

    /** The mask of one of the application's render targets. Holds references. */
    struct mask_t {
      ID3D11Texture2D *texture;
      ID3D11RenderTargetView *target;
      ID3D11ShaderResourceView *view;
      bool active;  ///< Something was drawn on it since its target was last cleared.

      void drop() {
        release(texture);
        release(target);
        release(view);
        active = false;
      }
    };

    bool find_mask(ID3D11Texture2D *of, mask_t &mask) {
      mask = {};
      mask.texture = attached<ID3D11Texture2D>(of, mask_texture_id);
      if (!mask.texture) {
        return false;
      }
      mask.target = attached<ID3D11RenderTargetView>(of, mask_target_id);
      mask.view = attached<ID3D11ShaderResourceView>(of, mask_view_id);
      mask.active = flagged(of, mask_active_id);
      if (!mask.target || !mask.view) {
        mask.drop();
        return false;
      }
      return true;
    }

    /**
     * Give a render target of the application a mask of its size. The three objects are attached
     * to the application's texture, none to another of the three, so they go when it goes.
     */
    bool make_mask(ID3D11Texture2D *of, const bool srgb, mask_t &mask) {
      mask = {};
      D3D11_TEXTURE2D_DESC theirs {};
      of->GetDesc(&theirs);
      if (theirs.ArraySize != 1u || theirs.Width == 0u || theirs.Height == 0u || theirs.Width > max_surface_extent || theirs.Height > max_surface_extent) {
        return false;
      }
      D3D11_TEXTURE2D_DESC own {};
      own.Width = theirs.Width;
      own.Height = theirs.Height;
      own.MipLevels = 1u;
      own.ArraySize = 1u;
      // The mask holds what the application's shaders write, encoded as its own target has it.
      own.Format = srgb ? DXGI_FORMAT_B8G8R8A8_UNORM_SRGB : DXGI_FORMAT_B8G8R8A8_UNORM;
      own.SampleDesc.Count = 1u;
      own.Usage = D3D11_USAGE_DEFAULT;
      own.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
      enter_call();
      const bool made = SUCCEEDED(device->CreateTexture2D(&own, nullptr, &mask.texture)) && mask.texture &&
                        SUCCEEDED(device->CreateRenderTargetView(mask.texture, nullptr, &mask.target)) && mask.target &&
                        SUCCEEDED(device->CreateShaderResourceView(mask.texture, nullptr, &mask.view)) && mask.view;
      leave_call();
      if (!made || FAILED(of->SetPrivateDataInterface(mask_texture_id, mask.texture)) || FAILED(of->SetPrivateDataInterface(mask_target_id, mask.target)) || FAILED(of->SetPrivateDataInterface(mask_view_id, mask.view))) {
        of->SetPrivateDataInterface(mask_texture_id, nullptr);
        of->SetPrivateDataInterface(mask_target_id, nullptr);
        of->SetPrivateDataInterface(mask_view_id, nullptr);
        mask.drop();
        return false;
      }
      flag(of, mask_active_id, false);
      log_line("Direct3D 11: mask %ux%u for a render target", own.Width, own.Height);
      return true;
    }

    /** Start a mask afresh if nothing is on it, and note that something now is. */
    void activate(ID3D11Texture2D *of, mask_t &mask) {
      if (!mask.active) {
        const FLOAT nothing[4] {0.0f, 0.0f, 0.0f, 0.0f};
        original_clear(context, mask.target, nothing);
        flag(of, mask_active_id, true);
        mask.active = true;
      }
    }

    // --- which textures hold text ---------------------------------------------------------------

    /** Bytes per texel and the offset of coverage within one, for the formats text is kept in. */
    bool layout_of(const DXGI_FORMAT format, unsigned &bytes, unsigned &coverage) {
      switch (format) {
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        case DXGI_FORMAT_B8G8R8A8_TYPELESS:
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        case DXGI_FORMAT_R8G8B8A8_TYPELESS:
          bytes = 4u;
          coverage = 3u;
          return true;
        case DXGI_FORMAT_A8_UNORM:
        case DXGI_FORMAT_R8_UNORM:
          bytes = 1u;
          coverage = 0u;
          return true;
        default:
          return false;
      }
    }

    image::measure_t examine(const void *data, const std::uint32_t pitch, const std::uint32_t width, const std::uint32_t height, const DXGI_FORMAT format, const bool slow_memory) {
      unsigned bytes = 0u, coverage_at = 0u;
      if (!layout_of(format, bytes, coverage_at)) {
        return {};
      }
      return image::examine(data, pitch, width, height, bytes, coverage_at, slow_memory ? examined_mapped_texels : examined_texels);
    }

    /** A text texture had this size and format. */
    bool confirmed(const D3D11_TEXTURE2D_DESC &desc) {
      for (unsigned index = 0u; index < shape_count; ++index) {
        if (shapes[index].width == desc.Width && shapes[index].height == desc.Height && shapes[index].format == desc.Format) {
          return true;
        }
      }
      return false;
    }

    /** A text texture has this size and format. When the table is full the oldest entry goes. */
    void confirm(const D3D11_TEXTURE2D_DESC &desc) {
      if (!confirmed(desc)) {
        shape_t &entry = shape_count < max_shapes ? shapes[shape_count++] : shapes[shape_turn++ % max_shapes];
        entry = {desc.Width, desc.Height, desc.Format};
      }
    }

    /** How firmly a texture is taken for a text texture: 0 for one that is not. */
    std::uint32_t mark_of(ID3D11Texture2D *texture) {
      std::uint32_t value = 0u;
      UINT size = sizeof(value);
      return SUCCEEDED(texture->GetPrivateData(text_mark_id, &size, &value)) && size == sizeof(value) ? value : 0u;
    }

    void set_mark(ID3D11Texture2D *texture, const std::uint32_t value) {
      texture->SetPrivateData(text_mark_id, sizeof(value), &value);
    }

    /** This texture holds text no longer, and what it was loaded with before says nothing now. */
    void unmark(ID3D11Texture2D *texture) {
      set_mark(texture, 0u);
      texture->SetPrivateData(image_loads_id, 0u, nullptr);
    }

    /** This texture holds text. `sum` tells whether it is other text than it held before. */
    void mark_text(ID3D11Texture2D *texture, const D3D11_TEXTURE2D_DESC &desc, const std::uint64_t sum) {
      if (mark_of(texture) == 0u) {
        log_line("Direct3D 11: text texture %p %ux%u format %u", static_cast<void *>(texture), desc.Width, desc.Height, static_cast<unsigned>(desc.Format));
      }
      set_mark(texture, text_mark_strength);
      std::uint64_t before = 0u;
      UINT size = sizeof(before);
      if (sum == 0u || FAILED(texture->GetPrivateData(image_sum_id, &size, &before)) || size != sizeof(before) || before != sum) {
        // An application may load the same image again for every frame it shows.
        texture->SetPrivateData(image_sum_id, sizeof(sum), &sum);
        text_changed = true;
      }
      marked_ever = true;
    }

    /** In how many frames a texture was loaded with an image that looked like text. */
    struct loads_t {
      std::uint32_t count;  ///< Counted up to `texture_confirmations`.
      std::uint32_t frame;  ///< The frame it was last counted in.
    };

    loads_t loads_of(ID3D11Texture2D *texture) {
      loads_t loads {};
      UINT size = sizeof(loads);
      return SUCCEEDED(texture->GetPrivateData(image_loads_id, &size, &loads)) && size == sizeof(loads) ? loads : loads_t {};
    }

    /** The application took glyphs from GDI in this frame or the one before. */
    bool glyphs_recent() {
      const std::uint32_t taken = glyph_frame;
      return taken != 0u && frame - taken <= 1u;
    }

    /**
     * An image was loaded into a texture from memory. It is text when it looks like text and
     * there is a reason to think of text at all: glyphs were just taken from GDI, or were for an
     * earlier such image, the texture holds text already, such images have been loaded into it
     * often enough, or a texture of its size and format became a text texture that way. The
     * last two are how an application is followed that renders its glyphs itself: its text
     * layer or its store of glyphs is loaded line after line, a picture or a sprite is not.
     */
    void note_image(ID3D11Texture2D *texture, const void *data, const std::uint32_t pitch, const std::uint32_t width, const std::uint32_t height, const bool whole, const bool slow_memory, const char *how) {
      // Text the application draws with Direct2D is that part's to follow.
      if (d2d::active()) {
        return;
      }
      D3D11_TEXTURE2D_DESC desc {};
      texture->GetDesc(&desc);
      unsigned texel_bytes = 0u, coverage_at = 0u;
      if (!layout_of(desc.Format, texel_bytes, coverage_at)) {
        return;  // not a format text is kept in, nor one the rest of this can read
      }
      const std::uint32_t mark = mark_of(texture);
      const bool was_text = mark != 0u;
      const bool recent = glyphs_recent();
      const bool counted = whole && static_cast<std::uint64_t>(desc.Width) * desc.Height >= least_counted_texels;
      ++counted_images;
      // The log tells how an application loads its images, also where none is taken for text.
      const bool told = logging() && logged_images < 400u;

      // For the log's owner: how often this texture has been loaded, and how often a picture of
      // it was kept. A texture that is loaded again and again is where text is looked for.
      struct traced_t {
        std::uint32_t loads, pictures;
      } traced {};

      if (logging() && whole) {
        UINT size = sizeof(traced);
        if (FAILED(texture->GetPrivateData(image_traced_id, &size, &traced)) || size != sizeof(traced)) {
          traced = {};
        }
        ++traced.loads;
        texture->SetPrivateData(image_traced_id, sizeof(traced), &traced);
      }
      if (!was_text && !recent && !counted && !told) {
        return;
      }
      if (!was_text && !recent) {
        // Resting: the frame it rests until, kept with the texture.
        std::uint32_t until = 0u;
        UINT size = sizeof(until);
        if (SUCCEEDED(texture->GetPrivateData(image_rest_id, &size, &until)) && size == sizeof(until) && until - frame <= picture_rest_frames) {
          return;
        }
      }
      const image::measure_t measure = examine(data, pitch, width, height, desc.Format, slow_memory);
      // A texture that holds text may fill up: an application that keeps its glyphs in one
      // draws every line from it and sets down more of them as the text goes on.
      const bool text = measure.verdict == image::verdict_e::text || (was_text && measure.glyphs);
      if (!was_text && !text && measure.verdict != image::verdict_e::empty) {
        const std::uint32_t until = frame + (measure.dense ? picture_rest_frames : other_rest_frames);
        texture->SetPrivateData(image_rest_id, sizeof(until), &until);
        if (!measure.faint) {
          // Whatever looked like text in it before, the application gave it to something else.
          texture->SetPrivateData(image_loads_id, 0u, nullptr);
        }
      }
      bool loaded_often = false;
      if (text && recent) {
        confirm(desc);
        composes_from_glyphs = composes_from_glyphs || counted;
      } else if (text && counted && !was_text) {
        loads_t loads = loads_of(texture);
        if (loads.count < texture_confirmations && loads.frame != frame) {
          loads = {loads.count + 1u, frame};
          texture->SetPrivateData(image_loads_id, sizeof(loads), &loads);
        }
        loaded_often = loads.count >= texture_confirmations;
        if (loaded_often) {
          confirm(desc);
        }
      }
      const bool reason = was_text || recent || (counted && (loaded_often || composes_from_glyphs || confirmed(desc)));
      const char *const verdict = text ? "looks like text" : measure.verdict == image::verdict_e::empty ? "empty" :
                                                                                                          "other";
      // One line for a texture that is loaded with the same kind of image again and again.
      if (told && (texture != logged_texture || static_cast<int>(measure.verdict) != logged_verdict)) {
        ++logged_images;
        logged_texture = texture;
        logged_verdict = static_cast<int>(measure.verdict);
        log_line("Direct3D 11: frame %u %s %ux%u%s into texture %p %ux%u format %u usage %u bind %x: %s%s%s%s (covered %u%%, solid %u%% of that, narrow %u%% of the solid)", frame, how, width, height, whole ? "" : " (part)", static_cast<void *>(texture), desc.Width, desc.Height, static_cast<unsigned>(desc.Format), static_cast<unsigned>(desc.Usage), static_cast<unsigned>(desc.BindFlags), verdict, recent ? ", glyphs just taken" : "", was_text ? ", was text" : "", reason && text ? ", taken for text" : "", measure.covered_percent, measure.solid_percent, measure.narrow_percent);
      }
      // For the log's owner: the mostly clear images that were not taken for text, a few of each
      // texture, some seconds apart.
      if (logging() && counted && measure.verdict != image::verdict_e::empty && !(reason && text) && !measure.dense && dumped_candidates < 24u && desc.Format != DXGI_FORMAT_A8_UNORM && desc.Format != DXGI_FORMAT_R8_UNORM) {
        bool known = false;
        for (unsigned index = 0u; index < dumped_candidates && !known; ++index) {
          known = dumped[index].texture == texture && frame - dumped[index].frame < 270u;
        }
        if (!known) {
          dumped[dumped_candidates] = {texture, frame};
          log_line("Direct3D 11: frame %u texture %p %s is picture candidate-%u (covered %u%%, solid %u%% of that, narrow %u%% of the solid)", frame, static_cast<void *>(texture), verdict, dumped_candidates, measure.covered_percent, measure.solid_percent, measure.narrow_percent);
          dump("candidate", dumped_candidates++, data, pitch, width, height);
        }
      }
      // And the image of a texture that is loaded again and again without being taken for text:
      // after its thirtieth load, and once more after its three hundredth. Not a plain picture,
      // as a film's frames are.
      if (traced.loads >= (traced.pictures == 0u ? 30u : 300u) && traced.pictures < 2u && !(reason && text) && measure.verdict != image::verdict_e::empty && !measure.dense && dumped_repeated < 12u && texel_bytes == 4u) {
        ++traced.pictures;
        texture->SetPrivateData(image_traced_id, sizeof(traced), &traced);
        log_line("Direct3D 11: frame %u texture %p, loaded %u times, %s, is picture repeated-%u (covered %u%%, solid %u%% of that, narrow %u%% of the solid%s)", frame, static_cast<void *>(texture), traced.loads, verdict, dumped_repeated, measure.covered_percent, measure.solid_percent, measure.narrow_percent, measure.faint ? ", faint" : "");
        dump("repeated", dumped_repeated++, data, pitch, width, height);
      }
      if (!reason) {
        return;
      }
      if (text) {
        if (!was_text && logging() && dumped_textures < 16u && whole && desc.Format != DXGI_FORMAT_A8_UNORM && desc.Format != DXGI_FORMAT_R8_UNORM) {
          log_line("Direct3D 11: texture %p is picture texture-%u", static_cast<void *>(texture), dumped_textures);
          dump("texture", dumped_textures++, data, pitch, width, height);
        }
        mark_text(texture, desc, measure.sum);
      } else if (!was_text) {
        return;
      } else if (measure.verdict == image::verdict_e::empty) {
        mark_text(texture, desc, measure.sum);  // the text is gone from it; it stays a text texture
      } else if (whole && !measure.faint) {
        // A picture ends it at once: the application gave the texture to something else. Any
        // other image wears the mark down, unless too little of it is covered to tell.
        if (measure.dense || mark == 1u) {
          unmark(texture);
        } else {
          set_mark(texture, mark - 1u);
        }
        text_changed = true;
      } else {
        text_changed = true;
      }
    }

    // --- following the draws --------------------------------------------------------------------

    /// Put the hooks back if Direct3D rewrote the followed context's table during this call.
    void keep_hooked(ID3D11DeviceContext *self);

    enum class effect_e {
      none,
      add,
      add_premultiplied,
      copy,  ///< Written as it comes, as a draw without blending writes.
      erase,
      erase_all,
    };

    /** What the application's blend state makes of a draw, for each kind of thing it samples. */
    void effects_of(ID3D11DeviceContext *self, effect_e &for_text, effect_e &for_masked, effect_e &for_other) {
      for_text = for_masked = for_other = effect_e::none;
      ID3D11BlendState *state = nullptr;
      FLOAT factor[4] {};
      UINT samples = 0u;
      self->OMGetBlendState(&state, factor, &samples);
      D3D11_BLEND_DESC desc {};
      desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;  // no state: no blending
      if (state) {
        state->GetDesc(&desc);
        state->Release();
      }
      const auto &target = desc.RenderTarget[0];
      // A draw that writes no colour covers nothing: a depth pass, or one that only sets alpha.
      if ((target.RenderTargetWriteMask & (D3D11_COLOR_WRITE_ENABLE_RED | D3D11_COLOR_WRITE_ENABLE_GREEN | D3D11_COLOR_WRITE_ENABLE_BLUE)) == 0u) {
        return;
      }
      // Nor does one that only darkens or tints what is there, as a shade over the picture does.
      if (target.BlendEnable && (target.SrcBlend == D3D11_BLEND_DEST_COLOR || target.SrcBlend == D3D11_BLEND_INV_DEST_COLOR || target.SrcBlend == D3D11_BLEND_DEST_ALPHA || target.SrcBlend == D3D11_BLEND_INV_DEST_ALPHA || target.SrcBlend == D3D11_BLEND_ZERO)) {
        return;
      }
      if (!target.BlendEnable || target.DestBlend == D3D11_BLEND_ZERO) {
        // It replaces what is there. Text is never drawn so; whatever this is covers the text.
        for_text = for_other = effect_e::erase_all;
        for_masked = effect_e::copy;
        return;
      }
      for_text = target.SrcBlend == D3D11_BLEND_ONE ? effect_e::add_premultiplied : effect_e::add;
      for_masked = effect_e::add_premultiplied;  // a mask holds colour times coverage
      // A draw that only adds light covers nothing.
      for_other = target.DestBlend == D3D11_BLEND_INV_SRC_ALPHA ? effect_e::erase : effect_e::none;
    }

    /**
     * Issue the application's draw once more onto a mask: its own shaders, geometry, viewport and
     * scissor, with only the target and the blending the helper's, and for a masked source the
     * mask in the texture's place. Everything is put back as it was.
     */
    template<class Draw>
    void repeat(ID3D11DeviceContext *self, const mask_t &onto, const effect_e effect, const int slot, ID3D11ShaderResourceView *instead, Draw draw) {
      ID3D11RenderTargetView *targets[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] {};
      ID3D11DepthStencilView *depth = nullptr;
      self->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, targets, &depth);
      ID3D11BlendState *blend = nullptr;
      FLOAT factor[4] {};
      UINT samples = 0u;
      self->OMGetBlendState(&blend, factor, &samples);
      ID3D11ShaderResourceView *sampled = nullptr;
      if (instead) {
        self->PSGetShaderResources(static_cast<UINT>(slot), 1u, &sampled);
        self->PSSetShaderResources(static_cast<UINT>(slot), 1u, &instead);
      }
      ID3D11BlendState *const own = effect == effect_e::add ? blend_add : effect == effect_e::add_premultiplied ? blend_add_premultiplied :
                                                                        effect == effect_e::erase               ? blend_erase :
                                                                        effect == effect_e::erase_all           ? blend_erase_all :
                                                                                                                  nullptr;
      const FLOAT no_factor[4] {};
      self->OMSetRenderTargets(1u, &onto.target, nullptr);
      self->OMSetBlendState(own, no_factor, 0xffffffffu);
      draw();
      UINT count = 0u;
      for (UINT index = 0u; index < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++index) {
        count = targets[index] ? index + 1u : count;
      }
      self->OMSetRenderTargets(count, targets, depth);
      self->OMSetBlendState(blend, factor, samples);
      if (instead) {
        self->PSSetShaderResources(static_cast<UINT>(slot), 1u, &sampled);
      }
      for (auto &target : targets) {
        release(target);
      }
      release(depth);
      release(blend);
      release(sampled);
    }

    bool followed(ID3D11DeviceContext *self) {
      counted_all_draws += self == context;
      return self == context && marked_ever && !nested() && !d2d::active();
    }

    /** The application drew; `draw` issues the same draw again. */
    template<class Draw>
    void follow(ID3D11DeviceContext *self, Draw draw) {
      EnterCriticalSection(&state_lock);
      ++counted_draws;
      ID3D11RenderTargetView *target = nullptr;
      self->OMGetRenderTargets(1u, &target, nullptr);
      ID3D11Texture2D *onto = target ? texture_of(target) : nullptr;
      D3D11_RENDER_TARGET_VIEW_DESC view {};
      if (onto) {
        target->GetDesc(&view);
      }
      // A mask is the size of its target's first level; a draw onto a smaller level or onto one
      // layer of several is not followed.
      if (!onto || !((view.ViewDimension == D3D11_RTV_DIMENSION_TEXTURE2D && view.Texture2D.MipSlice == 0u) || view.ViewDimension == D3D11_RTV_DIMENSION_TEXTURE2DMS)) {
        release(onto);
        release(target);
        LeaveCriticalSection(&state_lock);
        return;
      }
      // What it samples: a text texture, or a render target that has text on it.
      ID3D11ShaderResourceView *views[4] {};
      self->PSGetShaderResources(0u, 4u, views);
      bool text = false;
      int masked_slot = -1;
      mask_t source {};
      for (int slot = 0; slot < 4; ++slot) {
        if (!views[slot]) {
          continue;
        }
        if (ID3D11Texture2D *sampled = texture_of(views[slot])) {
          if (sampled != onto && mark_of(sampled) != 0u) {
            text = true;
          } else if (sampled != onto && masked_slot < 0 && find_mask(sampled, source)) {
            if (source.active) {
              masked_slot = slot;
            } else {
              source.drop();
            }
          }
          sampled->Release();
        }
        views[slot]->Release();
      }
      mask_t mask {};
      bool have = find_mask(onto, mask);
      if (text || masked_slot >= 0 || (have && mask.active)) {
        effect_e for_text, for_masked, for_other;
        effects_of(self, for_text, for_masked, for_other);
        const effect_e effect = text ? for_text : masked_slot >= 0 ? for_masked :
                                                                     for_other;
        const bool adds = effect == effect_e::add || effect == effect_e::add_premultiplied || effect == effect_e::copy;
        if (adds && !have) {
          have = make_mask(onto, view.Format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB || view.Format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, mask);
        }
        if (traced_frame == frame) {
          D3D11_TEXTURE2D_DESC onto_desc {};
          onto->GetDesc(&onto_desc);
          log_line("Direct3D 11: frame %u draw onto %p %ux%u: samples %s, effect %u, mask %s", frame, static_cast<void *>(onto), onto_desc.Width, onto_desc.Height, text ? "a text texture" : masked_slot >= 0 ? "a masked target" :
                                                                                                                                                                                                                 "neither",
                   static_cast<unsigned>(effect),
                   !have ? "none" : mask.active ? "active" :
                                                  "idle");
        }
        if (have && effect != effect_e::none && (adds || mask.active)) {
          if (adds) {
            activate(onto, mask);
            ++frame_text_draws;
            ++counted_text_draws;
          } else {
            ++counted_wipes;
          }
          repeat(self, mask, effect, text ? -1 : masked_slot, text ? nullptr : source.view, draw);
        }
      }
      mask.drop();
      source.drop();
      onto->Release();
      target->Release();
      keep_hooked(self);
      LeaveCriticalSection(&state_lock);
    }

    void STDMETHODCALLTYPE draw_indexed_hook(ID3D11DeviceContext *self, UINT count, UINT start, INT base) {
      original_draw_indexed(self, count, start, base);
      keep_hooked(self);
      if (followed(self)) {
        follow(self, [&] {
          original_draw_indexed(self, count, start, base);
        });
      }
    }

    void STDMETHODCALLTYPE draw_hook(ID3D11DeviceContext *self, UINT count, UINT start) {
      original_draw(self, count, start);
      keep_hooked(self);
      if (followed(self)) {
        follow(self, [&] {
          original_draw(self, count, start);
        });
      }
    }

    void STDMETHODCALLTYPE draw_indexed_instanced_hook(ID3D11DeviceContext *self, UINT count, UINT instances, UINT start, INT base, UINT first) {
      original_draw_indexed_instanced(self, count, instances, start, base, first);
      keep_hooked(self);
      if (followed(self)) {
        follow(self, [&] {
          original_draw_indexed_instanced(self, count, instances, start, base, first);
        });
      }
    }

    void STDMETHODCALLTYPE draw_instanced_hook(ID3D11DeviceContext *self, UINT count, UINT instances, UINT start, UINT first) {
      original_draw_instanced(self, count, instances, start, first);
      keep_hooked(self);
      if (followed(self)) {
        follow(self, [&] {
          original_draw_instanced(self, count, instances, start, first);
        });
      }
    }

    void STDMETHODCALLTYPE clear_hook(ID3D11DeviceContext *self, ID3D11RenderTargetView *target, const FLOAT *colour) {
      original_clear(self, target, colour);
      keep_hooked(self);
      if (followed(self) && target) {
        EnterCriticalSection(&state_lock);
        // Whatever was on this target is gone, and so is its mask's content.
        if (ID3D11Texture2D *texture = texture_of(target)) {
          if (flagged(texture, mask_active_id)) {
            flag(texture, mask_active_id, false);
          }
          texture->Release();
        }
        LeaveCriticalSection(&state_lock);
      }
    }

    /** One texture was copied onto another, wholly or in part: text and masks go along. */
    void carry(ID3D11DeviceContext *self, ID3D11Resource *to, ID3D11Resource *from, const bool whole, const UINT x, const UINT y, const D3D11_BOX *box) {
      ID3D11Texture2D *target = texture_of(to), *source = texture_of(from);
      if (target && source && target != source) {
        D3D11_TEXTURE2D_DESC target_desc {};
        target->GetDesc(&target_desc);
        // A render target takes text as its mask does, by what is drawn on it: an image copied
        // onto one is part of a picture from then on, and does not make it a text texture.
        const bool plain = (target_desc.BindFlags & D3D11_BIND_RENDER_TARGET) == 0u;
        if (plain && mark_of(source) != 0u) {
          // The same image as its source holds: other text only when that is.
          D3D11_TEXTURE2D_DESC desc {};
          target->GetDesc(&desc);
          std::uint64_t sum = 0u;
          UINT size = sizeof(sum);
          if (FAILED(source->GetPrivateData(image_sum_id, &size, &sum)) || size != sizeof(sum)) {
            sum = 0u;
          }
          mark_text(target, desc, sum);
        } else if (whole && mark_of(target) != 0u) {
          unmark(target);
          text_changed = true;
        }
        mask_t theirs {}, ours {};
        const bool had = find_mask(source, theirs) && theirs.active;
        bool have = find_mask(target, ours);
        if (had) {
          if (!have) {
            D3D11_TEXTURE2D_DESC desc {};
            theirs.texture->GetDesc(&desc);
            have = make_mask(target, desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, ours);
          }
          if (have) {
            activate(target, ours);
            if (whole) {
              original_copy(self, ours.texture, theirs.texture);
            } else {
              original_copy_region(self, ours.texture, 0u, x, y, 0u, theirs.texture, 0u, box);
            }
          }
        } else if (have && ours.active && whole) {
          flag(target, mask_active_id, false);
        }
        theirs.drop();
        ours.drop();
        keep_hooked(self);
      }
      release(target);
      release(source);
    }

    void STDMETHODCALLTYPE copy_hook(ID3D11DeviceContext *self, ID3D11Resource *to, ID3D11Resource *from) {
      original_copy(self, to, from);
      keep_hooked(self);
      if (self == context && !nested() && to && from) {
        EnterCriticalSection(&state_lock);
        carry(self, to, from, true, 0u, 0u, nullptr);
        LeaveCriticalSection(&state_lock);
      }
    }

    void STDMETHODCALLTYPE copy_region_hook(ID3D11DeviceContext *self, ID3D11Resource *to, UINT to_part, UINT x, UINT y, UINT z, ID3D11Resource *from, UINT from_part, const D3D11_BOX *box) {
      original_copy_region(self, to, to_part, x, y, z, from, from_part, box);
      keep_hooked(self);
      if (self == context && !nested() && to && from && to_part == 0u && from_part == 0u) {
        EnterCriticalSection(&state_lock);
        carry(self, to, from, false, x, y, box);
        LeaveCriticalSection(&state_lock);
      }
    }

    void STDMETHODCALLTYPE resolve_hook(ID3D11DeviceContext *self, ID3D11Resource *to, UINT to_part, ID3D11Resource *from, UINT from_part, DXGI_FORMAT format) {
      original_resolve(self, to, to_part, from, from_part, format);
      keep_hooked(self);
      if (self == context && !nested() && to && from && to_part == 0u && from_part == 0u) {
        EnterCriticalSection(&state_lock);
        carry(self, to, from, true, 0u, 0u, nullptr);
        LeaveCriticalSection(&state_lock);
      }
    }

    void STDMETHODCALLTYPE update_hook(ID3D11DeviceContext *self, ID3D11Resource *to, UINT part, const D3D11_BOX *box, const void *data, UINT pitch, UINT depth_pitch) {
      original_update(self, to, part, box, data, pitch, depth_pitch);
      keep_hooked(self);
      if (self == context && !nested() && to && data && part == 0u) {
        EnterCriticalSection(&state_lock);
        if (ID3D11Texture2D *texture = texture_of(to)) {
          D3D11_TEXTURE2D_DESC desc {};
          texture->GetDesc(&desc);
          const bool part_only = box && (box->left != 0u || box->top != 0u || box->right < desc.Width || box->bottom < desc.Height);
          const std::uint32_t width = box ? (box->right > box->left ? box->right - box->left : 0u) : desc.Width;
          const std::uint32_t height = box ? (box->bottom > box->top ? box->bottom - box->top : 0u) : desc.Height;
          note_image(texture, data, pitch, width, height, !part_only, false, "update");
          texture->Release();
        }
        LeaveCriticalSection(&state_lock);
      }
    }

    HRESULT STDMETHODCALLTYPE map_hook(ID3D11DeviceContext *self, ID3D11Resource *resource, UINT part, D3D11_MAP type, UINT flags, D3D11_MAPPED_SUBRESOURCE *result) {
      const HRESULT outcome = original_map(self, resource, part, type, flags, result);
      keep_hooked(self);
      D3D11_RESOURCE_DIMENSION kind = D3D11_RESOURCE_DIMENSION_UNKNOWN;
      if (SUCCEEDED(outcome) && self == context && !nested() && resource && result && result->pData && part == 0u && type != D3D11_MAP_READ && (resource->GetType(&kind), kind == D3D11_RESOURCE_DIMENSION_TEXTURE2D)) {
        EnterCriticalSection(&state_lock);
        mapped_t *free_entry = nullptr;
        for (auto &entry : mapped) {
          if (entry.resource == resource) {
            free_entry = &entry;
            break;
          }
          if (!entry.resource && !free_entry) {
            free_entry = &entry;
          }
        }
        if (free_entry) {
          *free_entry = {resource, result->pData, result->RowPitch};
        }
        LeaveCriticalSection(&state_lock);
      }
      return outcome;
    }

    void STDMETHODCALLTYPE unmap_hook(ID3D11DeviceContext *self, ID3D11Resource *resource, UINT part) {
      if (self == context && !nested() && resource && part == 0u) {
        EnterCriticalSection(&state_lock);
        for (auto &entry : mapped) {
          if (entry.resource != resource) {
            continue;
          }
          // What the application wrote is still there to be looked at, until the unmap below.
          if (ID3D11Texture2D *texture = texture_of(resource)) {
            D3D11_TEXTURE2D_DESC desc {};
            texture->GetDesc(&desc);
            note_image(texture, entry.data, entry.pitch, desc.Width, desc.Height, true, desc.Usage != D3D11_USAGE_STAGING, "mapping");
            texture->Release();
          }
          entry = {};
          break;
        }
        LeaveCriticalSection(&state_lock);
      }
      original_unmap(self, resource, part);
      keep_hooked(self);
    }

    // The calls below are followed for one reason only: Direct3D rewrites their table entries
    // along with the others, and a call that went past the hooks would take them all away. The
    // draws among them are followed as draws.

    void STDMETHODCALLTYPE draw_auto_hook(ID3D11DeviceContext *self) {
      original_draw_auto(self);
      keep_hooked(self);
      if (followed(self)) {
        follow(self, [&] {
          original_draw_auto(self);
        });
      }
    }

    void STDMETHODCALLTYPE draw_indexed_instanced_indirect_hook(ID3D11DeviceContext *self, ID3D11Buffer *arguments, UINT offset) {
      original_draw_indexed_instanced_indirect(self, arguments, offset);
      keep_hooked(self);
      if (followed(self)) {
        follow(self, [&] {
          original_draw_indexed_instanced_indirect(self, arguments, offset);
        });
      }
    }

    void STDMETHODCALLTYPE draw_instanced_indirect_hook(ID3D11DeviceContext *self, ID3D11Buffer *arguments, UINT offset) {
      original_draw_instanced_indirect(self, arguments, offset);
      keep_hooked(self);
      if (followed(self)) {
        follow(self, [&] {
          original_draw_instanced_indirect(self, arguments, offset);
        });
      }
    }

    void STDMETHODCALLTYPE dispatch_hook(ID3D11DeviceContext *self, UINT x, UINT y, UINT z) {
      original_dispatch(self, x, y, z);
      keep_hooked(self);
    }

    void STDMETHODCALLTYPE dispatch_indirect_hook(ID3D11DeviceContext *self, ID3D11Buffer *arguments, UINT offset) {
      original_dispatch_indirect(self, arguments, offset);
      keep_hooked(self);
    }

    void STDMETHODCALLTYPE copy_structure_count_hook(ID3D11DeviceContext *self, ID3D11Buffer *to, UINT offset, ID3D11UnorderedAccessView *from) {
      original_copy_structure_count(self, to, offset, from);
      keep_hooked(self);
    }

    void STDMETHODCALLTYPE clear_unordered_access_view_uint_hook(ID3D11DeviceContext *self, ID3D11UnorderedAccessView *view, const UINT *values) {
      original_clear_unordered_access_view_uint(self, view, values);
      keep_hooked(self);
    }

    void STDMETHODCALLTYPE clear_unordered_access_view_float_hook(ID3D11DeviceContext *self, ID3D11UnorderedAccessView *view, const FLOAT *values) {
      original_clear_unordered_access_view_float(self, view, values);
      keep_hooked(self);
    }

    void STDMETHODCALLTYPE clear_depth_stencil_view_hook(ID3D11DeviceContext *self, ID3D11DepthStencilView *view, UINT flags, FLOAT depth, UINT8 stencil) {
      original_clear_depth_stencil_view(self, view, flags, depth, stencil);
      keep_hooked(self);
    }

    void STDMETHODCALLTYPE generate_mips_hook(ID3D11DeviceContext *self, ID3D11ShaderResourceView *view) {
      original_generate_mips(self, view);
      keep_hooked(self);
    }

    void STDMETHODCALLTYPE flush_hook(ID3D11DeviceContext *self) {
      original_flush(self);
      keep_hooked(self);
    }

    // Waiting for a query's result and running a command list both flush.
    HRESULT STDMETHODCALLTYPE get_data_hook(ID3D11DeviceContext *self, ID3D11Asynchronous *query, void *data, UINT size, UINT flags) {
      const HRESULT outcome = original_get_data(self, query, data, size, flags);
      keep_hooked(self);
      return outcome;
    }

    void STDMETHODCALLTYPE execute_command_list_hook(ID3D11DeviceContext *self, ID3D11CommandList *list, BOOL restore) {
      original_execute_command_list(self, list, restore);
      keep_hooked(self);
    }

    /** A hook of the followed context's table: its entry, the hook, and where its original goes. */
    struct entry_t {
      unsigned slot;
      void *hook;
      void **original;
    };

#define SUNSHINE_ENTRY(slot, function) \
  {slot, reinterpret_cast<void *>(&function##_hook), reinterpret_cast<void **>(&original_##function)}
    const entry_t entries[] {
      SUNSHINE_ENTRY(slot_draw_indexed, draw_indexed),
      SUNSHINE_ENTRY(slot_draw, draw),
      SUNSHINE_ENTRY(slot_map, map),
      SUNSHINE_ENTRY(slot_unmap, unmap),
      SUNSHINE_ENTRY(slot_draw_indexed_instanced, draw_indexed_instanced),
      SUNSHINE_ENTRY(slot_draw_instanced, draw_instanced),
      SUNSHINE_ENTRY(slot_draw_auto, draw_auto),
      SUNSHINE_ENTRY(slot_draw_indexed_instanced_indirect, draw_indexed_instanced_indirect),
      SUNSHINE_ENTRY(slot_draw_instanced_indirect, draw_instanced_indirect),
      SUNSHINE_ENTRY(slot_dispatch, dispatch),
      SUNSHINE_ENTRY(slot_dispatch_indirect, dispatch_indirect),
      SUNSHINE_ENTRY(slot_copy_subresource_region, copy_region),
      SUNSHINE_ENTRY(slot_copy_resource, copy),
      SUNSHINE_ENTRY(slot_update_subresource, update),
      SUNSHINE_ENTRY(slot_copy_structure_count, copy_structure_count),
      SUNSHINE_ENTRY(slot_clear_render_target_view, clear),
      SUNSHINE_ENTRY(slot_clear_unordered_access_view_uint, clear_unordered_access_view_uint),
      SUNSHINE_ENTRY(slot_clear_unordered_access_view_float, clear_unordered_access_view_float),
      SUNSHINE_ENTRY(slot_clear_depth_stencil_view, clear_depth_stencil_view),
      SUNSHINE_ENTRY(slot_generate_mips, generate_mips),
      SUNSHINE_ENTRY(slot_resolve_subresource, resolve),
      SUNSHINE_ENTRY(slot_flush, flush),
      SUNSHINE_ENTRY(slot_get_data, get_data),
      SUNSHINE_ENTRY(slot_execute_command_list, execute_command_list),
    };
#undef SUNSHINE_ENTRY

    /**
     * Point the followed context's table at the hooks. Whatever an entry holds instead of its
     * hook is the function Direct3D now wants called there, and becomes that hook's original.
     */
    /**
     * Direct3D's own context keeps its function table in memory of its own, which is how the
     * table can be rewritten at all. A context that other code wraps, as an overlay or a
     * translation layer does, has the table its compiler made, in memory that cannot be written:
     * such a context is not followed.
     */
    bool table_writable(ID3D11DeviceContext *of) {
      void **table = *reinterpret_cast<void ***>(of);
      MEMORY_BASIC_INFORMATION memory {};
      if (VirtualQuery(table, &memory, sizeof(memory)) == 0u || memory.State != MEM_COMMIT || memory.Type != MEM_PRIVATE || (memory.Protect & 0xffu) != PAGE_READWRITE) {
        return false;
      }
      const auto *end = static_cast<const std::uint8_t *>(memory.BaseAddress) + memory.RegionSize;
      return reinterpret_cast<const std::uint8_t *>(&table[slot_flush + 1u]) <= end;
    }

    void hook_context() {
      void **table = *reinterpret_cast<void ***>(context);
      for (const auto &entry : entries) {
        if (table[entry.slot] != entry.hook) {
          *entry.original = table[entry.slot];
          table[entry.slot] = entry.hook;
        }
      }
    }

    void keep_hooked(ID3D11DeviceContext *self) {
      // Direct3D rewrites all of its entries together, so one of them tells for all.
      if (context && self == context && (*reinterpret_cast<void ***>(self))[slot_draw] != reinterpret_cast<void *>(&draw_hook)) {
        hook_context();
      }
    }

    HRESULT STDMETHODCALLTYPE create_texture_hook(ID3D11Device *self, const D3D11_TEXTURE2D_DESC *desc, const D3D11_SUBRESOURCE_DATA *initial, ID3D11Texture2D **texture) {
      const HRESULT outcome = original_create_texture(self, desc, initial, texture);
      if (SUCCEEDED(outcome) && self == device && !nested() && desc && initial && initial->pSysMem && texture && *texture) {
        EnterCriticalSection(&state_lock);
        note_image(*texture, initial->pSysMem, initial->SysMemPitch, desc->Width, desc->Height, true, false, "creation");
        LeaveCriticalSection(&state_lock);
      }
      return outcome;
    }

    // --- the published picture ------------------------------------------------------------------

    void release_own() {
      release(staging);
      staging_width = staging_height = 0u;
      copy_pending = false;
      release(blend_add);
      release(blend_add_premultiplied);
      release(blend_erase);
      release(blend_erase_all);
      release(context);
      release(device);
      std::memset(mapped, 0, sizeof(mapped));
    }

    ID3D11BlendState *make_blend(const D3D11_BLEND source, const D3D11_BLEND target, const D3D11_BLEND source_alpha, const D3D11_BLEND target_alpha) {
      D3D11_BLEND_DESC desc {};
      auto &first = desc.RenderTarget[0];
      first.BlendEnable = TRUE;
      first.SrcBlend = source;
      first.DestBlend = target;
      first.BlendOp = D3D11_BLEND_OP_ADD;
      first.SrcBlendAlpha = source_alpha;
      first.DestBlendAlpha = target_alpha;
      first.BlendOpAlpha = D3D11_BLEND_OP_ADD;
      first.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
      ID3D11BlendState *state = nullptr;
      device->CreateBlendState(&desc, &state);
      return state;
    }

    template<class Function>
    bool patch(void **vtable, const unsigned slot, void *hook, Function &original) {
      DWORD protection = 0u;
      if (!VirtualProtect(&vtable[slot], sizeof(void *), PAGE_EXECUTE_READWRITE, &protection)) {
        return false;
      }
      original = reinterpret_cast<Function>(vtable[slot]);
      vtable[slot] = hook;
      VirtualProtect(&vtable[slot], sizeof(void *), protection, &protection);
      return true;
    }

    /** Hook the device's own table, which every Direct3D 11 device shares, and its context's. */
    bool hook_device() {
      if (!table_writable(context)) {
        return false;
      }
      void **of_device = *reinterpret_cast<void ***>(device);
      if (of_device[slot_create_texture_2d] != reinterpret_cast<void *>(&create_texture_hook) && (original_create_texture || !patch(of_device, slot_create_texture_2d, reinterpret_cast<void *>(&create_texture_hook), original_create_texture))) {
        // A second kind of device, or a table that cannot be written: its textures are not seen.
        return false;
      }
      hook_context();
      return true;
    }

    /** Follow the device this swap chain presents with. False where it is not Direct3D 11. */
    bool adopt(IDXGISwapChain *chain) {
      ID3D11Device *presenting = nullptr;
      if (FAILED(chain->GetDevice(__uuidof(ID3D11Device), reinterpret_cast<void **>(&presenting))) || !presenting) {
        return false;
      }
      if (presenting == device) {
        presenting->Release();
        return context != nullptr;
      }
      release_own();
      device = presenting;
      device->GetImmediateContext(&context);
      const bool hooked = context && hook_device();
      enter_call();
      blend_add = make_blend(D3D11_BLEND_SRC_ALPHA, D3D11_BLEND_INV_SRC_ALPHA, D3D11_BLEND_ONE, D3D11_BLEND_INV_SRC_ALPHA);
      blend_add_premultiplied = make_blend(D3D11_BLEND_ONE, D3D11_BLEND_INV_SRC_ALPHA, D3D11_BLEND_ONE, D3D11_BLEND_INV_SRC_ALPHA);
      blend_erase = make_blend(D3D11_BLEND_ZERO, D3D11_BLEND_INV_SRC_ALPHA, D3D11_BLEND_ZERO, D3D11_BLEND_INV_SRC_ALPHA);
      blend_erase_all = make_blend(D3D11_BLEND_ZERO, D3D11_BLEND_ZERO, D3D11_BLEND_ZERO, D3D11_BLEND_ZERO);
      leave_call();
      const bool ready = hooked && blend_add && blend_add_premultiplied && blend_erase && blend_erase_all;
      log_line("Direct3D 11: device %p %s", static_cast<void *>(device), ready ? "followed" : hooked ? "cannot be drawn on by the helper" :
                                                                                                       "is of a kind that is not hooked");
      if (!ready) {
        // The device is remembered, so that this is not tried again at every frame; without a
        // context no hook takes any call for the followed one.
        release(context);
      }
      return ready;
    }

    bool ensure_own(const std::uint32_t width, const std::uint32_t height) {
      if (!ensure_surface(width, height)) {
        return false;
      }
      if (own_pixels && own_width == width && own_height == height) {
        return true;
      }
      if (own_pixels) {
        HeapFree(GetProcessHeap(), 0u, own_pixels);
      }
      own_pixels = static_cast<std::uint32_t *>(HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, static_cast<std::size_t>(width) * height * 4u));
      own_width = own_pixels ? width : 0u;
      own_height = own_pixels ? height : 0u;
      published_top = published_bottom = 0u;
      return own_pixels != nullptr;
    }

    /** Publish `rows` rows of BGRA at `data`, or no text at all for null. */
    void publish_picture(const void *data, const std::uint32_t pitch, const std::uint32_t width, const std::uint32_t height, const HWND window) {
      if (!ensure_own(width, height)) {
        return;
      }
      // The rows that hold anything, and how much of the picture is covered.
      std::uint32_t first = height, last = 0u;
      std::uint64_t covered = 0u;
      for (std::uint32_t y = 0u; data && y < height; ++y) {
        const auto *row = static_cast<const std::uint32_t *>(static_cast<const void *>(static_cast<const std::uint8_t *>(data) + static_cast<std::size_t>(y) * pitch));
        std::uint32_t any = 0u, solid = 0u;
        for (std::uint32_t x = 0u; x < width; ++x) {
          any |= row[x];
          solid += row[x] >> 31u;  // coverage of a half or more
        }
        covered += solid;
        if ((any & 0xff000000u) != 0u) {
          first = first == height ? y : first;
          last = y;
        }
      }
      const bool unsure = static_cast<float>(covered) > most_of_picture * static_cast<float>(static_cast<std::uint64_t>(width) * height);
      if (unsure) {
        static bool told = false;
        if (!told) {
          told = true;
          log_line("Direct3D 11: the mask covers %u%% of the picture, which is not text; nothing is published", static_cast<unsigned>(covered * 100u / (static_cast<std::uint64_t>(width) * height)));
        }
        first = height;
      }
      const bool text = first != height;
      const std::uint32_t top = text ? (first >= 2u ? first - 2u : 0u) : 0u;
      const std::uint32_t bottom = text ? min_of(last + 3u, height) : 0u;
      // The rows that held text before are published too: that text may be gone from them.
      const bool before = published_bottom > published_top;
      const std::uint32_t from = text && before ? min_of(top, published_top) : text ? top :
                                                                                      published_top;
      const std::uint32_t to = max_of(bottom, published_bottom);
      for (std::uint32_t y = from; y < to; ++y) {
        std::uint32_t *own = own_pixels + static_cast<std::size_t>(y) * width;
        if (y >= top && y < bottom) {
          std::memcpy(own, static_cast<const std::uint8_t *>(data) + static_cast<std::size_t>(y) * pitch, static_cast<std::size_t>(width) * 4u);
        } else {
          std::memset(own, 0, static_cast<std::size_t>(width) * 4u);
        }
      }
      if (data && logging() && GetTickCount64() - dumped_tick >= 2500u) {
        dumped_tick = GetTickCount64();
        log_line("Direct3D 11: frame %u the mask is picture mask-%u, text rows %u to %u, %u%% covered", frame, dumped_masks % 12u, top, bottom, static_cast<unsigned>(covered * 100u / (static_cast<std::uint64_t>(width) * height)));
        dump("mask", dumped_masks++ % 12u, data, pitch, width, height);
      }
      publish(own_pixels, from, to, unsure ? source_unsure : source_direct3d11, top, bottom, window);
      published_top = top;
      published_bottom = bottom;
    }

    void presented(IDXGISwapChain *chain) {
      // Text the application draws with Direct2D is that part's to publish.
      DXGI_SWAP_CHAIN_DESC chain_desc {};
      if (d2d::active() || FAILED(chain->GetDesc(&chain_desc))) {
        return;
      }
      // One swap chain is followed: the application's picture. Another one, of a film's window,
      // a launcher or a library that presents for itself, takes its place only when its picture
      // is larger or the followed one has stopped presenting. Following each in turn would undo
      // the helper's work at every present.
      const std::uint64_t tick = GetTickCount64();
      const std::uint64_t texels = static_cast<std::uint64_t>(chain_desc.BufferDesc.Width) * chain_desc.BufferDesc.Height;
      if (followed_chain && chain != followed_chain && texels <= followed_texels && tick - followed_tick < 1000u) {
        return;
      }
      followed_chain = chain;
      followed_tick = tick;
      followed_texels = texels;
      if (!adopt(chain)) {
        return;
      }
      ++frame;
      ++counted_frames;
      counted_tick = counted_tick ? counted_tick : tick;
      if (tick - counted_tick >= 5000u) {
        log_line("Direct3D 11, 5 s: %u frames, %u draws, %u images loaded, %u glyphs taken; %u draws looked at, %u of text, %u wipes, %u masks read", counted_frames, counted_all_draws, counted_images, static_cast<unsigned>(counted_glyphs), counted_draws, counted_text_draws, counted_wipes, counted_reads);
        counted_frames = counted_draws = counted_text_draws = counted_wipes = counted_reads = counted_all_draws = counted_images = 0u;
        counted_glyphs = 0u;
        counted_tick = tick;
        traced_frame = logging() ? frame + 1u : 0u;
      }
      const unsigned text_draws = frame_text_draws;
      const bool draws_changed = text_draws != last_frame_text_draws;
      last_frame_text_draws = text_draws;
      frame_text_draws = 0u;
      if (!marked_ever) {
        return;
      }
      ID3D11Texture2D *back = nullptr;
      if (FAILED(chain->GetBuffer(0u, __uuidof(ID3D11Texture2D), reinterpret_cast<void **>(&back))) || !back) {
        release(back);
        return;
      }
      D3D11_TEXTURE2D_DESC desc {};
      back->GetDesc(&desc);
      mask_t mask {};
      const bool masked = find_mask(back, mask) && mask.active;
      enter_call();
      if (masked) {
        D3D11_TEXTURE2D_DESC own {};
        mask.texture->GetDesc(&own);
        if (!staging || staging_width != own.Width || staging_height != own.Height || staging_format != own.Format) {
          release(staging);
          copy_pending = false;
          own.Usage = D3D11_USAGE_STAGING;
          own.BindFlags = 0u;
          own.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
          device->CreateTexture2D(&own, nullptr, &staging);
          staging_width = own.Width;
          staging_height = own.Height;
          staging_format = own.Format;
        }
        // Kept until it has been read: a change may come in a frame that is not read.
        text_changed = text_changed || draws_changed;
        if (staging && !copy_pending && frame - last_copy_frame >= (text_changed ? changed_read_frames : steady_read_frames)) {
          original_copy(context, staging, mask.texture);
          copy_pending = true;
          text_changed = false;
          last_copy_frame = frame;
        }
      } else {
        // Nothing of the text reached the picture in this frame; a copy still on its way is of
        // a frame that is over.
        copy_pending = false;
        if (published_bottom > published_top) {
          publish_picture(nullptr, 0u, desc.Width, desc.Height, chain_desc.OutputWindow);
        }
      }
      if (copy_pending && staging) {
        // The copy is taken when the graphics processor has made it, never waited for.
        D3D11_MAPPED_SUBRESOURCE memory {};
        const HRESULT read = original_map(context, staging, 0u, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &memory);
        if (read != DXGI_ERROR_WAS_STILL_DRAWING) {
          copy_pending = false;
          if (SUCCEEDED(read) && memory.pData) {
            ++counted_reads;
            publish_picture(memory.pData, memory.RowPitch, staging_width, staging_height, chain_desc.OutputWindow);
            original_unmap(context, staging, 0u);
          }
        }
      }
      leave_call();
      mask.drop();
      back->Release();
    }

    HRESULT STDMETHODCALLTYPE present_hook(IDXGISwapChain *self, UINT interval, UINT flags) {
      if (!nested() && (flags & DXGI_PRESENT_TEST) == 0u) {
        EnterCriticalSection(&state_lock);
        presented(self);
        LeaveCriticalSection(&state_lock);
      }
      enter_call();
      const HRESULT outcome = original_present(self, interval, flags);
      leave_call();
      // Presenting flushes.
      EnterCriticalSection(&state_lock);
      keep_hooked(context);
      LeaveCriticalSection(&state_lock);
      return outcome;
    }

    HRESULT STDMETHODCALLTYPE present_1_hook(IDXGISwapChain *self, UINT interval, UINT flags, const void *parameters) {
      if (!nested() && (flags & DXGI_PRESENT_TEST) == 0u) {
        EnterCriticalSection(&state_lock);
        presented(self);
        LeaveCriticalSection(&state_lock);
      }
      enter_call();
      const HRESULT outcome = original_present_1(self, interval, flags, parameters);
      leave_call();
      EnterCriticalSection(&state_lock);
      keep_hooked(context);
      LeaveCriticalSection(&state_lock);
      return outcome;
    }

  }  // namespace

  bool install() {
    if (install_settled) {
      return true;
    }
    // The helper brings Direct3D into no application that does not use it.
    const HMODULE direct3d = GetModuleHandleW(L"d3d11.dll");
    if (!direct3d) {
      return false;
    }
    install_settled = true;
    using create_fn = HRESULT(WINAPI *)(IDXGIAdapter *, D3D_DRIVER_TYPE, HMODULE, UINT, const D3D_FEATURE_LEVEL *, UINT, UINT, const DXGI_SWAP_CHAIN_DESC *, IDXGISwapChain **, ID3D11Device **, D3D_FEATURE_LEVEL *, ID3D11DeviceContext **);
    const auto create = reinterpret_cast<create_fn>(reinterpret_cast<void *>(GetProcAddress(direct3d, "D3D11CreateDeviceAndSwapChain")));
    // Every swap chain shares one vtable, whichever adapter it presents with: a software device
    // on a window that is never shown yields it. A device and its context are hooked when one
    // presents.
    const HWND window = CreateWindowExW(0u, L"STATIC", L"", WS_POPUP, 0, 0, 16, 16, nullptr, nullptr, nullptr, nullptr);
    DXGI_SWAP_CHAIN_DESC desc {};
    desc.BufferDesc.Width = 16u;
    desc.BufferDesc.Height = 16u;
    desc.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1u;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = 1u;
    desc.OutputWindow = window;
    desc.Windowed = TRUE;
    IDXGISwapChain *sample_chain = nullptr;
    ID3D11Device *sample_device = nullptr;
    ID3D11DeviceContext *sample_context = nullptr;
    enter_call();
    const bool created = create && window &&
                         SUCCEEDED(create(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0u, nullptr, 0u, D3D11_SDK_VERSION, &desc, &sample_chain, &sample_device, nullptr, &sample_context)) &&
                         sample_chain && sample_device && sample_context;
    leave_call();
    if (created) {
      void **of_chain = *reinterpret_cast<void ***>(sample_chain);
      const bool patched = patch(of_chain, slot_present_1, reinterpret_cast<void *>(&present_1_hook), original_present_1) &&
                           patch(of_chain, slot_present, reinterpret_cast<void *>(&present_hook), original_present);
      log_line("Direct3D 11: presenting %s", patched ? "hooked" : "could not be hooked");
    } else {
      log_line("Direct3D 11: no device to learn the vtables from");
    }
    enter_call();
    release(sample_chain);
    release(sample_context);
    release(sample_device);
    leave_call();
    if (window) {
      DestroyWindow(window);
    }
    return true;
  }

  void glyphs_taken() {
    glyph_frame = frame;
    counted_glyphs = counted_glyphs + 1u;
  }
}  // namespace sunshine_text_hook::helper::d3d11
