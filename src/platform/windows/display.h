/**
 * @file src/platform/windows/display.h
 * @brief Declarations for the Windows display backend.
 */
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <vector>

// platform includes
#include <d3d11.h>
#include <d3d11_4.h>
#include <d3dcommon.h>
#include <dwmapi.h>
#include <dxgi.h>
#include <dxgi1_6.h>
#include <Unknwn.h>
#include <winrt/windows.graphics.capture.h>

// local includes
#include "src/platform/common.h"
#include "src/utility.h"
#include "src/video.h"

namespace platf::dxgi {
  extern const char *format_str[];

  // Add D3D11_CREATE_DEVICE_DEBUG here to enable the D3D11 debug runtime.
  // You should have a debugger like WinDbg attached to receive debug messages.
  auto constexpr D3D11_CREATE_DEVICE_FLAGS = 0;

  template<class T>
  void Release(T *dxgi) {
    dxgi->Release();
  }

  using factory1_t = util::safe_ptr<IDXGIFactory1, Release<IDXGIFactory1>>;
  using dxgi_t = util::safe_ptr<IDXGIDevice, Release<IDXGIDevice>>;
  using dxgi1_t = util::safe_ptr<IDXGIDevice1, Release<IDXGIDevice1>>;
  using device_t = util::safe_ptr<ID3D11Device, Release<ID3D11Device>>;
  using device1_t = util::safe_ptr<ID3D11Device1, Release<ID3D11Device1>>;
  using device_ctx_t = util::safe_ptr<ID3D11DeviceContext, Release<ID3D11DeviceContext>>;
  using adapter_t = util::safe_ptr<IDXGIAdapter1, Release<IDXGIAdapter1>>;
  using output_t = util::safe_ptr<IDXGIOutput, Release<IDXGIOutput>>;
  using output1_t = util::safe_ptr<IDXGIOutput1, Release<IDXGIOutput1>>;
  using output5_t = util::safe_ptr<IDXGIOutput5, Release<IDXGIOutput5>>;
  using output6_t = util::safe_ptr<IDXGIOutput6, Release<IDXGIOutput6>>;
  using dup_t = util::safe_ptr<IDXGIOutputDuplication, Release<IDXGIOutputDuplication>>;
  using texture2d_t = util::safe_ptr<ID3D11Texture2D, Release<ID3D11Texture2D>>;
  using resource_t = util::safe_ptr<IDXGIResource, Release<IDXGIResource>>;
  using resource1_t = util::safe_ptr<IDXGIResource1, Release<IDXGIResource1>>;
  using vs_t = util::safe_ptr<ID3D11VertexShader, Release<ID3D11VertexShader>>;
  using ps_t = util::safe_ptr<ID3D11PixelShader, Release<ID3D11PixelShader>>;
  using blend_t = util::safe_ptr<ID3D11BlendState, Release<ID3D11BlendState>>;
  using render_target_t = util::safe_ptr<ID3D11RenderTargetView, Release<ID3D11RenderTargetView>>;
  using shader_res_t = util::safe_ptr<ID3D11ShaderResourceView, Release<ID3D11ShaderResourceView>>;
  using buf_t = util::safe_ptr<ID3D11Buffer, Release<ID3D11Buffer>>;
  using sampler_state_t = util::safe_ptr<ID3D11SamplerState, Release<ID3D11SamplerState>>;
  using blob_t = util::safe_ptr<ID3DBlob, Release<ID3DBlob>>;
  using keyed_mutex_t = util::safe_ptr<IDXGIKeyedMutex, Release<IDXGIKeyedMutex>>;

  namespace detail {
    /** Tracks whether a persistent encoder input already contains a converted Host-SBS output.
     *
     * Native NVENC registers one D3D11 input texture for the lifetime of an encode session. A
     * repeated packed SBS image therefore does not need another RGB-to-YUV draw once that exact
     * texture has been initialized. Local RGB presentation is deliberately excluded: its
     * swapchain backbuffer can rotate on every Present().
     */
    class host_sbs_encoder_input_state_t {
    public:
      [[nodiscard]] constexpr bool conversion_required(
        const bool repeats_prior_output,
        const bool local_rgb_presentation
      ) const noexcept {
        return local_rgb_presentation || !repeats_prior_output || !initialized_;
      }

      constexpr void mark_converted() noexcept {
        initialized_ = true;
      }

      constexpr void reset() noexcept {
        initialized_ = false;
      }

      [[nodiscard]] constexpr bool initialized() const noexcept {
        return initialized_;
      }

    private:
      bool initialized_ = false;
    };

    /** Accepted-frame synchronization is reserved for an old source that already exceeded the
     * bounded repeat window. Startup and ordinary analysis-domain resets have no stale source and
     * must remain on the normal asynchronous completion path. */
    [[nodiscard]] constexpr bool host_sbs_accepted_frame_needs_synchronous_recovery(
      const bool stale_prior_completion,
      const bool stale_prior_output
    ) noexcept {
      return stale_prior_completion || stale_prior_output;
    }

    [[nodiscard]] constexpr bool host_sbs_window_authority_observation_needed(
      const bool has_depth_estimator,
      const bool renderer_uses_depth_pipeline
    ) noexcept {
      return has_depth_estimator && renderer_uses_depth_pipeline;
    }

    inline constexpr auto host_sbs_full_source_reuse_max_age =
      std::chrono::milliseconds {250};
    inline constexpr unsigned host_sbs_full_source_reuse_max_skips = 16u;
    inline constexpr auto host_sbs_low_motion_reuse_max_age =
      std::chrono::milliseconds {50};
    inline constexpr unsigned host_sbs_low_motion_reuse_max_skips = 1u;
    // 1 / 400 = 0.25%. Keep this integer-ratio contract exact and overflow-free.
    inline constexpr std::uint64_t host_sbs_low_motion_damage_ratio_denominator = 400u;

    [[nodiscard]] constexpr bool host_sbs_cached_geometry_render_allowed(
      const bool dedup_gate_open,
      const bool renderer_authenticated,
      const bool cached_geometry_matches,
      const bool cached_resources_complete
    ) noexcept {
      return dedup_gate_open && renderer_authenticated && cached_geometry_matches &&
             cached_resources_complete;
    }

    /** Production-independent selector for the default-off completed-cache experiment. */
    [[nodiscard]] constexpr bool host_sbs_low_motion_cache_reuse_allowed(
      const bool experiment_enabled,
      const bool dedup_gate_open,
      const bool cache_authenticated,
      const bool inference_pending,
      const bool damage_candidate,
      const bool refresh_allowed
    ) noexcept {
      return experiment_enabled && dedup_gate_open && cache_authenticated &&
             !inference_pending && damage_candidate && refresh_allowed;
    }

    /** Bounded unchanged-content refresh state. A busy admission attempt is intentionally a no-op;
     * only a real enqueue resets the saturated age/skip cap. */
    class host_sbs_content_refresh_state_t {
    public:
      constexpr void reset() noexcept {
        last_enqueued_at_.reset();
        skipped_ = 0u;
      }

      constexpr void record_successful_enqueue(
        const std::chrono::steady_clock::time_point now
      ) noexcept {
        last_enqueued_at_ = now;
        skipped_ = 0u;
      }

      constexpr void record_reuse() noexcept {
        if (skipped_ < host_sbs_full_source_reuse_max_skips) {
          ++skipped_;
        }
      }

      [[nodiscard]] constexpr bool refresh_due(
        const std::chrono::steady_clock::time_point now
      ) const noexcept {
        return !last_enqueued_at_ ||
               skipped_ >= host_sbs_full_source_reuse_max_skips ||
               now - *last_enqueued_at_ >= host_sbs_full_source_reuse_max_age;
      }

      [[nodiscard]] constexpr unsigned skipped() const noexcept {
        return skipped_;
      }

    private:
      std::optional<std::chrono::steady_clock::time_point> last_enqueued_at_;
      unsigned skipped_ = 0u;
    };

    /** Separate quality-surface bound for approximate low-motion holds.
     *
     * Exact DDup reuse retains the wider 16/250 cap above. Approximate reuse gets one delivery
     * and 50 ms from the last real enqueue; exact holds neither consume nor refresh this budget.
     */
    class host_sbs_low_motion_refresh_state_t {
    public:
      constexpr void reset() noexcept {
        last_enqueued_at_.reset();
        skipped_ = 0u;
      }

      constexpr void record_successful_enqueue(
        const std::chrono::steady_clock::time_point now
      ) noexcept {
        last_enqueued_at_ = now;
        skipped_ = 0u;
      }

      constexpr void record_reuse() noexcept {
        if (skipped_ < host_sbs_low_motion_reuse_max_skips) {
          ++skipped_;
        }
      }

      [[nodiscard]] constexpr bool reuse_allowed(
        const std::chrono::steady_clock::time_point now
      ) const noexcept {
        return last_enqueued_at_ &&
               skipped_ < host_sbs_low_motion_reuse_max_skips &&
               now - *last_enqueued_at_ < host_sbs_low_motion_reuse_max_age;
      }

      [[nodiscard]] constexpr unsigned skipped() const noexcept {
        return skipped_;
      }

    private:
      std::optional<std::chrono::steady_clock::time_point> last_enqueued_at_;
      unsigned skipped_ = 0u;
    };

    /** CPU shadow for an immutable-by-value GPU upload. The caller commits only after the D3D
     * create/update operation has been submitted successfully. */
    template<typename T>
    class uploaded_value_state_t {
    public:
      [[nodiscard]] constexpr bool is_current(const T &value) const {
        return value_ && *value_ == value;
      }

      constexpr void commit(const T &value) {
        value_ = value;
      }

      constexpr void reset() noexcept {
        value_.reset();
      }

    private:
      std::optional<T> value_;
    };

    /** Non-owning scope guard for one successful IDXGIKeyedMutex acquisition.
     *
     * Keeping this generic makes the ownership contract testable without a D3D device. The
     * referenced mutex must outlive the guard, as it does for an image encoder context.
     */
    template<typename KeyedMutex>
    class keyed_mutex_lock_t {
    public:
      explicit keyed_mutex_lock_t(KeyedMutex *mutex) noexcept:
          mutex_ {mutex} {
      }

      keyed_mutex_lock_t(const keyed_mutex_lock_t &) = delete;
      keyed_mutex_lock_t &operator=(const keyed_mutex_lock_t &) = delete;

      ~keyed_mutex_lock_t() noexcept {
        if (locked_) {
          mutex_->ReleaseSync(release_key_);
        }
      }

      [[nodiscard]] HRESULT lock(
        UINT64 acquire_key = 0,
        DWORD timeout_ms = INFINITE,
        UINT64 release_key = 0
      ) noexcept {
        if (locked_) {
          return S_OK;
        }
        if (!mutex_) {
          return E_POINTER;
        }

        const auto status = mutex_->AcquireSync(acquire_key, timeout_ms);
        if (status == S_OK) {
          locked_ = true;
          release_key_ = release_key;
        }
        return status;
      }

      [[nodiscard]] bool owns_lock() const noexcept {
        return locked_;
      }

    private:
      KeyedMutex *mutex_;
      UINT64 release_key_ = 0;
      bool locked_ = false;
    };

    template<typename Timestamp>
    struct ddup_timestamp_selection_t {
      std::optional<Timestamp> presentation_timestamp;
      std::optional<Timestamp> content_timestamp;
    };

    /** Advance Desktop Duplication timestamp state for one acquired frame.
     *
     * Cursor movement advances presentation cadence but retains the last desktop-content time;
     * a present advances both, even when a newer cursor update shares the acquisition.
     */
    template<typename Timestamp>
    constexpr ddup_timestamp_selection_t<Timestamp> select_ddup_timestamps(
      std::optional<Timestamp> present_timestamp,
      std::optional<Timestamp> mouse_timestamp,
      std::optional<Timestamp> retained_content_timestamp
    ) {
      auto presentation_timestamp = present_timestamp;
      if (
        mouse_timestamp &&
        (!presentation_timestamp || *presentation_timestamp < *mouse_timestamp)
      ) {
        presentation_timestamp = mouse_timestamp;
      }
      if (present_timestamp) {
        retained_content_timestamp = present_timestamp;
      }
      return {presentation_timestamp, retained_content_timestamp};
    }

    inline constexpr std::size_t ddup_damage_history_frame_budget = 128u;
    inline constexpr std::size_t ddup_damage_history_rect_budget = 4096u;
    inline constexpr std::size_t ddup_damage_frame_rect_budget = 512u;
    inline constexpr UINT ddup_damage_metadata_byte_budget = 64u * 1024u;

    /** DDup only promises dirty/move lists when update metadata is present.
     *
     * A delivered content present with a zero metadata byte count is not proof that no pixels
     * changed. Keep that acquisition as an explicit history discontinuity instead of committing a
     * known-empty update.
     */
    [[nodiscard]] constexpr bool ddup_frame_damage_metadata_available(
      const DXGI_OUTDUPL_FRAME_INFO &frame_info
    ) noexcept {
      return frame_info.LastPresentTime.QuadPart != 0 &&
             frame_info.AccumulatedFrames != 0u &&
             frame_info.TotalMetadataBufferSize != 0u &&
             frame_info.TotalMetadataBufferSize <= ddup_damage_metadata_byte_budget &&
             !frame_info.ProtectedContentMaskedOut;
    }

    /** Conservative answer for damage accumulated between two committed DDup surfaces. */
    enum class ddup_damage_intersection_e : std::uint8_t {
      unknown,
      unchanged,
      changed,
    };

    /** Saturated upper bound on pixels that may have changed inside one queried region.
     *
     * Rectangles are deliberately not unioned: summing clipped overlaps and saturating at the
     * region area can only overestimate damage, so an accepted small fraction stays conservative.
     */
    struct ddup_damage_coverage_t {
      std::uint64_t potentially_changed_area = 0u;
      std::uint64_t region_area = 0u;
      bool known = false;
    };

    [[nodiscard]] constexpr bool host_sbs_low_motion_damage_candidate(
      const ddup_damage_coverage_t &coverage,
      const bool ocr_crop_unchanged
    ) noexcept {
      return coverage.known && ocr_crop_unchanged && coverage.region_area != 0u &&
             coverage.potentially_changed_area <=
               coverage.region_area /
                 host_sbs_low_motion_damage_ratio_denominator;
    }

    /** One acquired DDup present's normalized dirty coverage.
     *
     * Move rectangles contribute both their source and destination rectangles. `known == false`
     * is an explicit discontinuity: callers must not infer an unchanged ROI across this update.
     */
    struct ddup_damage_update_t {
      bool known = false;
      std::vector<RECT> rects;
    };

    class ddup_damage_history_t;

    /** Immutable identity of one desktop surface actually committed by CopyResource. */
    struct ddup_damage_snapshot_t {
      std::shared_ptr<const ddup_damage_history_t> history;
      std::uint64_t token = 0u;
    };

    /** Validate and normalize DDup dirty/move metadata into capture-texture coordinates. */
    [[nodiscard]] ddup_damage_update_t make_ddup_damage_update(
      std::span<const RECT> dirty_rects,
      std::span<const DXGI_OUTDUPL_MOVE_RECT> move_rects,
      LONG width,
      LONG height,
      bool metadata_valid = true
    );

    /** Bounded, thread-safe history shared by capture images and the encode thread. */
    class ddup_damage_history_t:
        public std::enable_shared_from_this<ddup_damage_history_t> {
    public:
      [[nodiscard]] ddup_damage_snapshot_t commit(ddup_damage_update_t update);

      [[nodiscard]] ddup_damage_intersection_e query(
        std::uint64_t from_exclusive,
        std::uint64_t through_inclusive,
        const RECT &region
      ) const;

      [[nodiscard]] ddup_damage_coverage_t query_coverage(
        std::uint64_t from_exclusive,
        std::uint64_t through_inclusive,
        const RECT &region
      ) const;

    private:
      struct entry_t {
        std::uint64_t token = 0u;
        bool known = false;
        std::vector<RECT> rects;
      };

      mutable std::mutex mutex_;
      std::deque<entry_t> entries_;
      std::size_t retained_rects_ = 0u;
      std::uint64_t next_token_ = 1u;
    };

    /** Query only when both snapshots belong to the same retained exact history range. */
    [[nodiscard]] ddup_damage_intersection_e query_ddup_damage_between(
      const std::optional<ddup_damage_snapshot_t> &from,
      const std::optional<ddup_damage_snapshot_t> &through,
      const RECT &region
    );

    /** Query conservative accumulated coverage only across one complete retained history range. */
    [[nodiscard]] ddup_damage_coverage_t query_ddup_damage_coverage_between(
      const std::optional<ddup_damage_snapshot_t> &from,
      const std::optional<ddup_damage_snapshot_t> &through,
      const RECT &region
    );

    enum class host_sbs_ddup_reuse_proof_e : std::uint8_t {
      none,
      content_clock,
      roi_damage,
    };

    /** Classify the pixel proof for bounded current-color reuse.
     *
     * Full-source reuse requires the established DDup content-clock identity. ROI reuse also
     * requires one continuous damage history and may bridge changed content timestamps only when
     * every committed dirty/move record stays outside the exact analysis crop.
     */
    [[nodiscard]] host_sbs_ddup_reuse_proof_e classify_host_sbs_ddup_reuse(
      const std::optional<std::chrono::steady_clock::time_point> &inferred_content,
      const std::optional<ddup_damage_snapshot_t> &inferred_damage,
      bool video_region,
      const RECT &input_region,
      const std::optional<std::chrono::steady_clock::time_point> &current_content,
      const std::optional<ddup_damage_snapshot_t> &current_damage
    );
  }  // namespace detail

  class gpu_cursor_t {
  public:
    gpu_cursor_t():
        cursor_view {0, 0, 0, 0, 0.0f, 1.0f} {};

    void set_pos(LONG topleft_x, LONG topleft_y, LONG display_width, LONG display_height, DXGI_MODE_ROTATION display_rotation, bool visible) {
      this->topleft_x = topleft_x;
      this->topleft_y = topleft_y;
      this->display_width = display_width;
      this->display_height = display_height;
      this->display_rotation = display_rotation;
      this->visible = visible;
      update_viewport();
    }

    void set_texture(LONG texture_width, LONG texture_height, texture2d_t &&texture) {
      this->texture = std::move(texture);
      this->texture_width = texture_width;
      this->texture_height = texture_height;
      update_viewport();
    }

    void update_viewport() {
      switch (display_rotation) {
        case DXGI_MODE_ROTATION_UNSPECIFIED:
        case DXGI_MODE_ROTATION_IDENTITY:
          cursor_view.TopLeftX = topleft_x;
          cursor_view.TopLeftY = topleft_y;
          cursor_view.Width = texture_width;
          cursor_view.Height = texture_height;
          break;

        case DXGI_MODE_ROTATION_ROTATE90:
          cursor_view.TopLeftX = topleft_y;
          cursor_view.TopLeftY = display_width - texture_width - topleft_x;
          cursor_view.Width = texture_height;
          cursor_view.Height = texture_width;
          break;

        case DXGI_MODE_ROTATION_ROTATE180:
          cursor_view.TopLeftX = display_width - texture_width - topleft_x;
          cursor_view.TopLeftY = display_height - texture_height - topleft_y;
          cursor_view.Width = texture_width;
          cursor_view.Height = texture_height;
          break;

        case DXGI_MODE_ROTATION_ROTATE270:
          cursor_view.TopLeftX = display_height - texture_height - topleft_y;
          cursor_view.TopLeftY = topleft_x;
          cursor_view.Width = texture_height;
          cursor_view.Height = texture_width;
          break;
      }
    }

    texture2d_t texture;
    LONG texture_width;
    LONG texture_height;

    LONG topleft_x;
    LONG topleft_y;

    LONG display_width;
    LONG display_height;
    DXGI_MODE_ROTATION display_rotation;

    shader_res_t input_res;

    D3D11_VIEWPORT cursor_view;

    bool visible;
  };

  class display_base_t: public display_t {
  public:
    int init(const ::video::config_t &config, const std::string &display_name, capture_backend_e backend);

    capture_e capture(const push_captured_image_cb_t &push_captured_image_cb, const pull_free_image_cb_t &pull_free_image_cb, bool *cursor) override;

    void set_client_frame_rate(int framerate, int framerate_x100) override;

    factory1_t factory;
    adapter_t adapter;
    output_t output;
    device_t device;
    device_ctx_t device_ctx;
    DXGI_RATIONAL display_refresh_rate;
    int display_refresh_rate_rounded;

    DXGI_MODE_ROTATION display_rotation = DXGI_MODE_ROTATION_UNSPECIFIED;
    int width_before_rotation;
    int height_before_rotation;

    // Guarded by client_frame_rate_mutex once capture() is running: a live 0x3007 video-mode
    // change republishes the cadence from the encode thread while capture is in flight.
    int client_frame_rate;
    DXGI_RATIONAL client_frame_rate_strict {};
    // Bumped by set_client_frame_rate() whenever the published cadence actually changes. The
    // capture loop watches it so it can re-derive its pacing interval without a display reinit.
    std::atomic<std::uint64_t> client_frame_rate_generation {0};
    std::mutex client_frame_rate_mutex;

    DXGI_FORMAT capture_format;
    D3D_FEATURE_LEVEL feature_level;

    std::unique_ptr<high_precision_timer> timer = create_high_precision_timer();

    typedef enum _D3DKMT_SCHEDULINGPRIORITYCLASS {
      D3DKMT_SCHEDULINGPRIORITYCLASS_IDLE,  ///< Idle priority class
      D3DKMT_SCHEDULINGPRIORITYCLASS_BELOW_NORMAL,  ///< Below normal priority class
      D3DKMT_SCHEDULINGPRIORITYCLASS_NORMAL,  ///< Normal priority class
      D3DKMT_SCHEDULINGPRIORITYCLASS_ABOVE_NORMAL,  ///< Above normal priority class
      D3DKMT_SCHEDULINGPRIORITYCLASS_HIGH,  ///< High priority class
      D3DKMT_SCHEDULINGPRIORITYCLASS_REALTIME  ///< Realtime priority class
    } D3DKMT_SCHEDULINGPRIORITYCLASS;

    typedef UINT D3DKMT_HANDLE;

    typedef struct _D3DKMT_OPENADAPTERFROMLUID {
      LUID AdapterLuid;
      D3DKMT_HANDLE hAdapter;
    } D3DKMT_OPENADAPTERFROMLUID;

    typedef struct _D3DKMT_WDDM_2_7_CAPS {
      union {
        struct
        {
          UINT HwSchSupported : 1;
          UINT HwSchEnabled : 1;
          UINT HwSchEnabledByDefault : 1;
          UINT IndependentVidPnVSyncControl : 1;
          UINT Reserved : 28;
        };

        UINT Value;
      };
    } D3DKMT_WDDM_2_7_CAPS;

    typedef struct _D3DKMT_QUERYADAPTERINFO {
      D3DKMT_HANDLE hAdapter;
      UINT Type;
      VOID *pPrivateDriverData;
      UINT PrivateDriverDataSize;
    } D3DKMT_QUERYADAPTERINFO;

    const UINT KMTQAITYPE_WDDM_2_7_CAPS = 70;

    typedef struct _D3DKMT_CLOSEADAPTER {
      D3DKMT_HANDLE hAdapter;
    } D3DKMT_CLOSEADAPTER;

    typedef NTSTATUS(WINAPI *PD3DKMTSetProcessSchedulingPriorityClass)(HANDLE, D3DKMT_SCHEDULINGPRIORITYCLASS);
    typedef NTSTATUS(WINAPI *PD3DKMTOpenAdapterFromLuid)(D3DKMT_OPENADAPTERFROMLUID *);
    typedef NTSTATUS(WINAPI *PD3DKMTQueryAdapterInfo)(D3DKMT_QUERYADAPTERINFO *);
    typedef NTSTATUS(WINAPI *PD3DKMTCloseAdapter)(D3DKMT_CLOSEADAPTER *);

    virtual bool is_hdr() override;
    virtual bool get_hdr_metadata(SS_HDR_METADATA &metadata) override;
    std::optional<float> get_sdr_white_nits();

    const char *dxgi_format_to_string(DXGI_FORMAT format);
    const char *colorspace_to_string(DXGI_COLOR_SPACE_TYPE type);
    virtual std::vector<DXGI_FORMAT> get_supported_capture_formats() = 0;

  protected:
    int get_pixel_pitch() {
      return (capture_format == DXGI_FORMAT_R16G16B16A16_FLOAT) ? 8 : 4;
    }

    virtual capture_e snapshot(const pull_free_image_cb_t &pull_free_image_cb, std::shared_ptr<platf::img_t> &img_out, std::chrono::milliseconds timeout, bool cursor_visible) = 0;
    virtual capture_e release_snapshot() = 0;
    virtual int complete_img(img_t *img, bool dummy) = 0;
  };

  /**
   * Display component for devices that use hardware encoders.
   */
  class display_vram_t: public display_base_t, public std::enable_shared_from_this<display_vram_t> {
  public:
    std::shared_ptr<img_t> alloc_img() override;
    int dummy_img(img_t *img_base) override;
    int complete_img(img_t *img_base, bool dummy) override;
    std::vector<DXGI_FORMAT> get_supported_capture_formats() override;

    bool is_codec_supported(std::string_view name, const ::video::config_t &config) override;

    std::unique_ptr<nvenc_encode_device_t> make_nvenc_encode_device(pix_fmt_e pix_fmt) override;

    std::atomic<uint32_t> next_image_id;
  };

  /**
   * Display duplicator that uses the DirectX Desktop Duplication API.
   */
  class duplication_t {
  public:
    dup_t dup;
    bool has_frame {};
    std::chrono::steady_clock::time_point last_protected_content_warning_time {};

    int init(display_base_t *display, const ::video::config_t &config);
    capture_e next_frame(DXGI_OUTDUPL_FRAME_INFO &frame_info, std::chrono::milliseconds timeout, resource_t::pointer *res_p);
    detail::ddup_damage_update_t damage_update(
      const DXGI_OUTDUPL_FRAME_INFO &frame_info,
      DXGI_MODE_ROTATION rotation,
      LONG width,
      LONG height
    );
    capture_e reset(dup_t::pointer dup_p = dup_t::pointer());
    capture_e release_frame();

    ~duplication_t();
  };

  /**
   * Display backend that uses DDAPI with a hardware encoder.
   */
  class display_ddup_vram_t: public display_vram_t {
  public:
    int init(const ::video::config_t &config, const std::string &display_name);

    capture_backend_e capture_backend() const noexcept override {
      return capture_backend_e::ddup;
    }

    capture_e snapshot(const pull_free_image_cb_t &pull_free_image_cb, std::shared_ptr<platf::img_t> &img_out, std::chrono::milliseconds timeout, bool cursor_visible) override;
    capture_e release_snapshot() override;

    duplication_t dup;
    sampler_state_t sampler_linear;

    blend_t blend_alpha;
    blend_t blend_invert;
    blend_t blend_disable;

    ps_t cursor_ps;
    vs_t cursor_vs;

    gpu_cursor_t cursor_alpha;
    gpu_cursor_t cursor_xor;

    texture2d_t old_surface_delayed_destruction;
    std::chrono::steady_clock::time_point old_surface_timestamp;
    std::variant<std::monostate, texture2d_t, std::shared_ptr<platf::img_t>> last_frame_variant;
    std::optional<std::chrono::steady_clock::time_point> last_content_timestamp;
    std::shared_ptr<detail::ddup_damage_history_t> damage_history;
    std::optional<detail::ddup_damage_snapshot_t> last_ddup_damage;
    bool damage_chain_valid = true;
  };

  /**
   * Display duplicator that uses the Windows.Graphics.Capture API.
   */
  class wgc_capture_t {
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice uwp_device {nullptr};
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem item {nullptr};
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool frame_pool {nullptr};
    winrt::Windows::Graphics::Capture::GraphicsCaptureSession capture_session {nullptr};
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame produced_frame {nullptr}, consumed_frame {nullptr};
    SRWLOCK frame_lock = SRWLOCK_INIT;
    CONDITION_VARIABLE frame_present_cv;

    void on_frame_arrived(winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool const &sender, winrt::Windows::Foundation::IInspectable const &);

  public:
    wgc_capture_t();
    ~wgc_capture_t();

    int init(display_base_t *display, const ::video::config_t &config);
    capture_e next_frame(std::chrono::milliseconds timeout, ID3D11Texture2D **out, uint64_t &out_time);
    capture_e release_frame();
    int set_cursor_visible(bool);
  };

  /**
   * Display backend that uses Windows.Graphics.Capture with a hardware encoder.
   */
  class display_wgc_vram_t: public display_vram_t {
    wgc_capture_t dup;

  public:
    int init(const ::video::config_t &config, const std::string &display_name);

    capture_backend_e capture_backend() const noexcept override {
      return capture_backend_e::wgc;
    }

    capture_e snapshot(const pull_free_image_cb_t &pull_free_image_cb, std::shared_ptr<platf::img_t> &img_out, std::chrono::milliseconds timeout, bool cursor_visible) override;
    capture_e release_snapshot() override;
  };

  /** Return whether Windows.Graphics.Capture can create an item for this output. */
  bool test_wgc_capture(output_t &output);

  struct local_presenter_cursor_clip_t {
    RECT previous_clip {};
    RECT owned_clip {};
    bool clip_saved = false;
    bool previous_clip_unbounded = false;
    bool owns_clip = false;
    bool clip_yielded = false;
    bool clip_unavailable_logged = false;
    unsigned retry_failures = 0;
    std::chrono::steady_clock::time_point retry_after {};

    bool restore();
    ~local_presenter_cursor_clip_t();
  };

  struct local_presenter_config_t {
    std::string source_display_name;
    RECT target_rect {};
    LUID target_adapter_id {};
    int target_refresh_millihz = 60000;
    bool hdr = false;
    int sbs_mode = ::video::SBS_OFF;
    config::video_t::sbs_t sbs_config {};
    std::shared_ptr<::video::capture_backend_failover_t> capture_failover;

    struct target_t {
      std::mutex mutex;
      RECT rect {};
      std::string display_name;
      // DisplayConfig paths are stable across the \\.\DISPLAYn renumbering that accompanies
      // physical 2D/SBS and Advanced Color transitions. The topology controller seeds these
      // before the presenter opens either volatile GDI name.
      std::wstring source_device_path;
      std::wstring target_device_path;
    };

    std::shared_ptr<target_t> live_target;
    std::shared_ptr<std::atomic<std::uint64_t>> presented_frames;
    std::shared_ptr<local_presenter_cursor_clip_t> cursor_clip;
  };

  enum class local_presenter_result_e {
    stopped,
    reinit,
    error,
  };

  /** Reconcile the session-owned cursor boundary while presenter resources are paused. */
  void refresh_local_presenter_pointer_isolation(
    const std::string &source_display_name,
    const std::shared_ptr<local_presenter_config_t::target_t> &live_target,
    const std::shared_ptr<local_presenter_cursor_clip_t> &cursor_clip
  );

  /**
   * Capture a local display and present it to a borderless window on another display.
   * SBS mode uses the same matched-frame depth and warp implementation as the encoder path.
   */
  local_presenter_result_e run_local_presenter(const local_presenter_config_t &config, std::stop_token stop_token);
}  // namespace platf::dxgi
