/**
 * @file src/nvenc/nvenc_base.h
 * @brief Declarations for abstract platform-agnostic base of standalone NVENC encoder.
 */
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

// lib includes
#include <ffnvcodec/nvEncodeAPI.h>

// local includes
#include "nvenc_colorspace.h"
#include "nvenc_config.h"
#include "nvenc_encoded_frame.h"
#include "src/logging.h"
#include "src/video.h"

/**
 * @brief Standalone NVENC encoder
 */
namespace nvenc {

  enum class nvenc_event_wait_status {
    ready,
    timeout,
    failed
  };

  struct nvenc_event_wait_result {
    nvenc_event_wait_status status = nvenc_event_wait_status::failed;
    std::uint32_t native_wait_result = 0;
    // Captured immediately on a native wait failure; never reuse a stale timeout error.
    std::uint32_t native_error = 0;
    // Empty when no device was queried, zero for a healthy D3D device, negative on removal.
    std::optional<std::int32_t> device_removed_reason;
  };

  /** Whether the GPU finished the work that wrote a submitted picture's input surface. */
  enum class input_producer_state {
    unknown,
    pending,
    complete
  };

  /** Last per-codec width capability reported by the active NVENC driver probe. */
  std::optional<int> max_encode_width_for_codec(int video_format);

  /** Last per-codec height capability reported by the active NVENC driver probe. */
  std::optional<int> max_encode_height_for_codec(int video_format);

  struct nvenc_hdr_metadata_t {
    std::optional<MASTERING_DISPLAY_INFO> mastering_display;
    std::optional<CONTENT_LIGHT_LEVEL> content_light_level;
  };

  /**
   * Convert Sunshine's wire metadata to the codec-specific integer units expected by NVENC.
   * HEVC and AV1 use different mastering-display denominators.
   */
  nvenc_hdr_metadata_t hdr_metadata_from_sunshine(
    const std::optional<SS_HDR_METADATA> &metadata,
    int video_format
  );

  /**
   * @brief Abstract platform-agnostic base of standalone NVENC encoder.
   *        Derived classes perform platform-specific operations.
   */
  class nvenc_base {
  public:
    /**
     * @param device_type Underlying device type used by derived class.
     */
    explicit nvenc_base(NV_ENC_DEVICE_TYPE device_type);
    virtual ~nvenc_base();

    nvenc_base(const nvenc_base &) = delete;
    nvenc_base &operator=(const nvenc_base &) = delete;

    /**
     * @brief Create the encoder.
     * @param config NVENC encoder configuration.
     * @param client_config Stream configuration requested by the client.
     * @param colorspace YUV colorspace.
     * @param buffer_format Platform-agnostic input surface format.
     * @param hdr_metadata Mastering-display and optional content-light metadata.
     * @return `true` on success, `false` on error
     */
    bool create_encoder(
      const nvenc_config &config,
      const video::config_t &client_config,
      const nvenc_colorspace_t &colorspace,
      NV_ENC_BUFFER_FORMAT buffer_format,
      const std::optional<SS_HDR_METADATA> &hdr_metadata
    );

    /**
     * @brief Destroy the encoder.
     *        Derived classes classes call it in the destructor.
     */
    void destroy_encoder();

    /** Pictures an encoder can have in flight at once (input surface, bitstream and event each). */
    static constexpr unsigned max_pipeline_depth = 2;

    /**
     * Pictures that may be submitted before the oldest is retrieved, fixed by create_encoder():
     * `client_config`'s video::nvenc_pipeline_depth() when the encoder runs asynchronously and
     * every picture got its own completion event, else 1.
     */
    [[nodiscard]] unsigned pipeline_depth() const noexcept {
      return pipeline_depth_;
    }

    /**
     * @brief Encode the next frame using platform-specific input surface: submit_frame() and then
     *        retrieve_frame(), on the calling thread.
     * @param frame_index Frame index that uniquely identifies the frame.
     *        Afterwards serves as parameter for `invalidate_ref_frames()`.
     *        No restrictions on the first frame index, but later frame indexes must be subsequent.
     * @param force_idr Whether to encode frame as forced IDR.
     * @param frame_buffer Recyclable storage for the copied bitstream.
     * @return Encoded frame.
     */
    nvenc_encoded_frame encode_frame(
      uint64_t frame_index,
      bool force_idr,
      std::vector<std::uint8_t> frame_buffer = {}
    );

    /**
     * Submit the current input surface as the next picture without waiting for it. With a pipeline
     * depth above 1 the surface is first copied into the picture's own input, so the caller may
     * convert the next frame while this one encodes. Call from one thread (the encode thread).
     * @return `false` on failure, or while `pipeline_depth()` pictures are still in flight.
     */
    bool submit_frame(uint64_t frame_index, bool force_idr);

    /**
     * Wait for the oldest submitted picture (the 250 ms encoder budget, from its input's completion
     * when the GPU was still producing it) and copy its bitstream. Pictures are retrieved strictly
     * in submission order. May run on a second thread concurrently with submit_frame(), as NVENC's
     * asynchronous mode intends; never concurrently with itself or any other call.
     * @return Encoded frame, or an empty one on failure (new encoders then wait for teardown).
     */
    nvenc_encoded_frame retrieve_frame(std::vector<std::uint8_t> frame_buffer = {});

    /** Submitted pictures not yet retrieved. */
    [[nodiscard]] unsigned frames_in_flight() const noexcept {
      return in_flight_.load(std::memory_order_acquire);
    }

    /**
     * Reconfigure only the active encoder's CBR bitrate and matching VBV size.
     *
     * The caller must serialize this between encode_frame() calls, with no picture in flight. A
     * failure leaves the cached configuration unchanged so the caller can fall back to rebuilding
     * the session. This path deliberately neither resets encoder state nor requests an IDR.
     */
    bool reconfigure_bitrate(int bitrate_kbps);

    /**
     * @brief Perform reference frame invalidation (RFI) procedure. Call with no picture in flight,
     *        so that the range extends to the last submitted picture and the next submitted one is
     *        the first encoded after it.
     * @param first_frame First frame index of the invalidation range.
     * @param last_frame Last frame index of the invalidation range.
     * @return `true` on success, `false` on error.
     *         After error next frame must be encoded with `force_idr = true`.
     */
    bool invalidate_ref_frames(uint64_t first_frame, uint64_t last_frame);

    /** Diagnostics: CPU time the last encode_frame() spent submitting its picture and waiting for
     *  its completion. Zero while diagnostics are disabled and for a step the frame did not reach.
     *  Only for a caller that submits and retrieves on one thread. */
    struct frame_timing_t {
      std::chrono::nanoseconds submit {};
      std::chrono::nanoseconds completion_wait {};
    };

    [[nodiscard]] frame_timing_t last_frame_timing() const noexcept {
      return {last_submit_, last_completion_wait_};
    }

    /** Diagnostics: CPU time of the last submit_frame()'s picture submission; submitting thread. */
    [[nodiscard]] std::chrono::nanoseconds last_submit_timing() const noexcept {
      return last_submit_;
    }

  protected:
    /**
     * @brief Required. Used for loading NvEnc library and setting `nvenc` variable with `NvEncodeAPICreateInstance()`.
     *        Called during `create_encoder()` if `nvenc` variable is not initialized.
     * @return `true` on success, `false` on error
     */
    virtual bool init_library() = 0;

    /**
     * @brief Required. Used for creating outside-facing input surface and registering with
     *        `nvenc->nvEncRegisterResource()` the input of each picture slot below
     *        `pipeline_depth()`, setting `registered_input_buffers[slot]`. With a depth of 1 that
     *        is the outside-facing surface itself. Called during `create_encoder()`.
     * @return `true` on success, `false` on error
     */
    virtual bool create_and_register_input_buffer() = 0;

    /**
     * @brief Required for a depth above 1, where it alone is called: fill picture `slot`'s
     *        registered input from the outside-facing surface (a GPU copy) before it is mapped.
     *        Submitting thread only.
     */
    virtual void prepare_input(unsigned slot) {}

    /**
     * @brief Optional. Completion event of picture slot `slot` (1 and up; slot 0 uses
     *        `async_event_handle`), owned by the derived class until `release_async_event()`.
     *        Without one the encoder keeps a pipeline depth of 1.
     */
    virtual void *create_completion_event(unsigned slot) {
      return nullptr;
    }

    /**
     * @brief Optional. Override if you want to create encoder in async mode.
     *        In this case must also set `async_event_handle` variable.
     * @param event A picture slot's completion event.
     * @param timeout_ms Wait timeout in milliseconds
     * @return Completion, elapsed timeout, or a wait failure with its native diagnostics.
     */
    virtual nvenc_event_wait_result wait_for_async_event(void *event, uint32_t timeout_ms) {
      return {};
    }

    /**
     * @brief Optional. Mark the end of the GPU work that writes picture slot `slot`'s input,
     *        immediately before its asynchronous picture is submitted. Submitting thread only;
     *        never wait.
     */
    virtual void mark_input_producer_end(unsigned slot) {}

    /**
     * @brief Optional. Non-blocking check of slot `slot`'s point marked above, from the retrieving
     *        thread. Polled only after a slow picture, to separate a late input producer from a
     *        slow encoder.
     */
    virtual input_producer_state poll_input_producer(unsigned slot) {
      return input_producer_state::unknown;
    }

    /** Monotonic clock shared by the frame's initial wait and its bounded grace period. */
    virtual std::chrono::steady_clock::time_point async_wait_clock_now() const {
      return std::chrono::steady_clock::now();
    }

    /** Teardown-only EOS event, distinct from the auto-reset picture-completion event. */
    virtual void *create_flush_event() {
      return nullptr;
    }

    virtual nvenc_event_wait_result wait_for_flush_event(uint32_t timeout_ms) {
      return {};
    }

    /**
     * @brief Close the platform-owned async event and clear `async_event_handle`.
     *        Derived classes that install an async event must implement this operation and call it
     *        from their destructor. The base also calls it when the selected codec reports that
     *        asynchronous encoding is unsupported.
     */
    virtual void release_async_event() = 0;

    /** Apply the driver's asynchronous-encode capability to the optional event. */
    void apply_async_encode_capability(int capability);

    bool nvenc_failed(NVENCSTATUS status);

    const NV_ENC_DEVICE_TYPE device_type;

    void *encoder = nullptr;

    struct {
      uint32_t width = 0;
      uint32_t height = 0;
      NV_ENC_BUFFER_FORMAT buffer_format = NV_ENC_BUFFER_FORMAT_UNDEFINED;
      uint32_t ref_frames_in_dpb = 0;
      bool rfi = false;
      int video_format = -1;
    } encoder_params;

    nvenc_hdr_metadata_t hdr_metadata;
    // Per thread: a retrieving thread reports its own failures while the submitting one runs.
    static thread_local std::string last_nvenc_error_string;

    // Derived classes set these variables
    void *device = nullptr;  ///< Platform-specific handle of encoding device.
                             ///< Should be set in constructor or `init_library()`.
    std::shared_ptr<NV_ENCODE_API_FUNCTION_LIST> nvenc;  ///< Function pointers list produced by `NvEncodeAPICreateInstance()`.
                                                         ///< Should be set in `init_library()`.
    std::array<NV_ENC_REGISTERED_PTR, max_pipeline_depth> registered_input_buffers {};  ///< Each picture slot's input registered with `NvEncRegisterResource()`.
                                                                                        ///< Should be set in `create_and_register_input_buffer()`.
    void *async_event_handle = nullptr;  ///< (optional) Platform-specific handle of event object event; slot 0's completion event.
                                         ///< Can be set in constructor or `init_library()`, must override `wait_for_async_event()`.

  private:
    enum class input_phase_t {
      unmapped,
      mapped,  ///< Mapped and not in flight: before its submission or after its retrieval.
      submitted,
      completion_seen,
      locked
    };

    // One picture in flight. The submitting thread owns an unmapped or mapped slot and the
    // retrieving thread a submitted one; `phase` hands it over (release/acquire).
    struct picture_slot_t {
      NV_ENC_OUTPUT_PTR output_bitstream = nullptr;
      void *completion_event = nullptr;
      bool event_registered = false;
      NV_ENC_INPUT_PTR mapped_input = nullptr;
      std::atomic<input_phase_t> phase {input_phase_t::unmapped};
      uint64_t frame_index = 0;
      bool after_ref_frame_invalidation = false;
    };

    std::array<picture_slot_t, max_pipeline_depth> slots;
    unsigned pipeline_depth_ = 1;
    unsigned next_submit_slot = 0;  ///< Submitting thread.
    unsigned next_retrieve_slot = 0;  ///< Retrieving thread.
    std::atomic<unsigned> in_flight_ {0};
    void *flush_event_handle = nullptr;
    bool flush_event_registered = false;
    bool flush_submitted = false;
    bool flush_completed = false;
    bool encoder_used = false;
    std::atomic<bool> cleanup_blocked {false};  ///< Set by either thread; cleared by teardown.
    bool teardown_wait_failure_logged = false;

    void block_new_encoders();
    bool cleanup_succeeded(NVENCSTATUS status, const char *operation);
    bool unlock_slot(picture_slot_t &slot);
    bool unmap_slot(picture_slot_t &slot);
    bool release_completed_input(picture_slot_t &slot);
    bool submit_flush();
    bool drain_slot(picture_slot_t &slot);
    bool drain_input();
    bool release_encoder_resources();
    bool teardown_wait_ready(const nvenc_event_wait_result &result, const char *operation);
    bool wait_for_frame_completion(unsigned slot_index);

    struct stage_diagnostics_t {
      logging::time_delta_periodic_logger input_map {info, "Video NVENC: input map CPU"};
      logging::time_delta_periodic_logger submit {info, "Video NVENC: picture submission CPU"};
      logging::time_delta_periodic_logger completion_wait {info, "Video NVENC: asynchronous completion wait"};
      logging::time_delta_periodic_logger bitstream_lock {info, "Video NVENC: bitstream lock CPU"};
      logging::time_delta_periodic_logger bitstream_copy {info, "Video NVENC: bitstream copy CPU"};
      logging::time_delta_periodic_logger bitstream_unlock {info, "Video NVENC: bitstream unlock CPU"};
      logging::time_delta_periodic_logger input_unmap {info, "Video NVENC: input unmap CPU"};
    };

    std::optional<stage_diagnostics_t> stage_diagnostics;
    std::chrono::nanoseconds last_submit_ {};  ///< Submitting thread.
    std::chrono::nanoseconds last_completion_wait_ {};  ///< Retrieving thread.

    struct {
      uint64_t last_encoded_frame_index = 0;
      bool rfi_needs_confirmation = false;
      std::pair<uint64_t, uint64_t> last_rfi_range;
      bool bitrate_reconfiguration_supported = false;
      bool custom_vbv_supported = false;
      int framerate = 0;
      int vbv_percentage_increase = 0;
      NV_ENC_INITIALIZE_PARAMS initialize_params {};
      NV_ENC_CONFIG encode_config {};
      logging::min_max_avg_periodic_logger<double> frame_size_logger = {info, "NvEnc: encoded frame sizes in kB", ""};
      logging::min_max_avg_periodic_logger<std::uint32_t> frame_qp_logger = {info, "NvEnc: frame average QP", ""};
      logging::min_max_avg_periodic_logger<std::uint32_t> frame_satd_logger = {info, "NvEnc: frame SATD", ""};
    } encoder_state;
  };

}  // namespace nvenc
