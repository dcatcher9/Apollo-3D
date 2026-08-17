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
#include <string_view>
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
#include "src/generated/sbs_adaptive_state_contract.h"
#include "src/host_sbs_adaptive_submission.h"
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
    inline constexpr auto host_sbs_gpu_observation_owner_max_age =
      std::chrono::microseconds {
        models::gpu_adaptive_max_infer_owner_observation_age_us
      };
    // Same-frame completion polling is allowed only inside encode-loop cadence slack. The
    // downstream reserve covers completed-depth postprocess, SBS warp/output, and NVENC submit;
    // late/high-rate frames therefore remain on the ordinary nonblocking path.
    // Low-rate streams have more real cadence slack, so let the scheduler-owned target expand the
    // wait beyond the original 3 ms tail. The absolute cap bounds added presentation latency and
    // yielded-query CPU time even when the next cadence target is far away.
    inline constexpr auto host_sbs_same_frame_poll_max_wait =
      std::chrono::milliseconds {8};
    inline constexpr auto host_sbs_same_frame_poll_downstream_reserve =
      std::chrono::milliseconds {3};
    inline constexpr auto host_sbs_same_frame_poll_min_budget =
      std::chrono::microseconds {250};
    inline constexpr std::uint32_t host_sbs_same_frame_poll_max_queries = 4096u;
    struct host_sbs_same_frame_poll_plan_t {
      std::chrono::steady_clock::time_point deadline {};
      std::chrono::steady_clock::duration budget {};
      bool eligible = false;
      bool limited_by_hard_cap = false;
    };

    enum class host_sbs_same_frame_poll_hit_bucket_e : std::uint8_t {
      within_2_ms,
      between_2_and_2_5_ms,
      between_2_5_and_3_ms,
      over_3_ms,
    };

    /** Classify a ready repeated-wait result without rounding its steady-clock duration. */
    [[nodiscard]] constexpr host_sbs_same_frame_poll_hit_bucket_e
    host_sbs_same_frame_poll_hit_bucket(
      const std::chrono::steady_clock::duration wait_duration
    ) noexcept {
      if (wait_duration <= std::chrono::milliseconds {2}) {
        return host_sbs_same_frame_poll_hit_bucket_e::within_2_ms;
      }
      if (wait_duration <= std::chrono::microseconds {2500}) {
        return host_sbs_same_frame_poll_hit_bucket_e::between_2_and_2_5_ms;
      }
      if (wait_duration <= std::chrono::milliseconds {3}) {
        return host_sbs_same_frame_poll_hit_bucket_e::between_2_5_and_3_ms;
      }
      // With cadence-adaptive polling this is the intentional extended readiness tail. The
      // separate planned-budget and timeout-limit telemetry distinguishes useful slack from a
      // hard-cap or cadence-bound miss.
      return host_sbs_same_frame_poll_hit_bucket_e::over_3_ms;
    }

    enum class host_sbs_same_frame_completion_e : std::uint8_t {
      keep_pending,
      adopt_exact,
      discard_ready,
    };

    /** Decide slot ownership after polling without inspecting mutable estimator resources. */
    [[nodiscard]] constexpr host_sbs_same_frame_completion_e
    host_sbs_same_frame_completion(
      const bool ready,
      const bool completed_frame_valid,
      const std::uint64_t completed_frame_id,
      const std::uint64_t candidate_frame_id
    ) noexcept {
      if (!ready) {
        return host_sbs_same_frame_completion_e::keep_pending;
      }
      return completed_frame_valid && completed_frame_id == candidate_frame_id ?
               host_sbs_same_frame_completion_e::adopt_exact :
               host_sbs_same_frame_completion_e::discard_ready;
    }

    /** Partition every newly submitted same-frame query into exactly one diagnostic outcome. */
    enum class host_sbs_same_frame_poll_outcome_e : std::uint8_t {
      immediate_hit,
      wait_hit,
      cadence_ineligible_busy,
      eligible_timeout,
      ready_failure,
      wait_unavailable_busy,
    };

    [[nodiscard]] constexpr host_sbs_same_frame_poll_outcome_e
    host_sbs_same_frame_poll_outcome(
      const bool plan_eligible,
      const bool wait_attempted,
      const bool timed_out,
      const host_sbs_same_frame_completion_e completion
    ) noexcept {
      if (completion == host_sbs_same_frame_completion_e::adopt_exact) {
        return wait_attempted ?
                 host_sbs_same_frame_poll_outcome_e::wait_hit :
                 host_sbs_same_frame_poll_outcome_e::immediate_hit;
      }
      if (completion == host_sbs_same_frame_completion_e::discard_ready) {
        return host_sbs_same_frame_poll_outcome_e::ready_failure;
      }
      if (!plan_eligible) {
        return host_sbs_same_frame_poll_outcome_e::cadence_ineligible_busy;
      }
      return timed_out ?
               host_sbs_same_frame_poll_outcome_e::eligible_timeout :
               host_sbs_same_frame_poll_outcome_e::wait_unavailable_busy;
    }

    /** Bound a newly submitted matched-frame transaction query by the encode cadence scheduler.
     * Capture/content timestamps are deliberately absent: they identify pixels but do not own the
     * encode deadline. An unavailable, late, or too-small budget fails open to nonblocking output.
     */
    [[nodiscard]] constexpr host_sbs_same_frame_poll_plan_t
    host_sbs_same_frame_poll_plan(
      const bool transaction_enqueued,
      const bool snapshot_debug_inputs,
      const std::optional<std::chrono::steady_clock::time_point> next_encode_target,
      const std::chrono::steady_clock::time_point now
    ) noexcept {
      if (!transaction_enqueued || snapshot_debug_inputs || !next_encode_target) {
        return {};
      }
      const auto cadence_deadline =
        *next_encode_target - host_sbs_same_frame_poll_downstream_reserve;
      if (cadence_deadline <= now ||
          cadence_deadline - now < host_sbs_same_frame_poll_min_budget) {
        return {};
      }
      const auto hard_deadline = now + host_sbs_same_frame_poll_max_wait;
      const bool limited_by_hard_cap = hard_deadline <= cadence_deadline;
      const auto deadline = limited_by_hard_cap ? hard_deadline : cadence_deadline;
      return {
        deadline,
        deadline - now,
        true,
        limited_by_hard_cap,
      };
    }

    enum class host_sbs_depth_reuse_kind_e : std::uint8_t {
      none,
      exact_content,
      exact_roi_damage,
    };

    /** GPU-undecided admission has one anti-chaining latch outside opaque follow-ups. */
    enum class host_sbs_approximate_reuse_provider_e : std::uint8_t {
      none,
      gpu_undecided,
    };

    enum class host_sbs_depth_reuse_refresh_e : std::uint8_t {
      none,
      bounded_content,
    };

    /** CPU-side disposition using host metadata only.
     *
     * Exact DDup proof may reuse the completed cache. An otherwise eligible changed frame is
     * deliberately undecided so the GPU can select inference or reuse without a decision
     * readback. Every other frame sends a force-infer request through the same wrapper.
     */
    enum class host_sbs_depth_admission_e : std::uint8_t {
      reuse_cached,
      force_infer,
      gpu_undecided,
    };

    [[nodiscard]] constexpr host_sbs_depth_admission_e host_sbs_depth_admission(
      const bool cached_reuse_authorized,
      const bool gpu_undecided_eligible,
      const bool must_observe,
      const bool opaque_followup_authorized = false
    ) noexcept {
      if (must_observe) {
        // The observation barrier continues to block every host-owned cache path. It may admit
        // only the immediately-prior opaque follow-up: the authenticated device history owner
        // compares after infer and forces inference after reuse/invalid without branch readback.
        return opaque_followup_authorized && gpu_undecided_eligible ?
                 host_sbs_depth_admission_e::gpu_undecided :
                 host_sbs_depth_admission_e::force_infer;
      }
      if (cached_reuse_authorized) {
        return host_sbs_depth_admission_e::reuse_cached;
      }
      return gpu_undecided_eligible ? host_sbs_depth_admission_e::gpu_undecided :
                                      host_sbs_depth_admission_e::force_infer;
    }

    [[nodiscard]] constexpr bool host_sbs_gpu_followup_fresh(
      const std::uint64_t anchor_frame_id,
      const std::uint64_t barrier_frame_id,
      const std::chrono::steady_clock::time_point enqueued_at,
      const std::chrono::steady_clock::time_point now
    ) noexcept {
      return anchor_frame_id != 0u && anchor_frame_id == barrier_frame_id &&
             enqueued_at.time_since_epoch().count() != 0 &&
             now.time_since_epoch().count() != 0 && now >= enqueued_at &&
             now - enqueued_at < host_sbs_gpu_observation_owner_max_age;
    }

    /** Live alias of the shared production/offline conditional transaction policy. */
    using host_sbs_gpu_observation_barrier_t =
      models::gpu_adaptive_transaction_policy_t;

    /** One current-color/cache/render authority regardless of proof acquisition path. */
    struct host_sbs_depth_reuse_authorization_t {
      host_sbs_depth_reuse_kind_e kind = host_sbs_depth_reuse_kind_e::none;
      host_sbs_depth_reuse_refresh_e refresh =
        host_sbs_depth_reuse_refresh_e::none;
      std::uint64_t baseline_frame_id = 0u;
      std::uint64_t current_frame_id = 0u;
      bool ocr_safe = false;

      [[nodiscard]] constexpr bool valid() const noexcept {
        if (
          kind == host_sbs_depth_reuse_kind_e::none || !ocr_safe ||
          baseline_frame_id == 0u || current_frame_id <= baseline_frame_id
        ) {
          return false;
        }
        switch (kind) {
          case host_sbs_depth_reuse_kind_e::none:
            return false;
          case host_sbs_depth_reuse_kind_e::exact_content:
          case host_sbs_depth_reuse_kind_e::exact_roi_damage:
            return refresh == host_sbs_depth_reuse_refresh_e::bounded_content;
        }
        return false;
      }
    };

    [[nodiscard]] constexpr host_sbs_depth_reuse_authorization_t
    make_host_sbs_depth_reuse_authorization(
      const host_sbs_depth_reuse_kind_e kind,
      const std::uint64_t baseline_frame_id,
      const std::uint64_t current_frame_id,
      const bool ocr_safe
    ) noexcept {
      host_sbs_depth_reuse_authorization_t result {
        .kind = kind,
        .baseline_frame_id = baseline_frame_id,
        .current_frame_id = current_frame_id,
        .ocr_safe = ocr_safe,
      };
      switch (kind) {
        case host_sbs_depth_reuse_kind_e::none:
          break;
        case host_sbs_depth_reuse_kind_e::exact_content:
        case host_sbs_depth_reuse_kind_e::exact_roi_damage:
          result.refresh = host_sbs_depth_reuse_refresh_e::bounded_content;
          break;
      }
      return result.valid() ? result : host_sbs_depth_reuse_authorization_t {};
    }

    class ddup_damage_history_t;

    /** Display-owned complete-DDup candidate retained across the private-copy route recheck. */
    struct host_sbs_gpu_undecided_candidate_t {
      std::uint64_t baseline_frame_id = 0u;
      std::uint64_t current_frame_id = 0u;
      const ddup_damage_history_t *damage_history = nullptr;
      std::uint64_t baseline_damage_token = 0u;
      std::uint64_t current_damage_token = 0u;
      bool damage_history_complete = false;
      bool opaque_followup = false;

      [[nodiscard]] constexpr bool valid() const noexcept {
        return baseline_frame_id != 0u && current_frame_id > baseline_frame_id &&
               damage_history != nullptr && baseline_damage_token != 0u &&
               current_damage_token > baseline_damage_token &&
               damage_history_complete;
      }
    };

    /** Host-owned approximate providers cannot chain without an intervening known inference.
     * Device-authenticated opaque follow-ups are admitted separately under the barrier contract.
     */
    [[nodiscard]] constexpr bool host_sbs_approximate_reuse_provider_allowed(
      const host_sbs_approximate_reuse_provider_e prior_since_enqueue,
      const host_sbs_approximate_reuse_provider_e candidate,
      const bool adaptive_refresh_required
    ) noexcept {
      if (prior_since_enqueue != host_sbs_approximate_reuse_provider_e::none) {
        return false;
      }
      return !adaptive_refresh_required &&
             candidate != host_sbs_approximate_reuse_provider_e::none;
    }

    [[nodiscard]] constexpr bool host_sbs_cached_geometry_render_allowed(
      const bool dedup_gate_open,
      const bool renderer_authenticated,
      const host_sbs_depth_reuse_authorization_t &authorization,
      const std::uint64_t cached_frame_id,
      const std::uint64_t current_frame_id,
      const bool cached_resources_complete
    ) noexcept {
      return dedup_gate_open && renderer_authenticated && authorization.valid() &&
             authorization.baseline_frame_id == cached_frame_id &&
             authorization.current_frame_id == current_frame_id &&
             cached_resources_complete;
    }

    /** Retain authenticated V2 lineage independently of current-frame reuse authorization. */
    [[nodiscard]] constexpr bool host_sbs_latest_v2_completion_retention_allowed(
      const bool renderer_authenticated,
      const bool result_authenticated,
      const bool completion_route_matches_current,
      const bool known_force_infer_completion
    ) noexcept {
      return renderer_authenticated && result_authenticated &&
             completion_route_matches_current && known_force_infer_completion;
    }

    /** Reset retained lineage only when its resource aliases or authority route are invalid. */
    [[nodiscard]] constexpr bool host_sbs_latest_v2_lineage_reset_required(
      const bool lineage_authenticated,
      const bool route_matches_current,
      const bool snapshot_debug_inputs,
      const bool authority_reprocess_pending,
      const bool producer_terminal
    ) noexcept {
      return lineage_authenticated &&
             (!route_matches_current || snapshot_debug_inputs || authority_reprocess_pending ||
              producer_terminal);
    }

    /** Cache-independent live route fingerprint for invalidating predictive evidence. */
    struct host_sbs_adaptive_motion_route_epoch_t {
      std::uint32_t source_width = 0u;
      std::uint32_t source_height = 0u;
      std::uint32_t mip_levels = 0u;
      std::uint32_t array_size = 0u;
      std::uint32_t source_format = 0u;
      std::uint32_t sample_count = 0u;
      std::uint32_t sample_quality = 0u;
      std::uint32_t input_color_space = 0u;
      std::uint64_t root_authority_generation = 0u;
      std::uint64_t region_authority_generation = 0u;
      std::uint64_t browser_authority_epoch = 0u;
      bool interactive_move_size = false;

      bool operator==(const host_sbs_adaptive_motion_route_epoch_t &) const = default;
    };

    class host_sbs_adaptive_motion_route_state_t {
    public:
      constexpr void reset() noexcept {
        observed_.reset();
      }

      /** Returns true only when an already-observed live route changes. */
      [[nodiscard]] constexpr bool observe(
        const host_sbs_adaptive_motion_route_epoch_t &current
      ) noexcept {
        const bool changed = observed_ && *observed_ != current;
        observed_ = current;
        return changed;
      }

    private:
      std::optional<host_sbs_adaptive_motion_route_epoch_t> observed_;
    };

    enum class host_sbs_adaptive_hold_decision_e : std::uint8_t {
      infer,
      hold_candidate,
      hold_same_identity,
    };

    /** Own initial-candidate cadence for GPU near-identical reuse.
     *
     * A candidate consumes the host arm. Device-authenticated opaque follow-ups are governed by
     * their separate route/age/owner contract; a real observation must be enqueued before another
     * initial host candidate may be armed.
     */
    class host_sbs_adaptive_hold_cadence_t {
    public:
      constexpr void reset() noexcept {
        armed_ = false;
        refresh_required_ = false;
        last_enqueued_identity_.reset();
        refresh_identity_.reset();
        last_enqueued_at_ = {};
      }

      constexpr void record_successful_enqueue(
        const std::optional<std::chrono::steady_clock::time_point> &identity,
        const std::chrono::steady_clock::time_point enqueued_at
      ) noexcept {
        armed_ = identity.has_value() && enqueued_at.time_since_epoch().count() != 0;
        refresh_required_ = false;
        last_enqueued_identity_ = identity;
        refresh_identity_.reset();
        last_enqueued_at_ = enqueued_at;
      }

      [[nodiscard]] constexpr host_sbs_adaptive_hold_decision_e observe_changed(
        const std::optional<std::chrono::steady_clock::time_point> &identity,
        const bool candidate_eligible,
        const std::chrono::steady_clock::time_point now
      ) noexcept {
        if (!identity || now.time_since_epoch().count() == 0) {
          reset();
          return host_sbs_adaptive_hold_decision_e::infer;
        }
        if (last_enqueued_identity_ && *identity == *last_enqueued_identity_) {
          return host_sbs_adaptive_hold_decision_e::infer;
        }
        if (refresh_required_) {
          const bool fresh = last_enqueued_at_.time_since_epoch().count() != 0 &&
                             now >= last_enqueued_at_ &&
                             now - last_enqueued_at_ <
                               host_sbs_gpu_observation_owner_max_age;
          if (refresh_identity_ && *identity == *refresh_identity_ && fresh) {
            return host_sbs_adaptive_hold_decision_e::hold_same_identity;
          }
          return host_sbs_adaptive_hold_decision_e::infer;
        }
        const bool fresh = last_enqueued_at_.time_since_epoch().count() != 0 &&
                           now >= last_enqueued_at_ &&
                           now - last_enqueued_at_ <
                             host_sbs_gpu_observation_owner_max_age;
        if (armed_ && candidate_eligible && fresh) {
          armed_ = false;
          refresh_required_ = true;
          refresh_identity_ = identity;
          return host_sbs_adaptive_hold_decision_e::hold_candidate;
        }
        armed_ = false;
        return host_sbs_adaptive_hold_decision_e::infer;
      }

      [[nodiscard]] constexpr bool refresh_required() const noexcept {
        return refresh_required_;
      }

      /** Revalidate the consumed candidate immediately before GPU submission.
       *
       * Private frame copy and live-authority checks happen after `observe_changed()`. A delayed
       * submit must not extend the shared observation-owner budget merely because the arm was
       * consumed while it was still fresh.
       */
      [[nodiscard]] constexpr bool hold_candidate_still_fresh(
        const std::optional<std::chrono::steady_clock::time_point> &identity,
        const std::chrono::steady_clock::time_point now
      ) const noexcept {
        return refresh_required_ && identity && refresh_identity_ &&
               *identity == *refresh_identity_ &&
               last_enqueued_at_.time_since_epoch().count() != 0 &&
               now >= last_enqueued_at_ &&
               now - last_enqueued_at_ < host_sbs_gpu_observation_owner_max_age;
      }

    private:
      bool armed_ = false;
      bool refresh_required_ = false;
      std::optional<std::chrono::steady_clock::time_point> last_enqueued_identity_;
      std::optional<std::chrono::steady_clock::time_point> refresh_identity_;
      std::chrono::steady_clock::time_point last_enqueued_at_ {};
    };

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
      // A lower bound on metadata coverage: unlike the saturated sum above, overlapping dirty
      // rectangles and a move's source/destination cannot inflate this value.
      std::uint64_t max_single_intersection_area = 0u;
    };

    /** Complete DDup history is admission evidence, never a similarity verdict.
     *
     * Every authority-valid changed frame may reach the device detector. Localized/broad shape is
     * deliberately absent: the authenticated current-vs-history tensor comparison owns reuse.
     */
    [[nodiscard]] constexpr bool host_sbs_adaptive_motion_damage_candidate(
      const ddup_damage_coverage_t &coverage
    ) noexcept {
      return coverage.known && coverage.region_area != 0u;
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
        const RECT &region,
        bool collect_max_single_intersection = false
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
      const RECT &region,
      bool collect_max_single_intersection = false
    );

    enum class host_sbs_ddup_reuse_proof_e : std::uint8_t {
      none,
      content_clock,
      roi_damage,
    };

    /** Convert cache proof acquisition into the shared render authorization. Exact proof wins. */
    [[nodiscard]] constexpr host_sbs_depth_reuse_authorization_t
    select_host_sbs_cached_depth_reuse(
      const host_sbs_ddup_reuse_proof_e exact_proof,
      const std::uint64_t baseline_frame_id,
      const std::uint64_t current_frame_id
    ) noexcept {
      switch (exact_proof) {
        case host_sbs_ddup_reuse_proof_e::content_clock:
          return make_host_sbs_depth_reuse_authorization(
            host_sbs_depth_reuse_kind_e::exact_content,
            baseline_frame_id,
            current_frame_id,
            true
          );
        case host_sbs_ddup_reuse_proof_e::roi_damage:
          return make_host_sbs_depth_reuse_authorization(
            host_sbs_depth_reuse_kind_e::exact_roi_damage,
            baseline_frame_id,
            current_frame_id,
            true
          );
        case host_sbs_ddup_reuse_proof_e::none:
          break;
      }
      return {};
    }

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
