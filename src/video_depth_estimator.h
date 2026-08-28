#pragma once

#include "config.h"
#include "host_sbs_adaptive_submission.h"
#include "host_sbs_resolution.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <d3d11.h>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <wrl/client.h>

namespace models {

  /** Immutable identity of the exact model bytes behind raw depth.
   *
   * preprocess_profile is populated only when the complete model name/URL/SHA identity resolves
   * to a calibrated coordinate-v2 entry and the runtime RGB-to-NCHW source closure has the exact
   * calibrated digest. An empty profile is explicitly uncalibrated rather than an inferred match
   * by registry position.
   */
  struct raw_model_provenance_t {
    std::string depth_model;
    std::string depth_model_url;
    std::string onnx_sha256;
    std::string preprocess_profile;
    std::string preprocess_source_closure_sha256;
  };

  /** Immutable identity of the optional fused runtime wrapped around the raw DAV2 producer.
   *
   * Raw DAV2 calibration remains owned by raw_model_provenance_t.  This record deliberately
   * describes only the separately authenticated composite ONNX/engine artifact selected from
   * local assets, so a diagnostic consumer cannot mistake the wrapper hash for DAV2 provenance.
   */
  struct composite_depth_runtime_provenance_t {
    std::string model;
    std::string onnx_sha256;
    std::string embedded_dav2_onnx_sha256;
    std::string zipdepth_checkpoint_sha256;
    std::string guidance_preprocess_source_closure_sha256;
    std::string engine_recipe;
    std::string engine_artifact;
    std::string active_engine_manifest;
  };

  /** Immutable identity of the complete producer closure behind a Host SBS parallax result. */
  struct parallax_v2_shader_provenance_t {
    std::uint32_t source_closure_schema = 0;
    std::uint32_t source_compile_flags = 0;
    std::uint32_t source_macro_count = 0;
    std::string source_closure_sha256;
  };

  /** Immutable identity of the optional diagnostic completion-ring shader. */
  struct host_sbs_gpu_trace_provenance_t {
    std::uint32_t source_closure_schema = 0;
    std::uint32_t source_compile_flags = 0;
    std::uint32_t source_macro_count = 0;
    std::string source_closure_sha256;
  };

  inline bool parallax_v2_shader_provenance_matches_current_contract(
    const parallax_v2_shader_provenance_t &identity
  ) noexcept {
    return identity.source_closure_schema ==
             depth_coordinate_v2::shader_source_closure_schema &&
           identity.source_compile_flags ==
             depth_coordinate_v2::shader_source_compile_flags &&
           identity.source_macro_count == depth_coordinate_v2::shader_source_macro_count &&
           identity.source_closure_sha256 ==
             depth_coordinate_v2::shader_source_closure_sha256;
  }

  enum class engine_build_status {
    unknown,
    building,
    ready,
    failed,
  };

  namespace detail {

    /** Failure phase for a context that has already completed its startup warmup.
     *
     * Pre-enqueue interop and binding failures do not mutate asynchronous TensorRT execution
     * state, so the context remains reusable by a later estimator. Once execution may have
     * started, or teardown cannot prove the stream clean, the MinGW/MSVC ABI prevents destroying
     * the context and it must instead be quarantined.
     */
    enum class warmed_execution_context_failure_e : std::uint8_t {
      pre_enqueue_interop_or_binding,
      asynchronous_execution_or_query,
      unsafe_teardown,
    };

    constexpr bool warmed_execution_context_requires_quarantine(
      const warmed_execution_context_failure_e failure
    ) noexcept {
      return failure !=
             warmed_execution_context_failure_e::pre_enqueue_interop_or_binding;
    }

    class warmed_execution_context_health_t {
    public:
      constexpr void observe(
        const warmed_execution_context_failure_e failure
      ) noexcept {
        poisoned_ = poisoned_ ||
                    warmed_execution_context_requires_quarantine(failure);
      }

      constexpr bool poisoned() const noexcept {
        return poisoned_;
      }

    private:
      bool poisoned_ = false;
    };

    /** Conservative ownership response for a CUDA API failure around the joined root stream.
     *
     * Required interop/binding operations always fail the estimator terminally. Optional timing
     * instrumentation may be ignored only before the first asynchronous inference submission.
     * After any private bootstrap or joined-root launch, CUDA is allowed to surface an earlier
     * asynchronous failure through a later map, pointer, event, or query call, so the depth context
     * is quarantined; OCR is quarantined once it has ever been submitted or armed in that stream
     * lifetime.
     */
    struct joined_cuda_failure_policy_t {
      bool terminal = false;
      bool quarantine_depth = false;
      bool quarantine_ocr = false;

      bool operator==(const joined_cuda_failure_policy_t &) const = default;
    };

    [[nodiscard]] constexpr joined_cuda_failure_policy_t joined_cuda_failure_policy(
      const bool required_operation,
      const bool async_work_ever_submitted,
      const bool ocr_ever_submitted_or_armed
    ) noexcept {
      return {
        .terminal = required_operation || async_work_ever_submitted,
        .quarantine_depth = async_work_ever_submitted,
        .quarantine_ocr = async_work_ever_submitted && ocr_ever_submitted_or_armed,
      };
    }

    /** Normalized result of a nonblocking asynchronous-stream query.
     *
     * Keeping CUDA's numeric result codes out of the policy makes the exact-frame joined-root
     * query independently testable. Any query failure is context-wide because CUDA may surface an
     * earlier asynchronous child or root launch error through this later stream operation.
     */
    enum class async_stream_readiness_e : std::uint8_t {
      ready,
      busy,
      failed,
    };

    /** Bounded teardown action for one nonblocking stream-query observation. */
    enum class teardown_quiescence_action_e : std::uint8_t {
      release_operands,
      retry_query,
      retain_operands,
    };

    [[nodiscard]] constexpr teardown_quiescence_action_e teardown_quiescence_action(
      const async_stream_readiness_e readiness,
      const bool deadline_remaining
    ) noexcept {
      if (readiness == async_stream_readiness_e::ready) {
        return teardown_quiescence_action_e::release_operands;
      }
      if (readiness == async_stream_readiness_e::busy && deadline_remaining) {
        return teardown_quiescence_action_e::retry_query;
      }
      return teardown_quiescence_action_e::retain_operands;
    }

    /** A raw CUDA teardown handle may be forgotten only when absent or positively destroyed. */
    [[nodiscard]] constexpr bool teardown_cuda_handle_may_be_forgotten(
      const bool handle_present,
      const bool destroy_api_available,
      const bool destroy_succeeded
    ) noexcept {
      return !handle_present || (destroy_api_available && destroy_succeeded);
    }

    /** A poisoned conditional-bridge failure inherits every prior optional-stream participant. */
    [[nodiscard]] constexpr bool bridge_failure_quarantines_ocr(
      const bool poison_execution_context,
      const bool ocr_ever_submitted_or_armed
    ) noexcept {
      return poison_execution_context && ocr_ever_submitted_or_armed;
    }

    /** Once a CUDA teardown chain fails, later successes cannot make further releases safe. */
    [[nodiscard]] constexpr bool cuda_teardown_chain_may_continue(
      const bool chain_was_clean,
      const bool operation_succeeded
    ) noexcept {
      return chain_was_clean && operation_succeeded;
    }

    /** Whether CUDA operands may be released after a possible asynchronous submission.
     *
     * enqueueV3(false) does not prove that no work was partially submitted. Once submission was
     * attempted, only an available synchronize operation that returns success proves the stream
     * quiescent enough to free captured operands or destroy its stream.
     */
    constexpr bool asynchronous_operands_may_be_released(
      const bool submission_attempted,
      const bool synchronize_available,
      const async_stream_readiness_e synchronize_result
    ) noexcept {
      return !submission_attempted ||
             (synchronize_available && synchronize_result == async_stream_readiness_e::ready);
    }

    class execution_context_accounting_t {
    public:
      constexpr std::size_t allocated() const noexcept {
        return usable_ + quarantined_;
      }

      constexpr std::size_t usable() const noexcept {
        return usable_;
      }

      constexpr std::size_t warmed() const noexcept {
        return warmed_;
      }

      constexpr std::size_t quarantined() const noexcept {
        return quarantined_;
      }

      constexpr bool reserve(const std::size_t limit) noexcept {
        if (allocated() >= limit) {
          return false;
        }
        ++usable_;
        return true;
      }

      constexpr void release_reservation() noexcept {
        if (usable_ > 0u) {
          --usable_;
        }
      }

      constexpr void mark_warmed() noexcept {
        ++warmed_;
      }

      constexpr void quarantine(const bool was_warmed) noexcept {
        release_reservation();
        if (was_warmed && warmed_ > 0u) {
          --warmed_;
        }
        ++quarantined_;
      }

      constexpr void detach_pooled(const std::size_t pooled_count) noexcept {
        quarantined_ += pooled_count;
        usable_ = 0u;
        warmed_ = 0u;
      }

    private:
      std::size_t usable_ = 0u;
      std::size_t warmed_ = 0u;
      std::size_t quarantined_ = 0u;
    };

    struct cuda_graph_signature_t {
      std::uint64_t input = 0u;
      std::uint64_t output = 0u;
      int width = 0;
      int height = 0;
      std::uint64_t guidance_input = 0u;
      std::uint64_t refined_output = 0u;
      std::uint32_t optimization_profile = 0u;

      constexpr bool operator==(const cuda_graph_signature_t &) const noexcept = default;
    };

    /** Fail-closed selection for the local Stage-1 composite asset. */
    enum class composite_depth_asset_resolution_e : std::uint8_t {
      legacy,
      fused,
      fail,
    };

    [[nodiscard]] constexpr composite_depth_asset_resolution_e
    resolve_composite_depth_asset(
      const bool present,
      const bool regular_file,
      const bool sha256_matches
    ) noexcept {
      if (!present) {
        return composite_depth_asset_resolution_e::legacy;
      }
      return regular_file && sha256_matches ?
               composite_depth_asset_resolution_e::fused :
               composite_depth_asset_resolution_e::fail;
    }

    /** TensorRT 11.2 may require a 512-byte output allocation for a logical FP32 tensor.
     *
     * Inputs remain exact-sized. Output SRVs still expose only the logical elements; this
     * checked padding is private storage required by enqueueV3/getMaxOutputSize.
     */
    inline constexpr std::size_t tensorrt_output_allocation_alignment = 512u;

    [[nodiscard]] constexpr std::optional<std::size_t>
    checked_tensorrt_output_allocation_bytes(
      const std::size_t logical_bytes
    ) noexcept {
      const std::size_t remainder =
        logical_bytes % tensorrt_output_allocation_alignment;
      if (remainder == 0u) {
        return logical_bytes;
      }
      const std::size_t padding =
        tensorrt_output_allocation_alignment - remainder;
      if (logical_bytes > std::numeric_limits<std::size_t>::max() - padding) {
        return std::nullopt;
      }
      return logical_bytes + padding;
    }

    /** TensorRT 11.2 optimization profiles cannot be shared by concurrent contexts.
     *
     * The fused engine's six profiles are shape identities, not concurrency duplicates, so one
     * physical context owns the engine for the process lifetime. A quarantined fused context
     * therefore requires process restart. Legacy DAV2 and OCR retain the established transition
     * pool bound.
     */
    inline constexpr std::size_t standard_tensorrt_context_limit = 4u;
    inline constexpr std::size_t fused_tensorrt_context_limit = 1u;

    [[nodiscard]] constexpr std::size_t tensorrt_context_limit(
      const bool fused_depth_runtime
    ) noexcept {
      return fused_depth_runtime ?
               fused_tensorrt_context_limit :
               standard_tensorrt_context_limit;
    }

    struct cuda_graph_replay_policy_t {
      cuda_graph_signature_t signature;
      bool signature_warmed = false;
      bool capture_failed = false;
    };

    constexpr bool select_cuda_graph_signature(
      cuda_graph_replay_policy_t &state,
      const cuda_graph_signature_t &signature
    ) noexcept {
      if (state.signature == signature) {
        return false;
      }
      state.signature = signature;
      state.signature_warmed = false;
      return true;
    }

    constexpr bool cuda_graph_signature_matches(
      const cuda_graph_replay_policy_t &state,
      const cuda_graph_signature_t &signature
    ) noexcept {
      return state.signature_warmed && state.signature == signature;
    }

  }  // namespace detail

  enum class input_color_space : uint32_t {
    srgb = 0,  ///< gamma-encoded SDR UNORM capture
    linear_sdr = 1,  ///< linear FP16 capture targeting an SDR stream
    scrgb_hdr = 2,  ///< linear scRGB FP16 HDR capture; tone-map for the SDR-trained model
  };

  constexpr bool input_color_space_is_linear(const input_color_space color_space) {
    return color_space != input_color_space::srgb;
  }

  /** Semantic authority that selected one DAV2 analysis domain. */
  enum class depth_analysis_authority_e : std::uint8_t {
    full_source,
    chromium_video,
    foreground_client,
  };

  /**
   * Exact logical analysis raster and its placement in the retained full source.
   *
   * full_source authority means the estimator input and final parallax cover the whole source and
   * analysis_generation is zero. Chromium-video and foreground-client authority mean a bounded
   * analysis ROI with a nonzero analysis generation. Its dimensions
   * describe crop-local analysis coordinates, while the shaders sample that rectangle directly
   * from the full retained source. Position alone is not part of the analysis domain, so moving an
   * otherwise-identical authority may retain scene state.
   */
  struct depth_input_region_t {
    std::uint32_t source_width = 0u;
    std::uint32_t source_height = 0u;
    std::uint32_t left = 0u;
    std::uint32_t top = 0u;
    std::uint32_t right = 0u;
    std::uint32_t bottom = 0u;
    depth_tensor_content_rect_t tensor_content {};
    std::uint64_t analysis_generation = 0u;
    depth_analysis_authority_e authority = depth_analysis_authority_e::full_source;

    constexpr bool is_video_region() const noexcept {
      return authority == depth_analysis_authority_e::chromium_video ||
             authority == depth_analysis_authority_e::foreground_client;
    }

    constexpr std::uint32_t width() const noexcept {
      return right > left ? right - left : 0u;
    }

    constexpr std::uint32_t height() const noexcept {
      return bottom > top ? bottom - top : 0u;
    }

    constexpr bool valid() const noexcept {
      if (source_width == 0u || source_height == 0u || left >= right || top >= bottom ||
          right > source_width || bottom > source_height || !tensor_content.valid()) {
        return false;
      }
      if (is_video_region()) {
        return analysis_generation != 0u;
      }
      return authority == depth_analysis_authority_e::full_source &&
             left == 0u && top == 0u && right == source_width &&
             bottom == source_height && tensor_content.left == 0u &&
             tensor_content.top == 0u &&
             analysis_generation == 0u;
    }

    constexpr bool same_analysis_domain(const depth_input_region_t &other) const noexcept {
      return authority == other.authority &&
             source_width == other.source_width && source_height == other.source_height &&
             width() == other.width() && height() == other.height() &&
             tensor_content == other.tensor_content &&
             analysis_generation == other.analysis_generation;
    }

    constexpr bool operator==(const depth_input_region_t &) const = default;
  };

  /** Stable semantic identity used to assign an ROI analysis generation.
   *
   * Authority kind participates in identity so a Chromium video and a foreground client can never
   * share temporal state accidentally.
   * Position and observer snapshot generation are deliberately absent. Moving the same authority
   * or receiving an otherwise-identical observer heartbeat does not create a new analysis domain.
   */
  struct depth_analysis_domain_key_t {
    std::uint32_t source_width = 0u;
    std::uint32_t source_height = 0u;
    std::uint32_t semantic_width = 0u;
    std::uint32_t semantic_height = 0u;
    std::uint32_t crop_width = 0u;
    std::uint32_t crop_height = 0u;
    std::uint64_t hwnd = 0u;
    std::uint32_t process_id = 0u;
    std::int32_t document_id = 0;
    std::int32_t video_id = 0;
    depth_analysis_authority_e authority = depth_analysis_authority_e::full_source;

    constexpr bool operator==(const depth_analysis_domain_key_t &) const = default;
  };

  /** Assign monotonically changing nonzero generations to stable ROI semantic identities. */
  class depth_analysis_generation_tracker_t {
  public:
    std::uint64_t select(const depth_analysis_domain_key_t &key) noexcept {
      if (!key_ || *key_ != key) {
        key_ = key;
        if (++generation_ == 0u) {
          ++generation_;
        }
      }
      return generation_;
    }

    /** Mark the active route full-source; a later ROI must rearm even for the prior identity. */
    void select_full_source() noexcept {
      key_.reset();
    }

  private:
    std::optional<depth_analysis_domain_key_t> key_;
    std::uint64_t generation_ = 0u;
  };

  /** Pure transition decision used immediately before temporal/camera state is reset. */
  class depth_input_domain_tracker_t {
  public:
    [[nodiscard]] bool matches_analysis_domain(
      const depth_input_region_t &region,
      const input_color_space color_space
    ) const noexcept {
      return initialized_ && color_space_ == color_space &&
             region_.same_analysis_domain(region);
    }

    bool update(
      const depth_input_region_t &region,
      const input_color_space color_space
    ) noexcept {
      const bool changed = !initialized_ || color_space_ != color_space ||
                           !region_.same_analysis_domain(region);
      if (changed) {
        region_ = region;
        color_space_ = color_space;
        initialized_ = true;
      }
      return changed;
    }

  private:
    depth_input_region_t region_ {};
    input_color_space color_space_ = input_color_space::srgb;
    bool initialized_ = false;
  };

  /** Build and warm the active model and prewarm fixed-shape Host SBS shader bytecode. */
  bool prepare_tensorrt_model(
    const std::filesystem::path &assets_dir,
    const config::depth_model_info &model,
    const std::string &adapter_name
  );
  /** Build and warm the optional authenticated PP-OCRv6 tiny detector. Failure is fail-flat. */
  bool prepare_ocr_tensorrt_model(
    const std::filesystem::path &assets_dir,
    const std::string &adapter_name
  );
  engine_build_status tensorrt_model_prepare_status(const config::depth_model_info &model);

  /**
   * @brief Result of one estimate call: private cut-analysis depth, cut state, and the
   *        authenticated Host SBS parallax field/state.
   */
  [[nodiscard]] constexpr bool ocr_signature_refresh_satisfied(
    const bool refresh_required,
    const bool ocr_available,
    const bool force_infer_enqueued,
    const depth_optional_work_mode_e accepted_work,
    const bool ocr_child_enqueued
  ) noexcept {
    return !refresh_required || !ocr_available ||
           (force_infer_enqueued &&
            (accepted_work == depth_optional_work_mode_e::ordinary ||
             accepted_work == depth_optional_work_mode_e::ordinary_due) &&
            ocr_child_enqueued);
  }

  /** Whether a joined DAV2 wrapper may embed, retain, or must drop its OCR sibling.
   *
   * `retain_if_present` is used while OCR is deliberately unarmed and its interop resources are
   * unmapped. An already-instantiated superset remains valid because the authenticated setter
   * leaves that child dormant, but a replacement wrapper must be built depth-only until a mapped
   * ordinary transaction can validate and capture the OCR signature again.
   */
  enum class conditional_optional_child_policy_e : std::uint8_t {
    disabled,
    retain_if_present,
    build_ready,
  };

  struct conditional_optional_topology_selection_t {
    bool reuse_existing = false;
    bool build_with_optional = false;

    constexpr bool operator==(const conditional_optional_topology_selection_t &) const = default;
  };

  [[nodiscard]] constexpr conditional_optional_topology_selection_t
  select_conditional_optional_topology(
    const bool wrapper_base_matches,
    const bool existing_optional_present,
    const bool existing_optional_matches_requested,
    const conditional_optional_child_policy_e policy
  ) noexcept {
    switch (policy) {
      case conditional_optional_child_policy_e::disabled:
        return {
          .reuse_existing = wrapper_base_matches && !existing_optional_present,
          .build_with_optional = false,
        };
      case conditional_optional_child_policy_e::retain_if_present:
        return {
          .reuse_existing = wrapper_base_matches,
          .build_with_optional = false,
        };
      case conditional_optional_child_policy_e::build_ready:
        return {
          .reuse_existing =
            wrapper_base_matches && existing_optional_matches_requested,
          .build_with_optional = true,
        };
    }
    return {};
  }

  /** Fixed GPU-only near-identical decision and history-owner contracts. */
  inline constexpr std::uint32_t near_identical_history_owner_contract_tag = 0x3142484Eu;
  inline constexpr std::uint32_t near_identical_history_owner_contract_schema = 2u;
  inline constexpr std::size_t near_identical_history_owner_word_count = 10u;

  inline constexpr std::uint32_t near_identical_decision_cookie = 0xD1EC15A5u;
  inline constexpr std::uint32_t near_identical_token_low_cookie = 0xA3756C91u;
  inline constexpr std::uint32_t near_identical_token_high_cookie = 0x5C8A936Eu;
  inline constexpr std::uint32_t near_identical_proposal_magic = 0x504F5250u;  // PROP
  inline constexpr std::uint32_t near_identical_receipt_magic = 0x47524243u;  // CBRG
  inline constexpr std::uint32_t near_identical_request_magic = 0x54535152u;  // RQST
  inline constexpr std::uint32_t near_identical_optional_receipt_magic = 0x52434F4Fu;
  inline constexpr std::uint32_t near_identical_work_flags_cookie = 0x6F435257u;
  inline constexpr std::uint32_t near_identical_work_optional_ocr = 1u << 0u;
  inline constexpr std::uint32_t near_identical_work_subtitle_observation = 1u << 1u;
  inline constexpr std::uint32_t near_identical_work_optional_ocr_due = 1u << 3u;
  inline constexpr std::uint32_t near_identical_work_subtitle_observation_due = 1u << 4u;
  inline constexpr std::uint32_t near_identical_work_optional_infer =
    near_identical_work_optional_ocr;
  inline constexpr std::size_t near_identical_gpu_decision_word_count = 64u;
  inline constexpr std::size_t near_identical_gpu_decision_byte_count =
    near_identical_gpu_decision_word_count * sizeof(std::uint32_t);

  enum class near_identical_gpu_branch_e : std::uint32_t {
    reuse = 0u,
    infer = 1u,
  };

  enum class near_identical_gpu_decision_word_e : std::size_t {
    decision,
    decision_cookie,
    decision_token_low,
    decision_token_high,
    decision_token_low_cookie,
    decision_token_high_cookie,
    decision_magic,
    decision_reserved,
    request_token_low,
    request_token_high,
    request_token_low_cookie,
    request_token_high_cookie,
    request_magic,
    request_work_flags,
    request_work_flags_cookie,
    request_reserved,
    infer_reduce_x,
    infer_reduce_y,
    infer_reduce_z,
    infer_reduce_padding,
    infer_one_x,
    infer_one_y,
    infer_one_z,
    infer_one_padding,
    infer_grid16_x,
    infer_grid16_y,
    infer_grid16_z,
    infer_grid16_padding,
    infer_grid8_x,
    infer_grid8_y,
    infer_grid8_z,
    infer_grid8_padding,
    infer_columns_x,
    infer_columns_y,
    infer_columns_z,
    infer_columns_padding,
    infer_rows_x,
    infer_rows_y,
    infer_rows_z,
    infer_rows_padding,
    reuse_grid16_x,
    reuse_grid16_y,
    reuse_grid16_z,
    reuse_grid16_padding,
    optional_preprocess_x,
    optional_preprocess_y,
    optional_preprocess_z,
    optional_preprocess_padding,
    optional_cells_x,
    optional_cells_y,
    optional_cells_z,
    optional_cells_padding,
    optional_one_x,
    optional_one_y,
    optional_one_z,
    optional_one_padding,
    infer_without_optional_one_x,
    infer_without_optional_one_y,
    infer_without_optional_one_z,
    infer_without_optional_one_padding,
    observation_one_x,
    observation_one_y,
    observation_one_z,
    observation_one_padding,
  };

  [[nodiscard]] constexpr std::size_t near_identical_gpu_decision_word_index(
    const near_identical_gpu_decision_word_e word
  ) noexcept {
    return static_cast<std::size_t>(word);
  }

  [[nodiscard]] constexpr std::uint32_t near_identical_gpu_decision_byte_offset(
    const near_identical_gpu_decision_word_e word
  ) noexcept {
    return static_cast<std::uint32_t>(
      near_identical_gpu_decision_word_index(word) * sizeof(std::uint32_t)
    );
  }

  inline constexpr std::uint32_t near_identical_gpu_decision_record_byte_offset =
    near_identical_gpu_decision_byte_offset(
      near_identical_gpu_decision_word_e::decision
    );
  inline constexpr std::uint32_t near_identical_gpu_request_record_byte_offset =
    near_identical_gpu_decision_byte_offset(
      near_identical_gpu_decision_word_e::request_token_low
    );
  inline constexpr std::uint32_t near_identical_gpu_infer_reduce_byte_offset =
    near_identical_gpu_decision_byte_offset(
      near_identical_gpu_decision_word_e::infer_reduce_x
    );
  inline constexpr std::uint32_t near_identical_gpu_infer_one_byte_offset =
    near_identical_gpu_decision_byte_offset(
      near_identical_gpu_decision_word_e::infer_one_x
    );
  inline constexpr std::uint32_t near_identical_gpu_infer_grid16_byte_offset =
    near_identical_gpu_decision_byte_offset(
      near_identical_gpu_decision_word_e::infer_grid16_x
    );
  inline constexpr std::uint32_t near_identical_gpu_infer_grid8_byte_offset =
    near_identical_gpu_decision_byte_offset(
      near_identical_gpu_decision_word_e::infer_grid8_x
    );
  inline constexpr std::uint32_t near_identical_gpu_infer_columns_byte_offset =
    near_identical_gpu_decision_byte_offset(
      near_identical_gpu_decision_word_e::infer_columns_x
    );
  inline constexpr std::uint32_t near_identical_gpu_infer_rows_byte_offset =
    near_identical_gpu_decision_byte_offset(
      near_identical_gpu_decision_word_e::infer_rows_x
    );
  inline constexpr std::uint32_t near_identical_gpu_reuse_grid16_byte_offset =
    near_identical_gpu_decision_byte_offset(
      near_identical_gpu_decision_word_e::reuse_grid16_x
    );
  inline constexpr std::uint32_t near_identical_gpu_optional_preprocess_byte_offset =
    near_identical_gpu_decision_byte_offset(
      near_identical_gpu_decision_word_e::optional_preprocess_x
    );
  // D3D consumes this slot before CUDA mapping. The authenticated post-CUDA args writer reuses it
  // for the complete out-of-place subtitle-conditioner grid.
  inline constexpr std::uint32_t
    near_identical_gpu_subtitle_condition_grid16_byte_offset =
      near_identical_gpu_optional_preprocess_byte_offset;
  inline constexpr std::uint32_t near_identical_gpu_optional_cells_byte_offset =
    near_identical_gpu_decision_byte_offset(
      near_identical_gpu_decision_word_e::optional_cells_x
    );
  inline constexpr std::uint32_t near_identical_gpu_optional_one_byte_offset =
    near_identical_gpu_decision_byte_offset(
      near_identical_gpu_decision_word_e::optional_one_x
    );
  inline constexpr std::uint32_t
    near_identical_gpu_infer_without_optional_one_byte_offset =
      near_identical_gpu_decision_byte_offset(
        near_identical_gpu_decision_word_e::infer_without_optional_one_x
      );
  inline constexpr std::uint32_t near_identical_gpu_subtitle_record_one_byte_offset =
    near_identical_gpu_infer_without_optional_one_byte_offset;
  inline constexpr std::uint32_t near_identical_gpu_observation_one_byte_offset =
    near_identical_gpu_decision_byte_offset(
      near_identical_gpu_decision_word_e::observation_one_x
    );

  static_assert(near_identical_gpu_decision_record_byte_offset == 0u);
  static_assert(near_identical_gpu_request_record_byte_offset == 32u);
  static_assert(near_identical_gpu_decision_record_byte_offset % 16u == 0u);
  static_assert(near_identical_gpu_request_record_byte_offset % 16u == 0u);
  static_assert(near_identical_gpu_infer_reduce_byte_offset == 64u);
  static_assert(near_identical_gpu_infer_one_byte_offset == 80u);
  static_assert(near_identical_gpu_infer_grid16_byte_offset == 96u);
  static_assert(near_identical_gpu_infer_grid8_byte_offset == 112u);
  static_assert(near_identical_gpu_infer_columns_byte_offset == 128u);
  static_assert(near_identical_gpu_infer_rows_byte_offset == 144u);
  static_assert(near_identical_gpu_reuse_grid16_byte_offset == 160u);
  static_assert(near_identical_gpu_optional_preprocess_byte_offset == 176u);
  static_assert(near_identical_gpu_optional_cells_byte_offset == 192u);
  static_assert(near_identical_gpu_optional_one_byte_offset == 208u);
  static_assert(near_identical_gpu_infer_without_optional_one_byte_offset == 224u);
  static_assert(near_identical_gpu_observation_one_byte_offset == 240u);
  static_assert(near_identical_gpu_decision_byte_count % 16u == 0u);
  static_assert(
    near_identical_gpu_decision_word_index(
      near_identical_gpu_decision_word_e::observation_one_padding
    ) + 1u == near_identical_gpu_decision_word_count
  );

  /** Stable host-only identity for the analysis domain whose NCHW history the GPU owns.
   *
   * Position is intentionally absent, matching depth_input_region_t::same_analysis_domain(). The
   * display still owns exact positioned-route authorization independently. This tag is a compact
   * lineage binding, not a cryptographic authentication claim.
   */
  [[nodiscard]] constexpr std::uint64_t near_identical_input_domain_tag(
    const depth_input_region_t &region,
    const input_color_space color_space,
    const std::uint32_t field_width,
    const std::uint32_t field_height
  ) noexcept {
    std::uint64_t hash = 1469598103934665603ull;
    const auto append = [&hash](const std::uint64_t value) constexpr {
      for (unsigned byte = 0u; byte < 8u; ++byte) {
        hash ^= (value >> (byte * 8u)) & 0xffu;
        hash *= 1099511628211ull;
      }
    };
    append(region.source_width);
    append(region.source_height);
    append(region.width());
    append(region.height());
    append(region.tensor_content.left);
    append(region.tensor_content.top);
    append(region.tensor_content.right);
    append(region.tensor_content.bottom);
    append(region.analysis_generation);
    // Preserve the established serialized/tag ABI bit while deriving it from semantic authority.
    append(region.is_video_region() ? 1u : 0u);
    append(static_cast<std::uint8_t>(region.authority));
    append(static_cast<std::uint32_t>(color_space));
    append(field_width);
    append(field_height);
    return hash != 0u ? hash : 1u;
  }

  struct estimate_result {
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> cut_state;  ///< Cut-analysis state shared with telemetry and coordinate production.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_frame_state;  ///< {min,max,initialized,frame_state}; frame_state 0 means an all-invalid completion held the prior depth.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> ema_motion_mask;  ///< Edge-selective EMA snap mask.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> raw_model_depth;  ///< Raw model output buffer, before normalization/EMA/curvature; primarily for the offline evaluator.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> raw_model_depth_snapshot;  ///< Optional stable copy of the completed frame's raw output for a live Dump 3D request.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> model_input_snapshot;  ///< Optional stable NCHW/ImageNet-normalized input for the same live Dump 3D frame.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> guidance_model_input_snapshot;  ///< Exact-force-only stable fused guidance NCHW [1,3,2H,2W].
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> refined_model_depth_snapshot;  ///< Exact-force-only stable diagnostic refined output [1,2H,2W].
    std::shared_ptr<const raw_model_provenance_t> raw_model_provenance;  ///< Capture-time model-byte identity; copied by pointer on ordinary frames.
    std::shared_ptr<const composite_depth_runtime_provenance_t>
      composite_depth_runtime_provenance;  ///< Separate fused ONNX/engine identity; never raw DAV2 calibration authority.
    // Production V2 outputs. candidate_parallax is immutable pre-conditioner evidence;
    // ownership_refined_parallax is the full-resolution source-contour ownership result consumed
    // by the vertical pass, vertical_majorant is the upper-envelope diagnostic,
    // vertical_conditioned is the fixed upper/lower vertical share consumed by the pure row
    // majorant. base_final_parallax is independently observable as the ordinary post-limiter field
    // for explicit diagnostics and padded ROI. Subtitle conditioning always writes a separate
    // complete texture, so base_final_parallax remains immutable. final_parallax is that atomic
    // OCR-conditioned publication and is sampled directly by the renderer. Adaptive depth reuse
    // always holds the DAV2 field; ordinary subtitle work holds the OCR/SLR/final tuple, while a
    // cadence-due subtitle observation may advance that tuple independently.
    // In ROI mode
    // both are crop-local producer q; only final_parallax becomes renderer authority through
    // input_region's authenticated scale/collar embedding. coordinate is an optional
    // Dump-3D-only snapshot, never a live resource or authentication prerequisite. The legacy
    // `shadow_*` prefix remains for dump compatibility.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> shadow_coordinate;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> shadow_candidate_parallax;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> shadow_ownership_refined_parallax;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> shadow_vertical_majorant;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> shadow_vertical_conditioned;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> shadow_base_final_parallax;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> shadow_final_parallax;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> shadow_state;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> shadow_frame_stats;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> ocr_box_record;  ///< Exact-frame OCR8 only when subtitle_evidence_is_exact_frame() succeeds.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> subtitle_locator_state;  ///< Exact-frame SLR13 only when subtitle_evidence_is_exact_frame() succeeds.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> gpu_trace_ring;  ///< Optional diagnostic-only rolling completion trace; never rendering authority.
    std::shared_ptr<const parallax_v2_shader_provenance_t>
      parallax_v2_shader_provenance;  ///< Exact producer shader closure when V2 is active.
    std::shared_ptr<const host_sbs_gpu_trace_provenance_t>
      gpu_trace_provenance;  ///< Source identity for gpu_trace_ring when available.
    int raw_width = 0;
    int raw_height = 0;
    int field_width = 0;  ///< Spatial dimensions of every live parallax texture; raw_* stays coarse DAV2.
    int field_height = 0;
    depth_tensor_content_rect_t field_content {};  ///< Exact half-open live-field realization of input_region.tensor_content.
    int guidance_width = 0;
    int guidance_height = 0;
    int refined_width = 0;
    int refined_height = 0;
    bool refined_live_geometry_active = false;  ///< Session-latched exact-2x landscape spatial path.
    // A wrapper transaction completed and its GPU normalization passes were submitted. The associated
    // depth_frame_state decides on-GPU whether this completion contains valid depth or must hold
    // the previous matched color/depth output.
    bool completed_frame_valid = false;
    std::uint64_t completed_frame_id = 0;  ///< Caller-provided identity of that completed result.
    bool inference_enqueued = false;  ///< This call submitted a force-infer wrapper transaction for the supplied input frame.
    bool gpu_undecided_transaction_enqueued = false;  ///< This call submitted a GPU-conditional depth transaction; its branch remains device-owned.
    bool gpu_undecided_completion = false;  ///< The completed transaction's actual branch remains GPU-owned and is deliberately not read back.
    bool cuda_graph_active = false;  ///< The mandatory DAV2 wrapper is ready and owns its embedded inference child.
    bool parallax_v2_producer_active = false;  ///< All production V2 producer shaders/resources are active.
    float parallax_v2_raw_coordinate_scale = 0.0f;  ///< Fixed authenticated model/shape coordinate scale.
    float parallax_v2_requested_pop_strength = 0.0f;  ///< Fixed V2 request from cfg.pop_strength only; no legacy adaptive ratio or ceiling is consumed.
    float parallax_v2_requested_gain = 0.0f;  ///< One-eye source-U gain before safety attenuation.
    depth_input_region_t input_region {};  ///< Exact source domain that owns this completion.
    input_color_space color_space = input_color_space::srgb;  ///< Exact transfer domain used for this completion.
    bool input_domain_reset = false;  ///< Temporal/camera state was reset before this completion.
    bool subtitle_work_suppressed = false;  ///< This completion published Base and did not advance same-domain locator state.
    bool subtitle_ocr_inference_enqueued = false;  ///< This call requested OCR participation; ordinary work is infer-coupled and cadence-due work may execute on either authenticated depth branch.
  };

  [[nodiscard]] inline bool subtitle_evidence_is_exact_frame(
    const estimate_result &result
  ) noexcept {
    return result.completed_frame_valid && !result.gpu_undecided_completion &&
           !result.subtitle_work_suppressed &&
           result.ocr_box_record.Get() != nullptr &&
           result.subtitle_locator_state.Get() != nullptr;
  }

  struct pending_depth_poll_result {
    estimate_result result;
    bool ready = false;  ///< Joined query proved completion; result may still report a failure.
    bool wait_attempted = false;  ///< At least one yielded repeat was attempted after the initial query.
    bool timed_out = false;  ///< The exact pending unit remained busy at the deadline/query fuse.
    std::uint32_t query_count = 0;  ///< Joined DAV2/OCR readiness queries issued by this call.
    std::chrono::steady_clock::duration wait_duration {};  ///< Query time only; excludes completed-depth postprocess.
  };

  /** Fail-closed CPU authentication for a completed live V2 result.
   *
   * This verifies the complete model/preprocess/shape and producer source identities, the exact
   * full-source/analysis-region shape relation, the presence of every production V2 resource, and
   * the fixed-pop gain relation. Dump-only canonical coordinate evidence is deliberately excluded.
   * It does not map GPU state; the live shader authenticates the per-frame contract tag before
   * sampling geometry. Subtitle-suppressed completions still authenticate their live geometry,
   * but their OCR/SLR views are not exact-frame evidence. A device-conditional completion also
   * cannot expose exact-current CPU OCR lineage because an opaque reuse may either hold ordinary
   * subtitle state or advance a cadence-due observation. Callers gate diagnostic and baseline uses on
   * subtitle_evidence_is_exact_frame(); the diagnostic GPU trace owns per-branch evidence.
   */
  bool parallax_v2_result_is_authenticated(const estimate_result &result);

  enum class host_sbs_renderer_e {
    awaiting_v2,
    parallax_v2,
    failed_flat,
  };

  /** One-way production latch. Host SBS authenticates the first completed V2 field before it may
   * render geometry; rejection is terminal for this stream and can only render live identity.
   */
  constexpr host_sbs_renderer_e latch_host_sbs_renderer(
    const host_sbs_renderer_e current,
    const bool result_authenticated
  ) {
    if (current != host_sbs_renderer_e::awaiting_v2) {
      return current;
    }
    return result_authenticated ?
             host_sbs_renderer_e::parallax_v2 :
             host_sbs_renderer_e::failed_flat;
  }

  /** A failed V2 producer is terminal for geometry, but not for the video stream. The host
   * stops producing unused depth and draws each current color frame through flat identity.
   */
  constexpr bool host_sbs_renderer_uses_depth_pipeline(
    const host_sbs_renderer_e renderer
  ) {
    return renderer != host_sbs_renderer_e::failed_flat;
  }

  inline constexpr auto host_sbs_v2_max_matched_repeat_age =
    std::chrono::milliseconds {250};

  /** A retained capture may be reconverted after an arbitrarily long static interval. Its exact
   * matched completion is still current; wall-clock age only expires geometry for an older source.
   */
  constexpr bool host_sbs_matched_completion_is_current(
    const bool source_unchanged,
    const std::chrono::steady_clock::duration completion_age
  ) {
    return source_unchanged ||
           completion_age <= host_sbs_v2_max_matched_repeat_age;
  }

  /** Select the already-rendered packed SBS presentation while a newer matched unit is pending.
   *
   * This is presentation continuity only. `presentation_route_matches` is supplied by the live
   * route/domain owner and must never be inferred from the opaque depth decision. The retained
   * packed image carries no reusable depth, OCR, damage, or adaptive-submission authority.
   */
  constexpr bool host_sbs_should_repeat_packed_presentation(
    const host_sbs_renderer_e renderer,
    const bool has_matched_frame,
    const bool packed_output_valid,
    const bool presentation_route_matches,
    const std::chrono::steady_clock::duration repeat_source_age =
      std::chrono::steady_clock::duration::zero(),
    const bool source_unchanged = false
  ) {
    return renderer == host_sbs_renderer_e::parallax_v2 &&
           !has_matched_frame && packed_output_valid && presentation_route_matches &&
           host_sbs_matched_completion_is_current(
             source_unchanged,
             repeat_source_age
           );
  }

  /** A completed authenticated V2 draw may seed only the packed presentation cache.
   *
   * The result has already been rendered into a self-contained SBS image. Its private infer/reuse
   * branch is therefore irrelevant to redelivering those exact pixels. Geometry/OCR/adaptive
   * lineage remains governed separately by host_sbs_latest_v2_completion_retention_allowed().
   */
  constexpr bool host_sbs_packed_output_can_enter_presentation_cache(
    const bool has_matched_frame,
    const bool has_authenticated_field
  ) {
    return has_matched_frame && has_authenticated_field;
  }

  /** A terminal producer failure moves every nonfailed Host SBS stream to live flat identity. */
  constexpr host_sbs_renderer_e fail_host_sbs_renderer_flat(
    const host_sbs_renderer_e current
  ) {
    return current == host_sbs_renderer_e::failed_flat ? current :
                                                        host_sbs_renderer_e::failed_flat;
  }

  /**
   * One opportunistic, nonblocking readback of the append-only diagnostic portion of
   * cut-analysis state. Live samples may be skipped or coalesced while the GPU is busy; absence
   * of a sample is not evidence that the detector did not update.
   */
  struct depth_telemetry_sample {
    int depth_width = 0;
    int depth_height = 0;
    float adaptive_pop_ratio = 1.0f;  ///< Retained wire field; V2 always publishes 1.
    float edge_fraction = -1.0f;  ///< Retained wire field; unavailable in V2.
    float change_fraction = 0.0f;
    float zero_anchor_shift_px = 0.0f;  ///< Retained wire field; unavailable in V2.
    float subject_depth = 0.0f;  ///< Retained wire field; unavailable in V2.
    float valid_depth_fraction = 0.0f;
    float effective_range_width = 0.0f;
    // Current-frame cut evidence used to distinguish detector state from incoming observations.
    float current_edge_fraction = -1.0f;
    float current_zero_anchor_candidate_shift_px = -1.0f;
    float structural_change_fraction = -1.0f;
    float raw_rgb_change_fraction = -1.0f;
    std::uint32_t analysis_flags = 0;
    std::uint32_t model_input_history_state = 0;
    std::uint32_t scene_age = 0;
    std::uint32_t cut_flags = 0;
    std::uint32_t hard_cut_count = 0;
    std::uint32_t external_cut_count = 0;
    std::uint32_t empty_raw_count = 0;
    std::uint32_t collapsed_raw_count = 0;
    std::uint64_t sampled_frame_id = 0;
    // Exact wall-clock owner of the CopyResource that captured this CutBridge state. Readback may
    // complete much later and must never make old motion evidence look fresh.
    std::chrono::steady_clock::time_point sampled_at {};
    bool profile_initialized = false;
    bool anchor_valid = false;
    bool range_collapsed = false;
    bool depth_ready = false;
    bool hard_cut_pulse = false;
  };

  struct depth_telemetry_poll_result {
    std::optional<depth_telemetry_sample> sample;
    bool copy_scheduled = false;
    bool failed = false;
  };

  class video_depth_estimator {
  public:
    /**
     * @brief Construct a new video depth estimator
     *
     * @param device D3D11 Device used for the capture pipeline
     * @param context D3D11 Device Context
     * @param assets_dir Path to the assets directory (for model loading)
     * @param cfg Shared SBS controls; see config::video_t::sbs_t (the estimator consumes
     *            pop_strength; the depth-side analysis parameters are the fixed
     *            config::host_sbs_v2_live_calibration values).
     * @param model The selected depth model: name/url (which engine to load/build) plus the
     *            DA-V2-compatible model contract (pixel_values -> predicted_depth).
     */
    video_depth_estimator(
      Microsoft::WRL::ComPtr<ID3D11Device> device,
      Microsoft::WRL::ComPtr<ID3D11DeviceContext> context,
      const std::filesystem::path &assets_dir,
      const config::video_t::sbs_t &cfg,
      const config::depth_model_info &model
    );

    ~video_depth_estimator();

    /** True only when every mandatory engine, shader, and session resource initialized. */
    bool is_valid() const;

    /**
     * @brief Nonblocking producer-side readiness check for matched-frame capture.
     *
     * Returns true only when the estimator can accept a new input immediately. A false return
     * accounts for the source opportunity (and, when CUDA is still working, one busy drop) in the
     * throughput telemetry. It never consumes a completed depth result; the next estimate_depth()
     * call performs that consumption. The live pipeline uses this before copying a full-resolution
     * color frame into its private matched slot.
     */
    bool can_accept_frame();

    /** True after CUDA/TensorRT reports a non-recoverable execution/interoperability error. */
    bool has_terminal_failure() const;

    // Non-copyable
    video_depth_estimator(const video_depth_estimator &) = delete;
    video_depth_estimator &operator=(const video_depth_estimator &) = delete;

    /**
     * @brief Estimate private analysis depth and the authenticated Host SBS parallax field.
     *
     * @param input_srv D3D11 ShaderResourceView containing the RGB image (usually B8G8R8A8_UNORM or R8G8B8A8_UNORM).
     * @param snapshot_debug_inputs Copy the completed frame's exact model input and raw output
     *        before the next asynchronous inference can overwrite them. Intended only for an
     *        explicit Dump 3D.
     * @return estimate_result; all views are owned by the estimator and overwritten by later calls.
     */
    estimate_result estimate_depth(
      ID3D11ShaderResourceView *input_srv,
      input_color_space color_space = input_color_space::srgb,
      std::uint64_t frame_id = 0,
      bool snapshot_debug_inputs = false,
      depth_input_region_t input_region = {},
      depth_optional_work_mode_e optional_work = depth_optional_work_mode_e::ordinary,
      gpu_adaptive_reuse_request adaptive_reuse = {}
    );

    /**
     * @brief Finish and consume exactly one inference previously submitted by estimate_depth().
     *
     * It synchronizes the estimator stream and applies normalization, EMA, cut analysis, and
     * coordinate production exactly once without enqueueing another inference. This is an
     * offline-evaluation quality path; live capture uses only bounded/nonblocking completion polls.
     */
    estimate_result finish_pending_depth_for_evaluation(
      input_color_space color_space = input_color_space::srgb,
      bool snapshot_debug_inputs = false
    );

    /** Query the pending joined DAV2/OCR completion fence once and consume the exact unit when ready. */
    pending_depth_poll_result try_finish_pending_depth_nonblocking(
      input_color_space color_space,
      bool snapshot_debug_inputs = false
    );

    /** Query one pending exact-frame DAV2/OCR unit until ready or an absolute CPU deadline.
     *
     * When available, every query is a nonblocking CUDA event query recorded after the joined
     * conditional root and every interop-unmap tail. Ready postprocess is therefore ordered behind
     * all fixed-buffer releases without synchronizing the root stream. This
     * never flushes or enqueues replacement work; the query-count fuse is a second bound behind
     * the steady-clock deadline.
     * A timeout preserves the pending inference, event state, and exact owner. If event setup was
     * unavailable before enqueue, this degrades to one full-stream query with no repeated wait;
     * admission and fixed-resource reuse always retain their full-stream query.
     */
    pending_depth_poll_result try_finish_pending_depth_until(
      input_color_space color_space,
      std::chrono::steady_clock::time_point deadline,
      std::uint32_t max_queries,
      bool snapshot_debug_inputs = false
    );

    /**
     * Poll completed telemetry copies and optionally enqueue one new copy after the caller has
     * submitted the critical warp/output work. Never flushes, waits, or maps an unsignaled slot.
     * A busy three-slot ring deliberately drops the sampling opportunity rather than delaying
     * capture, so callers must not compare its sample count one-for-one with offline traces.
     */
    depth_telemetry_poll_result poll_depth_telemetry(
      bool schedule_copy,
      std::uint64_t sampled_frame_id
    );

  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
  };

}  // namespace models
