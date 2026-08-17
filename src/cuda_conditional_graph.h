#pragma once

#include "cuda_driver_api.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace cuda_conditional_graph {

  // These cookies authenticate publication/lineage against stale or torn in-process records;
  // they are deliberately not a cryptographic boundary against an adversarial GPU writer.
  inline constexpr std::uint32_t decision_cookie = 0xd1ec15a5u;
  inline constexpr std::uint32_t token_low_cookie = 0xa3756c91u;
  inline constexpr std::uint32_t token_high_cookie = 0x5c8a936eu;
  inline constexpr std::uint32_t work_flags_cookie = 0x6f435257u;

  // Integer values spell the tags in little-endian GPU memory. Keep these in lockstep with the
  // D3D shader contract: the producer publishes PROP/RQST and CUDA replaces PROP with CBRG.
  inline constexpr std::uint32_t proposal_magic = 0x504f5250u;  // PROP
  inline constexpr std::uint32_t request_magic = 0x54535152u;  // RQST
  inline constexpr std::uint32_t receipt_magic = 0x47524243u;  // CBRG
  inline constexpr std::uint32_t optional_ocr_receipt_magic = 0x52434f4fu;  // OOCR
  // Compatibility name for the stable transaction ABI. The marker authenticates execution of
  // the optional OCR sibling. Ordinary OCR requires an authenticated infer branch; cadence-due
  // OCR may execute alongside either infer or reuse.
  inline constexpr std::uint32_t optional_infer_receipt_magic =
    optional_ocr_receipt_magic;

  enum class branch_e : std::uint32_t {
    reuse = 0u,
    infer = 1u,
  };

  enum class work_flag_e : std::uint32_t {
    none = 0u,
    optional_ocr = 1u << 0u,
    subtitle_observation = 1u << 1u,
    optional_ocr_due = 1u << 3u,
    subtitle_observation_due = 1u << 4u,
    optional_infer = optional_ocr,  // Stable source-compatible spelling.
  };

  [[nodiscard]] constexpr std::uint32_t work_flags_value(
    const work_flag_e flags
  ) noexcept {
    return static_cast<std::uint32_t>(flags);
  }

  static_assert(work_flags_value(work_flag_e::optional_ocr) == 1u);
  static_assert(work_flags_value(work_flag_e::subtitle_observation) == 2u);
  static_assert(work_flags_value(work_flag_e::optional_ocr_due) == 8u);
  static_assert(work_flags_value(work_flag_e::subtitle_observation_due) == 16u);

  [[nodiscard]] constexpr bool authenticated_work_flags(
    const std::uint32_t flags
  ) noexcept {
    return flags == work_flags_value(work_flag_e::none) ||
           flags == work_flags_value(work_flag_e::optional_ocr) ||
           flags == work_flags_value(work_flag_e::subtitle_observation) ||
           flags == work_flags_value(work_flag_e::optional_ocr_due) ||
           flags == work_flags_value(work_flag_e::subtitle_observation_due);
  }

  [[nodiscard]] constexpr bool optional_ocr_executes(
    const std::uint32_t flags,
    const branch_e resolved
  ) noexcept {
    return (
             flags == work_flags_value(work_flag_e::optional_ocr) &&
             resolved == branch_e::infer
           ) ||
           flags == work_flags_value(work_flag_e::optional_ocr_due);
  }

  /** GPU producer proposal, overwritten in place by CUDA with the resolved receipt.
   *
   * Publication tags are written last. A proposal is never accepted by a postprocess consumer;
   * only a CBRG receipt matching a valid RQST record is authoritative.
   */
  struct alignas(16) decision_record_t {
    std::uint32_t decision;
    std::uint32_t decision_cookie;
    std::uint32_t token_low;
    std::uint32_t token_high;
    std::uint32_t token_low_cookie;
    std::uint32_t token_high_cookie;
    std::uint32_t magic;
    std::uint32_t reserved;
  };

  /** GPU-visible identity for the current transaction. */
  struct alignas(16) request_record_t {
    std::uint32_t token_low;
    std::uint32_t token_high;
    std::uint32_t token_low_cookie;
    std::uint32_t token_high_cookie;
    std::uint32_t magic;
    std::uint32_t work_flags;
    std::uint32_t work_flags_cookie;
    std::uint32_t reserved;
  };

  static_assert(sizeof(decision_record_t) == 32u);
  static_assert(sizeof(request_record_t) == 32u);
  static_assert(alignof(decision_record_t) == 16u);
  static_assert(alignof(request_record_t) == 16u);

  [[nodiscard]] constexpr decision_record_t make_proposal(
    const branch_e decision,
    const std::uint64_t token
  ) noexcept {
    const auto low = static_cast<std::uint32_t>(token);
    const auto high = static_cast<std::uint32_t>(token >> 32u);
    const auto value = static_cast<std::uint32_t>(decision);
    return {
      value,
      value ^ decision_cookie,
      low,
      high,
      low ^ token_low_cookie,
      high ^ token_high_cookie,
      proposal_magic,
      0u,
    };
  }

  [[nodiscard]] constexpr request_record_t make_request(
    const std::uint64_t token,
    const work_flag_e work_flags = work_flag_e::none
  ) noexcept {
    const auto low = static_cast<std::uint32_t>(token);
    const auto high = static_cast<std::uint32_t>(token >> 32u);
    const auto flags = work_flags_value(work_flags);
    return {
      low,
      high,
      low ^ token_low_cookie,
      high ^ token_high_cookie,
      request_magic,
      flags,
      flags == 0u ? 0u : flags ^ work_flags_cookie,
      0u,
    };
  }

  [[nodiscard]] constexpr bool authenticated_request(
    const request_record_t &request
  ) noexcept {
    return (request.token_low != 0u || request.token_high != 0u) &&
           request.token_low_cookie == (request.token_low ^ token_low_cookie) &&
           request.token_high_cookie == (request.token_high ^ token_high_cookie) &&
           request.magic == request_magic &&
           authenticated_work_flags(request.work_flags) &&
           request.work_flags_cookie ==
             (request.work_flags == 0u ? 0u : request.work_flags ^ work_flags_cookie) &&
           request.reserved == 0u;
  }

  [[nodiscard]] constexpr bool authenticated_reuse_proposal(
    const decision_record_t &proposal,
    const request_record_t &request
  ) noexcept {
    return authenticated_request(request) &&
           proposal.decision == static_cast<std::uint32_t>(branch_e::reuse) &&
           proposal.decision_cookie == (proposal.decision ^ decision_cookie) &&
           proposal.token_low == request.token_low &&
           proposal.token_high == request.token_high &&
           proposal.token_low_cookie == (proposal.token_low ^ token_low_cookie) &&
           proposal.token_high_cookie == (proposal.token_high ^ token_high_cookie) &&
           proposal.magic == proposal_magic && proposal.reserved == 0u;
  }

  [[nodiscard]] constexpr bool authenticated_proposal(
    const decision_record_t &proposal,
    const request_record_t &request
  ) noexcept {
    return authenticated_request(request) &&
           (proposal.decision == static_cast<std::uint32_t>(branch_e::reuse) ||
            proposal.decision == static_cast<std::uint32_t>(branch_e::infer)) &&
           proposal.decision_cookie == (proposal.decision ^ decision_cookie) &&
           proposal.token_low == request.token_low &&
           proposal.token_high == request.token_high &&
           proposal.token_low_cookie == (proposal.token_low ^ token_low_cookie) &&
           proposal.token_high_cookie == (proposal.token_high ^ token_high_cookie) &&
           proposal.magic == proposal_magic && proposal.reserved == 0u;
  }

  [[nodiscard]] constexpr decision_record_t resolve_proposal(
    const decision_record_t &proposal,
    const request_record_t &request,
    const bool optional_child_present = true
  ) noexcept {
    const bool proposal_valid = authenticated_proposal(proposal, request);
    const auto resolved = proposal_valid &&
                            proposal.decision == static_cast<std::uint32_t>(branch_e::reuse) ?
                            branch_e::reuse :
                            branch_e::infer;
    const auto value = static_cast<std::uint32_t>(resolved);
    const bool optional_ocr = optional_child_present && proposal_valid &&
                              optional_ocr_executes(request.work_flags, resolved);
    return {
      value,
      value ^ decision_cookie ^
        (optional_ocr ? optional_ocr_receipt_magic : 0u),
      request.token_low,
      request.token_high,
      request.token_low ^ token_low_cookie,
      request.token_high ^ token_high_cookie,
      receipt_magic,
      optional_ocr ? optional_ocr_receipt_magic : 0u,
    };
  }

  [[nodiscard]] constexpr bool authenticated_receipt(
    const decision_record_t &receipt,
    const request_record_t &request
  ) noexcept {
    const bool optional_receipt = receipt.reserved == 0u ||
                                  (receipt.reserved == optional_ocr_receipt_magic &&
                                   optional_ocr_executes(
                                     request.work_flags,
                                     static_cast<branch_e>(receipt.decision)
                                   ));
    return authenticated_request(request) &&
           (receipt.decision == static_cast<std::uint32_t>(branch_e::reuse) ||
            receipt.decision == static_cast<std::uint32_t>(branch_e::infer)) &&
           receipt.decision_cookie ==
             (receipt.decision ^ decision_cookie ^ receipt.reserved) &&
           receipt.token_low == request.token_low &&
           receipt.token_high == request.token_high &&
           receipt.token_low_cookie == (receipt.token_low ^ token_low_cookie) &&
           receipt.token_high_cookie == (receipt.token_high ^ token_high_cookie) &&
           receipt.magic == receipt_magic && optional_receipt;
  }

  [[nodiscard]] constexpr bool authenticated_optional_ocr_receipt(
    const decision_record_t &receipt,
    const request_record_t &request
  ) noexcept {
    return authenticated_receipt(receipt, request) &&
           receipt.reserved == optional_ocr_receipt_magic;
  }

  [[nodiscard]] constexpr bool authenticated_optional_infer_receipt(
    const decision_record_t &receipt,
    const request_record_t &request
  ) noexcept {
    return authenticated_optional_ocr_receipt(receipt, request);
  }

  /** Exact embedded PTX passed to nvcuda.dll's JIT; no toolkit artifact is loaded. */
  [[nodiscard]] std::string_view bridge_ptx_source() noexcept;

  /** Node types permitted directly in a CUDA conditional body by the Driver API contract. */
  [[nodiscard]] constexpr bool is_conditional_body_node_type_allowed(
    const CUgraphNodeType type
  ) noexcept {
    return type == CU_GRAPH_NODE_TYPE_KERNEL || type == CU_GRAPH_NODE_TYPE_MEMCPY ||
           type == CU_GRAPH_NODE_TYPE_MEMSET || type == CU_GRAPH_NODE_TYPE_GRAPH ||
           type == CU_GRAPH_NODE_TYPE_EMPTY || type == CU_GRAPH_NODE_TYPE_CONDITIONAL;
  }

  enum class audit_failure_e {
    none,
    invalid_graph,
    driver_api_unavailable,
    cuda_error,
    host_allocation_failed,
    unsupported_node_type,
    nested_conditional,
    recursion_limit,
    missing_inference_kernel,
  };

  struct audit_result_t {
    audit_failure_e failure = audit_failure_e::none;
    CUresult cuda_result = CUDA_SUCCESS;
    CUgraphNodeType rejected_type = static_cast<CUgraphNodeType>(-1);
    std::size_t visited_graphs = 0u;
    std::size_t visited_nodes = 0u;
    // CUDA currently defines node values 0..13. Retain a recursive count for every known type so
    // a rejected captured TensorRT graph reports its exact topology (not merely the first event).
    std::array<std::size_t, 14u> node_type_counts {};

    [[nodiscard]] constexpr bool legal() const noexcept {
      return failure == audit_failure_e::none;
    }
  };

  /** Recursively audits a graph before embedding it as a conditional-body child graph.
   *
   * Conditional nodes are legal directly in a body but not in a graph that is itself embedded as
   * a child, so this audit rejects them. Instantiation remains authoritative for memcpy parameter
   * restrictions that are not represented by CUgraphNodeType (arrays are not permitted).
   */
  [[nodiscard]] audit_result_t audit_embeddable_child_graph(
    const cuda_driver_api &cuda,
    CUgraph graph
  ) noexcept;

  /** Audit a TensorRT inference child and require recursively observable kernel work.
   *
   * A non-null capture with no kernel nodes can otherwise be embedded successfully while doing no
   * inference. The conditional receipt would then authenticate stale output from an earlier
   * bootstrap. Reuse children intentionally use the generic embeddable audit because a legitimate
   * reuse body may contain only a copy or memset.
   */
  [[nodiscard]] audit_result_t audit_inference_child_graph(
    const cuda_driver_api &cuda,
    CUgraph graph
  ) noexcept;

  struct build_desc_t {
    CUcontext context = nullptr;
    CUgraph infer_child = nullptr;
    // Optional sibling IF child. An authenticated, preprocess-ready ordinary request runs it only
    // when DAV2 resolves to infer; optional_ocr_due runs it on either resolved branch. It has no
    // dependency on infer_child, so CUDA may schedule both graphs concurrently and the root joins
    // them before completion. Host SBS uses this for the isolated OCR TensorRT context while
    // keeping the published depth/OCR/SLR tuple atomic across reuse.
    CUgraph optional_infer_child = nullptr;
    CUgraph reuse_child = nullptr;  // Optional IF/ELSE false body; null means skip on reuse.
    CUdeviceptr decision_record = 0u;  // Both records must be 16-byte aligned.
    CUdeviceptr request_record = 0u;
  };

  enum class build_failure_e {
    none,
    invalid_descriptor,
    driver_api_unavailable,
    context_query_failed,
    context_switch_failed,
    infer_child_rejected,
    optional_infer_child_rejected,
    infer_child_missing_inference_kernel,
    optional_infer_child_missing_inference_kernel,
    reuse_child_rejected,
    module_load_failed,
    module_function_missing,
    graph_create_failed,
    conditional_handle_create_failed,
    setter_node_add_failed,
    conditional_node_add_failed,
    conditional_body_missing,
    infer_child_scalar_prefix_rejected,
    optional_infer_child_scalar_prefix_rejected,
    infer_child_scalar_mirror_failed,
    infer_child_scalar_parent_copy_failed,
    infer_child_embedded_scalar_prefix_rejected,
    optional_infer_child_embedded_scalar_prefix_rejected,
    infer_child_scalar_rewrite_failed,
    optional_infer_child_scalar_rewrite_failed,
    infer_child_add_failed,
    optional_infer_child_add_failed,
    optional_handle_create_failed,
    optional_conditional_node_add_failed,
    optional_conditional_body_missing,
    reuse_child_add_failed,
    instantiate_failed,
    context_restore_failed,
  };

  /** Owns the PTX module, wrapper graph, and instantiated conditional executable.
   *
   * Build failures return an object with ready()==false; this neutral helper does not select the
   * caller's recovery policy. During an active GPU-adaptive transaction, construction or launch
   * failure is terminal for that pipeline and the caller must not retry the raw inference child.
   * Neutral helper users may define a different policy, but production DAV2 never bypasses the
   * wrapper after its private bootstrap. The caller owns
   * both input child graphs. The decision/request CUdeviceptrs are captured by value and must stay
   * mapped and unchanged through each launch, completion, and interop-unmap tail. A later remap
   * must reproduce those captured addresses; rebuild before launch if it does not. For the fixed
   * DAV2 graph, two TensorRT-owned pageable scalar
   * copies are validated in the source graph. cuGraphAddChildGraphNode clones that source into its
   * node-owned embedded graph; only that embedded graph is rewritten in place to read device
   * mirrors, and the wrapper refreshes those mirrors outside the conditional on every launch.
   * The source graph and any raw CUgraphExec instantiated from it remain unmodified and independent.
   * Those outer refresh nodes capture the original two pageable source addresses, so their owning
   * TensorRT context/storage must outlive this executable and keep both addresses stable.
   *
   * The caller must quiesce every stream that can launch this executable before reset() or
   * destruction. Destroy this object before its CUDA context is released.
   */
  class executable_t {
  public:
    executable_t() = default;
    ~executable_t();

    executable_t(const executable_t &) = delete;
    executable_t &operator=(const executable_t &) = delete;
    executable_t(executable_t &&other) noexcept;
    executable_t &operator=(executable_t &&other) noexcept;

    [[nodiscard]] static executable_t build(
      cuda_driver_api &cuda,
      const build_desc_t &desc
    ) noexcept;

    [[nodiscard]] bool ready() const noexcept {
      return executable_ != nullptr && failure_ == build_failure_e::none;
    }

    [[nodiscard]] CUgraphExec get() const noexcept {
      return executable_;
    }

    [[nodiscard]] build_failure_e failure() const noexcept {
      return failure_;
    }

    [[nodiscard]] CUresult cuda_result() const noexcept {
      return cuda_result_;
    }

    [[nodiscard]] const audit_result_t &audit_result() const noexcept {
      return audit_result_;
    }

    /** Releases every CUDA object owned by the wrapper.
     *
     * Returns false when the owning context could not be made current/restored or a wrapper object
     * could not be destroyed. Mirrors remain owned whenever the parent graph cannot be destroyed,
     * because its embedded child may still reference them. The caller must quiesce all wrapper
     * launch streams before calling this function.
     */
    [[nodiscard]] bool reset() noexcept;

    /** Relinquishes bookkeeping without issuing any CUDA call.
     *
     * This is the last-resort process-lifetime quarantine for a context or launch stream whose
     * quiescence cannot be proved. Every owned CUDA allocation/object is intentionally leaked;
     * callers must likewise retain all graph operands and registrations. After this call the
     * value is empty, so its destructor cannot retry unsafe teardown.
     */
    void abandon_unsafe() noexcept;

    [[nodiscard]] bool empty() const noexcept {
      return module_ == nullptr && graph_ == nullptr && executable_ == nullptr &&
             scalar_mirrors_[0] == 0u && scalar_mirrors_[1] == 0u &&
             scalar_mirrors_[2] == 0u && scalar_mirrors_[3] == 0u;
    }

  private:
    [[nodiscard]] bool destroy_current_context() noexcept;
    void move_from(executable_t &&other) noexcept;

    cuda_driver_api *cuda_ = nullptr;
    CUcontext context_ = nullptr;
    CUmodule module_ = nullptr;
    CUgraph graph_ = nullptr;
    CUgraphExec executable_ = nullptr;
    // The fixed DAV2 and optional fixed OCR TensorRT graphs each own at most the two captured
    // pageable scalar-prefix copies validated by build(). Their device mirrors live outside the
    // conditional body and are refreshed before its setter node on every root launch.
    CUdeviceptr scalar_mirrors_[4] {};
    build_failure_e failure_ = build_failure_e::invalid_descriptor;
    CUresult cuda_result_ = CUDA_SUCCESS;
    audit_result_t audit_result_ {};
  };

}  // namespace cuda_conditional_graph
