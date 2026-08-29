#include "cuda_conditional_graph.h"

#include <algorithm>
#include <array>
#include <limits>
#include <utility>
#include <vector>

namespace cuda_conditional_graph {
  namespace {

    // Hand-written PTX keeps the bridge reproducible on hosts that have only the NVIDIA driver.
    // nvcuda.dll resolves cudaGraphSetConditional as a device builtin during module JIT. The
    // request pointer, rather than a captured scalar token, makes one graph reusable across frames.
    constexpr char conditional_bridge_ptx[] = R"ptx(
.version 8.4
.target sm_70
.address_size 64

.extern .func cudaGraphSetConditional(
    .param .b64 cudaGraphSetConditional_param_0,
    .param .b32 cudaGraphSetConditional_param_1
);

.visible .entry set_condition_from_records(
    .param .u64 set_condition_param_0,
    .param .u64 set_condition_param_1,
    .param .u64 set_condition_param_2,
    .param .u64 set_condition_param_3,
    .param .u32 set_condition_param_4
)
{
    .reg .pred %p<32>;
    .reg .b32 %r<40>;
    .reg .b64 %rd<5>;

    ld.param.u64 %rd1, [set_condition_param_0];
    ld.param.u64 %rd2, [set_condition_param_1];
    ld.param.u64 %rd3, [set_condition_param_2];
    ld.param.u64 %rd4, [set_condition_param_3];
    ld.param.u32 %r31, [set_condition_param_4];

    ld.global.v4.u32 {%r1, %r2, %r3, %r4}, [%rd3];
    ld.global.v4.u32 {%r5, %r6, %r7, %r8}, [%rd3+16];
    ld.global.v4.u32 {%r9, %r10, %r11, %r12}, [%rd4];
    ld.global.v4.u32 {%r13, %r14, %r15, %r16}, [%rd4+16];

    xor.b32 %r17, %r1, 0xd1ec15a5;
    xor.b32 %r18, %r3, 0xa3756c91;
    xor.b32 %r19, %r4, 0x5c8a936e;
    xor.b32 %r20, %r9, 0xa3756c91;
    xor.b32 %r21, %r10, 0x5c8a936e;
    xor.b32 %r26, %r14, 0x6f435257;
    or.b32 %r32, %r9, %r10;

    setp.eq.u32 %p1, %r11, %r20;
    setp.eq.u32 %p2, %r12, %r21;
    setp.eq.u32 %p3, %r13, 0x54535152;
    setp.eq.u32 %p4, %r14, 0;
    setp.eq.u32 %p24, %r14, 1;
    setp.eq.u32 %p28, %r14, 2;
    setp.eq.u32 %p29, %r14, 8;
    setp.eq.u32 %p30, %r14, 16;
    or.pred %p4, %p4, %p24;
    or.pred %p4, %p4, %p28;
    or.pred %p4, %p4, %p29;
    or.pred %p4, %p4, %p30;
    setp.eq.u32 %p18, %r14, 0;
    selp.u32 %r27, 0, %r26, %p18;
    setp.eq.u32 %p5, %r15, %r27;
    setp.eq.u32 %p6, %r16, 0;
    setp.ne.u32 %p23, %r32, 0;
    and.pred %p7, %p1, %p2;
    and.pred %p7, %p7, %p3;
    and.pred %p7, %p7, %p4;
    and.pred %p7, %p7, %p5;
    and.pred %p7, %p7, %p6;
    and.pred %p7, %p7, %p23;

    setp.le.u32 %p8, %r1, 1;
    setp.eq.u32 %p9, %r2, %r17;
    setp.eq.u32 %p10, %r3, %r9;
    setp.eq.u32 %p11, %r4, %r10;
    setp.eq.u32 %p12, %r5, %r18;
    setp.eq.u32 %p13, %r6, %r19;
    setp.eq.u32 %p14, %r7, 0x504f5250;
    setp.eq.u32 %p15, %r8, 0;
    and.pred %p16, %p8, %p9;
    and.pred %p16, %p16, %p10;
    and.pred %p16, %p16, %p11;
    and.pred %p16, %p16, %p12;
    and.pred %p16, %p16, %p13;
    and.pred %p16, %p16, %p14;
    and.pred %p16, %p16, %p15;

    and.pred %p17, %p7, %p16;
    selp.u32 %r22, %r1, 1, %p17;
    xor.b32 %r23, %r22, 0xd1ec15a5;

    setp.eq.u32 %p20, %r14, 1;
    setp.eq.u32 %p26, %r14, 8;
    setp.eq.u32 %p22, %r31, 1;
    setp.eq.u32 %p25, %r22, 1;
    and.pred %p27, %p20, %p25;
    or.pred %p27, %p27, %p26;
    and.pred %p21, %p17, %p27;
    and.pred %p21, %p21, %p22;
    selp.u32 %r28, 0x52434f4f, 0, %p21;
    selp.u32 %r29, 1, 0, %p21;
    xor.b32 %r30, %r23, %r28;

    // PROP is not a consumable receipt. Invalidate its tag first, write the resolved record, and
    // publish CBRG last. A valid request plus any malformed proposal produces an infer receipt,
    // but only a valid preprocess-ready proposal may arm ordinary infer OCR or cadence-due OCR.
    st.global.u32 [%rd3+24], 0;
    membar.gl;
    st.global.v4.u32 [%rd3], {%r22, %r30, %r9, %r10};
    st.global.v2.u32 [%rd3+16], {%r20, %r21};
    st.global.u32 [%rd3+28], %r28;
    membar.gl;
    mov.u32 %r24, 0x47524243;
    st.global.u32 [%rd3+24], %r24;
    membar.gl;

    {
        .param .b64 call_param_0;
        .param .b32 call_param_1;
        st.param.b64 [call_param_0], %rd1;
        st.param.b32 [call_param_1], %r22;
        call.uni cudaGraphSetConditional, (call_param_0, call_param_1);
    }
    {
        .param .b64 call_param_0;
        .param .b32 call_param_1;
        st.param.b64 [call_param_0], %rd2;
        st.param.b32 [call_param_1], %r29;
        call.uni cudaGraphSetConditional, (call_param_0, call_param_1);
    }
    ret;
}
)ptx";

    constexpr std::size_t max_audit_depth = 64u;
    constexpr std::size_t tensor_rt_scalar_copy_count = 2u;
    constexpr std::size_t tensor_rt_scalar_copy_bytes = 8u;

    enum class scalar_prefix_result_e {
      absent,
      ready,
      rejected,
      cuda_error,
      host_allocation_failed,
    };

    struct scalar_prefix_t {
      scalar_prefix_result_e result = scalar_prefix_result_e::absent;
      CUresult cuda_result = CUDA_SUCCESS;
      std::array<CUgraphNode, tensor_rt_scalar_copy_count> nodes {};
      std::array<CUDA_MEMCPY3D, tensor_rt_scalar_copy_count> params {};
    };

    bool exact_tensor_rt_scalar_copy(const CUDA_MEMCPY3D &copy) noexcept {
      return copy.srcXInBytes == 0u && copy.srcY == 0u && copy.srcZ == 0u &&
             copy.srcLOD == 0u && copy.srcMemoryType == CU_MEMORYTYPE_UNIFIED &&
             copy.srcHost == nullptr && copy.srcDevice != 0u && copy.srcArray == nullptr &&
             copy.reserved0 == nullptr && copy.srcPitch == 0u && copy.srcHeight == 0u &&
             copy.dstXInBytes == 0u && copy.dstY == 0u && copy.dstZ == 0u &&
             copy.dstLOD == 0u && copy.dstMemoryType == CU_MEMORYTYPE_UNIFIED &&
             copy.dstHost == nullptr && copy.dstDevice != 0u && copy.dstArray == nullptr &&
             copy.reserved1 == nullptr && copy.dstPitch == 0u && copy.dstHeight == 0u &&
             copy.WidthInBytes == tensor_rt_scalar_copy_bytes && copy.Height == 1u &&
             copy.Depth == 1u;
    }

    bool disjoint_scalar_ranges(const CUdeviceptr first, const CUdeviceptr second) noexcept {
      const CUdeviceptr distance = first > second ? first - second : second - first;
      return distance >= tensor_rt_scalar_copy_bytes;
    }

    scalar_prefix_t inspect_tensor_rt_scalar_prefix(
      const cuda_driver_api &cuda,
      CUgraph graph
    ) noexcept {
      scalar_prefix_t result;
      if (!cuda.cuGraphGetNodes || !cuda.cuGraphGetRootNodes ||
          (!cuda.cuGraphNodeGetDependencies_v2 && !cuda.cuGraphNodeGetDependencies) ||
          (!cuda.cuGraphNodeGetDependentNodes_v2 && !cuda.cuGraphNodeGetDependentNodes) ||
          !cuda.cuGraphNodeGetType || !cuda.cuGraphMemcpyNodeGetParams) {
        result.result = scalar_prefix_result_e::cuda_error;
        result.cuda_result = CUDA_ERROR_INVALID_HANDLE;
        return result;
      }

      try {
        std::size_t node_count = 0u;
        result.cuda_result = cuda.cuGraphGetNodes(graph, nullptr, &node_count);
        if (result.cuda_result != CUDA_SUCCESS) {
          result.result = scalar_prefix_result_e::cuda_error;
          return result;
        }
        std::vector<CUgraphNode> nodes(node_count);
        if (node_count != 0u) {
          result.cuda_result = cuda.cuGraphGetNodes(graph, nodes.data(), &node_count);
          if (result.cuda_result != CUDA_SUCCESS || node_count > nodes.size()) {
            result.result = scalar_prefix_result_e::cuda_error;
            return result;
          }
          nodes.resize(node_count);
        }

        std::array<CUgraphNode, tensor_rt_scalar_copy_count> copies {};
        std::size_t copy_count = 0u;
        for (const CUgraphNode node : nodes) {
          CUgraphNodeType type = static_cast<CUgraphNodeType>(-1);
          result.cuda_result = cuda.cuGraphNodeGetType(node, &type);
          if (result.cuda_result != CUDA_SUCCESS) {
            result.result = scalar_prefix_result_e::cuda_error;
            return result;
          }
          if (type == CU_GRAPH_NODE_TYPE_MEMCPY) {
            if (copy_count >= copies.size()) {
              result.result = scalar_prefix_result_e::rejected;
              return result;
            }
            copies[copy_count++] = node;
          }
        }
        if (copy_count == 0u) {
          result.result = scalar_prefix_result_e::absent;
          return result;
        }
        if (copy_count != copies.size()) {
          result.result = scalar_prefix_result_e::rejected;
          return result;
        }

        std::size_t root_count = 0u;
        result.cuda_result = cuda.cuGraphGetRootNodes(graph, nullptr, &root_count);
        if (result.cuda_result != CUDA_SUCCESS) {
          result.result = scalar_prefix_result_e::cuda_error;
          return result;
        }
        if (root_count != 1u) {
          result.result = scalar_prefix_result_e::rejected;
          return result;
        }
        CUgraphNode first = nullptr;
        result.cuda_result = cuda.cuGraphGetRootNodes(graph, &first, &root_count);
        if (result.cuda_result != CUDA_SUCCESS || root_count != 1u ||
            (first != copies[0] && first != copies[1])) {
          result.result = result.cuda_result == CUDA_SUCCESS ?
                            scalar_prefix_result_e::rejected :
                            scalar_prefix_result_e::cuda_error;
          return result;
        }

        std::size_t dependency_count = 0u;
        result.cuda_result = cuda.graph_node_get_dependencies(
          first, nullptr, &dependency_count
        );
        if (result.cuda_result != CUDA_SUCCESS || dependency_count != 0u) {
          result.result = result.cuda_result == CUDA_SUCCESS ?
                            scalar_prefix_result_e::rejected :
                            scalar_prefix_result_e::cuda_error;
          return result;
        }

        std::size_t dependent_count = 0u;
        result.cuda_result = cuda.graph_node_get_dependents(first, nullptr, &dependent_count);
        if (result.cuda_result != CUDA_SUCCESS || dependent_count != 1u) {
          result.result = result.cuda_result == CUDA_SUCCESS ?
                            scalar_prefix_result_e::rejected :
                            scalar_prefix_result_e::cuda_error;
          return result;
        }
        CUgraphNode second = nullptr;
        result.cuda_result = cuda.graph_node_get_dependents(
          first, &second, &dependent_count
        );
        if (result.cuda_result != CUDA_SUCCESS || dependent_count != 1u ||
            (second != copies[0] && second != copies[1]) || second == first) {
          result.result = result.cuda_result == CUDA_SUCCESS ?
                            scalar_prefix_result_e::rejected :
                            scalar_prefix_result_e::cuda_error;
          return result;
        }

        dependency_count = 0u;
        result.cuda_result = cuda.graph_node_get_dependencies(
          second, nullptr, &dependency_count
        );
        if (result.cuda_result != CUDA_SUCCESS || dependency_count != 1u) {
          result.result = result.cuda_result == CUDA_SUCCESS ?
                            scalar_prefix_result_e::rejected :
                            scalar_prefix_result_e::cuda_error;
          return result;
        }
        CUgraphNode second_dependency = nullptr;
        result.cuda_result = cuda.graph_node_get_dependencies(
          second, &second_dependency, &dependency_count
        );
        if (result.cuda_result != CUDA_SUCCESS || dependency_count != 1u ||
            second_dependency != first) {
          result.result = result.cuda_result == CUDA_SUCCESS ?
                            scalar_prefix_result_e::rejected :
                            scalar_prefix_result_e::cuda_error;
          return result;
        }

        dependent_count = 0u;
        result.cuda_result = cuda.graph_node_get_dependents(second, nullptr, &dependent_count);
        if (result.cuda_result != CUDA_SUCCESS || dependent_count != 1u) {
          result.result = result.cuda_result == CUDA_SUCCESS ?
                            scalar_prefix_result_e::rejected :
                            scalar_prefix_result_e::cuda_error;
          return result;
        }
        CUgraphNode first_kernel = nullptr;
        result.cuda_result = cuda.graph_node_get_dependents(
          second, &first_kernel, &dependent_count
        );
        CUgraphNodeType first_kernel_type = static_cast<CUgraphNodeType>(-1);
        if (result.cuda_result == CUDA_SUCCESS) {
          result.cuda_result = cuda.cuGraphNodeGetType(first_kernel, &first_kernel_type);
        }
        if (result.cuda_result != CUDA_SUCCESS || dependent_count != 1u ||
            first_kernel_type != CU_GRAPH_NODE_TYPE_KERNEL) {
          result.result = result.cuda_result == CUDA_SUCCESS ?
                            scalar_prefix_result_e::rejected :
                            scalar_prefix_result_e::cuda_error;
          return result;
        }

        dependency_count = 0u;
        result.cuda_result = cuda.graph_node_get_dependencies(
          first_kernel, nullptr, &dependency_count
        );
        if (result.cuda_result != CUDA_SUCCESS || dependency_count != 1u) {
          result.result = result.cuda_result == CUDA_SUCCESS ?
                            scalar_prefix_result_e::rejected :
                            scalar_prefix_result_e::cuda_error;
          return result;
        }
        CUgraphNode kernel_dependency = nullptr;
        result.cuda_result = cuda.graph_node_get_dependencies(
          first_kernel, &kernel_dependency, &dependency_count
        );
        if (result.cuda_result != CUDA_SUCCESS || dependency_count != 1u ||
            kernel_dependency != second) {
          result.result = result.cuda_result == CUDA_SUCCESS ?
                            scalar_prefix_result_e::rejected :
                            scalar_prefix_result_e::cuda_error;
          return result;
        }

        result.nodes = {first, second};
        for (std::size_t i = 0u; i < result.nodes.size(); ++i) {
          result.cuda_result = cuda.cuGraphMemcpyNodeGetParams(
            result.nodes[i], &result.params[i]
          );
          if (result.cuda_result != CUDA_SUCCESS) {
            result.result = scalar_prefix_result_e::cuda_error;
            return result;
          }
          if (!exact_tensor_rt_scalar_copy(result.params[i])) {
            result.result = scalar_prefix_result_e::rejected;
            return result;
          }
        }
        if (!disjoint_scalar_ranges(
              result.params[0].srcDevice, result.params[1].srcDevice
            ) ||
            !disjoint_scalar_ranges(
              result.params[0].dstDevice, result.params[1].dstDevice
            )) {
          result.result = scalar_prefix_result_e::rejected;
          return result;
        }
        for (const CUDA_MEMCPY3D &source : result.params) {
          for (const CUDA_MEMCPY3D &destination : result.params) {
            if (!disjoint_scalar_ranges(source.srcDevice, destination.dstDevice)) {
              result.result = scalar_prefix_result_e::rejected;
              return result;
            }
          }
        }

        result.result = scalar_prefix_result_e::ready;
        return result;
      } catch (...) {
        result.result = scalar_prefix_result_e::host_allocation_failed;
        return result;
      }
    }

    CUDA_MEMCPY3D parent_scalar_copy(
      const CUDA_MEMCPY3D &captured,
      const CUdeviceptr mirror
    ) noexcept {
      CUDA_MEMCPY3D result = captured;
      result.dstMemoryType = CU_MEMORYTYPE_DEVICE;
      result.dstHost = nullptr;
      result.dstDevice = mirror;
      result.dstArray = nullptr;
      result.reserved1 = nullptr;
      result.dstPitch = 0u;
      result.dstHeight = 0u;
      return result;
    }

    CUDA_MEMCPY3D mirrored_child_scalar_copy(
      const CUDA_MEMCPY3D &captured,
      const CUdeviceptr mirror
    ) noexcept {
      CUDA_MEMCPY3D result = captured;
      result.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      result.srcHost = nullptr;
      result.srcDevice = mirror;
      result.srcArray = nullptr;
      result.reserved0 = nullptr;
      result.srcPitch = 0u;
      result.srcHeight = 0u;
      return result;
    }

    bool audit_graph_impl(
      const cuda_driver_api &cuda,
      CUgraph graph,
      const std::size_t depth,
      std::vector<CUgraph> &visited,
      audit_result_t &result
    ) {
      if (depth > max_audit_depth) {
        result.failure = audit_failure_e::recursion_limit;
        return false;
      }
      if (std::find(visited.begin(), visited.end(), graph) != visited.end()) {
        return true;
      }
      visited.push_back(graph);
      ++result.visited_graphs;

      std::size_t node_count = 0u;
      result.cuda_result = cuda.cuGraphGetNodes(graph, nullptr, &node_count);
      if (result.cuda_result != CUDA_SUCCESS) {
        result.failure = audit_failure_e::cuda_error;
        return false;
      }
      std::vector<CUgraphNode> nodes(node_count);
      if (node_count != 0u) {
        result.cuda_result = cuda.cuGraphGetNodes(graph, nodes.data(), &node_count);
        if (result.cuda_result != CUDA_SUCCESS || node_count > nodes.size()) {
          result.failure = audit_failure_e::cuda_error;
          return false;
        }
        nodes.resize(node_count);
      }

      result.visited_nodes += nodes.size();
      for (const CUgraphNode node : nodes) {
        CUgraphNodeType type = static_cast<CUgraphNodeType>(-1);
        result.cuda_result = cuda.cuGraphNodeGetType(node, &type);
        if (result.cuda_result != CUDA_SUCCESS) {
          result.failure = audit_failure_e::cuda_error;
          return false;
        }
        const auto type_index = static_cast<std::size_t>(type);
        if (type_index < result.node_type_counts.size()) {
          ++result.node_type_counts[type_index];
        }
        if (!is_conditional_body_node_type_allowed(type)) {
          if (result.failure == audit_failure_e::none) {
            result.failure = audit_failure_e::unsupported_node_type;
            result.rejected_type = type;
          }
          continue;
        }
        // CUDA graphs containing conditional nodes cannot themselves be child-graph nodes.
        if (type == CU_GRAPH_NODE_TYPE_CONDITIONAL) {
          if (result.failure == audit_failure_e::none) {
            result.failure = audit_failure_e::nested_conditional;
            result.rejected_type = type;
          }
          continue;
        }
        if (type == CU_GRAPH_NODE_TYPE_GRAPH) {
          CUgraph child = nullptr;
          result.cuda_result = cuda.cuGraphChildGraphNodeGetGraph(node, &child);
          if (result.cuda_result != CUDA_SUCCESS || child == nullptr) {
            result.failure = audit_failure_e::cuda_error;
            return false;
          }
          if (!audit_graph_impl(cuda, child, depth + 1u, visited, result) &&
              result.failure != audit_failure_e::unsupported_node_type &&
              result.failure != audit_failure_e::nested_conditional) {
            return false;
          }
        }
      }
      return result.failure == audit_failure_e::none;
    }

  }  // namespace

  std::string_view bridge_ptx_source() noexcept {
    return conditional_bridge_ptx;
  }

  audit_result_t audit_embeddable_child_graph(
    const cuda_driver_api &cuda,
    CUgraph graph
  ) noexcept {
    audit_result_t result {};
    if (graph == nullptr) {
      result.failure = audit_failure_e::invalid_graph;
      return result;
    }
    if (!cuda.cuGraphGetNodes || !cuda.cuGraphNodeGetType ||
        !cuda.cuGraphChildGraphNodeGetGraph) {
      result.failure = audit_failure_e::driver_api_unavailable;
      return result;
    }
    try {
      std::vector<CUgraph> visited;
      audit_graph_impl(cuda, graph, 0u, visited, result);
    } catch (...) {
      result.failure = audit_failure_e::host_allocation_failed;
    }
    return result;
  }

  audit_result_t audit_inference_child_graph(
    const cuda_driver_api &cuda,
    CUgraph graph
  ) noexcept {
    auto result = audit_embeddable_child_graph(cuda, graph);
    if (result.legal() &&
        result.node_type_counts[CU_GRAPH_NODE_TYPE_KERNEL] == 0u) {
      result.failure = audit_failure_e::missing_inference_kernel;
    }
    return result;
  }

  executable_t::~executable_t() {
    (void) reset();
  }

  executable_t::executable_t(executable_t &&other) noexcept {
    move_from(std::move(other));
  }

  bool executable_t::adopt_from_empty(executable_t &&other) noexcept {
    if (this == &other) {
      return true;
    }
    if (!empty()) {
      return false;
    }
    move_from(std::move(other));
    return true;
  }

  void executable_t::move_from(executable_t &&other) noexcept {
    cuda_ = std::exchange(other.cuda_, nullptr);
    context_ = std::exchange(other.context_, nullptr);
    module_ = std::exchange(other.module_, nullptr);
    graph_ = std::exchange(other.graph_, nullptr);
    executable_ = std::exchange(other.executable_, nullptr);
    scalar_mirrors_[0] = std::exchange(other.scalar_mirrors_[0], 0u);
    scalar_mirrors_[1] = std::exchange(other.scalar_mirrors_[1], 0u);
    scalar_mirrors_[2] = std::exchange(other.scalar_mirrors_[2], 0u);
    scalar_mirrors_[3] = std::exchange(other.scalar_mirrors_[3], 0u);
    failure_ = std::exchange(other.failure_, build_failure_e::invalid_descriptor);
    cuda_result_ = std::exchange(other.cuda_result_, CUDA_SUCCESS);
    audit_result_ = other.audit_result_;
    other.audit_result_ = {};
  }

  bool executable_t::destroy_current_context() noexcept {
    if (!cuda_) {
      return empty();
    }
    if (executable_) {
      if (!cuda_->cuGraphExecDestroy ||
          cuda_->cuGraphExecDestroy(executable_) != CUDA_SUCCESS) {
        return false;
      }
      executable_ = nullptr;
    }
    if (graph_) {
      if (!cuda_->cuGraphDestroy ||
          cuda_->cuGraphDestroy(graph_) != CUDA_SUCCESS) {
        return false;
      }
      graph_ = nullptr;
    }
    if (module_) {
      if (!cuda_->cuModuleUnload ||
          cuda_->cuModuleUnload(module_) != CUDA_SUCCESS) {
        return false;
      }
      module_ = nullptr;
    }
    for (CUdeviceptr &mirror : scalar_mirrors_) {
      if (mirror != 0u) {
        if (!cuda_->cuMemFree || cuda_->cuMemFree(mirror) != CUDA_SUCCESS) {
          return false;
        }
        mirror = 0u;
      }
    }
    return true;
  }

  bool executable_t::reset() noexcept {
    if (!cuda_ || empty()) {
      return empty();
    }
    CUcontext previous = nullptr;
    if (!cuda_->cuCtxGetCurrent ||
        cuda_->cuCtxGetCurrent(&previous) != CUDA_SUCCESS) {
      return false;
    }
    const bool switched = previous != context_;
    if (switched && (!cuda_->cuCtxSetCurrent ||
                     cuda_->cuCtxSetCurrent(context_) != CUDA_SUCCESS)) {
      return false;
    }
    const bool destroyed = destroy_current_context();
    bool restored = true;
    if (switched) {
      restored = cuda_->cuCtxSetCurrent(previous) == CUDA_SUCCESS;
    }
    return destroyed && restored;
  }

  void executable_t::abandon_unsafe() noexcept {
    cuda_ = nullptr;
    context_ = nullptr;
    module_ = nullptr;
    graph_ = nullptr;
    executable_ = nullptr;
    for (CUdeviceptr &mirror : scalar_mirrors_) {
      mirror = 0u;
    }
    failure_ = build_failure_e::invalid_descriptor;
    cuda_result_ = CUDA_SUCCESS;
    audit_result_ = {};
  }

  executable_t executable_t::build(
    cuda_driver_api &cuda,
    const build_desc_t &desc
  ) noexcept {
    executable_t output;
    output.cuda_ = &cuda;
    output.context_ = desc.context;
    output.failure_ = build_failure_e::invalid_descriptor;

    const auto record_fits_address_space = [](const CUdeviceptr address) {
      return address != 0u &&
             address <= std::numeric_limits<CUdeviceptr>::max() -
                          (sizeof(decision_record_t) - 1u);
    };
    const auto record_distance = desc.decision_record > desc.request_record ?
                                   desc.decision_record - desc.request_record :
                                   desc.request_record - desc.decision_record;
    if (!desc.context || !desc.infer_child || !desc.decision_record ||
        !desc.request_record || (desc.decision_record & 15u) != 0u ||
        (desc.request_record & 15u) != 0u ||
        !record_fits_address_space(desc.decision_record) ||
        !record_fits_address_space(desc.request_record) ||
        record_distance < sizeof(decision_record_t)) {
      return output;
    }
    if (!cuda.has_conditional_graph_support()) {
      output.failure_ = build_failure_e::driver_api_unavailable;
      return output;
    }

    CUcontext previous = nullptr;
    output.cuda_result_ = cuda.cuCtxGetCurrent(&previous);
    if (output.cuda_result_ != CUDA_SUCCESS) {
      output.failure_ = build_failure_e::context_query_failed;
      return output;
    }
    const bool switched = previous != desc.context;
    if (switched) {
      output.cuda_result_ = cuda.cuCtxSetCurrent(desc.context);
      if (output.cuda_result_ != CUDA_SUCCESS) {
        output.failure_ = build_failure_e::context_switch_failed;
        return output;
      }
    }

    const auto finish = [&](const bool success) -> executable_t {
      if (!success) {
        (void) output.destroy_current_context();
      }
      if (switched) {
        const CUresult restore_result = cuda.cuCtxSetCurrent(previous);
        if (restore_result != CUDA_SUCCESS) {
          if (success) {
            // A failed restoration normally leaves the target context current, so resources can
            // still be torn down safely. Never return a live graph after violating context state.
            (void) output.destroy_current_context();
          }
          output.failure_ = build_failure_e::context_restore_failed;
          output.cuda_result_ = restore_result;
        }
      }
      return std::move(output);
    };

    output.audit_result_ = audit_inference_child_graph(cuda, desc.infer_child);
    if (!output.audit_result_.legal()) {
      output.failure_ =
        output.audit_result_.failure == audit_failure_e::missing_inference_kernel ?
          build_failure_e::infer_child_missing_inference_kernel :
          build_failure_e::infer_child_rejected;
      output.cuda_result_ = output.audit_result_.cuda_result;
      return finish(false);
    }
    if (desc.optional_infer_child) {
      output.audit_result_ = audit_inference_child_graph(
        cuda, desc.optional_infer_child
      );
      if (!output.audit_result_.legal()) {
        output.failure_ =
          output.audit_result_.failure == audit_failure_e::missing_inference_kernel ?
            build_failure_e::optional_infer_child_missing_inference_kernel :
            build_failure_e::optional_infer_child_rejected;
        output.cuda_result_ = output.audit_result_.cuda_result;
        return finish(false);
      }
    }
    if (desc.reuse_child) {
      output.audit_result_ = audit_embeddable_child_graph(cuda, desc.reuse_child);
      if (!output.audit_result_.legal()) {
        output.failure_ = build_failure_e::reuse_child_rejected;
        output.cuda_result_ = output.audit_result_.cuda_result;
        return finish(false);
      }
    }

    const std::array<CUgraph, 2u> infer_children {
      desc.infer_child,
      desc.optional_infer_child,
    };
    const std::size_t infer_child_count = desc.optional_infer_child ? 2u : 1u;
    std::array<scalar_prefix_t, 2u> scalar_prefixes {};
    std::array<std::size_t, 2u> scalar_mirror_bases {};
    std::size_t scalar_mirror_count = 0u;
    for (std::size_t child_index = 0u; child_index < infer_child_count; ++child_index) {
      scalar_prefixes[child_index] = inspect_tensor_rt_scalar_prefix(
        cuda, infer_children[child_index]
      );
      const auto result = scalar_prefixes[child_index].result;
      if (result == scalar_prefix_result_e::rejected ||
          result == scalar_prefix_result_e::host_allocation_failed ||
          result == scalar_prefix_result_e::cuda_error) {
        output.failure_ = child_index == 0u ?
                            build_failure_e::infer_child_scalar_prefix_rejected :
                            build_failure_e::optional_infer_child_scalar_prefix_rejected;
        output.cuda_result_ = scalar_prefixes[child_index].cuda_result;
        return finish(false);
      }
      scalar_mirror_bases[child_index] = scalar_mirror_count;
      if (result == scalar_prefix_result_e::ready) {
        scalar_mirror_count += tensor_rt_scalar_copy_count;
      }
    }
    if (scalar_mirror_count != 0u &&
        (!cuda.cuGraphMemcpyNodeSetParams || !cuda.cuGraphAddMemcpyNode ||
         !cuda.cuMemAlloc || !cuda.cuMemFree)) {
      output.failure_ = build_failure_e::infer_child_scalar_prefix_rejected;
      output.cuda_result_ = CUDA_ERROR_INVALID_HANDLE;
      return finish(false);
    }

    output.cuda_result_ = cuda.cuModuleLoadDataEx(
      &output.module_, conditional_bridge_ptx, 0u, nullptr, nullptr
    );
    if (output.cuda_result_ != CUDA_SUCCESS) {
      output.failure_ = build_failure_e::module_load_failed;
      return finish(false);
    }

    CUfunction setter = nullptr;
    output.cuda_result_ = cuda.cuModuleGetFunction(
      &setter, output.module_, "set_condition_from_records"
    );
    if (output.cuda_result_ != CUDA_SUCCESS || !setter) {
      output.failure_ = build_failure_e::module_function_missing;
      return finish(false);
    }

    output.cuda_result_ = cuda.cuGraphCreate(&output.graph_, 0u);
    if (output.cuda_result_ != CUDA_SUCCESS) {
      output.failure_ = build_failure_e::graph_create_failed;
      return finish(false);
    }

    CUgraphNode scalar_copy_tail = nullptr;
    for (std::size_t child_index = 0u; child_index < infer_child_count; ++child_index) {
      const auto &scalar_prefix = scalar_prefixes[child_index];
      if (scalar_prefix.result != scalar_prefix_result_e::ready) {
        continue;
      }
      for (std::size_t i = 0u; i < tensor_rt_scalar_copy_count; ++i) {
        const std::size_t mirror_index = scalar_mirror_bases[child_index] + i;
        output.cuda_result_ = cuda.cuMemAlloc(
          &output.scalar_mirrors_[mirror_index], tensor_rt_scalar_copy_bytes
        );
        if (output.cuda_result_ != CUDA_SUCCESS ||
            output.scalar_mirrors_[mirror_index] == 0u) {
          output.failure_ = build_failure_e::infer_child_scalar_mirror_failed;
          return finish(false);
        }

        const CUDA_MEMCPY3D parent_copy = parent_scalar_copy(
          scalar_prefix.params[i], output.scalar_mirrors_[mirror_index]
        );
        CUgraphNode parent_copy_node = nullptr;
        const CUgraphNode *dependencies = scalar_copy_tail ? &scalar_copy_tail : nullptr;
        output.cuda_result_ = cuda.cuGraphAddMemcpyNode(
          &parent_copy_node,
          output.graph_,
          dependencies,
          scalar_copy_tail ? 1u : 0u,
          &parent_copy,
          desc.context
        );
        if (output.cuda_result_ != CUDA_SUCCESS || !parent_copy_node) {
          output.failure_ = build_failure_e::infer_child_scalar_parent_copy_failed;
          return finish(false);
        }
        scalar_copy_tail = parent_copy_node;
      }
    }

    CUgraphConditionalHandle handle = 0u;
    output.cuda_result_ = cuda.cuGraphConditionalHandleCreate(
      &handle,
      output.graph_,
      desc.context,
      static_cast<unsigned int>(branch_e::infer),
      CU_GRAPH_COND_ASSIGN_DEFAULT
    );
    if (output.cuda_result_ != CUDA_SUCCESS) {
      output.failure_ = build_failure_e::conditional_handle_create_failed;
      return finish(false);
    }

    CUgraphConditionalHandle optional_handle = 0u;
    output.cuda_result_ = cuda.cuGraphConditionalHandleCreate(
      &optional_handle,
      output.graph_,
      desc.context,
      0u,
      CU_GRAPH_COND_ASSIGN_DEFAULT
    );
    if (output.cuda_result_ != CUDA_SUCCESS) {
      output.failure_ = build_failure_e::optional_handle_create_failed;
      return finish(false);
    }

    CUdeviceptr decision_record = desc.decision_record;
    CUdeviceptr request_record = desc.request_record;
    unsigned int optional_child_present = desc.optional_infer_child ? 1u : 0u;
    void *setter_args[] = {
      &handle,
      &optional_handle,
      &decision_record,
      &request_record,
      &optional_child_present,
    };
    CUDA_KERNEL_NODE_PARAMS setter_params {};
    setter_params.func = setter;
    setter_params.gridDimX = 1u;
    setter_params.gridDimY = 1u;
    setter_params.gridDimZ = 1u;
    setter_params.blockDimX = 1u;
    setter_params.blockDimY = 1u;
    setter_params.blockDimZ = 1u;
    setter_params.kernelParams = setter_args;
    CUgraphNode setter_node = nullptr;
    const CUgraphNode *setter_dependencies = scalar_copy_tail ? &scalar_copy_tail : nullptr;
    output.cuda_result_ = cuda.cuGraphAddKernelNode(
      &setter_node,
      output.graph_,
      setter_dependencies,
      scalar_copy_tail ? 1u : 0u,
      &setter_params
    );
    if (output.cuda_result_ != CUDA_SUCCESS) {
      output.failure_ = build_failure_e::setter_node_add_failed;
      return finish(false);
    }

    CUgraphNodeParams conditional_params {};
    conditional_params.type = CU_GRAPH_NODE_TYPE_CONDITIONAL;
    conditional_params.params.conditional.handle = handle;
    conditional_params.params.conditional.type = CU_GRAPH_COND_TYPE_IF;
    conditional_params.params.conditional.size = desc.reuse_child ? 2u : 1u;
    conditional_params.params.conditional.ctx = desc.context;
    CUgraphNode conditional_node = nullptr;
    output.cuda_result_ = cuda.graph_add_node(
      &conditional_node, output.graph_, &setter_node, 1u, &conditional_params
    );
    if (output.cuda_result_ != CUDA_SUCCESS) {
      output.failure_ = build_failure_e::conditional_node_add_failed;
      return finish(false);
    }

    CUgraph *bodies = conditional_params.params.conditional.phGraph_out;
    if (!bodies || !bodies[0] || (desc.reuse_child && !bodies[1])) {
      output.failure_ = build_failure_e::conditional_body_missing;
      output.cuda_result_ = CUDA_ERROR_INVALID_HANDLE;
      return finish(false);
    }

    CUgraphNodeParams optional_conditional_params {};
    optional_conditional_params.type = CU_GRAPH_NODE_TYPE_CONDITIONAL;
    optional_conditional_params.params.conditional.handle = optional_handle;
    optional_conditional_params.params.conditional.type = CU_GRAPH_COND_TYPE_IF;
    optional_conditional_params.params.conditional.size = 1u;
    optional_conditional_params.params.conditional.ctx = desc.context;
    CUgraphNode optional_conditional_node = nullptr;
    output.cuda_result_ = cuda.graph_add_node(
      &optional_conditional_node,
      output.graph_,
      &setter_node,
      1u,
      &optional_conditional_params
    );
    if (output.cuda_result_ != CUDA_SUCCESS) {
      output.failure_ = build_failure_e::optional_conditional_node_add_failed;
      return finish(false);
    }
    CUgraph *optional_bodies =
      optional_conditional_params.params.conditional.phGraph_out;
    if (!optional_bodies || !optional_bodies[0]) {
      output.failure_ = build_failure_e::optional_conditional_body_missing;
      output.cuda_result_ = CUDA_ERROR_INVALID_HANDLE;
      return finish(false);
    }

    for (std::size_t child_index = 0u; child_index < infer_child_count; ++child_index) {
      CUgraphNode infer_child_node = nullptr;
      output.cuda_result_ = cuda.cuGraphAddChildGraphNode(
        &infer_child_node,
        child_index == 0u ? bodies[0] : optional_bodies[0],
        nullptr,
        0u,
        infer_children[child_index]
      );
      if (output.cuda_result_ != CUDA_SUCCESS || !infer_child_node) {
        output.failure_ = child_index == 0u ?
                            build_failure_e::infer_child_add_failed :
                            build_failure_e::optional_infer_child_add_failed;
        if (output.cuda_result_ == CUDA_SUCCESS) {
          output.cuda_result_ = CUDA_ERROR_INVALID_HANDLE;
        }
        return finish(false);
      }
      const auto &scalar_prefix = scalar_prefixes[child_index];
      if (scalar_prefix.result != scalar_prefix_result_e::ready) {
        continue;
      }
      CUgraph embedded_infer_graph = nullptr;
      output.cuda_result_ = cuda.cuGraphChildGraphNodeGetGraph(
        infer_child_node, &embedded_infer_graph
      );
      if (output.cuda_result_ != CUDA_SUCCESS || !embedded_infer_graph) {
        output.failure_ = child_index == 0u ?
                            build_failure_e::infer_child_embedded_scalar_prefix_rejected :
                            build_failure_e::optional_infer_child_embedded_scalar_prefix_rejected;
        return finish(false);
      }
      const scalar_prefix_t embedded_prefix = inspect_tensor_rt_scalar_prefix(
        cuda, embedded_infer_graph
      );
      if (embedded_prefix.result != scalar_prefix_result_e::ready) {
        output.failure_ = child_index == 0u ?
                            build_failure_e::infer_child_embedded_scalar_prefix_rejected :
                            build_failure_e::optional_infer_child_embedded_scalar_prefix_rejected;
        output.cuda_result_ = embedded_prefix.cuda_result;
        return finish(false);
      }
      for (std::size_t i = 0u; i < tensor_rt_scalar_copy_count; ++i) {
        if (embedded_prefix.params[i].srcDevice != scalar_prefix.params[i].srcDevice ||
            embedded_prefix.params[i].dstDevice != scalar_prefix.params[i].dstDevice) {
          output.failure_ = child_index == 0u ?
                              build_failure_e::infer_child_embedded_scalar_prefix_rejected :
                              build_failure_e::optional_infer_child_embedded_scalar_prefix_rejected;
          output.cuda_result_ = CUDA_SUCCESS;
          return finish(false);
        }
        const CUDA_MEMCPY3D child_copy = mirrored_child_scalar_copy(
          embedded_prefix.params[i],
          output.scalar_mirrors_[scalar_mirror_bases[child_index] + i]
        );
        output.cuda_result_ = cuda.cuGraphMemcpyNodeSetParams(
          embedded_prefix.nodes[i], &child_copy
        );
        if (output.cuda_result_ != CUDA_SUCCESS) {
          output.failure_ = child_index == 0u ?
                              build_failure_e::infer_child_scalar_rewrite_failed :
                              build_failure_e::optional_infer_child_scalar_rewrite_failed;
          return finish(false);
        }
      }
    }
    if (desc.reuse_child) {
      CUgraphNode reuse_child_node = nullptr;
      output.cuda_result_ = cuda.cuGraphAddChildGraphNode(
        &reuse_child_node, bodies[1], nullptr, 0u, desc.reuse_child
      );
      if (output.cuda_result_ != CUDA_SUCCESS) {
        output.failure_ = build_failure_e::reuse_child_add_failed;
        return finish(false);
      }
    }

    output.cuda_result_ = cuda.cuGraphInstantiateWithFlags(
      &output.executable_, output.graph_, 0u
    );
    if (output.cuda_result_ != CUDA_SUCCESS) {
      output.failure_ = build_failure_e::instantiate_failed;
      return finish(false);
    }
    output.failure_ = build_failure_e::none;
    return finish(true);
  }

}  // namespace cuda_conditional_graph
