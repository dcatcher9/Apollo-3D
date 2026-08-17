#include <gtest/gtest.h>

#include "src/cuda_conditional_graph.h"
#include "src/model_manager.h"
#include "src/video_depth_estimator.h"

#include <NvInfer.h>
#include <NvInferPlugin.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

// Complete the opaque CUDA tags for the in-process graph-audit fake. Hardware handles remain
// opaque pointers and are never dereferenced by these definitions.
struct CUgraph_st {
  std::vector<CUgraphNode> nodes;
};

struct CUgraphNode_st {
  CUgraphNodeType type = CU_GRAPH_NODE_TYPE_EMPTY;
  CUgraph child = nullptr;
  std::vector<CUgraphNode> dependencies;
  std::vector<CUgraphNode> dependents;
  CUDA_MEMCPY3D memcpy {};
};

namespace {
  using namespace cuda_conditional_graph;

  constexpr char scalar_prefix_consumer_ptx[] = R"ptx(
.version 8.4
.target sm_70
.address_size 64

.visible .entry consume_grid_scalars(
    .param .u64 height_ptr,
    .param .u64 width_ptr,
    .param .u64 output_ptr
)
{
    .reg .b32 %r<5>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [height_ptr];
    ld.param.u64 %rd2, [width_ptr];
    ld.param.u64 %rd3, [output_ptr];
    ld.global.u32 %r1, [%rd1];
    ld.global.u32 %r2, [%rd2];
    mul.lo.u32 %r3, %r1, 1000;
    add.u32 %r4, %r3, %r2;
    st.global.u32 [%rd3], %r4;
    ret;
}

.visible .entry write_inference_marker(
    .param .u64 output_ptr,
    .param .u32 marker_value
)
{
    .reg .b32 %r1;
    .reg .b64 %rd1;
    ld.param.u64 %rd1, [output_ptr];
    ld.param.u32 %r1, [marker_value];
    st.global.u32 [%rd1], %r1;
    ret;
}
)ptx";

  static_assert(proposal_magic == models::near_identical_proposal_magic);
  static_assert(request_magic == models::near_identical_request_magic);
  static_assert(receipt_magic == models::near_identical_receipt_magic);
  static_assert(decision_cookie == models::near_identical_decision_cookie);
  static_assert(token_low_cookie == models::near_identical_token_low_cookie);
  static_assert(token_high_cookie == models::near_identical_token_high_cookie);
  static_assert(work_flags_value(work_flag_e::optional_ocr_due) == 8u);
  static_assert(work_flags_value(work_flag_e::subtitle_observation_due) == 16u);
  static_assert(models::near_identical_gpu_decision_record_byte_offset == 0u);
  static_assert(models::near_identical_gpu_request_record_byte_offset == 32u);

  CUresult __stdcall fake_graph_get_nodes(
    CUgraph graph,
    CUgraphNode *nodes,
    std::size_t *node_count
  ) {
    if (!graph || !node_count) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    if (!nodes) {
      *node_count = graph->nodes.size();
      return CUDA_SUCCESS;
    }
    if (*node_count < graph->nodes.size()) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    std::copy(graph->nodes.begin(), graph->nodes.end(), nodes);
    *node_count = graph->nodes.size();
    return CUDA_SUCCESS;
  }

  CUresult __stdcall fake_graph_node_get_type(
    CUgraphNode node,
    CUgraphNodeType *type
  ) {
    if (!node || !type) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    *type = node->type;
    return CUDA_SUCCESS;
  }

  CUresult __stdcall fake_graph_get_root_nodes(
    CUgraph graph,
    CUgraphNode *roots,
    std::size_t *root_count
  ) {
    if (!graph || !root_count) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    std::vector<CUgraphNode> discovered;
    for (const CUgraphNode node : graph->nodes) {
      if (node && node->dependencies.empty()) {
        discovered.push_back(node);
      }
    }
    if (!roots) {
      *root_count = discovered.size();
      return CUDA_SUCCESS;
    }
    if (*root_count < discovered.size()) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    std::copy(discovered.begin(), discovered.end(), roots);
    *root_count = discovered.size();
    return CUDA_SUCCESS;
  }

  CUresult __stdcall fake_graph_node_get_dependencies(
    CUgraphNode node,
    CUgraphNode *dependencies,
    std::size_t *dependency_count
  ) {
    if (!node || !dependency_count) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    if (!dependencies) {
      *dependency_count = node->dependencies.size();
      return CUDA_SUCCESS;
    }
    if (*dependency_count < node->dependencies.size()) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    std::copy(node->dependencies.begin(), node->dependencies.end(), dependencies);
    *dependency_count = node->dependencies.size();
    return CUDA_SUCCESS;
  }

  CUresult __stdcall fake_graph_node_get_dependents(
    CUgraphNode node,
    CUgraphNode *dependents,
    std::size_t *dependent_count
  ) {
    if (!node || !dependent_count) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    if (!dependents) {
      *dependent_count = node->dependents.size();
      return CUDA_SUCCESS;
    }
    if (*dependent_count < node->dependents.size()) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    std::copy(node->dependents.begin(), node->dependents.end(), dependents);
    *dependent_count = node->dependents.size();
    return CUDA_SUCCESS;
  }

  CUresult __stdcall fake_graph_memcpy_node_get_params(
    CUgraphNode node,
    CUDA_MEMCPY3D *params
  ) {
    if (!node || !params || node->type != CU_GRAPH_NODE_TYPE_MEMCPY) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    *params = node->memcpy;
    return CUDA_SUCCESS;
  }

  CUresult __stdcall fake_child_graph_get_graph(CUgraphNode node, CUgraph *child) {
    if (!node || !child || node->type != CU_GRAPH_NODE_TYPE_GRAPH || !node->child) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    *child = node->child;
    return CUDA_SUCCESS;
  }

  CUresult __stdcall fake_ctx_get_current(CUcontext *context) {
    if (!context) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    *context = nullptr;
    return CUDA_SUCCESS;
  }

  CUresult __stdcall fake_ctx_set_current(CUcontext) {
    return CUDA_SUCCESS;
  }

  CUresult __stdcall fake_module_load_data(
    CUmodule *, const void *, unsigned int, int *, void **
  ) {
    return CUDA_ERROR_INVALID_HANDLE;
  }

  CUresult __stdcall fake_module_get_function(CUfunction *, CUmodule, const char *) {
    return CUDA_ERROR_INVALID_HANDLE;
  }

  CUresult __stdcall fake_module_unload(CUmodule) {
    return CUDA_ERROR_INVALID_HANDLE;
  }

  CUresult __stdcall fake_graph_create(CUgraph *, unsigned int) {
    return CUDA_ERROR_INVALID_HANDLE;
  }

  CUresult __stdcall fake_graph_destroy(CUgraph) {
    return CUDA_ERROR_INVALID_HANDLE;
  }

  CUresult __stdcall fake_graph_exec_destroy(CUgraphExec) {
    return CUDA_ERROR_INVALID_HANDLE;
  }

  CUresult __stdcall fake_graph_instantiate(
    CUgraphExec *, CUgraph, unsigned long long
  ) {
    return CUDA_ERROR_INVALID_HANDLE;
  }

  CUresult __stdcall fake_graph_launch(CUgraphExec, CUstream) {
    return CUDA_ERROR_INVALID_HANDLE;
  }

  CUresult __stdcall fake_conditional_handle_create(
    CUgraphConditionalHandle *, CUgraph, CUcontext, unsigned int, unsigned int
  ) {
    return CUDA_ERROR_INVALID_HANDLE;
  }

  CUresult __stdcall fake_graph_add_kernel_node(
    CUgraphNode *, CUgraph, const CUgraphNode *, std::size_t,
    const CUDA_KERNEL_NODE_PARAMS *
  ) {
    return CUDA_ERROR_INVALID_HANDLE;
  }

  CUresult __stdcall fake_graph_add_node(
    CUgraphNode *, CUgraph, const CUgraphNode *, std::size_t, CUgraphNodeParams *
  ) {
    return CUDA_ERROR_INVALID_HANDLE;
  }

  CUresult __stdcall fake_graph_add_child_node(
    CUgraphNode *, CUgraph, const CUgraphNode *, std::size_t, CUgraph
  ) {
    return CUDA_ERROR_INVALID_HANDLE;
  }

  cuda_driver_api fake_audit_api() {
    cuda_driver_api cuda;
    cuda.cuGraphGetNodes = fake_graph_get_nodes;
    cuda.cuGraphNodeGetType = fake_graph_node_get_type;
    cuda.cuGraphChildGraphNodeGetGraph = fake_child_graph_get_graph;
    return cuda;
  }

  cuda_driver_api fake_scalar_prefix_api() {
    cuda_driver_api cuda = fake_audit_api();
    cuda.cuCtxGetCurrent = fake_ctx_get_current;
    cuda.cuCtxSetCurrent = fake_ctx_set_current;
    cuda.cuModuleLoadDataEx = fake_module_load_data;
    cuda.cuModuleGetFunction = fake_module_get_function;
    cuda.cuModuleUnload = fake_module_unload;
    cuda.cuGraphCreate = fake_graph_create;
    cuda.cuGraphDestroy = fake_graph_destroy;
    cuda.cuGraphExecDestroy = fake_graph_exec_destroy;
    cuda.cuGraphInstantiateWithFlags = fake_graph_instantiate;
    cuda.cuGraphLaunch = fake_graph_launch;
    cuda.cuGraphConditionalHandleCreate = fake_conditional_handle_create;
    cuda.cuGraphAddKernelNode = fake_graph_add_kernel_node;
    cuda.cuGraphAddNode = fake_graph_add_node;
    cuda.cuGraphAddChildGraphNode = fake_graph_add_child_node;
    cuda.cuGraphGetRootNodes = fake_graph_get_root_nodes;
    cuda.cuGraphMemcpyNodeGetParams = fake_graph_memcpy_node_get_params;
    cuda.cuGraphNodeGetDependencies = fake_graph_node_get_dependencies;
    cuda.cuGraphNodeGetDependentNodes = fake_graph_node_get_dependents;
    return cuda;
  }

  CUDA_MEMCPY3D fake_scalar_copy(
    const CUdeviceptr source,
    const CUdeviceptr destination
  ) {
    CUDA_MEMCPY3D copy {};
    copy.srcMemoryType = CU_MEMORYTYPE_UNIFIED;
    copy.srcDevice = source;
    copy.dstMemoryType = CU_MEMORYTYPE_UNIFIED;
    copy.dstDevice = destination;
    copy.WidthInBytes = sizeof(std::uint64_t);
    copy.Height = 1u;
    copy.Depth = 1u;
    return copy;
  }

  struct transactional_graph_fake_t {
    CUgraph_st wrapper;
    std::array<CUgraph_st, 2u> bodies;
    CUgraph_st embedded_infer;
    std::array<CUgraph, 2u> body_handles {};
    std::array<CUgraphNode_st, 8u> wrapper_nodes;
    std::array<CUgraphNode_st, 8u> embedded_nodes;
    std::size_t wrapper_node_count = 0u;
    std::array<CUdeviceptr, 2u> allocated_mirrors {};
    std::vector<CUdeviceptr> freed_mirrors;
    std::size_t set_params_calls = 0u;
    std::size_t fail_set_params_call = 0u;
    bool fail_graph_destroy = false;
    bool graph_destroyed = false;
    bool executable_destroyed = false;
    bool module_unloaded = false;
  };

  transactional_graph_fake_t *transactional_graph_fake = nullptr;

  CUresult __stdcall transaction_module_load_data(
    CUmodule *module, const void *, unsigned int, int *, void **
  ) {
    if (!module || !transactional_graph_fake) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    *module = reinterpret_cast<CUmodule>(0x1010u);
    return CUDA_SUCCESS;
  }

  CUresult __stdcall transaction_module_get_function(
    CUfunction *function, CUmodule, const char *
  ) {
    if (!function || !transactional_graph_fake) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    *function = reinterpret_cast<CUfunction>(0x2020u);
    return CUDA_SUCCESS;
  }

  CUresult __stdcall transaction_module_unload(CUmodule) {
    if (!transactional_graph_fake) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    transactional_graph_fake->module_unloaded = true;
    return CUDA_SUCCESS;
  }

  CUresult __stdcall transaction_graph_create(CUgraph *graph, unsigned int) {
    if (!graph || !transactional_graph_fake) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    transactional_graph_fake->wrapper.nodes.clear();
    for (CUgraph_st &body : transactional_graph_fake->bodies) {
      body.nodes.clear();
    }
    transactional_graph_fake->body_handles = {
      &transactional_graph_fake->bodies[0],
      &transactional_graph_fake->bodies[1],
    };
    *graph = &transactional_graph_fake->wrapper;
    return CUDA_SUCCESS;
  }

  CUresult __stdcall transaction_graph_destroy(CUgraph graph) {
    if (!transactional_graph_fake || graph != &transactional_graph_fake->wrapper) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    if (transactional_graph_fake->fail_graph_destroy) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    transactional_graph_fake->graph_destroyed = true;
    return CUDA_SUCCESS;
  }

  CUresult __stdcall transaction_graph_exec_destroy(CUgraphExec executable) {
    if (!transactional_graph_fake ||
        executable != reinterpret_cast<CUgraphExec>(0x3030u)) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    transactional_graph_fake->executable_destroyed = true;
    return CUDA_SUCCESS;
  }

  CUresult __stdcall transaction_graph_instantiate(
    CUgraphExec *executable, CUgraph graph, unsigned long long
  ) {
    if (!transactional_graph_fake || !executable ||
        graph != &transactional_graph_fake->wrapper) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    *executable = reinterpret_cast<CUgraphExec>(0x3030u);
    return CUDA_SUCCESS;
  }

  CUresult __stdcall transaction_graph_launch(CUgraphExec, CUstream) {
    return CUDA_SUCCESS;
  }

  CUresult __stdcall transaction_mem_alloc(CUdeviceptr *pointer, std::size_t bytes) {
    if (!transactional_graph_fake || !pointer || bytes != sizeof(std::uint64_t)) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    for (std::size_t i = 0u; i < transactional_graph_fake->allocated_mirrors.size(); ++i) {
      if (transactional_graph_fake->allocated_mirrors[i] == 0u) {
        const CUdeviceptr allocated = 0x9000u + i * 0x100u;
        transactional_graph_fake->allocated_mirrors[i] = allocated;
        *pointer = allocated;
        return CUDA_SUCCESS;
      }
    }
    return CUDA_ERROR_INVALID_HANDLE;
  }

  CUresult __stdcall transaction_mem_free(const CUdeviceptr pointer) {
    if (!transactional_graph_fake || pointer == 0u) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    transactional_graph_fake->freed_mirrors.push_back(pointer);
    return CUDA_SUCCESS;
  }

  CUresult __stdcall transaction_memcpy_node_set_params(
    CUgraphNode node, const CUDA_MEMCPY3D *params
  ) {
    if (!transactional_graph_fake || !node || !params) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    ++transactional_graph_fake->set_params_calls;
    if (transactional_graph_fake->fail_set_params_call ==
        transactional_graph_fake->set_params_calls) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    node->memcpy = *params;
    return CUDA_SUCCESS;
  }

  CUgraphNode next_transaction_node(const CUgraphNodeType type) {
    if (!transactional_graph_fake ||
        transactional_graph_fake->wrapper_node_count >=
          transactional_graph_fake->wrapper_nodes.size()) {
      return nullptr;
    }
    CUgraphNode node = &transactional_graph_fake->wrapper_nodes[
      transactional_graph_fake->wrapper_node_count++
    ];
    node->type = type;
    node->child = nullptr;
    node->dependencies.clear();
    node->dependents.clear();
    node->memcpy = {};
    return node;
  }

  CUresult __stdcall transaction_graph_add_memcpy_node(
    CUgraphNode *node,
    CUgraph graph,
    const CUgraphNode *dependencies,
    const std::size_t dependency_count,
    const CUDA_MEMCPY3D *params,
    CUcontext
  ) {
    if (!transactional_graph_fake || !node || !params ||
        graph != &transactional_graph_fake->wrapper) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    CUgraphNode created = next_transaction_node(CU_GRAPH_NODE_TYPE_MEMCPY);
    if (!created) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    created->memcpy = *params;
    if (dependencies && dependency_count != 0u) {
      created->dependencies.assign(dependencies, dependencies + dependency_count);
    }
    graph->nodes.push_back(created);
    *node = created;
    return CUDA_SUCCESS;
  }

  CUresult __stdcall transaction_conditional_handle_create(
    CUgraphConditionalHandle *handle,
    CUgraph graph,
    CUcontext,
    unsigned int,
    unsigned int
  ) {
    if (!transactional_graph_fake || !handle ||
        graph != &transactional_graph_fake->wrapper) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    *handle = 0x4040u;
    return CUDA_SUCCESS;
  }

  CUresult __stdcall transaction_graph_add_kernel_node(
    CUgraphNode *node,
    CUgraph graph,
    const CUgraphNode *dependencies,
    const std::size_t dependency_count,
    const CUDA_KERNEL_NODE_PARAMS *
  ) {
    if (!transactional_graph_fake || !node ||
        graph != &transactional_graph_fake->wrapper) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    CUgraphNode created = next_transaction_node(CU_GRAPH_NODE_TYPE_KERNEL);
    if (!created) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    if (dependencies && dependency_count != 0u) {
      created->dependencies.assign(dependencies, dependencies + dependency_count);
    }
    graph->nodes.push_back(created);
    *node = created;
    return CUDA_SUCCESS;
  }

  CUresult __stdcall transaction_graph_add_node(
    CUgraphNode *node,
    CUgraph graph,
    const CUgraphNode *dependencies,
    const std::size_t dependency_count,
    CUgraphNodeParams *params
  ) {
    if (!transactional_graph_fake || !node || !params ||
        graph != &transactional_graph_fake->wrapper ||
        params->type != CU_GRAPH_NODE_TYPE_CONDITIONAL) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    CUgraphNode created = next_transaction_node(CU_GRAPH_NODE_TYPE_CONDITIONAL);
    if (!created) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    if (dependencies && dependency_count != 0u) {
      created->dependencies.assign(dependencies, dependencies + dependency_count);
    }
    params->params.conditional.phGraph_out =
      transactional_graph_fake->body_handles.data();
    graph->nodes.push_back(created);
    *node = created;
    return CUDA_SUCCESS;
  }

  CUresult __stdcall transaction_graph_add_child_node(
    CUgraphNode *node,
    CUgraph graph,
    const CUgraphNode *,
    std::size_t,
    CUgraph child
  ) {
    if (!transactional_graph_fake || !node || !graph || !child) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    CUgraphNode created = next_transaction_node(CU_GRAPH_NODE_TYPE_GRAPH);
    if (!created) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    if (graph == &transactional_graph_fake->bodies[0]) {
      if (child->nodes.size() > transactional_graph_fake->embedded_nodes.size()) {
        return CUDA_ERROR_INVALID_HANDLE;
      }
      transactional_graph_fake->embedded_infer.nodes.clear();
      for (std::size_t i = 0u; i < child->nodes.size(); ++i) {
        const CUgraphNode source = child->nodes[i];
        CUgraphNode embedded = &transactional_graph_fake->embedded_nodes[i];
        embedded->type = source->type;
        embedded->child = source->child;
        embedded->dependencies.clear();
        embedded->dependents.clear();
        embedded->memcpy = source->memcpy;
        transactional_graph_fake->embedded_infer.nodes.push_back(embedded);
      }
      const auto embedded_for = [&](const CUgraphNode source) -> CUgraphNode {
        const auto found = std::find(child->nodes.begin(), child->nodes.end(), source);
        return found == child->nodes.end() ? nullptr :
          transactional_graph_fake->embedded_infer.nodes[
            static_cast<std::size_t>(found - child->nodes.begin())
          ];
      };
      for (std::size_t i = 0u; i < child->nodes.size(); ++i) {
        const CUgraphNode source = child->nodes[i];
        CUgraphNode embedded = transactional_graph_fake->embedded_infer.nodes[i];
        for (const CUgraphNode dependency : source->dependencies) {
          embedded->dependencies.push_back(embedded_for(dependency));
        }
        for (const CUgraphNode dependent : source->dependents) {
          embedded->dependents.push_back(embedded_for(dependent));
        }
      }
      created->child = &transactional_graph_fake->embedded_infer;
    } else {
      created->child = child;
    }
    graph->nodes.push_back(created);
    *node = created;
    return CUDA_SUCCESS;
  }

  cuda_driver_api transactional_graph_api(transactional_graph_fake_t &state) {
    transactional_graph_fake = &state;
    cuda_driver_api cuda = fake_scalar_prefix_api();
    cuda.cuMemAlloc = transaction_mem_alloc;
    cuda.cuMemFree = transaction_mem_free;
    cuda.cuModuleLoadDataEx = transaction_module_load_data;
    cuda.cuModuleGetFunction = transaction_module_get_function;
    cuda.cuModuleUnload = transaction_module_unload;
    cuda.cuGraphCreate = transaction_graph_create;
    cuda.cuGraphDestroy = transaction_graph_destroy;
    cuda.cuGraphExecDestroy = transaction_graph_exec_destroy;
    cuda.cuGraphInstantiateWithFlags = transaction_graph_instantiate;
    cuda.cuGraphLaunch = transaction_graph_launch;
    cuda.cuGraphConditionalHandleCreate = transaction_conditional_handle_create;
    cuda.cuGraphAddKernelNode = transaction_graph_add_kernel_node;
    cuda.cuGraphAddMemcpyNode = transaction_graph_add_memcpy_node;
    cuda.cuGraphAddNode = transaction_graph_add_node;
    cuda.cuGraphAddChildGraphNode = transaction_graph_add_child_node;
    cuda.cuGraphMemcpyNodeSetParams = transaction_memcpy_node_set_params;
    return cuda;
  }

  struct scalar_prefix_graph_t {
    CUgraphNode_st first_copy {.type = CU_GRAPH_NODE_TYPE_MEMCPY};
    CUgraphNode_st second_copy {.type = CU_GRAPH_NODE_TYPE_MEMCPY};
    CUgraphNode_st first_kernel {.type = CU_GRAPH_NODE_TYPE_KERNEL};
    CUgraph_st graph;
    std::array<CUDA_MEMCPY3D, 2u> original_params;

    scalar_prefix_graph_t() {
      first_copy.memcpy = fake_scalar_copy(0x1000u, 0x2000u);
      second_copy.memcpy = fake_scalar_copy(0x1100u, 0x2100u);
      original_params = {first_copy.memcpy, second_copy.memcpy};
      first_copy.dependents = {&second_copy};
      second_copy.dependencies = {&first_copy};
      second_copy.dependents = {&first_kernel};
      first_kernel.dependencies = {&second_copy};
      graph.nodes = {&first_copy, &second_copy, &first_kernel};
    }
  };

  class conditional_hardware_fixture_t {
  public:
    enum class initialize_result_e {
      ready,
      unavailable,
      failed,
    };

    ~conditional_hardware_fixture_t() {
      cleanup();
    }

    initialize_result_e initialize(
      const bool with_reuse_child = false,
      const bool with_scalar_prefix = false,
      const bool with_optional_child = false
    ) {
      cuda_ = &cuda_driver_api::get();
      if (!cuda_->is_valid() || !cuda_->has_conditional_graph_support() ||
          !cuda_->cuDevicePrimaryCtxRetain || !cuda_->cuDevicePrimaryCtxRelease ||
          !cuda_->cuGraphAddKernelNode || !cuda_->cuGraphAddMemsetNode ||
          !cuda_->cuModuleLoadDataEx || !cuda_->cuModuleGetFunction ||
          !cuda_->cuModuleUnload || !cuda_->cuMemcpyHtoD ||
          !cuda_->cuMemcpyDtoH || !cuda_->cuGraphLaunch) {
        return initialize_result_e::unavailable;
      }
      if (cuda_->cuInit(0u) != CUDA_SUCCESS ||
          cuda_->cuCtxGetCurrent(&previous_context_) != CUDA_SUCCESS ||
          cuda_->cuDeviceGet(&device_, 0) != CUDA_SUCCESS ||
          cuda_->cuDevicePrimaryCtxRetain(&context_, device_) != CUDA_SUCCESS) {
        return initialize_result_e::unavailable;
      }
      retained_ = true;
      if (cuda_->cuCtxSetCurrent(context_) != CUDA_SUCCESS ||
          cuda_->cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS ||
          cuda_->cuMemAlloc(&records_, sizeof(decision_record_t) +
                                        sizeof(request_record_t)) != CUDA_SUCCESS ||
          cuda_->cuMemAlloc(&output_, sizeof(std::uint32_t)) != CUDA_SUCCESS ||
          (with_optional_child &&
           cuda_->cuMemAlloc(&optional_output_, sizeof(std::uint32_t)) != CUDA_SUCCESS) ||
          cuda_->cuGraphCreate(&child_, 0u) != CUDA_SUCCESS ||
          cuda_->cuModuleLoadDataEx(
            &scalar_module_, scalar_prefix_consumer_ptx, 0u, nullptr, nullptr
          ) != CUDA_SUCCESS ||
          cuda_->cuModuleGetFunction(
            &marker_writer_, scalar_module_, "write_inference_marker"
          ) != CUDA_SUCCESS || !marker_writer_) {
        return initialize_result_e::failed;
      }

      scalar_prefix_ = with_scalar_prefix;
      if (scalar_prefix_) {
        if (!cuda_->cuGraphAddMemcpyNode ||
            cuda_->cuMemAlloc(&scalar_targets_[0], sizeof(std::uint64_t)) != CUDA_SUCCESS ||
            cuda_->cuMemAlloc(&scalar_targets_[1], sizeof(std::uint64_t)) != CUDA_SUCCESS ||
            cuda_->cuModuleGetFunction(
              &scalar_consumer_, scalar_module_, "consume_grid_scalars"
            ) != CUDA_SUCCESS || !scalar_consumer_) {
          return initialize_result_e::failed;
        }
      }

      if (scalar_prefix_) {
        CUgraphNode copy_tail = nullptr;
        for (std::size_t i = 0u; i < scalar_seeds_.size(); ++i) {
          CUDA_MEMCPY3D copy {};
          copy.srcMemoryType = CU_MEMORYTYPE_UNIFIED;
          copy.srcDevice = reinterpret_cast<CUdeviceptr>(&scalar_seeds_[i]);
          copy.dstMemoryType = CU_MEMORYTYPE_UNIFIED;
          copy.dstDevice = scalar_targets_[i];
          copy.WidthInBytes = sizeof(std::uint64_t);
          copy.Height = 1u;
          copy.Depth = 1u;
          CUgraphNode copy_node = nullptr;
          const CUgraphNode *dependencies = copy_tail ? &copy_tail : nullptr;
          if (cuda_->cuGraphAddMemcpyNode(
                &copy_node,
                child_,
                dependencies,
                copy_tail ? 1u : 0u,
                &copy,
                context_
              ) != CUDA_SUCCESS || !copy_node) {
            return initialize_result_e::failed;
          }
          scalar_copy_nodes_[i] = copy_node;
          copy_tail = copy_node;
        }
        CUdeviceptr height = scalar_targets_[0];
        CUdeviceptr width = scalar_targets_[1];
        CUdeviceptr output = output_;
        void *kernel_args[] = {&height, &width, &output};
        CUDA_KERNEL_NODE_PARAMS kernel {};
        kernel.func = scalar_consumer_;
        kernel.gridDimX = 1u;
        kernel.gridDimY = 1u;
        kernel.gridDimZ = 1u;
        kernel.blockDimX = 1u;
        kernel.blockDimY = 1u;
        kernel.blockDimZ = 1u;
        kernel.kernelParams = kernel_args;
        CUgraphNode kernel_node = nullptr;
        if (cuda_->cuGraphAddKernelNode(
              &kernel_node, child_, &copy_tail, 1u, &kernel
            ) != CUDA_SUCCESS || !kernel_node) {
          return initialize_result_e::failed;
        }
      } else {
        CUdeviceptr output = output_;
        std::uint32_t value = marker;
        void *kernel_args[] = {&output, &value};
        CUDA_KERNEL_NODE_PARAMS kernel {};
        kernel.func = marker_writer_;
        kernel.gridDimX = 1u;
        kernel.gridDimY = 1u;
        kernel.gridDimZ = 1u;
        kernel.blockDimX = 1u;
        kernel.blockDimY = 1u;
        kernel.blockDimZ = 1u;
        kernel.kernelParams = kernel_args;
        CUgraphNode kernel_node = nullptr;
        if (cuda_->cuGraphAddKernelNode(
              &kernel_node, child_, nullptr, 0u, &kernel
            ) != CUDA_SUCCESS || !kernel_node) {
          return initialize_result_e::failed;
        }
      }
      if (with_reuse_child) {
        CUDA_MEMSET_NODE_PARAMS memset_params {};
        memset_params.dst = output_;
        memset_params.pitch = sizeof(std::uint32_t);
        memset_params.elementSize = sizeof(std::uint32_t);
        memset_params.width = 1u;
        memset_params.height = 1u;
        memset_params.value = reuse_marker;
        CUgraphNode reuse_memset_node = nullptr;
        if (cuda_->cuGraphCreate(&reuse_child_, 0u) != CUDA_SUCCESS ||
            cuda_->cuGraphAddMemsetNode(
              &reuse_memset_node, reuse_child_, nullptr, 0u, &memset_params, context_
            ) != CUDA_SUCCESS) {
          return initialize_result_e::failed;
        }
      }
      if (with_optional_child) {
        CUdeviceptr optional_output = optional_output_;
        std::uint32_t value = optional_marker;
        void *kernel_args[] = {&optional_output, &value};
        CUDA_KERNEL_NODE_PARAMS kernel {};
        kernel.func = marker_writer_;
        kernel.gridDimX = 1u;
        kernel.gridDimY = 1u;
        kernel.gridDimZ = 1u;
        kernel.blockDimX = 1u;
        kernel.blockDimY = 1u;
        kernel.blockDimZ = 1u;
        kernel.kernelParams = kernel_args;
        CUgraphNode optional_kernel_node = nullptr;
        if (cuda_->cuGraphCreate(&optional_child_, 0u) != CUDA_SUCCESS ||
            cuda_->cuGraphAddKernelNode(
              &optional_kernel_node, optional_child_, nullptr, 0u, &kernel
            ) != CUDA_SUCCESS || !optional_kernel_node) {
          return initialize_result_e::failed;
        }
      }

      if (cuda_->cuGraphInstantiateWithFlags(&raw_executable_, child_, 0u) != CUDA_SUCCESS ||
          !raw_executable_) {
        return initialize_result_e::failed;
      }

      // Exercise build's context preservation rather than building only in an already-current
      // context. A live application may construct this bridge from a different host thread.
      if (cuda_->cuCtxSetCurrent(nullptr) != CUDA_SUCCESS) {
        return initialize_result_e::failed;
      }
      bridge_ = executable_t::build(
        *cuda_,
        {
          .context = context_,
          .infer_child = child_,
          .optional_infer_child = optional_child_,
          .reuse_child = reuse_child_,
          .decision_record = records_,
          .request_record = records_ + sizeof(decision_record_t),
        }
      );
      if (!bridge_.ready() && bridge_.cuda_result() == CUDA_ERROR_NOT_SUPPORTED) {
        return initialize_result_e::unavailable;
      }
      CUcontext after_build = context_;
      if (!bridge_.ready() ||
          cuda_->cuCtxGetCurrent(&after_build) != CUDA_SUCCESS || after_build != nullptr ||
          cuda_->cuCtxSetCurrent(context_) != CUDA_SUCCESS) {
        return initialize_result_e::failed;
      }
      return initialize_result_e::ready;
    }

    bool run(
      const decision_record_t &proposal,
      const request_record_t &request,
      const std::uint32_t expected_output,
      const branch_e expected_branch,
      const bool expected_authenticated_receipt
    ) {
      const std::uint32_t zero = 0u;
      decision_record_t receipt {};
      if (cuda_->cuMemcpyHtoD(output_, &zero, sizeof(zero)) != CUDA_SUCCESS ||
          cuda_->cuMemcpyHtoD(records_, &proposal, sizeof(proposal)) != CUDA_SUCCESS ||
          cuda_->cuMemcpyHtoD(
            records_ + sizeof(proposal), &request, sizeof(request)
          ) != CUDA_SUCCESS ||
          cuda_->cuGraphLaunch(bridge_.get(), stream_) != CUDA_SUCCESS ||
          cuda_->cuStreamSynchronize(stream_) != CUDA_SUCCESS) {
        return false;
      }

      std::uint32_t observed_output = 0u;
      if (cuda_->cuMemcpyDtoH(
            &observed_output, output_, sizeof(observed_output)
          ) != CUDA_SUCCESS ||
          cuda_->cuMemcpyDtoH(&receipt, records_, sizeof(receipt)) != CUDA_SUCCESS) {
        return false;
      }
      return observed_output == expected_output &&
             authenticated_receipt(receipt, request) == expected_authenticated_receipt &&
             (!expected_authenticated_receipt ||
              receipt.decision == static_cast<std::uint32_t>(expected_branch));
    }

    bool run_optional(
      const decision_record_t &proposal,
      const request_record_t &request,
      const std::uint32_t expected_depth_output,
      const std::uint32_t expected_optional_output,
      const branch_e expected_branch,
      const bool expected_authenticated_receipt,
      const bool expected_optional_receipt
    ) {
      if (!optional_child_ || !optional_output_) {
        return false;
      }
      const std::uint32_t zero = 0u;
      decision_record_t receipt {};
      if (cuda_->cuMemcpyHtoD(output_, &zero, sizeof(zero)) != CUDA_SUCCESS ||
          cuda_->cuMemcpyHtoD(optional_output_, &zero, sizeof(zero)) != CUDA_SUCCESS ||
          cuda_->cuMemcpyHtoD(records_, &proposal, sizeof(proposal)) != CUDA_SUCCESS ||
          cuda_->cuMemcpyHtoD(
            records_ + sizeof(proposal), &request, sizeof(request)
          ) != CUDA_SUCCESS ||
          cuda_->cuGraphLaunch(bridge_.get(), stream_) != CUDA_SUCCESS ||
          cuda_->cuStreamSynchronize(stream_) != CUDA_SUCCESS) {
        return false;
      }
      std::uint32_t observed_depth = 0u;
      std::uint32_t observed_optional = 0u;
      if (cuda_->cuMemcpyDtoH(
            &observed_depth, output_, sizeof(observed_depth)
          ) != CUDA_SUCCESS ||
          cuda_->cuMemcpyDtoH(
            &observed_optional, optional_output_, sizeof(observed_optional)
          ) != CUDA_SUCCESS ||
          cuda_->cuMemcpyDtoH(&receipt, records_, sizeof(receipt)) != CUDA_SUCCESS) {
        return false;
      }
      return observed_depth == expected_depth_output &&
             observed_optional == expected_optional_output &&
             authenticated_receipt(receipt, request) == expected_authenticated_receipt &&
             authenticated_optional_infer_receipt(receipt, request) ==
               expected_optional_receipt &&
             (!expected_authenticated_receipt ||
              receipt.decision == static_cast<std::uint32_t>(expected_branch));
    }

    bool run_raw(const std::uint32_t expected_output) {
      const std::uint32_t zero = 0u;
      std::uint32_t observed = 0u;
      return raw_executable_ &&
             cuda_->cuMemcpyHtoD(output_, &zero, sizeof(zero)) == CUDA_SUCCESS &&
             cuda_->cuGraphLaunch(raw_executable_, stream_) == CUDA_SUCCESS &&
             cuda_->cuStreamSynchronize(stream_) == CUDA_SUCCESS &&
             cuda_->cuMemcpyDtoH(&observed, output_, sizeof(observed)) == CUDA_SUCCESS &&
             observed == expected_output;
    }

    bool destroy_raw_executable() {
      if (!raw_executable_) {
        return true;
      }
      if (cuda_->cuStreamSynchronize(stream_) != CUDA_SUCCESS ||
          cuda_->cuGraphExecDestroy(raw_executable_) != CUDA_SUCCESS) {
        return false;
      }
      raw_executable_ = nullptr;
      return true;
    }

    bool scalar_source_unchanged() const {
      if (!scalar_prefix_) {
        return false;
      }
      for (std::size_t i = 0u; i < scalar_copy_nodes_.size(); ++i) {
        CUDA_MEMCPY3D params {};
        if (!scalar_copy_nodes_[i] ||
            cuda_->cuGraphMemcpyNodeGetParams(scalar_copy_nodes_[i], &params) != CUDA_SUCCESS ||
            params.srcMemoryType != CU_MEMORYTYPE_UNIFIED ||
            params.srcDevice != reinterpret_cast<CUdeviceptr>(&scalar_seeds_[i]) ||
            params.dstMemoryType != CU_MEMORYTYPE_UNIFIED ||
            params.dstDevice != scalar_targets_[i] ||
            params.WidthInBytes != sizeof(std::uint64_t) || params.Height != 1u ||
            params.Depth != 1u) {
          return false;
        }
      }
      return true;
    }

    bool reset_restores_null_context() {
      if (cuda_->cuCtxSetCurrent(nullptr) != CUDA_SUCCESS) {
        return false;
      }
      const bool reset = bridge_.reset();
      CUcontext current = context_;
      return reset && bridge_.empty() && !bridge_.ready() &&
             cuda_->cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == nullptr &&
             cuda_->cuCtxSetCurrent(context_) == CUDA_SUCCESS;
    }

    void set_scalar_seeds(const std::uint64_t height, const std::uint64_t width) noexcept {
      scalar_seeds_ = {height, width};
    }

    build_failure_e bridge_failure() const noexcept {
      return bridge_.failure();
    }

    CUresult bridge_cuda_result() const noexcept {
      return bridge_.cuda_result();
    }

  private:
    void cleanup() noexcept {
      if (!cuda_) {
        return;
      }
      if (context_) {
        cuda_->cuCtxSetCurrent(context_);
      }
      if (stream_) {
        cuda_->cuStreamSynchronize(stream_);
      }
      (void) bridge_.reset();
      if (raw_executable_ && cuda_->cuGraphExecDestroy) {
        cuda_->cuGraphExecDestroy(raw_executable_);
      }
      if (child_ && cuda_->cuGraphDestroy) {
        cuda_->cuGraphDestroy(child_);
      }
      if (reuse_child_ && cuda_->cuGraphDestroy) {
        cuda_->cuGraphDestroy(reuse_child_);
      }
      if (optional_child_ && cuda_->cuGraphDestroy) {
        cuda_->cuGraphDestroy(optional_child_);
      }
      if (scalar_module_ && cuda_->cuModuleUnload) {
        cuda_->cuModuleUnload(scalar_module_);
      }
      for (CUdeviceptr &target : scalar_targets_) {
        if (target && cuda_->cuMemFree) {
          cuda_->cuMemFree(target);
        }
        target = 0u;
      }
      if (output_ && cuda_->cuMemFree) {
        cuda_->cuMemFree(output_);
      }
      if (optional_output_ && cuda_->cuMemFree) {
        cuda_->cuMemFree(optional_output_);
      }
      if (records_ && cuda_->cuMemFree) {
        cuda_->cuMemFree(records_);
      }
      if (stream_ && cuda_->cuStreamDestroy) {
        cuda_->cuStreamDestroy(stream_);
      }
      child_ = nullptr;
      reuse_child_ = nullptr;
      optional_child_ = nullptr;
      raw_executable_ = nullptr;
      scalar_module_ = nullptr;
      scalar_consumer_ = nullptr;
      marker_writer_ = nullptr;
      output_ = 0u;
      optional_output_ = 0u;
      records_ = 0u;
      stream_ = nullptr;
      cuda_->cuCtxSetCurrent(previous_context_);
      if (retained_) {
        cuda_->cuDevicePrimaryCtxRelease(device_);
      }
      retained_ = false;
      context_ = nullptr;
    }

    static constexpr std::uint32_t marker = 0xa5a5a5a5u;
    static constexpr std::uint32_t reuse_marker = 0x5a5a5a5au;
    static constexpr std::uint32_t optional_marker = 0xc3c3c3c3u;
    cuda_driver_api *cuda_ = nullptr;
    CUdevice device_ = 0;
    CUcontext previous_context_ = nullptr;
    CUcontext context_ = nullptr;
    CUstream stream_ = nullptr;
    CUdeviceptr records_ = 0u;
    CUdeviceptr output_ = 0u;
    CUdeviceptr optional_output_ = 0u;
    CUgraph child_ = nullptr;
    CUgraph reuse_child_ = nullptr;
    CUgraph optional_child_ = nullptr;
    CUgraphExec raw_executable_ = nullptr;
    CUmodule scalar_module_ = nullptr;
    CUfunction scalar_consumer_ = nullptr;
    CUfunction marker_writer_ = nullptr;
    std::array<std::uint64_t, 2u> scalar_seeds_ {31u, 55u};
    std::array<CUdeviceptr, 2u> scalar_targets_ {};
    std::array<CUgraphNode, 2u> scalar_copy_nodes_ {};
    bool scalar_prefix_ = false;
    executable_t bridge_;
    bool retained_ = false;

  public:
    static constexpr std::uint32_t infer_marker = marker;
    static constexpr std::uint32_t reuse_branch_marker = reuse_marker;
    static constexpr std::uint32_t optional_infer_marker = optional_marker;
  };

}  // namespace

TEST(CudaConditionalGraphContract, ResolvesOnlyExactCurrentReuseProposal) {
  constexpr std::uint64_t token = 0x18cc257e17cfdd6cull;
  const auto request = make_request(token);
  const auto reuse = make_proposal(branch_e::reuse, token);
  const auto infer = make_proposal(branch_e::infer, token);

  const auto reuse_receipt = resolve_proposal(reuse, request);
  EXPECT_TRUE(authenticated_reuse_proposal(reuse, request));
  EXPECT_TRUE(authenticated_receipt(reuse_receipt, request));
  EXPECT_EQ(reuse_receipt.decision, static_cast<std::uint32_t>(branch_e::reuse));

  const auto infer_receipt = resolve_proposal(infer, request);
  EXPECT_FALSE(authenticated_reuse_proposal(infer, request));
  EXPECT_TRUE(authenticated_receipt(infer_receipt, request));
  EXPECT_EQ(infer_receipt.decision, static_cast<std::uint32_t>(branch_e::infer));
  EXPECT_FALSE(authenticated_receipt(reuse, request)) << "PROP is never a resolved receipt";
}

TEST(CudaConditionalGraphContract, AuthenticatesOptionalOcrOnlyOnInfer) {
  constexpr std::uint64_t token = 0x8877665544332211ull;
  const auto request = make_request(token, work_flag_e::optional_ocr);
  ASSERT_TRUE(authenticated_request(request));

  const auto infer_receipt = resolve_proposal(
    make_proposal(branch_e::infer, token), request
  );
  EXPECT_TRUE(authenticated_receipt(infer_receipt, request));
  EXPECT_TRUE(authenticated_optional_ocr_receipt(infer_receipt, request));
  EXPECT_EQ(infer_receipt.reserved, optional_ocr_receipt_magic);
  EXPECT_EQ(
    infer_receipt.decision_cookie,
    static_cast<std::uint32_t>(branch_e::infer) ^ decision_cookie ^
      optional_ocr_receipt_magic
  );

  const auto reuse_receipt = resolve_proposal(
    make_proposal(branch_e::reuse, token), request
  );
  EXPECT_TRUE(authenticated_receipt(reuse_receipt, request));
  EXPECT_FALSE(authenticated_optional_ocr_receipt(reuse_receipt, request));
  EXPECT_EQ(reuse_receipt.reserved, 0u);

  auto forged_reuse_ocr = reuse_receipt;
  forged_reuse_ocr.reserved = optional_ocr_receipt_magic;
  forged_reuse_ocr.decision_cookie ^= optional_ocr_receipt_magic;
  EXPECT_FALSE(authenticated_receipt(forged_reuse_ocr, request))
    << "Ordinary OCR may not authenticate on a reuse receipt";

  auto malformed = make_proposal(branch_e::infer, token);
  malformed.decision_cookie ^= 1u;
  const auto fail_open_receipt = resolve_proposal(malformed, request);
  EXPECT_TRUE(authenticated_receipt(fail_open_receipt, request));
  EXPECT_EQ(
    fail_open_receipt.decision,
    static_cast<std::uint32_t>(branch_e::infer)
  );
  EXPECT_EQ(fail_open_receipt.reserved, 0u);
  EXPECT_FALSE(authenticated_optional_ocr_receipt(fail_open_receipt, request));

  auto forged_optional = fail_open_receipt;
  forged_optional.reserved = optional_ocr_receipt_magic;
  EXPECT_FALSE(authenticated_receipt(forged_optional, request))
    << "The OCR-run disposition is bound into the receipt cookie";

  const auto child_absent_receipt = resolve_proposal(
    make_proposal(branch_e::infer, token), request, false
  );
  EXPECT_TRUE(authenticated_receipt(child_absent_receipt, request));
  EXPECT_EQ(child_absent_receipt.reserved, 0u);
  EXPECT_FALSE(authenticated_optional_ocr_receipt(child_absent_receipt, request))
    << "An authenticated request cannot authorize OCR when no optional child exists";

  EXPECT_TRUE(authenticated_request(make_request(
    token, work_flag_e::subtitle_observation
  )));
  const auto retired_flag4 = make_request(
    token, static_cast<work_flag_e>(4u)
  );
  EXPECT_FALSE(authenticated_request(retired_flag4));
  auto combined_work = make_request(token, work_flag_e::optional_ocr);
  combined_work.work_flags = 3u;
  combined_work.work_flags_cookie = 3u ^ work_flags_cookie;
  EXPECT_FALSE(authenticated_request(combined_work))
    << "Subtitle dispositions are mutually exclusive authenticated modes";
}

TEST(CudaConditionalGraphContract, AuthenticatesCadenceDueOcrOnInferAndReuse) {
  constexpr std::uint64_t token = 0x9a8b7c6d5e4f3021ull;
  const auto request = make_request(token, work_flag_e::optional_ocr_due);
  ASSERT_TRUE(authenticated_request(request));

  for (const auto branch : {branch_e::infer, branch_e::reuse}) {
    const auto receipt = resolve_proposal(make_proposal(branch, token), request);
    EXPECT_TRUE(authenticated_receipt(receipt, request));
    EXPECT_TRUE(authenticated_optional_ocr_receipt(receipt, request));
    EXPECT_EQ(receipt.decision, static_cast<std::uint32_t>(branch));
    EXPECT_EQ(receipt.reserved, optional_ocr_receipt_magic);
  }

  auto malformed = make_proposal(branch_e::reuse, token);
  malformed.magic = 0u;
  const auto fail_open_receipt = resolve_proposal(malformed, request);
  EXPECT_TRUE(authenticated_receipt(fail_open_receipt, request));
  EXPECT_EQ(
    fail_open_receipt.decision,
    static_cast<std::uint32_t>(branch_e::infer)
  );
  EXPECT_FALSE(authenticated_optional_ocr_receipt(fail_open_receipt, request));

  const auto child_absent_receipt = resolve_proposal(
    make_proposal(branch_e::reuse, token), request, false
  );
  EXPECT_TRUE(authenticated_receipt(child_absent_receipt, request));
  EXPECT_FALSE(authenticated_optional_ocr_receipt(child_absent_receipt, request));
}

TEST(CudaConditionalGraphContract, CadenceDueAbstentionNeverAuthenticatesOptionalOcr) {
  constexpr std::uint64_t token = 0x6b5a493827160f1eull;
  const auto request = make_request(token, work_flag_e::subtitle_observation_due);
  ASSERT_TRUE(authenticated_request(request));

  for (const auto branch : {branch_e::infer, branch_e::reuse}) {
    const auto receipt = resolve_proposal(make_proposal(branch, token), request);
    EXPECT_TRUE(authenticated_receipt(receipt, request));
    EXPECT_EQ(receipt.decision, static_cast<std::uint32_t>(branch));
    EXPECT_EQ(receipt.reserved, 0u);
    EXPECT_FALSE(authenticated_optional_ocr_receipt(receipt, request));

    auto forged_ocr = receipt;
    forged_ocr.reserved = optional_ocr_receipt_magic;
    forged_ocr.decision_cookie ^= optional_ocr_receipt_magic;
    EXPECT_FALSE(authenticated_receipt(forged_ocr, request))
      << "A branch-independent abstention may not claim optional OCR execution";
  }
}

TEST(CudaConditionalGraphContract, MalformedAndStaleInputsFailOpenToInfer) {
  constexpr std::uint64_t token = 0x1020304050607080ull;
  const auto request = make_request(token);
  const auto valid_reuse = make_proposal(branch_e::reuse, token);

  std::array<decision_record_t, 6> malformed_proposals {};
  malformed_proposals[0] = {};
  malformed_proposals[1] = make_proposal(branch_e::reuse, token - 1u);
  malformed_proposals[2] = valid_reuse;
  malformed_proposals[2].decision_cookie ^= 1u;
  malformed_proposals[3] = valid_reuse;
  malformed_proposals[3].magic ^= 1u;
  malformed_proposals[4] = valid_reuse;
  malformed_proposals[4].reserved = 1u;
  malformed_proposals[5] = valid_reuse;
  malformed_proposals[5].decision = 7u;
  malformed_proposals[5].decision_cookie = 7u ^ decision_cookie;

  for (const auto &proposal : malformed_proposals) {
    const auto receipt = resolve_proposal(proposal, request);
    EXPECT_TRUE(authenticated_receipt(receipt, request));
    EXPECT_EQ(receipt.decision, static_cast<std::uint32_t>(branch_e::infer));
  }

  auto bad_request = request;
  bad_request.token_low_cookie ^= 1u;
  const auto receipt = resolve_proposal(valid_reuse, bad_request);
  EXPECT_EQ(receipt.decision, static_cast<std::uint32_t>(branch_e::infer));
  EXPECT_FALSE(authenticated_receipt(receipt, bad_request));

  const auto zero_request = make_request(0u, work_flag_e::optional_infer);
  EXPECT_FALSE(authenticated_request(zero_request));
  EXPECT_FALSE(authenticated_optional_infer_receipt(
    resolve_proposal(make_proposal(branch_e::infer, 0u), zero_request),
    zero_request
  ));
}

TEST(CudaConditionalGraphContract, EmbeddedPtxPublishesReceiptBeforeSettingCondition) {
  const std::string_view ptx = bridge_ptx_source();
  EXPECT_NE(ptx.find(".extern .func cudaGraphSetConditional"), std::string_view::npos);
  EXPECT_NE(ptx.find("0x504f5250"), std::string_view::npos);  // PROP
  EXPECT_NE(ptx.find("0x54535152"), std::string_view::npos);  // RQST
  EXPECT_NE(ptx.find("0x47524243"), std::string_view::npos);  // CBRG
  EXPECT_NE(ptx.find("0x52434f4f"), std::string_view::npos);  // OOCR
  EXPECT_NE(ptx.find("setp.eq.u32 %p4, %r14, 0"), std::string_view::npos);
  EXPECT_NE(ptx.find("setp.eq.u32 %p24, %r14, 1"), std::string_view::npos);
  EXPECT_NE(ptx.find("setp.eq.u32 %p28, %r14, 2"), std::string_view::npos);
  EXPECT_NE(ptx.find("setp.eq.u32 %p29, %r14, 8"), std::string_view::npos);
  EXPECT_NE(ptx.find("setp.eq.u32 %p30, %r14, 16"), std::string_view::npos);
  EXPECT_NE(ptx.find("setp.eq.u32 %p26, %r14, 8"), std::string_view::npos);
  const auto receipt_publish = ptx.find("st.global.u32 [%rd3+24], %r24");
  const auto conditional_call = ptx.find("call.uni cudaGraphSetConditional");
  ASSERT_NE(receipt_publish, std::string_view::npos);
  ASSERT_NE(conditional_call, std::string_view::npos);
  EXPECT_LT(receipt_publish, conditional_call);
  const auto post_publish_barrier = ptx.find("membar.gl", receipt_publish);
  ASSERT_NE(post_publish_barrier, std::string_view::npos);
  EXPECT_LT(post_publish_barrier, conditional_call);
  EXPECT_EQ(ptx.find("st.global", receipt_publish + 1u), std::string_view::npos)
    << "The receipt tag must be the final global store";
  const auto first_conditional_call = ptx.find("call.uni cudaGraphSetConditional");
  ASSERT_NE(first_conditional_call, std::string_view::npos);
  EXPECT_NE(
    ptx.find("call.uni cudaGraphSetConditional", first_conditional_call + 1u),
    std::string_view::npos
  ) << "Depth and optional inference own sibling conditional handles";
}

TEST(CudaConditionalGraphContract, BuilderFailuresExposeNoExecutable) {
  cuda_driver_api unavailable;
  auto result = executable_t::build(unavailable, {});
  EXPECT_FALSE(result.ready());
  EXPECT_EQ(result.get(), nullptr);
  EXPECT_EQ(result.failure(), build_failure_e::invalid_descriptor);

  CUgraph_st child;
  result = executable_t::build(
    unavailable,
    {
      .context = reinterpret_cast<CUcontext>(1u),
      .infer_child = &child,
      .decision_record = 16u,
      .request_record = 32u,
    }
  );
  EXPECT_FALSE(result.ready());
  EXPECT_EQ(result.failure(), build_failure_e::invalid_descriptor)
    << "The two 32-byte authenticated records must not overlap";

  result = executable_t::build(
    unavailable,
    {
      .context = reinterpret_cast<CUcontext>(1u),
      .infer_child = &child,
      .decision_record = 16u,
      .request_record = 48u,
    }
  );
  EXPECT_FALSE(result.ready());
  EXPECT_EQ(result.get(), nullptr);
  EXPECT_EQ(result.failure(), build_failure_e::driver_api_unavailable);

  constexpr CUdeviceptr last_aligned_address =
    std::numeric_limits<CUdeviceptr>::max() - 15u;
  result = executable_t::build(
    unavailable,
    {
      .context = reinterpret_cast<CUcontext>(1u),
      .infer_child = &child,
      .decision_record = last_aligned_address,
      .request_record = 0x1000u,
    }
  );
  EXPECT_EQ(result.failure(), build_failure_e::invalid_descriptor)
    << "The complete 32-byte decision record must not wrap the device address space";

  result = executable_t::build(
    unavailable,
    {
      .context = reinterpret_cast<CUcontext>(1u),
      .infer_child = &child,
      .decision_record = 0x1000u,
      .request_record = last_aligned_address,
    }
  );
  EXPECT_EQ(result.failure(), build_failure_e::invalid_descriptor)
    << "The complete 32-byte request record must not wrap the device address space";

  result = executable_t::build(
    unavailable,
    {
      .context = reinterpret_cast<CUcontext>(1u),
      .infer_child = &child,
      .decision_record = std::numeric_limits<CUdeviceptr>::max() - 31u,
      .request_record = 0x1000u,
    }
  );
  EXPECT_EQ(result.failure(), build_failure_e::driver_api_unavailable)
    << "A 32-byte record whose final byte is exactly the address maximum does not wrap";
}

TEST(CudaConditionalGraphContract, CapabilityIncludesScalarPrefixInspectionApis) {
  cuda_driver_api cuda = fake_scalar_prefix_api();
  ASSERT_TRUE(cuda.has_conditional_graph_support());

  cuda.cuGraphGetRootNodes = nullptr;
  EXPECT_FALSE(cuda.has_conditional_graph_support());
  cuda.cuGraphGetRootNodes = fake_graph_get_root_nodes;

  cuda.cuGraphMemcpyNodeGetParams = nullptr;
  EXPECT_FALSE(cuda.has_conditional_graph_support());
  cuda.cuGraphMemcpyNodeGetParams = fake_graph_memcpy_node_get_params;

  cuda.cuGraphNodeGetDependencies = nullptr;
  cuda.cuGraphNodeGetDependencies_v2 = nullptr;
  EXPECT_FALSE(cuda.has_conditional_graph_support());
  cuda.cuGraphNodeGetDependencies = fake_graph_node_get_dependencies;

  cuda.cuGraphNodeGetDependentNodes = nullptr;
  cuda.cuGraphNodeGetDependentNodes_v2 = nullptr;
  EXPECT_FALSE(cuda.has_conditional_graph_support());
}

TEST(CudaConditionalGraphContract, RejectsScalarSourceAliasingAnotherDestination) {
  CUgraphNode_st first_copy {.type = CU_GRAPH_NODE_TYPE_MEMCPY};
  CUgraphNode_st second_copy {.type = CU_GRAPH_NODE_TYPE_MEMCPY};
  CUgraphNode_st first_kernel {.type = CU_GRAPH_NODE_TYPE_KERNEL};
  first_copy.memcpy = fake_scalar_copy(0x1000u, 0x2000u);
  second_copy.memcpy = fake_scalar_copy(0x2000u, 0x3000u);
  first_copy.dependents = {&second_copy};
  second_copy.dependencies = {&first_copy};
  second_copy.dependents = {&first_kernel};
  first_kernel.dependencies = {&second_copy};
  CUgraph_st graph {{&first_copy, &second_copy, &first_kernel}};

  cuda_driver_api cuda = fake_scalar_prefix_api();
  auto result = executable_t::build(
    cuda,
    {
      .context = reinterpret_cast<CUcontext>(1u),
      .infer_child = &graph,
      .decision_record = 0x4000u,
      .request_record = 0x4040u,
    }
  );

  EXPECT_FALSE(result.ready());
  EXPECT_TRUE(result.empty());
  EXPECT_EQ(result.failure(), build_failure_e::infer_child_scalar_prefix_rejected);
  EXPECT_EQ(result.cuda_result(), CUDA_SUCCESS);
}

TEST(CudaConditionalGraphContract, RewritesOnlyNodeOwnedEmbeddedScalarPrefix) {
  scalar_prefix_graph_t infer;
  transactional_graph_fake_t state;
  cuda_driver_api cuda = transactional_graph_api(state);

  auto result = executable_t::build(
    cuda,
    {
      .context = reinterpret_cast<CUcontext>(1u),
      .infer_child = &infer.graph,
      .decision_record = 0x4000u,
      .request_record = 0x4040u,
    }
  );

  ASSERT_TRUE(result.ready())
    << "failure=" << static_cast<unsigned>(result.failure())
    << " cuda=" << result.cuda_result();
  ASSERT_NE(state.allocated_mirrors[0], 0u);
  ASSERT_NE(state.allocated_mirrors[1], 0u);
  EXPECT_EQ(infer.first_copy.memcpy.srcMemoryType, infer.original_params[0].srcMemoryType);
  EXPECT_EQ(infer.first_copy.memcpy.srcDevice, infer.original_params[0].srcDevice);
  EXPECT_EQ(infer.first_copy.memcpy.dstDevice, infer.original_params[0].dstDevice);
  EXPECT_EQ(infer.second_copy.memcpy.srcMemoryType, infer.original_params[1].srcMemoryType);
  EXPECT_EQ(infer.second_copy.memcpy.srcDevice, infer.original_params[1].srcDevice);
  EXPECT_EQ(infer.second_copy.memcpy.dstDevice, infer.original_params[1].dstDevice);
  ASSERT_EQ(state.embedded_infer.nodes.size(), 3u);
  EXPECT_EQ(state.embedded_infer.nodes[0]->memcpy.srcMemoryType, CU_MEMORYTYPE_DEVICE);
  EXPECT_EQ(state.embedded_infer.nodes[0]->memcpy.srcDevice, state.allocated_mirrors[0]);
  EXPECT_EQ(state.embedded_infer.nodes[0]->memcpy.dstDevice, infer.original_params[0].dstDevice);
  EXPECT_EQ(state.embedded_infer.nodes[1]->memcpy.srcMemoryType, CU_MEMORYTYPE_DEVICE);
  EXPECT_EQ(state.embedded_infer.nodes[1]->memcpy.srcDevice, state.allocated_mirrors[1]);
  EXPECT_EQ(state.embedded_infer.nodes[1]->memcpy.dstDevice, infer.original_params[1].dstDevice);

  // The two outer copies still read the original pageable addresses and populate the mirrors.
  ASSERT_GE(state.wrapper.nodes.size(), 2u);
  EXPECT_EQ(state.wrapper.nodes[0]->memcpy.srcDevice, infer.original_params[0].srcDevice);
  EXPECT_EQ(state.wrapper.nodes[0]->memcpy.dstDevice, state.allocated_mirrors[0]);
  EXPECT_EQ(state.wrapper.nodes[1]->memcpy.srcDevice, infer.original_params[1].srcDevice);
  EXPECT_EQ(state.wrapper.nodes[1]->memcpy.dstDevice, state.allocated_mirrors[1]);

  EXPECT_TRUE(result.reset());
  EXPECT_TRUE(result.empty());
  EXPECT_TRUE(state.executable_destroyed);
  EXPECT_TRUE(state.graph_destroyed);
  EXPECT_TRUE(state.module_unloaded);
  EXPECT_EQ(state.freed_mirrors.size(), 2u);
  EXPECT_EQ(infer.first_copy.memcpy.srcMemoryType, infer.original_params[0].srcMemoryType);
  EXPECT_EQ(infer.first_copy.memcpy.srcDevice, infer.original_params[0].srcDevice);
  EXPECT_EQ(infer.first_copy.memcpy.dstDevice, infer.original_params[0].dstDevice);
  EXPECT_EQ(infer.second_copy.memcpy.srcMemoryType, infer.original_params[1].srcMemoryType);
  EXPECT_EQ(infer.second_copy.memcpy.srcDevice, infer.original_params[1].srcDevice);
  EXPECT_EQ(infer.second_copy.memcpy.dstDevice, infer.original_params[1].dstDevice);
  transactional_graph_fake = nullptr;
}

TEST(CudaConditionalGraphContract, ParentDestroyFailureRetainsEmbeddedMirrorsForRetry) {
  scalar_prefix_graph_t infer;
  transactional_graph_fake_t state;
  cuda_driver_api cuda = transactional_graph_api(state);
  auto result = executable_t::build(
    cuda,
    {
      .context = reinterpret_cast<CUcontext>(1u),
      .infer_child = &infer.graph,
      .decision_record = 0x4000u,
      .request_record = 0x4040u,
    }
  );
  ASSERT_TRUE(result.ready());

  state.fail_graph_destroy = true;
  EXPECT_FALSE(result.reset());
  EXPECT_FALSE(result.empty());
  EXPECT_FALSE(result.ready());
  EXPECT_TRUE(state.executable_destroyed);
  EXPECT_FALSE(state.graph_destroyed);
  EXPECT_TRUE(state.freed_mirrors.empty());
  EXPECT_EQ(infer.first_copy.memcpy.srcDevice, infer.original_params[0].srcDevice);
  EXPECT_EQ(infer.second_copy.memcpy.srcDevice, infer.original_params[1].srcDevice);
  EXPECT_EQ(state.embedded_infer.nodes[1]->memcpy.srcMemoryType, CU_MEMORYTYPE_DEVICE);
  EXPECT_EQ(state.embedded_infer.nodes[1]->memcpy.srcDevice, state.allocated_mirrors[1]);

  state.fail_graph_destroy = false;
  EXPECT_TRUE(result.reset());
  EXPECT_TRUE(result.empty());
  EXPECT_EQ(state.freed_mirrors.size(), 2u);
  transactional_graph_fake = nullptr;
}

TEST(CudaConditionalGraphContract, UnsafeAbandonMakesDestructorIssueNoCudaCalls) {
  scalar_prefix_graph_t infer;
  transactional_graph_fake_t state;
  cuda_driver_api cuda = transactional_graph_api(state);
  {
    auto result = executable_t::build(
      cuda,
      {
        .context = reinterpret_cast<CUcontext>(1u),
        .infer_child = &infer.graph,
        .decision_record = 0x4000u,
        .request_record = 0x4040u,
      }
    );
    ASSERT_TRUE(result.ready());
    ASSERT_FALSE(result.empty());

    result.abandon_unsafe();
    EXPECT_TRUE(result.empty());
    EXPECT_FALSE(result.ready());
    EXPECT_FALSE(state.executable_destroyed);
    EXPECT_FALSE(state.graph_destroyed);
    EXPECT_FALSE(state.module_unloaded);
    EXPECT_TRUE(state.freed_mirrors.empty());
  }
  EXPECT_FALSE(state.executable_destroyed);
  EXPECT_FALSE(state.graph_destroyed);
  EXPECT_FALSE(state.module_unloaded);
  EXPECT_TRUE(state.freed_mirrors.empty());
  transactional_graph_fake = nullptr;
}

TEST(CudaConditionalGraphContract, PartialEmbeddedRewriteFailureLeavesSourceUnmodified) {
  scalar_prefix_graph_t infer;
  transactional_graph_fake_t state;
  state.fail_set_params_call = 2u;
  cuda_driver_api cuda = transactional_graph_api(state);

  auto result = executable_t::build(
    cuda,
    {
      .context = reinterpret_cast<CUcontext>(1u),
      .infer_child = &infer.graph,
      .decision_record = 0x4000u,
      .request_record = 0x4040u,
    }
  );

  EXPECT_FALSE(result.ready());
  EXPECT_TRUE(result.empty());
  EXPECT_EQ(result.failure(), build_failure_e::infer_child_scalar_rewrite_failed);
  EXPECT_EQ(result.cuda_result(), CUDA_ERROR_INVALID_HANDLE);
  EXPECT_TRUE(state.graph_destroyed);
  EXPECT_EQ(state.freed_mirrors.size(), 2u);
  EXPECT_EQ(infer.first_copy.memcpy.srcDevice, infer.original_params[0].srcDevice);
  EXPECT_EQ(infer.first_copy.memcpy.srcMemoryType, infer.original_params[0].srcMemoryType);
  EXPECT_EQ(infer.second_copy.memcpy.srcDevice, infer.original_params[1].srcDevice);
  EXPECT_EQ(infer.second_copy.memcpy.srcMemoryType, infer.original_params[1].srcMemoryType);
  transactional_graph_fake = nullptr;
}

TEST(CudaConditionalGraphAudit, RecursesThroughLegalChildGraphs) {
  CUgraphNode_st kernel {.type = CU_GRAPH_NODE_TYPE_KERNEL};
  CUgraph_st nested {{&kernel}};
  CUgraphNode_st child {.type = CU_GRAPH_NODE_TYPE_GRAPH, .child = &nested};
  CUgraph_st root {{&child}};
  const auto result = audit_embeddable_child_graph(fake_audit_api(), &root);
  EXPECT_TRUE(result.legal());
  EXPECT_EQ(result.visited_graphs, 2u);
  EXPECT_EQ(result.visited_nodes, 2u);
  EXPECT_EQ(result.node_type_counts[CU_GRAPH_NODE_TYPE_GRAPH], 1u);
  EXPECT_EQ(result.node_type_counts[CU_GRAPH_NODE_TYPE_KERNEL], 1u);
}

TEST(CudaConditionalGraphAudit, InferenceChildrenRequireRecursiveKernelWork) {
  CUgraph_st empty_graph;
  auto result = audit_inference_child_graph(fake_audit_api(), &empty_graph);
  EXPECT_EQ(result.failure, audit_failure_e::missing_inference_kernel);
  EXPECT_EQ(result.visited_graphs, 1u);
  EXPECT_EQ(result.visited_nodes, 0u);

  CUgraphNode_st empty_node {.type = CU_GRAPH_NODE_TYPE_EMPTY};
  CUgraph_st empty_node_graph {{&empty_node}};
  result = audit_inference_child_graph(fake_audit_api(), &empty_node_graph);
  EXPECT_EQ(result.failure, audit_failure_e::missing_inference_kernel);
  EXPECT_EQ(result.node_type_counts[CU_GRAPH_NODE_TYPE_EMPTY], 1u);

  CUgraphNode_st nested_kernel {.type = CU_GRAPH_NODE_TYPE_KERNEL};
  CUgraph_st nested_graph {{&nested_kernel}};
  CUgraphNode_st child_node {.type = CU_GRAPH_NODE_TYPE_GRAPH, .child = &nested_graph};
  CUgraph_st root {{&child_node}};
  result = audit_inference_child_graph(fake_audit_api(), &root);
  EXPECT_TRUE(result.legal());
  EXPECT_EQ(result.node_type_counts[CU_GRAPH_NODE_TYPE_KERNEL], 1u);
}

TEST(CudaConditionalGraphContract, RejectsKernelFreeMandatoryAndOptionalInferenceChildren) {
  CUgraph_st empty_graph;
  transactional_graph_fake_t mandatory_state;
  cuda_driver_api mandatory_cuda = transactional_graph_api(mandatory_state);
  auto result = executable_t::build(
    mandatory_cuda,
    {
      .context = reinterpret_cast<CUcontext>(1u),
      .infer_child = &empty_graph,
      .decision_record = 0x4000u,
      .request_record = 0x4040u,
    }
  );
  EXPECT_FALSE(result.ready());
  EXPECT_TRUE(result.empty());
  EXPECT_EQ(
    result.failure(), build_failure_e::infer_child_missing_inference_kernel
  );
  EXPECT_EQ(
    result.audit_result().failure, audit_failure_e::missing_inference_kernel
  );
  transactional_graph_fake = nullptr;

  scalar_prefix_graph_t infer;
  CUgraphNode_st empty_node {.type = CU_GRAPH_NODE_TYPE_EMPTY};
  CUgraph_st empty_optional {{&empty_node}};
  transactional_graph_fake_t optional_state;
  cuda_driver_api optional_cuda = transactional_graph_api(optional_state);
  result = executable_t::build(
    optional_cuda,
    {
      .context = reinterpret_cast<CUcontext>(1u),
      .infer_child = &infer.graph,
      .optional_infer_child = &empty_optional,
      .decision_record = 0x4000u,
      .request_record = 0x4040u,
    }
  );
  EXPECT_FALSE(result.ready());
  EXPECT_TRUE(result.empty());
  EXPECT_EQ(
    result.failure(), build_failure_e::optional_infer_child_missing_inference_kernel
  );
  EXPECT_EQ(
    result.audit_result().failure, audit_failure_e::missing_inference_kernel
  );
  transactional_graph_fake = nullptr;
}

TEST(CudaConditionalGraphAudit, RejectsUnsupportedAndNestedConditionalNodes) {
  CUgraphNode_st host {.type = CU_GRAPH_NODE_TYPE_HOST};
  CUgraph_st host_graph {{&host}};
  auto result = audit_embeddable_child_graph(fake_audit_api(), &host_graph);
  EXPECT_EQ(result.failure, audit_failure_e::unsupported_node_type);
  EXPECT_EQ(result.rejected_type, CU_GRAPH_NODE_TYPE_HOST);
  EXPECT_EQ(result.node_type_counts[CU_GRAPH_NODE_TYPE_HOST], 1u);

  CUgraphNode_st conditional {.type = CU_GRAPH_NODE_TYPE_CONDITIONAL};
  CUgraph_st conditional_graph {{&conditional}};
  result = audit_embeddable_child_graph(fake_audit_api(), &conditional_graph);
  EXPECT_EQ(result.failure, audit_failure_e::nested_conditional);
  EXPECT_EQ(result.rejected_type, CU_GRAPH_NODE_TYPE_CONDITIONAL);
  EXPECT_EQ(result.node_type_counts[CU_GRAPH_NODE_TYPE_CONDITIONAL], 1u);
}

TEST(CudaConditionalGraphHardware, AlternatesBranchesAndFailsMalformedProposalToInfer) {
  constexpr std::uint64_t token = 0x18cc257e17cfdd6cull;
  conditional_hardware_fixture_t fixture;
  const auto initialized = fixture.initialize();
  if (initialized == conditional_hardware_fixture_t::initialize_result_e::unavailable) {
    GTEST_SKIP() << "CUDA conditional graphs are unavailable on this driver/device";
  }
  ASSERT_EQ(initialized, conditional_hardware_fixture_t::initialize_result_e::ready)
    << "failure=" << static_cast<int>(fixture.bridge_failure())
    << " cuda=" << static_cast<int>(fixture.bridge_cuda_result());

  const auto request = make_request(token);
  const auto reuse = make_proposal(branch_e::reuse, token);
  const auto infer = make_proposal(branch_e::infer, token);
  for (unsigned repetition = 0u; repetition < 8u; ++repetition) {
    EXPECT_TRUE(fixture.run(reuse, request, 0u, branch_e::reuse, true));
    EXPECT_TRUE(fixture.run(
      infer, request, conditional_hardware_fixture_t::infer_marker, branch_e::infer, true
    ));
  }

  auto malformed = reuse;
  malformed.decision_cookie ^= 1u;
  EXPECT_TRUE(fixture.run(
    malformed, request, conditional_hardware_fixture_t::infer_marker, branch_e::infer, true
  ));
  auto stale = make_proposal(branch_e::reuse, token - 1u);
  EXPECT_TRUE(fixture.run(
    stale, request, conditional_hardware_fixture_t::infer_marker, branch_e::infer, true
  ));
  auto bad_request = request;
  bad_request.magic ^= 1u;
  EXPECT_TRUE(fixture.run(
    reuse, bad_request, conditional_hardware_fixture_t::infer_marker, branch_e::infer, false
  ));
  EXPECT_TRUE(fixture.reset_restores_null_context());
}

TEST(CudaConditionalGraphHardware, ExecutesOptionalReuseElseChild) {
  constexpr std::uint64_t token = 0xaabbccddeeff0011ull;
  conditional_hardware_fixture_t fixture;
  const auto initialized = fixture.initialize(true);
  if (initialized == conditional_hardware_fixture_t::initialize_result_e::unavailable) {
    GTEST_SKIP() << "CUDA IF/ELSE conditional graphs are unavailable on this driver/device";
  }
  ASSERT_EQ(initialized, conditional_hardware_fixture_t::initialize_result_e::ready)
    << "failure=" << static_cast<int>(fixture.bridge_failure())
    << " cuda=" << static_cast<int>(fixture.bridge_cuda_result());

  const auto request = make_request(token);
  EXPECT_TRUE(fixture.run(
    make_proposal(branch_e::reuse, token), request,
    conditional_hardware_fixture_t::reuse_branch_marker, branch_e::reuse, true
  ));
  EXPECT_TRUE(fixture.run(
    make_proposal(branch_e::infer, token), request,
    conditional_hardware_fixture_t::infer_marker, branch_e::infer, true
  ));
}

TEST(CudaConditionalGraphHardware, OptionalSiblingRequiresAuthenticatedProposalAndWorkMode) {
  constexpr std::uint64_t token = 0x3141592653589793ull;
  conditional_hardware_fixture_t fixture;
  const auto initialized = fixture.initialize(false, false, true);
  if (initialized == conditional_hardware_fixture_t::initialize_result_e::unavailable) {
    GTEST_SKIP() << "CUDA conditional graphs are unavailable on this driver/device";
  }
  ASSERT_EQ(initialized, conditional_hardware_fixture_t::initialize_result_e::ready)
    << "failure=" << static_cast<int>(fixture.bridge_failure())
    << " cuda=" << static_cast<int>(fixture.bridge_cuda_result());

  const auto request = make_request(token, work_flag_e::optional_ocr);
  EXPECT_TRUE(fixture.run_optional(
    make_proposal(branch_e::infer, token),
    request,
    conditional_hardware_fixture_t::infer_marker,
    conditional_hardware_fixture_t::optional_infer_marker,
    branch_e::infer,
    true,
    true
  ));
  EXPECT_TRUE(fixture.run_optional(
    make_proposal(branch_e::reuse, token),
    request,
    0u,
    0u,
    branch_e::reuse,
    true,
    false
  ));

  const auto due_request = make_request(token, work_flag_e::optional_ocr_due);
  EXPECT_TRUE(fixture.run_optional(
    make_proposal(branch_e::infer, token),
    due_request,
    conditional_hardware_fixture_t::infer_marker,
    conditional_hardware_fixture_t::optional_infer_marker,
    branch_e::infer,
    true,
    true
  ));
  EXPECT_TRUE(fixture.run_optional(
    make_proposal(branch_e::reuse, token),
    due_request,
    0u,
    conditional_hardware_fixture_t::optional_infer_marker,
    branch_e::reuse,
    true,
    true
  ));

  const auto due_abstention_request = make_request(
    token, work_flag_e::subtitle_observation_due
  );
  EXPECT_TRUE(fixture.run_optional(
    make_proposal(branch_e::infer, token),
    due_abstention_request,
    conditional_hardware_fixture_t::infer_marker,
    0u,
    branch_e::infer,
    true,
    false
  ));
  EXPECT_TRUE(fixture.run_optional(
    make_proposal(branch_e::reuse, token),
    due_abstention_request,
    0u,
    0u,
    branch_e::reuse,
    true,
    false
  ));

  auto malformed = make_proposal(branch_e::infer, token);
  malformed.magic = 0u;
  EXPECT_TRUE(fixture.run_optional(
    malformed,
    request,
    conditional_hardware_fixture_t::infer_marker,
    0u,
    branch_e::infer,
    true,
    false
  )) << "Malformed PROP must fail depth open while leaving optional OCR off";

  auto invalid_request = request;
  invalid_request.magic = 0u;
  EXPECT_TRUE(fixture.run_optional(
    make_proposal(branch_e::infer, token),
    invalid_request,
    conditional_hardware_fixture_t::infer_marker,
    0u,
    branch_e::infer,
    false,
    false
  ));

  const auto retired_flag4_request = make_request(
    token, static_cast<work_flag_e>(4u)
  );
  EXPECT_TRUE(fixture.run_optional(
    make_proposal(branch_e::reuse, token),
    retired_flag4_request,
    conditional_hardware_fixture_t::infer_marker,
    0u,
    branch_e::infer,
    false,
    false
  )) << "Retired work value 4 must fail request authentication and keep OCR dormant";
}

TEST(CudaConditionalGraphHardware, MirrorsFixedDav2ScalarPrefixOnEveryLaunch) {
  conditional_hardware_fixture_t fixture;
  const auto initialized = fixture.initialize(false, true);
  if (initialized == conditional_hardware_fixture_t::initialize_result_e::unavailable) {
    GTEST_SKIP() << "CUDA conditional graphs are unavailable on this driver/device";
  }
  ASSERT_EQ(initialized, conditional_hardware_fixture_t::initialize_result_e::ready)
    << "CUDA result " << fixture.bridge_cuda_result() << ", bridge stage "
    << static_cast<unsigned>(fixture.bridge_failure());
  EXPECT_TRUE(fixture.scalar_source_unchanged());

  // A raw executable instantiated from the TensorRT-style source before wrapper construction
  // remains direct and independent. Destroying it must not invalidate the embedded child clone.
  fixture.set_scalar_seeds(37u, 61u);
  EXPECT_TRUE(fixture.run_raw(37061u));
  EXPECT_TRUE(fixture.destroy_raw_executable());

  constexpr std::uint64_t first_token = 0x1020304050607080ull;
  const auto first_request = make_request(first_token);
  EXPECT_TRUE(fixture.run(
    make_proposal(branch_e::infer, first_token),
    first_request,
    37061u,
    branch_e::infer,
    true
  ));

  fixture.set_scalar_seeds(29u, 43u);
  constexpr std::uint64_t reuse_token = first_token + 1u;
  const auto reuse_request = make_request(reuse_token);
  EXPECT_TRUE(fixture.run(
    make_proposal(branch_e::reuse, reuse_token),
    reuse_request,
    0u,
    branch_e::reuse,
    true
  ));

  // The host values are captured by address, not snapshotted at wrapper construction. The next
  // infer must observe the values submitted during this launch through the two device mirrors.
  constexpr std::uint64_t second_infer_token = reuse_token + 1u;
  const auto second_infer_request = make_request(second_infer_token);
  EXPECT_TRUE(fixture.run(
    make_proposal(branch_e::infer, second_infer_token),
    second_infer_request,
    29043u,
    branch_e::infer,
    true
  ));
  EXPECT_TRUE(fixture.scalar_source_unchanged());
  EXPECT_TRUE(fixture.reset_restores_null_context());
  EXPECT_TRUE(fixture.scalar_source_unchanged());
}

namespace {
  class topology_test_logger_t: public nvinfer1::ILogger {
  public:
#ifdef __GNUC__
    void msvc_dummy_destructor(char) noexcept override {}
#endif
    void log(Severity severity, const char *message) noexcept override {
      if (severity <= Severity::kWARNING) {
        std::cerr << "TensorRT topology fixture: " << message << '\n';
      }
    }
  };

  template<typename T>
  void release_trt_test_interface(T *&value) {
    if (!value) {
      return;
    }
#ifdef __GNUC__
    value->msvc_dummy_destructor(1);
#else
    delete value;
#endif
    value = nullptr;
  }
}

// Opt-in hardware evidence for a real serialized OCR plan. The environment variable keeps this
// machine-local engine out of portable unit-test assumptions while letting reviewers inspect the
// exact recursively captured node topology before it is embedded in a conditional body.
TEST(CudaConditionalGraphHardware, AuditsCapturedTensorRtOcrTopology) {
  const char *const engine_env = std::getenv("SUNSHINE_TEST_OCR_ENGINE");
  if (!engine_env || *engine_env == '\0') {
    GTEST_SKIP() << "SUNSHINE_TEST_OCR_ENGINE is not set";
  }
  const std::filesystem::path engine_path {engine_env};
  std::ifstream input(engine_path, std::ios::binary);
  ASSERT_TRUE(input.is_open()) << engine_path;
  const std::vector<char> engine_bytes {
    std::istreambuf_iterator<char> {input}, std::istreambuf_iterator<char> {}
  };
  ASSERT_FALSE(engine_bytes.empty());

  auto &cuda = cuda_driver_api::get();
  ASSERT_TRUE(cuda.is_valid());
  ASSERT_EQ(cuda.cuInit(0u), CUDA_SUCCESS);
  CUdevice device = 0;
  ASSERT_EQ(cuda.cuDeviceGet(&device, 0), CUDA_SUCCESS);
  CUcontext cuda_context = nullptr;
  ASSERT_EQ(cuda.cuDevicePrimaryCtxRetain(&cuda_context, device), CUDA_SUCCESS);
  ASSERT_NE(cuda_context, nullptr);
  ASSERT_EQ(cuda.cuCtxSetCurrent(cuda_context), CUDA_SUCCESS);

  topology_test_logger_t logger;
  ASSERT_TRUE(initLibNvInferPlugins(&logger, ""));
  nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
  ASSERT_NE(runtime, nullptr);
  nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(
    engine_bytes.data(), engine_bytes.size()
  );
  ASSERT_NE(engine, nullptr);
  nvinfer1::IExecutionContext *execution = engine->createExecutionContext();
  ASSERT_NE(execution, nullptr);

  nvinfer1::Dims input_dims {};
  input_dims.nbDims = 4;
  input_dims.d[0] = 1;
  input_dims.d[1] = 3;
  input_dims.d[2] = models::ocr_engine_height;
  input_dims.d[3] = models::ocr_engine_width;
  ASSERT_TRUE(execution->setInputShape("x", input_dims));
  constexpr std::size_t input_bytes =
    3u * static_cast<std::size_t>(models::ocr_engine_width) *
    static_cast<std::size_t>(models::ocr_engine_height) * sizeof(float);
  constexpr std::size_t output_bytes =
    static_cast<std::size_t>(models::ocr_engine_width) *
    static_cast<std::size_t>(models::ocr_engine_height) * sizeof(float);
  CUdeviceptr device_input = 0u;
  CUdeviceptr device_output = 0u;
  ASSERT_EQ(cuda.cuMemAlloc(&device_input, input_bytes), CUDA_SUCCESS);
  ASSERT_EQ(cuda.cuMemAlloc(&device_output, output_bytes), CUDA_SUCCESS);
  ASSERT_TRUE(execution->setTensorAddress(
    "x", reinterpret_cast<void *>(device_input)
  ));
  ASSERT_TRUE(execution->setTensorAddress(
    "fetch_name_0", reinterpret_cast<void *>(device_output)
  ));
  CUstream stream = nullptr;
  ASSERT_EQ(cuda.cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  ASSERT_TRUE(execution->enqueueV3(stream));
  ASSERT_EQ(cuda.cuStreamSynchronize(stream), CUDA_SUCCESS);

  CUgraph graph = nullptr;
  ASSERT_EQ(
    cuda.cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED),
    CUDA_SUCCESS
  );
  const bool capture_enqueue = execution->enqueueV3(stream);
  const CUresult capture_end = cuda.cuStreamEndCapture(stream, &graph);
  ASSERT_TRUE(capture_enqueue);
  ASSERT_EQ(capture_end, CUDA_SUCCESS);
  ASSERT_NE(graph, nullptr);

  const auto audit = audit_embeddable_child_graph(cuda, graph);
  std::cout
    << "OCR captured topology: aux=" << engine->getNbAuxStreams()
    << " graphs=" << audit.visited_graphs << " nodes=" << audit.visited_nodes
    << " kernel=" << audit.node_type_counts[CU_GRAPH_NODE_TYPE_KERNEL]
    << " memcpy=" << audit.node_type_counts[CU_GRAPH_NODE_TYPE_MEMCPY]
    << " memset=" << audit.node_type_counts[CU_GRAPH_NODE_TYPE_MEMSET]
    << " child=" << audit.node_type_counts[CU_GRAPH_NODE_TYPE_GRAPH]
    << " wait-event=" << audit.node_type_counts[CU_GRAPH_NODE_TYPE_WAIT_EVENT]
    << " record-event=" << audit.node_type_counts[CU_GRAPH_NODE_TYPE_EVENT_RECORD]
    << " legal=" << audit.legal() << '\n';
  EXPECT_GT(audit.visited_nodes, 0u);
  if (const char *const expected = std::getenv("SUNSHINE_EXPECT_OCR_GRAPH_LEGAL")) {
    if (std::string_view {expected} == "1") {
      EXPECT_TRUE(audit.legal());
    } else if (std::string_view {expected} == "0") {
      EXPECT_FALSE(audit.legal());
    }
  }

  CUgraph mandatory_graph = nullptr;
  ASSERT_EQ(cuda.cuGraphCreate(&mandatory_graph, 0u), CUDA_SUCCESS);
  CUdeviceptr mandatory_marker = 0u;
  ASSERT_EQ(cuda.cuMemAlloc(&mandatory_marker, sizeof(std::uint32_t)), CUDA_SUCCESS);
  CUmodule marker_module = nullptr;
  CUfunction marker_writer = nullptr;
  ASSERT_EQ(
    cuda.cuModuleLoadDataEx(
      &marker_module, scalar_prefix_consumer_ptx, 0u, nullptr, nullptr
    ),
    CUDA_SUCCESS
  );
  ASSERT_EQ(
    cuda.cuModuleGetFunction(
      &marker_writer, marker_module, "write_inference_marker"
    ),
    CUDA_SUCCESS
  );
  ASSERT_NE(marker_writer, nullptr);
  CUdeviceptr mandatory_output = mandatory_marker;
  std::uint32_t mandatory_value = 0x51u;
  void *mandatory_args[] = {&mandatory_output, &mandatory_value};
  CUDA_KERNEL_NODE_PARAMS marker_params {};
  marker_params.func = marker_writer;
  marker_params.gridDimX = 1u;
  marker_params.gridDimY = 1u;
  marker_params.gridDimZ = 1u;
  marker_params.blockDimX = 1u;
  marker_params.blockDimY = 1u;
  marker_params.blockDimZ = 1u;
  marker_params.kernelParams = mandatory_args;
  CUgraphNode marker_node = nullptr;
  ASSERT_EQ(
    cuda.cuGraphAddKernelNode(
      &marker_node, mandatory_graph, nullptr, 0u, &marker_params
    ),
    CUDA_SUCCESS
  );
  ASSERT_NE(marker_node, nullptr);

  CUdeviceptr transaction = 0u;
  ASSERT_EQ(cuda.cuMemAlloc(&transaction, 64u), CUDA_SUCCESS);
  constexpr std::uint64_t token = 0x48f0ccab7a662511ull;
  const auto proposal = make_proposal(branch_e::infer, token);
  const auto request = make_request(token, work_flag_e::optional_ocr);
  ASSERT_EQ(
    cuda.cuMemcpyHtoD(transaction, &proposal, sizeof(proposal)), CUDA_SUCCESS
  );
  ASSERT_EQ(
    cuda.cuMemcpyHtoD(transaction + 32u, &request, sizeof(request)), CUDA_SUCCESS
  );
  auto wrapper = executable_t::build(
    cuda,
    {
      .context = cuda_context,
      .infer_child = mandatory_graph,
      .optional_infer_child = graph,
      .decision_record = transaction,
      .request_record = transaction + 32u,
    }
  );
  ASSERT_TRUE(wrapper.ready())
    << "failure=" << static_cast<unsigned>(wrapper.failure())
    << " cuda=" << wrapper.cuda_result()
    << " rejected_type=" << static_cast<int>(wrapper.audit_result().rejected_type);
  constexpr std::uint32_t output_sentinel = 0xdeadbeefu;
  ASSERT_EQ(
    cuda.cuMemcpyHtoD(device_output, &output_sentinel, sizeof(output_sentinel)),
    CUDA_SUCCESS
  );
  const std::uint32_t zero_marker = 0u;
  ASSERT_EQ(
    cuda.cuMemcpyHtoD(mandatory_marker, &zero_marker, sizeof(zero_marker)),
    CUDA_SUCCESS
  );
  ASSERT_EQ(cuda.cuGraphLaunch(wrapper.get(), stream), CUDA_SUCCESS);
  ASSERT_EQ(cuda.cuStreamSynchronize(stream), CUDA_SUCCESS);
  decision_record_t receipt {};
  ASSERT_EQ(
    cuda.cuMemcpyDtoH(&receipt, transaction, sizeof(receipt)), CUDA_SUCCESS
  );
  EXPECT_TRUE(authenticated_optional_ocr_receipt(receipt, request));
  std::uint32_t inferred_output_word = output_sentinel;
  ASSERT_EQ(
    cuda.cuMemcpyDtoH(
      &inferred_output_word, device_output, sizeof(inferred_output_word)
    ),
    CUDA_SUCCESS
  );
  EXPECT_NE(inferred_output_word, output_sentinel)
    << "The authenticated optional OCR child must execute the real OCR graph";

  // Keep the same instantiated superset but omit the optional request flag. The mandatory DAV2
  // marker must still execute while the embedded 132-kernel OCR graph remains completely dormant.
  // Production uses this launch shape for native suppression while OCR interop is intentionally
  // unmapped.
  const auto no_optional_request = make_request(token, work_flag_e::none);
  ASSERT_EQ(
    cuda.cuMemcpyHtoD(transaction, &proposal, sizeof(proposal)), CUDA_SUCCESS
  );
  ASSERT_EQ(
    cuda.cuMemcpyHtoD(
      transaction + 32u, &no_optional_request, sizeof(no_optional_request)
    ),
    CUDA_SUCCESS
  );
  ASSERT_EQ(
    cuda.cuMemcpyHtoD(device_output, &output_sentinel, sizeof(output_sentinel)),
    CUDA_SUCCESS
  );
  ASSERT_EQ(cuda.cuGraphLaunch(wrapper.get(), stream), CUDA_SUCCESS);
  ASSERT_EQ(cuda.cuStreamSynchronize(stream), CUDA_SUCCESS);
  ASSERT_EQ(
    cuda.cuMemcpyDtoH(&receipt, transaction, sizeof(receipt)), CUDA_SUCCESS
  );
  EXPECT_TRUE(authenticated_receipt(receipt, no_optional_request));
  EXPECT_EQ(receipt.decision, static_cast<std::uint32_t>(branch_e::infer));
  EXPECT_FALSE(authenticated_optional_infer_receipt(receipt, no_optional_request));
  std::uint32_t mandatory_output_word = 0u;
  ASSERT_EQ(
    cuda.cuMemcpyDtoH(
      &mandatory_output_word, mandatory_marker, sizeof(mandatory_output_word)
    ),
    CUDA_SUCCESS
  );
  EXPECT_EQ(mandatory_output_word, 0x51u)
    << "The mandatory infer child must execute when only the optional handle is dormant";
  std::uint32_t dormant_output_word = 0u;
  ASSERT_EQ(
    cuda.cuMemcpyDtoH(
      &dormant_output_word, device_output, sizeof(dormant_output_word)
    ),
    CUDA_SUCCESS
  );
  EXPECT_EQ(dormant_output_word, output_sentinel)
    << "An embedded OCR child must remain dormant when the request does not arm it";

  const auto reuse_proposal = make_proposal(branch_e::reuse, token);
  ASSERT_EQ(
    cuda.cuMemcpyHtoD(transaction, &reuse_proposal, sizeof(reuse_proposal)),
    CUDA_SUCCESS
  );
  ASSERT_EQ(
    cuda.cuMemcpyHtoD(transaction + 32u, &request, sizeof(request)), CUDA_SUCCESS
  );
  ASSERT_EQ(
    cuda.cuMemcpyHtoD(device_output, &output_sentinel, sizeof(output_sentinel)),
    CUDA_SUCCESS
  );
  ASSERT_EQ(cuda.cuGraphLaunch(wrapper.get(), stream), CUDA_SUCCESS);
  ASSERT_EQ(cuda.cuStreamSynchronize(stream), CUDA_SUCCESS);
  ASSERT_EQ(
    cuda.cuMemcpyDtoH(&receipt, transaction, sizeof(receipt)), CUDA_SUCCESS
  );
  EXPECT_TRUE(authenticated_receipt(receipt, request));
  EXPECT_EQ(receipt.decision, static_cast<std::uint32_t>(branch_e::reuse));
  EXPECT_FALSE(authenticated_optional_ocr_receipt(receipt, request));
  std::uint32_t reused_output_word = 0u;
  ASSERT_EQ(
    cuda.cuMemcpyDtoH(&reused_output_word, device_output, sizeof(reused_output_word)),
    CUDA_SUCCESS
  );
  EXPECT_EQ(reused_output_word, output_sentinel)
    << "Authenticated reuse must keep the optional OCR child dormant";
  EXPECT_TRUE(wrapper.reset());
  EXPECT_EQ(cuda.cuGraphDestroy(mandatory_graph), CUDA_SUCCESS);
  EXPECT_EQ(cuda.cuModuleUnload(marker_module), CUDA_SUCCESS);
  EXPECT_EQ(cuda.cuMemFree(transaction), CUDA_SUCCESS);
  EXPECT_EQ(cuda.cuMemFree(mandatory_marker), CUDA_SUCCESS);

  EXPECT_EQ(cuda.cuGraphDestroy(graph), CUDA_SUCCESS);
  EXPECT_EQ(cuda.cuStreamDestroy(stream), CUDA_SUCCESS);
  EXPECT_EQ(cuda.cuMemFree(device_output), CUDA_SUCCESS);
  EXPECT_EQ(cuda.cuMemFree(device_input), CUDA_SUCCESS);
  release_trt_test_interface(execution);
  release_trt_test_interface(engine);
  release_trt_test_interface(runtime);
  EXPECT_EQ(cuda.cuCtxSetCurrent(nullptr), CUDA_SUCCESS);
  EXPECT_EQ(cuda.cuDevicePrimaryCtxRelease(device), CUDA_SUCCESS);
}
