#include "src/cuda_conditional_graph.h"
#include "src/host_sbs_direct_glyph_gpu.h"
#include "src/model_manager.h"
#include "src/utility.h"
#include "src/video_depth_estimator.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <gtest/gtest.h>
#include <iostream>
#include <iterator>
#include <limits>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <numeric>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
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
  CUevent event = nullptr;
  CUfunction kernel_function = nullptr;
};

namespace {
  using namespace cuda_conditional_graph;

  static_assert(std::is_move_constructible_v<executable_t>);
  static_assert(!std::is_move_assignable_v<executable_t>);

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
// Test-only companion: observe receipt while a child is active, then write its original marker.
// clock64 delay has a finite loop cap; cycles are not a wall-time/performance measurement.
.visible .entry observe_staged_then_mark(
    .param .u64 output_ptr,
    .param .u32 marker_value,
    .param .u64 record_ptr,
    .param .u64 probe_ptr,
    .param .u64 delay_cycles
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<9>;
    ld.param.u64 %rd1, [output_ptr];
    ld.param.u32 %r1, [marker_value];
    ld.param.u64 %rd2, [record_ptr];
    ld.param.u64 %rd3, [probe_ptr];
    ld.param.u64 %rd4, [delay_cycles];
    mov.u64 %rd5, %clock64;
    mov.u32 %r3, 0;
probe_delay:
    mov.u64 %rd6, %clock64;
    sub.u64 %rd7, %rd6, %rd5;
    setp.ge.u64 %p1, %rd7, %rd4;
    @%p1 bra probe_sample;
    add.u32 %r3, %r3, 1;
    setp.lt.u32 %p2, %r3, 10000000;
    @%p2 bra probe_delay;
probe_sample:
    ld.global.u32 %r2, [%rd2+24];
    st.global.u32 [%rd3], %r2;
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
    std::array<CUgraphNode_st, 16u> wrapper_nodes;
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
    bool fail_commit_function = false;
    bool fail_commit_node = false;
    std::size_t event_create_calls = 0u;
    std::size_t event_node_calls = 0u;
    std::size_t event_elapsed_calls = 0u;
    std::size_t fail_event_create_call = 0u;
    std::size_t fail_event_node_call = 0u;
    std::size_t fail_event_destroy_call = 0u;
    std::size_t event_destroy_calls = 0u;
    bool event_destroyed_before_graph = false;
    std::vector<CUevent> created_events;
    std::vector<CUevent> destroyed_events;
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
    CUfunction *function,
    CUmodule,
    const char *name
  ) {
    if (!function || !transactional_graph_fake) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    const bool commit = std::string_view(name) == "commit_after_children";
    if (commit && transactional_graph_fake->fail_commit_function) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    *function = reinterpret_cast<CUfunction>(commit ? 0x2120u : 0x2020u);
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
    node->event = nullptr;
    node->kernel_function = nullptr;
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
    const CUDA_KERNEL_NODE_PARAMS *params
  ) {
    if (!transactional_graph_fake || !node || !params || graph != &transactional_graph_fake->wrapper) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    if (params->func == reinterpret_cast<CUfunction>(0x2120u) && transactional_graph_fake->fail_commit_node) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    CUgraphNode created = next_transaction_node(CU_GRAPH_NODE_TYPE_KERNEL);
    if (!created) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    if (dependencies && dependency_count != 0u) {
      created->dependencies.assign(dependencies, dependencies + dependency_count);
    }
    created->kernel_function = params->func;
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

  CUresult __stdcall transaction_event_create(CUevent *event, unsigned int flags) {
    if (!transactional_graph_fake || !event || flags != CU_EVENT_DEFAULT) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    auto &state = *transactional_graph_fake;
    if (++state.event_create_calls == state.fail_event_create_call) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    *event = reinterpret_cast<CUevent>(0x5000u + state.event_create_calls * 0x100u);
    state.created_events.push_back(*event);
    return CUDA_SUCCESS;
  }

  CUresult __stdcall transaction_event_destroy(CUevent event) {
    if (!transactional_graph_fake || !event) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    auto &state = *transactional_graph_fake;
    if (++state.event_destroy_calls == state.fail_event_destroy_call) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    state.event_destroyed_before_graph |= !state.graph_destroyed;
    state.destroyed_events.push_back(event);
    return CUDA_SUCCESS;
  }

  CUresult __stdcall transaction_event_elapsed(float *ms, CUevent start, CUevent stop) {
    if (!transactional_graph_fake || !ms || !start || !stop) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    ++transactional_graph_fake->event_elapsed_calls;
    *ms = 0.5f;
    return CUDA_SUCCESS;
  }

  CUresult __stdcall transaction_graph_add_event_record_node(
    CUgraphNode *node, CUgraph graph, const CUgraphNode *dependencies,
    const std::size_t dependency_count, CUevent event
  ) {
    if (!transactional_graph_fake || !node || !event ||
        graph != &transactional_graph_fake->wrapper) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    auto &state = *transactional_graph_fake;
    if (++state.event_node_calls == state.fail_event_node_call) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    CUgraphNode created = next_transaction_node(CU_GRAPH_NODE_TYPE_EVENT_RECORD);
    if (!created) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    created->event = event;
    if (dependencies && dependency_count != 0u) {
      created->dependencies.assign(dependencies, dependencies + dependency_count);
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

  void enable_transaction_timing_api(cuda_driver_api &cuda) {
    cuda.cuEventCreate = transaction_event_create;
    cuda.cuEventDestroy = transaction_event_destroy;
    cuda.cuEventElapsedTime = transaction_event_elapsed;
    cuda.cuGraphAddEventRecordNode = transaction_graph_add_event_record_node;
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
      const bool with_optional_child = false,
      const bool with_branch_timing = false,
      const bool with_receipt_probe = false,
      const std::uint64_t depth_delay_cycles = 0u,
      const std::uint64_t optional_delay_cycles = 0u
    ) {
      cuda_ = &cuda_driver_api::get();
      receipt_probe_enabled_ = with_receipt_probe;
      if (with_receipt_probe && with_scalar_prefix) {
        return initialize_result_e::failed;
      }
      if (!cuda_->is_valid() || !cuda_->has_conditional_graph_support() ||
          !cuda_->cuDevicePrimaryCtxRetain || !cuda_->cuDevicePrimaryCtxRelease ||
          !cuda_->cuGraphAddKernelNode || !cuda_->cuGraphAddMemsetNode ||
          !cuda_->cuModuleLoadDataEx || !cuda_->cuModuleGetFunction ||
          !cuda_->cuModuleUnload || !cuda_->cuMemcpyHtoD ||
          !cuda_->cuMemcpyDtoH || !cuda_->cuGraphLaunch) {
        return initialize_result_e::unavailable;
      }
      if (with_branch_timing && !cuda_->has_graph_branch_timing_support()) {
        return initialize_result_e::unavailable;
      }
      if (cuda_->cuInit(0u) != CUDA_SUCCESS ||
          cuda_->cuCtxGetCurrent(&previous_context_) != CUDA_SUCCESS ||
          cuda_->cuDeviceGet(&device_, 0) != CUDA_SUCCESS ||
          cuda_->cuDevicePrimaryCtxRetain(&context_, device_) != CUDA_SUCCESS) {
        return initialize_result_e::unavailable;
      }
      retained_ = true;
      // The fixture writes its records with cuMemcpyHtoD from pageable memory, which returns
      // once the bytes are staged; the DMA to the device can still be queued on the legacy
      // stream. A blocking stream waits for it, so a launch never reads the previous records.
      if (cuda_->cuCtxSetCurrent(context_) != CUDA_SUCCESS || cuda_->cuStreamCreate(&stream_, CU_STREAM_DEFAULT) != CUDA_SUCCESS || cuda_->cuMemAlloc(&records_, sizeof(decision_record_t) + sizeof(request_record_t)) != CUDA_SUCCESS || cuda_->cuMemAlloc(&output_, sizeof(std::uint32_t)) != CUDA_SUCCESS || (with_optional_child && cuda_->cuMemAlloc(&optional_output_, sizeof(std::uint32_t)) != CUDA_SUCCESS) || (with_receipt_probe && cuda_->cuMemAlloc(&receipt_probe_, 2u * sizeof(std::uint32_t)) != CUDA_SUCCESS) || cuda_->cuGraphCreate(&child_, 0u) != CUDA_SUCCESS || cuda_->cuModuleLoadDataEx(&scalar_module_, scalar_prefix_consumer_ptx, 0u, nullptr, nullptr) != CUDA_SUCCESS || cuda_->cuModuleGetFunction(&marker_writer_, scalar_module_, "write_inference_marker") != CUDA_SUCCESS || !marker_writer_) {
        return initialize_result_e::failed;
      }

      scalar_prefix_ = with_scalar_prefix;
      if (with_receipt_probe && (cuda_->cuModuleGetFunction(&receipt_probe_writer_, scalar_module_, "observe_staged_then_mark") != CUDA_SUCCESS || !receipt_probe_writer_)) {
        return initialize_result_e::failed;
      }

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
        CUdeviceptr records = records_, probe = receipt_probe_;
        auto delay = depth_delay_cycles;
        void *kernel_args[] = {&output, &value};
        void *probe_args[] = {&output, &value, &records, &probe, &delay};
        CUDA_KERNEL_NODE_PARAMS kernel {};
        kernel.func = with_receipt_probe ? receipt_probe_writer_ : marker_writer_;
        kernel.gridDimX = 1u;
        kernel.gridDimY = 1u;
        kernel.gridDimZ = 1u;
        kernel.blockDimX = 1u;
        kernel.blockDimY = 1u;
        kernel.blockDimZ = 1u;
        kernel.kernelParams = with_receipt_probe ? probe_args : kernel_args;
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
        CUdeviceptr records = records_, probe = receipt_probe_ + sizeof(std::uint32_t);
        auto delay = optional_delay_cycles;
        void *kernel_args[] = {&optional_output, &value};
        void *probe_args[] = {&optional_output, &value, &records, &probe, &delay};
        CUDA_KERNEL_NODE_PARAMS kernel {};
        kernel.func = with_receipt_probe ? receipt_probe_writer_ : marker_writer_;
        kernel.gridDimX = 1u;
        kernel.gridDimY = 1u;
        kernel.gridDimZ = 1u;
        kernel.blockDimX = 1u;
        kernel.blockDimY = 1u;
        kernel.blockDimZ = 1u;
        kernel.kernelParams = with_receipt_probe ? probe_args : kernel_args;
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
      auto bridge_candidate = executable_t::build(
        *cuda_,
        {
          .context = context_,
          .infer_child = child_,
          .optional_infer_child = optional_child_,
          .reuse_child = reuse_child_,
          .decision_record = records_,
          .request_record = records_ + sizeof(decision_record_t),
          .record_branch_timing = with_branch_timing,
        }
      );
      if (!bridge_.adopt_from_empty(std::move(bridge_candidate))) {
        return initialize_result_e::failed;
      }
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

    bool branch_timing(bool current_root_complete, branch_completion_timing_t &timing) const {
      return bridge_.read_branch_completion_timing(timing, current_root_complete) == CUDA_SUCCESS;
    }

    bool reset_receipt_probe() {
      if (!receipt_probe_enabled_) {
        return true;
      }
      const std::array<std::uint32_t, 2u> unset {0xffffffffu, 0xffffffffu};
      return receipt_probe_ && cuda_->cuMemcpyHtoD(
                                 receipt_probe_,
                                 unset.data(),
                                 sizeof(unset)
                               ) == CUDA_SUCCESS;
    }

    bool staged_receipt_was_uncommitted(const bool depth_ran, const bool optional_ran) {
      if (!receipt_probe_enabled_ || !receipt_probe_) {
        return false;
      }
      std::array<std::uint32_t, 2u> observed {};
      return cuda_->cuMemcpyDtoH(observed.data(), receipt_probe_, sizeof(observed)) == CUDA_SUCCESS &&
             observed[0] == (depth_ran ? 0u : 0xffffffffu) &&
             observed[1] == (optional_ran ? 0u : 0xffffffffu);
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
      if (!reset_receipt_probe() || cuda_->cuMemcpyHtoD(output_, &zero, sizeof(zero)) != CUDA_SUCCESS || cuda_->cuMemcpyHtoD(records_, &proposal, sizeof(proposal)) != CUDA_SUCCESS || cuda_->cuMemcpyHtoD(records_ + sizeof(proposal), &request, sizeof(request)) != CUDA_SUCCESS || cuda_->cuGraphLaunch(bridge_.get(), stream_) != CUDA_SUCCESS || cuda_->cuStreamSynchronize(stream_) != CUDA_SUCCESS) {
        return false;
      }

      std::uint32_t observed_output = 0u;
      if (cuda_->cuMemcpyDtoH(
            &observed_output, output_, sizeof(observed_output)
          ) != CUDA_SUCCESS ||
          cuda_->cuMemcpyDtoH(&receipt, records_, sizeof(receipt)) != CUDA_SUCCESS) {
        return false;
      }
      const auto expected = resolve_proposal(proposal, request, optional_child_ != nullptr);
      return std::memcmp(&receipt, &expected, sizeof(receipt)) == 0 &&
             observed_output == expected_output &&
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
      if (!reset_receipt_probe() || cuda_->cuMemcpyHtoD(output_, &zero, sizeof(zero)) != CUDA_SUCCESS || cuda_->cuMemcpyHtoD(optional_output_, &zero, sizeof(zero)) != CUDA_SUCCESS || cuda_->cuMemcpyHtoD(records_, &proposal, sizeof(proposal)) != CUDA_SUCCESS || cuda_->cuMemcpyHtoD(records_ + sizeof(proposal), &request, sizeof(request)) != CUDA_SUCCESS || cuda_->cuGraphLaunch(bridge_.get(), stream_) != CUDA_SUCCESS || cuda_->cuStreamSynchronize(stream_) != CUDA_SUCCESS) {
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
      const auto expected = resolve_proposal(proposal, request, true);
      return std::memcmp(&receipt, &expected, sizeof(receipt)) == 0 &&
             observed_depth == expected_depth_output &&
             observed_optional == expected_optional_output &&
             authenticated_receipt(receipt, request) == expected_authenticated_receipt &&
             authenticated_optional_ocr_receipt(receipt, request) ==
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
      if (receipt_probe_ && cuda_->cuMemFree) {
        cuda_->cuMemFree(receipt_probe_);
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
      receipt_probe_ = 0u;
      receipt_probe_writer_ = nullptr;
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
    CUdeviceptr receipt_probe_ = 0u;
    CUfunction receipt_probe_writer_ = nullptr;
    bool receipt_probe_enabled_ = false;
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

TEST(CudaConditionalGraphContract, RetiredIndependentSubtitleWorkIsRejected) {
  constexpr std::uint64_t token = 0x9a8b7c6d5e4f3021ull;
  for (const auto work : {4u, 8u, 16u, 3u}) {
    const auto request = make_request(token, static_cast<work_flag_e>(work));
    EXPECT_FALSE(authenticated_request(request));
    for (const auto branch : {branch_e::infer, branch_e::reuse}) {
      const auto receipt = resolve_proposal(make_proposal(branch, token), request);
      EXPECT_EQ(receipt.decision, static_cast<std::uint32_t>(branch_e::infer));
      EXPECT_FALSE(authenticated_receipt(receipt, request));
      EXPECT_EQ(receipt.reserved, 0u);
    }
  }
}

TEST(CudaConditionalGraphContract, SubtitleAbstentionRemainsCoupledToTheDepthBranch) {
  constexpr std::uint64_t token = 0x6b5a493827160f1eull;
  const auto request = make_request(token, work_flag_e::subtitle_observation);
  ASSERT_TRUE(authenticated_request(request));
  for (const auto branch : {branch_e::infer, branch_e::reuse}) {
    const auto receipt = resolve_proposal(make_proposal(branch, token), request);
    EXPECT_TRUE(authenticated_receipt(receipt, request));
    EXPECT_EQ(receipt.decision, static_cast<std::uint32_t>(branch));
    EXPECT_EQ(receipt.reserved, 0u);
    auto forged_ocr = receipt;
    forged_ocr.reserved = optional_ocr_receipt_magic;
    forged_ocr.decision_cookie ^= optional_ocr_receipt_magic;
    EXPECT_FALSE(authenticated_receipt(forged_ocr, request));
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
  EXPECT_EQ(receipt.magic, 0u) << "Invalid RQST must remain uncommitted after child completion";

  const auto zero_request = make_request(0u, work_flag_e::optional_infer);
  EXPECT_FALSE(authenticated_request(zero_request));
  EXPECT_FALSE(authenticated_optional_ocr_receipt(
    resolve_proposal(make_proposal(branch_e::infer, 0u), zero_request),
    zero_request
  ));
}

TEST(CudaConditionalGraphContract, EmbeddedPtxStagesThenCommitsAfterChildren) {
  const std::string_view ptx = bridge_ptx_source();
  EXPECT_NE(ptx.find(".extern .func cudaGraphSetConditional"), std::string_view::npos);
  EXPECT_NE(ptx.find("0x504f5250"), std::string_view::npos);  // PROP
  EXPECT_NE(ptx.find("0x54535152"), std::string_view::npos);  // RQST
  EXPECT_NE(ptx.find("0x47524243"), std::string_view::npos);  // CBRG
  EXPECT_NE(ptx.find("0x52434f4f"), std::string_view::npos);  // OOCR
  EXPECT_NE(ptx.find("setp.eq.u32 %p4, %r14, 0"), std::string_view::npos);
  EXPECT_NE(ptx.find("setp.eq.u32 %p24, %r14, 1"), std::string_view::npos);
  EXPECT_NE(ptx.find("setp.eq.u32 %p28, %r14, 2"), std::string_view::npos);
  EXPECT_EQ(ptx.find("setp.eq.u32 %p29, %r14, 8"), std::string_view::npos);
  EXPECT_EQ(ptx.find("setp.eq.u32 %p30, %r14, 16"), std::string_view::npos);
  EXPECT_EQ(ptx.find("setp.eq.u32 %p26, %r14, 8"), std::string_view::npos);
  EXPECT_NE(ptx.find("setp.eq.u32 %p20, %r14, 1"), std::string_view::npos);
  EXPECT_NE(ptx.find("setp.eq.u32 %p25, %r22, 1"), std::string_view::npos);
  EXPECT_NE(ptx.find("and.pred %p27, %p20, %p25"), std::string_view::npos)
    << "The OCR sibling requires both optional work and the infer branch";
  EXPECT_EQ(ptx.find("or.pred %p27"), std::string_view::npos)
    << "No independent subtitle mode can bypass the joint infer branch";
  const auto commit_entry = ptx.find(".visible .entry commit_after_children");
  ASSERT_NE(commit_entry, std::string_view::npos);
  const auto setter = ptx.substr(0u, commit_entry);
  const auto commit = ptx.substr(commit_entry);
  EXPECT_NE(setter.find("st.global.u32 [%rd3+24], 0"), std::string_view::npos);
  EXPECT_EQ(setter.find("0x47524243"), std::string_view::npos);
  EXPECT_NE(commit.find("setp.eq.u32 %p14, %r7, 0"), std::string_view::npos);
  const auto guard = commit.find("@!%p17 ret;");
  const auto receipt_publish = commit.find("st.global.u32 [%rd1+24], %r24");
  ASSERT_NE(guard, std::string_view::npos);
  ASSERT_NE(receipt_publish, std::string_view::npos);
  EXPECT_LT(guard, receipt_publish);
  EXPECT_EQ(commit.find("st.global", receipt_publish + 1u), std::string_view::npos);
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
  ASSERT_TRUE(result.adopt_from_empty(executable_t::build(
    unavailable,
    {
      .context = reinterpret_cast<CUcontext>(1u),
      .infer_child = &child,
      .decision_record = 16u,
      .request_record = 32u,
    }
  )));
  EXPECT_FALSE(result.ready());
  EXPECT_EQ(result.failure(), build_failure_e::invalid_descriptor)
    << "The two 32-byte authenticated records must not overlap";

  ASSERT_TRUE(result.adopt_from_empty(executable_t::build(
    unavailable,
    {
      .context = reinterpret_cast<CUcontext>(1u),
      .infer_child = &child,
      .decision_record = 16u,
      .request_record = 48u,
    }
  )));
  EXPECT_FALSE(result.ready());
  EXPECT_EQ(result.get(), nullptr);
  EXPECT_EQ(result.failure(), build_failure_e::driver_api_unavailable);

  constexpr CUdeviceptr last_aligned_address =
    std::numeric_limits<CUdeviceptr>::max() - 15u;
  ASSERT_TRUE(result.adopt_from_empty(executable_t::build(
    unavailable,
    {
      .context = reinterpret_cast<CUcontext>(1u),
      .infer_child = &child,
      .decision_record = last_aligned_address,
      .request_record = 0x1000u,
    }
  )));
  EXPECT_EQ(result.failure(), build_failure_e::invalid_descriptor)
    << "The complete 32-byte decision record must not wrap the device address space";

  ASSERT_TRUE(result.adopt_from_empty(executable_t::build(
    unavailable,
    {
      .context = reinterpret_cast<CUcontext>(1u),
      .infer_child = &child,
      .decision_record = 0x1000u,
      .request_record = last_aligned_address,
    }
  )));
  EXPECT_EQ(result.failure(), build_failure_e::invalid_descriptor)
    << "The complete 32-byte request record must not wrap the device address space";

  ASSERT_TRUE(result.adopt_from_empty(executable_t::build(
    unavailable,
    {
      .context = reinterpret_cast<CUcontext>(1u),
      .infer_child = &child,
      .decision_record = std::numeric_limits<CUdeviceptr>::max() - 31u,
      .request_record = 0x1000u,
    }
  )));
  EXPECT_EQ(result.failure(), build_failure_e::driver_api_unavailable)
    << "A 32-byte record whose final byte is exactly the address maximum does not wrap";
}


TEST(CudaConditionalGraphContract, BranchTimingIsOptionalAndDisabledAllocatesNothing) {
  for (const bool requested : {false, true}) {
    transactional_graph_fake_t state;
    cuda_driver_api cuda = transactional_graph_api(state);
    enable_transaction_timing_api(cuda);
    if (requested) {
      cuda.cuGraphAddEventRecordNode = nullptr;
    }
    EXPECT_TRUE(cuda.has_conditional_graph_support());
    CUgraphNode_st kernel {.type = CU_GRAPH_NODE_TYPE_KERNEL};
    CUgraph_st child {{&kernel}};
    auto graph = executable_t::build(cuda, {
      .context = reinterpret_cast<CUcontext>(1u),
      .infer_child = &child,
      .decision_record = 0x1000u,
      .request_record = 0x1100u,
      .record_branch_timing = requested,
    });
    ASSERT_TRUE(graph.ready());
    EXPECT_FALSE(graph.has_branch_completion_timing());
    EXPECT_EQ(state.event_create_calls, 0u);
    EXPECT_EQ(state.event_node_calls, 0u);
    branch_completion_timing_t timing;
    EXPECT_EQ(graph.read_branch_completion_timing(timing, true), CUDA_ERROR_NOT_SUPPORTED);
    EXPECT_EQ(state.event_elapsed_calls, 0u);
    ASSERT_TRUE(graph.reset());
  }
}

TEST(CudaConditionalGraphContract, BranchEventNodesStayInParentAndPreserveSiblingDependencies) {
  transactional_graph_fake_t state;
  cuda_driver_api cuda = transactional_graph_api(state);
  enable_transaction_timing_api(cuda);
  CUgraphNode_st kernel {.type = CU_GRAPH_NODE_TYPE_KERNEL};
  CUgraph_st child {{&kernel}};
  auto graph = executable_t::build(cuda, {
    .context = reinterpret_cast<CUcontext>(1u),
    .infer_child = &child,
    .optional_infer_child = &child,
    .decision_record = 0x1000u,
    .request_record = 0x1100u,
    .record_branch_timing = true,
  });
  ASSERT_TRUE(graph.ready());
  EXPECT_TRUE(graph.has_branch_completion_timing());
  ASSERT_EQ(state.created_events.size(), 3u);
  std::vector<CUgraphNode> events;
  std::vector<CUgraphNode> conditionals;
  for (const auto node : state.wrapper.nodes) {
    if (node->type == CU_GRAPH_NODE_TYPE_EVENT_RECORD) {
      events.push_back(node);
    } else if (node->type == CU_GRAPH_NODE_TYPE_CONDITIONAL) {
      conditionals.push_back(node);
    }
  }
  ASSERT_EQ(events.size(), 3u);
  ASSERT_EQ(conditionals.size(), 2u);
  ASSERT_EQ(events[0]->dependencies.size(), 1u);
  EXPECT_EQ(events[0]->dependencies[0]->type, CU_GRAPH_NODE_TYPE_KERNEL);
  for (std::size_t i = 0u; i < conditionals.size(); ++i) {
    EXPECT_EQ(conditionals[i]->dependencies, (std::vector<CUgraphNode> {events[0]}));
    EXPECT_EQ(events[i + 1u]->dependencies, (std::vector<CUgraphNode> {conditionals[i]}));
  }
  for (const auto &body : state.bodies) {
    for (const auto node : body.nodes) {
      EXPECT_NE(node->type, CU_GRAPH_NODE_TYPE_EVENT_RECORD);
      EXPECT_NE(node->type, CU_GRAPH_NODE_TYPE_WAIT_EVENT);
    }
  }
  branch_completion_timing_t timing;
  EXPECT_EQ(graph.read_branch_completion_timing(timing, false), CUDA_ERROR_NOT_READY);
  EXPECT_EQ(state.event_elapsed_calls, 0u) << "Prior-launch event records are not current-root proof";
  ASSERT_EQ(graph.read_branch_completion_timing(timing, true), CUDA_SUCCESS);
  EXPECT_EQ(state.event_elapsed_calls, 2u);
  EXPECT_TRUE(timing.has_optional);
  EXPECT_GE(timing.depth_ms, 0.0f);
  EXPECT_GE(timing.optional_ms, 0.0f);
  ASSERT_TRUE(graph.reset());
  EXPECT_EQ(state.destroyed_events, state.created_events);
  EXPECT_FALSE(state.event_destroyed_before_graph);
}

TEST(CudaConditionalGraphContract, CommitJoinsBothSiblingsWithOrWithoutOptionalWork) {
  for (const bool optional : {false, true}) {
    for (const bool timing : {false, true}) {
      transactional_graph_fake_t state;
      cuda_driver_api cuda = transactional_graph_api(state);
      enable_transaction_timing_api(cuda);
      CUgraphNode_st kernel {.type = CU_GRAPH_NODE_TYPE_KERNEL};
      CUgraph_st child {{&kernel}};
      auto graph = executable_t::build(cuda, {
                                               .context = reinterpret_cast<CUcontext>(1u),
                                               .infer_child = &child,
                                               .optional_infer_child = optional ? &child : nullptr,
                                               .decision_record = 0x1000u,
                                               .request_record = 0x1100u,
                                               .record_branch_timing = timing,
                                             });
      ASSERT_TRUE(graph.ready());
      std::vector<CUgraphNode> conditionals;
      std::vector<CUgraphNode> commits;
      for (const auto node : state.wrapper.nodes) {
        if (node->type == CU_GRAPH_NODE_TYPE_CONDITIONAL) {
          conditionals.push_back(node);
        }
        if (node->kernel_function == reinterpret_cast<CUfunction>(0x2120u)) {
          commits.push_back(node);
        }
      }
      ASSERT_EQ(conditionals.size(), 2u);
      ASSERT_EQ(commits.size(), 1u);
      EXPECT_EQ(commits[0]->dependencies, conditionals);
      ASSERT_TRUE(graph.reset());
    }
  }
}

TEST(CudaConditionalGraphContract, CommitBuildFailureNeverExposesExecutable) {
  for (const bool missing_function : {false, true}) {
    transactional_graph_fake_t state;
    state.fail_commit_function = missing_function;
    state.fail_commit_node = !missing_function;
    cuda_driver_api cuda = transactional_graph_api(state);
    CUgraphNode_st kernel {.type = CU_GRAPH_NODE_TYPE_KERNEL};
    CUgraph_st child {{&kernel}};
    auto graph = executable_t::build(cuda, {
                                             .context = reinterpret_cast<CUcontext>(1u),
                                             .infer_child = &child,
                                             .decision_record = 0x1000u,
                                             .request_record = 0x1100u,
                                           });
    EXPECT_FALSE(graph.ready());
    EXPECT_EQ(graph.get(), nullptr);
    EXPECT_EQ(graph.failure(), missing_function ? build_failure_e::commit_function_missing : build_failure_e::commit_node_add_failed);
    EXPECT_TRUE(state.module_unloaded);
    EXPECT_EQ(state.graph_destroyed, !missing_function);
    EXPECT_TRUE(graph.empty());
  }
}

TEST(CudaConditionalGraphContract, TimingRequiresExactRootAndCannotOverwritePendingLaunch) {
  branch_timing_pending_t pending;
  EXPECT_FALSE(pending.arm(0u, 11u));
  ASSERT_TRUE(pending.arm(101u, 11u));
  EXPECT_FALSE(pending.arm(102u, 12u));
  EXPECT_FALSE(pending.consume(101u, false));
  EXPECT_FALSE(pending.consume(100u, true));
  EXPECT_EQ(pending.token, 101u);
  EXPECT_EQ(pending.consume(101u, true), std::optional<std::uint64_t> {11u});
  EXPECT_FALSE(pending.consume(101u, true));
  ASSERT_TRUE(pending.arm(102u, 12u));
  pending.clear();
  EXPECT_FALSE(pending.consume(102u, true));
}

TEST(CudaConditionalGraphContract, TimingPartialFailuresReleaseOnlySafeGraphOperands) {
  for (const bool create_failure : {false, true}) {
    for (std::size_t failure = 1u; failure <= 3u; ++failure) {
      transactional_graph_fake_t state;
      cuda_driver_api cuda = transactional_graph_api(state);
      enable_transaction_timing_api(cuda);
      if (create_failure) {
        state.fail_event_create_call = failure;
      } else {
        state.fail_event_node_call = failure;
      }
      CUgraphNode_st kernel {.type = CU_GRAPH_NODE_TYPE_KERNEL};
      CUgraph_st child {{&kernel}};
      auto graph = executable_t::build(cuda, {
        .context = reinterpret_cast<CUcontext>(1u),
        .infer_child = &child,
        .optional_infer_child = &child,
        .decision_record = 0x1000u,
        .request_record = 0x1100u,
        .record_branch_timing = true,
      });
      EXPECT_FALSE(graph.ready());
      EXPECT_TRUE(graph.empty());
      EXPECT_EQ(graph.failure(), create_failure ? build_failure_e::diagnostic_event_create_failed :
                                                 build_failure_e::diagnostic_event_node_add_failed);
      EXPECT_EQ(state.destroyed_events, state.created_events);
      EXPECT_FALSE(state.event_destroyed_before_graph);
    }
  }
}

TEST(CudaConditionalGraphContract, TimingEventsFollowMoveAndMonotonicQuiescentTeardown) {
  transactional_graph_fake_t state;
  cuda_driver_api cuda = transactional_graph_api(state);
  enable_transaction_timing_api(cuda);
  CUgraphNode_st kernel {.type = CU_GRAPH_NODE_TYPE_KERNEL};
  CUgraph_st child {{&kernel}};
  auto original = executable_t::build(cuda, {
    .context = reinterpret_cast<CUcontext>(1u),
    .infer_child = &child,
    .optional_infer_child = &child,
    .decision_record = 0x1000u,
    .request_record = 0x1100u,
    .record_branch_timing = true,
  });
  ASSERT_TRUE(original.ready());
  executable_t moved(std::move(original));
  EXPECT_TRUE(original.empty());
  EXPECT_TRUE(moved.has_branch_completion_timing());
  state.fail_graph_destroy = true;
  EXPECT_FALSE(moved.reset());
  EXPECT_TRUE(state.destroyed_events.empty());
  EXPECT_FALSE(moved.empty());
  state.fail_graph_destroy = false;
  state.fail_event_destroy_call = 2u;
  EXPECT_FALSE(moved.reset());
  ASSERT_EQ(state.destroyed_events.size(), 1u);
  EXPECT_FALSE(state.module_unloaded);
  EXPECT_FALSE(moved.empty());
  state.fail_event_destroy_call = 0u;
  EXPECT_TRUE(moved.reset());
  EXPECT_EQ(state.destroyed_events, state.created_events);
  EXPECT_FALSE(state.event_destroyed_before_graph);
}

TEST(CudaConditionalGraphContract, UnsafeAbandonRetainsGraphTimingEvents) {
  transactional_graph_fake_t state;
  cuda_driver_api cuda = transactional_graph_api(state);
  enable_transaction_timing_api(cuda);
  CUgraphNode_st kernel {.type = CU_GRAPH_NODE_TYPE_KERNEL};
  CUgraph_st child {{&kernel}};
  {
    auto graph = executable_t::build(cuda, {
      .context = reinterpret_cast<CUcontext>(1u),
      .infer_child = &child,
      .decision_record = 0x1000u,
      .request_record = 0x1100u,
      .record_branch_timing = true,
    });
    ASSERT_TRUE(graph.ready());
    graph.abandon_unsafe();
    EXPECT_TRUE(graph.empty());
  }
  EXPECT_TRUE(state.destroyed_events.empty());
  EXPECT_FALSE(state.graph_destroyed);
  EXPECT_FALSE(state.executable_destroyed);
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

TEST(CudaConditionalGraphContract, AdoptFromEmptyTransfersOwnershipWithoutCudaCalls) {
  scalar_prefix_graph_t infer;
  transactional_graph_fake_t state;
  cuda_driver_api cuda = transactional_graph_api(state);
  auto source = executable_t::build(
    cuda,
    {
      .context = reinterpret_cast<CUcontext>(1u),
      .infer_child = &infer.graph,
      .decision_record = 0x4000u,
      .request_record = 0x4040u,
    }
  );
  ASSERT_TRUE(source.ready());
  const CUgraphExec source_handle = source.get();
  const auto source_audit = source.audit_result();
  executable_t destination;

  ASSERT_TRUE(destination.adopt_from_empty(std::move(source)));
  EXPECT_TRUE(destination.ready());
  EXPECT_EQ(destination.get(), source_handle);
  EXPECT_EQ(destination.audit_result().failure, source_audit.failure);
  EXPECT_EQ(destination.audit_result().visited_graphs, source_audit.visited_graphs);
  EXPECT_TRUE(source.empty());
  EXPECT_FALSE(source.ready());
  EXPECT_FALSE(state.executable_destroyed);
  EXPECT_FALSE(state.graph_destroyed);
  EXPECT_FALSE(state.module_unloaded);
  EXPECT_TRUE(state.freed_mirrors.empty());

  EXPECT_TRUE(destination.reset());
  transactional_graph_fake = nullptr;
}

TEST(CudaConditionalGraphContract, AdoptFromEmptyRejectsTwoLiveOwners) {
  scalar_prefix_graph_t destination_infer;
  transactional_graph_fake_t destination_state;
  cuda_driver_api destination_cuda = transactional_graph_api(destination_state);
  auto destination = executable_t::build(
    destination_cuda,
    {
      .context = reinterpret_cast<CUcontext>(1u),
      .infer_child = &destination_infer.graph,
      .decision_record = 0x4000u,
      .request_record = 0x4040u,
    }
  );
  ASSERT_TRUE(destination.ready());

  scalar_prefix_graph_t source_infer;
  transactional_graph_fake_t source_state;
  cuda_driver_api source_cuda = transactional_graph_api(source_state);
  auto source = executable_t::build(
    source_cuda,
    {
      .context = reinterpret_cast<CUcontext>(1u),
      .infer_child = &source_infer.graph,
      .decision_record = 0x5000u,
      .request_record = 0x5040u,
    }
  );
  ASSERT_TRUE(source.ready());
  const CUgraphExec destination_handle = destination.get();
  const CUgraphExec source_handle = source.get();

  EXPECT_FALSE(destination.adopt_from_empty(std::move(source)));
  EXPECT_TRUE(destination.ready());
  EXPECT_EQ(destination.get(), destination_handle);
  EXPECT_TRUE(source.ready());
  EXPECT_EQ(source.get(), source_handle);
  EXPECT_FALSE(destination_state.executable_destroyed);
  EXPECT_FALSE(destination_state.graph_destroyed);
  EXPECT_TRUE(destination_state.freed_mirrors.empty());
  EXPECT_FALSE(source_state.executable_destroyed);
  EXPECT_FALSE(source_state.graph_destroyed);
  EXPECT_TRUE(source_state.freed_mirrors.empty());

  transactional_graph_fake = &destination_state;
  EXPECT_TRUE(destination.reset());
  transactional_graph_fake = &source_state;
  EXPECT_TRUE(source.reset());
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
  ASSERT_TRUE(result.adopt_from_empty(executable_t::build(
    optional_cuda,
    {
      .context = reinterpret_cast<CUcontext>(1u),
      .infer_child = &infer.graph,
      .optional_infer_child = &empty_optional,
      .decision_record = 0x4000u,
      .request_record = 0x4040u,
    }
  )));
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

TEST(CudaConditionalGraphHardware, ReceiptStaysUncommittedInsideEitherSiblingOrder) {
  constexpr std::uint64_t token = 0x13572468abcdef01ull;
  // Delay opposite siblings in separate immutable wrappers. Tests assert ordering, not timing.
  for (const bool optional : {false, true}) {
    for (const bool delay_depth : {false, true}) {
      conditional_hardware_fixture_t fixture;
      const auto initialized = fixture.initialize(false, false, optional, false, true, delay_depth ? 2000000u : 0u, delay_depth ? 0u : 2000000u);
      if (initialized == conditional_hardware_fixture_t::initialize_result_e::unavailable) {
        GTEST_SKIP() << "CUDA conditional graphs unavailable";
      }
      ASSERT_EQ(initialized, conditional_hardware_fixture_t::initialize_result_e::ready);
      for (const auto work : {work_flag_e::none, work_flag_e::optional_ocr, work_flag_e::subtitle_observation}) {
        for (const auto branch : {branch_e::reuse, branch_e::infer}) {
          const auto request = make_request(token, work);
          const bool depth_ran = branch == branch_e::infer;
          const bool optional_ran = optional && depth_ran && work == work_flag_e::optional_ocr;
          const auto proposal = make_proposal(branch, token);
          if (optional) {
            ASSERT_TRUE(fixture.run_optional(proposal, request, depth_ran ? conditional_hardware_fixture_t::infer_marker : 0u, optional_ran ? conditional_hardware_fixture_t::optional_infer_marker : 0u, branch, true, optional_ran));
          } else {
            ASSERT_TRUE(fixture.run(proposal, request, depth_ran ? conditional_hardware_fixture_t::infer_marker : 0u, branch, true));
          }
          EXPECT_TRUE(fixture.staged_receipt_was_uncommitted(depth_ran, optional_ran));
        }
      }
    }
  }
}

TEST(CudaConditionalGraphHardware, InvalidRequestsNeverCommitButKeepInferFallback) {
  constexpr std::uint64_t token = 0x86421357fedcba09ull;
  conditional_hardware_fixture_t fixture;
  const auto initialized = fixture.initialize(false, false, true, false, true);
  if (initialized == conditional_hardware_fixture_t::initialize_result_e::unavailable) {
    GTEST_SKIP() << "CUDA conditional graphs unavailable";
  }
  ASSERT_EQ(initialized, conditional_hardware_fixture_t::initialize_result_e::ready);
  const auto valid = make_request(token, work_flag_e::optional_ocr);
  std::array<request_record_t, 8u> bad;
  bad.fill(valid);
  bad[0] = make_request(0u, work_flag_e::optional_ocr);
  bad[1].token_low_cookie ^= 1u;
  bad[2].token_high_cookie ^= 1u;
  bad[3].magic = 0u;
  bad[4].work_flags = 3u;
  bad[4].work_flags_cookie = 3u ^ work_flags_cookie;
  bad[5].work_flags = 4u;
  bad[5].work_flags_cookie = 4u ^ work_flags_cookie;
  bad[6].work_flags_cookie ^= 1u;
  bad[7].reserved = 1u;
  for (const auto &request : bad) {
    EXPECT_FALSE(authenticated_request(request));
    ASSERT_TRUE(fixture.run_optional(make_proposal(branch_e::reuse, token), request, conditional_hardware_fixture_t::infer_marker, 0u, branch_e::infer, false, false));
    EXPECT_TRUE(fixture.staged_receipt_was_uncommitted(true, false));
  }
  // A bad PROP under good RQST still commits infer/noOOCR: preserve the established fallback.
  auto bad_proposal = make_proposal(branch_e::reuse, token);
  bad_proposal.magic = 0u;
  ASSERT_TRUE(fixture.run_optional(bad_proposal, valid, conditional_hardware_fixture_t::infer_marker, 0u, branch_e::infer, true, false));
  EXPECT_TRUE(fixture.staged_receipt_was_uncommitted(true, false));
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

  for (const auto work : {8u, 16u}) {
    const auto retired_request = make_request(token, static_cast<work_flag_e>(work));
    for (const auto branch : {branch_e::infer, branch_e::reuse}) {
      EXPECT_TRUE(fixture.run_optional(
        make_proposal(branch, token), retired_request,
        conditional_hardware_fixture_t::infer_marker, 0u,
        branch_e::infer, false, false
      ));
    }
  }
  const auto abstention_request = make_request(token, work_flag_e::subtitle_observation);
  for (const auto branch : {branch_e::infer, branch_e::reuse}) {
    EXPECT_TRUE(fixture.run_optional(
      make_proposal(branch, token), abstention_request,
      branch == branch_e::infer ? conditional_hardware_fixture_t::infer_marker : 0u,
      0u, branch, true, false
    ));
  }

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


TEST(CudaConditionalGraphHardware, ParentBranchTimingCoversInferReuseAndSuppressedOptionalWork) {
  constexpr std::uint64_t token = 0x3132333435363738ull;
  conditional_hardware_fixture_t fixture;
  const auto initialized = fixture.initialize(false, false, true, true);
  if (initialized == conditional_hardware_fixture_t::initialize_result_e::unavailable) {
    GTEST_SKIP() << "Conditional graph branch timing is unavailable on this driver/device";
  }
  ASSERT_EQ(initialized, conditional_hardware_fixture_t::initialize_result_e::ready)
    << "failure=" << static_cast<int>(fixture.bridge_failure())
    << " cuda=" << static_cast<int>(fixture.bridge_cuda_result());
  for (const auto work : {work_flag_e::optional_ocr, work_flag_e::subtitle_observation,
                         work_flag_e::none}) {
    for (const auto branch : {branch_e::infer, branch_e::reuse}) {
      const bool optional_executes = optional_ocr_executes(work_flags_value(work), branch);
      const auto request = make_request(token, work);
      ASSERT_TRUE(fixture.run_optional(
        make_proposal(branch, token), request,
        branch == branch_e::infer ? conditional_hardware_fixture_t::infer_marker : 0u,
        optional_executes ? conditional_hardware_fixture_t::optional_infer_marker : 0u,
        branch, true, optional_executes
      ));
      // The fixture synchronized this exact current root before asserting its marker/receipt.
      branch_completion_timing_t timing;
      EXPECT_FALSE(fixture.branch_timing(false, timing));
      ASSERT_TRUE(fixture.branch_timing(true, timing));
      EXPECT_TRUE(timing.has_optional);
      EXPECT_TRUE(std::isfinite(timing.depth_ms));
      EXPECT_TRUE(std::isfinite(timing.optional_ms));
      EXPECT_GE(timing.depth_ms, 0.0f);
      EXPECT_GE(timing.optional_ms, 0.0f);
    }
  }
  EXPECT_TRUE(fixture.reset_restores_null_context());
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

TEST(DirectGlyphHardware, HoldsClearsOwnerChangesAndWorkerRetirement) {
  namespace glyph = models::host_sbs_direct_glyph_gpu;
  using Microsoft::WRL::ComPtr;
  const char *path = std::getenv("SUNSHINE_TEST_DIRECT_GLYPH_ENGINE");
  if (!path || !*path) GTEST_SKIP() << "Set SUNSHINE_TEST_DIRECT_GLYPH_ENGINE to the native batch engine";
  std::ifstream input(path, std::ios::binary);
  ASSERT_TRUE(input) << path;
  const std::vector<char> bytes {std::istreambuf_iterator<char>(input), {}};
  ComPtr<ID3D11Device> device;
  ComPtr<ID3D11DeviceContext> context;
  ASSERT_HRESULT_SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
                                             D3D11_SDK_VERSION, &device, nullptr, &context));
  ComPtr<IDXGIDevice> dx;
  ComPtr<IDXGIAdapter> adapter;
  ASSERT_HRESULT_SUCCEEDED(device.As(&dx));
  ASSERT_HRESULT_SUCCEEDED(dx->GetAdapter(&adapter));
  auto &cuda = cuda_driver_api::get();
  ASSERT_TRUE(cuda.is_valid() && cuda.cuD3D11GetDevice);
  ASSERT_EQ(cuda.cuInit(0), CUDA_SUCCESS);
  CUdevice ordinal {};
  CUcontext primary = nullptr;
  ASSERT_EQ(cuda.cuD3D11GetDevice(&ordinal, adapter.Get()), CUDA_SUCCESS);
  ASSERT_EQ(cuda.cuDevicePrimaryCtxRetain(&primary, ordinal), CUDA_SUCCESS);
  const auto release_primary = util::fail_guard([&] { cuda.cuDevicePrimaryCtxRelease(ordinal); });
  glyph::detail::current_cuda_context_t current(cuda, primary);
  topology_test_logger_t logger;
  auto *runtime = nvinfer1::createInferRuntime(logger);
  ASSERT_NE(runtime, nullptr);
  auto *engine = runtime->deserializeCudaEngine(bytes.data(), bytes.size());
  ASSERT_NE(engine, nullptr);
  auto *execution = engine->createExecutionContext();
  ASSERT_NE(execution, nullptr);
  glyph::resources_t score;
  const auto cleanup = util::fail_guard([&] {
    if (score.release()) {
      release_trt_test_interface(execution);
      release_trt_test_interface(engine);
      release_trt_test_interface(runtime);
    }
  });
  std::string error;
  ASSERT_TRUE(score.prepare(device.Get(), context.Get(), SUNSHINE_SHADERS_DIR, primary, execution, error)) << error;

  constexpr UINT w = 64, h = 32;
  const std::array<std::uint16_t, w * h * 4> rgb {};
  D3D11_TEXTURE2D_DESC texture {w, h, 1, 1, DXGI_FORMAT_R16G16B16A16_FLOAT, {1, 0}, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE};
  D3D11_SUBRESOURCE_DATA initial {rgb.data(), w * 8, 0};
  ComPtr<ID3D11Texture2D> source;
  ComPtr<ID3D11ShaderResourceView> source_view, cap_view;
  ASSERT_HRESULT_SUCCEEDED(device->CreateTexture2D(&texture, &initial, &source));
  ASSERT_HRESULT_SUCCEEDED(device->CreateShaderResourceView(source.Get(), nullptr, &source_view));
  D3D11_BUFFER_DESC raw {glyph::work_bytes, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
                         0, D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS};
  ComPtr<ID3D11Buffer> work, caps, staging;
  ASSERT_HRESULT_SUCCEEDED(device->CreateBuffer(&raw, nullptr, &work));
  ASSERT_HRESULT_SUCCEEDED(device->CreateBuffer(&raw, nullptr, &caps));
  D3D11_SHADER_RESOURCE_VIEW_DESC view {};
  view.Format = DXGI_FORMAT_R32_TYPELESS;
  view.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
  view.BufferEx.NumElements = glyph::work_bytes / sizeof(UINT);
  view.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
  ASSERT_HRESULT_SUCCEEDED(device->CreateShaderResourceView(caps.Get(), &view, &cap_view));
  glyph::request_t job {.source_rgb = source_view.Get(), .work = work.Get(), .width = w, .band_height = h,
                        .detector_caps = cap_view.Get(), .generation = 1};
  const auto submit = [&](UINT state, UINT64 owner) {
    job.frame = ++job.token;
    job.subtitle_context_generation = owner;
    std::array<UINT, glyph::work_bytes / sizeof(UINT)> packet {state};
    if (state == 2) packet = {2, 1, 0, 0, 0, 0, w, h};
    context->UpdateSubresource(work.Get(), 0, nullptr, packet.data(), 0, 0);
    packet = {1, 1, UINT(job.token), 0, 0, 0, w, h};
    context->UpdateSubresource(caps.Get(), 0, nullptr, packet.data(), 0, 0);
    return score.try_submit(job, error);
  };
  const auto poll = [&](const glyph::request_t *request) {
    glyph::result_t result;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    glyph::poll_status_e status;
    do {
      status = score.poll(request, result, error);
      if (status != glyph::poll_status_e::pending) break;
      std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < deadline);
    return status;
  };
  D3D11_BUFFER_DESC readback {16 * sizeof(UINT), D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ};
  ASSERT_HRESULT_SUCCEEDED(device->CreateBuffer(&readback, nullptr, &staging));
  // state.x is the matched previous slot + 1. A held observation keeps its last line record.
  const auto check_line = [&](UINT state, UINT64 owner, UINT expected_previous) {
    ASSERT_EQ(submit(state, owner), glyph::submit_status_e::accepted) << error;
    ASSERT_EQ(poll(&job), glyph::poll_status_e::ready) << error;
    score.retire();
    const D3D11_BOX first_line {0, 0, 0, readback.ByteWidth, 1, 1};
    context->CopySubresourceRegion(staging.Get(), 0, 0, 0, 0, score.native_lines_for_evaluation(), 0, &first_line);
    D3D11_MAPPED_SUBRESOURCE map {};
    ASSERT_HRESULT_SUCCEEDED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &map));
    EXPECT_EQ(static_cast<const UINT *>(map.pData)[12], expected_previous) << job.frame;
    context->Unmap(staging.Get(), 0);
  };
  check_line(2, 1, 0);
  check_line(2, 1, 1);
  check_line(3, 1, 1);
  check_line(3, 1, 1);
  check_line(2, 1, 1);
  check_line(2, 2, 0);
  check_line(1, 2, 0);
  check_line(2, 2, 0);
  check_line(3, 3, 0);
  check_line(2, 3, 0);

  ASSERT_EQ(submit(2, 3), glyph::submit_status_e::accepted) << error;
  ComPtr<ID3D11Multithread> immediate_lock;
  ASSERT_HRESULT_SUCCEEDED(context.As(&immediate_lock));
  immediate_lock->SetMultithreadProtected(TRUE);
  immediate_lock->Enter();
  auto discarded = std::async(std::launch::async, [&] { return poll(nullptr); });
  const auto ready = discarded.wait_for(std::chrono::seconds(2));
  immediate_lock->Leave();
  EXPECT_EQ(ready, std::future_status::ready) << "Worker needed the render thread's immediate context";
  EXPECT_EQ(discarded.get(), glyph::poll_status_e::discarded) << error;
  EXPECT_TRUE(score.release());
}

// One observation's whole lines share an encoder batch, and its scoring fence, for as long as
// their tiles fit it. Each line matches scoring it alone; a line wider than a batch keeps one
// fence per chunk.
TEST(DirectGlyphHardware, ScoresTheLinesThatFitOneBatchUnderOneFence) {
  namespace glyph = models::host_sbs_direct_glyph_gpu;
  using Microsoft::WRL::ComPtr;
  const auto *enabled = std::getenv("APOLLO_RUN_TENSORRT_TESTS");
  if (!enabled || std::string_view {enabled} != "1") {
    GTEST_SKIP() << "Set APOLLO_RUN_TENSORRT_TESTS=1 for the local NVIDIA/TensorRT integration check.";
  }
  ComPtr<ID3D11Device> device;
  ComPtr<ID3D11DeviceContext> context;
  ASSERT_HRESULT_SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
                                             D3D11_SDK_VERSION, &device, nullptr, &context));
  ComPtr<IDXGIDevice> dx;
  ComPtr<IDXGIAdapter> adapter;
  ASSERT_HRESULT_SUCCEEDED(device.As(&dx));
  ASSERT_HRESULT_SUCCEEDED(dx->GetAdapter(&adapter));
  auto &cuda = cuda_driver_api::get();
  ASSERT_TRUE(cuda.is_valid() && cuda.cuD3D11GetDevice);
  ASSERT_EQ(cuda.cuInit(0), CUDA_SUCCESS);
  CUdevice ordinal {};
  CUcontext primary = nullptr;
  ASSERT_EQ(cuda.cuD3D11GetDevice(&ordinal, adapter.Get()), CUDA_SUCCESS);
  ASSERT_EQ(cuda.cuDevicePrimaryCtxRetain(&primary, ordinal), CUDA_SUCCESS);
  const auto release_primary = util::fail_guard([&] { cuda.cuDevicePrimaryCtxRelease(ordinal); });
  glyph::detail::current_cuda_context_t current(cuda, primary);
  topology_test_logger_t logger;
  auto *runtime = nvinfer1::createInferRuntime(logger);
  ASSERT_NE(runtime, nullptr);
  nvinfer1::IHostMemory *serialized = nullptr;
  {
    // Build the packaged encoder with the production batch profile of one to six tiles.
    const auto onnx = std::filesystem::path {SUNSHINE_TEST_BIN_DIR}.parent_path() / "assets" /
                      std::filesystem::path {models::direct_glyph_model_asset_path};
    ASSERT_EQ(models::file_sha256_hex(onnx), models::direct_glyph_onnx_sha256) << onnx;
    auto *builder = nvinfer1::createInferBuilder(logger);
    ASSERT_NE(builder, nullptr);
#if NV_TENSORRT_MAJOR < 11
    constexpr auto network_flags = 1u << static_cast<std::uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
#else
    constexpr auto network_flags = 0u;
#endif
    auto *network = builder->createNetworkV2(network_flags);
    auto *config = builder->createBuilderConfig();
    auto *parser = network ? nvonnxparser::createParser(*network, logger) : nullptr;
    const auto release_builder = util::fail_guard([&] {
      release_trt_test_interface(parser);
      release_trt_test_interface(config);
      release_trt_test_interface(network);
      release_trt_test_interface(builder);
    });
    ASSERT_TRUE(network && config && parser);
    config->clearFlag(nvinfer1::BuilderFlag::kTF32);
    ASSERT_TRUE(parser->parseFromFile(onnx.string().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)));
    const auto shape = [](int tiles) {
      return nvinfer1::Dims4 {tiles, 3, int(glyph::palette_tile.height), int(glyph::palette_tile.width)};
    };
    auto *profile = builder->createOptimizationProfile();
    using point = nvinfer1::OptProfileSelector;
    ASSERT_TRUE(profile && profile->setDimensions("rgb", point::kMIN, shape(1)) && profile->setDimensions("rgb", point::kOPT, shape(4)) &&
                profile->setDimensions("rgb", point::kMAX, shape(int(glyph::tile_capacity))) && config->addOptimizationProfile(profile) >= 0);
    serialized = builder->buildSerializedNetwork(*network, *config);
    ASSERT_NE(serialized, nullptr);
  }
  auto *engine = runtime->deserializeCudaEngine(serialized->data(), serialized->size());
  release_trt_test_interface(serialized);
  ASSERT_NE(engine, nullptr);
  auto *execution = engine->createExecutionContext();
  ASSERT_NE(execution, nullptr);
  std::string error;
  glyph::resources_t transport;
  const auto cleanup = util::fail_guard([&] {
    if (transport.release()) {
      release_trt_test_interface(execution);
      release_trt_test_interface(engine);
      release_trt_test_interface(runtime);
    }
  });
  ASSERT_TRUE(transport.prepare(device.Get(), context.Get(), SUNSHINE_SHADERS_DIR, primary, execution, error)) << error;
  // Live preparation runs on the background build worker, which must leave CUDA's D3D11
  // registration to the thread that submits (and draws).
  EXPECT_FALSE(transport.native_interop_registered_for_tests());

  // Light strokes with dark outlines on a textured backdrop, in four caption rows.
  constexpr UINT width = 3840, height = 212, scale = 4, advance = 6 * scale;
  constexpr std::array<std::array<std::uint8_t, 7>, 4> font {{
    {{14, 17, 17, 31, 17, 17, 17}},
    {{30, 17, 17, 30, 17, 17, 30}},
    {{14, 17, 16, 16, 16, 17, 14}},
    {{31, 16, 16, 30, 16, 16, 31}},
  }};
  struct line_t {
    UINT left, top, characters;
  };
  // Each short line has one encoder tile; the long lines have four and three, and the last one
  // has more than a batch holds.
  constexpr std::array<line_t, 11> lines {{
    {40, 10, 10},
    {40, 62, 10},
    {40, 114, 10},
    {400, 10, 80},
    {400, 62, 56},
    {400, 114, 10},
    {760, 114, 10},
    {1120, 114, 10},
    {1480, 114, 10},
    {1840, 114, 10},
    {40, 166, 150},
  }};
  const auto half = [](float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    return std::uint16_t((((bits >> 23) & 0xffu) - 112u) << 10 | ((bits >> 13) & 0x3ffu));
  };
  std::vector<std::array<float, 3>> rgb(std::size_t(width) * height);
  for (UINT y = 0; y < height; ++y) {
    for (UINT x = 0; x < width; ++x) {
      const float shade = 0.22f + 0.12f * float((x / 16 + y / 8) % 2) + 0.1f * float(x) / width;
      rgb[std::size_t(y) * width + x] = {shade, shade * 0.9f + 0.05f, shade * 0.8f + 0.1f};
    }
  }
  for (const auto &line : lines) {
    for (const UINT radius : {2u, 0u}) {
      for (UINT c = 0; c < line.characters; ++c) {
        const auto &rows = font[(c * 7 + c / 3) % font.size()];
        for (UINT row = 0; row < 7; ++row) {
          for (UINT column = 0; column < 5; ++column) {
            if (!(rows[row] & (16u >> column))) continue;
            const UINT x0 = line.left + c * advance + column * scale, y0 = line.top + row * scale;
            for (UINT y = y0 - radius; y < y0 + scale + radius; ++y) {
              for (UINT x = x0 - radius; x < x0 + scale + radius; ++x) {
                rgb[std::size_t(y) * width + x] = radius ? std::array {0.03f, 0.03f, 0.03f} : std::array {0.96f, 0.94f, 0.9f};
              }
            }
          }
        }
      }
    }
  }
  std::vector<std::uint16_t> pixels(rgb.size() * 4);
  for (std::size_t i = 0; i < rgb.size(); ++i) {
    pixels[4 * i] = half(rgb[i][0]);
    pixels[4 * i + 1] = half(rgb[i][1]);
    pixels[4 * i + 2] = half(rgb[i][2]);
    pixels[4 * i + 3] = half(1.0f);
  }
  D3D11_TEXTURE2D_DESC texture {width, height, 1, 1, DXGI_FORMAT_R16G16B16A16_FLOAT, {1, 0}, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE};
  const D3D11_SUBRESOURCE_DATA initial {pixels.data(), width * 8, 0};
  ComPtr<ID3D11Texture2D> source;
  ComPtr<ID3D11ShaderResourceView> source_view, cap_view;
  ASSERT_HRESULT_SUCCEEDED(device->CreateTexture2D(&texture, &initial, &source));
  ASSERT_HRESULT_SUCCEEDED(device->CreateShaderResourceView(source.Get(), nullptr, &source_view));
  D3D11_BUFFER_DESC raw {glyph::work_bytes, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
                         0, D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS};
  ComPtr<ID3D11Buffer> work, caps;
  ASSERT_HRESULT_SUCCEEDED(device->CreateBuffer(&raw, nullptr, &work));
  ASSERT_HRESULT_SUCCEEDED(device->CreateBuffer(&raw, nullptr, &caps));
  D3D11_SHADER_RESOURCE_VIEW_DESC view {};
  view.Format = DXGI_FORMAT_R32_TYPELESS;
  view.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
  view.BufferEx.NumElements = glyph::work_bytes / sizeof(UINT);
  view.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
  ASSERT_HRESULT_SUCCEEDED(device->CreateShaderResourceView(caps.Get(), &view, &cap_view));

  const auto read = [&](ID3D11Resource *resource, D3D11_TEXTURE2D_DESC desc, std::vector<UINT> &words) {
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> staging;
    ASSERT_HRESULT_SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &staging));
    context->CopyResource(staging.Get(), resource);
    D3D11_MAPPED_SUBRESOURCE map {};
    ASSERT_HRESULT_SUCCEEDED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &map));
    words.resize(std::size_t(desc.Width) * desc.Height);
    for (UINT y = 0; y < desc.Height; ++y) {
      std::memcpy(words.data() + std::size_t(y) * desc.Width, static_cast<const char *>(map.pData) + std::size_t(y) * map.RowPitch, desc.Width * sizeof(UINT));
    }
    context->Unmap(staging.Get(), 0);
  };
  struct scored_t {
    unsigned scoring_fences = 0;
    unsigned polls = 0;
    double milliseconds = 0.0;
    std::vector<UINT> alpha;
  };
  glyph::request_t job {.source_rgb = source_view.Get(), .work = work.Get(), .width = width, .band_height = height,
                        .detector_caps = cap_view.Get(), .generation = 1, .reset_history = true};
  const auto score = [&](std::initializer_list<line_t> planned, scored_t &scored) {
    job.frame = ++job.token;
    std::array<UINT, glyph::work_bytes / sizeof(UINT)> plan {2, UINT(planned.size())};
    std::array<UINT, glyph::work_bytes / sizeof(UINT)> covers {1, UINT(planned.size()), UINT(job.token), UINT(job.token >> 32)};
    UINT slot = 0;
    for (const auto &line : planned) {
      const UINT right = line.left - 2 + line.characters * advance, bottom = line.top + 7 * scale + 2;
      const std::array box {line.left - 2, line.top - 2, right, bottom};
      const std::array cover {line.left - 8, line.top - 8, right + 8, bottom + 6};
      std::copy(box.begin(), box.end(), plan.begin() + 4 + 4 * slot);
      std::copy(cover.begin(), cover.end(), covers.begin() + 4 + 4 * slot);
      ++slot;
    }
    context->UpdateSubresource(work.Get(), 0, nullptr, plan.data(), 0, 0);
    context->UpdateSubresource(caps.Get(), 0, nullptr, covers.data(), 0, 0);
    const auto submitted = transport.native_fences_for_tests();
    ASSERT_EQ(transport.try_submit(job, error), glyph::submit_status_e::accepted) << error;
    EXPECT_TRUE(transport.native_interop_registered_for_tests());
    glyph::result_t result;
    glyph::poll_status_e status;
    const auto started = std::chrono::steady_clock::now();
    const auto deadline = started + std::chrono::seconds(5);
    // A deadline already past ends the wait at once and leaves the score as it is.
    transport.wait_until(started);
    EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::milliseconds(50));
    scored.polls = 0;
    do {
      status = transport.poll(&job, result, error);
      ++scored.polls;
      if (status != glyph::poll_status_e::pending) break;
      // As a live completion does within its budget: wait for the step, then take the next.
      transport.wait_until(deadline);
    } while (std::chrono::steady_clock::now() < deadline);
    scored.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    ASSERT_EQ(status, glyph::poll_status_e::ready) << error;
    // The first fence covers planning.
    scored.scoring_fences = unsigned(transport.native_fences_for_tests() - submitted - 1u);
    // Each wait ends on a finished step, so the polls are counted by the steps and not by time:
    // one that finds the plan pending, one for each fence, and one spare for a late readback.
    EXPECT_LE(scored.polls, scored.scoring_fences + 3u);
    ComPtr<ID3D11Resource> alpha;
    result.alpha8->GetResource(&alpha);
    ASSERT_NO_FATAL_FAILURE(read(alpha.Get(), {width, height, 1, 1, DXGI_FORMAT_R32_UINT, {1, 0}}, scored.alpha));
    D3D11_BUFFER_DESC staged {glyph::work_bytes, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ};
    ComPtr<ID3D11Buffer> staging;
    ASSERT_HRESULT_SUCCEEDED(device->CreateBuffer(&staged, nullptr, &staging));
    context->CopyResource(staging.Get(), work.Get());
    D3D11_MAPPED_SUBRESOURCE map {};
    ASSERT_HRESULT_SUCCEEDED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &map));
    const auto *words = static_cast<const UINT *>(map.pData);
    EXPECT_EQ(words[1], planned.size());
    EXPECT_EQ(words[3], 1u) << "Every planned line must publish a valid score";
    context->Unmap(staging.Get(), 0);
    transport.retire();
  };
  const auto expect_ink = [&](const scored_t &scored, const line_t &line) {
    UINT ink = 0;
    for (UINT y = line.top; y < line.top + 7 * scale; ++y) {
      for (UINT x = line.left; x < line.left + line.characters * advance; ++x) {
        ink += scored.alpha[std::size_t(y) * width + x] >= 38u;
      }
    }
    EXPECT_GT(ink, 0u) << line.left << "," << line.top;
  };
  const auto expect_lines = [&](std::initializer_list<line_t> planned, unsigned scoring_fences) {
    std::vector<UINT> expected(std::size_t(width) * height);
    for (const auto &line : planned) {
      SCOPED_TRACE(line.characters);
      scored_t alone;
      ASSERT_NO_FATAL_FAILURE(score({line}, alone));
      // Only the line wider than a batch needs a second one.
      EXPECT_EQ(alone.scoring_fences, line.characters == lines[10].characters ? 2u : 1u);
      expect_ink(alone, line);
      std::transform(expected.begin(), expected.end(), alone.alpha.begin(), expected.begin(), [](UINT a, UINT b) {
        return std::max(a, b);
      });
    }
    scored_t together;
    ASSERT_NO_FATAL_FAILURE(score(planned, together));
    EXPECT_EQ(together.scoring_fences, scoring_fences);
    std::cout << "[direct glyph] " << planned.size() << " line(s), " << together.scoring_fences << " scoring fence(s): "
              << together.milliseconds << " ms in " << together.polls << " poll(s)" << std::endl;
    const auto mismatched = std::inner_product(expected.begin(), expected.end(), together.alpha.begin(), std::size_t {0},
                                               std::plus<> {}, std::not_equal_to<> {});
    EXPECT_EQ(mismatched, 0u);
  };
  // Three one-tile lines fit one batch: planning plus one scoring fence.
  ASSERT_NO_FATAL_FAILURE(expect_lines({lines[0], lines[1], lines[2]}, 1u));
  // Seven tiles exceed the six-tile batch, so each line keeps its own batch and fence.
  ASSERT_NO_FATAL_FAILURE(expect_lines({lines[3], lines[4]}, 2u));
  // A screen of eight short lines, as a menu has: six share the first batch and two the second.
  ASSERT_NO_FATAL_FAILURE(expect_lines({lines[0], lines[1], lines[2], lines[5], lines[6], lines[7], lines[8], lines[9]}, 2u));
  // Short lines around long ones: one and four tiles share a batch, then three, one and one.
  ASSERT_NO_FATAL_FAILURE(expect_lines({lines[0], lines[3], lines[4], lines[1], lines[5]}, 2u));
  // A line wider than a batch is scored in its own two chunks between the batches of the others.
  ASSERT_NO_FATAL_FAILURE(expect_lines({lines[0], lines[1], lines[10], lines[4]}, 4u));

  // A line whose rectangle and pixels are those of the last observation is carried over: a still
  // screen of text costs the plan and the finish, and no score.
  {
    const std::initializer_list<line_t> menu {lines[0], lines[1], lines[2], lines[5], lines[6], lines[7], lines[8], lines[9]};
    scored_t fresh, again, edited;
    ASSERT_NO_FATAL_FAILURE(score(menu, fresh));
    EXPECT_EQ(transport.planned_lines().scored, 8u);
    EXPECT_EQ(transport.planned_lines().carried, 0u) << "an observation without history scores every line";
    job.reset_history = false;
    ASSERT_NO_FATAL_FAILURE(score(menu, again));
    EXPECT_EQ(transport.planned_lines().scored, 0u);
    EXPECT_EQ(transport.planned_lines().carried, 8u);
    EXPECT_EQ(again.alpha, fresh.alpha) << "a carried line publishes what its score would";
    std::cout << "[direct glyph] 8 line(s) unchanged: " << again.milliseconds << " ms in " << again.polls << " poll(s), against "
              << fresh.milliseconds << " ms scored" << std::endl;

    // One texel of one line changes: that line is scored again and the others are still carried.
    const std::array<std::uint16_t, 4u> dot {half(0.5f), half(0.1f), half(0.1f), half(1.0f)};
    const D3D11_BOX changed {lines[1].left + 4u, lines[1].top + 4u, 0u, lines[1].left + 5u, lines[1].top + 5u, 1u};
    context->UpdateSubresource(source.Get(), 0u, &changed, dot.data(), 8u, 0u);
    ASSERT_NO_FATAL_FAILURE(score(menu, edited));
    EXPECT_EQ(transport.planned_lines().scored, 1u);
    EXPECT_EQ(transport.planned_lines().carried, 7u);
    for (const auto &line : menu) {
      if (line.left == lines[1].left && line.top == lines[1].top) continue;
      for (UINT y = line.top - 8u; y < line.top + 7u * scale + 8u; ++y) {
        for (UINT x = line.left - 8u; x < line.left - 2u + line.characters * advance + 8u; ++x) {
          ASSERT_EQ(edited.alpha[std::size_t(y) * width + x], fresh.alpha[std::size_t(y) * width + x]) << x << ',' << y;
        }
      }
    }
    expect_ink(edited, lines[1]);
    job.reset_history = true;
  }
  EXPECT_TRUE(transport.release());
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
  EXPECT_FALSE(authenticated_optional_ocr_receipt(receipt, no_optional_request));
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
