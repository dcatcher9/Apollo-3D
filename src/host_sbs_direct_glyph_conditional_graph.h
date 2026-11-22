#pragma once

#include "cuda_driver_api.h"
#include "host_sbs_direct_glyph_tiles.h"
#include <NvInferRuntime.h>

#include <array>
#include <stdexcept>
#include <string>

namespace models::host_sbs_direct_glyph_gpu::detail {
  // Capture retains fixed private tensor addresses. A different mapping invalidates the graph.
  class conditional_score_graph_t {
    static_assert(tile_capacity == 6u, "The PTX setter binds six conditional handles");

    inline static constexpr char gate_ptx[] = R"ptx(
.version 8.4
.target sm_70
.address_size 64
.extern .func cudaGraphSetConditional(.param .b64 handle, .param .b32 value);

.visible .entry set_glyph_condition(
    .param .u64 cond0, .param .u64 cond1, .param .u64 cond2, .param .u64 cond3,
    .param .u64 cond4, .param .u64 cond5,
    .param .u64 placement) {
    .reg .pred %p<10>;
    .reg .b32 %r<7>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [placement];
    mov.u32 %r1, %tid.x;
    setp.eq.u32 %p1, %r1, 0;
    setp.eq.u32 %p2, %r1, 1;
    setp.eq.u32 %p3, %r1, 2;
    setp.eq.u32 %p4, %r1, 3;
    setp.eq.u32 %p5, %r1, 4;
    setp.eq.u32 %p6, %r1, 5;
    @%p1 ld.param.u64 %rd2, [cond0];
    @%p2 ld.param.u64 %rd2, [cond1];
    @%p3 ld.param.u64 %rd2, [cond2];
    @%p4 ld.param.u64 %rd2, [cond3];
    @%p5 ld.param.u64 %rd2, [cond4];
    @%p6 ld.param.u64 %rd2, [cond5];
    ld.global.u32 %r2, [%rd1];
    ld.global.u32 %r3, [%rd1+4];
    ld.global.u32 %r4, [%rd1+8];
    ld.global.u32 %r5, [%rd1+12];
    or.b32 %r3, %r3, %r4;
    or.b32 %r3, %r3, %r5;
    setp.eq.u32 %p7, %r3, 0;
    setp.le.u32 %p8, %r2, 6;
    setp.gt.u32 %p9, %r2, %r1;
    and.pred %p7, %p7, %p8;
    and.pred %p7, %p7, %p9;
    selp.u32 %r6, 1, 0, %p7;
    {
        .param .b64 h;
        .param .b32 v;
        st.param.b64 [h], %rd2;
        st.param.b32 [v], %r6;
        call.uni cudaGraphSetConditional, (h, v);
    }
    ret;
}

.visible .entry attest_glyph_child(.param .u64 placement, .param .u32 completed_bit) {
    .reg .b32 %r<3>;
    .reg .b64 %rd;
    ld.param.u64 %rd, [placement];
    ld.param.u32 %r1, [completed_bit];
    ld.global.u32 %r2, [%rd+4];
    or.b32 %r2, %r2, %r1;
    st.global.u32 [%rd+4], %r2;
    ret;
}
)ptx";

    cuda_driver_api &cuda_;
    CUmodule module_ = nullptr;
    CUgraph child_ = nullptr, parent_ = nullptr;
    CUgraphExec execution_ = nullptr;
    std::array<CUdeviceptr, 3> addresses_{};

    static void prove(CUresult status, const char *reason) {
      if (status != CUDA_SUCCESS) throw std::runtime_error(std::string(reason) + " CUDA " + std::to_string(status));
    }

    CUgraphNode add_kernel(CUgraph graph, CUfunction kernel, void **args, CUgraphNode prerequisite = nullptr,
                          unsigned int threads = 1u) {
      CUDA_KERNEL_NODE_PARAMS params{};
      params.func = kernel;
      params.gridDimX = params.gridDimY = params.gridDimZ = 1u;
      params.blockDimX = threads;
      params.blockDimY = params.blockDimZ = 1u;
      params.kernelParams = args;
      CUgraphNode result = nullptr;
      prove(cuda_.cuGraphAddKernelNode(&result, graph, prerequisite ? &prerequisite : nullptr, prerequisite ? 1u : 0u, &params),
            "Add exact model IF bridge kernel");
      return result;
    }

    CUgraphNode add_tile_copy(CUgraph body, CUcontext primary, CUdeviceptr source, CUdeviceptr destination,
                              std::size_t bytes, CUgraphNode prerequisite) {
      CUDA_MEMCPY3D copy{};
      copy.srcMemoryType = copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
      copy.srcDevice = source;
      copy.dstDevice = destination;
      copy.WidthInBytes = bytes;
      copy.Height = copy.Depth = 1u;
      CUgraphNode result = nullptr;
      prove(cuda_.cuGraphAddMemcpyNode(&result, body, prerequisite ? &prerequisite : nullptr,
                                      prerequisite ? 1u : 0u, &copy, primary), "Add direct glyph tile atlas copy");
      return result;
    }

    CUgraph add_branch(CUcontext primary, CUgraphConditionalHandle handle, CUgraphNode &previous) {
      CUgraphNodeParams branch{};
      branch.type = CU_GRAPH_NODE_TYPE_CONDITIONAL;
      branch.params.conditional.handle = handle;
      branch.params.conditional.type = CU_GRAPH_COND_TYPE_IF;
      branch.params.conditional.size = 1u;
      branch.params.conditional.ctx = primary;
      CUgraphNode node = nullptr;
      prove(cuda_.graph_add_node(&node, parent_, &previous, 1u, &branch), "Add ordered direct glyph tile IF");
      previous = node;
      const auto bodies = branch.params.conditional.phGraph_out;
      if (!bodies || !bodies[0]) throw std::runtime_error("No CUDA condition body");
      return bodies[0];
    }

    void add_tile(CUgraph body, CUcontext primary, CUfunction attest, unsigned int tile,
                  tile_shape_t shape) {
      const std::size_t rgb_bytes = shape.input_words() * sizeof(float);
      const std::size_t logits_bytes = shape.output_words() * sizeof(float);
      CUgraphNode before_model = nullptr;
      if (tile != 0u) {
        // Preserve the preceding tile before the fixed TensorRT binding is overwritten.
        before_model = add_tile_copy(body, primary, addresses_[1], addresses_[1] + tile * logits_bytes,
                                     logits_bytes, nullptr);
        before_model = add_tile_copy(body, primary, addresses_[0] + tile * rgb_bytes, addresses_[0],
                                     rgb_bytes, before_model);
      }

      CUgraphNode completed = nullptr;
      prove(cuda_.cuGraphAddChildGraphNode(&completed, body, before_model ? &before_model : nullptr,
                                          before_model ? 1u : 0u, child_), "Insert captured TensorRT score child");
      unsigned int completed_bit = 1u << tile;
      void *proof_args[]{&addresses_[2], &completed_bit};
      (void)add_kernel(body, attest, proof_args, completed);
    }

    void setup(CUcontext primary, CUstream stream, nvinfer1::IExecutionContext &inference,
               tile_shape_t shape) {
      if (!cuda_.has_conditional_graph_support() || !cuda_.cuStreamBeginCapture || !cuda_.cuStreamEndCapture ||
          !cuda_.cuGraphAddMemcpyNode)
        throw std::runtime_error("GPU-conditioned score graph is unavailable");
      if (!inference.setTensorAddress("rgb",reinterpret_cast<void *>(addresses_[0])) ||
          !inference.setTensorAddress("logits",reinterpret_cast<void *>(addresses_[1])) ||
          !inference.enqueueV3(stream)) throw std::runtime_error("Initialize fixed model context before capture");
      // Setup runs on private zero tensors before accepting any observation.
      prove(cuda_.cuStreamSynchronize(stream), "Finish one-time TensorRT warmup");
      prove(cuda_.cuStreamBeginCapture(stream,CU_STREAM_CAPTURE_MODE_RELAXED), "Capture fixed score child");
      const bool ok = inference.enqueueV3(stream);
      prove(cuda_.cuStreamEndCapture(stream,&child_), "Finish fixed score capture");
      std::size_t nodes = 0;
      if (!ok || !child_) throw std::runtime_error("TensorRT score capture was rejected");
      prove(cuda_.cuGraphGetNodes(child_,nullptr,&nodes), "Inspect TensorRT child presence");
      if (nodes == 0) throw std::runtime_error("TensorRT score child is empty");

      prove(cuda_.cuModuleLoadDataEx(&module_,gate_ptx,0u,nullptr,nullptr), "Load conditional score bridge");
      CUfunction set = nullptr, attest = nullptr;
      prove(cuda_.cuModuleGetFunction(&set,module_,"set_glyph_condition"), "Find score IF setter");
      prove(cuda_.cuModuleGetFunction(&attest,module_,"attest_glyph_child"), "Find score IF completion");
      prove(cuda_.cuGraphCreate(&parent_,0u), "Create private conditional root");
      std::array<CUgraphConditionalHandle, tile_capacity> handles{};
      for (auto &handle : handles) {
        prove(cuda_.cuGraphConditionalHandleCreate(&handle, parent_, primary, 0u, CU_GRAPH_COND_ASSIGN_DEFAULT),
              "Default score conditional to OFF");
      }
      // Placement starts with tile count, completed mask, numeric fault and a zero reserved word.
      std::array<void *, tile_capacity + 1u> set_args {};
      for (unsigned int tile = 0; tile < tile_capacity; ++tile) set_args[tile] = &handles[tile];
      set_args.back() = &addresses_[2];
      auto previous = add_kernel(parent_, set, set_args.data(), nullptr, tile_capacity);
      // Every child reuses the same context, activation and fixed tensor bindings.
      for (unsigned int tile = 0; tile < tile_capacity; ++tile) {
        add_tile(add_branch(primary, handles[tile], previous), primary, attest, tile, shape);
      }
      prove(cuda_.cuGraphInstantiateWithFlags(&execution_,parent_,0u), "Instantiate fail-closed TensorRT score graph");
    }

   public:
    explicit conditional_score_graph_t(cuda_driver_api &cuda) : cuda_(cuda) {}

    void prepare(CUcontext primary, CUstream stream, nvinfer1::IExecutionContext &inference,
                 std::array<CUdeviceptr, 3> mapped, tile_shape_t shape) {
      if (execution_) throw std::runtime_error("Score graph already prepared");
      addresses_ = mapped;
      setup(primary,stream,inference,shape);
    }

    // Requires a current context and established retirement of any submissions.
    bool release() noexcept {
      if (execution_ && cuda_.cuGraphExecDestroy(execution_) != CUDA_SUCCESS) return false;
      execution_ = nullptr;
      if (parent_ && cuda_.cuGraphDestroy(parent_) != CUDA_SUCCESS) return false;
      parent_ = nullptr;
      if (child_ && cuda_.cuGraphDestroy(child_) != CUDA_SUCCESS) return false;
      child_ = nullptr;
      if (module_ && cuda_.cuModuleUnload(module_) != CUDA_SUCCESS) return false;
      module_ = nullptr;
      return true;
    }

    void validate_mapping(const std::array<CUdeviceptr, 3> &mapped) const {
      if (!execution_) throw std::runtime_error("Score graph was not warmed at startup");
      if (mapped != addresses_) {
        throw std::runtime_error("Mapped CUDA graph addresses changed");
      }
    }

    void launch(CUstream stream, std::array<CUdeviceptr, 3> mapped) {
      validate_mapping(mapped);
      prove(cuda_.cuGraphLaunch(execution_,stream), "Dispatch device-authenticated score IF graph");
    }
  };
}
