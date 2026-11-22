/**
 * @file src/host_sbs_direct_glyph_gpu.h
 * @brief Single-slot GPU scoring transport for the current direct subtitle glyph plan.
 */
#pragma once

#include "host_sbs_direct_glyph_conditional_graph.h"
#include "host_sbs_direct_glyph_tiles.h"
#include "host_sbs_shader_cache.h"

#include <wrl/client.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace models::host_sbs_direct_glyph_gpu {
  using Microsoft::WRL::ComPtr;
  namespace cache = host_sbs_shader_cache;

  inline constexpr UINT work_bytes = 144u;

  struct request_t {
    ID3D11ShaderResourceView *source_rgb = nullptr;
    ID3D11Buffer *work = nullptr;
    UINT width = 0;
    UINT band_height = 0;
    UINT crop_top = 0;
    std::uint64_t token = 0;
    bool capture_plan = false;
    bool adaptive_context = false;
    tile_profile_e profile = tile_profile_e::common;
    ID3D11ShaderResourceView *detector_caps = nullptr;
    std::uint64_t generation = 0, frame = 0;
    UINT origin_x = 0, origin_y = 0, color_domain = 0;
    bool reset_history = false;

    bool operator==(const request_t &) const = default;
  };

  struct result_t {
    ComPtr<ID3D11ShaderResourceView> alpha8;
    ComPtr<ID3D11ShaderResourceView> completed_placement;
  };

  enum class submit_status_e { accepted, busy, failed };
  enum class poll_status_e { idle, pending, ready, discarded, failed };
  using execution_profiles_t = std::array<nvinfer1::IExecutionContext *, tile_profile_count>;

  namespace detail {
    inline void require(bool condition, const char *reason) {
      if (!condition) throw std::runtime_error(reason);
    }

    inline void check_cuda(CUresult status, const char *reason) {
      if (status != CUDA_SUCCESS) {
        throw std::runtime_error(std::string(reason) + " (CUDA " + std::to_string(status) + ")");
      }
    }

    // The caller may already be using another primary context on this thread.
    class current_cuda_context_t {
     public:
      current_cuda_context_t(cuda_driver_api &cuda, CUcontext primary): cuda_(cuda) {
        check_cuda(cuda_.cuCtxGetCurrent(&previous_), "Read current CUDA context");
        if (previous_ != primary) {
          check_cuda(cuda_.cuCtxSetCurrent(primary), "Select the direct glyph CUDA context");
          changed_ = true;
        }
      }

      ~current_cuda_context_t() {
        if (changed_) cuda_.cuCtxSetCurrent(previous_);
      }

      current_cuda_context_t(const current_cuda_context_t &) = delete;
      current_cuda_context_t &operator=(const current_cuda_context_t &) = delete;

     private:
      cuda_driver_api &cuda_;
      CUcontext previous_ = nullptr;
      bool changed_ = false;
    };

    template<std::size_t Sources = 3, std::size_t Targets = 3>
    class compute_state_t {
     public:
      explicit compute_state_t(ID3D11DeviceContext *context): context_(context) {
        context_->CSGetShaderResources(0u, UINT(srvs_.size()), srvs_.data());
        context_->CSGetUnorderedAccessViews(0u, UINT(uavs_.size()), uavs_.data());
        context_->CSGetConstantBuffers(0u, 1u, constants_.GetAddressOf());
        context_->CSGetShader(shader_.GetAddressOf(), instances_.data(), &instance_count_);
        context_->GetPredication(predicate_.GetAddressOf(), &predicate_value_);
        context_->SetPredication(nullptr, FALSE);
      }

      ~compute_state_t() {
        context_->CSSetShaderResources(0u, UINT(srvs_.size()), srvs_.data());
        context_->CSSetUnorderedAccessViews(0u, UINT(uavs_.size()), uavs_.data(), nullptr);
        context_->CSSetConstantBuffers(0u, 1u, constants_.GetAddressOf());
        context_->CSSetShader(shader_.Get(), instances_.data(), instance_count_);
        context_->SetPredication(predicate_.Get(), predicate_value_);
        for (auto *srv : srvs_) if (srv) srv->Release();
        for (auto *uav : uavs_) if (uav) uav->Release();
        for (UINT i = 0u; i < instance_count_; ++i) if (instances_[i]) instances_[i]->Release();
      }

      compute_state_t(const compute_state_t &) = delete;
      compute_state_t &operator=(const compute_state_t &) = delete;

     private:
      ID3D11DeviceContext *context_;
      std::array<ID3D11ShaderResourceView *, Sources> srvs_ {};
      std::array<ID3D11UnorderedAccessView *, Targets> uavs_ {};
      std::array<ID3D11ClassInstance *, 256> instances_ {};
      ComPtr<ID3D11ComputeShader> shader_;
      ComPtr<ID3D11Buffer> constants_;
      ComPtr<ID3D11Predicate> predicate_;
      UINT instance_count_ = UINT(instances_.size());
      BOOL predicate_value_ = FALSE;
    };

    class transport_t {
      static constexpr UINT cutoff = 38u;
      static constexpr unsigned int event_disable_timing = 2u;

      struct tensor_t {
        ComPtr<ID3D11Buffer> buffer;
        ComPtr<ID3D11ShaderResourceView> srv;
        ComPtr<ID3D11UnorderedAccessView> uav;
      };

      enum stage_e { plan_tile, pack_rgb, resolve_matte, finish_score };
      using srvs_t = std::array<ID3D11ShaderResourceView *, 3>;
      using uavs_t = std::array<ID3D11UnorderedAccessView *, 3>;

      cuda_driver_api &cuda_ = cuda_driver_api::get();
      static_assert(tile_profile_count == 2u);
      std::array<conditional_score_graph_t, tile_profile_count> graphs_ {
        conditional_score_graph_t {cuda_}, conditional_score_graph_t {cuda_}
      };
      CUdevice cuda_device_ {};
      CUcontext primary_ = nullptr;
      CUstream stream_ = nullptr;
      CUevent completed_ = nullptr;
      std::array<CUgraphicsResource, 3> interop_ {};
      ComPtr<ID3D11Device> device_;
      ComPtr<ID3D11DeviceContext> context_;
      std::array<ComPtr<ID3D11ComputeShader>, cache::direct_glyph_transport_specs.size()> shaders_;
      tensor_t input_, output_, placement_;
      ComPtr<ID3D11Buffer> constants_;
      ComPtr<ID3D11Buffer> work_;
      ComPtr<ID3D11UnorderedAccessView> work_uav_;
      ComPtr<ID3D11Buffer> post_plan_work_;
      ComPtr<ID3D11Buffer> tile_dispatch_;
      ComPtr<ID3D11UnorderedAccessView> tile_dispatch_uav_;
      ComPtr<ID3D11ShaderResourceView> alpha8_;
      ComPtr<ID3D11UnorderedAccessView> alpha8_uav_;
      UINT alpha_width_ = 0, alpha_height_ = 0;

      request_t submitted_ {};
      ComPtr<ID3D11Buffer> submitted_work_;
      ComPtr<ID3D11ShaderResourceView> submitted_rgb_;

      // Set before each map: a partial CUDA submission may outlive a failed API call.
      bool unsettled_ = false;
      enum class submission_e { idle, score, rejected_mapping };
      submission_e submission_ = submission_e::idle;
      bool leased_ = false;

      tensor_t make_tensor(UINT words, bool initialize_zero = false) const {
        const D3D11_BUFFER_DESC desc {words * 4u, D3D11_USAGE_DEFAULT,
          D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, 0u, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, 4u};
        const std::vector<std::uint32_t> zero(initialize_zero ? words : 0u, 0u);
        const D3D11_SUBRESOURCE_DATA initial {zero.data(), 0u, 0u};
        tensor_t result;
        require(SUCCEEDED(device_->CreateBuffer(&desc, initialize_zero ? &initial : nullptr, &result.buffer)) &&
                SUCCEEDED(device_->CreateShaderResourceView(result.buffer.Get(), nullptr, &result.srv)) &&
                SUCCEEDED(device_->CreateUnorderedAccessView(result.buffer.Get(), nullptr, &result.uav)),
                "Create direct glyph transport tensor");
        return result;
      }

      void load_shaders(const std::filesystem::path &shader_root) {
        constexpr auto &specs = cache::direct_glyph_transport_specs;
        static_assert(specs.size() == 4u);
        const auto sources = cache::snapshot_sources(shader_root, specs);
        require(sources && cache::source_closure_sha256(sources) == cache::direct_glyph_transport_source_closure_sha256,
                "Direct glyph transport source closure mismatch");
        for (std::size_t i = 0; i < specs.size(); ++i) {
          const auto code = cache::get(sources, specs[i]);
          if (!code || FAILED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &shaders_[i]))) {
            throw std::runtime_error("Create direct glyph transport shader " + std::string(specs[i].entrypoint));
          }
        }
      }

      void prepare_d3d_resources(const std::filesystem::path &shader_root) {
        load_shaders(shader_root);
        input_ = make_tensor(input_atlas_words, true);
        output_ = make_tensor(output_atlas_words);
        placement_ = make_tensor(placement_words, true);
        const D3D11_BUFFER_DESC params {32u, D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0u, 0u, 0u};
        require(SUCCEEDED(device_->CreateBuffer(&params, nullptr, &constants_)), "Create direct glyph constants");
        const D3D11_BUFFER_DESC work_desc {work_bytes, D3D11_USAGE_DEFAULT, D3D11_BIND_UNORDERED_ACCESS,
          0u, D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS, 0u};
        D3D11_UNORDERED_ACCESS_VIEW_DESC work_view {};
        work_view.Format = DXGI_FORMAT_R32_TYPELESS;
        work_view.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        work_view.Buffer.NumElements = work_bytes / 4u;
        work_view.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
        require(SUCCEEDED(device_->CreateBuffer(&work_desc, nullptr, &work_)) &&
                SUCCEEDED(device_->CreateUnorderedAccessView(work_.Get(), &work_view, &work_uav_)),
                "Create direct glyph private Work buffer");
        D3D11_BUFFER_DESC dispatch_desc = work_desc;
        dispatch_desc.ByteWidth = 16u;
        dispatch_desc.MiscFlags |= D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
        work_view.Buffer.NumElements = dispatch_desc.ByteWidth / sizeof(std::uint32_t);
        require(SUCCEEDED(device_->CreateBuffer(&dispatch_desc, nullptr, &tile_dispatch_)) &&
                SUCCEEDED(device_->CreateUnorderedAccessView(tile_dispatch_.Get(), &work_view, &tile_dispatch_uav_)),
                "Create direct glyph tile dispatch arguments");
      }

      std::array<CUdeviceptr, 3> mapped_addresses() const {
        std::array<CUdeviceptr, 3> pointers {};
        std::array<std::size_t, 3> sizes {};
        for (std::size_t i = 0; i < interop_.size(); ++i) {
          check_cuda(cuda_.cuGraphicsResourceGetMappedPointer(&pointers[i], &sizes[i], interop_[i]),
                     "Get a direct glyph CUDA tensor address");
        }
        require(pointers[0] && pointers[1] && pointers[2] &&
                sizes[0] == input_atlas_words * sizeof(float) && sizes[1] == output_atlas_words * sizeof(float) &&
                sizes[2] >= placement_words * sizeof(std::uint32_t), "Wrong direct glyph CUDA tensor extent");
        return pointers;
      }

      void ensure_alpha8(UINT width, UINT height) {
        if (alpha8_ && width == alpha_width_ && height == alpha_height_) return;
        const D3D11_TEXTURE2D_DESC desc {width, height, 1u, 1u, DXGI_FORMAT_R32_UINT, {1u, 0u}, D3D11_USAGE_DEFAULT,
          D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, 0u, 0u};
        ComPtr<ID3D11Texture2D> texture;
        ComPtr<ID3D11ShaderResourceView> srv;
        ComPtr<ID3D11UnorderedAccessView> uav;
        require(SUCCEEDED(device_->CreateTexture2D(&desc, nullptr, &texture)) &&
                SUCCEEDED(device_->CreateShaderResourceView(texture.Get(), nullptr, &srv)) &&
                SUCCEEDED(device_->CreateUnorderedAccessView(texture.Get(), nullptr, &uav)),
                "Create direct glyph alpha8 texture");
        alpha8_ = std::move(srv);
        alpha8_uav_ = std::move(uav);
        alpha_width_ = width;
        alpha_height_ = height;
      }

      void dispatch(stage_e stage, srvs_t sources, uavs_t targets, ID3D11Buffer *groups = nullptr) const {
        context_->CSSetShader(shaders_[stage].Get(), nullptr, 0u);
        context_->CSSetShaderResources(0u, UINT(sources.size()), sources.data());
        context_->CSSetUnorderedAccessViews(0u, UINT(targets.size()), targets.data(), nullptr);
        if (groups) {
          context_->DispatchIndirect(groups, 0u);
        } else {
          context_->Dispatch(1u, 1u, 1u);
        }
        sources.fill(nullptr);
        targets.fill(nullptr);
        context_->CSSetShaderResources(0u, UINT(sources.size()), sources.data());
        context_->CSSetUnorderedAccessViews(0u, UINT(targets.size()), targets.data(), nullptr);
      }

      void drop_request() noexcept {
        submitted_ = {};
        submitted_work_.Reset();
        submitted_rgb_.Reset();
      }

     public:
      void prepare(ID3D11Device *device, ID3D11DeviceContext *context, const std::filesystem::path &shader_root,
                   CUcontext primary, const execution_profiles_t &executions) {
        require(device && context && primary && executions[0u] && executions[1u] &&
                executions[0u] != executions[1u],
                "Direct glyph requires an immediate D3D11 context and distinct TensorRT profile leases");
        ComPtr<ID3D11DeviceContext> immediate;
        device->GetImmediateContext(&immediate);
        require(immediate.Get() == context, "Foreign or deferred D3D11 context for direct glyph");
        device_ = device;
        context_ = context;

        require(cuda_.is_valid() && cuda_.has_conditional_graph_support() && cuda_.cuStreamBeginCapture &&
                cuda_.cuStreamEndCapture && cuda_.cuD3D11GetDevice && cuda_.cuDevicePrimaryCtxRetain &&
                cuda_.cuDevicePrimaryCtxRelease && cuda_.cuEventCreate && cuda_.cuEventQuery &&
                cuda_.cuEventRecord && cuda_.cuEventDestroy, "Direct glyph CUDA score transport is unavailable");
        check_cuda(cuda_.cuInit(0u), "Initialize direct glyph CUDA transport");
        ComPtr<IDXGIDevice> dxgi;
        ComPtr<IDXGIAdapter> adapter;
        require(SUCCEEDED(device_.As(&dxgi)) && SUCCEEDED(dxgi->GetAdapter(&adapter)),
                "Resolve the direct glyph D3D11 adapter");
        check_cuda(cuda_.cuD3D11GetDevice(&cuda_device_, adapter.Get()), "Join direct glyph to the exact D3D11 adapter");
        check_cuda(cuda_.cuDevicePrimaryCtxRetain(&primary_, cuda_device_), "Retain the direct glyph CUDA primary");
        require(primary_ == primary, "Direct glyph TensorRT lease used a different CUDA primary");

        prepare_d3d_resources(shader_root);
        const current_cuda_context_t guard(cuda_, primary_);
        check_cuda(cuda_.cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING), "Create a private direct glyph stream");
        check_cuda(cuda_.cuEventCreate(&completed_, event_disable_timing), "Create direct glyph completion event");
        const std::array buffers {input_.buffer.Get(), output_.buffer.Get(), placement_.buffer.Get()};
        for (std::size_t i = 0; i < buffers.size(); ++i) {
          check_cuda(cuda_.cuGraphicsD3D11RegisterResource(&interop_[i], buffers[i], 0u), "Register direct glyph CUDA interop");
        }

        for (unsigned int i = 0u; i < tile_profile_count; ++i) {
          const auto shape = tile_shape(static_cast<tile_profile_e>(i));
          const auto output_bytes = static_cast<std::int64_t>(shape.output_words() * sizeof(float));
          require(executions[i]->getMaxOutputSize("logits") == output_bytes,
                  "Wrong direct glyph TensorRT profile output size");
        }
        unsettled_ = true;
        check_cuda(cuda_.cuGraphicsMapResources(UINT(interop_.size()), interop_.data(), stream_),
                   "Map direct glyph startup tensors");
        try {
          const auto mapped = mapped_addresses();
          for (unsigned int i = 0u; i < tile_profile_count; ++i) {
            const auto shape = tile_shape(static_cast<tile_profile_e>(i));
            graphs_[i].prepare(primary_, stream_, *executions[i], mapped, shape);
          }
        } catch (...) {
          // Startup may prove retirement before reporting failure and giving the lease back.
          if (cuda_.cuGraphicsUnmapResources(UINT(interop_.size()), interop_.data(), stream_) == CUDA_SUCCESS &&
              cuda_.cuStreamSynchronize(stream_) == CUDA_SUCCESS) {
            unsettled_ = false;
          }
          throw;
        }
        check_cuda(cuda_.cuGraphicsUnmapResources(UINT(interop_.size()), interop_.data(), stream_),
                   "Unmap direct glyph startup tensors");
        check_cuda(cuda_.cuStreamSynchronize(stream_), "Finish direct glyph graph preparation");
        unsettled_ = false;
      }

      bool busy() const noexcept {
        return submission_ != submission_e::idle || leased_;
      }

      bool returning_rejected_mapping() const noexcept {
        return submission_ == submission_e::rejected_mapping;
      }

      void submit(const request_t &request) {
        constexpr auto limit = D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION;
        require(request.work && request.token && request.width && request.width <= limit &&
                request.band_height && request.band_height <= limit && request.crop_top <= limit - request.band_height,
                "Direct glyph submission lacks exact Work ownership or geometry");
        const auto profile_index = static_cast<unsigned int>(request.profile);
        require(profile_index < tile_profile_count, "Invalid direct glyph score profile");
        auto &graph = graphs_[profile_index];
        const auto tile = tile_shape(request.profile);
        ComPtr<ID3D11Device> work_device;
        request.work->GetDevice(&work_device);
        D3D11_BUFFER_DESC desc {};
        request.work->GetDesc(&desc);
        require(work_device.Get() == device_.Get() && desc.ByteWidth == work_bytes &&
                (desc.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS) &&
                desc.Usage == D3D11_USAGE_DEFAULT, "Wrong direct glyph caller Work buffer ABI");
        ID3D11ShaderResourceView *source = request.source_rgb;
        if (source) {
          ComPtr<ID3D11Device> source_device;
          source->GetDevice(&source_device);
          require(source_device.Get() == device_.Get(), "Foreign direct glyph source SRV");
          D3D11_SHADER_RESOURCE_VIEW_DESC source_desc {};
          source->GetDesc(&source_desc);
          if (source_desc.Format != DXGI_FORMAT_R16G16B16A16_FLOAT ||
              source_desc.ViewDimension != D3D11_SRV_DIMENSION_TEXTURE2D ||
              source_desc.Texture2D.MostDetailedMip != 0u || source_desc.Texture2D.MipLevels != 1u) {
            source = nullptr;  // plan_tile records unavailable for active Work without a valid RGB source.
          }
        }
        ensure_alpha8(request.width, request.band_height);
        if (request.capture_plan && !post_plan_work_) {
          const D3D11_BUFFER_DESC snapshot {work_bytes, D3D11_USAGE_DEFAULT, 0u, 0u, 0u, 0u};
          require(SUCCEEDED(device_->CreateBuffer(&snapshot, nullptr, &post_plan_work_)),
                  "Create a direct glyph plan snapshot for evaluation");
        }
        const current_cuda_context_t guard(cuda_, primary_);
        submitted_ = request;
        submitted_work_ = request.work;
        submitted_rgb_ = request.source_rgb;
        {
          const compute_state_t saved(context_.Get());
          context_->CopyResource(work_.Get(), submitted_work_.Get());
          const std::array<UINT, 8> shape {request.width, request.band_height, request.crop_top, cutoff,
                                          tile.width, tile.height, request.adaptive_context ? 1u : 0u, 0u};
          context_->UpdateSubresource(constants_.Get(), 0u, nullptr, shape.data(), 0u, 0u);
          context_->CSSetConstantBuffers(0u, 1u, constants_.GetAddressOf());
          dispatch(plan_tile, {source, nullptr, nullptr}, {placement_.uav.Get(), work_uav_.Get(), tile_dispatch_uav_.Get()});
          if (request.capture_plan) context_->CopyResource(post_plan_work_.Get(), work_.Get());
          dispatch(pack_rgb, {source, nullptr, placement_.srv.Get()}, {input_.uav.Get(), nullptr},
                   tile_dispatch_.Get());
        }
        unsettled_ = true;
        check_cuda(cuda_.cuGraphicsMapResources(UINT(interop_.size()), interop_.data(), stream_),
                   "Map direct glyph scoring tensors");
        std::array<CUdeviceptr, 3> mapped;
        try {
          mapped = mapped_addresses();
          graph.validate_mapping(mapped);
        } catch (...) {
          // This rejection precedes any graph launch. Return the resources to D3D and keep
          // their lease until the unmap event retires, without waiting in the capture loop.
          if (cuda_.cuGraphicsUnmapResources(UINT(interop_.size()), interop_.data(), stream_) == CUDA_SUCCESS &&
              cuda_.cuEventRecord(completed_, stream_) == CUDA_SUCCESS) {
            submission_ = submission_e::rejected_mapping;
          }
          throw;
        }
        graph.launch(stream_, mapped);
        check_cuda(cuda_.cuGraphicsUnmapResources(UINT(interop_.size()), interop_.data(), stream_),
                   "Unmap direct glyph scoring tensors");
        check_cuda(cuda_.cuEventRecord(completed_, stream_), "Record direct glyph ownership completion");
        submission_ = submission_e::score;
      }

      poll_status_e poll(const request_t *current, result_t &result) {
        if (submission_ == submission_e::idle) return poll_status_e::idle;
        const current_cuda_context_t guard(cuda_, primary_);
        const auto status = cuda_.cuEventQuery(completed_);
        if (status == CUDA_ERROR_NOT_READY) return poll_status_e::pending;
        check_cuda(status, "Query direct glyph ownership completion");
        const bool rejected = returning_rejected_mapping();
        submission_ = submission_e::idle;
        unsettled_ = false;
        if (rejected || !current || *current != submitted_) {
          drop_request();
          return poll_status_e::discarded;
        }
        {
          const compute_state_t saved(context_.Get());
          context_->CSSetConstantBuffers(0u, 1u, constants_.GetAddressOf());
          dispatch(resolve_matte, {output_.srv.Get(), nullptr, nullptr}, {alpha8_uav_.Get(), placement_.uav.Get()},
                   tile_dispatch_.Get());
          dispatch(finish_score, {nullptr, nullptr, placement_.srv.Get()}, {nullptr, work_uav_.Get()});
          context_->CopyResource(submitted_work_.Get(), work_.Get());
        }
        result.alpha8 = alpha8_;
        result.completed_placement = placement_.srv;
        leased_ = true;
        return poll_status_e::ready;
      }

      void retire() noexcept {
        if (!leased_) return;
        leased_ = false;
        drop_request();
      }

      ID3D11Buffer *placement_for_evaluation() const noexcept {
        return placement_.buffer.Get();
      }

      ID3D11Buffer *post_plan_work_for_evaluation() const noexcept {
        return post_plan_work_.Get();
      }

      bool release() noexcept {
        if (unsettled_ || busy()) return false;
        if (!primary_) return true;
        try {
          const current_cuda_context_t guard(cuda_, primary_);
          for (auto &graph : graphs_) {
            if (!graph.release()) return false;
          }
          for (auto &resource : interop_) {
            if (resource && cuda_.cuGraphicsUnregisterResource(resource) != CUDA_SUCCESS) return false;
            resource = nullptr;
          }
          if (completed_ && cuda_.cuEventDestroy(completed_) != CUDA_SUCCESS) return false;
          completed_ = nullptr;
          if (stream_ && cuda_.cuStreamDestroy(stream_) != CUDA_SUCCESS) return false;
          stream_ = nullptr;
        } catch (...) {
          return false;
        }
        if (cuda_.cuDevicePrimaryCtxRelease(cuda_device_) != CUDA_SUCCESS) return false;
        primary_ = nullptr;
        return true;
      }
    };
    #include "host_sbs_direct_glyph_palette.inc"
  }  // namespace detail

  // Serialized by the caller's immediate context. It owns publication and gives each
  // observation a unique token. Work must stay reserved until ready or discarded.
  class resources_t {
   public:
    resources_t() = default;
    resources_t(const resources_t &) = delete;
    resources_t &operator=(const resources_t &) = delete;

    ~resources_t() {
      // Retain all operands if CUDA may still be using them. Both execution leases must
      // be quarantined by the caller whenever release() cannot prove safe teardown.
      if (!release()) {
        (void)transport_.release();
        (void)palette_.release();
      }
    }

    bool prepare(ID3D11Device *device, ID3D11DeviceContext *context, const std::filesystem::path &shader_root,
                 CUcontext primary, const execution_profiles_t &executions, std::string &error, bool palette = false) {
      if (transport_ || palette_ || failed_) {
        error = "Direct glyph score transport already owns an execution lease";
        return false;
      }
      try {
        if (palette) {
          palette_ = std::make_unique<detail::palette_transport_t>();
          palette_->prepare(device, context, shader_root, primary, executions[0]);
        } else {
          transport_ = std::make_unique<detail::transport_t>();
          transport_->prepare(device, context, shader_root, primary, executions);
        }
        error.clear();
        return true;
      } catch (const std::exception &why) {
        failed_ = true;
        error = why.what();
        return false;
      }
    }

    submit_status_e try_submit(const request_t &request, std::string &error) {
      if ((!transport_ && !palette_) || failed_) {
        error = "Direct glyph score transport is unavailable";
        return submit_status_e::failed;
      }
      error.clear();
      if (palette_ ? palette_->busy() : transport_->busy()) return submit_status_e::busy;
      try {
        if (palette_) palette_->submit(request);
        else transport_->submit(request);
        return submit_status_e::accepted;
      } catch (const std::exception &why) {
        failed_ = true;
        error = why.what();
        return submit_status_e::failed;
      }
    }

    // Returns a score lease only for this exact request. GPU Work separately attests
    // EMPTY, valid score, or unavailable. A ready event alone is never matte authority.
    poll_status_e poll(const request_t *current, result_t &result, std::string &error) {
      result = {};
      if (palette_ && !failed_) {
        try {
          return palette_->poll(current, result);
        } catch (const std::exception &why) {
          failed_ = true;
          error = why.what();
          return poll_status_e::failed;
        }
      }
      error.clear();
      if (failed_ && (current || !transport_ || !transport_->returning_rejected_mapping())) {
        error = "Direct glyph score transport failed";
        return poll_status_e::failed;
      }
      if (!transport_) return poll_status_e::idle;
      try {
        return transport_->poll(current, result);
      } catch (const std::exception &why) {
        failed_ = true;
        error = why.what();
        return poll_status_e::failed;
      }
    }

    // Call after recording every same-context consumer of alpha8 and updated Work.
    void retire() noexcept {
      if (palette_) palette_->retire();
      if (transport_) transport_->retire();
    }

    [[nodiscard]] ID3D11Buffer *placement_for_evaluation() const noexcept {
      return transport_ ? transport_->placement_for_evaluation() : nullptr;
    }

    [[nodiscard]] ID3D11Buffer *post_plan_work_for_evaluation() const noexcept {
      return transport_ ? transport_->post_plan_work_for_evaluation() : nullptr;
    }
    [[nodiscard]] ID3D11Buffer *native_lines_for_evaluation() const noexcept {
      return palette_ ? palette_->committed_lines() : nullptr;
    }

    bool release() noexcept {
      if (palette_ && !palette_->release()) return false;
      palette_.reset();
      if (transport_ && !transport_->release()) return false;
      transport_.reset();
      return true;
    }

    bool safe_to_recycle_context() const noexcept {
      return !transport_ && !palette_;
    }

   private:
    std::unique_ptr<detail::transport_t> transport_;
    std::unique_ptr<detail::palette_transport_t> palette_;
    bool failed_ = false;
  };
}  // namespace models::host_sbs_direct_glyph_gpu
