#pragma once

// External private research backend. True means ordered D3D commands were queued, never that
// a GPU receipt is available, a subtitle is present/absent, or these outputs may be drawn.
// Inputs and shader lifetimes must remain stable through their immediate-context commands.
// Every private working resource must be unbound outside the touched CS slots0..8/UAV0..7;
// this scoped helper cannot restore bindings in another pipeline stage. AddRef preserves
// storage, never its contents. Exposed views remain mutable across begin/ensure/end.
// Direct hooks save/get/restore repeatedly; no fast-path or command-list cost is qualified.
#ifdef _WIN32

  #include <algorithm>
  #include <array>
  #include <cstddef>
  #include <cstdint>
  #include <d3d11.h>
  #include <limits>
  #include <string>
  #include <utility>
  #include <wrl/client.h>

namespace models::host_sbs_body_witness_temporal_gpu {
  using Microsoft::WRL::ComPtr;

  inline constexpr std::array<const char *, 19u> entry_names {
    "begin_main",
    "frame_clear_main",
    "unit_main",
    "mode_roots_main",
    "mode_body_main",
    "mode_alloc_main",
    "mode_context_main",
    "prepare_end_main",
    "hash_main",
    "component_stats_main",
    "duplicate_order_main",
    "decide_main",
    "overlap_main",
    "overlap_reduce_main",
    "budget_main",
    "budget_finish_main",
    "scatter_main",
    "publish_main",
    "commit_main"
  };

  struct caps_t {
    UINT components = 4096u;
    UINT members = 1048576u;
    UINT bit_words = 1048576u;
    UINT hash_words = 8192u;
    bool operator==(const caps_t &) const = default;
  };

  // Exact ABI b0: shape4, SOURCE identity4, control4, caps4, scope32,
  // MODEL identity4 at48..51, SourceContinuity{bool,0,0,0} at52..55, reserved8.
  // Source serial/clock are genuine captured-material
  // identity, never a conversion/finish counter. Model identity joins exact currentCB.
  // Scope is a caller-declared, authenticated source namespace; serialization is not authority.
  struct alignas(16) frame_constants_t {
    std::array<std::uint32_t, 64u> words {};
  };

  static_assert(sizeof(frame_constants_t) == 256u);

  struct mutable_views_t {
    UINT width = 0u;
    UINT height = 0u;
    caps_t capacities;
    ComPtr<ID3D11ShaderResourceView> body;
    ComPtr<ID3D11ShaderResourceView> matte;
    ComPtr<ID3D11ShaderResourceView> output_array;
    ComPtr<ID3D11ShaderResourceView> state;
    ComPtr<ID3D11ShaderResourceView> records;
    ComPtr<ID3D11ShaderResourceView> members;
    ComPtr<ID3D11ShaderResourceView> bits;
    ComPtr<ID3D11ShaderResourceView> root_rgb;
    ComPtr<ID3D11ShaderResourceView> images;
    ComPtr<ID3D11ShaderResourceView> receipt;
    ComPtr<ID3D11Buffer> receipt_storage;
    ComPtr<ID3D11Buffer> end_arguments;
    ComPtr<ID3D11Buffer> hook_arguments;
    ComPtr<ID3D11Buffer> frame_constants;
    ComPtr<ID3D11Buffer> hook_constants;
    ComPtr<ID3D11Buffer> native_arguments;
  };

  class resources_t {
    struct buffer_t {
      ComPtr<ID3D11Buffer> resource;
      ComPtr<ID3D11ShaderResourceView> srv;
      ComPtr<ID3D11UnorderedAccessView> uav;
    };

    struct texture_t {
      ComPtr<ID3D11Texture2D> resource;
      ComPtr<ID3D11ShaderResourceView> srv;
      ComPtr<ID3D11UnorderedAccessView> uav;
    };

    class saved_cs_t {
    public:
      explicit saved_cs_t(ID3D11DeviceContext *context, bool capture = true):
          context_(context),
          capture_(capture) {
        if (!capture_) {
          return;
        }
        std::array<ID3D11ShaderResourceView *, 9u> srvs {};
        std::array<ID3D11UnorderedAccessView *, 8u> uavs {};
        std::array<ID3D11Buffer *, 3u> constants {};
        context_->CSGetShaderResources(0u, UINT(srvs.size()), srvs.data());
        context_->CSGetUnorderedAccessViews(0u, UINT(uavs.size()), uavs.data());
        context_->CSGetConstantBuffers(0u, UINT(constants.size()), constants.data());
        for (UINT i = 0u; i < srvs.size(); ++i) {
          srvs_[i].Attach(srvs[i]);
        }
        for (UINT i = 0u; i < uavs.size(); ++i) {
          uavs_[i].Attach(uavs[i]);
        }
        for (UINT i = 0u; i < constants.size(); ++i) {
          constants_[i].Attach(constants[i]);
        }
        std::array<ID3D11ClassInstance *, 256u> instances {};
        count_ = UINT(instances.size());
        ID3D11ComputeShader *shader = nullptr;
        context_->CSGetShader(&shader, instances.data(), &count_);
        shader_.Attach(shader);
        for (UINT i = 0u; i < count_; ++i) {
          instances_[i].Attach(instances[i]);
        }
      }

      saved_cs_t(const saved_cs_t &) = delete;
      saved_cs_t &operator=(const saved_cs_t &) = delete;

      ~saved_cs_t() {
        unbind(context_);
        if (!capture_) {
          return;
        }
        std::array<ID3D11ShaderResourceView *, 9u> srvs {};
        std::array<ID3D11UnorderedAccessView *, 8u> uavs {};
        std::array<ID3D11Buffer *, 3u> constants {};
        std::array<ID3D11ClassInstance *, 256u> instances {};
        for (UINT i = 0u; i < srvs.size(); ++i) {
          srvs[i] = srvs_[i].Get();
        }
        for (UINT i = 0u; i < uavs.size(); ++i) {
          uavs[i] = uavs_[i].Get();
        }
        for (UINT i = 0u; i < constants.size(); ++i) {
          constants[i] = constants_[i].Get();
        }
        for (UINT i = 0u; i < count_; ++i) {
          instances[i] = instances_[i].Get();
        }
        context_->CSSetUnorderedAccessViews(0u, UINT(uavs.size()), uavs.data(), nullptr);
        context_->CSSetShaderResources(0u, UINT(srvs.size()), srvs.data());
        context_->CSSetConstantBuffers(0u, UINT(constants.size()), constants.data());
        context_->CSSetShader(shader_.Get(), count_ ? instances.data() : nullptr, count_);
      }

      static void unbind(ID3D11DeviceContext *context) noexcept {
        std::array<ID3D11ShaderResourceView *, 9u> srvs {};
        std::array<ID3D11UnorderedAccessView *, 8u> uavs {};
        context->CSSetShaderResources(0u, UINT(srvs.size()), srvs.data());
        context->CSSetUnorderedAccessViews(0u, UINT(uavs.size()), uavs.data(), nullptr);
      }

    private:
      ID3D11DeviceContext *context_;
      bool capture_ = true;
      std::array<ComPtr<ID3D11ShaderResourceView>, 9u> srvs_;
      std::array<ComPtr<ID3D11UnorderedAccessView>, 8u> uavs_;
      std::array<ComPtr<ID3D11Buffer>, 3u> constants_;
      std::array<ComPtr<ID3D11ClassInstance>, 256u> instances_;
      ComPtr<ID3D11ComputeShader> shader_;
      UINT count_ = 0u;
    };

  public:
    resources_t() = default;
    resources_t(const resources_t &) = delete;
    resources_t &operator=(const resources_t &) = delete;
    resources_t(resources_t &&) = default;
    resources_t &operator=(resources_t &&) = default;

    bool ensure(ID3D11Device *device, UINT width, UINT height, caps_t capacities, std::string &error) {
      error.clear();
      const auto pixels = std::uint64_t {width} * height;
      if (open_ || poisoned_ || !device || width == 0u || height == 0u || width > 5120u || height > 5120u || capacities.components == 0u || capacities.components > 65535u || capacities.members == 0u || capacities.members > 65535u * 256u || capacities.bit_words == 0u || capacities.bit_words > (1u << 25u) || capacities.hash_words == 0u || (capacities.hash_words & (capacities.hash_words - 1u)) != 0u || std::uint64_t {capacities.hash_words} < 2ull * capacities.components || !buffer_size(512u, 4u) || !buffer_size(2ull * (512ull + 64ull * capacities.components + capacities.hash_words), 4u) || !buffer_size(2ull * capacities.members, 8u) || !buffer_size(2ull * capacities.bit_words, 4u) || !buffer_size(5ull * pixels + 1u, 4u)) {
        error = "Temporal backend extent/capacity invalid, unsupported, or a session is still open.";
        return false;
      }
      if (device_.Get() == device && width_ == width && height_ == height && caps_ == capacities) {
        return true;
      }
      resources_t next;
      next.device_ = device;
      next.width_ = width;
      next.height_ = height;
      next.pixels_ = UINT(pixels);
      next.caps_ = capacities;
      if (!next.allocate(error)) {
        return false;
      }
      *this = std::move(next);
      return true;
    }

    // Owned coordinator only: one outer state/predication guard belongs to the caller. Execute
    // BEFORE every command-list replay. Live256B is never captured by the cached program.
    bool prepare_live_frame_owned(ID3D11DeviceContext *context, const frame_constants_t &frame, std::string &error) {
      error.clear();
      if (open_ || poisoned_) {
        return fail(error, "Open/poisoned temporal owned scope cannot prepare.");
      }
      if (!context_ok(context, error)) {
        return false;
      }
      if (!frame_ok(frame)) {
        return fail(error, "Owned temporal frame shape/caps differ allocation.");
      }
      saved_cs_t::unbind(context);
      if (needs_initialize_) {
        constexpr std::array<UINT, 4u> zero {};
        for (auto *view : uav_array()) {
          context->ClearUnorderedAccessViewUint(view, zero.data());
        }
        needs_initialize_ = false;
      }
      context->UpdateSubresource(frame_constants_.Get(), 0u, nullptr, &frame, 0u, 0u);
      frame_prepared_ = true;
      return true;
    }

    // EXCLUSIVE known-state owned context only (immediate/deferred). The coordinator rejects
    // caller predication, ClearState/SetPredication(null) on its OWN deferred context, and
    // initializes/updates liveframe on immediate. No CSGet* executes in this owned emission.
    bool begin_owned_emission(ID3D11DeviceContext *context, const std::array<ID3D11ComputeShader *, 19u> &shaders, ID3D11Buffer *current_constants96, ID3D11ShaderResourceView *current_rgb, ID3D11ShaderResourceView *numeric_receipt32, ID3D11Buffer *root_arguments, UINT root_offset, std::string &error) {
      error.clear();
      if (open_ || poisoned_ || needs_initialize_ || !frame_prepared_) {
        return fail(error, "Owned temporal emission requires one prepared private scope.");
      }
      if (!owned_context_ok(context, error)) {
        return false;
      }
      for (auto *shader : shaders) {
        if (!child_ok(shader)) {
          return fail(error, "Missing/wrong-device owned temporal shader.");
        }
      }
      if (!constant_ok(current_constants96, 96u) || !texture_ok(current_rgb, DXGI_FORMAT_R16G16B16A16_FLOAT) || !structured_ok(numeric_receipt32, 32u) || !arguments_ok(root_arguments, root_offset)) {
        return fail(error, "Owned temporal descriptors invalid.");
      }
      const saved_cs_t saved {context, false};
      saved_cs_t::unbind(context);
      constexpr std::array<UINT, 4u> zero {};
      context->ClearUnorderedAccessViewUint(output_.uav.Get(), zero.data());
      context->ClearUnorderedAccessViewUint(receipt_.uav.Get(), zero.data());
      copy_words(context, native_arguments_.Get(), 8u, root_arguments, root_offset, 4u);
      current_constants_ = current_constants96;
      current_rgb_ = current_rgb;
      validation_ = numeric_receipt32;
      root_arguments_ = root_arguments;
      root_offset_ = root_offset;
      context_ = context;
      for (UINT i = 0u; i < shaders.size(); ++i) {
        shaders_[i] = shaders[i];
      }
      inputs_.fill(nullptr);
      inputs_[0] = current_rgb_.Get();
      inputs_[6] = validation_.Get();
      owned_emission_ = true;
      bind_stage(context);
      dispatch(context, 0u, root_arguments_.Get(), root_offset_);
      dispatch(context, 1u, native_arguments_.Get(), 0u);
      open_ = true;
      next_unit_ = 0u;
      next_mode_ = 8u;
      return true;
    }

    // Discard a half-recorded list BEFORE this call. Revoke current outputs on immediate;
    // preserve the historical watermark and poison this private scope rather than reuse it.
    bool quarantine_owned(ID3D11DeviceContext *context, std::string &error) {
      error.clear();
      if (!context_ok(context, error)) {
        return false;
      }
      saved_cs_t::unbind(context);
      constexpr std::array<UINT, 4u> zero {};
      for (auto *view : {output_.uav.Get(), receipt_.uav.Get(), body_.uav.Get(), matte_.uav.Get()}) {
        context->ClearUnorderedAccessViewUint(view, zero.data());
      }
      release_session();
      poisoned_ = true;
      return true;
    }

    // Fixed host schedule: eight units, each palette then eight modes, then end. Inactive modes
    // still call the hook with canonical000 arguments; no CPU palette or segmentation verdict.
    // The baseline producer must provide exact frozenV5 selected mode planes, not authored masks.
    bool begin(ID3D11DeviceContext *context, const std::array<ID3D11ComputeShader *, 19u> &shaders, const frame_constants_t &frame, ID3D11Buffer *current_constants96, ID3D11ShaderResourceView *current_rgb, ID3D11ShaderResourceView *numeric_receipt32, ID3D11Buffer *root_arguments, UINT root_offset, std::string &error) {
      error.clear();
      if (open_ || poisoned_) {
        return fail(error, "Temporal session still open or abandoned; a new authenticated private scope is required.");
      }
      if (!context_ok(context, error)) {
        return false;
      }
      for (auto *shader : shaders) {
        if (!child_ok(shader)) {
          return fail(error, "Missing/wrong-device temporal shader.");
        }
      }
      if (!constant_ok(current_constants96, 96u) || !texture_ok(current_rgb, DXGI_FORMAT_R16G16B16A16_FLOAT) || !structured_ok(numeric_receipt32, 32u) || !arguments_ok(root_arguments, root_offset)) {
        return fail(error, "Temporal current source/CB/validation/root-argument descriptors invalid.");
      }
      if (frame.words[0] != width_ || frame.words[1] != height_ || frame.words[2] != pixels_ || frame.words[3] != 0u || frame.words[12] != caps_.components || frame.words[13] != caps_.members || frame.words[14] != caps_.bit_words || frame.words[15] != caps_.hash_words) {
        return fail(error, "Temporal frame shape/caps differ physical allocation.");
      }
      const saved_cs_t saved {context, !owned_emission_};
      saved_cs_t::unbind(context);
      if (needs_initialize_) {
        constexpr std::array<UINT, 4u> zero {};
        for (auto *view : uav_array()) {
          context->ClearUnorderedAccessViewUint(view, zero.data());
        }
        needs_initialize_ = false;  // Queued initialization, not GPU completion or pattern validity.
      }
      constexpr std::array<UINT, 4u> zero {};
      context->ClearUnorderedAccessViewUint(output_.uav.Get(), zero.data());
      context->ClearUnorderedAccessViewUint(receipt_.uav.Get(), zero.data());
      context->UpdateSubresource(frame_constants_.Get(), 0u, nullptr, &frame, 0u, 0u);
      copy_words(context, native_arguments_.Get(), 8u, root_arguments, root_offset, 4u);
      current_constants_ = current_constants96;
      current_rgb_ = current_rgb;
      validation_ = numeric_receipt32;
      root_arguments_ = root_arguments;
      root_offset_ = root_offset;
      context_ = context;
      for (UINT i = 0u; i < shaders.size(); ++i) {
        shaders_[i] = shaders[i];
      }
      inputs_.fill(nullptr);
      inputs_[0] = current_rgb_.Get();
      inputs_[6] = validation_.Get();
      bind_stage(context);
      dispatch(context, 0u, root_arguments_.Get(), root_offset_);
      dispatch(context, 1u, native_arguments_.Get(), 0u);
      open_ = true;
      next_unit_ = 0u;
      next_mode_ = 8u;
      return true;
    }

    bool unit_palette(ID3D11DeviceContext *context, UINT slot, ID3D11ShaderResourceView *palette128, std::string &error) {
      error.clear();
      if (!session_ok(context, error)) {
        return false;
      }
      if (slot != next_unit_ || slot >= 8u || next_mode_ != 8u || !structured_ok(palette128, 128u)) {
        return fail(error, "Temporal unit hook order/palette invalid; incomplete session is unavailable.");
      }
      const saved_cs_t saved {context, !owned_emission_};
      saved_cs_t::unbind(context);
      set_hook(context, slot, 0u);
      inputs_.fill(nullptr);
      inputs_[0] = current_rgb_.Get();
      inputs_[5] = palette128;
      inputs_[6] = validation_.Get();
      bind_stage(context);
      dispatch(context, 2u, root_arguments_.Get(), root_offset_);
      next_mode_ = 0u;
      return true;
    }

    bool mode_after_ccresolve(ID3D11DeviceContext *context, UINT slot, UINT mode, ID3D11ShaderResourceView *seeds, ID3D11ShaderResourceView *mode_body, ID3D11ShaderResourceView *parent, ID3D11ShaderResourceView *root_flags, ID3D11ShaderResourceView *palette128, ID3D11ShaderResourceView *cover, ID3D11Buffer *mode_arguments, UINT mode_offset, std::string &error) {
      error.clear();
      if (!session_ok(context, error)) {
        return false;
      }
      if (slot != next_unit_ || mode != next_mode_ || slot >= 8u || mode >= 8u || !texture_ok(seeds, DXGI_FORMAT_R32_UINT) || !texture_ok(mode_body, DXGI_FORMAT_R32_UINT) || !texture_ok(cover, DXGI_FORMAT_R32_UINT) || !structured_ok(parent, pixels_ + 1u) || !structured_ok(root_flags, pixels_ + 1u) || !structured_ok(palette128, 128u) || !arguments_ok(mode_arguments, mode_offset)) {
        return fail(error, "Temporal mode hook order/planes/graph/arguments invalid.");
      }
      const saved_cs_t saved {context, !owned_emission_};
      saved_cs_t::unbind(context);
      // Distinct D3D-only argument storage. NativeXY with rootX asZ also defeats stale
      // nonzero mode arguments on root0. Canonical inactiveXY0 continues to dispatch0.
      copy_words(context, hook_arguments_.Get(), 0u, mode_arguments, mode_offset, 8u);
      copy_words(context, hook_arguments_.Get(), 8u, root_arguments_.Get(), root_offset_, 4u);
      set_hook(context, slot, mode);
      inputs_ = {current_rgb_.Get(), seeds, mode_body, parent, root_flags, palette128, validation_.Get(), cover};
      bind_stage(context);
      for (UINT entry : {3u, 4u, 5u, 6u}) {
        dispatch(context, entry, hook_arguments_.Get(), 0u);
      }
      ++next_mode_;
      if (next_mode_ == 8u) {
        ++next_unit_;
      }
      return true;
    }

    bool end(ID3D11DeviceContext *context, std::string &error) {
      error.clear();
      if (!session_ok(context, error)) {
        return false;
      }
      if (next_unit_ != 8u || next_mode_ != 8u) {
        return fail(error, "Temporal end requires all8x8 hook positions.");
      }
      const saved_cs_t saved {context, !owned_emission_};
      saved_cs_t::unbind(context);
      inputs_.fill(nullptr);
      inputs_[0] = current_rgb_.Get();
      inputs_[6] = validation_.Get();
      bind_stage(context);
      dispatch(context, 7u, root_arguments_.Get(), root_offset_);
      saved_cs_t::unbind(context);
      // A UAV may not simultaneously supply DispatchIndirect arguments. These literal six
      // GPU words are copied once after preparation; no component count reaches the CPU.
      copy_words(context, end_arguments_.Get(), 0u, receipt_.resource.Get(), 192u * 4u, 24u);
      bind_stage(context);
      for (UINT entry : {8u, 9u, 10u, 11u}) {
        dispatch(context, entry, end_arguments_.Get(), 0u);
      }
      dispatch(context, 12u, end_arguments_.Get(), 12u);
      for (UINT entry : {13u, 14u}) {
        dispatch(context, entry, end_arguments_.Get(), 0u);
      }
      dispatch(context, 15u, root_arguments_.Get(), root_offset_);
      dispatch(context, 16u, end_arguments_.Get(), 12u);
      dispatch(context, 17u, native_arguments_.Get(), 0u);
      dispatch(context, 18u, root_arguments_.Get(), root_offset_);
      saved_cs_t::unbind(context);
      context->CopySubresourceRegion(body_.resource.Get(), 0u, 0u, 0u, 0u, output_.resource.Get(), 0u, nullptr);
      context->CopySubresourceRegion(matte_.resource.Get(), 0u, 0u, 0u, 0u, output_.resource.Get(), 1u, nullptr);
      release_session();
      return true;
    }

    // A failed/incomplete hook sequence must never be exported as current. This method clears
    // current output/receipt and revokes this host session; it does not advance GPU history.
    bool abandon(ID3D11DeviceContext *context, std::string &error) {
      error.clear();
      if (!session_ok(context, error)) {
        return false;
      }
      const saved_cs_t saved {context, !owned_emission_};
      saved_cs_t::unbind(context);
      constexpr std::array<UINT, 4u> zero {};
      context->ClearUnorderedAccessViewUint(output_.uav.Get(), zero.data());
      context->ClearUnorderedAccessViewUint(receipt_.uav.Get(), zero.data());
      context->ClearUnorderedAccessViewUint(body_.uav.Get(), zero.data());
      context->ClearUnorderedAccessViewUint(matte_.uav.Get(), zero.data());
      release_session();
      // Never silently clear the retained watermark after a partial observation. A later begin
      // is unavailable; caller must construct a new authenticated private source/pipeline scope.
      poisoned_ = true;
      return true;
    }

    [[nodiscard]] mutable_views_t mutable_views() const {
      return {width_, height_, caps_, body_.srv, matte_.srv, output_.srv, state_.srv, records_.srv, members_.srv, bits_.srv, root_rgb_.srv, images_.srv, receipt_.srv, receipt_.resource, end_arguments_, hook_arguments_, frame_constants_, hook_constants_, native_arguments_};
    }

  private:
    static bool fail(std::string &error, const char *message) {
      error = message;
      return false;
    }

    static bool buffer_size(std::uint64_t elements, UINT stride) {
      return elements > 0u && elements <= (1ull << 27u) && elements * stride <= std::numeric_limits<UINT>::max();
    }

    bool child_ok(ID3D11DeviceChild *child) const {
      if (!child) {
        return false;
      }
      ComPtr<ID3D11Device> device;
      child->GetDevice(&device);
      return device.Get() == device_.Get();
    }

    bool context_ok(ID3D11DeviceContext *context, std::string &error) const {
      if (!context || !device_ || context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE || !child_ok(context)) {
        return fail(error, "Temporal backend requires its own immediate device context.");
      }
      ComPtr<ID3D11Predicate> predicate;
      BOOL value = FALSE;
      context->GetPredication(&predicate, &value);
      if (predicate) {
        return fail(error, "Active caller predication is unsupported; no private commands were issued.");
      }
      return true;
    }

    bool owned_context_ok(ID3D11DeviceContext *context, std::string &error) const {
      if (!context || !device_ || !child_ok(context)) {
        return fail(error, "Owned temporal device/context mismatch.");
      }
      const auto type = context->GetType();
      if (type != D3D11_DEVICE_CONTEXT_IMMEDIATE && type != D3D11_DEVICE_CONTEXT_DEFERRED) {
        return fail(error, "Unsupported owned temporal context.");
      }
      // GetPredication/CSGet is neither legal nor needed on exclusive known-state deferred.
      return true;
    }

    bool frame_ok(const frame_constants_t &frame) const {
      return frame.words[0] == width_ && frame.words[1] == height_ && frame.words[2] == pixels_ && frame.words[3] == 0u &&
             frame.words[12] == caps_.components && frame.words[13] == caps_.members && frame.words[14] == caps_.bit_words && frame.words[15] == caps_.hash_words;
    }

    bool session_ok(ID3D11DeviceContext *context, std::string &error) const {
      return (owned_emission_ ? owned_context_ok(context, error) : context_ok(context, error)) && ((open_ && context_.Get() == context) || fail(error, "No matching open temporal hook session."));
    }

    bool constant_ok(ID3D11Buffer *buffer, UINT bytes) const {
      if (!child_ok(buffer)) {
        return false;
      }
      D3D11_BUFFER_DESC descriptor {};
      buffer->GetDesc(&descriptor);
      return descriptor.ByteWidth == bytes && (descriptor.BindFlags & D3D11_BIND_CONSTANT_BUFFER) != 0u;
    }

    bool arguments_ok(ID3D11Buffer *buffer, UINT offset) const {
      if (!child_ok(buffer)) {
        return false;
      }
      D3D11_BUFFER_DESC descriptor {};
      buffer->GetDesc(&descriptor);
      return (descriptor.MiscFlags & D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS) != 0u && offset % 4u == 0u &&
             offset <= descriptor.ByteWidth && descriptor.ByteWidth - offset >= 12u;
    }

    bool texture_ok(ID3D11ShaderResourceView *view, DXGI_FORMAT format) const {
      if (!child_ok(view)) {
        return false;
      }
      D3D11_SHADER_RESOURCE_VIEW_DESC descriptor {};
      view->GetDesc(&descriptor);
      if (descriptor.ViewDimension != D3D11_SRV_DIMENSION_TEXTURE2D || descriptor.Format != format || descriptor.Texture2D.MostDetailedMip != 0u || descriptor.Texture2D.MipLevels != 1u) {
        return false;
      }
      ComPtr<ID3D11Resource> resource;
      ComPtr<ID3D11Texture2D> texture;
      view->GetResource(&resource);
      if (FAILED(resource.As(&texture))) {
        return false;
      }
      D3D11_TEXTURE2D_DESC actual {};
      texture->GetDesc(&actual);
      return actual.Width == width_ && actual.Height == height_ && actual.ArraySize == 1u && actual.MipLevels == 1u &&
             actual.SampleDesc.Count == 1u && actual.Format == format;
    }

    bool structured_ok(ID3D11ShaderResourceView *view, UINT elements) const {
      if (!child_ok(view)) {
        return false;
      }
      D3D11_SHADER_RESOURCE_VIEW_DESC descriptor {};
      view->GetDesc(&descriptor);
      if (descriptor.ViewDimension != D3D11_SRV_DIMENSION_BUFFER || descriptor.Format != DXGI_FORMAT_UNKNOWN || descriptor.Buffer.FirstElement != 0u || descriptor.Buffer.NumElements != elements) {
        return false;
      }
      ComPtr<ID3D11Resource> resource;
      ComPtr<ID3D11Buffer> buffer;
      view->GetResource(&resource);
      if (FAILED(resource.As(&buffer))) {
        return false;
      }
      D3D11_BUFFER_DESC actual {};
      buffer->GetDesc(&actual);
      return actual.ByteWidth == std::uint64_t {elements} * 4u && actual.StructureByteStride == 4u &&
             (actual.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED) != 0u;
    }

    static void copy_words(ID3D11DeviceContext *context, ID3D11Buffer *destination, UINT destination_offset, ID3D11Buffer *source, UINT source_offset, UINT bytes) {
      const D3D11_BOX box {source_offset, 0u, 0u, source_offset + bytes, 1u, 1u};
      context->CopySubresourceRegion(destination, 0u, destination_offset, 0u, 0u, source, 0u, &box);
    }

    std::array<ID3D11UnorderedAccessView *, 8u> uav_array() const {
      return {state_.uav.Get(), records_.uav.Get(), members_.uav.Get(), bits_.uav.Get(), root_rgb_.uav.Get(), images_.uav.Get(), output_.uav.Get(), receipt_.uav.Get()};
    }

    void bind_stage(ID3D11DeviceContext *context) {
      ID3D11Buffer *constants[3] {frame_constants_.Get(), hook_constants_.Get(), current_constants_.Get()};
      context->CSSetConstantBuffers(0u, 3u, constants);
      context->CSSetShaderResources(0u, UINT(inputs_.size()), inputs_.data());
      const auto uavs = uav_array();
      context->CSSetUnorderedAccessViews(0u, UINT(uavs.size()), uavs.data(), nullptr);
    }

    void set_hook(ID3D11DeviceContext *context, UINT slot, UINT mode) {
      const std::array<UINT, 4u> request {slot, mode, 0u, 0u};
      context->UpdateSubresource(hook_constants_.Get(), 0u, nullptr, request.data(), 0u, 0u);
    }

    void dispatch(ID3D11DeviceContext *context, UINT entry, ID3D11Buffer *arguments, UINT offset) {
      context->CSSetShader(shaders_[entry].Get(), nullptr, 0u);
      context->DispatchIndirect(arguments, offset);
    }

    void release_session() {
      open_ = false;
      owned_emission_ = false;
      frame_prepared_ = false;
      context_.Reset();
      current_constants_.Reset();
      current_rgb_.Reset();
      validation_.Reset();
      root_arguments_.Reset();
      for (auto &shader : shaders_) {
        shader.Reset();
      }
      inputs_.fill(nullptr);
    }

    bool make_structured(std::uint64_t elements, UINT stride, buffer_t &out) {
      if (!buffer_size(elements, stride)) {
        return false;
      }
      D3D11_BUFFER_DESC descriptor {};
      descriptor.ByteWidth = UINT(elements * stride);
      descriptor.Usage = D3D11_USAGE_DEFAULT;
      descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      descriptor.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      descriptor.StructureByteStride = stride;
      D3D11_SHADER_RESOURCE_VIEW_DESC srv {};
      srv.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
      srv.Buffer.NumElements = UINT(elements);
      D3D11_UNORDERED_ACCESS_VIEW_DESC uav {};
      uav.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
      uav.Buffer.NumElements = UINT(elements);
      return SUCCEEDED(device_->CreateBuffer(&descriptor, nullptr, &out.resource)) &&
             SUCCEEDED(device_->CreateShaderResourceView(out.resource.Get(), &srv, &out.srv)) &&
             SUCCEEDED(device_->CreateUnorderedAccessView(out.resource.Get(), &uav, &out.uav));
    }

    bool make_texture(UINT layers, texture_t &out) {
      D3D11_TEXTURE2D_DESC descriptor {};
      descriptor.Width = width_;
      descriptor.Height = height_;
      descriptor.MipLevels = 1u;
      descriptor.ArraySize = layers;
      descriptor.Format = DXGI_FORMAT_R32_UINT;
      descriptor.SampleDesc.Count = 1u;
      descriptor.Usage = D3D11_USAGE_DEFAULT;
      descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      return SUCCEEDED(device_->CreateTexture2D(&descriptor, nullptr, &out.resource)) &&
             SUCCEEDED(device_->CreateShaderResourceView(out.resource.Get(), nullptr, &out.srv)) &&
             SUCCEEDED(device_->CreateUnorderedAccessView(out.resource.Get(), nullptr, &out.uav));
    }

    bool make_constant(UINT bytes, ComPtr<ID3D11Buffer> &out) {
      D3D11_BUFFER_DESC descriptor {};
      descriptor.ByteWidth = bytes;
      descriptor.Usage = D3D11_USAGE_DEFAULT;
      descriptor.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      return SUCCEEDED(device_->CreateBuffer(&descriptor, nullptr, &out));
    }

    bool make_arguments(UINT words, const UINT *values, ComPtr<ID3D11Buffer> &out) {
      D3D11_BUFFER_DESC descriptor {};
      descriptor.ByteWidth = words * 4u;
      descriptor.Usage = D3D11_USAGE_DEFAULT;
      descriptor.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      D3D11_SUBRESOURCE_DATA initial {};
      initial.pSysMem = values;
      return SUCCEEDED(device_->CreateBuffer(&descriptor, &initial, &out));
    }

    bool allocate(std::string &error) {
      const std::uint64_t record_words = 2ull * (512ull + 64ull * caps_.components + caps_.hash_words);
      if (!make_structured(512u, 4u, state_) || !make_structured(record_words, 4u, records_) || !make_structured(2ull * caps_.members, 8u, members_) || !make_structured(2ull * caps_.bit_words, 4u, bits_) || !make_structured(5ull * pixels_ + 1u, 4u, root_rgb_) || !make_texture(9u, images_) || !make_texture(2u, output_) || !make_texture(1u, body_) || !make_texture(1u, matte_) || !make_constant(256u, frame_constants_) || !make_constant(16u, hook_constants_)) {
        return fail(error, "Could not allocate exact temporal storage; no partial backend is usable.");
      }
      D3D11_BUFFER_DESC descriptor {};
      descriptor.ByteWidth = 1024u;
      descriptor.Usage = D3D11_USAGE_DEFAULT;
      descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      descriptor.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS | D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      D3D11_SHADER_RESOURCE_VIEW_DESC srv {};
      srv.Format = DXGI_FORMAT_R32_TYPELESS;
      srv.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
      srv.BufferEx.NumElements = 256u;
      srv.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
      D3D11_UNORDERED_ACCESS_VIEW_DESC uav {};
      uav.Format = DXGI_FORMAT_R32_TYPELESS;
      uav.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
      uav.Buffer.NumElements = 256u;
      uav.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
      if (FAILED(device_->CreateBuffer(&descriptor, nullptr, &receipt_.resource)) || FAILED(device_->CreateShaderResourceView(receipt_.resource.Get(), &srv, &receipt_.srv)) || FAILED(device_->CreateUnorderedAccessView(receipt_.resource.Get(), &uav, &receipt_.uav))) {
        return fail(error, "Could not allocate1024B temporal receipt/arguments.");
      }
      const std::array<UINT, 3u> native {(width_ + 15u) / 16u, (height_ + 15u) / 16u, 0u};
      const std::array<UINT, 6u> zero {};
      if (!make_arguments(3u, native.data(), native_arguments_) || !make_arguments(3u, zero.data(), hook_arguments_) || !make_arguments(6u, zero.data(), end_arguments_)) {
        return fail(error, "Could not allocate distinct temporal GPU indirect arguments.");
      }
      return true;
    }

    ComPtr<ID3D11Device> device_;
    UINT width_ = 0u, height_ = 0u, pixels_ = 0u;
    caps_t caps_;
    buffer_t state_, records_, members_, bits_, root_rgb_, receipt_;
    texture_t images_, output_, body_, matte_;
    ComPtr<ID3D11Buffer> frame_constants_, hook_constants_, native_arguments_, hook_arguments_, end_arguments_;
    bool needs_initialize_ = true;
    bool open_ = false;
    bool poisoned_ = false;
    bool owned_emission_ = false;
    bool frame_prepared_ = false;
    UINT next_unit_ = 0u, next_mode_ = 8u, root_offset_ = 0u;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<ID3D11Buffer> current_constants_, root_arguments_;
    ComPtr<ID3D11ShaderResourceView> current_rgb_, validation_;
    std::array<ComPtr<ID3D11ComputeShader>, 19u> shaders_;
    std::array<ID3D11ShaderResourceView *, 8u> inputs_ {};
  };
}  // namespace models::host_sbs_body_witness_temporal_gpu

#endif
