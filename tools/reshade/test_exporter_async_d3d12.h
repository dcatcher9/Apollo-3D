// SPDX-License-Identifier: GPL-3.0-only
#pragma once

// Native D3D12 transport for the opt-in publisher ring test. Only the two
// ReShade operations used by production generation_t::submit are implemented.
namespace exporter_async_fixture {
  using namespace reshade::api;
  inline void check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
  inline constexpr unsigned width = 64, height = 32;

  inline void transition(ID3D12GraphicsCommandList *commands, ID3D12Resource *resource,
      D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    commands->ResourceBarrier(1, &barrier);
  }

  struct native_commands final : command_list {
    ID3D12GraphicsCommandList *native{};
    uint64_t get_native() const override { return reinterpret_cast<uint64_t>(native); }
    static D3D12_RESOURCE_STATES state(resource_usage usage) {
      switch (usage) {
        case resource_usage::general: return D3D12_RESOURCE_STATE_COMMON;
        case resource_usage::shader_resource: return D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        case resource_usage::copy_source: return D3D12_RESOURCE_STATE_COPY_SOURCE;
        case resource_usage::copy_dest: return D3D12_RESOURCE_STATE_COPY_DEST;
        default: throw std::runtime_error("Unexpected production copy barrier state");
      }
    }
    void barrier(uint32_t count, const resource *resources, const resource_usage *old_states, const resource_usage *new_states) override {
      for (uint32_t i = 0; i != count; ++i)
        transition(native, reinterpret_cast<ID3D12Resource *>(resources[i].handle), state(old_states[i]), state(new_states[i]));
    }
    void copy_resource(resource source, resource dest) override {
      native->CopyResource(reinterpret_cast<ID3D12Resource *>(dest.handle), reinterpret_cast<ID3D12Resource *>(source.handle));
    }
#define UNEXPECTED_COMMAND(...) __VA_ARGS__ override { throw std::runtime_error("Unexpected native copy adapter call: " #__VA_ARGS__); }
    UNEXPECTED_COMMAND(void get_private_data(const uint8_t[16], uint64_t *) const)
    UNEXPECTED_COMMAND(void set_private_data(const uint8_t[16], uint64_t))
    UNEXPECTED_COMMAND(device *get_device())
    UNEXPECTED_COMMAND(void end_render_pass())
    UNEXPECTED_COMMAND(void bind_render_targets_and_depth_stencil(uint32_t count, const resource_view *rtvs, resource_view dsv))
    UNEXPECTED_COMMAND(void bind_pipeline(pipeline_stage stages, pipeline pipeline))
    UNEXPECTED_COMMAND(void bind_pipeline_states(uint32_t count, const dynamic_state *states, const uint32_t *values))
    UNEXPECTED_COMMAND(void bind_viewports(uint32_t first, uint32_t count, const viewport *viewports))
    UNEXPECTED_COMMAND(void bind_scissor_rects(uint32_t first, uint32_t count, const rect *rects))
    UNEXPECTED_COMMAND(void push_constants(shader_stage stages, pipeline_layout layout, uint32_t param, uint32_t first, uint32_t count, const void *values))
    UNEXPECTED_COMMAND(void push_descriptors(shader_stage stages, pipeline_layout layout, uint32_t param, const descriptor_table_update &update))
    UNEXPECTED_COMMAND(void bind_index_buffer(resource buffer, uint64_t offset, uint32_t index_size))
    UNEXPECTED_COMMAND(void bind_vertex_buffers(uint32_t first, uint32_t count, const resource *buffers, const uint64_t *offsets, const uint32_t *strides))
    UNEXPECTED_COMMAND(void bind_stream_output_buffers(uint32_t first, uint32_t count, const resource *buffers, const uint64_t *offsets, const uint64_t *max_sizes, const resource *counter_buffers, const uint64_t *counter_offsets))
    UNEXPECTED_COMMAND(void draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance))
    UNEXPECTED_COMMAND(void draw_indexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t vertex_offset, uint32_t first_instance))
    UNEXPECTED_COMMAND(void dispatch(uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z))
    UNEXPECTED_COMMAND(void draw_or_dispatch_indirect(indirect_command type, resource buffer, uint64_t offset, uint32_t draw_count, uint32_t stride))
    UNEXPECTED_COMMAND(void copy_buffer_region(resource source, uint64_t source_offset, resource dest, uint64_t dest_offset, uint64_t size))
    UNEXPECTED_COMMAND(void copy_buffer_to_texture(resource source, uint64_t source_offset, uint32_t row_length, uint32_t slice_height, resource dest, uint32_t dest_subresource, const subresource_box *dest_box))
    UNEXPECTED_COMMAND(void copy_texture_region(resource source, uint32_t source_subresource, const subresource_box *source_box, resource dest, uint32_t dest_subresource, const subresource_box *dest_box, filter_mode filter))
    UNEXPECTED_COMMAND(void copy_texture_to_buffer(resource source, uint32_t source_subresource, const subresource_box *source_box, resource dest, uint64_t dest_offset, uint32_t row_length, uint32_t slice_height))
    UNEXPECTED_COMMAND(void resolve_texture_region(resource source, uint32_t source_subresource, const subresource_box *source_box, resource dest, uint32_t dest_subresource, uint32_t dest_x, uint32_t dest_y, uint32_t dest_z, format format))
    UNEXPECTED_COMMAND(void clear_depth_stencil_view(resource_view dsv, const float *depth, const uint8_t *stencil, uint32_t rect_count, const rect *rects))
    UNEXPECTED_COMMAND(void clear_render_target_view(resource_view rtv, const float color[4], uint32_t rect_count, const rect *rects))
    UNEXPECTED_COMMAND(void clear_unordered_access_view_uint(resource_view uav, const uint32_t values[4], uint32_t rect_count, const rect *rects))
    UNEXPECTED_COMMAND(void clear_unordered_access_view_float(resource_view uav, const float values[4], uint32_t rect_count, const rect *rects))
    UNEXPECTED_COMMAND(void generate_mipmaps(resource_view srv))
    UNEXPECTED_COMMAND(void begin_query(query_heap heap, query_type type, uint32_t index))
    UNEXPECTED_COMMAND(void end_query(query_heap heap, query_type type, uint32_t index))
    UNEXPECTED_COMMAND(void copy_query_heap_results(query_heap heap, query_type type, uint32_t first, uint32_t count, resource dest, uint64_t dest_offset, uint32_t stride))
    UNEXPECTED_COMMAND(void begin_debug_event(const char *label, const float color[4]))
    UNEXPECTED_COMMAND(void end_debug_event())
    UNEXPECTED_COMMAND(void insert_debug_marker(const char *label, const float color[4]))
    UNEXPECTED_COMMAND(void dispatch_mesh(uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z))
    UNEXPECTED_COMMAND(void dispatch_rays(resource raygen, uint64_t raygen_offset, uint64_t raygen_size, resource miss, uint64_t miss_offset, uint64_t miss_size, uint64_t miss_stride, resource hit_group, uint64_t hit_group_offset, uint64_t hit_group_size, uint64_t hit_group_stride, resource callable, uint64_t callable_offset, uint64_t callable_size, uint64_t callable_stride, uint32_t width, uint32_t height, uint32_t depth))
    UNEXPECTED_COMMAND(void copy_acceleration_structure(resource_view source, resource_view dest, acceleration_structure_copy_mode mode))
    UNEXPECTED_COMMAND(void build_acceleration_structure(acceleration_structure_type type, acceleration_structure_build_flags flags, uint32_t input_count, const acceleration_structure_build_input *inputs, resource scratch, uint64_t scratch_offset, resource_view source, resource_view dest, acceleration_structure_build_mode mode))
    UNEXPECTED_COMMAND(void query_acceleration_structures(uint32_t count, const resource_view *acceleration_structures, query_heap heap, query_type type, uint32_t first))
    UNEXPECTED_COMMAND(void update_buffer_region(const void *data, resource dest, uint64_t dest_offset, uint64_t size))
    UNEXPECTED_COMMAND(void update_texture_region(const subresource_data &data, resource dest, uint32_t dest_subresource, const subresource_box *dest_box))
    UNEXPECTED_COMMAND(void begin_render_pass2(uint32_t count, const render_pass_render_target_desc *rts, const render_pass_depth_stencil_desc *ds, render_pass_flags flags))
    UNEXPECTED_COMMAND(void bind_descriptor_tables2(shader_stage stages, pipeline_layout layout, uint32_t first, uint32_t count, const descriptor_table *tables, uint32_t dynamic_offset_count, const uint32_t *dynamic_offsets))
#undef UNEXPECTED_COMMAND
  };

  struct queued_copy {
    com_ptr<ID3D12CommandAllocator> allocator;
    com_ptr<ID3D12GraphicsCommandList> commands;
    com_ptr<ID3D12Resource> upload, readback;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
  };

  struct gpu_t {
    com_ptr<ID3D12Device> device;
    com_ptr<ID3D12CommandQueue> queue;
    com_ptr<ID3D12Fence> gate, drained;
    std::array<queued_copy, 4> copies;
    bool waiting{};

    gpu_t() {
      check(SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.put()))), "Async exporter D3D12 device creation failed");
      D3D12_COMMAND_QUEUE_DESC desc{};
      check(SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(queue.put()))), "Async exporter queue creation failed");
      check(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(gate.put()))) &&
        SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(drained.put()))), "Async exporter gate creation failed");
    }

    ~gpu_t() {
      // Even an assertion failure must unblock recorded work before COM owners
      // are destroyed. A separate process watchdog bounds a broken driver.
      if (waiting) gate->Signal(1);
      if (queue && SUCCEEDED(queue->Signal(drained.get(), 1))) {
        const auto deadline = GetTickCount64() + 2000;
        while (drained->GetCompletedValue() < 1 && GetTickCount64() < deadline) Sleep(1);
        if (drained->GetCompletedValue() < 1) {
          std::fputs("FAIL async exporter cleanup did not drain within two seconds\n", stderr);
          TerminateProcess(GetCurrentProcess(), 3);
        }
      }
    }

    void hold() {
      check(SUCCEEDED(queue->Wait(gate.get(), 1)), "Could not hold async exporter GPU queue");
      waiting = true;
    }

    void release() {
      check(SUCCEEDED(gate->Signal(1)), "Could not release async exporter GPU queue");
      waiting = false;
    }

    static std::uint32_t pixel(unsigned pattern, unsigned x, unsigned y) {
      return 0xff000000u | ((17u + pattern * 39u) << 16) | ((y * 7u & 255u) << 8) | (x * 3u & 255u);
    }

    com_ptr<ID3D12Resource> prepare(unsigned index) {
      auto &copy = copies[index];
      check(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(copy.allocator.put()))) &&
        SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, copy.allocator.get(), nullptr, IID_PPV_ARGS(copy.commands.put()))),
        "Could not create independent async copy recording");
      D3D12_RESOURCE_DESC desc{};
      desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      desc.Width = width; desc.Height = height;
      desc.DepthOrArraySize = desc.MipLevels = 1;
      desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
      desc.SampleDesc.Count = 1;
      D3D12_HEAP_PROPERTIES heap{};
      heap.Type = D3D12_HEAP_TYPE_DEFAULT;
      com_ptr<ID3D12Resource> source;
      check(SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(source.put()))),
        "Could not allocate independent async copy source");
      UINT64 bytes{};
      device->GetCopyableFootprints(&desc, 0, 1, 0, &copy.footprint, nullptr, nullptr, &bytes);
      desc = {};
      desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
      desc.Width = bytes; desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
      desc.SampleDesc.Count = 1; desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
      heap.Type = D3D12_HEAP_TYPE_UPLOAD;
      check(SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(copy.upload.put()))),
        "Could not allocate async pixel upload");
      heap.Type = D3D12_HEAP_TYPE_READBACK;
      check(SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(copy.readback.put()))),
        "Could not allocate async pixel readback");
      void *mapped{};
      D3D12_RANGE none{};
      check(SUCCEEDED(copy.upload->Map(0, &none, &mapped)), "Could not map async pixel upload");
      for (unsigned y = 0; y != height; ++y) {
        auto *row = reinterpret_cast<std::uint32_t *>(static_cast<std::uint8_t *>(mapped) + y * copy.footprint.Footprint.RowPitch);
        for (unsigned x = 0; x != width; ++x) row[x] = pixel(index, x, y);
      }
      copy.upload->Unmap(0, nullptr);
      D3D12_TEXTURE_COPY_LOCATION input{};
      input.pResource = copy.upload.get(); input.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; input.PlacedFootprint = copy.footprint;
      D3D12_TEXTURE_COPY_LOCATION output{};
      output.pResource = source.get(); output.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      copy.commands->CopyTextureRegion(&output, 0, 0, 0, &input, nullptr);
      transition(copy.commands.get(), source.get(), D3D12_RESOURCE_STATE_COPY_DEST, native_commands::state(resource_usage::shader_resource));
      return source;
    }

    void submit_and_read(unsigned index, ID3D12Resource *exported) {
      auto &copy = copies[index];
      transition(copy.commands.get(), exported, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
      D3D12_TEXTURE_COPY_LOCATION input{};
      input.pResource = exported; input.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      D3D12_TEXTURE_COPY_LOCATION output{};
      output.pResource = copy.readback.get(); output.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; output.PlacedFootprint = copy.footprint;
      copy.commands->CopyTextureRegion(&output, 0, 0, 0, &input, nullptr);
      transition(copy.commands.get(), exported, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
      check(SUCCEEDED(copy.commands->Close()), "Could not close async copy recording");
      ID3D12CommandList *lists[]{copy.commands.get()};
      queue->ExecuteCommandLists(1, lists);
    }

    void check_pixels(unsigned index) {
      auto &copy = copies[index];
      void *mapped{};
      check(SUCCEEDED(copy.readback->Map(0, nullptr, &mapped)), "Could not map completed async copy pixels");
      bool valid = true;
      for (unsigned y = 0; y != height; ++y) {
        const auto *row = reinterpret_cast<const std::uint32_t *>(static_cast<const std::uint8_t *>(mapped) + y * copy.footprint.Footprint.RowPitch);
        for (unsigned x = 0; x != width; ++x) valid = valid && row[x] == pixel(index, x, y);
      }
      D3D12_RANGE none{};
      copy.readback->Unmap(0, &none);
      check(valid, "An asynchronous exporter slot contained another frame's pixels");
    }
  };
}
