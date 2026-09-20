// SPDX-License-Identifier: GPL-3.0-only
// Opt-in GPU regression using the production capture owner and native observer.
// This models a legal game queue dependency, not an already-deadlocked game:
// producer waits for a fence the consumer will signal after its current effects
// callback. Only the add-on's reverse Wait can turn that ordering into a cycle.
// Every path releases the game's fence from the CPU before bounded GPU cleanup.
#include "streamline_depth_capture.h"
#include "streamline_native_observer.h"

#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
  namespace capture = sunshine_streamline::depth_capture;
  namespace observer = sunshine_streamline::native_observer;
  template<class T> struct com_ptr {
    T *value{};
    ~com_ptr() { if (value) value->Release(); }
    T *operator->() const { return value; }
    T **put() { return &value; }
    std::uint64_t native() const { return reinterpret_cast<std::uint64_t>(value); }
  };
  void require(bool okay, const char *message) {
    if (!okay) throw std::runtime_error(message);
  }
  void checked(HRESULT value, const char *message) {
    if (FAILED(value)) {
      char text[256]{};
      std::snprintf(text, sizeof(text), "%s: HRESULT=0x%08lx", message, static_cast<unsigned long>(value));
      throw std::runtime_error(text);
    }
  }
  bool wait_until(ID3D12Fence *fence, UINT64 value, DWORD milliseconds) {
    const auto start = GetTickCount64();
    do {
      const auto completed = fence->GetCompletedValue();
      if (completed == UINT64_MAX) throw std::runtime_error("D3D12 device was removed");
      if (completed >= value) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (GetTickCount64() - start < milliseconds);
    return false;
  }

  struct fixture {
    com_ptr<ID3D12Device> device;
    com_ptr<ID3D12CommandQueue> producer, consumer;
    com_ptr<ID3D12Fence> game_gate, producer_done, consumer_done;
    com_ptr<ID3D12CommandAllocator> producer_allocator, producer_next_allocator, consumer_allocator;
    com_ptr<ID3D12GraphicsCommandList> producer_commands, consumer_commands;
    com_ptr<ID3D12Resource> source, destination;
    UINT64 rescue_gate_value{1};
    bool capture_initialized{}, drained{};

    fixture(unsigned width = 64, unsigned height = 64, bool packed = false) {
      checked(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_ID3D12Device,
        reinterpret_cast<void **>(device.put())), "Create hardware D3D12 device");
      D3D12_COMMAND_QUEUE_DESC queue{};
      queue.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
      checked(device->CreateCommandQueue(&queue, IID_ID3D12CommandQueue,
        reinterpret_cast<void **>(producer.put())), "Create producer queue");
      checked(device->CreateCommandQueue(&queue, IID_ID3D12CommandQueue,
        reinterpret_cast<void **>(consumer.put())), "Create consumer queue");
      checked(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_ID3D12Fence,
        reinterpret_cast<void **>(game_gate.put())), "Create game ordering fence");
      checked(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_ID3D12Fence,
        reinterpret_cast<void **>(producer_done.put())), "Create producer completion fence");
      checked(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_ID3D12Fence,
        reinterpret_cast<void **>(consumer_done.put())), "Create consumer completion fence");
      for (auto *allocator : {&producer_allocator, &producer_next_allocator, &consumer_allocator})
        checked(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_ID3D12CommandAllocator,
          reinterpret_cast<void **>(allocator->put())), "Create command allocator");
      checked(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, producer_allocator.value,
        nullptr, IID_ID3D12GraphicsCommandList, reinterpret_cast<void **>(producer_commands.put())), "Create producer recording");
      checked(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, consumer_allocator.value,
        nullptr, IID_ID3D12GraphicsCommandList, reinterpret_cast<void **>(consumer_commands.put())), "Create consumer recording");
      D3D12_HEAP_PROPERTIES heap{};
      heap.Type = D3D12_HEAP_TYPE_DEFAULT;
      D3D12_RESOURCE_DESC texture{};
      texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      texture.Width = width;
      texture.Height = height;
      texture.DepthOrArraySize = texture.MipLevels = texture.SampleDesc.Count = 1;
      texture.Format = packed ? DXGI_FORMAT_R32G8X24_TYPELESS : DXGI_FORMAT_R32_FLOAT;
      texture.Flags = packed ? D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL : D3D12_RESOURCE_FLAG_NONE;
      texture.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
      checked(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture,
        packed ? D3D12_RESOURCE_STATE_DEPTH_WRITE : D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr, IID_ID3D12Resource,
        reinterpret_cast<void **>(source.put())), "Create source depth texture");
      checked(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture,
        packed ? D3D12_RESOURCE_STATE_DEPTH_WRITE : D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_ID3D12Resource,
        reinterpret_cast<void **>(destination.put())), "Create consumer depth texture");
      capture::initialize(true);
      capture_initialized = true;
      capture::observe_queue(producer.native());
      capture::observe_queue(consumer.native());
      capture::observe_command(producer_commands.native());
      capture::observe_command(consumer_commands.native());
      capture::poll();
      require(observer::queue_ready(producer.native()) && observer::queue_ready(consumer.native()) &&
        observer::command_ready(producer_commands.native()) && observer::command_ready(consumer_commands.native()),
        "Production native observer did not establish complete queue/command coverage");
      // Installing coverage invalidates pre-install recordings. Begin fresh,
      // fully observed recordings exactly as a game does at its next Reset.
      checked(producer_commands->Close(), "Close pre-observer producer recording");
      checked(consumer_commands->Close(), "Close pre-observer consumer recording");
      checked(producer_commands->Reset(producer_allocator.value, nullptr), "Start observed producer recording");
      checked(consumer_commands->Reset(consumer_allocator.value, nullptr), "Start observed consumer recording");
    }

    void drain() {
      if (drained) return;
      // Signal on the CPU, never by queuing more work behind the suspect cycle.
      checked(game_gate->Signal(rescue_gate_value), "Release game fence from CPU for cleanup");
      checked(producer->Signal(producer_done.value, 100), "Fence producer cleanup");
      checked(consumer->Signal(consumer_done.value, 100), "Fence consumer cleanup");
      require(wait_until(producer_done.value, 100, 5000) && wait_until(consumer_done.value, 100, 5000),
        "GPU cleanup did not finish within its bounded deadline");
      drained = true;
    }

    ~fixture() {
      try { drain(); }
      catch (const std::exception &error) {
        std::fprintf(stderr, "FATAL bounded cleanup: %s\n", error.what());
        std::fflush(stderr);
        // Do not enter potentially blocking driver destruction after cleanup
        // failed. This is this fixture process only, never an attached game.
        std::_Exit(2);
      }
      if (capture_initialized) {
        // Production shutdown intentionally preserves queue owners whose GPU
        // work may still be live. This fixture has proved completion above;
        // report command/queue destruction just as the game integration does,
        // or sequential cases consume the owner's bounded four-queue registry.
        capture::command_destroyed(producer_commands.native());
        capture::command_destroyed(consumer_commands.native());
        capture::retire_queue(producer.native());
        capture::retire_queue(consumer.native());
        capture::poll();
        capture::shutdown();
      }
    }
  };

  void run(bool expect_cycle) {
    fixture gpu;
    capture::input input;
    input.epoch = input.sequence = 1;
    input.tick = GetTickCount64();
    input.viewport = 1;
    input.resource = {gpu.source.native(), 64, 64, {0, 0, 64, 64}};
    input.native_state = D3D12_RESOURCE_STATE_COPY_SOURCE;
    input.proof = sunshine_scene_depth::state_proof::declared;
    input.valid_until = sunshine_scene_depth::lifetime::until_present;
    input.source = capture::retain_source(gpu.source.native());
    input.source_present_generation = capture::source_present_generation(input.source);
    input.force_snapshot = true;
    require(bool(input.source), "Production owner failed to retain source identity");

    // The application has submitted independent producer work gated on a later
    // consumer signal. This is acyclic before the add-on intervenes.
    checked(gpu.producer->Wait(gpu.game_gate.value, 1), "Queue game producer dependency");
    capture::record_diagnostic recorded;
    const auto ticket = capture::nominate_evaluation(gpu.producer_commands.native(), input, UINT64_MAX, &recorded);
    std::printf("record: ticket=%llu status=%s stage=%s\n", static_cast<unsigned long long>(ticket),
      capture::name(recorded.result), capture::name(recorded.stage));
    require(ticket && recorded.result == capture::status::recorded, "Production capture rejected the legal producer copy");
    capture::finish(ticket, true);
    checked(gpu.producer_commands->Close(), "Close producer copy recording");
    ID3D12CommandList *producer_lists[]{gpu.producer_commands.value};
    gpu.producer->ExecuteCommandLists(1, producer_lists);
    // Resetting the list onto a DIFFERENT allocator is legal while the previous
    // submission runs; that previous allocator is kept alive and never reset.
    checked(gpu.producer_commands->Reset(gpu.producer_next_allocator.value, nullptr),
      "Retire producer recording without resetting its in-flight allocator");

    // Earlier consumer work is empty/already submitted. This meets the provider's
    // flush-before-Wait contract, without a synthetic unsubmitted command list.
    capture::packet packet;
    capture::capture_diagnostic acquired;
    const bool metadata = capture::acquire(gpu.consumer.native(), 1, packet, &acquired);
    require(acquired.submitted && acquired.success && acquired.producer_recording_retired &&
      acquired.producer_completion_valid && acquired.producer_completed < acquired.producer_fence,
      "Fixture did not reach a valid immutable-but-pending foreign-queue snapshot");
    std::printf("acquire: metadata=%d pixel_ready=%d status=%s producer=%llu completed=%llu retired=%d\n",
      int(metadata), int(packet.pixel_ready), capture::name(acquired.result),
      static_cast<unsigned long long>(acquired.producer_fence),
      static_cast<unsigned long long>(acquired.producer_completed), int(acquired.producer_recording_retired));

    bool copied = false;
    if (metadata && packet.pixel_ready) {
      capture::consumer_diagnostic consumer_result;
      copied = capture::copy_current(gpu.consumer_commands.native(), packet, gpu.destination.native(),
        D3D12_RESOURCE_STATE_COPY_DEST, &consumer_result);
      std::printf("copy: accepted=%d status=%s\n", int(copied), capture::name(consumer_result.result));
    }
    checked(gpu.consumer_commands->Close(), "Close consumer recording");
    ID3D12CommandList *consumer_lists[]{gpu.consumer_commands.value};
    gpu.consumer->ExecuteCommandLists(1, consumer_lists);
    checked(gpu.consumer->Signal(gpu.game_gate.value, 1), "Queue application's later consumer signal");
    checked(gpu.consumer->Signal(gpu.consumer_done.value, 1), "Fence application consumer progress");
    checked(gpu.producer->Signal(gpu.producer_done.value, 1), "Fence application producer progress");

    const bool consumer_progressed = wait_until(gpu.consumer_done.value, 1, 250);
    const bool producer_progressed = gpu.producer_done->GetCompletedValue() >= 1;
    std::printf("before CPU rescue: consumer_progressed=%d producer_progressed=%d copied=%d\n",
      int(consumer_progressed), int(producer_progressed), int(copied));
    if (expect_cycle) {
      gpu.drain();
      require(copied && !consumer_progressed && !producer_progressed,
        "The production owner did not reproduce the expected injected queue cycle");
      std::puts("PASS: production consumer Wait reproduced bounded queue cycle; CPU rescue drained both queues");
    } else {
      require(!copied && consumer_progressed,
        "Production capture injected a GPU dependency cycle into an acyclic application schedule");
      require(wait_until(gpu.producer_done.value, 1, 2000), "Original producer did not finish after natural consumer signal");
      // The deferred exact frame remains eligible when its immutable copy is
      // complete. Retry before the ordinary 250 ms freshness window expires;
      // neither metadata timestamps nor evaluation sequence are refreshed here.
      checked(gpu.consumer_allocator->Reset(), "Reset completed consumer allocator");
      checked(gpu.consumer_commands->Reset(gpu.consumer_allocator.value, nullptr), "Start completion retry recording");
      capture::packet recovered;
      capture::capture_diagnostic recovered_status;
      require(capture::acquire(gpu.consumer.native(), 2, recovered, &recovered_status) && recovered.pixel_ready &&
        recovered.capture_id == ticket && recovered.metadata.sequence == input.sequence,
        "Exact deferred capture did not recover after natural producer completion");
      capture::consumer_diagnostic recovered_copy;
      require(capture::copy_current(gpu.consumer_commands.native(), recovered, gpu.destination.native(),
        D3D12_RESOURCE_STATE_COPY_DEST, &recovered_copy), "Completed foreign-queue copy was rejected");
      capture::complete_frame(recovered, 2);
      checked(gpu.consumer_commands->Close(), "Close recovered consumer recording");
      gpu.consumer->ExecuteCommandLists(1, consumer_lists);
      checked(gpu.consumer->Signal(gpu.consumer_done.value, 3), "Fence completed-depth recovery");
      require(wait_until(gpu.consumer_done.value, 3, 5000), "Recovered depth copy failed to complete");
      std::printf("recovery: same_capture=%llu sequence=%llu pixel_ready=1 producer_completed=%llu\n",
        static_cast<unsigned long long>(recovered.capture_id),
        static_cast<unsigned long long>(recovered.metadata.sequence),
        static_cast<unsigned long long>(recovered_status.producer_completed));

      // Existing same-queue ordering must remain usable without waiting for the
      // CPU to observe producer completion. The new producer recording has been
      // open since the legitimate Reset above and its allocator remains alive.
      gpu.rescue_gate_value = 2;
      checked(gpu.producer->Wait(gpu.game_gate.value, 2), "Gate same-queue producer fixture");
      input.sequence = 2;
      input.tick = GetTickCount64();
      const auto same_ticket = capture::nominate_evaluation(gpu.producer_commands.native(), input,
        UINT64_MAX, &recorded);
      require(same_ticket && recorded.result == capture::status::recorded, "Record same-queue source");
      capture::finish(same_ticket, true);
      checked(gpu.producer_commands->Close(), "Close same-queue producer recording");
      gpu.producer->ExecuteCommandLists(1, producer_lists);
      capture::packet same_queue;
      capture::capture_diagnostic same_status;
      require(capture::acquire(gpu.producer.native(), 3, same_queue, &same_status) && same_queue.pixel_ready &&
        same_queue.capture_id == same_ticket && same_status.producer_completed < same_status.producer_fence,
        "Same-queue capture unnecessarily required CPU-visible producer completion");
      checked(gpu.consumer_allocator->Reset(), "Reset completed consumer allocator for same-queue read");
      checked(gpu.consumer_commands->Reset(gpu.consumer_allocator.value, nullptr), "Start same-queue consumer recording");
      require(capture::copy_current(gpu.consumer_commands.native(), same_queue, gpu.destination.native(),
        D3D12_RESOURCE_STATE_COPY_DEST), "Same-queue pending copy rejected");
      capture::complete_frame(same_queue, 3);
      checked(gpu.consumer_commands->Close(), "Close same-queue consumer recording");
      gpu.producer->ExecuteCommandLists(1, consumer_lists);
      checked(gpu.producer->Signal(gpu.producer_done.value, 4), "Fence same-queue copy");
      checked(gpu.game_gate->Signal(2), "Release intentional same-queue test gate");
      require(wait_until(gpu.producer_done.value, 4, 5000), "Same-queue copy failed to complete");
      std::puts("same-queue: pending producer accepted through queue ordering and copy completed");
      gpu.drain();
      std::puts("PASS: pending foreign producer was deferred; original application ordering completed without rescue");
    }
  }

  enum class pipeline_case {
    recorded, submitted, failed, missing, observation_reset, feedback_reset,
    reset_flag, epoch_change, layout_change
  };

  const char *name(pipeline_case value) {
    switch (value) {
      case pipeline_case::recorded: return "CPU-recorded successor";
      case pipeline_case::submitted: return "pending foreign-queue successor";
      case pipeline_case::failed: return "failed successor";
      case pipeline_case::missing: return "missing successor";
      case pipeline_case::observation_reset: return "observation revision changed";
      case pipeline_case::feedback_reset: return "feedback revision changed";
      case pipeline_case::reset_flag: return "explicit reset";
      case pipeline_case::epoch_change: return "provider epoch changed";
      case pipeline_case::layout_change: return "active depth crop changed";
    }
    return "unknown";
  }

  void run_pipeline(pipeline_case scenario) {
    // Each case owns its queues and observer lifetime. Extra allocators outlive
    // the fixture's bounded drain, including the deliberately blocked submit.
    com_ptr<ID3D12CommandAllocator> newest_allocator, retired_allocator;
    fixture gpu;
    for (auto *allocator : {&newest_allocator, &retired_allocator})
      checked(gpu.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_ID3D12CommandAllocator,
        reinterpret_cast<void **>(allocator->put())), "Create pipeline allocator");

    capture::input input;
    input.provider = sunshine_scene_depth::provider_kind::ngx;
    input.epoch = 47;
    input.sequence = 1;
    input.source_id = 51;
    input.viewport = 1;
    input.observation_revision = 7;
    input.feedback.revision = 11;
    input.resource = {gpu.source.native(), 64, 64, {0, 0, 64, 64}};
    input.native_state = D3D12_RESOURCE_STATE_COPY_SOURCE;
    input.proof = sunshine_scene_depth::state_proof::declared;
    input.valid_until = sunshine_scene_depth::lifetime::until_present;
    input.source = capture::retain_source(gpu.source.native());
    input.source_present_generation = capture::source_present_generation(input.source);
    require(bool(input.source), "Retain pipeline source");

    const auto record = [&](bool successful = true) {
      input.tick = GetTickCount64();
      capture::record_diagnostic diagnostic;
      const auto ticket = capture::nominate_evaluation(gpu.producer_commands.native(), input, UINT64_MAX, &diagnostic);
      require(ticket && diagnostic.result == capture::status::recorded, "Record pipeline snapshot");
      capture::finish(ticket, successful);
      return ticket;
    };
    const auto submit = [&](UINT64 completion, ID3D12CommandAllocator *next_allocator, bool wait) {
      checked(gpu.producer_commands->Close(), "Close pipeline producer");
      ID3D12CommandList *lists[]{gpu.producer_commands.value};
      gpu.producer->ExecuteCommandLists(1, lists);
      checked(gpu.producer->Signal(gpu.producer_done.value, completion), "Fence pipeline producer");
      checked(gpu.producer_commands->Reset(next_allocator, nullptr), "Retire pipeline producer recording");
      if (wait) require(wait_until(gpu.producer_done.value, completion, 5000), "Pipeline producer did not finish");
    };
    const auto consume = [&](const capture::packet &packet, UINT64 present) {
      require(capture::copy_current(gpu.consumer_commands.native(), packet, gpu.destination.native(),
        D3D12_RESOURCE_STATE_COPY_DEST), "Copy completed pipeline snapshot");
      capture::complete_frame(packet, present);
      checked(gpu.consumer_commands->Close(), "Close pipeline consumer");
      ID3D12CommandList *lists[]{gpu.consumer_commands.value};
      gpu.consumer->ExecuteCommandLists(1, lists);
      checked(gpu.consumer->Signal(gpu.consumer_done.value, present), "Fence pipeline consumer");
      require(wait_until(gpu.consumer_done.value, present, 2000), "Completed snapshot injected a queue wait");
      checked(gpu.consumer_allocator->Reset(), "Reset finished pipeline consumer allocator");
      checked(gpu.consumer_commands->Reset(gpu.consumer_allocator.value, nullptr), "Restart pipeline consumer");
    };

    const auto first_ticket = record();
    submit(1, gpu.producer_next_allocator.value, true);
    capture::packet first;
    capture::capture_diagnostic first_diagnostic;
    const bool first_acquired = capture::acquire(gpu.consumer.native(), 1, first, &first_diagnostic);
    std::printf("pipeline %s seed: selected=%llu ready=%d status=%s failure=%s\n", name(scenario),
      static_cast<unsigned long long>(first.metadata.sequence), int(first.pixel_ready),
      capture::name(first_diagnostic.result), capture::name(first_diagnostic.failure));
    require(first_acquired && first.pixel_ready &&
      !first.shared_preservation && first.capture_id == first_ticket, "Establish NGX pipeline with frame one");
    consume(first, 1);
    first = {};

    // Complete a second real frame without presenting it. When the next API
    // evaluation arrives, that immutable copy must survive pool reuse and be
    // available to effects while the CPU/GPU is still working on frame three.
    input.sequence = 2;
    const auto completed_ticket = record();
    submit(2, newest_allocator.value, true);
    input.sequence = 3;
    if (scenario == pipeline_case::observation_reset) ++input.observation_revision;
    if (scenario == pipeline_case::feedback_reset) ++input.feedback.revision;
    if (scenario == pipeline_case::reset_flag) input.feedback.reset = true;
    if (scenario == pipeline_case::epoch_change) ++input.epoch;
    if (scenario == pipeline_case::layout_change) input.resource.area.width = 32;
    if (scenario == pipeline_case::missing) {
      capture::begin_evaluation(input.epoch, input.sequence, input.viewport, input.provider, input.source_id);
    } else {
      record(scenario != pipeline_case::failed);
      if (scenario == pipeline_case::submitted) {
        checked(gpu.producer->Wait(gpu.game_gate.value, 1), "Gate newest pipeline submission");
        submit(3, retired_allocator.value, false);
        require(gpu.producer_done->GetCompletedValue() < 3, "Newest pipeline producer was not pending");
      }
    }

    capture::packet selected;
    capture::capture_diagnostic diagnostic;
    const bool acquired = capture::acquire(gpu.consumer.native(), 2, selected, &diagnostic);
    const bool allows_completed = scenario == pipeline_case::recorded || scenario == pipeline_case::submitted;
    std::printf("pipeline %s: selected=%llu newest=%llu ready=%d status=%s\n", name(scenario),
      static_cast<unsigned long long>(selected.metadata.sequence),
      static_cast<unsigned long long>(diagnostic.newest_sequence), int(selected.pixel_ready), capture::name(diagnostic.result));
    if (allows_completed) {
      require(acquired && selected.pixel_ready && !selected.shared_preservation && selected.capture_id == completed_ticket &&
        selected.metadata.sequence == 2 && diagnostic.newest_sequence == 3,
        "Completed NGX frame was starved by a newer pending evaluation");
      // Submission behind the producer's blocked frame would hang here if the
      // production owner inserted a reverse GPU wait instead of using frame 2.
      consume(selected, 2);
      if (scenario == pipeline_case::submitted)
        require(gpu.producer_done->GetCompletedValue() < 3, "Consumer unexpectedly released the game's producer gate");
    } else {
      require(!selected.pixel_ready && selected.capture_id != completed_ticket,
        "Invalid/missing/changed input reused an older NGX snapshot");
      if (scenario == pipeline_case::failed || scenario == pipeline_case::missing) {
        // Acquiring a known gap raises the admission watermark. A subsequent
        // valid but pending attempt cannot bring a pre-gap frame back to life.
        input.sequence = 4;
        record();
        capture::packet after_gap;
        capture::acquire(gpu.consumer.native(), 3, after_gap);
        require(!after_gap.pixel_ready && after_gap.capture_id != completed_ticket,
          "A later pending evaluation resurrected pre-gap pixels");
      }
    }
    gpu.drain();
    std::printf("PASS: pipeline %s\n", name(scenario));
  }

  unsigned preservation_queries{};
  bool has_generic_preservation(std::uint64_t, std::uint64_t, std::uint64_t) {
    ++preservation_queries;
    return true;
  }

  void transition(ID3D12GraphicsCommandList *commands, ID3D12Resource *resource,
      D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    commands->ResourceBarrier(1, &barrier);
  }

  struct plane_readback {
    com_ptr<ID3D12Resource> buffer;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows{};
    UINT64 row_bytes{}, total{};

    void record(ID3D12Device *device, ID3D12GraphicsCommandList *commands, ID3D12Resource *texture, UINT plane) {
      const auto desc = texture->GetDesc();
      device->GetCopyableFootprints(&desc, plane, 1, 0, &footprint, &rows, &row_bytes, &total);
      const auto format = footprint.Footprint.Format;
      const bool planar = plane ?
        format == DXGI_FORMAT_R8_TYPELESS || format == DXGI_FORMAT_R8_UINT || format == DXGI_FORMAT_R8_UNORM :
        format == DXGI_FORMAT_R32_TYPELESS || format == DXGI_FORMAT_R32_FLOAT || format == DXGI_FORMAT_R32_UINT;
      require(planar && rows == desc.Height && row_bytes == desc.Width * (plane ? 1 : sizeof(float)) &&
        footprint.Footprint.Width == desc.Width && footprint.Footprint.Height == desc.Height &&
        footprint.Footprint.Depth == 1 && row_bytes <= footprint.Footprint.RowPitch && rows &&
        footprint.Offset + UINT64(rows - 1) * footprint.Footprint.RowPitch + row_bytes <= total,
        "Unexpected D32S8 planar readback footprint");
      D3D12_HEAP_PROPERTIES heap{};
      heap.Type = D3D12_HEAP_TYPE_READBACK;
      D3D12_RESOURCE_DESC storage{};
      storage.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
      storage.Width = total;
      storage.Height = storage.DepthOrArraySize = storage.MipLevels = storage.SampleDesc.Count = 1;
      storage.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
      checked(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &storage, D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr, IID_ID3D12Resource, reinterpret_cast<void **>(buffer.put())), "Create content readback");
      D3D12_TEXTURE_COPY_LOCATION from{}, to{};
      from.pResource = texture;
      from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      from.SubresourceIndex = plane;
      to.pResource = buffer.value;
      to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
      to.PlacedFootprint = footprint;
      commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    }

    template<class Expected> void verify(UINT width, bool stencil, Expected expected, const char *message) {
      void *mapped{};
      const D3D12_RANGE range{static_cast<SIZE_T>(footprint.Offset), static_cast<SIZE_T>(total)};
      checked(buffer->Map(0, &range, &mapped), "Map completed content readback");
      bool matches = true;
      for (UINT y = 0; y < rows && matches; ++y) {
        const auto *row = static_cast<const unsigned char *>(mapped) + footprint.Offset + SIZE_T(y) * footprint.Footprint.RowPitch;
        for (UINT x = 0; x < width; ++x) {
          float actual{};
          if (stencil) actual = row[x];
          else std::memcpy(&actual, row + SIZE_T(x) * sizeof(float), sizeof(actual));
          const float wanted = expected(x, y);
          if (actual != wanted) {
            std::fprintf(stderr, "%s at (%u,%u): got %.9g expected %.9g\n", message, x, y, actual, wanted);
            matches = false;
            break;
          }
        }
      }
      const D3D12_RANGE no_write{0, 0};
      buffer->Unmap(0, &no_write);
      require(matches, message);
    }
  };

  void run_content(unsigned width, unsigned height) {
    require(width >= 16 && height >= 16 && width <= 8192 && height <= 8192, "Content size must be between 16 and 8192 pixels");
    // Declare resources before the fixture so exceptional cleanup drains the
    // queues before any descriptor or readback allocation can be destroyed.
    com_ptr<ID3D12DescriptorHeap> descriptors;
    plane_readback captured_depth, captured_stencil, source_depth, source_stencil;
    fixture gpu(width, height, true);
    struct callback_scope {
      callback_scope() { preservation_queries = 0; capture::set_preservation_available(has_generic_preservation); }
      ~callback_scope() { capture::set_preservation_available(nullptr); }
    } registered_generic;
    D3D12_DESCRIPTOR_HEAP_DESC heap{};
    heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    heap.NumDescriptors = 2;
    checked(gpu.device->CreateDescriptorHeap(&heap, IID_ID3D12DescriptorHeap,
      reinterpret_cast<void **>(descriptors.put())), "Create content DSV heap");
    D3D12_DEPTH_STENCIL_VIEW_DESC view{};
    view.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    const auto source_dsv = descriptors->GetCPUDescriptorHandleForHeapStart();
    auto destination_dsv = source_dsv;
    destination_dsv.ptr += gpu.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    gpu.device->CreateDepthStencilView(gpu.source.value, &view, source_dsv);
    gpu.device->CreateDepthStencilView(gpu.destination.value, &view, destination_dsv);

    capture::input input;
    input.provider = sunshine_scene_depth::provider_kind::ngx;
    input.epoch = input.sequence = input.source_id = 1;
    input.tick = GetTickCount64();
    input.resource = {gpu.source.native(), width, height, {0, 0, width, height}};
    input.native_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    input.proof = sunshine_scene_depth::state_proof::observed_nonzero;
    input.valid_until = sunshine_scene_depth::lifetime::until_present;
    input.source = capture::retain_source(gpu.source.native());
    input.source_present_generation = capture::source_present_generation(input.source);
    require(bool(input.source), "Retain content source");

    // Independent foreground and background values make an incomplete later
    // pass distinguishable from the full scene consumed by the middleware.
    gpu.producer_commands->ClearDepthStencilView(source_dsv,
      D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, .125f, 17, 0, nullptr);
    for (unsigned band = 0; band < 4; ++band) {
      const D3D12_RECT rect{0, LONG(band * height / 4), LONG(width), LONG((band + 1) * height / 4)};
      gpu.producer_commands->ClearDepthStencilView(source_dsv, D3D12_CLEAR_FLAG_DEPTH,
        .125f * float(band + 1), 0, 1, &rect);
    }
    const D3D12_RECT character{LONG(width / 3), LONG(height / 4), LONG(2 * width / 3), LONG(3 * height / 4)};
    gpu.producer_commands->ClearDepthStencilView(source_dsv, D3D12_CLEAR_FLAG_DEPTH, .75f, 0, 1, &character);
    transition(gpu.producer_commands.value, gpu.source.value, D3D12_RESOURCE_STATE_DEPTH_WRITE,
      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    capture::record_diagnostic recorded;
    const auto ticket = capture::nominate_evaluation(gpu.producer_commands.native(), input, UINT64_MAX, &recorded);
    require(ticket && recorded.result == capture::status::recorded, "Record the full NGX scene at the API boundary");
    std::printf("content record: ticket=%llu generic_queries=%u status=%s stage=%s\n",
      static_cast<unsigned long long>(ticket), preservation_queries, capture::name(recorded.result), capture::name(recorded.stage));
    require(!preservation_queries, "API capture delegated its timing to Generic preservation");
    capture::finish(ticket, true);

    // Same address, different contents after evaluation: any live resource or
    // later-clear selection returns .875 instead of the original scene below.
    transition(gpu.producer_commands.value, gpu.source.value, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
      D3D12_RESOURCE_STATE_DEPTH_WRITE);
    gpu.producer_commands->ClearDepthStencilView(source_dsv, D3D12_CLEAR_FLAG_DEPTH, .875f, 0, 0, nullptr);
    transition(gpu.producer_commands.value, gpu.source.value, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    checked(gpu.producer_commands->Close(), "Close API content producer");
    ID3D12CommandList *producer_lists[]{gpu.producer_commands.value};
    gpu.producer->ExecuteCommandLists(1, producer_lists);
    checked(gpu.producer->Signal(gpu.producer_done.value, 1), "Fence API content producer");
    checked(gpu.producer_commands->Reset(gpu.producer_next_allocator.value, nullptr), "Retire API content producer recording");
    require(wait_until(gpu.producer_done.value, 1, 5000), "API content producer did not finish");

    capture::packet packet;
    capture::capture_diagnostic acquired;
    require(capture::acquire(gpu.consumer.native(), 1, packet, &acquired) && packet.pixel_ready &&
      !packet.shared_preservation && packet.capture_id == ticket && packet.texture != gpu.source.native(),
      "NGX capture did not return an independent API-boundary snapshot");
    gpu.consumer_commands->ClearDepthStencilView(destination_dsv,
      D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.f, 203, 0, nullptr);
    transition(gpu.consumer_commands.value, gpu.destination.value, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_DEST);
    require(capture::copy_current(gpu.consumer_commands.native(), packet, gpu.destination.native(), D3D12_RESOURCE_STATE_COPY_DEST),
      "Copy API snapshot through the production consumer path");
    capture::complete_frame(packet, 1);
    transition(gpu.consumer_commands.value, gpu.destination.value, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    captured_depth.record(gpu.device.value, gpu.consumer_commands.value, gpu.destination.value, 0);
    captured_stencil.record(gpu.device.value, gpu.consumer_commands.value, gpu.destination.value, 1);
    source_depth.record(gpu.device.value, gpu.consumer_commands.value, gpu.source.value, 0);
    source_stencil.record(gpu.device.value, gpu.consumer_commands.value, gpu.source.value, 1);
    checked(gpu.consumer_commands->Close(), "Close API content consumer");
    ID3D12CommandList *consumer_lists[]{gpu.consumer_commands.value};
    gpu.consumer->ExecuteCommandLists(1, consumer_lists);
    checked(gpu.consumer->Signal(gpu.consumer_done.value, 1), "Fence API content readbacks");
    require(wait_until(gpu.consumer_done.value, 1, 5000), "API content readbacks did not finish");

    captured_depth.verify(width, false, [&](UINT x, UINT y) {
      if (x >= UINT(character.left) && x < UINT(character.right) && y >= UINT(character.top) && y < UINT(character.bottom)) return .75f;
      for (unsigned band = 0; band < 4; ++band) if (y < (band + 1) * height / 4) return .125f * float(band + 1);
      return -1.f;
    }, "Captured depth differs from the complete API scene");
    captured_stencil.verify(width, true, [](UINT, UINT) { return 203.f; }, "Depth capture modified consumer stencil");
    source_depth.verify(width, false, [](UINT, UINT) { return .875f; }, "Post-evaluation source overwrite did not execute");
    source_stencil.verify(width, true, [](UINT, UINT) { return 17.f; }, "API capture modified original stencil");
    require(!preservation_queries, "API path consulted Generic preservation during capture/consumption");
    gpu.drain();
    std::printf("PASS: %ux%u D32S8 API snapshot matches all %llu original pixels; later source overwrite and both stencil planes exact; Generic queries=0\n",
      width, height, static_cast<unsigned long long>(width) * height);
  }
}

int main(int argc, char **argv) {
  const bool expect_cycle = argc == 2 && std::strcmp(argv[1], "--expect-cycle") == 0;
  const bool pipeline = argc == 2 && std::strcmp(argv[1], "--pipeline") == 0;
  const bool content = (argc == 2 || argc == 4) && std::strcmp(argv[1], "--content") == 0;
  if ((!content && argc > 2) || (argc == 2 && !expect_cycle && !pipeline && !content)) {
    std::fprintf(stderr, "Usage: %s [--expect-cycle|--pipeline|--content [width height]]\n", argv[0]);
    return 2;
  }
  try {
    if (content) {
      run_content(argc == 4 ? unsigned(std::stoul(argv[2])) : 64, argc == 4 ? unsigned(std::stoul(argv[3])) : 64);
    } else if (pipeline) {
      for (const auto scenario : {pipeline_case::recorded, pipeline_case::submitted, pipeline_case::failed,
        pipeline_case::missing, pipeline_case::observation_reset, pipeline_case::feedback_reset,
        pipeline_case::reset_flag, pipeline_case::epoch_change, pipeline_case::layout_change})
        run_pipeline(scenario);
    } else run(expect_cycle);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
