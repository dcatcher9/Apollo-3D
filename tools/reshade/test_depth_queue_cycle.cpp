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

    fixture() {
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
      texture.Width = texture.Height = 64;
      texture.DepthOrArraySize = texture.MipLevels = texture.SampleDesc.Count = 1;
      texture.Format = DXGI_FORMAT_R32_FLOAT;
      texture.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
      checked(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture,
        D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr, IID_ID3D12Resource,
        reinterpret_cast<void **>(source.put())), "Create source depth texture");
      checked(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_ID3D12Resource,
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
      if (capture_initialized) capture::shutdown();
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
}

int main(int argc, char **argv) {
  const bool expect_cycle = argc == 2 && std::strcmp(argv[1], "--expect-cycle") == 0;
  if (argc > 2 || (argc == 2 && !expect_cycle)) {
    std::fprintf(stderr, "Usage: %s [--expect-cycle]\n", argv[0]);
    return 2;
  }
  try {
    run(expect_cycle);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
