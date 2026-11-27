/**
 * @file tests/unit/test_host_sbs_depth_gpu_timing.cpp
 * @brief CPU tests of the depth estimator's CUDA event ring and its failure results.
 */
#include "../tests_common.h"

#include <src/host_sbs_depth_gpu_timing.h>

#include <cstdint>

namespace {
  using models::host_sbs_depth_gpu_timing::cuda_event_ring_t;

  // Driver results the trimmed CUresult declaration does not name.
  constexpr auto invalid_value = static_cast<CUresult>(1);
  constexpr auto launch_failed = static_cast<CUresult>(719);

  struct fake_events_t {
    int created = 0;
    int records = 0;
    int destroyed = 0;
    CUresult create_result = CUDA_SUCCESS;
    CUresult record_result = CUDA_SUCCESS;
    CUresult query_result = CUDA_SUCCESS;
    CUresult elapsed_result = CUDA_SUCCESS;
    CUresult destroy_result = CUDA_SUCCESS;
  };
  thread_local fake_events_t fake_events;

  cuda_driver_api fake_event_api() {
    cuda_driver_api api;
    api.cuEventCreate = [](CUevent *event, unsigned int) -> CUresult {
      if (fake_events.create_result != CUDA_SUCCESS) {
        return fake_events.create_result;
      }
      *event = reinterpret_cast<CUevent>(static_cast<std::uintptr_t>(0x1000 + ++fake_events.created));
      return CUDA_SUCCESS;
    };
    api.cuEventRecord = [](CUevent, CUstream) -> CUresult {
      ++fake_events.records;
      return fake_events.record_result;
    };
    api.cuEventQuery = [](CUevent) -> CUresult {
      return fake_events.query_result;
    };
    api.cuEventElapsedTime = [](float *ms, CUevent, CUevent) -> CUresult {
      *ms = 2.5f;
      return fake_events.elapsed_result;
    };
    api.cuEventDestroy = [](CUevent) -> CUresult {
      ++fake_events.destroyed;
      return fake_events.destroy_result;
    };
    return api;
  }
}  // namespace

TEST(HostSbsDepthCudaEventRingTest, ReadsBackEachSampleOnceAfterItsStopCompletes) {
  fake_events = {};
  auto api = fake_event_api();
  cuda_event_ring_t ring;
  ASSERT_TRUE(cuda_event_ring_t::recording_available(api));

  const int slot = ring.next_slot();
  EXPECT_EQ(slot, 0);
  EXPECT_FALSE(ring.record_start(slot, nullptr, api));
  EXPECT_EQ(fake_events.created, 2);
  EXPECT_FALSE(ring.in_flight(slot));
  const auto idle = ring.try_resolve(slot, api);
  EXPECT_FALSE(idle.elapsed_ms);
  EXPECT_FALSE(idle.failure);

  EXPECT_FALSE(ring.record_stop(slot, nullptr, api));
  EXPECT_TRUE(ring.in_flight(slot));
  EXPECT_EQ(ring.next_slot(), 1);

  fake_events.query_result = CUDA_ERROR_NOT_READY;
  const auto running = ring.try_resolve(slot, api);
  EXPECT_FALSE(running.elapsed_ms);
  EXPECT_FALSE(running.failure);
  EXPECT_TRUE(ring.in_flight(slot));

  fake_events.query_result = CUDA_SUCCESS;
  const auto ready = ring.try_resolve(slot, api);
  ASSERT_TRUE(ready.elapsed_ms);
  EXPECT_FLOAT_EQ(*ready.elapsed_ms, 2.5f);
  EXPECT_FALSE(ready.failure);
  EXPECT_FALSE(ring.in_flight(slot));
  EXPECT_FALSE(ring.try_resolve(slot, api).elapsed_ms);

  // A slot keeps its events for its next measurement.
  EXPECT_FALSE(ring.record_start(slot, nullptr, api));
  EXPECT_EQ(fake_events.created, 2);
  EXPECT_TRUE(ring.destroy_events(api));
  EXPECT_EQ(fake_events.destroyed, 2);
  EXPECT_TRUE(ring.destroy_events(api));
  EXPECT_EQ(fake_events.destroyed, 2);
}

TEST(HostSbsDepthCudaEventRingTest, ReturnsEachFailureAndLeavesPolicyToTheOwner) {
  fake_events = {};
  auto api = fake_event_api();
  cuda_event_ring_t ring;

  fake_events.create_result = CUDA_ERROR_NOT_SUPPORTED;
  const auto creation = ring.record_start(0, nullptr, api);
  ASSERT_TRUE(creation);
  EXPECT_EQ(creation->operation, "diagnostic timing-event creation");
  EXPECT_EQ(creation->result, CUDA_ERROR_NOT_SUPPORTED);
  EXPECT_EQ(fake_events.records, 0);

  fake_events.create_result = CUDA_SUCCESS;
  fake_events.record_result = CUDA_ERROR_INVALID_HANDLE;
  const auto start = ring.record_start(0, nullptr, api);
  ASSERT_TRUE(start);
  EXPECT_EQ(start->operation, "diagnostic timing-event start record");
  EXPECT_EQ(start->result, CUDA_ERROR_INVALID_HANDLE);

  const auto stop = ring.record_stop(0, nullptr, api);
  ASSERT_TRUE(stop);
  EXPECT_EQ(stop->operation, "diagnostic timing-event stop record");
  EXPECT_FALSE(ring.in_flight(0));
  EXPECT_EQ(ring.next_slot(), 0);

  fake_events.record_result = CUDA_SUCCESS;
  ASSERT_FALSE(ring.record_start(0, nullptr, api));
  ASSERT_FALSE(ring.record_stop(0, nullptr, api));
  fake_events.query_result = launch_failed;
  const auto query = ring.try_resolve(0, api);
  ASSERT_TRUE(query.failure);
  EXPECT_FALSE(query.elapsed_ms);
  EXPECT_EQ(query.failure->operation, "diagnostic timing-event query");
  EXPECT_EQ(query.failure->result, launch_failed);
  EXPECT_TRUE(ring.in_flight(0));

  fake_events.query_result = CUDA_SUCCESS;
  fake_events.elapsed_result = invalid_value;
  const auto elapsed = ring.try_resolve(0, api);
  ASSERT_TRUE(elapsed.failure);
  EXPECT_EQ(elapsed.failure->operation, "diagnostic timing-event elapsed query");
  EXPECT_TRUE(ring.in_flight(0));

  // The first failed destroy keeps that handle and issues no later destroy.
  ASSERT_FALSE(ring.record_start(1, nullptr, api));
  fake_events.destroy_result = launch_failed;
  EXPECT_FALSE(ring.destroy_events(api));
  EXPECT_EQ(fake_events.destroyed, 1);
  EXPECT_FALSE(ring.destroy_events(api));
  EXPECT_EQ(fake_events.destroyed, 2);
}
