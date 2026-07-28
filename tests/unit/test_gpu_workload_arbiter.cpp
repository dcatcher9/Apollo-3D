#include <gtest/gtest.h>

#include "src/gpu_workload_arbiter.h"
#include "src/rtsp.h"

TEST(GpuWorkloadArbiter, OfflineAndLiveReservationsAreAtomicAndExclusive) {
  {
    auto live = gpu_workload::try_acquire(gpu_workload::kind_e::live_stream);
    ASSERT_TRUE(live);
    EXPECT_TRUE(gpu_workload::live_stream_active());
    EXPECT_FALSE(
      gpu_workload::try_acquire(gpu_workload::kind_e::offline_sbs)
    );
  }
  EXPECT_FALSE(gpu_workload::live_stream_active());

  {
    auto offline = gpu_workload::try_acquire(gpu_workload::kind_e::offline_sbs);
    ASSERT_TRUE(offline);
    EXPECT_TRUE(gpu_workload::offline_sbs_active());
    EXPECT_FALSE(
      gpu_workload::try_acquire(gpu_workload::kind_e::live_stream)
    );
  }
  EXPECT_FALSE(gpu_workload::offline_sbs_active());
}

TEST(GpuWorkloadArbiter, MoveTransfersExactlyOneLease) {
  auto original = gpu_workload::try_acquire(gpu_workload::kind_e::offline_sbs);
  ASSERT_TRUE(original);
  gpu_workload::lease_t moved = std::move(*original);
  original.reset();
  EXPECT_TRUE(gpu_workload::offline_sbs_active());
  moved.reset();
  EXPECT_FALSE(gpu_workload::offline_sbs_active());
}

TEST(GpuWorkloadArbiter, PendingLaunchLeaseTransfersWithoutOfflineAdmissionGap) {
  rtsp_stream::launch_session_t launch {};
  ASSERT_TRUE(launch.reserve_live_gpu());
  EXPECT_TRUE(gpu_workload::live_stream_active());
  EXPECT_FALSE(
    gpu_workload::try_acquire(gpu_workload::kind_e::offline_sbs)
  );

  auto active = launch.take_live_gpu_lease();
  ASSERT_TRUE(active);
  EXPECT_FALSE(launch.pending_live_gpu_lease);
  EXPECT_TRUE(gpu_workload::live_stream_active());
  EXPECT_FALSE(
    gpu_workload::try_acquire(gpu_workload::kind_e::offline_sbs)
  ) << "Moving pending ownership into the active slot must never open an admission gap";

  active.reset();
  EXPECT_FALSE(gpu_workload::live_stream_active());
  EXPECT_TRUE(
    gpu_workload::try_acquire(gpu_workload::kind_e::offline_sbs)
  );
}

TEST(GpuWorkloadArbiter, OfflineLeaseRejectsLaunchBeforeGpuWorkBegins) {
  auto offline =
    gpu_workload::try_acquire(gpu_workload::kind_e::offline_sbs);
  ASSERT_TRUE(offline);

  rtsp_stream::launch_session_t launch {};
  EXPECT_FALSE(launch.reserve_live_gpu());
  EXPECT_FALSE(launch.pending_live_gpu_lease);
}

TEST(GpuWorkloadArbiter, RejectedPublicationReleasesLeaseEvenWhenLaunchRecordSurvives) {
  rtsp_stream::launch_session_t retained_launch {};
  ASSERT_TRUE(retained_launch.reserve_live_gpu());
  EXPECT_TRUE(gpu_workload::live_stream_active());

  retained_launch.release_unpublished_live_gpu();

  EXPECT_FALSE(retained_launch.pending_live_gpu_lease);
  EXPECT_FALSE(gpu_workload::live_stream_active());
  EXPECT_TRUE(
    gpu_workload::try_acquire(gpu_workload::kind_e::offline_sbs)
  );
}
