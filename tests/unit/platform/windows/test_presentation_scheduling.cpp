#include "src/platform/windows/presentation_scheduling.h"

#include <barrier>
#include <gtest/gtest.h>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
  struct scheduling_calls_t {
    int starts = 0;
    int stops = 0;
    bool throw_on_start = false;
  };

  struct fake_scheduling_operations_t {
    scheduling_calls_t *calls;

    void start() {
      ++calls->starts;
      if (calls->throw_on_start) {
        throw std::runtime_error("partial setup failure");
      }
    }

    void stop() noexcept {
      ++calls->stops;
    }
  };

  using scheduling_t = platf::detail::presentation_scheduling_t<fake_scheduling_operations_t>;
}  // namespace

TEST(PresentationScheduling, RemoteStopPreservesOverlappingLocalOwner) {
  scheduling_calls_t calls;
  scheduling_t scheduling {{&calls}};
  auto remote = scheduling.acquire();
  auto local = scheduling.acquire();
  EXPECT_EQ(calls.starts, 1);
  remote.reset();
  EXPECT_EQ(calls.stops, 0);
  local.reset();
  EXPECT_EQ(calls.stops, 1);
}

TEST(PresentationScheduling, LocalStopPreservesOverlappingRemoteOwner) {
  scheduling_calls_t calls;
  scheduling_t scheduling {{&calls}};
  auto local = scheduling.acquire();
  auto remote = scheduling.acquire();
  local.reset();
  EXPECT_EQ(calls.starts, 1);
  EXPECT_EQ(calls.stops, 0);
  remote.reset();
  EXPECT_EQ(calls.stops, 1);
}

TEST(PresentationScheduling, MoveTransfersOneOwnerAndCleanupRunsOnce) {
  scheduling_calls_t calls;
  scheduling_t scheduling {{&calls}};
  auto first = scheduling.acquire();
  auto second = scheduling.acquire();
  auto moved = std::move(first);
  first.reset();
  second = std::move(moved);
  EXPECT_EQ(calls.starts, 1);
  EXPECT_EQ(calls.stops, 0);
  second.reset();
  second.reset();
  EXPECT_EQ(calls.stops, 1);
}

TEST(PresentationScheduling, NewLifetimeRefreshesOnlyAfterLastOwnerExits) {
  scheduling_calls_t calls;
  scheduling_t scheduling {{&calls}};
  {
    auto first = scheduling.acquire();
    auto second = scheduling.acquire();
  }
  EXPECT_EQ(calls.starts, 1);
  EXPECT_EQ(calls.stops, 1);
  {
    auto next = scheduling.acquire();
    EXPECT_EQ(calls.starts, 2);
  }
  EXPECT_EQ(calls.stops, 2);
}

TEST(PresentationScheduling, PartialSetupExceptionCleansUpWithoutPublishingOwner) {
  scheduling_calls_t calls {.throw_on_start = true};
  scheduling_t scheduling {{&calls}};
  EXPECT_THROW((void) scheduling.acquire(), std::runtime_error);
  EXPECT_EQ(calls.stops, 1);
  calls.throw_on_start = false;
  {
    auto next = scheduling.acquire();
    EXPECT_EQ(calls.starts, 2);
  }
  EXPECT_EQ(calls.stops, 2);
}

TEST(PresentationScheduling, ConcurrentOwnersHaveOneSerializedSetupAndCleanup) {
  scheduling_calls_t calls;
  scheduling_t scheduling {{&calls}};
  constexpr int owner_count = 8;
  std::barrier start {owner_count};
  std::barrier acquired {owner_count};
  std::vector<std::jthread> workers;
  for (int index = 0; index < owner_count; ++index) {
    workers.emplace_back([&]() {
      start.arrive_and_wait();
      auto lease = scheduling.acquire();
      acquired.arrive_and_wait();
    });
  }
  workers.clear();
  EXPECT_EQ(calls.starts, 1);
  EXPECT_EQ(calls.stops, 1);
}

TEST(PresentationSchedulingTimer, FailedAcquisitionsNeverReleaseAnUnownedRequest) {
  platf::detail::presentation_timer_request_t timer;
  int native_releases = 0, multimedia_releases = 0;
  timer.start([]() {
    return std::optional<std::uint32_t> {};
  },
              []() {
                return false;
              });
  timer.stop([&](auto) {
    ++native_releases;
    return true;
  },
             [&]() {
               ++multimedia_releases;
               return true;
             });
  EXPECT_EQ(native_releases, 0);
  EXPECT_EQ(multimedia_releases, 0);
}

TEST(PresentationSchedulingTimer, NativeRequestReleasesItsExactSuccessfulResolutionOnce) {
  platf::detail::presentation_timer_request_t timer;
  int multimedia_acquires = 0, native_releases = 0, multimedia_releases = 0;
  timer.start([]() {
    return std::optional<std::uint32_t> {5000};
  },
              [&]() {
                ++multimedia_acquires;
                return true;
              });
  const auto release_native = [&](auto resolution) {
    EXPECT_EQ(resolution, 5000u);
    ++native_releases;
    return true;
  };
  const auto release_multimedia = [&]() {
    ++multimedia_releases;
    return true;
  };
  timer.stop(release_native, release_multimedia);
  timer.stop(release_native, release_multimedia);
  EXPECT_EQ(multimedia_acquires, 0);
  EXPECT_EQ(native_releases, 1);
  EXPECT_EQ(multimedia_releases, 0);
}

TEST(PresentationSchedulingTimer, NativeFailureUsesAndReleasesOnlySuccessfulFallback) {
  platf::detail::presentation_timer_request_t timer;
  int native_releases = 0, multimedia_releases = 0;
  timer.start([]() {
    return std::optional<std::uint32_t> {};
  },
              []() {
                return true;
              });
  timer.stop([&](auto) {
    ++native_releases;
    return true;
  },
             [&]() {
               ++multimedia_releases;
               return true;
             });
  EXPECT_EQ(native_releases, 0);
  EXPECT_EQ(multimedia_releases, 1);
}

TEST(PresentationSchedulingTimer, FailedNativeReleaseRetainsOwnershipWithoutDoubleAcquire) {
  platf::detail::presentation_timer_request_t timer;
  int acquisitions = 0, releases = 0;
  const auto acquire = [&]() {
    ++acquisitions;
    return std::optional<std::uint32_t> {5000};
  };
  timer.start(acquire, []() {
    return false;
  });
  timer.stop([&](auto) {
    ++releases;
    return false;
  },
             []() {
               return true;
             });
  timer.start(acquire, []() {
    return false;
  });
  timer.stop([&](auto resolution) {
    EXPECT_EQ(resolution, 5000u);
    ++releases;
    return true;
  },
             []() {
               return true;
             });
  EXPECT_EQ(acquisitions, 1);
  EXPECT_EQ(releases, 2);
}

TEST(PresentationSchedulingTimer, FailedFallbackReleaseRetainsOwnershipWithoutDoubleAcquire) {
  platf::detail::presentation_timer_request_t timer;
  int acquisitions = 0, releases = 0;
  const auto acquire = [&]() {
    ++acquisitions;
    return true;
  };
  timer.start([]() {
    return std::optional<std::uint32_t> {};
  },
              acquire);
  timer.stop([](auto) {
    return true;
  },
             [&]() {
               ++releases;
               return false;
             });
  timer.start([]() {
    return std::optional<std::uint32_t> {};
  },
              acquire);
  timer.stop([](auto) {
    return true;
  },
             [&]() {
               ++releases;
               return true;
             });
  EXPECT_EQ(acquisitions, 1);
  EXPECT_EQ(releases, 2);
}
