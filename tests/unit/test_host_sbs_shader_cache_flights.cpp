#include <gtest/gtest.h>

#include "src/host_sbs_shader_cache_flights.h"

#include <atomic>
#include <chrono>
#include <exception>
#include <future>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
  namespace cache = models::host_sbs_shader_cache;
  using flights_t = cache::detail::bytecode_cache_t;
  using namespace std::chrono_literals;

  struct outcome_t {
    cache::bytecode_t code;
    std::exception_ptr error;
  };

  struct setup_error_t: std::runtime_error {
    std::shared_ptr<const int> token;

    explicit setup_error_t(std::shared_ptr<const int> identity):
        std::runtime_error("post-insert setup failed"), token(std::move(identity)) {}
  };

  struct nonstd_error_t {
    std::shared_ptr<const int> token;
  };

  template<class Error>
  void expect_error(const std::exception_ptr &error, const std::shared_ptr<const int> &token) {
    ASSERT_TRUE(error);
    try {
      std::rethrow_exception(error);
    } catch (const Error &caught) {
      EXPECT_EQ(caught.token, token);
      if constexpr (std::is_base_of_v<std::exception, Error>) {
        EXPECT_STREQ(caught.what(), "post-insert setup failed");
      }
    } catch (...) {
      FAIL() << "Attached caller received a different exception type";
    }
  }

  template<class Work>
  outcome_t observe(Work work) {
    try {
      return {work(), {}};
    } catch (...) {
      return {{}, std::current_exception()};
    }
  }

  void await_release(const std::shared_future<void> &release) {
    if (release.wait_for(5s) != std::future_status::ready) {
      throw std::runtime_error("Shader-cache test producer gate timed out");
    }
  }

  flights_t::key_t key(const char *name) {
    return {std::filesystem::path(name), "closure", "fixture.hlsl", "main", "cs_5_0"};
  }

  cache::bytecode_t bytes(unsigned char value) {
    return std::make_shared<const std::vector<unsigned char>>(4u, value);
  }

  TEST(HostSbsShaderCacheFlightTest, SetupExceptionSettlesAttachedWaitersAndSameKeyRetryHasOneProducer) {
    // This is the production map/promise implementation used by cache::get. Its factory
    // includes path/hash/configuration setup, before a compiler or disk resolver can run.
    flights_t flights;
    const auto shader = key("setup-failure");
    const auto failure_token = std::make_shared<const int>(17);
    const auto failure = std::make_exception_ptr(setup_error_t(failure_token));
    std::promise<void> started, release, attached;
    auto started_signal = started.get_future();
    auto release_signal = release.get_future().share();
    auto attached_signal = attached.get_future();
    std::atomic_uint producers {0u}, resolver_calls {0u}, waiters {0u};
    const auto setup = [&] {
      producers.fetch_add(1u);
      started.set_value();
      await_release(release_signal);
      std::rethrow_exception(failure);
    };
    const auto resolve = [&]() -> cache::bytecode_t {
      setup();
      resolver_calls.fetch_add(1u);
      return bytes(17u);
    };
    const auto on_hit = [&] {
      if (waiters.fetch_add(1u) + 1u == 2u) attached.set_value();
    };
    const auto request = [&] {
      return observe([&] { return flights.get(shader, resolve, on_hit); });
    };
    auto producer = std::async(std::launch::async, request);
    EXPECT_EQ(started_signal.wait_for(5s), std::future_status::ready);
    auto waiter1 = std::async(std::launch::async, request);
    auto waiter2 = std::async(std::launch::async, request);
    EXPECT_EQ(attached_signal.wait_for(5s), std::future_status::ready);
    release.set_value();
    for (auto *result : {&producer, &waiter1, &waiter2}) {
      const auto outcome = result->get();
      EXPECT_FALSE(outcome.code);
      // Exception propagation may copy an object on some SDKs. Its original dynamic type,
      // message and unique shared payload must survive for every attached caller.
      expect_error<setup_error_t>(outcome.error, failure_token);
    }
    EXPECT_EQ(producers.load(), 1u);
    EXPECT_EQ(resolver_calls.load(), 0u);
    EXPECT_EQ(waiters.load(), 2u);

    const auto replacement = bytes(29u);
    std::promise<void> retry_started, retry_release, retry_attached;
    auto retry_started_signal = retry_started.get_future();
    auto retry_release_signal = retry_release.get_future().share();
    auto retry_attached_signal = retry_attached.get_future();
    std::atomic_uint retry_producers {0u}, retry_waiters {0u};
    const auto retry = [&] {
      return flights.get(shader, [&] {
        retry_producers.fetch_add(1u);
        retry_started.set_value();
        await_release(retry_release_signal);
        return replacement;
      }, [&] {
        if (retry_waiters.fetch_add(1u) + 1u == 2u) retry_attached.set_value();
      });
    };
    auto retry_producer = std::async(std::launch::async, retry);
    EXPECT_EQ(retry_started_signal.wait_for(5s), std::future_status::ready);
    auto retry_waiter1 = std::async(std::launch::async, retry);
    auto retry_waiter2 = std::async(std::launch::async, retry);
    EXPECT_EQ(retry_attached_signal.wait_for(5s), std::future_status::ready);
    retry_release.set_value();
    EXPECT_EQ(retry_producer.get(), replacement);
    EXPECT_EQ(retry_waiter1.get(), replacement);
    EXPECT_EQ(retry_waiter2.get(), replacement);
    EXPECT_EQ(retry_producers.load(), 1u);
    EXPECT_EQ(retry_waiters.load(), 2u);
    EXPECT_EQ(flights.get(shader, [&]() -> cache::bytecode_t {
      ADD_FAILURE() << "The successful retry must remain cached";
      return {};
    }), replacement);
  }

  TEST(HostSbsShaderCacheFlightTest, OldExceptionOrNullCompletionCannotEraseReplacementAfterClear) {
    for (const bool throw_old : {false, true}) {
      SCOPED_TRACE(throw_old);
      flights_t flights;
      const auto shader = key("replaced-in-flight");
      const auto failure_token = std::make_shared<const int>(41);
      const auto failure = std::make_exception_ptr(nonstd_error_t {failure_token});
      std::promise<void> started, release, attached;
      auto started_signal = started.get_future();
      auto release_signal = release.get_future().share();
      auto attached_signal = attached.get_future();
      std::atomic_uint old_producers {0u}, waiters {0u};
      const auto old = [&] {
        return observe([&] {
          return flights.get(shader, [&]() -> cache::bytecode_t {
            old_producers.fetch_add(1u);
            started.set_value();
            await_release(release_signal);
            if (throw_old) std::rethrow_exception(failure);
            return {};
          }, [&] {
            if (waiters.fetch_add(1u) == 0u) attached.set_value();
          });
        });
      };
      auto old_producer = std::async(std::launch::async, old);
      EXPECT_EQ(started_signal.wait_for(5s), std::future_status::ready);
      auto old_waiter = std::async(std::launch::async, old);
      EXPECT_EQ(attached_signal.wait_for(5s), std::future_status::ready);
      flights.clear();
      const auto replacement = bytes(41u);
      std::atomic_uint new_producers {0u};
      const auto new_value = [&] {
        return flights.get(shader, [&] {
          new_producers.fetch_add(1u);
          return replacement;
        });
      };
      EXPECT_EQ(new_value(), replacement);
      release.set_value();
      for (auto *result : {&old_producer, &old_waiter}) {
        const auto outcome = result->get();
        EXPECT_FALSE(outcome.code);
        if (throw_old) {
          expect_error<nonstd_error_t>(outcome.error, failure_token);
        } else {
          EXPECT_FALSE(outcome.error);
        }
      }
      EXPECT_EQ(old_producers.load(), 1u);
      EXPECT_EQ(waiters.load(), 1u);
      EXPECT_EQ(new_value(), replacement);
      EXPECT_EQ(new_producers.load(), 1u) << "An old failed producer must not evict the new shared state";
    }
  }

  TEST(HostSbsShaderCacheFlightTest, NullFailureRetriesAndWaiterObserverCannotPoisonSuccessfulEntry) {
    flights_t flights;
    const auto shader = key("null-retry");
    std::atomic_uint producers {0u};
    EXPECT_FALSE(flights.get(shader, [&]() -> cache::bytecode_t {
      producers.fetch_add(1u);
      return {};
    }));
    const auto replacement = bytes(53u);
    EXPECT_EQ(flights.get(shader, [&] {
      producers.fetch_add(1u);
      return replacement;
    }), replacement);
    EXPECT_THROW(flights.get(shader, [&]() -> cache::bytecode_t {
      ADD_FAILURE() << "The observer belongs to an existing entry";
      return {};
    }, [] {
      throw std::logic_error("waiter observer failure");
    }), std::logic_error);
    EXPECT_EQ(flights.get(shader, [&]() -> cache::bytecode_t {
      ADD_FAILURE() << "Observer failure must preserve the successful producer";
      return {};
    }), replacement);
    EXPECT_EQ(producers.load(), 2u);
  }
}  // namespace
