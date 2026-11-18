#include "src/platform/windows/virtual_display_session.h"

#include <algorithm>
#include <chrono>
#include <deque>
#include <gtest/gtest.h>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
  using namespace std::chrono_literals;
  using VDISPLAY::desktop_detach_state_e;
  using VDISPLAY::display_identity_state_e;

  struct retirement_probe_t {
    using time_point = std::chrono::steady_clock::time_point;
    time_point current {};
    VDISPLAY::creation_result_t binding {
      .display_name = L"DISPLAY42",
      .device_path = L"\\\\?\\DISPLAY#SUDOVDA#probe",
      .friendly_name = L"Probe",
      .identity = SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT {{17, 23}, 41},
      .render_adapter_luid = LUID {17, 23},
    };
    GUID guid {0x38F72B96, 0xB00C, 0x4F21, {0x8B, 0x6C, 0xE1, 0xBF, 0xF1, 0x60, 0x2B, 0x0E}};
    std::vector<std::string> events;
    std::vector<std::chrono::milliseconds> detach_times;
    std::vector<std::chrono::milliseconds> detach_budgets;
    std::vector<bool> detach_received_context;
    std::vector<std::chrono::milliseconds> removal_times;
    std::vector<std::chrono::milliseconds> query_times;
    std::vector<std::chrono::milliseconds> sleeps;
    std::deque<desktop_detach_state_e> detach_results;
    std::deque<display_identity_state_e> active_results;
    std::deque<display_identity_state_e> retirement_results;
    desktop_detach_state_e default_detach = desktop_detach_state_e::detached;
    display_identity_state_e default_active = display_identity_state_e::absent;
    display_identity_state_e default_retirement = display_identity_state_e::absent;
    bool removal_accepted = true;
    bool consume_detach_budget = false;
    unsigned active_queries = 0;
    unsigned clock_reads = 0;
    std::shared_ptr<int> context_owner = std::make_shared<int>(42);
    // The injected detach operation only compares and retains this opaque marker; it never
    // dereferences a native desktop-detach context or invokes a physical operation.
    std::shared_ptr<VDISPLAY::desktop_detach_context_t> context_marker {
      context_owner,
      reinterpret_cast<VDISPLAY::desktop_detach_context_t *>(context_owner.get()),
    };

    template<class T>
    static T next(std::deque<T> &values, T fallback) {
      if (values.empty()) {
        return fallback;
      }
      const auto value = values.front();
      values.pop_front();
      return value;
    }

    std::chrono::milliseconds elapsed() const {
      return std::chrono::duration_cast<std::chrono::milliseconds>(current.time_since_epoch());
    }

    void check_binding(const VDISPLAY::creation_result_t &value) const {
      EXPECT_EQ(value.display_name, binding.display_name);
      EXPECT_EQ(value.device_path, binding.device_path);
      EXPECT_EQ(value.friendly_name, binding.friendly_name);
      EXPECT_EQ(value.identity.has_value(), binding.identity.has_value());
      if (value.identity && binding.identity) {
        EXPECT_EQ(value.identity->AdapterLuid.LowPart, binding.identity->AdapterLuid.LowPart);
        EXPECT_EQ(value.identity->AdapterLuid.HighPart, binding.identity->AdapterLuid.HighPart);
        EXPECT_EQ(value.identity->TargetId, binding.identity->TargetId);
      }
    }

    VDISPLAY::probe_retirement_io_t io() {
      return {
        .detach = [&](const VDISPLAY::creation_result_t &value, std::chrono::milliseconds budget, auto &context) {
          check_binding(value);
          events.emplace_back("detach");
          detach_times.push_back(elapsed());
          detach_budgets.push_back(budget);
          detach_received_context.push_back(static_cast<bool>(context));
          if (context) {
            EXPECT_EQ(context, context_marker);
          } else {
            context = context_marker;
          }
          if (consume_detach_budget) {
            current += budget;
          }
          return VDISPLAY::desktop_detach_result_t {.state = next(detach_results, default_detach)};
        },
        .query_active = [&](const VDISPLAY::creation_result_t &value) {
          check_binding(value);
          events.emplace_back("active");
          ++active_queries;
          return next(active_results, default_active);
        },
        .query_retirement = [&](const VDISPLAY::creation_result_t &value) {
          check_binding(value);
          events.emplace_back("retirement");
          query_times.push_back(elapsed());
          return next(retirement_results, default_retirement);
        },
        .remove = [&](const GUID &value) {
          events.emplace_back("remove");
          removal_times.push_back(elapsed());
          EXPECT_EQ(value.Data1, guid.Data1);
          EXPECT_EQ(value.Data2, guid.Data2);
          EXPECT_EQ(value.Data3, guid.Data3);
          EXPECT_TRUE(std::equal(std::begin(value.Data4), std::end(value.Data4), std::begin(guid.Data4)));
          return removal_accepted;
        },
        .now = [&]() {
          ++clock_reads;
          return current;
        },
        .sleep = [&](std::chrono::milliseconds delay) {
          EXPECT_GT(delay, 0ms);
          sleeps.push_back(delay);
          current += delay;
        },
      };
    }

    bool retire() {
      return VDISPLAY::retire_probe_display_for_test(binding, guid, io());
    }
  };

  TEST(StartupProbeDisplayTest, PublishedIdentityRetiresWithExactArgumentsAndRecordsTheProof) {
    retirement_probe_t probe;
    ASSERT_TRUE(probe.retire());
    EXPECT_EQ(probe.detach_times.size(), 1u);
    EXPECT_EQ(probe.detach_budgets, (std::vector<std::chrono::milliseconds> {1500ms}));
    EXPECT_EQ(probe.active_queries, 1u);
    EXPECT_EQ(probe.removal_times.size(), 1u);
    EXPECT_GE(probe.query_times.size(), 3u);
    EXPECT_GE(probe.elapsed(), 500ms);
    EXPECT_LT(probe.elapsed(), 8s);
    ASSERT_GE(probe.events.size(), 4u);
    EXPECT_EQ((std::vector<std::string>(probe.events.begin(), probe.events.begin() + 4)), (std::vector<std::string> {"detach", "active", "remove", "retirement"}));
    // The startup policy will deliberately adopt the shared owner's extra settled/final
    // confirmation. Observe the actual old and new proof without asserting their equality.
    RecordProperty("retirement_queries", static_cast<int>(probe.query_times.size()));
    RecordProperty("retirement_elapsed_ms", static_cast<int>(probe.elapsed().count()));
  }

  TEST(StartupProbeDisplayTest, FailedAddHasNoIdentityAndPerformsNoRetirementIo) {
    retirement_probe_t probe;
    probe.binding = {};
    EXPECT_TRUE(probe.retire());
    EXPECT_TRUE(probe.events.empty());
    EXPECT_TRUE(probe.sleeps.empty());
    EXPECT_EQ(probe.clock_reads, 0u);
  }

  TEST(StartupProbeDisplayTest, UnpublishedAddRetainsItsExactIdentityThroughThePublicationQuarantine) {
    retirement_probe_t probe;
    probe.binding.display_name.clear();
    probe.binding.device_path.clear();
    ASSERT_TRUE(probe.retire());
    EXPECT_EQ(probe.removal_times.size(), 1u);
    EXPECT_EQ(probe.detach_received_context, (std::vector<bool> {false}));
    EXPECT_GE(probe.elapsed(), 750ms);
    EXPECT_LT(probe.elapsed(), 8s);
    EXPECT_GE(probe.query_times.size(), 4u);
    RecordProperty("unpublished_retirement_queries", static_cast<int>(probe.query_times.size()));
    RecordProperty("unpublished_retirement_elapsed_ms", static_cast<int>(probe.elapsed().count()));
  }

  TEST(StartupProbeDisplayTest, SoleActiveDisplayBypassesOnlyTheActivePathRemovalCheck) {
    retirement_probe_t probe;
    probe.default_detach = desktop_detach_state_e::skipped_only_active;
    probe.default_active = display_identity_state_e::indeterminate;
    ASSERT_TRUE(probe.retire());
    EXPECT_EQ(probe.active_queries, 0u);
    EXPECT_EQ(probe.removal_times.size(), 1u);
    EXPECT_GE(probe.query_times.size(), 3u);
    ASSERT_GE(probe.events.size(), 3u);
    EXPECT_EQ((std::vector<std::string>(probe.events.begin(), probe.events.begin() + 3)), (std::vector<std::string> {"detach", "remove", "retirement"}));
  }

  TEST(StartupProbeDisplayTest, DetachRetriesRetainContextAndReattachmentRearmsItBeforeRemoval) {
    retirement_probe_t probe;
    probe.detach_results = {
      desktop_detach_state_e::ambiguous_identity,
      desktop_detach_state_e::already_inactive,
      desktop_detach_state_e::detached,
    };
    probe.active_results = {display_identity_state_e::present, display_identity_state_e::absent};
    ASSERT_TRUE(probe.retire());
    EXPECT_EQ(probe.detach_received_context, (std::vector<bool> {false, true, false}));
    EXPECT_EQ(probe.active_queries, 2u);
    EXPECT_EQ(probe.removal_times.size(), 1u);
    ASSERT_EQ(probe.detach_times.size(), 3u);
    EXPECT_EQ(probe.detach_times.front(), 0ms);
    EXPECT_EQ(probe.detach_times[1], 250ms);
    EXPECT_EQ(probe.detach_times[2], 250ms);
    ASSERT_GE(probe.events.size(), 7u);
    EXPECT_EQ((std::vector<std::string>(probe.events.begin(), probe.events.begin() + 7)), (std::vector<std::string> {"detach", "detach", "active", "detach", "active", "remove", "retirement"}));
  }

  TEST(StartupProbeDisplayTest, RemoveAcknowledgementDoesNotCompleteOrRepeatRemovalDuringAmbiguousAbsence) {
    retirement_probe_t probe;
    probe.retirement_results = {
      display_identity_state_e::present,
      display_identity_state_e::absent,
      display_identity_state_e::indeterminate,
      display_identity_state_e::absent,
      display_identity_state_e::absent,
      display_identity_state_e::absent,
    };
    ASSERT_TRUE(probe.retire());
    EXPECT_EQ(probe.removal_times.size(), 1u);
    EXPECT_EQ(probe.active_queries, 1u);
    EXPECT_GE(probe.query_times.size(), 6u);
    EXPECT_GE(probe.elapsed(), 1250ms);
  }

  TEST(StartupProbeDisplayTest, RejectedRemoveCanCompleteFromExternalExactAbsenceWithBoundedRetries) {
    retirement_probe_t probe;
    probe.removal_accepted = false;
    ASSERT_TRUE(probe.retire());
    ASSERT_GE(probe.removal_times.size(), 3u);
    EXPECT_EQ(probe.active_queries, probe.removal_times.size());
    for (std::size_t index = 1; index < probe.removal_times.size(); ++index) {
      EXPECT_GE(probe.removal_times[index] - probe.removal_times[index - 1], 250ms);
    }
    EXPECT_GE(probe.query_times.size(), 3u);
  }

  TEST(StartupProbeDisplayTest, AmbiguousDetachKeepsOwnershipUntilTheBudgetExpiresWithoutRemoval) {
    retirement_probe_t probe;
    probe.default_detach = desktop_detach_state_e::ambiguous_identity;
    probe.consume_detach_budget = true;
    EXPECT_FALSE(probe.retire());
    ASSERT_GE(probe.detach_budgets.size(), 2u);
    EXPECT_EQ(probe.active_queries, 0u);
    EXPECT_TRUE(probe.removal_times.empty());
    EXPECT_TRUE(probe.query_times.empty());
    EXPECT_FALSE(probe.detach_received_context.front());
    EXPECT_TRUE(std::all_of(probe.detach_received_context.begin() + 1, probe.detach_received_context.end(), [](bool received) {
      return received;
    }));
    for (std::size_t index = 0; index < probe.detach_budgets.size(); ++index) {
      EXPECT_EQ(probe.detach_budgets[index], std::min(1500ms, 8000ms - probe.detach_times[index]));
      EXPECT_GT(probe.detach_budgets[index], 0ms);
    }
    EXPECT_LT(probe.detach_budgets.back(), 1500ms);
    EXPECT_GE(probe.elapsed(), 8s);
    EXPECT_LE(probe.elapsed(), 8250ms);
  }

  TEST(StartupProbeDisplayTest, AcknowledgedRemovalCannotAuthorizeAnIndeterminateCompleteTopology) {
    retirement_probe_t probe;
    probe.default_retirement = display_identity_state_e::indeterminate;
    EXPECT_FALSE(probe.retire());
    EXPECT_EQ(probe.removal_times.size(), 1u);
    EXPECT_EQ(probe.active_queries, 1u);
    EXPECT_GE(probe.query_times.size(), 3u);
    EXPECT_GE(probe.elapsed(), 8s);
    EXPECT_LE(probe.elapsed(), 8250ms);
  }

  TEST(StartupProbeDisplayTest, MissingInjectedOperationsAreRejectedBeforeAnyIo) {
    for (unsigned missing = 0; missing < 6; ++missing) {
      SCOPED_TRACE(missing);
      retirement_probe_t probe;
      auto io = probe.io();
      switch (missing) {
        case 0:
          io.detach = {};
          break;
        case 1:
          io.query_active = {};
          break;
        case 2:
          io.query_retirement = {};
          break;
        case 3:
          io.remove = {};
          break;
        case 4:
          io.now = {};
          break;
        case 5:
          io.sleep = {};
          break;
      }
      EXPECT_THROW(VDISPLAY::retire_probe_display_for_test(probe.binding, probe.guid, std::move(io)), std::invalid_argument);
      EXPECT_TRUE(probe.events.empty());
      EXPECT_TRUE(probe.sleeps.empty());
      EXPECT_EQ(probe.clock_reads, 0u);
    }
  }
}  // namespace

namespace {
  TEST(StartupProbeDisplayTest, ReappearanceAtTheSettledFourthObservationCannotCompleteRetirement) {
    retirement_probe_t probe;
    probe.retirement_results = {display_identity_state_e::absent, display_identity_state_e::absent, display_identity_state_e::absent};
    probe.default_retirement = display_identity_state_e::present;
    EXPECT_FALSE(probe.retire());
    EXPECT_EQ(probe.removal_times.size(), 1u);
    ASSERT_GE(probe.query_times.size(), 4u);
    EXPECT_GT(probe.query_times[3], probe.query_times[2]);
    EXPECT_TRUE(std::all_of(probe.query_times.begin(), probe.query_times.end(), [](auto at) {
      return at < 8000ms;
    }));
    EXPECT_EQ(probe.elapsed(), 8s);
  }

  TEST(StartupProbeDisplayTest, UnknownSettledFourthObservationCannotCompleteRetirement) {
    retirement_probe_t probe;
    probe.retirement_results = {display_identity_state_e::absent, display_identity_state_e::absent, display_identity_state_e::absent};
    probe.default_retirement = display_identity_state_e::indeterminate;
    EXPECT_FALSE(probe.retire());
    EXPECT_EQ(probe.removal_times.size(), 1u);
    ASSERT_GE(probe.query_times.size(), 4u);
    EXPECT_GT(probe.query_times[3], probe.query_times[2]);
    EXPECT_TRUE(std::all_of(probe.query_times.begin(), probe.query_times.end(), [](auto at) {
      return at < 8000ms;
    }));
    EXPECT_EQ(probe.elapsed(), 8s);
  }

  TEST(StartupProbeDisplayTest, ReappearanceAtTheFinalFifthObservationCannotReleaseTheOwner) {
    retirement_probe_t probe;
    probe.retirement_results = {display_identity_state_e::absent, display_identity_state_e::absent, display_identity_state_e::absent, display_identity_state_e::absent};
    probe.default_retirement = display_identity_state_e::present;
    EXPECT_FALSE(probe.retire());
    EXPECT_EQ(probe.removal_times.size(), 1u);
    ASSERT_GE(probe.query_times.size(), 5u);
    EXPECT_GT(probe.query_times[3], probe.query_times[2]);
    EXPECT_EQ(probe.query_times[4], probe.query_times[3]);
    EXPECT_TRUE(std::all_of(probe.query_times.begin(), probe.query_times.end(), [](auto at) {
      return at < 8000ms;
    }));
    EXPECT_EQ(probe.elapsed(), 8s);
  }

  TEST(StartupProbeDisplayTest, UnknownFinalFifthObservationCannotReleaseTheOwner) {
    retirement_probe_t probe;
    probe.retirement_results = {display_identity_state_e::absent, display_identity_state_e::absent, display_identity_state_e::absent, display_identity_state_e::absent};
    probe.default_retirement = display_identity_state_e::indeterminate;
    EXPECT_FALSE(probe.retire());
    EXPECT_EQ(probe.removal_times.size(), 1u);
    ASSERT_GE(probe.query_times.size(), 5u);
    EXPECT_GT(probe.query_times[3], probe.query_times[2]);
    EXPECT_EQ(probe.query_times[4], probe.query_times[3]);
    EXPECT_TRUE(std::all_of(probe.query_times.begin(), probe.query_times.end(), [](auto at) {
      return at < 8000ms;
    }));
    EXPECT_EQ(probe.elapsed(), 8s);
  }

  TEST(StartupProbeDisplayTest, DetachConsumingTheRemainingBudgetCannotIssueLaterNativeQueriesOrRemoval) {
    retirement_probe_t probe;
    probe.consume_detach_budget = true;
    probe.detach_results = {
      desktop_detach_state_e::ambiguous_identity,
      desktop_detach_state_e::ambiguous_identity,
      desktop_detach_state_e::ambiguous_identity,
      desktop_detach_state_e::ambiguous_identity,
      desktop_detach_state_e::detached,
    };
    EXPECT_FALSE(probe.retire());
    ASSERT_EQ(probe.detach_budgets.size(), 5u);
    EXPECT_EQ(probe.detach_budgets.back(), 1000ms);
    EXPECT_EQ(probe.detach_times.back() + probe.detach_budgets.back(), 8s);
    EXPECT_EQ(probe.active_queries, 0u);
    EXPECT_TRUE(probe.removal_times.empty());
    EXPECT_TRUE(probe.query_times.empty());
    EXPECT_EQ(probe.elapsed(), 8s);
  }

  TEST(StartupProbeDisplayTest, PerpetualImmediateReattachmentMakesBoundedProgressWithoutUnsafeRemoval) {
    retirement_probe_t probe;
    probe.default_active = display_identity_state_e::present;
    auto io = probe.io();
    io.query_active = [&, query_active = std::move(io.query_active)](const auto &binding) {
      // Fail promptly if a missing progress bound would otherwise spin forever at fake time zero.
      if (probe.active_queries >= 128) {
        throw std::runtime_error("probe retirement did not advance its bounded retry clock");
      }
      return query_active(binding);
    };
    EXPECT_FALSE(VDISPLAY::retire_probe_display_for_test(probe.binding, probe.guid, std::move(io)));
    EXPECT_GT(probe.active_queries, 1u);
    EXPECT_LT(probe.active_queries, 128u);
    EXPECT_TRUE(probe.removal_times.empty());
    EXPECT_TRUE(probe.query_times.empty());
    EXPECT_FALSE(probe.sleeps.empty());
    EXPECT_TRUE(std::all_of(probe.sleeps.begin(), probe.sleeps.end(), [](auto delay) {
      return delay > 0ms && delay <= 250ms;
    }));
    EXPECT_EQ(probe.elapsed(), 8s);
  }
}  // namespace
