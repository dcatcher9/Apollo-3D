#include "src/platform/windows/virtual_display_session.h"

#include <deque>
#include <gtest/gtest.h>
#include <stdexcept>

namespace {
  using namespace std::chrono_literals;
  using namespace VDISPLAY;

  struct owner_probe_t {
    session_t::time_point now {};
    std::vector<std::string> events;
    bool prepare_ok = true;
    bool bind_ok = true;
    bool restore_ok = true;
    bool pause_ok = true;
    bool reactivate_ok = true;
    bool remove_ok = true;
    unsigned prepares = 0;
    unsigned creates = 0;
    unsigned binds = 0;
    unsigned restores = 0;
    unsigned pauses = 0;
    unsigned reactivations = 0;
    unsigned cursor_restores = 0;
    unsigned removes = 0;
    unsigned active_queries = 0;
    unsigned retirement_queries = 0;
    display_identity_state_e retirement_state = display_identity_state_e::absent;
    std::deque<display_identity_state_e> retirement_observations;
    display_spec_t prepared_spec;
    creation_result_t created {
      .display_name = L"DISPLAY7",
      .device_path = L"stable-A",
      .friendly_name = L"Virtual Display",
      .identity = SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT {{3, 4}, 5},
    };
    display_identity_query_t observed {
      display_identity_state_e::present, L"DISPLAY7", L"stable-A", L"Virtual Display",
    };
    platf::primary_display::retained_display_ptr retained_marker {
      reinterpret_cast<platf::primary_display::retained_display_t *>(this), [](auto *) {},
    };

    display_spec_t spec() const {
      display_spec_t result;
      result.client_uid = "uid";
      result.client_name = "display";
      result.guid.Data1 = 11;
      result.width = 1920;
      result.height = 1080;
      result.refresh_millihz = 120013;
      result.exclusive = true;
      result.local_sink = L"glasses";
      return result;
    }

    session_io_t io() {
      return {
        .prepare = [&](const display_spec_t &value) {
          events.emplace_back("prepare");
          ++prepares;
          prepared_spec = value;
          return prepare_ok;
        },
        .create = [&](const display_spec_t &value) {
          events.emplace_back("create");
          ++creates;
          EXPECT_EQ(value.refresh_millihz, 120013u);
          return created;
        },
        .bind = [&](std::wstring_view path) {
          events.emplace_back("bind");
          ++binds;
          EXPECT_EQ(path, L"stable-A");
          return bind_ok;
        },
        .promote = [&](std::wstring_view path, bool) {
          events.emplace_back("promote");
          EXPECT_EQ(path, L"stable-A");
          return true;
        },
        .restore = [&](std::wstring_view) {
          events.emplace_back("restore");
          ++restores;
          return restore_ok;
        },
        .pause = [&](std::wstring_view path, auto &retained, std::wstring_view sink) {
          events.emplace_back("pause");
          ++pauses;
          EXPECT_EQ(path, L"stable-A");
          EXPECT_EQ(sink, L"glasses");
          if (retained) {
            EXPECT_EQ(retained, retained_marker);
          }
          retained = retained_marker;
          return pause_ok;
        },
        .reactivate = [&](const auto &retained, bool) {
          events.emplace_back("reactivate");
          ++reactivations;
          EXPECT_EQ(retained, retained_marker);
          return reactivate_ok;
        },
        .query = [&](const creation_result_t &binding) {
          events.emplace_back("query");
          ++active_queries;
          EXPECT_TRUE(binding.identity);
          return observed;
        },
        .query_retirement = [&](const creation_result_t &binding) {
          events.emplace_back("query-retirement");
          ++retirement_queries;
          EXPECT_TRUE(binding.identity);
          if (retirement_observations.empty()) {
            return retirement_state;
          }
          const auto state = retirement_observations.front();
          retirement_observations.pop_front();
          return state;
        },
        .remove = [&](const GUID &guid) {
          events.emplace_back("remove");
          ++removes;
          EXPECT_EQ(guid.Data1, 11u);
          return remove_ok;
        },
        .now = [&]() { return now; },
        .sleep = [&](std::chrono::milliseconds delay) { now += delay; },
        .restore_cursor = [&](const auto &retained) {
          events.emplace_back("restore-cursor");
          ++cursor_restores;
          EXPECT_EQ(retained, retained_marker);
        },
      };
    }
  };

  TEST(VirtualDisplaySession, AcquisitionOrdersDurablePreparationBeforeAddAndExactBinding) {
    owner_probe_t fake;
    session_t owner(fake.io());
    ASSERT_TRUE(owner.acquire(fake.spec()));
    EXPECT_EQ(fake.events, (std::vector<std::string> {"prepare", "create", "bind"}));
    EXPECT_TRUE(fake.prepared_spec.exclusive);
    EXPECT_EQ(fake.prepared_spec.local_sink, L"glasses");
    EXPECT_TRUE(owner.owns_display());
    EXPECT_TRUE(owner.prepared());
    EXPECT_TRUE(owner.was_published());
    EXPECT_FALSE(owner.acquire(fake.spec()));
    EXPECT_EQ(fake.creates, 1u);
  }

  TEST(VirtualDisplaySession, ExtendedLocalPreparationUsesItsAdapterAndSkipsPrimaryBinding) {
    owner_probe_t fake;
    session_t owner(fake.io());
    auto spec = fake.spec();
    spec.primary_binding = false;
    spec.exclusive = false;
    ASSERT_TRUE(owner.acquire(spec, [&]() {
      fake.events.emplace_back("row-journal");
      return true;
    }));
    EXPECT_EQ(fake.events, (std::vector<std::string> {"row-journal", "create"}));
    EXPECT_EQ(fake.prepares, 0u);
    EXPECT_EQ(fake.binds, 0u);
  }

  TEST(VirtualDisplaySession, FailedPreparationDoesNotCreateOrRemoveADisplay) {
    owner_probe_t fake;
    fake.prepare_ok = false;
    session_t owner(fake.io());
    EXPECT_FALSE(owner.acquire(fake.spec()));
    EXPECT_FALSE(owner.owns_display());
    EXPECT_FALSE(owner.prepared());
    EXPECT_EQ(fake.creates, 0u);
    EXPECT_TRUE(owner.retire_until(fake.now));
    EXPECT_EQ(fake.restores, 0u);
    EXPECT_EQ(fake.removes, 0u);
  }

  TEST(VirtualDisplaySession, CancellationAfterPreparationRetainsBaselineWithoutAdding) {
    owner_probe_t fake;
    session_t owner(fake.io());
    bool cancelled = false;
    EXPECT_FALSE(owner.acquire(fake.spec(), [&]() {
      cancelled = true;
      return true;
    }, [&]() { return cancelled; }));
    EXPECT_TRUE(owner.prepared());
    EXPECT_EQ(fake.creates, 0u);
    fake.restore_ok = false;
    EXPECT_FALSE(owner.retire_until(fake.now));
    EXPECT_TRUE(owner.prepared());
    fake.restore_ok = true;
    EXPECT_TRUE(owner.retire_until(fake.now));
    EXPECT_TRUE(owner.retirement_complete());
    EXPECT_EQ(fake.restores, 2u);
    EXPECT_EQ(fake.active_queries, 0u);
    EXPECT_EQ(fake.retirement_queries, 0u);
    EXPECT_EQ(fake.removes, 0u);
    EXPECT_EQ(fake.now, session_t::time_point {});
  }

  TEST(VirtualDisplaySession, FailedBindKeepsTheAddedResourceUntilExplicitRetirement) {
    owner_probe_t fake;
    fake.bind_ok = false;
    session_t owner(fake.io());
    EXPECT_FALSE(owner.acquire(fake.spec()));
    EXPECT_TRUE(owner.owns_display());
    EXPECT_TRUE(owner.binding().identity);
    EXPECT_FALSE(owner.adopt(fake.spec(), fake.created));
    EXPECT_TRUE(owner.retire_until(fake.now + 1s));
    EXPECT_EQ(fake.removes, 1u);
  }

  TEST(VirtualDisplaySession, UnpublishedAddMovesWithItsIdentityAndLatePublicationQuarantine) {
    owner_probe_t fake;
    fake.created.display_name.clear();
    fake.created.device_path.clear();
    session_t original(fake.io());
    EXPECT_FALSE(original.acquire(fake.spec()));
    ASSERT_TRUE(original.owns_display());
    const auto generation = original.generation();
    session_t retired(std::move(original));
    EXPECT_FALSE(original.owns_display());
    EXPECT_FALSE(original.prepared());
    EXPECT_EQ(original.generation(), 0u);
    EXPECT_EQ(retired.generation(), generation);
    EXPECT_FALSE(retired.retire_until(fake.now + 400ms));
    EXPECT_EQ(retired.binding().device_path, L"stable-A");
    EXPECT_TRUE(retired.was_published());
    EXPECT_TRUE(retired.owns_display());
    EXPECT_TRUE(retired.retire_until(fake.now + 1s));
    EXPECT_GE(fake.now.time_since_epoch(), 850ms);
    EXPECT_EQ(fake.removes, 1u);
  }

  TEST(VirtualDisplaySession, FailedPauseRetainsSnapshotButCannotAuthorizeReactivation) {
    owner_probe_t fake;
    fake.pause_ok = false;
    session_t owner(fake.io());
    ASSERT_TRUE(owner.acquire(fake.spec()));
    EXPECT_FALSE(owner.pause());
    EXPECT_TRUE(owner.has_retained());
    EXPECT_TRUE(owner.pause_requested());
    EXPECT_FALSE(owner.paused());
    EXPECT_TRUE(owner.binding().display_name.empty());
    EXPECT_FALSE(owner.begin_resume());
    EXPECT_EQ(fake.reactivations, 0u);
    fake.pause_ok = true;
    ASSERT_TRUE(owner.begin_resume());
    EXPECT_EQ(fake.pauses, 3u);
    EXPECT_TRUE(owner.has_retained());
    EXPECT_TRUE(owner.pause_requested());
    owner.commit_resume();
    EXPECT_FALSE(owner.has_retained());
    EXPECT_FALSE(owner.pause_requested());
  }

  TEST(VirtualDisplaySession, PartialResumeInvalidatesPauseProofAndKeepsExactSnapshotForRollback) {
    owner_probe_t fake;
    session_t owner(fake.io());
    ASSERT_TRUE(owner.acquire(fake.spec()));
    ASSERT_TRUE(owner.pause());
    // A local row-cleanup retry must not repeat the already successful manager pause.
    ASSERT_TRUE(owner.pause());
    EXPECT_EQ(fake.pauses, 1u);
    fake.reactivate_ok = false;
    EXPECT_FALSE(owner.begin_resume());
    EXPECT_FALSE(owner.paused());
    EXPECT_TRUE(owner.has_retained());
    ASSERT_TRUE(owner.pause());
    EXPECT_EQ(fake.pauses, 2u);
    fake.reactivate_ok = true;
    ASSERT_TRUE(owner.begin_resume());
    EXPECT_TRUE(owner.has_retained());
    owner.commit_resume();
    EXPECT_FALSE(owner.has_retained());
  }

  TEST(VirtualDisplaySession, ReactivationLeavesAsynchronousBindingVerificationToTheAdapter) {
    owner_probe_t fake;
    session_t owner(fake.io());
    ASSERT_TRUE(owner.acquire(fake.spec()));
    ASSERT_TRUE(owner.pause());
    fake.observed = {display_identity_state_e::indeterminate};
    ASSERT_TRUE(owner.begin_resume());
    EXPECT_EQ(fake.reactivations, 1u);
    EXPECT_EQ(fake.active_queries, 0u);
    EXPECT_TRUE(owner.has_retained());
    EXPECT_TRUE(owner.binding().display_name.empty());
    EXPECT_FALSE(owner.refresh());
    EXPECT_TRUE(owner.has_retained());
    fake.observed = {display_identity_state_e::present, L"DISPLAY8", L"stable-A", L"Virtual Display"};
    ASSERT_TRUE(owner.refresh());
    EXPECT_EQ(owner.binding().display_name, L"DISPLAY8");
    owner.commit_resume();
    EXPECT_FALSE(owner.has_retained());
  }

  TEST(VirtualDisplaySession, ResumeWithoutPauseDoesNotQueryOrMutateTheBinding) {
    owner_probe_t fake;
    session_t owner(fake.io());
    ASSERT_TRUE(owner.acquire(fake.spec()));
    fake.observed = {display_identity_state_e::indeterminate};
    EXPECT_TRUE(owner.begin_resume());
    EXPECT_EQ(fake.active_queries, 0u);
    EXPECT_EQ(fake.pauses, 0u);
    EXPECT_EQ(fake.reactivations, 0u);
    EXPECT_EQ(owner.binding().display_name, L"DISPLAY7");
    owner.commit_resume();
    EXPECT_EQ(fake.cursor_restores, 0u);
  }

  TEST(VirtualDisplaySession, CursorRestoresOnceAfterSuccessfulResumePromotion) {
    owner_probe_t fake;
    session_t owner(fake.io());
    ASSERT_TRUE(owner.acquire(fake.spec()));
    ASSERT_TRUE(owner.pause());
    ASSERT_TRUE(owner.begin_resume());
    ASSERT_TRUE(owner.refresh());
    ASSERT_TRUE(owner.promote());
    EXPECT_EQ(fake.cursor_restores, 0u);
    owner.commit_resume();
    ASSERT_GE(fake.events.size(), 2u);
    EXPECT_EQ(fake.events[fake.events.size() - 2], "promote");
    EXPECT_EQ(fake.events.back(), "restore-cursor");
    owner.commit_resume();
    EXPECT_EQ(fake.cursor_restores, 1u);
  }

  TEST(VirtualDisplaySession, FailedResumeKeepsCursorSnapshotAcrossRollbackAndOwnerMove) {
    owner_probe_t fake;
    session_t original(fake.io());
    ASSERT_TRUE(original.acquire(fake.spec()));
    ASSERT_TRUE(original.pause());
    fake.reactivate_ok = false;
    EXPECT_FALSE(original.begin_resume());
    ASSERT_TRUE(original.pause());
    EXPECT_EQ(fake.cursor_restores, 0u);
    session_t owner(std::move(original));
    fake.reactivate_ok = true;
    ASSERT_TRUE(owner.begin_resume());
    ASSERT_TRUE(owner.refresh());
    ASSERT_TRUE(owner.promote());
    owner.commit_resume();
    EXPECT_EQ(fake.cursor_restores, 1u);
    EXPECT_FALSE(owner.has_retained());
  }

  TEST(VirtualDisplaySession, RetiringPausedSessionDoesNotRestoreCursor) {
    owner_probe_t fake;
    session_t owner(fake.io());
    ASSERT_TRUE(owner.acquire(fake.spec()));
    ASSERT_TRUE(owner.pause());
    owner.begin_retirement();
    owner.commit_resume();
    EXPECT_EQ(fake.cursor_restores, 0u);
    ASSERT_TRUE(owner.retire_until(fake.now + 1s));
    EXPECT_EQ(fake.cursor_restores, 0u);
  }

  TEST(VirtualDisplaySession, BindingRenumberingRequiresTheSameLearnedPhysicalIdentity) {
    owner_probe_t fake;
    session_t owner(fake.io());
    ASSERT_TRUE(owner.acquire(fake.spec()));
    auto renumbered = owner.binding();
    renumbered.identity->TargetId = 99;
    renumbered.display_name = L"DISPLAY99";
    ASSERT_TRUE(owner.update_binding(renumbered));
    EXPECT_EQ(owner.binding().identity->TargetId, 99u);
    auto replacement = renumbered;
    replacement.device_path = L"replacement-physical-monitor";
    EXPECT_FALSE(owner.update_binding(replacement));
    EXPECT_EQ(owner.binding().device_path, L"stable-A");
    fake.observed = {display_identity_state_e::indeterminate};
    EXPECT_FALSE(owner.refresh());
    EXPECT_TRUE(owner.binding().display_name.empty());
    EXPECT_EQ(owner.binding().device_path, L"stable-A");
  }

  TEST(VirtualDisplaySession, ExplicitModeAndPolicyUpdatePreservesResourceAndPublicationIdentity) {
    owner_probe_t fake;
    session_t owner(fake.io());
    ASSERT_TRUE(owner.acquire(fake.spec()));
    const auto generation = owner.generation();
    owner.set_exclusive(false);
    owner.update_mode(3840, 2160, 90000);
    EXPECT_FALSE(owner.spec().exclusive);
    EXPECT_EQ(owner.spec().width, 3840u);
    EXPECT_EQ(owner.spec().refresh_millihz, 90000u);
    EXPECT_EQ(owner.generation(), generation);
    EXPECT_EQ(fake.creates, 1u);
  }

  TEST(VirtualDisplaySession, MoveAssignmentCannotDiscardUnresolvedOwnership) {
    owner_probe_t fake;
    session_t first(fake.io());
    session_t second(fake.io());
    ASSERT_TRUE(first.acquire(fake.spec()));
    ASSERT_TRUE(second.acquire(fake.spec()));
    const auto first_generation = first.generation();
    const auto second_generation = second.generation();
    EXPECT_THROW(first = std::move(second), std::logic_error);
    EXPECT_EQ(first.generation(), first_generation);
    EXPECT_EQ(second.generation(), second_generation);
    EXPECT_NE(first_generation, second_generation);
  }

  TEST(VirtualDisplaySession, MovedFromOwnerCanBeResetAndOldCompletionCannotTouchItsSuccessor) {
    owner_probe_t fake;
    session_t active(fake.io());
    ASSERT_TRUE(active.acquire(fake.spec()));
    session_t retired(std::move(active));
    ASSERT_TRUE(retired.retire_until(fake.now + 1s));
    const auto old_generation = retired.generation();
    active = session_t(fake.io());
    ASSERT_TRUE(active.acquire(fake.spec()));
    EXPECT_NE(active.generation(), old_generation);
    const auto removes = fake.removes;
    const auto queries = fake.retirement_queries;
    EXPECT_TRUE(retired.retire_until(fake.now + 1s));
    EXPECT_EQ(fake.removes, removes);
    EXPECT_EQ(fake.retirement_queries, queries);
    EXPECT_TRUE(active.owns_display());
    EXPECT_EQ(active.binding().device_path, L"stable-A");
  }

  TEST(VirtualDisplaySession, RetirementRequiresRestoreAndNewAbsenceProofBeyondRemoveAcknowledgement) {
    owner_probe_t fake;
    session_t owner(fake.io());
    ASSERT_TRUE(owner.acquire(fake.spec()));
    fake.restore_ok = false;
    EXPECT_FALSE(owner.retire_until(fake.now + 1s));
    EXPECT_EQ(fake.removes, 0u);
    fake.restore_ok = true;
    fake.retirement_state = display_identity_state_e::present;
    EXPECT_FALSE(owner.retire_until(fake.now + 200ms));
    EXPECT_TRUE(owner.removal_accepted());
    EXPECT_TRUE(owner.owns_display());
    fake.retirement_state = display_identity_state_e::absent;
    EXPECT_TRUE(owner.retire_until(fake.now + 200ms));
    EXPECT_EQ(fake.removes, 1u);
    EXPECT_FALSE(owner.owns_display());
  }

  TEST(VirtualDisplaySession, RemovalSafetyIsRevalidatedAndPreparationRearmedBeforeRetry) {
    owner_probe_t fake;
    session_t owner(fake.io());
    ASSERT_TRUE(owner.acquire(fake.spec()));
    bool safe = false;
    unsigned preparations = 0;
    owner.begin_retirement({
      .prepare = [&](session_t &) { ++preparations; return true; },
      .before_remove = [&](session_t &) { return safe; },
    });
    EXPECT_FALSE(owner.retire_until(fake.now + 1s));
    EXPECT_EQ(fake.removes, 0u);
    safe = true;
    EXPECT_TRUE(owner.retire_until(fake.now + 1s));
    EXPECT_EQ(preparations, 2u);
    EXPECT_EQ(fake.removes, 1u);
  }

  TEST(VirtualDisplaySession, CleanupSurvivesMoveAndIsNotReplayedAfterIndeterminateFinalQuery) {
    owner_probe_t fake;
    session_t original(fake.io());
    ASSERT_TRUE(original.acquire(fake.spec()));
    auto cleanup_calls = std::make_shared<unsigned>(0);
    original.begin_retirement({
      .finish = [cleanup_calls](session_t &) { ++*cleanup_calls; return true; },
    });
    session_t owner(std::move(original));
    fake.retirement_observations = {
      display_identity_state_e::absent,
      display_identity_state_e::absent,
      display_identity_state_e::absent,
      display_identity_state_e::absent,
      display_identity_state_e::indeterminate,
    };
    EXPECT_FALSE(owner.retire_until(fake.now + 200ms));
    EXPECT_EQ(*cleanup_calls, 1u);
    EXPECT_TRUE(owner.owns_display());
    EXPECT_TRUE(owner.retire_until(fake.now + 200ms));
    EXPECT_EQ(*cleanup_calls, 1u);
    EXPECT_EQ(fake.restores, 1u);
    EXPECT_EQ(fake.removes, 1u);
    const auto queries = fake.retirement_queries;
    EXPECT_TRUE(owner.retire_until(fake.now + 1s));
    EXPECT_EQ(fake.retirement_queries, queries);
  }

  TEST(VirtualDisplaySession, RetryBackoffSurvivesWaitSlicesAndRejectedRemove) {
    owner_probe_t fake;
    fake.remove_ok = false;
    fake.retirement_state = display_identity_state_e::present;
    session_t owner(fake.io());
    ASSERT_TRUE(owner.acquire(fake.spec()));
    owner.begin_retirement({}, {.remove_retry_interval = 250ms});
    EXPECT_FALSE(owner.retire_until(fake.now + 100ms));
    EXPECT_EQ(fake.removes, 1u);
    EXPECT_FALSE(owner.retire_until(fake.now + 100ms));
    EXPECT_EQ(fake.removes, 1u);
    EXPECT_FALSE(owner.retire_until(fake.now + 150ms));
    EXPECT_EQ(fake.removes, 2u);
  }

  TEST(VirtualDisplaySession, LocalPolicyRetriesSettlingPreparationAndAcknowledgedRemovalOnLaterSlice) {
    owner_probe_t fake;
    fake.retirement_state = display_identity_state_e::present;
    session_t owner(fake.io());
    ASSERT_TRUE(owner.acquire(fake.spec()));
    unsigned preparations = 0;
    owner.begin_retirement({
      .prepare = [&](session_t &) { return ++preparations >= 3; },
    }, {
      .retry_prepare_within_slice = true,
      .retry_acknowledged_removal_each_slice = true,
    });
    EXPECT_FALSE(owner.retire_until(fake.now + 200ms));
    EXPECT_EQ(preparations, 3u);
    EXPECT_EQ(fake.removes, 1u);
    fake.retirement_state = display_identity_state_e::absent;
    EXPECT_TRUE(owner.retire_until(fake.now + 200ms));
    EXPECT_EQ(fake.removes, 2u);
  }

  TEST(VirtualDisplaySession, DestructorNeverPerformsImplicitDisplayIo) {
    owner_probe_t fake;
    std::size_t events_before_destruction;
    {
      session_t owner(fake.io());
      ASSERT_TRUE(owner.acquire(fake.spec()));
      events_before_destruction = fake.events.size();
    }
    EXPECT_EQ(fake.events.size(), events_before_destruction);
  }
}  // namespace
