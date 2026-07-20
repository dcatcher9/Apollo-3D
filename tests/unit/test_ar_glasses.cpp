#ifdef _WIN32

  #include "src/platform/windows/ar_glasses.h"
  #include "src/platform/windows/virtual_display.h"

  #include <chrono>
  #include <thread>

  #include <nlohmann/json.hpp>

  #include <gtest/gtest.h>

using namespace std::chrono_literals;

namespace {
  nlohmann::json recovery_record(std::string device_path = R"(\\?\DISPLAY#TCL03D4#test)") {
    return {
      {"device_path", std::move(device_path)},
      {"original_left", 5120},
      {"original_top", 0},
      {"original_right", 7040},
      {"original_bottom", 1080},
      {"applied_left", 5119},
      {"applied_top", 2160},
      {"applied_right", 7039},
      {"applied_bottom", 3240},
    };
  }

  nlohmann::json rect_json(LONG left, LONG top, LONG right, LONG bottom) {
    return {
      {"left", left},
      {"top", top},
      {"right", right},
      {"bottom", bottom},
    };
  }

  nlohmann::json transactional_recovery_record(
    nlohmann::json owned_rects,
    nlohmann::json pending_rect = nullptr
  ) {
    return {
      {"device_path", R"(\\?\DISPLAY#TCL03D4#test)"},
      {"original_rect", rect_json(5120, 0, 7040, 1080)},
      {"owned_rects", std::move(owned_rects)},
      {"pending_rect", std::move(pending_rect)},
    };
  }

  nlohmann::json recovery_document(nlohmann::json recoveries, int version = 3) {
    return {
      {"version", version},
      {"recoveries", std::move(recoveries)},
    };
  }
}  // namespace

TEST(ArGlassesMode, SelectsNormalForNativeTwoDimensionalMode) {
  EXPECT_EQ(
    ar_glasses::classify_mode(1920, 1080),
    ar_glasses::presentation_mode_e::normal
  );
}

TEST(ArGlassesMode, SelectsFullSbsForDoubleWidthMode) {
  EXPECT_EQ(
    ar_glasses::classify_mode(3840, 1080),
    ar_glasses::presentation_mode_e::sbs_ai
  );
}

TEST(ArGlassesMode, RejectsUnrecognizedModes) {
  EXPECT_EQ(
    ar_glasses::classify_mode(2560, 1080),
    ar_glasses::presentation_mode_e::unsupported
  );
  EXPECT_EQ(
    ar_glasses::classify_mode(3840, 2160),
    ar_glasses::presentation_mode_e::unsupported
  );
}

TEST(ArGlassesDiscovery, RecognizesSpecificModelsAndNames) {
  EXPECT_TRUE(ar_glasses::is_recognized_ar_display("DISPLAY:TCL03D4", "Generic Monitor"));
  EXPECT_TRUE(ar_glasses::is_recognized_ar_display("DISPLAY:ABC1234", "XREAL Air 2 Pro"));
  EXPECT_TRUE(ar_glasses::is_recognized_ar_display("DISPLAY:ABC1234", "SmartGlasses"));
}

TEST(ArGlassesDiscovery, DoesNotGuessFromOrdinaryMonitorNames) {
  EXPECT_FALSE(ar_glasses::is_recognized_ar_display("DISPLAY:SMKD1CE", "Apollo AR Des"));
  EXPECT_FALSE(ar_glasses::is_recognized_ar_display("DISPLAY:GSM1234", "LG ULTRAGEAR"));
  EXPECT_FALSE(ar_glasses::is_recognized_ar_display("DISPLAY:AUS4321", "ROG PG32UCDM"));
  EXPECT_FALSE(ar_glasses::is_recognized_ar_display("DISPLAY:ACI9999", "ARZOPA Portable Monitor"));
}

TEST(ArGlassesOwnership, RenewedRemoteConnectWindowBlocksLocalPresentation) {
  constexpr ar_glasses::remote_virtual_display_lease_t lease = 1001;
  ar_glasses::remote_virtual_display_ended(lease);
  ASSERT_TRUE(ar_glasses::remote_virtual_display_starting(lease, 0ms));

  // This is called after display creation, encoder probing, and app preparation. Its fresh lease
  // must remain visible to the local topology controller until RTSP activates or the lease ends.
  ar_glasses::remote_virtual_display_awaiting_client(lease, 0ms);
  EXPECT_TRUE(ar_glasses::remote_virtual_display_blocks_local());

  ar_glasses::remote_virtual_display_ended(lease);
  EXPECT_FALSE(ar_glasses::remote_virtual_display_blocks_local());
}

TEST(ArGlassesOwnership, ProcessSetupPinsLeasePastTheInitialConnectWindow) {
  constexpr ar_glasses::remote_virtual_display_lease_t lease = 1051;
  ar_glasses::remote_virtual_display_ended(lease);
  ASSERT_TRUE(ar_glasses::remote_virtual_display_starting(lease, 0ms, true));

  // A zero connect timeout still has the production two-second handshake floor. Process setup can
  // legitimately exceed it while probing encoders or preparing an app, so the setup pin must keep
  // the lease alive until the post-setup renewal starts the ordinary bounded client window.
  std::this_thread::sleep_for(2200ms);
  EXPECT_TRUE(ar_glasses::remote_virtual_display_blocks_local());
  ar_glasses::remote_virtual_display_awaiting_client(lease, 0ms);
  EXPECT_TRUE(ar_glasses::remote_virtual_display_active(lease));

  ar_glasses::remote_virtual_display_ended(lease);
  EXPECT_FALSE(ar_glasses::remote_virtual_display_blocks_local());
}

TEST(ArGlassesOwnership, OlderLifecycleCannotClearNewerReconnectLease) {
  constexpr ar_glasses::remote_virtual_display_lease_t old_lease = 1101;
  constexpr ar_glasses::remote_virtual_display_lease_t reconnect_lease = 1102;
  ASSERT_TRUE(ar_glasses::remote_virtual_display_starting(old_lease, 0ms));
  ASSERT_TRUE(ar_glasses::remote_virtual_display_starting(reconnect_lease, 0ms));

  ar_glasses::remote_virtual_display_ended(old_lease);
  EXPECT_TRUE(ar_glasses::remote_virtual_display_blocks_local());
  EXPECT_TRUE(ar_glasses::remote_virtual_display_active(reconnect_lease));

  ar_glasses::remote_virtual_display_ended(old_lease);
  EXPECT_TRUE(ar_glasses::remote_virtual_display_blocks_local());
  ar_glasses::remote_virtual_display_ended(reconnect_lease);
  EXPECT_FALSE(ar_glasses::remote_virtual_display_blocks_local());
}

TEST(ArGlassesOwnership, ReservationTracksLongConfiguredPingTimeout) {
  EXPECT_EQ(
    ar_glasses::detail::remote_pending_duration(120s),
    122s
  );
  EXPECT_EQ(
    ar_glasses::detail::remote_pending_duration(-1ms),
    2s
  );
}

TEST(ArGlassesAdapterContract, TracksThePhysicalOutputAcrossTopologyChanges) {
  const LUID physical_before {.LowPart = 0x12345678u, .HighPart = 0x1234};
  const LUID same_physical_after {.LowPart = 0x12345678u, .HighPart = 0x1234};
  const LUID different_virtual_adapter {.LowPart = 0x87654321u, .HighPart = 0x4321};

  // The virtual target may identify SudoVDA rather than its render GPU. Only the stable physical
  // target before/after topology mutation is authoritative.
  EXPECT_TRUE(ar_glasses::detail::physical_adapter_contract_valid_for_test(
    physical_before,
    same_physical_after,
    different_virtual_adapter
  ));
}

TEST(ArGlassesAdapterContract, RejectsAPhysicalOutputThatMigratedAdapters) {
  const LUID physical_before {.LowPart = 0x12345678u, .HighPart = 0x1234};
  const LUID physical_after {.LowPart = 0x87654321u, .HighPart = 0x4321};
  const LUID virtual_adapter = physical_after;

  // Matching the new physical adapter to SudoVDA must not hide a real physical-output migration.
  EXPECT_FALSE(ar_glasses::detail::physical_adapter_contract_valid_for_test(
    physical_before,
    physical_after,
    virtual_adapter
  ));
}

TEST(ArGlassesLinearLayout, PlacesSourceThenSinkAfterTheRightmostInteractiveOutput) {
  const RECT anchor {-2560, 300, 1280, 1740};

  const auto layout = ar_glasses::detail::compute_linear_layout_for_test(
    anchor,
    1920,
    1080,
    3840,
    1080
  );

  EXPECT_EQ(layout.virtual_rect.left, 1280);
  EXPECT_EQ(layout.virtual_rect.top, 300);
  EXPECT_EQ(layout.virtual_rect.right, 3200);
  EXPECT_EQ(layout.virtual_rect.bottom, 1380);
  EXPECT_EQ(layout.physical_rect.left, 3200);
  EXPECT_EQ(layout.physical_rect.top, 300);
  EXPECT_EQ(layout.physical_rect.right, 7040);
  EXPECT_EQ(layout.physical_rect.bottom, 1380);
}

TEST(ArGlassesLinearLayout, DependsOnTheAnchorRatherThanAbsoluteDesktopCoordinates) {
  const RECT first_anchor {0, 0, 2560, 1440};
  const RECT shifted_anchor {-4000, -700, -1440, 740};

  const auto first = ar_glasses::detail::compute_linear_layout_for_test(
    first_anchor,
    1920,
    1080,
    1920,
    1080
  );
  const auto shifted = ar_glasses::detail::compute_linear_layout_for_test(
    shifted_anchor,
    1920,
    1080,
    1920,
    1080
  );

  EXPECT_EQ(first.virtual_rect.left, first_anchor.right);
  EXPECT_EQ(first.physical_rect.left, first.virtual_rect.right);
  EXPECT_EQ(shifted.virtual_rect.left, shifted_anchor.right);
  EXPECT_EQ(shifted.physical_rect.left, shifted.virtual_rect.right);
  EXPECT_EQ(shifted.virtual_rect.top, shifted_anchor.top);
  EXPECT_EQ(shifted.physical_rect.top, shifted_anchor.top);
}

TEST(ArGlassesLinearLayout, SelectsAnchorDeterministicallyWhenRightEdgesTie) {
  const ar_glasses::detail::anchor_candidate_t upper {
    .rect = {0, -1080, 1920, 0},
    .source_adapter_id = {.LowPart = 2, .HighPart = 0},
    .source_id = 2,
  };
  const ar_glasses::detail::anchor_candidate_t lower {
    .rect = {0, 0, 1920, 1080},
    .source_adapter_id = {.LowPart = 1, .HighPart = 0},
    .source_id = 1,
  };

  const auto forward = ar_glasses::detail::select_anchor_for_test({lower, upper});
  const auto reversed = ar_glasses::detail::select_anchor_for_test({upper, lower});
  ASSERT_TRUE(forward);
  ASSERT_TRUE(reversed);
  EXPECT_EQ(forward->rect.top, -1080);
  EXPECT_EQ(reversed->rect.top, -1080);
}

TEST(ArGlassesLinearLayout, RejectsAnyCloneOfThePhysicalSource) {
  const LUID adapter {.LowPart = 7, .HighPart = 0};
  const std::vector<ar_glasses::detail::topology_path_identity_t> paths {
    {adapter, 3, adapter, 10},
    {adapter, 3, adapter, 11},
    {adapter, 4, adapter, 12},
  };

  EXPECT_TRUE(ar_glasses::detail::source_is_cloned_for_test(paths, 0));
  EXPECT_TRUE(ar_glasses::detail::source_is_cloned_for_test(paths, 1));
  EXPECT_FALSE(ar_glasses::detail::source_is_cloned_for_test(paths, 2));
}

TEST(ArGlassesTopologyEvidence, RejectsMissingOrStalePrimarySourceEvidence) {
  const std::vector<std::wstring> active_sources {LR"(\\.\DISPLAY1)", LR"(\\.\DISPLAY4)"};

  EXPECT_TRUE(ar_glasses::detail::primary_source_is_authoritative_for_test(
    LR"(\\.\DISPLAY1)",
    active_sources
  ));
  EXPECT_FALSE(ar_glasses::detail::primary_source_is_authoritative_for_test(L"", active_sources));
  EXPECT_FALSE(ar_glasses::detail::primary_source_is_authoritative_for_test(
    LR"(\\.\DISPLAY2)",
    active_sources
  ));
}

TEST(ArGlassesLinearLayout, RequiresBothExactRectanglesAndSharedEdge) {
  const ar_glasses::detail::linear_layout_t expected {
    .virtual_rect = {1920, 0, 3840, 1080},
    .physical_rect = {3840, 0, 7680, 1080},
  };
  EXPECT_TRUE(ar_glasses::detail::isolated_layout_matches_for_test(
    expected,
    expected.virtual_rect,
    expected.physical_rect
  ));

  RECT normalized_virtual = expected.virtual_rect;
  normalized_virtual.top = 1;
  normalized_virtual.bottom = 1081;
  EXPECT_FALSE(ar_glasses::detail::isolated_layout_matches_for_test(
    expected,
    normalized_virtual,
    expected.physical_rect
  ));
}

TEST(ArGlassesTopologyRecovery, SerializesAPlainRecordArray) {
  auto recoveries = nlohmann::json::array();
  auto owned = nlohmann::json::array();
  owned.emplace_back(rect_json(7040, 0, 8960, 1080));
  recoveries.emplace_back(transactional_recovery_record(std::move(owned)));

  const auto result = ar_glasses::detail::parse_topology_recovery_json_for_test(
    recovery_document(std::move(recoveries), 4).dump()
  );

  ASSERT_TRUE(result.valid);
  EXPECT_FALSE(result.rewrite_required);
  EXPECT_EQ(result.record_count, 1u);
  const auto normalized = nlohmann::json::parse(result.normalized_json);
  EXPECT_EQ(normalized.at("version"), 4);
  ASSERT_TRUE(normalized.at("recoveries").is_array());
  ASSERT_EQ(normalized.at("recoveries").size(), 1u);
  EXPECT_TRUE(normalized.at("recoveries").front().is_object());
}

TEST(ArGlassesTopologyRecovery, MigratesLegacyVersionTwoRecord) {
  auto legacy = recovery_record();
  legacy["version"] = 2;

  const auto result = ar_glasses::detail::parse_topology_recovery_json_for_test(legacy.dump());

  ASSERT_TRUE(result.valid);
  EXPECT_TRUE(result.rewrite_required);
  EXPECT_EQ(result.record_count, 1u);
  const auto normalized = nlohmann::json::parse(result.normalized_json);
  EXPECT_EQ(normalized.at("version"), 4);
  ASSERT_TRUE(normalized.at("recoveries").is_array());
  ASSERT_EQ(normalized.at("recoveries").size(), 1u);
  EXPECT_TRUE(normalized.at("recoveries").front().is_object());
}

TEST(ArGlassesTopologyRecovery, PendingMoveOwnsOnlyExactPendingAndConfirmedPositions) {
  auto owned = nlohmann::json::array();
  owned.emplace_back(rect_json(7040, 0, 8960, 1080));
  owned.emplace_back(rect_json(7040, -1080, 8960, 0));
  auto recoveries = nlohmann::json::array();
  recoveries.emplace_back(transactional_recovery_record(
    std::move(owned),
    rect_json(8960, 0, 10880, 1080)
  ));
  const auto document = recovery_document(std::move(recoveries), 4);

  const auto parsed = ar_glasses::detail::parse_topology_recovery_json_for_test(document.dump());
  ASSERT_TRUE(parsed.valid);
  const auto normalized = nlohmann::json::parse(parsed.normalized_json);
  const auto &record = normalized.at("recoveries").front();
  EXPECT_EQ(record.at("owned_rects").size(), 2u);
  EXPECT_FALSE(record.at("pending_rect").is_null());

  const RECT exact_pending {8960, 0, 10880, 1080};
  EXPECT_TRUE(ar_glasses::detail::topology_recovery_should_restore_for_test(
    document.dump(),
    exact_pending
  ));
  const RECT prior_confirmed {7040, -1080, 8960, 0};
  EXPECT_TRUE(ar_glasses::detail::topology_recovery_should_restore_for_test(
    document.dump(),
    prior_confirmed
  ));

  // Apollo may stop after writing the pending marker but before SetDisplayConfig. A different
  // rectangle can therefore be a later user move and must not be claimed by the transaction.
  const RECT unrelated_user_position {8500, 200, 10420, 1280};
  EXPECT_FALSE(ar_glasses::detail::topology_recovery_should_restore_for_test(
    document.dump(),
    unrelated_user_position
  ));
}

TEST(ArGlassesTopologyRecovery, DoesNotOwnAnUnrecognizedRectWithoutAPendingMove) {
  auto owned = nlohmann::json::array();
  owned.emplace_back(rect_json(7040, 0, 8960, 1080));
  auto recoveries = nlohmann::json::array();
  recoveries.emplace_back(transactional_recovery_record(std::move(owned)));
  const auto document = recovery_document(std::move(recoveries), 4);
  const RECT user_position {100, 100, 2020, 1180};

  EXPECT_FALSE(ar_glasses::detail::topology_recovery_should_restore_for_test(
    document.dump(),
    user_position
  ));
}

TEST(ArGlassesTopologyRecovery, RepairsTheKnownLeadingEmptyArrayArtifact) {
  auto recoveries = nlohmann::json::array();
  recoveries.emplace_back(nlohmann::json::array());
  recoveries.emplace_back(recovery_record());

  const auto result = ar_glasses::detail::parse_topology_recovery_json_for_test(
    recovery_document(std::move(recoveries)).dump()
  );

  ASSERT_TRUE(result.valid);
  EXPECT_TRUE(result.rewrite_required);
  EXPECT_EQ(result.record_count, 1u);
  const auto normalized = nlohmann::json::parse(result.normalized_json);
  ASSERT_EQ(normalized.at("recoveries").size(), 1u);
  EXPECT_TRUE(normalized.at("recoveries").front().is_object());
}

TEST(ArGlassesTopologyRecovery, RejectsOtherMalformedRecoveryLists) {
  auto sentinel_only = nlohmann::json::array();
  sentinel_only.emplace_back(nlohmann::json::array());

  auto trailing_array = nlohmann::json::array();
  trailing_array.emplace_back(recovery_record());
  trailing_array.emplace_back(nlohmann::json::array());

  auto repeated_sentinel = nlohmann::json::array();
  repeated_sentinel.emplace_back(nlohmann::json::array());
  repeated_sentinel.emplace_back(nlohmann::json::array());
  repeated_sentinel.emplace_back(recovery_record());

  auto null_entry = nlohmann::json::array();
  null_entry.emplace_back(nullptr);
  null_entry.emplace_back(recovery_record());

  auto duplicate_devices = nlohmann::json::array();
  duplicate_devices.emplace_back(recovery_record());
  duplicate_devices.emplace_back(recovery_record());

  auto invalid_rectangle = recovery_record();
  invalid_rectangle["applied_right"] = invalid_rectangle["applied_left"];
  auto malformed_record = nlohmann::json::array();
  malformed_record.emplace_back(std::move(invalid_rectangle));

  for (auto &invalid : {
         std::move(sentinel_only),
         std::move(trailing_array),
         std::move(repeated_sentinel),
         std::move(null_entry),
         std::move(duplicate_devices),
         std::move(malformed_record),
       }) {
    EXPECT_FALSE(ar_glasses::detail::parse_topology_recovery_json_for_test(
                   recovery_document(std::move(invalid)).dump()
                 ).valid);
  }
}

TEST(ArGlassesModeTransition, KeepsVirtualDesktopForSupportedModesOnSameOutput) {
  ar_glasses::detail::local_session_contract_t before;
  before.device_path = LR"(\\?\DISPLAY#TCL03D4#test)";
  before.adapter_id.LowPart = 42;
  before.adapter_id.HighPart = 7;
  before.mode = ar_glasses::presentation_mode_e::normal;
  before.hdr_known = true;

  auto after = before;
  after.mode = ar_glasses::presentation_mode_e::sbs_ai;

  EXPECT_TRUE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, after));
  EXPECT_TRUE(ar_glasses::detail::local_session_can_reconfigure_for_test(after, before));
}

TEST(ArGlassesModeTransition, RebuildsOnlyForOutputAdapterOrUnsupportedChanges) {
  ar_glasses::detail::local_session_contract_t before;
  before.device_path = LR"(\\?\DISPLAY#TCL03D4#test)";
  before.adapter_id.LowPart = 42;
  before.mode = ar_glasses::presentation_mode_e::normal;
  before.hdr_known = true;

  auto changed = before;
  changed.device_path = LR"(\\?\DISPLAY#OTHER#test)";
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed));

  changed = before;
  changed.adapter_id.LowPart = 43;
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed));

  changed = before;
  changed.hdr_active = true;
  EXPECT_TRUE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed));

  changed = before;
  changed.hdr_known = false;
  EXPECT_TRUE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed));

  changed = before;
  changed.mode = ar_glasses::presentation_mode_e::unsupported;
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed));

  changed = before;
  changed.is_primary = true;
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed));

  changed = before;
  changed.is_cloned = true;
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed));
}

TEST(ArGlassesModeTransition, RetirementDoesNotFollowAReusedPhysicalTargetId) {
  ar_glasses::detail::virtual_display_identity_contract_t retiring;
  retiring.adapter_id.LowPart = 42;
  retiring.target_id = 7;
  retiring.device_path = LR"(\\?\DISPLAY#SUDOVDA#retiring)";
  retiring.gdi_name = LR"(\\.\DISPLAY5)";

  auto reused_physical = retiring;
  reused_physical.device_path = LR"(\\?\DISPLAY#TCL03D4#physical)";
  reused_physical.gdi_name = LR"(\\.\DISPLAY2)";
  reused_physical.friendly_name = L"SmartGlasses";
  EXPECT_FALSE(ar_glasses::detail::retirement_identity_matches_for_test(
    retiring,
    reused_physical
  ));

  auto renumbered_virtual = reused_physical;
  renumbered_virtual.adapter_id.LowPart = 43;
  renumbered_virtual.target_id = 9;
  renumbered_virtual.friendly_name = L"Apollo AR Desktop";
  EXPECT_FALSE(ar_glasses::detail::retirement_identity_matches_for_test(
    retiring,
    renumbered_virtual
  ));

  auto exact_sudo_identity = reused_physical;
  exact_sudo_identity.device_path = LR"(\\?\DISPLAY#SMKD1CE#replacement-path)";
  EXPECT_TRUE(ar_glasses::detail::retirement_identity_matches_for_test(
    retiring,
    exact_sudo_identity
  ));

  auto exact_apollo_identity = reused_physical;
  exact_apollo_identity.friendly_name = L"Apollo AR Desktop";
  EXPECT_TRUE(ar_glasses::detail::retirement_identity_matches_for_test(
    retiring,
    exact_apollo_identity
  ));

  auto learned_path = reused_physical;
  learned_path.device_path = retiring.device_path;
  EXPECT_TRUE(ar_glasses::detail::retirement_identity_matches_for_test(retiring, learned_path));
}

TEST(ArGlassesModeTransition, RecognizesTheProductionSudoVirtualDisplayHardwarePath) {
  EXPECT_TRUE(VDISPLAY::isSudoVirtualDisplayPathForTest(
    LR"(\\?\DISPLAY#SMKD1CE#5&production&0&UID4352)"
  ));
  EXPECT_TRUE(VDISPLAY::isSudoVirtualDisplayPathForTest(
    LR"(\\?\DISPLAY#SUDOVDA#legacy)"
  ));
  EXPECT_FALSE(VDISPLAY::isSudoVirtualDisplayPathForTest(
    LR"(\\?\DISPLAY#TCL03D4#physical)"
  ));
}

TEST(ArGlassesModeTransition, RebasesOwnedRecoveryRectsToNewPhysicalWidth) {
  auto owned = nlohmann::json::array();
  owned.emplace_back(rect_json(7040, 0, 8960, 1080));
  auto recoveries = nlohmann::json::array();
  recoveries.emplace_back(transactional_recovery_record(
    std::move(owned),
    rect_json(7040, -1080, 8960, 0)
  ));
  const auto document = recovery_document(std::move(recoveries), 4);
  const RECT previous_original {5120, 0, 7040, 1080};
  const RECT expanded_owned_position {7040, 0, 10880, 1080};

  const auto rebased = ar_glasses::detail::rebase_topology_recovery_json_for_test(
    document.dump(),
    previous_original,
    expanded_owned_position
  );

  ASSERT_TRUE(rebased);
  const auto normalized = nlohmann::json::parse(*rebased);
  const auto &record = normalized.at("recoveries").front();
  EXPECT_EQ(record.at("original_rect"), rect_json(5120, 0, 8960, 1080));
  ASSERT_EQ(record.at("owned_rects").size(), 1u);
  EXPECT_EQ(record.at("owned_rects").front(), rect_json(7040, 0, 10880, 1080));
  EXPECT_EQ(record.at("pending_rect"), rect_json(7040, -1080, 10880, 0));
}

TEST(ArGlassesModeTransition, PreservesTheAttachedRightEdgeForALeftSideOriginal) {
  auto owned = nlohmann::json::array();
  owned.emplace_back(rect_json(7040, 0, 8960, 1080));
  auto recoveries = nlohmann::json::array();
  recoveries.emplace_back(transactional_recovery_record(std::move(owned)));
  auto document = recovery_document(std::move(recoveries), 4);
  document["recoveries"].front()["original_rect"] = rect_json(-1920, 0, 0, 1080);
  const RECT previous_original {-1920, 0, 0, 1080};
  const RECT expanded_owned_position {7040, 0, 10880, 1080};

  const auto rebased = ar_glasses::detail::rebase_topology_recovery_json_for_test(
    document.dump(),
    previous_original,
    expanded_owned_position
  );

  ASSERT_TRUE(rebased);
  const auto normalized = nlohmann::json::parse(*rebased);
  const auto &record = normalized.at("recoveries").front();
  EXPECT_EQ(record.at("original_rect"), rect_json(-3840, 0, 0, 1080));
  EXPECT_EQ(record.at("owned_rects").front(), rect_json(7040, 0, 10880, 1080));
}

TEST(ArGlassesModeTransition, RebasesOwnershipForAnUnsupportedSameOriginWidth) {
  auto owned = nlohmann::json::array();
  owned.emplace_back(rect_json(7040, 0, 8960, 1080));
  auto recoveries = nlohmann::json::array();
  recoveries.emplace_back(transactional_recovery_record(std::move(owned)));
  const auto document = recovery_document(std::move(recoveries), 4);
  const RECT previous_original {5120, 0, 7040, 1080};
  const RECT unsupported_owned_position {7040, 0, 9600, 1080};

  const auto rebased = ar_glasses::detail::rebase_topology_recovery_json_for_test(
    document.dump(),
    previous_original,
    unsupported_owned_position
  );

  ASSERT_TRUE(rebased);
  const auto normalized = nlohmann::json::parse(*rebased);
  const auto &record = normalized.at("recoveries").front();
  EXPECT_EQ(record.at("original_rect"), rect_json(5120, 0, 7680, 1080));
  EXPECT_EQ(record.at("owned_rects").front(), rect_json(7040, 0, 9600, 1080));
}

TEST(ArGlassesModeTransition, DoesNotClaimAUserMovedRectDuringModeChange) {
  auto owned = nlohmann::json::array();
  owned.emplace_back(rect_json(7040, 0, 8960, 1080));
  auto recoveries = nlohmann::json::array();
  recoveries.emplace_back(transactional_recovery_record(std::move(owned)));
  const auto document = recovery_document(std::move(recoveries), 4);
  const RECT previous_original {5120, 0, 7040, 1080};
  const RECT user_moved_expanded {8000, 200, 11840, 1280};

  EXPECT_FALSE(ar_glasses::detail::rebase_topology_recovery_json_for_test(
    document.dump(),
    previous_original,
    user_moved_expanded
  ));
}

#endif
