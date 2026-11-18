#ifdef _WIN32
  #include "src/platform/windows/ar_glasses_discovery_test.h"
  #include "src/platform/windows/misc.h"
  #include "src/config.h"

  #include <algorithm>
  #include <array>
  #include <atomic>
  #include <chrono>
  #include <cwchar>
  #include <filesystem>
  #include <fstream>
  #include <gtest/gtest.h>
  #include <iterator>
  #include <limits>
  #include <memory>
  #include <nlohmann/json.hpp>
  #include <optional>
  #include <stdexcept>
  #include <string>
  #include <vector>

namespace {
  namespace fs = std::filesystem;
  namespace discovery = ar_glasses::testing;
  using json = nlohmann::json;

  struct monitor_t {
    std::wstring path;
    std::wstring friendly;
    std::wstring gdi;
    LUID source_adapter {101, 11};
    UINT32 source_id = 1;
    LUID target_adapter {201, 21};
    UINT32 target_id = 11;
    LONG x = 0;
    LONG y = 0;
    UINT32 width = 1920;
    UINT32 height = 1080;
    UINT32 refresh_numerator = 60000;
    UINT32 refresh_denominator = 1000;
    LONG modern_status = ERROR_SUCCESS;
    LONG legacy_status = ERROR_SUCCESS;
    bool hdr_supported = true;
    bool hdr_user_enabled = true;
    bool hdr_active = true;
    bool hdr_limited = false;
    UINT32 bits = 10;
  };

  monitor_t monitor(std::wstring model, UINT32 index, std::wstring friendly = L"Office Monitor") {
    monitor_t result;
    result.path = L"\\\\?\\DISPLAY#" + model + L"#" + std::to_wstring(index);
    result.friendly = std::move(friendly);
    result.gdi = L"\\\\.\\DISPLAY" + std::to_wstring(index);
    result.source_id = index;
    result.target_id = index + 10;
    result.x = static_cast<LONG>((index - 1) * 1920);
    return result;
  }

  template<std::size_t N>
  void copy_wide(WCHAR (&destination)[N], const std::wstring &source) {
    EXPECT_LT(source.size(), N);
    std::wcsncpy(destination, source.c_str(), N - 1);
    destination[N - 1] = L'\0';
  }

  bool same_luid(const LUID &left, const LUID &right) {
    return left.LowPart == right.LowPart && left.HighPart == right.HighPart;
  }

  json target_json(const discovery::discovery_target_t &target) {
    return {
      {"device_id", target.device_id},
      {"device_path", platf::to_utf8(target.device_path)},
      {"friendly_name", target.friendly_name},
      {"gdi_name", target.gdi_name},
      {"desktop_topology", target.desktop_topology},
      {"source_adapter", {target.source_adapter_id.LowPart, target.source_adapter_id.HighPart}},
      {"source_id", target.source_id},
      {"target_adapter", {target.adapter_id.LowPart, target.adapter_id.HighPart}},
      {"target_id", target.target_id},
      {"hdr", {
                {"known", target.hdr.known},
                {"supported", target.hdr.supported},
                {"user_enabled", target.hdr.user_enabled},
                {"active", target.hdr.active},
                {"limited_by_policy", target.hdr.limited_by_policy},
                {"bits_per_color", target.hdr.bits_per_color},
              }},
      {"rect", {target.rect.left, target.rect.top, target.rect.right, target.rect.bottom}},
      {"refresh_millihz", target.refresh_millihz},
      {"mode", static_cast<int>(target.mode)},
      {"is_primary", target.is_primary},
      {"is_cloned", target.is_cloned},
    };
  }

  json device_json() {
    auto result = json::array();
    for (const auto &device : ar_glasses::devices()) {
      result.push_back({
        {"id", device.id},
        {"name", device.name},
        {"decision", static_cast<int>(device.decision)},
        {"connected", device.connected},
        {"auto_detected", device.auto_detected},
      });
    }
    return result;
  }

  class ArTargetDiscoveryTest: public testing::Test {
  protected:
    void SetUp() override {
      static std::atomic<unsigned> next {0};
      temp_base_ = fs::weakly_canonical(fs::temp_directory_path());
      root_ = temp_base_ / ("sunshine3d-ar-discovery-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" + std::to_string(++next));
      ASSERT_TRUE(fs::create_directory(root_));
      config_path_ = root_ / "sunshine.conf";
      write_config();
    }

    void TearDown() override {
      session_.reset();
      std::error_code error;
      const auto resolved = fs::weakly_canonical(root_, error);
      if (error || resolved.parent_path() != temp_base_ || !resolved.filename().string().starts_with("sunshine3d-ar-discovery-")) {
        ADD_FAILURE() << "Refusing recursive cleanup outside the exact owned temporary child";
        return;
      }
      fs::remove_all(resolved, error);
      EXPECT_FALSE(error) << error.message();
    }

    void write_config(const json &decisions = json::array()) {
      std::ofstream output(config_path_, std::ios::binary | std::ios::trunc);
      output << "# owned discovery fixture\r\nkeep = unchanged\r\nar_glass_devices = " << decisions.dump() << "\r\n";
      ASSERT_TRUE(output.good());
    }

    std::string read_config() const {
      std::ifstream input(config_path_, std::ios::binary);
      return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    json persisted_devices() const {
      return json::parse(config::parse_config(read_config()).at("ar_glass_devices"));
    }

    void check_native_lock() {
      EXPECT_TRUE(discovery::discovery_mutex_available_for_test());
    }

    discovery::discovery_hooks_t hooks() {
      return {
        .query_active = [this](auto &paths, auto &modes) {
          check_native_lock();
          ++queries_;
          if (!query_ok_) {
            return false;
          }
          for (const auto &row : rows_) {
            DISPLAYCONFIG_PATH_INFO path {};
            path.flags = DISPLAYCONFIG_PATH_ACTIVE;
            path.sourceInfo.adapterId = row.source_adapter;
            path.sourceInfo.id = row.source_id;
            path.sourceInfo.modeInfoIdx = static_cast<UINT32>(modes.size());
            path.targetInfo.adapterId = row.target_adapter;
            path.targetInfo.id = row.target_id;
            path.targetInfo.refreshRate = {row.refresh_numerator, row.refresh_denominator};
            paths.push_back(path);
            DISPLAYCONFIG_MODE_INFO mode {};
            mode.infoType = DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE;
            mode.id = row.source_id;
            mode.adapterId = row.source_adapter;
            mode.sourceMode.position = {row.x, row.y};
            mode.sourceMode.width = row.width;
            mode.sourceMode.height = row.height;
            modes.push_back(mode);
          }
          if (corrupt_snapshot_) {
            corrupt_snapshot_(paths, modes);
          }
          return true;
        },
        .match_display = [this](std::wstring name) {
          check_native_lock();
          ++match_display_queries_;
          EXPECT_EQ(name, L"SudoMaker Virtual Display Adapter");
          return virtual_sources_;
        },
        .get_info = [this](DISPLAYCONFIG_DEVICE_INFO_HEADER *header) -> LONG {
          check_native_lock();
          EXPECT_NE(header, nullptr);
          if (!header) {
            return ERROR_INVALID_PARAMETER;
          }
          const bool source = header->type == DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
          const auto row = std::find_if(rows_.begin(), rows_.end(), [&](const auto &candidate) {
            return same_luid(header->adapterId, source ? candidate.source_adapter : candidate.target_adapter) && header->id == (source ? candidate.source_id : candidate.target_id);
          });
          if (row == rows_.end()) {
            ADD_FAILURE() << "Unexpected native query identity " << header->id;
            return ERROR_NOT_FOUND;
          }
          if (source) {
            ++source_name_queries_;
            EXPECT_EQ(header->size, sizeof(DISPLAYCONFIG_SOURCE_DEVICE_NAME));
            if (failed_source_ == header->id) {
              return ERROR_GEN_FAILURE;
            }
            auto *info = reinterpret_cast<DISPLAYCONFIG_SOURCE_DEVICE_NAME *>(header);
            copy_wide(info->viewGdiDeviceName, row->gdi);
            return ERROR_SUCCESS;
          }
          if (header->type == DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME) {
            ++target_name_queries_;
            EXPECT_EQ(header->size, sizeof(DISPLAYCONFIG_TARGET_DEVICE_NAME));
            if (failed_target_ == header->id) {
              return ERROR_GEN_FAILURE;
            }
            auto *info = reinterpret_cast<DISPLAYCONFIG_TARGET_DEVICE_NAME *>(header);
            copy_wide(info->monitorDevicePath, row->path);
            copy_wide(info->monitorFriendlyDeviceName, row->friendly);
            info->edidManufactureId = 0x1234;
            info->edidProductCodeId = 0xABCD;
            return ERROR_SUCCESS;
          }
          if (header->type == DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO_2) {
            EXPECT_EQ(header->size, sizeof(DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO_2));
            ++modern_queries_;
            if (row->modern_status != ERROR_SUCCESS) {
              return row->modern_status;
            }
            auto *info = reinterpret_cast<DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO_2 *>(header);
            info->highDynamicRangeSupported = row->hdr_supported;
            info->highDynamicRangeUserEnabled = row->hdr_user_enabled;
            info->advancedColorActive = row->hdr_active;
            info->advancedColorLimitedByPolicy = row->hdr_limited;
            info->bitsPerColorChannel = row->bits;
            info->activeColorMode = row->hdr_active ? DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR : DISPLAYCONFIG_ADVANCED_COLOR_MODE_SDR;
            return ERROR_SUCCESS;
          }
          if (header->type == DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO) {
            EXPECT_EQ(header->size, sizeof(DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO));
            ++legacy_queries_;
            if (row->legacy_status != ERROR_SUCCESS) {
              return row->legacy_status;
            }
            auto *info = reinterpret_cast<DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO *>(header);
            info->advancedColorSupported = row->hdr_supported;
            info->advancedColorEnabled = row->hdr_active;
            info->advancedColorForceDisabled = row->hdr_limited;
            info->bitsPerColorChannel = row->bits;
            return ERROR_SUCCESS;
          }
          ADD_FAILURE() << "Unexpected native request type " << header->type;
          return ERROR_INVALID_PARAMETER;
        },
        .enum_devices = [this](LPCWSTR name, DWORD index, PDISPLAY_DEVICEW device, DWORD flags) -> BOOL {
          check_native_lock();
          ++enum_devices_queries_;
          EXPECT_EQ(name, nullptr);
          EXPECT_EQ(flags, 0u);
          EXPECT_NE(device, nullptr);
          if (!device) {
            return FALSE;
          }
          EXPECT_EQ(device->cb, sizeof(DISPLAY_DEVICEW));
          if (index != 0 || primary_.empty()) {
            return FALSE;
          }
          copy_wide(device->DeviceName, primary_);
          device->StateFlags = DISPLAY_DEVICE_PRIMARY_DEVICE;
          return TRUE;
        },
        .notify = [this](const std::string &name) {
          check_native_lock();
          notifications_.push_back(name);
          return accept_notification_;
        },
      };
    }

    void start(discovery::discovery_request_t request = {}) {
      ASSERT_FALSE(session_);
      session_ = std::make_unique<discovery::discovery_session_t>(config_path_, hooks(), std::move(request));
    }

    discovery::discovery_result_t find(std::wstring_view required = {}, std::wstring_view preferred = {}, bool collect = true) {
      auto result = session_->find(required, preferred, collect);
      EXPECT_TRUE(result.approved_prefix_preserved);
      return result;
    }

    void record_costs() const {
      const auto costs = session_->costs();
      RecordProperty("hdr_queries", std::to_string(costs.hdr_queries));
      RecordProperty("fingerprints", std::to_string(costs.fingerprints));
      RecordProperty("persistence_attempts", std::to_string(costs.persistence_attempts));
    }

    fs::path temp_base_;
    fs::path root_;
    fs::path config_path_;
    std::vector<monitor_t> rows_;
    std::vector<std::wstring> virtual_sources_;
    std::wstring primary_ = L"\\\\.\\DISPLAY1";
    bool query_ok_ = true;
    std::optional<UINT32> failed_source_;
    std::optional<UINT32> failed_target_;
    std::function<void(std::vector<DISPLAYCONFIG_PATH_INFO> &, std::vector<DISPLAYCONFIG_MODE_INFO> &)> corrupt_snapshot_;
    unsigned queries_ = 0;
    unsigned match_display_queries_ = 0;
    unsigned source_name_queries_ = 0;
    unsigned target_name_queries_ = 0;
    unsigned enum_devices_queries_ = 0;
    unsigned modern_queries_ = 0;
    unsigned legacy_queries_ = 0;
    bool accept_notification_ = true;
    std::vector<std::string> notifications_;
    // Destroy the facade before every callback capture, owned path and native-input container.
    std::unique_ptr<discovery::discovery_session_t> session_;
  };
}  // namespace

TEST_F(ArTargetDiscoveryTest, OrdinaryMonitorsPersistRealDecisionsAndNotifyOnlyOnce) {
  rows_ = {monitor(L"BBB0001", 1), monitor(L"ZZZ0002", 2, L"\u663e\u793a\u5668")};
  auto internal = monitor(L"SMKD1CE", 3, L"Apollo AR Desktop");
  rows_.push_back(internal);
  start();
  const auto first = find();
  EXPECT_TRUE(first.query_succeeded);
  EXPECT_FALSE(first.selected);
  EXPECT_TRUE(first.approved.empty());
  EXPECT_EQ(notifications_, (std::vector<std::string> {"Office Monitor", "\xE6\x98\xBE\xE7\xA4\xBA\xE5\x99\xA8"}));
  const auto stored = persisted_devices();
  ASSERT_EQ(stored.size(), 2u);
  EXPECT_EQ(stored[0], (json {{"id", "DISPLAY:BBB0001"}, {"name", "Office Monitor"}, {"decision", "pending"}, {"auto_detected", false}}));
  EXPECT_EQ(stored[1].at("id"), "DISPLAY:ZZZ0002");
  EXPECT_NE(read_config().find("# owned discovery fixture\r\nkeep = unchanged\r\n"), std::string::npos);
  const auto devices = device_json();
  ASSERT_EQ(devices.size(), 2u);
  EXPECT_TRUE(devices[0].at("connected").get<bool>());
  EXPECT_TRUE(devices[1].at("connected").get<bool>());
  const auto bytes = read_config();
  EXPECT_TRUE(find().query_succeeded);
  EXPECT_EQ(read_config(), bytes);
  EXPECT_EQ(notifications_.size(), 2u);
  record_costs();
}

TEST_F(ArTargetDiscoveryTest, ApprovedTargetsOwnEveryFieldAndPreferredSelectionKeepsSortedOrder) {
  rows_ = {monitor(L"ZZZ0001", 1), monitor(L"AAA0002", 2, L"XREAL Air"), monitor(L"BBB0003", 3, L"VITURE Pro")};
  rows_[1].width = 3840;
  rows_[1].refresh_numerator = 60000;
  rows_[1].refresh_denominator = 1001;
  rows_[1].hdr_user_enabled = false;
  rows_[1].hdr_limited = true;
  start();
  const auto result = find({}, rows_[2].path);
  ASSERT_TRUE(result.query_succeeded);
  ASSERT_TRUE(result.selected);
  ASSERT_EQ(result.approved.size(), 2u);
  EXPECT_EQ(result.approved[0].device_id, "DISPLAY:AAA0002");
  EXPECT_EQ(result.approved[1].device_id, "DISPLAY:BBB0003");
  EXPECT_EQ(result.selected->device_path, rows_[2].path);
  EXPECT_EQ(target_json(result.approved[0]), (json {
                                               {"device_id", "DISPLAY:AAA0002"},
                                               {"device_path", R"(\\?\DISPLAY#AAA0002#2)"},
                                               {"friendly_name", "XREAL Air"},
                                               {"gdi_name", R"(\\.\DISPLAY2)"},
                                               {"desktop_topology", R"(primary=\\?\DISPLAY#ZZZ0001#1;\\?\DISPLAY#BBB0003#3@3840,0:1920x1080@60000;\\?\DISPLAY#ZZZ0001#1@0,0:1920x1080@60000)"},
                                               {"source_adapter", {101, 11}},
                                               {"source_id", 2},
                                               {"target_adapter", {201, 21}},
                                               {"target_id", 12},
                                               {"hdr", {{"known", true}, {"supported", true}, {"user_enabled", false}, {"active", true}, {"limited_by_policy", true}, {"bits_per_color", 10}}},
                                               {"rect", {1920, 0, 5760, 1080}},
                                               {"refresh_millihz", 59940},
                                               {"mode", static_cast<int>(ar_glasses::presentation_mode_e::sbs_ai)},
                                               {"is_primary", false},
                                               {"is_cloned", false},
                                             }));
  auto caller_copy = *result.selected;
  caller_copy.device_path.clear();
  caller_copy.hdr.active = false;
  caller_copy.desktop_topology.assign("caller mutation");
  const auto repeated = find({}, rows_[2].path);
  ASSERT_TRUE(repeated.selected);
  EXPECT_EQ(target_json(*repeated.selected), target_json(*result.selected));
  record_costs();
}

TEST_F(ArTargetDiscoveryTest, SameModelPromotionUsesEachRowsExistingDecisionPoint) {
  rows_ = {monitor(L"DUP0001", 1), monitor(L"DUP0001", 2, L"XREAL Air")};
  start();
  const auto first = find({}, rows_[0].path);
  ASSERT_TRUE(first.query_succeeded);
  ASSERT_TRUE(first.selected);
  ASSERT_EQ(first.approved.size(), 1u);
  EXPECT_EQ(first.selected->device_path, rows_[1].path);
  EXPECT_EQ(first.approved[0].device_path, rows_[1].path);
  ASSERT_EQ(notifications_.size(), 1u);
  EXPECT_EQ(notifications_[0], "Office Monitor");
  ASSERT_EQ(persisted_devices().size(), 1u);
  EXPECT_EQ(persisted_devices()[0].at("decision"), "approved");
  const auto second = find();
  ASSERT_TRUE(second.selected);
  ASSERT_EQ(second.approved.size(), 2u);
  EXPECT_EQ(second.selected->device_path, rows_[0].path);
  EXPECT_EQ(notifications_.size(), 1u);
  record_costs();
}

TEST_F(ArTargetDiscoveryTest, PrimaryPathUsesRawCcdOrderBeforeSortingClonedTargets) {
  rows_ = {monitor(L"ZZZ0001", 1, L"XREAL Air"), monitor(L"AAA0002", 2, L"VITURE Pro")};
  rows_[1].source_id = rows_[0].source_id;
  rows_[1].gdi = rows_[0].gdi;
  rows_[1].x = 0;
  start();
  const auto result = find();
  ASSERT_TRUE(result.selected);
  ASSERT_EQ(result.approved.size(), 2u);
  EXPECT_EQ(result.selected->device_id, "DISPLAY:AAA0002");
  EXPECT_TRUE(result.approved[0].is_primary);
  EXPECT_TRUE(result.approved[1].is_primary);
  EXPECT_TRUE(result.approved[0].is_cloned);
  EXPECT_TRUE(result.approved[1].is_cloned);
  EXPECT_EQ(result.approved[0].desktop_topology, R"(primary=\\?\DISPLAY#ZZZ0001#1;\\?\DISPLAY#ZZZ0001#1@0,0:1920x1080@60000)");
  record_costs();
}

TEST_F(ArTargetDiscoveryTest, ExcludedVirtualPrimaryAndVirtualCloneKeepTheirDistinctEvidence) {
  rows_ = {monitor(L"SMKD1CE", 1), monitor(L"TCL03D4", 2), monitor(L"VIRTUAL", 3)};
  virtual_sources_.push_back(rows_[0].gdi);
  rows_[2].path = L"\\\\?\\SUDOVDA#clone";
  rows_[2].source_id = rows_[1].source_id;
  rows_[2].gdi = rows_[1].gdi;
  start();
  const auto result = find();
  ASSERT_TRUE(result.query_succeeded);
  ASSERT_TRUE(result.selected);
  ASSERT_EQ(result.approved.size(), 1u);
  EXPECT_FALSE(result.selected->is_primary);
  EXPECT_TRUE(result.selected->is_cloned);
  EXPECT_EQ(result.selected->desktop_topology, "primary=");
  EXPECT_EQ(device_json().size(), 1u);
  record_costs();
}

TEST_F(ArTargetDiscoveryTest, RequiredRejectedAndMissingPathsNeverMutateDiscoveryOutputs) {
  rows_ = {monitor(L"TCL03D4", 1), monitor(L"BBB0002", 2)};
  write_config(json::array({{{"id", "DISPLAY:TCL03D4"}, {"name", "Office Monitor"}, {"decision", "rejected"}, {"auto_detected", false}}}));
  start();
  const auto initial_devices = device_json();
  const auto initial_config = read_config();
  const auto required = find(rows_[0].path);
  ASSERT_TRUE(required.query_succeeded);
  ASSERT_TRUE(required.selected);
  EXPECT_EQ(required.selected->device_id, "DISPLAY:TCL03D4");
  EXPECT_TRUE(required.selected->hdr.known);
  EXPECT_TRUE(required.approved.empty());
  const auto absent = find(L"\\\\?\\DISPLAY#MISSING#1");
  EXPECT_TRUE(absent.query_succeeded);
  EXPECT_FALSE(absent.selected);
  EXPECT_TRUE(absent.approved.empty());
  EXPECT_EQ(device_json(), initial_devices);
  EXPECT_EQ(read_config(), initial_config);
  EXPECT_TRUE(notifications_.empty());
  EXPECT_EQ(session_->costs().persistence_attempts, 0u);
  record_costs();
}

TEST_F(ArTargetDiscoveryTest, CompleteSnapshotFailuresNeverPublishPartialConnectionsOrDisconnects) {
  rows_ = {monitor(L"TCL03D4", 1), monitor(L"SMKD1CE", 2)};
  virtual_sources_ = {rows_[1].gdi};
  start();
  ASSERT_TRUE(find().selected);
  const auto stable_devices = device_json();
  const auto stable_config = read_config();
  const auto notifications = notifications_;
  for (unsigned failure = 0; failure != 8; ++failure) {
    SCOPED_TRACE(failure);
    query_ok_ = failure != 0;
    failed_source_ = failure == 1 ? std::make_optional(rows_[1].source_id) : std::nullopt;
    failed_target_ = failure == 2 ? std::make_optional(rows_[0].target_id) : std::nullopt;
    primary_ = failure == 5 ? L"" : failure == 6 ? L"\\\\.\\STALE" :
                                                   rows_[0].gdi;
    corrupt_snapshot_ = [failure](auto &paths, auto &modes) {
      if (failure == 3) {
        paths[0].sourceInfo.modeInfoIdx = DISPLAYCONFIG_PATH_MODE_IDX_INVALID;
      } else if (failure == 4) {
        modes[0].infoType = DISPLAYCONFIG_MODE_INFO_TYPE_TARGET;
      } else if (failure == 7) {
        paths[0].sourceInfo.modeInfoIdx = static_cast<UINT32>(modes.size());
      }
    };
    const auto result = find();
    EXPECT_FALSE(result.query_succeeded);
    EXPECT_FALSE(result.selected);
    EXPECT_TRUE(result.approved.empty());
    EXPECT_EQ(device_json(), stable_devices);
    EXPECT_EQ(read_config(), stable_config);
    EXPECT_EQ(notifications_, notifications);
  }
  query_ok_ = true;
  failed_source_.reset();
  failed_target_.reset();
  corrupt_snapshot_ = {};
  primary_ = rows_[1].gdi;
  rows_.erase(rows_.begin());
  const auto absent = find();
  EXPECT_TRUE(absent.query_succeeded);
  EXPECT_FALSE(absent.selected);
  EXPECT_FALSE(device_json()[0].at("connected").get<bool>());
  record_costs();
}

TEST_F(ArTargetDiscoveryTest, ModernHdrFailureRetainsLegacyFallbackAndUnknownIsNotTopologyFailure) {
  rows_ = {monitor(L"TCL03D4", 1)};
  rows_[0].modern_status = ERROR_ACCESS_DENIED;
  rows_[0].bits = 8;
  start();
  const auto legacy = find();
  ASSERT_TRUE(legacy.selected);
  EXPECT_TRUE(legacy.query_succeeded);
  EXPECT_TRUE(legacy.selected->hdr.known);
  EXPECT_TRUE(legacy.selected->hdr.active);
  EXPECT_TRUE(legacy.selected->hdr.user_enabled);
  EXPECT_EQ(legacy.selected->hdr.bits_per_color, 8u);
  EXPECT_EQ(modern_queries_, 1u);
  EXPECT_EQ(legacy_queries_, 1u);
  rows_[0].legacy_status = ERROR_GEN_FAILURE;
  const auto unknown = find();
  ASSERT_TRUE(unknown.selected);
  EXPECT_TRUE(unknown.query_succeeded);
  EXPECT_EQ(target_json(*unknown.selected).at("hdr"), (json {{"known", false}, {"supported", false}, {"user_enabled", false}, {"active", false}, {"limited_by_policy", false}, {"bits_per_color", 0}}));
  EXPECT_EQ(legacy_queries_, 2u);
  record_costs();
}

TEST_F(ArTargetDiscoveryTest, EveryCallReobservesCurrentGeometryColorNamesAndApproval) {
  rows_ = {monitor(L"BBB0001", 1), monitor(L"TCL03D4", 2)};
  start();
  const auto first = find();
  ASSERT_TRUE(first.selected);
  const auto first_owned = target_json(*first.selected);
  rows_[0].x = -1920;
  rows_[0].refresh_numerator = 75000;
  rows_[1].friendly = L"Updated RayNeo";
  rows_[1].gdi = L"\\\\.\\DISPLAY7";
  rows_[1].target_adapter = {777, 77};
  rows_[1].target_id = 97;
  rows_[1].width = 3840;
  rows_[1].hdr_active = false;
  rows_[1].bits = 12;
  const auto second = find();
  ASSERT_TRUE(second.selected);
  EXPECT_EQ(queries_, 2u);
  EXPECT_EQ(second.selected->friendly_name, "Updated RayNeo");
  EXPECT_EQ(second.selected->gdi_name, R"(\\.\DISPLAY7)");
  EXPECT_EQ(second.selected->adapter_id.LowPart, 777u);
  EXPECT_EQ(second.selected->target_id, 97u);
  EXPECT_EQ(second.selected->mode, ar_glasses::presentation_mode_e::sbs_ai);
  EXPECT_FALSE(second.selected->hdr.active);
  EXPECT_EQ(second.selected->hdr.bits_per_color, 12u);
  EXPECT_NE(second.selected->desktop_topology.find("@-1920,0:1920x1080@75000"), std::string::npos);
  EXPECT_EQ(target_json(*first.selected), first_owned);
  ASSERT_TRUE(ar_glasses::set_device_decision("DISPLAY:TCL03D4", ar_glasses::device_decision_e::rejected));
  EXPECT_FALSE(find().selected);
  ASSERT_TRUE(ar_glasses::set_device_decision("DISPLAY:BBB0001", ar_glasses::device_decision_e::approved));
  const auto changed = find({}, rows_[1].path, false);
  ASSERT_TRUE(changed.selected);
  EXPECT_EQ(changed.selected->device_id, "DISPLAY:BBB0001");
  EXPECT_TRUE(changed.approved.empty());
  record_costs();
}

TEST_F(ArTargetDiscoveryTest, DuplicateDevicePathsAreExcludedFromTheirOwnNeighborFingerprint) {
  rows_ = {monitor(L"TCL03D4", 1), monitor(L"TCL03D4", 2)};
  rows_[1].path = rows_[0].path;
  start();
  const auto result = find();
  ASSERT_TRUE(result.selected);
  ASSERT_EQ(result.approved.size(), 2u);
  EXPECT_EQ(result.approved[0].desktop_topology, R"(primary=\\?\DISPLAY#TCL03D4#1)");
  EXPECT_EQ(result.approved[1].desktop_topology, result.approved[0].desktop_topology);
  record_costs();
}

TEST_F(ArTargetDiscoveryTest, FailedAtomicPersistenceKeepsOriginalBytesAndExplicitDecisionCanRetry) {
  rows_ = {monitor(L"AAA0001", 1), monitor(L"BBB0002", 2)};
  accept_notification_ = false;
  const auto original_bytes = read_config();
  auto blocked_temporary = config_path_;
  blocked_temporary += L".apollo-ar.tmp";
  ASSERT_TRUE(fs::create_directory(blocked_temporary));
  start();
  const auto first = find();
  EXPECT_TRUE(first.query_succeeded);
  EXPECT_FALSE(first.selected);
  EXPECT_EQ(read_config(), original_bytes);
  EXPECT_EQ(notifications_.size(), 2u);
  // Rejected notification delivery must remain retryable, independent of persistence backoff.
  EXPECT_TRUE(find().query_succeeded);
  EXPECT_EQ(notifications_.size(), 4u);
  EXPECT_EQ(read_config(), original_bytes);
  ASSERT_TRUE(fs::remove(blocked_temporary));
  ASSERT_TRUE(ar_glasses::set_device_decision("DISPLAY:AAA0001", ar_glasses::device_decision_e::approved));
  ASSERT_EQ(persisted_devices().size(), 2u);
  EXPECT_EQ(persisted_devices()[0].at("decision"), "approved");
  accept_notification_ = true;
  const auto approved = find();
  ASSERT_TRUE(approved.selected);
  EXPECT_EQ(approved.selected->device_id, "DISPLAY:AAA0001");
  EXPECT_EQ(notifications_.size(), 5u);
  EXPECT_TRUE(find().query_succeeded);
  EXPECT_EQ(notifications_.size(), 5u);
  record_costs();
}

TEST_F(ArTargetDiscoveryTest, MissingHooksFailClosedAndScopedDiscoveryRestoresCallerState) {
  const auto original_config = config::sunshine.config_file;
  const auto original_devices = device_json();
  for (unsigned missing = 0; missing != 5; ++missing) {
    SCOPED_TRACE(missing);
    auto incomplete = hooks();
    if (missing == 0) {
      incomplete.query_active = {};
    }
    if (missing == 1) {
      incomplete.match_display = {};
    }
    if (missing == 2) {
      incomplete.get_info = {};
    }
    if (missing == 3) {
      incomplete.enum_devices = {};
    }
    if (missing == 4) {
      incomplete.notify = {};
    }
    EXPECT_THROW(discovery::discovery_session_t(config_path_, std::move(incomplete)), std::invalid_argument);
    EXPECT_EQ(config::sunshine.config_file, original_config);
    EXPECT_EQ(device_json(), original_devices);
  }
  rows_ = {monitor(L"TCL03D4", 1)};
  start();
  ASSERT_TRUE(find().selected);
  EXPECT_EQ(config::sunshine.config_file, platf::to_utf8(config_path_.native()));
  session_.reset();
  EXPECT_EQ(config::sunshine.config_file, original_config);
  EXPECT_EQ(device_json(), original_devices);
}

TEST_F(ArTargetDiscoveryTest, DiscoveryKeepsApprovalWithoutSelectingBeforeStart) {
  rows_ = {monitor(L"TCL03D4", 1)};
  start({.ar_requested = false});
  const auto result = find();
  ASSERT_TRUE(result.query_succeeded);
  EXPECT_FALSE(result.selected);
  EXPECT_TRUE(result.approved.empty());
  EXPECT_EQ(device_json().at(0).at("decision"), static_cast<int>(ar_glasses::device_decision_e::approved));
  EXPECT_EQ(session_->costs().hdr_queries, 0u);
  EXPECT_EQ(session_->costs().fingerprints, 0u);
}

TEST_F(ArTargetDiscoveryTest, ExplicitMonitorMaterializesOnlyItsExactInstance) {
  rows_ = {monitor(L"TCL03D4", 1), monitor(L"BBB0002", 2)};
  start({.ar_requested = false, .monitor_requested = true, .monitor_device_path = rows_[1].path});
  const auto result = find();
  ASSERT_TRUE(result.query_succeeded);
  ASSERT_TRUE(result.selected);
  EXPECT_EQ(result.selected->device_path, rows_[1].path);
  ASSERT_EQ(result.approved.size(), 1u);
  EXPECT_EQ(result.approved[0].device_path, rows_[1].path);
  EXPECT_TRUE(result.selected->hdr.known);
  EXPECT_FALSE(result.selected->is_primary);
  EXPECT_FALSE(result.selected->is_cloned);
  EXPECT_EQ(session_->costs().hdr_queries, 1u);
  EXPECT_EQ(session_->costs().fingerprints, 1u);
  rows_.pop_back();
  const auto unplugged = find();
  EXPECT_TRUE(unplugged.query_succeeded);
  EXPECT_FALSE(unplugged.selected);
  EXPECT_TRUE(unplugged.approved.empty());
  EXPECT_EQ(session_->costs().hdr_queries, 1u);
}

TEST_F(ArTargetDiscoveryTest, StoppedMonitorDoesNotSelectAnApprovedGlassesModel) {
  rows_ = {monitor(L"TCL03D4", 1)};
  start({.monitor_device_path = rows_[0].path});
  const auto result = find();
  ASSERT_TRUE(result.query_succeeded);
  EXPECT_FALSE(result.selected);
  EXPECT_TRUE(result.approved.empty());
  EXPECT_EQ(session_->costs().hdr_queries, 0u);
}

TEST_F(ArTargetDiscoveryTest, WatchdogRectRefreshesExactGeometryWithoutDiscoveryCosts) {
  rows_ = {monitor(L"TCL03D4", 1), monitor(L"TCL03D4", 2), monitor(L"SMKD1CE", 3, L"Apollo AR Desktop")};
  virtual_sources_ = {rows_[2].gdi};
  const auto selected_path = rows_[1].path;
  const auto original_config = read_config();
  start({.ar_requested = false, .monitor_requested = true, .monitor_device_path = selected_path});

  for (unsigned sample = 0; sample != 2; ++sample) {
    SCOPED_TRACE(sample);
    const auto baseline = find(selected_path);
    ASSERT_TRUE(baseline.query_succeeded);
    ASSERT_TRUE(baseline.selected);
    const auto costs = session_->costs();
    const auto snapshots = queries_;
    const auto target_names = target_name_queries_;
    const auto source_names = source_name_queries_;
    const auto virtual_matches = match_display_queries_;
    const auto primary_queries = enum_devices_queries_;
    const auto modern_hdr_queries = modern_queries_;
    const auto legacy_hdr_queries = legacy_queries_;
    const auto rect = session_->watchdog_rect(selected_path);
    ASSERT_TRUE(rect);
    EXPECT_EQ(rect->left, rows_[1].x);
    EXPECT_EQ(rect->top, rows_[1].y);
    EXPECT_EQ(rect->left, baseline.selected->rect.left);
    EXPECT_EQ(rect->top, baseline.selected->rect.top);
    EXPECT_EQ(rect->right, baseline.selected->rect.right);
    EXPECT_EQ(rect->bottom, baseline.selected->rect.bottom);
    EXPECT_EQ(queries_, snapshots + 1u);
    EXPECT_EQ(target_name_queries_, target_names + rows_.size());
    EXPECT_EQ(source_name_queries_, source_names);
    EXPECT_EQ(match_display_queries_, virtual_matches);
    EXPECT_EQ(enum_devices_queries_, primary_queries);
    EXPECT_EQ(modern_queries_, modern_hdr_queries);
    EXPECT_EQ(legacy_queries_, legacy_hdr_queries);
    EXPECT_EQ(session_->costs().hdr_queries, costs.hdr_queries);
    EXPECT_EQ(session_->costs().fingerprints, costs.fingerprints);

    // The exact path survives source-name changes and a move to a different desktop rectangle.
    rows_[1].gdi = L"\\\\.\\DISPLAY8";
    rows_[1].source_id = 8;
    rows_[1].target_id = 18;
    rows_[1].x = -3200;
    rows_[1].y = 350;
    rows_[1].width = 1600;
    rows_[1].height = 900;
  }
  EXPECT_EQ(session_->costs().hdr_queries, 2u);
  EXPECT_EQ(session_->costs().fingerprints, 2u);
  EXPECT_EQ(session_->costs().persistence_attempts, 0u);
  EXPECT_TRUE(notifications_.empty());
  EXPECT_EQ(read_config(), original_config);
}

TEST_F(ArTargetDiscoveryTest, WatchdogRectRejectsUnboundOrInvalidGeometry) {
  const auto removed = monitor(L"TCL03D4", 1);
  rows_ = {monitor(L"TCL03D4", 2)};
  rows_[0].gdi = removed.gdi;
  start();
  EXPECT_FALSE(session_->watchdog_rect(removed.path));
  EXPECT_FALSE(session_->watchdog_rect(removed.gdi));
  EXPECT_FALSE(session_->watchdog_rect({}));

  rows_.push_back(removed);
  ASSERT_TRUE(session_->watchdog_rect(removed.path));
  rows_[0].path = removed.path;
  EXPECT_FALSE(session_->watchdog_rect(removed.path));
  rows_.erase(rows_.begin());

  query_ok_ = false;
  EXPECT_FALSE(session_->watchdog_rect(removed.path));
  query_ok_ = true;
  for (unsigned failure = 0; failure != 7; ++failure) {
    SCOPED_TRACE(failure);
    corrupt_snapshot_ = [failure](auto &paths, auto &modes) {
      switch (failure) {
        case 0:
          paths[0].sourceInfo.modeInfoIdx = DISPLAYCONFIG_PATH_MODE_IDX_INVALID;
          break;
        case 1:
          paths[0].sourceInfo.modeInfoIdx = static_cast<UINT32>(modes.size());
          break;
        case 2:
          modes[0].infoType = DISPLAYCONFIG_MODE_INFO_TYPE_TARGET;
          break;
        case 3:
          ++modes[0].id;
          break;
        case 4:
          ++modes[0].adapterId.LowPart;
          break;
        case 5:
          modes[0].sourceMode.width = 0;
          break;
        case 6:
          modes[0].sourceMode.position.x = std::numeric_limits<LONG>::max();
          break;
      }
    };
    EXPECT_FALSE(session_->watchdog_rect(removed.path));
  }
}

TEST_F(ArTargetDiscoveryTest, WatchdogRectPreservesLegacyVirtualSourceFallback) {
  rows_ = {monitor(L"TCL03D4", 1), monitor(L"SMKD1CE", 2, L"Apollo AR Desktop")};
  virtual_sources_ = {rows_[1].gdi};
  failed_target_ = rows_[1].target_id;
  start();
  const auto baseline = find(rows_[0].path);
  ASSERT_TRUE(baseline.query_succeeded);
  ASSERT_TRUE(baseline.selected);
  const auto costs = session_->costs();
  const auto snapshots = queries_;
  const auto rect = session_->watchdog_rect(rows_[0].path);
  ASSERT_TRUE(rect);
  EXPECT_EQ(rect->left, baseline.selected->rect.left);
  EXPECT_EQ(rect->top, baseline.selected->rect.top);
  EXPECT_EQ(rect->right, baseline.selected->rect.right);
  EXPECT_EQ(rect->bottom, baseline.selected->rect.bottom);
  EXPECT_EQ(queries_, snapshots + 1u);
  EXPECT_EQ(session_->costs().hdr_queries, costs.hdr_queries);
  EXPECT_EQ(session_->costs().fingerprints, costs.fingerprints);

  // An unresolved physical identity still makes the lookup indeterminate.
  virtual_sources_.clear();
  EXPECT_FALSE(session_->watchdog_rect(rows_[0].path));
}

#endif
