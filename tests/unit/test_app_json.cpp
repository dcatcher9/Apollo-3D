/**
 * @file tests/unit/test_app_json.cpp
 * @brief Application JSON migration and ordering contracts.
 */
#include "../tests_common.h"
#include "src/apps_store.h"
#include "src/confighttp.h"

#include <nlohmann/json.hpp>
#include <utility>

using nlohmann::json;

TEST(ConfigHttpAppOrder, DuplicateRequestsConsumeFirstRemainingMatchAndPreserveOtherEntries) {
  const json first {{"uuid", "duplicate"}, {"name", "first"}, {"prep-cmd", {{{"do", "prepare"}}}}};
  const json second {{"uuid", "duplicate"}, {"name", "second"}};
  const json invalid_uuid {{"uuid", 17}};
  const json missing_uuid {{"name", "without UUID"}};
  const json tail {{"uuid", "tail"}, {"extension", {{"future", true}}}};
  auto apps = json::array({first, nullptr, invalid_uuid, missing_uuid, second, tail});

  const auto result = confighttp::reorder_app_list(
    std::move(apps),
    json::array({false, "missing", "duplicate", "duplicate", "duplicate"})
  );

  EXPECT_EQ(result, json::array({first, second, nullptr, invalid_uuid, missing_uuid, tail}));
}

TEST(ConfigHttpAppOrder, EmptyOrderRetainsAllEntriesAndTheirOriginalOrder) {
  const auto expected = json::array({{{"uuid", "one"}}, 3, {{"name", "no UUID"}}, {{"uuid", "two"}}});
  auto apps = expected;

  EXPECT_EQ(confighttp::reorder_app_list(std::move(apps), json::array()), expected);
  EXPECT_EQ(confighttp::reorder_app_list(json::array(), json::array({"unknown", 3})), json::array());
}

TEST(AppJsonMigration, ReplacesOnlyFirstDuplicateAndRemovesTransientInputFields) {
  const json first {{"uuid", "duplicate"}, {"name", "first"}};
  const json second {{"uuid", "duplicate"}, {"name", "second"}, {"future", {1, 2, 3}}};
  const json tail {{"uuid", "tail"}};
  json tree {{"apps", json::array({first, second, tail})}, {"env", {{"CUSTOM", "value"}}}};
  json input {
    {"uuid", "duplicate"},
    {"name", "replacement"},
    {"prep-cmd", json::array()},
    {"detached", json::array()},
    {"launching", true},
    {"index", 7},
    {"extension", {{"keep", true}}}
  };
  const json replacement {{"uuid", "duplicate"}, {"name", "replacement"}, {"extension", {{"keep", true}}}};

  proc::migrate_apps(&tree, &input);

  EXPECT_EQ(input, replacement);
  EXPECT_EQ(tree.at("apps"), json::array({replacement, second, tail}));
  EXPECT_EQ(tree.at("env"), (json {{"CUSTOM", "value"}}));
}

TEST(AppJsonMigration, AppendsGeneratedUuidWithoutChangingExistingApplications) {
  const json existing {{"uuid", "existing"}, {"prep-cmd", {{{"do", "keep"}}}}, {"extra", {1, 2}}};
  json tree {{"apps", json::array({existing})}};
  json input {{"uuid", ""}, {"name", "new application"}, {"detached", {"start helper"}}};

  proc::migrate_apps(&tree, &input);

  ASSERT_EQ(tree.at("apps").size(), 2u);
  ASSERT_TRUE(input.at("uuid").is_string());
  EXPECT_FALSE(input.at("uuid").get_ref<const json::string_t &>().empty());
  EXPECT_EQ(tree.at("apps").at(0), existing);
  EXPECT_EQ(tree.at("apps").at(1), input);
}

TEST(AppJsonMigration, NonArrayApplicationsAreReplacedByTheNewApplication) {
  json tree {{"apps", "malformed"}, {"extension", 42}};
  json input {{"uuid", "new"}, {"name", "application"}};

  proc::migrate_apps(&tree, &input);

  EXPECT_EQ(tree.at("apps"), json::array({input}));
  EXPECT_EQ(tree.at("extension"), 42);
}

TEST(AppJsonMigration, ExistingApplicationWithoutUuidRetainsItsErrorBehavior) {
  json tree {{"apps", json::array({{{"name", "missing UUID"}}})}};
  json input {{"uuid", "new"}, {"name", "application"}};

  EXPECT_THROW(proc::migrate_apps(&tree, &input), json::out_of_range);
}
