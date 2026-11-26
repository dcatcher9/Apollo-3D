/**
 * @file tests/unit/test_apps_store.cpp
 * @brief Contracts of the app library's variable expansion and image-path resolution.
 */
#include "../tests_common.h"
#include "src/apps_store.h"
#include "src/process.h"
#include "src/utility.h"
#include "src/uuid.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <windows.h>

namespace {
  class AppsStoreParseTest: public testing::Test {
  protected:
    void SetUp() override {
      // Without the generated Virtual Display entry the listing holds only the configured apps.
      previous_status_ = proc::vDisplayDriverStatus.exchange(VDISPLAY::DRIVER_STATUS::UNKNOWN);
      const auto candidate = std::filesystem::temp_directory_path() /
                             ("apollo_apps_store_" + uuid_util::uuid_t::generate().string());
      ASSERT_TRUE(std::filesystem::create_directory(candidate));
      directory_ = candidate;
    }

    void TearDown() override {
      proc::vDisplayDriverStatus.store(previous_status_);
      if (!directory_.empty()) {
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
        EXPECT_FALSE(error) << error.message();
      }
    }

    std::optional<proc::proc_t> parse(std::string_view apps_json) {
      const auto apps = directory_ / "apps.json";
      std::ofstream file(apps, std::ios::binary | std::ios::trunc);
      file.write(apps_json.data(), static_cast<std::streamsize>(apps_json.size()));
      file.close();
      EXPECT_TRUE(file.good());
      return proc::parse(apps.string());
    }

    VDISPLAY::DRIVER_STATUS previous_status_ {};
    std::filesystem::path directory_;
  };
}  // namespace

TEST(AppsStoreFindMatch, ClosedBracketsReturnTheirMatchingClose) {
  constexpr std::string_view text = "(A(B)C)D)";
  EXPECT_EQ(proc::find_match(text.begin(), text.end()), text.begin() + 6);

  constexpr auto inner = text.substr(2);
  EXPECT_EQ(proc::find_match(inner.begin(), inner.end()), inner.begin() + 2);

  constexpr std::string_view empty_name = "()";
  EXPECT_EQ(proc::find_match(empty_name.begin(), empty_name.end()), empty_name.begin() + 1);
}

TEST(AppsStoreFindMatch, UnclosedBracketThrowsWithoutReadingPastTheEnd) {
  // The text ends exactly at a no-access page, so any read of *end faults.
  SYSTEM_INFO system_info;
  GetSystemInfo(&system_info);
  const auto page_size = static_cast<std::size_t>(system_info.dwPageSize);
  auto *const pages = static_cast<char *>(VirtualAlloc(nullptr, page_size * 2, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
  ASSERT_NE(pages, nullptr);
  const auto release = util::fail_guard([pages]() {
    VirtualFree(pages, 0, MEM_RELEASE);
  });
  DWORD previous_protection = 0;
  ASSERT_TRUE(VirtualProtect(pages + page_size, page_size, PAGE_NOACCESS, &previous_protection));

  for (const std::string_view text : {"(", "(VAR", "((VAR)", "(A(B)C"}) {
    SCOPED_TRACE(text);
    auto *const start = pages + page_size - text.size();
    std::memcpy(start, text.data(), text.size());
    const std::string_view guarded {start, text.size()};
    EXPECT_THROW((void) proc::find_match(guarded.begin(), guarded.end()), std::out_of_range);
  }
}

TEST_F(AppsStoreParseTest, ExpandsClosedVariablesAndFallsBackOnAnUnclosedOne) {
  const auto restore_variable = util::fail_guard([]() {
    SetEnvironmentVariableW(L"APOLLO_APPS_STORE_TEST_NAME", nullptr);
  });
  const auto parsed = parse(R"({"version":2,"env":{"APOLLO_APPS_STORE_TEST_NAME":"Name"},"apps":[
    {"uuid":"00000000-0000-0000-0000-000000000001","name":"$(APOLLO_APPS_STORE_TEST_NAME) costs $$5","exclude-global-prep-cmd":true}
  ]})");
  ASSERT_TRUE(parsed);
  const auto listing = parsed->get_app_list();
  ASSERT_EQ(listing.size(), 1u);
  EXPECT_EQ(listing[0].name, "Name costs $5");

  // An unclosed reference makes the whole file invalid, as before: only the fallback Desktop remains.
  const auto unclosed = parse(R"({"version":2,"env":{},"apps":[
    {"uuid":"00000000-0000-0000-0000-000000000001","name":"Broken $(UNCLOSED","exclude-global-prep-cmd":true}
  ]})");
  ASSERT_TRUE(unclosed);
  const auto fallback = unclosed->get_app_list();
  ASSERT_EQ(fallback.size(), 1u);
  EXPECT_EQ(fallback[0].uuid, FALLBACK_DESKTOP_UUID);
}

TEST(AppsStoreImagePath, PathThatIsNotUtf8IsAnImageThatIsNotFound) {
  // Windows-1252 "J\xFCrgen": the filesystem library cannot convert these bytes into a path.
  const std::string image_path = "C:\\Users\\J\xFCrgen\\Pictures\\game.png";
  const auto default_image = proc::validate_app_image_path("");
  std::string resolved;
  EXPECT_NO_THROW(resolved = proc::validate_app_image_path(image_path));
  EXPECT_EQ(resolved, default_image);
  EXPECT_EQ(proc::calculate_app_id("Game", image_path, 3), proc::calculate_app_id("Game", "", 3));
}

TEST_F(AppsStoreParseTest, KeepsAnAppWhoseExpandedImagePathIsNotUtf8) {
  // The environment is read through the ANSI code page, so a non-ASCII value need not be UTF-8.
  ASSERT_TRUE(SetEnvironmentVariableW(L"APOLLO_APPS_STORE_TEST_IMAGE_DIR", L"C:\\Users\\J\u00fcrgen"));
  const auto restore_variable = util::fail_guard([]() {
    SetEnvironmentVariableW(L"APOLLO_APPS_STORE_TEST_IMAGE_DIR", nullptr);
  });
  auto parsed = parse(R"({"version":2,"env":{},"apps":[
    {"uuid":"00000000-0000-0000-0000-000000000002","name":"Game","image-path":"$(APOLLO_APPS_STORE_TEST_IMAGE_DIR)\\Pictures\\game.png","exclude-global-prep-cmd":true}
  ]})");
  ASSERT_TRUE(parsed);
  const auto listing = parsed->get_app_list();
  ASSERT_EQ(listing.size(), 1u);
  EXPECT_EQ(listing[0].uuid, "00000000-0000-0000-0000-000000000002");
  std::string image;
  EXPECT_NO_THROW(image = parsed->get_app_image(std::stoi(listing[0].id)));
  EXPECT_EQ(image, proc::validate_app_image_path(""));
}
