/**
 * @file tests/unit/test_app_id.cpp
 * @brief Compatibility contracts for application IDs derived from names and image bytes.
 */
#include "../tests_common.h"
#include "src/apps_store.h"
#include "src/process.h"
#include "src/uuid.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <locale>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace {
  using app_ids_t = std::tuple<std::string, std::string>;

  class scoped_app_id_locale_t {
  public:
    explicit scoped_app_id_locale_t(std::locale locale):
        previous_(std::locale::global(std::move(locale))) {
    }

    ~scoped_app_id_locale_t() {
      std::locale::global(previous_);
    }

  private:
    std::locale previous_;
  };

  class grouped_app_id_numbers_t: public std::numpunct<char> {
    char do_thousands_sep() const override {
      return '_';
    }

    std::string do_grouping() const override {
      return "\3";
    }
  };

  // A complete 1x1 RGBA PNG. The hash consumes all 70 bytes, including embedded NULs.
  constexpr std::string_view png_bytes {
    "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52"
    "\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    "\x00\x00\x00\x0d\x49\x44\x41\x54\x78\x9c\x63\x60\x68\xf8\xff\x1f"
    "\x00\x04\x82\x02\x7f\xe7\xd8\x5a\x6c\x00\x00\x00\x00\x49\x45\x4e"
    "\x44\xae\x42\x60\x82",
    70
  };

  class AppIdTest: public testing::Test {
  protected:
    void SetUp() override {
      const auto candidate = std::filesystem::temp_directory_path() /
                             ("apollo_app_id_" + uuid_util::uuid_t::generate().string());
      ASSERT_TRUE(std::filesystem::create_directory(candidate));
      directory_ = candidate;
    }

    void TearDown() override {
      for (auto path = created_paths_.rbegin(); path != created_paths_.rend(); ++path) {
        std::error_code error;
        std::filesystem::remove(*path, error);
        EXPECT_FALSE(error) << path->string() << ": " << error.message();
      }
      if (!directory_.empty()) {
        std::error_code error;
        std::filesystem::remove(directory_, error);
        EXPECT_FALSE(error) << error.message();
      }
    }

    bool write_file(const std::filesystem::path &path, std::string_view bytes) {
      created_paths_.push_back(path);
      std::ofstream file(path, std::ios::binary | std::ios::trunc);
      file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
      file.close();
      EXPECT_TRUE(file.good()) << path.string();
      return file.good();
    }

    scoped_app_id_locale_t locale_ {std::locale::classic()};
    std::filesystem::path directory_;
    std::vector<std::filesystem::path> created_paths_;
  };
}  // namespace

// Fixed goldens were generated independently with Python zlib.crc32/hashlib.sha256 over the
// literal UTF-8/binary bytes, then the existing signed-32-bit absolute-value ID conversion.
// None of these CRCs is the INT32_MIN absolute-value edge case.
TEST_F(AppIdTest, NamesAndIndexBoundariesKeepFixedIds) {
  struct case_t {
    const char *label;
    std::string name;
    const char *unindexed;
    std::array<const char *, 5> indexed;
  };

  const std::array<case_t, 5> cases {{
    {"ASCII", "Desktop", "958645192", {"601900157", "507581505", "110480674", "285854599", "1485717048"}},
    {"UTF-8", "Caf\xc3\xa9 \xe9\x9b\xaa", "1110622627", {"507756714", "389705312", "744592622", "2130639414", "533228782"}},
    {"embedded NUL", std::string {"Name\0Suffix", 11}, "412809428", {"958081339", "2115761271", "221293632", "1311641871", "2061286913"}},
    {"long", std::string(4096, 'n'), "2085072247", {"1614568695", "1234535043", "840676362", "550519336", "276205418"}},
    {"empty", "", "0", {"186917087", "649138913", "873121252", "1080688651", "365954768"}},
  }};
  constexpr std::array indices {0, -7, 12345, std::numeric_limits<int>::min(), std::numeric_limits<int>::max()};
  for (const auto &item : cases) {
    SCOPED_TRACE(item.label);
    for (std::size_t index = 0; index < indices.size(); ++index) {
      SCOPED_TRACE(indices[index]);
      EXPECT_EQ(proc::calculate_app_id(item.name, "", indices[index]), (app_ids_t {item.unindexed, item.indexed[index]}));
    }
  }
}

TEST_F(AppIdTest, EmptyDefaultMissingAndNonPngImagesUseOnlyTheName) {
  const auto non_png = directory_ / "existing.jpg";
  ASSERT_TRUE(write_file(non_png, png_bytes));
  for (const auto &path : std::array<std::string, 4> {
         "",
         SUNSHINE_ASSETS_DIR "/box.png",
         (directory_ / "missing.png").string(),
         non_png.string()
       }) {
    SCOPED_TRACE(path);
    EXPECT_EQ(proc::calculate_app_id("Desktop", path, 12345), (app_ids_t {"958645192", "110480674"}));
  }
}

TEST_F(AppIdTest, EqualImageBytesIgnorePathAndChangedBytesChangeIds) {
  const auto first = directory_ / "first.png";
  const auto second = directory_ / "second.PNG";
  ASSERT_TRUE(write_file(first, png_bytes));
  ASSERT_TRUE(write_file(second, png_bytes));
  const app_ids_t original {"280436797", "1139264893"};
  EXPECT_EQ(proc::calculate_app_id("Image App", first.string(), 12345), original);
  EXPECT_EQ(proc::calculate_app_id("Image App", second.string(), 12345), original);

  std::string changed {png_bytes};
  changed.append("\0tail\0", 6);
  ASSERT_TRUE(write_file(first, changed));
  EXPECT_EQ(proc::calculate_app_id("Image App", first.string(), 12345), (app_ids_t {"273707526", "578754184"}));
  EXPECT_EQ(proc::calculate_app_id("Image App", second.string(), 12345), original);
}

TEST_F(AppIdTest, UnreadableImageDirectoryRetainsTheEmptyDigestIdentity) {
  const auto image_directory = directory_ / "unreadable.png";
  ASSERT_TRUE(std::filesystem::create_directory(image_directory));
  created_paths_.push_back(image_directory);
  ASSERT_EQ(proc::validate_app_image_path(image_directory.string()), image_directory.string());

  // The existing file-hash helper finalizes SHA-256(empty) when opening/reading fails.
  // Its nullopt fallback concerns OpenSSL failures, not an unreadable image path.
  EXPECT_EQ(proc::calculate_app_id("Image App", image_directory.string(), 12345), (app_ids_t {"500936078", "2062496200"}));
}

TEST_F(AppIdTest, IndexFormattingKeepsTheStreamsGlobalLocale) {
  const scoped_app_id_locale_t grouped {std::locale {std::locale::classic(), new grouped_app_id_numbers_t}};
  const std::array cases {
    std::pair {12345, "2004365008"},
    std::pair {-12345, "2012429244"},
    std::pair {std::numeric_limits<int>::max(), "1238206768"},
    std::pair {std::numeric_limits<int>::min(), "1417179694"},
  };
  for (const auto &[index, expected] : cases) {
    SCOPED_TRACE(index);
    EXPECT_EQ(proc::calculate_app_id("Desktop", "", index), (app_ids_t {"958645192", expected}));
  }
}

TEST_F(AppIdTest, ParsingDuplicateNamesKeepsFirstIdAndIndexedFallbackOrder) {
  const auto previous_status = proc::vDisplayDriverStatus.exchange(VDISPLAY::DRIVER_STATUS::UNKNOWN);
  const auto restore_status = util::fail_guard([previous_status]() {
    proc::vDisplayDriverStatus.store(previous_status);
  });
  const auto apps = directory_ / "apps.json";
  ASSERT_TRUE(write_file(apps, R"({"version":2,"env":{},"apps":[
    {"uuid":"00000000-0000-0000-0000-000000000001","name":"Duplicate","exclude-global-prep-cmd":true},
    {"uuid":"00000000-0000-0000-0000-000000000002","name":"Duplicate","exclude-global-prep-cmd":true},
    {"uuid":"00000000-0000-0000-0000-000000000003","name":"Duplicate","exclude-global-prep-cmd":true}
  ]})"));

  const auto parsed = proc::parse(apps.string());
  ASSERT_TRUE(parsed);
  const auto listing = parsed->get_app_list();
  ASSERT_EQ(listing.size(), 3u);
  const std::array expected {"1867119688", "1190552074", "536923216"};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(listing[index].idx, std::to_string(index));
    EXPECT_EQ(listing[index].id, expected[index]);
  }
}
