#include "src/offline_sbs_job.h"
#include "src/offline_sbs_progress_test.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <windows.h>

namespace {
  namespace fs = std::filesystem;
  using namespace std::chrono_literals;

  constexpr std::string_view job_id = "11111111-2222-3333-4444-555555555555";

  class progress_files_t {
  public:
    progress_files_t() {
      root = fs::temp_directory_path() /
             ("sunshine3d-progress-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
      if (!fs::create_directory(root)) {
        throw std::runtime_error("cannot create an exclusively owned progress test directory");
      }
      progress = root / fs::path {u8"progress-\u96ea.json"};
      temporary = progress;
      temporary += L".part";
    }

    ~progress_files_t() {
      // Only these known test-owned leaves are removed; there is no recursive cleanup.
      for (const auto &path : {temporary / "blocker", temporary, progress, root}) {
        std::error_code error;
        fs::remove(path, error);
        if (error) {
          ADD_FAILURE() << "Cannot remove an owned progress test path: " << path << ": " << error.message();
        }
      }
    }

    fs::path root, progress, temporary;
  };

  offline_sbs::media_contract_t timed_media() {
    offline_sbs::media_contract_t media;
    media.time_base = {1, 1000};
    media.frames = {{1, 1000, 40}, {2, 1040, 60}, {3, 1100, 50}};
    return media;
  }

  std::string read_bytes(const fs::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("cannot read the owned progress snapshot");
    }
    return {std::istreambuf_iterator<char> {input}, std::istreambuf_iterator<char> {}};
  }

  void write_bytes(const fs::path &path, std::string_view bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!output) {
      throw std::runtime_error("cannot write the owned progress fixture");
    }
  }

  offline_sbs::worker_context_t reader_context(const fs::path &path) {
    offline_sbs::worker_context_t context;
    context.job_id = job_id;
    context.worker_progress = path;
    return context;
  }
}  // namespace

TEST(OfflineSbsProgressTest, PublishesExactProbeFirstFrameLastFrameAndPhaseSnapshots) {
  progress_files_t files;
  const auto media = timed_media();
  const auto context = reader_context(files.progress);
  offline_sbs::worker_progress_test_publisher_t publisher {files.progress, std::string {job_id}};

  struct update_t {
    const char *phase;
    std::uint64_t processed;
    const offline_sbs::media_contract_t *media;
    nlohmann::json source_time;
  };

  const std::vector<update_t> updates {
    {"probe", 0, nullptr, nullptr},
    {"convert", 0, &media, nullptr},
    {"convert", 1, &media, 0.0},
    {"convert", 3, &media, 0.1},
    {"mux", 3, &media, 0.1},
    {"complete", 3, &media, 0.1},
  };
  for (std::size_t index = 0; index < updates.size(); ++index) {
    const auto &update = updates[index];
    SCOPED_TRACE(update.phase);
    publisher.publish(update.phase, update.processed, update.media, std::chrono::steady_clock::time_point {} + 200ms * static_cast<int>(index));
    const nlohmann::json expected {
      {"schema", 1},
      {"job_id", job_id},
      {"phase", update.phase},
      {"processed_frames", update.processed},
      {"total_frames", update.media ? nlohmann::json(3) : nlohmann::json(nullptr)},
      {"source_time_seconds", update.source_time},
      {"source_duration_seconds", update.media ? nlohmann::json(0.15) : nlohmann::json(nullptr)},
    };
    EXPECT_EQ(read_bytes(files.progress), expected.dump(2) + "\n");
    EXPECT_FALSE(fs::exists(files.temporary));
    const auto consumed = offline_sbs::read_worker_progress_for_test(context);
    EXPECT_EQ(consumed.phase, update.phase);
    EXPECT_EQ(consumed.processed_frames, update.processed);
    EXPECT_EQ(consumed.total_frames.has_value(), update.media != nullptr);
    if (update.media) {
      EXPECT_EQ(consumed.total_frames, 3u);
      EXPECT_EQ(consumed.source_duration_seconds, 0.15);
    }
  }
  EXPECT_EQ(publisher.successful_publications(), updates.size());
  RecordProperty("successful_publications", static_cast<int>(publisher.successful_publications()));
}

TEST(OfflineSbsProgressTest, FailedPublicationWhilePreviousReaderIsOpenCanRetryAfterItCloses) {
  progress_files_t files;
  const auto media = timed_media();
  offline_sbs::worker_progress_test_publisher_t publisher {files.progress, std::string {job_id}};
  publisher.publish("convert", 0, &media, std::chrono::steady_clock::time_point {});
  const auto previous = read_bytes(files.progress);
  const auto handle = CreateFileW(files.progress.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  ASSERT_NE(handle, INVALID_HANDLE_VALUE);
  std::unique_ptr<void, decltype(&CloseHandle)> old_file {handle, &CloseHandle};
  const auto due = std::chrono::steady_clock::time_point {} + 200ms;
  EXPECT_THROW(publisher.publish("convert", 1, &media, due), std::runtime_error);
  EXPECT_EQ(read_bytes(files.progress), previous);
  EXPECT_FALSE(fs::exists(files.temporary));
  EXPECT_EQ(publisher.successful_publications(), 1u);
  std::string retained(previous.size(), '\0');
  DWORD received = 0;
  ASSERT_TRUE(ReadFile(handle, retained.data(), static_cast<DWORD>(retained.size()), &received, nullptr));
  EXPECT_EQ(received, previous.size());
  EXPECT_EQ(retained, previous);
  old_file.reset();
  publisher.publish("convert", 1, &media, due);
  const auto current = nlohmann::json::parse(read_bytes(files.progress));
  EXPECT_EQ(current.at("processed_frames"), 1);
  EXPECT_EQ(current.at("source_time_seconds"), 0.0);
  EXPECT_FALSE(fs::exists(files.temporary));
  EXPECT_EQ(publisher.successful_publications(), 2u);
}

TEST(OfflineSbsProgressTest, FailedPublicationPreservesThePreviousSnapshotAndCanRetry) {
  progress_files_t files;
  const auto media = timed_media();
  offline_sbs::worker_progress_test_publisher_t publisher {files.progress, std::string {job_id}};
  publisher.publish("convert", 0, &media, std::chrono::steady_clock::time_point {});
  const auto previous = read_bytes(files.progress);
  ASSERT_TRUE(fs::create_directory(files.temporary));
  write_bytes(files.temporary / "blocker", "owned blocker");
  const auto due = std::chrono::steady_clock::time_point {} + 200ms;
  EXPECT_THROW(publisher.publish("convert", 1, &media, due), std::runtime_error);
  EXPECT_EQ(read_bytes(files.progress), previous);
  EXPECT_EQ(publisher.successful_publications(), 1u);
  ASSERT_TRUE(fs::remove(files.temporary / "blocker"));
  ASSERT_TRUE(fs::remove(files.temporary));
  publisher.publish("convert", 1, &media, due);
  EXPECT_EQ(nlohmann::json::parse(read_bytes(files.progress)).at("processed_frames"), 1);
  EXPECT_FALSE(fs::exists(files.temporary));
  EXPECT_EQ(publisher.successful_publications(), 2u);
}

TEST(OfflineSbsProgressTest, ActualManagerReaderRejectsWrongIdentityDuplicateKeysAndInvalidCounts) {
  progress_files_t files;
  const auto media = timed_media();
  auto context = reader_context(files.progress);
  offline_sbs::worker_progress_test_publisher_t publisher {files.progress, std::string {job_id}};
  publisher.publish("evaluate", 1, &media, std::chrono::steady_clock::time_point {});
  EXPECT_EQ(offline_sbs::read_worker_progress_for_test(context).processed_frames, 1u);
  context.job_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
  EXPECT_THROW(offline_sbs::read_worker_progress_for_test(context), std::runtime_error);
  context.job_id = job_id;
  publisher.publish("evaluate", 4, &media, std::chrono::steady_clock::time_point {} + 200ms);
  EXPECT_EQ(nlohmann::json::parse(read_bytes(files.progress)).at("processed_frames"), 4);
  EXPECT_THROW(offline_sbs::read_worker_progress_for_test(context), std::runtime_error);
  write_bytes(files.progress, R"({"schema":1,"schema":1,"job_id":"11111111-2222-3333-4444-555555555555","phase":"evaluate","processed_frames":1})");
  EXPECT_THROW(offline_sbs::read_worker_progress_for_test(context), std::runtime_error);
  publisher.publish("complete", 3, &media, std::chrono::steady_clock::time_point {} + 400ms);
  EXPECT_EQ(offline_sbs::read_worker_progress_for_test(context).processed_frames, 3u);
  EXPECT_FALSE(fs::exists(files.temporary));
}

TEST(OfflineSbsProgressTest, UnpacedUpdatesRetainTheExactFinalSnapshot) {
  progress_files_t files;
  offline_sbs::media_contract_t media;
  media.time_base = {1, 1000};
  media.frames.reserve(1000);
  for (std::uint64_t frame = 0; frame < 1000; ++frame) {
    media.frames.push_back({frame + 1, 1000 + 20 * static_cast<std::int64_t>(frame), 20});
  }
  offline_sbs::worker_progress_test_publisher_t publisher {files.progress, std::string {job_id}};
  for (std::uint64_t processed = 0; processed <= 1000; ++processed) {
    publisher.publish("convert", processed, &media, std::chrono::steady_clock::time_point {} + std::chrono::milliseconds {static_cast<std::int64_t>(processed)});
  }
  const nlohmann::json expected {
    {"schema", 1},
    {"job_id", job_id},
    {"phase", "convert"},
    {"processed_frames", 1000},
    {"total_frames", 1000},
    {"source_time_seconds", 19.98},
    {"source_duration_seconds", 20.0},
  };
  EXPECT_EQ(read_bytes(files.progress), expected.dump(2) + "\n");
  const auto consumed = offline_sbs::read_worker_progress_for_test(reader_context(files.progress));
  EXPECT_EQ(consumed.phase, "convert");
  EXPECT_EQ(consumed.processed_frames, 1000u);
  EXPECT_EQ(consumed.total_frames, 1000u);
  EXPECT_EQ(consumed.source_time_seconds, 19.98);
  EXPECT_EQ(consumed.source_duration_seconds, 20.0);
  EXPECT_FALSE(fs::exists(files.temporary));
  RecordProperty("requested_updates", 1001);
  RecordProperty("successful_publications", static_cast<int>(publisher.successful_publications()));
}

TEST(OfflineSbsProgressTest, SuppressedUpdatesDoNotMaterializeOrWriteAndTheExactDeadlineCanRetry) {
  progress_files_t files;
  auto media = timed_media();
  offline_sbs::worker_progress_test_publisher_t publisher {files.progress, std::string {job_id}};
  const auto start = std::chrono::steady_clock::time_point {};
  publisher.publish("convert", 0, &media, start);
  const auto previous = read_bytes(files.progress);
  ASSERT_TRUE(fs::create_directory(files.temporary));
  write_bytes(files.temporary / "blocker", "owned blocker");
  // A suppressed call must not evaluate the media-derived JSON fields either.
  media.time_base.denominator = 0;
  EXPECT_NO_THROW(publisher.publish("convert", 1, &media, start + 1ms));
  media.time_base.denominator = 1000;
  EXPECT_NO_THROW(publisher.publish("convert", 2, &media, start + 199ms));
  EXPECT_EQ(read_bytes(files.progress), previous);
  EXPECT_EQ(publisher.successful_publications(), 1u);
  EXPECT_TRUE(fs::exists(files.temporary / "blocker"));
  EXPECT_THROW(publisher.publish("convert", 1, &media, start + 200ms), std::runtime_error);
  EXPECT_EQ(read_bytes(files.progress), previous);
  EXPECT_EQ(publisher.successful_publications(), 1u);
  ASSERT_TRUE(fs::remove(files.temporary / "blocker"));
  ASSERT_TRUE(fs::remove(files.temporary));
  publisher.publish("convert", 1, &media, start + 200ms);
  EXPECT_EQ(publisher.successful_publications(), 2u);
  EXPECT_EQ(offline_sbs::read_worker_progress_for_test(reader_context(files.progress)).processed_frames, 1u);
  EXPECT_FALSE(fs::exists(files.temporary));
}

TEST(OfflineSbsProgressTest, PhaseChangeAndFinalSourceFramePublishWithoutWaitingForTheDeadline) {
  progress_files_t files;
  const auto media = timed_media();
  offline_sbs::worker_progress_test_publisher_t publisher {files.progress, std::string {job_id}};
  const auto start = std::chrono::steady_clock::time_point {};
  publisher.publish("probe", 0, nullptr, start);
  publisher.publish("convert", 0, &media, start + 1ms);
  EXPECT_EQ(publisher.successful_publications(), 2u);
  publisher.publish("convert", 3, &media, start + 2ms);
  EXPECT_EQ(publisher.successful_publications(), 3u);
  auto actual = offline_sbs::read_worker_progress_for_test(reader_context(files.progress));
  EXPECT_EQ(actual.phase, "convert");
  EXPECT_EQ(actual.processed_frames, 3u);
  publisher.publish("complete", 3, &media, start + 3ms);
  EXPECT_EQ(publisher.successful_publications(), 4u);
  actual = offline_sbs::read_worker_progress_for_test(reader_context(files.progress));
  EXPECT_EQ(actual.phase, "complete");
  EXPECT_EQ(actual.total_frames, 3u);
  EXPECT_FALSE(fs::exists(files.temporary));
}

TEST(OfflineSbsProgressTest, FailedForcedPhaseKeepsTheLastSuccessfulPhaseAndDeadline) {
  progress_files_t files;
  const auto media = timed_media();
  offline_sbs::worker_progress_test_publisher_t publisher {files.progress, std::string {job_id}};
  const auto start = std::chrono::steady_clock::time_point {};
  publisher.publish("convert", 0, &media, start);
  const auto previous = read_bytes(files.progress);
  ASSERT_TRUE(fs::create_directory(files.temporary));
  write_bytes(files.temporary / "blocker", "owned blocker");
  EXPECT_THROW(publisher.publish("mux", 2, &media, start + 1ms), std::runtime_error);
  EXPECT_EQ(publisher.successful_publications(), 1u);
  // A failed phase change must not make the still-current phase a forced update.
  EXPECT_NO_THROW(publisher.publish("convert", 1, &media, start + 2ms));
  EXPECT_EQ(read_bytes(files.progress), previous);
  ASSERT_TRUE(fs::remove(files.temporary / "blocker"));
  ASSERT_TRUE(fs::remove(files.temporary));
  publisher.publish("mux", 2, &media, start + 3ms);
  EXPECT_EQ(publisher.successful_publications(), 2u);
  const auto actual = offline_sbs::read_worker_progress_for_test(reader_context(files.progress));
  EXPECT_EQ(actual.phase, "mux");
  EXPECT_EQ(actual.processed_frames, 2u);
  EXPECT_FALSE(fs::exists(files.temporary));
}
