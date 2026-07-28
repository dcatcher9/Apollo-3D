#include "src/offline_sbs_filesystem.h"

#include <chrono>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
  #include <windows.h>
  #include <winioctl.h>
#endif

namespace {
  namespace fs = std::filesystem;
  namespace safe_fs = offline_sbs::safe_filesystem;

  class temporary_tree_t {
  public:
    temporary_tree_t() {
      path = fs::temp_directory_path() /
             ("sunshine3d-no-follow-" +
              std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()
              ));
      fs::create_directories(path);
    }

    ~temporary_tree_t() {
      std::error_code ignored;
      fs::remove_all(path, ignored);
    }

    fs::path path;
  };

  void write_bytes(
    const fs::path &path,
    const std::size_t count,
    const char byte = 'x'
  ) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    const std::string contents(count, byte);
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  }

#ifdef _WIN32
  struct mount_point_reparse_buffer_t {
    DWORD reparse_tag;
    WORD reparse_data_length;
    WORD reserved;
    WORD substitute_name_offset;
    WORD substitute_name_length;
    WORD print_name_offset;
    WORD print_name_length;
    WCHAR path_buffer[1];
  };

  static_assert(offsetof(mount_point_reparse_buffer_t, path_buffer) == 16);

  bool create_junction(
    const fs::path &junction,
    const fs::path &target,
    std::string &error
  ) {
    std::error_code ec;
    fs::create_directory(junction, ec);
    if (ec) {
      error = "cannot create junction directory: " + ec.message();
      return false;
    }
    const auto absolute_target = fs::absolute(target).lexically_normal();
    const std::wstring substitute = LR"(\??\)" + absolute_target.native();
    const std::wstring print_name = absolute_target.native();
    const auto substitute_bytes =
      substitute.size() * sizeof(wchar_t);
    const auto print_bytes = print_name.size() * sizeof(wchar_t);
    const auto path_bytes =
      substitute_bytes + sizeof(wchar_t) +
      print_bytes + sizeof(wchar_t);
    const auto buffer_bytes =
      offsetof(mount_point_reparse_buffer_t, path_buffer) + path_bytes;
    std::vector<std::byte> storage(buffer_bytes);
    auto *buffer = reinterpret_cast<mount_point_reparse_buffer_t *>(
      storage.data()
    );
    buffer->reparse_tag = IO_REPARSE_TAG_MOUNT_POINT;
    buffer->reparse_data_length = static_cast<WORD>(
      sizeof(WORD) * 4 + path_bytes
    );
    buffer->reserved = 0;
    buffer->substitute_name_offset = 0;
    buffer->substitute_name_length =
      static_cast<WORD>(substitute_bytes);
    buffer->print_name_offset =
      static_cast<WORD>(substitute_bytes + sizeof(wchar_t));
    buffer->print_name_length = static_cast<WORD>(print_bytes);
    std::memcpy(
      buffer->path_buffer,
      substitute.data(),
      substitute_bytes
    );
    buffer->path_buffer[substitute.size()] = L'\0';
    std::memcpy(
      reinterpret_cast<std::byte *>(buffer->path_buffer) +
        buffer->print_name_offset,
      print_name.data(),
      print_bytes
    );
    *reinterpret_cast<wchar_t *>(
      reinterpret_cast<std::byte *>(buffer->path_buffer) +
      buffer->print_name_offset +
      print_bytes
    ) = L'\0';

    const HANDLE handle = CreateFileW(
      junction.c_str(),
      GENERIC_WRITE,
      0,
      nullptr,
      OPEN_EXISTING,
      FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
      nullptr
    );
    if (handle == INVALID_HANDLE_VALUE) {
      error =
        "cannot open junction directory (Windows error " +
        std::to_string(GetLastError()) + ")";
      fs::remove(junction, ec);
      return false;
    }
    DWORD returned = 0;
    const BOOL created = DeviceIoControl(
      handle,
      FSCTL_SET_REPARSE_POINT,
      buffer,
      static_cast<DWORD>(
        offsetof(mount_point_reparse_buffer_t, path_buffer) + path_bytes
      ),
      nullptr,
      0,
      &returned,
      nullptr
    );
    const auto create_error = created ? ERROR_SUCCESS : GetLastError();
    CloseHandle(handle);
    if (!created) {
      error =
        "cannot create junction reparse point (Windows error " +
        std::to_string(create_error) + ")";
      fs::remove(junction, ec);
      return false;
    }
    return true;
  }

  class junction_guard_t {
  public:
    explicit junction_guard_t(fs::path path):
        path_(std::move(path)) {
    }

    ~junction_guard_t() {
      if (!path_.empty()) {
        RemoveDirectoryW(path_.c_str());
      }
    }

    junction_guard_t(const junction_guard_t &) = delete;
    junction_guard_t &operator=(const junction_guard_t &) = delete;

  private:
    fs::path path_;
  };
#endif
}  // namespace

TEST(OfflineSbsFilesystem, MeasuresAndRemovesTheSamePinnedTreeIdentity) {
  temporary_tree_t tree;
  const auto root = tree.path / "managed";
  write_bytes(root / "first.bin", 7, 'a');
  write_bytes(root / "nested" / "second.bin", 11, 'b');

  std::string error;
  auto pinned = safe_fs::pinned_tree_t::open(
    root,
    safe_fs::tree_access_e::remove,
    error
  );
  ASSERT_TRUE(pinned) << error;
#ifdef _WIN32
  const auto swapped = tree.path / "managed-swapped";
  EXPECT_FALSE(MoveFileExW(
    root.c_str(),
    swapped.c_str(),
    MOVEFILE_WRITE_THROUGH
  ));
#endif

  const auto measurement = pinned->measure(error);
  ASSERT_TRUE(measurement) << error;
  EXPECT_TRUE(measurement->root_exists);
  EXPECT_EQ(measurement->regular_file_bytes, 18u);
  EXPECT_EQ(measurement->regular_file_count, 2u);
  EXPECT_EQ(measurement->directory_count, 2u);
  EXPECT_EQ(measurement->reparse_point_count, 0u);

  ASSERT_TRUE(pinned->remove(error)) << error;
  EXPECT_FALSE(fs::exists(root));
}

TEST(OfflineSbsFilesystem, MissingOneShotTreesAreEmptyAndAlreadyRemoved) {
  temporary_tree_t tree;
  const auto missing = tree.path / "missing";
  std::string error;

  const auto measurement =
    safe_fs::measure_tree_no_follow(missing, error);
  ASSERT_TRUE(measurement) << error;
  EXPECT_FALSE(measurement->root_exists);
  EXPECT_EQ(measurement->regular_file_bytes, 0u);
  EXPECT_TRUE(safe_fs::remove_tree_no_follow(missing, error)) << error;
}

TEST(OfflineSbsFilesystem, AggregateMeasurementCoexistsWithChildRemovalPin) {
  temporary_tree_t tree;
  const auto aggregate_root = tree.path / "aggregate";
  const auto retained_child = aggregate_root / "retained-job";
  write_bytes(retained_child / "artifact.bin", 37, 'a');

  std::string error;
  auto child_pin = safe_fs::pinned_tree_t::open(
    retained_child,
    safe_fs::tree_access_e::remove,
    error
  );
  ASSERT_TRUE(child_pin) << error;

  const auto aggregate =
    safe_fs::measure_tree_no_follow(aggregate_root, error);
  ASSERT_TRUE(aggregate) << error;
  EXPECT_EQ(aggregate->regular_file_bytes, 37u);
  EXPECT_EQ(aggregate->regular_file_count, 1u);
  EXPECT_EQ(aggregate->directory_count, 2u);

  ASSERT_TRUE(child_pin->remove(error)) << error;
  ASSERT_TRUE(
    safe_fs::remove_tree_no_follow(aggregate_root, error)
  ) << error;
}

TEST(OfflineSbsFilesystem, RemovesDirectChildrenAndKeepsRootPinUsable) {
  temporary_tree_t tree;
  const auto root = tree.path / "managed";
  write_bytes(root / "native-work" / "nested" / "partial.bin", 23, 'p');
  write_bytes(root / "keep.bin", 7, 'k');

  std::string error;
  auto pinned = safe_fs::pinned_tree_t::open(
    root,
    safe_fs::tree_access_e::remove,
    error
  );
  ASSERT_TRUE(pinned) << error;

  ASSERT_TRUE(pinned->remove_child("native-work", error)) << error;
  EXPECT_FALSE(fs::exists(root / "native-work"));
  EXPECT_TRUE(fs::is_regular_file(root / "keep.bin"));

  ASSERT_TRUE(pinned->remove_child("native-work", error)) << error;
  write_bytes(root / "second-work" / "later.bin", 13, 'l');
  ASSERT_TRUE(pinned->remove_child("second-work", error)) << error;

  const auto measurement = pinned->measure(error);
  ASSERT_TRUE(measurement) << error;
  EXPECT_EQ(measurement->regular_file_bytes, 7u);
  EXPECT_EQ(measurement->regular_file_count, 1u);
  EXPECT_EQ(measurement->directory_count, 1u);

  ASSERT_TRUE(pinned->remove(error)) << error;
  EXPECT_FALSE(fs::exists(root));
}

TEST(OfflineSbsFilesystem, RejectsUnsafeDirectChildNames) {
  temporary_tree_t tree;
  const auto root = tree.path / "managed";
  write_bytes(root / "native-work" / "sentinel.bin", 9, 's');

  std::string error;
  auto pinned = safe_fs::pinned_tree_t::open(
    root,
    safe_fs::tree_access_e::remove,
    error
  );
  ASSERT_TRUE(pinned) << error;

  for (const fs::path &unsafe : {
         fs::path {},
         fs::path {"."},
         fs::path {".."},
         fs::path {"nested"} / "child",
         tree.path / "outside",
       }) {
    error.clear();
    EXPECT_FALSE(pinned->remove_child(unsafe, error))
      << "unexpectedly accepted " << unsafe;
    EXPECT_NE(error.find("one safe relative filename"), std::string::npos)
      << error;
  }
  EXPECT_TRUE(fs::is_regular_file(root / "native-work" / "sentinel.bin"));

  auto measure_only = safe_fs::pinned_tree_t::open(
    root,
    safe_fs::tree_access_e::measure_only,
    error
  );
  ASSERT_TRUE(measure_only) << error;
  EXPECT_FALSE(measure_only->remove_child("native-work", error));
  EXPECT_NE(error.find("removal access"), std::string::npos);
}

TEST(OfflineSbsFilesystem, RejectsRelativeAndFilesystemRootPaths) {
  std::string error;
  EXPECT_FALSE(safe_fs::measure_tree_no_follow("relative-tree", error));
  EXPECT_NE(error.find("absolute"), std::string::npos);

  error.clear();
  EXPECT_FALSE(
    safe_fs::remove_tree_no_follow(fs::current_path().root_path(), error)
  );
  EXPECT_NE(error.find("root"), std::string::npos);
}

#if defined(__linux__)
TEST(OfflineSbsFilesystem, PosixPinSurvivesAncestorRename) {
  temporary_tree_t tree;
  const auto ancestor = tree.path / "ancestor";
  const auto root = ancestor / "managed";
  const auto moved_ancestor = tree.path / "ancestor-original";
  write_bytes(root / "original.bin", 17, 'o');

  std::string error;
  auto pinned = safe_fs::pinned_tree_t::open(
    root,
    safe_fs::tree_access_e::remove,
    error
  );
  ASSERT_TRUE(pinned) << error;

  fs::rename(ancestor, moved_ancestor);
  write_bytes(root / "replacement.bin", 31, 'r');

  const auto measurement = pinned->measure(error);
  ASSERT_TRUE(measurement) << error;
  EXPECT_EQ(measurement->regular_file_bytes, 17u);
  EXPECT_EQ(measurement->regular_file_count, 1u);

  ASSERT_TRUE(pinned->remove(error)) << error;
  EXPECT_FALSE(fs::exists(moved_ancestor / "managed"));
  EXPECT_TRUE(fs::is_regular_file(root / "replacement.bin"));
  EXPECT_EQ(fs::file_size(root / "replacement.bin"), 31u);
}

TEST(OfflineSbsFilesystem, PosixPinRefusesReplacementAtRootName) {
  temporary_tree_t tree;
  const auto root = tree.path / "managed";
  const auto moved_root = tree.path / "managed-original";
  write_bytes(root / "original.bin", 19, 'o');

  std::string error;
  auto pinned = safe_fs::pinned_tree_t::open(
    root,
    safe_fs::tree_access_e::remove,
    error
  );
  ASSERT_TRUE(pinned) << error;

  fs::rename(root, moved_root);
  write_bytes(root / "replacement.bin", 37, 'r');

  const auto measurement = pinned->measure(error);
  ASSERT_TRUE(measurement) << error;
  EXPECT_EQ(measurement->regular_file_bytes, 19u);
  EXPECT_EQ(measurement->regular_file_count, 1u);

  EXPECT_FALSE(pinned->remove(error));
  EXPECT_NE(error.find("identity"), std::string::npos) << error;
  EXPECT_TRUE(fs::is_regular_file(moved_root / "original.bin"));
  EXPECT_EQ(fs::file_size(moved_root / "original.bin"), 19u);
  EXPECT_TRUE(fs::is_regular_file(root / "replacement.bin"));
  EXPECT_EQ(fs::file_size(root / "replacement.bin"), 37u);
}

TEST(OfflineSbsFilesystem, PosixRegularFilePinRefusesReplacement) {
  temporary_tree_t tree;
  const auto root = tree.path / "artifact.bin";
  const auto moved_root = tree.path / "artifact-original.bin";
  write_bytes(root, 23, 'o');

  std::string error;
  auto pinned = safe_fs::pinned_tree_t::open(
    root,
    safe_fs::tree_access_e::remove,
    error
  );
  ASSERT_TRUE(pinned) << error;

  fs::rename(root, moved_root);
  write_bytes(root, 43, 'r');

  const auto measurement = pinned->measure(error);
  ASSERT_TRUE(measurement) << error;
  EXPECT_EQ(measurement->regular_file_bytes, 23u);
  EXPECT_EQ(measurement->regular_file_count, 1u);

  EXPECT_FALSE(pinned->remove(error));
  EXPECT_NE(error.find("identity"), std::string::npos) << error;
  EXPECT_EQ(fs::file_size(moved_root), 23u);
  EXPECT_EQ(fs::file_size(root), 43u);
}

TEST(OfflineSbsFilesystem, PosixSymlinkTargetsStayOutsideTheTree) {
  temporary_tree_t tree;
  const auto root = tree.path / "managed";
  const auto target = tree.path / "outside";
  write_bytes(root / "owned.bin", 7, 'o');
  write_bytes(target / "sentinel.bin", 64 * 1024, 's');
  fs::create_directory_symlink(target, root / "redirect");
  fs::create_directory_symlink(root, root / "loop");

  std::string error;
  auto pinned = safe_fs::pinned_tree_t::open(
    root,
    safe_fs::tree_access_e::remove,
    error
  );
  ASSERT_TRUE(pinned) << error;

  for (int repetition = 0; repetition < 2; ++repetition) {
    const auto measurement = pinned->measure(error);
    ASSERT_TRUE(measurement) << error;
    EXPECT_EQ(measurement->regular_file_bytes, 7u);
    EXPECT_EQ(measurement->regular_file_count, 1u);
    EXPECT_EQ(measurement->directory_count, 1u);
    EXPECT_EQ(measurement->reparse_point_count, 2u);
  }

  ASSERT_TRUE(pinned->remove(error)) << error;
  EXPECT_FALSE(fs::exists(root));
  EXPECT_TRUE(fs::is_regular_file(target / "sentinel.bin"));
  EXPECT_EQ(fs::file_size(target / "sentinel.bin"), 64u * 1024u);
}

TEST(OfflineSbsFilesystem, PosixRootSymlinkRemovesOnlyTheLink) {
  temporary_tree_t tree;
  const auto target = tree.path / "outside";
  const auto link = tree.path / "root-link";
  write_bytes(target / "sentinel.bin", 29, 's');
  fs::create_directory_symlink(target, link);

  std::string error;
  EXPECT_FALSE(safe_fs::measure_tree_no_follow(link, error));
  EXPECT_NE(error.find("symbolic link"), std::string::npos) << error;
  ASSERT_TRUE(safe_fs::remove_tree_no_follow(link, error)) << error;
  EXPECT_FALSE(fs::exists(link));
  EXPECT_TRUE(fs::is_regular_file(target / "sentinel.bin"));
  EXPECT_EQ(fs::file_size(target / "sentinel.bin"), 29u);
}

TEST(OfflineSbsFilesystem, PosixRejectsSymlinkAncestor) {
  temporary_tree_t tree;
  const auto real_parent = tree.path / "real-parent";
  const auto linked_parent = tree.path / "linked-parent";
  write_bytes(real_parent / "managed" / "artifact.bin", 13, 'a');
  fs::create_directory_symlink(real_parent, linked_parent);

  std::string error;
  EXPECT_FALSE(
    safe_fs::measure_tree_no_follow(linked_parent / "managed", error)
  );
  EXPECT_NE(error.find("plain directory"), std::string::npos) << error;
}

TEST(OfflineSbsFilesystem, PosixMeasurementEnforcesDepthBound) {
  temporary_tree_t tree;
  const auto root = tree.path / "managed";
  auto current = root;
  fs::create_directory(current);
  for (int depth = 0; depth <= 256; ++depth) {
    current /= "d";
    fs::create_directory(current);
  }
  write_bytes(current / "artifact.bin", 1, 'a');

  std::string error;
  EXPECT_FALSE(safe_fs::measure_tree_no_follow(root, error));
  EXPECT_NE(error.find("directory depth"), std::string::npos) << error;
}
#endif  // defined(__linux__)

#ifdef _WIN32
TEST(OfflineSbsFilesystem, MeasurePinPreventsAncestorReplacement) {
  temporary_tree_t tree;
  const auto ancestor = tree.path / "ancestor";
  const auto root = ancestor / "managed";
  const auto replacement = tree.path / "ancestor-replacement";
  write_bytes(root / "artifact.bin", 17, 'a');

  std::string error;
  auto pinned = safe_fs::pinned_tree_t::open(
    root,
    safe_fs::tree_access_e::measure_only,
    error
  );
  ASSERT_TRUE(pinned) << error;
  EXPECT_FALSE(MoveFileExW(
    ancestor.c_str(),
    replacement.c_str(),
    MOVEFILE_WRITE_THROUGH
  ));

  const auto measurement = pinned->measure(error);
  ASSERT_TRUE(measurement) << error;
  EXPECT_EQ(measurement->regular_file_bytes, 17u);
  EXPECT_EQ(measurement->regular_file_count, 1u);
}

TEST(OfflineSbsFilesystem, JunctionTargetIsNeverMeasuredOrDeleted) {
  temporary_tree_t tree;
  const auto root = tree.path / "managed";
  const auto target = tree.path / "outside";
  const auto junction = root / "redirect";
  write_bytes(root / "owned.bin", 7, 'o');
  write_bytes(target / "sentinel.bin", 64 * 1024, 's');

  std::string error;
  ASSERT_TRUE(create_junction(junction, target, error)) << error;
  junction_guard_t junction_guard {junction};

  const auto measurement =
    safe_fs::measure_tree_no_follow(root, error);
  ASSERT_TRUE(measurement) << error;
  EXPECT_EQ(measurement->regular_file_bytes, 7u);
  EXPECT_EQ(measurement->regular_file_count, 1u);
  EXPECT_EQ(measurement->directory_count, 1u);
  EXPECT_EQ(measurement->reparse_point_count, 1u);

  ASSERT_TRUE(safe_fs::remove_tree_no_follow(root, error)) << error;
  EXPECT_FALSE(fs::exists(root));
  EXPECT_TRUE(fs::is_regular_file(target / "sentinel.bin"));
  EXPECT_EQ(fs::file_size(target / "sentinel.bin"), 64u * 1024u);
}

TEST(OfflineSbsFilesystem, RootJunctionFailsMeasurementAndRemovesOnlyLink) {
  temporary_tree_t tree;
  const auto target = tree.path / "outside";
  const auto junction = tree.path / "root-junction";
  write_bytes(target / "sentinel.bin", 29, 's');

  std::string error;
  ASSERT_TRUE(create_junction(junction, target, error)) << error;
  junction_guard_t junction_guard {junction};

  EXPECT_FALSE(safe_fs::measure_tree_no_follow(junction, error));
  EXPECT_NE(error.find("reparse point"), std::string::npos);
  ASSERT_TRUE(safe_fs::remove_tree_no_follow(junction, error)) << error;
  EXPECT_FALSE(fs::exists(junction));
  EXPECT_TRUE(fs::is_regular_file(target / "sentinel.bin"));
  EXPECT_EQ(fs::file_size(target / "sentinel.bin"), 29u);
}

TEST(OfflineSbsFilesystem, DirectChildJunctionRemovalKeepsTargetAndRoot) {
  temporary_tree_t tree;
  const auto root = tree.path / "managed";
  const auto target = tree.path / "outside";
  const auto junction = root / "native-work";
  write_bytes(root / "keep.bin", 5, 'k');
  write_bytes(target / "sentinel.bin", 41, 's');

  std::string error;
  ASSERT_TRUE(create_junction(junction, target, error)) << error;
  junction_guard_t junction_guard {junction};
  auto pinned = safe_fs::pinned_tree_t::open(
    root,
    safe_fs::tree_access_e::remove,
    error
  );
  ASSERT_TRUE(pinned) << error;

  ASSERT_TRUE(pinned->remove_child("native-work", error)) << error;
  EXPECT_FALSE(fs::exists(junction));
  EXPECT_TRUE(fs::is_regular_file(target / "sentinel.bin"));
  EXPECT_EQ(fs::file_size(target / "sentinel.bin"), 41u);

  const auto measurement = pinned->measure(error);
  ASSERT_TRUE(measurement) << error;
  EXPECT_EQ(measurement->regular_file_bytes, 5u);
  EXPECT_EQ(measurement->regular_file_count, 1u);
  EXPECT_EQ(measurement->directory_count, 1u);
}
#endif
