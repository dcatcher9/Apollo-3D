/**
 * @file src/offline_sbs_filesystem.h
 * @brief No-follow filesystem traversal for offline SBS retained artifacts.
 */
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace offline_sbs::safe_filesystem {
  /**
   * Access reserved when a tree is pinned.
   *
   * On Windows, `remove` reserves delete access up front. On POSIX it records
   * removal intent while retaining descriptor pins; final parent-directory
   * permissions are evaluated by `unlinkat`. In both cases operations stay
   * bound to the pinned root rather than resolving its pathname again.
   */
  enum class tree_access_e {
    measure_only,
    remove,
  };

  /**
   * Logical unnamed-stream usage below one tree root.
   *
   * Directory counts include the root when the root is a directory. Reparse
   * points are counted but never traversed and do not contribute file bytes.
   * Hard links are counted once per directory entry.
   */
  struct tree_measurement_t {
    bool root_exists = true;
    std::uint64_t regular_file_bytes = 0;
    std::uint64_t regular_file_count = 0;
    std::uint64_t directory_count = 0;
    std::uint64_t reparse_point_count = 0;
  };

  /**
   * A move-only identity pin for a tree root.
   *
   * On Windows, `open()` opens every existing path component with
   * `FILE_FLAG_OPEN_REPARSE_POINT`. Ancestor pins and removal-access root pins
   * omit `FILE_SHARE_DELETE` to preserve path identity. A measure-only root pin
   * shares delete so an aggregate scan can coexist with retained removal pins,
   * and verifies every enumerated child against the volume and file identity
   * of its opened no-follow handle. The root handle remains open for this
   * object's lifetime. Measurement and removal never recurse through a symlink,
   * junction, or other reparse point.
   *
   * On POSIX, `open()` walks from the filesystem root with descriptor-relative
   * no-follow opens and retains descriptors for the final parent and root
   * object. Measurement is performed only from those descriptors, so renaming
   * an ancestor or the root does not redirect a pin to a replacement pathname.
   * Removal validates the retained root identity at its original parent/name
   * before the final `unlinkat`; a moved or replaced final name fails closed.
   *
   * The root must be an absolute path below a filesystem root. Opening a volume
   * root itself is deliberately rejected. A reparse point may be accepted only
   * as the final root object (and only where the platform can open the link
   * itself) so `remove()` can safely remove the link; `measure()` rejects a
   * reparse-point root rather than reporting a misleading zero-byte tree.
   */
  class pinned_tree_t {
  public:
    pinned_tree_t(pinned_tree_t &&other) noexcept;
    pinned_tree_t &operator=(pinned_tree_t &&other) noexcept;
    ~pinned_tree_t();

    pinned_tree_t(const pinned_tree_t &) = delete;
    pinned_tree_t &operator=(const pinned_tree_t &) = delete;

    /**
     * Pin `root` and reserve the requested access.
     *
     * Returns `std::nullopt` for a missing path or any safety/access failure and
     * writes a diagnostic to `error`.
     */
    static std::optional<pinned_tree_t> open(
      const std::filesystem::path &root,
      tree_access_e access,
      std::string &error
    );

    /**
     * Measure regular files reachable without crossing a reparse point.
     *
     * Sizes and identities are queried from the same opened child handles or
     * descriptors.
     * Identity changes observed during traversal, access failure, unsupported
     * identity queries, arithmetic overflow, excessive depth, or excessive
     * entry count fail closed with `std::nullopt`. This is a bounded
     * point-in-time best effort, not a frozen filesystem snapshot: a child
     * created after its parent was enumerated or a file grown after its size
     * query may not be reflected in that measurement.
     */
    [[nodiscard]] std::optional<tree_measurement_t> measure(
      std::string &error
    ) const;

    /**
     * Remove the pinned root in postorder without following reparse points.
     *
     * On Windows, every entry is removed by dispositioning the exact no-follow
     * handle that was inspected. On POSIX, traversal stays descriptor-relative
     * and each name is identity-validated before `unlinkat`; POSIX does not
     * provide an atomic identity-conditional unlink, so callers must prevent
     * adversarial mutation of the pinned parent during the final validation /
     * unlink interval. A reparse point is removed as a link; its target is
     * never opened. Concurrent mutation, traversal depth, and entry count are
     * bounded and then fail safely. This requires `tree_access_e::remove`.
     */
    [[nodiscard]] bool remove(std::string &error);

    /**
     * Remove one direct child of the pinned root without resolving the root
     * pathname again.
     *
     * `child_name` must be exactly one non-empty relative filename component;
     * absolute paths, separators, "." and ".." are rejected. A missing child
     * is successful. Reparse points are removed as links and never traversed.
     * The root remains pinned and usable after the operation. This requires a
     * directory root opened with `tree_access_e::remove`.
     */
    [[nodiscard]] bool remove_child(
      const std::filesystem::path &child_name,
      std::string &error
    );

    [[nodiscard]] const std::filesystem::path &path() const noexcept;
    [[nodiscard]] tree_access_e access() const noexcept;

  private:
    struct impl_t;
    explicit pinned_tree_t(std::unique_ptr<impl_t> impl) noexcept;

    std::unique_ptr<impl_t> impl_;
  };

  /**
   * One-shot no-follow measurement.
   *
   * A missing root is successful and returns `root_exists == false`. The root
   * and its ancestors are identity-pinned for the entire call. The same
   * point-in-time best-effort concurrency limits as `measure()` apply.
   */
  [[nodiscard]] std::optional<tree_measurement_t> measure_tree_no_follow(
    const std::filesystem::path &root,
    std::string &error
  );

  /**
   * One-shot no-follow removal.
   *
   * A missing root is successful. The function atomically binds the root that
   * exists when it opens it, then removes only that object and descendants
   * reached through non-reparse directory handles. Callers needing identity
   * continuity from an earlier point in time should retain a `pinned_tree_t`
   * instead.
   */
  [[nodiscard]] bool remove_tree_no_follow(
    const std::filesystem::path &root,
    std::string &error
  );
}  // namespace offline_sbs::safe_filesystem
