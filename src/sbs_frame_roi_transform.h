/**
 * @file src/sbs_frame_roi_transform.h
 * @brief CPU ownership identity for double-buffered Host SBS GPU ROI transforms.
 *
 * ROI geometry, flags, reset debt, and ROI generation are GPU authority and will live in the
 * paired GPU transform bank. This file intentionally mirrors none of those decisions. It only
 * prevents the asynchronous depth pipeline from pairing a GPU bank with the wrong source frame,
 * model shape, backend generation, or accepted-enqueue version.
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace models {
  inline constexpr std::uint32_t frame_roi_transform_contract_version = 1;
  inline constexpr std::uint32_t frame_roi_model_patch_size = 14;
  inline constexpr std::uint32_t frame_roi_transform_bank_count = 2;
  inline constexpr std::uint32_t invalid_frame_roi_transform_bank =
    std::numeric_limits<std::uint32_t>::max();

  struct frame_roi_transform_identity {
    std::uint32_t contract_version = frame_roi_transform_contract_version;
    std::uint32_t gpu_bank_index = invalid_frame_roi_transform_bank;
    std::uint64_t transform_version = 0;
    std::uint64_t source_frame_id = 0;
    std::uint32_t backend_generation = 0;
    std::uint32_t source_width = 0;
    std::uint32_t source_height = 0;
    std::uint32_t model_width = 0;
    std::uint32_t model_height = 0;

    [[nodiscard]] bool is_valid_candidate() const {
      return contract_version == frame_roi_transform_contract_version &&
             gpu_bank_index < frame_roi_transform_bank_count &&
             transform_version == 0 &&
             source_width > 0 && source_height > 0 &&
             model_width > 0 && model_height > 0 &&
             model_width % frame_roi_model_patch_size == 0 &&
             model_height % frame_roi_model_patch_size == 0;
    }

    [[nodiscard]] bool is_committed() const {
      if (transform_version == 0) {
        return false;
      }
      auto candidate = *this;
      candidate.transform_version = 0;
      return candidate.is_valid_candidate();
    }
  };

  [[nodiscard]] inline frame_roi_transform_identity
  make_frame_roi_transform_identity(
    std::uint64_t source_frame_id,
    std::uint32_t source_width,
    std::uint32_t source_height,
    std::uint32_t model_width,
    std::uint32_t model_height,
    std::uint32_t backend_generation,
    std::uint32_t gpu_bank_index
  ) {
    frame_roi_transform_identity result;
    result.gpu_bank_index = gpu_bank_index;
    result.source_frame_id = source_frame_id;
    result.backend_generation = backend_generation;
    result.source_width = source_width;
    result.source_height = source_height;
    result.model_width = model_width;
    result.model_height = model_height;
    return result;
  }

  /**
   * CPU lifecycle for two paired GPU transform banks.
   *
   * reserve() assigns a monotonic nonzero version before GPU dispatch while preserving the prior
   * completed bank. Work not accepted by TensorRT is rolled back; accepted work transitions to
   * pending only after CUDA/D3D unmap succeeds, or to an explicit orphan that must be dropped.
   */
  class frame_roi_transform_buffer {
  public:
    [[nodiscard]] std::optional<std::uint32_t> writable_bank() const {
      if (reserved_bank_ || pending_bank_ || orphaned_frame_id_) {
        return std::nullopt;
      }
      auto bank = next_write_bank_;
      if (completed_bank_ && bank == *completed_bank_) {
        bank ^= 1u;
      }
      return bank;
    }

    [[nodiscard]] bool can_reserve(
      const frame_roi_transform_identity &candidate
    ) const {
      const auto writable = writable_bank();
      return writable && candidate.is_valid_candidate() &&
              candidate.gpu_bank_index == *writable;
    }

    /**
     * Reserve one free GPU bank and allocate its accepted-enqueue version before any GPU pass
     * writes that bank. A reservation never overwrites the completed bank and is reversible until
     * TensorRT accepts the matching inference.
     */
    [[nodiscard]] std::optional<frame_roi_transform_identity> reserve(
      frame_roi_transform_identity candidate
    ) {
      if (!can_reserve(candidate)) {
        return std::nullopt;
      }
      candidate.transform_version = next_transform_version_;
      if (++next_transform_version_ == 0) {
        // Version zero is reserved for uncommitted candidates.
        next_transform_version_ = 1;
      }
      const auto bank = candidate.gpu_bank_index;
      slots_[bank] = candidate;
      reserved_bank_ = bank;
      return candidate;
    }

    [[nodiscard]] bool is_reserved(
      const frame_roi_transform_identity &expected
    ) const {
      return matching_identity(reserved_bank_, expected) != nullptr;
    }

    /**
     * Roll back work that was never accepted by TensorRT. Version numbers remain monotonic and
     * may contain gaps; reusing a rejected version would make stale GPU contents harder to detect.
     */
    bool rollback_reserved(const frame_roi_transform_identity &expected) {
      if (!matching_identity(reserved_bank_, expected)) {
        return false;
      }
      const auto bank = *reserved_bank_;
      slots_[bank] = {};
      reserved_bank_.reset();
      return true;
    }

    /**
     * Commit the exact reservation after enqueue and CUDA/D3D unmap both succeed.
     *
     * With one estimator thread and one inference in flight this transition cannot fail after a
     * successful reserve(). The boolean remains a defensive contract check for tests and future
     * callers; an accepted inference whose transition fails must use orphan_reserved_enqueued().
     */
    bool commit_reserved_enqueued(
      const frame_roi_transform_identity &expected
    ) {
      if (!matching_identity(reserved_bank_, expected)) {
        return false;
      }
      const auto bank = *reserved_bank_;
      pending_bank_ = bank;
      reserved_bank_.reset();
      next_write_bank_ = bank ^ 1u;
      return true;
    }

    /**
     * Record an inference that TensorRT accepted but which cannot safely be paired with a GPU
     * transform bank. Its eventual output must be consumed and dropped without normalization.
     */
    void orphan_reserved_enqueued(std::uint64_t source_frame_id) {
      if (reserved_bank_) {
        const auto bank = *reserved_bank_;
        slots_[bank] = {};
        reserved_bank_.reset();
      }
      // There can be only one accepted inference. Preserve the source identity even for frame 0.
      orphaned_frame_id_ = source_frame_id;
    }

    bool complete(const frame_roi_transform_identity &expected) {
      if (!matching_identity(pending_bank_, expected)) {
        return false;
      }
      completed_bank_ = pending_bank_;
      pending_bank_.reset();
      return true;
    }

    [[nodiscard]] const frame_roi_transform_identity *pending_for(
      std::uint64_t source_frame_id
    ) const {
      return matching_bank(pending_bank_, source_frame_id);
    }

    [[nodiscard]] const frame_roi_transform_identity *completed_for(
      std::uint64_t source_frame_id
    ) const {
      return matching_bank(completed_bank_, source_frame_id);
    }

    [[nodiscard]] bool orphaned_for(std::uint64_t source_frame_id) const {
      return orphaned_frame_id_ && *orphaned_frame_id_ == source_frame_id;
    }

    /**
     * Drop one accepted completion without promoting it to completed. This handles both an exact
     * pending bank and an explicitly orphaned accepted inference.
     */
    bool drop_in_flight(std::uint64_t source_frame_id) {
      if (pending_bank_ &&
          slots_[*pending_bank_].source_frame_id == source_frame_id) {
        const auto bank = *pending_bank_;
        slots_[bank] = {};
        pending_bank_.reset();
        return true;
      }
      if (orphaned_for(source_frame_id)) {
        orphaned_frame_id_.reset();
        return true;
      }
      return false;
    }

    /**
     * Terminal CUDA errors invalidate every non-completed ownership state. The last completed bank
     * remains available for repeated output, but no failed raw output can later be normalized.
     */
    void abandon_in_flight() {
      if (reserved_bank_) {
        slots_[*reserved_bank_] = {};
      }
      if (pending_bank_) {
        slots_[*pending_bank_] = {};
      }
      reserved_bank_.reset();
      pending_bank_.reset();
      orphaned_frame_id_.reset();
    }

    [[nodiscard]] std::optional<std::uint32_t> reserved_bank() const {
      return reserved_bank_;
    }

    [[nodiscard]] std::optional<std::uint32_t> pending_bank() const {
      return pending_bank_;
    }

    [[nodiscard]] std::optional<std::uint32_t> completed_bank() const {
      return completed_bank_;
    }

    [[nodiscard]] bool has_pending() const {
      return pending_bank_.has_value();
    }

    [[nodiscard]] bool has_reserved() const {
      return reserved_bank_.has_value();
    }

    [[nodiscard]] bool has_orphaned() const {
      return orphaned_frame_id_.has_value();
    }

  private:
    [[nodiscard]] static bool identities_match(
      const frame_roi_transform_identity &actual,
      const frame_roi_transform_identity &expected
    ) {
      return actual.is_committed() && expected.is_committed() &&
             actual.contract_version == expected.contract_version &&
             actual.gpu_bank_index == expected.gpu_bank_index &&
             actual.transform_version == expected.transform_version &&
             actual.source_frame_id == expected.source_frame_id &&
             actual.backend_generation == expected.backend_generation &&
             actual.source_width == expected.source_width &&
             actual.source_height == expected.source_height &&
             actual.model_width == expected.model_width &&
             actual.model_height == expected.model_height;
    }

    [[nodiscard]] const frame_roi_transform_identity *matching_identity(
      const std::optional<std::uint32_t> &bank,
      const frame_roi_transform_identity &expected
    ) const {
      if (!bank || !identities_match(slots_[*bank], expected)) {
        return nullptr;
      }
      return &slots_[*bank];
    }

    [[nodiscard]] const frame_roi_transform_identity *matching_bank(
      const std::optional<std::uint32_t> &bank,
      std::uint64_t source_frame_id
    ) const {
      if (!bank || slots_[*bank].source_frame_id != source_frame_id ||
          !slots_[*bank].is_committed()) {
        return nullptr;
      }
      return &slots_[*bank];
    }

    std::array<
      frame_roi_transform_identity,
      frame_roi_transform_bank_count
    > slots_ {};
    std::optional<std::uint32_t> reserved_bank_;
    std::optional<std::uint32_t> pending_bank_;
    std::optional<std::uint32_t> completed_bank_;
    std::optional<std::uint64_t> orphaned_frame_id_;
    std::uint32_t next_write_bank_ = 0;
    std::uint64_t next_transform_version_ = 1;
  };
}  // namespace models
