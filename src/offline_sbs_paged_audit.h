/** Bounded append-only scene diagnostics. The caller owns atomic filesystem publication. */
#pragma once

#include "offline_sbs_contract.h"
#include "offline_sbs_wire_contract.h"

#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace offline_sbs {
  class paged_scene_audit_writer_t {
  public:
    // Publish these exact bytes to a new manager-owned filename and return their
    // lowercase SHA-256. Never overwrite an already-published page.
    using publisher_t = std::function<std::string(std::string_view, std::string_view)>;

    explicit paged_scene_audit_writer_t(publisher_t publish) : publish_(std::move(publish)) {}

    void append(const scene_plan_t &scene) {
      if (ended_ || scene.scene_id != scene_count_ + 1 || scene.start_sequence != covered_end_) {
        throw std::runtime_error("scene audit append is not contiguous");
      }
      // Size each new record once. Compact JSON keeps whitespace independent of
      // nesting; the fixed reserve covers the envelope, index and array commas.
      const auto record_bytes = wire::scene_record_json(scene).dump().size() +
        (scene.boundary.semantic_cut ? wire::boundary_audit_json(scene.boundary).dump().size() : 0) + 2;
      constexpr std::size_t envelope_reserve = 256;
      if (record_bytes + envelope_reserve > scene_audit_page_max_bytes) {
        throw std::runtime_error("one scene exceeds the 2 MiB audit page storage limit");
      }
      if (pending_record_bytes_ + record_bytes + envelope_reserve > scene_audit_page_max_bytes) {
        flush();
      }
      if (pages_.size() >= scene_audit_max_pages) {
        throw std::runtime_error("scene audit exceeds its bounded page index");
      }
      pending_.scenes.push_back(scene);
      if (scene.boundary.semantic_cut) pending_.boundary_revisions.push_back(scene.boundary);
      pending_record_bytes_ += record_bytes;
      ++scene_count_;
      covered_end_ = scene.end_sequence_exclusive;
      ended_ = scene.boundary.decision == boundary_decision_e::end_of_stream;
      recent_.push_back(scene);
      if (recent_.size() > scene_audit_recent_scenes) recent_.erase(recent_.begin());
      if (pending_.scenes.size() == scene_audit_page_max_scenes) flush();
    }

    void flush() {
      if (pending_.scenes.empty()) return;
      const auto bytes = serialize(pending_);
      if (bytes.size() > scene_audit_page_max_bytes || bytes.size() > scene_audit_storage_max_bytes - total_bytes_) {
        throw std::runtime_error("scene audit exceeds its 1 GiB storage limit");
      }
      wire::scene_audit_page_descriptor_t descriptor {
        .index = pending_.index,
        .first_scene_id = pending_.scenes.front().scene_id,
        .scene_count = pending_.scenes.size(),
        .boundary_count = pending_.boundary_revisions.size(),
        .start_sequence = pending_.scenes.front().start_sequence,
        .end_sequence_exclusive = pending_.scenes.back().end_sequence_exclusive,
        .bytes = bytes.size(),
      };
      wire::validate_scene_audit_page(pending_, descriptor,
        pending_.scenes.back().boundary.decision == boundary_decision_e::end_of_stream);
      descriptor.sha256 = publish_(wire::scene_audit_page_filename(descriptor.index), bytes);
      if (!wire::valid_sha256_hex(descriptor.sha256)) throw std::runtime_error("audit publisher did not attest a SHA-256 digest");
      total_bytes_ += bytes.size();
      boundary_count_ += descriptor.boundary_count;
      pages_.push_back(std::move(descriptor));
      pending_ = {};
      pending_record_bytes_ = 0;
      pending_.index = static_cast<std::uint32_t>(pages_.size());
    }

    // Checkpoints flush at most one small page. The manifest has a bounded index;
    // previously published scene records are never read or rewritten.
    [[nodiscard]] nlohmann::json manifest(wire::scene_audit_contract_t summary) {
      if ((summary.status == "complete") != ended_) throw std::runtime_error("audit completion disagrees with its final scene");
      flush();
      summary.scenes.clear();
      summary.boundary_revisions.clear();
      summary.paged = true;
      summary.scene_count = scene_count_;
      summary.boundary_count = boundary_count_;
      summary.covered_end_sequence = covered_end_;
      summary.total_page_bytes = total_bytes_;
      summary.pages = pages_;
      auto result = wire::to_json(summary);
      if (result.dump(2).size() + 1 > scene_audit_max_bytes) throw std::runtime_error("audit manifest exceeds its serialized bound");
      (void) wire::parse_scene_audit_contract(result);
      return result;
    }

    [[nodiscard]] std::uint64_t scene_count() const noexcept { return scene_count_; }
    [[nodiscard]] const std::vector<scene_plan_t> &recent_scenes() const noexcept { return recent_; }
    [[nodiscard]] std::size_t pending_scene_count() const noexcept { return pending_.scenes.size(); }
    [[nodiscard]] std::size_t page_count() const noexcept { return pages_.size(); }

  private:
    static std::string serialize(const wire::scene_audit_page_t &page) { return wire::to_json(page).dump() + "\n"; }
    publisher_t publish_;
    wire::scene_audit_page_t pending_;
    std::size_t pending_record_bytes_ = 0;
    std::vector<wire::scene_audit_page_descriptor_t> pages_;
    std::vector<scene_plan_t> recent_;
    std::uint64_t scene_count_ = 0;
    std::uint64_t covered_end_ = 1;
    std::uint64_t boundary_count_ = 0;
    std::uint64_t total_bytes_ = 0;
    bool ended_ = false;
  };
}  // namespace offline_sbs
