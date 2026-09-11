/**
 * @file src/host_sbs_conversion_work.h
 * @brief Live Host SBS work demand, independent of retained geometry ownership.
 */
#pragma once

namespace models {

  enum class engine_build_status {
    unknown,
    building,
    ready,
    failed,
  };

  /** Snapshot of the converter's existing owners; this stores no parallel lifecycle state.
   *
   * An opaque adaptive watermark restricts reuse after its transaction completes, but is not
   * unfinished work. Per-device construction has its own completion event. Global preparation
   * is sampled at the adapter's ordinary idle boundary and requests conversion only once it can
   * start construction or report failure, rather than reconverting flat frames while it builds.
   */
  struct host_sbs_conversion_work_t {
    bool pipeline_enabled = false;
    bool estimator_present = false;
    bool pipeline_build_pending = false;
    bool depth_completion_pending = false;
    bool authority_reprocess_pending = false;
    bool dump_pending = false;

    template<class ReadModelStatus>
    [[nodiscard]] bool needs_service(ReadModelStatus read_model_status) const {
      if (depth_completion_pending || authority_reprocess_pending || dump_pending) {
        return true;
      }
      if (!pipeline_enabled || estimator_present || pipeline_build_pending) {
        return false;
      }
      const auto status = read_model_status();
      return status == engine_build_status::ready || status == engine_build_status::failed;
    }
  };

}  // namespace models
