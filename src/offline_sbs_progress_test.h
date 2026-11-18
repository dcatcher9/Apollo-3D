/** Private access to the actual offline worker's progress publisher. */
#pragma once

#ifdef SUNSHINE_TESTS
  #include "offline_sbs_worker.h"

  #include <chrono>
  #include <cstddef>
  #include <cstdint>
  #include <filesystem>
  #include <memory>
  #include <string>
  #include <string_view>

namespace offline_sbs {
  // This facade calls the actual worker-owned publisher with a deterministic clock and writes
  // real atomic files. It does not reproduce the publication policy.
  class worker_progress_test_publisher_t {
  public:
    worker_progress_test_publisher_t(std::filesystem::path progress_path, std::string job_id);
    ~worker_progress_test_publisher_t();

    worker_progress_test_publisher_t(const worker_progress_test_publisher_t &) = delete;
    worker_progress_test_publisher_t &operator=(const worker_progress_test_publisher_t &) = delete;

    void publish(
      std::string_view phase,
      std::uint64_t processed_frames,
      const media_contract_t *media,
      std::chrono::steady_clock::time_point now
    );
    std::size_t successful_publications() const;

  private:
    struct impl_t;
    std::unique_ptr<impl_t> impl_;
  };
}  // namespace offline_sbs
#endif
