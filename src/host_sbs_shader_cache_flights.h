/** Internal shared bytecode work; no source authentication or D3D device operations. */
#pragma once

#include "host_sbs_shader_cache.h"

#include <functional>
#include <map>
#include <mutex>
#include <tuple>

namespace models::host_sbs_shader_cache::detail {
  class bytecode_cache_t {
    struct entry_t;

  public:
    using key_t = std::tuple<std::filesystem::path, std::string, std::string, std::string, std::string>;

    // The factory owns all producer setup after publication of the in-flight entry. Attached
    // callers share its value or original exception; a failed/null entry permits a later retry.
    bytecode_t get(
      const key_t &key,
      const std::function<bytecode_t()> &resolve,
      const std::function<void()> &on_hit = {}
    );

    // Detach cached entries without cancelling work or the callers already attached to it.
    void clear();

  private:
    void erase_if_current(const key_t &key, const std::shared_ptr<entry_t> &entry);

    std::mutex mutex_;
    std::map<key_t, std::shared_ptr<entry_t>> entries_;
  };
}  // namespace models::host_sbs_shader_cache::detail
