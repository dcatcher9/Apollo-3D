#pragma once

#ifdef SUNSHINE_TESTS
  #include "src/platform/windows/ar_glasses.h"

  #include <cstdint>
  #include <filesystem>
  #include <functional>
  #include <memory>
  #include <optional>
  #include <string>
  #include <string_view>
  #include <vector>

namespace ar_glasses::testing {
  struct discovery_hooks_t {
    std::function<bool(std::vector<DISPLAYCONFIG_PATH_INFO> &, std::vector<DISPLAYCONFIG_MODE_INFO> &)> query_active;
    std::function<std::vector<std::wstring>(std::wstring)> match_display;
    std::function<LONG(DISPLAYCONFIG_DEVICE_INFO_HEADER *)> get_info;
    std::function<BOOL(LPCWSTR, DWORD, PDISPLAY_DEVICEW, DWORD)> enum_devices;
    std::function<bool(const std::string &)> notify;
  };

  struct discovery_hdr_t {
    bool known = false;
    bool supported = false;
    bool user_enabled = false;
    bool active = false;
    bool limited_by_policy = false;
    UINT32 bits_per_color = 0;
  };

  /** Every field of the actual private target, copied without retaining native or global state. */
  struct discovery_target_t {
    std::string device_id;
    std::wstring device_path;
    std::string friendly_name;
    std::string gdi_name;
    std::string desktop_topology;
    LUID source_adapter_id {};
    UINT32 source_id = 0;
    LUID adapter_id {};
    UINT32 target_id = 0;
    discovery_hdr_t hdr;
    RECT rect {};
    int refresh_millihz = 60000;
    presentation_mode_e mode = presentation_mode_e::unsupported;
    bool is_primary = false;
    bool is_cloned = false;
  };

  struct discovery_result_t {
    bool query_succeeded = false;
    bool approved_prefix_preserved = true;
    std::optional<discovery_target_t> selected;
    std::vector<discovery_target_t> approved;
  };

  struct discovery_costs_t {
    std::uint64_t hdr_queries = 0;
    std::uint64_t fingerprints = 0;
    std::uint64_t persistence_attempts = 0;
  };

  /** Runs only real synchronous discovery/decision code against complete finite native inputs.
   * The caller owns the absolute temporary configuration file. No controller is constructed.
   */
  class discovery_session_t {
  public:
    discovery_session_t(const std::filesystem::path &config_path, discovery_hooks_t hooks);
    ~discovery_session_t();
    discovery_session_t(const discovery_session_t &) = delete;
    discovery_session_t &operator=(const discovery_session_t &) = delete;
    discovery_session_t(discovery_session_t &&) = delete;
    discovery_session_t &operator=(discovery_session_t &&) = delete;

    void reload_decisions();
    discovery_result_t find(std::wstring_view required = {}, std::wstring_view preferred = {}, bool collect_approved = true);
    [[nodiscard]] discovery_costs_t costs() const;

  private:
    class impl_t;
    std::unique_ptr<impl_t> impl_;
  };

  /** A separate joined observer avoids same-thread try_lock on the nonrecursive policy mutex. */
  bool discovery_mutex_available_for_test();
}  // namespace ar_glasses::testing
#endif
