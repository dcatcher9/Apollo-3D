/**
 * @file tools/video_dom_probe/main.cpp
 * @brief Out-of-process Chromium IAccessible2 tag:video geometry helper.
 */

#define WIN32_LEAN_AND_MEAN

#include "tools/video_dom_probe/ia2_minimal.h"
#include "tools/video_dom_probe/probe_logic.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <dwmapi.h>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <oleacc.h>
#include <optional>
#include <servprov.h>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <windows.h>

namespace {
  using namespace std::chrono_literals;
  namespace probe = video_dom_probe;

  constexpr std::size_t maximum_nodes = 20000;
  constexpr std::size_t maximum_depth = 128;
  constexpr LONG maximum_children_per_node = 10000;
  constexpr auto maximum_scan_work_time = 15s;
  constexpr auto scan_service_interval = 100ms;
  constexpr auto machine_heartbeat_interval = 1s;
  constexpr auto periodic_full_scan_interval = 15s;
  constexpr auto event_full_scan_min_interval = 3s;
  // How long a selection dropped for a foreground or geometry change is kept to be staged again.
  constexpr auto remembered_selection_lifetime = 10min;
  constexpr std::size_t maximum_remembered_selections = 4;
  constexpr unsigned maximum_cold_warmup_scans = 3;
  constexpr LONG coordinate_rounding_tolerance = 1;

  std::atomic_bool keep_running {true};
  std::atomic_uint64_t foreground_change_generation {1};
  std::atomic_uint64_t object_change_generation {1};
  std::atomic_uintptr_t monitored_window_value {};

  void pump_messages();

  template<typename T>
  class com_ptr_t {
  public:
    com_ptr_t() = default;

    explicit com_ptr_t(T *value):
        value_(value) {}

    ~com_ptr_t() {
      reset();
    }

    com_ptr_t(const com_ptr_t &) = delete;
    com_ptr_t &operator=(const com_ptr_t &) = delete;

    com_ptr_t(com_ptr_t &&other) noexcept:
        value_(std::exchange(other.value_, nullptr)) {}

    com_ptr_t &operator=(com_ptr_t &&other) noexcept {
      if (this != &other) {
        reset(std::exchange(other.value_, nullptr));
      }
      return *this;
    }

    [[nodiscard]] T *get() const noexcept {
      return value_;
    }

    [[nodiscard]] T *operator->() const noexcept {
      return value_;
    }

    [[nodiscard]] explicit operator bool() const noexcept {
      return value_ != nullptr;
    }

    void reset(T *value = nullptr) noexcept {
      if (value_) {
        value_->Release();
      }
      value_ = value;
    }

  private:
    T *value_ {};
  };

  template<typename T>
  [[nodiscard]] com_ptr_t<T> retain_com(T *value) noexcept {
    if (value) {
      value->AddRef();
    }
    return com_ptr_t<T> {value};
  }

  class bstr_t {
  public:
    bstr_t() = default;

    ~bstr_t() {
      if (value_) {
        SysFreeString(value_);
      }
    }

    bstr_t(const bstr_t &) = delete;
    bstr_t &operator=(const bstr_t &) = delete;

    [[nodiscard]] BSTR *put() noexcept {
      return &value_;
    }

    [[nodiscard]] BSTR get() const noexcept {
      return value_;
    }

  private:
    BSTR value_ {};
  };

  struct observed_candidate_t {
    probe::video_candidate_t geometry;
    DWORD state {};
    LONG document_unique_id {};
    probe::rect_t document_rect {};
    com_ptr_t<IAccessible> accessible;
    com_ptr_t<IAccessible> document_accessible;
  };

  enum class scan_status_e {
    ok,
    no_foreground_window,
    unsupported_foreground,
    unavailable_window,
    accessibility_unavailable,
    warming_up,
    traversal_incomplete,
    foreground_changed,
  };

  struct scan_result_t {
    scan_status_e status {scan_status_e::no_foreground_window};
    HWND window {};
    DWORD process_id {};
    std::wstring process_name;
    probe::rect_t client_rect {};
    bool client_fills_monitor {};
    std::size_t visited_nodes {};
    std::size_t ia2_nodes {};
    std::size_t document_nodes {};
    std::size_t tagged_nodes {};
    std::size_t errors {};
    // For --trace: the source line of each failed step, and why the walk stopped early.
    std::map<int, std::size_t> error_lines;
    std::vector<std::string> error_notes;
    std::size_t vanished_nodes {};
    const char *stopped = "";
    bool extended_properties_ready {};
    std::chrono::microseconds elapsed {};
    std::vector<observed_candidate_t> candidates;
    probe::selection_t selection;
  };

  void note_error(scan_result_t &result, int line) {
    ++result.errors;
    ++result.error_lines[line];
  }

  [[nodiscard]] const char *status_name(scan_status_e status) noexcept {
    switch (status) {
      case scan_status_e::ok:
        return "ok";
      case scan_status_e::no_foreground_window:
        return "no-foreground-window";
      case scan_status_e::unsupported_foreground:
        return "unsupported-foreground";
      case scan_status_e::unavailable_window:
        return "unavailable-window";
      case scan_status_e::accessibility_unavailable:
        return "accessibility-unavailable";
      case scan_status_e::warming_up:
        return "warming-up";
      case scan_status_e::traversal_incomplete:
        return "traversal-incomplete";
      case scan_status_e::foreground_changed:
        return "foreground-changed";
    }
    return "unknown";
  }

  [[nodiscard]] std::string utf8(std::wstring_view input) {
    if (input.empty()) {
      return {};
    }
    const int required = WideCharToMultiByte(
      CP_UTF8,
      WC_ERR_INVALID_CHARS,
      input.data(),
      static_cast<int>(input.size()),
      nullptr,
      0,
      nullptr,
      nullptr
    );
    if (required <= 0) {
      return "<invalid-unicode>";
    }
    std::string output(static_cast<std::size_t>(required), '\0');
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, input.data(), static_cast<int>(input.size()), output.data(), required, nullptr, nullptr) != required) {
      return "<invalid-unicode>";
    }
    return output;
  }

  [[nodiscard]] std::wstring lower_ascii(std::wstring value) {
    std::transform(value.begin(), value.end(), value.begin(), [](wchar_t character) {
      if (character >= L'A' && character <= L'Z') {
        return static_cast<wchar_t>(character - L'A' + L'a');
      }
      return character;
    });
    return value;
  }

  [[nodiscard]] std::wstring process_name_for_window(HWND window, DWORD &process_id) {
    process_id = 0;
    GetWindowThreadProcessId(window, &process_id);
    if (!process_id) {
      return {};
    }

    const HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, process_id);
    if (!process) {
      return {};
    }
    std::array<wchar_t, 32768> path {};
    DWORD path_length = static_cast<DWORD>(path.size());
    const BOOL success = QueryFullProcessImageNameW(process, 0, path.data(), &path_length);
    CloseHandle(process);
    if (!success || path_length == 0 || path_length >= path.size()) {
      return {};
    }
    return std::filesystem::path(std::wstring(path.data(), path_length)).filename().wstring();
  }

  [[nodiscard]] bool supported_browser(std::wstring_view process_name) {
    const auto lowered = lower_ascii(std::wstring(process_name));
    return lowered == L"chrome.exe" || lowered == L"msedge.exe";
  }

  [[nodiscard]] std::optional<probe::rect_t> foreground_client_rect(HWND window) {
    RECT client {};
    if (!GetClientRect(window, &client)) {
      return std::nullopt;
    }
    std::array<POINT, 2> points {
      POINT {client.left, client.top},
      POINT {client.right, client.bottom},
    };
    SetLastError(ERROR_SUCCESS);
    if (!MapWindowPoints(window, nullptr, points.data(), 2)) {
      const DWORD error = GetLastError();
      if (error != ERROR_SUCCESS) {
        return std::nullopt;
      }
    }
    probe::rect_t result {points[0].x, points[0].y, points[1].x, points[1].y};
    if (!probe::rect_valid(result)) {
      return std::nullopt;
    }
    return result;
  }

  [[nodiscard]] bool window_available(HWND window) {
    if (!IsWindow(window) || !IsWindowVisible(window) || IsIconic(window)) {
      return false;
    }
    DWORD cloaked = 0;
    const HRESULT cloak_status = DwmGetWindowAttribute(
      window,
      DWMWA_CLOAKED,
      &cloaked,
      sizeof(cloaked)
    );
    return SUCCEEDED(cloak_status) && cloaked == 0;
  }

  [[nodiscard]] bool client_fills_monitor(HWND window, const probe::rect_t &client) {
    const auto monitor = MonitorFromWindow(window, MONITOR_DEFAULTTONULL);
    MONITORINFO info {};
    info.cbSize = sizeof(info);
    return monitor && GetMonitorInfoW(monitor, &info) && probe::client_fills_monitor(client, {info.rcMonitor.left, info.rcMonitor.top, info.rcMonitor.right, info.rcMonitor.bottom});
  }

  /**
   * Whether `window` is still the available root foreground window of `process_id`, with the
   * client rectangle it was observed with. A census and a cached-selection refresh both keep
   * their authority only while this holds.
   */
  [[nodiscard]] bool foreground_unchanged(HWND window, DWORD process_id, const probe::rect_t &client_rect) {
    const HWND foreground = GetForegroundWindow();
    if (!foreground || GetAncestor(foreground, GA_ROOT) != window || !window_available(window)) {
      return false;
    }
    DWORD current_process_id = 0;
    GetWindowThreadProcessId(window, &current_process_id);
    const auto client = foreground_client_rect(window);
    return current_process_id == process_id && client && *client == client_rect;
  }

  template<typename T>
  struct interface_query_t {
    HRESULT status {E_POINTER};
    com_ptr_t<T> object;
  };

  [[nodiscard]] interface_query_t<IAccessible> accessible_from_dispatch(IDispatch *dispatch) {
    if (!dispatch) {
      return {.status = E_POINTER};
    }
    IAccessible *accessible = nullptr;
    const HRESULT status = dispatch->QueryInterface(
      IID_IAccessible,
      reinterpret_cast<void **>(&accessible)
    );
    return {.status = status, .object = com_ptr_t<IAccessible> {accessible}};
  }

  [[nodiscard]] interface_query_t<probe::ia2::IAccessible2> query_ia2(
    IAccessible *accessible
  ) {
    if (!accessible) {
      return {.status = E_POINTER};
    }
    IServiceProvider *provider_raw = nullptr;
    const HRESULT provider_status = accessible->QueryInterface(
      IID_IServiceProvider,
      reinterpret_cast<void **>(&provider_raw)
    );
    if (FAILED(provider_status)) {
      return {.status = provider_status};
    }
    com_ptr_t<IServiceProvider> provider {provider_raw};

    probe::ia2::IAccessible2 *ia2_accessible = nullptr;
    const HRESULT ia2_status = provider->QueryService(
      IID_IAccessible,
      probe::ia2::IID_IAccessible2,
      reinterpret_cast<void **>(&ia2_accessible)
    );
    return {
      .status = ia2_status,
      .object = com_ptr_t<probe::ia2::IAccessible2> {ia2_accessible},
    };
  }

  [[nodiscard]] std::optional<DWORD> accessible_state(IAccessible *accessible) {
    VARIANT self;
    VariantInit(&self);
    self.vt = VT_I4;
    self.lVal = CHILDID_SELF;
    VARIANT state;
    VariantInit(&state);
    const HRESULT status = accessible->get_accState(self, &state);
    std::optional<DWORD> result;
    if (SUCCEEDED(status)) {
      if (state.vt == VT_I4) {
        result = static_cast<DWORD>(state.lVal);
      } else if (state.vt == VT_UI4) {
        result = state.ulVal;
      }
    }
    VariantClear(&state);
    return result;
  }

  [[nodiscard]] std::optional<probe::rect_t> accessible_rect(IAccessible *accessible) {
    VARIANT self;
    VariantInit(&self);
    self.vt = VT_I4;
    self.lVal = CHILDID_SELF;
    LONG left = 0;
    LONG top = 0;
    LONG width = 0;
    LONG height = 0;
    if (accessible->accLocation(&left, &top, &width, &height, self) != S_OK || width <= 0 || height <= 0) {
      return std::nullopt;
    }
    const auto right = static_cast<std::int64_t>(left) + width;
    const auto bottom = static_cast<std::int64_t>(top) + height;
    if (right > std::numeric_limits<LONG>::max() || bottom > std::numeric_limits<LONG>::max()) {
      return std::nullopt;
    }
    probe::rect_t result {
      left,
      top,
      static_cast<LONG>(right),
      static_cast<LONG>(bottom),
    };
    if (!probe::rect_valid(result)) {
      return std::nullopt;
    }
    return result;
  }

  [[nodiscard]] interface_query_t<IUnknown> query_com_identity(IAccessible *accessible) {
    IUnknown *identity = nullptr;
    const HRESULT status = accessible->QueryInterface(
      IID_IUnknown,
      reinterpret_cast<void **>(&identity)
    );
    return {.status = status, .object = com_ptr_t<IUnknown> {identity}};
  }

  struct document_context_t {
    LONG unique_id {};
    probe::rect_t rect {};
    bool state_available {};
    IAccessible *accessible {};
  };

  // Set by --trace: nodes are described so that a failed step can name its node's parent.
  bool trace_nodes = false;

  [[nodiscard]] std::string narrow(std::wstring_view wide, std::size_t limit) {
    wide = wide.substr(0, limit);
    std::string out(wide.size() * 3u, '\0');
    const int written = WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), out.data(), static_cast<int>(out.size()), nullptr, nullptr);
    out.resize(written > 0 ? static_cast<std::size_t>(written) : 0u);
    return out;
  }

  [[nodiscard]] std::shared_ptr<const std::string> describe_node(IAccessible *accessible, std::wstring_view attributes) {
    std::ostringstream out;
    VARIANT self;
    VariantInit(&self);
    self.vt = VT_I4;
    self.lVal = CHILDID_SELF;
    VARIANT role;
    VariantInit(&role);
    const HRESULT role_status = accessible->get_accRole(self, &role);
    out << "role=" << (role_status == S_OK && role.vt == VT_I4 ? std::to_string(role.lVal) : "?");
    VariantClear(&role);
    BSTR name = nullptr;
    if (accessible->get_accName(self, &name) == S_OK && name) {
      out << " name='" << narrow(std::wstring_view(name, SysStringLen(name)), 40u) << "'";
    }
    SysFreeString(name);
    out << " attrs='" << narrow(attributes, 160u) << "'";
    return std::make_shared<const std::string>(out.str());
  }

  struct observed_candidate_t;
  // A video element as the helper records it: where it is and whether it can be selected.
  [[nodiscard]] observed_candidate_t observe_candidate(
    IAccessible *accessible,
    LONG unique_id,
    DWORD state,
    const probe::rect_t &element_rect,
    const std::optional<document_context_t> &document,
    const probe::rect_t &client_rect
  );

  // A node that has children, as the walk met it: what its children need of it, and what the walk
  // needs to read its children again.
  struct lineage_t {
    com_ptr_t<IAccessible> accessible;
    std::shared_ptr<lineage_t> parent;
    std::size_t depth {};
    std::optional<document_context_t> children_document;  // The document context of its children.
    std::shared_ptr<const std::string> described;  // For --trace.
    unsigned rewalks {};  // How often its children were read again.
    unsigned reading {};  // Counts the readings of its children; a child knows which listed it.
    bool reread_pending {};  // Queued to have its children read again.
    bool gone {};  // Taken as having left the tree; its queued descendants are dropped.
  };

  struct queued_node_t {
    com_ptr_t<IAccessible> accessible;
    std::size_t depth {};
    std::optional<document_context_t> document;
    std::shared_ptr<lineage_t> parent;
    LONG index_in_parent = -1;
    unsigned parent_reading {};  // The reading of its parent's children that listed it.
    // Set for a node queued again after a child of it had left the tree: only its children are
    // read, and they are visited before anything else.
    std::shared_ptr<lineage_t> read_again;
  };

  template<typename Service>
  scan_result_t scan_foreground(Service service) {
    const auto started = std::chrono::steady_clock::now();
    scan_result_t result;

    HWND foreground = GetForegroundWindow();
    if (!foreground) {
      result.elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started
      );
      return result;
    }
    foreground = GetAncestor(foreground, GA_ROOT);
    result.window = foreground;
    result.process_name = process_name_for_window(foreground, result.process_id);
    if (!supported_browser(result.process_name)) {
      result.status = scan_status_e::unsupported_foreground;
      result.elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started
      );
      return result;
    }
    if (!window_available(foreground)) {
      result.status = scan_status_e::unavailable_window;
      result.elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started
      );
      return result;
    }
    const auto client_rect = foreground_client_rect(foreground);
    if (!client_rect) {
      result.status = scan_status_e::unavailable_window;
      result.elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started
      );
      return result;
    }
    result.client_rect = *client_rect;
    result.client_fills_monitor = client_fills_monitor(foreground, *client_rect);
    // The first census can pump WinEvents before it finishes. Route its descendant events
    // now, rather than losing them against the previous foreground window or a null HWND.
    monitored_window_value.store(reinterpret_cast<std::uintptr_t>(foreground), std::memory_order_relaxed);
    const auto initial_foreground_generation = foreground_change_generation.load(std::memory_order_relaxed);

    IAccessible *root_raw = nullptr;
    const HRESULT root_status = AccessibleObjectFromWindow(
      foreground,
      OBJID_CLIENT,
      IID_IAccessible,
      reinterpret_cast<void **>(&root_raw)
    );
    if (root_status != S_OK || !root_raw) {
      result.status = scan_status_e::accessibility_unavailable;
      result.elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started
      );
      return result;
    }

    bool complete = true;
    bool aborted = false;
    bool foreground_changed = false;
    auto last_service = started;
    const auto foreground_current = [&]() {
      pump_messages();
      if (foreground_change_generation.load(std::memory_order_relaxed) != initial_foreground_generation || !foreground_unchanged(foreground, result.process_id, result.client_rect) || client_fills_monitor(foreground, result.client_rect) != result.client_fills_monitor) {
        foreground_changed = true;
        aborted = true;
        return false;
      }
      return true;
    };
    const auto checkpoint = [&]() {
      const auto now = std::chrono::steady_clock::now();
      const auto action = probe::census_checkpoint_policy(
        now - started, now - last_service, result.visited_nodes, maximum_nodes,
        maximum_scan_work_time, scan_service_interval
      );
      if (!keep_running.load(std::memory_order_relaxed) || action == probe::census_checkpoint_e::exhausted) {
        result.stopped = "time-or-node-limit";
        complete = false;
        aborted = true;
        return false;
      }
      if (action != probe::census_checkpoint_e::service) {
        return true;
      }
      // This runs on the scanning thread: a blocked cross-process COM call also blocks
      // heartbeats, so the host's existing watchdog can still terminate a hung helper.
      if (!foreground_current() || !service(foreground_current)) {
        result.stopped = "foreground-or-service";
        complete = false;
        aborted = true;
        return false;
      }
      last_service = std::chrono::steady_clock::now();
      return true;
    };
    std::deque<queued_node_t> queue;
    queue.push_back({com_ptr_t<IAccessible> {root_raw}, 0, std::nullopt});
    std::unordered_set<std::uintptr_t> visited;
    visited.reserve(4096);
    std::vector<com_ptr_t<IUnknown>> retained_identities;
    retained_identities.reserve(4096);
    std::vector<com_ptr_t<IAccessible>> retained_documents;
    retained_documents.reserve(16);
    unsigned census_rewalks = 0;

    while (!queue.empty() && !aborted) {
      if (!checkpoint()) {
        break;
      }
      auto node = std::move(queue.front());
      queue.pop_front();
      if (!node.accessible) {
        note_error(result, __LINE__);
        complete = false;
        continue;
      }

      // What this visit has recorded so far, so that a node found to have left the tree is
      // taken out of the census again.
      std::optional<std::uintptr_t> identity;
      bool counted_ia2 = false;
      bool counted_document = false;
      bool counted_tagged = false;
      bool added_candidate = false;
      // The lineage this node's children refer to; made when its children are read.
      std::shared_ptr<lineage_t> self = node.read_again;
      if (self) {
        self->reread_pending = false;
      }
      auto current_document = node.read_again ? node.read_again->children_document : node.document;
      std::shared_ptr<const std::string> described = node.read_again ? node.read_again->described : nullptr;

      // A page replaces nodes while the walk is on its way to them, and such a node answers no
      // query any more. A failed step is therefore first taken as the node having left the tree:
      // everything recorded of it is withdrawn, and the children of its parent are read again and
      // visited at once, which finds the node again if it is still there. Only when the bounds on
      // that are used up does the step count as the error that leaves the census incomplete.
      const auto left_the_tree = [&](int line, HRESULT status) {
        if (self) {
          self->gone = true;
        }
        const auto up = node.read_again ? node.read_again->parent : node.parent;
        bool under_gone_ancestor = false;
        for (auto ancestor = up; ancestor && !under_gone_ancestor; ancestor = ancestor->parent) {
          under_gone_ancestor = ancestor->gone;
        }
        const auto decision = probe::vanished_node_policy({
          .has_parent = up != nullptr,
          .under_gone_ancestor = under_gone_ancestor,
          .older_reading = up && !node.read_again && node.parent_reading != up->reading,
          .reread_pending = up && up->reread_pending,
          .parent_rewalks = up ? up->rewalks : 0u,
          .census_rewalks = census_rewalks,
        });
        if (decision == probe::vanished_node_e::read_parent_again) {
          ++up->rewalks;
          ++census_rewalks;
          up->reread_pending = true;
          queue.push_front({retain_com(up->accessible.get()), up->depth, up->children_document, up->parent, -1, 0u, up});
        }
        if (decision == probe::vanished_node_e::error) {
          if (trace_nodes) {
            std::ostringstream out;
            out << "line " << line << " hr=0x" << std::hex << static_cast<unsigned long>(status) << std::dec << " depth=" << node.depth
                << " child#" << node.index_in_parent << " of {" << (up && up->described ? *up->described : std::string("?")) << "}";
            result.error_notes.push_back(out.str());
          }
          note_error(result, line);
          complete = false;
          return false;
        }
        ++result.vanished_nodes;
        if (identity) {
          visited.erase(*identity);
          --result.visited_nodes;
        }
        result.ia2_nodes -= counted_ia2 ? 1u : 0u;
        result.tagged_nodes -= counted_tagged ? 1u : 0u;
        if (counted_document) {
          // The retained object stays: children already queued under it name it as their document.
          --result.document_nodes;
        }
        if (added_candidate) {
          result.candidates.pop_back();
        }
        return true;
      };

      if (!node.read_again) {
        auto identity_query = query_com_identity(node.accessible.get());
        if (identity_query.status != S_OK || !identity_query.object) {
          (void) left_the_tree(__LINE__, identity_query.status);
          continue;
        }
        const auto identity_value = reinterpret_cast<std::uintptr_t>(identity_query.object.get());
        if (!visited.insert(identity_value).second) {
          continue;
        }
        identity = identity_value;
        retained_identities.push_back(std::move(identity_query.object));
        ++result.visited_nodes;

        auto ia2_query = query_ia2(node.accessible.get());
        if (ia2_query.status == S_OK && ia2_query.object) {
          auto &ia2_accessible = ia2_query.object;
          ++result.ia2_nodes;
          counted_ia2 = true;

          LONG unique_id_before = 0;
          LONG role = 0;
          if (ia2_accessible->get_uniqueID(&unique_id_before) != S_OK || unique_id_before == 0 || ia2_accessible->role(&role) != S_OK) {
            (void) left_the_tree(__LINE__, E_FAIL);
            continue;
          }
          if (role == ROLE_SYSTEM_DOCUMENT) {
            const auto state = accessible_state(node.accessible.get());
            const auto document_rect = accessible_rect(node.accessible.get());
            LONG unique_id_after = 0;
            if (!state || !document_rect || ia2_accessible->get_uniqueID(&unique_id_after) != S_OK || unique_id_after == 0 || unique_id_after != unique_id_before) {
              (void) left_the_tree(__LINE__, E_FAIL);
              continue;
            }
            constexpr DWORD unavailable_states =
              STATE_SYSTEM_UNAVAILABLE | STATE_SYSTEM_INVISIBLE | STATE_SYSTEM_OFFSCREEN;
            const bool state_available = (*state & unavailable_states) == 0;
            retained_documents.push_back(retain_com(node.accessible.get()));
            current_document = document_context_t {
              .unique_id = unique_id_before,
              .rect = *document_rect,
              .state_available = state_available,
              .accessible = retained_documents.back().get(),
            };
            ++result.document_nodes;
            counted_document = true;
          }

          bstr_t attributes;
          const HRESULT attributes_status = ia2_accessible->get_attributes(attributes.put());
          if (FAILED(attributes_status) || (attributes_status == S_OK && !attributes.get())) {
            (void) left_the_tree(__LINE__, attributes_status);
            continue;
          }
          if (trace_nodes) {
            described = describe_node(node.accessible.get(), attributes.get() ? std::wstring_view(attributes.get(), SysStringLen(attributes.get())) : std::wstring_view {});
          }
          if (attributes_status == S_OK) {
            const auto length = static_cast<std::size_t>(SysStringLen(attributes.get()));
            const auto match = probe::match_video_tag(
              std::wstring_view(attributes.get(), length)
            );
            if (!match.valid) {
              (void) left_the_tree(__LINE__, E_FAIL);
              continue;
            }
            if (match.has_tag) {
              ++result.tagged_nodes;
              counted_tagged = true;
            }
            if (match.is_video) {
              const auto state = accessible_state(node.accessible.get());
              const auto element_rect = accessible_rect(node.accessible.get());
              LONG unique_id_after = 0;
              if (!state || !element_rect || ia2_accessible->get_uniqueID(&unique_id_after) != S_OK || unique_id_after == 0 || unique_id_after != unique_id_before) {
                (void) left_the_tree(__LINE__, E_FAIL);
                continue;
              }
              result.candidates.push_back(observe_candidate(
                node.accessible.get(), unique_id_before, *state, *element_rect, current_document, result.client_rect
              ));
              added_candidate = true;
            }
          }
        } else if (ia2_query.status != E_NOINTERFACE) {
          (void) left_the_tree(__LINE__, ia2_query.status);
          continue;
        }
      }

      if (node.depth >= maximum_depth) {
        LONG child_count = 0;
        if (node.accessible->get_accChildCount(&child_count) == S_OK && child_count > 0) {
          complete = false;
        }
        continue;
      }

      LONG child_count = 0;
      const HRESULT child_count_status = node.accessible->get_accChildCount(&child_count);
      if (child_count_status != S_OK || child_count < 0) {
        (void) left_the_tree(__LINE__, child_count_status);
        continue;
      }
      if (child_count > maximum_children_per_node) {
        note_error(result, __LINE__);
        complete = false;
        continue;
      }
      if (child_count == 0) {
        continue;
      }
      if (!self) {
        self = std::make_shared<lineage_t>();
        self->accessible = retain_com(node.accessible.get());
        self->parent = node.parent;
        self->depth = node.depth;
      }
      self->children_document = current_document;
      self->described = described;
      // Read again and answering: it is in the tree.
      self->gone = false;
      ++self->reading;
      // The children of a node read again are visited before anything else, while they are fresh.
      const bool at_front = node.read_again != nullptr;
      const auto queue_child = [&](IDispatch *dispatch, LONG index) {
        auto query = accessible_from_dispatch(dispatch);
        if (query.status != S_OK || !query.object) {
          return false;
        }
        queued_node_t child {std::move(query.object), node.depth + 1, current_document, self, index, self->reading};
        if (at_front) {
          queue.push_front(std::move(child));
        } else {
          queue.push_back(std::move(child));
        }
        return true;
      };
      // A child list that changes while it is read is read again as a whole, by the same rule.
      bool children_read = true;
      for (LONG first = 0; first < child_count && children_read;) {
        if (!checkpoint()) {
          break;
        }
        if (queue.size() + result.visited_nodes + 64u >= maximum_nodes) {
          note_error(result, __LINE__);
          complete = false;
          break;
        }
        std::array<VARIANT, 64> children;
        for (auto &child : children) {
          VariantInit(&child);
        }
        const LONG requested = std::min<LONG>(
          static_cast<LONG>(children.size()),
          child_count - first
        );
        LONG obtained = 0;
        const HRESULT children_status = AccessibleChildren(
          node.accessible.get(),
          first,
          requested,
          children.data(),
          &obtained
        );
        children_read = children_status == S_OK && obtained > 0 && obtained <= requested;
        for (LONG index = 0; index < obtained && children_read; ++index) {
          auto &child = children[static_cast<std::size_t>(index)];
          if (child.vt == VT_DISPATCH && child.pdispVal) {
            children_read = queue_child(child.pdispVal, first + index);
          } else if (child.vt == VT_I4) {
            IDispatch *dispatch = nullptr;
            const HRESULT child_status = node.accessible->get_accChild(child, &dispatch);
            // S_FALSE names a simple element without an object of its own: nothing to visit.
            if (child_status == S_OK && dispatch) {
              children_read = queue_child(dispatch, first + index);
              dispatch->Release();
            } else if (child_status == S_OK || FAILED(child_status)) {
              children_read = false;
            }
          } else {
            children_read = false;
          }
        }
        for (auto &child : children) {
          VariantClear(&child);
        }
        if (!children_read) {
          (void) left_the_tree(__LINE__, children_status);
          break;
        }
        first += obtained;
      }
    }

    if (!aborted) {
      // The walk is not one moment: a video element the page replaced during it was met twice,
      // as the node it was and as the node it is. A candidate that no longer answers as itself
      // has left the tree, and is not a second video.
      result.vanished_nodes += std::erase_if(result.candidates, [](const observed_candidate_t &candidate) {
        auto ia2_query = query_ia2(candidate.accessible.get());
        LONG unique_id = 0;
        return ia2_query.status != S_OK || !ia2_query.object ||
               ia2_query.object->get_uniqueID(&unique_id) != S_OK || unique_id != candidate.geometry.unique_id;
      });
    }
    if (std::chrono::steady_clock::now() - started >= maximum_scan_work_time) {
      complete = false;
    }
    DWORD process_id_after = 0;
    const auto process_name_after = process_name_for_window(foreground, process_id_after);
    if (foreground_changed || !foreground_unchanged(foreground, result.process_id, result.client_rect) || process_name_after != result.process_name || client_fills_monitor(foreground, result.client_rect) != result.client_fills_monitor) {
      result.status = scan_status_e::foreground_changed;
      result.candidates.clear();
    } else if (!complete) {
      result.status = scan_status_e::traversal_incomplete;
    } else if (result.ia2_nodes == 0 || result.document_nodes == 0) {
      result.status = scan_status_e::accessibility_unavailable;
    } else {
      result.extended_properties_ready = result.tagged_nodes > 0;
      result.status = scan_status_e::warming_up;
    }

    result.elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - started
    );
    return result;
  }

  struct cached_selection_t {
    probe::cached_selection_phase_e phase {
      probe::cached_selection_phase_e::provisional
    };
    HWND window {};
    DWORD process_id {};
    probe::rect_t client_rect {};
    LONG document_unique_id {};
    LONG video_unique_id {};
    probe::rect_t document_rect {};
    probe::rect_t element_rect {};
    probe::rect_t visible_rect {};
    bool fullscreen {};
    bool client_fills_monitor {};
    std::string semantic_fingerprint;
    com_ptr_t<IAccessible> accessible;
    com_ptr_t<IAccessible> document_accessible;
    // It has been established, here or in the selection it was staged again from: worth
    // remembering when it is dropped for anything but a census that found no video to select.
    bool established_once {};
    // How often in a row it was staged again from remembered objects without being established.
    unsigned restages {};
  };

  struct accessible_snapshot_t {
    LONG unique_id {};
    LONG role {};
    DWORD state {};
    probe::rect_t rect {};
  };

  [[nodiscard]] std::optional<accessible_snapshot_t> snapshot_accessible(
    IAccessible *accessible,
    LONG expected_unique_id,
    bool require_document,
    bool require_video
  ) {
    auto ia2_query = query_ia2(accessible);
    if (ia2_query.status != S_OK || !ia2_query.object) {
      return std::nullopt;
    }

    LONG unique_id_before = 0;
    LONG role = 0;
    if (ia2_query.object->get_uniqueID(&unique_id_before) != S_OK ||
        unique_id_before == 0 || unique_id_before != expected_unique_id ||
        ia2_query.object->role(&role) != S_OK ||
        (require_document && role != ROLE_SYSTEM_DOCUMENT)) {
      return std::nullopt;
    }

    if (require_video) {
      bstr_t attributes;
      if (ia2_query.object->get_attributes(attributes.put()) != S_OK ||
          !attributes.get()) {
        return std::nullopt;
      }
      const auto match = probe::match_video_tag(std::wstring_view(
        attributes.get(),
        static_cast<std::size_t>(SysStringLen(attributes.get()))
      ));
      if (!match.valid || !match.is_video) {
        return std::nullopt;
      }
    }

    const auto state = accessible_state(accessible);
    const auto rect = accessible_rect(accessible);
    LONG unique_id_after = 0;
    if (!state || !rect ||
        ia2_query.object->get_uniqueID(&unique_id_after) != S_OK ||
        unique_id_after != unique_id_before) {
      return std::nullopt;
    }
    constexpr DWORD unavailable_states =
      STATE_SYSTEM_UNAVAILABLE | STATE_SYSTEM_INVISIBLE | STATE_SYSTEM_OFFSCREEN;
    if ((*state & unavailable_states) != 0) {
      return std::nullopt;
    }
    return accessible_snapshot_t {
      .unique_id = unique_id_before,
      .role = role,
      .state = *state,
      .rect = *rect,
    };
  }

  [[nodiscard]] std::optional<cached_selection_t> cache_complete_ready_selection(
    const scan_result_t &result,
    const probe::selection_t &selection,
    std::string semantics,
    probe::cached_selection_phase_e phase
  ) {
    if (result.status != scan_status_e::warming_up ||
        !result.extended_properties_ready ||
        !probe::selection_can_stage_provisional(selection) ||
        *selection.index >= result.candidates.size()) {
      return std::nullopt;
    }
    const auto &candidate = result.candidates[*selection.index];
    const std::array geometry {candidate.geometry};
    const probe::selection_t local_selection {
      .index = 0,
      .reason = selection.reason,
    };
    const auto authority_rect = probe::selection_authority_rect(
      geometry,
      local_selection,
      result.client_rect,
      coordinate_rounding_tolerance
    );
    if (!candidate.accessible || !candidate.document_accessible || !authority_rect) {
      return std::nullopt;
    }
    return cached_selection_t {
      .phase = phase,
      .window = result.window,
      .process_id = result.process_id,
      .client_rect = result.client_rect,
      .document_unique_id = candidate.document_unique_id,
      .video_unique_id = candidate.geometry.unique_id,
      .document_rect = candidate.document_rect,
      .element_rect = candidate.geometry.element_rect,
      .visible_rect = *authority_rect,
      .fullscreen = selection.reason == probe::selection_reason_e::fullscreen,
      .client_fills_monitor = result.client_fills_monitor,
      .semantic_fingerprint = std::move(semantics),
      .accessible = retain_com(candidate.accessible.get()),
      .document_accessible = retain_com(candidate.document_accessible.get()),
    };
  }

  [[nodiscard]] bool refresh_cached_selection(const cached_selection_t &cached) {
    const auto foreground_generation = foreground_change_generation.load(std::memory_order_relaxed);
    const auto foreground_current = [&]() {
      pump_messages();
      const auto events = probe::machine_event_policy(
        {.foreground = foreground_generation},
        {.foreground = foreground_change_generation.load(std::memory_order_relaxed)}
      );
      if (events.invalidate_semantic_state) {
        return false;
      }
      return foreground_unchanged(cached.window, cached.process_id, cached.client_rect) &&
             (!cached.fullscreen || !cached.client_fills_monitor || client_fills_monitor(cached.window, cached.client_rect));
    };
    if (!foreground_current()) {
      return false;
    }

    const auto document = snapshot_accessible(
      cached.document_accessible.get(),
      cached.document_unique_id,
      true,
      false
    );
    const auto video = snapshot_accessible(
      cached.accessible.get(),
      cached.video_unique_id,
      false,
      true
    );
    // COM calls may span a foreground/client/topology change, including A->B->A with matching
    // geometry. Pump events and recheck the original foreground epoch plus Win32 authority after
    // querying the retained semantic objects.
    if (!document || !video || !foreground_current()) {
      return false;
    }

    if (cached.fullscreen) {
      return probe::fullscreen_cache_refresh_matches(
        cached.video_unique_id,
        video->unique_id,
        cached.visible_rect,
        cached.client_rect,
        video->rect,
        coordinate_rounding_tolerance
      );
    }

    return probe::windowed_cache_refresh_matches(
      cached.element_rect,
      cached.visible_rect,
      cached.client_rect,
      document->rect,
      video->rect
    );
  }

  [[nodiscard]] std::string rect_string(const probe::rect_t &rect) {
    std::ostringstream output;
    output << rect.left << ':' << rect.top << '-' << rect.right << ':' << rect.bottom;
    return output.str();
  }

  [[nodiscard]] probe::selection_t select_video(const scan_result_t &result) {
    std::vector<probe::video_candidate_t> geometry;
    geometry.reserve(result.candidates.size());
    for (const auto &candidate : result.candidates) {
      geometry.push_back(candidate.geometry);
    }
    return probe::select_authority_candidate(
      geometry,
      result.client_rect,
      coordinate_rounding_tolerance,
      result.client_fills_monitor
    );
  }

  [[nodiscard]] std::string semantic_fingerprint(
    const scan_result_t &result,
    const probe::selection_t &selection
  ) {
    std::ostringstream output;
    output << reinterpret_cast<std::uintptr_t>(result.window) << ':'
           << result.process_id << ':' << rect_string(result.client_rect) << ':'
           << static_cast<int>(selection.reason) << ':';
    if (selection.index) {
      const auto &candidate = result.candidates[*selection.index];
      output << candidate.document_unique_id << '@'
             << candidate.geometry.unique_id << '@';
      if (selection.reason == probe::selection_reason_e::fullscreen) {
        // Fullscreen authority is the client. Document clipping and harmless element overscan
        // changes do not alter the semantic identity of the retained video object.
        output << "fullscreen";
      } else {
        output << rect_string(candidate.geometry.element_rect) << '@'
               << rect_string(candidate.geometry.visible_rect);
      }
    }
    return output.str();
  }

  [[nodiscard]] std::string scan_fingerprint(const scan_result_t &result) {
    std::ostringstream output;
    output << static_cast<int>(result.status) << ':'
           << semantic_fingerprint(result, result.selection) << ':';
    output << ':' << static_cast<int>(result.selection.reason) << ':';
    if (result.selection.index) {
      output << result.candidates[*result.selection.index].geometry.unique_id;
    }
    return output.str();
  }

  void print_scan(const scan_result_t &result, bool unchanged = false) {
    std::cout << "status=" << status_name(result.status)
              << " browser=" << utf8(result.process_name)
              << " pid=" << result.process_id
              << " hwnd=0x" << std::hex
              << reinterpret_cast<std::uintptr_t>(result.window) << std::dec
              << " client=" << rect_string(result.client_rect)
              << " client_fills_monitor=" << result.client_fills_monitor
              << " nodes=" << result.visited_nodes
              << " ia2=" << result.ia2_nodes
              << " documents=" << result.document_nodes
              << " tagged=" << result.tagged_nodes
              << " candidates=" << result.candidates.size()
              << " errors=" << result.errors
              << " scan_ms=" << std::fixed << std::setprecision(2)
              << static_cast<double>(result.elapsed.count()) / 1000.0;
    if (unchanged) {
      std::cout << " unchanged=1";
    }
    std::cout << '\n';

    if (unchanged) {
      return;
    }
    for (std::size_t index = 0; index < result.candidates.size(); ++index) {
      const auto &candidate = result.candidates[index];
      std::cout << "  video[" << index << "] id=" << candidate.geometry.unique_id
                << " element=" << rect_string(candidate.geometry.element_rect)
                << " visible=" << rect_string(candidate.geometry.visible_rect)
                << " available=" << candidate.geometry.available
                << " fullscreen_available="
                << candidate.geometry.fullscreen_available
                << " contained=" << candidate.geometry.fully_contained
                << " credible=" << candidate.geometry.credible_size
                << " document=" << candidate.document_unique_id
                << " state=0x" << std::hex << candidate.state << std::dec
                << '\n';
    }
    std::cout << "  selection=" << probe::selection_reason_name(result.selection.reason);
    if (result.selection.index) {
      const auto &selected = result.candidates[*result.selection.index];
      const std::array geometry {selected.geometry};
      const probe::selection_t local_selection {
        .index = 0,
        .reason = result.selection.reason,
      };
      const auto authority_rect = probe::selection_authority_rect(
        geometry,
        local_selection,
        result.client_rect,
        coordinate_rounding_tolerance
      );
      std::cout << " id=" << selected.geometry.unique_id
                << " rect=" << rect_string(
                     authority_rect.value_or(selected.geometry.visible_rect)
                   );
    }
    std::cout << '\n'
              << std::flush;
  }

  BOOL WINAPI console_handler(DWORD event) {
    if (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT || event == CTRL_CLOSE_EVENT) {
      keep_running.store(false, std::memory_order_relaxed);
      return TRUE;
    }
    return FALSE;
  }

  void CALLBACK accessibility_event(
    HWINEVENTHOOK,
    DWORD event,
    HWND window,
    LONG,
    LONG,
    DWORD,
    DWORD
  ) {
    if (event == EVENT_SYSTEM_FOREGROUND) {
      foreground_change_generation.fetch_add(1, std::memory_order_relaxed);
      object_change_generation.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    if (event != EVENT_OBJECT_CREATE && event != EVENT_OBJECT_DESTROY &&
        event != EVENT_OBJECT_SHOW && event != EVENT_OBJECT_HIDE &&
        event != EVENT_OBJECT_REORDER && event != EVENT_OBJECT_LOCATIONCHANGE) {
      return;
    }
    const auto monitored = reinterpret_cast<HWND>(
      monitored_window_value.load(std::memory_order_relaxed)
    );
    if (monitored && window && GetAncestor(window, GA_ROOT) == monitored) {
      // DOM churn is only a request to audit. The cached video object is independently
      // revalidated every machine tick, so unrelated adverts or controls must not erase it.
      object_change_generation.fetch_add(1, std::memory_order_relaxed);
    }
  }

  void pump_messages() {
    MSG message;
    while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
      TranslateMessage(&message);
      DispatchMessageW(&message);
    }
  }

  struct options_t {
    std::chrono::milliseconds interval {1000};
    bool all_scans {};
    bool machine {};
    bool trace {};
    bool interval_explicit {};
  };

  [[nodiscard]] std::optional<options_t> parse_options(int argc, char **argv) {
    options_t options;
    for (int index = 1; index < argc; ++index) {
      const std::string_view argument = argv[index];
      if (argument == "--all-scans") {
        options.all_scans = true;
      } else if (argument == "--trace") {
        options.trace = true;
      } else if (argument == "--machine") {
        options.machine = true;
      } else if (argument == "--interval-ms" && index + 1 < argc) {
        const std::string_view value = argv[++index];
        int milliseconds = 0;
        const auto [end, error] = std::from_chars(
          value.data(),
          value.data() + value.size(),
          milliseconds
        );
        if (error != std::errc {} || end != value.data() + value.size() || milliseconds < 50 || milliseconds > 10000) {
          std::cerr << "--interval-ms must be between 50 and 10000\n";
          return std::nullopt;
        }
        options.interval = std::chrono::milliseconds(milliseconds);
        options.interval_explicit = true;
      } else if (argument == "--help" || argument == "-h") {
        std::cout
          << "Usage: video-dom-info [--all-scans] [--interval-ms 50..10000]\n"
          << "       video-dom-info --machine [--interval-ms 50..1000]\n"
          << "Keep Chrome or Edge in the foreground while the probe runs.\n"
          << "Machine mode stages a unique selection from one complete census and authorizes it "
             "only after an exact cached-object refresh on the next tick.\n"
          << "Human diagnostics retain the two-scan semantic display filter.\n"
          << "--machine emits only strict SUNSHINE_VIDEO_DOM_V1 TSV records on stdout.\n"
          << "--trace adds one line per census on stderr: how it ended, and the source line of each failed step.\n";
        return std::nullopt;
      } else {
        std::cerr << "Unknown option: " << argument << '\n';
        return std::nullopt;
      }
    }
    if (options.machine && options.all_scans) {
      std::cerr << "--machine and --all-scans cannot be combined\n";
      return std::nullopt;
    }
    if (options.machine) {
      if (!options.interval_explicit) {
        options.interval = 100ms;
      }
      if (options.interval > 1000ms) {
        std::cerr << "--machine interval must not exceed 1000 ms\n";
        return std::nullopt;
      }
    } else if (options.interval < 1000ms) {
      std::cerr << "human diagnostic interval must be at least 1000 ms\n";
      return std::nullopt;
    }
    return options;
  }

  struct machine_record_t {
    std::string_view status {"warming"};
    std::uintptr_t window {};
    DWORD process_id {};
    LONG document_unique_id {};
    LONG video_unique_id {};
    probe::rect_t rect {};

    [[nodiscard]] bool operator==(const machine_record_t &) const = default;
  };

  [[nodiscard]] machine_record_t invalid_machine_record(std::string_view status) {
    return {.status = status};
  }

  [[nodiscard]] machine_record_t machine_record_for_result(
    const scan_result_t &result
  ) {
    switch (result.status) {
      case scan_status_e::no_foreground_window:
        return invalid_machine_record("no-foreground");
      case scan_status_e::unsupported_foreground:
        return invalid_machine_record("unsupported");
      case scan_status_e::unavailable_window:
        return invalid_machine_record("unavailable");
      case scan_status_e::accessibility_unavailable:
        return invalid_machine_record("accessibility");
      case scan_status_e::warming_up:
        return invalid_machine_record("warming");
      case scan_status_e::traversal_incomplete:
        return invalid_machine_record("incomplete");
      case scan_status_e::foreground_changed:
        return invalid_machine_record("changed");
      case scan_status_e::ok:
        break;
    }
    if (result.selection.reason == probe::selection_reason_e::ambiguous) {
      return invalid_machine_record("ambiguous");
    }
    if (!result.selection.index || *result.selection.index >= result.candidates.size()) {
      return invalid_machine_record("no-video");
    }
    const auto &candidate = result.candidates[*result.selection.index];
    const std::array geometry {candidate.geometry};
    const probe::selection_t local_selection {
      .index = 0,
      .reason = result.selection.reason,
    };
    const auto authority_rect = probe::selection_authority_rect(
      geometry,
      local_selection,
      result.client_rect,
      coordinate_rounding_tolerance
    );
    if (!authority_rect ||
        result.window == nullptr || result.process_id == 0 ||
        candidate.document_unique_id == 0 || candidate.geometry.unique_id == 0) {
      return invalid_machine_record("incomplete");
    }
    return {
      .status = result.selection.reason == probe::selection_reason_e::fullscreen ?
                  "ok-fullscreen" :
                  "ok",
      .window = reinterpret_cast<std::uintptr_t>(result.window),
      .process_id = result.process_id,
      .document_unique_id = candidate.document_unique_id,
      .video_unique_id = candidate.geometry.unique_id,
      .rect = *authority_rect,
    };
  }

  [[nodiscard]] machine_record_t machine_record_for_cache(
    const cached_selection_t &cached
  ) {
    return {
      .status = cached.fullscreen ? "ok-fullscreen" : "ok",
      .window = reinterpret_cast<std::uintptr_t>(cached.window),
      .process_id = cached.process_id,
      .document_unique_id = cached.document_unique_id,
      .video_unique_id = cached.video_unique_id,
      .rect = cached.visible_rect,
    };
  }

  void print_machine_record(std::uint64_t sequence, const machine_record_t &record) {
    std::cout << "SUNSHINE_VIDEO_DOM_V1\t" << sequence
              << '\t' << record.status
              << '\t' << record.window
              << '\t' << record.process_id
              << '\t' << record.document_unique_id
              << '\t' << record.video_unique_id
              << '\t' << record.rect.left
              << '\t' << record.rect.top
              << '\t' << record.rect.right
              << '\t' << record.rect.bottom
              << '\n'
              << std::flush;
  }

  int run_human(const options_t &options) {
    std::cout
      << "This human mode is diagnostics only; production uses --machine.\n"
      << "Switch to a Chrome or Edge window. Press Ctrl+C to stop.\n";

    std::string last_fingerprint;
    std::string pending_semantic_fingerprint;
    unsigned stable_ready_scans = 0;
    auto last_report = std::chrono::steady_clock::time_point {};
    do {
      pump_messages();
      auto result = scan_foreground([](const auto &) { return true; });
      const auto tentative_selection = select_video(result);
      if (result.status == scan_status_e::warming_up && result.extended_properties_ready) {
        const auto current_semantics = semantic_fingerprint(result, tentative_selection);
        if (current_semantics == pending_semantic_fingerprint) {
          ++stable_ready_scans;
        } else {
          pending_semantic_fingerprint = current_semantics;
          stable_ready_scans = 1;
        }
        if (stable_ready_scans >= 2) {
          result.status = scan_status_e::ok;
        }
      } else {
        pending_semantic_fingerprint.clear();
        stable_ready_scans = 0;
      }

      if (result.status == scan_status_e::ok) {
        result.selection = tentative_selection;
      }
      const auto fingerprint = scan_fingerprint(result);
      const auto now = std::chrono::steady_clock::now();
      const bool changed = fingerprint != last_fingerprint;
      const bool heartbeat = now - last_report >= 5s;
      if (options.all_scans || changed || heartbeat) {
        print_scan(result, !changed && !options.all_scans);
        last_report = now;
      }
      last_fingerprint = fingerprint;
      std::this_thread::sleep_for(options.interval);
    } while (keep_running.load(std::memory_order_relaxed));
    return 0;
  }

  observed_candidate_t observe_candidate(
    IAccessible *accessible,
    const LONG unique_id,
    const DWORD state,
    const probe::rect_t &element_rect,
    const std::optional<document_context_t> &document,
    const probe::rect_t &client_rect
  ) {
    constexpr DWORD unavailable_states =
      STATE_SYSTEM_UNAVAILABLE | STATE_SYSTEM_INVISIBLE | STATE_SYSTEM_OFFSCREEN;
    const auto visible_rect = document ?
                                probe::windowed_video_viewport(client_rect, document->rect, element_rect) :
                                std::nullopt;
    const bool state_available = (state & unavailable_states) == 0;
    observed_candidate_t candidate;
    candidate.geometry.unique_id = unique_id;
    candidate.geometry.element_rect = element_rect;
    candidate.geometry.visible_rect = visible_rect.value_or(probe::rect_t {});
    candidate.geometry.available =
      state_available && visible_rect.has_value() && document && document->state_available;
    candidate.geometry.fully_contained =
      document &&
      probe::clip_if_within_tolerance(client_rect, element_rect, coordinate_rounding_tolerance).has_value() &&
      probe::clip_if_within_tolerance(document->rect, element_rect, coordinate_rounding_tolerance).has_value();
    candidate.geometry.credible_size = probe::credible_video_extent(element_rect, client_rect);
    candidate.geometry.fullscreen_available = probe::fullscreen_semantic_available(
      state_available,
      document.has_value(),
      document && document->state_available
    );
    candidate.state = state;
    candidate.document_unique_id = document ? document->unique_id : 0;
    candidate.document_rect = document ? document->rect : probe::rect_t {};
    candidate.accessible = retain_com(accessible);
    candidate.document_accessible = retain_com(document ? document->accessible : nullptr);
    return candidate;
  }

  // The window the helper had a video in is in front again, or was moved or resized, and its page
  // is most likely the one it was. The retained document and video objects are asked again: if
  // they are the same available objects, the selection is staged from them as a census stages
  // it, with the geometry they have now. Like every staged selection it is published only after
  // the next tick's exact refresh, and the caller has a census follow at once as its audit, which
  // revokes it if the page has since gained a second video.
  [[nodiscard]] std::optional<cached_selection_t> restage_remembered_selection(const cached_selection_t &old) {
    const HWND foreground = GetForegroundWindow();
    if (!foreground || GetAncestor(foreground, GA_ROOT) != old.window || !window_available(old.window)) {
      return std::nullopt;
    }
    DWORD process_id = 0;
    GetWindowThreadProcessId(old.window, &process_id);
    const auto client = foreground_client_rect(old.window);
    if (process_id != old.process_id || !client) {
      return std::nullopt;
    }
    const auto document = snapshot_accessible(old.document_accessible.get(), old.document_unique_id, true, false);
    const auto video = snapshot_accessible(old.accessible.get(), old.video_unique_id, false, true);
    if (!document || !video) {
      return std::nullopt;
    }
    scan_result_t result;
    result.status = scan_status_e::warming_up;
    result.extended_properties_ready = true;
    result.window = old.window;
    result.process_id = process_id;
    result.client_rect = *client;
    result.client_fills_monitor = client_fills_monitor(old.window, *client);
    const document_context_t context {
      .unique_id = document->unique_id,
      .rect = document->rect,
      .state_available = true,  // snapshot_accessible() accepts no other.
      .accessible = old.document_accessible.get(),
    };
    result.candidates.push_back(observe_candidate(
      old.accessible.get(), video->unique_id, video->state, video->rect, context, result.client_rect
    ));
    const auto selection = select_video(result);
    auto staged = cache_complete_ready_selection(
      result,
      selection,
      semantic_fingerprint(result, selection),
      probe::cached_selection_phase_e::provisional
    );
    if (staged) {
      staged->established_once = true;
      staged->restages = old.restages + 1u;
    }
    return staged;
  }

  int run_machine(const options_t &options) {
    std::optional<cached_selection_t> cached;
    // The selection last dropped for a foreground or geometry change, or a refresh that failed:
    // staged again, without a census, when its window is in front. See
    // restage_remembered_selection().
    struct remembered_t {
      cached_selection_t selection;
      std::chrono::steady_clock::time_point at;
    };
    std::vector<remembered_t> remembered;  // One for each of the last few windows.
    const auto forget_window = [&](const HWND window) {
      std::erase_if(remembered, [&](const remembered_t &entry) {
        return entry.selection.window == window;
      });
    };
    const auto set_aside = [&]() {
      if (cached && probe::selection_worth_remembering(cached->established_once, cached->restages)) {
        forget_window(cached->window);
        if (remembered.size() >= maximum_remembered_selections) {
          remembered.erase(remembered.begin());
        }
        remembered.push_back({std::move(*cached), std::chrono::steady_clock::now()});
      }
      cached.reset();
    };
    const auto note_phase = [&]() {
      if (cached && cached->phase == probe::cached_selection_phase_e::established) {
        cached->established_once = true;
        cached->restages = 0u;
      }
    };
    unsigned cold_warmup_scans = 0;
    machine_record_t current = invalid_machine_record("warming");
    machine_record_t last_emitted = invalid_machine_record("changed");
    std::uint64_t sequence = 0;
    auto last_emit = std::chrono::steady_clock::time_point {};
    auto next_full_scan = std::chrono::steady_clock::now();
    auto last_full_scan = std::chrono::steady_clock::time_point {};
    auto observed_foreground_generation = foreground_change_generation.load(
      std::memory_order_relaxed
    );
    auto observed_object_generation = object_change_generation.load(
      std::memory_order_relaxed
    );

    auto publish = [&](machine_record_t record, bool force = false) {
      const auto now = std::chrono::steady_clock::now();
      current = record;
      if (force || current != last_emitted ||
          now - last_emit >= machine_heartbeat_interval) {
        print_machine_record(++sequence, current);
        last_emitted = current;
        last_emit = now;
        // The records go to the host that started this helper. Once it no longer reads them it
        // has ended, however it ended, and a helper nobody reads has no reason to go on walking
        // every browser page that comes to the front.
        if (!std::cout) {
          keep_running.store(false, std::memory_order_relaxed);
        }
      }
    };
    publish(current, true);

    while (keep_running.load(std::memory_order_relaxed)) {
      pump_messages();
      const auto foreground_generation = foreground_change_generation.load(
        std::memory_order_relaxed
      );
      if (foreground_generation != observed_foreground_generation) {
        observed_foreground_generation = foreground_generation;
        observed_object_generation = object_change_generation.load(
          std::memory_order_relaxed
        );
        set_aside();
        cold_warmup_scans = 0;
        next_full_scan = std::chrono::steady_clock::now();
        publish(invalid_machine_record("changed"), true);
        std::this_thread::sleep_for(options.interval);
        continue;
      }

      if (!cached && !remembered.empty()) {
        std::erase_if(remembered, [](const remembered_t &entry) {
          return std::chrono::steady_clock::now() - entry.at > remembered_selection_lifetime;
        });
        const HWND foreground = GetForegroundWindow();
        const HWND root = foreground ? GetAncestor(foreground, GA_ROOT) : nullptr;
        const auto entry = std::find_if(remembered.begin(), remembered.end(), [&](const remembered_t &candidate) {
          return candidate.selection.window == root;
        });
        if (root && entry != remembered.end()) {
          // One attempt: objects that no longer answer are not asked again.
          auto staged = restage_remembered_selection(entry->selection);
          remembered.erase(entry);
          if (staged) {
            cached = std::move(staged);
            // The census that audits it follows the exact refresh of the next tick.
            next_full_scan = std::chrono::steady_clock::now();
            if (options.trace) {
              std::cerr << "staged again without a census" << std::endl;
            }
            publish(invalid_machine_record("warming"));
            std::this_thread::sleep_for(options.interval);
            continue;
          }
        }
      }

      const auto now = std::chrono::steady_clock::now();
      const auto object_generation = object_change_generation.load(
        std::memory_order_relaxed
      );
      const auto tick_event_policy = probe::machine_event_policy(
        {
          .foreground = observed_foreground_generation,
          .object = observed_object_generation,
        },
        {
          .foreground = foreground_generation,
          .object = object_generation,
        }
      );
      observed_object_generation = object_generation;
      if (tick_event_policy.request_follow_up_audit) {
        const auto earliest_event_audit =
          last_full_scan.time_since_epoch().count() == 0 ?
            now : std::max(now, last_full_scan + event_full_scan_min_interval);
        next_full_scan = std::min(next_full_scan, earliest_event_audit);
      }
      if (cached) {
        const auto refresh_policy = probe::cached_refresh_policy(
          cached->phase,
          refresh_cached_selection(*cached),
          tick_event_policy
        );
        if (!refresh_policy.retain_selection) {
          set_aside();
          next_full_scan = now;
          publish(invalid_machine_record("changed"), true);
          std::this_thread::sleep_for(options.interval);
          continue;
        }
        cached->phase = refresh_policy.next_phase;
        note_phase();
        current = refresh_policy.publish_ok ?
                    machine_record_for_cache(*cached) :
                    invalid_machine_record("warming");
        if (now < next_full_scan) {
          publish(current);
          std::this_thread::sleep_for(options.interval);
          continue;
        }
      } else if (now < next_full_scan) {
        publish(current);
        std::this_thread::sleep_for(options.interval);
        continue;
      }

      const auto foreground_generation_before_scan = foreground_change_generation.load(
        std::memory_order_relaxed
      );
      const auto object_generation_before_scan = object_change_generation.load(
        std::memory_order_relaxed
      );
      // A complete tree walk may outlast the heartbeat budget. Checkpoints retain its traversal
      // progress while independently refreshing only existing authority; partial candidates stay
      // private until the complete census and the following tick's exact refresh succeed.
      publish(current, true);
      auto result = scan_foreground([&](const auto &foreground_current) {
        return probe::service_census_checkpoint([&]() {
          if (cached) {
            const auto refresh_policy = probe::cached_refresh_policy(
              cached->phase, refresh_cached_selection(*cached), {}
            );
            if (!refresh_policy.retain_selection) {
              set_aside();
              return invalid_machine_record("changed");
            }
            cached->phase = refresh_policy.next_phase;
        note_phase();
            return refresh_policy.publish_ok ? machine_record_for_cache(*cached) :
                                               invalid_machine_record("warming");
          }
          return invalid_machine_record("warming");
        }, foreground_current, [&](const auto &record) { publish(record); });
      });
      if (options.trace) {
        std::cerr << "census status=" << status_name(result.status) << " nodes=" << result.visited_nodes
                  << " candidates=" << result.candidates.size() << " errors=" << result.errors
                  << " ms=" << result.elapsed.count() / 1000 << " vanished=" << result.vanished_nodes
                  << " stopped=" << result.stopped << " failed-steps=";
        for (const auto &[line, count] : result.error_lines) {
          std::cerr << line << 'x' << count << ' ';
        }
        std::cerr << std::endl;
        for (const auto &note : result.error_notes) {
          std::cerr << "  " << note << std::endl;
        }
        for (const auto &candidate : result.candidates) {
          std::cerr << "  video id=" << candidate.geometry.unique_id << " document=" << candidate.document_unique_id
                    << " element=" << rect_string(candidate.geometry.element_rect) << " visible=" << rect_string(candidate.geometry.visible_rect)
                    << " available=" << candidate.geometry.available << " credible=" << candidate.geometry.credible_size
                    << " contained=" << candidate.geometry.fully_contained << " state=0x" << std::hex << candidate.state << std::dec << std::endl;
        }
      }
      // Successful scans route events before their first checkpoint. Also clear stale routing
      // when the census returned early for an unsupported foreground window.
      monitored_window_value.store(
        reinterpret_cast<std::uintptr_t>(
          result.status == scan_status_e::unsupported_foreground ? nullptr :
                                                                   result.window
        ),
        std::memory_order_relaxed
      );
      pump_messages();
      const auto foreground_generation_after_scan = foreground_change_generation.load(
        std::memory_order_relaxed
      );
      const auto object_generation_after_scan = object_change_generation.load(
        std::memory_order_relaxed
      );
      last_full_scan = std::chrono::steady_clock::now();
      const auto scan_event_policy = probe::machine_event_policy(
        {
          .foreground = foreground_generation_before_scan,
          .object = object_generation_before_scan,
        },
        {
          .foreground = foreground_generation_after_scan,
          .object = object_generation_after_scan,
        }
      );
      if (scan_event_policy.discard_completed_scan) {
        observed_foreground_generation = foreground_generation_after_scan;
        observed_object_generation = object_generation_after_scan;
        set_aside();
        cold_warmup_scans = 0;
        current = invalid_machine_record("changed");
        publish(current, true);
        next_full_scan = std::chrono::steady_clock::now() + options.interval;
        continue;
      }
      const bool object_changed_during_scan =
        scan_event_policy.request_follow_up_audit;

      if (
        result.status == scan_status_e::warming_up &&
        !result.extended_properties_ready
      ) {
        ++cold_warmup_scans;
        if (cold_warmup_scans >= maximum_cold_warmup_scans) {
          // Some Chromium policies/providers never expose extended IA2 properties. Do not keep
          // walking the entire tree every second forever; WinEvents still request an earlier audit.
          result.status = scan_status_e::accessibility_unavailable;
        }
      } else if (result.status == scan_status_e::warming_up) {
        cold_warmup_scans = 0;
      }
      const auto tentative_selection = select_video(result);
      const bool ready_observation =
        result.status == scan_status_e::warming_up &&
        result.extended_properties_ready;
      std::string semantics;
      bool staged_provisional = false;
      if (ready_observation) {
        semantics = semantic_fingerprint(result, tentative_selection);
        result.selection = tentative_selection;
        const bool same_established_selection =
          cached &&
          cached->phase == probe::cached_selection_phase_e::established &&
          semantics == cached->semantic_fingerprint;
        const auto census_policy = probe::complete_census_policy(
          tentative_selection,
          same_established_selection
        );
        if (census_policy.stage_selection) {
          auto replacement = cache_complete_ready_selection(
            result,
            tentative_selection,
            semantics,
            census_policy.phase
          );
          if (!replacement) {
            result.status = scan_status_e::traversal_incomplete;
            result.selection = {};
          } else {
            cached = std::move(replacement);
            note_phase();
            if (census_policy.publish_ok) {
              // The same established object was authenticated before this census. Replacing its
              // retained COM handles does not reduce its already-established authority.
              result.status = scan_status_e::ok;
            } else {
              // Do not publish from the census alone. The next machine tick must strictly refresh
              // these exact retained objects before this selection becomes authoritative.
              staged_provisional = true;
            }
          }
        } else if (census_policy.revoke_cached_selection) {
          // A complete census is immediately authoritative for fail-closed negative results.
          // Partial or equal-largest candidates are represented as ambiguous by select_video().
          result.status = scan_status_e::ok;
          cached.reset();
          forget_window(result.window);
        } else {
          result.status = scan_status_e::traversal_incomplete;
          result.selection = {};
        }
      }

      bool retained_cache_after_inconclusive_scan = false;
      if (!staged_provisional && result.status != scan_status_e::ok && (
        result.status == scan_status_e::warming_up ||
        result.status == scan_status_e::traversal_incomplete ||
        result.status == scan_status_e::accessibility_unavailable
      )) {
        // A partial census cannot disprove an exact selected DOM object that still independently
        // authenticates. Keep it through transient provider/tree failures and retry the census.
        retained_cache_after_inconclusive_scan =
          cached && refresh_cached_selection(*cached);
        if (!retained_cache_after_inconclusive_scan) {
          set_aside();
        }
      } else if (!staged_provisional && result.status != scan_status_e::ok) {
        // Foreground/window failures invalidate the old browser identity immediately.
        set_aside();
      }

      if (cached &&
          cached->phase == probe::cached_selection_phase_e::established &&
          (result.status == scan_status_e::ok ||
           retained_cache_after_inconclusive_scan)) {
        current = machine_record_for_cache(*cached);
      } else if (cached &&
                 cached->phase == probe::cached_selection_phase_e::provisional &&
                 (staged_provisional || retained_cache_after_inconclusive_scan)) {
        current = invalid_machine_record("warming");
      } else {
        current = machine_record_for_result(result);
      }
      pump_messages();
      const auto foreground_generation_before_publish = foreground_change_generation.load(
        std::memory_order_relaxed
      );
      const auto object_generation_before_publish = object_change_generation.load(
        std::memory_order_relaxed
      );
      const auto publish_event_policy = probe::machine_event_policy(
        {
          .foreground = foreground_generation_after_scan,
          .object = object_generation_after_scan,
        },
        {
          .foreground = foreground_generation_before_publish,
          .object = object_generation_before_publish,
        }
      );
      if (publish_event_policy.discard_completed_scan) {
        observed_foreground_generation = foreground_generation_before_publish;
        observed_object_generation = object_generation_before_publish;
        set_aside();
        cold_warmup_scans = 0;
        publish(invalid_machine_record("changed"), true);
        next_full_scan = std::chrono::steady_clock::now() + options.interval;
        continue;
      }
      const bool object_changed_before_publish =
        publish_event_policy.request_follow_up_audit;
      const bool object_changed_before_observation =
        object_changed_during_scan || object_changed_before_publish;
      observed_object_generation = object_generation_before_publish;
      if (object_changed_before_observation && cached) {
        // Descendant DOM activity is not proof that the selected video changed. In particular,
        // dynamic controls and adverts can emit location/reorder events throughout every full
        // tree walk. Established authority is revalidated here. A newly staged provisional stays
        // internal until the next machine tick; object churn neither promotes nor vetoes it, and
        // the event still requests an earlier full audit below.
        if (cached->phase == probe::cached_selection_phase_e::established) {
          const auto refresh_policy = probe::cached_refresh_policy(
            cached->phase,
            refresh_cached_selection(*cached),
            publish_event_policy
          );
          if (!refresh_policy.retain_selection) {
            set_aside();
            publish(invalid_machine_record("changed"), true);
            next_full_scan = last_full_scan + options.interval;
            std::this_thread::sleep_for(options.interval);
            continue;
          }
          cached->phase = refresh_policy.next_phase;
        note_phase();
          current = machine_record_for_cache(*cached);
        }
      }
      publish(current);
      const auto object_generation_after_publish = object_change_generation.load(
        std::memory_order_relaxed
      );
      const bool object_change_observed =
        object_changed_before_observation ||
        object_generation_after_publish != observed_object_generation;
      observed_object_generation = object_generation_after_publish;
      if (staged_provisional || result.status == scan_status_e::ok) {
        next_full_scan = last_full_scan +
                         (object_change_observed ? event_full_scan_min_interval :
                                                   periodic_full_scan_interval);
      } else if (retained_cache_after_inconclusive_scan) {
        next_full_scan = last_full_scan + event_full_scan_min_interval;
      } else {
        const auto outcome =
          result.status == scan_status_e::warming_up ?
            probe::uncached_scan_outcome_e::warming_up :
          result.status == scan_status_e::traversal_incomplete ?
            probe::uncached_scan_outcome_e::traversal_incomplete :
          result.status == scan_status_e::accessibility_unavailable ?
            probe::uncached_scan_outcome_e::accessibility_unavailable :
            probe::uncached_scan_outcome_e::other;
        auto retry_delay = probe::uncached_scan_retry_delay(outcome);
        if (object_change_observed) {
          retry_delay = std::min(
            retry_delay,
            std::chrono::duration_cast<std::chrono::milliseconds>(
              event_full_scan_min_interval
            )
          );
        }
        next_full_scan = last_full_scan + retry_delay;
      }
      std::this_thread::sleep_for(options.interval);
    }
    monitored_window_value.store(0, std::memory_order_relaxed);
    return 0;
  }
}  // namespace

int main(int argc, char **argv) {
  const auto options = parse_options(argc, argv);
  trace_nodes = options && options->trace;
  if (!options) {
    return argc > 1 && (std::string_view(argv[1]) == "--help" ||
                        std::string_view(argv[1]) == "-h") ?
             0 :
             2;
  }

  SetConsoleOutputCP(CP_UTF8);
  SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
  if (!AreDpiAwarenessContextsEqual(
        GetThreadDpiAwarenessContext(),
        DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
      )) {
    std::cerr << "Per-monitor-v2 DPI awareness is unavailable; refusing mixed coordinate spaces.\n";
    return 1;
  }
  SetConsoleCtrlHandler(console_handler, TRUE);

  const HRESULT com_status = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
  if (FAILED(com_status)) {
    std::cerr << "CoInitializeEx failed: 0x" << std::hex
              << static_cast<unsigned long>(com_status) << '\n';
    return 1;
  }

  HWINEVENTHOOK foreground_hook = nullptr;
  HWINEVENTHOOK object_hook = nullptr;
  if (options->machine) {
    foreground_hook = SetWinEventHook(
      EVENT_SYSTEM_FOREGROUND,
      EVENT_SYSTEM_FOREGROUND,
      nullptr,
      accessibility_event,
      0,
      0,
      WINEVENT_OUTOFCONTEXT | WINEVENT_SKIPOWNPROCESS
    );
    object_hook = SetWinEventHook(
      EVENT_OBJECT_CREATE,
      EVENT_OBJECT_LOCATIONCHANGE,
      nullptr,
      accessibility_event,
      0,
      0,
      WINEVENT_OUTOFCONTEXT | WINEVENT_SKIPOWNPROCESS
    );
    if (!foreground_hook || !object_hook) {
      std::cerr << "SetWinEventHook failed; refusing stale machine geometry\n";
      if (foreground_hook) {
        UnhookWinEvent(foreground_hook);
      }
      if (object_hook) {
        UnhookWinEvent(object_hook);
      }
      CoUninitialize();
      return 1;
    }
  }

  const int result = options->machine ? run_machine(*options) : run_human(*options);

  if (foreground_hook) {
    UnhookWinEvent(foreground_hook);
  }
  if (object_hook) {
    UnhookWinEvent(object_hook);
  }

  CoUninitialize();
  return result;
}
