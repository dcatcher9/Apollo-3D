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
  constexpr auto maximum_scan_work_time = 2s;
  constexpr auto machine_heartbeat_interval = 1s;
  constexpr auto periodic_full_scan_interval = 15s;
  constexpr auto event_full_scan_min_interval = 3s;
  constexpr unsigned maximum_cold_warmup_scans = 3;
  constexpr LONG coordinate_rounding_tolerance = 1;

  std::atomic_bool keep_running {true};
  std::atomic_uint64_t foreground_change_generation {1};
  std::atomic_uint64_t object_change_generation {1};
  std::atomic_uintptr_t monitored_window_value {};

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
    std::size_t visited_nodes {};
    std::size_t ia2_nodes {};
    std::size_t document_nodes {};
    std::size_t tagged_nodes {};
    std::size_t errors {};
    bool extended_properties_ready {};
    std::chrono::microseconds elapsed {};
    std::vector<observed_candidate_t> candidates;
    probe::selection_t selection;
  };

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
    bool available {};
    IAccessible *accessible {};
  };

  struct queued_node_t {
    com_ptr_t<IAccessible> accessible;
    std::size_t depth {};
    std::optional<document_context_t> document;
  };

  [[nodiscard]] bool queue_dispatch(
    std::deque<queued_node_t> &queue,
    IDispatch *dispatch,
    std::size_t depth,
    const std::optional<document_context_t> &document,
    std::size_t visited_nodes
  ) {
    if (queue.size() + visited_nodes >= maximum_nodes) {
      return false;
    }
    auto query = accessible_from_dispatch(dispatch);
    if (query.status != S_OK || !query.object) {
      return false;
    }
    queue.push_back({std::move(query.object), depth, document});
    return true;
  }

  scan_result_t scan_foreground() {
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
    const auto work_deadline = started + maximum_scan_work_time;
    std::deque<queued_node_t> queue;
    queue.push_back({com_ptr_t<IAccessible> {root_raw}, 0, std::nullopt});
    std::unordered_set<std::uintptr_t> visited;
    visited.reserve(4096);
    std::vector<com_ptr_t<IUnknown>> retained_identities;
    retained_identities.reserve(4096);
    std::vector<com_ptr_t<IAccessible>> retained_documents;
    retained_documents.reserve(16);

    while (!queue.empty()) {
      if (result.visited_nodes >= maximum_nodes || std::chrono::steady_clock::now() > work_deadline) {
        complete = false;
        break;
      }
      auto node = std::move(queue.front());
      queue.pop_front();
      if (!node.accessible) {
        ++result.errors;
        complete = false;
        continue;
      }

      auto identity_query = query_com_identity(node.accessible.get());
      if (identity_query.status != S_OK || !identity_query.object) {
        ++result.errors;
        complete = false;
        continue;
      }
      const auto identity = reinterpret_cast<std::uintptr_t>(identity_query.object.get());
      if (!visited.insert(identity).second) {
        continue;
      }
      retained_identities.push_back(std::move(identity_query.object));
      ++result.visited_nodes;

      auto current_document = node.document;
      auto ia2_query = query_ia2(node.accessible.get());
      if (ia2_query.status == S_OK && ia2_query.object) {
        auto &ia2_accessible = ia2_query.object;
        ++result.ia2_nodes;

        LONG unique_id_before = 0;
        LONG role = 0;
        if (ia2_accessible->get_uniqueID(&unique_id_before) != S_OK || unique_id_before == 0 || ia2_accessible->role(&role) != S_OK) {
          ++result.errors;
          complete = false;
        } else {
          if (role == ROLE_SYSTEM_DOCUMENT) {
            const auto state = accessible_state(node.accessible.get());
            const auto document_rect = accessible_rect(node.accessible.get());
            LONG unique_id_after = 0;
            if (!state || !document_rect || ia2_accessible->get_uniqueID(&unique_id_after) != S_OK || unique_id_after == 0 || unique_id_after != unique_id_before) {
              ++result.errors;
              complete = false;
            } else {
              constexpr DWORD unavailable_states =
                STATE_SYSTEM_UNAVAILABLE | STATE_SYSTEM_INVISIBLE | STATE_SYSTEM_OFFSCREEN;
              retained_documents.push_back(retain_com(node.accessible.get()));
              current_document = document_context_t {
                .unique_id = unique_id_before,
                .rect = *document_rect,
                .available = (*state & unavailable_states) == 0 &&
                             probe::clip_if_within_tolerance(
                               result.client_rect,
                               *document_rect,
                               coordinate_rounding_tolerance
                             ).has_value(),
                .accessible = retained_documents.back().get(),
              };
              ++result.document_nodes;
            }
          }

          bstr_t attributes;
          const HRESULT attributes_status = ia2_accessible->get_attributes(attributes.put());
          if (attributes_status == S_OK) {
            if (!attributes.get()) {
              ++result.errors;
              complete = false;
            } else {
              const auto length = static_cast<std::size_t>(SysStringLen(attributes.get()));
              const auto match = probe::match_video_tag(
                std::wstring_view(attributes.get(), length)
              );
              if (!match.valid) {
                ++result.errors;
                complete = false;
              } else {
                if (match.has_tag) {
                  ++result.tagged_nodes;
                }
                if (match.is_video) {
                  const auto state = accessible_state(node.accessible.get());
                  const auto element_rect = accessible_rect(node.accessible.get());
                  LONG unique_id_after = 0;
                  if (!state || !element_rect || ia2_accessible->get_uniqueID(&unique_id_after) != S_OK || unique_id_after == 0 || unique_id_after != unique_id_before) {
                    ++result.errors;
                    complete = false;
                  } else {
                    constexpr DWORD unavailable_states =
                      STATE_SYSTEM_UNAVAILABLE | STATE_SYSTEM_INVISIBLE |
                      STATE_SYSTEM_OFFSCREEN;
                    auto visible_rect = probe::rect_intersection(
                      *element_rect,
                      result.client_rect
                    );
                    if (visible_rect && current_document) {
                      visible_rect = probe::rect_intersection(
                        *visible_rect,
                        current_document->rect
                      );
                    }
                    const auto client_area = probe::rect_area(result.client_rect);
                    const auto element_area = probe::rect_area(*element_rect);
                    observed_candidate_t candidate;
                    candidate.geometry.unique_id = unique_id_before;
                    candidate.geometry.element_rect = *element_rect;
                    candidate.geometry.visible_rect = visible_rect.value_or(probe::rect_t {});
                    candidate.geometry.available =
                      (*state & unavailable_states) == 0 && visible_rect.has_value() &&
                      current_document && current_document->available;
                    candidate.geometry.fully_contained =
                      current_document &&
                      probe::clip_if_within_tolerance(
                        result.client_rect,
                        *element_rect,
                        coordinate_rounding_tolerance
                      ).has_value() &&
                      probe::clip_if_within_tolerance(
                        current_document->rect,
                        *element_rect,
                        coordinate_rounding_tolerance
                      ).has_value();
                    candidate.geometry.credible_size =
                      client_area > 0 && element_area >= client_area / 20;
                    candidate.state = *state;
                    candidate.document_unique_id =
                      current_document ? current_document->unique_id : 0;
                    candidate.document_rect =
                      current_document ? current_document->rect : probe::rect_t {};
                    candidate.accessible = retain_com(node.accessible.get());
                    candidate.document_accessible = retain_com(
                      current_document ? current_document->accessible : nullptr
                    );
                    result.candidates.push_back(std::move(candidate));
                  }
                }
              }
            }
          } else if (FAILED(attributes_status)) {
            ++result.errors;
            complete = false;
          }
        }
      } else if (ia2_query.status != E_NOINTERFACE) {
        ++result.errors;
        complete = false;
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
      if (child_count_status != S_OK || child_count < 0 || child_count > maximum_children_per_node) {
        ++result.errors;
        complete = false;
        continue;
      }
      for (LONG first = 0; first < child_count;) {
        if (std::chrono::steady_clock::now() > work_deadline) {
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
        if (children_status != S_OK || obtained < 0 || obtained > requested) {
          for (auto &child : children) {
            VariantClear(&child);
          }
          ++result.errors;
          complete = false;
          break;
        }

        for (LONG index = 0; index < obtained; ++index) {
          auto &child = children[static_cast<std::size_t>(index)];
          if (child.vt == VT_DISPATCH && child.pdispVal) {
            if (!queue_dispatch(
                  queue,
                  child.pdispVal,
                  node.depth + 1,
                  current_document,
                  result.visited_nodes
                )) {
              ++result.errors;
              complete = false;
            }
          } else if (child.vt == VT_I4) {
            IDispatch *dispatch = nullptr;
            const HRESULT child_status = node.accessible->get_accChild(child, &dispatch);
            if (child_status == S_OK && dispatch) {
              if (!queue_dispatch(
                    queue,
                    dispatch,
                    node.depth + 1,
                    current_document,
                    result.visited_nodes
                  )) {
                ++result.errors;
                complete = false;
              }
              dispatch->Release();
            } else if (child_status == S_OK) {
              ++result.errors;
              complete = false;
            } else if (FAILED(child_status)) {
              ++result.errors;
              complete = false;
            }
          } else {
            ++result.errors;
            complete = false;
          }
        }
        for (auto &child : children) {
          VariantClear(&child);
        }
        if (obtained == 0) {
          if (first < child_count) {
            complete = false;
          }
          break;
        }
        first += obtained;
      }
    }

    const HWND foreground_after = GetAncestor(GetForegroundWindow(), GA_ROOT);
    DWORD process_id_after = 0;
    const auto process_name_after = process_name_for_window(
      foreground_after,
      process_id_after
    );
    const auto client_rect_after = foreground_client_rect(foreground_after);
    if (foreground_after != foreground || process_id_after != result.process_id || process_name_after != result.process_name || !client_rect_after || *client_rect_after != result.client_rect || !window_available(foreground_after)) {
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
    HWND window {};
    DWORD process_id {};
    probe::rect_t client_rect {};
    LONG document_unique_id {};
    LONG video_unique_id {};
    probe::rect_t document_rect {};
    probe::rect_t element_rect {};
    probe::rect_t visible_rect {};
    std::string semantic_fingerprint;
    com_ptr_t<IAccessible> accessible;
    com_ptr_t<IAccessible> document_accessible;
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

  [[nodiscard]] std::optional<cached_selection_t> cache_selection(
    const scan_result_t &result,
    const probe::selection_t &selection,
    std::string semantics
  ) {
    if (result.status != scan_status_e::ok || !selection.index ||
        *selection.index >= result.candidates.size()) {
      return std::nullopt;
    }
    const auto &candidate = result.candidates[*selection.index];
    if (!candidate.accessible || !candidate.document_accessible ||
        !probe::rect_valid(candidate.geometry.visible_rect)) {
      return std::nullopt;
    }
    return cached_selection_t {
      .window = result.window,
      .process_id = result.process_id,
      .client_rect = result.client_rect,
      .document_unique_id = candidate.document_unique_id,
      .video_unique_id = candidate.geometry.unique_id,
      .document_rect = candidate.document_rect,
      .element_rect = candidate.geometry.element_rect,
      .visible_rect = candidate.geometry.visible_rect,
      .semantic_fingerprint = std::move(semantics),
      .accessible = retain_com(candidate.accessible.get()),
      .document_accessible = retain_com(candidate.document_accessible.get()),
    };
  }

  [[nodiscard]] bool refresh_cached_selection(const cached_selection_t &cached) {
    HWND foreground = GetForegroundWindow();
    if (!foreground || GetAncestor(foreground, GA_ROOT) != cached.window ||
        !window_available(cached.window)) {
      return false;
    }
    DWORD process_id = 0;
    GetWindowThreadProcessId(cached.window, &process_id);
    const auto client_rect = foreground_client_rect(cached.window);
    if (process_id != cached.process_id || !client_rect ||
        *client_rect != cached.client_rect) {
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
    if (!document || !video || document->rect != cached.document_rect ||
        video->rect != cached.element_rect) {
      return false;
    }

    const auto document_in_client = probe::clip_if_within_tolerance(
      *client_rect,
      document->rect,
      coordinate_rounding_tolerance
    );
    const auto video_in_client = probe::clip_if_within_tolerance(
      *client_rect,
      video->rect,
      coordinate_rounding_tolerance
    );
    const auto video_in_document = probe::clip_if_within_tolerance(
      document->rect,
      video->rect,
      coordinate_rounding_tolerance
    );
    if (!document_in_client || !video_in_client || !video_in_document ||
        probe::rect_area(video->rect) < probe::rect_area(*client_rect) / 20) {
      return false;
    }
    const auto visible = probe::rect_intersection(*video_in_client, *video_in_document);
    return visible && *visible == cached.visible_rect;
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
    return probe::select_candidate(geometry);
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
             << candidate.geometry.unique_id << '@'
             << rect_string(candidate.geometry.element_rect) << '@'
             << rect_string(candidate.geometry.visible_rect);
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
                << " contained=" << candidate.geometry.fully_contained
                << " credible=" << candidate.geometry.credible_size
                << " document=" << candidate.document_unique_id
                << " state=0x" << std::hex << candidate.state << std::dec
                << '\n';
    }
    std::cout << "  selection=" << probe::selection_reason_name(result.selection.reason);
    if (result.selection.index) {
      const auto &selected = result.candidates[*result.selection.index];
      std::cout << " id=" << selected.geometry.unique_id
                << " rect=" << rect_string(selected.geometry.visible_rect);
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
    bool interval_explicit {};
  };

  [[nodiscard]] std::optional<options_t> parse_options(int argc, char **argv) {
    options_t options;
    for (int index = 1; index < argc; ++index) {
      const std::string_view argument = argv[index];
      if (argument == "--all-scans") {
        options.all_scans = true;
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
          << "A cold Chromium accessibility tree needs two matching scans before status=ok.\n"
          << "--machine emits only strict SUNSHINE_VIDEO_DOM_V1 TSV records on stdout.\n";
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
    if (!probe::rect_valid(candidate.geometry.visible_rect) ||
        result.window == nullptr || result.process_id == 0 ||
        candidate.document_unique_id == 0 || candidate.geometry.unique_id == 0) {
      return invalid_machine_record("incomplete");
    }
    return {
      .status = "ok",
      .window = reinterpret_cast<std::uintptr_t>(result.window),
      .process_id = result.process_id,
      .document_unique_id = candidate.document_unique_id,
      .video_unique_id = candidate.geometry.unique_id,
      .rect = candidate.geometry.visible_rect,
    };
  }

  [[nodiscard]] machine_record_t machine_record_for_cache(
    const cached_selection_t &cached
  ) {
    return {
      .status = "ok",
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
      << "video-dom-info is diagnostics only; it never changes capture or SBS output.\n"
      << "Switch to a Chrome or Edge window. Press Ctrl+C to stop.\n";

    std::string last_fingerprint;
    std::string pending_semantic_fingerprint;
    unsigned stable_ready_scans = 0;
    auto last_report = std::chrono::steady_clock::time_point {};
    do {
      pump_messages();
      auto result = scan_foreground();
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

  int run_machine(const options_t &options) {
    std::optional<cached_selection_t> cached;
    std::string pending_semantic_fingerprint;
    unsigned stable_ready_scans = 0;
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
        cached.reset();
        pending_semantic_fingerprint.clear();
        stable_ready_scans = 0;
        cold_warmup_scans = 0;
        next_full_scan = std::chrono::steady_clock::now();
        publish(invalid_machine_record("changed"), true);
        std::this_thread::sleep_for(options.interval);
        continue;
      }

      const auto now = std::chrono::steady_clock::now();
      const auto object_generation = object_change_generation.load(
        std::memory_order_relaxed
      );
      if (object_generation != observed_object_generation) {
        observed_object_generation = object_generation;
        const auto earliest_event_audit =
          last_full_scan.time_since_epoch().count() == 0 ?
            now : std::max(now, last_full_scan + event_full_scan_min_interval);
        next_full_scan = std::min(next_full_scan, earliest_event_audit);
      }
      if (cached) {
        if (!refresh_cached_selection(*cached)) {
          cached.reset();
          pending_semantic_fingerprint.clear();
          stable_ready_scans = 0;
          next_full_scan = now;
          publish(invalid_machine_record("changed"), true);
          std::this_thread::sleep_for(options.interval);
          continue;
        }
        current = machine_record_for_cache(*cached);
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
      // A bounded tree walk may consume most of the host freshness budget. Publish the latest
      // independently revalidated cache immediately before entering cross-process COM calls.
      publish(current, true);
      auto result = scan_foreground();
      pump_messages();
      const auto foreground_generation_after_scan = foreground_change_generation.load(
        std::memory_order_relaxed
      );
      last_full_scan = std::chrono::steady_clock::now();
      if (foreground_generation_after_scan != foreground_generation_before_scan) {
        observed_foreground_generation = foreground_generation_after_scan;
        observed_object_generation = object_change_generation.load(
          std::memory_order_relaxed
        );
        cached.reset();
        pending_semantic_fingerprint.clear();
        stable_ready_scans = 0;
        current = invalid_machine_record("changed");
        publish(current, true);
        next_full_scan = std::chrono::steady_clock::now() + options.interval;
        continue;
      }
      const auto object_generation_after_scan = object_change_generation.load(
        std::memory_order_relaxed
      );
      if (object_generation_after_scan != object_generation_before_scan) {
        observed_object_generation = object_generation_after_scan;
        if (cached && refresh_cached_selection(*cached)) {
          current = machine_record_for_cache(*cached);
          publish(current);
        } else {
          cached.reset();
          pending_semantic_fingerprint.clear();
          stable_ready_scans = 0;
          publish(invalid_machine_record("changed"), true);
        }
        next_full_scan = last_full_scan + event_full_scan_min_interval;
        std::this_thread::sleep_for(options.interval);
        continue;
      }

      monitored_window_value.store(
        reinterpret_cast<std::uintptr_t>(
          result.status == scan_status_e::unsupported_foreground ? nullptr :
                                                                   result.window
        ),
        std::memory_order_relaxed
      );
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
      if (ready_observation) {
        semantics = semantic_fingerprint(result, tentative_selection);
        if (cached && semantics == cached->semantic_fingerprint) {
          stable_ready_scans = 2;
          pending_semantic_fingerprint = semantics;
        } else if (semantics == pending_semantic_fingerprint) {
          ++stable_ready_scans;
        } else {
          pending_semantic_fingerprint = semantics;
          stable_ready_scans = 1;
        }
        if (stable_ready_scans >= 2) {
          result.status = scan_status_e::ok;
          result.selection = tentative_selection;
        }
      } else {
        pending_semantic_fingerprint.clear();
        stable_ready_scans = 0;
      }

      bool retained_cache_after_inconclusive_scan = false;
      if (result.status == scan_status_e::ok && result.selection.index) {
        auto replacement = cache_selection(result, result.selection, semantics);
        if (!replacement) {
          result.status = scan_status_e::traversal_incomplete;
          result.selection = {};
        } else {
          cached = std::move(replacement);
        }
      }

      if (result.status == scan_status_e::ok) {
        if (!result.selection.index) {
          // A complete, stable census authoritatively found no unique video.
          cached.reset();
        }
      } else if (
        result.status == scan_status_e::warming_up ||
        result.status == scan_status_e::traversal_incomplete ||
        result.status == scan_status_e::accessibility_unavailable
      ) {
        // A partial census cannot disprove an exact selected DOM object that still independently
        // authenticates. Keep it through transient provider/tree failures and retry the census.
        retained_cache_after_inconclusive_scan =
          cached && refresh_cached_selection(*cached);
        if (!retained_cache_after_inconclusive_scan) {
          cached.reset();
        }
      } else {
        // Foreground/window failures invalidate the old browser identity immediately.
        cached.reset();
      }

      current = retained_cache_after_inconclusive_scan ?
                  machine_record_for_cache(*cached) :
                  machine_record_for_result(result);
      pump_messages();
      const auto object_generation_before_publish = object_change_generation.load(
        std::memory_order_relaxed
      );
      if (object_generation_before_publish != object_generation_after_scan) {
        observed_object_generation = object_generation_before_publish;
        if (cached && refresh_cached_selection(*cached)) {
          current = machine_record_for_cache(*cached);
          publish(current);
        } else {
          cached.reset();
          pending_semantic_fingerprint.clear();
          stable_ready_scans = 0;
          publish(invalid_machine_record("changed"), true);
        }
        next_full_scan = last_full_scan + event_full_scan_min_interval;
        std::this_thread::sleep_for(options.interval);
        continue;
      }
      publish(current);
      const auto object_generation_after_publish = object_change_generation.load(
        std::memory_order_relaxed
      );
      const bool object_changed_during_scan =
        object_generation_after_publish != observed_object_generation;
      observed_object_generation = object_generation_after_publish;
      if (result.status == scan_status_e::ok) {
        next_full_scan = last_full_scan +
                         (object_changed_during_scan ? event_full_scan_min_interval :
                                                       periodic_full_scan_interval);
      } else if (ready_observation) {
        next_full_scan = last_full_scan + options.interval;
      } else if (retained_cache_after_inconclusive_scan) {
        next_full_scan = last_full_scan + event_full_scan_min_interval;
      } else if (result.status == scan_status_e::warming_up) {
        next_full_scan = last_full_scan + 1s;
      } else {
        next_full_scan = last_full_scan + periodic_full_scan_interval;
      }
      std::this_thread::sleep_for(options.interval);
    }
    monitored_window_value.store(0, std::memory_order_relaxed);
    return 0;
  }
}  // namespace

int main(int argc, char **argv) {
  const auto options = parse_options(argc, argv);
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
