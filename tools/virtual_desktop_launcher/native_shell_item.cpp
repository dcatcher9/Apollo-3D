#ifndef NOMINMAX
  #define NOMINMAX
#endif
#include "native_shell_item.h"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <uiautomation.h>
#include <utility>

namespace desktop_launcher {
  namespace {
    constexpr std::uint64_t answer_lifetime_ms = 500;
    constexpr std::uint64_t refresh_after_ms = 200;

    struct process_handle_t {
      explicit process_handle_t(HANDLE value):
          value(value) {}

      ~process_handle_t() {
        if (value) {
          CloseHandle(value);
        }
      }

      HANDLE value;
    };

    struct desktop_handle_t {
      explicit desktop_handle_t(HDESK value):
          value(value) {}

      ~desktop_handle_t() {
        if (value) {
          CloseDesktop(value);
        }
      }

      HDESK value;
    };

    struct request_t {
      HWND root = nullptr;
      POINT point {};
      DWORD pid = 0, thread = 0;
      std::uint64_t tick = 0, generation = 0;
      std::shared_ptr<process_handle_t> process;

      bool matches(HWND window, POINT position, DWORD process_id, DWORD thread_id) const {
        return root == window && point.x == position.x && point.y == position.y &&
               pid == process_id && thread == thread_id;
      }

      bool fresh(std::uint64_t now, std::uint64_t limit = answer_lifetime_ms) const {
        return now >= tick && now - tick <= limit;
      }

      bool live() const {
        DWORD current_pid = 0;
        return process && WaitForSingleObject(process->value, 0) == WAIT_TIMEOUT &&
               GetWindowThreadProcessId(root, &current_pid) == thread && current_pid == pid;
      }
    };

    template<class Interface>
    struct release_com_t {
      void operator()(Interface *value) const {
        if (value) {
          value->Release();
        }
      }
    };
    template<class Interface>
    using com_ptr_t = std::unique_ptr<Interface, release_com_t<Interface>>;

    bool cached_boolean(IUIAutomationElement *element, PROPERTYID property) {
      VARIANT value {};
      const auto status = element->GetCachedPropertyValue(property, &value);
      const bool result = SUCCEEDED(status) && value.vt == VT_BOOL && value.boolVal == VARIANT_TRUE;
      VariantClear(&value);
      return result;
    }

    std::optional<bool> actionable_item(IUIAutomationElement *element, POINT point) {
      CONTROLTYPEID type = 0;
      if (FAILED(element->get_CachedControlType(&type))) {
        return std::nullopt;
      }
      // A pane, toolbar or empty list is never an invocation just because it accepts focus.
      if (type != UIA_ButtonControlTypeId && type != UIA_MenuItemControlTypeId && type != UIA_ListItemControlTypeId && type != UIA_HyperlinkControlTypeId) {
        return false;
      }
      BOOL enabled = FALSE, offscreen = TRUE;
      RECT bounds {};
      if (FAILED(element->get_CachedIsEnabled(&enabled)) || FAILED(element->get_CachedIsOffscreen(&offscreen)) || FAILED(element->get_CachedBoundingRectangle(&bounds))) {
        return std::nullopt;
      }
      // UIA may expose an inactive desktop's controls as offscreen; that is unavailable
      // evidence, not a verified negative answer about the control's invocation semantics.
      if (offscreen) {
        return std::nullopt;
      }
      if (!enabled || !PtInRect(&bounds, point)) {
        return false;
      }
      return cached_boolean(element, UIA_IsInvokePatternAvailablePropertyId) ||
             cached_boolean(element, UIA_IsSelectionItemPatternAvailablePropertyId);
    }

    std::optional<bool> query_item(IUIAutomation *automation, IUIAutomationTreeWalker *walker, IUIAutomationCacheRequest *cache, const request_t &request) {
      if (!request.live() || !request.fresh(GetTickCount64()) || !IsWindowVisible(request.root) || GetAncestor(WindowFromPoint(request.point), GA_ROOT) != request.root) {
        return std::nullopt;
      }
      IUIAutomationElement *raw = nullptr;
      if (FAILED(automation->ElementFromPointBuildCache(request.point, cache, &raw)) || !raw) {
        return std::nullopt;
      }
      com_ptr_t<IUIAutomationElement> current(raw);
      bool actionable = false, unavailable = false;
      // Walking only ancestors is bounded and cannot accidentally select another shell item.
      for (unsigned depth = 0; current && depth < 32; ++depth) {
        int pid = 0;
        UIA_HWND native_window = nullptr;
        if (!request.fresh(GetTickCount64()) || FAILED(current->get_CachedProcessId(&pid)) || static_cast<DWORD>(pid) != request.pid || FAILED(current->get_CachedNativeWindowHandle(&native_window))) {
          return std::nullopt;
        }
        const auto window = reinterpret_cast<HWND>(native_window);
        if (window == request.root) {
          // The root itself is not an item. Only proven descendants can qualify.
          if (!request.live() || !request.fresh(GetTickCount64()) || GetAncestor(WindowFromPoint(request.point), GA_ROOT) != request.root) {
            return std::nullopt;
          }
          return actionable ? std::optional {true} : unavailable ? std::nullopt :
                                                                   std::optional {false};
        }
        if (window && GetAncestor(window, GA_ROOT) != request.root) {
          return std::nullopt;
        }
        const auto item = actionable_item(current.get(), request.point);
        actionable = actionable || item.value_or(false);
        unavailable = unavailable || !item;
        IUIAutomationElement *parent = nullptr;
        if (FAILED(walker->GetParentElementBuildCache(current.get(), cache, &parent))) {
          return std::nullopt;
        }
        current.reset(parent);
      }
      return std::nullopt;
    }
  }  // namespace

  struct native_shell_item_query_t::state_t {
    std::mutex mutex;
    std::condition_variable changed;
    bool stopped = false;
    std::uint64_t generation = 0;
    std::shared_ptr<desktop_handle_t> desktop;
    std::optional<request_t> pending, active;
    std::optional<std::pair<request_t, bool>> completed;
  };

  native_shell_item_query_t::native_shell_item_query_t():
      state_(std::make_shared<state_t>()) {
    std::array<wchar_t, 256> name {};
    DWORD required = 0;
    const auto creator_desktop = GetThreadDesktop(GetCurrentThreadId());
    if (!GetUserObjectInformationW(creator_desktop, UOI_NAME, name.data(), sizeof(name), &required)) {
      state_->stopped = true;
      return;
    }
    const auto desktop = OpenDesktopW(name.data(), 0, FALSE, DESKTOP_READOBJECTS | DESKTOP_WRITEOBJECTS);
    if (!desktop) {
      state_->stopped = true;
      return;
    }
    state_->desktop = std::make_shared<desktop_handle_t>(desktop);
    try {
      std::thread(worker, state_).detach();
    } catch (const std::system_error &) {
      state_->stopped = true;
    }
  }

  native_shell_item_query_t::~native_shell_item_query_t() {
    {
      std::lock_guard lock(state_->mutex);
      state_->stopped = true;
      ++state_->generation;
      state_->pending.reset();
      state_->completed.reset();
    }
    state_->changed.notify_one();
  }

  void native_shell_item_query_t::request(HWND source_root, POINT point) {
    DWORD pid = 0;
    const auto thread = GetWindowThreadProcessId(source_root, &pid);
    if (!source_root || !thread || !pid || GetAncestor(source_root, GA_ROOT) != source_root) {
      return;
    }
    const auto now = GetTickCount64();
    std::lock_guard lock(state_->mutex);
    if (state_->stopped) {
      return;
    }
    if (state_->completed && state_->completed->first.matches(source_root, point, pid, thread) && state_->completed->first.fresh(now, refresh_after_ms)) {
      return;
    }
    for (const auto *request : {&state_->pending, &state_->active}) {
      if (*request && (*request)->matches(source_root, point, pid, thread) && (*request)->fresh(now)) {
        return;
      }
    }
    const auto process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, FALSE, pid);
    if (!process) {
      return;
    }
    request_t request {source_root, point, pid, thread, now, ++state_->generation, std::make_shared<process_handle_t>(process)};
    if (!request.live()) {
      return;
    }
    state_->pending = std::move(request);
    state_->changed.notify_one();
  }

  std::optional<bool> native_shell_item_query_t::result(HWND source_root, POINT point) const {
    DWORD pid = 0;
    const auto thread = GetWindowThreadProcessId(source_root, &pid);
    std::lock_guard lock(state_->mutex);
    if (state_->stopped || !state_->completed) {
      return std::nullopt;
    }
    const auto &[request, answer] = *state_->completed;
    if (!request.matches(source_root, point, pid, thread) || !request.fresh(GetTickCount64()) || !request.live()) {
      return std::nullopt;
    }
    return answer;
  }

  void native_shell_item_query_t::clear() {
    std::lock_guard lock(state_->mutex);
    ++state_->generation;
    state_->pending.reset();
    state_->active.reset();
    state_->completed.reset();
  }

  void native_shell_item_query_t::worker(std::shared_ptr<state_t> state) {
    const auto original_desktop = GetThreadDesktop(GetCurrentThreadId());
    if (!SetThreadDesktop(state->desktop->value)) {
      std::lock_guard lock(state->mutex);
      state->stopped = true;
      return;
    }

    struct restore_desktop_t {
      HDESK original;

      ~restore_desktop_t() {
        SetThreadDesktop(original);
      }
    } restore_desktop {original_desktop};

    const auto initialized = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(initialized)) {
      std::lock_guard lock(state->mutex);
      state->stopped = true;
      return;
    }

    struct apartment_t {
      ~apartment_t() {
        CoUninitialize();
      }
    } apartment;

    IUIAutomation *raw = nullptr;
    if (FAILED(CoCreateInstance(__uuidof(CUIAutomation8), nullptr, CLSCTX_INPROC_SERVER, __uuidof(IUIAutomation), reinterpret_cast<void **>(&raw)))) {
      std::lock_guard lock(state->mutex);
      state->stopped = true;
      return;
    }
    com_ptr_t<IUIAutomation> automation(raw);
    IUIAutomation2 *timeouts_raw = nullptr;
    if (SUCCEEDED(automation->QueryInterface(__uuidof(IUIAutomation2), reinterpret_cast<void **>(&timeouts_raw)))) {
      com_ptr_t<IUIAutomation2> timeouts(timeouts_raw);
      timeouts->put_ConnectionTimeout(150);
      timeouts->put_TransactionTimeout(150);
    }
    IUIAutomationTreeWalker *walker_raw = nullptr;
    IUIAutomationCacheRequest *cache_raw = nullptr;
    const auto walker_status = automation->get_RawViewWalker(&walker_raw);
    const auto cache_status = automation->CreateCacheRequest(&cache_raw);
    com_ptr_t<IUIAutomationTreeWalker> walker(walker_raw);
    com_ptr_t<IUIAutomationCacheRequest> cache(cache_raw);
    bool ready = SUCCEEDED(walker_status) && walker && SUCCEEDED(cache_status) && cache;
    if (ready) {
      for (const auto property : {UIA_ProcessIdPropertyId, UIA_NativeWindowHandlePropertyId, UIA_ControlTypePropertyId, UIA_IsEnabledPropertyId, UIA_IsOffscreenPropertyId, UIA_BoundingRectanglePropertyId, UIA_IsInvokePatternAvailablePropertyId, UIA_IsSelectionItemPatternAvailablePropertyId}) {
        ready = SUCCEEDED(cache->AddProperty(property)) && ready;
      }
    }
    if (!ready) {
      std::lock_guard lock(state->mutex);
      state->stopped = true;
      return;
    }
    while (true) {
      request_t request;
      {
        std::unique_lock lock(state->mutex);
        state->changed.wait(lock, [&] {
          return state->stopped || state->pending;
        });
        if (state->stopped) {
          return;
        }
        request = std::move(*state->pending);
        state->pending.reset();
        state->active = request;
      }
      const auto answer = query_item(automation.get(), walker.get(), cache.get(), request);
      {
        std::lock_guard lock(state->mutex);
        if (state->stopped) {
          return;
        }
        if (answer && state->generation == request.generation && request.fresh(GetTickCount64())) {
          state->completed = std::pair {request, *answer};
        }
        state->active.reset();
      }
    }
  }

  int native_shell_item_self_test() {
    const auto started = GetTickCount64();
    const auto original = GetThreadDesktop(GetCurrentThreadId());
    const auto name = L"SunshineShellItemTest-" + std::to_wstring(GetCurrentProcessId()) +
                      L"-" + std::to_wstring(started);
    const auto desktop = CreateDesktopW(name.c_str(), nullptr, nullptr, 0, GENERIC_ALL, nullptr);
    if (!desktop || !SetThreadDesktop(desktop)) {
      if (desktop) {
        CloseDesktop(desktop);
      }
      std::fwprintf(stderr, L"Native shell item UIA test unsupported: private desktop unavailable.\n");
      return 77;
    }
    WNDCLASSW cls {};
    cls.lpfnWndProc = DefWindowProcW;
    cls.hInstance = GetModuleHandleW(nullptr);
    cls.lpszClassName = L"SunshineShellItemFixture";
    const bool registered = RegisterClassW(&cls) != 0;
    const auto create = [&](int x) {
      return registered ? CreateWindowExW(0, cls.lpszClassName, L"Private UIA fixture", WS_POPUP | WS_BORDER, x, 30, 280, 200, nullptr, nullptr, cls.hInstance, nullptr) : nullptr;
    };
    const auto root = create(20), blank = create(340);
    const auto button = root ? CreateWindowExW(0, L"BUTTON", L"Action", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 20, 20, 140, 40, root, nullptr, cls.hInstance, nullptr) : nullptr;
    int result = 0;
    if (!root || !blank || !button) {
      result = 31;
    } else {
      // This desktop is never switched to; no test HWND appears on the user's display.
      ShowWindow(root, SW_SHOWNOACTIVATE);
      ShowWindow(blank, SW_SHOWNOACTIVATE);
      POINT button_point {60, 40}, blank_point {60, 40};
      ClientToScreen(root, &button_point);
      ClientToScreen(blank, &blank_point);
      native_shell_item_query_t query;
      const auto wait_for_answer = [&](HWND window, POINT point, std::uint64_t deadline) {
        query.clear();
        while (GetTickCount64() < deadline) {
          query.request(window, point);
          if (const auto answer = query.result(window, point)) {
            return answer;
          }
          MsgWaitForMultipleObjectsEx(0, nullptr, 10, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
          MSG message {};
          while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&message);
            DispatchMessageW(&message);
          }
        }
        return std::optional<bool> {};
      };
      const auto actionable = wait_for_answer(root, button_point, started + 2500);
      if (!actionable) {
        result = 77;
      } else if (!*actionable) {
        result = 32;
      } else {
        const auto empty = wait_for_answer(blank, blank_point, started + 4500);
        result = !empty ? 77 : *empty ? 33 :
                                        0;
      }
    }
    if (button) {
      DestroyWindow(button);
    }
    if (blank) {
      DestroyWindow(blank);
    }
    if (root) {
      DestroyWindow(root);
    }
    if (registered) {
      UnregisterClassW(cls.lpszClassName, cls.hInstance);
    }
    if (!SetThreadDesktop(original)) {
      result = 34;
    }
    CloseDesktop(desktop);
    std::fwprintf(stderr, L"Native shell item UIA test result=%d (%ls); no desktop switch or input injection.\n", result, result == 0 ? L"Button accepted; blank rejected" : result == 77 ? L"private-desktop UIA unsupported" :
                                                                                                                                                                                            L"failed");
    return result;
  }
}  // namespace desktop_launcher
