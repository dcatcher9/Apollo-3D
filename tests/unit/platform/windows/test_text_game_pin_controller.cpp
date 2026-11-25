// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file tests/unit/platform/windows/test_text_game_pin_controller.cpp
 * @brief The Game 3D text-game pin: which game is in front, when its helper's text is pinned, when
 *        the pin is released, and when the packed route switches.
 */
#include "src/platform/windows/text_game_pin_controller.h"
#include "tools/text_hook/text_hook_shared.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <d3d11.h>
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <vector>
#include <windows.h>
#include <wrl/client.h>

namespace {
  namespace foreground_window = platf::foreground_window;
  namespace hook = platf::game3d::text_hook;
  namespace shared = sunshine_text_hook;
  using controller_t = platf::dxgi::text_game_pin_controller_t;
  using Microsoft::WRL::ComPtr;
  using namespace std::chrono_literals;

  constexpr std::uint64_t created = 0x01dc0000'2468ace0ull;
  constexpr foreground_window::rect_t screen {0, 0, 1920, 1080};
  // The game's 640x480 picture, placed at (100, 50) of this display.
  constexpr foreground_window::rect_t client {100, 50, 740, 530};
  constexpr std::uint32_t surface_width = 640u, surface_height = 480u;
  constexpr std::uintptr_t game_window = 0x5150u;

  /** What a helper publishes, made by the test under a process id no process has. */
  class fake_helper_t {
  public:
    explicit fake_helper_t(const std::uint32_t process_id) {
      const auto state_name = std::wstring {shared::section_prefix} + std::to_wstring(process_id);
      state_section_ = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0u, sizeof(shared::snapshot_t), state_name.c_str());
      state = state_section_ ? static_cast<shared::snapshot_t *>(MapViewOfFile(state_section_, FILE_MAP_ALL_ACCESS, 0u, 0u, sizeof(shared::snapshot_t))) : nullptr;
      constexpr std::size_t bytes = std::size_t {surface_width} * surface_height * 4u;
      const auto coverage_name = std::wstring {shared::coverage_prefix} + std::to_wstring(process_id) + L"-" + std::to_wstring(surface_width) + L"x" + std::to_wstring(surface_height);
      coverage_section_ = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0u, static_cast<DWORD>(bytes), coverage_name.c_str());
      pixels_ = coverage_section_ ? static_cast<std::uint8_t *>(MapViewOfFile(coverage_section_, FILE_MAP_ALL_ACCESS, 0u, 0u, bytes)) : nullptr;
      if (state && pixels_) {
        *state = {};
        state->magic = shared::magic;
        state->version = shared::version;
        state->process_created = created;
        state->source = shared::source_direct3d11;
        state->surface_width = surface_width;
        state->surface_height = surface_height;
        // White text coverage on the rows that hold text.
        std::memset(pixels_ + std::size_t {400u} * surface_width * 4u, 0xff, std::size_t {40u} * surface_width * 4u);
      }
    }

    fake_helper_t(const fake_helper_t &) = delete;
    fake_helper_t &operator=(const fake_helper_t &) = delete;

    ~fake_helper_t() {
      for (auto *view : {static_cast<void *>(state), static_cast<void *>(pixels_)}) {
        if (view) {
          UnmapViewOfFile(view);
        }
      }
      for (const HANDLE handle : {state_section_, coverage_section_}) {
        if (handle) {
          CloseHandle(handle);
        }
      }
    }

    [[nodiscard]] bool ready() const {
      return state && pixels_;
    }

    /** Publish one complete state: text on surface rows [top, bottom), none when they are equal. */
    void publish(const std::uint32_t top, const std::uint32_t bottom, const std::uint32_t source = shared::source_direct3d11) {
      state->sequence = state->sequence + 1u;
      state->source = source;
      state->text_top = top;
      state->text_bottom = bottom;
      state->window = game_window;
      state->sequence = state->sequence + 1u;
    }

    shared::snapshot_t *state = nullptr;

  private:
    HANDLE state_section_ = nullptr;
    HANDLE coverage_section_ = nullptr;
    std::uint8_t *pixels_ = nullptr;
  };

  /** The foreground as foreground_window::sample() reports it. */
  foreground_window::observation_t in_front(
    const std::uint32_t process_id,
    const foreground_window::status_e status = foreground_window::status_e::ok,
    const foreground_window::rect_t &picture = client
  ) {
    foreground_window::observation_t observation;
    observation.status = status;
    observation.window = game_window;
    observation.process_id = process_id;
    observation.client_screen_rect = picture;
    return observation;
  }

  class TextGamePinController: public testing::Test {
  protected:
    void SetUp() override {
      constexpr D3D_FEATURE_LEVEL requested = D3D_FEATURE_LEVEL_11_0;
      if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0u, &requested, 1u, D3D11_SDK_VERSION, &device, nullptr, &context))) {
        GTEST_SKIP() << "the WARP device is unavailable";
      }
    }

    void TearDown() override {
      for (const auto process : approved) {
        hook::set_consent(process, created, false);
      }
    }

    /** A game its owner approved, under a process id no process has. */
    std::uint32_t approve(const std::uint32_t process) {
      hook::set_consent(process, created, true);
      approved.push_back(process);
      return process;
    }

    std::optional<controller_t::identity_t> observe(controller_t &pin, const controller_t::time_point_t now, const foreground_window::observation_t &foreground) {
      return pin.observe(now, screen, [&] {
        ++samples;
        return foreground;
      }, [&](std::uint32_t) {
        ++creation_queries;
        return created;
      });
    }

    bool prepare(controller_t &pin, std::array<UINT, 4u> &shape) {
      D3D11_TEXTURE2D_DESC colour {};
      colour.Width = 1920u;
      colour.Height = 1080u;
      return pin.prepare(device.Get(), context.Get(), colour, 203.0f / 80.0f, shape);
    }

    /** Draw as the converter does: the pin only for prepared text that the picture shows. */
    bool pinned(controller_t &pin) {
      std::array<UINT, 4u> shape {};
      return prepare(pin, shape) && pin.text_shown();
    }

    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    std::vector<std::uint32_t> approved;
    int samples = 0;
    int creation_queries = 0;
  };

  // Verdict words: light glyph cores and of them those shown, dark cores and those shown.
  constexpr std::array<std::uint32_t, 4u> shown {64u, 64u, 32u, 32u};
  constexpr std::array<std::uint32_t, 4u> unshown {64u, 0u, 0u, 0u};

  TEST_F(TextGamePinController, ObservesOnlyAnApprovedGameInFrontOnThisDisplay) {
    const auto now = std::chrono::steady_clock::now();
    const auto process = approve(0x7ffffe01u);
    controller_t pin;
    // Approved, and its picture lies on this display.
    const auto identity = observe(pin, now, in_front(process));
    ASSERT_TRUE(identity);
    EXPECT_EQ(*identity, (controller_t::identity_t {process, created, 0u}));
    // The foreground is sampled a few times a second, not at every conversion.
    EXPECT_EQ(observe(pin, now + 100ms, in_front(process)), identity);
    EXPECT_EQ(samples, 1);
    // A game its owner did not approve is not converted.
    EXPECT_FALSE(observe(pin, now + 300ms, in_front(0x7ffffe02u)));
    // Nor one whose picture leaves this display: its process is not even opened.
    creation_queries = 0;
    EXPECT_FALSE(observe(pin, now + 600ms, in_front(process, foreground_window::status_e::ok, {1800, 50, 2440, 530})));
    EXPECT_FALSE(observe(pin, now + 900ms, in_front(process, foreground_window::status_e::minimized)));
    EXPECT_EQ(creation_queries, 0);
    EXPECT_TRUE(observe(pin, now + 1200ms, in_front(process)));
    // While its window is dragged, the game stays the one in front, with its place unknown: its
    // text is left unpinned.
    fake_helper_t helper(process);
    ASSERT_TRUE(helper.ready());
    helper.publish(400u, 440u);
    EXPECT_EQ(observe(pin, now + 1500ms, in_front(process, foreground_window::status_e::interactive_move_size, {})), identity);
    std::array<UINT, 4u> shape {};
    EXPECT_FALSE(prepare(pin, shape));
    // A new session starts with no game known to be in front.
    pin.restart_session();
    samples = 0;
    EXPECT_TRUE(observe(pin, now + 1600ms, in_front(process)));
    EXPECT_EQ(samples, 1);
  }

  TEST_F(TextGamePinController, NoHelperTextMeansNoPin) {
    const auto process = approve(0x7ffffe03u);
    controller_t pin;
    ASSERT_TRUE(observe(pin, std::chrono::steady_clock::now(), in_front(process)));
    std::array<UINT, 4u> shape {};
    // No helper runs in the game.
    EXPECT_FALSE(prepare(pin, shape));
    EXPECT_FALSE(pin.text_shown());
    pin.note_protection(true, false, false, std::chrono::steady_clock::now());
    EXPECT_EQ(hook::protection(process, created), (hook::protection_t {hook::protection_e::picture, hook::fallback_e::no_helper}));

    // A helper that reports no text.
    fake_helper_t helper(process);
    ASSERT_TRUE(helper.ready());
    helper.publish(0u, 0u);
    EXPECT_FALSE(prepare(pin, shape));
    pin.note_protection(true, false, false, std::chrono::steady_clock::now());
    EXPECT_EQ(hook::protection(process, created).fallback, hook::fallback_e::no_text);

    // A helper not sure of what it follows.
    helper.publish(0u, 0u, shared::source_unsure);
    EXPECT_FALSE(prepare(pin, shape));
    pin.note_protection(true, false, false, std::chrono::steady_clock::now());
    EXPECT_EQ(hook::protection(process, created).fallback, hook::fallback_e::unsure);

    // A verdict that the picture shows text pins nothing while there is no text to prepare.
    pin.take_verdict(shown, std::chrono::steady_clock::now());
    EXPECT_FALSE(pinned(pin));
    // The picture-based search covers this game.
    EXPECT_TRUE(pin.picture_search_covers(process, std::chrono::steady_clock::now()));
  }

  TEST_F(TextGamePinController, HelperTextThePictureShowsIsPinned) {
    const auto process = approve(0x7ffffe04u);
    fake_helper_t helper(process);
    ASSERT_TRUE(helper.ready());
    helper.publish(400u, 440u);
    controller_t pin;
    ASSERT_TRUE(observe(pin, std::chrono::steady_clock::now(), in_front(process)));
    std::array<UINT, 4u> shape {};
    ASSERT_TRUE(prepare(pin, shape));
    // The pinned band covers the text's source rows 450 to 489, in rows of 16, inside the game's
    // picture of the 1920x1080 source.
    EXPECT_EQ(shape[0], 1920u);
    EXPECT_EQ(shape[1], 1080u);
    EXPECT_EQ(shape[2] % 16u, 0u);
    EXPECT_GE(shape[2], 48u);
    EXPECT_LE(shape[2], 450u);
    EXPECT_GE(shape[2] + shape[3], 489u);
    EXPECT_LE(shape[2] + shape[3], 544u);
    // Until the picture is found to show the text, it is only observed.
    EXPECT_FALSE(pin.text_shown());
    pin.note_protection(true, true, false, std::chrono::steady_clock::now());
    EXPECT_EQ(hook::protection(process, created), (hook::protection_t {hook::protection_e::picture, hook::fallback_e::not_shown}));
    EXPECT_TRUE(pin.picture_search_covers(process, std::chrono::steady_clock::now()));

    // The picture shows it: the text is pinned, and the helper serves the game.
    pin.take_verdict(shown, std::chrono::steady_clock::now());
    EXPECT_TRUE(pinned(pin));
    const auto now = std::chrono::steady_clock::now();
    EXPECT_TRUE(pin.helper_serves(now));
    EXPECT_FALSE(pin.picture_search_covers(process, now));
    pin.note_protection(true, true, true, now);
    EXPECT_EQ(hook::protection(process, created), (hook::protection_t {hook::protection_e::game_text, hook::fallback_e::none}));
  }

  TEST_F(TextGamePinController, ThePictureShowsTheTextOnlyWithEnoughOfEachKind) {
    controller_t pin;
    const auto now = std::chrono::steady_clock::now();
    // Fewer than 16 cores in all say nothing.
    pin.take_verdict({10u, 10u, 5u, 5u}, now);
    EXPECT_FALSE(pin.text_shown());
    // Each kind with 16 cores or more must show a sixteenth of them.
    pin.take_verdict({64u, 3u, 0u, 0u}, now);
    EXPECT_FALSE(pin.text_shown());
    pin.take_verdict({64u, 4u, 0u, 0u}, now);
    EXPECT_TRUE(pin.text_shown());
    pin.take_verdict({64u, 4u, 32u, 1u}, now);
    EXPECT_FALSE(pin.text_shown());
    // A kind with fewer than 16 cores is not judged.
    pin.take_verdict({64u, 4u, 15u, 0u}, now);
    EXPECT_TRUE(pin.text_shown());
  }

  TEST_F(TextGamePinController, ThePinIsReleasedWhenTheTextGoes) {
    const auto process = approve(0x7ffffe05u);
    fake_helper_t helper(process);
    ASSERT_TRUE(helper.ready());
    helper.publish(400u, 440u);
    controller_t pin;
    ASSERT_TRUE(observe(pin, std::chrono::steady_clock::now(), in_front(process)));
    std::array<UINT, 4u> shape {};
    ASSERT_TRUE(prepare(pin, shape));
    pin.take_verdict(shown, std::chrono::steady_clock::now());
    ASSERT_TRUE(pinned(pin));

    // The helper reports no text any more: nothing is pinned.
    helper.publish(0u, 0u);
    const auto before = std::chrono::steady_clock::now();
    EXPECT_FALSE(prepare(pin, shape));
    const auto after = std::chrono::steady_clock::now();
    EXPECT_FALSE(pin.text_shown());
    EXPECT_FALSE(pinned(pin));
    // A pause between two lines keeps the helper serving the game for three seconds of such
    // findings; past that the picture-based search covers it again.
    EXPECT_TRUE(pin.helper_serves(before + 3s));
    EXPECT_FALSE(pin.helper_serves(after + 3s + 1ms));
    EXPECT_FALSE(pin.picture_search_covers(process, before + 2s));
    EXPECT_TRUE(pin.picture_search_covers(process, after + 4s));
    pin.note_protection(true, false, false, before + 2s);
    EXPECT_EQ(hook::protection(process, created).protection, hook::protection_e::game_text);
    pin.note_protection(true, false, false, after + 4s);
    EXPECT_EQ(hook::protection(process, created), (hook::protection_t {hook::protection_e::picture, hook::fallback_e::no_text}));

    // The text is back, and the picture shows it again.
    helper.publish(400u, 440u);
    pin.take_verdict(shown, std::chrono::steady_clock::now());
    EXPECT_TRUE(pinned(pin));
    // The picture no longer shows it: released at once.
    pin.take_verdict(unshown, std::chrono::steady_clock::now());
    EXPECT_FALSE(pinned(pin));
    EXPECT_TRUE(pin.helper_serves(std::chrono::steady_clock::now()));

    // A helper that is not sure of its text asks for the picture-based search without delay.
    pin.take_verdict(shown, std::chrono::steady_clock::now());
    ASSERT_TRUE(pinned(pin));
    helper.publish(0u, 0u, shared::source_unsure);
    EXPECT_FALSE(pinned(pin));
    EXPECT_FALSE(pin.helper_serves(std::chrono::steady_clock::now()));
  }

  TEST_F(TextGamePinController, AnotherGameInFrontLearnsAgain) {
    const auto first = approve(0x7ffffe06u);
    const auto second = approve(0x7ffffe07u);
    fake_helper_t helper(first);
    ASSERT_TRUE(helper.ready());
    helper.publish(400u, 440u);
    controller_t pin;
    const auto now = std::chrono::steady_clock::now();
    ASSERT_TRUE(observe(pin, now, in_front(first)));
    std::array<UINT, 4u> shape {};
    ASSERT_TRUE(prepare(pin, shape));
    pin.take_verdict(shown, now);
    ASSERT_TRUE(pinned(pin));
    ASSERT_TRUE(pin.helper_serves(now));
    // Nothing learned about one game is another's.
    ASSERT_TRUE(observe(pin, now + 300ms, in_front(second)));
    EXPECT_FALSE(pin.text_shown());
    EXPECT_FALSE(pin.helper_serves(now + 300ms));
    EXPECT_TRUE(pin.picture_search_covers(second, now + 300ms));
    EXPECT_FALSE(pin.picture_search_covers(first, now + 300ms));
    // reset() forgets the helper as well.
    ASSERT_TRUE(observe(pin, now + 600ms, in_front(first)));
    pin.take_verdict(shown, now + 600ms);
    ASSERT_TRUE(pin.helper_serves(now + 600ms));
    pin.reset();
    EXPECT_FALSE(pin.text_shown());
    EXPECT_FALSE(pin.helper_serves(now + 600ms));
  }

  TEST_F(TextGamePinController, TheRouteSwitchesOncePerChange) {
    controller_t pin;
    // The first frame of a session takes a route, which is a switch from none.
    EXPECT_TRUE(pin.switch_route(false));
    EXPECT_FALSE(pin.switch_route(false));
    // A text game comes in front: the Host AI path.
    EXPECT_TRUE(pin.switch_route(true));
    EXPECT_FALSE(pin.switch_route(true));
    // It goes: the export or flat path again.
    EXPECT_TRUE(pin.switch_route(false));
    // A new session starts without a route, whichever one it takes first.
    pin.restart_session();
    EXPECT_TRUE(pin.switch_route(false));
    pin.restart_session();
    EXPECT_TRUE(pin.switch_route(true));
  }
}  // namespace
