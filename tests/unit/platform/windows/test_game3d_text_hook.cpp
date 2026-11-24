// SPDX-License-Identifier: GPL-3.0-only
#include "src/platform/windows/game3d_detection.h"
#include "src/platform/windows/game3d_text_hook.h"
#include "tools/text_hook/text_hook_shared.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <gtest/gtest.h>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <Windows.h>

namespace {
  namespace fs = std::filesystem;
  namespace hook = platf::game3d::text_hook;
  namespace shared = sunshine_text_hook;
  using namespace std::chrono_literals;

  TEST(Game3DTextHook, LoadSkipsARunningHelperAndRequiresItToReportIn) {
    int runs = 0;
    const auto yes = [] {
      return true;
    };
    const auto no = [] {
      return false;
    };
    // Already running: the loader is not started again.
    auto result = hook::detail::load_with(32u, yes, yes, [&] {
      ++runs;
      return 0u;
    });
    EXPECT_EQ(result.state, hook::load_state_e::loaded);
    EXPECT_EQ(runs, 0);
    // A build without the helper for this kind of game offers nothing to run, and says which.
    result = hook::detail::load_with(64u, no, no, [&] {
      ++runs;
      return 0u;
    });
    EXPECT_EQ(result.state, hook::load_state_e::unavailable);
    EXPECT_NE(result.detail.find("64-bit"), std::string::npos);
    EXPECT_EQ(runs, 0);
    // A loader failure is reported with its reason.
    result = hook::detail::load_with(32u, no, yes, [&] {
      ++runs;
      return 5u;
    });
    EXPECT_EQ(result.state, hook::load_state_e::failed);
    EXPECT_EQ(result.detail, hook::detail::injector_failure(5u));
    EXPECT_EQ(runs, 1);
    // The helper reports in shortly after the loader returns.
    int polls = 0;
    result = hook::detail::load_with(32u, [&] {
      return ++polls >= 3;
    }, yes, [&] {
      ++runs;
      return 0u;
    });
    EXPECT_EQ(result.state, hook::load_state_e::loaded);
    EXPECT_EQ(runs, 2);
    // A game closed before its approval is not a failure, and nothing is started for it.
    result = hook::detail::load_with(32u, no, yes, [&] {
      ++runs;
      return 0u;
    }, no);
    EXPECT_EQ(result.state, hook::load_state_e::not_running);
    EXPECT_EQ(runs, 2);
  }

  TEST(Game3DTextHook, WaitsForGraphicsReadinessAndObservesExitDuringStartup) {
    const auto ready_at = std::chrono::steady_clock::now() + 2200ms;
    const auto ready = [&] { return std::chrono::steady_clock::now() >= ready_at; };
    const auto present = [] { return true; };
    const auto started = [] { return 0u; };
    EXPECT_EQ(hook::detail::load_with(64u, ready, present, started).state, hook::load_state_e::loaded);

    bool running = true;
    const auto closed = hook::detail::load_with(64u, [] { return false; }, present, [&] {
      running = false;
      return 0u;
    }, [&] { return running; });
    EXPECT_EQ(closed.state, hook::load_state_e::not_running);
  }

  // The setup card shows what the capture path last reported for the very game it shows, and
  // says nothing once that is not this game.
  TEST(Game3DTextHook, TheProtectionReportIsForOneGameAndSaysWhyTheFallbackIsInForce) {
    using hook::fallback_e;
    using hook::protection_e;
    hook::report_protection(4242u, 77u, {protection_e::picture, fallback_e::unsure});
    EXPECT_EQ(hook::protection(4242u, 77u), (hook::protection_t {protection_e::picture, fallback_e::unsure}));
    EXPECT_EQ(hook::protection(4242u, 78u).protection, protection_e::unknown);
    EXPECT_EQ(hook::protection(4243u, 77u).protection, protection_e::unknown);
    hook::report_protection(4242u, 77u, {protection_e::game_text, fallback_e::none});
    EXPECT_EQ(hook::protection(4242u, 77u).protection, protection_e::game_text);

    EXPECT_TRUE(hook::describe({}).empty());
    EXPECT_NE(hook::describe({protection_e::game_text, fallback_e::none}).find("game's own text"), std::string::npos);
    std::set<std::string> sentences;
    for (const auto reason : {fallback_e::none, fallback_e::no_helper, fallback_e::no_text, fallback_e::unsure, fallback_e::not_shown, fallback_e::unavailable}) {
      const auto sentence = hook::describe({protection_e::picture, reason});
      EXPECT_NE(sentence.find("fallback"), std::string::npos) << sentence;
      sentences.insert(sentence);
    }
    // Every reason reads differently.
    EXPECT_EQ(sentences.size(), 6u);
  }

  TEST(Game3DTextHook, RunningNamesOneProcessByItsIdAndStartTime) {
    FILETIME created {}, unused {};
    ASSERT_TRUE(GetProcessTimes(GetCurrentProcess(), &created, &unused, &unused, &unused));
    const auto creation_time = (static_cast<std::uint64_t>(created.dwHighDateTime) << 32u) | created.dwLowDateTime;
    EXPECT_TRUE(hook::running(GetCurrentProcessId(), creation_time));
    // The same id given to another program since, and no process at all.
    EXPECT_FALSE(hook::running(GetCurrentProcessId(), creation_time + 1u));
    EXPECT_FALSE(hook::running(GetCurrentProcessId(), 0u));
    EXPECT_FALSE(hook::running(0u, creation_time));
  }

  TEST(Game3DTextHook, LoadRefusesAProcessThatIsNotTheApprovedOne) {
    // This test's own process, named with another creation time: a process id that has been
    // given to a different program since the approval. Nothing is started for it.
    const auto result = hook::load(GetCurrentProcessId(), 1u, fs::temp_directory_path());
    EXPECT_EQ(result.state, hook::load_state_e::not_running);
  }

  TEST(Game3DTextHook, ConsentNamesOneRunningProcessAndCanBeWithdrawn) {
    constexpr std::uint32_t process = 0x7ffffff1u;
    EXPECT_FALSE(hook::consented(process, 100u));
    hook::set_consent(process, 100u, true);
    EXPECT_TRUE(hook::consented(process, 100u));
    // Another process that reuses the id later is a different game.
    EXPECT_FALSE(hook::consented(process, 101u));
    EXPECT_FALSE(hook::consented(process + 1u, 100u));
    hook::set_consent(process, 101u, true);
    EXPECT_FALSE(hook::consented(process, 100u));
    EXPECT_TRUE(hook::consented(process, 101u));
    hook::set_consent(process, 101u, false);
    EXPECT_FALSE(hook::consented(process, 101u));
    // No process and no start time are never a consent.
    hook::set_consent(0u, 100u, true);
    hook::set_consent(process, 0u, true);
    EXPECT_FALSE(hook::consented(0u, 100u));
    EXPECT_FALSE(hook::consented(process, 0u));
  }

  /** What a helper publishes, made by the test under a process id no process has. */
  class fake_helper_t {
  public:
    explicit fake_helper_t(const std::uint32_t process_id):
        process_id_(process_id) {
      const auto name = std::wstring {shared::section_prefix} + std::to_wstring(process_id);
      section_ = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0u, sizeof(shared::snapshot_t), name.c_str());
      state = section_ ? static_cast<shared::snapshot_t *>(MapViewOfFile(section_, FILE_MAP_ALL_ACCESS, 0u, 0u, sizeof(shared::snapshot_t))) : nullptr;
      if (state) {
        *state = {};
        state->magic = shared::magic;
        state->version = shared::version;
        state->process_created = created;
      }
    }

    ~fake_helper_t() {
      for (auto *view : {static_cast<void *>(state), static_cast<void *>(pixels)}) {
        if (view) {
          UnmapViewOfFile(view);
        }
      }
      for (const HANDLE handle : {section_, coverage_}) {
        if (handle) {
          CloseHandle(handle);
        }
      }
    }

    /** A coverage section named for one size and holding `bytes` bytes. */
    void picture(const std::uint32_t width, const std::uint32_t height, const std::size_t bytes) {
      const auto name = std::wstring {shared::coverage_prefix} + std::to_wstring(process_id_) + L"-" + std::to_wstring(width) + L"x" + std::to_wstring(height);
      coverage_ = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0u, static_cast<DWORD>(bytes), name.c_str());
      pixels = coverage_ ? static_cast<std::uint8_t *>(MapViewOfFile(coverage_, FILE_MAP_ALL_ACCESS, 0u, 0u, bytes)) : nullptr;
      state->surface_width = width;
      state->surface_height = height;
    }

    static constexpr std::uint64_t created = 0x01dc0000'12345678ull;
    shared::snapshot_t *state = nullptr;
    std::uint8_t *pixels = nullptr;

  private:
    std::uint32_t process_id_;
    HANDLE section_ = nullptr;
    HANDLE coverage_ = nullptr;
  };

  // A helper that does not take what it follows for text publishes no rows and says so; the
  // reader hands that on, so the capture path falls back at once instead of waiting.
  TEST(Game3DTextHook, ReaderTellsWhenTheHelperIsNotSureOfItsText) {
    constexpr std::uint32_t process = 0x7ffffff4u;
    fake_helper_t helper(process);
    ASSERT_NE(helper.state, nullptr);
    helper.picture(64u, 32u, 64u * 32u * 4u);
    hook::reader_t reader;
    hook::text_t text;
    helper.state->source = shared::source_direct3d11;
    helper.state->sequence = 2u;
    ASSERT_TRUE(reader.read(process, fake_helper_t::created, text));
    EXPECT_FALSE(text.unsure);
    helper.state->source = shared::source_unsure;
    helper.state->sequence = 4u;
    ASSERT_TRUE(reader.read(process, fake_helper_t::created, text));
    EXPECT_TRUE(text.unsure);
    EXPECT_EQ(text.top, text.bottom);
    // Sure again, with text.
    helper.state->source = shared::source_direct3d11;
    helper.state->text_top = 4u;
    helper.state->text_bottom = 20u;
    helper.state->sequence = 6u;
    ASSERT_TRUE(reader.read(process, fake_helper_t::created, text));
    EXPECT_FALSE(text.unsure);
    EXPECT_EQ(text.bottom, 20u);
  }

  TEST(Game3DTextHook, ReaderTrustsOnlyACompleteStateOfTheSameProcess) {
    constexpr std::uint32_t process = 0x7ffffff2u;
    hook::reader_t reader;
    hook::text_t text;
    // No helper at all.
    EXPECT_FALSE(reader.read(process, fake_helper_t::created, text));
    EXPECT_FALSE(hook::loaded(process, fake_helper_t::created));

    fake_helper_t helper(process);
    ASSERT_NE(helper.state, nullptr);
    // A helper that has not seen the game draw yet is running, without text.
    ASSERT_TRUE(reader.read(process, fake_helper_t::created, text));
    EXPECT_EQ(text.top, text.bottom);
    EXPECT_EQ(text.pixels, nullptr);
    EXPECT_TRUE(hook::loaded(process, fake_helper_t::created));
    // The same section name under another process start is not this process's helper, nor is a
    // section of another contract.
    EXPECT_FALSE(reader.read(process, fake_helper_t::created + 1u, text));
    EXPECT_FALSE(reader.read(process, 0u, text));
    helper.state->version = shared::version + 1u;
    EXPECT_FALSE(reader.read(process, fake_helper_t::created, text));
    helper.state->version = shared::version;

    helper.picture(64u, 32u, 64u * 32u * 4u);
    ASSERT_NE(helper.pixels, nullptr);
    helper.pixels[(10u * 64u + 5u) * 4u + 3u] = 200u;
    helper.state->text_top = 8u;
    helper.state->text_bottom = 14u;
    helper.state->window = 0x1234u;
    helper.state->sequence = 6u;
    ASSERT_TRUE(reader.read(process, fake_helper_t::created, text));
    EXPECT_EQ(text.sequence, 6u);
    EXPECT_EQ(text.surface_width, 64u);
    EXPECT_EQ(text.surface_height, 32u);
    EXPECT_EQ(text.top, 8u);
    EXPECT_EQ(text.bottom, 14u);
    EXPECT_EQ(text.window, 0x1234u);
    ASSERT_NE(text.pixels, nullptr);
    EXPECT_EQ(text.pixels[(10u * 64u + 5u) * 4u + 3u], 200u);
    EXPECT_TRUE(reader.still_current(text));

    // Mid-publication the last complete state is returned, and a copy made now is not current.
    helper.state->sequence = 7u;
    helper.state->text_top = 0u;
    hook::text_t during;
    ASSERT_TRUE(reader.read(process, fake_helper_t::created, during));
    EXPECT_EQ(during.sequence, 6u);
    EXPECT_EQ(during.top, 8u);
    EXPECT_FALSE(reader.still_current(during));
    helper.state->text_top = 9u;
    helper.state->sequence = 8u;
    ASSERT_TRUE(reader.read(process, fake_helper_t::created, text));
    EXPECT_EQ(text.top, 9u);

    // Rows outside the picture, and a picture larger than any helper makes, are not trusted.
    helper.state->text_bottom = 33u;
    EXPECT_FALSE(reader.read(process, fake_helper_t::created, text));
    helper.state->text_bottom = 14u;
    helper.state->text_top = 15u;
    EXPECT_FALSE(reader.read(process, fake_helper_t::created, text));
    helper.state->text_top = 9u;
    helper.state->surface_width = shared::max_surface_extent + 1u;
    EXPECT_FALSE(reader.read(process, fake_helper_t::created, text));
  }

  TEST(Game3DTextHook, ReaderNeverReadsPastASectionSmallerThanItsNamePromises) {
    constexpr std::uint32_t process = 0x7ffffff3u;
    fake_helper_t helper(process);
    ASSERT_NE(helper.state, nullptr);
    // Named 2048 by 2048, holding one page.
    helper.picture(2048u, 2048u, 4096u);
    helper.state->text_top = 1u;
    helper.state->text_bottom = 2u;
    hook::reader_t reader;
    hook::text_t text;
    EXPECT_FALSE(reader.read(process, fake_helper_t::created, text));
    EXPECT_EQ(text.pixels, nullptr);
  }

#if defined(SUNSHINE_TEST_TEXT_HOOK_DIR) && defined(SUNSHINE_TEST_TEXT_HOOK_SAMPLE_DIR)
  // The built helper, loaded by the built loader into a sample program that draws one line of
  // text the way a game may. The sample's window is all but transparent and takes no input.
  struct sample_case_t {
    unsigned bits;
    const char *mode;
    std::uint32_t first_row, last_row;  ///< The line's glyphs lie between these rows.
    bool has_window;
  };

  void PrintTo(const sample_case_t &value, std::ostream *out) {
    *out << value.bits << "-bit " << value.mode;
  }

  class Game3DTextHookSample: public testing::TestWithParam<sample_case_t> {
  protected:
    void TearDown() override {
      if (process_.hProcess) {
        if (window_) {
          PostMessageW(window_, WM_CLOSE, 0u, 0);
        }
        if (WaitForSingleObject(process_.hProcess, 3000u) != WAIT_OBJECT_0) {
          TerminateProcess(process_.hProcess, 1u);
        }
        CloseHandle(process_.hThread);
        CloseHandle(process_.hProcess);
      }
    }

    static bool wait_until(const std::function<bool()> &condition, const std::chrono::milliseconds limit = 5000ms) {
      const auto end = std::chrono::steady_clock::now() + limit;
      while (std::chrono::steady_clock::now() < end) {
        if (condition()) {
          return true;
        }
        std::this_thread::sleep_for(30ms);
      }
      return condition();
    }

    bool start(const sample_case_t &sample) {
      const auto executable = fs::path {SUNSHINE_TEST_TEXT_HOOK_SAMPLE_DIR} / ("sunshine-text-hook-sample" + std::to_string(sample.bits) + ".exe");
      std::error_code error;
      if (!fs::is_regular_file(executable, error) || !fs::is_regular_file(fs::path {SUNSHINE_TEST_TEXT_HOOK_DIR} / ("SunshineTextHook" + std::to_string(sample.bits) + ".dll"), error)) {
        return false;
      }
      std::wstring command = L"\"" + executable.wstring() + L"\" " + std::wstring {sample.mode, sample.mode + std::strlen(sample.mode)};
      STARTUPINFOW startup {};
      startup.cb = sizeof(startup);
      if (!CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, FALSE, 0u, nullptr, nullptr, &startup, &process_)) {
        process_ = {};
        return false;
      }
      FILETIME created {}, exited {}, kernel {}, user {};
      GetProcessTimes(process_.hProcess, &created, &exited, &kernel, &user);
      created_ = (static_cast<std::uint64_t>(created.dwHighDateTime) << 32u) | created.dwLowDateTime;
      // Without a desktop to show a window on, there is nothing to test.
      return wait_until([&] {
        for (HWND window = nullptr; (window = FindWindowExW(nullptr, window, L"SunshineTextHookSample", nullptr)) != nullptr;) {
          DWORD owner = 0u;
          GetWindowThreadProcessId(window, &owner);
          if (owner == process_.dwProcessId && IsWindowVisible(window)) {
            window_ = window;
            return true;
          }
        }
        return false;
      });
    }

    void show_text(const bool show) const {
      PostMessageW(window_, WM_APP + 1u, show ? 1u : 0u, 0);
    }

    void show_second_line() const {
      PostMessageW(window_, WM_APP + 1u, 2u, 0);
    }

    bool read(hook::text_t &text) {
      return reader_.read(process_.dwProcessId, created_, text);
    }

    /** Count what the published picture holds: solid texels, those in the line's own colour, and texels outside the published rows. */
    bool count_picture(hook::text_t &text, unsigned &cores, unsigned &coloured, unsigned &outside) {
      std::vector<std::uint8_t> pixels;
      const bool copied = wait_until([&] {
        if (!read(text) || text.bottom <= text.top) {
          return false;
        }
        pixels.resize(static_cast<std::size_t>(text.surface_width) * text.surface_height * 4u);
        std::memcpy(pixels.data(), text.pixels, pixels.size());
        return reader_.still_current(text);
      });
      cores = coloured = outside = 0u;
      if (!copied) {
        return false;
      }
      for (std::uint32_t y = 0u; y < text.surface_height; ++y) {
        for (std::uint32_t x = 0u; x < text.surface_width; ++x) {
          const std::uint8_t *pixel = &pixels[(static_cast<std::size_t>(y) * text.surface_width + x) * 4u];
          if (pixel[3] == 0u) {
            continue;
          }
          outside += y < text.top || y >= text.bottom;
          if (pixel[3] >= 250u) {
            ++cores;
            // BGRA, premultiplied: RGB(240, 232, 224) at full coverage.
            coloured += std::abs(pixel[2] - 240) <= 6 && std::abs(pixel[1] - 232) <= 6 && std::abs(pixel[0] - 224) <= 6;
          }
        }
      }
      return true;
    }

    PROCESS_INFORMATION process_ {};
    std::uint64_t created_ = 0u;
    HWND window_ = nullptr;
    hook::reader_t reader_;
  };

  TEST_P(Game3DTextHookSample, PublishesTheTextWhileItIsDrawnAndNoLonger) {
    const auto sample = GetParam();
    if (!start(sample)) {
      GTEST_SKIP() << "the text helper sample for this bitness is not built, or no window can be shown";
    }
    const auto bundle = fs::path {SUNSHINE_TEST_TEXT_HOOK_DIR};
    // The process id under another start time names no process to load into.
    EXPECT_EQ(hook::load(process_.dwProcessId, created_ + 1u, bundle).state, hook::load_state_e::not_running);
    const auto loaded = hook::load(process_.dwProcessId, created_, bundle);
    ASSERT_EQ(loaded.state, hook::load_state_e::loaded) << loaded.detail;
    // A second load finds the helper running and starts nothing.
    EXPECT_EQ(hook::load(process_.dwProcessId, created_, bundle).state, hook::load_state_e::loaded);
    EXPECT_TRUE(hook::loaded(process_.dwProcessId, created_));
    EXPECT_FALSE(hook::loaded(process_.dwProcessId, created_ + 1u));

    hook::text_t text;
    const auto has_text = [&] {
      return read(text) && text.bottom > text.top;
    };
    const auto has_none = [&] {
      return read(text) && text.surface_width != 0u && text.bottom == text.top;
    };
    // Text drawn before the helper was there, on a bitmap that is not drawn again, is unknown
    // to it. Every mode draws its text anew when it changes.
    show_text(false);
    std::this_thread::sleep_for(300ms);
    show_text(true);
    ASSERT_TRUE(wait_until(has_text));

    EXPECT_EQ(text.surface_width, 640u);
    EXPECT_EQ(text.surface_height, 360u);
    EXPECT_EQ(text.window != 0u, sample.has_window);
    if (sample.has_window) {
      EXPECT_EQ(static_cast<std::uint32_t>(text.window), static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(window_)));
    }
    EXPECT_LE(text.top, sample.first_row);
    EXPECT_GE(text.bottom, sample.last_row);
    EXPECT_LE(text.bottom - text.top, 80u);
    // The picture holds the line in the sample's own colour, inside the published rows only.
    unsigned cores = 0u, coloured = 0u, outside = 0u;
    ASSERT_TRUE(count_picture(text, cores, coloured, outside));
    EXPECT_GT(cores, 500u);
    EXPECT_EQ(coloured, cores);
    EXPECT_EQ(outside, 0u);

    // A sample that draws the same line ten times a second publishes it once.
    const auto steady = text.sequence;
    std::this_thread::sleep_for(500ms);
    ASSERT_TRUE(has_text());
    EXPECT_EQ(text.sequence, steady);

    // The line goes when the sample stops drawing it, and comes back with it.
    show_text(false);
    EXPECT_TRUE(wait_until(has_none));
    show_text(true);
    EXPECT_TRUE(wait_until(has_text));
  }

  // A program that keeps its glyphs in one texture fills it as the text goes on, until more of
  // it is covered than of any text layer. It is the text's texture all the same: the line stays
  // published, and none of the other glyphs in the cache is taken for a part of it.
  class Game3DTextHookSampleGlyphCache: public Game3DTextHookSample {};

  TEST_P(Game3DTextHookSampleGlyphCache, KeepsTheLineWhenTheCacheHasFilled) {
    const auto sample = GetParam();
    if (!start(sample)) {
      GTEST_SKIP() << "the text helper sample for this bitness is not built, or no window can be shown";
    }
    const auto loaded = hook::load(process_.dwProcessId, created_, fs::path {SUNSHINE_TEST_TEXT_HOOK_DIR});
    ASSERT_EQ(loaded.state, hook::load_state_e::loaded) << loaded.detail;
    hook::text_t text;
    const auto has_text = [&] {
      return read(text) && text.bottom > text.top;
    };
    // The cache is begun anew when the text is shown again, so the helper sees it from its start.
    show_text(false);
    std::this_thread::sleep_for(300ms);
    show_text(true);
    ASSERT_TRUE(wait_until(has_text));
    // Seven more frames fill it, at ten frames a second.
    std::this_thread::sleep_for(2000ms);
    unsigned cores = 0u, coloured = 0u, outside = 0u;
    ASSERT_TRUE(count_picture(text, cores, coloured, outside));
    EXPECT_LE(text.top, sample.first_row);
    EXPECT_GE(text.bottom, sample.last_row);
    EXPECT_LE(text.bottom - text.top, 80u);
    EXPECT_GT(cores, 500u);
    EXPECT_EQ(coloured, cores);
    EXPECT_EQ(outside, 0u);
    // And it stays so while the full cache is drawn from.
    std::this_thread::sleep_for(500ms);
    EXPECT_TRUE(has_text());
  }

  INSTANTIATE_TEST_SUITE_P(
    Sources,
    Game3DTextHookSampleGlyphCache,
    testing::Values(
      sample_case_t {32u, "d3d11-atlas", 76u, 98u, true},
      sample_case_t {64u, "d3d11-atlas", 76u, 98u, true}
    )
  );

  // A game loads the parts of a figure's face, as small as glyphs, each into a new texture of
  // one size whenever it shows the figure. New textures alone are no reason to think of text,
  // however many of them look like it: the helper publishes none of it.
  class Game3DTextHookSampleParts: public Game3DTextHookSample {};

  TEST_P(Game3DTextHookSampleParts, NewTexturesOfOneSizeAreNotTextByThemselves) {
    const auto sample = GetParam();
    if (!start(sample)) {
      GTEST_SKIP() << "the text helper sample for this bitness is not built, or no window can be shown";
    }
    const auto loaded = hook::load(process_.dwProcessId, created_, fs::path {SUNSHINE_TEST_TEXT_HOOK_DIR});
    ASSERT_EQ(loaded.state, hook::load_state_e::loaded) << loaded.detail;
    hook::text_t text;
    show_text(false);
    std::this_thread::sleep_for(300ms);
    show_text(true);
    // Twenty frames, a new texture with every third.
    std::this_thread::sleep_for(2000ms);
    EXPECT_FALSE(read(text) && text.bottom > text.top);
    EXPECT_TRUE(hook::loaded(process_.dwProcessId, created_));
  }

  INSTANTIATE_TEST_SUITE_P(
    Sources,
    Game3DTextHookSampleParts,
    testing::Values(
      sample_case_t {32u, "d3d11-parts", 76u, 98u, true},
      sample_case_t {64u, "d3d11-parts", 76u, 98u, true}
    )
  );

  INSTANTIATE_TEST_SUITE_P(
    Sources, Game3DTextHookSample,
    testing::Values(
      sample_case_t {32u, "gdi-window", 76u, 98u, true},
      sample_case_t {32u, "gdi-buffer", 76u, 98u, true},
      sample_case_t {32u, "gdi-layer", 226u, 248u, true},
      sample_case_t {32u, "gdi-bits", 76u, 98u, true},
      sample_case_t {32u, "d3d11-layer", 76u, 98u, true},
      sample_case_t {32u, "d3d11-direct", 76u, 98u, true},
      sample_case_t {32u, "d3d11-own", 76u, 98u, true},
      sample_case_t {32u, "d3d11-cache", 76u, 98u, true},
      sample_case_t {32u, "d2d", 76u, 98u, false},
      sample_case_t {64u, "gdi-window", 76u, 98u, true},
      sample_case_t {64u, "gdi-buffer", 76u, 98u, true},
      sample_case_t {64u, "gdi-layer", 226u, 248u, true},
      sample_case_t {64u, "gdi-bits", 76u, 98u, true},
      sample_case_t {64u, "d3d11-layer", 76u, 98u, true},
      sample_case_t {64u, "d3d11-direct", 76u, 98u, true},
      sample_case_t {64u, "d3d11-own", 76u, 98u, true},
      sample_case_t {64u, "d3d11-cache", 76u, 98u, true},
      sample_case_t {64u, "d2d", 76u, 98u, false}
    )
  );

  // A Direct2D frame that repeats the one before is only compared with it. A frame that then
  // adds a line draws that line alone, and one that drops it again is redone from what the
  // helper kept of the lines that stayed, which the sample has by then drawn only for comparison.
  class Game3DTextHookDirect2DSample: public Game3DTextHookSample {};

  TEST_P(Game3DTextHookDirect2DSample, FollowsALineAddedToARepeatedFrameAndTakenAwayAgain) {
    const auto sample = GetParam();
    if (!start(sample)) {
      GTEST_SKIP() << "the text helper sample for this bitness is not built, or no window can be shown";
    }
    const auto loaded = hook::load(process_.dwProcessId, created_, fs::path {SUNSHINE_TEST_TEXT_HOOK_DIR});
    ASSERT_EQ(loaded.state, hook::load_state_e::loaded) << loaded.detail;
    hook::text_t text;
    const auto rows_end_between = [&](const std::uint32_t least, const std::uint32_t most) {
      return read(text) && text.bottom > text.top && text.bottom >= least && text.bottom <= most;
    };
    // The glyphs of the second line lie 80 rows below those of the first.
    const std::uint32_t one_line = sample.last_row + 40u, two_lines = sample.last_row + 80u;
    show_text(false);
    std::this_thread::sleep_for(300ms);
    show_text(true);
    ASSERT_TRUE(wait_until([&] {
      return rows_end_between(sample.last_row, one_line);
    }));
    // Let the frame repeat before it changes.
    std::this_thread::sleep_for(400ms);
    const auto cores_in = [&](const std::uint32_t first, const std::uint32_t end) {
      std::vector<std::uint8_t> pixels(static_cast<std::size_t>(text.surface_width) * text.surface_height * 4u);
      unsigned cores = 0u;
      EXPECT_TRUE(wait_until([&] {
        if (!read(text) || !text.pixels) {
          return false;
        }
        std::memcpy(pixels.data(), text.pixels, pixels.size());
        return reader_.still_current(text);
      }));
      for (std::uint32_t y = first; y < end && y < text.surface_height; ++y) {
        for (std::uint32_t x = 0u; x < text.surface_width; ++x) {
          cores += pixels[(static_cast<std::size_t>(y) * text.surface_width + x) * 4u + 3u] >= 250u;
        }
      }
      return cores;
    };
    show_second_line();
    ASSERT_TRUE(wait_until([&] {
      return rows_end_between(two_lines, two_lines + 40u);
    }));
    EXPECT_GT(cores_in(0u, one_line), 500u);
    EXPECT_GT(cores_in(one_line, text.surface_height), 500u);
    std::this_thread::sleep_for(400ms);
    show_text(true);
    ASSERT_TRUE(wait_until([&] {
      return rows_end_between(sample.last_row, one_line);
    }));
    EXPECT_GT(cores_in(0u, one_line), 500u);
    EXPECT_EQ(cores_in(one_line, text.surface_height), 0u);
  }

  INSTANTIATE_TEST_SUITE_P(
    Sources,
    Game3DTextHookDirect2DSample,
    testing::Values(sample_case_t {32u, "d2d", 76u, 98u, false}, sample_case_t {64u, "d2d", 76u, 98u, false})
  );

  TEST(Game3DTextHookSampleImports, AnExecutablesOwnImportsTellHowItDrawsText) {
    for (const unsigned bits : {32u, 64u}) {
      const auto executable = fs::path {SUNSHINE_TEST_TEXT_HOOK_SAMPLE_DIR} / ("sunshine-text-hook-sample" + std::to_string(bits) + ".exe");
      std::error_code error;
      if (!fs::is_regular_file(executable, error)) {
        continue;
      }
      // The sample links Direct2D and DirectWrite and calls TextOutW and GetGlyphOutlineW; the
      // loader does none of that.
      const auto sample = platf::game3d::detail::inspect_imports(executable);
      EXPECT_TRUE(sample.direct2d_text) << bits;
      EXPECT_TRUE(sample.gdi_text) << bits;
      EXPECT_TRUE(sample.gdi_glyphs) << bits;
      const auto loader = platf::game3d::detail::inspect_imports(fs::path {SUNSHINE_TEST_TEXT_HOOK_DIR} / ("sunshine-text-hook-inject" + std::to_string(bits) + ".exe"));
      EXPECT_FALSE(loader.direct2d_text) << bits;
      EXPECT_FALSE(loader.gdi_text) << bits;
      EXPECT_FALSE(loader.gdi_glyphs) << bits;
    }
  }
#endif

  TEST(Game3DTextHookImports, AFileThatIsNoExecutableImportsNothing) {
    const auto file = fs::temp_directory_path() / ("sunshine-text-hook-imports-" + std::to_string(GetCurrentProcessId()) + ".exe");
    {
      std::ofstream out(file, std::ios::binary);
      out << "MZ this is not an image, however it begins";
    }
    const auto imports = platf::game3d::detail::inspect_imports(file);
    EXPECT_FALSE(imports.direct2d_text);
    EXPECT_FALSE(imports.gdi_text);
    EXPECT_FALSE(imports.gdi_glyphs);
    std::error_code error;
    fs::remove(file, error);
    const auto missing = platf::game3d::detail::inspect_imports(file);
    EXPECT_FALSE(missing.direct2d_text);
    EXPECT_FALSE(missing.gdi_text);
  }
}  // namespace
