# SPDX-License-Identifier: GPL-3.0-only
# reshade_clock_domains, in script mode: every clock read and thread-pool
# submission in the add-on's sources has a stated domain (README, Test
# builds). Game time is read only through sunshine_game3d::clock::game_now()
# (game3d_clock.h), which the test add-on can drive, and every pool callback
# starts through sunshine_game3d::pool::submit (game3d_pool_submit.h), which
# the test add-on counts until it returns; every other clock read and the
# one pool submission are listed below, counted per file with their domain
# and reason. A new site, or one that moved, fails here until it names its
# domain: game time through game_now(), pool work through pool::submit,
# anything else as a new entry.
#
# Scanned: every .cpp, .h and .inl under SOURCE_DIR except the fixtures
# (test_*), experiments/ and the replay tools, whose clocks are their own; and
# the test add-on's own sources (TEST_ADDON_SOURCES, CMakeLists.txt's
# sunshine_test_addon_sources), which run in the game process as add-on code
# although their names start with test_. Comments, string literals and
# character literals are not code.
#
# The fixtures that play game time (converted_fixtures below, plan C3 item
# 10) are scanned for the real tick only: they read game time through the
# fixture clock (test_game3d_clock.h: clock.now(), hitch(), the frame
# boundary) and wall time only through game_clock::wall_now(), so a
# GetTickCount64 in one of them fails here as in the add-on. Their threads,
# waits and other clocks (watchdogs, hang bounds) are the fixtures' own.
#
#   cmake -DSOURCE_DIR=<tools/reshade> -DTEST_ADDON_SOURCES=<a.cpp|b.cpp|...> -P check_clock_domains.cmake
cmake_minimum_required(VERSION 3.24)

if(NOT SOURCE_DIR)
  message(FATAL_ERROR "SOURCE_DIR is required")
endif()
if(NOT TEST_ADDON_SOURCES)
  message(FATAL_ERROR "TEST_ADDON_SOURCES is required: the test add-on's own sources, joined with | "
    "(CMakeLists.txt, sunshine_test_addon_sources)")
endif()
string(REPLACE "|" ";" test_addon_sources "${TEST_ADDON_SOURCES}")

# The kinds of site the check counts.
#   tick:  GetTickCount64, the real tick behind game_now();
#   os:    the operating system's other clocks, and the C runtime's (in
#          call form: time and clock are also plain words);
#   qpc:   QueryPerformanceCounter;
#   chrono: the std::chrono clocks, named or read (an alias's clock::now());
#   pool:  a thread-pool submission, a new thread, a timer, or a wait or
#          work object whose callback runs on the pool.
set(os_clocks GetTickCount timeGetTime GetSystemTime GetLocalTime GetSystemTimeAsFileTime GetSystemTimePreciseAsFileTime
  QueryInterruptTime QueryInterruptTimePrecise QueryUnbiasedInterruptTime QueryUnbiasedInterruptTimePrecise
  NtQuerySystemTime __rdtsc _rdtsc)
set(std_clocks steady_clock system_clock high_resolution_clock utc_clock tai_clock gps_clock file_clock)
set(pool_calls TrySubmitThreadpoolCallback SubmitThreadpoolWork CreateThreadpoolWork QueueUserWorkItem CreateThread
  CreateRemoteThread CreateRemoteThreadEx _beginthread _beginthreadex CreateThreadpoolTimer SetThreadpoolTimer
  CreateTimerQueueTimer CreateWaitableTimer CreateWaitableTimerW CreateWaitableTimerExW CreateThreadpoolWait
  CreateThreadpoolIo RegisterWaitForSingleObject SetThreadpoolWait SetThreadpoolWaitEx StartThreadpoolIo
  BindIoCompletionCallback CreateTimerQueue SetTimer SetCoalescableTimer QueueUserAPC QueueUserAPC2)
set(pool_std thread jthread async)

# The allowed sites: "<file>|<kind>|<count>|<domain>|<reason>". Domains:
#   game: the game clock itself;
#   wall: real elapsed time that must not follow the test clock (a hang
#         bound, a lifetime another process sees, a CPU time slice);
#   perf: a measurement only (durations in logs and statistics, the wire
#         slot's timestamp the host reads);
#   file: file times;
#   pool: work handed to the thread pool.
set(allowed
  "game3d_clock.h|tick|3|game|game_now() and the test clock's engage read the real tick"
  "depth_capture/pool.inl|tick|3|wall|release_device: the real 10 s bound of its fence wait"
  "game3d_diagnostic_metadata.cpp|tick|2|wall|the diagnostics library, built outside the views, cannot read the test clock: its armed and snapshot ticks reach only Dump 3D metadata"
  "game3d_shader_cache.h|os|1|file|shader-cache pruning ages files by their write time (GetSystemTimeAsFileTime)"
  "depth_capture/lock_stats.inl|qpc|1|perf|capture lock wait and hold durations"
  "depth_capture/report_storage_release.inl|qpc|2|perf|the storage release's duration in its log line"
  "depth_display.inl|qpc|2|perf|the display release's duration in its log line"
  "exporter/present_diagnostics.inl|qpc|1|perf|the Present's CPU sections in the timing line"
  "exporter_internal.h|qpc|4|perf|the Present's QPC, its timing marks and the wire slot's timestamp the host reads"
  "game3d_debug_dump.cpp|qpc|1|perf|Dump 3D's capture timestamp"
  "game3d_slow_step.h|qpc|2|perf|the slow step's duration and start"
  "exporter/export_ring.inl|chrono|1|wall|the export ring's inactive_since: ring lifetimes the receiver process sees"
  "exporter/export_ring_class.inl|chrono|8|wall|the export ring's 500 ms idle reuse, retry and cached-ring idle: lifetimes the receiver process sees"
  "exporter_internal.h|chrono|2|wall|the export ring's 2 s retry stamps"
  "game3d_renderer.cpp|chrono|6|wall|the background build's 4 ms present-thread time slice, and its build_ms"
  "game3d_ui_layer.cpp|chrono|1|perf|the skip gap's microsecond clock, for the timing line's maximum"
  "test_depth_addon_exports.cpp|chrono|2|perf|SunshineDepthTestSnapshot's timed copies of the depth selection snapshot, a measurement"
  "test_addon_lifetime_exports.cpp|chrono|2|wall|SunshineGame3DTestQuiesce's bound: real time, so that a barrier that is never reached cannot hang"
  "game3d_pool_submit.h|pool|2|pool|pool::submit, shipping and test branch: every pool callback of the add-on starts here, by family, so the test add-on counts it until it returns"
  "test_game3d_clock.h|tick|2|wall|the fixture clock itself: game_clock::wall_now(), and now() until the clock engages"
  "test_depth3d_runtime_d3d12.cpp|tick|1|wall|the crash dump's file name, unique per crash")

# The fixtures that play game time on the fixture clock (README, Test
# builds): every one a fixture converts joins this list.
set(converted_fixtures
  test_game3d_clock.h test_depth3d_runtime_d3d12.cpp test_game3d_continuity.h
  test_game3d_session_timeline_runtime.cpp test_game3d_api_contract_runtime.cpp test_game3d_native_provider_runtime.cpp
  test_streamline_direct_runtime.cpp test_ngx_depth_runtime.cpp test_depth_probe_round_runtime.cpp
  test_adaptive_raw_runtime.cpp test_depth_alternating_runtime.cpp test_game3d_native_depth_runtime.cpp
  test_game3d_native_observation.h test_raw_runtime_fixture.h
  test_streamline_camera_probe.cpp test_upscaler_call_trace.cpp test_game3d_ui_capture.cpp
  test_diagnostic_texture_capture.cpp test_game3d_native_runtime.cpp test_game3d_native_runtime_d3d11.cpp
  test_game3d_debug_dump_runtime.h test_game3d_seam_characterization.cpp test_streamline_depth_capture.cpp
  test_streamline_depth_capture_fixture.h test_game3d_renderer_build.h test_game3d_startup_measure.h
  test_game3d_view_quiesce.cpp)

# Code with its comments and its string and character literals blanked (raw
# strings without a delimiter included), lines kept as lines. Each pattern is
# unrolled (a run of plain characters, then each escape, star run or
# parenthesis run with the plain run after it), so that CMake's recursive
# regex engine recurses once per escape or run, not once per character: a
# long literal or comment cannot overflow its stack (the per-character form
# crashed, with no output, on a literal of 25,000 characters).
function(clock_code text out)
  string(REPLACE ";" "<semicolon>" text "${text}")
  string(REPLACE "\r" "" text "${text}")
  string(REGEX REPLACE
    "//[^\n]*|/\\*[^*]*\\*+([^*/][^*]*\\*+)*/|R\"\\([^)]*\\)+([^)\"][^)]*\\)+)*\"|\"[^\"\\\\\n]*(\\\\.[^\"\\\\\n]*)*\"|'[^'\\\\\n]*(\\\\.[^'\\\\\n]*)*'"
    " " text "${text}")
  set(${out} "${text}" PARENT_SCOPE)
endfunction()

# The sites of one kind in code: a list of the matched tokens.
function(clock_sites code kind out)
  set(found "")
  if(kind STREQUAL "tick")
    string(REGEX MATCHALL "[A-Za-z0-9_]*GetTickCount64[A-Za-z0-9_]*" names "${code}")
    foreach(name IN LISTS names)
      if(name STREQUAL "GetTickCount64")
        list(APPEND found "${name}")
      endif()
    endforeach()
  elseif(kind STREQUAL "os")
    string(REGEX MATCHALL "[A-Za-z0-9_]*(GetTickCount|Time|_rdtsc)[A-Za-z0-9_]*" names "${code}")
    foreach(name IN LISTS names)
      if(name IN_LIST os_clocks)
        list(APPEND found "${name}")
      endif()
    endforeach()
    # time(, std::time(, ::clock(, _time64(, _ftime(...: a call, not a member
    # (. or ->) or another namespace's function (the blanked ; reads n>).
    string(REGEX MATCHALL
      "(^|[^A-Za-z0-9_.:>]|[^-]>)(std[ \t]*::[ \t]*|::[ \t]*)?(time|clock|_time32|_time64|_ftime|_ftime32|_ftime64|_ftime_s|_ftime32_s|_ftime64_s|timespec_get|_timespec32_get|_timespec64_get)[ \t]*\\("
      calls "${code}")
    foreach(call IN LISTS calls)
      string(REGEX MATCH "(std[ \t]*::[ \t]*|::[ \t]*)?[A-Za-z0-9_]+[ \t]*\\($" call "${call}")
      list(APPEND found "${call}")
    endforeach()
  elseif(kind STREQUAL "qpc")
    string(REGEX MATCHALL "[A-Za-z0-9_]*QueryPerformanceCounter[A-Za-z0-9_]*" names "${code}")
    foreach(name IN LISTS names)
      if(name STREQUAL "QueryPerformanceCounter")
        list(APPEND found "${name}")
      endif()
    endforeach()
  elseif(kind STREQUAL "chrono")
    string(REGEX MATCHALL "[A-Za-z0-9_]*clock[A-Za-z0-9_]*([ \t]*::[ \t]*now[ \t]*\\()?" names "${code}")
    foreach(name IN LISTS names)
      string(REGEX REPLACE "[ \t]*::.*$" "" base "${name}")
      if(base IN_LIST std_clocks OR name MATCHES "::[ \t]*now[ \t]*\\($")
        list(APPEND found "${name}")
      endif()
    endforeach()
  elseif(kind STREQUAL "pool")
    string(REGEX MATCHALL "(std[ \t]*::[ \t]*)?[A-Za-z0-9_]*(Thread|Timer|WorkItem|Wait|APC|Completion|thread|async)[A-Za-z0-9_]*"
      names "${code}")
    foreach(name IN LISTS names)
      string(REGEX REPLACE "^std[ \t]*::[ \t]*" "" bare "${name}")
      if(bare IN_LIST pool_calls OR (NOT bare STREQUAL name AND bare IN_LIST pool_std))
        list(APPEND found "${name}")
      endif()
    endforeach()
  else()
    message(FATAL_ERROR "unknown kind ${kind}")
  endif()
  set(${out} "${found}" PARENT_SCOPE)
endfunction()

set(kinds tick os qpc chrono pool)

# The check must itself find what it exists to find, also after a literal or
# comment far longer than any in the add-on.
string(REPEAT "x" 40000 long_text)
foreach(sample IN ITEMS
    "tick|  const auto now = GetTickCount64();\n"
    "tick|  auto read = &GetTickCount64;\n"
    "tick|  return (GetTickCount64)();\n"
    "tick|  a = GetTickCount64() + GetTickCount64();|2"
    "os|  const DWORD now = GetTickCount();\n"
    "os|  GetSystemTimeAsFileTime(&now);\n"
    "os|  const auto now = time(nullptr);\n"
    "os|  stamp = std::time(nullptr); used = clock();|2"
    "os|  _ftime64(&now); x = ::time (nullptr);|2"
    "qpc|  QueryPerformanceCounter(&start);\n"
    "chrono|  const auto now = std::chrono::steady_clock::now();\n"
    "chrono|  using c = std::chrono::steady_clock; auto t = c::now();\n"
    "chrono|  const auto t = clock::now();\n"
    "chrono|  std::chrono::system_clock::time_point at{};\n"
    "pool|  TrySubmitThreadpoolCallback(task, context, nullptr);\n"
    "pool|  std::thread worker(run);\n"
    "pool|  auto f = std::async(std::launch::async, run);\n"
    "pool|  RegisterWaitForSingleObject(&wait, event, run, nullptr, INFINITE, WT_EXECUTEONLYONCE);\n"
    "pool|  SetThreadpoolWait(wait, event, nullptr);\n"
    "pool|  StartThreadpoolIo(io);\n"
    "pool|  BindIoCompletionCallback(file, done, 0);\n"
    "pool|  const auto queue = CreateTimerQueue();\n"
    "pool|  SetTimer(window, 1, 100, nullptr);\n"
    "pool|  QueueUserAPC(run, thread, 0);\n"
    "pool|  const char *text = \"${long_text}\";\n  TrySubmitThreadpoolCallback(task, context, nullptr);\n"
    "tick|  /* ${long_text}\n ${long_text} */\n  const auto now = GetTickCount64();\n"
    "chrono|  const auto shader = R\"(${long_text})\";\n  const auto now = std::chrono::steady_clock::now();\n"
    "qpc|  const char quote = '\\\"', escaped[] = \"\\\\\\\" ${long_text}\";\n  QueryPerformanceCounter(&start);\n")
  string(REPLACE ";" "<semicolon>" sample "${sample}")
  string(REPLACE "|" ";" parts "${sample}")
  list(GET parts 0 kind)
  list(GET parts 1 text)
  set(want 1)
  list(LENGTH parts n)
  if(n GREATER 2)
    list(GET parts 2 want)
  endif()
  clock_code("${text}" code)
  clock_sites("${code}" ${kind} found)
  list(LENGTH found got)
  if(NOT got EQUAL want)
    message(FATAL_ERROR "The clock domain check found ${got} ${kind} sites, not ${want}, in:\n${text}")
  endif()
endforeach()
set(sample "// GetTickCount64() ticks, steady_clock::now(), QueryPerformanceCounter, TrySubmitThreadpoolCallback\n/* GetTickCount64() and\n std::thread */\nconst char *a = \"GetTickCount64 ms; steady_clock\";\nconstexpr char b[] = R\"(QueryPerformanceCounter(\"x\"); GetTickCount())\";\nconst char c = '\"';\nconst auto t = sunshine_game3d::clock::game_now() + lock_clock() + scope_clock + MyGetTickCount64Helper();\nstd::this_thread::yield();\nframe.time(); frame->clock(); runtime(now); const auto time_ms = mktime(&t) + localtime(&s);\nstd::chrono::time_point<C> at; WaitForSingleObject(event, INFINITE); SwitchToThread();\n")
clock_code("${sample}" code)
foreach(kind IN LISTS kinds)
  clock_sites("${code}" ${kind} found)
  if(found)
    message(FATAL_ERROR "The clock domain check counted ${kind} sites in comments, literals or other names: ${found}")
  endif()
endforeach()

# The add-on's sources.
file(GLOB_RECURSE sources RELATIVE "${SOURCE_DIR}" "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.h" "${SOURCE_DIR}/*.inl")
set(replay_tools replay_game3d_dump.cpp preview_game3d_dump.cpp ui_detection_replay.cpp replay_game3d_package.h)
set(scanned "")
foreach(source IN LISTS sources)
  get_filename_component(name "${source}" NAME)
  if(source MATCHES "^experiments/" OR source MATCHES "(^|/)(cmake-build-[^/]*|CMakeFiles)/"
      OR (name MATCHES "^test_" AND NOT source IN_LIST test_addon_sources) OR source IN_LIST replay_tools)
    continue()
  endif()
  list(APPEND scanned "${source}")
endforeach()
if(NOT "game3d_clock.h" IN_LIST scanned OR NOT "game3d_pool_submit.h" IN_LIST scanned)
  message(FATAL_ERROR "SOURCE_DIR ${SOURCE_DIR} does not hold the add-on's sources")
endif()
foreach(source IN LISTS test_addon_sources)
  if(NOT source IN_LIST scanned)
    message(FATAL_ERROR "The test add-on's source ${source} (TEST_ADDON_SOURCES) is not under ${SOURCE_DIR}")
  endif()
endforeach()
foreach(source IN LISTS converted_fixtures)
  if(NOT EXISTS "${SOURCE_DIR}/${source}" OR source IN_LIST scanned)
    message(FATAL_ERROR "The converted fixture ${source} is not a fixture under ${SOURCE_DIR}")
  endif()
endforeach()

# Every entry names a scanned file and a known kind, at most once.
set(entries "")
foreach(entry IN LISTS allowed)
  string(REPLACE "|" ";" fields "${entry}")
  list(LENGTH fields n)
  list(GET fields 0 file)
  list(GET fields 1 kind)
  if(NOT n EQUAL 5 OR NOT kind IN_LIST kinds)
    message(FATAL_ERROR "Malformed clock domain entry: ${entry}")
  endif()
  if(NOT file IN_LIST scanned AND NOT (file IN_LIST converted_fixtures AND kind STREQUAL "tick"))
    message(FATAL_ERROR "The clock domain entry names a file that is not an add-on source (or a converted fixture's tick): ${entry}")
  endif()
  if("${file}|${kind}" IN_LIST entries)
    message(FATAL_ERROR "Two clock domain entries for ${file} (${kind})")
  endif()
  list(APPEND entries "${file}|${kind}")
endforeach()

set(failures "")
set(total 0)
foreach(source IN LISTS scanned converted_fixtures)
  file(READ "${SOURCE_DIR}/${source}" text)
  clock_code("${text}" code)
  set(source_kinds ${kinds})
  if(source IN_LIST converted_fixtures)
    set(source_kinds tick)
  endif()
  foreach(kind IN LISTS source_kinds)
    clock_sites("${code}" ${kind} found)
    list(LENGTH found count)
    set(want 0)
    set(domain "")
    foreach(entry IN LISTS allowed)
      string(REPLACE "|" ";" fields "${entry}")
      list(GET fields 0 file)
      list(GET fields 1 entry_kind)
      if(file STREQUAL source AND entry_kind STREQUAL kind)
        list(GET fields 2 want)
        list(GET fields 3 domain)
      endif()
    endforeach()
    math(EXPR total "${total} + ${count}")
    if(NOT count EQUAL want)
      set(lines "")
      string(REPLACE "\n" ";" code_lines "${code}")
      foreach(line IN LISTS code_lines)
        clock_sites("${line}" ${kind} in_line)
        if(in_line)
          string(STRIP "${line}" line)
          string(APPEND lines "\n    ${line}")
        endif()
      endforeach()
      if(want EQUAL 0)
        set(rule "no ${kind} site may appear here")
      else()
        set(rule "${want} allowed (${domain})")
      endif()
      if(kind STREQUAL "tick" AND source IN_LIST converted_fixtures)
        string(APPEND rule " (a fixture on the game clock reads game time with clock.now() and wall time with "
          "game_clock::wall_now(), test_game3d_clock.h)")
      elseif(kind STREQUAL "tick")
        string(APPEND rule " (game time is read with sunshine_game3d::clock::game_now(), game3d_clock.h)")
      endif()
      string(APPEND failures "\n${source}: ${count} ${kind} sites, ${rule}${lines}")
    endif()
  endforeach()
endforeach()

if(failures)
  string(REPLACE "<semicolon>" ";" failures "${failures}")
  message(FATAL_ERROR "Clock reads or pool submissions without a stated domain (check_clock_domains.cmake lists the "
    "allowed ones, README, Test builds):${failures}")
endif()
list(LENGTH scanned files)
list(LENGTH converted_fixtures fixtures)
list(LENGTH allowed listed)
message(STATUS "PASS ${files} add-on sources and ${fixtures} fixtures on the game clock: ${total} clock reads and "
  "pool submissions outside game_now() and the fixture clock, all at the ${listed} listed sites with a stated domain")
