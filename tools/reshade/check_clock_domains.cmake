# SPDX-License-Identifier: GPL-3.0-only
# reshade_clock_domains, in script mode (README, Test builds). The test
# add-on's clock seams rely on every clock read and every piece of
# background work of the add-on having chosen its domain, so each rule below
# names a pattern and the only production files allowed to use it. The
# production files are every .cpp, .h and .inl under SOURCE_DIR except the
# tests (test_*), experiments/ and the Dump 3D replay and preview tools;
# comments and string literals do not count.
#
#   cmake -DSOURCE_DIR=<tools/reshade> -P check_clock_domains.cmake
cmake_minimum_required(VERSION 3.24)

if(NOT SOURCE_DIR)
  message(FATAL_ERROR "SOURCE_DIR is required")
endif()

set(rules "")
# domain_rule(<name> <regex> [allowed file...]): <regex> may match only in
# the allowed files (paths relative to SOURCE_DIR); none allowed = nowhere.
function(domain_rule name regex)
  set(rules ${rules} ${name} PARENT_SCOPE)
  set(rule_regex_${name} "${regex}" PARENT_SCOPE)
  set(rule_allowed_${name} ${ARGN} PARENT_SCOPE)
endfunction()

# Pool work: pool::submit starts every thread-pool callback, so that the test
# add-on counts each one until it returns (game3d_pool_submit.h,
# SunshineGame3DTestQuiesce).
domain_rule(pool "(^|[^A-Za-z0-9_])TrySubmitThreadpoolCallback[ \t\n]*\\(" game3d_pool_submit.h)
# Nothing else runs the add-on's code in the background: a thread, timer,
# wait or work object of its own would escape that count.
domain_rule(background
  "(^|[^A-Za-z0-9_])(CreateThreadpool(Work|Timer|Wait|Io)|SubmitThreadpoolWork|QueueUserWorkItem|CreateTimerQueueTimer|RegisterWaitForSingleObject|CreateRemoteThread(Ex)?|CreateThread|_beginthread(ex)?|std::(thread|jthread|async))([^A-Za-z0-9_]|$)")

file(GLOB_RECURSE sources RELATIVE "${SOURCE_DIR}" "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.h" "${SOURCE_DIR}/*.inl")
list(FILTER sources EXCLUDE REGEX "(^|/)test_[^/]*$")
list(FILTER sources EXCLUDE REGEX "^experiments/")
list(FILTER sources EXCLUDE REGEX "(^|/)(cmake-build-[^/]*|CMakeFiles)/")
list(REMOVE_ITEM sources replay_game3d_dump.cpp replay_game3d_package.h preview_game3d_dump.cpp ui_detection_replay.cpp)
list(LENGTH sources source_count)
if(source_count LESS 50)
  message(FATAL_ERROR "Found ${source_count} production files under ${SOURCE_DIR}; is SOURCE_DIR tools/reshade?")
endif()

set(violations "")
foreach(source IN LISTS sources)
  file(READ "${SOURCE_DIR}/${source}" text)
  string(REPLACE ";" "<semicolon>" code "${text}")
  string(REPLACE "\r" "" code "${code}")
  # Comments and literals, in one left-to-right pass, so that whichever
  # starts first wins: a "//" inside a string is no comment, and a quote
  # inside a comment opens no string.
  string(REGEX REPLACE "/\\*([^*]|\\*+[^*/])*\\*+/|//[^\n]*|\"([^\"\\\\\n]|\\\\.)*\"|'([^'\\\\\n]|\\\\.)*'" " " code "${code}")
  foreach(rule IN LISTS rules)
    if(source IN_LIST rule_allowed_${rule})
      continue()
    endif()
    string(REGEX MATCHALL "${rule_regex_${rule}}" found "${code}")
    list(LENGTH found count)
    if(count)
      list(GET found 0 first)
      string(REGEX REPLACE "^[^A-Za-z_]" "" first "${first}")
      string(REGEX REPLACE "[^A-Za-z0-9_:]+$" "" first "${first}")
      list(APPEND violations "${source}: ${count} use(s) outside the ${rule} domain's files, the first ${first}")
    endif()
  endforeach()
endforeach()

if(violations)
  list(JOIN violations "\n  " report)
  message(FATAL_ERROR "FAIL clock and pool domains:\n  ${report}")
endif()
list(LENGTH rules rule_count)
message("PASS clock and pool domains: ${rule_count} rules over ${source_count} production files")
