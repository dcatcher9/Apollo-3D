# SPDX-License-Identifier: GPL-3.0-only
# reshade_contract_oracle_independence, in script mode. The API contract
# oracle derives every expected outcome from the Streamline, D3D12 and NGX
# contracts, never from the add-on, so its headers may include only <d3d12.h>,
# <cstdint>, each other and generated SDK constant tables, and may name nothing
# the add-on declares; the case generator includes only the standard library.
# With COMPILER set, each header must also compile on its own, which proves it
# uses no declaration it did not include.
#
#   cmake -DSOURCE_DIR=<tools/reshade> [-DWORK_DIR=<dir> -DCOMPILER=<c++>
#     -DCOMPILER_FRONTEND=<GNU|MSVC>] -P test_game3d_api_contract_independence.cmake
cmake_minimum_required(VERSION 3.24)

if(NOT SOURCE_DIR)
  message(FATAL_ERROR "SOURCE_DIR is required")
endif()

set(survey_tables "test_sl_survey_tables.generated.h")
set(generated_tables "\"${survey_tables}\"")
set(dimensions "test_game3d_api_contract_dimensions.h")
set(oracle "test_game3d_api_contract_oracle.h")
set(known_gaps "test_game3d_api_contract_known_gaps.h")
set(generator "test_contract_matrix_generator.h")
set(headers ${survey_tables} ${dimensions} ${oracle} ${known_gaps} ${generator})
# The generated tables hold survey data only (contract_sl_survey_tables.cmake).
set(allowed_${survey_tables} "<cstdint>")
set(allowed_${dimensions} "<cstdint>" "<d3d12.h>" ${generated_tables})
set(allowed_${oracle} "<cstdint>" "<d3d12.h>" ${generated_tables} "\"${dimensions}\"")
# The oracle never includes the known-gap table: gaps describe the add-on.
set(allowed_${known_gaps} "<cstdint>" "<d3d12.h>" "\"${dimensions}\"" "\"${oracle}\"")
# The case generator knows only dimension names and values (standard library).
set(allowed_${generator} "<algorithm>" "<cstdint>" "<cstdio>" "<cstdlib>" "<cstring>" "<functional>" "<string>" "<vector>")

# Appends one message per violation in text to the list named by out.
function(contract_violations text allowed out)
  set(found "")
  string(REPLACE ";" "<semicolon>" text "${text}")
  string(REPLACE "\r" "" text "${text}")
  string(REGEX REPLACE "/\\*([^*]|\\*+[^*/])*\\*+/" "" code "${text}")
  # Every include-like directive must name an allowed header.
  string(REGEX MATCHALL "(^|\n)[ \t]*#[ \t]*(include|include_next|import)[^\n]*" directives "${code}")
  foreach(directive IN LISTS directives)
    string(REGEX REPLACE "^\n?[ \t]*#[ \t]*[a-z_]+[ \t]*" "" operand "${directive}")
    string(REGEX REPLACE "[ \t]*//.*$" "" operand "${operand}")
    string(STRIP "${operand}" operand)
    if(NOT operand IN_LIST allowed)
      list(APPEND found "includes ${operand}")
    endif()
  endforeach()
  # No add-on name may appear in code (string literals and comments aside).
  string(REGEX REPLACE "\"([^\"\\\\\n]|\\\\.)*\"" "\"\"" code "${code}")
  string(REGEX REPLACE "//[^\n]*" "" code "${code}")
  string(REGEX MATCHALL "(^|[^A-Za-z0-9_])(__has_include|[Ss]unshine[A-Za-z0-9_]*|sl::|reshade::|ReShade[A-Za-z0-9_]*)" names "${code}")
  foreach(name IN LISTS names)
    string(REGEX REPLACE "^[^A-Za-z_]" "" name "${name}")
    list(APPEND found "names ${name}")
  endforeach()
  set(${out} "${found}" PARENT_SCOPE)
endfunction()

# The check must itself refuse what it exists to refuse.
set(self_allowed "<cstdint>" "<d3d12.h>")
foreach(sample IN ITEMS
    "#include \"capture_state_policy.h\"\n"
    "  #  include <streamline_depth_capture.h> // a comment\n"
    "#include SOME_MACRO\n"
    "#if __has_include(<x.h>)\n#endif\n"
    "namespace alias = sunshine_streamline::depth_capture\n"
    "constexpr auto x = sl::Result::eOk\n")
  contract_violations("${sample}" "${self_allowed}" refused)
  if(NOT refused)
    message(FATAL_ERROR "The independence check accepted a production dependency:\n${sample}")
  endif()
endforeach()
set(sample "#include <cstdint>\n// #include \"capture_state_policy.h\" in a comment\n/* sunshine_streamline */\nconstexpr const char *c = \"docs/reshade-sbs.md sunshine_x\"\n")
contract_violations("${sample}" "${self_allowed}" refused)
if(refused)
  message(FATAL_ERROR "The independence check refused a clean header: ${refused}")
endif()

set(failures "")
foreach(header IN LISTS headers)
  set(path "${SOURCE_DIR}/${header}")
  if(NOT EXISTS "${path}")
    list(APPEND failures "${header}: missing")
    continue()
  endif()
  file(READ "${path}" text)
  contract_violations("${text}" "${allowed_${header}}" refused)
  foreach(violation IN LISTS refused)
    list(APPEND failures "${header} ${violation}")
  endforeach()
endforeach()

if(COMPILER AND WORK_DIR AND NOT failures)
  file(MAKE_DIRECTORY "${WORK_DIR}")
  foreach(header IN LISTS headers)
    string(REGEX REPLACE "\\.h$" ".cpp" unit "${WORK_DIR}/standalone_${header}")
    file(WRITE "${unit}" "#include \"${header}\"\n")
    if(COMPILER_FRONTEND STREQUAL "MSVC")
      set(command "${COMPILER}" /nologo /Zs /TP /std:c++17 /DWIN32_LEAN_AND_MEAN /DNOMINMAX "/I${SOURCE_DIR}" "${unit}")
    else()
      set(command "${COMPILER}" -std=c++17 -fsyntax-only -DWIN32_LEAN_AND_MEAN -DNOMINMAX "-I${SOURCE_DIR}" "${unit}")
    endif()
    execute_process(COMMAND ${command} RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE output)
    if(NOT result EQUAL 0)
      list(APPEND failures "${header} does not compile on its own:\n${output}")
    endif()
  endforeach()
endif()

if(failures)
  string(REPLACE ";" "\n" failures "${failures}")
  string(REPLACE "<semicolon>" ";" failures "${failures}")
  message(FATAL_ERROR "FAIL API contract oracle independence:\n${failures}")
endif()
list(LENGTH headers header_count)
message(STATUS "PASS API contract oracle independence: ${header_count} headers include and name nothing from the add-on")
