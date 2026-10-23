# SPDX-License-Identifier: GPL-3.0-only
# contract_sl_survey_tables.cmake, in script mode. Writes the Streamline survey
# tables that the API contract oracle reads (test_sl_survey_tables.generated.h)
# from the Streamline ABI survey, or checks the checked-in copy against it
# (reshade_contract_sl_survey_tables). The oracle takes every buffer-type
# meaning from these tables, never from streamline_buffer_contract.h.
#
#   cmake -DSURVEY_DIR=<sl-abi-survey> -DOUTPUT=<header> -P contract_sl_survey_tables.cmake
#   cmake -DSURVEY_DIR=<sl-abi-survey> -DCHECK=<checked-in header> -P contract_sl_survey_tables.cmake
#
# Each survey directory v<major>.<minor>.<patch>[suffix]/include holds that
# official release's unmodified public headers. Per release the table records
# its version (sl_version.h where it exists, which must agree with the tag),
# every declared buffer type (the 2.x kBufferType* constants, or the 1.x
# BufferType enumerators in declaration order), the declared struct versions
# of the structures the add-on reads, whether the Resource carries a state
# word, and the interposer DLL's version resource from
# all-release-dll-results.json (its fixed version and strings only, never the
# classifier's verdict).
cmake_minimum_required(VERSION 3.24)

if(NOT SURVEY_DIR OR NOT IS_DIRECTORY "${SURVEY_DIR}")
  message(FATAL_ERROR "SURVEY_DIR must name the sl-abi-survey directory")
endif()
if(NOT OUTPUT AND NOT CHECK)
  message(FATAL_ERROR "Set OUTPUT to write the table or CHECK to compare a checked-in copy")
endif()

# The structures the add-on reads, by their SDK names.
set(struct_names Constants Resource ResourceTag PrecisionInfo ViewportHandle DLSSGOptions)
set(struct_fields constants resource resource_tag precision_info viewport_handle dlssg_options)

# The interposer version resources, by release tag.
set(results_path "${SURVEY_DIR}/all-release-dll-results.json")
if(NOT EXISTS "${results_path}")
  message(FATAL_ERROR "${results_path} is missing")
endif()
file(READ "${results_path}" results_json)
string(JSON result_count LENGTH "${results_json}" results)
math(EXPR result_last "${result_count} - 1")
foreach(index RANGE ${result_last})
  string(JSON result_tag GET "${results_json}" results ${index} tag)
  string(JSON result_output GET "${results_json}" results ${index} output)
  if(NOT result_output MATCHES "^fixed=([0-9]+)\\.([0-9]+)\\.([0-9]+)\\.([0-9]+) strings='([^'\\\\\"]*)'/'([^'\\\\\"]*)'")
    message(FATAL_ERROR "${result_tag}: unreadable version resource '${result_output}'")
  endif()
  set(dll_${result_tag} "{${CMAKE_MATCH_1}, ${CMAKE_MATCH_2}, ${CMAKE_MATCH_3}, ${CMAKE_MATCH_4}}, \"${CMAKE_MATCH_5}\", \"${CMAKE_MATCH_6}\"")
endforeach()

file(GLOB release_dirs LIST_DIRECTORIES true "${SURVEY_DIR}/v*")
list(SORT release_dirs COMPARE NATURAL)
set(type_arrays "")
set(release_rows "")
set(release_count 0)
foreach(release_dir IN LISTS release_dirs)
  if(NOT IS_DIRECTORY "${release_dir}/include")
    continue()
  endif()
  get_filename_component(tag "${release_dir}" NAME)
  if(NOT tag MATCHES "^v([0-9]+)\\.([0-9]+)\\.([0-9]+)[a-z]*$")
    message(FATAL_ERROR "${tag}: not a release tag")
  endif()
  set(major ${CMAKE_MATCH_1})
  set(minor ${CMAKE_MATCH_2})
  set(patch ${CMAKE_MATCH_3})
  set(include_dir "${release_dir}/include")
  if(EXISTS "${include_dir}/sl_version.h")
    foreach(part IN ITEMS MAJOR MINOR PATCH)
      file(STRINGS "${include_dir}/sl_version.h" define REGEX "^#define[ \t]+SL_VERSION_${part}[ \t]+[0-9]+")
      string(REGEX REPLACE "^#define[ \t]+SL_VERSION_${part}[ \t]+([0-9]+).*$" "\\1" declared "${define}")
      string(TOLOWER "${part}" field)
      if(NOT declared STREQUAL "${${field}}")
        message(FATAL_ERROR "${tag}: sl_version.h declares ${part} '${declared}'")
      endif()
    endforeach()
  endif()

  file(GLOB headers "${include_dir}/*.h")
  list(SORT headers)
  set(values "")
  set(names "")
  set(enumerator 0)
  foreach(field IN LISTS struct_fields)
    set(${field} 0)
  endforeach()
  foreach(header IN LISTS headers)
    # 2.x: constexpr BufferType kBufferType<Name> = <value>;
    file(STRINGS "${header}" constant_lines REGEX "^[ \t]*constexpr[ \t]+BufferType[ \t]+kBufferType[A-Za-z0-9_]+[ \t]*=[ \t]*[0-9]+[ \t]*;")
    foreach(line IN LISTS constant_lines)
      string(REGEX REPLACE "^[ \t]*constexpr[ \t]+BufferType[ \t]+kBufferType([A-Za-z0-9_]+)[ \t]*=[ \t]*([0-9]+).*$" "\\1;\\2" pair "${line}")
      list(GET pair 0 name)
      list(GET pair 1 value)
      list(APPEND names "${name}")
      list(APPEND values "${value}")
    endforeach()
    # 1.x: enum BufferType { eBufferType<Name>, ... eBufferTypeCount }.
    file(STRINGS "${header}" enumerator_lines REGEX "^[ \t]*eBufferType[A-Za-z0-9_]+[ \t]*(=[ \t]*[0-9]+[ \t]*)?,?[ \t]*(//.*)?$")
    foreach(line IN LISTS enumerator_lines)
      string(REGEX REPLACE "^[ \t]*eBufferType([A-Za-z0-9_]+).*$" "\\1" name "${line}")
      if(line MATCHES "=[ \t]*([0-9]+)")
        set(enumerator ${CMAKE_MATCH_1})
      endif()
      # The enumeration's terminator is a count, not a buffer type.
      if(NOT name STREQUAL "Count")
        list(APPEND names "${name}")
        list(APPEND values "${enumerator}")
      endif()
      math(EXPR enumerator "${enumerator} + 1")
    endforeach()
    # SL_STRUCT[_BEGIN|_PROTECTED[_BEGIN]](<Name>, StructType(...), kStructVersion<N>)
    foreach(struct_name field IN ZIP_LISTS struct_names struct_fields)
      file(STRINGS "${header}" declaration REGEX "^[ \t]*SL_STRUCT(_PROTECTED)?(_BEGIN)?[ \t]*\\([ \t]*${struct_name}[ \t]*,.*kStructVersion[0-9]+[ \t]*\\)")
      if(declaration)
        list(LENGTH declaration declaration_count)
        if(NOT declaration_count EQUAL 1 OR NOT ${field} EQUAL 0)
          message(FATAL_ERROR "${tag}: ${struct_name} is declared more than once")
        endif()
        string(REGEX REPLACE "^.*kStructVersion([0-9]+)[ \t]*\\).*$" "\\1" ${field} "${declaration}")
      endif()
    endforeach()
  endforeach()

  list(LENGTH values type_count)
  if(type_count EQUAL 0)
    message(FATAL_ERROR "${tag}: no buffer types were found")
  endif()
  set(seen_values "")
  set(seen_names "")
  set(rows "")
  foreach(name value IN ZIP_LISTS names values)
    if(value IN_LIST seen_values OR name IN_LIST seen_names)
      message(FATAL_ERROR "${tag}: buffer type ${name} = ${value} is declared twice")
    endif()
    list(APPEND seen_values "${value}")
    list(APPEND seen_names "${name}")
    string(APPEND rows "    {${value}, \"${name}\"},\n")
  endforeach()

  # A legacy Resource is a plain struct; the typed 2.x Resource always
  # declares its state (SLT:331-389).
  set(resource_state true)
  if(resource EQUAL 0)
    file(READ "${include_dir}/sl.h" sl_text)
    string(REGEX MATCH "\nstruct Resource[ \t\r]*\n" resource_start "${sl_text}")
    if(NOT resource_start)
      message(FATAL_ERROR "${tag}: sl.h declares no struct Resource")
    endif()
    string(FIND "${sl_text}" "${resource_start}" resource_offset)
    string(SUBSTRING "${sl_text}" ${resource_offset} -1 resource_text)
    string(FIND "${resource_text}" "\n};" resource_end)
    string(SUBSTRING "${resource_text}" 0 ${resource_end} resource_text)
    string(REGEX REPLACE "//[^\n]*" "" resource_text "${resource_text}")
    if(NOT resource_text MATCHES "[ \t\n]state[ \t]*[{=;]")
      set(resource_state false)
    endif()
  endif()

  # The 2.x enumerations the oracle reads per release: how many ResourceType
  # and DLSSGMode values come before eCount (0 where the release declares no
  # such enumeration), and whether numFramesToGenerate may exceed 1 (before
  # 2.7 its comment reads "Must be 1").
  set(enumeration_names ResourceType DLSSGMode)
  set(enumeration_fields resource_types dlssg_modes)
  set(resource_types 0)
  set(dlssg_modes 0)
  set(dlssg_multi_frame false)
  foreach(header IN LISTS headers)
    file(READ "${header}" header_text)
    string(REGEX REPLACE "//[^\n]*" "" header_code "${header_text}")
    foreach(enumeration field IN ZIP_LISTS enumeration_names enumeration_fields)
      string(REGEX MATCH "enum class ${enumeration}[^{;]*{[^}]*}" body "${header_code}")
      if(NOT body)
        continue()
      endif()
      if(NOT ${field} EQUAL 0)
        message(FATAL_ERROR "${tag}: ${enumeration} is declared more than once")
      endif()
      string(REGEX REPLACE "^[^{]*{([^}]*)}$" "\\1" body "${body}")
      string(REGEX REPLACE "[ \t\r\n]" "" body "${body}")
      string(REPLACE "," ";" enumerators "${body}")
      list(FIND enumerators eCount count)
      if(count LESS 1 OR body MATCHES "=")
        message(FATAL_ERROR "${tag}: ${enumeration} is not a plain enumeration ending in eCount")
      endif()
      set(${field} ${count})
    endforeach()
    # The DLSSGOptions member and the comment line right above it.
    file(STRINGS "${header}" header_lines)
    set(previous "")
    foreach(line IN LISTS header_lines)
      if(line MATCHES "^[ \t]*uint32_t[ \t]+numFramesToGenerate[ \t]*=")
        set(dlssg_multi_frame true)
        if(previous MATCHES "Must be 1")
          set(dlssg_multi_frame false)
        endif()
      endif()
      set(previous "${line}")
    endforeach()
  endforeach()
  if(dlssg_modes EQUAL 0 AND dlssg_multi_frame)
    message(FATAL_ERROR "${tag}: numFramesToGenerate is declared without DLSSGMode")
  endif()

  if(NOT DEFINED dll_${tag})
    message(FATAL_ERROR "${tag}: all-release-dll-results.json has no version resource")
  endif()
  string(REPLACE "." "_" array "${tag}_types")
  string(APPEND type_arrays "\n  inline constexpr buffer_type ${array}[] {\n${rows}  };\n")
  string(APPEND release_rows "    {\"${tag}\", ${major}, ${minor}, ${patch}, ${array}, ${type_count}, ${constants}, ${resource}, ${resource_tag}, ${precision_info}, ${viewport_handle}, ${dlssg_options}, ${resource_state}, ${resource_types}, ${dlssg_modes}, ${dlssg_multi_frame}, ${dll_${tag}}},\n")
  math(EXPR release_count "${release_count} + 1")
endforeach()
if(release_count EQUAL 0)
  message(FATAL_ERROR "${SURVEY_DIR} holds no release headers")
endif()

string(CONCAT generated
  "// SPDX-License-Identifier: GPL-3.0-only\n"
  "#pragma once\n"
  "// Generated by contract_sl_survey_tables.cmake from the Streamline ABI survey\n"
  "// (sl-abi-survey): every surveyed official release's public headers and its\n"
  "// interposer DLL's version resource. Do not edit; regenerate it as\n"
  "// tools/reshade/README.md (API contract matrix) describes.\n"
  "// reshade_contract_sl_survey_tables compares this copy with the survey.\n"
  "#include <cstdint>\n"
  "\n"
  "namespace game3d_contract::survey {\n"
  "  struct buffer_type {\n"
  "    std::uint32_t value;\n"
  "    const char *name;  // The declaration's name without kBufferType or eBufferType.\n"
  "  };\n"
  "\n"
  "  struct release {\n"
  "    const char *tag;\n"
  "    std::uint32_t major, minor, patch;\n"
  "    const buffer_type *types;\n"
  "    std::uint32_t type_count;\n"
  "    // Declared struct versions; 0 where the release declares no versioned struct.\n"
  "    std::uint32_t constants, resource, resource_tag, precision_info, viewport_handle, dlssg_options;\n"
  "    bool resource_state;  // The Resource carries a state word.\n"
  "    // The ResourceType and DLSSGMode values declared before eCount (0 without\n"
  "    // the enumeration), and whether numFramesToGenerate may exceed 1.\n"
  "    std::uint32_t resource_types, dlssg_modes;\n"
  "    bool dlssg_multi_frame;\n"
  "    // The interposer DLL's fixed file version and its FileVersion and\n"
  "    // ProductVersion strings.\n"
  "    std::uint32_t dll_fixed[4];\n"
  "    const char *dll_file_version, *dll_product_version;\n"
  "  };\n"
  "${type_arrays}"
  "\n"
  "  inline constexpr release releases[] {\n"
  "${release_rows}"
  "  };\n"
  "}  // namespace game3d_contract::survey\n")

if(OUTPUT)
  file(WRITE "${OUTPUT}" "${generated}")
  message(STATUS "Wrote ${OUTPUT}: ${release_count} surveyed releases")
endif()
if(CHECK)
  if(NOT EXISTS "${CHECK}")
    message(FATAL_ERROR "FAIL Streamline survey tables: ${CHECK} is missing")
  endif()
  file(READ "${CHECK}" checked_in)
  string(REPLACE "\r\n" "\n" checked_in "${checked_in}")
  if(NOT checked_in STREQUAL generated)
    message(FATAL_ERROR "FAIL Streamline survey tables: ${CHECK} differs from the survey in ${SURVEY_DIR}; "
      "regenerate it with -DOUTPUT (tools/reshade/README.md, API contract matrix) and review the change")
  endif()
  message(STATUS "PASS Streamline survey tables: the checked-in copy matches ${release_count} surveyed releases")
endif()
