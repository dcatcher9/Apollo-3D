# Keep development/runtime copies unchanged. Only the installer representation
# changes: one solid archive avoids NSIS's 2 GiB uncompressed solid-block limit.
option(SUNSHINE_PACKAGE_TENSORRT_ARCHIVE "Bundle TensorRT in a local solid archive for installer extraction" OFF)
set(SUNSHINE_TENSORRT_ARCHIVE_FILE "" CACHE FILEPATH "Optional existing local TensorRT 7z archive; its complete contents must match the configured SDK")
set(SUNSHINE_NSIS_TENSORRT_UNINSTALL "")
if(NOT SUNSHINE_PACKAGE_TENSORRT_ARCHIVE)
    if(PROJECT_TENSORRT_DLLS)
        install(FILES ${PROJECT_TENSORRT_DLLS} DESTINATION "." COMPONENT application)
    endif()
    return()
endif()

find_program(SUNSHINE_7ZIP_EXECUTABLE NAMES 7z.exe 7z
    HINTS "$ENV{ProgramW6432}/7-Zip" "$ENV{ProgramFiles}/7-Zip"
    DOC "Existing Windows 7-Zip executable used to build and extract the bundled TensorRT archive")
if(NOT SUNSHINE_7ZIP_EXECUTABLE)
    message(FATAL_ERROR "SUNSHINE_PACKAGE_TENSORRT_ARCHIVE requires an existing 64-bit 7-Zip installation. Set SUNSHINE_7ZIP_EXECUTABLE.")
endif()
get_filename_component(_sevenzip_dir "${SUNSHINE_7ZIP_EXECUTABLE}" DIRECTORY)
set(_sevenzip_dll "${_sevenzip_dir}/7z.dll")
set(_sevenzip_license "${_sevenzip_dir}/License.txt")
foreach(_tool IN ITEMS "${SUNSHINE_7ZIP_EXECUTABLE}" "${_sevenzip_dll}" "${_sevenzip_license}")
    if(NOT EXISTS "${_tool}" OR IS_DIRECTORY "${_tool}")
        message(FATAL_ERROR "The complete 7-Zip executable, 7z.dll, and License.txt must be present: ${_tool}")
    endif()
endforeach()
file(READ "${SUNSHINE_7ZIP_EXECUTABLE}" _pe_offset_hex OFFSET 60 LIMIT 4 HEX)
string(REGEX REPLACE "(..)(..)(..)(..)" "0x\\4\\3\\2\\1" _pe_offset_hex "${_pe_offset_hex}")
math(EXPR _pe_offset "${_pe_offset_hex}")
file(READ "${SUNSHINE_7ZIP_EXECUTABLE}" _pe_header OFFSET ${_pe_offset} LIMIT 6 HEX)
if(NOT _pe_header STREQUAL "504500006486")
    message(FATAL_ERROR "The bundled 7-Zip executable must be a Windows x64 PE executable.")
endif()
file(READ "${_sevenzip_license}" _license_text)
if(NOT _license_text MATCHES "GNU LGPL" OR NOT _license_text MATCHES "unRAR license restriction")
    message(FATAL_ERROR "7-Zip License.txt must include the LGPL information and unRAR restriction.")
endif()

# The bundled extractor has the same complete TensorRT 11 runtime contract.
# Never select architecture resources based on the packager's GPU.
set(_expected_names
    nvinfer_11.dll
    nvinfer_builder_resource_ptx_11.dll
    nvinfer_builder_resource_sm100_11.dll
    nvinfer_builder_resource_sm120_11.dll
    nvinfer_builder_resource_sm75_11.dll
    nvinfer_builder_resource_sm80_11.dll
    nvinfer_builder_resource_sm86_11.dll
    nvinfer_builder_resource_sm89_11.dll
    nvinfer_builder_resource_sm90_11.dll
    nvinfer_dispatch_11.dll
    nvinfer_lean_11.dll
    nvinfer_plugin_11.dll
    nvinfer_vc_plugin_11.dll
    nvonnxparser_11.dll)
set(_runtime_sources ${PROJECT_TENSORRT_DLLS})
list(SORT _runtime_sources)
set(_actual_names "")
set(_files_json "[]")
set(_listfile_content "")
set(_validation "")
set(_index 0)
foreach(_dll IN LISTS _runtime_sources)
    if(NOT EXISTS "${_dll}" OR IS_DIRECTORY "${_dll}")
        message(FATAL_ERROR "TensorRT runtime DLL is missing: ${_dll}")
    endif()
    get_filename_component(_name "${_dll}" NAME)
    if(NOT _name IN_LIST _expected_names OR _name IN_LIST _actual_names)
        message(FATAL_ERROR "Unsupported or duplicate TensorRT runtime DLL: ${_name}")
    endif()
    list(APPEND _actual_names "${_name}")
    file(SHA256 "${_dll}" _sha256)
    file(SIZE "${_dll}" _size)
    string(JSON _entry SET "{}" name "\"${_name}\"")
    string(JSON _entry SET "${_entry}" sha256 "\"${_sha256}\"")
    string(JSON _entry SET "${_entry}" size "${_size}")
    string(JSON _files_json SET "${_files_json}" ${_index} "${_entry}")
    math(EXPR _index "${_index} + 1")
    string(APPEND _listfile_content "${_dll}\n")
    string(APPEND _validation
        "file(SHA256 \"${_dll}\" _actual)\nif(NOT _actual STREQUAL \"${_sha256}\")\n  message(FATAL_ERROR \"TensorRT input changed after configure: ${_name}\")\nendif()\n")
    string(APPEND SUNSHINE_NSIS_TENSORRT_UNINSTALL "Delete \"$INSTDIR\\${_name}\"\n")
endforeach()
list(SORT _actual_names)
list(SORT _expected_names)
if(NOT _actual_names STREQUAL _expected_names)
    message(FATAL_ERROR "Archived TensorRT packaging requires the complete set of 14 TensorRT 11 DLLs.")
endif()

set(_cache "${CMAKE_BINARY_DIR}/tensorrt-package")
set(SUNSHINE_TENSORRT_RUNTIME_ARCHIVE "${_cache}/runtime.7z")
set(SUNSHINE_TENSORRT_RUNTIME_MANIFEST "${_cache}/tensorrt-runtime.json")
file(MAKE_DIRECTORY "${_cache}")
file(LOCK "${_cache}/prepare.lock" GUARD PROCESS TIMEOUT 0 RESULT_VARIABLE _lock_result)
if(NOT _lock_result STREQUAL "0")
    message(FATAL_ERROR "Another process is preparing the TensorRT package. Wait for it to finish: ${_cache}")
endif()
set(_compression "7z/LZMA2/mx9/d512m/solid/threads4/no-timestamps/v1")
if(SUNSHINE_TENSORRT_ARCHIVE_FILE)
    if(NOT EXISTS "${SUNSHINE_TENSORRT_ARCHIVE_FILE}" OR IS_DIRECTORY "${SUNSHINE_TENSORRT_ARCHIVE_FILE}")
        message(FATAL_ERROR "Supplied TensorRT archive does not exist: ${SUNSHINE_TENSORRT_ARCHIVE_FILE}")
    endif()
    file(SHA256 "${SUNSHINE_TENSORRT_ARCHIVE_FILE}" _supplied_archive_hash)
    set(_compression "supplied-7z/extracted-sha256-verified/v1/${_supplied_archive_hash}")
endif()
set(_input_fingerprint "${_compression}\n${_files_json}")
foreach(_tool IN ITEMS "${SUNSHINE_7ZIP_EXECUTABLE}" "${_sevenzip_dll}" "${_sevenzip_license}")
    file(SHA256 "${_tool}" _tool_hash)
    string(APPEND _input_fingerprint "\n${_tool_hash}")
    string(APPEND _validation
        "file(SHA256 \"${_tool}\" _actual)\nif(NOT _actual STREQUAL \"${_tool_hash}\")\n  message(FATAL_ERROR \"7-Zip input changed after configure: ${_tool}\")\nendif()\n")
endforeach()
string(SHA256 _input_hash "${_input_fingerprint}")
set(_reuse FALSE)
if(EXISTS "${_cache}/inputs.sha256" AND EXISTS "${SUNSHINE_TENSORRT_RUNTIME_ARCHIVE}"
    AND EXISTS "${SUNSHINE_TENSORRT_RUNTIME_MANIFEST}")
    file(READ "${_cache}/inputs.sha256" _previous_input_hash)
    if(_previous_input_hash STREQUAL _input_hash)
        file(READ "${SUNSHINE_TENSORRT_RUNTIME_MANIFEST}" _previous_manifest)
        string(JSON _previous_archive_hash ERROR_VARIABLE _manifest_error GET "${_previous_manifest}" archiveSha256)
        string(JSON _previous_files ERROR_VARIABLE _files_error GET "${_previous_manifest}" files)
        if(NOT _manifest_error AND NOT _files_error AND _previous_files STREQUAL _files_json)
            file(SHA256 "${SUNSHINE_TENSORRT_RUNTIME_ARCHIVE}" _archive_hash)
            if(_archive_hash STREQUAL _previous_archive_hash)
                set(_reuse TRUE)
            endif()
        endif()
    endif()
endif()
if(NOT _reuse)
    file(WRITE "${_cache}/inputs.txt" "${_listfile_content}")
    set(_partial "${_cache}/runtime.partial.7z")
    file(REMOVE "${_partial}")
    if(SUNSHINE_TENSORRT_ARCHIVE_FILE)
        file(COPY_FILE "${SUNSHINE_TENSORRT_ARCHIVE_FILE}" "${_partial}")
        file(SHA256 "${_partial}" _copied_archive_hash)
        if(NOT _copied_archive_hash STREQUAL _supplied_archive_hash)
            message(FATAL_ERROR "Supplied TensorRT archive changed while it was being copied.")
        endif()
    else()
        message(STATUS "Compressing all 14 TensorRT DLLs with solid LZMA2 (512 MiB, 4 threads). Log: ${_cache}/compression.log")
        execute_process(
            COMMAND "${SUNSHINE_7ZIP_EXECUTABLE}" a -t7z -mx=9 -m0=LZMA2 -md=512m
                -ms=on -mmt=4 -mtc=off -mta=off -mtm=off -scsUTF-8 -bsp0 -y
                "${_partial}" "@${_cache}/inputs.txt"
            RESULT_VARIABLE _compress_result
            OUTPUT_FILE "${_cache}/compression.log" ERROR_FILE "${_cache}/compression.log")
        if(NOT _compress_result STREQUAL "0")
            message(FATAL_ERROR "TensorRT archive creation failed (${_compress_result}); see ${_cache}/compression.log")
        endif()
    endif()
    # Validate flat membership before extraction, including for packager-supplied
    # archives. Then compare every extracted byte with the complete SDK input.
    execute_process(COMMAND "${SUNSHINE_7ZIP_EXECUTABLE}" l -slt -ba "${_partial}"
        RESULT_VARIABLE _list_result OUTPUT_VARIABLE _listing ERROR_VARIABLE _list_error)
    if(NOT _list_result STREQUAL "0")
        message(FATAL_ERROR "Cannot inspect TensorRT archive: ${_list_error}")
    endif()
    string(REPLACE "\r" "" _listing "${_listing}")
    string(REGEX MATCHALL "(^|\n)Path = [^\n]+" _archive_paths "${_listing}")
    set(_archive_names "")
    foreach(_entry IN LISTS _archive_paths)
        string(REGEX REPLACE "^\n?Path = " "" _entry "${_entry}")
        list(APPEND _archive_names "${_entry}")
    endforeach()
    list(SORT _archive_names)
    if(NOT _archive_names STREQUAL _expected_names OR _listing MATCHES "(^|\n)(Folder = \\+|Symbolic Link =|Hard Link =)")
        message(FATAL_ERROR "TensorRT archive must contain exactly the 14 supported, flat regular DLL files.")
    endif()
    string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef _verification_suffix)
    set(_verification_directory "${_cache}/verify-${_verification_suffix}")
    file(MAKE_DIRECTORY "${_verification_directory}")
    message(STATUS "Verifying all 14 archived TensorRT DLLs against the SDK. Log: ${_cache}/verification.log")
    execute_process(COMMAND "${SUNSHINE_7ZIP_EXECUTABLE}" x -bsp0 -y "-o${_verification_directory}" "${_partial}"
        RESULT_VARIABLE _test_result
        OUTPUT_FILE "${_cache}/verification.log" ERROR_FILE "${_cache}/verification.log")
    if(NOT _test_result STREQUAL "0")
        message(FATAL_ERROR "TensorRT archive extraction failed; see ${_cache}/verification.log")
    endif()
    foreach(_index RANGE 0 13)
        string(JSON _name GET "${_files_json}" ${_index} name)
        string(JSON _expected_hash GET "${_files_json}" ${_index} sha256)
        string(JSON _expected_size GET "${_files_json}" ${_index} size)
        file(SHA256 "${_verification_directory}/${_name}" _extracted_hash)
        file(SIZE "${_verification_directory}/${_name}" _extracted_size)
        if(NOT _extracted_hash STREQUAL _expected_hash OR NOT _extracted_size EQUAL _expected_size)
            message(FATAL_ERROR "Archived TensorRT DLL differs from the configured SDK: ${_name}")
        endif()
    endforeach()
    cmake_path(IS_PREFIX _cache "${_verification_directory}" NORMALIZE _safe_verification_directory)
    if(NOT _safe_verification_directory)
        message(FATAL_ERROR "Verification cleanup path is outside the TensorRT package cache.")
    endif()
    file(REMOVE_RECURSE "${_verification_directory}")
    # Refuse a manifest built from files modified while compression was running.
    file(WRITE "${_cache}/validate-inputs.cmake" "${_validation}")
    include("${_cache}/validate-inputs.cmake")
    file(RENAME "${_partial}" "${SUNSHINE_TENSORRT_RUNTIME_ARCHIVE}")
    file(SHA256 "${SUNSHINE_TENSORRT_RUNTIME_ARCHIVE}" _archive_hash)
    string(JSON _manifest SET "{}" archiveSha256 "\"${_archive_hash}\"")
    string(JSON _manifest SET "${_manifest}" files "${_files_json}")
    file(WRITE "${SUNSHINE_TENSORRT_RUNTIME_MANIFEST}" "${_manifest}\n")
    file(WRITE "${_cache}/inputs.sha256" "${_input_hash}")
else()
    message(STATUS "Reusing verified complete TensorRT archive: ${SUNSHINE_TENSORRT_RUNTIME_ARCHIVE}")
endif()
file(LOCK "${_cache}/prepare.lock" RELEASE)

foreach(_artifact IN ITEMS "${SUNSHINE_TENSORRT_RUNTIME_ARCHIVE}" "${SUNSHINE_TENSORRT_RUNTIME_MANIFEST}")
    file(SHA256 "${_artifact}" _artifact_hash)
    string(APPEND _validation
        "file(SHA256 \"${_artifact}\" _actual)\nif(NOT _actual STREQUAL \"${_artifact_hash}\")\n  message(FATAL_ERROR \"TensorRT package changed after configure: ${_artifact}\")\nendif()\n")
endforeach()
file(WRITE "${_cache}/validate-package.cmake" "${_validation}")
file(WRITE "${_cache}/7ZIP-NOTICE.txt"
    "This package includes the unmodified 7-Zip command-line executable and 7z.dll.\n7-Zip is licensed under the GNU LGPL (version 2.1 or later), with additional BSD terms and the unRAR restriction described in License.txt.\n7-Zip source code and project information: https://www.7-zip.org/\n")

# Escape the extra uninstall commands through CPack's second CMake parse.
string(REPLACE "\\" "\\\\" SUNSHINE_NSIS_TENSORRT_UNINSTALL "${SUNSHINE_NSIS_TENSORRT_UNINSTALL}")
string(REPLACE "$" "\\$" SUNSHINE_NSIS_TENSORRT_UNINSTALL "${SUNSHINE_NSIS_TENSORRT_UNINSTALL}")
string(REPLACE "\"" "\\\"" SUNSHINE_NSIS_TENSORRT_UNINSTALL "${SUNSHINE_NSIS_TENSORRT_UNINSTALL}")

# A standalone CMake preparation script can warm the same build-directory cache.
# The normal project configures the install rules without compressing a second time.
if(CMAKE_SCRIPT_MODE_FILE)
    return()
endif()
install(SCRIPT "${_cache}/validate-package.cmake" COMPONENT application)
install(FILES "${SUNSHINE_TENSORRT_RUNTIME_ARCHIVE}"
    DESTINATION "tools/tensorrt" COMPONENT application)
install(FILES "${SUNSHINE_TENSORRT_RUNTIME_MANIFEST}"
    DESTINATION "scripts" COMPONENT application)
install(PROGRAMS "${SUNSHINE_7ZIP_EXECUTABLE}" DESTINATION "tools/tensorrt/7zip"
    RENAME "7z.exe" COMPONENT application)
install(FILES "${_sevenzip_dll}" "${_sevenzip_license}" "${_cache}/7ZIP-NOTICE.txt"
    DESTINATION "tools/tensorrt/7zip" COMPONENT application)
install(FILES "${SUNSHINE_SOURCE_ASSETS_DIR}/windows/misc/runtime/install-tensorrt-bundle.ps1"
    DESTINATION "scripts" COMPONENT application)
