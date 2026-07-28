cmake_minimum_required(VERSION 3.20)
# `cmake_path(CONVERT ... TO_NATIVE_PATH_LIST ...)` requires 3.20

project(Apollo VERSION 0.0.0
        DESCRIPTION "Sunshine 3D host for Moonlight 3D"
        HOMEPAGE_URL "https://github.com/ClassicOldSong/Apollo")

# User-facing product name. Keep PROJECT_NAME as "Apollo" for upgrade, package-path,
# service, and executable compatibility.
set(PROJECT_DISPLAY_NAME "Sunshine 3D")

if(NOT WIN32)
    message(FATAL_ERROR "Sunshine 3D supports Windows with native NVIDIA NVENC only")
endif()

set(PROJECT_LICENSE "GPL-3.0-only")

# Native offline conversion intentionally does not search PATH. Official/package builds can
# provide one audited matching pair here; they will be copied beside Sunshine under tools/.
set(SUNSHINE_OFFLINE_FFMPEG "" CACHE FILEPATH
        "Absolute path to the approved ffmpeg.exe bundled for offline SBS conversion")
set(SUNSHINE_OFFLINE_FFPROBE "" CACHE FILEPATH
        "Absolute path to the matching approved ffprobe.exe bundled for offline SBS conversion")
if((SUNSHINE_OFFLINE_FFMPEG AND NOT SUNSHINE_OFFLINE_FFPROBE) OR
   (SUNSHINE_OFFLINE_FFPROBE AND NOT SUNSHINE_OFFLINE_FFMPEG))
    message(FATAL_ERROR
            "SUNSHINE_OFFLINE_FFMPEG and SUNSHINE_OFFLINE_FFPROBE must be supplied together")
endif()
if(SUNSHINE_OFFLINE_FFMPEG)
    foreach(media_tool IN ITEMS SUNSHINE_OFFLINE_FFMPEG SUNSHINE_OFFLINE_FFPROBE)
        if(NOT IS_ABSOLUTE "${${media_tool}}" OR NOT EXISTS "${${media_tool}}")
            message(FATAL_ERROR "${media_tool} must name an existing absolute file")
        endif()
    endforeach()
    if(NOT SUNSHINE_OFFLINE_FFMPEG MATCHES "[/\\\\]ffmpeg\\.exe$" OR
       NOT SUNSHINE_OFFLINE_FFPROBE MATCHES "[/\\\\]ffprobe\\.exe$")
        message(FATAL_ERROR "Offline media tools must retain the names ffmpeg.exe and ffprobe.exe")
    endif()
else()
    message(STATUS
            "Offline SBS conversion media tools are not bundled in this build; "
            "install ffmpeg.exe and ffprobe.exe under the Sunshine tools directory to enable it")
endif()

set(PROJECT_FQDN "dev.lizardbyte.app.Sunshine")

set(PROJECT_BRIEF_DESCRIPTION "3D streaming host for Moonlight 3D")  # must be <= 35 characters

set(PROJECT_LONG_DESCRIPTION "Windows XR streaming host for Moonlight 3D. Sunshine 3D uses native NVIDIA \
NVENC for H.264, HEVC, and AV1 video, with HDR and host-side stereoscopic conversion support. A web UI provides \
configuration and device pairing.")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# set the module path, used for includes
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# export compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set version info for this build
include(${CMAKE_MODULE_PATH}/prep/build_version.cmake)

# cmake build flags
include(${CMAKE_MODULE_PATH}/prep/options.cmake)

# initial prep
include(${CMAKE_MODULE_PATH}/prep/init.cmake)

# project constants
include(${CMAKE_MODULE_PATH}/prep/constants.cmake)

# load macros
include(${CMAKE_MODULE_PATH}/macros/common.cmake)

# load dependencies
include(${CMAKE_MODULE_PATH}/dependencies/common.cmake)

# The adaptive-state manifest is the sole editable definition shared by native code, shaders,
# and Python tooling. Refuse every normal build when either committed generated consumer is
# stale; otherwise a same-sized field reorder could silently change live or offline telemetry.
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
set(SBS_ADAPTIVE_STATE_GENERATOR
        "${CMAKE_SOURCE_DIR}/tools/sbsbench/generate_adaptive_state_contract.py")
set(SBS_ADAPTIVE_STATE_MANIFEST
        "${CMAKE_SOURCE_DIR}/tools/sbsbench/contracts/sbs-adaptive-state-v3.json")
set(SBS_ADAPTIVE_STATE_CPP
        "${CMAKE_SOURCE_DIR}/src/generated/sbs_adaptive_state_contract.h")
set(SBS_ADAPTIVE_STATE_HLSL
        "${CMAKE_SOURCE_DIR}/src_assets/windows/assets/shaders/directx/include/sbs_adaptive_state_contract.generated.hlsl")
add_custom_target(verify_sbs_adaptive_state_contract ALL
        COMMAND "${Python3_EXECUTABLE}" "${SBS_ADAPTIVE_STATE_GENERATOR}" --check
        DEPENDS
            "${SBS_ADAPTIVE_STATE_GENERATOR}"
            "${SBS_ADAPTIVE_STATE_MANIFEST}"
            "${SBS_ADAPTIVE_STATE_CPP}"
            "${SBS_ADAPTIVE_STATE_HLSL}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Verifying generated Host SBS adaptive-state contract"
        VERBATIM)
list(APPEND SUNSHINE_TARGET_DEPENDENCIES verify_sbs_adaptive_state_contract)

# setup compile definitions
include(${CMAKE_MODULE_PATH}/compile_definitions/common.cmake)

# target definitions
include(${CMAKE_MODULE_PATH}/targets/common.cmake)

# packaging
include(${CMAKE_MODULE_PATH}/packaging/common.cmake)
