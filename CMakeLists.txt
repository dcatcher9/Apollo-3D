cmake_minimum_required(VERSION 3.20)
# `cmake_path(CONVERT ... TO_NATIVE_PATH_LIST ...)` requires 3.20

project(Apollo VERSION 0.0.0
        DESCRIPTION "Apollo XR host for Artemis"
        HOMEPAGE_URL "https://github.com/ClassicOldSong/Apollo")

# User-facing product name. Keep PROJECT_NAME as "Apollo" for upgrade, package-path,
# service, and executable compatibility.
set(PROJECT_DISPLAY_NAME "Apollo XR")

if(NOT WIN32)
    message(FATAL_ERROR "Apollo XR host supports Windows with native NVIDIA NVENC only")
endif()

set(PROJECT_LICENSE "GPL-3.0-only")

set(PROJECT_FQDN "dev.lizardbyte.app.Sunshine")

set(PROJECT_BRIEF_DESCRIPTION "GameStream host for Artemis")  # must be <= 35 characters

set(PROJECT_LONG_DESCRIPTION "Windows XR streaming host for modern Artemis clients. Apollo XR uses native NVIDIA \
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

# setup compile definitions
include(${CMAKE_MODULE_PATH}/compile_definitions/common.cmake)

# target definitions
include(${CMAKE_MODULE_PATH}/targets/common.cmake)

# packaging
include(${CMAKE_MODULE_PATH}/packaging/common.cmake)
