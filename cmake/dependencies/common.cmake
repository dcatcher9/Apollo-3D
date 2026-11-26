# load common dependencies
# this file will also load platform specific dependencies

# boost, this should be before Simple-Web-Server as it also depends on boost
include(dependencies/Boost_Sunshine)

# submodules
# moonlight common library
set(ENET_NO_INSTALL ON CACHE BOOL "Don't install any libraries built for enet")
add_subdirectory("${CMAKE_SOURCE_DIR}/third-party/moonlight-common-c/enet")

# web server
add_subdirectory("${CMAKE_SOURCE_DIR}/third-party/Simple-Web-Server")

# Simple-Web-Server fixes one global request-buffer limit at Session construction time. Sunshine
# needs a stricter offline-API body cap without regressing existing app/config/cover payloads.
# Generate a patched header overlay in the build tree so the upstream submodule stays immutable
# and a clean checkout contains the complete, reproducible change.
include(dependencies/simple_web_server_overlay)
set(SIMPLE_WEB_SERVER_OVERLAY_INCLUDE_DIR
    "${CMAKE_BINARY_DIR}/generated")
set(SIMPLE_WEB_SERVER_OVERLAY_DIR
    "${SIMPLE_WEB_SERVER_OVERLAY_INCLUDE_DIR}/Simple-Web-Server")
sunshine_prepare_simple_web_server_overlay(
    "${CMAKE_SOURCE_DIR}/third-party/Simple-Web-Server"
    "${CMAKE_SOURCE_DIR}/cmake/patches/simple-web-server-request-limit.patch"
    "${SIMPLE_WEB_SERVER_OVERLAY_DIR}")
add_library(sunshine-simple-web-overlay INTERFACE)
target_include_directories(
    sunshine-simple-web-overlay BEFORE INTERFACE
    "${SIMPLE_WEB_SERVER_OVERLAY_INCLUDE_DIR}")
# Apollo historically consumes Simple-Web-Server through a global third-party include. Propagate
# only the generated override through a dedicated interface target; the upstream target also
# carries a Boost::boost dependency that this fork's custom Boost build intentionally does not
# define.
list(APPEND SUNSHINE_EXTERNAL_LIBRARIES sunshine-simple-web-overlay)

# libdisplaydevice
add_subdirectory("${CMAKE_SOURCE_DIR}/third-party/libdisplaydevice")

# common dependencies
include("${CMAKE_MODULE_PATH}/dependencies/nlohmann_json.cmake")
find_package(OpenSSL 3.0 REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
pkg_check_modules(CURL REQUIRED libcurl)

# Apollo supports only the native Windows/NVIDIA host path.
include("${CMAKE_MODULE_PATH}/dependencies/windows.cmake")

# TensorRT (required for Depth Anything zero-copy SBS)
include("${CMAKE_MODULE_PATH}/dependencies/tensorrt.cmake")
list(APPEND SUNSHINE_EXTERNAL_LIBRARIES TensorRT::TensorRT)
if(TARGET TensorRT::NvOnnxParser)
    list(APPEND SUNSHINE_EXTERNAL_LIBRARIES TensorRT::NvOnnxParser)
endif()
if(TARGET TensorRT::NvInferPlugin)
    list(APPEND SUNSHINE_EXTERNAL_LIBRARIES TensorRT::NvInferPlugin)
endif()
