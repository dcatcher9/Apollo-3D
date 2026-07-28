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
find_package(Git REQUIRED)
set(SIMPLE_WEB_SERVER_OVERLAY_INCLUDE_DIR
    "${CMAKE_BINARY_DIR}/generated")
set(SIMPLE_WEB_SERVER_OVERLAY_DIR
    "${SIMPLE_WEB_SERVER_OVERLAY_INCLUDE_DIR}/Simple-Web-Server")
set(SIMPLE_WEB_SERVER_REQUEST_LIMIT_PATCH
    "${CMAKE_SOURCE_DIR}/cmake/patches/simple-web-server-request-limit.patch")
file(MAKE_DIRECTORY "${SIMPLE_WEB_SERVER_OVERLAY_DIR}")
set(SIMPLE_WEB_SERVER_HEADERS
    asio_compatibility.hpp
    client_http.hpp
    client_https.hpp
    crypto.hpp
    mutex.hpp
    server_http.hpp
    server_https.hpp
    status_code.hpp
    utility.hpp)
foreach(SIMPLE_WEB_SERVER_HEADER IN LISTS SIMPLE_WEB_SERVER_HEADERS)
    configure_file(
        "${CMAKE_SOURCE_DIR}/third-party/Simple-Web-Server/${SIMPLE_WEB_SERVER_HEADER}"
        "${SIMPLE_WEB_SERVER_OVERLAY_DIR}/${SIMPLE_WEB_SERVER_HEADER}"
        COPYONLY)
endforeach()
set_property(
    DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${SIMPLE_WEB_SERVER_REQUEST_LIMIT_PATCH}")
file(REMOVE_RECURSE "${SIMPLE_WEB_SERVER_OVERLAY_DIR}/.git")
execute_process(
    COMMAND "${GIT_EXECUTABLE}" init --quiet
    WORKING_DIRECTORY "${SIMPLE_WEB_SERVER_OVERLAY_DIR}"
    RESULT_VARIABLE SIMPLE_WEB_SERVER_INIT_RESULT
    OUTPUT_VARIABLE SIMPLE_WEB_SERVER_INIT_OUTPUT
    ERROR_VARIABLE SIMPLE_WEB_SERVER_INIT_ERROR)
if(NOT SIMPLE_WEB_SERVER_INIT_RESULT EQUAL 0)
    message(FATAL_ERROR
            "Could not initialize the Simple-Web-Server overlay.\n"
            "${SIMPLE_WEB_SERVER_INIT_OUTPUT}${SIMPLE_WEB_SERVER_INIT_ERROR}")
endif()
execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply --whitespace=nowarn
            "${SIMPLE_WEB_SERVER_REQUEST_LIMIT_PATCH}"
    WORKING_DIRECTORY "${SIMPLE_WEB_SERVER_OVERLAY_DIR}"
    RESULT_VARIABLE SIMPLE_WEB_SERVER_PATCH_RESULT
    OUTPUT_VARIABLE SIMPLE_WEB_SERVER_PATCH_OUTPUT
    ERROR_VARIABLE SIMPLE_WEB_SERVER_PATCH_ERROR)
file(REMOVE_RECURSE "${SIMPLE_WEB_SERVER_OVERLAY_DIR}/.git")
if(NOT SIMPLE_WEB_SERVER_PATCH_RESULT EQUAL 0)
    message(FATAL_ERROR
            "Could not build the Simple-Web-Server request-limit overlay. "
            "The upstream header may have changed.\n"
            "${SIMPLE_WEB_SERVER_PATCH_OUTPUT}${SIMPLE_WEB_SERVER_PATCH_ERROR}")
endif()
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
find_package(OpenSSL REQUIRED)
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
