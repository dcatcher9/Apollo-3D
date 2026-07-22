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
