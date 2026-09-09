cmake_minimum_required(VERSION 3.20)
project(shared_client_features LANGUAGES C CXX)

get_filename_component(DEFAULT_HOST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(HOST_ROOT "${DEFAULT_HOST_ROOT}" CACHE PATH "Sunshine 3D checkout")
set(CLIENT_COMMON_ROOT "" CACHE PATH "Published shared moonlight-common-c checkout")
foreach(required
    "${HOST_ROOT}/src/client_features.h"
    "${CLIENT_COMMON_ROOT}/src/Ds5HapticsStream.c"
    "${CLIENT_COMMON_ROOT}/src/Ds5HapticsStream.h")
  if(NOT EXISTS "${required}")
    message(FATAL_ERROR "Missing production source: ${required}. Set HOST_ROOT and CLIENT_COMMON_ROOT explicitly.")
  endif()
endforeach()

add_executable(client_features_interop
  client_features_interop.cpp
  "${CLIENT_COMMON_ROOT}/src/Ds5HapticsStream.c")
set_target_properties(client_features_interop PROPERTIES C_STANDARD 11 CXX_STANDARD 20)
target_include_directories(client_features_interop PRIVATE
  "${HOST_ROOT}"
  "${CLIENT_COMMON_ROOT}/src"
  "${CLIENT_COMMON_ROOT}/enet/include"
  "${CLIENT_COMMON_ROOT}/reedsolomon")

enable_testing()
add_test(NAME client_features_interop COMMAND client_features_interop)
set_tests_properties(client_features_interop PROPERTIES TIMEOUT 10)
