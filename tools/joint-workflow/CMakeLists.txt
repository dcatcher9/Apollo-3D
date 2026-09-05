cmake_minimum_required(VERSION 3.20)
project(joint_workflow_portable LANGUAGES CXX)

# Compile the same boundary tests used by test_sunshine without initializing a GPU, loading
# TensorRT, or configuring the complete Windows application and its packaged model assets.
get_filename_component(HOST_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory("${HOST_ROOT}/third-party/googletest" googletest EXCLUDE_FROM_ALL)
add_executable(joint_workflow_portable
  "${HOST_ROOT}/tests/unit/platform/test_windows_capture_timing.cpp"
  "${HOST_ROOT}/tests/unit/test_offline_frame_transport.cpp"
)
target_compile_features(joint_workflow_portable PRIVATE cxx_std_20)
target_include_directories(joint_workflow_portable PRIVATE "${HOST_ROOT}")
target_link_libraries(joint_workflow_portable PRIVATE GTest::gtest_main)
enable_testing()
include(GoogleTest)
gtest_discover_tests(joint_workflow_portable)
