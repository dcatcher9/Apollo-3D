# common target definitions
# this file will also load platform specific macros

add_executable(sunshine ${SUNSHINE_TARGET_FILES})
foreach(dep ${SUNSHINE_TARGET_DEPENDENCIES})
    add_dependencies(sunshine ${dep})  # compile these before sunshine
endforeach()

# Apollo supports only the native Windows/NVIDIA host path.
include(${CMAKE_MODULE_PATH}/targets/windows.cmake)

target_link_libraries(sunshine ${SUNSHINE_EXTERNAL_LIBRARIES} ${EXTRA_LIBS})
target_compile_definitions(sunshine PUBLIC ${SUNSHINE_DEFINITIONS})
set_target_properties(sunshine PROPERTIES CXX_STANDARD 23
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR})

target_compile_options(sunshine PRIVATE ${SUNSHINE_COMPILE_OPTIONS})

#WebUI build
find_program(NPM npm REQUIRED)

if (NPM_OFFLINE)
    set(NPM_INSTALL_FLAGS "--offline")
else()
    set(NPM_INSTALL_FLAGS "")
endif()

add_custom_target(web-ui ALL
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Installing NPM Dependencies and Building the Web UI"
        COMMAND cmd /C "${NPM}" install ${NPM_INSTALL_FLAGS}
        COMMAND "${CMAKE_COMMAND}" -E env "SUNSHINE_SOURCE_ASSETS_DIR=${SUNSHINE_SOURCE_ASSETS_DIR}" "SUNSHINE_ASSETS_DIR=${CMAKE_BINARY_DIR}" cmd /C "${NPM}" run build  # cmake-lint: disable=C0301
        COMMAND_EXPAND_LISTS
        VERBATIM)

# docs
if(BUILD_DOCS)
    add_subdirectory(third-party/doxyconfig docs)
endif()

# tests
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# custom compile flags, must be after adding tests

if (NOT BUILD_TESTS)
    set(TEST_DIR "")
else()
    set(TEST_DIR "${CMAKE_SOURCE_DIR}/tests")
endif()

# third-party/ViGEmClient
set(VIGEM_COMPILE_FLAGS "")
string(APPEND VIGEM_COMPILE_FLAGS "-Wno-unknown-pragmas ")
string(APPEND VIGEM_COMPILE_FLAGS "-Wno-misleading-indentation ")
string(APPEND VIGEM_COMPILE_FLAGS "-Wno-class-memaccess ")
string(APPEND VIGEM_COMPILE_FLAGS "-Wno-unused-function ")
string(APPEND VIGEM_COMPILE_FLAGS "-Wno-unused-variable ")
set_source_files_properties("${CMAKE_SOURCE_DIR}/third-party/ViGEmClient/src/ViGEmClient.cpp"
        DIRECTORY "${CMAKE_SOURCE_DIR}" "${TEST_DIR}"
        PROPERTIES
        COMPILE_DEFINITIONS "UNICODE=1;ERROR_INVALID_DEVICE_OBJECT_PARAMETER=650"
        COMPILE_FLAGS ${VIGEM_COMPILE_FLAGS})

# src/nvhttp
string(TOUPPER "x${CMAKE_BUILD_TYPE}" BUILD_TYPE)
if("${BUILD_TYPE}" STREQUAL "XDEBUG")
    if (NOT BUILD_TESTS)
        set_source_files_properties("${CMAKE_SOURCE_DIR}/src/nvhttp.cpp"
                DIRECTORY "${CMAKE_SOURCE_DIR}"
                PROPERTIES COMPILE_FLAGS -O2)
    else()
        set_source_files_properties("${CMAKE_SOURCE_DIR}/src/nvhttp.cpp"
                DIRECTORY "${CMAKE_SOURCE_DIR}" "${CMAKE_SOURCE_DIR}/tests"
                PROPERTIES COMPILE_FLAGS -O2)
    endif()
else()
    add_definitions(-DNDEBUG)
endif()

# Copy TensorRT DLLs to the build directory post-build
if(PROJECT_TENSORRT_DLLS)
    add_custom_command(TARGET sunshine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${PROJECT_TENSORRT_DLLS}
        $<TARGET_FILE_DIR:sunshine>
        COMMENT "Copying TensorRT DLLs to output directory"
    )
endif()
