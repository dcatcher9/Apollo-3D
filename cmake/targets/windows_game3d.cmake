# Build the production add-on from this checkout, rather than downloading a DLL
# that may speak a different protocol. Its independent project keeps SDK/test
# targets and compiler settings out of the host project.
include(ExternalProject)
set(SUNSHINE_GAME3D_BUILD_DIR "${CMAKE_BINARY_DIR}/reshade" CACHE PATH
    "Build directory for the matching Sunshine 3D add-on (an existing standalone build can be reused)")
set(SUNSHINE_GAME3D_SDK_DIR "" CACHE PATH "Optional offline checkout of the pinned ReShade SDK")
set(SUNSHINE_GAME3D_IMGUI_DIR "" CACHE PATH "Optional offline checkout of the pinned ReShade ImGui headers")

set(_game3d_cmake_args
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
    "-DSUNSHINE_SBS_RUNTIME_TESTS=OFF"
    "-DBUILD_TESTING=OFF")
if(nlohmann_json_DIR)
    list(APPEND _game3d_cmake_args "-Dnlohmann_json_DIR=${nlohmann_json_DIR}")
endif()
if(SUNSHINE_GAME3D_SDK_DIR)
    list(APPEND _game3d_cmake_args "-DRESHADE_SDK_DIR=${SUNSHINE_GAME3D_SDK_DIR}")
endif()
if(SUNSHINE_GAME3D_IMGUI_DIR)
    list(APPEND _game3d_cmake_args "-DRESHADE_IMGUI_DIR=${SUNSHINE_GAME3D_IMGUI_DIR}")
endif()

ExternalProject_Add(game3d_addon_build
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/tools/reshade"
    BINARY_DIR "${SUNSHINE_GAME3D_BUILD_DIR}"
    CMAKE_ARGS ${_game3d_cmake_args}
    BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --config $<CONFIG> --target Sunshine3DPackage
    BUILD_ALWAYS TRUE
    BUILD_BYPRODUCTS
        "${SUNSHINE_GAME3D_BUILD_DIR}/package/SunshineSBS.addon64"
        "${SUNSHINE_GAME3D_BUILD_DIR}/package/SunshineSBS-LICENSES.txt"
        "${SUNSHINE_GAME3D_BUILD_DIR}/package/install.ps1"
        "${SUNSHINE_GAME3D_BUILD_DIR}/package/install-runtime.ps1"
    INSTALL_COMMAND "")

add_custom_target(game3d_addon_bundle ALL
    COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:sunshine>/tools/reshade"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${SUNSHINE_GAME3D_BUILD_DIR}/package/SunshineSBS.addon64"
        "${SUNSHINE_GAME3D_BUILD_DIR}/package/SunshineSBS-LICENSES.txt"
        "${SUNSHINE_GAME3D_BUILD_DIR}/package/install.ps1"
        "${SUNSHINE_GAME3D_BUILD_DIR}/package/install-runtime.ps1"
        "$<TARGET_FILE_DIR:sunshine>/tools/reshade"
    DEPENDS game3d_addon_build
    COMMENT "Bundling the matching Game 3D add-on and installer"
    VERBATIM)
add_dependencies(sunshine game3d_addon_bundle)
