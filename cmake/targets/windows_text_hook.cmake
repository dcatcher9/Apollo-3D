# The Game 3D text helper is loaded into the game, so it is built once for each kind of game: by
# the host's own compiler for 64-bit games, and by a 32-bit compiler, which the host build does
# not otherwise need, for 32-bit games. Without the 32-bit compiler the host still builds and
# reports that it has no text helper for 32-bit games; see docs/building.md.
set(SUNSHINE_TEXT_HOOK_MINGW32_DIR "C:/msys64/mingw32/bin" CACHE PATH
    "Directory of the 32-bit MinGW-w64 compiler used for the Game 3D text helper")
find_program(SUNSHINE_TEXT_HOOK_GCC32 NAMES i686-w64-mingw32-gcc
    HINTS "${SUNSHINE_TEXT_HOOK_MINGW32_DIR}" NO_DEFAULT_PATH)
find_program(SUNSHINE_TEXT_HOOK_GXX32 NAMES i686-w64-mingw32-g++
    HINTS "${SUNSHINE_TEXT_HOOK_MINGW32_DIR}" NO_DEFAULT_PATH)

set(SUNSHINE_TEXT_HOOK_BUILD_DIR "${CMAKE_BINARY_DIR}/text-hook-package")
# The sample is a test fixture: built beside the helper, never bundled or installed.
set(SUNSHINE_TEXT_HOOK_SAMPLE_DIR "${CMAKE_BINARY_DIR}/text-hook-sample")
set(SUNSHINE_TEXT_HOOK_FILES "")
set(SUNSHINE_TEXT_HOOK_SAMPLES "")

# One helper, loader and sample for games of `bits` bits, built by the C driver `gcc` and the C++
# driver `gxx` found in `compiler_dir`.
function(sunshine_text_hook bits gcc gxx compiler_dir)
    set(_source_dir "${CMAKE_SOURCE_DIR}/tools/text_hook")
    set(_library "${SUNSHINE_TEXT_HOOK_BUILD_DIR}/SunshineTextHook${bits}.dll")
    set(_injector "${SUNSHINE_TEXT_HOOK_BUILD_DIR}/sunshine-text-hook-inject${bits}.exe")
    set(_sample "${SUNSHINE_TEXT_HOOK_SAMPLE_DIR}/sunshine-text-hook-sample${bits}.exe")
    # The compiler's own DLLs sit beside it. Its directory is the whole search path, so the
    # other toolchain's libraries of the same names can never be picked up instead.
    set(_env "${CMAKE_COMMAND}" -E env "PATH=${compiler_dir}")
    set(_flags -std=c++20 -O2 -Wall -Wextra -municode -fno-exceptions -fno-rtti -static-libgcc)
    set(_helper_sources
        "${_source_dir}/text_hook.cpp"
        "${_source_dir}/text_hook_d2d.cpp"
        "${_source_dir}/text_hook_d3d11.cpp"
        "${_source_dir}/text_hook_gdi.cpp")
    set(_helper_headers
        "${_source_dir}/text_hook_core.h"
        "${_source_dir}/text_hook_image.h"
        "${_source_dir}/text_hook_shared.h")

    # Loaded into another application, so it must not need a C++ runtime DLL there: it is
    # compiled without exceptions or RTTI and linked by the C driver. It does not link Direct2D
    # or DirectWrite either: it uses them only where the application itself has loaded them.
    add_custom_command(OUTPUT "${_library}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${SUNSHINE_TEXT_HOOK_BUILD_DIR}"
        COMMAND ${_env} "${gcc}" -x c++ ${_flags} -shared -o "${_library}" ${_helper_sources}
            -lgdi32 -luser32 -luuid
        DEPENDS ${_helper_sources} ${_helper_headers}
        COMMENT "Building the ${bits}-bit Game 3D text helper"
        VERBATIM)
    add_custom_command(OUTPUT "${_injector}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${SUNSHINE_TEXT_HOOK_BUILD_DIR}"
        COMMAND ${_env} "${gxx}" ${_flags} -static -o "${_injector}" "${_source_dir}/text_hook_inject.cpp"
        DEPENDS "${_source_dir}/text_hook_inject.cpp"
        COMMENT "Building the ${bits}-bit Game 3D text helper loader"
        VERBATIM)
    add_custom_command(OUTPUT "${_sample}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${SUNSHINE_TEXT_HOOK_SAMPLE_DIR}"
        COMMAND ${_env} "${gxx}" ${_flags} -static -mwindows -o "${_sample}" "${_source_dir}/text_hook_sample.cpp"
            -ld2d1 -ld3d11 -ldwrite -lgdi32 -luser32 -lmsimg32 -luuid
        DEPENDS "${_source_dir}/text_hook_sample.cpp"
        COMMENT "Building the ${bits}-bit Game 3D text helper sample"
        VERBATIM)
    set(SUNSHINE_TEXT_HOOK_FILES ${SUNSHINE_TEXT_HOOK_FILES} "${_library}" "${_injector}" PARENT_SCOPE)
    set(SUNSHINE_TEXT_HOOK_SAMPLES ${SUNSHINE_TEXT_HOOK_SAMPLES} "${_sample}" PARENT_SCOPE)
endfunction()

get_filename_component(_text_hook_host_compiler_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
sunshine_text_hook(64 "${CMAKE_C_COMPILER}" "${CMAKE_CXX_COMPILER}" "${_text_hook_host_compiler_dir}")
if(SUNSHINE_TEXT_HOOK_GCC32 AND SUNSHINE_TEXT_HOOK_GXX32)
    sunshine_text_hook(32 "${SUNSHINE_TEXT_HOOK_GCC32}" "${SUNSHINE_TEXT_HOOK_GXX32}"
        "${SUNSHINE_TEXT_HOOK_MINGW32_DIR}")
else()
    message(WARNING "No 32-bit MinGW-w64 compiler in ${SUNSHINE_TEXT_HOOK_MINGW32_DIR}: "
        "the Game 3D text helper for 32-bit games is not built.")
endif()

add_custom_target(text_hook_bundle ALL
    COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:sunshine>/tools/text-hook"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${SUNSHINE_TEXT_HOOK_FILES}
        "$<TARGET_FILE_DIR:sunshine>/tools/text-hook"
    DEPENDS ${SUNSHINE_TEXT_HOOK_FILES}
    COMMENT "Bundling the Game 3D text helper"
    VERBATIM)
add_dependencies(sunshine text_hook_bundle)
add_custom_target(text_hook_samples DEPENDS ${SUNSHINE_TEXT_HOOK_SAMPLES})
